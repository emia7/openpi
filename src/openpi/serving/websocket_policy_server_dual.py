import asyncio
import http
import logging
import time
import traceback
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation as R

from openpi_client import base_policy as _base_policy
from openpi_client import msgpack_numpy
import websockets.asyncio.server as _server
import websockets.frames

logger = logging.getLogger(__name__)


def _as_np(x: Any, *, dtype=np.float32) -> np.ndarray:
    if x is None:
        return np.asarray([], dtype=dtype)
    return np.asarray(x, dtype=dtype)


def _pose7_to_pos_rotvec(pose7: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """pose7: [x,y,z,qx,qy,qz,qw] -> (pos(3), rotvec(3))"""
    pose7 = _as_np(pose7, dtype=np.float32).reshape(-1)
    if pose7.shape[0] != 7:
        raise ValueError(f"Expected pose7 shape (7,), got {pose7.shape}")
    pos = pose7[:3]
    quat = pose7[3:7]  # (qx,qy,qz,qw)
    rotvec = R.from_quat(quat).as_rotvec().astype(np.float32)
    return pos.astype(np.float32), rotvec


def _pick_image_from_obs(obs: dict, *, preferred_keys: list[str]) -> np.ndarray | None:
    """
    Try to pick an image from:
      - obs["left_view"]/["right_view"]/["third_view"]
      - obs["images"][...]
      - obs["image"]
    """
    for k in preferred_keys:
        if k in obs and obs[k] is not None:
            return np.asarray(obs[k])

    images = obs.get("images", None)
    if isinstance(images, dict):
        # Try exact preferred_keys inside images
        for k in preferred_keys:
            if k in images and images[k] is not None:
                return np.asarray(images[k])
        # Otherwise any image
        for _, v in images.items():
            if v is not None:
                return np.asarray(v)

    if "image" in obs and obs["image"] is not None:
        return np.asarray(obs["image"])

    return None


class WebsocketPolicyServerDual:
    """
    Websocket policy server adapter for **dual-arm** realRL -> OpenPI `xv_dual_policy`.

    Expected incoming obs (from realRL PolicyActionProvider dual_arm=True) commonly contains:
      - "image" (single view) OR "left_view"/"right_view"/"third_view" OR "images" dict
      - "left_tcp_pose"/"right_tcp_pose": pose7 (x,y,z,qx,qy,qz,qw)
      - "left_gripper_pose"/"right_gripper_pose": (1,) (normalized width or state)
      - optional "prompt"/"task"

    This adapter converts to the keys expected by `XVDualInputs`:
      - left_view/right_view/third_view
      - left_eef_pos/left_eef_rotvec/left_gripper
      - right_eef_pos/right_eef_rotvec/right_gripper
      - demo_start_pose_left/demo_start_pose_right (pose6, pos3+rotvec3)
    """

    def __init__(
        self,
        policy: _base_policy.BasePolicy,
        host: str = "0.0.0.0",
        port: int | None = None,
        metadata: dict | None = None,
    ) -> None:
        self._policy = policy
        self._host = host
        self._port = port
        self._metadata = metadata or {}
        logging.getLogger("websockets.server").setLevel(logging.INFO)

        # Per-connection demo-start pose cache (pose6 per arm)
        self._start_pose_left_by_conn: dict[int, np.ndarray] = {}
        self._start_pose_right_by_conn: dict[int, np.ndarray] = {}

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self):
        async with _server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
            process_request=_health_check,
        ) as server:
            await server.serve_forever()

    def _adapt_obs_for_xv_dual(self, obs: dict, conn_id: int) -> dict:
        # -----------------------
        # 1) Images (strict: require all three views)
        # -----------------------
        left_img = _pick_image_from_obs(obs, preferred_keys=["left_view", "left/image", "left", "left_wrist", "left/wrist"])
        right_img = _pick_image_from_obs(obs, preferred_keys=["right_view", "right/image", "right", "right_wrist", "right/wrist"])
        third_img = _pick_image_from_obs(obs, preferred_keys=["third_view", "third/image", "base", "side", "front", "fish_eye_front"])
        missing_views = []
        if left_img is None:
            missing_views.append("left_view")
        if right_img is None:
            missing_views.append("right_view")
        if third_img is None:
            missing_views.append("third_view")
        if missing_views:
            raise KeyError(
                "Dual server requires all three image views. "
                f"Missing: {missing_views}. "
                "Provide left/right/third view explicitly; fallback broadcasting is disabled."
            )

        # -----------------------
        # 2) Poses & grippers
        # -----------------------
        # Preferred direct keys from realRL wrapper
        left_tcp = obs.get("left_tcp_pose", None)
        right_tcp = obs.get("right_tcp_pose", None)

        # Fallback: sometimes the client may put them inside obs["state"]
        state = obs.get("state", None)
        if isinstance(state, dict):
            left_tcp = left_tcp if left_tcp is not None else state.get("left/tcp_pose", None)
            right_tcp = right_tcp if right_tcp is not None else state.get("right/tcp_pose", None)

        if left_tcp is None or right_tcp is None:
            raise KeyError(
                "Dual server expects both left/right tcp poses. "
                "Missing one of: left_tcp_pose/right_tcp_pose (or state['left/tcp_pose']/state['right/tcp_pose'])."
            )

        l_pos, l_rotvec = _pose7_to_pos_rotvec(left_tcp)
        r_pos, r_rotvec = _pose7_to_pos_rotvec(right_tcp)

        left_grip = obs.get("left_gripper_pose", None)
        right_grip = obs.get("right_gripper_pose", None)
        if isinstance(state, dict):
            left_grip = left_grip if left_grip is not None else state.get("left/gripper_pose", state.get("left/gripper", None))
            right_grip = right_grip if right_grip is not None else state.get("right/gripper_pose", state.get("right/gripper", None))

        l_g = _as_np(left_grip, dtype=np.float32).reshape(-1)
        r_g = _as_np(right_grip, dtype=np.float32).reshape(-1)
        l_g = np.array([float(l_g[0])], dtype=np.float32) if l_g.size else np.array([0.0], dtype=np.float32)
        r_g = np.array([float(r_g[0])], dtype=np.float32) if r_g.size else np.array([0.0], dtype=np.float32)

        # -----------------------
        # 3) Demo-start poses (pose6)
        # -----------------------
        start_l = obs.get("demo_start_pose_left", None)
        start_r = obs.get("demo_start_pose_right", None)

        if start_l is not None and start_r is not None:
            demo_start_left = _as_np(start_l, dtype=np.float32).reshape(-1)
            demo_start_right = _as_np(start_r, dtype=np.float32).reshape(-1)
        else:
            cached_l = self._start_pose_left_by_conn.get(conn_id, None)
            cached_r = self._start_pose_right_by_conn.get(conn_id, None)
            if cached_l is None or cached_r is None:
                cached_l = np.concatenate([l_pos, l_rotvec], axis=0).astype(np.float32)
                cached_r = np.concatenate([r_pos, r_rotvec], axis=0).astype(np.float32)
                self._start_pose_left_by_conn[conn_id] = cached_l
                self._start_pose_right_by_conn[conn_id] = cached_r
            demo_start_left = cached_l
            demo_start_right = cached_r

        if demo_start_left.shape[0] != 6 or demo_start_right.shape[0] != 6:
            raise ValueError(
                f"demo_start_pose_left/right must be pose6 (6,), got "
                f"{demo_start_left.shape} / {demo_start_right.shape}"
            )

        out = {
            "third_view": third_img,
            "left_view": left_img,
            "right_view": right_img,
            "left_eef_pos": l_pos,
            "left_eef_rotvec": l_rotvec,
            "left_gripper": l_g,
            "right_eef_pos": r_pos,
            "right_eef_rotvec": r_rotvec,
            "right_gripper": r_g,
            "demo_start_pose_left": demo_start_left.astype(np.float32),
            "demo_start_pose_right": demo_start_right.astype(np.float32),
        }

        if "prompt" in obs:
            out["prompt"] = obs["prompt"]
        elif "task" in obs:
            out["task"] = obs["task"]

        return out

    async def _handler(self, websocket: _server.ServerConnection):
        logger.info(f"Connection from {websocket.remote_address} opened")
        packer = msgpack_numpy.Packer()
        conn_id = id(websocket)

        # Reset cached start poses for this connection (new episode/client)
        self._start_pose_left_by_conn.pop(conn_id, None)
        self._start_pose_right_by_conn.pop(conn_id, None)

        await websocket.send(packer.pack(self._metadata))

        prev_total_time = None
        while True:
            try:
                start_time = time.monotonic()
                obs = msgpack_numpy.unpackb(await websocket.recv())

                adapted = self._adapt_obs_for_xv_dual(obs, conn_id)

                infer_time = time.monotonic()
                action = self._policy.infer(adapted)
                infer_time = time.monotonic() - infer_time

                action["server_timing"] = {
                    "infer_ms": infer_time * 1000,
                }
                if prev_total_time is not None:
                    action["server_timing"]["prev_total_ms"] = prev_total_time * 1000

                await websocket.send(packer.pack(action))
                prev_total_time = time.monotonic() - start_time

            except websockets.ConnectionClosed:
                logger.info(f"Connection from {websocket.remote_address} closed")
                break
            except Exception:
                await websocket.send(traceback.format_exc())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback included in previous frame.",
                )
                raise


def _health_check(connection: _server.ServerConnection, request: _server.Request) -> _server.Response | None:
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    return None
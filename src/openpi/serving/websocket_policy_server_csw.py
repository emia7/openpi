import asyncio
import http
import logging
import time
import traceback

from openpi_client import base_policy as _base_policy
from openpi_client import msgpack_numpy
import websockets.asyncio.server as _server
import websockets.frames
import numpy as np
from scipy.spatial.transform import Rotation as R

logger = logging.getLogger(__name__)


class WebsocketPolicyServer:
    """Serves a policy using the websocket protocol. See websocket_client_policy.py for a client implementation.

    Currently only implements the `load` and `infer` methods.
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

        self._episode_start_pose_by_conn: dict[int, np.ndarray] = {}

    def adapt_obs_for_xv(self, obs: dict, conn_id: int) -> dict:

        image = obs["image"]

        tcp = np.asarray(obs["tcp_pose"], dtype=np.float32).reshape(-1)
        if tcp.shape[0] != 7:
            raise ValueError(f"tcp_pose must be shape (7,) = (x,y,z,qx,qy,qz,qw), got {tcp.shape}")

        pos = tcp[0:3]
        quat = tcp[3:7]  # (qx,qy,qz,qw)
        rotvec = R.from_quat(quat).as_rotvec().astype(np.float32)

        grip = np.asarray(obs["gripper_pose"], dtype=np.float32).reshape(-1)
        gripper_width = np.array([float(grip[0])], dtype=np.float32) if grip.size else np.array([0.0], dtype=np.float32)

        start_tcp = obs["demo_start_tcp_pose"]
        start_pos = start_tcp[0:3]
        start_quat = start_tcp[3:7]
        start_rotvec = R.from_quat(start_quat).as_rotvec().astype(np.float32)
        start_pose = np.concatenate([start_pos, start_rotvec], axis=0).astype(np.float32)
 
        out = {
            "image": image,
            "eef_pos": pos.astype(np.float32),
            "eef_rot_axis_angle": rotvec.astype(np.float32),
            "gripper_width": gripper_width,
            "demo_start_pose": start_pose,
        }

        if "prompt" in obs:
            out["prompt"] = obs["prompt"]
        elif "task" in obs:
            out["task"] = obs["task"]

        return out
    
    def adapt_obs_for_xv_old(self, obs: dict, conn_id: int) -> dict:


        image = obs["image"]

        tcp = np.asarray(obs["tcp_pose"], dtype=np.float32).reshape(-1)
        if tcp.shape[0] != 7:
            raise ValueError(f"tcp_pose must be shape (7,) = (x,y,z,qx,qy,qz,qw), got {tcp.shape}")

        pos = tcp[0:3]
        quat = tcp[3:7]  # (qx,qy,qz,qw)
        rotvec = R.from_quat(quat).as_rotvec().astype(np.float32)

        grip = np.asarray(obs["gripper_pose"], dtype=np.float32).reshape(-1)
        gripper_width = np.array([float(grip[0])], dtype=np.float32) if grip.size else np.array([0.0], dtype=np.float32)

        start_pose = self._episode_start_pose_by_conn.get(conn_id, None)
        if start_pose is None:
            start_pose = np.concatenate([pos, rotvec], axis=0).astype(np.float32)
            self._episode_start_pose_by_conn[conn_id] = start_pose

        out = {
            "image": image,
            "eef_pos": pos.astype(np.float32),
            "eef_rot_axis_angle": rotvec.astype(np.float32),
            "gripper_width": gripper_width,
            "demo_start_pose": start_pose,
        }

        if "prompt" in obs:
            out["prompt"] = obs["prompt"]
        elif "task" in obs:
            out["task"] = obs["task"]

        return out

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

    async def _handler(self, websocket: _server.ServerConnection):
        logger.info(f"Connection from {websocket.remote_address} opened")
        packer = msgpack_numpy.Packer()

        await websocket.send(packer.pack(self._metadata))

        prev_total_time = None
        while True:
            try:
                start_time = time.monotonic()
                obs = msgpack_numpy.unpackb(await websocket.recv())
                adapt_obs = self.adapt_obs_for_xv(obs,1)
                
                infer_time = time.monotonic()
                action = self._policy.infer(adapt_obs)
                infer_time = time.monotonic() - infer_time

                action["server_timing"] = {
                    "infer_ms": infer_time * 1000,
                }
                if prev_total_time is not None:
                    # We can only record the last total time since we also want to include the send time.
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
    # Continue with the normal request handling.
    return None

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ROS1: 3-camera 10Hz sampler + L-R stamp delta monitor (A方案)

- Left/Right (xv_sdk): sensor_msgs/Image @ ~60Hz -> publish 10Hz (latest cached)
- Third view (D435): sensor_msgs/CompressedImage @ ~30Hz -> publish 10Hz (latest cached)
- Timer-driven sampling (10Hz) publishes the latest cached frame from each camera.
- Keeps ORIGINAL header.stamp (for downstream pose alignment on the same time base).
- ONLY monitors |L-R| stamp delta (ignores D435 stamp time base to avoid false warnings).

Usage:
  1) Put this file in a catkin package's scripts/ directory
  2) chmod +x scripts/tri_image_sampler_10hz_lr_only.py
  3) rosrun <your_pkg> tri_image_sampler_10hz_lr_only.py

Configure topics below in CONFIG (no command-line params needed).
"""

import rospy
from threading import Lock
from collections import deque
import math

from sensor_msgs.msg import Image, CompressedImage


# =========================
# Edit config here
# =========================
CONFIG = {
    "rate_hz": 10.0,

    # Print controls (avoid spamming logs)
    "print_every_sec": 1.0,     # print deltas every N seconds
    "window_size": 200,         # sliding window length (in timer ticks, ~20s at 10Hz)
    "warn_if_gap_ms": 30.0,     # warn if abs L-R stamp delta exceeds this
    "require_all": True,        # if True: only publish when all 3 cams have frames

    # Camera topics
    "cameras": {
        "left": {
            "in":  "/xv_sdk/250801DR48FP25002268/color_camera/image",
            "out": "/xv_sdk/250801DR48FP25002268/color_camera/image_10hz",
            "type": "raw",  # raw = sensor_msgs/Image
        },
        "right": {
            "in":  "/xv_sdk/250801DR48FP25002565/color_camera/image",
            "out": "/xv_sdk/250801DR48FP25002565/color_camera/image_10hz",
            "type": "raw",
        },
        "d435": {
            "in":  "/camera/color/image_raw/compressed",
            "out": "/camera/color/image_raw/compressed_10hz",
            "type": "compressed",  # compressed = sensor_msgs/CompressedImage
        },
    },
}


class TriImageSampler10HzLROnly:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.rate_hz = float(cfg["rate_hz"])
        self.print_every_sec = float(cfg["print_every_sec"])
        self.warn_if_gap_ms = float(cfg["warn_if_gap_ms"])
        self.require_all = bool(cfg["require_all"])

        self.window_size = int(cfg["window_size"])
        self.lr_deltas = deque(maxlen=self.window_size)
        self.last_print_time = rospy.Time(0)

        cams = cfg["cameras"]
        self.left_in  = cams["left"]["in"]
        self.right_in = cams["right"]["in"]
        self.d435_in  = cams["d435"]["in"]

        self.left_out  = cams["left"]["out"]
        self.right_out = cams["right"]["out"]
        self.d435_out  = cams["d435"]["out"]

        self.pub_left  = rospy.Publisher(self.left_out, Image, queue_size=1)
        self.pub_right = rospy.Publisher(self.right_out, Image, queue_size=1)
        self.pub_d435  = rospy.Publisher(self.d435_out, CompressedImage, queue_size=1)

        self.lock = Lock()
        self.latest_left = None   # Image
        self.latest_right = None  # Image
        self.latest_d435 = None   # CompressedImage

        # Keep newest only; avoid lag. buff_size large for images.
        rospy.Subscriber(self.left_in,  Image, self._cb_left,  queue_size=1, buff_size=2**24)
        rospy.Subscriber(self.right_in, Image, self._cb_right, queue_size=1, buff_size=2**24)
        rospy.Subscriber(self.d435_in,  CompressedImage, self._cb_d435, queue_size=1, buff_size=2**24)

        self.timer = rospy.Timer(rospy.Duration(1.0 / self.rate_hz), self._on_timer)

        rospy.loginfo("TriImageSampler10HzLROnly started.")
        rospy.loginfo("rate_hz=%.3f, require_all=%s", self.rate_hz, str(self.require_all))
        rospy.loginfo("Left : %s -> %s", self.left_in, self.left_out)
        rospy.loginfo("Right: %s -> %s", self.right_in, self.right_out)
        rospy.loginfo("D435 : %s -> %s", self.d435_in, self.d435_out)
        rospy.loginfo("L-R delta print every %.2fs, warn_if_gap_ms=%.1f, window_size=%d",
                      self.print_every_sec, self.warn_if_gap_ms, self.window_size)

    def _cb_left(self, msg: Image):
        with self.lock:
            self.latest_left = msg

    def _cb_right(self, msg: Image):
        with self.lock:
            self.latest_right = msg

    def _cb_d435(self, msg: CompressedImage):
        with self.lock:
            self.latest_d435 = msg

    @staticmethod
    def _abs_dt_ms(t0: rospy.Time, t1: rospy.Time) -> float:
        return abs((t0 - t1).to_sec()) * 1000.0

    def _maybe_print_lr(self, now: rospy.Time, lr_ms: float, tL: rospy.Time, tR: rospy.Time):
        if (now - self.last_print_time).to_sec() < self.print_every_sec:
            return
        self.last_print_time = now

        if len(self.lr_deltas) == 0:
            lr_mean, lr_max = (math.nan, math.nan)
        else:
            lr_mean = sum(self.lr_deltas) / len(self.lr_deltas)
            lr_max = max(self.lr_deltas)

        msg = (
            f"[stamp_delta @10Hz] "
            f"tL={tL.to_sec():.6f}  tR={tR.to_sec():.6f}  |  "
            f"|L-R|={lr_ms:.2f}ms (mean={lr_mean:.2f}, max={lr_max:.2f})  "
            f"window={len(self.lr_deltas)}"
        )

        if lr_ms >= self.warn_if_gap_ms:
            rospy.logwarn(msg + f"  (>= {self.warn_if_gap_ms:.1f}ms)")
        else:
            rospy.loginfo(msg)

    def _on_timer(self, _evt):
        with self.lock:
            mL = self.latest_left
            mR = self.latest_right
            mD = self.latest_d435

        if self.require_all:
            if (mL is None) or (mR is None) or (mD is None):
                return
        else:
            if (mL is None) and (mR is None) and (mD is None):
                return

        # Publish latest (keep original header.stamp)
        if mL is not None:
            self.pub_left.publish(mL)
        if mR is not None:
            self.pub_right.publish(mR)
        if mD is not None:
            self.pub_d435.publish(mD)

        # Only monitor L-R stamp delta (ignore D435)
        if (mL is not None) and (mR is not None):
            tL = mL.header.stamp
            tR = mR.header.stamp
            lr_ms = self._abs_dt_ms(tL, tR)

            self.lr_deltas.append(lr_ms)
            self._maybe_print_lr(rospy.Time.now(), lr_ms, tL, tR)


if __name__ == "__main__":
    rospy.init_node("tri_image_sampler_10hz_lr_only", anonymous=False)
    TriImageSampler10HzLROnly(CONFIG)
    rospy.spin()
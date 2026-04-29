#!/usr/bin/env python3
"""
本机只读起 HTTP 服务，为「标定时间轴」静态页白名单提供：
- ``/``、``/app.js``、``/app.css`` 静态资源
- ``/api/markers``：只读已指定的 markers.json
- ``/stream/video``：只读已指定的视频，支持 **Range**（大 MP4 可拖进度）

使用::

  cd umi_scripts_csw
  python3 nano_sync_freq/annotate_server.py \\
    --video /path/to/record.mp4 --markers /path/to/markers.json
  浏览器打开终端打印的 http://127.0.0.1:PORT/
"""

from __future__ import annotations

import argparse
import mimetypes
import os
import re
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import ClassVar
from urllib.parse import unquote, urlparse

_ROOT = Path(__file__).resolve().parent
_STATIC = _ROOT / "annotate"

_VIDEO_PATH: str | None = None
_MARKERS_PATH: str | None = None


def _read_video_range(
    f,
    file_size: int,
    start: int,
    end: int,
) -> bytes:
    f.seek(start)
    return f.read(end - start + 1)


class AnnotateRequestHandler(BaseHTTPRequestHandler):
    _video_path: ClassVar[str | None] = None
    _markers_path: ClassVar[str | None] = None
    _static_root: ClassVar[Path] = _STATIC

    def log_message(self, format: str, *args) -> None:  # noqa: A003
        sys.stderr.write(
            f"[{self.log_date_time_string()}] {self.client_address[0]} - {format % args}\n"
        )

    def _send_404(self) -> None:
        self.send_response(404)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.end_headers()
        self.wfile.write(b"not found")

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = unquote(parsed.path)

        if path in ("/", "/index.html"):
            return self._send_static("index.html", "text/html; charset=utf-8")
        if path == "/app.js":
            return self._send_static("app.js", "application/javascript; charset=utf-8")
        if path == "/app.css":
            return self._send_static("app.css", "text/css; charset=utf-8")
        if path == "/api/markers":
            return self._send_markers()
        if path == "/stream/video":
            return self._send_video()

        return self._send_404()

    def _send_static(self, name: str, content_type: str) -> None:
        p = self._static_root / name
        if not p.is_file():
            return self._send_404()
        data = p.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_markers(self) -> None:
        if not self._markers_path or not os.path.isfile(self._markers_path):
            return self._send_404()
        data = Path(self._markers_path).read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _send_video(self) -> None:
        vp = self._video_path
        if not vp or not os.path.isfile(vp):
            return self._send_404()
        file_size = os.path.getsize(vp)
        range_h = self.headers.get("Range")
        ctype, _ = mimetypes.guess_type(vp)
        if not ctype or ctype == "text/plain":
            ctype = "application/octet-stream"
        with open(vp, "rb") as f:
            if not range_h:
                self.send_response(200)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(file_size))
                self.send_header("Accept-Ranges", "bytes")
                self.end_headers()
                while True:
                    chunk = f.read(1024 * 1024)
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                return
            m = re.match(r"bytes=(\d*)-(\d*)", range_h.strip(), re.IGNORECASE)
            if not m:
                return self._send_404()
            a, b = m.group(1), m.group(2)
            if a == "" and b == "":
                return self._send_404()
            start = int(a) if a else 0
            end = int(b) if b else file_size - 1
            if start < 0 or end >= file_size or start > end:
                return self._send_404()
            clen = end - start + 1
            self.send_response(206, "Partial Content")
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(clen))
            self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
            self.send_header("Accept-Ranges", "bytes")
            self.end_headers()
            f.seek(start)
            remain = clen
            while remain > 0:
                n = min(1024 * 1024, remain)
                self.wfile.write(f.read(n))
                remain -= n


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--video", type=Path, required=True, help="源 mp4 等，白名单只读单文件")
    p.add_argument(
        "--markers", type=Path, required=True, help="segment 写出的 markers.json"
    )
    p.add_argument(
        "--host", default="127.0.0.1", help="默认仅本机回环，勿对公网暴露"
    )
    p.add_argument("--port", type=int, default=8765)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    vp = args.video.expanduser().resolve()
    mp = args.markers.expanduser().resolve()
    if not vp.is_file():
        raise SystemExit(f"not a file: {vp}")
    if not mp.is_file():
        raise SystemExit(f"not a file: {mp}")
    if not _STATIC.is_dir():
        raise SystemExit(f"static dir missing: {_STATIC}")
    global _VIDEO_PATH, _MARKERS_PATH
    _VIDEO_PATH = str(vp)
    _MARKERS_PATH = str(mp)
    AnnotateRequestHandler._video_path = _VIDEO_PATH
    AnnotateRequestHandler._markers_path = _MARKERS_PATH
    # Windows 上 Threading 利于 Range 与静态并发
    httpd = ThreadingHTTPServer(
        (args.host, int(args.port)), AnnotateRequestHandler
    )
    url = f"http://{args.host}:{args.port}/"
    print(f"annotate: video={_VIDEO_PATH}", flush=True)
    print(f"annotate: markers={_MARKERS_PATH}", flush=True)
    print(f"打开浏览器: {url}  (Ctrl+C 结束)", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nbye", flush=True)
        httpd.server_close()


if __name__ == "__main__":
    main()

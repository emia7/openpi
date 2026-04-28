#!/usr/bin/env python3
"""
将本地音频文件上传到阿里云 OSS，打印公网 HTTPS URL，供 ``segment_by_record_markers --file-url`` 使用。

依赖::

    pip install oss2

环境变量（可写入 ``umi_scripts_csw/.env``，勿提交）::

    OSS_ACCESS_KEY_ID
    OSS_ACCESS_KEY_SECRET
    OSS_ENDPOINT          例: https://oss-cn-hangzhou.aliyuncs.com
    OSS_BUCKET
    可选 OSS_OBJECT_PREFIX   对象名前缀，默认 ``nano-audio/``
    可选 OSS_PUBLIC_BASE     自定义域名或 CDN 根，例 ``https://static.example.com``（无尾斜杠）

未设置 ``OSS_PUBLIC_BASE`` 时，使用 virtual-hosted 形式
``https://<bucket>.<oss-host>/<key>``。百炼 filetrans 需能**直接 GET** 该 URL，请为对象配置
公共读、或先将 bucket/前缀设为可读。

用法（在 umi_scripts_csw 下）::

    python nano_sync/upload_wav_oss.py /path/to/track.wav
    python nano_sync/upload_wav_oss.py /path/to/track.wav --key mylab/session01.wav
"""

from __future__ import annotations

import os
import sys
import time
import uuid
from pathlib import Path
from urllib.parse import urlparse

_NS = Path(__file__).resolve().parent
_UMI = _NS.parent
for d in (_NS, _UMI):
    if str(d) not in sys.path:
        sys.path.insert(0, str(d))

from qwen_filetrans_asr import load_env_around_script  # noqa: E402


def _public_https_url(
    endpoint: str,
    bucket: str,
    key: str,
    public_base: str | None,
) -> str:
    key = key.lstrip("/")
    if public_base and public_base.strip():
        b = public_base.rstrip("/")
        return f"{b}/{key}"
    parsed = urlparse(endpoint)
    host = parsed.netloc or parsed.path.split("/")[0]
    if not host:
        raise SystemExit(f"无法从 OSS_ENDPOINT 解析 host: {endpoint!r}")
    return f"https://{bucket}.{host}/{key}"


def main() -> None:
    import argparse

    load_env_around_script()
    p = argparse.ArgumentParser(
        description="上传音频到阿里云 OSS，输出公网 HTTPS URL（给 --file-url）",
    )
    p.add_argument("path", type=Path, help="本地 WAV/MP3 等文件")
    p.add_argument(
        "--key",
        type=str,
        default=None,
        help="对象键（含路径）。默认: OSS_OBJECT_PREFIX + 时间戳 + 短 uuid + 原扩展名",
    )
    p.add_argument(
        "--public-base",
        type=str,
        default=None,
        help="覆盖环境变量 OSS_PUBLIC_BASE",
    )
    args = p.parse_args()
    path = args.path.expanduser()
    if not path.is_file():
        raise SystemExit(f"file not found: {path}")

    try:
        import oss2  # type: ignore[import-not-found]
    except ImportError as e:
        raise SystemExit(
            "需要 oss2: pip install oss2\n"
            "阿里云 OSS Python SDK: https://help.aliyun.com/zh/oss/developer-reference/preface-15/"
        ) from e

    access_key = os.environ.get("OSS_ACCESS_KEY_ID", "").strip()
    secret = os.environ.get("OSS_ACCESS_KEY_SECRET", "").strip()
    endpoint = os.environ.get("OSS_ENDPOINT", "").strip()
    bucket_name = os.environ.get("OSS_BUCKET", "").strip()
    prefix = (os.environ.get("OSS_OBJECT_PREFIX", "nano-audio/") or "").strip()
    if prefix and not prefix.endswith("/"):
        prefix += "/"
    public_base = (args.public_base or os.environ.get("OSS_PUBLIC_BASE", "")).strip() or None

    if not access_key or not secret:
        raise SystemExit("请设置 OSS_ACCESS_KEY_ID 与 OSS_ACCESS_KEY_SECRET")
    if not endpoint:
        raise SystemExit("请设置 OSS_ENDPOINT，例如 https://oss-cn-hangzhou.aliyuncs.com")
    if not bucket_name:
        raise SystemExit("请设置 OSS_BUCKET")

    ext = path.suffix.lower() or ".wav"
    if args.key:
        object_key = args.key.lstrip("/")
    else:
        object_key = (
            f"{prefix}{int(time.time())}_{uuid.uuid4().hex[:8]}{ext}"
        )

    auth = oss2.Auth(access_key, secret)
    if not endpoint.startswith("http"):
        endpoint = "https://" + endpoint.lstrip("/")
    bucket = oss2.Bucket(auth, endpoint, bucket_name)
    print(f"[oss] 上传: {path.name} -> {object_key!r}", flush=True)
    with open(path, "rb") as f:
        bucket.put_object(object_key, f)

    url = _public_https_url(endpoint, bucket_name, object_key, public_base)
    print(url, flush=True)


if __name__ == "__main__":
    main()

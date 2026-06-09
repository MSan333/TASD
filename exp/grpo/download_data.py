"""从 OSS 下载 GRPO v3 训练数据 + TipBank.

用法:
  # 下载到默认目录
  python -m exp.grpo.download_data

  # 指定 OSS 路径和本地目录
  python -m exp.grpo.download_data --oss_path oss://your-bucket/grpo/v3/ --local_dir exp/grpo/data/grpo_v3
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import subprocess
import sys

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_OSS_PATH = "oss://lazada-ads-algo/rank-agent/grpo/v3/"
DEFAULT_LOCAL_DIR = os.path.join(os.path.dirname(__file__), "data", "grpo_v3")


def download_from_oss(oss_path: str, local_dir: str) -> bool:
    """从 OSS 下载数据到本地."""
    ossutil_cmd = None
    for cmd in ["ossutil64", "ossutil"]:
        if shutil.which(cmd):
            ossutil_cmd = cmd
            break

    if not ossutil_cmd:
        logger.error("ossutil/ossutil64 未找到, 请安装 ossutil 或手动下载")
        logger.info("手动下载: ossutil64 cp -r %s %s/", oss_path, local_dir)
        return False

    os.makedirs(local_dir, exist_ok=True)
    if not oss_path.endswith("/"):
        oss_path += "/"

    full_cmd = [ossutil_cmd, "cp", "-r", oss_path, local_dir + "/", "--update"]
    logger.info("Downloading: %s", " ".join(full_cmd))

    try:
        result = subprocess.run(full_cmd, capture_output=True, text=True, timeout=600)
        if result.returncode == 0:
            logger.info("Download success: %s → %s", oss_path, local_dir)
            return True
        else:
            logger.error("Download failed: %s", result.stderr)
            return False
    except Exception as e:
        logger.error("Download error: %s", e)
        return False


def verify_data(local_dir: str) -> None:
    """验证下载的数据完整性."""
    required_files = ["train.parquet", "train.jsonl"]
    optional_dirs = ["tipbank"]

    for f in required_files:
        path = os.path.join(local_dir, f)
        if os.path.isfile(path):
            size_mb = os.path.getsize(path) / (1024 * 1024)
            logger.info("  [OK] %s (%.2f MB)", f, size_mb)
        else:
            logger.warning("  [MISSING] %s", f)

    for d in optional_dirs:
        path = os.path.join(local_dir, d)
        if os.path.isdir(path):
            n_files = len(os.listdir(path))
            logger.info("  [OK] %s/ (%d files)", d, n_files)
        else:
            logger.warning("  [MISSING] %s/", d)

    # Check tipbank also exists at package level
    pkg_tipbank = os.path.join(os.path.dirname(__file__), "tipbank")
    if os.path.isdir(pkg_tipbank):
        logger.info("  [OK] exp/grpo/tipbank/ (package-level, %d files)",
                    len(os.listdir(pkg_tipbank)))
    else:
        logger.warning("  [MISSING] exp/grpo/tipbank/ — copy from downloaded data or rank-agent")


def main():
    parser = argparse.ArgumentParser(description="下载 GRPO v3 训练数据")
    parser.add_argument("--oss_path", type=str, default=DEFAULT_OSS_PATH)
    parser.add_argument("--local_dir", type=str, default=DEFAULT_LOCAL_DIR)
    parser.add_argument("--verify_only", action="store_true",
                        help="只验证本地数据, 不下载")
    args = parser.parse_args()

    if args.verify_only:
        logger.info("验证本地数据: %s", args.local_dir)
        verify_data(args.local_dir)
        return

    success = download_from_oss(args.oss_path, args.local_dir)
    if success:
        logger.info("验证下载数据:")
        verify_data(args.local_dir)

        # 同步 tipbank 到 package level
        downloaded_tipbank = os.path.join(args.local_dir, "tipbank")
        pkg_tipbank = os.path.join(os.path.dirname(__file__), "tipbank")
        if os.path.isdir(downloaded_tipbank) and not os.path.isdir(pkg_tipbank):
            shutil.copytree(downloaded_tipbank, pkg_tipbank)
            logger.info("Synced tipbank to exp/grpo/tipbank/")


if __name__ == "__main__":
    main()

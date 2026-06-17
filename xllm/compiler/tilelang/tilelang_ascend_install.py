from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from scripts.build_support.env import set_npu_envs
from scripts.logger import logger

from .common.toolchain import find_installed_tilelang_root, prepare_tilelang_import, repo_root

PREPARE_ASCEND_COMMAND = "python xllm/compiler/tilelang_launcher.py prepare-ascend --target-platform <a2|a3|a5> --arch <arm|x86>"

TILELANG_ASCEND_WHEELS: dict[tuple[str, str], str] = {
    ("a2", "arm"): "https://gitcode.com/xLLM-AI/tilelang-ascend/releases/download/v0.1.1.010-release/tilelang-0.1.0%2Bascendc_pto.56d3f9b8.a2.cann850-cp311-cp311-linux_aarch64.whl",
    ("a2", "x86"): "https://gitcode.com/xLLM-AI/tilelang-ascend/releases/download/v0.1.1.010-release/tilelang-0.1.0%2Bascendc_pto.56d3f9b8.a2.cann850-cp311-cp311-linux_x86_64.whl",
    ("a3", "arm"): "https://gitcode.com/xLLM-AI/tilelang-ascend/releases/download/v0.1.1.010-release/tilelang-0.1.0%2Bascendc_pto.56d3f9b8.a3.cann850-cp311-cp311-linux_aarch64.whl",
}


REQUIRED_TILELANG_FILES = (
    "__init__.py",
    "lib/libtilelang_module.so",
    "lib/libtilelang.so",
    "lib/libtvm.so",
    "3rdparty/catlass/include/catlass/catlass.hpp",
    "3rdparty/shmem/include/shmem.h",
    "3rdparty/shmem/src/device/shmemi_device_common.h",
    "src/tl_templates/ascend/common.h",
)


def _resolve_wheel_location(target_platform: str, arch: str) -> str | Path:
    location = TILELANG_ASCEND_WHEELS.get((target_platform, arch))
    if location is None:
        raise RuntimeError(f"No TileLang Ascend wheel for platform={target_platform}, arch={arch}.")

    if location.startswith(("http://", "https://")):
        return location

    wheel = Path(location).expanduser()
    if not wheel.is_absolute():
        wheel = repo_root() / wheel
    wheel = wheel.resolve()
    if not wheel.is_file():
        raise RuntimeError(f"Cannot find TileLang Ascend wheel: {wheel}")
    return wheel


def tilelang_artifacts_ready(tilelang_root: str | Path) -> bool:
    tl_root = Path(tilelang_root).resolve()
    return all((tl_root / path).exists() for path in REQUIRED_TILELANG_FILES)


def ensure_ascend_ready() -> Path:
    set_npu_envs()
    tilelang_root = find_installed_tilelang_root()
    if tilelang_root is None:
        raise RuntimeError(f"tilelang package is not installed.\nRun `{PREPARE_ASCEND_COMMAND}` first.")
    if not tilelang_artifacts_ready(tilelang_root):
        raise RuntimeError(
            f"tilelang-ascend artifacts are missing under {tilelang_root}.\n"
            f"Run `{PREPARE_ASCEND_COMMAND}` first."
        )

    prepare_tilelang_import(tilelang_root)
    return tilelang_root


def prepare_ascend(target_platform: str, arch: str, *, force: bool = False) -> Path:
    set_npu_envs()
    try:
        import Cython  # noqa: F401
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-deps", "Cython"])

    tilelang_root = find_installed_tilelang_root()
    if not force and tilelang_root is not None and tilelang_artifacts_ready(tilelang_root):
        prepare_tilelang_import(tilelang_root)
        logger.info(f"tilelang is already installed and ready: {tilelang_root}")
        return tilelang_root

    wheel_location = _resolve_wheel_location(target_platform, arch)
    cmd = [sys.executable, "-m", "pip", "install", "--no-deps"]
    if force or tilelang_root is not None:
        cmd.append("--force-reinstall")
    logger.info(f"Installing TileLang Ascend wheel: {wheel_location}")
    subprocess.check_call(cmd + [str(wheel_location)])
    return ensure_ascend_ready()

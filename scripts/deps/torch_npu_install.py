"""torch_npu wheel management for xLLM NPU builds.

Ensures the correct torch_npu version is installed before compilation.
Auto-installs or upgrades as needed — mirrors the TileLang pattern in
xllm/compiler/tilelang/tilelang_ascend_install.py.
"""

from __future__ import annotations

import importlib.metadata
import os
import platform
import subprocess
import sys
import tempfile
import urllib.request

from scripts.logger import logger

TORCH_NPU_WHEELS: dict[tuple[str, str], str] = {
    # (cann_version, arch) → wheel URL (from xLLM-AI/pytorch releases)
    ("9.0", "arm"): (
        "https://gitcode.com/xLLM-AI/pytorch/releases/download/"
        "v2.9.0-26.0.0.0/"
        "torch_npu-2.9.0.post2%2Bgita5f47a6-cp311-cp311-linux_aarch64.whl"
    ),
    ("9.0", "x86"): (
        "https://gitcode.com/xLLM-AI/pytorch/releases/download/"
        "v2.9.0-26.0.0.0/"
        "torch_npu-2.9.0.post2%2Bgita5f47a6-cp311-cp311-linux_x86_64.whl"
    ),
}

# PEP 440 compliant local filename for pip install (upstream wheel name
# contains URL-encoded '+' that pip 26+ may reject).
_LOCAL_WHEEL_NAMES = {
    "arm": "torch_npu-2.9.0.post2-cp311-cp311-linux_aarch64.whl",
    "x86": "torch_npu-2.9.0.post2-cp311-cp311-linux_x86_64.whl",
}

REQUIRED_VERSION = "2.9.0.post2+gita5f47a6"


def _get_cann_version() -> str:
    """Detect CANN major.minor from toolkit path."""
    toolkit = os.environ.get(
        "NPU_TOOLKIT_HOME", "/usr/local/Ascend/ascend-toolkit/latest"
    )
    version_file = os.path.join(toolkit, "version.cfg")
    try:
        with open(version_file) as f:
            for line in f:
                if "=" in line:
                    _, ver = line.strip().split("=", 1)
                    parts = ver.split(".")
                    if len(parts) >= 2:
                        return f"{parts[0]}.{parts[1]}"
    except (OSError, ValueError):
        pass
    return "9.0"


def _get_arch() -> str:
    machine = platform.machine()
    if machine in ("aarch64", "arm64"):
        return "arm"
    return "x86"


def _installed_version() -> str | None:
    try:
        return importlib.metadata.version("torch_npu")
    except importlib.metadata.PackageNotFoundError:
        return None


def _resolve_wheel(cann_version: str, arch: str) -> str:
    key = (cann_version, arch)
    url = TORCH_NPU_WHEELS.get(key)
    if url is None:
        raise RuntimeError(
            f"No torch_npu wheel configured for CANN {cann_version}, arch={arch}.\n"
            f"Available: {list(TORCH_NPU_WHEELS.keys())}"
        )
    return url


def _download_and_install(wheel_url: str) -> None:
    """Download wheel, rename to PEP 440 compliant name, then pip install."""
    arch = _get_arch()
    local_name = _LOCAL_WHEEL_NAMES[arch]
    with tempfile.TemporaryDirectory() as tmpdir:
        local_path = os.path.join(tmpdir, local_name)
        logger.info(f"Downloading {wheel_url} ...")
        urllib.request.urlretrieve(wheel_url, local_path)
        subprocess.check_call([
            sys.executable, "-m", "pip", "install",
            "--force-reinstall", "--no-deps", local_path,
        ])


def ensure_torch_npu_ready() -> None:
    """Ensure torch_npu matches REQUIRED_VERSION; install or upgrade if not."""
    installed = _installed_version()
    if installed is not None and REQUIRED_VERSION in installed:
        logger.info(f"torch_npu {installed} is ready.")
        return

    cann_version = _get_cann_version()
    arch = _get_arch()
    wheel_url = _resolve_wheel(cann_version, arch)

    action = f"Upgrading torch_npu {installed}" if installed else "Installing torch_npu"
    logger.info(f"{action} → {REQUIRED_VERSION}")

    _download_and_install(wheel_url)

    new_version = _installed_version()
    if new_version is None or REQUIRED_VERSION not in new_version:
        raise RuntimeError(
            f"torch_npu installation failed: expected {REQUIRED_VERSION}, "
            f"got {new_version}"
        )
    logger.info(f"torch_npu {new_version} installed successfully.")

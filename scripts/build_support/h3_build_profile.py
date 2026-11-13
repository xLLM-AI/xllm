# Copyright 2026 The xLLM Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Explicit, device-free validation of the qualified H3 A3 build environment."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import os
import platform
import subprocess
import sys
import tempfile
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from scripts.logger import logger

H3_BUILD_PROFILE = "h3_a3_rf_cann91_cp312"


@dataclass(frozen=True)
class H3BuildProfile:
    name: str = H3_BUILD_PROFILE
    host_arch: str = "aarch64"
    python_version: tuple[int, int] = (3, 12)
    cann_version: str = "9.1"
    torch_version: str = "2.10.0+cpu"
    torch_npu_version: str = "2.10.0.post4+git5dd8ef3"
    tilelang_version: str = "0.1.4+ubuntu.24.3.cann910"
    torch_npu_wheel: str = (
        "https://gitcode.com/xLLM-AI/pytorch/releases/download/v2.10.0-26.1.0alpha001/"
        "torch_npu-2.10.0.post4%2Bgit5dd8ef3-cp312-cp312-linux_aarch64.whl"
    )
    torch_npu_sha256: str = "7ec86a4f2a403533aa561178ddf1b6c5f389a9f079dc362286c92bd5d9c92e9f"
    tilelang_wheel: str = (
        "https://github.com/tile-ai/tilelang-ascend/releases/download/TileLang-ascend-v0.1.2.000-release/"
        "tilelang-0.1.4%2Bubuntu.24.3.cann910-cp312-cp312-linux_aarch64.whl"
    )
    tilelang_sha256: str = "6545d48550959eef4208c934d09c5a529ce720ff8a8bcd3e0f9e5a7687138922"


def resolve_h3_build_profile(name: str | None) -> H3BuildProfile | None:
    if name is None or name == "native":
        return None
    if name != H3_BUILD_PROFILE:
        raise ValueError(f"Unknown xLLM build profile: {name!r}")
    return H3BuildProfile()


def _cann_version() -> str:
    roots = [os.environ.get(key) for key in ("NPU_TOOLKIT_HOME", "ASCEND_TOOLKIT_HOME", "ASCEND_HOME_PATH")]
    roots.append("/usr/local/Ascend/ascend-toolkit/latest")
    for value in dict.fromkeys(root for root in roots if root):
        root = Path(value)
        for path in (root / "version.cfg", *sorted(root.glob("*/ascend_toolkit_install.info"))):
            if not path.is_file():
                continue
            for line in path.read_text(encoding="utf-8").splitlines():
                key, separator, version = line.strip().partition("=")
                if separator and key == "version":
                    parts = version.split(".")
                    if len(parts) >= 2 and all(part.isdigit() for part in parts[:2]):
                        return ".".join(parts[:2])
    raise RuntimeError("Cannot identify CANN version for the qualified H3 build profile")


def _require_package(name: str, expected: str) -> None:
    try:
        actual = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError as error:
        raise RuntimeError(f"H3 build profile requires {name}=={expected}; package is missing") from error
    if actual != expected:
        raise RuntimeError(f"H3 build profile requires {name}=={expected}, found {actual}")


def validate_h3_build_profile(
    profile: H3BuildProfile,
    *,
    device: str = "a3",
    require_torch_npu: bool = True,
    require_tilelang: bool = True,
) -> None:
    if device != "a3":
        raise ValueError("The H3 RainFusion build profile requires Ascend A3")
    if sys.version_info[:2] != profile.python_version:
        raise RuntimeError("The H3 RainFusion build profile requires Python cp312")
    if platform.machine() != profile.host_arch:
        raise RuntimeError(f"The H3 RainFusion build profile requires host arch {profile.host_arch}")
    actual_cann = _cann_version()
    if actual_cann != profile.cann_version:
        raise RuntimeError(f"H3 build profile requires CANN {profile.cann_version}, found {actual_cann}")
    _require_package("torch", profile.torch_version)
    if require_torch_npu:
        _require_package("torch_npu", profile.torch_npu_version)
    if require_tilelang:
        _require_package("tilelang", profile.tilelang_version)


def validate_h3_tilelang_root(profile: H3BuildProfile, root: Path) -> None:
    """Prevent a TL_ROOT override from silently compiling a different DSL runtime."""
    _require_package("tilelang", profile.tilelang_version)
    distribution = importlib.metadata.distribution("tilelang")
    for file in distribution.files or ():
        if str(file) == "tilelang/__init__.py":
            installed = Path(file.locate()).resolve().parent
            if root.resolve() != installed:
                raise RuntimeError(f"H3 build profile requires installed TileLang root {installed}, found {root}")
            return
    raise RuntimeError("Installed TileLang distribution has no tilelang/__init__.py record")


def install_locked_wheel(location: str, expected_sha256: str) -> None:
    """Verify local or downloaded bytes before invoking pip; preserve wheel tags."""
    if len(expected_sha256) != 64 or any(char not in "0123456789abcdef" for char in expected_sha256):
        raise ValueError("A complete lowercase SHA256 is required for a qualified wheel")
    parsed = urllib.parse.urlparse(location)
    name = urllib.parse.unquote(Path(parsed.path).name)
    if not name.endswith("-cp312-cp312-linux_aarch64.whl"):
        raise ValueError(f"Unqualified H3 wheel filename: {name}")
    with tempfile.TemporaryDirectory(prefix="xllm-h3-wheel-") as temporary:
        if parsed.scheme in ("http", "https"):
            wheel = Path(temporary) / name
            urllib.request.urlretrieve(location, wheel)
        elif not parsed.scheme:
            wheel = Path(location).expanduser().resolve()
        else:
            raise ValueError(f"Unsupported wheel location scheme: {parsed.scheme}")
        digest = hashlib.sha256()
        with wheel.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != expected_sha256:
            raise ValueError(f"H3 wheel SHA256 mismatch: {wheel.name}")
        logger.info(f"Installing verified H3 wheel: {wheel.name} sha256={expected_sha256}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-deps", "--force-reinstall", str(wheel)])


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True, choices=[H3_BUILD_PROFILE])
    parser.add_argument("--device", default="a3")
    args = parser.parse_args(argv)
    profile = resolve_h3_build_profile(args.profile)
    assert profile is not None
    validate_h3_build_profile(profile, device=args.device)
    logger.info(f"Qualified H3 build environment verified: {profile.name}")


if __name__ == "__main__":
    main()

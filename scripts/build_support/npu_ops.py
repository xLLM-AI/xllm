# Copyright 2026 The xLLM Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/xLLM-AI/xllm/blob/main/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Collect the custom operator vendor used by an NPU build into its wheel."""

import shutil
import subprocess
import tempfile
from collections.abc import Mapping
from pathlib import Path

from scripts.logger import logger

_BUNDLED_VENDOR = "custom_xllm_math"
_VERSION_MARKER = ".xllm_ops_git_head"


def _source_git_head(base_dir: str) -> str:
    source = Path(base_dir) / "third_party" / "xllm_ops"
    if not (source / ".git").exists():
        raise RuntimeError(f"Initialize the xllm_ops submodule before packaging NPU operators: {source}")
    try:
        return subprocess.check_output(
            ["git", "-c", f"safe.directory={source}", "-C", str(source), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.PIPE,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError(f"Cannot determine the xllm_ops source revision at {source}") from error


def _vendor_path(env: Mapping[str, str]) -> Path:
    opp_root = env.get("ASCEND_OPP_PATH")
    if opp_root:
        opp = Path(opp_root).expanduser().resolve()
    elif env.get("ASCEND_HOME_PATH"):
        opp = (Path(env["ASCEND_HOME_PATH"]).expanduser() / "opp").resolve()
    else:
        raise RuntimeError("ASCEND_OPP_PATH or ASCEND_HOME_PATH is required to package NPU operators")

    # CMake links the operator API from NPU_HOME_PATH, while its revision
    # marker is written under ASCEND_OPP_PATH. Both must describe this build.
    npu_home = env.get("NPU_HOME_PATH")
    if npu_home and (Path(npu_home).expanduser() / "opp").resolve() != opp:
        raise RuntimeError("NPU_HOME_PATH and ASCEND_OPP_PATH select different operator installations")
    # Match CMake's revision-marker selection, including when both layouts
    # remain installed after a toolkit upgrade. Do not select a different
    # vendor just because the build-selected installation is incomplete.
    vendors = opp / "vendors"
    current_vendor = vendors / "custom_xllm_math"
    if (current_vendor / "op_api/include/aclnnop").is_dir():
        return current_vendor
    return vendors / "xllm"


def _validate_vendor(vendor: Path, source_git_head: str) -> None:
    for relative_path in (_VERSION_MARKER, "op_api/lib/libcust_opapi.so", "bin/set_env.bash", "version.info"):
        resource = vendor / relative_path
        if not resource.is_file() or resource.stat().st_size == 0:
            raise RuntimeError(f"Incomplete NPU operator vendor: missing or empty {resource}")

    installed_git_head = (vendor / _VERSION_MARKER).read_text(encoding="utf-8").strip()
    if installed_git_head != source_git_head:
        raise RuntimeError(
            f"NPU operator revision mismatch at {vendor}: installed {installed_git_head}, source {source_git_head}"
        )

    resources = (
        ("op_proto", "*.so*"),
        ("op_impl/ai_core/tbe/config", "*.json"),
        ("op_impl/ai_core/tbe/kernel", "*.o"),
        ("op_impl/ai_core/tbe/op_tiling", "*.so*"),
    )
    for relative_path, pattern in resources:
        directory = vendor / relative_path
        if not any(resource.is_file() and resource.stat().st_size for resource in directory.rglob(pattern)):
            raise RuntimeError(f"Incomplete NPU operator vendor: no {pattern} resources under {directory}")


def stage_npu_ops_vendor(base_dir: str, extdir: str, device: str, env: Mapping[str, str]) -> None:
    """Ship the matching vendor, including metadata, kernels and host libraries.

    Normalize both supported installation names to the wheel's fixed vendor
    path, shared by its RPATH and runtime registration. Copy the entire
    installation and materialize symbolic links: a wheel must
    not depend on resources remaining in the build machine's CANN directory.
    Other backends remove this build-owned subtree to avoid stale NPU assets
    after reusing a setuptools build directory.
    """
    destination_root = Path(extdir) / "opp"
    if destination_root.is_symlink() or destination_root.is_file():
        destination_root.unlink()
    elif destination_root.exists():
        shutil.rmtree(destination_root)
    if device != "npu":
        return

    vendor = _vendor_path(env)
    source_git_head = _source_git_head(base_dir)
    if not vendor.is_dir():
        raise RuntimeError(f"NPU operator vendor was not installed by this build: {vendor}")

    Path(extdir).mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".xllm-ops-", dir=extdir) as temporary_root:
        staged_vendor = Path(temporary_root) / _BUNDLED_VENDOR
        try:
            shutil.copytree(vendor, staged_vendor, symlinks=False)
        except OSError as error:
            raise RuntimeError(f"Failed to collect the complete NPU operator vendor from {vendor}: {error}") from error
        _validate_vendor(staged_vendor, source_git_head)
        destination = destination_root / "vendors" / _BUNDLED_VENDOR
        destination.parent.mkdir(parents=True)
        staged_vendor.rename(destination)
    logger.info("Staged complete NPU operator vendor from %s into %s", vendor, destination)

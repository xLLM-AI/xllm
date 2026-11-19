# Copyright 2026 The xLLM Authors.
#
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

"""Static gate or read-only target inventory. Does not start a service/kernel."""

from __future__ import annotations

import argparse
import ast
import ctypes
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def _static_checks() -> Any:
    manifest = json.loads((ROOT / "reference_manifest.json").read_text(encoding="utf-8"))
    for record in manifest["files"]:
        actual = hashlib.sha256((ROOT / record["path"]).read_bytes()).hexdigest()
        if actual != record["sha256"]:
            raise ValueError(f"Reference changed: {record['path']}")
    for path in ROOT.rglob("*"):
        if any(p in {".git", "node_modules", "build", "research"} for p in path.relative_to(ROOT).parts):
            continue
        if path.is_file() and path.suffix in {".sh", ".cmd"}:
            raw = path.read_bytes()
            if b"\r" in raw or raw.startswith(b"\xef\xbb\xbf"):
                raise ValueError(f"Linux script must be LF, UTF-8 without BOM: {path}")
            raw.decode("utf-8")
    return {
        "reference_unchanged": True,
        "script_encoding": "UTF-8 no BOM, LF",
        "original_manifest_valid": manifest["original_manifest_valid"],
    }


def _command(argv: Any) -> Any:
    try:
        result = subprocess.run(argv, capture_output=True, text=True, timeout=30)
        return {"command": argv, "exit_code": result.returncode, "stdout": result.stdout, "stderr": result.stderr}
    except (OSError, subprocess.SubprocessError) as exc:
        return {"command": argv, "error": str(exc)}


def _inventory() -> Any:
    data = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "executable": sys.executable,
        "environment": {
            k: os.environ.get(k)
            for k in (
                "ASCEND_HOME_PATH",
                "ASCEND_RT_VISIBLE_DEVICES",
                "ASCEND_CUSTOM_OPP_PATH",
                "T5_OPP_ROOT",
                "T5_ACLNN_EXECUTOR_REUSE",
                "TASK_QUEUE_ENABLE",
                "T5_FUNCTIONAL_DEVICE_SCOPE",
                "T5_PHYSICAL_DEVICE",
                "T5_CONTROL_REPORT",
                "T5_CONTROL_SOLUTION",
                "T5_CONTROL_COMMIT",
            )
        },
        "commands": [_command(x) for x in (["npu-smi", "info"], ["ss", "-lntp"], ["ps", "-eo", "pid,ppid,comm,args"])],
        "versions": {},
    }
    for name in ("torch", "torch_npu", "numpy", "ml_dtypes"):
        try:
            from importlib.metadata import version

            data["versions"][name] = version(name)
        except Exception as exc:
            data["versions"][name] = {"unavailable": str(exc)}
    # Package metadata omits the git suffix; read the shipped version source
    # without creating an NPU context merely to inventory the installation.
    try:
        from importlib.metadata import distribution

        source = Path(distribution("torch_npu").locate_file("torch_npu/version.py"))
        for node in ast.parse(source.read_text(encoding="utf-8")).body:
            if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "git_version" for t in node.targets
            ):
                data["torch_npu_git_version"] = ast.literal_eval(node.value)
    except Exception as exc:
        data["torch_npu_git_version"] = {"unavailable": str(exc)}
    data["listening_tcp_ports"] = sorted(
        {
            int(line.split()[1].split(":")[1], 16)
            for filename in ("/proc/net/tcp", "/proc/net/tcp6")
            if Path(filename).exists()
            for line in Path(filename).read_text().splitlines()[1:]
            if line.split()[3] == "0A"
        }
    )
    for path in (
        Path("/etc/os-release"),
        Path("/usr/local/Ascend/driver/version.info"),
        Path(os.environ.get("ASCEND_HOME_PATH", "/nonexistent")) / "version.info",
    ):
        if path.is_file():
            data[str(path)] = path.read_text(encoding="utf-8", errors="replace")
    return data


def _target_inventory() -> Any:
    # Official read-only runtime queries; no set-device, allocation or kernel.
    acl = ctypes.CDLL("libascendcl.so")
    acl.aclInit.argtypes = [ctypes.c_char_p]
    acl.aclrtGetSocName.restype = ctypes.c_char_p
    acl.aclrtGetDeviceCount.argtypes = [ctypes.POINTER(ctypes.c_uint32)]
    rc = acl.aclInit(None)
    if rc != 0:
        raise RuntimeError(f"aclInit failed: {rc}")
    try:
        count = ctypes.c_uint32()
        rc = acl.aclrtGetDeviceCount(ctypes.byref(count))
        if rc != 0:
            raise RuntimeError(f"aclrtGetDeviceCount failed: {rc}")
        name = acl.aclrtGetSocName()
        return {"soc": name.decode() if name else None, "visible_device_count": count.value}
    finally:
        rc = acl.aclFinalize()
        if rc != 0:
            raise RuntimeError(f"aclFinalize failed: {rc}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--static", action="store_true")
    p.add_argument("--require-soc")
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    result = _static_checks()
    if not args.static:
        result["inventory"] = _inventory()
    if args.require_soc:
        result["target"] = _target_inventory()
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=True, indent=2))
    if args.require_soc:
        assert result["target"]["soc"] == args.require_soc, "Wrong SoC for fixed acceptance"
        assert result["target"]["visible_device_count"] == 1, "Select exactly one runtime device"

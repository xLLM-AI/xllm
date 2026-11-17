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

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output must not exist")
    result = {"python": sys.version, "platform": platform.platform(), "commands": {}, "cann_files": {}}
    for command in (["npu-smi", "info"], ["c++", "--version"], ["ninja", "--version"]):
        try:
            process = subprocess.run(command, capture_output=True, text=True, timeout=30)
            record = {"returncode": process.returncode, "stdout": process.stdout, "stderr": process.stderr}
        except (OSError, subprocess.TimeoutExpired) as exc:
            record = {"error": repr(exc)}
        result["commands"][" ".join(command)] = record
    result["packages"] = {}
    for name in ("torch", "torch-npu", "triton-ascend", "numpy", "ninja"):
        try:
            result["packages"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result["packages"][name] = None
    roots = [Path("/usr/local/Ascend/ascend-toolkit/latest"), Path("/usr/local/Ascend/ascend-toolkit/9.1.0")]
    for key in ("ASCEND_HOME_PATH", "ASCEND_TOOLKIT_HOME"):
        if os.environ.get(key):
            roots.append(Path(os.environ[key]))
    for root in roots:
        if root.is_dir():
            for pattern in ("version.cfg", "*-linux/ascend_toolkit_install.info", "*-linux/version.info"):
                for file in root.glob(pattern):
                    result["cann_files"][str(file.resolve())] = file.read_text(errors="replace")
    try:
        import torch
        import torch_npu
        import triton

        torch.npu.set_device(0)
        result["device"] = torch.npu.get_device_name(0)
        result["triton_target"] = str(triton.runtime.driver.active.get_current_target())
    except Exception as exc:
        result["device_error"] = repr(exc)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if "device_error" in result or any(value is None for value in result["packages"].values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()

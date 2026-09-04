# Copyright 2026 The xLLM Authors. All Rights Reserved.
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

import importlib.util
import sys
from pathlib import Path
from types import ModuleType


def _load_extension(module_path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("xllm_export", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to create import spec for {module_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules["xllm_export"] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop("xllm_export", None)
        raise
    return module


def main() -> None:
    if len(sys.argv) != 2:
        raise RuntimeError("Expected the xllm_export shared-library path")

    module = _load_extension(Path(sys.argv[1]))
    if not hasattr(module, "LLMMaster"):
        raise RuntimeError("xllm_export loaded without LLMMaster")
    if not hasattr(sys.modules.get("xllm_runtime"), "AttentionMetadataView"):
        raise RuntimeError("xllm_runtime was not registered")
    if not hasattr(sys.modules.get("xllm_weight_loader"), "StateDict"):
        raise RuntimeError("xllm_weight_loader was not registered")


if __name__ == "__main__":
    main()

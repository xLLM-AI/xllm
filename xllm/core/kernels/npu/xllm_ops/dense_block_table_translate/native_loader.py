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

import json
from functools import lru_cache
from pathlib import Path
from types import ModuleType


@lru_cache(maxsize=1)
def load_native() -> ModuleType:
    from torch.utils.cpp_extension import load

    return load(
        name="dense_block_table_native",
        sources=[str(Path(__file__).with_name("native_dense.cpp"))],
        extra_cflags=["-O3", "-std=c++17"],
        verbose=True,
    )


if __name__ == "__main__":
    import torch_npu

    load_native()
    print(json.dumps({"build": "PASS"}), flush=True)

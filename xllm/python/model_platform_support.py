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

"""Python model support declared by implementation and platform."""

MODEL_PLATFORM_SUPPORT: dict[str, dict[str, bool]] = {
    "qwen3": {"cuda": True, "npu": True},
    "qwen3_5": {"cuda": True, "npu": False},
    "qwen3_vl": {"cuda": False, "npu": True},
    "deepseek_v32": {"cuda": False, "npu": True},
    "glm5_2": {"cuda": False, "npu": True},
    "deepseek_v32_mtp": {"cuda": False, "npu": True},
}

# Copyright 2025-2026 The xLLM Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/jd-opensource/xllm/blob/main/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Model-runner subpackage: graph capture / replay over the pure-GPU model
graph. Mirrors SGLang's ``model_executor`` runners; a model constructs a
``GraphRunner`` and calls it, staying free of graph-capture concerns.
"""

from .graph_runner import GraphRunner, TC_BACKEND_ENV, maybe_compile

__all__ = ["GraphRunner", "TC_BACKEND_ENV", "maybe_compile"]

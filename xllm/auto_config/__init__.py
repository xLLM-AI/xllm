# Copyright 2026 The xLLM Authors. All Rights Reserved.
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
# ==============================================================================

"""Per-model-type auto-tuning profiles for `xllm serve`.

Each supported model type ships two files named after its `model_type`
(read from the model directory's config.json):

* `<model_type>.json` — the base "optimal" startup config, using the JSON
  keys documented in docs/.../cli_reference.md.
* `<model_type>.py` — a tuning module exposing a top-level
  `tune(base_config: dict, context: dict) -> dict` callable that adapts the
  base config to the current machine (device count, hardware/arch).

The launcher (xllm/launch_server.py) resolves this directory relative to its
own location so the profiles are found both in the source tree and after a
wheel install.
"""

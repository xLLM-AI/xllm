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

"""Auto-tuning profile for the `qwen3` model type.

The launcher loads the base `qwen3.json` config, builds a `context` describing
the current machine, and calls the module-level `tune(base_config, context)`.
`tune` delegates to `Qwen3Tuner`, a `BaseTuner` subclass that implements the
two mandatory hooks (`tune_common` and `tune_npu`) and returns an adjusted copy
of the base config, which the launcher then writes next to the launch command
and passes to the xllm binary via `--config_json_file`.

Shared machinery (`BaseTuner`, `Platform`, `detect_hardware`,
`check_device_count`) lives in `xllm.auto_config.utils`; this module only holds
the qwen3 tuning policy.
"""

from __future__ import annotations

from typing import Any, Dict

from scripts.logger import logger
from xllm.auto_config.utils import BaseTuner, CpuArchEnum, Platform


class Qwen3Tuner(BaseTuner):
    """qwen3 auto-tuning policy.

    Only the two mandatory hooks are implemented: `tune_common` (topology and
    ARM adjustments) and `tune_npu`. Other platforms fall back to `BaseTuner`'s 
    no-op defaults until qwen3 has validated tuning for them.
    """

    MODEL_TYPE = "qwen3"

    def tune_common(
        self,
        config: Dict[str, Any],
        context: Dict[str, Any],
    ) -> None:
        # Align the launch topology with the devices visible on this host.
        visible_device_count = context.get("visible_device_count")
        if isinstance(visible_device_count, int) and visible_device_count > 0:
            config["nnodes"] = visible_device_count

        # ARM hosts get a smaller prefill batch regardless of accelerator.
        if Platform.get_cpu_architecture() == CpuArchEnum.ARM:
            config["max_tokens_per_batch"] = min(
                config.get("max_tokens_per_batch", 8192), 4096
            )

    def tune_npu(
        self,
        config: Dict[str, Any],
        context: Dict[str, Any],
    ) -> None:
        # TODO
        pass


def tune(base_config: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    """Launcher entry point: adapt the base qwen3 config to this machine."""
    return Qwen3Tuner().tune(base_config, context)

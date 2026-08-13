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

"""Auto-tuning machinery for `xllm serve`.

Hosts the per-model tuning base class (`BaseTuner`) and the launcher helpers
that resolve a model's tuning profile and generate its tuned config.

Hardware typing (`Platform`, `PlatformEnum`, `CpuArchEnum`, `detect_hardware`)
lives in `xllm/python/platform.py`, the single home shared with the C++ worker.
The launcher has no hard `torch` dependency and cannot `import
xllm.python.platform` normally -- that would execute `xllm/python/__init__.py`,
which imports `torch` and binds a kernel package. So this module loads that one
file by path (the same technique `load_tuning_module` uses for tuning profiles)
and re-exports the symbols, keeping the tuning profiles' imports unchanged.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import os
from collections.abc import Sequence
from types import ModuleType
from typing import Any

from scripts.logger import logger


def _load_platform_module() -> ModuleType:
    """Load `xllm/python/platform.py` by file path, dodging its package init.

    `platform.py` only imports the stdlib and the shared logger, so it loads
    cleanly here without pulling in `torch` or a kernel package. A private
    module name keeps it from clashing with the worker's real
    `xllm.python.platform`.
    """
    platform_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.realpath(__file__))),
        "python",
        "platform.py",
    )
    spec = importlib.util.spec_from_file_location("xllm.auto_config._platform", platform_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"failed to load platform module: {platform_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_platform_module = _load_platform_module()

# Re-export the shared hardware-typing surface so tuning profiles keep importing
# it from `xllm.auto_config.utils` while the definitions live in one place.
Platform = _platform_module.Platform
PlatformEnum = _platform_module.PlatformEnum
CpuArchEnum = _platform_module.CpuArchEnum
current_platform = _platform_module.current_platform
detect_hardware = _platform_module.detect_hardware


def check_device_count(
    model_type: str,
    base_config: dict[str, Any],
    context: dict[str, Any],
) -> bool:
    """Check the visible device count against the profile's optimal `nnodes`.

    Shared across model profiles: `model_type` only labels the log lines.
    Returns True when they match. A mismatch is non-fatal: it is logged as a
    warning so the operator can decide whether to adjust the launch topology.
    """
    optimal_nnodes = base_config.get("nnodes")
    visible_device_count = context.get("visible_device_count")

    if optimal_nnodes is None or visible_device_count is None:
        logger.warning(
            "%s auto-tuning: cannot verify device count (optimal nnodes=%s, visible devices=%s)",
            model_type,
            optimal_nnodes,
            visible_device_count,
        )
        return False

    if visible_device_count != optimal_nnodes:
        logger.warning(
            "%s auto-tuning: visible device count %s does not match the "
            "profile's optimal nnodes %s; the tuned config keeps nnodes=%s.",
            model_type,
            visible_device_count,
            optimal_nnodes,
            visible_device_count,
        )
        return False

    logger.info(
        "%s auto-tuning: visible device count %s matches optimal nnodes %s.",
        model_type,
        visible_device_count,
        optimal_nnodes,
    )
    return True


class BaseTuner:
    """Base class for per-model auto-tuning profiles.

    A profile subclasses this, sets `MODEL_TYPE`, and implements the two
    mandatory hooks -- `tune_common` (cross-platform adjustments) and
    `tune_npu` (the primary, validated target). The remaining platform hooks
    (`tune_cuda`, `tune_mlu`, `tune_musa`, `tune_dcu`, `tune_ilu`) have base
    no-op defaults, so a profile overrides one only when it has platform
    specific tuning; otherwise the base config is used unchanged there.

    `tune()` is the fixed orchestration entry point the launcher calls (via
    each profile module's `tune()` wrapper):

    1. deep-copy the base config (the input is never mutated),
    2. log the detected hardware and check the device count,
    3. apply cross-platform adjustments in `tune_common()`,
    4. dispatch to the platform-specific hook selected from `Platform.enum()`.

    The hooks mutate the working config in place (they return `None`); `tune()`
    owns and returns the copy.
    """

    # Subclasses MUST override with their config.json / <type>.py base name.
    MODEL_TYPE: str = "base"

    def tune(
        self,
        base_config: dict[str, Any],
        context: dict[str, Any],
    ) -> dict[str, Any]:
        tuned_config = copy.deepcopy(base_config)

        hardware = context.get("hardware") or detect_hardware()
        logger.info(
            "%s auto-tuning: detected hardware device_type=%s chip=%s arch=%s.",
            self.MODEL_TYPE,
            hardware.get("device_type"),
            hardware.get("chip"),
            hardware.get("arch"),
        )

        check_device_count(self.MODEL_TYPE, base_config, context)

        self.tune_common(tuned_config, context)
        self._dispatch_platform(tuned_config, context)
        return tuned_config

    def _dispatch_platform(
        self,
        config: dict[str, Any],
        context: dict[str, Any],
    ) -> None:
        dispatch = {
            PlatformEnum.NPU: self.tune_npu,
            PlatformEnum.CUDA: self.tune_cuda,
            PlatformEnum.MLU: self.tune_mlu,
            PlatformEnum.MUSA: self.tune_musa,
            PlatformEnum.DCU: self.tune_dcu,
            PlatformEnum.ILU: self.tune_ilu,
        }
        hook = dispatch.get(Platform.enum())
        if hook is None:
            logger.warning(
                "%s auto-tuning: no tuning hook for platform %s; using base config unchanged.",
                self.MODEL_TYPE,
                Platform.enum(),
            )
            return
        hook(config, context)

    def tune_common(
        self,
        config: dict[str, Any],
        context: dict[str, Any],
    ) -> None:
        """Cross-platform adjustments applied before the platform hook.

        Mandatory: a profile with no cross-platform tuning implements this as a
        one-line `pass`.
        """
        pass

    def tune_npu(
        self,
        config: dict[str, Any],
        context: dict[str, Any],
    ) -> None:
        """Tune for Ascend NPU. Mutate `config` in place."""
        pass

    def tune_cuda(
        self,
        config: dict[str, Any],
        context: dict[str, Any],
    ) -> None:
        """Tune for NVIDIA CUDA. Base default: no change. Override as needed."""
        pass

    def tune_mlu(
        self,
        config: dict[str, Any],
        context: dict[str, Any],
    ) -> None:
        """Tune for Cambricon MLU. Base default: no change. Override as needed."""
        pass

    def tune_musa(
        self,
        config: dict[str, Any],
        context: dict[str, Any],
    ) -> None:
        """Tune for Moore Threads MUSA. Base default: no change. Override as needed."""
        pass

    def tune_dcu(
        self,
        config: dict[str, Any],
        context: dict[str, Any],
    ) -> None:
        """Tune for Hygon DCU. Base default: no change. Override as needed."""
        pass

    def tune_ilu(
        self,
        config: dict[str, Any],
        context: dict[str, Any],
    ) -> None:
        """Tune for Iluvatar GPU. Base default: no change. Override as needed."""
        pass


class AutoTuningError(Exception):
    """Raised when auto-tuning cannot produce a config.

    The launcher catches this and surfaces the message via `parser.error`, so
    these functions stay independent of argparse.
    """


def auto_tuning_config_dir() -> str:
    """Directory holding the per-model tuning profiles (`xllm/auto_config`)."""
    return os.path.dirname(os.path.realpath(__file__))


def extract_model_path(extra_args: Sequence[str]) -> str | None:
    """Recover `--model` from the launcher's passthrough args.

    `--model` is forwarded to the binary rather than parsed by the launcher, so
    it must be read out of `extra_args`. Supports `--model VALUE` and
    `--model=VALUE`.
    """
    for index, arg in enumerate(extra_args):
        if arg == "--model":
            if index + 1 < len(extra_args):
                return extra_args[index + 1]
            return None
        if arg.startswith("--model="):
            return arg[len("--model=") :]
    return None


def read_model_type(model_path: str) -> str:
    """Read `model_type` (fallback `model_name`) from `<model_path>/config.json`.

    Mirrors C++ `util::get_model_type`. Raises `AutoTuningError` on any problem.
    """
    config_path = os.path.join(os.path.realpath(os.path.expanduser(model_path)), "config.json")
    try:
        with open(config_path, encoding="utf-8") as config_file:
            model_config = json.load(config_file)
    except FileNotFoundError:
        raise AutoTuningError(f"model config.json not found: {config_path}")
    except json.JSONDecodeError as error:
        raise AutoTuningError(f"failed to parse {config_path}: {error}")
    except OSError as error:
        raise AutoTuningError(f"failed to read {config_path}: {error}")

    if not isinstance(model_config, dict):
        raise AutoTuningError(f"{config_path} must contain a JSON object")

    model_type = model_config.get("model_type") or model_config.get("model_name")
    if not isinstance(model_type, str) or not model_type:
        raise AutoTuningError(f"{config_path} must contain a string `model_type` or `model_name`")
    return model_type


def load_tuning_module(py_path: str, model_type: str) -> ModuleType:
    """Import a `<model_type>.py` tuning profile by file path.

    Loading by path avoids triggering `xllm/__init__`'s lazy `xllm_export` .so
    load. Raises `AutoTuningError` if the module cannot be imported or does not
    expose a callable `tune`.
    """
    spec = importlib.util.spec_from_file_location(f"xllm.auto_config.{model_type}", py_path)
    if spec is None or spec.loader is None:
        raise AutoTuningError(f"failed to load tuning module: {py_path}")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as error:
        raise AutoTuningError(f"failed to import {py_path}: {error}")
    if not callable(getattr(module, "tune", None)):
        raise AutoTuningError(f"{py_path} must define a callable `tune(base_config, context)`")
    return module


def generate_tuned_config(extra_args: Sequence[str], output_dir: str) -> str:
    """Generate a tuned JSON config for the model and write it to `output_dir`.

    Resolves the model's `model_type`, loads its base config and tuning
    profile, applies the profile's `tune()`, and writes
    `<output_dir>/<model_type>.tuned.json`. Returns the written path. Raises
    `AutoTuningError` on any failure.
    """
    model_path = extract_model_path(extra_args)
    if not model_path:
        raise AutoTuningError("auto-tuning requires --model <path>")

    model_type = read_model_type(model_path)

    config_dir = auto_tuning_config_dir()
    base_json_path = os.path.join(config_dir, f"{model_type}.json")
    tuning_py_path = os.path.join(config_dir, f"{model_type}.py")
    if not os.path.isfile(base_json_path) or not os.path.isfile(tuning_py_path):
        raise AutoTuningError(
            f"auto-tuning is not supported for model_type `{model_type}`: "
            f"expected both {base_json_path} and {tuning_py_path} to exist."
        )

    try:
        with open(base_json_path, encoding="utf-8") as base_file:
            base_config = json.load(base_file)
    except (OSError, json.JSONDecodeError) as error:
        raise AutoTuningError(f"failed to read {base_json_path}: {error}")
    if not isinstance(base_config, dict):
        raise AutoTuningError(f"{base_json_path} must contain a JSON object")

    module = load_tuning_module(tuning_py_path, model_type)

    detect_hardware_fn = getattr(module, "detect_hardware", None)
    hardware = detect_hardware_fn() if callable(detect_hardware_fn) else None
    context = {
        "model_path": model_path,
        "model_type": model_type,
        "visible_device_count": Platform.get_device_count(),
        "hardware": hardware,
    }

    try:
        tuned_config = module.tune(base_config, context)
    except Exception as error:
        raise AutoTuningError(f"{tuning_py_path} tune() failed: {error}")
    if not isinstance(tuned_config, dict):
        raise AutoTuningError(f"{tuning_py_path} tune() must return a dict")

    output_path = os.path.join(output_dir, f"{model_type}.tuned.json")
    try:
        with open(output_path, "w", encoding="utf-8") as output_file:
            json.dump(tuned_config, output_file, indent=2)
            output_file.write("\n")
    except OSError as error:
        raise AutoTuningError(f"failed to write tuned config {output_path}: {error}")

    logger.info("auto-tuning: wrote tuned config for %s to %s", model_type, output_path)
    return output_path

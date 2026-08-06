---
title: "Auto Config Development Guide"
sidebar:
  order: 5
---

## Background

xLLM's **Auto Config** feature lets `xllm serve`, at startup, generate a tuned JSON launch config based on the model type (`model_type`) and the current machine's hardware (accelerator kind, chip generation, visible device count, CPU architecture), then launch the server with it.

From the user's side it is a single switch:

```bash
xllm serve --model /path/to/model --enable-auto-tuning-gflags
```

The launcher then:

1. reads `model_type` from the model directory's `config.json`;
2. locates the built-in base config `xllm/auto_config/<model_type>.json` and the tuning script `xllm/auto_config/<model_type>.py`;
3. detects the current hardware and calls the script's `tune()` to produce a tuned config;
4. writes the result to `<model_type>.tuned.json` in the **current working directory** and passes it to the xllm binary via `--config_json_file` to start the service.

> If a `model_type` has no matching `.json` + `.py`, the launcher fails and exits (no implicit fallback). So **adapting a new model = adding these two files**.

This guide is for developers and explains how to:

- **Step 0**: export a base JSON config to start from;
- add a tuning profile for a **new model** (`<model_type>.json` + `<model_type>.py`);
- adapt a **new hardware** by extending detection in `Platform` and filling in each model's `tune_<platform>` hooks.

All the code lives under `xllm/auto_config/`:

| File | Purpose |
|:-----|:--------|
| `utils.py` | Shared machinery: `Platform` hardware typing, the `BaseTuner` base class, the `generate_tuned_config` flow, etc. |
| `<model_type>.json` | The model's **base** optimal config (keys match the CLI launch parameters) |
| `<model_type>.py` | The model's tuning policy, subclassing `BaseTuner` |

---

## Step 0: Export a base JSON config

You don't need to hand-write `<model_type>.json` for a new model. The most reliable approach is to **first launch the service normally with the command-line flags you have already tuned, let xllm dump the effective config to JSON**, then trim that down into your base.

The xllm binary provides two gflags (see the `ConfigJsonUtils` section of [Service Startup Parameters](/en/cli_reference/)):

| Parameter | Description |
|:----------|:------------|
| `enable_dump_config_json` | Whether to dump the resolved startup config as JSON |
| `dump_config_json_file` | Output path, default `xllm_config.json`, only used when `enable_dump_config_json=true` |

Launch the service with your usual optimal flags plus the dump switches:

```bash
xllm serve --model /path/to/Qwen3-0.6B \
  --enable_dump_config_json=true \
  --dump_config_json_file=qwen3.dumped.json \
  <the other flags you normally tune with...>
```

Once the service is up, `qwen3.dumped.json` contains the **effective** full config.

> **Note**: the dumped JSON only contains fields that **differ from the default** (see `APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT` in `config_utils.cpp`). This conveniently narrows things down to the tuning items actually worth pinning.

Turn that dump into the base `<model_type>.json`:

- keep only the tuning keys you actually want to pin (e.g. `block_size`, `max_tokens_per_batch`, `enable_graph`, ...);
- add `nnodes` (the profile's "optimal node count", used for the device-count check) and `port` (optional, used for local multi-rank port computation);
- drop fields tied to a specific machine/path that should not be pinned.

---

## 1. Adapting a new model

Suppose the new model's `config.json` has `model_type` equal to `my_model`. Two files are needed.

### 1. Base config `xllm/auto_config/my_model.json`

Derived from Step 0 and trimmed. Keys must be valid xllm launch parameters (see [Service Startup Parameters](/en/cli_reference/)). Example (modeled on `qwen3.json`):

```json
{
  "nnodes": 1,
  "port": 8010,
  "block_size": 128,
  "max_memory_utilization": 0.9,
  "enable_prefix_cache": true,
  "max_tokens_per_batch": 8192,
  "max_seqs_per_batch": 256,
  "enable_schedule_overlap": true,
  "enable_chunked_prefill": true,
  "max_tokens_per_chunk_for_prefill": 2048,
  "npu_kernel_backend": "ATB",
  "enable_graph": true,
  "max_tokens_for_graph_mode": 2048
}
```

### 2. Tuning script `xllm/auto_config/my_model.py`

Create a class that subclasses `BaseTuner`, and provide a module-level `tune()` wrapper as the launcher entry point.

The `BaseTuner` contract:

- **`tune()` is the fixed orchestration entry point** (do not override it). It: deep-copies the base config → logs the detected hardware → checks the device count → calls `tune_common()` → dispatches to the platform-specific hook selected from `Platform.enum()` → returns the tuned copy.
- Each hook **mutates the passed-in `config` in place** (returns `None`); do not return a new object.
- Only **two abstract methods are mandatory**: `tune_common` (cross-platform adjustments) and `tune_npu` (the primary validated target). When there is no cross-platform adjustment, implement `tune_common` as a one-line `pass`.
- `tune_cuda` / `tune_mlu` / `tune_musa` / `tune_dcu` / `tune_ilu` have **no-op defaults** in the base class; override one only when the model has dedicated tuning for that platform.

The `context` dict passed to the hooks contains:

| Key | Meaning |
|:----|:--------|
| `model_path` | The model path from `--model` |
| `model_type` | The resolved model type |
| `visible_device_count` | Visible/usable device count (from `Platform.get_device_count()`) |
| `hardware` | `{"arch", "device_type", "chip"}`, provided by `detect_hardware()` |

Full example (modeled on `qwen3.py`):

```python
# Copyright 2026 The xLLM Authors. All Rights Reserved.
# ...(reuse the repo's standard Apache-2.0 header)...

from __future__ import annotations

from typing import Any, Dict

from scripts.logger import logger
from xllm.auto_config.utils import BaseTuner, CpuArchEnum, Platform


class MyModelTuner(BaseTuner):
    """Auto-tuning policy for my_model."""

    MODEL_TYPE = "my_model"

    def tune_common(
        self,
        config: Dict[str, Any],
        context: Dict[str, Any],
    ) -> None:
        # Align the launch topology with the devices visible on this host.
        visible_device_count = context.get("visible_device_count")
        if isinstance(visible_device_count, int) and visible_device_count > 0:
            config["nnodes"] = visible_device_count

        # ARM hosts get a smaller prefill batch (accelerator-independent).
        if Platform.get_cpu_architecture() == CpuArchEnum.ARM:
            config["max_tokens_per_batch"] = min(
                config.get("max_tokens_per_batch", 8192), 4096
            )

    def tune_npu(
        self,
        config: Dict[str, Any],
        context: Dict[str, Any],
    ) -> None:
        # Example: graph mode is only validated on Ascend A2 (910b); disable
        # it on other generations.
        if Platform.get_ascend_soc_generation() != "a2":
            config["enable_graph"] = False
            logger.info(
                "%s auto-tuning: graph mode only validated on Ascend A2; "
                "disabling on chip=%s.",
                self.MODEL_TYPE,
                Platform.get_npu_chip(),
            )


def tune(base_config: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    """Launcher entry point: adapt the base config to this machine."""
    return MyModelTuner().tune(base_config, context)
```

> **Conventions**: `MODEL_TYPE` must match the file name and the `model_type` in `config.json`; the module-level `tune(base_config, context)` must exist and be callable — the launcher relies on it as the unified entry point.

### 3. Verify

After adding the files, verify the generation flow with `--dry-run` first (it does not actually start the service):

```bash
xllm serve --model /path/to/my_model \
  --enable-auto-tuning-gflags --dry-run
```

Expected: a log line `auto-tuning: wrote tuned config for my_model to <cwd>/my_model.tuned.json`, and the printed launch command carries `--config_json_file=<cwd>/my_model.tuned.json`. Inspect the generated `my_model.tuned.json` to confirm it matches expectations.

> **Packaging note**: the whole `xllm/auto_config/` directory is bundled into the wheel by `setup.py` (both `.py` and `.json`), and the launcher locates it relative to its own path, so it is found both from the source tree and after a wheel install. New files require no changes to the packaging script.

---

## 2. Adapting a new hardware

Hardware adaptation has two layers: the **detection layer** (`Platform` can recognize the new hardware) and the **policy layer** (each model's `tune_<platform>` hook for the new platform).

### 1. Detection layer: extend `Platform` (`utils.py`)

`Platform` is a streamlined hardware-typing facade that uses **torch-optional layered detection**: prefer the framework runtime (`torch` / `torch_npu` / `torch_mlu` ...), and fall back to visible-device env masks, `npu-smi`, etc. when it is unavailable. It never raises in any environment.

`PlatformEnum` currently supports: `CUDA / NPU / MLU / MUSA / DCU / ILU / CPU / UNSPECIFIED`. To add a new accelerator:

1. **Add the enum member**: add a member to `PlatformEnum`, e.g. `XPU = enum.auto()`.
2. **Add the visible-device env var**: add an entry `PlatformEnum.XPU: "XPU_VISIBLE_DEVICES"` to `_VISIBLE_DEVICE_ENV_VARS` (used for torch-less fallback detection and device counting).
3. **Add the detection logic**: add a branch for the corresponding framework runtime in `_torch_device_type()` (align the order and style with `scripts/build_support/utils.py::get_device_type`).
4. **Add a typing method** (optional but recommended): add an `is_xpu()` modeled on `is_npu()` for convenience in the hooks.
5. If chip generation/model matters, add query methods modeled on `get_npu_chip()` / `get_ascend_soc_generation()`.

`Platform`'s existing public methods can be used directly in the hooks: `enum()`, `is_cuda()/is_npu()/...`, `device_type()`, `get_cpu_architecture()`, `get_cpu_arch_str()`, `get_npu_chip()`, `get_ascend_soc_generation()`, `get_device_count()`, `get_device_name()`.

### 2. Dispatch layer: `BaseTuner`'s dispatch table

`BaseTuner._dispatch_platform()` looks up a `PlatformEnum -> hook` table by `Platform.enum()`. **When adding a new platform, remember to add a row to this table**, e.g.:

```python
dispatch = {
    PlatformEnum.NPU: self.tune_npu,
    PlatformEnum.CUDA: self.tune_cuda,
    ...
    PlatformEnum.XPU: self.tune_xpu,   # new
}
```

And add a hook `tune_xpu` with a **no-op default implementation** to `BaseTuner` (so existing models need no changes and take the safe "base config unchanged" path on the new platform):

```python
def tune_xpu(
    self,
    config: Dict[str, Any],
    context: Dict[str, Any],
) -> None:
    """Tune for XPU. Base default: no change. Override as needed."""
    pass
```

> If a platform has no hook in the dispatch table, `_dispatch_platform` logs a warning and returns the base config unchanged — the fallback behavior for an "unadapted platform".

### 3. Policy layer: override the hook in the model profile

In models that need dedicated tuning for the new hardware, override the corresponding `tune_xpu`:

```python
class MyModelTuner(BaseTuner):
    MODEL_TYPE = "my_model"

    def tune_common(self, config, context) -> None:
        ...

    def tune_npu(self, config, context) -> None:
        ...

    def tune_xpu(self, config, context) -> None:
        # XPU-specific tuning; mutate config in place
        config["some_flag"] = True
```

Models with no dedicated tuning need not override it and automatically take the base class's no-op default.

---

## Design highlights

- **Add files, don't touch the flow**: adapting a new model = adding `<model_type>.json` + `<model_type>.py`; the `generate_tuned_config()` flow is fully generic.
- **Two mandatory hooks**: `tune_common` + `tune_npu` are abstract methods; omitting one raises `TypeError` at instantiation. The other platform hooks have base defaults.
- **In-place mutation semantics**: hooks mutate `config` itself and return `None`; `tune()` owns the deep copy and the return.
- **Hardware typing centralized in `Platform`**: new hardware is recognized in one place, and models reuse it through `Platform.*` and the `tune_<platform>` hooks.
- **No implicit fallback**: a `model_type` with no profile fails outright; a platform with no hook warns and uses the base config.

Reference implementation: `xllm/auto_config/qwen3.py`, `xllm/auto_config/qwen3.json`, `xllm/auto_config/utils.py`.

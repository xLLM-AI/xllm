---
title: "自动调优配置 (Auto Config) 开发指南"
sidebar:
  order: 5
---

## 背景

xLLM 的 **Auto Config（自动调优配置）** 功能让 `xllm serve` 在启动时，根据模型类型（`model_type`）和当前机器的硬件（加速器种类、芯片代际、可见设备数、CPU 架构），自动生成一份调优后的 JSON 启动配置，再用它拉起服务。

用户侧只需一个开关：

```bash
xllm serve --model /path/to/model --enable-auto-tuning-gflags
```

启动器会：

1. 从模型目录的 `config.json` 读出 `model_type`；
2. 找到内置的基线配置 `xllm/auto_config/<model_type>.json` 和调优脚本 `xllm/auto_config/<model_type>.py`；
3. 探测当前硬件，调用脚本里的 `tune()` 生成调优后的配置；
4. 把结果写到**当前工作目录**下的 `<model_type>.tuned.json`，并通过 `--config_json_file` 传给 xllm 二进制启动服务。

> 如果某个 `model_type` 没有对应的 `.json` + `.py`，启动器会直接报错退出（不做隐式回退）。因此**为新模型适配 = 补齐这两个文件**。

本文面向开发者，说明如何：

- **第 0 步**：导出一份基线 JSON 配置作为起点；
- 为**新模型**新增一份调优 profile（`<model_type>.json` + `<model_type>.py`）；
- 为**新硬件**在 `Platform` 中扩展探测、并在各模型的 `tune_<platform>` 钩子里补上调优策略。

相关代码都在 `xllm/auto_config/` 下：

| 文件 | 作用 |
|:-----|:-----|
| `utils.py` | `Platform` 硬件判型、`BaseTuner` 基类、`generate_tuned_config` 生成流程等公共机制 |
| `<model_type>.json` | 该模型的**基线**最优配置（键名与 CLI 启动参数一致） |
| `<model_type>.py` | 该模型的调优策略，继承 `BaseTuner` |

---

## 第 0 步：导出一份基线 JSON 配置

新模型的 `<model_type>.json` 不用手写。最可靠的做法是**先用你已经调好的命令行参数正常启动一次服务，让 xllm 把最终生效的配置导出成 JSON**，再以它为基线裁剪。

xllm 二进制提供了两个 gflags（参见 [服务启动参数](/zh/cli_reference/) 的 `ConfigJsonUtils` 一节）：

| 参数 | 说明 |
|:-----|:-----|
| `enable_dump_config_json` | 是否把最终生效的启动配置导出为 JSON |
| `dump_config_json_file` | 导出路径，默认 `xllm_config.json`，仅在 `enable_dump_config_json=true` 时生效 |

用你平时的最优参数启动服务，并加上导出开关：

```bash
xllm serve --model /path/to/Qwen3-0.6B \
  --enable_dump_config_json=true \
  --dump_config_json_file=qwen3.dumped.json \
  <你平时调优用的其它参数...>
```

服务起来后，`qwen3.dumped.json` 里就是**最终生效**的完整配置。

> **注意**：导出的 JSON 只包含**与默认值不同**的字段（导出逻辑见 `config_utils.cpp` 的 `APPEND_CONFIG_JSON_VALUE_IF_NOT_DEFAULT`）。这正好方便你挑出真正需要固化的调优项。

把这份导出结果整理成基线 `<model_type>.json`：

- 只保留你确实想固化的调优键（例如 `block_size`、`max_tokens_per_batch`、`enable_graph` 等）；
- 补上 `nnodes`（该 profile 的“最优节点数”，供设备数校验用）和 `port`（可选，供本地多 rank 计算端口）；
- 删掉与具体机器/路径强绑定、不该固化的字段。

---

## 一、为新模型适配

假设新模型的 `config.json` 里 `model_type` 为 `my_model`，需要新增两个文件。

### 1. 基线配置 `xllm/auto_config/my_model.json`

由第 0 步导出并裁剪得到。键名必须是合法的 xllm 启动参数（详见 [服务启动参数](/zh/cli_reference/)）。示例（参考 `qwen3.json`）：

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

### 2. 调优脚本 `xllm/auto_config/my_model.py`

新建一个继承 `BaseTuner` 的类，并提供一个模块级 `tune()` 包装函数作为启动器入口。

`BaseTuner` 的契约：

- **`tune()` 是固定的编排入口**（不要重写），它会：deepcopy 基线配置 → 打印硬件信息 → 校验设备数 → 调用 `tune_common()` → 按 `Platform.enum()` 分发到对应平台钩子 → 返回调优后的副本。
- 各钩子**原地修改**传入的 `config`（返回 `None`），不要返回新对象。
- **必须实现**的抽象方法只有两个：`tune_common`（跨平台调整）和 `tune_npu`（主验证目标）。没有跨平台调整时把 `tune_common` 写成 `pass` 即可。
- `tune_cuda` / `tune_mlu` / `tune_musa` / `tune_dcu` / `tune_ilu` 在基类里有**空默认实现**，只在该模型于对应平台有专门调优时才重写。

传入钩子的 `context` 字典包含：

| 键 | 含义 |
|:-----|:-----|
| `model_path` | `--model` 指定的模型路径 |
| `model_type` | 解析出的模型类型 |
| `visible_device_count` | 可见/可用设备数（来自 `Platform.get_device_count()`） |
| `hardware` | `{"arch", "device_type", "chip"}`，由 `detect_hardware()` 提供 |

完整示例（参考 `qwen3.py`）：

```python
# Copyright 2026 The xLLM Authors. All Rights Reserved.
# ...（沿用仓库统一的 Apache-2.0 头）...

from __future__ import annotations

from typing import Any, Dict

from scripts.logger import logger
from xllm.auto_config.utils import BaseTuner, CpuArchEnum, Platform


class MyModelTuner(BaseTuner):
    """my_model 的自动调优策略。"""

    MODEL_TYPE = "my_model"

    def tune_common(
        self,
        config: Dict[str, Any],
        context: Dict[str, Any],
    ) -> None:
        # 把启动拓扑对齐到本机实际可见的设备数。
        visible_device_count = context.get("visible_device_count")
        if isinstance(visible_device_count, int) and visible_device_count > 0:
            config["nnodes"] = visible_device_count

        # ARM 主机上收窄 prefill 批大小（与加速器无关的调整）。
        if Platform.get_cpu_architecture() == CpuArchEnum.ARM:
            config["max_tokens_per_batch"] = min(
                config.get("max_tokens_per_batch", 8192), 4096
            )

    def tune_npu(
        self,
        config: Dict[str, Any],
        context: Dict[str, Any],
    ) -> None:
        # 示例：图模式只在 Ascend A2(910b) 上验证过，其它代际关掉。
        if Platform.get_ascend_soc_generation() != "a2":
            config["enable_graph"] = False
            logger.info(
                "%s auto-tuning: graph mode only validated on Ascend A2; "
                "disabling on chip=%s.",
                self.MODEL_TYPE,
                Platform.get_npu_chip(),
            )


def tune(base_config: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    """启动器入口：把基线配置适配到当前机器。"""
    return MyModelTuner().tune(base_config, context)
```

> **约定**：`MODEL_TYPE` 必须与文件名、`config.json` 里的 `model_type` 保持一致；模块级 `tune(base_config, context)` 必须存在且可调用——启动器就是靠它作为统一入口。

### 3. 验证

新增文件后，用 `--dry-run` 先验证生成流程（不会真正拉起服务）：

```bash
xllm serve --model /path/to/my_model \
  --enable-auto-tuning-gflags --dry-run
```

预期：日志里出现 `auto-tuning: wrote tuned config for my_model to <cwd>/my_model.tuned.json`，打印的启动命令带 `--config_json_file=<cwd>/my_model.tuned.json`。检查生成的 `my_model.tuned.json` 内容是否符合预期。

> **打包说明**：`xllm/auto_config/` 整个目录会被 `setup.py` 打进 wheel（`.py` 和 `.json` 都包含），启动器按相对自身的路径定位，源码树运行和 wheel 安装后都能找到。新增文件无需改打包脚本。

---

## 二、为新硬件适配

硬件适配分两层：**判型层**（`Platform` 能识别新硬件）和 **策略层**（各模型在新平台上的 `tune_<platform>` 钩子）。

### 1. 判型层：扩展 `Platform`（`utils.py`）

`Platform` 是一个精简的硬件判型门面，采用 **torch-optional 分层探测**：优先用框架运行时（`torch` / `torch_npu` / `torch_mlu` …），拿不到再回退到可见设备环境变量、`npu-smi` 等，任何环境都不抛异常。

当前 `PlatformEnum` 支持：`CUDA / NPU / MLU / MUSA / DCU / ILU / CPU / UNSPECIFIED`。若要新增一种加速器：

1. **加枚举**：在 `PlatformEnum` 里新增成员，例如 `XPU = enum.auto()`。
2. **加可见设备环境变量**：在 `_VISIBLE_DEVICE_ENV_VARS` 里补一条 `PlatformEnum.XPU: "XPU_VISIBLE_DEVICES"`（用于无 torch 时的回退探测和设备计数）。
3. **加探测逻辑**：在 `_torch_device_type()` 里补上对应框架运行时的探测分支（对齐 `scripts/build_support/utils.py::get_device_type` 的顺序与写法）。
4. **加判型方法**（可选但推荐）：仿照 `is_npu()` 加一个 `is_xpu()`，方便钩子里调用。
5. 如涉及芯片代际/型号，可仿照 `get_npu_chip()` / `get_ascend_soc_generation()` 增加相应查询方法。

`Platform` 现有的公共方法可直接在钩子里使用：`enum()`、`is_cuda()/is_npu()/...`、`device_type()`、`get_cpu_architecture()`、`get_cpu_arch_str()`、`get_npu_chip()`、`get_ascend_soc_generation()`、`get_device_count()`、`get_device_name()`。

### 2. 分发层：`BaseTuner` 的分发表

`BaseTuner._dispatch_platform()` 按 `Platform.enum()` 查一张 `PlatformEnum -> 钩子` 的表。**新增一种平台时，记得在这张表里补一行**，例如：

```python
dispatch = {
    PlatformEnum.NPU: self.tune_npu,
    PlatformEnum.CUDA: self.tune_cuda,
    ...
    PlatformEnum.XPU: self.tune_xpu,   # 新增
}
```

并在 `BaseTuner` 里为新平台加一个**带空默认实现**的钩子 `tune_xpu`（这样已有模型无需改动就能在新平台上走“基线配置不变”的安全路径）：

```python
def tune_xpu(
    self,
    config: Dict[str, Any],
    context: Dict[str, Any],
) -> None:
    """Tune for XPU. Base default: no change. Override as needed."""
    pass
```

> 如果某平台在分发表里查不到钩子，`_dispatch_platform` 会打印一条 warning 并原样返回基线配置——这是“未适配平台”的兜底行为。

### 3. 策略层：在模型 profile 里重写钩子

在需要为新硬件做专门调优的模型里，重写对应的 `tune_xpu`：

```python
class MyModelTuner(BaseTuner):
    MODEL_TYPE = "my_model"

    def tune_common(self, config, context) -> None:
        ...

    def tune_npu(self, config, context) -> None:
        ...

    def tune_xpu(self, config, context) -> None:
        # 针对 XPU 的调优，原地修改 config
        config["some_flag"] = True
```

没有专门调优的模型可以不重写，自动走基类的空默认实现。

---

## 设计要点小结

- **只加文件，不改流程**：为新模型适配 = 加 `<model_type>.json` + `<model_type>.py`；生成流程 `generate_tuned_config()` 完全通用。
- **两个必需钩子**：`tune_common` + `tune_npu` 是抽象方法，漏写会在实例化时报 `TypeError`；其余平台钩子有基类默认。
- **原地修改语义**：钩子改 `config` 本身、返回 `None`；`tune()` 负责 deepcopy 与返回。
- **硬件判型集中在 `Platform`**：新硬件的识别只在一处扩展，各模型通过 `Platform.*` 与 `tune_<platform>` 钩子复用。
- **不隐式回退**：`model_type` 无 profile 直接报错；平台无钩子则告警并用基线配置。

参考实现：`xllm/auto_config/qwen3.py`、`xllm/auto_config/qwen3.json`、`xllm/auto_config/utils.py`。

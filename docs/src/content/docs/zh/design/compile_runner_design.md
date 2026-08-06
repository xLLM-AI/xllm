---
title: "Compile Runner 多后端扩展设计文档"
sidebar:
  order: 4
---

## 概述

xLLM 的 Python Model Execution 层通过 `torch.compile` 提供图编译优化能力。当前实现以
`InductorRunner` 命名，仅支持单一 backend 字符串透传，未覆盖 `torch.compile` 的完整
配置空间。本文将 `InductorRunner` 扩展为 `CompileRunner`，支持多 backend（inductor、
torchair）、`fullgraph` 和 `dynamic` 编译选项。

本文档面向需要理解实现决策和扩展方式的开发者，重点说明：

- 为什么需要扩展 `InductorRunner`
- 多 backend 的配置模型和编译路径差异
- 配置如何从 C++ 传递到 Python runner
- 对现有 executor 调度逻辑的影响

本文档的设计目标包括：

- 统一 `torch.compile` 多 backend 的配置和构造逻辑
- 保持与现有 `ModelExecutor` 调度路径的兼容
- 为后续接入新 backend 提供可扩展的构造模式

本文档的非目标包括：

- 不改变 decode graph（CUDA Graph / ACL Graph）的 capture/replay 机制
- 不修改 eager runner 的行为
- 不覆盖 `torch.compile` 内部的编译优化细节

相关设计文档：

- [C++ Serving Framework + Python Model Execution 架构决策](/zh/design/cpp_framework_python_model_architecture/)
- [Graph Mode 设计文档](/zh/design/graph_mode_design/)

## 1. 背景和问题

### 1.1 当前实现

`InductorRunner` 在 `runners/inductor.py` 中实现，构造逻辑如下：

```python
class InductorRunner(BaseRunner):
    def __init__(self, model, attention_backend, device, backend: str) -> None:
        super().__init__(model, attention_backend, device)
        self.compiled_model = torch.compile(model, backend=backend)
```

`ModelExecutor` 在 `executor.py` 中根据 `python_graph_backend` 配置创建 runner：

```python
else:
    from xllm.python.model_executor.runners.inductor import InductorRunner
    self.inductor_runner = InductorRunner(
        execution_model, self.attention_backend, device, graph_backend
    )
```

### 1.2 问题

当前实现存在三个限制：

1. **Backend 覆盖不完整**：`torch.compile` 支持多种 backend，但 `InductorRunner` 的命名
   和实现暗示只服务于 inductor。接入 torchair（Ascend NPU GE 后端）时，需要特殊的
   backend 构造逻辑（`torchair.CompilerConfig()` + `torchair.get_npu_backend()`），
   当前无法表达。

2. **缺少 `fullgraph` 配置**：`torch.compile` 的 `fullgraph=True` 要求整图捕获，能获
   得更彻底的图级优化。当前实现未传递该参数，使用 PyTorch 默认值（`False`），可能导
   致 graph break 和性能损失。

3. **缺少 `dynamic` 配置**：`torch.compile` 的 `dynamic` 参数控制是否启用动态 shape 编
   译。对于 shape 稳定的推理场景，`dynamic=False` 可以获得更好的编译优化和运行时性能；
   对于 shape 多变的场景，`dynamic=True` 可以避免重复编译。当前实现未暴露该选项。

## 2. 设计

### 2.1 类重命名

将 `InductorRunner` 重命名为 `CompileRunner`，文件名从 `inductor.py` 改为
`compile_runner.py`。重命名的原因是该 runner 的职责是"通过 `torch.compile` 编译模
型并执行"，不局限于某一个特定 backend。

`ModelExecutor` 中的属性名同步从 `inductor_runner` 改为 `compile_runner`。

### 2.2 配置模型

新增两个 C++ `ExecutionConfig` 属性，与已有的 `python_graph_backend` 配合使用：

| 配置项 | 类型 | 默认值 | 说明 |
|:-------|:-----|:-------|:-----|
| `python_graph_backend` | `string` | `"off"` | 已有字段。值为 `"inductor"` 或 `"torchair"` 时创建 `CompileRunner` |
| `python_compile_fullgraph` | `bool` | `false` | 是否启用 `torch.compile(fullgraph=True)` 整图捕获 |
| `python_compile_dynamic` | `bool` | `false` | 是否启用 `torch.compile(dynamic=True)` 动态 shape 编译 |

配置传递路径：

```text
C++ gflags / JSON config
    → ExecutionConfig (python_graph_backend, python_compile_fullgraph, python_compile_dynamic)
    → py_causal_lm.cpp build_config_dict()
    → Python config dict
    → ModelExecutor.__init__()
    → CompileRunner.__init__()
```

### 2.3 Backend 构造差异

不同 backend 的构造逻辑不同，`CompileRunner` 在 `__init__` 中按 backend 名称分发：

**inductor**：

```python
compiled_model = torch.compile(
    model, backend="inductor", fullgraph=fullgraph, dynamic=dynamic
)
```

**torchair**：

```python
import torchair
config = torchair.CompilerConfig()
npu_backend = torchair.get_npu_backend(compiler_config=config)
compiled_model = torch.compile(
    model, backend=npu_backend, fullgraph=fullgraph, dynamic=dynamic
)
```

torchair 的 backend 不是字符串，而是通过 `torchair.get_npu_backend()` 返回的 callable
对象。这是 torchair 与 inductor 的关键差异——torchair 需要在 Python 侧构造 backend
对象，而不能仅通过字符串名称指定。

### 2.4 CompileRunner 构造逻辑

```python
class CompileRunner(BaseRunner):
    def __init__(
        self, model, attention_backend, device,
        backend: str, fullgraph: bool = False, dynamic: bool = False,
    ) -> None:
        super().__init__(model, attention_backend, device)
        compile_backend = self._resolve_compile_backend(backend)
        self.compiled_model = torch.compile(
            model, backend=compile_backend, fullgraph=fullgraph, dynamic=dynamic
        )

    @staticmethod
    def _resolve_compile_backend(backend: str):
        if backend == "torchair":
            import torchair
            config = torchair.CompilerConfig()
            return torchair.get_npu_backend(compiler_config=config)
        return backend
```

### 2.5 ModelExecutor 调度逻辑

`ModelExecutor.__init__` 中的 `CompileRunner` 创建逻辑更新为：

```python
else:
    from xllm.python.model_executor.runners.compile_runner import CompileRunner
    self.compile_runner = CompileRunner(
        execution_model, self.attention_backend, device,
        backend=graph_backend,
        fullgraph=bool(config.get("python_compile_fullgraph", False)),
        dynamic=bool(config.get("python_compile_dynamic", False)),
    )
```

`execute()` 中的调度逻辑保持不变——`compile_runner` 优先于 `eager_runner`，与原来
`inductor_runner` 的优先级一致。

### 2.6 与 decode graph runner 的关系

`CompileRunner` 和 decode graph runner（`DecodeCudaGraphRunner` /
`DecodeAclGraphRunner`）是两条独立的执行路径：

- **decode graph runner**：在 decode 阶段通过 CUDA Graph / ACL Graph 的 capture/replay
  减少 Host 调度开销。由 `enable_graph` 控制，backend 为 `"cudagraphs"` 或
  `"aclgraph"`。
- **CompileRunner**：通过 `torch.compile` 对模型进行图编译优化，覆盖 prefill 和 decode
  全阶段。由 `python_graph_backend` 控制。

两者可以在同一 `ModelExecutor` 中共存：decode 阶段优先使用 decode graph runner（如
果 `can_execute` 返回 `True`），其余步骤使用 `CompileRunner` 或 `EagerRunner`。

## 3. 修改范围

### 3.1 Python 侧

| 文件 | 修改内容 |
|------|----------|
| `runners/inductor.py` → `runners/compile_runner.py` | 重命名文件；`InductorRunner` → `CompileRunner`；增加 `fullgraph`、`dynamic` 参数和 torchair backend 构造 |
| `executor.py` | 更新 import 路径；属性名 `inductor_runner` → `compile_runner`；传递 `fullgraph`、`dynamic` 配置 |
| `pybind/args.py` | 新增 `--python_compile_fullgraph`、`--python_compile_dynamic` CLI 参数 |
| `pybind/llm.py` | 新增 `python_compile_fullgraph`、`python_compile_dynamic` 构造参数 |

### 3.2 C++ 侧

| 文件 | 修改内容 |
|------|----------|
| `core/common/global_flags.h` | 声明 `python_compile_fullgraph`、`python_compile_dynamic` flag |
| `core/framework/config/execution_config.h` | 新增 `python_compile_fullgraph`、`python_compile_dynamic` PROPERTY |
| `core/framework/config/execution_config.cpp` | 注册新属性到 flag/json 映射 |
| `models/llm/py_causal_lm.cpp` | `build_config_dict()` 中传递新配置到 Python |

### 3.3 文档和测试

| 文件 | 修改内容 |
|------|----------|
| `docs/src/content/docs/zh/cli_reference.md` | 新增参数文档 |
| `tests/python/test_model_executor.py` | 更新 `inductor_runner` → `compile_runner` 引用 |

## 4. 使用方式

### 4.1 inductor backend + fullgraph

```shell
--python_graph_backend=inductor \
--python_compile_fullgraph=true
```

### 4.2 torchair backend + fullgraph + static shape

```shell
--python_graph_backend=torchair \
--python_compile_fullgraph=true \
--python_compile_dynamic=false
```

### 4.3 torchair backend + dynamic shape

```shell
--python_graph_backend=torchair \
--python_compile_fullgraph=true \
--python_compile_dynamic=true
```

### 4.4 JSON 配置

```json
{
  "python_graph_backend": "torchair",
  "python_compile_fullgraph": true,
  "python_compile_dynamic": false
}
```

## 5. 后续扩展

当前设计支持通过增加 `_resolve_compile_backend` 分支来接入新 backend。如果未来需要
支持更多 backend 特有的 `CompilerConfig` 选项（如 torchair 的 `experimental_config`、
`session_options` 等），可以在 `CompileRunner.__init__` 中扩展 backend-specific 的
配置构造逻辑，而不需要修改 `ModelExecutor` 的调度路径。

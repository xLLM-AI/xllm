---
title: "xLLM-Kernel 算子开发指南"
---

`xllm-kernel/` 负责 Python 模型使用的算子接口、实现注册与准备，以及独立的数值验证和性能测量。
本文说明新增算子、扩展实现、模型接入与验证的通用流程。RMSNorm 仅作为示例，其他算子沿用相同接口。
安装方法见
[组件 README](https://github.com/xLLM-AI/xllm/blob/main/xllm-kernel/README.md)。

## 1. 职责与目录

组件按算子组织实现，使模型侧公共 API 与具体 solution 解耦。契约显式描述数值行为、mutation、
alias、支持设备与输入域；实现既可以复用厂商算子，也可以提供包内扩展，无需改变模型 API。

注册、依赖检查和绑定在模型执行与 graph capture 前完成。模型保留已准备的 callable，避免在
服务执行路径中选择实现或编译。数值验证与性能测量可以独立于 xLLM server 运行；新增算子族或
平台沿用这些边界，并逐实现补充测试和测量。

目录按职责划分：

| 位置 | 职责与接入点 |
| --- | --- |
| `python/xllm_kernel/ops/<operation>/` | 算子契约、公共 API、实现与注册 |
| `python/xllm_kernel/registry.py` | `OpContract`、`KernelSpec`、`PreparedKernel` 与注册表 |
| `python/xllm_kernel/{catalog,runtime,selection}.py` | 发现算子、加载依赖、选择实现并发布绑定 |
| `python/xllm_kernel/custom_ops.py` | 本包 Torch schema、设备实现与 fake 注册 |
| `csrc/<solution>/` | 原生扩展源文件与设备 launcher bridge 的目录约定 |
| `pyproject.toml`、`setup.py`、`MANIFEST.in` | 包元数据、扩展构建与分发产物 |
| `python/xllm_kernel/numerics/<operation>.py` | 算子专属参数、输入、reference、容差与 suite |
| `python/xllm_kernel/numerics/catalog.py` | 在 `_BUILTINS` 中登记 suite 模块 |
| `python/xllm_kernel/numerics/{suite,factory,inputs,tensor_case}.py` | 用例接口、输入构造、tensor-tree 验证与恢复 |
| `python/xllm_kernel/numerics/{arguments,verify,report,distributed}.py` | 公共 CLI、验证、报告与分布式运行 |
| `python/xllm_kernel/benchmark/` | 复用 suite，提供计时与性能报告 |
| `test/`、`test/npu/` | 公共接口测试与设备实现测试 |

以上路径均相对于 `xllm-kernel/`。模型结构、batch metadata、KV cache 管理和请求调度仍属于
xLLM runtime。算子接收准备好的输入，不在 solution 内另建请求调度或 cache 生命周期。

算子的输入、shape、reference、容差和支持域由对应模块定义；公共注册表、runner 和报告通过接口复用。
公共 API 导出、catalog 模块名单和扩展源文件名单是新增算子的接入点。

## 2. 先定义契约，再实现 kernel

开发前明确以下内容，并让代码、测试和算子文档使用同一套定义：

- 输入/输出角色、shape、dtype、设备、布局、stride、允许的空维度和标量参数范围。
- 是否修改输入、输出是否允许与输入 alias，是否有 residual、out-buffer 或量化语义。
- 数值 reference、累积精度、输出精度、容差规则和非有限值处理。
- 支持 eager、ACLGraph 中的哪些模式，以及 workspace、stream、设备上下文和生命周期要求。
- 依赖的 Torch/torch_npu、设备 SDK、第三方库与构建产物。

[`OpContract`](https://github.com/xLLM-AI/xllm/blob/main/xllm-kernel/python/xllm_kernel/registry.py) 描述算子语义：
`operation`、`version`、`mutates`、`aliases_inputs`、`numerical_mode`。
`KernelSpec` 描述具体实现：名称、设备、执行模式、输入域、solution、variant、binding 和优先级。
其中 `input_domain` 是描述字符串，实际参数由底层实现或厂商算子校验。

给同一个算子新增 solution 时，只有语义一致的实现才能使用相同的 `OpContract`。
改变 mutation、alias、量化或输出语义时，需要明确区分契约，不能靠实现名字掩盖差异。
实现间的数值等价可以按声明的容差判定，不应默认承诺逐位一致。

## 3. 实现、注册与模型接入

### 3.1 实现与注册

实现放在 `ops/<operation>/`，通常按 `<solution>.py` 组织，也可按绑定或变体拆分模块。
在模块顶层用 `@register_kernel(...)` 声明 callable；导入时登记到共享注册表，原样返回 callable，
不增加 forward 包装。

`name` 采用以下命名规则：

```text
<device>.<solution>.<operation>[.<variant>]
```

- `device`：目标设备，与 `device` 字段一致，例如 `npu`。
- `solution`：执行方案，与 `solution` 字段一致，例如 `torch`。
- `operation`：`OpContract.operation` 中的算子名，例如 `rms_norm`。
- `variant`：可选的实现变体后缀，用于区分同一设备、同一方案下同一算子的不同实现。

方括号表示可选部分，不写入实际名称。省略 `variant` 时，也省略前面的点号；例如 RMSNorm 的
名称为 `npu.torch.rms_norm`。名称须在整个进程的注册表中唯一。此格式属于命名约定，注册表不会
解析名称来填充元数据，对应字段仍须单独声明。
下面以 RMSNorm 复用厂商接口为例：

```python
import torch
import torch_npu
from xllm_kernel import register_kernel
from xllm_kernel.ops.rms_norm import RMS_NORM_CONTRACT

_native_rms_norm = torch_npu.npu_rms_norm


@register_kernel(
    name="npu.torch.rms_norm",
    contract=RMS_NORM_CONTRACT,
    device="npu",
    execution_modes=("eager", "aclgraph"),
    input_domain="torch_npu.npu_rms_norm tensor domain",
    solution="torch",
    binding="torch_npu",
    priority=20,
)
def rms_norm(input: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    return _native_rms_norm(input, weight, epsilon=eps)[0]
```

### 3.2 自动选择与准备阶段

在模型导入和 graph capture 前调用 `initialize()`，返回只读的算子绑定：

```python
from xllm_kernel import initialize

bindings = initialize(device="npu", operations=("rms_norm",))
plan = bindings["rms_norm"]
```

```text
发现算子定义 → 按需导入并注册实现 → 冻结注册表 → 筛选实现 → 发布绑定
```

- `operations` 一次声明本进程所需的完整算子集合，省略时准备整个 catalog。实现先按契约、设备和执行模式过滤，再按
  priority 从高到低选择，同优先级按名称排序。
- 可用 `overrides` 或 `XLLM_KERNEL_OVERRIDE_<OPERATION>` 指定完整实现名或 solution；环境变量优先。
- 同一配置复用原绑定，forward/replay 不再选择实现；更改配置须重启 worker。
- 缺失依赖的实现被排除，已安装实现的加载错误直接抛出；所需算子全部准备成功后才发布绑定。

### 3.3 厂商算子与自定义注册

厂商已有的 Torch schema 直接复用；新增本包 Torch 算子时，使用 `register_custom_op()`：

```python
from xllm_kernel.custom_ops import register_custom_op

# Python 实现：显式声明 mutation 和 dispatch key。
register_custom_op(
    "my_op", my_kernel, mutates_args=(),
    dispatch_key="PrivateUse1", fake_impl=my_fake,
)

# 已有 C++ schema：先加载对应扩展，只补充 fake。
register_custom_op("my_native_op", fake_impl=my_native_fake)
```

- 原生扩展放在 `csrc/<solution>/`，同步配置 `setup.py`、源码 manifest 和 wheel 产物。
- 扩展加载与注册在准备阶段完成并复用。
- fake 用于推导 shape、dtype、device 等元信息，不能替代数值验证。

### 3.4 公共 API 与模型接入

公共定义模块保持 SDK-free，设备依赖由实现模块加载。新增算子时，完成以下接入：

1. 在 `ops/<operation>/__init__.py` 声明契约与 `OPERATION`，通过 `BackendModule` 列出实现模块、依赖和实现名。
2. 将算子定义加入 `catalog._BUILTINS`，并从 `ops/__init__.py` 导出公共 API。
3. 公共 API 通过 `OPERATION.get_prepared()` 获取固定绑定；模型初始化时将该算子加入 `operations`。
4. 接入模型或 layer 调用点，并补充对应的离线 suite 和验证。

初始化完成后的公共调用示例：

```python
from xllm_kernel.ops import rms_norm

output = rms_norm(input, weight, eps)
```

模型也可保留 `bindings[operation].function`，在 graph 的整个生命周期中复用。

## 4. 提供算子专属的离线 suite

数值验证与 benchmark 共用一份 `OperationSuite`，输入、reference 和容差由算子定义。

### 4.1 `OperationSuite`

在 `numerics/<operation>.py` 导出 `SUITE`，并加入 `numerics/catalog.py` 的 `_BUILTINS`。以 RMSNorm 为例：

```python
from xllm_kernel.numerics.suite import OperationSuite

SUITE = OperationSuite(
    operation="rms_norm",
    implementations=("npu.torch.rms_norm",),
    add_arguments=_add_arguments,  # 注册 CLI 参数
    cases=_cases,                # 生成用例
    prepare=_prepare,            # 准备实现
)
```

suite 发现阶段只加载元数据；Torch、设备 SDK 和扩展在准备或执行时导入，保证 `--help` 可独立使用。

### 4.2 `KernelCase` 与 `PreparedCase`

| 对象 | 接口与字段 |
| --- | --- |
| `KernelCase` | `case_id`、`parameters`、`unsupported_reason(plan)`、`prepare(plan)` |
| `PreparedCase` | `function`、`validate`、`metadata`、可选的 `reset` |

`prepare(plan)` 创建输入并返回 `PreparedCase`。`validate(function)` 使用独立 reference 和容差，
检查传入函数的输出、mutation 和 alias；ACLGraph 下检查捕获输出。可变输入通过 `reset` 恢复。
不支持的用例由 `unsupported_reason(plan)` 返回原因，支持时返回 `None`。

### 4.3 可复用的 tensor-tree 用例

tensor-tree 输出可复用 `numerics/factory.py` 的 `tensor_suite()`，按正数维度矩阵生成 suite，builder 返回 `Example`。
`TensorCase` 在准备时绑定输入，`numerics/inputs.py` 提供固定 CPU seed 的随机输入；空维度用例自定义 `KernelCase`。
完整示例见 [`numerics/rms_norm.py`](https://github.com/xLLM-AI/xllm/blob/main/xllm-kernel/python/xllm_kernel/numerics/rms_norm.py)。

## 5. 公共验证、计时和报告流程

```text
准备实现 → 检查支持域并生成输入 → 数值验证 → warmup
        → 可选 graph capture/replay 与输出验证 → benchmark 计时 → JSON / Markdown
```

只有验证通过的用例才计时；不适用记录原因，执行或图错误直接抛出，零通过用例不能生成成功报告。

| 模式 | 执行与计时 |
| --- | --- |
| Eager | 逐次调用，报告主机入队时间、设备 event 间隔和同步墙钟时间 |
| ACLGraph | 捕获后 replay，先验证图输出；benchmark 默认每图 64 次调用，有 `reset` 时为 1 次 |

- 准备、warmup 和 capture 不进入稳态计时；数值 CLI 每张图只捕获一次调用。
- `reset` 在设备 event 和主机入队计时区间外执行，可变输入路径的墙钟时间包含其成本。
- 报告记录实现、环境、覆盖率和验证结果，性能报告另含原始计时样本；`solution`、`variant`、`binding` 分别记录。
- `--distributed` 由 torchrun 启动至少两个 NPU rank，逐 rank 输出报告，并汇总验证一致性和计时最大值。

性能对比应保持输入和配置一致，记录设备占用，并用 baseline/candidate/baseline 检查漂移。
Eager 的 event 间隔包含主机提交间隙；单算子结果与端到端性能分别报告。

## 6. 测试与验证

以下命令从仓库根目录执行，先安装包以及被测实现所需的依赖。命令以 RMSNorm 为例，
其他算子替换 `--operation`、`--implementation` 和各自的专属参数。

### 6.1 独立数值验证与 benchmark

```bash
python -m xllm_kernel.numerics --operation rms_norm --help
python -m xllm_kernel.benchmark --operation rms_norm --help

python -m xllm_kernel.numerics --operation rms_norm --device npu --mode eager \
  --implementation npu.torch.rms_norm --rows 1 16 --widths 6144 \
  --revision "$(git rev-parse HEAD)" --output numerics.json
python -m xllm_kernel.benchmark --operation rms_norm --device npu --mode aclgraph \
  --implementation npu.torch.rms_norm --rows 1 16 --widths 6144 \
  --iterations 50 --revision "$(git rev-parse HEAD)" --output benchmark.json
```

各算子的参数与默认值由对应 suite 定义，`--operation NAME --help` 用于查看可用选项。

### 6.2 包测试

包测试通过 pytest 独立运行，不接入宿主 `tests/CMakeLists.txt`、CTest 或 `all_tests`。
公共测试需要安装本包、PyTorch、pytest 和 setuptools：

```bash
python -m pytest -q xllm-kernel/test --ignore=xllm-kernel/test/npu
```

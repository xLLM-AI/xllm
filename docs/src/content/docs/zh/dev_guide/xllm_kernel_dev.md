---
title: "xLLM-Kernel 算子开发指南"
---

`xllm-kernel/` 负责 Python 模型使用的算子接口、实现注册与准备，以及独立的数值验证和性能测量。
本文统一说明组件设计、公共 API、新增算子或实现、模型接入与验证流程。安装方法见
[组件 README](https://github.com/xLLM-AI/xllm/blob/main/xllm-kernel/README.md)。

## 1. 职责与目录

组件按算子组织实现，使模型侧公共 API 与具体 solution 解耦。契约显式描述数值行为、mutation、
alias、支持设备与输入域；实现既可以复用厂商算子，也可以提供包内扩展，无需改变模型 API。

注册、依赖检查和绑定在模型执行与 graph capture 前完成。模型保留已准备的 callable，避免在
服务执行路径中选择实现或编译。数值验证与性能测量可以独立于 xLLM server 运行；新增算子族或
平台沿用这些边界，并逐实现补充测试和测量。

以下目录将算子定义、公共基础设施与 backend 依赖分开：

| 位置 | 负责内容 | 新增算子时的处理 |
| --- | --- | --- |
| `pyproject.toml`、`python/xllm_kernel/__init__.py` | 独立包元数据、初始化与公共重导出；普通 import 不加载 SDK | 导出需要的公共 API |
| `python/xllm_kernel/catalog.py` | `Operation` / `BackendModule` 定义与运行时算子发现 | 将算子模块加入 `_BUILTINS` |
| `python/xllm_kernel/selection.py` | 统一 overrides、兼容性过滤与优先级选择 | 复用选择器，声明实现元数据 |
| `python/xllm_kernel/runtime.py` | 加载所有算子候选、统一冻结并一起发布固定绑定 | 复用统一初始化 |
| `python/xllm_kernel/ops/<operation>/` | 公共算子语义、各 solution 的实现与注册、运行时绑定 | 定义接口或增加实现 |
| `python/xllm_kernel/registry.py` | `OpContract`、`KernelSpec`、`PreparedKernel` 与准备阶段选择 | 复用；不增加按算子名分支 |
| `python/xllm_kernel/custom_ops.py` | Torch 注册与可选 `fake_impl`；fake 函数由算子实现模块持有 | 新 Python op 和已有 C++ schema 复用此入口 |
| `csrc/<solution>/` | 可选本地扩展、Torch bridge 和设备 launcher 适配；当前为 `xlite/` | 增加对应实现与设备注册 |
| `setup.py` | 扩展源文件、依赖库和独立构建选项 | 接入所需扩展，保持默认安装为纯 Python |
| `python/xllm_kernel/numerics/<operation>.py` | 该算子的参数、输入、reference、容差、支持域和 `OperationSuite` | 每个算子提供自己的 suite |
| `python/xllm_kernel/numerics/catalog.py` | 显式发现内置 suite | 增加模块入口；不放算子逻辑 |
| `python/xllm_kernel/numerics/{arguments,suite,verify,report}.py` | 公共 CLI 参数、用例接口、验证生命周期、数值报告 | 通过接口复用 |
| `python/xllm_kernel/benchmark/` | 公共性能 CLI、NPU 计时与报告 | 复用已准备的用例，不解析算子 tensor 参数 |
| `test/`、`test/npu/` | 公共基础设施测试、独立设备测试与 fixture | 增加对应 pytest 用例 |

以上路径均相对于 `xllm-kernel/`。模型结构、batch metadata、KV cache 管理和请求调度仍属于
xLLM runtime。算子接收准备好的输入，不在 solution 内另建请求调度或 cache 生命周期。

公共文件不能写死具体算子的报告标题、输入角色、shape、容差、reference 或某个实现的支持域。
公共 API 的重导出、运行时与 suite catalog 的模块名单、构建脚本中的扩展源文件名单属于显式接入点，
可以列出具体算子；计算语义和条件判断仍应留在对应算子模块。

## 2. 先定义契约，再实现 kernel

开发前明确以下内容，并让代码、测试和算子文档使用同一套定义：

- 输入/输出角色、shape、dtype、设备、布局、stride、允许的空维度和标量参数范围。
- 是否修改输入、输出是否允许与输入 alias，是否有 residual、out-buffer 或量化语义。
- 数值 reference、累积精度、输出精度、容差规则和非有限值处理。
- 支持 eager、ACLGraph 中的哪些模式，以及 workspace、stream、设备上下文和生命周期要求。
- 依赖的 Torch/torch_npu、设备 SDK、第三方库与构建产物。

[`OpContract`](https://github.com/xLLM-AI/xllm/blob/main/xllm-kernel/python/xllm_kernel/registry.py) 当前表达 operation、version、mutates、
aliases_inputs 和 numerical_mode。`KernelSpec` 另外声明实现名、device、execution_modes、
input_domain、solution、variant、binding 和 priority。`input_domain` 当前是描述字符串，注册表不会据此自动验证
shape 或布局；真实实现仍须在 launch 前检查参数。

给同一个算子新增 solution 时，只有语义一致的实现才能使用相同的 `OpContract`。
改变 mutation、alias、量化或输出语义时，需要明确区分契约，不能靠实现名字掩盖差异。
实现间的数值等价可以按声明的容差判定，不应默认承诺逐位一致。

## 3. 实现、注册与模型接入

### 3.1 实现与注册

在 `ops/<operation>/<solution>.py` 模块顶层用 `@register_kernel(...)` 声明 callable。
导入模块时解析依赖并执行装饰器：构造 `KernelSpec`，登记到 `KernelRegistry.get()`，再原样返回
callable，不增加 forward 包装。backend 无需提供 `register(registry)` 入口，也无需传入 registry。

```python
import torch
import torch_npu
from xllm_kernel import register_kernel
from xllm_kernel.ops.rms_norm import RMS_NORM_CONTRACT

_native_rms_norm = torch_npu.npu_rms_norm


@register_kernel(
    name="npu.aclnn.rms_norm.torch_npu",
    contract=RMS_NORM_CONTRACT,
    device="npu",
    execution_modes=("eager", "aclgraph"),
    input_domain="torch_npu.npu_rms_norm tensor domain",
    solution="aclnn",
    variant="torch_npu",
    binding="torch_npu",
    priority=20,
)
def rms_norm(input: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    return _native_rms_norm(input, weight, epsilon=eps)[0]
```

已有 Torch op 或 callable 实例可以使用 `register_kernel(...)(existing_callable)`；xlite 在检查和
加载 bridge、注册 fake 后使用此形式。实现名、contract、device、execution_modes、input_domain
和 solution 必须显式提供，priority 默认为 0。重名注册和 `freeze()` 后注册直接报错。

包及装饰器的导入不依赖 Torch 或设备 SDK。backend 模块被导入时加载自身依赖。每个算子导出
包含契约与 `BackendModule` 声明的 `OPERATION`；统一 catalog 先发现这些不依赖 SDK 的定义，
再导入可用 backend 候选。

`register_kernel` 负责实现选择元数据；`custom_ops.register_custom_op` 另行负责 Torch schema、
设备 kernel 和可选 fake 的注册。对照现有
[native adapter](https://github.com/xLLM-AI/xllm/blob/main/xllm-kernel/python/xllm_kernel/ops/rms_norm/aclnn_torch_npu.py) 和
[xlite adapter](https://github.com/xLLM-AI/xllm/blob/main/xllm-kernel/python/xllm_kernel/ops/rms_norm/xlite.py) 阅读。

复用宿主已有 Torch op 时，保留原有 schema、实现和 fake 的所有权。
新增本包拥有的 Torch op 时，使用本包 namespace，同时提供真实设备 bridge 和相应 fake。
构建产物及版本检查应在准备阶段完成，不能在请求执行或 graph replay 时下载依赖、编译或重选实现。

NPU 实现使用当前 PyTorch stream，并遵循 torch_npu 的提交队列和 Tensor 生命周期。
禁止在算子内部为方便实现而创建私有 stream 或执行设备同步；跨 stream、设备切换和异步资源释放
必须有对应测试。FakeTensor 负责可检查的逻辑 shape/dtype/device 语义，真实设备实现仍负责
物理格式等设备条件。

### 3.2 自动选择与准备阶段

```text
initialize(device, execution_mode, operations, overrides)
    -> 从 catalog._BUILTINS 发现所有 OPERATION 定义
    -> 解析 XLLM_KERNEL_OVERRIDE_<OPERATION> 和显式 overrides
    -> 导入可用 backend 模块，由装饰器登记 callable
    -> 统一冻结共享 registry
    -> 按契约/设备/执行模式过滤，再按优先级和稳定名称排序
    -> 所有算子准备成功后，一次发布只读绑定映射
    -> 模型保留 callable，用于 eager 或 graph 执行
```

正常启动无需设置 override。选择基于声明元数据与优先级，不读取输入 Tensor、不运行候选性能
测试、不做 shape 特化，也不在 forward/replay 时重选。实际参数仍由选中的实现校验。RMSNorm
的当前实现和支持范围见[第 3.4 节](#34-公共-api-与当前支持范围)。

宿主在加载 native 算子后、导入模型或 graph capture 前调用
`xllm_kernel.initialize(device="npu", operations=("rms_norm",))`，并记录各算子的选中实现。直接调用返回
`Mapping[str, PreparedKernel]`：

```python
from xllm_kernel import initialize

bindings = initialize(device="npu", operations=("rms_norm",), overrides={"rms_norm": "xlite"})
plan = bindings["rms_norm"]
```

启动 worker 时可用 `XLLM_KERNEL_OVERRIDE_RMS_NORM=xlite` 指定同样的选择。统一规则为
`XLLM_KERNEL_OVERRIDE_<OPERATION>`，算子名转为大写；值接受 solution 或完整实现名，精确
实现名优先于同名 solution。环境变量优先于显式映射。未知算子、未知目标和不兼容的显式选择
在发布绑定前报错；override 不会绕过契约、设备与执行模式校验。

缺失声明依赖时，该 backend 不导入，原因记录在 `PreparedKernel.excluded`。已安装 backend
导入失败、ABI 或 launcher hash 不匹配等错误直接向上传播，不会触发切换后端。所有算子必须
准备成功后才发布绑定。同一有效配置重复初始化会返回相同映射；改变配置需要重建 worker 和
graph。数值验证和 benchmark 复用同一初始化入口，环境覆盖若改变了 CLI 指定的实现会报错。

新增算子时，在公共模块导出 `OPERATION = Operation(contract, backends)`，并把模块加入
`catalog._BUILTINS`。每个 `BackendModule` 声明模块路径、设备和所需依赖；增加 solution 时
扩展该算子的 backend 列表，并用 `@register_kernel` 声明元数据与优先级。宿主无需新增逐算子
环境变量处理逻辑。离线 suite 和模型调用点仍分别扩展。

catalog 现包含 1 个算子、3 个实现变体；完整清单见第 8 节。模型目前只接入 RMSNorm，
新增注册不自动替换宿主调用点。新增算子或改变语义后，仍需验证模型接入路径。


`operations` 显式声明本进程需要的算子集合；省略时准备整个 catalog。模型宿主当前传入
`("rms_norm",)`，避免加载未使用的通信算子。独立 CLI 仅准备指定算子；多算子调用方应在一次
`initialize` 中给出完整集合。集合、设备、模式或有效 override 改变都要求重启 worker。

`BackendModule.implementations` 提供不依赖 SDK 的 CLI 名单；装饰器仍拥有完整选择元数据。
同一 solution 的绑定分为独立模块，如 `aclnn_torch_npu.py` 与 `aclnn_xllm.py`。
宿主绑定声明 `xllm_export` 依赖：模型在初始化前加载它；独立 CLI 只在显式选择宿主绑定时加载。
未加载宿主时，独立 torch_npu/xlite 路径仍可准备。已安装 backend 的 ABI/import 错误继续向上传播。

### 3.3 厂商算子、自定义注册与扩展构建

当前 RMSNorm 的实现入口如下：

| 实现名 | 执行入口 | Torch schema 所有者 | 包内构建选项 |
| --- | --- | --- | --- |
| `npu.aclnn.rms_norm.torch_npu` | `torch_npu.npu_rms_norm(input, weight, epsilon=eps)[0]` | torch_npu | 无 |
| `npu.xlite.rms_norm` | `csrc/xlite/rms_norm.cpp` | `xllm_kernel::xlite_rms_norm` | `XLLM_KERNEL_BUILD_XLITE=1` |

Native 直接调用 torch_npu，只返回归一化结果，与原 C++ wrapper 取 tuple 第一个元素的语义一致。
它不需要 xLLM host 或包内 native 扩展；实现名为 `npu.aclnn.rms_norm.torch_npu`。Torch schema 和
FakeTensor 实现由厂商包持有。

`custom_ops.register_custom_op` 是准备阶段共用的 Torch 注册入口。新增 Python kernel 时，
传入 callable、显式的 `mutates_args` 和 `dispatch_key`，以及可选 `fake_impl`。如果 C++
扩展已经注册 schema 和设备 kernel，则不传 callable、mutation 元数据与 dispatch key；
helper 解析已有算子，并注册可选 fake。helper 保持 Torch library 的生命周期，注册失败直接
向上传播；重复初始化由 backend 复用准备状态，不重复定义 schema 或 fake。fake 函数放在
对应算子实现模块中，不再设置独立 `fake/` 目录。

```python
from xllm_kernel.custom_ops import register_custom_op

# 新 Python kernel：my_kernel、my_fake 由算子模块提供。
register_custom_op(
    "my_op", my_kernel, mutates_args=(),
    dispatch_key="PrivateUse1", fake_impl=my_fake,
)
# 已有 C++ schema：先加载 xlite bridge，再添加 fake。
register_custom_op("xlite_rms_norm", fake_impl=_fake_rms_norm)
```

Native 不需要包内本地扩展。开发 xlite bridge 时，使用匹配的 Torch/torch_npu、CANN 环境，
以及包内 `setup.py` 要求的 GVirt revision
`4a6dc3102d5778928e4d21116bc4d4694d51e584`。构建清单记录依赖版本和 launcher hash。
安装方法见组件 README；修改 C++ bridge 后，应先原地重建扩展，再启动新 worker：

```bash
(cd xllm-kernel && XLLM_KERNEL_BUILD_XLITE=1 MAX_JOBS=4 python setup.py build_ext --inplace)
```

扩展通过 `torch.ops.load_library` 在准备阶段加载，不在普通包导入时加载。已有宿主
`xllm_ops` schema 和 C++ 模型调用链保留其所有权与行为。FakeTensor 支持用于追踪、编译时
推导元信息；它不是 eager/设备图执行本身的必要条件，也不承担数值 reference。应分别验证这些路径。

### 3.4 公共 API 与当前支持范围

公共算子从 `xllm_kernel.ops` 导出。完成包初始化后，模型可以调用公共函数，也可以在 graph 的
整个生命周期中保留已准备的 callable：

```python
from xllm_kernel import prepare_rms_norm
from xllm_kernel.ops import rms_norm

plan = prepare_rms_norm()
output = rms_norm(input, weight, eps)
# 模型也可保留 plan.function，用于重复调用或 graph capture。
```

| 算子 | 实现名 | 执行模式 | 优先级与选择 |
| --- | --- | --- | --- |
| `rms_norm` | `npu.aclnn.rms_norm.torch_npu` | Eager、ACLGraph | 优先级 20；同算子的默认选择 |
| `rms_norm` | `npu.aclnn.rms_norm.xllm` | Eager、ACLGraph | 优先级 10；需要已加载的宿主扩展 |
| `rms_norm` | `npu.xlite.rms_norm` | Eager、ACLGraph | 优先级 0；可通过统一 override 指定 |

Native 保留 torch_npu 的输入域，以较高优先级将这一更广的支持范围作为默认；这不表示 native
在所有规模下都更快。xlite 支持同一 NPU、同一 dtype 的 FP16/BF16 输入与权重、ND 布局、
64 的倍数且不超过 8192 的宽度、至多 `2**32 - 1` 个输入元素，以及非负有限 FP32 epsilon。
支持非连续 ND 输入和空行。两种实现都不修改输入，也不让输出与输入 alias；数值等价按容差判断。

当前宿主接入覆盖共享层的普通 RMSNorm 和 GLM-5.3 MLA 预处理。Residual、量化和 Gemma
normalization 保留现有路径。其他算子的模型接入、CUDA adapter、shape 特化、profiling/replay、编译监控
和调优仍属于后续扩展；增加注册声明或测试替身并不代表设备或模型已经支持。

## 4. 提供算子专属的离线 suite

公共接口定义在 [`numerics/suite.py`](https://github.com/xLLM-AI/xllm/blob/main/xllm-kernel/python/xllm_kernel/numerics/suite.py)。
数值 CLI 和性能 CLI 使用同一份 suite，避免各自维护不同的 shape、输入或 reference。
`numerics/` 负责输入生成、独立 reference、容差、契约验证与数值报告；`benchmark/` 复用已准备
用例并增加计时。包内 `test/` 负责 pytest 用例与 fixture，不进入运行时 wheel，也不增加中间的
`testing/` 包。

### 4.1 `OperationSuite`

每个 `numerics/<operation>.py` 导出一个 `SUITE`：

| 字段 | 约定 |
| --- | --- |
| `operation` | 与已准备实现的 `OpContract.operation` 一致 |
| `implementations` | 该 suite 支持的完整实现名；catalog 中不可重复 |
| `add_arguments(parser)` | 注册该算子的专属 CLI 参数 |
| `cases(args)` | 校验专属参数并返回 `KernelCase` 用例；此时不分配设备输入 |
| `prepare(device, mode, implementation)` | 加载所选实现并返回 `PreparedKernel` |

在 `numerics/catalog.py` 的 `_BUILTINS` 中增加模块入口。catalog 导入 suite 时只能使用
Python 元数据；Torch、设备 SDK 和 native extension 的导入留到真正准备/执行时，保证
`--help` 在无设备环境可用。

公共 CLI 处理 device、执行模式、显式实现名、warmup、revision、output；benchmark 另外处理
iterations 和 repeats。`--operation` 可显式指定，也可由 `--implementation` 推导。
shape、dtype 组合等算子参数由 `add_arguments` 和 `cases` 负责，不能塞进公共 CLI 的条件分支。

### 4.2 `KernelCase` 与 `PreparedCase`

每个逻辑用例实现以下接口：

| 接口 | 返回内容与职责 |
| --- | --- |
| `case_id` | 可复现、能区分用例参数和 seed 的标识 |
| `parameters` | 可写入 JSON 的参数字典 |
| `unsupported_reason(plan)` | 不适用时返回原因，适用时返回 `None`；不得通过执行失败来猜支持域 |
| `prepare(plan)` | 生成输入，返回 `PreparedCase` |

`PreparedCase` 的字段是：

- `function()`：绑定输入、标量参数和已准备实现的零参数 callable，返回值可以是 tensor、tuple
  或其他由该算子定义的结构。
- `validate(function)`：调用传入的零参数 callable，按该算子的 reference 和容差返回包含
  `status` 的验证结果；校验输出结构、数值、mutation 和 alias 等契约。
- `metadata`：输入 stride、格式等可序列化的专属记录。
- `reset()`：可选的输入恢复函数；在验证、warmup 和每次可变输入计时前调用。

验证器必须检查传入 callable 的结果，不能无视它而重新调用原始实现。ACLGraph 验证传入的是
捕获输出的读取函数；重新执行 eager 算子会掩盖错误的 graph 输出。验证结果和 metadata
也不应覆盖公共 case ID、operation、implementation 等身份字段。

默认输入用固定 seed 在 CPU 生成，再按实现需要搬到设备，保证候选实现接收相同逻辑数据。
reference 应独立于被测实现，不作为服务实现或执行错误的回退。支持域过滤必须和真实实现的
参数检查一致；边界仍要由直接设备
测试覆盖，不能通过全部标为 `not_applicable` 规避验证。

现有 [`numerics/rms_norm.py`](https://github.com/xLLM-AI/xllm/blob/main/xllm-kernel/python/xllm_kernel/numerics/rms_norm.py) 提供完整实例。
公共 CLI 测试另外使用参数为 size/scale、返回字典的测试算子，验证 runner 不依赖 RMSNorm 的
input/weight/eps 结构。这个测试算子只是公共流程的单测，不代表新增设备算子支持。

## 5. 公共验证、计时和报告流程

```text
解析公共参数 + suite 专属参数
    -> suite.prepare
    -> 用例支持域检查
    -> case.prepare
    -> validate(function)
    -> warmup
    -> 可选 graph capture/replay + validate(captured_output)
    -> benchmark 计时
    -> JSON / Markdown 报告
```

`numerics/verify.py` 只管理准备、状态和报告生命周期；不解释 tensor 角色，也不选择某个算子的
reference。`numerics/report.py` 和 `benchmark/report.py` 使用通用标题，逐行记录 operation，
不通过名字猜测算子类型或读取某个算子的 shape 字段。JSON 和 Markdown 报告记录实现、
环境、覆盖率、验证结果和失败原因。

失败状态需保持可见：不适用记录原因；数值失败不计时；capture/replay 失败直接报告并抛出；
没有任何通过的用例不能产生成功报告。对设备执行错误不得切换到 eager 或其他 solution 后
继续给出有效性能结果。

当前计时器支持 NPU eager 和 ACLGraph，报告主机入队时间、NPU event 间隔和同步墙钟时间。
无输入修改时，每张图捕获 64 次独立调用；有 `reset` 时只捕获一次，并在 capture/replay
和每次计时调用前恢复输入。reset 在设备 event 与主机提交计时区间外；可变输入路径的墙钟时间
包含 reset 成本，报告中显式记录该边界。

准备、warmup 和 capture 不进入稳态计时。保持输入、case、设备、执行模式和计时设置一致，
报告原始样本与离散程度，并记录资源占用。eager 的设备事件间隔包含主机提交间隙，不能简单
等同于单个 kernel 的纯设备执行时间。当前复用输入、采用热缓存策略；计时过程中持续监控其他
任务，并用重复的 baseline/candidate/baseline 测量检查漂移。性能结论还应区分单算子和端到端结果。

## 6. 测试与验证

以下命令从仓库根目录执行，先安装包以及被测实现所需的依赖。

### 6.1 独立数值验证与 benchmark

Native suite 直接调用 torch_npu，xlite suite 加载自己的可选 bridge；两者都不导入 xLLM host，
也不需要启动 server。两套 CLI 的 `--help` 均可在不加载 Torch 或设备 SDK 时使用；
`--operation NAME --help` 查看指定 suite 的参数。例如：

```bash
python -m xllm_kernel.numerics --operation rms_norm --device npu --mode eager \
  --implementation npu.aclnn.rms_norm.torch_npu --rows 1 16 --widths 6144 \
  --revision "$(git rev-parse HEAD)" --output numerics.json
python -m xllm_kernel.benchmark --operation rms_norm --device npu --mode aclgraph \
  --implementation npu.aclnn.rms_norm.torch_npu --rows 1 16 --widths 6144 \
  --iterations 50 --revision "$(git rev-parse HEAD)" --output benchmark.json
```

将实现名改为 `npu.xlite.rms_norm` 可验证 xlite bridge。省略 shape 和 dtype 选项时，RMSNorm
suite 使用默认矩阵：BF16/FP16、行数 1/16/128/4096、宽度 512/2048/6144。输入在 CPU 上确定性
生成，并与 FP32 CPU reference 对比。新增算子通过相同接口提供自己的参数与 reference。

只有数值验证通过的用例才计时，ACLGraph 还要求捕获输出验证通过。零个通过用例或验证失败都
不能产生成功报告；实现间性能比较应遵循第 5 节的计时与资源控制要求。

### 6.2 包测试与宿主集成测试

包测试通过 pytest 独立运行，不接入宿主 `tests/CMakeLists.txt`、CTest 或 `all_tests`。
公共测试需要安装本包、PyTorch、pytest 和 setuptools，不需要 NPU 或宿主扩展：

```bash
python -m pytest -q xllm-kernel/test --ignore=xllm-kernel/test/npu --ignore=xllm-kernel/test/distributed
```

NPU 集成测试需要匹配的 host 与 backend 构建产物。两种实现应在空闲设备上顺序、分进程运行：

```bash
XLLM_KERNEL_OVERRIDE_RMS_NORM=npu.aclnn.rms_norm.torch_npu \
  python -m pytest -q xllm-kernel/test/npu/test_kernel_rms_norm_npu.py
XLLM_KERNEL_OVERRIDE_RMS_NORM=npu.xlite.rms_norm \
  python -m pytest -q xllm-kernel/test/npu/test_kernel_xlite_rms_norm_npu.py
```

设备 fixture 在 `test/npu/conftest.py` 中加载宿主并初始化 Python kernels；pytest 不负责构建。
Native 测试比较 torch_npu adapter 与宿主结果，并通过子进程检查不依赖宿主或包内 native 扩展
的执行与 FakeTensor 行为。同一进程保留原有绑定，因此 native/xlite 使用不同进程。直接 pytest
不获得 CTest 设备资源锁，需要自行确保设备空闲；device guard 测试需要两张可见 NPU。
若验证范围不包含宿主构建产物，可使用前面的独立 CLI。

### 6.3 验证顺序与证据范围

新增算子的测试按以下顺序展开，范围以实际契约为准：

1. SDK-free import、注册/准备、独立安装和 CLI 参数测试。
2. CPU reference、容差、错误输出、非有限值、mutation/alias 检测。
3. 真设备 dtype/shape/layout/stride/空输入/边界与非法参数测试。
4. 非默认 stream、设备 guard、异步对象生命周期、重复调用和内存稳定性。
5. eager/ACLGraph 输出一致性，以及改变输入后的交错 replay。
6. 独占设备上的单算子性能比较，最后按模型接入范围做集成验证。

CPU double 和 FakeTensor 只证明对应逻辑，不能作为 NPU 精度或性能证据。

## 7. 交付检查项

新增算子时，先定义契约和公共 API，再实现并注册 backend、接入 `OPERATION` 与运行时 catalog，
随后提供离线 suite 和测试，最后接入模型调用点。同一契约下增加 solution 时，复用公共 API 与
算子定义，扩展 backend 列表、suite 的实现名单、支持用例和设备测试；统一初始化负责选择，
无需在宿主新增逐算子分支。

- 契约、真实参数检查、fake 检查、suite 支持域和算子文档一致。
- 公共 runner、report、registry 中没有为新算子增加名字、shape、输入角色或容差分支。
- suite 导入不加载设备依赖，未知实现和不匹配的 operation 在准备执行前失败。
- 新 solution 通过算子 backend 声明、离线 suite 和设备测试接入，保持当前流与资源生命周期规则。
- 单测仍位于独立包 `test/`，不修改宿主的测试目标来自动运行它们。
- 检查格式、安装包内容、CLI 示例和报告；提交说明区分源码检查、CPU、构建、NPU 和模型证据。
- 如需 JIT、状态 reset、多设备或新平台服务，先补齐显式接口及生命周期，再增加实现。

## 8. GLM-5.3 算子清单与显式后端接入

本轮扩展对照 `xllm/python/models/glm5_2.py`、共享的 `deepseek_v32.py` 路径、
`layers/embedding.py`、`layers/linear.py`，以及
`kernels_npu/{normalization,activation,quantization,rotary_embedding,moe,mla,sparse_attention}.py`。
这里指完整的 GLM-5.3 DSA 模型，不是 GLM-5.3 Flash。代码基于 main 的统一 catalog、runtime、selection 和装饰器注册架构。

| 算子 | 方案及可选变体 |
| --- | --- |
| `rms_norm` | `aclnn.xllm`, `aclnn.torch_npu`, `xlite` |

清单共 1 个算子契约、3 个实现变体，归属表中列出的方案。`aclnn` 包括标准 CANN 算子和通过 ACLNN
API 调用的自定义算子。xLLM 绑定内部可能分派到不同 ACLNN kernel，变体名标识封装入口，
不承诺固定二进制实现。`atb` 对应实际 ATB 调用，`hccl` 对应通信库，`torch` 是显式 PyTorch NPU
组合实现，`tilelang` 解析已有 AOT 算子。xlite 使用固定 GVirt 版本的 launcher，提交到当前 stream。
初始化导入所需算子的可用候选模块，然后统一冻结和选择。

新增精度 suite 通过 `tensor_suite` 组织算子自己的形状矩阵。`Example` 绑定输入 tensor tree、
CPU 参考、声明的输入修改、容差及可选的语义归一化比较方法。`TensorCase` 只携带元信息，
发现 suite 时不加载设备 SDK。`PreparedCase.reset` 在计时 event 区间外恢复可变输入；
有输入修改的算子每张图只捕获一次调用。各算子的测试文件复用
`test/npu/operator_checks.py` 的执行流程，不接入 `tests/CMakeLists.txt`。

注册清单和 CPU 测试不代表新增实现已通过设备或完整模型验证。真实 NPU 精度、性能和
端到端结果必须记录实际 revision、输入、设备占用与退出状态，不能复用重构前的结论。

路由及 SwiGLU 语义同时参考官方
[MoeInitRoutingV2 API](https://www.hiascend.com/document/detail/en/Pytorch/2610/apiref/customapi/docs/en/custom_APIs/torch_npu/torch_npu-npu_moe_init_routing_v2.md) 和
[DequantSwigluQuant API](https://www.hiascend.com/document/detail/en/Pytorch/2610/apiref/customapi/docs/en/custom_APIs/torch_npu/torch_npu-npu_dequant_swiglu_quant.md)。
安装的 runtime 必须提供所选接口；缺失时准备或执行明确报错。文档不能替代二进制 ABI 验证。

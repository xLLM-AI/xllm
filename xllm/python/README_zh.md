<!--
Copyright 2026 The xLLM Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/xLLM-AI/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# xLLM Python 包

[English](README.md) | [简体中文](README_zh.md)

Python 执行器将模型语义与硬件执行分离，保持单向依赖：

```text
models  ->  layers  ->  kernels
        model_executor 协调执行
        distributed 负责并行通信
```

## 包职责

### `models/`

负责模型结构、配置解析、权重加载和前向计算的组合。模型使用 layer 和逻辑算子描述网络。

### `layers/`

提供可复用的神经网络层，并管理参数。简单 layer 直接调用当前平台的 `kernels`。
当不同设备需要不同的算子边界、参数布局、持久工作区或计算图生命周期时，复杂模型可以
分别实现 `layers/<device>/<model>/`。

Qwen3.5 是这种组织方式的示例：共用模型入口，在构造时一次性选择 CUDA 或 NPU 的
decoder layer 实现。后端选择不能进入前向计算的热路径，各平台的 layer 包也不能互相导入。

### `kernels_<device>/`

每个平台拥有独立的硬件 kernel 包，与 C++ 的 `xllm/core/kernels/` 划分对应：

```text
kernels_cuda/
kernels_npu/
```

各平台包的实现彼此独立，不互相导入。runtime 初始化会把当前平台包发布为
`xllm.python.kernels`，不同入口的初始化流程见[导入与初始化](#导入与初始化)。

通用 layer 和平台专属 layer 都可以通过 `from xllm.python import kernels` 使用初始化时
选定的平台包。平台专属 layer 在选定设备后导入，其 API 可以遵循该平台的原生算子融合边界。
`setup.py` 只打包与 `--device` 对应的 kernel 包。xLLM 的构建平台多于 Python 执行器
支持的平台；没有对应 kernel 包时，其余 `xllm.python` 内容仍会打包，但为该平台初始化
Python runtime 时会报错。

`model_executor/executor.py` 独立选择 attention 后端和计算图 runner。attention 后端
持有跨执行步骤的状态，例如 wrapper、工作区和缓存计划，生命周期由 executor 管理；
kernel 包则导出无状态函数，因此 attention 后端的选择由 executor 负责。

同级模块使用相对导入，跨包使用以 `xllm` 为根的绝对导入。

平台包负责其硬件相关内容：

- Triton、FlashInfer 和厂商库的调用入口，按框架放入 `triton/`、`flashinfer/` 等子目录；
- Python 实现算子的 `torch.library.custom_op` 注册及 FakeTensor 契约；
- C++ 算子的 FakeTensor 契约，集中在 `_custom_op.py`；
- 张量修改声明与 `torch.compile` 计算图边界；
- kernel 所需的权重布局。

TileLang、Triton、FlashInfer 等较重的调用实现保持在语义算子函数内部延迟导入，
构建工具显式导入底层 DSL 模块的情况除外。

NPU 包包含两个独立的底层实现库：

```text
kernels_npu/
├── tilelang/
└── triton/
```

它们只依赖各自的 DSL、框架和内部辅助模块，不导入 NPU 语义 API、`_custom_op.py`、
`torch.ops.xllm_ops`、AOT 编译器或 xLLM 原生扩展。因此 C++ 构建工具可以在
xLLM 二进制尚未生成时导入同一份 Python DSL 实现。

### Python DSL 归属与 AOT 复用

`kernels_npu/tilelang/` 和 `kernels_npu/triton/` 拥有 Python DSL 源码。
底层模块包含程序构造器、JIT 调用入口、实现内部的校验，以及参考实现和调试辅助函数。
运行时语义模块可以调用这些底层模块，底层模块不能反向依赖语义包。

Ascend TileLang AOT 适配器位于 `xllm/compiler/tilelang/targets/ascend/aot/`，
只负责构建元数据和编译转换，例如 kernel family 注册、分发 schema、特化、导出 ABI
及源码生成。AOT 适配器从 `kernels_npu/tilelang/` 导入程序构造器，不应拥有或复制
DSL kernel 主体。

允许的依赖方向是：

```text
kernels_npu 语义 API  ->  kernels_npu/{tilelang,triton}
Ascend TileLang 编译器  ->  kernels_npu/tilelang
kernels_npu/{tilelang,triton}  ->  仅依赖 DSL/框架
```

构建工具直接导入具体的底层 DSL 模块，不导入 `kernels_npu` 语义 API，也不调用
`initialize_runtime()`。构建期和运行期行为不通过环境变量切换。

### 各平台的 kernel API

每个平台包独立定义公开 API，并在自己的 `__all__` 中声明。各平台不必导出相同名称：
模型和 layer 复用稳定的 `xllm.python.kernels` 绑定，由当前平台提供其支持模型所需的算子。

已有的未实现函数如果通过明确的 `NotImplementedError` 提供诊断，可以保留；新增算子
不需要在其他平台添加对应的占位实现。`model_platform_support.py` 记录模型与平台的
支持关系，registry 在导入模型实现前拒绝不支持的组合。

语义相似的算子在不同平台上可以使用不同的公开函数、Torch schema、融合边界、参数和
模型层组合。各平台的测试定义自己的 kernel 契约，不应仅为让共享复杂 layer 通过编译，
就添加模仿其他平台接口的适配层。

### `distributed/`

负责并行进程组、拓扑和集合通信。共享调度逻辑选择 `cuda/` 和 `npu/` 中的平台专属通信实现。

### `model_executor/`

负责执行协调，包括 eager 或计算图 runner、前向上下文、attention 后端配置、cache 绑定和生命周期。

## 导入与初始化

`import xllm` 和 `import xllm.python` 不会加载原生扩展，也不会初始化 kernels。
原生算子注册、Python runtime 初始化、模型构造是不同的阶段：

| 入口 | 原生代码何时可用 | Python kernels 何时初始化 |
| --- | --- | --- |
| `setup.py` / AOT 编译 | 编译期间不加载扩展 | 不初始化 |
| xLLM server 使用 embedded Python | 宿主先提供原生算子，再由 `ensure_python_interpreter()` 初始化 Python | `ensure_python_interpreter()` 调用 `initialize_runtime()` |
| 通过 pytest 运行 `tests/python` | `conftest.py` 的公共初始化在收集测试前加载 `xllm_export` | `conftest.py` 随后调用 `initialize_runtime()` |
| Python 离线推理 API（`xllm.LLM`） | `_load_public_api()` 在准备 API 时加载 `xllm_export`，早于引擎构造 | 选择 Python 模型执行器时，由 worker 调用 `initialize_runtime()` |

加载扩展让 Python 进程能够使用 xLLM 的 C++ 引擎和算子，由离线 API 和测试公共准备负责。
server 启动时已经具备原生代码，随后才嵌入 Python；构建流程则导入 DSL 源码以生成
原生代码。这两种情况下的包导入都不需要加载独立扩展。

扩展加载后，原生算子可用。`initialize_runtime()` 随后初始化平台 Python kernels，
它本身不加载原生库。各入口负责安排下述执行顺序。

### 公共 runtime 初始化

`initialize_runtime()` 要求原生算子已经注册，不负责加载 `xllm_export`。
它选择平台包，调用该包的 `_initialize_runtime()` 钩子，再同时设置
`xllm.python.kernels` 属性和 `sys.modules` 中的对应条目。磁盘上没有 `kernels/`
目录，这些导入方式指向同一个已选定的平台包。

NPU 初始化先导入 `_custom_op.py`，注册原生算子的 FakeTensor 实现，再按照 `_EXPORTS`
导入语义模块并发布函数。CUDA 在导入平台包时加载 Python 算子封装，但同样把原生算子的
FakeTensor 注册延迟到 runtime 钩子。FakeTensor 实现用于描述计算图追踪时的算子行为。

每个解释器在成功初始化后复用现有状态。模型创建、权重加载和进程组配置分别进行；
各 worker 或 pytest 进程拥有各自的初始化状态。

### 构建阶段：`setup.py` 与 AOT 编译

NPU 的 `setup.py build` 在 C++ 构建前执行 TileLang AOT 编译。AOT 适配器导入
`kernels_npu.tilelang.rope` 等 DSL 模块，Python 会先执行父包，但这条路径不会初始化
runtime。随后构建引擎、暂存 Python 包及平台资源，再构建 `export_module`。
`bdist_wheel` 将暂存产物打包。

NPU 包必须在这里延迟导入语义模块：`activation` 等模块会在导入时绑定
`torch.ops.xllm_ops` 函数。如果立即导入它们，就会提前要求构建尚未产出的原生算子。
当前通过 `_EXPORTS` 实现这种延迟绑定。AOT 编译不运行 pytest，也不会加载其 `conftest.py`。

### xLLM server 使用 embedded Python

xLLM server 选择 Python 模型实现后，`PyCausalLM` 调用 `ensure_python_interpreter()`。
该函数确保原生算子注册代码被保留在链接产物中，按需创建解释器，设置包搜索路径，导入
`xllm.python`，并在构造模型前调用 `initialize_runtime()`。如果解释器已存在，则复用它。
NPU 新建解释器时，还会在 runtime 初始化前执行已有的 `_npu_bootstrap` 适配逻辑。

worker 通过自身链接的代码提供原生算子，因此这条路径不需要独立的 `xllm_export` 扩展。
`--python_model_path` 指定包含 `xllm` 包的目录；参数为空时读取
`XLLM_PYTHON_MODEL_PATH`。两者均未设置时，使用正常的 Python 包搜索路径。

### Python 测试

构建完成后，从 checkout 根目录直接运行测试：

```bash
python -m pytest tests/python/test_model_executor.py
```

pytest 在收集测试模块前自动加载 `tests/python/conftest.py`。它先加载 `xllm_export`，
再调用 `initialize_runtime()`，因此测试可以正常导入模型、layer 和 kernel 模块。
测试文件无需重复执行这些步骤。直接运行 pytest 不会自动构建扩展。

通过 `python setup.py test --test-name python_tests` 运行 NPU Python 测试集合。
各文件对应的构建目标都依赖 `xllm_export`；CTest 为每个文件启动独立的 pytest 进程，
使用同一份 `conftest.py`。

### Python 离线推理 API

此模式通过 `xllm.LLM` 等公开 Python API 调用 C++ 引擎：

```text
from xllm import LLM
  -> 加载 xllm_export -> 导入 Python API 包装层
LLM(...)
  -> 构造 Options -> 调用 C++ LLMMaster / VLMMaster -> 创建模型 worker
```

在 API 初始化阶段，`_load_public_api()` 先加载扩展，再导入 Python API 包装层。
这发生在引擎或模型构造之前，并不初始化 Python 模型 runtime。C++ 引擎可以使用原生模型，
也可以使用 Python 模型。选择 Python 模型后，会进入前述 embedded 初始化链路；
API 调用者无需自行调用 `initialize_runtime()`。
worker 进程可能创建自己的解释器，而非复用 API 调用者的解释器。

### 源码 checkout 与已安装的 wheel

加载 `xllm_export` 时，源码 checkout 使用自身标准 `build/` 目录中与当前 Python
解释器和平台匹配的产物。缺少产物会报错，无需向源码包复制或链接扩展。
已安装的 wheel 从自己的包目录加载扩展。要使用 wheel，应从源码树外运行，并确保
`sys.path` 中源码根目录没有排在安装位置之前。

扩展加载后，会在当前进程中复用。重新构建磁盘上的扩展不会替换进程中已经加载的代码，
需要启动新进程才能使用新构建。

当 `--python_model_path` 指向 checkout 根目录时，重启服务即可使用 Python 模型和
layer 的修改。修改 C++ 或编译为原生 AOT 产物的 Python DSL kernel 后，需要重新构建对应产物。

## 新增算子

1. 在对应平台包的框架子目录中添加实现。Python DSL 实现如果由 JIT runtime 和原生
   AOT 构建共用，应保持独立。
2. 原生构建使用 TileLang 实现时，在 `xllm/compiler/tilelang/targets/ascend/aot/`
   添加薄适配器，不复制 DSL 程序主体。
3. 在平台对应的算子模块及包导出中绑定名称：NPU 更新 `_EXPORTS`，CUDA 更新显式导入，
   并更新 `__all__`。需要稳定计算图节点、FakeTensor 传播或张量修改追踪时，注册
   `custom_op` 及其 FakeTensor 实现。C++ 算子的 FakeTensor 契约放在平台的 `_custom_op.py`。
4. 添加该平台的底层数值测试和计算图、FakeTensor 测试。不为对齐导出列表而在其他平台添加占位实现。
5. 算子改变模型支持范围时，更新 `model_platform_support.py`。
6. 确认依赖仍然从 models 经 layers 指向 kernels，没有反向依赖。

## 新增平台

1. 在现有平台包旁创建 `kernels_<device>/`，实现该平台首批模型所需的 API。
2. 保持包独立：管理自己的导出，不导入其他平台包。
3. 添加 `_custom_op.py`，实现该平台 C++ 算子的 FakeTensor 契约。
   在原生算子注册后，通过包的 `_initialize_runtime()` 钩子导入它。
4. 修改 `xllm-kernel/python/xllm_kernel/platform.py` 中的 `Platform`，扩展 `PlatformEnum` 和
   `_torch_device_type()`，使其识别 `<device>`；在 `xllm.python.initialize_runtime()`
   中增加对应分支。
5. `python setup.py build --device <device>` 随后会暂存该包。
   对应包尚不存在时，该设备的构建会记录已有平台包，并且不打包任何平台 kernel 包。
6. 只有平台路径通过功能测试后，才在 `model_platform_support.py` 中标记模型支持该平台。

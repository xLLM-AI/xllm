---
title: "原生 C++ 扩展"
---

<!-- Copyright 2026 The xLLM Authors. -->

xLLM 可以在查询模型或 executor 注册表之前加载原生共享库。共享库使用现有的
`REGISTER_*` 宏注册实现，并使用宿主的注册表单例。扩展必须使用与加载它的
xLLM 构建一同安装的 SDK 进行构建，且二者的构建版本必须完全匹配。

## 构建 SDK

xLLM 的标准安装包含 C++ 头文件和 CMake 包 `XLLMNative`。构建并安装 xLLM 时，
设备、Torch、编译器和 C++ ABI 设置必须与扩展保持一致：

```sh
cmake -S . -B build -DUSE_NPU=ON -DUSE_CXX11_ABI=OFF \
  -DXLLM_ATB_LAYERS_SOURCE_DIR="${PWD}/third_party/xllm_atb_layers"
cmake --build build --target xllm
cmake --install build --prefix /opt/xllm
```

从源码归档构建发布版本时，应传入构建扩展时使用的确切源码修订版本，例如
`-DNATIVE_LIBRARY_BUILD_REVISION=0123456789abcdef`。构建发布包时，不能将此值保留为
`unknown`，因为宿主会拒绝未提供已知修订版本的扩展。

对于 NPU 构建，配置使用 SDK 的项目时，需要设置 `NPU_HOME_PATH`、`ATB_HOME_PATH`
和 `PYTORCH_NPU_INSTALL_PATH`，以指定匹配的 CANN、ATB 和 torch_npu 头文件位置。
SDK 不会将 xLLM 的 `models` 或 `runtime` 静态库链接到扩展中。

对于 MLU 构建，需要将 `NEUWARE_HOME` 和 `PYTORCH_MLU_INSTALL_PATH` 设置为匹配的
Neuware SDK 和 torch_mlu 包路径。对于 MUSA 构建，需要将 `MUSA_HOME` 和
`TORCH_MUSA_PYTHONPATH` 设置为匹配的 MUSA SDK 路径和 TorchMusa CMake 包前缀。
使用 SDK 的项目从自身安装的设备厂商 SDK 中查找这些头文件。

## 构建扩展

使用安装后的接口目标获取头文件、Torch 的 ABI 标志和后端定义。在 DSO（动态共享对象）
中使用一次描述符宏，并使用与内置实现相同的静态注册宏：

```cmake
find_package(XLLMNative CONFIG REQUIRED)
add_library(example_executor SHARED example_executor.cpp)
target_link_libraries(example_executor PRIVATE XLLM::native_extension)
set_target_properties(example_executor PROPERTIES PREFIX "")
```

```cpp
#include "core/common/native_library.h"
#include "core/runtime/executor_impl_factory.h"

DEFINE_NATIVE_LIBRARY()
REGISTER_EXECUTOR("llm", "custom", ExampleExecutor, 100);
```

`extension_impl` 用于在模型、executor 等组件的注册表中选择同一个扩展族。
原有的 `model_impl` 继续选择模型后端（`native` 或 `python`），不再承担扩展选择职责。
每个注册表按具体类型和扩展族选择优先级最高的实现，优先级相同时保留最先注册的实现。

模型实现使用相同的注册入口。内置的双参数 `REGISTER_CAUSAL_MODEL` 和
`REGISTER_CAUSAL_VLM_MODEL` 宏仍用于默认的原生实现注册，扩展可以通过
`REGISTER_CAUSAL_LM` 或 `REGISTER_CAUSAL_VLM` 注册直接子类。两个宏都要求
`CausalLM` 子类；VLM 扩展需派生自 `CausalVLMImpl<...>`（同时实现
`CausalVLM` 的 `CausalLM`）：

```cpp
#include "models/model_registry.h"

REGISTER_CAUSAL_LM(example_model, "custom", ExampleCausalLM, 100);
REGISTER_CAUSAL_VLM(example_vlm, "custom", ExampleVlmAdapter, 100);
```

启动 xLLM 时指定匹配的扩展族，例如 `--extension_impl=custom`。未知扩展族会被拒绝。
如果扩展族没有注册某类组件，该类组件继续使用原有选择方式，因此扩展可以只替换模型或
只替换 executor。如果扩展族已注册某类组件，则必须提供当前请求的具体类型，缺失注册
会报错。后续新增组件类别时，也可以复用同一个扩展族选择参数和目录。

扩展族不能使用模型后端保留名称 `native`、`cpp`、`auto`、`python`、`py` 或空字符串。

原生模型扩展要求使用 native 模型后端。扩展的模型类型会在内置 NPU 后端别名之前解析，
因此扩展可以替换受支持模型类型的实现，而无需向宿主暴露具体的类实现。在 NPU 上，
选定的模型扩展会将 `npu_kernel_backend=AUTO` 解析为 `TORCH`；可根据扩展需求显式
指定 `ATB`、`TORCH` 或 `XLITE`。`XLITE` 仍要求 xLLM 构建启用 `USE_XLITE=ON`。
扩展负责支持所请求的后端。新的模型类型也可以注册为 `native`，加载其工厂后便可
使用默认的模型选择方式。

四参数 `REGISTER_EXECUTOR` 示例用于注册指定扩展族的 executor。现有的双参数注册和
带优先级的三参数注册保留默认选择方式。模型也可以通过 `CausalLM::executor_type()`
返回已注册的 executor 键，executor 工厂会对该键应用选定的扩展族。扩展拥有子类及其
算子；xLLM 负责注册表、生命周期、流、缓存和工厂调用。

描述符记录 xLLM 源码修订版本、Torch 版本、编译器、标准库及 C++ ABI、架构、
指针位宽、已启用的设备支持及 kernel 后端特性。可选的 XLITE NPU kernel 后端
与设备支持分别记录。
当前描述符版本为 2；升级 xLLM 时，需使用匹配的 SDK 重新构建扩展。
加载器会在 DSO 的静态初始化器执行后检查这些字段，
发现不匹配时终止启动。此检查用于发现意外的 ABI 不匹配，不用于验证扩展的可信性。

## 加载扩展

启动 xLLM 之前，将 `XLLM_NATIVE_LIBRARIES` 设置为由冒号分隔的共享库绝对路径列表：

```sh
XLLM_NATIVE_LIBRARIES=/opt/xllm/plugins/example_executor.so \
  /opt/xllm/bin/xllm --model=/models/example --extension_impl=custom
```

通过 spawn 启动的 worker 会继承同一个环境变量，并接收父进程选定的
`model_impl` 后端和 `extension_impl` 扩展族。
Python 模块和 C/C++ API 会在创建模型或查询注册表之前加载扩展。
为每个公共 API 实例分别设置选择器：Python 的 `LLM`、`VLM`、`Embedding`
传入 `extension_impl="custom"`；直接创建 master 时设置
`Options::extension_impl("custom")`；C/C++ API 则设置初始化选项的
`extension_impl` 字段。空选择器保留常规组件选择，一个实例的选择不会改变其他实例。
C 初始化结构的 ABI 版本为 2，调用方必须使用匹配的头文件和动态库重新编译。

共享库仅加载一次，并在进程的整个生命周期内保持驻留，因为注册的工厂、对象和虚函数表
可能引用其中的代码。不支持卸载、热重载，也不支持在已经开始处理的请求中进行注册。

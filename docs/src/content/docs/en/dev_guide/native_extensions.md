---
title: "Native C++ Extensions"
---

<!-- Copyright 2026 The xLLM Authors. -->

xLLM can load native shared libraries before it queries model or executor
registries. A library registers implementations with the existing `REGISTER_*`
macros and uses the host's registry singleton. The extension must be built with
the SDK installed alongside the exact xLLM build that will load it.

## Build the SDK

The normal xLLM installation includes C++ headers and the CMake package
`XLLMNative`. Build and install xLLM with the same device, Torch, compiler, and
C++ ABI settings used by the extension:

```sh
cmake -S . -B build -DUSE_NPU=ON -DUSE_CXX11_ABI=OFF \
  -DXLLM_ATB_LAYERS_SOURCE_DIR="${PWD}/third_party/xllm_atb_layers"
cmake --build build --target xllm
cmake --install build --prefix /opt/xllm
```

Release builds made from a source archive should pass the exact source revision
used to build the extension, for example
`-DNATIVE_LIBRARY_BUILD_REVISION=0123456789abcdef`. A package build must not leave
this value as `unknown`, because the host rejects extensions without a known
revision.

For NPU builds, set `NPU_HOME_PATH`, `ATB_HOME_PATH`, and
`PYTORCH_NPU_INSTALL_PATH` when configuring the consumer. They identify the
matching CANN, ATB, and torch_npu headers. The SDK does not link xLLM's static
`models` or `runtime` archives into an extension.

For MLU builds, set `NEUWARE_HOME` and `PYTORCH_MLU_INSTALL_PATH` to the matching
Neuware SDK and torch_mlu package. For MUSA builds, set `MUSA_HOME` and
`TORCH_MUSA_PYTHONPATH` to the matching MUSA SDK and TorchMusa CMake package
prefix. The consumer resolves these headers from its own vendor installation.

## Build an extension

Use the installed interface target for headers, Torch's ABI flags, and backend
definitions. Include the descriptor macro once in the DSO and use the same
static registration macros as built-in implementations:

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

`extension_impl` selects an extension family across component registries,
including models and executors. The original `model_impl` setting continues to
select the model backend (`native` or `python`); it does not select extensions.
Each registry chooses the highest-priority registration for its concrete type
and implementation family. Equal priorities keep the first registration.

Model implementations use the same registration boundary. The built-in
two-argument `REGISTER_CAUSAL_MODEL` and `REGISTER_CAUSAL_VLM_MODEL` macros
remain the default native registrations, while an extension can register a
direct subclass with `REGISTER_CAUSAL_LM` or `REGISTER_CAUSAL_VLM`. Both
macros require a `CausalLM` subclass; a VLM extension derives from
`CausalVLMImpl<...>` (a `CausalLM` that also implements `CausalVLM`):

```cpp
#include "models/model_registry.h"

REGISTER_CAUSAL_LM(example_model, "custom", ExampleCausalLM, 100);
REGISTER_CAUSAL_VLM(example_vlm, "custom", ExampleVlmAdapter, 100);
```

Start xLLM with the matching extension family, for example
`--extension_impl=custom`. An unknown family is rejected. If the family has no
registrations for a component category, that component keeps its normal
selection. This allows an extension to replace only models or only executors.
If the family registers that category, it must supply the requested concrete
type; a missing registration is an error. New component categories can use the
same family selector and catalog.

Family names must not use the reserved model backend names `native`, `cpp`,
`auto`, `python`, or `py`, or an empty string.

Native model extensions require the native model backend. Their model types
are resolved before the built-in NPU backend aliases, so an extension can
replace a supported model type without exposing its class implementation to
the host. On NPU, a selected model extension uses `TORCH` when
`npu_kernel_backend=AUTO`; set `ATB`, `TORCH`, or `XLITE` explicitly when needed
by the extension. `XLITE` still requires an xLLM build with `USE_XLITE=ON`.
The extension is responsible for supporting the requested backend. A new
model type can also register as `native` and use the default model selection
once its factory has been loaded.

The four-argument `REGISTER_EXECUTOR` example registers a family-specific
executor. Existing two-argument registrations and three-argument registrations
with a priority retain the default selection. A model can also return a
registered executor key from `CausalLM::executor_type()`; the executor factory
applies the selected extension family to that key. The extension owns the
subclass and its kernels; xLLM owns the registry, lifecycle, streams, caches,
and factory call.

The descriptor records the xLLM source revision, Torch version, compiler,
standard library and C++ ABI, architecture, pointer width, enabled device
support, and kernel backend features. The optional XLITE NPU kernel backend
is recorded separately from device support.
The current descriptor is version 2; rebuild extensions with the matching SDK
when updating xLLM.
The loader checks these fields after the DSO's static initializers
run and terminates startup on a mismatch. This check detects accidental ABI
mismatches; it does not authenticate the extension.

## Load extensions

Set `XLLM_NATIVE_LIBRARIES` to a colon-separated list of absolute shared-library
paths before starting xLLM:

```sh
XLLM_NATIVE_LIBRARIES=/opt/xllm/plugins/example_executor.so \
  /opt/xllm/bin/xllm --model=/models/example --extension_impl=custom
```

Spawned workers inherit the same environment variable and receive the selected
`model_impl` backend and `extension_impl` family from the parent process. The
Python module and C/C++ APIs load extensions before model creation or registry
queries. Set the selector on each public API instance: pass
`extension_impl="custom"` to Python `LLM`, `VLM`, or `Embedding`; set
`Options::extension_impl("custom")` for direct master construction; or set the
`extension_impl` field on the C/C++ initialization options. Empty selectors
preserve normal component selection, and one instance's choice does not change
another's. The C initialization structure is ABI version 2; rebuild callers
against the matching headers and library.

Libraries are loaded once and remain resident for the process lifetime because
registered factories, objects, and vtables may refer to their code. Unloading,
hot reload, and registering from an already running request are unsupported.

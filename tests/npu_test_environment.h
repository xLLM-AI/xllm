/* Copyright 2026 The xLLM Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/jd-opensource/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#pragma once

namespace xllm::testing {

// Mirror xllm.cpp::main()::init_npu_python_runtime(): aclInit tolerating
// ACL_ERROR_INTERNAL_ERROR, Py_InitializeEx, import torch_npu + _npu_init(),
// then PyEval_SaveThread() to release the GIL. Idempotent — safe to call from
// multiple Environments in the same process.
void init_npu_test_runtime();

// Reacquire the GIL saved by init_npu_test_runtime so pybind11 objects held by
// torch_npu / torch can dec_ref safely during process exit. Call from the
// Environment's TearDown; idempotent.
void finalize_npu_test_runtime();

}  // namespace xllm::testing

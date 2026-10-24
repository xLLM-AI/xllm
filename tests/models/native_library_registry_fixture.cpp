/* Copyright 2026 The xLLM Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/xLLM-AI/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "core/common/native_library.h"
#include "core/runtime/executor_impl_factory.h"
#include "models/model_registry.h"

DEFINE_NATIVE_LIBRARY()

namespace xllm {

REGISTER_MODEL_BACKEND(native_library_fixture, "shared_object");

class NativeLibraryExecutor final : public ExecutorImpl {
 public:
  NativeLibraryExecutor(CausalLM* /*model*/,
                        const ModelArgs& /*args*/,
                        const torch::Device& /*device*/,
                        const runtime::Options& /*options*/) {}

  bool supports_prepared_attention_metadata() const override { return true; }

  ModelOutput run(const torch::Tensor& /*tokens*/,
                  const torch::Tensor& /*positions*/,
                  std::vector<KVCache>& /*kv_caches*/,
                  const ModelParamsView& /*params*/) override {
    return {};
  }
};

REGISTER_EXECUTOR("native_library_executor",
                  "custom",
                  NativeLibraryExecutor,
                  100);

}  // namespace xllm

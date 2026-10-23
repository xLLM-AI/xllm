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

#include "core/runtime/rec_executor_impl_factory.h"

#include "core/framework/model/model_family_args.h"
#include "core/runtime/rec_executor_impl.h"
#if defined(USE_NPU)
#include "core/runtime/acl_graph_executor_impl.h"
#elif defined(USE_MLU)
#include "core/runtime/mlu_graph_executor_impl.h"
#elif defined(USE_CUDA)
#include "core/runtime/cuda_graph_executor_impl.h"
#elif defined(USE_DCU)
#include "core/runtime/dcu_graph_executor_impl.h"
#elif defined(USE_MUSA)
#include "core/runtime/musa_graph_executor_impl.h"
#endif

namespace xllm {

RecExecutorImplFactory& RecExecutorImplFactory::get_instance() {
  static RecExecutorImplFactory instance;
  return instance;
}

bool RecExecutorImplFactory::register_creator(const std::string& name,
                                              Creator creator) {
  return creators_.emplace(name, std::move(creator)).second;
}

std::unique_ptr<RecExecutorBackend>
RecExecutorImplFactory::create_executor_impl(RecModel* model,
                                             const TransformerModelArgs& args,
                                             const torch::Device& device,
                                             const runtime::Options& options,
                                             const std::string& backend) {
  auto it = creators_.find(backend);
  CHECK(it != creators_.end())
      << "No Rec executor backend registered: " << backend;
  return it->second(model, args, device, options);
}

}  // namespace xllm

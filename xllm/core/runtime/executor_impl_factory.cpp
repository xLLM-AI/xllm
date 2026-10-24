/* Copyright 2025-2026 The xLLM Authors.

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

#include "core/runtime/executor_impl_factory.h"

#include <glog/logging.h>

#include <stdexcept>
#include <utility>

#include "core/common/extension_registry.h"
#include "core/framework/model/model_family_args.h"
#include "platform/device.h"
#include "runtime/base_executor_impl.h"
#include "runtime/py_executor_impl.h"
#include "runtime/vlm_executor_impl.h"
#if defined(USE_NPU)
#include "runtime/acl_graph_executor_impl.h"
#if defined(USE_XLITE)
#include "runtime/xlite_executor_impl.h"
#endif
#elif defined(USE_MLU)
#include "runtime/mlu_graph_executor_impl.h"
#elif defined(USE_CUDA)
#include "runtime/cuda_graph_executor_impl.h"
#elif defined(USE_DCU)
#include "runtime/dcu_graph_executor_impl.h"
#endif

namespace xllm {

ExecutorImplFactory& ExecutorImplFactory::get_instance() {
  static ExecutorImplFactory instance;
  return instance;
}

bool ExecutorImplFactory::register_creator(const std::string& executor_type,
                                           Creator creator) {
  return register_creator(executor_type, std::move(creator), /*priority=*/0);
}

bool ExecutorImplFactory::register_creator(const std::string& executor_type,
                                           Creator creator,
                                           int32_t priority) {
  return register_creator(executor_type,
                          /*extension_impl=*/"",
                          std::move(creator),
                          priority);
}

bool ExecutorImplFactory::register_creator(std::string executor_type,
                                           std::string extension_impl,
                                           Creator creator,
                                           int32_t priority) {
  auto& implementations = creators_[std::move(executor_type)];
  auto it = implementations.find(extension_impl);
  if (it != implementations.end() && priority <= it->second.priority) {
    return false;
  }

  if (!extension_impl.empty()) {
    ExtensionRegistry::get_instance().register_implementation(
        extension_impl, /*component=*/"executor");
  }
  implementations.insert_or_assign(std::move(extension_impl),
                                   Registration{priority, std::move(creator)});
  return true;
}

std::unique_ptr<ExecutorImpl> ExecutorImplFactory::create_executor_impl(
    CausalLM* model,
    const ModelArgsBundle& args,
    const torch::Device& device,
    const runtime::Options& options,
    const std::string& executor_type) {
  const std::string& requested_impl = options.extension_impl();
  const auto& extensions = ExtensionRegistry::get_instance();
  CHECK(requested_impl.empty() || extensions.has_implementation(requested_impl))
      << "Unknown extension implementation: " << requested_impl;

  const std::string default_impl;
  const std::string& extension_impl =
      extensions.has_implementation(requested_impl, /*component=*/"executor")
          ? requested_impl
          : default_impl;

  auto executor_it = creators_.find(executor_type);
  if (executor_it == creators_.end()) {
    CHECK(extension_impl.empty())
        << "No executor registered for extension implementation: "
        << extension_impl << ", executor type: " << executor_type;
    throw std::runtime_error("No valid executor type found: " + executor_type);
  }

  auto implementation_it = executor_it->second.find(extension_impl);
  CHECK(extension_impl.empty() ||
        implementation_it != executor_it->second.end())
      << "No executor registered for extension implementation: "
      << extension_impl << ", executor type: " << executor_type;
  if (implementation_it == executor_it->second.end()) {
    throw std::runtime_error("No valid executor type found: " + executor_type);
  }

  return implementation_it->second.creator(model, args, device, options);
}

}  // namespace xllm

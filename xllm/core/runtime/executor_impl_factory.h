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

#pragma once

#include <torch/types.h>

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>

#include "core/common/macros.h"
#include "core/framework/model/causal_lm.h"
#include "core/framework/model/model_family_args.h"
#include "core/runtime/executor_impl.h"
#include "core/runtime/options.h"

namespace xllm {

class ExecutorImplFactory {
 public:
  using Creator =
      std::function<std::unique_ptr<ExecutorImpl>(CausalLM*,
                                                  const TransformerModelArgs&,
                                                  const torch::Device&,
                                                  const runtime::Options&)>;

  static ExecutorImplFactory& get_instance();

  bool register_creator(const std::string& executor_type, Creator creator);
  bool register_creator(const std::string& executor_type,
                        Creator creator,
                        int32_t priority);
  bool register_creator(std::string executor_type,
                        std::string extension_impl,
                        Creator creator,
                        int32_t priority = 0);

  std::unique_ptr<ExecutorImpl> create_executor_impl(
      CausalLM* model,
      const TransformerModelArgs& args,
      const torch::Device& device,
      const runtime::Options& options,
      const std::string& executor_type);

  DISALLOW_COPY_AND_ASSIGN(ExecutorImplFactory);

 private:
  ExecutorImplFactory() = default;

  ~ExecutorImplFactory() = default;

  struct Registration {
    int32_t priority;
    Creator creator;
  };

  using Implementations = std::unordered_map<std::string, Registration>;
  std::unordered_map<std::string, Implementations> creators_;
};

#define CONCAT_EXECUTOR_IMPL(left, right) left##right
#define CONCAT_EXECUTOR(left, right) CONCAT_EXECUTOR_IMPL(left, right)

// Higher priorities replace lower priorities within the same implementation;
// ties keep the first registration. Legacy registrations use the built-in
// implementation, and omitting priority preserves priority zero.
#define REGISTER_EXECUTOR_WITH_IMPL(                                         \
    executor_type, extension_impl, class_type, priority)                     \
  namespace {                                                                \
  const bool CONCAT_EXECUTOR(xllm_executor_registration_,                    \
                             __COUNTER__) = []() -> bool {                   \
    return ::xllm::ExecutorImplFactory::get_instance().register_creator(     \
        executor_type,                                                       \
        extension_impl,                                                      \
        [](::xllm::CausalLM* model,                                          \
           const ::xllm::TransformerModelArgs& args,                                    \
           const torch::Device& device,                                      \
           const ::xllm::runtime::Options& options)                          \
            -> std::unique_ptr<::xllm::ExecutorImpl> {                       \
          return std::make_unique<class_type>(model, args, device, options); \
        },                                                                   \
        priority);                                                           \
  }();                                                                       \
  }

#define REGISTER_EXECUTOR_DEFAULT(executor_type, class_type) \
  REGISTER_EXECUTOR_WITH_IMPL(executor_type, "", class_type, 0)
#define REGISTER_EXECUTOR_WITH_PRIORITY(executor_type, class_type, priority) \
  REGISTER_EXECUTOR_WITH_IMPL(executor_type, "", class_type, priority)
#define SELECT_EXECUTOR_REGISTRATION(_1, _2, _3, _4, registration, ...) \
  registration
#define REGISTER_EXECUTOR(...)                                  \
  SELECT_EXECUTOR_REGISTRATION(__VA_ARGS__,                     \
                               REGISTER_EXECUTOR_WITH_IMPL,     \
                               REGISTER_EXECUTOR_WITH_PRIORITY, \
                               REGISTER_EXECUTOR_DEFAULT)(__VA_ARGS__)

}  // namespace xllm

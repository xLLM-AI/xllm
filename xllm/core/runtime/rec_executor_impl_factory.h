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

#pragma once

#include <functional>
#include <memory>
#include <string>
#include <unordered_map>

#include "core/common/macros.h"
#include "core/framework/model/model_family_args.h"
#include "core/framework/model/rec_model.h"
#include "core/runtime/options.h"
#include "core/runtime/rec_executor_backend.h"

namespace xllm {

class RecExecutorImplFactory final {
 public:
  using Creator = std::function<std::unique_ptr<RecExecutorBackend>(
      RecModel*,
      const ModelArgs&,
      const torch::Device&,
      const runtime::Options&)>;

  static RecExecutorImplFactory& get_instance();
  bool register_creator(const std::string& name, Creator creator);
  std::unique_ptr<RecExecutorBackend> create_executor_impl(
      RecModel* model,
      const ModelArgs& args,
      const torch::Device& device,
      const runtime::Options& options,
      const std::string& backend);

  DISALLOW_COPY_AND_ASSIGN(RecExecutorImplFactory);

 private:
  RecExecutorImplFactory() = default;
  std::unordered_map<std::string, Creator> creators_;
};

#define REGISTER_REC_EXECUTOR(backend, class_type)                           \
  namespace {                                                                \
  const bool class_type##_rec_registered = []() {                            \
    return RecExecutorImplFactory::get_instance().register_creator(          \
        backend,                                                             \
        [](RecModel* model,                                                  \
           const ModelArgs& args,                                            \
           const torch::Device& device,                                      \
           const runtime::Options& options)                                  \
            -> std::unique_ptr<RecExecutorBackend> {                         \
          return std::make_unique<class_type>(model, args, device, options); \
        });                                                                  \
  }();                                                                       \
  }

}  // namespace xllm

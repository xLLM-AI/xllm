/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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

#include <glog/logging.h>

#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>

#include "models/dit/attn_processor/attn_processor.h"
#include "models/dit/parallel_mode.h"

namespace xllm::dit {

template <typename ProcessorBaseType, typename AttentionType>
class AttnProcessorFactory final {
 public:
  using Creator = std::function<std::unique_ptr<ProcessorBaseType>(
      AttentionType& attention,
      ProcessGroup* process_group)>;

  static AttnProcessorFactory& get_instance() {
    static AttnProcessorFactory instance;
    return instance;
  }

  bool register_creator(const std::string& model_name,
                        ParallelMode parallel_mode,
                        Creator creator) {
    return creators_[model_name]
        .emplace(parallel_mode, std::move(creator))
        .second;
  }

  std::unique_ptr<ProcessorBaseType> create_attn_processor(
      const std::string& model_name,
      ParallelMode parallel_mode,
      AttentionType& attention,
      ProcessGroup* process_group) const {
    auto model_it = creators_.find(model_name);
    CHECK(model_it != creators_.end())
        << "No attention processors registered for model: " << model_name;

    auto processor_it = model_it->second.find(parallel_mode);
    CHECK(processor_it != model_it->second.end())
        << "No attention processor registered for model: " << model_name
        << ", parallel mode: " << static_cast<int32_t>(parallel_mode);
    return processor_it->second(attention, process_group);
  }

  AttnProcessorFactory(const AttnProcessorFactory&) = delete;
  AttnProcessorFactory& operator=(const AttnProcessorFactory&) = delete;

 private:
  using ModeCreators = std::unordered_map<ParallelMode, Creator>;

  AttnProcessorFactory() = default;
  ~AttnProcessorFactory() = default;

  std::unordered_map<std::string, ModeCreators> creators_;
};

}  // namespace xllm::dit

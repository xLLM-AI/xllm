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

#include "core/framework/parallel_state/process_group.h"

namespace xllm::dit {

template <typename AttentionType, typename OutputType, typename... InputTypes>
class AttnProcessor {
 public:
  explicit AttnProcessor(AttentionType& attention) : attention_(attention) {}
  virtual ~AttnProcessor() = default;

  virtual OutputType forward(InputTypes... inputs) = 0;

 protected:
  AttentionType& attention() { return attention_; }

 private:
  AttentionType& attention_;
};

template <typename AttentionType, typename OutputType, typename... InputTypes>
class SequenceParallelAttnProcessor
    : public AttnProcessor<AttentionType, OutputType, InputTypes...> {
 public:
  SequenceParallelAttnProcessor(AttentionType& attention,
                                ProcessGroup* process_group)
      : AttnProcessor<AttentionType, OutputType, InputTypes...>(attention),
        process_group_(process_group) {
    CHECK(process_group_ != nullptr)
        << "Sequence-parallel attention requires a process group";
    CHECK_GT(process_group_->world_size(), 1)
        << "Sequence-parallel attention requires world_size greater than one";
  }

 protected:
  ProcessGroup* process_group() const { return process_group_; }

 private:
  ProcessGroup* process_group_{nullptr};
};

}  // namespace xllm::dit

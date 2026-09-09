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

#include <glog/logging.h>
#include <torch/torch.h>

#include <cstdint>

#include "core/framework/model_context.h"
#include "framework/parallel_state/parallel_args.h"
#include "framework/parallel_state/parallel_state.h"
#include "framework/state_dict/state_dict.h"
#include "framework/state_dict/utils.h"

namespace xllm {
namespace layer {

// Embedding parallelized in the embedding dimension.
class WordEmbeddingImpl : public torch::nn::Module {
 public:
  WordEmbeddingImpl(const ModelContext& context);
  WordEmbeddingImpl(int64_t num_embeddings,
                    int64_t embedding_dim,
                    const ParallelArgs& parallel_args,
                    const torch::TensorOptions& options);

  // The input to the module is a list of indices, and the output is the
  // corresponding word embeddings.
  torch::Tensor forward(torch::Tensor input);

  // load the weight from the checkpoint
  void load_state_dict(const StateDict& state_dict);

  void pretty_print(std::ostream& stream) const override {
    stream << name() << " " << weight_.sizes() << " " << weight_.device();
  }

  // return the weight (for testing)
  torch::Tensor weight() const { return weight_; }

 private:
  // rank of current process
  PROPERTY(int32_t, rank) = 0;

  // world size
  PROPERTY(int32_t, world_size) = 0;

  // parameter members, must be registered
  DEFINE_WEIGHT(weight);

  // parallel args
  ParallelArgs parallel_args_;
};

}  // namespace layer
}  // namespace xllm

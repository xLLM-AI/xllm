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

#include "word_embedding_impl.h"

namespace xllm {
namespace layer {

WordEmbeddingImpl::WordEmbeddingImpl(const ModelContext& context)
    : WordEmbeddingImpl(context.get_model_args().vocab_size(),
                        context.get_model_args().hidden_size(),
                        context.get_parallel_args(),
                        context.get_tensor_options()) {}

WordEmbeddingImpl::WordEmbeddingImpl(int64_t num_embeddings,
                                     int64_t embedding_dim,
                                     const ParallelArgs& parallel_args,
                                     const torch::TensorOptions& options)
    : parallel_args_(parallel_args) {
  rank_ = parallel_args_.tp_group_->rank();
  world_size_ = parallel_args_.tp_group_->world_size();

  CHECK(embedding_dim % world_size_ == 0)
      << "out_features " << embedding_dim << " not divisible by world_size "
      << world_size_;
  const int64_t embedding_dim_per_partition = embedding_dim / world_size_;

  // register the weight parameter
  weight_ = register_parameter(
      "weight",
      torch::empty({num_embeddings, embedding_dim_per_partition}, options),
      /*requires_grad=*/false);
}

// The input to the module is a list of indices, and the output is the
// corresponding word embeddings.
torch::Tensor WordEmbeddingImpl::forward(torch::Tensor input) {
  namespace F = torch::nn::functional;
  auto output = F::embedding(input, weight_);
  if (world_size_ > 1) {
    output = xllm::parallel_state::gather(output, parallel_args_.tp_group_);
  }
  return output;
}

// load the weight from the checkpoint
void WordEmbeddingImpl::load_state_dict(const StateDict& state_dict) {
  const int64_t rank = rank_;
  const int64_t world_size = world_size_;
  LOAD_SHARDED_WEIGHT(weight, 1);
}

}  // namespace layer
}  // namespace xllm

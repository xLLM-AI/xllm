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

#include <future>
#include <vector>

#include "core/framework/batch/rec_forward_input_builder.h"
#include "core/framework/model/model_args.h"
#include "core/framework/model/model_input_params.h"
#include "core/framework/multimodal/mm_data.h"
#include "core/framework/request/sequence.h"
#include "core/framework/request/sequences_group.h"
#include "core/runtime/forward_params.h"
#include "core/util/threadpool.h"

namespace xllm {

class OneRecForwardInputBuilder : public RecForwardInputBuilder {
 public:
  explicit OneRecForwardInputBuilder(const BatchInputData& data,
                                     const ModelArgs* args,
                                     MPMCThreadPool* thread_pool = nullptr);

 public:
  ForwardInput build_rec_forward_input(
      uint32_t num_decoding_tokens,
      uint32_t min_decoding_batch_size) override;

 private:
  const std::vector<SequencesGroup*>& sequence_groups_;
  const std::vector<uint32_t>& allowed_max_tokens_;
  const std::vector<torch::Tensor>& input_embeddings_vec_;
  const std::vector<MMData>& mm_data_vec_;
  std::vector<BlockTransferInfo>* swap_block_transfer_infos_ = nullptr;
  const uint64_t batch_id_;
  const ModelArgs* args_ = nullptr;
  MPMCThreadPool* thread_pool_ = nullptr;
  BatchForwardType batch_forward_type_;
  struct HighPerformanceCache {
    struct MemoryPool {
      std::vector<std::vector<int32_t>> int32_pools;
      size_t pool_index = 0;

      std::vector<int32_t>& get_int32_vector(size_t reserve_size = 0) {
        if (pool_index >= int32_pools.size()) {
          int32_pools.emplace_back();
        }
        auto& vec = int32_pools[pool_index++];
        vec.clear();
        if (reserve_size > 0) {
          vec.reserve(reserve_size);
        }
        return vec;
      }

      void reset() { pool_index = 0; }
    };

    struct CacheData {
      std::vector<int32_t> encoder_tokens;
      std::vector<int32_t> encoder_seq_lens;
      std::vector<torch::Tensor> encoder_sparse_embeddings;
      std::vector<torch::Tensor> decoder_context_embeddings;
    };

    torch::Tensor fixed_positions_tensor;
    torch::Tensor fixed_encoder_positions_tensor;
    torch::Tensor empty_tensor;
    bool tensors_initialized = false;

    MemoryPool memory_pool;
    CacheData cache_data;

    HighPerformanceCache() = default;

    void ensure_tensors_initialized() {
      if (!tensors_initialized) {
        fixed_positions_tensor = torch::tensor({0}, torch::kInt);
        fixed_encoder_positions_tensor = torch::tensor({0}, torch::kInt);
        empty_tensor = torch::tensor(std::vector<int32_t>{}, torch::kInt);
        tensors_initialized = true;
      }
    }
  };

  static HighPerformanceCache& get_perf_cache();
};

}  // namespace xllm

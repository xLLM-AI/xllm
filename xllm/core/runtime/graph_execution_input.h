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

#include <torch/torch.h>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <optional>

#include "core/common/global_flags.h"
#include "core/framework/model/model_args.h"
#include "core/framework/model/model_params_view.h"
#include "core/framework/model/rec_model_params.h"
#include "core/framework/model/rec_model_utils.h"
#include "core/framework/model/seq_lens.h"
#include "core/layers/common/attention_metadata.h"
#include "core/layers/common/attention_metadata_builder.h"

namespace xllm {

// ModelParamsView retains its cloned owner; Rec snapshots own the native
// aggregate directly. Tensor handles intentionally share storage.
inline ModelParamsView copy_graph_execution_input(
    const ModelParamsView& params) {
  return params.clone().view();
}

inline RecModelParams copy_graph_execution_input(const RecModelParams& params) {
  return params.clone();
}

// Borrow unchanged execution input. Detach host metadata only when graph
// preparation changes it; tensor storage deliberately remains shared.
// The source must outlive this adapter, including when it already owns a view.
template <typename Params>
class GraphExecutionInput final {
 public:
  explicit GraphExecutionInput(const Params& source) : source_(source) {}
  GraphExecutionInput(Params&&) = delete;
  GraphExecutionInput(const GraphExecutionInput&) = delete;
  GraphExecutionInput& operator=(const GraphExecutionInput&) = delete;
  GraphExecutionInput(GraphExecutionInput&&) = default;
  GraphExecutionInput& operator=(GraphExecutionInput&&) = delete;

  const Params& params() const {
    return owned_params_ ? *owned_params_ : source_;
  }

  Params& mutable_params() {
    if (!owned_params_) {
      owned_params_.emplace(copy_graph_execution_input(source_));
    }
    return *owned_params_;
  }

  void pad_dp_token_counts(int32_t token_count) {
    const auto& token_counts = params().parallel.dp_global_token_nums;
    if (token_counts.size() <= 1 || std::all_of(token_counts.begin(),
                                                token_counts.end(),
                                                [token_count](int32_t count) {
                                                  return count == token_count;
                                                })) {
      return;
    }
    auto& padded_counts = mutable_params().parallel.dp_global_token_nums;
    std::fill(padded_counts.begin(), padded_counts.end(), token_count);
  }

 private:
  const Params& source_;
  std::optional<Params> owned_params_;
};

inline bool is_rec_multi_round_graph_input(const ModelParamsView&) {
  return false;
}

inline bool is_rec_multi_round_graph_input(const RecModelParams&) {
  return is_rec_multi_round_mode();
}

inline bool has_llmrec_graph_input(const ModelParamsView&) { return false; }

inline bool has_llmrec_graph_input(const RecModelParams& params) {
  return params.has_llmrec_params();
}

template <typename Params>
inline void fill_persistent_graph_cu_seq_lens(const Params& params,
                                              int64_t actual_batch_size,
                                              int64_t actual_num_tokens,
                                              int64_t padded_num_tokens,
                                              torch::Tensor& q_cu_seq_lens,
                                              torch::Tensor& kv_cu_seq_lens) {
  VLOG(kGraphExecutorLogVerboseLevel)
      << "cumsum q_seq_lens into persistent buffer: src shape="
      << params.attention.device.q_seq_lens.sizes()
      << ", rows=" << actual_batch_size;
  util::fill_query_start_loc(
      params.attention.device.q_seq_lens, actual_batch_size, q_cu_seq_lens);

  VLOG(kGraphExecutorLogVerboseLevel)
      << "cumsum kv_seq_lens into persistent buffer: src shape="
      << params.attention.device.kv_seq_lens.sizes()
      << ", rows=" << actual_batch_size;
  util::fill_query_start_loc(
      params.attention.device.kv_seq_lens, actual_batch_size, kv_cu_seq_lens);

  if (params.meta.batch_forward_type.is_decode() &&
      padded_num_tokens > actual_num_tokens) {
    util::pad_query_start_loc(
        kv_cu_seq_lens, actual_batch_size + 1, padded_num_tokens + 1);
  }
}

template <typename Params>
inline std::shared_ptr<layer::AttentionMetadata> build_graph_attention_metadata(
    const GraphExecutionInput<Params>& graph_input,
    const ModelArgs& args,
    bool skip_cu_seq_lens) {
  const layer::AttentionMetadataBuildOptions build_options{
      .skip_cu_seq_lens = skip_cu_seq_lens};
  std::shared_ptr<layer::AttentionMetadata> attn_metadata =
      std::make_shared<layer::AttentionMetadata>(
          layer::AttentionMetadataBuilder::build(graph_input.params(),
                                                 args.enable_mla(),
                                                 /*attn_mask=*/{},
                                                 /*device=*/std::nullopt,
                                                 build_options));
  attn_metadata->enable_cuda_graph = true;
  return attn_metadata;
}

}  // namespace xllm

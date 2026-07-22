/* Copyright 2025-2026 The xLLM Authors.

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

#include "deepseek_v2_attention.h"

#include <cstdint>
#include <optional>

#include "framework/parallel_state/parallel_state.h"
#include "kernels/mlu/mlu_ops_api.h"
#include "kernels/ops_api.h"
#include "platform/platform.h"

namespace xllm {
namespace layer {

namespace {
void check_phase1_dcp_geometry(int32_t dcp_size,
                               int64_t tp_heads_attn,
                               int64_t full_heads_attn) {
  CHECK_EQ(dcp_size * tp_heads_attn, full_heads_attn)
      << "Phase 1 DCP requires dcp_size * tp_heads == full_heads (i.e. "
         "tp_size == dcp_size); tp_size > dcp_size is a Phase 2 task.";
}

}  // namespace

torch::Tensor DeepseekV2AttentionImpl::forward_dcp(
    const torch::Tensor& positions,
    const torch::Tensor& hidden_states,
    const AttentionMetadata& attn_metadata,
    KVCache& kv_cache) {
  CHECK_GT(dcp_size_, 1) << "forward_dcp requires dcp_size_ > 1.";

  check_phase1_dcp_geometry(dcp_size_, tp_heads_.attn, full_heads_.attn);

  (void)attn_metadata;

  const int64_t tokens = hidden_states.size(0);
  auto k_cache = kv_cache.get_k_cache();
  auto k_cache_scale = kv_cache.get_k_cache_scale();
  auto query_prep = prep_query(hidden_states, tp_heads_);
  torch::Tensor q_input = torch::empty(
      {tokens, tp_heads_.attn, kv_lora_rank_ + qk_rope_head_dim_},
      hidden_states.options());
  torch::Tensor latent_cache = kv_a_proj_with_mqa_(hidden_states);
  fill_q_input(q_input,
               query_prep.q,
               positions,
               attn_metadata,
               /*use_prompt_rope=*/false);
  decode_kv_pre_base(latent_cache, positions, attn_metadata, /*use_prompt_rope=*/false);
  AttentionMetadata local_meta = build_mla_attention_metadata(
      positions,
      hidden_states,
      query_prep.q_norm,
      latent_cache,
      attn_metadata,
      kv_cache,
      k_cache_scale,
      /*is_prefill_phase=*/false,
      /*slot_mapping=*/std::nullopt,
      /*new_block_tables=*/std::nullopt,
      /*new_context_lens=*/std::nullopt);

  {
    torch::Tensor key = latent_cache.unsqueeze(1);
    xllm::kernel::ReshapePagedCacheParams params;
    params.key = key;
    params.k_cache = k_cache;
    params.slot_mapping = local_meta.slot_mapping;
    if (k_cache_scale.has_value()) {
      params.k_cache_scale = k_cache_scale;
      xllm::kernel::quant_to_paged_cache(params);
    } else {
      xllm::kernel::reshape_paged_cache(params);
    }
  }

  q_input = q_input.view({tokens, tp_heads_.attn, -1});
  auto q_gather_ctx = parallel_state::launch_all_gather(
      q_input.contiguous(), dcp_group_);
  torch::Tensor q_full = parallel_state::finish_all_gather(std::move(q_gather_ctx));
  const int64_t head_dim = q_input.size(-1);
  q_full = q_full.permute({1, 0, 2, 3}).contiguous()
               .view({tokens, dcp_size_ * tp_heads_.attn, head_dim});

  const int64_t local_heads = full_heads_.attn;
  torch::Tensor q_decode = q_full.view({tokens, local_heads, head_dim})
                               .unsqueeze(1)
                               .contiguous();  // [tokens, 1, heads, D]
  torch::Tensor partial_out =
      torch::empty({tokens, 1, local_heads, kv_lora_rank_}, hidden_states.options());
  std::optional<torch::Tensor> partial_lse =
      torch::empty({tokens, local_heads, 1},
                   hidden_states.options().dtype(torch::kFloat32));
  int64_t kv_cache_quant_bit_size = -1;
  std::optional<torch::Tensor> k_cache_quant_scale;
  if (k_cache_scale.has_value()) {
    k_cache_quant_scale = k_cache_scale;
    kv_cache_quant_bit_size = 8;
  }

  auto max_seq_len = (local_meta.max_seq_len + dcp_size_ - 1) / dcp_size_;  // ceil div
  auto kv_seq_lens = (local_meta.kv_seq_lens + dcp_size_ - 1) / dcp_size_;  // ceil div
  kv_seq_lens = kv_seq_lens.to(torch::kInt32);
  xllm::kernel::mlu::batch_decode(
      q_decode,
      k_cache,
      partial_out,
      local_meta.block_table,
      kv_seq_lens,
      /*v_cache=*/std::nullopt,
      partial_lse,
      /*q_quant_scale=*/std::nullopt,
      k_cache_quant_scale,
      /*v_cache_quant_scale=*/std::nullopt,
      /*out_quant_scale=*/std::nullopt,
      /*alibi_slope=*/std::nullopt,
      /*mask=*/std::nullopt,
      local_meta.compute_dtype,
      max_seq_len,
      std::max<int64_t>(sliding_window_ - 1, -1),
      /*window_size_right=*/-1,
      attn_scale_,
      /*return_lse=*/true,
      kv_cache_quant_bit_size,
      /*cu_seq_q=*/std::nullopt,
      /*max_seq_q=*/-1,
      /*sink=*/std::nullopt);

  torch::Tensor partial_out_flat =
      partial_out.permute({2, 0, 1, 3}).contiguous()
          .view({local_heads, tokens * kv_lora_rank_});
  auto o_a2a_ctx = parallel_state::launch_all_to_all(partial_out_flat, dcp_group_);
  torch::Tensor partial_out_redistributed =
      parallel_state::finish_all_to_all(std::move(o_a2a_ctx));

  torch::Tensor partial_lse_flat =
      partial_lse.value().permute({1, 0, 2}).contiguous()
          .view({local_heads, tokens});
  auto lse_a2a_ctx = parallel_state::launch_all_to_all(partial_lse_flat, dcp_group_);
  torch::Tensor partial_lse_redistributed =
      parallel_state::finish_all_to_all(std::move(lse_a2a_ctx));

  partial_out_redistributed =
      partial_out_redistributed.view({dcp_size_, 1, tp_heads_.attn, tokens, kv_lora_rank_})
          .permute({0, 3, 1, 2, 4}).contiguous();
  partial_lse_redistributed =
      partial_lse_redistributed.view({dcp_size_, tp_heads_.attn, tokens, 1})
          .permute({0, 2, 1, 3}).contiguous();

  torch::Tensor merged_out = partial_out_redistributed[0].clone();
  torch::Tensor merged_lse = partial_lse_redistributed[0].clone();
  for (int32_t i = 1; i < dcp_size_; ++i) {
    xllm::kernel::mlu::update_out_and_lse(
        merged_out, merged_lse,
        partial_out_redistributed[i], partial_lse_redistributed[i]);
  }

  return project_output(merged_out, tp_heads_);
}

}  // namespace layer
}  // namespace xllm

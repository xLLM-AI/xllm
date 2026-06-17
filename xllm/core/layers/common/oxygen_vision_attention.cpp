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

#include "oxygen_vision_attention.h"

#if defined(USE_MLU)
#include "kernels/mlu/mlu_ops_api.h"
#endif
#include "kernels/ops_api.h"
namespace xllm {
namespace layer {

OxygenVisionAttentionImpl::OxygenVisionAttentionImpl(
    const ModelContext& context)
    : Qwen2VisionAttentionImpl(context, false) {}

torch::Tensor OxygenVisionAttentionImpl::forward(
    torch::Tensor& hidden_states,
    torch::Tensor& m_cos_pos,
    torch::Tensor& m_sin_pos,
    torch::Tensor& cu_seq_len,
    std::vector<int32_t>& cu_seq_len_vec) {
  // 1. qkv projection
  auto qkv = qkv_proj_->forward(hidden_states);
  // 2. split qkv
  auto qkv_split = split_qkv(qkv);
  // 3. transpose [s, b, h, d] -> [b, s, h, d]
  for (auto& tensor : qkv_split) {
    tensor = tensor.transpose(0, 1).contiguous();
  }
  auto q = qkv_split[0];
  auto k = qkv_split[1];
  auto v = qkv_split[2];
  int64_t B = q.size(0);
  int64_t S = q.size(1);
  int64_t head_dim = q.size(3);
  CHECK_EQ(head_dim, hidden_size_per_attention_head_) << "head_dim mismatch";
  int32_t max_seqlen = get_max_sequence_length(cu_seq_len_vec);

  // 4. rope
  // Reshape q, k from [B, S, H, D] to [B*S, H, D] before applying RoPE so
  // that the RoPE kernel sees the correct total token count (B*S = seq_len),
  // not just the batch dimension (B=1).
  q = q.reshape({B * S, num_attention_heads_per_partition_, head_dim});
  k = k.reshape({B * S, num_attention_heads_per_partition_, head_dim});

  // Apply rotary position embedding to both q and k seperately.
  xllm::kernel::RotaryParams rotary_params;
  rotary_params.q = q;
  rotary_params.sin = m_sin_pos;
  rotary_params.cos = m_cos_pos;
  rotary_params.interleaved = false;
  rotary_params.discrete = false;
  rotary_params.cu_query_lens = cu_seq_len;
  rotary_params.max_query_len = max_seqlen;
  xllm::kernel::apply_rotary(rotary_params);
  rotary_params.q = k;
  xllm::kernel::apply_rotary(rotary_params);

  // q, k, v = (rearrange(x, "b s ... -> (b s) ...") for x in [q, k, v])
  // q and k are already [B*S, H, D] after the reshape above; just
  // flatten v to the same shape.
  v = v.view({B * S, v.size(2), v.size(3)});
  torch::Tensor output = torch::zeros_like(q);

  // 5. store k/v cache and do attention
#if defined(USE_MLU)
  std::optional<torch::Tensor> output_lse = std::nullopt;

  xllm::kernel::mlu::batch_prefill(q,
                                   k,
                                   v,
                                   output,
                                   output_lse,
                                   cu_seq_len,
                                   cu_seq_len,
                                   /*alibi_slope=*/std::nullopt,
                                   /*alibi_bias=*/std::nullopt,
                                   /*q_quant_scale=*/std::nullopt,
                                   /*k_quant_scale=*/std::nullopt,
                                   /*v_quant_scale=*/std::nullopt,
                                   /*out_quant_scale=*/std::nullopt,
                                   /*block_table=*/std::nullopt,
                                   max_seqlen,
                                   max_seqlen,
                                   scale_,
                                   /*is_causal=*/false,
                                   /*window_size_left=*/-1,
                                   /*window_size_right=*/-1,
                                   /*compute_dtype=*/"half",
                                   /*return_lse=*/false);
#endif

  // context_layer = rearrange(output, "(b s) h d -> s b (h d)", b=batch_size)
  output = output.view({B, S, -1});
  // [B, S, ...] -> [S, B, ...]
  output = output.transpose(0, 1).reshape({-1, output.size(-1)});
  // 6. output projection
  return proj_->forward(output);
}

}  // namespace layer
}  // namespace xllm

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

#include "qwen2_vision_attention.h"

#include <algorithm>

#include "framework/parallel_state/parallel_state.h"
#if defined(USE_MLU)
#include "kernels/mlu/mlu_ops_api.h"
#endif
#include "kernels/ops_api.h"
#include "layers/common/attention_metadata.h"
namespace xllm {
namespace layer {

Qwen2VisionAttentionImpl::Qwen2VisionAttentionImpl(const ModelContext& context,
                                                   bool has_bias) {
  const auto& args = context.get_model_args();
  const auto& quant_args = context.get_quant_args();
  const auto& parallel_args = context.get_parallel_args();
  const auto& options = context.get_tensor_options();
  const int64_t hidden_size = args.mm_hidden_size();
  const int64_t num_heads = args.mm_num_attention_heads();
  const int64_t tp_size = parallel_args.tp_group_->world_size();
  CHECK(num_heads % tp_size == 0);

  tp_group_ = parallel_args.tp_group_;
  hidden_size_per_attention_head_ = args.mm_head_dim();
  num_attention_heads_per_partition_ = num_heads / tp_size;
  scale_ = 1.0 / std::sqrt(static_cast<float>(hidden_size_per_attention_head_));

  qkv_proj_ =
      register_module("qkv_proj",
                      QKVParallelLinear(hidden_size,
                                        num_attention_heads_per_partition_,
                                        num_attention_heads_per_partition_,
                                        hidden_size_per_attention_head_,
                                        /*num_kv_head_replicas=*/1,
                                        /*bias=*/has_bias,
                                        /*gather_output=*/false,
                                        parallel_args,
                                        options));

  proj_ = register_module("proj",
                          RowParallelLinear(hidden_size,
                                            hidden_size,
                                            /*bias=*/has_bias,
                                            /*input_is_parallelized=*/true,
                                            /*if_reduce_results=*/true,
                                            quant_args,
                                            parallel_args.tp_group_,
                                            options));
}

std::vector<torch::Tensor> Qwen2VisionAttentionImpl::split_qkv(
    const torch::Tensor& qkv) {
  // [s, b, 3 * head * head_dim]
  auto sizes = qkv.sizes();
  int64_t seq_len = qkv.size(0);
  int64_t bs = qkv.sizes() == 3 ? qkv.size(1) : 1;
  torch::Tensor qkv_gathered =
      xllm::parallel_state::all_gather_interleaved(qkv, tp_group_);

  // [s, b, 3 * head * head_dim] -> 3 * [s, b, head * head_dim]
  auto qkv_chunks = qkv_gathered.chunk(3, /*dim=*/-1);
  auto q = qkv_chunks[0];
  auto k = qkv_chunks[1];
  auto v = qkv_chunks[2];

  // 3 * [s, b, head * head_dim]
  if (tp_group_->world_size() > 1) {
    q = xllm::parallel_state::scatter(q, tp_group_);
    k = xllm::parallel_state::scatter(k, tp_group_);
    v = xllm::parallel_state::scatter(v, tp_group_);
  }

  // 3 * [s, b, head * head_dim] -> 3 * [s, b, head, head_dim]
  std::vector<int64_t> new_shape = {seq_len,
                                    bs,
                                    num_attention_heads_per_partition_,
                                    hidden_size_per_attention_head_};
  q = q.reshape(new_shape);
  k = k.reshape(new_shape);
  v = v.reshape(new_shape);

  return {q, k, v};
}

int32_t Qwen2VisionAttentionImpl::get_max_sequence_length(
    const std::vector<int32_t>& cu_seq_len_vec) {
  if (cu_seq_len_vec.size() < static_cast<size_t>(2)) {
    return 0;
  }
  int32_t max_seq_len = 0;
  for (size_t i = 1; i < cu_seq_len_vec.size(); ++i) {
    CHECK_GE(cu_seq_len_vec[i], cu_seq_len_vec[i - 1]);
    max_seq_len =
        std::max(max_seq_len, cu_seq_len_vec[i] - cu_seq_len_vec[i - 1]);
  }
  return max_seq_len;
}

namespace {

#if defined(USE_CUDA) || defined(USE_DCU)
// Pure PyTorch scaled dot-product attention for Qwen2 vision.
void compute_qwen2_vision_attention_torch(
    torch::Tensor& q,
    torch::Tensor& k,
    torch::Tensor& v,
    torch::Tensor& output,
    const std::vector<int32_t>& cu_seq_len_vec,
    float scale) {
  if (cu_seq_len_vec.size() < 2) return;

  const int32_t num_seqs = static_cast<int32_t>(cu_seq_len_vec.size()) - 1;
  for (int32_t i = 0; i < num_seqs; ++i) {
    int32_t start = cu_seq_len_vec[i];
    int32_t end = cu_seq_len_vec[i + 1];
    int32_t len = end - start;
    if (len <= 0) continue;

    // Current sequence: [len, heads, head_dim]
    auto q_i = q.slice(/*dim=*/0, /*start=*/start, /*end=*/end);
    auto k_i = k.slice(0, start, end);
    auto v_i = v.slice(0, start, end);

    // [len, H, D] -> [H, len, D]
    q_i = q_i.permute({1, 0, 2});
    k_i = k_i.permute({1, 0, 2});
    v_i = v_i.permute({1, 0, 2});

    // Scaled dot-product attention per head.
    auto q_scaled = q_i * scale;
    auto k_t = k_i.transpose(1, 2);              // [H, D, len]
    auto scores = torch::matmul(q_scaled, k_t);  // [H, len, len]
    auto attn = torch::softmax(scores, /*dim=*/-1);
    auto out_i = torch::matmul(attn, v_i);  // [H, len, D]

    // Back to [len, H, D] and write into output.
    out_i = out_i.permute({1, 0, 2}).contiguous();
    output.slice(/*dim=*/0, /*start=*/start, /*end=*/end).copy_(out_i);
  }
}
#endif  // defined(USE_CUDA) || defined(USE_DCU)

}  // namespace

torch::Tensor Qwen2VisionAttentionImpl::forward(
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

  // Apply rotary position embedding to both q and k in a single call.
  // NOTE: Do NOT call apply_rotary twice; the first call already handles both
  // q and k. A second call would incorrectly apply RoPE to k a second time.
  xllm::kernel::RotaryParams rotary_params;
  rotary_params.q = q;
  rotary_params.k = k;
  rotary_params.sin = m_sin_pos;
  rotary_params.cos = m_cos_pos;
  rotary_params.interleaved = false;
  rotary_params.discrete = false;
#if defined(USE_CUDA) || defined(USE_DCU) || defined(USE_MUSA)
  rotary_params.position_ids =
      torch::arange(
          q.size(0),
          torch::TensorOptions().dtype(torch::kInt64).device(q.device()))
          .contiguous();
  rotary_params.cu_query_lens = std::nullopt;
#else
  rotary_params.cu_query_lens = cu_seq_len;
#endif
  rotary_params.max_query_len = max_seqlen;
  xllm::kernel::apply_rotary(rotary_params);
  q = rotary_params.q.reshape(
      {B * S, num_attention_heads_per_partition_, head_dim});
  k = rotary_params.k.reshape(
      {B * S, num_attention_heads_per_partition_, head_dim});

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
#elif defined(USE_CUDA) || defined(USE_DCU)
  // CUDA/DCU path: use a pure PyTorch vision attention implementation that
  // matches Transformers Qwen2.5-VL VisionAttention. FlashInfer's precompiled
  // AOT kernels in this project do not support head_dim=80, so we intentionally
  // do not call FlashInfer here and run attention entirely in PyTorch instead.
  compute_qwen2_vision_attention_torch(q, k, v, output, cu_seq_len_vec, scale_);
#endif

  // context_layer = rearrange(output, "(b s) h d -> s b (h d)", b=batch_size)
  output = output.view({B, S, -1});
  // [B, S, ...] -> [S, B, ...]
  output = output.transpose(0, 1).reshape({-1, output.size(-1)});
  // 6. output projection
  return proj_->forward(output);
}

void Qwen2VisionAttentionImpl::load_state_dict(const StateDict& state_dict) {
  qkv_proj_->load_state_dict(state_dict.get_dict_with_prefix("qkv."));
  proj_->load_state_dict(state_dict.get_dict_with_prefix("proj."));
}

}  // namespace layer
}  // namespace xllm

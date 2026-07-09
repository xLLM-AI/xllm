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

#include "rotary_embedding.h"

#include <algorithm>

#include "kernels/ops_api.h"
#include "platform/device.h"
#include "platform/platform.h"

namespace xllm {
namespace layer {

RotaryEmbeddingImpl::RotaryEmbeddingImpl(const ModelContext& context) {
  LOG(FATAL) << "Not implement currently.";
}

RotaryEmbeddingImpl::RotaryEmbeddingImpl(int64_t rotary_dim,
                                         int64_t max_position_embeddings,
                                         int64_t rope_theta,
                                         bool interleaved,
                                         const torch::TensorOptions& options)
    : interleaved_(interleaved) {
  auto inv_freq = rotary::compute_inv_freq(rotary_dim, rope_theta, options);
  const auto cos_sin = rotary::compute_cos_sin_cache(
      rotary_dim, max_position_embeddings, interleaved, inv_freq, options);
  cos_sin_cache_ = register_buffer("cos_sin_cache", cos_sin);

  auto cos_sin_vec = cos_sin_cache_.chunk(2, /*dim=*/-1);
  cos_ = cos_sin_vec[0].view({-1, rotary_dim});
  sin_ = cos_sin_vec[1].view({-1, rotary_dim});

  // Pre-compute [cos_half, sin_half] format used by the CUDA/ILU/MUSA kernels.
  if (Platform::is_cuda() || Platform::is_ilu() || Platform::is_musa() ||
      Platform::is_dcu()) {
    auto chunks = cos_sin_cache_.chunk(4, -1);
    precomputed_cos_sin_cache_ =
        torch::cat({chunks[0], chunks[2]}, -1).contiguous();
  }
}

void RotaryEmbeddingImpl::forward(torch::Tensor& q,
                                  torch::Tensor& k,
                                  const torch::Tensor& positions,
                                  const torch::Tensor& cu_query_lens,
                                  int64_t max_query_len,
                                  bool is_prompt) {
  bool discrete;
  std::optional<torch::Tensor> position_ids;
  if (is_prompt) {
    discrete = false;
    if (Platform::is_cuda() || Platform::is_npu() || Platform::is_ilu() ||
        Platform::is_musa() || Platform::is_dcu()) {
      position_ids = positions;
    }
  } else {
    discrete = true;
    position_ids = positions;
  }

  xllm::kernel::RotaryParams rotary_params;
  rotary_params.q = q;
  rotary_params.k = k;
  rotary_params.sin = sin_;
  rotary_params.cos = cos_;
  rotary_params.cos_sin = cos_sin_cache_;
  rotary_params.precomputed_cos_sin = precomputed_cos_sin_cache_;
  rotary_params.position_ids = position_ids;
  rotary_params.cu_query_lens = cu_query_lens;
  rotary_params.interleaved = interleaved_;
  rotary_params.discrete = discrete;
  rotary_params.max_query_len = max_query_len;
  xllm::kernel::apply_rotary(rotary_params);

  q = rotary_params.q;
  k = rotary_params.k;
}

// Single tensor forward for MLA architecture
void RotaryEmbeddingImpl::forward(torch::Tensor& input,
                                  const torch::Tensor& positions,
                                  const torch::Tensor& cu_query_lens,
                                  int64_t max_query_len,
                                  bool is_prompt) {
  bool discrete;
  std::optional<torch::Tensor> position_ids;
  if (is_prompt) {
    discrete = false;
    if (Platform::is_cuda() || Platform::is_npu() || Platform::is_ilu() ||
        Platform::is_dcu()) {
      position_ids = positions;
    }
  } else {
    discrete = true;
    position_ids = positions;
  }

  xllm::kernel::RotaryParams rotary_params;
  rotary_params.q = input;
  rotary_params.sin = sin_;
  rotary_params.cos = cos_;
  rotary_params.cos_sin = cos_sin_cache_;
  rotary_params.position_ids = position_ids;
  rotary_params.cu_query_lens = cu_query_lens;
  rotary_params.interleaved = interleaved_;
  rotary_params.discrete = discrete;
  rotary_params.max_query_len = max_query_len;
  xllm::kernel::apply_rotary(rotary_params);

  input = rotary_params.q;
}

MRotaryEmbeddingImpl::MRotaryEmbeddingImpl(
    int64_t rotary_dim,
    int64_t max_position_embeddings,
    int64_t rope_theta,
    bool interleaved,
    const std::vector<int64_t>& rope_scaling_mrope_section,
    const torch::TensorOptions& options)
    : RotaryEmbeddingImpl(rotary_dim,
                          max_position_embeddings,
                          rope_theta,
                          interleaved,
                          options),
      mrope_section_(rope_scaling_mrope_section) {
  mrope_cu_seq_lens_ = torch::zeros(2, torch::kInt32).to(options.device());
}

void MRotaryEmbeddingImpl::forward(torch::Tensor& q,
                                   torch::Tensor& k,
                                   const torch::Tensor& positions,
                                   const AttentionMetadata& attn_metadata) {
  if (attn_metadata.dsa_metadata && attn_metadata.dsa_metadata->cos.defined() &&
      attn_metadata.dsa_metadata->sin.defined()) {
    torch::Tensor dsa_q = q;
    torch::Tensor dsa_k = k;
    if (q.dim() == 2) {
      dsa_q = q.unsqueeze(1);
    }
    if (k.dim() == 2) {
      dsa_k = k.unsqueeze(1);
    }

    xllm::kernel::RotaryParams rotary_params;
    rotary_params.q = dsa_q;
    rotary_params.k = dsa_k;
    rotary_params.sin = attn_metadata.dsa_metadata->sin;
    rotary_params.cos = attn_metadata.dsa_metadata->cos;
    rotary_params.cos_sin = get_cos_sin_cache();
    rotary_params.position_ids = std::nullopt;
    rotary_params.cu_query_lens = attn_metadata.q_cu_seq_lens;
    rotary_params.interleaved = interleaved_;
    rotary_params.discrete = false;
    rotary_params.max_query_len =
        std::max<int64_t>(attn_metadata.max_query_len, 1);
    xllm::kernel::apply_rotary(rotary_params);

    q = (q.dim() == 2) ? rotary_params.q.squeeze(1) : rotary_params.q;
    k = (k.dim() == 2) ? rotary_params.k.squeeze(1) : rotary_params.k;
    return;
  }

  bool only_prefill =
      (attn_metadata.is_prefill || attn_metadata.is_chunked_prefill);
  if (!only_prefill || mrope_section_.empty()) {
    torch::Tensor position_ids = positions;
    if (positions.dim() == 2) {
      position_ids = positions[0];
    }
    return RotaryEmbeddingImpl::forward(q,
                                        k,
                                        position_ids,
                                        attn_metadata.q_cu_seq_lens,
                                        attn_metadata.max_query_len,
                                        attn_metadata.is_prefill);
  }

  int64_t num_tokens = positions.size(-1);
  mrope_cu_seq_lens_[1] = num_tokens;
  CHECK(attn_metadata.mrope_cos.defined() && attn_metadata.mrope_sin.defined());
  xllm::kernel::RotaryParams rotary_params;
  rotary_params.q = q;
  rotary_params.k = k;
  rotary_params.sin = attn_metadata.mrope_sin;
  rotary_params.cos = attn_metadata.mrope_cos;
  rotary_params.cos_sin = cos_sin_cache_;
  rotary_params.precomputed_cos_sin = precomputed_cos_sin_cache_;
  rotary_params.position_ids = std::nullopt;
  rotary_params.cu_query_lens = mrope_cu_seq_lens_;
  rotary_params.interleaved = interleaved_;
  rotary_params.discrete = false;
  rotary_params.max_query_len = num_tokens;
  xllm::kernel::apply_rotary(rotary_params);

  q = rotary_params.q;
  k = rotary_params.k;
}

DeepseekScalingRotaryEmbeddingImpl::DeepseekScalingRotaryEmbeddingImpl(
    int64_t head_size,
    int64_t rotary_dim,
    int64_t max_position_embeddings,
    int64_t rope_scaling_original_max_position_embeddings,
    int64_t rope_theta,
    bool interleaved,
    float scaling_factor,
    float extrapolation_factor,
    float attn_factor,
    float beta_fast,
    float beta_slow,
    float mscale,
    float mscale_all_dim,
    const torch::TensorOptions& options)
    : head_size_(head_size),
      rotary_dim_(rotary_dim),
      interleaved_(interleaved) {
  auto inv_freq = rotary::apply_deepseek_yarn_rope_scaling(
      scaling_factor,
      extrapolation_factor,
      beta_fast,
      beta_slow,
      rotary_dim,
      rope_theta,
      rope_scaling_original_max_position_embeddings);
  const auto cos_sin = rotary::compute_cos_sin_cache(rotary_dim,
                                                     max_position_embeddings,
                                                     interleaved,
                                                     scaling_factor,
                                                     attn_factor,
                                                     mscale,
                                                     mscale_all_dim,
                                                     inv_freq,
                                                     options);
  cos_sin_cache_ = register_buffer("cos_sin_cache", cos_sin);

  auto cos_sin_vec = cos_sin_cache_.chunk(2, /*dim=*/-1);
  cos_ = cos_sin_vec[0].view({-1, rotary_dim});
  sin_ = cos_sin_vec[1].view({-1, rotary_dim});

  // Pre-compute [cos_half, sin_half] format used by the CUDA/ILU/MUSA kernels.
  if (Platform::is_cuda() || Platform::is_ilu() || Platform::is_musa() ||
      Platform::is_dcu()) {
    auto chunks = cos_sin_cache_.chunk(4, -1);
    precomputed_cos_sin_cache_ =
        torch::cat({chunks[0], chunks[2]}, -1).contiguous();
  }
}

void DeepseekScalingRotaryEmbeddingImpl::forward(
    torch::Tensor& input,
    const torch::Tensor& positions,
    const torch::Tensor& cu_query_lens,
    int64_t max_query_len,
    bool is_prompt) {
  const int32_t dim = -1;
  bool discrete;
  std::optional<torch::Tensor> position_ids;
  if (is_prompt) {
    discrete = false;
    position_ids = std::nullopt;
    if (Platform::is_dcu()) {
      position_ids = positions;
    }
  } else {
    discrete = true;
    position_ids = positions;
    max_query_len = 1;
  }
  auto input_rot = input.slice(dim, 0, rotary_dim_);
  torch::Tensor input_pass;
  if (rotary_dim_ < head_size_) {
    input_pass = input.slice(dim, rotary_dim_, head_size_);
  }

  xllm::kernel::RotaryParams rotary_params;
  rotary_params.q = input_rot;
  rotary_params.sin = sin_;
  rotary_params.cos = cos_;
  rotary_params.cos_sin = cos_sin_cache_;
  rotary_params.precomputed_cos_sin = precomputed_cos_sin_cache_;
  rotary_params.position_ids = position_ids;
  rotary_params.cu_query_lens = cu_query_lens;
  rotary_params.interleaved = interleaved_;
  rotary_params.discrete = discrete;
  rotary_params.max_query_len = max_query_len;
  xllm::kernel::apply_rotary(rotary_params);
  input_rot = rotary_params.q;

  if (rotary_dim_ < head_size_) {
    input = torch::cat({input_rot, input_pass}, dim);
  } else {
    input = input_rot;
  }
}

// Factory function: creates the appropriate RoPE type based on model args
std::shared_ptr<RotaryEmbeddingBase> create_mla_rotary_embedding(
    const ModelArgs& args,
    int64_t rotary_dim,
    int64_t max_position_embeddings,
    bool interleaved,
    const torch::TensorOptions& options) {
  if (args.rope_scaling_rope_type() == "deepseek_yarn") {
    return std::make_shared<DeepseekScalingRotaryEmbeddingImpl>(
        rotary_dim,  // head_size (same as rotary_dim for MLA)
        rotary_dim,
        max_position_embeddings,
        args.rope_scaling_original_max_position_embeddings(),
        args.rope_theta(),
        interleaved,
        args.rope_scaling_factor(),
        args.rope_extrapolation_factor(),
        args.rope_scaling_attn_factor(),
        args.rope_scaling_beta_fast(),
        args.rope_scaling_beta_slow(),
        args.rope_scaling_mscale(),
        args.rope_scaling_mscale_all_dim(),
        options);
  } else {
    // default rope type
    return std::make_shared<RotaryEmbeddingImpl>(rotary_dim,
                                                 max_position_embeddings,
                                                 args.rope_theta(),
                                                 interleaved,
                                                 options);
  }
}

}  // namespace layer
}  // namespace xllm

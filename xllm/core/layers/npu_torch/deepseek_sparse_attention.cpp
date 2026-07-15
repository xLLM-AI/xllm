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
#include "deepseek_sparse_attention.h"

#include <glog/logging.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <optional>
#include <tuple>
#include <vector>

#include "kernels/ops_api.h"
#include "xllm/core/kernels/npu/xllm_ops/xllm_ops_api.h"

DECLARE_bool(enable_chunked_prefill);
namespace xllm {
namespace layer {
namespace {

struct Dsv4PreprocessOutputs {
  torch::Tensor qr;
  std::optional<torch::Tensor> qr_pertoken_scale;
  torch::Tensor q;
  torch::Tensor kv;
};

torch::Tensor w8a8_dynamic_linear_forward(
    const torch::Tensor& quantized_input,
    const torch::Tensor& weight,
    const torch::Tensor& weight_scale,
    const torch::Tensor& pertoken_scale,
    const std::optional<torch::Tensor>& bias,
    at::ScalarType output_dtype) {
  xllm::kernel::QuantMatmulParams params;
  params.x1 = quantized_input;
  params.x2 = weight;
  params.transpose2 = true;
  params.scale = weight_scale;
  params.pertoken_scale = pertoken_scale;
  params.bias = bias;
  params.output_dtype = output_dtype;
  return xllm::kernel::quant_matmul(params);
}

struct DsaCacheMapping {
  int64_t cmp_cache_idx = -1;
  int64_t index_cache_idx = -1;
  int64_t indexer_scale_cache_idx = -1;
  int64_t ori_cache_idx = -1;
  int64_t kv_state_cache_idx = -1;
  int64_t score_state_cache_idx = -1;
  int64_t index_kv_state_cache_idx = -1;
  int64_t index_score_state_cache_idx = -1;
};

std::optional<torch::Tensor> as_optional(const torch::Tensor& tensor) {
  if (tensor.defined() && tensor.numel() > 0) {
    return std::optional<torch::Tensor>(tensor);
  }
  return std::nullopt;
}

torch::Tensor get_layer_cache_tensor(
    const std::vector<std::vector<torch::Tensor>>& layer_tensors,
    int32_t layer_id,
    int64_t cache_idx) {
  if (layer_id < 0 || layer_id >= static_cast<int32_t>(layer_tensors.size()) ||
      cache_idx < 0 ||
      cache_idx >= static_cast<int64_t>(layer_tensors[layer_id].size())) {
    return torch::Tensor();
  }
  return layer_tensors[layer_id][cache_idx];
}

DsaCacheMapping resolve_cache_mapping(const DSAMetadata& attn_metadata,
                                      int64_t compress_ratio) {
  DsaCacheMapping mapping;
  if (!attn_metadata.caches_info || attn_metadata.layer_id < 0 ||
      attn_metadata.layer_id >=
          static_cast<int32_t>(attn_metadata.caches_info->size())) {
    return mapping;
  }

  const auto& layer_caches =
      (*(attn_metadata.caches_info))[attn_metadata.layer_id];

  std::vector<int64_t> token_ratio_indices;
  std::vector<int64_t> swa_indices;
  token_ratio_indices.reserve(layer_caches.size());
  swa_indices.reserve(layer_caches.size());

  for (int64_t cache_idx = 0;
       cache_idx < static_cast<int64_t>(layer_caches.size());
       ++cache_idx) {
    const auto& cache_info = layer_caches[cache_idx];
    if (cache_info.type == DSACacheType::TOKEN &&
        cache_info.ratio == static_cast<int32_t>(compress_ratio)) {
      token_ratio_indices.push_back(cache_idx);
    }
    if (cache_info.type == DSACacheType::SLIDING_WINDOW) {
      swa_indices.push_back(cache_idx);
    }
  }

  if (!token_ratio_indices.empty() && compress_ratio > 1) {
    mapping.cmp_cache_idx = token_ratio_indices[0];
  }
  if (token_ratio_indices.size() > 1) {
    mapping.index_cache_idx = token_ratio_indices[1];
  }
  if (token_ratio_indices.size() > 2) {
    mapping.indexer_scale_cache_idx = token_ratio_indices[2];
  }

  if (!swa_indices.empty()) {
    mapping.ori_cache_idx = swa_indices[0];
  }
  if (swa_indices.size() > 1) {
    mapping.kv_state_cache_idx = swa_indices[1];
  }
  if (swa_indices.size() > 2) {
    mapping.score_state_cache_idx = swa_indices[2];
  }
  if (swa_indices.size() > 3) {
    mapping.index_kv_state_cache_idx = swa_indices[3];
  }
  if (swa_indices.size() > 4) {
    mapping.index_score_state_cache_idx = swa_indices[4];
  }

  return mapping;
}

void apply_partial_rope(torch::Tensor& input,
                        int64_t rope_start_dim,
                        int64_t rope_head_dim,
                        const torch::Tensor& cos,
                        const torch::Tensor& sin,
                        bool inverse = false) {
  if (!input.defined() || !cos.defined() || !sin.defined() ||
      rope_head_dim <= 0 || rope_start_dim < 0) {
    return;
  }

  const int64_t input_last_dim = input.size(input.dim() - 1);
  if (input_last_dim < rope_start_dim + rope_head_dim) {
    return;
  }

  auto sin_cache = inverse ? -sin : sin;
  auto cos_cache = cos;
  CHECK(input.dim() == 2 || input.dim() == 3)
      << "apply_partial_rope only supports input dim 2/3, got: " << input.dim();
  CHECK(cos_cache.dim() == 2 && sin_cache.dim() == 2)
      << "apply_partial_rope expects cos/sin dim=2, got cos dim "
      << cos_cache.dim() << ", sin dim " << sin_cache.dim();
  CHECK(cos_cache.size(0) == input.size(0) &&
        sin_cache.size(0) == input.size(0))
      << "apply_partial_rope expects cos/sin batch == input.size(0), got cos "
      << cos_cache.size(0) << ", sin " << sin_cache.size(0) << ", input "
      << input.size(0);
  CHECK(cos_cache.size(1) == rope_head_dim &&
        sin_cache.size(1) == rope_head_dim)
      << "apply_partial_rope expects cos/sin last dim == rope_head_dim("
      << rope_head_dim << "), got cos " << cos_cache.size(1) << ", sin "
      << sin_cache.size(1);

  auto cos_4d = cos_cache.view({cos_cache.size(0), 1, 1, rope_head_dim});
  auto sin_4d = sin_cache.view({sin_cache.size(0), 1, 1, rope_head_dim});
  auto input_4d =
      (input.dim() == 3) ? input.unsqueeze(2) : input.unsqueeze(1).unsqueeze(1);
  xllm::kernel::NpuInplacePartialRotaryMulParams rope_params;
  rope_params.x = input_4d;
  rope_params.r1 = cos_4d;
  rope_params.r2 = sin_4d;
  rope_params.rotary_mode = "interleave";
  rope_params.partial_slice = {rope_start_dim, rope_start_dim + rope_head_dim};
  xllm::kernel::npu_inplace_partial_rotary_mul(rope_params);
  input =
      (input.dim() == 3) ? input_4d.squeeze(2) : input_4d.squeeze(1).squeeze(1);
}

void scatter_by_slot(torch::Tensor& cache,
                     const torch::Tensor& slot_mapping,
                     const torch::Tensor& value) {
  if (!cache.defined() || !slot_mapping.defined() || !value.defined()) {
    return;
  }
  if (slot_mapping.numel() == 0 || value.numel() == 0) {
    return;
  }

  auto value_2d = value.reshape({-1, value.size(value.dim() - 1)});
  auto cache_2d = cache.view({-1, value_2d.size(1)});

  auto slots = slot_mapping.reshape({-1}).to(torch::kLong).to(cache.device());
  const int64_t update_rows =
      std::min<int64_t>(slots.size(0), value_2d.size(0));
  if (update_rows <= 0) {
    return;
  }

  auto slots_slice = slots.slice(/*dim=*/0, /*start=*/0, /*end=*/update_rows);
  auto value_slice =
      value_2d.slice(/*dim=*/0, /*start=*/0, /*end=*/update_rows);

  const int64_t cache_rows = cache_2d.size(0);
  CHECK_GT(cache_rows, 0) << "scatter_by_slot requires cache rows > 0, cache "
                          << cache.sizes();

  if (!cache.device().is_cpu()) {
    auto safe_slots = slots_slice.clamp_min(0);
    auto valid_mask = slots_slice.ge(0).unsqueeze(1);
    auto old_values = cache_2d.index_select(/*dim=*/0, safe_slots);
    auto safe_values = torch::where(valid_mask, value_slice, old_values);
    xllm::kernel::npu::scatter_nd_update(
        cache_2d, safe_slots.reshape({-1, 1}), safe_values);
    return;
  }

  auto valid_mask = slots_slice.ge(0);
  auto valid_slots = slots_slice.index({valid_mask});
  if (valid_slots.numel() == 0) {
    return;
  }
  auto valid_values = value_slice.index({valid_mask});

  const int64_t max_slot = valid_slots.max().item<int64_t>();
  CHECK_LT(max_slot, cache_rows)
      << "scatter_by_slot slot index out of range: max_slot=" << max_slot
      << ", cache_rows=" << cache_rows
      << ", slot_mapping_shape=" << slot_mapping.sizes()
      << ", value_shape=" << value.sizes()
      << ", value_rows=" << value_2d.size(0) << ", update_rows=" << update_rows
      << ", cache_shape=" << cache.sizes();
  cache_2d.index_copy_(/*dim=*/0, valid_slots, valid_values);
}

// Pack prefill TND KV [total_tokens, n, d] into temporary PA_ND blocks
// [num_blocks + 1, block_size, n, d]. This mirrors vllm-ascend's
// pad_to_blocks path and lets sparse_attn_sharedkv read prefill KV through a
// block table without depending on the persistent SWA ring cache.
std::tuple<torch::Tensor, torch::Tensor> build_prefill_pa_nd_kv(
    const torch::Tensor& kv,
    const torch::Tensor& cu_seqlens_q,
    const torch::Tensor& block_table_hint,
    int64_t block_size) {
  if (!kv.defined() || !cu_seqlens_q.defined() || cu_seqlens_q.numel() <= 1 ||
      block_size <= 0) {
    return {torch::Tensor(), torch::Tensor()};
  }

  const int64_t batch_size = cu_seqlens_q.numel() - 1;
  const auto cu_cpu = cu_seqlens_q.to(torch::kCPU).to(torch::kInt64);
  std::vector<int64_t> q_starts(batch_size + 1);
  auto cu_acc = cu_cpu.accessor<int64_t, 1>();
  int64_t total_blocks = 0;
  int64_t max_blocks_per_req = 0;
  // cu_seqlens_q describes the packed token ranges for each request. Convert
  // those ranges into the number of PA_ND blocks each request needs.
  for (int64_t i = 0; i <= batch_size; ++i) {
    q_starts[i] = cu_acc[i];
    if (i > 0) {
      const int64_t q_len = q_starts[i] - q_starts[i - 1];
      const int64_t blocks = (q_len + block_size - 1) / block_size;
      total_blocks += blocks;
      max_blocks_per_req = std::max(max_blocks_per_req, blocks);
    }
  }
  if (total_blocks <= 0) {
    return {torch::Tensor(), torch::Tensor()};
  }

  const int64_t table_cols =
      std::max<int64_t>(block_table_hint.defined() && block_table_hint.dim() > 1
                            ? block_table_hint.size(1)
                            : 0,
                        max_blocks_per_req);
  std::vector<int32_t> table_data(static_cast<size_t>(batch_size * table_cols),
                                  0);

  auto packed_kv = torch::zeros(
      {total_blocks + 1, block_size, kv.size(1), kv.size(2)}, kv.options());

  // Block 0 stays zero-filled as the padding block; real requests use 1-based
  // block ids, matching the block tables consumed by sparse_attn_sharedkv.
  int64_t next_block = 1;
  for (int64_t req = 0; req < batch_size; ++req) {
    const int64_t q_start = q_starts[req];
    const int64_t q_len = q_starts[req + 1] - q_start;
    const int64_t blocks = (q_len + block_size - 1) / block_size;
    if (q_len <= 0 || blocks <= 0) {
      continue;
    }
    for (int64_t j = 0; j < blocks; ++j) {
      table_data[static_cast<size_t>(req * table_cols + j)] =
          static_cast<int32_t>(next_block + j);
    }
    // Copy this request's contiguous prefill KV into its temporary PA_ND block
    // range. The zero-initialized tail of the last block is padding.
    auto target = packed_kv
                      .slice(/*dim=*/0,
                             /*start=*/next_block,
                             /*end=*/next_block + blocks)
                      .view({blocks * block_size, kv.size(1), kv.size(2)});
    target.narrow(/*dim=*/0, /*start=*/0, /*length=*/q_len)
        .copy_(kv.narrow(/*dim=*/0, /*start=*/q_start, /*length=*/q_len));
    next_block += blocks;
  }

  auto table_cpu =
      torch::tensor(table_data, torch::TensorOptions().dtype(torch::kInt32))
          .view({batch_size, table_cols});
  auto table = table_cpu.to(
      block_table_hint.defined() ? block_table_hint.device() : kv.device());
  return {packed_kv, table};
}

Dsv4PreprocessOutputs run_dsv4_preprocess_fallback(
    ReplicatedLinear& q_a_proj,
    RMSNorm& q_layernorm,
    ColumnParallelLinear& q_b_proj,
    ReplicatedLinear& kv_proj,
    RMSNorm& kv_layernorm,
    const torch::Tensor& hidden_states,
    int64_t n_local_heads,
    int64_t head_dim,
    int64_t qk_head_dim,
    int64_t nope_head_dim,
    int64_t rope_head_dim,
    double eps,
    const torch::Tensor& q_rms_gamma,
    const torch::Tensor& cos,
    const torch::Tensor& sin) {
  Dsv4PreprocessOutputs outputs;

  auto q_down = q_a_proj->forward(hidden_states);
  if (q_b_proj->uses_w8a8_dynamic_quant()) {
    xllm::kernel::RmsNormDynamicQuantParams rms_quant_params;
    rms_quant_params.input = q_down;
    rms_quant_params.weight = q_layernorm->weight();
    rms_quant_params.eps = eps;
    torch::Tensor q_pertoken_scale;
    std::tie(outputs.qr, q_pertoken_scale) =
        xllm::kernel::rms_norm_dynamic_quant(rms_quant_params);
    outputs.qr_pertoken_scale = q_pertoken_scale;
    outputs.q =
        w8a8_dynamic_linear_forward(outputs.qr,
                                    q_b_proj->weight(),
                                    q_b_proj->w8a8_dynamic_weight_scale(),
                                    q_pertoken_scale,
                                    q_b_proj->bias(),
                                    hidden_states.scalar_type())
            .view({-1, n_local_heads, head_dim});
  } else {
    outputs.qr = std::get<0>(q_layernorm->forward(q_down));
    outputs.q =
        q_b_proj->forward(outputs.qr).view({-1, n_local_heads, head_dim});
  }

  xllm::kernel::FusedLayerNormParams q_rmsnorm_params;
  q_rmsnorm_params.input = outputs.q;
  q_rmsnorm_params.weight = q_rms_gamma;
  q_rmsnorm_params.mode = "rmsnorm";
  q_rmsnorm_params.eps = eps;
  xllm::kernel::fused_layernorm(q_rmsnorm_params);
  outputs.q = q_rmsnorm_params.output;

  auto kv_down = kv_proj->forward(hidden_states);
  outputs.kv = std::get<0>(kv_layernorm->forward(kv_down));
  outputs.kv = outputs.kv.view({-1, 1, qk_head_dim});

  apply_partial_rope(outputs.q, nope_head_dim, rope_head_dim, cos, sin);
  apply_partial_rope(outputs.kv, nope_head_dim, rope_head_dim, cos, sin);

  return outputs;
}

AttentionMetadata build_indexer_attention_metadata(
    const DSAMetadata& attn_metadata,
    const torch::Tensor& block_table,
    const torch::Tensor& slot_mapping,
    bool is_prefill,
    int64_t max_query_len,
    int64_t max_seq_len) {
  AttentionMetadata metadata;
  metadata.is_prefill = is_prefill;
  metadata.is_chunked_prefill = false;
  metadata.is_dummy = false;
  metadata.is_causal = true;

  metadata.block_table = block_table;
  metadata.slot_mapping = slot_mapping;

  metadata.q_cu_seq_lens = attn_metadata.actual_seq_lengths_query;
  metadata.kv_seq_lens = attn_metadata.actual_seq_lengths_kv;
  if (!metadata.kv_seq_lens.defined()) {
    metadata.kv_seq_lens = attn_metadata.seq_lens;
  }

  if (attn_metadata.actual_seq_lengths_query.defined() &&
      attn_metadata.actual_seq_lengths_query.dim() > 0 &&
      attn_metadata.actual_seq_lengths_query.size(0) > 1) {
    metadata.q_seq_lens = attn_metadata.actual_seq_lengths_query.slice(
        /*dim=*/0,
        /*start=*/1,
        /*end=*/attn_metadata.actual_seq_lengths_query.size(0));
  } else {
    metadata.q_seq_lens = attn_metadata.seq_lens_q;
  }

  if (attn_metadata.kv_cu_seq_lens.defined() &&
      attn_metadata.kv_cu_seq_lens.numel() > 0) {
    // Reuse the kv cumulative sequence lengths computed once per forward in
    // DSAMetadataBuilder, avoiding a redundant host-side cumsum on every DSA
    // layer.
    metadata.kv_cu_seq_lens = attn_metadata.kv_cu_seq_lens;
  } else if (metadata.kv_seq_lens.defined()) {
    auto kv_seq_lens = metadata.kv_seq_lens.to(torch::kInt32);
    auto kv_cumsum = torch::cumsum(kv_seq_lens, /*dim=*/0);
    metadata.kv_cu_seq_lens =
        torch::cat({torch::zeros({1}, kv_seq_lens.options()), kv_cumsum});
  }

  metadata.max_query_len = max_query_len;
  metadata.max_seq_len = max_seq_len;

  metadata.dsa_metadata = std::make_shared<DSAMetadata>(attn_metadata);

  return metadata;
}

}  // namespace

DSAttentionImpl::DSAttentionImpl(const ModelContext& context, int32_t layer_id)
    : DSAttentionImpl(context.get_model_args(),
                      context.get_quant_args(),
                      context.get_parallel_args(),
                      context.get_tensor_options(),
                      layer_id) {}

DSAttentionImpl::DSAttentionImpl(const ModelArgs& args,
                                 const QuantArgs& quant_args,
                                 const ParallelArgs& parallel_args,
                                 const torch::TensorOptions& options,
                                 int32_t layer_id)
    : num_heads_(args.n_heads()),
      head_size_(args.head_dim()),
      head_dim_(args.head_dim()),
      n_kv_heads_(args.n_kv_heads().value()),
      sliding_window_(-1),
      q_lora_rank_(args.q_lora_rank()),
      o_lora_rank_(args.o_lora_rank()),
      o_groups_(args.o_groups()),
      rope_head_dim_(args.rope_head_dim()),
      window_size_(args.window_size()),
      compress_ratio_(1.0),
      index_n_heads_(args.index_n_heads()),
      index_head_dim_(args.index_head_dim()),
      index_topk_(args.index_topk()),
      eps_(args.rms_norm_eps()) {
  const auto& compress_ratios = args.compress_ratios();
  CHECK(!compress_ratios.empty())
      << "DSAttention requires non-empty compress_ratios for DeepSeek V4";
  CHECK_GE(layer_id, 0) << "DSAttention requires valid layer_id, got "
                        << layer_id;
  CHECK_LT(layer_id, static_cast<int32_t>(compress_ratios.size()))
      << "DSAttention layer_id " << layer_id << " exceeds compress_ratios size "
      << compress_ratios.size();
  int64_t compress_ratio = compress_ratios[static_cast<size_t>(layer_id)];

  CHECK(compress_ratio == 1 || compress_ratio == 4 || compress_ratio == 128)
      << "DSAttention unsupported compress_ratio " << compress_ratio
      << " at layer " << layer_id;

  compress_ratio_ = static_cast<double>(compress_ratio);

  softmax_scale_ = std::pow(head_dim_, static_cast<double>(-0.5));
  scale_ = static_cast<float>(softmax_scale_);
  nope_head_dim_ = head_dim_ - rope_head_dim_;
  qk_head_dim_ = nope_head_dim_ + rope_head_dim_;

  const int64_t tp_size = parallel_args.tp_group_->world_size();
  tp_rank_ = parallel_args.tp_group_->rank();
  tp_size_ = tp_size;
  int64_t hidden_size = args.hidden_size();
  int64_t num_heads = args.n_heads();

  CHECK_EQ(o_groups_ % tp_size, 0)
      << "o_groups must be divisible by tensor parallel size";
  CHECK_EQ(num_heads % tp_size, 0)
      << "num_heads must be divisible by tensor parallel size";
  n_local_heads_ = num_heads / tp_size;
  n_local_groups_ = o_groups_ / tp_size;

  attn_sink_ = register_parameter(
      "attn_sink",
      torch::zeros({n_local_heads_}, options.dtype(torch::kFloat32)),
      /*requires_grad=*/false);

  q_a_proj_ = register_module(
      "q_a_proj",
      ReplicatedLinear(
          hidden_size, q_lora_rank_, /*bias=*/false, quant_args, options));

  q_layernorm_ =
      register_module("q_a_layernorm", RMSNorm(q_lora_rank_, eps_, options));

  q_b_proj_ = register_module("q_b_proj",
                              ColumnParallelLinear(q_lora_rank_,
                                                   num_heads * head_dim_,
                                                   false,
                                                   false,
                                                   quant_args,
                                                   parallel_args.tp_group_,
                                                   options));

  kv_proj_ = register_module(
      "kv_proj",
      ReplicatedLinear(
          hidden_size, head_dim_, /*bias=*/false, quant_args, options));
  kv_layernorm_ =
      register_module("kv_layernorm", RMSNorm(head_dim_, eps_, options));

  if (compress_ratio_ > 1) {
    compressor_ =
        register_module("compressor",
                        Compressor(static_cast<int64_t>(compress_ratio_),
                                   head_dim_,
                                   rope_head_dim_,
                                   /*rot_mode=*/2,
                                   eps_,
                                   options));
  }

  if (compress_ratio_ == 4) {
    if (index_n_heads_ <= 0) {
      index_n_heads_ = num_heads_;
    }
    if (index_head_dim_ <= 0) {
      index_head_dim_ = head_dim_;
    }
    if (index_topk_ <= 0) {
      index_topk_ = 512;
    }

    if (index_n_heads_ > 0 && index_head_dim_ > 0 && index_topk_ > 0) {
      indexer_ = register_module(
          "indexer",
          DeepseekV4Indexer(hidden_size,
                            index_n_heads_,
                            index_head_dim_,
                            rope_head_dim_,
                            index_topk_,
                            q_lora_rank_,
                            static_cast<int64_t>(compress_ratio_),
                            eps_,
                            quant_args,
                            options));
    } else {
      LOG(FATAL) << "DSAttention indexer disabled due to invalid config: "
                 << "index_n_heads=" << index_n_heads_
                 << ", index_head_dim=" << index_head_dim_
                 << ", index_topk=" << index_topk_;
    }
  }

  q_rms_gamma_ = register_buffer("q_rms_gamma",
                                 torch::ones({head_dim_},
                                             torch::TensorOptions()
                                                 .dtype(options.dtype())
                                                 .device(options.device())));

  o_a_proj_ =
      register_module("o_a_proj",
                      ColumnParallelLinear(num_heads * head_dim_ / o_groups_,
                                           o_groups_ * o_lora_rank_,
                                           false,
                                           true,
                                           quant_args,
                                           parallel_args.tp_group_,
                                           options));

  o_b_proj_ = register_module("o_b_proj",
                              RowParallelLinear(o_groups_ * o_lora_rank_,
                                                hidden_size,
                                                false,
                                                true,
                                                /*reduce=*/true,
                                                quant_args,
                                                parallel_args.tp_group_,
                                                options));
}

std::tuple<torch::Tensor, std::optional<torch::Tensor>>
DSAttentionImpl::forward(const DSAMetadata& attn_metadata,
                         torch::Tensor& hidden_states,
                         KVCache& kv_cache,
                         KVState& kv_state,
                         bool is_prefill,
                         bool is_chunked_prefill,
                         const std::tuple<torch::Tensor,
                                          torch::Tensor,
                                          torch::Tensor,
                                          torch::Tensor>& compress_metadata) {
  auto [c1_metadata, c4_metadata, c128_metadata, qli_metadata] =
      compress_metadata;
  auto cos = attn_metadata.cos;
  auto sin = attn_metadata.sin;
  Dsv4PreprocessOutputs preprocess_outputs =
      run_dsv4_preprocess_fallback(q_a_proj_,
                                   q_layernorm_,
                                   q_b_proj_,
                                   kv_proj_,
                                   kv_layernorm_,
                                   hidden_states,
                                   n_local_heads_,
                                   head_dim_,
                                   qk_head_dim_,
                                   nope_head_dim_,
                                   rope_head_dim_,
                                   eps_,
                                   q_rms_gamma_,
                                   cos,
                                   sin);
  auto qr = preprocess_outputs.qr;
  auto qr_pertoken_scale = preprocess_outputs.qr_pertoken_scale;
  auto q = preprocess_outputs.q;
  auto kv = preprocess_outputs.kv;

  // 4) resolve per-layer cache mapping
  const int64_t compress_ratio_i = static_cast<int64_t>(compress_ratio_);
  DsaCacheMapping mapping =
      resolve_cache_mapping(attn_metadata, compress_ratio_i);

  auto cmp_block_table = get_layer_cache_tensor(attn_metadata.block_tables,
                                                attn_metadata.layer_id,
                                                mapping.cmp_cache_idx);
  auto ori_block_table = get_layer_cache_tensor(attn_metadata.block_tables,
                                                attn_metadata.layer_id,
                                                mapping.ori_cache_idx);
  auto kv_block_table = get_layer_cache_tensor(attn_metadata.block_tables,
                                               attn_metadata.layer_id,
                                               mapping.kv_state_cache_idx);
  auto score_block_table =
      get_layer_cache_tensor(attn_metadata.block_tables,
                             attn_metadata.layer_id,
                             mapping.score_state_cache_idx);
  auto index_kv_block_table =
      get_layer_cache_tensor(attn_metadata.block_tables,
                             attn_metadata.layer_id,
                             mapping.index_kv_state_cache_idx);
  auto index_score_block_table =
      get_layer_cache_tensor(attn_metadata.block_tables,
                             attn_metadata.layer_id,
                             mapping.index_score_state_cache_idx);
  auto index_block_table = get_layer_cache_tensor(attn_metadata.block_tables,
                                                  attn_metadata.layer_id,
                                                  mapping.index_cache_idx);

  auto cmp_slot = get_layer_cache_tensor(attn_metadata.slot_mappings,
                                         attn_metadata.layer_id,
                                         mapping.cmp_cache_idx);
  auto ori_slot = get_layer_cache_tensor(attn_metadata.slot_mappings,
                                         attn_metadata.layer_id,
                                         mapping.ori_cache_idx);
  auto index_slot = get_layer_cache_tensor(attn_metadata.slot_mappings,
                                           attn_metadata.layer_id,
                                           mapping.index_cache_idx);

  auto ori_kv = std::get<0>(kv_state);
  if (!ori_kv.defined()) {
    ori_kv = kv_cache.get_swa_cache();
  }

  auto compressor_kv_state = std::get<1>(kv_state);
  if (!compressor_kv_state.defined()) {
    compressor_kv_state = kv_cache.get_compress_kv_state();
  }

  auto compressor_score_state = std::get<2>(kv_state);
  if (!compressor_score_state.defined()) {
    compressor_score_state = kv_cache.get_compress_score_state();
  }

  auto index_kv_state = std::get<3>(kv_state);
  if (!index_kv_state.defined()) {
    index_kv_state = kv_cache.get_compress_index_kv_state();
  }

  auto index_score_state = std::get<4>(kv_state);
  if (!index_score_state.defined()) {
    index_score_state = kv_cache.get_compress_index_score_state();
  }

  // 5) Prepare ori_kv for attention.
  // Full prefill can read a temporary PA_ND cache built from current KV.
  // Chunked prefill needs prefix KV, so it reads the persistent SWA cache.
  torch::Tensor ori_kv_for_attn;
  torch::Tensor ori_block_table_for_attn = ori_block_table;
  const std::string ori_kv_layout = "PA_ND";
  const bool use_prefill_attn = is_prefill || is_chunked_prefill;
  const bool use_temporary_prefill_kv = is_prefill && !is_chunked_prefill;
  if (use_temporary_prefill_kv) {
    const int64_t block_size =
        ori_kv.defined() && ori_kv.dim() > 1 ? ori_kv.size(1) : 128;
    std::tie(ori_kv_for_attn, ori_block_table_for_attn) =
        build_prefill_pa_nd_kv(kv,
                               attn_metadata.actual_seq_lengths_query,
                               ori_block_table,
                               block_size);
    CHECK(ori_kv_for_attn.defined())
        << "Failed to build PA_ND KV for DeepSeek V4 prefill attention.";
  } else {
    scatter_by_slot(ori_kv, ori_slot, kv);
    ori_kv_for_attn = ori_kv;
  }

  // 6) optional compressor for cmp cache
  // Token compressed cache is PA_ND for both prefill and decode.
  torch::Tensor cmp_kv_for_attn;
  auto cmp_kv = kv_cache.get_k_cache();
  if (compress_ratio_i > 1 && compressor_ && cmp_kv.defined() &&
      cmp_slot.defined() && compressor_kv_state.defined() &&
      compressor_score_state.defined()) {
    torch::Tensor compress_cos;
    torch::Tensor compress_sin;
    if (compress_ratio_i == 4) {
      compress_cos = attn_metadata.c4_cos;
      compress_sin = attn_metadata.c4_sin;
    } else if (compress_ratio_i == 128) {
      compress_cos = attn_metadata.c128_cos;
      compress_sin = attn_metadata.c128_sin;
    }

    std::tuple<torch::Tensor, torch::Tensor> compressor_states{
        compressor_kv_state, compressor_score_state};
    std::tuple<torch::Tensor, torch::Tensor> compressor_block_tables{
        kv_block_table, score_block_table};

    auto compressed_kv =
        compressor_->forward(attn_metadata,
                             hidden_states,
                             compressor_states,
                             compressor_block_tables,
                             compress_sin,
                             compress_cos,
                             attn_metadata.actual_seq_lengths_query);
    scatter_by_slot(cmp_kv, cmp_slot, compressed_kv);
    cmp_kv_for_attn = cmp_kv;
  }

  torch::Tensor compress_topk_idxs;
  if (compress_ratio_i == 4 && cmp_kv.defined()) {
    auto index_cache = kv_cache.get_index_cache();
    std::optional<torch::Tensor> indexer_cache_scale =
        kv_cache.get_indexer_cache_scale();
    torch::Tensor& indexer_cache_scale_tensor = indexer_cache_scale.value();

    std::tuple<torch::Tensor, torch::Tensor> indexer_states{index_kv_state,
                                                            index_score_state};
    std::tuple<torch::Tensor, torch::Tensor> indexer_block_tables{
        index_kv_block_table, index_score_block_table};
    auto indexer_metadata =
        build_indexer_attention_metadata(attn_metadata,
                                         index_block_table,
                                         index_slot,
                                         use_prefill_attn,
                                         attn_metadata.max_query_len,
                                         attn_metadata.max_seq_len);
    CHECK(qli_metadata.defined()) << "DSAttention requires precomputed "
                                     "qli_metadata for compress_ratio==4.";
    auto qli_metadata_opt = std::optional<torch::Tensor>(qli_metadata);
    compress_topk_idxs =
        indexer_->select_qli(hidden_states,
                             qr,
                             qr_pertoken_scale,
                             index_cache,
                             &indexer_cache_scale_tensor,
                             indexer_metadata,
                             cos,
                             sin,
                             attn_metadata.c4_cos,
                             attn_metadata.c4_sin,
                             attn_metadata.actual_seq_lengths_query,
                             attn_metadata.actual_seq_lengths_kv,
                             qli_metadata_opt,
                             use_prefill_attn,
                             &indexer_states,
                             &indexer_block_tables);
    CHECK(compress_topk_idxs.defined())
        << "DSAttention indexer returned undefined topk indices for "
           "compress_ratio==4.";
  }

  // 7) sparse shared-kv attention
  std::optional<torch::Tensor> sparse_metadata = std::nullopt;
  if (compress_ratio_i == 1) {
    sparse_metadata = as_optional(c1_metadata);
  } else if (compress_ratio_i == 4) {
    sparse_metadata = as_optional(c4_metadata);
  } else if (compress_ratio_i == 128) {
    sparse_metadata = as_optional(c128_metadata);
  }

  CHECK(sparse_metadata.has_value())
      << "DSAttention requires precomputed sparse metadata for compress_ratio="
      << compress_ratio_i;

  std::optional<torch::Tensor> cu_seqlens_ori_kv_for_attn = std::nullopt;
  if (use_prefill_attn) {
    // Prefill-style sparse metadata uses query cu-seqlens for ori_kv.
    cu_seqlens_ori_kv_for_attn =
        as_optional(attn_metadata.actual_seq_lengths_query);
  }

  auto [attn_output, output_lse] = xllm::kernel::npu::sparse_attn_sharedkv(
      /*q=*/q,
      /*ori_kv=*/as_optional(ori_kv_for_attn),
      /*cmp_kv=*/compress_ratio_i > 1 ? as_optional(cmp_kv_for_attn)
                                      : std::nullopt,
      /*ori_sparse_indices=*/std::nullopt,
      /*cmp_sparse_indices=*/
      compress_ratio_i == 4 ? as_optional(compress_topk_idxs) : std::nullopt,
      /*ori_block_table=*/as_optional(ori_block_table_for_attn),
      /*cmp_block_table=*/
      compress_ratio_i > 1 ? as_optional(cmp_block_table) : std::nullopt,
      /*cu_seqlens_q=*/as_optional(attn_metadata.actual_seq_lengths_query),
      /*cu_seqlens_ori_kv=*/cu_seqlens_ori_kv_for_attn,
      /*cu_seqlens_cmp_kv=*/std::nullopt,
      /*seqused_q=*/std::nullopt,
      /*seqused_kv=*/as_optional(attn_metadata.actual_seq_lengths_kv),
      /*sinks=*/attn_sink_loaded_ ? as_optional(attn_sink_) : std::nullopt,
      /*metadata=*/sparse_metadata,
      /*softmax_scale=*/softmax_scale_,
      /*cmp_ratio=*/compress_ratio_i,
      /*ori_mask_mode=*/4,
      /*cmp_mask_mode=*/3,
      /*ori_win_left=*/std::max<int64_t>(window_size_ - 1, 0),
      /*ori_win_right=*/0,
      /*layout_q=*/"TND",
      /*layout_kv=*/ori_kv_layout,
      /*return_softmax_lse=*/false);

  // 8) Deferred cache write for full prefill.
  if (use_temporary_prefill_kv) {
    scatter_by_slot(ori_kv, ori_slot, kv);
  }

  // 9) output RoPE + projection
  auto o = attn_output.view({-1, n_local_heads_, head_dim_});
  apply_partial_rope(
      o, nope_head_dim_, rope_head_dim_, cos, sin, /*inverse=*/true);

  const int64_t num_tokens = o.size(0);
  auto o_group = o.view({num_tokens, n_local_groups_, -1});
  auto wo_a = o_a_proj_->weight().view({n_local_groups_, o_lora_rank_, -1});
  auto o_low_rank = torch::einsum("tgd,grd->tgr", {o_group, wo_a});
  auto output = o_b_proj_->forward(o_low_rank.reshape({num_tokens, -1}));
  std::optional<torch::Tensor> final_lse = std::nullopt;
  (void)output_lse;

  return std::make_tuple(output, final_lse);
}

void DSAttentionImpl::load_state_dict(const StateDict& state_dict) {
  q_a_proj_->load_state_dict(state_dict.get_dict_with_prefix("wq_a."));
  q_b_proj_->load_state_dict(state_dict.get_dict_with_prefix("wq_b."));
  q_layernorm_->load_state_dict(state_dict.get_dict_with_prefix("q_norm."));

  kv_proj_->load_state_dict(state_dict.get_dict_with_prefix("wkv."));
  kv_layernorm_->load_state_dict(state_dict.get_dict_with_prefix("kv_norm."));
  o_a_proj_->load_state_dict(state_dict.get_dict_with_prefix("wo_a."));
  o_b_proj_->load_state_dict(state_dict.get_dict_with_prefix("wo_b."));

  auto attn_sink = state_dict.get_tensor("attn_sink");
  if (!attn_sink.defined()) {
    attn_sink = state_dict.get_tensor("attn_sink.weight");
  }
  if (attn_sink.defined()) {
    if (attn_sink.dim() == 1 && attn_sink.size(0) == num_heads_ &&
        tp_size_ > 1) {
      CHECK_EQ(num_heads_ % tp_size_, 0)
          << "attn_sink full-head tensor size is not divisible by tp_size.";
      const int64_t shard_size = num_heads_ / tp_size_;
      const int64_t shard_start = tp_rank_ * shard_size;
      attn_sink = attn_sink.slice(/*dim=*/0,
                                  /*start=*/shard_start,
                                  /*end=*/shard_start + shard_size);
    }

    CHECK(attn_sink.dim() == 1 && attn_sink.size(0) == n_local_heads_)
        << "attn_sink shape mismatch, expected [" << n_local_heads_ << "], got "
        << attn_sink.sizes();

    torch::NoGradGuard no_grad;
    attn_sink_.copy_(attn_sink.to(attn_sink_.device()).to(attn_sink_.dtype()));
    attn_sink_loaded_ = true;
  }

  if (compressor_ && compress_ratio_ >= 4) {
    auto compressor_state = state_dict.get_dict_with_prefix("compressor.");
    if (compressor_state.size() == 0) {
      compressor_state = state_dict.get_dict_with_prefix("compress.");
    }
    if (compressor_state.size() > 0) {
      compressor_->load_state_dict(compressor_state);
    }
  }

  if (indexer_ && compress_ratio_ == 4) {
    indexer_->load_state_dict(state_dict.get_dict_with_prefix("indexer."));
  }
}

int64_t DSAttentionImpl::non_registered_weight_bytes() const {
  if (!compressor_) {
    return 0;
  }
  return compressor_->weight_bytes();
}

}  // namespace layer
}  // namespace xllm

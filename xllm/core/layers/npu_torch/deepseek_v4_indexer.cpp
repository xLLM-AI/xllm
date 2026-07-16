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

#include "layers/npu_torch/deepseek_v4_indexer.h"

#include <glog/logging.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#include "kernels/ops_api.h"
#include "xllm/core/kernels/npu/xllm_ops/xllm_ops_api.h"

namespace xllm {
namespace layer {
namespace {

inline bool is_power_of_two(int64_t n) { return n > 0 && ((n & (n - 1)) == 0); }

torch::Tensor create_hadamard_matrix(int64_t n,
                                     torch::Dtype dtype,
                                     torch::Device device,
                                     bool normalize) {
  CHECK(is_power_of_two(n)) << "hadamard_matrix: n must be a power of two.";
  auto options = torch::TensorOptions().dtype(dtype).device(device);
  torch::Tensor matrix = torch::ones({1, 1}, options);
  for (int64_t m = 1; m < n; m <<= 1) {
    auto top = torch::cat({matrix, matrix}, 1);
    auto bottom = torch::cat({matrix, -matrix}, 1);
    matrix = torch::cat({top, bottom}, 0);
  }
  if (normalize) {
    matrix = matrix / std::sqrt(static_cast<double>(n));
  }
  return matrix;
}

const torch::Tensor& get_hadamard_matrix(const AttentionMetadata& attn_metadata,
                                         const torch::Tensor& fallback) {
  if (attn_metadata.dsa_metadata != nullptr &&
      attn_metadata.dsa_metadata->hadamard.defined()) {
    return attn_metadata.dsa_metadata->hadamard;
  }
  return fallback;
}

torch::Tensor hadamard_transform_ref(const torch::Tensor& x,
                                     const torch::Tensor& hadamard_matrix) {
  auto x_shape = x.sizes();
  int64_t dim = x.size(-1);
  auto x2d = x.reshape({-1, dim});
  int64_t dim_padded = hadamard_matrix.size(0);
  if (dim != dim_padded) {
    x2d = torch::nn::functional::pad(
        x2d,
        torch::nn::functional::PadFuncOptions({0, dim_padded - dim})
            .mode(torch::kConstant)
            .value(0));
  }
  auto out = torch::nn::functional::linear(x2d, hadamard_matrix);
  using torch::indexing::Slice;
  out = out.index({Slice(), Slice(0, dim)});
  return out.reshape(x_shape);
}

torch::Tensor rotate_activation_with_hadamard(const torch::Tensor& x,
                                              const torch::Tensor& hadamard,
                                              double scale) {
  auto out = hadamard_transform_ref(x, hadamard);
  if (scale != 1.0) {
    out = out * scale;
  }
  return out;
}

struct SlotScatterPlan {
  torch::Tensor slots_slice;
  torch::Tensor safe_indices;
  int64_t update_rows = 0;
};

SlotScatterPlan prepare_slot_scatter_plan(const torch::Tensor& slot_mapping,
                                          int64_t value_rows,
                                          const c10::Device& device) {
  SlotScatterPlan plan;
  if (!slot_mapping.defined() || slot_mapping.numel() == 0 || value_rows <= 0) {
    return plan;
  }

  torch::Tensor slots = slot_mapping.reshape({-1}).to(torch::kLong).to(device);
  plan.update_rows = std::min<int64_t>(slots.size(0), value_rows);
  if (plan.update_rows <= 0) {
    return plan;
  }

  plan.slots_slice =
      slots.slice(/*dim=*/0, /*start=*/0, /*end=*/plan.update_rows);
  if (!device.is_cpu()) {
    plan.safe_indices = plan.slots_slice.clamp_min(0).reshape({-1, 1});
  }
  return plan;
}

// Scatter a row-major tensor into a flattened cache according to slot ids.
// Each valid slot in `slot_mapping` selects the destination row for the same
// row in `value`; negative slots are treated as padding and ignored.
void scatter_rows_by_prepared_slot(torch::Tensor& cache,
                                   const SlotScatterPlan& plan,
                                   const torch::Tensor& value) {
  if (!cache.defined() || !value.defined() || !plan.slots_slice.defined()) {
    return;
  }
  if (plan.update_rows <= 0 || value.numel() == 0) {
    return;
  }

  auto value_2d = value.reshape({-1, value.size(value.dim() - 1)});
  auto cache_2d = cache.view({-1, value_2d.size(1)});
  const int64_t update_rows =
      std::min<int64_t>(plan.update_rows, value_2d.size(0));
  if (update_rows <= 0) {
    return;
  }

  torch::Tensor slots_slice =
      plan.slots_slice.slice(/*dim=*/0, /*start=*/0, /*end=*/update_rows);
  torch::Tensor value_slice =
      value_2d.slice(/*dim=*/0, /*start=*/0, /*end=*/update_rows);

  if (!cache.device().is_cpu()) {
    torch::Tensor safe_indices =
        plan.safe_indices.defined()
            ? plan.safe_indices.slice(/*dim=*/0,
                                      /*start=*/0,
                                      /*end=*/update_rows)
            : slots_slice.clamp_min(0).reshape({-1, 1});
    xllm::kernel::npu::scatter_nd_update(cache_2d, safe_indices, value_slice);
    return;
  }

  // Negative slots mean "unused" entries in the metadata. Skip them so the
  // caller can pass padded or partially-filled mappings safely.
  torch::Tensor valid_mask = slots_slice.ge(0);
  torch::Tensor valid_slots = slots_slice.index({valid_mask});
  if (valid_slots.numel() == 0) {
    return;
  }

  torch::Tensor valid_values = value_slice.index({valid_mask});
  const int64_t cache_rows = cache_2d.size(0);
  const int64_t max_slot = valid_slots.max().item<int64_t>();
  CHECK_LT(max_slot, cache_rows)
      << "scatter_rows_by_slot slot index out of range: max_slot=" << max_slot
      << ", cache_rows=" << cache_rows << ", value_shape=" << value.sizes();
  // Write the valid rows into the flattened cache view.
  cache_2d.index_copy_(/*dim=*/0, valid_slots, valid_values);
}

void scatter_rows_by_slot(torch::Tensor& cache,
                          const torch::Tensor& slot_mapping,
                          const torch::Tensor& value) {
  if (!cache.defined() || !slot_mapping.defined() || !value.defined()) {
    return;
  }

  torch::Tensor value_2d = value.reshape({-1, value.size(value.dim() - 1)});
  SlotScatterPlan plan =
      prepare_slot_scatter_plan(slot_mapping, value_2d.size(0), cache.device());
  scatter_rows_by_prepared_slot(cache, plan, value);
}

torch::Tensor apply_partial_rope(torch::Tensor q,
                                 int64_t rope_start_dim,
                                 int64_t rope_head_dim,
                                 const torch::Tensor& cos,
                                 const torch::Tensor& sin) {
  if (!q.defined() || !cos.defined() || !sin.defined() || rope_head_dim <= 0 ||
      rope_start_dim < 0 || rope_start_dim + rope_head_dim > q.size(-1)) {
    return q;
  }

  auto cos_cache = cos;
  auto sin_cache = sin;
  CHECK(q.dim() == 2 || q.dim() == 3)
      << "apply_partial_rope only supports q dim 2/3, got: " << q.dim();
  CHECK(cos_cache.dim() == 2 && sin_cache.dim() == 2)
      << "apply_partial_rope expects cos/sin dim=2, got cos dim "
      << cos_cache.dim() << ", sin dim " << sin_cache.dim();
  CHECK(cos_cache.size(0) == q.size(0) && sin_cache.size(0) == q.size(0))
      << "apply_partial_rope expects cos/sin batch == q.size(0), got cos "
      << cos_cache.size(0) << ", sin " << sin_cache.size(0) << ", q "
      << q.size(0);
  CHECK(cos_cache.size(1) == rope_head_dim &&
        sin_cache.size(1) == rope_head_dim)
      << "apply_partial_rope expects cos/sin last dim == rope_head_dim("
      << rope_head_dim << "), got cos " << cos_cache.size(1) << ", sin "
      << sin_cache.size(1);

  auto cos_4d = cos_cache.view({cos_cache.size(0), 1, 1, rope_head_dim});
  auto sin_4d = sin_cache.view({sin_cache.size(0), 1, 1, rope_head_dim});
  auto q_4d = (q.dim() == 3) ? q.unsqueeze(1) : q.unsqueeze(1).unsqueeze(1);
  xllm::kernel::NpuInplacePartialRotaryMulParams rope_params;
  rope_params.x = q_4d;
  rope_params.r1 = cos_4d;
  rope_params.r2 = sin_4d;
  rope_params.rotary_mode = "interleave";
  rope_params.partial_slice = {rope_start_dim, rope_start_dim + rope_head_dim};
  xllm::kernel::npu_inplace_partial_rotary_mul(rope_params);
  return (q.dim() == 3) ? q_4d.squeeze(1) : q_4d.squeeze(1).squeeze(1);
}

std::tuple<torch::Tensor, torch::Tensor> dynamic_quant_int8(
    const torch::Tensor& input) {
  if (!input.device().is_cpu()) {
    xllm::kernel::NpuQuantizeParams quant_params;
    quant_params.input = input;

    torch::Tensor quant;
    std::optional<torch::Tensor> scale;
    std::tie(quant, scale) = xllm::kernel::dynamic_quant(quant_params);
    CHECK(scale.has_value() && scale->defined())
        << "DeepseekV4Indexer dynamic_quant must return scale.";
    return {quant, scale.value()};
  }

  auto max_abs = input.abs().amax(-1, true).to(torch::kFloat32);
  auto safe_max = torch::where(max_abs > 0, max_abs, torch::ones_like(max_abs));
  auto scale = safe_max / 127.0;
  auto quant = torch::round(input.to(torch::kFloat32) / scale)
                   .clamp(-128, 127)
                   .to(torch::kInt8);
  return {quant, scale.squeeze(-1)};
}

}  // namespace

DeepseekV4IndexerImpl::DeepseekV4IndexerImpl(
    int64_t dim,
    int64_t index_n_heads,
    int64_t index_head_dim,
    int64_t rope_head_dim,
    int64_t index_topk,
    int64_t q_lora_rank,
    int64_t compress_ratio,
    double norm_eps,
    const QuantArgs& quant_args,
    const torch::TensorOptions& options)
    : dim_(dim),
      n_heads_(index_n_heads),
      head_dim_(index_head_dim),
      rope_head_dim_(rope_head_dim),
      index_topk_(index_topk),
      q_lora_rank_(q_lora_rank),
      compress_ratio_(compress_ratio),
      softmax_scale_(std::pow(static_cast<double>(index_head_dim), -0.5)) {
  CHECK(dim_ > 0) << "DeepseekV4Indexer: dim must be > 0";
  CHECK(n_heads_ > 0) << "DeepseekV4Indexer: index_n_heads must be > 0";
  CHECK(head_dim_ > 0) << "DeepseekV4Indexer: index_head_dim must be > 0";
  CHECK(q_lora_rank_ > 0) << "DeepseekV4Indexer: q_lora_rank must be > 0";
  CHECK(compress_ratio_ > 0) << "DeepseekV4Indexer: compress_ratio must be > 0";

  wq_b_ = register_module("wq_b",
                          ReplicatedLinear(q_lora_rank_,
                                           n_heads_ * head_dim_,
                                           /*bias=*/false,
                                           quant_args,
                                           options));

  weights_proj_ = register_module("weights_proj",
                                  ReplicatedLinear(dim_,
                                                   n_heads_,
                                                   /*bias=*/false,
                                                   quant_args,
                                                   options));

  compressor_ = register_module("compressor",
                                Compressor(compress_ratio_,
                                           head_dim_,
                                           rope_head_dim_,
                                           /*rot_mode=*/2,
                                           norm_eps,
                                           options));

  indexer_softmax_mul_head_dim_sqrt_ =
      softmax_scale_ * std::pow(static_cast<double>(n_heads_), -0.5);

  hadamard_scale_ = std::pow(static_cast<double>(head_dim_), -0.5);
  index_head_dim_padded_ =
      static_cast<int64_t>(std::pow(2, std::ceil(std::log2(head_dim_))));
  hadamard_matrix_ = create_hadamard_matrix(index_head_dim_padded_,
                                            options.dtype().toScalarType(),
                                            options.device(),
                                            /*normalize=*/false);
}

torch::Tensor DeepseekV4IndexerImpl::build_query(const torch::Tensor& qr) {
  CHECK(qr.defined()) << "DeepseekV4Indexer::build_query: qr is undefined";
  auto q = wq_b_->forward(qr);
  q = q.view({q.size(0), n_heads_, head_dim_});
  return q;
}

torch::Tensor DeepseekV4IndexerImpl::build_query(
    const torch::Tensor& qr,
    const std::optional<torch::Tensor>& qr_pertoken_scale) {
  CHECK(qr.defined()) << "DeepseekV4Indexer::build_query: qr is undefined";
  if (wq_b_->uses_w8a8_dynamic_quant() && qr_pertoken_scale.has_value() &&
      qr_pertoken_scale->defined()) {
    xllm::kernel::QuantMatmulParams params;
    params.x1 = qr;
    params.x2 = wq_b_->weight();
    params.transpose2 = true;
    params.scale = wq_b_->w8a8_dynamic_weight_scale();
    params.pertoken_scale = qr_pertoken_scale;
    params.bias = wq_b_->bias();
    params.output_dtype = wq_b_->output_dtype();
    auto q = xllm::kernel::quant_matmul(params);
    return q.view({q.size(0), n_heads_, head_dim_});
  }
  return build_query(qr);
}

torch::Tensor DeepseekV4IndexerImpl::build_weights(const torch::Tensor& x) {
  CHECK(x.defined()) << "DeepseekV4Indexer::build_weights: x is undefined";
  return weights_proj_->forward(x) * indexer_softmax_mul_head_dim_sqrt_;
}

torch::Tensor DeepseekV4IndexerImpl::compress_kv(
    const torch::Tensor& x,
    const AttentionMetadata& attn_metadata,
    const std::optional<torch::Tensor>& compressed_cos,
    const std::optional<torch::Tensor>& compressed_sin,
    const std::optional<torch::Tensor>& actual_seq_lengths_query,
    std::tuple<torch::Tensor, torch::Tensor>* compressor_states,
    std::tuple<torch::Tensor, torch::Tensor>* compressor_block_tables) {
  CHECK(x.defined()) << "DeepseekV4Indexer::compress_kv: x is undefined";
  CHECK(compressor_)
      << "DeepseekV4Indexer::compress_kv: compressor is not initialized";
  CHECK(compressor_states != nullptr)
      << "DeepseekV4Indexer::compress_kv: compressor_states is required";
  CHECK(compressor_block_tables != nullptr)
      << "DeepseekV4Indexer::compress_kv: compressor_block_tables is required";
  CHECK(compressed_cos.has_value())
      << "DeepseekV4Indexer::compress_kv: compressed_cos is required";
  CHECK(compressed_sin.has_value())
      << "DeepseekV4Indexer::compress_kv: compressed_sin is required";
  CHECK(actual_seq_lengths_query.has_value())
      << "DeepseekV4Indexer::compress_kv: actual_seq_lengths_query is required";
  CHECK(attn_metadata.dsa_metadata != nullptr)
      << "DeepseekV4Indexer::compress_kv: dsa_metadata is required";
  CHECK(attn_metadata.dsa_metadata->start_pos.defined())
      << "DeepseekV4Indexer::compress_kv: dsa_metadata.start_pos is required";

  auto dsa_metadata = DSAMetadata{};
  dsa_metadata.start_pos = attn_metadata.dsa_metadata->start_pos;

  auto hidden_states = x;
  auto compressed_sin_view = compressed_sin.value();
  auto compressed_cos_view = compressed_cos.value();
  auto actual_q_lens = actual_seq_lengths_query.value();

  return compressor_->forward(dsa_metadata,
                              hidden_states,
                              *compressor_states,
                              *compressor_block_tables,
                              compressed_sin_view,
                              compressed_cos_view,
                              actual_q_lens);
}

std::tuple<torch::Tensor, torch::Tensor> DeepseekV4IndexerImpl::forward(
    const torch::Tensor& x,
    const torch::Tensor& qr) {
  auto q = build_query(qr);
  auto weights = build_weights(x);
  return {q, weights};
}

torch::Tensor DeepseekV4IndexerImpl::select_qli(
    const torch::Tensor& x,
    const torch::Tensor& qr,
    const std::optional<torch::Tensor>& qr_pertoken_scale,
    torch::Tensor& index_cache,
    torch::Tensor* quant_index_cache,
    const AttentionMetadata& attn_metadata,
    const std::optional<torch::Tensor>& cos,
    const std::optional<torch::Tensor>& sin,
    const std::optional<torch::Tensor>& compressed_cos,
    const std::optional<torch::Tensor>& compressed_sin,
    const std::optional<torch::Tensor>& actual_seq_lengths_query,
    const std::optional<torch::Tensor>& actual_seq_lengths_key,
    const std::optional<torch::Tensor>& qli_metadata,
    bool with_prefill,
    std::tuple<torch::Tensor, torch::Tensor>* compressor_states,
    std::tuple<torch::Tensor, torch::Tensor>* compressor_block_tables,
    const std::optional<torch::Tensor>& compress_x,
    const std::optional<torch::Tensor>& compress_cu_seqlens) {
  CHECK(index_cache.defined())
      << "DeepseekV4Indexer::select_qli: index_cache is undefined";

  (void)with_prefill;
  auto q = build_query(qr, qr_pertoken_scale);
  if (cos.has_value() && sin.has_value()) {
    const int64_t rope_start_dim =
        std::max<int64_t>(head_dim_ - rope_head_dim_, 0);
    q = apply_partial_rope(
        q, rope_start_dim, rope_head_dim_, cos.value(), sin.value());
  }
  auto hadamard = get_hadamard_matrix(attn_metadata, hadamard_matrix_);
  q = rotate_activation_with_hadamard(q, hadamard, hadamard_scale_);

  // DSA-CP: the index-key cache is written for the FULL token stream
  // (replicated per CP rank, like vllm-ascend _update_indexer_cache),
  // so the compressor sees full hidden + full cu-seqlens; its full c4
  // RoPE row count then matches. query/weights/top-k below stay local.
  const torch::Tensor& compress_hidden =
      compress_x.has_value() ? compress_x.value() : x;
  const std::optional<torch::Tensor>& compress_q_cu =
      compress_cu_seqlens.has_value() ? compress_cu_seqlens
                                      : actual_seq_lengths_query;
  auto kv = compress_kv(compress_hidden,
                        attn_metadata,
                        compressed_cos,
                        compressed_sin,
                        compress_q_cu,
                        compressor_states,
                        compressor_block_tables);
  if (kv.numel() > 0) {
    kv = rotate_activation_with_hadamard(kv, hadamard, hadamard_scale_);
  }

  auto weights = build_weights(x);
  auto [q_quant, q_scale] = dynamic_quant_int8(q);
  q_scale = q_scale.to(torch::kFloat16);
  torch::Tensor kv_quant;
  torch::Tensor kv_scale;
  if (kv.numel() > 0) {
    std::tie(kv_quant, kv_scale) = dynamic_quant_int8(kv);
    kv_scale = kv_scale.unsqueeze(-1);
    kv_scale = kv_scale.to(torch::kFloat16);
  }

  if (kv.numel() > 0) {
    torch::Tensor kv_quant_2d =
        kv_quant.reshape({-1, kv_quant.size(kv_quant.dim() - 1)});
    SlotScatterPlan scatter_plan = prepare_slot_scatter_plan(
        attn_metadata.slot_mapping, kv_quant_2d.size(0), index_cache.device());
    scatter_rows_by_prepared_slot(index_cache, scatter_plan, kv_quant);
    if (quant_index_cache != nullptr && quant_index_cache->defined()) {
      scatter_rows_by_prepared_slot(*quant_index_cache, scatter_plan, kv_scale);
    }
  }

  torch::Tensor query_seq_lens;
  if (actual_seq_lengths_query.has_value()) {
    auto query_cu_seq_lens = actual_seq_lengths_query.value();
    query_seq_lens =
        (query_cu_seq_lens.dim() > 0 && query_cu_seq_lens.size(0) > 1)
            ? query_cu_seq_lens.slice(0, 1, query_cu_seq_lens.size(0))
            : query_cu_seq_lens;
  } else if (attn_metadata.q_seq_lens.defined()) {
    query_seq_lens = attn_metadata.q_seq_lens;
  } else if (attn_metadata.q_cu_seq_lens.defined() &&
             attn_metadata.q_cu_seq_lens.dim() > 0 &&
             attn_metadata.q_cu_seq_lens.size(0) > 1) {
    query_seq_lens = attn_metadata.q_cu_seq_lens.slice(
        0, 1, attn_metadata.q_cu_seq_lens.size(0));
  } else {
    query_seq_lens = attn_metadata.kv_seq_lens;
  }

  torch::Tensor key_seq_lens = actual_seq_lengths_key.has_value()
                                   ? actual_seq_lengths_key.value()
                                   : attn_metadata.kv_seq_lens;
  if (!key_seq_lens.defined() && attn_metadata.kv_cu_seq_lens.defined() &&
      attn_metadata.kv_cu_seq_lens.dim() > 0 &&
      attn_metadata.kv_cu_seq_lens.size(0) > 1) {
    auto kv_cu_seq_lens = attn_metadata.kv_cu_seq_lens;
    key_seq_lens = kv_cu_seq_lens.slice(0, 1, kv_cu_seq_lens.size(0));
  }

  torch::Tensor key_dequant_scale;
  if (quant_index_cache != nullptr && quant_index_cache->defined()) {
    key_dequant_scale = *quant_index_cache;
  } else {
    auto scale_sizes = index_cache.sizes().vec();
    CHECK(!scale_sizes.empty())
        << "DeepseekV4Indexer::select_qli: index_cache rank must be > 0";
    scale_sizes.back() = 1;
    key_dequant_scale = torch::ones(scale_sizes,
                                    torch::TensorOptions()
                                        .dtype(torch::kFloat16)
                                        .device(index_cache.device()));
  }

  c10::optional<torch::Tensor> metadata_opt = c10::nullopt;
  if (qli_metadata.has_value() && qli_metadata.value().defined()) {
    metadata_opt = qli_metadata.value();
  }

  xllm::kernel::QuantLightningIndexerParams qli_params;
  qli_params.query = q_quant;
  qli_params.key = index_cache;
  qli_params.weights = weights.to(torch::kFloat16);
  qli_params.query_dequant_scale = q_scale;
  qli_params.key_dequant_scale = key_dequant_scale;
  qli_params.query_quant_mode = 0;
  qli_params.key_quant_mode = 0;
  qli_params.actual_seq_lengths_query =
      c10::optional<torch::Tensor>(query_seq_lens);
  qli_params.actual_seq_lengths_key =
      c10::optional<torch::Tensor>(key_seq_lens);
  qli_params.block_table =
      c10::optional<torch::Tensor>(attn_metadata.block_table);
  qli_params.metadata = metadata_opt;
  qli_params.layout_query = "TND";
  qli_params.layout_key = "PA_BSND";
  qli_params.sparse_count = index_topk_;
  qli_params.sparse_mode = 3;
  qli_params.pre_tokens = std::numeric_limits<int64_t>::max();
  qli_params.next_tokens = std::numeric_limits<int64_t>::max();
  qli_params.cmp_ratio = compress_ratio_;
  qli_params.return_value = false;

  auto [topk_indices, sparse_values] =
      xllm::kernel::quant_lightning_indexer(qli_params);
  (void)sparse_values;

  (void)key_seq_lens;
  return topk_indices;
}

torch::Tensor DeepseekV4IndexerImpl::select_qli(
    const torch::Tensor& x,
    const torch::Tensor& qr,
    const std::optional<torch::Tensor>& qr_pertoken_scale,
    torch::Tensor& index_cache,
    const AttentionMetadata& attn_metadata,
    const std::optional<torch::Tensor>& cos,
    const std::optional<torch::Tensor>& sin,
    const std::optional<torch::Tensor>& compressed_cos,
    const std::optional<torch::Tensor>& compressed_sin,
    const std::optional<torch::Tensor>& actual_seq_lengths_query,
    const std::optional<torch::Tensor>& actual_seq_lengths_key,
    const std::optional<torch::Tensor>& qli_metadata,
    bool with_prefill,
    std::tuple<torch::Tensor, torch::Tensor>* compressor_states,
    std::tuple<torch::Tensor, torch::Tensor>* compressor_block_tables) {
  return select_qli(x,
                    qr,
                    qr_pertoken_scale,
                    index_cache,
                    /*quant_index_cache=*/nullptr,
                    attn_metadata,
                    cos,
                    sin,
                    compressed_cos,
                    compressed_sin,
                    actual_seq_lengths_query,
                    actual_seq_lengths_key,
                    qli_metadata,
                    with_prefill,
                    compressor_states,
                    compressor_block_tables);
}

void DeepseekV4IndexerImpl::load_state_dict(const StateDict& state_dict) {
  if (state_dict.size() == 0) {
    return;
  }

  wq_b_->load_state_dict(state_dict.get_dict_with_prefix("wq_b."));
  weights_proj_->load_state_dict(
      state_dict.get_dict_with_prefix("weights_proj."));
  compressor_->load_state_dict(state_dict.get_dict_with_prefix("compressor."));
}

}  // namespace layer
}  // namespace xllm

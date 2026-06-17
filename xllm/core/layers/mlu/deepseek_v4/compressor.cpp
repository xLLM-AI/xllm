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

#include "layers/mlu/deepseek_v4/compressor.h"

#include <glog/logging.h>

#include <algorithm>
#include <cmath>
#include <optional>
#include <tuple>
#include <vector>

#include "kernels/mlu/mlu_ops_api.h"
#include "kernels/ops_api.h"
#include "util/linalg.h"

namespace {

struct PrefillPadPlan {
  bool needs_padding = false;
  std::vector<int64_t> q_lens;
  std::vector<int64_t> prefix_lens;
  std::vector<bool> has_prev_windows;
  std::vector<int32_t> padded_cu_lens;
  std::vector<int64_t> keep_indices;
  int64_t max_padded_len = 0;
  int64_t padded_tokens = 0;
  int64_t padded_rows = 0;
};

void write_cache(const torch::Tensor& key,
                 torch::Tensor& cache,
                 const torch::Tensor& slot_mapping) {
  xllm::kernel::ReshapePagedCacheParams params;
  params.key = key.contiguous();
  params.value = std::nullopt;
  params.k_cache = cache;
  params.v_cache = std::nullopt;
  params.slot_mapping = slot_mapping.contiguous();
  params.direction = false;
  xllm::kernel::reshape_paged_cache(params);
}

void apply_rotary(torch::Tensor& kv,
                  const torch::Tensor& sin_table,
                  const torch::Tensor& cos_table,
                  const torch::Tensor& positions,
                  int64_t rope_head_dim) {
  if (rope_head_dim == 0 || kv.numel() == 0) {
    return;
  }

  torch::Tensor position_ids =
      positions.slice(/*dim=*/0, /*start=*/0, /*end=*/kv.size(0));
  torch::Tensor rope_part =
      kv.slice(/*dim=*/-1, /*start=*/kv.size(-1) - rope_head_dim).unsqueeze(1);
  xllm::kernel::RotaryParams params;
  params.q = rope_part;
  params.k = torch::Tensor();
  params.sin = sin_table;
  params.cos = cos_table;
  params.position_ids = position_ids;
  params.cu_query_lens = std::nullopt;
  params.interleaved = true;
  params.discrete = true;
  params.dynamic_ntk = false;
  params.max_query_len = kv.size(0);
  xllm::kernel::apply_rotary(params);
  rope_part.copy_(params.q);
}

const torch::Tensor& compressed_positions(const xllm::layer::DSAMetadata& dsa,
                                          int64_t compress_ratio) {
  if (compress_ratio == 4) {
    return dsa.c4_pad_positions;
  }
  return dsa.c128_pad_positions;
}

torch::Tensor empty_output(const torch::Tensor& hidden_states,
                           int64_t head_dim) {
  return torch::empty({0, head_dim}, hidden_states.options());
}

torch::Tensor make_compress_slots(const torch::Tensor& block_table,
                                  int64_t batch_size,
                                  int64_t seq_len,
                                  int64_t block_size,
                                  const torch::Device& device) {
  torch::Tensor positions = torch::arange(
      seq_len, torch::TensorOptions().dtype(torch::kInt64).device(device));
  torch::Tensor block_col = torch::floor_divide(positions, block_size);
  torch::Tensor block_offset = positions - block_col * block_size;
  torch::Tensor bt_i64 = block_table.to(torch::kInt64);
  torch::Tensor idx = block_col.unsqueeze(0)
                          .expand({batch_size, -1})
                          .to(bt_i64.device())
                          .to(torch::kInt32);
  torch::Tensor block_ids = bt_i64.gather(/*dim=*/1, idx);
  return (block_ids * block_size + block_offset.unsqueeze(0))
      .to(torch::kInt32)
      .reshape({-1})
      .contiguous();
}

PrefillPadPlan make_prefill_pad_plan(
    const xllm::layer::AttentionMetadata& attn_metadata,
    int64_t compress_ratio,
    int64_t coff) {
  PrefillPadPlan plan;
  const xllm::layer::DSAMetadata& dsa = *attn_metadata.dsa_metadata;
  const int64_t batch_size = static_cast<int64_t>(dsa.start_pos_vec.size());
  plan.q_lens.reserve(static_cast<size_t>(batch_size));
  plan.prefix_lens.reserve(static_cast<size_t>(batch_size));
  plan.has_prev_windows.reserve(static_cast<size_t>(batch_size));
  plan.padded_cu_lens.reserve(static_cast<size_t>(batch_size + 1));
  plan.padded_cu_lens.emplace_back(0);

  torch::Tensor q_cu_cpu =
      attn_metadata.q_cu_seq_lens.to(torch::kCPU).to(torch::kInt64);
  int64_t row_begin = 0;
  for (int64_t seq_idx = 0; seq_idx < batch_size; ++seq_idx) {
    const int64_t q_begin = q_cu_cpu[seq_idx].item<int64_t>();
    const int64_t q_end = q_cu_cpu[seq_idx + 1].item<int64_t>();
    const int64_t q_len = q_end - q_begin;
    const int64_t start_pos = dsa.start_pos_vec[static_cast<size_t>(seq_idx)];
    const int64_t prefix_len = start_pos % compress_ratio;
    const bool has_prev_window = coff == 2 && start_pos >= compress_ratio;
    const int64_t synthetic_len = has_prev_window ? compress_ratio : 0;
    const int64_t padded_len = synthetic_len + prefix_len + q_len;
    const int64_t seq_rows = padded_len / compress_ratio;

    plan.q_lens.emplace_back(q_len);
    plan.prefix_lens.emplace_back(prefix_len);
    plan.has_prev_windows.emplace_back(has_prev_window);
    plan.padded_tokens += padded_len;
    plan.padded_rows += seq_rows;
    plan.max_padded_len = std::max(plan.max_padded_len, padded_len);
    plan.padded_cu_lens.emplace_back(static_cast<int32_t>(plan.padded_tokens));
    plan.needs_padding =
        plan.needs_padding || prefix_len > 0 || has_prev_window;

    const int64_t drop_rows = has_prev_window ? 1 : 0;
    for (int64_t row_idx = drop_rows; row_idx < seq_rows; ++row_idx) {
      plan.keep_indices.emplace_back(row_begin + row_idx);
    }
    row_begin += seq_rows;
  }
  return plan;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> pad_prefill_pack(
    const torch::Tensor& kv_pack,
    const torch::Tensor& score_pack,
    const torch::Tensor& kv_state,
    const torch::Tensor& score_state,
    const torch::Tensor& q_cu_seq_lens,
    const torch::Tensor& ape,
    const PrefillPadPlan& plan,
    int64_t compress_ratio,
    int64_t coff) {
  std::vector<torch::Tensor> kv_parts;
  std::vector<torch::Tensor> score_parts;
  kv_parts.reserve(static_cast<size_t>(plan.q_lens.size()) * 3);
  score_parts.reserve(static_cast<size_t>(plan.q_lens.size()) * 3);
  torch::Tensor q_cu_cpu = q_cu_seq_lens.to(torch::kCPU).to(torch::kInt64);
  const int64_t batch_size = static_cast<int64_t>(plan.q_lens.size());
  for (int64_t seq_idx = 0; seq_idx < batch_size; ++seq_idx) {
    torch::Tensor seq_kv_state = kv_state.select(/*dim=*/0, seq_idx);
    torch::Tensor seq_score_state = score_state.select(/*dim=*/0, seq_idx);
    if (plan.has_prev_windows[static_cast<size_t>(seq_idx)]) {
      torch::Tensor prev_kv =
          seq_kv_state.slice(/*dim=*/0, /*start=*/0, /*end=*/compress_ratio);
      torch::Tensor prev_score = seq_score_state.slice(
          /*dim=*/0, /*start=*/0, /*end=*/compress_ratio);
      kv_parts.emplace_back(prev_kv);
      score_parts.emplace_back(prev_score - ape.slice(/*dim=*/0,
                                                      /*start=*/0,
                                                      /*end=*/compress_ratio));
    }

    const int64_t prefix_len = plan.prefix_lens[static_cast<size_t>(seq_idx)];
    if (prefix_len > 0) {
      const int64_t state_offset = coff == 2 ? compress_ratio : 0;
      torch::Tensor prefix_kv =
          seq_kv_state.slice(/*dim=*/0,
                             /*start=*/state_offset,
                             /*end=*/state_offset + prefix_len);
      torch::Tensor prefix_score =
          seq_score_state.slice(/*dim=*/0,
                                /*start=*/state_offset,
                                /*end=*/state_offset + prefix_len);
      kv_parts.emplace_back(prefix_kv);
      score_parts.emplace_back(
          prefix_score - ape.slice(/*dim=*/0, /*start=*/0, /*end=*/prefix_len));
    }

    const int64_t q_begin = q_cu_cpu[seq_idx].item<int64_t>();
    const int64_t q_end = q_cu_cpu[seq_idx + 1].item<int64_t>();
    if (q_end > q_begin) {
      kv_parts.emplace_back(
          kv_pack.slice(/*dim=*/0, /*start=*/q_begin, /*end=*/q_end));
      score_parts.emplace_back(
          score_pack.slice(/*dim=*/0, /*start=*/q_begin, /*end=*/q_end));
    }
  }

  torch::Tensor padded_kv =
      kv_parts.empty() ? kv_pack.slice(0, 0, 0) : torch::cat(kv_parts, 0);
  torch::Tensor padded_score = score_parts.empty() ? score_pack.slice(0, 0, 0)
                                                   : torch::cat(score_parts, 0);
  torch::Tensor padded_cu_lens =
      torch::tensor(plan.padded_cu_lens,
                    torch::TensorOptions()
                        .dtype(torch::kInt32)
                        .device(q_cu_seq_lens.device()));
  return {padded_kv.contiguous(), padded_score.contiguous(), padded_cu_lens};
}

torch::Tensor drop_synthetic_rows(const torch::Tensor& compressed_kv,
                                  const PrefillPadPlan& plan,
                                  const torch::TensorOptions& options,
                                  int64_t head_dim) {
  if (static_cast<int64_t>(plan.keep_indices.size()) == compressed_kv.size(0)) {
    return compressed_kv;
  }
  if (plan.keep_indices.empty()) {
    return torch::empty({0, head_dim}, options);
  }
  torch::Tensor keep = torch::tensor(plan.keep_indices,
                                     torch::TensorOptions()
                                         .dtype(torch::kInt64)
                                         .device(compressed_kv.device()));
  return compressed_kv.index_select(/*dim=*/0, keep);
}

}  // namespace

namespace xllm {
namespace layer {

CompressorImpl::CompressorImpl(int64_t compress_ratio,
                               int64_t hidden_dim,
                               int64_t head_dim,
                               int64_t rope_head_dim,
                               bool rotate,
                               double norm_eps,
                               const torch::TensorOptions& options,
                               const QuantArgs& quant_args)
    : compress_ratio_(compress_ratio),
      hidden_dim_(hidden_dim),
      head_dim_(head_dim),
      rope_head_dim_(rope_head_dim),
      rotate_(rotate),
      eps_(norm_eps),
      overlap_(compress_ratio == 4 ? true : false),
      coff_(compress_ratio == 4 ? 2 : 1) {
  compress_len_ = compress_ratio_ * coff_;
  wkv_ = register_module("wkv",
                         ReplicatedLinear(hidden_dim_,
                                          coff_ * head_dim_,
                                          /*bias=*/false,
                                          quant_args,
                                          options));
  wgate_ = register_module("wgate",
                           ReplicatedLinear(hidden_dim_,
                                            coff_ * head_dim_,
                                            /*bias=*/false,
                                            quant_args,
                                            options));
  norm_ = register_module(
      "norm", RMSNorm(head_dim_, eps_, options.dtype(torch::kFloat32)));
  ape_ = register_parameter("ape",
                            torch::empty({compress_ratio_, coff_ * head_dim_},
                                         options.dtype(torch::kFloat32)),
                            /*requires_grad=*/false);

  if (rotate_) {
    const double log_dim = std::ceil(std::log2(static_cast<double>(head_dim_)));
    const int64_t padded_dim =
        static_cast<int64_t>(1ull << static_cast<uint64_t>(log_dim));
    hadamard_matrix_ = util::create_hadamard_matrix(padded_dim,
                                                    torch::kFloat32,
                                                    torch::Device(torch::kCPU),
                                                    /*normalize=*/true);
    hadamard_matrix_ =
        hadamard_matrix_.to(options.device(), options.dtype().toScalarType());
  }
}

torch::Tensor CompressorImpl::forward_prefill(
    const AttentionMetadata& attn_metadata,
    torch::Tensor& hidden_states,
    torch::Tensor& kv_cache,
    const torch::Tensor& slot_mapping,
    std::tuple<torch::Tensor, torch::Tensor>& kv_states,
    std::tuple<torch::Tensor, torch::Tensor>& block_tables,
    const torch::Tensor& compressed_sin_table,
    const torch::Tensor& compressed_cos_table) {
  torch::Tensor kv_pack = wkv_->forward(hidden_states);
  torch::Tensor score_pack = wgate_->forward(hidden_states);

  torch::Tensor& kv_state = std::get<0>(kv_states);
  torch::Tensor& score_state = std::get<1>(kv_states);

  torch::Tensor& kv_block_table = std::get<0>(block_tables);
  torch::Tensor& score_block_table = std::get<1>(block_tables);
  const int64_t batch_size =
      static_cast<int64_t>(attn_metadata.kv_seq_lens.size(0));
  auto kv_slots = make_compress_slots(kv_block_table,
                                      batch_size,
                                      compress_len_,
                                      kv_state.size(1),
                                      hidden_states.device());
  auto score_slots = make_compress_slots(score_block_table,
                                         batch_size,
                                         compress_len_,
                                         score_state.size(1),
                                         hidden_states.device());

  auto kv_state_flat =
      kv_state.view({kv_state.size(0) * kv_state.size(1), kv_state.size(2)});
  auto new_kv_state =
      kv_state_flat.index_select(/*dim=*/0, kv_slots.to(torch::kLong));
  new_kv_state = new_kv_state.view({-1, compress_len_, kv_state.size(2)});

  auto score_state_flat = score_state.view(
      {score_state.size(0) * score_state.size(1), score_state.size(2)});
  auto new_score_state =
      score_state_flat.index_select(/*dim=*/0, score_slots.to(torch::kLong));
  new_score_state =
      new_score_state.view({-1, compress_len_, score_state.size(2)});

  PrefillPadPlan pad_plan =
      make_prefill_pad_plan(attn_metadata, compress_ratio_, coff_);
  torch::Tensor fused_kv_pack = kv_pack;
  torch::Tensor fused_score_pack = score_pack;
  torch::Tensor fused_q_cu_seq_lens = attn_metadata.q_cu_seq_lens;
  int64_t fused_max_query_len = attn_metadata.max_query_len;
  int64_t fused_output_rows = slot_mapping.numel();
  if (pad_plan.needs_padding) {
    std::tie(fused_kv_pack, fused_score_pack, fused_q_cu_seq_lens) =
        pad_prefill_pack(kv_pack,
                         score_pack,
                         new_kv_state,
                         new_score_state,
                         attn_metadata.q_cu_seq_lens,
                         ape_,
                         pad_plan,
                         compress_ratio_,
                         coff_);
    fused_max_query_len = pad_plan.max_padded_len;
    fused_output_rows = pad_plan.padded_rows;
  }

  torch::Tensor compressed_kv =
      torch::empty({fused_output_rows, head_dim_}, hidden_states.options());
  torch::Tensor state_ids = torch::arange(batch_size,
                                          torch::TensorOptions()
                                              .dtype(torch::kInt32)
                                              .device(hidden_states.device()));

  xllm::kernel::mlu::fused_compress_multi_kv(fused_kv_pack,
                                             fused_score_pack,
                                             new_kv_state,
                                             new_score_state,
                                             fused_q_cu_seq_lens,
                                             state_ids,
                                             ape_,
                                             fused_max_query_len,
                                             overlap_,
                                             compressed_kv);
  if (pad_plan.needs_padding) {
    compressed_kv = drop_synthetic_rows(
        compressed_kv, pad_plan, hidden_states.options(), head_dim_);
  }

  auto score_state_cache = score_state.unsqueeze(1);
  auto kv_state_cache = kv_state.unsqueeze(1);
  write_cache(new_score_state.view({-1, 1, score_state.size(2)}),
              score_state_cache,
              score_slots);
  write_cache(
      new_kv_state.view({-1, 1, kv_state.size(2)}), kv_state_cache, kv_slots);

  if (slot_mapping.numel() == 0) {
    return empty_output(hidden_states, head_dim_);
  }
  auto kv = compressed_kv.to(torch::kFloat32);
  auto output = std::get<0>(norm_->forward(kv));
  output = output.to(hidden_states.scalar_type());
  const DSAMetadata& dsa = *attn_metadata.dsa_metadata;
  apply_rotary(output,
               compressed_sin_table,
               compressed_cos_table,
               compressed_positions(dsa, compress_ratio_),
               rope_head_dim_);
  if (rotate_) {
    output = util::rotate_activation(output, hadamard_matrix_);
  }

  write_cache(output.unsqueeze(1), kv_cache, slot_mapping);
  return output;
}

torch::Tensor CompressorImpl::forward_decode(
    const AttentionMetadata& attn_metadata,
    torch::Tensor& hidden_states,
    torch::Tensor& kv_cache,
    const torch::Tensor& slot_mapping,
    std::tuple<torch::Tensor, torch::Tensor>& kv_states,
    std::tuple<torch::Tensor, torch::Tensor>& block_tables,
    const torch::Tensor& compressed_sin_table,
    const torch::Tensor& compressed_cos_table) {
  torch::Tensor& kv_state = std::get<0>(kv_states);
  torch::Tensor& score_state = std::get<1>(kv_states);
  torch::Tensor& kv_block_table = std::get<0>(block_tables);
  torch::Tensor& score_block_table = std::get<1>(block_tables);

  const DSAMetadata& dsa = *attn_metadata.dsa_metadata;
  const int64_t batch_size =
      static_cast<int64_t>(attn_metadata.kv_seq_lens.size(0));

  torch::Tensor kv_pack = wkv_->forward(hidden_states);
  torch::Tensor score_pack = wgate_->forward(hidden_states);

  torch::Tensor kv_slots = make_compress_slots(kv_block_table,
                                               batch_size,
                                               compress_len_,
                                               kv_state.size(1),
                                               hidden_states.device());
  torch::Tensor score_slots = make_compress_slots(score_block_table,
                                                  batch_size,
                                                  compress_len_,
                                                  score_state.size(1),
                                                  hidden_states.device());
  torch::Tensor kv_state_flat =
      kv_state.view({kv_state.size(0) * kv_state.size(1), kv_state.size(2)});
  torch::Tensor new_kv_state =
      kv_state_flat.index_select(/*dim=*/0, kv_slots.to(torch::kLong));
  new_kv_state = new_kv_state.view({-1, compress_len_, kv_state.size(2)});

  torch::Tensor score_state_flat = score_state.view(
      {score_state.size(0) * score_state.size(1), score_state.size(2)});
  torch::Tensor new_score_state =
      score_state_flat.index_select(/*dim=*/0, score_slots.to(torch::kLong));
  new_score_state =
      new_score_state.view({-1, compress_len_, score_state.size(2)});

  if (kv_pack.dim() == 2) {
    kv_pack = kv_pack.unsqueeze(/*dim=*/1);
  }
  if (score_pack.dim() == 2) {
    score_pack = score_pack.unsqueeze(/*dim=*/1);
  }

  CHECK(slot_mapping.defined())
      << "CompressorImpl::forward_decode requires decode slot_mapping.";
  CHECK_EQ(slot_mapping.numel(), batch_size)
      << "CompressorImpl::forward_decode expects one decode slot per "
         "sequence.";
  torch::Tensor decode_slots = slot_mapping;
  torch::Tensor positions = dsa.input_positions;
  torch::Tensor kv_cache_view = kv_cache.reshape({-1, 1, head_dim_});
  torch::Tensor state_ids = torch::arange(new_kv_state.size(0),
                                          torch::TensorOptions()
                                              .dtype(torch::kInt32)
                                              .device(hidden_states.device()));
  std::optional<torch::Tensor> hadamard_matrix = std::nullopt;
  if (rotate_) {
    hadamard_matrix = hadamard_matrix_;
  }

  xllm::kernel::mlu::fused_compress_single_kv(kv_pack,
                                              score_pack,
                                              positions,
                                              state_ids,
                                              ape_,
                                              new_kv_state,
                                              new_score_state,
                                              norm_->weight(),
                                              compressed_sin_table,
                                              compressed_cos_table,
                                              hadamard_matrix,
                                              decode_slots,
                                              kv_cache_view,
                                              std::nullopt,
                                              eps_,
                                              overlap_,
                                              std::nullopt,
                                              /*mtp_token_num=*/0);

  auto score_state_cache = score_state.unsqueeze(1);
  auto kv_state_cache = kv_state.unsqueeze(1);
  write_cache(new_score_state.view({-1, 1, score_state.size(2)}),
              score_state_cache,
              score_slots);
  write_cache(
      new_kv_state.view({-1, 1, kv_state.size(2)}), kv_state_cache, kv_slots);
  return empty_output(hidden_states, head_dim_);
}

torch::Tensor CompressorImpl::forward(
    const AttentionMetadata& attn_metadata,
    torch::Tensor& hidden_states,
    torch::Tensor& kv_cache,
    const torch::Tensor& slot_mapping,
    std::tuple<torch::Tensor, torch::Tensor>& kv_states,
    std::tuple<torch::Tensor, torch::Tensor>& block_tables,
    const torch::Tensor& compressed_sin_table,
    const torch::Tensor& compressed_cos_table) {
  const bool is_prefill =
      attn_metadata.is_prefill || attn_metadata.is_chunked_prefill;
  torch::Tensor output;
  if (is_prefill) {
    output = forward_prefill(attn_metadata,
                             hidden_states,
                             kv_cache,
                             slot_mapping,
                             kv_states,
                             block_tables,
                             compressed_sin_table,
                             compressed_cos_table);
  } else {
    output = forward_decode(attn_metadata,
                            hidden_states,
                            kv_cache,
                            slot_mapping,
                            kv_states,
                            block_tables,
                            compressed_sin_table,
                            compressed_cos_table);
  }
  return output;
}

void CompressorImpl::load_state_dict(const StateDict& state_dict) {
  if (state_dict.size() == 0) {
    return;
  }
  wkv_->load_state_dict(state_dict.get_dict_with_prefix("wkv."));
  wgate_->load_state_dict(state_dict.get_dict_with_prefix("wgate."));
  norm_->load_state_dict(state_dict.get_dict_with_prefix("norm."));
  LOAD_WEIGHT(ape);
}

}  // namespace layer
}  // namespace xllm

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

#include "core/runtime/dcu_graph_executor_impl.h"

#include <glog/logging.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <numeric>
#include <vector>

#include "core/common/metrics.h"
#include "core/framework/config/execution_config.h"
#include "core/framework/config/scheduler_config.h"
#include "core/layers/common/attention_metadata.h"
#include "core/layers/common/attention_metadata_builder.h"
#include "core/util/rec_model_utils.h"
#include "core/util/utils.h"
#include "kernels/dcu/attention_runner.h"
#include "kernels/dcu/global_capture_instance.h"
#include "kernels/dcu/piecewise_graphs.h"

namespace xllm::runtime::dcu {

namespace {

torch::Tensor make_int_tensor_on_device(const std::vector<int32_t>& values,
                                        const torch::TensorOptions& options) {
  return torch::tensor(values, options);
}

void extend_q_cu_seq_lens(std::vector<int32_t>* cu_seq_lens,
                          uint32_t padding_needed,
                          uint32_t num_decoding_tokens) {
  if (cu_seq_lens == nullptr || cu_seq_lens->empty()) {
    return;
  }

  cu_seq_lens->reserve(cu_seq_lens->size() + padding_needed);
  for (uint32_t i = 0; i < padding_needed; ++i) {
    cu_seq_lens->push_back(cu_seq_lens->back() +
                           static_cast<int32_t>(num_decoding_tokens));
  }
}

void extend_kv_cu_seq_lens_for_padding(std::vector<int32_t>* cu_seq_lens,
                                       uint32_t padding_needed) {
  if (cu_seq_lens == nullptr || cu_seq_lens->empty()) {
    return;
  }

  // Padded decode rows have query tokens so the graph shape is stable, but
  // they do not own KV pages. Keep their KV length at zero; otherwise flash
  // attention may try to read a padding block table entry (-1).
  cu_seq_lens->resize(cu_seq_lens->size() + padding_needed,
                      cu_seq_lens->back());
}

std::vector<int32_t> compute_cu_seq_lens_diff(
    const std::vector<int32_t>& cu_seq_lens) {
  std::vector<int32_t> seq_lens;
  if (cu_seq_lens.size() < 2) {
    return seq_lens;
  }

  seq_lens.reserve(cu_seq_lens.size() - 1);
  for (std::size_t i = 1; i < cu_seq_lens.size(); ++i) {
    seq_lens.push_back(cu_seq_lens[i] - cu_seq_lens[i - 1]);
  }
  return seq_lens;
}

std::size_t tensor_bytes(const torch::Tensor& tensor) {
  if (!tensor.defined()) {
    return 0;
  }

  return static_cast<std::size_t>(tensor.numel()) * tensor.element_size();
}

}  // namespace

DcuGraphPersistentParam::DcuGraphPersistentParam(
    const ModelArgs& args,
    const torch::Device& device,
    const runtime::Options& options)
    : args_(args),
      device_(device),
      options_(options),
      num_decoding_tokens_(options.num_decoding_tokens()) {
  const int64_t max_tokens_per_batch =
      SchedulerConfig::get_instance().max_tokens_per_batch();

  int64_t max_seqs_per_batch = options.max_seqs_per_batch();
  if (is_rec_multi_round_mode()) {
    max_seqs_per_batch *= options.beam_width();
  }

  const int64_t max_seq_len = args.max_position_embeddings();
  const uint32_t block_size = options.block_size();
  const int64_t max_num_blocks_per_req =
      (max_seq_len + block_size - 1) / block_size + 1;

  torch::ScalarType dtype = util::parse_dtype(args.dtype(), device);
  if (args.dtype() == "float" || args.dtype() == "float32") {
    LOG(WARNING) << "DCU graph executor init hidden_states with float32 dtype. "
                 << "This is usually only expected in tests.";
    dtype = torch::kFloat32;
  }

  torch::TensorOptions tensor_options =
      torch::TensorOptions().device(device).dtype(dtype);
  torch::TensorOptions int_tensor_options = tensor_options.dtype(torch::kInt32);

  tokens_ = torch::zeros({max_tokens_per_batch}, int_tensor_options);

  if (args.rope_scaling_mrope_section().empty()) {
    positions_ = torch::zeros({max_tokens_per_batch}, int_tensor_options);
  } else {
    positions_ = torch::zeros({3, max_tokens_per_batch}, int_tensor_options);
    use_mrope_ = true;
  }

  new_cache_slots_ = torch::zeros({max_tokens_per_batch}, int_tensor_options);

  block_table_ = torch::zeros({max_seqs_per_batch, max_num_blocks_per_req},
                              int_tensor_options);

  q_seq_lens_ = torch::zeros({max_seqs_per_batch + 1}, int_tensor_options);
  kv_seq_lens_ = torch::zeros({max_seqs_per_batch + 1}, int_tensor_options);
  q_seq_lens_values_ = torch::zeros({max_seqs_per_batch}, int_tensor_options);
  kv_seq_lens_values_ = torch::zeros({max_seqs_per_batch}, int_tensor_options);

  output_ =
      torch::zeros({max_tokens_per_batch, args.hidden_size()}, tensor_options);

  paged_kv_indptr_ = torch::zeros({max_seqs_per_batch + 1}, int_tensor_options);

  paged_kv_indices_ = torch::zeros(
      {max_seqs_per_batch * max_num_blocks_per_req}, int_tensor_options);

  paged_kv_last_page_len_ =
      torch::zeros({max_seqs_per_batch}, int_tensor_options);

  decode_qo_indptr_ = torch::arange(
      0, max_seqs_per_batch + 1, torch::dtype(torch::kInt32).device(device));
}

std::optional<ModelInputParams> DcuGraphPersistentParam::update(
    const torch::Tensor& tokens,
    const torch::Tensor& positions,
    const ModelInputParams& params,
    uint32_t padded_num_tokens,
    bool return_capture_params,
    uint32_t graph_max_seq_len) {
  const uint32_t actual_num_tokens =
      static_cast<uint32_t>(tokens.size(/*dim=*/0));

  if (padded_num_tokens == 0) {
    padded_num_tokens = actual_num_tokens;
  }

  CHECK_GE(padded_num_tokens, actual_num_tokens)
      << "padded_num_tokens must be >= actual_num_tokens";

  const uint32_t padding_needed = padded_num_tokens - actual_num_tokens;

  std::optional<ModelInputParams> params_for_capture;
  if (return_capture_params) {
    params_for_capture = std::make_optional<ModelInputParams>(params);
  }

  std::shared_ptr<layer::AttentionMetadata> attn_metadata =
      std::make_shared<layer::AttentionMetadata>(
          layer::AttentionMetadataBuilder::build(params, args_.enable_mla()));
  CHECK(attn_metadata) << "attn_metadata should not be null";
  attn_metadata->enable_cuda_graph = true;
  if (params.meta.batch_forward_type.is_decode()) {
    // Flash decode receives max_seq_len as a scalar kernel argument. Full graph
    // replay cannot update that scalar, so capture with a bucketed upper bound
    // instead of the first decode step's exact kv_max_seq_len.
    attn_metadata->max_seq_len = graph_max_seq_len > 0
                                     ? graph_max_seq_len
                                     : args_.max_position_embeddings();
  }

  const auto build_capture_params_if_needed =
      [&]() -> std::optional<ModelInputParams> {
    if (!return_capture_params) {
      return std::nullopt;
    }

    CHECK(params_for_capture.has_value())
        << "params_for_capture should exist when return_capture_params=true";

    params_for_capture->enable_graph = true;
    params_for_capture->attn_metadata = attn_metadata;

    params_for_capture->attention.device.q_seq_lens =
        attn_metadata->q_cu_seq_lens;
    params_for_capture->attention.device.kv_seq_lens =
        attn_metadata->kv_cu_seq_lens;
    params_for_capture->attention.device.new_cache_slots =
        attn_metadata->slot_mapping;
    params_for_capture->attention.device.block_tables =
        attn_metadata->block_table;
    params_for_capture->attention.device.paged_kv_indptr =
        attn_metadata->paged_kv_indptr;
    params_for_capture->attention.device.paged_kv_indices =
        attn_metadata->paged_kv_indices;
    params_for_capture->attention.device.paged_kv_last_page_len =
        attn_metadata->paged_kv_last_page_len;

    if (params.embedding.input_embedding.defined()) {
      params_for_capture->embedding.input_embedding =
          input_embeds_.slice(0, 0, padded_num_tokens);
    }

    return params_for_capture;
  };

  tokens_.slice(0, 0, actual_num_tokens).copy_(tokens, /*non_blocking=*/true);
  if (padding_needed > 0) {
    tokens_.slice(0, actual_num_tokens, padded_num_tokens).fill_(0);
  }

  const int32_t slice_dim = use_mrope_ ? 1 : 0;
  positions_.slice(slice_dim, 0, positions.size(slice_dim))
      .copy_(positions, /*non_blocking=*/true);
  if (padding_needed > 0) {
    positions_.slice(slice_dim, actual_num_tokens, padded_num_tokens).fill_(0);
  }

  const uint32_t graph_batch_size = padded_num_tokens;

  std::vector<int32_t> q_seq_lens_vec(params.attention.host.q_seq_lens);
  std::vector<int32_t> kv_seq_lens_vec(params.attention.host.kv_seq_lens);

  if (!q_seq_lens_vec.empty()) {
    extend_q_cu_seq_lens(&q_seq_lens_vec, padding_needed, num_decoding_tokens_);

    auto q_seq_lens =
        make_int_tensor_on_device(q_seq_lens_vec, q_seq_lens_.options());

    q_seq_lens_.slice(0, 0, q_seq_lens.size(0))
        .copy_(q_seq_lens, /*non_blocking=*/true);

    attn_metadata->q_cu_seq_lens = q_seq_lens_.slice(0, 0, q_seq_lens.size(0));

    if (q_seq_lens.size(0) > 1) {
      auto q_seq_lens_diff_vec = compute_cu_seq_lens_diff(q_seq_lens_vec);
      auto q_seq_lens_diff = make_int_tensor_on_device(
          q_seq_lens_diff_vec, q_seq_lens_values_.options());
      q_seq_lens_values_.slice(0, 0, q_seq_lens_diff.size(0))
          .copy_(q_seq_lens_diff, /*non_blocking=*/true);
      attn_metadata->q_seq_lens = q_seq_lens_values(q_seq_lens_diff.size(0));
    }
  } else if (params.attention.device.q_seq_lens.defined()) {
    q_seq_lens_.slice(0, 0, params.attention.device.q_seq_lens.size(0))
        .copy_(params.attention.device.q_seq_lens, /*non_blocking=*/true);

    attn_metadata->q_cu_seq_lens =
        q_seq_lens_.slice(0, 0, params.attention.device.q_seq_lens.size(0));

    if (params.attention.device.q_seq_lens.size(0) > 1) {
      auto q_seq_lens_diff = torch::diff(attn_metadata->q_cu_seq_lens);
      q_seq_lens_values_.slice(0, 0, q_seq_lens_diff.size(0))
          .copy_(q_seq_lens_diff, /*non_blocking=*/true);
      attn_metadata->q_seq_lens = q_seq_lens_values(q_seq_lens_diff.size(0));
    }
  }

  if (!kv_seq_lens_vec.empty()) {
    extend_kv_cu_seq_lens_for_padding(&kv_seq_lens_vec, padding_needed);

    auto kv_seq_lens =
        make_int_tensor_on_device(kv_seq_lens_vec, kv_seq_lens_.options());

    kv_seq_lens_.slice(0, 0, kv_seq_lens.size(0))
        .copy_(kv_seq_lens, /*non_blocking=*/true);

    attn_metadata->kv_cu_seq_lens =
        kv_seq_lens_.slice(0, 0, kv_seq_lens.size(0));

    if (kv_seq_lens.size(0) > 1) {
      auto kv_seq_lens_diff_vec = compute_cu_seq_lens_diff(kv_seq_lens_vec);
      auto kv_seq_lens_diff = make_int_tensor_on_device(
          kv_seq_lens_diff_vec, kv_seq_lens_values_.options());
      kv_seq_lens_values_.slice(0, 0, kv_seq_lens_diff.size(0))
          .copy_(kv_seq_lens_diff, /*non_blocking=*/true);
      attn_metadata->kv_seq_lens = kv_seq_lens_values(kv_seq_lens_diff.size(0));
    }
  } else if (params.attention.device.kv_seq_lens.defined()) {
    kv_seq_lens_.slice(0, 0, params.attention.device.kv_seq_lens.size(0))
        .copy_(params.attention.device.kv_seq_lens, /*non_blocking=*/true);

    attn_metadata->kv_cu_seq_lens =
        kv_seq_lens_.slice(0, 0, params.attention.device.kv_seq_lens.size(0));

    if (params.attention.device.kv_seq_lens.size(0) > 1) {
      auto kv_seq_lens_diff = torch::diff(attn_metadata->kv_cu_seq_lens);
      kv_seq_lens_values_.slice(0, 0, kv_seq_lens_diff.size(0))
          .copy_(kv_seq_lens_diff, /*non_blocking=*/true);
      attn_metadata->kv_seq_lens = kv_seq_lens_values(kv_seq_lens_diff.size(0));
    }
  }

  if (params.attention.device.new_cache_slots.defined()) {
    const uint32_t src_size =
        static_cast<uint32_t>(params.attention.device.new_cache_slots.size(0));

    new_cache_slots_.slice(0, 0, src_size)
        .copy_(params.attention.device.new_cache_slots, /*non_blocking=*/true);

    if (padded_num_tokens > src_size) {
      new_cache_slots_.slice(0, src_size, padded_num_tokens).fill_(-1);
    }

    attn_metadata->slot_mapping = new_cache_slots(padded_num_tokens);
  }

  if (params.attention.device.block_tables.defined()) {
    const int64_t actual_rows = params.attention.device.block_tables.size(0);
    const int64_t actual_cols = params.attention.device.block_tables.size(1);

    auto block_table_slice =
        block_table_.slice(0, 0, actual_rows).slice(1, 0, actual_cols);

    block_table_slice.copy_(params.attention.device.block_tables,
                            /*non_blocking=*/true);

    if (static_cast<uint32_t>(actual_rows) < graph_batch_size) {
      block_table_.slice(0, actual_rows, graph_batch_size).fill_(0);
    }

    if (!attn_metadata->is_prefill || args_.enable_mla()) {
      attn_metadata->block_table = block_tables(graph_batch_size);
    }
  }

  if (params.embedding.input_embedding.defined()) {
    if (!input_embeds_.defined() || input_embeds_.numel() == 0) {
      auto shape = params.embedding.input_embedding.sizes().vec();
      shape[0] = SchedulerConfig::get_instance().max_tokens_per_batch();
      input_embeds_ = torch::zeros(
          shape, params.embedding.input_embedding.options().device(device_));
    }

    input_embeds_.slice(0, 0, params.embedding.input_embedding.size(0))
        .copy_(params.embedding.input_embedding, /*non_blocking=*/true);

    if (padding_needed > 0) {
      input_embeds_.slice(0, actual_num_tokens, padded_num_tokens).fill_(0);
    }
  }

  if (params.attention.device.paged_kv_indptr.defined()) {
    const int64_t indptr_size = params.attention.device.paged_kv_indptr.size(0);

    paged_kv_indptr_.slice(0, 0, indptr_size)
        .copy_(params.attention.device.paged_kv_indptr,
               /*non_blocking=*/true);

    if (graph_batch_size + 1 > static_cast<uint32_t>(indptr_size)) {
      auto last_value =
          paged_kv_indptr_.slice(0, indptr_size - 1, indptr_size).clone();

      for (uint32_t i = indptr_size; i < graph_batch_size + 1; ++i) {
        paged_kv_indptr_.slice(0, i, i + 1)
            .copy_(last_value, /*non_blocking=*/true);
      }
    }

    attn_metadata->paged_kv_indptr = paged_kv_indptr(graph_batch_size);
  }

  if (params.attention.device.paged_kv_indices.defined()) {
    const int64_t indices_size =
        params.attention.device.paged_kv_indices.size(0);

    paged_kv_indices_.slice(0, 0, indices_size)
        .copy_(params.attention.device.paged_kv_indices,
               /*non_blocking=*/true);

    attn_metadata->paged_kv_indices = paged_kv_indices_;
  }

  if (params.attention.device.paged_kv_last_page_len.defined()) {
    const int64_t len_size =
        params.attention.device.paged_kv_last_page_len.size(0);

    paged_kv_last_page_len_.slice(0, 0, len_size)
        .copy_(params.attention.device.paged_kv_last_page_len,
               /*non_blocking=*/true);

    if (graph_batch_size > static_cast<uint32_t>(len_size)) {
      paged_kv_last_page_len_.slice(0, len_size, graph_batch_size).fill_(0);
    }

    attn_metadata->paged_kv_last_page_len =
        paged_kv_last_page_len(graph_batch_size);
  }

  attn_metadata->qo_indptr = decode_qo_indptr(graph_batch_size);

  return build_capture_params_if_needed();
}

void DcuGraphPersistentParam::update_decode_input_buffer(
    const torch::Tensor& tokens,
    const torch::Tensor& positions,
    const ModelInputParams& params,
    uint32_t padded_num_tokens) {
  const uint32_t actual_num_tokens =
      static_cast<uint32_t>(tokens.size(/*dim=*/0));

  if (padded_num_tokens == 0) {
    padded_num_tokens = actual_num_tokens;
  }

  CHECK_GE(padded_num_tokens, actual_num_tokens)
      << "padded_num_tokens must be >= actual_num_tokens";

  const uint32_t padding_needed = padded_num_tokens - actual_num_tokens;

  tokens_.slice(0, 0, actual_num_tokens).copy_(tokens, /*non_blocking=*/true);
  if (padding_needed > 0) {
    tokens_.slice(0, actual_num_tokens, padded_num_tokens).fill_(0);
  }

  const int32_t slice_dim = use_mrope_ ? 1 : 0;
  positions_.slice(slice_dim, 0, positions.size(slice_dim))
      .copy_(positions, /*non_blocking=*/true);
  if (padding_needed > 0) {
    positions_.slice(slice_dim, actual_num_tokens, padded_num_tokens).fill_(0);
  }

  std::vector<int32_t> q_seq_lens_vec(params.attention.host.q_seq_lens);
  std::vector<int32_t> kv_seq_lens_vec(params.attention.host.kv_seq_lens);

  if (!q_seq_lens_vec.empty()) {
    extend_q_cu_seq_lens(&q_seq_lens_vec, padding_needed, num_decoding_tokens_);

    auto q_seq_lens =
        make_int_tensor_on_device(q_seq_lens_vec, q_seq_lens_.options());
    q_seq_lens_.slice(0, 0, q_seq_lens.size(0))
        .copy_(q_seq_lens, /*non_blocking=*/true);

    auto q_seq_lens_diff_vec = compute_cu_seq_lens_diff(q_seq_lens_vec);
    if (!q_seq_lens_diff_vec.empty()) {
      auto q_seq_lens_diff = make_int_tensor_on_device(
          q_seq_lens_diff_vec, q_seq_lens_values_.options());
      q_seq_lens_values_.slice(0, 0, q_seq_lens_diff.size(0))
          .copy_(q_seq_lens_diff, /*non_blocking=*/true);
    }
  } else if (params.attention.device.q_seq_lens.defined()) {
    q_seq_lens_.slice(0, 0, params.attention.device.q_seq_lens.size(0))
        .copy_(params.attention.device.q_seq_lens, /*non_blocking=*/true);

    if (params.attention.device.q_seq_lens.size(0) > 1) {
      auto q_seq_lens_diff = torch::diff(q_seq_lens(
          static_cast<uint32_t>(params.attention.device.q_seq_lens.size(0))));
      q_seq_lens_values_.slice(0, 0, q_seq_lens_diff.size(0))
          .copy_(q_seq_lens_diff, /*non_blocking=*/true);
    }
  }

  if (!kv_seq_lens_vec.empty()) {
    extend_kv_cu_seq_lens_for_padding(&kv_seq_lens_vec, padding_needed);

    auto kv_seq_lens =
        make_int_tensor_on_device(kv_seq_lens_vec, kv_seq_lens_.options());
    kv_seq_lens_.slice(0, 0, kv_seq_lens.size(0))
        .copy_(kv_seq_lens, /*non_blocking=*/true);

    auto kv_seq_lens_diff_vec = compute_cu_seq_lens_diff(kv_seq_lens_vec);
    if (!kv_seq_lens_diff_vec.empty()) {
      auto kv_seq_lens_diff = make_int_tensor_on_device(
          kv_seq_lens_diff_vec, kv_seq_lens_values_.options());
      kv_seq_lens_values_.slice(0, 0, kv_seq_lens_diff.size(0))
          .copy_(kv_seq_lens_diff, /*non_blocking=*/true);
    }
  } else if (params.attention.device.kv_seq_lens.defined()) {
    kv_seq_lens_.slice(0, 0, params.attention.device.kv_seq_lens.size(0))
        .copy_(params.attention.device.kv_seq_lens, /*non_blocking=*/true);

    if (params.attention.device.kv_seq_lens.size(0) > 1) {
      auto kv_seq_lens_diff = torch::diff(kv_seq_lens(
          static_cast<uint32_t>(params.attention.device.kv_seq_lens.size(0))));
      kv_seq_lens_values_.slice(0, 0, kv_seq_lens_diff.size(0))
          .copy_(kv_seq_lens_diff, /*non_blocking=*/true);
    }
  }

  if (params.attention.device.new_cache_slots.defined()) {
    const uint32_t src_size =
        static_cast<uint32_t>(params.attention.device.new_cache_slots.size(0));

    new_cache_slots_.slice(0, 0, src_size)
        .copy_(params.attention.device.new_cache_slots, /*non_blocking=*/true);

    if (padded_num_tokens > src_size) {
      new_cache_slots_.slice(0, src_size, padded_num_tokens).fill_(-1);
    }
  }

  if (params.attention.device.block_tables.defined()) {
    const int64_t actual_rows = params.attention.device.block_tables.size(0);
    const int64_t actual_cols = params.attention.device.block_tables.size(1);

    auto block_table_slice =
        block_table_.slice(0, 0, actual_rows).slice(1, 0, actual_cols);
    block_table_slice.copy_(params.attention.device.block_tables,
                            /*non_blocking=*/true);

    if (static_cast<uint32_t>(actual_rows) < padded_num_tokens) {
      block_table_.slice(0, actual_rows, padded_num_tokens).fill_(0);
    }
  }

  if (params.embedding.input_embedding.defined()) {
    if (!input_embeds_.defined() || input_embeds_.numel() == 0) {
      auto shape = params.embedding.input_embedding.sizes().vec();
      shape[0] = SchedulerConfig::get_instance().max_tokens_per_batch();
      input_embeds_ = torch::zeros(
          shape, params.embedding.input_embedding.options().device(device_));
    }

    input_embeds_.slice(0, 0, params.embedding.input_embedding.size(0))
        .copy_(params.embedding.input_embedding, /*non_blocking=*/true);

    if (padding_needed > 0) {
      input_embeds_.slice(0, actual_num_tokens, padded_num_tokens).fill_(0);
    }
  }
}

ModelInputParams DcuGraphPersistentParam::init_decode_params(
    const torch::Tensor& tokens,
    const torch::Tensor& positions,
    const ModelInputParams& params,
    uint32_t padded_num_tokens,
    uint32_t graph_max_seq_len) {
  update_decode_input_buffer(tokens, positions, params, padded_num_tokens);

  ModelInputParams decode_params = params;
  decode_params.enable_graph = true;
  decode_params.attention.device.q_seq_lens = q_seq_lens(padded_num_tokens + 1);
  decode_params.attention.device.kv_seq_lens =
      kv_seq_lens(padded_num_tokens + 1);
  decode_params.attention.device.new_cache_slots =
      new_cache_slots(padded_num_tokens);
  decode_params.attention.device.block_tables = block_tables(padded_num_tokens);

  if (params.embedding.input_embedding.defined()) {
    decode_params.embedding.input_embedding =
        input_embeds_.slice(0, 0, padded_num_tokens);
  }

  auto attn_metadata = std::make_shared<layer::AttentionMetadata>();
  attn_metadata->q_cu_seq_lens = decode_params.attention.device.q_seq_lens;
  attn_metadata->kv_cu_seq_lens = decode_params.attention.device.kv_seq_lens;
  attn_metadata->q_seq_lens = q_seq_lens_values(padded_num_tokens);
  attn_metadata->kv_seq_lens = kv_seq_lens_values(padded_num_tokens);
  attn_metadata->block_table = decode_params.attention.device.block_tables;
  attn_metadata->slot_mapping = decode_params.attention.device.new_cache_slots;
  attn_metadata->max_query_len = params.meta.q_max_seq_len;
  attn_metadata->max_seq_len = graph_max_seq_len > 0
                                   ? graph_max_seq_len
                                   : args_.max_position_embeddings();
  attn_metadata->compute_dtype = "float";
  attn_metadata->is_prefill = false;
  attn_metadata->is_chunked_prefill = false;
  attn_metadata->is_dummy = (params.meta.q_max_seq_len == 0);
  attn_metadata->is_causal = false;
  attn_metadata->enable_cuda_graph = true;

  if (!params.attention.host.kv_seq_lens.empty()) {
    const bool is_cu_seq_lens =
        params.attention.host.kv_seq_lens.size() ==
            static_cast<std::size_t>(params.meta.num_sequences + 1) &&
        params.attention.host.kv_seq_lens.front() == 0;
    attn_metadata->total_kv_len =
        is_cu_seq_lens
            ? params.attention.host.kv_seq_lens.back()
            : std::accumulate(params.attention.host.kv_seq_lens.begin(),
                              params.attention.host.kv_seq_lens.end(),
                              int64_t{0});
  }

  decode_params.attn_metadata = attn_metadata;
  return decode_params;
}

void DcuGraphPersistentParam::set_hidden_states(const torch::Tensor& value) {
  if (!value.defined()) {
    return;
  }

  const int64_t n_tokens = value.size(0);
  output_.slice(0, 0, n_tokens).copy_(value, /*non_blocking=*/true);
}

void DcuGraphPersistentParam::set_aux_hidden_states(
    const torch::Tensor& value) {
  if (!value.defined()) {
    return;
  }

  const int64_t n_tokens = value.size(0);

  if (!aux_hidden_states_.defined() || aux_hidden_states_.numel() == 0) {
    auto shape = value.sizes().vec();
    shape[0] = SchedulerConfig::get_instance().max_tokens_per_batch();
    aux_hidden_states_ = torch::zeros(shape, value.options().device(device_));
  }

  auto slice = aux_hidden_states_.slice(0, 0, n_tokens);
  if (slice.sizes() == value.sizes()) {
    slice.copy_(value, /*non_blocking=*/true);
  }
}

std::size_t DcuGraphPersistentParam::get_persistent_tensor_bytes() const {
  std::size_t total = 0;

  total += tensor_bytes(tokens_);
  total += tensor_bytes(positions_);
  total += tensor_bytes(new_cache_slots_);
  total += tensor_bytes(block_table_);
  total += tensor_bytes(q_seq_lens_);
  total += tensor_bytes(kv_seq_lens_);
  total += tensor_bytes(q_seq_lens_values_);
  total += tensor_bytes(kv_seq_lens_values_);

  total += tensor_bytes(paged_kv_indptr_);
  total += tensor_bytes(paged_kv_indices_);
  total += tensor_bytes(paged_kv_last_page_len_);
  total += tensor_bytes(decode_qo_indptr_);

  total += tensor_bytes(input_embeds_);
  total += tensor_bytes(output_);
  total += tensor_bytes(aux_hidden_states_);

  return total;
}

bool DcuGraph::capture(CausalLM* model,
                       const runtime::Options& options,
                       const torch::Tensor& tokens,
                       const torch::Tensor& positions,
                       const ModelInputParams& params,
                       std::vector<KVCache>& kv_cache,
                       uint32_t bucket_num_tokens,
                       const at::hip::MempoolId_t& pool,
                       bool use_piecewise,
                       uint32_t graph_max_seq_len) {
  padded_num_tokens_ = bucket_num_tokens;
  is_piecewise_ = use_piecewise;
  graph_max_seq_len_ = graph_max_seq_len;

  const uint32_t actual_num_tokens =
      static_cast<uint32_t>(tokens.size(/*dim=*/0));

  CHECK_GE(padded_num_tokens_, actual_num_tokens)
      << "bucket_num_tokens must be >= actual_num_tokens";

  auto original_stream = at::hip::getCurrentHIPStream(device_index_);
  auto capture_stream = capture_stream_;

  if (original_stream != capture_stream) {
    original_stream.synchronize();
    capture_stream.synchronize();
  }

  std::optional<c10::hip::HIPStreamGuard> stream_guard;
  stream_guard.emplace(capture_stream);

  std::optional<ModelInputParams> graph_params_opt;
  if (is_piecewise_) {
    graph_params_opt = persistent_param_.update(tokens,
                                                positions,
                                                params,
                                                padded_num_tokens_,
                                                /*return_capture_params=*/true,
                                                graph_max_seq_len);
  } else {
    graph_params_opt = persistent_param_.init_decode_params(
        tokens, positions, params, padded_num_tokens_, graph_max_seq_len);
  }

  CHECK(graph_params_opt.has_value())
      << "DcuGraphPersistentParam::update should return ModelInputParams "
         "during capture";

  VLOG(1) << "DCU graph capture begin, bucket_num_tokens=" << bucket_num_tokens
          << ", actual_num_tokens=" << actual_num_tokens
          << ", graph_max_seq_len=" << graph_max_seq_len
          << ", piecewise=" << is_piecewise_;

  if (is_piecewise_) {
    // Warmup: execute forward once without capture to initialize hipBLAS
    // handles and other HIP resources that cannot be created during capture.
    model->forward(persistent_param_.persistent_tokens(padded_num_tokens_),
                   persistent_param_.persistent_positions(padded_num_tokens_),
                   kv_cache,
                   graph_params_opt.value());

    auto& capture = ::xllm::runtime::dcu::GlobalCaptureInstance::get_instance();

    capture.begin_capture(pool);

    auto forward_result = model->forward(
        persistent_param_.persistent_tokens(padded_num_tokens_),
        persistent_param_.persistent_positions(padded_num_tokens_),
        kv_cache,
        graph_params_opt.value());

    persistent_param_.set_hidden_states(forward_result.hidden_states);

    if (options.enable_graph_aux_hidden_states() &&
        forward_result.aux_hidden_states.defined()) {
      persistent_param_.set_aux_hidden_states(forward_result.aux_hidden_states);
    }

    piecewise_graph_ = capture.end_capture();
    CHECK(piecewise_graph_ != nullptr)
        << "DCU piecewise graph capture returned null";
  } else {
    graph_.capture_begin(pool, hipStreamCaptureModeThreadLocal);

    auto forward_result = model->forward(
        persistent_param_.persistent_tokens(padded_num_tokens_),
        persistent_param_.persistent_positions(padded_num_tokens_),
        kv_cache,
        graph_params_opt.value());

    persistent_param_.set_hidden_states(forward_result.hidden_states);

    if (options.enable_graph_aux_hidden_states() &&
        forward_result.aux_hidden_states.defined()) {
      persistent_param_.set_aux_hidden_states(forward_result.aux_hidden_states);
    }

    graph_.capture_end();
  }

  capture_stream.synchronize();
  stream_guard.reset();

  VLOG(1) << "DCU graph capture end, bucket_num_tokens=" << bucket_num_tokens
          << ", piecewise=" << is_piecewise_;

  return true;
}

ModelOutput DcuGraph::replay(const torch::Tensor& tokens,
                             const torch::Tensor& positions,
                             std::vector<KVCache>& kv_cache,
                             const ModelInputParams& params) {
  const uint32_t actual_num_tokens =
      static_cast<uint32_t>(tokens.size(/*dim=*/0));

  CHECK_LE(actual_num_tokens, padded_num_tokens_)
      << "actual_num_tokens must be <= padded_num_tokens_";

  if (is_piecewise_) {
    auto updated_params =
        persistent_param_.update(tokens,
                                 positions,
                                 params,
                                 padded_num_tokens_,
                                 /*return_capture_params=*/true,
                                 graph_max_seq_len_);

    CHECK(piecewise_graph_ != nullptr)
        << "piecewise_graph_ should not be null for piecewise replay";
    CHECK(updated_params.has_value())
        << "update() should return ModelInputParams for piecewise replay";
    CHECK(updated_params->attn_metadata)
        << "attn_metadata is required for piecewise replay";

    ::xllm::kernel::dcu::AttentionReplayParams runner_params;
    runner_params.actual_num_tokens = actual_num_tokens;
    runner_params.attn_metadata = updated_params->attn_metadata;
    piecewise_graph_->replay(runner_params);
  } else {
    persistent_param_.update_decode_input_buffer(
        tokens, positions, params, padded_num_tokens_);
    graph_.replay();
  }

  return ModelOutput(get_hidden_states(actual_num_tokens));
}

DcuGraphExecutorImpl::DcuGraphExecutorImpl(CausalLM* model,
                                           const ModelArgs& args,
                                           const torch::Device& device,
                                           const runtime::Options& options)
    : model_(model),
      args_(args),
      device_(device),
      options_(options),
      graph_pool_(at::cuda::graph_pool_handle()) {
  max_tokens_for_graph_mode_ =
      ::xllm::ExecutionConfig::get_instance().max_tokens_for_graph_mode();
  if (max_tokens_for_graph_mode_ < options_.max_seqs_per_batch()) {
    max_tokens_for_graph_mode_ = options_.max_seqs_per_batch();
  }

  persistent_param_ =
      std::make_unique<DcuGraphPersistentParam>(args_, device_, options_);

  const std::size_t persistent_bytes =
      persistent_param_->get_persistent_tensor_bytes();

  VLOG(1) << "DCU graph persistent tensor total size: " << persistent_bytes
          << " bytes (" << (persistent_bytes / (1024 * 1024)) << " MB)";
}

DcuGraphExecutorImpl::~DcuGraphExecutorImpl() {
  prefill_graphs_.clear();
  graphs_.clear();
}

ForwardInput DcuGraphExecutorImpl::prepare_inputs(Batch& batch) {
  return batch.prepare_forward_input(
      options_.num_decoding_tokens(), 0, args_, options_.cp_size());
}

uint32_t DcuGraphExecutorImpl::get_bucket_num_tokens(
    uint32_t num_tokens) const {
  if (::xllm::ExecutionConfig::get_instance()
          .enable_graph_mode_decode_no_padding()) {
    return num_tokens;
  }

  if (num_tokens <= 1) {
    return 1;
  }

  if (num_tokens <= 2) {
    return 2;
  }

  if (num_tokens <= 4) {
    return 4;
  }

  if (num_tokens <= 8) {
    return 8;
  }

  constexpr uint32_t kGraphStep = 16;
  return ((num_tokens + kGraphStep - 1) / kGraphStep) * kGraphStep;
}

uint32_t DcuGraphExecutorImpl::get_graph_max_seq_len(
    uint32_t kv_max_seq_len) const {
  constexpr uint32_t kSmallSeqLenThreshold = 4096;
  constexpr uint32_t kSmallSeqLenStep = 1024;
  constexpr uint32_t kLargeSeqLenStep = 4096;

  const uint32_t max_seq_len =
      static_cast<uint32_t>(args_.max_position_embeddings());
  if (kv_max_seq_len == 0 || kv_max_seq_len >= max_seq_len) {
    return max_seq_len;
  }

  // Keep early decode reasonably tight, but avoid repeatedly recapturing long
  // CoT generations every 128 tokens.
  const uint32_t graph_seq_len_step = kv_max_seq_len <= kSmallSeqLenThreshold
                                          ? kSmallSeqLenStep
                                          : kLargeSeqLenStep;
  const uint32_t bucket =
      ((kv_max_seq_len + graph_seq_len_step - 1) / graph_seq_len_step) *
      graph_seq_len_step;
  return std::min(bucket, max_seq_len);
}

uint32_t DcuGraphExecutorImpl::get_graph_shape_id(
    uint32_t bucket_num_tokens,
    uint32_t graph_max_seq_len) const {
  constexpr uint32_t kTokenBits = 12;
  constexpr uint32_t kMaxBucketNumTokens = (1u << kTokenBits) - 1;
  CHECK_LE(bucket_num_tokens, kMaxBucketNumTokens)
      << "bucket_num_tokens is too large for DCU graph shape id";

  return (graph_max_seq_len << kTokenBits) | bucket_num_tokens;
}

ModelOutput DcuGraphExecutorImpl::attach_aux_hidden_states_if_needed(
    const torch::Tensor& hidden_states,
    uint32_t n_tokens) const {
  if (options_.enable_graph_aux_hidden_states()) {
    auto aux_hidden_states = persistent_param_->aux_hidden_states(n_tokens);
    if (aux_hidden_states.defined() && aux_hidden_states.numel() > 0) {
      return ModelOutput(hidden_states, torch::Tensor(), aux_hidden_states);
    }
  }

  return ModelOutput(hidden_states);
}

DcuStream DcuGraphExecutorImpl::get_capture_stream(
    c10::DeviceIndex device_index) {
  thread_local DcuStream thread_capture_stream =
      at::hip::getStreamFromPool(/*isHighPriority=*/true, device_index);

  thread_local bool initialized = false;
  if (!initialized) {
    VLOG(1) << "Initialized DCU graph capture stream for thread "
            << std::this_thread::get_id() << ", device_index=" << device_index;
    initialized = true;
  }

  return thread_capture_stream;
}

ModelOutput DcuGraphExecutorImpl::run(const torch::Tensor& tokens,
                                      const torch::Tensor& positions,
                                      std::vector<KVCache>& kv_caches,
                                      const ModelInputParams& params) {
  const bool is_prefill = params.meta.batch_forward_type.is_prefill();
  const bool is_decode = params.meta.batch_forward_type.is_decode();

  VLOG(1) << "DCU executor run: "
          << "is_prefill=" << is_prefill << ", is_decode=" << is_decode
          << ", enable_graph="
          << ::xllm::ExecutionConfig::get_instance().enable_graph()
          << ", enable_prefill_piecewise_graph="
          << ::xllm::ExecutionConfig::get_instance()
                 .enable_prefill_piecewise_graph()
          << ", num_tokens=" << tokens.size(0);

  uint32_t actual_num_tokens = static_cast<uint32_t>(tokens.size(/*dim=*/0));

  bool graph_mode =
      is_decode && ::xllm::ExecutionConfig::get_instance().enable_graph();

  if (params.parallel.dp_global_token_nums.size() > 1) {
    auto max_it = std::max_element(params.parallel.dp_global_token_nums.begin(),
                                   params.parallel.dp_global_token_nums.end());
    CHECK(max_it != params.parallel.dp_global_token_nums.end());

    actual_num_tokens = static_cast<uint32_t>(*max_it);

    const auto& dp_is_decode = params.parallel.dp_is_decode;
    CHECK_EQ(dp_is_decode.size(), params.parallel.dp_global_token_nums.size());

    graph_mode = ::xllm::ExecutionConfig::get_instance().enable_graph() &&
                 std::find(dp_is_decode.begin(), dp_is_decode.end(), 0) ==
                     dp_is_decode.end();
  }

  const bool enable_prefill_piecewise =
      ::xllm::ExecutionConfig::get_instance().enable_prefill_piecewise_graph();

  if (is_prefill && enable_prefill_piecewise) {
    if (static_cast<int64_t>(actual_num_tokens) > max_tokens_for_graph_mode_) {
      VLOG(1) << "DCU prefill token count " << actual_num_tokens
              << " exceeds max_tokens_for_graph_mode ("
              << max_tokens_for_graph_mode_ << "), falling back to eager";
      COUNTER_INC(num_model_execution_total_eager);
      return model_->forward(tokens, positions, kv_caches, params);
    }
    const uint32_t bucket_num_tokens = actual_num_tokens;
    const uint32_t local_actual_tokens =
        static_cast<uint32_t>(tokens.size(/*dim=*/0));

    auto it = prefill_graphs_.find(bucket_num_tokens);
    if (it != prefill_graphs_.end()) {
      VLOG(1) << "DCU prefill piecewise graph replay, bucket_num_tokens="
              << bucket_num_tokens
              << ", actual_num_tokens=" << local_actual_tokens;

      auto result = it->second->replay(tokens, positions, kv_caches, params);
      auto hidden_states =
          result.hidden_states.slice(0, 0, local_actual_tokens);

      return attach_aux_hidden_states_if_needed(hidden_states,
                                                local_actual_tokens);
    }

    VLOG(1) << "DCU prefill piecewise graph capture, bucket_num_tokens="
            << bucket_num_tokens
            << ", actual_num_tokens=" << local_actual_tokens;

    auto graph =
        std::make_unique<DcuGraph>(*persistent_param_,
                                   device_.index(),
                                   get_capture_stream(device_.index()));

    const bool capture_success = graph->capture(model_,
                                                options_,
                                                tokens,
                                                positions,
                                                params,
                                                kv_caches,
                                                bucket_num_tokens,
                                                graph_pool_,
                                                /*use_piecewise=*/true);

    if (!capture_success) {
      LOG(WARNING) << "Failed to capture DCU prefill piecewise graph for "
                   << "bucket_num_tokens=" << bucket_num_tokens
                   << ". Falling back to eager.";
      COUNTER_INC(num_model_execution_total_eager);
      return model_->forward(tokens, positions, kv_caches, params);
    }

    prefill_graphs_[bucket_num_tokens] = std::move(graph);

    auto result = prefill_graphs_[bucket_num_tokens]->replay(
        tokens, positions, kv_caches, params);

    auto hidden_states = result.hidden_states.slice(0, 0, local_actual_tokens);

    return attach_aux_hidden_states_if_needed(hidden_states,
                                              local_actual_tokens);
  }

  if (!graph_mode || is_prefill) {
    COUNTER_INC(num_model_execution_total_eager);
    return model_->forward(tokens, positions, kv_caches, params);
  }

  if (params.has_llmrec_params()) {
    VLOG(1) << "DCU graph does not support LLMRec/xAttention yet; "
            << "falling back to eager.";
    COUNTER_INC(num_model_execution_total_eager);
    return model_->forward(tokens, positions, kv_caches, params);
  }

  const int64_t max_seq_len = args_.max_position_embeddings();
  if (params.meta.kv_max_seq_len > max_seq_len) {
    LOG(WARNING) << "Not suitable for DCU graph: kv_max_seq_len="
                 << params.meta.kv_max_seq_len
                 << ", max_seq_len=" << max_seq_len
                 << ". Falling back to eager.";
    COUNTER_INC(num_model_execution_total_eager);
    return model_->forward(tokens, positions, kv_caches, params);
  }

  const uint32_t bucket_num_tokens = get_bucket_num_tokens(actual_num_tokens);
  const uint32_t graph_max_seq_len =
      get_graph_max_seq_len(static_cast<uint32_t>(params.meta.kv_max_seq_len));
  const uint32_t graph_shape_id =
      get_graph_shape_id(bucket_num_tokens, graph_max_seq_len);
  const uint32_t local_actual_tokens =
      static_cast<uint32_t>(tokens.size(/*dim=*/0));

  auto it = graphs_.find(graph_shape_id);
  if (it != graphs_.end()) {
    VLOG(1) << "DCU graph replay, bucket_num_tokens=" << bucket_num_tokens
            << ", graph_max_seq_len=" << graph_max_seq_len
            << ", actual_num_tokens=" << local_actual_tokens;

    auto result = it->second->replay(tokens, positions, kv_caches, params);
    auto hidden_states = result.hidden_states.slice(0, 0, local_actual_tokens);

    return attach_aux_hidden_states_if_needed(hidden_states,
                                              local_actual_tokens);
  }

  VLOG(1) << "DCU graph capture, bucket_num_tokens=" << bucket_num_tokens
          << ", graph_max_seq_len=" << graph_max_seq_len
          << ", actual_num_tokens=" << local_actual_tokens;

  auto graph = std::make_unique<DcuGraph>(
      *persistent_param_, device_.index(), get_capture_stream(device_.index()));

  const bool capture_success = graph->capture(model_,
                                              options_,
                                              tokens,
                                              positions,
                                              params,
                                              kv_caches,
                                              bucket_num_tokens,
                                              graph_pool_,
                                              /*use_piecewise=*/false,
                                              graph_max_seq_len);

  if (!capture_success) {
    LOG(WARNING) << "Failed to capture DCU graph for bucket_num_tokens="
                 << bucket_num_tokens << ". Falling back to eager.";
    COUNTER_INC(num_model_execution_total_eager);
    return model_->forward(tokens, positions, kv_caches, params);
  }

  graphs_[graph_shape_id] = std::move(graph);

  auto result =
      graphs_[graph_shape_id]->replay(tokens, positions, kv_caches, params);

  auto hidden_states = result.hidden_states.slice(0, 0, local_actual_tokens);

  return attach_aux_hidden_states_if_needed(hidden_states, local_actual_tokens);
}

}  // namespace xllm::runtime::dcu

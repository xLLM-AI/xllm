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

#include "cuda_graph_executor_impl.h"

#include <c10/core/Device.h>
#include <c10/core/TensorOptions.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAGuard.h>
#include <glog/logging.h>
#include <torch/torch.h>

#include <algorithm>
#include <numeric>
#include <shared_mutex>
#include <unordered_map>

#include "core/common/global_flags.h"
#include "core/common/metrics.h"
#include "core/framework/config/execution_config.h"
#include "core/framework/config/rec_config.h"
#include "core/layers/common/attention_metadata.h"
#include "core/layers/common/attention_metadata_builder.h"
#include "core/layers/cuda/flashinfer_planinfo.h"
#include "core/layers/cuda/xattention_planinfo.h"
#include "core/platform/cuda/device_capture_lock.h"
#include "core/platform/device.h"
#include "core/platform/shared_vmm_allocator.h"
#include "core/platform/stream.h"
#include "core/platform/vmm_torch_allocator.h"
#include "core/util/rec_model_utils.h"
#include "core/util/utils.h"
#include "kernels/cuda/global_capture_instance.h"
#include "kernels/cuda/utils.h"

namespace xllm::runtime::cuda {

namespace {

struct GraphPoolMemoryUsage {
  size_t reserved_bytes = 0;
  size_t allocated_bytes = 0;
  size_t active_bytes = 0;
};

GraphPoolMemoryUsage get_graph_pool_memory_usage(
    c10::DeviceIndex device_index,
    const at::cuda::MempoolId_t& pool_id) {
  GraphPoolMemoryUsage usage;
  const auto snapshot = c10::cuda::CUDACachingAllocator::snapshot();
  for (const auto& segment : snapshot.segments) {
    if (segment.device != device_index ||
        segment.owner_private_pool_id != pool_id) {
      continue;
    }
    usage.reserved_bytes += segment.total_size;
    usage.allocated_bytes += segment.allocated_size;
    usage.active_bytes += segment.active_size;
  }
  return usage;
}

GraphPoolMemoryUsage get_private_pools_memory_usage(
    c10::DeviceIndex device_index) {
  GraphPoolMemoryUsage usage;
  const auto snapshot = c10::cuda::CUDACachingAllocator::snapshot();
  for (const auto& segment : snapshot.segments) {
    if (segment.device != device_index) {
      continue;
    }
    if (segment.owner_private_pool_id == at::cuda::MempoolId_t{0, 0}) {
      continue;
    }
    usage.reserved_bytes += segment.total_size;
    usage.allocated_bytes += segment.allocated_size;
    usage.active_bytes += segment.active_size;
  }
  return usage;
}

size_t get_allocator_reserved_bytes(c10::DeviceIndex device_index) {
  const auto device_stats =
      c10::cuda::CUDACachingAllocator::getDeviceStats(device_index);
  const size_t stat_index =
      static_cast<size_t>(c10::CachingAllocator::StatType::AGGREGATE);
  return static_cast<size_t>(device_stats.reserved_bytes[stat_index].current);
}

bool is_cuda_contiguous_int32_tensor(const torch::Tensor& tensor) {
  return tensor.defined() && tensor.is_cuda() &&
         tensor.scalar_type() == torch::kInt32 && tensor.is_contiguous();
}

int64_t get_graph_bucket_num_tokens(int64_t num_tokens, bool is_prefill) {
  // no_padding only works for decode, prefill requires padding for graph reuse
  if (::xllm::ExecutionConfig::get_instance()
          .enable_graph_mode_decode_no_padding() &&
      !is_prefill) {
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
  return ((num_tokens + 15) / 16) * 16;
}

}  // namespace

// CudaGraphPersistentParam implementation
CudaGraphPersistentParam::CudaGraphPersistentParam(
    const ModelArgs& args,
    const torch::Device& device,
    const runtime::Options& options)
    : args_(args), device_(device), options_(options) {
  // Use max_tokens_per_batch for first dimension size
  const int64_t max_tokens_per_batch = options.max_tokens_per_batch();
  // Round the sequence capacity to the same bucket used by graph execution so
  // the largest configured decode batch can be padded safely.
  int64_t max_seqs_per_batch;
  if (is_rec_multi_round_mode()) {
    // max_seqs_per_batch is the max sequence count per Batch in a scheduler
    // group.
    // When is_rec_multi_round_mode() == true, multiply by beam_width to account
    // for beam search.
    max_seqs_per_batch = options.max_seqs_per_batch() * options_.beam_width();
  } else {
    max_seqs_per_batch = options.max_seqs_per_batch();
  }
  max_seqs_per_batch =
      get_graph_bucket_num_tokens(max_seqs_per_batch, /*is_prefill=*/false);
  auto tensor_options = torch::TensorOptions().device(device);

  const int64_t max_seq_len = args_.max_position_embeddings();

  // Create persistent tensors with max_tokens_per_batch as first dimension
  persistent_tokens_ = torch::zeros({max_tokens_per_batch},
                                    torch::dtype(torch::kInt).device(device));
  persistent_positions_ = torch::zeros(
      {max_tokens_per_batch}, torch::dtype(torch::kInt).device(device));
  persistent_new_cache_slots_ = torch::zeros(
      {max_tokens_per_batch}, torch::dtype(torch::kInt).device(device));
  persistent_linear_state_indices_ = torch::zeros(
      {max_seqs_per_batch}, torch::dtype(torch::kInt).device(device));

  // q_seq_lens is q_cu_seq_lens in GPU Model.
  // kv_seq_lens is kv_cu_seq_lens in GPU Model.
  q_seq_lens_ = torch::zeros({max_seqs_per_batch + 1},
                             torch::dtype(torch::kInt).device(device));
  kv_seq_lens_ = torch::zeros({max_seqs_per_batch + 1},
                              torch::dtype(torch::kInt).device(device));

  // Block table tensors with maximum possible size
  const auto block_size = options.block_size();
  const int64_t max_block_table_len =
      (max_seq_len + block_size - 1) / block_size + 1;
  persistent_block_tables_ =
      torch::zeros({max_seqs_per_batch, max_block_table_len},
                   torch::dtype(torch::kInt).device(device));

  // Output tensor for hidden states
  torch::ScalarType dtype = util::parse_dtype(args.dtype(), device);
  if (args.dtype() == "float" || args.dtype() == "float32") {
    LOG(WARNING)
        << "Cuda graph executor init hidden_states compatible with float32 "
           "dtype: float32. This should not happen in production but for test.";
    dtype = torch::kFloat32;
  }
  hidden_states_ = torch::zeros({max_tokens_per_batch, args.hidden_size()},
                                torch::dtype(dtype).device(device));

  // FlashInfer decode mode parameters
  // paged_kv_indptr: shape [max_seqs_per_batch + 1]
  persistent_paged_kv_indptr_ = torch::zeros(
      {max_seqs_per_batch + 1}, torch::dtype(torch::kInt).device(device));

  // paged_kv_indices: maximum size based on max blocks
  // Estimate max blocks: max_seqs_per_batch * max_block_table_len
  const int64_t max_paged_kv_indices_size =
      max_seqs_per_batch * max_block_table_len;
  persistent_paged_kv_indices_ = torch::zeros(
      {max_paged_kv_indices_size}, torch::dtype(torch::kInt).device(device));

  // paged_kv_last_page_len: shape [max_seqs_per_batch]
  persistent_paged_kv_last_page_len_ = torch::zeros(
      {max_seqs_per_batch}, torch::dtype(torch::kInt).device(device));

  // For decode mode, each sequence has 1 token, so qo_indptr = [0, 1, 2, ...,
  // max_seqs_per_batch]
  persistent_decode_qo_indptr_ = torch::arange(
      0, max_seqs_per_batch + 1, torch::dtype(torch::kInt).device(device));
  persistent_kv_seq_lens_delta_ = torch::zeros(
      {max_seqs_per_batch}, torch::dtype(torch::kInt).device(device));
  // will be updated by q_cu_seq_lens, q_cu_seq_lens is the cumulative sum of
  // q_seq_lens
  persistent_chunked_prefill_qo_indptr_ = torch::zeros(
      {max_seqs_per_batch + 1}, torch::dtype(torch::kInt).device(device));
  // aux_hidden_states will be lazily initialized when needed
}

bool CudaGraphPersistentParam::can_use_llm_decode_fast_path(
    const torch::Tensor& tokens,
    const torch::Tensor& positions,
    const ModelInputParams& params) const {
  if (!params.meta.batch_forward_type.is_decode() ||
      is_rec_multi_round_mode() || params.has_llmrec_params() ||
      params.embedding.input_embedding.defined()) {
    return false;
  }
  return is_cuda_contiguous_int32_tensor(tokens) &&
         is_cuda_contiguous_int32_tensor(positions) &&
         is_cuda_contiguous_int32_tensor(
             params.attention.device.new_cache_slots) &&
         is_cuda_contiguous_int32_tensor(params.attention.device.kv_seq_lens) &&
         is_cuda_contiguous_int32_tensor(
             params.attention.device.paged_kv_indptr) &&
         is_cuda_contiguous_int32_tensor(
             params.attention.device.paged_kv_indices) &&
         is_cuda_contiguous_int32_tensor(
             params.attention.device.paged_kv_last_page_len);
}

void CudaGraphPersistentParam::update_llm_decode_metadata_fast_path(
    const torch::Tensor& tokens,
    const torch::Tensor& positions,
    const ModelInputParams& params,
    uint32_t padded_num_tokens,
    int64_t actual_batch_size,
    int64_t actual_num_tokens) {
  CHECK_GE(actual_batch_size, 0) << "actual_batch_size must be >= 0";
  CHECK_GE(actual_num_tokens, 0) << "actual_num_tokens must be >= 0";
  const int64_t actual_indices_size =
      params.attention.device.paged_kv_indices.size(0);
  xllm::kernel::cuda::LlmDecodeMetadataUpdateParams update_params{
      .src_tokens = tokens.data_ptr<int32_t>(),
      .src_positions = positions.data_ptr<int32_t>(),
      .src_new_cache_slots =
          params.attention.device.new_cache_slots.data_ptr<int32_t>(),
      .src_kv_seq_lens =
          params.attention.device.kv_seq_lens.data_ptr<int32_t>(),
      .src_paged_kv_indptr =
          params.attention.device.paged_kv_indptr.data_ptr<int32_t>(),
      .src_paged_kv_indices =
          params.attention.device.paged_kv_indices.data_ptr<int32_t>(),
      .src_paged_kv_last_page_len =
          params.attention.device.paged_kv_last_page_len.data_ptr<int32_t>(),
      .dst_tokens = persistent_tokens_.data_ptr<int32_t>(),
      .dst_positions = persistent_positions_.data_ptr<int32_t>(),
      .dst_new_cache_slots = persistent_new_cache_slots_.data_ptr<int32_t>(),
      .dst_kv_seq_lens = kv_seq_lens_.data_ptr<int32_t>(),
      .dst_kv_seq_lens_delta =
          persistent_kv_seq_lens_delta_.data_ptr<int32_t>(),
      .dst_paged_kv_indptr = persistent_paged_kv_indptr_.data_ptr<int32_t>(),
      .dst_paged_kv_indices = persistent_paged_kv_indices_.data_ptr<int32_t>(),
      .dst_paged_kv_last_page_len =
          persistent_paged_kv_last_page_len_.data_ptr<int32_t>(),
      .actual_num_tokens = actual_num_tokens,
      .padded_num_tokens = static_cast<int64_t>(padded_num_tokens),
      .actual_batch_size = actual_batch_size,
      .actual_indices_size = actual_indices_size,
  };
  const cudaStream_t stream = c10::cuda::getCurrentCUDAStream(device_.index());
  xllm::kernel::cuda::update_llm_decode_metadata(update_params, stream);
}

void CudaGraphPersistentParam::set_aux_hidden_states(
    const torch::Tensor& value) {
  if (!value.defined()) {
    return;
  }
  const uint32_t result_tokens = value.size(0);
  if (aux_hidden_states_.numel() == 0) {
    // Lazy initialization: create aux_hidden_states tensor if not already
    // created
    const int64_t max_tokens_per_batch = options_.max_tokens_per_batch();
    auto shape = value.sizes().vec();
    shape[0] = max_tokens_per_batch;
    torch::ScalarType dtype = util::parse_dtype(args_.dtype(), device_);
    if (args_.dtype() == "float" || args_.dtype() == "float32") {
      dtype = torch::kFloat32;
    }
    aux_hidden_states_ =
        torch::zeros(shape, torch::dtype(dtype).device(device_));
  }
  // Slice to match the actual shape
  auto slice =
      aux_hidden_states_.slice(/*dim=*/0, /*start=*/0, /*end=*/result_tokens);
  // Reshape slice if needed to match value shape
  if (slice.sizes() == value.sizes()) {
    slice.copy_(value, /*non_blocking=*/true);
  }
}

size_t CudaGraphPersistentParam::get_persistent_tensor_bytes() const {
  auto bytes = [](const torch::Tensor& t) {
    return t.defined() ? static_cast<size_t>(t.numel()) * t.element_size() : 0;
  };
  size_t total = 0;
  total += bytes(persistent_tokens_);
  total += bytes(persistent_positions_);
  total += bytes(persistent_new_cache_slots_);
  total += bytes(persistent_linear_state_indices_);
  total += bytes(persistent_block_tables_);
  total += bytes(hidden_states_);
  total += bytes(q_seq_lens_);
  total += bytes(kv_seq_lens_);
  total += bytes(persistent_embedding_);
  total += bytes(aux_hidden_states_);
  total += bytes(persistent_paged_kv_indptr_);
  total += bytes(persistent_paged_kv_indices_);
  total += bytes(persistent_paged_kv_last_page_len_);
  total += bytes(persistent_decode_qo_indptr_);
  total += bytes(persistent_kv_seq_lens_delta_);
  total += bytes(persistent_chunked_prefill_qo_indptr_);
  return total;
}

std::optional<ModelInputParams> CudaGraphPersistentParam::update(
    const torch::Tensor& tokens,
    const torch::Tensor& k_cache,
    const torch::Tensor& v_cache,
    const torch::Tensor& positions,
    const ModelInputParams& params,
    uint32_t padded_num_tokens,
    bool return_capture_params) {
  std::optional<ModelInputParams> params_for_capture;
  if (return_capture_params) {
    CHECK_GT(padded_num_tokens, 0)
        << "padded_num_tokens must be > 0 when return_capture_params is true";
    params_for_capture = std::make_optional<ModelInputParams>(params);
  }
  // Build attn_metadata with original model_input_params. So we can set actual
  // batch size in plan_info.
  std::shared_ptr<layer::AttentionMetadata> attn_metadata =
      std::make_shared<layer::AttentionMetadata>(
          layer::AttentionMetadataBuilder::build(params, args_.enable_mla()));
  CHECK(attn_metadata) << "attn_metadata should not be null";
  attn_metadata->enable_cuda_graph = true;
  auto build_capture_params_if_needed =
      [&]() -> std::optional<ModelInputParams> {
    if (!return_capture_params) {
      return std::nullopt;
    }
    CHECK(params_for_capture.has_value())
        << "params_for_capture should be initialized when "
           "return_capture_params "
           "is true";
    if (params.embedding.input_embedding.defined()) {
      params_for_capture->embedding.input_embedding =
          persistent_embedding(padded_num_tokens);
    }
    if (!params.embedding.linear_state_ids.empty()) {
      params_for_capture->embedding.linear_state_ids =
          params.embedding.linear_state_ids;
      params_for_capture->embedding.linear_state_indices =
          persistent_linear_state_indices(params.meta.num_sequences);
    }
    params_for_capture->attn_metadata = attn_metadata;
    return params_for_capture;
  };

  const uint32_t actual_num_tokens = tokens.size(0);
  const int64_t actual_batch_size = params.meta.num_sequences;
  const bool use_llm_decode_fast_path =
      can_use_llm_decode_fast_path(tokens, positions, params);

  // Copy data from input parameters to persistent graph tensors
  if (use_llm_decode_fast_path) {
    VLOG(kGraphExecutorLogVerboseLevel)
        << "use fast path for LLM decode metadata update";
    update_llm_decode_metadata_fast_path(tokens,
                                         positions,
                                         params,
                                         padded_num_tokens,
                                         actual_batch_size,
                                         actual_num_tokens);
  } else {
    VLOG(kGraphExecutorLogVerboseLevel)
        << "copy_ tokens: src shape=" << tokens.sizes() << ", dst slice shape=["
        << actual_num_tokens << "]";
    persistent_tokens_.slice(/*dim=*/0, /*start=*/0, /*end=*/actual_num_tokens)
        .copy_(tokens, /*non_blocking=*/true);

    if (padded_num_tokens > actual_num_tokens) {
      VLOG(kGraphExecutorLogVerboseLevel)
          << "fill_ tokens padding: [" << actual_num_tokens << ", "
          << padded_num_tokens << "] with 0";
      persistent_tokens_
          .slice(/*dim=*/0,
                 /*start=*/actual_num_tokens,
                 /*end=*/padded_num_tokens)
          .fill_(0);
    }

    VLOG(kGraphExecutorLogVerboseLevel)
        << "copy_ positions: src shape=" << positions.sizes()
        << ", dst slice shape=[" << actual_num_tokens << "]";
    persistent_positions_
        .slice(/*dim=*/0, /*start=*/0, /*end=*/actual_num_tokens)
        .copy_(positions, /*non_blocking=*/true);
    if (padded_num_tokens > actual_num_tokens) {
      persistent_positions_
          .slice(/*dim=*/0,
                 /*start=*/actual_num_tokens,
                 /*end=*/padded_num_tokens)
          .fill_(0);
    }
  }

  if (!is_rec_multi_round_mode() && !use_llm_decode_fast_path) {
    // q_seq_lens is q_cu_seq_lens in GPU Model.
    // kv_seq_lens is kv_cu_seq_lens in GPU Model.
    VLOG(kGraphExecutorLogVerboseLevel)
        << "copy_ q_seq_lens: src shape="
        << params.attention.device.q_seq_lens.sizes() << ", dst slice shape=["
        << actual_batch_size + 1 << "]";
    q_seq_lens_.slice(/*dim=*/0, /*start=*/0, /*end=*/actual_batch_size + 1)
        .copy_(params.attention.device.q_seq_lens, /*non_blocking=*/true);

    VLOG(kGraphExecutorLogVerboseLevel)
        << "copy_ kv_seq_lens: src shape="
        << params.attention.device.kv_seq_lens.sizes() << ", dst slice shape=["
        << actual_batch_size + 1 << "]";
    kv_seq_lens_.slice(/*dim=*/0, /*start=*/0, /*end=*/actual_batch_size + 1)
        .copy_(params.attention.device.kv_seq_lens, /*non_blocking=*/true);
    if (params.meta.batch_forward_type.is_decode() &&
        padded_num_tokens > actual_num_tokens) {
      kv_seq_lens_
          .slice(/*dim=*/0,
                 /*start=*/actual_batch_size + 1,
                 /*end=*/padded_num_tokens + 1)
          .fill_(params.attention.device.kv_seq_lens[-1].item<int32_t>());
    }

    VLOG(kGraphExecutorLogVerboseLevel)
        << "copy_ new_cache_slots: src shape="
        << params.attention.device.new_cache_slots.sizes()
        << ", dst slice shape=[" << actual_num_tokens << "]";
    persistent_new_cache_slots_
        .slice(/*dim=*/0, /*start=*/0, /*end=*/actual_num_tokens)
        .copy_(params.attention.device.new_cache_slots, /*non_blocking=*/true);
    if (padded_num_tokens > actual_num_tokens) {
      persistent_new_cache_slots_
          .slice(/*dim=*/0,
                 /*start=*/actual_num_tokens,
                 /*end=*/padded_num_tokens)
          .fill_(-1);
    }

    // Keep metadata tensors pointing to persistent buffers used by graph
    // capture/replay so their addresses are stable and shapes match padded
    // tensors in capture path.
    attn_metadata->q_cu_seq_lens = q_seq_lens(/*actual_batch_size=*/
                                              actual_batch_size + 1);
    attn_metadata->kv_cu_seq_lens = kv_seq_lens(/*actual_batch_size=*/
                                                actual_batch_size + 1);
    const uint32_t slot_mapping_tokens =
        padded_num_tokens > 0 ? padded_num_tokens : actual_num_tokens;
    attn_metadata->slot_mapping =
        persistent_new_cache_slots(slot_mapping_tokens);
  }

  if (!is_rec_multi_round_mode() &&
      !params.embedding.linear_state_ids.empty()) {
    if (params.embedding.linear_state_indices.defined()) {
      persistent_linear_state_indices_
          .slice(/*dim=*/0, /*start=*/0, /*end=*/actual_batch_size)
          .copy_(params.embedding.linear_state_indices, /*non_blocking=*/true);
    } else {
      persistent_linear_state_indices_
          .slice(/*dim=*/0, /*start=*/0, /*end=*/actual_batch_size)
          .copy_(torch::tensor(params.embedding.linear_state_ids, torch::kInt)
                     .to(device_),
                 /*non_blocking=*/true);
    }
  }

  // Copy block table data. In rec multi-round, block_tables may already be
  // expanded to batch_size * beam_width rows while num_sequences still tracks
  // the logical request count. Use the tensor's real row count here.
  const int64_t actual_block_table_batch =
      is_rec_multi_round_mode() ? params.attention.device.block_tables.size(0)
                                : actual_batch_size;
  const int64_t actual_block_table_len =
      params.attention.device.block_tables.size(1);
  torch::Tensor slice_persistent_block_tables =
      persistent_block_tables_
          .slice(/*dim=*/0, /*start=*/0, /*end=*/actual_block_table_batch)
          .slice(/*dim=*/1, /*start=*/0, /*end=*/actual_block_table_len);

  VLOG(kGraphExecutorLogVerboseLevel)
      << "copy_ block_tables: src shape="
      << params.attention.device.block_tables.sizes()
      << ", dst slice shape=" << slice_persistent_block_tables.sizes();
  slice_persistent_block_tables.copy_(params.attention.device.block_tables,
                                      /*non_blocking=*/true);
  if (!attn_metadata->is_prefill || args_.enable_mla()) {
    attn_metadata->block_table = slice_persistent_block_tables;
  }

  // Update persistent embedding from input_embedding if available
  const auto& embedding = params.embedding.input_embedding;
  if (embedding.defined()) {
    const int64_t embedding_tokens = embedding.size(0);

    // Initialize persistent_embedding_ if needed and not already initialized
    if (persistent_embedding_.numel() == 0) {
      const int64_t max_tokens_per_batch = options_.max_tokens_per_batch();
      const int64_t embedding_dim = embedding.size(1);
      torch::ScalarType dtype = util::parse_dtype(args_.dtype(), device_);
      persistent_embedding_ =
          torch::zeros({max_tokens_per_batch, embedding_dim},
                       torch::dtype(dtype).device(device_));
    }

    // Copy embedding data to persistent buffer
    VLOG(kGraphExecutorLogVerboseLevel)
        << "copy_ embedding: src shape=" << embedding.sizes()
        << ", dst slice shape=[" << embedding_tokens << ", "
        << embedding.size(1) << "]";
    persistent_embedding_
        .slice(/*dim=*/0, /*start=*/0, /*end=*/embedding_tokens)
        .copy_(embedding, /*non_blocking=*/true);
  }

  const bool is_decode_with_llmrec =
      params.meta.batch_forward_type.is_decode() && params.has_llmrec_params();
  const bool use_two_stage_decode =
      !::xllm::RecConfig::get_instance().enable_xattention_one_stage() &&
      is_decode_with_llmrec;
  const int32_t head_dim = args_.head_dim();
  const int64_t tp_size =
      options_.world_size() / std::max(options_.dp_size(), 1);
  const int64_t n_heads = args_.n_heads() / std::max(tp_size, int64_t{1});
  const int64_t total_kv_heads = args_.n_kv_heads().value_or(args_.n_heads());
  const int64_t n_kv_heads =
      (total_kv_heads >= tp_size)
          ? (total_kv_heads / std::max(tp_size, int64_t{1}))
          : 1;
  const int64_t block_size = options_.block_size();

  // Get sliding_window from ModelArgs (default to -1 if not available)
  // Note: sliding_window in ModelArgs is the actual window size, but in
  // attention it's used as window_size_left which is typically sliding_window
  // - 1. This matches the behavior in attention.cpp where sliding_window_ is
  // initialized as sliding_window - 1 regardless of the value.
  int32_t sliding_window = args_.sliding_window();
  sliding_window =
      sliding_window - 1;  // Convert to window_size_left (always subtract 1)

  // Get dtype from k_cache
  const auto dtype = k_cache.scalar_type();
  // Determine backend
  const std::string backend = xllm::kernel::cuda::determine_attention_backend(
      /*pos_encoding_mode=*/0,
      /*use_fp16_qk_reduction=*/false,
      /*use_custom_mask=*/false);

  bool use_tensor_core =
      xllm::kernel::cuda::should_use_tensor_core(dtype, n_heads, n_kv_heads);
  if (use_two_stage_decode) {
    if (params.attention.device.q_seq_lens.defined() &&
        params.attention.device.q_seq_lens.numel() > 0) {
      const int64_t q_numel = params.attention.device.q_seq_lens.numel();
      q_seq_lens_.slice(/*dim=*/0, /*start=*/0, /*end=*/q_numel)
          .copy_(params.attention.device.q_seq_lens, /*non_blocking=*/true);
      attn_metadata->q_cu_seq_lens = q_seq_lens(/*actual_batch_size=*/q_numel);
    }

    if (params.attention.device.kv_seq_lens.defined() &&
        params.attention.device.kv_seq_lens.numel() > 0) {
      const int64_t kv_numel = params.attention.device.kv_seq_lens.numel();
      kv_seq_lens_.slice(/*dim=*/0, /*start=*/0, /*end=*/kv_numel)
          .copy_(params.attention.device.kv_seq_lens, /*non_blocking=*/true);
      attn_metadata->kv_cu_seq_lens =
          kv_seq_lens(/*actual_batch_size=*/kv_numel);
      if (kv_numel > 1) {
        attn_metadata->kv_seq_lens = torch::diff(attn_metadata->kv_cu_seq_lens);
      }
    }

    attn_metadata->paged_kv_indptr = torch::Tensor();
    attn_metadata->paged_kv_indices = torch::Tensor();
    attn_metadata->paged_kv_last_page_len = torch::Tensor();
    attn_metadata->qo_indptr = torch::Tensor();

    // Update plan_info if attn_metadata exists and enable_cuda_graph is true.
    const auto& llmrec_params = *params.llmrec_params();
    auto cache = attn_metadata->xattention_two_stage_decode_cache.value();
    CHECK(cache.q_cu_seq_lens_shared.defined())
        << "q_cu_seq_lens_shared must be initialized in rec worker";
    CHECK(cache.qo_indptr_expanded.defined())
        << "qo_indptr_expanded must be initialized in rec worker";
    CHECK(cache.paged_kv_indptr_expanded.defined() &&
          cache.paged_kv_indices_expanded.defined() &&
          cache.paged_kv_last_page_len_expanded.defined())
        << "paged_kv_* expanded tensors must be initialized in rec worker";
    CHECK(cache.shared_lse.defined() && cache.shared_o.defined() &&
          cache.unshared_lse.defined() && cache.unshared_o.defined())
        << "two-stage shared/unshared output cache tensors must be "
           "initialized in rec worker";

    const int64_t request_batch_size =
        cache.q_cu_seq_lens_shared.numel() > 0
            ? (cache.q_cu_seq_lens_shared.numel() - 1)
            : 0;
    CHECK_GT(request_batch_size, 0)
        << "request_batch_size must be > 0 for two-stage xattention";
    const int64_t beam_width = std::max<int64_t>(llmrec_params.beam_width, 1);
    const int64_t total_beam = request_batch_size * beam_width;

    CHECK_EQ(cache.shared_o.size(0), total_beam)
        << "shared_o first dim mismatch: expected total_beam=" << total_beam
        << ", got " << cache.shared_o.size(0);
    CHECK_EQ(cache.unshared_o.size(0), total_beam)
        << "unshared_o first dim mismatch: expected total_beam=" << total_beam
        << ", got " << cache.unshared_o.size(0);
    CHECK_EQ(cache.qo_indptr_expanded.numel(), total_beam + 1)
        << "qo_indptr_expanded size mismatch: expected " << (total_beam + 1)
        << ", got " << cache.qo_indptr_expanded.numel();
    CHECK_EQ(cache.paged_kv_indptr_expanded.numel(), total_beam + 1)
        << "paged_kv_indptr_expanded size mismatch";
    CHECK_EQ(cache.paged_kv_indices_expanded.numel(), total_beam)
        << "paged_kv_indices_expanded size mismatch";
    CHECK_EQ(cache.paged_kv_last_page_len_expanded.numel(), total_beam)
        << "paged_kv_last_page_len_expanded size mismatch";

    const int64_t max_decode_step =
        !llmrec_params.unshared_k_caches.empty()
            ? llmrec_params.unshared_k_caches[0].size(2)
            : std::max<int64_t>(
                  ::xllm::RecConfig::get_instance().max_decode_rounds() - 1, 1);
    CHECK_GT(max_decode_step, 0)
        << "max_decode_step must be > 0 for two-stage decode";

    cache.cached_batch_size = static_cast<int32_t>(request_batch_size);
    cache.cached_beam_size = static_cast<int32_t>(beam_width);
    cache.cached_num_heads = static_cast<int32_t>(n_heads);
    cache.cached_head_size = static_cast<int32_t>(head_dim);
    cache.cached_max_decode_step = static_cast<int32_t>(max_decode_step);
    cache.cached_step = (llmrec_params.current_round_tensor.defined() &&
                         llmrec_params.current_round_tensor.numel() > 0)
                            ? llmrec_params.current_round_tensor.item<int32_t>()
                            : 0;

    attn_metadata->xattention_two_stage_decode_cache = cache;

    layer::AttentionMetadata shared_attn_meta = *attn_metadata;
    shared_attn_meta.plan_info = attn_metadata->shared_plan_info;
    shared_attn_meta.q_cu_seq_lens = cache.q_cu_seq_lens_shared;
    shared_attn_meta.is_causal = false;

    attn_metadata->shared_plan_info->layer_id = 0;
    layer::xattention::update_xattention_plan_info(
        attn_metadata->shared_plan_info,
        backend,
        shared_attn_meta,
        dtype,
        dtype,
        dtype,
        head_dim,
        head_dim,
        static_cast<int32_t>(n_heads),
        static_cast<int32_t>(n_kv_heads),
        /*block_size=*/1,
        /*window_size_left=*/-1,
        /*enable_cuda_graph=*/true,
        /*causal=*/false,
        /*use_tensor_core=*/true,
        /*is_shared_stage_plan*/ true);

    layer::AttentionMetadata unshared_attn_meta = *attn_metadata;
    unshared_attn_meta.plan_info = attn_metadata->unshared_plan_info;
    unshared_attn_meta.qo_indptr = cache.qo_indptr_expanded;
    unshared_attn_meta.paged_kv_indptr = cache.paged_kv_indptr_expanded;
    unshared_attn_meta.paged_kv_indices = cache.paged_kv_indices_expanded;
    unshared_attn_meta.paged_kv_last_page_len =
        cache.paged_kv_last_page_len_expanded;
    unshared_attn_meta.is_causal = false;

    attn_metadata->unshared_plan_info->layer_id = 0;
    layer::xattention::update_xattention_plan_info(
        attn_metadata->unshared_plan_info,
        backend,
        unshared_attn_meta,
        dtype,
        dtype,
        dtype,
        head_dim,
        head_dim,
        static_cast<int32_t>(n_heads),
        static_cast<int32_t>(n_kv_heads),
        static_cast<int32_t>(max_decode_step),
        sliding_window,
        /*enable_cuda_graph=*/true,
        /*causal=*/false,
        use_tensor_core,
        /*is_shared_stage_plan*/ false);
    return build_capture_params_if_needed();
  }
  if (use_llm_decode_fast_path) {
    const uint32_t slot_mapping_tokens =
        padded_num_tokens > 0 ? padded_num_tokens : actual_num_tokens;
    // Decode has one query token per sequence, so its graph-facing metadata
    // must describe the padded token bucket rather than only the real batch.
    const uint32_t metadata_batch =
        padded_num_tokens > 0 ? padded_num_tokens
                              : static_cast<uint32_t>(actual_batch_size);
    attn_metadata->q_cu_seq_lens = persistent_decode_qo_indptr(metadata_batch);
    attn_metadata->kv_cu_seq_lens = kv_seq_lens(metadata_batch + 1);
    attn_metadata->kv_seq_lens = persistent_kv_seq_lens_delta(metadata_batch);
    attn_metadata->slot_mapping =
        persistent_new_cache_slots(slot_mapping_tokens);
    attn_metadata->paged_kv_indptr = persistent_paged_kv_indptr(metadata_batch);
    attn_metadata->paged_kv_indices = persistent_paged_kv_indices_;
    attn_metadata->paged_kv_last_page_len =
        persistent_paged_kv_last_page_len(metadata_batch);
    attn_metadata->qo_indptr = persistent_decode_qo_indptr(metadata_batch);
  } else {
    CHECK(params.attention.device.paged_kv_indptr.defined())
        << "paged_kv_indptr should not be null";
    VLOG(kGraphExecutorLogVerboseLevel)
        << "copy_ paged_kv_indptr: src shape="
        << params.attention.device.paged_kv_indptr.sizes()
        << ", dst slice shape=[" << (actual_batch_size + 1) << "]";
    if (VLOG_IS_ON(kGraphExecutorLogVerboseLevel)) {
      torch::Tensor paged_kv_indptr_cpu =
          params.attention.device.paged_kv_indptr.to(torch::kCPU);
      VLOG(kGraphExecutorLogVerboseLevel)
          << "copy_ paged_kv_indptr: src values=" << paged_kv_indptr_cpu;
    }
    persistent_paged_kv_indptr_
        .slice(/*dim=*/0,
               /*start=*/0,
               /*end=*/actual_batch_size + 1)
        .copy_(params.attention.device.paged_kv_indptr, /*non_blocking=*/true);
    const bool pad_decode_metadata =
        params.meta.batch_forward_type.is_decode() &&
        padded_num_tokens > actual_num_tokens;
    if (pad_decode_metadata) {
      persistent_paged_kv_indptr_
          .slice(/*dim=*/0,
                 /*start=*/actual_batch_size + 1,
                 /*end=*/padded_num_tokens + 1)
          .fill_(params.attention.device.paged_kv_indptr[-1].item<int32_t>());
    }
    CHECK(params.attention.device.paged_kv_indices.defined())
        << "paged_kv_indices should not be null";
    const int64_t actual_indices_size =
        params.attention.device.paged_kv_indices.size(0);
    VLOG(kGraphExecutorLogVerboseLevel)
        << "copy_ paged_kv_indices: src shape="
        << params.attention.device.paged_kv_indices.sizes()
        << ", dst slice shape=[" << actual_indices_size << "]";
    persistent_paged_kv_indices_
        .slice(/*dim=*/0,
               /*start=*/0,
               /*end=*/actual_indices_size)
        .copy_(params.attention.device.paged_kv_indices, /*non_blocking=*/true);
    CHECK(params.attention.device.paged_kv_last_page_len.defined())
        << "paged_kv_last_page_len should not be null";
    VLOG(kGraphExecutorLogVerboseLevel)
        << "copy_ paged_kv_last_page_len: src shape="
        << params.attention.device.paged_kv_last_page_len.sizes()
        << ", dst slice shape=[" << actual_batch_size << "]";
    persistent_paged_kv_last_page_len_
        .slice(/*dim=*/0,
               /*start=*/0,
               /*end=*/actual_batch_size)
        .copy_(params.attention.device.paged_kv_last_page_len,
               /*non_blocking=*/true);
    if (pad_decode_metadata) {
      persistent_paged_kv_last_page_len_
          .slice(/*dim=*/0,
                 /*start=*/actual_batch_size,
                 /*end=*/padded_num_tokens)
          .fill_(1);
    }
    const int64_t metadata_batch =
        params.meta.batch_forward_type.is_decode() && padded_num_tokens > 0
            ? padded_num_tokens
            : actual_batch_size;
    if (params.meta.batch_forward_type.is_decode()) {
      attn_metadata->q_cu_seq_lens =
          persistent_decode_qo_indptr(metadata_batch);
      attn_metadata->kv_cu_seq_lens = kv_seq_lens(metadata_batch + 1);
      torch::Tensor kv_seq_lens_delta =
          persistent_kv_seq_lens_delta(metadata_batch);
      kv_seq_lens_delta.copy_(torch::diff(attn_metadata->kv_cu_seq_lens),
                              /*non_blocking=*/true);
      attn_metadata->kv_seq_lens = kv_seq_lens_delta;
    } else {
      attn_metadata->kv_seq_lens =
          torch::diff(kv_seq_lens(/*actual_batch_size=*/metadata_batch + 1));
    }
    attn_metadata->paged_kv_indptr = persistent_paged_kv_indptr(metadata_batch);
    attn_metadata->paged_kv_indices = persistent_paged_kv_indices_;
    attn_metadata->paged_kv_last_page_len =
        persistent_paged_kv_last_page_len(metadata_batch);
    attn_metadata->qo_indptr = persistent_decode_qo_indptr(metadata_batch);
  }
  // Update plan_info if attn_metadata exists and enable_cuda_graph is true
  // This ensures plan_info is updated before CUDA graph capture/replay
  {
    // Determine if causal (prefill mode)
    const bool causal =
        attn_metadata->is_prefill || attn_metadata->is_chunked_prefill;

    // Update plan_info
    // Note: plan_info is only updated at layer 0, so we set layer_id to 0
    attn_metadata->plan_info->layer_id = 0;
    CHECK_EQ(dtype, torch::ScalarType::BFloat16)
        << "only support bf16 kvcache for now";

    VLOG(kGraphExecutorLogVerboseLevel)
        << "CudaGraphPersistentParam::update() calling update_plan_info: "
        << "is_prefill=" << attn_metadata->is_prefill
        << ", is_chunked_prefill=" << attn_metadata->is_chunked_prefill
        << ", causal=" << causal << ", backend=" << backend
        << ", enable_cuda_graph=" << attn_metadata->enable_cuda_graph;

    if (attn_metadata->is_prefill) {
      layer::flashinfer::update_prefill_plan_info(
          attn_metadata->plan_info,
          backend,
          *attn_metadata,
          dtype,                             // query_dtype
          dtype,                             // key_dtype
          dtype,                             // output_dtype
          head_dim,                          // head_dim_qk
          head_dim,                          // head_dim_vo
          static_cast<int32_t>(n_heads),     // num_qo_heads
          static_cast<int32_t>(n_kv_heads),  // num_kv_heads
          /*enable_cuda_graph=*/true);
    } else if (attn_metadata->is_chunked_prefill) {
      layer::flashinfer::update_chunked_prefill_plan_info(
          attn_metadata->plan_info,
          /*backend=*/"fa2",  // flashinfer paged fa3 is slow, use fa2 instead
          *attn_metadata,
          dtype,                             // query_dtype
          dtype,                             // key_dtype
          dtype,                             // output_dtype
          head_dim,                          // head_dim_qk
          head_dim,                          // head_dim_vo
          static_cast<int32_t>(n_heads),     // num_qo_heads
          static_cast<int32_t>(n_kv_heads),  // num_kv_heads
          static_cast<int32_t>(block_size),  // block_size
          sliding_window,                    // window_size_left
          /*enable_cuda_graph=*/true);
    } else {
      layer::flashinfer::update_decode_plan_info(
          attn_metadata->plan_info,
          /*backend=*/"fa2",  // flashinfer paged fa3 is slow, use fa2 instead
          *attn_metadata,
          dtype,                             // query_dtype
          dtype,                             // key_dtype
          dtype,                             // output_dtype
          head_dim,                          // head_dim_qk
          head_dim,                          // head_dim_vo
          static_cast<int32_t>(n_heads),     // num_qo_heads
          static_cast<int32_t>(n_kv_heads),  // num_kv_heads
          static_cast<int32_t>(block_size),  // block_size
          sliding_window,                    // window_size_left
          /*enable_cuda_graph=*/true,
          use_tensor_core);
    }

    VLOG(kGraphExecutorLogVerboseLevel)
        << "CudaGraphPersistentParam::update() plan_info updated: uri="
        << attn_metadata->plan_info->uri << ", plan_info.defined="
        << attn_metadata->plan_info->plan_info.defined() << ", plan_info.size="
        << (attn_metadata->plan_info->plan_info.defined()
                ? attn_metadata->plan_info->plan_info.size()
                : 0);
  }

  // Return ModelInputParams with persistent buffer references if requested
  return build_capture_params_if_needed();
}

// CudaGraph implementation
bool CudaGraph::capture(CausalLM* model,
                        const ModelArgs& args,
                        const runtime::Options& options,
                        const torch::Tensor& tokens,
                        const torch::Tensor& positions,
                        const ModelInputParams& params,
                        std::vector<KVCache>& kv_cache,
                        uint32_t bucket_num_tokens,
                        const at::cuda::MempoolId_t& pool,
                        TorchMemPool* pool_ptr) {
  padded_num_tokens_ = bucket_num_tokens;
  const uint32_t actual_num_tokens = tokens.size(0);
  CHECK_GE(padded_num_tokens_, actual_num_tokens)
      << "bucket_num_tokens >= actual_num_tokens";

  // Guard CUDA graph capture region with a device-level exclusive lock to
  // prevent conflicting GPU work from other streams (e.g., prepare streams) on
  // the same device when using cudaStreamCaptureModeGlobal. Capture requires
  // exclusive access, so we use write lock.
  std::optional<std::unique_lock<std::shared_mutex>> capture_lock_guard;
  if (::xllm::ExecutionConfig::get_instance().enable_graph()) {
    auto& capture_lock =
        ::xllm::cuda::DeviceCaptureLock::get_instance().get_write_lock(
            device_index_);
    capture_lock_guard.emplace(capture_lock);
  }
  // Use the returned ModelInputParams for graph capture
  // Always use capture stream for plan/update + capture + forward.
  at::cuda::CUDAStream original_stream =
      at::cuda::getCurrentCUDAStream(device_index_);
  at::cuda::CUDAStream capture_stream = capture_stream_;
  if (original_stream != capture_stream) {
    original_stream.synchronize();
    capture_stream.synchronize();
  }
  std::optional<at::cuda::CUDAStreamGuard> stream_guard;
  stream_guard.emplace(capture_stream);

  // auto& tensor_options = model->options();

  // Update persistent parameters with input data before capture (includes
  // FlashInfer plan/update).
  auto full_attention_cache =
      CudaGraphExecutorImpl::find_first_full_attention_cache(kv_cache);
  CHECK(full_attention_cache.has_value())
      << "CUDA graph capture requires at least one full-attention KV cache";
  const torch::Tensor& k_cache = full_attention_cache->first;
  const torch::Tensor& v_cache = full_attention_cache->second;
  auto graph_params_opt =
      persistent_param_.update(tokens,
                               k_cache,
                               v_cache,
                               positions,
                               params,
                               padded_num_tokens_,
                               /*return_capture_params=*/true);

  // Use the returned ModelInputParams for graph capture
  CHECK(graph_params_opt.has_value())
      << "update() should return ModelInputParams when "
         "return_capture_params=true";

  LOG(INFO) << "CUDA graph capture begin, bucket_num_tokens: "
            << bucket_num_tokens << ", actual_num_tokens: " << actual_num_tokens
            << ", is_piecewise: " << is_piecewise_;

  if (is_piecewise_) {
    // Piecewise capture mode (for prefill)
    // Warmup: execute forward once without capture to initialize cuBLAS handles
    // and other CUDA resources. This is necessary because these resources
    // cannot be created during CUDA graph capture mode.
    model->forward(persistent_param_.persistent_tokens(padded_num_tokens_),
                   persistent_param_.persistent_positions(padded_num_tokens_),
                   kv_cache,
                   graph_params_opt.value());

    // MemPoolContext has been deprecated in torch >= 2.8
#if TORCH_VERSION_MAJOR <= 2 && TORCH_VERSION_MINOR <= 7
    // Activate VMM mempool only for the actual capture to keep plan_info
    // allocations out of the shared physical memory pool.
    std::optional<c10::cuda::MemPoolContext> mempool_ctx;
    if (pool_ptr != nullptr) {
      mempool_ctx.emplace(pool_ptr);
    }
#endif

    // Begin piecewise capture via GlobalCaptureInstance.
    GlobalCaptureInstance::get_instance().begin_capture(pool);

    // Execute forward pass - attention operations will be captured separately
    auto forward_result = model->forward(
        persistent_param_.persistent_tokens(padded_num_tokens_),
        persistent_param_.persistent_positions(padded_num_tokens_),
        kv_cache,
        graph_params_opt.value());

    // Store result in persistent buffer
    persistent_param_.set_hidden_states(forward_result.hidden_states);
    // Only capture aux_hidden_states when enable_graph_aux_hidden_states is on
    // (e.g. main worker in EAGLE-3); draft worker has this option false.
    if (options.enable_graph_aux_hidden_states() &&
        forward_result.aux_hidden_states.defined()) {
      persistent_param_.set_aux_hidden_states(forward_result.aux_hidden_states);
    }
    VLOG(kGraphExecutorLogVerboseLevel)
        << "Piecewise capture forward_result shape: "
        << forward_result.hidden_states.sizes();

    // End capture and get piecewise graphs
    auto piecewise_graphs = GlobalCaptureInstance::get_instance().end_capture();

    if (!piecewise_graphs || piecewise_graphs->empty()) {
      LOG(WARNING) << "Failed to capture piecewise graph: no graphs captured";
      return false;
    }

    // Move piecewise graphs to member
    piecewise_graph_ = std::move(*piecewise_graphs);

    LOG(INFO) << "Piecewise graph capture end, bucket_num_tokens: "
              << bucket_num_tokens
              << ", num_graphs: " << piecewise_graph_.size()
              << ", num_runners: " << piecewise_graph_.num_runners();
  } else {
    // Normal capture mode (for decode)

    // MemPoolContext has been deprecated in torch >= 2.8
#if TORCH_VERSION_MAJOR <= 2 && TORCH_VERSION_MINOR <= 7
    // Activate VMM mempool only for the actual capture to keep plan_info
    // allocations out of the shared physical memory pool.
    std::optional<c10::cuda::MemPoolContext> mempool_ctx;
    if (pool_ptr != nullptr) {
      mempool_ctx.emplace(pool_ptr);
    }
#endif

    // Begin graph capture (capture_mode defaults to
    // cudaStreamCaptureModeGlobal)
    // graph_.capture_begin(pool);
    graph_.capture_begin(pool, cudaStreamCaptureModeThreadLocal);

    // Execute forward pass - CUDA graph will capture this
    auto forward_result = model->forward(
        persistent_param_.persistent_tokens(padded_num_tokens_),
        persistent_param_.persistent_positions(padded_num_tokens_),
        kv_cache,
        graph_params_opt.value());

    // Store result in persistent buffer
    persistent_param_.set_hidden_states(forward_result.hidden_states);
    if (options.enable_graph_aux_hidden_states() &&
        forward_result.aux_hidden_states.defined()) {
      persistent_param_.set_aux_hidden_states(forward_result.aux_hidden_states);
    }

    // End graph capture
    graph_.capture_end();
  }

  // Synchronize to ensure graph capture is completed.
  capture_stream.synchronize();

  // Explicitly restore stream after capture before replay logic.
  stream_guard.reset();

  // Replay is unified in CudaGraphExecutorImpl::run() after capture success
  // for both prefill and decode.

  LOG(INFO) << "CUDA graph capture end, bucket_num_tokens: "
            << bucket_num_tokens;
  return true;
}

ModelOutput CudaGraph::replay(const torch::Tensor& tokens,
                              const torch::Tensor& positions,
                              std::vector<KVCache>& kv_cache,
                              const ModelInputParams& params) {
  const uint32_t actual_num_tokens = tokens.size(0);
  CHECK_LE(actual_num_tokens, padded_num_tokens_)
      << "num_tokens mismatch: expected <= " << padded_num_tokens_ << ", got "
      << actual_num_tokens;

  // Guard CUDA graph replay with a device-level shared lock to allow multiple
  // replay operations to run concurrently while preventing conflicts with
  // capture operations. Replay can share the lock with other replay/prepare
  // operations.
  std::optional<std::shared_lock<std::shared_mutex>> replay_lock_guard;
  if (::xllm::ExecutionConfig::get_instance().enable_graph()) {
    auto& replay_lock =
        ::xllm::cuda::DeviceCaptureLock::get_instance().get_read_lock(
            device_index_);
    replay_lock_guard.emplace(replay_lock);
  }

  // Update persistent parameters with new input data
  auto full_attention_cache =
      CudaGraphExecutorImpl::find_first_full_attention_cache(kv_cache);
  CHECK(full_attention_cache.has_value())
      << "CUDA graph replay requires at least one full-attention KV cache";
  const torch::Tensor& k_cache = full_attention_cache->first;
  const torch::Tensor& v_cache = full_attention_cache->second;

  if (is_piecewise_) {
    // Piecewise replay mode (for prefill)
    // Need to get updated params with attn_metadata for attention replay
    auto updated_params_opt =
        persistent_param_.update(tokens,
                                 k_cache,
                                 v_cache,
                                 positions,
                                 params,
                                 padded_num_tokens_,
                                 /*return_capture_params=*/true);
    CHECK(updated_params_opt.has_value())
        << "update() should return ModelInputParams for piecewise replay";

    const auto& updated_params = updated_params_opt.value();
    CHECK(piecewise_graph_.num_runners() > 0)
        << "Piecewise graph must have attention runners";
    CHECK(updated_params.attn_metadata)
        << "attn_metadata is required for piecewise replay";
    CHECK(updated_params.attn_metadata->plan_info)
        << "plan_info is required for piecewise replay";

    VLOG(kGraphExecutorLogVerboseLevel)
        << "CudaGraph::replay() piecewise replay with uri="
        << updated_params.attn_metadata->plan_info->uri
        << ", plan_info.defined="
        << updated_params.attn_metadata->plan_info->plan_info.defined();

    // Build AttentionReplayParams from updated attn_metadata
    ::xllm::kernel::cuda::AttentionReplayParams replay_params;
    replay_params.actual_num_tokens = actual_num_tokens;
    replay_params.plan_info =
        updated_params.attn_metadata->plan_info->plan_info;
    replay_params.q_cu_seq_lens = updated_params.attn_metadata->q_cu_seq_lens;
    replay_params.kv_cu_seq_lens = updated_params.attn_metadata->kv_cu_seq_lens;

    // Replay piecewise graphs and attention runners
    piecewise_graph_.replay(replay_params);
  } else {
    // Normal replay mode (for decode)
    persistent_param_.update(tokens,
                             k_cache,
                             v_cache,
                             positions,
                             params,
                             padded_num_tokens_,
                             /*return_capture_params=*/false);
    graph_.replay();
  }

  // Return the actual num_tokens portion of ModelOutput
  // Note: aux_hidden_states handling is done in CudaGraphExecutorImpl::run()
  // since replay() doesn't have access to options
  return ModelOutput(get_hidden_states(actual_num_tokens));
}

// CudaGraphExecutorImpl implementation
CudaGraphExecutorImpl::CudaGraphExecutorImpl(CausalLM* model,
                                             const ModelArgs& args,
                                             const torch::Device& device,
                                             const runtime::Options& options)
    : model_(model),
      args_(args),
      device_(device),
      options_(options),
      enable_prefill_piecewise_graph_(::xllm::ExecutionConfig::get_instance()
                                          .enable_prefill_piecewise_graph()) {
  max_tokens_for_graph_mode_ =
      ::xllm::ExecutionConfig::get_instance().max_tokens_for_graph_mode();
  if (max_tokens_for_graph_mode_ < options_.max_seqs_per_batch()) {
    max_tokens_for_graph_mode_ = options_.max_seqs_per_batch();
  }
  // Keep one pool per executor instance so all captured graphs can reuse it,
  // while avoiding cross-instance stale-handle reuse.
  graph_pool_ = at::cuda::graph_pool_handle();
  // Create single persistent parameter object shared by all CudaGraph instances
  persistent_param_ =
      std::make_unique<CudaGraphPersistentParam>(args_, device_, options_);
  const size_t persistent_bytes =
      persistent_param_->get_persistent_tensor_bytes();
  LOG(INFO) << "Persistent input tensor total size: " << persistent_bytes
            << " bytes (" << (persistent_bytes / (1024 * 1024)) << " MB)";

  const auto private_pool_usage =
      get_private_pools_memory_usage(device_.index());
  baseline_private_pool_reserved_bytes_ = private_pool_usage.reserved_bytes;
  baseline_private_pool_allocated_bytes_ = private_pool_usage.allocated_bytes;
  baseline_private_pool_active_bytes_ = private_pool_usage.active_bytes;
  baseline_allocator_reserved_bytes_ =
      get_allocator_reserved_bytes(device_.index());
}

std::optional<std::pair<torch::Tensor, torch::Tensor>>
CudaGraphExecutorImpl::find_first_full_attention_cache(
    const std::vector<KVCache>& kv_caches) {
  for (const auto& cache : kv_caches) {
    if (cache.empty()) {
      continue;
    }
    auto k_cache = cache.get_k_cache();
    auto v_cache = cache.get_v_cache();
    if (k_cache.defined() && v_cache.defined() && k_cache.numel() > 0 &&
        v_cache.numel() > 0) {
      return std::make_pair(std::move(k_cache), std::move(v_cache));
    }
  }
  return std::nullopt;
}

// ============== VMM Allocator Support ==============
// These functions provide VMM-based memory pool for CUDA Graph capture,
// enabling memory reuse across different shape captures.

namespace {
// Physical pool id: same id => reuse across different shapes (prefill vs decode
// are different physical pools).
constexpr uint32_t kPhysicalPoolIdPrefill = 0;
constexpr uint32_t kPhysicalPoolIdDecode = 1;
}  // namespace

struct CudaGraphExecutorImpl::VmmPoolState {
  std::unique_ptr<xllm::SharedVMMAllocator> allocator;
  std::unique_ptr<xllm::VMMTorchAllocator> torch_allocator;
  std::unordered_map<uint32_t, std::unique_ptr<TorchMemPool>> mempools_by_shape;
};

CudaGraphExecutorImpl::~CudaGraphExecutorImpl() {
  // Release captured graphs before MemPool objects to avoid PyTorch MemPool
  // use_count assertion during destruction.
  prefill_graphs_.clear();
  graphs_.clear();
  vmm_pools_.clear();
}

CudaGraphExecutorImpl::VmmPoolState&
CudaGraphExecutorImpl::get_or_create_vmm_pool_state(uint32_t physical_pool_id) {
  std::lock_guard<std::mutex> lock(vmm_mutex_);
  auto& slot = vmm_pools_[physical_pool_id];
  if (!slot) {
    auto state = std::make_unique<VmmPoolState>();
    state->allocator = std::make_unique<xllm::SharedVMMAllocator>();
    state->allocator->init(device_.index());
    state->torch_allocator =
        std::make_unique<xllm::VMMTorchAllocator>(state->allocator.get());
    slot = std::move(state);
    LOG(INFO) << "Created VMM pool state for executor " << this << ", device "
              << device_.index() << ", physical_pool_id: " << physical_pool_id;
  }
  return *slot;
}

TorchMemPool* CudaGraphExecutorImpl::get_or_create_vmm_mempool(
    uint32_t physical_pool_id,
    uint32_t shape_id) {
  VmmPoolState& state = get_or_create_vmm_pool_state(physical_pool_id);
  std::lock_guard<std::mutex> lock(vmm_mutex_);
  auto& mempools = state.mempools_by_shape;
  auto it = mempools.find(shape_id);
  if (it != mempools.end()) {
    return it->second.get();
  }
  auto pool = std::make_unique<TorchMemPool>(state.torch_allocator.get(),
                                             /*is_user_created=*/true);
  TorchMemPool* ptr = pool.get();
  mempools[shape_id] = std::move(pool);
  VLOG(kGraphExecutorLogVerboseLevel)
      << "Created per-shape VMM MemPool for executor " << this << ", device "
      << device_.index() << ", physical_pool_id: " << physical_pool_id
      << ", shape_id: " << shape_id << ", pool_id: {" << ptr->id().first << ", "
      << ptr->id().second << "}";
  return ptr;
}

TorchMemPool* CudaGraphExecutorImpl::get_vmm_mempool(uint32_t physical_pool_id,
                                                     uint32_t shape_id) {
  std::lock_guard<std::mutex> lock(vmm_mutex_);
  auto it = vmm_pools_.find(physical_pool_id);
  if (it == vmm_pools_.end() || !it->second) {
    return nullptr;
  }
  auto& mempools = it->second->mempools_by_shape;
  auto it_pool = mempools.find(shape_id);
  if (it_pool == mempools.end()) {
    return nullptr;
  }
  return it_pool->second.get();
}

CudaGraphExecutorImpl::GraphMemoryUsageStats
CudaGraphExecutorImpl::get_graph_memory_usage_stats() {
  GraphMemoryUsageStats stats;

  if (!::xllm::ExecutionConfig::get_instance().enable_graph_vmm_pool()) {
    const auto pool = get_mem_pool();
    const auto usage = get_graph_pool_memory_usage(device_.index(), pool);
    stats.executor_total_bytes = usage.reserved_bytes;
    stats.allocated_pool_bytes = usage.allocated_bytes;
    stats.active_pool_bytes = usage.active_bytes;
    stats.pool_high_water_mark_bytes = usage.allocated_bytes;

    if (stats.executor_total_bytes == 0) {
      const auto private_pool_usage =
          get_private_pools_memory_usage(device_.index());
      if (private_pool_usage.reserved_bytes >
          baseline_private_pool_reserved_bytes_) {
        stats.executor_total_bytes = private_pool_usage.reserved_bytes -
                                     baseline_private_pool_reserved_bytes_;
      }
      if (private_pool_usage.allocated_bytes >
          baseline_private_pool_allocated_bytes_) {
        stats.allocated_pool_bytes = private_pool_usage.allocated_bytes -
                                     baseline_private_pool_allocated_bytes_;
      }
      if (private_pool_usage.active_bytes >
          baseline_private_pool_active_bytes_) {
        stats.active_pool_bytes = private_pool_usage.active_bytes -
                                  baseline_private_pool_active_bytes_;
      }
      stats.pool_high_water_mark_bytes = stats.allocated_pool_bytes;
    }

    if (stats.executor_total_bytes == 0) {
      const size_t allocator_reserved_bytes =
          get_allocator_reserved_bytes(device_.index());
      if (allocator_reserved_bytes > baseline_allocator_reserved_bytes_) {
        stats.executor_total_bytes =
            allocator_reserved_bytes - baseline_allocator_reserved_bytes_;
      }
    }
  } else {
    std::lock_guard<std::mutex> lock(vmm_mutex_);
    for (const auto& kv : vmm_pools_) {
      const VmmPoolState& pool_state = *kv.second;
      stats.executor_total_bytes += pool_state.allocator->mapped_size();
      stats.allocated_pool_bytes += pool_state.allocator->high_water_mark();
      stats.active_pool_bytes += pool_state.allocator->current_offset();
    }
    stats.pool_high_water_mark_bytes = stats.allocated_pool_bytes;
  }

  stats.persistent_param_bytes =
      persistent_param_ ? persistent_param_->get_persistent_tensor_bytes() : 0;
  stats.executor_total_bytes += stats.persistent_param_bytes;

  return stats;
}

size_t CudaGraphExecutorImpl::get_graph_memory_usage_bytes() {
  return get_graph_memory_usage_stats().executor_total_bytes;
}

void CudaGraphExecutorImpl::log_graph_memory_after_capture() {
  const auto stats = get_graph_memory_usage_stats();
  const size_t executor_total_bytes = stats.executor_total_bytes;
  const size_t persistent_param_bytes = stats.persistent_param_bytes;
  const size_t allocated_bytes = stats.allocated_pool_bytes;
  const size_t active_bytes = stats.active_pool_bytes;
  const size_t high_water_mark_bytes = stats.pool_high_water_mark_bytes;

  if (executor_total_bytes <= last_logged_executor_total_bytes_) {
    return;
  }
  last_logged_executor_total_bytes_ = executor_total_bytes;

  const bool vmm_enabled =
      ::xllm::ExecutionConfig::get_instance().enable_graph_vmm_pool();
  auto format_size = [](size_t bytes) {
    return c10::CachingDeviceAllocator::format_size(bytes);
  };
  LOG(INFO) << "GraphExecutorMemory Usage:"
            << " enable_graph_vmm_pool=" << vmm_enabled
            << " executor_total_memory=" << format_size(executor_total_bytes)
            << " persistent_param=" << format_size(persistent_param_bytes)
            << " allocated_pool_memory=" << format_size(allocated_bytes)
            << " active_pool_memory=" << format_size(active_bytes)
            << " pool_high_water_mark=" << format_size(high_water_mark_bytes);
}

// Switch to new virtual address space before capture for the given physical
// pool. Enables physical memory reuse within that physical pool across shapes.
void CudaGraphExecutorImpl::reset_vmm_allocator_offset(
    uint32_t physical_pool_id) {
  auto& state = get_or_create_vmm_pool_state(physical_pool_id);
  state.allocator->switch_to_new_virtual_space();
  VLOG(kGraphExecutorLogVerboseLevel)
      << "Reset VMM allocator for device " << device_.index()
      << ", physical_pool_id: " << physical_pool_id;
}

// Get graph memory pool id for capture. When VMM is enabled, uses per-shape
// MemPool under (physical_pool_id, shape_id).
at::cuda::MempoolId_t CudaGraphExecutorImpl::get_mem_pool(
    uint32_t physical_pool_id,
    uint32_t shape_id) {
  if (!::xllm::ExecutionConfig::get_instance().enable_graph_vmm_pool()) {
    // Non-VMM mode intentionally uses one pool per executor instance.
    // Rationale: this executor is designed for single-threaded invocation, and
    // concurrent run() on the same executor instance is not allowed.
    // Under this contract, a per-instance pool is safe and keeps graph memory
    // ownership tied to the executor lifecycle.
    return graph_pool_;
  }
  // Per-shape VMM MemPool: look up pool for (physical_pool_id, shape_id).
  TorchMemPool* pool = get_vmm_mempool(physical_pool_id, shape_id);
  CHECK(pool != nullptr)
      << "VMM MemPool for shape_id=" << shape_id
      << ", physical_pool_id=" << physical_pool_id
      << " not found; get_or_create_vmm_mempool must be called before capture";
  return pool->id();
}

// Static method to get CUDA capture stream for current thread
// Each thread gets its own high-priority capture stream
c10::cuda::CUDAStream CudaGraphExecutorImpl::get_capture_stream(
    c10::DeviceIndex device_index) {
  // Use thread_local to ensure each thread has its own capture stream
  // This is required because CUDA graphs must be captured on a non-default
  // stream. We use high-priority streams for better performance.
  thread_local c10::cuda::CUDAStream thread_capture_stream =
      c10::cuda::getStreamFromPool(/*isHighPriority=*/true, device_index);

  // Thread-local counter to log initialization only once per thread
  thread_local bool initialized = false;
  if (!initialized) {
    LOG(INFO) << "Initialized capture_stream for thread: "
              << std::this_thread::get_id()
              << ", stream: " << thread_capture_stream
              << ", device_index: " << device_index;
    initialized = true;
  }

  return thread_capture_stream;
}

ForwardInput CudaGraphExecutorImpl::prepare_inputs(Batch& batch) {
  // Prepare inputs for workers
  return batch.prepare_forward_input(
      options_.num_decoding_tokens(), 0, args_, options_.cp_size());
}

ModelOutput CudaGraphExecutorImpl::attach_aux_hidden_states_if_needed(
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

ModelOutput CudaGraphExecutorImpl::run(const torch::Tensor& tokens,
                                       const torch::Tensor& positions,
                                       std::vector<KVCache>& kv_caches,
                                       const ModelInputParams& params) {
  const bool is_prefill = params.meta.batch_forward_type.is_prefill();
  const bool is_decode = params.meta.batch_forward_type.is_decode();

  // Get actual num_tokens from tokens shape
  const uint32_t n_tokens = tokens.size(/*dim=*/0);
  const uint32_t bucket_num_tokens =
      get_bucket_num_tokens(n_tokens, is_prefill);

  // Prefill phase with piecewise graph
  if (is_prefill && enable_prefill_piecewise_graph_) {
    // Check if token count is within limit
    const bool graph_mode_supported = n_tokens <= max_tokens_for_graph_mode_;

    if (!graph_mode_supported) {
      VLOG(kGraphExecutorLogVerboseLevel)
          << "Token count " << n_tokens
          << " exceeds max_tokens_for_graph_mode ("
          << max_tokens_for_graph_mode_ << "), falling back to eager mode";
      COUNTER_INC(num_model_execution_total_eager);
      return model_->forward(tokens, positions, kv_caches, params);
    }

    // Check if piecewise graph exists for this bucket
    auto it = prefill_graphs_.find(bucket_num_tokens);
    if (it != prefill_graphs_.end()) {
      // Replay existing piecewise graph
      VLOG(kGraphExecutorLogVerboseLevel)
          << "CudaGraphExecutorImpl::run() in prefill piecewise replay mode";
      auto result = it->second->replay(tokens, positions, kv_caches, params);
      return attach_aux_hidden_states_if_needed(result.hidden_states, n_tokens);
    }

    // Graph doesn't exist, try to create it lazily with piecewise capture
    auto graph =
        std::make_unique<CudaGraph>(*persistent_param_,
                                    device_.index(),
                                    get_capture_stream(device_.index()),
                                    /*is_piecewise=*/true);
    VLOG(kGraphExecutorLogVerboseLevel)
        << "CudaGraphExecutorImpl::run() in prefill piecewise capture mode";

    TorchMemPool* pool_ptr = nullptr;
    if (::xllm::ExecutionConfig::get_instance().enable_graph_vmm_pool()) {
      reset_vmm_allocator_offset(kPhysicalPoolIdPrefill);
      const uint32_t shape_id = bucket_num_tokens;
      pool_ptr = get_or_create_vmm_mempool(kPhysicalPoolIdPrefill, shape_id);
    }
    const at::cuda::MempoolId_t mem_pool =
        get_mem_pool(kPhysicalPoolIdPrefill, bucket_num_tokens);

    bool capture_success = graph->capture(model_,
                                          args_,
                                          options_,
                                          tokens,
                                          positions,
                                          params,
                                          kv_caches,
                                          bucket_num_tokens,
                                          mem_pool,
                                          pool_ptr);

    if (capture_success) {
      LOG(INFO) << "Lazy capturing piecewise CUDA graph for bucket num_tokens: "
                << bucket_num_tokens << " (actual num_tokens: " << n_tokens
                << ") done";

      log_graph_memory_after_capture();

      // Save the graph for future reuse
      prefill_graphs_[bucket_num_tokens] = std::move(graph);

      // Run replay after capture so first request uses same execution path as
      // subsequent requests.
      auto result = prefill_graphs_[bucket_num_tokens]->replay(
          tokens, positions, kv_caches, params);
      return attach_aux_hidden_states_if_needed(result.hidden_states, n_tokens);
    }

    // Fail fast intentionally: after entering graph mode, silently falling back
    // to eager can hide allocator/capture regressions and make latency behavior
    // non-deterministic in production. Operators can disable graph mode via
    // ::xllm::ExecutionConfig::get_instance().enable_graph() or
    // ::xllm::ExecutionConfig::get_instance().enable_prefill_piecewise_graph()
    // when fallback behavior is preferred over strict graph-mode enforcement.
    LOG(FATAL)
        << "Failed to capture piecewise CUDA graph for bucket num_tokens: "
        << bucket_num_tokens << " (actual num_tokens: " << n_tokens << ")";
  }

  // Prefill without piecewise graph: use eager mode
  if (is_prefill) {
    COUNTER_INC(num_model_execution_total_eager);
    return model_->forward(tokens, positions, kv_caches, params);
  }

  // Decode phase with full graph
  if (is_decode) {
    // Check if conditions are suitable for graph execution (replay or capture)
    const auto max_seq_len = args_.max_position_embeddings();
    const bool seq_len_supported = params.meta.kv_max_seq_len <= max_seq_len;

    // Early return if conditions are not suitable for graph operations
    if (!seq_len_supported) {
      LOG(WARNING) << "Not suitable for CUDA graph operations, falling back to "
                      "eager mode.";
      COUNTER_INC(num_model_execution_total_eager);
      return model_->forward(tokens, positions, kv_caches, params);
    }

    // Check if captured graph exists for this bucket num_tokens
    auto it = graphs_.find(bucket_num_tokens);
    if (it != graphs_.end()) {
      // Replay the existing graph
      VLOG(kGraphExecutorLogVerboseLevel)
          << "CudaGraphExecutorImpl::run() in decode replay mode";
      auto result = it->second->replay(tokens, positions, kv_caches, params);
      return attach_aux_hidden_states_if_needed(result.hidden_states, n_tokens);
    }

    // Graph doesn't exist for this bucket num_tokens, try to create it lazily
    auto graph =
        std::make_unique<CudaGraph>(*persistent_param_,
                                    device_.index(),
                                    get_capture_stream(device_.index()));
    VLOG(kGraphExecutorLogVerboseLevel)
        << "CudaGraphExecutorImpl::run() in decode capture mode";

    TorchMemPool* pool_ptr = nullptr;
    if (::xllm::ExecutionConfig::get_instance().enable_graph_vmm_pool()) {
      reset_vmm_allocator_offset(kPhysicalPoolIdDecode);
      const uint32_t shape_id = bucket_num_tokens;
      pool_ptr = get_or_create_vmm_mempool(kPhysicalPoolIdDecode, shape_id);
    }
    const at::cuda::MempoolId_t mem_pool =
        get_mem_pool(kPhysicalPoolIdDecode, bucket_num_tokens);

    bool capture_success = graph->capture(model_,
                                          args_,
                                          options_,
                                          tokens,
                                          positions,
                                          params,
                                          kv_caches,
                                          bucket_num_tokens,
                                          mem_pool,
                                          pool_ptr);

    if (capture_success) {
      LOG(INFO) << "Lazy capturing CUDA graph for bucket num_tokens: "
                << bucket_num_tokens << " (actual num_tokens: " << n_tokens
                << ") done";

      log_graph_memory_after_capture();

      // Save the graph for future reuse
      graphs_[bucket_num_tokens] = std::move(graph);

      // Run replay after capture so first request uses same execution path as
      // subsequent requests.
      auto result = graphs_[bucket_num_tokens]->replay(
          tokens, positions, kv_caches, params);
      return attach_aux_hidden_states_if_needed(result.hidden_states, n_tokens);
    }

    // Same fail-fast policy as prefill capture above: keep graph-mode behavior
    // explicit and avoid silently switching execution semantics after a capture
    // failure. Use ::xllm::ExecutionConfig::get_instance().enable_graph() to
    // turn off graph mode if eager fallback is desired for resiliency.
    LOG(FATAL) << "Failed to capture CUDA graph for bucket num_tokens: "
               << bucket_num_tokens << " (actual num_tokens: " << n_tokens
               << ")";
  }
  // Defensive fallback for unsupported forward types (should be unreachable for
  // normal prefill/decode paths).
  LOG(ERROR) << "Failed to capture CUDA graph for bucket num_tokens: "
             << bucket_num_tokens;
  COUNTER_INC(num_model_execution_total_eager);
  return model_->forward(tokens, positions, kv_caches, params);
}

// bucket will be [1, 2, 4, 8, 16, 32, 48, 64, ..., max_seqs_per_batch]
uint32_t CudaGraphExecutorImpl::get_bucket_num_tokens(uint32_t num_tokens,
                                                      bool is_prefill) const {
  return static_cast<uint32_t>(
      get_graph_bucket_num_tokens(num_tokens, is_prefill));
}

}  // namespace xllm::runtime::cuda

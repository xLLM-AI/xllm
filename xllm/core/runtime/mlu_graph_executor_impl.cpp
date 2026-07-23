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

#include "mlu_graph_executor_impl.h"

#include <cnrt.h>
#include <framework/core/caching_allocator.h>
#include <framework/core/stream_guard.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <iomanip>
#include <sstream>
#include <string>

#include "common/global_flags.h"
#include "common/metrics.h"
#include "core/common/constants.h"
#include "core/framework/config/execution_config.h"
#include "framework/model/causal_vlm.h"
#include "util/utils.h"
#include "vlm_executor_impl.h"

namespace {
struct GraphPoolMemoryUsage {
  std::size_t reserved_bytes = 0;
  std::size_t allocated_bytes = 0;
  std::size_t active_bytes = 0;
  std::size_t segment_count = 0;
};

std::size_t tensor_bytes(const torch::Tensor& tensor) {
  if (!tensor.defined()) {
    return 0;
  }

  return static_cast<std::size_t>(tensor.numel()) * tensor.element_size();
}

std::string format_memory_size(std::size_t bytes) {
  if (bytes < 1024) {
    return std::to_string(bytes) + " B";
  }

  double value = static_cast<double>(bytes) / 1024.0;
  std::string unit = " KiB";
  if (value >= 1024.0) {
    value /= 1024.0;
    unit = " MiB";
  }
  if (value >= 1024.0) {
    value /= 1024.0;
    unit = " GiB";
  }

  std::ostringstream oss;
  oss << std::fixed << std::setprecision(2) << value << unit;
  return oss.str();
}

GraphPoolMemoryUsage get_graph_pool_usage(
    c10::DeviceIndex device_index,
    const torch_mlu::MempoolId_t& pool_id) {
  GraphPoolMemoryUsage usage;
  const auto snapshot = torch_mlu::MLUCachingAllocator::snapshot();
  for (const auto& segment : snapshot.segments) {
    if (segment.device != device_index ||
        segment.owner_private_pool_id != pool_id) {
      continue;
    }
    usage.reserved_bytes += segment.total_size;
    usage.allocated_bytes += segment.allocated_size;
    usage.active_bytes += segment.active_size;
    usage.segment_count += 1;
  }
  return usage;
}

// bucket will be [1, 2, 4, 8, 16, 32, 48, 64, ..., max_seqs_per_batch]
uint32_t get_bucket_num_tokens(uint32_t num_tokens) {
  if (::xllm::ExecutionConfig::get_instance()
          .enable_graph_mode_decode_no_padding()) {
    return num_tokens;
  }
  const uint32_t graph_step = 16;
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

  return ((num_tokens + graph_step - 1) / graph_step) * graph_step;
}

xllm::ModelOutput make_graph_output(const torch::Tensor& hidden_states,
                                    const torch::Tensor& aux_hidden_states,
                                    bool enable_aux_hidden_states) {
  if (enable_aux_hidden_states && aux_hidden_states.defined() &&
      aux_hidden_states.numel() > 0) {
    return xllm::ModelOutput(hidden_states, torch::Tensor(), aux_hidden_states);
  }
  return xllm::ModelOutput(hidden_states);
}

enum class RunMode : int8_t {
  kGraph = 0,
  kPaddedDpGraph,
  kDraft,
  kNonDecode,
  kDummy,
  kUnevenDp,
  kMixedDp,
  kBadDpMeta,
};

bool has_zero_tokens(const std::vector<int32_t>& dp_token_nums) {
  return std::any_of(dp_token_nums.begin(),
                     dp_token_nums.end(),
                     [](int32_t token_num) { return token_num == 0; });
}

bool dp_tokens_equal(const std::vector<int32_t>& dp_token_nums) {
  return dp_token_nums.empty() ||
         std::all_of(
             dp_token_nums.begin(),
             dp_token_nums.end(),
             [first_token_num = dp_token_nums.front()](int32_t token_num) {
               return token_num == first_token_num;
             });
}

bool allow_graph(RunMode run_mode) {
  return run_mode == RunMode::kGraph || run_mode == RunMode::kPaddedDpGraph;
}

uint32_t align_tokens(uint32_t tokens, uint32_t align) {
  CHECK_GT(align, 0U) << "align must be positive";
  uint32_t rem = tokens % align;
  return rem == 0 ? tokens : tokens + align - rem;
}

uint32_t get_tp_size(const xllm::runtime::Options& options) {
  int32_t world_size = options.world_size();
  int32_t dp_size = options.dp_size();
  if (world_size <= 1 || dp_size <= 1 || world_size < dp_size ||
      world_size % dp_size != 0) {
    return 1;
  }

  return static_cast<uint32_t>(world_size / dp_size);
}

int64_t get_graph_token_capacity(const xllm::runtime::Options& options) {
  int64_t capacity = options.max_seqs_per_batch();

  if (options.enable_speculative_decode() && !options.is_draft_engine()) {
    capacity *= options.num_decoding_tokens();
  }

  capacity = static_cast<int64_t>(get_bucket_num_tokens(
      static_cast<uint32_t>(std::max<int64_t>(capacity, 1))));

  const uint32_t tp_size = get_tp_size(options);
  capacity = static_cast<int64_t>(align_tokens(
      static_cast<uint32_t>(std::max<int64_t>(capacity, tp_size)), tp_size));

  return capacity;
}

uint32_t get_graph_dp_tokens(uint32_t actual_tokens,
                             const xllm::ModelInputParams& params,
                             const xllm::runtime::Options& options) {
  if (params.parallel.dp_global_token_nums.size() <= 1) {
    return get_bucket_num_tokens(actual_tokens);
  }

  const auto max_token_num =
      std::max_element(params.parallel.dp_global_token_nums.begin(),
                       params.parallel.dp_global_token_nums.end());
  CHECK(max_token_num != params.parallel.dp_global_token_nums.end())
      << "dp_global_token_nums is empty";
  uint32_t bucket_tokens =
      get_bucket_num_tokens(static_cast<uint32_t>(*max_token_num));
  uint32_t tp_size = get_tp_size(options);
  return align_tokens(std::max(bucket_tokens, tp_size), tp_size);
}

xllm::ModelInputParams make_graph_params(const xllm::ModelInputParams& params,
                                         uint32_t padding_num_tokens) {
  xllm::ModelInputParams graph_params = params;
  if (params.parallel.dp_global_token_nums.size() > 1) {
    graph_params.parallel.dp_global_token_nums =
        std::vector<int32_t>(params.parallel.dp_global_token_nums.size(),
                             static_cast<int32_t>(padding_num_tokens));
  }
  return graph_params;
}

RunMode get_run_mode(const xllm::runtime::Options& options,
                     const xllm::ModelInputParams& params) {
  if (options.is_draft_engine()) {
    return RunMode::kDraft;
  }

  if (!params.meta.batch_forward_type.is_decode()) {
    return RunMode::kNonDecode;
  }

  if (params.meta.q_max_seq_len == 0) {
    return RunMode::kDummy;
  }

  if (params.parallel.dp_global_token_nums.size() <= 1) {
    return RunMode::kGraph;
  }

  if (has_zero_tokens(params.parallel.dp_global_token_nums)) {
    return RunMode::kDummy;
  }

  if (params.parallel.dp_is_decode.size() !=
      params.parallel.dp_global_token_nums.size()) {
    return RunMode::kBadDpMeta;
  }

  if (std::find(params.parallel.dp_is_decode.begin(),
                params.parallel.dp_is_decode.end(),
                0) != params.parallel.dp_is_decode.end()) {
    return RunMode::kMixedDp;
  }

  if (!dp_tokens_equal(params.parallel.dp_global_token_nums)) {
    if (params.meta.q_max_seq_len == 1) {
      return RunMode::kPaddedDpGraph;
    }
    return RunMode::kUnevenDp;
  }

  return RunMode::kGraph;
}

}  // namespace

namespace xllm::mlu {

GraphPersistentParam::GraphPersistentParam(const ModelArgs& args,
                                           const torch::Device& device,
                                           const runtime::Options& options)
    : num_decoding_tokens_(options.num_decoding_tokens()) {
  const int64_t max_tokens = options.max_tokens_per_batch();
  const int64_t graph_tokens_capacity = get_graph_token_capacity(options);
  // Sequence lengths are cumulative offsets for graph token rows, including
  // the terminal offset.
  const int64_t max_seq_lens = graph_tokens_capacity + 1;
  const int64_t max_seq_len = args.max_position_embeddings();
  const uint32_t block_size = options.block_size();
  const int64_t max_num_blocks_per_req =
      (max_seq_len + block_size - 1) / block_size + 1;
  torch::ScalarType torch_type = util::parse_dtype(args.dtype(), device);
  auto tensor_options = torch::TensorOptions().device(device).dtype(torch_type);
  auto int_tensor_options = tensor_options.dtype(torch::kInt32);

  // output buffer
  output_ = torch::zeros({max_tokens, args.hidden_size()}, tensor_options);
  // aux_hidden_states will be lazily initialized when needed

  // input buffers
  if (args.rope_scaling_mrope_section().empty()) {
    positions_ = torch::zeros({max_tokens}, int_tensor_options);
  } else {
    positions_ = torch::zeros({3, max_tokens}, int_tensor_options);
    use_mrope_ = true;
  }
  tokens_ = torch::zeros({max_tokens}, int_tensor_options);
  new_cache_slots_ = torch::zeros({max_tokens}, int_tensor_options);
  block_table_ = torch::zeros({graph_tokens_capacity, max_num_blocks_per_req},
                              int_tensor_options);
  // MTP validate expands decode rows from N to N * (K + 1), where K is the
  // speculative token count. Draft-extend only doubles rows, so the same
  // bound covers both paths when speculative decode is enabled.
  q_seq_lens_ = torch::zeros({max_seq_lens}, int_tensor_options);
  kv_seq_lens_ = torch::zeros({max_seq_lens}, int_tensor_options);
  // Padding decode rows can still execute stateful graph kernels. Point them
  // at the reserved padding slot so they cannot update a live request state.
  linear_state_indices_ =
      torch::full({max_seq_lens}, kPaddingLinearStateId, int_tensor_options);
}

void GraphPersistentParam::init_params(const ModelInputParams& params,
                                       uint32_t padding_num_tokens,
                                       uint32_t padding_needed) {
  params_ = params.to(tokens_.device());
  params_.enable_graph = true;
  params_.attention.device.q_seq_lens = q_seq_lens_.slice(
      0, 0, params.attention.device.q_seq_lens.size(0) + padding_needed);
  params_.attention.device.kv_seq_lens = kv_seq_lens_.slice(
      0, 0, params.attention.device.kv_seq_lens.size(0) + padding_needed);
  params_.attention.device.new_cache_slots =
      new_cache_slots_.slice(0, 0, padding_num_tokens);
  params_.attention.device.block_tables =
      block_table_.slice(0, 0, padding_num_tokens);
  if (params.embedding.input_embedding.defined()) {
    if (!input_embeds_.defined()) {
      input_embeds_ = torch::zeros_like(output_);
    }
    params_.embedding.input_embedding =
        input_embeds_.slice(0, 0, padding_num_tokens);
  }

  if (!params.embedding.linear_state_ids.empty()) {
    params_.embedding.linear_state_ids = params.embedding.linear_state_ids;
    params_.embedding.linear_state_indices =
        linear_state_indices(padding_num_tokens);
  }
}

void GraphPersistentParam::update_input_buffer(const torch::Tensor& tokens,
                                               const torch::Tensor& positions,
                                               const ModelInputParams& params,
                                               uint32_t padding_needed) {
  // Copy data from input parameters to persistent graph tensors
  int32_t slice_dim = use_mrope_ ? 1 : 0;
  const int64_t actual_tokens = tokens.size(0);
  const int64_t padded_tokens = actual_tokens + padding_needed;
  const int64_t actual_batch =
      params.attention.device.block_tables.defined()
          ? params.attention.device.block_tables.size(0)
          : (!params.multi_block_tables.empty()
                 ? params.multi_block_tables[0].size(0)
                 : actual_tokens);
  const int64_t block_rows_end = actual_batch + padding_needed;
  auto position_slice =
      positions_.slice(slice_dim, 0, positions.size(slice_dim));
  auto token_slice = tokens_.slice(0, 0, tokens.size(0));
  auto cache_slot_slice = new_cache_slots_.slice(
      0, 0, params.attention.device.new_cache_slots.size(0));
  position_slice.copy_(positions, true);
  token_slice.copy_(tokens, true);
  cache_slot_slice.copy_(params.attention.device.new_cache_slots, true);
  if (padding_needed > 0) {
    positions_.slice(slice_dim, actual_tokens, padded_tokens).zero_();
    tokens_.slice(0, actual_tokens, padded_tokens).zero_();
    new_cache_slots_.slice(0, actual_tokens, padded_tokens).zero_();
  }
  params_.meta.num_sequences = params.meta.num_sequences;

  // Apply padding if required number of tokens exceeds actual input
  // Generate padded sequence lengths by extending the last valid value
  std::vector<int32_t> q_seq_lens_vec(params.attention.host.q_seq_lens);
  std::vector<int32_t> kv_seq_lens_vec(params.attention.host.kv_seq_lens);
  if (padding_needed > 0) {
    q_seq_lens_vec.reserve(q_seq_lens_vec.size() + padding_needed);
    kv_seq_lens_vec.reserve(kv_seq_lens_vec.size() + padding_needed);
    for (size_t i = 0; i < padding_needed; i++) {
      q_seq_lens_vec.push_back(q_seq_lens_vec.back() + num_decoding_tokens_);
      kv_seq_lens_vec.push_back(kv_seq_lens_vec.back() + num_decoding_tokens_);
    }
  }

  params_.attention.host.q_seq_lens = q_seq_lens_vec;
  params_.attention.host.kv_seq_lens = kv_seq_lens_vec;

  auto q_seq_lens = torch::tensor(q_seq_lens_vec, q_seq_lens_.options());
  auto kv_seq_lens = torch::tensor(kv_seq_lens_vec, kv_seq_lens_.options());
  auto q_seq_slice = q_seq_lens_.slice(0, 0, q_seq_lens.size(0));
  auto kv_seq_slice = kv_seq_lens_.slice(0, 0, kv_seq_lens.size(0));
  q_seq_slice.copy_(q_seq_lens, true);
  kv_seq_slice.copy_(kv_seq_lens, true);

  // Copy block table data
  if (params.attention.device.block_tables.defined()) {
    const int64_t actual_block_batch =
        params.attention.device.block_tables.size(0);
    const int64_t actual_n_block = params.attention.device.block_tables.size(1);
    auto slice_block_tables = block_table_.slice(0, 0, actual_block_batch)
                                  .slice(1, 0, actual_n_block);
    slice_block_tables.copy_(params.attention.device.block_tables, true);
    if (actual_n_block < block_table_.size(1)) {
      block_table_.slice(0, 0, actual_block_batch)
          .slice(1, actual_n_block, block_table_.size(1))
          .zero_();
    }
    if (block_rows_end > actual_block_batch) {
      block_table_.slice(0, actual_block_batch, block_rows_end).zero_();
    }
  }

  if (!params.multi_block_tables.empty()) {
    params_.multi_block_tables = params.multi_block_tables;
  }

  if (params.embedding.input_embedding.defined()) {
    auto input_embed_slice =
        input_embeds_.slice(0, 0, params.embedding.input_embedding.size(0));
    input_embed_slice.copy_(params.embedding.input_embedding, true);
    if (padding_needed > 0) {
      input_embeds_
          .slice(0, params.embedding.input_embedding.size(0), padded_tokens)
          .zero_();
    }
  }

  if (!params.embedding.linear_state_ids.empty()) {
    const int64_t actual_batch_size = params.embedding.linear_state_ids.size();
    if (params.embedding.linear_state_indices.defined()) {
      linear_state_indices_
          .slice(/*dim=*/0, /*start=*/0, /*end=*/actual_batch_size)
          .copy_(params.embedding.linear_state_indices, /*non_blocking=*/true);
    } else {
      linear_state_indices_
          .slice(/*dim=*/0, /*start=*/0, /*end=*/actual_batch_size)
          .copy_(torch::tensor(params.embedding.linear_state_ids,
                               linear_state_indices_.options()),
                 /*non_blocking=*/true);
    }
    if (padded_tokens > actual_batch_size) {
      linear_state_indices_
          .slice(/*dim=*/0, /*start=*/actual_batch_size, /*end=*/padded_tokens)
          .fill_(kPaddingLinearStateId);
    }
    params_.embedding.linear_state_ids = params.embedding.linear_state_ids;
    params_.embedding.linear_state_indices =
        linear_state_indices(padded_tokens);
  }
}

std::size_t GraphPersistentParam::get_persistent_tensor_bytes() const {
  std::size_t total = 0;

  total += tensor_bytes(output_);
  total += tensor_bytes(positions_);
  total += tensor_bytes(tokens_);
  total += tensor_bytes(new_cache_slots_);
  total += tensor_bytes(block_table_);
  total += tensor_bytes(q_seq_lens_);
  total += tensor_bytes(kv_seq_lens_);
  total += tensor_bytes(input_embeds_);
  total += tensor_bytes(aux_hidden_states_);

  return total;
}

MluGraph::MluGraph(GraphPersistentParam* persistent_param,
                   uint32_t padding_num_tokens)
    : persistent_param_(persistent_param),
      padding_num_tokens_(padding_num_tokens) {}

void MluGraph::prepare_model_graph_metadata(CausalLM* model,
                                            const ModelInputParams& params) {
  if (!model->requires_graph_forward_metadata()) {
    return;
  }

  if (!model_graph_metadata_state_) {
    model_graph_metadata_state_ = model->create_graph_forward_metadata_state();
  }
  auto graph_params = make_graph_params(params, padding_num_tokens_);
  int32_t slice_dim = persistent_param_->use_mrope_ ? 1 : 0;
  model->prepare_graph_forward_metadata(
      model_graph_metadata_state_.get(),
      persistent_param_->positions_.slice(slice_dim, 0, padding_num_tokens_),
      graph_params);

  persistent_param_->params_.attn_metadata = graph_params.attn_metadata;
}

void MluGraph::capture(CausalLM* model,
                       std::vector<KVCache>& kv_cache,
                       const torch_mlu::MempoolId_t& pool,
                       const torch_mlu::MLUStream& capture_stream,
                       const runtime::Options& options) {
  int32_t slice_dim = persistent_param_->use_mrope_ ? 1 : 0;
  torch_mlu::synchronize();
  auto prev_stream = torch_mlu::getCurrentMLUStream();
  torch_mlu::mlu::MLUStreamGuard guard(capture_stream);
  graph_ = torch_mlu::MLUGraph();
  graph_.capture_begin(pool, cnrtQueueCaptureModeRelaxed);
  auto forward_result = model->forward(
      persistent_param_->tokens_.slice(0, 0, padding_num_tokens_),
      persistent_param_->positions_.slice(slice_dim, 0, padding_num_tokens_),
      kv_cache,
      persistent_param_->params_);
  persistent_param_->output_.slice(0, 0, forward_result.hidden_states.size(0))
      .copy_(forward_result.hidden_states, true);
  // Only capture aux_hidden_states when enable_graph_aux_hidden_states is on
  // (e.g. main worker in EAGLE-3); draft worker has this option false.
  if (options.enable_graph_aux_hidden_states() &&
      forward_result.aux_hidden_states.defined()) {
    if (persistent_param_->aux_hidden_states_.numel() == 0) {
      // Lazy initialization
      auto shape = forward_result.aux_hidden_states.sizes().vec();
      shape[0] = persistent_param_->output_.size(0);
      persistent_param_->aux_hidden_states_ =
          torch::zeros(shape, persistent_param_->output_.options());
    }
    auto slice = persistent_param_->aux_hidden_states_.slice(
        0, 0, forward_result.aux_hidden_states.size(0));
    if (slice.sizes() == forward_result.aux_hidden_states.sizes()) {
      slice.copy_(forward_result.aux_hidden_states, true);
    }
  }
  graph_.capture_end();
  torch_mlu::setCurrentMLUStream(prev_stream);
  torch_mlu::synchronize();
  graph_.replay();
}

ModelOutput MluGraph::replay() {
  graph_.replay();
  const uint32_t actual_tokens = padding_num_tokens_;
  // Note: aux_hidden_states handling is done in MluGraphExecutorImpl::run()
  // since replay() doesn't have access to options
  return ModelOutput(persistent_param_->output_.slice(0, 0, actual_tokens));
}

void MluGraph::update_input_buffer(CausalLM* model,
                                   const torch::Tensor& tokens,
                                   const torch::Tensor& positions,
                                   const ModelInputParams& params,
                                   bool is_init) {
  uint32_t padding_needed = padding_num_tokens_ - tokens.size(0);
  if (is_init) {
    persistent_param_->init_params(params, padding_num_tokens_, padding_needed);
  }
  persistent_param_->update_input_buffer(
      tokens, positions, params, padding_needed);
  // For some models (e.g. DeepSeekV4), the metadata depends on variable host
  // data, which needs to be updated outside of capture.
  prepare_model_graph_metadata(model, params);
}

MluGraphExecutorImpl::MluGraphExecutorImpl(CausalLM* model,
                                           const ModelArgs& args,
                                           const torch::Device& device,
                                           const runtime::Options& options)
    : model_(model),
      args_(args),
      device_(device),
      options_(options),
      graph_pool_(torch_mlu::graph_pool_handle()) {
  max_tokens_for_graph_mode_ =
      ::xllm::ExecutionConfig::get_instance().max_tokens_for_graph_mode();
  if (max_tokens_for_graph_mode_ < options_.max_seqs_per_batch()) {
    max_tokens_for_graph_mode_ = options_.max_seqs_per_batch();
  }
}

ForwardInput MluGraphExecutorImpl::prepare_inputs(Batch& batch) {
  return batch.prepare_forward_input(
      options_.num_decoding_tokens(), 0, args_, options_.cp_size());
}

ModelOutput MluGraphExecutorImpl::run_eager(const torch::Tensor& tokens,
                                            const torch::Tensor& positions,
                                            std::vector<KVCache>& kv_caches,
                                            const ModelInputParams& params) {
  RunMode run_mode = get_run_mode(options_, params);
  if (run_mode == RunMode::kDraft) {
    LOG_FIRST_N(INFO, 1) << "MLU graph fallback to eager for draft worker";
  } else if (run_mode == RunMode::kDummy) {
    LOG_FIRST_N(INFO, 1)
        << "MLU graph fallback to eager when decode inputs contain dummy run";
  } else if (run_mode == RunMode::kUnevenDp) {
    LOG_FIRST_N(INFO, 1)
        << "MLU graph fallback to eager for uneven dp decode batch";
  } else if (run_mode == RunMode::kMixedDp) {
    LOG_FIRST_N(INFO, 1)
        << "MLU graph fallback to eager for mixed dp prefill/decode batch";
  } else if (run_mode == RunMode::kBadDpMeta) {
    LOG_FIRST_N(WARNING, 1)
        << "MLU graph fallback to eager because dp_is_decode is invalid";
  }
  COUNTER_INC(num_model_execution_total_eager);
  ModelOutput result = model_->forward(tokens, positions, kv_caches, params);
  ModelOutput output =
      make_graph_output(result.hidden_states,
                        result.aux_hidden_states,
                        options_.enable_graph_aux_hidden_states());
  output.mtp_topk_state = std::move(result.mtp_topk_state);
  return output;
}

void MluGraphExecutorImpl::init_param_once() {
  if (persistent_param_ == nullptr) {
    persistent_param_ =
        std::make_unique<GraphPersistentParam>(args_, device_, options_);
  }
}

void MluGraphExecutorImpl::log_memory_after_capture() {
  std::size_t reserved_bytes = 0;
  std::size_t allocated_bytes = 0;
  std::size_t active_bytes = 0;
  std::size_t segment_count = 0;

  try {
    const GraphPoolMemoryUsage usage =
        get_graph_pool_usage(device_.index(), graph_pool_);
    reserved_bytes = usage.reserved_bytes;
    allocated_bytes = usage.allocated_bytes;
    active_bytes = usage.active_bytes;
    segment_count = usage.segment_count;
  } catch (const std::exception& e) {
    VLOG(1) << "Skip MLU graph memory usage log: " << e.what();
  } catch (...) {
    VLOG(1) << "Skip MLU graph memory usage log: unknown allocator error";
  }

  const std::size_t persistent_param_bytes =
      persistent_param_ ? persistent_param_->get_persistent_tensor_bytes() : 0;
  const std::size_t executor_total_bytes =
      reserved_bytes + persistent_param_bytes;

  // Per-capture delta of the shared graph pool. When scratch is being reused
  // across buckets, this collapses to ~0 after the first (largest) capture;
  // a steady positive delta means each bucket pins its own scratch instead.
  const bool reserved_grew = reserved_bytes >= last_pool_reserved_bytes_;
  const std::size_t reserved_delta =
      reserved_grew ? reserved_bytes - last_pool_reserved_bytes_
                    : last_pool_reserved_bytes_ - reserved_bytes;
  last_pool_reserved_bytes_ = reserved_bytes;
  peak_pool_reserved_bytes_ =
      std::max(peak_pool_reserved_bytes_, reserved_bytes);

  LOG(INFO) << "MluGraphExecutorMemory Usage:"
            << " executor_total_memory="
            << format_memory_size(executor_total_bytes) << " persistent_param="
            << format_memory_size(persistent_param_bytes)
            << " pool_reserved=" << format_memory_size(reserved_bytes)
            << " pool_reserved_delta=" << (reserved_grew ? "+" : "-")
            << format_memory_size(reserved_delta) << " pool_reserved_peak="
            << format_memory_size(peak_pool_reserved_bytes_)
            << " pool_segments=" << segment_count
            << " allocated_pool_memory=" << format_memory_size(allocated_bytes)
            << " active_pool_memory=" << format_memory_size(active_bytes);
}

// Main execution method with graph optimization for decode phase
// tokens: [num_decode_tokens]
// positions: [num_decode_tokens] token pos in the sequence
// returns: ModelOutput
ModelOutput MluGraphExecutorImpl::run(const torch::Tensor& tokens,
                                      const torch::Tensor& positions,
                                      std::vector<KVCache>& kv_caches,
                                      const ModelInputParams& params) {
  const RunMode run_mode = get_run_mode(options_, params);
  if (!allow_graph(run_mode)) {
    return run_eager(tokens, positions, kv_caches, params);
  }

  const uint32_t actual_tokens = static_cast<uint32_t>(tokens.size(0));
  const uint32_t graph_tokens =
      get_graph_dp_tokens(actual_tokens, params, options_);
  if (static_cast<int64_t>(graph_tokens) > max_tokens_for_graph_mode_) {
    LOG_FIRST_N(INFO, 1)
        << "MLU graph fallback to eager because graph bucket num_tokens "
        << graph_tokens << " exceeds max_tokens_for_graph_mode ("
        << max_tokens_for_graph_mode_ << ")";
    return run_eager(tokens, positions, kv_caches, params);
  }

  init_param_once();

  const ModelInputParams graph_params = make_graph_params(params, graph_tokens);

  if (graph_params.parallel.dp_global_token_nums !=
      params.parallel.dp_global_token_nums) {
    LOG_FIRST_N(INFO, 4) << "MLU graph padded dp decode path: raw "
                         << "dp_global_token_nums="
                         << params.parallel.dp_global_token_nums
                         << ", graph dp_global_token_nums="
                         << graph_params.parallel.dp_global_token_nums
                         << ", tp_size=" << get_tp_size(options_)
                         << ", graph_tokens=" << graph_tokens;
  }

  auto it = graphs_.find(graph_tokens);
  if (it != graphs_.end()) {
    MluGraph* cur_graph = it->second.get();
    cur_graph->update_input_buffer(model_, tokens, positions, graph_params);
    ModelOutput result = cur_graph->replay();
    // Return only the actual num_tokens portion
    auto hidden_states = result.hidden_states.slice(0, 0, actual_tokens);
    if (options_.enable_graph_aux_hidden_states()) {
      auto aux_hidden_states =
          persistent_param_->aux_hidden_states_.numel() > 0
              ? persistent_param_->aux_hidden_states_.slice(0, 0, actual_tokens)
              : torch::Tensor();
      return make_graph_output(
          hidden_states, aux_hidden_states, /*enable_aux_hidden_states=*/true);
    }
    return ModelOutput(hidden_states);
  }

  std::unique_ptr<MluGraph> graph =
      std::make_unique<MluGraph>(persistent_param_.get(), graph_tokens);
  graph->update_input_buffer(model_, tokens, positions, graph_params, true);
  if (!graph_capture_stream_.has_value()) {
    graph_capture_stream_ =
        torch_mlu::getStreamFromPool(/*isHighPriority=*/false, device_.index());
  }
  graph->capture(
      model_, kv_caches, graph_pool_, *graph_capture_stream_, options_);
  log_memory_after_capture();
  graphs_[graph_tokens] = std::move(graph);
  // Return the output from capture
  auto hidden_states = persistent_param_->output_.slice(0, 0, actual_tokens);
  if (options_.enable_graph_aux_hidden_states()) {
    auto aux_hidden_states =
        persistent_param_->aux_hidden_states_.numel() > 0
            ? persistent_param_->aux_hidden_states_.slice(0, 0, actual_tokens)
            : torch::Tensor();
    return make_graph_output(
        hidden_states, aux_hidden_states, /*enable_aux_hidden_states=*/true);
  }

  return ModelOutput(hidden_states);
}

}  // namespace xllm::mlu

/* Copyright 2025-2026 The xLLM Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/xLLM-AI/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "core/runtime/mlu_graph_executor_impl.h"

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
#include "compiler/triton/include/jit_kernel.h"
#include "core/common/constants.h"
#include "core/framework/config/execution_config.h"
#include "core/framework/model/seq_lens.h"
#include "core/framework/parallel_state/process_group.h"
#include "core/runtime/graph_execution_input.h"
#include "core/util/tensor_options.h"
#include "runtime/decode_graph_bucket.h"
#include "util/utils.h"

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

xllm::ModelOutput make_graph_output(const torch::Tensor& hidden_states,
                                    const torch::Tensor& aux_hidden_states,
                                    bool enable_aux_hidden_states) {
  if (enable_aux_hidden_states && aux_hidden_states.defined() &&
      aux_hidden_states.numel() > 0) {
    return xllm::ModelOutput(hidden_states, torch::Tensor(), aux_hidden_states);
  }
  return xllm::ModelOutput(hidden_states);
}

int64_t extra_rows_of(const xllm::mlu::GraphLayout& layout) {
  return (layout.padded_num_reqs - layout.num_reqs) *
         layout.tokens_per_request / layout.attention_row_query_len;
}

xllm::GraphExecutionInput<xllm::ModelParamsView> prepare_graph_input(
    const xllm::ModelParamsView& params,
    int64_t padding_num_tokens) {
  xllm::GraphExecutionInput<xllm::ModelParamsView> graph_input(params);
  graph_input.pad_dp_token_counts(static_cast<int32_t>(padding_num_tokens));
  return graph_input;
}

enum class GraphAction : int8_t { EAGER, CAPTURE, REPLAY };

// Compare fields directly so hash collisions cannot split DP graph actions.
constexpr int64_t kGraphPacketFields = 12;
constexpr int64_t kGraphPacketSize = 32;

GraphAction agree_graph_action(const xllm::mlu::GraphPlan* plan,
                               bool cache_hit,
                               xllm::ProcessGroup* dp_group,
                               const torch::Device& device,
                               xllm::mlu::GraphActionWorkspace& workspace) {
  if (dp_group == nullptr || dp_group->world_size() <= 1) {
    return plan == nullptr
               ? GraphAction::EAGER
               : (cache_hit ? GraphAction::REPLAY : GraphAction::CAPTURE);
  }
  if (!workspace.input_device.defined()) {
    const auto options = torch::TensorOptions().dtype(torch::kInt64);
    workspace.input_host = torch::empty({kGraphPacketSize}, options);
    workspace.input_device =
        torch::empty({kGraphPacketSize}, options.device(device));
    workspace.output_device = torch::empty(
        {dp_group->world_size(), kGraphPacketSize}, options.device(device));
    workspace.output_host =
        torch::empty({dp_group->world_size(), kGraphPacketSize}, options);
  }
  int64_t* packet = workspace.input_host.data_ptr<int64_t>();
  std::fill_n(packet, kGraphPacketSize, 0);
  packet[0] = plan != nullptr;
  if (plan != nullptr) {
    CHECK_LE(plan->key.multi_block_table_column_counts.size(),
             static_cast<std::size_t>(kGraphPacketSize - kGraphPacketFields));
    packet[1] = cache_hit ? 1 : 0;
    packet[2] = plan->layout.padded_num_tokens;
    packet[3] = plan->layout.padded_num_reqs;
    packet[4] = plan->layout.tokens_per_request;
    packet[5] = plan->layout.attention_row_query_len;
    packet[6] = plan->layout.cache_pad_slot;
    packet[7] = plan->history_capacity;
    packet[8] = plan->main_block_table_columns;
    packet[9] = plan->key.has_input_embedding;
    packet[10] =
        static_cast<int64_t>(plan->key.multi_block_table_column_counts.size());
    packet[11] = plan->key.grouped;
    std::copy(plan->key.multi_block_table_column_counts.begin(),
              plan->key.multi_block_table_column_counts.end(),
              packet + kGraphPacketFields);
  }
  workspace.input_device.copy_(workspace.input_host);
  dp_group
      ->allgather_base_async(workspace.input_device, workspace.output_device)
      ->wait();
  workspace.output_host.copy_(workspace.output_device);
  const int64_t* values = workspace.output_host.data_ptr<int64_t>();
  bool all_hit = true;
  for (int32_t rank = 0; rank < dp_group->world_size(); ++rank) {
    const int64_t* peer = values + rank * kGraphPacketSize;
    if (peer[0] == 0) {
      return GraphAction::EAGER;
    }
    if (!std::equal(packet + 2, packet + kGraphPacketSize, peer + 2)) {
      LOG_FIRST_N(WARNING, 4)
          << "MLU graph layout differs across dp ranks; this step runs eager "
             "on every rank";
      return GraphAction::EAGER;
    }
    all_hit = all_hit && peer[1] != 0;
  }
  return all_hit ? GraphAction::REPLAY : GraphAction::CAPTURE;
}

void log_eager_reason(xllm::mlu::EagerReason reason) {
  switch (reason) {
    case xllm::mlu::EagerReason::NON_DECODE:
      LOG_FIRST_N(INFO, 1)
          << "MLU graph fallback to eager for non-decode or draft step";
      break;
    case xllm::mlu::EagerReason::UNSUPPORTED_INPUT:
      LOG_FIRST_N(INFO, 1)
          << "MLU graph fallback to eager for unsupported input layout";
      break;
    case xllm::mlu::EagerReason::INVALID_LAYOUT:
      LOG_FIRST_N(INFO, 1)
          << "MLU graph fallback to eager for a non-uniform step layout";
      break;
    case xllm::mlu::EagerReason::TOKEN_LIMIT:
      LOG_FIRST_N(INFO, 1)
          << "MLU graph fallback to eager because the padded token bucket "
             "exceeds max_tokens_for_graph_mode";
      break;
    case xllm::mlu::EagerReason::HISTORY_LIMIT:
      LOG_FIRST_N(INFO, 1)
          << "MLU graph fallback to eager because kv history exceeds the "
             "captured graph capacity";
      break;
  }
}

}  // namespace

namespace xllm::mlu {

GraphPersistentParam::GraphPersistentParam(
    const ModelArgs& args,
    const torch::Device& device,
    const runtime::Options& options,
    const MtpModelCapabilities& capabilities) {
  const int64_t graph_tokens_capacity =
      get_graph_token_capacity(options, capabilities);
  const int64_t max_tokens =
      std::max<int64_t>(options.max_tokens_per_batch(), graph_tokens_capacity);
  const int64_t max_seq_lens = graph_tokens_capacity;
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
  q_seq_lens_ = torch::zeros({max_seq_lens}, int_tensor_options);
  kv_seq_lens_ = torch::zeros({max_seq_lens}, int_tensor_options);
  // Padding decode rows can still execute stateful graph kernels. Point them
  // at the reserved padding slot so they cannot update a live request state.
  linear_state_indices_ =
      torch::full({max_seq_lens}, kPaddingLinearStateId, int_tensor_options);
}

void GraphPersistentParam::bind_buffers(ModelParamsView& graph_params,
                                        const GraphPlan& plan) {
  const GraphLayout& layout = plan.layout;
  const int64_t num_seqs =
      layout.padded_num_tokens / layout.attention_row_query_len;
  auto& attention = graph_params.attention.device;
  attention.q_seq_lens = q_seq_lens_.slice(0, 0, num_seqs);
  attention.q_cu_seq_lens = torch::Tensor();
  attention.kv_seq_lens = kv_seq_lens_.slice(0, 0, num_seqs);
  attention.new_cache_slots =
      new_cache_slots_.slice(0, 0, layout.padded_num_tokens);
  if (attention.block_tables.defined()) {
    const int64_t columns = plan.main_block_table_columns > 0
                                ? plan.main_block_table_columns
                                : attention.block_tables.size(1);
    attention.block_tables =
        block_table_.slice(0, 0, num_seqs).slice(1, 0, columns);
  }
  if (graph_params.embedding.input_embedding.defined()) {
    if (!input_embeds_.defined()) {
      input_embeds_ = torch::zeros_like(output_);
    }
    graph_params.embedding.input_embedding =
        input_embeds_.slice(0, 0, layout.padded_num_tokens);
  }
  if (!graph_params.embedding.linear_state_ids.empty()) {
    graph_params.embedding.linear_state_indices =
        linear_state_indices_.slice(0, 0, layout.padded_num_reqs);
  }
  // The EPLB decode mask is rebuilt per step and cannot be replayed.
  graph_params.expert.eplb_decode_token_mask = torch::Tensor();
}

void GraphPersistentParam::update_input_buffer(const torch::Tensor& tokens,
                                               const torch::Tensor& positions,
                                               const ModelParamsView& params,
                                               ModelParamsView& graph_params,
                                               const GraphPlan& plan) {
  const GraphLayout& layout = plan.layout;
  const int64_t extra_rows = extra_rows_of(layout);
  const int32_t dim = use_mrope_ ? 1 : 0;
  const auto copy_rows = [](const torch::Tensor& dst,
                            const torch::Tensor& src,
                            int64_t fill,
                            int64_t dim = 0) {
    if (!dst.defined()) {
      return;
    }
    const int64_t rows = src.defined() ? src.size(dim) : 0;
    if (rows < dst.size(dim)) {
      dst.narrow(dim, rows, dst.size(dim) - rows).fill_(fill);
    }
    if (src.defined()) {
      dst.narrow(dim, 0, rows).copy_(src, true);
    }
  };
  const auto copy_host = [](const torch::Tensor& dst,
                            const std::vector<int32_t>& values) {
    dst.copy_(
        torch::tensor(values, torch::TensorOptions().dtype(torch::kInt32)),
        true);
  };

  copy_rows(tokens_.slice(0, 0, layout.padded_num_tokens), tokens, 0);
  copy_rows(
      positions_.slice(dim, 0, layout.padded_num_tokens), positions, 0, dim);

  auto& host = graph_params.attention.host;
  host.q_seq_lens = params.attention.host.q_seq_lens;
  host.kv_seq_lens = params.attention.host.kv_seq_lens;
  host.kpool_query_lens = params.attention.host.kpool_query_lens;
  util::pad_uniform_decode_lens(host.q_seq_lens,
                                host.kv_seq_lens,
                                extra_rows,
                                layout.attention_row_query_len - 1);
  if (!host.kpool_query_lens.empty()) {
    host.kpool_query_lens.resize(layout.padded_num_reqs,
                                 layout.tokens_per_request);
  }

  auto& attention = graph_params.attention.device;
  copy_rows(attention.q_seq_lens,
            params.attention.device.q_seq_lens,
            layout.attention_row_query_len);
  copy_rows(attention.kv_seq_lens, params.attention.device.kv_seq_lens, 0);

  copy_rows(attention.new_cache_slots,
            params.attention.device.new_cache_slots,
            layout.cache_pad_slot);

  if (attention.block_tables.defined()) {
    const auto& source_table = params.attention.device.block_tables;
    const int64_t block_rows = source_table.size(0);
    const int64_t copy_width =
        std::min(attention.block_tables.size(1), source_table.size(1));
    if (block_rows > 0 && copy_width > 0) {
      attention.block_tables.narrow(0, 0, block_rows)
          .narrow(1, 0, copy_width)
          .copy_(source_table.narrow(1, 0, copy_width), true);
    }
    if (copy_width < attention.block_tables.size(1)) {
      attention.block_tables.narrow(0, 0, block_rows)
          .narrow(1, copy_width, attention.block_tables.size(1) - copy_width)
          .zero_();
    }
    if (block_rows < attention.block_tables.size(0)) {
      attention.block_tables
          .narrow(0, block_rows, attention.block_tables.size(0) - block_rows)
          .zero_();
    }
  }

  if (params.embedding.linear_state_indices.defined() ||
      params.embedding.linear_state_ids.empty()) {
    copy_rows(graph_params.embedding.linear_state_indices,
              params.embedding.linear_state_indices,
              kPaddingLinearStateId);
  } else {
    graph_params.embedding.linear_state_indices.fill_(kPaddingLinearStateId);
    copy_host(graph_params.embedding.linear_state_indices.narrow(
                  0, 0, layout.num_reqs),
              params.embedding.linear_state_ids);
  }
  copy_rows(graph_params.embedding.input_embedding,
            params.embedding.input_embedding,
            0);
  copy_rows(graph_params.num_accepted_tokens, params.num_accepted_tokens, 1);
  if (graph_params.num_accepted_tokens.defined()) {
    graph_params.num_accepted_tokens_host = params.num_accepted_tokens_host;
    graph_params.num_accepted_tokens_host.resize(layout.padded_num_reqs, 1);
  }
  if (plan.key.grouped) {
    for (std::size_t i = 0; i < graph_params.multi_block_tables.size(); ++i) {
      copy_rows(
          graph_params.multi_block_tables[i], params.multi_block_tables[i], 0);
    }
  } else {
    graph_params.multi_block_tables = params.multi_block_tables;
  }

  if (!graph_params.embedding.linear_state_ids.empty()) {
    graph_params.embedding.linear_state_ids = params.embedding.linear_state_ids;
    graph_params.embedding.linear_state_ids.resize(layout.padded_num_reqs,
                                                   kPaddingLinearStateId);
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
  total += tensor_bytes(linear_state_indices_);
  return total;
}

MluGraph::MluGraph(GraphPersistentParam* persistent_param, GraphPlan plan)
    : plan_(std::move(plan)), persistent_param_(persistent_param) {}

void MluGraph::prepare_model_graph_metadata(CausalLM* model,
                                            const ModelParamsView& params) {
  if (!model->requires_graph_forward_metadata()) {
    return;
  }

  if (!model_graph_metadata_state_) {
    model_graph_metadata_state_ = model->create_graph_forward_metadata_state();
  }
  auto graph_input =
      prepare_graph_input(params, plan_.layout.padded_num_tokens);
  ModelParamsView& graph_params = graph_input.mutable_params();
  // Captured metadata must use bucket-shaped tensors.
  const ModelParamsView& captured = params_.value();
  graph_params.attention.device.q_seq_lens =
      captured.attention.device.q_seq_lens;
  graph_params.attention.device.kv_seq_lens =
      captured.attention.device.kv_seq_lens;
  graph_params.attention.device.q_cu_seq_lens =
      captured.attention.device.q_cu_seq_lens;
  graph_params.attention.device.new_cache_slots =
      captured.attention.device.new_cache_slots;
  graph_params.attention.device.block_tables =
      captured.attention.device.block_tables;
  graph_params.attention.host.q_seq_lens = captured.attention.host.q_seq_lens;
  graph_params.attention.host.kv_seq_lens = captured.attention.host.kv_seq_lens;
  graph_params.attention.host.kpool_query_lens =
      captured.attention.host.kpool_query_lens;
  graph_params.embedding.linear_state_indices =
      captured.embedding.linear_state_indices;
  graph_params.num_accepted_tokens = captured.num_accepted_tokens;
  graph_params.num_accepted_tokens_host = captured.num_accepted_tokens_host;
  graph_params.multi_block_tables = captured.multi_block_tables;
  graph_params.enable_graph = true;
  model->prepare_graph_forward_metadata(
      model_graph_metadata_state_.get(),
      persistent_param_->positions_.slice(persistent_param_->use_mrope_ ? 1 : 0,
                                          0,
                                          plan_.layout.padded_num_tokens),
      graph_params);
  params_->attn_metadata = std::move(graph_params.attn_metadata);
}

ModelOutput MluGraph::capture(CausalLM* model,
                              std::vector<KVCache>& kv_cache,
                              const torch_mlu::MempoolId_t& pool,
                              const torch_mlu::MLUStream& capture_stream,
                              const runtime::Options& options) {
  const int32_t slice_dim = persistent_param_->use_mrope_ ? 1 : 0;
  enable_aux_hidden_states_ = options.enable_graph_aux_hidden_states();
  const auto forward = [&]() {
    return model->forward(
        persistent_param_->tokens_.slice(0, 0, plan_.layout.padded_num_tokens),
        persistent_param_->positions_.slice(
            slice_dim, 0, plan_.layout.padded_num_tokens),
        kv_cache,
        params_.value());
  };
  const auto store_output = [&](const ModelOutput& result) {
    persistent_param_->output_.slice(0, 0, result.hidden_states.size(0))
        .copy_(result.hidden_states, true);
    if (enable_aux_hidden_states_ && result.aux_hidden_states.defined()) {
      if (persistent_param_->aux_hidden_states_.numel() == 0) {
        auto shape = result.aux_hidden_states.sizes().vec();
        shape[0] = persistent_param_->output_.size(0);
        persistent_param_->aux_hidden_states_ =
            torch::zeros(shape, persistent_param_->output_.options());
      }
      auto slice = persistent_param_->aux_hidden_states_.slice(
          0, 0, result.aux_hidden_states.size(0));
      if (slice.sizes() == result.aux_hidden_states.sizes()) {
        slice.copy_(result.aux_hidden_states, true);
      }
    }
  };
  const bool has_state =
      std::any_of(kv_cache.begin(), kv_cache.end(), [](const KVCache& cache) {
        return cache.has_request_state();
      });
  if (has_state) {
    store_output(forward());
  }
  triton_jit::JITKernel::initialize_device();
  torch_mlu::synchronize();
  auto prev_stream = torch_mlu::getCurrentMLUStream();
  torch_mlu::mlu::MLUStreamGuard guard(capture_stream);
  graph_ = torch_mlu::MLUGraph();
  graph_.capture_begin(pool, cnrtQueueCaptureModeRelaxed);
  store_output(forward());
  graph_.capture_end();
  torch_mlu::setCurrentMLUStream(prev_stream);
  torch_mlu::synchronize();
  if (!has_state) {
    graph_.replay();
  }
  return persistent_output();
}

ModelOutput MluGraph::persistent_output() const {
  return make_graph_output(
      persistent_param_->output_.slice(0, 0, plan_.layout.padded_num_tokens),
      persistent_param_->aux_hidden_states_.numel() > 0
          ? persistent_param_->aux_hidden_states_.slice(
                0, 0, plan_.layout.padded_num_tokens)
          : torch::Tensor(),
      enable_aux_hidden_states_);
}

ModelOutput MluGraph::replay() {
  graph_.replay();
  return persistent_output();
}

void MluGraph::update_input_buffer(CausalLM* model,
                                   const torch::Tensor& tokens,
                                   const torch::Tensor& positions,
                                   const ModelParamsView& params,
                                   bool is_init) {
  if (is_init) {
    params_.emplace(params.to(persistent_param_->tokens_.device()).view());

    ModelParamsView& graph_params = params_.value();
    graph_params.enable_graph = true;
    graph_params.meta.num_sequences =
        plan_.layout.padded_num_tokens / plan_.layout.attention_row_query_len;
    if (plan_.history_capacity > 0) {
      graph_params.meta.kv_max_seq_len = plan_.history_capacity;
    }
    graph_params.attn_metadata.reset();
    if (plan_.key.grouped) {
      const auto options = torch::TensorOptions().device(tokens.device());
      owned_buffers_.num_accepted_tokens = torch::empty(
          {plan_.layout.padded_num_reqs}, options.dtype(torch::kInt32));
      owned_buffers_.multi_block_tables.clear();
      for (const torch::Tensor& table : params.multi_block_tables) {
        owned_buffers_.multi_block_tables.emplace_back(torch::empty(
            {table.size(0) + extra_rows_of(plan_.layout), table.size(1)},
            table.options()));
      }
      graph_params.num_accepted_tokens = owned_buffers_.num_accepted_tokens;
      graph_params.multi_block_tables = owned_buffers_.multi_block_tables;
    }

    persistent_param_->bind_buffers(graph_params, plan_);
  }
  persistent_param_->update_input_buffer(
      tokens, positions, params, params_.value(), plan_);
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
  std::string model_name;
  std::string resolve_error;
  if (!resolve_model_registration_name(
          args_.model_type(), &model_name, &resolve_error)) {
    model_name = args_.model_type();
  }
  mtp_capabilities_ = ModelRegistry::get_mtp_capabilities(model_name);
  int64_t history_capacity = 0;
  if (args_.max_position_embeddings() > 0) {
    history_capacity = (get_block_capacity(args_.max_position_embeddings(),
                                           options_.block_size()) +
                        1) *
                       options_.block_size();
  }
  if (mtp_capabilities_.graph_context_policy ==
      GraphContextPolicy::KPOOL_FIXED) {
    history_capacity = ModelRegistry::get_graph_kv_capacity(
        model_name, args_, history_capacity);
  } else if (mtp_capabilities_.graph_context_policy ==
             GraphContextPolicy::QWEN_FULL_ATTENTION) {
    CHECK_GT(args_.max_position_embeddings(), 0)
        << "Qwen3.5 graph capture requires positive max_position_embeddings";
    history_capacity = args_.max_position_embeddings();
  }
  GraphPlannerConfig planner_config;
  planner_config.options = options_;
  planner_config.capabilities = mtp_capabilities_;
  planner_config.state_required =
      model_ != nullptr && model_->requires_graph_forward_metadata();
  planner_config.history_capacity = history_capacity;
  planner_config.max_tokens = std::min<int64_t>(
      std::max<int64_t>(
          ::xllm::ExecutionConfig::get_instance().max_tokens_for_graph_mode(),
          options_.max_seqs_per_batch()),
      get_graph_token_capacity(options_, mtp_capabilities_));
  planner_ = std::make_unique<MluGraphPlanner>(std::move(planner_config));
}

void MluGraphExecutorImpl::set_dp_process_group(ProcessGroup* group) {
  graph_action_workspace_ = {};
  dp_group_ = group;
}

ModelOutput MluGraphExecutorImpl::run_eager(const torch::Tensor& tokens,
                                            const torch::Tensor& positions,
                                            std::vector<KVCache>& kv_caches,
                                            const ModelParamsView& params) {
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
    persistent_param_ = std::make_unique<GraphPersistentParam>(
        args_, device_, options_, mtp_capabilities_);
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
  const std::size_t graph_action_bytes =
      tensor_bytes(graph_action_workspace_.input_device) +
      tensor_bytes(graph_action_workspace_.output_device);
  const std::size_t executor_total_bytes =
      reserved_bytes + persistent_param_bytes + graph_action_bytes;

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
            << " graphs_cached=" << graphs_.size() << " executor_total_memory="
            << format_memory_size(executor_total_bytes) << " persistent_param="
            << format_memory_size(persistent_param_bytes)
            << " graph_action_memory=" << format_memory_size(graph_action_bytes)
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
                                      const ModelParamsView& params) {
  const int64_t actual_tokens = tokens.size(0);
  GraphDecision decision = planner_->plan(actual_tokens, params);
  const GraphPlan* plan = std::get_if<GraphPlan>(&decision);
  MluGraph* existing = nullptr;
  if (plan != nullptr) {
    const auto it = graphs_.find(plan->key);
    existing = it != graphs_.end() ? it->second.get() : nullptr;
  }
  GraphAction action = GraphAction::EAGER;
  const auto& decode_flags = params.parallel.dp_is_decode;
  const auto& token_nums = params.parallel.dp_global_token_nums;
  const bool skip_dp_agreement =
      dp_group_ != nullptr && dp_group_->world_size() > 1 &&
      (decode_flags.empty() ||
       std::find(decode_flags.begin(), decode_flags.end(), 0) !=
           decode_flags.end() ||
       std::find(token_nums.begin(), token_nums.end(), 0) != token_nums.end());
  if (!skip_dp_agreement) {
    action = agree_graph_action(
        plan, existing != nullptr, dp_group_, device_, graph_action_workspace_);
  }
  if (action == GraphAction::EAGER) {
    if (plan == nullptr) {
      log_eager_reason(std::get<EagerReason>(decision));
    }
    return run_eager(tokens, positions, kv_caches, params);
  }
  if (action == GraphAction::CAPTURE &&
      params.embedding.input_embedding.defined()) {
    CHECK(params.embedding.input_embedding.device() == device_)
        << "input_embedding graph input device changed";
  }
  init_param_once();
  ModelOutput result;
  if (action == GraphAction::REPLAY && existing != nullptr) {
    existing->update_input_buffer(model_, tokens, positions, params);
    result = existing->replay();
  } else {
    auto graph_input =
        prepare_graph_input(params, plan->layout.padded_num_tokens);
    if (mtp_capabilities_.model_family == MtpModelFamily::GLM5 ||
        mtp_capabilities_.model_family == MtpModelFamily::QWEN35) {
      if (!params.is_spec_verify &&
          (params.num_accepted_tokens.defined() ||
           !params.num_accepted_tokens_host.empty())) {
        auto& graph_params = graph_input.mutable_params();
        graph_params.num_accepted_tokens = torch::Tensor();
        graph_params.num_accepted_tokens_host.clear();
      }
    }

    auto graph = std::make_unique<MluGraph>(persistent_param_.get(), *plan);
    graph->update_input_buffer(
        model_, tokens, positions, graph_input.params(), true);
    if (!graph_capture_stream_.has_value()) {
      graph_capture_stream_ = torch_mlu::getStreamFromPool(
          /*isHighPriority=*/false, device_.index());
    }
    result = graph->capture(
        model_, kv_caches, graph_pool_, *graph_capture_stream_, options_);
    // Old replay work may still use these resources on the caller stream.
    if (existing != nullptr) {
      torch_mlu::synchronize();
    }
    graphs_[plan->key] = std::move(graph);

    LOG(INFO) << "MLU graph captured: padded_num_reqs="
              << plan->layout.padded_num_reqs
              << ", tokens_per_request=" << plan->layout.tokens_per_request
              << ", grouped=" << plan->key.grouped
              << ", draft=" << options_.is_draft_model()
              << ", verify=" << params.is_spec_verify
              << ", padded_num_tokens=" << plan->layout.padded_num_tokens;
    log_memory_after_capture();
  }
  return make_graph_output(
      result.hidden_states.slice(0, 0, actual_tokens),
      result.aux_hidden_states.defined()
          ? result.aux_hidden_states.slice(0, 0, actual_tokens)
          : torch::Tensor(),
      options_.enable_graph_aux_hidden_states());
}

}  // namespace xllm::mlu

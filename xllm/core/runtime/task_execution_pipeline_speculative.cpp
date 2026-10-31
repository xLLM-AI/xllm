/* Copyright 2026 The xLLM Authors.

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

#include "core/runtime/task_execution_pipeline_speculative.h"

#include <c10/core/DeviceGuard.h>

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <limits>
#include <nlohmann/json.hpp>

#include "core/common/metrics.h"
#include "core/framework/parallel_state/process_group.h"
#include "core/framework/sampling/sampler.h"
#include "core/framework/sampling/sampling_params.h"
#include "core/framework/speculative/draft_proposal.h"
#include "core/framework/speculative/spec_verify.h"
#include "core/framework/speculative/verify_layout.h"
#include "core/kv_cache/linear_state_restore.h"
#include "core/kv_cache/transfer/kv_transfer_completion.h"
#include "core/layers/common/attention_metadata.h"
#include "core/runtime/dflash2_worker_impl.h"
#include "core/runtime/dspark_worker_impl.h"
#if defined(USE_NPU)
#include "core/kernels/npu/tilelang/tilelang_ops_api.h"
#endif
#include "core/platform/device.h"
#include "core/platform/platform.h"
#include "core/runtime/decode_graph_bucket.h"
#include "core/runtime/py_executor_impl.h"
#include "core/runtime/task_execution_pipeline.h"
#include "core/util/tensor_helper.h"

namespace xllm {
bool TaskExecutionPipeline::block_draft() const {
  return speculative_capacity_->kind != SpeculativeTaskKind::MTP;
}

bool TaskExecutionPipeline::block_draft_graphs() const {
  return speculative_capacity_->kind == SpeculativeTaskKind::DSPARK ||
         speculative_capacity_->kind == SpeculativeTaskKind::DFLASH2;
}

uint32_t TaskExecutionPipeline::draft_calls(const SpeculativeSlot& slot) const {
  return block_draft() ? 1U : slot.draft_steps;
}

uint32_t TaskExecutionPipeline::context_hidden_size() const {
  return block_draft() ? speculative_capacity_->context_hidden_size
                       : speculative_capacity_->common.hidden_size;
}

namespace {

int64_t paged_cache_blocks(const std::vector<KVCache>& caches) {
  for (const auto& cache : caches) {
    if (cache.empty() || cache.has_request_state()) {
      continue;
    }
    const auto key = cache.get_k_cache();
    if (key.defined() && key.numel() != 0) {
      return key.size(0);
    }
  }
  return 0;
}

ModelInputBatch model_batch(const LlmForwardInput& input) {
  const auto& meta = input.input_params.meta;
  const uint32_t rows = input.input_params.attention.host.q_seq_lens.size();
  return {rows == 0 ? BatchForwardType::EMPTY : meta.batch_forward_type,
          rows,
          meta.batch_id,
          false};
}

ParallelInput expanded_parallel(const ParallelInput& base,
                                uint32_t scale,
                                bool pad_empty) {
  ParallelInput result = base;
  for (int32_t& count : result.dp_global_token_nums) {
    count = count == 0 && pad_empty ? scale : count * scale;
  }
  for (int32_t& count : result.raw_dp_global_token_nums) {
    count *= scale;
  }
  return result;
}

uint32_t task_graph_batch_size(const ParallelInput& input,
                               std::span<const int64_t> buckets) {
  int32_t maximum = 0;
  for (uint32_t rank = 0; rank < input.dp_global_token_nums.size(); ++rank) {
    const int32_t count = input.dp_global_token_nums[rank];
    if (count != 0 && input.dp_is_decode[rank] != 1) {
      return 0;
    }
    maximum = std::max(maximum, count);
  }
  const auto bucket = std::lower_bound(buckets.begin(), buckets.end(), maximum);
  return maximum == 0 || bucket == buckets.end() ? 0 : *bucket;
}

void pad_task_parallel(ParallelInput& input, uint32_t rows) {
  if (rows != 0) {
    std::fill(input.dp_global_token_nums.begin(),
              input.dp_global_token_nums.end(),
              rows);
    std::fill(input.dp_is_decode.begin(), input.dp_is_decode.end(), 1);
  }
}

Status invalid(const char* message) {
  return Status(StatusCode::INVALID_ARGUMENT, message);
}

StreamEventPtr reusable_event(const torch::Device& device) {
  return std::make_shared<StreamEvent>(device.type());
}

void record(const Stream& stream, const StreamEventPtr& event) {
  stream.record_event(*event);
}

torch::Tensor prefix(const torch::Tensor& storage, int64_t count) {
  return storage.narrow(/*dim=*/0, /*start=*/0, count);
}

std::vector<int64_t> decode_graph_batch_sizes(int64_t maximum) {
  std::vector<int64_t> sizes;
  sizes.reserve(maximum / 16 + 5);
  for (int64_t size = 1; size <= maximum;) {
    const int64_t bucket = std::min(runtime::get_decode_graph_token_bucket(
                                        size, /*enable_no_padding=*/false),
                                    maximum);
    sizes.emplace_back(bucket);
    size = bucket + 1;
  }
  return sizes;
}

void copy_input(const torch::Tensor& device,
                const torch::Tensor& host,
                const Stream& stream) {
  if (host.numel() == 0) {
    return;
  }
  CHECK_EQ(device.nbytes(), host.nbytes());
  auto stream_guard = stream.set_stream_guard();
  device.copy_(host, /*non_blocking=*/true);
}

SampleOutput sample_views(const TokenResultTensors& result) {
  SampleOutput output;
  if (!result.tokens.defined()) {
    return output;
  }
  output.next_tokens = result.tokens.view({-1});
  if (result.logprobs.defined()) {
    output.logprobs = result.logprobs.view({-1});
  }
  if (result.top_tokens.defined()) {
    const int64_t top = result.top_tokens.size(/*dim=*/2);
    output.top_tokens = result.top_tokens.view({-1, top});
    output.top_logprobs = result.top_logprobs.view({-1, top});
  }
  return output;
}

uint64_t state_bytes(const MtpContextTensors& state) {
  uint64_t bytes = 0;
  for (const auto* tensor : {&state.previous_tokens,
                             &state.hidden,
                             &state.positions,
                             &state.kv_seq_lens,
                             &state.accepted_lengths,
                             &state.repair_required}) {
    if (tensor->defined()) {
      bytes += tensor->nbytes();
    }
  }
  return bytes;
}

bool neutral_host_values(const torch::Tensor& tensor, float neutral) {
  if (!tensor.defined() || tensor.numel() == 0) {
    return true;
  }
  if (!tensor.device().is_cpu()) {
    return false;
  }
  const torch::Tensor values = tensor.to(torch::kFloat32).contiguous();
  const float* begin = values.const_data_ptr<float>();
  return std::all_of(begin, begin + values.numel(), [neutral](float value) {
    return value == neutral;
  });
}

bool adaptive_target_sampling(const SamplingParameters& sampling) {
  if (sampling.logprobs() || sampling.max_top_logprobs() != 0 ||
      sampling.use_beam_search() || sampling.is_embeddings() ||
      sampling.num_return_sequences() != 0 ||
      sampling.acc_logprob().defined() ||
      (sampling.filter_mask().defined() &&
       sampling.filter_mask().numel() != 0) ||
      (sampling.filter_bitmask().defined() &&
       sampling.filter_bitmask().numel() != 0) ||
      !neutral_host_values(sampling.frequency_penalties(),
                           /*neutral=*/0.0F) ||
      !neutral_host_values(sampling.presence_penalties(),
                           /*neutral=*/0.0F) ||
      !neutral_host_values(sampling.repetition_penalties(),
                           /*neutral=*/1.0F)) {
    return false;
  }
  if (sampling.sampling_mode() == SamplingMode::GREEDY ||
      sampling.sampling_mode() == SamplingMode::RANDOM) {
    return !sampling.do_sample().defined();
  }
  if (sampling.sampling_mode() != SamplingMode::MIXED ||
      !sampling.do_sample().defined() || sampling.do_sample().numel() == 0 ||
      !sampling.do_sample().device().is_cpu() ||
      sampling.do_sample().scalar_type() != torch::kBool) {
    return false;
  }
  const torch::Tensor values = sampling.do_sample().contiguous();
  const bool* begin = values.const_data_ptr<bool>();
  const bool any_random = std::any_of(
      begin, begin + values.numel(), [](bool value) { return value; });
  const bool all_random = std::all_of(
      begin, begin + values.numel(), [](bool value) { return value; });
  return any_random && !all_random;
}

uint64_t dspark_graph_kv_capacity(const LlmTaskCapacity& capacity,
                                  uint32_t query_width,
                                  int32_t maximum_kv_length) {
  // Match PreparedAclGraphRunner.warmup_prepared's compact block-draft
  // table: at least 16 context pages plus the query, then round to a power
  // of two and clamp to the fixed Slot table width. This records the startup
  // capacity for calibration; serialized Launch may capture larger variants.
  const uint64_t page_size = capacity.logical_block_size;
  const uint64_t required_pages =
      (static_cast<uint64_t>(maximum_kv_length) + page_size - 1) / page_size;
  const uint64_t minimum_pages =
      16 + (static_cast<uint64_t>(query_width) + page_size - 1) / page_size;
  const uint64_t pages =
      std::min(std::bit_ceil(std::max(required_pages, minimum_pages)),
               static_cast<uint64_t>(capacity.model.max_blocks_per_sequence));
  return pages * page_size;
}

int32_t dspark_local_kv_bound(std::span<const int32_t> upper_kv_lengths,
                              uint32_t draft_width) {
  CHECK(!upper_kv_lengths.empty());
  return *std::max_element(upper_kv_lengths.begin(), upper_kv_lengths.end()) +
         static_cast<int32_t>(draft_width);
}

uint64_t adaptive_unified_graph_key(uint32_t batch_size, uint32_t graph_rows) {
  return (static_cast<uint64_t>(batch_size) << 32) | graph_rows;
}

std::vector<int64_t> adaptive_unified_graph_rows(
    uint32_t batch_size,
    std::span<const int64_t> captured_graph_rows,
    uint32_t draft_width) {
  std::vector<int64_t> result;
  result.reserve(4);
  // Whole-round graphs duplicate model nodes, unlike prepared Target graphs.
  // Four physical tiers bound startup variants without restricting raw B.
  for (const uint32_t width : {1U, 2U, 4U, draft_width + 1}) {
    const int64_t required = static_cast<int64_t>(batch_size) * width;
    const auto bucket = std::lower_bound(
        captured_graph_rows.begin(), captured_graph_rows.end(), required);
    if (bucket == captured_graph_rows.end()) {
      return {};
    }
    if (result.empty() || result.back() != *bucket) {
      result.emplace_back(*bucket);
    }
  }
  return result;
}

}  // namespace

bool supports_unified_block_graph(const SpeculativeTaskCapacity& capacity,
                                  uint32_t rows,
                                  const LlmForwardInput& input) {
  const auto& sampling = input.sampling_params;
  return capacity.unified_graph && !capacity.adaptive_target_verify &&
         capacity.kind != SpeculativeTaskKind::MTP && rows > 0 &&
         static_cast<uint64_t>(rows) *
                 (static_cast<uint64_t>(capacity.num_speculative_tokens) + 1) <=
             capacity.common.max_graph_batch_size &&
         input.input_params.meta.batch_forward_type.is_decode() &&
         !input.input_params.meta.is_graph_warmup &&
         input.json_object_states.empty() &&
         input.input_params.multi_block_tables.empty() &&
         sampling.sampling_mode() == SamplingMode::GREEDY &&
         !sampling.use_beam_search() && !sampling.logprobs() &&
         !sampling.has_sampling_fields();
}

Status TaskExecutionPipeline::create(
    ThreadPool& state_executor,
    TaskModel target,
    TaskModel draft,
    const SpeculativeTaskCapacity& capacity,
    std::unique_ptr<TaskExecutionPipeline>& output) {
  auto prepared_capacity = capacity;
  if (prepared_capacity.hybrid_recurrent_target) {
    // Fixed blocks share the ordinary recurrent Slot layout. Only adaptive
    // verification binds request state through a packed graph descriptor.
    prepared_capacity.common.model.enable_linear_attention =
        !capacity.adaptive_target_verify;
  }
  const auto& common = prepared_capacity.common;
  const uint64_t width =
      static_cast<uint64_t>(capacity.num_speculative_tokens) + 1;
  const uint64_t rows =
      static_cast<uint64_t>(common.model.max_sequences) * width;
  if (capacity.kind == SpeculativeTaskKind::DFLASH2 &&
      common.max_graph_batch_size != 0 &&
      capacity.num_speculative_tokens > 16) {
    return invalid(
        "DFlash2 graph execution supports 1..16 speculative tokens.");
  }
  if (common.model.enable_linear_attention &&
      !capacity.hybrid_recurrent_target &&
      (capacity.kind != SpeculativeTaskKind::MTP ||
       capacity.num_speculative_tokens == 0 ||
       capacity.num_speculative_tokens > 16 || common.dp_size != 1 ||
       capacity.adaptive_target_verify || capacity.adaptive_draft_step)) {
    return invalid(
        "Hybrid MTP requires fixed widths, DP=1 and 1..16 speculative tokens.");
  }
  if (capacity.kind == SpeculativeTaskKind::DSPARK && common.dp_size != 1 &&
      capacity.unified_graph) {
    return invalid("DP DSpark task pipeline requires split graphs.");
  }
  if (capacity.startup_calibration &&
      (!capacity.adaptive_target_verify || capacity.adaptive_costs)) {
    return invalid(
        "Internal DSpark startup calibration requires adaptive graphs "
        "without an injected startup cost cache.");
  }
  if (capacity.adaptive_target_verify &&
      (!Platform::is_npu() ||
       (capacity.kind != SpeculativeTaskKind::DSPARK &&
        capacity.kind != SpeculativeTaskKind::MTP) ||
       (capacity.num_speculative_tokens != 5 &&
        capacity.num_speculative_tokens != 7) ||
       (!(capacity.sparse_dsa_target && common.enable_mla) &&
        !capacity.hybrid_recurrent_target) ||
       common.max_graph_batch_size == 0 || common.model.max_sequences > 128 ||
       capacity.is_probabilistic_draft ||
       !std::isfinite(capacity.adaptive_min_gain) ||
       capacity.adaptive_min_gain < 0.0 ||
       (capacity.kind == SpeculativeTaskKind::DSPARK &&
        !draft.model.has_dspark_confidence_head()) ||
       (capacity.kind == SpeculativeTaskKind::MTP &&
        (common.dp_size != 1 || common.cp_sampling_group != nullptr ||
         capacity.unified_graph || !capacity.refresh_target_prefix)))) {
    return invalid(
        "Adaptive task pipeline requires NPU MTP/DSpark Draft5/7, sparse DSA "
        "or recurrent Target graphs, greedy draft sampling and supported "
        "request/page capacity.");
  }
  if (capacity.adaptive_draft_step &&
      (!Platform::is_npu() || capacity.kind != SpeculativeTaskKind::MTP ||
       common.dp_size != 1 || common.cp_sampling_group != nullptr ||
       capacity.adaptive_target_verify || capacity.unified_graph ||
       capacity.num_speculative_tokens > 64)) {
    return invalid("Adaptive MTP requires ordinary NPU MTP tasks and DP=CP=1.");
  }
  if (capacity.hybrid_recurrent_target &&
      ((capacity.kind != SpeculativeTaskKind::DSPARK &&
        capacity.kind != SpeculativeTaskKind::DFLASH2 &&
        !(capacity.kind == SpeculativeTaskKind::MTP &&
          capacity.adaptive_target_verify)) ||
       !Platform::is_npu() || common.max_graph_batch_size == 0 ||
       common.model.max_sequences > 128 ||
       capacity.num_speculative_tokens == 0 ||
       capacity.num_speculative_tokens > 16 || capacity.unified_graph ||
       common.dp_size != 1 || common.cp_sampling_group != nullptr ||
       !target.restore_request_state)) {
    return invalid(
        "Recurrent block drafts require prepared Draft1..16 graphs, request "
        "state "
        "restoration and DP=CP=1 without unified graphs.");
  }
  if ((capacity.adaptive_target_verify || capacity.hybrid_recurrent_target) &&
      (rows * common.model.max_blocks_per_sequence >
           std::numeric_limits<int32_t>::max() ||
       static_cast<uint64_t>(common.logical_block_size) *
               common.model.max_blocks_per_sequence >
           std::numeric_limits<int32_t>::max())) {
    return invalid("Adaptive DSpark block tables exceed INT32 indexing.");
  }
  if (capacity.adaptive_target_verify && capacity.unified_graph &&
      (common.cp_sampling_group != nullptr ||
       !target.executor.supports_adaptive_unified_graph() ||
       !draft.executor.supports_adaptive_unified_graph())) {
    return invalid(
        "Adaptive DSpark unified graphs require supported Target/Draft "
        "executors and CP=1.");
  }
  if (common.enable_kv_push && (capacity.kind != SpeculativeTaskKind::MTP ||
                                !target.push_kv || !draft.push_kv)) {
    return invalid(
        "PD MTP Prefill requires Target and Draft KV PUSH providers.");
  }
  if (state_executor.size() != 1 || common.slot_count == 0 ||
      common.slot_count > 2 || common.model.max_sequences == 0 ||
      common.model.max_tokens == 0 || common.max_positions == 0 ||
      common.max_kv_seq_len == 0 ||
      common.max_kv_seq_len > common.max_positions ||
      common.max_positions > std::numeric_limits<int32_t>::max() ||
      common.hidden_size == 0 || common.logical_block_size == 0 ||
      common.logical_block_size > std::numeric_limits<int32_t>::max() ||
      common.vocab_size == 0 ||
      (capacity.refresh_target_prefix &&
       capacity.kind != SpeculativeTaskKind::MTP) ||
      (capacity.reuse_prefix_refresh_output &&
       (!capacity.refresh_target_prefix || capacity.reuse_topk ||
        capacity.num_speculative_tokens <= 1 || common.dp_size != 1 ||
        common.cp_sampling_group != nullptr ||
        capacity.adaptive_target_verify || capacity.adaptive_draft_step)) ||
      (capacity.kind == SpeculativeTaskKind::MTP && !common.enable_mla &&
       !common.model.enable_linear_attention &&
       !capacity.hybrid_recurrent_target) ||
      (capacity.kind != SpeculativeTaskKind::MTP &&
       (capacity.context_hidden_size == 0 || capacity.mask_token_id < 0 ||
        static_cast<uint32_t>(capacity.mask_token_id) >= common.vocab_size)) ||
      (capacity.kind == SpeculativeTaskKind::DFLASH &&
       capacity.is_probabilistic_draft) ||
      (capacity.kind == SpeculativeTaskKind::DFLASH2 &&
       capacity.selector_top_k == 0) ||
      capacity.num_speculative_tokens == 0 || common.dp_size == 0 ||
      common.dp_rank >= common.dp_size ||
      rows > std::numeric_limits<int32_t>::max() ||
      (capacity.reuse_topk && capacity.index_topk == 0) ||
      target.model.device().type() != Platform::type_torch() ||
      !target.model.device().has_index() ||
      target.model.device() != draft.model.device() ||
      target.model.options().dtype() != draft.model.options().dtype() ||
      !target.executor.supports_prepared_attention_metadata() ||
      !draft.executor.supports_prepared_attention_metadata()) {
    return invalid("Invalid MTP capacity or prepared MLA model resources.");
  }
  c10::DeviceGuard guard(target.model.device());
  auto program = std::unique_ptr<TaskExecutionPipeline>(
      new TaskExecutionPipeline(state_executor,
                                target.model,
                                target.executor,
                                target.kv_caches,
                                common,
                                std::move(target.push_kv),
                                std::move(target.load_kv),
                                std::move(target.fence_kv)));
  program->speculative_capacity_ =
      std::make_unique<SpeculativeTaskCapacity>(std::move(prepared_capacity));
  program->draft_ = std::make_unique<TaskModel>(std::move(draft));
  program->restore_request_state_ = std::move(target.restore_request_state);
  program->hidden_dtype_ = target.model.options().dtype().toScalarType();
  Status status = program->initialize_speculative();
  if (!status.ok()) {
    return status;
  }
  CHECK_EQ(Device(target.model.device()).current_stream()->synchronize(), 0);
  program->start();
  output = std::move(program);
  return Status();
}

Status TaskExecutionPipeline::initialize_speculative() {
  const auto& c = capacity_;
  const uint32_t count = c.model.max_sequences;
  const uint32_t width = speculative_capacity_->num_speculative_tokens + 1;
  const uint32_t expanded = count * width;
  Status status = create_mtp_context(0,
                                     block_draft() ? 0 : c.hidden_size,
                                     hidden_dtype_,
                                     device_.unwrap(),
                                     context_);
  if (!status.ok()) {
    return status;
  }
  reserve_draft_prefix_context(*context_, 0);
  const uint32_t draft_samples = block_draft() ? count * (width - 1) : count;
  const uint32_t max_batch_size = std::min(c.max_graph_batch_size, expanded);
  graph_batch_sizes_ = decode_graph_batch_sizes(c.model.enable_linear_attention
                                                    ? max_batch_size / width
                                                    : max_batch_size);
  if (c.model.enable_linear_attention) {
    // A captured recurrent invocation owns complete S-token sequence chunks.
    for (int64_t& size : graph_batch_sizes_) {
      size *= width;
    }
  }
  captured_graph_batch_sizes_.reserve(graph_batch_sizes_.size());
  const auto options = model_.options();
  const auto pinned = options.device(torch::kCPU).pinned_memory(true);
  if (speculative_capacity_->adaptive_target_verify) {
    adaptive_verify_ = std::make_unique<AdaptiveVerifyState>();
    if (!speculative_capacity_->adaptive_costs) {
      adaptive_verify_->owned_costs = std::make_shared<SpecCostProfileStore>();
      speculative_capacity_->adaptive_costs = adaptive_verify_->owned_costs;
    }
  }
  SlotBufferCapacity base_capacity{c.model,
                                   c.max_unique_tokens,
                                   c.vocab_size,
                                   c.max_top_logprobs,
                                   c.parameter_dtype,
                                   c.enable_mla};
  base_capacity.max_result_width = width;
  SlotBufferCapacity target_capacity = base_capacity;
  target_capacity.max_sample_rows = expanded;
  target_capacity.max_result_width = 0;
  slots_.reserve(c.slot_count);
  for (uint32_t slot_id = 0; slot_id < c.slot_count; ++slot_id) {
    auto task_slot = std::make_unique<Slot>();
    auto slot = std::make_unique<SpeculativeSlot>();
    slot->draft_steps = width - 1;
    if (!block_draft()) {
      slot->dummy_hidden = torch::zeros({1, c.hidden_size}, options);
    }
    status = SlotBuffer::create(
        base_capacity, device_.unwrap(), slot->target_prefill);
    if (!status.ok()) {
      return status;
    }
    ModelInputCapacity draft_model = c.model;
    draft_model.enable_linear_attention = false;
    MtpInputSpec prefill_spec{draft_model, context_hidden_size()};
    prefill_spec.enable_mla = c.enable_mla;
    prefill_spec.context_only = block_draft();
    status = SlotBuffer::create_mtp_input(
        prefill_spec, hidden_dtype_, device_.unwrap(), slot->draft_prefill);
    if (!status.ok()) {
      return status;
    }
    const MtpInputSpec validate_spec{c.model,
                                     0,
                                     c.logical_block_size,
                                     width - 1,
                                     MtpInvocationKind::VALIDATE,
                                     0,
                                     c.enable_mla};
    status = SlotBuffer::create_mtp_input(
        validate_spec, hidden_dtype_, device_.unwrap(), slot->validate_input);
    if (!status.ok()) {
      return status;
    }
    status = slot->validate_input->configure_sampling(target_capacity);
    if (!status.ok()) {
      return status;
    }
    status = context_create(*context_, count, slot->state);
    if (!status.ok()) {
      return status;
    }
    const auto indices = options.dtype(torch::kInt32);
    if (speculative_capacity_->hybrid_recurrent_target) {
      slot->recurrent = std::make_unique<SpeculativeSlot::Recurrent>();
      slot->recurrent->host_state_indices =
          torch::zeros({count}, pinned.dtype(torch::kInt32));
      slot->recurrent->state_indices = torch::zeros({count}, indices);
      slot->recurrent->accepted_lengths = torch::ones({count}, indices);
    }
    if (block_draft()) {
      slot->draft_block_indices =
          speculative_capacity_->kind == SpeculativeTaskKind::DSPARK
              ? torch::arange(draft_samples, indices)
              : torch::arange(expanded, indices)
                    .view({count, width})
                    .narrow(1, 1, width - 1)
                    .reshape({-1});
    } else {
      slot->draft_first_indices = torch::arange(count, indices).mul(2).add(1);
      if (speculative_capacity_->reuse_prefix_refresh_output) {
        slot->draft_prefix_indices = torch::arange(count, indices);
      }
    }
    if (adaptive_verify_) {
      slot->target_token_storage =
          torch::empty({expanded}, options.dtype(torch::kInt64));
      slot->verify = std::make_unique<SpeculativeSlot::PackedVerify>();
      auto& adaptive = *slot->verify;
      adaptive.arena_storage =
          torch::empty({static_cast<int64_t>(count) * (12 * (width - 1) + 4)},
                       options.dtype(torch::kUInt8));
      adaptive.host_feedback_storage =
          torch::empty({static_cast<int64_t>(count) * (4 * width)},
                       pinned.dtype(torch::kUInt8));
      adaptive.previous_storage =
          torch::empty({count, width - 1}, options.dtype(torch::kInt64));
      adaptive.offsets_storage =
          torch::empty({count + 1}, options.dtype(torch::kInt32));
      adaptive.owners_storage =
          torch::empty({expanded}, options.dtype(torch::kInt32));
      adaptive.columns_storage =
          torch::empty({expanded}, options.dtype(torch::kInt32));
      adaptive.embedding_ids.reserve(count);
      adaptive.request_ids.reserve(count);
      adaptive.lagged_confidence.reserve(count * (width - 1));
      adaptive.draft_counts.reserve(count);
      adaptive.unified_graph_bindings.reserve(static_cast<size_t>(count) * 4);
      if (!block_draft()) {
        adaptive.hidden_columns =
            torch::arange(width, options.dtype(torch::kInt64)).view({1, width});
      }
    } else {
      slot->proposal_storage =
          torch::empty({count * (width - 1)}, options.dtype(torch::kInt64));
    }
    slot->accepted_mask_storage =
        torch::empty({std::max<int64_t>(expanded, adaptive_verify_ ? 8 : 0)},
                     options.dtype(torch::kBool));
    if (adaptive_verify_) {
      // Packed input preparation reads control before fixed rejection writes
      // the mask on the same task stream. Adaptive commit never uses the mask.
      // Reuse its first eight bytes instead of another Device allocation.
      slot->verify->control =
          prefix(slot->accepted_mask_storage, /*count=*/8).view(torch::kInt64);
      // Consume copies feedback into owned history before releasing this Slot.
      // Prepare's control H2D precedes Launch and the next feedback D2H, so
      // this pinned prefix can be reused without overlapping either DMA reader.
      slot->verify->host_control =
          prefix(slot->verify->host_feedback_storage, /*count=*/8)
              .view(torch::kInt64);
    }
    slot->host_hidden_storage =
        torch::empty({count, context_hidden_size()}, pinned);
    if (block_draft() || slot->verify) {
      slot->context_slots_storage =
          torch::empty({expanded}, options.dtype(torch::kInt32));
    }
    slot->upper_positions.reserve(count);
    slot->upper_kv_lengths.reserve(count);
    auto& bootstrap = slot->bootstrap;
    bootstrap.host_rows = torch::empty({count}, pinned.dtype(torch::kInt64));
    bootstrap.row_storage = torch::empty({count}, options.dtype(torch::kInt64));
    bootstrap.host_tokens = torch::empty({count}, pinned.dtype(torch::kInt64));
    bootstrap.token_storage =
        torch::empty({count}, options.dtype(torch::kInt64));
    for (auto* state : {&bootstrap.host_state, &bootstrap.storage}) {
      const auto state_options =
          state == &bootstrap.host_state ? pinned : options;
      state->positions =
          torch::empty({count}, state_options.dtype(torch::kInt32));
      state->kv_seq_lens =
          torch::empty({count}, state_options.dtype(torch::kInt32));
      if (!block_draft()) {
        state->hidden = torch::empty({count, 2, c.hidden_size}, state_options);
        state->repair_required =
            torch::zeros({count}, state_options.dtype(torch::kBool));
      }
    }
    const uint32_t draft_calls = block_draft() ? 1 : width - 1;
    slot->drafts.reserve(draft_calls);
    for (uint32_t step = 0; step < draft_calls; ++step) {
      auto invocation = std::make_unique<SpeculativeSlot::Draft>();
      auto spec = validate_spec;
      spec.model.enable_linear_attention = false;
      spec.kind = block_draft() ? MtpInvocationKind::BLOCK_DRAFT
                                : MtpInvocationKind::DRAFT;
      spec.draft_step = step;
      spec.enable_mla = !block_draft() && c.enable_mla;
      spec.mask_token_id =
          block_draft() ? speculative_capacity_->mask_token_id : 0;
      spec.sample_from_anchor =
          speculative_capacity_->kind == SpeculativeTaskKind::DSPARK;
      status = SlotBuffer::create_mtp_input(
          spec, hidden_dtype_, device_.unwrap(), invocation->input);
      if (!status.ok()) {
        return status;
      }
      invocation->token_storage =
          torch::empty({draft_samples}, options.dtype(torch::kInt64));
      const uint32_t rows_per_sequence =
          invocation->input->input_scratch_->rows_per_sequence_;
      if (!block_draft()) {
        invocation->graph_batch_sizes = decode_graph_batch_sizes(
            max_batch_size / width * rows_per_sequence);
        invocation->captured_graph_batch_sizes.reserve(
            invocation->graph_batch_sizes.size());
      } else if (block_draft_graphs() ||
                 speculative_capacity_->hybrid_recurrent_target) {
        invocation->graph_batch_sizes =
            decode_graph_batch_sizes(max_batch_size / width);
        for (int64_t& request_bucket : invocation->graph_batch_sizes) {
          request_bucket *= rows_per_sequence;
        }
        invocation->captured_graph_batch_sizes.reserve(
            invocation->graph_batch_sizes.size());
      }
      if (step != 0 || (!block_draft() && max_batch_size != 0)) {
        invocation->hidden_storage =
            torch::empty({count * rows_per_sequence, c.hidden_size}, options);
        if (step != 0 && speculative_capacity_->reuse_topk) {
          invocation->topk_storage =
              torch::empty({count, 1, speculative_capacity_->index_topk},
                           options.dtype(torch::kInt32));
        }
      }
      slot->drafts.emplace_back(std::move(invocation));
    }
    if (speculative_capacity_->refresh_target_prefix) {
      auto refresh = std::make_unique<SpeculativeSlot::Draft>();
      auto refresh_spec = validate_spec;
      refresh_spec.model.enable_linear_attention = false;
      status = SlotBuffer::create_mtp_input(
          refresh_spec, hidden_dtype_, device_.unwrap(), refresh->input);
      if (!status.ok()) {
        return status;
      }
      refresh->hidden_storage =
          torch::empty({expanded, c.hidden_size}, options);
      refresh->graph_batch_sizes = graph_batch_sizes_;
      refresh->captured_graph_batch_sizes.reserve(graph_batch_sizes_.size());
      slot->prefix_refresh = std::move(refresh);
    }
    task_slot->input_ready = reusable_event(device_.unwrap());
    task_slot->output_ready = reusable_event(device_.unwrap());
    task_slot->speculative = std::move(slot);
    slots_.emplace_back(std::move(task_slot));
  }
  if (prepares_mtp_context()) {
    // At most one producer and predecessor per occupied Slot, plus the last
    // admitted generation, can be retained. Keep one spare for admission.
    const uint32_t publications = 2 * capacity_.slot_count + 1;
    context_publications_.reserve(publications);
    for (uint32_t index = 0; index < publications; ++index) {
      auto publication = std::make_shared<MtpContextPublication>();
      publication->ready = reusable_event(device_.unwrap());
      context_publications_.emplace_back(std::move(publication));
    }
  }
  return Status();
}

Status TaskExecutionPipeline::plan_parallel(const ParallelInput& input,
                                            uint32_t dp_size,
                                            uint32_t dp_rank,
                                            uint32_t local_tokens,
                                            bool local_decode,
                                            ParallelInput& output,
                                            bool& run_models,
                                            bool& decode) {
  if (dp_size == 0 || dp_rank >= dp_size ||
      local_tokens > std::numeric_limits<int32_t>::max()) {
    return Status(StatusCode::INVALID_ARGUMENT, "Invalid Task DP topology.");
  }
  ParallelInput plan;
  if (dp_size == 1 && input.dp_global_token_nums.empty()) {
    if (!input.raw_dp_global_token_nums.empty() ||
        !input.dp_is_decode.empty() ||
        !input.dp_global_kv_max_seq_lens.empty() ||
        !input.dp_global_batch_generations.empty() ||
        !input.dp_global_sequence_nums.empty()) {
      return Status(StatusCode::INVALID_ARGUMENT,
                    "Task DP summaries require token counts.");
    }
    plan.dp_global_token_nums = {static_cast<int32_t>(local_tokens)};
    plan.raw_dp_global_token_nums = plan.dp_global_token_nums;
    plan.dp_is_decode = {local_decode ? 1 : 0};
  } else {
    if (input.dp_global_token_nums.size() != dp_size ||
        input.raw_dp_global_token_nums.size() != dp_size ||
        input.dp_is_decode.size() != dp_size ||
        (!input.dp_global_kv_max_seq_lens.empty() &&
         input.dp_global_kv_max_seq_lens.size() != dp_size) ||
        (!input.dp_global_batch_generations.empty() &&
         input.dp_global_batch_generations.size() != dp_size) ||
        (!input.dp_global_sequence_nums.empty() &&
         input.dp_global_sequence_nums.size() != dp_size) ||
        std::any_of(input.dp_global_sequence_nums.begin(),
                    input.dp_global_sequence_nums.end(),
                    [](int32_t count) { return count < 0; }) ||
        input.dp_global_token_nums[dp_rank] !=
            static_cast<int32_t>(local_tokens) ||
        !std::equal(input.dp_global_token_nums.begin(),
                    input.dp_global_token_nums.end(),
                    input.raw_dp_global_token_nums.begin()) ||
        std::any_of(input.dp_global_token_nums.begin(),
                    input.dp_global_token_nums.end(),
                    [](int32_t count) { return count < 0; }) ||
        std::any_of(input.dp_is_decode.begin(),
                    input.dp_is_decode.end(),
                    [](int32_t phase) { return phase != 0 && phase != 1; }) ||
        std::any_of(input.dp_global_kv_max_seq_lens.begin(),
                    input.dp_global_kv_max_seq_lens.end(),
                    [](int32_t length) { return length < 0; }) ||
        (local_tokens != 0 &&
         input.dp_is_decode[dp_rank] != static_cast<int32_t>(local_decode))) {
      return Status(StatusCode::INVALID_ARGUMENT,
                    "Task requires aligned unpadded DP counts and phases.");
    }
    plan.dp_global_token_nums.assign(input.dp_global_token_nums.begin(),
                                     input.dp_global_token_nums.end());
    plan.raw_dp_global_token_nums = plan.dp_global_token_nums;
    plan.dp_is_decode.assign(input.dp_is_decode.begin(),
                             input.dp_is_decode.end());
    plan.dp_global_kv_max_seq_lens.assign(
        input.dp_global_kv_max_seq_lens.begin(),
        input.dp_global_kv_max_seq_lens.end());
    plan.dp_global_batch_generations.assign(
        input.dp_global_batch_generations.begin(),
        input.dp_global_batch_generations.end());
    plan.dp_global_sequence_nums.assign(input.dp_global_sequence_nums.begin(),
                                        input.dp_global_sequence_nums.end());
    plan.dp_dspark_confidence = input.dp_dspark_confidence;
    plan.dp_dspark_rounds_since_full_width =
        input.dp_dspark_rounds_since_full_width;
  }
  bool all_decode = true;
  bool active = false;
  for (uint32_t rank = 0; rank < dp_size; ++rank) {
    if (plan.dp_global_token_nums[rank] == 0) {
      continue;
    }
    active = true;
    all_decode = all_decode && plan.dp_is_decode[rank] == 1;
  }
  run_models = active;
  decode = active && all_decode;
  output = std::move(plan);
  return Status();
}

Status TaskExecutionPipeline::validate_bootstrap(
    const LlmForwardInput& input) const {
  const auto host = SlotBuffer::model_input_view(input);
  const auto batch = model_batch(input);
  const auto& rows = input.input_params.embedding.mtp_bootstrap_row_idxes;
  const auto& hidden = input.input_params.embedding.mtp_bootstrap_embeddings;
  if (rows.empty()) {
    if (hidden.defined() && hidden.numel() != 0) {
      return invalid("MTP bootstrap hidden requires explicit sequence rows.");
    }
    return Status();
  }
  if (!batch.forward_type.is_decode() || !hidden.defined() ||
      !hidden.device().is_cpu() || hidden.dim() != 2 ||
      hidden.size(/*dim=*/0) != static_cast<int64_t>(rows.size()) ||
      hidden.size(/*dim=*/1) != context_hidden_size() ||
      !hidden.is_floating_point()) {
    return invalid("MTP bootstrap requires CPU hidden rows for Decode.");
  }
  std::vector<uint8_t> seen(host.q_seq_lens.size(), /*value=*/0);
  for (const int32_t row : rows) {
    if (row < 0 || static_cast<uint64_t>(row) >= seen.size() || seen[row] ||
        host.token_ids[row] < 0) {
      return invalid(
          "MTP bootstrap rows must be unique and have known tokens.");
    }
    seen[row] = 1;
  }
  return Status();
}

Status TaskExecutionPipeline::validate_input(SpeculativeSlot& slot,
                                             const LlmForwardInput& input) {
  const uint32_t requested_steps = input.input_params.meta.mtp_draft_steps;
  if (requested_steps != 0 &&
      (!speculative_capacity_->adaptive_draft_step ||
       requested_steps > speculative_capacity_->num_speculative_tokens)) {
    return invalid("Task MTP depth is unsupported or exceeds maximum depth.");
  }
  slot.draft_steps = requested_steps == 0
                         ? speculative_capacity_->num_speculative_tokens
                         : requested_steps;
  if (speculative_capacity_->adaptive_draft_step) {
    Status width_status = SlotBuffer::set_mtp_validate_width(
        *slot.validate_input, slot.draft_steps);
    if (!width_status.ok()) {
      return width_status;
    }
    if (slot.prefix_refresh) {
      width_status = SlotBuffer::set_mtp_validate_width(
          *slot.prefix_refresh->input, slot.draft_steps);
      if (!width_status.ok()) {
        return width_status;
      }
    }
  }
  const auto host = SlotBuffer::model_input_view(input);
  const auto batch = model_batch(input);
  Status status = slot.target_prefill->validate(host, batch);
  if (!status.ok()) {
    return status;
  }
  status = slot.target_prefill->validate_model_state(input);
  if (!status.ok()) {
    return status;
  }
  if (capacity_.model.enable_linear_attention &&
      batch.forward_type.is_decode() &&
      !input.input_params.meta.is_graph_warmup) {
    const auto& params = input.input_params;
    for (uint32_t row = 0; row < host.q_seq_lens.size(); ++row) {
      bool valid = host.kv_seq_lens[row] > host.q_seq_lens[row];
      if (!params.linear_state_validity_mask.empty()) {
        valid = params.linear_state_validity_mask[row] != 0;
      }
      const auto& read_ids = params.embedding.linear_state_read_ids;
      if (!read_ids.empty() &&
          read_ids[row] != params.embedding.linear_state_ids[row]) {
        valid = true;
      }
      if (!valid) {
        return invalid("Hybrid MTP decode requires a warm recurrent state.");
      }
    }
  }
  if (adaptive_verify_ && !adaptive_target_sampling(input.sampling_params)) {
    return invalid(
        "Adaptive DSpark task pipeline requires consistent sampling flags "
        "without log probabilities, structured sampling or non-neutral "
        "penalties.");
  }
  if (adaptive_verify_ && speculative_capacity_->unified_graph &&
      input.sampling_params.sampling_mode() != SamplingMode::GREEDY) {
    return invalid(
        "Adaptive DSpark unified graphs require greedy target sampling; "
        "disable unified graphs for positive or mixed temperatures.");
  }
  if (adaptive_verify_ && speculative_capacity_->unified_graph &&
      input.sampling_params.min_token_ids().defined()) {
    return invalid(
        "Adaptive minimum-length sampling requires split target/draft graphs.");
  }

  const auto& c = speculative_capacity_->common;
  const uint64_t count = host.q_seq_lens.size();
  if (batch.num_actual_sequences != count ||
      input.input_params.embedding.embedding_ids.size() > count ||
      input.input_params.embedding.request_ids.size() !=
          input.input_params.embedding.embedding_ids.size() ||
      batch.is_graph_warmup ||
      (!input.sampling_params.logprobs() &&
       input.sampling_params.max_top_logprobs() != 0) ||
      (!batch.forward_type.is_decode() && !batch.forward_type.is_prefill() &&
       !batch.forward_type.is_chunked_prefill() &&
       !batch.forward_type.is_mixed() &&
       batch.forward_type.value() != BatchForwardType::EMPTY)) {
    return invalid(
        "MTP requires unpadded ordinary sampling and sequence keys.");
  }
  auto& base_parallel = slot.target_prefill->model_params().parallel;
  status = plan_parallel(input.input_params.parallel,
                         c.dp_size,
                         c.dp_rank,
                         static_cast<uint32_t>(host.token_ids.size()),
                         batch.forward_type.is_decode(),
                         base_parallel,
                         slot.run_models,
                         slot.decode);
  if (!status.ok()) {
    return status;
  }
  const uint32_t width = slot.draft_steps + 1;
  for (const int32_t tokens : base_parallel.dp_global_token_nums) {
    if (static_cast<uint64_t>(tokens) * width >
        std::numeric_limits<int32_t>::max()) {
      return invalid("MTP DP token expansion exceeds int32 capacity.");
    }
  }
  slot.validate_input->model_params().parallel =
      expanded_parallel(base_parallel, slot.decode ? width : 1, false);
  if (slot.prefix_refresh) {
    slot.prefix_refresh->input->model_params().parallel =
        slot.validate_input->model_params().parallel;
  }
  for (uint32_t step = 0; step < draft_calls(slot); ++step) {
    const uint32_t scale = !slot.decode ? 1
                           : block_draft()
                               ? width - (speculative_capacity_->kind ==
                                                  SpeculativeTaskKind::DSPARK
                                              ? 1U
                                              : 0U)
                           : step == 0 ? 2
                                       : 1;
    slot.drafts[step]->input->model_params().parallel =
        expanded_parallel(base_parallel, scale, slot.decode && block_draft());
  }
  if (slot.decode &&
      input.input_params.embedding.embedding_ids.size() != count) {
    return Status(StatusCode::INVALID_ARGUMENT,
                  "MTP decode requires one embedding id per model row.");
  }
  slot.target_padded_batch_size =
      task_graph_batch_size(slot.validate_input->model_params().parallel,
                            captured_graph_batch_sizes_);
  if (slot.prefix_refresh) {
    auto& refresh = *slot.prefix_refresh;
    refresh.padded_batch_size =
        task_graph_batch_size(refresh.input->model_params().parallel,
                              refresh.captured_graph_batch_sizes);
  }
  for (uint32_t step = 0; step < draft_calls(slot); ++step) {
    auto& draft = *slot.drafts[step];
    draft.padded_batch_size = task_graph_batch_size(
        draft.input->model_params().parallel, draft.captured_graph_batch_sizes);
  }
  const uint32_t calibration_rows =
      input.input_params.meta.mtp_target_graph_rows;
  if (calibration_rows != 0) {
    if (!speculative_capacity_->adaptive_draft_step || !slot.decode ||
        requested_steps == 0 || calibration_rows < count * width ||
        !std::binary_search(captured_graph_batch_sizes_.begin(),
                            captured_graph_batch_sizes_.end(),
                            calibration_rows) ||
        (slot.prefix_refresh &&
         !std::binary_search(
             slot.prefix_refresh->captured_graph_batch_sizes.begin(),
             slot.prefix_refresh->captured_graph_batch_sizes.end(),
             calibration_rows))) {
      return invalid(
          "MTP calibration requires a captured padded verification shape.");
    }
    slot.target_padded_batch_size = calibration_rows;
    if (slot.prefix_refresh) {
      slot.prefix_refresh->padded_batch_size = calibration_rows;
    }
  }
  if (speculative_capacity_->adaptive_draft_step && requested_steps != 0 &&
      capacity_.max_graph_batch_size != 0 && slot.decode && count != 0 &&
      count <= capacity_.max_graph_batch_size /
                   (speculative_capacity_->num_speculative_tokens + 1) &&
      !input.input_params.meta.is_graph_warmup &&
      (slot.target_padded_batch_size == 0 ||
       (slot.prefix_refresh && slot.prefix_refresh->padded_batch_size == 0) ||
       std::any_of(
           slot.drafts.begin(),
           slot.drafts.begin() + draft_calls(slot),
           [](const auto& draft) { return draft->padded_batch_size == 0; }))) {
    return invalid(
        "Adaptive MTP requires prewarmed target/draft graph buckets.");
  }
  if (block_draft_graphs() && capacity_.max_graph_batch_size != 0 &&
      slot.decode && count != 0 && !input.input_params.meta.is_graph_warmup &&
      (slot.target_padded_batch_size == 0 ||
       slot.drafts.front()->padded_batch_size == 0)) {
    return invalid(
        "Block draft graph task requires prewarmed Target and Draft batch "
        "buckets.");
  }
  slot.rows = static_cast<uint32_t>(count);
  slot.model_tokens = static_cast<uint32_t>(host.token_ids.size());
  status = slot.target_prefill->validate_sampling(input.sampling_params,
                                                  slot.model_tokens);
  if (!status.ok()) {
    return status;
  }
  if (slot.decode &&
      (host.token_ids.size() != count ||
       std::any_of(host.q_seq_lens.begin(),
                   host.q_seq_lens.end(),
                   [](int32_t length) { return length != 1; }))) {
    return invalid("MTP Decode requires one base token per sequence.");
  }
  status = validate_bootstrap(input);
  if (!status.ok()) {
    return status;
  }
  if (!slot.run_models) {
    return Status();
  }
  const int64_t blocks = std::min(paged_cache_blocks(kv_caches_),
                                  paged_cache_blocks(draft_->kv_caches));
  if (blocks == 0) {
    return Status(StatusCode::UNAVAILABLE, "MTP requires Target and Draft KV.");
  }
  if (count == 0) {
    if (adaptive_verify_ && slot.decode && c.dp_size > 1) {
      return plan_adaptive_verify(slot, input);
    }
    return Status();
  }
  if (std::any_of(
          host.block_tables.begin(),
          host.block_tables.end(),
          [blocks](int32_t block) { return block < 0 || block >= blocks; })) {
    return invalid("MTP page table is outside allocated Target/Draft cache.");
  }
  slot.upper_positions.clear();
  slot.upper_kv_lengths.clear();
  uint64_t offset = 0;
  for (uint64_t row = 0; row < count; ++row) {
    const int64_t kv = host.kv_seq_lens[row];
    const int64_t q = host.q_seq_lens[row];
    // Cache hits may have q < kv even when scheduler chunking is disabled.
    if (kv > c.max_kv_seq_len ||
        (kv + c.logical_block_size - 1) / c.logical_block_size >
            host.block_table_width) {
      return invalid("MTP KV length or Prefill mode exceeds capacity.");
    }
    for (int64_t item = 0; item < q; ++item, ++offset) {
      const int64_t position = host.positions[offset];
      const int32_t token = host.token_ids[offset];
      if (position != kv - q + item || position < 0 ||
          position >= c.max_positions ||
          (token >= 0 && static_cast<uint32_t>(token) >= c.vocab_size) ||
          (!slot.decode && token < 0)) {
        return invalid("Invalid MTP base token or rotary position.");
      }
      const int32_t block = host.block_tables[row * host.block_table_width +
                                              position / c.logical_block_size];
      if (host.new_cache_slots[offset] !=
          static_cast<int64_t>(block) * c.logical_block_size +
              position % c.logical_block_size) {
        return invalid("MTP KV write slot does not match the page table.");
      }
    }
    if (slot.decode) {
      // Scheduler's one pending placeholder can represent K+1 accepted tokens.
      const int64_t pending =
          host.token_ids[row] < 0
              ? speculative_capacity_->num_speculative_tokens
              : 0;
      const int64_t upper_position = host.positions[row] + pending;
      const int64_t upper_kv = kv + pending;
      if (upper_position + speculative_capacity_->num_speculative_tokens >=
              c.max_positions ||
          upper_kv + speculative_capacity_->num_speculative_tokens >
              c.max_kv_seq_len ||
          host.positions[row] < 1) {
        return invalid(
            "MTP speculative position upper bound exceeds capacity.");
      }
      slot.upper_positions.push_back(static_cast<int32_t>(upper_position));
      slot.upper_kv_lengths.push_back(static_cast<int32_t>(upper_kv));
    } else if (kv >= c.max_positions || kv >= c.max_kv_seq_len) {
      return invalid("MTP Prefill must leave capacity for its next token.");
    }
  }
  for (const int32_t token : input.input_params.embedding.extra_token_ids) {
    if (token < -1 ||
        (token >= 0 && static_cast<uint32_t>(token) >= c.vocab_size)) {
      return invalid("Invalid MTP Prefill extra token.");
    }
  }
  if (!slot.decode) {
    return SlotBuffer::plan_mtp_prefill(
        *slot.draft_prefill,
        host,
        batch,
        input.input_params.embedding.extra_token_ids,
        input.sampling_params);
  }
  auto conservative = host;
  conservative.positions = slot.upper_positions;
  conservative.kv_seq_lens = slot.upper_kv_lengths;
  status =
      SlotBuffer::plan_mtp_decode(*slot.validate_input, conservative, batch);
  if (!status.ok()) {
    return status;
  }
  for (auto& invocation : std::span(slot.drafts).first(draft_calls(slot))) {
    status =
        SlotBuffer::plan_mtp_decode(*invocation->input, conservative, batch);
    if (!status.ok()) {
      return status;
    }
  }
  if (slot.prefix_refresh) {
    status = SlotBuffer::plan_mtp_decode(
        *slot.prefix_refresh->input, conservative, batch);
    if (!status.ok()) {
      return status;
    }
  }
  if (block_draft_graphs() && capacity_.max_graph_batch_size != 0) {
    const auto& invocation = *slot.drafts.front();
    const uint32_t graph_rows =
        input.input_params.meta.is_graph_warmup
            ? task_graph_batch_size(invocation.input->model_params().parallel,
                                    invocation.graph_batch_sizes)
            : invocation.padded_batch_size;
    if (!input.input_params.meta.is_graph_warmup &&
        !invocation.captured_graph_kv_capacities.contains(graph_rows)) {
      return invalid(
          "Block draft graph task requires a warmed Draft Slot binding.");
    }
  }
  if (adaptive_verify_) {
    if (c.dp_size > 1) {
      return plan_adaptive_verify(slot, input);
    }
    if (input.input_params.meta.is_graph_warmup) {
      slot.verify->plan = {
          slot.rows * speculative_capacity_->num_speculative_tokens,
          task_graph_batch_size(slot.validate_input->model_params().parallel,
                                graph_batch_sizes_)};
      slot.verify->max_kv_length =
          static_cast<uint32_t>(*std::max_element(
              slot.upper_kv_lengths.begin(), slot.upper_kv_lengths.end())) +
          speculative_capacity_->num_speculative_tokens;
      return Status();
    }
    if (adaptive_verify_) {
      return plan_adaptive_verify(slot, input);
    }
  }
  return Status();
}

Status TaskExecutionPipeline::plan_adaptive_verify(
    SpeculativeSlot& slot,
    const LlmForwardInput& input) {
  CHECK(slot.verify != nullptr);
  if (capacity_.dp_size > 1) {
    return plan_dp_adaptive_dspark(slot, input);
  }
  auto& adaptive = *slot.verify;
  const uint32_t draft_width = speculative_capacity_->num_speculative_tokens;
  const uint64_t generation = next_task_id_;
  const uint64_t eligible_generation = generation > 2 ? generation - 2 : 0;
  // Engine coordinated submission retires n-2 on every sampling rank before
  // Step(n) admission (ContinuousScheduler::step_with_schedule_overlap and
  // LLMEngine::collect_step_outputs). Never use opportunistic n-1 feedback:
  // its availability can differ between rank-local state executors.
  if (adaptive_verify_->retired_generation < eligible_generation) {
    return invalid(
        "Adaptive DSpark requires coordinated retirement through Task n-2.");
  }
  const auto& embedding_ids = input.input_params.embedding.embedding_ids;
  const auto& request_ids = input.input_params.embedding.request_ids;
  adaptive.lagged_confidence.clear();
  uint64_t rounds_since_full_width = 0;
  bool complete_feedback = eligible_generation != 0;
  for (uint32_t row = 0; row < slot.rows; ++row) {
    const int32_t id = embedding_ids[row];
    if (id <= 0 ||
        static_cast<uint64_t>(id) >= adaptive_verify_->histories.size()) {
      complete_feedback = false;
      continue;
    }
    const auto& history = adaptive_verify_->histories[id];
    const AdaptiveVerifyState::Feedback* feedback = nullptr;
    for (const auto& version : history.versions) {
      if (version.generation != 0 &&
          version.generation <= eligible_generation &&
          (feedback == nullptr || version.generation > feedback->generation)) {
        feedback = &version;
      }
    }
    if (history.request_id != request_ids[row] || feedback == nullptr) {
      complete_feedback = false;
      continue;
    }
    adaptive.lagged_confidence.insert(
        adaptive.lagged_confidence.end(),
        feedback->confidence.begin(),
        feedback->confidence.begin() + draft_width);
    rounds_since_full_width =
        std::max(rounds_since_full_width, feedback->rounds_since_full_width);
  }
  if (!complete_feedback) {
    adaptive.lagged_confidence.clear();
  }
  adaptive.max_kv_length =
      static_cast<uint32_t>(*std::max_element(slot.upper_kv_lengths.begin(),
                                              slot.upper_kv_lengths.end())) +
      draft_width;
  const auto unified_rows =
      speculative_capacity_->unified_graph
          ? adaptive_unified_graph_rows(
                slot.rows, captured_graph_batch_sizes_, draft_width)
          : std::vector<int64_t>();
  const std::span<const int64_t> candidate_rows =
      speculative_capacity_->unified_graph
          ? std::span<const int64_t>(unified_rows)
          : std::span<const int64_t>(captured_graph_batch_sizes_);
  std::vector<SpecGraphCost> costs;
  costs.reserve(candidate_rows.size());
  for (const int64_t graph_rows : candidate_rows) {
    costs.emplace_back(SpecGraphCost{static_cast<uint32_t>(graph_rows), 0.0});
  }
  VerifyBudgetOptions options;
  options.draft_width = draft_width;
  options.min_gain = speculative_capacity_->adaptive_min_gain;
  // MTP's first call has two rows per request (repair plus current token);
  // DSpark generates the whole block in one call.
  const uint32_t draft_rows = slot.drafts.front()->padded_batch_size;
  const uint32_t rows_per_request = block_draft() ? draft_width : 2;
  CHECK_EQ(draft_rows % rows_per_request, 0U);
  const uint32_t physical_requests = draft_rows / rows_per_request;
  const auto profile = speculative_capacity_->adaptive_costs->lookup(
      physical_requests, adaptive.max_kv_length, draft_width);
  if (profile) {
    options.cost_scope = profile->cost_scope;
    options.fixed_draft_time_ms = profile->fixed_draft_time_ms;
    options.commit_time_ms = profile->commit_time_ms;
    for (auto& cost : costs) {
      const auto profiled =
          std::lower_bound(profile->target_graph_costs.begin(),
                           profile->target_graph_costs.end(),
                           cost.graph_rows,
                           [](const SpecGraphCost& candidate, uint32_t rows) {
                             return candidate.graph_rows < rows;
                           });
      if (profiled != profile->target_graph_costs.end() &&
          profiled->graph_rows == cost.graph_rows) {
        cost.target_time_ms = profiled->target_time_ms;
      }
    }
  }
  std::optional<VerifyBudgetPlan> previous;
  uint64_t previous_generation = 0;
  for (const auto& retired : adaptive_verify_->retired_plans) {
    if (retired.generation != 0 && retired.generation <= eligible_generation &&
        retired.generation > previous_generation &&
        retired.embedding_ids == embedding_ids &&
        retired.request_ids == request_ids) {
      previous = retired.plan;
      previous_generation = retired.generation;
    }
  }
  std::optional<VerifyBudgetPlan> plan;
  if (adaptive_verify_->calibration_point) {
    const uint32_t graph_rows =
        adaptive_verify_->calibration_point->target_graph_rows;
    if (speculative_capacity_->unified_graph) {
      std::vector<SpecGraphCost> target_graphs;
      target_graphs.reserve(captured_graph_batch_sizes_.size());
      for (const int64_t rows : captured_graph_batch_sizes_) {
        target_graphs.emplace_back(
            SpecGraphCost{static_cast<uint32_t>(rows), 0.0});
      }
      plan = VerifyBudgetPlanner::calibration_plan(
          slot.rows,
          physical_requests,
          graph_rows,
          target_graphs,
          draft_width,
          costs,
          adaptive_verify_->calibration_point
              ? std::optional(adaptive_verify_->calibration_point->requests)
              : std::nullopt);
    } else {
      plan = VerifyBudgetPlanner::calibration_plan(
          slot.rows, physical_requests, graph_rows, costs, draft_width);
    }
    if (!plan) {
      return invalid(
          "DSpark calibration requires a known G and prewarmed full-width "
          "capacity for the physical draft request bucket.");
    }
  } else {
    plan = VerifyBudgetPlanner::plan(slot.rows,
                                     adaptive.lagged_confidence,
                                     costs,
                                     options,
                                     previous,
                                     rounds_since_full_width);
  }
  if (!plan) {
    return invalid("Adaptive DSpark requires a prewarmed full-width graph.");
  }
  if (speculative_capacity_->unified_graph &&
      !adaptive.unified_graph_bindings.contains(
          adaptive_unified_graph_key(slot.rows, plan->graph_rows))) {
    return invalid(
        "Adaptive DSpark unified graph requires a prewarmed batch/verify "
        "binding.");
  }
  adaptive.plan = *plan;
  adaptive.generation = generation;
  adaptive.selected_draft_tokens = plan->draft_budget;
  adaptive.embedding_ids.assign(embedding_ids.begin(), embedding_ids.end());
  adaptive.request_ids.assign(request_ids.begin(), request_ids.end());
  slot.target_padded_batch_size = plan->graph_rows;
  auto& parallel = slot.validate_input->model_params().parallel;
  parallel.raw_dp_global_token_nums = {
      static_cast<int32_t>(slot.rows + plan->draft_budget)};
  parallel.dp_global_token_nums = {static_cast<int32_t>(plan->graph_rows)};
  parallel.dp_is_decode = {1};
  return Status();
}

Status TaskExecutionPipeline::plan_dp_adaptive_dspark(
    SpeculativeSlot& slot,
    const LlmForwardInput& input) {
  const auto& source = input.input_params.parallel;
  const auto& requests = source.dp_global_sequence_nums;
  const uint32_t width = speculative_capacity_->num_speculative_tokens;
  const bool warmup = input.input_params.meta.is_graph_warmup;
  const uint64_t eligible_generation =
      next_task_id_ > 2 ? next_task_id_ - 2 : 0;
  if (!warmup && adaptive_verify_->retired_generation < eligible_generation) {
    return invalid(
        "Adaptive DSpark DP requires coordinated retirement through Task n-2.");
  }
  if (requests.size() != capacity_.dp_size ||
      source.dp_global_kv_max_seq_lens.size() != capacity_.dp_size ||
      requests[capacity_.dp_rank] != slot.rows ||
      std::any_of(requests.begin(),
                  requests.end(),
                  [&](int32_t count) {
                    return count < 0 || static_cast<uint32_t>(count) >
                                            capacity_.model.max_sequences;
                  }) ||
      std::any_of(source.dp_global_kv_max_seq_lens.begin(),
                  source.dp_global_kv_max_seq_lens.end(),
                  [](int32_t length) { return length < 0; })) {
    return invalid(
        "Adaptive DSpark DP requires complete global request and KV metadata.");
  }
  const uint32_t maximum_requests =
      *std::max_element(requests.begin(), requests.end());
  if (maximum_requests == 0) {
    return invalid("Adaptive DSpark DP decode requires an active shard.");
  }
  auto& adaptive = *slot.verify;
  adaptive.max_kv_length =
      *std::max_element(source.dp_global_kv_max_seq_lens.begin(),
                        source.dp_global_kv_max_seq_lens.end()) +
      width;
  const auto& draft = *slot.drafts.front();
  const auto captured =
      draft.captured_graph_kv_capacities.find(draft.padded_batch_size);
  if (!warmup && (draft.padded_batch_size == 0 ||
                  captured == draft.captured_graph_kv_capacities.end())) {
    return invalid(
        "Adaptive DSpark DP requires a prewarmed global Draft batch bucket.");
  }
  const std::span<const int64_t> candidates =
      warmup ? std::span<const int64_t>(graph_batch_sizes_)
             : std::span<const int64_t>(captured_graph_batch_sizes_);
  std::vector<SpecGraphCost> costs;
  costs.reserve(candidates.size());
  VerifyBudgetOptions options;
  options.draft_width = width;
  options.min_gain = speculative_capacity_->adaptive_min_gain;
  const uint32_t draft_rows = slot.drafts.front()->padded_batch_size;
  CHECK_EQ(draft_rows % width, 0U);
  const uint32_t physical_requests = draft_rows / width;
  const auto profile = speculative_capacity_->adaptive_costs->lookup(
      physical_requests, adaptive.max_kv_length, width);
  if (profile) {
    options.cost_scope = profile->cost_scope;
    options.fixed_draft_time_ms = profile->fixed_draft_time_ms;
    options.commit_time_ms = profile->commit_time_ms;
  }
  for (const int64_t graph_rows : candidates) {
    double latency = 0.0;
    if (profile && !warmup) {
      const auto cost =
          std::lower_bound(profile->target_graph_costs.begin(),
                           profile->target_graph_costs.end(),
                           graph_rows,
                           [](const SpecGraphCost& value, int64_t rows) {
                             return value.graph_rows < rows;
                           });
      if (cost != profile->target_graph_costs.end() &&
          cost->graph_rows == graph_rows) {
        latency = cost->target_time_ms;
      }
    }
    costs.emplace_back(
        SpecGraphCost{static_cast<uint32_t>(graph_rows), latency});
  }
  std::optional<std::vector<VerifyBudgetPlan>> plans;
  if (adaptive_verify_->calibration_point && !warmup) {
    const auto plan = VerifyBudgetPlanner::calibration_plan(
        maximum_requests,
        physical_requests,
        adaptive_verify_->calibration_point->target_graph_rows,
        costs,
        width);
    if (plan) {
      plans.emplace();
      plans->reserve(requests.size());
      for (const int32_t count : requests) {
        plans->emplace_back(VerifyBudgetPlan{
            std::min(static_cast<uint32_t>(count) * width,
                     plan->graph_rows - static_cast<uint32_t>(count)),
            plan->graph_rows});
      }
    }
  } else {
    plans = VerifyBudgetPlanner::parallel_plan(
        requests,
        source.dp_dspark_confidence,
        source.dp_dspark_rounds_since_full_width,
        costs,
        options);
  }
  if (!plans) {
    return invalid(
        "Adaptive DSpark DP requires aligned feedback and a prewarmed common "
        "graph.");
  }
  adaptive.plan = (*plans)[capacity_.dp_rank];
  adaptive.generation = next_task_id_;
  adaptive.selected_draft_tokens = adaptive.plan.draft_budget;
  adaptive.embedding_ids = input.input_params.embedding.embedding_ids;
  adaptive.request_ids = input.input_params.embedding.request_ids;
  adaptive.lagged_confidence.clear();
  if (!source.dp_dspark_confidence.empty() && slot.rows != 0) {
    uint64_t preceding_rows = 0;
    for (uint32_t shard = 0; shard < capacity_.dp_rank; ++shard) {
      preceding_rows += requests[shard];
    }
    const auto first =
        source.dp_dspark_confidence.begin() + preceding_rows * width;
    adaptive.lagged_confidence.assign(first, first + slot.rows * width);
  }
  slot.target_padded_batch_size = adaptive.plan.graph_rows;
  auto& parallel = slot.validate_input->model_params().parallel;
  parallel.raw_dp_global_token_nums.clear();
  parallel.raw_dp_global_token_nums.reserve(requests.size());
  for (uint32_t shard = 0; shard < requests.size(); ++shard) {
    parallel.raw_dp_global_token_nums.emplace_back(
        requests[shard] + (*plans)[shard].draft_budget);
  }
  parallel.dp_global_token_nums.assign(requests.size(),
                                       adaptive.plan.graph_rows);
  parallel.dp_is_decode.assign(requests.size(), 1);
  VLOG(1) << "DSpark adaptive DP budget: rank=" << capacity_.dp_rank
          << ", B=" << slot.rows
          << ", selected_K=" << adaptive.plan.draft_budget
          << ", physical_G=" << adaptive.plan.graph_rows;
  return Status();
}

Status TaskExecutionPipeline::plan_sampling(SpeculativeSlot& slot,
                                            const LlmForwardInput& input) {
  const auto& base = input.sampling_params;
  slot.samples = static_cast<uint32_t>(base.num_sample_seqs());
  if (slot.rows == 0 || !slot.decode) {
    return Status();
  }
  // The verify pipeline carries no per-token grammar masks into the
  // rejection sampling; JSON-constrained decode runs through the MTP worker
  // path, which installs full-width masks. Reject here instead of silently
  // verifying unconstrained.
  if (base.filter_bitmask().defined() || base.filter_mask().defined()) {
    return invalid(
        "Task pipeline speculative decode does not support JSON grammar "
        "constraints.");
  }
  if (slot.samples != slot.rows) {
    return invalid("MTP Decode must sample every base row.");
  }
  const auto* selected = base.selected_token_idxes().const_data_ptr<int32_t>();
  for (uint32_t row = 0; row < slot.rows; ++row) {
    if (selected[row] != static_cast<int32_t>(row)) {
      return invalid("MTP Decode sampling rows must be in sequence order.");
    }
  }
  if (adaptive_verify_) {
    const auto indices =
        torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU);
    const uint32_t graph_rows = slot.verify->plan.graph_rows;
    slot.host_validate_sampling = SamplingParameters();
    slot.host_validate_sampling.selected_token_idxes() =
        torch::arange(graph_rows, indices);
    return slot.validate_input->validate_sampling(slot.host_validate_sampling,
                                                  graph_rows);
  }
  const int64_t width = slot.draft_steps + 1;
  slot.host_validate_sampling = base;
  expand_sampling_params_for_verify(slot.host_validate_sampling,
                                    width,
                                    slot.rows * width,
                                    torch::Device(torch::kCPU));
  // Rejection sampling produces the final log probabilities once.
  slot.host_validate_sampling.logprobs() = false;
  slot.host_validate_sampling.max_top_logprobs() = 0;
  return slot.validate_input->validate_sampling(slot.host_validate_sampling,
                                                slot.rows * width);
}

void TaskExecutionPipeline::prepare_bootstrap(SpeculativeSlot& slot,
                                              const LlmForwardInput& input) {
  const auto host = SlotBuffer::model_input_view(input);
  auto& b = slot.bootstrap;
  const int64_t count =
      input.input_params.embedding.mtp_bootstrap_row_idxes.size();
  b.rows = prefix(b.row_storage, count);
  b.tokens = prefix(b.token_storage, count);
  b.state.hidden =
      block_draft() ? torch::Tensor() : prefix(b.storage.hidden, count);
  b.state.positions = prefix(b.storage.positions, count);
  b.state.kv_seq_lens = prefix(b.storage.kv_seq_lens, count);
  b.state.repair_required = block_draft()
                                ? torch::Tensor()
                                : prefix(b.storage.repair_required, count);
  if (count == 0) {
    return;
  }
  auto host_rows = prefix(b.host_rows, count);
  auto host_tokens = prefix(b.host_tokens, count);
  auto host_positions = prefix(b.host_state.positions, count);
  auto host_kv = prefix(b.host_state.kv_seq_lens, count);
  for (int64_t index = 0; index < count; ++index) {
    const int32_t row =
        input.input_params.embedding.mtp_bootstrap_row_idxes[index];
    host_rows.data_ptr<int64_t>()[index] = row;
    host_tokens.data_ptr<int64_t>()[index] = host.token_ids[row];
    host_positions.data_ptr<int32_t>()[index] = host.positions[row];
    host_kv.data_ptr<int32_t>()[index] = host.kv_seq_lens[row];
  }
  if (!block_draft()) {
    auto host_hidden = prefix(b.host_state.hidden, count);
    auto host_repair = prefix(b.host_state.repair_required, count);
    host_hidden.select(/*dim=*/1, /*index=*/0).zero_();
    host_hidden.select(/*dim=*/1, /*index=*/1)
        .copy_(input.input_params.embedding.mtp_bootstrap_embeddings);
    copy_input(b.state.hidden, host_hidden, prepare_stream_);
    copy_input(b.state.repair_required, host_repair, prepare_stream_);
  }
  copy_input(b.rows, host_rows, prepare_stream_);
  copy_input(b.tokens, host_tokens, prepare_stream_);
  copy_input(b.state.positions, host_positions, prepare_stream_);
  copy_input(b.state.kv_seq_lens, host_kv, prepare_stream_);
}

void TaskExecutionPipeline::apply_bootstrap(SpeculativeSlot& slot) {
  const auto& b = slot.bootstrap;
  if (b.rows.numel() == 0) {
    return;
  }
  const auto& state = slot.state->state_;
  slot.state->tokens_.index_copy_(/*dim=*/0, b.rows, b.tokens);
  if (!block_draft()) {
    state.previous_tokens.index_copy_(/*dim=*/0, b.rows, b.tokens);
    state.hidden.index_copy_(/*dim=*/0, b.rows, b.state.hidden);
    state.repair_required.index_copy_(
        /*dim=*/0, b.rows, b.state.repair_required);
  }
  state.positions.index_copy_(/*dim=*/0, b.rows, b.state.positions);
  state.kv_seq_lens.index_copy_(/*dim=*/0, b.rows, b.state.kv_seq_lens);
  state.accepted_lengths.index_fill_(/*dim=*/0, b.rows, /*value=*/1);
}

Status TaskExecutionPipeline::warmup_speculative_graphs(
    const LlmForwardInput& input) {
  const auto host = SlotBuffer::model_input_view(input);
  const auto batch = model_batch(input);
  CHECK(input.input_params.meta.is_graph_warmup);
  if (graph_batch_sizes_.empty()) {
    return Status();
  }
  if (speculative_capacity_->adaptive_draft_step &&
      input.input_params.meta.mtp_draft_steps == 0 &&
      batch.forward_type.is_decode()) {
    // Warm every reachable depth while startup owns the drained pipeline.
    // Smaller depths reuse the same maximum-capacity Slot storage, not another
    // model or independent pipeline. Ordinary serving never triggers capture.
    for (uint32_t depth = 1;
         depth <= speculative_capacity_->num_speculative_tokens;
         ++depth) {
      LlmForwardInput candidate = input.clone();
      candidate.input_params.meta.mtp_draft_steps = depth;
      const Status status = warmup_speculative_graphs(candidate);
      if (!status.ok()) {
        return status;
      }
    }
    return Status();
  }
  if (adaptive_verify_) {
    // Scheduler startup runs Prefill warmup before the Decode bucket sweep.
    // Preserve its normal context/bootstrap execution; only Decode owns the
    // prepared block-draft/verify bindings and their adaptive cost
    // measurements.
    if (!batch.forward_type.is_decode()) {
      return Status();
    }
    return warmup_packed_verify_graphs(input);
  }
  // Reuse scheduler-owned warmup KV and each SpeculativeSlot's final model
  // input views.
  for (auto& task_slot : slots_) {
    auto* slot = task_slot->speculative.get();
    Status status = validate_input(*slot, input);
    if (!status.ok()) {
      return status;
    }
  }
  const uint32_t padded_batch_size = task_graph_batch_size(
      slots_.front()->speculative->validate_input->model_params().parallel,
      graph_batch_sizes_);
  const auto bucket = std::lower_bound(captured_graph_batch_sizes_.begin(),
                                       captured_graph_batch_sizes_.end(),
                                       padded_batch_size);
  c10::DeviceGuard device_guard(device_.unwrap());
  auto guard = task_stream_.set_stream_guard();
  if (padded_batch_size != 0 && (bucket == captured_graph_batch_sizes_.end() ||
                                 *bucket != padded_batch_size)) {
    for (auto& task_slot : slots_) {
      auto* slot = task_slot->speculative.get();
      auto& binding = *slot->validate_input;
      Status status;
      if (slot->rows == 0) {
        status = binding.prepare_decode_padded(
            {},
            {BatchForwardType::EMPTY, 0, batch.batch_id},
            padded_batch_size,
            prepare_stream_);
      } else {
        auto conservative = host;
        conservative.positions = slot->upper_positions;
        conservative.kv_seq_lens = slot->upper_kv_lengths;
        status = SlotBuffer::prepare_planned_mtp_decode(*slot->validate_input,
                                                        conservative,
                                                        batch,
                                                        prepare_stream_,
                                                        padded_batch_size);
      }
      CHECK(status.ok()) << status.message();
      if (capacity_.model.enable_linear_attention) {
        auto input_guard = prepare_stream_.set_stream_guard();
        binding.prepare_model_state(
            input, speculative_capacity_->num_speculative_tokens + 1);
      }
      CHECK_EQ(prepare_stream_.synchronize(), 0);

      pad_task_parallel(binding.model_params().parallel, padded_batch_size);
      executor_.prepare_attention_metadata(kv_caches_, binding.model_params());
      executor_.warmup_prepared_graph(binding.tokens(),
                                      binding.positions(),
                                      kv_caches_,
                                      binding.model_params());
    }
    CHECK_EQ(task_stream_.synchronize(), 0);
    captured_graph_batch_sizes_.insert(bucket, padded_batch_size);
  }
  // Different sequence counts can share a target bucket while requiring
  // different draft buckets. Always visit the draft bindings independently.
  if (!block_draft()) {
    warmup_draft_graphs(input);
  } else if ((block_draft_graphs() ||
              speculative_capacity_->hybrid_recurrent_target) &&
             padded_batch_size != 0) {
    // Prefill and idle warmup do not plan the expanded decode bindings.
    // DSpark has N query rows; DFlash2 includes its anchor for N+1 rows.
    // Capture each final draft Slot binding independently of target padding.
    for (auto& task_slot : slots_) {
      auto& slot = *task_slot->speculative;
      auto& binding = *slot.drafts.front()->input;
      auto conservative = host;
      conservative.positions = slot.upper_positions;
      conservative.kv_seq_lens = slot.upper_kv_lengths;
      auto& invocation = *slot.drafts.front();
      const uint32_t draft_rows = task_graph_batch_size(
          binding.model_params().parallel, invocation.graph_batch_sizes);
      const Status status = SlotBuffer::prepare_planned_mtp_decode(
          binding, conservative, batch, prepare_stream_, draft_rows);
      CHECK(status.ok()) << status.message();
      CHECK_EQ(prepare_stream_.synchronize(), 0);
      pad_task_parallel(binding.model_params().parallel, draft_rows);
      draft_->executor.prepare_attention_metadata(draft_->kv_caches,
                                                  binding.model_params());
      draft_->executor.warmup_prepared_graph(binding.tokens(),
                                             binding.positions(),
                                             draft_->kv_caches,
                                             binding.model_params());
      const int32_t maximum_kv_length = *std::max_element(
          binding.model_params().attention.host.kv_seq_lens.begin(),
          binding.model_params().attention.host.kv_seq_lens.end());
      invocation.captured_graph_kv_capacities.try_emplace(
          draft_rows,
          dspark_graph_kv_capacity(capacity_,
                                   binding.input_scratch_->rows_per_sequence_,
                                   maximum_kv_length));
      // Warm every Slot's context binding, even if synthetic Tasks only
      // retire through the first Slot. No live or reserved KV row is written.
      auto& target_input = *slot.validate_input;
      slot.target_output = executor_.forward(target_input.tokens(),
                                             target_input.positions(),
                                             kv_caches_,
                                             target_input.model_params());
      const int64_t context_rows = target_input.tokens().numel();
      auto context_slots = prefix(slot.context_slots_storage, context_rows);
      context_slots.fill_(-1);
      // Slot inputs clear the output-only warmup marker. Context capture
      // explicitly needs it to admit this final borrowed binding.
      auto& context_params = target_input.model_params();
      const bool was_warmup = context_params.meta.is_graph_warmup;
      context_params.meta.is_graph_warmup = true;
      std::vector<torch::Tensor> context_outputs{
          slot.target_output.aux_hidden_states};
      if (auto* python = executor_.python_impl(); python != nullptr) {
        // KV tiers own distinct output buffers. Warm every borrowed address,
        // not just the tier selected by the synthetic startup sequence.
        context_outputs = python->get_prepared_aux_hidden_states(
            target_input.tokens(), target_input.positions(), context_params);
      }
      for (const auto& hidden : context_outputs) {
        const auto context_output = draft_->model.write_context_kv(
            hidden.narrow(0, 0, context_rows),
            target_input.positions().narrow(0, 0, context_rows),
            context_slots,
            draft_->kv_caches,
            context_params);
        CHECK(context_output.hidden_states.defined());
      }
      context_params.meta.is_graph_warmup = was_warmup;
      auto& captured = invocation.captured_graph_batch_sizes;
      const auto draft_bucket =
          std::lower_bound(captured.begin(), captured.end(), draft_rows);
      if (draft_bucket == captured.end() || *draft_bucket != draft_rows) {
        captured.insert(draft_bucket, draft_rows);
      }
    }
    CHECK_EQ(task_stream_.synchronize(), 0);
  }
  return Status();
}

void TaskExecutionPipeline::bind_packed_verify_views(SpeculativeSlot& slot) {
  auto& adaptive = *slot.verify;
  const int64_t draft_width = speculative_capacity_->num_speculative_tokens;
  const int64_t proposal_bytes =
      static_cast<int64_t>(slot.rows) * draft_width * 8;
  const int64_t confidence_bytes =
      static_cast<int64_t>(slot.rows) * draft_width * 4;
  const int64_t feedback_bytes =
      static_cast<int64_t>(slot.rows) * (draft_width + 1) * 4;
  slot.proposal = prefix(adaptive.arena_storage, proposal_bytes)
                      .view(torch::kInt64)
                      .view({slot.rows, draft_width});
  adaptive.confidence =
      adaptive.arena_storage.narrow(/*dim=*/0, proposal_bytes, confidence_bytes)
          .view(torch::kFloat32)
          .view({slot.rows, draft_width});
  adaptive.widths =
      adaptive.arena_storage
          .narrow(/*dim=*/0, proposal_bytes + confidence_bytes, slot.rows * 4)
          .view(torch::kInt32);
  adaptive.consensus =
      prefix(adaptive.arena_storage, proposal_bytes + confidence_bytes);
  adaptive.feedback = adaptive.arena_storage.narrow(
      /*dim=*/0, proposal_bytes, feedback_bytes);
  adaptive.host_feedback =
      prefix(adaptive.host_feedback_storage, feedback_bytes);
  adaptive.host_confidence = prefix(adaptive.host_feedback, confidence_bytes)
                                 .view(torch::kFloat32)
                                 .view({slot.rows, draft_width});
  adaptive.host_widths =
      adaptive.host_feedback.narrow(/*dim=*/0, confidence_bytes, slot.rows * 4)
          .view(torch::kInt32);
  adaptive.previous_tokens = prefix(adaptive.previous_storage, slot.rows);
  adaptive.offsets = prefix(adaptive.offsets_storage, slot.rows + 1);
  adaptive.owners = prefix(adaptive.owners_storage, adaptive.plan.graph_rows);
  adaptive.columns = prefix(adaptive.columns_storage, adaptive.plan.graph_rows);
  slot.target_tokens =
      prefix(slot.target_token_storage, adaptive.plan.graph_rows);
  slot.context_slots =
      prefix(slot.context_slots_storage, adaptive.plan.graph_rows);
}

void TaskExecutionPipeline::bind_packed_verify_graph_input(
    SpeculativeSlot& slot,
    bool suppress_kv_writes) {
  auto& adaptive = *slot.verify;
  CHECK(slot.rows != 0 || (capacity_.dp_size > 1 && suppress_kv_writes));
  CHECK_LE(slot.rows, capacity_.model.max_sequences);
  CHECK_GE(adaptive.plan.graph_rows, slot.rows);
  DSparkAdaptiveGraphInput input;
  input.arena = adaptive.arena_storage;
  input.control = adaptive.control;
  input.anchors = slot.state->token_storage_;
  input.base_positions = slot.state->storage_.positions;
  input.base_kv_lengths = slot.state->storage_.kv_seq_lens;
  input.base_block_table =
      (block_draft() ? slot.drafts.front() : slot.drafts.back())
          ->input->model_device_.block_tables.narrow(
              /*dim=*/0, /*start=*/0, capacity_.model.max_sequences);
  input.offsets = adaptive.offsets_storage;
  input.owners = prefix(adaptive.owners_storage, adaptive.plan.graph_rows);
  input.columns = prefix(adaptive.columns_storage, adaptive.plan.graph_rows);
  if (slot.recurrent) {
    input.linear_state_indices = slot.recurrent->state_indices;
    input.num_accepted_tokens = slot.recurrent->accepted_lengths;
  }
  input.block_size = capacity_.logical_block_size;
  input.draft_width = speculative_capacity_->num_speculative_tokens;
  slot.validate_input->model_params().graph.dspark_adaptive = std::move(input);
  // B and the startup write mask are Device values, not capture-time scalars.
  // All sources retain their full Slot capacity so a G graph can serve any
  // admitted B without changing pointers, strides or compact-arena bindings.
  const uint64_t control = static_cast<uint64_t>(std::max(slot.rows, 1U)) |
                           (static_cast<uint64_t>(suppress_kv_writes) << 32);
  adaptive.host_control.data_ptr<int64_t>()[0] = static_cast<int64_t>(control);
  copy_input(adaptive.control, adaptive.host_control, prepare_stream_);
}

Status TaskExecutionPipeline::warmup_packed_verify_graphs(
    const LlmForwardInput& input) {
  const auto host = SlotBuffer::model_input_view(input);
  const auto batch = model_batch(input);
  if (!batch.forward_type.is_decode() || host.q_seq_lens.empty()) {
    return invalid("Adaptive DSpark graph warmup requires decode requests.");
  }
  for (auto& task_slot : slots_) {
    const Status status = validate_input(*task_slot->speculative, input);
    if (!status.ok()) {
      return status;
    }
    const auto& slot = *task_slot->speculative;
    if (slot.verify->plan.graph_rows == 0 ||
        task_graph_batch_size(
            slot.drafts.front()->input->model_params().parallel,
            slot.drafts.front()->graph_batch_sizes) == 0) {
      return invalid(
          "Adaptive DSpark warmup exceeds Target/Draft graph capacity.");
    }
  }
  c10::DeviceGuard device_guard(device_.unwrap());
  auto guard = task_stream_.set_stream_guard();
  // Prepared metadata binds the Draft's KV caches, including its context
  // graph runner. Target warmup writes context before ordinary Draft execution.
  if (!block_draft()) {
    warmup_draft_graphs(input);
  }
  for (auto& task_slot : slots_) {
    auto& slot = *task_slot->speculative;
    if (!block_draft()) {
      bind_packed_verify_views(slot);
      continue;
    }
    auto& invocation = *slot.drafts.front();
    auto& binding = *invocation.input;
    const uint32_t draft_rows = task_graph_batch_size(
        binding.model_params().parallel, invocation.graph_batch_sizes);
    if (draft_rows == 0) {
      return invalid(
          "Adaptive DSpark requires a warmed Draft5/7 request bucket.");
    }
    auto conservative = host;
    conservative.positions = slot.upper_positions;
    conservative.kv_seq_lens = slot.upper_kv_lengths;
    const Status status = SlotBuffer::prepare_planned_mtp_decode(
        binding, conservative, batch, prepare_stream_, draft_rows);
    CHECK(status.ok()) << status.message();
    CHECK_EQ(prepare_stream_.synchronize(), 0);
    pad_task_parallel(binding.model_params().parallel, draft_rows);
    draft_->executor.prepare_attention_metadata(draft_->kv_caches,
                                                binding.model_params());
    draft_->executor.warmup_prepared_graph(binding.tokens(),
                                           binding.positions(),
                                           draft_->kv_caches,
                                           binding.model_params());
    const int32_t maximum_kv_length = *std::max_element(
        binding.model_params().attention.host.kv_seq_lens.begin(),
        binding.model_params().attention.host.kv_seq_lens.end());
    invocation.captured_graph_kv_capacities.try_emplace(
        draft_rows,
        dspark_graph_kv_capacity(capacity_,
                                 binding.input_scratch_->rows_per_sequence_,
                                 maximum_kv_length));
    auto& captured = invocation.captured_graph_batch_sizes;
    const auto bucket =
        std::lower_bound(captured.begin(), captured.end(), draft_rows);
    if (bucket == captured.end() || *bucket != draft_rows) {
      captured.insert(bucket, draft_rows);
    }
    invocation.padded_batch_size = draft_rows;
    bind_packed_verify_views(slot);
  }
  CHECK_EQ(task_stream_.synchronize(), 0);
  for (const int64_t graph_rows : graph_batch_sizes_) {
    if (std::binary_search(captured_graph_batch_sizes_.begin(),
                           captured_graph_batch_sizes_.end(),
                           graph_rows)) {
      continue;
    }
    for (auto& task_slot : slots_) {
      auto& slot = *task_slot->speculative;
      auto& binding = *slot.validate_input;
      const uint32_t physical_rows = static_cast<uint32_t>(graph_rows);
      const uint32_t original_rows = slot.rows;
      const VerifyBudgetPlan original_plan = slot.verify->plan;
      // The Device control selects a legal dummy batch even for G smaller than
      // the scheduler's synthetic warmup. These are the same full-capacity
      // sources used by serving; no bootstrap-only graph binding is created.
      slot.rows = std::min(original_rows, physical_rows);
      slot.verify->plan = {
          std::min(slot.rows * speculative_capacity_->num_speculative_tokens,
                   physical_rows - slot.rows),
          physical_rows};
      bind_packed_verify_views(slot);
      slot.verify->arena_storage.zero_();
      slot.state->token_storage_.fill_(1);
      slot.state->storage_.positions.zero_();
      slot.state->storage_.kv_seq_lens.fill_(1);
      (block_draft() ? slot.drafts.front() : slot.drafts.back())
          ->input->model_device_.block_tables.zero_();
      const uint32_t logical_rows = slot.rows + slot.verify->plan.draft_budget;
      const Status status = SlotBuffer::prepare_dspark_validate(
          binding,
          logical_rows,
          physical_rows,
          host.block_table_width,
          dspark_local_kv_bound(slot.upper_kv_lengths,
                                speculative_capacity_->num_speculative_tokens),
          batch.batch_id,
          prepare_stream_);
      CHECK(status.ok()) << status.message();
      bind_packed_verify_graph_input(slot, /*suppress_kv_writes=*/true);
      auto& raw_counts =
          binding.model_params().parallel.raw_dp_global_token_nums;
      if (capacity_.dp_size == 1) {
        raw_counts = {static_cast<int32_t>(logical_rows)};
      } else {
        raw_counts.clear();
        for (const int32_t count :
             input.input_params.parallel.dp_global_sequence_nums) {
          raw_counts.emplace_back(
              std::min(static_cast<uint32_t>(count) *
                           (speculative_capacity_->num_speculative_tokens + 1),
                       physical_rows));
        }
      }
      pad_task_parallel(binding.model_params().parallel, physical_rows);
      executor_.prepare_attention_metadata(kv_caches_, binding.model_params());
      CHECK_EQ(prepare_stream_.synchronize(), 0);
      executor_.warmup_prepared_graph(binding.tokens(),
                                      binding.positions(),
                                      kv_caches_,
                                      binding.model_params());
      slot.target_output = executor_.forward(binding.tokens(),
                                             binding.positions(),
                                             kv_caches_,
                                             binding.model_params());
      if (block_draft()) {
        auto context_slots = prefix(slot.context_slots_storage, physical_rows);
        context_slots.fill_(-1);
        auto& context_params = binding.model_params();
        context_params.meta.is_graph_warmup = true;
        const auto context_output =
            draft_->model.write_context_kv(slot.target_output.aux_hidden_states,
                                           binding.positions(),
                                           context_slots,
                                           draft_->kv_caches,
                                           context_params);
        context_params.meta.is_graph_warmup = false;
        CHECK(context_output.hidden_states.defined());
      }
      slot.rows = original_rows;
      slot.verify->plan = original_plan;
      bind_packed_verify_views(slot);
    }
    CHECK_EQ(task_stream_.synchronize(), 0);
    captured_graph_batch_sizes_.emplace_back(graph_rows);
  }
  std::sort(captured_graph_batch_sizes_.begin(),
            captured_graph_batch_sizes_.end());
  if (speculative_capacity_->unified_graph) {
    return warmup_adaptive_unified_graphs(input);
  }
  return Status();
}

std::vector<SpecCalibrationGraphCoverage>
TaskExecutionPipeline::adaptive_verify_graph_coverage() const {
  if (!speculative_capacity_->unified_graph) {
    return {};
  }
  const auto& captured_requests =
      slots_.front()->speculative->drafts.front()->captured_graph_batch_sizes;
  const uint32_t draft_width = speculative_capacity_->num_speculative_tokens;
  std::vector<SpecCalibrationGraphCoverage> coverage;
  coverage.reserve(captured_requests.size());
  for (const int64_t draft_rows : captured_requests) {
    const bool common_draft =
        std::all_of(slots_.begin(), slots_.end(), [&](const auto& task_slot) {
          const auto& captured = task_slot->speculative->drafts.front()
                                     ->captured_graph_batch_sizes;
          return std::binary_search(
              captured.begin(), captured.end(), draft_rows);
        });
    if (!common_draft) {
      continue;
    }
    CHECK_EQ(draft_rows % draft_width, 0);
    const uint32_t requests = static_cast<uint32_t>(draft_rows / draft_width);
    SpecCalibrationGraphCoverage entry;
    entry.requests = requests;
    entry.target_graph_rows.reserve(4);
    for (const int64_t rows : captured_graph_batch_sizes_) {
      const bool common =
          std::all_of(slots_.begin(), slots_.end(), [&](const auto& task_slot) {
            return task_slot->speculative->verify->unified_graph_bindings
                .contains(adaptive_unified_graph_key(
                    requests, static_cast<uint32_t>(rows)));
          });
      if (common) {
        entry.target_graph_rows.emplace_back(static_cast<uint32_t>(rows));
      }
    }
    coverage.emplace_back(std::move(entry));
  }
  return coverage;
}

Status TaskExecutionPipeline::warmup_adaptive_unified_graphs(
    const LlmForwardInput& input) {
  const auto host = SlotBuffer::model_input_view(input);
  const auto batch = model_batch(input);
  const uint32_t draft_width = speculative_capacity_->num_speculative_tokens;
  CHECK(input.input_params.meta.is_graph_warmup);
  CHECK(batch.forward_type.is_decode());
  c10::DeviceGuard device_guard(device_.unwrap());
  auto guard = task_stream_.set_stream_guard();
  uint64_t new_variants = 0;
  for (auto& task_slot : slots_) {
    auto& slot = *task_slot->speculative;
    auto& adaptive = *slot.verify;
    auto& invocation = *slot.drafts.front();
    auto& draft = *invocation.input;
    const uint32_t original_rows = slot.rows;
    const VerifyBudgetPlan original_plan = adaptive.plan;
    const uint32_t original_draft_rows = invocation.padded_batch_size;
    // ProfileManager sweeps the largest batch first. Enumerate every raw B,
    // including non-buckets such as three, before accepting serving Tasks.
    // The graph keeps actual-B head/result shapes and a fixed physical G.
    for (uint32_t rows = 1; rows <= original_rows; ++rows) {
      slot.rows = rows;
      const ModelInputHostView prefix_host{
          host.token_ids.first(rows),
          std::span<const int32_t>(slot.upper_positions).first(rows),
          host.new_cache_slots.first(rows),
          host.q_seq_lens.first(rows),
          std::span<const int32_t>(slot.upper_kv_lengths).first(rows),
          host.q_cu_seq_lens.first(rows),
          host.block_tables.first(static_cast<size_t>(rows) *
                                  host.block_table_width),
          host.block_table_width};
      const ModelInputBatch prefix_batch{
          BatchForwardType::DECODE, rows, batch.batch_id, false};
      Status status =
          SlotBuffer::plan_mtp_decode(draft, prefix_host, prefix_batch);
      CHECK(status.ok()) << status.message();
      auto& draft_parallel = draft.model_params().parallel;
      draft_parallel.raw_dp_global_token_nums = {
          static_cast<int32_t>(rows * draft_width)};
      draft_parallel.dp_global_token_nums =
          draft_parallel.raw_dp_global_token_nums;
      draft_parallel.dp_is_decode = {1};
      const uint32_t draft_rows =
          task_graph_batch_size(draft_parallel, invocation.graph_batch_sizes);
      CHECK_GT(draft_rows, 0U);
      status = SlotBuffer::prepare_planned_mtp_decode(
          draft, prefix_host, prefix_batch, prepare_stream_, draft_rows);
      CHECK(status.ok()) << status.message();
      pad_task_parallel(draft_parallel, draft_rows);
      CHECK_EQ(prepare_stream_.synchronize(), 0);
      // Cold prepared capture precedes the composite graph's write mask.
      // Borrow reserved page zero for reads and suppress every cache write.
      draft.model_params().attention.device.new_cache_slots.fill_(-1);
      draft.model_params().attention.device.block_tables.zero_();
      draft_->executor.prepare_attention_metadata(draft_->kv_caches,
                                                  draft.model_params());
      // Smaller physical Draft buckets may not yet have appeared in the
      // scheduler's descending warmup. Keep their prepared path available too.
      draft_->executor.warmup_prepared_graph(draft.tokens(),
                                             draft.positions(),
                                             draft_->kv_caches,
                                             draft.model_params());
      const int32_t maximum_kv_length = *std::max_element(
          draft.model_params().attention.host.kv_seq_lens.begin(),
          draft.model_params().attention.host.kv_seq_lens.end());
      invocation.captured_graph_kv_capacities.try_emplace(
          draft_rows,
          dspark_graph_kv_capacity(capacity_,
                                   draft.input_scratch_->rows_per_sequence_,
                                   maximum_kv_length));
      auto& captured = invocation.captured_graph_batch_sizes;
      const auto bucket =
          std::lower_bound(captured.begin(), captured.end(), draft_rows);
      if (bucket == captured.end() || *bucket != draft_rows) {
        captured.insert(bucket, draft_rows);
      }
      invocation.padded_batch_size = draft_rows;
      const uint32_t kv_bound = static_cast<uint32_t>(std::min(
          {invocation.captured_graph_kv_capacities.at(draft_rows) + 1,
           static_cast<uint64_t>(capacity_.max_kv_seq_len),
           static_cast<uint64_t>(capacity_.max_positions),
           static_cast<uint64_t>(
               draft.model_params().attention.device.block_tables.size(1)) *
               capacity_.logical_block_size}));
      status = slot.target_prefill->bind_result(rows,
                                                draft_width + 1,
                                                /*top_logprobs=*/0,
                                                /*logprobs=*/false);
      CHECK(status.ok()) << status.message();
      MtpContextView dummy_state{};
      dummy_state.prepared_ = true;
      dummy_state.tokens_ = prefix(slot.state->token_storage_, rows);
      dummy_state.state_.positions =
          prefix(slot.state->storage_.positions, rows);
      dummy_state.state_.kv_seq_lens =
          prefix(slot.state->storage_.kv_seq_lens, rows);
      // Cold sources borrow Slot storage, not central published request state.
      // Suppression is a Device value shared with serving's final binding.
      adaptive.arena_storage.zero_();
      dummy_state.tokens_.fill_(1);
      dummy_state.state_.positions.zero_();
      dummy_state.state_.kv_seq_lens.fill_(1);
      const auto graph_tiers = adaptive_unified_graph_rows(
          rows, captured_graph_batch_sizes_, draft_width);
      CHECK(!graph_tiers.empty());
      for (const int64_t captured_rows : graph_tiers) {
        const uint32_t graph_rows = static_cast<uint32_t>(captured_rows);
        const uint64_t key = adaptive_unified_graph_key(rows, graph_rows);
        if (adaptive.unified_graph_bindings.contains(key)) {
          continue;
        }
        adaptive.plan = {std::min(rows * draft_width, graph_rows - rows),
                         graph_rows};
        bind_packed_verify_views(slot);
        auto& target = *slot.validate_input;
        status = SlotBuffer::prepare_dspark_validate(
            target,
            rows + adaptive.plan.draft_budget,
            graph_rows,
            static_cast<uint32_t>(target.model_device_.block_tables.size(1)),
            static_cast<int32_t>(kv_bound),
            batch.batch_id,
            prepare_stream_);
        CHECK(status.ok()) << status.message();
        auto& parallel = target.model_params().parallel;
        parallel.raw_dp_global_token_nums = {
            static_cast<int32_t>(rows + adaptive.plan.draft_budget)};
        parallel.dp_global_token_nums = {static_cast<int32_t>(graph_rows)};
        parallel.dp_is_decode = {1};
        bind_packed_verify_graph_input(slot, /*suppress_kv_writes=*/true);
        executor_.prepare_attention_metadata(kv_caches_, target.model_params());
        CHECK_EQ(prepare_stream_.synchronize(), 0);
        SlotBuffer::patch_mtp_decode(draft, dummy_state);
        slot.context_slots.fill_(-1);
        const AdaptiveDSparkGraphTask task{target,
                                           draft,
                                           slot.target_prefill->device_result(),
                                           slot.context_slots};
        executor_.warmup_adaptive_unified_graph(
            draft_->executor, task, capacity_.sampling_group);
        // Replay returns the composite stream's dependency to task_stream_.
        // Finish its borrowed inputs before rewriting the next control/Slot.
        executor_.run_adaptive_unified_graph(
            draft_->executor, task, capacity_.sampling_group);
        CHECK_EQ(task_stream_.synchronize(), 0);
        adaptive.unified_graph_bindings.insert(key);
        ++new_variants;
      }
      // Repeated scheduler warmup may find every G already captured. Its
      // dummy writes still precede the next raw-B Prepare stream uploads.
      CHECK_EQ(task_stream_.synchronize(), 0);
    }
    slot.rows = original_rows;
    adaptive.plan = original_plan;
    invocation.padded_batch_size = original_draft_rows;
    bind_packed_verify_views(slot);
  }
  LOG(INFO) << "Adaptive DSpark unified graph startup: new_variants="
            << new_variants << ", slots=" << slots_.size()
            << ", max_batch=" << host.q_seq_lens.size() << ", maximum_variants="
            << static_cast<uint64_t>(slots_.size()) * host.q_seq_lens.size() *
                   4;
  return Status();
}

void TaskExecutionPipeline::warmup_draft_graphs(const LlmForwardInput& input) {
  const auto host = SlotBuffer::model_input_view(input);
  const auto batch = model_batch(input);
  for (auto& task_slot : slots_) {
    auto* slot = task_slot->speculative.get();
    const uint32_t forwards =
        draft_calls(*slot) + (slot->prefix_refresh ? 1U : 0U);
    for (uint32_t step = 0; step < forwards; ++step) {
      const bool refresh = step == draft_calls(*slot);
      auto& invocation = refresh ? *slot->prefix_refresh : *slot->drafts[step];
      const auto& parallel = invocation.input->model_params().parallel;
      const uint32_t padded_batch_size =
          task_graph_batch_size(parallel, invocation.graph_batch_sizes);
      auto& captured = invocation.captured_graph_batch_sizes;
      const auto bucket =
          std::lower_bound(captured.begin(), captured.end(), padded_batch_size);
      if (padded_batch_size == 0 ||
          (bucket != captured.end() && *bucket == padded_batch_size)) {
        continue;
      }
      auto& binding = *invocation.input;
      Status status;
      if (slot->rows == 0) {
        status = binding.prepare_decode_padded(
            {},
            {BatchForwardType::EMPTY, 0, batch.batch_id},
            padded_batch_size,
            prepare_stream_);
      } else {
        auto conservative = host;
        conservative.positions = slot->upper_positions;
        conservative.kv_seq_lens = slot->upper_kv_lengths;
        status = SlotBuffer::prepare_planned_mtp_decode(*invocation.input,
                                                        conservative,
                                                        batch,
                                                        prepare_stream_,
                                                        padded_batch_size);
      }
      CHECK(status.ok()) << status.message();
      if (refresh) {
        prepare_prefix_refresh_state(*slot, /*warmup=*/true);
      } else {
        prepare_draft_state(*slot, step, /*warmup=*/true);
      }
      CHECK_EQ(prepare_stream_.synchronize(), 0);
      pad_task_parallel(binding.model_params().parallel, padded_batch_size);
      draft_->executor.prepare_attention_metadata(draft_->kv_caches,
                                                  binding.model_params());
      draft_->executor.warmup_prepared_graph(binding.tokens(),
                                             binding.positions(),
                                             draft_->kv_caches,
                                             binding.model_params());
      CHECK_EQ(task_stream_.synchronize(), 0);
      captured.insert(bucket, padded_batch_size);
    }
  }
}

void TaskExecutionPipeline::prepare_draft_state(SpeculativeSlot& slot,
                                                uint32_t step,
                                                bool warmup) {
  auto& invocation = *slot.drafts[step];
  auto& binding = *invocation.input;
  auto& params = binding.model_params();
  params.mtp_topk_state.reset();
  if (step == 0 && !params.enable_graph) {
    params.embedding.input_embedding =
        slot.rows == 0
            ? slot.dummy_hidden
            : slot.state->state_.hidden.view(
                  {2 * slot.rows, speculative_capacity_->common.hidden_size});
    return;
  }
  const int64_t logical_rows =
      slot.rows * invocation.input->input_scratch_->rows_per_sequence_;
  const int64_t physical_rows = binding.tokens().numel();
  invocation.next_hidden = prefix(invocation.hidden_storage, logical_rows);
  params.embedding.input_embedding =
      prefix(invocation.hidden_storage, physical_rows);
  auto guard = prepare_stream_.set_stream_guard();
  // Launch supplies logical rows after their producer completes. Padding has
  // no producer, so initialize it here; capture warmup initializes every row.
  const int64_t clear_from = warmup ? 0 : logical_rows;
  params.embedding.input_embedding
      .narrow(/*dim=*/0, clear_from, physical_rows - clear_from)
      .zero_();
  if (step != 0 && speculative_capacity_->reuse_topk) {
    invocation.next_topk = prefix(invocation.topk_storage, slot.rows);
    auto topk = prefix(invocation.topk_storage, physical_rows);
    topk.narrow(/*dim=*/0, clear_from, physical_rows - clear_from).zero_();
    params.mtp_topk_state = MtpTopkState::from_tensor(std::move(topk));
  }
}

void TaskExecutionPipeline::prepare_prefix_refresh_state(SpeculativeSlot& slot,
                                                         bool warmup) {
  auto& refresh = *slot.prefix_refresh;
  auto& binding = *refresh.input;
  const int64_t logical = slot.rows * (slot.draft_steps + 1);
  const int64_t physical = binding.tokens().numel();
  auto& params = binding.model_params();
  params.embedding.input_embedding = prefix(refresh.hidden_storage, physical);
  params.mtp_topk_state.reset();
  auto guard = prepare_stream_.set_stream_guard();
  const int64_t clear_from = warmup ? 0 : logical;
  params.embedding.input_embedding.narrow(0, clear_from, physical - clear_from)
      .zero_();
}

Status TaskExecutionPipeline::prepare_speculative(
    uint32_t slot_id,
    const LlmForwardInput& input) {
  const auto host = SlotBuffer::model_input_view(input);
  const auto batch = model_batch(input);
  if (slot_id >= slots_.size()) {
    return invalid("Invalid MTP SpeculativeSlot index.");
  }
  SpeculativeSlot& slot = *slots_[slot_id]->speculative;
#if defined(USE_NPU) || defined(USE_MLU) || defined(USE_DCU)
  slot.target_prefill->model_params().parallel.layer_synchronizer.reset();
  slot.draft_prefill->model_params().parallel.layer_synchronizer.reset();
#endif
  slot.target_prefill->model_params().graph.dspark_adaptive = {};
  slot.validate_input->model_params().graph.dspark_adaptive = {};
  Status status = validate_input(slot, input);
  if (!status.ok()) {
    return status;
  }
  status = plan_sampling(slot, input);
  if (!status.ok()) {
    return status;
  }
  if (adaptive_verify_) {
    slot.verify->generation = next_task_id_;
  }
  c10::DeviceGuard guard(device_.unwrap());
  const int64_t embedding_rows = paged_cache_blocks(kv_caches_);
  if (embedding_rows == 0) {
    return Status(StatusCode::UNAVAILABLE,
                  "MTP embedding resources are not allocated.");
  }
  if (embedding_rows > std::numeric_limits<int32_t>::max()) {
    return invalid("Invalid allocated MTP embedding capacity.");
  }
  if (context_->request_ids_.empty()) {
    std::unique_ptr<MtpContextStorage> context;
    status = create_mtp_context(static_cast<uint32_t>(embedding_rows),
                                block_draft() ? 0 : capacity_.hidden_size,
                                hidden_dtype_,
                                device_.unwrap(),
                                context);
    if (!status.ok()) {
      return status;
    }
    reserve_draft_prefix_context(*context,
                                 static_cast<uint32_t>(embedding_rows));
    // Views retain the storage object's address, not its allocation contents.
    *context_ = std::move(*context);
    if (adaptive_verify_) {
      adaptive_verify_->histories.resize(embedding_rows);
    }
    if (speculative_capacity_->hybrid_recurrent_target) {
      auto stream_guard = prepare_stream_.set_stream_guard();
      recurrent_accepted_lengths_ =
          torch::ones({embedding_rows}, model_.options().dtype(torch::kInt32));
    }
    LOG(INFO) << "Task pipeline embedding context: rows=" << embedding_rows
              << ", device_bytes=" << mtp_context_bytes(*context_);
  }
  CHECK_EQ(context_->request_ids_.size(), embedding_rows);
  if (slot.decode && slot.rows != 0) {
    status = context_prepare_decode(
        *slot.state,
        input.input_params.embedding.embedding_ids,
        input.input_params.embedding.request_ids,
        input.input_params.embedding.mtp_bootstrap_row_idxes,
        prepare_stream_);
  } else {
    status = context_prepare_prefill(
        *slot.state,
        input.input_params.embedding.embedding_ids,
        input.input_params.embedding.request_ids,
        input.input_params.embedding.extra_token_ids,
        slot.rows == 0 ? std::span<const uint32_t>()
                       : slot.draft_prefill->input_scratch_->planned_rows_,
        prepare_stream_);
  }
  if (!status.ok()) {
    return status;
  }
  if (adaptive_verify_) {
    const auto& ids = input.input_params.embedding.embedding_ids;
    const auto& requests = input.input_params.embedding.request_ids;
    for (uint32_t row = 0; row < ids.size(); ++row) {
      auto& history = adaptive_verify_->histories[ids[row]];
      if (slot.decode && history.request_id == requests[row]) {
        continue;
      }
      history = AdaptiveVerifyState::History();
      history.request_id = requests[row];
      history.request_generation = next_task_id_;
    }
  }
  status = slot.target_prefill->prepare_sampling(
      input.sampling_params, slot.model_tokens, prepare_stream_);
  CHECK(status.ok()) << status.message();
  const uint32_t width = slot.decode ? slot.draft_steps + 1 : 1;
  status = slot.samples == 0
               ? slot.target_prefill->bind_result(0, 0, 0, false)
               : slot.target_prefill->bind_result(
                     slot.samples,
                     width,
                     input.sampling_params.logprobs()
                         ? input.sampling_params.max_top_logprobs()
                         : 0,
                     input.sampling_params.logprobs());
  CHECK(status.ok()) << status.message();
  slot.is_warmup = input.input_params.meta.is_graph_warmup;
  slot.state->reuse_draft_prefix_ =
      slot.state->reuse_draft_prefix_ &&
      speculative_capacity_->reuse_prefix_refresh_output && slot.decode &&
      !slot.is_warmup;
  slot.unified_graph = adaptive_verify_
                           ? speculative_capacity_->unified_graph
                           : supports_unified_block_graph(
                                 *speculative_capacity_, slot.rows, input);
  if (speculative_capacity_->common.dp_size > 1) {
    VLOG(1)
        << "MTP DP task batch=" << batch.batch_id
        << " dp_rank=" << speculative_capacity_->common.dp_rank
        << " local_tokens=" << slot.model_tokens << " local_rows=" << slot.rows
        << " samples=" << slot.samples << " run_models=" << slot.run_models
        << " speculative_decode=" << slot.decode
        << " padded_batch_size=" << slot.target_padded_batch_size
        << " local_decode=" << batch.forward_type.is_decode() << " base_counts="
        << slot.target_prefill->model_params().parallel.dp_global_token_nums
        << " first_draft_counts="
        << slot.drafts.front()
               ->input->model_params()
               .parallel.dp_global_token_nums
        << " validate_counts="
        << slot.validate_input->model_params().parallel.dp_global_token_nums;
  }
  slot.host_hidden =
      prefix(slot.host_hidden_storage, slot.decode ? 0 : slot.samples);
  if (slot.rows == 0) {
    if (slot.run_models) {
      prepare_empty_shard(slot, batch.batch_id);
    }
    bind_mtp_publication(slot);
    record(prepare_stream_, slots_[slot_id]->input_ready);
    return Status();
  }
  if (slot.decode) {
    if (capacity_.model.enable_linear_attention) {
      status = slot.target_prefill->prepare(host, batch, prepare_stream_);
      CHECK(status.ok()) << status.message();
      auto input_guard = prepare_stream_.set_stream_guard();
      slot.target_prefill->prepare_model_state(input);
    }
    prepare_bootstrap(slot, input);
    auto conservative = host;
    conservative.positions = slot.upper_positions;
    conservative.kv_seq_lens = slot.upper_kv_lengths;
    if (adaptive_verify_) {
      // Cost planning uses the global DP KV maximum, but this shard's
      // placeholders must fit its own reserved page table.
      status = SlotBuffer::prepare_dspark_validate(
          *slot.validate_input,
          slot.rows + slot.verify->plan.draft_budget,
          slot.verify->plan.graph_rows,
          host.block_table_width,
          dspark_local_kv_bound(slot.upper_kv_lengths,
                                speculative_capacity_->num_speculative_tokens),
          batch.batch_id,
          prepare_stream_);
    } else {
      status =
          SlotBuffer::prepare_planned_mtp_decode(*slot.validate_input,
                                                 conservative,
                                                 batch,
                                                 prepare_stream_,
                                                 slot.target_padded_batch_size);
    }
    CHECK(status.ok()) << status.message();
    if (capacity_.model.enable_linear_attention) {
      auto input_guard = prepare_stream_.set_stream_guard();
      slot.validate_input->prepare_model_state(input, width);
    }
    status = slot.validate_input->prepare_sampling(
        slot.host_validate_sampling,
        adaptive_verify_ ? slot.verify->plan.graph_rows : slot.rows * width,
        prepare_stream_);
    CHECK(status.ok()) << status.message();
    if (adaptive_verify_) {
      bind_packed_verify_views(slot);
    } else {
      slot.proposal = prefix(slot.proposal_storage, slot.rows * (width - 1))
                          .view({slot.rows, width - 1});
    }
    slot.accepted_mask = prefix(slot.accepted_mask_storage, slot.rows * width)
                             .view({slot.rows, width});
    if (block_draft()) {
      slot.context_slots = prefix(
          slot.context_slots_storage,
          adaptive_verify_ || speculative_capacity_->hybrid_recurrent_target ||
                  block_draft_graphs()
              ? slot.validate_input->tokens().numel()
              : slot.rows * width);
    }
    const auto& base = slot.target_prefill->sampling_params();
    for (uint32_t step = 0; step < draft_calls(slot); ++step) {
      auto& invocation = *slot.drafts[step];
      status =
          SlotBuffer::prepare_planned_mtp_decode(*invocation.input,
                                                 conservative,
                                                 batch,
                                                 prepare_stream_,
                                                 invocation.padded_batch_size);
      CHECK(status.ok()) << status.message();
      auto& binding = *invocation.input;
      if (block_draft()) {
        binding.model_params().embedding.input_embedding = torch::Tensor();
        if (block_draft_graphs() ||
            speculative_capacity_->hybrid_recurrent_target) {
          binding.model_params().enable_graph =
              capacity_.max_graph_batch_size != 0;
        }
      } else {
        prepare_draft_state(slot, step, /*warmup=*/false);
      }
      const uint32_t draft_rows = slot.rows * (block_draft() ? width - 1 : 1);
      SamplingParameters params = base;
      prepare_draft_sampling(speculative_capacity_->is_probabilistic_draft,
                             params);
      if (block_draft()) {
        auto stream_guard = prepare_stream_.set_stream_guard();
        if (speculative_capacity_->kind == SpeculativeTaskKind::DFLASH) {
          for (const auto& field : kSamplingFields) {
            if (!field.per_request) {
              repeat_interleave_if_defined(field.member(params), width - 1);
            }
          }
        }
        params.selected_token_idxes() =
            prefix(slot.draft_block_indices, draft_rows);
      } else if (step == 0) {
        params.selected_token_idxes() =
            prefix(slot.state->reuse_draft_prefix_ ? slot.draft_prefix_indices
                                                   : slot.draft_first_indices,
                   slot.rows);
      }
      SampleOutput draft_output;
      draft_output.next_tokens = prefix(invocation.token_storage, draft_rows);
      invocation.sampling = Sampling{std::move(params), draft_output, {}};

      pad_task_parallel(binding.model_params().parallel,
                        invocation.padded_batch_size);
      draft_->executor.prepare_attention_metadata(draft_->kv_caches,
                                                  binding.model_params());
    }
    if (slot.prefix_refresh) {
      auto& refresh = *slot.prefix_refresh;
      status =
          SlotBuffer::prepare_planned_mtp_decode(*refresh.input,
                                                 conservative,
                                                 batch,
                                                 prepare_stream_,
                                                 refresh.padded_batch_size);
      CHECK(status.ok()) << status.message();
      prepare_prefix_refresh_state(slot, /*warmup=*/false);
      pad_task_parallel(refresh.input->model_params().parallel,
                        refresh.padded_batch_size);
      draft_->executor.prepare_attention_metadata(
          draft_->kv_caches, refresh.input->model_params());
    }
    auto& binding = *slot.validate_input;

    pad_task_parallel(binding.model_params().parallel,
                      slot.target_padded_batch_size);
    if (adaptive_verify_) {
      bind_packed_verify_graph_input(slot,
                                     /*suppress_kv_writes=*/slot.is_warmup);
    }
    status = prepare_recurrent_state(slot, input);
    CHECK(status.ok()) << status.message();
    executor_.prepare_attention_metadata(kv_caches_, binding.model_params());
  } else {
    status = slot.target_prefill->prepare(host, batch, prepare_stream_);
    CHECK(status.ok()) << status.message();
    if (capacity_.model.enable_linear_attention) {
      auto input_guard = prepare_stream_.set_stream_guard();
      slot.target_prefill->prepare_model_state(input);
    }
    status = SlotBuffer::prepare_planned_mtp_prefill(
        *slot.draft_prefill, host, batch, prepare_stream_);
    CHECK(status.ok()) << status.message();
    slot.target_sampling =
        Sampling{slot.target_prefill->sampling_params(),
                 sample_views(slot.target_prefill->device_result()),
                 {}};
    auto& target_input = *slot.target_prefill;
    status = prepare_recurrent_state(slot, input);
    CHECK(status.ok()) << status.message();
    executor_.prepare_attention_metadata(kv_caches_,
                                         target_input.model_params());
    if (!block_draft()) {
      auto& draft_input = *slot.draft_prefill;
      draft_input.model_params().parallel =
          slot.target_prefill->model_params().parallel;
      draft_->executor.prepare_attention_metadata(draft_->kv_caches,
                                                  draft_input.model_params());
    }
  }
  if (speculative_capacity_->reuse_prefix_refresh_output) {
    const uint64_t generation =
        slot.decode && !slot.is_warmup ? next_task_id_ : 0;
    for (const int32_t id : input.input_params.embedding.embedding_ids) {
      context_->draft_prefix_generations_[id] = generation;
    }
  }
  bind_mtp_publication(slot);
  if (prepares_mtp_context() && slot.decode && !slot.is_warmup) {
    prepare_mtp_context(slot);
  }
  record(prepare_stream_, slots_[slot_id]->input_ready);
  return Status();
}

bool TaskExecutionPipeline::prepares_mtp_context() const {
  return speculative_capacity_->kind == SpeculativeTaskKind::MTP &&
         !speculative_capacity_->adaptive_target_verify &&
         !speculative_capacity_->adaptive_draft_step &&
         !speculative_capacity_->unified_graph &&
         capacity_.model.enable_linear_attention && !capacity_.enable_mla &&
         capacity_.dp_size == 1 && capacity_.cp_sampling_group == nullptr &&
         speculative_capacity_->num_speculative_tokens >= 1 &&
         speculative_capacity_->num_speculative_tokens <= 16;
}

void TaskExecutionPipeline::bind_mtp_publication(SpeculativeSlot& slot) {
  slot.context_prepared = false;
  slot.context_published = false;
  if (!prepares_mtp_context()) {
    return;
  }
  CHECK(slot.publication == nullptr);
  CHECK(slot.predecessor_publication == nullptr);
  const auto available = std::find_if(
      context_publications_.begin(),
      context_publications_.end(),
      [](const auto& publication) { return publication.use_count() == 1; });
  CHECK(available != context_publications_.end())
      << "MTP publication leases exceeded the bounded Slot ownership.";
  slot.publication = *available;
  slot.publication->task_id = next_task_id_;
  slot.publication->submitted = false;
  slot.predecessor_publication = last_context_publication_;
  last_context_publication_ = slot.publication;
}

void TaskExecutionPipeline::prepare_mtp_context(SpeculativeSlot& slot) {
  const auto& predecessor = slot.predecessor_publication;
  if (predecessor != nullptr) {
    // No Python/GIL or device synchronization is held here. Launch can submit
    // this acknowledgement before entering its independent prefix-refresh body.
    std::unique_lock<std::mutex> lock(predecessor->mutex);
    predecessor->submitted_cv.wait(
        lock, [&predecessor] { return predecessor->submitted; });
    CHECK_LT(predecessor->task_id, slot.publication->task_id);
  }
  auto guard = prepare_stream_.set_stream_guard();
  if (predecessor != nullptr) {
    CHECK(prepare_stream_.wait_event(predecessor->ready));
  }
  context_gather_state(*slot.state);
  apply_bootstrap(slot);
  for (uint32_t step = 0; step < slot.drafts.size(); ++step) {
    if (step == 0 && slot.state->reuse_draft_prefix_) {
      continue;
    }
    SlotBuffer::patch_mtp_decode_metadata(*slot.drafts[step]->input,
                                          *slot.state);
  }
  SlotBuffer::patch_mtp_decode_metadata(*slot.validate_input, *slot.state);
  prefix(slot.validate_input->model_params().num_accepted_tokens, slot.rows)
      .copy_(slot.state->state_.accepted_lengths);
  if (slot.prefix_refresh) {
    // Refresh positions use the old base, before Launch advances the context.
    SlotBuffer::patch_mtp_decode_metadata(*slot.prefix_refresh->input,
                                          *slot.state);
  }
  slot.context_prepared = true;
}

void TaskExecutionPipeline::publish_mtp_context(SpeculativeSlot& slot) {
  if (slot.publication == nullptr || slot.context_published) {
    return;
  }
  record(task_stream_, slot.publication->ready);
  {
    std::lock_guard<std::mutex> lock(slot.publication->mutex);
    slot.publication->submitted = true;
  }
  slot.context_published = true;
  slot.publication->submitted_cv.notify_all();
}

void TaskExecutionPipeline::restore_recurrent_state(
    const SpeculativeSlot& slot,
    const ModelParamsView& model_params) {
  if (!slot.recurrent) {
    return;
  }
  CHECK(restore_request_state_);
  restore_request_state_(model_params);
}

Status TaskExecutionPipeline::prepare_recurrent_state(
    SpeculativeSlot& slot,
    const LlmForwardInput& input) {
  if (!slot.recurrent) {
    return Status();
  }
  const auto& source = input.input_params;
  const auto& ids = source.embedding.linear_state_ids;
  if (ids.size() != slot.rows) {
    return invalid("Recurrent DSpark requires one state slot per request.");
  }
  const auto cache = std::find_if(
      kv_caches_.begin(), kv_caches_.end(), [](const KVCache& value) {
        return value.has_request_state();
      });
  CHECK(cache != kv_caches_.end());
  const int64_t state_capacity = cache->get_conv_cache().size(0);
  const bool warmup = source.meta.is_graph_warmup;
  if (std::any_of(ids.begin(), ids.end(), [state_capacity, warmup](int32_t id) {
        return id < (warmup ? 0 : 1) || id >= state_capacity;
      })) {
    return invalid("Recurrent DSpark state slot is outside allocated cache.");
  }
  if (slot.decode && !slot.verify) {
    // prepare_model_state already bound the complete fixed-width bucket,
    // including padding state zero. Do not shrink its graph inputs to live B.
    return Status();
  }
  auto& recurrent = *slot.recurrent;
  const int64_t capacity = recurrent.host_state_indices.size(0);
  if (capacity > static_cast<int64_t>(slot.rows)) {
    recurrent.host_state_indices.narrow(0, slot.rows, capacity - slot.rows)
        .zero_();
  }
  std::copy(
      ids.begin(), ids.end(), recurrent.host_state_indices.data_ptr<int32_t>());
  copy_input(
      recurrent.state_indices, recurrent.host_state_indices, prepare_stream_);
  auto& target = slot.decode ? slot.validate_input->model_params()
                             : slot.target_prefill->model_params();
  target.embedding.linear_state_ids = ids;
  target.embedding.linear_state_indices =
      prefix(recurrent.state_indices, slot.rows);
  target.embedding.linear_state_read_ids =
      source.embedding.linear_state_read_ids;
  target.embedding.linear_state_read_indices =
      source.embedding.linear_state_read_indices;
  target.linear_state_validity_mask = source.linear_state_validity_mask;
  if (warmup) {
    // Synthetic warmup owns padding slot zero, never a live reset/copy lease.
    target.embedding.linear_state_read_ids.clear();
    target.embedding.linear_state_read_indices = torch::Tensor();
    target.linear_state_validity_mask.assign(slot.rows, 0);
  } else if (target.linear_state_validity_mask.empty()) {
    const auto host = SlotBuffer::model_input_view(input);
    std::vector<int32_t> cached;
    cached.reserve(slot.rows);
    for (uint32_t row = 0; row < slot.rows; ++row) {
      cached.emplace_back(host.kv_seq_lens[row] - host.q_seq_lens[row]);
    }
    target.linear_state_validity_mask =
        build_linear_state_mask(cached, slot.rows);
  }
  const auto& read_ids = target.embedding.linear_state_read_ids;
  for (size_t row = 0; row < read_ids.size(); ++row) {
    if (read_ids[row] != target.embedding.linear_state_ids[row]) {
      target.linear_state_validity_mask[row] = 1;
    }
  }
  if (!slot.decode) {
    target.attn_metadata->has_initial_states = torch::tensor(
        target.linear_state_validity_mask,
        torch::TensorOptions().dtype(torch::kBool).device(device_.unwrap()));
  }
  return Status();
}

void TaskExecutionPipeline::prepare_empty_shard(SpeculativeSlot& slot,
                                                uint64_t batch_id) {
  // Physical rows participate in MoE collectives without a Sequence lease,
  // sampling or publication. Only reserved padding block zero is written.
  // Each zero-count peer executes one row,
  // including first Draft and Validate; only active peers expand by 2 or K+1.
  const auto prepare = [&](SlotBuffer& binding,
                           const ParallelInput& parallel,
                           TaskModel resource,
                           bool draft) {
    const uint32_t padded_batch_size =
        draft ? 0 : slot.target_padded_batch_size;
    const Status status = padded_batch_size == 0
                              ? binding.prepare_empty_shard(
                                    slot.decode, batch_id, prepare_stream_)
                              : binding.prepare_decode_padded(
                                    {},
                                    {BatchForwardType::EMPTY, 0, batch_id},
                                    padded_batch_size,
                                    prepare_stream_);
    CHECK(status.ok()) << status.message();
    binding.model_params().parallel = parallel;
    pad_task_parallel(binding.model_params().parallel, padded_batch_size);
    binding.model_params().mtp_topk_state.reset();
    binding.model_params().embedding.input_embedding =
        draft ? slot.dummy_hidden : torch::Tensor();
    resource.executor.prepare_attention_metadata(resource.kv_caches,
                                                 binding.model_params());
  };
  if (slot.decode) {
    if (block_draft()) {
      const std::array<int32_t, 1> zero{0};
      const std::array<int32_t, 1> one{1};
      const ModelInputHostView host{zero, zero, zero, one, one, one, zero, 1};
      const ModelInputBatch batch{BatchForwardType::DECODE, 1, batch_id};
      auto& invocation = *slot.drafts.front();
      Status status =
          SlotBuffer::plan_mtp_decode(*invocation.input, host, batch);
      CHECK(status.ok()) << status.message();
      status =
          SlotBuffer::prepare_planned_mtp_decode(*invocation.input,
                                                 host,
                                                 batch,
                                                 prepare_stream_,
                                                 invocation.padded_batch_size);
      CHECK(status.ok()) << status.message();
      auto& binding = *invocation.input;
      binding.model_params().meta.actual_num_sequences = 0;
      binding.model_params().enable_graph = capacity_.max_graph_batch_size != 0;
      pad_task_parallel(binding.model_params().parallel,
                        invocation.padded_batch_size);
      draft_->executor.prepare_attention_metadata(draft_->kv_caches,
                                                  binding.model_params());
    } else {
      for (uint32_t step = 0; step < draft_calls(slot); ++step) {
        auto& invocation = *slot.drafts[step];
        auto& binding = *invocation.input;
        const Status status =
            invocation.padded_batch_size == 0
                ? binding.prepare_empty_shard(
                      /*decode=*/true, batch_id, prepare_stream_)
                : binding.prepare_decode_padded(
                      {},
                      {BatchForwardType::EMPTY, 0, batch_id},
                      invocation.padded_batch_size,
                      prepare_stream_);
        CHECK(status.ok()) << status.message();
        prepare_draft_state(slot, step, /*warmup=*/false);

        pad_task_parallel(binding.model_params().parallel,
                          invocation.padded_batch_size);
        draft_->executor.prepare_attention_metadata(draft_->kv_caches,
                                                    binding.model_params());
      }
    }
    if (adaptive_verify_) {
      auto& binding = *slot.validate_input;
      const Status status = SlotBuffer::prepare_dspark_validate(
          binding,
          1,
          slot.target_padded_batch_size,
          capacity_.model.max_blocks_per_sequence,
          static_cast<int32_t>(slot.verify->max_kv_length),
          batch_id,
          prepare_stream_);
      CHECK(status.ok()) << status.message();
      binding.model_params().meta.actual_num_sequences = 0;
      bind_packed_verify_views(slot);
      auto stream_guard = prepare_stream_.set_stream_guard();
      slot.verify->arena_storage.zero_();
      slot.verify->offsets_storage.zero_();
      slot.verify->offsets_storage.narrow(0, 1, 1).fill_(1);
      slot.state->token_storage_.fill_(1);
      slot.state->storage_.positions.zero_();
      slot.state->storage_.kv_seq_lens.fill_(1);
      bind_packed_verify_graph_input(slot, /*suppress_kv_writes=*/true);
      executor_.prepare_attention_metadata(kv_caches_, binding.model_params());
    } else {
      prepare(*slot.validate_input,
              slot.validate_input->model_params().parallel,
              TaskModel{model_, executor_, kv_caches_},
              /*draft=*/false);
    }
    if (slot.prefix_refresh) {
      auto& refresh = *slot.prefix_refresh;
      auto& binding = *refresh.input;
      const auto status =
          refresh.padded_batch_size == 0
              ? binding.prepare_empty_shard(true, batch_id, prepare_stream_)
              : binding.prepare_decode_padded(
                    {},
                    {BatchForwardType::EMPTY, 0, batch_id},
                    refresh.padded_batch_size,
                    prepare_stream_);
      CHECK(status.ok()) << status.message();
      prepare_prefix_refresh_state(slot, /*warmup=*/false);
      pad_task_parallel(binding.model_params().parallel,
                        refresh.padded_batch_size);
      draft_->executor.prepare_attention_metadata(draft_->kv_caches,
                                                  binding.model_params());
    }
  } else {
    prepare(*slot.target_prefill,
            slot.target_prefill->model_params().parallel,
            TaskModel{model_, executor_, kv_caches_},
            /*draft=*/false);
    if (!block_draft()) {
      prepare(*slot.draft_prefill,
              slot.target_prefill->model_params().parallel,
              *draft_,
              /*draft=*/true);
    }
  }
}

void TaskExecutionPipeline::launch_empty_shard(SpeculativeSlot& slot) {
  const auto run = [](SlotBuffer& binding, TaskModel resource) {
    return resource.executor.forward(binding.tokens(),
                                     binding.positions(),
                                     resource.kv_caches,
                                     binding.model_params());
  };
  if (slot.decode) {
    for (auto& invocation : std::span(slot.drafts).first(draft_calls(slot))) {
      invocation->output = run(*invocation->input, *draft_);
    }
    slot.target_output =
        run(*slot.validate_input, TaskModel{model_, executor_, kv_caches_});
    if (slot.prefix_refresh) {
      launch_prefix_refresh(slot);
    }
  } else {
    slot.target_output =
        run(*slot.target_prefill, TaskModel{model_, executor_, kv_caches_});
    if (block_draft()) {
      write_block_context(slot, /*prefill=*/true);
      return;
    }
    slot.drafts.front()->output = run(*slot.draft_prefill, *draft_);
  }
}

void TaskExecutionPipeline::launch_prefill(SpeculativeSlot& slot) {
  auto& target_input = *slot.target_prefill;
  restore_recurrent_state(slot, target_input.model_params());
  slot.target_output = executor_.forward(target_input.tokens(),
                                         target_input.positions(),
                                         kv_caches_,
                                         target_input.model_params());
  if (slot.samples != 0) {
    slot.target_prefill->refresh_min_tokens();
    slot.target_logits = model_.logits(
        slot.target_output.hidden_states,
        slot.target_prefill->sampling_params().selected_token_idxes());
    sample(*slot.target_sampling, slot.target_logits);
    slot.target_prefill->device_result().lengths.fill_(1);
    if (block_draft() || capacity_.cp_sampling_group != nullptr) {
      synchronize_samples(
          slot.target_prefill->device_result(),
          slot.target_prefill->sampling_params().sampling_mode());
    } else {
      auto tokens = slot.target_prefill->device_result().tokens;
      synchronize_tokens(
          tokens,
          slot.target_prefill->sampling_params().sampling_mode() ==
              SamplingMode::GREEDY);
    }
  }
  if (slot.recurrent) {
    recurrent_accepted_lengths_.index_fill_(0, slot.state->rows_, 1);
  }
  if (block_draft()) {
    SlotBuffer::initialize_mtp_context(
        *slot.draft_prefill,
        slot.target_output.aux_hidden_states,
        slot.target_prefill->device_result().tokens,
        *slot.state);
    write_block_context(slot, /*prefill=*/true);
    context_publish(*slot.state);
    return;
  }
  SlotBuffer::patch_mtp_prefill(*slot.draft_prefill,
                                slot.target_output.hidden_states,
                                slot.target_prefill->device_result().tokens,
                                *slot.state);
  auto& invocation = *slot.drafts.front();
  auto& draft_input = *slot.draft_prefill;
  invocation.output = draft_->executor.forward(draft_input.tokens(),
                                               draft_input.positions(),
                                               draft_->kv_caches,
                                               draft_input.model_params());
  context_publish(*slot.state);
}

void TaskExecutionPipeline::launch_prefix_refresh(SpeculativeSlot& slot) {
  auto& refresh = *slot.prefix_refresh;
  auto& binding = *refresh.input;
  refresh.output = draft_->executor.forward(binding.tokens(),
                                            binding.positions(),
                                            draft_->kv_caches,
                                            binding.model_params());
}

void TaskExecutionPipeline::launch_decode(SpeculativeSlot& slot) {
  if (slot.context_prepared) {
    // Prefix output is published after the previous refresh, on this ordered
    // stream. Its base metadata was already snapshotted by Prepare.
    context_gather_prefix_hidden(*slot.state);
  } else {
    context_gather(*slot.state);
    apply_bootstrap(slot);
  }
  // The Slot may never have run Prefill. Read the published request position,
  // after its producing Task, rather than stale or unbound prefill positions.
  slot.target_prefill->refresh_min_tokens(slot.state->state_.positions);
  if (adaptive_verify_ && block_draft()) {
    launch_packed_verify(slot);
    return;
  }
  if (launch_unified_block(slot)) {
    return;
  }
  const uint32_t width = slot.draft_steps + 1;
  if (block_draft()) {
    launch_block_draft(slot);
  } else {
    for (uint32_t step = 0; step < draft_calls(slot); ++step) {
      auto& invocation = *slot.drafts[step];
      const bool reuse_prefix = step == 0 && slot.state->reuse_draft_prefix_;
      if (reuse_prefix) {
        invocation.output = ModelOutput(slot.state->draft_prefix_hidden_);
      } else {
        if (slot.context_prepared) {
          SlotBuffer::patch_mtp_decode_tokens(*invocation.input, *slot.state);
        } else {
          SlotBuffer::patch_mtp_decode(*invocation.input, *slot.state);
        }
        auto& binding = *invocation.input;
        if (step == 0 && binding.model_params().enable_graph) {
          invocation.next_hidden.copy_(slot.state->state_.hidden.view(
              {2 * slot.rows, speculative_capacity_->common.hidden_size}));
        }
        if (step != 0) {
          const auto& previous = *slot.drafts[step - 1];
          prefix(binding.tokens(), slot.rows)
              .copy_(previous.sampled.next_tokens);
          const int64_t previous_rows =
              step == 1 && !slot.state->reuse_draft_prefix_ ? 2 : 1;
          invocation.next_hidden.copy_(
              previous.output.hidden_states
                  .narrow(/*dim=*/0, /*start=*/0, slot.rows * previous_rows)
                  .view({slot.rows,
                         previous_rows,
                         speculative_capacity_->common.hidden_size})
                  .select(/*dim=*/1, previous_rows - 1));
          if (speculative_capacity_->reuse_topk) {
            CHECK(previous.output.mtp_topk_state != nullptr)
                << "GLM Draft did not publish its configured TopK state.";
            const auto topk = previous.output.mtp_topk_state->as_tensor();
            CHECK(topk.has_value())
                << "Prepared draft requires tensor-backed TopK state.";
            invocation.next_topk.copy_(
                topk->narrow(/*dim=*/0, /*start=*/0, slot.rows * previous_rows)
                    .view({slot.rows,
                           previous_rows,
                           1,
                           speculative_capacity_->index_topk})
                    .select(/*dim=*/1, previous_rows - 1));
          }
        }
        invocation.output = draft_->executor.forward(binding.tokens(),
                                                     binding.positions(),
                                                     draft_->kv_caches,
                                                     binding.model_params());
      }
      invocation.logits = draft_->model.logits(
          invocation.output.hidden_states,
          invocation.sampling->params.selected_token_idxes());
      invocation.sampled = sample(*invocation.sampling, invocation.logits);
      synchronize_tokens(
          invocation.sampled.next_tokens,
          invocation.sampling->params.sampling_mode() == SamplingMode::GREEDY &&
              capacity_.cp_sampling_group == nullptr);
      slot.proposal.select(/*dim=*/1, step)
          .copy_(invocation.sampled.next_tokens);
      if (adaptive_verify_) {
        // Confidence is a proxy for acceptance, not a target probability.
        // Work on Device; only normal result retirement returns this feedback.
        const auto logits = invocation.logits.to(torch::kFloat32);
        const auto selected = logits.gather(
            /*dim=*/1, invocation.sampled.next_tokens.view({slot.rows, 1}));
        slot.verify->confidence.select(/*dim=*/1, step)
            .copy_((selected.view({slot.rows}) -
                    torch::logsumexp(logits, /*dim=*/{-1}))
                       .exp());
      }
    }
  }
  if (adaptive_verify_) {
    launch_packed_verify(slot);
    return;
  }
  if (slot.context_prepared) {
    SlotBuffer::patch_mtp_decode_tokens(*slot.validate_input, *slot.state);
  } else {
    SlotBuffer::patch_mtp_decode(*slot.validate_input, *slot.state);
  }
  auto& validate = *slot.validate_input;
  if (capacity_.model.enable_linear_attention && !slot.context_prepared) {
    // The verifier commits the previous round's true accepted checkpoint.
    // Keep the captured Slot address stable and update only its contents.
    prefix(validate.model_params().num_accepted_tokens, slot.rows)
        .copy_(slot.state->state_.accepted_lengths);
  }
  restore_recurrent_state(slot, validate.model_params());
  validate.tokens()
      .narrow(/*dim=*/0, /*start=*/0, slot.rows * width)
      .view({slot.rows, width})
      .narrow(/*dim=*/1, /*start=*/1, width - 1)
      .copy_(slot.proposal);
  slot.target_output = executor_.forward(validate.tokens(),
                                         validate.positions(),
                                         kv_caches_,
                                         validate.model_params());
  slot.target_output.hidden_states = slot.target_output.hidden_states.narrow(
      /*dim=*/0, /*start=*/0, slot.rows * width);
  if (block_draft()) {
    if (!block_draft_graphs() &&
        !speculative_capacity_->hybrid_recurrent_target) {
      slot.target_output.aux_hidden_states =
          slot.target_output.aux_hidden_states.narrow(
              /*dim=*/0, /*start=*/0, slot.rows * width);
    }
  }
  slot.target_logits = model_.logits(
      slot.target_output.hidden_states,
      slot.validate_input->sampling_params().selected_token_idxes());
  finish_fixed_verify(slot);
}

void TaskExecutionPipeline::finish_fixed_verify(SpeculativeSlot& slot) {
  const uint32_t width = slot.draft_steps + 1;
  const auto& base = slot.target_prefill->sampling_params();
  // AR drafts merge their per-step logits into the proposal's
  // [batch, n_spec, vocab] buffer; block drafts already wrote one during
  // drafting (prepare_draft_sampling decided logits per mode). Greedy AR
  // drafts carry tokens only; the other legs wrap the slot's preallocated
  // buffer.
  DraftProposal proposal = DraftProposal(slot.proposal);
  if (block_draft()) {
    proposal =
        DraftProposal(slot.proposal, slot.drafts.front()->sampled.draft_logits);
  } else if (speculative_capacity_->is_probabilistic_draft &&
             base.sampling_mode() != SamplingMode::GREEDY) {
    const torch::Tensor& first_step_logits =
        slot.drafts.front()->sampled.draft_logits;
    proposal = DraftProposal::from_token_buffer(slot.proposal,
                                                first_step_logits.size(-1),
                                                /*is_probabilistic_draft=*/true,
                                                first_step_logits.options());
    int64_t step = 0;
    for (const auto& invocation :
         std::span(slot.drafts).first(draft_calls(slot))) {
      proposal.logits_at(step).copy_(invocation->sampled.draft_logits);
      ++step;
    }
  }
  SampleOutput accepted = rejection_sample(base, proposal, slot.target_logits);
  const auto& result = slot.target_prefill->device_result();
  result.tokens.copy_(accepted.next_tokens);
  if (result.logprobs.defined()) {
    result.logprobs.copy_(accepted.logprobs);
  }
  if (result.top_tokens.defined()) {
    result.top_tokens.copy_(accepted.top_tokens);
    result.top_logprobs.copy_(accepted.top_logprobs);
  }
  if (block_draft() || capacity_.cp_sampling_group != nullptr) {
    synchronize_samples(result, base.sampling_mode());
  } else {
    auto tokens = result.tokens;
    synchronize_tokens(tokens, base.sampling_mode() == SamplingMode::GREEDY);
  }
  torch::ge_out(slot.accepted_mask, result.tokens, /*other=*/0);
  auto lengths = result.lengths;
  torch::sum_out(
      lengths, slot.accepted_mask, {1}, /*keepdim=*/false, torch::kInt32);
  if (slot.recurrent) {
    recurrent_accepted_lengths_.index_copy_(
        0, slot.state->rows_, result.lengths);
  }
  if (block_draft()) {
    write_block_context(slot, /*prefill=*/false);
  }
  const auto& context = block_draft() ? slot.target_output.aux_hidden_states
                                      : slot.target_output.hidden_states;
  if (slot.prefix_refresh) {
    auto& refresh = *slot.prefix_refresh;
    // Snapshot before another body can reuse the verifier's output pool.
    if (!slot.context_prepared) {
      SlotBuffer::patch_mtp_decode(*refresh.input, *slot.state);
    }
    prefix(refresh.hidden_storage, slot.rows * width)
        .copy_(slot.target_output.hidden_states);
    auto tokens = refresh.input->tokens().narrow(0, 0, slot.rows * width);
    tokens.copy_(result.tokens.reshape({-1})).clamp_min_(0);
  }
  context_advance(
      *slot.state,
      result.tokens,
      result.lengths,
      block_draft() ? torch::Tensor()
                    : context.view({slot.rows, width, context_hidden_size()}));
  context_publish(*slot.state);
  publish_mtp_context(slot);
  if (slot.prefix_refresh) {
    launch_prefix_refresh(slot);
    if (speculative_capacity_->reuse_prefix_refresh_output && !slot.is_warmup) {
      auto& state = *slot.state;
      const auto refreshed =
          slot.prefix_refresh->output.hidden_states
              .narrow(/*dim=*/0, /*start=*/0, slot.rows * width)
              .view({slot.rows, width, capacity_.hidden_size});
      auto selected = state.draft_prefix_hidden_.unsqueeze(/*dim=*/1);
      torch::gather_out(selected,
                        refreshed,
                        /*dim=*/1,
                        state.last_hidden_indices_);
      // Snapshot the borrowed output before any following model can reuse it.
      state.context_->draft_prefix_hidden_.index_copy_(
          /*dim=*/0, state.rows_, state.draft_prefix_hidden_);
    }
  }
}

bool TaskExecutionPipeline::launch_unified_block(SpeculativeSlot& slot) {
  if (!slot.unified_graph) {
    return false;
  }
  auto& draft = *slot.drafts.front()->input;
  auto& target = *slot.validate_input;
  SlotBuffer::patch_mtp_decode(draft, *slot.state);
  SlotBuffer::patch_mtp_decode(target, *slot.state);
  const auto& result = slot.target_prefill->device_result();
  slot.target_output.aux_hidden_states =
      executor_.python_impl()->run_unified_block(
          *draft_->executor.python_impl(),
          target,
          draft,
          result,
          speculative_capacity_->kind == SpeculativeTaskKind::DSPARK,
          capacity_.sampling_group);
  const uint32_t width = speculative_capacity_->num_speculative_tokens + 1;
  context_advance(*slot.state,
                  result.tokens,
                  result.lengths,
                  slot.target_output.aux_hidden_states.view(
                      {slot.rows, width, context_hidden_size()}));
  context_publish(*slot.state);
  COUNTER_INC(speculative_unified_graph_executions_total);
  return true;
}

void TaskExecutionPipeline::launch_block_draft(SpeculativeSlot& slot) {
  auto& invocation = *slot.drafts.front();
  SlotBuffer::patch_mtp_decode(*invocation.input, *slot.state);
  auto& input = *invocation.input;
  const int64_t steps = speculative_capacity_->num_speculative_tokens;
  const auto& selected = invocation.sampling->params.selected_token_idxes();
  SamplingParameters selector_sampling = slot.target_prefill->sampling_params();
  prepare_draft_sampling(speculative_capacity_->is_probabilistic_draft,
                         selector_sampling);
  if (speculative_capacity_->kind == SpeculativeTaskKind::DFLASH2) {
    slot.block_gumbel =
        broadcast_gumbel_noise(slot.rows,
                               selector_sampling,
                               steps,
                               speculative_capacity_->selector_top_k,
                               device_.unwrap(),
                               capacity_.sampling_group);
  }
  invocation.output = draft_->executor.forward(input.tokens(),
                                               input.positions(),
                                               draft_->kv_caches,
                                               input.model_params());
  invocation.logits =
      draft_->model.logits(invocation.output.hidden_states, selected);
  if (speculative_capacity_->kind == SpeculativeTaskKind::DFLASH2) {
    auto hidden = invocation.output.hidden_states.index_select(
        /*dim=*/0, selected.to(torch::kInt64));
    const auto candidates = draft_->model.dflash2_candidates(
        hidden.view(
            {slot.rows, steps, speculative_capacity_->common.hidden_size}),
        invocation.logits.view(
            {slot.rows, steps, speculative_capacity_->common.vocab_size}),
        slot.state->tokens_);
    invocation.sampled.draft_logits =
        sample_dflash2_path(candidates,
                            selector_sampling,
                            slot.block_gumbel,
                            speculative_capacity_->common.vocab_size,
                            slot.proposal)
            .logits()
            .value_or(torch::Tensor());
  } else if (speculative_capacity_->kind == SpeculativeTaskKind::DSPARK) {
    if (adaptive_verify_) {
      generate_adaptive_dspark_proposals(slot, slot.state->tokens_);
      return;
    }
    const auto base_logits = invocation.logits.view(
        {slot.rows, steps, speculative_capacity_->common.vocab_size});
    invocation.sampled.draft_logits =
        sample_dspark_markov_block(
            [this](const torch::Tensor& tokens) {
              return draft_->model.dspark_markov_bias(tokens);
            },
            base_logits,
            slot.state->tokens_,
            selector_sampling,
            [this](torch::Tensor& tokens) {
              synchronize_tokens(tokens, /*skip_sampling_group=*/false);
            },
            speculative_capacity_->is_probabilistic_draft,
            /*token_ids_out=*/slot.proposal)
            .logits()
            .value_or(torch::Tensor());
  } else {
    invocation.sampled = sample(*invocation.sampling, invocation.logits);
    slot.proposal.copy_(
        invocation.sampled.next_tokens.view({slot.rows, steps}));
  }
}

void TaskExecutionPipeline::generate_adaptive_dspark_proposals(
    SpeculativeSlot& slot,
    const torch::Tensor& anchors) {
  auto& invocation = *slot.drafts.front();
  auto& adaptive = *slot.verify;
  const uint32_t draft_width = speculative_capacity_->num_speculative_tokens;
  // Admission checked every penalty is neutral and the draft is greedy. Use
  // the same minimal sampler in serving and startup profiling; do not launch
  // no-op penalty kernels merely because the client supplied neutral tensors.
  SamplingParameters sampling;
  const auto base_logits =
      invocation.logits.view({slot.rows, draft_width, capacity_.vocab_size});
  sample_dspark_markov_block(
      [this](const torch::Tensor& tokens) {
        return draft_->model.dspark_markov_bias(tokens);
      },
      base_logits,
      anchors,
      sampling,
      [](torch::Tensor& /*sampled_tokens*/) {},
      /*is_probabilistic_draft=*/false,
      /*token_ids_out=*/slot.proposal);
  invocation.sampled.draft_logits = torch::Tensor();
  adaptive.previous_tokens.select(/*dim=*/1, /*index=*/0).copy_(anchors);
  adaptive.previous_tokens
      .narrow(/*dim=*/1, /*start=*/1, /*length=*/draft_width - 1)
      .copy_(slot.proposal.narrow(
          /*dim=*/1, /*start=*/0, /*length=*/draft_width - 1));
  const auto confidence = draft_->model.dspark_confidence_probs(
      invocation.output.hidden_states
          .narrow(/*dim=*/0, /*start=*/0, slot.rows * draft_width)
          .view({slot.rows, draft_width, capacity_.hidden_size}),
      adaptive.previous_tokens);
  CHECK(confidence.defined());
  CHECK(confidence.sizes() == adaptive.confidence.sizes());
  CHECK_EQ(confidence.device(), device_.unwrap());
  CHECK_EQ(confidence.scalar_type(), torch::kFloat32);
  // Snapshot before write_context_kv can overwrite the borrowed draft output.
  // ConfidenceHead consumes local hidden states, unlike full-vocabulary logits
  // gathered across TP ranks. Keep leader-authoritative proposals and
  // confidence until cross-rank bitwise parity is verified, even for greedy
  // sampling.
  adaptive.confidence.copy_(confidence);
  sync_sampling_tensor(adaptive.consensus, capacity_.sampling_group);
  sync_sampling_tensor(adaptive.consensus, capacity_.cp_sampling_group);
}

void TaskExecutionPipeline::launch_packed_verify(SpeculativeSlot& slot) {
#if defined(USE_NPU)
  if (slot.unified_graph) {
    auto& draft = *slot.drafts.front()->input;
    SlotBuffer::patch_mtp_decode(draft, *slot.state);
    const AdaptiveDSparkGraphTask task{*slot.validate_input,
                                       draft,
                                       slot.target_prefill->device_result(),
                                       slot.context_slots};
    slot.target_output.aux_hidden_states = executor_.run_adaptive_unified_graph(
        draft_->executor, task, capacity_.sampling_group);
    CHECK(slot.target_output.aux_hidden_states.defined());
    CHECK_EQ(slot.target_output.aux_hidden_states.size(0),
             slot.verify->plan.graph_rows);
    CHECK_EQ(slot.target_output.aux_hidden_states.size(1),
             context_hidden_size());
    const auto& result = slot.target_prefill->device_result();
    // The composite graph has already committed ragged results and written
    // accepted context in physical G order. Only request state remains here.
    context_advance(
        *slot.state, result.tokens, result.lengths, torch::Tensor());
    context_publish(*slot.state);
    COUNTER_INC(speculative_unified_graph_executions_total);
    return;
  }
  if (block_draft()) {
    launch_block_draft(slot);
  }
  auto& adaptive = *slot.verify;
  if (slot.recurrent) {
    prefix(slot.recurrent->accepted_lengths, slot.rows)
        .copy_(recurrent_accepted_lengths_.index_select(0, slot.state->rows_));
  }
  auto& validate = *slot.validate_input;
  restore_recurrent_state(slot, validate.model_params());
  auto& attention = validate.model_params().attention.device;
  // Proposal/confidence consensus is complete. The prepared Target graph owns
  // Device allocation, packing and attention metadata refresh in that order.
  slot.target_output = executor_.forward(validate.tokens(),
                                         validate.positions(),
                                         kv_caches_,
                                         validate.model_params());
  slot.target_logits =
      model_.logits(slot.target_output.hidden_states,
                    validate.sampling_params().selected_token_idxes());
  if (!adaptive_verify_) {
    // Full-width packed rows are request-major, exactly the fixed verifier's
    // sampling order. Keep its penalties, random sampling and logprobs intact.
    finish_fixed_verify(slot);
    return;
  }
  const auto& sampling = slot.target_prefill->sampling_params();
  if (sampling.sampling_mode() == SamplingMode::GREEDY &&
      !sampling.min_token_ids().defined()) {
    torch::argmax_out(slot.target_tokens,
                      slot.target_logits,
                      /*dim=*/-1,
                      /*keepdim=*/false);
  } else {
    slot.target_tokens.copy_(spec_verify::rejection_sample_ragged(
        slot.target_logits, sampling, adaptive.owners, adaptive.columns));
  }
  // A sampled target must agree across TP ranks before prefix acceptance and
  // context-KV writes, not merely synchronize the final returned tokens.
  synchronize_tokens(
      slot.target_tokens,
      sampling.sampling_mode() == SamplingMode::GREEDY &&
          (speculative_capacity_->kind == SpeculativeTaskKind::DSPARK ||
           capacity_.cp_sampling_group == nullptr));
  const auto& result = slot.target_prefill->device_result();
  auto result_tokens = result.tokens;
  auto result_lengths = result.lengths;
  kernel::npu::tilelang::dspark_greedy_verify_commit(slot.proposal,
                                                     slot.target_tokens,
                                                     adaptive.widths,
                                                     adaptive.offsets,
                                                     adaptive.owners,
                                                     adaptive.columns,
                                                     attention.new_cache_slots,
                                                     result_tokens,
                                                     result_lengths,
                                                     slot.context_slots);
  if (slot.recurrent) {
    recurrent_accepted_lengths_.index_copy_(
        0, slot.state->rows_, result.lengths.to(torch::kInt32));
  }
  if (block_draft()) {
    write_block_context(slot, /*prefill=*/false);
    context_advance(
        *slot.state, result.tokens, result.lengths, torch::Tensor());
  } else {
    // Packed Target rows cannot be viewed as a dense [B, width] sequence.
    // Recover each request's prefix using Device offsets, clamping unused
    // columns to its last verified row. Refresh's existing storage owns this
    // snapshot before another prepared forward can reuse Target output.
    auto& refresh = *slot.prefix_refresh;
    const int64_t width = speculative_capacity_->num_speculative_tokens + 1;
    const auto columns =
        torch::minimum(adaptive.hidden_columns,
                       adaptive.widths.to(torch::kInt64).unsqueeze(1) - 1);
    const auto indices =
        columns + prefix(adaptive.offsets, slot.rows).unsqueeze(1);
    auto hidden = prefix(refresh.hidden_storage, slot.rows * width);
    torch::index_select_out(hidden,
                            slot.target_output.hidden_states,
                            /*dim=*/0,
                            indices.reshape({-1}));
    SlotBuffer::patch_mtp_decode(*refresh.input, *slot.state);
    auto tokens = prefix(refresh.input->tokens(), slot.rows * width);
    tokens.copy_(result.tokens.reshape({-1})).clamp_min_(0);
    context_advance(*slot.state,
                    result.tokens,
                    result.lengths,
                    hidden.view({slot.rows, width, context_hidden_size()}));
    // All draft steps executed: retain the fixed generated-width repair
    // condition. A pruned bonus lies inside the already generated draft KV;
    // treating every fully accepted short prefix as full width would also
    // enable an invalid predecessor repair when zero drafts were selected.
  }
  context_publish(*slot.state);
  if (!block_draft()) {
    launch_prefix_refresh(slot);
  }
#else
  LOG(FATAL) << "Adaptive DSpark task pipeline requires NPU kernels.";
#endif
}

void TaskExecutionPipeline::synchronize_tokens(torch::Tensor& tokens,
                                               bool skip_sampling_group) {
  if (!skip_sampling_group) {
    sync_sampling_tensor(tokens, capacity_.sampling_group);
  }
  sync_sampling_tensor(tokens, capacity_.cp_sampling_group);
}

void TaskExecutionPipeline::synchronize_samples(
    const TokenResultTensors& result,
    SamplingMode sampling_mode) {
  // Greedy DSpark's commit kernel already broadcast the tokens.
  const bool already_broadcast =
      speculative_capacity_ != nullptr &&
      speculative_capacity_->kind == SpeculativeTaskKind::DSPARK &&
      sampling_mode == SamplingMode::GREEDY;
  for (const auto* tensor : {&result.tokens,
                             &result.logprobs,
                             &result.top_tokens,
                             &result.top_logprobs}) {
    if (tensor->defined()) {
      auto view = *tensor;
      synchronize_tokens(view, already_broadcast);
    }
  }
}

void TaskExecutionPipeline::write_block_context(SpeculativeSlot& slot,
                                                bool prefill) {
  auto& input = prefill ? *slot.target_prefill : *slot.validate_input;
  const auto& hidden = slot.target_output.aux_hidden_states;
  CHECK(hidden.defined())
      << "Block draft requires captured target auxiliary hidden states.";
  CHECK_EQ(hidden.size(/*dim=*/1), context_hidden_size());
  torch::Tensor cache_slots =
      input.model_params().attention.device.new_cache_slots;
  torch::Tensor positions = input.positions();
  if (!prefill && !adaptive_verify_) {
    const int64_t rows =
        slot.rows * (speculative_capacity_->num_speculative_tokens + 1);
    slot.context_slots.copy_(cache_slots.narrow(
        /*dim=*/0, /*start=*/0, slot.context_slots.numel()));
    slot.context_slots.narrow(/*dim=*/0, /*start=*/0, rows)
        .view(slot.accepted_mask.sizes())
        .masked_fill_(slot.accepted_mask.logical_not(), /*value=*/-1);
    slot.context_slots
        .narrow(/*dim=*/0, rows, slot.context_slots.numel() - rows)
        .fill_(/*value=*/-1);
    cache_slots = slot.context_slots;
    positions = positions.narrow(
        /*dim=*/0, /*start=*/0, slot.context_slots.numel());
  } else if (!prefill) {
    // Ragged commit already produced the G-row accepted-context mapping.
    // Keep borrowed target hidden states in physical order; never scatter them
    // to a dense Bx8 tensor or reconstruct owners on the Host.
    cache_slots = slot.context_slots;
  }
  // The serial task stream orders accepted-context publication before any
  // subsequent block reads the draft cache. Rejected rows disable the scatter;
  // no accepted length or token is copied to Host to build this operation.
  slot.drafts.front()->output = draft_->model.write_context_kv(
      hidden, positions, cache_slots, draft_->kv_caches, input.model_params());
  CHECK(slot.drafts.front()->output.hidden_states.defined());
}

void TaskExecutionPipeline::launch_speculative(uint32_t slot_id) {
  CHECK_LT(slot_id, slots_.size());
  SpeculativeSlot& slot = *slots_[slot_id]->speculative;
  c10::DeviceGuard device_guard(device_.unwrap());
  auto guard = task_stream_.set_stream_guard();
  CHECK(task_stream_.wait_event(slots_[slot_id]->input_ready));
  if (capacity_.model.enable_linear_attention && !slot.recurrent &&
      slot.rows != 0) {
    // launch() has already waited for Host KV reloads, including Conv/SSM.
    // Reset/restore and all recurrent forwards share this ordered stream.
    auto& params = slot.target_prefill->model_params();
    restore_linear_state_slots(kv_caches_,
                               params.embedding.linear_state_ids,
                               params.embedding.linear_state_read_ids,
                               params.linear_state_validity_mask,
                               /*reads_distinct_state=*/false);
  }
  KVTransferCompletion kv_transfers;
  const auto& transfers = slots_[slot_id]->transfer_kv_infos;
  if (!transfers.empty()) {
    CHECK(!slot.decode && !block_draft() && slot.rows != 0);
    kv_transfers.add(push_kv_(transfers, slot.target_prefill->model_params()));
    kv_transfers.add(
        draft_->push_kv(transfers, slot.draft_prefill->model_params()));
  }
  if (slot.rows == 0 && slot.run_models) {
    launch_empty_shard(slot);
  } else if (slot.rows != 0) {
    if (slot.decode) {
      launch_decode(slot);
    } else {
      launch_prefill(slot);
    }
  }
  // Target and Draft have independent layer events and cache namespaces.
  // Both transfers must finish before publishing tokens/bootstrap hidden.
  CHECK(kv_transfers.wait()) << "Task pipeline MTP KV cache push failed";
  // Prefill, empty and unchanged legacy paths still publish a generation for
  // a following Prepare. Decode normally records it before prefix refresh.
  publish_mtp_context(slot);
  record(task_stream_, slots_[slot_id]->output_ready);
  if (!slot.decode && slot.samples != 0) {
    CHECK(result_stream_.wait_event(slots_[slot_id]->output_ready));
    auto result_guard = result_stream_.set_stream_guard();
    const auto& hidden = slot.draft_prefill->input_scratch_->bootstrap_hidden_;
    CHECK_EQ(hidden.nbytes(), slot.host_hidden.nbytes());
    slot.host_hidden.copy_(hidden, /*non_blocking=*/true);
  }
  if (adaptive_verify_ && slot.decode && slot.rows != 0) {
    CHECK(result_stream_.wait_event(slots_[slot_id]->output_ready));
    auto result_guard = result_stream_.set_stream_guard();
    slot.verify->host_feedback.copy_(slot.verify->feedback,
                                     /*non_blocking=*/true);
  }
  const Status status = slot.target_prefill->copy_result_to_host(
      result_stream_, slots_[slot_id]->output_ready);
  CHECK(status.ok()) << status.message();
}

ForwardOutput TaskExecutionPipeline::consume_speculative(uint32_t slot_id) {
  CHECK_LT(slot_id, slots_.size());
  SpeculativeSlot& slot = *slots_[slot_id]->speculative;
  c10::DeviceGuard guard(device_.unwrap());
  auto tokens = slot.target_prefill->take_result();
  uint64_t completed_ns = 0;
  const bool record_calibration =
      adaptive_verify_ && adaptive_verify_->calibration_session;
  if (record_calibration) {
    completed_ns = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch())
            .count());
  }
  if (adaptive_verify_) {
    if (slot.decode && slot.rows != 0) {
      consume_adaptive_verify(slot);
    }
    adaptive_verify_->retired_generation = slot.verify->generation;
  }
  const bool prefill = !slot.decode;
  // tokens.lengths counts committed tokens (accepted + bonus); both consumers
  // want draft-only counts.
  std::vector<int32_t> num_accepted_tokens;
  if (!prefill && (record_calibration || tokens.tokens.defined())) {
    num_accepted_tokens = tensor_to_vector<int32_t>(tokens.lengths);
    for (int32_t& num_accepted : num_accepted_tokens) {
      CHECK_GE(num_accepted, 1);
      --num_accepted;
    }
  }
  if (record_calibration) {
    SpecSteadyStep step;
    step.generation = slot.verify->generation;
    step.completed_ns = completed_ns;
    if (!slot.is_warmup && slot.decode && slot.rows != 0) {
      step.logical_requests = slot.rows;
      step.draft_requests =
          slot.drafts.front()->padded_batch_size /
          (block_draft() ? speculative_capacity_->num_speculative_tokens : 2);
      step.target_graph_rows = slot.verify->plan.graph_rows;
      step.max_kv_length = slot.verify->max_kv_length;
      step.draft_tokens = slot.verify->selected_draft_tokens;
      step.request_ids = slot.verify->request_ids;
      for (const int32_t num_accepted : num_accepted_tokens) {
        step.accepted_tokens += static_cast<uint32_t>(num_accepted);
      }
    }
    if (adaptive_verify_->calibration_session) {
      adaptive_verify_->calibration_session->observe(step);
    }
  }
  ForwardOutput output;
  output.is_graph_warmup = slot.is_warmup;
  if (adaptive_verify_ && slot.decode && slot.rows != 0 &&
      capacity_.dp_size > 1 && !slot.is_warmup) {
    const uint32_t width = speculative_capacity_->num_speculative_tokens;
    const float* confidence =
        slot.verify->host_confidence.const_data_ptr<float>();
    output.dspark_feedback.confidence.assign(confidence,
                                             confidence + slot.rows * width);
    output.dspark_feedback.rounds_since_full_width.reserve(slot.rows);
    for (const int32_t id : slot.verify->embedding_ids) {
      output.dspark_feedback.rounds_since_full_width.emplace_back(
          adaptive_verify_->histories[id].versions[0].rounds_since_full_width);
    }
  }
  if (tokens.tokens.defined()) {
    output.sample_output.next_tokens =
        prefill ? tokens.tokens.squeeze(1) : tokens.tokens;
    if (!prefill) {
      output.spec_verify_stats = make_verify_stats(
          tokens.tokens,
          adaptive_verify_ ? &slot.verify->draft_counts : nullptr,
          num_accepted_tokens,
          {});
    }
    if (tokens.logprobs.defined()) {
      output.logprobs = true;
      output.sample_output.logprobs =
          prefill ? tokens.logprobs.squeeze(1) : tokens.logprobs;
    }
    if (tokens.top_tokens.defined()) {
      output.max_top_logprobs = tokens.top_tokens.size(-1);
      output.sample_output.top_tokens =
          prefill ? tokens.top_tokens.squeeze(1) : tokens.top_tokens;
      output.sample_output.top_logprobs =
          prefill ? tokens.top_logprobs.squeeze(1) : tokens.top_logprobs;
    }
  }
  if (prefill && slot.samples != 0) {
    output.sample_output.embeddings =
        torch::empty(slot.host_hidden.sizes(),
                     slot.host_hidden.options().pinned_memory(false));
    output.sample_output.embeddings.copy_(slot.host_hidden);
  }
  release_outputs(slot);
  slots_[slot_id]->transfer_kv_infos.clear();
  return output;
}

void TaskExecutionPipeline::consume_adaptive_verify(SpeculativeSlot& slot) {
  auto& adaptive = *slot.verify;
  const uint32_t draft_width = speculative_capacity_->num_speculative_tokens;
  const int32_t* widths = adaptive.host_widths.const_data_ptr<int32_t>();
  const float* confidence = adaptive.host_confidence.const_data_ptr<float>();
  adaptive.draft_counts.clear();
  uint32_t selected = 0;
  for (uint32_t row = 0; row < slot.rows; ++row) {
    CHECK_GE(widths[row], 1);
    CHECK_LE(widths[row], draft_width + 1);
    adaptive.draft_counts.emplace_back(widths[row] - 1);
    selected += static_cast<uint32_t>(widths[row] - 1);
  }
  CHECK_EQ(selected, adaptive.plan.draft_budget);
  if (slot.is_warmup) {
    return;
  }
  const bool full_width = adaptive.plan.draft_budget == slot.rows * draft_width;
  for (uint32_t row = 0; row < slot.rows; ++row) {
    const int32_t id = adaptive.embedding_ids[row];
    CHECK_GT(id, 0);
    CHECK_LT(static_cast<uint64_t>(id), adaptive_verify_->histories.size());
    auto& history = adaptive_verify_->histories[id];
    // A newly prepared Prefill may already reuse this embedding row while
    // an older Slot retires. Identity and generation both guard its feedback.
    if (context_->request_ids_[id] != adaptive.request_ids[row] ||
        history.request_id != adaptive.request_ids[row] ||
        adaptive.generation < history.request_generation ||
        adaptive.generation <= history.versions[0].generation) {
      continue;
    }
    AdaptiveVerifyState::Feedback feedback;
    feedback.generation = adaptive.generation;
    feedback.rounds_since_full_width =
        full_width ? 0 : history.versions[0].rounds_since_full_width + 1;
    for (uint32_t column = 0; column < draft_width; ++column) {
      const float probability = confidence[row * draft_width + column];
      CHECK(std::isfinite(probability));
      CHECK_GE(probability, 0.0F);
      CHECK_LE(probability, 1.0F);
      feedback.confidence[column] = probability;
    }
    history.versions[1] = history.versions[0];
    history.versions[0] = feedback;
  }
  auto& retired = adaptive_verify_->retired_plans;
  retired[1] = std::move(retired[0]);
  retired[0].generation = adaptive.generation;
  retired[0].embedding_ids = adaptive.embedding_ids;
  retired[0].request_ids = adaptive.request_ids;
  retired[0].plan = adaptive.plan;
}

void TaskExecutionPipeline::discard_speculative(uint32_t slot_id) {
  CHECK_LT(slot_id, slots_.size());
  SpeculativeSlot& slot = *slots_[slot_id]->speculative;
  c10::DeviceGuard guard(device_.unwrap());
  slot.target_prefill->discard_result();
  if (adaptive_verify_ && adaptive_verify_->calibration_session) {
    adaptive_verify_->calibration_session->observe(SpecSteadyStep{});
  }
  release_outputs(slot);
  slots_[slot_id]->transfer_kv_infos.clear();
}

void TaskExecutionPipeline::release_outputs(SpeculativeSlot& slot) {
#if defined(USE_NPU) || defined(USE_MLU) || defined(USE_DCU)
  slot.target_prefill->model_params().parallel.layer_synchronizer.reset();
  slot.draft_prefill->model_params().parallel.layer_synchronizer.reset();
#endif
  context_release(*slot.state);
  slot.publication.reset();
  slot.predecessor_publication.reset();
  slot.context_prepared = false;
  slot.draft_prefill->model_params().embedding.input_embedding =
      torch::Tensor();
  slot.target_output = ModelOutput();
  slot.target_logits = torch::Tensor();
  slot.target_sampling.reset();
  for (auto& invocation : slot.drafts) {
    invocation->output = ModelOutput();
    invocation->logits = torch::Tensor();
    invocation->sampled = SampleOutput();
    invocation->sampling.reset();
  }
  if (slot.prefix_refresh) {
    slot.prefix_refresh->output = ModelOutput();
  }
  slot.host_validate_sampling = SamplingParameters();
  slot.block_gumbel = torch::Tensor();
}

uint64_t TaskExecutionPipeline::speculative_pinned_bytes() const {
  uint64_t bytes = 0;
  for (const auto& task_slot : slots_) {
    const auto& slot = *task_slot->speculative;
    bytes +=
        slot.target_prefill->pinned_bytes() +
        slot.draft_prefill->pinned_bytes() +
        slot.validate_input->pinned_bytes() + slot.state->host_rows_.nbytes() +
        slot.host_hidden_storage.nbytes() + slot.bootstrap.host_rows.nbytes() +
        slot.bootstrap.host_tokens.nbytes() +
        state_bytes(slot.bootstrap.host_state);
    for (const auto& invocation : slot.drafts) {
      bytes += invocation->input->pinned_bytes();
    }
    if (adaptive_verify_) {
      bytes += slot.verify->host_feedback_storage.nbytes();
    }
    if (slot.recurrent) {
      bytes += slot.recurrent->host_state_indices.nbytes();
    }
    if (slot.prefix_refresh) {
      bytes += slot.prefix_refresh->input->pinned_bytes();
    }
  }
  return bytes;
}

uint64_t TaskExecutionPipeline::speculative_device_bytes() const {
  uint64_t bytes = mtp_context_bytes(*context_);
  if (recurrent_accepted_lengths_.defined()) {
    bytes += recurrent_accepted_lengths_.nbytes();
  }
  for (const auto& task_slot : slots_) {
    const auto& slot = *task_slot->speculative;
    bytes +=
        slot.target_prefill->device_bytes() +
        slot.draft_prefill->device_bytes() +
        slot.validate_input->device_bytes() +
        (slot.target_token_storage.defined()
             ? slot.target_token_storage.nbytes()
             : 0) +
        (slot.proposal_storage.defined() ? slot.proposal_storage.nbytes() : 0) +
        slot.accepted_mask_storage.nbytes() +
        context_device_bytes(*slot.state) +
        slot.bootstrap.row_storage.nbytes() +
        slot.bootstrap.token_storage.nbytes() +
        state_bytes(slot.bootstrap.storage);
    for (const auto& invocation : slot.drafts) {
      bytes += invocation->input->device_bytes();
      bytes += invocation->token_storage.nbytes();
      for (const auto* tensor :
           {&invocation->hidden_storage, &invocation->topk_storage}) {
        bytes += tensor->defined() ? tensor->nbytes() : 0;
      }
    }
    if (slot.prefix_refresh) {
      bytes += slot.prefix_refresh->input->device_bytes() +
               slot.prefix_refresh->hidden_storage.nbytes();
    }
    if (slot.recurrent) {
      bytes += slot.recurrent->state_indices.nbytes() +
               slot.recurrent->accepted_lengths.nbytes();
    }
    for (const auto* tensor : {&slot.context_slots_storage,
                               &slot.draft_block_indices,
                               &slot.draft_first_indices,
                               &slot.draft_prefix_indices,
                               &slot.dummy_hidden}) {
      bytes += tensor->defined() ? tensor->nbytes() : 0;
    }
    if (adaptive_verify_) {
      for (const auto* tensor : {&slot.verify->arena_storage,
                                 &slot.verify->previous_storage,
                                 &slot.verify->offsets_storage,
                                 &slot.verify->owners_storage,
                                 &slot.verify->columns_storage,
                                 &slot.verify->hidden_columns}) {
        bytes += tensor->defined() ? tensor->nbytes() : 0;
      }
    }
  }
  return bytes;
}

const SampleOutput& TaskExecutionPipeline::sample(Sampling& sampling,
                                                  torch::Tensor& logits) {
  if (sampling.params.num_sample_seqs() == 0) {
    return sampling.output;
  }
  CHECK_EQ(logits.dim(), 2);
  CHECK_EQ(logits.size(/*dim=*/1), speculative_capacity_->common.vocab_size);
  CHECK_EQ(logits.size(/*dim=*/0), sampling.params.num_sample_seqs());
  sampling.sampled = ::xllm::sample(logits, sampling.params);
  sampling.output.next_tokens.copy_(sampling.sampled.next_tokens);
  if (sampling.output.logprobs.defined()) {
    sampling.output.logprobs.copy_(sampling.sampled.logprobs);
  }
  if (sampling.output.top_tokens.defined()) {
    sampling.output.top_tokens.copy_(sampling.sampled.top_tokens);
    sampling.output.top_logprobs.copy_(sampling.sampled.top_logprobs);
  }
  sampling.output.draft_logits = sampling.sampled.draft_logits;
  return sampling.output;
}

namespace {
constexpr std::array<torch::Tensor MtpContextTensors::*, 6> kContextFields = {
    &MtpContextTensors::previous_tokens,
    &MtpContextTensors::hidden,
    &MtpContextTensors::positions,
    &MtpContextTensors::kv_seq_lens,
    &MtpContextTensors::accepted_lengths,
    &MtpContextTensors::repair_required};
}  // namespace

Status TaskExecutionPipeline::create_mtp_context(
    uint32_t capacity,
    uint32_t hidden_size,
    torch::ScalarType dtype,
    const torch::Device& device,
    std::unique_ptr<MtpContextStorage>& output) {
  const uint64_t element_bytes =
      dtype == torch::kFloat32 ? sizeof(float) : sizeof(uint16_t);
  if (capacity > std::numeric_limits<int32_t>::max() ||
      device.type() != Platform::type_torch() || !device.has_index() ||
      hidden_size > std::numeric_limits<int32_t>::max() ||
      (dtype != torch::kFloat16 && dtype != torch::kBFloat16 &&
       dtype != torch::kFloat32) ||
      static_cast<uint64_t>(capacity) * hidden_size >
          static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) /
              (2 * element_bytes)) {
    return invalid("Invalid MTP embedding context capacity or dtype.");
  }
  auto context = std::make_unique<MtpContextStorage>();
  context->request_ids_.resize(capacity);
  context->published_.resize(capacity);
  c10::DeviceGuard guard(device);
  const auto options = torch::TensorOptions().device(device);
  context->tokens_ = torch::empty({capacity}, options.dtype(torch::kInt64));
  if (hidden_size != 0) {
    context->mtp_state_.previous_tokens =
        torch::empty({capacity}, options.dtype(torch::kInt64));
    context->mtp_state_.hidden =
        torch::empty({capacity, 2, hidden_size}, options.dtype(dtype));
    context->mtp_state_.repair_required =
        torch::empty({capacity}, options.dtype(torch::kBool));
  }
  context->mtp_state_.positions =
      torch::empty({capacity}, options.dtype(torch::kInt32));
  context->mtp_state_.kv_seq_lens =
      torch::empty({capacity}, options.dtype(torch::kInt32));
  context->mtp_state_.accepted_lengths =
      torch::ones({capacity}, options.dtype(torch::kInt32));
  output = std::move(context);
  return Status();
}

uint64_t TaskExecutionPipeline::mtp_context_bytes(
    const MtpContextStorage& cache) {
  return cache.tokens_.nbytes() + state_bytes(cache.mtp_state_) +
         (cache.draft_prefix_hidden_.defined()
              ? cache.draft_prefix_hidden_.nbytes()
              : 0);
}

void TaskExecutionPipeline::reserve_draft_prefix_context(
    MtpContextStorage& context,
    uint32_t capacity) {
  if (!speculative_capacity_->reuse_prefix_refresh_output) {
    return;
  }
  context.draft_prefix_generations_.resize(capacity);
  context.draft_prefix_hidden_ =
      torch::empty({capacity, capacity_.hidden_size}, model_.options());
}

// Block drafts keep their hidden context in KV. Their cross-round state only
// needs the next token and its position; MTP additionally keeps two hidden
// rows.
Status TaskExecutionPipeline::context_create(
    MtpContextStorage& context,
    uint32_t capacity,
    std::unique_ptr<MtpContextView>& output) {
  if (!context.tokens_.defined() || !context.mtp_state_.positions.defined() ||
      !context.mtp_state_.kv_seq_lens.defined() || capacity == 0 ||
      capacity > std::numeric_limits<int32_t>::max()) {
    return invalid("Speculative context requires a fixed Slot capacity.");
  }
  auto storage = std::make_unique<MtpContextView>();
  auto& view = *storage;
  view.context_ = &context;
  view.capacity_ = capacity;
  view.sorted_ids_.reserve(capacity);

  c10::DeviceGuard guard(context.tokens_.device());
  const auto options = context.tokens_.options();
  view.host_rows_ =
      torch::empty({capacity}, options.device(torch::kCPU).pinned_memory(true));
  view.device_rows_ = torch::empty({capacity}, options);
  view.token_storage_ = torch::empty({capacity}, options);
  if (context.draft_prefix_hidden_.defined()) {
    view.draft_prefix_hidden_storage_ =
        torch::empty({capacity, context.draft_prefix_hidden_.size(/*dim=*/1)},
                     context.draft_prefix_hidden_.options());
  }
  view.storage_.positions =
      torch::empty({capacity}, options.dtype(torch::kInt32));
  view.storage_.kv_seq_lens =
      torch::empty({capacity}, options.dtype(torch::kInt32));
  view.storage_.accepted_lengths =
      torch::ones({capacity}, options.dtype(torch::kInt32));
  const bool history = context.mtp_state_.hidden.defined();
  // Both algorithms gather the last accepted token. Only MTP also gathers
  // its predecessor and tracks full-acceptance repair.
  view.index_storage_ = torch::empty({history ? 2 : 1, capacity}, options);
  if (history) {
    view.storage_.previous_tokens = torch::empty({capacity}, options);
    view.storage_.hidden =
        torch::empty({capacity, 2, context.mtp_state_.hidden.size(/*dim=*/2)},
                     context.mtp_state_.hidden.options());
    view.storage_.repair_required =
        torch::empty({capacity}, options.dtype(torch::kBool));
    view.mask_storage_ = torch::empty({capacity}, options.dtype(torch::kBool));
  }
  view.identity_rows_.reserve(capacity);
  view.model_to_embedding_.resize(capacity);
  view.publication_mask_.resize(capacity);
  view.bootstrap_mask_.resize(capacity);
  output = std::move(storage);
  return Status();
}

Status TaskExecutionPipeline::context_prepare(
    MtpContextView& view,
    std::span<const int32_t> embedding_ids,
    std::span<const std::string> request_ids,
    bool read_published_state,
    const Stream& stream) {
  if (embedding_ids.size() > view.capacity_) {
    return Status(StatusCode::RESOURCE_EXHAUSTED,
                  "MTP binding exceeds fixed Slot capacity.");
  }
  view.identity_rows_.clear();
  for (uint32_t row = 0; row < embedding_ids.size(); ++row) {
    view.identity_rows_.push_back(row);
  }
  return context_prepare_impl(view,
                              embedding_ids,
                              request_ids,
                              view.identity_rows_,
                              read_published_state,
                              stream,
                              {});
}

Status TaskExecutionPipeline::context_prepare_prefill(
    MtpContextView& view,
    std::span<const int32_t> embedding_ids,
    std::span<const std::string> request_ids,
    std::span<const int32_t> extra_token_ids,
    std::span<const uint32_t> published_sequence_rows,
    const Stream& stream) {
  if (extra_token_ids.size() > view.capacity_) {
    return Status(StatusCode::RESOURCE_EXHAUSTED,
                  "MTP prefill exceeds fixed Slot capacity.");
  }
  // ForwardInputBuilder emits embedding ids only for completed chunks, in
  // model order. Publication follows sampling order, which may differ.
  uint32_t completed = 0;
  for (uint32_t row = 0; row < extra_token_ids.size(); ++row) {
    view.model_to_embedding_[row] =
        extra_token_ids[row] == -1 ? static_cast<int32_t>(completed++) : -1;
  }
  if (embedding_ids.size() != completed ||
      published_sequence_rows.size() != completed) {
    return Status(StatusCode::INVALID_ARGUMENT,
                  "MTP prefill requires one embedding id per completed row.");
  }
  view.identity_rows_.clear();
  for (const uint32_t row : published_sequence_rows) {
    if (row >= extra_token_ids.size() || view.model_to_embedding_[row] < 0) {
      return Status(StatusCode::INVALID_ARGUMENT,
                    "MTP prefill publication requires completed model rows.");
    }
    view.identity_rows_.push_back(view.model_to_embedding_[row]);
  }
  return context_prepare_impl(view,
                              embedding_ids,
                              request_ids,
                              view.identity_rows_,
                              /*read_published_state=*/false,
                              stream,
                              {});
}

Status TaskExecutionPipeline::context_prepare_decode(
    MtpContextView& view,
    std::span<const int32_t> embedding_ids,
    std::span<const std::string> request_ids,
    std::span<const int32_t> bootstrap_rows,
    const Stream& stream) {
  if (embedding_ids.size() > view.capacity_) {
    return Status(StatusCode::RESOURCE_EXHAUSTED,
                  "MTP decode exceeds fixed Slot capacity.");
  }
  view.identity_rows_.clear();
  for (uint32_t row = 0; row < embedding_ids.size(); ++row) {
    view.identity_rows_.push_back(row);
  }
  return context_prepare_impl(view,
                              embedding_ids,
                              request_ids,
                              view.identity_rows_,
                              /*read_published_state=*/true,
                              stream,
                              bootstrap_rows);
}

Status TaskExecutionPipeline::context_prepare_impl(
    MtpContextView& view,
    std::span<const int32_t> embedding_ids,
    std::span<const std::string> request_ids,
    std::span<const uint32_t> published_sequence_rows,
    bool read_published_state,
    const Stream& stream,
    std::span<const int32_t> bootstrap_rows) {
  if (view.prepared_ || embedding_ids.size() > view.capacity_) {
    return Status(StatusCode::RESOURCE_EXHAUSTED,
                  "MTP binding is occupied or exceeds fixed Slot capacity.");
  }
  if (embedding_ids.size() != request_ids.size()) {
    return Status(StatusCode::INVALID_ARGUMENT,
                  "MTP embedding and request ids must be row aligned.");
  }
  view.sorted_ids_.assign(embedding_ids.begin(), embedding_ids.end());
  std::sort(view.sorted_ids_.begin(), view.sorted_ids_.end());
  if (!view.sorted_ids_.empty() &&
      (view.sorted_ids_.front() <= 0 ||
       static_cast<uint64_t>(view.sorted_ids_.back()) >=
           view.context_->request_ids_.size() ||
       std::adjacent_find(view.sorted_ids_.begin(), view.sorted_ids_.end()) !=
           view.sorted_ids_.end())) {
    return Status(
        StatusCode::INVALID_ARGUMENT,
        "MTP requires unique allocated embedding rows; zero is padding.");
  }
  if (stream.get_stream()->device_index() !=
      view.context_->tokens_.device().index()) {
    return Status(StatusCode::INVALID_ARGUMENT,
                  "MTP binding stream uses a different device.");
  }
  std::fill(
      view.publication_mask_.begin(), view.publication_mask_.end(), uint8_t{0});
  for (const uint32_t row : published_sequence_rows) {
    if (row >= embedding_ids.size() || view.publication_mask_[row] != 0) {
      return Status(StatusCode::INVALID_ARGUMENT,
                    "MTP Prefill publication rows must be unique model rows.");
    }
    view.publication_mask_[row] = 1;
  }
  std::fill(
      view.bootstrap_mask_.begin(), view.bootstrap_mask_.end(), uint8_t{0});
  for (const int32_t row : bootstrap_rows) {
    if (row < 0 || static_cast<uint64_t>(row) >= embedding_ids.size() ||
        view.bootstrap_mask_[row] != 0) {
      return Status(StatusCode::INVALID_ARGUMENT,
                    "MTP bootstrap rows must be unique model rows.");
    }
    view.bootstrap_mask_[row] = 1;
  }
  bool reuse_prefix = read_published_state && !embedding_ids.empty() &&
                      view.context_->draft_prefix_hidden_.defined();
  for (uint32_t row = 0; row < embedding_ids.size(); ++row) {
    const int32_t id = embedding_ids[row];
    if (read_published_state && view.bootstrap_mask_[row] == 0 &&
        (!view.context_->published_[id] ||
         view.context_->request_ids_[id] != request_ids[row])) {
      return Status(StatusCode::INVALID_ARGUMENT,
                    "MTP decode has no published context for this request.");
    }
    if (reuse_prefix && (view.bootstrap_mask_[row] != 0 ||
                         view.context_->request_ids_[id] != request_ids[row] ||
                         view.context_->draft_prefix_generations_[id] == 0)) {
      reuse_prefix = false;
    }
  }
  // Validation is complete. Host metadata only promises FIFO publication;
  // Prepare never overwrites the device row while an earlier Task uses it.
  for (uint32_t row = 0; row < embedding_ids.size(); ++row) {
    const int32_t id = embedding_ids[row];
    view.context_->request_ids_[id] = request_ids[row];
    view.context_->published_[id] = view.publication_mask_[row];
  }
  c10::DeviceGuard guard(view.context_->tokens_.device());
  view.prepared_ = true;
  view.read_published_state_ = read_published_state;
  view.reuse_draft_prefix_ = reuse_prefix;
  const int64_t count = static_cast<int64_t>(published_sequence_rows.size());
  int64_t* host = view.host_rows_.data_ptr<int64_t>();
  for (int64_t index = 0; index < count; ++index) {
    host[index] = embedding_ids[published_sequence_rows[index]];
  }
  view.rows_ = view.device_rows_.narrow(/*dim=*/0, /*start=*/0, count);
  view.tokens_ = view.token_storage_.narrow(/*dim=*/0, /*start=*/0, count);
  if (view.draft_prefix_hidden_storage_.defined()) {
    view.draft_prefix_hidden_ =
        prefix(view.draft_prefix_hidden_storage_, count);
  }
  for (const auto member : kContextFields) {
    const torch::Tensor& storage = view.storage_.*member;
    view.state_.*member = storage.defined()
                              ? storage.narrow(/*dim=*/0, /*start=*/0, count)
                              : torch::Tensor();
  }
  view.last_indices_ = view.index_storage_.select(/*dim=*/0, /*index=*/0)
                           .narrow(/*dim=*/0, /*start=*/0, count)
                           .view({count, 1});
  view.current_token_output_ = view.tokens_.unsqueeze(/*dim=*/1);
  if (view.state_.hidden.defined()) {
    view.previous_indices_ = view.index_storage_.select(/*dim=*/0, /*index=*/1)
                                 .narrow(/*dim=*/0, /*start=*/0, count)
                                 .view({count, 1});
    const int64_t hidden_size = view.state_.hidden.size(/*dim=*/2);
    view.last_hidden_indices_ =
        view.last_indices_.unsqueeze(/*dim=*/2).expand({count, 1, hidden_size});
    view.previous_hidden_indices_ =
        view.previous_indices_.unsqueeze(/*dim=*/2).expand(
            {count, 1, hidden_size});
    view.previous_token_output_ =
        view.state_.previous_tokens.unsqueeze(/*dim=*/1);
    view.current_hidden_output_ =
        view.state_.hidden.narrow(/*dim=*/1, /*start=*/1, /*length=*/1);
    view.previous_hidden_output_ =
        view.state_.hidden.narrow(/*dim=*/1, /*start=*/0, /*length=*/1);
    view.no_previous_ =
        view.mask_storage_.narrow(/*dim=*/0, /*start=*/0, count);
    view.no_previous_mask_ = view.no_previous_.view({count, 1, 1});
  }
  if (count != 0) {
    auto stream_guard = stream.set_stream_guard();
    view.rows_.copy_(view.host_rows_.narrow(/*dim=*/0, /*start=*/0, count),
                     /*non_blocking=*/true);
  }
  return Status();
}

void TaskExecutionPipeline::context_gather(MtpContextView& view) {
  context_gather_state(view);
  context_gather_prefix_hidden(view);
}

void TaskExecutionPipeline::context_gather_state(MtpContextView& view) {
  if (!view.prepared_ || view.tokens_.numel() == 0) {
    return;
  }
  CHECK(view.read_published_state_);
  torch::index_select_out(
      view.tokens_, view.context_->tokens_, /*dim=*/0, view.rows_);
  for (const auto member : kContextFields) {
    const torch::Tensor& source = view.context_->mtp_state_.*member;
    if (!source.defined()) {
      continue;
    }
    torch::index_select_out(view.state_.*member, source, /*dim=*/0, view.rows_);
  }
}

void TaskExecutionPipeline::context_gather_prefix_hidden(MtpContextView& view) {
  if (!view.prepared_ || view.tokens_.numel() == 0 ||
      !view.reuse_draft_prefix_) {
    return;
  }
  torch::index_select_out(view.draft_prefix_hidden_,
                          view.context_->draft_prefix_hidden_,
                          /*dim=*/0,
                          view.rows_);
}

void TaskExecutionPipeline::context_publish(MtpContextView& view) {
  if (!view.prepared_ || view.tokens_.numel() == 0) {
    return;
  }
  view.context_->tokens_.index_copy_(/*dim=*/0, view.rows_, view.tokens_);
  for (const auto member : kContextFields) {
    const torch::Tensor& destination = view.context_->mtp_state_.*member;
    if (!destination.defined()) {
      continue;
    }
    destination.index_copy_(/*dim=*/0, view.rows_, view.state_.*member);
  }
}

void TaskExecutionPipeline::context_advance(
    MtpContextView& view,
    const torch::Tensor& accepted_tokens,
    const torch::Tensor& accepted_lengths,
    const torch::Tensor& target_hidden) {
  if (!view.prepared_ || view.tokens_.numel() == 0) {
    return;
  }
  const int64_t count = view.tokens_.numel();
  CHECK_EQ(accepted_tokens.device(), view.tokens_.device());
  CHECK_EQ(accepted_tokens.scalar_type(), torch::kInt64);
  CHECK_EQ(accepted_tokens.dim(), 2);
  CHECK_EQ(accepted_tokens.size(/*dim=*/0), count);
  CHECK_GE(accepted_tokens.size(/*dim=*/1), 2);
  CHECK_EQ(accepted_lengths.device(), view.tokens_.device());
  CHECK_EQ(accepted_lengths.scalar_type(), torch::kInt32);
  CHECK_EQ(accepted_lengths.dim(), 1);
  CHECK_EQ(accepted_lengths.numel(), count);
  if (view.state_.hidden.defined()) {
    CHECK_EQ(target_hidden.device(), view.tokens_.device());
    CHECK_EQ(target_hidden.scalar_type(), view.state_.hidden.scalar_type());
    CHECK_EQ(target_hidden.dim(), 3);
    CHECK_EQ(target_hidden.size(/*dim=*/0), count);
    CHECK_EQ(target_hidden.size(/*dim=*/1), accepted_tokens.size(/*dim=*/1));
    CHECK_EQ(target_hidden.size(/*dim=*/2), view.state_.hidden.size(/*dim=*/2));
  }
  // The rejection sampler guarantees one valid prefix of length [1, width].
  // Keep lengths on Device; only the statically known width is read on Host.
  view.last_indices_.copy_(accepted_lengths.unsqueeze(/*dim=*/1));
  view.last_indices_.sub_(/*other=*/1);
  torch::gather_out(view.current_token_output_,
                    accepted_tokens,
                    /*dim=*/1,
                    view.last_indices_);
  if (view.state_.hidden.defined()) {
    view.previous_indices_.copy_(view.last_indices_);
    view.previous_indices_.sub_(/*other=*/1).clamp_min_(/*min=*/0);
    torch::gather_out(view.previous_token_output_,
                      accepted_tokens,
                      /*dim=*/1,
                      view.previous_indices_);
    torch::gather_out(view.current_hidden_output_,
                      target_hidden,
                      /*dim=*/1,
                      view.last_hidden_indices_);
    torch::gather_out(view.previous_hidden_output_,
                      target_hidden,
                      /*dim=*/1,
                      view.previous_hidden_indices_);
    torch::le_out(view.no_previous_, accepted_lengths, /*other=*/1);
    view.previous_hidden_output_.masked_fill_(view.no_previous_mask_,
                                              /*value=*/0);
    torch::eq_out(view.state_.repair_required,
                  accepted_lengths,
                  accepted_tokens.size(/*dim=*/1));
  }
  view.state_.positions.add_(accepted_lengths);
  view.state_.kv_seq_lens.add_(accepted_lengths);
  view.state_.accepted_lengths.copy_(accepted_lengths);
}

void TaskExecutionPipeline::context_release(MtpContextView& view) {
  view.prepared_ = false;
  view.read_published_state_ = false;
  view.reuse_draft_prefix_ = false;
}

uint64_t TaskExecutionPipeline::context_device_bytes(
    const MtpContextView& view) {
  return view.device_rows_.nbytes() + view.token_storage_.nbytes() +
         state_bytes(view.storage_) + view.index_storage_.nbytes() +
         (view.mask_storage_.defined() ? view.mask_storage_.nbytes() : 0) +
         (view.draft_prefix_hidden_storage_.defined()
              ? view.draft_prefix_hidden_storage_.nbytes()
              : 0);
}

}  // namespace xllm

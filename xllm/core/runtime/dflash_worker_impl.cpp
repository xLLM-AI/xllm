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

#include "runtime/dflash_worker_impl.h"

#include <glog/logging.h>

#include <algorithm>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "common/metrics.h"
#include "core/framework/config/execution_config.h"
#include "core/framework/config/kernel_config.h"
#include "core/framework/config/model_config.h"
#include "core/framework/config/scheduler_config.h"
#include "core/framework/config/speculative_config.h"
#include "core/framework/model/model_args.h"
#include "core/framework/parallel_state/process_group.h"
#include "core/framework/sampling/sampling_params.h"
#include "core/framework/speculative/draft_proposal.h"
#if defined(USE_NPU) || defined(USE_MLU)
#include "core/kv_cache/transfer/mooncake_kv_cache_transfer.h"
#endif
#if defined(USE_NPU)
#include "core/kv_cache/transfer/kv_transfer_completion.h"
#include "core/layers/common/expanded_decode_metadata_builder.h"
#include "core/layers/npu_torch/deepseek_sparse_attention.h"
#endif
#include "core/framework/batch/decode_input_builder.h"
#include "core/framework/model_loader/hf_model_config.h"
#include "core/framework/speculative/draft_extend_input.h"
#include "core/framework/speculative/spec_verify.h"
#include "core/platform/platform.h"
#include "core/runtime/json_object_decode_continuation.h"
#include "core/runtime/mtp_runtime_helpers.h"
#include "core/runtime/task_execution_pipeline.h"
#include "runtime/llm_worker_impl.h"
#include "util/json_reader.h"
#include "util/tensor_helper.h"
#include "util/timer.h"
#include "util/utils.h"

namespace xllm {

namespace {

runtime::Options target_options(const runtime::Options& options,
                                bool hybrid_pipeline) {
  runtime::Options opts = options;
  opts.enable_schedule_overlap(false)
      .is_draft_engine(false)
      .enable_graph_aux_hidden_states(true);
  if (hybrid_pipeline) {
    // The outer worker owns the pipeline; the recurrent target keeps its
    // ordinary ACL runner, just like the draft leaf.
    opts.enable_task_pipeline(false);
  }
  return opts;
}

runtime::Options draft_options(const runtime::Options& options,
                               bool hybrid_pipeline) {
  // DSpark sizes its attention window from num_speculative_tokens; other
  // DFlash-style drafts still run one step at a time.
  const bool is_dspark =
      SpeculativeConfig::is_dspark_algorithm(options.speculative_algorithm());
  const bool is_dflash2 =
      SpeculativeConfig::is_dflash2_algorithm(options.speculative_algorithm());
  const int32_t draft_num_speculative_tokens =
      is_dspark ? options.num_speculative_tokens() : 0;
  runtime::Options opts = options;
  opts.enable_schedule_overlap(false)
      .is_draft_engine(true)
      .num_decoding_tokens(is_dflash2 ? options.num_speculative_tokens() + 1
                                      : 1)
      .num_speculative_tokens(draft_num_speculative_tokens)
      .enable_graph_aux_hidden_states(false);
  if (hybrid_pipeline) {
    // The outer worker owns the task queue. Its leaf executors retain their
    // recurrent-aware ACL runners instead of borrowing full-attention slots.
    opts.enable_task_pipeline(false);
  } else if (options.enable_task_pipeline() && !is_dspark && !is_dflash2) {
    opts.enable_graph(false);
  }
  return opts;
}

std::optional<ForwardOutput> run_worker_no_sync_impl(
    WorkerImpl& worker,
    const LlmForwardInput& input,
    Stream& prepare_stream,
    Stream& compute_stream,
    LlmForwardInput* processed_output = nullptr) {
  LlmForwardInput processed_input;
  worker.prepare_work_before_execute_on_stream(
      input, processed_input, prepare_stream);
  std::optional<ForwardOutput> output = worker.execute_no_sync_on_stream(
      processed_input,
      compute_stream,
      WorkerImpl::ForwardCompletion::CALLER_MANAGED);
  if (processed_output != nullptr) {
    *processed_output = std::move(processed_input);
  }
  return output;
}

std::vector<int64_t> build_accepted_context_rows(
    const LlmForwardInput& input,
    const torch::Tensor& accepted_tokens_cpu,
    int32_t block_size,
    DecodeBuildBuffers& buf) {
  const int32_t batch_size = static_cast<int32_t>(accepted_tokens_cpu.size(0));
  const int32_t token_width = static_cast<int32_t>(accepted_tokens_cpu.size(1));
  CHECK_EQ(input.input_params.meta.num_sequences, batch_size)
      << "DFlash accepted token batch mismatch.";

  DecodeBatchContext row_ctx = make_decode_batch_context(input);
  std::vector<int64_t> accepted_idxes;
  accepted_idxes.reserve(static_cast<size_t>(accepted_tokens_cpu.numel()));
  reserve_entries(buf,
                  row_ctx,
                  static_cast<size_t>(accepted_tokens_cpu.numel()),
                  DecodeFields::kSlotsOnly);

  const int64_t* accepted_tokens_data =
      accepted_tokens_cpu.const_data_ptr<int64_t>();
  for (int32_t seq_id = 0; seq_id < batch_size; ++seq_id) {
    const int64_t row_offset = static_cast<int64_t>(seq_id) * token_width;
    for (int32_t token_idx = 0; token_idx < token_width; ++token_idx) {
      if (accepted_tokens_data[row_offset + token_idx] < 0) {
        break;
      }

      QueryEntrySpec row;
      row.seq_id = seq_id;
      row.position_offset = token_idx;
      row.append_token = false;
      append_decode_entry(row_ctx, row, block_size, buf);
      accepted_idxes.emplace_back(row_offset + token_idx);
    }
  }

  CHECK(!accepted_idxes.empty())
      << "DFlash accepted context must not be empty.";
  CHECK_EQ(buf.out_new_cache_slots.size(), buf.out_positions.size())
      << "DFlash accepted context slots/positions mismatch.";
  return accepted_idxes;
}

}  // namespace

namespace dflash_detail {

bool pad_empty_attention_dp_batch(LlmForwardInput& input,
                                  const ExpandDecodePlan& plan) {
  auto& params = input.input_params;
  const int32_t query_width = plan.num_val_tokens;
  for (int32_t& count : params.parallel.dp_global_token_nums) {
    if (count == 0) {
      count = query_width;
    }
  }
  if (params.meta.num_sequences != 0) {
    return false;
  }

  // An idle rank is a graph-bound ACL replay, which binds rows + 1 cu entries.
  DecodeBuildBuffers buf = make_idle_entries(plan);
  // Idle ranks stay host-only: the rows are not forwarded, so skip the H2D.
  input.runtime.device_tensors_ready = false;
  input.token_ids_host = make_cpu_tensor(buf.out_token_ids);
  input.positions_host = make_cpu_tensor(buf.out_positions);
  input.token_ids = input.token_ids_host;
  input.positions = input.positions_host;
  commit_decode_metadata(params, buf, plan);
  params.meta.actual_num_sequences = 0;
  input.runtime.input_host_buffer_has_layout = false;
  return true;
}

}  // namespace dflash_detail

bool DFlashWorkerImpl::prepare_draft_query_parallel_state(
    LlmForwardInput& query_input,
    const ExpandDecodePlan& plan) {
  scale_speculative_parallel_token_counts(query_input.input_params,
                                          plan.num_val_tokens);
  if (!draft_needs_dummy_rows_ ||
      !dflash_detail::pad_empty_attention_dp_batch(query_input, plan)) {
    return false;
  }
  c10::StreamGuard stream_guard = prepare_stream_->set_stream_guard();
  query_input.input_params.attention.rebuild_device_buffer(device_.unwrap());
  return true;
}

namespace {

// Ordinary DFlash retains its worker-owned recurrent target path. DSpark and
// DFlash2 use the shared block task pipeline and its capability validation.
bool uses_hybrid_worker_pipeline(const runtime::Options& options,
                                 const std::string& model_type) {
  return options.enable_task_pipeline() && Platform::is_npu() &&
         !options.enable_adaptive_target_verify() &&
         SpeculativeConfig::is_dflash_algorithm(
             options.speculative_algorithm()) &&
         is_qwen3_5_target_model_type(model_type);
}

}  // namespace

DFlashWorkerImpl::DFlashWorkerImpl(const ParallelArgs& parallel_args,
                                   const torch::Device& device,
                                   const runtime::Options& options)
    : DFlashWorkerImpl(
          parallel_args,
          device,
          options,
          options.model_path().empty()
              ? std::string()
              : get_model_type(options.model_path(), options.backend())) {}

DFlashWorkerImpl::DFlashWorkerImpl(const ParallelArgs& parallel_args,
                                   const torch::Device& device,
                                   const runtime::Options& options,
                                   const std::string& model_type)
    : DraftModelSpecWorkerImpl<LlmForwardInput>(
          parallel_args,
          device,
          options,
          target_options(options,
                         uses_hybrid_worker_pipeline(options, model_type)),
          WorkerType::LLM,
          [&parallel_args, &device, &options, &model_type] {
            // The draft consumes the target's aux hidden states, which the
            // worker does not expose under CP; reject cp_size > 1 except
            // DeepSeek-V4 on NPU.
            if (parallel_args.cp_size() > 1) {
              CHECK(Platform::is_npu() &&
                    util::is_deepseek_v4_model_type(model_type))
                  << "Block-diffusion speculative decoding does not support "
                     "context parallelism (cp_size > 1).";
            }
            return std::make_unique<LLMWorkerImpl>(
                parallel_args,
                device,
                draft_options(
                    options, uses_hybrid_worker_pipeline(options, model_type)));
          }),
      draft_query_width_(options.num_speculative_tokens() + 1),
      hybrid_pipeline_(uses_hybrid_worker_pipeline(options, model_type)),
      sampling_process_group_(parallel_args.tp_group_ != nullptr
                                  ? parallel_args.tp_group_
                                  : parallel_args.process_group_) {
  CHECK((options.adaptive_target_verify_cost_table().empty() &&
         options.adaptive_target_verify_calibration().empty()) ||
        (options.enable_task_pipeline() &&
         options.enable_adaptive_target_verify() &&
         SpeculativeConfig::is_dspark_algorithm(
             options.speculative_algorithm()) &&
         (options.num_speculative_tokens() == 5 ||
          options.num_speculative_tokens() == 7)))
      << "DSpark startup calibration options require an adaptive DSpark "
         "Draft5/7 "
         "task "
         "pipeline.";
}

DFlashWorkerImpl::~DFlashWorkerImpl() {
  // Keep launch-thread-owned context alive until queued execution completes.
  drain_pending_execution();
}

::xllm::Status DFlashWorkerImpl::create_task_pipeline(
    std::unique_ptr<TaskExecutionPipeline>& output) {
  if (target_is_hybrid_recurrent_ && hybrid_pipeline_) {
    if (!options_.enable_graph() || options_.enable_adaptive_target_verify() ||
        parallel_args_.dp_size() != 1) {
      return ::xllm::Status(
          StatusCode::INVALID_ARGUMENT,
          "Hybrid block task pipeline requires fixed-width Qwen3.5 Python "
          "ACL graphs with DP=1.");
    }
    LlmTaskCapacity capacity;
    ::xllm::Status status = impl_->task_capacity(options_, capacity);
    if (!status.ok()) {
      return status;
    }
    status = TaskExecutionPipeline::create(
        threadpool_, impl_->task_model(), *this, capacity, output);
    if (status.ok()) {
      LOG(INFO) << options_.speculative_algorithm()
                << " hybrid worker task execution pipeline: slots="
                << capacity.slot_count
                << ", speculative_tokens=" << options_.num_speculative_tokens();
    }
    return status;
  }
  if (impl_ == nullptr || draft_impl_ == nullptr ||
      (options_.enable_adaptive_target_verify() &&
       !SpeculativeConfig::is_dspark_algorithm(
           options_.speculative_algorithm())) ||
      options_.num_speculative_tokens() <= 0) {
    return ::xllm::Status(StatusCode::INVALID_ARGUMENT,
                          "Block task pipeline requires fixed-width or "
                          "adaptive DSpark full-attention models.");
  }
  SpeculativeTaskCapacity capacity;
  ::xllm::Status status = impl_->task_capacity(options_, capacity.common);
  if (!status.ok()) {
    return status;
  }
  capacity.hybrid_recurrent_target = target_is_hybrid_recurrent_;
  if (!capacity.common.enable_mla && !capacity.hybrid_recurrent_target) {
    return ::xllm::Status(
        StatusCode::INVALID_ARGUMENT,
        "Block task pipeline currently requires a GLM DSA target; "
        "dense target attention needs a device-length prepared contract.");
  }
  const auto& draft_args = draft_impl_->context_.get_model_args();
  if (SpeculativeConfig::is_dspark_algorithm(
          options_.speculative_algorithm()) &&
      parallel_args_.dp_size() > 1 &&
      ExecutionConfig::get_instance().enable_unified_mtp_graph()) {
    return ::xllm::Status(StatusCode::INVALID_ARGUMENT,
                          "DP DSpark task pipeline requires split ACL graphs; "
                          "disable unified graphs.");
  }
  if (draft_args.hidden_size() != capacity.common.hidden_size ||
      (draft_args.vocab_size() != capacity.common.vocab_size &&
       !draft_vocab_.draft_to_target_ids_.defined()) ||
      expected_context_hidden_size_ <= 0 ||
      expected_context_hidden_size_ > std::numeric_limits<int32_t>::max()) {
    return ::xllm::Status(StatusCode::INVALID_ARGUMENT,
                          "Block task model hidden/vocabulary or captured "
                          "context geometry mismatch.");
  }
  capacity.kind =
      SpeculativeConfig::is_dspark_algorithm(options_.speculative_algorithm())
          ? SpeculativeTaskKind::DSPARK
      : SpeculativeConfig::is_dflash2_algorithm(
            options_.speculative_algorithm())
          ? SpeculativeTaskKind::DFLASH2
          : SpeculativeTaskKind::DFLASH;
  capacity.num_speculative_tokens = options_.num_speculative_tokens();
  capacity.context_hidden_size =
      static_cast<uint32_t>(expected_context_hidden_size_);
  capacity.mask_token_id = mask_token_id_;
  capacity.selector_top_k = draft_args.dflash2_selector_top_k();
  capacity.common.sampling_group = sampling_process_group_;
  capacity.is_probabilistic_draft = is_probabilistic_draft_;
  capacity.draft_vocab = draft_vocab_;
  capacity.unified_graph =
      Platform::is_npu() && options_.enable_graph() &&
      ExecutionConfig::get_instance().enable_unified_mtp_graph() &&
      impl_->context_.get_model_args().model_type() == "glm_moe_dsa" &&
      (draft_args.model_type() == "DFlashDraftModel" ||
       is_dflash2_draft_model_type(draft_args.model_type()) ||
       draft_args.model_type() == "DSparkDraftModel") &&
      parallel_args_.dp_size() == 1 && parallel_args_.cp_size() == 1 &&
      parallel_args_.kv_split_size_effective() == 1;
  capacity.adaptive_target_verify = options_.enable_adaptive_target_verify();
  capacity.adaptive_min_gain = options_.adaptive_target_verify_min_gain();
  capacity.startup_calibration = capacity.adaptive_target_verify;
  capacity.calibration_leader = parallel_args_.rank() == 0;
  if (capacity.startup_calibration &&
      (!options_.enable_graph() || !options_.enable_schedule_overlap() ||
       ExecutionConfig::get_instance().disable_graph_warmup() ||
       (!options_.adaptive_target_verify_calibration().empty() &&
        !options_.adaptive_target_verify_cost_table().empty()))) {
    return ::xllm::Status(
        StatusCode::INVALID_ARGUMENT,
        "Adaptive DSpark calibration requires graph warmup, schedule overlap "
        "and at most one calibration-config or saved-table input.");
  }
  const auto& target_args = impl_->context_.get_model_args();
  capacity.sparse_dsa_target = target_args.model_type() == "glm_moe_dsa" &&
                               target_args.enable_mla() &&
                               target_args.index_topk() > 0;
  status = TaskExecutionPipeline::create(threadpool_,
                                         impl_->task_model(),
                                         draft_impl_->task_model(),
                                         capacity,
                                         output);
  if (!status.ok()) {
    return status;
  }
  LOG(INFO) << options_.speculative_algorithm()
            << " task execution pipeline: slots=" << capacity.common.slot_count
            << ", speculative_tokens=" << capacity.num_speculative_tokens
            << ", pinned_bytes=" << output->pinned_bytes()
            << ", device_bytes=" << output->device_bytes();
  return ::xllm::Status();
}

bool DFlashWorkerImpl::init_model(const std::string& model_weights_path,
                                  int32_t random_seed,
                                  MasterStatus master_status) {
  // DFlash draft attends each block non-causally, which the shared QWen3 model
  // only wires up on the chunked-prefill mask path. Without it the draft falls
  // back to a causal mask and proposal quality silently degrades, so require
  // the flag rather than accept a misconfigured run.
  CHECK(::xllm::SchedulerConfig::get_instance().enable_chunked_prefill())
      << "Block-diffusion speculative decoding requires "
         "--enable_chunked_prefill=true.";
  CHECK(SpeculativeConfig::is_probabilistic_draft_sampling_supported(
            options_.speculative_algorithm()) ||
        !is_probabilistic_draft_)
      << "This drafter only supports greedy drafts.";
  // Draft config's use_sliding_window / sliding_window is intentionally
  // ignored: xLLM NPU FIA hard-codes pre_tokens=INT_MAX, next_tokens=0
  // and never plumbs sliding_window into the aclnn call. Attended kv_len
  // on the draft path is bounded by the target sequence length, not by
  // block_size, so enforcing a block_size-vs-window relationship here
  // would compare the wrong quantities.
  const bool result = DraftModelSpecWorkerImpl<LlmForwardInput>::init_model(
      model_weights_path, random_seed, master_status);

  if (impl_->get_status() == WorkerImpl::Status::LOADED) {
    const ModelArgs& target_args = impl_->context_.get_model_args();
    target_is_hybrid_recurrent_ = has_linear_attention_layers(target_args);
    target_spec_verify_mode_ = mtp_async::classify_target_spec_verify_mode(
        target_args.model_type(),
        ModelConfig::is_python_model_impl(impl_->context_.get_model_impl()));
  }

  if (draft_impl_->get_status() == WorkerImpl::Status::LOADED) {
    const ModelArgs& draft_args = draft_impl_->context_.get_model_args();
    // DeepSeek-V4 DSpark checkpoints carry trained mtp.0.embed /
    // mtp.<last>.head tensors (with the draft-side QuaRot transform); replacing
    // them after load destroys proposal quality.
    const bool uses_own_head_and_embedding =
        util::is_deepseek_v4_dspark_model_type(draft_args.model_type());
    if (uses_own_head_and_embedding) {
      LOG(INFO) << "Configured DeepSeek-V4 DSpark draft block size: "
                << draft_args.dspark_block_size();
      LOG(INFO) << "Configured DeepSeek-V4 DSpark SAS mode: "
                << (draft_args.dspark_use_native_sas()
                        ? "native explicit indices"
                        : "CANN 9.0-compatible q_len=1 fallback");
#if defined(USE_NPU)
      if (draft_args.dspark_use_native_sas()) {
        LOG(WARNING)
            << "Native DeepSeek-V4 DSpark SAS requires an operator that "
               "accepts non-empty ori_sparse_indices and ori_win_left="
            << layer::deepseek_v4_ori_window_left(
                   draft_args.window_size(),
                   draft_args.dspark_block_size(),
                   /*use_native_dspark_sas=*/true)
            << ".";
      }
#endif
      // Keep the trained mtp.0.embed and mtp.<last>.head modules loaded by
      // DeepseekV4DSparkForCausalLMImpl. Sharing the target modules here makes
      // the draft backbone/Markov head project through the wrong vocabulary
      // basis and reduces acceptance to near-random levels.
    } else {
      const bool python_weights_shared =
          draft_impl_->WorkerImpl::share_weights_from(*impl_);
      if (!python_weights_shared) {
#if defined(USE_NPU)
        if (!SpeculativeConfig::is_dflash2_algorithm(
                options_.speculative_algorithm())) {
          auto head = impl_->get_npu_lm_head();
          draft_impl_->set_npu_lm_head(head);
          auto word_embedding = impl_->get_npu_word_embedding();
          draft_impl_->set_npu_word_embedding(word_embedding);
        } else
#endif
        {
          auto head = impl_->get_lm_head();
          draft_impl_->set_lm_head(head);
          auto word_embedding = impl_->get_word_embedding();
          draft_impl_->set_word_embedding(word_embedding);
        }
      }
    }

    JsonReader reader;
    const std::string config_path = model_weights_path + "/config.json";
    CHECK(reader.parse(config_path))
        << "Failed to parse block-diffusion draft config: " << config_path;
    mask_token_id_ = reader.value_or<int32_t>({"dflash_config.mask_token_id",
                                               "mask_token_id",
                                               "dspark_noise_token_id"},
                                              /*default=*/-1);
    CHECK_GE(mask_token_id_, 0)
        << "Block-diffusion draft config requires mask_token_id, "
           "dflash_config.mask_token_id, or dspark_noise_token_id.";

    const int64_t draft_vocab_size = draft_args.draft_vocab_size() > 0
                                         ? draft_args.draft_vocab_size()
                                         : draft_args.vocab_size();
    const ModelArgs& target_args = impl_->context_.get_model_args();
    CHECK_GT(draft_vocab_size, 0)
        << "Block-diffusion draft vocab_size must be set.";
    CHECK_LT(mask_token_id_, target_args.vocab_size())
        << "Block-diffusion mask_token_id (" << mask_token_id_
        << ") must be < target vocab_size (" << target_args.vocab_size()
        << ").";
    if (SpeculativeConfig::is_dflash2_algorithm(
            options_.speculative_algorithm())) {
      const int32_t requested_block_size =
          options_.num_speculative_tokens() + 1;
      CHECK_EQ(requested_block_size, draft_args.dflash2_block_size())
          << "DFlash2 runtime block size must match the checkpoint's trained "
             "dflash_config.block_size.";
      CHECK(!target_is_hybrid_recurrent_ ||
            ::xllm::ExecutionConfig::get_instance().enable_graph())
          << "DFlash2 with a hybrid recurrent target requires ACL Graph: the "
             "expanded spec-verify replay path preserves the accepted GDN "
             "checkpoint, while eager validation is not lossless.";
    }
    // Context hidden comes from the target.
    const int64_t num_target_layers =
        static_cast<int64_t>(target_args.layers_to_capture().size());
    CHECK_GT(num_target_layers, 0)
        << "Block-diffusion draft config requires dspark_target_layer_ids, "
           "target_layer_ids, dflash_config.target_layer_ids, or "
           "aux_hidden_state_layer_ids.";
    expected_context_hidden_size_ =
        static_cast<int64_t>(target_args.hidden_size()) * num_target_layers;
  }
  return result;
}

std::tuple<int64_t, int64_t> DFlashWorkerImpl::estimate_kv_cache_capacity() {
  CHECK(impl_ != nullptr);
  return estimate_kv_cache_capacity_with_draft(
      target_options(options_, hybrid_pipeline_),
      draft_options(options_, hybrid_pipeline_));
}

std::optional<ForwardOutput> DFlashWorkerImpl::step_empty(
    const LlmForwardInput& input) {
  if (!input.input_params.meta.batch_forward_type.is_decode()) {
    std::optional<ForwardOutput> output = run_worker_no_sync_impl(
        *impl_, input, *prepare_stream_, *compute_stream_);
    // Active prefill ranks write the draft context KV without a draft forward.
    // Keep idle ranks symmetric: a draft MoE forward here would enter EP
    // collectives that active ranks never join and deadlock the whole group.
    // Sync the target forward before its staged input can be reused.
    compute_stream_->synchronize();
    if (output.has_value()) {
      clear_all_output_embeddings(output.value());
    }
    return output;
  }

  const ExpandDecodePlan plan = draft_query_plan();
  const int32_t num_draft_tokens = plan.num_val_tokens;
  LlmForwardInput query_input = input.clone();
  dflash_detail::invalidate_draft_model_geometry(query_input.input_params);
  const int32_t num_sequences = query_input.input_params.meta.num_sequences;
  // Chunked rows commit one entry per sequence, flat rows one per val token.
  apply_commit_layout_meta(query_input.input_params,
                           plan,
                           is_chunked_layout(plan.layout)
                               ? num_sequences
                               : num_sequences * num_draft_tokens);
  prepare_draft_query_parallel_state(query_input, plan);
  // Warmup only: prime the draft; its output is unused. Keep it alive until the
  // sync below so the no-sync draft input is not freed while the target forward
  // launched next can reuse the buffer.
  std::optional<ForwardOutput> draft_output = run_worker_no_sync_impl(
      *draft_impl_, query_input, *prepare_stream_, *compute_stream_);

  LlmForwardInput validate_input = input.clone();
  // DSpark's N-wide draft geometry must be rescaled to (N+1) for the target's
  // anchor + drafts forward.
  scale_speculative_parallel_token_counts(
      validate_input.input_params, options_.num_speculative_tokens() + 1);
  ForwardOutput output =
      run_worker_no_sync_impl(
          *impl_, validate_input, *prepare_stream_, *compute_stream_)
          .value();
  // See above: sync the no-sync draft and target forwards before returning.
  compute_stream_->synchronize();
  clear_all_output_embeddings(output);
  return output;
}

std::optional<ForwardOutput> DFlashWorkerImpl::step_prefill(
    const LlmForwardInput& input) {
  LlmForwardInput processed_target_input;
  ForwardOutput output;
  {
    AUTO_COUNTER(speculative_execution_latency_seconds_target);
    output = run_worker_no_sync_impl(*impl_,
                                     input,
                                     *prepare_stream_,
                                     *compute_stream_,
                                     &processed_target_input)
                 .value();
  }

  const torch::Tensor& embeddings = output.sample_output.embeddings;
  if (embeddings.defined()) {
    const Slice<int32_t> positions =
        tensor_slice(processed_target_input.positions_host);
    CHECK_EQ(positions.size(), static_cast<size_t>(embeddings.size(0)))
        << "DFlash prefill hidden/position count mismatch.";
    torch::Tensor context_cache_slots =
        processed_target_input.input_params.attention.device.new_cache_slots;
    if (!processed_target_input.input_params.multi_block_tables.empty()) {
      const DecodeBatchContext ctx =
          make_decode_batch_context(processed_target_input);
      std::vector<int32_t> grouped_swa_slots =
          build_grouped_prefill_swa_slots(ctx, options_.block_size());
      c10::StreamGuard stream_guard = compute_stream_->set_stream_guard();
      context_cache_slots =
          async_h2d_tensor(grouped_swa_slots, device_.unwrap());
    }
    CHECK(context_cache_slots.defined())
        << "DFlash prefill requires context cache slots.";
    CHECK_EQ(static_cast<size_t>(context_cache_slots.numel()), positions.size())
        << "DFlash prefill hidden/cache slot count mismatch.";

    AUTO_COUNTER(speculative_execution_latency_seconds_draft);
    write_context_kv(processed_target_input,
                     embeddings,
                     processed_target_input.positions,
                     context_cache_slots);
  }

  if (input.sampling_params.num_sample_seqs() > 0) {
    prepare_first_draft_inputs(*embedding_cache_, input, output);
  } else {
    clear_all_output_embeddings(output);
  }

  if (!enable_schedule_overlap() && !driver_ && !dp_driver_) {
    return std::nullopt;
  }
  return output;
}

std::optional<ForwardOutput> DFlashWorkerImpl::step_decode(
    const LlmForwardInput& raw_input) {
  LlmForwardInput input = raw_input.clone();
  LlmForwardInput validate_input;

  CHECK(embedding_cache_ != nullptr)
      << "DFlash embedding cache is not allocated";

  prepare_draft_input(input);

  std::vector<EmbeddingCache::DecodeState> last_states =
      embedding_cache_->read_decode_states(
          input.input_params.embedding.embedding_ids,
          input.input_params.embedding.request_ids);
  CHECK_EQ(last_states.size(),
           input.input_params.embedding.embedding_ids.size())
      << "DFlash decode target state count mismatch";

  mtp_detail::update_decode_step_input(
      input,
      last_states,
      /*enable_cache_correction=*/enable_schedule_overlap(),
      mtp_detail::derive_expected_kv_policy(
          target_spec_verify_mode_ != mtp_async::TargetSpecVerifyMode::GENERIC,
          positions_are_decoupled_from_kv_length(),
          options_.num_speculative_tokens() + 1,
          logical_block_size()));
  DraftProposal draft_proposal = run_decode_draft(input, validate_input);
  return run_validate(input, draft_proposal, validate_input);
}

DraftProposal DFlashWorkerImpl::run_decode_draft(
    const LlmForwardInput& input,
    LlmForwardInput& validate_input) {
  Timer draft_timer;

  LlmForwardInput query_input;
  prepare_query_inputs(input, query_input);

  ForwardOutput draft_output =
      run_worker_no_sync_impl(
          *draft_impl_, query_input, *prepare_stream_, *compute_stream_)
          .value();
  // Overlap validate input preparation with the async draft forward: the draft
  // launch above returns immediately, so building validate_input here (it only
  // reads the original input; draft tokens are injected later in
  // fill_validate_input_from_draft_outputs) runs on the host while the draft
  // computes on device, instead of delaying the draft launch.
  prepare_validate_inputs(input, validate_input);
  // Keep draft tokens consistent across tensor-parallel ranks.
  maybe_broadcast_spec_tokens(draft_output.sample_output.next_tokens);
  COUNTER_ADD(speculative_execution_latency_seconds_draft,
              draft_timer.elapsed_seconds());

  // Draft emits the whole block in one forward; reshape the flat outputs into
  // [batch, num_speculative_tokens] instead of splitting into per-step outputs.
  const int32_t num_speculative_tokens = options_.num_speculative_tokens();
  const int32_t num_draft_tokens =
      static_cast<int32_t>(draft_output.sample_output.next_tokens.numel());
  CHECK_EQ(num_draft_tokens % num_speculative_tokens, 0)
      << "DFlash draft token count mismatch.";
  const int32_t batch_size = num_draft_tokens / num_speculative_tokens;
  DraftProposal draft_proposal(draft_output.sample_output.next_tokens.view(
      {batch_size, num_speculative_tokens}));
  draft_proposal.keep_alive() = take_retained_inputs(draft_output);
  return draft_proposal;
}

void DFlashWorkerImpl::fill_validate_input_from_draft_outputs(
    const DraftProposal& draft_proposal,
    LlmForwardInput& validate_input,
    Stream& compute_stream) {
  const int32_t num_speculative_tokens = options_.num_speculative_tokens();
  CHECK_GT(num_speculative_tokens, 0);
  const int32_t num_val_tokens = num_speculative_tokens + 1;
  const torch::Tensor& draft_token_ids = draft_proposal.token_ids();
  CHECK(draft_token_ids.defined())
      << "DFlash draft token_ids must be defined for validate token fill";
  CHECK_EQ(draft_token_ids.dim(), 2)
      << "DFlash draft token_ids must be [batch, num_speculative_tokens]";
  CHECK_EQ(draft_token_ids.size(1), num_speculative_tokens)
      << "DFlash draft token_ids width mismatch";
  CHECK(validate_input.token_ids.defined())
      << "DFlash validate token_ids must be prepared before draft token fill";
  CHECK_EQ(validate_input.token_ids.dim(), 1)
      << "DFlash validate token_ids must be flat";
  CHECK_EQ(validate_input.token_ids.numel() % num_val_tokens, 0)
      << "DFlash validate token_ids size must be divisible by validation width";

  const int64_t total_num_val_tokens = validate_input.token_ids.numel();
  const int64_t num_sequences = total_num_val_tokens / num_val_tokens;
  const torch::TensorOptions token_options = validate_input.token_ids.options();
  c10::StreamGuard stream_guard = compute_stream.set_stream_guard();
  CHECK(compute_stream.wait_event(validate_input.runtime.metadata_ready_event))
      << "failed to wait DFlash metadata ready event";
  torch::Tensor validate_token_rows =
      validate_input.token_ids.view({num_sequences, num_val_tokens});

  validate_input.runtime.device_tensors_ready = false;
  torch::Tensor draft_tokens = safe_to(
      draft_token_ids.contiguous(), token_options, /*non_blocking=*/true);
  validate_token_rows.slice(/*dim=*/1, /*start=*/1, /*end=*/num_val_tokens)
      .copy_(draft_tokens, /*non_blocking=*/true);
  validate_input.runtime.device_tensors_ready = true;
  // Publish this compute-stream write so the target's prepare stage (which
  // consumes validate_input.token_ids under ACL-graph double buffering) waits
  // for the copy to complete before staging into the graph's persistent
  // buffer. Without this, prepare could read stale/placeholder token ids.
  validate_input.runtime.metadata_ready_event =
      compute_stream.record_event_or_sync();
}

std::optional<ForwardOutput> DFlashWorkerImpl::run_validate(
    const LlmForwardInput& input,
    const DraftProposal& draft_proposal,
    LlmForwardInput& validate_input) {
  Timer target_timer;
  fill_validate_input_from_draft_outputs(
      draft_proposal, validate_input, *compute_stream_);

  ForwardOutput target_output =
      run_worker_no_sync_impl(
          *impl_, validate_input, *prepare_stream_, *compute_stream_)
          .value();
  COUNTER_ADD(speculative_execution_latency_seconds_target,
              target_timer.elapsed_seconds());

  Timer validation_timer;
  SampleOutput val_output = spec_verify::run_rejection_sampling(
      input.sampling_params,
      draft_proposal,
      target_output.logits,
      target_output.sample_output.embeddings);
  COUNTER_ADD(speculative_execution_latency_seconds_validation,
              validation_timer.elapsed_seconds());

  // target forward and validate()'s reads share compute_stream_, so they
  // serialize without a cross-stream wait; the sync below only makes the
  // accepted tokens host-visible for the D2H copy and context-cache write.
  maybe_broadcast_spec_tokens(val_output.next_tokens);
  compute_stream_->synchronize();
  val_output.next_tokens = to_cpu_contiguous(val_output.next_tokens);
  write_target_context_to_cache(input, val_output);

  if (!enable_schedule_overlap() && !driver_ && !dp_driver_) {
    return std::nullopt;
  }
  return finalize_verify_output(std::move(target_output),
                                std::move(val_output),
                                input.json_object_states);
}

torch::Tensor DFlashWorkerImpl::anchor_token_ids(
    const LlmForwardInput& input) const {
  const int32_t batch_size = input.input_params.meta.num_sequences;
  CHECK(input.token_ids_host.defined())
      << "block drafts require host token ids for the anchor.";
  CHECK_GE(input.token_ids_host.numel(), batch_size)
      << "anchor token count must cover the decode batch.";
  return input.token_ids_host.slice(/*dim=*/0, /*start=*/0, /*end=*/batch_size)
      .to(draft_impl_->device(), torch::kLong);
}

void DFlashWorkerImpl::maybe_broadcast_spec_tokens(torch::Tensor& tokens) {
  if (get_optimization_config().enable_spec_token_broadcast) {
    c10::StreamGuard stream_guard = compute_stream_->set_stream_guard();
    sync_sampling_tensor(tokens, sampling_process_group_);
  }
}

ExpandDecodePlan DFlashWorkerImpl::draft_query_plan() const {
  ExpandDecodePlan plan;
  plan.num_val_tokens = draft_query_width_;
  plan.block_size = options_.block_size();
  plan.layout = draft_query_layout_;
  plan.mask_token_id = mask_token_id_;
  return plan;
}

void DFlashWorkerImpl::prepare_validate_inputs(
    const LlmForwardInput& input,
    LlmForwardInput& validate_input) {
  c10::StreamGuard stream_guard = prepare_stream_->set_stream_guard();
  LlmForwardInput prepared_input = input.clone();
  prepared_input.runtime.metadata_ready_event.reset();
  const bool use_linear_spec_verify = target_is_hybrid_recurrent_;
  prepared_input.input_params.is_spec_verify = use_linear_spec_verify;
  SpeculativeWorkerImpl<LlmForwardInput>::prepare_validate_inputs(
      prepared_input, validate_input);
  validate_input.input_params.embedding.input_embedding = torch::Tensor();
  if (use_linear_spec_verify) {
    LlmModelParams& input_params = validate_input.input_params;
    const std::vector<int32_t> accepted_prefix_lengths =
        num_accepted_tokens_or_default(
            embedding_cache_.get(),
            static_cast<size_t>(input.input_params.meta.num_sequences),
            input.input_params.embedding.embedding_ids,
            input.input_params.embedding.request_ids);
    input_params.num_accepted_tokens_host.assign(
        accepted_prefix_lengths.begin(), accepted_prefix_lengths.end());
    input_params.num_accepted_tokens =
        torch::tensor(accepted_prefix_lengths,
                      validate_input.token_ids.options().dtype(torch::kInt32));
#if defined(USE_NPU)
    // DFlash binds the expanded layout only under graph mode; the MTP caller
    // gates this itself, so the shared builder stays unconditional.
    if (::xllm::ExecutionConfig::get_instance().enable_graph()) {
      layer::ExpandedDecodeMetadataBuilder::build_spec_verify_graph_input(
          ModelParamsView(input_params),
          device_.unwrap(),
          options_.block_size());
    }
#endif
  }
  validate_input.runtime.metadata_ready_event =
      prepare_stream_->record_event_or_sync();
}

void DFlashWorkerImpl::prepare_query_inputs(const LlmForwardInput& input,
                                            LlmForwardInput& query_input) {
  c10::StreamGuard stream_guard = prepare_stream_->set_stream_guard();
  query_input = input.clone();
  query_input.runtime.device_tensors_ready = false;
  LlmModelParams& input_params = query_input.input_params;
  input_params.embedding.input_embedding = torch::Tensor();
  dflash_detail::invalidate_draft_model_geometry(input_params);

  ExpandDecodePlan plan = draft_query_plan();
  const int32_t query_width = plan.num_val_tokens;
  const bool idle_rank = prepare_draft_query_parallel_state(query_input, plan);
  if (idle_rank) {
    query_input.sampling_params.selected_token_idxes() = torch::Tensor();
  } else {
    DecodeBatchContext row_ctx = make_decode_batch_context(input);
    DecodeBuildBuffers buf = expand_decode_inputs(row_ctx, plan);
    // A chunked-prefill commit keeps the anchor row unselected; a chunked
    // draft that commits as DECODE samples every row.
    const bool selects_anchor_row =
        !commit_forward_type(plan.layout).is_chunked_prefill();
    const int32_t rows_per_seq =
        selects_anchor_row ? query_width : query_width - 1;
    const int32_t total_rows = row_ctx.num_sequences * rows_per_seq;
    const torch::Tensor& draft_query_indices =
        draft_query_indices_.get(total_rows, [&]() {
          return selects_anchor_row
                     ? arange_indices(total_rows, device_)
                     : sample_flattened_indices(
                           row_ctx.num_sequences, query_width, device_);
        });
    commit_decode_input(query_input, buf, plan, device_);
    query_input.sampling_params.selected_token_idxes() = draft_query_indices;
  }
  prepare_draft_sampling(/*probabilistic=*/false, query_input.sampling_params);
  for (const auto& field : kSamplingFields) {
    if (!field.per_request) {
      repeat_interleave_if_defined(field.member(query_input.sampling_params),
                                   options_.num_speculative_tokens());
    }
  }
  query_input.runtime.device_tensors_ready = true;
}

void DFlashWorkerImpl::write_context_kv(
    const LlmForwardInput& input,
    const torch::Tensor& context_hidden,
    const torch::Tensor& positions_device,
    const torch::Tensor& new_cache_slots_device) {
  CHECK(context_hidden.defined()) << "DFlash context hidden is undefined.";
  CHECK_EQ(context_hidden.dim(), 2) << "DFlash context hidden must be 2D.";
  CHECK_EQ(context_hidden.size(1), expected_context_hidden_size_)
      << "DFlash context hidden size must be hidden_size * "
      << "target_layer_ids.size().";

  CHECK(context_hidden.device() == device_.unwrap())
      << "DFlash context hidden must already be on the compute device.";

  // Both the target forward that produced context_hidden and this pass run
  // on compute_stream_, so no explicit event dance is needed — the stream
  // orders them. Model methods below use torch ops on the same stream.
  c10::StreamGuard stream_guard = compute_stream_->set_stream_guard();
  CHECK(input.input_params.synchronize_draft_layer())
      << "Failed to wait for Draft Host KV load completion.";

#if defined(USE_NPU)
  // PD PUSH: the draft context-KV scattered below is not covered by the target
  // push, so wire the draft push here. Overwrite layer_synchronizer with a
  // draft-sized one (one event per draft layer); the target's was already
  // waited on before this runs, so the overwrite is safe. No-op when
  // transfer_kv_infos is empty. NPU-only: the scatter's record_event loop is
  // NPU-gated, so a non-NPU push would stall on events that never record.
  KVTransferCompletion kv_transfers;
  if (options_.kv_cache_transfer_mode() == "PUSH" &&
      !input.transfer_kv_infos.empty()) {
    std::shared_ptr<NPULayerSynchronizerImpl> layer_synchronizer =
        std::make_shared<NPULayerSynchronizerImpl>(
            draft_impl_->context_.get_model_args().n_layers());
    input.input_params.parallel.layer_synchronizer = layer_synchronizer;
    kv_transfers.add(kv_cache_transfer_->push_kv_blocks_async(
        input.transfer_kv_infos,
        draft_impl_->context_.get_parallel_args(),
        layer_synchronizer,
        /*is_spec_draft=*/true));
  }
#endif

  ModelOutput scatter_output =
      draft_impl_->write_context_kv(context_hidden,
                                    positions_device,
                                    new_cache_slots_device,
                                    input.input_params);
  // Model returns an empty ModelOutput (hidden_states undefined) when the
  // per-layer NPULayerSynchronizer::record_event fails mid-scatter under
  // PD-PUSH transfer. Fail fast instead of silently continuing: subsequent
  // layers won't have been scattered and the PD transfer side would block
  // forever waiting on the missing per-layer event.
  CHECK(scatter_output.hidden_states.defined())
      << "DFlash context-KV scatter failed (layer_synchronizer record_event "
         "returned false); PD-PUSH transfer would deadlock.";

  // The context-KV scatter is the last device op of the step and is launched
  // no-sync. Under schedule-overlap the scheduler dispatches the next step's
  // host prep as soon as this step's host call returns, while this scatter may
  // still be in flight. Sync so the next step's draft query reads a fully
  // written context-KV cache instead of a partially scattered one.
  compute_stream_->synchronize();

#if defined(USE_NPU)
  // Wait for the draft KV push (if any) so the source draft cache is not
  // overwritten by the next step while the transfer is still reading it
  // (mirrors step_internal's wait_kv_push()). No-op when no push was issued.
  CHECK(kv_transfers.wait()) << "DFlash draft context-KV push failed";
#endif
}

void DFlashWorkerImpl::write_target_context_to_cache(
    const LlmForwardInput& input,
    const SampleOutput& validate_output) {
  const torch::Tensor& accepted_embeddings = validate_output.embeddings;
  CHECK(accepted_embeddings.defined())
      << "DFlash validate target embeddings are undefined.";
  CHECK_EQ(accepted_embeddings.dim(), 3)
      << "DFlash validate target embeddings must be [batch,width,hidden].";

  CHECK(validate_output.next_tokens.defined())
      << "DFlash accepted tokens are undefined.";
  const torch::Tensor accepted_tokens =
      to_cpu_contiguous(validate_output.next_tokens, torch::kInt64);

  CHECK_EQ(accepted_tokens.dim(), 2)
      << "DFlash accepted tokens must be [batch,width].";
  const int64_t batch_size = accepted_tokens.size(0);
  const int64_t token_width = accepted_tokens.size(1);
  CHECK_EQ(accepted_embeddings.size(0), batch_size)
      << "DFlash accepted token/embedding batch mismatch.";
  CHECK_EQ(accepted_embeddings.size(1), token_width)
      << "DFlash accepted token/embedding width mismatch.";

  DecodeBuildBuffers buf;
  std::vector<int64_t> accepted_idxes = build_accepted_context_rows(
      input, accepted_tokens, options_.block_size(), buf);
  c10::StreamGuard stream_guard = prepare_stream_->set_stream_guard();
  torch::Tensor accepted_index =
      async_h2d_tensor(accepted_idxes, accepted_embeddings.device());
  torch::Tensor flat_embeddings = accepted_embeddings.reshape(
      {batch_size * token_width, accepted_embeddings.size(/*dim=*/2)});
  torch::Tensor context_hidden =
      flat_embeddings.index_select(/*dim=*/0, accepted_index);
  torch::Tensor positions_device =
      async_h2d_tensor(buf.out_positions, device_.unwrap());
  torch::Tensor new_cache_slots_device =
      async_h2d_tensor(buf.out_new_cache_slots, device_.unwrap());
  // Publish the prepare_stream_ work (index_select producing context_hidden +
  // pinned H2D copies for positions/slots) so compute_stream_ waits for it
  // before the model reads these tensors. Without this, torch does not
  // enforce cross-stream ordering: the write_context_kv scatter could launch
  // before index_select finishes, producing corrupt KV cache. The prefill
  // caller runs its producer on compute_stream_, so no event is needed there.
  StreamEventPtr context_hidden_ready_event =
      prepare_stream_->record_event_or_sync();
  if (context_hidden_ready_event != nullptr) {
    CHECK(compute_stream_->wait_event(context_hidden_ready_event))
        << "failed to wait DFlash context hidden ready event";
  }
  write_context_kv(
      input, context_hidden, positions_device, new_cache_slots_device);
  CHECK(!input.input_params.embedding.embedding_ids.empty())
      << "DFlash target context cache write requires embedding ids";
  embedding_cache_->write_target_context(
      input.input_params.embedding.embedding_ids,
      input.input_params.embedding.request_ids,
      validate_output.next_tokens,
      validate_output.embeddings,
      options_.num_speculative_tokens(),
      /*retain_previous_embedding=*/false);
}

void DFlashWorkerImpl::prepare_task_pipeline_input(const LlmForwardInput& input,
                                                   LlmForwardInput& prepared) {
  // Prepare owns only transport storage. The target leaf restores recurrent
  // state on the ordered execution stream, after preceding model work.
  prepare_input_on_stream(input,
                          prepared,
                          *prepare_stream_,
                          /*record_ready_event=*/true,
                          /*restore_linear_state=*/false,
                          /*wait_for_compute_stream=*/false);
}

std::optional<ForwardOutput> DFlashWorkerImpl::execute_task_pipeline(
    const LlmForwardInput& prepared) {
  // A worker-owned queue must not use the driver's legacy last-step slot.
  if (!enable_schedule_overlap() || prepared.json_object_states.empty()) {
    task_json_output_ = ForwardOutput();
    task_json_sample_sequence_ids_.clear();
    return step(prepared);
  }
  LlmForwardInput input = prepared.clone();
  c10::StreamGuard stream_guard = compute_stream_->set_stream_guard();
  CHECK(compute_stream_->wait_event(input.runtime.metadata_ready_event));
  if (input.token_ids.numel() > 0 &&
      input.input_params.meta.batch_forward_type.has_decode()) {
    JsonObjectDecodeContinuation::prepare(input,
                                          task_json_output_,
                                          task_json_sample_sequence_ids_,
                                          device_,
                                          *compute_stream_);
  }
  auto output = step(input);
  task_json_output_ = ForwardOutput();
  task_json_sample_sequence_ids_.clear();
  if (output.has_value()) {
    output->json_object_errors.insert(output->json_object_errors.end(),
                                      input.json_object_errors.begin(),
                                      input.json_object_errors.end());
    if (output->sample_output.next_tokens.defined()) {
      if (output->ready_event != nullptr) {
        CHECK(output->ready_event->synchronize());
      }
      task_json_output_.sample_output.next_tokens =
          safe_to(
              output->sample_output.next_tokens,
              torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU),
              /*non_blocking=*/false)
              .clone();
      task_json_sample_sequence_ids_ = input.sample_sequence_ids;
    }
  }
  return output;
}

}  // namespace xllm

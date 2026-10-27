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
#include <iterator>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "common/metrics.h"
#include "core/framework/config/execution_config.h"
#include "core/framework/config/kernel_config.h"
#include "core/framework/config/scheduler_config.h"
#include "core/framework/model/model_args.h"
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
#include "core/common/types.h"
#include "core/framework/batch/decode_input_builder.h"
#include "core/framework/model/model_type.h"
#include "core/framework/model_loader/hf_model_config.h"
#include "core/framework/speculative/draft_extend_input.h"
#include "core/framework/speculative/spec_verify.h"
#include "core/platform/platform.h"
#include "core/util/tensor_helper.h"
#include "runtime/llm_worker_impl.h"
#include "util/json_reader.h"
#include "util/timer.h"
#include "util/utils.h"

namespace xllm {

runtime::Options DFlashWorkerImpl::draft_options(
    const runtime::Options& options,
    int32_t num_speculative_tokens) {
  runtime::Options opts = options;
  opts.enable_schedule_overlap(false)
      .is_draft_model(true)
      .num_speculative_tokens(num_speculative_tokens)
      .enable_graph_aux_hidden_states(false);
  return opts;
}

namespace {

int32_t configure_draft_config(ModelArgs& args,
                               const runtime::Options& options) {
  args.layers_to_capture({});
  args.model_type("DFlashDraftModel");
  return options.num_speculative_tokens();
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

  DecodeBatchContext batch_ctx = make_decode_batch_context(input);
  std::vector<int64_t> accepted_idxes;
  accepted_idxes.reserve(static_cast<size_t>(accepted_tokens_cpu.numel()));
  prepare_decode_buffers(buf,
                         batch_ctx,
                         static_cast<size_t>(accepted_tokens_cpu.numel()),
                         /*emit_kv_len=*/false,
                         /*emit_block_tables=*/false,
                         /*emit_token=*/false);

  const int64_t* accepted_tokens_data =
      accepted_tokens_cpu.const_data_ptr<int64_t>();
  for (int32_t seq_id = 0; seq_id < batch_size; ++seq_id) {
    const int64_t row_offset = static_cast<int64_t>(seq_id) * token_width;
    for (int32_t token_idx = 0; token_idx < token_width; ++token_idx) {
      if (accepted_tokens_data[row_offset + token_idx] < 0) {
        break;
      }

      DecodeTokenInput token_input;
      token_input.req_id = seq_id;
      token_input.position_offset = token_idx;
      token_input.append_token = false;
      append_decode_token(batch_ctx, token_input, block_size, buf);
      accepted_idxes.emplace_back(row_offset + token_idx);
    }
  }

  CHECK(!accepted_idxes.empty())
      << "DFlash accepted context must not be empty.";
  return accepted_idxes;
}

}  // namespace

bool DFlashWorkerImpl::prepare_parallel_draft_metadata(
    LlmForwardInput& query_input,
    const DecodeExpansionOptions& decode_options) {
  scale_speculative_parallel_token_counts(
      query_input.input_params, decode_options.num_speculative_tokens);
  if (!draft_needs_dummy_tokens_) {
    return false;
  }
  auto& params = query_input.input_params;
  std::replace(params.parallel.dp_global_token_nums.begin(),
               params.parallel.dp_global_token_nums.end(),
               0,
               (decode_options.num_speculative_tokens + 1));
  if (params.meta.num_sequences != 0) {
    return false;
  }

  build_idle_decode_input(query_input, decode_options);
  params.meta.actual_num_sequences = 0;
  query_input.runtime.input_host_buffer_has_layout = false;
  c10::StreamGuard stream_guard = prepare_stream_->set_stream_guard();
  query_input.input_params.attention.rebuild_device_buffer(device_.unwrap());
  return true;
}

DFlashWorkerImpl::DFlashWorkerImpl(const ParallelArgs& parallel_args,
                                   const torch::Device& device,
                                   const runtime::Options& options)
    : DFlashWorkerImpl(
          parallel_args,
          device,
          options,
          [&] {
            return std::make_unique<LLMWorkerImpl>(
                configure_draft_parallel_args(parallel_args),
                device,
                draft_options(options),
                parallel_args.dp_size() > 1);
          },
          configure_draft_config) {}

DFlashWorkerImpl::DFlashWorkerImpl(
    const ParallelArgs& parallel_args,
    const torch::Device& device,
    const runtime::Options& options,
    absl::FunctionRef<std::unique_ptr<LLMWorkerImpl>()> draft_factory,
    int32_t (*configure_draft_config)(ModelArgs&, const runtime::Options&))
    : DraftModelSpecWorkerImpl<LlmForwardInput>(
          parallel_args,
          device,
          options,
          WorkerType::LLM,
          [&] {
            // The draft consumes the target's aux hidden states, which the
            // worker does not expose under CP; reject cp_size > 1 except
            // DeepSeek-V4 on NPU.
            if (parallel_args.cp_size() > 1) {
              CHECK(Platform::is_npu() &&
                    util::is_deepseek_v4_model_type(get_model_type(
                        options.model_path(), options.backend())))
                  << "Block-diffusion speculative decoding does not support "
                     "context parallelism (cp_size > 1).";
            }
            return draft_factory();
          },
          configure_draft_config) {}

DFlashWorkerImpl::~DFlashWorkerImpl() {
  // Keep launch-thread-owned context alive until queued execution completes.
  drain_pending_execution();
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
                << draft_args.block_size();
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
                   draft_args.block_size(),
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
        auto head = impl_->get_lm_head();
        draft_impl_->set_lm_head(head);
        auto word_embedding = impl_->get_word_embedding();
        draft_impl_->set_word_embedding(word_embedding);
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

  const DecodeExpansionOptions decode_options = prepare_for_draft();
  LlmForwardInput query_input = input.clone();
  dflash_detail::invalidate_draft_model_geometry(query_input.input_params);
  const int32_t num_sequences = query_input.input_params.meta.num_sequences;
  if (!prepare_parallel_draft_metadata(query_input, decode_options)) {
    const int32_t rows_per_sequence =
        decode_options.per_request
            ? 1
            : (decode_options.num_speculative_tokens + 1);
    update_decode_batch_metadata(query_input.input_params,
                                 decode_options,
                                 num_sequences * rows_per_sequence);
  }
  // Warmup only: prime the draft; its output is unused. Keep it alive until the
  // sync below so the no-sync draft input is not freed while the target forward
  // launched next can reuse the buffer.
  std::optional<ForwardOutput> draft_output = run_worker_no_sync_impl(
      *draft_impl_, query_input, *prepare_stream_, *compute_stream_);

  LlmForwardInput validate_input = input.clone();
  // DSpark's N-wide draft geometry must be rescaled to (N+1) for the target's
  // anchor + drafts forward.
  scale_speculative_parallel_token_counts(validate_input.input_params,
                                          options_.num_speculative_tokens());
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
                                     processed_target_input)
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
    append_failed_request_ids(output,
                              write_context_kv(processed_target_input,
                                               embeddings,
                                               processed_target_input.positions,
                                               context_cache_slots));
  }

  if (input.sampling_params.num_sample_seqs() > 0) {
    prepare_first_draft_inputs(*embedding_cache_, input, output);
  } else {
    clear_all_output_embeddings(output);
  }

  return finalize_step_output(std::move(output));
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

  update_decode_step_input(input, last_states);
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
  maybe_broadcast_spec_tokens(draft_output.sample_output.next_tokens,
                              input.sampling_params.sampling_mode());
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
  maybe_broadcast_spec_tokens(val_output.next_tokens,
                              input.sampling_params.sampling_mode());
  compute_stream_->synchronize();
  val_output.next_tokens = to_cpu_contiguous(val_output.next_tokens);
  append_failed_request_ids(target_output,
                            write_target_context_to_cache(input, val_output));

  return finalize_step_output(std::move(target_output),
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
DecodeExpansionOptions DFlashWorkerImpl::prepare_for_draft() const {
  DecodeExpansionOptions decode_options;
  decode_options.num_speculative_tokens =
      options_.num_speculative_tokens() -
      static_cast<int32_t>(sample_from_anchor_);
  decode_options.block_size = options_.block_size();
  decode_options.per_request = draft_input_per_request_;
  decode_options.causal = false;
  decode_options.forward_type = draft_forward_type_;
  decode_options.sample_from_anchor = sample_from_anchor_;
  decode_options.mask_token_id = mask_token_id_;
  return decode_options;
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
    const std::vector<int32_t> accepted_prefix_lengths = num_accepted_tokens(
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
    if (::xllm::ExecutionConfig::get_instance().enable_graph()) {
      layer::ExpandedDecodeMetadataBuilder::update_expanded_decode_metadata(
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

  const DecodeExpansionOptions decode_options = prepare_for_draft();
  const int32_t query_width = (decode_options.num_speculative_tokens + 1);
  const bool idle_rank =
      prepare_parallel_draft_metadata(query_input, decode_options);
  if (idle_rank) {
    query_input.sampling_params.logits_indices() = torch::Tensor();
  } else {
    DecodeBatchContext batch_ctx = make_decode_batch_context(input);
    DecodeBuildBuffers buf = expand_decode_inputs(batch_ctx, decode_options);
    const int32_t num_logits_per_request =
        decode_options.sample_from_anchor ? query_width : query_width - 1;
    const int32_t num_logits = batch_ctx.num_requests * num_logits_per_request;
    const torch::Tensor& draft_query_indices =
        draft_logits_indices_.get(num_logits, [&]() {
          return decode_options.sample_from_anchor
                     ? arange_indices(num_logits, device_)
                     : torch::arange(
                           static_cast<int64_t>(batch_ctx.num_requests) *
                               query_width,
                           torch::TensorOptions()
                               .dtype(torch::kInt)
                               .device(device_))
                           .view({batch_ctx.num_requests, query_width})
                           .narrow(/*dim=*/1, 1, query_width - 1)
                           .reshape({-1});
        });
    build_decode_input(query_input, buf, decode_options, device_);
    query_input.sampling_params.logits_indices() = draft_query_indices;
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

std::vector<std::string> DFlashWorkerImpl::write_context_kv(
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
  KVTransferCompletion kv_transfers = [&]() {
    if (options_.kv_push_enabled() && !input.transfer_kv_infos.empty()) {
      std::shared_ptr<NPULayerSynchronizerImpl> layer_synchronizer =
          std::make_shared<NPULayerSynchronizerImpl>(
              draft_impl_->context_.get_model_args().n_layers());
      input.input_params.parallel.layer_synchronizer = layer_synchronizer;
      auto request_ids =
          KVCacheTransfer::unique_request_ids(input.transfer_kv_infos);
      return KVTransferCompletion(kv_cache_transfer_->push_kv_blocks_async(
                                      input.transfer_kv_infos,
                                      draft_impl_->context_.get_parallel_args(),
                                      layer_synchronizer,
                                      /*is_spec_draft=*/true),
                                  std::move(request_ids));
    }
    return KVTransferCompletion{};
  }();
#endif

  ModelParamsView execution_params(input.input_params);
  ModelOutput scatter_output =
      draft_impl_->get_causal_lm()->write_context_kv(context_hidden,
                                                     positions_device,
                                                     new_cache_slots_device,
                                                     draft_impl_->kv_caches(),
                                                     execution_params);
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
  return kv_transfers.wait(
      draft_impl_->context_.get_parallel_args().kv_transfer_groups(), device_);
#else
  return {};
#endif
}

std::vector<std::string> DFlashWorkerImpl::write_target_context_to_cache(
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
  std::vector<std::string> failed_request_ids = write_context_kv(
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
  return failed_request_ids;
}

}  // namespace xllm

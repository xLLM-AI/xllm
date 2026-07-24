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

#include "llm_worker_impl.h"

#include <c10/core/DeviceGuard.h>
#include <folly/Unit.h>
#include <folly/futures/Future.h>
#include <glog/logging.h>
#include <torch/torch.h>

#include <memory>
#include <optional>
#include <utility>

#include "common/device_monitor.h"
#include "common/metrics.h"
#include "common/types.h"
#include "core/framework/config/beam_search_config.h"
#include "core/framework/config/eplb_config.h"
#include "core/framework/config/kv_cache_config.h"
#include "core/framework/config/load_config.h"
#include "core/framework/config/model_config.h"
#include "core/platform/platform.h"
#include "framework/kv_cache/kv_cache.h"
#include "framework/kv_cache/linear_state_restore.h"
#include "framework/kv_cache_transfer/kv_transfer_completion.h"
#include "framework/model/model_input_params.h"
#include "framework/state_dict/state_dict.h"
#if defined(USE_CUDA) || defined(USE_ILU) || defined(USE_MUSA)
#include "layers/cuda/flashinfer_workspace.h"
#endif
#include "models/model_registry.h"
#include "util/threadpool.h"
#include "util/timer.h"

namespace xllm {

namespace {

bool uses_worker_cp_partition(const runtime::Options& options) {
  return options.cp_size() > 1 && !Platform::uses_model_cp_partition();
}

void wait_input_ready_events(const ForwardInput& input, const Stream& stream) {
  CHECK(stream.wait_event(input.metadata_ready_event))
      << "failed to wait ForwardInput metadata ready event";
}

StreamEventPtr record_current_stream_event(const Device& device) {
  std::unique_ptr<Stream> stream = device.current_stream();
  StreamEventPtr event = stream->record_event();
  if (event == nullptr) {
    stream->synchronize();
  }
  return event;
}

}  // namespace

LLMWorkerImpl::LLMWorkerImpl(const ParallelArgs& parallel_args,
                             const torch::Device& device,
                             const runtime::Options& options)
    : WorkerImpl(parallel_args, device, options) {
  device_.set_device();
#if defined(USE_CUDA) || defined(USE_MUSA)
  const auto& model_config = ModelConfig::get_instance();
  if (!ModelConfig::is_python_model_impl(model_config.model_impl())) {
    threadpool_.schedule([this]() mutable {
      // initialize flashinfer workspace
      ::xllm::layer::flashinfer::FlashinferWorkspace::get_instance().initialize(
          device_);
    });
  }
#endif
}

bool LLMWorkerImpl::init_model(ModelContext& context) {
  CHECK(model_ == nullptr) << "Model is already initialized.";
  const auto& model_config = ModelConfig::get_instance();

#if defined(USE_CUDA)
  // Ensure FlashinferWorkspace is initialized on the calling thread before
  // constructing model layers. When called synchronously from
  // SpeculativeWorkerImpl (e.g. MTP target/draft setup), init_model runs on
  // the MTP worker's thread (T_MTP) rather than on the LLMWorkerImpl's own
  // threadpool thread (T_worker) where the scheduled initialize() runs.
  // FlashinferWorkspace is thread_local, so T_MTP's instance must be
  // explicitly initialized here; otherwise FlashInferAttentionImpl captures
  // an undefined int_workspace_buffer_ and crashes at prefill time.
  //
  // Skip when model_impl=python: Python executor uses flashinfer's Python API
  // directly; initializing the C++ workspace would conflict with Python-side
  // TVM-FFI type registration.
  if (!ModelConfig::is_python_model_impl(model_config.model_impl())) {
    auto& ws = ::xllm::layer::flashinfer::FlashinferWorkspace::get_instance();
    if (!ws.get_int_workspace_buffer().defined()) {
      ws.initialize(device_);
    }
  }
#endif

  // Try to create a causal LM model
  context.set_model_impl(model_config.model_impl());
  model_ = create_llm_model(context);

  // Dont find model in causal models
  CHECK(model_ != nullptr) << "Failed to create model.";
  model_executor_ = std::make_unique<Executor>(
      model_.get(), context.get_model_args(), device_, options_);

  if (::xllm::EPLBConfig::get_instance().enable_eplb()) {
    eplb_executor_ = std::make_unique<EplbExecutor>(model_.get(), device_);
  }

  if (::xllm::BeamSearchConfig::get_instance().enable_beam_search_kernel()) {
    beam_searcher_ = std::make_unique<BeamSearcher>();
  }
  return true;
}

std::optional<ForwardOutput> LLMWorkerImpl::step_no_sync(
    const ForwardInput& input) {
  ForwardInput input_on_device;
  prepare_work_before_execute(input, input_on_device);
  std::unique_ptr<Stream> current_stream = device_.current_stream();
  return execute_no_sync_on_stream(input_on_device, *current_stream);
}

std::optional<ForwardOutput> LLMWorkerImpl::execute_no_sync_on_stream(
    const ForwardInput& input,
    Stream& compute_stream,
    bool record_ready_event) {
  const ForwardSyncPolicy sync_policy = ForwardSyncPolicy::NO_SYNC;
  c10::StreamGuard stream_guard = compute_stream.set_stream_guard();
  if (::xllm::LoadConfig::get_instance().enable_manual_loader()) {
#if defined(USE_NPU)
    if (!enable_schedule_overlap() && options_.backend() == "llm") {
      aclrtStream current_acl_stream =
          c10_npu::getCurrentNPUStream(device_.index()).stream();
      atb::Context* atb_context =
          const_cast<atb::Context*>(context_.get_atb_context());
      atb_context->SetExecuteStream(current_acl_stream);
      wait_input_ready_events(input, compute_stream);
      return step_internal(input, sync_policy, record_ready_event);
    } else {
      SET_ATB_EXECUTE_STREAM((&compute_stream), device_, context_);
      wait_input_ready_events(input, compute_stream);
      return step_internal(input, sync_policy, record_ready_event);
    }
#else
    wait_input_ready_events(input, compute_stream);
    return step_internal(input, sync_policy, record_ready_event);
#endif
  }
  wait_input_ready_events(input, compute_stream);
  return step_internal(input, sync_policy, record_ready_event);
}

std::optional<ForwardOutput> LLMWorkerImpl::step(const ForwardInput& input) {
#if defined(USE_NPU)
  if (::xllm::LoadConfig::get_instance().enable_manual_loader()) {
    if (!enable_schedule_overlap() && options_.backend() == "llm") {
      aclrtStream current_stream =
          c10_npu::getCurrentNPUStream(device_.index()).stream();
      atb::Context* atb_context =
          const_cast<atb::Context*>(context_.get_atb_context());
      atb_context->SetExecuteStream(current_stream);
    } else {
      SET_ATB_EXECUTE_STREAM(compute_stream_, device_, context_);
      wait_input_ready_events(input, *compute_stream_);
      return step_internal(input, ForwardSyncPolicy::LEGACY);
    }
  }
#endif

  std::unique_ptr<Stream> stream = device_.current_stream();
  wait_input_ready_events(input, *stream);
  return step_internal(input, ForwardSyncPolicy::LEGACY);
}

folly::SemiFuture<std::optional<ForwardOutput>>
LLMWorkerImpl::step_async_no_sync(const ForwardInput& input) {
  CHECK(!enable_schedule_overlap())
      << "step_async_no_sync is only supported for non-overlap workers";
  ForwardInput input_on_device;

  prepare_work_before_execute(input, input_on_device);

  folly::Promise<std::optional<ForwardOutput>> promise;
  auto future = promise.getSemiFuture();
  threadpool_.schedule([this,
                        input = std::move(input_on_device),
                        promise = std::move(promise)]() mutable {
    // hierarchy temporarily disabled during the block-manager refactor
    // if (hierarchy_kv_cache_transfer_ != nullptr) {
    //   hierarchy_kv_cache_transfer_->set_layer_synchronizer(input.input_params);
    // }

    const auto output = this->step_no_sync(input);
    promise.setValue(output);
  });
  return future;
}

std::optional<ForwardOutput> LLMWorkerImpl::step_for_schedule_overlap(
    const ForwardInput& input) {
#if defined(USE_NPU)
  // Restore live recurrent-state slots from saved checkpoints here (worker
  // thread, on compute_stream_) instead of in prepare_work_before_execute on
  // prepare_stream_. The single-threaded worker pool guarantees the previous
  // chunk's forward kernels are already enqueued on compute_stream_ before
  // this task runs, so the restore copy is automatically stream-ordered
  // after those writes without needing a cross-stream barrier.
  if (has_linear_attention_layers(context_.get_model_args())) {
    c10::StreamGuard restore_guard = compute_stream_->set_stream_guard();
    auto& mutable_params = const_cast<ModelInputParams&>(input.input_params);
    restore_linear_state_slots(kv_caches_,
                               mutable_params.linear_state_cache_ops,
                               mutable_params.parallel.has_initial_state);
  }
#endif
  std::optional<ForwardOutput> output =
      execute_no_sync_on_stream(input, *compute_stream_);
#if defined(USE_NPU)
  // Single-process schedule overlap publishes this step's sample outputs to
  // last_step_output_ for consumption by other threads (the engine-facing
  // device->host copy in forward_output_to_raw runs on a folly executor thread
  // with no NPU stream binding, and cannot be ordered against compute_stream_).
  // The forward ran NO_SYNC, so without a barrier here those consumers race the
  // still-running compute-stream kernels and read reused device memory, which
  // aborts with an NPU MTE out-of-range error. Multi-process serializes through
  // the RPC/shm response so it never hit this and keeps its overlap unchanged.
  // Sync compute_stream_ before the output leaves this device-bound worker
  // thread; the next step's prepare still overlaps on prepare_stream_.
  if (options_.enable_single_process() && compute_stream_ != nullptr) {
    compute_stream_->synchronize();
  }
#endif
  return output;
}

ForwardInput
LLMWorkerImpl::update_input_by_last_step_output_for_schedule_overlap(
    ForwardInput& input) {
  c10::StreamGuard stream_guard = compute_stream_->set_stream_guard();
  CHECK(compute_stream_->wait_event(last_step_output_.ready_event))
      << "failed to wait last step output ready event";
  return update_input_by_last_step_output(input);
}

std::optional<ForwardOutput> LLMWorkerImpl::step_internal(
    const ForwardInput& input,
    ForwardSyncPolicy sync_policy,
    bool record_ready_event) {
  MULTI_MODEL_STEP_LOCK(::xllm::KVCacheConfig::get_instance().enable_xtensor());

  Timer timer;
  auto& sampling_params = input.sampling_params;

  KVTransferCompletion kv_transfers;

  if (options_.kv_cache_transfer_mode() == "PUSH" &&
      !input.transfer_kv_infos.empty()) {
#if defined(USE_NPU)
    std::shared_ptr<NPULayerSynchronizerImpl> layer_synchronizer =
        std::make_shared<NPULayerSynchronizerImpl>(
            context_.get_model_args().n_layers());
#elif defined(USE_MLU)
    std::shared_ptr<MLULayerSynchronizerImpl> layer_synchronizer =
        std::make_shared<MLULayerSynchronizerImpl>(
            context_.get_model_args().n_layers());
#elif defined(USE_DCU)
    std::shared_ptr<DCULayerSynchronizerImpl> layer_synchronizer =
        std::make_shared<DCULayerSynchronizerImpl>(
            context_.get_model_args().n_layers());
#endif
#if defined(USE_NPU) || defined(USE_MLU) || defined(USE_DCU)
    const_cast<ModelInputParams*>(&(input.input_params))
        ->parallel.layer_synchronizer = layer_synchronizer;

    kv_transfers.add(
        kv_cache_transfer_->push_kv_blocks_async(input.transfer_kv_infos,
                                                 context_.get_parallel_args(),
                                                 layer_synchronizer,
                                                 is_spec_draft_));
#endif
  }
  auto wait_kv_push = [&kv_transfers]() {
    CHECK(kv_transfers.wait()) << "KV cache push failed";
  };
  if (::xllm::EPLBConfig::get_instance().enable_eplb()) {
    eplb_executor_->eplb_execute(input.input_params.expert.eplb_info);
  }

  // call model executor forward to get hidden states
  auto model_output = model_executor_->forward(
      input.token_ids, input.positions, kv_caches_, input.input_params);
  if (!model_output.hidden_states.defined()) {
    wait_kv_push();
    return std::nullopt;
  }

  torch::Tensor logits;
  torch::Tensor selected_hidden_from_lm_head;
  if (sampling_params.selected_token_idxes.defined()) {
    torch::Tensor selected_token_idxes = sampling_params.selected_token_idxes;
    if (model_output.hidden_states.defined() &&
        selected_token_idxes.device() != model_output.hidden_states.device()) {
      selected_token_idxes = selected_token_idxes
                                 .to(model_output.hidden_states.device(),
                                     /*non_blocking=*/false)
                                 .contiguous();
    }
    if (uses_worker_cp_partition(options_)) {
      logits = model_->logits(model_output.hidden_states,
                              selected_token_idxes,
                              selected_hidden_from_lm_head);
    } else {
      logits = model_->logits(model_output.hidden_states, selected_token_idxes);
    }
  }

  ForwardOutput output;
  output.mtp_topk_state = std::move(model_output.mtp_topk_state);
  if (::xllm::EPLBConfig::get_instance().enable_eplb()) {
    output.expert_load_data = expert_load_data_;
    output.prepared_layer_id = eplb_executor_->get_ready_layer_id();
    if (output.prepared_layer_id != -1) {
      eplb_executor_->reset_ready_layer_id();
    }
  }

  if (!enable_schedule_overlap() && !driver_ && !dp_driver_ &&
      !options_.enable_speculative_decode()) {
    MULTI_MODEL_STEP_UNLOCK();
    if (sync_policy == ForwardSyncPolicy::NO_SYNC) {
      wait_kv_push();
      return std::nullopt;
    }
    int ret = device_.synchronize_default_stream();
    CHECK_EQ(ret, 0) << "synchronize_default_stream failed";
    wait_kv_push();
    if (::xllm::EPLBConfig::get_instance().enable_eplb()) {
      return output;
    }
    return std::nullopt;
  }

  // driver prepare model output
  if (sampling_params.selected_token_idxes.defined()) {
    output.logits = logits;
    output.do_sample = sampling_params.do_sample;
    output.logprobs = sampling_params.logprobs;
    output.max_top_logprobs = sampling_params.max_top_logprobs;
    if (!input.skip_sampling_for_logits_only) {
      auto sample_output = sampler_->forward(logits, sampling_params);

      // beam search kernel
      BeamSearchOutput beam_search_output;
      if (sampling_params.use_beam_search &&
          sampling_params.acc_logprob.defined() &&
          sampling_params.acc_logprob.numel() > 0) {
        beam_search_output =
            beam_searcher_->forward(sampling_params.acc_logprob,
                                    sample_output.top_tokens,
                                    sample_output.top_logprobs);
      }

      // set sample output to output
      output.sample_output = sample_output;
      // set beam search output to output
      output.beam_search_output = beam_search_output;
    }
  }

  if (options_.enable_speculative_decode()) {
    torch::Tensor embeddings;
    if (model_output.aux_hidden_states.defined()) {
      embeddings = model_output.aux_hidden_states;
    } else {
      embeddings = model_output.hidden_states;
    }
    if (!input.input_params.meta.batch_forward_type.is_decode() &&
        !is_spec_draft_) {
      // Target prefill keeps the full hidden in `embeddings` for the draft
      // input_embedding. Under CP this is the LOCAL token shard, whose rows
      // cannot be indexed by the CP all-gather-space selected_token_idxes.
      // Expose the LmHead-gathered per-sequence hidden (rows = num_seq)
      // separately so the embedding cache stores it without re-selecting on
      // the local shard.
      output.sample_output.embeddings = embeddings;
      if (uses_worker_cp_partition(options_) &&
          selected_hidden_from_lm_head.defined()) {
        output.sample_output.selected_embeddings = selected_hidden_from_lm_head;
      }
    } else if (sampling_params.selected_token_idxes.defined()) {
      if (uses_worker_cp_partition(options_)) {
        CHECK(selected_hidden_from_lm_head.defined())
            << "selected_hidden_from_lm_head must be defined when "
               "selected_token_idxes is defined.";
        output.sample_output.embeddings = selected_hidden_from_lm_head;
      } else {
        output.sample_output.embeddings = embeddings.index_select(
            /*dim=*/0, sampling_params.selected_token_idxes);
      }
    }
  }

  MULTI_MODEL_STEP_UNLOCK();
  bool should_sync_default_stream = true;
#if defined(USE_NPU)
  should_sync_default_stream =
      !can_skip_npu_graph_decode_sync(input.input_params);
#endif
  if (sync_policy == ForwardSyncPolicy::NO_SYNC) {
    wait_kv_push();
    output.retained_input = std::make_shared<ForwardInput>(input);
    if (enable_schedule_overlap() && record_ready_event) {
      output.ready_event = record_current_stream_event(device_);
    }
    return output;
  }
  if (should_sync_default_stream) {
    int ret = device_.synchronize_default_stream();
    CHECK_EQ(ret, 0) << "synchronize_default_stream failed";
  }

  wait_kv_push();

  COUNTER_ADD(execution_latency_seconds_model, timer.elapsed_seconds());
  if (should_sync_default_stream) {
    DeviceMonitor::get_instance().update_active_activation_memory(
        device_.index());
  }

  return output;
}

}  // namespace xllm

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

#include "vlm_worker_impl.h"

#include <c10/core/DeviceGuard.h>
#include <folly/Unit.h>
#include <folly/futures/Future.h>
#include <glog/logging.h>
#include <torch/torch.h>

#include <memory>
#include <optional>
#include <utility>

#include "common/metrics.h"
#include "core/framework/config/load_config.h"
#include "framework/kv_cache/kv_cache.h"
#include "framework/model/model_input_params.h"
#include "framework/state_dict/state_dict.h"
#include "models/model_registry.h"
#include "util/threadpool.h"
#include "util/timer.h"

namespace xllm {

namespace {

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

torch::Tensor choose_lm_head_selected_token_idxes(
    const torch::Tensor& selected_token_idxes,
    const ModelInputParams& input_params,
    const ParallelArgs& parallel_args,
    int64_t hidden_num_rows,
    const torch::Device& device) {
  const auto& mapping = parallel_args.mapping_data();
  if (!selected_token_idxes.defined() || selected_token_idxes.numel() == 0 ||
      mapping.empty() || !mapping.contains("attnDp") ||
      !mapping["attnDp"].contains("rank") ||
      input_params.parallel.dp_global_token_nums.size() <= 1 ||
      hidden_num_rows <= 0) {
    return selected_token_idxes;
  }

  const int64_t dp_rank = mapping["attnDp"]["rank"].get<int64_t>();
  CHECK_GE(dp_rank, 0) << "invalid attnDp rank";
  CHECK_LT(
      dp_rank,
      static_cast<int64_t>(input_params.parallel.dp_global_token_nums.size()))
      << "attnDp rank exceeds dp_global_token_nums";

  const int64_t local_selected_rows = selected_token_idxes.numel();
  const int64_t local_token_count =
      input_params.parallel.dp_global_token_nums.at(dp_rank);
  if (hidden_num_rows == local_token_count ||
      hidden_num_rows == local_selected_rows) {
    return selected_token_idxes;
  }

  int64_t dp_offset = 0;
  for (int64_t i = 0; i < dp_rank; ++i) {
    dp_offset += input_params.parallel.dp_global_token_nums[i];
  }

  torch::Tensor selected_cpu =
      selected_token_idxes.to(torch::dtype(torch::kLong).device(torch::kCPU));
  torch::Tensor logical_selected_cpu = selected_cpu + dp_offset;

  const auto& padding_idx = input_params.parallel.dp_ep_padding_data
                                .lm_head_skip_padding_token_indices();
  if (padding_idx.defined() && padding_idx.numel() > 0 &&
      hidden_num_rows > padding_idx.numel()) {
    torch::Tensor padding_cpu =
        padding_idx.to(torch::dtype(torch::kLong).device(torch::kCPU));
    const int64_t max_logical_selected =
        logical_selected_cpu.max().item<int64_t>();
    if (max_logical_selected < padding_cpu.numel()) {
      return padding_cpu.index_select(/*dim=*/0, logical_selected_cpu)
          .to(torch::dtype(selected_token_idxes.scalar_type()).device(device),
              /*non_blocking=*/false)
          .contiguous();
    }
  }

  if (dp_offset == 0) {
    return selected_token_idxes;
  }

  const int64_t max_selected_idx = selected_cpu.max().item<int64_t>();
  if (max_selected_idx + dp_offset >= hidden_num_rows) {
    return selected_token_idxes;
  }

  torch::Tensor remapped =
      selected_token_idxes.to(device, /*non_blocking=*/false).contiguous();
  return (remapped + dp_offset).to(remapped.scalar_type()).contiguous();
}

torch::Tensor select_hidden_rows(const torch::Tensor& hidden_states,
                                 const torch::Tensor& selected_idxes) {
  if (!hidden_states.defined() || !selected_idxes.defined() ||
      selected_idxes.numel() == 0) {
    return torch::Tensor();
  }
  torch::Tensor idxes = selected_idxes.to(
      torch::dtype(torch::kLong).device(hidden_states.device()),
      /*non_blocking=*/false);
  return hidden_states.index_select(/*dim=*/0, idxes).contiguous();
}

}  // namespace

VLMWorkerImpl::VLMWorkerImpl(const ParallelArgs& parallel_args,
                             const torch::Device& device,
                             const runtime::Options& options)
    : WorkerImpl(parallel_args, device, options) {
  device_.set_device();
}

bool VLMWorkerImpl::init_model(ModelContext& context) {
  CHECK(model_ == nullptr) << "Model is already initialized.";

  // initialize model
  context.set_encoder_embedding_mode(false);
  model_ = create_vlm_model(context);
  CHECK(model_ != nullptr) << "Failed to create model.";
  model_executor_ = std::make_unique<Executor>(
      model_.get(), context.get_model_args(), device_, options_);
  return true;
}

std::optional<ForwardOutput> VLMWorkerImpl::step(const ForwardInput& input) {
  if (::xllm::LoadConfig::get_instance().enable_manual_loader()) {
#if defined(USE_NPU)
    if (!enable_schedule_overlap() && options_.backend() == "vlm") {
      aclrtStream current_stream =
          c10_npu::getCurrentNPUStream(device_.index()).stream();
      atb::Context* atb_context =
          const_cast<atb::Context*>(context_.get_atb_context());
      atb_context->SetExecuteStream(current_stream);
      std::unique_ptr<Stream> stream = device_.current_stream();
      wait_input_ready_events(input, *stream);
      return step_internal(input, ForwardSyncPolicy::LEGACY);
    } else {
      SET_ATB_EXECUTE_STREAM(compute_stream_, device_, context_);
      wait_input_ready_events(input, *compute_stream_);
      return step_internal(input, ForwardSyncPolicy::LEGACY);
    }
#else
    std::unique_ptr<Stream> stream = device_.current_stream();
    wait_input_ready_events(input, *stream);
    return step_internal(input, ForwardSyncPolicy::LEGACY);
#endif
  }
  std::unique_ptr<Stream> stream = device_.current_stream();
  wait_input_ready_events(input, *stream);
  return step_internal(input, ForwardSyncPolicy::LEGACY);
}

std::optional<ForwardOutput> VLMWorkerImpl::execute_no_sync_on_stream(
    const ForwardInput& input,
    Stream& compute_stream) {
  const ForwardSyncPolicy sync_policy = ForwardSyncPolicy::NO_SYNC;
  c10::StreamGuard stream_guard = compute_stream.set_stream_guard();
  if (::xllm::LoadConfig::get_instance().enable_manual_loader()) {
#if defined(USE_NPU)
    if (!enable_schedule_overlap() && options_.backend() == "vlm") {
      aclrtStream current_acl_stream =
          c10_npu::getCurrentNPUStream(device_.index()).stream();
      atb::Context* atb_context =
          const_cast<atb::Context*>(context_.get_atb_context());
      atb_context->SetExecuteStream(current_acl_stream);
      wait_input_ready_events(input, compute_stream);
      return step_internal(input, sync_policy);
    } else {
      SET_ATB_EXECUTE_STREAM((&compute_stream), device_, context_);
      wait_input_ready_events(input, compute_stream);
      return step_internal(input, sync_policy);
    }
#else
    wait_input_ready_events(input, compute_stream);
    return step_internal(input, sync_policy);
#endif
  }
  wait_input_ready_events(input, compute_stream);
  return step_internal(input, sync_policy);
}

std::optional<ForwardOutput> VLMWorkerImpl::step_internal(
    const ForwardInput& input,
    ForwardSyncPolicy sync_policy) {
  Timer timer;
  const bool empty_shard =
      input.input_params.meta.num_sequences == 0 &&
      (!input.token_ids.defined() || input.token_ids.numel() == 0);
  if (empty_shard) {
    return ForwardOutput{};
  }

  // TODO guojinrong, to adapt multi stream parallel later
  // call model executor forward to get hidden states
  auto model_output = model_executor_->forward(
      input.token_ids, input.positions, kv_caches_, input.input_params);
  auto& sampling_params = input.sampling_params;
  const bool has_aux_hidden_states = model_output.aux_hidden_states.defined();
  torch::Tensor logits;
  torch::Tensor lm_head_selected_token_idxes;
  torch::Tensor selected_hidden_from_lm_head;
  torch::Tensor selected_aux_hidden;
  torch::Tensor selected_hidden_for_target_cache;
  if (sampling_params.selected_token_idxes.defined()) {
    lm_head_selected_token_idxes = choose_lm_head_selected_token_idxes(
        sampling_params.selected_token_idxes,
        input.input_params,
        context_.get_parallel_args(),
        model_output.hidden_states.size(0),
        model_output.hidden_states.device());
    if (options_.enable_speculative_decode()) {
      logits = model_->logits(model_output.hidden_states,
                              lm_head_selected_token_idxes,
                              selected_hidden_from_lm_head);
      if (has_aux_hidden_states) {
        selected_aux_hidden = select_hidden_rows(model_output.aux_hidden_states,
                                                 lm_head_selected_token_idxes);
      } else if (selected_hidden_from_lm_head.defined()) {
        selected_hidden_for_target_cache = selected_hidden_from_lm_head;
      } else if (!input.input_params.meta.batch_forward_type.is_decode() &&
                 !is_spec_draft_) {
        selected_hidden_for_target_cache = select_hidden_rows(
            has_aux_hidden_states ? model_output.aux_hidden_states
                                  : model_output.hidden_states,
            lm_head_selected_token_idxes);
      }
    } else {
      logits = model_->logits(model_output.hidden_states,
                              lm_head_selected_token_idxes);
    }
  }

  COUNTER_ADD(execution_latency_seconds_model, timer.elapsed_seconds());

  if (!enable_schedule_overlap() && !driver_ && !dp_driver_ &&
      !options_.enable_speculative_decode()) {
    if (sync_policy == ForwardSyncPolicy::LEGACY) {
      auto ret = device_.synchronize_default_stream();
      (void)ret;
    }
    return std::nullopt;
  }

  ForwardOutput output;
  if (sampling_params.selected_token_idxes.defined()) {
    auto sample_output = sampler_->forward(logits, sampling_params);
    output.logits = logits;
    COUNTER_ADD(execution_latency_seconds_sampling, timer.elapsed_seconds());

    // set sample output to output
    output.sample_output = sample_output;

    // carry over the sampling params
    output.do_sample = sampling_params.do_sample;
    output.logprobs = sampling_params.logprobs;
    output.max_top_logprobs = sampling_params.max_top_logprobs;
  }

  if (options_.enable_speculative_decode()) {
    torch::Tensor embeddings;
    if (has_aux_hidden_states) {
      embeddings = model_output.aux_hidden_states;
    } else {
      embeddings = model_output.hidden_states;
    }
    if (!input.input_params.meta.batch_forward_type.is_decode() &&
        !is_spec_draft_) {
      output.sample_output.embeddings = embeddings;
      if (selected_hidden_for_target_cache.defined()) {
        output.sample_output.selected_embeddings =
            selected_hidden_for_target_cache;
      }
    } else if (sampling_params.selected_token_idxes.defined()) {
      if (selected_aux_hidden.defined()) {
        output.sample_output.embeddings = selected_aux_hidden;
      } else if (selected_hidden_from_lm_head.defined()) {
        output.sample_output.embeddings = selected_hidden_from_lm_head;
      } else {
        output.sample_output.embeddings =
            select_hidden_rows(embeddings, lm_head_selected_token_idxes);
      }
    }
  }

  if (sync_policy == ForwardSyncPolicy::NO_SYNC) {
    output.retained_input = std::make_shared<ForwardInput>(input);
    if (enable_schedule_overlap()) {
      output.ready_event = record_current_stream_event(device_);
    }
    return output;
  }

  auto ret = device_.synchronize_default_stream();
  (void)ret;
  return output;
}

}  // namespace xllm

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

#include "core/runtime/speculative_worker_impl.h"

#include <type_traits>

#include "common/global_flags.h"
#include "core/framework/batch/decode_input_builder.h"
#include "core/framework/config/speculative_config.h"
#include "core/framework/constrained/json_object_grammar.h"
#include "core/framework/eplb/eplb_utils.h"
#include "core/framework/speculative/verify_layout.h"
#include "core/kv_cache/layout/kv_cache_shape.h"
#include "core/kv_cache/layout/seq_lens_layout.h"
#include "runtime/llm_worker_impl.h"
#include "runtime/vlm_worker_impl.h"
#include "util/slice.h"
#include "util/tensor_helper.h"
#include "util/timer.h"
#include "util/utils.h"

namespace xllm {

namespace {
template <typename Params>
bool should_run_speculative_decode_impl(const Params& params) {
  if (!params.meta.batch_forward_type.is_decode()) {
    return false;
  }

  const auto& dp_token_nums = params.parallel.dp_global_token_nums;
  const auto& dp_is_decode = params.parallel.dp_is_decode;
  if (dp_is_decode.empty()) {
    return dp_token_nums.size() <= 1;
  }
  if (dp_is_decode.size() != dp_token_nums.size()) {
    return false;
  }

  // Idle DP ranks (no scheduled tokens this step) must not veto speculative
  // decode for the active ranks. Under enable_graph=False these idle ranks keep
  // dp_is_decode=0 (the backfill in llm_engine only fires when enable_graph=
  // True), which made an all-of-ones check fail for any bs<dp_size batch and
  // silently fell back to the non-speculative path (validate never ran). Only
  // ranks that actually carry tokens gate the decision; require every such rank
  // to be in decode.
  bool any_active = false;
  for (size_t i = 0; i < dp_is_decode.size(); ++i) {
    if (dp_token_nums[i] == 0) {
      continue;  // idle rank: does not participate in the vote
    }
    any_active = true;
    if (dp_is_decode[i] != 1) {
      return false;
    }
  }
  return any_active;
}

template <typename Params>
void scale_speculative_parallel_token_counts_impl(Params& params,
                                                  int32_t multiplier) {
  scale_parallel_token_counts(params.parallel, multiplier);
  params.expert.eplb_decode_token_mask = eplb::expand_decode_token_mask(
      params.expert.eplb_decode_token_mask, multiplier);
}

}  // namespace

bool should_run_speculative_decode(const LlmModelParams& params) {
  return should_run_speculative_decode_impl(params);
}

bool should_run_speculative_decode(const VlmModelParams& params) {
  return should_run_speculative_decode_impl(params);
}

void scale_speculative_parallel_token_counts(LlmModelParams& params,
                                             int32_t multiplier) {
  scale_speculative_parallel_token_counts_impl(params, multiplier);
}

void scale_speculative_parallel_token_counts(VlmModelParams& params,
                                             int32_t multiplier) {
  scale_speculative_parallel_token_counts_impl(params, multiplier);
}

ForwardOutput finalize_verify_output(
    ForwardOutput target_output,
    SampleOutput val_output,
    const std::vector<JsonObjectGrammarState>& json_object_states) {
  const std::vector<int32_t> num_accepted_tokens =
      tensor_to_vector<int32_t>(val_output.num_accepted_tokens);
  target_output.spec_verify_stats =
      make_verify_stats(val_output.next_tokens,
                        /*per_seq_draft_counts=*/nullptr,
                        num_accepted_tokens,
                        constrained_mask(json_object_states));
  val_output.embeddings = torch::Tensor();
  target_output.sample_output = std::move(val_output);
  return target_output;
}

template <typename TargetInput>
SpeculativeWorkerImpl<TargetInput>::SpeculativeWorkerImpl(
    const ParallelArgs& parallel_args,
    const torch::Device& device,
    const runtime::Options& options,
    const runtime::Options& target_options,
    WorkerType worker_type)
    : WorkerImpl(parallel_args, device, options) {
  if (worker_type == WorkerType::LLM) {
    impl_ =
        std::make_unique<LLMWorkerImpl>(parallel_args, device, target_options);
  } else if (worker_type == WorkerType::VLM) {
    impl_ =
        std::make_unique<VLMWorkerImpl>(parallel_args, device, target_options);
  } else {
    LOG(FATAL) << "Unsupported speculative worker type: "
               << worker_type.to_string();
  }
}

template <typename TargetInput>
SpeculativeWorkerImpl<TargetInput>::~SpeculativeWorkerImpl() {
  drain_pending_execution();
  // Draft-model subclasses shut down the shared transfer's threadpool first
  // (see ~DraftModelSpecWorkerImpl); this base only releases refs.
  if (impl_ != nullptr) {
    impl_->clear_hierarchy_kv_cache_transfer();
  }
  clear_hierarchy_kv_cache_transfer();
}

template <typename TargetInput>
void SpeculativeWorkerImpl<TargetInput>::drain_pending_execution() {
  WorkerImpl::drain_pending_execution();
  if (impl_ != nullptr) {
    impl_->drain_pending_execution();
  }
}

template <typename TargetInput>
bool SpeculativeWorkerImpl<TargetInput>::init_model(
    const std::string& model_weights_path,
    int32_t random_seed,
    MasterStatus master_status) {
  // Base class only loads the target model.
  bool result = true;
  CHECK(impl_ != nullptr);
  if (impl_->get_status() == WorkerImpl::Status::UNINITIALIZED) {
    result = impl_->WorkerImpl::init_model(
        model_weights_path, random_seed, master_status);
    if (result) {
      dtype_ = impl_->dtype();
    }
  }
  return result;
}

template <typename TargetInput>
bool SpeculativeWorkerImpl<TargetInput>::allocate_kv_cache(
    const KVCacheShape& kv_cache_shape) {
  return impl_->allocate_kv_cache(kv_cache_shape);
}

#if defined(USE_NPU)
template <typename TargetInput>
bool SpeculativeWorkerImpl<TargetInput>::allocate_kv_cache_with_transfer(
    const KVCacheShape& kv_cache_shape) {
  return impl_->allocate_kv_cache_with_transfer(kv_cache_shape);
}
#endif

template <typename TargetInput>
std::optional<ForwardOutput> SpeculativeWorkerImpl<TargetInput>::step(
    const TargetInput& input) {
  if constexpr (std::is_same_v<TargetInput, VlmForwardInput>) {
    if (input.runtime.input_host_buffer_has_layout) {
      // Rebuild host views before speculative code derives text draft inputs.
      VlmForwardInput unpacked_input;
      if (input.runtime.device_tensors_ready) {
        unpacked_input = input.clone();
        unpacked_input.runtime.input_host_buffer_has_layout = false;
      } else {
        CHECK(detail::unpack_from_input_host_buffer(
            input, torch::Device(torch::kCPU), unpacked_input))
            << "failed to unpack the packed VLM input host buffer";
      }
      return step(unpacked_input);
    }
  }
  auto& mutable_params = input.input_params;
  set_hierarchy_layer_synchronizer(mutable_params);
  const bool run_speculative_decode =
      should_run_speculative_decode(input.input_params);
  if (input.input_params.meta.num_sequences == 0 ||
      input.token_ids.numel() == 0) {
    if (input.input_params.meta.batch_forward_type.is_decode() &&
        !run_speculative_decode) {
      // Realign a non-speculative decode rank to EMPTY so every rank agrees
      // on the batch type. Only this branch needs its own copy.
      TargetInput aligned_input = input.clone();
      aligned_input.input_params.meta.batch_forward_type =
          BatchForwardType::EMPTY;
      return step_empty(aligned_input);
    }
    return step_empty(input);
  }

  if (run_speculative_decode) {
    return step_decode(input);
  }
  return step_prefill(input);
}

template <typename TargetInput>
bool SpeculativeWorkerImpl<TargetInput>::prepare_model_decode_continuation(
    TargetInput& inputs,
    const ForwardOutput& prior_output) {
  // Only decode batches need draft input rebuilding.
  auto& input_params = inputs.input_params;
  const int32_t num_sequences = input_params.meta.num_sequences;
  const int32_t block_size = options_.block_size();

  const torch::Tensor last_token_ids =
      to_cpu_contiguous(prior_output.sample_output.next_tokens.flatten());
  const Slice<int64_t> last_tokens_ids_slice =
      tensor_slice<int64_t>(last_token_ids);

  // Determine how many tokens were decoded in the last step
  // If the output is 2D, it means multiple tokens were generated per sequence
  int32_t last_step_decode_num = 1;
  if (prior_output.sample_output.next_tokens.dim() == 2) {
    last_step_decode_num = prior_output.sample_output.next_tokens.size(1);
  }

  DecodeBuildBuffers buf;
  DecodeBatchContext row_ctx = make_decode_batch_context(inputs);
  reserve_entries(buf,
                  row_ctx,
                  static_cast<size_t>(num_sequences),
                  DecodeFields::kSlotsAndKvLen);

  for (int32_t seq_id = 0; seq_id < num_sequences; ++seq_id) {
    const TokenWithOffset resolved = resolve_token_with_position_offset(
        token_at(row_ctx, seq_id), last_tokens_ids_slice, last_step_decode_num);
    QueryEntrySpec row;
    row.seq_id = seq_id;
    row.token_id = resolved.token_id;
    row.position_offset = resolved.position_offset;
    append_decode_entry(row_ctx, row, block_size, buf);
  }

  set_token_position_tensors(inputs, buf.out_token_ids, buf.out_positions);
  commit_kv_and_slots(input_params, buf);
  input_params.attention.rebuild_device_buffer(device_);

  return true;
}

template <typename TargetInput>
void SpeculativeWorkerImpl<TargetInput>::prepare_validate_inputs(
    const TargetInput& input,
    TargetInput& validate_input) {
  // Clear the transform fields on the host copy first so .to() skips them;
  // expand_sampling_params_for_verify later installs the validate-row idxes.
  // The JSON grammar states are not consumed by the logits-only validate
  // forward (the verify layouts are built from the original input), so they
  // are moved aside instead of deep-copied through clone() and .to().
  TargetInput host_input = input.clone();
  std::vector<JsonObjectGrammarState> json_states =
      std::move(host_input.json_object_states);
  std::vector<JsonObjectGrammarSnapshot> json_snapshots =
      std::move(host_input.json_object_state_snapshots);
  clear_all_sampling_fields(host_input.sampling_params);
  validate_input = host_input.to(device_, dtype_);
  validate_input.json_object_states = std::move(json_states);
  validate_input.json_object_state_snapshots = std::move(json_snapshots);
  validate_input.skip_sampling_for_logits_only = true;
  validate_input.runtime.device_tensors_ready = false;
  auto& input_params = validate_input.input_params;
  const int32_t num_speculative_tokens = options_.num_speculative_tokens();
  const int32_t num_sequences = input_params.meta.num_sequences;
  const int32_t num_val_tokens = num_speculative_tokens + 1;
  const int32_t total_num_val_tokens = num_sequences * num_val_tokens;
  const int32_t block_size = options_.block_size();
  // Hybrid targets (for example Qwen3.8 GDN) mark validation as spec-verify
  // before entering this generic builder.  They must keep one sequence row
  // with an N+1-token query so recurrent state is checkpointed and committed
  // by the model's spec-verify kernel instead of being expanded into N+1
  // independent decode rows.
  const bool use_chunked_spec_verify =
      ::xllm::SpeculativeConfig::get_instance().enable_atb_spec_kernel() ||
      input.input_params.is_spec_verify;
  DecodeBatchContext row_ctx = make_decode_batch_context(input);

  const ExpandDecodePlan plan = make_verify_plan(
      num_val_tokens,
      block_size,
      use_chunked_spec_verify ? QueryLayout::kChunked : QueryLayout::kFlat,
      /*leading_zero_cu=*/input.input_params.is_spec_verify);
  DecodeBuildBuffers buf = expand_decode_inputs(row_ctx, plan);

  commit_decode_input(validate_input, buf, plan, device_);

  // Clearing here drops the ~15 dead per-token tensors the clone carried in.
  expand_sampling_params_for_verify(validate_input.sampling_params,
                                    num_val_tokens,
                                    total_num_val_tokens,
                                    device_);

  scale_speculative_parallel_token_counts(input_params, num_val_tokens);
  validate_input.runtime.device_tensors_ready = true;
}

template <typename TargetInput>
void SpeculativeWorkerImpl<TargetInput>::restore_json_object_states(
    TargetInput& input) {
  impl_->restore_json_object_states(input);
}
template class SpeculativeWorkerImpl<LlmForwardInput>;
template class SpeculativeWorkerImpl<VlmForwardInput>;

}  // namespace xllm

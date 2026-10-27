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

#include <algorithm>
#include <type_traits>

#include "common/global_flags.h"
#include "core/common/types.h"
#include "core/framework/batch/decode_input_builder.h"
#include "core/framework/config/execution_config.h"
#include "core/framework/config/speculative_config.h"
#include "core/framework/constrained/json_object_grammar.h"
#include "core/framework/eplb/eplb_utils.h"
#include "core/framework/model/model_type.h"
#include "core/framework/speculative/draft_config.h"
#include "core/framework/speculative/verify_layout.h"
#include "core/kv_cache/layout/kv_cache_shape.h"
#include "core/util/tensor_helper.h"
#include "framework/model_loader/model_loader.h"
#include "runtime/decode_graph_bucket.h"
#include "runtime/llm_worker_impl.h"
#include "runtime/vlm_worker_impl.h"
#include "util/slice.h"
#include "util/timer.h"
#include "util/utils.h"

namespace xllm {

namespace {
template <typename Input>
void prepare_dummy_input_impl(Input& input,
                              const ModelContext& context,
                              const torch::Device& device) {
  auto& params = input.input_params;
  const auto& args = context.get_model_args();
  const auto& parallel = context.get_parallel_args();
  const bool idle_block =
      params.meta.num_sequences == 0 && context.num_speculative_tokens() > 0;
  const bool empty =
      params.meta.num_sequences == 0 &&
      (!input.token_ids.defined() || input.token_ids.numel() == 0);
  if (!(empty || idle_block) || params.meta.batch_forward_type.is_empty() ||
      (parallel.dp_size() <= 1 && parallel.ep_size() <= 1 &&
       parallel.mapping_data().empty())) {
    return;
  }
  const int32_t tokens = context.num_speculative_tokens() + 1;
  build_dummy_forward_input(input, tokens, device);
  if (context.num_speculative_tokens() > 0 &&
      has_linear_attention_layers(args)) {
    params.num_accepted_tokens_host.assign(1, 1);
    params.num_accepted_tokens =
        torch::ones({1}, input.token_ids.options().dtype(torch::kInt32));
  }
  if (idle_block) {
    params.meta.batch_forward_type = BatchForwardType::CHUNKED_PREFILL;
    params.meta.q_max_seq_len = tokens;
    params.meta.kv_max_seq_len = tokens;
    params.embedding.input_embedding = torch::Tensor();
    params.attn_metadata.reset();
    params.attention = {};
    params.attention.host.q_seq_lens = {tokens};
    params.attention.host.q_cu_seq_lens = {0, tokens};
    params.attention.host.kv_seq_lens = {tokens};
    params.attention.host.kv_cache_tokens_nums = {0};
    params.attention.host.new_cache_slots.assign(tokens, -1);
    params.attention.rebuild_device_buffer(device);
  }
}
}  // namespace

void prepare_dummy_input(LlmForwardInput& input,
                         const ModelContext& context,
                         const torch::Device& device) {
  prepare_dummy_input_impl(input, context, device);
}

void prepare_dummy_input(VlmForwardInput& input,
                         const ModelContext& context,
                         const torch::Device& device) {
  prepare_dummy_input_impl(input, context, device);
}

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
void scale_speculative_parallel_token_counts_impl(
    Params& params,
    int32_t num_speculative_tokens) {
  scale_parallel_token_counts(params.parallel, num_speculative_tokens + 1);
  params.expert.eplb_decode_token_mask = eplb::expand_decode_token_mask(
      params.expert.eplb_decode_token_mask, num_speculative_tokens + 1);
}

}  // namespace

bool should_run_speculative_decode(const LlmModelParams& params) {
  return should_run_speculative_decode_impl(params);
}

bool should_run_speculative_decode(const VlmModelParams& params) {
  return should_run_speculative_decode_impl(params);
}

void scale_speculative_parallel_token_counts(LlmModelParams& params,
                                             int32_t num_speculative_tokens) {
  scale_speculative_parallel_token_counts_impl(params, num_speculative_tokens);
}

void scale_speculative_parallel_token_counts(VlmModelParams& params,
                                             int32_t num_speculative_tokens) {
  scale_speculative_parallel_token_counts_impl(params, num_speculative_tokens);
}

template <typename TargetInput>
SpeculativeWorkerImpl<TargetInput>::SpeculativeWorkerImpl(
    const ParallelArgs& parallel_args,
    const torch::Device& device,
    const runtime::Options& options,
    const runtime::Options& target_options,
    WorkerType worker_type)
    : WorkerImpl(parallel_args, device, options, parallel_args.dp_size() > 1) {
  if (worker_type == WorkerType::LLM) {
    impl_ = std::make_unique<LLMWorkerImpl>(
        parallel_args, device, target_options, parallel_args.dp_size() > 1);
  } else if (worker_type == WorkerType::VLM) {
    impl_ = std::make_unique<VLMWorkerImpl>(
        parallel_args, device, target_options, parallel_args.dp_size() > 1);
  } else {
    LOG(FATAL) << "Unsupported speculative worker type: "
               << worker_type.to_string();
  }
  impl_->set_input_preparer(
      std::function<void(TargetInput&)>([this](TargetInput& input) {
        ::xllm::prepare_dummy_input(input, impl_->context_, device_);
#if defined(USE_NPU)
        prepare_target_padding(input.input_params);
#endif
      }));
}

#if defined(USE_NPU)
template <typename TargetInput>
template <typename Params>
void SpeculativeWorkerImpl<TargetInput>::prepare_target_padding(
    Params& params) {
  if (!impl_->uses_npu_dp_ep_padding()) {
    return;
  }
  const auto& tokens = params.parallel.dp_global_token_nums;
  const auto& raw_tokens = params.parallel.raw_dp_global_token_nums;
  const bool graph_decode =
      params.meta.batch_forward_type.is_decode() &&
      ExecutionConfig::get_instance().enable_graph() && tokens.size() > 1 &&
      params.parallel.dp_is_decode.size() == tokens.size() &&
      std::all_of(params.parallel.dp_is_decode.begin(),
                  params.parallel.dp_is_decode.end(),
                  [](int32_t decode) { return decode != 0; });
  if (graph_decode) {
    const int32_t max_tokens = *std::max_element(tokens.begin(), tokens.end());
    const int32_t bucket =
        static_cast<int32_t>(runtime::get_decode_graph_token_bucket(
            max_tokens,
            ExecutionConfig::get_instance()
                .enable_graph_mode_decode_no_padding()));
    if (std::any_of(tokens.begin(), tokens.end(), [bucket](int32_t count) {
          return count != bucket;
        })) {
      const auto padded_tokens =
          runtime::get_decode_graph_dp_token_counts(tokens, bucket);
      params.parallel.dp_ep_padding_data = impl_->build_dp_ep_padding_data(
          padded_tokens,
          raw_tokens.empty() ? tokens : raw_tokens,
          params.meta.batch_forward_type.no_decode());
      return;
    }
  }
  impl_->prepare_dp_ep_padding(params);
}
#endif

template <typename TargetInput>
SpeculativeWorkerImpl<TargetInput>::~SpeculativeWorkerImpl() {
  drain_pending_execution();
  // Draft-model subclasses shut down the shared transfer's threadpool first
  // (see ~DraftModelSpecWorkerImpl); this base only releases refs.
  impl_->clear_hierarchy_kv_cache_transfer();
  clear_hierarchy_kv_cache_transfer();
}

template <typename TargetInput>
void SpeculativeWorkerImpl<TargetInput>::drain_pending_execution() {
  WorkerImpl::drain_pending_execution();
  impl_->drain_pending_execution();
}

template <typename TargetInput>
bool SpeculativeWorkerImpl<TargetInput>::init_model(
    const std::string& model_weights_path,
    int32_t random_seed,
    MasterStatus master_status) {
  // Base class only loads the target model.
  bool result = true;
  if (impl_->get_status() == WorkerImpl::Status::UNINITIALIZED) {
    auto loader = ModelLoader::create(model_weights_path);
    auto args = loader->model_args();
    configure_speculative_model_args(args, impl_->options());
    auto tokenizer = impl_->load_tokenizer(*loader);
    auto context =
        impl_->load_model_config(std::move(args),
                                 loader->quant_args(),
                                 impl_->options().num_speculative_tokens(),
                                 tokenizer.get());
    result = impl_->init_model(std::move(loader),
                               std::move(context),
                               std::move(tokenizer),
                               random_seed,
                               master_status);
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
  DecodeBatchContext batch_ctx = make_decode_batch_context(inputs);
  prepare_decode_buffers(buf,
                         batch_ctx,
                         static_cast<size_t>(num_sequences),
                         /*emit_kv_len=*/true,
                         /*emit_block_tables=*/false);

  for (int32_t seq_id = 0; seq_id < num_sequences; ++seq_id) {
    const auto [token_id, position_offset] =
        resolve_token_with_position_offset(batch_ctx.token_ids[seq_id],
                                           last_tokens_ids_slice,
                                           last_step_decode_num);
    DecodeTokenInput token_input;
    token_input.req_id = seq_id;
    token_input.token_id = token_id;
    token_input.position_offset = position_offset;
    append_decode_token(batch_ctx, token_input, block_size, buf);
  }

  set_token_position_tensors(inputs, buf.out_token_ids, buf.out_positions);
  update_kv_lengths_and_slots(input_params, buf);
  input_params.attention.rebuild_device_buffer(device_);

  return true;
}

ForwardOutput finalize_verify_output(
    ForwardOutput target_output,
    SampleOutput val_output,
    const std::vector<JsonObjectGrammarState>& json_object_states) {
  std::vector<char> constrained_mask;
  constrained_mask.reserve(json_object_states.size());
  for (const JsonObjectGrammarState& state : json_object_states) {
    constrained_mask.push_back(static_cast<char>(state.initialized()));
  }
  if (val_output.next_tokens.is_cpu()) {
    target_output.spec_verify_stats =
        make_verify_stats(val_output.next_tokens, constrained_mask);
  } else {
    target_output.is_verify_stats_deferred = true;
    target_output.constrained_mask = std::move(constrained_mask);
  }
  // The verify output replaces sample_output wholesale, so only its own
  // embeddings need clearing.
  val_output.embeddings = torch::Tensor();
  target_output.sample_output = std::move(val_output);
  return target_output;
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
  const bool use_chunked_spec_verify = input.input_params.is_spec_verify;
  DecodeBatchContext batch_ctx = make_decode_batch_context(input);

  DecodeExpansionOptions decode_options = make_verify_layout(
      num_speculative_tokens, block_size, use_chunked_spec_verify);
  DecodeBuildBuffers buf = expand_decode_inputs(batch_ctx, decode_options);

  build_decode_input(validate_input, buf, decode_options, device_);

  // Clearing here drops the ~15 dead per-token tensors the clone carried in.
  expand_sampling_params_for_verify(validate_input.sampling_params,
                                    num_speculative_tokens,
                                    total_num_val_tokens,
                                    device_);

  scale_speculative_parallel_token_counts(input_params, num_speculative_tokens);
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

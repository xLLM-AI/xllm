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

#include "common/global_flags.h"
#include "core/common/types.h"
#include "core/framework/batch/decode_input_builder.h"
#include "core/framework/config/speculative_config.h"
#include "core/framework/constrained/json_object_grammar.h"
#include "core/framework/eplb/eplb_utils.h"
#include "core/framework/model/model_args.h"
#include "core/framework/speculative/verify_layout.h"
#include "core/kv_cache/layout/kv_cache_shape.h"
#include "framework/model_loader/model_loader.h"
#include "runtime/decode_graph_bucket.h"
#include "runtime/llm_worker_impl.h"
#include "runtime/vlm_worker_impl.h"

namespace xllm {

template <typename Input>
void prepare_dummy_input(Input& input, const WorkerImpl& worker) {
  auto& params = input.input_params;
  const auto& context = worker.context_;
  const auto& parallel = context.get_parallel_args();
  if (params.meta.num_sequences != 0 ||
      params.meta.batch_forward_type.is_empty() ||
      (parallel.dp_size() <= 1 && parallel.ep_size() <= 1 &&
       parallel.mapping_data().empty())) {
    return;
  }
  const auto& options = worker.options();
  const int32_t num_speculative_tokens = options.num_speculative_tokens();
  int32_t query_width = num_speculative_tokens + 1;
  if (options.is_draft_engine()) {
    const auto& algorithm = options.speculative_algorithm();
    if (SpeculativeConfig::is_dspark_algorithm(algorithm)) {
      query_width = num_speculative_tokens;
    } else if (SpeculativeConfig::is_eagle_family(algorithm)) {
      query_width = 1;
    }
  }
  if (query_width == 1 && input.token_ids.defined() &&
      input.token_ids.numel() != 0) {
    return;
  }
  const auto device = context.get_tensor_options().device();
  build_dummy_forward_input(input, query_width, device);
  if (query_width > 1) {
    if (has_linear_attention_layers(context.get_model_args())) {
      params.num_accepted_tokens_host = {1};
      params.num_accepted_tokens =
          torch::ones({1}, input.token_ids.options().dtype(torch::kInt32));
    }
    params.meta.batch_forward_type = BatchForwardType::CHUNKED_PREFILL;
    params.meta.q_max_seq_len = query_width;
    params.meta.kv_max_seq_len = query_width;
    params.embedding.input_embedding = torch::Tensor();
    params.attn_metadata.reset();
    params.attention = {};
    params.attention.host.q_seq_lens = {query_width};
    params.attention.host.q_cu_seq_lens = {0, query_width};
    params.attention.host.kv_seq_lens = {query_width};
    params.attention.host.kv_cache_tokens_nums = {0};
    params.attention.host.new_cache_slots.assign(query_width,
                                                 /*value=*/-1);
    params.attention.rebuild_device_buffer(device);
  }
}

template <typename Params>
void scale_decode_parallel_token_counts(Params& params, int32_t query_width) {
  scale_parallel_token_counts(params.parallel, query_width);
  params.expert.eplb_decode_token_mask = eplb::expand_decode_token_mask(
      params.expert.eplb_decode_token_mask, query_width);
}

template void prepare_dummy_input(LlmForwardInput&, const WorkerImpl&);
template void scale_decode_parallel_token_counts(LlmModelParams&, int32_t);
template void scale_decode_parallel_token_counts(VlmModelParams&, int32_t);

template <typename TargetInput>
void SpeculativeWorkerImpl<TargetInput>::apply_target_model_overrides(
    ModelArgs&) const {}

template <typename TargetInput>
SpeculativeWorkerImpl<TargetInput>::SpeculativeWorkerImpl(
    const ParallelArgs& parallel_args,
    const torch::Device& device,
    const runtime::Options& options,
    runtime::Options target_options,
    WorkerType worker_type)
    : WorkerImpl(parallel_args, device, options, parallel_args.dp_size() > 1) {
  target_options.enable_schedule_overlap(/*enable_schedule_overlap=*/false);
  auto prepare_input = [this](TargetInput& input) {
    ::xllm::prepare_dummy_input(input, *impl_);
#if defined(USE_NPU)
    if (!impl_->uses_npu_dp_ep_padding()) {
      return;
    }
    auto& params = input.input_params;
    const auto& tokens = params.parallel.dp_global_token_nums;
    const auto& raw_tokens = params.parallel.raw_dp_global_token_nums;
    const auto& target_options = impl_->options();
    const bool graph_decode =
        params.meta.batch_forward_type.is_decode() &&
        params.meta.q_max_seq_len <= 1 && target_options.enable_graph() &&
        tokens.size() > 1 &&
        params.parallel.dp_is_decode.size() == tokens.size() &&
        std::all_of(params.parallel.dp_is_decode.begin(),
                    params.parallel.dp_is_decode.end(),
                    [](int32_t decode) { return decode != 0; });
    if (graph_decode) {
      const int32_t max_tokens =
          *std::max_element(tokens.begin(), tokens.end());
      const int32_t bucket =
          static_cast<int32_t>(runtime::get_decode_graph_token_bucket(
              max_tokens,
              target_options.enable_graph_mode_decode_no_padding()));
      if (std::any_of(tokens.begin(), tokens.end(), [bucket](int32_t count) {
            return count != bucket;
          })) {
        const auto padded_tokens =
            runtime::get_decode_graph_dp_token_counts(tokens, bucket);
        params.parallel.dp_ep_padding_data = impl_->build_dp_ep_padding_data(
            padded_tokens,
            raw_tokens.empty() ? tokens : raw_tokens,
            /*is_prefill=*/false);
        return;
      }
    }
    impl_->prepare_dp_ep_padding(ModelParamsView(params));
#endif
  };
  auto make_impl = [&]<typename BaseWorker>() {
    return std::make_unique<
        SpeculativeModelWorkerImpl<BaseWorker,
                                   TargetInput,
                                   decltype(prepare_input)>>(
        std::move(prepare_input),
        parallel_args,
        device,
        target_options,
        parallel_args.dp_size() > 1);
  };
  if (worker_type == WorkerType::LLM) {
    impl_ = make_impl.template operator()<LLMWorkerImpl>();
  } else if (worker_type == WorkerType::VLM) {
    impl_ = make_impl.template operator()<VLMWorkerImpl>();
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
  impl_->clear_hierarchy_kv_transfer();
  clear_hierarchy_kv_transfer();
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
    apply_target_model_overrides(args);
    auto tokenizer = loader->tokenizer();
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
std::shared_ptr<const JsonObjectGrammar>
SpeculativeWorkerImpl<TargetInput>::ensure_json_object_grammar(
    bool reasoning_enabled) {
  return impl_->ensure_json_object_grammar(reasoning_enabled);
}

template <typename TargetInput>
std::optional<ForwardOutput> SpeculativeWorkerImpl<TargetInput>::step(
    const TargetInput& input) {
  auto& mutable_params = input.input_params;
  set_hierarchy_layer_synchronizer(mutable_params);
  const bool is_decode = mutable_params.meta.batch_forward_type.is_decode();
  bool run_speculative_decode = false;
  if (is_decode) {
    const auto& dp_token_nums = mutable_params.parallel.dp_global_token_nums;
    const auto& dp_is_decode = mutable_params.parallel.dp_is_decode;
    if (dp_is_decode.empty()) {
      run_speculative_decode = dp_token_nums.size() <= 1;
    } else if (dp_is_decode.size() == dp_token_nums.size()) {
      for (size_t i = 0; i < dp_is_decode.size(); ++i) {
        if (dp_token_nums[i] == 0) {
          continue;
        }
        run_speculative_decode = dp_is_decode[i] == 1;
        if (!run_speculative_decode) {
          break;
        }
      }
    }
  }
  if (input.input_params.meta.num_sequences == 0 ||
      input.token_ids.numel() == 0) {
    if (is_decode && !run_speculative_decode) {
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
    TargetInput& validate_input,
    bool expand_decode_metadata_per_token) {
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
  DecodeBatchContext batch_ctx = make_decode_batch_context(input);

  const DecodeExpansionOptions decode_options{
      {.query_width = num_val_tokens,
       .per_request = !expand_decode_metadata_per_token},
      block_size};
  DecodeBuildBuffers buf = expand_decode_inputs(batch_ctx, decode_options);

  build_decode_input(validate_input, buf, decode_options, device_);

  expand_sampling_params_for_verify(validate_input.sampling_params,
                                    num_speculative_tokens,
                                    total_num_val_tokens,
                                    device_);

  scale_decode_parallel_token_counts(input_params, decode_options.query_width);
  validate_input.runtime.device_tensors_ready = true;
}

template class SpeculativeWorkerImpl<LlmForwardInput>;
template class SpeculativeWorkerImpl<VlmForwardInput>;

}  // namespace xllm

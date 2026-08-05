/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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

#pragma once

#include <glog/logging.h>
#include <torch/torch.h>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "framework/model_loader.h"
#include "framework/parallel_state/process_group.h"
#include "framework/sampling/dspark_sampler.h"
#include "framework/state_dict/state_dict.h"
#include "models/llm/npu/qwen3_dflash.h"
#include "models/model_registry.h"

namespace xllm::npu::model {

// Low-rank Markov sampler for DSpark block-diffusion drafting. It owns the
// Markov weights and the sequential sampling loop, while the draft worker owns
// the backbone forward and logits lifecycle.
//
// Weights (no bias, no activation):
//   markov_w1: [vocab_size, markov_rank]        embedded by the target-vocab
//                                               prev token id.
//   markov_w2: [draft_vocab_size, markov_rank]  projects the embed to a
//                                               draft-vocab bias.
//   bias(prev) = markov_w1[prev] @ markov_w2^T   -> [num_reqs, draft_vocab]
//
// Both weights are replicated on every TP rank. The Markov loop is sequential,
// so sharding would add an all-reduce and a full-vocab gather for every draft
// position. The backbone LM head still performs its normal TP gather once for
// the whole block before this sampler runs. Random and mixed sampling retain a
// token broadcast per position because xLLM sampling uses rank-local RNG state.
class DSparkBlockDraftSampler final : public BlockDraftSampler {
 public:
  DSparkBlockDraftSampler() = default;

  void initialize(const torch::TensorOptions& options,
                  int64_t markov_rank,
                  ProcessGroup* process_group) {
    tensor_options_ = options;
    markov_rank_ = markov_rank;
    process_group_ = process_group;
    CHECK_GT(markov_rank_, 0) << "DSpark requires markov_rank > 0.";
  }

  // vLLM keys: markov_head.markov_w1.weight [vocab, rank],
  //            markov_head.markov_w2.weight [draft_vocab, rank].
  void load_state_dict(const StateDict& state_dict) {
    torch::Tensor w1 = state_dict.get_tensor("markov_head.markov_w1.weight");
    torch::Tensor w2 = state_dict.get_tensor("markov_head.markov_w2.weight");
    if (w1.defined()) {
      markov_w1_ = w1.to(tensor_options_);
    }
    if (w2.defined()) {
      markov_w2_ = w2.to(tensor_options_);
    }
  }

  void verify_loaded_weights(const std::string& prefix) const {
    CHECK(markov_w1_.defined()) << "Failed to find DSpark " << prefix
                                << "markov_head.markov_w1.weight.";
    CHECK(markov_w2_.defined()) << "Failed to find DSpark " << prefix
                                << "markov_head.markov_w2.weight.";
    CHECK_EQ(markov_w1_.dim(), 2)
        << "DSpark markov_w1 must be two-dimensional.";
    CHECK_EQ(markov_w2_.dim(), 2)
        << "DSpark markov_w2 must be two-dimensional.";
    CHECK_EQ(markov_w1_.size(1), markov_rank_)
        << "DSpark markov_w1 rank mismatch.";
    CHECK_EQ(markov_w2_.size(1), markov_rank_)
        << "DSpark markov_w2 rank mismatch.";
    CHECK_EQ(markov_w1_.size(0), markov_w2_.size(0))
        << "DSpark reduced-vocab drafts need draft-to-target remapping, not "
           "yet "
           "implemented.";
  }

  bool defined() const { return markov_w1_.defined() && markov_w2_.defined(); }

  int64_t vocab_size() const { return markov_w1_.size(0); }

  BlockDraftSampleOutput sample(
      const torch::Tensor& base_logits,
      const torch::Tensor& anchor_token_ids,
      const SamplingParameters& sampling_params) override {
    CHECK(defined()) << "DSpark Markov sampler weights are not initialized.";
    CHECK_EQ(base_logits.dim(), 3)
        << "DSpark base_logits must be [num_reqs, n_spec, draft_vocab].";
    const int64_t draft_vocab_size = base_logits.size(/*dim=*/2);
    CHECK_EQ(draft_vocab_size, vocab_size())
        << "DSpark reduced-vocab drafts (draft_vocab=" << draft_vocab_size
        << " != vocab=" << vocab_size()
        << ") need draft-to-target remapping, not yet implemented.";

    return dspark::sample_block(
        base_logits,
        anchor_token_ids,
        sampling_params,
        [this](const torch::Tensor& previous_token_ids) {
          return bias(previous_token_ids);
        },
        [this, &sampling_params](torch::Tensor& sampled_token_ids) {
          synchronize_sampled_token_ids(sampled_token_ids, sampling_params);
        });
  }

 private:
  void synchronize_sampled_token_ids(
      torch::Tensor& sampled_token_ids,
      const SamplingParameters& sampling_params) const {
    if (sampling_params.all_greedy_sample || process_group_ == nullptr ||
        process_group_->world_size() <= 1) {
      return;
    }
    sampled_token_ids = sampled_token_ids.contiguous();
    process_group_->broadcast(sampled_token_ids, /*root_rank=*/0);
  }

  torch::Tensor bias(const torch::Tensor& prev_token_ids) const {
    CHECK(defined()) << "DSpark Markov head weights are not initialized.";
    namespace F = torch::nn::functional;
    torch::Tensor markov_embedding = F::embedding(prev_token_ids, markov_w1_);
    return F::linear(markov_embedding, markov_w2_);
  }

  torch::Tensor markov_w1_;  // [vocab_size, markov_rank]
  torch::Tensor markov_w2_;  // [draft_vocab_size, markov_rank]
  torch::TensorOptions tensor_options_;
  ProcessGroup* process_group_ = nullptr;
  int64_t markov_rank_ = 0;
};

// DSpark draft model = DFlash block-diffusion backbone (context-K/V injection,
// prefill, weight loading all inherited unchanged) + a block-draft sampler held
// by the ForCausalLM layer. The backbone remains independent from sampling.
class DSparkQwen3ModelImpl final : public DFlashQwen3ModelImpl {
 public:
  explicit DSparkQwen3ModelImpl(const ModelContext& context)
      : DFlashQwen3ModelImpl(context) {}
};
TORCH_MODULE(DSparkQwen3Model);

class DSparkQwen3ForCausalLMImpl final
    : public LlmForCausalLMImplBase<DSparkQwen3Model> {
 public:
  explicit DSparkQwen3ForCausalLMImpl(const ModelContext& context)
      : LlmForCausalLMImplBase<DSparkQwen3Model>(context) {
    const ModelArgs& model_args = context.get_model_args();
    const ParallelArgs& parallel_args = context.get_parallel_args();
    ProcessGroup* process_group = parallel_args.tp_group_ != nullptr
                                      ? parallel_args.tp_group_
                                      : parallel_args.process_group_;
    block_draft_sampler_.initialize(
        context.get_tensor_options(), model_args.markov_rank(), process_group);
  }

  BlockDraftSampler* block_draft_sampler() { return &block_draft_sampler_; }

  void load_model(std::unique_ptr<ModelLoader> loader,
                  std::string prefix = "model.") override {
    for (const std::unique_ptr<StateDict>& state_dict :
         loader->get_state_dicts()) {
      StateDict sub_dict = state_dict->get_dict_with_prefix(prefix);
      if (sub_dict.size() == 0) {
        sub_dict = state_dict->get_dict_with_prefix("");
      }
      model_->load_state_dict(sub_dict);
      block_draft_sampler_.load_state_dict(sub_dict);
    }
    model_->verify_loaded_weights("");
    model_->merge_loaded_weights();
    block_draft_sampler_.verify_loaded_weights("");
  }

  ModelOutput write_context_kv(const torch::Tensor& target_hidden,
                               const torch::Tensor& positions,
                               const torch::Tensor& device_cache_slots,
                               std::vector<KVCache>& kv_caches,
                               const ModelInputParams& input_params) {
    return model_->write_context_kv(
        target_hidden, positions, device_cache_slots, kv_caches, input_params);
  }

 private:
  DSparkBlockDraftSampler block_draft_sampler_;
};
TORCH_MODULE(DSparkQwen3ForCausalLM);

// Draft config carries model_type="qwen3"; worker_impl overwrites
// args.model_type to "DSparkDraftModel" so this factory builds the draft body.
REGISTER_CAUSAL_MODEL_WITH_VARNAME(dspark_draft_model,
                                   DSparkDraftModel,
                                   DSparkQwen3ForCausalLM);

}  // namespace xllm::npu::model

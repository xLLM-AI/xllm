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
#include "framework/state_dict/state_dict.h"
#include "models/llm/npu/qwen3_dflash.h"
#include "models/model_registry.h"

namespace xllm::npu::model {

// Low-rank Markov head for DSpark block-diffusion drafting. The model owns the
// trained projection weights, while DSparkWorkerImpl owns the sequential
// sampling loop and distributed token synchronization.
//
// Weights (no bias, no activation):
//   markov_w1: [vocab_size, markov_rank]        embedded by the target-vocab
//                                               prev token id.
//   markov_w2: [draft_vocab_size, markov_rank]  projects the embed to a
//                                               draft-vocab bias.
//   bias(prev) = markov_w1[prev] @ markov_w2^T   -> [num_reqs, draft_vocab]
//
// Both weights are replicated on every TP rank. Sharding would add an
// all-reduce and a full-vocab gather for every sequential draft position.
class DSparkMarkovHead final {
 public:
  DSparkMarkovHead() = default;

  void initialize(const torch::TensorOptions& options, int64_t markov_rank) {
    tensor_options_ = options;
    markov_rank_ = markov_rank;
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

  torch::Tensor bias(const torch::Tensor& prev_token_ids) const {
    CHECK(defined()) << "DSpark Markov head weights are not initialized.";
    namespace F = torch::nn::functional;
    torch::Tensor markov_embedding = F::embedding(prev_token_ids, markov_w1_);
    return F::linear(markov_embedding, markov_w2_);
  }

  // Expose the shared markov_w1 embedding so ConfidenceHead can reuse the same
  // rank-256 features without a redundant lookup table copy.
  torch::Tensor markov_embed(const torch::Tensor& prev_token_ids) const {
    CHECK(defined()) << "DSpark Markov head weights are not initialized.";
    namespace F = torch::nn::functional;
    return F::embedding(prev_token_ids, markov_w1_);
  }

 private:
  bool defined() const { return markov_w1_.defined() && markov_w2_.defined(); }

  torch::Tensor markov_w1_;  // [vocab_size, markov_rank]
  torch::Tensor markov_w2_;  // [draft_vocab_size, markov_rank]
  torch::TensorOptions tensor_options_;
  int64_t markov_rank_ = 0;
};

// DSpark ConfidenceHead: given a draft-step hidden state and the previous
// token id, produces an acceptance-prob logit. When
// confidence_head_with_markov=true (the standard config), the head is applied
// on `concat(hidden, markov_embedding[prev])` — a low-rank Markov feature
// that reuses markov_w1 (the same embedding table the MarkovHead uses).
//
// Weights (bias-less linear that xllm sees as `confidence_head.proj`):
//   proj.weight: [1, hidden_size + markov_rank]  when with_markov=true
//   proj.weight: [1, hidden_size]                 otherwise
//   proj.bias:   [1]
//
// forward returns [num_reqs] float acceptance probabilities in [0, 1] (sigmoid
// applied). The controller consumes them as per-step draft accept
// probabilities.
class DSparkConfidenceHead final {
 public:
  DSparkConfidenceHead() = default;

  void initialize(const torch::TensorOptions& options,
                  int64_t hidden_size,
                  int64_t markov_rank,
                  bool with_markov) {
    tensor_options_ = options;
    hidden_size_ = hidden_size;
    markov_rank_ = markov_rank;
    with_markov_ = with_markov;
    CHECK_GT(hidden_size_, 0) << "DSpark ConfidenceHead requires hidden_size>0";
    if (with_markov_) {
      CHECK_GT(markov_rank_, 0)
          << "DSpark ConfidenceHead with_markov requires markov_rank>0";
    }
  }

  // vLLM keys: confidence_head.proj.{weight,bias}
  void load_state_dict(const StateDict& state_dict) {
    if (hidden_size_ <= 0) {
      // Not initialized (enable_confidence_head=false); skip.
      return;
    }
    torch::Tensor w = state_dict.get_tensor("confidence_head.proj.weight");
    torch::Tensor b = state_dict.get_tensor("confidence_head.proj.bias");
    if (w.defined()) {
      proj_weight_ = w.to(tensor_options_);
    }
    if (b.defined()) {
      proj_bias_ = b.to(tensor_options_);
    }
  }

  void verify_loaded_weights(const std::string& prefix) const {
    CHECK(proj_weight_.defined())
        << "Failed to find " << prefix << "confidence_head.proj.weight";
    CHECK(proj_bias_.defined())
        << "Failed to find " << prefix << "confidence_head.proj.bias";
    CHECK_EQ(proj_weight_.dim(), 2)
        << "confidence_head.proj.weight must be [1, in_dim]";
    CHECK_EQ(proj_bias_.dim(), 1) << "confidence_head.proj.bias must be [1]";
    CHECK_EQ(proj_weight_.size(0), 1)
        << "confidence_head.proj.weight must output a single logit";
    const int64_t expected_in =
        with_markov_ ? hidden_size_ + markov_rank_ : hidden_size_;
    CHECK_EQ(proj_weight_.size(1), expected_in)
        << "confidence_head.proj.weight in_dim mismatch, expected "
        << expected_in << " got " << proj_weight_.size(1);
    CHECK_EQ(proj_bias_.size(0), 1)
        << "confidence_head.proj.bias size mismatch";
  }

  // hidden: [num_reqs, hidden_size]
  // markov_embed: [num_reqs, markov_rank] — from `markov_w1[prev_token_ids]`.
  //   Only consumed when `with_markov_` is true.
  // returns: [num_reqs] float32 acceptance probabilities in [0, 1].
  torch::Tensor forward(const torch::Tensor& hidden,
                        const torch::Tensor& markov_embed) const {
    CHECK(defined()) << "DSpark ConfidenceHead weights are not initialized";
    CHECK_EQ(hidden.dim(), 2) << "ConfidenceHead hidden must be [B, H]";
    CHECK_EQ(hidden.size(-1), hidden_size_)
        << "ConfidenceHead hidden size mismatch";
    torch::Tensor input;
    if (with_markov_) {
      CHECK(markov_embed.defined())
          << "ConfidenceHead with_markov requires markov_embed";
      CHECK_EQ(markov_embed.size(0), hidden.size(0))
          << "ConfidenceHead markov_embed batch mismatch";
      CHECK_EQ(markov_embed.size(-1), markov_rank_)
          << "ConfidenceHead markov_embed rank mismatch";
      input = torch::cat({hidden.to(proj_weight_.dtype()),
                          markov_embed.to(proj_weight_.dtype())},
                         /*dim=*/-1);
    } else {
      input = hidden.to(proj_weight_.dtype());
    }
    namespace F = torch::nn::functional;
    torch::Tensor logit = F::linear(input, proj_weight_, proj_bias_);
    const double temperature = confidence_temperature();
    if (temperature != 1.0) {
      logit = logit / temperature;
    }
    return torch::sigmoid(logit).squeeze(-1).to(torch::kFloat32);
  }

  // Batched over the whole draft block: applies the same head as `forward` to
  // all gamma steps at once, so the sample loop pays one linear+sigmoid instead
  // of gamma. Numerically identical to gamma per-step `forward` calls.
  //   hidden_all:       [B, gamma, hidden_size]
  //   markov_embed_all: [B, gamma, markov_rank]  (consumed only when
  //   with_markov_)
  // returns:            [B, gamma] float32 acceptance probabilities in [0, 1].
  torch::Tensor forward_batched(const torch::Tensor& hidden_all,
                                const torch::Tensor& markov_embed_all) const {
    CHECK(defined()) << "DSpark ConfidenceHead weights are not initialized";
    CHECK_EQ(hidden_all.dim(), 3)
        << "ConfidenceHead hidden_all must be [B, gamma, H]";
    CHECK_EQ(hidden_all.size(-1), hidden_size_)
        << "ConfidenceHead hidden size mismatch";
    const int64_t batch = hidden_all.size(0);
    const int64_t gamma = hidden_all.size(1);
    torch::Tensor input;
    if (with_markov_) {
      CHECK(markov_embed_all.defined())
          << "ConfidenceHead with_markov requires markov_embed_all";
      CHECK_EQ(markov_embed_all.dim(), 3)
          << "ConfidenceHead markov_embed_all must be [B, gamma, rank]";
      CHECK_EQ(markov_embed_all.size(0), batch)
          << "ConfidenceHead markov_embed_all batch mismatch";
      CHECK_EQ(markov_embed_all.size(1), gamma)
          << "ConfidenceHead markov_embed_all gamma mismatch";
      CHECK_EQ(markov_embed_all.size(-1), markov_rank_)
          << "ConfidenceHead markov_embed_all rank mismatch";
      input = torch::cat({hidden_all.to(proj_weight_.dtype()),
                          markov_embed_all.to(proj_weight_.dtype())},
                         /*dim=*/-1);
    } else {
      input = hidden_all.to(proj_weight_.dtype());
    }
    namespace F = torch::nn::functional;
    torch::Tensor logit =
        F::linear(input.reshape({batch * gamma, -1}), proj_weight_, proj_bias_)
            .view({batch, gamma});
    const double temperature = confidence_temperature();
    if (temperature != 1.0) {
      logit = logit / temperature;
    }
    return torch::sigmoid(logit).to(torch::kFloat32);
  }

  bool defined() const {
    return proj_weight_.defined() && proj_bias_.defined();
  }

 private:
  // Optional temperature scaling on the confidence logit before sigmoid, shared
  // by both `forward` and `forward_batched` so the two paths stay numerically
  // identical. Raw neural confidence estimates are typically overconfident (Guo
  // et al. 2017); the DSpark paper (Section 3.2.1 "Post-hoc Calibration") calls
  // for Sequential Temperature Scaling to keep the cumulative product ∏ c_i
  // aligned with the empirical acceptance rate, which higher T approximates by
  // flattening confidence toward 0.5. The scaling path is kept for that future
  // calibration, but T is fixed at 1.0 (no scaling) here; when it becomes
  // tunable it should arrive as a ModelArgs field, not a process-global knob.
  static constexpr double confidence_temperature() { return 1.0; }

  torch::Tensor proj_weight_;  // [1, hidden_size (+ markov_rank)]
  torch::Tensor proj_bias_;    // [1]
  torch::TensorOptions tensor_options_;
  int64_t hidden_size_ = 0;
  int64_t markov_rank_ = 0;
  bool with_markov_ = false;
};

// DSpark draft model = DFlash block-diffusion backbone (context-K/V injection,
// prefill, weight loading all inherited unchanged) + a low-rank Markov head
// held by the ForCausalLM layer. The backbone remains independent from
// sampling.
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
    markov_head_.initialize(context.get_tensor_options(),
                            model_args.markov_rank());
    if (model_args.enable_confidence_head()) {
      confidence_head_.initialize(context.get_tensor_options(),
                                  model_args.hidden_size(),
                                  model_args.markov_rank(),
                                  model_args.confidence_head_with_markov());
    }
  }

  torch::Tensor dspark_markov_bias(
      const torch::Tensor& previous_token_ids) const {
    return markov_head_.bias(previous_token_ids);
  }

  // Compute per-request acceptance probability using the trained ConfidenceHead
  // over the draft-step hidden state and the previous token embedding.
  // hidden: [num_reqs, hidden_size], prev_token_ids: [num_reqs].
  // Returns [num_reqs] fp32 in [0, 1]. Defined only when
  // enable_confidence_head.
  torch::Tensor dspark_confidence_probs(
      const torch::Tensor& hidden,
      const torch::Tensor& previous_token_ids) const {
    CHECK(confidence_head_.defined())
        << "DSpark ConfidenceHead is not initialized (enable_confidence_head?)";
    torch::Tensor markov_embed;
    if (previous_token_ids.defined()) {
      markov_embed = markov_head_.markov_embed(previous_token_ids);
    }
    return confidence_head_.forward(hidden, markov_embed);
  }

  // Batched variant of dspark_confidence_probs over the whole draft block.
  //   hidden_all:  [num_reqs, num_spec, hidden_size]
  //   prev_matrix: [num_reqs, num_spec] int64 — column k is step k's "prev"
  //                token (col 0 = anchor, col k = draft token sampled at k-1).
  // Returns [num_reqs, num_spec] fp32 in [0, 1]. Defined only when
  // enable_confidence_head.
  torch::Tensor dspark_confidence_probs_batched(
      const torch::Tensor& hidden_all,
      const torch::Tensor& prev_matrix) const {
    CHECK(confidence_head_.defined())
        << "DSpark ConfidenceHead is not initialized (enable_confidence_head?)";
    torch::Tensor markov_embed_all;
    if (prev_matrix.defined()) {
      markov_embed_all = markov_head_.markov_embed(prev_matrix);
    }
    return confidence_head_.forward_batched(hidden_all, markov_embed_all);
  }

  bool has_dspark_confidence_head() const { return confidence_head_.defined(); }

  void load_model(std::unique_ptr<ModelLoader> loader,
                  std::string prefix = "model.") override {
    for (const std::unique_ptr<StateDict>& state_dict :
         loader->get_state_dicts()) {
      StateDict sub_dict = state_dict->get_dict_with_prefix(prefix);
      if (sub_dict.size() == 0) {
        sub_dict = state_dict->get_dict_with_prefix("");
      }
      model_->load_state_dict(sub_dict);
      markov_head_.load_state_dict(sub_dict);
      confidence_head_.load_state_dict(sub_dict);
    }
    model_->verify_loaded_weights("");
    model_->merge_loaded_weights();
    markov_head_.verify_loaded_weights("");
    if (confidence_head_.defined()) {
      confidence_head_.verify_loaded_weights("");
    }
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
  DSparkMarkovHead markov_head_;
  DSparkConfidenceHead confidence_head_;
};
TORCH_MODULE(DSparkQwen3ForCausalLM);

// Draft config carries model_type="qwen3"; worker_impl overwrites
// args.model_type to "DSparkDraftModel" so this factory builds the draft body.
REGISTER_CAUSAL_MODEL_WITH_VARNAME(dspark_draft_model,
                                   DSparkDraftModel,
                                   DSparkQwen3ForCausalLM);

}  // namespace xllm::npu::model

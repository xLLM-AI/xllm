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

#pragma once

#include <torch/torch.h>

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "core/framework/model/model_args.h"
#include "core/framework/model/model_traits.h"
#include "core/framework/model_context/model_context.h"
#include "core/framework/model_loader/model_loader.h"
#include "core/framework/state_dict/state_dict.h"
#include "core/layers/common/lm_head.h"
#include "core/layers/common/word_embedding.h"
#if defined(USE_NPU)
#include "core/layers/npu/npu_lm_head_impl.h"
#endif

namespace xllm {
namespace detail {

// Vocabulary-head ownership for family model classes (the layer that also
// owns embed_tokens and the decoder stack, mirroring `self.lm_head` on HF /
// vLLM / SGLang / TRT-LLM family models). The host must derive from
// torch::nn::Module; the head is registered as a child module under the host.
// HeadType selects the head flavor the family uses: layer::NpuLmHead for NPU
// families (ATB head), layer::LmHead otherwise.
template <typename HeadType>
class LmHeadOwnerImpl;

// State and adoption logic shared by both head flavors.
class LmHeadOwnerBase {
 protected:
  // Reads the ModelArgs flags both flavors need.
  void read_model_args(const ModelContext& context) {
    const auto& model_args = context.get_model_args();
    tie_word_embeddings_ = model_args.tie_word_embeddings();
    embedding_mode_ = model_args.embedding_mode();
  }

  // Adopts the target's head when this owner hosts none of its own loaded. An
  // already-allocated eager head is replaced so its vocab-sized device tensor
  // is released instead of lingering as an orphaned child.
  template <typename Head>
  void adopt_head_impl(torch::nn::Module& host,
                       const char* name,
                       Head& slot,
                       Head target) {
    if (weight_loaded(slot)) {
      return;
    }
    if (slot.is_empty()) {
      slot = host.register_module(name, std::move(target));
    } else {
      slot = host.replace_module(name, std::move(target));
    }
  }

  bool tie_word_embeddings_ = false;
  bool embedding_mode_ = false;
};

#if defined(USE_NPU)
// NPU families host the ATB head; this specialization exists only under NPU.
template <>
class LmHeadOwnerImpl<layer::NpuLmHead> : public LmHeadOwnerBase {
 public:
  void init_lm_head(torch::nn::Module& host, const ModelContext& context) {
    read_model_args(context);
    if (embedding_mode_) {
      return;
    }
    npu_lm_head_ =
        host.register_module("npu_lm_head", layer::NpuLmHead(context));
  }

  // hidden_states: [num_tokens, hidden_size]; selected_idxes: [num_tokens] or
  // undefined; out_hidden (optional export): [num_selected_tokens,
  // hidden_size] gathered rows; returns [num_selected_tokens, vocab_size].
  torch::Tensor forward_lm_head(const torch::Tensor& hidden_states,
                                const torch::Tensor& selected_idxes,
                                torch::Tensor& out_hidden) {
    return npu_lm_head_->forward_with_hidden(
        hidden_states, selected_idxes, out_hidden, /*nodeId=*/0);
  }

  // The CausalLM hook name for forward_lm_head, so owner hosts and their
  // derived families satisfy has_lm_head_logits through inheritance.
  torch::Tensor logits(const torch::Tensor& hidden_states,
                       const torch::Tensor& selected_idxes,
                       torch::Tensor& out_hidden) {
    return forward_lm_head(hidden_states, selected_idxes, out_hidden);
  }

  // The CausalLM head-accessor hook name for lm_head().
  layer::NpuLmHead get_npu_lm_head() { return npu_lm_head_; }

  // Customization point mirroring the CPU flavor: families override this
  // virtual for checkpoint layouts that don't match the default prefixes
  // below. Declared on the owner so derived families inherit it.
  virtual void load_lm_head_weights(const StateDict& state_dict,
                                    layer::NpuLmHead& head,
                                    bool tie_word_embeddings) {
    // NPU-style checkpoints: tied heads reuse the embedding weights under
    // the backbone's embed_tokens prefixes.
    static const std::vector<std::string> kNpuTiedEmbeddingPrefixes = {
        "model.embed_tokens.", "embed_tokens."};
    static const std::vector<std::string> kLmHeadPrefixes = {
        "lm_head.",
        "model.lm_head.",
        "language_model.lm_head.",
        "model.language_model.lm_head."};
    if (tie_word_embeddings) {
      head->load_state_dict(
          state_dict.get_dict_with_prefix(kNpuTiedEmbeddingPrefixes));
    } else {
      head->load_state_dict(state_dict.get_dict_with_prefix(kLmHeadPrefixes));
    }
  }

  // `pinned_host` keeps the NPU head weight in pinned host memory (rolling/
  // lazy paths) instead of merging onto the device.
  void load_lm_head_weights(const std::unique_ptr<ModelLoader>& loader,
                            bool pinned_host = false) {
    if (embedding_mode_) {
      return;
    }
    for (const auto& state_dict : loader->get_state_dicts()) {
      load_lm_head_weights(*state_dict, npu_lm_head_, tie_word_embeddings_);
    }
    if (npu_lm_head_->is_weight_loaded()) {
      npu_lm_head_->verify_loaded_weights(tie_word_embeddings_ ? "embed_tokens."
                                                               : "lm_head.");
      if (pinned_host) {
        npu_lm_head_->merge_and_move_pinned_host();
      } else {
        npu_lm_head_->merge_loaded_weights();
      }
    }
  }

  layer::NpuLmHead lm_head() const { return npu_lm_head_; }

  // Head lifecycle hooks; no-ops unless the head is hosted.
  void free_head_weights() {
    if (!npu_lm_head_.is_empty()) {
      npu_lm_head_->free_weights();
    }
  }

  void reload_head_weights() {
    if (!npu_lm_head_.is_empty()) {
      npu_lm_head_->reload_weights();
    }
  }

  void reload_head_weights_from_device() {
    if (!npu_lm_head_.is_empty()) {
      npu_lm_head_->reload_weights_from_device();
    }
  }

  // Adopts the target's head when this owner hosts none of its own loaded. An
  // already-allocated eager head is replaced so its vocab-sized device tensor
  // is released instead of lingering as an orphaned child.
  void adopt_lm_head(torch::nn::Module& host, layer::NpuLmHead target_head) {
    adopt_head_impl(host, "npu_lm_head", npu_lm_head_, std::move(target_head));
  }

 protected:
  layer::NpuLmHead npu_lm_head_{nullptr};
};
#endif

template <>
class LmHeadOwnerImpl<layer::LmHead> : public LmHeadOwnerBase {
 public:
  void init_lm_head(torch::nn::Module& host, const ModelContext& context) {
    read_model_args(context);
    if (embedding_mode_) {
      return;
    }
    lm_head_ = host.register_module("lm_head", layer::LmHead(context));
  }

  // hidden_states: [num_tokens, hidden_size]; selected_idxes: [num_tokens] or
  // undefined; out_hidden (optional export): [num_selected_tokens,
  // hidden_size] gathered rows; returns [num_selected_tokens, vocab_size].
  torch::Tensor forward_lm_head(const torch::Tensor& hidden_states,
                                const torch::Tensor& selected_idxes,
                                torch::Tensor& out_hidden) {
    return lm_head_->forward_with_hidden(
        hidden_states, selected_idxes, out_hidden);
  }

  // The CausalLM hook names for forward_lm_head / lm_head(), so owner hosts
  // and their derived families satisfy the has_* traits through inheritance.
  torch::Tensor logits(const torch::Tensor& hidden_states,
                       const torch::Tensor& selected_idxes,
                       torch::Tensor& out_hidden) {
    return forward_lm_head(hidden_states, selected_idxes, out_hidden);
  }

  layer::LmHead get_lm_head() { return lm_head_; }

  // Customization point for checkpoint layouts that don't match the default
  // prefixes below: families override this virtual (declared on the owner
  // so derived families inherit it). Virtual dispatch replaces the previous
  // compile-time probe, which could not see hooks defined on derived
  // classes.
  virtual void load_lm_head_weights(const StateDict& state_dict,
                                    layer::LmHead& head,
                                    bool tie_word_embeddings) {
    if (tie_word_embeddings) {
      // Torch-style checkpoints: tied heads read the embedding weights
      // directly (T5-style "embed." on some families).
      static const std::vector<std::string> kTorchTiedEmbeddingPrefixes = {
          "embed_tokens.", "embed."};
      head->load_state_dict(
          state_dict.get_dict_with_prefix(kTorchTiedEmbeddingPrefixes));
    } else {
      head->load_state_dict(state_dict.get_dict_with_prefix("lm_head."));
    }
  }

  void load_lm_head_weights(const std::unique_ptr<ModelLoader>& loader,
                            bool /*pinned_host*/ = false) {
    if (embedding_mode_) {
      return;
    }
    for (const auto& state_dict : loader->get_state_dicts()) {
      load_lm_head_weights(*state_dict, lm_head_, tie_word_embeddings_);
    }
  }

  layer::LmHead lm_head() const { return lm_head_; }

  // Bind the tied CPU lm_head weight to embed_tokens.weight so they share one
  // storage. TP>1 shards them on orthogonal dims, so they cannot alias and the
  // head keeps its own copy.
  void tie_lm_head_weights(layer::WordEmbedding embed_tokens) {
    if (!tie_word_embeddings_ || embedding_mode_ || lm_head_.is_empty()) {
      return;
    }
    if (embed_tokens.is_empty()) {
      return;
    }
    auto head_weight = lm_head_->weight();
    auto embed_weight = embed_tokens->weight();
    if (!head_weight.defined() || !embed_weight.defined() ||
        head_weight.sizes() != embed_weight.sizes() ||
        head_weight.scalar_type() != embed_weight.scalar_type()) {
      return;
    }
    lm_head_->tie_weights(embed_weight);
  }

  // Adopts the target's head when this owner hosts none of its own loaded. An
  // already-allocated eager head is replaced so its vocab-sized device tensor
  // is released instead of lingering as an orphaned child.
  void adopt_lm_head(torch::nn::Module& host, layer::LmHead target_head) {
    adopt_head_impl(host, "lm_head", lm_head_, std::move(target_head));
  }

 protected:
  layer::LmHead lm_head_{nullptr};
};

// Convenience alias: families inherit `LmHeadOwner<HeadType>` with the head
// flavor they host.
template <typename HeadType = layer::LmHead>
using LmHeadOwner = LmHeadOwnerImpl<HeadType>;

}  // namespace detail
}  // namespace xllm

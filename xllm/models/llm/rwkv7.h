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

// RWKV-7 "Goose" causal LM implementation for xLLM inference.
//
// Architecture reference:
//   https://github.com/BlinkDL/RWKV-LM/tree/main/RWKV-v7
//
// Key characteristics that distinguish RWKV-7 from transformer LLMs:
//   • No positional encoding (recurrence replaces attention).
//   • Fixed-size per-layer recurrent state (W-matrix + shift states).
//   • State stored in the linear-attention KV cache (conv + ssm tensors).
//   • v_first cross-layer tensor managed in the custom forward loop.
//
// Checkpoint weight layout (BlinkDL / HuggingFace RWKV):
//   emb.weight, blocks.{i}.{ln0/ln1/ln2/att/ffn}.*, ln_out.*, head.weight

#pragma once

#include <torch/torch.h>

#include <cstdint>
#include <string>
#include <unordered_set>
#include <vector>

#include "core/framework/kv_cache/kv_cache.h"
#include "core/framework/model/model_args.h"
#include "core/framework/model/model_input_params.h"
#include "core/framework/model/model_output.h"
#include "core/framework/model_context.h"
#include "core/framework/model_loader.h"
#include "core/layers/rwkv7_decoder_layer.h"
#include "models/model_registry.h"

namespace xllm {

// ---------------------------------------------------------------------------
// RWKV7Model – embedding + N blocks + final layer-norm
// ---------------------------------------------------------------------------

class RWKV7ModelImpl : public torch::nn::Module {
 public:
  explicit RWKV7ModelImpl(const ModelContext& context) {
    const ModelArgs& args = context.get_model_args();
    const int32_t n_layers = static_cast<int32_t>(args.n_layers());
    const int64_t hidden_size = args.hidden_size();
    const int64_t vocab_size = args.vocab_size();
    const float ln_eps =
        args.layer_norm_eps() > 0.0f ? args.layer_norm_eps() : 1e-5f;
    const torch::TensorOptions opts = context.get_tensor_options();

    emb_ = register_module("emb",
                           torch::nn::Embedding(torch::nn::EmbeddingOptions(
                               vocab_size, hidden_size)));
    emb_->to(opts.device());
    emb_->to(opts.dtype().toScalarType());

    blocks_module_list_ = register_module("blocks", torch::nn::ModuleList());
    blocks_.reserve(static_cast<size_t>(n_layers));
    for (int32_t i = 0; i < n_layers; ++i) {
      auto block = layer::RWKV7DecoderLayer(context, i);
      blocks_.emplace_back(block);
      blocks_module_list_->push_back(block);
    }

    ln_out_ = register_module(
        "ln_out",
        torch::nn::LayerNorm(
            torch::nn::LayerNormOptions({hidden_size}).eps(ln_eps)));
    ln_out_->to(opts.device());
    ln_out_->to(opts.dtype().toScalarType());
  }

  // tokens:    [total_tokens] (all sequences packed)
  // positions: unused — RWKV-7 has no positional encoding
  // kv_caches: one KVCache per layer; each holds conv+ssm state tensors
  //
  // Returns: ModelOutput with hidden_states [total_tokens, hidden_size]
  ModelOutput forward(const torch::Tensor& tokens,
                      const torch::Tensor& /*positions*/,
                      std::vector<KVCache>& kv_caches,
                      const ModelInputParams& input_params) {
    torch::Tensor h = emb_(tokens);  // [total_tokens, hidden_size]

    // v_first is a cross-layer state initialised by block 0 and passed
    // through all subsequent blocks for value-residual blending.
    torch::Tensor v_first;

    for (size_t i = 0; i < blocks_.size(); ++i) {
      h = blocks_[i]->forward(
          h, kv_caches[i], input_params, static_cast<int32_t>(i), v_first);
    }

    torch::Tensor hidden_states = ln_out_(h);
    return ModelOutput(hidden_states, std::nullopt);
  }

  void load_state_dict(const StateDict& state_dict) {
    // Embedding: RWKV checkpoints use "emb.weight"
    torch::Tensor emb_w =
        state_dict.get_dict_with_prefix("emb.").get_tensor("weight");
    if (emb_w.defined()) {
      emb_->weight =
          emb_w.to(emb_->weight.device()).to(emb_->weight.scalar_type());
    }

    // Decoder blocks
    for (size_t i = 0; i < blocks_.size(); ++i) {
      const std::string prefix = "blocks." + std::to_string(i) + ".";
      blocks_[i]->load_state_dict(state_dict.get_dict_with_prefix(prefix));
    }

    // Final layer norm: load weight and bias directly (xLLM StateDict API)
    const StateDict ln_dict = state_dict.get_dict_with_prefix("ln_out.");
    torch::Tensor ln_w = ln_dict.get_tensor("weight");
    if (ln_w.defined()) {
      ln_out_->weight =
          ln_w.to(ln_out_->weight.device()).to(ln_out_->weight.scalar_type());
    }
    torch::Tensor ln_b = ln_dict.get_tensor("bias");
    if (ln_b.defined()) {
      ln_out_->bias =
          ln_b.to(ln_out_->bias.device()).to(ln_out_->bias.scalar_type());
    }
  }

 private:
  torch::nn::Embedding emb_{nullptr};
  torch::nn::ModuleList blocks_module_list_{nullptr};
  std::vector<layer::RWKV7DecoderLayer> blocks_;
  torch::nn::LayerNorm ln_out_{nullptr};
};

TORCH_MODULE(RWKV7Model);

// ---------------------------------------------------------------------------
// RWKV7ForCausalLM – adds the unembedding LM head
// ---------------------------------------------------------------------------

class RWKV7ForCausalLMImpl : public torch::nn::Module {
 public:
  explicit RWKV7ForCausalLMImpl(const ModelContext& context) {
    const ModelArgs& args = context.get_model_args();
    const int64_t hidden_size = args.hidden_size();
    const int64_t vocab_size = args.vocab_size();
    const torch::TensorOptions opts = context.get_tensor_options();

    model_ = register_module("model", RWKV7Model(context));

    // LM head: linear, no bias (BlinkDL convention)
    head_ = register_module(
        "head",
        torch::nn::Linear(
            torch::nn::LinearOptions(hidden_size, vocab_size).bias(false)));
    head_->to(opts.device());
    head_->to(opts.dtype().toScalarType());
  }

  // Forward: compute hidden states for the packed token sequence.
  ModelOutput forward(const torch::Tensor& tokens,
                      const torch::Tensor& positions,
                      std::vector<KVCache>& kv_caches,
                      const ModelInputParams& input_params) {
    return model_(tokens, positions, kv_caches, input_params);
  }

  // Compute logits for selected token positions.
  //   hidden_states:  [total_tokens, hidden_size]
  //   selected_idxes: optional token selector
  //   returns:        [selected_tokens, vocab_size]
  torch::Tensor logits(const torch::Tensor& hidden_states,
                       const torch::Tensor& selected_idxes) {
    torch::Tensor h = hidden_states;
    if (selected_idxes.defined()) {
      h = h.index_select(/*dim=*/0, selected_idxes);
    }
    return head_(h);
  }

  // Embedding / rerank path: return selected hidden states as-is.
  torch::Tensor pooler(const torch::Tensor& hidden_states,
                       const torch::Tensor& selected_idxes) {
    torch::Tensor h = hidden_states;
    if (selected_idxes.defined()) {
      h = h.index_select(/*dim=*/0, selected_idxes);
    }
    return h;
  }

  // Load weights from checkpoint.
  //
  // Supported checkpoint layouts:
  //   • BlinkDL native: flat keys (emb.weight, blocks.*.*, ln_out.*,
  //   head.weight) • HuggingFace:    "model." prefix (model.emb.weight, …,
  //   model.ln_out.*, lm_head.weight)
  void load_model(std::unique_ptr<ModelLoader> loader) {
    for (const auto& state_dict : loader->get_state_dicts()) {
      // Try to strip common top-level prefixes for the body weights.
      // An empty prefix ("") matches the native BlinkDL flat format.
      model_->load_state_dict(state_dict->get_dict_with_prefix(
          std::vector<std::string>{"model.", ""}));

      // LM head weight: "head.weight" (native) or "lm_head.weight" (HF)
      for (const auto& key : std::vector<std::string>{
               "head.weight", "lm_head.weight", "model.head.weight"}) {
        torch::Tensor head_w = state_dict->get_tensor(key);
        if (head_w.defined()) {
          head_->weight =
              head_w.to(head_->weight.device()).to(head_->weight.scalar_type());
          break;
        }
      }
    }
  }

  // Required by CausalLMImpl (no-ops for non-MoE models)
  void prepare_expert_weight(int32_t /*layer_id*/,
                             const std::vector<int32_t>& /*expert_ids*/) {}
  void update_expert_weight(int32_t /*layer_id*/) {}

 private:
  RWKV7Model model_{nullptr};
  torch::nn::Linear head_{nullptr};
};

TORCH_MODULE(RWKV7ForCausalLM);

// ---------------------------------------------------------------------------
// Model registration
// ---------------------------------------------------------------------------

REGISTER_CAUSAL_MODEL(rwkv7, RWKV7ForCausalLM);

// Model args — supports the RWKV-7 HuggingFace config.json format.
//
// Minimal config.json example:
//   {
//     "model_type": "rwkv7",
//     "hidden_size": 768,
//     "num_hidden_layers": 12,
//     "vocab_size": 65536,
//     "head_size": 64
//   }
REGISTER_MODEL_ARGS(rwkv7, [&] {
  LOAD_ARG_OR(model_type, "model_type", "rwkv7");
  LOAD_ARG_OR(dtype, "torch_dtype", "");

  LOAD_ARG_OR(vocab_size, "vocab_size", 65536);
  LOAD_ARG_OR(hidden_size, "hidden_size", 768);
  LOAD_ARG_OR(n_layers, "num_hidden_layers", 12);

  // RWKV-7 head_size_a — stored as "head_size" in the config.
  LOAD_ARG_OR(head_dim, "head_size", 64);

  // Derive n_heads from hidden_size / head_size.
  LOAD_ARG_OR_FUNC(n_heads, "num_attention_heads", [&] {
    return args->hidden_size() / args->head_dim();
  });

  // FFN intermediate size (4 × hidden_size by default).
  LOAD_ARG_OR_FUNC(intermediate_size, "intermediate_size", [&] {
    return 4LL * args->hidden_size();
  });

  // LayerNorm eps used for ln0 / ln1 / ln2 / ln_out (default 1e-5).
  LOAD_ARG_OR(layer_norm_eps, "layer_norm_eps", 1e-5f);

  // RWKV-7 World checkpoints are trained with ctx4096.
  LOAD_ARG_OR(max_position_embeddings, "max_position_embeddings", 4096);

  LOAD_ARG_OR(bos_token_id, "bos_token_id", 0);
  LOAD_ARG_OR(eos_token_id, "eos_token_id", 0);

  // RWKV-7 has no standard transformer attention; use a single dummy KV head
  // so the standard paged KV cache is allocated as small as possible.
  SET_ARG(n_kv_heads, static_cast<int64_t>(1));

  // Linear-attention KV cache configuration.
  //
  // conv_cache per slot: [1, head_size * (2*n_heads + n_heads)] = [1, 3*H]
  //   [0:H]   → att_x_prev  (time-mix shift state)
  //   [H:2H]  → ffn_x_prev  (channel-mix shift state)
  //   [2H:3H] → unused (artefact of the KVCacheShape formula)
  //
  // ssm_cache per slot: [n_heads, head_size, head_size]  ← W-matrix state
  LOAD_ARG_OR_FUNC(linear_num_key_heads, "num_attention_heads", [&] {
    return args->n_heads();  // int64_t matches linear_num_key_heads type
  });
  LOAD_ARG_OR_FUNC(linear_key_head_dim, "head_size", [&] {
    return static_cast<int32_t>(args->head_dim());
  });
  LOAD_ARG_OR_FUNC(linear_num_value_heads, "num_attention_heads", [&] {
    return static_cast<int32_t>(args->n_heads());
  });
  LOAD_ARG_OR_FUNC(linear_value_head_dim, "head_size", [&] {
    return static_cast<int32_t>(args->head_dim());
  });
  // conv_state_len = linear_conv_kernel_dim - 1 = 1 (one step of history)
  SET_ARG(linear_conv_kernel_dim, static_cast<int32_t>(2));

  // Set full_attention_interval = n_layers + 1 so that:
  //
  //   1. has_linear_attention_layers(args) == true
  //      (because full_attention_interval > 1)
  //      → worker_impl sets enable_linear_attention = true
  //      → KVCacheShape allocates conv_cache + ssm_cache tensors
  //
  //   2. is_linear_attention_layer(i, full_attention_interval) == true
  //      for ALL layer indices i ∈ [0, n_layers-1]
  //      (because (i+1) % (n_layers+1) ≠ 0 for i < n_layers)
  //      → create_kv_cache_impl creates LinearAttentionKVCacheImpl per layer
  //
  // Without this, KVCacheImpl (standard paged blocks) would be created for
  // every layer, and get_conv_cache() / get_ssm_cache() would return undefined
  // tensors, causing a DCHECK failure at the first forward pass.
  SET_ARG(full_attention_interval, static_cast<int32_t>(args->n_layers() + 1));

  // Mark all layers as "rwkv7" so that has_linear_attention_layers() also
  // returns true via the layer_types path, providing extra clarity.
  SET_ARG(
      layer_types,
      std::vector<std::string>(static_cast<size_t>(args->n_layers()), "rwkv7"));

  SET_ARG(stop_token_ids, std::unordered_set<int32_t>({args->eos_token_id()}));
});

// Tokenizer args.
//
// RWKV-7 World models use the official trie tokenizer backed by
// rwkv_vocab_v20230424.txt (exported by tools/convert_rwkv7_world.py).
REGISTER_TOKENIZER_ARGS(rwkv7, [&] {
  SET_ARG(tokenizer_type, "rwkv");
  SET_ARG(vocab_file, "rwkv_vocab_v20230424.txt");
});

}  // namespace xllm

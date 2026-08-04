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

#include "core/layers/rwkv7_decoder_layer.h"

#include <glog/logging.h>
#include <torch/torch.h>

#include <cstdint>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "framework/model/model_args.h"
#include "framework/model/model_input_params.h"
#include "framework/model_context.h"
#include "framework/state_dict/state_dict.h"

namespace xllm {
namespace layer {

namespace {

// ---- Weight-loading helpers ------------------------------------------------

// Load a tensor from state_dict and place it on the same device/dtype as ref.
// Returns an undefined tensor if the key is absent.
torch::Tensor load_to(const StateDict& sd,
                      const std::string& name,
                      const torch::Device& dev,
                      torch::ScalarType dtype) {
  torch::Tensor t = sd.get_tensor(name);
  if (!t.defined()) {
    return torch::Tensor();
  }
  return t.to(dev).to(dtype);
}

// Load weight (and optionally bias) for a torch::nn::Linear layer.
void load_linear(torch::nn::Linear& linear, const StateDict& state_dict) {
  torch::Tensor w = state_dict.get_tensor("weight");
  if (w.defined()) {
    linear->weight =
        w.to(linear->weight.device()).to(linear->weight.scalar_type());
  }
  if (linear->options.bias()) {
    torch::Tensor b = state_dict.get_tensor("bias");
    if (b.defined()) {
      linear->bias = b.to(linear->bias.device()).to(linear->bias.scalar_type());
    }
  }
}

// Load weight and bias for a torch::nn::LayerNorm layer.
void load_layer_norm(torch::nn::LayerNorm& ln, const StateDict& state_dict) {
  torch::Tensor w = state_dict.get_tensor("weight");
  if (w.defined()) {
    ln->weight = w.to(ln->weight.device()).to(ln->weight.scalar_type());
  }
  torch::Tensor b = state_dict.get_tensor("bias");
  if (b.defined()) {
    ln->bias = b.to(ln->bias.device()).to(ln->bias.scalar_type());
  }
}

// Load weight and bias for a torch::nn::GroupNorm layer.
void load_group_norm(torch::nn::GroupNorm& gn, const StateDict& state_dict) {
  torch::Tensor w = state_dict.get_tensor("weight");
  if (w.defined()) {
    gn->weight = w.to(gn->weight.device()).to(gn->weight.scalar_type());
  }
  torch::Tensor b = state_dict.get_tensor("bias");
  if (b.defined()) {
    gn->bias = b.to(gn->bias.device()).to(gn->bias.scalar_type());
  }
}

}  // namespace

// ---------------------------------------------------------------------------
// RWKV7TimeMixImpl
// ---------------------------------------------------------------------------

RWKV7TimeMixImpl::RWKV7TimeMixImpl(const ModelContext& context,
                                   int32_t /*layer_id*/) {
  const ModelArgs& args = context.get_model_args();
  hidden_size_ = args.hidden_size();
  head_size_ = args.head_dim();  // RWKV-7 head_size_a
  n_heads_ = hidden_size_ / head_size_;

  const torch::TensorOptions opts = context.get_tensor_options();

  // Token-shift mixing scalars [1, 1, C] — registered parameters so that
  // model->to(device) moves them automatically.
  x_r_ = register_parameter("x_r", torch::zeros({1, 1, hidden_size_}, opts));
  x_w_ = register_parameter("x_w", torch::zeros({1, 1, hidden_size_}, opts));
  x_k_ = register_parameter("x_k", torch::zeros({1, 1, hidden_size_}, opts));
  x_v_ = register_parameter("x_v", torch::zeros({1, 1, hidden_size_}, opts));
  x_a_ = register_parameter("x_a", torch::zeros({1, 1, hidden_size_}, opts));
  x_g_ = register_parameter("x_g", torch::zeros({1, 1, hidden_size_}, opts));

  // Decay / a-gate / v-first base scalars [1, 1, C]
  w0_ = register_parameter("w0", torch::zeros({1, 1, hidden_size_}, opts));
  a0_ = register_parameter("a0", torch::zeros({1, 1, hidden_size_}, opts));
  v0_ = register_parameter("v0", torch::zeros({1, 1, hidden_size_}, opts));

  // Key modifiers [1, 1, C]
  k_k_ = register_parameter("k_k", torch::ones({1, 1, hidden_size_}, opts));
  k_a_ = register_parameter("k_a", torch::ones({1, 1, hidden_size_}, opts));

  // Receptance-key interaction [n_heads_, head_size_]
  r_k_ = register_parameter("r_k", torch::zeros({n_heads_, head_size_}, opts));

  // Linear projections: C → C, no bias (BlinkDL checkpoint convention)
  receptance_ = register_module(
      "receptance",
      torch::nn::Linear(
          torch::nn::LinearOptions(hidden_size_, hidden_size_).bias(false)));
  key_ = register_module(
      "key",
      torch::nn::Linear(
          torch::nn::LinearOptions(hidden_size_, hidden_size_).bias(false)));
  value_ = register_module(
      "value",
      torch::nn::Linear(
          torch::nn::LinearOptions(hidden_size_, hidden_size_).bias(false)));
  output_ = register_module(
      "output",
      torch::nn::Linear(
          torch::nn::LinearOptions(hidden_size_, hidden_size_).bias(false)));

  // GroupNorm: n_heads_ groups, hidden_size_ channels, eps = 64e-5 (RWKV-7
  // spec)
  ln_x_ = register_module(
      "ln_x",
      torch::nn::GroupNorm(torch::nn::GroupNormOptions(
                               static_cast<int64_t>(n_heads_), hidden_size_)
                               .eps(64e-5)));

  receptance_->to(opts.device());
  receptance_->to(opts.dtype().toScalarType());
  key_->to(opts.device());
  key_->to(opts.dtype().toScalarType());
  value_->to(opts.device());
  value_->to(opts.dtype().toScalarType());
  output_->to(opts.device());
  output_->to(opts.dtype().toScalarType());
  ln_x_->to(opts.device());
  ln_x_->to(opts.dtype().toScalarType());

  // LoRA weights (w1/w2, a1/a2, v1/v2, g1/g2): shapes vary per checkpoint
  // rank.  They are NOT registered parameters — loaded as plain tensors in
  // load_state_dict() and placed on the same device as x_r_.
}

void RWKV7TimeMixImpl::load_state_dict(const StateDict& state_dict) {
  const torch::Device dev = x_r_.device();
  const torch::ScalarType dtype = x_r_.scalar_type();

  // Registered shift-mix scalars and key modifiers
  const std::vector<std::pair<std::string, torch::Tensor*>> param_map = {
      {"x_r", &x_r_},
      {"x_w", &x_w_},
      {"x_k", &x_k_},
      {"x_v", &x_v_},
      {"x_a", &x_a_},
      {"x_g", &x_g_},
      {"w0", &w0_},
      {"a0", &a0_},
      {"v0", &v0_},
      {"k_k", &k_k_},
      {"k_a", &k_a_},
      {"r_k", &r_k_},
  };
  for (const auto& [name, ptr] : param_map) {
    torch::Tensor t = state_dict.get_tensor(name);
    if (t.defined()) {
      *ptr = t.to(dev).to(dtype);
    }
  }

  // LoRA tensors: infer rank from checkpoint tensor shapes
  w1_ = load_to(state_dict, "w1", dev, dtype);
  w2_ = load_to(state_dict, "w2", dev, dtype);
  a1_ = load_to(state_dict, "a1", dev, dtype);
  a2_ = load_to(state_dict, "a2", dev, dtype);
  v1_ = load_to(state_dict, "v1", dev, dtype);
  v2_ = load_to(state_dict, "v2", dev, dtype);
  g1_ = load_to(state_dict, "g1", dev, dtype);
  g2_ = load_to(state_dict, "g2", dev, dtype);

  // Sub-modules: load weight tensors directly
  load_linear(receptance_, state_dict.get_dict_with_prefix("receptance."));
  load_linear(key_, state_dict.get_dict_with_prefix("key."));
  load_linear(value_, state_dict.get_dict_with_prefix("value."));
  load_linear(output_, state_dict.get_dict_with_prefix("output."));
  load_group_norm(ln_x_, state_dict.get_dict_with_prefix("ln_x."));
}

torch::Tensor RWKV7TimeMixImpl::compute_decay(const torch::Tensor& xw) const {
  // xw: [T, C]
  // w = log_sigmoid(w0 + tanh(xw @ w1) @ w2) - 0.5
  //   ≡ -softplus(-(w0 + tanh(xw @ w1) @ w2)) - 0.5
  // decay = exp(-exp(w))  ∈ (0, 1)
  CHECK(w1_.defined() && w2_.defined())
      << "RWKV-7 decay LoRA weights (w1, w2) not loaded";
  torch::Tensor lora = torch::tanh(xw.matmul(w1_)).matmul(w2_);  // [T, C]
  torch::Tensor w_raw = w0_.view({hidden_size_}) + lora;         // [T, C]
  torch::Tensor w = torch::log_sigmoid(w_raw) - 0.5f;            // [T, C]
  torch::Tensor decay = torch::exp(-torch::exp(w));              // [T, C]
  return decay.view({-1LL, n_heads_, head_size_});               // [T, H, N]
}

std::pair<torch::Tensor, torch::Tensor> RWKV7TimeMixImpl::rwkv7_recurrence(
    const torch::Tensor& r,
    const torch::Tensor& w,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& a,
    const torch::Tensor& kk,
    const torch::Tensor& state) const {
  // r, w, k, v, a, kk: [T, H, N]
  // state:              [H, N, N]
  const int64_t T = r.size(0);
  const int64_t H = n_heads_;
  const int64_t N = head_size_;

  torch::Tensor out = torch::zeros({T, H * N}, r.options());
  // Float32 for numerical stability of the W-matrix state
  torch::Tensor s = state.to(torch::kFloat32);

  for (int64_t t = 0; t < T; ++t) {
    // Outer product: vk = v ⊗ k  →  [H, N, N]
    torch::Tensor vt = v[t].to(torch::kFloat32).view({H, N, 1});
    torch::Tensor kt = k[t].to(torch::kFloat32).view({H, 1, N});
    torch::Tensor vk = vt.bmm(kt);

    // In-context learning correction: ab = (-kk) ⊗ (kk * a)  →  [H, N, N]
    torch::Tensor kkt = kk[t].to(torch::kFloat32);  // [H, N]
    torch::Tensor at = a[t].to(torch::kFloat32);    // [H, N]
    torch::Tensor ab = (-kkt).view({H, N, 1}).bmm((kkt * at).view({H, 1, N}));

    // State update: s = s * w + s @ ab + vk
    torch::Tensor wt = w[t].to(torch::kFloat32).view({H, 1, N});
    s = s * wt + s.bmm(ab) + vk;

    // Read-out: out[t] = (s @ r)^T  →  [H, N]
    torch::Tensor rt = r[t].to(torch::kFloat32).view({H, N, 1});
    out[t] = s.bmm(rt).view({H * N}).to(r.dtype());
  }

  return {out, s.to(state.dtype())};
}

std::pair<torch::Tensor, torch::Tensor> RWKV7TimeMixImpl::forward(
    const torch::Tensor& x,
    const torch::Tensor& att_x_prev,
    torch::Tensor& att_kv,
    bool is_layer0,
    torch::Tensor& v_first,
    const std::vector<int32_t>& seq_lens) {
  // x:          [total_tokens, C]
  // att_x_prev: [num_seqs, C]
  // att_kv:     [num_seqs, H, N, N]

  const int64_t total_tokens = x.size(0);
  const int32_t num_seqs = static_cast<int32_t>(seq_lens.size());

  torch::Tensor output = torch::zeros_like(x);
  torch::Tensor new_att_x_prev = torch::zeros_like(att_x_prev);

  if (is_layer0 && !v_first.defined()) {
    v_first = torch::zeros_like(x);
  }

  int64_t token_offset = 0;
  for (int32_t s = 0; s < num_seqs; ++s) {
    const int64_t T = static_cast<int64_t>(seq_lens[static_cast<size_t>(s)]);
    torch::Tensor xs = x.slice(0, token_offset, token_offset + T);  // [T, C]
    torch::Tensor x_prev_s = att_x_prev[s].unsqueeze(0);            // [1, C]

    // Time-shift: xx[t] = x[t-1] - x[t],  with x[-1] = att_x_prev
    torch::Tensor shifted;
    if (T > 1) {
      shifted = torch::cat({x_prev_s, xs.slice(0, 0, T - 1)}, 0);
    } else {
      shifted = x_prev_s;
    }
    torch::Tensor xx = shifted - xs;  // [T, C]

    // Mixed inputs using shift scalars (broadcast [1,1,C] → [T,C])
    auto mix = [&](const torch::Tensor& coeff) {
      return xs + xx * coeff.view({hidden_size_});  // [T, C]
    };
    torch::Tensor xr = mix(x_r_);
    torch::Tensor xw = mix(x_w_);
    torch::Tensor xk = mix(x_k_);
    torch::Tensor xv = mix(x_v_);
    torch::Tensor xa = mix(x_a_);
    torch::Tensor xg = mix(x_g_);

    // Linear projections
    torch::Tensor r_proj = receptance_(xr);  // [T, C]
    torch::Tensor k_proj = key_(xk);         // [T, C]
    torch::Tensor v_proj = value_(xv);       // [T, C]

    // Decay [T, H, N]
    torch::Tensor w_decay = compute_decay(xw);

    // A-gate: sigmoid(a0 + xa @ a1 @ a2)  →  [T, C]
    CHECK(a1_.defined() && a2_.defined())
        << "RWKV-7 a-gate LoRA weights (a1, a2) not loaded";
    torch::Tensor a_gate =
        torch::sigmoid(a0_.view({hidden_size_}) + xa.matmul(a1_).matmul(a2_));

    // Gate: sigmoid(xg @ g1) @ g2  →  [T, C]
    CHECK(g1_.defined() && g2_.defined())
        << "RWKV-7 gate LoRA weights (g1, g2) not loaded";
    torch::Tensor gate = torch::sigmoid(xg.matmul(g1_)).matmul(g2_);

    // Head layout reshaping: [T, H, N]
    torch::Tensor r_h = r_proj.view({T, n_heads_, head_size_});
    torch::Tensor k_h = k_proj.view({T, n_heads_, head_size_});
    torch::Tensor v_h = v_proj.view({T, n_heads_, head_size_});
    torch::Tensor a_h = a_gate.view({T, n_heads_, head_size_});

    // Key normalisation (per-head L2 norm)
    torch::Tensor kk_h = torch::nn::functional::normalize(
        k_h * k_k_.view({1LL, n_heads_, head_size_}),
        torch::nn::functional::NormalizeFuncOptions().dim(-1).p(2));

    // Key modulation with a-gate
    torch::Tensor k_mod =
        k_h * (1.0f + (a_h - 1.0f) * k_a_.view({1LL, n_heads_, head_size_}));

    // V-first blending
    torch::Tensor v_h_blend = v_h;
    if (is_layer0) {
      v_first.slice(0, token_offset, token_offset + T).copy_(v_proj);
    } else {
      CHECK(v1_.defined() && v2_.defined())
          << "RWKV-7 v-first LoRA weights (v1, v2) not loaded";
      torch::Tensor vf = v_first.slice(0, token_offset, token_offset + T)
                             .view({T, n_heads_, head_size_});
      torch::Tensor blend =
          torch::sigmoid(v0_.view({hidden_size_}) + xv.matmul(v1_).matmul(v2_))
              .view({T, n_heads_, head_size_});
      v_h_blend = v_h + (vf - v_h) * blend;
    }

    // Core RWKV-7 recurrence
    auto [seq_out, new_state] =
        rwkv7_recurrence(r_h, w_decay, k_mod, v_h_blend, a_h, kk_h, att_kv[s]);
    att_kv[s].copy_(new_state);  // update W-matrix state in-place

    // GroupNorm over the channel dimension
    torch::Tensor normed = ln_x_(seq_out.view({T, hidden_size_}));

    // Receptance-key residual: per-head dot-product correction
    torch::Tensor rk_dot = (r_h * k_mod * r_k_.unsqueeze(0))
                               .sum(/*dim=*/-1, /*keepdim=*/true);  // [T, H, 1]
    torch::Tensor rk_res = (rk_dot * v_h_blend).view({T, hidden_size_});

    // Output projection
    torch::Tensor block_out = output_((normed + rk_res) * gate);  // [T, C]

    output.slice(0, token_offset, token_offset + T).copy_(block_out);
    new_att_x_prev[s].copy_(xs[T - 1]);  // last time-mix input for next shift

    token_offset += T;
  }

  return {output, new_att_x_prev};
}

// ---------------------------------------------------------------------------
// RWKV7ChannelMixImpl
// ---------------------------------------------------------------------------

RWKV7ChannelMixImpl::RWKV7ChannelMixImpl(const ModelContext& context,
                                         int32_t /*layer_id*/) {
  const ModelArgs& args = context.get_model_args();
  const int64_t hidden_size = args.hidden_size();
  const int64_t intermediate_size = args.intermediate_size();
  const torch::TensorOptions opts = context.get_tensor_options();

  x_k_ = register_parameter("x_k", torch::zeros({1, 1, hidden_size}, opts));

  key_ = register_module(
      "key",
      torch::nn::Linear(torch::nn::LinearOptions(hidden_size, intermediate_size)
                            .bias(false)));
  value_ = register_module(
      "value",
      torch::nn::Linear(torch::nn::LinearOptions(intermediate_size, hidden_size)
                            .bias(false)));
  key_->to(opts.device());
  key_->to(opts.dtype().toScalarType());
  value_->to(opts.device());
  value_->to(opts.dtype().toScalarType());
}

void RWKV7ChannelMixImpl::load_state_dict(const StateDict& state_dict) {
  torch::Tensor xk_t = state_dict.get_tensor("x_k");
  if (xk_t.defined()) {
    x_k_ = xk_t.to(x_k_.device()).to(x_k_.scalar_type());
  }
  load_linear(key_, state_dict.get_dict_with_prefix("key."));
  load_linear(value_, state_dict.get_dict_with_prefix("value."));
}

std::pair<torch::Tensor, torch::Tensor> RWKV7ChannelMixImpl::forward(
    const torch::Tensor& x,
    const torch::Tensor& ffn_x_prev,
    const std::vector<int32_t>& seq_lens) {
  const int32_t num_seqs = static_cast<int32_t>(seq_lens.size());
  const int64_t C = x.size(-1);

  torch::Tensor output = torch::zeros_like(x);
  torch::Tensor new_ffn_x_prev = torch::zeros_like(ffn_x_prev);

  int64_t token_offset = 0;
  for (int32_t s = 0; s < num_seqs; ++s) {
    const int64_t T = static_cast<int64_t>(seq_lens[static_cast<size_t>(s)]);
    torch::Tensor xs = x.slice(0, token_offset, token_offset + T);
    torch::Tensor x_prev_s = ffn_x_prev[s].unsqueeze(0);  // [1, C]

    torch::Tensor shifted;
    if (T > 1) {
      shifted = torch::cat({x_prev_s, xs.slice(0, 0, T - 1)}, 0);
    } else {
      shifted = x_prev_s;
    }
    torch::Tensor xx = shifted - xs;
    torch::Tensor k = xs + xx * x_k_.view({C});  // [T, C]

    // Gated FFN: relu(key(k)) ^ 2 → value
    torch::Tensor k_act = torch::pow(torch::relu(key_(k)), 2.0f);
    torch::Tensor out_s = value_(k_act);  // [T, C]

    output.slice(0, token_offset, token_offset + T).copy_(out_s);
    new_ffn_x_prev[s].copy_(xs[T - 1]);

    token_offset += T;
  }

  return {output, new_ffn_x_prev};
}

// ---------------------------------------------------------------------------
// RWKV7DecoderLayerImpl
// ---------------------------------------------------------------------------

RWKV7DecoderLayerImpl::RWKV7DecoderLayerImpl(const ModelContext& context,
                                             int32_t layer_id)
    : layer_id_(layer_id), has_ln0_(layer_id == 0) {
  const ModelArgs& args = context.get_model_args();
  hidden_size_ = args.hidden_size();
  head_size_ = args.head_dim();
  n_heads_ = hidden_size_ / head_size_;

  const float ln_eps =
      args.layer_norm_eps() > 0.0f ? args.layer_norm_eps() : 1e-5f;
  const torch::TensorOptions opts = context.get_tensor_options();

  if (has_ln0_) {
    ln0_ = register_module(
        "ln0",
        torch::nn::LayerNorm(
            torch::nn::LayerNormOptions({hidden_size_}).eps(ln_eps)));
    ln0_->to(opts.device());
    ln0_->to(opts.dtype().toScalarType());
  }
  ln1_ = register_module(
      "ln1",
      torch::nn::LayerNorm(
          torch::nn::LayerNormOptions({hidden_size_}).eps(ln_eps)));
  ln2_ = register_module(
      "ln2",
      torch::nn::LayerNorm(
          torch::nn::LayerNormOptions({hidden_size_}).eps(ln_eps)));
  ln1_->to(opts.device());
  ln1_->to(opts.dtype().toScalarType());
  ln2_->to(opts.device());
  ln2_->to(opts.dtype().toScalarType());

  att_ = register_module("att", RWKV7TimeMix(context, layer_id));
  ffn_ = register_module("ffn", RWKV7ChannelMix(context, layer_id));
}

void RWKV7DecoderLayerImpl::load_state_dict(const StateDict& state_dict) {
  if (has_ln0_) {
    load_layer_norm(ln0_, state_dict.get_dict_with_prefix("ln0."));
  }
  load_layer_norm(ln1_, state_dict.get_dict_with_prefix("ln1."));
  load_layer_norm(ln2_, state_dict.get_dict_with_prefix("ln2."));
  att_->load_state_dict(state_dict.get_dict_with_prefix("att."));
  ffn_->load_state_dict(state_dict.get_dict_with_prefix("ffn."));
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
RWKV7DecoderLayerImpl::read_state(KVCache& kv_cache,
                                  const torch::Tensor& state_indices,
                                  int32_t /*num_seqs*/) const {
  torch::Tensor conv = kv_cache.get_conv_cache();
  torch::Tensor ssm = kv_cache.get_ssm_cache();

  CHECK(conv.defined()) << "conv_cache is not allocated; ensure RWKV-7 uses "
                           "the linear-attention KV cache.";
  CHECK(ssm.defined()) << "ssm_cache is not allocated; ensure RWKV-7 uses "
                          "the linear-attention KV cache.";

  torch::Tensor sel_conv = conv.index_select(0, state_indices);  // [S, 1, 3H]
  torch::Tensor sel_ssm = ssm.index_select(0, state_indices);    // [S, H, N, N]

  // Unpack shift states from conv_cache[slot, 0, 0:2H]
  torch::Tensor flat = sel_conv.squeeze(1);  // [S, 3H]
  torch::Tensor att_x_prev =
      flat.slice(/*dim=*/-1, 0LL, hidden_size_).contiguous();
  torch::Tensor ffn_x_prev =
      flat.slice(/*dim=*/-1, hidden_size_, 2LL * hidden_size_).contiguous();

  return {att_x_prev, ffn_x_prev, sel_ssm.contiguous()};
}

void RWKV7DecoderLayerImpl::write_state(KVCache& kv_cache,
                                        const torch::Tensor& state_indices,
                                        const torch::Tensor& att_x_prev,
                                        const torch::Tensor& ffn_x_prev,
                                        const torch::Tensor& att_kv) const {
  torch::Tensor conv = kv_cache.get_conv_cache();
  torch::Tensor ssm = kv_cache.get_ssm_cache();

  const int64_t num_seqs = att_x_prev.size(0);
  torch::Tensor new_conv =
      torch::zeros({num_seqs, 1LL, 3LL * hidden_size_}, conv.options());

  // Pack shift states into new_conv[s, 0, 0:H] and [H:2H]
  new_conv.squeeze(1).slice(/*dim=*/-1, 0LL, hidden_size_).copy_(att_x_prev);
  new_conv.squeeze(1)
      .slice(/*dim=*/-1, hidden_size_, 2LL * hidden_size_)
      .copy_(ffn_x_prev);

  conv.index_copy_(0, state_indices, new_conv);
  ssm.index_copy_(0, state_indices, att_kv);
}

torch::Tensor RWKV7DecoderLayerImpl::forward(
    const torch::Tensor& x,
    KVCache& kv_cache,
    const ModelInputParams& input_params,
    int32_t layer_id,
    torch::Tensor& v_first) {
  const int32_t num_seqs = input_params.meta.num_sequences;

  std::vector<int32_t> seq_lens(static_cast<size_t>(num_seqs));
  for (int32_t s = 0; s < num_seqs; ++s) {
    seq_lens[static_cast<size_t>(s)] = input_params.get_q_seq_len(s);
  }

  // Use scheduler-provided slot indices when available; otherwise fall back
  // to sequential slot assignment [0, 1, …, num_seqs-1].
  torch::Tensor state_indices;
  if (input_params.embedding.linear_state_indices.defined()) {
    state_indices = input_params.embedding.linear_state_indices;
  } else {
    state_indices = torch::arange(
        num_seqs,
        torch::TensorOptions().dtype(torch::kLong).device(x.device()));
  }
  if (state_indices.scalar_type() != torch::kLong) {
    state_indices = state_indices.to(torch::kLong);
  }

  auto [att_x_prev, ffn_x_prev, att_kv] =
      read_state(kv_cache, state_indices, num_seqs);

  // ln0: applied at block 0 before ln1 to normalise the token embedding
  torch::Tensor h = x;
  if (has_ln0_) {
    h = ln0_(h);
  }

  // Time-mix (attention replacement)
  torch::Tensor h_att = ln1_(h);
  auto [att_out, new_att_x_prev] = att_->forward(
      h_att, att_x_prev, att_kv, layer_id == 0, v_first, seq_lens);
  h = h + att_out;

  // Channel-mix (FFN replacement)
  torch::Tensor h_ffn = ln2_(h);
  auto [ffn_out, new_ffn_x_prev] = ffn_->forward(h_ffn, ffn_x_prev, seq_lens);
  h = h + ffn_out;

  write_state(kv_cache, state_indices, new_att_x_prev, new_ffn_x_prev, att_kv);

  return h;
}

}  // namespace layer
}  // namespace xllm

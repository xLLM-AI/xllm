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

#include "qwen3_gated_delta_net_base.h"

#include <glog/logging.h>
#include <torch/torch.h>

#include <cstdlib>
#include <string_view>
#include <tuple>

#include "qwen3_5_gated_delta_net.h"
#include "qwen3_gated_delta_net_impls.h"

namespace xllm {
namespace layer {

namespace {

// Runtime backend selection. Both flags default off (use optimized kernel).
// Setting them to "1" swaps that operator to the pure-torch fallback for
// debugging / A-B testing.
inline bool use_torch_conv1d() {
  static const bool v = [] {
    const char* e = std::getenv("XLLM_DCU_USE_TORCH_CONV1D");
    return e && std::string_view(e) == "1";
  }();
  return v;
}

inline bool use_torch_recurrent_prefill() {
  static const bool v = [] {
    const char* e = std::getenv("XLLM_DCU_USE_TORCH_RECURRENT");
    return e && std::string_view(e) == "1";
  }();
  return v;
}

// -------------------------------------------------------------------------
// Backend dispatch — a single entry point per operator; picks torch_impl or
// kernel_impl based on the env flag. Callers stay free of if/else clutter.
// -------------------------------------------------------------------------

torch::Tensor causal_conv1d_prefill(const torch::Tensor& flat_input,
                                    const torch::Tensor& conv_weight,
                                    torch::Tensor& conv_cache,
                                    const std::vector<int64_t>& cu_seqlens,
                                    const std::vector<int64_t>& state_indices,
                                    int32_t kernel_size) {
  namespace gdn = qwen3_gdn;
  return (use_torch_conv1d() ? gdn::torch_impl::causal_conv1d_prefill
                             : gdn::kernel_impl::causal_conv1d_prefill)(
      flat_input.contiguous(),
      conv_weight,
      conv_cache,
      cu_seqlens,
      state_indices,
      kernel_size,
      /*activation=*/true);
}

torch::Tensor causal_conv1d_decode(const torch::Tensor& flat_input,
                                   const torch::Tensor& conv_weight,
                                   torch::Tensor& conv_cache,
                                   const torch::Tensor& state_indices,
                                   int32_t kernel_size) {
  namespace gdn = qwen3_gdn;
  return (use_torch_conv1d()
              ? gdn::torch_impl::causal_conv1d_decode
              : gdn::kernel_impl::causal_conv1d_decode)(flat_input,
                                                        conv_weight,
                                                        conv_cache,
                                                        state_indices,
                                                        kernel_size,
                                                        /*activation=*/true);
}

// Prefill selects between torch_recurrent (slow, exact) and aiter chunked
// (fast). Decode always uses torch_recurrent (T=1, no chunk speedup possible).
std::tuple<torch::Tensor, torch::Tensor> gated_delta_rule_prefill(
    const torch::Tensor& cat_q,
    const torch::Tensor& cat_k,
    const torch::Tensor& cat_v,
    const torch::Tensor& cat_g,
    const torch::Tensor& cat_beta,
    const torch::Tensor& initial_state,
    const std::vector<int64_t>& cu_seqlens) {
  namespace gdn = qwen3_gdn;
  if (use_torch_recurrent_prefill()) {
    return gdn::torch_impl::gated_delta_rule(
        cat_q, cat_k, cat_v, cat_g, cat_beta, initial_state);
  }
  return gdn::kernel_impl::gated_delta_rule_prefill(
      cat_q,
      cat_k,
      cat_v,
      cat_g,
      cat_beta,
      std::optional<torch::Tensor>(initial_state),
      cu_seqlens,
      /*output_final_state=*/true,
      /*use_qk_l2norm_in_kernel=*/true);
}

std::tuple<torch::Tensor, torch::Tensor> gated_delta_rule_decode(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& g,
    const torch::Tensor& beta,
    const torch::Tensor& init_state) {
  return qwen3_gdn::torch_impl::gated_delta_rule(q, k, v, g, beta, init_state);
}

// -------------------------------------------------------------------------
// Small helpers used only by the layer's forward orchestration.
// -------------------------------------------------------------------------

// Split the fused QKVZ/BA projections back into their four component tensors.
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
split_qkvz_ba(const torch::Tensor& qkvz_flat,
              const torch::Tensor& ba_flat,
              int64_t num_k_heads_local,
              int64_t num_v_heads_local,
              int64_t head_k_dim,
              int64_t head_v_dim) {
  const int64_t k_size = num_k_heads_local * head_k_dim;
  const int64_t v_size = num_v_heads_local * head_v_dim;

  auto qkvz_split =
      torch::split(qkvz_flat, {k_size, k_size, v_size, v_size}, 1);
  const auto& q = qkvz_split[0];
  const auto& k = qkvz_split[1];
  const auto& v = qkvz_split[2];
  const auto& z = qkvz_split[3];
  auto ba_split =
      torch::split(ba_flat, {num_v_heads_local, num_v_heads_local}, 1);
  const auto& b = ba_split[0];
  const auto& a = ba_split[1];
  auto mixed_qkv = torch::cat({q, k, v}, 1).contiguous();
  return std::make_tuple(mixed_qkv, z, b, a);
}

int64_t get_checkpoint_stride(const torch::Tensor& conv_cache,
                              const torch::Tensor& ssm_cache) {
  if (!conv_cache.defined() || !ssm_cache.defined() ||
      conv_cache.numel() == 0 || ssm_cache.numel() == 0) {
    return 1;
  }
  CHECK_GT(conv_cache.size(0), 0) << "conv cache must have positive batch dim";
  CHECK_EQ(ssm_cache.size(0) % conv_cache.size(0), 0)
      << "ssm cache checkpoint layout mismatch";
  return ssm_cache.size(0) / conv_cache.size(0);
}

torch::Tensor build_linear_state_base_indices(
    const torch::Tensor& logical_state_indices,
    int64_t checkpoint_stride) {
  if (checkpoint_stride == 1) {
    return logical_state_indices;
  }
  return logical_state_indices * checkpoint_stride;
}

// cu_seqlens with only the positive-length sequences retained. Zero-length
// padding entries in q_seq_lens_vec must not become separate cu_seqlens rows
// because they don't consume linear state slots.
std::vector<int64_t> build_real_cu_seqlens(
    const std::vector<int32_t>& q_seq_lens_vec) {
  std::vector<int64_t> cu;
  cu.reserve(q_seq_lens_vec.size() + 1);
  cu.push_back(0);
  for (const int32_t vlen : q_seq_lens_vec) {
    if (vlen > 0) {
      cu.push_back(cu.back() + vlen);
    }
  }
  return cu;
}

// Extract cu_seqlens from host-side lens (or fall back to device tensor).
std::vector<int64_t> extract_cu_seqlens(
    const AttentionMetadata& attn_metadata) {
  std::vector<int64_t> cu;
  if (!attn_metadata.q_seq_lens_vec.empty()) {
    cu.assign(attn_metadata.q_seq_lens_vec.begin(),
              attn_metadata.q_seq_lens_vec.end());
  } else {
    auto cpu_lens = attn_metadata.q_seq_lens.cpu();
    auto* ptr = cpu_lens.data_ptr<int64_t>();
    cu.assign(ptr, ptr + cpu_lens.numel());
  }
  return cu;
}

}  // namespace

// =========================================================================
// Qwen3GatedDeltaNetBaseImpl
// =========================================================================

Qwen3GatedDeltaNetBaseImpl::Qwen3GatedDeltaNetBaseImpl(
    const ModelArgs& args,
    const QuantArgs& quant_args,
    const ParallelArgs& parallel_args,
    const torch::TensorOptions& options) {
  tp_size_ = parallel_args.tp_group_->world_size();
  rank_ = parallel_args.tp_group_->rank();
  num_k_heads_ = args.linear_num_key_heads();
  num_v_heads_ = args.linear_num_value_heads();
  head_k_dim_ = args.linear_key_head_dim();
  head_v_dim_ = args.linear_value_head_dim();
  k_size_ = num_k_heads_ * head_k_dim_;
  v_size_ = num_v_heads_ * head_v_dim_;
  conv_kernel_size_ = args.linear_conv_kernel_dim();

  // Shared causal conv projection over mixed QKV states.
  conv1d_ = register_module("conv1d",
                            ColumnParallelLinear(args.linear_conv_kernel_dim(),
                                                 k_size_ * 2 + v_size_,
                                                 /*bias=*/false,
                                                 /*gather_output=*/false,
                                                 quant_args,
                                                 parallel_args.tp_group_,
                                                 options));

  auto opts = options.dtype(torch::kFloat32);
  dt_bias_ = register_parameter("dt_bias",
                                torch::ones({num_v_heads_ / tp_size_}, opts),
                                /*requires_grad=*/false);

  A_log_ = register_parameter("A_log",
                              torch::empty({num_v_heads_ / tp_size_}, opts),
                              /*requires_grad=*/false);

  o_proj_ = register_module("out_proj",
                            RowParallelLinear(v_size_,
                                              args.hidden_size(),
                                              /*bias=*/false,
                                              /*input_is_parallelized=*/true,
                                              /*if_reduce_results=*/true,
                                              quant_args,
                                              parallel_args.tp_group_,
                                              options));

  norm_ = register_module(
      "norm", RmsNormGated(head_v_dim_, args.rms_norm_eps(), options));
}

void Qwen3GatedDeltaNetBaseImpl::load_common_state_dict(
    const StateDict& state_dict) {
  const int64_t rank = rank_;
  const int64_t world_size = tp_size_;
  const int32_t shard_tensor_count = 3;
  const std::vector<int64_t> shard_sizes = {
      k_size_ / tp_size_, k_size_ / tp_size_, v_size_ / tp_size_};

  if (auto w = state_dict.get_tensor("conv1d.weight"); w.defined()) {
    conv1d_->load_state_dict(
        StateDict({{"weight", w.squeeze(1)}}), shard_tensor_count, shard_sizes);
    conv1d_->weight().set_(conv1d_->weight().transpose(0, 1).contiguous());
  }
  o_proj_->load_state_dict(state_dict.get_dict_with_prefix("out_proj."));
  if (auto w = state_dict.get_tensor("norm.weight"); w.defined()) {
    norm_->load_state_dict(StateDict({{"weight", w}}));
  }
  LOAD_SHARDED_WEIGHT(dt_bias, 0);
  LOAD_SHARDED_WEIGHT(A_log, 0);
}

void Qwen3GatedDeltaNetBaseImpl::verify_common_loaded_weights(
    const std::string& prefix) const {
  CHECK(dt_bias_is_loaded_)
      << "Missing required weight: " << prefix << "dt_bias";
  CHECK(A_log_is_loaded_) << "Missing required weight: " << prefix << "A_log";
}

std::pair<torch::Tensor, torch::Tensor>
Qwen3GatedDeltaNetBaseImpl::project_padded_inputs(
    const torch::Tensor& hidden_states,
    const AttentionMetadata& attn_metadata) {
  if (attn_metadata.is_prefill || attn_metadata.is_chunked_prefill) {
    auto [qkvz_flat, ba_flat] = project_flat_inputs(hidden_states);
    return {reshape_qkvz_with_pad(attn_metadata, qkvz_flat),
            reshape_qkvz_with_pad(attn_metadata, ba_flat)};
  }
  return project_decode_inputs(hidden_states);
}

torch::Tensor Qwen3GatedDeltaNetBaseImpl::forward(
    const torch::Tensor& hidden_states,
    const AttentionMetadata& attn_metadata,
    KVCache& kv_cache,
    const ModelInputParams& input_params) {
  const int64_t original_num_tokens = hidden_states.size(0);
  auto [qkvz_padded, ba_padded] =
      project_padded_inputs(hidden_states, attn_metadata);
  const int64_t batch_size = qkvz_padded.size(0);
  const int64_t seq_len = qkvz_padded.size(1);

  auto qkvz_flat =
      qkvz_padded.reshape({batch_size * seq_len, qkvz_padded.size(-1)});
  auto ba_flat = ba_padded.reshape({batch_size * seq_len, ba_padded.size(-1)});

  torch::Tensor mixed_qkv, z, b, a;
  std::tie(mixed_qkv, z, b, a) = split_qkvz_ba(qkvz_flat,
                                               ba_flat,
                                               num_k_heads_ / tp_size_,
                                               num_v_heads_ / tp_size_,
                                               head_k_dim_,
                                               head_v_dim_);

  mixed_qkv = mixed_qkv.reshape({batch_size, seq_len, mixed_qkv.size(-1)});
  z = z.reshape({batch_size, seq_len, num_v_heads_ / tp_size_, head_v_dim_});
  b = b.reshape({batch_size, seq_len, num_v_heads_ / tp_size_});
  a = a.reshape({batch_size, seq_len, num_v_heads_ / tp_size_});

  torch::Tensor conv_cache = kv_cache.get_conv_cache();
  torch::Tensor ssm_cache = kv_cache.get_ssm_cache();
  torch::Tensor conv_weight = conv1d_->weight();
  torch::Tensor logical_state_indices =
      get_linear_state_indices(input_params, mixed_qkv.device());
  const int64_t checkpoint_stride =
      get_checkpoint_stride(conv_cache, ssm_cache);
  torch::Tensor linear_state_base_indices =
      build_linear_state_base_indices(logical_state_indices, checkpoint_stride);
  const bool is_prefill =
      attn_metadata.is_prefill || attn_metadata.is_chunked_prefill;

  // ---- Causal Conv1d ----
  if (is_prefill) {
    auto cu_seqlens_vec = extract_cu_seqlens(attn_metadata);
    auto conv_input = reshape_qkvz_unpad(attn_metadata, mixed_qkv);
    std::vector<int64_t> state_indices_vec(
        input_params.embedding.linear_state_ids.begin(),
        input_params.embedding.linear_state_ids.end());
    mixed_qkv = causal_conv1d_prefill(conv_input,
                                      conv_weight,
                                      conv_cache,
                                      cu_seqlens_vec,
                                      state_indices_vec,
                                      conv_kernel_size_);
    mixed_qkv = reshape_qkvz_with_pad(attn_metadata, mixed_qkv);
    mixed_qkv = mixed_qkv.transpose(1, 2);
  } else {
    auto flat_input =
        mixed_qkv.reshape({batch_size * seq_len, mixed_qkv.size(-1)});
    auto updated = causal_conv1d_decode(flat_input,
                                        conv_weight,
                                        conv_cache,
                                        logical_state_indices,
                                        conv_kernel_size_);
    mixed_qkv = updated.reshape({batch_size, seq_len, -1}).transpose(1, 2);
  }

  // ---- Gating ----
  const auto beta = torch::sigmoid(b);
  torch::Tensor g;
  {
    auto A_log_exp = A_log_.exp();
    auto softplus_out = torch::nn::functional::softplus(
        a.to(torch::kFloat32) + dt_bias_,
        torch::nn::functional::SoftplusFuncOptions().beta(1.0).threshold(20.0));
    g = (-A_log_exp * softplus_out).to(a.dtype()).contiguous();
  }

  auto [processed_q, processed_k, processed_v] = process_mixed_qkv(mixed_qkv);

  // ---- Gated Delta Rule ----
  torch::Tensor core_attn_out;
  if (is_prefill) {
    CHECK_GE(attn_metadata.q_seq_lens_vec.size(),
             static_cast<size_t>(batch_size))
        << "q_seq_lens_vec must be populated for Qwen3.5 prefill.";

    std::vector<torch::Tensor> pq, pk, pv, pg, pbeta;
    pq.reserve(batch_size);
    pk.reserve(batch_size);
    pv.reserve(batch_size);
    pg.reserve(batch_size);
    pbeta.reserve(batch_size);
    for (int64_t bidx = 0; bidx < batch_size; ++bidx) {
      const int64_t vlen = attn_metadata.q_seq_lens_vec[bidx];
      pq.push_back(processed_q[bidx].narrow(0, 0, vlen));
      pk.push_back(processed_k[bidx].narrow(0, 0, vlen));
      pv.push_back(processed_v[bidx].narrow(0, 0, vlen));
      pg.push_back(g[bidx].narrow(0, 0, vlen));
      pbeta.push_back(beta[bidx].narrow(0, 0, vlen));
    }
    auto cat_q = torch::cat(pq, 0).unsqueeze(0);
    auto cat_k = torch::cat(pk, 0).unsqueeze(0);
    auto cat_v = torch::cat(pv, 0).unsqueeze(0);
    auto cat_g = torch::cat(pg, 0).unsqueeze(0);
    auto cat_beta = torch::cat(pbeta, 0).unsqueeze(0);

    // ssm_cache stores state as [N, H, V, K] (non-fla) or [N, H, K, V] (fla);
    // gated_delta_rule expects [N, H, K, V] fla layout.
    auto initial_state =
        torch::index_select(ssm_cache, 0, linear_state_base_indices);
    if (!use_fla_ssm_state_layout()) {
      initial_state = initial_state.transpose(-1, -2).contiguous();
    }

    auto real_cu = build_real_cu_seqlens(attn_metadata.q_seq_lens_vec);
    torch::Tensor last_state;
    std::tie(core_attn_out, last_state) = gated_delta_rule_prefill(
        cat_q, cat_k, cat_v, cat_g, cat_beta, initial_state, real_cu);

    // Scatter back to per-batch output.
    core_attn_out = core_attn_out.squeeze(0);
    auto final_out = torch::zeros_like(processed_v);
    int64_t offset = 0;
    for (int64_t bidx = 0; bidx < batch_size; ++bidx) {
      const int64_t vlen = attn_metadata.q_seq_lens_vec[bidx];
      final_out[bidx]
          .narrow(0, 0, vlen)
          .copy_(core_attn_out.narrow(0, offset, vlen));
      offset += vlen;
    }
    core_attn_out = final_out;

    auto state_to_store =
        use_fla_ssm_state_layout() ? last_state : last_state.transpose(-1, -2);
    ssm_cache.index_put_({linear_state_base_indices},
                         state_to_store.to(ssm_cache.dtype()));
  } else {
    torch::Tensor init_state;
    if (ssm_cache.defined() && ssm_cache.numel() > 0) {
      init_state = torch::index_select(ssm_cache, 0, linear_state_base_indices);
      if (!use_fla_ssm_state_layout()) {
        init_state = init_state.transpose(-1, -2);
      }
    }
    torch::Tensor last_state;
    std::tie(core_attn_out, last_state) = gated_delta_rule_decode(
        processed_q, processed_k, processed_v, g, beta, init_state);

    auto state_to_store =
        use_fla_ssm_state_layout() ? last_state : last_state.transpose(-1, -2);
    ssm_cache.index_put_({linear_state_base_indices},
                         state_to_store.to(ssm_cache.dtype()));
  }

  // ---- Z-gate + Output Projection ----
  auto z_reshaped = z.reshape({-1, z.size(-1)});
  auto core_attn_out_reshaped =
      core_attn_out.reshape({-1, core_attn_out.size(-1)});
  auto norm_out = norm_->forward(core_attn_out_reshaped, z_reshaped);
  norm_out = norm_out.view(z.sizes().vec());
  norm_out = norm_out.reshape({-1, norm_out.size(2), norm_out.size(3)});

  auto rearranged_norm =
      norm_out.reshape({norm_out.size(0), norm_out.size(1) * norm_out.size(2)});
  rearranged_norm = reshape_qkvz_unpad(attn_metadata, rearranged_norm);
  if (rearranged_norm.size(0) > original_num_tokens) {
    rearranged_norm =
        rearranged_norm.slice(0, 0, original_num_tokens).contiguous();
  }
  return o_proj_->forward(rearranged_norm);
}

torch::Tensor Qwen3GatedDeltaNetBaseImpl::reshape_qkvz_unpad(
    const AttentionMetadata& attn_metadata,
    const torch::Tensor& padded_qkvz) const {
  const bool has_padded_queries =
      attn_metadata.is_prefill || attn_metadata.is_chunked_prefill;
  if (!has_padded_queries) {
    return padded_qkvz;
  }
  const bool has_host_lens = !attn_metadata.q_seq_lens_vec.empty();
  const int64_t bs =
      has_host_lens ? static_cast<int64_t>(attn_metadata.q_seq_lens_vec.size())
                    : attn_metadata.q_seq_lens.size(0);
  const int64_t max_len = attn_metadata.max_query_len;
  auto reshaped_qkvz = padded_qkvz.reshape({bs, max_len, -1});
  std::vector<torch::Tensor> valid_batches;
  valid_batches.reserve(bs);
  for (int64_t b = 0; b < bs; ++b) {
    const int64_t ori_len =
        has_host_lens ? attn_metadata.q_seq_lens_vec[b]
                      : attn_metadata.q_seq_lens[b].template item<int64_t>();
    valid_batches.emplace_back(reshaped_qkvz[b].slice(0, 0, ori_len));
  }
  if (valid_batches.size() == 1) {
    return valid_batches[0].contiguous();
  }
  return torch::cat(valid_batches, 0).contiguous();
}

torch::Tensor Qwen3GatedDeltaNetBaseImpl::get_linear_state_indices(
    const ModelInputParams& input_params,
    const torch::Device& device) const {
  CHECK(!input_params.embedding.linear_state_ids.empty())
      << "linear_state_ids must be populated for gated delta net";
  return torch::tensor(
      input_params.embedding.linear_state_ids,
      torch::TensorOptions().dtype(torch::kInt).device(device));
}

torch::Tensor Qwen3GatedDeltaNetBaseImpl::reshape_qkvz_with_pad(
    const AttentionMetadata& attn_metadata,
    const torch::Tensor& qkvz) const {
  const bool has_host_lens = !attn_metadata.q_seq_lens_vec.empty();
  const int64_t bs =
      has_host_lens ? static_cast<int64_t>(attn_metadata.q_seq_lens_vec.size())
                    : attn_metadata.q_seq_lens.size(0);
  const int64_t max_len = attn_metadata.max_query_len;
  const bool need_padding =
      attn_metadata.is_prefill || attn_metadata.is_chunked_prefill;
  if (!need_padding) {
    return qkvz.reshape({bs, -1, qkvz.size(-1)});
  }
  std::vector<torch::Tensor> batches;
  batches.reserve(bs);
  int64_t idx = 0;
  for (int64_t b = 0; b < bs; ++b) {
    const int64_t cur_len = has_host_lens ? attn_metadata.q_seq_lens_vec[b] : 0;
    torch::Tensor batch = qkvz.slice(0, idx, idx + cur_len).contiguous();
    idx += cur_len;
    if (batch.size(0) != max_len) {
      batch = batch.size(0) > max_len
                  ? batch.slice(0, 0, max_len).contiguous()
                  : torch::nn::functional::pad(
                        batch,
                        torch::nn::functional::PadFuncOptions(
                            {0, 0, 0, max_len - batch.size(0)}))
                        .contiguous();
    }
    batches.emplace_back(batch);
  }
  return torch::stack(batches, 0).contiguous();
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
Qwen3GatedDeltaNetBaseImpl::process_mixed_qkv(torch::Tensor& mixed_qkv) const {
  mixed_qkv = mixed_qkv.transpose(1, 2);
  const int64_t batch_size = mixed_qkv.size(0);
  const int64_t seq_len = mixed_qkv.size(1);
  const std::vector<int64_t> split_sizes = {
      k_size_ / tp_size_, k_size_ / tp_size_, v_size_ / tp_size_};
  auto processed_qkv = torch::split(mixed_qkv, split_sizes, 2);
  auto processed_q = processed_qkv[0].reshape(
      {batch_size, seq_len, num_k_heads_ / tp_size_, head_k_dim_});
  auto processed_k = processed_qkv[1].reshape(
      {batch_size, seq_len, num_k_heads_ / tp_size_, head_k_dim_});
  auto processed_v = processed_qkv[2].reshape(
      {batch_size, seq_len, num_v_heads_ / tp_size_, head_v_dim_});
  return std::make_tuple(processed_q, processed_k, processed_v);
}

// =========================================================================
// Qwen3_5GatedDeltaNet DCU forward (direct projections, no merge/split)
// =========================================================================

torch::Tensor Qwen3_5GatedDeltaNetImpl::forward(
    const torch::Tensor& hidden_states,
    const AttentionMetadata& attn_metadata,
    KVCache& kv_cache,
    const ModelInputParams& input_params) {
  auto mixed_qkv_flat = in_proj_qkv_->forward(hidden_states);
  auto z_flat = in_proj_z_->forward(hidden_states);
  auto b_flat = in_proj_b_->forward(hidden_states);
  auto a_flat = in_proj_a_->forward(hidden_states);

  const int64_t num_tokens = hidden_states.size(0);
  const bool is_prefill =
      attn_metadata.is_prefill || attn_metadata.is_chunked_prefill;

  int64_t batch_size, seq_len;
  torch::Tensor mixed_qkv, z, b, a;

  if (is_prefill) {
    const bool has_host = !attn_metadata.q_seq_lens_vec.empty();
    batch_size = has_host
                     ? static_cast<int64_t>(attn_metadata.q_seq_lens_vec.size())
                     : attn_metadata.q_seq_lens.size(0);
    seq_len = attn_metadata.max_query_len;
    mixed_qkv = reshape_qkvz_with_pad(attn_metadata, mixed_qkv_flat);
    z = reshape_qkvz_with_pad(attn_metadata, z_flat);
    b = reshape_qkvz_with_pad(attn_metadata, b_flat);
    a = reshape_qkvz_with_pad(attn_metadata, a_flat);
  } else {
    batch_size = mixed_qkv_flat.size(0);
    seq_len = 1;
    mixed_qkv = mixed_qkv_flat.reshape({batch_size, seq_len, -1});
    z = z_flat.reshape({batch_size, seq_len, -1});
    b = b_flat.reshape({batch_size, seq_len, -1});
    a = a_flat.reshape({batch_size, seq_len, -1});
  }

  z = z.reshape({batch_size, seq_len, num_v_heads_ / tp_size_, head_v_dim_});

  // ---- Causal Conv1d ----
  torch::Tensor conv_cache = kv_cache.get_conv_cache();
  torch::Tensor conv_weight = conv1d_->weight();

  if (is_prefill) {
    auto cu_seqlens_vec = extract_cu_seqlens(attn_metadata);
    auto conv_input = reshape_qkvz_unpad(attn_metadata, mixed_qkv);
    std::vector<int64_t> state_indices(
        input_params.embedding.linear_state_ids.begin(),
        input_params.embedding.linear_state_ids.end());
    mixed_qkv = causal_conv1d_prefill(conv_input,
                                      conv_weight,
                                      conv_cache,
                                      cu_seqlens_vec,
                                      state_indices,
                                      conv_kernel_size_);
    mixed_qkv = reshape_qkvz_with_pad(attn_metadata, mixed_qkv);
  } else {
    auto flat_input = mixed_qkv.reshape({batch_size * seq_len, -1});
    auto state_indices =
        get_linear_state_indices(input_params, mixed_qkv.device());
    mixed_qkv = causal_conv1d_decode(flat_input,
                                     conv_weight,
                                     conv_cache,
                                     state_indices,
                                     conv_kernel_size_)
                    .reshape({batch_size, seq_len, -1});
  }

  mixed_qkv = mixed_qkv.transpose(1, 2);
  auto [q, k, v] = process_mixed_qkv(mixed_qkv);

  // ---- Gating ----
  const auto beta = torch::sigmoid(b);
  auto softplus_out = torch::nn::functional::softplus(
      a.to(torch::kFloat32) + dt_bias_,
      torch::nn::functional::SoftplusFuncOptions().beta(1.0).threshold(20.0));
  auto g = (-A_log_.exp() * softplus_out).to(b.dtype()).contiguous();

  // ---- Gated Delta Rule ----
  torch::Tensor ssm_cache = kv_cache.get_ssm_cache();
  auto state_indices = get_linear_state_indices(input_params, q.device());
  auto linear_state_base = state_indices;

  torch::Tensor core_attn_out;
  if (is_prefill) {
    std::vector<torch::Tensor> pq, pk, pv, pg, pbeta;
    for (int64_t bi = 0; bi < batch_size; ++bi) {
      const int64_t vlen = attn_metadata.q_seq_lens_vec[bi];
      pq.push_back(q[bi].narrow(0, 0, vlen));
      pk.push_back(k[bi].narrow(0, 0, vlen));
      pv.push_back(v[bi].narrow(0, 0, vlen));
      pg.push_back(g[bi].narrow(0, 0, vlen));
      pbeta.push_back(beta[bi].narrow(0, 0, vlen));
    }
    auto cat_q = torch::cat(pq, 0).unsqueeze(0);
    auto cat_k = torch::cat(pk, 0).unsqueeze(0);
    auto cat_v = torch::cat(pv, 0).unsqueeze(0);
    auto cat_g = torch::cat(pg, 0).unsqueeze(0);
    auto cat_beta = torch::cat(pbeta, 0).unsqueeze(0);

    auto init_state = torch::index_select(ssm_cache, 0, linear_state_base);
    auto real_cu = build_real_cu_seqlens(attn_metadata.q_seq_lens_vec);

    torch::Tensor last_state;
    std::tie(core_attn_out, last_state) = gated_delta_rule_prefill(
        cat_q, cat_k, cat_v, cat_g, cat_beta, init_state, real_cu);

    core_attn_out = core_attn_out.squeeze(0);
    auto out3d = torch::zeros_like(v);
    int64_t off = 0;
    for (int64_t bi = 0; bi < batch_size; ++bi) {
      const int64_t vlen = attn_metadata.q_seq_lens_vec[bi];
      out3d[bi].narrow(0, 0, vlen).copy_(core_attn_out.narrow(0, off, vlen));
      off += vlen;
    }
    core_attn_out = out3d;
    ssm_cache.index_put_({linear_state_base}, last_state.to(ssm_cache.dtype()));
  } else {
    auto init_state = torch::index_select(ssm_cache, 0, linear_state_base);
    torch::Tensor last_state;
    std::tie(core_attn_out, last_state) =
        gated_delta_rule_decode(q, k, v, g, beta, init_state);
    ssm_cache.index_put_({linear_state_base}, last_state.to(ssm_cache.dtype()));
  }

  // ---- Z-gate + Output Projection ----
  auto z_2d = z.reshape({-1, head_v_dim_});
  auto co_2d = core_attn_out.reshape({-1, head_v_dim_});
  auto norm_out = norm_->forward(co_2d, z_2d);
  norm_out = norm_out.reshape(
      {batch_size, seq_len, num_v_heads_ / tp_size_, head_v_dim_});
  norm_out = norm_out.reshape({norm_out.size(0),
                               norm_out.size(1),
                               norm_out.size(2) * norm_out.size(3)});
  auto unpad = reshape_qkvz_unpad(attn_metadata, norm_out);
  if (unpad.size(0) > num_tokens) {
    unpad = unpad.slice(0, 0, num_tokens).contiguous();
  }
  return o_proj_->forward(unpad);
}

}  // namespace layer
}  // namespace xllm

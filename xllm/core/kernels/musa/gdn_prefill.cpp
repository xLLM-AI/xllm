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

#include <glog/logging.h>
#include <unistd.h>

#include <cstdlib>
#include <optional>
#include <sstream>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "core/common/macros.h"
#include "core/kernels/musa/musa_ops_api.h"
#include "core/kernels/param.h"
#include "core/util/env_var.h"

namespace xllm {
namespace kernel {
namespace musa {

namespace {

inline torch::Tensor l2norm_last(const torch::Tensor& x, double eps) {
  return x / (x.pow(2).sum(-1, /*keepdim=*/true) + eps).sqrt();
}

}  // namespace

std::pair<torch::Tensor, torch::Tensor> fused_recurrent_gated_delta_rule(
    FusedRecurrentGatedDeltaRuleParams& params) {
  auto query = params.q;
  auto key = params.k;
  auto value = params.v;
  auto g = params.g;
  const auto initial_dtype = query.scalar_type();

  if (params.use_qk_l2norm_in_kernel) {
    query = l2norm_last(query, 1e-6);
    key = l2norm_last(key, 1e-6);
  }

  auto to_f32_bhtd = [](const torch::Tensor& x) {
    return x.transpose(1, 2).contiguous().to(torch::kFloat32);
  };
  query = to_f32_bhtd(query);
  key = to_f32_bhtd(key);
  value = to_f32_bhtd(value);
  g = to_f32_bhtd(g);
  torch::Tensor beta_f32;
  if (params.beta.has_value() && params.beta.value().defined()) {
    beta_f32 = to_f32_bhtd(params.beta.value());
  } else {
    beta_f32 = torch::ones_like(g);
  }

  const int64_t batch_size = query.size(0);
  const int64_t num_heads = query.size(1);
  const int64_t sequence_length = query.size(2);
  const int64_t k_head_dim = key.size(-1);
  const int64_t v_head_dim = value.size(-1);
  const float scale_val =
      params.scale.value_or(1.0f / std::sqrt(static_cast<float>(k_head_dim)));
  query = query * scale_val;

  torch::Tensor last_recurrent_state;
  if (params.initial_state.has_value() &&
      params.initial_state.value().defined()) {
    last_recurrent_state =
        params.initial_state.value().to(torch::kFloat32).transpose(-1, -2);
  } else {
    last_recurrent_state = torch::zeros(
        {batch_size, num_heads, k_head_dim, v_head_dim},
        torch::TensorOptions().dtype(torch::kFloat32).device(value.device()));
  }

  auto core_attn_out = torch::zeros(
      {batch_size, num_heads, sequence_length, v_head_dim},
      torch::TensorOptions().dtype(torch::kFloat32).device(value.device()));

  for (int64_t i = 0; i < sequence_length; ++i) {
    auto q_t = query.select(2, i);
    auto k_t = key.select(2, i);
    auto v_t = value.select(2, i);
    auto g_t = g.select(2, i);
    auto beta_t = beta_f32.select(2, i);
    auto g_exp = g_t.exp().unsqueeze(-1).unsqueeze(-1);
    last_recurrent_state.mul_(g_exp);
    auto kv_mem = (last_recurrent_state * k_t.unsqueeze(-1)).sum(-2);
    auto delta = (v_t - kv_mem) * beta_t.unsqueeze(-1);
    last_recurrent_state.add_(k_t.unsqueeze(-1) * delta.unsqueeze(-2));
    core_attn_out.select(2, i) =
        (last_recurrent_state * q_t.unsqueeze(-1)).sum(-2);
  }

  core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype);
  last_recurrent_state = last_recurrent_state.transpose(-1, -2);
  return {core_attn_out, last_recurrent_state};
}

std::pair<torch::Tensor, torch::Tensor> chunk_gated_delta_rule(
    ChunkGatedDeltaRuleParams& params) {
  auto query = params.q;
  auto key = params.k;
  auto value = params.v;
  auto g = params.g;
  auto beta = params.beta;
  const int64_t chunk_size = 64;
  const auto initial_dtype = query.dtype();

  if (params.use_qk_l2norm_in_kernel) {
    query = l2norm_last(query, 1e-6);
    key = l2norm_last(key, 1e-6);
  }

  const int64_t Hqk = query.size(2);
  const int64_t Hv = value.size(2);
  if (Hqk != Hv) {
    CHECK(Hv % Hqk == 0) << "chunk_gated_delta_rule: Hv (" << Hv
                         << ") must be a multiple of Hqk (" << Hqk
                         << ") for GQA expansion";
    const int64_t repeat = Hv / Hqk;
    query = query.repeat_interleave(repeat, /*dim=*/2);
    key = key.repeat_interleave(repeat, /*dim=*/2);
  }

  auto to_f32_thd = [](const torch::Tensor& x) {
    return x.transpose(1, 2).contiguous().to(torch::kFloat32);
  };
  query = to_f32_thd(query);
  key = to_f32_thd(key);
  value = to_f32_thd(value);
  beta = beta.transpose(1, 2).contiguous().to(torch::kFloat32);
  g = g.transpose(1, 2).contiguous().to(torch::kFloat32);

  const int64_t batch_size = query.size(0);
  const int64_t num_heads = query.size(1);
  const int64_t sequence_length = query.size(2);
  const int64_t k_head_dim = key.size(-1);
  const int64_t v_head_dim = value.size(-1);

  const int64_t pad_size =
      (chunk_size - sequence_length % chunk_size) % chunk_size;
  using PadOpts = torch::nn::functional::PadFuncOptions;
  if (pad_size != 0) {
    query = torch::nn::functional::pad(query, PadOpts({0, 0, 0, pad_size}));
    key = torch::nn::functional::pad(key, PadOpts({0, 0, 0, pad_size}));
    value = torch::nn::functional::pad(value, PadOpts({0, 0, 0, pad_size}));
    beta = torch::nn::functional::pad(beta, PadOpts({0, pad_size}));
    g = torch::nn::functional::pad(g, PadOpts({0, pad_size}));
  }
  const int64_t total_sequence_length = sequence_length + pad_size;
  const float scale =
      params.scale.value_or(1.0f / std::sqrt(static_cast<float>(k_head_dim)));
  query = query * scale;
  auto v_beta = value * beta.unsqueeze(-1);
  auto k_beta = key * beta.unsqueeze(-1);

  auto reshape_to_chunks = [chunk_size](const torch::Tensor& x) {
    return x.reshape(
        {x.size(0), x.size(1), x.size(2) / chunk_size, chunk_size, x.size(3)});
  };
  query = reshape_to_chunks(query);
  key = reshape_to_chunks(key);
  value = reshape_to_chunks(value);
  k_beta = reshape_to_chunks(k_beta);
  v_beta = reshape_to_chunks(v_beta);
  g = g.reshape({g.size(0), g.size(1), g.size(2) / chunk_size, chunk_size});

  auto mask = torch::triu(
      torch::ones(
          {chunk_size, chunk_size},
          torch::TensorOptions().dtype(torch::kBool).device(query.device())),
      0);
  g = g.cumsum(-1);
  auto g_diff = g.unsqueeze(-1) - g.unsqueeze(-2);
  auto decay_mask = g_diff.tril().exp().to(torch::kFloat32).tril();
  auto attn = -(torch::matmul(k_beta, key.transpose(-1, -2)) * decay_mask)
                   .masked_fill(mask, 0.0);
  for (int64_t i = 1; i < chunk_size; ++i) {
    if (!attn.is_contiguous()) {
      attn = attn.contiguous();
    }
    auto row = attn.slice(-2, i, i + 1).slice(-1, 0, i).squeeze(-2).clone();
    auto sub = attn.slice(-2, 0, i).slice(-1, 0, i).clone();
    auto row_final = row + (row.unsqueeze(-1) * sub).sum(-2);
    attn.index_put_({torch::indexing::Ellipsis,
                     torch::indexing::Slice(i, i + 1),
                     torch::indexing::Slice(0, i)},
                    row_final.unsqueeze(-2));
  }
  attn = attn +
         torch::eye(
             chunk_size,
             torch::TensorOptions().dtype(attn.dtype()).device(attn.device()));
  value = torch::matmul(attn, v_beta);
  auto k_cumdecay = torch::matmul(attn, k_beta * g.exp().unsqueeze(-1));

  torch::Tensor last_recurrent_state;
  if (params.initial_state.has_value() &&
      params.initial_state.value().defined()) {
    last_recurrent_state = params.initial_state.value().to(value.dtype());
  } else {
    last_recurrent_state = torch::zeros(
        {batch_size, num_heads, k_head_dim, v_head_dim},
        torch::TensorOptions().dtype(value.dtype()).device(value.device()));
  }
  auto core_attn_out = torch::zeros_like(value);
  const int64_t num_chunks = total_sequence_length / chunk_size;

  auto upper_mask = torch::triu(
      torch::ones(
          {chunk_size, chunk_size},
          torch::TensorOptions().dtype(torch::kBool).device(query.device())),
      1);
  for (int64_t i = 0; i < num_chunks; ++i) {
    auto q_i = query.select(2, i);
    auto k_i = key.select(2, i);
    auto v_i = value.select(2, i);
    auto attn_i =
        (torch::matmul(q_i, k_i.transpose(-1, -2)) * decay_mask.select(2, i))
            .masked_fill_(upper_mask, 0.0);
    auto v_prime = torch::matmul(k_cumdecay.select(2, i), last_recurrent_state);
    auto v_new = v_i - v_prime;
    auto attn_inter = torch::matmul(q_i * g.select(2, i).unsqueeze(-1).exp(),
                                    last_recurrent_state);
    core_attn_out.select(2, i) = attn_inter + torch::matmul(attn_i, v_new);
    auto g_i_last = g.select(2, i).select(-1, -1).unsqueeze(-1);
    auto g_exp_term = (g_i_last - g.select(2, i)).exp().unsqueeze(-1);
    auto k_g_exp = (k_i * g_exp_term).transpose(-1, -2).contiguous();
    last_recurrent_state = last_recurrent_state * g_i_last.unsqueeze(-1).exp() +
                           torch::matmul(k_g_exp, v_new);
  }
  const auto s = core_attn_out.sizes();
  core_attn_out = core_attn_out.reshape({s[0], s[1], s[2] * s[3], s[4]});
  core_attn_out = core_attn_out.slice(2, 0, sequence_length);
  core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype);
  return {core_attn_out, last_recurrent_state};
}

namespace {

constexpr int64_t kGdnChunkSize = 64;

bool is_readable_file(const std::string& path) {
  static thread_local std::unordered_map<std::string, bool> availability;
  const auto cached = availability.find(path);
  if (cached != availability.end()) {
    return cached->second;
  }
  const bool available = ::access(path.c_str(), R_OK) == 0;
  availability.emplace(path, available);
  return available;
}

int64_t chunk_pad_size(int64_t seq_len, int64_t chunk_size) {
  return (chunk_size - seq_len % chunk_size) % chunk_size;
}

torch::Tensor pad_time_dim_4d(const torch::Tensor& tensor, int64_t pad_size) {
  if (pad_size == 0) {
    return tensor;
  }
  return torch::nn::functional::pad(
      tensor, torch::nn::functional::PadFuncOptions({0, 0, 0, 0, 0, pad_size}));
}

torch::Tensor pad_time_dim_3d(const torch::Tensor& tensor,
                              int64_t pad_size,
                              double pad_value) {
  if (pad_size == 0) {
    return tensor;
  }
  return torch::nn::functional::pad(
      tensor,
      torch::nn::functional::PadFuncOptions({0, 0, 0, pad_size})
          .mode(torch::kConstant)
          .value(pad_value));
}

std::string mate_gdn_dtype_suffix(torch::ScalarType dtype) {
  if (dtype == torch::kBFloat16) {
    return "bf16";
  }
  if (dtype == torch::kFloat16) {
    return "f16";
  }
  LOG(FATAL) << "mate GDN prefill expects bfloat16 or float16 q/k/v";
}

void l2norm_last_dim(torch::Tensor& tensor) {
  constexpr double kEps = 1e-6;
  tensor =
      tensor / (tensor.pow(2).sum(/*dim=*/-1, /*keepdim=*/true) + kEps).sqrt();
}

void l2norm_qk_last_dim(torch::Tensor& query,
                        torch::Tensor& key,
                        bool allow_inplace) {
  const bool pair_supported = query.sizes() == key.sizes() &&
                              query.scalar_type() == key.scalar_type() &&
                              query.dim() == 4 && query.stride(-1) == 1 &&
                              key.stride(-1) == 1 && query.size(-1) == 128;
  if (pair_supported) {
    if (allow_inplace) {
      l2_norm_pair_fused_inplace(query, key, /*eps=*/1e-6);
    } else {
      std::tie(query, key) = l2_norm_pair_fused(query, key, /*eps=*/1e-6);
    }
    return;
  }
  l2norm_last_dim(query);
  l2norm_last_dim(key);
}

// Exact-shape scratch reused sequentially across GDN layers.
struct MateGdnPrefillScratch {
  torch::Tensor a;
  torch::Tensor output;
  torch::Tensor final_state;
  torch::Tensor cu_seqlens;
  torch::Tensor beta_f32;
  torch::Tensor g_f32;
  torch::Tensor h0;
};

MateGdnPrefillScratch& mate_gdn_prefill_scratch() {
  static thread_local MateGdnPrefillScratch scratch;
  return scratch;
}

torch::Tensor ensure_scratch_tensor(torch::Tensor& buf,
                                    torch::IntArrayRef sizes,
                                    const torch::TensorOptions& opts) {
  // TileLang buffers must retain exact contiguous strides.
  const bool exact_reuse = buf.defined() && buf.sizes().equals(sizes) &&
                           buf.is_contiguous() &&
                           buf.scalar_type() == opts.dtype().toScalarType() &&
                           buf.device() == opts.device();
  if (exact_reuse) {
    return buf;
  }
  buf = torch::empty(sizes, opts);
  return buf;
}

torch::Tensor as_fp32_contig(const torch::Tensor& src,
                             torch::Tensor& scratch_buf) {
  if (src.scalar_type() == torch::kFloat32 && src.is_contiguous()) {
    return src;
  }
  auto out = ensure_scratch_tensor(
      scratch_buf, src.sizes(), src.options().dtype(torch::kFloat32));
  out.copy_(src);
  return out;
}

// Host cu_seqlens helpers require CPU lengths.
bool cu_seqlens_all_full(c10::ArrayRef<int32_t> cu, int64_t max_len) {
  const int64_t num_seqs = static_cast<int64_t>(cu.size()) - 1;
  for (int64_t seq_idx = 0; seq_idx < num_seqs; ++seq_idx) {
    if (static_cast<int64_t>(cu[static_cast<size_t>(seq_idx) + 1] -
                             cu[static_cast<size_t>(seq_idx)]) != max_len) {
      return false;
    }
  }
  return true;
}

std::vector<int32_t> materialize_cu_seqlens_host(
    const musa::MateGatedDeltaRulePrefillParams& params) {
  if (params.cu_seqlens_host.has_value() && !params.cu_seqlens_host->empty()) {
    return *params.cu_seqlens_host;
  }
  CHECK(params.cu_seqlens.has_value() && params.cu_seqlens->defined())
      << "mate GDN prefill needs cu_seqlens_host or device cu_seqlens";
  // Eager multi-sequence fallback materializes lengths on the host.
  const auto cu_cpu =
      params.cu_seqlens->to(torch::kCPU).to(torch::kInt32).contiguous();
  const int32_t* ptr = cu_cpu.data_ptr<int32_t>();
  return std::vector<int32_t>(ptr, ptr + cu_cpu.numel());
}

// Compute per-chunk cumulative log-decay without host access for one sequence.
torch::Tensor chunk_local_cumsum_log_space(const torch::Tensor& g_log,
                                           int64_t num_seqs,
                                           c10::ArrayRef<int32_t> cu) {
  CHECK_EQ(g_log.dim(), 3) << "g_log must be [1, T, H]";
  CHECK_EQ(g_log.size(0), 1) << "g_log must be packed as batch=1";
  const int64_t total_tokens = g_log.size(1);
  CHECK_EQ(total_tokens % kGdnChunkSize, 0)
      << "g_log T must be padded to a multiple of " << kGdnChunkSize;
  CHECK_EQ(static_cast<int64_t>(cu.size()) - 1, num_seqs);

  if (num_seqs == 1) {
    const int64_t num_chunks = total_tokens / kGdnChunkSize;
    const int64_t num_heads = g_log.size(2);
    return g_log.view({1, num_chunks, kGdnChunkSize, num_heads})
        .cumsum(/*dim=*/2)
        .reshape({1, total_tokens, num_heads})
        .contiguous();
  }

  auto out = torch::empty_like(g_log);
  const int64_t num_heads = g_log.size(2);
  for (int64_t seq_idx = 0; seq_idx < num_seqs; ++seq_idx) {
    const int64_t start =
        static_cast<int64_t>(cu[static_cast<size_t>(seq_idx)]);
    const int64_t end =
        static_cast<int64_t>(cu[static_cast<size_t>(seq_idx) + 1]);
    CHECK_GE(end, start);
    CHECK_LE(end, total_tokens);
    const int64_t len = end - start;
    if (len == 0) {
      continue;
    }
    const int64_t n_full = len / kGdnChunkSize;
    const int64_t rem = len % kGdnChunkSize;
    if (n_full > 0) {
      const int64_t full_end = start + n_full * kGdnChunkSize;
      out.slice(/*dim=*/1, start, full_end) =
          g_log.slice(/*dim=*/1, start, full_end)
              .view({1, n_full, kGdnChunkSize, num_heads})
              .cumsum(/*dim=*/2)
              .reshape({1, n_full * kGdnChunkSize, num_heads});
    }
    if (rem > 0) {
      const int64_t rem_start = start + n_full * kGdnChunkSize;
      out.slice(/*dim=*/1, rem_start, end) =
          g_log.slice(/*dim=*/1, rem_start, end).cumsum(/*dim=*/1);
    }
  }
  return out;
}

torch::Tensor pack_time_dim_4d(const torch::Tensor& padded,
                               c10::ArrayRef<int32_t> cu) {
  CHECK_EQ(padded.dim(), 4);
  const int64_t num_seqs = static_cast<int64_t>(cu.size()) - 1;
  CHECK_EQ(padded.size(0), num_seqs);
  const int64_t max_len = padded.size(1);
  // Homogeneous batch: [B, T, H, D] -> [1, B*T, H, D] is a view (no cat).
  if (cu_seqlens_all_full(cu, max_len)) {
    return padded.reshape(
        {1, num_seqs * max_len, padded.size(2), padded.size(3)});
  }
  std::vector<torch::Tensor> parts;
  parts.reserve(static_cast<size_t>(num_seqs));
  for (int64_t seq_idx = 0; seq_idx < num_seqs; ++seq_idx) {
    const int64_t len =
        static_cast<int64_t>(cu[static_cast<size_t>(seq_idx) + 1] -
                             cu[static_cast<size_t>(seq_idx)]);
    parts.emplace_back(padded.select(/*dim=*/0, seq_idx)
                           .narrow(
                               /*dim=*/0, /*start=*/0, /*length=*/len));
  }
  return torch::cat(parts, /*dim=*/0).unsqueeze(0).contiguous();
}

torch::Tensor pack_time_dim_3d(const torch::Tensor& padded,
                               c10::ArrayRef<int32_t> cu) {
  CHECK_EQ(padded.dim(), 3);
  const int64_t num_seqs = static_cast<int64_t>(cu.size()) - 1;
  CHECK_EQ(padded.size(0), num_seqs);
  const int64_t max_len = padded.size(1);
  if (cu_seqlens_all_full(cu, max_len)) {
    return padded.reshape({1, num_seqs * max_len, padded.size(2)});
  }
  std::vector<torch::Tensor> parts;
  parts.reserve(static_cast<size_t>(num_seqs));
  for (int64_t seq_idx = 0; seq_idx < num_seqs; ++seq_idx) {
    const int64_t len =
        static_cast<int64_t>(cu[static_cast<size_t>(seq_idx) + 1] -
                             cu[static_cast<size_t>(seq_idx)]);
    parts.emplace_back(padded.select(/*dim=*/0, seq_idx)
                           .narrow(
                               /*dim=*/0, /*start=*/0, /*length=*/len));
  }
  return torch::cat(parts, /*dim=*/0).unsqueeze(0).contiguous();
}

torch::Tensor unpack_time_dim_4d(const torch::Tensor& packed,
                                 c10::ArrayRef<int32_t> cu,
                                 int64_t max_len) {
  CHECK_EQ(packed.dim(), 4);
  CHECK_EQ(packed.size(0), 1);
  const int64_t num_seqs = static_cast<int64_t>(cu.size()) - 1;
  if (cu_seqlens_all_full(cu, max_len)) {
    CHECK_EQ(packed.size(1), num_seqs * max_len);
    return packed.reshape({num_seqs, max_len, packed.size(2), packed.size(3)});
  }
  auto out = torch::zeros({num_seqs, max_len, packed.size(2), packed.size(3)},
                          packed.options());
  for (int64_t seq_idx = 0; seq_idx < num_seqs; ++seq_idx) {
    const int64_t start =
        static_cast<int64_t>(cu[static_cast<size_t>(seq_idx)]);
    const int64_t end =
        static_cast<int64_t>(cu[static_cast<size_t>(seq_idx) + 1]);
    const int64_t len = end - start;
    out.select(/*dim=*/0, seq_idx)
        .narrow(/*dim=*/0, /*start=*/0, /*length=*/len)
        .copy_(packed.select(/*dim=*/0, 0)
                   .narrow(
                       /*dim=*/0, /*start=*/start, /*length=*/len));
  }
  return out;
}

bool mate_kkt_enabled() {
  static const bool enabled = [] {
    const char* env = std::getenv("XLLM_MATE_KKT");
    if (env == nullptr) {
      return true;
    }
    return env[0] != '0';
  }();
  return enabled;
}

bool mate_gdn_strided_abi_enabled() {
  static const bool enabled = [] {
    const char* env = std::getenv("XLLM_MATE_GDN_DISABLE_STRIDED_ABI");
    return env == nullptr || env[0] != '1';
  }();
  return enabled;
}

bool mate_gdn_kkt_cu_alias_enabled() {
  static const bool enabled = [] {
    const char* env = std::getenv("XLLM_MATE_GDN_DISABLE_KKT_CU_ALIAS");
    return env == nullptr || env[0] != '1';
  }();
  return enabled;
}

bool mate_gdn_c1_partial_kkt_enabled() {
  static const bool enabled = [] {
    const char* env = std::getenv("XLLM_MATE_GDN_C1_PARTIAL_KKT");
    // The fixed KKT kernel masks a partial final chunk.
    return env == nullptr || env[0] != '0';
  }();
  return enabled;
}

std::string get_mate_kkt_solve_uri(int64_t num_q_heads,
                                   int64_t num_v_heads,
                                   torch::ScalarType dtype,
                                   bool is_varlen = false,
                                   bool is_strided = false) {
  std::ostringstream oss;
  oss << "mate_kkt_solve_";
  if (is_varlen) {
    oss << "varlen_";
  }
  if (is_strided) {
    oss << "strided_";
  }
  oss << "hq" << num_q_heads << "_hv" << num_v_heads << "_"
      << mate_gdn_dtype_suffix(dtype);
  return oss.str();
}

bool mate_kkt_module_available(const std::string& uri) {
  const std::string ops_path = util::get_string_env("FLASHINFER_OPS_PATH");
  if (ops_path.empty()) {
    return false;
  }
  const std::string so_path = ops_path + "/" + uri + "/" + uri + ".so";
  return is_readable_file(so_path);
}

// Reference host KKT implementation for [B, T, Hqk, D] keys.
torch::Tensor kkt_solve_torch(const torch::Tensor& key,
                              const torch::Tensor& beta,
                              int64_t chunk_size) {
  const int64_t batch_size = key.size(0);
  const int64_t num_tokens = key.size(1);
  const int64_t num_q_heads = key.size(2);
  const int64_t head_dim = key.size(3);
  const int64_t num_v_heads = beta.size(2);

  auto k_f32 = key.to(torch::kFloat32).contiguous();
  auto beta_f32 = beta.to(torch::kFloat32).contiguous();

  if (num_v_heads != num_q_heads) {
    CHECK(num_v_heads % num_q_heads == 0)
        << "kkt_solve: num_v_heads (" << num_v_heads
        << ") must be a multiple of num_q_heads (" << num_q_heads << ")";
    const int64_t repeat = num_v_heads / num_q_heads;
    k_f32 = k_f32.repeat_interleave(repeat, /*dim=*/2);
  }

  auto k_beta = k_f32 * beta_f32.unsqueeze(-1);

  CHECK_EQ(num_tokens % chunk_size, 0)
      << "torch KKT fallback requires chunk-aligned input";
  const int64_t num_chunks = num_tokens / chunk_size;

  auto k_chunks =
      k_f32.reshape({batch_size, num_chunks, chunk_size, num_v_heads, head_dim})
          .permute({0, 3, 1, 2, 4})
          .contiguous();
  auto kb_chunks =
      k_beta
          .reshape({batch_size, num_chunks, chunk_size, num_v_heads, head_dim})
          .permute({0, 3, 1, 2, 4})
          .contiguous();

  auto gram = torch::matmul(kb_chunks, k_chunks.transpose(-1, -2));

  auto mask = torch::triu(
      torch::ones(
          {chunk_size, chunk_size},
          torch::TensorOptions().dtype(torch::kBool).device(key.device())),
      0);
  auto attn = (-gram).masked_fill(mask, 0.0);

  for (int64_t i = 1; i < chunk_size; ++i) {
    if (!attn.is_contiguous()) {
      attn = attn.contiguous();
    }
    auto row = attn.slice(-2, i, i + 1).slice(-1, 0, i).squeeze(-2).clone();
    auto sub = attn.slice(-2, 0, i).slice(-1, 0, i).clone();
    auto row_final = row + (row.unsqueeze(-1) * sub).sum(-2);
    attn.index_put_({torch::indexing::Ellipsis,
                     torch::indexing::Slice(i, i + 1),
                     torch::indexing::Slice(0, i)},
                    row_final.unsqueeze(-2));
  }

  attn = attn +
         torch::eye(
             chunk_size,
             torch::TensorOptions().dtype(attn.dtype()).device(attn.device()));

  attn = attn.permute({0, 2, 3, 1, 4}).contiguous();
  attn = attn.reshape({batch_size, num_tokens, num_v_heads, chunk_size});

  return attn.to(key.scalar_type());
}

// Mate TileLang KKT solve via TVM FFI.
// ABI: main(k, b, a, num_chunks) with
//   k: [B, T, Hq, K], b: [B, T, Hv], a: [B, T, Hv, 64],
//   num_chunks: B * ceil_div(T, 64). The TileLang kernel masks a partial
//   final chunk, so T need not be padded when B == 1.
torch::Tensor kkt_solve_mate_ffi(
    const torch::Tensor& key,
    const torch::Tensor& beta,
    int64_t chunk_size,
    bool use_strided_abi,
    const std::optional<torch::Tensor>& output = std::nullopt) {
  CHECK_EQ(chunk_size, kGdnChunkSize)
      << "mate KKT solve currently requires chunk_size=" << kGdnChunkSize;
  CHECK_EQ(key.size(3), 128) << "mate KKT solve currently requires K=128";

  const int64_t batch_size = key.size(0);
  const int64_t num_tokens = key.size(1);
  const int64_t num_q_heads = key.size(2);
  const int64_t num_v_heads = beta.size(2);
  CHECK(beta.device() == key.device())
      << "Mate KKT inputs must be on the same device";
  const std::string uri = get_mate_kkt_solve_uri(num_q_heads,
                                                 num_v_heads,
                                                 key.scalar_type(),
                                                 /*is_varlen=*/false,
                                                 use_strided_abi);

  torch::Tensor key_input = use_strided_abi ? key : key.contiguous();
  auto beta_contig = beta.to(torch::kFloat32).contiguous();
  torch::Tensor a;
  if (output.has_value() && output->defined()) {
    a = *output;
    CHECK_EQ(a.dim(), 4);
    CHECK_EQ(a.size(0), batch_size);
    CHECK_EQ(a.size(1), num_tokens);
    CHECK_EQ(a.size(2), num_v_heads);
    CHECK_EQ(a.size(3), chunk_size);
    CHECK(a.is_contiguous() && a.device() == key.device() &&
          a.scalar_type() == key.scalar_type())
        << "Mate KKT output must match key dtype/device and be contiguous";
  } else {
    a = torch::empty({batch_size, num_tokens, num_v_heads, chunk_size},
                     key.options());
  }
  const int32_t num_chunks = static_cast<int32_t>(
      batch_size * ((num_tokens + chunk_size - 1) / chunk_size));

  auto main = get_function(uri, "main");
  main(to_ffi_tensor(key_input),
       to_ffi_tensor(beta_contig),
       to_ffi_tensor(a),
       num_chunks);
  return a;
}

// Varlen KKT ABI: main(k, b, cu_seqlens, a) for packed [1, T, ...] inputs.
torch::Tensor kkt_solve_mate_ffi_varlen(
    const torch::Tensor& key,
    const torch::Tensor& beta,
    const torch::Tensor& cu_seqlens,
    int64_t chunk_size,
    bool use_strided_abi,
    const std::optional<torch::Tensor>& output = std::nullopt) {
  CHECK_EQ(chunk_size, kGdnChunkSize)
      << "mate KKT solve currently requires chunk_size=" << kGdnChunkSize;
  CHECK_EQ(key.size(0), 1) << "varlen KKT expects packed batch dim == 1";
  CHECK_EQ(key.size(3), 128) << "mate KKT solve currently requires K=128";
  CHECK_EQ(cu_seqlens.dim(), 1);
  CHECK_GE(cu_seqlens.size(0), 2);

  const int64_t num_tokens = key.size(1);
  const int64_t num_q_heads = key.size(2);
  const int64_t num_v_heads = beta.size(2);
  CHECK(beta.device() == key.device() && cu_seqlens.device() == key.device())
      << "Mate varlen KKT inputs must be on the same device";
  const std::string uri = get_mate_kkt_solve_uri(num_q_heads,
                                                 num_v_heads,
                                                 key.scalar_type(),
                                                 /*is_varlen=*/true,
                                                 use_strided_abi);

  torch::Tensor key_input = use_strided_abi ? key : key.contiguous();
  auto beta_contig = beta.to(torch::kFloat32).contiguous();
  auto cu_contig = cu_seqlens.to(torch::kInt32).contiguous();
  torch::Tensor a;
  if (output.has_value() && output->defined()) {
    a = *output;
    CHECK_EQ(a.dim(), 4);
    CHECK_EQ(a.size(0), 1);
    CHECK_EQ(a.size(1), num_tokens);
    CHECK_EQ(a.size(2), num_v_heads);
    CHECK_EQ(a.size(3), chunk_size);
    CHECK(a.is_contiguous() && a.device() == key.device() &&
          a.scalar_type() == key.scalar_type())
        << "Mate KKT output must match key dtype/device and be contiguous";
  } else {
    a = torch::empty({1, num_tokens, num_v_heads, chunk_size}, key.options());
  }

  auto main = get_function(uri, "main");
  main(to_ffi_tensor(key_input),
       to_ffi_tensor(beta_contig),
       to_ffi_tensor(cu_contig),
       to_ffi_tensor(a));
  return a;
}

torch::Tensor kkt_solve(
    const torch::Tensor& key,
    const torch::Tensor& beta,
    int64_t chunk_size,
    const std::optional<torch::Tensor>& cu_seqlens = std::nullopt,
    const std::optional<torch::Tensor>& kkt_cu_seqlens = std::nullopt,
    const std::optional<torch::Tensor>& output = std::nullopt,
    bool use_strided_abi = false) {
  const bool is_varlen = cu_seqlens.has_value() && cu_seqlens->defined();
  const bool c1_partial_kkt_candidate =
      is_varlen && mate_gdn_c1_partial_kkt_enabled() && key.size(0) == 1 &&
      cu_seqlens->size(0) == 2;
  // Fall back when only the varlen module is available.
  const bool use_c1_partial_kkt =
      c1_partial_kkt_candidate &&
      mate_kkt_module_available(get_mate_kkt_solve_uri(key.size(2),
                                                       beta.size(2),
                                                       key.scalar_type(),
                                                       /*is_varlen=*/false,
                                                       use_strided_abi));
  const bool use_varlen_kkt = is_varlen && !use_c1_partial_kkt;
  if (mate_kkt_enabled()) {
    const std::string uri = get_mate_kkt_solve_uri(key.size(2),
                                                   beta.size(2),
                                                   key.scalar_type(),
                                                   use_varlen_kkt,
                                                   use_strided_abi);
    if (mate_kkt_module_available(uri)) {
      if (ensure_tilelang_loader()) {
        if (is_varlen) {
          if (use_c1_partial_kkt) {
            return kkt_solve_mate_ffi(
                key, beta, chunk_size, use_strided_abi, output);
          }
          const auto& kkt_cu =
              kkt_cu_seqlens.has_value() && kkt_cu_seqlens->defined()
                  ? *kkt_cu_seqlens
                  : *cu_seqlens;
          return kkt_solve_mate_ffi_varlen(
              key, beta, kkt_cu, chunk_size, use_strided_abi, output);
        }
        return kkt_solve_mate_ffi(
            key, beta, chunk_size, use_strided_abi, output);
      }
      LOG_FIRST_N(WARNING, 1)
          << "[MateKktSolve] TileLang MUSA module loader unavailable; "
             "falling back to torch KKT";
    } else {
      LOG_FIRST_N(WARNING, 1)
          << "[MateKktSolve] module not found for uri=" << uri
          << " under FLASHINFER_OPS_PATH; falling back to torch KKT";
    }
  }
  CHECK(!is_varlen)
      << "torch KKT fallback does not support varlen cu_seqlens; "
         "deploy mate_kkt_solve_varlen_* under FLASHINFER_OPS_PATH";
  return kkt_solve_torch(key, beta, chunk_size);
}
}  // namespace

std::string get_mate_gdn_prefill_simple_uri(int64_t num_q_heads,
                                            int64_t num_v_heads,
                                            torch::ScalarType dtype) {
  std::ostringstream oss;
  oss << "mate_gdn_prefill_hq" << num_q_heads << "_hv" << num_v_heads << "_"
      << mate_gdn_dtype_suffix(dtype);
  return oss.str();
}

std::string get_mate_gdn_prefill_full_uri(int64_t num_q_heads,
                                          int64_t num_v_heads,
                                          torch::ScalarType dtype) {
  std::ostringstream oss;
  oss << "mate_gdn_prefill_full_hq" << num_q_heads << "_hv" << num_v_heads
      << "_" << mate_gdn_dtype_suffix(dtype);
  return oss.str();
}

std::string get_mate_gdn_prefill_full_varlen_uri(int64_t num_q_heads,
                                                 int64_t num_v_heads,
                                                 torch::ScalarType dtype) {
  std::ostringstream oss;
  oss << "mate_gdn_prefill_full_varlen_hq" << num_q_heads << "_hv"
      << num_v_heads << "_" << mate_gdn_dtype_suffix(dtype);
  return oss.str();
}

std::string get_mate_gdn_prefill_full_strided_uri(int64_t num_q_heads,
                                                  int64_t num_v_heads,
                                                  torch::ScalarType dtype) {
  std::ostringstream oss;
  oss << "mate_gdn_prefill_full_strided_hq" << num_q_heads << "_hv"
      << num_v_heads << "_" << mate_gdn_dtype_suffix(dtype);
  return oss.str();
}

std::string get_mate_gdn_prefill_full_varlen_strided_uri(
    int64_t num_q_heads,
    int64_t num_v_heads,
    torch::ScalarType dtype) {
  std::ostringstream oss;
  oss << "mate_gdn_prefill_full_varlen_strided_hq" << num_q_heads << "_hv"
      << num_v_heads << "_" << mate_gdn_dtype_suffix(dtype);
  return oss.str();
}

bool mate_gdn_force_simple_kernel() {
  static const bool enabled = [] {
    const char* env = std::getenv("XLLM_MATE_GDN_PREFILL_SIMPLE");
    return env != nullptr && env[0] == '1';
  }();
  return enabled;
}

bool mate_gdn_force_varlen_kernel() {
  static const bool enabled = [] {
    const char* env = std::getenv("XLLM_MATE_GDN_PREFILL_VARLEN");
    return env != nullptr && env[0] == '1';
  }();
  return enabled;
}

bool mate_gdn_unpadded_c1_enabled() {
  static const bool enabled = [] {
    const char* env = std::getenv("XLLM_MATE_GDN_UNPADDED_C1");
    return env == nullptr || env[0] != '0';
  }();
  return enabled;
}

bool mate_gdn_c1_partial_fixed_enabled() {
  static const bool enabled = [] {
    const char* env = std::getenv("XLLM_MATE_GDN_C1_PARTIAL_FIXED");
    return env == nullptr || env[0] != '0';
  }();
  return enabled;
}

bool mate_gdn_module_available(const std::string& uri) {
  const std::string ops_path = util::get_string_env("FLASHINFER_OPS_PATH");
  if (ops_path.empty()) {
    return false;
  }
  const std::string so_path = ops_path + "/" + uri + "/" + uri + ".so";
  return is_readable_file(so_path);
}

// Use packed varlen when padding exceeds this fraction.
constexpr double kMateGdnPaddedWasteThreshold = 0.05;

double mate_gdn_padded_waste_ratio(c10::ArrayRef<int32_t> cu, int64_t max_len) {
  const int64_t num_seqs = static_cast<int64_t>(cu.size()) - 1;
  if (num_seqs <= 0 || max_len <= 0) {
    return 0.0;
  }
  const int64_t real_tokens =
      static_cast<int64_t>(cu.back()) - static_cast<int64_t>(cu.front());
  const int64_t padded_tokens = num_seqs * max_len;
  if (padded_tokens <= 0 || real_tokens >= padded_tokens) {
    return 0.0;
  }
  return static_cast<double>(padded_tokens - real_tokens) /
         static_cast<double>(padded_tokens);
}

bool mate_gdn_needs_varlen_packing(
    const musa::MateGatedDeltaRulePrefillParams& params,
    int64_t input_batch,
    int64_t input_seq_len) {
  if (mate_gdn_force_varlen_kernel()) {
    return true;
  }
  if (params.cu_seqlens_host.has_value() && !params.cu_seqlens_host->empty()) {
    const auto& cu = *params.cu_seqlens_host;
    if (cu.size() < 2) {
      return false;
    }
    const int64_t num_seqs = static_cast<int64_t>(cu.size()) - 1;
    if (input_batch == 1 && num_seqs > 1) {
      return true;
    }
    if (num_seqs > 1) {
      return mate_gdn_padded_waste_ratio(cu, input_seq_len) >
             kMateGdnPaddedWasteThreshold;
    }
    return false;
  }
  if (params.cu_seqlens.has_value() && params.cu_seqlens->defined()) {
    const int64_t cu_n = params.cu_seqlens->size(0);
    // A device-only multi-sequence B=1 input is already packed.
    return input_batch == 1 && cu_n > 2;
  }
  return false;
}

std::pair<torch::Tensor, torch::Tensor> mate_gated_delta_rule_prefill(
    musa::MateGatedDeltaRulePrefillParams& params) {
  torch::Tensor query = params.q;
  torch::Tensor key = params.k;
  torch::Tensor value = params.v;
  CHECK(query.dim() == 4 && key.dim() == 4 && value.dim() == 4)
      << "mate GDN prefill expects q/k/v shaped [B, T, H, D]";
  CHECK(query.scalar_type() == key.scalar_type() &&
        query.scalar_type() == value.scalar_type())
      << "mate GDN prefill expects q/k/v to share dtype";
  CHECK(query.device() == key.device() && query.device() == value.device())
      << "mate GDN prefill expects q/k/v on the same device";
  CHECK(query.size(0) == key.size(0) && query.size(0) == value.size(0) &&
        query.size(1) == key.size(1) && query.size(1) == value.size(1) &&
        query.size(2) == key.size(2) && query.size(3) == key.size(3))
      << "mate GDN prefill q/k/v shape mismatch";

  const int64_t input_batch = query.size(0);
  const int64_t input_seq_len = query.size(1);
  const int64_t num_q_heads = query.size(2);
  const int64_t num_v_heads = value.size(2);
  const int64_t head_k_dim = query.size(3);
  const int64_t head_v_dim = value.size(3);
  CHECK_GT(num_q_heads, 0);
  CHECK_GT(num_v_heads, 0);
  CHECK_GT(head_k_dim, 0);
  CHECK_GT(head_v_dim, 0);
  CHECK(head_k_dim == head_v_dim)
      << "mate GDN prefill currently requires K == V, got K=" << head_k_dim
      << " V=" << head_v_dim;
  CHECK(num_v_heads % num_q_heads == 0)
      << "mate GDN prefill expects Hv divisible by Hqk";
  CHECK(params.beta.dim() == 3 && params.g.dim() == 3 &&
        params.beta.size(0) == input_batch && params.g.size(0) == input_batch &&
        params.beta.size(1) == input_seq_len &&
        params.g.size(1) == input_seq_len &&
        params.beta.size(2) == num_v_heads && params.g.size(2) == num_v_heads)
      << "mate GDN prefill beta/g shape mismatch";
  CHECK(params.beta.device() == query.device() &&
        params.g.device() == query.device())
      << "mate GDN prefill inputs must be on the same device";

  if (input_batch == 0 || input_seq_len == 0) {
    int64_t state_batch = input_batch;
    if (params.cu_seqlens_host.has_value() &&
        !params.cu_seqlens_host->empty()) {
      state_batch = static_cast<int64_t>(params.cu_seqlens_host->size()) - 1;
    } else if (params.cu_seqlens.has_value() && params.cu_seqlens->defined()) {
      state_batch = params.cu_seqlens->size(0) - 1;
    }
    CHECK_GE(state_batch, 0);
    torch::Tensor output = torch::empty(
        {input_batch, input_seq_len, num_v_heads, head_v_dim}, value.options());
    torch::Tensor final_state;
    if (params.initial_state.has_value() && params.initial_state->defined()) {
      CHECK(params.initial_state->device() == query.device())
          << "Mate GDN initial_state must be on the input device";
      final_state = params.initial_state->to(torch::kFloat32).contiguous();
      CHECK(final_state.dim() == 4 && final_state.size(0) == state_batch &&
            final_state.size(1) == num_v_heads &&
            final_state.size(2) == head_v_dim &&
            final_state.size(3) == head_k_dim)
          << "Mate GDN initial_state shape mismatch";
    } else {
      final_state = torch::zeros(
          {state_batch, num_v_heads, head_v_dim, head_k_dim},
          torch::TensorOptions().dtype(torch::kFloat32).device(query.device()));
    }
    return {output, final_state};
  }

  // Register the TileLang loader before resolving FFI functions.
  ensure_tilelang_loader();

  const std::string full_varlen_uri = get_mate_gdn_prefill_full_varlen_uri(
      num_q_heads, num_v_heads, query.scalar_type());
  const std::string full_uri = get_mate_gdn_prefill_full_uri(
      num_q_heads, num_v_heads, query.scalar_type());
  const std::string full_varlen_strided_uri =
      get_mate_gdn_prefill_full_varlen_strided_uri(
          num_q_heads, num_v_heads, query.scalar_type());
  const std::string full_strided_uri = get_mate_gdn_prefill_full_strided_uri(
      num_q_heads, num_v_heads, query.scalar_type());
  const bool strided_input_supported =
      query.stride(-1) == 1 && key.stride(-1) == 1 && value.stride(-1) == 1;
  const bool allow_strided_abi =
      mate_gdn_strided_abi_enabled() && strided_input_supported;
  const bool strided_varlen_available =
      !mate_gdn_force_simple_kernel() && allow_strided_abi &&
      mate_gdn_module_available(full_varlen_strided_uri) &&
      mate_kkt_module_available(get_mate_kkt_solve_uri(num_q_heads,
                                                       num_v_heads,
                                                       query.scalar_type(),
                                                       /*is_varlen=*/true,
                                                       /*is_strided=*/true));
  const bool legacy_varlen_available =
      !mate_gdn_force_simple_kernel() &&
      mate_gdn_module_available(full_varlen_uri) &&
      mate_kkt_module_available(get_mate_kkt_solve_uri(
          num_q_heads, num_v_heads, query.scalar_type(), /*is_varlen=*/true));
  const bool varlen_available =
      strided_varlen_available || legacy_varlen_available;
  const bool strided_padded_available =
      !mate_gdn_force_simple_kernel() && allow_strided_abi &&
      mate_gdn_module_available(full_strided_uri) &&
      mate_kkt_module_available(get_mate_kkt_solve_uri(num_q_heads,
                                                       num_v_heads,
                                                       query.scalar_type(),
                                                       /*is_varlen=*/false,
                                                       /*is_strided=*/true));
  const bool legacy_padded_available =
      !mate_gdn_force_simple_kernel() && mate_gdn_module_available(full_uri) &&
      mate_kkt_module_available(get_mate_kkt_solve_uri(
          num_q_heads, num_v_heads, query.scalar_type()));

  const bool use_c1_partial_fixed =
      mate_gdn_c1_partial_fixed_enabled() && input_batch == 1 &&
      input_seq_len >= kGdnChunkSize && input_seq_len % kGdnChunkSize != 0 &&
      (strided_padded_available || legacy_padded_available);
  // Full-varlen kernels accept a partial final chunk.
  const bool use_unpadded_c1_varlen =
      !use_c1_partial_fixed && mate_gdn_unpadded_c1_enabled() &&
      input_batch == 1 && input_seq_len >= kGdnChunkSize &&
      input_seq_len % kGdnChunkSize != 0 && varlen_available;
  const bool use_full_varlen =
      !use_c1_partial_fixed && varlen_available &&
      ((params.output.has_value() && params.output->defined()) ||
       (params.final_state.has_value() && params.final_state->defined()) ||
       (params.kkt_output.has_value() && params.kkt_output->defined()) ||
       mate_gdn_needs_varlen_packing(params, input_batch, input_seq_len) ||
       use_unpadded_c1_varlen);
  const bool use_full_padded =
      !use_full_varlen && (strided_padded_available || legacy_padded_available);
  const bool use_strided_abi = (use_full_varlen && strided_varlen_available) ||
                               (use_full_padded && strided_padded_available);
  const std::string uri =
      use_full_varlen
          ? (use_strided_abi ? full_varlen_strided_uri : full_varlen_uri)
          : (use_full_padded
                 ? (use_strided_abi ? full_strided_uri : full_uri)
                 : get_mate_gdn_prefill_simple_uri(
                       num_q_heads, num_v_heads, query.scalar_type()));

  if (!use_strided_abi) {
    query = query.contiguous();
    key = key.contiguous();
    value = value.contiguous();
  }

  // Packed varlen path.
  if (use_full_varlen) {
    std::vector<int32_t> cu_host;
    std::vector<int32_t> cu_host_unpadded;
    int64_t num_seqs = 1;
    bool need_unpack = false;
    const int64_t max_len = input_seq_len;

    const bool has_host_cu =
        params.cu_seqlens_host.has_value() && !params.cu_seqlens_host->empty();
    const bool has_device_cu =
        params.cu_seqlens.has_value() && params.cu_seqlens->defined();

    if (has_host_cu || has_device_cu) {
      cu_host = materialize_cu_seqlens_host(params);
      CHECK_GE(cu_host.size(), 2u);
      num_seqs = static_cast<int64_t>(cu_host.size()) - 1;
      cu_host_unpadded = cu_host;
      if (input_batch == 1) {
        // Already packed [1, total].
      } else {
        CHECK_EQ(input_batch, num_seqs)
            << "padded mate full-varlen batch must match cu_seqlens";
        query = pack_time_dim_4d(query, cu_host);
        key = pack_time_dim_4d(key, cu_host);
        value = pack_time_dim_4d(value, cu_host);
        need_unpack = true;
      }
    } else {
      // Rectangular [B, T] without cu_seqlens: synthesize equal-length cu and
      // pack to [1, B*T].
      CHECK_GE(input_batch, 1);
      num_seqs = input_batch;
      cu_host.assign(static_cast<size_t>(num_seqs) + 1, 0);
      for (int64_t i = 0; i < num_seqs; ++i) {
        cu_host[static_cast<size_t>(i) + 1] =
            static_cast<int32_t>((i + 1) * input_seq_len);
      }
      cu_host_unpadded = cu_host;
      if (input_batch > 1) {
        query = pack_time_dim_4d(query, cu_host);
        key = pack_time_dim_4d(key, cu_host);
        value = pack_time_dim_4d(value, cu_host);
        need_unpack = true;
      }
    }

    const int64_t packed_seq_len = query.size(1);
    const bool skip_chunk_padding =
        use_unpadded_c1_varlen && num_seqs == 1 && !need_unpack;
    const int64_t pad_size =
        skip_chunk_padding ? 0 : chunk_pad_size(packed_seq_len, kGdnChunkSize);
    if (pad_size > 0) {
      query = pad_time_dim_4d(query, pad_size);
      key = pad_time_dim_4d(key, pad_size);
      value = pad_time_dim_4d(value, pad_size);
      cu_host.back() += static_cast<int32_t>(pad_size);
    }
    const int64_t num_tokens = query.size(1);
    CHECK_EQ(query.size(0), 1);

    if (params.use_qk_l2norm_in_kernel) {
      l2norm_qk_last_dim(query, key, params.allow_inplace_qk_l2norm);
    }
    if (!use_strided_abi) {
      query = query.contiguous();
      key = key.contiguous();
      value = value.contiguous();
    }

    auto beta = params.beta.to(torch::kFloat32).contiguous();
    auto g_log = params.g.to(torch::kFloat32).contiguous();
    if (need_unpack) {
      beta = pack_time_dim_3d(beta, cu_host_unpadded);
      g_log = pack_time_dim_3d(g_log, cu_host_unpadded);
    }
    if (pad_size > 0) {
      beta = pad_time_dim_3d(beta, pad_size, 0.0);
      g_log = pad_time_dim_3d(g_log, pad_size, 0.0);
    }

    auto cu_seqlens = [&]() -> torch::Tensor {
      // Prefer an already-on-device cu_seqlens when we did not mutate lengths
      // because host-to-device tensor creation is not graph safe.
      const bool has_device_cu =
          params.cu_seqlens.has_value() && params.cu_seqlens->defined();
      if (has_device_cu && !need_unpack && pad_size == 0) {
        return params.cu_seqlens->to(torch::kInt32).contiguous();
      }
      return torch::tensor(
          cu_host,
          torch::TensorOptions().dtype(torch::kInt32).device(query.device()));
    }();

    torch::Tensor kkt_cu_seqlens;
    if (mate_gdn_kkt_cu_alias_enabled() && !need_unpack && pad_size == 0) {
      // With no pack/pad mutation, live cu_seqlens already ends at
      // num_tokens. KKT only reads the tensor, so alias it instead of doing a
      // redundant device clone and endpoint fill on every GDN layer.
      kkt_cu_seqlens = cu_seqlens;
    } else {
      // Keep the KKT ABI's B+1 shape stable. A bucket endpoint is used only by
      // KKT to cover the fixed launch tail; the recurrent kernel receives the
      // live cu_seqlens above.
      kkt_cu_seqlens = cu_seqlens.clone();
      kkt_cu_seqlens.select(/*dim=*/0, /*index=*/num_seqs)
          .fill_(static_cast<int32_t>(num_tokens));
    }

    torch::Tensor a;
    torch::Tensor h0;
    torch::Tensor output;
    torch::Tensor final_state;
    {
      TvmffiStreamGuard stream_guard(query.device());
      a = kkt_solve(key,
                    beta,
                    kGdnChunkSize,
                    cu_seqlens,
                    kkt_cu_seqlens,
                    params.kkt_output,
                    use_strided_abi);
      if (params.initial_state.has_value() && params.initial_state->defined()) {
        CHECK(params.initial_state->device() == query.device())
            << "Mate GDN initial_state must be on the input device";
        h0 = params.initial_state->scalar_type() == torch::kFloat32 &&
                     params.initial_state->is_contiguous()
                 ? *params.initial_state
                 : params.initial_state->to(torch::kFloat32).contiguous();
      } else {
        h0 = torch::zeros({num_seqs, num_v_heads, head_v_dim, head_k_dim},
                          torch::TensorOptions()
                              .dtype(torch::kFloat32)
                              .device(query.device()));
      }
      CHECK_EQ(h0.size(0), num_seqs);
      CHECK_EQ(h0.size(1), num_v_heads);
      CHECK_EQ(h0.size(2), head_v_dim);
      CHECK_EQ(h0.size(3), head_k_dim);
      if (params.output.has_value() && params.output->defined()) {
        output = *params.output;
        CHECK_EQ(output.dim(), 4);
        CHECK_EQ(output.size(0), 1);
        CHECK_EQ(output.size(1), num_tokens);
        CHECK_EQ(output.size(2), num_v_heads);
        CHECK_EQ(output.size(3), head_v_dim);
        CHECK(output.is_contiguous() && output.device() == value.device() &&
              output.scalar_type() == value.scalar_type())
            << "Mate GDN output must match value dtype/device and be "
               "contiguous";
      } else {
        output = torch::empty({1, num_tokens, num_v_heads, head_v_dim},
                              value.options());
      }
      if (params.final_state.has_value() && params.final_state->defined()) {
        final_state = *params.final_state;
        CHECK_EQ(final_state.dim(), 4);
        CHECK_EQ(final_state.size(0), num_seqs);
        CHECK_EQ(final_state.size(1), num_v_heads);
        CHECK_EQ(final_state.size(2), head_v_dim);
        CHECK_EQ(final_state.size(3), head_k_dim);
        CHECK(final_state.is_contiguous() &&
              final_state.scalar_type() == torch::kFloat32 &&
              final_state.device() == query.device())
            << "Mate GDN final_state must be contiguous FP32 on the input "
               "device";
      } else {
        final_state =
            torch::empty({num_seqs, num_v_heads, head_v_dim, head_k_dim},
                         torch::TensorOptions()
                             .dtype(torch::kFloat32)
                             .device(query.device()));
      }

      auto run = get_function(uri, use_strided_abi ? "main" : "run");
      run(to_ffi_tensor(query),
          to_ffi_tensor(key),
          to_ffi_tensor(value),
          to_ffi_tensor(a),
          to_ffi_tensor(g_log),
          to_ffi_tensor(beta),
          to_ffi_tensor(h0),
          to_ffi_tensor(cu_seqlens),
          to_ffi_tensor(output),
          to_ffi_tensor(final_state));
    }

    if (pad_size > 0) {
      output = output.slice(/*dim=*/1, /*start=*/0, /*end=*/packed_seq_len)
                   .contiguous();
    }
    if (need_unpack) {
      output = unpack_time_dim_4d(output, cu_host_unpadded, max_len);
    }
    return {output, final_state};
  }

  // Rectangular padded path.
  if (use_full_padded) {
    CHECK(!(params.output.has_value() && params.output->defined()) &&
          !(params.final_state.has_value() && params.final_state->defined()) &&
          !(params.kkt_output.has_value() && params.kkt_output->defined()))
        << "caller-owned Mate GDN buffers require the varlen kernel";
    const int64_t batch_size = input_batch;
    const int64_t pad_size =
        use_c1_partial_fixed ? 0 : chunk_pad_size(input_seq_len, kGdnChunkSize);
    if (pad_size > 0) {
      query = pad_time_dim_4d(query, pad_size);
      key = pad_time_dim_4d(key, pad_size);
      value = pad_time_dim_4d(value, pad_size);
    }
    const int64_t num_tokens = query.size(1);
    auto& scratch = mate_gdn_prefill_scratch();

    if (params.use_qk_l2norm_in_kernel) {
      l2norm_qk_last_dim(query, key, params.allow_inplace_qk_l2norm);
    }
    if (!use_strided_abi) {
      if (!query.is_contiguous()) {
        query = query.contiguous();
      }
      if (!key.is_contiguous()) {
        key = key.contiguous();
      }
      if (!value.is_contiguous()) {
        value = value.contiguous();
      }
    }

    auto beta = as_fp32_contig(params.beta, scratch.beta_f32);
    auto g_log = as_fp32_contig(params.g, scratch.g_f32);
    if (pad_size > 0) {
      beta = pad_time_dim_3d(beta, pad_size, 0.0);
      g_log = pad_time_dim_3d(g_log, pad_size, 0.0);
    }

    // The fixed-batch Mate ABI requires this tensor but does not read it.
    auto cu_seqlens = ensure_scratch_tensor(
        scratch.cu_seqlens,
        {batch_size + 1},
        torch::TensorOptions().dtype(torch::kInt32).device(query.device()));

    torch::Tensor a;
    torch::Tensor h0;
    torch::Tensor output;
    torch::Tensor final_state;
    {
      TvmffiStreamGuard stream_guard(query.device());
      CHECK(ensure_tilelang_loader())
          << "TileLang MUSA FFI loader required for Mate KKT reuse path";
      a = ensure_scratch_tensor(
          scratch.a,
          {batch_size, num_tokens, num_v_heads, kGdnChunkSize},
          key.options());
      torch::Tensor key_input =
          use_strided_abi || key.is_contiguous() ? key : key.contiguous();
      auto beta_contig = beta.is_contiguous() ? beta : beta.contiguous();
      const int32_t num_chunks = static_cast<int32_t>(
          batch_size * ((num_tokens + kGdnChunkSize - 1) / kGdnChunkSize));
      const std::string kkt_uri = get_mate_kkt_solve_uri(num_q_heads,
                                                         num_v_heads,
                                                         key.scalar_type(),
                                                         /*is_varlen=*/false,
                                                         use_strided_abi);
      auto main = get_function(kkt_uri, "main");
      main(to_ffi_tensor(key_input),
           to_ffi_tensor(beta_contig),
           to_ffi_tensor(a),
           num_chunks);
      if (params.initial_state.has_value() && params.initial_state->defined()) {
        const auto& init = *params.initial_state;
        CHECK(init.device() == query.device())
            << "Mate GDN initial_state must be on the input device";
        if (init.scalar_type() == torch::kFloat32 && init.is_contiguous()) {
          h0 = init;
        } else {
          h0 = as_fp32_contig(init, scratch.h0);
        }
      } else {
        h0 = ensure_scratch_tensor(
            scratch.h0,
            {batch_size, num_v_heads, head_v_dim, head_k_dim},
            torch::TensorOptions()
                .dtype(torch::kFloat32)
                .device(query.device()));
        h0.zero_();
      }
      CHECK_EQ(h0.size(0), batch_size);
      CHECK_EQ(h0.size(1), num_v_heads);
      CHECK_EQ(h0.size(2), head_v_dim);
      CHECK_EQ(h0.size(3), head_k_dim);
      output = ensure_scratch_tensor(
          scratch.output,
          {batch_size, num_tokens, num_v_heads, head_v_dim},
          value.options());
      final_state = ensure_scratch_tensor(
          scratch.final_state,
          {batch_size, num_v_heads, head_v_dim, head_k_dim},
          torch::TensorOptions().dtype(torch::kFloat32).device(query.device()));

      auto run = get_function(uri, use_strided_abi ? "main" : "run");
      run(to_ffi_tensor(query),
          to_ffi_tensor(key),
          to_ffi_tensor(value),
          to_ffi_tensor(a),
          to_ffi_tensor(g_log),
          to_ffi_tensor(beta),
          to_ffi_tensor(h0),
          to_ffi_tensor(cu_seqlens),
          to_ffi_tensor(output),
          to_ffi_tensor(final_state));
    }

    if (pad_size > 0) {
      output = output.slice(/*dim=*/1, /*start=*/0, /*end=*/input_seq_len)
                   .contiguous();
    }
    return {output, final_state};
  }

  // Simple varlen fallback.
  torch::Tensor cu_seqlens;
  bool packed_input = false;
  int64_t num_seqs = 1;
  std::vector<int32_t> cu_host;
  std::vector<int32_t> cu_host_unpadded;
  const bool has_device_cu =
      params.cu_seqlens.has_value() && params.cu_seqlens->defined();
  const bool has_host_cu =
      params.cu_seqlens_host.has_value() && !params.cu_seqlens_host->empty();

  if (has_host_cu) {
    cu_host = *params.cu_seqlens_host;
    CHECK_GE(cu_host.size(), 2u);
    num_seqs = static_cast<int64_t>(cu_host.size()) - 1;
    cu_host_unpadded = cu_host;
    if (input_batch == 1) {
      packed_input = true;
    } else {
      CHECK_EQ(input_batch, num_seqs)
          << "padded mate prefill batch must match cu_seqlens sequences";
      query = pack_time_dim_4d(query, cu_host);
      key = pack_time_dim_4d(key, cu_host);
      value = pack_time_dim_4d(value, cu_host);
      packed_input = true;
    }
  } else if (has_device_cu) {
    CHECK_EQ(params.cu_seqlens->dim(), 1);
    CHECK_GE(params.cu_seqlens->size(0), 2);
    num_seqs = params.cu_seqlens->size(0) - 1;
    if (input_batch == 1) {
      packed_input = true;
      if (num_seqs > 1) {
        cu_host = materialize_cu_seqlens_host(params);
        cu_host_unpadded = cu_host;
      }
    } else {
      CHECK_EQ(input_batch, num_seqs)
          << "padded mate prefill batch must match cu_seqlens sequences";
      cu_host = materialize_cu_seqlens_host(params);
      cu_host_unpadded = cu_host;
      query = pack_time_dim_4d(query, cu_host);
      key = pack_time_dim_4d(key, cu_host);
      value = pack_time_dim_4d(value, cu_host);
      packed_input = true;
    }
  } else {
    CHECK_EQ(input_batch, 1)
        << "mate GDN varlen prefill requires cu_seqlens when batch > 1";
    num_seqs = 1;
  }

  const int64_t packed_seq_len = query.size(1);
  const int64_t pad_size = chunk_pad_size(packed_seq_len, kGdnChunkSize);
  if (pad_size > 0) {
    query = pad_time_dim_4d(query, pad_size);
    key = pad_time_dim_4d(key, pad_size);
    value = pad_time_dim_4d(value, pad_size);
  }
  const int64_t num_tokens = query.size(1);

  if (num_seqs == 1) {
    cu_seqlens = torch::zeros(
        {2},
        torch::TensorOptions().dtype(torch::kInt32).device(query.device()));
    cu_seqlens.select(0, 1).fill_(static_cast<int64_t>(num_tokens));
    cu_host = {0, static_cast<int32_t>(num_tokens)};
  } else {
    CHECK(!cu_host.empty())
        << "multi-seq mate GDN prefill requires host cu_seqlens";
    if (pad_size > 0) {
      cu_host.back() += static_cast<int32_t>(pad_size);
    }
    cu_seqlens = torch::tensor(
        cu_host,
        torch::TensorOptions().dtype(torch::kInt32).device(query.device()));
  }

  if (params.use_qk_l2norm_in_kernel) {
    l2norm_qk_last_dim(query, key, params.allow_inplace_qk_l2norm);
  }
  query = query.contiguous();
  key = key.contiguous();
  value = value.contiguous();

  auto beta = params.beta.to(torch::kFloat32).contiguous();
  auto g_log = params.g.to(torch::kFloat32).contiguous();
  if (packed_input && input_batch > 1) {
    beta = pack_time_dim_3d(beta, cu_host_unpadded);
    g_log = pack_time_dim_3d(g_log, cu_host_unpadded);
  }
  if (pad_size > 0) {
    beta = pad_time_dim_3d(beta, pad_size, 0.0);
    g_log = pad_time_dim_3d(g_log, pad_size, 0.0);
  }
  g_log = chunk_local_cumsum_log_space(g_log, num_seqs, cu_host);

  torch::Tensor a;
  torch::Tensor h0;
  torch::Tensor output;
  torch::Tensor final_state;
  {
    TvmffiStreamGuard stream_guard(query.device());
    auto run = get_function(uri, "run");
    a = torch::empty({1, num_tokens, num_v_heads, kGdnChunkSize},
                     query.options());
    if (params.initial_state.has_value() && params.initial_state->defined()) {
      CHECK(params.initial_state->device() == query.device())
          << "Mate GDN initial_state must be on the input device";
      h0 = params.initial_state->to(torch::kFloat32).contiguous();
      CHECK_EQ(h0.dim(), 4) << "mate GDN prefill initial_state must be 4D";
      CHECK_EQ(h0.size(0), num_seqs)
          << "mate GDN prefill initial_state batch mismatch";
      CHECK_EQ(h0.size(1), num_v_heads)
          << "mate GDN prefill initial_state head mismatch";
      CHECK_EQ(h0.size(2), head_v_dim)
          << "mate GDN prefill initial_state V dim mismatch";
      CHECK_EQ(h0.size(3), head_k_dim)
          << "mate GDN prefill initial_state K dim mismatch";
    } else {
      h0 = torch::zeros(
          {num_seqs, num_v_heads, head_v_dim, head_k_dim},
          torch::TensorOptions().dtype(torch::kFloat32).device(query.device()));
    }
    output =
        torch::empty({1, num_tokens, num_v_heads, head_v_dim}, value.options());
    final_state = torch::empty(
        {num_seqs, num_v_heads, head_v_dim, head_k_dim},
        torch::TensorOptions().dtype(torch::kFloat32).device(query.device()));

    run(to_ffi_tensor(query),
        to_ffi_tensor(key),
        to_ffi_tensor(value),
        to_ffi_tensor(a),
        to_ffi_tensor(g_log),
        to_ffi_tensor(beta),
        to_ffi_tensor(h0),
        to_ffi_tensor(cu_seqlens),
        to_ffi_tensor(output),
        to_ffi_tensor(final_state));
  }

  if (pad_size > 0) {
    output = output.slice(/*dim=*/1, /*start=*/0, /*end=*/packed_seq_len)
                 .contiguous();
  }

  if (input_batch > 1) {
    output = unpack_time_dim_4d(output, cu_host_unpadded, input_seq_len);
  }
  return {output, final_state};
}

torch::Tensor causal_conv1d(
    const torch::Tensor& x,
    const torch::Tensor& weight,
    const torch::Tensor& conv_state,
    const std::optional<torch::Tensor>& bias_opt,
    const torch::IntArrayRef query_start_loc_opt,
    const torch::IntArrayRef cache_indices_opt,
    const torch::IntArrayRef initial_state_mode_opt,
    const torch::IntArrayRef /*num_accepted_tokens_opt*/,
    int64_t activation_mode,
    int64_t /*pad_slot_id*/,
    int64_t /*run_mode*/) {
  const auto device = x.device();

  auto x_t = x.t().contiguous();
  auto out_t = torch::empty_like(x_t);

  auto qsl_cpu = torch::empty(
      {static_cast<int64_t>(query_start_loc_opt.size())}, torch::kInt32);
  for (size_t i = 0; i < query_start_loc_opt.size(); ++i) {
    qsl_cpu.data_ptr<int32_t>()[i] =
        static_cast<int32_t>(query_start_loc_opt[i]);
  }
  auto query_start_loc = qsl_cpu.to(device);

  torch::Tensor cache_indices;
  if (!cache_indices_opt.empty()) {
    auto ci_cpu = torch::empty({static_cast<int64_t>(cache_indices_opt.size())},
                               torch::kInt32);
    for (size_t i = 0; i < cache_indices_opt.size(); ++i) {
      ci_cpu.data_ptr<int32_t>()[i] =
          static_cast<int32_t>(cache_indices_opt[i]);
    }
    cache_indices = ci_cpu.to(device);
  }

  torch::Tensor has_initial_state;
  if (!initial_state_mode_opt.empty()) {
    auto his_cpu = torch::empty(
        {static_cast<int64_t>(initial_state_mode_opt.size())}, torch::kBool);
    for (size_t i = 0; i < initial_state_mode_opt.size(); ++i) {
      his_cpu.data_ptr<bool>()[i] = (initial_state_mode_opt[i] != 0);
    }
    has_initial_state = his_cpu.to(device);
  }

  const bool silu_activation = (activation_mode != 0);
  const int64_t pad_slot_id = -1;

  causal_conv1d_fwd(x_t,
                    weight,
                    out_t,
                    bias_opt,
                    conv_state,
                    query_start_loc,
                    cache_indices,
                    has_initial_state,
                    silu_activation,
                    pad_slot_id);

  return out_t.t().contiguous().to(x.dtype());
}

torch::Tensor causal_conv1d_prefill(const torch::Tensor& x,
                                    const torch::Tensor& weight,
                                    const torch::Tensor& conv_state,
                                    const std::optional<torch::Tensor>& bias,
                                    const torch::Tensor& query_start_loc,
                                    const torch::Tensor& cache_indices,
                                    const torch::Tensor& has_initial_state,
                                    bool silu_activation) {
  CHECK(query_start_loc.defined() && is_torch_device(query_start_loc.device()))
      << "causal_conv1d_prefill requires device query_start_loc";
  CHECK(cache_indices.defined() && is_torch_device(cache_indices.device()))
      << "causal_conv1d_prefill requires device cache_indices";
  CHECK(has_initial_state.defined() &&
        is_torch_device(has_initial_state.device()))
      << "causal_conv1d_prefill requires device has_initial_state";

  static const bool use_token_major = [] {
    const char* env = std::getenv("XLLM_TOKEN_MAJOR_PREFILL_CONV");
    return env == nullptr || std::string(env) != "0";
  }();
  // Each sequence owns a disjoint output and cache-state row.
  if (use_token_major && x.is_contiguous() && weight.dim() == 2 &&
      weight.size(1) == 4) {
    torch::Tensor out = torch::empty_like(x);
    causal_conv1d_fwd_token_major(x,
                                  weight,
                                  out,
                                  bias,
                                  conv_state,
                                  query_start_loc,
                                  cache_indices,
                                  has_initial_state,
                                  silu_activation,
                                  /*pad_slot_id=*/-1);
    return out;
  }

  auto x_t = x.t().contiguous();
  auto out_t = torch::empty_like(x_t);
  causal_conv1d_fwd(x_t,
                    weight,
                    out_t,
                    bias,
                    conv_state,
                    query_start_loc,
                    cache_indices,
                    has_initial_state,
                    silu_activation,
                    /*pad_slot_id=*/-1);
  return out_t.t().contiguous();
}

}  // namespace musa
}  // namespace kernel
}  // namespace xllm

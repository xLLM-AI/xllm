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

#include "qwen3_5_gated_delta_net.h"

#include <glog/logging.h>
#include <torch/torch.h>

#include <algorithm>
#include <limits>
#include <tuple>
#include <unordered_set>
#include <vector>

#include "core/kernels/npu/xllm_ops/mega_gdn_constants.h"
#include "core/kernels/ops_api.h"
#include "core/util/tensor_helper.h"
#include "core/util/utils.h"

namespace xllm {
namespace layer {

namespace {

// MegaGdnDecode and MegaGdnMtpDecode accept at most 32 sequences per call.
constexpr int64_t kMegaGdnMaxDecodeBatchSize = 32;

void check_tensor(const torch::Tensor& tensor,
                  const char* name,
                  torch::ScalarType dtype,
                  int64_t dim,
                  const torch::Device& device) {
  CHECK(tensor.defined()) << name << " must be defined.";
  CHECK_EQ(tensor.scalar_type(), dtype) << name << " has an invalid dtype.";
  CHECK_EQ(tensor.dim(), dim) << name << " has an invalid rank.";
  CHECK_EQ(tensor.device(), device) << name << " must be on the input device.";
}

void check_live_slots(const std::vector<int32_t>& live_slots,
                      int64_t batch_size,
                      int64_t num_slots) {
  CHECK_EQ(static_cast<int64_t>(live_slots.size()), batch_size)
      << "linear_state_ids must be sequence-scoped.";
  CHECK_GT(num_slots, 0) << "GDN cache must contain at least one slot.";
  std::unordered_set<int32_t> unique_slots;
  unique_slots.reserve(live_slots.size());
  for (const int32_t slot : live_slots) {
    CHECK_GE(slot, 0) << "GDN live slot must be non-negative.";
    CHECK_LT(static_cast<int64_t>(slot), num_slots)
        << "GDN live slot exceeds cache capacity.";
    if (slot != kPaddingLinearStateId) {
      CHECK(unique_slots.emplace(slot).second)
          << "GDN write slots must be unique within a batch, duplicate="
          << slot;
    }
  }
}

torch::Tensor slots_to_device(const std::vector<int32_t>& slots,
                              const torch::Device& device) {
  return make_cpu_tensor(slots).to(device);
}

torch::Tensor graph_safe_indices(const torch::Tensor& indices,
                                 const std::vector<int32_t>& host_values,
                                 int64_t batch_size,
                                 const torch::Device& device,
                                 const char* name) {
  if (!indices.defined()) {
    return slots_to_device(host_values, device);
  }
  check_tensor(indices, name, torch::kInt, 1, device);
  CHECK_EQ(indices.numel(), batch_size) << name << " must be sequence-scoped.";
  CHECK(indices.is_contiguous()) << name << " must be contiguous.";
  return indices;
}

bool is_supported_prefill_head_count(int64_t heads) {
  switch (heads) {
    case 1:
    case 2:
    case 3:
    case 4:
    case 6:
    case 8:
    case 12:
    case 16:
    case 24:
    case 32:
    case 48:
    case 64:
      return true;
    default:
      return false;
  }
}

void check_decode_head_geometry(int64_t key_heads, int64_t value_heads) {
  CHECK_GE(key_heads, 1)
      << "MegaGdn decode operators require at least one local key head.";
  CHECK_LE(key_heads, 16)
      << "MegaGdn decode operators support at most 16 local key heads.";
  CHECK_EQ(key_heads & (key_heads - 1), 0)
      << "MegaGdn decode operators require a power-of-two local key-head "
         "count.";
  CHECK_EQ(value_heads % key_heads, 0)
      << "MegaGdn decode operators require NV to be divisible by NK.";
  const int64_t value_heads_per_key = value_heads / key_heads;
  CHECK_GE(value_heads_per_key, 1) << "MegaGdn decode operators require at "
                                      "least one value head per key head.";
  CHECK_LE(value_heads_per_key, 4)
      << "MegaGdn decode operators support at most four value heads per key "
         "head.";
}

int64_t compute_num_matrices(const std::vector<int32_t>& query_lengths,
                             int64_t num_value_heads) {
  int64_t num_chunks = 0;
  for (const int32_t query_length : query_lengths) {
    CHECK_GT(query_length, 0) << "prefill query lengths must be positive.";
    num_chunks +=
        util::ceil_div<int64_t>(query_length, kernel::npu::kMegaGdnChunkSize);
  }
  return num_chunks * num_value_heads;
}

}  // namespace

namespace qwen3_5_gdn_internal {

const MegaGdnPrefillIndicesCache& get_or_build_prefill_indices(
    const AttentionMetadata& attn_metadata,
    const std::vector<int32_t>& live_slots,
    const std::vector<int64_t>& validity_mask,
    int64_t checkpoint_stride,
    int64_t num_slots,
    const torch::Device& device,
    const std::vector<LinearStateCacheOp>& cache_ops) {
  const int64_t batch_size = static_cast<int64_t>(live_slots.size());
  CHECK(cache_ops.empty() ||
        static_cast<int64_t>(cache_ops.size()) == batch_size)
      << "linear_state_cache_ops must be empty or sequence-scoped.";
  const auto read_slot_at = [&](int64_t row) {
    CHECK(validity_mask[row] == 0 || validity_mask[row] == 1)
        << "linear state validity must be 0 or 1.";
    int32_t slot = validity_mask[row] == 0 ? -1 : live_slots[row];
    if (!cache_ops.empty()) {
      const LinearStateCacheOp& op = cache_ops[row];
      CHECK_EQ(op.linear_state_id, live_slots[row]);
      CHECK(!(op.reset_requested && op.restore_requested))
          << "linear-state reset and restore are mutually exclusive.";
      if (op.reset_requested) {
        CHECK_LT(op.restore_src_slot_id, 0)
            << "linear-state reset must not carry a restore source.";
        CHECK_EQ(validity_mask[row], 0)
            << "linear-state reset row must remain cold after restore.";
      } else if (op.restore_requested) {
        CHECK_GE(op.restore_src_slot_id, 0)
            << "linear-state restore requires a valid source slot.";
        CHECK_EQ(validity_mask[row], 1)
            << "linear-state restored row must be warm after restore.";
      } else if (op.restore_src_slot_id >= 0) {
        CHECK_EQ(validity_mask[row], 1)
            << "linear-state direct-read row must be warm after restore.";
      }
      if (op.restore_src_slot_id >= 0) {
        CHECK_GT(op.restore_src_slot_id, kPaddingLinearStateId);
        CHECK_LT(op.restore_src_slot_id, num_slots);
      }
      if (op.is_direct_read()) {
        slot = op.restore_src_slot_id;
      }
    }
    return slot;
  };
  const auto& cache = attn_metadata.mega_gdn_prefill_indices;
  if (cache.has_value()) {
    const auto& key = cache->key;
    CHECK_EQ(key.device, device) << "cache device changed within one forward";
    CHECK_EQ(static_cast<int64_t>(key.linear_state_ids.size()), batch_size)
        << "cache batch size changed within one forward";
    CHECK_EQ(key.num_slots, num_slots)
        << "cache slot geometry changed within one forward";
    CHECK_EQ(key.checkpoint_stride, checkpoint_stride)
        << "checkpoint stride changed within one forward";
    CHECK(key.linear_state_ids == live_slots)
        << "linear state ids changed within one forward";
    CHECK(key.linear_state_validity_mask == validity_mask)
        << "validity changed within one forward";
    for (int64_t row = 0; row < batch_size; ++row) {
      CHECK_EQ(key.linear_state_read_slots[row], read_slot_at(row))
          << "linear state read slots changed within one forward";
    }
    return cache.value();
  }

  check_live_slots(live_slots, batch_size, num_slots);
  CHECK_EQ(static_cast<int64_t>(validity_mask.size()), batch_size)
      << "linear_state_validity_mask must be sequence-scoped.";
  CHECK_GT(checkpoint_stride, 0) << "checkpoint stride must be positive.";
  CHECK_LE(checkpoint_stride,
           static_cast<int64_t>(std::numeric_limits<int32_t>::max()))
      << "checkpoint stride does not fit the operator int32 ABI.";
  CHECK_LE(num_slots - 1,
           static_cast<int64_t>(std::numeric_limits<int32_t>::max()) /
               checkpoint_stride)
      << "SSM state index does not fit the operator int32 ABI.";

  std::vector<int32_t> conv_read;
  std::vector<int32_t> ssm_read;
  std::vector<int32_t> ssm_write;
  conv_read.reserve(batch_size);
  ssm_read.reserve(batch_size);
  ssm_write.reserve(batch_size);
  const int32_t stride = static_cast<int32_t>(checkpoint_stride);
  for (int64_t i = 0; i < batch_size; ++i) {
    const int32_t live_slot = live_slots[i];
    const int32_t read_slot = read_slot_at(i);
    conv_read.emplace_back(read_slot);
    ssm_read.emplace_back(read_slot < 0 ? -1 : read_slot * stride);
    ssm_write.emplace_back(live_slot * stride);
  }

  MegaGdnPrefillIndicesCache indices{
      .key = {.device = device,
              .num_slots = num_slots,
              .checkpoint_stride = checkpoint_stride,
              .linear_state_ids = live_slots,
              .linear_state_read_slots = conv_read,
              .linear_state_validity_mask = validity_mask}};
  indices.conv_read = slots_to_device(conv_read, device);
  indices.conv_write = slots_to_device(live_slots, device);
  indices.ssm_read = slots_to_device(ssm_read, device);
  indices.ssm_write = slots_to_device(ssm_write, device);
  attn_metadata.mega_gdn_prefill_indices.emplace(std::move(indices));
  return attn_metadata.mega_gdn_prefill_indices.value();
}

}  // namespace qwen3_5_gdn_internal

Qwen3_5GatedDeltaNetImpl::Qwen3_5GatedDeltaNetImpl(
    const ModelArgs& args,
    const QuantArgs& quant_args,
    const ParallelArgs& parallel_args,
    const torch::TensorOptions& options)
    : Qwen3NextGatedDeltaNetImpl(args,
                                 quant_args,
                                 parallel_args,
                                 options,
                                 /*init_projections=*/false) {
  CHECK_EQ(num_k_heads_ % tp_size_, 0)
      << "MegaGdn operators require key heads divisible by TP size.";
  CHECK_EQ(num_v_heads_ % tp_size_, 0)
      << "MegaGdn operators require value heads divisible by TP size.";
  const int64_t local_value_heads = num_v_heads_ / tp_size_;
  CHECK_EQ(conv_kernel_size_, 4)
      << "MegaGdn operators require a width-four causal convolution.";
  CHECK_EQ(head_k_dim_, 128)
      << "MegaGdn operators require a key head dimension of 128.";
  CHECK_EQ(head_v_dim_, 128)
      << "MegaGdn operators require a value head dimension of 128.";
  const int64_t local_key_heads = num_k_heads_ / tp_size_;
  check_decode_head_geometry(local_key_heads, local_value_heads);
  CHECK(is_supported_prefill_head_count(local_value_heads))
      << "MegaGdnPrefill does not support this local value-head count.";
  in_proj_qkv_ = register_module("in_proj_qkv",
                                 ColumnParallelLinear(args.hidden_size(),
                                                      k_size_ * 2 + v_size_,
                                                      /*bias=*/false,
                                                      /*gather_output=*/false,
                                                      quant_args,
                                                      parallel_args.tp_group_,
                                                      options));
  in_proj_z_ = register_module("in_proj_z",
                               ColumnParallelLinear(args.hidden_size(),
                                                    v_size_,
                                                    /*bias=*/false,
                                                    /*gather_output=*/false,
                                                    quant_args,
                                                    parallel_args.tp_group_,
                                                    options));
  in_proj_b_ = register_module("in_proj_b",
                               ColumnParallelLinear(args.hidden_size(),
                                                    num_v_heads_,
                                                    /*bias=*/false,
                                                    /*gather_output=*/false,
                                                    quant_args,
                                                    parallel_args.tp_group_,
                                                    options));
  in_proj_a_ = register_module("in_proj_a",
                               ColumnParallelLinear(args.hidden_size(),
                                                    num_v_heads_,
                                                    /*bias=*/false,
                                                    /*gather_output=*/false,
                                                    quant_args,
                                                    parallel_args.tp_group_,
                                                    options));
}

torch::Tensor Qwen3_5GatedDeltaNetImpl::forward(
    const torch::Tensor& hidden_states,
    const AttentionMetadata& attn_metadata,
    KVCache& kv_cache,
    const ModelInputParams& input_params) {
  if (attn_metadata.is_dummy) {
    return torch::zeros_like(hidden_states);
  }

  const FlashComm1Context* fc1_context = get_current_flash_comm1_context();
  torch::Tensor gathered_hidden_states = hidden_states;
  if (fc1_context != nullptr && is_sequence_sharded(*fc1_context)) {
    gathered_hidden_states = gather_sequence(hidden_states, *fc1_context);
  }
  const int64_t original_num_tokens = gathered_hidden_states.size(0);
  const bool use_spec_verify = input_params.is_spec_verify;
  const bool is_any_prefill =
      attn_metadata.is_prefill || attn_metadata.is_chunked_prefill;

  auto split_inputs =
      project_split_inputs(gathered_hidden_states, attn_metadata);
  CHECK(split_inputs.has_value())
      << "Qwen3.5 requires separate qkv, z, b, and a projections.";
  auto [mixed_qkv, z, b, a] = split_inputs.value();
  const int64_t batch_size = mixed_qkv.size(0);
  const int64_t sequence_length = mixed_qkv.size(1);

  CHECK_GT(batch_size, 0) << "GDN batch size must be positive.";
  CHECK_GT(sequence_length, 0) << "GDN sequence length must be positive.";
  const int64_t local_value_heads = num_v_heads_ / tp_size_;
  const int64_t local_conv_dim = 2 * (num_k_heads_ / tp_size_) * head_k_dim_ +
                                 local_value_heads * head_v_dim_;
  const torch::Device device = mixed_qkv.device();
  torch::Tensor conv_cache = kv_cache.get_conv_cache();
  torch::Tensor ssm_cache = kv_cache.get_ssm_cache();
  torch::Tensor conv_weight = conv1d_->weight();

  check_tensor(mixed_qkv, "mixed_qkv", torch::kBFloat16, 3, device);
  check_tensor(z, "z", torch::kBFloat16, 4, device);
  check_tensor(b, "b", torch::kBFloat16, 3, device);
  check_tensor(a, "a", torch::kBFloat16, 3, device);
  check_tensor(conv_weight, "conv_weight", torch::kBFloat16, 2, device);
  check_tensor(conv_cache, "conv_cache", torch::kBFloat16, 3, device);
  check_tensor(ssm_cache, "ssm_cache", torch::kFloat32, 4, device);
  check_tensor(A_log_, "A_log", torch::kFloat32, 1, device);
  check_tensor(dt_bias_, "dt_bias", torch::kFloat32, 1, device);
  check_tensor(norm_->weight(), "norm_weight", torch::kBFloat16, 1, device);
  CHECK_EQ(mixed_qkv.size(2), local_conv_dim)
      << "mixed_qkv channel count does not match the model geometry.";
  CHECK_EQ(z.size(0), batch_size);
  CHECK_EQ(z.size(1), sequence_length);
  CHECK_EQ(b.size(0), batch_size);
  CHECK_EQ(b.size(1), sequence_length);
  CHECK_EQ(a.size(0), batch_size);
  CHECK_EQ(a.size(1), sequence_length);
  CHECK_EQ(conv_weight.size(0), conv_kernel_size_)
      << "conv weight kernel width does not match the model configuration.";
  CHECK_EQ(conv_weight.size(1), local_conv_dim)
      << "conv weight channel count does not match mixed_qkv.";
  CHECK_EQ(conv_cache.size(2), local_conv_dim)
      << "conv cache channel count does not match mixed_qkv.";
  CHECK_EQ(ssm_cache.size(1), local_value_heads)
      << "SSM cache head count does not match the model geometry.";
  CHECK_EQ(ssm_cache.size(2), head_k_dim_)
      << "SSM cache key dimension does not match the model geometry.";
  CHECK_EQ(ssm_cache.size(3), head_v_dim_)
      << "SSM cache value dimension does not match the model geometry.";
  CHECK_EQ(z.size(2), local_value_heads);
  CHECK_EQ(z.size(3), head_v_dim_);
  CHECK_EQ(b.size(2), local_value_heads);
  CHECK_EQ(a.size(2), local_value_heads);

  const int64_t num_slots = conv_cache.size(0);
  CHECK_GT(num_slots, 0) << "GDN cache must contain at least one slot.";
  CHECK_EQ(ssm_cache.size(0) % num_slots, 0)
      << "SSM cache rows must be divisible by conv cache slots.";
  const int64_t checkpoint_stride = ssm_cache.size(0) / num_slots;
  const std::vector<int32_t>& live_slots =
      input_params.embedding.linear_state_ids;
  if (use_spec_verify || !is_any_prefill) {
    check_live_slots(live_slots, batch_size, num_slots);
    CHECK_EQ(
        static_cast<int64_t>(input_params.linear_state_validity_mask.size()),
        batch_size)
        << "decode linear state validity must be sequence-scoped.";
    for (int64_t i = 0; i < batch_size; ++i) {
      const int64_t validity = input_params.linear_state_validity_mask[i];
      CHECK(validity == 1 ||
            (live_slots[i] == kPaddingLinearStateId && validity == 0))
          << "decode requires a warm GDN state for every real sequence.";
    }
  }

  const torch::Tensor contiguous_conv_weight = conv_weight.contiguous();
  const torch::Tensor contiguous_a_log = A_log_.contiguous();
  const torch::Tensor contiguous_dt_bias = dt_bias_.contiguous();
  const torch::Tensor contiguous_norm_weight = norm_->weight().contiguous();

  torch::Tensor fused_norm_output;
  // Route order is semantic: spec verify is also decode-shaped and must win.
  if (use_spec_verify) {
    CHECK_EQ(checkpoint_stride, sequence_length)
        << "MegaGdnMtpDecode requires one checkpoint per verify token.";
    CHECK_EQ(conv_cache.size(1), sequence_length + 2)
        << "MegaGdnMtpDecode requires an S+2 conv history.";
    CHECK_EQ(static_cast<int64_t>(attn_metadata.q_seq_lens_vec.size()),
             batch_size)
        << "MTP query lengths must be sequence-scoped.";
    CHECK_EQ(static_cast<int64_t>(input_params.num_accepted_tokens_host.size()),
             batch_size)
        << "MTP accepted-token values must be sequence-scoped host data.";
    CHECK_GE(sequence_length, 2)
        << "MegaGdnMtpDecode requires at least two verify tokens.";
    CHECK_LE(sequence_length, 17)
        << "MegaGdnMtpDecode supports at most 17 verify tokens.";
    std::vector<int32_t> accepted_tokens;
    accepted_tokens.reserve(batch_size);
    int64_t valid_verify_tokens = 0;
    for (int64_t i = 0; i < batch_size; ++i) {
      const int32_t query_length = attn_metadata.q_seq_lens_vec[i];
      CHECK_GT(query_length, 0)
          << "MTP query lengths must be positive, including padded graph rows.";
      CHECK_LE(query_length, sequence_length)
          << "MTP query length exceeds the dense operator width.";
      valid_verify_tokens += query_length;
      const int64_t accepted = input_params.num_accepted_tokens_host[i];
      CHECK_GE(accepted, 1) << "MTP accepted-token count must be in [1, S].";
      CHECK_LE(accepted, sequence_length)
          << "MTP accepted-token count must be in [1, S].";
      accepted_tokens.emplace_back(static_cast<int32_t>(accepted));
    }
    CHECK_EQ(valid_verify_tokens, original_num_tokens)
        << "MTP host query lengths do not match the unpadded model input.";
    const torch::Tensor state_indices =
        graph_safe_indices(input_params.embedding.linear_state_indices,
                           live_slots,
                           batch_size,
                           device,
                           "linear_state_indices");
    const torch::Tensor accepted_indices =
        graph_safe_indices(input_params.num_accepted_tokens,
                           accepted_tokens,
                           batch_size,
                           device,
                           "num_accepted_tokens");
    std::vector<torch::Tensor> chunk_outputs;
    chunk_outputs.reserve(
        util::ceil_div(batch_size, kMegaGdnMaxDecodeBatchSize));
    for (int64_t start = 0; start < batch_size;
         start += kMegaGdnMaxDecodeBatchSize) {
      const int64_t end =
          std::min(start + kMegaGdnMaxDecodeBatchSize, batch_size);
      xllm::kernel::MegaGdnMtpDecodeParams params;
      params.qkv = mixed_qkv.slice(0, start, end).contiguous();
      params.z = z.slice(0, start, end).contiguous();
      params.b = b.slice(0, start, end).contiguous();
      params.a = a.slice(0, start, end).contiguous();
      params.conv_weight = contiguous_conv_weight;
      params.conv_state = conv_cache;
      params.a_log = contiguous_a_log;
      params.dt_bias = contiguous_dt_bias;
      params.ssm_state = ssm_cache;
      params.read_state_indices =
          state_indices.slice(0, start, end).contiguous();
      params.write_state_indices = params.read_state_indices;
      params.num_accepted_tokens =
          accepted_indices.slice(0, start, end).contiguous();
      params.norm_weight = contiguous_norm_weight;
      params.fla_ssm_state_layout = use_fla_ssm_state_layout();
      torch::Tensor chunk_output;
      std::tie(std::ignore, std::ignore, std::ignore, chunk_output) =
          xllm::kernel::mega_gdn_mtp_decode(params);
      chunk_outputs.emplace_back(std::move(chunk_output));
    }
    fused_norm_output = chunk_outputs.size() == 1
                            ? std::move(chunk_outputs.front())
                            : torch::cat(chunk_outputs, 0);
  } else if (is_any_prefill) {
    CHECK_EQ(static_cast<int64_t>(attn_metadata.q_seq_lens_vec.size()),
             batch_size)
        << "prefill query lengths must be sequence-scoped.";
    CHECK_EQ(attn_metadata.q_cu_seq_lens.numel(), batch_size + 1)
        << "prefill cu_seqlens must contain B+1 entries.";
    CHECK_EQ(conv_cache.size(1), checkpoint_stride + 2)
        << "MegaGdnPrefill conv history must be R+2.";

    torch::Tensor packed_qkv = reshape_qkvz_unpad(
        attn_metadata,
        mixed_qkv.reshape({batch_size * sequence_length, local_conv_dim}));
    torch::Tensor packed_z =
        reshape_qkvz_unpad(attn_metadata,
                           z.reshape({batch_size * sequence_length,
                                      local_value_heads * head_v_dim_}));
    torch::Tensor packed_b = reshape_qkvz_unpad(
        attn_metadata,
        b.reshape({batch_size * sequence_length, local_value_heads}));
    torch::Tensor packed_a = reshape_qkvz_unpad(
        attn_metadata,
        a.reshape({batch_size * sequence_length, local_value_heads}));
    const int64_t total_tokens = packed_qkv.size(0);
    int64_t expected_tokens = 0;
    for (const int32_t query_length : attn_metadata.q_seq_lens_vec) {
      CHECK_GT(query_length, 0);
      CHECK_LE(query_length, sequence_length);
      expected_tokens += query_length;
    }
    CHECK_EQ(total_tokens, expected_tokens)
        << "packed prefill token count does not match host sequence lengths.";

    const MegaGdnPrefillIndicesCache& indices =
        qwen3_5_gdn_internal::get_or_build_prefill_indices(
            attn_metadata,
            live_slots,
            input_params.linear_state_validity_mask,
            checkpoint_stride,
            num_slots,
            device,
            input_params.linear_state_cache_ops);
    xllm::kernel::MegaGdnPrefillParams params;
    params.mixed_qkv = packed_qkv.contiguous();
    params.b = packed_b.view({total_tokens, local_value_heads}).contiguous();
    params.a = packed_a.view({total_tokens, local_value_heads}).contiguous();
    params.z = packed_z.view({total_tokens, local_value_heads, head_v_dim_})
                   .contiguous();
    params.conv_weight = contiguous_conv_weight;
    params.conv_state = conv_cache;
    params.a_log = contiguous_a_log;
    params.dt_bias = contiguous_dt_bias;
    params.conv_state_read_indices = indices.conv_read;
    params.conv_state_write_indices = indices.conv_write;
    params.ssm_state_read_indices = indices.ssm_read;
    params.ssm_state_write_indices = indices.ssm_write;
    params.ssm_cache = ssm_cache;
    params.cu_seqlens =
        attn_metadata.q_cu_seq_lens.to(torch::kInt).contiguous();
    params.norm_weight = contiguous_norm_weight;
    params.num_matrices =
        compute_num_matrices(attn_metadata.q_seq_lens_vec, local_value_heads);
    std::tie(fused_norm_output, std::ignore, std::ignore) =
        xllm::kernel::mega_gdn_prefill(params);
  } else {
    CHECK_EQ(sequence_length, 1)
        << "MegaGdnDecode requires one token per sequence.";
    CHECK_EQ(checkpoint_stride, 1)
        << "MegaGdnDecode requires one SSM row per slot.";
    CHECK_EQ(conv_cache.size(1), 3)
        << "MegaGdnDecode requires a three-token conv history.";
    const torch::Tensor state_indices =
        graph_safe_indices(input_params.embedding.linear_state_indices,
                           live_slots,
                           batch_size,
                           device,
                           "linear_state_indices");
    const torch::Tensor decode_qkv =
        mixed_qkv.reshape({batch_size, local_conv_dim});
    const torch::Tensor decode_z =
        z.reshape({batch_size, local_value_heads, head_v_dim_});
    const torch::Tensor decode_b = b.reshape({batch_size, local_value_heads});
    const torch::Tensor decode_a = a.reshape({batch_size, local_value_heads});
    std::vector<torch::Tensor> chunk_outputs;
    chunk_outputs.reserve(
        util::ceil_div(batch_size, kMegaGdnMaxDecodeBatchSize));
    for (int64_t start = 0; start < batch_size;
         start += kMegaGdnMaxDecodeBatchSize) {
      const int64_t end =
          std::min(start + kMegaGdnMaxDecodeBatchSize, batch_size);
      xllm::kernel::MegaGdnDecodeParams params;
      params.qkv = decode_qkv.slice(0, start, end).contiguous();
      params.z = decode_z.slice(0, start, end).contiguous();
      params.b = decode_b.slice(0, start, end).contiguous();
      params.a = decode_a.slice(0, start, end).contiguous();
      params.conv_weight = contiguous_conv_weight;
      params.conv_state = conv_cache;
      params.a_log = contiguous_a_log;
      params.dt_bias = contiguous_dt_bias;
      params.ssm_state = ssm_cache;
      params.read_state_indices =
          state_indices.slice(0, start, end).contiguous();
      params.write_state_indices = params.read_state_indices;
      params.norm_weight = contiguous_norm_weight;
      params.fla_ssm_state_layout = use_fla_ssm_state_layout();
      torch::Tensor chunk_output;
      std::tie(std::ignore, std::ignore, std::ignore, chunk_output) =
          xllm::kernel::mega_gdn_decode(params);
      chunk_outputs.emplace_back(std::move(chunk_output));
    }
    fused_norm_output = chunk_outputs.size() == 1
                            ? std::move(chunk_outputs.front())
                            : torch::cat(chunk_outputs, 0);
  }

  CHECK(fused_norm_output.defined()) << "MegaGdn operator returned no output.";
  CHECK(fused_norm_output.dim() == 3 || fused_norm_output.dim() == 4)
      << "MegaGdn norm output has an invalid rank.";
  torch::Tensor projected_input =
      fused_norm_output.reshape({-1, local_value_heads * head_v_dim_});
  // MegaGdnMtpDecode returns dense [B,S,...]. Spec verify is represented as
  // chunked prefill by the ACL executor, so remove padding per sequence rather
  // than truncating the flattened output (which would retain an earlier row's
  // padding and drop a later row's valid tokens for ragged metadata).
  if (use_spec_verify) {
    projected_input = reshape_qkvz_unpad(attn_metadata, projected_input);
  }
  CHECK_EQ(projected_input.size(0), original_num_tokens)
      << "MegaGdn output token layout does not match the model input.";
  if (fc1_context != nullptr && is_sequence_sharded(*fc1_context)) {
    return o_proj_->forward(projected_input,
                            row_parallel_reduce_mode_for_fc1(*fc1_context));
  }
  return o_proj_->forward(projected_input);
}

std::optional<
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>>
Qwen3_5GatedDeltaNetImpl::project_split_inputs(
    const torch::Tensor& hidden_states,
    const AttentionMetadata& attn_metadata) {
  auto qkv = reshape_projected_tokens_with_pad(
      attn_metadata, in_proj_qkv_->forward(hidden_states));
  auto z_proj = reshape_projected_tokens_with_pad(
      attn_metadata, in_proj_z_->forward(hidden_states));
  auto b_proj = reshape_projected_tokens_with_pad(
      attn_metadata, in_proj_b_->forward(hidden_states));
  auto a_proj = reshape_projected_tokens_with_pad(
      attn_metadata, in_proj_a_->forward(hidden_states));

  const int64_t batch_size = qkv.size(0);
  const int64_t seq_len = qkv.size(1);
  auto z =
      z_proj.view({batch_size, seq_len, num_v_heads_ / tp_size_, head_v_dim_});
  auto b = b_proj.view({batch_size, seq_len, num_v_heads_ / tp_size_});
  auto a = a_proj.view({batch_size, seq_len, num_v_heads_ / tp_size_});
  return std::make_tuple(qkv, z, b, a);
}

void Qwen3_5GatedDeltaNetImpl::load_projection_state_dict(
    const StateDict& state_dict) {
  auto in_proj_qkv_state_dict = state_dict.get_dict_with_prefix("in_proj_qkv.");
  if (in_proj_qkv_state_dict.size() > 0 && !in_proj_qkv_->is_weight_loaded()) {
    in_proj_qkv_->load_state_dict(
        in_proj_qkv_state_dict,
        /*shard_tensor_count=*/3,
        /*shard_sizes=*/
        {k_size_ / tp_size_, k_size_ / tp_size_, v_size_ / tp_size_});
  }

  auto in_proj_z_state_dict = state_dict.get_dict_with_prefix("in_proj_z.");
  if (in_proj_z_state_dict.size() > 0 && !in_proj_z_->is_weight_loaded()) {
    in_proj_z_->load_state_dict(in_proj_z_state_dict);
  }

  auto in_proj_b_state_dict = state_dict.get_dict_with_prefix("in_proj_b.");
  if (in_proj_b_state_dict.size() > 0 && !in_proj_b_->is_weight_loaded()) {
    in_proj_b_->load_state_dict(in_proj_b_state_dict);
  }

  auto in_proj_a_state_dict = state_dict.get_dict_with_prefix("in_proj_a.");
  if (in_proj_a_state_dict.size() > 0 && !in_proj_a_->is_weight_loaded()) {
    in_proj_a_->load_state_dict(in_proj_a_state_dict);
  }
}

void Qwen3_5GatedDeltaNetImpl::verify_projection_weights(
    const std::string& prefix) const {
  CHECK(in_proj_qkv_ && in_proj_qkv_->is_weight_loaded())
      << "Missing required weight after all shards loaded: " << prefix
      << "in_proj_qkv.weight";
  CHECK(in_proj_z_ && in_proj_z_->is_weight_loaded())
      << "Missing required weight after all shards loaded: " << prefix
      << "in_proj_z.weight";
  CHECK(in_proj_b_ && in_proj_b_->is_weight_loaded())
      << "Missing required weight after all shards loaded: " << prefix
      << "in_proj_b.weight";
  CHECK(in_proj_a_ && in_proj_a_->is_weight_loaded())
      << "Missing required weight after all shards loaded: " << prefix
      << "in_proj_a.weight";
}

}  // namespace layer
}  // namespace xllm

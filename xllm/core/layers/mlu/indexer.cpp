/* Copyright 2025 The xLLM Authors. All Rights Reserved.

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
#include "indexer.h"

#include <glog/logging.h>
#include <torch/torch.h>

#include <cmath>

#include "core/framework/config/kv_cache_config.h"
#include "kernels/ops_api.h"
#include "util/linalg.h"

namespace xllm {
namespace layer {

IndexerImpl::IndexerImpl(int64_t dim,
                         int64_t index_n_heads,
                         int64_t index_head_dim,
                         int64_t qk_rope_head_dim,
                         int64_t index_topk,
                         int64_t q_lora_rank,
                         bool enable_fused_qk,
                         const std::shared_ptr<RotaryEmbeddingBase>& rotary_emb,
                         const QuantArgs& quant_args,
                         const ParallelArgs& parallel_args,
                         const torch::TensorOptions& options)
    : n_heads_(index_n_heads),
      head_dim_(index_head_dim),
      rope_head_dim_(qk_rope_head_dim),
      index_topk_(index_topk),
      rotary_emb_(rotary_emb),
      softmax_scale_(std::pow(head_dim_, -0.5) * std::pow(n_heads_, -0.5)),
      enable_fused_qk_(enable_fused_qk) {
  // Note: The current Indexer implementation does not yet support quantization
  // or parallelization strategies. These features are planned for future
  // updates. For now, the entire indexer computation runs independently on each
  // MLU on any parallel strategy.
  (void)parallel_args;

  // Register modules
  wq_b_ = register_module("wq_b",
                          ReplicatedLinear(q_lora_rank,
                                           n_heads_ * head_dim_,
                                           /*bias=*/false,
                                           quant_args,
                                           options));
  wk_ = register_module("wk",
                        ReplicatedLinear(dim,
                                         head_dim_,
                                         /*bias=*/false,
                                         quant_args,
                                         options));

  weights_proj_ = register_module("weights_proj",
                                  ReplicatedLinear(dim,
                                                   n_heads_,
                                                   /*bias=*/false,
                                                   quant_args,
                                                   options));

  // the default eps is defined as 1e-6 in indexer implementation of
  // DeepSeek-V3.2.
  double default_eps = 1e-6;
  k_norm_ = register_module(
      "k_norm",
      RMSNorm(head_dim_, default_eps, options.dtype(torch::kFloat32)));
  k_norm_->set_layernorm_mode();

  // Create hadamard matrix
  int64_t head_dim_padded = std::pow(2, std::ceil(std::log2(head_dim_)));
  // Construct the Hadamard matrix on CPU with float32, then cast to target
  // dtype and device set normalize=true is equivalent to scale=hidden_size **
  // -0.5
  hadamard_matrix_ = util::create_hadamard_matrix(head_dim_padded,
                                                  torch::kFloat32,
                                                  torch::Device(torch::kCPU),
                                                  /*normalize=*/true);
  hadamard_matrix_ =
      hadamard_matrix_.to(options.device(), options.dtype().toScalarType());

  // indexer config
  // TODO: this part should be obtained via model config instead
  q_rope_at_front_ = true;
}

torch::Tensor IndexerImpl::rotate_activation(
    const torch::Tensor& input,
    const torch::Tensor& hadamard_matrix) {
  // Ensure the input is bfloat16 as per interface contract
  CHECK(input.dtype() == torch::kBFloat16)
      << "rotate_activation: input must be bfloat16";
  return util::rotate_activation(input, hadamard_matrix);
}

IndexerRuntimeContext IndexerImpl::prepare_runtime_context(
    const torch::Tensor& k_current_dense,
    torch::Tensor& k_cache_paged,
    torch::Tensor& q,
    torch::Tensor& weights,
    const AttentionMetadata& attn_metadata,
    bool is_prefill,
    int64_t num_tokens) {
  IndexerRuntimeContext ctx;
  auto device = attn_metadata.block_table.device();

  // Allocate context_lens buffer
  ctx.new_context_lens = torch::empty(
      {num_tokens}, torch::TensorOptions().dtype(torch::kInt32).device(device));

  if (is_prefill) {
    // Prefill: flatten Q and weights
    ctx.q = q;
    ctx.weights = weights;
    ctx.cu_seq_q_lens = attn_metadata.q_cu_seq_lens;
    ctx.k_block_table = std::nullopt;

    ctx.new_block_tables = torch::empty(
        {num_tokens, index_topk_},
        torch::TensorOptions().dtype(torch::kInt32).device(device));

    if (attn_metadata.is_chunked_prefill) {
      // NOTE: the kv_cu_seq_lens should already include the history tokens
      ctx.cu_seq_k_lens = attn_metadata.kv_cu_seq_lens;

      // Reuse the host-side total length cached in attention metadata to avoid
      // synchronizing the device cu_seq tensor back to CPU.
      int64_t total_k_len = attn_metadata.total_kv_len;
      ctx._storage_k_full = torch::empty(
          {total_k_len, head_dim_},
          torch::TensorOptions().dtype(k_cache_paged.dtype()).device(device));

      // Calculate sequence lengths by diff of offsets
      auto seq_lens = torch::diff(ctx.cu_seq_k_lens);
      int64_t max_context_len = attn_metadata.max_seq_len;
      ;

      ctx.k_context_lens = seq_lens;

      // Gather k from cache
      xllm::kernel::ReshapeFromCacheParams gather_params;
      gather_params.key = ctx._storage_k_full.unsqueeze(1);
      gather_params.value = std::nullopt;
      gather_params.key_cache = k_cache_paged;
      gather_params.value_cache = std::nullopt;
      gather_params.context_lengths = seq_lens;
      gather_params.max_context_len = max_context_len;
      gather_params.block_tables = attn_metadata.block_table;
      gather_params.context_seq_offset = std::nullopt;
      gather_params.cache_seq_offset = std::nullopt;

      xllm::kernel::reshape_from_cache(gather_params);
      ctx.k_cache_tensor = ctx._storage_k_full;
    } else {
      // Standard prefill: k is dense
      ctx.cu_seq_k_lens = attn_metadata.q_cu_seq_lens;
      ctx.k_context_lens = attn_metadata.kv_seq_lens;
      ctx.k_cache_tensor = k_current_dense;
    }
  } else {
    // Decode mode
    int64_t batch_size = attn_metadata.kv_seq_lens.size(0);
    auto seq_len = num_tokens / batch_size;

    // Reshape q and weights for decode
    ctx.q = q.view({batch_size, seq_len, n_heads_, head_dim_});
    ctx.weights = weights.view({batch_size, seq_len, n_heads_});

    ctx.new_block_tables = torch::empty(
        {batch_size, seq_len, index_topk_},
        torch::TensorOptions().dtype(torch::kInt32).device(device));

    ctx.cu_seq_q_lens = std::nullopt;
    ctx.cu_seq_k_lens = attn_metadata.q_cu_seq_lens;
    ctx.k_block_table = attn_metadata.block_table;
    ctx.k_cache_tensor = k_cache_paged;
    ctx.k_context_lens = attn_metadata.kv_seq_lens;
  }

  return ctx;
}

torch::Tensor IndexerImpl::preprocess_indexer_q(
    const torch::Tensor& q_norm,
    const torch::Tensor& positions,
    const AttentionMetadata& attn_metadata) {
  // Forward pass through wq_b
  auto q = wq_b_->forward(q_norm);
  q = q.view({q.size(0), n_heads_, head_dim_});
  auto q_pe = q.slice(-1, 0, rope_head_dim_);
  rotary_emb_->forward(q_pe,
                       positions,
                       attn_metadata.q_cu_seq_lens,
                       attn_metadata.max_query_len,
                       attn_metadata.is_prefill);

  // Apply rotation activation
  q = rotate_activation(q, hadamard_matrix_);
  return q;
}

std::tuple<torch::Tensor, torch::Tensor> IndexerImpl::preprocess_indexer_k(
    const torch::Tensor& x,
    const torch::Tensor& positions,
    torch::Tensor& k_cache,
    const AttentionMetadata& attn_metadata,
    bool write_k_cache) {
  // Forward pass through wk and normalize
  auto k = wk_->forward(x);
  auto k_dtype = k.dtype();
  // follow the implementation of DeepSeek-V3.2,
  // the k_norm is applied on the float32 tensor.
  auto k_fp32 = k.to(torch::kFloat32);
  k = std::get<0>(k_norm_->forward(k_fp32)).to(k_dtype);

  // Apply rotary embedding to positional parts only (like Python)
  auto k_pe = k.slice(-1, 0, rope_head_dim_).unsqueeze(1);
  rotary_emb_->forward(k_pe,
                       positions,
                       attn_metadata.q_cu_seq_lens,
                       attn_metadata.max_query_len,
                       attn_metadata.is_prefill);
  k = rotate_activation(k, hadamard_matrix_);

  if (write_k_cache) {
    write_prefill_k_cache(k, k_cache, attn_metadata.slot_mapping);
  }

  // Forward pass through weights projection
  auto weights = weights_proj_->forward(x);

  return {k, weights};
}

torch::Tensor IndexerImpl::preprocess_indexer_q_fused(
    const torch::Tensor& q_norm,
    const torch::Tensor& positions) {
  // fuses the query projection(Matmul), Rotary Position Embedding (RoPE), and
  // an optional Hadamard transformation(Matmul) into a single high-performance
  // kernel
  auto output =
      torch::empty({q_norm.size(0), n_heads_, head_dim_}, q_norm.options());
  auto w_q = wq_b_->weight().view({n_heads_, head_dim_, -1});
  kernel::FusedIndexerQParams q_params;
  q_params.input_q = q_norm;
  q_params.output = output;
  q_params.output_scale = std::nullopt;
  q_params.w_q = w_q;
  q_params.w_q_scale = std::nullopt;
  q_params.hadamard_matrix = hadamard_matrix_;
  q_params.sin = rotary_emb_->get_sin_cache();
  q_params.cos = rotary_emb_->get_cos_cache();
  q_params.position_id = positions;
  q_params.quant_mode = "none";
  q_params.interleaved = rotary_emb_->get_interleaved();
  q_params.rope_at_front = q_rope_at_front_;
  kernel::fused_indexer_q(q_params);
  return output;
}

torch::Tensor IndexerImpl::preprocess_indexer_k_fused(
    const torch::Tensor& x,
    const torch::Tensor& positions,
    torch::Tensor& k_cache,
    const AttentionMetadata& attn_metadata) {
  // Perform wk(x), layernorm, rope, wproj(x) and quant to paged k_cache
  auto wproj_weight = weights_proj_->weight();
  auto head_weights =
      torch::empty({x.size(0), wproj_weight.size(0)}, x.options());
  kernel::FusedIndexerKParams k_params;
  k_params.x = x;
  k_params.wk = wk_->weight();
  k_params.wproj = wproj_weight;
  k_params.sin_table = rotary_emb_->get_sin_cache();
  k_params.cos_table = rotary_emb_->get_cos_cache();
  k_params.position_id = positions;
  k_params.slot_mapping = attn_metadata.slot_mapping;
  k_params.head_weights = head_weights;
  k_params.k_cache = k_cache;
  k_params.k_cache_scale = std::nullopt;
  k_params.hadamard_matrix = hadamard_matrix_;
  k_params.interleaved = rotary_emb_->get_interleaved();
  k_params.gamma = k_norm_->weight();
  k_params.beta = k_norm_->bias();
  k_params.eps = k_norm_->eps();
  kernel::fused_indexer_k(k_params);
  return head_weights;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
IndexerImpl::preprocess_indexer_inputs(const torch::Tensor& x,
                                       const torch::Tensor& q_norm,
                                       const torch::Tensor& positions,
                                       torch::Tensor& k_cache,
                                       const AttentionMetadata& attn_metadata,
                                       bool is_prefill,
                                       bool write_k_cache) {
  torch::Tensor q, k, weights;
  if (!is_prefill && enable_fused_qk_) {
    q = preprocess_indexer_q_fused(q_norm, positions);
    weights = preprocess_indexer_k_fused(x, positions, k_cache, attn_metadata);
  } else {
    q = preprocess_indexer_q(q_norm, positions, attn_metadata);
    std::tie(k, weights) = preprocess_indexer_k(
        x, positions, k_cache, attn_metadata, write_k_cache);
  }
  return {q, k, weights};
}

IndexerSPPreOut IndexerImpl::sp_pre(
    const torch::Tensor& x,
    const torch::Tensor& q_norm,
    const torch::Tensor& positions,
    const AttentionMetadata& attn_metadata,
    const v32_sp::DeepseekV32SPContext& sp_ctx) {
  (void)sp_ctx;
  IndexerSPPreOut out;
  std::tie(out.q, out.k_local, out.weights) =
      preprocess_indexer_inputs(x,
                                q_norm,
                                positions,
                                out.k_local,
                                attn_metadata,
                                /*is_prefill=*/true,
                                /*write_k_cache=*/false);
  return out;
}

v32_sp::PaddedGatherHandle IndexerImpl::sp_comm(
    const torch::Tensor& k_local,
    const v32_sp::DeepseekV32SPContext& sp_ctx) {
  if (!k_local.defined()) {
    return {};
  }
  return parallel_state::launch_gather(
      k_local, sp_ctx.process_group, sp_ctx.comm_plan.tokens_per_rank);
}

torch::Tensor IndexerImpl::sp_wait_k(
    const torch::Tensor& k_local,
    const v32_sp::PaddedGatherHandle& gather_handle,
    const v32_sp::DeepseekV32SPContext& sp_ctx) {
  if (gather_handle.stacked.defined()) {
    (void)sp_ctx;
    return parallel_state::finish_gather(gather_handle);
  }
  return k_local;
}

std::tuple<torch::Tensor, torch::Tensor> IndexerImpl::sp_post(
    const IndexerSPPreOut& pre_out,
    const torch::Tensor& k_gathered,
    torch::Tensor& k_cache,
    const AttentionMetadata& attn_metadata,
    const torch::Tensor& gathered_slot_mapping,
    const v32_sp::DeepseekV32SPContext& sp_ctx) {
  CHECK(attn_metadata.is_prefill || attn_metadata.is_chunked_prefill)
      << "deepseek_v32 sequence parallel indexer only supports prefill "
         "batches.";
  CHECK(sp_ctx.batch_forward_type.no_decode())
      << "deepseek_v32 sequence parallel indexer only supports prefill "
         "batches.";
  CHECK(attn_metadata.block_table.defined())
      << "deepseek_v32 sequence parallel indexer requires block_table.";
  CHECK(attn_metadata.slot_mapping.defined())
      << "deepseek_v32 sequence parallel indexer requires slot_mapping.";
  CHECK(gathered_slot_mapping.defined())
      << "deepseek_v32 sequence parallel indexer requires gathered "
         "slot_mapping.";
  for (const auto& segment : sp_ctx.local_segments) {
    CHECK_GE(segment.req_idx, 0)
        << "deepseek_v32 sequence parallel expects non-negative req_idx.";
    CHECK_LT(segment.req_idx, attn_metadata.block_table.size(0))
        << "deepseek_v32 sequence parallel segment req_idx is out of range.";
  }

  // For chunked SP, keep the runtime contract aligned with the normal chunked
  // indexer path: write the freshly computed suffix K into paged cache first,
  // then rebuild the full-context dense K from cache before segmented select.
  // Feeding suffix-only K here would truncate the effective context seen by
  // the indexer on long prompts.
  write_prefill_k_cache(k_gathered, k_cache, gathered_slot_mapping);
  // `k_gathered` stays in rank-packed gather order for paged-cache placement.
  // The non-chunked segmented select path, however, slices dense K with
  // request-global offsets and therefore must consume original token order.
  torch::Tensor k_source =
      attn_metadata.is_chunked_prefill
          ? build_sp_chunked_dense_k_source(k_cache, attn_metadata)
          : v32_sp::restore_gathered_to_global_order(k_gathered, sp_ctx);
  return run_indexer_select_kernel_sp_segmented(
      pre_out, k_source, attn_metadata, sp_ctx);
}

void IndexerImpl::write_prefill_k_cache(const torch::Tensor& k,
                                        torch::Tensor& k_cache,
                                        const torch::Tensor& slot_mapping) {
  auto k_unsqueezed = k.unsqueeze(1);
  xllm::kernel::ReshapePagedCacheParams reshape_paged_cache_params;
  reshape_paged_cache_params.key = k_unsqueezed;
  reshape_paged_cache_params.value = std::nullopt;
  reshape_paged_cache_params.k_cache = k_cache;
  reshape_paged_cache_params.v_cache = std::nullopt;
  reshape_paged_cache_params.slot_mapping = slot_mapping;
  reshape_paged_cache_params.direction = false;
  xllm::kernel::reshape_paged_cache(reshape_paged_cache_params);
}

torch::Tensor IndexerImpl::build_sp_chunked_dense_k_source(
    torch::Tensor& k_cache,
    const AttentionMetadata& attn_metadata) {
  auto device = attn_metadata.block_table.device();
  int64_t total_k_len = attn_metadata.total_kv_len;
  torch::Tensor k_full = torch::empty(
      {total_k_len, head_dim_},
      torch::TensorOptions().dtype(k_cache.dtype()).device(device));

  auto seq_lens = torch::diff(attn_metadata.kv_cu_seq_lens);
  int64_t max_context_len = attn_metadata.max_seq_len;

  xllm::kernel::ReshapeFromCacheParams gather_params;
  gather_params.key = k_full.unsqueeze(1);
  gather_params.value = std::nullopt;
  gather_params.key_cache = k_cache;
  gather_params.value_cache = std::nullopt;
  gather_params.context_lengths = seq_lens;
  gather_params.max_context_len = max_context_len;
  gather_params.block_tables = attn_metadata.block_table;
  gather_params.context_seq_offset = std::nullopt;
  gather_params.cache_seq_offset = std::nullopt;
  xllm::kernel::reshape_from_cache(gather_params);
  return k_full;
}

std::tuple<torch::Tensor, torch::Tensor> IndexerImpl::run_indexer_select_kernel(
    const AttentionMetadata& attn_metadata,
    bool is_prefill,
    IndexerRuntimeContext& ctx) {
  // Call masked indexer select paged kv
  kernel::MaskedIndexerSelectPagedKVParams params;
  params.query = ctx.q;
  params.k_cache = ctx.k_cache_tensor;
  params.weights = ctx.weights;
  params.kv_cache_block_table = attn_metadata.block_table;
  params.cu_seq_q_lens = ctx.cu_seq_q_lens;
  params.cu_seq_k_lens = ctx.cu_seq_k_lens;
  params.k_context_lens = ctx.k_context_lens;
  params.k_cache_block_table = ctx.k_block_table;
  params.is_prefill = is_prefill;
  params.softmax_scale = softmax_scale_;
  params.q_scale = std::nullopt;        // empty tensor as q_scale
  params.k_scale_cache = std::nullopt;  // empty tensor as k_scale_cache
  params.index_topk = index_topk_;
  params.kv_cache_block_size =
      ::xllm::KVCacheConfig::get_instance().block_size();
  params.sparse_block_table = ctx.new_block_tables;
  params.sparse_context_lens = ctx.new_context_lens;

  xllm::kernel::masked_indexer_select_paged_kv(params);

  if (!is_prefill) {
    ctx.new_block_tables =
        ctx.new_block_tables.view({-1, ctx.new_block_tables.size(-1)});
  }

  return {ctx.new_block_tables, ctx.new_context_lens};
}

std::tuple<torch::Tensor, torch::Tensor>
IndexerImpl::run_indexer_select_kernel_sp_segmented(
    const IndexerSPPreOut& pre_out,
    const torch::Tensor& k_source,
    const AttentionMetadata& attn_metadata,
    const v32_sp::DeepseekV32SPContext& sp_ctx) {
  auto device = attn_metadata.block_table.device();
  auto int32_options =
      torch::TensorOptions().dtype(torch::kInt32).device(device);

  torch::Tensor new_block_tables =
      torch::empty({pre_out.q.size(0), index_topk_}, int32_options);
  torch::Tensor new_context_lens =
      torch::empty({pre_out.q.size(0)}, int32_options);

  for (int64_t i = 0; i < static_cast<int64_t>(sp_ctx.local_segments.size());
       ++i) {
    const auto& segment = sp_ctx.local_segments[i];
    if (segment.q_tokens == 0) {
      continue;
    }

    const int32_t q_start = sp_ctx.seg_q_starts_cpu[i];
    const int32_t req_q_start = sp_ctx.req_q_offsets_cpu[segment.req_idx];
    const int32_t req_ctx_start = sp_ctx.req_ctx_offsets_cpu[segment.req_idx];

    torch::Tensor q_seg = pre_out.q.narrow(0, q_start, segment.q_tokens);
    torch::Tensor weights_seg =
        pre_out.weights.narrow(0, q_start, segment.q_tokens);

    torch::Tensor k_seg;
    torch::Tensor cu_seq_k_lens_seg;
    if (attn_metadata.is_chunked_prefill) {
      k_seg = k_source.narrow(0, req_ctx_start, segment.ctx_k_len);
      cu_seq_k_lens_seg = sp_ctx.seg_ctx_k_cu_lens_2col.select(0, i);
    } else {
      k_seg = k_source.narrow(0, req_q_start, segment.suffix_k_len);
      cu_seq_k_lens_seg = sp_ctx.seg_suffix_k_cu_lens_2col.select(0, i);
    }

    torch::Tensor cu_seq_q_lens_seg = sp_ctx.seg_q_cu_lens_2col.select(0, i);
    torch::Tensor k_context_lens_seg = sp_ctx.seg_ctx_lens_1col.narrow(0, i, 1);
    torch::Tensor block_table_seg =
        attn_metadata.block_table.narrow(0, segment.req_idx, 1);
    torch::Tensor out_block_seg =
        new_block_tables.narrow(0, q_start, segment.q_tokens);
    torch::Tensor out_ctx_seg =
        new_context_lens.narrow(0, q_start, segment.q_tokens);

    kernel::MaskedIndexerSelectPagedKVParams params;
    params.query = q_seg;
    params.k_cache = k_seg;
    params.weights = weights_seg;
    params.kv_cache_block_table = block_table_seg;
    params.cu_seq_q_lens = cu_seq_q_lens_seg;
    params.cu_seq_k_lens = cu_seq_k_lens_seg;
    params.k_context_lens = k_context_lens_seg;
    params.k_cache_block_table = std::nullopt;
    params.is_prefill = true;
    params.softmax_scale = softmax_scale_;
    params.q_scale = std::nullopt;
    params.k_scale_cache = std::nullopt;
    params.index_topk = index_topk_;
    params.kv_cache_block_size =
        ::xllm::KVCacheConfig::get_instance().block_size();
    params.sparse_block_table = out_block_seg;
    params.sparse_context_lens = out_ctx_seg;
    xllm::kernel::masked_indexer_select_paged_kv(params);
  }

  return {new_block_tables, new_context_lens};
}

std::tuple<torch::Tensor, torch::Tensor> IndexerImpl::forward(
    const torch::Tensor& x,
    const torch::Tensor& q_norm,
    const torch::Tensor& positions,
    torch::Tensor& k_cache,
    const AttentionMetadata& attn_metadata,
    bool is_prefill,
    const std::optional<torch::Tensor>& mask) {
  (void)mask;
  torch::Tensor q, k, weights;
  std::tie(q, k, weights) = preprocess_indexer_inputs(x,
                                                      q_norm,
                                                      positions,
                                                      k_cache,
                                                      attn_metadata,
                                                      is_prefill,
                                                      /*write_k_cache=*/true);
  // Unified parameter setup for both prefill and decode modes
  IndexerRuntimeContext ctx = prepare_runtime_context(
      k, k_cache, q, weights, attn_metadata, is_prefill, x.size(0));

  return run_indexer_select_kernel(attn_metadata, is_prefill, ctx);
}

// load the weight from the checkpoint
void IndexerImpl::load_state_dict(const StateDict& state_dict) {
  if (state_dict.size() == 0) {
    return;
  }
  // Load weights for each linear layer
  wq_b_->load_state_dict(state_dict.get_dict_with_prefix("wq_b."));
  wk_->load_state_dict(state_dict.get_dict_with_prefix("wk."));
  weights_proj_->load_state_dict(
      state_dict.get_dict_with_prefix("weights_proj."));
  k_norm_->load_state_dict(state_dict.get_dict_with_prefix("k_norm."));
}

}  // namespace layer
}  // namespace xllm

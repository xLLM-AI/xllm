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

#include "layers/mlu/deepseek_v4/deepseek_v4_attention.h"

#include <glog/logging.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <tuple>
#include <vector>

#include "kernels/mlu/mlu_ops_api.h"
#include "kernels/ops_api.h"

namespace xllm {
namespace layer {
namespace {

bool has_tensor(const torch::Tensor& tensor) {
  return tensor.defined() && tensor.numel() > 0;
}

int64_t effective_compress_ratio(int64_t ratio) {
  return ratio <= 1 ? 1 : ratio;
}

torch::Tensor flatten_slot_cache(const torch::Tensor& cache) {
  if (!has_tensor(cache) || cache.dim() != 4 || cache.size(2) == 1) {
    return cache;
  }
  return cache.reshape({-1, cache.size(1), 1, cache.size(3)}).contiguous();
}

torch::Tensor layer_tensor(
    const std::vector<std::vector<torch::Tensor>>& tensors,
    int32_t layer_id,
    int64_t cache_idx) {
  if (layer_id < 0 || layer_id >= static_cast<int32_t>(tensors.size()) ||
      cache_idx < 0 ||
      cache_idx >= static_cast<int64_t>(tensors[layer_id].size())) {
    return torch::Tensor();
  }
  return tensors[layer_id][cache_idx];
}

void write_cache(const torch::Tensor& rows,
                 torch::Tensor& cache,
                 const torch::Tensor& slot_mapping) {
  if (!has_tensor(rows) || !has_tensor(slot_mapping)) {
    return;
  }

  xllm::kernel::ReshapePagedCacheParams params;
  params.key = rows.contiguous();
  params.value = std::nullopt;
  params.k_cache = cache;
  params.v_cache = std::nullopt;
  params.slot_mapping = slot_mapping.contiguous();
  params.direction = false;
  xllm::kernel::reshape_paged_cache(params);
}

AttentionMetadata make_indexer_metadata(const AttentionMetadata& attn_metadata,
                                        const torch::Tensor& block_table,
                                        const torch::Tensor& slot_mapping) {
  const DSAMetadata& dsa = *attn_metadata.dsa_metadata;
  AttentionMetadata metadata = attn_metadata;
  metadata.is_causal = true;
  metadata.block_table = block_table;
  metadata.slot_mapping = slot_mapping;
  metadata.max_seq_len = dsa.index_max_c4_len;
  metadata.total_kv_len = dsa.index_total_c4_len;
  metadata.compute_dtype = "float";
  return metadata;
}

void run_decode(torch::Tensor& q,
                torch::Tensor& output,
                std::optional<torch::Tensor>& output_lse,
                const torch::Tensor& cache,
                const torch::Tensor& block_table,
                const torch::Tensor& context_lens,
                int64_t max_context_len,
                int64_t window_left,
                float scale,
                const std::optional<torch::Tensor>& sink,
                bool return_lse) {
  output_lse = std::nullopt;
  if (return_lse) {
    output_lse = torch::empty({q.size(0), q.size(2), 1},
                              q.options().dtype(torch::kFloat32));
  }
  xllm::kernel::mlu::batch_decode(q,
                                  cache,
                                  output,
                                  block_table,
                                  context_lens,
                                  std::optional<torch::Tensor>(cache),
                                  output_lse,
                                  std::nullopt,
                                  std::nullopt,
                                  std::nullopt,
                                  std::nullopt,
                                  std::nullopt,
                                  std::nullopt,
                                  "float",
                                  max_context_len,
                                  window_left,
                                  /*window_size_right=*/-1,
                                  scale,
                                  return_lse,
                                  /*kv_cache_quant_bit_size=*/-1,
                                  std::nullopt,
                                  /*max_seq_q=*/-1,
                                  sink);
}

}  // namespace

DeepseekV4AttentionImpl::DeepseekV4AttentionImpl(
    const ModelArgs& args,
    const QuantArgs& quant_args,
    const ParallelArgs& parallel_args,
    const torch::TensorOptions& options,
    int32_t layer_id)
    : layer_id_(layer_id),
      hidden_dim_(args.hidden_size()),
      q_lora_rank_(args.q_lora_rank()),
      n_heads_(args.n_heads()),
      head_dim_(args.head_dim()),
      rope_head_dim_(args.rope_head_dim()),
      nope_head_dim_(args.head_dim() - args.rope_head_dim()),
      o_groups_(args.o_groups()),
      o_lora_rank_(args.o_lora_rank()),
      window_size_(args.window_size()),
      index_n_heads_(args.index_n_heads()),
      index_head_dim_(args.index_head_dim()),
      index_topk_(args.index_topk()),
      eps_(args.rms_norm_eps()),
      scale_(static_cast<float>(
          std::pow(static_cast<double>(args.head_dim()), -0.5))) {
  compress_ratio_ = effective_compress_ratio(
      args.compress_ratios()[static_cast<size_t>(layer_id_)]);

  tp_rank_ = parallel_args.tp_group_->rank();
  tp_size_ = parallel_args.tp_group_->world_size();
  n_local_heads_ = n_heads_ / tp_size_;
  n_local_groups_ = o_groups_ / tp_size_;

  // Resolve indexer fallbacks early so hidden_proj_sizes_ can use them.
  if (compress_ratio_ == 4) {
    if (index_n_heads_ <= 0) {
      index_n_heads_ = n_heads_;
    }
    if (index_head_dim_ <= 0) {
      index_head_dim_ = head_dim_;
    }
    if (index_topk_ <= 0) {
      index_topk_ = 512;
    }
  }

  // DeepSeek V4 MLU checkpoints keep wo_a and DSA helper weights dense.
  const QuantArgs empty_quant_args;

  // Build the per-segment output dims for the merged hidden->* GEMM. The order
  // must match both the checkpoint cat in load_state_dict and the split() in
  // forward. See deepseek_v4.py hidden_proj_sizes for the reference layout.
  hidden_proj_sizes_.clear();
  hidden_proj_sizes_.push_back(q_lora_rank_);  // segment 0: wq_a
  hidden_proj_sizes_.push_back(head_dim_);     // segment 1: wkv
  if (compress_ratio_ == 4 || compress_ratio_ == 128) {
    const int64_t coff = (compress_ratio_ == 4) ? 2 : 1;
    hidden_proj_sizes_.push_back(coff *
                                 head_dim_);  // segment 2: compressor.wkv
    hidden_proj_sizes_.push_back(coff *
                                 head_dim_);  // segment 3: compressor.wgate
    if (compress_ratio_ == 4) {
      hidden_proj_sizes_.push_back(
          index_n_heads_);  // seg 4: indexer.weights_proj
      hidden_proj_sizes_.push_back(
          2 * index_head_dim_);  // seg 5: indexer.compressor.wkv (coff=2)
      hidden_proj_sizes_.push_back(
          2 * index_head_dim_);  // seg 6: indexer.compressor.wgate (coff=2)
    }
  }
  int64_t fused_out = 0;
  for (auto s : hidden_proj_sizes_) {
    fused_out += s;
  }
  fused_wqa_wkv_ = register_module("fused_wqa_wkv",
                                   ReplicatedLinear(hidden_dim_,
                                                    fused_out,
                                                    /*bias=*/false,
                                                    empty_quant_args,
                                                    options));

  q_norm_ = register_module("q_norm", RMSNorm(q_lora_rank_, eps_, options));
  wq_b_ = register_module("wq_b",
                          ColumnParallelLinear(q_lora_rank_,
                                               n_heads_ * head_dim_,
                                               /*bias=*/false,
                                               /*gather_output=*/false,
                                               quant_args,
                                               parallel_args.tp_group_,
                                               options));
  kv_norm_ = register_module("kv_norm", RMSNorm(head_dim_, eps_, options));
  wo_a_ = register_module("wo_a",
                          ColumnParallelLinear(n_heads_ * head_dim_ / o_groups_,
                                               o_groups_ * o_lora_rank_,
                                               /*bias=*/false,
                                               /*gather_output=*/true,
                                               empty_quant_args,
                                               parallel_args.tp_group_,
                                               options));
  wo_b_ = register_module("wo_b",
                          RowParallelLinear(o_groups_ * o_lora_rank_,
                                            hidden_dim_,
                                            /*bias=*/false,
                                            /*input_is_parallelized=*/true,
                                            /*reduce=*/true,
                                            quant_args,
                                            parallel_args.tp_group_,
                                            options));
  attn_sink_ = register_parameter(
      "attn_sink",
      torch::zeros({n_local_heads_}, options.dtype(torch::kFloat32)),
      /*requires_grad=*/false);
  if (compress_ratio_ == 4 || compress_ratio_ == 128) {
    compressor_ = register_module("compressor",
                                  Compressor(compress_ratio_,
                                             hidden_dim_,
                                             head_dim_,
                                             rope_head_dim_,
                                             /*rotate=*/false,
                                             eps_,
                                             options,
                                             args,
                                             empty_quant_args));
  }
  if (compress_ratio_ == 4) {
    indexer_ = register_module("indexer",
                               DeepseekV4Indexer(hidden_dim_,
                                                 index_n_heads_,
                                                 index_head_dim_,
                                                 rope_head_dim_,
                                                 index_topk_,
                                                 q_lora_rank_,
                                                 eps_,
                                                 options,
                                                 args,
                                                 empty_quant_args));
  }
}

void DeepseekV4AttentionImpl::apply_last_rope(
    torch::Tensor& tensor,
    const torch::Tensor& sin_table,
    const torch::Tensor& cos_table,
    const torch::Tensor& input_positions,
    int64_t rope_dim) {
  if (rope_dim == 0 || tensor.numel() == 0) {
    return;
  }

  torch::Tensor rope_part =
      tensor.slice(/*dim=*/-1, /*start=*/tensor.size(-1) - rope_dim);
  xllm::kernel::RotaryParams params;
  params.q = rope_part;
  params.k = torch::Tensor();
  params.sin = sin_table;
  params.cos = cos_table;
  params.position_ids = input_positions;
  params.cu_query_lens = std::nullopt;
  params.interleaved = true;
  params.discrete = true;
  params.dynamic_ntk = false;
  params.max_query_len = tensor.size(0);
  xllm::kernel::apply_rotary(params);
}

torch::Tensor DeepseekV4AttentionImpl::project_q(torch::Tensor& q_down,
                                                 torch::Tensor& qr) {
  // q_down is the segment-0 slice of the fused hidden->* projection.
  qr = std::get<0>(q_norm_->forward(q_down));
  torch::Tensor q = wq_b_->forward(qr).view({-1, n_local_heads_, head_dim_});
  torch::Tensor output = torch::empty_like(q);
  const std::string mode = "rmsnorm";
  xllm::kernel::mlu::fused_layernorm(q,
                                     output,
                                     std::nullopt,
                                     std::nullopt,
                                     std::nullopt,
                                     std::nullopt,
                                     std::nullopt,
                                     std::nullopt,
                                     std::nullopt,
                                     std::nullopt,
                                     mode,
                                     eps_,
                                     /*store_output_before_norm=*/false,
                                     /*store_output_after_norm=*/false,
                                     /*dynamic_quant=*/false);
  return output;
}

torch::Tensor DeepseekV4AttentionImpl::project_kv(torch::Tensor& kv_down) {
  // kv_down is the segment-1 slice of the fused hidden->* projection.
  torch::Tensor kv = std::get<0>(kv_norm_->forward(kv_down));
  return kv.view({-1, 1, head_dim_});
}

torch::Tensor DeepseekV4AttentionImpl::project_output(
    torch::Tensor& attn_output) {
  const int64_t num_tokens = attn_output.size(0);
  torch::Tensor grouped =
      attn_output.reshape({num_tokens, n_local_groups_, -1});
  torch::Tensor wo_a =
      wo_a_->weight().view({n_local_groups_, o_lora_rank_, -1});
  torch::Tensor low_rank =
      torch::einsum("tgd,grd->tgr", {grouped, wo_a}).reshape({num_tokens, -1});
  return wo_b_->forward(low_rank);
}

std::tuple<torch::Tensor, std::optional<torch::Tensor>>
DeepseekV4AttentionImpl::forward(const AttentionMetadata& attn_metadata,
                                 torch::Tensor& hidden_states,
                                 KVCache& kv_cache) {
  CHECK(attn_metadata.dsa_metadata)
      << "DeepseekV4Attention requires DSAMetadata.";
  CHECK(attn_metadata.q_seq_lens.defined())
      << "DeepseekV4Attention requires AttentionMetadata.q_seq_lens.";
  CHECK(attn_metadata.kv_seq_lens.defined())
      << "DeepseekV4Attention requires AttentionMetadata.kv_seq_lens.";
  const DSAMetadata& dsa = *attn_metadata.dsa_metadata;
  const bool is_prefill = attn_metadata.is_prefill;
  const bool is_chunked_prefill = attn_metadata.is_chunked_prefill;
  torch::Tensor output = torch::empty({hidden_states.size(0), hidden_dim_},
                                      hidden_states.options());
  if (attn_metadata.is_dummy) {
    return {output, std::nullopt};
  }

  // One fused GEMM over all hidden->* projections, then split along the output
  // dim into per-segment views. split() returns views sharing storage; the
  // downstream ops (rmsnorm/rotary/fused_compress) do not mutate their inputs
  // in place, so the views can be consumed zero-copy. If any downstream kernel
  // is later changed to be in-place, the relevant slice must be cloned first.
  torch::Tensor hidden_proj = fused_wqa_wkv_->forward(hidden_states);
  std::vector<torch::Tensor> parts =
      hidden_proj.split(/*split_size=*/hidden_proj_sizes_, /*dim=*/-1);

  torch::Tensor qr;
  torch::Tensor q = project_q(parts[0], qr);
  torch::Tensor kv = project_kv(parts[1]);

  const bool uses_compressed_rope =
      compress_ratio_ == 4 || compress_ratio_ == 128;
  const torch::Tensor& active_sin_table =
      uses_compressed_rope ? dsa.compressed_sin_table : dsa.sin_table;
  const torch::Tensor& active_cos_table =
      uses_compressed_rope ? dsa.compressed_cos_table : dsa.cos_table;
  const torch::Tensor& active_inverse_sin_table =
      uses_compressed_rope ? dsa.compressed_inverse_sin_table
                           : dsa.inverse_sin_table;
  apply_last_rope(q,
                  active_sin_table,
                  active_cos_table,
                  dsa.input_positions,
                  rope_head_dim_);
  apply_last_rope(kv,
                  active_sin_table,
                  active_cos_table,
                  dsa.input_positions,
                  rope_head_dim_);

  const DSACacheMapping& mapping = cache_mapping_;
  torch::Tensor ori_block_table =
      layer_tensor(dsa.block_tables, layer_id_, mapping.ori_cache_idx);
  torch::Tensor ori_slot =
      layer_tensor(dsa.slot_mappings, layer_id_, mapping.ori_cache_idx);

  torch::Tensor swa_cache = kv_cache.get_swa_cache();
  torch::Tensor ori_cache_for_attn;
  torch::Tensor ori_table_rows;
  torch::Tensor ori_context_lens;
  int64_t ori_max_context_len = 0;
  if (is_prefill || is_chunked_prefill) {
    write_cache(kv, swa_cache, ori_slot);
    ori_cache_for_attn = swa_cache;
    ori_context_lens = dsa.input_positions + 1;
    ori_max_context_len = attn_metadata.max_seq_len;
    ori_table_rows =
        torch::repeat_interleave(ori_block_table, attn_metadata.q_seq_lens, 0);
  } else {
    write_cache(kv, swa_cache, ori_slot);
    ori_cache_for_attn = swa_cache;
    ori_context_lens = dsa.input_positions + 1;
    ori_max_context_len = attn_metadata.max_seq_len;
    ori_table_rows = ori_block_table;
  }

  torch::Tensor q_decode = q.unsqueeze(1).contiguous();
  torch::Tensor ori_out = torch::empty_like(q_decode);
  std::optional<torch::Tensor> ori_lse;
  const bool has_cmp_context =
      (compress_ratio_ == 4 && dsa.index_total_c4_len > 0) ||
      (compress_ratio_ == 128 && dsa.c128_attn_metadata.max_context_len > 0);
  run_decode(q_decode,
             ori_out,
             ori_lse,
             ori_cache_for_attn,
             ori_table_rows,
             ori_context_lens,
             ori_max_context_len,
             std::max<int64_t>(window_size_ - 1, 0),
             scale_,
             attn_sink_loaded_ ? std::optional<torch::Tensor>(attn_sink_)
                               : std::nullopt,
             has_cmp_context);

  torch::Tensor merged_out = ori_out;
  torch::Tensor merged_lse;
  if (has_cmp_context) {
    merged_lse = ori_lse.value();
  }

  if (compress_ratio_ == 4 || compress_ratio_ == 128) {
    torch::Tensor cmp_cache = kv_cache.get_k_cache();
    torch::Tensor cmp_slot =
        layer_tensor(dsa.slot_mappings, layer_id_, mapping.cmp_cache_idx);
    torch::Tensor cmp_state = kv_cache.get_compress_state();
    torch::Tensor cmp_state_block_table =
        layer_tensor(dsa.block_tables, layer_id_, mapping.kv_state_cache_idx);
    compressor_->forward(attn_metadata,
                         hidden_states,
                         cmp_cache,
                         cmp_slot,
                         cmp_state,
                         cmp_state_block_table,
                         dsa.compressed_sin_table,
                         dsa.compressed_cos_table,
                         /*projected_kv=*/parts[2],
                         /*projected_score=*/parts[3]);
    torch::Tensor cmp_context_lens;
    torch::Tensor cmp_table_for_attn;
    torch::Tensor cmp_cache_for_attn = cmp_cache;
    int64_t cmp_max_context = 0;
    if (compress_ratio_ == 4) {
      torch::Tensor index_cache = kv_cache.get_index_cache();
      torch::Tensor index_state = kv_cache.get_compress_index_state();
      torch::Tensor index_block_table =
          layer_tensor(dsa.block_tables, layer_id_, mapping.index_cache_idx);
      torch::Tensor index_slot =
          layer_tensor(dsa.slot_mappings, layer_id_, mapping.index_cache_idx);
      DeepseekV4IndexerCacheRefs refs{
          index_block_table,
          index_slot,
          layer_tensor(
              dsa.block_tables, layer_id_, mapping.index_kv_state_cache_idx)};
      AttentionMetadata indexer_metadata =
          make_indexer_metadata(attn_metadata, index_block_table, index_slot);
      std::tie(cmp_table_for_attn, cmp_context_lens) =
          indexer_->forward(hidden_states,
                            qr,
                            index_cache,
                            index_state,
                            indexer_metadata,
                            refs,
                            is_prefill || is_chunked_prefill,
                            dsa.compressed_sin_table,
                            dsa.compressed_cos_table,
                            /*projected_weights=*/parts[4],
                            /*projected_kv=*/parts[5],
                            /*projected_score=*/parts[6]);
      cmp_cache_for_attn = flatten_slot_cache(cmp_cache);
      cmp_max_context = dsa.index_total_c4_len > 0 ? index_topk_ : 0;
    } else {
      cmp_context_lens = dsa.c128_attn_metadata.context_lens;
      cmp_table_for_attn = dsa.c128_attn_metadata.block_table_for_attn;
      cmp_max_context = dsa.c128_attn_metadata.max_context_len;
    }

    if (cmp_max_context > 0) {
      torch::Tensor cmp_out = torch::empty_like(q_decode);
      std::optional<torch::Tensor> cmp_lse;
      run_decode(q_decode,
                 cmp_out,
                 cmp_lse,
                 cmp_cache_for_attn,
                 cmp_table_for_attn,
                 cmp_context_lens,
                 cmp_max_context,
                 /*window_left=*/-1,
                 scale_,
                 std::nullopt,
                 /*return_lse=*/true);
      xllm::kernel::mlu::update_out_and_lse(
          merged_out, merged_lse, cmp_out, cmp_lse.value());
    }
  }

  torch::Tensor attn_output = merged_out.squeeze(1).contiguous();
  apply_last_rope(attn_output,
                  active_inverse_sin_table,
                  active_cos_table,
                  dsa.input_positions,
                  rope_head_dim_);
  output = project_output(attn_output);
  return {output, std::nullopt};
}

void DeepseekV4AttentionImpl::load_state_dict(const StateDict& state_dict) {
  // Concatenate the per-segment hidden->* projection weights (kept under their
  // original checkpoint keys) into the single fused_wqa_wkv_ weight along the
  // output dim. The order must match hidden_proj_sizes_ exactly.
  std::vector<torch::Tensor> proj_parts;
  proj_parts.push_back(state_dict.get_tensor("wq_a.weight"));  // segment 0
  proj_parts.push_back(state_dict.get_tensor("wkv.weight"));   // segment 1
  if (compress_ratio_ == 4 || compress_ratio_ == 128) {
    proj_parts.push_back(
        state_dict.get_tensor("compressor.wkv.weight"));  // seg 2
    proj_parts.push_back(
        state_dict.get_tensor("compressor.wgate.weight"));  // seg 3
    if (compress_ratio_ == 4) {
      proj_parts.push_back(
          state_dict.get_tensor("indexer.weights_proj.weight"));  // seg 4
      proj_parts.push_back(
          state_dict.get_tensor("indexer.compressor.wkv.weight"));  // seg 5
      proj_parts.push_back(
          state_dict.get_tensor("indexer.compressor.wgate.weight"));  // seg 6
    }
  }
  for (size_t i = 0; i < proj_parts.size(); ++i) {
    CHECK(proj_parts[i].defined())
        << "fused_wqa_wkv: missing weight segment " << i;
  }
  torch::Tensor fused_weight = torch::cat(proj_parts, /*dim=*/0);
  std::unordered_map<std::string, torch::Tensor> fused_dict;
  fused_dict["weight"] = fused_weight;
  fused_wqa_wkv_->load_state_dict(StateDict(fused_dict));

  q_norm_->load_state_dict(state_dict.get_dict_with_prefix("q_norm."));
  wq_b_->load_state_dict(state_dict.get_dict_with_prefix("wq_b."));
  kv_norm_->load_state_dict(state_dict.get_dict_with_prefix("kv_norm."));
  wo_a_->load_state_dict(state_dict.get_dict_with_prefix("wo_a."));
  wo_b_->load_state_dict(state_dict.get_dict_with_prefix("wo_b."));

  torch::Tensor attn_sink = state_dict.get_tensor("attn_sink");
  if (!attn_sink.defined()) {
    attn_sink = state_dict.get_tensor("attn_sink.weight");
  }
  if (attn_sink.defined()) {
    if (attn_sink.dim() == 1 && attn_sink.size(0) == n_heads_ && tp_size_ > 1) {
      const int64_t shard_size = n_heads_ / tp_size_;
      const int64_t shard_start = tp_rank_ * shard_size;
      attn_sink = attn_sink.slice(
          /*dim=*/0, /*start=*/shard_start, /*end=*/shard_start + shard_size);
    }
    torch::NoGradGuard no_grad;
    attn_sink_.copy_(attn_sink.to(attn_sink_.device()).to(attn_sink_.dtype()));
    attn_sink_loaded_ = true;
  }

  // The projection weights (wkv/wgate/weights_proj) have been merged into the
  // fused GEMM, so skip loading them here; norm/ape/wq_b are still loaded.
  if (compressor_) {
    compressor_->load_state_dict(state_dict.get_dict_with_prefix("compressor."),
                                 /*skip_proj_weights=*/true);
  }
  if (indexer_) {
    indexer_->load_state_dict(state_dict.get_dict_with_prefix("indexer."),
                              /*skip_proj_weights=*/true);
  }
}

void DeepseekV4AttentionImpl::set_cache_mapping(
    const DSACacheMapping& mapping) {
  cache_mapping_ = mapping;
}

}  // namespace layer
}  // namespace xllm

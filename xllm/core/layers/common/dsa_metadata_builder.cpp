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

#include "dsa_metadata_builder.h"

#include <algorithm>

#include "attention_metadata.h"
#include "attention_metadata_builder.h"
#include "dsa_metadata.h"
#include "framework/model/model_input_params.h"
#include "util/tensor_helper.h"

namespace xllm::layer {

namespace {

torch::Device infer_metadata_device(const ModelInputParams& params,
                                    const torch::Tensor& positions) {
  if (positions.defined()) {
    return positions.device();
  }
  if (params.attention.device.kv_seq_lens.defined()) {
    return params.attention.device.kv_seq_lens.device();
  }
  if (params.attention.device.q_seq_lens.defined()) {
    return params.attention.device.q_seq_lens.device();
  }
  if (params.attention.device.new_cache_slots.defined()) {
    return params.attention.device.new_cache_slots.device();
  }
  if (params.attention.device.block_tables.defined()) {
    return params.attention.device.block_tables.device();
  }
  return torch::Device(torch::kCPU);
}

int64_t vector_max_or_zero(const std::vector<int32_t>& values) {
  if (values.empty()) {
    return 0;
  }
  return *std::max_element(values.begin(), values.end());
}

torch::Tensor pad_block_table(const torch::Tensor& block_table,
                              int32_t target_rows,
                              int32_t target_cols,
                              int32_t pad_value) {
  if (!block_table.defined() || block_table.dim() != 2 ||
      (block_table.size(0) >= target_rows &&
       block_table.size(1) >= target_cols)) {
    return block_table;
  }

  const int64_t rows = std::max<int64_t>(target_rows, block_table.size(0));
  const int64_t cols = std::max<int64_t>(target_cols, block_table.size(1));
  auto padded = torch::full({rows, cols}, pad_value, block_table.options());
  padded.slice(/*dim=*/0, /*start=*/0, /*end=*/block_table.size(0))
      .slice(/*dim=*/1, /*start=*/0, /*end=*/block_table.size(1))
      .copy_(block_table);
  return padded;
}

}  // namespace

AttentionMetadata DSAMetadataBuilder::build(
    const ModelInputParams& params,
    const torch::Tensor& positions,
    const torch::Tensor& dsa_cos_sin,
    const std::vector<std::vector<DSACacheInfo>>& caches_info,
    const std::vector<DSAGroupInfo>& group_infos,
    const torch::Tensor& dsa_c4_cos_sin,
    const torch::Tensor& dsa_c128_cos_sin) {
  // 1. Build base AttentionMetadata (q_cu_seq_lens, block_table, etc.)
  AttentionMetadata attn_metadata =
      AttentionMetadataBuilder::build(params,
                                      /*enable_mla=*/false,
                                      /*attn_mask=*/{},
                                      infer_metadata_device(params, positions));

  // 2. Build DSA-specific fields
  auto dsa_metadata = std::make_shared<DSAMetadata>();
  build_dsa_fields(params,
                   positions,
                   dsa_cos_sin,
                   dsa_c4_cos_sin,
                   dsa_c128_cos_sin,
                   caches_info,
                   group_infos,
                   *dsa_metadata);

  // 3. Keep DSA metadata independent while syncing base attention tensors.
  if (attn_metadata.attn_mask.defined()) {
    dsa_metadata->attn_mask = attn_metadata.attn_mask.clone();
  }

  if (attn_metadata.mrope_cos.defined() && !dsa_metadata->cos_table.defined()) {
    dsa_metadata->cos_table = attn_metadata.mrope_cos;
  }
  if (attn_metadata.mrope_sin.defined() && !dsa_metadata->sin_table.defined()) {
    dsa_metadata->sin_table = attn_metadata.mrope_sin;
  }

  // 4. Attach to AttentionMetadata
  attn_metadata.dsa_metadata = std::move(dsa_metadata);

  return attn_metadata;
}

void DSAMetadataBuilder::build_dsa_fields(
    const ModelInputParams& params,
    const torch::Tensor& positions,
    const torch::Tensor& dsa_cos_sin,
    const torch::Tensor& dsa_c4_cos_sin,
    const torch::Tensor& dsa_c128_cos_sin,
    const std::vector<std::vector<DSACacheInfo>>& caches_info,
    const std::vector<DSAGroupInfo>& group_infos,
    DSAMetadata& dsa) {
  const int32_t batch_size =
      static_cast<int32_t>(params.attention.host.kv_seq_lens.size());
  std::vector<int32_t> q_lens_vec;
  q_lens_vec.reserve(batch_size);

  dsa.input_positions = positions;
  const bool is_acl_graph = params.enable_graph;
  const torch::Device metadata_device(torch::kCPU);

  // Build per-batch sequence length metadata.
  build_seq_lengths(params, metadata_device, batch_size, dsa);
  dsa.is_acl_graph = is_acl_graph;
  if (static_cast<int32_t>(params.attention.host.q_seq_lens.size()) ==
      batch_size) {
    q_lens_vec.assign(params.attention.host.q_seq_lens.begin(),
                      params.attention.host.q_seq_lens.end());
  } else if (params.meta.batch_forward_type.no_decode()) {
    q_lens_vec.assign(params.attention.host.kv_seq_lens.begin(),
                      params.attention.host.kv_seq_lens.end());
  } else {
    q_lens_vec.assign(batch_size, 1);
  }

  // Keep base RoPE tables in metadata. Per-forward cos/sin slices are
  // calculated in DeepseekV4ModelImpl::forward to align with MindIE timing.
  if (dsa_cos_sin.defined()) {
    auto cos_sin_chunks = dsa_cos_sin.chunk(/*chunks=*/2, /*dim=*/-1);
    dsa.cos_table = cos_sin_chunks[0].contiguous();
    dsa.sin_table = cos_sin_chunks[1].contiguous();
  }

  if (positions.defined()) {
    build_positions(params, batch_size, dsa);
  }

  (void)dsa_c4_cos_sin;
  (void)dsa_c128_cos_sin;

  // --- Block tables / slots expansion ---
  if (!params.multi_block_tables.empty() && !caches_info.empty()) {
    std::vector<torch::Tensor> active_multi_block_tables =
        params.multi_block_tables;
    int32_t manager_num =
        static_cast<int32_t>(active_multi_block_tables.size());
    if (manager_num == 1 && batch_size == 1 &&
        active_multi_block_tables[0].defined() &&
        active_multi_block_tables[0].dim() == 2 &&
        active_multi_block_tables[0].size(0) > 1 &&
        static_cast<size_t>(active_multi_block_tables[0].size(0)) <=
            group_infos.size()) {
      const auto packed = active_multi_block_tables[0].contiguous();
      std::vector<torch::Tensor> unpacked_tables;
      unpacked_tables.reserve(packed.size(0));
      for (int64_t m = 0; m < packed.size(0); ++m) {
        unpacked_tables.push_back(packed[m].unsqueeze(0).contiguous());
      }
      active_multi_block_tables = std::move(unpacked_tables);
      manager_num = static_cast<int32_t>(active_multi_block_tables.size());
      LOG(WARNING)
          << "DSAMetadataBuilder detected packed multi_block_tables layout "
             "([manager, blocks]) while batch_size==1; auto-unpacked to "
             "[manager][batch, blocks]. manager_num="
          << manager_num;
    }

    CHECK_EQ(batch_size,
             static_cast<int32_t>(params.attention.host.kv_seq_lens.size()))
        << "DSAMetadataBuilder: batch_size mismatch with kv_seq_lens_vec size.";
    CHECK_LE(manager_num, static_cast<int32_t>(group_infos.size()))
        << "DSAMetadataBuilder: manager_num(" << manager_num
        << ") exceeds group_infos size(" << group_infos.size()
        << "), cannot align manager/group mapping.";
    int32_t graph_block_table_capacity_cols = 0;
    if (is_acl_graph && params.attention.device.block_tables.defined() &&
        params.attention.device.block_tables.dim() == 2) {
      graph_block_table_capacity_cols =
          static_cast<int32_t>(params.attention.device.block_tables.size(1));
    }
    if (is_acl_graph && graph_block_table_capacity_cols > 0) {
      for (int32_t m = 0; m < manager_num; ++m) {
        const auto& block_table = active_multi_block_tables[m];
        CHECK(block_table.defined() && block_table.dim() == 2)
            << "DSAMetadataBuilder: ACL graph multi_block_tables manager " << m
            << " must be a 2-D tensor.";
        CHECK_LE(block_table.size(1), graph_block_table_capacity_cols)
            << "DSAMetadataBuilder: ACL graph multi_block_tables exceeds "
            << "bucket column capacity: manager_id=" << m
            << ", required_cols=" << block_table.size(1)
            << ", capacity_cols=" << graph_block_table_capacity_cols;
      }
    }
    const int32_t n_layers = static_cast<int32_t>(caches_info.size());
    const auto& ctx_lens = params.attention.host.kv_seq_lens;
    const int64_t graph_slot_capacity =
        is_acl_graph && positions.defined() ? positions.numel() : 0;
    int64_t total_tokens = 0;
    for (int32_t len : ctx_lens) {
      total_tokens += len;
    }

    std::vector<torch::Tensor> proc_slots(manager_num);
    std::vector<torch::Tensor> proc_bt(manager_num);
    for (int32_t m = 0; m < manager_num; ++m) {
      process_group(active_multi_block_tables[m],
                    group_infos[m],
                    ctx_lens,
                    q_lens_vec,
                    batch_size,
                    total_tokens,
                    graph_slot_capacity,
                    graph_block_table_capacity_cols,
                    proc_bt[m],
                    proc_slots[m]);
    }

    // Keep expanded metadata on host. DeepSeek V4 packs these small tensors
    // into one contiguous transfer for both graph and non-graph NPU forwards.

    // Step 3: expand by layer using group_id
    dsa.block_tables.resize(n_layers);
    dsa.slot_mappings.resize(n_layers);
    for (int32_t lid = 0; lid < n_layers; ++lid) {
      const auto& lci = caches_info[lid];
      dsa.block_tables[lid].resize(lci.size());
      dsa.slot_mappings[lid].resize(lci.size());
      for (size_t ci = 0; ci < lci.size(); ++ci) {
        int32_t gid = lci[ci].group_id;
        if (gid < manager_num) {
          dsa.block_tables[lid][ci] = proc_bt[gid];
          dsa.slot_mappings[lid][ci] = proc_slots[gid];
        }
      }
    }
  }

  // Attach cache spec pointer
  dsa.caches_info = &caches_info;
}

torch::Tensor DSAMetadataBuilder::expand_blocks_to_slots(
    const torch::Tensor& block_table,
    const DSAGroupInfo& gi,
    const std::vector<int32_t>& ctx_lens,
    int32_t batch_size,
    int64_t total_tokens) {
  const int32_t bs = gi.block_size;
  auto slots = torch::full({total_tokens}, -1, torch::kInt32);
  auto slots_acc = slots.accessor<int32_t, 1>();
  auto bt_acc = block_table.accessor<int32_t, 2>();
  const int32_t max_blocks = static_cast<int32_t>(block_table.size(1));

  int64_t start_idx = 0;
  for (int32_t seq = 0; seq < batch_size; ++seq) {
    int64_t token_len = ctx_lens[seq];
    int64_t slot_num = compute_slot_num(gi, token_len);
    if (seq >= block_table.size(0)) {
      // ACL graph capture pads sequence-len vectors to bucket size while the
      // request-shaped CPU multi_block_tables only contains real batch rows.
      // Treat missing padded rows as dummy rows filled with -1.
      start_idx += token_len;
      continue;
    }

    int64_t filled = 0;
    for (int32_t blk = 0; blk < max_blocks && filled < slot_num; ++blk) {
      int32_t block_id = bt_acc[seq][blk];
      if (block_id < 0) break;
      for (int32_t off = 0; off < bs && filled < slot_num; ++off) {
        slots_acc[start_idx + filled] =
            static_cast<int32_t>(static_cast<int64_t>(block_id) * bs + off);
        ++filled;
      }
    }
    start_idx += token_len;
  }
  return slots;
}

int64_t DSAMetadataBuilder::compute_slot_num(const DSAGroupInfo& gi,
                                             int64_t token_len) {
  if (gi.type == DSACacheType::TOKEN) {
    return token_len / gi.ratio;
  }
  // SLIDING_WINDOW
  const int32_t bs = gi.block_size;
  if (token_len > bs) {
    return token_len % bs + bs;
  }
  int64_t n = token_len % bs;
  return (n == 0 && token_len > 0) ? bs : n;
}

void DSAMetadataBuilder::process_group(const torch::Tensor& raw_bt,
                                       const DSAGroupInfo& gi,
                                       const std::vector<int32_t>& ctx_lens,
                                       const std::vector<int32_t>& q_lens,
                                       int32_t batch_size,
                                       int64_t total_tokens,
                                       int64_t graph_slot_capacity,
                                       int32_t block_table_capacity_cols,
                                       torch::Tensor& out_bt,
                                       torch::Tensor& out_slots) {
  if (gi.type == DSACacheType::TOKEN) {
    process_token_group(raw_bt,
                        gi.ratio,
                        gi.block_size,
                        ctx_lens,
                        q_lens,
                        batch_size,
                        total_tokens,
                        graph_slot_capacity,
                        block_table_capacity_cols,
                        out_bt,
                        out_slots);
  } else if (gi.type == DSACacheType::SLIDING_WINDOW) {
    process_swa_group(raw_bt,
                      gi.block_size,
                      ctx_lens,
                      q_lens,
                      batch_size,
                      graph_slot_capacity,
                      block_table_capacity_cols,
                      out_bt,
                      out_slots);
  } else {
    auto raw_slots =
        expand_blocks_to_slots(raw_bt, gi, ctx_lens, batch_size, total_tokens);
    out_slots =
        torch::where(raw_slots.eq(-1), torch::zeros_like(raw_slots), raw_slots);
    out_bt = raw_bt;
  }
}

void DSAMetadataBuilder::process_token_group(
    const torch::Tensor& raw_bt,
    int32_t ratio,
    int32_t block_size,
    const std::vector<int32_t>& ctx_lens,
    const std::vector<int32_t>& q_lens,
    int32_t batch_size,
    int64_t total_tokens,
    int64_t graph_slot_capacity,
    int32_t block_table_capacity_cols,
    torch::Tensor& out_bt,
    torch::Tensor& out_slots) {
  CHECK_EQ(static_cast<int32_t>(ctx_lens.size()), batch_size)
      << "process_token_group requires ctx_lens.size == batch_size, got "
      << ctx_lens.size() << " vs " << batch_size;
  CHECK_EQ(static_cast<int32_t>(q_lens.size()), batch_size)
      << "process_token_group requires q_lens.size == batch_size, got "
      << q_lens.size() << " vs " << batch_size;
  CHECK_GT(ratio, 0) << "process_token_group requires ratio > 0, got " << ratio;
  CHECK_GT(block_size, 0) << "process_token_group requires block_size > 0, got "
                          << block_size;
  CHECK_EQ(raw_bt.dim(), 2)
      << "process_token_group requires raw_bt dim == 2, got " << raw_bt.dim();

  int64_t query_total_tokens = 0;
  for (const int q_len : q_lens) {
    query_total_tokens += static_cast<int64_t>(q_len);
  }

  // Token caches commit one row only when a sequence crosses a compression
  // boundary. Count rows per sequence so multi-batch padding/dummy rows do not
  // become cache writes.
  int64_t committed_rows = 0;
  for (int32_t seq = 0; seq < batch_size; ++seq) {
    const int64_t ctx_len = static_cast<int64_t>(ctx_lens[seq]);
    const int64_t q_len =
        std::clamp<int64_t>(static_cast<int64_t>(q_lens[seq]), 0, ctx_len);
    const int64_t prev_ctx_len = ctx_len - q_len;
    committed_rows += ctx_len / ratio - prev_ctx_len / ratio;
  }

  // Token caches write only the compressed rows produced by the current
  // forward step. Padded RoPE/compressor rows must not become cache writes.
  const int64_t out_slot_rows =
      graph_slot_capacity > 0
          ? std::max<int64_t>(graph_slot_capacity, committed_rows)
          : committed_rows;
  auto out_slots_tensor = torch::full({out_slot_rows}, -1, raw_bt.options());
  auto out_slots_acc = out_slots_tensor.accessor<int32_t, 1>();
  auto raw_bt_acc = raw_bt.accessor<int32_t, 2>();
  const int64_t semantic_cols = raw_bt.size(1);
  const int64_t block_size_i64 = static_cast<int64_t>(block_size);

  auto slot_for_compressed_index = [&](int32_t seq,
                                       int64_t compressed_idx) -> int32_t {
    if (seq >= raw_bt.size(0) || semantic_cols <= 0) {
      return -1;
    }
    const int64_t block_idx = compressed_idx / block_size_i64;
    if (block_idx >= semantic_cols) {
      return -1;
    }
    const int32_t block_id = raw_bt_acc[seq][block_idx];
    if (block_id < 0) {
      return -1;
    }
    const int64_t block_offset = compressed_idx % block_size_i64;
    return static_cast<int32_t>(
        static_cast<int64_t>(block_id) * block_size_i64 + block_offset);
  };

  int64_t write_idx = 0;
  for (int32_t seq = 0; seq < batch_size; ++seq) {
    const int64_t ctx_len = static_cast<int64_t>(ctx_lens[seq]);
    const int64_t q_len =
        std::clamp<int64_t>(static_cast<int64_t>(q_lens[seq]), 0, ctx_len);
    const int64_t prev_ctx_len = ctx_len - q_len;
    const int64_t prev_committed = prev_ctx_len / ratio;
    const int64_t committed = ctx_len / ratio;
    const int64_t new_committed = committed - prev_committed;
    for (int64_t i = 0; i < new_committed; ++i) {
      out_slots_acc[write_idx++] =
          slot_for_compressed_index(seq, prev_committed + i);
    }
  }

  CHECK_EQ(write_idx, committed_rows)
      << "process_token_group committed slot count mismatch: write_idx="
      << write_idx << ", committed_rows=" << committed_rows
      << ", query_total_tokens=" << query_total_tokens << ", ratio=" << ratio
      << ", batch_size=" << batch_size << ", total_tokens=" << total_tokens;
  out_slots = out_slots_tensor;
  out_bt = graph_slot_capacity > 0
               ? pad_block_table(
                     raw_bt,
                     batch_size,
                     std::max<int32_t>(block_table_capacity_cols,
                                       static_cast<int32_t>(raw_bt.size(1))),
                     /*pad_value=*/-1)
               : raw_bt;
}

void DSAMetadataBuilder::process_swa_group(const torch::Tensor& raw_bt,
                                           int32_t block_size,
                                           const std::vector<int32_t>& ctx_lens,
                                           const std::vector<int32_t>& q_lens,
                                           int32_t batch_size,
                                           int64_t graph_slot_capacity,
                                           int32_t block_table_capacity_cols,
                                           torch::Tensor& out_bt,
                                           torch::Tensor& out_slots) {
  CHECK_EQ(static_cast<int32_t>(ctx_lens.size()), batch_size)
      << "process_swa_group requires ctx_lens.size == batch_size, got "
      << ctx_lens.size() << " vs " << batch_size;
  CHECK_EQ(static_cast<int32_t>(q_lens.size()), batch_size)
      << "process_swa_group requires q_lens.size == batch_size, got "
      << q_lens.size() << " vs " << batch_size;
  CHECK_GT(block_size, 0) << "process_swa_group requires block_size > 0, got "
                          << block_size;
  CHECK_EQ(raw_bt.dim(), 2)
      << "process_swa_group requires raw_bt dim == 2, got " << raw_bt.dim();

  int64_t query_total_tokens = 0;
  for (int32_t seq = 0; seq < batch_size; ++seq) {
    query_total_tokens += std::clamp<int64_t>(
        static_cast<int64_t>(q_lens[seq]), 0, ctx_lens[seq]);
  }

  // SWA cache writes only the tokens produced by the current forward step.
  // Do not reuse the full-context raw_slots prefix here: decode has one kv row,
  // and prefix slot 0 would incorrectly overwrite cache row 0.
  const int64_t out_slot_rows =
      graph_slot_capacity > 0
          ? std::max<int64_t>(graph_slot_capacity, query_total_tokens)
          : query_total_tokens;
  auto out_slots_tensor = torch::full({out_slot_rows}, -1, raw_bt.options());
  auto out_slots_acc = out_slots_tensor.accessor<int32_t, 1>();
  auto raw_bt_acc = raw_bt.accessor<int32_t, 2>();
  const int64_t semantic_cols = raw_bt.size(1);
  const int64_t storage_cols =
      graph_slot_capacity > 0 && block_table_capacity_cols > 0
          ? std::max<int64_t>(block_table_capacity_cols, raw_bt.size(1))
          : raw_bt.size(1);
  const int64_t block_size_i64 = static_cast<int64_t>(block_size);

  auto slot_for_position = [&](int32_t seq, int64_t pos) -> int32_t {
    if (semantic_cols <= 0) {
      return -1;
    }
    const int64_t block_idx = (pos / block_size_i64) % semantic_cols;
    const int32_t block_id = raw_bt_acc[seq][block_idx];
    if (block_id < 0) {
      return -1;
    }
    const int64_t block_offset = pos % block_size_i64;
    return static_cast<int32_t>(
        static_cast<int64_t>(block_id) * block_size_i64 + block_offset);
  };

  int64_t write_idx = 0;
  for (int32_t seq = 0; seq < batch_size; ++seq) {
    const int64_t ctx_len = static_cast<int64_t>(ctx_lens[seq]);
    const int64_t q_len =
        std::clamp<int64_t>(static_cast<int64_t>(q_lens[seq]), 0, ctx_len);
    if (seq >= raw_bt.size(0)) {
      write_idx += q_len;
      continue;
    }
    const int64_t q_start = ctx_len - q_len;
    for (int64_t i = 0; i < q_len; ++i) {
      out_slots_acc[write_idx++] = slot_for_position(seq, q_start + i);
    }
  }

  out_slots = out_slots_tensor;

  const int32_t current_cols = static_cast<int32_t>(semantic_cols);
  int32_t max_dst_len = 0;
  std::vector<int32_t> dst_lens(batch_size);
  for (int32_t s = 0; s < batch_size; ++s) {
    const int64_t ctx_len = std::max<int64_t>(ctx_lens[s], 0);
    dst_lens[s] = static_cast<int32_t>((ctx_len + block_size - 1) / block_size);
    max_dst_len = std::max(max_dst_len, dst_lens[s]);
  }
  max_dst_len = std::max(max_dst_len, current_cols);
  if (graph_slot_capacity > 0) {
    max_dst_len =
        std::max<int32_t>(max_dst_len, static_cast<int32_t>(storage_cols));
  }

  auto new_bt = torch::full({batch_size, max_dst_len}, -1, raw_bt.options());
  auto new_acc = new_bt.accessor<int32_t, 2>();
  auto old_acc = raw_bt.accessor<int32_t, 2>();

  for (int32_t s = 0; s < batch_size; ++s) {
    if (s >= raw_bt.size(0)) {
      continue;
    }
    const int32_t retained_cols = std::min(current_cols, dst_lens[s]);
    const int32_t start_col = dst_lens[s] - retained_cols;
    for (int32_t j = 0; j < retained_cols; ++j) {
      const int32_t logical_col = start_col + j;
      // Keep the read-side block table aligned with slot_for_position().
      const int32_t physical_col = logical_col % current_cols;
      new_acc[s][logical_col] = old_acc[s][physical_col];
    }
  }
  out_bt = new_bt;
}

void DSAMetadataBuilder::build_seq_lengths(const ModelInputParams& params,
                                           const torch::Device& target_device,
                                           int32_t batch_size,
                                           DSAMetadata& dsa_metadata) {
  auto int_options =
      torch::TensorOptions().dtype(torch::kInt32).device(target_device);
  torch::Tensor kv_lens = params.attention.device.kv_seq_lens;
  if (target_device.is_cpu() &&
      static_cast<int32_t>(params.attention.host.kv_seq_lens.size()) ==
          batch_size) {
    kv_lens = torch::tensor(
        std::vector<int32_t>(params.attention.host.kv_seq_lens.begin(),
                             params.attention.host.kv_seq_lens.end()),
        int_options);
  } else if (!kv_lens.defined() || kv_lens.numel() == 0) {
    kv_lens = torch::tensor(
        std::vector<int32_t>(params.attention.host.kv_seq_lens.begin(),
                             params.attention.host.kv_seq_lens.end()),
        int_options);
  } else if (kv_lens.device() != target_device) {
    kv_lens = safe_to(kv_lens, int_options, true);
  }
  dsa_metadata.seq_lens = kv_lens;
  dsa_metadata.actual_seq_lengths_kv = kv_lens;

  torch::Tensor q_lens;
  q_lens = params.attention.device.q_seq_lens;
  if (static_cast<int32_t>(params.attention.host.q_seq_lens.size()) ==
      batch_size) {
    // Prefer explicit per-sequence query lengths from ModelInputParams.
    // This is accurate for prefill/decode/chunked/mixed batches.
    if (target_device.is_cpu()) {
      q_lens = torch::tensor(
          std::vector<int32_t>(params.attention.host.q_seq_lens.begin(),
                               params.attention.host.q_seq_lens.end()),
          int_options);
    } else if (!q_lens.defined() || q_lens.numel() == 0) {
      q_lens = torch::tensor(
          std::vector<int32_t>(params.attention.host.q_seq_lens.begin(),
                               params.attention.host.q_seq_lens.end()),
          int_options);
    } else if (q_lens.device() != target_device) {
      q_lens = safe_to(q_lens, int_options, true);
    }
  } else if (params.meta.batch_forward_type.no_decode()) {
    // Pure prefill path fallback: query lengths follow KV context lengths.
    q_lens = kv_lens;
  } else {
    // Decode fallback: each sequence contributes one query token.
    q_lens = torch::ones({batch_size}, int_options);
  }
  // cumsum with leading 0: shape (batch_size+1,)
  auto cumsum = torch::cumsum(q_lens, /*dim=*/0, /*dtype=*/torch::kInt32);
  dsa_metadata.actual_seq_lengths_query =
      torch::cat({torch::zeros({1}, int_options), cumsum});
  dsa_metadata.seq_lens_q = q_lens;

  // Precompute the kv cumulative sequence lengths once per forward so the
  // per-layer indexer metadata builder can reuse it instead of running a
  // host-side cumsum on every DSA layer (kv_lens is identical across layers
  // within one forward).
  if (kv_lens.numel() > 0) {
    torch::Tensor kv_lens_i32 = kv_lens.to(torch::kInt32);
    torch::Tensor kv_cumsum =
        torch::cumsum(kv_lens_i32, /*dim=*/0, /*dtype=*/torch::kInt32);
    dsa_metadata.kv_cu_seq_lens =
        torch::cat({torch::zeros({1}, int_options), kv_cumsum});
  }

  if (kv_lens.numel() > 0) {
    dsa_metadata.max_seqlen_kv = torch::max(kv_lens).to(torch::kInt32);
  } else {
    dsa_metadata.max_seqlen_kv = torch::zeros({1}, int_options);
  }

  if (q_lens.numel() > 0) {
    dsa_metadata.max_seqlen_q = torch::max(q_lens).to(torch::kInt32);
  } else {
    dsa_metadata.max_seqlen_q = torch::zeros({1}, int_options);
  }

  dsa_metadata.max_query_len =
      std::max<int64_t>(params.meta.q_max_seq_len,
                        vector_max_or_zero(params.attention.host.q_seq_lens));
  dsa_metadata.max_seq_len =
      std::max<int64_t>(params.meta.kv_max_seq_len,
                        vector_max_or_zero(params.attention.host.kv_seq_lens));
}

void DSAMetadataBuilder::build_positions(const ModelInputParams& params,
                                         int32_t batch_size,
                                         DSAMetadata& dsa_metadata) {
  if (!dsa_metadata.input_positions.defined()) return;

  auto input_positions = dsa_metadata.input_positions;
  int64_t num_tokens = input_positions.size(0);
  const bool is_acl_graph = params.enable_graph;
  const auto target_device = input_positions.device();
  const auto pos_dtype = input_positions.scalar_type();
  auto cpu_options =
      torch::TensorOptions().dtype(pos_dtype).device(torch::kCPU);

  const bool has_host_lens =
      static_cast<int32_t>(params.attention.host.kv_seq_lens.size()) ==
      batch_size;
  if (has_host_lens) {
    std::vector<int64_t> c4_positions;
    std::vector<int64_t> c128_positions;
    c4_positions.reserve(
        static_cast<size_t>(std::max<int64_t>(num_tokens / 4 + batch_size, 0)));
    c128_positions.reserve(static_cast<size_t>(
        std::max<int64_t>(num_tokens / 128 + batch_size, 0)));
    for (int32_t seq = 0; seq < batch_size; ++seq) {
      const int64_t kv_len = params.attention.host.kv_seq_lens[seq];
      int64_t q_len = 1;
      if (static_cast<int32_t>(params.attention.host.q_seq_lens.size()) ==
          batch_size) {
        q_len = params.attention.host.q_seq_lens[seq];
      } else if (params.meta.batch_forward_type.no_decode()) {
        q_len = kv_len;
      }
      q_len = std::clamp<int64_t>(q_len, 0, kv_len);
      const int64_t start_pos = kv_len - q_len;
      for (int64_t i = 0; i < q_len; ++i) {
        const int64_t pos = start_pos + i;
        const int64_t next_pos = pos + 1;
        if (next_pos % 4 == 0) {
          c4_positions.push_back(next_pos - 4);
        }
        if (next_pos % 128 == 0) {
          c128_positions.push_back(next_pos - 128);
        }
      }
    }

    const int64_t c4_target =
        is_acl_graph
            ? num_tokens
            : std::min<int64_t>(num_tokens, num_tokens / 4 + batch_size);
    const int64_t c128_target =
        is_acl_graph
            ? num_tokens
            : std::min<int64_t>(num_tokens, num_tokens / 128 + batch_size);
    c4_positions.resize(static_cast<size_t>(std::max<int64_t>(c4_target, 0)),
                        0);
    c128_positions.resize(
        static_cast<size_t>(std::max<int64_t>(c128_target, 0)), 0);

    dsa_metadata.c4_pad_positions = torch::tensor(c4_positions, cpu_options);
    dsa_metadata.c128_pad_positions =
        torch::tensor(c128_positions, cpu_options);
    return;
  }

  std::vector<int64_t> host_positions;
  host_positions.reserve(static_cast<size_t>(std::max<int64_t>(num_tokens, 0)));
  if (static_cast<int64_t>(host_positions.size()) < num_tokens &&
      input_positions.device().is_cpu()) {
    auto positions_cpu = input_positions.contiguous();
    if (positions_cpu.scalar_type() == torch::kInt64) {
      auto acc = positions_cpu.accessor<int64_t, 1>();
      for (int64_t i = static_cast<int64_t>(host_positions.size());
           i < num_tokens;
           ++i) {
        host_positions.push_back(acc[i]);
      }
    } else {
      auto positions_i64 = positions_cpu.to(torch::kInt64);
      auto acc = positions_i64.accessor<int64_t, 1>();
      for (int64_t i = static_cast<int64_t>(host_positions.size());
           i < num_tokens;
           ++i) {
        host_positions.push_back(acc[i]);
      }
    }
  }

  if (static_cast<int64_t>(host_positions.size()) > num_tokens) {
    host_positions.resize(static_cast<size_t>(num_tokens));
  }
  if (static_cast<int64_t>(host_positions.size()) < num_tokens) {
    host_positions.resize(static_cast<size_t>(num_tokens), 0);
  }

  auto build_compressed_positions = [&](int64_t ratio) {
    std::vector<int64_t> compressed;
    compressed.reserve(host_positions.size() / ratio + batch_size);
    for (const int64_t pos : host_positions) {
      if ((pos + 1) % ratio == 0) {
        compressed.push_back((pos + 1) - ratio);
      }
    }
    const int64_t target =
        is_acl_graph
            ? num_tokens
            : std::min<int64_t>(num_tokens, num_tokens / ratio + batch_size);
    compressed.resize(static_cast<size_t>(std::max<int64_t>(target, 0)), 0);
    auto tensor = torch::tensor(compressed, cpu_options);
    return tensor;
  };

  dsa_metadata.c4_pad_positions = build_compressed_positions(4);
  dsa_metadata.c128_pad_positions = build_compressed_positions(128);
}

DSACPMetadata DSAMetadataBuilder::build_cp_local_metadata(
    const torch::Tensor& query_start_loc,
    const torch::Tensor& seq_lens,
    int32_t cp_size,
    int32_t cp_rank,
    const torch::Tensor& full_cos,
    const torch::Tensor& full_sin) {
  TORCH_CHECK(cp_size >= 1, "cp_size must be >= 1, got ", cp_size);
  TORCH_CHECK(cp_rank >= 0 && cp_rank < cp_size,
              "cp_rank must be in [0, cp_size), got cp_rank=",
              cp_rank,
              ", cp_size=",
              cp_size);
  TORCH_CHECK(query_start_loc.defined() && query_start_loc.dim() == 1,
              "query_start_loc must be a defined 1-D tensor");
  TORCH_CHECK(query_start_loc.numel() >= 1,
              "query_start_loc must have at least one element (leading 0)");
  TORCH_CHECK(seq_lens.defined() && seq_lens.dim() == 1,
              "seq_lens must be a defined 1-D tensor");

  // Work in int64 on CPU for exact host-side index math, then cast back to
  // the caller's dtype at the end so the result matches DSAMetadata
  // conventions.
  const auto in_dtype = query_start_loc.scalar_type() == torch::kInt64
                            ? torch::kInt64
                            : torch::kInt32;
  auto qsl = query_start_loc.to(torch::kCPU).to(torch::kInt64).contiguous();
  auto slens = seq_lens.to(torch::kCPU).to(torch::kInt64).contiguous();

  const int64_t num_reqs = qsl.numel() - 1;
  TORCH_CHECK(slens.numel() == num_reqs,
              "seq_lens size (",
              slens.numel(),
              ") must equal num_reqs (",
              num_reqs,
              ") implied by query_start_loc");

  const int64_t num_input_tokens =
      num_reqs > 0 ? qsl[num_reqs].item<int64_t>() : 0;

  // Split the flattened token stream evenly across CP ranks. Padding keeps
  // every rank's local slice the same length, matching vllm-ascend.
  const int64_t num_tokens_pad =
      ((num_input_tokens + cp_size - 1) / cp_size) * cp_size;
  const int64_t tokens_per_rank = num_tokens_pad / cp_size;
  const int64_t local_start = static_cast<int64_t>(cp_rank) * tokens_per_rank;
  const int64_t local_end = local_start + tokens_per_rank;

  auto int_cpu =
      torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU);

  DSACPMetadata cp;
  cp.local_start = local_start;
  cp.local_end = local_end;
  cp.tokens_per_rank = tokens_per_rank;
  cp.num_tokens_pad = num_tokens_pad;

  if (num_reqs <= 0) {
    cp.local_query_start_loc = torch::zeros({1}, int_cpu).to(in_dtype);
    cp.local_seq_lens = torch::zeros({0}, int_cpu).to(in_dtype);
  } else {
    // Intersect each request's global token interval with this rank's local
    // token interval, then build the per-rank query_start_loc from lengths.
    auto q_start = qsl.slice(/*dim=*/0, /*start=*/0, /*end=*/num_reqs);
    auto q_end = qsl.slice(/*dim=*/0, /*start=*/1, /*end=*/num_reqs + 1);
    auto local_q_start = q_start.clamp(local_start, local_end);
    auto local_q_end = q_end.clamp(local_start, local_end);
    auto local_q_lens = local_q_end - local_q_start;  // (num_reqs,)

    auto local_qsl = torch::zeros({num_reqs + 1}, int_cpu);
    // NOTE: assigning to a slice temporary does NOT write into local_qsl.
    // Use narrow().copy_() to write into the underlying storage.
    local_qsl.narrow(/*dim=*/0, /*start=*/1, /*length=*/num_reqs)
        .copy_(torch::cumsum(local_q_lens, /*dim=*/0));

    // For requests that cross the local slice boundary, offset removes the
    // tokens that live on later ranks so local_seq_lens matches local
    // queries.
    auto offset = q_end - local_q_end;  // (num_reqs,)
    auto has_local = (local_q_lens > 0).to(torch::kInt64);
    auto local_slens = has_local * (slens - offset);  // (num_reqs,)

    cp.local_query_start_loc = local_qsl.to(in_dtype);
    cp.local_seq_lens = local_slens.to(in_dtype);
  }

  // Slice RoPE tables to the local token interval when provided. Tables are
  // expected to cover the PADDED global token stream (length
  // num_tokens_pad).
  if (full_cos.defined() && full_cos.numel() > 0) {
    TORCH_CHECK(full_cos.size(0) >= local_end,
                "full_cos length (",
                full_cos.size(0),
                ") must cover local_end (",
                local_end,
                "); pass a table padded to num_tokens_pad");
    cp.local_cos =
        full_cos.slice(/*dim=*/0, /*start=*/local_start, /*end=*/local_end)
            .contiguous();
  }
  if (full_sin.defined() && full_sin.numel() > 0) {
    TORCH_CHECK(full_sin.size(0) >= local_end,
                "full_sin length (",
                full_sin.size(0),
                ") must cover local_end (",
                local_end,
                "); pass a table padded to num_tokens_pad");
    cp.local_sin =
        full_sin.slice(/*dim=*/0, /*start=*/local_start, /*end=*/local_end)
            .contiguous();
  }

  return cp;
}

}  // namespace xllm::layer

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

#include "layers/mlu/deepseek_v4/dsa_metadata_builder_mlu.h"

#include <glog/logging.h>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <vector>

#include "framework/model/model_input_params.h"
#include "layers/common/attention_metadata.h"
#include "layers/common/attention_metadata_builder.h"
#include "layers/common/dsa_metadata.h"
#include "util/tensor_helper.h"

namespace xllm::layer {

namespace {

constexpr int64_t kIndexC4Ratio = 4;
constexpr int64_t kC128Ratio = 128;

void build_compressed_pad_positions(const torch::Tensor& input_positions,
                                    int64_t ratio,
                                    int32_t batch_size,
                                    int64_t num_tokens,
                                    torch::Tensor& out) {
  torch::Tensor mask = ((input_positions + 1) % ratio).eq(0);
  torch::Tensor cmp_pos = input_positions.index({mask});
  cmp_pos = (cmp_pos + 1) - ratio;
  const int64_t target = std::min(num_tokens, num_tokens / ratio + batch_size);
  const int64_t pad_right = target - cmp_pos.size(0);
  if (pad_right > 0) {
    out = torch::cat({cmp_pos, torch::zeros({pad_right}, cmp_pos.options())});
  } else {
    out = cmp_pos.slice(0, 0, target);
  }
}

std::vector<int32_t> tensor_to_vec(const torch::Tensor& tensor,
                                   const char* name) {
  CHECK(tensor.defined()) << "DSAMetadataBuilderMlu requires " << name << ".";
  CHECK_EQ(tensor.dim(), 1)
      << "DSAMetadataBuilderMlu expects " << name << " to be 1D.";
  torch::Tensor cpu_tensor =
      tensor.to(torch::TensorOptions().device(torch::kCPU).dtype(torch::kInt32))
          .contiguous();
  const int64_t numel = cpu_tensor.numel();
  std::vector<int32_t> values;
  values.reserve(static_cast<size_t>(numel));
  if (numel > 0) {
    const int32_t* data = cpu_tensor.data_ptr<int32_t>();
    values.assign(data, data + numel);
  }
  return values;
}

torch::Tensor sanitize_block_table(const torch::Tensor& table) {
  if (!table.defined()) {
    return table;
  }
  return table.clamp_min(0);
}

torch::Tensor sanitize_swa_table(const torch::Tensor& table,
                                 int32_t block_size,
                                 const std::vector<int32_t>& ctx_lens,
                                 const std::vector<int32_t>& q_lens,
                                 int32_t batch_size,
                                 int64_t window_size) {
  if (!table.defined()) {
    return table;
  }
  CHECK_GT(block_size, 0) << "SWA block_size must be positive.";
  CHECK_EQ(static_cast<int32_t>(ctx_lens.size()), batch_size)
      << "SWA ctx_lens size must match batch_size.";
  CHECK_EQ(static_cast<int32_t>(q_lens.size()), batch_size)
      << "SWA q_lens size must match batch_size.";
  CHECK_EQ(table.dim(), 2) << "SWA block table must be 2D.";
  CHECK_GE(table.size(0), batch_size)
      << "SWA block table rows must cover the active batch.";

  torch::Tensor contig_table = table.contiguous();
  auto table_acc = contig_table.accessor<int32_t, 2>();
  const int64_t cols = contig_table.size(1);
  const int64_t block_size_i64 = static_cast<int64_t>(block_size);
  const int64_t window_left =
      window_size > 0 ? std::max<int64_t>(window_size - 1, 0) : 0;

  for (int32_t seq = 0; seq < batch_size; ++seq) {
    const int64_t ctx_len = static_cast<int64_t>(ctx_lens[seq]);
    if (ctx_len <= 0) {
      continue;
    }
    const int64_t q_len =
        std::clamp<int64_t>(static_cast<int64_t>(q_lens[seq]), 0, ctx_len);
    if (q_len <= 0) {
      continue;
    }
    const int64_t q_start = ctx_len - q_len;
    const int64_t live_token_begin =
        std::max<int64_t>(0, q_start - window_left);
    const int64_t live_col_begin = live_token_begin / block_size_i64;
    const int64_t live_col_end = (ctx_len - 1) / block_size_i64;

    CHECK_LT(live_col_end, cols)
        << "SWA absolute block table is too short for the live window. seq="
        << seq << ", ctx_len=" << ctx_len << ", q_len=" << q_len
        << ", block_size=" << block_size << ", live_col_end=" << live_col_end
        << ", cols=" << cols;
    for (int64_t col = live_col_begin; col <= live_col_end; ++col) {
      CHECK_GE(table_acc[seq][col], 0)
          << "SWA live window has an invalid absolute block. seq=" << seq
          << ", col=" << col << ", ctx_len=" << ctx_len << ", q_len=" << q_len
          << ", block_size=" << block_size;
    }
  }

  return sanitize_block_table(contig_table);
}

void sync_dsa_seq_metadata(AttentionMetadata& attn_metadata,
                           const DSAMetadata& dsa_metadata) {
  attn_metadata.q_cu_seq_lens = dsa_metadata.q_cu_seq_lens;
  attn_metadata.kv_cu_seq_lens = dsa_metadata.kv_cu_seq_lens;
  attn_metadata.q_seq_lens = dsa_metadata.q_seq_lens;
  attn_metadata.kv_seq_lens = dsa_metadata.kv_seq_lens;
}

torch::Tensor build_decode_slot_mapping(const torch::Tensor& cache_block_table,
                                        const xllm::layer::DSAMetadata& dsa,
                                        int64_t compress_ratio,
                                        int64_t block_size,
                                        const torch::Device& device) {
  const int64_t batch_size = static_cast<int64_t>(dsa.start_pos_vec.size());
  if (batch_size == 0 || !dsa.input_positions.defined() ||
      dsa.input_positions.numel() == 0) {
    return torch::empty(
        {0}, torch::TensorOptions().dtype(torch::kInt32).device(device));
  }
  CHECK_EQ(static_cast<int64_t>(dsa.query_start_offsets.size()), batch_size + 1)
      << "DSAMetadataBuilderMlu: query_start_offsets size must equal "
         "batch_size + 1.";

  auto block_table_cpu = cache_block_table.contiguous().to(torch::kCPU);
  auto block_table_acc = block_table_cpu.accessor<int32_t, 2>();
  auto pos_cpu = dsa.input_positions.contiguous().to(torch::kCPU);
  auto pos_acc = pos_cpu.accessor<int32_t, 1>();
  std::vector<int32_t> slot_mapping_vec;
  slot_mapping_vec.reserve(static_cast<size_t>(dsa.input_positions.numel()));
  for (int64_t seq_idx = 0; seq_idx < batch_size; ++seq_idx) {
    const int64_t q_begin = dsa.query_start_offsets[seq_idx];
    const int64_t q_end = dsa.query_start_offsets[seq_idx + 1];
    for (int64_t ci = q_begin; ci < q_end; ++ci) {
      int32_t lid = pos_acc[ci] / compress_ratio;
      int32_t block_col = lid / block_size;
      slot_mapping_vec.push_back(
          block_table_acc[seq_idx][block_col] * block_size + lid % block_size);
    }
  }
  return torch::tensor(
      slot_mapping_vec,
      torch::TensorOptions().dtype(torch::kInt32).device(device));
}

}  // namespace

AttentionMetadata DSAMetadataBuilderMlu::build(
    const ModelInputParams& params,
    const torch::Tensor& positions,
    const std::vector<std::vector<DSACacheInfo>>& caches_info,
    const std::vector<DSAGroupInfo>& group_infos,
    int64_t window_size) {
  // 1. Build base AttentionMetadata (q_cu_seq_lens, block_table, etc.)
  AttentionMetadata attn_metadata =
      AttentionMetadataBuilder::build(params, /*enable_mla=*/true);

  // 2. Keep DSA metadata independent while syncing base attention tensors.
  auto dsa_metadata = std::make_shared<DSAMetadata>();
  if (attn_metadata.is_dummy) {
    attn_metadata.dsa_metadata = std::move(dsa_metadata);
    return attn_metadata;
  }

  // 3. Build DSA-specific fields
  build_dsa_fields(params,
                   attn_metadata,
                   positions,
                   caches_info,
                   group_infos,
                   window_size,
                   *dsa_metadata);

  // 4. Sync canonical DSA sequence metadata to AttentionMetadata.
  sync_dsa_seq_metadata(attn_metadata, *dsa_metadata);

  // 5. Attach to AttentionMetadata
  attn_metadata.dsa_metadata = std::move(dsa_metadata);

  return attn_metadata;
}

void DSAMetadataBuilderMlu::build_dsa_fields(
    const ModelInputParams& params,
    const AttentionMetadata& attn_metadata,
    const torch::Tensor& positions,
    const std::vector<std::vector<DSACacheInfo>>& caches_info,
    const std::vector<DSAGroupInfo>& group_infos,
    int64_t window_size,
    DSAMetadata& dsa) {
  const bool is_decode = params.meta.batch_forward_type.is_decode();
  const int32_t batch_size = params.meta.num_sequences;
  std::vector<int32_t> q_lens_vec;
  std::vector<int32_t> kv_lens_vec;

  dsa.input_positions = positions;

  // Build per-batch sequence length metadata.
  build_seq_lengths(attn_metadata, batch_size, dsa, q_lens_vec, kv_lens_vec);

  if (positions.defined() && !is_decode) {
    build_positions(params, batch_size, dsa);
  }

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
          << "DSAMetadataBuilderMlu detected packed multi_block_tables layout "
          << "([manager, blocks]) while batch_size==1; auto-unpacked to "
          << "[manager][batch, blocks]. manager_num=" << manager_num;
    }

    CHECK_LE(manager_num, static_cast<int32_t>(group_infos.size()))
        << "DSAMetadataBuilderMlu: manager_num(" << manager_num
        << ") exceeds group_infos size(" << group_infos.size()
        << "), cannot align manager/group mapping.";
    const int32_t n_layers = static_cast<int32_t>(caches_info.size());
    const std::vector<int32_t>& ctx_lens = kv_lens_vec;
    int64_t total_tokens = 0;
    for (int32_t len : ctx_lens) {
      total_tokens += static_cast<int64_t>(len);
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
                    proc_bt[m],
                    proc_slots[m]);
      if (group_infos[m].type == DSACacheType::SLIDING_WINDOW) {
        proc_bt[m] = sanitize_swa_table(proc_bt[m],
                                        group_infos[m].block_size,
                                        ctx_lens,
                                        q_lens_vec,
                                        batch_size,
                                        window_size);
      } else {
        proc_bt[m] = sanitize_block_table(proc_bt[m]);
      }
    }

    build_c128_meta(dsa, proc_bt, group_infos, batch_size);

    // Metadata expansion uses CPU accessor paths. Move processed tables and
    // slots back to model device once per forward for downstream MLU kernels.
    const torch::Device target_device =
        positions.defined() ? positions.device() : torch::Device(torch::kCPU);
    if (!target_device.is_cpu()) {
      for (int32_t m = 0; m < manager_num; ++m) {
        if (proc_bt[m].defined() && proc_bt[m].device() != target_device) {
          proc_bt[m] = safe_to(
              proc_bt[m], proc_bt[m].options().device(target_device), true);
        }
        if (proc_slots[m].defined() &&
            proc_slots[m].device() != target_device) {
          proc_slots[m] = safe_to(proc_slots[m],
                                  proc_slots[m].options().device(target_device),
                                  true);
        }
      }
    }

    std::vector<torch::Tensor> decode_slots(manager_num);
    if (is_decode) {
      for (int32_t m = 0; m < manager_num; ++m) {
        const int64_t ratio = static_cast<int64_t>(group_infos[m].ratio);
        if (group_infos[m].type == DSACacheType::TOKEN && ratio > 1) {
          decode_slots[m] = build_decode_slot_mapping(
              proc_bt[m], dsa, ratio, group_infos[m].block_size, target_device);
        }
      }
    }

    // Step 3: expand by layer using group_id.
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
          torch::Tensor slot_mapping = proc_slots[gid];
          if (is_decode && lci[ci].type == DSACacheType::TOKEN &&
              lci[ci].ratio > 1 && decode_slots[gid].defined()) {
            slot_mapping = decode_slots[gid];
          }
          dsa.slot_mappings[lid][ci] = slot_mapping;
        }
      }
    }
  }

  // Attach cache spec pointer
  dsa.caches_info = &caches_info;
}

torch::Tensor DSAMetadataBuilderMlu::expand_blocks_to_slots(
    const torch::Tensor& block_table,
    const DSAGroupInfo& gi,
    const std::vector<int32_t>& ctx_lens,
    int32_t batch_size,
    int64_t total_tokens) {
  const int32_t bs = gi.block_size;
  auto slots = torch::full({total_tokens}, -1, torch::kInt32);
  auto slots_acc = slots.accessor<int32_t, 1>();
  auto bt_acc = block_table.accessor<int32_t, 2>();
  const int32_t max_blocks = block_table.size(1);

  int64_t start_idx = 0;
  for (int32_t seq = 0; seq < batch_size; ++seq) {
    int64_t token_len = ctx_lens[seq];
    int64_t slot_num = compute_slot_num(gi, token_len);

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

int64_t DSAMetadataBuilderMlu::compute_slot_num(const DSAGroupInfo& gi,
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

void DSAMetadataBuilderMlu::process_group(const torch::Tensor& raw_bt,
                                          const DSAGroupInfo& gi,
                                          const std::vector<int32_t>& ctx_lens,
                                          const std::vector<int32_t>& q_lens,
                                          int32_t batch_size,
                                          int64_t total_tokens,
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
                        out_bt,
                        out_slots);
  } else if (gi.type == DSACacheType::SLIDING_WINDOW) {
    process_swa_group(
        raw_bt, gi.block_size, ctx_lens, q_lens, batch_size, out_bt, out_slots);
  } else {
    torch::Tensor raw_slots =
        expand_blocks_to_slots(raw_bt, gi, ctx_lens, batch_size, total_tokens);
    out_slots = raw_slots.clamp_min(0);
    out_bt = raw_bt;
  }
}

void DSAMetadataBuilderMlu::process_token_group(
    const torch::Tensor& raw_bt,
    int32_t ratio,
    int32_t block_size,
    const std::vector<int32_t>& ctx_lens,
    const std::vector<int32_t>& q_lens,
    int32_t batch_size,
    int64_t total_tokens,
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
  for (const int32_t q_len : q_lens) {
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
  auto out_slots_tensor = torch::full({committed_rows}, -1, raw_bt.options());
  auto out_slots_acc = out_slots_tensor.accessor<int32_t, 1>();
  auto raw_bt_acc = raw_bt.accessor<int32_t, 2>();
  const int64_t max_blocks = raw_bt.size(1);
  const int64_t block_size_i64 = static_cast<int64_t>(block_size);

  auto slot_for_compressed_index = [&](int32_t seq,
                                       int64_t compressed_idx) -> int32_t {
    if (max_blocks <= 0) {
      return -1;
    }
    const int64_t block_idx = compressed_idx / block_size_i64;
    if (block_idx >= max_blocks) {
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
  out_bt = raw_bt;  // keep original right-padded block_tables
}

void DSAMetadataBuilderMlu::process_swa_group(
    const torch::Tensor& raw_bt,
    int32_t block_size,
    const std::vector<int32_t>& ctx_lens,
    const std::vector<int32_t>& q_lens,
    int32_t batch_size,
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
  CHECK_GE(raw_bt.size(0), batch_size)
      << "process_swa_group requires raw_bt rows >= batch_size, got "
      << raw_bt.size(0) << " vs " << batch_size;

  int64_t query_total_tokens = 0;
  for (int32_t seq = 0; seq < batch_size; ++seq) {
    query_total_tokens += std::clamp<int64_t>(
        static_cast<int64_t>(q_lens[seq]), 0, ctx_lens[seq]);
  }

  // SWA cache writes only the tokens produced by the current forward step.
  // Decode has one kv row; prefix slot 0 would incorrectly overwrite cache row
  // 0.
  auto out_slots_tensor =
      torch::full({query_total_tokens}, -1, raw_bt.options());
  auto out_slots_acc = out_slots_tensor.accessor<int32_t, 1>();
  auto raw_bt_acc = raw_bt.accessor<int32_t, 2>();
  const int64_t current_cols = raw_bt.size(1);
  const int64_t block_size_i64 = static_cast<int64_t>(block_size);

  auto slot_for_position = [&](int32_t seq, int64_t pos) -> int32_t {
    if (current_cols <= 0) {
      return -1;
    }
    const int64_t block_idx = pos / block_size_i64;
    if (block_idx >= current_cols) {
      return -1;
    }
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
    const int64_t q_start = ctx_len - q_len;
    for (int64_t i = 0; i < q_len; ++i) {
      out_slots_acc[write_idx++] = slot_for_position(seq, q_start + i);
    }
  }

  out_slots = out_slots_tensor;

  int32_t max_dst_len = 0;
  for (int32_t s = 0; s < batch_size; ++s) {
    const int32_t dst_len = (ctx_lens[s] + block_size - 1) / block_size;
    max_dst_len = std::max(max_dst_len, dst_len);
  }
  max_dst_len = std::max(max_dst_len, static_cast<int32_t>(current_cols));

  // To adapt to the fused_compress_single_kv and fused_compress_multi_kv
  // operators, the input cache tensor must have a shape of [bs, compress_len,
  // dim]. So the max_dst_len can not be less than C128 compress_len.
  int32_t min_compress_len = static_cast<int32_t>(kC128Ratio / block_size);
  max_dst_len = std::max(max_dst_len, min_compress_len);

  if (current_cols >= max_dst_len && raw_bt.size(0) == batch_size) {
    out_bt = raw_bt;
    return;
  }

  torch::Tensor new_bt =
      torch::full({batch_size, max_dst_len}, -1, raw_bt.options());
  auto new_acc = new_bt.accessor<int32_t, 2>();
  auto old_acc = raw_bt.accessor<int32_t, 2>();

  for (int32_t s = 0; s < batch_size; ++s) {
    for (int64_t col = 0; col < current_cols; ++col) {
      new_acc[s][col] = old_acc[s][col];
    }
  }
  out_bt = new_bt;
}

void DSAMetadataBuilderMlu::build_c128_meta(
    DSAMetadata& dsa_metadata,
    const std::vector<torch::Tensor>& proc_bt,
    const std::vector<DSAGroupInfo>& group_infos,
    int32_t batch_size) {
  const int32_t manager_num = static_cast<int32_t>(proc_bt.size());
  int32_t c128_gid = -1;
  for (int32_t gid = 0; gid < static_cast<int32_t>(group_infos.size()); ++gid) {
    if (group_infos[gid].type == DSACacheType::TOKEN &&
        group_infos[gid].ratio == kC128Ratio) {
      c128_gid = gid;
      break;
    }
  }
  if (c128_gid < 0) {
    return;
  }

  CHECK_LT(c128_gid, manager_num)
      << "DSAMetadataBuilderMlu: c128 group id exceeds processed block tables.";
  const DSAGroupInfo& c128_group = group_infos[c128_gid];
  const torch::Tensor& block_table = proc_bt[c128_gid];
  const int32_t block_size = c128_group.block_size;

  const int64_t total_tokens = dsa_metadata.query_start_offsets.back();
  int64_t max_context_len = 0;
  for (int32_t seq_idx = 0; seq_idx < batch_size; ++seq_idx) {
    const int64_t q_begin = dsa_metadata.query_start_offsets[seq_idx];
    const int64_t q_end = dsa_metadata.query_start_offsets[seq_idx + 1];
    const int64_t q_len = q_end - q_begin;
    if (q_len > 0) {
      const int64_t last_context_len =
          (dsa_metadata.start_pos_vec[seq_idx] + q_len) / kC128Ratio;
      max_context_len = std::max(max_context_len, last_context_len);
    }
  }

  const torch::TensorOptions int_options =
      torch::TensorOptions().dtype(torch::kInt32);
  const int64_t table_cols =
      std::max<int64_t>((max_context_len + block_size - 1) / block_size, 1);
  torch::Tensor table =
      torch::full({total_tokens, table_cols}, -1, int_options);
  torch::Tensor src =
      block_table
          .to(torch::TensorOptions().device(torch::kCPU).dtype(torch::kInt32))
          .contiguous();
  auto table_acc = table.accessor<int32_t, 2>();
  auto src_acc = src.accessor<int32_t, 2>();

  std::vector<int32_t> context_lens;
  context_lens.reserve(static_cast<size_t>(total_tokens));
  int64_t row = 0;
  for (int32_t seq_idx = 0; seq_idx < batch_size; ++seq_idx) {
    const int64_t q_len = dsa_metadata.query_start_offsets[seq_idx + 1] -
                          dsa_metadata.query_start_offsets[seq_idx];
    for (int64_t token_idx = 0; token_idx < q_len; ++token_idx) {
      const int64_t context_len =
          (dsa_metadata.start_pos_vec[seq_idx] + token_idx + 1) / kC128Ratio;
      context_lens.emplace_back(static_cast<int32_t>(context_len));
      const int64_t blocks = (context_len + block_size - 1) / block_size;
      const int64_t cols = std::min<int64_t>(blocks, src.size(1));
      for (int64_t col = 0; col < cols; ++col) {
        table_acc[row][col] = src_acc[seq_idx][col];
      }
      ++row;
    }
  }
  dsa_metadata.c128_attn_metadata.context_lens =
      torch::tensor(context_lens, int_options);
  dsa_metadata.c128_attn_metadata.max_context_len = max_context_len;
  dsa_metadata.c128_attn_metadata.block_table_for_attn = table;
}

void DSAMetadataBuilderMlu::build_seq_lengths(
    const AttentionMetadata& attn_metadata,
    int32_t batch_size,
    DSAMetadata& dsa_metadata,
    std::vector<int32_t>& q_lens_vec,
    std::vector<int32_t>& kv_lens_vec) {
  torch::Tensor q_cu_lens;
  torch::Tensor kv_cu_lens;
  torch::Tensor q_lens;
  torch::Tensor kv_lens;

  q_lens_vec = tensor_to_vec(attn_metadata.q_seq_lens, "q_seq_lens");
  kv_lens_vec = tensor_to_vec(attn_metadata.kv_seq_lens, "kv_seq_lens");

  q_cu_lens = attn_metadata.q_cu_seq_lens.to(torch::kInt32).contiguous();
  kv_cu_lens = attn_metadata.kv_cu_seq_lens.to(torch::kInt32).contiguous();
  q_lens = attn_metadata.q_seq_lens.to(torch::kInt32).contiguous();
  kv_lens = attn_metadata.kv_seq_lens.to(torch::kInt32).contiguous();

  std::vector<int32_t> index_c4_lens_vec;
  index_c4_lens_vec.reserve(static_cast<size_t>(batch_size));
  dsa_metadata.index_total_c4_len = 0;
  dsa_metadata.index_max_c4_len = 0;
  for (int32_t seq_idx = 0; seq_idx < batch_size; ++seq_idx) {
    const int64_t q_len = static_cast<int64_t>(q_lens_vec[seq_idx]);
    const int64_t kv_len = static_cast<int64_t>(kv_lens_vec[seq_idx]);
    CHECK_GE(q_len, 0) << "DSAMetadataBuilderMlu: q_len must be non-negative.";
    CHECK_GE(kv_len, q_len)
        << "DSAMetadataBuilderMlu: kv_len must be >= q_len.";
    const int64_t c4_len = kv_len / kIndexC4Ratio;
    dsa_metadata.index_total_c4_len += c4_len;
    dsa_metadata.index_max_c4_len =
        std::max(dsa_metadata.index_max_c4_len, c4_len);
    index_c4_lens_vec.emplace_back(static_cast<int32_t>(c4_len));
  }

  torch::Tensor index_c4_lens = torch::tensor(index_c4_lens_vec, torch::kInt32);

  dsa_metadata.q_cu_seq_lens = q_cu_lens;
  dsa_metadata.kv_cu_seq_lens = kv_cu_lens;
  dsa_metadata.q_seq_lens = q_lens;
  dsa_metadata.kv_seq_lens = kv_lens;
  dsa_metadata.index_c4_seq_lens = index_c4_lens;
  dsa_metadata.seq_lens = kv_lens;

  dsa_metadata.query_start_offsets.clear();
  dsa_metadata.query_start_offsets.reserve(static_cast<size_t>(batch_size) + 1);
  dsa_metadata.query_start_offsets.emplace_back(0);
  dsa_metadata.start_pos_vec.clear();
  dsa_metadata.start_pos_vec.reserve(static_cast<size_t>(batch_size));
  int64_t query_offset = 0;
  for (int32_t seq_idx = 0; seq_idx < batch_size; ++seq_idx) {
    const int64_t q_len = static_cast<int64_t>(q_lens_vec[seq_idx]);
    const int64_t kv_len = static_cast<int64_t>(kv_lens_vec[seq_idx]);
    query_offset += q_len;
    dsa_metadata.query_start_offsets.emplace_back(query_offset);
    dsa_metadata.start_pos_vec.emplace_back(kv_len - q_len);
  }
}

void DSAMetadataBuilderMlu::build_positions(const ModelInputParams& params,
                                            int32_t batch_size,
                                            DSAMetadata& dsa_metadata) {
  (void)params;
  if (!dsa_metadata.input_positions.defined()) {
    return;
  }

  const torch::Tensor input_positions = dsa_metadata.input_positions;
  const int64_t num_tokens = input_positions.size(0);

  // C4 / C128 compressed positions share the same construction; only the
  // compression ratio differs.
  build_compressed_pad_positions(input_positions,
                                 /*ratio=*/kIndexC4Ratio,
                                 batch_size,
                                 num_tokens,
                                 dsa_metadata.c4_pad_positions);
  build_compressed_pad_positions(input_positions,
                                 /*ratio=*/kC128Ratio,
                                 batch_size,
                                 num_tokens,
                                 dsa_metadata.c128_pad_positions);
}

}  // namespace xllm::layer

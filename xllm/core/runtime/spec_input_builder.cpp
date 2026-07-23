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

#include "spec_input_builder.h"

#include <glog/logging.h>

#include <algorithm>
#include <limits>

#include "framework/model/model_input_params.h"
#include "framework/sampling/sampling_params.h"
#include "runtime/forward_params.h"
#include "util/tensor_helper.h"

namespace xllm::specBuilder {

namespace {

// Builds cumulative seq-lens layout: [0, l0, l0+l1, ...].
void push_cumsum(std::vector<int32_t>& vec, int32_t len) {
  if (vec.empty()) {
    vec.emplace_back(0);
  }
  vec.emplace_back(vec.back() + len);
}

Slice<int32_t> tensor_slice(const torch::Tensor& tensor) {
  return {tensor.data_ptr<int32_t>(), static_cast<size_t>(tensor.numel())};
}

Slice<int32_t> get_token_ids(const ForwardInput& input) {
  return tensor_slice(input.token_ids_host);
}

Slice<int32_t> get_positions(const ForwardInput& input) {
  return tensor_slice(input.positions_host);
}

Slice<int32_t> get_kv_seq_lens(const ForwardInput& input) {
  return input.input_params.attention.host.kv_seq_lens;
}

// Resolves a row token from either input token_ids[seq_id] or row.token_id.
int32_t resolve_row_token_id(const DecodeRowContext& ctx, const RowSpec& row) {
  if (!row.use_input_token) {
    return row.token_id;
  }
  CHECK_LT(static_cast<size_t>(row.seq_id), ctx.token_ids.size())
      << "seq_id out of range for token_ids, seq_id=" << row.seq_id
      << ", token_ids_size=" << ctx.token_ids.size();
  return ctx.token_ids[row.seq_id];
}

Slice<int32_t> get_block_table_slice(const DecodeRowContext& ctx,
                                     int32_t seq_id) {
  CHECK_GE(seq_id, 0) << "invalid seq_id=" << seq_id;
  CHECK_LT(seq_id, ctx.num_sequences)
      << "seq_id out of range for block tables, seq_id=" << seq_id
      << ", num_sequences=" << ctx.num_sequences;
  CHECK_GT(ctx.block_table_stride, 0)
      << "invalid block table row stride=" << ctx.block_table_stride;

  const size_t row_offset =
      static_cast<size_t>(seq_id) * static_cast<size_t>(ctx.block_table_stride);
  CHECK_LE(row_offset + static_cast<size_t>(ctx.block_table_stride),
           ctx.block_tables.size())
      << "block table row out of range, seq_id=" << seq_id
      << ", row_offset=" << row_offset
      << ", row_stride=" << ctx.block_table_stride
      << ", block_tables_size=" << ctx.block_tables.size();
  return {ctx.block_tables.data() + row_offset,
          static_cast<size_t>(ctx.block_table_stride)};
}

template <typename T>
void pad_2d_vector(std::vector<std::vector<T>>& vec, T pad_value) {
  size_t max_col_size = 0;
  for (const std::vector<T>& row : vec) {
    max_col_size = std::max(max_col_size, row.size());
  }

  for (std::vector<T>& row : vec) {
    row.resize(max_col_size, pad_value);
  }
}

torch::Tensor create_flat_2d_tensor(const std::vector<int32_t>& values,
                                    int32_t rows,
                                    int32_t stride) {
  if (rows == 0) {
    return torch::Tensor();
  }
  CHECK_GT(rows, 0) << "invalid rows=" << rows;
  CHECK_GT(stride, 0) << "invalid stride=" << stride;
  CHECK_EQ(values.size(), static_cast<size_t>(rows) * stride)
      << "flat 2D tensor size mismatch, rows=" << rows << ", stride=" << stride
      << ", values_size=" << values.size();
  auto tensor = torch::empty({rows, stride},
                             torch::TensorOptions()
                                 .dtype(torch::kInt)
                                 .device(torch::kCPU)
                                 .pinned_memory(true));
  std::copy(values.begin(), values.end(), tensor.data_ptr<int32_t>());
  return tensor;
}

void fill_multi_block_table_slices(DecodeRowContext& ctx) {
  ctx.model_managed_multiblock = !ctx.multi_block_tables_owner.empty();
  ctx.multi_block_tables.resize(ctx.multi_block_tables_owner.size());
  for (size_t m = 0; m < ctx.multi_block_tables_owner.size(); ++m) {
    const torch::Tensor& manager_table = ctx.multi_block_tables_owner[m];
    CHECK(manager_table.defined())
        << "multi_block_tables[" << m << "] is undefined";
    CHECK_EQ(manager_table.dim(), 2)
        << "multi_block_tables[" << m << "] must be 2D, got "
        << manager_table.sizes();
    CHECK_LE(ctx.num_sequences, manager_table.size(0))
        << "num_sequences exceeds multi_block_tables[" << m
        << "] rows, num_sequences=" << ctx.num_sequences
        << ", rows=" << manager_table.size(0);
    ctx.multi_block_tables[m].reserve(static_cast<size_t>(ctx.num_sequences));
    for (int32_t seq_id = 0; seq_id < ctx.num_sequences; ++seq_id) {
      torch::Tensor row = manager_table[seq_id];
      ctx.multi_block_tables[m].emplace_back(row.data_ptr<int32_t>(),
                                             static_cast<size_t>(row.numel()));
    }
  }
}

}  // namespace

MtpTopkStatePtr select_mtp_topk_state_for_next_step(
    const MtpTopkStatePtr& state,
    const SamplingParameters& sampling_params) {
  if (state == nullptr || !sampling_params.selected_token_idxes.defined()) {
    return state;
  }
  const torch::Tensor& selected_idxes = sampling_params.selected_token_idxes;
  if (selected_idxes.numel() == 0 ||
      state->num_rows() == selected_idxes.numel()) {
    return state;
  }
  if (selected_idxes.device().is_cpu()) {
    CHECK_GT(state->num_rows(), selected_idxes.max().item<int64_t>())
        << "MTP selected top-k index exceeds top-k rows.";
  }
  torch::Tensor index =
      selected_idxes.to(torch::dtype(torch::kLong).device(state->device()))
          .contiguous();
  return state->index_select_rows(index);
}

int32_t calc_slot_id(int32_t position,
                     const Slice<int32_t>& block_table_slice,
                     int32_t block_size) {
  CHECK_GT(block_size, 0) << "invalid block_size=" << block_size;
  CHECK_GE(position, 0) << "invalid position=" << position;
  const int32_t block_idx = position / block_size;
  CHECK_LT(static_cast<size_t>(block_idx), block_table_slice.size())
      << "block table index out of range, block_idx=" << block_idx
      << ", block_table_size=" << block_table_slice.size()
      << ", position=" << position << ", block_size=" << block_size;
  const int32_t block_id = block_table_slice[block_idx];
  CHECK_GE(block_id, 0) << "invalid block_id=" << block_id;
  const int32_t block_offset = position % block_size;
  return block_id * block_size + block_offset;
}

int32_t calc_kv_len(const Slice<int32_t>& kv_seq_lens_slice,
                    int32_t seq_id,
                    int32_t offset) {
  CHECK_GE(seq_id, 0) << "invalid seq_id=" << seq_id;
#if defined(USE_NPU)
  CHECK_LT(static_cast<size_t>(seq_id), kv_seq_lens_slice.size())
      << "seq_id out of range, seq_id=" << seq_id
      << ", kv_seq_lens_size=" << kv_seq_lens_slice.size();
  return kv_seq_lens_slice[seq_id] + offset;
#else
  CHECK_LT(static_cast<size_t>(seq_id + 1), kv_seq_lens_slice.size())
      << "seq_id out of range for cumulative layout, seq_id=" << seq_id
      << ", kv_seq_lens_size=" << kv_seq_lens_slice.size();
  return kv_seq_lens_slice[seq_id + 1] - kv_seq_lens_slice[seq_id] + offset;
#endif
}

void append_seq_len_by_layout(std::vector<int32_t>& vec, int32_t len) {
#if defined(USE_NPU)
  vec.emplace_back(len);
#else
  push_cumsum(vec, len);
#endif
}

void append_q_seq_len(std::vector<int32_t>& q_seq_lens,
                      std::vector<int32_t>& q_cu_seq_lens,
                      int32_t len) {
  append_seq_len_by_layout(q_seq_lens, len);
  q_cu_seq_lens.emplace_back(
      (q_cu_seq_lens.empty() ? 0 : q_cu_seq_lens.back()) + len);
}

void update_kv_seq_lens_and_max(std::vector<int32_t>& kv_seq_lens_vec,
                                int32_t kv_len,
                                int32_t& kv_max_seq_len) {
  if (kv_len > kv_max_seq_len) {
    kv_max_seq_len = kv_len;
  }
  append_seq_len_by_layout(kv_seq_lens_vec, kv_len);
}

DecodeRowContext make_decode_row_context(const ForwardInput& input) {
  DecodeRowContext ctx;
  ctx.num_sequences = input.input_params.meta.num_sequences;
  CHECK_GE(ctx.num_sequences, 0) << "invalid num_sequences";

  if (input.token_ids_host.defined()) {
    ctx.token_ids = get_token_ids(input);
  }
  CHECK(input.positions_host.defined())
      << "positions_host must be defined for decode row build";
  ctx.positions = get_positions(input);
  CHECK_GE(static_cast<int32_t>(ctx.positions.size()), ctx.num_sequences)
      << "positions size is smaller than num_sequences, positions_size="
      << ctx.positions.size() << ", num_sequences=" << ctx.num_sequences;

  ctx.kv_seq_lens = get_kv_seq_lens(input);
  if (!input.input_params.multi_block_tables.empty()) {
    ctx.multi_block_tables_owner.reserve(
        input.input_params.multi_block_tables.size());
    for (const torch::Tensor& block_table :
         input.input_params.multi_block_tables) {
      torch::Tensor cpu_block_table = block_table.device().is_cpu()
                                          ? block_table
                                          : block_table.to(torch::kCPU);
      ctx.multi_block_tables_owner.emplace_back(cpu_block_table.contiguous());
    }
    fill_multi_block_table_slices(ctx);
    return ctx;
  }

  CHECK(input.input_params.attention.host.block_tables.defined())
      << "host block_tables must be defined for decode row build";
  ctx.block_tables_owner =
      input.input_params.attention.host.block_tables.contiguous();
  CHECK_EQ(ctx.block_tables_owner.dim(), 2)
      << "block_tables must be 2D, got " << ctx.block_tables_owner.sizes();
  CHECK_LE(ctx.num_sequences, ctx.block_tables_owner.size(0))
      << "num_sequences exceeds block table rows, num_sequences="
      << ctx.num_sequences
      << ", block_table_rows=" << ctx.block_tables_owner.size(0);
  ctx.block_table_stride = static_cast<int32_t>(ctx.block_tables_owner.size(1));
  CHECK_GT(ctx.block_table_stride, 0)
      << "invalid block table row stride=" << ctx.block_table_stride;
  ctx.block_tables = tensor_slice(ctx.block_tables_owner);
  return ctx;
}

void append_decode_row(const DecodeRowContext& ctx,
                       const RowSpec& row,
                       int32_t block_size,
                       DecodeBuildBuffers& buf) {
  CHECK_GE(row.seq_id, 0);
  CHECK_LT(row.seq_id, ctx.num_sequences);
  CHECK_LT(static_cast<size_t>(row.seq_id), ctx.positions.size());
  const int32_t new_position = ctx.positions[row.seq_id] + row.position_offset;
  CHECK_GE(new_position, 0) << "invalid decode position";

  // All decode paths can toggle which fields are emitted, so one row builder
  // can serve draft/validate/first-decode/update-last-step scenarios.
  if (row.append_token) {
    buf.out_token_ids.emplace_back(resolve_row_token_id(ctx, row));
  }
  buf.out_positions.emplace_back(new_position);
  if (ctx.model_managed_multiblock) {
    buf.out_new_cache_slots.emplace_back(0);
    if (row.append_block_table) {
      if (buf.out_multi_block_tables.size() < ctx.multi_block_tables.size()) {
        buf.out_multi_block_tables.resize(ctx.multi_block_tables.size());
      }
      for (size_t m = 0; m < ctx.multi_block_tables.size(); ++m) {
        CHECK_LT(static_cast<size_t>(row.seq_id),
                 ctx.multi_block_tables[m].size());
        const Slice<int32_t>& block_table_slice =
            ctx.multi_block_tables[m][row.seq_id];
        buf.out_multi_block_tables[m].emplace_back(block_table_slice.begin(),
                                                   block_table_slice.end());
      }
    }
  } else {
    const Slice<int32_t> block_table_slice =
        get_block_table_slice(ctx, row.seq_id);
    buf.out_new_cache_slots.emplace_back(
        calc_slot_id(new_position, block_table_slice, block_size));
    if (row.append_block_table) {
      if (buf.out_block_table_stride == 0) {
        buf.out_block_table_stride =
            static_cast<int32_t>(block_table_slice.size());
      }
      CHECK_EQ(buf.out_block_table_stride,
               static_cast<int32_t>(block_table_slice.size()))
          << "block table stride mismatch";
      buf.out_block_tables.insert(buf.out_block_tables.end(),
                                  block_table_slice.begin(),
                                  block_table_slice.end());
      ++buf.out_block_table_rows;
    }
  }

  if (row.append_kv_len) {
    int32_t kv_len =
        calc_kv_len(ctx.kv_seq_lens, row.seq_id, row.position_offset);
    update_kv_seq_lens_and_max(
        buf.out_kv_seq_lens, kv_len, buf.meta.kv_max_seq_len);
  }
  if (row.append_q_len_one) {
    append_q_seq_len(buf.out_q_seq_lens, buf.out_q_cu_seq_lens, 1);
  }
}

TokenWithOffset resolve_token_with_position_offset(
    int32_t input_token_id,
    int32_t seq_id,
    const Slice<int64_t>& last_step_tokens,
    int32_t last_step_decode_num) {
  CHECK_GT(last_step_decode_num, 0)
      << "invalid last_step_decode_num=" << last_step_decode_num;
  if (input_token_id >= 0) {
    TokenWithOffset direct;
    direct.token_id = input_token_id;
    direct.position_offset = 0;
    return direct;
  }

  const int32_t placeholder_idx = -input_token_id - 1;
  CHECK_GE(placeholder_idx, 0)
      << "invalid placeholder token id=" << input_token_id
      << ", seq_id=" << seq_id;
  const int32_t base_idx = placeholder_idx * last_step_decode_num;
  CHECK_LE(base_idx + last_step_decode_num,
           static_cast<int32_t>(last_step_tokens.size()))
      << "last_step_tokens out of range, seq_id=" << seq_id
      << ", placeholder_idx=" << placeholder_idx
      << ", last_step_decode_num=" << last_step_decode_num
      << ", last_step_tokens_size=" << last_step_tokens.size();

  TokenWithOffset resolved;
  resolved.position_offset = -1;
  for (int32_t i = 0; i < last_step_decode_num; ++i) {
    const int64_t candidate = last_step_tokens[base_idx + i];
    if (candidate >= 0) {
      CHECK_LE(candidate,
               static_cast<int64_t>(std::numeric_limits<int32_t>::max()))
          << "token id overflow, seq_id=" << seq_id
          << ", candidate=" << candidate;
      resolved.token_id = static_cast<int32_t>(candidate);
      resolved.position_offset += 1;
    }
  }
  return resolved;
}

void append_decode_row_from_last_step(const DecodeRowContext& ctx,
                                      int32_t seq_id,
                                      int32_t input_token_id,
                                      const Slice<int64_t>& last_step_tokens,
                                      int32_t last_step_decode_num,
                                      int32_t block_size,
                                      DecodeBuildBuffers& buf) {
  // Placeholder tokens (-1/-2/...) are resolved from last-step outputs first,
  // then appended via the same row builder used by all decode paths.
  const TokenWithOffset resolved = resolve_token_with_position_offset(
      input_token_id, seq_id, last_step_tokens, last_step_decode_num);

  RowSpec row;
  row.seq_id = seq_id;
  row.token_id = resolved.token_id;
  row.position_offset = resolved.position_offset;
  append_decode_row(ctx, row, block_size, buf);
}

torch::Tensor build_q_cu_seq_lens_tensor(const ModelInputParams& params,
                                         torch::Device device,
                                         bool include_leading_zero) {
  CHECK_EQ(params.attention.host.q_seq_lens.empty(),
           params.attention.host.q_cu_seq_lens.empty())
      << "q_seq_lens and q_cu_seq_lens must be provided together";
  if (!include_leading_zero) {
    return torch::tensor(params.attention.host.q_cu_seq_lens,
                         torch::dtype(torch::kInt).device(device));
  }
  std::vector<int32_t> q_cu_seq_lens_vec;
  q_cu_seq_lens_vec.reserve(params.meta.num_sequences + 1);
  q_cu_seq_lens_vec.emplace_back(0);
  for (int32_t i = 0; i < params.meta.num_sequences; ++i) {
    q_cu_seq_lens_vec.emplace_back(q_cu_seq_lens_vec.back() +
                                   params.get_q_seq_len(i));
  }
  return torch::tensor(q_cu_seq_lens_vec,
                       torch::dtype(torch::kInt).device(device));
}

void update_input_params(ModelInputParams& input_params,
                         DecodeBuildBuffers& buf,
                         int32_t q_max_seq_len,
                         std::vector<int32_t> q_seq_lens_vec,
                         std::vector<int32_t> q_cu_seq_lens_vec,
                         int32_t kv_max_seq_len,
                         std::vector<int32_t> kv_seq_lens_vec,
                         bool update_block_tables) {
  CHECK_EQ(q_seq_lens_vec.empty(), q_cu_seq_lens_vec.empty())
      << "q_seq_lens and q_cu_seq_lens must be provided together";
  input_params.meta.q_max_seq_len = q_max_seq_len;
  input_params.attention.host.q_seq_lens = std::move(q_seq_lens_vec);
  input_params.attention.host.q_cu_seq_lens = std::move(q_cu_seq_lens_vec);
  input_params.meta.kv_max_seq_len = kv_max_seq_len;
  input_params.attention.host.kv_seq_lens = std::move(kv_seq_lens_vec);
  input_params.attention.host.new_cache_slots =
      std::move(buf.out_new_cache_slots);
  if (update_block_tables) {
    if (!buf.out_multi_block_tables.empty()) {
      input_params.multi_block_tables.clear();
      input_params.multi_block_tables.reserve(
          buf.out_multi_block_tables.size());
      for (std::vector<std::vector<int32_t>>& manager_tables :
           buf.out_multi_block_tables) {
        pad_2d_vector(manager_tables, /*pad_value=*/-1);
        input_params.multi_block_tables.emplace_back(
            create_2d_tensor(manager_tables, torch::kInt));
      }
      input_params.attention.host.block_tables = torch::Tensor();
    } else {
      input_params.attention.host.block_tables =
          create_flat_2d_tensor(buf.out_block_tables,
                                buf.out_block_table_rows,
                                buf.out_block_table_stride);
      input_params.multi_block_tables.clear();
    }
  }
}

torch::Tensor make_cpu_int_tensor(const std::vector<int32_t>& values) {
  return torch::tensor(values,
                       torch::TensorOptions()
                           .dtype(torch::kInt)
                           .device(torch::kCPU)
                           .pinned_memory(true));
}

void set_token_position_tensors(ForwardInput& input,
                                const std::vector<int32_t>& token_ids,
                                const std::vector<int32_t>& positions,
                                const torch::TensorOptions& token_options,
                                const torch::TensorOptions& position_options) {
  input.device_tensors_ready = false;
  input.token_ids_host = make_cpu_int_tensor(token_ids);
  input.positions_host = make_cpu_int_tensor(positions);
  input.token_ids =
      safe_to(input.token_ids_host, token_options, /*non_blocking=*/true);
  input.positions =
      safe_to(input.positions_host, position_options, /*non_blocking=*/true);
  input.device_tensors_ready = true;
}

namespace draftProbs {

namespace {

torch::Tensor extract_selected_probs(const torch::Tensor& draft_probs,
                                     const torch::Tensor& draft_token_ids) {
  CHECK(draft_probs.defined()) << "draft_probs must be defined";
  CHECK(draft_token_ids.defined()) << "draft_token_ids must be defined";

  if (draft_probs.dim() == 1) {
    return draft_probs;
  }

  if (draft_probs.dim() == 2) {
    CHECK_EQ(draft_probs.size(0), draft_token_ids.numel())
        << "draft_probs batch size mismatch";
    if (draft_probs.size(1) == 1) {
      return draft_probs.squeeze(-1);
    }
    auto ids = draft_token_ids.view({-1, 1}).to(torch::kLong);
    return draft_probs.gather(/*dim=*/-1, ids).squeeze(-1);
  }

  CHECK(false) << "draft_probs must be [batch], [batch,1] or [batch,vocab]";
  return torch::Tensor();
}

// Scatter selected-only probs into a dense last-vocab-dim tensor. `ids` and
// `selected_probs` share shape [..., width]; the result is [..., width, vocab],
// with each selected id's prob placed at its vocab slot and zeros elsewhere.
torch::Tensor scatter_selected_to_dense(const torch::Tensor& ids,
                                        const torch::Tensor& selected_probs,
                                        int32_t vocab_size) {
  CHECK_GT(vocab_size, 0) << "vocab_size must be > 0 for dense draft probs";
  std::vector<int64_t> dense_shape(ids.sizes().begin(), ids.sizes().end());
  dense_shape.emplace_back(vocab_size);
  torch::Tensor dense_probs =
      torch::zeros(dense_shape, selected_probs.options());
  dense_probs.scatter_(
      /*dim=*/-1, ids.unsqueeze(-1), selected_probs.unsqueeze(-1));
  return dense_probs;
}

}  // namespace

torch::Tensor compress_for_cache(const torch::Tensor& draft_probs,
                                 const torch::Tensor& draft_token_ids) {
  return extract_selected_probs(draft_probs, draft_token_ids);
}

std::pair<torch::Tensor, torch::Tensor> build_validate_tensors(
    const std::vector<torch::Tensor>& draft_token_ids_steps,
    const std::vector<torch::Tensor>& draft_probs_steps,
    int32_t batch_size,
    int32_t vocab_size,
    bool enable_opt_validate_probs,
    bool draft_probs_required) {
  CHECK_GT(batch_size, 0) << "batch_size must be > 0";
  CHECK_GT(vocab_size, 0) << "vocab_size must be > 0";
  CHECK_EQ(draft_token_ids_steps.size(), draft_probs_steps.size())
      << "draft steps mismatch";
  CHECK(!draft_token_ids_steps.empty()) << "draft steps must not be empty";

  std::vector<torch::Tensor> token_ids_vec;
  std::vector<torch::Tensor> probs_vec;
  token_ids_vec.reserve(draft_token_ids_steps.size());
  probs_vec.reserve(draft_probs_steps.size());

  for (size_t i = 0; i < draft_token_ids_steps.size(); ++i) {
    auto draft_token_ids =
        draft_token_ids_steps[i].view({batch_size, 1}).to(torch::kLong);
    token_ids_vec.emplace_back(draft_token_ids);
    if (!draft_probs_required) {
      continue;
    }

    torch::Tensor selected_probs =
        extract_selected_probs(draft_probs_steps[i], draft_token_ids)
            .view({batch_size, 1});
    if (enable_opt_validate_probs) {
      probs_vec.emplace_back(selected_probs);
    } else {
      probs_vec.emplace_back(scatter_selected_to_dense(
          draft_token_ids, selected_probs, vocab_size));
    }
  }

  auto draft_token_ids = torch::cat(token_ids_vec, /*dim=*/1);
  if (!draft_probs_required) {
    return {draft_token_ids, torch::Tensor()};
  }
  auto draft_probs = torch::cat(probs_vec, /*dim=*/1);
  return {draft_token_ids, draft_probs};
}

std::pair<torch::Tensor, torch::Tensor> build_validate_tensors_from_block(
    const torch::Tensor& token_ids_block,
    const torch::Tensor& probs_block,
    int32_t vocab_size,
    bool enable_opt_validate_probs) {
  CHECK(token_ids_block.defined()) << "token_ids_block must be defined";
  CHECK(probs_block.defined()) << "probs_block must be defined";
  CHECK_EQ(token_ids_block.dim(), 2)
      << "token_ids_block must be [batch, n_speculative_tokens], got "
      << token_ids_block.sizes();
  CHECK_EQ(probs_block.dim(), 2)
      << "probs_block must be selected-only [batch, n_speculative_tokens], got "
      << probs_block.sizes();
  CHECK_EQ(probs_block.size(0), token_ids_block.size(0))
      << "probs/token block batch mismatch";
  CHECK_EQ(probs_block.size(1), token_ids_block.size(1))
      << "probs/token block width mismatch";

  auto draft_token_ids = token_ids_block.to(torch::kLong);
  if (enable_opt_validate_probs) {
    // The draft block is already stacked as [batch, n_speculative_tokens]; pass
    // the selected-only probs straight to the rejection sampler.
    return {draft_token_ids, probs_block};
  }

  // Default path: scatter the selected probs into a dense
  // [batch, n_speculative_tokens, vocab_size] tensor, matching MTP's
  // build_validate_tensors output so the shared rejection sampler (and its
  // fused kernel) always receives dense draft_probs.
  return {draft_token_ids,
          scatter_selected_to_dense(draft_token_ids, probs_block, vocab_size)};
}

}  // namespace draftProbs

}  // namespace xllm::specBuilder

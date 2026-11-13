/* Copyright 2026 The xLLM Authors.

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

#include <glog/logging.h>
#include <torch/torch.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <optional>

#include "core/kernels/npu/tilelang/dispatch_registry.h"
#include "core/kernels/npu/tilelang/mtp_prepare_next_draft.h"

#ifndef XLLM_TL_MTP_PREPARE_NEXT_DRAFT_REGISTRY_INC
#error "XLLM_TL_MTP_PREPARE_NEXT_DRAFT_REGISTRY_INC is not defined"
#endif

namespace xllm::kernel::npu {
namespace tilelang {
namespace {
#include XLLM_TL_MTP_PREPARE_NEXT_DRAFT_REGISTRY_INC
}  // namespace
}  // namespace tilelang

namespace {
constexpr int64_t kMaxOffset = std::numeric_limits<int32_t>::max();

bool is_npu_tensor(const torch::Tensor& tensor) {
  return tensor.defined() && tensor.numel() > 0 &&
         tensor.device().type() == c10::DeviceType::PrivateUse1;
}

bool is_supported_mtp_prepare_input(const torch::Tensor& accepted_tokens,
                                    const torch::Tensor& accepted_embeddings,
                                    const torch::Tensor& embedding_placeholder,
                                    const torch::Tensor& base_positions,
                                    const torch::Tensor& base_kv_seq_lens,
                                    const torch::Tensor& block_tables,
                                    int64_t block_size) {
  if (!is_npu_tensor(accepted_tokens) || !is_npu_tensor(accepted_embeddings) ||
      !is_npu_tensor(embedding_placeholder) || !is_npu_tensor(base_positions) ||
      !is_npu_tensor(base_kv_seq_lens) || !is_npu_tensor(block_tables) ||
      block_size <= 0 || block_size > kMaxOffset ||
      accepted_tokens.dim() != 2 ||
      (accepted_embeddings.dim() != 2 && accepted_embeddings.dim() != 3) ||
      block_tables.dim() != 2) {
    return false;
  }

  const int64_t batch_size = accepted_tokens.size(0);
  const int64_t speculative_width = accepted_tokens.size(1);
  const int64_t hidden_size = accepted_embeddings.size(-1);
  const torch::Device device = accepted_tokens.device();
  const torch::ScalarType embedding_type = accepted_embeddings.scalar_type();
  return batch_size > 0 && speculative_width > 0 && hidden_size > 0 &&
         batch_size <= kMaxOffset / 2 &&
         speculative_width <= kMaxOffset / batch_size &&
         hidden_size <= kMaxOffset / (batch_size * 2) &&
         accepted_embeddings.numel() <= kMaxOffset &&
         block_tables.numel() <= kMaxOffset &&
         accepted_tokens.scalar_type() == torch::kLong &&
         (embedding_type == torch::kFloat16 ||
          embedding_type == torch::kBFloat16) &&
         embedding_placeholder.scalar_type() == embedding_type &&
         (base_positions.scalar_type() == torch::kInt ||
          base_positions.scalar_type() == torch::kLong) &&
         (base_kv_seq_lens.scalar_type() == torch::kInt ||
          base_kv_seq_lens.scalar_type() == torch::kLong) &&
         (block_tables.scalar_type() == torch::kInt ||
          block_tables.scalar_type() == torch::kLong) &&
         (accepted_embeddings.dim() == 2
              ? accepted_embeddings.size(0) == batch_size * 2
              : (accepted_embeddings.size(0) == batch_size &&
                 accepted_embeddings.size(1) == speculative_width)) &&
         embedding_placeholder.numel() == hidden_size &&
         base_positions.numel() >= batch_size &&
         base_kv_seq_lens.numel() >= batch_size &&
         block_tables.size(0) == batch_size &&
         (hidden_size * accepted_embeddings.element_size()) % 32 == 0 &&
         accepted_tokens.is_contiguous() &&
         accepted_embeddings.is_contiguous() &&
         embedding_placeholder.is_contiguous() &&
         base_positions.is_contiguous() && base_kv_seq_lens.is_contiguous() &&
         block_tables.is_contiguous() &&
         accepted_embeddings.device() == device &&
         embedding_placeholder.device() == device &&
         base_positions.device() == device &&
         base_kv_seq_lens.device() == device && block_tables.device() == device;
}

}  // namespace

std::optional<MtpPrepareNextDraftOutput> try_mtp_prepare_next_draft(
    const torch::Tensor& accepted_tokens,
    const torch::Tensor& accepted_embeddings,
    const torch::Tensor& embedding_placeholder,
    const torch::Tensor& base_positions,
    const torch::Tensor& base_kv_seq_lens,
    const torch::Tensor& block_tables,
    int64_t block_size,
    MtpPrepareNextDraftWorkspace* reusable_workspace,
    const torch::Tensor& shared_block_tables) {
  if (!is_supported_mtp_prepare_input(accepted_tokens,
                                      accepted_embeddings,
                                      embedding_placeholder,
                                      base_positions,
                                      base_kv_seq_lens,
                                      block_tables,
                                      block_size)) {
    return std::nullopt;
  }
  const bool use_shared_table = shared_block_tables.defined();
  if (use_shared_table &&
      (shared_block_tables.device() != accepted_tokens.device() ||
       shared_block_tables.scalar_type() != torch::kInt ||
       shared_block_tables.sizes() != block_tables.sizes() ||
       !shared_block_tables.is_contiguous())) {
    return std::nullopt;
  }

  const int64_t batch_size = accepted_tokens.size(0);
  const int64_t hidden_size = accepted_embeddings.size(-1);
  torch::Tensor position_rows = base_positions;
  if (position_rows.scalar_type() != torch::kInt) {
    position_rows = position_rows.to(torch::kInt);
  }
  torch::Tensor kv_seq_len_rows = base_kv_seq_lens;
  if (kv_seq_len_rows.scalar_type() != torch::kInt) {
    kv_seq_len_rows = kv_seq_len_rows.to(torch::kInt);
  }
  torch::Tensor block_table_input = block_tables;
  if (block_table_input.scalar_type() != torch::kInt) {
    block_table_input = block_table_input.to(torch::kInt);
  }

  MtpPrepareNextDraftOutput local_output;
  MtpPrepareNextDraftOutput& output =
      reusable_workspace == nullptr ? local_output : reusable_workspace->output;
  auto ensure_tensor = [](torch::Tensor& tensor,
                          const torch::IntArrayRef& sizes,
                          const torch::TensorOptions& options) {
    if (!tensor.defined() || tensor.sizes() != sizes ||
        tensor.scalar_type() != options.dtype().toScalarType() ||
        tensor.device() != options.device()) {
      tensor = torch::empty(sizes, options);
    }
  };
  ensure_tensor(output.token_ids,
                {batch_size * 2},
                accepted_tokens.options().dtype(torch::kInt));
  ensure_tensor(output.embeddings,
                {batch_size * 2, hidden_size},
                accepted_embeddings.options());
  const torch::TensorOptions int_options =
      accepted_tokens.options().dtype(torch::kInt);
  ensure_tensor(output.positions, {batch_size * 2}, int_options);
  ensure_tensor(output.kv_seq_lens,
                {accepted_embeddings.dim() == 2 ? batch_size * 2 : batch_size},
                int_options);
  ensure_tensor(output.cache_slots, {batch_size * 2}, int_options);
  if (use_shared_table) {
    ensure_tensor(output.shared_cache_slots, {batch_size * 2}, int_options);
  } else {
    output.shared_cache_slots = torch::Tensor();
  }
  const torch::Tensor& shared_table_input =
      use_shared_table ? shared_block_tables : block_table_input;
  const torch::Tensor& shared_slot_output =
      use_shared_table ? output.shared_cache_slots : output.cache_slots;

  const auto spec = tilelang::make_mtp_prepare_next_draft_specialization(
      tilelang::MtpPrepareNextDraftCompactHidden{
          accepted_embeddings.dim() == 2 ? 1 : 0});
  const auto* entry = tilelang::find_mtp_prepare_next_draft_kernel_entry(spec);
  CHECK(entry != nullptr)
      << tilelang::available_mtp_prepare_next_draft_variant_keys();
  aclrtStream stream =
      c10_npu::getCurrentNPUStream(accepted_tokens.device().index()).stream();
  entry->fn(static_cast<uint8_t*>(accepted_tokens.data_ptr()),
            static_cast<uint8_t*>(accepted_embeddings.data_ptr()),
            static_cast<uint8_t*>(embedding_placeholder.data_ptr()),
            static_cast<uint8_t*>(position_rows.data_ptr()),
            static_cast<uint8_t*>(kv_seq_len_rows.data_ptr()),
            static_cast<uint8_t*>(block_table_input.data_ptr()),
            static_cast<uint8_t*>(shared_table_input.data_ptr()),
            static_cast<uint8_t*>(output.token_ids.data_ptr()),
            static_cast<uint8_t*>(output.embeddings.data_ptr()),
            static_cast<uint8_t*>(output.positions.data_ptr()),
            static_cast<uint8_t*>(output.kv_seq_lens.data_ptr()),
            static_cast<uint8_t*>(output.cache_slots.data_ptr()),
            static_cast<uint8_t*>(shared_slot_output.data_ptr()),
            static_cast<int32_t>(batch_size),
            static_cast<int32_t>(accepted_tokens.size(1)),
            static_cast<int32_t>(hidden_size),
            static_cast<int32_t>(block_tables.size(1)),
            static_cast<int32_t>(block_size),
            static_cast<int32_t>(std::min(batch_size * 2, int64_t{32})),
            static_cast<int32_t>(use_shared_table),
            stream);
  return output;
}

}  // namespace xllm::kernel::npu

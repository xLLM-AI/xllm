/* Copyright 2026 The xLLM Authors. All Rights Reserved. */

#include <glog/logging.h>
#include <torch/torch.h>
#include <torch_npu/csrc/libs/init_npu.h>
#include <torch_npu/torch_npu.h>

#include <cstdint>

#include "acl/acl.h"
#include "aclnn_mtp_prepare_next_draft.h"
#include "core/common/macros.h"
#include "core/kernels/npu/utils.h"
#include "xllm_ops_api.h"

namespace xllm::kernel::npu {
namespace {

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
      block_size <= 0 || accepted_tokens.dim() != 2 ||
      accepted_embeddings.dim() != 3 || block_tables.dim() != 2) {
    return false;
  }

  const int64_t batch_size = accepted_tokens.size(0);
  const int64_t speculative_width = accepted_tokens.size(1);
  const int64_t hidden_size = accepted_embeddings.size(2);
  const torch::Device device = accepted_tokens.device();
  const torch::ScalarType embedding_type = accepted_embeddings.scalar_type();
  return batch_size > 0 && speculative_width > 0 && hidden_size > 0 &&
         accepted_tokens.scalar_type() == torch::kLong &&
         (embedding_type == torch::kFloat16 ||
          embedding_type == torch::kBFloat16) &&
         embedding_placeholder.scalar_type() == embedding_type &&
         base_positions.scalar_type() == torch::kInt &&
         base_kv_seq_lens.scalar_type() == torch::kInt &&
         block_tables.scalar_type() == torch::kInt &&
         accepted_embeddings.size(0) == batch_size &&
         accepted_embeddings.size(1) == speculative_width &&
         embedding_placeholder.numel() == hidden_size &&
         base_positions.numel() >= batch_size &&
         base_kv_seq_lens.numel() >= batch_size &&
         block_tables.size(0) >= batch_size &&
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
    int64_t block_size) {
  if (!is_supported_mtp_prepare_input(accepted_tokens,
                                      accepted_embeddings,
                                      embedding_placeholder,
                                      base_positions,
                                      base_kv_seq_lens,
                                      block_tables,
                                      block_size)) {
    return std::nullopt;
  }

  const int64_t batch_size = accepted_tokens.size(0);
  const int64_t hidden_size = accepted_embeddings.size(2);
  const torch::Tensor position_rows =
      base_positions.flatten().slice(0, 0, batch_size);
  const torch::Tensor kv_seq_len_rows =
      base_kv_seq_lens.flatten().slice(0, 0, batch_size);

  MtpPrepareNextDraftOutput output;
  output.token_ids = torch::empty({batch_size * 2},
                                  accepted_tokens.options().dtype(torch::kInt));
  output.embeddings = torch::empty({batch_size * 2, hidden_size},
                                   accepted_embeddings.options());
  output.positions = torch::empty({batch_size * 2}, base_positions.options());
  output.kv_seq_lens = torch::empty({batch_size}, base_kv_seq_lens.options());
  output.cache_slots = torch::empty({batch_size * 2}, base_positions.options());

  aclTensor* accepted_tokens_acl = nullptr;
  aclTensor* accepted_embeddings_acl = nullptr;
  aclTensor* embedding_placeholder_acl = nullptr;
  aclTensor* base_positions_acl = nullptr;
  aclTensor* base_kv_seq_lens_acl = nullptr;
  aclTensor* block_tables_acl = nullptr;
  aclTensor* token_ids_acl = nullptr;
  aclTensor* embeddings_acl = nullptr;
  aclTensor* positions_acl = nullptr;
  aclTensor* kv_seq_lens_acl = nullptr;
  aclTensor* cache_slots_acl = nullptr;
  create_acltensor(&accepted_tokens_acl, accepted_tokens);
  create_acltensor(&accepted_embeddings_acl, accepted_embeddings);
  create_acltensor(&embedding_placeholder_acl, embedding_placeholder);
  create_acltensor(&base_positions_acl, position_rows);
  create_acltensor(&base_kv_seq_lens_acl, kv_seq_len_rows);
  create_acltensor(&block_tables_acl, block_tables);
  create_acltensor(&token_ids_acl, output.token_ids);
  create_acltensor(&embeddings_acl, output.embeddings);
  create_acltensor(&positions_acl, output.positions);
  create_acltensor(&kv_seq_lens_acl, output.kv_seq_lens);
  create_acltensor(&cache_slots_acl, output.cache_slots);

  const int32_t device_id = accepted_tokens.device().index();
  aclrtStream stream = c10_npu::getCurrentNPUStream(device_id).stream();
  uint64_t workspace_size = 0;
  aclOpExecutor* executor = nullptr;
  CHECK_ACL_SUCCESS(
      aclnnMtpPrepareNextDraftGetWorkspaceSize(accepted_tokens_acl,
                                               accepted_embeddings_acl,
                                               embedding_placeholder_acl,
                                               base_positions_acl,
                                               base_kv_seq_lens_acl,
                                               block_tables_acl,
                                               block_size,
                                               token_ids_acl,
                                               embeddings_acl,
                                               positions_acl,
                                               kv_seq_lens_acl,
                                               cache_slots_acl,
                                               &workspace_size,
                                               &executor),
      "mtp_prepare_next_draft: failed to get workspace size");
  torch::Tensor workspace;
  void* workspace_addr = nullptr;
  if (workspace_size > 0) {
    workspace = torch::empty({static_cast<int64_t>(workspace_size)},
                             accepted_tokens.options().dtype(torch::kUInt8));
    workspace_addr = workspace.data_ptr();
  }
  CHECK_ACL_SUCCESS(aclnnMtpPrepareNextDraft(
                        workspace_addr, workspace_size, executor, stream),
                    "mtp_prepare_next_draft: kernel launch failed");

  aclDestroyTensor(accepted_tokens_acl);
  aclDestroyTensor(accepted_embeddings_acl);
  aclDestroyTensor(embedding_placeholder_acl);
  aclDestroyTensor(base_positions_acl);
  aclDestroyTensor(base_kv_seq_lens_acl);
  aclDestroyTensor(block_tables_acl);
  aclDestroyTensor(token_ids_acl);
  aclDestroyTensor(embeddings_acl);
  aclDestroyTensor(positions_acl);
  aclDestroyTensor(kv_seq_lens_acl);
  aclDestroyTensor(cache_slots_acl);
  return output;
}

}  // namespace xllm::kernel::npu

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

void check_contiguous(const torch::Tensor& tensor, const char* name) {
  CHECK(tensor.is_contiguous())
      << "mtp_prepare_next_draft: " << name << " must be contiguous";
}

}  // namespace

MtpPrepareNextDraftOutput mtp_prepare_next_draft(
    const torch::Tensor& accepted_tokens,
    const torch::Tensor& accepted_embeddings,
    const torch::Tensor& embedding_placeholder,
    const torch::Tensor& base_positions,
    const torch::Tensor& base_kv_seq_lens,
    const torch::Tensor& block_tables,
    int64_t block_size) {
  check_tensor(accepted_tokens,
               "accepted_tokens",
               "mtp_prepare_next_draft");
  check_tensor(accepted_embeddings,
               "accepted_embeddings",
               "mtp_prepare_next_draft");
  check_tensor(embedding_placeholder,
               "embedding_placeholder",
               "mtp_prepare_next_draft");
  check_tensor(base_positions, "base_positions", "mtp_prepare_next_draft");
  check_tensor(base_kv_seq_lens,
               "base_kv_seq_lens",
               "mtp_prepare_next_draft");
  check_tensor(block_tables, "block_tables", "mtp_prepare_next_draft");

  CHECK_EQ(accepted_tokens.scalar_type(), torch::kLong);
  CHECK(accepted_embeddings.scalar_type() == torch::kFloat16 ||
        accepted_embeddings.scalar_type() == torch::kBFloat16);
  CHECK_EQ(embedding_placeholder.scalar_type(),
           accepted_embeddings.scalar_type());
  CHECK_EQ(base_positions.scalar_type(), torch::kInt);
  CHECK_EQ(base_kv_seq_lens.scalar_type(), torch::kInt);
  CHECK_EQ(block_tables.scalar_type(), torch::kInt);
  CHECK_EQ(accepted_tokens.dim(), 2);
  CHECK_EQ(accepted_embeddings.dim(), 3);
  CHECK_EQ(block_tables.dim(), 2);
  CHECK_GT(block_size, 0);

  const int64_t batch_size = accepted_tokens.size(0);
  const int64_t speculative_width = accepted_tokens.size(1);
  const int64_t hidden_size = accepted_embeddings.size(2);
  CHECK_EQ(accepted_embeddings.size(0), batch_size);
  CHECK_EQ(accepted_embeddings.size(1), speculative_width);
  CHECK_EQ(embedding_placeholder.numel(), hidden_size);
  CHECK_GE(base_positions.numel(), batch_size);
  CHECK_GE(base_kv_seq_lens.numel(), batch_size);
  CHECK_GE(block_tables.size(0), batch_size);
  CHECK_EQ((hidden_size * accepted_embeddings.element_size()) % 32, 0)
      << "embedding row must be 32-byte aligned";

  check_contiguous(accepted_tokens, "accepted_tokens");
  check_contiguous(accepted_embeddings, "accepted_embeddings");
  check_contiguous(embedding_placeholder, "embedding_placeholder");
  check_contiguous(base_positions, "base_positions");
  check_contiguous(base_kv_seq_lens, "base_kv_seq_lens");
  check_contiguous(block_tables, "block_tables");

  MtpPrepareNextDraftOutput output;
  output.token_ids = torch::empty(
      {batch_size * 2}, accepted_tokens.options().dtype(torch::kInt));
  output.embeddings = torch::empty(
      {batch_size * 2, hidden_size}, accepted_embeddings.options());
  output.positions =
      torch::empty({batch_size * 2}, base_positions.options());
  output.kv_seq_lens =
      torch::empty({batch_size}, base_kv_seq_lens.options());
  output.cache_slots =
      torch::empty({batch_size * 2}, base_positions.options());

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
  create_acltensor(&base_positions_acl, base_positions);
  create_acltensor(&base_kv_seq_lens_acl, base_kv_seq_lens);
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
    workspace = torch::empty(
        {static_cast<int64_t>(workspace_size)},
        accepted_tokens.options().dtype(torch::kUInt8));
    workspace_addr = workspace.data_ptr();
  }
  CHECK_ACL_SUCCESS(aclnnMtpPrepareNextDraft(
                        workspace_addr,
                        workspace_size,
                        executor,
                        stream),
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

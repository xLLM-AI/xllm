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

#include <c10/core/Device.h>
#include <glog/logging.h>
#include <torch/torch.h>
#include <torch_npu/csrc/libs/init_npu.h>
#include <torch_npu/torch_npu.h>

#include <optional>
#include <sstream>
#include <string>
#include <tuple>

#ifdef TORCH_HIGHER_THAN_PTA6
#include <torch_npu/csrc/framework/OpCommand.h>
#else
#include <torch_npu/csrc/aten/NPUNativeFunctions.h>
#include <torch_npu/csrc/framework/utils/OpPreparation.h>
#endif

#include "acl/acl.h"
#include "aclnn_rec_constrained_top_k.h"
#include "core/common/macros.h"
#include "core/kernels/npu/utils.h"
#include "xllm_ops_api.h"

namespace xllm::kernel::npu {
namespace {

constexpr int64_t kRecConstrainedTopKFusedMaxK = 512;

bool tensors_on_same_device(const torch::Tensor& first,
                            const torch::Tensor& second) {
  return !first.defined() || !second.defined() ||
         first.device() == second.device();
}

std::string tensor_summary(const char* name, const torch::Tensor& tensor) {
  if (!tensor.defined()) {
    return std::string(name) + "=undefined";
  }
  std::ostringstream out;
  out << name << "(sizes=" << tensor.sizes()
      << ", dtype=" << c10::toString(tensor.scalar_type())
      << ", device=" << tensor.device() << ")";
  return out.str();
}

std::optional<std::string> fused_inputs_unsupported_reason(
    const torch::Tensor& logits,
    const torch::Tensor& sequence_group,
    const torch::Tensor& first_token_ids,
    const torch::Tensor& prefix1_offsets,
    const torch::Tensor& prefix1_values,
    const torch::Tensor& prefix1_pair_keys,
    const torch::Tensor& prefix2_value_offsets,
    const torch::Tensor& prefix2_values,
    const torch::Tensor& temperatures,
    int64_t current_step,
    int64_t top_k) {
  if (!logits.defined() || logits.dim() != 2 || logits.numel() == 0) {
    return "logits must be defined non-empty 2-D";
  }
  if (current_step < 0 || current_step > 2 || top_k <= 0) {
    return "current_step/top_k out of supported range";
  }
  if (top_k > kRecConstrainedTopKFusedMaxK) {
    return "top_k exceeds fused kernel max";
  }
  const torch::ScalarType logits_dtype = logits.scalar_type();
  if (logits_dtype != torch::kFloat16 && logits_dtype != torch::kBFloat16 &&
      logits_dtype != torch::kFloat32) {
    return "logits dtype must be fp16, bf16, or fp32";
  }
  if (!first_token_ids.defined() || first_token_ids.numel() == 0 ||
      first_token_ids.scalar_type() != torch::kInt32) {
    return "first_token_ids must be non-empty int32";
  }
  if (!prefix1_offsets.defined() || prefix1_offsets.numel() == 0 ||
      prefix1_offsets.scalar_type() != torch::kInt32) {
    return "prefix1_offsets must be non-empty int32";
  }
  if (!prefix1_values.defined() || prefix1_values.numel() == 0 ||
      prefix1_values.scalar_type() != torch::kInt32) {
    return "prefix1_values must be non-empty int32";
  }
  if (!prefix1_pair_keys.defined() || prefix1_pair_keys.numel() == 0 ||
      prefix1_pair_keys.scalar_type() != torch::kInt64) {
    return "prefix1_pair_keys must be non-empty int64";
  }
  if (!prefix2_value_offsets.defined() || prefix2_value_offsets.numel() < 2 ||
      prefix2_value_offsets.scalar_type() != torch::kInt32) {
    return "prefix2_value_offsets must have at least 2 int32 values";
  }
  if (!prefix2_values.defined() || prefix2_values.numel() == 0 ||
      prefix2_values.scalar_type() != torch::kInt32) {
    return "prefix2_values must be non-empty int32";
  }
  if (current_step > 0 && (!sequence_group.defined() ||
                           sequence_group.scalar_type() != torch::kInt32 ||
                           sequence_group.size(-1) <= current_step)) {
    return "sequence_group must be int32 and contain current prefix columns";
  }
  if (temperatures.defined() &&
      (temperatures.scalar_type() != torch::kFloat32 ||
       (temperatures.numel() != 1 && temperatures.numel() != logits.size(0)))) {
    return "temperatures must be float32 scalar or per-row tensor";
  }
  if (!tensors_on_same_device(logits, sequence_group) ||
      !tensors_on_same_device(logits, first_token_ids) ||
      !tensors_on_same_device(logits, prefix1_offsets) ||
      !tensors_on_same_device(logits, prefix1_values) ||
      !tensors_on_same_device(logits, prefix1_pair_keys) ||
      !tensors_on_same_device(logits, prefix2_value_offsets) ||
      !tensors_on_same_device(logits, prefix2_values) ||
      !tensors_on_same_device(logits, temperatures)) {
    return "all fused inputs must be on logits device";
  }
  return std::nullopt;
}

void destroy_tensor_if_needed(aclTensor* tensor) {
  if (tensor != nullptr) {
    aclDestroyTensor(tensor);
  }
}

}  // namespace

std::optional<std::tuple<torch::Tensor, torch::Tensor>>
rec_constrained_topk_fused(const torch::Tensor& logits,
                           const torch::Tensor& sequence_group,
                           const torch::Tensor& first_token_ids,
                           const torch::Tensor& prefix1_offsets,
                           const torch::Tensor& prefix1_values,
                           const torch::Tensor& prefix1_pair_keys,
                           const torch::Tensor& prefix2_value_offsets,
                           const torch::Tensor& prefix2_values,
                           const torch::Tensor& temperatures,
                           int64_t current_step,
                           int64_t top_k,
                           int64_t max_prefix1_degree,
                           int64_t max_prefix2_degree) {
  const std::optional<std::string> unsupported_reason =
      fused_inputs_unsupported_reason(logits,
                                      sequence_group,
                                      first_token_ids,
                                      prefix1_offsets,
                                      prefix1_values,
                                      prefix1_pair_keys,
                                      prefix2_value_offsets,
                                      prefix2_values,
                                      temperatures,
                                      current_step,
                                      top_k);
  if (unsupported_reason.has_value()) {
    LOG_FIRST_N(WARNING, 8)
        << "rec_constrained_topk_fused: unsupported inputs, reason="
        << unsupported_reason.value() << ", current_step=" << current_step
        << ", top_k=" << top_k << ", " << tensor_summary("logits", logits)
        << ", " << tensor_summary("sequence_group", sequence_group) << ", "
        << tensor_summary("first_token_ids", first_token_ids) << ", "
        << tensor_summary("prefix1_offsets", prefix1_offsets) << ", "
        << tensor_summary("prefix1_values", prefix1_values) << ", "
        << tensor_summary("prefix1_pair_keys", prefix1_pair_keys) << ", "
        << tensor_summary("prefix2_value_offsets", prefix2_value_offsets)
        << ", " << tensor_summary("prefix2_values", prefix2_values) << ", "
        << tensor_summary("temperatures", temperatures);
    return std::nullopt;
  }

  torch::Tensor logits_contiguous = logits.contiguous();
  torch::Tensor sequence_group_contiguous =
      sequence_group.defined() ? sequence_group.contiguous()
                               : torch::empty({0},
                                              torch::TensorOptions()
                                                  .dtype(torch::kInt32)
                                                  .device(logits.device()));
  torch::Tensor fused_temperatures =
      temperatures.defined() && temperatures.numel() > 0
          ? temperatures.contiguous()
          : torch::ones({1},
                        torch::TensorOptions()
                            .dtype(torch::kFloat32)
                            .device(logits.device()));
  torch::Tensor first_token_ids_contiguous = first_token_ids.contiguous();
  torch::Tensor prefix1_offsets_contiguous = prefix1_offsets.contiguous();
  torch::Tensor prefix1_values_contiguous = prefix1_values.contiguous();
  torch::Tensor prefix1_pair_keys_contiguous = prefix1_pair_keys.contiguous();
  torch::Tensor prefix2_value_offsets_contiguous =
      prefix2_value_offsets.contiguous();
  torch::Tensor prefix2_values_contiguous = prefix2_values.contiguous();
  torch::Tensor out_tokens = torch::empty(
      {logits.size(0), top_k},
      torch::TensorOptions().dtype(torch::kInt32).device(logits.device()));
  torch::Tensor out_logprobs = torch::empty(
      {logits.size(0), top_k},
      torch::TensorOptions().dtype(torch::kFloat32).device(logits.device()));

  aclTensor* logits_ids = nullptr;
  aclTensor* sequence_group_ids = nullptr;
  aclTensor* first_token_ids_ids = nullptr;
  aclTensor* prefix1_offsets_ids = nullptr;
  aclTensor* prefix1_values_ids = nullptr;
  aclTensor* prefix1_pair_keys_ids = nullptr;
  aclTensor* prefix2_value_offsets_ids = nullptr;
  aclTensor* prefix2_values_ids = nullptr;
  aclTensor* temperatures_ids = nullptr;
  aclTensor* out_tokens_ids = nullptr;
  aclTensor* out_logprobs_ids = nullptr;

  create_acltensor(&logits_ids, logits_contiguous);
  create_acltensor(&sequence_group_ids, sequence_group_contiguous);
  create_acltensor(&first_token_ids_ids, first_token_ids_contiguous);
  create_acltensor(&prefix1_offsets_ids, prefix1_offsets_contiguous);
  create_acltensor(&prefix1_values_ids, prefix1_values_contiguous);
  create_acltensor(&prefix1_pair_keys_ids, prefix1_pair_keys_contiguous);
  create_acltensor(&prefix2_value_offsets_ids,
                   prefix2_value_offsets_contiguous);
  create_acltensor(&prefix2_values_ids, prefix2_values_contiguous);
  create_acltensor(&temperatures_ids, fused_temperatures);
  create_acltensor(&out_tokens_ids, out_tokens);
  create_acltensor(&out_logprobs_ids, out_logprobs);

  uint64_t workspace_size = 0;
  aclOpExecutor* executor = nullptr;
  const aclnnStatus workspace_status =
      aclnnRecConstrainedTopKGetWorkspaceSize(logits_ids,
                                              sequence_group_ids,
                                              first_token_ids_ids,
                                              prefix1_offsets_ids,
                                              prefix1_values_ids,
                                              prefix1_pair_keys_ids,
                                              prefix2_value_offsets_ids,
                                              prefix2_values_ids,
                                              temperatures_ids,
                                              current_step,
                                              top_k,
                                              max_prefix1_degree,
                                              max_prefix2_degree,
                                              out_tokens_ids,
                                              out_logprobs_ids,
                                              &workspace_size,
                                              &executor);
  if (workspace_status != 0) {
    LOG(WARNING) << "rec_constrained_topk_fused: failed to get workspace, "
                 << "status=" << workspace_status
                 << ", detail=" << aclGetRecentErrMsg();
    destroy_tensor_if_needed(logits_ids);
    destroy_tensor_if_needed(sequence_group_ids);
    destroy_tensor_if_needed(first_token_ids_ids);
    destroy_tensor_if_needed(prefix1_offsets_ids);
    destroy_tensor_if_needed(prefix1_values_ids);
    destroy_tensor_if_needed(prefix1_pair_keys_ids);
    destroy_tensor_if_needed(prefix2_value_offsets_ids);
    destroy_tensor_if_needed(prefix2_values_ids);
    destroy_tensor_if_needed(temperatures_ids);
    destroy_tensor_if_needed(out_tokens_ids);
    destroy_tensor_if_needed(out_logprobs_ids);
    return std::nullopt;
  }

  void* workspace_addr = nullptr;
  if (workspace_size > 0) {
    const aclError malloc_status =
        aclrtMalloc(&workspace_addr, workspace_size, ACL_MEM_MALLOC_HUGE_FIRST);
    if (malloc_status != ACL_ERROR_NONE) {
      LOG(WARNING) << "rec_constrained_topk_fused: failed to allocate "
                   << "workspace, status=" << malloc_status;
      destroy_tensor_if_needed(logits_ids);
      destroy_tensor_if_needed(sequence_group_ids);
      destroy_tensor_if_needed(first_token_ids_ids);
      destroy_tensor_if_needed(prefix1_offsets_ids);
      destroy_tensor_if_needed(prefix1_values_ids);
      destroy_tensor_if_needed(prefix1_pair_keys_ids);
      destroy_tensor_if_needed(prefix2_value_offsets_ids);
      destroy_tensor_if_needed(prefix2_values_ids);
      destroy_tensor_if_needed(temperatures_ids);
      destroy_tensor_if_needed(out_tokens_ids);
      destroy_tensor_if_needed(out_logprobs_ids);
      return std::nullopt;
    }
  }

  const int32_t device_id = logits.device().index();
  const aclrtStream stream = c10_npu::getCurrentNPUStream(device_id).stream();
  const aclnnStatus run_status =
      aclnnRecConstrainedTopK(workspace_addr, workspace_size, executor, stream);
  if (run_status != 0) {
    LOG(WARNING) << "rec_constrained_topk_fused: failed to execute, status="
                 << run_status << ", detail=" << aclGetRecentErrMsg();
  }

  if (workspace_size > 0) {
    const aclError free_status = aclrtFree(workspace_addr);
    if (free_status != ACL_ERROR_NONE) {
      LOG(WARNING) << "rec_constrained_topk_fused: failed to free workspace, "
                   << "status=" << free_status;
    }
  }
  destroy_tensor_if_needed(logits_ids);
  destroy_tensor_if_needed(sequence_group_ids);
  destroy_tensor_if_needed(first_token_ids_ids);
  destroy_tensor_if_needed(prefix1_offsets_ids);
  destroy_tensor_if_needed(prefix1_values_ids);
  destroy_tensor_if_needed(prefix1_pair_keys_ids);
  destroy_tensor_if_needed(prefix2_value_offsets_ids);
  destroy_tensor_if_needed(prefix2_values_ids);
  destroy_tensor_if_needed(temperatures_ids);
  destroy_tensor_if_needed(out_tokens_ids);
  destroy_tensor_if_needed(out_logprobs_ids);
  if (run_status != 0) {
    return std::nullopt;
  }
  return std::make_tuple(out_tokens, out_logprobs);
}

}  // namespace xllm::kernel::npu

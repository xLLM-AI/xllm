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

#include <cstdlib>
#include <cstring>
#include <map>
#include <nlohmann/json.hpp>
#include <utility>
#include <vector>
#ifdef TORCH_HIGHER_THAN_PTA6
#include <torch_npu/csrc/framework/OpCommand.h>
#else
#include <torch_npu/csrc/aten/NPUNativeFunctions.h>
#include <torch_npu/csrc/framework/utils/OpPreparation.h>
#endif

#include "acl/acl.h"
#include "aclnn_select_unshared_kv.h"
#include "core/common/macros.h"
#include "core/kernels/npu/utils.h"
#include "xllm_ops_api.h"

namespace xllm::kernel::npu {

namespace {

constexpr const char* kAsyncCacheSelectEnv =
    "XLLM_ONEREC_XATTN_ASYNC_CACHE_SELECT";

struct SelectUnsharedKvWorkspace {
  void* addr = nullptr;
  uint64_t size = 0;
  std::vector<void*> retired_addrs;
};

using SelectUnsharedKvWorkspaceKey = std::pair<int32_t, uint64_t>;

thread_local std::map<SelectUnsharedKvWorkspaceKey, SelectUnsharedKvWorkspace>
    g_select_unshared_kv_workspaces;

bool enable_async_cache_select() {
  static const bool enabled = []() {
    const char* env = std::getenv(kAsyncCacheSelectEnv);
    return env != nullptr && std::strcmp(env, "1") == 0;
  }();
  return enabled;
}

SelectUnsharedKvWorkspaceKey make_workspace_key(int32_t device_id,
                                                aclrtStream stream) {
  return std::make_pair(
      device_id, static_cast<uint64_t>(reinterpret_cast<uintptr_t>(stream)));
}

void* get_persistent_select_workspace(uint64_t workspace_size,
                                      int32_t device_id,
                                      aclrtStream stream) {
  if (workspace_size == 0) {
    return nullptr;
  }

  auto& workspace =
      g_select_unshared_kv_workspaces[make_workspace_key(device_id, stream)];
  if (workspace.addr != nullptr && workspace.size >= workspace_size) {
    return workspace.addr;
  }

  void* workspace_addr = nullptr;
  CHECK_ACL_SUCCESS(
      aclrtMalloc(&workspace_addr, workspace_size, ACL_MEM_MALLOC_HUGE_FIRST),
      "select_unshared_kv: failed to allocate persistent workspace");
  if (workspace.addr != nullptr) {
    // Async mode removes the post-kernel host sync, so the old workspace may
    // still be used by earlier same-stream work until the worker's later sync.
    workspace.retired_addrs.push_back(workspace.addr);
  }
  workspace.addr = workspace_addr;
  workspace.size = workspace_size;
  return workspace.addr;
}

}  // namespace

// Reorder per-layer unshared KV caches after beam selection in REC
// multi-round decoding.
// Inputs:
//   beam_index: source beam chosen for each output beam.
//   x_key_block/x_value_block: per-layer unshared K/V caches to update.
//   block_table/group_offset: request mapping and per-request beam offsets
//   expected by the NPU kernel.
//   decode_step/beam_size/layer_num: cache slot and layout metadata.
// Output:
//   x_key_block/x_value_block are updated in place to match the selected
//   beams for the next round.
void select_unshared_kv(const torch::Tensor& beam_index,
                        const std::vector<torch::Tensor>& x_key_block,
                        const std::vector<torch::Tensor>& x_value_block,
                        const torch::Tensor& block_table,
                        const torch::Tensor& group_offset,
                        int64_t decode_step,
                        int64_t beam_size,
                        int64_t layer_num) {
  check_tensor(beam_index, "beam_index", "select_unshared_kv");
  check_tensor(block_table, "block_table", "select_unshared_kv");
  check_tensor(group_offset, "group_offset", "select_unshared_kv");
  for (const auto& t : x_key_block) {
    check_tensor(t, "x_key_block[i]", "select_unshared_kv");
  }
  for (const auto& t : x_value_block) {
    check_tensor(t, "x_value_block[i]", "select_unshared_kv");
  }

  aclTensor* beam_index_ids = nullptr;
  aclTensor* group_offset_ids = nullptr;
  aclTensor* block_table_ids = nullptr;
  aclTensorList* x_key_block_list_ids = nullptr;
  aclTensorList* x_value_block_list_ids = nullptr;
  std::vector<aclTensor*> x_key_block_list_ids_vec;
  std::vector<aclTensor*> x_value_block_list_ids_vec;
  for (auto& x_key_block_tensor : x_key_block) {
    aclTensor* x_key_block_id = nullptr;
    create_acltensor(&x_key_block_id, x_key_block_tensor);
    x_key_block_list_ids_vec.push_back(x_key_block_id);
  }
  for (auto& x_value_block_tensor : x_value_block) {
    aclTensor* x_value_block_id = nullptr;
    create_acltensor(&x_value_block_id, x_value_block_tensor);
    x_value_block_list_ids_vec.push_back(x_value_block_id);
  }
  x_key_block_list_ids = aclCreateTensorList(x_key_block_list_ids_vec.data(),
                                             x_key_block_list_ids_vec.size());
  x_value_block_list_ids = aclCreateTensorList(
      x_value_block_list_ids_vec.data(), x_value_block_list_ids_vec.size());
  create_acltensor(&beam_index_ids, beam_index);
  create_acltensor(&group_offset_ids, group_offset);
  create_acltensor(&block_table_ids, block_table);

  int32_t device_id = beam_index.device().index();
  aclrtStream stream = c10_npu::getCurrentNPUStream(device_id).stream();
  uint64_t workspace_size = 0;
  aclOpExecutor* executor = nullptr;
  const bool async_cache_select = enable_async_cache_select();

  CHECK_ACL_SUCCESS(
      aclnnSelectUnsharedKVGetWorkspaceSize(beam_index_ids,
                                            block_table_ids,
                                            x_key_block_list_ids,
                                            x_value_block_list_ids,
                                            group_offset_ids,
                                            decode_step,
                                            beam_size,
                                            layer_num,
                                            x_key_block_list_ids,
                                            x_value_block_list_ids,
                                            &workspace_size,
                                            &executor),
      "select_unshared_kv: failed to get workspace size");
  void* workspace_addr = nullptr;
  if (workspace_size > 0) {
    if (async_cache_select) {
      workspace_addr =
          get_persistent_select_workspace(workspace_size, device_id, stream);
    } else {
      CHECK_ACL_SUCCESS(
          aclrtMalloc(
              &workspace_addr, workspace_size, ACL_MEM_MALLOC_HUGE_FIRST),
          "select_unshared_kv: failed to allocate workspace");
    }
  }
  CHECK_ACL_SUCCESS(
      aclnnSelectUnsharedKV(workspace_addr, workspace_size, executor, stream),
      "select_unshared_kv: failed to reorder caches");
  if (!async_cache_select) {
    CHECK_ACL_SUCCESS(aclrtSynchronizeStream(stream),
                      "select_unshared_kv: failed to synchronize stream");
  }
  aclDestroyTensor(beam_index_ids);
  aclDestroyTensor(group_offset_ids);
  aclDestroyTensor(block_table_ids);
  aclDestroyTensorList(x_key_block_list_ids);
  aclDestroyTensorList(x_value_block_list_ids);
  if (!async_cache_select && workspace_size > 0) {
    CHECK_ACL_SUCCESS(aclrtFree(workspace_addr),
                      "select_unshared_kv: failed to free workspace");
  }
}
}  // namespace xllm::kernel::npu

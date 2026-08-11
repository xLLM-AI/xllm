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

#include <dlfcn.h>
#include <torch/library.h>

#include <string>
#include <tuple>

#include "core/kernels/npu/aclnn/pytorch_npu_helper.hpp"
#include "xllm_ops_api.h"

namespace xllm::kernel::npu {
namespace {

// The PR #6 SparseFlashAttention aclnn op (3 outputs / 8 attrs, rope=None
// supported) exports the *same* symbol name (``aclnnSparseFlashAttention``)
// as the legacy op in ``custom_xllm_math`` (1 output / 5 attrs, rope hardcoded
// 64). The global ``get_op_api_func_addr`` resolver cannot distinguish them --
// whichever vendor wins ``config.ini`` load_priority serves both -- so to call
// the PR #6 op without disturbing the other ops served by ``custom_xllm_math``,
// this implementation dlopens the new-contract vendor ``libcust_opapi.so``
// directly and dlsyms the two entry points, bypassing the global resolver.
//
// The new-contract SFA aclnn op is shipped out-of-tree by different vendor
// packages depending on the deployment: ``custom_transformer`` (the fork repo
// ``build_aclnn.sh`` target named by the original PR) on some boxes, or
// ``glm_next_transformer`` (Huawei CANN GLM-Next transformer op package) on
// others. Both export an identical 3-output / 8-attr
// ``aclnnSparseFlashAttention``. We probe ``custom_transformer`` first to keep
// the original PR semantics, then fall back to ``glm_next_transformer`` so the
// op resolves on boxes where that is the only installed copy. If neither vendor
// has the symbol, a clear error is reported rather than silently falling back
// to the legacy op.
constexpr const char* kNewSfaVendorLibs[] = {
    "/vendors/custom_transformer/op_api/lib/libcust_opapi.so",
    "/vendors/glm_next_transformer/op_api/lib/libcust_opapi.so",
};

struct CustomTransformerOpApi {
  void* handler = nullptr;
  void* get_workspace_size = nullptr;
  void* op_api = nullptr;
};

// Loads the new-contract SFA vendor libcust_opapi.so once and resolves the two
// aclnnSparseFlashAttention entry points. Probes kNewSfaVendorLibs in order and
// keeps the first vendor whose .so both dlopens and dlsyms the two symbols.
// Thread-safe via call_once; the .so stays loaded for the process lifetime
// (matching the global resolver's behaviour).
const CustomTransformerOpApi& load_custom_transformer_op_api() {
  static CustomTransformerOpApi api;
  static std::once_flag flag;
  std::call_once(flag, []() {
    const char* ascend_opp = std::getenv("ASCEND_OPP_PATH");
    if (ascend_opp == nullptr) {
      TORCH_CHECK(false,
                  "ASCEND_OPP_PATH is not set; cannot locate the "
                  "new-contract SFA vendor for sparse_flash_attention.");
    }
    std::string tried_paths;
    bool resolved = false;
    for (const char* rel : kNewSfaVendorLibs) {
      std::string lib_path = std::string(ascend_opp) + rel;
      tried_paths += "  " + lib_path + "\n";
      void* handler = dlopen(lib_path.c_str(), RTLD_LAZY);
      if (handler == nullptr) {
        continue;
      }
      void* get_ws =
          dlsym(handler, "aclnnSparseFlashAttentionGetWorkspaceSize");
      void* op_api = dlsym(handler, "aclnnSparseFlashAttention");
      if (get_ws != nullptr && op_api != nullptr) {
        api.handler = handler;
        api.get_workspace_size = get_ws;
        api.op_api = op_api;
        resolved = true;
        break;
      }
      // Opened but missing the symbol: close and try the next vendor.
      dlclose(handler);
    }
    TORCH_CHECK(resolved,
                "sparse_flash_attention: aclnnSparseFlashAttention or "
                "GetWorkspaceSize not found in any new-contract SFA vendor "
                "libcust_opapi.so. Tried:\n",
                tried_paths,
                "Install the PR #6 SFA vendor (custom_transformer or "
                "glm_next_transformer) first.");
  });
  return api;
}

// Allocates the op's three outputs, matching the PR #6 op contract.
// attention_out always matches query's shape/dtype; softmax_max/sum are empty
// ({0}) float32 unless return_softmax_lse is set.
std::tuple<at::Tensor, at::Tensor, at::Tensor>
construct_sparse_flash_attention_outputs(const at::Tensor& query,
                                         const at::Tensor& key,
                                         const std::string& layout_query_str,
                                         const std::string& layout_kv_str,
                                         bool return_softmax_lse) {
  constexpr int64_t kDim0 = 0;
  constexpr int64_t kDim1 = 1;
  constexpr int64_t kDim2 = 2;
  constexpr int64_t kDim3 = 3;

  TORCH_CHECK(layout_query_str == "BSND" || layout_query_str == "TND",
              "The layout of query only support BSND and TND, but got ",
              layout_query_str);

  at::Tensor attention_output =
      at::empty(query.sizes(), query.options().dtype(query.dtype()));

  at::SmallVector<int64_t, 4> softmax_size;
  if (return_softmax_lse) {
    if (query.dim() == 3) {  // TND
      const int64_t kv_head_num =
          layout_kv_str == "PA_BSND" ? key.size(kDim2) : key.size(kDim1);
      softmax_size = {
          kv_head_num, query.size(kDim0), query.size(kDim1) / kv_head_num};
    } else {  // BSND
      softmax_size = {query.size(kDim0),
                      key.size(kDim2),
                      query.size(kDim1),
                      query.size(kDim2) / key.size(kDim2)};
    }
  } else {
    softmax_size = {0};
  }
  at::Tensor softmax_max =
      at::empty(softmax_size, query.options().dtype(at::kFloat));
  at::Tensor softmax_sum =
      at::empty(softmax_size, query.options().dtype(at::kFloat));
  return std::make_tuple(attention_output, softmax_max, softmax_sum);
}

void check_sparse_flash_attention_shape_and_dtype(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const at::Tensor& sparse_indices,
    int64_t sparse_block_size,
    const c10::string_view& layout_query,
    const c10::string_view& layout_kv) {
  TORCH_CHECK(query.dim() >= 1,
              "query's dim num should be at least 1, actual ",
              query.dim(),
              ".");
  TORCH_CHECK(query.dtype() == at::kHalf || query.dtype() == at::kBFloat16,
              "query should be FLOAT16 or BFLOAT16.");
  TORCH_CHECK(key.dtype() == query.dtype(),
              "key's dtype should be equal to query's dtype.");
  TORCH_CHECK(value.dtype() == query.dtype(),
              "value's dtype should be equal to query's dtype.");
  TORCH_CHECK(sparse_indices.dtype() == at::kInt,
              "sparse_indices should be INT32.");
  TORCH_CHECK(sparse_block_size > 0,
              "sparse_block_size should be greater than 0, actual ",
              sparse_block_size,
              ".");
  TORCH_CHECK(!layout_query.empty(), "layout_query should not be empty.");
  TORCH_CHECK(!layout_kv.empty(), "layout_kv should not be empty.");
}

}  // namespace

// PR #6 SparseFlashAttention: 3 outputs / 8 attrs, supports rope=None.
// Resolves the op from the new-contract SFA vendor directly (dlopen+dlsym,
// probing custom_transformer then glm_next_transformer) rather than the global
// aclnn resolver, so it does not disturb the other ops served by
// custom_xllm_math (which keeps load_priority via config.ini).
std::tuple<at::Tensor, at::Tensor, at::Tensor> sparse_flash_attention(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const at::Tensor& sparse_indices,
    const c10::optional<at::Tensor>& block_table,
    const c10::optional<at::Tensor>& actual_seq_lengths_query,
    const c10::optional<at::Tensor>& actual_seq_lengths_kv,
    const c10::optional<at::Tensor>& query_rope,
    const c10::optional<at::Tensor>& key_rope,
    double scale_value,
    int64_t sparse_block_size,
    c10::string_view layout_query,
    c10::string_view layout_kv,
    int64_t sparse_mode,
    int64_t pre_tokens,
    int64_t next_tokens,
    int64_t attention_mode,
    bool return_softmax_lse) {
  check_sparse_flash_attention_shape_and_dtype(query,
                                               key,
                                               value,
                                               sparse_indices,
                                               sparse_block_size,
                                               layout_query,
                                               layout_kv);

  std::string query_layout_str = std::string(layout_query);
  std::string kv_layout_str = std::string(layout_kv);
  auto [out, softmax_max, softmax_sum] =
      construct_sparse_flash_attention_outputs(
          query, key, query_layout_str, kv_layout_str, return_softmax_lse);

  char* query_layout_ptr = const_cast<char*>(query_layout_str.c_str());
  char* kv_layout_ptr = const_cast<char*>(kv_layout_str.c_str());

  const CustomTransformerOpApi& api = load_custom_transformer_op_api();

  // Mirror EXEC_NPU_CMD (pytorch_npu_helper.hpp:703): the GetWorkspaceSize /
  // op entry points come from the resolved new-contract SFA vendor .so above,
  // while the optional HugeMem helpers are resolved through the global
  // resolver (they live in the built-in libopapi.so, not the vendor .so).
  static const auto init_mem_addr =
      ::xllm::kernel::npu::aclnn::detail::get_op_api_func_addr(
          "InitHugeMemThreadLocal");
  static const auto uninit_mem_addr =
      ::xllm::kernel::npu::aclnn::detail::get_op_api_func_addr(
          "UnInitHugeMemThreadLocal");
  static const auto release_mem_addr =
      ::xllm::kernel::npu::aclnn::detail::get_op_api_func_addr(
          "ReleaseHugeMem");

  auto acl_stream = c10_npu::getCurrentNPUStream().stream(false);
  uint64_t workspace_size = 0;
  uint64_t* workspace_size_addr = &workspace_size;
  ::aclOpExecutor* executor = nullptr;
  ::aclOpExecutor** executor_addr = &executor;

  ::xllm::kernel::npu::aclnn::detail::InitHugeMemThreadLocalFn init_mem_func =
      reinterpret_cast<
          ::xllm::kernel::npu::aclnn::detail::InitHugeMemThreadLocalFn>(
          init_mem_addr);
  ::xllm::kernel::npu::aclnn::detail::UnInitHugeMemThreadLocalFn
      uninit_mem_func = reinterpret_cast<
          ::xllm::kernel::npu::aclnn::detail::UnInitHugeMemThreadLocalFn>(
          uninit_mem_addr);
  if (init_mem_func) {
    init_mem_func(nullptr, false);
  }

  auto converted_params = ::xllm::kernel::npu::aclnn::detail::convert_types(
      query,
      key,
      value,
      sparse_indices,
      block_table,
      actual_seq_lengths_query,
      actual_seq_lengths_kv,
      query_rope,
      key_rope,
      scale_value,
      sparse_block_size,
      query_layout_ptr,
      kv_layout_ptr,
      sparse_mode,
      pre_tokens,
      next_tokens,
      attention_mode,
      return_softmax_lse,
      out,
      softmax_max,
      softmax_sum,
      workspace_size_addr,
      executor_addr);

  static auto get_workspace_size_func =
      ::xllm::kernel::npu::aclnn::detail::convert_to_op_api_func(
          converted_params, api.get_workspace_size);
  auto workspace_status = ::xllm::kernel::npu::aclnn::detail::call(
      get_workspace_size_func, converted_params);
  TORCH_CHECK(workspace_status == 0,
              "call aclnnSparseFlashAttention failed, detail: ",
              aclGetRecentErrMsg());

  void* workspace_addr = nullptr;
  at::Tensor workspace_tensor;
  if (workspace_size != 0) {
    at::TensorOptions options =
        at::TensorOptions(torch_npu::utils::get_npu_device_type());
    workspace_tensor = at::empty({static_cast<int64_t>(workspace_size)},
                                 options.dtype(at::kByte));
    workspace_addr = const_cast<void*>(workspace_tensor.storage().data());
  }

  auto acl_call = [=]() -> int {
    using OpApiFunc =
        int (*)(void*, uint64_t, ::aclOpExecutor*, const aclrtStream);
    OpApiFunc op_api_func = reinterpret_cast<OpApiFunc>(api.op_api);
    auto api_ret =
        op_api_func(workspace_addr, workspace_size, executor, acl_stream);
    TORCH_CHECK(api_ret == 0,
                "call aclnnSparseFlashAttention failed, detail: ",
                aclGetRecentErrMsg());
    ::xllm::kernel::npu::aclnn::detail::release_convert_types(converted_params);
    ::xllm::kernel::npu::aclnn::detail::ReleaseHugeMemFn release_mem_func =
        reinterpret_cast<::xllm::kernel::npu::aclnn::detail::ReleaseHugeMemFn>(
            release_mem_addr);
    if (release_mem_func) {
      release_mem_func(nullptr, false);
    }
    return api_ret;
  };
  at_npu::native::OpCommand cmd;
  cmd.Name("aclnnSparseFlashAttention");
  cmd.SetCustomHandler(acl_call);
  cmd.Run();

  if (uninit_mem_func) {
    uninit_mem_func(nullptr, false);
  }

  return std::make_tuple(out, softmax_max, softmax_sum);
}

}  // namespace xllm::kernel::npu

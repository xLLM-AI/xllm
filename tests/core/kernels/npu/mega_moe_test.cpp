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

#include <gtest/gtest.h>
#include <torch/torch.h>

#include <cstdlib>
#include <optional>
#include <tuple>

#include "core/kernels/npu/mega_moe_acl_contract.h"
#include "core/kernels/npu/npu_ops_api.h"

namespace xllm::kernel::npu {
namespace {

TEST(MegaMoeTest, AcceptsOnlyExactCustomVendorAbiProvenance) {
  const std::string expected =
      "/isolated/vendor/op_api/lib/libcust_opapi.so";
  auto compatible = validate_mega_moe_op_api_paths(
      expected, expected, expected);
  EXPECT_TRUE(compatible.compatible);
  EXPECT_TRUE(compatible.same_library);

  auto stock = validate_mega_moe_op_api_paths(
      expected,
      "/usr/local/Ascend/cann/lib64/libopapi.so",
      "/usr/local/Ascend/cann/lib64/libopapi.so");
  EXPECT_FALSE(stock.compatible);
  EXPECT_TRUE(stock.same_library);

  auto split = validate_mega_moe_op_api_paths(
      expected, expected, "/other/vendor/libcust_opapi.so");
  EXPECT_FALSE(split.compatible);
  EXPECT_FALSE(split.same_library);
}

TEST(MegaMoeTest, CustomAbiKeepsOptionalInputsNullAndTopologyBeforeOutputs) {
  const auto bf16 = torch::TensorOptions().dtype(torch::kBFloat16);
  const auto int32 = torch::TensorOptions().dtype(torch::kInt32);
  auto context = torch::empty({1}, int32);
  auto x = torch::empty({1, 2048}, bf16);
  auto ids = torch::empty({1, 8}, int32);
  auto weights = torch::empty({1, 8}, bf16);
  std::vector<torch::Tensor> w1_storage = {
      torch::empty({2048, 1024}, bf16)};
  std::vector<torch::Tensor> w2_storage = {
      torch::empty({512, 2048}, bf16)};
  torch::TensorList w1(w1_storage);
  torch::TensorList w2(w2_storage);
  std::optional<torch::TensorList> missing_list = std::nullopt;
  std::optional<torch::Tensor> missing_mask = std::nullopt;
  auto output = torch::empty_like(x);
  auto token_counts = torch::empty({1}, int32);
  int64_t moe_expert_num = 256;
  int64_t ep_world_size = 16;
  int64_t ccl_buffer_size = 1;
  int64_t max_recv_token_num = 0;
  int64_t dispatch_quant_mode = 0;
  int64_t dispatch_quant_out_dtype = 28;
  int64_t combine_quant_mode = 0;
  char comm_alg_storage[] = "";
  char* comm_alg = comm_alg_storage;
  int64_t num_max_tokens_per_rank = 1;
  char activation_storage[] = "swiglu";
  char* activation = activation_storage;
  float activation_clamp = 3.4e38F;
  int64_t topo_type = 0;
  int64_t rank_num_per_server = 2;

  bool called = false;
  auto fake_workspace = [&](auto&... arguments) {
    auto args = std::forward_as_tuple(arguments...);
    static_assert(std::tuple_size_v<decltype(args)> == 26);
    EXPECT_FALSE(std::get<6>(args).has_value());
    EXPECT_FALSE(std::get<7>(args).has_value());
    EXPECT_FALSE(std::get<8>(args).has_value());
    EXPECT_FALSE(std::get<9>(args).has_value());
    EXPECT_FALSE(std::get<10>(args).has_value());
    EXPECT_EQ(std::get<22>(args), 0);
    EXPECT_EQ(std::get<23>(args), 2);
    EXPECT_EQ(&std::get<24>(args), &output);
    EXPECT_EQ(&std::get<25>(args), &token_counts);
    called = true;
  };
  execute_mega_moe_acl_contract(fake_workspace,
                                context, x, ids, weights, w1, w2,
                                missing_list, missing_list,
                                missing_list, missing_list, missing_mask,
                                moe_expert_num, ep_world_size, ccl_buffer_size,
                                max_recv_token_num, dispatch_quant_mode,
                                dispatch_quant_out_dtype, combine_quant_mode,
                                comm_alg, num_max_tokens_per_rank, activation,
                                activation_clamp, topo_type,
                                rank_num_per_server, output, token_counts);
  EXPECT_TRUE(called);
}

TEST(MegaMoeTest, ExactVendorExportsBothAclnnSymbols) {
  const char* enabled = std::getenv("XLLM_RUN_CANN91_MEGA_MOE_SMOKE");
  if (enabled == nullptr || std::string(enabled) != "1") {
    GTEST_SKIP() << "CANN 9.1 custom-vendor ABI smoke is disabled; set "
                    "XLLM_RUN_CANN91_MEGA_MOE_SMOKE=1 in the isolated vendor "
                    "environment.";
  }
  EXPECT_TRUE(has_mega_moe());
}

TEST(MegaMoeTest, RejectsNonMatrixInputBeforeLaunch) {
  const auto bf16 = torch::TensorOptions().dtype(torch::kBFloat16);
  const auto int32 = torch::TensorOptions().dtype(torch::kInt);
  const auto fp32 = torch::TensorOptions().dtype(torch::kFloat);
  const torch::Tensor context = torch::empty({1}, int32);
  const torch::Tensor x = torch::empty({2048}, bf16);
  const torch::Tensor topk_ids = torch::empty({1, 8}, int32);
  const torch::Tensor topk_weights = torch::empty({1, 8}, fp32);
  const std::vector<torch::Tensor> weight1 = {
      torch::empty({2048, 1024}, bf16)};
  const std::vector<torch::Tensor> weight2 = {
      torch::empty({512, 2048}, bf16)};

  EXPECT_THROW(apply_npu_mega_moe(context,
                                  x,
                                  topk_ids,
                                  topk_weights,
                                  weight1,
                                  weight2,
                                  256,
                                  16,
                                  1),
               c10::Error);
}

}  // namespace
}  // namespace xllm::kernel::npu

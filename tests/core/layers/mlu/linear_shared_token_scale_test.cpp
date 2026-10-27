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

#include <gtest/gtest.h>
#include <torch/torch.h>

#include <string>
#include <unordered_map>

#include "core/framework/config/parallel_config.h"
#include "core/layers/mlu/multi_device_test_utils.h"
#include "framework/parallel_state/collective_communicator.h"
#include "framework/state_dict/state_dict.h"
#include "layers/common/linear.h"
#include "platform/device.h"

namespace xllm::layer {
namespace {
void run_row_quantization(int32_t rank, int32_t port, bool compressed) {
  torch::set_num_threads(1);
  torch::NoGradGuard no_grad;
  Device dev(rank);
  dev.set_device();
  const torch::Device device = dev.unwrap();
  ParallelConfig::get_instance().kv_split_size(1);
  CollectiveCommunicator communicator(rank, 2, 1, 1, 1);
  communicator.create_process_groups("127.0.0.1:" + std::to_string(port),
                                     device);
  const ParallelArgs& parallel = *communicator.parallel_args();
  constexpr int64_t kInputs = 256;
  constexpr int64_t kOutputs = 128;
  const auto options =
      torch::TensorOptions().dtype(torch::kBFloat16).device(device);
  QuantArgs quant;
  quant.quant_method() = compressed ? "compressed-tensors" : "smoothquant";
  quant.bits() = 8;
  quant.is_sym() = true;
  quant.activation_dynamic() = true;
  quant.is_compressed_tensors_w8a8_dynamic() = compressed;
  quant.preserve_smooth() = compressed;
  LinearExtraArgs shared_scale_extra_args;
  shared_scale_extra_args.share_token_scale = true;
  RowParallelLinear full(kInputs,
                         kOutputs,
                         /*bias=*/false,
                         /*input_is_parallelized=*/true,
                         /*enable_result_reduction=*/false,
                         quant,
                         parallel.single_rank_group_,
                         options,
                         shared_scale_extra_args);
  RowParallelLinear sharded(kInputs,
                            kOutputs,
                            /*bias=*/false,
                            /*input_is_parallelized=*/true,
                            /*enable_result_reduction=*/false,
                            quant,
                            parallel.tp_group_,
                            options,
                            shared_scale_extra_args);
  RowParallelLinear local(kInputs,
                          kOutputs,
                          /*bias=*/false,
                          /*input_is_parallelized=*/true,
                          /*enable_result_reduction=*/false,
                          quant,
                          parallel.tp_group_,
                          options);
  RowParallelLinear scattered(kInputs,
                              kOutputs,
                              /*bias=*/false,
                              /*input_is_parallelized=*/false,
                              /*enable_result_reduction=*/true,
                              quant,
                              parallel.tp_group_,
                              options,
                              shared_scale_extra_args);
  ColumnParallelLinear column(kInputs,
                              kOutputs,
                              /*bias=*/false,
                              /*gather_output=*/true,
                              quant,
                              parallel.tp_group_,
                              options);
  torch::Tensor weight = torch::zeros({kOutputs, kInputs}, torch::kInt8);
  weight.index_put_({0, kInputs / 2}, 127);
  weight.index_put_({1, 0}, 1);
  StateDict weights(std::unordered_map<std::string, torch::Tensor>{
      {compressed ? "weight" : "qweight", weight},
      {compressed ? "weight_scale" : "per_channel_scale",
       torch::ones({kOutputs}, torch::kFloat32)},
      {"smooth", torch::ones({kInputs}, torch::kFloat32)}});
  full->load_state_dict(weights);
  sharded->load_state_dict(weights);
  local->load_state_dict(weights);
  scattered->load_state_dict(weights);
  column->load_state_dict(weights);
  torch::Tensor input = torch::zeros({4, kInputs}, torch::kBFloat16);
  input.index_put_({0, 0}, 1.0);
  input.index_put_({0, kInputs / 2}, 1.0 / 256.0);
  input.index_put_({1, 0}, 1.0 / 256.0);
  input.index_put_({1, kInputs / 2}, 1.0);
  input.index_put_({2, 0}, 1.0);
  input.index_put_({2, kInputs / 2}, 1.0);
  input = input.to(device);
  torch::Tensor expected = torch::zeros({4, kOutputs}, torch::kBFloat16);
  expected.index_put_({0, 1}, 1.0);
  expected.index_put_({1, 0}, 127.0);
  expected.index_put_({2, 0}, 127.0);
  expected.index_put_({2, 1}, 1.0);
  const torch::Tensor reference = full->forward(input);
  // Each output channel selects one input channel, so partial GEMM rounding
  // cannot hide a change in the per-token activation quantization domain.
  ASSERT_TRUE(torch::equal(reference.cpu(), expected));
  EXPECT_TRUE(torch::equal(column->forward(input).cpu(), expected));
  EXPECT_TRUE(torch::equal(scattered->forward(input).cpu(), expected));
  const torch::Tensor local_input =
      input.chunk(/*chunks=*/2, /*dim=*/1)[rank].contiguous();
  torch::Tensor local_expected = expected.clone();
  local_expected.index_put_({0, 0}, 127.0 / 256.0);
  local_expected.index_put_({1, 1}, 1.0 / 256.0);
  torch::Tensor local_output = local->forward(local_input);
  parallel.tp_group_->allreduce(local_output);
  EXPECT_TRUE(torch::equal(local_output.cpu(), local_expected));
  torch::Tensor actual = sharded->forward(local_input);
  parallel.tp_group_->allreduce(actual);
  dev.synchronize_default_stream();
  EXPECT_EQ(actual.scalar_type(), torch::kBFloat16);
  EXPECT_TRUE(torch::equal(actual.cpu(), expected))
      << "Row-parallel INT8 quantization changed the full-token scale domain: "
      << actual.cpu().slice(/*dim=*/1, /*start=*/0, /*end=*/2);
}

TEST(RowParallelLinearTest, PreservesGlobalTokenQuantization) {
  xllm::test::run_multi_device(2, [](int32_t rank, int32_t port) {
    run_row_quantization(rank, port, /*compressed=*/false);
  });
}

TEST(RowParallelLinearTest, PreservesGlobalTokenQuantizationCompressed) {
  xllm::test::run_multi_device(2, [](int32_t rank, int32_t port) {
    run_row_quantization(rank, port, /*compressed=*/true);
  });
}

}  // namespace
}  // namespace xllm::layer

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

#include "core/kernels/npu/atb_vision_attention.h"

#include <c10/core/DeviceGuard.h>
#include <glog/logging.h>

#include <limits>

#include "custom_functions_npu/atb_common.h"

namespace xllm::kernel::npu {

torch::Tensor atb_vision_attention(
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& value,
    const std::vector<int64_t>& cumulative_seq_lens,
    double scale) {
  CHECK_EQ(query.device().type(), c10::DeviceType::PrivateUse1);
  CHECK_EQ(query.device(), key.device());
  CHECK_EQ(query.device(), value.device());
  CHECK(query.scalar_type() == torch::kBFloat16 ||
        query.scalar_type() == torch::kFloat16);
  CHECK_EQ(query.scalar_type(), key.scalar_type());
  CHECK_EQ(query.scalar_type(), value.scalar_type());
  CHECK_EQ(query.dim(), 3) << "vision attention requires [T,N,D] tensors";
  CHECK(query.sizes() == key.sizes());
  CHECK(query.sizes() == value.sizes());
  CHECK_GT(query.size(1), 0);
  CHECK_GT(query.size(2), 0);
  CHECK_LE(query.size(0), std::numeric_limits<int32_t>::max());
  CHECK(!cumulative_seq_lens.empty());
  CHECK_EQ(cumulative_seq_lens.back(), query.size(0));

  const c10::DeviceGuard device_guard(query.device());
  torch::Tensor seq_lens = torch::empty(
      {static_cast<int64_t>(cumulative_seq_lens.size())},
      torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU));
  int32_t* lengths = seq_lens.data_ptr<int32_t>();
  int64_t previous = 0;
  for (size_t index = 0; index < cumulative_seq_lens.size(); ++index) {
    const int64_t end = cumulative_seq_lens[index];
    CHECK_GT(end, previous) << "vision frame lengths must be positive";
    lengths[index] = static_cast<int32_t>(end - previous);
    previous = end;
  }

  // Match the native Qwen vision encoder, including head_dim=72 without
  // padding. ATB consumes flattened TND inputs and per-frame host lengths.
  atb::infer::SelfAttentionParam param;
  param.headNum = static_cast<int32_t>(query.size(1));
  param.kvHeadNum = param.headNum;
  param.calcType = atb::infer::SelfAttentionParam::PA_ENCODER;
  param.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_UNDEFINED;
  param.qkScale = scale;
  auto& cache =
      atb::OpParamCache<atb::infer::SelfAttentionParam>::getInstance();
  atb::Operation* operation =
      cache.get_operation(param, "SelfAttentionOperation");

  torch::Tensor output = torch::empty(query.sizes(), query.options());
  torch::Tensor flat_output = output.view({query.size(0), -1});
  atb::ParamSetter setter;
  setter.Input(query.flatten(1), /*format_trans=*/true)
      .Input(key.flatten(1), /*format_trans=*/true)
      .Input(value.flatten(1), /*format_trans=*/true)
      .Input(seq_lens)
      .Output(flat_output);
  atb::run_atb_cmd(operation, setter, "SelfAttentionOperation");
  return output;
}

}  // namespace xllm::kernel::npu

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

#include <cstdint>
#include <cstring>

#include "npu_ops_api.h"

namespace xllm::kernel::npu {

namespace {

constexpr int64_t kPackedInt8InInt32 = 4;

bool is_defined(const std::optional<at::Tensor>& tensor) {
  return tensor.has_value() && tensor->defined();
}

int64_t get_tensor_npu_format(const at::Tensor& tensor) {
#ifdef TORCH_HIGHER_THAN_PTA6
  return at_npu::native::get_npu_format(tensor);
#else
  return at_npu::native::NPUNativeFunctions::get_npu_format(tensor);
#endif
}

at::Tensor npu_format_cast(const at::Tensor& tensor, int64_t format) {
#ifdef TORCH_HIGHER_THAN_PTA6
  return at_npu::native::npu_format_cast(tensor, format);
#else
  return at_npu::native::NPUNativeFunctions::npu_format_cast(tensor, format);
#endif
}

int64_t fp32_bits_to_int64(float value, bool use_even_uint32_buffer_layout) {
  uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));

  if (!use_even_uint32_buffer_layout) {
    return static_cast<int64_t>(bits);
  }

  // vllm-ascend writes scale bits into the even uint32 slots of a temporary
  // uint32 buffer, leaves the odd slots zero, then views that buffer as int64.
  uint32_t words[2] = {bits, 0};
  int64_t packed = 0;
  std::memcpy(&packed, words, sizeof(packed));
  return packed;
}

at::Tensor fp32_bits_to_int64_tensor(const at::Tensor& float_tensor,
                                     bool use_even_uint32_buffer_layout) {
  auto cpu_float =
      float_tensor.to(at::TensorOptions().dtype(at::kFloat).device(at::kCPU))
          .contiguous();
  auto cpu_bits =
      at::empty(cpu_float.sizes(), cpu_float.options().dtype(at::kLong));

  const auto* input = cpu_float.data_ptr<float>();
  auto* output = cpu_bits.data_ptr<int64_t>();
  const auto numel = cpu_float.numel();
  for (int64_t i = 0; i < numel; ++i) {
    output[i] = fp32_bits_to_int64(input[i], use_even_uint32_buffer_layout);
  }
  return cpu_bits.to(float_tensor.device()).contiguous();
}

at::Tensor process_scale(const at::Tensor& weight,
                         const at::Tensor& scale,
                         const std::optional<at::Tensor>& per_group_scale,
                         bool is_per_channel_weight) {
  TORCH_CHECK(weight.dim() == 3,
              "W4A8_DYNAMIC preprocess expects 3D expert weight, got ",
              weight.sizes());
  TORCH_CHECK(scale.dim() == 3,
              "W4A8_DYNAMIC preprocess expects 3D weight scale, got ",
              scale.sizes());
  TORCH_CHECK(weight.size(0) == scale.size(0),
              "W4A8_DYNAMIC preprocess expert count mismatch. weight=",
              weight.sizes(),
              ", scale=",
              scale.sizes());

  auto transposed_scale = scale.transpose(1, 2).contiguous();
  const auto logical_n = weight.size(2) * 2;
  TORCH_CHECK(transposed_scale.size(2) == logical_n,
              "W4A8_DYNAMIC preprocess scale output dim mismatch. weight=",
              weight.sizes(),
              ", scale=",
              scale.sizes(),
              ", expected logical_n=",
              logical_n);
  if (is_per_channel_weight) {
    return fp32_bits_to_int64_tensor(transposed_scale, false);
  }

  TORCH_CHECK(per_group_scale.has_value() && per_group_scale->defined(),
              "W4A8_DYNAMIC per-group preprocess requires "
              "weight_scale_second when group_size > 0.");
  auto transposed_per_group_scale =
      per_group_scale.value().transpose(1, 2).contiguous();

  const auto group_num = weight.size(0);
  const auto n = logical_n;
  TORCH_CHECK(transposed_per_group_scale.numel() % (group_num * n) == 0,
              "Invalid W4A8_DYNAMIC per-group scale shape. scale_second=",
              transposed_per_group_scale.sizes(),
              ", weight=",
              weight.sizes(),
              ", restored_n=",
              n);
  transposed_per_group_scale =
      transposed_per_group_scale.reshape({group_num, -1, n});

  auto scale_fp32 = (transposed_scale.to(at::kFloat) *
                     transposed_per_group_scale.to(at::kFloat))
                        .to(at::kHalf)
                        .to(at::kFloat);
  return fp32_bits_to_int64_tensor(scale_fp32, true);
}

at::Tensor maybe_trans_nz(const at::Tensor& weight) {
  if (weight.device().is_cpu()) {
    return weight;
  }
  if (get_tensor_npu_format(weight) == ACL_FORMAT_FRACTAL_NZ) {
    return weight;
  }
  return npu_format_cast(weight, ACL_FORMAT_FRACTAL_NZ);
}

at::Tensor pack_to_int32(const at::Tensor& weight) {
  TORCH_CHECK(weight.size(-1) % kPackedInt8InInt32 == 0,
              "W4A8_DYNAMIC version 1.0.0 weight last dim must be "
              "divisible by 4 before dtype view to int32, got shape ",
              weight.sizes());
  return weight.view(at::kInt).contiguous();
}

}  // namespace

std::tuple<at::Tensor,
           at::Tensor,
           at::Tensor,
           at::Tensor,
           std::optional<at::Tensor>,
           std::optional<at::Tensor>>
w4a8_dynamic_moe_preprocess(
    const at::Tensor& w13_weight,
    const at::Tensor& w2_weight,
    const at::Tensor& w13_weight_scale,
    const at::Tensor& w2_weight_scale,
    const std::optional<at::Tensor>& w13_weight_scale_second,
    const std::optional<at::Tensor>& w2_weight_scale_second,
    const std::optional<at::Tensor>& w13_scale_bias,
    const std::optional<at::Tensor>& w2_scale_bias,
    int64_t group_size) {
  TORCH_CHECK(group_size >= 0,
              "W4A8_DYNAMIC group_size must be >= 0, got ",
              group_size);
  const bool is_per_channel_weight = group_size == 0;

  auto processed_w13 = w13_weight.transpose(1, 2).contiguous();
  auto processed_w2 = w2_weight.transpose(1, 2).contiguous();

  auto processed_w13_scale = process_scale(processed_w13,
                                           w13_weight_scale,
                                           w13_weight_scale_second,
                                           is_per_channel_weight);
  auto processed_w2_scale = process_scale(processed_w2,
                                          w2_weight_scale,
                                          w2_weight_scale_second,
                                          is_per_channel_weight);

  TORCH_CHECK(is_defined(w13_scale_bias),
              "W4A8_DYNAMIC version 1.0.0 requires w13_scale_bias.");
  TORCH_CHECK(is_defined(w2_scale_bias),
              "W4A8_DYNAMIC version 1.0.0 requires w2_scale_bias.");
  auto processed_w13_scale_bias =
      w13_scale_bias.value().transpose(1, 2).contiguous().sum(1);
  auto processed_w2_scale_bias =
      w2_scale_bias.value().transpose(1, 2).contiguous().sum(1);

  processed_w13 = maybe_trans_nz(processed_w13);
  processed_w2 = maybe_trans_nz(processed_w2);
  processed_w13 = pack_to_int32(processed_w13);
  processed_w2 = pack_to_int32(processed_w2);

  return std::make_tuple(processed_w13,
                         processed_w2,
                         processed_w13_scale,
                         processed_w2_scale,
                         processed_w13_scale_bias,
                         processed_w2_scale_bias);
}

}  // namespace xllm::kernel::npu

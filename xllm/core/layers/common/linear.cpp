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

#include "linear.h"

#include <glog/logging.h>
#include <torch/torch.h>

#include <algorithm>
#include <cctype>

#include "core/layers/common/quant_utils.h"
#include "framework/parallel_state/parallel_args.h"
#include "framework/parallel_state/parallel_state.h"
#include "kernels/ops_api.h"

namespace xllm {
namespace layer {

namespace {

// ============================================================================
// FP8 Fused Weight Utilities
// ============================================================================
// Unlike INT8/SmoothQuant (per-channel), FP8 usually uses per-tensor scaling.
// When fusing separate layers (e.g., gate_proj + up_proj) into one, we cannot
// simply concatenate them if they have different scaling factors. We must
// requantize all partitions to align with a single global max_scale.

struct Fp8PartitionInfo {
  std::vector<float> scales;
  std::vector<int64_t> logical_widths;

  bool empty() const { return scales.empty(); }
  size_t size() const { return scales.size(); }
};

inline float compute_max_scale(const std::vector<float>& scales) {
  if (scales.empty()) {
    return 1.0f;
  }
  return *std::max_element(scales.begin(), scales.end());
}

// Detect if the checkpoint contains valid separate scales for each partition.
// The check on the last element serves as a heuristic to ensure the scales
// are fully populated and not just initialized to a sentinel/minimum value.
inline bool is_unfused_checkpoint(const std::vector<float>& scales) {
  return scales.size() > 1 &&
         scales.back() > std::numeric_limits<float>::lowest();
}

bool is_fp8_dtype(torch::ScalarType dtype) {
  return dtype == torch::kFloat8_e4m3fn || dtype == torch::kFloat8_e5m2;
}

// Realigns FP8 partitions to a unified global scale to enable fusion.
// Logic:
// 1. Recover original values (FP8 -> FP16) using partition-specific scales.
// 2. Re-quantize (FP16 -> FP8) using the new global max_scale.
void requantize_fp8_weight(torch::Tensor& weight,
                           const std::vector<float>& partition_scales,
                           const std::vector<int64_t>& logical_widths,
                           float max_scale) {
  if (partition_scales.size() != logical_widths.size()) {
    return;
  }

  int64_t start = 0;
  for (size_t idx = 0; idx < logical_widths.size(); ++idx) {
    int64_t logical_width = logical_widths[idx];
    if (logical_width == 0) {
      continue;
    }
    int64_t end = start + logical_width;

    // Dequantize: FP8 -> FP16 with original scale
    auto weight_slice = weight.slice(0, start, end);
    auto weight_fp16 = weight_slice.to(torch::kFloat16) * partition_scales[idx];

    // Requantize: FP16 -> FP8 with unified max_scale
    auto scale_tensor = torch::tensor(
        {max_scale}, weight_fp16.options().dtype(torch::kFloat32));
    auto weight_quantized =
        torch::empty_like(weight_slice, torch::kFloat8_e4m3fn);

    xllm::kernel::StaticScaledFp8QuantParams quant_params;
    quant_params.output = weight_quantized;
    quant_params.input = weight_fp16;
    quant_params.scale = scale_tensor;
    xllm::kernel::static_scaled_fp8_quant(quant_params);

    weight.slice(0, start, end).copy_(weight_quantized);
    start = end;
  }
}

// Load max input scale from multiple prefixes
torch::Tensor load_max_input_scale(const StateDict& state_dict,
                                   const std::vector<std::string>& prefixes) {
  torch::Tensor max_scale;
  for (const auto& prefix : prefixes) {
    auto scale_tensor = state_dict.get_tensor(prefix + "input_scale");
    if (scale_tensor.defined()) {
      auto scale_val = scale_tensor.flatten().max();
      if (!max_scale.defined()) {
        max_scale = scale_val;
      } else {
        max_scale = torch::max(max_scale, scale_val);
      }
    }
  }
  return max_scale;
}

// ============================================================================
// FP8 Forward Helper
// ============================================================================
// Performs FP8 W8A8 quantized linear: input quantization + scaled matmul.
// Consolidates repeated logic from Column/QKV/RowParallelLinear forward paths.
//
// Performance Optimization:
// - If input is already FP8 (from fused RMSNorm+FP8 quantization), skip
//   quantization step and use input_scale directly. This avoids redundant
//   memory reads/writes.
// - For non-FP8 inputs, quantization is performed based on input_scale:
//   - input_scale provided: static quantization (faster, no absmax compute)
//   - input_scale not provided: dynamic quantization (computes absmax)

torch::Tensor fp8_linear_forward(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& weight_scale,
    const std::optional<torch::Tensor>& input_scale,
    const std::optional<torch::Tensor>& bias,
    at::ScalarType output_dtype) {
  // Flatten input to 2D for matmul
  auto input_2d = input.view({-1, input.size(-1)});

  torch::Tensor quantized_input;
  torch::Tensor a_scale;

  // Check if input is already FP8 quantized (from fused RMSNorm+FP8)
  if (input.dtype() == torch::kFloat8_e4m3fn) {
    // Input is already FP8, use directly (skip quantization)
    // This is the fast path when using fused RMSNorm+FP8 quantization
    CHECK(input_scale.has_value())
        << "input_scale must be provided when input is already FP8";
    quantized_input = input_2d;
    a_scale = input_scale.value();
  } else {
    // Input is not FP8, perform quantization
    // (static if input_scale provided, dynamic otherwise)
    xllm::kernel::Fp8ScaledQuantizeParams quantize_params;
    quantize_params.input = input_2d;
    quantize_params.output = std::nullopt;
    quantize_params.scale = input_scale;

    std::tie(quantized_input, a_scale) =
        xllm::kernel::fp8_scaled_quantize(quantize_params);
  }

  // FP8 scaled matmul
  xllm::kernel::Fp8ScaledMatmulParams matmul_params;
  matmul_params.a = quantized_input;
  matmul_params.b = weight;
  matmul_params.a_scale = a_scale;
  matmul_params.b_scale = weight_scale;
  matmul_params.bias = bias;
  matmul_params.output = std::nullopt;
  matmul_params.output_dtype = output_dtype;
  matmul_params.input_shape = input.sizes().vec();

  return xllm::kernel::fp8_scaled_matmul(matmul_params);
}

std::string to_lower_copy(std::string value) {
  std::transform(
      value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
      });
  return value;
}

bool is_w8a8_dynamic_quant(
    const std::optional<std::string>& resolved_weight_quant_method) {
  return resolved_weight_quant_method.has_value() &&
         resolved_weight_quant_method.value() == "w8a8_dynamic";
}

bool is_w8a8_quant(
    const std::optional<std::string>& resolved_weight_quant_method) {
  return resolved_weight_quant_method.has_value() &&
         resolved_weight_quant_method.value() == "w8a8";
}

bool is_fp8_channelwise_w8a8(const QuantArgs& quant_args) {
#if defined(USE_DCU)
  // Compressed-tensors FP8 W8A8 currently pairs dynamic activations with
  // per-output-channel weight scales on DCU.
  return quant_args.quant_method() == kQuantMethodFp8 &&
         quant_args.activation_dynamic();
#else
  (void)quant_args;
  return false;
#endif
}

void check_fp8_activation_dynamic_supported(const QuantArgs& quant_args) {
  if (!quant_args.activation_dynamic()) {
    return;
  }
  CHECK(is_fp8_channelwise_w8a8(quant_args))
      << "DCU FP8 currently supports only compressed-tensors dynamic "
         "channelwise W8A8.";
}

torch::Dtype get_w8a8_deq_scale_dtype(const torch::TensorOptions& options) {
  const torch::Dtype dtype = c10::typeMetaToScalarType(options.dtype());
  if (dtype == torch::kFloat16) {
    return torch::kInt64;
  }
  if (dtype == torch::kBFloat16) {
    return torch::kFloat32;
  }
  LOG(WARNING) << "W8A8 deq_scale defaults to float32 for dtype " << dtype;
  return torch::kFloat32;
}

struct W8A8LinearParamRefs {
  torch::Tensor& weight;
  bool& weight_is_loaded;
  torch::Tensor& input_scale;
  bool& input_scale_is_loaded;
  torch::Tensor& input_offset;
  bool& input_offset_is_loaded;
  torch::Tensor& deq_scale;
  bool& deq_scale_is_loaded;
  torch::Tensor& quant_bias;
  bool& quant_bias_is_loaded;
  torch::Tensor& weight_scale;
  bool& weight_scale_is_loaded;
  torch::Tensor& weight_offset;
  bool& weight_offset_is_loaded;
};

void ensure_w8a8_params_for_linear_load(
    torch::nn::Module* module,
    const QuantArgs& quant_args,
    const torch::TensorOptions& options,
    const std::optional<std::string>& resolved_weight_quant_method,
    int64_t shared_input_param_size,
    W8A8LinearParamRefs refs) {
  std::vector<weight::LazyParameterSpec> specs;
  auto push = [&](torch::Tensor& tensor,
                  bool& tensor_is_loaded,
                  const char* name,
                  std::vector<int64_t> sizes,
                  const torch::TensorOptions& tensor_options) {
    specs.push_back(weight::LazyParameterSpec{
        &tensor, &tensor_is_loaded, name, std::move(sizes), tensor_options});
  };

  if (!is_w8a8_quant(resolved_weight_quant_method) &&
      !is_w8a8_dynamic_quant(resolved_weight_quant_method)) {
    if (!quant_args.quant_descs().empty() ||
        quant_args.is_compressed_tensors_w8a8_dynamic()) {
      // Quant args indicated a checkpoint that may be quantized, so the
      // constructor initialized weights as kInt8. If the actual checkpoint is
      // not resolved to a W8A8 method, re-register the weight in the original
      // dtype so the subsequent load can copy checkpoint weights correctly.
      CHECK(refs.weight.defined())
          << "weight must be registered before lazy quant fallback";
      const int64_t out_features = refs.weight.size(0);
      const int64_t in_features = refs.weight.size(1);
      specs.reserve(1);
      push(refs.weight,
           refs.weight_is_loaded,
           "weight",
           {out_features, in_features},
           options);
      weight::ensure_parameter_storage(module, specs);
    }
    return;
  }

  CHECK(refs.weight.defined())
      << "weight must be registered before lazy quant init";
  const int64_t out_features = refs.weight.size(0);
  const int64_t in_features = refs.weight.size(1);

  specs.reserve(4);
  if (is_w8a8_quant(resolved_weight_quant_method)) {
    push(refs.input_scale,
         refs.input_scale_is_loaded,
         "input_scale",
         {shared_input_param_size},
         options.dtype(torch::kFloat32));
    push(refs.input_offset,
         refs.input_offset_is_loaded,
         "input_offset",
         {shared_input_param_size},
         options.dtype(torch::kInt8));
    push(refs.deq_scale,
         refs.deq_scale_is_loaded,
         "deq_scale",
         {out_features},
         options.dtype(get_w8a8_deq_scale_dtype(options)));
    push(refs.quant_bias,
         refs.quant_bias_is_loaded,
         "quant_bias",
         {out_features},
         options.dtype(torch::kInt32));
  } else {
    push(refs.weight_scale,
         refs.weight_scale_is_loaded,
         "weight_scale",
         {out_features},
         options.dtype(torch::kFloat32));
    push(refs.weight_offset,
         refs.weight_offset_is_loaded,
         "weight_offset",
         {out_features},
         options.dtype(torch::kFloat32));
  }
  weight::ensure_parameter_storage(module, specs);
}

bool tensors_allclose_as_fp32(const torch::Tensor& lhs,
                              const torch::Tensor& rhs) {
  return torch::allclose(lhs.to(torch::kFloat32), rhs.to(torch::kFloat32));
}

bool load_shared_tensor_from_prefixes_or_fail(
    const StateDict& state_dict,
    const std::vector<std::string>& prefixes,
    const std::string& name,
    torch::Tensor& tensor,
    bool& tensor_is_loaded) {
  // W8A8 fused input_scale/offset shoul be same
  if (tensor_is_loaded || !tensor.defined()) {
    return tensor_is_loaded;
  }
  torch::Tensor first_candidate;
  std::string first_prefix;
  for (const auto& prefix : prefixes) {
    auto candidate = state_dict.get_tensor(prefix + name);
    if (!candidate.defined()) {
      continue;
    }
    auto flattened = candidate.flatten();
    if (!first_candidate.defined()) {
      first_candidate = flattened;
      first_prefix = prefix;
      continue;
    }
    CHECK_EQ(flattened.sizes(), first_candidate.sizes())
        << "Shared tensor size for " << name << ": prefix '" << prefix
        << "' has shape " << flattened.sizes() << ", but prefix '"
        << first_prefix << "' has shape " << first_candidate.sizes() << ".";
    CHECK(tensors_allclose_as_fp32(flattened, first_candidate))
        << "Shared tensor value for " << name << ": prefix '" << prefix
        << "' differs from prefix '" << first_prefix << "'.";
  }
  if (!first_candidate.defined()) {
    return false;
  }
  CHECK_EQ(first_candidate.numel(), tensor.numel())
      << "Tensor size mismatch for shared: " << state_dict.prefix() << name;
  tensor.copy_(first_candidate.view(tensor.sizes()));
  tensor_is_loaded = true;
  return true;
}

void collapse_shared_tensor_to_scalar_or_fail(torch::Tensor& tensor,
                                              const char* name) {
  // W8A8 fused input_scale/offset shoul be same
  CHECK(tensor.defined()) << name << " must be defined.";
  CHECK_GT(tensor.numel(), 0) << name << " must contain at least one element.";
  if (tensor.numel() <= 1) {
    return;
  }
  auto flattened = tensor.flatten();
  auto first = flattened.slice(0, 0, 1).expand_as(flattened);
  CHECK(tensors_allclose_as_fp32(flattened, first))
      << "Shared tensor value for " << name
      << " in fused static W8A8 should be same.";
  tensor = tensor.flatten().slice(0, 0, 1);
}

torch::Tensor npu_w8a8_linear_forward(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& input_scale,
    const torch::Tensor& input_offset,
    const torch::Tensor& deq_scale,
    const std::optional<torch::Tensor>& quant_bias,
    at::ScalarType output_dtype) {
  xllm::kernel::NpuQuantizeParams quant_params;
  quant_params.input = input;
  quant_params.scale = input_scale;
  quant_params.zero_point = input_offset;
  // quant_params.output_dtype = at::ScalarType::QInt8;
  quant_params.axis = -1;

  auto quantized_input = xllm::kernel::quantize(quant_params);

  xllm::kernel::QuantMatmulParams quant_matmul_params;
  quant_matmul_params.x1 = quantized_input;
  quant_matmul_params.x2 = weight;
  quant_matmul_params.transpose2 = true;
  quant_matmul_params.scale = deq_scale;
  quant_matmul_params.bias = quant_bias;
  quant_matmul_params.output_dtype = output_dtype;

  return xllm::kernel::quant_matmul(quant_matmul_params);
}

#if defined(USE_DCU)
torch::Tensor dcu_w8a8_dynamic_linear_forward(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& weight_scale,
    const std::optional<torch::Tensor>& bias,
    at::ScalarType output_dtype) {
  xllm::kernel::ScaledQuantizeParams quantize_params;
  quantize_params.x = input;
  quantize_params.smooth = torch::Tensor();  // no smooth factor

  torch::Tensor quantized_input;
  torch::Tensor input_scale;
  std::tie(quantized_input, input_scale) =
      xllm::kernel::scaled_quantize(quantize_params);

  xllm::kernel::ScaledMatmulParams matmul_params;
  matmul_params.a = quantized_input;
  matmul_params.b = weight;
  matmul_params.a_scale = input_scale;
  matmul_params.b_scale = weight_scale;
  matmul_params.output_dtype = output_dtype;
  matmul_params.bias = bias;
  matmul_params.beta = 0.0;
  matmul_params.a_quant_bit_size = 8;

  return xllm::kernel::scaled_matmul(matmul_params);
}
#endif  // USE_DCU

}  // namespace

ColumnParallelLinearImpl::ColumnParallelLinearImpl(const ModelContext& context)
    : ColumnParallelLinearImpl(
          context.get_model_args().hidden_size(),
          context.get_model_args().vocab_size(),
          /*bias=*/false,
          /*gather_output=*/true,
          QuantArgs{},  // do not use quantization for lm_head
          context.get_parallel_args().lm_head_group_ != nullptr
              ? context.get_parallel_args().lm_head_group_
              : context.get_parallel_args().tp_group_,
          context.get_tensor_options()) {}

// Linear layer with column parallelism.
ColumnParallelLinearImpl::ColumnParallelLinearImpl(
    int64_t in_features,
    int64_t out_features,
    bool bias,
    bool gather_output,
    const QuantArgs& quant_args,
    ProcessGroup* process_group,
    const torch::TensorOptions& options,
    const LinearExtraArgs& linear_extra_args,
    int32_t output_replicas)
    : gather_output_(gather_output),
      device_(options.device()),
      process_group_(process_group),
      quant_args_(quant_args),
      options_(options),
      linear_extra_args_(linear_extra_args),
      output_dtype_(c10::typeMetaToScalarType(options.dtype())) {
  rank_ = process_group_->rank();
  world_size_ = process_group_->world_size();
  int32_t valid_output_replicas = output_replicas;
  if (valid_output_replicas <= 0 || world_size_ % valid_output_replicas != 0 ||
      (valid_output_replicas != 1 && gather_output)) {
    valid_output_replicas = 1;
  }
  weight_rank_ = rank_ / valid_output_replicas;
  weight_world_size_ = world_size_ / valid_output_replicas;
  CHECK(out_features % weight_world_size_ == 0)
      << "out_features " << out_features
      << " not divisible by weight_world_size " << weight_world_size_;
  const int64_t out_features_per_partition = out_features / weight_world_size_;
  // Note: torch.nn.functional.linear performs XA^T + b and as a result
  // we allocate the transpose.
  if (quant_args_.quant_method() == kQuantMethodSmoothquant) {
    qweight_ = register_parameter(
        "qweight",
        torch::empty({out_features_per_partition, in_features},
                     options.dtype(torch::kInt8)),
        /*requires_grad=*/false);
    per_channel_scale_ =
        register_parameter("per_channel_scale",
                           torch::empty({out_features_per_partition},
                                        options.dtype(torch::kFloat32)),
                           /*requires_grad=*/false);
    smooth_ = register_parameter(
        "smooth",
        torch::empty({in_features}, options.dtype(torch::kFloat32)),
        /*requires_grad=*/false);
    // output dtype for scaled_matmul
    output_dtype_ = c10::typeMetaToScalarType(options.dtype());
  } else if (quant_args_.quant_method() == kQuantMethodFp8) {
    // FP8 W8A8 quantization - weight is stored as FP8 (float8_e4m3fn)
    weight_ = register_parameter(
        "weight",
        torch::empty({out_features_per_partition, in_features},
                     options.dtype(torch::kFloat8_e4m3fn)),
        /*requires_grad=*/false);
    const int64_t weight_scale_size =
        is_fp8_channelwise_w8a8(quant_args_) ? out_features_per_partition : 1;
    weight_scale_ = register_parameter(
        "weight_scale",
        torch::empty({weight_scale_size}, options.dtype(torch::kFloat32)),
        /*requires_grad=*/false);
    // For static activation quantization, input_scale is pre-computed
    if (!quant_args_.activation_dynamic()) {
      input_scale_ =
          register_parameter("input_scale",
                             torch::empty({1}, options.dtype(torch::kFloat32)),
                             /*requires_grad=*/false);
    }
  } else if (!quant_args_.quant_descs().empty() ||
             quant_args_.is_compressed_tensors_w8a8_dynamic()) {
    // quant_descs is not empty: default initialize weight as kInt8.
    // During load_state_dict, the weight will be lazily re-registered to the
    // appropriate dtype based on the resolved quant method.
    weight_ = register_parameter(
        "weight",
        torch::empty({out_features_per_partition, in_features},
                     options.dtype(torch::kInt8)),
        /*requires_grad=*/false);
  } else {
    weight_ = register_parameter(
        "weight",
        torch::empty({out_features_per_partition, in_features}, options),
        /*requires_grad=*/false);
  }

  if (bias) {
    bias_ =
        register_parameter("bias",
                           torch::empty({out_features_per_partition}, options),
                           /*requires_grad=*/false);
  }
}

torch::Tensor ColumnParallelLinearImpl::forward(torch::Tensor input) {
  input = input.to(device_);
  auto bias =
      bias_.defined() ? std::optional<torch::Tensor>(bias_) : std::nullopt;
  torch::Tensor output;

  if (quant_args_.quant_method() == kQuantMethodSmoothquant) {
    CHECK(qweight_.defined()) << "qweight is required for smoothquant.";
    CHECK(per_channel_scale_.defined())
        << "per_channel_scale is required for smoothquant.";

    torch::Tensor quantized_input;
    torch::Tensor input_scale;

    xllm::kernel::ScaledQuantizeParams quantize_params;
    quantize_params.x = input;
    quantize_params.smooth = smooth_;
    quantize_params.zero = std::nullopt;
    quantize_params.token_count = std::nullopt;
    quantize_params.gather_index = std::nullopt;
    quantize_params.gather_index_start_position = std::nullopt;
    quantize_params.output = std::nullopt;
    quantize_params.output_scale = std::nullopt;
    quantize_params.act_mode = linear_extra_args_.act_mode;
    quantize_params.active_coef = 1.0;
    quantize_params.is_gated = linear_extra_args_.is_gated;

    std::tie(quantized_input, input_scale) =
        xllm::kernel::scaled_quantize(quantize_params);

    xllm::kernel::ScaledMatmulParams matmul_params;
    matmul_params.a = quantized_input;
    matmul_params.b = qweight_;
    matmul_params.a_scale = input_scale;
    matmul_params.b_scale = per_channel_scale_;
    matmul_params.output_dtype = output_dtype_;
    matmul_params.bias = bias;
    matmul_params.c = std::nullopt;
    matmul_params.act_mode = "none";
    matmul_params.quant_bit_size = 8;
    matmul_params.alpha = 1.0;
    matmul_params.beta = 0.0;
    matmul_params.use_hp_active = false;
    matmul_params.a_quant_bit_size = 8;
    matmul_params.a_calib = std::nullopt;
    matmul_params.b_calib = std::nullopt;
    matmul_params.output = std::nullopt;

    output = xllm::kernel::scaled_matmul(matmul_params);
  } else if (quant_args_.quant_method() == kQuantMethodFp8) {
    check_fp8_activation_dynamic_supported(quant_args_);
    auto scale = input_scale_.defined()
                     ? std::optional<torch::Tensor>(input_scale_)
                     : std::nullopt;
    output = fp8_linear_forward(
        input, weight_, weight_scale_, scale, bias, output_dtype_);
  } else if (is_w8a8_quant(resolved_weight_quant_method_)) {
    CHECK(input_scale_is_loaded_ && input_scale_.defined())
        << "input_scale is required for w8a8 quant matmul.";
    CHECK(input_offset_is_loaded_ && input_offset_.defined())
        << "input_offset is required for w8a8 quant matmul.";
    CHECK(deq_scale_is_loaded_ && deq_scale_.defined())
        << "deq_scale is required for w8a8 quant matmul.";
    auto quant_bias = quant_bias_is_loaded_ && quant_bias_.defined()
                          ? std::optional<torch::Tensor>(quant_bias_)
                          : std::nullopt;
    output = npu_w8a8_linear_forward(input,
                                     weight_,
                                     input_scale_,
                                     input_offset_,
                                     deq_scale_,
                                     quant_bias,
                                     output_dtype_);
  } else if (is_w8a8_dynamic_quant(resolved_weight_quant_method_)) {
    auto weight_scale = weight_scale_is_loaded_
                            ? std::optional<torch::Tensor>(weight_scale_)
                            : std::nullopt;
    CHECK(weight_scale.has_value() && weight_scale.value().defined())
        << "weight_scale is required for w8a8_dynamic quant matmul.";
#if defined(USE_DCU)
    output = dcu_w8a8_dynamic_linear_forward(
        input, weight_, weight_scale.value(), bias, output_dtype_);
#elif defined(USE_NPU)
    output = npu_w8a8_dynamic_linear_forward(
        input, weight_, weight_scale.value(), bias, output_dtype_);
#endif
  } else {
    xllm::kernel::MatmulParams matmul_params;
    matmul_params.a = input;
    matmul_params.b = weight_;
    matmul_params.bias = bias;
    output = xllm::kernel::matmul(matmul_params);
  }

  if (world_size_ > 1 && gather_output_) {
    output = xllm::parallel_state::gather(output, process_group_);
  }
  return output;
}

bool ColumnParallelLinearImpl::uses_w8a8_dynamic_quant() const {
  return is_w8a8_dynamic_quant(resolved_weight_quant_method_);
}

torch::Tensor ColumnParallelLinearImpl::w8a8_dynamic_weight_scale() const {
  CHECK(uses_w8a8_dynamic_quant())
      << "w8a8_dynamic_weight_scale requires w8a8_dynamic quant method.";
  CHECK(weight_scale_is_loaded_ && weight_scale_.defined())
      << "weight_scale is required for w8a8_dynamic quant matmul.";
  return weight_scale_;
}

std::optional<torch::Tensor> ColumnParallelLinearImpl::bias() const {
  if (bias_.defined()) {
    return bias_;
  }
  return std::nullopt;
}

// load the weight from the checkpoint
void ColumnParallelLinearImpl::load_state_dict(const StateDict& state_dict) {
  if (state_dict.size() == 0) {
    return;
  }
  const int64_t rank = weight_world_size_ == 1 ? 0 : weight_rank_;
  const int64_t world_size = weight_world_size_;
  resolve_weight_quant_method_for_linear_load(
      quant_args_, state_dict, nullptr, resolved_weight_quant_method_);
  ensure_w8a8_params_for_linear_load(
      this,
      quant_args_,
      options_,
      resolved_weight_quant_method_,
      /*shared_input_param_size=*/1,
      W8A8LinearParamRefs{weight_,
                          weight_is_loaded_,
                          input_scale_,
                          input_scale_is_loaded_,
                          input_offset_,
                          input_offset_is_loaded_,
                          deq_scale_,
                          deq_scale_is_loaded_,
                          quant_bias_,
                          quant_bias_is_loaded_,
                          weight_scale_,
                          weight_scale_is_loaded_,
                          weight_offset_,
                          weight_offset_is_loaded_});

  // load and merge the weights on dim 0
  // If quant_args_ indicates SmoothQuant, load qweight; otherwise, load
  // normal weight
  if (quant_args_.quant_method() == kQuantMethodSmoothquant) {
    LOAD_SHARDED_WEIGHT(qweight, 0);
    LOAD_SHARDED_WEIGHT(per_channel_scale, 0);
    // for input, there is one smooth value
    LOAD_WEIGHT(smooth);
  } else if (quant_args_.quant_method() == kQuantMethodFp8) {
    // FP8 quantization: load FP8 weight and scales
    LOAD_SHARDED_WEIGHT(weight, 0);
    if (is_fp8_channelwise_w8a8(quant_args_)) {
      LOAD_SHARDED_WEIGHT(weight_scale, 0);
    } else {
      LOAD_WEIGHT(weight_scale);
    }
    // For static activation quantization, load input_scale
    if (!quant_args_.activation_dynamic() && input_scale_.defined()) {
      LOAD_WEIGHT(input_scale);
    }
  } else if (is_w8a8_quant(resolved_weight_quant_method_)) {
    LOAD_SHARDED_WEIGHT(weight, 0);
    LOAD_WEIGHT(input_scale);
    LOAD_WEIGHT(input_offset);
    LOAD_SHARDED_WEIGHT(deq_scale, 0);
    LOAD_SHARDED_WEIGHT(quant_bias, 0);
  } else if (is_w8a8_dynamic_quant(resolved_weight_quant_method_)) {
    LOAD_SHARDED_WEIGHT(weight, 0);
    LOAD_SHARDED_WEIGHT(weight_scale, 0);
    if (weight_offset_.defined()) {
      LOAD_SHARDED_WEIGHT(weight_offset, 0);
    }
  } else {
    LOAD_SHARDED_WEIGHT(weight, 0);
  }

  if (bias_.defined()) {
    LOAD_SHARDED_WEIGHT(bias, 0);
  }
}

// special load_state_dict for fused cases
void ColumnParallelLinearImpl::load_state_dict(
    const StateDict& state_dict,
    const std::vector<std::string>& prefixes) {
  if (state_dict.size() == 0) {
    return;
  }
  const int64_t rank = weight_world_size_ == 1 ? 0 : weight_rank_;
  const int64_t world_size = weight_world_size_;
  resolve_weight_quant_method_for_linear_load(
      quant_args_, state_dict, &prefixes, resolved_weight_quant_method_);
  ensure_w8a8_params_for_linear_load(
      this,
      quant_args_,
      options_,
      resolved_weight_quant_method_,
      /*shared_input_param_size=*/1,
      W8A8LinearParamRefs{weight_,
                          weight_is_loaded_,
                          input_scale_,
                          input_scale_is_loaded_,
                          input_offset_,
                          input_offset_is_loaded_,
                          deq_scale_,
                          deq_scale_is_loaded_,
                          quant_bias_,
                          quant_bias_is_loaded_,
                          weight_scale_,
                          weight_scale_is_loaded_,
                          weight_offset_,
                          weight_offset_is_loaded_});

  // load and merge the weights on dim 0
  // If quant_args_ indicates SmoothQuant, load qweight
  if (quant_args_.quant_method() == kQuantMethodSmoothquant) {
    // Find the first available "smooth" tensor in prefixes (e.g.,
    // "gate.smooth", "up_proj.smooth", etc.)
    for (const auto& prefix : prefixes) {
      auto smooth_tensor_candidate = state_dict.get_tensor(prefix + "smooth");
      if (smooth_tensor_candidate.defined()) {
        // Copy the found smooth tensor to the module parameter
        CHECK_EQ(smooth_.sizes(), smooth_tensor_candidate.sizes())
            << "smooth weight size mismatch for " << state_dict.prefix()
            << "smooth";
        smooth_.copy_(smooth_tensor_candidate);
        smooth_is_loaded_ = true;
        break;
      }
    }
    LOAD_FUSED_WEIGHT(qweight, 0);
    LOAD_FUSED_WEIGHT(per_channel_scale, 0);
  } else if (quant_args_.quant_method() == kQuantMethodFp8) {
    if (is_fp8_channelwise_w8a8(quant_args_)) {
      LOAD_FUSED_WEIGHT(weight, 0);
      LOAD_FUSED_WEIGHT(weight_scale, 0);
    } else {
      // FP8 fused layer loading: each partition may have its own per-tensor
      // scale (unfused checkpoint). We must requantize all partitions with
      // max_scale.

      // Step 1: Collect partition info BEFORE LOAD_FUSED_WEIGHT (clears list)
      Fp8PartitionInfo partition_info;
      if (!weight_scale_is_loaded_) {
        for (const auto& prefix : prefixes) {
          auto scale_tensor = state_dict.get_tensor(prefix + "weight_scale");
          if (scale_tensor.defined()) {
            partition_info.scales.push_back(
                scale_tensor.flatten().item<float>());
          }
          auto weight_tensor = state_dict.get_sharded_tensor(
              prefix + "weight", 0, rank, world_size);
          if (weight_tensor.defined()) {
            partition_info.logical_widths.push_back(weight_tensor.size(0));
          }
        }
      }

      // Step 2: Load fused weight
      LOAD_FUSED_WEIGHT(weight, 0);

      // Step 3: Requantize if needed (unfused checkpoint case)
      if (!weight_scale_is_loaded_ && !partition_info.empty()) {
        float max_scale = compute_max_scale(partition_info.scales);

        if (is_unfused_checkpoint(partition_info.scales) && weight_.defined() &&
            partition_info.logical_widths.size() ==
                partition_info.scales.size()) {
          requantize_fp8_weight(weight_,
                                partition_info.scales,
                                partition_info.logical_widths,
                                max_scale);
        }

        weight_scale_.fill_(max_scale);
        weight_scale_is_loaded_ = true;
      }

      // Step 4: Load input_scale for static activation quantization
      if (input_scale_.defined() && !input_scale_is_loaded_) {
        auto max_input_scale = load_max_input_scale(state_dict, prefixes);
        if (max_input_scale.defined()) {
          input_scale_.copy_(max_input_scale.view({1}));
          input_scale_is_loaded_ = true;
        }
      }
    }
  } else if (is_w8a8_quant(resolved_weight_quant_method_)) {
    LOAD_FUSED_WEIGHT(weight, 0);
    // Fused static W8A8 quantizes the shared input only once, so keep a single
    // input_scale/input_offset slot and pull the first available tensor.
    load_shared_tensor_from_prefixes_or_fail(state_dict,
                                             prefixes,
                                             "input_scale",
                                             input_scale_,
                                             input_scale_is_loaded_);
    load_shared_tensor_from_prefixes_or_fail(state_dict,
                                             prefixes,
                                             "input_offset",
                                             input_offset_,
                                             input_offset_is_loaded_);
    LOAD_FUSED_WEIGHT(deq_scale, 0);
    LOAD_FUSED_WEIGHT(quant_bias, 0);
  } else if (is_w8a8_dynamic_quant(resolved_weight_quant_method_)) {
    LOAD_FUSED_WEIGHT(weight, 0);
    LOAD_FUSED_WEIGHT(weight_scale, 0);
    if (weight_offset_.defined()) {
      LOAD_FUSED_WEIGHT(weight_offset, 0);
    }
  } else {
    LOAD_FUSED_WEIGHT(weight, 0);
  }

  if (bias_.defined()) {
    LOAD_FUSED_WEIGHT(bias, 0);
  }
}

std::optional<torch::Tensor> ColumnParallelLinearImpl::get_input_scale() const {
  if (quant_args_.quant_method() == kQuantMethodFp8 &&
      !quant_args_.activation_dynamic() && input_scale_.defined()) {
    return input_scale_;
  }
  return std::nullopt;
}

// load_state_dict for merged weights with variable shard sizes
void ColumnParallelLinearImpl::load_state_dict(
    const StateDict& state_dict,
    int32_t shard_tensor_count,
    const std::vector<int64_t>& shard_sizes) {
  if (state_dict.size() == 0) {
    return;
  }
  const int64_t rank = weight_rank_;
  const int64_t world_size = weight_world_size_;
  resolve_weight_quant_method_for_linear_load(
      quant_args_, state_dict, nullptr, resolved_weight_quant_method_);
  ensure_w8a8_params_for_linear_load(
      this,
      quant_args_,
      options_,
      resolved_weight_quant_method_,
      /*shared_input_param_size=*/1,
      W8A8LinearParamRefs{weight_,
                          weight_is_loaded_,
                          input_scale_,
                          input_scale_is_loaded_,
                          input_offset_,
                          input_offset_is_loaded_,
                          deq_scale_,
                          deq_scale_is_loaded_,
                          quant_bias_,
                          quant_bias_is_loaded_,
                          weight_scale_,
                          weight_scale_is_loaded_,
                          weight_offset_,
                          weight_offset_is_loaded_});

  // load and merge the weights on dim 0 with variable shard sizes
  if (quant_args_.quant_method() == "smoothquant") {
    // For smoothquant, load quantized weights with variable shard sizes
    LOAD_MERGED_WEIGHT_V2(qweight, 0);
    LOAD_MERGED_WEIGHT_V2(per_channel_scale, 0);
  } else {
    if (is_w8a8_quant(resolved_weight_quant_method_)) {
      LOAD_MERGED_WEIGHT_V2(weight, 0);
      LOAD_WEIGHT(input_scale);
      LOAD_WEIGHT(input_offset);
      LOAD_MERGED_WEIGHT_V2(deq_scale, 0);
      LOAD_MERGED_WEIGHT_V2(quant_bias, 0);
    } else if (is_w8a8_dynamic_quant(resolved_weight_quant_method_)) {
      LOAD_MERGED_WEIGHT_V2(weight, 0);
      LOAD_MERGED_WEIGHT_V2(weight_scale, 0);
      if (weight_offset_.defined()) {
        LOAD_MERGED_WEIGHT_V2(weight_offset, 0);
      }
    } else {
      // For regular weights, use the new merged weight loading with variable
      // shard sizes
      LOAD_MERGED_WEIGHT_V2(weight, 0);
    }
  }

  if (bias_.defined()) {
    // For bias, we might need to handle it differently based on the use case
    // For now, we'll use the same approach if bias is also sharded
    LOAD_MERGED_WEIGHT_V2(bias, 0);
  }
}

QKVParallelLinearImpl::QKVParallelLinearImpl(
    int64_t hidden_size,
    int64_t num_heads,
    int64_t num_kv_heads,
    int64_t head_size,
    int64_t num_kv_head_replicas,
    bool bias,
    bool gather_output,
    const ParallelArgs& parallel_args,
    const torch::TensorOptions& options,
    const QuantArgs& quant_args,
    const LinearExtraArgs& linear_extra_args)
    : hidden_size_(hidden_size),
      num_heads_(num_heads),
      num_kv_heads_(num_kv_heads),
      head_size_(head_size),
      num_kv_head_replicas_(num_kv_head_replicas),
      gather_output_(gather_output),
      parallel_args_(parallel_args),
      options_(options),
      device_(options.device()),
      quant_args_(quant_args),
      output_dtype_(c10::typeMetaToScalarType(options.dtype())) {
  rank_ = parallel_args_.tp_group_->rank();
  world_size_ = parallel_args_.tp_group_->world_size();
  const int64_t out_features_per_partition =
      (num_heads + 2 * num_kv_heads) * head_size;
  (void)linear_extra_args;
  // Note: torch.nn.functional.linear performs XA^T + b and as a result
  // we allocate the transpose.
  if (quant_args_.quant_method() == kQuantMethodFp8) {
    // FP8 W8A8 quantization - weight is stored as FP8 (float8_e4m3fn)
    weight_ = register_parameter(
        "weight",
        torch::empty({out_features_per_partition, hidden_size},
                     options.dtype(torch::kFloat8_e4m3fn)),
        /*requires_grad=*/false);
    const int64_t weight_scale_size =
        is_fp8_channelwise_w8a8(quant_args_) ? out_features_per_partition : 3;
    weight_scale_ = register_parameter(
        "weight_scale",
        torch::empty({weight_scale_size}, options.dtype(torch::kFloat32)),
        /*requires_grad=*/false);
    // For static activation quantization, input_scale is pre-computed
    // Also create {3} for Q/K/V, will use max() after loading
    if (!quant_args_.activation_dynamic()) {
      input_scale_ =
          register_parameter("input_scale",
                             torch::empty({3}, options.dtype(torch::kFloat32)),
                             /*requires_grad=*/false);
    }
  } else if (!quant_args_.quant_descs().empty() ||
             quant_args_.is_compressed_tensors_w8a8_dynamic()) {
    // quant_descs is not empty: default initialize weight as kInt8.
    // During load_state_dict, the weight will be lazily re-registered to the
    // appropriate dtype based on the resolved quant method.
    weight_ = register_parameter(
        "weight",
        torch::empty({out_features_per_partition, hidden_size},
                     options.dtype(torch::kInt8)),
        /*requires_grad=*/false);
  } else {
    weight_ = register_parameter(
        "weight",
        torch::empty({out_features_per_partition, hidden_size}, options),
        /*requires_grad=*/false);
  }

  if (bias) {
    bias_ =
        register_parameter("bias",
                           torch::empty({out_features_per_partition}, options),
                           /*requires_grad=*/false);
  }
}

torch::Tensor QKVParallelLinearImpl::forward(torch::Tensor input) {
  input = input.to(device_);
  auto bias =
      bias_.defined() ? std::optional<torch::Tensor>(bias_) : std::nullopt;

  torch::Tensor output;
  if (quant_args_.quant_method() == kQuantMethodFp8) {
    check_fp8_activation_dynamic_supported(quant_args_);
    auto a_scale = input_scale_.defined()
                       ? std::optional<torch::Tensor>(input_scale_)
                       : std::nullopt;
    output = fp8_linear_forward(
        input, weight_, weight_scale_, a_scale, bias, output_dtype_);
  } else if (is_w8a8_quant(resolved_weight_quant_method_)) {
    CHECK(input_scale_is_loaded_ && input_scale_.defined())
        << "input_scale is required for w8a8 quant matmul.";
    CHECK(input_offset_is_loaded_ && input_offset_.defined())
        << "input_offset is required for w8a8 quant matmul.";
    CHECK(deq_scale_is_loaded_ && deq_scale_.defined())
        << "deq_scale is required for w8a8 quant matmul.";
    auto quant_bias = quant_bias_is_loaded_ && quant_bias_.defined()
                          ? std::optional<torch::Tensor>(quant_bias_)
                          : std::nullopt;
    output = npu_w8a8_linear_forward(input,
                                     weight_,
                                     input_scale_,
                                     input_offset_,
                                     deq_scale_,
                                     quant_bias,
                                     output_dtype_);
  } else if (is_w8a8_dynamic_quant(resolved_weight_quant_method_)) {
    auto weight_scale = weight_scale_is_loaded_
                            ? std::optional<torch::Tensor>(weight_scale_)
                            : std::nullopt;
    CHECK(weight_scale.has_value() && weight_scale.value().defined())
        << "weight_scale is required for w8a8_dynamic quant matmul.";
#if defined(USE_DCU)
    output = dcu_w8a8_dynamic_linear_forward(
        input, weight_, weight_scale.value(), bias, output_dtype_);
#elif defined(USE_NPU)
    output = npu_w8a8_dynamic_linear_forward(
        input, weight_, weight_scale.value(), bias, output_dtype_);
#endif
  } else {
    xllm::kernel::MatmulParams matmul_params;
    matmul_params.a = input;
    matmul_params.b = weight_;
    matmul_params.bias = bias;

    output = xllm::kernel::matmul(matmul_params);
  }

  if (world_size_ > 1 && gather_output_) {
    output = xllm::parallel_state::gather(output, parallel_args_.tp_group_);
  }
  return output;
}

void QKVParallelLinearImpl::load_state_dict(
    const StateDict& state_dict,
    const std::vector<std::string>& prefixes) {
  if (state_dict.size() == 0) {
    return;
  }
  const int64_t rank = rank_;
  const int64_t world_size = world_size_;
  resolve_weight_quant_method_for_linear_load(
      quant_args_, state_dict, &prefixes, resolved_weight_quant_method_);
  ensure_w8a8_params_for_linear_load(
      this,
      quant_args_,
      options_,
      resolved_weight_quant_method_,
      /*shared_input_param_size=*/1,
      W8A8LinearParamRefs{weight_,
                          weight_is_loaded_,
                          input_scale_,
                          input_scale_is_loaded_,
                          input_offset_,
                          input_offset_is_loaded_,
                          deq_scale_,
                          deq_scale_is_loaded_,
                          quant_bias_,
                          quant_bias_is_loaded_,
                          weight_scale_,
                          weight_scale_is_loaded_,
                          weight_offset_,
                          weight_offset_is_loaded_});
  LOAD_QKV_WEIGHT(weight, 0, num_kv_head_replicas_);
  if (bias_.defined()) {
    LOAD_QKV_WEIGHT(bias, 0, num_kv_head_replicas_);
  }
  // FP8: load weight_scale and input_scale, requantize if needed
  if (quant_args_.quant_method() == kQuantMethodFp8) {
    if (is_fp8_channelwise_w8a8(quant_args_)) {
      LOAD_QKV_WEIGHT(weight_scale, 0, num_kv_head_replicas_);
      return;
    }

    // Build partition info for Q/K/V
    Fp8PartitionInfo partition_info;
    int64_t num_heads_per_partition = num_heads_ / world_size_;
    int64_t num_kv_heads_per_partition = num_kv_heads_ / world_size_;
    partition_info.logical_widths = {
        num_heads_per_partition * head_size_,      // Q
        num_kv_heads_per_partition * head_size_,   // K
        num_kv_heads_per_partition * head_size_};  // V

    for (const auto& prefix : prefixes) {
      auto scale_tensor = state_dict.get_tensor(prefix + "weight_scale");
      if (scale_tensor.defined()) {
        partition_info.scales.push_back(scale_tensor.flatten().item<float>());
      }
    }

    // Requantize if unfused checkpoint (multiple scales)
    if (partition_info.scales.size() > 1 && weight_.defined()) {
      float max_scale = compute_max_scale(partition_info.scales);

      if (is_unfused_checkpoint(partition_info.scales)) {
        requantize_fp8_weight(weight_,
                              partition_info.scales,
                              partition_info.logical_widths,
                              max_scale);
      }
      weight_scale_.fill_(max_scale);
    } else if (partition_info.scales.size() == 1) {
      weight_scale_.fill_(partition_info.scales[0]);
    } else {
      LOAD_FUSED_WEIGHT(weight_scale, 0);
    }

    if (!quant_args_.activation_dynamic() && input_scale_.defined()) {
      LOAD_FUSED_WEIGHT(input_scale, 0);
    }

    // For per-tensor quantization with fused QKV, replace scale tensors with
    // scalar max values to avoid recomputing max() in every forward() call.
    // Only apply for per-tensor quantization.
    // Per-channel/per-block quantization should NOT take max.
    if (weight_scale_.defined() && weight_scale_.numel() > 1) {
      weight_scale_ = weight_scale_.max();
    }
    if (input_scale_.defined() && input_scale_.numel() > 1) {
      input_scale_ = input_scale_.max();
    }
  } else if (is_w8a8_quant(resolved_weight_quant_method_)) {
    // input_scale/input_offset are shared activation-quant params and should
    // not inherit the KV-head replication logic used by output-channel tensors.
    load_shared_tensor_from_prefixes_or_fail(state_dict,
                                             prefixes,
                                             "input_scale",
                                             input_scale_,
                                             input_scale_is_loaded_);
    load_shared_tensor_from_prefixes_or_fail(state_dict,
                                             prefixes,
                                             "input_offset",
                                             input_offset_,
                                             input_offset_is_loaded_);
    LOAD_QKV_WEIGHT(deq_scale, 0, num_kv_head_replicas_);
    LOAD_QKV_WEIGHT(quant_bias, 0, num_kv_head_replicas_);
  } else if (is_w8a8_dynamic_quant(resolved_weight_quant_method_)) {
    LOAD_QKV_WEIGHT(weight_scale, 0, num_kv_head_replicas_);
    if (weight_offset_.defined()) {
      LOAD_QKV_WEIGHT(weight_offset, 0, num_kv_head_replicas_);
    }
  }
}

void QKVParallelLinearImpl::load_state_dict(const StateDict& state_dict) {
  if (state_dict.size() == 0) {
    return;
  }
  const int64_t rank = rank_;
  const int64_t world_size = world_size_;
  const int32_t shard_tensor_count = 3;
  const int64_t shard_size = num_heads_ * head_size_;
  resolve_weight_quant_method_for_linear_load(
      quant_args_, state_dict, nullptr, resolved_weight_quant_method_);
  ensure_w8a8_params_for_linear_load(
      this,
      quant_args_,
      options_,
      resolved_weight_quant_method_,
      /*shared_input_param_size=*/1,
      W8A8LinearParamRefs{weight_,
                          weight_is_loaded_,
                          input_scale_,
                          input_scale_is_loaded_,
                          input_offset_,
                          input_offset_is_loaded_,
                          deq_scale_,
                          deq_scale_is_loaded_,
                          quant_bias_,
                          quant_bias_is_loaded_,
                          weight_scale_,
                          weight_scale_is_loaded_,
                          weight_offset_,
                          weight_offset_is_loaded_});
  CHECK_EQ(num_heads_, num_kv_heads_);
  LOAD_MERGED_WEIGHT(weight, 0);

  if (bias_.defined()) {
    LOAD_MERGED_WEIGHT(bias, 0);
  }
  if (is_w8a8_quant(resolved_weight_quant_method_)) {
    const std::vector<std::string> shared_input_prefixes{""};
    load_shared_tensor_from_prefixes_or_fail(state_dict,
                                             shared_input_prefixes,
                                             "input_scale",
                                             input_scale_,
                                             input_scale_is_loaded_);
    load_shared_tensor_from_prefixes_or_fail(state_dict,
                                             shared_input_prefixes,
                                             "input_offset",
                                             input_offset_,
                                             input_offset_is_loaded_);
    LOAD_SHARDED_WEIGHT(deq_scale, 0);
    LOAD_SHARDED_WEIGHT(quant_bias, 0);
  } else if (is_w8a8_dynamic_quant(resolved_weight_quant_method_)) {
    LOAD_SHARDED_WEIGHT(weight_scale, 0);
    if (weight_offset_.defined()) {
      LOAD_SHARDED_WEIGHT(weight_offset, 0);
    }
  }
}

std::optional<torch::Tensor> QKVParallelLinearImpl::get_input_scale() const {
  if (quant_args_.quant_method() == kQuantMethodFp8 &&
      !quant_args_.activation_dynamic() && input_scale_.defined()) {
    // input_scale_ is already reduced to per-tensor scale in load_state_dict.
    return input_scale_;
  }
  return std::nullopt;
}

// Linear layer with row parallelism.
RowParallelLinearImpl::RowParallelLinearImpl(
    int64_t in_features,
    int64_t out_features,
    bool bias,
    bool input_is_parallelized,
    bool enable_result_reduction,
    const QuantArgs& quant_args,
    ProcessGroup* process_group,
    const torch::TensorOptions& options,
    const LinearExtraArgs& linear_extra_args)
    : input_is_parallelized_(input_is_parallelized),
      enable_result_reduction_(enable_result_reduction),
      quant_args_(quant_args),
      options_(options),
      process_group_(process_group),
      linear_extra_args_(linear_extra_args),
      output_dtype_(c10::typeMetaToScalarType(options.dtype())) {
  rank_ = process_group_->rank();
  world_size_ = process_group_->world_size();
  CHECK(in_features % world_size_ == 0)
      << "in_features " << in_features << " not divisible by world_size "
      << world_size_;
  const int64_t in_features_per_partition = in_features / world_size_;
  // Allocate the transpose since linear performs XA^T.
  if (quant_args_.quant_method() == kQuantMethodSmoothquant) {
    qweight_ = register_parameter(
        "qweight",
        torch::empty({out_features, in_features_per_partition},
                     options.dtype(torch::kInt8)),
        /*requires_grad=*/false);
    per_channel_scale_ = register_parameter(
        "per_channel_scale",
        torch::empty({out_features}, options.dtype(torch::kFloat32)),
        /*requires_grad=*/false);
    smooth_ = register_parameter("smooth",
                                 torch::empty({in_features_per_partition},
                                              options.dtype(torch::kFloat32)),
                                 /*requires_grad=*/false);
    // Output dtype for scaled_matmul
    output_dtype_ = c10::typeMetaToScalarType(options.dtype());
  } else if (quant_args_.quant_method() == kQuantMethodFp8) {
    // FP8 W8A8 quantization - weight is stored as FP8 (float8_e4m3fn)
    weight_ = register_parameter(
        "weight",
        torch::empty({out_features, in_features_per_partition},
                     options.dtype(torch::kFloat8_e4m3fn)),
        /*requires_grad=*/false);
    const int64_t weight_scale_size =
        is_fp8_channelwise_w8a8(quant_args_) ? out_features : 1;
    weight_scale_ = register_parameter(
        "weight_scale",
        torch::empty({weight_scale_size}, options.dtype(torch::kFloat32)),
        /*requires_grad=*/false);
    // For static activation quantization, input_scale is pre-computed
    if (!quant_args_.activation_dynamic()) {
      input_scale_ =
          register_parameter("input_scale",
                             torch::empty({1}, options.dtype(torch::kFloat32)),
                             /*requires_grad=*/false);
    }
  } else if (!quant_args_.quant_descs().empty() ||
             quant_args_.is_compressed_tensors_w8a8_dynamic()) {
    // quant_descs is not empty: default initialize weight as kInt8.
    // During load_state_dict, the weight will be lazily re-registered to the
    // appropriate dtype based on the resolved quant method.
    weight_ = register_parameter(
        "weight",
        torch::empty({out_features, in_features_per_partition},
                     options.dtype(torch::kInt8)),
        /*requires_grad=*/false);
  } else {
    weight_ = register_parameter(
        "weight",
        torch::empty({out_features, in_features_per_partition}, options),
        /*requires_grad=*/false);
  }

  if (bias) {
    bias_ = register_parameter("bias",
                               torch::empty({out_features}, options),
                               /*requires_grad=*/false);
  }
}

torch::Tensor RowParallelLinearImpl::forward(torch::Tensor input) {
  auto bias = bias_.defined() && rank_ == 0
                  ? std::optional<torch::Tensor>(bias_)
                  : std::nullopt;
  torch::Tensor output;
  if (quant_args_.quant_method() == kQuantMethodSmoothquant) {
    CHECK(smooth_.defined()) << "smooth is required for smoothquant.";
    CHECK(qweight_.defined()) << "qweight is required for smoothquant.";
    CHECK(per_channel_scale_.defined())
        << "per_channel_scale is required for smoothquant.";

    torch::Tensor quantized_input;
    torch::Tensor input_scale;

    if (!input_is_parallelized_) {
      input = xllm::parallel_state::scatter(input, process_group_);
    }

    xllm::kernel::ScaledQuantizeParams quantize_params;
    quantize_params.x = input;
    quantize_params.smooth = smooth_;
    quantize_params.zero = std::nullopt;
    quantize_params.token_count = std::nullopt;
    quantize_params.gather_index = std::nullopt;
    quantize_params.gather_index_start_position = std::nullopt;
    quantize_params.output = std::nullopt;
    quantize_params.output_scale = std::nullopt;
    quantize_params.act_mode = linear_extra_args_.act_mode;
    quantize_params.active_coef = 1.0;
    quantize_params.is_gated = linear_extra_args_.is_gated;

    std::tie(quantized_input, input_scale) =
        xllm::kernel::scaled_quantize(quantize_params);

    xllm::kernel::ScaledMatmulParams matmul_params;
    matmul_params.a = quantized_input;
    matmul_params.b = qweight_;
    matmul_params.a_scale = input_scale;
    matmul_params.b_scale = per_channel_scale_;
    matmul_params.output_dtype = output_dtype_;
    matmul_params.bias = bias;
    matmul_params.c = std::nullopt;
    matmul_params.act_mode = "none";
    matmul_params.quant_bit_size = 8;
    matmul_params.alpha = 1.0;
    matmul_params.beta = 0.0;
    matmul_params.use_hp_active = false;
    matmul_params.a_quant_bit_size = 8;
    matmul_params.a_calib = std::nullopt;
    matmul_params.b_calib = std::nullopt;
    matmul_params.output = std::nullopt;

    output = xllm::kernel::scaled_matmul(matmul_params);
  } else if (quant_args_.quant_method() == kQuantMethodFp8) {
    check_fp8_activation_dynamic_supported(quant_args_);
    if (!input_is_parallelized_) {
      input = xllm::parallel_state::scatter(input, process_group_);
    }

    auto scale = input_scale_.defined()
                     ? std::optional<torch::Tensor>(input_scale_)
                     : std::nullopt;
    output = fp8_linear_forward(
        input, weight_, weight_scale_, scale, bias, output_dtype_);
  } else if (is_w8a8_quant(resolved_weight_quant_method_)) {
    CHECK(input_scale_is_loaded_ && input_scale_.defined())
        << "input_scale is required for w8a8 quant matmul.";
    CHECK(input_offset_is_loaded_ && input_offset_.defined())
        << "input_offset is required for w8a8 quant matmul.";
    CHECK(deq_scale_is_loaded_ && deq_scale_.defined())
        << "deq_scale is required for w8a8 quant matmul.";
    if (!input_is_parallelized_) {
      input = xllm::parallel_state::scatter(input, process_group_);
    }
    auto quant_bias = quant_bias_is_loaded_ && quant_bias_.defined()
                          ? std::optional<torch::Tensor>(quant_bias_)
                          : std::nullopt;
    output = npu_w8a8_linear_forward(input,
                                     weight_,
                                     input_scale_,
                                     input_offset_,
                                     deq_scale_,
                                     quant_bias,
                                     output_dtype_);
  } else if (is_w8a8_dynamic_quant(resolved_weight_quant_method_)) {
    if (!input_is_parallelized_) {
      input = xllm::parallel_state::scatter(input, process_group_);
    }
    auto weight_scale = weight_scale_is_loaded_
                            ? std::optional<torch::Tensor>(weight_scale_)
                            : std::nullopt;
    CHECK(weight_scale.has_value() && weight_scale.value().defined())
        << "weight_scale is required for w8a8_dynamic quant matmul.";
#if defined(USE_DCU)
    output = dcu_w8a8_dynamic_linear_forward(
        input, weight_, weight_scale.value(), bias, output_dtype_);
#elif defined(USE_NPU)
    output = npu_w8a8_dynamic_linear_forward(
        input, weight_, weight_scale.value(), bias, output_dtype_);
#endif
  } else {
    if (!input_is_parallelized_) {
      input = xllm::parallel_state::scatter(input, process_group_);
    }
    xllm::kernel::MatmulParams matmul_params;
    matmul_params.a = input;
    matmul_params.b = weight_;
    matmul_params.bias = bias;
    output = xllm::kernel::matmul(matmul_params);
  }
  if (enable_result_reduction_ && world_size_ > 1) {
    output = xllm::parallel_state::reduce(output, process_group_);
  }
  return output;
}

// load the weight from the checkpoint
void RowParallelLinearImpl::load_state_dict(const StateDict& state_dict) {
  if (state_dict.size() == 0) {
    return;
  }
  const int64_t rank = world_size_ == 1 ? 0 : rank_;
  const int64_t world_size = world_size_;
  resolve_weight_quant_method_for_linear_load(
      quant_args_, state_dict, nullptr, resolved_weight_quant_method_);
  ensure_w8a8_params_for_linear_load(
      this,
      quant_args_,
      options_,
      resolved_weight_quant_method_,
      /*shared_input_param_size=*/1,
      W8A8LinearParamRefs{weight_,
                          weight_is_loaded_,
                          input_scale_,
                          input_scale_is_loaded_,
                          input_offset_,
                          input_offset_is_loaded_,
                          deq_scale_,
                          deq_scale_is_loaded_,
                          quant_bias_,
                          quant_bias_is_loaded_,
                          weight_scale_,
                          weight_scale_is_loaded_,
                          weight_offset_,
                          weight_offset_is_loaded_});

  // If quant_args_ indicates SmoothQuant, load qweight; otherwise, load
  // normal weight.
  if (quant_args_.quant_method() == kQuantMethodSmoothquant) {
    LOAD_SHARDED_WEIGHT(qweight, 1);
    LOAD_WEIGHT(per_channel_scale);
    LOAD_SHARDED_WEIGHT(smooth, 0);
  } else if (quant_args_.quant_method() == kQuantMethodFp8) {
    // FP8 quantization: load FP8 weight and scales
    LOAD_SHARDED_WEIGHT(weight, 1);
    LOAD_WEIGHT(weight_scale);
    // For static activation quantization, load input_scale
    if (!quant_args_.activation_dynamic() && input_scale_.defined()) {
      LOAD_WEIGHT(input_scale);
    }
  } else if (is_w8a8_quant(resolved_weight_quant_method_)) {
    LOAD_SHARDED_WEIGHT(weight, 1);
    LOAD_WEIGHT(input_scale);
    LOAD_WEIGHT(input_offset);
    LOAD_WEIGHT(deq_scale);
    if (rank_ == 0) {
      LOAD_WEIGHT(quant_bias);
    } else if (quant_bias_.defined()) {
      quant_bias_.zero_();
      quant_bias_is_loaded_ = true;
    }
  } else if (is_w8a8_dynamic_quant(resolved_weight_quant_method_)) {
    LOAD_SHARDED_WEIGHT(weight, 1);
    LOAD_WEIGHT(weight_scale);
    if (weight_offset_.defined()) {
      LOAD_WEIGHT(weight_offset);
    }
  } else {
    LOAD_SHARDED_WEIGHT(weight, 1);
  }

  if (bias_.defined()) {
    LOAD_WEIGHT(bias);
  }
}

// Linear layer with row parallelism.
ReplicatedLinearImpl::ReplicatedLinearImpl(
    int64_t in_features,
    int64_t out_features,
    bool bias,
    const QuantArgs& quant_args,
    const torch::TensorOptions& options,
    const LinearExtraArgs& linear_extra_args)
    : quant_args_(quant_args),
      options_(options),
      output_dtype_(c10::typeMetaToScalarType(options.dtype())) {
  (void)linear_extra_args;
  if (quant_args_.quant_method() == kQuantMethodFp8) {
    // Replicated projections are mixed in DeepSeek checkpoints: attention
    // low-rank projections can be FP8, while router gates remain BF16. Keep the
    // storage in runtime dtype initially and switch to FP8 lazily after seeing
    // the checkpoint tensor dtype in load_state_dict().
    weight_ =
        register_parameter("weight",
                           torch::empty({out_features, in_features}, options),
                           /*requires_grad=*/false);
  } else if (!quant_args_.quant_descs().empty() ||
             quant_args_.is_compressed_tensors_w8a8_dynamic()) {
    // quant_descs is not empty: default initialize weight as kInt8.
    // During load_state_dict, the weight will be lazily re-registered to the
    // appropriate dtype based on the resolved quant method.
    weight_ = register_parameter(
        "weight",
        torch::empty({out_features, in_features}, options.dtype(torch::kInt8)),
        /*requires_grad=*/false);
  } else {
    weight_ =
        register_parameter("weight",
                           torch::empty({out_features, in_features}, options),
                           /*requires_grad=*/false);
  }

  if (bias) {
    bias_ = register_parameter("bias",
                               torch::empty({out_features}, options),
                               /*requires_grad=*/false);
  }
}

torch::Tensor ReplicatedLinearImpl::forward(torch::Tensor input) {
  auto bias =
      bias_.defined() ? std::optional<torch::Tensor>(bias_) : std::nullopt;
  if (is_fp8_dtype(weight_.scalar_type())) {
    check_fp8_activation_dynamic_supported(quant_args_);
    CHECK(weight_scale_.defined())
        << "weight_scale is required for FP8 replicated linear.";
    auto scale = input_scale_.defined()
                     ? std::optional<torch::Tensor>(input_scale_)
                     : std::nullopt;
    return fp8_linear_forward(
        input, weight_, weight_scale_, scale, bias, output_dtype_);
  }
  if (is_w8a8_quant(resolved_weight_quant_method_)) {
    CHECK(input_scale_is_loaded_ && input_scale_.defined())
        << "input_scale is required for w8a8 quant matmul.";
    CHECK(input_offset_is_loaded_ && input_offset_.defined())
        << "input_offset is required for w8a8 quant matmul.";
    CHECK(deq_scale_is_loaded_ && deq_scale_.defined())
        << "deq_scale is required for w8a8 quant matmul.";
    auto quant_bias = quant_bias_is_loaded_ && quant_bias_.defined()
                          ? std::optional<torch::Tensor>(quant_bias_)
                          : std::nullopt;
    return npu_w8a8_linear_forward(input,
                                   weight_,
                                   input_scale_,
                                   input_offset_,
                                   deq_scale_,
                                   quant_bias,
                                   input.scalar_type());
  }
  if (is_w8a8_dynamic_quant(resolved_weight_quant_method_)) {
    auto weight_scale = weight_scale_is_loaded_
                            ? std::optional<torch::Tensor>(weight_scale_)
                            : std::nullopt;
    CHECK(weight_scale.has_value() && weight_scale.value().defined())
        << "weight_scale is required for w8a8_dynamic quant matmul.";
#if defined(USE_DCU)
    return dcu_w8a8_dynamic_linear_forward(
        input, weight_, weight_scale.value(), bias, input.scalar_type());
#elif defined(USE_NPU)
    return npu_w8a8_dynamic_linear_forward(
        input, weight_, weight_scale.value(), bias, input.scalar_type());
#endif
  }
  xllm::kernel::MatmulParams matmul_params;
  matmul_params.a = input;
  matmul_params.b = weight_;
  matmul_params.bias = bias;

  auto output = xllm::kernel::matmul(matmul_params);
  return output;
}

bool ReplicatedLinearImpl::uses_w8a8_dynamic_quant() const {
  return is_w8a8_dynamic_quant(resolved_weight_quant_method_);
}

torch::Tensor ReplicatedLinearImpl::w8a8_dynamic_weight_scale() const {
  CHECK(uses_w8a8_dynamic_quant())
      << "w8a8_dynamic_weight_scale requires w8a8_dynamic quant method.";
  CHECK(weight_scale_is_loaded_ && weight_scale_.defined())
      << "weight_scale is required for w8a8_dynamic quant matmul.";
  return weight_scale_;
}

at::ScalarType ReplicatedLinearImpl::output_dtype() const {
  return output_dtype_;
}

std::optional<torch::Tensor> ReplicatedLinearImpl::bias() const {
  if (bias_.defined()) {
    return bias_;
  }
  return std::nullopt;
}

// load the weight from the checkpoint
void ReplicatedLinearImpl::load_state_dict(const StateDict& state_dict) {
  if (state_dict.size() == 0) {
    return;
  }
  resolve_weight_quant_method_for_linear_load(
      quant_args_, state_dict, nullptr, resolved_weight_quant_method_);
  ensure_w8a8_params_for_linear_load(
      this,
      quant_args_,
      options_,
      resolved_weight_quant_method_,
      /*shared_input_param_size=*/1,
      W8A8LinearParamRefs{weight_,
                          weight_is_loaded_,
                          input_scale_,
                          input_scale_is_loaded_,
                          input_offset_,
                          input_offset_is_loaded_,
                          deq_scale_,
                          deq_scale_is_loaded_,
                          quant_bias_,
                          quant_bias_is_loaded_,
                          weight_scale_,
                          weight_scale_is_loaded_,
                          weight_offset_,
                          weight_offset_is_loaded_});

  if (quant_args_.quant_method() == kQuantMethodFp8) {
    torch::Tensor checkpoint_weight = state_dict.get_tensor("weight");
    if (checkpoint_weight.defined() &&
        is_fp8_dtype(checkpoint_weight.scalar_type())) {
      const int64_t out_features = weight_.size(0);
      const int64_t in_features = weight_.size(1);
      const int64_t weight_scale_size =
          is_fp8_channelwise_w8a8(quant_args_) ? out_features : 1;
      std::vector<weight::LazyParameterSpec> specs;
      specs.reserve(quant_args_.activation_dynamic() ? 2 : 3);
      specs.push_back(weight::LazyParameterSpec{
          &weight_,
          &weight_is_loaded_,
          "weight",
          {out_features, in_features},
          options_.dtype(checkpoint_weight.scalar_type())});
      specs.push_back(
          weight::LazyParameterSpec{&weight_scale_,
                                    &weight_scale_is_loaded_,
                                    "weight_scale",
                                    {weight_scale_size},
                                    options_.dtype(torch::kFloat32)});
      if (!quant_args_.activation_dynamic()) {
        specs.push_back(
            weight::LazyParameterSpec{&input_scale_,
                                      &input_scale_is_loaded_,
                                      "input_scale",
                                      {1},
                                      options_.dtype(torch::kFloat32)});
      }
      weight::ensure_parameter_storage(this, specs);
    }
  }

  LOAD_WEIGHT(weight);
  if (is_fp8_dtype(weight_.scalar_type())) {
    LOAD_WEIGHT(weight_scale);
    if (!quant_args_.activation_dynamic() && input_scale_.defined()) {
      LOAD_WEIGHT(input_scale);
    }
  } else if (is_w8a8_quant(resolved_weight_quant_method_)) {
    LOAD_WEIGHT(input_scale);
    LOAD_WEIGHT(input_offset);
    LOAD_WEIGHT(deq_scale);
    LOAD_WEIGHT(quant_bias);
  } else if (is_w8a8_dynamic_quant(resolved_weight_quant_method_)) {
    LOAD_WEIGHT(weight_scale);
    if (weight_offset_.defined()) {
      LOAD_WEIGHT(weight_offset);
    }
  }
  if (bias_.defined()) {
    LOAD_WEIGHT(bias);
  }
}

}  // namespace layer
}  // namespace xllm

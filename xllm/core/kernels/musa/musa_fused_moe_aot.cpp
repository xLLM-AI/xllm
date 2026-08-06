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

#include <glog/logging.h>
#include <musa.h>
#include <tvm/ffi/extra/stl.h>
#include <unistd.h>

#include <array>
#include <cstdint>
#include <cstdlib>
#include <map>
#include <mutex>
#include <set>
#include <string>
#include <tuple>

#include "kernels/musa/musa_ops_api.h"
#include "kernels/musa/musa_tvmffi_stream.h"
#include "torch_musa/csrc/core/MUSAGuard.h"
#include "torch_musa/csrc/core/MUSAStream.h"

namespace xllm::kernel::musa {
namespace {

constexpr const char* kAlignUri = "xllm_musa_moe_align_block_size";
constexpr const char* kActivationUri = "xllm_musa_moe_act_and_mul";
constexpr const char* kSumUri = "xllm_musa_moe_sum_reduce";
constexpr const char* kKernelName = "fused_moe_kernel";
constexpr int64_t kTopK = 8;
constexpr int64_t kGroupSize = 128;

enum class MoeStage : int32_t { kGate = 0, kDown = 1 };

struct KernelConfig {
  int64_t batch_size;
  int64_t block_m;
  int64_t block_n;
  uint32_t threads_per_block;
  uint32_t shared_memory;
};

constexpr std::array<KernelConfig, 8> kKernelConfigs = {
    KernelConfig{1, 32, 32, 128, 8192},
    KernelConfig{2, 16, 64, 128, 10240},
    KernelConfig{3, 16, 64, 128, 10240},
    KernelConfig{4, 16, 64, 128, 10240},
    KernelConfig{5, 32, 64, 128, 12288},
    KernelConfig{6, 32, 64, 128, 12288},
    KernelConfig{7, 32, 64, 128, 12288},
    KernelConfig{8, 32, 64, 128, 12288},
};

constexpr std::array<KernelConfig, 8> kBf16KernelConfigs = {
    KernelConfig{1, 32, 32, 128, 16384},
    KernelConfig{2, 16, 64, 128, 10240},
    KernelConfig{3, 16, 64, 128, 10240},
    KernelConfig{4, 32, 64, 256, 24576},
    KernelConfig{5, 32, 64, 256, 24576},
    KernelConfig{6, 32, 64, 256, 24576},
    KernelConfig{7, 32, 64, 256, 24576},
    KernelConfig{8, 32, 64, 256, 24576},
};

struct KernelHandle {
  MUmodule module = nullptr;
  MUfunction function = nullptr;
};

struct KernelArguments {
  uint64_t input;
  uint64_t weights;
  uint64_t output;
  uint64_t input_scale;
  uint64_t weight_scale;
  uint64_t topk_weights;
  uint64_t sorted_token_ids;
  uint64_t expert_ids;
  uint64_t num_tokens_post_padded;
  int32_t output_size;
  int32_t input_size;
  int32_t max_padded_tokens;
  int32_t valid_assignments;
  int32_t input_stride;
  int32_t weight_expert_stride;
  int32_t weight_output_stride;
  int32_t bias_expert_stride;
  int32_t bias_output_stride;
  int32_t output_assignment_stride;
  int32_t input_scale_row_stride;
  int32_t weight_scale_expert_stride;
  int32_t weight_scale_output_stride;
};

const KernelConfig* find_config(int64_t batch_size, bool use_fp8) {
  const auto& configs = use_fp8 ? kKernelConfigs : kBf16KernelConfigs;
  for (const KernelConfig& config : configs) {
    if (config.batch_size == batch_size) {
      return &config;
    }
  }
  return nullptr;
}

int64_t max_padded_tokens(int64_t assignment_count,
                          int64_t num_experts,
                          int64_t block_size) {
  if (assignment_count < num_experts + 1) {
    return assignment_count * block_size;
  }
  return assignment_count + (num_experts + 1) * (block_size - 1);
}

std::string artifact_root() {
  const char* explicit_path = std::getenv("XLLM_MUSA_FUSED_MOE_AOT_PATH");
  if (explicit_path != nullptr && explicit_path[0] != '\0') {
    return explicit_path;
  }
  const char* ops_path = std::getenv("FLASHINFER_OPS_PATH");
  if (ops_path == nullptr || ops_path[0] == '\0') {
    return {};
  }
  return std::string(ops_path) + "/xllm_musa_fused_moe_aot/mp31";
}

std::string bf16_artifact_root() {
  const char* explicit_path = std::getenv("XLLM_MUSA_FUSED_MOE_BF16_AOT_PATH");
  if (explicit_path != nullptr && explicit_path[0] != '\0') {
    return explicit_path;
  }
  const char* ops_path = std::getenv("FLASHINFER_OPS_PATH");
  if (ops_path == nullptr || ops_path[0] == '\0') {
    return {};
  }
  return std::string(ops_path) + "/xllm_musa_fused_moe_aot/bf16_mp31";
}

std::string artifact_path(int64_t batch_size, MoeStage stage, bool use_fp8) {
  const std::string root = use_fp8 ? artifact_root() : bf16_artifact_root();
  if (root.empty()) {
    return {};
  }
  const char* stage_name = stage == MoeStage::kGate ? "gate" : "down";
  return root + "/b" + std::to_string(batch_size) + "_" + stage_name + ".mubin";
}

bool readable(const std::string& path) {
  return !path.empty() && ::access(path.c_str(), R_OK) == 0;
}

bool ffi_artifact_available(const char* uri) {
  const char* ops_path = std::getenv("FLASHINFER_OPS_PATH");
  if (ops_path == nullptr || ops_path[0] == '\0') {
    return false;
  }
  const std::string path =
      std::string(ops_path) + "/" + uri + "/" + uri + ".so";
  return readable(path);
}

void check_driver_result(MUresult result, const char* operation) {
  if (result == MUresult::MUSA_SUCCESS) {
    return;
  }
  const char* error = nullptr;
  muGetErrorString(result, &error);
  CHECK(false) << operation << " failed with MUSA driver error "
               << static_cast<int32_t>(result) << ": "
               << (error == nullptr ? "unknown" : error);
}

KernelHandle& get_kernel(int32_t device_index,
                         int64_t batch_size,
                         MoeStage stage,
                         bool use_fp8) {
  using KernelKey = std::tuple<int32_t, int64_t, int32_t, bool>;
  static std::mutex mutex;
  static std::map<KernelKey, KernelHandle> handles;

  const KernelKey key = std::make_tuple(
      device_index, batch_size, static_cast<int32_t>(stage), use_fp8);
  std::lock_guard<std::mutex> lock(mutex);
  auto [iterator, inserted] = handles.try_emplace(key);
  if (!inserted) {
    return iterator->second;
  }

  const std::string path = artifact_path(batch_size, stage, use_fp8);
  CHECK(readable(path)) << "MUSA fused MoE artifact is not readable: " << path;
  check_driver_result(muModuleLoad(&iterator->second.module, path.c_str()),
                      "muModuleLoad");
  check_driver_result(
      muModuleGetFunction(
          &iterator->second.function, iterator->second.module, kKernelName),
      "muModuleGetFunction");
  return iterator->second;
}

uint64_t tensor_pointer(const torch::Tensor& tensor) {
  if (!tensor.defined()) {
    return 0;
  }
  return static_cast<uint64_t>(reinterpret_cast<uintptr_t>(tensor.data_ptr()));
}

int32_t checked_int32(int64_t value, const char* name) {
  CHECK_GE(value, 0) << name << " must be non-negative.";
  CHECK_LE(value, static_cast<int64_t>(INT32_MAX))
      << name << " exceeds int32 range: " << value;
  return static_cast<int32_t>(value);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> align_topk_ids(
    const torch::Tensor& topk_ids,
    int64_t num_experts,
    int64_t block_size) {
  const int64_t padded_tokens =
      max_padded_tokens(topk_ids.numel(), num_experts, block_size);
  auto int_options = topk_ids.options().dtype(torch::kInt32);
  auto sorted_token_ids = torch::empty({padded_tokens}, int_options);
  auto expert_ids = torch::empty(
      {(padded_tokens + block_size - 1) / block_size}, int_options);
  auto num_tokens_post_padded = torch::empty({1}, int_options);
  auto cumsum_buffer = torch::empty({num_experts + 2}, int_options);

  TvmffiStreamGuard stream_guard(topk_ids.device());
  get_function(kAlignUri, "xllm_musa_moe_align_block_size")(
      to_ffi_tensor_view(topk_ids),
      num_experts + 1,
      block_size,
      to_ffi_tensor_view(sorted_token_ids),
      to_ffi_tensor_view(expert_ids),
      to_ffi_tensor_view(num_tokens_post_padded),
      to_ffi_tensor_view(cumsum_buffer),
      /*pad_sorted_token_ids=*/true);
  return std::make_tuple(sorted_token_ids, expert_ids, num_tokens_post_padded);
}

torch::Tensor activate_gate_up(const torch::Tensor& gate_up,
                               const torch::Tensor& topk_ids) {
  auto output =
      torch::empty({gate_up.size(0), gate_up.size(1) / 2}, gate_up.options());
  TvmffiStreamGuard stream_guard(gate_up.device());
  get_function(kActivationUri, "xllm_musa_moe_act_and_mul")(
      to_ffi_tensor_view(gate_up),
      to_ffi_tensor_view(output),
      to_ffi_tensor_view(topk_ids.reshape({-1})),
      /*expert_step=*/static_cast<int64_t>(1),
      /*activation_type=*/static_cast<int64_t>(0),
      /*skip_expert_check=*/static_cast<int64_t>(1));
  return output;
}

torch::Tensor reduce_assignments(const torch::Tensor& assignments) {
  auto output = torch::empty({assignments.size(0), assignments.size(2)},
                             assignments.options());
  TvmffiStreamGuard stream_guard(assignments.device());
  get_function(kSumUri, "xllm_musa_moe_sum_reduce")(
      to_ffi_tensor_view(assignments),
      to_ffi_tensor_view(output),
      /*routed_scaling_factor=*/1.0);
  return output;
}

torch::Tensor launch_gemm(const torch::Tensor& input,
                          const torch::Tensor& input_scale,
                          const torch::Tensor& weights,
                          const torch::Tensor& weight_scale,
                          const torch::Tensor& topk_weights,
                          const torch::Tensor& sorted_token_ids,
                          const torch::Tensor& expert_ids,
                          const torch::Tensor& num_tokens_post_padded,
                          const KernelConfig& config,
                          MoeStage stage,
                          bool use_fp8) {
  const int64_t batch_size = topk_weights.size(0);
  const int64_t assignment_count = batch_size * kTopK;
  torch::Tensor output;
  if (stage == MoeStage::kGate) {
    output = torch::empty({assignment_count, weights.size(1)},
                          input.options().dtype(torch::kBFloat16));
  } else {
    output = torch::empty({batch_size, kTopK, weights.size(1)},
                          input.options().dtype(torch::kBFloat16));
  }

  KernelArguments arguments = {
      .input = tensor_pointer(input),
      .weights = tensor_pointer(weights),
      .output = tensor_pointer(output),
      .input_scale = tensor_pointer(input_scale),
      .weight_scale = tensor_pointer(weight_scale),
      .topk_weights = tensor_pointer(topk_weights),
      .sorted_token_ids = tensor_pointer(sorted_token_ids),
      .expert_ids = tensor_pointer(expert_ids),
      .num_tokens_post_padded = tensor_pointer(num_tokens_post_padded),
      .output_size = checked_int32(weights.size(1), "output_size"),
      .input_size = checked_int32(weights.size(2), "input_size"),
      .max_padded_tokens =
          checked_int32(sorted_token_ids.size(0), "max_padded_tokens"),
      .valid_assignments = checked_int32(assignment_count, "valid_assignments"),
      .input_stride = checked_int32(input.stride(0), "input_stride"),
      .weight_expert_stride =
          checked_int32(weights.stride(0), "weight_expert_stride"),
      .weight_output_stride =
          checked_int32(weights.stride(1), "weight_output_stride"),
      .bias_expert_stride = 0,
      .bias_output_stride = 0,
      .output_assignment_stride =
          checked_int32(output.stride(-2), "output_assignment_stride"),
      .input_scale_row_stride =
          input_scale.defined()
              ? checked_int32(input_scale.stride(0), "input_scale_row_stride")
              : 0,
      .weight_scale_expert_stride =
          weight_scale.defined() ? checked_int32(weight_scale.stride(0),
                                                 "weight_scale_expert_stride")
                                 : 0,
      .weight_scale_output_stride =
          weight_scale.defined() ? checked_int32(weight_scale.stride(1),
                                                 "weight_scale_output_stride")
                                 : 0,
  };
  std::array<void*, 22> parameters = {
      &arguments.input,
      &arguments.weights,
      &arguments.output,
      &arguments.input_scale,
      &arguments.weight_scale,
      &arguments.topk_weights,
      &arguments.sorted_token_ids,
      &arguments.expert_ids,
      &arguments.num_tokens_post_padded,
      &arguments.output_size,
      &arguments.input_size,
      &arguments.max_padded_tokens,
      &arguments.valid_assignments,
      &arguments.input_stride,
      &arguments.weight_expert_stride,
      &arguments.weight_output_stride,
      &arguments.bias_expert_stride,
      &arguments.bias_output_stride,
      &arguments.output_assignment_stride,
      &arguments.input_scale_row_stride,
      &arguments.weight_scale_expert_stride,
      &arguments.weight_scale_output_stride,
  };

  c10::musa::MUSAGuard device_guard(input.device().index());
  KernelHandle& kernel =
      get_kernel(input.device().index(), batch_size, stage, use_fp8);
  const uint32_t grid_x = static_cast<uint32_t>(
      ((sorted_token_ids.size(0) + config.block_m - 1) / config.block_m) *
      ((weights.size(1) + config.block_n - 1) / config.block_n));
  MUstream stream = reinterpret_cast<MUstream>(
      c10::musa::getCurrentMUSAStream(input.device().index()).stream());
  check_driver_result(muLaunchKernel(kernel.function,
                                     grid_x,
                                     1,
                                     1,
                                     config.threads_per_block,
                                     1,
                                     1,
                                     config.shared_memory,
                                     stream,
                                     parameters.data(),
                                     nullptr),
                      "muLaunchKernel");
  return output;
}

void check_common_inputs(const torch::Tensor& hidden_states,
                         const torch::Tensor& w13,
                         const torch::Tensor& w2,
                         const torch::Tensor& topk_weights,
                         const torch::Tensor& topk_ids) {
  CHECK_EQ(hidden_states.dim(), 2);
  CHECK_EQ(hidden_states.scalar_type(), torch::kBFloat16);
  CHECK(hidden_states.is_contiguous());
  CHECK_EQ(w13.dim(), 3);
  CHECK_EQ(w2.dim(), 3);
  CHECK(w13.is_contiguous() && w2.is_contiguous());
  CHECK_EQ(topk_weights.dim(), 2);
  CHECK_EQ(topk_ids.dim(), 2);
  CHECK_EQ(topk_weights.size(0), hidden_states.size(0));
  CHECK_EQ(topk_ids.size(0), hidden_states.size(0));
  CHECK_EQ(topk_weights.size(1), kTopK);
  CHECK_EQ(topk_ids.size(1), kTopK);
  CHECK_EQ(topk_weights.scalar_type(), torch::kFloat32);
  CHECK_EQ(topk_ids.scalar_type(), torch::kInt32);
  CHECK(topk_weights.is_contiguous() && topk_ids.is_contiguous());
  CHECK_EQ(w13.size(0), w2.size(0));
  CHECK_EQ(w13.size(2), hidden_states.size(1));
  CHECK_EQ(w13.size(1) % 2, 0);
  CHECK_EQ(w2.size(1), hidden_states.size(1));
  CHECK_EQ(w2.size(2), w13.size(1) / 2);
  CHECK(hidden_states.device() == w13.device() &&
        hidden_states.device() == w2.device() &&
        hidden_states.device() == topk_weights.device() &&
        hidden_states.device() == topk_ids.device());
}

void check_fp8_inputs(const torch::Tensor& hidden_states,
                      const torch::Tensor& w13,
                      const torch::Tensor& w13_scale,
                      const torch::Tensor& w2,
                      const torch::Tensor& w2_scale,
                      const torch::Tensor& topk_weights,
                      const torch::Tensor& topk_ids) {
  check_common_inputs(hidden_states, w13, w2, topk_weights, topk_ids);
  CHECK_EQ(w13.scalar_type(), torch::kFloat8_e4m3fn);
  CHECK_EQ(w2.scalar_type(), torch::kFloat8_e4m3fn);
  CHECK_EQ(w13_scale.dim(), 3);
  CHECK_EQ(w2_scale.dim(), 3);
  CHECK_EQ(w13_scale.scalar_type(), torch::kFloat32);
  CHECK_EQ(w2_scale.scalar_type(), torch::kFloat32);
  CHECK(w13_scale.is_contiguous() && w2_scale.is_contiguous());
  CHECK_EQ(w13_scale.size(0), w13.size(0));
  CHECK_EQ(w13_scale.size(1) * kGroupSize, w13.size(1));
  CHECK_EQ(w13_scale.size(2) * kGroupSize, w13.size(2));
  CHECK_EQ(w2_scale.size(0), w2.size(0));
  CHECK_EQ(w2_scale.size(1) * kGroupSize, w2.size(1));
  CHECK_EQ(w2_scale.size(2) * kGroupSize, w2.size(2));
  CHECK(hidden_states.device() == w13_scale.device() &&
        hidden_states.device() == w2_scale.device() &&
        hidden_states.device() == w2.device());
}

void check_bf16_inputs(const torch::Tensor& hidden_states,
                       const torch::Tensor& w13,
                       const torch::Tensor& w2,
                       const torch::Tensor& topk_weights,
                       const torch::Tensor& topk_ids) {
  check_common_inputs(hidden_states, w13, w2, topk_weights, topk_ids);
  CHECK_EQ(w13.scalar_type(), torch::kBFloat16);
  CHECK_EQ(w2.scalar_type(), torch::kBFloat16);
}

}  // namespace

bool fused_moe_aot_available(int64_t num_tokens) {
  const KernelConfig* config = find_config(num_tokens, /*use_fp8=*/true);
  if (config == nullptr || artifact_root().empty()) {
    return false;
  }
  return readable(
             artifact_path(num_tokens, MoeStage::kGate, /*use_fp8=*/true)) &&
         readable(
             artifact_path(num_tokens, MoeStage::kDown, /*use_fp8=*/true)) &&
         ffi_artifact_available(kAlignUri) &&
         ffi_artifact_available(kActivationUri) &&
         ffi_artifact_available(kSumUri);
}

bool fused_moe_bf16_aot_available(int64_t num_tokens) {
  const KernelConfig* config = find_config(num_tokens, /*use_fp8=*/false);
  if (config == nullptr || bf16_artifact_root().empty()) {
    return false;
  }
  return readable(
             artifact_path(num_tokens, MoeStage::kGate, /*use_fp8=*/false)) &&
         readable(
             artifact_path(num_tokens, MoeStage::kDown, /*use_fp8=*/false)) &&
         ffi_artifact_available(kAlignUri) &&
         ffi_artifact_available(kActivationUri) &&
         ffi_artifact_available(kSumUri);
}

void prepare_fused_moe_aot(const torch::Device& device) {
  static std::mutex mutex;
  static std::set<int32_t> prepared_devices;
  const int32_t device_index = device.index();
  std::lock_guard<std::mutex> lock(mutex);
  if (prepared_devices.contains(device_index)) {
    return;
  }

  c10::musa::MUSAGuard device_guard(device_index);
  get_function(kAlignUri, "xllm_musa_moe_align_block_size");
  get_function(kActivationUri, "xllm_musa_moe_act_and_mul");
  get_function(kSumUri, "xllm_musa_moe_sum_reduce");
  for (const KernelConfig& config : kKernelConfigs) {
    if (!fused_moe_aot_available(config.batch_size)) {
      continue;
    }
    get_kernel(
        device_index, config.batch_size, MoeStage::kGate, /*use_fp8=*/true);
    get_kernel(
        device_index, config.batch_size, MoeStage::kDown, /*use_fp8=*/true);
  }
  prepared_devices.insert(device_index);
}

void prepare_fused_moe_bf16_aot(const torch::Device& device) {
  static std::mutex mutex;
  static std::set<int32_t> prepared_devices;
  const int32_t device_index = device.index();
  std::lock_guard<std::mutex> lock(mutex);
  if (prepared_devices.contains(device_index)) {
    return;
  }

  c10::musa::MUSAGuard device_guard(device_index);
  get_function(kAlignUri, "xllm_musa_moe_align_block_size");
  get_function(kActivationUri, "xllm_musa_moe_act_and_mul");
  get_function(kSumUri, "xllm_musa_moe_sum_reduce");
  for (const KernelConfig& config : kBf16KernelConfigs) {
    if (!fused_moe_bf16_aot_available(config.batch_size)) {
      continue;
    }
    get_kernel(
        device_index, config.batch_size, MoeStage::kGate, /*use_fp8=*/false);
    get_kernel(
        device_index, config.batch_size, MoeStage::kDown, /*use_fp8=*/false);
  }
  prepared_devices.insert(device_index);
}

torch::Tensor fused_moe_aot_fp8(const torch::Tensor& hidden_states,
                                const torch::Tensor& w13,
                                const torch::Tensor& w13_scale,
                                const torch::Tensor& w2,
                                const torch::Tensor& w2_scale,
                                const torch::Tensor& topk_weights,
                                const torch::Tensor& topk_ids) {
  check_fp8_inputs(
      hidden_states, w13, w13_scale, w2, w2_scale, topk_weights, topk_ids);
  const KernelConfig* config =
      find_config(hidden_states.size(0), /*use_fp8=*/true);
  CHECK(config != nullptr)
      << "MUSA fused MoE AOT supports decode batches 1 through 8; got "
      << hidden_states.size(0);
  CHECK(fused_moe_aot_available(hidden_states.size(0)))
      << "MUSA fused MoE AOT artifacts are unavailable for batch "
      << hidden_states.size(0);

  auto [sorted_token_ids, expert_ids, num_tokens_post_padded] =
      align_topk_ids(topk_ids, w13.size(0), config->block_m);
  auto [quantized_hidden, hidden_scale] =
      per_token_group_quant_fp8(hidden_states, kGroupSize);
  auto gate_up = launch_gemm(quantized_hidden,
                             hidden_scale,
                             w13,
                             w13_scale,
                             topk_weights,
                             sorted_token_ids,
                             expert_ids,
                             num_tokens_post_padded,
                             *config,
                             MoeStage::kGate,
                             /*use_fp8=*/true);
  auto activated = activate_gate_up(gate_up, topk_ids);
  auto [quantized_activated, activated_scale] =
      per_token_group_quant_fp8(activated, kGroupSize);
  auto down = launch_gemm(quantized_activated,
                          activated_scale,
                          w2,
                          w2_scale,
                          topk_weights,
                          sorted_token_ids,
                          expert_ids,
                          num_tokens_post_padded,
                          *config,
                          MoeStage::kDown,
                          /*use_fp8=*/true);
  return reduce_assignments(down);
}

torch::Tensor fused_moe_aot_bf16(const torch::Tensor& hidden_states,
                                 const torch::Tensor& w13,
                                 const torch::Tensor& w2,
                                 const torch::Tensor& topk_weights,
                                 const torch::Tensor& topk_ids) {
  check_bf16_inputs(hidden_states, w13, w2, topk_weights, topk_ids);
  const KernelConfig* config =
      find_config(hidden_states.size(0), /*use_fp8=*/false);
  CHECK(config != nullptr)
      << "MUSA BF16 fused MoE AOT supports decode batches 1 through 8; got "
      << hidden_states.size(0);
  CHECK(fused_moe_bf16_aot_available(hidden_states.size(0)))
      << "MUSA BF16 fused MoE AOT artifacts are unavailable for batch "
      << hidden_states.size(0);

  auto [sorted_token_ids, expert_ids, num_tokens_post_padded] =
      align_topk_ids(topk_ids, w13.size(0), config->block_m);
  const torch::Tensor no_scale;
  auto gate_up = launch_gemm(hidden_states,
                             no_scale,
                             w13,
                             no_scale,
                             topk_weights,
                             sorted_token_ids,
                             expert_ids,
                             num_tokens_post_padded,
                             *config,
                             MoeStage::kGate,
                             /*use_fp8=*/false);
  auto activated = activate_gate_up(gate_up, topk_ids);
  auto down = launch_gemm(activated,
                          no_scale,
                          w2,
                          no_scale,
                          topk_weights,
                          sorted_token_ids,
                          expert_ids,
                          num_tokens_post_padded,
                          *config,
                          MoeStage::kDown,
                          /*use_fp8=*/false);
  return reduce_assignments(down);
}

}  // namespace xllm::kernel::musa

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

#include "core/runtime/vlm_forward_params.h"

#include "core/framework/config/execution_config.h"

namespace xllm {

VlmForwardInput VlmForwardInput::to(const torch::Device& device,
                                    torch::ScalarType dtype) const {
  if (runtime.device_tensors_ready) {
    VlmForwardInput inputs = clone();
    inputs.runtime.input_host_buffer_has_layout = false;
    return inputs;
  }

  const bool use_contiguous_input_buffer =
      ::xllm::ExecutionConfig::get_instance().use_contiguous_input_buffer() &&
      detail::supports_contiguous_forward_input_buffer(device);
  if (runtime.input_host_buffer_has_layout) {
    VlmForwardInput buffer_inputs;
    if (detail::unpack_from_input_host_buffer(
            *this, device, dtype, buffer_inputs, use_contiguous_input_buffer)) {
      if (buffer_inputs.runtime.device_tensors_ready) {
        buffer_inputs.runtime.input_host_buffer_has_layout = false;
        return buffer_inputs;
      }
      return buffer_inputs.to(device, dtype);
    }
  }

  if (use_contiguous_input_buffer) {
    VlmForwardInput contiguous_inputs;
    if (to_contiguous_input_buffer(device, contiguous_inputs)) {
      return contiguous_inputs;
    }
  }

  VlmForwardInput inputs;
  set_host_views(inputs);
  const torch::Tensor& source_token_ids =
      inputs.token_ids_host.defined() ? inputs.token_ids_host : token_ids;
  const torch::Tensor& source_positions =
      inputs.positions_host.defined() ? inputs.positions_host : positions;
  inputs.token_ids = safe_to(source_token_ids, device, /*non_blocking=*/true);
  inputs.positions = detail::normalize_positions_for_device(
      safe_to(source_positions, device, /*non_blocking=*/true));
  inputs.input_params = input_params.to(device);
  inputs.sampling_params = sampling_params.to(device, dtype);
  copy_metadata_to(inputs);
  inputs.runtime.input_host_buffer = runtime.input_host_buffer;
  inputs.runtime.device_input_buffer = runtime.device_input_buffer;
  inputs.runtime.device_tensors_ready = true;
  inputs.runtime.kv_slot_layout = runtime.kv_slot_layout;
  return inputs;
}

bool VlmForwardInput::to_contiguous_input_buffer(
    const torch::Device& device,
    VlmForwardInput& inputs) const {
  copy_metadata_to(inputs);
  set_host_views(inputs);

  const VlmModelParams& source_params = input_params;
  if ((token_ids.defined() && !inputs.token_ids_host.defined()) ||
      (positions.defined() && !inputs.positions_host.defined()) ||
      source_params.multimodal.mm_data.valid() ||
      !source_params.multimodal.deep_stacks.empty()) {
    return false;
  }

  inputs.input_params = source_params.clone();
  detail::clear_contiguous_input_buffer_tensor_targets(inputs.input_params);
  inputs.sampling_params = sampling_params;

  torch::Tensor positions_for_device =
      detail::normalize_positions_for_device(inputs.positions_host);
  detail::ForwardInputBufferPlan plan;
  if (!plan.add(inputs.token_ids_host, &inputs.token_ids) ||
      !plan.add(positions_for_device, &inputs.positions) ||
      !detail::add_attention_to_plan(
          source_params.attention, inputs.input_params.attention, plan) ||
      !detail::add_model_tensors_to_plan(
          source_params, inputs.input_params, plan) ||
      !detail::add_sampling_to_plan(
          sampling_params, inputs.sampling_params, plan)) {
    return false;
  }

  const uint64_t total_bytes = plan.prepare_layout();
  if (total_bytes > 0) {
    inputs.runtime.input_host_buffer = plan.build_host_buffer(total_bytes);
    inputs.runtime.device_input_buffer =
        safe_to(inputs.runtime.input_host_buffer,
                torch::TensorOptions().dtype(torch::kUInt8).device(device),
                /*non_blocking=*/true);
    plan.bind_device_views(inputs.runtime.device_input_buffer, device);
  }

  inputs.runtime.device_tensors_ready = true;
  inputs.runtime.input_host_buffer_has_layout = false;
  return true;
}

}  // namespace xllm

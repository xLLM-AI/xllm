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
  return inputs;
}

}  // namespace xllm

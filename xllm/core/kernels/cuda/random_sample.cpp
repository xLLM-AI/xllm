/* Copyright 2025-2026 The xLLM Authors.

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

#include <ATen/cuda/CUDAGeneratorImpl.h>

#include <cstdint>
#include <mutex>
#include <tuple>
#include <unordered_map>

#include "cuda_ops_api.h"

namespace {

at::Generator get_default_generator(c10::DeviceIndex device_index) {
  static std::unordered_map<c10::DeviceIndex, at::Generator> cache;
  static std::mutex mu;
  std::lock_guard<std::mutex> lock(mu);
  auto it = cache.find(device_index);
  if (it != cache.end()) {
    return it->second;
  }
  at::globalContext().lazyInitCUDA();
  at::Generator gen = at::cuda::detail::getDefaultCUDAGenerator(device_index);
  cache.emplace(device_index, gen);
  return gen;
}

std::tuple<int64_t, int64_t> get_seed_and_offset(
    int64_t increment,
    const torch::Device& device,
    c10::optional<at::Generator> generator = c10::nullopt) {
  at::Generator gen = generator.has_value()
                          ? generator.value()
                          : get_default_generator(device.index());
  std::lock_guard<std::mutex> lock(gen.mutex());
  auto* cuda_gen = at::check_generator<at::CUDAGeneratorImpl>(gen);

  int64_t seed = static_cast<int64_t>(cuda_gen->current_seed());
  int64_t offset = static_cast<int64_t>(cuda_gen->get_offset());
  offset += (increment + 3) / 4 * 4;
  cuda_gen->set_offset(static_cast<uint64_t>(offset));

  return std::make_tuple(seed, offset);
}
}  // namespace

namespace xllm::kernel::cuda {

torch::Tensor random_sample(const torch::Tensor& probs) {
  CHECK(probs.dim() == 2 || probs.dim() == 3)
      << "probs must be a 2D or 3D tensor";

  torch::Tensor flat_probs;
  if (probs.dim() == 3) {
    flat_probs = probs.reshape({-1, probs.size(2)});
  } else {
    flat_probs = probs;
  }

  const torch::Device device = flat_probs.device();
  int64_t batch_size = flat_probs.size(0);
  torch::ScalarType out_dtype = torch::kInt64;
  torch::Tensor samples =
      torch::empty({batch_size}, torch::dtype(out_dtype).device(device));
  auto [seed, offset] = get_seed_and_offset(batch_size, device);

  get_function(/*uri=*/"sampling",
               /*func_name=*/"sampling_from_probs")(
      to_ffi_tensor(flat_probs),
      to_ffi_tensor(samples),
      /*maybe_indices=*/ffi::Optional<ffi::Tensor>(),
      /*deterministic=*/true,
      /*philox_seed=*/seed,
      /*philox_offset=*/offset);

  if (probs.dim() == 3) {
    return samples.reshape({probs.size(0), probs.size(1)});
  }
  return samples.flatten();
}

}  // namespace xllm::kernel::cuda

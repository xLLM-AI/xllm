/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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

#include "framework/sampling/gumbel_sampling.h"

#include <glog/logging.h>

namespace xllm {

torch::Tensor sample_gumbel_noise(int64_t batch_size,
                                  int64_t num_steps,
                                  int64_t num_classes,
                                  const SamplingParameters& sampling_params,
                                  const torch::Device& device) {
  CHECK_GT(batch_size, 0);
  CHECK_GT(num_steps, 0);
  CHECK_GT(num_classes, 0);
  const torch::TensorOptions float_options =
      torch::TensorOptions().dtype(torch::kFloat32).device(device);
  if (sampling_params.all_greedy_sample) {
    return torch::zeros({batch_size, num_steps, num_classes}, float_options);
  }
  CHECK(sampling_params.do_sample.defined())
      << "Gumbel sampling requires do_sample to be defined";

  // Gumbel(0, 1) = -log(-log(U)), U ~ Uniform(0, 1).
  torch::Tensor uniform =
      torch::rand({batch_size, num_steps, num_classes}, float_options);
  torch::Tensor gumbel = uniform.log().neg().log().neg();
  if (!sampling_params.all_random_sample) {
    torch::Tensor sample_mask =
        sampling_params.do_sample.to(float_options).view({batch_size, 1, 1});
    gumbel = gumbel * sample_mask;
  }
  return gumbel;
}

void apply_selector_temperatures(torch::Tensor& logits,
                                 const torch::Tensor& temperatures,
                                 int64_t batch_size) {
  CHECK(temperatures.defined());
  CHECK_EQ(temperatures.dim(), 1);
  CHECK_EQ(temperatures.size(0), batch_size);
  const torch::TensorOptions float_options =
      torch::TensorOptions().dtype(torch::kFloat32).device(logits.device());
  torch::Tensor scaled =
      temperatures.to(float_options).view({batch_size, 1, 1, 1});
  scaled = scaled.masked_fill(scaled.eq(0), 1.0);
  logits.div_(scaled);
}

torch::Tensor gumbel_argmax(const torch::Tensor& log_probs,
                            const torch::Tensor& gumbel_noise) {
  return (log_probs + gumbel_noise).argmax(/*dim=*/-1);
}

}  // namespace xllm

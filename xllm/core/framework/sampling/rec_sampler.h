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

#pragma once
#include <torch/torch.h>

#include <functional>
#include <memory>
#include <optional>

#include "sampling_params.h"
#include "util/rec_model_utils.h"

namespace xllm {

struct RecSamplingContext {
  using DeviceConstrainedSampler = std::function<std::optional<SampleOutput>(
      torch::Tensor& logits,
      const SamplingParameters& params,
      const torch::Tensor& sequence_group,
      int32_t current_step,
      int32_t beam_width)>;

  torch::Tensor sequence_group;
  int32_t current_step = -1;
  int32_t beam_width = 1;
  DeviceConstrainedSampler device_constrained_sampler;
};
class Sampler;

class RecSampler {
 public:
  explicit RecSampler(RecPipelineType pipeline_type);
  ~RecSampler() = default;

  // logits: [batch_size, vocab_size]
  SampleOutput forward(torch::Tensor& logits,
                       const SamplingParameters& params,
                       const torch::Tensor& filter_mask = torch::Tensor(),
                       const RecSamplingContext* context = nullptr) const;

 private:
  class SamplingStrategy {
   public:
    virtual ~SamplingStrategy() = default;
    virtual SampleOutput forward(torch::Tensor& logits,
                                 const SamplingParameters& params,
                                 const torch::Tensor& filter_mask,
                                 const RecSamplingContext* context) const = 0;
  };

  class DefaultSamplingStrategy final : public SamplingStrategy {
   public:
    explicit DefaultSamplingStrategy(const Sampler& sampler);
    SampleOutput forward(torch::Tensor& logits,
                         const SamplingParameters& params,
                         const torch::Tensor& filter_mask,
                         const RecSamplingContext* context) const override;

   private:
    const Sampler& sampler_;
  };

  class OneRecConstrainedSamplingStrategy final : public SamplingStrategy {
   public:
    explicit OneRecConstrainedSamplingStrategy(const Sampler& sampler);
    SampleOutput forward(torch::Tensor& logits,
                         const SamplingParameters& params,
                         const torch::Tensor& filter_mask,
                         const RecSamplingContext* context) const override;

   private:
    const Sampler& sampler_;
  };

  class MultiRoundFastPathSamplingStrategy final : public SamplingStrategy {
   public:
    explicit MultiRoundFastPathSamplingStrategy(const Sampler& sampler);
    SampleOutput forward(torch::Tensor& logits,
                         const SamplingParameters& params,
                         const torch::Tensor& filter_mask,
                         const RecSamplingContext* context) const override;

   private:
    const Sampler& sampler_;
  };

  static std::unique_ptr<SamplingStrategy> create_sampling_strategy(
      RecPipelineType type,
      const Sampler& sampler);

  std::unique_ptr<Sampler> sampler_;
  std::unique_ptr<SamplingStrategy> strategy_;
};

}  // namespace xllm

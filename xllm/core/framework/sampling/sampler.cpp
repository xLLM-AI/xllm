/* Copyright 2025-2026 The xLLM Authors.
Copyright 2024 The ScaleLLM Authors. All Rights Reserved.

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

#include "core/framework/sampling/sampler.h"

#include <glog/logging.h>
#include <torch/torch.h>

#include <algorithm>
#include <limits>
#include <mutex>
#include <tuple>

#if defined(USE_NPU)
#include <torch_npu/csrc/aten/NPUGeneratorImpl.h>

#include "core/kernels/npu/tilelang/tilelang_ops_api.h"
#endif

#include "core/framework/config/model_config.h"
#include "core/framework/sampling/logits_processor.h"
#include "core/framework/sampling/npu_sampling_policy.h"
#include "core/framework/sampling/sampling_params.h"

namespace xllm {
namespace {

uint64_t sampling_seed(int64_t seed, int64_t offset) {
  uint64_t value = static_cast<uint64_t>(seed) +
                   static_cast<uint64_t>(offset) * 0x9e3779b97f4a7c15ULL;
  value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
  value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
  return value ^ (value >> 31);
}

torch::Generator make_sampling_generator(const torch::Device& device) {
#if defined(USE_NPU)
  if (device.type() == torch::kPrivateUse1) {
    // Native execution does not register the generic default-generator hook.
    return at_npu::detail::createNPUGenerator(device.index());
  }
#endif
  auto generator = torch::globalContext().defaultGenerator(device);
  std::lock_guard<std::mutex> lock(generator.mutex());
  return generator.clone();
}

}  // namespace

torch::Tensor gumbel_noise(const SamplingParameters& params,
                           torch::IntArrayRef sizes,
                           const torch::Device& device) {
  torch::Tensor noise = torch::empty(
      sizes, torch::TensorOptions().dtype(torch::kFloat32).device(device));
  if (params.seeds().defined()) {
    CHECK(sizes.size() == 2 || sizes.size() == 3);
    const torch::Tensor seeds = params.seeds().to(torch::kCPU);
    const torch::Tensor offsets = params.seed_offsets().to(torch::kCPU);
    const torch::Tensor seeded = seeds.ne(-1);
    const torch::Tensor random_rows =
        params.sampling_mode() == SamplingMode::MIXED
            ? params.require_do_sample().to(torch::kCPU)
            : torch::ones({sizes[0]}, torch::kBool);
    torch::Generator generator = make_sampling_generator(device);
    for (int64_t row = 0; row < sizes[0]; ++row) {
      if (!random_rows[row].item<bool>()) {
        noise[row].fill_(1.0);
        continue;
      }
      const int64_t columns = sizes.size() == 3 ? sizes[1] : 1;
      for (int64_t column = 0; column < columns; ++column) {
        std::optional<torch::Generator> row_generator;
        if (seeded[row].item<bool>()) {
          generator.set_current_seed(
              sampling_seed(seeds[row].item<int64_t>(),
                            offsets[row].item<int64_t>() + column));
          row_generator = generator;
        }
        torch::Tensor values =
            sizes.size() == 3 ? noise[row][column] : noise[row];
        values.exponential_(/*lambd=*/1.0, row_generator);
      }
    }
  } else {
    noise.exponential_(/*lambd=*/1.0);
  }
  noise.clamp_min_(std::numeric_limits<float>::min()).log_();
  // Fill after the log_: masking first would turn the sentinel into
  // log(0) = -inf.
  if (params.sampling_mode() == SamplingMode::MIXED) {
    noise.masked_fill_(
        greedy_mask(params, device)
            .view(batch_broadcast_shape(noise.size(0), noise.dim())),
        0.0);
  }
  return noise;
}

torch::Tensor categorical_sample(const torch::Tensor& logits) {
#if defined(USE_NPU)
  if (npu_sampling_enabled(NpuSamplingOptimization::VERIFICATION) &&
      kernel::npu::tilelang::supports_sampling_logits(logits) &&
      logits.size(0) >= 8) {
    const torch::Tensor uniforms =
        torch::rand({logits.size(0)}, logits.options().dtype(torch::kFloat32));
    return kernel::npu::tilelang::sampling_logits(logits, {}, {}, uniforms);
  }
#endif
  return gumbel_sample(logits);
}

torch::Tensor gumbel_sample(const torch::Tensor& logits) {
  // clamp_min_ before log_: an exact 0.0 Exp draw would become -inf noise.
  return gumbel_argmax(
      logits,
      torch::empty(
          logits.sizes(),
          torch::TensorOptions().dtype(torch::kFloat32).device(logits.device()))
          .exponential_(/*lambd=*/1.0)
          .clamp_min_(std::numeric_limits<float>::min())
          .log_());
}

torch::Tensor gumbel_argmax(const torch::Tensor& logits,
                            torch::Tensor gumbel_noise) {
  CHECK(gumbel_noise.defined())
      << "gumbel_argmax requires noise; greedy callers use greedy_sample";
  CHECK_EQ(gumbel_noise.sizes(), logits.sizes());
  // Elementwise, so the noise buffer doubles as the output.
  torch::sub_out(gumbel_noise, logits, gumbel_noise);
  return gumbel_noise.argmax(/*dim=*/-1);
}

SampleOutput sample(torch::Tensor& logits,
                    const SamplingParameters& params,
                    const torch::Tensor& filter_mask) {
  SampleOutput output;
  // vLLM reports probabilities from the unmodified model distribution.
  bool compact_logprobs = false;
#if defined(USE_NPU)
  compact_logprobs =
      npu_sampling_enabled(NpuSamplingOptimization::COMPACT_NORMALIZATION) &&
      params.logprobs() && params.max_top_logprobs() == 0 &&
      !params.use_beam_search() &&
      !ModelConfig::get_instance().enable_qwen3_reranker() &&
      kernel::npu::tilelang::supports_sampling_logits(logits) &&
      logits.size(0) >= 128 && logits.size(1) >= 65536 &&
      (logits.scalar_type() == torch::kBFloat16 ||
       logits.scalar_type() == torch::kFloat16);
#endif
  const torch::Tensor raw_logits =
      compact_logprobs ? logits.clone() : torch::Tensor();
  torch::Tensor raw_max;
  torch::Tensor raw_log_sum;
#if defined(USE_NPU)
  if (compact_logprobs) {
    std::tie(raw_max, raw_log_sum) =
        kernel::npu::tilelang::row_log_normalization(logits);
  }
#endif
  const torch::Tensor raw_logprobs =
      params.logprobs() && !params.use_beam_search() && !compact_logprobs
          ? torch::log_softmax(logits, /*dim=*/-1, torch::kFloat32)
          : torch::Tensor();

  process_logits(logits, params, filter_mask);

  // Keep the model dtype; consumers upcast where the reduction needs fp32.
  if (params.return_logits()) {
    output.draft_logits = logits;
  }

  if (params.sampling_mode() == SamplingMode::MIXED) {
    const torch::Tensor& do_sample = params.require_do_sample();
    CHECK_EQ(do_sample.dim(), 1)
        << "params.do_sample() must be 1D [num_seqs], got "
        << do_sample.sizes();
    CHECK_EQ(logits.size(0), do_sample.size(0));
  }

  torch::Tensor samples;
#if defined(USE_NPU)
  const bool batched_seed_sampling =
      npu_sampling_enabled(NpuSamplingOptimization::BATCHED) &&
      params.sampling_mode() != SamplingMode::GREEDY &&
      kernel::npu::tilelang::supports_sampling_logits(logits) &&
      (!params.seeds().defined() ||
       (params.seeds().dim() == 1 && params.seed_offsets().defined() &&
        params.seed_offsets().dim() == 1 && params.seeds().is_contiguous() &&
        params.seed_offsets().is_contiguous()));
  if (batched_seed_sampling) {
    const auto uniform_options = logits.options().dtype(torch::kFloat32);
    const torch::Tensor uniforms =
        (!params.seeds().defined() || params.has_unseeded_sample())
            ? torch::rand({logits.size(0)}, uniform_options)
            : torch::Tensor();
#if defined(XLLM_NPU_SAMPLING_ABLATION)
    if (!npu_sampling_enabled(NpuSamplingOptimization::FUSED_LOGITS)) {
      const torch::Tensor weights =
          torch::softmax(logits, /*dim=*/-1, torch::kFloat32);
      samples = kernel::npu::tilelang::seeded_sample(
          weights, params.seeds(), params.seed_offsets(), uniforms);
    } else
#endif
    {
      samples = kernel::npu::tilelang::sampling_logits(
          logits, params.seeds(), params.seed_offsets(), uniforms);
    }
    if (params.sampling_mode() == SamplingMode::MIXED) {
      samples = torch::where(
          params.require_do_sample(), samples, greedy_sample(logits));
    }
  } else
#endif
      if (params.seeds().defined() &&
          params.sampling_mode() != SamplingMode::GREEDY) {
    samples = gumbel_argmax(
        logits, gumbel_noise(params, logits.sizes(), logits.device()));
  } else if (params.sampling_mode() == SamplingMode::RANDOM) {
    samples = gumbel_sample(logits);
  } else if (params.sampling_mode() == SamplingMode::GREEDY) {
    samples = greedy_sample(logits);
  } else {
    torch::Tensor noise = gumbel_noise(params, logits.sizes(), logits.device());
    samples = gumbel_argmax(logits, noise);
  }
  output.next_tokens = samples;

  if (params.logprobs()) {
    if (compact_logprobs) {
      output.logprobs = raw_logits.gather(/*dim=*/-1, samples.view({-1, 1}))
                            .to(torch::kFloat32)
                            .view({-1})
                            .sub(raw_max)
                            .sub_(raw_log_sum);
      return output;
    }
    if (::xllm::ModelConfig::get_instance().enable_qwen3_reranker()) {
      int32_t false_id = 2152;  // "no"
      int32_t true_id = 9693;   // "yes"
      auto indices =
          torch::tensor({false_id, true_id}, torch::kLong).to(samples.device());
      logits = logits.index_select(/*dim=*/1, indices);
      auto logprobs =
          torch::log_softmax(logits, /*dim=*/1, /*dtype=*/torch::kFloat32);
      logprobs = logprobs.index({torch::indexing::Slice(), 1});
      output.logprobs = logprobs.view({-1}).exp();
      return output;
    }
    // log_softmax is equivalent to log(softmax) but more numerically stable
    const torch::Tensor logprobs =
        raw_logprobs.defined()
            ? raw_logprobs
            : torch::log_softmax(logits, /*dim=*/-1, torch::kFloat32);
    auto selected_logprobs = logprobs.gather(/*dim=*/-1, samples.view({-1, 1}));
    output.logprobs = selected_logprobs.view({-1});

    if (params.max_top_logprobs() > 0) {
      // max_top_logprobs is the batch-wide max, so one request asking for
      // more than the vocabulary would fail every request in the batch. The
      // factories reject this up front; clamp as a last line of defense —
      // every consumer sizes by the returned row width.
      const int64_t k =
          std::min<int64_t>(params.max_top_logprobs(), logprobs.size(-1));
      auto [values, indices] = logprobs.topk(k, /*dim=*/-1);
      output.top_logprobs = values;
      output.top_tokens = indices;
    }
  }

  return output;
}

torch::Tensor greedy_sample(const torch::Tensor& logits) {
  return logits.argmax(/*dim=*/-1);
}

}  // namespace xllm

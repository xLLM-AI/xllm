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

#include "mm_embed_vlm_worker_impl.h"

#include <c10/core/DeviceGuard.h>
#include <folly/Unit.h>
#include <folly/futures/Future.h>
#include <glog/logging.h>
#include <torch/torch.h>

#include <memory>
#include <optional>
#include <utility>

#include "common/metrics.h"
#include "framework/kv_cache/kv_cache.h"
#include "framework/model/model_input_params.h"
#include "framework/state_dict/state_dict.h"
#include "models/model_registry.h"
#include "options.h"
#include "util/timer.h"

namespace xllm {

MMEmbedVLMWorkerImpl::MMEmbedVLMWorkerImpl(const ParallelArgs& parallel_args,
                                           const torch::Device& device,
                                           const runtime::Options& options)
    : WorkerImpl(parallel_args, device, options) {}

bool MMEmbedVLMWorkerImpl::init_model(ModelContext& context) {
  CHECK(model_ == nullptr) << "Model is already initialized.";

  context.set_encoder_embedding_mode(true);
  model_ = create_vlm_model(context);
  CHECK(model_ != nullptr) << "Failed to create model.";
  model_executor_ = std::make_unique<Executor>(
      model_.get(), context.get_model_args(), device_, options_);

  return true;
}

std::optional<ForwardOutput> MMEmbedVLMWorkerImpl::step(
    const ForwardInput& input) {
  torch::DeviceGuard device_guard(device_);
  auto ret = device_.synchronize_default_stream();

  Timer timer;

  // TODO remove language params in only vision model forward.
  // TODO to adapt multi stream parallel later, just use [0] temporarily
  // all tensors should be on the same device as model
  auto flatten_tokens = input.token_ids.to(device_);
  auto flatten_positions = input.positions.to(device_);
  auto params = input.input_params.to(device_);
  auto sampling_params = input.sampling_params.to(device_, dtype_);
  CHECK(input.sampling_params.is_embeddings)
      << "Only mm embedding is supported.";

  // call model executor forward to get hidden states
  CausalVLM* vlm_model = dynamic_cast<CausalVLM*>(model_.get());
  CHECK(vlm_model != nullptr) << "Model is not a CausalVLM.";
  auto encode_output = vlm_model->encode(params);
  const auto it = encode_output.find("image|embedding");
  if (it == encode_output.end() ||
      !std::holds_alternative<std::vector<torch::Tensor>>(it->second)) {
    LOG(ERROR) << "Invalid 'image|embedding' in encode output.";
    return std::nullopt;
  }
  const auto& mm_embeddings = std::get<std::vector<torch::Tensor>>(it->second);

  ret = device_.synchronize_default_stream();
  COUNTER_ADD(execution_latency_seconds_model, timer.elapsed_seconds());

  if (!driver_) {
    return std::nullopt;
  }

  // driver prepare model output

  ForwardOutput output;
  SampleOutput sample_output;
  // Group the flattened per-image embeddings by sequence, using each sequence's
  // image count from the host-side mm_data. Outer = sequence, inner = images.
  const auto& mm_data_vec = input.input_params.multimodal.mm_data.mm_data_vec();
  sample_output.mm_embeddings.reserve(mm_data_vec.size());
  size_t image_idx = 0;
  for (const auto& seq_mm_data : mm_data_vec) {
    const size_t seq_image_count = seq_mm_data.size();
    std::vector<torch::Tensor> seq_mm_embeddings;
    seq_mm_embeddings.reserve(seq_image_count);
    for (size_t i = 0; i < seq_image_count; ++i) {
      CHECK_LT(image_idx, mm_embeddings.size());
      seq_mm_embeddings.push_back(mm_embeddings[image_idx++]);
    }
    sample_output.mm_embeddings.push_back(std::move(seq_mm_embeddings));
  }
  CHECK_EQ(image_idx, mm_embeddings.size())
      << "mm_embedding count mismatch: grouped " << image_idx << " but got "
      << mm_embeddings.size();
  output.sample_output = sample_output;

  return output;
}

}  // namespace xllm
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

#include "runtime/speculative_worker_impl.h"

#include <gtest/gtest.h>

#include <memory>
#include <optional>
#include <type_traits>
#include <utility>
#include <vector>

#include "common/metrics.h"
#include "framework/model/causal_vlm.h"
#include "framework/multimodal/mm_data.h"
#include "models/model_registry.h"
#include "runtime/vlm_executor_impl.h"
#include "runtime/vlm_worker_impl.h"

namespace xllm {

namespace {

template <typename WorkerType, typename = void>
struct HasNoSyncForward : std::false_type {};

template <typename WorkerType>
struct HasNoSyncForward<
    WorkerType,
    std::void_t<decltype(&WorkerType::execute_no_sync_on_stream)>>
    : std::true_type {};

template <typename WorkerType, typename = void>
struct HasWeightSharingAccessors : std::false_type {};

template <typename WorkerType>
struct HasWeightSharingAccessors<
    WorkerType,
    std::void_t<decltype(&WorkerType::get_lm_head),
                decltype(&WorkerType::set_lm_head),
                decltype(&WorkerType::get_word_embedding),
                decltype(&WorkerType::set_word_embedding)>> : std::true_type {};

class FakeCausalVLM final : public CausalVLM {
 public:
  explicit FakeCausalVLM(const torch::Device& device)
      : device_(device),
        options_(torch::dtype(torch::kFloat32).device(device)) {}

  MMDict encode(const ModelInputParams& params) override {
    if (params.multimodal.mm_data.data().empty()) {
      return {};
    }
    return {{"image|embedding",
             std::vector<torch::Tensor>{torch::zeros({1, 1}, options_)}}};
  }

  torch::Tensor get_input_embeddings(
      const torch::Tensor& input_ids,
      const ModelInputParams& /*params*/) override {
    return torch::zeros({input_ids.size(0), 1}, options_);
  }

  ModelOutput forward(const torch::Tensor& tokens,
                      const torch::Tensor& /*positions*/,
                      std::vector<KVCache>& /*kv_caches*/,
                      const ModelInputParams& /*params*/) override {
    return ModelOutput(torch::zeros({tokens.size(0), 1}, options_));
  }

  torch::Tensor logits(const torch::Tensor& hidden_states,
                       const torch::Tensor& /*selected_idxes*/) override {
    return hidden_states;
  }

  void load_model(std::unique_ptr<ModelLoader> /*loader*/) override {}

  torch::Device device() const override { return device_; }

  void prepare_expert_weight(
      int32_t /*layer_id*/,
      const std::vector<int32_t>& /*expert_ids*/) override {}

  void update_expert_weight(int32_t /*layer_id*/) override {}

  const torch::TensorOptions& options() const override { return options_; }

 private:
  torch::Device device_;
  torch::TensorOptions options_;
};

ModelInputParams make_image_prefill_params() {
  MMData image_data;
  MMDataItem& image = image_data.add(MMType::IMAGE);
  image.add("pixel_values", torch::ones({1}, torch::kFloat));

  MMItemState& state = image.mutable_state();
  state.mutable_seq_index() = 0;
  state.mutable_token_pos().length = 1;
  state.mutable_mm_token_mask() = torch::ones({1}, torch::kBool);
  state.mutable_mm_token_num() = 1;
  state.mutable_schedule_data().end_pos = 1;

  ModelInputParams params;
  params.attention.host.kv_seq_lens = {0, 1};
  params.attention.host.q_seq_lens = {0, 1};
  params.multimodal.mm_data.batch({image_data});
  return params;
}

ModelInputParams make_text_params() {
  ModelInputParams params;
  params.attention.host.kv_seq_lens = {0, 1};
  params.attention.host.q_seq_lens = {0, 1};
  return params;
}

class InspectableSpeculativeWorker final : public SpeculativeWorkerImpl {
 public:
  using TargetWorker = decltype(impl_);

  static constexpr bool uses_llm_target_type() {
    return std::is_same<TargetWorker, std::unique_ptr<LLMWorkerImpl>>::value;
  }

 private:
  std::optional<ForwardOutput> step_prefill(
      const ForwardInput& /*input*/) override {
    return std::nullopt;
  }

  std::optional<ForwardOutput> step_decode(
      const ForwardInput& /*input*/) override {
    return std::nullopt;
  }

  std::optional<ForwardOutput> step_empty(
      const ForwardInput& /*input*/) override {
    return std::nullopt;
  }
};

TEST(SpeculativeOwnershipTest, TargetUsesLlmOrchestrationTypes) {
  static_assert(InspectableSpeculativeWorker::uses_llm_target_type());
}

TEST(SpeculativeOwnershipTest, SpeculativeInterfacesStayOnLlmWorker) {
  static_assert(!HasNoSyncForward<WorkerImpl>::value);
  static_assert(!HasNoSyncForward<VLMWorkerImpl>::value);
  static_assert(HasNoSyncForward<LLMWorkerImpl>::value);
  static_assert(!HasWeightSharingAccessors<WorkerImpl>::value);
  static_assert(!HasWeightSharingAccessors<VLMWorkerImpl>::value);
  static_assert(HasWeightSharingAccessors<LLMWorkerImpl>::value);
}

TEST(SpeculativeOwnershipTest, VlmModelAndExecutorTypesRemainCompatible) {
  using VlmModelFactoryResult =
      decltype(create_vlm_model(std::declval<const ModelContext&>()));
  static_assert(
      std::is_same_v<VlmModelFactoryResult, std::unique_ptr<CausalVLM>>);
  static_assert(std::is_base_of_v<ExecutorImpl, VlmExecutorImpl>);
}

TEST(VlmExecutorImplTest, CountsOnlyUncachedImagePrefill) {
  const torch::Device device(torch::kCPU);
  FakeCausalVLM model(device);
  ModelArgs args;
  runtime::Options options;
  VlmExecutorImpl executor(&model, args, device, options);
  std::vector<KVCache> kv_caches;
  const torch::Tensor tokens = torch::tensor({1}, torch::kInt64);
  const torch::Tensor positions = torch::tensor({0}, torch::kInt64);
  const double invocations_before =
      COUNTER_vlm_encoder_effective_invocations_total.get_value();

  ModelInputParams prefill_params = make_image_prefill_params();
  executor.run(tokens, positions, kv_caches, prefill_params);
  EXPECT_EQ(COUNTER_vlm_encoder_effective_invocations_total.get_value(),
            invocations_before + 1);

  ModelInputParams decode_params = make_text_params();
  executor.run(tokens, positions, kv_caches, decode_params);
  EXPECT_EQ(COUNTER_vlm_encoder_effective_invocations_total.get_value(),
            invocations_before + 1);

  ModelInputParams text_prefill_params = make_text_params();
  executor.run(tokens, positions, kv_caches, text_prefill_params);
  EXPECT_EQ(COUNTER_vlm_encoder_effective_invocations_total.get_value(),
            invocations_before + 1);
}

}  // namespace
}  // namespace xllm

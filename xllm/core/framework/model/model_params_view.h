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

#pragma once
#include <memory>
#include <type_traits>
#include <utility>

#include "core/framework/model/llm_model_params.h"
#include "core/framework/model/vlm_model_params.h"
namespace xllm {

class ModelParamsView;
class ModelInputSnapshotOwner {
 public:
  virtual ~ModelInputSnapshotOwner() = default;
  virtual ModelParamsView view() = 0;
};
class ModelInputSnapshot final {
 public:
  template <typename NativeParams>
  explicit ModelInputSnapshot(NativeParams params);
  ModelInputSnapshot(const ModelInputSnapshot&) = delete;
  ModelInputSnapshot& operator=(const ModelInputSnapshot&) = delete;
  ModelInputSnapshot(ModelInputSnapshot&&) = default;
  ModelInputSnapshot& operator=(ModelInputSnapshot&&) = default;
  // A returned view retains this explicit native snapshot and can outlive
  // the snapshot handle, including when returned from a graph helper.
  ModelParamsView view() const;

 private:
  std::shared_ptr<ModelInputSnapshotOwner> owner_;
};
// Shallow borrowed execution view: construction and copying never copy native
// host vectors or multimodal payloads.
class ModelParamsView final {
 private:
  enum class Domain { LLM, VLM };
  std::shared_ptr<ModelInputSnapshotOwner> snapshot_owner_;
  void* native_owner_;
  Domain domain_;

 public:
  template <typename Owner>
    requires(std::is_same_v<Owner, LlmModelParams> ||
             std::is_same_v<Owner, VlmModelParams>)
  explicit ModelParamsView(Owner& owner)
      : native_owner_(&owner),
        domain_(std::is_same_v<Owner, LlmModelParams> ? Domain::LLM
                                                      : Domain::VLM),
        meta(owner.meta),
        attention(owner.attention),
        embedding(owner.embedding),
        parallel(owner.parallel),
        expert(owner.expert),
        graph(owner.graph),
        linear_state_validity_mask(owner.linear_state_validity_mask),
        multi_block_tables(owner.multi_block_tables),
        prefill_without_cache(owner.prefill_without_cache),
        num_accepted_tokens(owner.num_accepted_tokens),
        mtp_topk_state(owner.mtp_topk_state),
        num_accepted_tokens_host(owner.num_accepted_tokens_host),
        attn_metadata(owner.attn_metadata),
        enable_graph(owner.enable_graph) {}
  ModelParamsView(const ModelParamsView&) = default;
  ModelParamsView(ModelParamsView&&) = default;
  ModelParamsView& operator=(const ModelParamsView&) = delete;
  ModelParamsView& operator=(ModelParamsView&&) = delete;
  ModelInputSnapshot clone() const;
  ModelInputSnapshot to(const torch::Device& device) const;
  bool synchronize_layer(int64_t layer_idx) const {
    if (parallel.layer_wise_load_synchronizer == nullptr) {
      return true;
    }
    CHECK_GE(layer_idx, 0) << "Layer index must be non-negative.";
    if (static_cast<uint64_t>(layer_idx) % parallel.layers_per_event == 0) {
      return parallel.layer_wise_load_synchronizer->synchronize_layer(
          layer_idx / parallel.layers_per_event);
    }
    return true;
  }

  bool record_layer(uint32_t layer_idx, const torch::Device& device) const {
#if defined(USE_MLU) || defined(USE_DCU)
    if (parallel.layer_synchronizer != nullptr) {
      return parallel.layer_synchronizer->record_current(layer_idx,
                                                         device.index());
    }
#else
    (void)layer_idx;
    (void)device;
#endif
    return true;
  }

  BatchInputMeta& meta;
  AttentionInputView attention;
  ModelEmbeddingInput& embedding;
  ParallelInput& parallel;
  ExpertInput& expert;
  ModelGraphInput& graph;
  std::vector<int64_t>& linear_state_validity_mask;
  std::vector<torch::Tensor>& multi_block_tables;
  bool& prefill_without_cache;
  torch::Tensor& num_accepted_tokens;
  MtpTopkStatePtr& mtp_topk_state;
  std::vector<int64_t>& num_accepted_tokens_host;
  std::shared_ptr<layer::AttentionMetadata>& attn_metadata;
  bool& enable_graph;

 private:
  friend class ModelInputSnapshot;
};
template <typename NativeParams>
class NativeModelInputSnapshotOwner final : public ModelInputSnapshotOwner {
 public:
  explicit NativeModelInputSnapshotOwner(NativeParams params)
      : params_(std::move(params)) {}
  ModelParamsView view() override { return ModelParamsView(params_); }

 private:
  NativeParams params_;
};
template <typename NativeParams>
ModelInputSnapshot::ModelInputSnapshot(NativeParams params)
    : owner_(std::make_shared<NativeModelInputSnapshotOwner<NativeParams>>(
          std::move(params))) {}
inline ModelParamsView ModelInputSnapshot::view() const {
  CHECK(owner_ != nullptr) << "cannot view an empty model input snapshot";
  ModelParamsView params = owner_->view();
  params.snapshot_owner_ = owner_;
  return params;
}
inline ModelInputSnapshot ModelParamsView::to(
    const torch::Device& device) const {
  if (domain_ == Domain::LLM) {
    return ModelInputSnapshot(
        static_cast<LlmModelParams*>(native_owner_)->to(device));
  }
  return ModelInputSnapshot(
      static_cast<VlmModelParams*>(native_owner_)->to(device));
}
inline ModelInputSnapshot ModelParamsView::clone() const {
  if (domain_ == Domain::LLM) {
    return ModelInputSnapshot(
        static_cast<LlmModelParams*>(native_owner_)->clone());
  }
  return ModelInputSnapshot(
      static_cast<VlmModelParams*>(native_owner_)->clone());
}
}  // namespace xllm

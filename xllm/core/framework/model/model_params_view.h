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
#include <utility>

#include "core/framework/model/llm_model_params.h"
#include "core/framework/model/vlm_model_params.h"
namespace xllm {
class EmbeddingInputView final {
 public:
  template <typename Common, typename Spec>
  EmbeddingInputView(Common& common, Spec& spec)
      : input_embedding(common.input_embedding),
        embedding_ids(common.embedding_ids),
        linear_state_ids(common.linear_state_ids),
        linear_state_indices(common.linear_state_indices),
        linear_state_read_ids(common.linear_state_read_ids),
        linear_state_read_indices(common.linear_state_read_indices),
        request_ids(common.request_ids),
        extra_token_ids(spec.extra_token_ids),
        mtp_shifted_token_ids(spec.mtp_shifted_token_ids),
        mtp_bootstrap_row_idxes(spec.mtp_bootstrap_row_idxes),
        mtp_bootstrap_embeddings(spec.mtp_bootstrap_embeddings) {}
  torch::Tensor& input_embedding;
  std::vector<int32_t>& embedding_ids;
  std::vector<int32_t>& linear_state_ids;
  torch::Tensor& linear_state_indices;
  std::vector<int32_t>& linear_state_read_ids;
  torch::Tensor& linear_state_read_indices;
  std::vector<std::string>& request_ids;
  std::vector<int32_t>& extra_token_ids;
  torch::Tensor& mtp_shifted_token_ids;
  std::vector<int32_t>& mtp_bootstrap_row_idxes;
  torch::Tensor& mtp_bootstrap_embeddings;
};

class GraphInputView final {
 public:
  template <typename Common, typename Spec>
  GraphInputView(Common& common, Spec& spec)
      : attn_mask(common.attn_mask),
        tiling_data(common.tiling_data),
        use_expanded_decode_for_spec_verify_attention(
            spec.use_expanded_decode_for_spec_verify_attention),
        expanded_kv_seq_lens(spec.expanded_kv_seq_lens),
        expanded_block_tables(spec.expanded_block_tables),
        expanded_paged_kv_indptr(spec.expanded_paged_kv_indptr),
        expanded_paged_kv_indices(spec.expanded_paged_kv_indices),
        expanded_paged_kv_last_page_len(spec.expanded_paged_kv_last_page_len),
        expanded_tiling_data(spec.expanded_tiling_data),
        expanded_kv_seq_lens_vec(spec.expanded_kv_seq_lens_vec),
        input_tokens_override(spec.input_tokens_override),
        spec_verify_draft_token_sources(spec.spec_verify_draft_token_sources),
        spec_verify_source_addresses_stable(
            spec.spec_verify_source_addresses_stable),
        spec_verify_static_graph_tasks_prepared(
            spec.spec_verify_static_graph_tasks_prepared),
        dspark_adaptive(spec.dspark_adaptive)
#if defined(USE_DCU)
        ,
        use_dense_flash_attention(common.use_dense_flash_attention)
#endif
#if defined(USE_NPU)
        ,
        acl_graph_task_update_context(common.acl_graph_task_update_context)
#endif
  {
  }
  torch::Tensor& attn_mask;
  torch::Tensor& tiling_data;
  bool& use_expanded_decode_for_spec_verify_attention;
  torch::Tensor& expanded_kv_seq_lens;
  torch::Tensor& expanded_block_tables;
  torch::Tensor& expanded_paged_kv_indptr;
  torch::Tensor& expanded_paged_kv_indices;
  torch::Tensor& expanded_paged_kv_last_page_len;
  torch::Tensor& expanded_tiling_data;
  std::vector<int32_t>& expanded_kv_seq_lens_vec;
  torch::Tensor& input_tokens_override;
  std::vector<torch::Tensor>& spec_verify_draft_token_sources;
  bool& spec_verify_source_addresses_stable;
  bool& spec_verify_static_graph_tasks_prepared;
  DSparkAdaptiveGraphInput& dspark_adaptive;

#if defined(USE_DCU)
  bool& use_dense_flash_attention;
#endif
#if defined(USE_NPU)
  std::shared_ptr<npu::AclGraphTaskUpdateContext>&
      acl_graph_task_update_context;
#endif
};

class ModelParamsView;
class ModelInputSnapshotOwner {
 public:
  virtual ~ModelInputSnapshotOwner() = default;
  virtual ModelParamsView view() = 0;
  virtual std::shared_ptr<ModelInputSnapshotOwner> to(
      const torch::Device& device) const = 0;
};
class ModelInputSnapshot final {
 public:
  ModelInputSnapshot() = default;
  template <typename NativeParams>
  explicit ModelInputSnapshot(NativeParams params);
  ModelInputSnapshot(const ModelInputSnapshot&) = delete;
  ModelInputSnapshot& operator=(const ModelInputSnapshot&) = delete;
  ModelInputSnapshot(ModelInputSnapshot&&) = default;
  ModelInputSnapshot& operator=(ModelInputSnapshot&&) = default;
  // A returned view retains this explicit native snapshot and can outlive
  // the snapshot handle, including when returned from a graph helper.
  ModelParamsView view() const;
  ModelInputSnapshot to(const torch::Device& device) const;

 private:
  friend class ModelParamsView;
  explicit ModelInputSnapshot(std::shared_ptr<ModelInputSnapshotOwner> owner)
      : owner_(std::move(owner)) {}
  std::shared_ptr<ModelInputSnapshotOwner> owner_;
};
// Shallow borrowed execution view: construction and copying never copy native
// host vectors or multimodal payloads.
class ModelParamsView final {
 private:
  enum class Domain { LLM, VLM };
  std::shared_ptr<ModelInputSnapshotOwner> snapshot_owner_;
  void* native_owner_ = nullptr;
  Domain domain_;
  VlmVisionInput* multimodal_ = nullptr;

 public:
  explicit ModelParamsView(LlmModelParams& owner)
      : ModelParamsView(owner, owner, Domain::LLM) {}
  explicit ModelParamsView(VlmModelParams& owner)
      : ModelParamsView(owner, owner, Domain::VLM) {
    multimodal_ = &owner.multimodal;
  }
  ModelParamsView(const ModelParamsView&) = default;
  ModelParamsView(ModelParamsView&&) = default;
  ModelParamsView& operator=(const ModelParamsView&) = delete;
  ModelParamsView& operator=(ModelParamsView&&) = delete;
  ModelInputSnapshot clone() const;
  ModelInputSnapshot to(const torch::Device& device) const;
  bool has_multimodal() const { return multimodal_ != nullptr; }
  VlmVisionInput& multimodal() const {
    CHECK(multimodal_ != nullptr) << "vision input requires VLM parameters";
    return *multimodal_;
  }
  void clear_linear_attention_state() {
    embedding.linear_state_ids.clear();
    embedding.linear_state_indices = torch::Tensor();
    embedding.linear_state_read_ids.clear();
    embedding.linear_state_read_indices = torch::Tensor();
    linear_state_validity_mask.clear();
  }

  void print() const {
    LOG(INFO) << "ModelParamsView: batch_forward_type is "
              << meta.batch_forward_type.to_string() << " , num_sequences is "
              << meta.num_sequences << " , kv_max_seq_len is "
              << meta.kv_max_seq_len << " , q_max_seq_len is "
              << meta.q_max_seq_len;
    LOG(INFO) << "ModelParamsView: attention.host.kv_seq_lens is "
              << attention.host.kv_seq_lens;
    LOG(INFO) << "ModelParamsView: attention.host.q_seq_lens is "
              << attention.host.q_seq_lens;
    LOG(INFO) << "ModelParamsView: batch_forward_type is "
              << meta.batch_forward_type.to_string();
    print_tensor(
        attention.device.kv_seq_lens, "ModelParamsView: kv_seq_lens", 4);
    print_tensor(attention.device.q_seq_lens, "ModelParamsView: q_seq_lens", 4);
    print_tensor(
        attention.device.q_cu_seq_lens, "ModelParamsView: q_cu_seq_lens", 4);
    print_tensor(attention.device.new_cache_slots,
                 "ModelParamsView: new_cache_slots",
                 4);
    print_tensor(
        attention.device.block_tables, "ModelParamsView: block_tables", 4);
    LOG(INFO) << "ModelParamsView: dp_global_token_nums is "
              << parallel.dp_global_token_nums
              << ", dp_is_decode: " << parallel.dp_is_decode;
    LOG(INFO) << "ModelParamsView: is_spec_verify is " << is_spec_verify;
    print_tensor(num_accepted_tokens,
                 "ModelParamsView: num_accepted_tokens",
                 /*max_elements=*/4);
  }

  int32_t get_q_seq_len(int32_t seq_idx) const {
    CHECK_GE(seq_idx, 0) << "seq_idx out of range";
#if defined(USE_NPU)
    CHECK_LT(seq_idx, static_cast<int32_t>(attention.host.q_seq_lens.size()))
        << "seq_idx out of range";
    return attention.host.q_seq_lens[seq_idx];
#else
    CHECK_LT(seq_idx + 1,
             static_cast<int32_t>(attention.host.q_seq_lens.size()))
        << "seq_idx out of range";
    return attention.host.q_seq_lens[seq_idx + 1] -
           attention.host.q_seq_lens[seq_idx];
#endif
  }

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

  bool synchronize_draft_layer() const {
    if (parallel.layer_wise_load_synchronizer == nullptr) {
      return true;
    }
    if (!parallel.draft_load_event_index.has_value()) {
      return true;
    }
    return parallel.layer_wise_load_synchronizer->synchronize_layer(
        static_cast<int64_t>(*parallel.draft_load_event_index));
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
  EmbeddingInputView embedding;
  ParallelInput& parallel;
  BlockCopyInput& block_copy;
  ExpertInput& expert;
  GraphInputView graph;
  LinearStateValidityMask& linear_state_validity_mask;
  std::vector<torch::Tensor>& multi_block_tables;
  CacheBlockTables& cache_block_tables;
  torch::Tensor& mtp_shifted_token_ids;
  bool& is_spec_verify;
  bool& prefill_without_cache;
  torch::Tensor& num_accepted_tokens;
  MtpTopkStatePtr& mtp_topk_state;
  std::vector<int64_t>& num_accepted_tokens_host;
  std::shared_ptr<layer::AttentionMetadata>& attn_metadata;
  std::shared_ptr<PythonAttentionMetadata>& python_attention_metadata;
  bool& enable_graph;

 private:
  friend class ModelInputSnapshot;
  template <typename Owner, typename Spec>
  ModelParamsView(Owner& owner, Spec& spec, Domain domain)
      : native_owner_(&owner),
        domain_(domain),
        meta(owner.meta),
        attention(owner.attention),
        embedding(owner.embedding, spec.embedding),
        parallel(owner.parallel),
        block_copy(owner.block_copy),
        expert(owner.expert),
        graph(owner.graph, spec.graph),
        linear_state_validity_mask(owner.linear_state_validity_mask),
        multi_block_tables(spec.multi_block_tables),
        cache_block_tables(spec.cache_block_tables),
        mtp_shifted_token_ids(spec.mtp_shifted_token_ids),
        is_spec_verify(spec.is_spec_verify),
        prefill_without_cache(spec.prefill_without_cache),
        num_accepted_tokens(spec.num_accepted_tokens),
        mtp_topk_state(spec.mtp_topk_state),
        num_accepted_tokens_host(spec.num_accepted_tokens_host),
        attn_metadata(owner.attn_metadata),
        python_attention_metadata(owner.python_attention_metadata),
        enable_graph(owner.enable_graph) {}
};
template <typename NativeParams>
class NativeModelInputSnapshotOwner final : public ModelInputSnapshotOwner {
 public:
  explicit NativeModelInputSnapshotOwner(NativeParams params)
      : params_(std::move(params)) {}
  ModelParamsView view() override { return ModelParamsView(params_); }
  std::shared_ptr<ModelInputSnapshotOwner> to(
      const torch::Device& device) const override {
    return std::make_shared<NativeModelInputSnapshotOwner>(params_.to(device));
  }

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
inline ModelInputSnapshot ModelInputSnapshot::to(
    const torch::Device& device) const {
  CHECK(owner_ != nullptr);
  return ModelInputSnapshot(owner_->to(device));
}
inline ModelInputSnapshot ModelParamsView::to(
    const torch::Device& device) const {
  CHECK(native_owner_ != nullptr);
  if (snapshot_owner_ != nullptr) {
    return ModelInputSnapshot(snapshot_owner_->to(device));
  }
  switch (domain_) {
    case Domain::LLM:
      return ModelInputSnapshot(
          static_cast<LlmModelParams*>(native_owner_)->to(device));
    case Domain::VLM:
      return ModelInputSnapshot(
          static_cast<VlmModelParams*>(native_owner_)->to(device));
  }
  LOG(FATAL) << "unknown model input domain";
  return ModelInputSnapshot();
}
inline ModelInputSnapshot ModelParamsView::clone() const {
  switch (domain_) {
    case Domain::LLM:
      return ModelInputSnapshot(
          static_cast<LlmModelParams*>(native_owner_)->clone());
    case Domain::VLM:
      return ModelInputSnapshot(
          static_cast<VlmModelParams*>(native_owner_)->clone());
  }
  LOG(FATAL) << "unknown model input domain";
  return ModelInputSnapshot();
}
}  // namespace xllm

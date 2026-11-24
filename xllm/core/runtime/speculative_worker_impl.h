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

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include "common/macros.h"
#include "core/framework/model/model_input_types.h"
#include "core/framework/speculative/spec_verify.h"
#include "framework/sampling/sampling_params.h"
#include "runtime/options.h"
#include "runtime/worker_impl.h"

namespace xllm {

class JsonObjectGrammarState;

template <typename Input>
void prepare_dummy_input(Input& input, const WorkerImpl& worker);

template <typename Params>
void scale_speculative_parallel_token_counts(Params& params,
                                             int32_t num_speculative_tokens);

// Shared tail of run_validate: marks the output as a speculative decode
// verify result, attaches the row metadata, clears embeddings, and installs
// the validation sample output.
ForwardOutput finalize_verify_output(
    ForwardOutput target_output,
    SampleOutput val_output,
    const std::vector<JsonObjectGrammarState>& json_object_states);

template <typename BaseWorker, typename Input, typename PrepareInput>
class SpeculativeModelWorkerImpl final : public BaseWorker {
 public:
  template <typename... Args>
  explicit SpeculativeModelWorkerImpl(PrepareInput prepare_input,
                                      Args&&... args)
      : BaseWorker(std::forward<Args>(args)...),
        prepare_input_(std::move(prepare_input)) {}

 protected:
  using BaseWorker::prepare_model_input;

  void prepare_model_input(Input& input) override { prepare_input_(input); }

  bool owns_npu_parallel_input_prepare() const override { return false; }
  bool can_prepare_without_compute_stream_wait() const override {
    return false;
  }

 private:
  PrepareInput prepare_input_;
};

// Base class for all speculative decoding workers.
// Provides common logic: target model management, step dispatch, and
// sampling parameter updates. Subclasses implement algorithm-specific
// draft generation and validation (MTP, Eagle3, Suffix, DFlash, etc.).
template <typename TargetInput>
class SpeculativeWorkerImpl : public WorkerImpl {
 public:
  using TargetModelParams = std::remove_reference_t<
      decltype(std::declval<TargetInput&>().input_params)>;

  ~SpeculativeWorkerImpl() override;

  void drain_pending_execution() override;

 protected:
  SpeculativeWorkerImpl(const ParallelArgs& parallel_args,
                        const torch::Device& device,
                        const runtime::Options& options,
                        runtime::Options target_options,
                        WorkerType worker_type = WorkerType::LLM);

 public:
  // initialize model, cache manager. blocking call
  bool init_model(ModelContext& context) override {
    // do nothing
    return true;
  };

  bool init_model(const std::string& model_weights_path,
                  int32_t random_seed,
                  MasterStatus master_status) override;

  bool link_cluster(const std::vector<uint64_t>& cluster_ids,
                    const std::vector<std::string>& addrs,
                    const std::vector<uint16_t>& ports) override {
    return impl_->link_cluster(cluster_ids, addrs, ports);
  };

  bool unlink_cluster(const std::vector<uint64_t>& cluster_ids,
                      const std::vector<std::string>& addrs,
                      const std::vector<uint16_t>& ports) override {
    return impl_->unlink_cluster(cluster_ids, addrs, ports);
  };

  std::tuple<int64_t, int64_t> estimate_kv_cache_capacity() override {
    return impl_->estimate_kv_cache_capacity();
  };

  // allocate kv cache. blocking call
  bool allocate_kv_cache(const KVCacheShape& kv_cache_shape) override;

#if defined(USE_NPU)
  bool allocate_kv_cache_with_transfer(
      const KVCacheShape& kv_cache_shape) override;
#endif

  void get_cache_info(uint64_t& cluster_id,
                      std::string& addr,
                      uint16_t& port) override {
    impl_->get_cache_info(cluster_id, addr, port);
  };

  std::shared_ptr<const JsonObjectGrammar> ensure_json_object_grammar(
      bool reasoning_enabled) override;

  // Common step dispatch: prefill / decode / empty
  std::optional<ForwardOutput> step(const TargetInput& input) override;

  folly::SemiFuture<bool> pull_kv_blocks_async(
      const uint64_t src_cluster_id,
      const std::string& src_addr,
      const std::vector<KVTransferMapping>& mappings) override {
    return impl_->pull_kv_blocks_async(src_cluster_id, src_addr, mappings);
  };

 protected:
  bool can_prepare_without_compute_stream_wait() const override {
    return false;
  }

  virtual void apply_target_model_overrides(ModelArgs&) const;

  // Algorithm-specific virtual methods for subclasses to implement
  virtual std::optional<ForwardOutput> step_prefill(
      const TargetInput& input) = 0;
  virtual std::optional<ForwardOutput> step_decode(
      const TargetInput& inputs) = 0;
  virtual std::optional<ForwardOutput> step_empty(
      const TargetInput& inputs) = 0;

  void prepare_validate_inputs(const TargetInput& inputs,
                               TargetInput& validate_inputs,
                               bool expand_decode_metadata_per_token);

 protected:
  // Target model worker
  std::unique_ptr<WorkerImpl> impl_;
};
}  // namespace xllm

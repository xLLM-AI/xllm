/* Copyright 2025 The xLLM Authors. All Rights Reserved.

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

#pragma once

#include <folly/futures/Future.h>
#include <torch/torch.h>

#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <vector>

#include "runtime/llm_worker_impl.h"
#include "util/rec_model_utils.h"
#include "util/threadpool.h"

namespace xllm {

class RecSampler;
class RecConstrainedDecoding;
struct RecConstraintTables;

class RecWorkerImpl : public LLMWorkerImpl {
  friend class RecWorkPipeline;

 public:
  RecWorkerImpl(const ParallelArgs& parallel_args,
                const torch::Device& device,
                const runtime::Options& options);

  virtual ~RecWorkerImpl();

  bool init_model(const std::string& model_weights_path,
                  int32_t random_seed,
                  MasterStatus master_status) override;

  bool init_model(ModelContext& context) override;

  void load_model(std::unique_ptr<ModelLoader> loader) override;

  bool init_onerec_model(ModelContext& context);

  ForwardInput prepare_inputs(Batch& batch) override;

  void prepare_work_before_execute(const ForwardInput& inputs,
                                   ForwardInput& processed_inputs) override;

  std::optional<ForwardOutput> step(const ForwardInput& input) override;

  folly::SemiFuture<std::optional<ForwardOutput>> step_async(
      const ForwardInput& input);

 protected:
  std::shared_ptr<MPMCThreadPool> input_builder_thread_pool_;

 private:
  struct RecPipelineRuntime {
    RecPipelineRuntime(RecWorkerImpl& worker) : worker(worker) {}
    RecPipelineRuntime(RecPipelineRuntime&& other) : worker(other.worker) {
      stream = std::move(other.stream);
      model = std::move(other.model);
      executor = std::move(other.executor);
      context = std::move(other.context);
      eplb_executor = std::move(other.eplb_executor);
      expert_load_data = std::move(other.expert_load_data);
    }
    std::unique_ptr<Stream> stream;
    std::unique_ptr<CausalLM> model;
    std::unique_ptr<Executor> executor;
    std::unique_ptr<ModelContext> context;
    std::unique_ptr<EplbExecutor> eplb_executor;
    torch::Tensor expert_load_data;

    RecWorkerImpl& worker;  // not owned
  };

  class RecWorkPipeline {
   public:
    RecWorkPipeline(RecPipelineRuntime& runtime)
        : runtime_(std::move(runtime)) {}
    virtual ~RecWorkPipeline() = default;

    virtual ForwardInput prepare_inputs(Batch& batch);

    virtual void prepare_work_before_execute(const ForwardInput& inputs,
                                             ForwardInput& processed_inputs);

    virtual std::optional<ForwardOutput> step(const ForwardInput& input);

    RecPipelineRuntime& runtime() { return runtime_; }

   protected:
    RecPipelineRuntime runtime_;
  };

  class LlmRecWorkPipeline final : public RecWorkPipeline {
   public:
    explicit LlmRecWorkPipeline(RecPipelineRuntime& runtime)
        : RecWorkPipeline(runtime) {}

    void prepare_work_before_execute(const ForwardInput& inputs,
                                     ForwardInput& processed_inputs) override;
  };

  class OneRecWorkPipeline : public RecWorkPipeline {
   public:
    explicit OneRecWorkPipeline(
        RecPipelineRuntime& runtime,
        RecPipelineType pipeline_type = RecPipelineType::kOneRecDefault);

    ForwardInput prepare_inputs(Batch& batch) override;

    void prepare_work_before_execute(const ForwardInput& inputs,
                                     ForwardInput& processed_inputs) override;

    std::optional<ForwardOutput> step(const ForwardInput& input) override;

   private:
    folly::SemiFuture<torch::Tensor> prepare_filter_mask_async(
        const std::vector<std::vector<int32_t>>& generated_tokens);

    std::unique_ptr<RecSampler> rec_sampler_;
    std::unique_ptr<RecConstrainedDecoding> constrained_decoding_;
    std::unique_ptr<ThreadPool> filter_mask_threadpool_;
  };

  class OneRecXAttentionWorkPipeline final : public RecWorkPipeline {
   public:
    explicit OneRecXAttentionWorkPipeline(RecPipelineRuntime& runtime);

    ForwardInput prepare_inputs(Batch& batch) override;

    void prepare_work_before_execute(const ForwardInput& inputs,
                                     ForwardInput& processed_inputs) override;

    std::optional<ForwardOutput> step(const ForwardInput& input) override;

   private:
    struct RecConstraintDeviceTensors {
      torch::Tensor first_token_ids;
      torch::Tensor prefix1_offsets;
      torch::Tensor prefix1_values;
      torch::Tensor prefix1_pair_keys;
      torch::Tensor prefix2_value_offsets;
      torch::Tensor prefix2_values;
      int64_t max_prefix1_degree = 0;
      int64_t max_prefix2_degree = 0;
      bool initialized = false;
    };

    folly::SemiFuture<torch::Tensor> prepare_filter_mask_async(
        const std::vector<std::vector<int32_t>>& generated_tokens);

    void initialize_constraint_device_tensors(
        const RecConstraintTables& tables);

    bool can_use_device_constraints(const SamplingParameters& sampling_params,
                                    int32_t current_step,
                                    int32_t beam_width) const;

    SampleOutput sample_with_device_constraints(
        torch::Tensor& logits,
        const SamplingParameters& sampling_params,
        const torch::Tensor& sequence_group,
        int32_t current_step) const;

    void allocate_unshared_kv_caches();

    void prepare_unshared_kv_caches_for_input(
        const ForwardInput& inputs,
        OneRecXAttentionParams& onerec_params);

    void execute_cache_select(const torch::Tensor& out_token_index,
                              const torch::Tensor& out_beam_count_prefix_sums,
                              OneRecXAttentionParams& onerec_params,
                              int32_t round,
                              int32_t batch_size,
                              int32_t beam_width,
                              int32_t num_layers);

    std::unique_ptr<RecSampler> rec_sampler_;
    std::unique_ptr<RecConstrainedDecoding> constrained_decoding_;
    std::unique_ptr<ThreadPool> filter_mask_threadpool_;
    RecConstraintDeviceTensors constraint_device_tensors_;
    std::vector<torch::Tensor> cached_unshared_k_caches_;
    std::vector<torch::Tensor> cached_unshared_v_caches_;
    std::vector<torch::Tensor> cached_decode_position_tensors_;
    std::vector<int32_t> cached_decode_positions_key_;
    torch::Tensor cached_decode_selected_token_idxes_;
    int64_t cached_decode_selected_token_count_ = 0;
    int32_t cached_decode_positions_batch_size_ = 0;
    int32_t cached_decode_positions_beam_width_ = 0;
    int32_t cached_decode_positions_total_rounds_ = 0;
    int32_t max_seqs_per_batch_ = 0;
    int32_t beam_width_ = 1;
    int32_t max_decode_step_ = 0;
  };

  class LlmRecWithMmDataWorkPipeline final : public RecWorkPipeline {
   public:
    explicit LlmRecWithMmDataWorkPipeline(RecPipelineRuntime& runtime)
        : RecWorkPipeline(runtime) {}

    void prepare_work_before_execute(const ForwardInput& inputs,
                                     ForwardInput& processed_inputs) override;
  };

  class LlmRecMultiRoundPipeline final : public RecWorkPipeline {
   public:
    explicit LlmRecMultiRoundPipeline(RecPipelineRuntime& runtime);

    ForwardInput prepare_inputs(Batch& batch) override;

    void prepare_work_before_execute(const ForwardInput& inputs,
                                     ForwardInput& processed_inputs) override;

    std::optional<ForwardOutput> step(const ForwardInput& input) override;

   private:
    // Beam search related tensors
    struct BeamSearchTensors {
      torch::Tensor sequence_group;   // [batch_size, beam_width, total_rounds]
      torch::Tensor acc_logprob;      // [num_seq, 1]
      torch::Tensor out_log_probs;    // [num_seq, 1]
      torch::Tensor out_token_ids;    // [num_seq, 1]
      torch::Tensor out_token_index;  // [num_seq, 1]
      torch::Tensor out_beam_count_prefix_sums;  // [num_seq, 1]
      torch::Tensor out_seqgroup;  // [batch_size, beam_width, total_rounds]
      torch::Tensor
          token_step_logprobs;  // [batch_size, beam_width, total_rounds]
    };

    // Prepare beam search tensors
    BeamSearchTensors prepare_beam_search_tensors(int32_t batch_size,
                                                  int32_t beam_width,
                                                  int32_t total_rounds,
                                                  const torch::Device& device);

    // Execute beam search kernel
    void execute_beam_search(const torch::Tensor& top_tokens,
                             const torch::Tensor& top_logprobs,
                             BeamSearchTensors& beam_tensors,
                             int32_t round,
                             int32_t batch_size,
                             int32_t requested_result_width,
                             int32_t total_rounds);

    void execute_final_beam_search(const torch::Tensor& top_tokens,
                                   const torch::Tensor& top_logprobs,
                                   BeamSearchTensors& beam_tensors,
                                   int32_t round,
                                   int32_t batch_size,
                                   int32_t beam_width,
                                   int32_t requested_result_width);

    // Execute cache select kernel
    void execute_cache_select(const BeamSearchTensors& beam_tensors,
                              ForwardInput& input,
                              int32_t round,
                              int32_t beam_width,
                              int32_t layer_num);

    // Build final output from beam search results
    void build_final_output(const torch::Tensor& logits,
                            const SampleOutput& sample_output,
                            const SamplingParameters& sampling_params,
                            const BeamSearchTensors& beam_tensors,
                            ForwardOutput& output);

    // Structure to hold async computation results for next round input
    struct NextRoundInputResults {
#if defined(USE_NPU)
// TODO: implement NextRoundInputResults for NPU
#elif defined(USE_CUDA)
      torch::Tensor paged_kv_indices;
      torch::Tensor paged_kv_indptr;
      torch::Tensor paged_kv_last_page_len;
#endif
    };

    // Compute next round input asynchronously (can overlap with GPU execution)
    folly::SemiFuture<NextRoundInputResults> compute_next_round_input_async(
        const torch::Tensor& kv_seq_lens,
        int32_t current_step,
        int32_t batch_size,
        int32_t beam_width,
        int32_t max_decode_step);

    // Apply async result to prepare decode input for current round
    void prepare_input_for_current_round(ForwardInput& input,
                                         const NextRoundInputResults& results,
                                         int32_t round,
                                         const torch::Tensor& top_tokens,
                                         const BeamSearchTensors& beam_tensors);

    void prepare_round_input_for_npu(ForwardInput& input,
                                     int32_t round,
                                     const torch::Tensor& top_tokens,
                                     const BeamSearchTensors& beam_tensors);

    void prepare_two_stage_round_input(ForwardInput& input,
                                       int32_t round,
                                       const torch::Tensor& top_tokens,
                                       const BeamSearchTensors& beam_tensors);

    // Consume async result for current round and schedule async computation for
    // next round.
    void prepare_round_input_and_schedule_next(
        ForwardInput& input,
        int32_t round,
        int32_t total_rounds,
        int32_t batch_size,
        int32_t beam_width,
        int32_t max_decode_step,
        const torch::Tensor& top_tokens,
        const BeamSearchTensors& beam_tensors,
        std::optional<folly::SemiFuture<NextRoundInputResults>>&
            next_round_async_result);

    void allocate_kv_caches_related();
    void prepare_kv_caches_related_for_input(const ForwardInput& inputs,
                                             ForwardInput& processed_inputs);

    struct FullKvCacheOffsets {
      explicit FullKvCacheOffsets(
          LlmRecMultiRoundPipeline* multi_round_pipeline);
#if defined(USE_NPU)
// TODO: implement FullKvCacheOffsets for NPU
#elif defined(USE_CUDA)
      torch::Tensor full_kv_offsets;
      torch::Tensor full_kv_mask;
      torch::Tensor full_kv_indices;
      torch::Tensor unshared_offsets;
      torch::Tensor max_decode_step_ids;
#endif
    };
    std::unique_ptr<FullKvCacheOffsets> full_kv_cache_offsets_;

    std::vector<torch::Tensor> cached_full_k_caches_;
    std::vector<torch::Tensor> cached_full_v_caches_;
    torch::Tensor cached_naive_block_table_;
    torch::Tensor cached_current_round_tensor_;
    torch::Tensor cached_two_stage_shared_lse_;
    torch::Tensor cached_two_stage_shared_o_;
    torch::Tensor cached_two_stage_unshared_lse_;
    torch::Tensor cached_two_stage_unshared_o_;
    torch::Tensor cached_two_stage_q_cu_seq_lens_shared_;
    torch::Tensor cached_two_stage_qo_indptr_expanded_;
    torch::Tensor cached_two_stage_paged_kv_indptr_expanded_;
    torch::Tensor cached_two_stage_paged_kv_indices_expanded_;
    torch::Tensor cached_two_stage_paged_kv_last_page_len_expanded_;
    torch::Tensor cached_beam_width_tensor_;

    std::unique_ptr<RecSampler> rec_sampler_;

    // for async scheduler
    ThreadPool threadpool_;

    int32_t max_seqs_per_batch_;
    int32_t max_tokens_per_batch_;
    int32_t max_token_per_req_;
    int32_t beam_width_;
  };

  // Factory method to create pipeline (can access private classes)
  static std::unique_ptr<RecWorkPipeline> create_pipeline(
      RecPipelineType type,
      RecPipelineRuntime& runtime);

  torch::Tensor merge_embeddings_by_indices(
      const torch::Tensor& input_tokens_embedding,
      const torch::Tensor& input_embedding,
      const std::vector<int64_t>& input_indices);

  void prepare_multi_modal_data(ForwardInput& processed_inputs);

  void initialize_xattention_workspace();

  std::vector<std::unique_ptr<RecWorkPipeline>> work_pipelines_;

  RecModelKind rec_model_kind_ = RecModelKind::kNone;
  RecPipelineType pipeline_type_ = RecPipelineType::kLlmRecDefault;

  std::unique_ptr<ThreadPool> step_threadpool_;

  std::mutex onerec_xattention_prepare_mutex_;
  std::mutex onerec_xattention_model_forward_mutex_;

  moodycamel::BlockingConcurrentQueue<size_t> index_queue_;
};

}  // namespace xllm

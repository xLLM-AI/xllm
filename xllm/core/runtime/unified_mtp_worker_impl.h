/* Copyright 2026 The xLLM Authors.

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
#include "core/runtime/mtp_worker_impl.h"

namespace xllm {
class UnifiedMtpExecutor;
template <typename TargetInput>
bool supports_unified_mtp_request(const TargetInput& input);
// Unified MTP worker. Shared model/cache lifetime is supplied by
// MTPWorkerImpl<TargetInput>; unsupported requests use its legacy decode
// path.
template <typename TargetInput>
class UnifiedMtpWorkerImpl : public MTPWorkerImpl<TargetInput> {
 public:
  UnifiedMtpWorkerImpl(const ParallelArgs& parallel_args,
                       const torch::Device& device,
                       const runtime::Options& options,
                       WorkerType worker_type);
  ~UnifiedMtpWorkerImpl() override;
  bool init_model(const std::string& model_weights_path,
                  int32_t random_seed,
                  MasterStatus master_status) override;
  bool uses_worker_task_pipeline() const override { return true; }
  ::xllm::Status create_task_pipeline(
      std::unique_ptr<TaskExecutionPipeline>& output) override;
  void prepare_task_pipeline_input(const TargetInput& input,
                                   TargetInput& prepared) override;
  std::optional<ForwardOutput> execute_task_pipeline(
      const TargetInput& prepared) override;
  std::optional<ForwardOutput> step(const TargetInput& input) override;

 protected:
  using MTPWorkerImpl<TargetInput>::compute_stream_;
  using MTPWorkerImpl<TargetInput>::context_;
  using MTPWorkerImpl<TargetInput>::device_;
  using MTPWorkerImpl<TargetInput>::draft_impl_;
  using MTPWorkerImpl<TargetInput>::driver_;
  using MTPWorkerImpl<TargetInput>::dp_driver_;
  using MTPWorkerImpl<TargetInput>::embedding_cache_;
  using MTPWorkerImpl<TargetInput>::impl_;
  using MTPWorkerImpl<TargetInput>::is_probabilistic_draft_;
  using MTPWorkerImpl<TargetInput>::options_;
  using MTPWorkerImpl<TargetInput>::parallel_args_;
  using MTPWorkerImpl<TargetInput>::prepare_stream_;
  using MTPWorkerImpl<TargetInput>::threadpool_;
  using MTPWorkerImpl<TargetInput>::combined_draft_execution_path_;
  using MTPWorkerImpl<TargetInput>::drain_pending_execution;
  using MTPWorkerImpl<TargetInput>::should_run_speculative_decode;
  using MTPWorkerImpl<TargetInput>::enable_schedule_overlap;
  using MTPWorkerImpl<TargetInput>::prepare_input_on_stream;
  using MTPWorkerImpl<TargetInput>::logical_block_size;
  using MTPWorkerImpl<TargetInput>::uses_step_major_validate_layout;
  using MTPWorkerImpl<TargetInput>::acquire_accepted_tokens_host_buffer;
  using MTPWorkerImpl<TargetInput>::stage_target_context_write;
  using MTPWorkerImpl<TargetInput>::snapshot_pending_target_tokens;
  using MTPWorkerImpl<TargetInput>::latest_pending_target_context;
  using MTPWorkerImpl<TargetInput>::retire_legacy_prelaunch;
  using MTPWorkerImpl<TargetInput>::flush_pending_target_context;
  using MTPWorkerImpl<TargetInput>::prepare_draft_input;
  using MTPWorkerImpl<TargetInput>::pending_target_context_matches;
  using MTPWorkerImpl<TargetInput>::update_decode_step_input;
  using MTPWorkerImpl<TargetInput>::prepare_draft_extend_inputs;
  using MTPWorkerImpl<TargetInput>::prepare_draft_prefix;
  using MTPWorkerImpl<TargetInput>::retain_draft_prefix;
  using MTPWorkerImpl<TargetInput>::uses_verifier_conditioned_draft_prefix;

  std::optional<ForwardOutput> step_decode(const TargetInput& input) override;
  bool supports_unified_python_mtp_graph(const TargetInput& input) const;
  void clear_unified_device_state();
  UnifiedMtpExecutor& unified_executor();

 private:
  bool supports_unified_configuration() const;
  bool unified_graph_capable_ = false;
  // Launch-thread-only JSON overlap context. Task results remain owned by
  // the shared pipeline; only a completed token copy survives slot reuse.
  ForwardOutput task_json_output_;
  std::vector<std::string> task_json_sample_sequence_ids_;
  std::optional<ForwardOutput> step_unified(const TargetInput& input);
  std::optional<ForwardOutput> run_unified_python_mtp_graph(
      const TargetInput& input,
      const LlmForwardInput& current_draft_input,
      int32_t num_speculative_tokens);
  std::unique_ptr<UnifiedMtpExecutor> unified_executor_;
};
}  // namespace xllm

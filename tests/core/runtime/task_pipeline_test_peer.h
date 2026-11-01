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
#include "core/runtime/task_execution_pipeline.h"
#include "core/runtime/task_execution_pipeline_speculative.h"
namespace xllm {
// Exercise fixed views and retirement invariants without exposing a second
// public runtime API or adding production helper objects.
struct TaskPipelineTestPeer {
  static constexpr auto plan_parallel = TaskExecutionPipeline::plan_parallel;
  static constexpr auto context_create = TaskExecutionPipeline::context_create;
  static constexpr auto context_prepare =
      TaskExecutionPipeline::context_prepare;
  static constexpr auto context_prepare_prefill =
      TaskExecutionPipeline::context_prepare_prefill;
  static constexpr auto context_prepare_decode =
      TaskExecutionPipeline::context_prepare_decode;
  static constexpr auto context_gather = TaskExecutionPipeline::context_gather;
  static constexpr auto speculative_slot =
      [](const TaskExecutionPipeline& pipeline,
         uint32_t slot_id) -> const auto& {
    return *pipeline.slots_[slot_id]->speculative;
  };
  static constexpr auto input_ready = [](const TaskExecutionPipeline& pipeline,
                                         uint32_t slot_id) {
    return pipeline.slots_[slot_id]->input_ready;
  };
  static constexpr auto context_publish =
      TaskExecutionPipeline::context_publish;
  static constexpr auto context_advance =
      TaskExecutionPipeline::context_advance;
  static constexpr auto context_release =
      TaskExecutionPipeline::context_release;
  static constexpr auto context_device_bytes =
      TaskExecutionPipeline::context_device_bytes;
  static constexpr auto create_mtp_input = SlotBuffer::create_mtp_input;
  static constexpr auto model_input_view = SlotBuffer::model_input_view;
  static constexpr auto input_scratch = [](SlotBuffer& input) -> auto& {
    return *input.input_scratch_;
  };
  static constexpr auto adaptive_costs =
      [](const TaskExecutionPipeline& pipeline) -> const auto& {
    return *pipeline.speculative_capacity_->adaptive_costs;
  };
  static constexpr auto mtp_context =
      [](const TaskExecutionPipeline& pipeline) -> const auto& {
    return *pipeline.context_;
  };
  static constexpr auto adaptive_confidence =
      [](const TaskExecutionPipeline& pipeline, uint32_t embedding_id) {
        return pipeline.adaptive_verify_->histories.at(embedding_id)
            .versions[0]
            .confidence;
      };
  static constexpr auto erase_adaptive_unified_binding =
      [](TaskExecutionPipeline& pipeline, uint32_t rows, uint32_t graph_rows) {
        const uint64_t key = (static_cast<uint64_t>(rows) << 32) | graph_rows;
        uint32_t erased = 0;
        for (auto& task_slot : pipeline.slots_) {
          erased += static_cast<uint32_t>(
              task_slot->speculative->verify->unified_graph_bindings.erase(
                  key));
        }
        return erased;
      };
  static constexpr auto plan_mtp_decode = SlotBuffer::plan_mtp_decode;
  static constexpr auto set_mtp_validate_width =
      SlotBuffer::set_mtp_validate_width;
  static constexpr auto prepare_mtp_decode = SlotBuffer::prepare_mtp_decode;
  static constexpr auto prepare_planned_mtp_decode =
      SlotBuffer::prepare_planned_mtp_decode;
  static constexpr auto prepare_dspark_validate =
      SlotBuffer::prepare_dspark_validate;
  static constexpr auto patch_mtp_decode = SlotBuffer::patch_mtp_decode;
  static constexpr auto prepare_mtp_prefill = SlotBuffer::prepare_mtp_prefill;
  static constexpr auto patch_mtp_prefill = SlotBuffer::patch_mtp_prefill;
  static constexpr auto create_mtp_context =
      TaskExecutionPipeline::create_mtp_context;
  static constexpr auto mtp_context_bytes =
      TaskExecutionPipeline::mtp_context_bytes;
};
}  // namespace xllm

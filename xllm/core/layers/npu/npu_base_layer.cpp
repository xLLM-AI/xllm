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

#include "npu_base_layer.h"

#ifdef TORCH_HIGHER_THAN_PTA6
#include <torch_npu/csrc/core/npu/NPUFormat.h>
#include <torch_npu/csrc/framework/OpCommand.h>
#else
#include <torch_npu/csrc/aten/NPUNativeFunctions.h>
#include <torch_npu/csrc/framework/utils/OpPreparation.h>
#endif
#include "core/common/global_flags.h"
#include "core/framework/config/execution_config.h"

namespace xllm {
namespace layer {

BaseLayer::BaseLayer(const ModelContext& context)
    : device_(context.get_tensor_options().device()),
      name_(""),
      parallel_args_(context.get_parallel_args()) {
  auto quant_args = context.get_quant_args();
  if (!quant_args.quantize_type().empty()) {
    quantize_type_ = quant_args.quantize_type();
  }

  if (!quant_args.torch_dtype().empty()) {
    torch_dtype_ = quant_args.torch_dtype();
  }

  dp_size_ = parallel_args_.dp_size();
  cp_size_ = parallel_args_.cp_size();
  dp_local_tp_size_ = parallel_args_.world_size() / (dp_size_ * cp_size_);
  dp_rank_ = parallel_args_.rank() / (dp_local_tp_size_ * cp_size_);
  CHECK_EQ(parallel_args_.world_size(),
           dp_size_ * dp_local_tp_size_ * cp_size_);
  dp_local_tp_rank_ = parallel_args_.rank() % dp_local_tp_size_;

  run_task_func_ = [this](const std::string& task_name,
                          std::function<int()> task) {
    this->run_task(task_name, task);
  };

  context_ = const_cast<atb::Context*>(context.get_atb_context());
  work_space_ = context.get_atb_workspace();
}

atb::Status BaseLayer::execute_node(atb_speed::Model::Node& node,
                                    int node_id,
                                    aclrtEvent* event,
                                    std::atomic<bool>* event_flag) {
  // TODO: Stream management needs to be refactored
  // for better separation of concerns Current issues:
  // 1. ACLGraph capture requires execution on a non-default stream, so we
  // temporarily set the current stream
  // 2. After ACLGraph capture ends, the stream will be modified back to the
  // default stream
  // 3. In non-ACL graph capture mode, the context stream should be set to the
  // default stream
  // 4. The actual requirement is to separate decode node context from prefill
  // node context
  //
  // Note: The commented code below will cause runtime errors because:
  // - aclmdlRICaptureGetInfo() may fail when called at inappropriate times
  // - The capture status check logic is not robust enough for all scenarios
  // - Stream management conflicts: ATB context stream must be consistent with
  // libtorch_npu current stream.
  //   However, libtorch_npu current stream is set to default stream after
  //   capture ends, causing inconsistency between ATB context and the actual
  //   execution stream
  if (::xllm::ExecutionConfig::get_instance().enable_graph()) {
    void* stream = c10_npu::getCurrentNPUStream(device_.index()).stream();
    context_->SetExecuteStream(stream);
  }
  // if (::xllm::ExecutionConfig::get_instance().enable_graph() &&
  // !graph_captured_) {
  //   void* stream = c10_npu::getCurrentNPUStream(device_.index()).stream();
  //   aclmdlRICaptureStatus status;
  //   aclmdlRI modelRI;
  //   auto error = aclmdlRICaptureGetInfo(stream, &status, &modelRI);
  //   if (error != ACL_SUCCESS) {
  //     LOG(ERROR) << "aclmdlRICaptureGetInfo failed, acl error code: " <<
  //     error;
  //   }
  //   if (status == ACL_MODEL_RI_CAPTURE_STATUS_ACTIVE) {
  //     context_->SetExecuteStream(stream);
  //     graph_captured_ = true;
  //   }
  // }
  atb::Status st =
      node.operation->Setup(node.variantPack, node.workspaceSize, context_);
  if (st != 0) {
    LOG(ERROR) << " setup layer node fail, not call execute";
    return st;
  }

  if (node.workspaceSize > 0) {
    node.workspace = work_space_->get_workspace_buffer(node.workspaceSize);
  }

  run_task_func_(name_ + std::to_string(node_id), [=, this]() {
    return execute_plan(
        node, name_ + std::to_string(node_id), event, event_flag);
  });

  return st;
}

atb::Status BaseLayer::execute_plan(const atb_speed::Model::Node& node,
                                    const std::string& op_name,
                                    aclrtEvent* event,
                                    std::atomic<bool>* event_flag) {
  atb::Status st = node.operation->Execute(
      node.variantPack, (uint8_t*)node.workspace, node.workspaceSize, context_);
  LOG_IF(ERROR, st != 0) << name_ << " execute plan fail, error code: " << st;
  if (st == 0 && event != nullptr) {
    aclrtStream stream = context_->GetExecuteStream();

    aclrtEvent* aclrt_event = reinterpret_cast<aclrtEvent*>(event);

    auto ret = aclrtRecordEvent(*aclrt_event, stream);
    if (ret != ACL_SUCCESS) {
      LOG(ERROR) << "Record event failed.";
      return st;
    }

    event_flag->store(true, std::memory_order_release);
  }

  return st;
}

void BaseLayer::run_task(std::string taskName,
                         std::function<int()> task) const {
  at_npu::native::OpCommand cmd;
  cmd.Name(taskName);
  cmd.SetCustomHandler(task);
  cmd.Run();
}

}  // namespace layer
}  // namespace xllm

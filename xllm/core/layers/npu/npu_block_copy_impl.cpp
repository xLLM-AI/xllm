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

#include "npu_block_copy_impl.h"

#include <glog/logging.h>

namespace xllm {
namespace layer {

NpuBlockCopyImpl::NpuBlockCopyImpl(const ModelContext& context)
    : BaseLayer(context) {
  auto options = context.get_tensor_options();
  dtype_ = c10::typeMetaToScalarType(options.dtype());
}

void NpuBlockCopyImpl::merge_loaded_weights() { init_layer(); }

int64_t NpuBlockCopyImpl::init_layer() {
  BaseLayer::name_ = "block_copy_layer";
  model_name_ = "llm";
  CHECK_OPERATION_STATUS_RETURN(init_node(node_, param_));
  return atb::NO_ERROR;
}

int64_t NpuBlockCopyImpl::init_node(atb_speed::Model::Node& node,
                                    atb::infer::BlockCopyParam& param) {
  atb::Operation* operation = nullptr;
  atb::Status atbStatus = atb::CreateOperation(param, &operation);
  if (atbStatus != atb::NO_ERROR) {
    return atbStatus;
  }

  node.operation.reset(operation);
  if (node.operation == nullptr) {
    LOG(ERROR) << "node.operation is null";
    return -1;
  }
  if (node.operation->GetInputNum() < 1) {
    LOG(ERROR) << "Can not resize number which is smaller than 1";
    return -1;
  }
  node.inTensors.resize(node.operation->GetInputNum());
  node.outTensors.resize(node.operation->GetOutputNum());

  node.variantPack.inTensors.reserve(node.inTensors.size());
  node.variantPack.inTensors.resize(node.inTensors.size());
  node.variantPack.outTensors.reserve(node.outTensors.size());
  node.variantPack.outTensors.resize(node.outTensors.size());

  return atb::NO_ERROR;
}

torch::Tensor NpuBlockCopyImpl::forward(const torch::Tensor& key_cache,
                                        const torch::Tensor& value_cache,
                                        const torch::Tensor& src_block_ids,
                                        const torch::Tensor& dst_block_ids,
                                        const torch::Tensor& cum_sum,
                                        int nodeId) {
  atb::Status st;
  build_node_variant_pack(
      node_, key_cache, value_cache, src_block_ids, dst_block_ids, cum_sum);
  st = execute_node(node_, nodeId);
  LOG_IF(FATAL, st != 0) << model_name_
                         << "infer shape fail, error code: " << st;
  return key_cache;
}

void NpuBlockCopyImpl::build_node_variant_pack(
    atb_speed::Model::Node& node,
    const torch::Tensor& key_cache,
    const torch::Tensor& value_cache,
    const torch::Tensor& src_block_ids,
    const torch::Tensor& dst_block_ids,
    const torch::Tensor& cum_sum) {
  internal_key_tensors_ = atb_speed::Utils::AtTensor2Tensor(key_cache);
  internal_value_tensors_ = atb_speed::Utils::AtTensor2Tensor(value_cache);
  internal_src_block_ids_tensors_ =
      atb_speed::Utils::AtTensor2Tensor(src_block_ids);
  internal_dst_block_ids_tensors_ =
      atb_speed::Utils::AtTensor2Tensor(dst_block_ids);
  internal_cum_sum_tensors_ = atb_speed::Utils::AtTensor2Tensor(cum_sum);

  node.variantPack.inTensors.at(0) = internal_key_tensors_;
  node.variantPack.inTensors.at(1) = internal_value_tensors_;
  node.variantPack.inTensors.at(2) = internal_src_block_ids_tensors_;
  node.variantPack.inTensors.at(3) = internal_dst_block_ids_tensors_;
  node.variantPack.inTensors.at(4) = internal_cum_sum_tensors_;
}

}  // namespace layer
}  // namespace xllm

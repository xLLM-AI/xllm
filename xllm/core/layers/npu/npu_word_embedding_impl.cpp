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

#include "npu_word_embedding_impl.h"

#include <glog/logging.h>

#include "core/framework/config/load_config.h"
#include "core/framework/config/parallel_config.h"
namespace xllm {
namespace layer {

void NpuWordEmbeddingImpl::param_from_args(
    atb_speed::common::WordEmbeddingParam& param,
    const xllm::ModelArgs& args,
    const xllm::ParallelArgs& parallel_args) {
  param.unpadInputs = true;

  if (parallel_args.world_size() > 1) {
    if (parallel_args.mapping_data().empty()) {
      const bool use_local_tp = (dp_size_ > 1) || (cp_size_ > 1);
      if (use_local_tp) {
        param.tensorParallelInfo.rank = dp_local_tp_rank_;
        param.tensorParallelInfo.worldSize = dp_local_tp_size_;
      } else {
        param.tensorParallelInfo.rank = parallel_args.rank();
        param.tensorParallelInfo.worldSize = parallel_args.world_size();
      }
      const int32_t tp_group_id =
          use_local_tp ? (parallel_args.rank() / dp_local_tp_size_) : 0;
      param.tensorParallelInfo.commDomain = std::to_string(tp_group_id);
      param.tensorParallelInfo.backend =
          ::xllm::ParallelConfig::get_instance().communication_backend();
    } else {
      atb_speed::common::ParallelInfo parallelInfo =
          parallel_args.mapping().Get(atb_speed::base::ATTN_TP);
      param.tensorParallelInfo.rank = parallelInfo.rank;
      param.tensorParallelInfo.worldSize = parallelInfo.rankIds.size();
      param.tensorParallelInfo.backend =
          ::xllm::ParallelConfig::get_instance().communication_backend();
      parallelInfo.InitCommDomain(param.tensorParallelInfo.hcommInfo,
                                  param.tensorParallelInfo.commDomain);
    }
  }
}

NpuWordEmbeddingImpl::NpuWordEmbeddingImpl(const ModelContext& context)
    : BaseLayer(context) {
  auto model_args = context.get_model_args();
  auto parallel_args = context.get_parallel_args();
  auto options = context.get_tensor_options();

  param_from_args(embedding_param_, model_args, parallel_args);
  atb_weight_tensors_.resize(1);
  atOutTensors_.resize(1);
  dtype_ = c10::typeMetaToScalarType(options.dtype());

  loader_ = std::make_unique<WordEmbeddingLoader>(
      1,
      context,
      ::xllm::LoadConfig::get_instance().enable_manual_loader()
          ? LoadMode::kManual
          : LoadMode::kEager);
}

int64_t NpuWordEmbeddingImpl::init_layer() {
  BaseLayer::name_ = "word_embedding_layer";
  modelName_ = "llm";
  CHECK_OPERATION_STATUS_RETURN(init_node(embedding_node_, embedding_param_));
  return atb::NO_ERROR;
}

int64_t NpuWordEmbeddingImpl::init_node(
    atb_speed::Model::Node& node,
    atb_speed::common::WordEmbeddingParam& param) {
  atb::Operation* operation = nullptr;
  atb_speed::common::WordEmbedding(param, &operation);
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
  // node.outTensors.resize(1);

  node.inTensors.at(0) = &atb_weight_tensors_[0];

  node.variantPack.inTensors.reserve(node.inTensors.size());
  node.variantPack.inTensors.resize(node.inTensors.size());
  node.variantPack.outTensors.reserve(1);
  node.variantPack.outTensors.resize(1);

  return atb::NO_ERROR;
}

torch::Tensor NpuWordEmbeddingImpl::forward(const torch::Tensor& x,
                                            int nodeId) {
  atb::Status st;
  // std::cout<<"x:"<<x<<std::endl;
  build_node_variant_pack(embedding_node_, x);
  st = execute_node(embedding_node_, nodeId);
  LOG_IF(FATAL, st != 0) << modelName_
                         << "infer shape fail, error code: " << st;
  return atOutTensors_.at(0);
}

void NpuWordEmbeddingImpl::build_node_variant_pack(atb_speed::Model::Node& node,
                                                   const torch::Tensor& x) {
  internalTensors = atb_speed::Utils::AtTensor2Tensor(x);
  // node.outTensors[0] = &internalTensors;

  atb::SVector<atb::TensorDesc> inTensorDescs;
  inTensorDescs.reserve(node.variantPack.inTensors.size());
  inTensorDescs.resize(node.variantPack.inTensors.size());

  atb::SVector<atb::TensorDesc> outTensorDescs;
  outTensorDescs.reserve(node.operation->GetOutputNum());
  outTensorDescs.resize(node.operation->GetOutputNum());

  node.variantPack.inTensors.at(0) = *node.inTensors.at(0);
  inTensorDescs.at(0) = node.inTensors.at(0)->desc;

  node.variantPack.inTensors.at(1) = internalTensors;
  inTensorDescs.at(1) = internalTensors.desc;

  atb::Status st = node.operation->InferShape(inTensorDescs, outTensorDescs);

  at::Tensor newTensor =
      atb_speed::Utils::CreateAtTensorFromTensorDesc(outTensorDescs.at(0));

  atOutTensors_.at(0) = newTensor;

  node.variantPack.outTensors.at(0) =
      atb_speed::Utils::AtTensor2Tensor(atOutTensors_.at(0));
}

}  // namespace layer
}  // namespace xllm

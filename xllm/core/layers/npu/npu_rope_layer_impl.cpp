/* Copyright 2026 The xLLM Authors.

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

#include "npu_rope_layer_impl.h"

#include <glog/logging.h>

#include "atb_speed/utils/tensor_util.h"

namespace xllm {
namespace layer {

NpuRopeLayerImpl::NpuRopeLayerImpl(const ModelContext& context)
    : BaseLayer(context) {
  const auto& args = context.get_model_args();
  head_dim_ = args.mm_head_dim();
  if (head_dim_ <= 0) {
    head_dim_ = args.mm_hidden_size() / args.mm_num_attention_heads();
  }
  CHECK_GT(head_dim_, 0);
  output_tensors_.resize(2);
  init_layer();
}

int64_t NpuRopeLayerImpl::init_layer() {
  name_ = "rope_layer";
  return init_node(rope_node_);
}

int64_t NpuRopeLayerImpl::init_node(atb_speed::Model::Node& node) {
  atb::infer::RopeParam rope_param;
  rope_param.rotaryCoeff = 2;

  atb::Operation* operation = nullptr;
  const atb::Status status = atb::CreateOperation(rope_param, &operation);
  if (status != atb::NO_ERROR || operation == nullptr) {
    LOG(ERROR) << "Failed to create RopeOperation, status=" << status;
    return status == atb::NO_ERROR ? -1 : status;
  }

  node.operation.reset(operation);
  CHECK_EQ(node.operation->GetInputNum(), 5);
  CHECK_EQ(node.operation->GetOutputNum(), 2);
  node.inTensors.resize(node.operation->GetInputNum());
  node.outTensors.resize(node.operation->GetOutputNum());
  node.variantPack.inTensors.resize(node.operation->GetInputNum());
  node.variantPack.outTensors.resize(node.operation->GetOutputNum());
  return atb::NO_ERROR;
}

std::tuple<torch::Tensor, torch::Tensor> NpuRopeLayerImpl::forward(
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& cos,
    const torch::Tensor& sin,
    const torch::Tensor& seq_len,
    std::vector<int32_t>& seq_len_host,
    int32_t node_id) {
  build_node_variant_pack(
      rope_node_, query, key, cos, sin, seq_len, seq_len_host);
  const atb::Status status = execute_node(rope_node_, node_id);
  LOG_IF(FATAL, status != atb::NO_ERROR)
      << "RopeOperation execute failed, status=" << status;
  return {output_tensors_.at(0), output_tensors_.at(1)};
}

void NpuRopeLayerImpl::build_node_variant_pack(
    atb_speed::Model::Node& node,
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& cos,
    const torch::Tensor& sin,
    const torch::Tensor& seq_len,
    std::vector<int32_t>& seq_len_host) {
  CHECK(query.dim() == 2 || query.dim() == 3);
  CHECK_EQ(query.dim(), key.dim());
  CHECK_EQ(query.sizes(), key.sizes());
  CHECK_EQ(cos.sizes(), sin.sizes());
  CHECK_EQ(query.size(0), cos.size(0));
  CHECK_EQ(head_dim_ % 2, 0);
  CHECK_EQ(seq_len_host.size(), static_cast<size_t>(seq_len.numel()));

  if (query.dim() == 3) {
    CHECK_EQ(query.size(-1), head_dim_);
  } else {
    CHECK_EQ(query.size(-1) % head_dim_, 0);
  }
  const int64_t rotary_dim = head_dim_ / 2;
  CHECK(cos.size(-1) == head_dim_ || cos.size(-1) == rotary_dim);
  auto to_nd = [](const torch::Tensor& tensor) {
    auto contiguous = tensor.is_contiguous() ? tensor : tensor.contiguous();
#ifdef TORCH_HIGHER_THAN_PTA6
    if (at_npu::native::get_npu_format(contiguous) != ACL_FORMAT_ND) {
      return at_npu::native::npu_format_cast(contiguous, ACL_FORMAT_ND);
    }
#else
    if (at_npu::native::NPUNativeFunctions::get_npu_format(contiguous) !=
        ACL_FORMAT_ND) {
      return at_npu::native::NPUNativeFunctions::npu_format_cast(contiguous,
                                                                 ACL_FORMAT_ND);
    }
#endif
    return contiguous;
  };
  auto query_input = to_nd(query);
  auto key_input = to_nd(key);
  CHECK(query_input.scalar_type() == torch::kFloat16 ||
        query_input.scalar_type() == torch::kBFloat16)
      << "ATB Rope only supports float16/bfloat16 query and key";
  CHECK_EQ(key_input.scalar_type(), query_input.scalar_type());
  internal_query_ = atb_speed::Utils::AtTensor2Tensor(query_input);
  internal_key_ = atb_speed::Utils::AtTensor2Tensor(key_input);
  CHECK_EQ(cos.scalar_type(), query_input.scalar_type());
  CHECK_EQ(sin.scalar_type(), query_input.scalar_type());
  CHECK_EQ(seq_len.scalar_type(), torch::kInt);
  auto cos_atb = to_nd(cos);
  auto sin_atb = to_nd(sin);
  auto seq_len_atb = to_nd(seq_len);
  CHECK_EQ(cos_atb.scalar_type(), query_input.scalar_type());
  CHECK_EQ(sin_atb.scalar_type(), query_input.scalar_type());
  CHECK_EQ(seq_len_atb.scalar_type(), torch::kInt);
  internal_cos_ = atb_speed::Utils::AtTensor2Tensor(cos_atb);
  internal_sin_ = atb_speed::Utils::AtTensor2Tensor(sin_atb);
  internal_seq_len_ = atb_speed::Utils::AtTensor2Tensor(seq_len_atb);

  node.variantPack.inTensors.at(0) = internal_query_;
  node.variantPack.inTensors.at(1) = internal_key_;
  node.variantPack.inTensors.at(2) = internal_cos_;
  node.variantPack.inTensors.at(3) = internal_sin_;
  node.variantPack.inTensors.at(4) = internal_seq_len_;
  node.variantPack.inTensors.at(4).hostData = seq_len_host.data();

  atb::SVector<atb::TensorDesc> input_descs;
  input_descs.resize(node.operation->GetInputNum());
  input_descs.at(0) = internal_query_.desc;
  input_descs.at(1) = internal_key_.desc;
  input_descs.at(2) = internal_cos_.desc;
  input_descs.at(3) = internal_sin_.desc;
  input_descs.at(4) = internal_seq_len_.desc;
  atb::SVector<atb::TensorDesc> output_descs;
  output_descs.resize(node.operation->GetOutputNum());
  const atb::Status status =
      node.operation->InferShape(input_descs, output_descs);
  CHECK_EQ(status, atb::NO_ERROR);

  output_tensors_.at(0) =
      atb_speed::Utils::CreateAtTensorFromTensorDesc(output_descs.at(0));
  output_tensors_.at(1) =
      atb_speed::Utils::CreateAtTensorFromTensorDesc(output_descs.at(1));
  node.variantPack.outTensors.at(0) =
      atb_speed::Utils::AtTensor2Tensor(output_tensors_.at(0));
  node.variantPack.outTensors.at(1) =
      atb_speed::Utils::AtTensor2Tensor(output_tensors_.at(1));
}

}  // namespace layer
}  // namespace xllm

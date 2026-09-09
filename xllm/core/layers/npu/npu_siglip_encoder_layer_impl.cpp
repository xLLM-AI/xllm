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

#include "npu_siglip_encoder_layer_impl.h"

#include "common/global_flags.h"
#include "nlohmann/json.hpp"

namespace xllm {
namespace layer {

NpuSiglipEncoderLayerUpImpl::NpuSiglipEncoderLayerUpImpl(
    const ModelContext& context,
    const std::string& prefix)
    : BaseLayer(context),
      graph_("siglip_encoder_layer_up"),
      model_args_(context.get_model_args()),
      options_(context.get_tensor_options()),
      prefix_(prefix) {
  loader_ = std::make_unique<SiglipEncoderUpLoader>(context);
  build_graph(prefix);
}

void NpuSiglipEncoderLayerUpImpl::build_graph(const std::string& prefix) {
  // set graph input names and output names
  std::vector<std::string> input_names = {
      "hidden_states",
      "seq_len",
      prefix + "layer_norm1.weight",
      prefix + "layer_norm1.bias",
      prefix + "self_attn.q_proj.weight",
      prefix + "self_attn.q_proj.bias",
      prefix + "self_attn.k_proj.weight",
      prefix + "self_attn.k_proj.bias",
      prefix + "self_attn.v_proj.weight",
      prefix + "self_attn.v_proj.bias",
  };
  std::vector<std::string> output_names = {"layer_up_out"};
  graph_.AddInputOutput(input_names, output_names);

  // layer_norm1
  // layer norm input shape: [batch, seq_len, hidden_size]
  // so set the parameter beginParamsAxis and beginNormAxis to 2
  nlohmann::json layer_norm1_op_param;
  layer_norm1_op_param["layerType"] = "LAYER_NORM_NORM";
  layer_norm1_op_param["normParam"] = {
      {"quantType", "QUANT_UNDEFINED"},
      {"epsilon", model_args_.mm_layer_norm_eps()},
      {"beginParamsAxis", 2},
      {"beginNormAxis", 2}};
  auto layer_norm1_op = std::make_shared<atb_torch::BaseOperation>(
      "LayerNorm", layer_norm1_op_param.dump(), "layer_norm1");
  ops_.emplace_back(layer_norm1_op);
  std::vector<std::string> layer_norm1_inputs = {"hidden_states",
                                                 prefix + "layer_norm1.weight",
                                                 prefix + "layer_norm1.bias"};
  std::vector<std::string> layer_norm1_outputs = {"attn_in"};
  graph_.AddOperation(
      layer_norm1_op.get(), layer_norm1_inputs, layer_norm1_outputs);

  // q_proj
  nlohmann::json q_proj_op_param;
  q_proj_op_param["transposeB"] = true;
  q_proj_op_param["hasBias"] = true;
  auto q_proj_op = std::make_shared<atb_torch::BaseOperation>(
      "Linear", q_proj_op_param.dump(), "q_proj_linear");
  ops_.emplace_back(q_proj_op);
  std::vector<std::string> q_proj_inputs = {"attn_in",
                                            prefix + "self_attn.q_proj.weight",
                                            prefix + "self_attn.q_proj.bias"};
  std::vector<std::string> q_proj_outputs = {"q"};
  graph_.AddOperation(q_proj_op.get(), q_proj_inputs, q_proj_outputs);

  // k_proj
  nlohmann::json k_proj_op_param;
  k_proj_op_param["transposeB"] = true;
  k_proj_op_param["hasBias"] = true;
  auto k_proj_op = std::make_shared<atb_torch::BaseOperation>(
      "Linear", k_proj_op_param.dump(), "k_proj_linear");
  ops_.emplace_back(k_proj_op);
  std::vector<std::string> k_proj_inputs = {"attn_in",
                                            prefix + "self_attn.k_proj.weight",
                                            prefix + "self_attn.k_proj.bias"};
  std::vector<std::string> k_proj_outputs = {"k"};
  graph_.AddOperation(k_proj_op.get(), k_proj_inputs, k_proj_outputs);

  // v_proj
  nlohmann::json v_proj_op_param;
  v_proj_op_param["transposeB"] = true;
  v_proj_op_param["hasBias"] = true;
  auto v_proj_op = std::make_shared<atb_torch::BaseOperation>(
      "Linear", v_proj_op_param.dump(), "v_proj_linear");
  ops_.emplace_back(v_proj_op);
  std::vector<std::string> v_proj_inputs = {"attn_in",
                                            prefix + "self_attn.v_proj.weight",
                                            prefix + "self_attn.v_proj.bias"};
  std::vector<std::string> v_proj_outputs = {"v"};
  graph_.AddOperation(v_proj_op.get(), v_proj_inputs, v_proj_outputs);

  // qkv reshape
  // [batch, seq_len, hidden_size] => [batch * seq_len, head_num, head_dim]
  atb_torch::ReshapeFunc reshape_qkv =
      [this](const std::vector<int64_t>& org_shape) -> std::vector<int64_t> {
    std::vector<int64_t> new_shape = {org_shape[0] * org_shape[1],
                                      model_args_.mm_num_attention_heads(),
                                      model_args_.mm_head_dim()};
    return new_shape;
  };

  graph_.AddReshape("q", "q_reshape", reshape_qkv);
  graph_.AddReshape("k", "k_reshape", reshape_qkv);
  graph_.AddReshape("v", "v_reshape", reshape_qkv);

  // self attn
  nlohmann::json self_attn_op_param;
  float qk_scale =
      1.0f / std::sqrt(static_cast<float>(model_args_.mm_head_dim()));
  self_attn_op_param["headNum"] = model_args_.mm_num_attention_heads();
  self_attn_op_param["kvHeadNum"] = model_args_.mm_num_attention_heads();
  self_attn_op_param["qkScale"] = qk_scale;
  self_attn_op_param["calcType"] = "PA_ENCODER";
  self_attn_op_param["kernelType"] = "KERNELTYPE_HIGH_PRECISION";
  auto self_attn_op = std::make_shared<atb_torch::BaseOperation>(
      "SelfAttention", self_attn_op_param.dump(), "self_attention");
  ops_.emplace_back(self_attn_op);
  std::vector<std::string> self_attn_inputs = {
      "q_reshape", "k_reshape", "v_reshape", "seq_len"};
  std::vector<std::string> self_attn_outputs = {"layer_up_out"};
  graph_.AddOperation(self_attn_op.get(), self_attn_inputs, self_attn_outputs);

  // build graph
  graph_.Build();
}

void NpuSiglipEncoderLayerUpImpl::load_state_dict(const StateDict& state_dict) {
  loader_->load_state_dict(state_dict);
  auto weights_map = loader_->get_weights_map();
  graph_.SetWeights(weights_map);
}

torch::Tensor NpuSiglipEncoderLayerUpImpl::forward(const torch::Tensor& x) {
  // set graph forward inputs
  atb_torch::TorchTensorMap inputs;
  atb_torch::TorchTensorMap outputs;
  atb_torch::TorchTensorMap binds;
  inputs["hidden_states"] = x;

  // create seq len from input shape
  torch::Tensor seq_len = torch::full(
      {x.size(0)}, x.size(1), torch::dtype(torch::kInt32).device(x.device()));
  torch::Tensor seq_len_cpu = seq_len.to(torch::kCPU);
  inputs["seq_len"] = seq_len;
  outputs["layer_up_out"] = x;
  binds["seq_len"] = seq_len_cpu;
  auto output = graph_.Forward(inputs, outputs, binds);
  return output["layer_up_out"];
}

NpuSiglipEncoderLayerDownImpl::NpuSiglipEncoderLayerDownImpl(
    const ModelContext& context,
    const std::string& prefix)
    : BaseLayer(context),
      graph_("siglip_encoder_layer_down"),
      model_args_(context.get_model_args()),
      options_(context.get_tensor_options()),
      prefix_(prefix) {
  loader_ = std::make_unique<SiglipEncoderDownLoader>(context);

  build_graph(prefix);
}

void NpuSiglipEncoderLayerDownImpl::build_graph(const std::string& prefix) {
  // set graph input names and output names
  std::vector<std::string> input_names = {"residual",
                                          "hidden_states",
                                          prefix + "self_attn.out_proj.weight",
                                          prefix + "self_attn.out_proj.bias",
                                          prefix + "layer_norm2.weight",
                                          prefix + "layer_norm2.bias",
                                          prefix + "mlp.fc1.weight",
                                          prefix + "mlp.fc1.bias",
                                          prefix + "mlp.fc2.weight",
                                          prefix + "mlp.fc2.bias"};
  std::vector<std::string> output_names = {"layer_down_out"};
  graph_.AddInputOutput(input_names, output_names);

  // out_proj
  nlohmann::json out_proj_op_param;
  out_proj_op_param["transposeB"] = true;
  out_proj_op_param["hasBias"] = true;
  auto out_proj_op = std::make_shared<atb_torch::BaseOperation>(
      "Linear", out_proj_op_param.dump(), "out_proj_linear");
  ops_.emplace_back(out_proj_op);
  std::vector<std::string> out_proj_inputs = {
      "hidden_states",
      prefix + "self_attn.out_proj.weight",
      prefix + "self_attn.out_proj.bias"};
  std::vector<std::string> out_proj_outputs = {"dense_out"};
  graph_.AddOperation(out_proj_op.get(), out_proj_inputs, out_proj_outputs);

  // add
  nlohmann::json attn_res_add_op_param;
  attn_res_add_op_param["elewiseType"] = "ELEWISE_ADD";
  auto attn_res_add_op = std::make_shared<atb_torch::BaseOperation>(
      "Elewise", attn_res_add_op_param.dump(), "attn_res_add");
  ops_.emplace_back(attn_res_add_op);
  std::vector<std::string> attn_res_add_inputs = {"residual", "dense_out"};
  std::vector<std::string> attn_res_add_outputs = {"attn_res_add_out"};
  graph_.AddOperation(
      attn_res_add_op.get(), attn_res_add_inputs, attn_res_add_outputs);

  // layer_norm2
  nlohmann::json layer_norm2_op_param;
  layer_norm2_op_param["layerType"] = "LAYER_NORM_NORM";
  layer_norm2_op_param["normParam"] = {
      {"quantType", "QUANT_UNDEFINED"},
      {"epsilon", model_args_.mm_layer_norm_eps()},
      {"beginParamsAxis", 2},
      {"beginNormAxis", 2}};
  auto layer_norm2_op = std::make_shared<atb_torch::BaseOperation>(
      "LayerNorm", layer_norm2_op_param.dump(), "layer_norm2");
  ops_.emplace_back(layer_norm2_op);
  std::vector<std::string> layer_norm2_inputs = {"attn_res_add_out",
                                                 prefix + "layer_norm2.weight",
                                                 prefix + "layer_norm2.bias"};
  std::vector<std::string> layer_norm2_outputs = {"mlp_in"};
  graph_.AddOperation(
      layer_norm2_op.get(), layer_norm2_inputs, layer_norm2_outputs);

  nlohmann::json fc1_op_param;
  fc1_op_param["transposeB"] = true;
  fc1_op_param["hasBias"] = true;
  auto fc1_op = std::make_shared<atb_torch::BaseOperation>(
      "Linear", fc1_op_param.dump(), "fc1_Linear");
  ops_.emplace_back(fc1_op);
  std::vector<std::string> fc1_inputs = {
      "mlp_in", prefix + "mlp.fc1.weight", prefix + "mlp.fc1.bias"};
  std::vector<std::string> fc1_outputs = {"fc1_out"};
  graph_.AddOperation(fc1_op.get(), fc1_inputs, fc1_outputs);

  nlohmann::json act_op_param;
  act_op_param["activationType"] = "ACTIVATION_GELU";
  auto act_op = std::make_shared<atb_torch::BaseOperation>(
      "Activation", act_op_param.dump(), "Activation");
  ops_.emplace_back(act_op);
  std::vector<std::string> act_inputs = {"fc1_out"};
  std::vector<std::string> act_outputs = {"act_out"};
  graph_.AddOperation(act_op.get(), act_inputs, act_outputs);

  nlohmann::json fc2_op_param;
  fc2_op_param["transposeB"] = true;
  fc2_op_param["hasBias"] = true;
  auto fc2_op = std::make_shared<atb_torch::BaseOperation>(
      "Linear", fc2_op_param.dump(), "fc2_Linear");
  ops_.emplace_back(fc2_op);
  std::vector<std::string> fc2_inputs = {
      "act_out", prefix + "mlp.fc2.weight", prefix + "mlp.fc2.bias"};
  std::vector<std::string> fc2_outputs = {"mlp_out"};
  graph_.AddOperation(fc2_op.get(), fc2_inputs, fc2_outputs);

  nlohmann::json mlp_res_add_op_param;
  mlp_res_add_op_param["elewiseType"] = "ELEWISE_ADD";
  auto mlp_res_add_op = std::make_shared<atb_torch::BaseOperation>(
      "Elewise", mlp_res_add_op_param.dump(), "mlp_res_add");
  ops_.emplace_back(mlp_res_add_op);
  std::vector<std::string> mlp_res_add_inputs = {"attn_res_add_out", "mlp_out"};
  std::vector<std::string> mlp_res_add_outputs = {"layer_down_out"};
  graph_.AddOperation(
      mlp_res_add_op.get(), mlp_res_add_inputs, mlp_res_add_outputs);

  // build graph
  graph_.Build();
}

void NpuSiglipEncoderLayerDownImpl::load_state_dict(
    const StateDict& state_dict) {
  loader_->load_state_dict(state_dict);
  auto& weights_map = loader_->get_weights_map();
  graph_.SetWeights(weights_map);
}

torch::Tensor NpuSiglipEncoderLayerDownImpl::forward(torch::Tensor& x,
                                                     torch::Tensor& y) {
  // set graph forward inputs
  atb_torch::TorchTensorMap inputs;
  atb_torch::TorchTensorMap outputs;
  atb_torch::TorchTensorMap binds;
  inputs["residual"] = x;
  inputs["hidden_states"] = y;

  outputs["layer_down_out"] = y;
  auto output = graph_.Forward(inputs, outputs, binds);
  return output["layer_down_out"];
}

NpuSiglipEncoderLayerImpl::NpuSiglipEncoderLayerImpl(
    const ModelContext& context,
    const std::string& prefix)
    : BaseLayer(context),
      model_args_(context.get_model_args()),
      options_(context.get_tensor_options()),
      prefix_(prefix) {
  up_ = NpuSiglipEncoderLayerUp(context, prefix);
  down_ = NpuSiglipEncoderLayerDown(context, prefix);
}

void NpuSiglipEncoderLayerImpl::load_state_dict(const StateDict& state_dict) {
  up_->load_state_dict(state_dict);
  down_->load_state_dict(state_dict);
}

torch::Tensor NpuSiglipEncoderLayerImpl::forward(const torch::Tensor& x) {
  auto residual = x.clone();
  auto batch = x.size(0);
  auto seq_len = x.size(1);

  auto out = up_->forward(x);
  out = out.view(
      {batch,
       seq_len,
       model_args_.mm_num_attention_heads() * model_args_.mm_head_dim()});

  return down_->forward(residual, out);
}

}  // namespace layer
}  // namespace xllm

/* Copyright 2025-2026 The xLLM Authors.

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

#include <torch/torch.h>

#include "framework/model/model_input_params.h"
#include "framework/request/sequence.h"
#include "runtime/forward_params.h"
#include "worker.pb.h"

namespace xllm {

class Stream;

bool forward_input_to_packed_proto(
    const ForwardInput& input,
    proto::PackedForwardInput* packed_forward_input);

void packed_proto_to_forward_input(
    const proto::PackedForwardInput& packed_forward_input,
    ForwardInput& forward_input,
    const torch::Device& device,
    Stream* stream);

void proto_to_forward_output(const proto::ForwardOutput& pb_output,
                             RawForwardOutput& raw_forward_output);

void forward_output_to_proto(
    const torch::Tensor& next_tokens,
    const torch::Tensor& logprobs,
    const torch::Tensor& top_tokens,
    const torch::Tensor& top_logprobs,
    const torch::Tensor& embeddings,
    const std::vector<std::vector<torch::Tensor>>& mm_embeddings,
    const torch::Tensor& expert_load_data,
    int32_t prepared_layer_id,
    const torch::Tensor& src_seq_idxes,
    const torch::Tensor& out_tokens,
    const torch::Tensor& out_logprobs,
    const std::vector<torch::Tensor>& dit_images,
    const std::vector<std::string>& dit_text_output,
    proto::ForwardOutput* pb_forward_output);

Token build_token(int64_t index,
                  torch::Tensor token_ids,
                  torch::Tensor logprobs,
                  torch::Tensor top_tokens,
                  torch::Tensor top_logprobs);

uint64_t proto_to_block_transfer_info(
    const proto::BlockTransferInfos& pb_block_transfer_info,
    std::vector<BlockTransferInfo>& block_transfer_info);

bool block_transfer_info_to_proto(
    const std::vector<BlockTransferInfo>& block_transfer_info,
    proto::BlockTransferInfos* pb_block_transfer_info);

bool block_transfer_info_to_proto(
    const uint64_t batch_id,
    const std::vector<BlockTransferInfo>& block_transfer_info,
    proto::BlockTransferInfos* pb_block_transfer_info);

bool dit_forward_input_to_proto(const DiTForwardInput& dit_inputs,
                                proto::DiTForwardInput* pb_dit_inputs);

bool generation_params_to_proto(
    const DiTGenerationParams& dit_generation_params,
    proto::DiTGenerationParams* pb_dit_generation_params);

bool proto_to_dit_forward_input(const proto::DiTForwardInput& pb_dit_inputs,
                                DiTForwardInput& dit_inputs);

bool proto_to_generation_params(
    const proto::DiTGenerationParams& pb_dit_generation_params,
    DiTGenerationParams& dit_generation_params);

bool proto_to_dit_forward_output(const proto::DiTForwardOutput& pb_dit_outputs,
                                 DiTForwardOutput& dit_outputs);

bool torch_tensor_to_proto_tensor(const torch::Tensor& torch_tensor,
                                  proto::Tensor* proto_tensor);
}  // namespace xllm

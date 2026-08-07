/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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

namespace xllm {

class ProcessGroup;

namespace layer {

struct DcpAttentionResult {
  torch::Tensor output;
  torch::Tensor lse;
};

DcpAttentionResult merge_dcp_attention_shards(
    const torch::Tensor& partial_outputs,
    const torch::Tensor& partial_lse);

DcpAttentionResult all_gather_and_merge_dcp_attention(
    const torch::Tensor& local_output,
    const torch::Tensor& local_lse,
    ProcessGroup& dcp_group);

}  // namespace layer
}  // namespace xllm

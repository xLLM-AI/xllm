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

#include <cstdint>
#include <torch/csrc/distributed/c10d/Backend.hpp>

namespace xllm {
class ProcessGroup;
}

namespace xllm::layer {

// Candidates after the DCP all-gather. The leading dimension is DCP rank.
struct DcpIndexerGatheredCandidates {
  torch::Tensor scores;
  torch::Tensor global_slots;
};

// In-flight DCP candidate all-gather. Holds the gathered buffers alive until
// both collectives retire; the caller enqueues independent compute between
// the launch and the finish to overlap it with the communication.
struct DcpIndexerGatherAsyncCtx {
  torch::Tensor gathered_scores;
  torch::Tensor gathered_global_slots;
  c10::intrusive_ptr<c10d::Work> score_work;
  c10::intrusive_ptr<c10d::Work> slot_work;
};

// Submits the rank-local candidate tuples for the DCP all-gather without
// waiting. The score and slot collectives intentionally remain separate so
// global int32 token ids are never cast to fp32 for transport.
DcpIndexerGatherAsyncCtx launch_dcp_indexer_candidate_gather(
    const torch::Tensor& local_scores,
    const torch::Tensor& local_global_slots,
    ProcessGroup* dcp_group);

// Waits for a launched candidate all-gather and returns the gathered buffers.
DcpIndexerGatheredCandidates finish_dcp_indexer_candidate_gather(
    DcpIndexerGatherAsyncCtx&& ctx);

}  // namespace xllm::layer

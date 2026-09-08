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

#include <glog/logging.h>
#include <gtest/gtest.h>
#include <sys/wait.h>
#include <torch/torch.h>
#include <unistd.h>

#include <cerrno>
#include <chrono>
#include <cstring>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "framework/parallel_state/process_group.h"
#include "layers/mlu/dcp_indexer_candidate.h"
#include "platform/device.h"
#include "platform/platform.h"
#include "util/net.h"

namespace xllm::layer {
namespace {

constexpr int32_t kExitCodeSkip = 77;
constexpr std::chrono::seconds kChildProcessTimeout{120};
constexpr std::chrono::milliseconds kChildPollInterval{10};

bool wait_for_children(const std::vector<pid_t>& child_pids) {
  std::vector<bool> finished(child_pids.size(), false);
  int64_t remaining = static_cast<int64_t>(child_pids.size());
  const std::chrono::steady_clock::time_point deadline =
      std::chrono::steady_clock::now() + kChildProcessTimeout;
  bool any_failed = false;
  bool any_skipped = false;

  while (remaining > 0 && std::chrono::steady_clock::now() < deadline) {
    bool made_progress = false;
    for (size_t rank = 0; rank < child_pids.size(); ++rank) {
      if (finished[rank]) {
        continue;
      }
      int status = 0;
      const pid_t result = ::waitpid(child_pids[rank], &status, WNOHANG);
      if (result == 0) {
        continue;
      }
      finished[rank] = true;
      --remaining;
      made_progress = true;
      if (result < 0) {
        any_failed = true;
        LOG(ERROR) << "Failed waiting for DCP candidate rank " << rank << ": "
                   << std::strerror(errno);
        continue;
      }
      if (!WIFEXITED(status)) {
        any_failed = true;
        LOG(ERROR) << "DCP candidate rank " << rank << " did not exit normally";
        continue;
      }
      const int32_t exit_code = WEXITSTATUS(status);
      if (exit_code == kExitCodeSkip) {
        any_skipped = true;
      } else if (exit_code != 0) {
        any_failed = true;
        LOG(ERROR) << "DCP candidate rank " << rank << " failed with code "
                   << exit_code;
      }
    }
    if (remaining > 0 && !made_progress) {
      std::this_thread::sleep_for(kChildPollInterval);
    }
  }

  for (size_t rank = 0; rank < child_pids.size(); ++rank) {
    if (!finished[rank]) {
      any_failed = true;
      LOG(ERROR) << "Timed out waiting for DCP candidate rank " << rank;
      if (::kill(child_pids[rank], SIGKILL) != 0 && errno != ESRCH) {
        LOG(ERROR) << "Failed to kill DCP candidate rank " << rank << ": "
                   << std::strerror(errno);
      }
      int status = 0;
      ::waitpid(child_pids[rank], &status, 0);
    }
  }
  return !any_failed && !any_skipped;
}

// Rank-encoded local fixtures: score[r] = 10r + {2, 1}, slot[r] = 100r + {5,
// 7}.
torch::Tensor make_local_scores(int32_t rank,
                                const torch::TensorOptions& options) {
  const float rank_score = static_cast<float>(rank * 10);
  return torch::tensor({{rank_score + 2.0f, rank_score + 1.0f}}, options);
}

torch::Tensor make_local_slots(int32_t rank,
                               const torch::TensorOptions& options) {
  const int32_t rank_slot = rank * 100;
  return torch::tensor({{rank_slot + 5, rank_slot + 7}}, options);
}

void verify_gathered_candidates(const DcpIndexerGatheredCandidates& gathered,
                                int32_t world_size,
                                const torch::Device& device) {
  const torch::TensorOptions score_options =
      torch::TensorOptions().dtype(torch::kFloat32).device(device);
  const torch::TensorOptions slot_options =
      torch::TensorOptions().dtype(torch::kInt32).device(device);
  CHECK_EQ(gathered.scores.sizes(), (torch::IntArrayRef{world_size, 1, 2}));
  CHECK_EQ(gathered.global_slots.sizes(),
           (torch::IntArrayRef{world_size, 1, 2}));
  for (int32_t source_rank = 0; source_rank < world_size; ++source_rank) {
    const torch::Tensor expected_scores =
        make_local_scores(source_rank, score_options);
    const torch::Tensor expected_slots =
        make_local_slots(source_rank, slot_options);
    CHECK(torch::equal(gathered.scores[source_rank], expected_scores));
    CHECK(torch::equal(gathered.global_slots[source_rank], expected_slots));
  }
}

int32_t run_candidate_collective_child(int32_t rank,
                                       int32_t world_size,
                                       int32_t port,
                                       const std::string& host) {
  try {
    if (Platform::device_count() < world_size) {
      LOG(WARNING) << "DCP candidate collective requires " << world_size
                   << " MLU devices";
      return kExitCodeSkip;
    }
    Device xllm_device(rank);
    xllm_device.set_device();
    const torch::Device device = xllm_device.unwrap();
    std::unique_ptr<ProcessGroup> process_group =
        create_process_group(rank,
                             world_size,
                             world_size,
                             port,
                             /*use_dp_master=*/false,
                             host,
                             "dcp_indexer_candidate_collective_test",
                             device);
    CHECK(process_group) << "Failed to create DCP candidate process group";

    torch::Tensor rendezvous = torch::ones(
        {1}, torch::TensorOptions().dtype(torch::kFloat32).device(device));
    process_group->allreduce(rendezvous);
    CHECK_EQ(rendezvous.item<float>(), static_cast<float>(world_size));

    const torch::TensorOptions score_options =
        torch::TensorOptions().dtype(torch::kFloat32).device(device);
    const torch::TensorOptions slot_options =
        torch::TensorOptions().dtype(torch::kInt32).device(device);

    DcpIndexerGatherAsyncCtx gather = launch_dcp_indexer_candidate_gather(
        make_local_scores(rank, score_options),
        make_local_slots(rank, slot_options),
        process_group.get());
    const DcpIndexerGatheredCandidates gathered =
        finish_dcp_indexer_candidate_gather(std::move(gather));
    xllm_device.synchronize_default_stream();
    verify_gathered_candidates(gathered, world_size, device);
    return 0;
  } catch (const std::exception& error) {
    LOG(ERROR) << "DCP candidate rank " << rank << " failed: " << error.what();
    return 1;
  }
}

// Exercises the launch/finish overlap contract: independent device compute is
// enqueued while the candidate all-gather is in flight, and both the compute
// result and the gathered buffers must stay intact after the finish.
int32_t run_candidate_async_collective_child(int32_t rank,
                                             int32_t world_size,
                                             int32_t port,
                                             const std::string& host) {
  constexpr int64_t kOverlapMatrixSize = 8;
  try {
    if (Platform::device_count() < world_size) {
      LOG(WARNING) << "DCP candidate collective requires " << world_size
                   << " MLU devices";
      return kExitCodeSkip;
    }
    Device xllm_device(rank);
    xllm_device.set_device();
    const torch::Device device = xllm_device.unwrap();
    std::unique_ptr<ProcessGroup> process_group =
        create_process_group(rank,
                             world_size,
                             world_size,
                             port,
                             /*use_dp_master=*/false,
                             host,
                             "dcp_indexer_candidate_async_collective_test",
                             device);
    CHECK(process_group) << "Failed to create DCP candidate process group";

    torch::Tensor rendezvous = torch::ones(
        {1}, torch::TensorOptions().dtype(torch::kFloat32).device(device));
    process_group->allreduce(rendezvous);
    CHECK_EQ(rendezvous.item<float>(), static_cast<float>(world_size));

    const torch::TensorOptions score_options =
        torch::TensorOptions().dtype(torch::kFloat32).device(device);
    const torch::TensorOptions slot_options =
        torch::TensorOptions().dtype(torch::kInt32).device(device);

    DcpIndexerGatherAsyncCtx gather = launch_dcp_indexer_candidate_gather(
        make_local_scores(rank, score_options),
        make_local_slots(rank, slot_options),
        process_group.get());
    const torch::Tensor overlap_probe = torch::matmul(
        torch::ones({kOverlapMatrixSize, kOverlapMatrixSize}, score_options),
        torch::full({kOverlapMatrixSize, kOverlapMatrixSize},
                    static_cast<float>(rank + 1),
                    score_options));
    const DcpIndexerGatheredCandidates gathered =
        finish_dcp_indexer_candidate_gather(std::move(gather));
    xllm_device.synchronize_default_stream();
    CHECK_EQ(overlap_probe.sum().item<float>(),
             static_cast<float>(kOverlapMatrixSize * kOverlapMatrixSize *
                                kOverlapMatrixSize * (rank + 1)));
    verify_gathered_candidates(gathered, world_size, device);
    return 0;
  } catch (const std::exception& error) {
    LOG(ERROR) << "DCP candidate rank " << rank << " failed: " << error.what();
    return 1;
  }
}

using CandidateChildFn = int32_t (*)(int32_t,
                                     int32_t,
                                     int32_t,
                                     const std::string&);

void run_candidate_collective_test(int32_t world_size,
                                   CandidateChildFn child_fn) {
  const int32_t port = net::get_local_free_port();
  ASSERT_GT(port, 0) << "Failed to allocate a DCP candidate rendezvous port";
  std::vector<pid_t> child_pids;
  child_pids.reserve(world_size);
  for (int32_t rank = 0; rank < world_size; ++rank) {
    const pid_t child_pid = ::fork();
    ASSERT_GE(child_pid, 0) << "Failed to fork DCP candidate rank " << rank;
    if (child_pid == 0) {
      const int32_t exit_code =
          child_fn(rank, world_size, port, /*host=*/"127.0.0.1");
      _exit(exit_code);
    }
    child_pids.emplace_back(child_pid);
  }
  EXPECT_TRUE(wait_for_children(child_pids));
}

// The four-rank sweep covers the richer interleave topology on real CNCL
// groups; the two-rank overlap contract below covers the launch/finish split.
TEST(DcpIndexerCandidateMultiDeviceTest, ExchangesCandidatesAcrossFourRanks) {
  run_candidate_collective_test(/*world_size=*/4,
                                run_candidate_collective_child);
}

TEST(DcpIndexerCandidateMultiDeviceTest,
     OverlapsComputeWithLaunchedCandidateGather) {
  run_candidate_collective_test(/*world_size=*/2,
                                run_candidate_async_collective_child);
}

}  // namespace
}  // namespace xllm::layer

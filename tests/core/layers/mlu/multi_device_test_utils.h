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

#include <glog/logging.h>
#include <gtest/gtest.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <functional>
#include <thread>
#include <vector>

#include "core/platform/platform.h"

namespace xllm::test {
namespace detail {

// The production communicator uses a contiguous TCPStore port window. Probe it
// below the usual Linux ephemeral range.
inline int32_t group_port() {
  for (int32_t attempt = 0; attempt < 100; ++attempt) {
    const int32_t base = 10000 + (getpid() + attempt * 64) % 15000;
    std::vector<int32_t> sockets;
    sockets.reserve(/*new_cap=*/64);
    bool available = true;
    for (int32_t offset = 0; offset < 64; ++offset) {
      const int32_t fd = socket(AF_INET, SOCK_STREAM, /*protocol=*/0);
      PCHECK(fd >= 0);
      sockets.emplace_back(fd);
      sockaddr_in addr{};
      addr.sin_family = AF_INET;
      addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
      addr.sin_port = htons(static_cast<uint16_t>(base + offset));
      if (bind(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0) {
        available = false;
        break;
      }
    }
    for (int32_t fd : sockets) {
      PCHECK(close(fd) == 0);
    }
    if (available) {
      return base;
    }
  }
  LOG(FATAL) << "Cannot reserve communicator port window";
  return -1;
}

inline void stop_ranks(const std::vector<pid_t>& pids,
                       std::vector<char>& done,
                       std::chrono::milliseconds termination_grace) {
  if (termination_grace.count() > 0) {
    for (size_t i = 0; i < pids.size(); ++i) {
      if (!done[i]) {
        kill(pids[i], SIGTERM);
      }
    }
    std::this_thread::sleep_for(termination_grace);
  }
  for (size_t i = 0; i < pids.size(); ++i) {
    if (!done[i]) {
      if (termination_grace.count() > 0) {
        int status = 0;
        pid_t result;
        do {
          result = waitpid(pids[i], &status, WNOHANG);
        } while (result < 0 && errno == EINTR);
        if (result != 0) {
          done[i] = 1;
          continue;
        }
      }
      kill(pids[i], SIGKILL);
    }
  }
  for (size_t i = 0; i < pids.size(); ++i) {
    if (done[i]) {
      continue;
    }
    int status = 0;
    while (waitpid(pids[i], &status, /*options=*/0) < 0 && errno == EINTR) {
    }
  }
}

}  // namespace detail

// Device discovery stays in the child so the parent never initializes MLU
// before forking the test ranks.
inline void run_multi_device(
    int32_t ranks,
    const std::function<void(int32_t, int32_t)>& run_rank,
    std::chrono::seconds timeout = std::chrono::minutes(30),
    std::chrono::milliseconds termination_grace =
        std::chrono::milliseconds(0)) {
  CHECK_GT(ranks, 0);
  constexpr int32_t kSkip = 77;
  const int32_t port = detail::group_port();
  std::vector<pid_t> pids;
  pids.reserve(ranks);
  std::vector<char> done(ranks, /*value=*/0);
  for (int32_t rank = 0; rank < ranks; ++rank) {
    const pid_t pid = fork();
    if (pid == 0) {
      try {
        if (Platform::device_count() < ranks) {
          _exit(kSkip);
        }
        run_rank(rank, port);
        const bool failed = ::testing::UnitTest::GetInstance()
                                ->current_test_info()
                                ->result()
                                ->Failed();
        std::fflush(nullptr);
        _exit(failed ? 1 : 0);
      } catch (const std::exception& error) {
        LOG(ERROR) << "rank=" << rank << " exception=" << error.what();
        _exit(/*status=*/1);
      }
    }
    if (pid < 0) {
      detail::stop_ranks(pids, done, termination_grace);
      FAIL() << "fork failed errno=" << errno;
    }
    pids.emplace_back(pid);
  }
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  int32_t remaining = ranks;
  int32_t skipped = 0;
  while (remaining > 0 && std::chrono::steady_clock::now() < deadline) {
    for (int32_t rank = 0; rank < ranks; ++rank) {
      if (done[rank]) {
        continue;
      }
      int status = 0;
      const pid_t result = waitpid(pids[rank], &status, WNOHANG);
      if (result == 0 || (result < 0 && errno == EINTR)) {
        continue;
      }
      done[rank] = 1;
      if (result < 0 || !WIFEXITED(status) ||
          (WEXITSTATUS(status) != 0 && WEXITSTATUS(status) != kSkip)) {
        detail::stop_ranks(pids, done, termination_grace);
        FAIL() << "rank=" << rank << " status=" << status;
      }
      --remaining;
      skipped += WEXITSTATUS(status) == kSkip;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(/*count=*/10));
  }
  if (remaining > 0) {
    detail::stop_ranks(pids, done, termination_grace);
    FAIL() << ranks << "-rank test timed out after " << timeout.count()
           << " seconds";
  }
  if (skipped == ranks) {
    GTEST_SKIP() << "Requires " << ranks << " visible MLU devices";
  }
  ASSERT_EQ(skipped, 0) << "Inconsistent device visibility";
}

}  // namespace xllm::test

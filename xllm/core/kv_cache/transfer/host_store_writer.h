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

#pragma once

#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <future>
#include <mutex>
#include <queue>
#include <thread>

namespace xllm {

// Serializes Host-to-Store writes and bounds the number of Host blocks pinned
// by accepted batches. The owner stops submitting before destroying the writer.
class HostStoreWriter final {
 public:
  explicit HostStoreWriter(size_t max_blocks);
  ~HostStoreWriter();

  std::future<uint32_t> submit(size_t blocks,
                               std::packaged_task<uint32_t()> put);

 private:
  struct Batch {
    size_t blocks;
    std::packaged_task<uint32_t()> put;
  };

  void run();

  const size_t max_blocks_;
  std::mutex mutex_;
  std::condition_variable work_;
  std::condition_variable capacity_;
  std::queue<Batch> queue_;
  size_t outstanding_blocks_ = 0;
  bool stopping_ = false;
  std::thread thread_;
};

}  // namespace xllm

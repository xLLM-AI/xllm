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

#include "core/kv_cache/transfer/host_store_writer.h"

#include <utility>

namespace xllm {

HostStoreWriter::HostStoreWriter(size_t max_blocks)
    : max_blocks_(max_blocks), thread_([this] { run(); }) {}

HostStoreWriter::~HostStoreWriter() {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    stopping_ = true;
  }
  work_.notify_one();
  thread_.join();
}

std::future<uint32_t> HostStoreWriter::submit(
    size_t blocks,
    std::packaged_task<uint32_t()> put) {
  auto completion = put.get_future();
  {
    std::unique_lock<std::mutex> lock(mutex_);
    capacity_.wait(lock, [this, blocks] {
      // An oversized batch can run alone without splitting a single RPC.
      return outstanding_blocks_ == 0 ||
             (blocks <= max_blocks_ &&
              outstanding_blocks_ <= max_blocks_ - blocks);
    });
    queue_.push({blocks, std::move(put)});
    outstanding_blocks_ += blocks;
  }
  work_.notify_one();
  return completion;
}

void HostStoreWriter::run() {
  for (;;) {
    Batch batch;
    {
      std::unique_lock<std::mutex> lock(mutex_);
      work_.wait(lock, [this] { return stopping_ || !queue_.empty(); });
      if (queue_.empty()) {
        return;
      }
      batch = std::move(queue_.front());
      queue_.pop();
    }
    batch.put();
    {
      std::lock_guard<std::mutex> lock(mutex_);
      outstanding_blocks_ -= batch.blocks;
    }
    capacity_.notify_all();
  }
}

}  // namespace xllm

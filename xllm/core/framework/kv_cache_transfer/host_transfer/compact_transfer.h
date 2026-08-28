/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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

#include <cstddef>
#include <memory>

#include "framework/kv_cache_transfer/host_transfer/transfer.h"
#include "platform/batch_memcpy.h"

namespace xllm {

class Device;
class Stream;
class CompactLoadExecutor;
class CompactOffloadExecutor;

struct CompactTransferConfig {
  size_t load_target_bytes = 64ULL * 1024 * 1024;
  size_t offload_target_bytes = 64ULL * 1024 * 1024;
};

class CompactHostKVTransfer final : public HostKVTransfer {
 public:
  CompactHostKVTransfer(HostKVLayout layout,
                        const Device& device,
                        const Stream& compute_stream,
                        uint32_t layer_copy_batches,
                        std::unique_ptr<BatchMemcpy> batch_memcpy = nullptr,
                        CompactTransferConfig config = {});
  ~CompactHostKVTransfer() override;

  HostKVLoadHandle prepare_load() override;
  void drain() override;

 protected:
  uint32_t load_event_count() const override;
  uint32_t layers_per_event() const override;
  bool load_impl(const HostKVRequest& request,
                 const HostKVLoadHandle& handle) override;
  bool offload_impl(const HostKVRequest& request) override;

 private:
  const Stream& compute_stream_;
  std::unique_ptr<BatchMemcpy> batch_memcpy_;
  std::unique_ptr<CompactLoadExecutor> load_executor_;
  std::unique_ptr<CompactOffloadExecutor> offload_executor_;
  bool drained_ = false;
};

}  // namespace xllm

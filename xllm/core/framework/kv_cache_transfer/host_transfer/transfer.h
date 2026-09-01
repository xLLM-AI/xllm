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

#include <cstdint>
#include <memory>
#include <optional>

#include "framework/kv_cache_transfer/host_transfer/layout.h"

namespace xllm {

class Device;
class LayerSynchronizer;
class Stream;

struct HostKVLoadHandle {
  std::shared_ptr<LayerSynchronizer> synchronizer;
  uint32_t layers_per_event = 1;
  std::optional<uint32_t> draft_event_index;
};

enum class HostKVTransferMode : uint8_t {
  AUTO,
  BASIC,
};

struct HostKVTransferConfig {
  uint32_t layer_copy_batches = 1;
  HostKVTransferMode mode = HostKVTransferMode::AUTO;
};

class HostKVTransfer {
 public:
  virtual ~HostKVTransfer() = default;

  virtual HostKVLoadHandle prepare_load(bool draft = false) = 0;
  // Success means all layer-ready events have been recorded.
  bool load(const HostKVRequest& request, const HostKVLoadHandle& handle);
  // Success means Host data is safe for CPU and Store access.
  bool offload(const HostKVRequest& request);
  // Shutdown calls this after request producers have stopped.
  virtual void drain() = 0;

 protected:
  explicit HostKVTransfer(HostKVLayout layout);

  const HostKVLayout& layout() const { return layout_; }
  virtual uint32_t load_event_count() const = 0;
  virtual uint32_t layers_per_event() const = 0;
  virtual bool load_impl(const HostKVRequest& request,
                         const HostKVLoadHandle& handle) = 0;
  virtual bool offload_impl(const HostKVRequest& request) = 0;

 private:
  bool valid_request(const HostKVRequest& request, bool is_load) const;
  bool valid_handle(const HostKVRequest& request,
                    const HostKVLoadHandle& handle) const;

  const HostKVLayout layout_;
};

std::unique_ptr<HostKVTransfer> create_host_kv_transfer(
    HostKVLayout layout,
    const Device& device,
    const Stream& compute_stream,
    const HostKVTransferConfig& config);

}  // namespace xllm

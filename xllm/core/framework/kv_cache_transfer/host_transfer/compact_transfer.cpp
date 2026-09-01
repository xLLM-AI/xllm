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

#include "framework/kv_cache_transfer/host_transfer/compact_transfer.h"

#include <glog/logging.h>

#include <cstdint>
#include <memory>
#include <utility>

#include "framework/kv_cache_transfer/host_transfer/compact_load_executor.h"
#include "framework/kv_cache_transfer/host_transfer/compact_offload_executor.h"
#include "platform/batch_memcpy.h"
#include "platform/device.h"
#include "platform/layer_synchronizer.h"

namespace xllm {

CompactHostKVTransfer::CompactHostKVTransfer(
    HostKVLayout layout,
    const Device& device,
    const Stream& compute_stream,
    uint32_t layer_copy_batches,
    std::unique_ptr<BatchMemcpy> batch_memcpy,
    CompactTransferConfig config)
    : HostKVTransfer(std::move(layout)),
      compute_stream_(compute_stream),
      batch_memcpy_(std::move(batch_memcpy)) {
  if (batch_memcpy_ == nullptr) {
    batch_memcpy_ = create_batch_memcpy(device);
  } else {
    batch_memcpy_->init(device.index());
  }
  CHECK(batch_memcpy_ != nullptr) << "Host KV batch memcpy is unavailable.";
  load_executor_ =
      std::make_unique<CompactLoadExecutor>(this->layout(),
                                            device,
                                            layer_copy_batches,
                                            *batch_memcpy_,
                                            config.load_target_bytes);
  offload_executor_ = std::make_unique<CompactOffloadExecutor>(
      this->layout(), device, *batch_memcpy_, config.offload_target_bytes);
}

CompactHostKVTransfer::~CompactHostKVTransfer() { drain(); }

HostKVLoadHandle CompactHostKVTransfer::prepare_load(bool draft) {
  const uint32_t base_event_count = load_executor_->event_count();
  const uint32_t event_count = base_event_count + static_cast<uint32_t>(draft);
  HostKVLoadHandle handle{create_layer_synchronizer(event_count),
                          load_executor_->layers_per_event()};
  if (draft) {
    handle.draft_event_index = base_event_count;
  }
  return handle;
}

void CompactHostKVTransfer::drain() {
  if (drained_) {
    return;
  }
  drained_ = true;
  offload_executor_->drain();
  load_executor_->drain();
}

uint32_t CompactHostKVTransfer::load_event_count() const {
  return load_executor_->event_count();
}

uint32_t CompactHostKVTransfer::layers_per_event() const {
  return load_executor_->layers_per_event();
}

bool CompactHostKVTransfer::load_impl(const HostKVRequest& request,
                                      const HostKVLoadHandle& handle) {
  return load_executor_->execute(request, handle.synchronizer);
}

bool CompactHostKVTransfer::offload_impl(const HostKVRequest& request) {
  offload_executor_->execute(request, compute_stream_);
  return true;
}

}  // namespace xllm

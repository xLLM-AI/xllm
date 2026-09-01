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

#include "framework/kv_cache_transfer/host_transfer/basic_transfer.h"

#include <glog/logging.h>

#include <algorithm>
#include <exception>
#include <memory>
#include <utility>
#include <vector>

#include "framework/kv_cache_transfer/host_transfer/transfer_utils.h"
#include "platform/batch_memcpy.h"
#include "platform/device.h"
#include "platform/layer_synchronizer.h"
#include "util/blockingconcurrentqueue.h"

namespace xllm {
namespace {

constexpr int32_t kStreamTimeoutMs = 60000;
constexpr size_t kCopyStreamCount = 6;

struct CopyPlan {
  std::vector<torch::Tensor> src_tensors;
  std::vector<torch::Tensor> dst_tensors;
};

using CopyStreamQueue =
    moodycamel::BlockingConcurrentQueue<std::unique_ptr<Stream>>;

class CopyStreamLease final {
 public:
  explicit CopyStreamLease(CopyStreamQueue* streams) : streams_(streams) {
    streams_->wait_dequeue(stream_);
  }

  ~CopyStreamLease() { streams_->enqueue(std::move(stream_)); }

  CopyStreamLease(const CopyStreamLease&) = delete;
  CopyStreamLease& operator=(const CopyStreamLease&) = delete;

  Stream* get() const { return stream_.get(); }

  void drain_or_die(const char* reason) const {
    try {
      const int32_t result = stream_->synchronize();
      if (result != 0) {
        LOG(FATAL) << "Failed to drain Host KV copy stream: reason=" << reason
                   << ", result=" << result;
      }
    } catch (const std::exception& error) {
      LOG(FATAL) << "Failed to drain Host KV copy stream: reason=" << reason
                 << ", error=" << error.what();
    }
  }

 private:
  CopyStreamQueue* streams_ = nullptr;
  std::unique_ptr<Stream> stream_;
};

CopyPlan build_plan(const HostKVLayout& layout,
                    const std::vector<HostKVMapping>& mappings,
                    const LayerRange& range,
                    bool is_load) {
  CopyPlan plan;
  const size_t estimated_copies =
      mappings.size() * static_cast<size_t>(range.end - range.begin);
  plan.src_tensors.reserve(estimated_copies);
  plan.dst_tensors.reserve(estimated_copies);
  for (const HostKVMapping& mapping : mappings) {
    const HostKVGroupLayout& group = layout.group(mapping.group_id);
    for (const HostKVLayerLayout& layer : group.layers) {
      if (layer.absolute_layer_id < range.begin ||
          layer.absolute_layer_id >= range.end) {
        continue;
      }
      for (const auto& [role, device_tensor] : layer.device_roles) {
        auto host_it = group.host_roles.find(role);
        if (host_it == group.host_roles.end()) {
          continue;
        }
        torch::Tensor host_block =
            host_it->second[mapping.host_block_id][layer.group_layer_slot];
        torch::Tensor device_block = device_tensor[mapping.device_block_id];
        if (is_load) {
          plan.src_tensors.emplace_back(host_block);
          plan.dst_tensors.emplace_back(device_block);
        } else {
          plan.src_tensors.emplace_back(device_block);
          plan.dst_tensors.emplace_back(host_block);
        }
      }
    }
  }
  return plan;
}

}  // namespace

class BasicHostKVTransfer::Impl final {
 public:
  Impl(const HostKVLayout& layout,
       const Device& device,
       const Stream& compute_stream,
       uint32_t layer_copy_batches,
       std::unique_ptr<BatchMemcpy> batch_memcpy)
      : layout_(layout),
        device_(device),
        compute_stream_(compute_stream),
        layers_per_event_(
            get_layers_per_event(layout.num_layers(), layer_copy_batches)),
        ranges_(build_layer_ranges(layout.num_layers(), layers_per_event_)),
        batch_memcpy_(std::move(batch_memcpy)) {
    if (batch_memcpy_ == nullptr) {
      batch_memcpy_ = create_batch_memcpy(device_);
    } else {
      batch_memcpy_->init(device_.index());
    }
    CHECK(batch_memcpy_ != nullptr) << "Host KV batch memcpy is unavailable.";
    for (size_t index = 0; index < kCopyStreamCount; ++index) {
      streams_.enqueue(device_.get_stream_from_pool(kStreamTimeoutMs));
    }
  }

  HostKVLoadHandle prepare_load(bool draft) const {
    const uint32_t count = event_count() + static_cast<uint32_t>(draft);
    HostKVLoadHandle handle{create_layer_synchronizer(count),
                            layers_per_event()};
    if (draft) {
      handle.draft_event_index = event_count();
    }
    return handle;
  }

  uint32_t event_count() const { return static_cast<uint32_t>(ranges_.size()); }

  uint32_t layers_per_event() const { return layers_per_event_; }

  bool load(const HostKVRequest& request, const HostKVLoadHandle& handle) {
    CopyStreamLease stream(&streams_);
    bool stream_has_work = false;
    for (size_t index = 0; index < ranges_.size(); ++index) {
      CopyPlan plan = build_plan(
          layout_, request.target_mappings, ranges_[index], /*is_load=*/true);
      if (!plan.src_tensors.empty()) {
        if (!batch_memcpy_->submit_h2d(
                plan.src_tensors, plan.dst_tensors, stream.get())) {
          return false;
        }
        stream_has_work = true;
      }
      if (!handle.synchronizer->record_stream(static_cast<int64_t>(index),
                                              stream.get())) {
        if (stream_has_work) {
          stream.drain_or_die("layer-ready event recording failed");
        }
        return false;
      }
    }
    for (const LayerRange& range : ranges_) {
      CopyPlan plan =
          build_plan(layout_, request.draft_mappings, range, /*is_load=*/true);
      if (!plan.src_tensors.empty() &&
          !batch_memcpy_->submit_h2d(
              plan.src_tensors, plan.dst_tensors, stream.get())) {
        if (stream_has_work) {
          stream.drain_or_die("draft cache copy submission failed");
        }
        return false;
      }
      stream_has_work = stream_has_work || !plan.src_tensors.empty();
    }
    if (!request.draft_mappings.empty() &&
        !handle.synchronizer->record_stream(
            static_cast<int64_t>(ranges_.size()), stream.get())) {
      if (stream_has_work) {
        stream.drain_or_die("draft cache completion event recording failed");
      }
      return false;
    }
    return true;
  }

  bool offload(const HostKVRequest& request) {
    CopyStreamLease stream(&streams_);
    stream.get()->wait_stream(compute_stream_);
    const auto offload_mappings =
        [this, &stream](const std::vector<HostKVMapping>& mappings) {
          for (const LayerRange& range : ranges_) {
            CopyPlan plan =
                build_plan(layout_, mappings, range, /*is_load=*/false);
            if (!plan.src_tensors.empty() &&
                !batch_memcpy_->copy_d2h(
                    plan.src_tensors, plan.dst_tensors, stream.get())) {
              return false;
            }
          }
          return true;
        };
    return offload_mappings(request.target_mappings) &&
           offload_mappings(request.draft_mappings);
  }

  void drain() {
    if (drained_) {
      return;
    }
    drained_ = true;
    std::unique_ptr<Stream> stream;
    while (streams_.try_dequeue(stream)) {
      if (stream != nullptr && stream->synchronize() != 0) {
        LOG(FATAL) << "Failed to drain Basic Host KV copy stream.";
      }
    }
  }

 private:
  const HostKVLayout& layout_;
  Device device_;
  const Stream& compute_stream_;
  uint32_t layers_per_event_ = 1;
  std::vector<LayerRange> ranges_;
  std::unique_ptr<BatchMemcpy> batch_memcpy_;
  CopyStreamQueue streams_;
  bool drained_ = false;
};

BasicHostKVTransfer::BasicHostKVTransfer(
    HostKVLayout layout,
    const Device& device,
    const Stream& compute_stream,
    uint32_t layer_copy_batches,
    std::unique_ptr<BatchMemcpy> batch_memcpy)
    : HostKVTransfer(std::move(layout)),
      impl_(std::make_unique<Impl>(this->layout(),
                                   device,
                                   compute_stream,
                                   layer_copy_batches,
                                   std::move(batch_memcpy))) {}

BasicHostKVTransfer::~BasicHostKVTransfer() { drain(); }

HostKVLoadHandle BasicHostKVTransfer::prepare_load(bool draft) {
  return impl_->prepare_load(draft);
}

void BasicHostKVTransfer::drain() { impl_->drain(); }

uint32_t BasicHostKVTransfer::load_event_count() const {
  return impl_->event_count();
}

uint32_t BasicHostKVTransfer::layers_per_event() const {
  return impl_->layers_per_event();
}

bool BasicHostKVTransfer::load_impl(const HostKVRequest& request,
                                    const HostKVLoadHandle& handle) {
  return impl_->load(request, handle);
}

bool BasicHostKVTransfer::offload_impl(const HostKVRequest& request) {
  return impl_->offload(request);
}

}  // namespace xllm

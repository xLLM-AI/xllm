/* Copyright 2025-2026 The xLLM Authors.

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

// VMMTorchAllocator is only available for platforms using PyTorch's
// CUDACachingAllocator interface (CUDA, ILU, ROCm).
#if defined(USE_CUDA) || defined(USE_ILU)
#if TORCH_VERSION_MAJOR >= 2 && TORCH_VERSION_MINOR >= 10
#include <ATen/cuda/MemPool.h>
#endif
#include <c10/cuda/CUDACachingAllocator.h>
#include <glog/logging.h>

#include "shared_vmm_allocator.h"

namespace xllm {

/**
 * @brief VMMTorchAllocator - A PyTorch-compatible allocator wrapper for
 * SharedVMMAllocator.
 *
 * Only raw_alloc/raw_alloc_with_stream/raw_delete are expected to be called.
 * Other methods are implemented to satisfy the interface but should not be
 * called at runtime.
 */
class VMMTorchAllocator
    : public c10::cuda::CUDACachingAllocator::CUDAAllocator {
 public:
  explicit VMMTorchAllocator(SharedVMMAllocator* vmm_allocator)
      : vmm_allocator_(vmm_allocator) {}

  ~VMMTorchAllocator() override = default;

  VMMTorchAllocator(const VMMTorchAllocator&) = delete;
  VMMTorchAllocator& operator=(const VMMTorchAllocator&) = delete;

  // ============== Core allocation methods (expected to be called)
  // ==============

  void* raw_alloc(size_t nbytes) override {
    void* ptr = vmm_allocator_->allocate(nbytes);
    total_allocated_ += nbytes;
    alloc_count_++;
    VLOG(10) << "VMMTorchAllocator::raw_alloc(" << nbytes << " bytes) -> "
             << ptr << ", total_allocated: " << total_allocated_
             << ", alloc_count: " << alloc_count_
             << ", current_offset: " << vmm_allocator_->current_offset()
             << ", high_water_mark: " << vmm_allocator_->high_water_mark();
    return ptr;
  }

  void* raw_alloc_with_stream(size_t nbytes, cudaStream_t /*stream*/) override {
    void* ptr = vmm_allocator_->allocate(nbytes);
    total_allocated_ += nbytes;
    alloc_count_++;
    VLOG(10) << "VMMTorchAllocator::raw_alloc_with_stream(" << nbytes
             << " bytes) -> " << ptr
             << ", total_allocated: " << total_allocated_
             << ", alloc_count: " << alloc_count_
             << ", current_offset: " << vmm_allocator_->current_offset()
             << ", high_water_mark: " << vmm_allocator_->high_water_mark();
    return ptr;
  }

  void raw_delete(void* ptr) override {
    VLOG(10) << "VMMTorchAllocator::raw_delete(" << ptr << ")";
    vmm_allocator_->deallocate(ptr);
  }

  // ============== c10::Allocator interface (should NOT be called)
  // ==============

  c10::DataPtr allocate(size_t n) override {
    LOG(FATAL) << "VMMTorchAllocator::allocate() called unexpectedly!";
    void* ptr = vmm_allocator_->allocate(n);
    return {ptr, ptr, &raw_deleter, c10::Device(c10::kCUDA)};
  }

  void copy_data(void* dest,
                 const void* src,
                 std::size_t count) const override {
    LOG(FATAL) << "VMMTorchAllocator::copy_data() called unexpectedly!";
    cudaMemcpy(dest, src, count, cudaMemcpyDefault);
  }

  // ============== CUDAAllocator interface (should NOT be called)
  // ==============

  void init(int /*device_count*/) override {
    LOG(FATAL) << "VMMTorchAllocator::init() called unexpectedly!";
  }

  bool initialized() override {
    LOG(FATAL) << "VMMTorchAllocator::initialized() called unexpectedly!";
    return vmm_allocator_->is_initialized();
  }

  std::string name() override { return "VMMTorchAllocator"; }

  c10::CachingDeviceAllocator::DeviceStats getDeviceStats(
      c10::DeviceIndex /*device*/) override {
    LOG(FATAL) << "VMMTorchAllocator::getDeviceStats() called unexpectedly!";
    return {};
  }

  void resetAccumulatedStats(c10::DeviceIndex /*device*/) override {
    LOG(FATAL)
        << "VMMTorchAllocator::resetAccumulatedStats() called unexpectedly!";
  }

  void resetPeakStats(c10::DeviceIndex /*device*/) override {
    LOG(FATAL) << "VMMTorchAllocator::resetPeakStats() called unexpectedly!";
  }

  double getMemoryFraction(c10::DeviceIndex /*device*/) override {
    LOG(FATAL) << "VMMTorchAllocator::getMemoryFraction() called unexpectedly!";
    return 1.0;
  }

  void setMemoryFraction(double /*fraction*/,
                         c10::DeviceIndex /*device*/) override {
    LOG(FATAL) << "VMMTorchAllocator::setMemoryFraction() called unexpectedly!";
  }

  void enable(bool /*value*/) override {
    LOG(FATAL) << "VMMTorchAllocator::enable() called unexpectedly!";
  }

  bool isEnabled() const override {
    LOG(FATAL) << "VMMTorchAllocator::isEnabled() called unexpectedly!";
    return true;
  }

  void cacheInfo(c10::DeviceIndex /*device*/, size_t* largestBlock) override {
    LOG(FATAL) << "VMMTorchAllocator::cacheInfo() called unexpectedly!";
    if (largestBlock) {
      *largestBlock =
          vmm_allocator_->reserved_size() - vmm_allocator_->current_offset();
    }
  }

  void* getBaseAllocation(void* ptr, size_t* size) override {
    LOG(FATAL) << "VMMTorchAllocator::getBaseAllocation() called unexpectedly!";
    if (size) {
      *size = vmm_allocator_->mapped_size();
    }
    return ptr;
  }

  void recordStream(const c10::DataPtr& /*ptr*/,
                    c10::cuda::CUDAStream /*stream*/) override {
    LOG(FATAL) << "VMMTorchAllocator::recordStream() called unexpectedly!";
  }

  c10::cuda::CUDACachingAllocator::ShareableHandle shareIpcHandle(
      void* /*ptr*/) override {
    LOG(ERROR) << "VMMTorchAllocator::shareIpcHandle() called - not supported!";
    TORCH_CHECK(false, name(), " does not support IPC");
    return {};
  }

  std::shared_ptr<void> getIpcDevPtr(std::string /*handle*/) override {
    LOG(ERROR) << "VMMTorchAllocator::getIpcDevPtr() called - not supported!";
    TORCH_CHECK(false, name(), " does not support IPC");
    return nullptr;
  }

  void attachOutOfMemoryObserver(
      c10::cuda::CUDACachingAllocator::OutOfMemoryObserver /*observer*/)
      override {
    LOG(FATAL) << "VMMTorchAllocator::attachOutOfMemoryObserver() called "
                  "unexpectedly!";
  }

  void attachAllocatorTraceTracker(
      c10::cuda::CUDACachingAllocator::AllocatorTraceTracker /*tracker*/)
      override {
    LOG(FATAL) << "VMMTorchAllocator::attachAllocatorTraceTracker() called "
                  "unexpectedly!";
  }

  void enablePeerAccess(c10::DeviceIndex /*dev*/,
                        c10::DeviceIndex /*dev_to_access*/) override {
    LOG(FATAL) << "VMMTorchAllocator::enablePeerAccess() called unexpectedly!";
  }

  cudaError_t memcpyAsync(void* dst,
                          int /*dstDevice*/,
                          const void* src,
                          int /*srcDevice*/,
                          size_t count,
                          cudaStream_t stream,
                          bool /*p2p_enabled*/) override {
    LOG(FATAL) << "VMMTorchAllocator::memcpyAsync() called unexpectedly!";
    return cudaMemcpyAsync(dst, src, count, cudaMemcpyDefault, stream);
  }

  c10::cuda::CUDACachingAllocator::CheckpointDelta setCheckpointPoolState(
      c10::DeviceIndex /*device*/,
      std::shared_ptr<c10::cuda::CUDACachingAllocator::AllocatorState> /*pps*/)
      override {
    LOG(ERROR) << "VMMTorchAllocator::setCheckpointPoolState() called - not "
                  "supported!";
    TORCH_CHECK(false, name(), " does not support checkpointing");
    return {};
  }

  void beginAllocateToPool(
      c10::DeviceIndex /*device*/,
      at::cuda::MempoolId_t /*mempool_id*/,
      std::function<bool(cudaStream_t)> /*filter*/) override {
    LOG(FATAL)
        << "VMMTorchAllocator::beginAllocateToPool() called unexpectedly!";
  }

  void endAllocateToPool(c10::DeviceIndex /*device*/,
                         at::cuda::MempoolId_t /*mempool_id*/) override {
    LOG(FATAL) << "VMMTorchAllocator::endAllocateToPool() called unexpectedly!";
  }

  void releasePool(c10::DeviceIndex /*device*/,
                   at::cuda::MempoolId_t /*mempool_id*/) override {
    LOG(FATAL) << "VMMTorchAllocator::releasePool() called unexpectedly!";
  }

  std::shared_ptr<c10::cuda::CUDACachingAllocator::AllocatorState>
  getCheckpointState(c10::DeviceIndex /*device*/,
                     at::cuda::MempoolId_t /*id*/) override {
    LOG(FATAL)
        << "VMMTorchAllocator::getCheckpointState() called unexpectedly!";
    return nullptr;
  }

#if (TORCH_VERSION_MAJOR >= 2 && TORCH_VERSION_MINOR >= 10) || defined(USE_MACA)
  void emptyCache(at::cuda::MempoolId_t /*mempool_id*/ = {0, 0}) override {
    LOG(FATAL) << "VMMTorchAllocator::emptyCache() called unexpectedly!";
  }

#if TORCH_VERSION_MAJOR >= 2 && TORCH_VERSION_MINOR >= 10
  std::vector<c10::cuda::CUDACachingAllocator::StreamSegmentSize>
  getExpandableSegmentSizes(c10::DeviceIndex /*device*/) override {
    LOG(FATAL) << "VMMTorchAllocator::getExpandableSegmentSizes() called "
                  "unexpectedly!";
    return {};
  }
#endif  // TORCH_VERSION_MAJOR >= 2 && TORCH_VERSION_MINOR >= 10

  c10::cuda::CUDACachingAllocator::SnapshotInfo snapshot(
      at::cuda::MempoolId_t /*mempool_id*/ = {0, 0}) override {
    LOG(FATAL) << "VMMTorchAllocator::snapshot() called unexpectedly!";
    return {};
  }

  void recordHistory(
      bool /*enabled*/,
      c10::cuda::CUDACachingAllocator::CreateContextFn /*context_recorder*/,
      size_t /*alloc_trace_max_entries*/,
      c10::cuda::CUDACachingAllocator::RecordContext /*when*/,
      bool /*clearHistory*/) override {
    LOG(FATAL) << "VMMTorchAllocator::recordHistory() called unexpectedly!";
  }
#else
  void emptyCache() override {
    LOG(FATAL) << "VMMTorchAllocator::emptyCache() called unexpectedly!";
  }

  c10::cuda::CUDACachingAllocator::SnapshotInfo snapshot() override {
    LOG(FATAL) << "VMMTorchAllocator::snapshot() called unexpectedly!";
    return {};
  }

  void recordHistory(
      bool /*enabled*/,
      c10::cuda::CUDACachingAllocator::CreateContextFn /*context_recorder*/,
      size_t /*alloc_trace_max_entries*/,
      c10::cuda::CUDACachingAllocator::RecordContext /*when*/) override {
    LOG(FATAL) << "VMMTorchAllocator::recordHistory() called unexpectedly!";
  }
#endif

 private:
  static void raw_deleter(void* ptr) {
    // No-op: VMM memory is not freed individually
    (void)ptr;
  }

  SharedVMMAllocator* vmm_allocator_;
  size_t total_allocated_ = 0;  // Total bytes allocated (for logging)
  size_t alloc_count_ = 0;      // Number of allocations (for logging)
};

}  // namespace xllm

#endif  // USE_CUDA || USE_ILU

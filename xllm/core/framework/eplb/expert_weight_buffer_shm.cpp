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

#include "expert_weight_buffer_shm.h"

#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <mutex>
#include <sstream>
#include <string_view>
#include <thread>

namespace xllm {
namespace {

constexpr size_t kTensorAlignment = 64;

uint64_t stable_namespace_hash(std::string_view value) {
  constexpr uint64_t kFnvOffsetBasis = 14695981039346656037ULL;
  constexpr uint64_t kFnvPrime = 1099511628211ULL;
  uint64_t hash = kFnvOffsetBasis;
  for (char byte : value) {
    hash ^= static_cast<uint8_t>(byte);
    hash *= kFnvPrime;
  }
  return hash;
}

std::string make_expert_shm_name(const std::string& service_namespace,
                                 int32_t expert_id) {
  CHECK(!service_namespace.empty())
      << "Expert shared memory requires a service namespace.";
  std::ostringstream name;
  name << "xllm_expert_" << std::hex << stable_namespace_hash(service_namespace)
       << std::dec << "_" << expert_id;
  return name.str();
}

size_t align_tensor_size(size_t size) {
  return (size + kTensorAlignment - 1) & ~(kTensorAlignment - 1);
}

int64_t aligned_layer_region_size(int64_t total_size, int32_t max_layers) {
  if (total_size <= 0 || max_layers <= 0) {
    return 0;
  }
  const int64_t unaligned_size = total_size / max_layers;
  return unaligned_size -
         unaligned_size % static_cast<int64_t>(kTensorAlignment);
}

void lock_shared_mutex(pthread_mutex_t& mutex, int32_t expert_id) {
  const int32_t result = pthread_mutex_lock(&mutex);
  if (result == EOWNERDEAD) {
    CHECK_EQ(pthread_mutex_consistent(&mutex), 0)
        << "Failed to recover shared mutex for expert " << expert_id;
    LOG(WARNING) << "Recovered orphaned shared mutex for expert " << expert_id;
    return;
  }
  CHECK_EQ(result, 0) << "Failed to acquire shared mutex for expert "
                      << expert_id << ": " << std::strerror(result);
}

void unlock_shared_mutex(pthread_mutex_t& mutex, int32_t expert_id) {
  const int32_t result = pthread_mutex_unlock(&mutex);
  CHECK_EQ(result, 0) << "Failed to release shared mutex for expert "
                      << expert_id << ": " << std::strerror(result);
}

class SharedMutexGuard final {
 public:
  SharedMutexGuard(pthread_mutex_t& mutex, int32_t expert_id)
      : mutex_(mutex), expert_id_(expert_id) {
    lock_shared_mutex(mutex_, expert_id_);
  }

  ~SharedMutexGuard() { unlock_shared_mutex(mutex_, expert_id_); }

  SharedMutexGuard(const SharedMutexGuard&) = delete;
  SharedMutexGuard& operator=(const SharedMutexGuard&) = delete;

 private:
  pthread_mutex_t& mutex_;
  int32_t expert_id_;
};

}  // namespace

ExpertBufferShm::ExpertBufferShm(const std::string& service_namespace,
                                 int32_t expert_id,
                                 int32_t max_layers,
                                 int64_t total_size)
    : shm_name_(make_expert_shm_name(service_namespace, expert_id)),
      expert_id_(expert_id),
      max_layers_(max_layers),
      data_region_size_(total_size),
      layer_data_region_size_(
          aligned_layer_region_size(total_size, max_layers)) {
  CHECK_GT(max_layers_, 0) << "Expert shared memory requires layers.";
  CHECK_LE(max_layers_, MAX_LAYERS_PER_EXPERT)
      << "Expert shared memory layer count exceeds supported maximum.";
  CHECK_GT(data_region_size_, 0)
      << "Expert shared memory data region must be positive.";
  CHECK_GT(layer_data_region_size_, 0)
      << "Expert shared memory requires at least " << kTensorAlignment
      << " aligned bytes per layer.";
  // Memory alignment calculation (64-byte alignment for performance)
  constexpr size_t kAlignment = 64;

  // Calculate aligned header size (header + padding)
  size_t header_size =
      ((sizeof(SharedHeader) + kAlignment - 1) / kAlignment) * kAlignment;

  // Calculate aligned metadata region size (all experts' metadata + padding)
  size_t meta_size = ((max_layers * MAX_TENSORS_PER_LAYER * sizeof(TensorMeta) +
                       kAlignment - 1) /
                      kAlignment) *
                     kAlignment;

  bool is_creator;
  // Create/attach shared memory segment with calculated size
  shm_ = std::make_shared<SharedMemoryManager>(
      shm_name_, header_size + meta_size + total_size, is_creator);

  // Memory region pointers setup:
  header_ = static_cast<SharedHeader*>(shm_->base_address());
  tensor_metas_ = reinterpret_cast<TensorMeta*>(
      static_cast<char*>(shm_->base_address()) + header_size);
  data_base_ =
      static_cast<char*>(shm_->base_address()) + header_size + meta_size;

  if (is_creator) {
    initialize_as_creator();
  }
  verify_and_recover();
  SharedMutexGuard lock(header_->allocation_mutex, expert_id_);
  rebuild_name_to_slot();
}

ExpertBufferShm::~ExpertBufferShm() {
  std::lock_guard<std::mutex> lock(local_mutex_);
  std::atomic_thread_fence(std::memory_order_seq_cst);
  header_ = nullptr;
  tensor_metas_ = nullptr;
  data_base_ = nullptr;
}

void ExpertBufferShm::initialize_as_creator() {
  header_->layout_version = kExpertShmLayoutVersion;
  header_->reserved = 0;

  pthread_mutexattr_t attr;
  pthread_mutexattr_init(&attr);
  pthread_mutexattr_setpshared(&attr, PTHREAD_PROCESS_SHARED);
  pthread_mutexattr_setrobust(&attr, PTHREAD_MUTEX_ROBUST);

  if (pthread_mutex_init(&header_->allocation_mutex, &attr) != 0) {
    pthread_mutexattr_destroy(&attr);
    LOG(FATAL) << "Mutex initialization failed.";
  }
  pthread_mutexattr_destroy(&attr);

  memset(tensor_metas_,
         0,
         max_layers_ * MAX_TENSORS_PER_LAYER * sizeof(TensorMeta));

  // Publish the initialized process-shared mutex and metadata as one release.
  // Attachers wait for this field before touching allocation_mutex.
  std::atomic_ref<uint64_t>(header_->magic)
      .store(kExpertShmMagic, std::memory_order_release);
}

void ExpertBufferShm::verify_and_recover() {
  // Reject segments that were created by a build with a different on-disk
  // layout (e.g. a mismatched SharedHeader / TensorMeta shape or a stale
  // segment left over from a previous xLLM version). Silent continuation
  // would let us read tensor bytes from arbitrary offsets and corrupt
  // weights across ranks.
  std::atomic_ref<uint64_t> magic(header_->magic);
  constexpr int32_t kInitializationRetries = 5000;
  uint64_t observed_magic = magic.load(std::memory_order_acquire);
  for (int32_t retry = 0; observed_magic == 0 && retry < kInitializationRetries;
       ++retry) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    observed_magic = magic.load(std::memory_order_acquire);
  }
  CHECK_EQ(observed_magic, kExpertShmMagic)
      << "Shared memory header magic mismatch for expert " << expert_id_
      << " (expected 0x" << std::hex << kExpertShmMagic << ", got 0x"
      << observed_magic << std::dec
      << "). This usually means a stale segment left over from a previous "
      << "build; remove /dev/shm/" << shm_name_ << " and retry.";
  CHECK_EQ(header_->layout_version, kExpertShmLayoutVersion)
      << "Shared memory layout version mismatch for expert " << expert_id_
      << " (expected " << kExpertShmLayoutVersion << ", got "
      << header_->layout_version
      << "). A stale segment from an incompatible xLLM version is attached.";

  SharedMutexGuard lock(header_->allocation_mutex, expert_id_);
}

void ExpertBufferShm::rebuild_name_to_slot() {
  // Reconstruct the per-layer name -> slot cache from what is already in
  // shared memory. Runs once at attach time so an existing segment (creator
  // or reader) has its lookup table populated before add/get touches it.
  name_to_slot_.assign(max_layers_, std::unordered_map<std::string, int32_t>());
  for (int32_t layer = 0; layer < max_layers_; ++layer) {
    rebuild_layer_index(layer);
  }
}

int32_t ExpertBufferShm::rebuild_layer_index(int32_t layer_id) {
  TensorMeta* layer_metas = &tensor_metas_[layer_id * MAX_TENSORS_PER_LAYER];
  auto& layer_index = name_to_slot_[layer_id];
  layer_index.clear();
  layer_index.reserve(MAX_TENSORS_PER_LAYER);
  int32_t available_slot = -1;
  for (int32_t slot = 0; slot < MAX_TENSORS_PER_LAYER; ++slot) {
    TensorMeta& meta = layer_metas[slot];
    const uint32_t publication_state =
        std::atomic_ref<uint32_t>(meta.publication_state)
            .load(std::memory_order_acquire);
    if (publication_state == kTensorMetaEmpty && available_slot < 0) {
      available_slot = slot;
    }
    if (publication_state != kTensorMetaPublished) {
      continue;
    }
    const size_t name_length =
        strnlen(meta.tensor_name, sizeof(meta.tensor_name));
    CHECK_LT(name_length, sizeof(meta.tensor_name))
        << "Corrupted tensor name for expert " << expert_id_ << " layer "
        << layer_id << " slot " << slot;
    layer_index.emplace(std::string(meta.tensor_name, name_length), slot);
  }
  return available_slot;
}

size_t ExpertBufferShm::get_layer_offset(int32_t layer_id) const {
  CHECK(layer_id >= 0 && layer_id < max_layers_)
      << "Invalid layer ID: " << layer_id << " for expert " << expert_id_
      << " (max_layers=" << max_layers_ << ")";
  return static_cast<size_t>(layer_id) *
         static_cast<size_t>(layer_data_region_size_);
}

void ExpertBufferShm::add_tensor(int32_t layer_id,
                                 const std::string& tensor_name,
                                 const torch::Tensor& tensor) {
  CHECK(layer_id >= 0 && layer_id < max_layers_)
      << "Invalid layer ID: " << layer_id << " for expert " << expert_id_
      << " (max_layers=" << max_layers_ << ")";
  CHECK(!tensor_name.empty()) << "Tensor name cannot be empty";
  CHECK_LT(tensor_name.size(), sizeof(TensorMeta::tensor_name))
      << "Tensor name exceeds shared metadata capacity";
  CHECK(tensor.defined() && tensor.is_contiguous())
      << "Tensor must be defined and contiguous";
  CHECK_GT(tensor.numel(), 0) << "Tensor must not be empty";
  CHECK(tensor.device().type() == torch::kCPU)
      << "Only CPU tensors can be stored in shared memory";
  CHECK_LE(tensor.dim(), 8) << "Tensor rank exceeds metadata capacity";

  std::lock_guard<std::mutex> lock(local_mutex_);
  SharedMutexGuard shared_lock(header_->allocation_mutex, expert_id_);

  // Get this expert's metadata block
  TensorMeta* layer_metas = &tensor_metas_[layer_id * MAX_TENSORS_PER_LAYER];
  const int32_t available_slot = rebuild_layer_index(layer_id);
  auto& layer_index = name_to_slot_[layer_id];

  // Duplicate check via the O(1) hash lookup instead of scanning every slot.
  CHECK(layer_index.find(tensor_name) == layer_index.end())
      << "Tensor '" << tensor_name << "' already exists for expert "
      << expert_id_ << " layer " << layer_id;

  CHECK_GE(available_slot, 0) << "No available slots for expert " << expert_id_
                              << " layer " << layer_id;

  // Prepare unpublished tensor metadata. tensor_name remains empty until all
  // metadata and bytes are complete, so attachers never discover a partial
  // tensor even if the writer exits between these steps.
  TensorMeta& meta = layer_metas[available_slot];
  std::memset(&meta, 0, sizeof(meta));
  meta.rank = static_cast<int32_t>(tensor.dim());
  for (int32_t i = 0; i < meta.rank; ++i) {
    meta.shape[i] = tensor.size(i);
  }
  meta.dtype = static_cast<int32_t>(tensor.scalar_type());

  const size_t raw_size = tensor.nbytes();
  const size_t aligned_size = align_tensor_size(raw_size);

  // Calculate offset by summing sizes of previous tensors in this expert
  size_t layer_data_offset = 0;
  for (int32_t i = 0; i < MAX_TENSORS_PER_LAYER; ++i) {
    if (&layer_metas[i] == &meta) {
      break;
    }
    const uint32_t publication_state =
        std::atomic_ref<uint32_t>(layer_metas[i].publication_state)
            .load(std::memory_order_acquire);
    if (publication_state == kTensorMetaPublished) {
      layer_data_offset += align_tensor_size(layer_metas[i].actual_size);
    }
  }

  CHECK_LE(layer_data_offset, static_cast<size_t>(layer_data_region_size_));
  CHECK_LE(aligned_size,
           static_cast<size_t>(layer_data_region_size_) - layer_data_offset)
      << "Insufficient space in expert " << expert_id_ << " layer " << layer_id
      << " (needs " << aligned_size << " bytes, has "
      << (layer_data_region_size_ - layer_data_offset) << " remaining)";

  // Set final storage location
  meta.data_offset = get_layer_offset(layer_id) + layer_data_offset;
  meta.actual_size = raw_size;

  // Copy tensor data to shared memory
  void* dest = data_base_ + meta.data_offset;
  memcpy(dest, tensor.data_ptr(), raw_size);

  // Zero-fill any alignment padding
  if (aligned_size > raw_size) {
    memset(static_cast<char*>(dest) + raw_size, 0, aligned_size - raw_size);
  }

  std::memcpy(meta.tensor_name, tensor_name.c_str(), tensor_name.size() + 1);
  std::atomic_ref<uint32_t>(meta.publication_state)
      .store(kTensorMetaPublished, std::memory_order_release);
  layer_index.emplace(tensor_name, available_slot);
}

torch::Tensor ExpertBufferShm::get_tensor(int32_t layer_id,
                                          const std::string& tensor_name) {
  CHECK(layer_id >= 0 && layer_id < max_layers_)
      << "Invalid layer ID: " << layer_id << " for expert " << expert_id_
      << " (max_layers=" << max_layers_ << ")";

  // Validate expert ID
  std::lock_guard<std::mutex> lock(local_mutex_);
  SharedMutexGuard shared_lock(header_->allocation_mutex, expert_id_);

  // Get this expert's metadata block
  TensorMeta* layer_metas = &tensor_metas_[layer_id * MAX_TENSORS_PER_LAYER];

  // O(1) lookup instead of scanning every slot in the layer.
  auto it = name_to_slot_[layer_id].find(tensor_name);
  if (it == name_to_slot_[layer_id].end()) {
    rebuild_layer_index(layer_id);
    it = name_to_slot_[layer_id].find(tensor_name);
  }
  CHECK(it != name_to_slot_[layer_id].end())
      << "Tensor " << tensor_name << " not found in expert " << expert_id_
      << " layer " << layer_id;
  TensorMeta& meta = layer_metas[it->second];

  const size_t data_region_size = static_cast<size_t>(data_region_size_);
  const size_t layer_begin = get_layer_offset(layer_id);
  const size_t layer_capacity = static_cast<size_t>(layer_data_region_size_);
  CHECK_LE(layer_begin, data_region_size);
  CHECK_LE(layer_capacity, data_region_size - layer_begin);
  const size_t layer_end = layer_begin + layer_capacity;
  CHECK_GT(meta.actual_size, 0)
      << "Corrupted empty tensor metadata for " << tensor_name;
  CHECK_GE(meta.data_offset, layer_begin)
      << "Tensor metadata points before its layer region for " << tensor_name;
  CHECK_LE(meta.data_offset, layer_end)
      << "Tensor metadata points beyond its layer region for " << tensor_name;
  CHECK_LE(meta.actual_size, layer_end - meta.data_offset)
      << "Tensor metadata exceeds its layer region for " << tensor_name;
  CHECK_LE(meta.data_offset, data_region_size);
  CHECK_LE(meta.actual_size, data_region_size - meta.data_offset);
  CHECK_GE(meta.rank, 0);
  CHECK_LE(meta.rank, 8);

  // Create tensor options from stored type
  auto options = torch::TensorOptions()
                     .dtype(static_cast<torch::ScalarType>(meta.dtype))
                     .device(torch::kCPU)
                     .requires_grad(false);

  // Convert shape array to vector
  std::vector<int64_t> shape(meta.shape, meta.shape + meta.rank);

  // Keep the mapping alive for as long as the tensor storage can be accessed.
  void* src = data_base_ + meta.data_offset;
  torch::Tensor tensor =
      torch::from_blob(src, shape, [shared_memory = shm_](void*) {}, options);
  return tensor;
}

}  // namespace xllm

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

#include <fcntl.h>
#include <pthread.h>
#include <torch/torch.h>
#include <unistd.h>

#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "core/util/shared_memory_manager.h"

namespace xllm {

// Maximum number of tensors each expert-layer pair can store
constexpr int32_t MAX_TENSORS_PER_LAYER = 16;
// Maximum number of layers per expert
constexpr int32_t MAX_LAYERS_PER_EXPERT = 128;

// Magic and layout version stamped into the shared-memory header when it is
// created. Any process that attaches to an existing segment MUST see the same
// magic and version, otherwise the layout has changed (binary upgrade with an
// incompatible on-disk shape) and continuing would corrupt weights. When they
// differ we abort loudly instead of silently reading garbage.
constexpr uint64_t kExpertShmMagic = 0x584C4C4D5F455850ULL;  // "XLLM_EXP"
constexpr uint32_t kExpertShmLayoutVersion = 2;
constexpr uint32_t kTensorMetaEmpty = 0;
constexpr uint32_t kTensorMetaPublished = 1;

// Shared memory header structure containing control information
struct SharedHeader {
  uint64_t magic;                    // Must equal kExpertShmMagic
  uint32_t layout_version;           // Must equal kExpertShmLayoutVersion
  uint32_t reserved;                 // Padding for alignment, keep zero
  pthread_mutex_t allocation_mutex;  // Cross-process synchronization mutex
};

// Metadata structure for each stored tensor
struct TensorMeta {
  uint32_t publication_state;  // Atomically published after data is complete
  char tensor_name[256];       // Null-terminated tensor identifier
  int32_t rank;                // Number of dimensions (1D, 2D, etc.)
  int64_t shape[8];            // Dimensions of the tensor (max 8D)
  int32_t dtype;               // Data type (matches torch::Dtype)
  size_t data_offset;          // Byte offset in shared memory
  size_t actual_size;          // Unpadded data size in bytes
};

class ExpertBufferShm {
 public:
  ExpertBufferShm(const std::string& service_namespace,
                  int32_t expert_id,
                  int32_t max_layers,
                  int64_t total_size);

  virtual ~ExpertBufferShm();

  void add_tensor(int32_t layer_id,
                  const std::string& tensor_name,
                  const torch::Tensor& tensor);

  torch::Tensor get_tensor(int32_t layer_id, const std::string& tensor_name);

 private:
  // Initializes shared memory when creating new region
  void initialize_as_creator();

  // Verifies and recovers shared memory state
  void verify_and_recover();

  // Rebuild the per-layer name -> slot cache from tensor_metas_ after attach.
  void rebuild_name_to_slot();

  int32_t rebuild_layer_index(int32_t layer_id);

  // Calculates base offset for a layer's data region
  size_t get_layer_offset(int32_t layer_id) const;

  std::mutex local_mutex_;                    // Thread synchronization
  std::shared_ptr<SharedMemoryManager> shm_;  // Retained by returned tensors
  SharedHeader* header_ = nullptr;            // Pointer to shared header
  TensorMeta* tensor_metas_ = nullptr;        // Array of all layers' metadata
  char* data_base_ = nullptr;                 // Base pointer to data region

  // Per-layer tensor-name -> slot index lookup, rebuilt from tensor_metas_
  // when the segment is attached and mutated in lock-step with add_tensor.
  // Replaces the previous O(MAX_TENSORS_PER_LAYER) linear scan in add/get
  // with an amortized O(1) hash lookup. This map lives in the attaching
  // process only — the shared segment still stores metadata in the same
  // slot-array layout, so on-disk compatibility is unchanged.
  std::vector<std::unordered_map<std::string, int32_t>> name_to_slot_;

  const std::string shm_name_;
  const int32_t expert_id_;               // Expert identifier
  const int32_t max_layers_;              // Maximum supported layers
  const int64_t data_region_size_;        // Total bytes in the data region
  const int64_t layer_data_region_size_;  // Bytes allocated per layer
};

}  // namespace xllm

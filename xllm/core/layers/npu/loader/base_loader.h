/* Copyright 2025 The xLLM Authors. All Rights Reserved.

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

#include <absl/strings/match.h>
#include <acl/acl.h>
#include <torch/torch.h>

#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

#include "framework/eplb/expert_buffer_manager.h"
#include "framework/kv_cache/kv_cache.h"
#include "framework/model/model_input_params.h"
#include "framework/model_context.h"
#include "framework/state_dict/state_dict.h"
#include "xllm_atb_layers/pytorch/atb_torch/core/include/base_operation.h"
#include "xllm_atb_layers/pytorch/atb_torch/core/include/graph_operation.h"

namespace xllm {
namespace layer {

class RollingWeightBuffer;

// Selects how a loader stages weights before they become device-resident.
// kEager  : state_dict is written directly into `at_weight_tensors_` on NPU
//           and format conversions (e.g. NZ) are done in-place.
// kManual : state_dict is first staged into `at_host_weight_tensors_` on CPU;
//           BaseLoader then packs the host tensors into a contiguous pinned
//           buffer, copies it to device storage (or a pre-allocated slot from
//           a RollingWeightBuffer) and rebuilds `at_weight_tensors_` views.
enum class LoadMode {
  kEager,
  kManual,
};

class BaseLoader {
 public:
  BaseLoader(uint64_t weight_count,
             const ModelContext& context,
             LoadMode mode = LoadMode::kEager);
  virtual ~BaseLoader();

  virtual void load_state_dict(const StateDict& state_dict) {};
  virtual void verify_loaded_weights() const {};
  virtual void verify_loaded_weights(const std::string& prefix) const {};

  // Default merge path:
  //   kEager  : subclass overrides this method (no-op here).
  //   kManual : runs `merge_host_at_weights` + slice init + H2D copy + device
  //             view rebuild.
  virtual void merge_loaded_weights();

  // Manual-mode only: stage weights into the pinned host buffer, do not touch
  // device storage. Asserts in eager mode.
  virtual void merge_and_move_pinned_host();

  virtual void resize_experts_weights(int num_of_device_experts) {};

  // Manual-mode: release the pinned host buffer. Harmless in eager mode.
  virtual void free_weights();

  // Manual-mode: re-allocate device storage, async H2D copy, rebuild views.
  virtual void reload_weights();

  // Manual-mode P2P path: device buffer already filled (e.g. by xtensor
  // allocator or rolling load), just rebuild views.
  virtual void reload_weights_from_device();

  // Rolling-load path: refresh device slot pointer from rolling buffer and
  // rebuild AT tensor views from latest device base.
  virtual void refresh_rolling_weights();

  torch::Dtype string2dtype(const std::string& dtype_str);

  void correct_tensor_dtype(torch::Tensor& tensor,
                            const std::string& tensorName);

  std::vector<at::Tensor>& get_at_weight_tensors() {
    return at_weight_tensors_;
  }

  std::vector<at::Tensor>& get_at_host_weight_tensors() {
    return at_host_weight_tensors_;
  }

  std::unordered_map<std::string, std::vector<torch::Tensor>>&
  get_experts_weight_tensors() {
    return experts_weights_;
  }

  std::unique_ptr<ExpertBufferManager>& get_expert_shared_buffer() {
    return shared_buffer_;
  }

  std::vector<int32_t>& get_device_expert_list() { return device_expert_list_; }

  atb_torch::TorchTensorMap& get_weights_map() { return weights_map_; }

  LoadMode mode() const { return mode_; }

  // Manual-mode helpers exposed publicly for RollingLoadManager.
  void* get_host_pinned_storage() const { return host_pinned_storage_; }
  uint64_t get_storage_size() const { return storage_size_; }
  void set_rolling_buffer(std::shared_ptr<RollingWeightBuffer> buf,
                          int32_t layer_index);
  void allocate_device_storage();

  // Manual-mode pipeline. Kept public for RollingLoadManager and legacy
  // downstream call sites; internally a no-op when mode_ == kEager.
  virtual void copy_weights_to_pinned_host();
  virtual void copy_weights_to_device();
  virtual void copy_weights_to_device_async();
  virtual void copy_weights_to_device_async(aclrtStream stream);
  virtual void init_device_at_weights();
  virtual void init_weight_slices();

 protected:
  struct WeightSlice {
    uint64_t offset = 0;
    uint64_t bytes = 0;
    std::vector<int64_t> sizes;
    torch::ScalarType dtype = torch::kFloat16;
  };

  // -------------------- uniform weight staging helpers --------------------
  // The working vector / target device / zero-fill / NZ-format policy differ
  // between eager and manual modes. Subclasses should prefer these helpers
  // over touching `at_weight_tensors_` / `at_host_weight_tensors_` directly
  // so that a single merge body works for both modes.
  std::vector<at::Tensor>& working_tensors() {
    return mode_ == LoadMode::kManual ? at_host_weight_tensors_
                                      : at_weight_tensors_;
  }
  const std::vector<at::Tensor>& working_tensors() const {
    return mode_ == LoadMode::kManual ? at_host_weight_tensors_
                                      : at_weight_tensors_;
  }
  at::Device target_device() const {
    return mode_ == LoadMode::kManual ? at::Device(at::kCPU) : device_;
  }
  bool load_to_host() const { return mode_ == LoadMode::kManual; }

  // Return a {1}-shaped zeros tensor with the same dtype as the working slot
  // at `idx`, placed on the correct target device for the current mode.
  at::Tensor zero_like_working(int idx) const;

  // Eager mode  : returns `at_npu::native::npu_format_cast(t.contiguous(),
  //               ACL_FORMAT_FRACTAL_NZ)`.
  // Manual mode : records `idx` in `nz_indices_` so that the later H2D copy
  //               converts this slice via `copy_host_nd_to_nz`; returns the
  //               tensor as contiguous ND for staging into the pinned host
  //               buffer.
  at::Tensor cast_nz(at::Tensor t, int idx);

  // Subclass hook executed by `merge_loaded_weights` /
  // `merge_and_move_pinned_host` to do model-specific tensor combining
  // (e.g. QKV cat, MLP cat, transpose) on `working_tensors()`.
  virtual void merge_host_at_weights() {}

  // Default impl consults `nz_indices_`; subclasses may override to return a
  // static policy (kept for backward compatibility with legacy loaders).
  virtual bool is_nz_format_tensor(int weight_index);

  // -------------------- manual-mode pipeline (ex BaseManualLoader) --------
  // (declarations moved to public section above for legacy call sites.)

  int copy_host_nd_to_nz(torch::Tensor host_tensor,
                         void* dst_ptr,
                         uint64_t len,
                         aclrtMemcpyKind kind = ACL_MEMCPY_DEVICE_TO_DEVICE);
  torch::Tensor convert_to_torch_tensor(const std::vector<int64_t>& dims,
                                        const torch::ScalarType dtype,
                                        const uintptr_t& dev_addr,
                                        int acl_format = ACL_FORMAT_ND);

  void release_device_storage();
  void release_host_storage();

  // -------------------- legacy set_weight helpers -------------------------
  void set_weight(const StateDict& state_dict,
                  const std::string& tensor_name,
                  int weight_position,
                  bool to_host = false);

  void set_weight(const StateDict& state_dict,
                  const std::string& tensor_name,
                  int weight_position,
                  int dim,
                  bool to_host = false);

  void set_weight(const StateDict& state_dict,
                  const std::string& tensor_name,
                  int weight_position,
                  int dim,
                  int rank,
                  int world_size,
                  bool to_host = false);

  void set_weight_with_padding(const StateDict& state_dict,
                               const std::string& tensor_name,
                               int weight_position,
                               int dim,
                               int64_t padded_vocab_size,
                               bool to_host = false);

  void set_weight_with_padding(const StateDict& state_dict,
                               const std::string& tensor_name,
                               int weight_position,
                               int dim,
                               int rank,
                               int world_size,
                               int64_t padded_vocab_size,
                               bool to_host = false);

  at::Tensor pad_vocab_tensor(const at::Tensor& tensor,
                              int64_t padded_vocab_size) const;

  at::Tensor shard_padded_tensor(const at::Tensor& padded_tensor,
                                 int dim,
                                 int rank,
                                 int world_size) const;

  int64_t get_padded_vocab_size(const ModelContext& context) const;

  // -------------------- data members --------------------------------------
  uint64_t weight_count_;
  xllm::ParallelArgs parallel_args_;
  std::string quantize_type_;
  std::string torch_dtype_;
  torch::ScalarType dtype_;
  torch::TensorOptions options_;
  std::vector<at::Tensor> at_weight_tensors_;
  std::vector<at::Tensor> at_host_weight_tensors_;
  std::unique_ptr<ExpertBufferManager> shared_buffer_ = nullptr;
  std::unordered_map<std::string, torch::Tensor> shared_experts_weights_;
  std::unordered_map<std::string, std::vector<torch::Tensor>> experts_weights_;
  std::vector<int32_t> device_expert_list_;
  atb_torch::TorchTensorMap weights_map_;

  at::Device device_;
  int32_t dp_size_;
  int32_t dp_local_tp_size_;
  int32_t dp_rank_;
  int32_t dp_local_tp_rank_;
  int32_t cp_size_;

  LoadMode mode_ = LoadMode::kEager;

  // Manual-mode storage: host pinned buffer, device buffer and per-weight
  // slice metadata. Rolling-load path replaces the device buffer with a slot
  // pointer borrowed from `rolling_buffer_`.
  std::string model_id_;
  void* host_pinned_storage_ = nullptr;
  void* device_storage_ = nullptr;
  uint64_t storage_size_ = 0;
  std::vector<WeightSlice> weight_slices_;
  std::unordered_set<int> nz_indices_;
  std::shared_ptr<RollingWeightBuffer> rolling_buffer_ = nullptr;
  int32_t layer_index_ = -1;
  static constexpr size_t kDeviceAlignment = 64;
  static constexpr size_t kHostAlignment = 64;
};

}  // namespace layer
}  // namespace xllm

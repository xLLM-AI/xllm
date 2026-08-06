/* Copyright 2025-2026 The xLLM Authors. All Rights Reserved.

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

#include <torch/torch.h>
#include <tvm/ffi/container/array.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/extra/module.h>
#include <tvm/ffi/optional.h>

#include <optional>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

namespace ffi = tvm::ffi;

namespace xllm::kernel::musa {

inline bool is_torch_device(const torch::Device& device) {
  return device.is_privateuseone() || device.is_cuda();
}

void bind_tvmffi_stream(const torch::Device& device);

bool is_stream_capturing();

void sync_current_stream(const torch::Device& device);

void sync_ffi_stream(const torch::Device& device);

void sync_graph_preparation_stage(const torch::Device& device);

class TvmffiPreparationSyncGuard final {
 public:
  TvmffiPreparationSyncGuard();
  ~TvmffiPreparationSyncGuard();

  TvmffiPreparationSyncGuard(const TvmffiPreparationSyncGuard&) = delete;
  TvmffiPreparationSyncGuard& operator=(const TvmffiPreparationSyncGuard&) =
      delete;

 private:
  bool previous_ = false;
};

// During graph capture, replaces only an invalid current MUSA stream with the
// capture stream. Individual FFI operators still rebind their stream handle.
class TvmffiStreamOverrideGuard final {
 public:
  TvmffiStreamOverrideGuard(const torch::Device& device, void* stream);
  ~TvmffiStreamOverrideGuard();

  TvmffiStreamOverrideGuard(const TvmffiStreamOverrideGuard&) = delete;
  TvmffiStreamOverrideGuard& operator=(const TvmffiStreamOverrideGuard&) =
      delete;

 private:
  torch::Device device_;
  bool active_ = false;
  void* previous_forced_stream_ = nullptr;
};

class TvmffiStreamGuard final {
 public:
  explicit TvmffiStreamGuard(const torch::Device& device);
  ~TvmffiStreamGuard();

  TvmffiStreamGuard(const TvmffiStreamGuard&) = delete;
  TvmffiStreamGuard& operator=(const TvmffiStreamGuard&) = delete;

 private:
  torch::Device device_;
  bool active_ = false;
  // True when the FFI kernel was bound to the pool stream (eager-mode
  // fallback). In that case the destructor must establish ordering from the
  // FFI stream to the compute stream before subsequent PyTorch operations.
  bool needs_sync_ = false;
  bool uses_event_handoff_ = false;
};

torch::Tensor get_cache_buffer(const int32_t seq_len,
                               const torch::Device& device);

#define DISPATCH_CASE_FLOATING_TYPES(...)              \
  AT_DISPATCH_CASE(at::ScalarType::Float, __VA_ARGS__) \
  AT_DISPATCH_CASE(at::ScalarType::Half, __VA_ARGS__)  \
  AT_DISPATCH_CASE(at::ScalarType::BFloat16, __VA_ARGS__)
#define DISPATCH_FLOATING_TYPES(TYPE, NAME, ...) \
  AT_DISPATCH_SWITCH(TYPE, NAME, DISPATCH_CASE_FLOATING_TYPES(__VA_ARGS__))
#define DISPATCH_CASE_HALF_TYPES(...)                 \
  AT_DISPATCH_CASE(at::ScalarType::Half, __VA_ARGS__) \
  AT_DISPATCH_CASE(at::ScalarType::BFloat16, __VA_ARGS__)
#define DISPATCH_HALF_TYPES(TYPE, NAME, ...) \
  AT_DISPATCH_SWITCH(TYPE, NAME, DISPATCH_CASE_HALF_TYPES(__VA_ARGS__))

bool should_use_tensor_core(torch::ScalarType kv_cache_dtype,
                            int64_t num_attention_heads,
                            int64_t num_kv_heads);

bool support_pdl();

std::string path_to_uri_so_lib(const std::string& uri);

std::string determine_attention_backend(int64_t pos_encoding_mode,
                                        bool use_fp16_qk_reduction,
                                        bool use_custom_mask);

std::string get_batch_prefill_uri(const std::string& backend,
                                  torch::ScalarType dtype_q,
                                  torch::ScalarType dtype_kv,
                                  torch::ScalarType dtype_o,
                                  torch::ScalarType dtype_idx,
                                  int64_t head_dim_qk,
                                  int64_t head_dim_vo,
                                  int64_t pos_encoding_mode,
                                  bool use_sliding_window,
                                  bool use_logits_soft_cap,
                                  bool use_fp16_qk_reduction);

std::string get_batch_decode_uri(torch::ScalarType dtype_q,
                                 torch::ScalarType dtype_kv,
                                 torch::ScalarType dtype_o,
                                 torch::ScalarType dtype_idx,
                                 int64_t head_dim_qk,
                                 int64_t head_dim_vo,
                                 int64_t pos_encoding_mode,
                                 bool use_sliding_window,
                                 bool use_logits_soft_cap);

std::tuple<torch::Tensor, double> split_scale_param(const torch::Tensor& scale);

DLDataType to_dl_data_type(torch::ScalarType scalar_type);

ffi::Tensor to_ffi_tensor(const torch::Tensor& torch_tensor);

// Creates an FFI tensor that owns only its DLPack metadata.  The caller must
// keep the Torch storage alive until the launched device work completes.
ffi::Tensor to_ffi_borrowed_tensor(const torch::Tensor& torch_tensor);

ffi::TensorView to_ffi_tensor_view(const torch::Tensor& torch_tensor);

ffi::Optional<ffi::Tensor> to_ffi_optional_tensor(
    const std::optional<torch::Tensor>& optional);

ffi::Array<ffi::Tensor> to_ffi_array_tensors(
    const std::vector<torch::Tensor>& torch_tensors);

ffi::Optional<ffi::Array<ffi::Tensor>> to_ffi_optional_array_tensors(
    const std::optional<std::vector<torch::Tensor>>& optional);

ffi::Module get_module(const std::string& uri);

ffi::Function get_function(const std::string& uri,
                           const std::string& func_name);

// Registers TileLang's embedded MUSA-module loader with TVM FFI. Returns false
// when libtilelang is unavailable so callers can use a non-TileLang fallback.
bool ensure_tilelang_loader();

enum class FfiAllocMode { kPassthrough, kRecord, kReplay };

void begin_ffi_alloc_record();

std::vector<torch::Tensor> end_ffi_alloc_record();

void begin_ffi_alloc_replay(const std::vector<torch::Tensor>* recorded);

void end_ffi_alloc_replay();

FfiAllocMode get_ffi_alloc_mode();

void bind_tvmffi_stream_to_current_torch_stream(const torch::Device& device);
}  // namespace xllm::kernel::musa

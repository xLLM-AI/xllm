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

#include "base_loader.h"

#include "core/common/global_flags.h"
#include "core/framework/config/kv_cache_config.h"
#include "framework/xtensor/xtensor_allocator.h"
#include "rolling_weight_buffer.h"

#ifdef TORCH_HIGHER_THAN_PTA6
#include <torch_npu/csrc/core/npu/NPUFormat.h>
#include <torch_npu/csrc/framework/OpCommand.h>
#else
#include <torch_npu/csrc/aten/NPUNativeFunctions.h>
#include <torch_npu/csrc/framework/utils/OpPreparation.h>
#endif

namespace xllm {
namespace layer {

namespace {
static inline size_t AlignUp(size_t value, size_t alignment) {
  return (value + alignment - 1) & ~(alignment - 1);
}
}  // namespace

BaseLoader::BaseLoader(uint64_t weight_count,
                       const ModelContext& context,
                       LoadMode mode)
    : weight_count_(weight_count),
      parallel_args_(context.get_parallel_args()),
      device_(context.get_tensor_options().device()),
      mode_(mode),
      model_id_(context.get_model_id()) {
  auto quant_args = context.get_quant_args();
  if (!quant_args.quantize_type().empty()) {
    quantize_type_ = quant_args.quantize_type();
  }

  if (!quant_args.torch_dtype().empty()) {
    torch_dtype_ = quant_args.torch_dtype();
  }

  dp_size_ = parallel_args_.dp_size();
  cp_size_ = parallel_args_.cp_size();
  dp_local_tp_size_ = parallel_args_.world_size() / (dp_size_ * cp_size_);
  dp_rank_ = parallel_args_.rank() / dp_local_tp_size_ * cp_size_;
  CHECK_EQ(parallel_args_.world_size(),
           dp_size_ * dp_local_tp_size_ * cp_size_);
  dp_local_tp_rank_ = parallel_args_.rank() % dp_local_tp_size_;

  at_weight_tensors_.resize(weight_count_);
  if (mode_ == LoadMode::kManual) {
    at_host_weight_tensors_.resize(weight_count_);
  }
}

BaseLoader::~BaseLoader() {
  release_host_storage();
  release_device_storage();
}

// ----------------------------- set_weight ---------------------------------

void BaseLoader::set_weight(const StateDict& state_dict,
                            const std::string& tensor_name,
                            int weight_position,
                            bool to_host) {
  auto device = to_host ? at::kCPU : device_;
  for (const auto& [name, tensor] : state_dict) {
    if (absl::EndsWith(name, tensor_name)) {
      at::Tensor mutable_tensor = tensor;
      correct_tensor_dtype(mutable_tensor, tensor_name);
      if (to_host) {
        at_host_weight_tensors_[weight_position] = mutable_tensor.to(device);
      } else {
        at_weight_tensors_[weight_position] = mutable_tensor.to(device);
      }
    }
  }
}

void BaseLoader::set_weight(const StateDict& state_dict,
                            const std::string& tensor_name,
                            int weight_position,
                            int dim,
                            bool to_host) {
  auto device = to_host ? at::kCPU : device_;
  if (parallel_args_.world_size() <= 1) {
    for (const auto& [name, tensor] : state_dict) {
      if (absl::EndsWith(name, tensor_name)) {
        at::Tensor mutable_tensor = tensor;
        correct_tensor_dtype(mutable_tensor, tensor_name);
        if (to_host) {
          at_host_weight_tensors_[weight_position] = mutable_tensor.to(device);
        } else {
          at_weight_tensors_[weight_position] = mutable_tensor.to(device);
        }
      }
    }
  } else {
    for (const auto& [name, tensor] : state_dict) {
      if (absl::EndsWith(name, tensor_name)) {
        at::Tensor mutable_tensor = state_dict.get_sharded_tensor(
            tensor_name,
            /*dim=*/dim,
            /*rank=*/parallel_args_.rank(),
            /*world_size=*/parallel_args_.world_size());
        correct_tensor_dtype(mutable_tensor, tensor_name);
        if (to_host) {
          at_host_weight_tensors_[weight_position] = mutable_tensor.to(device);
        } else {
          at_weight_tensors_[weight_position] = mutable_tensor.to(device);
        }
      }
    }
  }
}

void BaseLoader::set_weight(const StateDict& state_dict,
                            const std::string& tensor_name,
                            int weight_position,
                            int dim,
                            int rank,
                            int world_size,
                            bool to_host) {
  auto device = to_host ? at::kCPU : device_;
  if (world_size <= 1) {
    for (const auto& [name, tensor] : state_dict) {
      if (absl::EndsWith(name, tensor_name)) {
        at::Tensor mutable_tensor = tensor;
        correct_tensor_dtype(mutable_tensor, tensor_name);
        if (to_host) {
          at_host_weight_tensors_[weight_position] = mutable_tensor.to(device);
        } else {
          at_weight_tensors_[weight_position] = mutable_tensor.to(device);
        }
      }
    }
  } else {
    for (const auto& [name, tensor] : state_dict) {
      if (absl::EndsWith(name, tensor_name)) {
        at::Tensor mutable_tensor =
            state_dict.get_sharded_tensor(tensor_name,
                                          /*dim=*/dim,
                                          /*rank=*/rank,
                                          /*world_size=*/world_size);
        correct_tensor_dtype(mutable_tensor, tensor_name);
        if (to_host) {
          at_host_weight_tensors_[weight_position] = mutable_tensor.to(device);
        } else {
          at_weight_tensors_[weight_position] = mutable_tensor.to(device);
        }
      }
    }
  }
}

void BaseLoader::correct_tensor_dtype(torch::Tensor& tensor,
                                      const std::string& tensorName) {
  if (absl::EndsWith(tensorName, "deq_scale") &&
      (torch_dtype_.compare("bfloat16") == 0)) {
    return;
  }

  if (tensor.dtype() != torch::kInt8 && tensor.dtype() != torch::kInt32 &&
      tensor.dtype() != torch::kInt64) {
    torch::Dtype dtype = string2dtype(torch_dtype_);
    tensor = tensor.to(dtype);
  }
}

torch::Dtype BaseLoader::string2dtype(const std::string& dtype_str) {
  if (dtype_str.compare("float16") == 0) {
    return torch::kFloat16;
  } else if (dtype_str.compare("bfloat16") == 0) {
    return torch::kBFloat16;
  } else if (dtype_str.compare("float32") == 0) {
    return torch::kFloat32;
  } else if (dtype_str.compare("float64") == 0) {
    return torch::kFloat64;
  } else if (dtype_str.compare("int8") == 0) {
    return torch::kInt8;
  } else if (dtype_str.compare("int16") == 0) {
    return torch::kInt16;
  } else if (dtype_str.compare("int32") == 0) {
    return torch::kInt32;
  } else if (dtype_str.compare("int64") == 0) {
    return torch::kInt64;
  } else if (dtype_str.compare("uint8") == 0) {
    return torch::kUInt8;
  } else if (dtype_str.compare("bool") == 0) {
    return torch::kBool;
  }

  LOG(FATAL) << "Unsupported dtype string: " << dtype_str;
  return torch::kFloat16;
}

at::Tensor BaseLoader::pad_vocab_tensor(const at::Tensor& tensor,
                                        int64_t padded_vocab_size) const {
  if (tensor.size(0) >= padded_vocab_size) {
    return tensor;
  }
  at::Tensor padded_tensor =
      torch::zeros({padded_vocab_size, tensor.size(1)}, tensor.options());
  padded_tensor.slice(0, 0, tensor.size(0)) = tensor;
  return padded_tensor;
}

at::Tensor BaseLoader::shard_padded_tensor(const at::Tensor& padded_tensor,
                                           int dim,
                                           int rank,
                                           int world_size) const {
  if (world_size <= 1) {
    return padded_tensor;
  }
  auto chunks = padded_tensor.chunk(world_size, dim);
  return chunks[rank];
}

void BaseLoader::set_weight_with_padding(const StateDict& state_dict,
                                         const std::string& tensor_name,
                                         int weight_position,
                                         int dim,
                                         int64_t padded_vocab_size,
                                         bool to_host) {
  auto device = to_host ? at::kCPU : device_;
  for (const auto& [name, tensor] : state_dict) {
    if (absl::EndsWith(name, tensor_name)) {
      at::Tensor mutable_tensor = tensor;
      if (padded_vocab_size > tensor.size(0)) {
        mutable_tensor = pad_vocab_tensor(tensor, padded_vocab_size);
      }
      correct_tensor_dtype(mutable_tensor, tensor_name);
      if (to_host) {
        at_host_weight_tensors_[weight_position] = mutable_tensor.to(device);
      } else {
        at_weight_tensors_[weight_position] = mutable_tensor.to(device);
      }
    }
  }
}

void BaseLoader::set_weight_with_padding(const StateDict& state_dict,
                                         const std::string& tensor_name,
                                         int weight_position,
                                         int dim,
                                         int rank,
                                         int world_size,
                                         int64_t padded_vocab_size,
                                         bool to_host) {
  auto device = to_host ? at::kCPU : device_;
  if (world_size <= 1) {
    set_weight_with_padding(state_dict,
                            tensor_name,
                            weight_position,
                            dim,
                            padded_vocab_size,
                            to_host);
    return;
  }
  for (const auto& [name, tensor] : state_dict) {
    if (absl::EndsWith(name, tensor_name)) {
      at::Tensor mutable_tensor = tensor;
      if (padded_vocab_size > tensor.size(0)) {
        // Memory-optimized path for vocabulary dimension sharding
        if (dim == 0) {
          int64_t shard_size = padded_vocab_size / world_size;
          int64_t start_idx = rank * shard_size;
          int64_t end_idx = (rank + 1) * shard_size;
          if (start_idx >= tensor.size(0)) {
            mutable_tensor =
                torch::zeros({shard_size, tensor.size(1)}, tensor.options());
          } else {
            auto valid_part =
                tensor.slice(0, start_idx, std::min(end_idx, tensor.size(0)));
            if (valid_part.size(0) < shard_size) {
              mutable_tensor =
                  torch::zeros({shard_size, tensor.size(1)}, tensor.options());
              mutable_tensor.slice(0, 0, valid_part.size(0)).copy_(valid_part);
            } else {
              mutable_tensor = valid_part.clone();
            }
          }
        } else {
          // Non-vocabulary dimension: use original approach
          mutable_tensor = pad_vocab_tensor(tensor, padded_vocab_size);
          mutable_tensor =
              shard_padded_tensor(mutable_tensor, dim, rank, world_size);
        }
      } else {
        mutable_tensor =
            state_dict.get_sharded_tensor(tensor_name, dim, rank, world_size);
      }
      correct_tensor_dtype(mutable_tensor, tensor_name);
      if (to_host) {
        at_host_weight_tensors_[weight_position] = mutable_tensor.to(device);
      } else {
        at_weight_tensors_[weight_position] = mutable_tensor.to(device);
      }
    }
  }
}

int64_t BaseLoader::get_padded_vocab_size(const ModelContext& context) const {
  int64_t vocab_size = context.get_model_args().vocab_size();
  int32_t local_tp_size = dp_local_tp_size_;
  if (vocab_size > 0 && local_tp_size > 1 && vocab_size % local_tp_size != 0) {
    return ((vocab_size + local_tp_size - 1) / local_tp_size) * local_tp_size;
  }
  return vocab_size;
}

// ----------------------- uniform staging helpers --------------------------

at::Tensor BaseLoader::zero_like_working(int idx) const {
  const auto& ref = working_tensors()[idx];
  return torch::zeros(
      {1},
      torch::TensorOptions().dtype(ref.scalar_type()).device(target_device()));
}

at::Tensor BaseLoader::cast_nz(at::Tensor t, int idx) {
  if (mode_ == LoadMode::kManual) {
    // Defer NZ conversion; BaseLoader::copy_weights_to_device will route this
    // slice through `copy_host_nd_to_nz` based on `nz_indices_`.
    nz_indices_.insert(idx);
    return t.contiguous();
  }
  return at_npu::native::npu_format_cast(t.contiguous(), ACL_FORMAT_FRACTAL_NZ);
}

bool BaseLoader::is_nz_format_tensor(int weight_index) {
  return nz_indices_.count(weight_index) > 0;
}

// --------------------- manual-mode pipeline (ex BaseManualLoader) ---------

void BaseLoader::merge_loaded_weights() {
  // Shared subclass hook. Legacy eager loaders override
  // `merge_loaded_weights` directly and never reach this body; unified
  // (mode-aware) loaders override `merge_host_at_weights` and rely on this
  // dispatch to handle both modes.
  merge_host_at_weights();
  if (mode_ == LoadMode::kManual) {
    init_weight_slices();
    copy_weights_to_device();
    init_device_at_weights();
  }
}

void BaseLoader::merge_and_move_pinned_host() {
  CHECK(mode_ == LoadMode::kManual)
      << "merge_and_move_pinned_host is only valid in manual loader mode";
  merge_host_at_weights();
  init_weight_slices();
  copy_weights_to_pinned_host();
}

void BaseLoader::free_weights() { release_host_storage(); }

void BaseLoader::reload_weights() {
  CHECK(mode_ == LoadMode::kManual)
      << "reload_weights is only valid in manual loader mode";
  allocate_device_storage();
  copy_weights_to_device_async();
  init_device_at_weights();
}

void BaseLoader::reload_weights_from_device() {
  CHECK(mode_ == LoadMode::kManual)
      << "reload_weights_from_device is only valid in manual loader mode";
  // P2P path: weights already transferred to GlobalXTensor weight region.
  // Call allocate_weight to get the pointer into the pre-allocated region.
  allocate_device_storage();
  init_device_at_weights();
}

void BaseLoader::refresh_rolling_weights() {
  if (rolling_buffer_ == nullptr) {
    return;
  }
  allocate_device_storage();
  init_device_at_weights();
}

void BaseLoader::set_rolling_buffer(std::shared_ptr<RollingWeightBuffer> buf,
                                    int32_t layer_index) {
  rolling_buffer_ = std::move(buf);
  layer_index_ = layer_index;
}

void BaseLoader::allocate_device_storage() {
  if (rolling_buffer_ != nullptr) {
    // Rolling load path: use the pre-allocated slot instead of
    // XTensorAllocator. Decoder layer weights bypass XTensor regardless of
    // enable_xtensor flag.
    CHECK_GE(layer_index_, 0) << "layer_index_ not set for rolling buffer";
    device_storage_ = rolling_buffer_->get_slot_ptr(layer_index_);
    CHECK(device_storage_ != nullptr)
        << "RollingWeightBuffer slot is null for layer " << layer_index_;
    return;
  }
  if (::xllm::KVCacheConfig::get_instance().enable_xtensor()) {
    auto& allocator = XTensorAllocator::get_instance();
    bool ok =
        allocator.allocate_weight(model_id_, device_storage_, storage_size_);
    CHECK(ok) << "Failed to allocate contiguous device storage size="
              << storage_size_;
    return;
  }
  auto ret = aclrtMallocAlign32(
      &device_storage_, storage_size_, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_EQ(ret, ACL_SUCCESS)
      << "aclrtMallocAlign32 failed for BaseLoader, size=" << storage_size_
      << ", ret=" << ret;
}

void BaseLoader::init_weight_slices() {
  weight_slices_.resize(weight_count_);
  size_t offset = 0;
  for (size_t i = 0; i < weight_count_; ++i) {
    weight_slices_[i] = {};
    const auto& tensor = at_host_weight_tensors_[i];
    if (!tensor.defined() || tensor.numel() < 1) {
      continue;
    }
    offset = AlignUp(offset, kHostAlignment);
    weight_slices_[i].offset = offset;
    weight_slices_[i].bytes = tensor.nbytes();
    weight_slices_[i].sizes = tensor.sizes().vec();
    weight_slices_[i].dtype = tensor.scalar_type();
    offset += weight_slices_[i].bytes;
  }
  size_t max_alignment = std::max(kHostAlignment, kDeviceAlignment);
  storage_size_ = AlignUp(offset, max_alignment);
}

void BaseLoader::copy_weights_to_pinned_host() {
  CHECK_GT(storage_size_, 0) << "model size must be greater than 0.";
  CHECK_EQ(weight_slices_.size(), at_host_weight_tensors_.size())
      << "weight_slices_ size and at_host_weight_tensors_ size mismatch.";

  size_t max_alignment = std::max(kHostAlignment, kDeviceAlignment);
  storage_size_ = AlignUp(storage_size_, max_alignment);

  auto ret = aclrtMallocHost(&host_pinned_storage_, storage_size_);
  CHECK_EQ(ret, ACL_SUCCESS)
      << "Failed to allocate pinned host storage size=" << storage_size_;

  for (size_t i = 0; i < weight_slices_.size(); ++i) {
    const auto& slice = weight_slices_[i];
    if (!slice.bytes) {
      continue;
    }
    void* dst = static_cast<char*>(host_pinned_storage_) +
                static_cast<ptrdiff_t>(slice.offset);
    auto host_tensor = at_host_weight_tensors_[i].to(torch::kCPU).contiguous();

    if (is_nz_format_tensor(i)) {
      int err = copy_host_nd_to_nz(
          host_tensor, dst, slice.bytes, ACL_MEMCPY_DEVICE_TO_HOST);
      CHECK_EQ(err, ACL_SUCCESS)
          << "copy_host_nd_to_nz failed for tensor index " << i;
    } else {
      std::memcpy(dst, host_tensor.data_ptr(), slice.bytes);
    }
    at_host_weight_tensors_[i] = torch::zeros({1});
  }
}

void BaseLoader::copy_weights_to_device_async() {
  CHECK_EQ(weight_slices_.size(), at_weight_tensors_.size())
      << "weight_slices_ size and at_weight_tensors_ size mismatch.";
  copy_weights_to_device_async(c10_npu::getCurrentNPUStream().stream());
}

void BaseLoader::copy_weights_to_device_async(aclrtStream stream) {
  void* dst = static_cast<char*>(device_storage_);
  void* src = static_cast<char*>(host_pinned_storage_);

  auto ret = aclrtMemcpyAsync(dst,
                              storage_size_,
                              src,
                              storage_size_,
                              ACL_MEMCPY_HOST_TO_DEVICE,
                              stream);
  CHECK_EQ(ret, ACL_SUCCESS) << "aclrtMemcpyAsync failed (rolling)";
}

void BaseLoader::copy_weights_to_device() {
  CHECK_EQ(weight_slices_.size(), at_host_weight_tensors_.size())
      << "weight_slices_ size and at_host_weight_tensors_ size mismatch.";

  allocate_device_storage();

  for (size_t i = 0; i < weight_slices_.size(); ++i) {
    const auto& slice = weight_slices_[i];
    if (!slice.bytes) {
      continue;
    }
    void* dst = static_cast<char*>(device_storage_) +
                static_cast<ptrdiff_t>(slice.offset);
    auto host_tensor = at_host_weight_tensors_[i].contiguous();
    int err;
    if (is_nz_format_tensor(i)) {
      err = copy_host_nd_to_nz(host_tensor, dst, slice.bytes);
    } else {
      err = aclrtMemcpy(dst,
                        slice.bytes,
                        host_tensor.data_ptr(),
                        slice.bytes,
                        ACL_MEMCPY_HOST_TO_DEVICE);
    }
    CHECK_EQ(err, ACL_SUCCESS) << "aclrtMemcpy failed for tensor index " << i;
    at_host_weight_tensors_[i] = torch::zeros({1});
  }
}

int BaseLoader::copy_host_nd_to_nz(torch::Tensor host_tensor,
                                   void* dst_ptr,
                                   uint64_t len,
                                   aclrtMemcpyKind kind) {
  auto tmp_tensor = at_npu::native::npu_format_cast(host_tensor.to(device_),
                                                    ACL_FORMAT_FRACTAL_NZ);
  const void* src_ptr = tmp_tensor.data_ptr();
  auto stream = c10_npu::getCurrentNPUStream();
  auto err = aclrtMemcpyAsync(dst_ptr, len, src_ptr, len, kind, stream);
  stream.synchronize();
  tmp_tensor = torch::Tensor();

  return err;
}

void BaseLoader::init_device_at_weights() {
  for (size_t i = 0; i < weight_slices_.size(); ++i) {
    const auto& slice = weight_slices_[i];
    if (!slice.bytes) {
      continue;
    }
    void* base = static_cast<char*>(device_storage_) +
                 static_cast<ptrdiff_t>(slice.offset);
    if (is_nz_format_tensor(i)) {
      at_weight_tensors_[i] =
          convert_to_torch_tensor(slice.sizes,
                                  slice.dtype,
                                  reinterpret_cast<uintptr_t>(base),
                                  ACL_FORMAT_FRACTAL_NZ);
    } else {
      at_weight_tensors_[i] = convert_to_torch_tensor(
          slice.sizes, slice.dtype, reinterpret_cast<uintptr_t>(base));
    }
  }
}

void BaseLoader::release_device_storage() {
  if (device_storage_ == nullptr) {
    return;
  }
  if (!::xllm::KVCacheConfig::get_instance().enable_xtensor() &&
      !rolling_buffer_) {
    auto ret = aclrtFree(device_storage_);
    if (ret != ACL_SUCCESS) {
      LOG(ERROR) << "aclrtFree failed for BaseLoader, ret=" << ret;
    }
  }
  device_storage_ = nullptr;
}

void BaseLoader::release_host_storage() {
  if (host_pinned_storage_ == nullptr) {
    return;
  }
  auto ret = aclrtFreeHost(host_pinned_storage_);
  if (ret != ACL_SUCCESS) {
    LOG(ERROR) << "Failed to free pinned host storage, ret=" << ret;
  }
  host_pinned_storage_ = nullptr;
}

torch::Tensor BaseLoader::convert_to_torch_tensor(
    const std::vector<int64_t>& dims,
    const torch::ScalarType dtype,
    const uintptr_t& dev_addr,
    int acl_format) {
  c10::DeviceType device_type = c10::DeviceType::PrivateUse1;
  torch::TensorOptions option =
      torch::TensorOptions().dtype(dtype).device(device_type);

  auto tensor = torch::empty({0}, option);
  auto address = reinterpret_cast<void*>(dev_addr);
  torch::DataPtr c10_data_ptr(address, address, [](void*) {}, tensor.device());

  size_t tensor_nbytes = at::detail::computeStorageNbytesContiguous(
      dims, tensor.dtype().itemsize());
  torch::Storage storage;
  // get npu storage constructor from register and construct storage
  auto fptr = c10::GetStorageImplCreate(device_type);
  auto allocator = c10::GetAllocator(device_type);

  // PyTorch 2.7+: StorageImpl now takes DataPtr instead of raw allocator
  storage = fptr(c10::StorageImpl::use_byte_size_t(),
                 c10::SymInt(tensor_nbytes),
                 std::move(c10_data_ptr),
                 allocator,
                 true);

  tensor.set_(storage, 0, dims);
  // Notice: convert to NZ format forcefully, with the underlying data format
  // guaranteed by the developer.
  if (acl_format == ACL_FORMAT_FRACTAL_NZ) {
    auto* tensor_storage = static_cast<torch_npu::NPUStorageImpl*>(
        tensor.storage().unsafeGetStorageImpl());
    tensor_storage->npu_desc_.npu_format_ = ACL_FORMAT_FRACTAL_NZ;
  }

  return tensor;
}

}  // namespace layer
}  // namespace xllm

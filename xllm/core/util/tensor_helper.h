/* Copyright 2025-2026 The xLLM Authors.

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

#include <c10/core/TensorOptions.h>
#include <glog/logging.h>
#include <torch/torch.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <optional>
#include <source_location>
#include <span>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include "util/slice.h"

namespace xllm {

inline bool is_cpu_int_tensor(const torch::Tensor& tensor, int32_t dimensions) {
  return tensor.defined() && tensor.device().is_cpu() &&
         tensor.scalar_type() == torch::kInt32 && tensor.dim() == dimensions &&
         tensor.is_contiguous();
}

// CHECKed flat borrow; T defaults to int32, the dominant token/length type.
template <typename T = int32_t>
inline Slice<T> tensor_slice(const torch::Tensor& tensor) {
  CHECK(tensor.defined()) << "tensor_slice requires a defined tensor";
  CHECK(tensor.device().is_cpu()) << "tensor_slice requires a CPU tensor";
  CHECK(tensor.is_contiguous()) << "tensor_slice requires contiguous storage";
  return {tensor.const_data_ptr<T>(), static_cast<size_t>(tensor.numel())};
}

// Lenient twin of tensor_slice: empty view instead of CHECK on empty input.
inline std::span<const int32_t> int_span(const torch::Tensor& tensor) {
  if (!tensor.defined() || tensor.numel() == 0) {
    return {};
  }
  return {tensor.const_data_ptr<int32_t>(),
          static_cast<uint64_t>(tensor.numel())};
}

namespace detail {

// H2D-source options; the caching host allocator recycles pinned blocks.
inline torch::TensorOptions pinned_cpu_options(torch::ScalarType dtype) {
  return torch::TensorOptions()
      .dtype(dtype)
      .device(torch::kCPU)
      .pinned_memory(true);
}

// Host-only tensors stay pageable; pinning them is pure waste.
inline torch::TensorOptions pageable_cpu_options(torch::ScalarType dtype) {
  return torch::TensorOptions().dtype(dtype).device(torch::kCPU);
}

// empty+memcpy avoids torch::tensor's intermediate copy; vector<bool> is
// bit-packed with no data(), so it is filled element-by-element.
template <typename T>
inline torch::Tensor make_cpu_tensor_with_options(
    const std::vector<T>& values,
    const torch::TensorOptions& options) {
  torch::Tensor tensor =
      torch::empty({static_cast<int64_t>(values.size())}, options);
  if constexpr (std::is_same_v<T, bool>) {
    bool* data = tensor.data_ptr<bool>();
    for (size_t i = 0; i < values.size(); ++i) {
      data[i] = values[i];
    }
  } else {
    std::memcpy(tensor.data_ptr<T>(), values.data(), values.size() * sizeof(T));
  }
  return tensor;
}

}  // namespace detail

template <typename T>
inline torch::Tensor create_2d_tensor(const std::vector<std::vector<T> >& vec,
                                      torch::ScalarType dtype) {
  if (vec.empty()) {
    return {};
  }
  const size_t n_rows = vec.size();
  const size_t n_cols = vec[0].size();
  auto tensor =
      torch::empty({static_cast<int64_t>(n_rows), static_cast<int64_t>(n_cols)},
                   detail::pinned_cpu_options(dtype));
  // Fill the contiguous pinned buffer row-by-row with a plain memcpy. The
  // element type T must match `dtype`'s storage; every current caller pairs
  // int32_t with kInt and int64_t with kInt64. Copying directly avoids a
  // per-row pinned allocation plus an autograd-aware `tensor[i] = ...`
  // indexed assignment, which dominated large-batch input building.
  CHECK_EQ(
      torch::empty({0}, torch::TensorOptions().dtype(dtype)).element_size(),
      sizeof(T))
      << "create_2d_tensor element type does not match dtype storage size";
  auto* dst = static_cast<T*>(tensor.data_ptr());
  for (size_t i = 0; i < n_rows; ++i) {
    CHECK_EQ(vec[i].size(), n_cols);
    std::memcpy(dst + i * n_cols, vec[i].data(), n_cols * sizeof(T));
  }
  return tensor;
}

inline torch::Tensor safe_to(const torch::Tensor& t,
                             const torch::TensorOptions& options,
                             bool non_blocking = false) {
  return t.defined() ? t.to(options, non_blocking) : t;
}

// Copies directly into independent contiguous storage, detached from autograd.
inline torch::Tensor clone_contiguous_detached_tensor(
    const torch::Tensor& tensor) {
  return tensor.detach().clone(torch::MemoryFormat::Contiguous);
}

template <typename T>
constexpr torch::ScalarType get_scalar_type() {
  return c10::CppTypeToScalarType<T>::value;
}

// Pageable: pinned only pays off as a non_blocking H2D source; paths that
// upload directly need make_pinned_cpu_tensor instead.
template <typename T>
inline torch::Tensor make_cpu_tensor(const std::vector<T>& values) {
  return detail::make_cpu_tensor_with_options(
      values, detail::pageable_cpu_options(get_scalar_type<T>()));
}

// Pinned variant for host tensors that are the source of an async H2D copy;
// the caching host allocator recycles these blocks, so repeated staging is
// cheap, but a pinned allocation on a host-only path is pure waste.
template <typename T>
inline torch::Tensor make_pinned_cpu_tensor(const std::vector<T>& values) {
  return detail::make_cpu_tensor_with_options(
      values, detail::pinned_cpu_options(get_scalar_type<T>()));
}

// Identity index vector [0, count) as an int32 device tensor, built directly
// on device: copying a temporary pinned CPU source forces its allocator to
// synchronize before the async H2D completes.
inline torch::Tensor arange_indices(int64_t count,
                                    const torch::Device& device) {
  return torch::arange(
      count, torch::TensorOptions().dtype(torch::kInt).device(device));
}

// Stages a host vector onto the device with an async H2D copy from a pinned
// CPU tensor, issued on the caller's current stream: the caller must keep
// that stream active (e.g. via its stream guard) until the copy is ordered
// before any consumer, and must synchronize before reading the result on
// the host.
template <typename T>
inline torch::Tensor async_h2d_tensor(const std::vector<T>& values,
                                      const torch::Device& device) {
  return make_pinned_cpu_tensor(values).to(device, /*non_blocking=*/true);
}

inline std::vector<char> get_the_bytes(std::string filename) {
  std::ifstream input(filename, std::ios::binary);
  std::vector<char> bytes((std::istreambuf_iterator<char>(input)),
                          (std::istreambuf_iterator<char>()));

  input.close();
  return bytes;
}

inline torch::Tensor load_tensor(std::string filename) {
  std::vector<char> f = get_the_bytes(filename);
  torch::IValue x = torch::pickle_load(f);
  torch::Tensor my_tensor = x.toTensor();
  return my_tensor;
}

inline void print_tensor(
    const torch::Tensor& tensor,
    const std::string& tensor_name = "tensor",
    int num = 10,
    bool part = true,
    bool print_value = true,
    const std::source_location& loc = std::source_location::current()) {
  auto& log_stream =
      google::LogMessage(
          loc.file_name(), static_cast<int>(loc.line()), google::GLOG_INFO)
          .stream();
  if (!tensor.defined()) {
    log_stream << tensor_name << ", Undefined tensor." << std::endl;
    return;
  }

  log_stream << tensor_name << ": " << tensor.sizes()
             << ", dtype: " << tensor.dtype() << ", device: " << tensor.device()
             << std::endl;

  if (!print_value) {
    return;
  }

  if (part) {
    const auto& flat_tensor = tensor.contiguous().view(-1);
    int max_elements = std::min(static_cast<int>(flat_tensor.size(0)), num);
    // const auto& front_elements = flat_tensor.slice(0, 0, max_elements);
    const auto& front_elements =
        flat_tensor.slice(0, 0, max_elements).to(torch::kCPU);
    log_stream << "First " << max_elements << " elements: \n"
               << front_elements << std::endl;

    int back_num = flat_tensor.size(0) > num ? num : flat_tensor.size(0);
    // const auto& back_elements = flat_tensor.slice(0, flat_tensor.size(0) -
    // back_num, flat_tensor.size(0));
    const auto& back_elements =
        flat_tensor
            .slice(0, flat_tensor.size(0) - back_num, flat_tensor.size(0))
            .to(torch::kCPU);
    log_stream << "Last " << back_num << " elements: \n"
               << back_elements << std::endl;
  } else {
    log_stream << "All: \n" << tensor.to(torch::kCPU) << std::endl;
  }
}

inline bool file_exists(const std::string& path) {
  std::ifstream file(path);
  return file.good();
}

inline bool tensor_batch_signature_matches(const torch::Tensor& lhs,
                                           const torch::Tensor& rhs) {
  if (!lhs.defined() || !rhs.defined()) {
    return lhs.defined() == rhs.defined();
  }
  if (lhs.scalar_type() != rhs.scalar_type() || lhs.device() != rhs.device() ||
      lhs.dim() != rhs.dim()) {
    return false;
  }
  for (int64_t i = 0; i < lhs.dim(); ++i) {
    if (lhs.size(i) != rhs.size(i)) {
      return false;
    }
  }
  return true;
}

inline torch::Tensor safe_concat(const torch::Tensor& t1,
                                 const torch::Tensor& t2,
                                 const uint32_t dim) {
  if (t1.defined() && t2.defined()) {
    return torch::cat({t1, t2}, dim);
  } else if (!t1.defined()) {
    return t2;
  } else {
    return t1;
  }
}

inline bool safe_concat(const std::vector<torch::Tensor>& vec,
                        torch::Tensor& tar,
                        int64_t dim = 0) {
  auto check = [](const std::vector<torch::Tensor>& vec, int64_t dim) {
    if (vec.empty()) return false;

    const auto& ref = vec[0];
    if (!ref.defined()) return false;

    const int64_t ndim = ref.dim();
    if (ndim == 0) return false;

    if (dim < 0) dim += ndim;

    if (dim >= ndim) return false;

    for (size_t i = 1; i < vec.size(); ++i) {
      const auto& t = vec[i];
      if (!t.defined()) {
        return false;
      }

      if (t.dtype() != ref.dtype() || t.device() != ref.device() ||
          t.dim() != ref.dim()) {
        return false;
      }

      for (int64_t d = 0; d < ndim; ++d) {
        if (d == dim) continue;
        if (t.size(d) != ref.size(d)) {
          return false;
        }
      }
    }
    return true;
  };

  if (check(vec, dim)) {
    tar = torch::cat(vec, dim);
    return true;
  } else {
    return false;
  }
}

// save torch tensor to .pt file as pickle format, which is same as torch.save
// in python. .pt file can be loaded by torch.load in python. file_path must end
// with ".pt".
inline void save_tensor_as_pickle(const torch::Tensor& tensor,
                                  const std::string& file_path) {
  std::vector<char> pickled = torch::pickle_save(tensor);
  std::ofstream ofs(file_path, std::ios::binary);
  CHECK(ofs.good()) << "Cannot open file: " << file_path;
  ofs.write(pickled.data(), static_cast<std::streamsize>(pickled.size()));
  CHECK(ofs.good()) << "Write failed to: " << file_path;
}

// Computes the new shape for tensor view casting between dtypes by bytes, for
// use with from_blob.
inline std::vector<int64_t> compute_view_shape(const torch::Tensor& src,
                                               int64_t src_size,
                                               int64_t target_size) {
  std::vector<int64_t> new_sizes = src.sizes().vec();

  if (src_size == target_size) {
    // No size change, just return original shape
    return new_sizes;
  } else if (src_size > target_size) {
    // Splitting: each element will be split into more elements of smaller dtype
    // (e.g., BFloat16 -> char)
    int64_t ratio = src_size / target_size;
    if (new_sizes.empty()) {
      // Scalar tensor: introduce new dimension of length ratio
      new_sizes.push_back(ratio);
    } else {
      // For normal tensors: expand the last dimension accordingly
      // e.g. [8, 2048] -> [8, 4096]
      new_sizes.back() *= ratio;
    }
  } else {
    // Merging: multiple small dtype elements become one larger dtype element
    // (e.g., char -> BFloat16)
    int64_t ratio = target_size / src_size;

    // Ensure tensor is not scalar
    CHECK(!new_sizes.empty()) << "Cannot merge views for a scalar tensor.";

    int64_t last_dim = new_sizes.back();
    // Last dim size must be divisible by merge ratio
    CHECK(last_dim % ratio == 0)
        << "Last dimension size (" << last_dim
        << ") must be divisible by type ratio (" << ratio
        << ") when viewing as a larger dtype.";
    new_sizes.back() = last_dim / ratio;
  }
  return new_sizes;
}

// Simulates the Python tensor.view(dtype) functionality.
// Reinterprets a raw byte tensor (usually uint8) as a tensor of the target data
// type. Note: The input tensor must be contiguous in memory.
inline torch::Tensor view_as_dtype(const torch::Tensor& src,
                                   torch::ScalarType target_dtype) {
  // If the source type already matches the target type, just return as is.
  if (src.scalar_type() == target_dtype) {
    return src;
  }

  // core constraint: require the input tensor to be contiguous for raw memory
  // reinterpretation. use CHECK to enforce contiguity; if failed, the caller
  // must ensure .contiguous() is called beforehand.
  CHECK(src.is_contiguous())
      << "view_as_dtype expects a contiguous tensor. Please call .contiguous() "
         "before passing it in.";

  // calculate the source and target element sizes in bytes.
  int64_t src_element_size = src.element_size();
  int64_t target_element_size = torch::elementSize(target_dtype);
  std::vector<int64_t> new_shape =
      compute_view_shape(src, src_element_size, target_element_size);

  // we pass in a lambda, capture 'src' (by value, increase reference count)
  // when the returned tensor is destroyed, this lambda will be called, thus
  // releasing the reference to src. this makes the new tensor truly own the
  // "share" of the underlying storage, like python's view is safe.
  auto deleter = [src](void*) {
    // this empty lambda just captures src, can keep the memory alive.
    // src will automatically reduce reference count when the lambda is
    // destroyed
  };

  // Create a zero-copy view on the same memory.
  //    Notes:
  //    - data_ptr() directly points to the src tensor's memory.
  //    - The returned tensor does NOT own the memory.
  //    - The src tensor's lifetime MUST cover the returned tensor.
  return torch::from_blob(
      src.data_ptr(), new_shape, deleter, src.options().dtype(target_dtype));
}

// Contiguous CPU copy with the requested scalar type; the trailing
// contiguous() covers the same-device/dtype no-op path, where Tensor::to()
// returns a non-contiguous input as-is.
inline torch::Tensor to_cpu_contiguous(
    const torch::Tensor& tensor,
    std::optional<torch::ScalarType> dtype = std::nullopt) {
  return safe_to(tensor,
                 torch::TensorOptions()
                     .device(torch::kCPU)
                     .dtype(dtype.value_or(tensor.scalar_type())))
      .contiguous();
}

// Reads a tensor into a host vector, converting to T's scalar type if needed.
// Undefined input yields an empty vector: data_ptr() on an undefined tensor
// is not safe to dereference.
template <typename T>
inline std::vector<T> tensor_to_vector(const torch::Tensor& tensor) {
  if (!tensor.defined()) {
    return {};
  }
  const torch::Tensor cpu_tensor =
      to_cpu_contiguous(tensor, get_scalar_type<T>());
  const T* data_ptr = cpu_tensor.data_ptr<T>();
  const size_t size = static_cast<size_t>(cpu_tensor.numel());
  return std::vector<T>(data_ptr, data_ptr + size);
}

inline std::optional<torch::ScalarType> try_get_scalar_type_from_string(
    const std::string& dtype_str) {
  static const std::unordered_map<std::string, torch::ScalarType> kDtypeMap = {
      {"float16", torch::kFloat16},
      {"bfloat16", torch::kBFloat16},
      {"float32", torch::kFloat32},
      {"float64", torch::kFloat64},
      {"int8", torch::kInt8},
      {"int16", torch::kInt16},
      {"int32", torch::kInt32},
      {"int64", torch::kInt64},
      {"uint8", torch::kUInt8},
      {"bool", torch::kBool},
  };

  auto it = kDtypeMap.find(dtype_str);
  if (it == kDtypeMap.end()) {
    return std::nullopt;
  }
  return it->second;
}

inline torch::Tensor get_tensor_from_blob(const std::vector<int64_t>& dims,
                                          const torch::ScalarType dtype,
                                          const void* dev_addr) {
#if defined(USE_NPU)
  c10::DeviceType device_type = c10::DeviceType::PrivateUse1;
  torch::TensorOptions option =
      torch::TensorOptions().dtype(dtype).device(device_type);

  auto tensor = torch::empty({0}, option);
  auto address = const_cast<void*>(dev_addr);
  torch::DataPtr c10_data_ptr(address, address, [](void*) {}, tensor.device());

  size_t tensor_nbytes = at::detail::computeStorageNbytesContiguous(
      dims, tensor.dtype().itemsize());
  if (tensor_nbytes == 0) {
    return torch::empty(dims, option);
  }

  torch::Storage storage;
  auto fptr = c10::GetStorageImplCreate(device_type);
  auto allocator = c10::GetAllocator(device_type);

  storage = fptr(c10::StorageImpl::use_byte_size_t(),
                 c10::SymInt(tensor_nbytes),
                 std::move(c10_data_ptr),
                 allocator,
                 /*resizable=*/true);

  tensor.set_(storage, 0, dims);
  return tensor;
#elif defined(USE_CUDA) || defined(USE_MUSA) || defined(USE_MLU) || \
    defined(USE_DCU)
  auto options = torch::TensorOptions()
                     .dtype(dtype)
#if defined(USE_CUDA) || defined(USE_DCU)
                     .device(torch::kCUDA)
#else
                     .device(torch::kPrivateUse1)
#endif
                     .requires_grad(
                         /*requires_grad=*/false);
  return torch::from_blob(const_cast<void*>(dev_addr), dims, options);
#else
  LOG(FATAL)
      << "get_tensor_from_blob only supports NPU, CUDA, MUSA, MLU and DCU";
#endif
}

inline torch::Tensor get_tensor_from_blob(const std::vector<int64_t>& dims,
                                          const torch::ScalarType dtype,
                                          const void* dev_addr,
                                          const torch::Tensor& owner) {
#if defined(USE_CUDA) || defined(USE_MUSA) || defined(USE_MLU) || \
    defined(USE_DCU)
  CHECK(owner.defined())
      << "get_tensor_from_blob requires a valid owner tensor";

  auto options = torch::TensorOptions()
                     .dtype(dtype)
#if defined(USE_CUDA) || defined(USE_DCU)
                     .device(torch::kCUDA)
#else
                     .device(torch::kPrivateUse1)
#endif
                     .requires_grad(
                         /*requires_grad=*/false);
  auto owner_ref = owner;
  auto deleter = [owner_ref](void*) {};
  return torch::from_blob(const_cast<void*>(dev_addr), dims, deleter, options);
#else
  (void)owner;
  return get_tensor_from_blob(dims, dtype, dev_addr);
#endif
}

inline int32_t get_dtype_size(torch::ScalarType dtype) {
  return static_cast<int32_t>(torch::elementSize(dtype));
}

inline torch::ScalarType resolve_ssm_dtype(
    const std::string& mamba_ssm_dtype_str,
    torch::ScalarType default_dtype) {
  if (mamba_ssm_dtype_str.empty()) {
    return default_dtype;
  }
  auto parsed = try_get_scalar_type_from_string(mamba_ssm_dtype_str);
  if (parsed) {
    return parsed.value();
  }
  LOG(WARNING) << "Failed to parse mamba_ssm_dtype='" << mamba_ssm_dtype_str
               << "', falling back to default_dtype: " << default_dtype;
  return default_dtype;
}

inline int64_t resolve_ssm_dtype_size(const std::string& mamba_ssm_dtype_str,
                                      int64_t default_dtype_size) {
  if (mamba_ssm_dtype_str.empty()) {
    return default_dtype_size;
  }
  auto parsed = try_get_scalar_type_from_string(mamba_ssm_dtype_str);
  if (parsed) {
    return get_dtype_size(parsed.value());
  }
  LOG(WARNING) << "Failed to parse mamba_ssm_dtype='" << mamba_ssm_dtype_str
               << "', falling back to default dtype size";
  return default_dtype_size;
}

inline std::vector<int64_t> get_tensor_shape(const torch::Tensor& tensor) {
  if (!tensor.defined() || tensor.numel() == 0) {
    return {};
  }
  c10::IntArrayRef sizes = tensor.sizes();
  return std::vector<int64_t>(sizes.begin(), sizes.end());
}
}  // namespace xllm

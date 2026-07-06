/* Copyright 2025 The xLLM Authors. All Rights Reserved.
Copyright 2024 The ScaleLLM Authors. All Rights Reserved.

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

#include "state_dict.h"

#include <ATen/core/TensorBody.h>
#include <absl/strings/match.h>
#include <caffe2/serialize/inline_container.h>
#include <fcntl.h>
#include <glog/logging.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <torch/csrc/jit/serialization/import_read.h>
#include <torch/csrc/jit/serialization/storage_context.h>
#include <torch/torch.h>
#include <unistd.h>

#include <memory>

#include "core/util/env_var.h"
#include "safetensors/safetensors.h"

namespace xllm {
namespace {

const char* const ENV_MLOCK_ENABLED = "LLM_MLOCK_ENABLED";
const bool DEFAULT_MLOCK_ENABLED = false;

torch::ScalarType get_dtype(const Dtype& dtype) {
  switch (dtype) {
    case Dtype::BOOL:
      return torch::kBool;
    case Dtype::U8:
      return torch::kUInt8;
    case Dtype::I8:
      return torch::kInt8;
    case Dtype::F8_E5M2:
      return torch::kFloat8_e5m2;
    case Dtype::F8_E4M3:
      return torch::kFloat8_e4m3fn;
    case Dtype::I16:
      return torch::kInt16;
    case Dtype::F16:
      return torch::kFloat16;
    case Dtype::BF16:
      return torch::kBFloat16;
    case Dtype::I32:
      return torch::kInt32;
    case Dtype::F32:
      return torch::kFloat32;
    case Dtype::F64:
      return torch::kFloat64;
    case Dtype::I64:
      return torch::kInt64;
    case Dtype::U16:
    case Dtype::U32:
    case Dtype::U64:
    default:
      LOG(FATAL) << "Unsupported dtype " << static_cast<int>(dtype);
  }
  __builtin_unreachable();
}

std::vector<int64_t> get_sizes(const View* view) {
  std::vector<int64_t> sizes;
  sizes.reserve(view->rank);
  for (size_t i = 0; i < view->rank; i++) {
    sizes.push_back(view->shape[i]);
  }
  return sizes;
}

std::unique_ptr<MemoryMapping> create_memory_mapping(const char* weights_file) {
  auto mapping = std::make_unique<MemoryMapping>();

  mapping->fd = open(weights_file, O_RDONLY);
  if (mapping->fd == -1) {
    LOG(FATAL) << "Failed to open weight file: " << weights_file;
  }

  struct stat sb;
  if (fstat(mapping->fd, &sb) == -1) {
    LOG(FATAL) << "Failed to get file size for weight file: " << weights_file;
  }
  mapping->mapped_size = sb.st_size;

  mapping->mapped_addr = mmap(
      nullptr, mapping->mapped_size, PROT_READ, MAP_PRIVATE, mapping->fd, 0);
  if (mapping->mapped_addr == MAP_FAILED) {
    LOG(FATAL) << "Failed to map file: " << weights_file;
  }

  return mapping;
}

void destroy_memory_mapping(MemoryMapping* mapping) {
  if (mapping) {
    if (mapping->mapped_addr != MAP_FAILED) {
      munmap(mapping->mapped_addr, mapping->mapped_size);
    }
    if (mapping->fd != -1) {
      close(mapping->fd);
    }
    delete mapping;
  }
}
}  // namespace

StateDict::StateDict(std::unordered_map<std::string, torch::Tensor> dict,
                     const std::string& prefix)
    : dict_(std::move(dict)), prefix_(prefix) {}

torch::Tensor StateDict::get_tensor(const std::string& tensor_name) const {
  const auto it = dict_.find(tensor_name);
  if (it == dict_.end()) {
    return torch::Tensor{nullptr};
  }
  torch::Tensor tensor = it->second;
  if (tensor.dim() == 0) {
    tensor = tensor.reshape({tensor.numel()});
  }
  // apply transform function if exists
  return transform_func_ ? transform_func_(tensor_name, tensor) : tensor;
}

torch::Tensor StateDict::get_sharded_tensor(const std::string& tensor_name,
                                            int64_t dim,
                                            int rank,
                                            int world_size) const {
  CHECK(dim == 0 || dim == 1) << "Only support 1D or 2D sharding";
  CHECK(rank >= 0 && rank < world_size)
      << "Invalid rank " << rank << " for " << world_size << " shards";

  auto tensor = get_tensor(tensor_name);
  if (!tensor.defined()) {
    return tensor;
  }
  // chunk tensor along the dim
  const int64_t dim_size = tensor.size(dim);
  if (dim_size < world_size) {
    // too small to shard, return the whole tensor instead
    // TODO: assert dim_size >= world_size
    return tensor;
  }

  CHECK(dim_size % world_size == 0)
      << "can't devide tensor evenly on " << dim << " with dim: " << dim_size
      << " world_size: " << world_size;
  const auto chunks = tensor.chunk(world_size, dim);
  return chunks[rank];
}

// select all the tensors whose name starts with prefix.
StateDict StateDict::get_dict_with_prefix(const std::string& prefix) const {
  std::unordered_map<std::string, torch::Tensor> tensors;
  for (const auto& [name, tensor] : dict_) {
    if (absl::StartsWith(name, prefix)) {
      tensors[name.substr(prefix.length())] = tensor;
    }
  }
  return {std::move(tensors), prefix_ + prefix};
}

StateDict StateDict::get_dict_with_prefix(
    const std::vector<std::string>& prefixes) const {
  for (const auto& prefix : prefixes) {
    auto sd = get_dict_with_prefix(prefix);
    if (sd.size() > 0) {
      return sd;
    }
  }
  return StateDict({}, prefix_);
}

StateDict StateDict::get_dict_with_prefix(
    const std::string& prefix,
    TensorTransform transform_func) const {
  auto tensors = get_dict_with_prefix(prefix);
  tensors.transform_func_ = std::move(transform_func);
  return tensors;
}

bool StateDict::has(const std::string& tensor_name) const {
  const auto it = dict_.find(tensor_name);
  return it != dict_.end();
}

StateDictFromSafeTensor::StateDictFromSafeTensor(
    std::unique_ptr<MemoryMapping> mem_map,
    std::unordered_map<std::string, torch::Tensor> dict)
    : StateDict(std::move(dict)), mem_map_(std::move(mem_map)) {}

StateDictFromSafeTensor::~StateDictFromSafeTensor() {
  destroy_memory_mapping(mem_map_.release());
}

std::unique_ptr<StateDict> StateDictFromSafeTensor::load(
    const std::string& weights_file) {
  std::unique_ptr<MemoryMapping> mem_map = std::unique_ptr<MemoryMapping>(
      create_memory_mapping(weights_file.c_str()));

  if (!mem_map) {
    LOG(FATAL) << "Failed to create memory mapping for " << weights_file;
  }

  if (util::get_bool_env(ENV_MLOCK_ENABLED, DEFAULT_MLOCK_ENABLED)) {
    if (mlock(mem_map->mapped_addr, mem_map->mapped_size) == -1) {
      LOG(FATAL) << "Failed to lock memory for file: " << weights_file;
    }
  }

  const uint8_t* data = static_cast<const uint8_t*>(mem_map->mapped_addr);
  const size_t size = mem_map->mapped_size;

  std::unordered_map<std::string, torch::Tensor> dict;
  // safetensors
  Handle* handle = nullptr;
  CHECK(safetensors_deserialize(&handle, data, size) == Status::Ok)
      << "Failed to open safetensors file " << weights_file;

  const char* const* tensor_names = nullptr;
  size_t num_tensors = 0;
  CHECK(safetensors_names(handle, &tensor_names, &num_tensors) == Status::Ok)
      << "Failed to get tensor names from safetensors file " << weights_file;

  for (size_t i = 0; i < num_tensors; i++) {
    const char* tensor_name = tensor_names[i];
    View* tensor_view = nullptr;
    CHECK(safetensors_get_tensor(handle, &tensor_view, tensor_name) ==
          Status::Ok)
        << "Failed to get tensor " << tensor_name << " from safetensors file "
        << weights_file;

    const auto scalar_type = get_dtype(tensor_view->dtype);
    const void* tensor_data = data + tensor_view->start;
    const std::vector<int64_t> tensor_sizes = get_sizes(tensor_view);
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
    const auto tensor = at::from_blob(const_cast<void*>(tensor_data),
                                      tensor_sizes,
                                      torch::dtype(scalar_type));
    CHECK(safetensors_free_tensor(tensor_view) == Status::Ok)
        << "Failed to free tensor view";
    dict[tensor_name] = tensor;
  }
  CHECK(safetensors_free_names(tensor_names, num_tensors) == Status::Ok)
      << "Failed to free tensor names";
  CHECK(safetensors_destroy(handle) == Status::Ok)
      << "Failed to destroy safetensors handle";

  return std::make_unique<StateDictFromSafeTensor>(std::move(mem_map),
                                                   std::move(dict));
}

}  // namespace xllm

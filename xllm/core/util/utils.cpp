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

#include "util/utils.h"

#include <absl/strings/numbers.h>
#include <absl/strings/str_split.h>
#include <butil/base64.h>

#include <algorithm>
#include <boost/algorithm/string.hpp>

namespace xllm {
namespace util {

// Find the range of decode sequence indices (q_seq_lens == 1) in q_seq_lens
// Returns {start_index, end_index} of decode sequences,
// or {-1, -1} if no decode sequences found
std::pair<int, int> find_ones_indices(std::vector<int>& q_seq_lens) {
  int left = 0, right = q_seq_lens.size() - 1;
  int start_index = -1, end_index = -1;

  // Binary search for the start index of decode sequences (q_seq_lens == 1)
  while (left < right) {
    int mid = (left + right) / 2;
    if (q_seq_lens[mid] < 1) {
      left = mid + 1;
    } else {
      right = mid;
    }
  }
  if (q_seq_lens[left] == 1) {
    start_index = left;
  } else {
    return {start_index, end_index};  // No decode sequences found
  }

  left = 0;
  right = q_seq_lens.size() - 1;

  // Binary search for the end index of decode sequences (q_seq_lens == 1)
  while (left < right) {
    int mid = (left + right + 1) / 2;
    if (q_seq_lens[mid] > 1) {
      right = mid - 1;
    } else {
      left = mid;
    }
  }
  if (q_seq_lens[right] == 1) {
    end_index = right;
  }
  return {start_index, end_index};
}

torch::ScalarType parse_dtype(const std::string& dtype_str,
                              const std::optional<torch::Device>& device) {
  if (device.has_value() && device.value().is_cpu()) {
    // cpu only supports float32 for now
    return torch::kFloat32;
  }

  if (boost::iequals(dtype_str, "half") ||
      boost::iequals(dtype_str, "float16")) {
    return torch::kFloat16;
  }
  if (boost::iequals(dtype_str, "bfloat16")) {
    return torch::kBFloat16;
  }
  if ((boost::iequals(dtype_str, "float") ||
       boost::iequals(dtype_str, "float32"))) {
    return torch::kFloat;
  }

  if (dtype_str.empty() || boost::iequals(dtype_str, "auto")) {
    return torch::kFloat16;
  }
  CHECK(false) << "Unsupported dtype: " << dtype_str << " on device "
               << device.value();
}

std::optional<std::vector<uint32_t>> parse_batch_sizes(
    const std::string& batch_sizes_str) {
  if (batch_sizes_str.empty() || batch_sizes_str == "auto") {
    return std::nullopt;
  }

  // parse devices string
  const auto size_strs = absl::StrSplit(batch_sizes_str, ',');
  // remove duplicates
  std::unordered_set<uint32_t> sizes_set;
  for (const auto& size_str : size_strs) {
    uint32_t batch_size = 0;
    if (!absl::SimpleAtoi(size_str, &batch_size)) {
      LOG(ERROR) << "Failed to parse batch size: " << size_str;
      continue;
    }
    sizes_set.emplace(batch_size);
  }
  if (sizes_set.empty()) {
    return std::nullopt;
  }
  return std::vector<uint32_t>{sizes_set.begin(), sizes_set.end()};
}

bool match_suffix(const Slice<int32_t>& data, const Slice<int32_t>& suffix) {
  if (suffix.empty()) {
    return true;
  }

  const auto data_len = data.size();
  const auto suf_len = suffix.size();
  if (data_len < suf_len) {
    return false;
  }

  const auto data_start = data.data() + (data_len - suf_len);
  const auto data_end = data.data() + data_len;
  return std::equal(data_start, data_end, suffix.data());
}

std::vector<uint32_t> cal_vec_split_index(uint32_t vec_size,
                                          uint32_t part_num) {
  std::vector<uint32_t> split_index;
  split_index.reserve(part_num + 1);
  split_index.push_back(0);

  if (part_num == 1) {
    split_index.push_back(vec_size);
  } else {
    auto base = vec_size / part_num;
    auto remainder = vec_size % part_num;
    for (auto i = 0; i < part_num; ++i) {
      split_index.push_back(split_index[i] +
                            ((i < remainder) ? (base + 1) : base));
    }
  }
  return split_index;
}

torch::ScalarType convert_rec_type_to_torch(proto::DataType data_type) {
  // Future extensions go here.
  switch (data_type) {
    case proto::DataType::FLOAT:
      return torch::kFloat32;

    case proto::DataType::BFLOAT16:
      return torch::kBFloat16;

    case proto::DataType::BOOL:
      return torch::kBool;

    case proto::DataType::UINT8:
      return torch::kUInt8;

    case proto::DataType::INT8:
      return torch::kInt8;

    case proto::DataType::INT16:
      return torch::kInt16;

    default:
      LOG(FATAL) << "Unsupported data type: " << static_cast<int>(data_type);
  }
}

torch::Tensor convert_rec_tensor_to_torch(
    const proto::InferInputTensor& input_tensor) {
  std::vector<int64_t> shape;
  shape.reserve(input_tensor.shape_size());
  for (int i = 0; i < input_tensor.shape_size(); ++i) {
    shape.push_back(input_tensor.shape(i));
  }

  if (!input_tensor.has_contents()) {
    LOG(FATAL) << "Input tensor '" << input_tensor.name()
               << "' has no contents";
  }

  const auto& contents = input_tensor.contents();
  torch::ScalarType dtype = convert_rec_type_to_torch(input_tensor.data_type());

  switch (dtype) {
    case torch::kFloat32: {
      // Directly use protobuf's float array
      const auto& data = contents.fp32_contents();
      return torch::from_blob(
                 const_cast<float*>(data.data()),
                 shape,
                 torch::dtype(torch::kFloat32).requires_grad(false))
          .clone();  // Clone to ensure independent memory
    }
      // not support now.
      // case torch::kFloat16: {
      //   // Need type conversion (protobuf usually stores float16 as uint16)
      //   const auto& data = contents.bytes_contents();
      //   std::vector<at::Half> half_data;
      //   half_data.reserve(data.size());
      //   for (auto val : data) {
      //     half_data.push_back(static_cast<at::Half>(val));
      //   }
      //   return torch::tensor(half_data, torch::dtype(torch::kFloat16))
      //       .view(shape);
      // }

    case torch::kInt32: {
      const auto& data = contents.int_contents();
      return torch::from_blob(const_cast<int32_t*>(data.data()),
                              shape,
                              torch::dtype(torch::kInt32))
          .clone();
    }

    case torch::kInt64: {
      const auto& data = contents.int64_contents();
      return torch::from_blob(const_cast<int64_t*>(data.data()),
                              shape,
                              torch::dtype(torch::kInt64))
          .clone();
    }

    case torch::kBool: {
      const auto& data = contents.bool_contents();
      return torch::tensor(std::vector<uint8_t>(data.begin(), data.end()),
                           torch::dtype(torch::kBool))
          .view(shape);
    }

    default:
      LOG(FATAL) << "Unhandled data type conversion for: "
                 << static_cast<int>(dtype);
  }
}

torch::ScalarType datatype_proto_to_torch(const std::string& proto_datatype) {
  static const std::unordered_map<std::string, torch::ScalarType> kDatatypeMap =
      {{"BOOL", torch::kBool},
       {"INT32", torch::kInt},
       {"INT64", torch::kLong},
       {"UINT32", torch::kInt32},
       {"UINT64", torch::kInt64},
       {"FP32", torch::kFloat},
       {"FP64", torch::kDouble},
       {"BYTES", torch::kByte},
       {"FP16", torch::kHalf},
       {"BF16", torch::kBFloat16}};

  auto iter = kDatatypeMap.find(proto_datatype);
  if (iter == kDatatypeMap.end()) {
    LOG(FATAL)
        << "Unsupported proto datatype: " << proto_datatype
        << " (supported types: BOOL/INT32/INT64/UINT32/UINT64/FP32/FP64/BYTES)";
  }
  return iter->second;
}

namespace {
template <typename T>
const void* get_data_from_contents(const proto::TensorContents& contents,
                                   const std::string& datatype) {
  if constexpr (std::is_same_v<T, bool>) {
    if (contents.bool_contents().empty()) {
      LOG(ERROR) << "TensorContents.bool_contents is empty (datatype="
                 << datatype << ")";
      return nullptr;
    }
    return contents.bool_contents().data();
  } else if constexpr (std::is_same_v<T, int32_t>) {
    if (contents.int_contents().empty()) {
      LOG(ERROR) << "TensorContents.int_contents is empty (datatype="
                 << datatype << ")";
      return nullptr;
    }
    return contents.int_contents().data();
  } else if constexpr (std::is_same_v<T, int64_t>) {
    if (contents.int64_contents().empty()) {
      LOG(ERROR) << "TensorContents.int64_contents is empty (datatype="
                 << datatype << ")";
      return nullptr;
    }
    return contents.int64_contents().data();
  } else if constexpr (std::is_same_v<T, uint32_t>) {
    if (contents.uint_contents().empty()) {
      LOG(ERROR) << "TensorContents.uint_contents is empty (datatype="
                 << datatype << ")";
      return nullptr;
    }
    return contents.uint_contents().data();
  } else if constexpr (std::is_same_v<T, uint64_t>) {
    if (contents.uint64_contents().empty()) {
      LOG(ERROR) << "TensorContents.uint64_contents is empty (datatype="
                 << datatype << ")";
      return nullptr;
    }
    return contents.uint64_contents().data();
  } else if constexpr (std::is_same_v<T, float>) {
    if (contents.fp32_contents().empty()) {
      LOG(ERROR) << "TensorContents.fp32_contents is empty (datatype="
                 << datatype << ")";
      return nullptr;
    }
    return contents.fp32_contents().data();
  } else if constexpr (std::is_same_v<T, double>) {
    if (contents.fp64_contents().empty()) {
      LOG(ERROR) << "TensorContents.fp64_contents is empty (datatype="
                 << datatype << ")";
      return nullptr;
    }
    return contents.fp64_contents().data();
  } else if constexpr (std::is_same_v<T, uint8_t>) {
    return static_cast<const void*>(contents.bytes_contents().data());
  } else {
    LOG(FATAL) << "Unsupported data type for TensorContents: "
               << typeid(T).name();
    return nullptr;
  }
  return nullptr;
}

}  // namespace

std::string torch_datatype_to_proto(torch::ScalarType torch_dtype) {
  static const std::unordered_map<torch::ScalarType, std::string> kDatatypeMap =
      {{torch::kBool, "BOOL"},
       {torch::kInt, "INT32"},
       {torch::kLong, "INT64"},
       {torch::kInt32, "UINT32"},
       {torch::kInt64, "UINT64"},
       {torch::kFloat, "FP32"},
       {torch::kDouble, "FP64"},
       {torch::kByte, "BYTES"},
       {torch::kHalf, "FP16"},
       {torch::kBFloat16, "BF16"}};

  auto iter = kDatatypeMap.find(torch_dtype);
  if (iter == kDatatypeMap.end()) {
    LOG(ERROR) << "Unsupported torch datatype: " << torch::toString(torch_dtype)
               << " (supported types: "
                  "kBool/kInt/kLong/kInt32/kInt64/kFloat/kDouble/kByte)";
    return "";
  }
  return iter->second;
}

namespace {
template <typename T>
bool set_data_to_contents(proto::TensorContents* contents,
                          const torch::Tensor& tensor,
                          const std::string& proto_datatype) {
  torch::Tensor contig_tensor = tensor.contiguous().cpu();
  const T* data_ptr = contig_tensor.data_ptr<T>();
  size_t data_count = static_cast<size_t>(contig_tensor.numel());

  if (data_ptr == nullptr) {
    LOG(ERROR) << "Failed to get data pointer from torch Tensor (datatype="
               << proto_datatype << ")";
    return false;
  }

  if constexpr (std::is_same_v<T, bool>) {
    contents->mutable_bool_contents()->Reserve(data_count);
    for (size_t i = 0; i < data_count; ++i) {
      contents->add_bool_contents(data_ptr[i]);
    }
  } else if constexpr (std::is_same_v<T, int32_t>) {
    contents->mutable_int_contents()->Reserve(data_count);
    for (size_t i = 0; i < data_count; ++i) {
      contents->add_int_contents(data_ptr[i]);
    }
  } else if constexpr (std::is_same_v<T, int64_t>) {
    contents->mutable_int64_contents()->Reserve(data_count);
    for (size_t i = 0; i < data_count; ++i) {
      contents->add_int64_contents(data_ptr[i]);
    }
  } else if constexpr (std::is_same_v<T, uint32_t>) {
    contents->mutable_uint_contents()->Reserve(data_count);
    for (size_t i = 0; i < data_count; ++i) {
      contents->add_uint_contents(data_ptr[i]);
    }
  } else if constexpr (std::is_same_v<T, uint64_t>) {
    contents->mutable_uint64_contents()->Reserve(data_count);
    for (size_t i = 0; i < data_count; ++i) {
      contents->add_uint64_contents(data_ptr[i]);
    }
  } else if constexpr (std::is_same_v<T, float>) {
    contents->mutable_fp32_contents()->Reserve(data_count);
    for (size_t i = 0; i < data_count; ++i) {
      contents->add_fp32_contents(data_ptr[i]);
    }
  } else if constexpr (std::is_same_v<T, double>) {
    contents->mutable_fp64_contents()->Reserve(data_count);
    for (size_t i = 0; i < data_count; ++i) {
      contents->add_fp64_contents(data_ptr[i]);
    }
  } else if constexpr (std::is_same_v<T, uint8_t>) {
    const char* char_ptr = reinterpret_cast<const char*>(data_ptr);
    contents->set_bytes_contents(char_ptr, data_count * sizeof(T));
  } else {
    LOG(ERROR) << "Unsupported data type for TensorContents: "
               << typeid(T).name();
    return false;
  }

  return true;
}
}  // namespace

torch::Tensor proto_to_torch(const proto::Tensor& proto_tensor) {
  if (proto_tensor.datatype().empty()) {
    LOG(ERROR) << "Proto Tensor missing required field: datatype (e.g., "
                  "\"FP32\", \"INT64\")";
    return torch::Tensor();
  }
  if (proto_tensor.shape().empty()) {
    LOG(ERROR) << "Proto Tensor has empty shape (invalid tensor)";
    return torch::Tensor();
  }
  if (!proto_tensor.has_contents()) {
    LOG(ERROR)
        << "Proto Tensor missing required field: contents (TensorContents)";
    return torch::Tensor();
  }
  const auto& proto_contents = proto_tensor.contents();

  const std::string& proto_datatype = proto_tensor.datatype();
  torch::ScalarType torch_dtype = datatype_proto_to_torch(proto_datatype);
  const size_t element_size = torch::elementSize(torch_dtype);

  std::vector<int64_t> torch_shape;
  int64_t total_elements = 1;
  for (const auto& dim : proto_tensor.shape()) {
    if (dim <= 0) {
      LOG(ERROR) << "Proto Tensor has invalid dimension: " << dim
                 << " (must be positive, datatype=" << proto_datatype << ")";
      return torch::Tensor();
    }
    torch_shape.emplace_back(dim);
    total_elements *= dim;
  }
  torch::IntArrayRef tensor_shape(torch_shape);

  const void* data_ptr = nullptr;
  size_t data_count = 0;
  if (proto_datatype == "BOOL") {
    data_ptr = get_data_from_contents<bool>(proto_contents, proto_datatype);
    data_count = proto_contents.bool_contents_size();
  } else if (proto_datatype == "INT32") {
    data_ptr = get_data_from_contents<int32_t>(proto_contents, proto_datatype);
    data_count = proto_contents.int_contents_size();
  } else if (proto_datatype == "INT64") {
    data_ptr = get_data_from_contents<int64_t>(proto_contents, proto_datatype);
    data_count = proto_contents.int64_contents_size();
  } else if (proto_datatype == "UINT32") {
    data_ptr = get_data_from_contents<uint32_t>(proto_contents, proto_datatype);
    data_count = proto_contents.uint_contents_size();
  } else if (proto_datatype == "UINT64") {
    data_ptr = get_data_from_contents<uint64_t>(proto_contents, proto_datatype);
    data_count = proto_contents.uint64_contents_size();
  } else if (proto_datatype == "FP32") {
    data_ptr = get_data_from_contents<float>(proto_contents, proto_datatype);
    data_count = proto_contents.fp32_contents_size();
  } else if (proto_datatype == "FP64") {
    data_ptr = get_data_from_contents<double>(proto_contents, proto_datatype);
    data_count = proto_contents.fp64_contents_size();
  } else if (proto_datatype == "BF16") {
    data_ptr = get_data_from_contents<uint8_t>(proto_contents, proto_datatype);
    data_count = proto_contents.bytes_contents().size() /
                 static_cast<size_t>(sizeof(torch::BFloat16));
  } else if (proto_datatype == "FP16") {
    data_ptr = get_data_from_contents<uint8_t>(proto_contents, proto_datatype);
    data_count = proto_contents.bytes_contents().size() /
                 static_cast<size_t>(sizeof(torch::Half));
  } else if (proto_datatype == "BYTES") {
    data_ptr = get_data_from_contents<uint8_t>(proto_contents, proto_datatype);
    data_count = proto_contents.bytes_contents().size();
  }

  if (data_ptr == nullptr) {
    LOG(ERROR) << "Failed to get data from TensorContents (datatype="
               << proto_datatype << ")";
    return torch::Tensor();
  }
  if (data_count != static_cast<size_t>(total_elements)) {
    LOG(ERROR) << "Proto Tensor data count mismatch (datatype="
               << proto_datatype << "): "
               << "expected " << total_elements
               << " elements (shape=" << tensor_shape << "), "
               << "got " << data_count << " elements";
    return torch::Tensor();
  }

  torch::Tensor tensor =
      torch::from_blob(const_cast<void*>(data_ptr), tensor_shape, torch_dtype)
          .clone();
  return tensor;
}

bool torch_to_proto(const torch::Tensor& torch_tensor,
                    proto::Tensor* proto_tensor) {
  if (!torch_tensor.defined()) {
    LOG(ERROR) << "Input torch Tensor is undefined (null)";
    return false;
  }
  if (0 == torch_tensor.numel()) {
    LOG(ERROR) << "Input torch Tensor is empty (numel=0)";
    return false;
  }

  torch::ScalarType torch_dtype = torch_tensor.scalar_type();
  std::string proto_datatype = torch_datatype_to_proto(torch_dtype);
  if (proto_datatype.empty()) {
    return false;
  }
  proto_tensor->set_datatype(proto_datatype);

  proto_tensor->clear_shape();
  int64_t total_elements = 1;
  for (const auto& dim : torch_tensor.sizes()) {
    if (dim <= 0) {
      LOG(ERROR) << "Torch Tensor has invalid dimension: " << dim
                 << " (must be positive, datatype=" << proto_datatype << ")";
      return false;
    }
    proto_tensor->add_shape(dim);
    total_elements *= dim;
  }

  proto::TensorContents* proto_contents = proto_tensor->mutable_contents();
  proto_contents->Clear();

  bool data_set_success = false;
  switch (torch_dtype) {
    case torch::kBool:
      data_set_success = set_data_to_contents<bool>(
          proto_contents, torch_tensor, proto_datatype);
      break;
    case torch::kInt:
      data_set_success = set_data_to_contents<int32_t>(
          proto_contents, torch_tensor, proto_datatype);
      break;
    case torch::kLong:
      data_set_success = set_data_to_contents<int64_t>(
          proto_contents, torch_tensor, proto_datatype);
      break;
    case torch::kFloat:
      data_set_success = set_data_to_contents<float>(
          proto_contents, torch_tensor, proto_datatype);
      break;
    case torch::kDouble:
      data_set_success = set_data_to_contents<double>(
          proto_contents, torch_tensor, proto_datatype);
      break;
    case torch::kByte:
      data_set_success = set_data_to_contents<uint8_t>(
          proto_contents, torch_tensor, proto_datatype);
      break;
    case torch::kBFloat16: {
      // Need to convert bfloat16 to uint8_t for storage.
      // Ensure contiguous layout first — data_ptr() reads raw memory
      // which would be wrong for NHWC/ChannelsLast tensors.
      torch::Tensor contig_tensor = torch_tensor.contiguous();
      auto bfloat16_ptr = contig_tensor.data_ptr<torch::BFloat16>();
      uint8_t* uint8_ptr = reinterpret_cast<uint8_t*>(bfloat16_ptr);
      torch::Tensor uint8_tensor =
          torch::from_blob(uint8_ptr,
                           {static_cast<int64_t>(contig_tensor.numel() *
                                                 sizeof(torch::BFloat16))},
                           torch::dtype(torch::kUInt8));
      data_set_success = set_data_to_contents<uint8_t>(
          proto_contents, uint8_tensor, proto_datatype);
      break;
    }
    case torch::kHalf: {
      // Need to convert float16 to uint8_t for storage.
      // Ensure contiguous layout first — data_ptr() reads raw memory
      // which would be wrong for NHWC/ChannelsLast tensors.
      torch::Tensor contig_tensor = torch_tensor.contiguous();
      auto float16_ptr = contig_tensor.data_ptr<torch::Half>();
      uint8_t* uint8_ptr = reinterpret_cast<uint8_t*>(float16_ptr);
      torch::Tensor uint8_tensor = torch::from_blob(
          uint8_ptr,
          {static_cast<int64_t>(contig_tensor.numel() * sizeof(torch::Half))},
          torch::dtype(torch::kUInt8));
      data_set_success = set_data_to_contents<uint8_t>(
          proto_contents, uint8_tensor, proto_datatype);
      break;
    }
    default:
      LOG(ERROR) << "Unsupported torch dtype for serialization: "
                 << torch::toString(torch_dtype);
      data_set_success = false;
  }

  if (!data_set_success) {
    LOG(ERROR) << "Failed to set data to TensorContents (datatype="
               << proto_datatype << ")";
    return false;
  }

  size_t actual_count = 0;
  if (proto_datatype == "BOOL") {
    actual_count = proto_contents->bool_contents_size();
  } else if (proto_datatype == "INT32") {
    actual_count = proto_contents->int_contents_size();
  } else if (proto_datatype == "INT64") {
    actual_count = proto_contents->int64_contents_size();
  } else if (proto_datatype == "UINT32") {
    actual_count = proto_contents->uint_contents_size();
  } else if (proto_datatype == "UINT64") {
    actual_count = proto_contents->uint64_contents_size();
  } else if (proto_datatype == "FP32") {
    actual_count = proto_contents->fp32_contents_size();
  } else if (proto_datatype == "FP64") {
    actual_count = proto_contents->fp64_contents_size();
  } else if (proto_datatype == "STRING") {
    for (const auto& bytes : proto_contents->string_contents()) {
      actual_count += bytes.size() / sizeof(uint8_t);
    }
  } else if (proto_datatype == "BYTES") {
    actual_count += proto_contents->bytes_contents().size();
  } else if (proto_datatype == "BF16") {
    actual_count = proto_contents->bytes_contents().size() /
                   static_cast<size_t>(sizeof(torch::BFloat16));
  } else if (proto_datatype == "FP16") {
    actual_count = proto_contents->bytes_contents().size() /
                   static_cast<size_t>(sizeof(torch::Half));
  }

  if (actual_count != static_cast<size_t>(total_elements)) {
    LOG(WARNING) << "Torch Tensor data count mismatch (datatype="
                 << proto_datatype << "): "
                 << "expected " << total_elements << ", got " << actual_count;
  }

  return true;
}

// find the least power of 2 that is greater than or equal to x
int32_t ceil_pow2(int32_t n) {
  if (n <= 0) return 1;
  n--;
  n |= n >> 1;
  n |= n >> 2;
  n |= n >> 4;
  n |= n >> 8;
  n |= n >> 16;
  return n + 1;
}

}  // namespace util
}  // namespace xllm

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

#include <ATen/Tensor.h>
#include <dlfcn.h>
#include <glog/logging.h>
#include <limits.h>
#include <torch/extension.h>
#include <torch_npu/csrc/framework/utils/CalcuOpUtil.h>
#include <torch_npu/csrc/framework/utils/OpAdapter.h>
#include <unistd.h>

#include <array>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include "torch_npu/csrc/aten/NPUNativeFunctions.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/framework/OpCommand.h"
#include "torch_npu/csrc/framework/interface/EnvVariables.h"
#include "torch_npu/csrc/framework/utils/OpPreparation.h"

struct aclOpExecutor;
struct aclTensor;
struct aclScalar;
struct aclIntArray;
struct aclFloatArray;
struct aclBoolArray;
struct aclTensorList;

namespace xllm::kernel::npu::aclnn::detail {

using AclCreateTensorFn = ::aclTensor* (*)(const int64_t* view_dims,
                                           uint64_t view_dims_num,
                                           aclDataType data_type,
                                           const int64_t* stride,
                                           int64_t offset,
                                           aclFormat format,
                                           const int64_t* storage_dims,
                                           uint64_t storage_dims_num,
                                           void* tensor_data);
using AclCreateScalarFn = ::aclScalar* (*)(void* value, aclDataType data_type);
using AclCreateIntArrayFn = ::aclIntArray* (*)(const int64_t* value,
                                               uint64_t size);
using AclCreateFloatArrayFn = ::aclFloatArray* (*)(const float* value,
                                                   uint64_t size);
using AclCreateBoolArrayFn = ::aclBoolArray* (*)(const bool* value,
                                                 uint64_t size);
using AclCreateTensorListFn =
    ::aclTensorList* (*)(const ::aclTensor* const* value, uint64_t size);

using AclDestroyTensorFn = int (*)(const ::aclTensor* tensor);
using AclDestroyScalarFn = int (*)(const ::aclScalar* scalar);
using AclDestroyIntArrayFn = int (*)(const ::aclIntArray* array);
using AclDestroyFloatArrayFn = int (*)(const ::aclFloatArray* array);
using AclDestroyBoolArrayFn = int (*)(const ::aclBoolArray* array);
using AclDestroyTensorListFn = int (*)(const ::aclTensorList* array);

constexpr int32_t kHashBufSize = 8192;
constexpr int32_t kHashBufMaxSize = kHashBufSize + 1024;
extern thread_local char g_hashBuf[kHashBufSize];
extern thread_local int32_t g_hashOffset;

#ifdef XLLM_ACLNN_AT_ALL_SCALAR_TYPE_AND_ACL_DATATYPE_PAIR
#undef XLLM_ACLNN_AT_ALL_SCALAR_TYPE_AND_ACL_DATATYPE_PAIR
#endif
#define XLLM_ACLNN_AT_ALL_SCALAR_TYPE_AND_ACL_DATATYPE_PAIR(_) \
  _(at::ScalarType::Byte, ACL_UINT8)                           \
  _(at::ScalarType::Char, ACL_INT8)                            \
  _(at::ScalarType::Short, ACL_INT16)                          \
  _(at::ScalarType::Int, ACL_INT32)                            \
  _(at::ScalarType::Long, ACL_INT64)                           \
  _(at::ScalarType::Half, ACL_FLOAT16)                         \
  _(at::ScalarType::Float, ACL_FLOAT)                          \
  _(at::ScalarType::Double, ACL_DOUBLE)                        \
  _(at::ScalarType::ComplexHalf, ACL_DT_UNDEFINED)             \
  _(at::ScalarType::ComplexFloat, ACL_COMPLEX64)               \
  _(at::ScalarType::ComplexDouble, ACL_COMPLEX128)             \
  _(at::ScalarType::Bool, ACL_BOOL)                            \
  _(at::ScalarType::QInt8, ACL_DT_UNDEFINED)                   \
  _(at::ScalarType::QUInt8, ACL_DT_UNDEFINED)                  \
  _(at::ScalarType::QInt32, ACL_DT_UNDEFINED)                  \
  _(at::ScalarType::BFloat16, ACL_BF16)                        \
  _(at::ScalarType::QUInt4x2, ACL_DT_UNDEFINED)                \
  _(at::ScalarType::QUInt2x4, ACL_DT_UNDEFINED)                \
  _(at::ScalarType::Undefined, ACL_DT_UNDEFINED)               \
  _(at::ScalarType::NumOptions, ACL_DT_UNDEFINED)

inline std::vector<std::string> split_str(std::string s,
                                          const std::string& delimiter) {
  std::string::size_type end = s.find(delimiter);
  std::vector<std::string> path_list;
  while (end != std::string::npos) {
    path_list.push_back(s.substr(0, end));
    s.erase(0, end + 1);
    end = s.find(delimiter);
  }
  path_list.push_back(s);
  return path_list;
}

inline bool is_file_exist(const std::string& path) {
  if (path.empty() || path.size() > PATH_MAX) {
    return false;
  }
  return access(path.c_str(), F_OK) == 0;
}

inline std::string real_path(const std::string& path) {
  if (path.empty() || path.size() > PATH_MAX) {
    return {};
  }
  char real_path_buffer[PATH_MAX] = {0};
  if (realpath(path.c_str(), real_path_buffer) == nullptr) {
    return {};
  }
  return std::string(real_path_buffer);
}

inline std::vector<std::string> get_custom_lib_path() {
  const char* ascend_custom_opp_path = std::getenv("ASCEND_CUSTOM_OPP_PATH");
  std::vector<std::string> custom_lib_path_list;

  if (ascend_custom_opp_path == nullptr) {
    ASCEND_LOGW("ASCEND_CUSTOM_OPP_PATH is not exists");
    return {};
  }

  std::string ascend_custom_opp_path_str(ascend_custom_opp_path);
  // split string with ":"
  custom_lib_path_list = split_str(ascend_custom_opp_path_str, ":");
  if (custom_lib_path_list.empty()) {
    return {};
  }
  for (auto& it : custom_lib_path_list) {
    it += "/op_api/lib/";
  }

  return custom_lib_path_list;
}

inline std::vector<std::string> get_default_custom_lib_path() {
  const char* ascend_opp_path = std::getenv("ASCEND_OPP_PATH");
  std::vector<std::string> default_vendors_list;

  if (ascend_opp_path == nullptr) {
    ASCEND_LOGW("ASCEND_OPP_PATH is not exists");
    return {};
  }

  std::string vendors_path(ascend_opp_path);
  vendors_path = vendors_path + "/vendors";
  std::string vendors_config_file = real_path(vendors_path + "/config.ini");
  if (vendors_config_file.empty()) {
    ASCEND_LOGW("config.ini is not exists");
    return {};
  }

  if (!is_file_exist(vendors_config_file)) {
    ASCEND_LOGW("config.ini is not exists or the path length is more than %d",
                PATH_MAX);
    return {};
  }

  std::ifstream ifs(vendors_config_file);
  std::string line;
  while (std::getline(ifs, line)) {
    if (line.find("load_priority=") == 0) {
      break;
    }
  }
  std::string head = "load_priority=";
  line.erase(0, head.length());

  // split string with ","
  default_vendors_list = split_str(line, ",");
  if (default_vendors_list.empty()) {
    return {};
  }
  for (auto& it : default_vendors_list) {
    it = real_path(vendors_path + "/" + it + "/op_api/lib/");
  }

  return default_vendors_list;
}

const std::vector<std::string> g_custom_lib_path = get_custom_lib_path();
const std::vector<std::string> g_default_custom_lib_path =
    get_default_custom_lib_path();

constexpr aclDataType kATenScalarTypeToAclDataTypeTable
    [static_cast<int64_t>(at::ScalarType::NumOptions) + 1] = {
#define DEFINE_ENUM(_1, n) n,
        XLLM_ACLNN_AT_ALL_SCALAR_TYPE_AND_ACL_DATATYPE_PAIR(DEFINE_ENUM)
#undef DEFINE_ENUM
};

#ifdef XLLM_ACLNN_MEMCPY_TO_BUF
#undef XLLM_ACLNN_MEMCPY_TO_BUF
#endif
#define XLLM_ACLNN_MEMCPY_TO_BUF(data_expression, size_expression)         \
  if (g_hashOffset + (size_expression) > kHashBufSize) {                   \
    g_hashOffset = kHashBufMaxSize;                                        \
    return;                                                                \
  }                                                                        \
  std::memcpy(g_hashBuf + g_hashOffset, data_expression, size_expression); \
  g_hashOffset += size_expression;

inline const char* get_op_api_lib_name() { return "libopapi.so"; }

inline const char* get_cust_op_api_lib_name() { return "libcust_opapi.so"; }

inline void* get_op_api_func_addr_in_lib(void* handler,
                                         const char* lib_name,
                                         const char* api_name) {
  void* func_addr = dlsym(handler, api_name);
  if (func_addr == nullptr) {
    ASCEND_LOGW(
        "dlsym %s from %s failed, error:%s.", api_name, lib_name, dlerror());
  }
  return func_addr;
}

inline void* get_op_api_lib_handler(const char* lib_name) {
  void* handler = dlopen(lib_name, RTLD_LAZY);
  if (handler == nullptr) {
    ASCEND_LOGW("dlopen %s failed, error:%s.", lib_name, dlerror());
  }
  return handler;
}

inline void* get_op_api_func_addr(const char* api_name) {
  if (!g_custom_lib_path.empty()) {
    for (auto& it : g_custom_lib_path) {
      std::string cust_opapi_lib =
          real_path(it + "/" + get_cust_op_api_lib_name());
      if (cust_opapi_lib.empty()) {
        break;
      }
      void* cust_op_api_handler =
          get_op_api_lib_handler(cust_opapi_lib.c_str());
      if (cust_op_api_handler != nullptr) {
        void* func_addr = get_op_api_func_addr_in_lib(
            cust_op_api_handler, get_cust_op_api_lib_name(), api_name);
        if (func_addr != nullptr) {
          ASCEND_LOGI("%s is found in %s.", api_name, cust_opapi_lib.c_str());
          return func_addr;
        }
      }
    }
    ASCEND_LOGI("%s is not in custom lib.", api_name);
  }

  if (!g_default_custom_lib_path.empty()) {
    for (auto& it : g_default_custom_lib_path) {
      std::string default_cust_opapi_lib =
          real_path(it + "/" + get_cust_op_api_lib_name());
      if (default_cust_opapi_lib.empty()) {
        break;
      }
      void* cust_op_api_handler =
          get_op_api_lib_handler(default_cust_opapi_lib.c_str());
      if (cust_op_api_handler != nullptr) {
        void* func_addr = get_op_api_func_addr_in_lib(
            cust_op_api_handler, get_cust_op_api_lib_name(), api_name);
        if (func_addr != nullptr) {
          ASCEND_LOGI(
              "%s is found in %s.", api_name, default_cust_opapi_lib.c_str());
          return func_addr;
        }
      }
    }
    ASCEND_LOGI("%s is not in default custom lib.", api_name);
  }

  static void* op_api_handler = get_op_api_lib_handler(get_op_api_lib_name());
  if (op_api_handler == nullptr) {
    return nullptr;
  }
  return get_op_api_func_addr_in_lib(
      op_api_handler, get_op_api_lib_name(), api_name);
}

template <typename Func>
inline Func get_op_api_func(const char* api_name) {
  return reinterpret_cast<Func>(get_op_api_func_addr(api_name));
}

inline c10::Scalar convert_tensor_to_scalar(const at::Tensor& tensor) {
  c10::Scalar exp_scalar;
  const at::Tensor* acl_input = &tensor;
  if (acl_input->scalar_type() == at::ScalarType::Double) {
    double value = *reinterpret_cast<const double*>(acl_input->data_ptr());
    exp_scalar = c10::Scalar(value);
  } else if (acl_input->scalar_type() == at::ScalarType::Long) {
    int64_t value = *reinterpret_cast<const int64_t*>(acl_input->data_ptr());
    exp_scalar = c10::Scalar(value);
  } else if (acl_input->scalar_type() == at::ScalarType::Float) {
    float value = *reinterpret_cast<const float*>(acl_input->data_ptr());
    exp_scalar = c10::Scalar(value);
  } else if (acl_input->scalar_type() == at::ScalarType::Int) {
    int32_t value = *reinterpret_cast<const int32_t*>(acl_input->data_ptr());
    exp_scalar = c10::Scalar(value);
  } else if (acl_input->scalar_type() == at::ScalarType::Half) {
    c10::Half value =
        *reinterpret_cast<const c10::Half*>(acl_input->data_ptr());
    exp_scalar = c10::Scalar(value);
  } else if (acl_input->scalar_type() == at::ScalarType::Bool) {
    int8_t value = *reinterpret_cast<const int8_t*>(acl_input->data_ptr());
    exp_scalar = c10::Scalar(value);
  } else if (acl_input->scalar_type() == at::ScalarType::ComplexDouble) {
    c10::complex<double> value =
        *reinterpret_cast<const c10::complex<double>*>(acl_input->data_ptr());
    exp_scalar = c10::Scalar(value);
  } else if (acl_input->scalar_type() == at::ScalarType::ComplexFloat) {
    c10::complex<float> value =
        *reinterpret_cast<const c10::complex<float>*>(acl_input->data_ptr());
    exp_scalar = c10::Scalar(value);
  } else if (acl_input->scalar_type() == at::ScalarType::BFloat16) {
    c10::BFloat16 value =
        *reinterpret_cast<const c10::BFloat16*>(acl_input->data_ptr());
    exp_scalar = c10::Scalar(value);
  }
  return exp_scalar;
}

inline at::Tensor copy_tensor_host_to_device(const at::Tensor& cpu_tensor) {
  at::Tensor cpu_pin_mem_tensor = cpu_tensor.pin_memory();
  int32_t device_index = 0;
  return cpu_pin_mem_tensor.to(
      c10::Device(torch_npu::utils::get_npu_device_type(), device_index),
      cpu_pin_mem_tensor.scalar_type(),
      true,
      true);
}

inline at::Tensor copy_scalar_to_device(const c10::Scalar& cpu_scalar,
                                        at::ScalarType scalar_data_type) {
  return copy_tensor_host_to_device(
      scalar_to_tensor(cpu_scalar).to(scalar_data_type));
}

inline aclTensor* convert_type(const at::Tensor& at_tensor) {
  static const auto acl_create_tensor =
      get_op_api_func<AclCreateTensorFn>("aclCreateTensor");
  if (acl_create_tensor == nullptr) {
    return nullptr;
  }

  if (!at_tensor.defined()) {
    return nullptr;
  }
  at::ScalarType scalar_data_type = at_tensor.scalar_type();
  aclDataType acl_data_type =
      kATenScalarTypeToAclDataTypeTable[static_cast<int64_t>(scalar_data_type)];
  CHECK(acl_data_type != ACL_DT_UNDEFINED)
      << std::string(c10::toString(scalar_data_type))
      << " has not been supported";
  c10::SmallVector<int64_t, 5> storage_dims;
  // if acl_data_type is ACL_STRING, storage_dims is empty.
  int64_t item_size = at_tensor.itemsize();
  if (item_size == 0) {
    AT_ERROR("When ConvertType, tensor item size of cannot be zero.");
    return nullptr;
  }
  if (acl_data_type != ACL_STRING) {
    storage_dims.push_back(at_tensor.storage().nbytes() / item_size);
  }

  const auto dim_num = at_tensor.sizes().size();
  aclFormat format = ACL_FORMAT_ND;
  switch (dim_num) {
    case 3:
      format = ACL_FORMAT_NCL;
      break;
    case 4:
      format = ACL_FORMAT_NCHW;
      break;
    case 5:
      format = ACL_FORMAT_NCDHW;
      break;
    default:
      format = ACL_FORMAT_ND;
  }

  if (at_tensor.unsafeGetTensorImpl()->is_wrapped_number()) {
    c10::Scalar exp_scalar = convert_tensor_to_scalar(at_tensor);
    at::Tensor acl_input = copy_scalar_to_device(exp_scalar, scalar_data_type);
    return acl_create_tensor(acl_input.sizes().data(),
                             acl_input.sizes().size(),
                             acl_data_type,
                             acl_input.strides().data(),
                             acl_input.storage_offset(),
                             format,
                             storage_dims.data(),
                             storage_dims.size(),
                             const_cast<void*>(acl_input.storage().data()));
  }

  aclTensor* acl_tensor =
      acl_create_tensor(at_tensor.sizes().data(),
                        at_tensor.sizes().size(),
                        acl_data_type,
                        at_tensor.strides().data(),
                        at_tensor.storage_offset(),
                        format,
                        storage_dims.data(),
                        storage_dims.size(),
                        const_cast<void*>(at_tensor.storage().data()));
  return acl_tensor;
}

inline aclScalar* convert_type(const at::Scalar& at_scalar) {
  static const auto acl_create_scalar =
      get_op_api_func<AclCreateScalarFn>("aclCreateScalar");
  if (acl_create_scalar == nullptr) {
    return nullptr;
  }

  at::ScalarType scalar_data_type = at_scalar.type();
  aclDataType acl_data_type =
      kATenScalarTypeToAclDataTypeTable[static_cast<int64_t>(scalar_data_type)];
  CHECK(acl_data_type != ACL_DT_UNDEFINED)
      << std::string(c10::toString(scalar_data_type))
      << " has not been supported";
  aclScalar* acl_scalar = nullptr;
  switch (scalar_data_type) {
    case at::ScalarType::Double: {
      double value = at_scalar.toDouble();
      acl_scalar = acl_create_scalar(&value, acl_data_type);
      break;
    }
    case at::ScalarType::Long: {
      int64_t value = at_scalar.toLong();
      acl_scalar = acl_create_scalar(&value, acl_data_type);
      break;
    }
    case at::ScalarType::Bool: {
      bool value = at_scalar.toBool();
      acl_scalar = acl_create_scalar(&value, acl_data_type);
      break;
    }
    case at::ScalarType::ComplexDouble: {
      c10::complex<double> value = at_scalar.toComplexDouble();
      acl_scalar = acl_create_scalar(&value, acl_data_type);
      break;
    }
    default:
      acl_scalar = nullptr;
      break;
  }
  return acl_scalar;
}

inline aclIntArray* convert_type(const at::IntArrayRef& at_array) {
  static const auto acl_create_int_array =
      get_op_api_func<AclCreateIntArrayFn>("aclCreateIntArray");
  if (acl_create_int_array == nullptr) {
    return nullptr;
  }
  aclIntArray* array = acl_create_int_array(at_array.data(), at_array.size());
  return array;
}

template <std::size_t N>
inline aclBoolArray* convert_type(const std::array<bool, N>& value) {
  static const auto acl_create_bool_array =
      get_op_api_func<AclCreateBoolArrayFn>("aclCreateBoolArray");
  if (acl_create_bool_array == nullptr) {
    return nullptr;
  }

  aclBoolArray* array = acl_create_bool_array(value.data(), value.size());
  return array;
}

inline aclBoolArray* convert_type(const at::ArrayRef<bool>& value) {
  static const auto acl_create_bool_array =
      get_op_api_func<AclCreateBoolArrayFn>("aclCreateBoolArray");
  if (acl_create_bool_array == nullptr) {
    return nullptr;
  }

  aclBoolArray* array = acl_create_bool_array(value.data(), value.size());
  return array;
}

inline aclTensorList* convert_type(const at::TensorList& at_tensor_list) {
  static const auto acl_create_tensor_list =
      get_op_api_func<AclCreateTensorListFn>("aclCreateTensorList");
  if (acl_create_tensor_list == nullptr) {
    return nullptr;
  }

  std::vector<const aclTensor*> tensor_list(at_tensor_list.size());
  for (size_t index = 0; index < at_tensor_list.size(); ++index) {
    tensor_list[index] = convert_type(at_tensor_list[index]);
  }
  aclTensorList* acl_tensor_list =
      acl_create_tensor_list(tensor_list.data(), tensor_list.size());
  return acl_tensor_list;
}

inline aclTensorList* convert_type(
    const c10::optional<at::TensorList>& opt_tensor_list) {
  if (opt_tensor_list.has_value()) {
    return convert_type(opt_tensor_list.value());
  }
  return nullptr;
}

inline aclTensor* convert_type(const c10::optional<at::Tensor>& opt_tensor) {
  if (opt_tensor.has_value() && opt_tensor.value().defined()) {
    return convert_type(opt_tensor.value());
  }
  return nullptr;
}

inline aclIntArray* convert_type(
    const c10::optional<at::IntArrayRef>& opt_array) {
  if (opt_array.has_value()) {
    return convert_type(opt_array.value());
  }
  return nullptr;
}

inline aclScalar* convert_type(const c10::optional<at::Scalar>& opt_scalar) {
  if (opt_scalar.has_value()) {
    return convert_type(opt_scalar.value());
  }
  return nullptr;
}

inline aclDataType convert_type(const at::ScalarType scalar_type) {
  return kATenScalarTypeToAclDataTypeTable[static_cast<int64_t>(scalar_type)];
}

template <typename T>
T convert_type(T value) {
  return value;
}

template <typename Tuple, size_t... I>
auto convert_to_op_api_func(const Tuple& params,
                            void* op_api_addr,
                            std::index_sequence<I...>) {
  using OpApiFunc =
      int (*)(typename std::decay<decltype(std::get<I>(params))>::type...);
  OpApiFunc func = reinterpret_cast<OpApiFunc>(op_api_addr);
  return func;
}

template <typename Tuple>
auto convert_to_op_api_func(const Tuple& params, void* op_api_addr) {
  static constexpr auto size = std::tuple_size<Tuple>::value;
  return convert_to_op_api_func(
      params, op_api_addr, std::make_index_sequence<size>{});
}

inline void release(aclTensor* p) {
  static const auto acl_destroy_tensor =
      get_op_api_func<AclDestroyTensorFn>("aclDestroyTensor");
  if (acl_destroy_tensor == nullptr) {
    return;
  }
  acl_destroy_tensor(p);
}

inline void release(aclScalar* p) {
  static const auto acl_destroy_scalar =
      get_op_api_func<AclDestroyScalarFn>("aclDestroyScalar");
  if (acl_destroy_scalar == nullptr) {
    return;
  }
  acl_destroy_scalar(p);
}

inline void release(aclIntArray* p) {
  static const auto acl_destroy_int_array =
      get_op_api_func<AclDestroyIntArrayFn>("aclDestroyIntArray");
  if (acl_destroy_int_array == nullptr) {
    return;
  }

  acl_destroy_int_array(p);
}

inline void release(aclBoolArray* p) {
  static const auto acl_destroy_bool_array =
      get_op_api_func<AclDestroyBoolArrayFn>("aclDestroyBoolArray");
  if (acl_destroy_bool_array == nullptr) {
    return;
  }

  acl_destroy_bool_array(p);
}

inline void release(aclTensorList* p) {
  if (p == nullptr) {
    return;
  }
  static const auto acl_destroy_tensor_list =
      get_op_api_func<AclDestroyTensorListFn>("aclDestroyTensorList");
  if (acl_destroy_tensor_list == nullptr) {
    return;
  }

  acl_destroy_tensor_list(p);
}

template <typename T>
void release(T value) {
  (void)value;
}

template <typename Tuple, size_t... I>
void call_release(Tuple t, std::index_sequence<I...>) {
  (void)std::initializer_list<int32_t>{(release(std::get<I>(t)), 0)...};
}

template <typename Tuple>
void release_convert_types(Tuple& t) {
  static constexpr auto size = std::tuple_size<Tuple>::value;
  call_release(t, std::make_index_sequence<size>{});
}

template <typename... Ts>
constexpr auto convert_types(Ts&... args) {
  return std::make_tuple(convert_type(args)...);
}

template <typename Function, typename Tuple, size_t... I>
auto call(Function f, Tuple t, std::index_sequence<I...>) {
  return f(std::get<I>(t)...);
}

template <typename Function, typename Tuple>
auto call(Function f, Tuple t) {
  static constexpr auto size = std::tuple_size<Tuple>::value;
  return call(f, t, std::make_index_sequence<size>{});
}

template <std::size_t N>
void add_param_to_buf(const std::array<bool, N>& value) {
  XLLM_ACLNN_MEMCPY_TO_BUF(value.data(), value.size() * sizeof(bool));
}

template <typename T>
void add_param_to_buf(const T& value) {
  XLLM_ACLNN_MEMCPY_TO_BUF(&value, sizeof(T));
}

void add_param_to_buf(const at::Tensor&);
void add_param_to_buf(const at::Scalar&);
void add_param_to_buf(const at::IntArrayRef&);
void add_param_to_buf(const at::ArrayRef<bool>&);
void add_param_to_buf(const at::TensorList&);
void add_param_to_buf(const c10::optional<at::Tensor>&);
void add_param_to_buf(const c10::optional<at::IntArrayRef>&);
void add_param_to_buf(const c10::optional<at::Scalar>&);
void add_param_to_buf(const at::ScalarType);
void add_param_to_buf(const std::string&);
void add_param_to_buf();

template <typename T, typename... Args>
void add_param_to_buf(const T& arg, Args&... args) {
  add_param_to_buf(arg);
  add_param_to_buf(args...);
}

uint64_t calc_hash_id();

using InitHugeMemThreadLocalFn = int (*)(void*, bool);
using UnInitHugeMemThreadLocalFn = void (*)(void*, bool);
using ReleaseHugeMemFn = void (*)(void*, bool);

}  // namespace xllm::kernel::npu::aclnn::detail

#define EXEC_NPU_CMD(aclnn_api, ...)                                           \
  do {                                                                         \
    static const auto get_workspace_size_func_addr =                           \
        ::xllm::kernel::npu::aclnn::detail::get_op_api_func_addr(              \
            #aclnn_api "GetWorkspaceSize");                                    \
    static const auto op_api_func_addr =                                       \
        ::xllm::kernel::npu::aclnn::detail::get_op_api_func_addr(#aclnn_api);  \
    static const auto init_mem_addr =                                          \
        ::xllm::kernel::npu::aclnn::detail::get_op_api_func_addr(              \
            "InitHugeMemThreadLocal");                                         \
    static const auto uninit_mem_addr =                                        \
        ::xllm::kernel::npu::aclnn::detail::get_op_api_func_addr(              \
            "UnInitHugeMemThreadLocal");                                       \
    static const auto release_mem_addr =                                       \
        ::xllm::kernel::npu::aclnn::detail::get_op_api_func_addr(              \
            "ReleaseHugeMem");                                                 \
    CHECK(get_workspace_size_func_addr != nullptr &&                           \
          op_api_func_addr != nullptr)                                         \
        << #aclnn_api << " or " << #aclnn_api "GetWorkspaceSize" << " not in " \
        << ::xllm::kernel::npu::aclnn::detail::get_op_api_lib_name()           \
        << ", or "                                                             \
        << ::xllm::kernel::npu::aclnn::detail::get_op_api_lib_name()           \
        << "not found.";                                                       \
    auto acl_stream = c10_npu::getCurrentNPUStream().stream(false);            \
    uint64_t workspace_size = 0;                                               \
    uint64_t* workspace_size_addr = &workspace_size;                           \
    ::aclOpExecutor* executor = nullptr;                                       \
    ::aclOpExecutor** executor_addr = &executor;                               \
    ::xllm::kernel::npu::aclnn::detail::InitHugeMemThreadLocalFn               \
        init_mem_func = reinterpret_cast<                                      \
            ::xllm::kernel::npu::aclnn::detail::InitHugeMemThreadLocalFn>(     \
            init_mem_addr);                                                    \
    ::xllm::kernel::npu::aclnn::detail::UnInitHugeMemThreadLocalFn             \
        uninit_mem_func = reinterpret_cast<                                    \
            ::xllm::kernel::npu::aclnn::detail::UnInitHugeMemThreadLocalFn>(   \
            uninit_mem_addr);                                                  \
    if (init_mem_func) {                                                       \
      init_mem_func(nullptr, false);                                           \
    }                                                                          \
    auto converted_params = ::xllm::kernel::npu::aclnn::detail::convert_types( \
        __VA_ARGS__, workspace_size_addr, executor_addr);                      \
    static auto get_workspace_size_func =                                      \
        ::xllm::kernel::npu::aclnn::detail::convert_to_op_api_func(            \
            converted_params, get_workspace_size_func_addr);                   \
    auto workspace_status = ::xllm::kernel::npu::aclnn::detail::call(          \
        get_workspace_size_func, converted_params);                            \
    CHECK(workspace_status == 0)                                               \
        << "call " #aclnn_api " failed, detail:" << aclGetRecentErrMsg();      \
    void* workspace_addr = nullptr;                                            \
    at::Tensor workspace_tensor;                                               \
    if (workspace_size != 0) {                                                 \
      at::TensorOptions options =                                              \
          at::TensorOptions(torch_npu::utils::get_npu_device_type());          \
      workspace_tensor = at::empty({static_cast<int64_t>(workspace_size)},     \
                                   options.dtype(at::kByte));                  \
      workspace_addr = const_cast<void*>(workspace_tensor.storage().data());   \
    }                                                                          \
    auto acl_call = [=]() -> int {                                             \
      using OpApiFunc =                                                        \
          int (*)(void*, uint64_t, ::aclOpExecutor*, const aclrtStream);       \
      OpApiFunc op_api_func = reinterpret_cast<OpApiFunc>(op_api_func_addr);   \
      auto api_ret =                                                           \
          op_api_func(workspace_addr, workspace_size, executor, acl_stream);   \
      CHECK(api_ret == 0) << "call " #aclnn_api " failed, detail:"             \
                          << aclGetRecentErrMsg();                             \
      ::xllm::kernel::npu::aclnn::detail::release_convert_types(               \
          converted_params);                                                   \
      ::xllm::kernel::npu::aclnn::detail::ReleaseHugeMemFn release_mem_func =  \
          reinterpret_cast<                                                    \
              ::xllm::kernel::npu::aclnn::detail::ReleaseHugeMemFn>(           \
              release_mem_addr);                                               \
      if (release_mem_func) {                                                  \
        release_mem_func(nullptr, false);                                      \
      }                                                                        \
      return api_ret;                                                          \
    };                                                                         \
    at_npu::native::OpCommand cmd;                                             \
    cmd.Name(#aclnn_api);                                                      \
    cmd.SetCustomHandler(acl_call);                                            \
    cmd.Run();                                                                 \
    if (uninit_mem_func) {                                                     \
      uninit_mem_func(nullptr, false);                                         \
    }                                                                          \
  } while (false)

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

#pragma once

#include <mutex>

namespace xllm {

#define PROPERTY(T, property)                                                 \
 public:                                                                      \
  [[nodiscard]] const T& property() const& noexcept { return property##_; }   \
  [[nodiscard]] T& property() & noexcept { return property##_; }              \
  [[nodiscard]] T&& property() && noexcept { return std::move(property##_); } \
                                                                              \
  auto property(const T& value) & -> decltype(*this) {                        \
    property##_ = value;                                                      \
    return *this;                                                             \
  }                                                                           \
                                                                              \
  auto property(T&& value) & -> decltype(*this) {                             \
    property##_ = std::move(value);                                           \
    return *this;                                                             \
  }                                                                           \
                                                                              \
  void property(const T& value) && = delete;                                  \
  void property(T&& value) && = delete;                                       \
                                                                              \
  T property##_

#define SAFE_CONCAT(a, b) (a##b)

#ifndef UNUSED_PARAMETER
#define UNUSED_PARAMETER(x) ((void)(x))
#endif

#ifndef DISALLOW_COPY_AND_ASSIGN
#define DISALLOW_COPY_AND_ASSIGN(TypeName) \
  TypeName(const TypeName&) = delete;      \
  void operator=(const TypeName&) = delete
#endif

// Define a macro to simplify adding elements from a vector to a repeated field
#define ADD_VECTOR_TO_PROTO(proto_field, vec) \
  do {                                        \
    proto_field->Reserve(vec.size());         \
    for (const auto& value : vec) {           \
      *proto_field->Add() = value;            \
    }                                         \
  } while (0)

#define TORCH_TENSOR_VEC_TO_PROTO_TENSOR_LIST(proto_field, torch_tensor_vec) \
  do {                                                                       \
    proto_field->mutable_tensors()->Reserve(torch_tensor_vec.size());        \
    for (const auto& torch_tensor : torch_tensor_vec) {                      \
      proto::Tensor* pb_tensor = proto_field->add_tensors();                 \
      if (!util::torch_to_proto(torch_tensor, pb_tensor)) {                  \
        LOG(ERROR)                                                           \
            << "Failed to convert torch Tensor to PB Tensor (list item)";    \
      }                                                                      \
    }                                                                        \
  } while (0)

#define CALLBACK_WITH_ERROR_ARGS2(CODE, MSG) callback(Status{CODE, MSG})
#define CALLBACK_WITH_ERROR_ARGS4(CODE, MSG, ID, TARGET_XSERVICE_ADDR) \
  callback({Status{CODE, MSG}, ID, TARGET_XSERVICE_ADDR})
#define CALLBACK_WITH_ERROR_INVALID_ARITY(...)                        \
  static_assert(false,                                                \
                "CALLBACK_WITH_ERROR expects 2 or 4 args: "           \
                "CALLBACK_WITH_ERROR(code, msg) or "                  \
                "CALLBACK_WITH_ERROR(code, msg, service_request_id, " \
                "target_xservice_addr)")

#define GET_MACRO(_1, _2, _3, _4, _5, NAME, ...) NAME
#define CALLBACK_WITH_ERROR(...)               \
  GET_MACRO(__VA_ARGS__,                       \
            CALLBACK_WITH_ERROR_INVALID_ARITY, \
            CALLBACK_WITH_ERROR_ARGS4,         \
            CALLBACK_WITH_ERROR_INVALID_ARITY, \
            CALLBACK_WITH_ERROR_ARGS2)(__VA_ARGS__)

#define CHECK_ACL_SUCCESS(expr, msg)             \
  do {                                           \
    auto _ret = (expr);                          \
    if (_ret != ACL_SUCCESS) {                   \
      LOG(FATAL) << "CHECK ACL FAILED: " << msg; \
    }                                            \
  } while (0)

#define NOT_IMPLEMENTED()                            \
  do {                                               \
    LOG(FATAL) << __func__ << " is not implemented"; \
  } while (0)

#define NOT_IMPLEMENTED_WITH_MSG(msg)                         \
  do {                                                        \
    LOG(FATAL) << __func__ << " is not implemented: " << msg; \
  } while (0)

// Multi-model step lock/unlock macros for serializing step execution
// Only locks when the condition is true.
#define MULTI_MODEL_STEP_LOCK(condition)                \
  static std::mutex __multi_model_step_mutex;           \
  std::unique_lock<std::mutex> __multi_model_step_lock( \
      __multi_model_step_mutex, std::defer_lock);       \
  do {                                                  \
    if (condition) {                                    \
      __multi_model_step_lock.lock();                   \
    }                                                   \
  } while (0)

#define MULTI_MODEL_STEP_UNLOCK()              \
  do {                                         \
    if (__multi_model_step_lock.owns_lock()) { \
      __multi_model_step_lock.unlock();        \
    }                                          \
  } while (0)

// Set ATB execute stream with stream guard
#define SET_ATB_EXECUTE_STREAM(compute_stream, device, context)           \
  c10::StreamGuard __stream_guard = (compute_stream)->set_stream_guard(); \
  do {                                                                    \
    aclrtStream __current_stream =                                        \
        c10_npu::getCurrentNPUStream((device).index()).stream();          \
    auto __atb_context =                                                  \
        const_cast<atb::Context*>((context).get_atb_context());           \
    __atb_context->SetExecuteStream(__current_stream);                    \
  } while (0)

}  // namespace xllm

/* Copyright 2025-2026 The xLLM Authors.
Copyright 2024 The ScaleLLM Authors. All Rights Reserved.

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

#include <cstdint>
#include <cstdlib>

/// The various available dtypes. They MUST be in increasing alignment order
enum class Dtype {
  /// Boolan type
  BOOL,
  /// MXF4
  /// <https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf>_
  F4,
  /// MXF6
  /// <https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf>_
  F6_E2M3,
  /// MXF6
  /// <https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf>_
  F6_E3M2,
  /// Unsigned byte
  U8,
  /// Signed byte
  I8,
  /// FP8 <https://arxiv.org/pdf/2209.05433.pdf>_
  F8_E5M2,
  /// FP8 <https://arxiv.org/pdf/2209.05433.pdf>_
  F8_E4M3,
  /// F8_E8M0
  /// <https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf>_
  F8_E8M0,
  /// Signed integer (16-bit)
  I16,
  /// Unsigned integer (16-bit)
  U16,
  /// Half-precision floating point
  F16,
  /// Brain floating point
  BF16,
  /// Signed integer (32-bit)
  I32,
  /// Unsigned integer (32-bit)
  U32,
  /// Floating point (32-bit)
  F32,
  /// Floating point (64-bit)
  F64,
  /// Signed integer (64-bit)
  I64,
  /// Unsigned integer (64-bit)
  U64,
};

enum class Status {
  NullPointer = -2,
  Utf8Error,
  Ok,
  InvalidHeader,
  InvalidHeaderStart,
  InvalidHeaderDeserialization,
  HeaderTooLarge,
  HeaderTooSmall,
  InvalidHeaderLength,
  TensorNotFound,
  TensorInvalidInfo,
  InvalidOffset,
  IoError,
  JsonError,
  InvalidTensorView,
  MetadataIncompleteBuffer,
  ValidationOverflow,
  MisalignedSlice,
};

struct Handle;

struct View {
  Dtype dtype;
  size_t rank;
  const size_t* shape;
  size_t start;
  size_t stop;
};

extern "C" {

/// Attempt to deserialize the content of `buffer`, reading `buffer_len` bytes
/// as a safentesors data buffer.
///
/// # Arguments
///
/// * `handle`: In-Out pointer to store the resulting safetensors reference is
/// successfully deserialized
/// * `buffer`: Buffer to attempt to read data from
/// * `buffer_len`: Number of bytes we can safely read from the deserialize the
/// safetensors
///
/// returns: `Status::Ok == 0` if success, any other status code if an error
/// what caught up
Status safetensors_deserialize(Handle** handle,
                               const uint8_t* buffer,
                               size_t buffer_len);

/// Free the resources hold by the safetensors
///
/// # Arguments
///
/// * `handle`: Pointer to the safetensors we want to release the resources of
///
/// returns: `Status::Ok == 0` if success, any other status code if an error
/// what caught up
Status safetensors_destroy(Handle* handle);

/// Retrieve the list of tensor's names currently stored in the safetensors
///
/// # Arguments
///
/// * `handle`: Pointer to the underlying safetensors we want to query tensor's
/// names from
/// * `ptr`: In-Out pointer to store the array of strings representing all the
/// tensor's names
/// * `len`: Number of strings stored in `ptr`
///
/// returns: `Status::Ok == 0` if success, any other status code if an error
/// what caught up
Status safetensors_names(const Handle* handle,
                         const char* const** ptr,
                         size_t* len);

/// Free the resources used to represent the list of tensor's names stored in
/// the safetensors. This must follow any call to `safetensors_names()` to clean
/// up underlying resources.
///
/// # Arguments
///
/// * `names`: Pointer to the array of strings we want to release resources of
/// * `len`: Number of strings hold by `names` array
///
/// returns: `Status::Ok == 0` if success, any other status code if an error
/// what caught up
Status safetensors_free_names(const char* const* names, size_t len);

/// Attempt to retrieve the metadata and content for the tensor associated with
/// `name` storing the result to the memory location pointed by `view` pointer.
///
/// # Arguments
///
/// * `handle`: Pointer to the underlying safetensors we want to retrieve the
/// tensor from.
/// * `view`: In-Out pointer to store the tensor if successfully found to belong
/// to the safetensors
/// * `name`: The name of the tensor to retrieve from the safetensors
///
/// returns: `Status::Ok == 0` if success, any other status code if an error
/// what caught up
Status safetensors_get_tensor(const Handle* handle,
                              View** view,
                              const char* name);

/// Free the resources used by a TensorView to expose metadata + content to the
/// C-FFI layer
///
/// # Arguments
///
/// * `ptr`: Pointer to the TensorView we want to release the underlying
/// resources of
///
/// returns: `Status::Ok = 0` if resources were successfully freed
Status safetensors_free_tensor(View* ptr);

}  // extern "C"

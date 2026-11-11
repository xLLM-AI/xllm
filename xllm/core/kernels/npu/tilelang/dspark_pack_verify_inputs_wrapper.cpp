/* Copyright 2026 The xLLM Authors.

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

#include <acl/acl.h>
#include <glog/logging.h>
#include <torch_npu/csrc/core/npu/NPUGuard.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <vector>

#include "core/kernels/npu/tilelang/dispatch_registry.h"
#include "core/kernels/npu/tilelang/tilelang_ops_api.h"

#ifndef XLLM_TL_DSPARK_PACK_VERIFY_INPUTS_REGISTRY_INC
#error "XLLM_TL_DSPARK_PACK_VERIFY_INPUTS_REGISTRY_INC is not defined"
#endif

namespace xllm::kernel::npu::tilelang {
namespace {
#include XLLM_TL_DSPARK_PACK_VERIFY_INPUTS_REGISTRY_INC

constexpr int64_t kMaxBatchSize = 128;
constexpr int64_t kMaxOffset = std::numeric_limits<int32_t>::max();
constexpr int64_t kRowsPerTask = 32;
constexpr int64_t kVectorsPerCore = 2;

void check_vector(const torch::Tensor& tensor,
                  const torch::Device& device,
                  torch::ScalarType dtype,
                  int64_t rows) {
  CHECK(tensor.defined());
  CHECK_EQ(tensor.device(), device);
  CHECK_EQ(tensor.scalar_type(), dtype);
  CHECK_EQ(tensor.dim(), 1);
  CHECK_EQ(tensor.size(0), rows);
  CHECK(tensor.is_contiguous());
}

bool spans_overlap(const torch::Tensor& first, const torch::Tensor& second) {
  if (first.numel() == 0 || second.numel() == 0) {
    return false;
  }
  const uintptr_t first_start = reinterpret_cast<uintptr_t>(first.data_ptr());
  const uintptr_t second_start = reinterpret_cast<uintptr_t>(second.data_ptr());
  const uintptr_t first_bytes =
      static_cast<uintptr_t>(first.numel() * first.element_size());
  const uintptr_t second_bytes =
      static_cast<uintptr_t>(second.numel() * second.element_size());
  return first_start < second_start + second_bytes &&
         second_start < first_start + first_bytes;
}

DsparkPackVerifyInputsSpecialization build_runtime_specialization(
    int64_t batch_size,
    int64_t draft_width) {
  int32_t bucket = 1;
  while (bucket < batch_size) {
    bucket *= 2;
  }
  return make_dspark_pack_verify_inputs_specialization(
      DsparkPackVerifyInputsMaxBatchSize{bucket},
      DsparkPackVerifyInputsDraftWidth{static_cast<int32_t>(draft_width)});
}

int32_t copy_core_count(const torch::Device& device, int64_t capacity) {
  // Query immutable device metadata during the first warmup on each worker
  // thread, not on every serving round. Graph replay never enters the wrapper.
  thread_local int32_t cached_device = -1;
  thread_local int32_t cached_core_limit = 0;
  const int32_t device_index = static_cast<int32_t>(device.index());
  CHECK_GE(device_index, 0);
  if (cached_device != device_index) {
    int64_t vector_cores = 0;
    CHECK_EQ(aclrtGetDeviceInfo(static_cast<uint32_t>(device_index),
                                ACL_DEV_ATTR_VECTOR_CORE_NUM,
                                &vector_cores),
             ACL_SUCCESS);
    CHECK_GT(vector_cores, 0);
    CHECK_EQ(vector_cores % kVectorsPerCore, 0);
    CHECK_LE(vector_cores, kMaxOffset);
    cached_core_limit = static_cast<int32_t>(vector_cores / kVectorsPerCore);
    cached_device = device_index;
  }
  // An empty pack still initializes offsets[0] without touching row buffers.
  const int64_t requested =
      std::max<int64_t>(1, (capacity + kRowsPerTask - 1) / kRowsPerTask);
  return static_cast<int32_t>(
      std::min(requested, static_cast<int64_t>(cached_core_limit)));
}
}  // namespace

void dspark_pack_verify_inputs(const torch::Tensor& anchors,
                               const torch::Tensor& proposals,
                               const torch::Tensor& base_positions,
                               const torch::Tensor& base_kv_lengths,
                               const torch::Tensor& base_block_table,
                               const torch::Tensor& widths,
                               torch::Tensor& offsets,
                               torch::Tensor& tokens,
                               torch::Tensor& positions,
                               torch::Tensor& kv_seq_lens,
                               torch::Tensor& slot_mapping,
                               torch::Tensor& block_table,
                               torch::Tensor& q_seq_lens,
                               torch::Tensor& owners,
                               torch::Tensor& columns,
                               int64_t block_size) {
  CHECK(proposals.defined());
  CHECK_EQ(proposals.dim(), 2);
  const int64_t draft_width = proposals.size(1);
  CHECK(draft_width == 5 || draft_width == 7);
  CHECK(anchors.defined());
  CHECK(anchors.device().is_privateuseone());
  CHECK_EQ(anchors.dim(), 1);
  const torch::Device device = anchors.device();
  const int64_t batch_size = anchors.size(0);
  CHECK_LE(batch_size, kMaxBatchSize);
  check_vector(anchors, device, torch::kInt64, batch_size);
  check_vector(base_positions, device, torch::kInt32, batch_size);
  check_vector(base_kv_lengths, device, torch::kInt32, batch_size);
  CHECK(proposals.defined());
  CHECK_EQ(proposals.device(), device);
  CHECK(proposals.is_contiguous());
  check_vector(widths, device, torch::kInt32, batch_size);
  CHECK_EQ(proposals.scalar_type(), torch::kInt64);
  CHECK_EQ(proposals.dim(), 2);
  CHECK_EQ(proposals.size(0), batch_size);
  CHECK_EQ(proposals.size(1), draft_width);
  CHECK(base_block_table.defined());
  CHECK_EQ(base_block_table.device(), device);
  CHECK_EQ(base_block_table.scalar_type(), torch::kInt32);
  CHECK_EQ(base_block_table.dim(), 2);
  CHECK_EQ(base_block_table.size(0), batch_size);
  CHECK(base_block_table.is_contiguous());
  const int64_t table_columns = base_block_table.size(1);
  CHECK_GT(table_columns, 0);
  CHECK_LE(table_columns, kMaxOffset);
  CHECK_GT(block_size, 0);
  CHECK_LE(block_size, kMaxOffset / table_columns);
  CHECK(tokens.defined());
  CHECK_EQ(tokens.dim(), 1);
  const int64_t capacity = tokens.size(0);
  CHECK_GE(capacity, batch_size);
  CHECK_GE(capacity, 0);
  CHECK_LE(capacity, kMaxBatchSize * (draft_width + 1));
  CHECK_LE(table_columns,
           kMaxOffset / std::max<int64_t>({1, capacity, batch_size}));
  check_vector(offsets, device, torch::kInt32, batch_size + 1);
  for (const torch::Tensor* tensor : {&tokens,
                                      &positions,
                                      &kv_seq_lens,
                                      &slot_mapping,
                                      &q_seq_lens,
                                      &owners,
                                      &columns}) {
    check_vector(*tensor, device, torch::kInt32, capacity);
  }
  CHECK(block_table.defined());
  CHECK_EQ(block_table.device(), device);
  CHECK_EQ(block_table.scalar_type(), torch::kInt32);
  CHECK_EQ(block_table.dim(), 2);
  CHECK_EQ(block_table.size(0), capacity);
  CHECK_EQ(block_table.size(1), table_columns);
  CHECK(block_table.is_contiguous());
  std::vector<const torch::Tensor*> inputs;
  inputs.reserve(6);
  inputs.insert(inputs.end(),
                {&anchors,
                 &proposals,
                 &base_positions,
                 &base_kv_lengths,
                 &base_block_table,
                 &widths});
  std::vector<const torch::Tensor*> outputs;
  outputs.reserve(9);
  outputs.insert(outputs.end(),
                 {&offsets,
                  &tokens,
                  &positions,
                  &kv_seq_lens,
                  &slot_mapping,
                  &block_table,
                  &q_seq_lens,
                  &owners,
                  &columns});
  for (size_t output = 0; output < outputs.size(); ++output) {
    for (const torch::Tensor* input : inputs) {
      CHECK(!spans_overlap(*outputs[output], *input))
          << "DSpark packed outputs must not overlap live inputs";
    }
    for (size_t previous = 0; previous < output; ++previous) {
      CHECK(!spans_overlap(*outputs[output], *outputs[previous]))
          << "DSpark packed output spans must be disjoint";
    }
  }
  const auto specialization =
      build_runtime_specialization(batch_size, draft_width);
  const auto* entry =
      find_dspark_pack_verify_inputs_kernel_entry(specialization);
  CHECK(entry != nullptr) << available_dspark_pack_verify_inputs_variant_keys();
  const int32_t copy_cores = copy_core_count(device, capacity);
  const auto stream = c10_npu::getCurrentNPUStream(device.index());
  c10_npu::NPUStreamGuard stream_guard(stream);
  entry->fn(reinterpret_cast<uint8_t*>(anchors.data_ptr()),
            reinterpret_cast<uint8_t*>(proposals.data_ptr()),
            reinterpret_cast<uint8_t*>(base_positions.data_ptr()),
            reinterpret_cast<uint8_t*>(base_kv_lengths.data_ptr()),
            reinterpret_cast<uint8_t*>(base_block_table.data_ptr()),
            reinterpret_cast<uint8_t*>(widths.data_ptr()),
            reinterpret_cast<uint8_t*>(offsets.data_ptr()),
            reinterpret_cast<uint8_t*>(tokens.data_ptr()),
            reinterpret_cast<uint8_t*>(positions.data_ptr()),
            reinterpret_cast<uint8_t*>(kv_seq_lens.data_ptr()),
            reinterpret_cast<uint8_t*>(slot_mapping.data_ptr()),
            reinterpret_cast<uint8_t*>(block_table.data_ptr()),
            reinterpret_cast<uint8_t*>(q_seq_lens.data_ptr()),
            reinterpret_cast<uint8_t*>(owners.data_ptr()),
            reinterpret_cast<uint8_t*>(columns.data_ptr()),
            static_cast<int32_t>(batch_size),
            static_cast<int32_t>(capacity),
            static_cast<int32_t>(table_columns),
            static_cast<int32_t>(block_size),
            copy_cores,
            stream.stream());
}
}  // namespace xllm::kernel::npu::tilelang

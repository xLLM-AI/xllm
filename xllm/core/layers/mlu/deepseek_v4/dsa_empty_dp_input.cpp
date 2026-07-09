/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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

#include "layers/mlu/deepseek_v4/dsa_empty_dp_input.h"

#include <glog/logging.h>

#include <algorithm>
#include <cstdint>

#include "framework/model/model_input_params.h"

namespace {

int64_t div_up(int64_t value, int64_t divisor) {
  CHECK_GT(divisor, 0);
  return (value + divisor - 1) / divisor;
}

int32_t dummy_kv_len(const std::vector<xllm::DSAGroupInfo>& group_infos,
                     int64_t window_size) {
  int64_t kv_len = std::max<int64_t>(window_size, 1);
  for (const xllm::DSAGroupInfo& group_info : group_infos) {
    kv_len = std::max<int64_t>(kv_len, group_info.ratio);
  }
  return static_cast<int32_t>(kv_len);
}

int64_t group_block_num(const xllm::DSAGroupInfo& group_info, int32_t kv_len) {
  const int64_t block_size = std::max<int64_t>(group_info.block_size, 1);
  return std::max<int64_t>(div_up(kv_len, block_size), 1);
}

}  // namespace

namespace xllm::layer {

void fill_dsv4_empty_dp_params(ModelInputParams& params,
                               const std::vector<DSAGroupInfo>& group_infos,
                               int64_t window_size) {
  const bool is_chunked_prefill =
      params.meta.batch_forward_type.is_chunked_prefill();
  const int32_t kv_len = dummy_kv_len(group_infos, window_size);
  const torch::TensorOptions int_options =
      torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU);

  params.attn_metadata = nullptr;
  params.meta.num_sequences = 1;
  params.meta.actual_num_sequences = 1;
  params.meta.q_max_seq_len = 1;
  params.meta.kv_max_seq_len =
      std::max<int32_t>(params.meta.kv_max_seq_len, kv_len);
  params.meta.batch_forward_type = is_chunked_prefill
                                       ? BatchForwardType::CHUNKED_PREFILL
                                       : BatchForwardType::DECODE;

  params.attention.host.q_seq_lens = {0, 1};
  params.attention.host.kv_seq_lens = {0, kv_len};
  params.attention.host.q_cu_seq_lens = {1};
  params.attention.host.kv_cu_seq_lens = {kv_len};
  params.attention.host.new_cache_slots = {0};
  params.attention.host.kv_cache_tokens_nums = {1};
  params.attention.device.q_seq_lens =
      torch::tensor(params.attention.host.q_seq_lens, int_options);
  params.attention.device.kv_seq_lens =
      torch::tensor(params.attention.host.kv_seq_lens, int_options);
  params.attention.device.q_cu_seq_lens =
      torch::tensor(params.attention.host.q_cu_seq_lens, int_options);
  params.attention.device.new_cache_slots =
      torch::tensor(params.attention.host.new_cache_slots, int_options);
  params.attention.device.kv_cache_tokens_nums =
      torch::tensor(params.attention.host.kv_cache_tokens_nums, int_options);

  int64_t max_block_num = 1;
  for (const DSAGroupInfo& group_info : group_infos) {
    max_block_num =
        std::max(max_block_num, group_block_num(group_info, kv_len));
  }
  params.attention.device.block_tables =
      torch::zeros({1, max_block_num}, int_options);

  params.multi_block_tables.clear();
  params.multi_block_tables.reserve(group_infos.size());
  for (const DSAGroupInfo& group_info : group_infos) {
    const int64_t block_num = group_block_num(group_info, kv_len);
    params.multi_block_tables.emplace_back(
        torch::zeros({1, block_num}, int_options));
  }
}

}  // namespace xllm::layer

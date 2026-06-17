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

#include "onerec_xattention_batch_input_builder.h"

#include <algorithm>
#include <vector>

#include "common/global_flags.h"
#include "core/framework/config/scheduler_config.h"
#include "core/util/rec_model_utils.h"
#include "core/util/utils.h"
#include "util/tensor_helper.h"

namespace xllm {

namespace {

int32_t get_onerec_xattention_decode_position(const Sequence& sequence) {
  return static_cast<int32_t>(sequence.num_prompt_tokens() +
                              sequence.num_decoder_embeddings());
}

}  // namespace

ForwardInput OneRecXAttentionBatchInputBuilder::build_rec_forward_input(
    uint32_t num_decoding_tokens,
    uint32_t min_decoding_batch_size) {
  auto input = OneRecBatchInputBuilder::build_rec_forward_input(
      num_decoding_tokens, min_decoding_batch_size);
  if (const auto* onerec = input.input_params.onerec_params()) {
    OneRecModelInputParams legacy_params = *onerec;
    auto& xattn_params = input.input_params.mutable_onerec_xattention_params();
    static_cast<OneRecModelInputParams&>(xattn_params) =
        std::move(legacy_params);
  }

  input.input_params.meta.batch_forward_type = BatchForwardType::PREFILL;

  if (sequence_groups_.empty()) {
    return input;
  }

  std::vector<std::vector<int32_t>> block_tables_vec;
  std::vector<int32_t> new_cache_slots_vec;
  std::vector<const RequestSamplingParam*> decode_sampling_params;
  std::vector<int32_t> decode_selected_token_idxes;
  std::vector<int32_t> decode_sample_idxes;
  std::vector<std::vector<int64_t>> decode_unique_token_ids_vec;
  std::vector<std::vector<int32_t>> decode_unique_token_counts_vec;
  std::vector<int32_t> decode_unique_token_lens_vec;
  std::vector<int32_t> decode_positions_vec;
  int32_t decode_hidden_row_offset = 0;

  int32_t batch_size = 0;
  int32_t beam_width = 1;
  for (size_t group_idx = 0; group_idx < sequence_groups_.size(); ++group_idx) {
    auto* group = sequence_groups_[group_idx];
    if (group == nullptr) {
      continue;
    }
    for (const auto& sequence_ptr : group->sequences()) {
      if (!sequence_ptr) {
        continue;
      }
      ++batch_size;
      const int32_t total_seq_len = static_cast<int32_t>(
          sequence_ptr->num_tokens() + sequence_ptr->num_decoder_embeddings());
      const int32_t n_kv_cache_tokens =
          static_cast<int32_t>(sequence_ptr->kv_state().kv_cache_tokens_num());
      const auto blocks = sequence_ptr->kv_state().kv_blocks();
      std::vector<int32_t> block_ids;
      block_ids.reserve(blocks.size());
      for (const auto& block : blocks) {
        block_ids.emplace_back(block.id());
      }
      block_tables_vec.emplace_back(std::move(block_ids));
      if (total_seq_len > n_kv_cache_tokens) {
        auto slot_ids = sequence_ptr->kv_state().kv_cache_slots(
            n_kv_cache_tokens, total_seq_len);
        new_cache_slots_vec.insert(
            new_cache_slots_vec.end(), slot_ids.begin(), slot_ids.end());
      }

      const auto* sampling_param = sequence_ptr->sampling_param();
      if (sampling_param == nullptr) {
        continue;
      }
      beam_width = std::max<int32_t>(beam_width, sampling_param->beam_width);
      decode_positions_vec.emplace_back(
          get_onerec_xattention_decode_position(*sequence_ptr));
      const int32_t sel_start =
          static_cast<int32_t>(decode_selected_token_idxes.size());
      const int32_t decode_hidden_seq_len = std::max(total_seq_len, 1);
      for (int32_t beam_idx = 0; beam_idx < beam_width; ++beam_idx) {
        const int32_t idx = sel_start + beam_idx;
        const int32_t selected_row = decode_hidden_row_offset +
                                     beam_idx * decode_hidden_seq_len +
                                     (decode_hidden_seq_len - 1);
        decode_sampling_params.emplace_back(sampling_param);
        decode_selected_token_idxes.emplace_back(selected_row);
        decode_sample_idxes.emplace_back(idx);
        decode_unique_token_ids_vec.emplace_back();
        decode_unique_token_counts_vec.emplace_back();
        decode_unique_token_lens_vec.emplace_back(0);
      }
      decode_hidden_row_offset += beam_width * decode_hidden_seq_len;
    }
  }

  if (!decode_selected_token_idxes.empty()) {
    input.decoder_sampling_params.init(decode_sampling_params,
                                       decode_selected_token_idxes,
                                       decode_sample_idxes,
                                       decode_unique_token_ids_vec,
                                       decode_unique_token_counts_vec,
                                       decode_unique_token_lens_vec);
  }

  StepDecodeMeta step_meta;
  step_meta.batch_size = batch_size;
  step_meta.beam_width = beam_width;
  step_meta.current_round = 0;
  step_meta.total_round = std::max(1, get_rec_multi_round_decode_rounds());
  step_meta.decode_positions_vec = std::move(decode_positions_vec);

  if (args_ != nullptr) {
    const int64_t n_kv_heads = args_->decoder_n_kv_heads().value_or(
        args_->n_kv_heads().value_or(args_->decoder_n_heads()));
    const int64_t head_dim = args_->decoder_head_dim();
    step_meta.full_kv_shape = {
        ::xllm::SchedulerConfig::get_instance().max_tokens_per_batch() +
            ::xllm::SchedulerConfig::get_instance().max_seqs_per_batch() *
                beam_width * std::max(0, step_meta.total_round - 1),
        n_kv_heads,
        head_dim,
    };
  }

  if (!block_tables_vec.empty()) {
    util::pad_2d_vector(block_tables_vec, /*pad_value=*/0);
    input.input_params.attention.device.block_tables =
        create_2d_tensor(block_tables_vec, torch::kInt);
    input.input_params.attention.host.block_tables =
        input.input_params.attention.device.block_tables;
  }
  if (!new_cache_slots_vec.empty()) {
    input.input_params.attention.device.new_cache_slots =
        torch::tensor(new_cache_slots_vec, torch::kInt);
  }

  input.step_decode = std::move(step_meta);
  return input;
}

}  // namespace xllm

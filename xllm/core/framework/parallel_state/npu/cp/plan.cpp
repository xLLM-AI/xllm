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

#include "framework/parallel_state/npu/cp/plan.h"

#include "framework/model/model_input_params.h"
#include "framework/parallel_state/npu_cp_closure.h"
#include "util/tensor_helper.h"

namespace xllm::npu::cp {
namespace {

CpEpPaddingData move_padding_to(const CpEpPaddingData& data,
                                const torch::Device& device) {
  CpEpPaddingData out;
  out.attn_padding_idx(safe_to(data.attn_padding_idx(), device, true))
      .attn_unpadding_idx(safe_to(data.attn_unpadding_idx(), device, true))
      .ffn_padding_idx(safe_to(data.ffn_padding_idx(), device, true))
      .ffn_unpadding_idx(safe_to(data.ffn_unpadding_idx(), device, true))
      .lm_head_skip_padding_token_indices(
          safe_to(data.lm_head_skip_padding_token_indices(), device, true))
      .gather_prenorm_idx(safe_to(data.gather_prenorm_idx(), device, true))
      .padding_idx(safe_to(data.padding_idx(), device, true))
      .un_padding_idx(safe_to(data.un_padding_idx(), device, true))
      .dynamic_ep_idx(safe_to(data.dynamic_ep_idx(), device, true))
      .moe_idx(safe_to(data.moe_idx(), device, true))
      .expert_array(safe_to(data.expert_array(), device, true));
  return out;
}

}  // namespace

Plan Plan::build(const SourceLayout& source, const ParallelConfig& config) {
  CHECK_GT(config.size, 1) << "CP plan requires size > 1";
  CHECK_GE(config.rank, 0);
  CHECK_LT(config.rank, config.size);

  Plan plan;
  plan.layout_ = build_npu_cp_prefill_plan(config.size,
                                           config.rank,
                                           source.q_seq_lens,
                                           source.positions,
                                           source.have_prefix_slots,
                                           source.kv_cache_tokens_per_seq,
                                           config.block_size,
                                           config.kv_split_size);
  plan.prefill_inputs_ =
      prepare_cp_prefill_inputs_from_plan(plan.layout_,
                                          source.have_prefix_slots,
                                          source.kv_cache_tokens_per_seq,
                                          config.block_size,
                                          config.kv_split_size);
  plan.ep_padding_data_ = CpEpPadding(plan.layout_.local_padded_token_num,
                                      config.num_experts_per_tok,
                                      config.mapping_data,
                                      config.device,
                                      config.dtype,
                                      config.is_prefill)
                              .build();
  return plan.to(config.device);
}

Plan Plan::to(const torch::Device& device) const {
  Plan out = *this;
  out.layout_ = layout_.to(device);
  out.prefill_inputs_ = prefill_inputs_.to(device);
  out.ep_padding_data_ = move_padding_to(ep_padding_data_, device);
  return out;
}

void Plan::preprocess(torch::Tensor& hidden,
                      torch::Tensor& positions,
                      ModelInputParams& input_params) const {
  if (!enabled()) {
    return;
  }
  hidden = npu_cp::localize(hidden, layout_);
  positions = npu_cp::localize_positions(layout_, positions);
  apply_cp_local_metadata_from_plan(input_params, layout_, hidden.device());
}

torch::Tensor Plan::postprocess(const torch::Tensor& hidden,
                                ProcessGroup* process_group) const {
  return npu_cp::gather_restore(hidden, layout_, process_group);
}

torch::Tensor Plan::localize_slots_recovered(
    const torch::Tensor& global_slots) const {
  return npu_cp::localize_slots_recovered(
      global_slots, layout_, prefill_inputs_.cp_kv_recover_idx);
}

}  // namespace xllm::npu::cp

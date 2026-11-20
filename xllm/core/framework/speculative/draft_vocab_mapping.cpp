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

#include "core/framework/speculative/draft_vocab_mapping.h"

#include <glog/logging.h>

#include <limits>
#include <tuple>
#include <utility>

#include "core/framework/sampling/sampling_params.h"

namespace xllm {

DraftVocabMapping::DraftVocabMapping(
    const torch::Tensor& draft_token_id_offsets,
    int64_t draft_vocab_size,
    int64_t target_vocab_size,
    const torch::Device& device)
    : target_vocab_size_(target_vocab_size) {
  if (draft_vocab_size == target_vocab_size) {
    return;
  }
  CHECK(draft_token_id_offsets.defined())
      << "Different draft and target vocabulary sizes require a d2t tensor.";
  CHECK_EQ(draft_token_id_offsets.dim(), 1) << "Draft d2t table must be 1D.";
  CHECK_EQ(draft_token_id_offsets.numel(), draft_vocab_size)
      << "Draft d2t table size must match the draft vocab_size.";
  const auto mapped_ids =
      draft_token_id_offsets.to(torch::kInt64) +
      torch::arange(draft_vocab_size,
                    draft_token_id_offsets.options().dtype(torch::kInt64));
  CHECK_EQ(std::get<0>(torch::_unique2(mapped_ids)).numel(), draft_vocab_size)
      << "Draft d2t mapped target ids must be unique.";
  const auto [min_id, max_id] = torch::aminmax(mapped_ids);
  CHECK_GE(min_id.item<int64_t>(), 0)
      << "Draft d2t mapped target ids must be >= 0.";
  CHECK_LT(max_id.item<int64_t>(), target_vocab_size)
      << "Draft d2t mapped target ids must be < target vocab ("
      << target_vocab_size << ").";
  draft_to_target_ids_ = mapped_ids.to(device);
}

int64_t DraftVocabMapping::mapped_vocab_size(int64_t draft_vocab_size) const {
  return draft_to_target_ids_.defined() ? target_vocab_size_ : draft_vocab_size;
}

torch::Tensor DraftVocabMapping::draft_logits_view(
    const torch::Tensor& target_logits) const {
  return draft_to_target_ids_.defined() ? torch::Tensor() : target_logits;
}

void DraftVocabMapping::remap(SampleOutput& sample_output,
                              torch::Tensor target_logits) const {
  if (!draft_to_target_ids_.defined() || !sample_output.next_tokens.defined() ||
      sample_output.next_tokens.numel() == 0) {
    return;
  }
  if (sample_output.draft_logits.defined()) {
    const int64_t rows = sample_output.draft_logits.size(0);
    if (target_logits.defined()) {
      target_logits.fill_(-std::numeric_limits<float>::infinity());
    } else {
      target_logits = torch::full({rows, target_vocab_size_},
                                  -std::numeric_limits<float>::infinity(),
                                  sample_output.draft_logits.options());
    }
    target_logits.scatter_(
        /*dim=*/1,
        draft_to_target_ids_.view({1, -1}).expand(
            {rows, draft_to_target_ids_.size(0)}),
        sample_output.draft_logits);
    sample_output.draft_logits = std::move(target_logits);
  }
  sample_output.next_tokens = draft_to_target_ids_.index_select(
      0, sample_output.next_tokens.to(torch::kInt64));
}

}  // namespace xllm

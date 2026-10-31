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

#include "runtime/draft_vocabulary.h"

#include <glog/logging.h>

#include <limits>
#include <tuple>
#include <utility>

#include "core/framework/sampling/sampling_params.h"

namespace xllm {

DraftVocabulary::DraftVocabulary(torch::Tensor d2t,
                                 int64_t draft_vocab_size,
                                 int64_t target_vocab_size,
                                 const torch::Device& device)
    : target_vocab_size(target_vocab_size) {
  if (!d2t.defined()) {
    return;
  }
  CHECK_EQ(d2t.dim(), 1) << "Draft d2t table must be 1D.";
  torch::Tensor mapped_ids =
      d2t.to(torch::kInt64) +
      torch::arange(d2t.size(0), d2t.options().dtype(torch::kInt64));
  CHECK_EQ(mapped_ids.numel(), draft_vocab_size)
      << "Draft d2t table has " << mapped_ids.numel()
      << " entries but the draft vocab_size is " << draft_vocab_size << ".";
  CHECK_EQ(std::get<0>(torch::_unique2(mapped_ids)).numel(), draft_vocab_size)
      << "Draft d2t mapped target ids must be unique.";
  CHECK_GE(torch::min(mapped_ids).item<int64_t>(), 0)
      << "Draft d2t mapped target ids must be >= 0.";
  CHECK_LT(torch::max(mapped_ids).item<int64_t>(), target_vocab_size)
      << "Draft d2t mapped target ids must be < target vocab ("
      << target_vocab_size << ").";
  target_token_ids = mapped_ids.to(device);
}

void DraftVocabulary::remap(SampleOutput& sample_output,
                            torch::Tensor target_logits) const {
  if (!target_token_ids.defined() || !sample_output.next_tokens.defined() ||
      sample_output.next_tokens.numel() == 0) {
    return;
  }
  if (sample_output.draft_logits.defined()) {
    const int64_t rows = sample_output.draft_logits.size(0);
    if (target_logits.defined()) {
      target_logits.fill_(-std::numeric_limits<float>::infinity());
    } else {
      target_logits = torch::full({rows, target_vocab_size},
                                  -std::numeric_limits<float>::infinity(),
                                  sample_output.draft_logits.options());
    }
    target_logits.scatter_(
        /*dim=*/1,
        target_token_ids.view({1, -1}).expand({rows, target_token_ids.size(0)}),
        sample_output.draft_logits);
    sample_output.draft_logits = std::move(target_logits);
  }
  sample_output.next_tokens = target_token_ids.index_select(
      0, sample_output.next_tokens.to(torch::kInt64));
}

}  // namespace xllm

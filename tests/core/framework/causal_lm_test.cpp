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

#include "core/framework/model/causal_lm.h"

#include <gtest/gtest.h>

#include <memory>
#include <vector>

namespace xllm {
namespace {

// A model that only implements the 2-arg logits, like the models on the
// generic LlmForCausalLMImplBase whose lm_head cannot emit the selected
// hidden in the same projection.
class TwoArgLogitsModel {
 public:
  ModelOutput forward(const torch::Tensor& tokens,
                      const torch::Tensor& positions,
                      std::vector<KVCache>& kv_caches,
                      const ModelInputParams& parameters) {
    (void)tokens;
    (void)positions;
    (void)kv_caches;
    (void)parameters;
    return ModelOutput();
  }

  torch::Tensor pooler(const torch::Tensor& hidden_states,
                       const torch::Tensor& seleted_idxes) {
    if (seleted_idxes.defined()) {
      return hidden_states.index_select(/*dim=*/0, seleted_idxes);
    }
    return hidden_states;
  }

  // Stand-in projection with vocab_size == hidden_size.
  torch::Tensor logits(const torch::Tensor& hidden_states,
                       const torch::Tensor& seleted_idxes) {
    if (seleted_idxes.defined()) {
      return hidden_states.index_select(/*dim=*/0, seleted_idxes);
    }
    return hidden_states;
  }

  void load_model(std::unique_ptr<ModelLoader> loader) { (void)loader; }

  void prepare_expert_weight(int32_t layer_id,
                             const std::vector<int32_t>& expert_ids) {
    (void)layer_id;
    (void)expert_ids;
  }

  void update_expert_weight(int32_t layer_id) { (void)layer_id; }
};

class CausalLMImplLogitsWithHiddenTest : public ::testing::Test {
 protected:
  CausalLMImpl<std::shared_ptr<TwoArgLogitsModel>> lm_{
      std::make_shared<TwoArgLogitsModel>(),
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU)};

  torch::Tensor make_hidden() {
    return torch::arange(0, 12, torch::TensorOptions().dtype(torch::kFloat32))
        .reshape({4, 3});
  }
};

TEST_F(CausalLMImplLogitsWithHiddenTest,
       GathersSelectedHiddenWhenModelLacksFusedVariant) {
  torch::Tensor hidden = make_hidden();
  torch::Tensor idxes =
      torch::tensor({3, 0}, torch::TensorOptions().dtype(torch::kInt32));

  torch::Tensor out_hidden;
  torch::Tensor logits = lm_.logits(hidden, idxes, out_hidden);

  torch::Tensor expected =
      hidden.index_select(/*dim=*/0, idxes.to(torch::kLong));
  EXPECT_TRUE(out_hidden.defined());
  EXPECT_TRUE(torch::equal(out_hidden, expected));
  EXPECT_TRUE(torch::equal(logits, lm_.logits(hidden, idxes)));
}

TEST_F(CausalLMImplLogitsWithHiddenTest, PassesThroughHiddenWithoutSelection) {
  torch::Tensor hidden = make_hidden();
  torch::Tensor undefined_idxes;

  torch::Tensor out_hidden;
  torch::Tensor logits = lm_.logits(hidden, undefined_idxes, out_hidden);

  EXPECT_TRUE(out_hidden.defined());
  EXPECT_TRUE(torch::equal(out_hidden, hidden));
  EXPECT_TRUE(torch::equal(logits, lm_.logits(hidden, undefined_idxes)));
}

}  // namespace
}  // namespace xllm

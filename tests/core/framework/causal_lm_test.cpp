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

#include "core/framework/model/lm_head_owner.h"
#include "core/framework/model/model_output.h"
#include "tests/core/framework/lm_head_test_utils.h"

namespace xllm {
namespace {

// Family model hosting the vocabulary head through LmHeadOwner, mirroring
// LlmForCausalLMImplBase families. The backbone is a plain value member; only
// the head needs torch module registration.
class FamilyWithHead : public torch::nn::Module, public detail::LmHeadOwner<> {
 public:
  explicit FamilyWithHead(const ModelContext& context) {
    init_lm_head(*this, context);
  }

  ModelOutput forward(const torch::Tensor& tokens,
                      const torch::Tensor& positions,
                      std::vector<KVCache>& kv_caches,
                      const ModelParamsView& parameters) {
    (void)tokens;
    (void)positions;
    (void)kv_caches;
    (void)parameters;
    return ModelOutput();
  }

  void load_model(std::unique_ptr<ModelLoader> loader) {
    load_lm_head_weights(loader);
  }

  torch::Tensor logits(const torch::Tensor& hidden_states,
                       const torch::Tensor& seleted_idxes,
                       torch::Tensor& out_hidden) {
    return forward_lm_head(hidden_states, seleted_idxes, out_hidden);
  }
};

// Registry-style holder so CausalLMImpl can own the family by pointer.
using FamilyWithHeadHolder = torch::nn::ModuleHolder<FamilyWithHead>;

class CausalLMImplLogitsWithHiddenTest : public ::testing::Test {
 protected:
  CausalLMImplLogitsWithHiddenTest() {
    lm_ = std::make_unique<CausalLMImpl<FamilyWithHeadHolder>>(
        FamilyWithHeadHolder(testutil::make_lm_test_context(/*hidden_size=*/3,
                                                            /*vocab_size=*/3)),
        testutil::make_lm_test_context(/*hidden_size=*/3, /*vocab_size=*/3));
    lm_->load_model(std::make_unique<testutil::HeadOnlyModelLoader>(
        3.0f * torch::eye(3, torch::TensorOptions().dtype(torch::kFloat32))));
  }

  std::unique_ptr<CausalLMImpl<FamilyWithHeadHolder>> lm_;

  torch::Tensor make_hidden() {
    return torch::arange(0, 12, torch::TensorOptions().dtype(torch::kFloat32))
        .reshape({4, 3});
  }
};

TEST_F(CausalLMImplLogitsWithHiddenTest, GathersSelectedHiddenRows) {
  torch::Tensor hidden = make_hidden();
  torch::Tensor idxes =
      torch::tensor({3, 0}, torch::TensorOptions().dtype(torch::kInt32));

  torch::Tensor out_hidden;
  torch::Tensor logits = lm_->logits(hidden, idxes, out_hidden);

  torch::Tensor expected =
      hidden.index_select(/*dim=*/0, idxes.to(torch::kLong));
  EXPECT_TRUE(out_hidden.defined());
  EXPECT_TRUE(torch::equal(out_hidden, expected));
  EXPECT_TRUE(torch::equal(logits, expected * 3.0f));
}

TEST_F(CausalLMImplLogitsWithHiddenTest, PassesThroughHiddenWithoutSelection) {
  torch::Tensor hidden = make_hidden();
  torch::Tensor undefined_idxes;

  torch::Tensor out_hidden;
  torch::Tensor logits = lm_->logits(hidden, undefined_idxes, out_hidden);

  EXPECT_TRUE(out_hidden.defined());
  EXPECT_TRUE(torch::equal(out_hidden, hidden));
  EXPECT_TRUE(torch::equal(logits, hidden * 3.0f));
}

class HybridBackbone {
 public:
  ModelOutput forward(const torch::Tensor& tokens,
                      const torch::Tensor& positions,
                      std::vector<KVCache>& kv_caches,
                      const ModelParamsView& parameters) {
    (void)tokens;
    (void)positions;
    (void)kv_caches;
    (void)parameters;
    return ModelOutput();
  }

  void load_model(std::unique_ptr<ModelLoader> loader) { (void)loader; }

  bool is_hybrid_linear_attention() { return true; }
};

TEST(CausalLMImplForwardOptional, HookIsDispatchedThroughBasePointer) {
  auto model = std::make_shared<HybridBackbone>();
  CausalLMImpl<std::shared_ptr<HybridBackbone>> lm(
      model,
      testutil::make_lm_test_context(/*hidden_size=*/3, /*vocab_size=*/3));
  CausalLM* base = &lm;
  EXPECT_TRUE(base->is_hybrid_linear_attention());
}

// Headless backbone: no LmHeadOwner, exercises the base-default paths.
class HeadlessBackbone {
 public:
  ModelOutput forward(const torch::Tensor& tokens,
                      const torch::Tensor& positions,
                      std::vector<KVCache>& kv_caches,
                      const ModelParamsView& parameters) {
    (void)tokens;
    (void)positions;
    (void)kv_caches;
    (void)parameters;
    return ModelOutput();
  }

  void load_model(std::unique_ptr<ModelLoader> loader) { (void)loader; }
};

TEST(CausalLMImplForwardOptional, BaseDefaultUsedWhenBackboneLacksHook) {
  auto model = std::make_shared<HeadlessBackbone>();
  CausalLMImpl<std::shared_ptr<HeadlessBackbone>> lm(
      model,
      testutil::make_lm_test_context(/*hidden_size=*/3, /*vocab_size=*/3));
  CausalLM* base = &lm;
  EXPECT_FALSE(base->is_hybrid_linear_attention());
}

}  // namespace
}  // namespace xllm

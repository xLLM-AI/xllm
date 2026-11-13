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
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "core/framework/model/logits_processor.h"
#include "core/framework/model/model_output.h"
#include "core/framework/model_context/model_context.h"
#include "core/framework/model_loader/model_loader.h"
#include "core/framework/parallel_state/parallel_args.h"
#include "core/framework/parallel_state/process_group.h"
#include "core/framework/quantization/quant_args.h"

namespace xllm {
namespace {

inline ProcessGroup* test_process_group() {
  static ProcessGroup group(/*rank=*/0,
                            /*world_size=*/1,
                            torch::Device(torch::kCPU));
  return &group;
}

inline ModelContext make_lm_test_context(int64_t hidden_size,
                                         int64_t vocab_size) {
  ParallelArgs parallel_args(/*rank=*/0,
                             /*world_size=*/1,
                             test_process_group());
  parallel_args.tp_group_ = test_process_group();
  ModelArgs model_args;
  model_args.hidden_size(/*hidden_size=*/hidden_size);
  model_args.vocab_size(/*vocab_size=*/vocab_size);
  return ModelContext(
      parallel_args,
      model_args,
      QuantArgs(),
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
}

class HeadOnlyModelLoader final : public ModelLoader {
 public:
  explicit HeadOnlyModelLoader(torch::Tensor head_weight) {
    state_dicts_.emplace_back(std::make_unique<StateDict>(
        std::unordered_map<std::string, torch::Tensor>{
            {"lm_head.weight", std::move(head_weight)}}));
  }

  std::unique_ptr<Tokenizer> tokenizer() const override { return nullptr; }

  std::vector<std::unique_ptr<StateDict>>& get_state_dicts() override {
    return state_dicts_;
  }

  std::string model_weights_path() const override { return {}; }

 private:
  std::vector<std::unique_ptr<StateDict>> state_dicts_;
};

class FamilyWithHead : public torch::nn::Module, public detail::LmHeadOwner {
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
};

using FamilyWithHeadHolder = torch::nn::ModuleHolder<FamilyWithHead>;

class CausalLMImplLogitsWithHiddenTest : public ::testing::Test {
 protected:
  CausalLMImplLogitsWithHiddenTest() {
    lm_ = std::make_unique<CausalLMImpl<FamilyWithHeadHolder>>(
        FamilyWithHeadHolder(make_lm_test_context(/*hidden_size=*/3,
                                                  /*vocab_size=*/3)),
        make_lm_test_context(/*hidden_size=*/3, /*vocab_size=*/3));
    lm_->load_model(std::make_unique<HeadOnlyModelLoader>(
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

  LogitsProcessor logits_processor;
  torch::Tensor out_hidden;
  torch::Tensor logits =
      logits_processor.forward(*lm_, hidden, idxes, out_hidden);

  torch::Tensor expected =
      hidden.index_select(/*dim=*/0, idxes.to(torch::kLong));
  EXPECT_TRUE(out_hidden.defined());
  EXPECT_TRUE(torch::equal(out_hidden, expected));
  EXPECT_TRUE(torch::equal(logits, expected * 3.0f));
}

TEST_F(CausalLMImplLogitsWithHiddenTest, PassesThroughHiddenWithoutSelection) {
  torch::Tensor hidden = make_hidden();
  torch::Tensor undefined_idxes;

  LogitsProcessor logits_processor;
  torch::Tensor out_hidden;
  torch::Tensor logits =
      logits_processor.forward(*lm_, hidden, undefined_idxes, out_hidden);

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
      model, make_lm_test_context(/*hidden_size=*/3, /*vocab_size=*/3));
  CausalLM* base = &lm;
  EXPECT_TRUE(base->is_hybrid_linear_attention());
}

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
      model, make_lm_test_context(/*hidden_size=*/3, /*vocab_size=*/3));
  CausalLM* base = &lm;
  EXPECT_FALSE(base->is_hybrid_linear_attention());
}

}  // namespace
}  // namespace xllm

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

#include "core/framework/model_context/model_context.h"

#include <gtest/gtest.h>

#include <utility>

#include "core/framework/model_context/dit_model_context.h"

namespace xllm {
namespace {

template <typename Context>
concept HasRvalueLlmModelArgs =
    requires(Context&& context) { std::move(context).get_llm_model_args(); };

template <typename Context>
concept HasRvalueVlmModelArgs =
    requires(Context&& context) { std::move(context).get_vlm_model_args(); };

template <typename Context>
concept HasRvalueRecModelArgs =
    requires(Context&& context) { std::move(context).get_rec_model_args(); };

template <typename Context>
concept HasRvalueDitModelArgs =
    requires(Context&& context) { std::move(context).get_dit_model_args(); };

template <typename Context>
concept HasRvalueDitComponentModelArgs = requires(Context&& context) {
  std::move(context).get_dit_model_args("transformer");
};

static_assert(!HasRvalueLlmModelArgs<ModelContext>);
static_assert(!HasRvalueVlmModelArgs<ModelContext>);
static_assert(!HasRvalueRecModelArgs<ModelContext>);
static_assert(!HasRvalueDitModelArgs<ModelContext>);
static_assert(!HasRvalueDitComponentModelArgs<DiTModelContext>);

ModelContext make_context(const ModelArgs& args,
                          ModelContextIdentity identity = {}) {
  const ParallelArgs parallel_args(
      /*rank=*/0, /*world_size=*/1, /*process_group=*/nullptr);
  const QuantArgs quant_args;
  const torch::TensorOptions options =
      torch::TensorOptions().device(torch::kCPU);
#if defined(USE_NPU)
  // Reuse the external ATB context overload to avoid initializing a device.
  return ModelContext(parallel_args,
                      args,
                      quant_args,
                      options,
                      /*context=*/nullptr,
                      std::move(identity));
#else
  return ModelContext(
      parallel_args, args, quant_args, options, std::move(identity));
#endif
}

TEST(ModelContextTest, KeepsLegacyContextsUnclassified) {
  const ModelContext empty_context;
  EXPECT_EQ(empty_context.get_model_family(), ModelFamily::UNKNOWN);
  EXPECT_TRUE(empty_context.get_registration_name().empty());

  ModelArgs args{};
  args.model_type() = "parser_model_type";
  args.hidden_size() = 128;
  const ModelContext context = make_context(args);
  EXPECT_EQ(context.get_model_family(), ModelFamily::UNKNOWN);
  EXPECT_TRUE(context.get_registration_name().empty());
  EXPECT_EQ(context.get_model_args().hidden_size(), 128);
  EXPECT_DEATH(context.get_llm_model_args(), "require an LLM model context");
}

TEST(ModelContextTest, ExposesEachRegisteredFamilyThroughItsView) {
  ModelArgs args{};
  args.model_type() = "parser_model_type";
  args.hidden_size() = 128;
  args.use_moe() = true;
  args.mm_patch_size() = 14;
  args.query_num() = 64;
  args.in_channels() = 16;

  const ModelContext llm =
      make_context(args, {ModelFamily::LLM, "registered_llm"});
  EXPECT_EQ(llm.get_registration_name(), "registered_llm");
  EXPECT_TRUE(llm.get_llm_model_args().use_moe());
  EXPECT_EQ(&llm.get_llm_model_args().transformer().hidden_size(),
            &llm.get_model_args().hidden_size());

  const ModelContext vlm =
      make_context(args, {ModelFamily::VLM, "registered_vlm"});
  EXPECT_EQ(vlm.get_vlm_model_args().vision().mm_patch_size(), 14);
  EXPECT_EQ(vlm.get_vlm_model_args().transformer().hidden_size(), 128);

  const ModelContext rec =
      make_context(args, {ModelFamily::REC, "registered_rec"});
  EXPECT_EQ(rec.get_rec_model_args().query_num(), 64);

  const ModelContext dit =
      make_context(args, {ModelFamily::DIT, "registered_dit"});
  EXPECT_EQ(dit.get_dit_model_args().transformer().in_channels(), 16);
}

TEST(ModelContextTest, RejectsViewsFromOtherFamilies) {
  const ModelArgs args{};
  const ModelContext llm = make_context(args, {ModelFamily::LLM, "llm"});
  EXPECT_DEATH(llm.get_vlm_model_args(), "require a VLM model context");
  EXPECT_DEATH(llm.get_rec_model_args(), "require a Rec model context");
  EXPECT_DEATH(llm.get_dit_model_args(), "require a DiT model context");

  const ModelContext vlm = make_context(args, {ModelFamily::VLM, "vlm"});
  EXPECT_DEATH(vlm.get_llm_model_args(), "require an LLM model context");
}

TEST(ModelContextTest, CopiedContextsBorrowTheirOwnArguments) {
  ModelArgs args{};
  args.hidden_size() = 128;
  const ModelContext context = make_context(args, {ModelFamily::LLM, "llm"});
  const ModelContext copied_context = context;
  args.hidden_size() = 256;

  EXPECT_EQ(copied_context.get_llm_model_args().transformer().hidden_size(),
            128);
  EXPECT_EQ(&copied_context.get_llm_model_args().transformer().hidden_size(),
            &copied_context.get_model_args().hidden_size());
  EXPECT_NE(&copied_context.get_llm_model_args().transformer().hidden_size(),
            &context.get_llm_model_args().transformer().hidden_size());
  EXPECT_EQ(copied_context.get_model_family(), ModelFamily::LLM);
  EXPECT_EQ(copied_context.get_registration_name(), "llm");
}

TEST(ModelContextTest, DerivedContextsKeepRegistrationAndOwnArguments) {
  ModelArgs args{};
  args.hidden_size() = 128;
  const ModelContext context =
      make_context(args, {ModelFamily::VLM, "registered_vlm"});
  const ParallelArgs parallel_args(
      /*rank=*/1, /*world_size=*/2, /*process_group=*/nullptr);
  const ModelContext parallel_context =
      context.with_parallel_args(parallel_args);
  EXPECT_EQ(parallel_context.get_parallel_args().world_size(), 2);
  EXPECT_EQ(parallel_context.get_model_family(), ModelFamily::VLM);
  EXPECT_EQ(parallel_context.get_registration_name(), "registered_vlm");
  EXPECT_EQ(&parallel_context.get_vlm_model_args().transformer().hidden_size(),
            &parallel_context.get_model_args().hidden_size());

  QuantArgs quant_args;
  quant_args.bits() = 8;
  const ModelContext quant_context = context.with_quant_args(quant_args);
  EXPECT_EQ(quant_context.get_quant_args().bits(), 8);
  EXPECT_EQ(quant_context.get_model_family(), ModelFamily::VLM);
  EXPECT_EQ(quant_context.get_registration_name(), "registered_vlm");
  EXPECT_EQ(&quant_context.get_vlm_model_args().transformer().hidden_size(),
            &quant_context.get_model_args().hidden_size());
}

TEST(DiTModelContextTest, IdentifiesItsFamilyWithoutADevice) {
  const DiTModelContext context;
  EXPECT_EQ(context.get_model_family(), ModelFamily::DIT);
}

#if !defined(USE_NPU)
TEST(DiTModelContextTest, ViewsBorrowTheSelectedComponentArguments) {
  ModelArgs transformer_args{};
  transformer_args.model_type() = "unregistered_transformer_component";
  transformer_args.in_channels() = 16;
  ModelArgs vae_args{};
  vae_args.in_channels() = 4;
  vae_args.vae_dim() = 128;
  const ParallelArgs parallel_args(
      /*rank=*/0, /*world_size=*/1, /*process_group=*/nullptr);
  const DiTModelContext context(
      parallel_args,
      {{"transformer", transformer_args}, {"vae", vae_args}},
      {{"transformer", QuantArgs()}, {"vae", QuantArgs()}},
      torch::TensorOptions().device(torch::kCPU),
      DiTCacheConfig(),
      "pipeline_model_type");

  EXPECT_EQ(
      context.get_dit_model_args("transformer").transformer().in_channels(),
      16);
  EXPECT_EQ(context.get_dit_model_args("vae").transformer().in_channels(), 4);
  EXPECT_EQ(context.get_dit_model_args("vae").vae().vae_dim(), 128);
  EXPECT_EQ(&context.get_dit_model_args("vae").vae().vae_dim(),
            &context.get_model_args("vae").vae_dim());
  EXPECT_DEATH(context.get_dit_model_args("missing"), "model args not found");

#if defined(USE_CUDA) || defined(USE_MLU) || defined(USE_DCU)
  const ModelContext component_context =
      context.get_model_context("transformer");
  EXPECT_EQ(component_context.get_model_family(), ModelFamily::DIT);
  EXPECT_TRUE(component_context.get_registration_name().empty());
  EXPECT_EQ(component_context.get_dit_model_args().transformer().in_channels(),
            16);
#endif
}
#endif

}  // namespace
}  // namespace xllm

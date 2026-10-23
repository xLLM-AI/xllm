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

template <typename Context>
concept HasRvalueAttentionModelParams = requires(Context&& context) {
  std::move(context).get_attention_model_params();
};

template <typename Context>
concept HasRvalueLinearAttentionModelParams = requires(Context&& context) {
  std::move(context).get_linear_attention_model_params();
};

template <typename Context>
concept HasRvalueKVCacheModelParams = requires(Context&& context) {
  std::move(context).get_kv_cache_model_params();
};

template <typename Context>
concept HasRvalueMoEModelParams =
    requires(Context&& context) { std::move(context).get_moe_model_params(); };

template <typename Context>
concept HasRvalueSpeculativeModelParams = requires(Context&& context) {
  std::move(context).get_speculative_model_params();
};

template <typename Context>
concept HasRvalueEmbeddingModelParams = requires(Context&& context) {
  std::move(context).get_embedding_model_params();
};

template <typename Context>
concept HasRvalueExecutionModelParams = requires(Context&& context) {
  std::move(context).get_execution_model_params();
};

static_assert(!HasRvalueLlmModelArgs<ModelContext>);
static_assert(!HasRvalueVlmModelArgs<ModelContext>);
static_assert(!HasRvalueRecModelArgs<ModelContext>);
static_assert(!HasRvalueDitModelArgs<ModelContext>);
static_assert(!HasRvalueDitComponentModelArgs<DiTModelContext>);
static_assert(!HasRvalueAttentionModelParams<ModelContext>);
static_assert(!HasRvalueLinearAttentionModelParams<ModelContext>);
static_assert(!HasRvalueKVCacheModelParams<ModelContext>);
static_assert(!HasRvalueMoEModelParams<ModelContext>);
static_assert(!HasRvalueSpeculativeModelParams<ModelContext>);
static_assert(!HasRvalueEmbeddingModelParams<ModelContext>);
static_assert(!HasRvalueExecutionModelParams<ModelContext>);

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

template <typename FamilyArgs>
void expect_capability_storage(const ModelContext& context,
                               const FamilyArgs& args) {
  EXPECT_EQ(&context.get_attention_model_params(), &args.attention_params());
  EXPECT_EQ(&context.get_linear_attention_model_params(),
            &args.linear_attention_params());
  EXPECT_EQ(&context.get_kv_cache_model_params(), &args.kv_cache_params());
  EXPECT_EQ(&context.get_moe_model_params(), &args.moe_params());
  EXPECT_EQ(&context.get_speculative_model_params(),
            &args.speculative_params());
  EXPECT_EQ(&context.get_embedding_model_params(), &args.embedding_params());
  EXPECT_EQ(&context.get_execution_model_params(), &args.execution_params());
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
  EXPECT_DEATH(context.get_llm_model_args(), "require an LLM model context");
}

TEST(ModelContextTest, ExposesEachRegisteredFamilyThroughOwnedArguments) {
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
  EXPECT_EQ(&llm.get_llm_model_args(), &llm.get_llm_model_args());
  EXPECT_EQ(llm.get_llm_model_args().transformer().hidden_size(), 128);

  const ModelContext vlm =
      make_context(args, {ModelFamily::VLM, "registered_vlm"});
  EXPECT_EQ(vlm.get_vlm_model_args().vision().mm_patch_size(), 14);
  EXPECT_EQ(vlm.get_vlm_model_args().transformer().hidden_size(), 128);
  EXPECT_EQ(&vlm.get_vlm_model_args(), &vlm.get_vlm_model_args());

  const ModelContext rec =
      make_context(args, {ModelFamily::REC, "registered_rec"});
  EXPECT_EQ(rec.get_rec_model_args().query_num(), 64);
  EXPECT_EQ(&rec.get_rec_model_args(), &rec.get_rec_model_args());

  const ModelContext dit =
      make_context(args, {ModelFamily::DIT, "registered_dit"});
  EXPECT_EQ(dit.get_dit_model_args().transformer().in_channels(), 16);
  EXPECT_EQ(&dit.get_dit_model_args(), &dit.get_dit_model_args());
}

TEST(ModelContextTest, ExposesStableCapabilityParametersForEachFamily) {
  ModelArgs args{};
  args.head_dim() = 128;
  args.n_heads() = 16;
  args.full_attention_interval() = 2;
  args.max_seq_len() = 4096;
  args.use_moe() = true;
  args.n_routed_experts() = 32;
  args.query_num() = 64;

  const ModelContext llm =
      make_context(args, {ModelFamily::LLM, "registered_llm"});
  EXPECT_EQ(llm.get_attention_model_params().head_dim(), 128);
  EXPECT_EQ(llm.get_moe_model_params().n_routed_experts(), 32);
  EXPECT_EQ(llm.get_kv_cache_model_params().max_seq_len(), 4096);
  EXPECT_TRUE(llm.get_moe_model_params().use_moe());
  EXPECT_FALSE(llm.get_linear_attention_model_params().is_full_attention_layer(
      /*layer_id=*/0));
  EXPECT_TRUE(llm.get_linear_attention_model_params().is_full_attention_layer(
      /*layer_id=*/1));
  EXPECT_EQ(&llm.get_attention_model_params(),
            &llm.get_attention_model_params());
  expect_capability_storage(llm, llm.get_llm_model_args());

  const ModelContext vlm =
      make_context(args, {ModelFamily::VLM, "registered_vlm"});
  expect_capability_storage(vlm, vlm.get_vlm_model_args());

  const ModelContext rec =
      make_context(args, {ModelFamily::REC, "registered_rec"});
  EXPECT_EQ(rec.get_attention_model_params().head_dim(), 128);
  EXPECT_EQ(rec.get_embedding_model_params().query_num(), 64);
  EXPECT_EQ(rec.get_moe_model_params().n_routed_experts(), 32);
  EXPECT_EQ(&rec.get_kv_cache_model_params(), &rec.get_kv_cache_model_params());
  expect_capability_storage(rec, rec.get_rec_model_args());

  const ModelContext dit =
      make_context(args, {ModelFamily::DIT, "registered_dit"});
  expect_capability_storage(dit, dit.get_dit_model_args().transformer());
}

TEST(ModelContextTest, SynchronizesEmbeddingCapabilityOnMutation) {
  const ModelContext context =
      make_context(ModelArgs{}, {ModelFamily::VLM, "registered_vlm"});
  EXPECT_FALSE(context.get_embedding_model_params().encoder_embedding_mode());
  const EmbeddingModelParams& original_params =
      context.get_embedding_model_params();

  ModelContext updated = context;
  EXPECT_EQ(&updated.get_embedding_model_params(), &original_params);
  updated.set_encoder_embedding_mode(true);

  EXPECT_TRUE(updated.get_embedding_model_params().encoder_embedding_mode());
  EXPECT_TRUE(updated.get_vlm_model_args().encoder_embedding_mode());
  EXPECT_TRUE(updated.get_vlm_model_args()
                  .transformer()
                  .embedding_params()
                  .encoder_embedding_mode());
  EXPECT_FALSE(context.get_embedding_model_params().encoder_embedding_mode());
  EXPECT_FALSE(original_params.encoder_embedding_mode());
  EXPECT_EQ(&updated.get_embedding_model_params(),
            &updated.get_vlm_model_args().embedding_params());
  EXPECT_NE(&updated.get_embedding_model_params(),
            &context.get_embedding_model_params());
}

TEST(ModelContextTest, KeepsCapabilityReferencesOnUniqueContextMutation) {
  ModelContext context =
      make_context(ModelArgs{}, {ModelFamily::REC, "registered_rec"});
  const EmbeddingModelParams& embedding_params =
      context.get_embedding_model_params();
  const AttentionModelParams& attention_params =
      context.get_attention_model_params();

  context.set_encoder_embedding_mode(true);

  EXPECT_EQ(&context.get_embedding_model_params(), &embedding_params);
  EXPECT_TRUE(embedding_params.encoder_embedding_mode());
  EXPECT_TRUE(context.get_rec_model_args().encoder_embedding_mode());
  EXPECT_EQ(&context.get_attention_model_params(), &attention_params);
  expect_capability_storage(context, context.get_rec_model_args());
}

TEST(ModelContextTest, RejectsCapabilitiesForNonTransformerDitComponents) {
  ModelArgs args{};
  const ModelContext vae = make_context(
      args, {ModelFamily::DIT, "registered_vae", DiTComponentType::VAE});
  EXPECT_DEATH(vae.get_attention_model_params(),
               "require a DiT transformer component");
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

TEST(ModelContextTest, CopiedContextsShareArgumentStorage) {
  ModelArgs args{};
  args.hidden_size() = 128;
  const ModelContext context = make_context(args, {ModelFamily::LLM, "llm"});
  const ModelContext copied_context = context;
  args.hidden_size() = 256;

  EXPECT_EQ(copied_context.get_llm_model_args().transformer().hidden_size(),
            128);
  EXPECT_EQ(&copied_context.get_llm_model_args(),
            &context.get_llm_model_args());
  EXPECT_EQ(&copied_context.get_llm_model_args().transformer().hidden_size(),
            &context.get_llm_model_args().transformer().hidden_size());
  EXPECT_EQ(&copied_context.get_attention_model_params(),
            &context.get_attention_model_params());
  EXPECT_EQ(&copied_context.get_moe_model_params(),
            &context.get_moe_model_params());
  EXPECT_EQ(copied_context.get_model_family(), ModelFamily::LLM);
  EXPECT_EQ(copied_context.get_registration_name(), "llm");
}

TEST(ModelContextTest, KeepsArgumentsAfterSourceConfigIsDestroyed) {
  const ModelContext context = [] {
    ModelArgs args{};
    args.hidden_size() = 128;
    args.use_moe() = true;
    return make_context(args, {ModelFamily::LLM, "llm"});
  }();

  EXPECT_EQ(context.get_llm_model_args().transformer().hidden_size(), 128);
  EXPECT_TRUE(context.get_llm_model_args().use_moe());
  EXPECT_EQ(context.get_execution_model_params().hidden_size(), 128);
  EXPECT_TRUE(context.get_moe_model_params().use_moe());
  expect_capability_storage(context, context.get_llm_model_args());
}

TEST(ModelContextTest, MutatingCopiedContextDetachesArgumentStorage) {
  ModelArgs args{};
  args.hidden_size() = 128;
  const ModelContext context = make_context(args, {ModelFamily::VLM, "vlm"});
  ModelContext copied_context = context;

  copied_context.set_encoder_embedding_mode(true);

  EXPECT_TRUE(copied_context.get_vlm_model_args().encoder_embedding_mode());
  EXPECT_FALSE(context.get_vlm_model_args().encoder_embedding_mode());
  EXPECT_NE(&copied_context.get_vlm_model_args(),
            &context.get_vlm_model_args());
}

TEST(ModelContextTest, DerivedContextsShareArgumentStorage) {
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
  EXPECT_EQ(&parallel_context.get_vlm_model_args(),
            &context.get_vlm_model_args());
  EXPECT_EQ(&parallel_context.get_vlm_model_args().transformer().hidden_size(),
            &context.get_vlm_model_args().transformer().hidden_size());

  QuantArgs quant_args;
  quant_args.bits() = 8;
  const ModelContext quant_context = context.with_quant_args(quant_args);
  EXPECT_EQ(quant_context.get_quant_args().bits(), 8);
  EXPECT_EQ(quant_context.get_model_family(), ModelFamily::VLM);
  EXPECT_EQ(quant_context.get_registration_name(), "registered_vlm");
  EXPECT_EQ(&quant_context.get_vlm_model_args(), &context.get_vlm_model_args());
  EXPECT_EQ(&quant_context.get_vlm_model_args().transformer().hidden_size(),
            &context.get_vlm_model_args().transformer().hidden_size());
}

TEST(DiTModelContextTest, IdentifiesItsFamilyWithoutADevice) {
  const DiTModelContext context;
  EXPECT_EQ(context.get_model_family(), ModelFamily::DIT);
}

#if !defined(USE_NPU)
TEST(DiTModelContextTest, OwnsTheSelectedComponentArguments) {
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
  EXPECT_EQ(context.get_dit_model_args("vae").vae().vae_dim(), 128);
  EXPECT_DEATH(context.get_dit_model_args("vae").transformer(),
               "does not contain transformer");
  EXPECT_EQ(&context.get_dit_model_args("vae"),
            &context.get_dit_model_args("vae"));
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

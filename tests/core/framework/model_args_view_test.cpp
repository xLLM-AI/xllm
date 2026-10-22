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

#include "core/framework/model/model_args_view.h"

#include <gtest/gtest.h>

#include <type_traits>

namespace xllm {
namespace {

static_assert(!std::is_constructible_v<TransformerModelArgsView, ModelArgs&&>);
static_assert(!std::is_constructible_v<LlmModelArgsView, ModelArgs&&>);
static_assert(!std::is_constructible_v<VlmModelArgsView, ModelArgs&&>);
static_assert(!std::is_constructible_v<RecModelArgsView, ModelArgs&&>);
static_assert(
    !std::is_constructible_v<DiTTransformerModelArgsView, ModelArgs&&>);
static_assert(!std::is_constructible_v<DiTVaeModelArgsView, ModelArgs&&>);
static_assert(
    !std::is_constructible_v<DiTTextEncoderModelArgsView, ModelArgs&&>);
static_assert(!std::is_constructible_v<DiTSchedulerModelArgsView, ModelArgs&&>);
static_assert(!std::is_constructible_v<DiTModelArgsView, ModelArgs&&>);

TEST(ModelArgsViewTest, ExposesTransformerFieldsAsReadOnly) {
  ModelArgs args;
  args.model_type() = "llama";
  args.hidden_size() = 4096;
  args.n_layers() = 32;
  args.n_kv_heads() = 8;

  const TransformerModelArgsView view = make_transformer_model_args_view(args);
  EXPECT_EQ(view.model_type(), "llama");
  EXPECT_EQ(view.hidden_size(), 4096);
  EXPECT_EQ(view.n_layers(), 32);
  ASSERT_TRUE(view.n_kv_heads().has_value());
  EXPECT_EQ(view.n_kv_heads().value(), 8);
  static_assert(std::is_same_v<decltype(view.hidden_size()), const int64_t&>);
}

TEST(ModelArgsViewTest, ExposesLlmVlmRecAndDitContracts) {
  ModelArgs args;
  args.hidden_size() = 4096;
  args.use_moe() = true;
  args.vision_token_id() = 151655;
  args.mm_patch_size() = 14;
  args.query_num() = 64;
  args.encoder_embedding_mode() = true;
  args.in_channels() = 16;
  args.num_layers() = 19;
  args.txt_dim() = 1024;
  args.base_dim() = 128;
  args.vae_dim() = 256;
  args.vae_patch_size() = 4;
  args.d_model() = 4096;
  args.max_position_embeddings() = 77;
  args.num_train_timesteps() = 1000;
  args.use_dynamic_shifting() = true;

  const LlmModelArgsView llm = make_llm_model_args_view(args);
  EXPECT_TRUE(llm.use_moe());
  EXPECT_EQ(llm.transformer().hidden_size(), 4096);
  static_assert(std::is_same_v<decltype(llm.use_moe()), const bool&>);

  const VlmModelArgsView vlm = make_vlm_model_args_view(args);
  EXPECT_EQ(vlm.vision().vision_token_id(), 151655);
  EXPECT_EQ(vlm.vision().mm_patch_size(), 14);
  EXPECT_EQ(vlm.transformer().hidden_size(), 4096);

  const RecModelArgsView rec = make_rec_model_args_view(args);
  EXPECT_EQ(rec.query_num(), 64);
  EXPECT_TRUE(rec.encoder_embedding_mode());
  EXPECT_EQ(rec.transformer().hidden_size(), 4096);
  static_assert(std::is_same_v<decltype(rec.query_num()), const int32_t&>);

  const DiTModelArgsView dit = make_dit_model_args_view(args);
  EXPECT_EQ(dit.transformer().in_channels(), 16);
  EXPECT_EQ(dit.transformer().num_layers(), 19);
  EXPECT_EQ(dit.transformer().txt_dim(), 1024);
  EXPECT_EQ(dit.vae().base_dim(), 128);
  EXPECT_EQ(dit.vae().vae_dim(), 256);
  EXPECT_EQ(dit.vae().vae_patch_size(), 4);
  EXPECT_EQ(dit.text_encoder().d_model(), 4096);
  EXPECT_EQ(dit.text_encoder().max_position_embeddings(), 77);
  EXPECT_EQ(dit.scheduler().num_train_timesteps(), 1000);
  EXPECT_TRUE(dit.scheduler().use_dynamic_shifting());
}

}  // namespace
}  // namespace xllm

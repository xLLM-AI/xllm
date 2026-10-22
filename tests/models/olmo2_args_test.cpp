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

#include <gtest/gtest.h>

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <string>
#include <system_error>
#include <unordered_set>
#include <vector>

#include "core/framework/model_loader/hf_model_args.h"
#include "models/model_registry.h"

namespace xllm {
namespace {

class Olmo2ArgsTest : public ::testing::Test {
 protected:
  void SetUp() override {
    std::string pattern =
        (std::filesystem::temp_directory_path() / "xllm-olmo2-XXXXXX").string();
    std::vector<char> buffer(pattern.begin(), pattern.end());
    buffer.reserve(pattern.size() + 1);
    buffer.emplace_back('\0');
    const char* directory = mkdtemp(buffer.data());
    ASSERT_NE(directory, nullptr);
    directory_ = directory;
    // Exact official config: allenai/OLMo-2-0425-1B-Instruct, revision
    // 48d788eca847d4d7548f375ad03d3c9312f6139e. No weights/network needed.
    JsonReader reader;
    ASSERT_TRUE(reader.parse(OLMO2_CONFIG_PATH));
    config_ = reader.data();
  }

  void TearDown() override {
    if (!directory_.empty()) {
      // This fixture owns exactly one file; no recursive traversal is needed.
      std::error_code error;
      std::filesystem::remove(directory_ / "config.json", error);
      EXPECT_FALSE(error) << error.message();
      std::filesystem::remove(directory_, error);
      EXPECT_FALSE(error) << error.message();
    }
  }

  bool load(ModelArgs* args) const {
    {
      std::ofstream file(directory_ / "config.json");
      EXPECT_TRUE(file.is_open());
      file << config_.dump();
      EXPECT_TRUE(file.good());
    }
    return load_hf_model_args(directory_, "llm", args);
  }

  std::filesystem::path directory_;
  nlohmann::json config_;
};

TEST_F(Olmo2ArgsTest, LoadsOfficialCheckpointThroughProductionEntry) {
  ModelArgs args;
  ASSERT_TRUE(load(&args));
  EXPECT_EQ(args.model_type(), "olmo2");
  EXPECT_EQ(args.hidden_size(), 2048);
  EXPECT_EQ(args.n_layers(), 16);
  EXPECT_EQ(args.n_heads(), 16);
  ASSERT_TRUE(args.n_kv_heads().has_value());
  EXPECT_EQ(args.n_kv_heads().value(), 16);
  EXPECT_EQ(args.head_dim(), 128);
  EXPECT_EQ(args.intermediate_size(), 8192);
  EXPECT_EQ(args.vocab_size(), 100352);
  EXPECT_EQ(args.max_position_embeddings(), 4096);
  EXPECT_EQ(args.bos_token_id(), -1);
  EXPECT_EQ(args.eos_token_id(), 100257);
  EXPECT_EQ(args.pad_token_id(), 100277);
  EXPECT_EQ(args.stop_token_ids(), std::unordered_set<int32_t>{100257});
  EXPECT_FLOAT_EQ(args.rope_theta(), 500000.0f);
  EXPECT_FLOAT_EQ(args.rms_norm_eps(), 1e-6f);
  EXPECT_EQ(args.hidden_act(), "silu");
  EXPECT_EQ(args.dtype(), "bfloat16");
  EXPECT_FALSE(args.tie_word_embeddings());
  EXPECT_FALSE(args.attention_bias());
  EXPECT_FALSE(args.qkv_bias());
  EXPECT_FLOAT_EQ(args.attention_dropout(), 0.0f);
  EXPECT_FALSE(args.enable_mla());
}

TEST_F(Olmo2ArgsTest, RegistersConfigurationWithoutRuntimeFactories) {
  EXPECT_NE(ModelRegistry::get_model_args_loader("olmo2"), nullptr);
  EXPECT_EQ(ModelRegistry::get_causallm_factory("olmo2"), nullptr);
  EXPECT_EQ(ModelRegistry::get_rec_model_factory("olmo2"), nullptr);
  EXPECT_EQ(ModelRegistry::get_causalvlm_factory("olmo2"), nullptr);
  EXPECT_EQ(ModelRegistry::get_dit_model_factory("olmo2"), nullptr);
}

TEST_F(Olmo2ArgsTest, LoadsDistinctDimensionsWithoutArchitectureDefaults) {
  // Distinct layer/head counts catch swapped mappings that the official
  // checkpoint's equal 16-layer/16-head dimensions would hide.
  config_.update({{"hidden_size", 1536},
                  {"num_hidden_layers", 6},
                  {"num_attention_heads", 12},
                  {"num_key_value_heads", 3},
                  {"intermediate_size", 4096},
                  {"vocab_size", 65536},
                  {"max_position_embeddings", 8192},
                  {"bos_token_id", 1},
                  {"eos_token_id", 7},
                  {"pad_token_id", 0},
                  {"rope_theta", 20000},
                  {"rms_norm_eps", 2e-5},
                  {"torch_dtype", "float16"},
                  {"tie_word_embeddings", true}});
  ModelArgs args;
  ASSERT_TRUE(load(&args));
  EXPECT_EQ(args.hidden_size(), 1536);
  EXPECT_EQ(args.n_layers(), 6);
  EXPECT_EQ(args.n_heads(), 12);
  EXPECT_EQ(args.n_kv_heads().value(), 3);
  EXPECT_EQ(args.head_dim(), 128);
  EXPECT_EQ(args.intermediate_size(), 4096);
  EXPECT_EQ(args.vocab_size(), 65536);
  EXPECT_EQ(args.max_position_embeddings(), 8192);
  EXPECT_EQ(args.bos_token_id(), 1);
  EXPECT_EQ(args.eos_token_id(), 7);
  EXPECT_EQ(args.pad_token_id(), 0);
  EXPECT_FLOAT_EQ(args.rope_theta(), 20000.0f);
  EXPECT_FLOAT_EQ(args.rms_norm_eps(), 2e-5f);
  EXPECT_EQ(args.dtype(), "float16");
  EXPECT_TRUE(args.tie_word_embeddings());
}

TEST_F(Olmo2ArgsTest, RejectsMissingRequiredDimensions) {
  const nlohmann::json original = config_;
  for (const char* field : {"hidden_size",
                            "num_hidden_layers",
                            "num_attention_heads",
                            "intermediate_size",
                            "vocab_size",
                            "max_position_embeddings"}) {
    SCOPED_TRACE(field);
    config_ = original;
    config_.erase(field);
    ModelArgs args;
    EXPECT_FALSE(load(&args));
    config_[field] = nullptr;
    EXPECT_FALSE(load(&args));
  }
}

TEST_F(Olmo2ArgsTest, RejectsInvalidDimensionValuesAndTypes) {
  const nlohmann::json original = config_;
  for (const char* field : {"hidden_size",
                            "num_hidden_layers",
                            "num_attention_heads",
                            "intermediate_size",
                            "vocab_size",
                            "max_position_embeddings"}) {
    for (const nlohmann::json& value :
         {nlohmann::json(0),
          nlohmann::json(-1),
          nlohmann::json(1.5),
          nlohmann::json("16"),
          nlohmann::json(true),
          nlohmann::json(std::numeric_limits<uint64_t>::max())}) {
      SCOPED_TRACE(std::string(field) + "=" + value.dump());
      config_ = original;
      config_[field] = value;
      ModelArgs args;
      EXPECT_FALSE(load(&args));
    }
  }
}

TEST_F(Olmo2ArgsTest, RejectsInconsistentAttentionDimensions) {
  const nlohmann::json original = config_;
  for (const nlohmann::json& patch :
       {nlohmann::json{{"hidden_size", 2049}},
        nlohmann::json{{"num_attention_heads", 0}},
        nlohmann::json{{"num_key_value_heads", 0}},
        nlohmann::json{{"num_key_value_heads", 3}},
        nlohmann::json{{"num_key_value_heads", 32}},
        nlohmann::json{{"num_key_value_heads", 16.5}},
        nlohmann::json{{"head_dim", 64}},
        nlohmann::json{{"hidden_size", 2064}, {"head_dim", 129}}}) {
    SCOPED_TRACE(patch.dump());
    config_ = original;
    config_.update(patch);
    ModelArgs args;
    EXPECT_FALSE(load(&args));
  }
}

TEST_F(Olmo2ArgsTest, DerivesKvHeadsAndHeadDimension) {
  config_.erase("num_key_value_heads");
  ModelArgs args;
  ASSERT_TRUE(load(&args));
  EXPECT_EQ(args.n_kv_heads().value(), 16);
  EXPECT_EQ(args.head_dim(), 128);
  config_["num_key_value_heads"] = nullptr;
  ASSERT_TRUE(load(&args));
  EXPECT_EQ(args.n_kv_heads().value(), 16);
  config_["num_key_value_heads"] = 4;
  config_["head_dim"] = 128;
  ASSERT_TRUE(load(&args));
  EXPECT_EQ(args.n_kv_heads().value(), 4);
  EXPECT_EQ(args.head_dim(), 128);
}

TEST_F(Olmo2ArgsTest, UsesReferenceDefaultsForOptionalFields) {
  for (const char* field : {"num_key_value_heads",
                            "hidden_act",
                            "rms_norm_eps",
                            "rope_theta",
                            "rope_scaling",
                            "torch_dtype",
                            "tie_word_embeddings",
                            "attention_bias",
                            "attention_dropout",
                            "eos_token_id",
                            "pad_token_id"}) {
    config_.erase(field);
  }
  ModelArgs args;
  ASSERT_TRUE(load(&args));
  EXPECT_EQ(args.n_kv_heads().value(), 16);
  EXPECT_EQ(args.head_dim(), 128);
  EXPECT_EQ(args.hidden_act(), "silu");
  EXPECT_FLOAT_EQ(args.rms_norm_eps(), 1e-5f);
  EXPECT_FLOAT_EQ(args.rope_theta(), 10000.0f);
  EXPECT_TRUE(args.dtype().empty());
  EXPECT_FALSE(args.tie_word_embeddings());
  EXPECT_FALSE(args.attention_bias());
  EXPECT_EQ(args.bos_token_id(), -1);
  EXPECT_EQ(args.eos_token_id(), 50279);
  EXPECT_EQ(args.pad_token_id(), 1);
}

TEST_F(Olmo2ArgsTest, LoadsExplicitOptionalValuesAndDtypeAlias) {
  config_["dtype"] = "float32";
  config_["tie_word_embeddings"] = true;
  config_["attention_bias"] = true;
  config_["attention_dropout"] = 0.1;
  config_["hidden_act"] = "gelu";
  config_["bos_token_id"] = 100257;
  ModelArgs args;
  ASSERT_TRUE(load(&args));
  EXPECT_EQ(args.dtype(), "float32");
  EXPECT_TRUE(args.tie_word_embeddings());
  EXPECT_TRUE(args.attention_bias());
  EXPECT_TRUE(args.qkv_bias());
  EXPECT_FLOAT_EQ(args.attention_dropout(), 0.1f);
  EXPECT_EQ(args.hidden_act(), "gelu");
  EXPECT_EQ(args.bos_token_id(), 100257);
  config_.erase("torch_dtype");
  config_["dtype"] = "float16";
  ASSERT_TRUE(load(&args));
  EXPECT_EQ(args.dtype(), "float16");
}

TEST_F(Olmo2ArgsTest, AcceptsNullableSpecialTokens) {
  for (const char* field : {"bos_token_id", "pad_token_id", "eos_token_id"}) {
    config_[field] = nullptr;
  }
  ModelArgs args;
  ASSERT_TRUE(load(&args));
  EXPECT_EQ(args.bos_token_id(), -1);
  EXPECT_EQ(args.pad_token_id(), -1);
  EXPECT_EQ(args.eos_token_id(), -1);
  EXPECT_TRUE(args.stop_token_ids().empty());
}

TEST_F(Olmo2ArgsTest, LoadsUnscaledRopeParameters) {
  config_.erase("rope_theta");
  config_["rope_parameters"] = {{"rope_type", "default"},
                                {"rope_theta", 750000}};
  ModelArgs args;
  ASSERT_TRUE(load(&args));
  EXPECT_FLOAT_EQ(args.rope_theta(), 750000.0f);
  config_["rope_theta"] = 500000;
  ASSERT_TRUE(load(&args));
  EXPECT_FLOAT_EQ(args.rope_theta(), 750000.0f);
}

TEST_F(Olmo2ArgsTest, LoadsLegacyDefaultRopeParameters) {
  config_["rope_parameters"] = {{"type", "default"}, {"rope_theta", 750000}};
  ModelArgs args;
  ASSERT_TRUE(load(&args));
  EXPECT_FLOAT_EQ(args.rope_theta(), 750000.0f);
}

TEST_F(Olmo2ArgsTest, RejectsLegacyScaledRopeParameters) {
  // LongRoPE can omit factor and derive scaling from the original context.
  // Checking only factor would silently accept the legacy type alias.
  config_["rope_parameters"] = {{"type", "longrope"},
                                {"rope_theta", 500000},
                                {"short_factor", std::vector<double>(64, 1.0)},
                                {"long_factor", std::vector<double>(64, 2.0)},
                                {"original_max_position_embeddings", 2048}};
  ModelArgs args;
  args.model_type("unchanged");
  EXPECT_FALSE(load(&args));
  EXPECT_EQ(args.model_type(), "unchanged");
}

TEST_F(Olmo2ArgsTest, CanonicalRopeTypeOverridesLegacyAlias) {
  config_["rope_parameters"] = {
      {"type", "longrope"}, {"rope_type", "default"}, {"rope_theta", 750000}};
  ModelArgs args;
  ASSERT_TRUE(load(&args));
  EXPECT_FLOAT_EQ(args.rope_theta(), 750000.0f);
  config_["rope_parameters"]["type"] = "default";
  config_["rope_parameters"]["rope_type"] = "longrope";
  EXPECT_FALSE(load(&args));
}

TEST_F(Olmo2ArgsTest, LoadsMultipleEosTokens) {
  config_["eos_token_id"] = {100257, 100278};
  ModelArgs args;
  ASSERT_TRUE(load(&args));
  EXPECT_EQ(args.eos_token_id(), 100257);
  EXPECT_EQ(args.eos_token_id_vec(), (std::vector<int32_t>{100257, 100278}));
  EXPECT_EQ(args.stop_token_ids(),
            (std::unordered_set<int32_t>{100257, 100278}));
}

TEST_F(Olmo2ArgsTest, RejectsInvalidSpecialTokens) {
  const nlohmann::json original = config_;
  for (const char* field : {"bos_token_id", "pad_token_id", "eos_token_id"}) {
    for (const nlohmann::json& value :
         {nlohmann::json(-2),
          nlohmann::json(100352),
          nlohmann::json(3.5),
          nlohmann::json("100257"),
          nlohmann::json(true),
          nlohmann::json(std::numeric_limits<int64_t>::max()),
          nlohmann::json(std::numeric_limits<uint64_t>::max())}) {
      SCOPED_TRACE(std::string(field) + "=" + value.dump());
      config_ = original;
      config_[field] = value;
      ModelArgs args;
      EXPECT_FALSE(load(&args));
    }
  }
  for (const nlohmann::json& value : {nlohmann::json::array(),
                                      nlohmann::json::array({-1}),
                                      nlohmann::json::array({100352}),
                                      nlohmann::json::array({true}),
                                      nlohmann::json::array({1.5}),
                                      nlohmann::json::object()}) {
    config_ = original;
    config_["eos_token_id"] = value;
    ModelArgs args;
    EXPECT_FALSE(load(&args));
  }
}

TEST_F(Olmo2ArgsTest, RejectsInvalidNormRopeDtypeAndOptionalTypes) {
  const nlohmann::json original = config_;
  for (const nlohmann::json& patch :
       {nlohmann::json{{"rms_norm_eps", 0}},
        nlohmann::json{{"rms_norm_eps", -1.0}},
        nlohmann::json{{"rms_norm_eps", "1e-6"}},
        nlohmann::json{{"rope_theta", 0}},
        nlohmann::json{{"rope_theta", 1e100}},
        nlohmann::json{
            {"rope_scaling", {{"rope_type", "linear"}, {"factor", 2}}}},
        nlohmann::json{{"rope_parameters", 1}},
        nlohmann::json{{"rope_parameters", {{"rope_type", "linear"}}}},
        nlohmann::json{{"rope_parameters", {{"factor", 2}}}},
        nlohmann::json{{"rope_parameters", {{"rope_theta", 0}}}},
        nlohmann::json{{"rope_parameters", {{"rope_theta", "500000"}}}},
        nlohmann::json{{"rope_parameters", {{"rope_type", true}}}},
        nlohmann::json{{"torch_dtype", "int8"}},
        nlohmann::json{{"torch_dtype", 16}},
        nlohmann::json{{"tie_word_embeddings", "false"}},
        nlohmann::json{{"attention_bias", 1}},
        nlohmann::json{{"attention_dropout", -0.1}},
        nlohmann::json{{"attention_dropout", 1.0}},
        nlohmann::json{{"hidden_act", ""}}}) {
    SCOPED_TRACE(patch.dump());
    config_ = original;
    config_.update(patch);
    ModelArgs args;
    EXPECT_FALSE(load(&args));
  }
}

TEST_F(Olmo2ArgsTest, FailureLeavesOutputUnchanged) {
  config_["hidden_size"] = 2049;
  ModelArgs args;
  args.model_type("unchanged");
  args.hidden_size(777);
  EXPECT_FALSE(load(&args));
  EXPECT_EQ(args.model_type(), "unchanged");
  EXPECT_EQ(args.hidden_size(), 777);
}

TEST_F(Olmo2ArgsTest, RejectsNullOutput) { EXPECT_FALSE(load(nullptr)); }

}  // namespace
}  // namespace xllm

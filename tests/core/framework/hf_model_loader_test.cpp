/* Copyright 2025-2026 The xLLM Authors.

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

#include "core/framework/model_loader/hf_model_loader.h"

#include <gtest/gtest.h>

#include <filesystem>
#include <string>
#include <string_view>
#include <type_traits>

#include "core/framework/kv_cache/kv_cache_estimation.h"
#include "core/framework/kv_cache/kv_cache_shape.h"
#include "core/framework/model/rec_causal_lm.h"
#include "core/framework/speculative/mtp_utils.h"
#include "core/platform/device.h"
#include "core/platform/platform.h"
#include "core/runtime/options.h"
#include "core/util/model_config_utils.h"
#include "models/model_registry.h"

namespace xllm {

namespace {

using ExpectedRecModelFactory =
    std::function<std::unique_ptr<RecCausalLM>(const ModelContext& context)>;

static_assert(std::is_same_v<RecModelFactory, ExpectedRecModelFactory>,
              "RecModelFactory must return std::unique_ptr<RecCausalLM>.");
static_assert(std::is_base_of_v<CausalLM, RecCausalLM>,
              "RecCausalLM must derive from CausalLM.");

class DummyRecCausalLM final : public RecCausalLM {
 public:
  explicit DummyRecCausalLM(const torch::TensorOptions& options)
      : options_(options) {}

  ModelOutput forward(const torch::Tensor& tokens,
                      const torch::Tensor& positions,
                      std::vector<KVCache>& kv_caches,
                      const ModelInputParams& parameters) override {
    UNUSED_PARAMETER(tokens);
    UNUSED_PARAMETER(positions);
    UNUSED_PARAMETER(kv_caches);
    UNUSED_PARAMETER(parameters);
    return ModelOutput();
  }

  torch::Tensor logits(const torch::Tensor& hidden_states,
                       const torch::Tensor& seleted_idxes) override {
    UNUSED_PARAMETER(hidden_states);
    UNUSED_PARAMETER(seleted_idxes);
    return torch::Tensor();
  }

  void load_model(std::unique_ptr<ModelLoader> loader) override {
    UNUSED_PARAMETER(loader);
  }

  torch::Device device() const override { return options_.device(); }

  void prepare_expert_weight(int32_t layer_id,
                             const std::vector<int32_t>& expert_ids) override {
    UNUSED_PARAMETER(layer_id);
    UNUSED_PARAMETER(expert_ids);
  }

  void update_expert_weight(int32_t layer_id) override {
    UNUSED_PARAMETER(layer_id);
  }

  const torch::TensorOptions& options() const override { return options_; }

 private:
  torch::TensorOptions options_;
};

}  // namespace

TEST(HFModelLoaderTest, Qwen35DenseBackendAwareModelTypeSelection) {
  JsonReader reader;
  ASSERT_TRUE(reader.parse_text(R"json(
    {
      "architectures": ["Qwen3_5ForConditionalGeneration"],
      "image_token_id": 248056,
      "model_type": "qwen3_5",
      "text_config": {
        "model_type": "qwen3_5_text"
      },
      "video_token_id": 248057,
      "vision_config": {
        "model_type": "qwen3_5"
      }
    }
  )json"));

  const std::filesystem::path fake_model_path("/tmp/Qwen3.5-9B");
  EXPECT_EQ(util::get_model_type(reader, fake_model_path), "qwen3_5_text");
  EXPECT_EQ(util::get_model_type(reader, fake_model_path, "llm"),
            "qwen3_5_text");
  EXPECT_EQ(util::get_model_type(reader, fake_model_path, "vlm"), "qwen3_5");
  EXPECT_EQ(ModelRegistry::get_model_backend("qwen3_5_text"), "llm");
}

TEST(HFModelLoaderTest, Qwen35MoeBackendAwareModelTypeSelection) {
  JsonReader reader;
  ASSERT_TRUE(reader.parse_text(R"json(
    {
      "architectures": ["Qwen3_5MoeForConditionalGeneration"],
      "image_token_id": 248056,
      "model_type": "qwen3_5_moe",
      "text_config": {
        "model_type": "qwen3_5_moe_text",
        "num_experts": 256,
        "num_experts_per_tok": 8
      },
      "video_token_id": 248057,
      "vision_config": {
        "model_type": "qwen3_5_moe"
      }
    }
  )json"));

  const std::filesystem::path fake_model_path("/tmp/Qwen3.5-MoE");
  EXPECT_EQ(util::get_model_type(reader, fake_model_path), "qwen3_5_moe_text");
  EXPECT_EQ(util::get_model_type(reader, fake_model_path, "llm"),
            "qwen3_5_moe_text");
  EXPECT_EQ(util::get_model_type(reader, fake_model_path, "vlm"),
            "qwen3_5_moe");
  EXPECT_EQ(ModelRegistry::get_model_backend("qwen3_5_moe_text"), "llm");
}

#if defined(USE_MLU)
TEST(HFModelLoaderTest, Qwen35MoeRootTypeHasCausalModelFactory) {
  CausalLMFactory factory = ModelRegistry::get_causallm_factory("qwen3_5_moe");
  EXPECT_TRUE(static_cast<bool>(factory));
}
#endif

TEST(HFModelLoaderTest, LoadCompressedTensorsFp8StaticConfig) {
  JsonReader reader;
  ASSERT_TRUE(reader.parse_text(R"json(
    {
      "quantization_config": {
        "config_groups": {
          "group_0": {
            "input_activations": {
              "dynamic": false,
              "num_bits": 8,
              "type": "float"
            },
            "weights": {
              "num_bits": 8,
              "type": "float"
            }
          }
        },
        "ignore": [
          "lm_head",
          "model.layers.1.mlp.down_proj"
        ],
        "quant_method": "compressed-tensors"
      }
    }
  )json"));

  QuantArgs quant_args;
  ASSERT_TRUE(load_quant_cfg(reader, quant_args));
  EXPECT_EQ(quant_args.quant_method(), kQuantMethodFp8);
  EXPECT_EQ(quant_args.bits(), 8);
  EXPECT_EQ(quant_args.moe_weight_bits(), 8);
  EXPECT_FALSE(quant_args.activation_dynamic());
  ASSERT_EQ(quant_args.ignored_modules().size(), 2);
  EXPECT_EQ(quant_args.ignored_modules()[0], "lm_head");
  EXPECT_EQ(quant_args.ignored_modules()[1], "model.layers.1.mlp.down_proj");
}

TEST(HFModelLoaderTest, KeepLegacyFp8ConfigUnchanged) {
  JsonReader reader;
  ASSERT_TRUE(reader.parse_text(R"json(
    {
      "quantization_config": {
        "activation_scheme": "static",
        "quant_method": "fp8"
      }
    }
  )json"));

  QuantArgs quant_args;
  ASSERT_TRUE(load_quant_cfg(reader, quant_args));
  EXPECT_EQ(quant_args.quant_method(), kQuantMethodFp8);
  EXPECT_FALSE(quant_args.activation_dynamic());
}

TEST(HFModelLoaderTest, RegisterRecFactoryAcceptsRecCausalLmReturnType) {
  const std::string factory_name = "rec_causallm_factory_contract_test";
  RecModelFactory unregistered_factory =
      ModelRegistry::get_rec_model_factory(factory_name);
  EXPECT_FALSE(static_cast<bool>(unregistered_factory));

  ModelRegistry::register_rec_model_factory(
      factory_name,
      [](const ModelContext& context) -> std::unique_ptr<RecCausalLM> {
        UNUSED_PARAMETER(context);
        return nullptr;
      });

  RecModelFactory factory = ModelRegistry::get_rec_model_factory(factory_name);
  EXPECT_TRUE(static_cast<bool>(factory));
}

TEST(HFModelLoaderTest, RecFactoryCreatesRecCausalLmInstance) {
  const std::string kFactoryName = "rec_causallm_instance_contract_test";
  ModelRegistry::register_rec_model_factory(
      kFactoryName,
      [](const ModelContext& context) -> std::unique_ptr<RecCausalLM> {
        UNUSED_PARAMETER(context);
        const torch::TensorOptions options =
            torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
        return std::make_unique<DummyRecCausalLM>(options);
      });

  RecModelFactory factory = ModelRegistry::get_rec_model_factory(kFactoryName);
  ASSERT_TRUE(static_cast<bool>(factory));

  ModelContext context;
  std::unique_ptr<RecCausalLM> rec_model = factory(context);
  ASSERT_NE(rec_model, nullptr);

  CausalLM* causal_model = dynamic_cast<CausalLM*>(rec_model.get());
  EXPECT_NE(causal_model, nullptr);
  EXPECT_EQ(rec_model->device(), torch::Device(torch::kCPU));
}

#if defined(USE_NPU) || defined(USE_MLU)
#if defined(USE_NPU)
TEST(HFModelLoaderTest, Glm5NextNativeMtpUsesAppendedLayer) {
  auto loader = ModelRegistry::get_model_args_loader("glm5_next");
  ASSERT_NE(loader, nullptr);
  JsonReader reader;
  ASSERT_TRUE(reader.parse_text(R"json({
    "model_type": "glm5_next",
    "text_config": {
      "num_hidden_layers": 45,
      "num_nextn_predict_layers": 1,
      "first_k_dense_replace": 3,
      "index_share_for_mtp_iteration": true,
      "index_topk_freq": 1
    }
  })json"));
  ModelArgs target_args;
  ASSERT_TRUE(loader(reader, &target_args));
  EXPECT_EQ(target_args.num_nextn_predict_layers(), 1);
  EXPECT_EQ(target_args.mtp_start_layer_idx(), -1);
  EXPECT_FALSE(configure_glm5_next_mtp_args(target_args,
                                            /*speculative_algorithm=*/"MTP",
                                            /*is_draft_engine=*/false));
  ModelArgs draft_args = target_args;
  ASSERT_TRUE(configure_glm5_next_mtp_args(draft_args,
                                           /*speculative_algorithm=*/"MTP",
                                           /*is_draft_engine=*/true));
  EXPECT_EQ(target_args.model_type(), "glm5_next");
  EXPECT_EQ(target_args.n_layers(), 45);
  EXPECT_EQ(draft_args.model_type(), "glm5_next_mtp");
  EXPECT_EQ(draft_args.n_layers(), 1);
  EXPECT_EQ(draft_args.mtp_start_layer_idx(), 45);
  EXPECT_EQ(draft_args.first_k_dense_replace(), 0);
  EXPECT_EQ(draft_args.layer_types(),
            std::vector<std::string>({"deepseek_sparse_attention"}));
  EXPECT_EQ(draft_args.full_attn_layers(), std::vector<int32_t>({0}));
  EXPECT_EQ(draft_args.mlp_layer_types(), std::vector<std::string>({"sparse"}));
  EXPECT_EQ(draft_args.indexer_types(), std::vector<std::string>({"full"}));
  EXPECT_EQ(draft_args.index_topk_freq(), 2);
  EXPECT_EQ(draft_args.index_skip_topk_offset(), 0);
  EXPECT_EQ(draft_args.index_topk_pattern(), "S");
}

TEST(HFModelLoaderTest, Glm5NextExportedMtpKeepsNativeKeys) {
  auto loader = ModelRegistry::get_model_args_loader("glm5_next_mtp");
  ASSERT_NE(loader, nullptr);
  JsonReader reader;
  ASSERT_TRUE(reader.parse_text(R"json({
    "model_type": "glm5_next_mtp",
    "text_config": {
      "num_hidden_layers": 1,
      "num_nextn_predict_layers": 1,
      "first_k_dense_replace": 0,
      "layer_types": ["deepseek_sparse_attention"],
      "mlp_layer_types": ["sparse"],
      "indexer_types": ["full"]
    }
  })json"));
  ModelArgs args;
  ASSERT_TRUE(loader(reader, &args));
  EXPECT_FALSE(configure_glm5_next_mtp_args(args,
                                            /*speculative_algorithm=*/"MTP",
                                            /*is_draft_engine=*/true));
  EXPECT_EQ(args.model_type(), "glm5_next_mtp");
  EXPECT_EQ(args.n_layers(), 1);
  EXPECT_EQ(args.mtp_start_layer_idx(), -1);
}

TEST(HFModelLoaderTest, Glm5NextMtpNormalizationIsIdempotent) {
  ModelArgs args;
  args.model_type("glm5_next").n_layers(45).num_nextn_predict_layers(1);
  ASSERT_TRUE(configure_glm5_next_mtp_args(args,
                                           /*speculative_algorithm=*/"MTP",
                                           /*is_draft_engine=*/true));
  EXPECT_FALSE(configure_glm5_next_mtp_args(args,
                                            /*speculative_algorithm=*/"MTP",
                                            /*is_draft_engine=*/true));
  EXPECT_EQ(args.model_type(), "glm5_next_mtp");
  EXPECT_EQ(args.n_layers(), 1);
  EXPECT_EQ(args.mtp_start_layer_idx(), 45);
}

TEST(HFModelLoaderTest, Glm5NextMtpNormalizationIgnoresOtherAlgorithms) {
  for (const std::string_view algorithm : {"", "Eagle3", "DFlash", "DSpark"}) {
    SCOPED_TRACE(algorithm);
    ModelArgs args;
    args.model_type("glm5_next").n_layers(45).num_nextn_predict_layers(1);
    EXPECT_FALSE(configure_glm5_next_mtp_args(
        args, algorithm, /*is_draft_engine=*/true));
    EXPECT_EQ(args.model_type(), "glm5_next");
    EXPECT_EQ(args.n_layers(), 45);
    EXPECT_EQ(args.mtp_start_layer_idx(), -1);
  }
}

TEST(HFModelLoaderTest, Glm5NextMtpNormalizationIgnoresOtherModels) {
  for (const std::string model_type : {"deepseek_v3", "qwen3_5_text"}) {
    SCOPED_TRACE(model_type);
    ModelArgs args;
    args.model_type(model_type).n_layers(45).num_nextn_predict_layers(1);
    EXPECT_FALSE(configure_glm5_next_mtp_args(args,
                                              /*speculative_algorithm=*/"MTP",
                                              /*is_draft_engine=*/true));
    EXPECT_EQ(args.model_type(), model_type);
    EXPECT_EQ(args.n_layers(), 45);
    EXPECT_EQ(args.mtp_start_layer_idx(), -1);
  }
}

TEST(HFModelLoaderTest, Glm5NextMtpEngineAndWorkerMatchExportedCacheLayout) {
  auto loader = ModelRegistry::get_model_args_loader("glm5_next");
  ASSERT_NE(loader, nullptr);
  JsonReader reader;
  ASSERT_TRUE(reader.parse_text(R"json({
    "model_type": "glm5_next",
    "text_config": {
      "num_hidden_layers": 45,
      "num_nextn_predict_layers": 1,
      "index_share_for_mtp_iteration": true,
      "index_kpool_compress": true,
      "linear_attn_config": {
        "num_heads": 64,
        "head_dim": 128,
        "short_conv_kernel_size": 4
      }
    }
  })json"));
  ModelArgs engine_args;
  ASSERT_TRUE(loader(reader, &engine_args));
  ASSERT_TRUE(has_linear_attention_layers(engine_args));
  engine_args.enable_mla(true);
  ModelArgs worker_args = engine_args;
  runtime::Options engine_options;
  engine_options.is_draft_engine(true)
      .speculative_algorithm("mTp")
      .enable_speculative_decode(false);
  runtime::Options worker_options = engine_options;
  worker_options.enable_speculative_decode(true);
  ASSERT_TRUE(
      configure_glm5_next_mtp_args(engine_args,
                                   engine_options.speculative_algorithm(),
                                   engine_options.is_draft_engine()));
  ASSERT_TRUE(
      configure_glm5_next_mtp_args(worker_args,
                                   worker_options.speculative_algorithm(),
                                   worker_options.is_draft_engine()));

  JsonReader exported_reader;
  ASSERT_TRUE(exported_reader.parse_text(R"json({
    "model_type": "glm5_next_mtp",
    "text_config": {
      "num_hidden_layers": 1,
      "num_nextn_predict_layers": 1,
      "first_k_dense_replace": 0,
      "layer_types": ["deepseek_sparse_attention"],
      "mlp_layer_types": ["sparse"],
      "indexer_types": ["full"],
      "index_kpool_compress": true,
      "linear_attn_config": {
        "num_heads": 64,
        "head_dim": 128,
        "short_conv_kernel_size": 4
      }
    }
  })json"));
  ModelArgs exported_args;
  ASSERT_TRUE(loader(exported_reader, &exported_args));
  exported_args.enable_mla(true);
  EXPECT_FALSE(
      configure_glm5_next_mtp_args(exported_args,
                                   engine_options.speculative_algorithm(),
                                   engine_options.is_draft_engine()));

  KVCacheEstimateOptions cache_options;
  cache_options.cache_size_in_bytes = 1024 * 1024;
  cache_options.block_size = 128;
  cache_options.world_size = 8;
  cache_options.n_local_kv_heads = 8;
  cache_options.max_seqs_per_batch = 16;
  cache_options.is_draft_engine = true;
  const KVCacheCapacity exported_capacity =
      estimate_kv_cache_capacity(exported_args, cache_options);
  const KVCacheShape exported_shape(
      exported_capacity, exported_args, cache_options.world_size);
  for (const ModelArgs* model_args : {&engine_args, &worker_args}) {
    EXPECT_FALSE(has_linear_attention_layers(*model_args));
    EXPECT_EQ(model_args->mtp_start_layer_idx(), 45);
    const KVCacheCapacity capacity =
        estimate_kv_cache_capacity(*model_args, cache_options);
    EXPECT_EQ(capacity.n_layers(), 1);
    EXPECT_EQ(capacity.num_full_attention_layers(), 1);
    EXPECT_EQ(capacity.num_linear_attention_layers(), 0);
    EXPECT_EQ(capacity.num_indexer_layers(), 1);
    EXPECT_EQ(capacity.linear_cache_size_in_bytes(), 0);
    EXPECT_EQ(capacity.n_blocks(), exported_capacity.n_blocks());
    EXPECT_EQ(capacity.slot_size(), exported_capacity.slot_size());
    EXPECT_EQ(capacity.index_slot_size(), exported_capacity.index_slot_size());
    const KVCacheShape shape(capacity, *model_args, cache_options.world_size);
    EXPECT_EQ(shape.key_cache_shape(), exported_shape.key_cache_shape());
    EXPECT_EQ(shape.value_cache_shape(), exported_shape.value_cache_shape());
    ASSERT_TRUE(shape.has_index_cache_shape());
    EXPECT_EQ(shape.index_cache_shape(), exported_shape.index_cache_shape());
    EXPECT_FALSE(shape.has_conv_cache_shape());
    EXPECT_FALSE(shape.has_ssm_cache_shape());
  }
}

TEST(HFModelLoaderTest, Glm5NextNativeMtpRejectsUnsupportedLayerCounts) {
  ModelArgs args;
  args.model_type("glm5_next").n_layers(45);
  EXPECT_DEATH(configure_glm5_next_mtp_args(args,
                                            /*speculative_algorithm=*/"MTP",
                                            /*is_draft_engine=*/true),
               "exactly one");
  args.num_nextn_predict_layers(2);
  EXPECT_DEATH(configure_glm5_next_mtp_args(args,
                                            /*speculative_algorithm=*/"MTP",
                                            /*is_draft_engine=*/true),
               "exactly one");
  args.num_nextn_predict_layers(1).n_layers(0);
  EXPECT_DEATH(configure_glm5_next_mtp_args(args,
                                            /*speculative_algorithm=*/"MTP",
                                            /*is_draft_engine=*/true),
               "n_layers");
}

TEST(HFModelLoaderTest, Qwen3DSparkFieldsFromTorchConfig) {
  auto loader = ModelRegistry::get_model_args_loader("qwen3");
  ASSERT_NE(loader, nullptr);

  JsonReader reader;
  ASSERT_TRUE(reader.parse_text(R"json(
    {
      "model_type": "qwen3",
      "markov_rank": 256,
      "enable_confidence_head": true,
      "confidence_head_with_markov": true
    }
  )json"));

  ModelArgs args;
  ASSERT_TRUE(loader(reader, &args));
  EXPECT_EQ(args.markov_rank(), 256);
  EXPECT_TRUE(args.enable_confidence_head());
  EXPECT_TRUE(args.confidence_head_with_markov());
}

TEST(HFModelLoaderTest, Qwen3DFlash2AcceptsArrayEosTokenIds) {
  for (const char* model_name : {"qwen3", "qwen3_atb"}) {
    auto loader = ModelRegistry::get_model_args_loader(model_name);
    ASSERT_NE(loader, nullptr);

    JsonReader reader;
    ASSERT_TRUE(reader.parse_text(R"json(
      {
        "model_type": "qwen3",
        "eos_token_id": [154820, 154827, 154829],
        "dflash_config": {
          "block_size": 8,
          "conv_group_size": 16,
          "conv_kernel_size": 2,
          "selector_rank": 256,
          "selector_top_k": 16
        }
      }
    )json"));

    ModelArgs args;
    ASSERT_TRUE(loader(reader, &args));
    EXPECT_EQ(args.eos_token_id(), 154820);
    EXPECT_EQ(args.eos_token_id_vec(),
              std::vector<int32_t>({154820, 154827, 154829}));
    EXPECT_EQ(args.stop_token_ids(),
              std::unordered_set<int32_t>({154820, 154827, 154829}));
    if (std::string(model_name) == "qwen3") {
      EXPECT_EQ(args.dflash2_block_size(), 8);
      EXPECT_EQ(args.dflash2_selector_top_k(), 16);
    }
  }
}

TEST(HFModelLoaderTest, GlmMoeDsaMtpPreservesLayerTypeMetadata) {
  auto loader = ModelRegistry::get_model_args_loader("glm_moe_dsa_mtp");
  ASSERT_NE(loader, nullptr);

  JsonReader reader;
  ASSERT_TRUE(reader.parse_text(R"json(
    {
      "model_type": "glm_moe_dsa_mtp",
      "num_hidden_layers": 1,
      "indexer_types": ["full"],
      "mlp_layer_types": ["dense"],
      "index_topk_pattern": "S",
      "index_share_for_mtp_iteration": true
    }
  )json"));

  ModelArgs args;
  ASSERT_TRUE(loader(reader, &args));
  EXPECT_EQ(args.indexer_types(), std::vector<std::string>({"full"}));
  EXPECT_EQ(args.mlp_layer_types(), std::vector<std::string>({"dense"}));
  EXPECT_EQ(args.index_topk_pattern(), "S");
}
TEST(HFModelLoaderTest, DeepseekV4DSparkModelArgsFrom0731Config) {
  auto loader = ModelRegistry::get_model_args_loader("deepseek_v4");
  ASSERT_NE(loader, nullptr);

  JsonReader reader;
  ASSERT_TRUE(reader.parse_text(R"json(
    {
      "model_type": "deepseek_v4",
      "hidden_size": 4096,
      "num_hidden_layers": 43,
      "num_attention_heads": 32,
      "num_key_value_heads": 1,
      "compress_ratios": [1],
      "dspark_block_size": 5,
      "dspark_markov_rank": 256,
      "dspark_noise_token_id": 128799,
      "dspark_target_layer_ids": [40, 41, 42]
    }
  )json"));

  ModelArgs args;
  ASSERT_TRUE(loader(reader, &args));
  EXPECT_EQ(args.model_type(), "deepseek_v4");
  EXPECT_EQ(args.dspark_num_layers(), 3);
  // Base loader leaves this 0; applied later in
  // configure_deepseek_v4_dspark_args.
  EXPECT_EQ(args.dspark_block_size(), 0);
  EXPECT_EQ(args.markov_rank(), 256);
  ASSERT_EQ(args.compress_ratios().size(), 43);
}
#endif

TEST(HFModelLoaderTest, Qwen35MtpModelArgsFromDenseConfig) {
  auto loader = ModelRegistry::get_model_args_loader("qwen3_5_mtp");
  ASSERT_NE(loader, nullptr);

  JsonReader reader;
  ASSERT_TRUE(reader.parse_text(R"json(
    {
      "model_type": "qwen3_5",
      "text_config": {
        "mtp_num_hidden_layers": 1,
        "layer_types": ["linear_attention"]
      }
    }
  )json"));

  ModelArgs args;
  ASSERT_TRUE(loader(reader, &args));
  EXPECT_EQ(args.model_type(), "qwen3_5_mtp");
  EXPECT_EQ(args.num_nextn_predict_layers(), 1);
  EXPECT_EQ(args.n_layers(), 1);
  ASSERT_EQ(args.layer_types().size(), 1);
  EXPECT_EQ(args.layer_types()[0], "full_attention");
}

TEST(HFModelLoaderTest, Qwen35MtpModelArgsFromMoeConfig) {
  auto loader = ModelRegistry::get_model_args_loader("qwen3_5_moe_mtp");
  ASSERT_NE(loader, nullptr);

  JsonReader reader;
  ASSERT_TRUE(reader.parse_text(R"json(
    {
      "model_type": "qwen3_5_moe",
      "text_config": {
        "mtp_num_hidden_layers": 2,
        "layer_types": ["linear_attention", "linear_attention"]
      }
    }
  )json"));

  ModelArgs args;
  ASSERT_TRUE(loader(reader, &args));
  EXPECT_EQ(args.model_type(), "qwen3_5_moe_mtp");
  EXPECT_EQ(args.num_nextn_predict_layers(), 2);
  EXPECT_EQ(args.n_layers(), 2);
  ASSERT_EQ(args.layer_types().size(), 2);
  EXPECT_EQ(args.layer_types()[0], "full_attention");
  EXPECT_EQ(args.layer_types()[1], "full_attention");
}
#endif

#if defined(USE_CUDA)
TEST(HFModelLoaderTest, MiMoMtpModelArgsFromExportedConfig) {
  // export_mtp.py sets num_hidden_layers = mtp_layer_count (1) in the exported
  // config, matching mtp_layers.0.* weight layout loaded by MiMoMtpModelImpl.
  auto loader = ModelRegistry::get_model_args_loader("mimo_mtp");
  ASSERT_NE(loader, nullptr);

  JsonReader reader;
  ASSERT_TRUE(reader.parse_text(R"json(
    {
      "model_type": "mimo_mtp",
      "num_hidden_layers": 1,
      "num_attention_heads": 32,
      "num_key_value_heads": 8,
      "hidden_size": 4096,
      "intermediate_size": 11008,
      "rope_theta": 640000.0,
      "num_nextn_predict_layers": 1
    }
  )json"));

  ModelArgs args;
  ASSERT_TRUE(loader(reader, &args));
  EXPECT_EQ(args.model_type(), "mimo_mtp");
  EXPECT_EQ(args.n_layers(), 1);
  EXPECT_EQ(args.n_heads(), 32);
  EXPECT_EQ(args.n_kv_heads(), std::optional<int64_t>(8));
  EXPECT_EQ(args.hidden_size(), 4096);
  EXPECT_EQ(args.intermediate_size(), 11008);
  EXPECT_FLOAT_EQ(args.rope_theta(), 640000.0f);
  EXPECT_EQ(args.num_nextn_predict_layers(), 1);
  // head_dim is derived from hidden_size / n_heads when absent from config
  EXPECT_EQ(args.head_dim(), 128);
  EXPECT_EQ(args.stop_token_ids(),
            std::unordered_set<int32_t>({args.eos_token_id()}));
}

TEST(HFModelLoaderTest, MiMoMtpModelArgsDefaults) {
  // Verify REGISTER_MODEL_ARGS defaults match MiMo-7B-Base architecture.
  auto loader = ModelRegistry::get_model_args_loader("mimo_mtp");
  ASSERT_NE(loader, nullptr);

  JsonReader reader;
  ASSERT_TRUE(reader.parse_text(R"json({ "model_type": "mimo_mtp" })json"));

  ModelArgs args;
  ASSERT_TRUE(loader(reader, &args));
  EXPECT_EQ(args.model_type(), "mimo_mtp");
  EXPECT_EQ(args.n_layers(), 36);
  EXPECT_EQ(args.n_heads(), 32);
  EXPECT_EQ(args.hidden_size(), 4096);
  EXPECT_EQ(args.vocab_size(), 151680);
  EXPECT_FLOAT_EQ(args.rope_theta(), 640000.0f);
  EXPECT_EQ(args.num_nextn_predict_layers(), 1);
  EXPECT_EQ(args.head_dim(), 128);
  EXPECT_EQ(args.eos_token_id(), 151643);
  EXPECT_EQ(args.stop_token_ids(), std::unordered_set<int32_t>({151643}));
}

TEST(HFModelLoaderTest, Rwkv7ModelArgsFromConvertedConfig) {
  auto loader = ModelRegistry::get_model_args_loader("rwkv7");
  ASSERT_NE(loader, nullptr);

  JsonReader reader;
  ASSERT_TRUE(reader.parse_text(R"json(
    {
      "model_type": "rwkv7",
      "torch_dtype": "float16",
      "vocab_size": 65536,
      "hidden_size": 768,
      "num_hidden_layers": 12,
      "head_size": 64,
      "intermediate_size": 3072,
      "num_attention_heads": 12,
      "layer_norm_eps": 1e-5,
      "max_position_embeddings": 4096,
      "bos_token_id": 0,
      "eos_token_id": 0
    }
  )json"));

  ModelArgs args;
  ASSERT_TRUE(loader(reader, &args));
  EXPECT_EQ(args.model_type(), "rwkv7");
  EXPECT_EQ(args.vocab_size(), 65536);
  EXPECT_EQ(args.hidden_size(), 768);
  EXPECT_EQ(args.n_layers(), 12);
  EXPECT_EQ(args.head_dim(), 64);
  EXPECT_EQ(args.intermediate_size(), 3072);
  EXPECT_EQ(args.n_heads(), 12);
  EXPECT_EQ(args.max_position_embeddings(), 4096);
  EXPECT_FLOAT_EQ(args.layer_norm_eps(), 1e-5f);
  EXPECT_EQ(args.linear_conv_kernel_dim(), 2);
  EXPECT_EQ(args.full_attention_interval(), 13);
  EXPECT_TRUE(has_linear_attention_layers(args));
  ASSERT_EQ(args.layer_types().size(), 12U);
  EXPECT_EQ(args.layer_types().front(), "rwkv7");
  EXPECT_EQ(args.stop_token_ids(), std::unordered_set<int32_t>({0}));
}
#endif  // USE_CUDA

TEST(HFModelLoaderTest, LoadCompressedTensorsInt8Scheme) {
  struct TestCase {
    const char* name;
    const char* config;
    bool is_w8a8_dynamic;
  };
  const TestCase test_cases[] = {
      {"dynamic_activation",
       R"json(
         {
           "quantization_config": {
             "config_groups": {
               "group_0": {
                 "input_activations": {
                   "dynamic": true,
                   "num_bits": 8,
                   "type": "int"
                 },
                 "weights": {
                   "dynamic": false,
                   "num_bits": 8,
                   "type": "int"
                 }
               }
             },
             "ignore": [
               "lm_head",
               "model.layers.0.self_attn.o_proj"
             ],
             "quant_method": "compressed-tensors"
           }
         }
       )json",
       true},
      {"static_activation",
       R"json(
         {
           "quantization_config": {
             "config_groups": {
               "group_0": {
                 "input_activations": {
                   "dynamic": false,
                   "num_bits": 8,
                   "type": "int"
                 },
                 "weights": {
                   "dynamic": false,
                   "num_bits": 8,
                   "type": "int"
                 }
               }
             },
             "quant_method": "compressed-tensors"
           }
         }
       )json",
       false},
  };

  for (const TestCase& test_case : test_cases) {
    SCOPED_TRACE(test_case.name);
    JsonReader reader;
    ASSERT_TRUE(reader.parse_text(test_case.config));

    QuantArgs quant_args;
    EXPECT_EQ(load_quant_cfg(reader, quant_args), test_case.is_w8a8_dynamic);
    EXPECT_EQ(quant_args.quant_method(), "compressed-tensors");
    EXPECT_EQ(quant_args.is_compressed_tensors_w8a8_dynamic(),
              test_case.is_w8a8_dynamic);
    if (test_case.is_w8a8_dynamic) {
      EXPECT_EQ(quant_args.bits(), 8);
      EXPECT_EQ(quant_args.moe_weight_bits(), 8);
      EXPECT_TRUE(quant_args.activation_dynamic());
      ASSERT_EQ(quant_args.ignored_modules().size(), 2);
      EXPECT_EQ(quant_args.ignored_modules()[0], "lm_head");
      EXPECT_EQ(quant_args.ignored_modules()[1],
                "model.layers.0.self_attn.o_proj");
    }
  }
}

TEST(HFModelLoaderTest, LoadCompressedTensorsWithoutSmooth) {
  JsonReader reader;
  const std::string config = R"json({"quantization_config": {
    "quant_method": "compressed-tensors", "format": "int-quantized",
    "ignore": ["lm_head", "re:.*mlp\\.gate$"],
    "config_groups": {"group_0": {
      "targets": ["Linear"],
      "weights": {"type": "int", "num_bits": 8, "dynamic": false,
                  "strategy": "channel", "symmetric": true,
                  "preserve_smooth": false},
      "input_activations": {"type": "int", "num_bits": 8, "dynamic": true,
                            "strategy": "token", "symmetric": true}
    }}
  }})json";
  std::string text = config;
  ASSERT_TRUE(reader.parse_text(text));
  QuantArgs args;
  ASSERT_TRUE(load_quant_cfg(reader, args));
  EXPECT_EQ(args.quant_method(), "compressed-tensors");
  EXPECT_TRUE(args.is_compressed_tensors_w8a8_dynamic());
  EXPECT_EQ(args.bits(), 8);
  EXPECT_EQ(args.moe_weight_bits(), 8);
  EXPECT_TRUE(args.is_sym());
  EXPECT_TRUE(args.should_ignore_module("model.layers.0.mlp.gate"));
  text.replace(text.find("\"channel\""), 9, "\"group\"");
  ASSERT_TRUE(reader.parse_text(text));
  EXPECT_FALSE(load_quant_cfg(reader, args));
}

TEST(HFModelLoaderTest, RejectCompressedTensorsPreservedSmooth) {
  const auto base = nlohmann::json::parse(R"json({
    "quant_method": "compressed-tensors",
    "config_groups": {"group_0": {
      "weights": {"type": "int", "num_bits": 8},
      "input_activations": {"type": "int", "num_bits": 8, "dynamic": true}
    }}
  })json");
  for (const std::string& key : {"quantization_config", "compression_config"}) {
    for (const std::string& target : {"weights", "input_activations"}) {
      auto config = base;
      config["config_groups"]["group_0"][target]["preserve_smooth"] = true;
      SCOPED_TRACE(key + ": " + config.dump());
      JsonReader reader;
      ASSERT_TRUE(reader.parse_text(nlohmann::json({{key, config}}).dump()));
      QuantArgs args;
      EXPECT_FALSE(load_quant_cfg(reader, args));
    }
  }
}

TEST(HFModelLoaderTest, LoadCompressedTensorsConfigAlias) {
  for (const std::string& key : {"quantization_config", "compression_config"}) {
    JsonReader reader;
    ASSERT_TRUE(reader.parse_text("{\"" + key + R"json(": {
      "quant_method": "compressed-tensors",
      "ignore": ["lm_head"],
      "config_groups": {"group_0": {
        "weights": {"type": "int", "num_bits": 8},
        "input_activations": {"type": "int", "num_bits": 8, "dynamic": true}
      }}
    }})json"));
    QuantArgs args;
    ASSERT_TRUE(load_quant_cfg(reader, args));
    EXPECT_EQ(args.quant_method(), "compressed-tensors");
    EXPECT_TRUE(args.is_compressed_tensors_w8a8_dynamic());
    EXPECT_TRUE(args.is_sym());
    EXPECT_TRUE(args.should_ignore_module("lm_head"));
  }
}

TEST(HFModelLoaderTest, RejectUnsupportedCompressedTensorsScheme) {
  const auto base = nlohmann::json::parse(R"json({
    "quant_method": "compressed-tensors",
    "config_groups": {"group_0": {
      "weights": {"type": "int", "num_bits": 8},
      "input_activations": {"type": "int", "num_bits": 8, "dynamic": true}
    }}
  })json");
  const std::vector<nlohmann::json> overrides = {
      {{"format", "pack-quantized"}},
      {{"kv_cache_scheme", {{"num_bits", 8}}}},
      {{"transform_config", {{"rotation", true}}}},
      {{"config_groups", nullptr}},
      {{"config_groups", {{"group_0", {{"weights", {{"symmetric", false}}}}}}}},
      {{"config_groups", {{"group_0", {{"targets", {"Conv2d"}}}}}}},
      {{"config_groups",
        {{"group_0", {{"input_activations", {{"strategy", "tensor"}}}}}}}},
      {{"config_groups",
        {{"group_1", base.at("config_groups").at("group_0")}}}}};
  for (const auto& patch : overrides) {
    auto config = base;
    config.merge_patch(patch);
    SCOPED_TRACE(config.dump());
    JsonReader reader;
    ASSERT_TRUE(reader.parse_text(
        nlohmann::json({{"quantization_config", config}}).dump()));
    QuantArgs args;
    EXPECT_FALSE(load_quant_cfg(reader, args));
  }
}

}  // namespace xllm

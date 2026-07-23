/* Copyright 2025-2026 The xLLM Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/jd-opensource/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "hf_model_loader.h"

#include <gtest/gtest.h>

#include <filesystem>
#include <string>
#include <type_traits>

#include "core/framework/model/rec_causal_lm.h"
#include "core/platform/device.h"
#include "core/platform/platform.h"
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
}

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
  if (Platform::is_cuda()) {
    ASSERT_TRUE(load_quant_cfg(reader, quant_args));
    EXPECT_EQ(quant_args.quant_method(), kQuantMethodFp8);
    EXPECT_EQ(quant_args.bits(), 8);
    EXPECT_EQ(quant_args.moe_weight_bits(), 8);
    EXPECT_FALSE(quant_args.activation_dynamic());
    ASSERT_EQ(quant_args.ignored_modules().size(), 2);
    EXPECT_EQ(quant_args.ignored_modules()[0], "lm_head");
    EXPECT_EQ(quant_args.ignored_modules()[1], "model.layers.1.mlp.down_proj");
  }
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

#if defined(USE_NPU)
TEST(HFModelLoaderTest, Qwen35MtpModelArgsFromDenseConfig) {
  auto loader = ModelRegistry::get_model_args_loader("qwen3_5_mtp");
  ASSERT_TRUE(loader != nullptr);

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
  ASSERT_TRUE(loader != nullptr);

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

#if defined(USE_DCU)
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
    ASSERT_TRUE(load_quant_cfg(reader, quant_args));
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
#endif

}  // namespace xllm

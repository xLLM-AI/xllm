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

#include "models/llm/py_causal_lm.h"

#include <gtest/gtest.h>
#include <pybind11/eval.h>

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "core/framework/config/kernel_config.h"
#include "core/framework/model/llm_model_params.h"
#include "core/framework/model/model_args.h"
#include "core/framework/model/model_params_view.h"
#include "core/framework/model_context/model_context.h"
#include "core/framework/model_loader/hf_model_args.h"
#include "core/framework/parallel_state/parallel_args.h"
#include "core/framework/parallel_state/process_group.h"
#include "core/framework/quantization/quant_args.h"
#include "core/kv_cache/storage/kv_cache.h"
#include "core/util/scope_guard.h"
#include "core/util/utils.h"

namespace py = pybind11;

namespace xllm::detail {
namespace {

constexpr const char* kBridgeTestModelType = "py_causal_lm_bridge_test";
constexpr const char* kBridgeTestModule = "xllm_py_causal_lm_bridge_test";

void register_bridge_test_model() {
  py::module_ module_type = py::module_::import("types");
  py::object module = module_type.attr("ModuleType")(kBridgeTestModule);
  py::dict module_dict =
      py::reinterpret_borrow<py::dict>(module.attr("__dict__"));
  py::exec(R"(
class BridgeTestModel:
    def __init__(self, config):
        self.last_caches = None

    def eval(self):
        return self

    def write_context_kv(self, target_hidden, positions, cache_slots, kv_caches, layer_synchronizer):
        from xllm.python.attention.backend import normalize_layer_caches

        self.last_caches = kv_caches
        self.last_swa = normalize_layer_caches(kv_caches)[0].swa
        return target_hidden
)",
           module_dict);

  py::module_::import("sys").attr("modules")[kBridgeTestModule] = module;
  py::module_ registry = py::module_::import("xllm.python.registry");
  registry.attr("register_model")(kBridgeTestModelType)(
      module.attr("BridgeTestModel"));

  py::dict support;
  support["cpu"] = true;
  support["cuda"] = true;
  support["npu"] = true;
  py::module_::import("xllm.python.model_platform_support")
      .attr("MODEL_PLATFORM_SUPPORT")[kBridgeTestModule] = support;
}

TEST(PyCausalLMMegaMoeCapacityTest, DecodeCoversGraphAndEagerFallbackLayouts) {
  EXPECT_EQ(python_mega_moe_max_num_tokens_per_rank(
                /*max_seqs_per_batch=*/16,
                /*num_speculative_tokens=*/0,
                /*dp_size=*/1),
            16);
  EXPECT_EQ(python_mega_moe_max_num_tokens_per_rank(
                /*max_seqs_per_batch=*/16,
                /*num_speculative_tokens=*/0,
                /*dp_size=*/16),
            31);
  EXPECT_EQ(python_mega_moe_max_num_tokens_per_rank(
                /*max_seqs_per_batch=*/17,
                /*num_speculative_tokens=*/0,
                /*dp_size=*/16),
            32);
}

TEST(PyCausalLMMegaMoeCapacityTest, SpeculativeDecodeScalesRowsBySpecWidth) {
  EXPECT_EQ(python_mega_moe_max_num_tokens_per_rank(
                /*max_seqs_per_batch=*/16,
                /*num_speculative_tokens=*/3,
                /*dp_size=*/16),
            79);
  EXPECT_EQ(python_mega_moe_max_num_tokens_per_rank(
                /*max_seqs_per_batch=*/17,
                /*num_speculative_tokens=*/3,
                /*dp_size=*/16),
            128);
}

TEST(PyCausalLMMegaMoeCapacityTest,
     Qwen35CoversPrefillAndSpeculativeDecodeWithoutDpGather) {
  EXPECT_EQ(python_qwen3_5_mega_moe_max_num_tokens_per_rank(
                /*max_tokens_per_batch=*/4096,
                /*max_seqs_per_batch=*/256,
                /*num_speculative_tokens=*/3),
            4096);
  EXPECT_EQ(python_qwen3_5_mega_moe_max_num_tokens_per_rank(
                /*max_tokens_per_batch=*/1024,
                /*max_seqs_per_batch=*/512,
                /*num_speculative_tokens=*/4),
            2560);
}

// Shared scaffolding for tests that construct a PyCausalLM: registers the
// dependency-free BridgeTestModel and returns a CPU ModelContext carrying the
// given model_type (the Python side instantiates BridgeTestModel regardless —
// real model classes such as deepseek_v4 require full vocab/layer configs).
ModelContext make_bridge_model_context(const char* model_type,
                                       ModelArgs model_args = {}) {
  {
    py::gil_scoped_acquire gil;
    // Idempotent, but run the py::exec/registry writes only once across the
    // truth table's per-case calls.
    static const bool registered = [] {
      register_bridge_test_model();
      return true;
    }();
    (void)registered;
  }
  // The context only copies ParallelArgs' raw group pointers, so the group
  // must outlive every PyCausalLM built from the context. Keep it in
  // function-local static storage: the tests are CPU-only and never
  // communicate, and a stack-local would dangle after this helper returns.
  static ProcessGroup process_group(/*rank=*/0,
                                    /*world_size=*/1,
                                    torch::Device(torch::kCPU));
  ParallelArgs parallel_args(/*rank=*/0,
                             /*world_size=*/1,
                             &process_group);
  parallel_args.tp_group_ = &process_group;
  parallel_args.moe_tp_group_ = &process_group;
  parallel_args.python_rendezvous_host_ = "127.0.0.1";
  parallel_args.python_rendezvous_port_ = 1;

  model_args.model_type(model_type);
  return ModelContext(
      parallel_args,
      model_args,
      QuantArgs(),
      torch::TensorOptions().device(torch::kCPU).dtype(torch::kFloat32),
      /*context=*/nullptr);
}

TEST(PyCausalLMConfigBridgeTest, PreservesBlockDraftWindowEnablement) {
  std::string pattern =
      (std::filesystem::temp_directory_path() / "xllm_draft_window_XXXXXX")
          .string();
  const char* directory = ::mkdtemp(pattern.data());
  ASSERT_NE(directory, nullptr);
  const std::filesystem::path model_path(directory);
  ScopeGuard remove_directory([&] { std::filesystem::remove_all(model_path); });
  KernelConfig& kernel_config = KernelConfig::get_instance();
  const std::string previous_backend = kernel_config.npu_kernel_backend();
  ScopeGuard restore_backend(
      [&] { kernel_config.npu_kernel_backend(previous_backend); });
  kernel_config.npu_kernel_backend("TORCH");

  struct WindowCase {
    const char* draft_type;
    std::optional<bool> enabled;
    std::optional<int32_t> window;
    bool nested;
  };
  const WindowCase cases[] = {
      {"dflash2", std::nullopt, 8, false},
      {"dflash2", true, 8, false},
      {"dflash2", false, 8, false},
      {"dflash2", std::nullopt, 8, true},
      {"dflash2", false, 8, true},
      {"dspark", std::nullopt, 8, true},
      {"dspark", true, 8, true},
      {"dspark", false, 8, true},
      {"dspark", std::nullopt, std::nullopt, true},
      {"dspark", std::nullopt, 0, true},
      {"", std::nullopt, 8, false},
  };
  for (const WindowCase& test : cases) {
    const std::string draft_type = test.draft_type;
    nlohmann::json config = {
        {"model_type", "qwen3"},
        {"hidden_size", 4},
        {"num_hidden_layers", 1},
        {"num_attention_heads", 1},
        {"num_key_value_heads", 1},
        {"head_dim", 4},
        {"intermediate_size", 8},
        {"vocab_size", 8},
    };
    config["sliding_window"] = test.window.has_value()
                                   ? nlohmann::json(*test.window)
                                   : nlohmann::json(nullptr);
    if (test.enabled.has_value()) {
      config["use_sliding_window"] = *test.enabled;
    }
    if (draft_type == "dflash2") {
      config["dflash_config"] = {{"block_size", 4},
                                 {"conv_group_size", 2},
                                 {"conv_kernel_size", 2},
                                 {"selector_rank", 2},
                                 {"selector_top_k", 2}};
    } else if (draft_type == "dspark") {
      config["markov_rank"] = 2;
    }
    if (test.nested) {
      config = {{"speculators_model_type", draft_type},
                {"transformer_layer_config", config}};
    }
    SCOPED_TRACE(config.dump());
    {
      std::ofstream output(model_path / "config.json");
      output << config;
      ASSERT_TRUE(output.good());
    }
    ModelArgs args;
    ASSERT_TRUE(load_hf_model_args(model_path, /*backend=*/"llm", &args));
    const bool expected_enabled = test.enabled.value_or(
        !draft_type.empty() && test.window.value_or(0) > 0);
    EXPECT_EQ(args.use_sliding_window(), expected_enabled);
    if (draft_type.empty()) {
      continue;
    }
    const bool is_dflash2 = draft_type == "dflash2";
    const char* model_type =
        is_dflash2 ? "DFlash2DraftModel" : "DSparkDraftModel";
    ModelContext context =
        make_bridge_model_context(model_type, std::move(args));
    PyCausalLM model(context, kBridgeTestModelType);
    py::gil_scoped_acquire gil;
    const char* module = is_dflash2 ? "xllm.python.models.qwen3_dflash2"
                                    : "xllm.python.models.qwen3_dspark";
    const char* config_class =
        is_dflash2 ? "DFlash2Qwen3Config" : "Qwen3DSparkConfig";
    py::object python_config = py::module_::import(module)
                                   .attr(config_class)
                                   .attr("from_dict")(model.config_dict());
    const int32_t expected_window =
        expected_enabled ? test.window.value_or(0) : 0;
    EXPECT_EQ(python_config.attr("sliding_window").cast<int32_t>(),
              expected_window);
    if (is_dflash2 && expected_window == 0) {
      EXPECT_THROW(python_config.attr("validate")(), py::error_already_set);
    } else {
      python_config.attr("validate")();
    }
  }
}

TEST(PyCausalLMContextKVBridgeTest, PassesSwaCacheAsSixthPythonSlot) {
  ModelContext context = make_bridge_model_context(kBridgeTestModelType);
  PyCausalLM model(context);

  DeepSeekV4KVCacheTensors cache_tensors;
  cache_tensors.swa_cache =
      torch::arange(8, torch::kFloat32).reshape({1, 2, 1, 4});
  std::vector<KVCache> kv_caches;
  kv_caches.emplace_back(cache_tensors);

  torch::Tensor target_hidden = torch::ones({2, 4}, torch::kFloat32);
  torch::Tensor positions = torch::tensor({0, 1}, torch::kLong);
  torch::Tensor cache_slots = torch::tensor({0, 1}, torch::kLong);
  LlmModelParams params_owner;
  ModelParamsView input_params(params_owner);
  ModelOutput output = model.write_context_kv(
      target_hidden, positions, cache_slots, kv_caches, input_params);

  ASSERT_TRUE(output.hidden_states.defined());
  EXPECT_TRUE(torch::equal(output.hidden_states, target_hidden));

  py::gil_scoped_acquire gil;
  py::list python_caches = model.python_model().attr("last_caches");
  ASSERT_EQ(py::len(python_caches), 1);
  py::tuple layer_cache = python_caches[0].cast<py::tuple>();
  ASSERT_EQ(py::len(layer_cache), 12);
  py::object bridged_swa = model.python_model().attr("last_swa");
  EXPECT_TRUE(bridged_swa.is(layer_cache[5]));
  EXPECT_EQ(
      bridged_swa.attr("data_ptr")().cast<std::uintptr_t>(),
      reinterpret_cast<std::uintptr_t>(cache_tensors.swa_cache.data_ptr()));
}

// End-to-end guard for the model-type classification the Python executor
// performs: the truth table of (is_deepseek_v4_model, is_block_draft_model,
// is_dspark_draft_model) must match the naming predicates the Python side
// mirrors in model_platform_support. The cases below are sampled points, not
// the full sets: when adding a new DSpark or DFlash family member, add a case
// here AND to the Python twin test_classification_matches_cpp_truth_table in
// tests/python/test_model_executor.py — forgetting both sides leaves every
// test green while the classification drifts.
TEST(ModelTypeClassificationTest, ClassificationMatchesPredicates) {
  struct Expected {
    const char* model_type;
    bool is_deepseek_v4_model;
    bool is_block_draft_model;
    bool is_dspark_draft_model;
  };
  const Expected cases[] = {
      {"deepseek_v4", true, false, false},
      // MTP marker branch: starts_with("deepseek_v4") && contains("mtp").
      {"deepseek_v4_mtp", true, false, false},
      {"deepseek_v4_dspark", true, true, true},
      {"DFlashDraftModel", false, true, false},
      {"DFlash2DraftModel", false, true, false},
      {"DSparkDraftModel", false, true, true},
      {"glm5_dspark", false, true, true},
      {"glm_moe_dsa", false, false, false},
  };

  for (const Expected& expected : cases) {
    const std::string_view model_type = expected.model_type;
    EXPECT_EQ(util::is_deepseek_v4_model_type(model_type),
              expected.is_deepseek_v4_model)
        << "model_type=" << expected.model_type;
    EXPECT_EQ(is_block_draft_model_type(model_type),
              expected.is_block_draft_model)
        << "model_type=" << expected.model_type;
    EXPECT_EQ(is_dspark_draft_model_type(model_type),
              expected.is_dspark_draft_model)
        << "model_type=" << expected.model_type;
  }
}

}  // namespace
}  // namespace xllm::detail

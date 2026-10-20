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
#include <vector>

#include "core/framework/model/llm_model_params.h"
#include "core/framework/model/model_args.h"
#include "core/framework/model/model_input_params.h"
#include "core/framework/model_context/model_context.h"
#include "core/framework/parallel_state/parallel_args.h"
#include "core/framework/parallel_state/process_group.h"
#include "core/framework/quantization/quant_args.h"
#include "core/kv_cache/storage/kv_cache.h"

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
  registry.attr("_REGISTRY")[kBridgeTestModelType] =
      py::make_tuple(kBridgeTestModule, "BridgeTestModel");

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

TEST(PyCausalLMContextKVBridgeTest, PassesSwaCacheAsSixthPythonSlot) {
  {
    py::gil_scoped_acquire gil;
    register_bridge_test_model();
  }

  ProcessGroup process_group(/*rank=*/0,
                             /*world_size=*/1,
                             torch::Device(torch::kCPU));
  ParallelArgs parallel_args(/*rank=*/0,
                             /*world_size=*/1,
                             &process_group);
  parallel_args.tp_group_ = &process_group;
  parallel_args.moe_tp_group_ = &process_group;
  parallel_args.python_rendezvous_host_ = "127.0.0.1";
  parallel_args.python_rendezvous_port_ = 1;

  ModelArgs model_args;
  model_args.model_type(kBridgeTestModelType);
  ModelContext context(
      parallel_args,
      model_args,
      QuantArgs(),
      torch::TensorOptions().device(torch::kCPU).dtype(torch::kFloat32),
      /*context=*/nullptr);
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

}  // namespace
}  // namespace xllm::detail

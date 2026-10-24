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
#include <pybind11/embed.h>
#include <torch/python.h>

#include <memory>
#include <utility>
#include <vector>

#include "core/framework/parallel_state/process_group.h"
#include "core/layers/common/attention_metadata.h"
#include "core/platform/layer_synchronizer.h"
#include "core/runtime/py_attention_metadata.h"
#include "core/runtime/py_executor_impl.h"
#include "models/llm/py_causal_lm.h"

namespace py = pybind11;

namespace xllm {
namespace {

class TestLoadSynchronizer final : public LayerSynchronizer {
 public:
  explicit TestLoadSynchronizer(torch::Tensor ready)
      : ready_(std::move(ready)) {
    waited_.reserve(3);
  }

  bool synchronize_layer(int64_t /*layer_index*/) override {
    ADD_FAILURE() << "Python reload should enqueue a consumer stream wait.";
    return false;
  }
  bool wait_layer_on_current_stream(int64_t layer_index) override {
    waited_.emplace_back(layer_index);
    ready_.select(/*dim=*/0, layer_index).fill_(7);
    return true;
  }
  bool record_stream(int64_t /*layer_index*/, Stream* /*stream*/) override {
    return false;
  }
  void abort() override {}
  uint32_t size() const override { return 3; }

  const std::vector<int64_t>& waited() const { return waited_; }

 private:
  torch::Tensor ready_;
  std::vector<int64_t> waited_;
};

// Exercise the real C++ Python bridge with a small Python executor double.
// Neither checkpoint weights nor accelerator allocations are needed.
TEST(PyExecutorKVLoadTest, PrefillWaitsAtConsumersAndGraphKeepsCompositeFence) {
  py::gil_scoped_acquire gil;
  py::dict scope;
  scope["torch"] = py::module_::import("torch");
  const torch::Tensor ready = torch::zeros({3});
  scope["ready"] = py::cast(ready);
  py::exec(R"PY(
import sys
import types
import unittest.mock

class TestModel:
    def __init__(self, config: dict) -> None:
        pass
    def eval(self) -> "TestModel":
        return self

class TestExecutor:
    supports_prepared_metadata = True
    context_kv_graph_runner = None
    def __init__(self, *args: object) -> None:
        self.supports_layerwise_kv_load = layerwise
    def bind_kv_caches(self, caches: object) -> None:
        pass
    def execute(self, tokens: torch.Tensor, positions: torch.Tensor,
                metadata: object, *args: object, **kwargs: object) -> torch.Tensor:
        if 'kv_load_synchronizer' in kwargs:
            assert kwargs['kv_load_layers_per_event'] == 1
            load = kwargs['kv_load_synchronizer']
            assert load.size() == 3
            for group in range(2):
                assert ready[group].item() == 0
                assert load.wait_group(group)
                assert ready[group].item() == 7
                assert ready[group + 1].item() == 0
            # The target body must not wait for the draft before computing.
            assert ready[2].item() == 0
        else:
            assert bool((ready == 7).all())
        return tokens.float().view(-1, 1) + ready[0]

registry = types.ModuleType('xllm.python.registry')
registry.get_model_class = lambda name: TestModel
executor_module = types.ModuleType('xllm.python.model_executor.executor')
executor_module.ModelExecutor = TestExecutor
patch = unittest.mock.patch.dict(sys.modules, {
    'xllm.python.registry': registry,
    'xllm.python.model_executor.executor': executor_module,
})
patch.__enter__()
)PY",
           scope);
  {
    const torch::Device device(torch::kCPU);
    ProcessGroup group(/*rank=*/0, /*world_size=*/1, device);
    ParallelArgs parallel(/*rank=*/0, /*world_size=*/1, &group);
    parallel.moe_tp_group_ = &group;
    ModelArgs args;
    args.model_type("deepseek_v4")
        .dtype("float32")
        .n_heads(1)
        .n_kv_heads(1)
        .head_dim(8)
        .n_layers(1)
        .vocab_size(16);
    // This CPU executor double does not create an ATB device context.
    ModelContext context(parallel,
                         args,
                         QuantArgs(),
                         torch::TensorOptions().device(device),
                         /*context=*/nullptr);
    PyCausalLM model(context, "test_reload");
    std::vector<KVCache> caches;
    for (bool layerwise : {false, true}) {
      scope["layerwise"] = layerwise;
      PyExecutorImpl executor(&model, args, device, runtime::Options());
      for (bool graph : {false, true}) {
        ready.zero_();
        auto load = std::make_shared<TestLoadSynchronizer>(ready);
        LlmModelParams storage;
        storage.attn_metadata = std::make_shared<layer::AttentionMetadata>();
        storage.attn_metadata->is_prefill = true;
        storage.parallel.layer_wise_load_synchronizer = load;
        storage.parallel.layers_per_event = 1;
        storage.parallel.draft_load_event_index = 2;
        if (graph) {
          storage.enable_graph = true;
          storage.python_attention_metadata =
              std::make_shared<PythonAttentionMetadata>(py::none());
        }
        const ModelOutput output = executor.run(torch::tensor({3}),
                                                torch::tensor({0}),
                                                caches,
                                                ModelParamsView(storage));
        EXPECT_EQ(load->waited(), (std::vector<int64_t>{0, 1, 2}));
        EXPECT_EQ(ready[2].item<float>(), 7.0F);
        EXPECT_EQ(output.hidden_states.item<float>(), 10.0F);
      }
    }
  }
  scope["patch"].attr("__exit__")(py::none(), py::none(), py::none());
}

}  // namespace
}  // namespace xllm

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

#pragma once

#include <pybind11/pybind11.h>
#include <torch/torch.h>

#include <memory>
#include <vector>

#include "core/framework/model/causal_lm.h"
#include "core/framework/model/model_args.h"
#include "core/framework/model_context.h"
#include "core/layers/cuda/attention.h"

namespace xllm {

// Tensor-parallel process group (not owned; from ParallelArgs). Forward-declared
// so the header stays light; full definition in framework/parallel_state.
class ProcessGroup;

// CausalLM implementation whose graph lives in Python (an nn.Module from the
// xllm_models package) but whose weights, attention (paged-KV flashinfer) and
// forward orchestration are driven from C++. Selected by --model_impl=python.
//
// Ownership of the model graph is Python's; this class binds the C++ worker's
// input tensors into the Python forward and reads back hidden states.
// Marked hidden-visibility because it holds pybind11::object members, whose
// types have hidden visibility; matching the class visibility avoids
// -Werror=attributes (same pattern as processors/pywarpper_input_processor.cpp).
class __attribute__((visibility("hidden"))) PyCausalLM : public CausalLM {
 public:
  explicit PyCausalLM(const ModelContext& context);
  ~PyCausalLM() override;

  ModelOutput forward(const torch::Tensor& tokens,
                      const torch::Tensor& positions,
                      std::vector<KVCache>& kv_caches,
                      const ModelInputParams& parameters) override;

  torch::Tensor logits(const torch::Tensor& hidden_states,
                       const torch::Tensor& seleted_idxes) override;

  void load_model(std::unique_ptr<ModelLoader> loader) override;

  torch::Device device() const override { return device_; }

  const torch::TensorOptions& options() const override { return options_; }

  // Dense model: no expert weights.
  void prepare_expert_weight(int32_t /*layer_id*/,
                             const std::vector<int32_t>& /*expert_ids*/)
      override {}
  void update_expert_weight(int32_t /*layer_id*/) override {}

 private:
  // Builds the config dict handed to the Python model constructor.
  pybind11::dict build_config_dict() const;

  ModelArgs model_args_;
  torch::TensorOptions options_;
  torch::Device device_;
  bool enable_mla_ = false;

  // Tensor parallelism (read from ParallelArgs::tp_group_ in the ctor). tp_size_
  // 1 / tp_group_ null on a single card. tp_group_ is passed into the forward
  // context so the all_reduce / all_gather ops reach the right NCCL group.
  int64_t tp_size_ = 1;
  int64_t tp_rank_ = 0;
  ProcessGroup* tp_group_ = nullptr;

  // Config-only attention module shared by every layer (per-layer plan/KV is
  // keyed by layer_id in the forward context). Constructed after the worker has
  // initialized the flashinfer workspace on this thread.
  layer::Attention attn_{nullptr};

  // The Python nn.Module (Qwen3ForCausalLM) and cached constructors. Held as
  // pybind objects; all access is under the GIL.
  pybind11::object py_model_;
  pybind11::object forward_batch_cls_;
};

}  // namespace xllm

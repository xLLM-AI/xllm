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

#include <glog/logging.h>
#include <pybind11/pybind11.h>
#include <torch/torch.h>

#include <memory>
#include <vector>

#include "core/framework/model/causal_lm.h"
#include "core/framework/model/causal_vlm.h"
#include "core/framework/model/model_args.h"
#include "core/framework/model_context.h"

namespace xllm {

class ProcessGroup;

namespace detail {
void share_python_model_weights(pybind11::object& draft_model,
                                const pybind11::object& target_model);
}  // namespace detail

// Inherits CausalVLM so that ``--backend vlm --model_impl python`` can route a
// Python VLM (e.g. Qwen3-VL) through the VLM engine while PyExecutorImpl drives
// it. The CausalVLM ``encode`` / ``get_input_embeddings`` virtuals are stubbed:
// in the Python path, PyExecutorImpl::run calls the Python model's own
// encode/get_input_embeddings via pybind (they are never reached here).
class __attribute__((visibility("hidden"))) PyCausalLM : public CausalVLM {
 public:
  explicit PyCausalLM(const ModelContext& context);
  ~PyCausalLM() override;

  ModelOutput forward(const torch::Tensor& tokens,
                      const torch::Tensor& positions,
                      std::vector<KVCache>& kv_caches,
                      const ModelInputParams& parameters) override;

  MMDict encode(const ModelInputParams& parameters) override {
    LOG(FATAL) << "PyCausalLM::encode must not be called directly; "
               << "PyExecutorImpl drives the Python model's encode via pybind.";
    return MMDict{};
  }

  torch::Tensor get_input_embeddings(
      const torch::Tensor& input_ids,
      const ModelInputParams& input_params) override {
    LOG(FATAL) << "PyCausalLM::get_input_embeddings must not be called "
               << "directly; PyExecutorImpl drives it via pybind.";
    return torch::Tensor();
  }

  torch::Tensor logits(const torch::Tensor& hidden_states,
                       const torch::Tensor& seleted_idxes) override;

  void load_model(std::unique_ptr<ModelLoader> loader) override;

  torch::Device device() const override { return device_; }
  const torch::TensorOptions& options() const override { return options_; }

  void prepare_expert_weight(int32_t, const std::vector<int32_t>&) override {}
  void update_expert_weight(int32_t) override {}

  bool share_weights_from(CausalLM& source) override;

  pybind11::object& python_model() { return py_model_; }
  const pybind11::object& config_dict() const { return config_dict_; }

 private:
  pybind11::dict build_config_dict(const ParallelArgs& parallel_args) const;

  ModelArgs model_args_;
  torch::TensorOptions options_;
  torch::Device device_;
  bool enable_mla_ = false;

  int64_t tp_size_ = 1;
  int64_t tp_rank_ = 0;
  int64_t dp_size_ = 1;
  int64_t dp_rank_ = 0;
  int64_t moe_tp_size_ = 1;
  int64_t moe_tp_rank_ = 0;
  int64_t ep_size_ = 1;
  int64_t ep_rank_ = 0;
  ProcessGroup* tp_group_ = nullptr;

  pybind11::object py_model_;
  pybind11::object config_dict_;
};

}  // namespace xllm

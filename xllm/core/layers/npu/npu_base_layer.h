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

#include <torch/torch.h>
#include <torch_npu/torch_npu.h>

#include <atomic>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "atb/atb_infer.h"
#include "atb_speed/base/model.h"
#include "atb_speed/log.h"
#include "atb_speed/utils/singleton.h"
#include "atb_speed/utils/tensor_util.h"
#include "buffer/atb_workspace.h"
#include "framework/kv_cache/kv_cache.h"
#include "framework/model/model_input_params.h"
#include "framework/model_context.h"
#include "framework/state_dict/state_dict.h"
#include "framework/xtensor/xtensor.h"
#include "loader/base_loader.h"
#include "platform/device.h"
#include "pytorch/adapter/utils/utils.h"
#include "pytorch/adapter/workspace/workspace.h"

#ifdef TORCH_HIGHER_THAN_PTA6
#include <torch_npu/csrc/core/npu/NPUFormat.h>
#include <torch_npu/csrc/framework/OpCommand.h>
#else
#include <torch_npu/csrc/aten/NPUNativeFunctions.h>
#include <torch_npu/csrc/framework/utils/OpPreparation.h>
#endif

namespace xllm {
namespace layer {

enum class TransposeType : int {
  INVALID = -1,
  NOT_TRANSPOSE = 0,
  TRANSPOSE = 1
};

enum class LinearType : int { INVALID = -1, FP = 0, INT = 1 };

enum class PackType : int {
  PACK_QUANT_UNDEFINED = 0,
  ALL_FP = 1,
  ALL_W8A8 = 2,
  ALL_W8A8_ANTI = 3,
  MIX_W8A8 = 4,
  MIX_W8A8_ANTI = 5,
  ALL_W8A16 = 6,
  ALL_W8A8SC = 7,
  MIX_W8A8SC = 8,
  ALL_W8A8SC_ANTI = 9,
  MIX_W8A8SC_ANTI = 10,
  ALL_W4A16 = 11,
  ALL_W8A16_ANTI = 12,
  ALL_W4A16_ANTI = 13,
  MIX_W4A16 = 14,
  MIX_W4A16_ANTI = 15,
  MIX_W8A16 = 16,
  MIX_W8A16_ANTI = 17,
  ALL_W8A8_DYNAMIC = 18,
  ALL_W8A8_DYNAMIC_ANTI = 19,
  MIX_W8A8_DYNAMIC = 20,
  MIX_W8A8_DYNAMIC_ANTI = 21,
  ALL_W4A8 = 22,
  MIX_W4A8 = 23,
  ALL_W4A8_ANTI = 24,
  MIX_W4A8_ANTI = 25
};

enum class LinearTypeV2 : int {
  INVALID = -1,
  FLOAT16 = 0,
  BFLOAT16 = 1,
  W4A16 = 2,
  W8A16 = 3,
  W8A8 = 4,
  W8A8S = 5,
  W8A8SC = 6,
  W8A8_DYNAMIC = 7,
  W8A8_PDMIX = 8,
  W4A8_DYNAMIC = 9
};

class BaseLayer : public torch::nn::Module {
 public:
  explicit BaseLayer(const ModelContext& context);
  virtual ~BaseLayer() override = default;

  atb::Status execute_node(atb_speed::Model::Node& node,
                           int nodeId = 0,
                           aclrtEvent* event = nullptr,
                           std::atomic<bool>* event_flag = nullptr);

  atb::Status execute_plan(const atb_speed::Model::Node& node,
                           const std::string& op_name,
                           aclrtEvent* event,
                           std::atomic<bool>* event_flag);

  virtual void load_state_dict(const StateDict& state_dict) {
    if (loader_) {
      loader_->load_state_dict(state_dict);
    }
  };

  virtual void verify_loaded_weights() const {
    if (loader_) {
      loader_->verify_loaded_weights();
    }
  };

  virtual void verify_loaded_weights(const std::string& prefix) const {
    if (loader_) {
      loader_->verify_loaded_weights(prefix);
    }
  };

  virtual void merge_loaded_weights() {
    if (loader_) {
      loader_->merge_loaded_weights();
      auto& at_weight_tensors = loader_->get_at_weight_tensors();
      for (int i = 0; i < atb_weight_tensors_.size(); i++) {
        atb_weight_tensors_[i] =
            atb_speed::Utils::AtTensor2Tensor(at_weight_tensors[i]);
      }
      Device::empty_cache(device_.index());
      init_layer();
    }
  };

  virtual void free_weights() {
    if (loader_) {
      loader_->free_weights();
    }
  };

  virtual void reload_weights() {
    if (loader_) {
      loader_->reload_weights();
      auto& at_weight_tensors = loader_->get_at_weight_tensors();
      for (int i = 0; i < atb_weight_tensors_.size(); i++) {
        atb_weight_tensors_[i] =
            atb_speed::Utils::AtTensor2Tensor(at_weight_tensors[i]);
      }
    }
  };

  virtual void reload_weights_from_device() {
    if (loader_) {
      loader_->reload_weights_from_device();
      auto& at_weight_tensors = loader_->get_at_weight_tensors();
      for (int i = 0; i < atb_weight_tensors_.size(); i++) {
        atb_weight_tensors_[i] =
            atb_speed::Utils::AtTensor2Tensor(at_weight_tensors[i]);
      }
    }
  };

  virtual void merge_and_move_pinned_host() {
    if (loader_) {
      loader_->merge_and_move_pinned_host();
      init_layer();
    }
  };

  virtual void refresh_rolling_weights() {
    if (loader_) {
      loader_->refresh_rolling_weights();
      auto& at_weight_tensors = loader_->get_at_weight_tensors();
      for (int i = 0; i < atb_weight_tensors_.size(); i++) {
        atb_weight_tensors_[i] =
            atb_speed::Utils::AtTensor2Tensor(at_weight_tensors[i]);
      }
    }
  };

  // Returns the loader when it is running in manual mode, otherwise nullptr
  // (e.g., when enable_manual_loader=false). All manual-mode bookkeeping
  // (pinned host buffer, rolling buffer, slice metadata, ...) lives directly
  // on `BaseLoader`.
  virtual BaseLoader* get_loader() {
    return (loader_ && loader_->mode() == LoadMode::kManual) ? loader_.get()
                                                             : nullptr;
  }

  virtual int64_t init_layer() { return 0; };

  virtual void run_task(std::string taskName, std::function<int()> task) const;

  virtual void set_residual(torch::Tensor residual) { residual_ = residual; };

  virtual void set_layer_id(int32_t layer_id) {};

 protected:
  std::unique_ptr<BaseLoader> loader_ = nullptr;
  std::vector<at::Tensor> at_weight_tensors_;
  at::Device device_;
  std::string name_;
  torch::ScalarType dtype_;
  std::vector<int32_t> placeholder_vec_;
  std::vector<int32_t> placeholder_vec_zero_;
  torch::Tensor residual_;
  xllm::ParallelArgs parallel_args_;
  std::function<void(const std::string&, std::function<int()>)> run_task_func_;
  std::string quantize_type_;
  std::string torch_dtype_;
  int32_t dp_size_;
  int32_t dp_local_tp_size_;
  int32_t dp_rank_;
  int32_t dp_local_tp_rank_;
  int32_t cp_size_;
  atb::Context* context_;
  std::shared_ptr<AtbWorkspace> work_space_ = nullptr;
  std::vector<atb::Tensor> atb_weight_tensors_;
  bool graph_captured_{false};
};

}  // namespace layer
}  // namespace xllm

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

#include "core/framework/eplb/eplb_controller.h"

#include <glog/logging.h>
#include <torch/torch.h>

#include <deque>
#include <limits>
#include <optional>
#include <utility>
#include <vector>

#include "core/framework/config/eplb_config.h"
#include "core/framework/eplb/eplb_manager.h"
#include "core/framework/eplb/eplb_options.h"
#include "core/framework/eplb/eplb_utils.h"
#include "core/framework/model/model_args.h"

namespace xllm {

class EplbController::Impl final {
 public:
  Impl(int32_t num_layers,
       int32_t num_experts,
       int32_t worker_num,
       int32_t device_num,
       EplbOptions options)
      : num_layers_(num_layers),
        worker_num_(worker_num),
        device_experts_num_(
            eplb::local_physical_experts_num(num_experts,
                                             device_num,
                                             options.redundant_experts_num)),
        manager_(std::make_unique<EplbManager>(num_layers,
                                               device_num,
                                               num_experts,
                                               std::move(options),
                                               nullptr)) {}

  void annotate_inputs(std::vector<LlmForwardInput>& inputs,
                       const std::vector<int32_t>& dp_token_counts,
                       bool allow_eplb_command) {
    CHECK(!inputs.empty()) << "EPLB requires at least one input.";
    CHECK_EQ(inputs.size(), dp_token_counts.size())
        << "EPLB inputs must align with DP token counts.";

    const EplbInfo eplb_info = manager_->get_eplb_info(allow_eplb_command);
    std::vector<torch::Tensor> decode_masks;
    decode_masks.reserve(inputs.size());
    for (size_t dp_rank = 0; dp_rank < inputs.size(); ++dp_rank) {
      const torch::Tensor& local_mask =
          inputs[dp_rank].input_params.expert.eplb_decode_token_mask;
      if (!local_mask.defined()) {
        CHECK_EQ(inputs[dp_rank].host_token_ids().numel(), 0)
            << "EPLB requires a per-token decode mask.";
        decode_masks.emplace_back(torch::empty({0}, torch::kBool));
      } else {
        decode_masks.emplace_back(local_mask);
      }
    }
    const torch::Tensor global_decode_mask =
        eplb::build_global_decode_token_mask(decode_masks, dp_token_counts);
    for (LlmForwardInput& input : inputs) {
      input.input_params.expert.eplb_info = eplb_info;
      input.input_params.expert.eplb_decode_token_mask = global_decode_mask;
    }
  }

  void on_step_dispatched(const std::vector<LlmForwardInput>& inputs,
                          bool is_graph_warmup) {
    if (is_graph_warmup) {
      return;
    }
    CHECK(!inputs.empty()) << "EPLB requires at least one dispatched input.";
    const int64_t activation_token =
        inputs.front().input_params.expert.eplb_info.activation_token;
    for (const LlmForwardInput& input : inputs) {
      CHECK_EQ(input.input_params.expert.eplb_info.activation_token,
               activation_token)
          << "EPLB activation token must be identical across DP inputs.";
    }
    pending_activation_tokens_.push_back(activation_token);
  }

  void on_step_completed(
      const std::vector<folly::Try<std::optional<RawForwardOutput>>>& results,
      bool is_graph_warmup) {
    std::vector<WorkerResultView> views;
    views.reserve(results.size());
    for (size_t worker_rank = 0; worker_rank < results.size(); ++worker_rank) {
      const auto& result = results[worker_rank];
      CHECK(result.hasValue() && result.value().has_value())
          << "Missing EPLB result from worker " << worker_rank;
      const RawForwardOutput& output = result.value().value();
      views.push_back({&output.expert_load_data, output.prepared_token});
    }
    submit_worker_results(views, take_completion_token(is_graph_warmup));
  }

 private:
  struct WorkerResultView final {
    const std::vector<int64_t>* expert_load_data;
    int64_t prepared_token;
  };

  int64_t take_completion_token(bool is_graph_warmup) {
    if (is_graph_warmup) {
      return -1;
    }
    CHECK(!pending_activation_tokens_.empty())
        << "Missing EPLB activation metadata for completed step.";
    const int64_t completed_activation_token =
        pending_activation_tokens_.front();
    pending_activation_tokens_.pop_front();
    return completed_activation_token;
  }

  void submit_worker_results(const std::vector<WorkerResultView>& results,
                             int64_t completed_activation_token) {
    CHECK_EQ(results.size(), static_cast<size_t>(worker_num_))
        << "EPLB requires forward results from all workers.";

    std::vector<torch::Tensor> expert_loads;
    std::vector<int64_t> prepare_tokens(results.size(), -1);
    expert_loads.reserve(results.size());
    for (int32_t worker_rank = 0; worker_rank < worker_num_; ++worker_rank) {
      const WorkerResultView& result =
          results[static_cast<size_t>(worker_rank)];
      const size_t expected_size = static_cast<size_t>(num_layers_) *
                                   static_cast<size_t>(device_experts_num_);
      CHECK(result.expert_load_data != nullptr);
      CHECK_EQ(result.expert_load_data->size(), expected_size)
          << "EPLB expert_load_data size mismatch from worker " << worker_rank;
      expert_loads.emplace_back(
          torch::from_blob(
              const_cast<int64_t*>(result.expert_load_data->data()),
              {num_layers_, device_experts_num_},
              torch::TensorOptions().dtype(torch::kInt64))
              .clone());
      prepare_tokens[static_cast<size_t>(worker_rank)] = result.prepared_token;
    }

    manager_->set_prepared_tokens(prepare_tokens);
    manager_->update_expert_load(expert_loads, completed_activation_token);
  }

  const int32_t num_layers_;
  const int32_t worker_num_;
  const int32_t device_experts_num_;
  std::unique_ptr<EplbManager> manager_;
  std::deque<int64_t> pending_activation_tokens_;
};

std::unique_ptr<EplbController> EplbController::create(
    const ModelArgs& model_args,
    int32_t worker_num,
    int32_t ep_size) {
  if (!EPLBConfig::get_instance().enable_eplb()) {
    return nullptr;
  }
  const int64_t num_layers =
      model_args.n_layers() - model_args.first_k_dense_replace();
  const int32_t num_experts = model_args.n_routed_experts();
  CHECK_GT(num_layers, 0) << "EPLB num_layers must be positive.";
  CHECK_LE(num_layers, std::numeric_limits<int32_t>::max())
      << "EPLB num_layers exceeds int32_t range.";
  CHECK_GT(num_experts, 0) << "EPLB num_experts must be positive.";
  CHECK_GT(worker_num, 0) << "EPLB worker_num must be positive.";
  const int32_t device_num = eplb::effective_device_num(worker_num, ep_size);
  EplbOptions options = EplbOptions::from_global_config();
  return std::unique_ptr<EplbController>(
      new EplbController(static_cast<int32_t>(num_layers),
                         num_experts,
                         worker_num,
                         device_num,
                         std::move(options)));
}

EplbController::EplbController(int32_t num_layers,
                               int32_t num_experts,
                               int32_t worker_num,
                               int32_t device_num,
                               EplbOptions options)
    : impl_(std::make_unique<Impl>(num_layers,
                                   num_experts,
                                   worker_num,
                                   device_num,
                                   std::move(options))) {}

EplbController::~EplbController() = default;

void EplbController::annotate_inputs(
    std::vector<LlmForwardInput>& inputs,
    const std::vector<int32_t>& dp_token_counts,
    bool allow_eplb_command) {
  impl_->annotate_inputs(inputs, dp_token_counts, allow_eplb_command);
}

void EplbController::on_step_dispatched(
    const std::vector<LlmForwardInput>& inputs,
    bool is_graph_warmup) {
  impl_->on_step_dispatched(inputs, is_graph_warmup);
}

void EplbController::on_step_completed(
    const std::vector<folly::Try<std::optional<RawForwardOutput>>>& results,
    bool is_graph_warmup) {
  impl_->on_step_completed(results, is_graph_warmup);
}

}  // namespace xllm

/* Copyright 2025 The xLLM Authors. All Rights Reserved.

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

#include "eagle3_worker_impl.h"

#include <glog/logging.h>

#include "common/global_flags.h"
#include "core/framework/config/speculative_config.h"
#include "framework/model_loader.h"

namespace xllm {

namespace {

runtime::Options eagle3_main_options(const runtime::Options& options) {
  auto opts = options;
  opts.enable_schedule_overlap(false)
      .is_draft_engine(false)
      .enable_graph_aux_hidden_states(true);
  return opts;
}

runtime::Options eagle3_draft_options(const runtime::Options& options) {
  auto opts = options;
  opts.enable_schedule_overlap(false)
      .is_draft_engine(true)
      .num_decoding_tokens(1)
      .num_speculative_tokens(0)
      .enable_graph_aux_hidden_states(false);
  return opts;
}

}  // namespace

Eagle3WorkerImpl::Eagle3WorkerImpl(const ParallelArgs& parallel_args,
                                   const torch::Device& device,
                                   const runtime::Options& options)
    : MTPWorkerImpl(parallel_args,
                    device,
                    options,
                    eagle3_main_options(options),
                    eagle3_draft_options(options),
                    ::xllm::SpeculativeConfig::get_instance()
                        .enable_opt_validate_probs()) {}

bool Eagle3WorkerImpl::init_model(const std::string& model_weights_path,
                                  int32_t random_seed,
                                  MasterStatus master_status) {
  // Call parent's init_model first
  bool result =
      MTPWorkerImpl::init_model(model_weights_path, random_seed, master_status);

  // Load hot_token_id_ directly from state_dict (EAGLE-3 specific)
  // This should be done after draft model is loaded
  if (draft_impl_->get_status() == WorkerImpl::Status::LOADED) {
    // d2t stores diffs between draft id and target id
    // hot_token_id = d2t + arange(d2t.size(0))
    auto model_loader = ModelLoader::create(model_weights_path);
    auto& state_dicts = model_loader->get_state_dicts();
    for (const auto& state_dict : state_dicts) {
      torch::Tensor d2t_tensor = state_dict->get_tensor("d2t");
      if (d2t_tensor.defined()) {
        auto arange_tensor = torch::arange(d2t_tensor.size(0));
        hot_token_id_ = d2t_tensor + arange_tensor;
        hot_token_id_ = hot_token_id_.to(device_);
        LOG(INFO) << "Eagle3WorkerImpl: Loaded d2t tensor from state_dict, "
                     "hot_token_id size: "
                  << hot_token_id_.size(0);
        break;
      }
    }
  }

  return result;
}

int64_t Eagle3WorkerImpl::get_embedding_placeholder_size() {
  const int64_t target_hidden = context_.get_model_args().hidden_size();
  return 3 * target_hidden;
}

void Eagle3WorkerImpl::process_draft_sample_output(
    SampleOutput& sample_output) {
  // Keep probability compression behavior fully aligned with MTP.
  MTPWorkerImpl::process_draft_sample_output(sample_output);

  // EAGLE-3 specific: map draft token IDs to target token IDs.
  if (!hot_token_id_.defined() || !sample_output.next_tokens.defined() ||
      sample_output.next_tokens.numel() == 0) {
    return;
  }

  sample_output.next_tokens =
      hot_token_id_.index_select(0, sample_output.next_tokens);
}

}  // namespace xllm

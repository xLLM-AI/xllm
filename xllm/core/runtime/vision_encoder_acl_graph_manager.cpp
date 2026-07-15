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

#include "core/runtime/vision_encoder_acl_graph_manager.h"

#include <glog/logging.h>

#include <stdexcept>

namespace xllm::npu {

namespace {

// Hard cap: graph only helps at small token counts; the captured encoder is
// slower than eager past ~512, so larger images fall back to eager.
constexpr int64_t kEncoderGraphMaxReplayTokens = 512;

std::vector<int64_t> filter_budgets(const std::vector<int64_t>& budgets) {
  std::vector<int64_t> valid;
  valid.reserve(budgets.size());
  for (int64_t b : budgets) {
    if (b > kEncoderGraphMaxReplayTokens) {
      LOG(WARNING) << "[EncoderGraph] dropping budget " << b << " > cap "
                   << kEncoderGraphMaxReplayTokens
                   << " (graph only helps at small token counts; larger "
                   << "images fall back to eager)";
    } else {
      valid.push_back(b);
    }
  }
  if (!budgets.empty() && valid.empty()) {
    LOG(WARNING) << "[EncoderGraph] no budget <= "
                 << kEncoderGraphMaxReplayTokens
                 << "; encoder graph effectively disabled";
  }
  return valid;
}

}  // namespace

VisionEncoderAclGraphManager::VisionEncoderAclGraphManager(
    const ModelArgs& args,
    const torch::Device& device,
    const std::vector<int64_t>& budgets)
    : budgets_(filter_budgets(budgets)),
      device_(device),
      hidden_size_(args.mm_hidden_size()),
      head_dim_(args.mm_head_dim()),
      d_model_(args.mm_projection_dim()),
      spatial_merge_size_(args.mm_spatial_merge_size()),
      device_index_(device.index()),
      dtype_(args.dtype() == "bfloat16" ? torch::kBFloat16 : torch::kFloat16) {
  max_budget_ = budgets_.empty() ? 0 : budgets_.back();
  capture_stream_.emplace(
      c10_npu::getStreamFromPool(/*isHighPriority=*/true, device_index_));
  LOG(INFO) << "VisionEncoderAclGraphManager created with " << budgets_.size()
            << " budgets, max_budget=" << max_budget_
            << ", hidden_size=" << hidden_size_ << ", head_dim=" << head_dim_
            << ", d_model=" << d_model_;
}

void VisionEncoderAclGraphManager::initialize_persistent_param(
    EncoderPersistentParam& param,
    int64_t budget,
    int32_t num_deepstack,
    size_t num_segments) {
  auto opts = torch::TensorOptions().device(device_).dtype(dtype_);
  auto int_opts = torch::TensorOptions().device(device_).dtype(torch::kInt32);

  param.hidden_states = torch::zeros({budget, hidden_size_}, opts);
  param.cos_pos = torch::zeros({budget, head_dim_}, opts);
  param.sin_pos = torch::zeros({budget, head_dim_}, opts);
  param.cu_seqlens =
      torch::zeros({static_cast<int64_t>(num_segments)}, int_opts);
  param.cu_seqlens_vec.clear();
  param.cu_seqlens_vec.reserve(num_segments);

  param.output_hidden = torch::zeros({budget, hidden_size_}, opts);
  int64_t merge_sq = spatial_merge_size_ * spatial_merge_size_;
  int64_t ds_tokens = budget / merge_sq;
  param.deepstack_outs.reserve(num_deepstack);
  for (int32_t i = 0; i < num_deepstack; ++i) {
    param.deepstack_outs.emplace_back(
        torch::zeros({ds_tokens, d_model_}, opts));
  }
}

int64_t VisionEncoderAclGraphManager::select_budget(
    int64_t actual_tokens) const {
  for (auto b : budgets_) {
    if (b >= actual_tokens) {
      return b;
    }
  }
  return -1;
}

bool VisionEncoderAclGraphManager::can_replay(int64_t actual_tokens) const {
  return select_budget(actual_tokens) > 0;
}

void VisionEncoderAclGraphManager::copy_inputs_to_persistent(
    EncoderPersistentParam& param,
    const torch::Tensor& hidden_states,
    const torch::Tensor& cos_pos,
    const torch::Tensor& sin_pos,
    const torch::Tensor& /*cu_seqlens*/,
    const std::vector<int>& cu_seqlens_vec,
    int64_t actual_tokens,
    int64_t bucket_size) {
  // Zero the padding tail: the last segment is extended to bucket_size below,
  // so padding rows join real attention and stale input would leak K/V.
  if (actual_tokens < bucket_size) {
    param.hidden_states.slice(0, actual_tokens, bucket_size).zero_();
    param.cos_pos.slice(0, actual_tokens, bucket_size).zero_();
    param.sin_pos.slice(0, actual_tokens, bucket_size).zero_();
  }
  param.hidden_states.slice(0, 0, actual_tokens).copy_(hidden_states);
  param.cos_pos.slice(0, 0, actual_tokens).copy_(cos_pos);
  param.sin_pos.slice(0, 0, actual_tokens).copy_(sin_pos);

  param.cu_seqlens_vec = cu_seqlens_vec;
  if (!param.cu_seqlens_vec.empty()) {
    param.cu_seqlens_vec.back() = static_cast<int>(bucket_size);
  }
  param.cu_seqlens.copy_(torch::tensor(
      param.cu_seqlens_vec, torch::TensorOptions().dtype(torch::kInt32)));
}

bool VisionEncoderAclGraphManager::capture(VisionEncoderGraphAdapter* adapter,
                                           torch::Tensor& hidden_states,
                                           torch::Tensor& cos_pos,
                                           torch::Tensor& sin_pos,
                                           torch::Tensor& cu_seqlens,
                                           std::vector<int>& cu_seqlens_vec,
                                           int64_t budget) {
  CHECK(adapter != nullptr) << "encoder graph adapter must be set";
  const auto& deepstack_indexes = adapter->deepstack_indexes();
  const size_t num_segments = cu_seqlens_vec.size();

  LOG(INFO) << "Capturing encoder graph, budget=" << budget
            << ", actual_tokens=" << hidden_states.size(0)
            << ", num_segments=" << num_segments;

  auto bucket_graph = std::make_unique<BucketGraph>();
  bucket_graph->num_tokens = budget;
  bucket_graph->num_segments = num_segments;

  initialize_persistent_param(bucket_graph->param,
                              budget,
                              static_cast<int32_t>(deepstack_indexes.size()),
                              num_segments);

  int64_t actual_tokens = hidden_states.size(0);
  copy_inputs_to_persistent(bucket_graph->param,
                            hidden_states,
                            cos_pos,
                            sin_pos,
                            cu_seqlens,
                            cu_seqlens_vec,
                            actual_tokens,
                            budget);

  auto& param = bucket_graph->param;

  torch::npu::synchronize();

  {
    LOG(INFO) << "[EncoderGraph] Warm-up run with budget=" << budget;
    torch::Tensor warm_hidden = param.hidden_states.clone();
    for (int32_t i = 0; i < adapter->num_encoder_layers(); ++i) {
      warm_hidden = adapter->forward_encoder_layer(
          /*layer_idx=*/i,
          warm_hidden,
          param.cos_pos,
          param.sin_pos,
          param.cu_seqlens,
          param.cu_seqlens_vec);
    }
    torch::npu::synchronize();
    LOG(INFO) << "[EncoderGraph] Warm-up completed successfully";
  }

  param.hidden_states.zero_();
  param.hidden_states.slice(0, 0, actual_tokens).copy_(hidden_states);

  bool need_restore = false;
  aclrtStream capture_raw_stream = nullptr;
  if (c10_npu::getCurrentNPUStream(device_index_) ==
      c10_npu::getDefaultNPUStream(device_index_)) {
    c10_npu::setCurrentNPUStream(*capture_stream_);
    capture_raw_stream = capture_stream_->stream();
    aclrtSynchronizeStream(capture_raw_stream);
    need_restore = true;
  } else {
    capture_raw_stream = c10_npu::getCurrentNPUStream(device_index_).stream();
  }

  bool captured = false;
  try {
    bucket_graph->graph.capture_begin(
        {0, 0}, aclmdlRICaptureMode::ACL_MODEL_RI_CAPTURE_MODE_THREAD_LOCAL);

    torch::Tensor hidden = param.hidden_states;
    for (int32_t i = 0; i < adapter->num_encoder_layers(); ++i) {
      hidden = adapter->forward_encoder_layer(
          /*layer_idx=*/i,
          hidden,
          param.cos_pos,
          param.sin_pos,
          param.cu_seqlens,
          param.cu_seqlens_vec);

      for (size_t k = 0; k < deepstack_indexes.size(); ++k) {
        if (deepstack_indexes[k] == i) {
          auto ds_out = adapter->forward_deepstack_merger(
              static_cast<int32_t>(k), hidden);
          param.deepstack_outs[k].copy_(ds_out);
        }
      }
    }
    param.output_hidden.copy_(hidden);

    bucket_graph->graph.capture_end();
    captured = true;
  } catch (const std::exception& e) {
    LOG(ERROR) << "[EncoderGraph] capture threw for budget=" << budget << ": "
               << e.what() << "; falling back to eager";
    captured = false;
  }

  if (need_restore) {
    c10_npu::setCurrentNPUStream(c10_npu::getDefaultNPUStream(device_index_));
  }

  if (!captured) {
    return false;
  }

  aclrtSynchronizeStream(capture_raw_stream);
  bucket_graph->graph.replay();

  graphs_[budget] = std::move(bucket_graph);
  LOG(INFO) << "Encoder graph captured successfully for budget=" << budget;
  return true;
}

std::optional<EncoderGraphOutput> VisionEncoderAclGraphManager::replay(
    torch::Tensor& hidden_states,
    torch::Tensor& cos_pos,
    torch::Tensor& sin_pos,
    torch::Tensor& cu_seqlens,
    std::vector<int>& cu_seqlens_vec,
    int64_t actual_num_tokens) {
  int64_t budget = select_budget(actual_num_tokens);
  if (budget <= 0) {
    return std::nullopt;
  }

  auto it = graphs_.find(budget);
  if (it == graphs_.end()) {
    CHECK(adapter_ != nullptr) << "Encoder adapter not set for lazy capture";
    if (!capture(adapter_,
                 hidden_states,
                 cos_pos,
                 sin_pos,
                 cu_seqlens,
                 cu_seqlens_vec,
                 budget)) {
      return std::nullopt;
    }
    it = graphs_.find(budget);
  }

  auto& bucket = it->second;
  if (bucket->num_segments != cu_seqlens_vec.size()) {
    LOG_FIRST_N(WARNING, 1)
        << "[EncoderGraph] segment count mismatch (captured="
        << bucket->num_segments << ", request=" << cu_seqlens_vec.size()
        << ") for budget=" << budget << "; falling back to eager. "
        << "This message is logged only once.";
    return std::nullopt;
  }

  copy_inputs_to_persistent(bucket->param,
                            hidden_states,
                            cos_pos,
                            sin_pos,
                            cu_seqlens,
                            cu_seqlens_vec,
                            actual_num_tokens,
                            budget);

  bucket->graph.replay();

  EncoderGraphOutput output;
  output.hidden_states =
      bucket->param.output_hidden.slice(0, 0, actual_num_tokens);

  int64_t merge_sq = spatial_merge_size_ * spatial_merge_size_;
  int64_t ds_actual = actual_num_tokens / merge_sq;
  const auto& deepstack_indexes = adapter_->deepstack_indexes();
  output.deepstack_features.reserve(deepstack_indexes.size());
  for (size_t k = 0; k < deepstack_indexes.size(); ++k) {
    output.deepstack_features.push_back(
        bucket->param.deepstack_outs[k].slice(0, 0, ds_actual));
  }

  return output;
}

}  // namespace xllm::npu

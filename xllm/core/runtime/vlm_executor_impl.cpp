/* Copyright 2025-2026 The xLLM Authors.

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

#include "core/runtime/vlm_executor_impl.h"

#include <glog/logging.h>
#include <torch/torch.h>

#include "core/framework/multimodal/mm_visitor.h"
#include "platform/device.h"
#include "platform/platform.h"

namespace xllm {

VlmExecutorImpl::VlmExecutorImpl(CausalLM* model,
                                 const ModelArgs& args,
                                 const torch::Device& device,
                                 const runtime::Options& options)
    : model_(dynamic_cast<CausalVLM*>(model)),
      device_(device),
      options_(options),
      use_mrope_(args.rope_scaling_rope_type() == "mrope") {
  if (const auto cache_size_mb = options.max_encoder_cache_size();
      cache_size_mb > 0) {
    encoder_cache_ =
        std::make_unique<EncoderCache>(cache_size_mb * 1024 * 1024);
  }

  if (options_.enable_graph() || options_.enable_prefill_piecewise_graph()) {
    llm_executor_ = ExecutorImplFactory::get_instance().create_executor_impl(
        model, args, device, options_, Platform::type_str());
  }
}

void VlmExecutorImpl::clear_multimodal_cache() {
  if (encoder_cache_ != nullptr) {
    encoder_cache_->clear();
  }
  if (llm_executor_ != nullptr) {
    llm_executor_->clear_multimodal_cache();
  }
}

ModelOutput VlmExecutorImpl::run(const torch::Tensor& tokens,
                                 const torch::Tensor& positions,
                                 std::vector<KVCache>& kv_caches,
                                 const ModelParamsView& params) {
  torch::NoGradGuard no_grad;
  return run_backbone(tokens, positions, kv_caches, params);
}

ModelOutput VlmExecutorImpl::run_backbone(const torch::Tensor& tokens,
                                          const torch::Tensor& positions,
                                          std::vector<KVCache>& kv_caches,
                                          const ModelParamsView& params) {
  if (llm_executor_ && (options_.enable_graph() ||
                        params.meta.batch_forward_type.is_prefill())) {
    return llm_executor_->run(tokens, positions, kv_caches, params);
  }
  return model_->forward(tokens, positions, kv_caches, params);
}

ModelOutput VlmExecutorImpl::run_multimodal(const torch::Tensor& tokens,
                                            const torch::Tensor& positions,
                                            std::vector<KVCache>& kv_caches,
                                            const ModelParamsView& params,
                                            MMBatchData& mm_data) {
  torch::NoGradGuard no_grad;
  // Decode uses identical mRoPE rows and lets the graph embed persistent
  // tokens.
  if (params.meta.batch_forward_type.is_decode() &&
      params.meta.q_max_seq_len <= 1 && !mm_data.valid()) {
    return run_backbone(
        tokens, collapse_mrope_decode_positions(positions), kv_caches, params);
  }

  params.embedding.input_embedding =
      prepare_inputs_embeds(tokens, params, mm_data);
  return run_backbone(tokens, positions, kv_caches, params);
}

torch::Tensor VlmExecutorImpl::prepare_inputs_embeds(
    const torch::Tensor& tokens,
    const ModelParamsView& params,
    MMBatchData& mm_data) {
  execute_mm_encoder(
      mm_data, encoder_cache_.get(), [this](const MMBatchData& encoder_inputs) {
        return model_->encode(encoder_inputs);
      });

  EncoderEmbeddingGatherVisitor gather(device_,
                                       mm_data.type(),
                                       params.attention.host.kv_seq_lens,
                                       params.attention.host.q_seq_lens);
  CHECK(mm_data.foreach (gather));
  CHECK(gather.finish(mm_data));

  return model_->get_input_embeddings(
      tokens, mm_data, params.embedding.deep_stacks_);
}

void VlmExecutorImpl::prepare_graph_input(const torch::Tensor& tokens,
                                          const torch::Tensor& positions,
                                          std::vector<KVCache>& kv_caches,
                                          const ModelParamsView& params) {
  // Decode-only double-buffer graph prewarm. A prefill-only graph executor
  // must not prewarm decode slots. The worker gates this on decode-phase input.
  if (llm_executor_ && options_.enable_graph()) {
    llm_executor_->prepare_graph_input(
        tokens, collapse_mrope_decode_positions(positions), kv_caches, params);
  }
}

void VlmExecutorImpl::set_dp_process_group(ProcessGroup* group) {
  if (llm_executor_) {
    llm_executor_->set_dp_process_group(group);
  }
}

torch::Tensor VlmExecutorImpl::collapse_mrope_decode_positions(
    const torch::Tensor& positions) const {
  if (use_mrope_ && positions.dim() == 2) {
    return positions[0].contiguous();
  }
  return positions;
}

}  // namespace xllm

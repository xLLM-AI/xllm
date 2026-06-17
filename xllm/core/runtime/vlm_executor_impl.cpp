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

#include "vlm_executor_impl.h"

#include <glog/logging.h>

#include "common/global_flags.h"
#include "common/metrics.h"
#include "core/framework/config/execution_config.h"
#include "core/framework/multimodal/mm_visitor.h"
#include "platform/device.h"

namespace xllm {

VlmExecutorImpl::VlmExecutorImpl(CausalLM* model,
                                 const ModelArgs& args,
                                 const torch::Device& device,
                                 const runtime::Options& options)
    : model_(dynamic_cast<CausalVLM*>(model)),
      args_(args),
      device_(device),
      options_(options) {
  if (options_.max_encoder_cache_size() > 0) {
    encoder_cache_ = std::make_unique<EncoderCache>(
        options_.max_encoder_cache_size() * 1024 * 1024);
  }

  if (::xllm::ExecutionConfig::get_instance().enable_graph()) {
    llm_executor_ = ExecutorImplFactory::get_instance().create_executor_impl(
        model, args, device, options, Device::type_str());
  }
}

ForwardInput VlmExecutorImpl::prepare_inputs(Batch& batch) {
  return batch.prepare_forward_input(
      options_.num_decoding_tokens(), 0, args_, options_.cp_size());
}

MMDict VlmExecutorImpl::encode(const ModelInputParams& params) {
  return model_->encode(params);
}

ModelOutput VlmExecutorImpl::run(const torch::Tensor& tokens,
                                 const torch::Tensor& positions,
                                 std::vector<KVCache>& kv_caches,
                                 const ModelInputParams& params) {
  torch::NoGradGuard no_grad;
  auto& mm_data = params.multimodal.mm_data;
  if (encoder_cache_) {
    EncoderCacheLookupVisitor lookup(encoder_cache_.get());
    mm_data.foreach (lookup);
  }

  EncoderInputGatherVisitor input_gather;
  mm_data.foreach (input_gather);
  CHECK(input_gather.finish(mm_data));
  mm_data.to(device_);

  MMDict embedding = encode(params);
  EncoderOutputScatterVisitor scatter(embedding);
  mm_data.foreach (scatter);
  CHECK(scatter.finish());

  if (encoder_cache_) {
    EncoderCacheInsertVisitor insert(encoder_cache_.get());
    mm_data.foreach (insert);
  }

  EncoderEmbeddingGatherVisitor gather(device_,
                                       mm_data.type(),
                                       params.attention.host.kv_seq_lens,
                                       params.attention.host.q_seq_lens);
  mm_data.foreach (gather);
  CHECK(gather.finish(mm_data));

  params.embedding.input_embedding =
      model_->get_input_embeddings(tokens, params);

  if (llm_executor_) {
    return llm_executor_->run(tokens, positions, kv_caches, params);
  }

  return model_->forward(tokens, positions, kv_caches, params);
}

}  // namespace xllm

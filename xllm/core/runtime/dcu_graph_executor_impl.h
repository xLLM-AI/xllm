/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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

#include <ATen/hip/HIPContext.h>
#include <ATen/hip/HIPGeneratorImpl.h>
#include <ATen/hip/HIPGraph.h>
#include <c10/hip/HIPCachingAllocator.h>
#include <c10/hip/HIPGuard.h>
#include <c10/hip/HIPStream.h>
#include <torch/torch.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <thread>
#include <unordered_map>
#include <vector>

#include "core/framework/kv_cache/kv_cache.h"
#include "core/framework/model/causal_lm.h"
#include "core/framework/model/model_args.h"
#include "core/framework/model/model_input_params.h"
#include "core/runtime/executor_impl.h"
#include "core/runtime/executor_impl_factory.h"
#include "core/runtime/options.h"
#include "kernels/dcu/piecewise_graphs.h"

namespace xllm::runtime::dcu {

using DcuStream = c10::hip::HIPStream;

class DcuGraphPersistentParam final {
 public:
  DcuGraphPersistentParam(const ModelArgs& args,
                          const torch::Device& device,
                          const runtime::Options& options);

  ~DcuGraphPersistentParam() = default;

  // Updates persistent tensors.
  // When return_capture_params is true, returns a ModelInputParams copy whose
  // attn_metadata points to persistent buffers for graph capture.
  std::optional<ModelInputParams> update(const torch::Tensor& tokens,
                                         const torch::Tensor& positions,
                                         const ModelInputParams& params,
                                         uint32_t padded_num_tokens = 0,
                                         bool return_capture_params = false,
                                         uint32_t graph_max_seq_len = 0);

  ModelInputParams init_decode_params(const torch::Tensor& tokens,
                                      const torch::Tensor& positions,
                                      const ModelInputParams& params,
                                      uint32_t padded_num_tokens,
                                      uint32_t graph_max_seq_len);

  void update_decode_input_buffer(const torch::Tensor& tokens,
                                  const torch::Tensor& positions,
                                  const ModelInputParams& params,
                                  uint32_t padded_num_tokens);

  torch::Tensor persistent_tokens(uint32_t n_tokens) const {
    return n_tokens > 0 ? tokens_.slice(0, 0, n_tokens) : tokens_;
  }

  torch::Tensor persistent_positions(uint32_t n_tokens) const {
    if (n_tokens == 0) {
      return positions_;
    }

    const int32_t slice_dim = use_mrope_ ? 1 : 0;
    return positions_.slice(slice_dim, 0, n_tokens);
  }

  torch::Tensor hidden_states(uint32_t n_tokens) const {
    return n_tokens > 0 ? output_.slice(0, 0, n_tokens) : output_;
  }

  torch::Tensor aux_hidden_states(uint32_t n_tokens) const {
    if (!aux_hidden_states_.defined() || aux_hidden_states_.numel() == 0) {
      return aux_hidden_states_;
    }

    return n_tokens > 0 ? aux_hidden_states_.slice(0, 0, n_tokens)
                        : aux_hidden_states_;
  }

  void set_hidden_states(const torch::Tensor& value);
  void set_aux_hidden_states(const torch::Tensor& value);

  std::size_t get_persistent_tensor_bytes() const;

  bool use_mrope() const { return use_mrope_; }

 private:
  torch::Tensor q_seq_lens(uint32_t size) const {
    return size > 0 ? q_seq_lens_.slice(0, 0, size) : q_seq_lens_;
  }

  torch::Tensor kv_seq_lens(uint32_t size) const {
    return size > 0 ? kv_seq_lens_.slice(0, 0, size) : kv_seq_lens_;
  }

  torch::Tensor q_seq_lens_values(uint32_t size) const {
    return size > 0 ? q_seq_lens_values_.slice(0, 0, size) : q_seq_lens_values_;
  }

  torch::Tensor kv_seq_lens_values(uint32_t size) const {
    return size > 0 ? kv_seq_lens_values_.slice(0, 0, size)
                    : kv_seq_lens_values_;
  }

  torch::Tensor new_cache_slots(uint32_t size) const {
    return size > 0 ? new_cache_slots_.slice(0, 0, size) : new_cache_slots_;
  }

  torch::Tensor block_tables(uint32_t rows) const {
    return rows > 0 ? block_table_.slice(0, 0, rows) : block_table_;
  }

  torch::Tensor paged_kv_indptr(uint32_t batch_size) const {
    return batch_size > 0 ? paged_kv_indptr_.slice(0, 0, batch_size + 1)
                          : paged_kv_indptr_;
  }

  torch::Tensor paged_kv_indices(uint32_t size) const {
    return size > 0 ? paged_kv_indices_.slice(0, 0, size) : paged_kv_indices_;
  }

  torch::Tensor paged_kv_last_page_len(uint32_t batch_size) const {
    return batch_size > 0 ? paged_kv_last_page_len_.slice(0, 0, batch_size)
                          : paged_kv_last_page_len_;
  }

  torch::Tensor decode_qo_indptr(uint32_t batch_size) const {
    return batch_size > 0 ? decode_qo_indptr_.slice(0, 0, batch_size + 1)
                          : decode_qo_indptr_;
  }

 private:
  ModelArgs args_;
  torch::Device device_;
  runtime::Options options_;

  bool use_mrope_ = false;
  uint32_t num_decoding_tokens_ = 1;

  // persistent input tensors
  torch::Tensor tokens_;
  torch::Tensor positions_;
  torch::Tensor new_cache_slots_;
  torch::Tensor block_table_;
  torch::Tensor q_seq_lens_;
  torch::Tensor kv_seq_lens_;
  torch::Tensor q_seq_lens_values_;
  torch::Tensor kv_seq_lens_values_;

  // paged KV metadata
  torch::Tensor paged_kv_indptr_;
  torch::Tensor paged_kv_indices_;
  torch::Tensor paged_kv_last_page_len_;
  torch::Tensor decode_qo_indptr_;

  // optional embedding
  torch::Tensor input_embeds_;

  // persistent output tensors
  torch::Tensor output_;
  torch::Tensor aux_hidden_states_;
};

class DcuGraph final {
 public:
  DcuGraph(DcuGraphPersistentParam& persistent_param,
           c10::DeviceIndex device_index,
           DcuStream capture_stream)
      : persistent_param_(persistent_param),
        device_index_(device_index),
        capture_stream_(capture_stream) {}

  bool capture(CausalLM* model,
               const runtime::Options& options,
               const torch::Tensor& tokens,
               const torch::Tensor& positions,
               const ModelInputParams& params,
               std::vector<KVCache>& kv_cache,
               uint32_t bucket_num_tokens,
               const at::hip::MempoolId_t& pool,
               bool use_piecewise,
               uint32_t graph_max_seq_len = 0);

  ModelOutput replay(const torch::Tensor& tokens,
                     const torch::Tensor& positions,
                     std::vector<KVCache>& kv_cache,
                     const ModelInputParams& params);

  torch::Tensor get_hidden_states(uint32_t n_tokens) const {
    return persistent_param_.hidden_states(n_tokens);
  }

 private:
  at::cuda::CUDAGraph graph_;
  bool is_piecewise_ = false;
  std::unique_ptr<::xllm::runtime::dcu::PiecewiseGraphs> piecewise_graph_;

  uint32_t padded_num_tokens_ = 0;
  uint32_t graph_max_seq_len_ = 0;

  DcuGraphPersistentParam& persistent_param_;
  c10::DeviceIndex device_index_;
  DcuStream capture_stream_;
};

class DcuGraphExecutorImpl final : public ExecutorImpl {
 public:
  DcuGraphExecutorImpl(CausalLM* model,
                       const ModelArgs& args,
                       const torch::Device& device,
                       const runtime::Options& options);

  ~DcuGraphExecutorImpl() override;

  ForwardInput prepare_inputs(Batch& batch) override;

  ModelOutput run(const torch::Tensor& tokens,
                  const torch::Tensor& positions,
                  std::vector<KVCache>& kv_caches,
                  const ModelInputParams& params) override;

 private:
  uint32_t get_bucket_num_tokens(uint32_t num_tokens) const;

  uint32_t get_graph_max_seq_len(uint32_t kv_max_seq_len) const;

  uint32_t get_graph_shape_id(uint32_t bucket_num_tokens,
                              uint32_t graph_max_seq_len) const;

  ModelOutput attach_aux_hidden_states_if_needed(
      const torch::Tensor& hidden_states,
      uint32_t n_tokens) const;

  static DcuStream get_capture_stream(c10::DeviceIndex device_index);

 private:
  CausalLM* model_;  // not owned

  ModelArgs args_;
  torch::Device device_;
  runtime::Options options_;

  std::unordered_map<uint32_t, std::unique_ptr<DcuGraph>> graphs_;

  std::unordered_map<uint32_t, std::unique_ptr<DcuGraph>> prefill_graphs_;

  std::unique_ptr<DcuGraphPersistentParam> persistent_param_;

  at::hip::MempoolId_t graph_pool_;

  int64_t max_tokens_for_graph_mode_ = 0;
};

REGISTER_EXECUTOR("dcu", DcuGraphExecutorImpl);

}  // namespace xllm::runtime::dcu

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

#include "onerec_acl_graph_executor.h"

#include <absl/container/flat_hash_map.h>
#include <acl/acl.h>
#include <glog/logging.h>

#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wattributes"
#endif

#include <torch_npu/csrc/core/npu/NPUGraph.h>

#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

#include <torch_npu/csrc/libs/init_npu.h>
#include <torch_npu/torch_npu.h>

#include <algorithm>
#include <cstdint>
#include <mutex>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "core/common/global_flags.h"
#include "core/common/metrics.h"
#include "core/util/utils.h"
#include "platform/npu/device_capture_lock.h"

namespace xllm::npu {
namespace {

void append_tensor_key(std::ostringstream& os,
                       const char* name,
                       const torch::Tensor& tensor) {
  os << '|' << name << ':';
  if (!tensor.defined()) {
    os << "undef";
    return;
  }
  os << static_cast<int>(tensor.scalar_type()) << ':';
  for (const auto dim : tensor.sizes()) {
    os << dim << ',';
  }
}

void append_int_vec_key(std::ostringstream& os,
                        const char* name,
                        const std::vector<int32_t>& values) {
  os << '|' << name << ':';
  for (const auto value : values) {
    os << value << ',';
  }
}

void append_tensor_vec_key(std::ostringstream& os,
                           const char* name,
                           const std::vector<torch::Tensor>& tensors) {
  os << '|' << name << "_size:" << tensors.size();
  for (size_t i = 0; i < tensors.size(); ++i) {
    const std::string item_name = std::string(name) + "_" + std::to_string(i);
    append_tensor_key(os, item_name.c_str(), tensors[i]);
  }
}

void append_tensor_vec_identity_key(std::ostringstream& os,
                                    const char* name,
                                    const std::vector<torch::Tensor>& tensors) {
  append_tensor_vec_key(os, name, tensors);
  os << '|' << name << "_ptrs:";
  for (const auto& tensor : tensors) {
    os << (tensor.defined() ? reinterpret_cast<uintptr_t>(tensor.data_ptr())
                            : uintptr_t{0})
       << ',';
  }
}

int64_t get_decoder_output_tokens(const torch::Tensor& tokens,
                                  const ModelInputParams& params,
                                  const ModelArgs& args) {
  const auto* onerec_params = params.onerec_params();
  if (onerec_params != nullptr &&
      onerec_params->decoder_context_embedding.defined()) {
    const int64_t hidden_size = std::max<int64_t>(1, args.hidden_size());
    return onerec_params->decoder_context_embedding.numel() / hidden_size;
  }
  return tokens.defined() && tokens.dim() >= 1 ? tokens.size(0) : 0;
}

std::string build_graph_key(const torch::Tensor& tokens,
                            const torch::Tensor& positions,
                            const ModelInputParams& params,
                            const ModelArgs& args) {
  const auto* onerec_params = params.onerec_params();
  CHECK(onerec_params != nullptr);
  const bool is_decode =
      onerec_params->rec_stage == OneRecModelInputParams::RecStage::DECODE;
  const bool use_xattn = params.onerec_xattention_params() != nullptr;
  const bool is_xattn_decode = is_decode && use_xattn;
  std::ostringstream os;
  os << (is_decode ? "onerec_decode" : "onerec_prefill")
     << "|out_tokens:" << get_decoder_output_tokens(tokens, params, args)
     << "|is_first:" << onerec_params->is_first_prefill
     << "|has_encoder_output:" << onerec_params->has_encoder_output
     << "|bs:" << onerec_params->bs
     << "|group_width:" << onerec_params->group_width
     << "|seq_len:" << onerec_params->seq_len
     << "|num_sequences:" << params.num_sequences << "|use_xattn:" << use_xattn
     << "|use_moe:" << args.use_moe();
  if (!use_xattn) {
    os << "|encoder_max_seq_len:" << onerec_params->encoder_max_seq_len;
  }
  if (!is_xattn_decode) {
    os << "|q_max_seq_len:" << params.q_max_seq_len
       << "|kv_max_seq_len:" << params.kv_max_seq_len;
  }
  append_tensor_key(os, "tokens", tokens);
  append_tensor_key(os, "positions", positions);
  append_tensor_key(
      os, "decoder_context", onerec_params->decoder_context_embedding);
  append_tensor_key(
      os, "encoder_seq_lens_tensor", onerec_params->encoder_seq_lens_tensor);
  append_tensor_key(
      os, "cross_kv_cu", onerec_params->cross_attn_kv_cu_seq_lens);
  append_tensor_key(
      os, "cross_new_slots", onerec_params->cross_attn_new_cache_slots);
  append_tensor_key(
      os, "cross_block_tables", onerec_params->cross_attn_block_tables);
  append_tensor_key(os, "kv_seq_lens", params.kv_seq_lens);
  append_tensor_key(os, "new_cache_slots", params.new_cache_slots);
  append_tensor_key(os, "block_tables", params.block_tables);
  if (!use_xattn) {
    append_int_vec_key(os, "encoder_seq_lens", onerec_params->encoder_seq_lens);
    append_int_vec_key(
        os, "cross_kv_cu_vec", onerec_params->cross_attn_kv_cu_seq_lens_vec);
  }
  if (!is_xattn_decode) {
    append_int_vec_key(os, "kv_seq_lens_vec", params.kv_seq_lens_vec);
  }

  if (const auto* xattn_params = params.onerec_xattention_params()) {
    auto append_workspace_key = [&](const char* name,
                                    const std::vector<torch::Tensor>& tensors) {
      if (is_decode) {
        append_tensor_vec_identity_key(os, name, tensors);
      } else {
        append_tensor_vec_key(os, name, tensors);
      }
    };
    append_workspace_key("unshared_k", xattn_params->unshared_k_caches);
    append_workspace_key("unshared_v", xattn_params->unshared_v_caches);
    append_workspace_key("shared_k", xattn_params->shared_k_caches);
    append_workspace_key("shared_v", xattn_params->shared_v_caches);
    append_tensor_key(os, "beam_width", xattn_params->beam_width_tensor);
    append_tensor_key(os, "current_round", xattn_params->current_round_tensor);
  }
  return os.str();
}

class OneRecGraphParam {
 public:
  OneRecGraphParam(const ModelArgs& args,
                   const torch::Device& device,
                   const runtime::Options& options)
      : args_(args), device_(device), options_(options) {}

  ModelInputParams update(CausalLM* model,
                          const torch::Tensor& tokens,
                          const torch::Tensor& positions,
                          const ModelInputParams& params,
                          const torch::Tensor& encoder_output) {
    const auto* source_xattn_params = params.onerec_xattention_params();
    if (source_xattn_params != nullptr &&
        source_xattn_params->rec_stage ==
            OneRecModelInputParams::RecStage::PREFILL &&
        source_xattn_params->is_first_prefill && encoder_output.defined()) {
      CHECK_EQ(encoder_output.dim(), 3);
      CHECK(source_xattn_params->cross_attn_new_cache_slots.defined());
      CHECK_EQ(encoder_output.size(0) * encoder_output.size(1),
               source_xattn_params->cross_attn_new_cache_slots.numel());
    }

    copy_tensor(tokens, persistent_tokens_);
    copy_tensor(positions, persistent_positions_);
    copy_tensor(encoder_output, persistent_encoder_output_);

    params_for_capture_ = params;
    copy_tensor(params.kv_seq_lens, persistent_kv_seq_lens_);
    copy_tensor(params.new_cache_slots, persistent_new_cache_slots_);
    copy_tensor(params.block_tables, persistent_block_tables_);
    bind_tensor(persistent_kv_seq_lens_, params_for_capture_.kv_seq_lens);
    bind_tensor(persistent_new_cache_slots_,
                params_for_capture_.new_cache_slots);
    bind_tensor(persistent_block_tables_, params_for_capture_.block_tables);

    const auto* source_onerec_params = params.onerec_params();
    CHECK(source_onerec_params != nullptr);
    auto& onerec_params = params_for_capture_.mutable_onerec_params();
    copy_tensor(source_onerec_params->decoder_context_embedding,
                persistent_decoder_context_embedding_);
    copy_tensor(source_onerec_params->encoder_seq_lens_tensor,
                persistent_encoder_seq_lens_tensor_);
    copy_tensor(source_onerec_params->cross_attn_kv_cu_seq_lens,
                persistent_cross_attn_kv_cu_seq_lens_);
    copy_tensor(source_onerec_params->cross_attn_new_cache_slots,
                persistent_cross_attn_new_cache_slots_);
    copy_tensor(source_onerec_params->cross_attn_block_tables,
                persistent_cross_attn_block_tables_);
    bind_tensor(persistent_decoder_context_embedding_,
                onerec_params.decoder_context_embedding);
    bind_tensor(persistent_encoder_seq_lens_tensor_,
                onerec_params.encoder_seq_lens_tensor);
    bind_tensor(persistent_cross_attn_kv_cu_seq_lens_,
                onerec_params.cross_attn_kv_cu_seq_lens);
    bind_tensor(persistent_cross_attn_new_cache_slots_,
                onerec_params.cross_attn_new_cache_slots);
    bind_tensor(persistent_cross_attn_block_tables_,
                onerec_params.cross_attn_block_tables);

    if (source_xattn_params != nullptr) {
      auto& xattn_params =
          params_for_capture_.mutable_onerec_xattention_params();
      copy_tensor(source_xattn_params->current_round_tensor,
                  persistent_current_round_tensor_);
      bind_tensor(persistent_current_round_tensor_,
                  xattn_params.current_round_tensor);
    }

    ensure_hidden_states(tokens, params);
    const bool use_cross_block_cache =
        params.onerec_xattention_params() != nullptr;
    if (!use_cross_block_cache) {
      ensure_cross_kv_caches(params, encoder_output);
    }
    bind_model(model, onerec_params.is_first_prefill && !use_cross_block_cache);
    return params_for_capture_;
  }

  torch::Tensor tokens() const { return persistent_tokens_; }
  torch::Tensor positions() const { return persistent_positions_; }
  torch::Tensor hidden_states() const { return hidden_states_; }

  void set_hidden_states(const torch::Tensor& value) {
    CHECK(hidden_states_.defined());
    CHECK_EQ(hidden_states_.sizes(), value.sizes());
    hidden_states_.copy_(value, /*non_blocking=*/true);
  }

 private:
  static void copy_tensor(const torch::Tensor& src, torch::Tensor& dst) {
    if (!src.defined()) {
      dst = torch::Tensor();
      return;
    }
    if (!dst.defined()) {
      dst = torch::empty_like(src);
    }
    CHECK_EQ(dst.sizes(), src.sizes());
    CHECK_EQ(dst.scalar_type(), src.scalar_type());
    CHECK_EQ(dst.device(), src.device());
    if (src.numel() > 0) {
      dst.copy_(src, /*non_blocking=*/true);
    }
  }

  static void bind_tensor(const torch::Tensor& persistent,
                          torch::Tensor& target) {
    if (persistent.defined()) {
      target = persistent;
    }
  }

  void bind_model(CausalLM* model, bool bind_cross_kv_caches) {
    model->bind_onerec_prefill_graph_buffers(
        persistent_encoder_output_,
        bind_cross_kv_caches ? cross_k_caches_ : empty_tensor_vec_,
        bind_cross_kv_caches ? cross_v_caches_ : empty_tensor_vec_);
  }

  void ensure_hidden_states(const torch::Tensor& tokens,
                            const ModelInputParams& params) {
    const int64_t output_tokens =
        get_decoder_output_tokens(tokens, params, args_);
    CHECK_GT(output_tokens, 0) << "OneRec graph output is empty.";
    const auto options = torch::TensorOptions()
                             .dtype(util::parse_dtype(args_.dtype(), device_))
                             .device(device_);
    const std::vector<int64_t> shape = {output_tokens, args_.hidden_size()};
    if (!hidden_states_.defined() || hidden_states_.sizes().vec() != shape) {
      hidden_states_ = torch::empty(shape, options);
    }
  }

  void ensure_cross_kv_caches(const ModelInputParams& params,
                              const torch::Tensor& encoder_output) {
    const auto* onerec_params = params.onerec_params();
    if (onerec_params == nullptr || !onerec_params->is_first_prefill ||
        !encoder_output.defined()) {
      return;
    }
    const int64_t bs = encoder_output.size(0);
    const int64_t seq_len = encoder_output.size(1);
    const auto general_kv_heads = args_.n_kv_heads();
    const auto decoder_kv_heads = args_.decoder_n_kv_heads().has_value()
                                      ? args_.decoder_n_kv_heads()
                                      : general_kv_heads;
    const int64_t kv_heads = decoder_kv_heads.value_or(args_.decoder_n_heads());
    const int64_t kv_heads_per_rank =
        kv_heads / std::max<int32_t>(1, options_.tp_size());
    const int64_t kv_hidden_size = kv_heads_per_rank * args_.decoder_head_dim();
    const std::vector<int64_t> shape = {bs, seq_len, kv_hidden_size};
    const auto options = torch::TensorOptions()
                             .dtype(encoder_output.dtype())
                             .device(encoder_output.device());
    const size_t layer_num = static_cast<size_t>(args_.n_layers());
    cross_k_caches_.resize(layer_num);
    cross_v_caches_.resize(layer_num);
    for (size_t i = 0; i < layer_num; ++i) {
      if (!cross_k_caches_[i].defined() ||
          cross_k_caches_[i].sizes().vec() != shape ||
          cross_k_caches_[i].dtype() != encoder_output.dtype() ||
          cross_k_caches_[i].device() != encoder_output.device()) {
        cross_k_caches_[i] = torch::empty(shape, options);
      }
      if (!cross_v_caches_[i].defined() ||
          cross_v_caches_[i].sizes().vec() != shape ||
          cross_v_caches_[i].dtype() != encoder_output.dtype() ||
          cross_v_caches_[i].device() != encoder_output.device()) {
        cross_v_caches_[i] = torch::empty(shape, options);
      }
    }
  }

  const ModelArgs& args_;
  torch::Device device_;
  runtime::Options options_;
  ModelInputParams params_for_capture_;
  torch::Tensor persistent_tokens_;
  torch::Tensor persistent_positions_;
  torch::Tensor persistent_encoder_output_;
  torch::Tensor persistent_kv_seq_lens_;
  torch::Tensor persistent_new_cache_slots_;
  torch::Tensor persistent_block_tables_;
  torch::Tensor persistent_decoder_context_embedding_;
  torch::Tensor persistent_encoder_seq_lens_tensor_;
  torch::Tensor persistent_cross_attn_kv_cu_seq_lens_;
  torch::Tensor persistent_cross_attn_new_cache_slots_;
  torch::Tensor persistent_cross_attn_block_tables_;
  torch::Tensor persistent_current_round_tensor_;
  std::vector<torch::Tensor> cross_k_caches_;
  std::vector<torch::Tensor> cross_v_caches_;
  std::vector<torch::Tensor> empty_tensor_vec_;
  torch::Tensor hidden_states_;
};

class OneRecAclGraph {
 public:
  OneRecAclGraph(const ModelArgs& args,
                 const torch::Device& device,
                 const runtime::Options& options)
      : param_(args, device, options), device_index_(device.index()) {
    capture_stream_ = c10_npu::getStreamFromPool(true, device_index_);
  }

  bool capture(CausalLM* model,
               const torch::Tensor& tokens,
               const torch::Tensor& positions,
               std::vector<KVCache>& kv_caches,
               const ModelInputParams& params,
               const torch::Tensor& encoder_output) {
    // Device synchronization is not allowed while any stream on the device is
    // being captured. Serialize the complete quiesce-and-capture sequence so a
    // second pipeline cannot synchronize the device during another capture.
    auto& capture_lock =
        DeviceCaptureLock::get_instance().get_lock(device_index_);
    std::lock_guard<std::mutex> lock_guard(capture_lock);

    torch::npu::synchronize();
    auto params_for_capture =
        param_.update(model, tokens, positions, params, encoder_output);
    aclrtStream stream = c10_npu::getCurrentNPUStream(device_index_).stream();
    aclrtSynchronizeStream(stream);

    bool need_restore_stream = false;
    if (c10_npu::getCurrentNPUStream(device_index_) ==
        c10_npu::getDefaultNPUStream(device_index_)) {
      c10_npu::setCurrentNPUStream(capture_stream_.value());
      aclrtSynchronizeStream(capture_stream_.value().stream());
      need_restore_stream = true;
    }

    graph_.capture_begin(
        {0, 0}, aclmdlRICaptureMode::ACL_MODEL_RI_CAPTURE_MODE_THREAD_LOCAL);
    auto forward_result = model->forward(
        param_.tokens(), param_.positions(), kv_caches, params_for_capture);
    param_.set_hidden_states(forward_result.hidden_states);
    graph_.capture_end();

    if (need_restore_stream) {
      c10_npu::setCurrentNPUStream(c10_npu::getDefaultNPUStream(device_index_));
    }

    // NPUGraph capture and replay share the device's default generator. Keep
    // validation replay in the same device-wide critical section.
    aclrtSynchronizeStream(stream);
    graph_.replay();
    captured_ = true;
    return true;
  }

  ModelOutput replay(CausalLM* model,
                     const torch::Tensor& tokens,
                     const torch::Tensor& positions,
                     const ModelInputParams& params,
                     const torch::Tensor& encoder_output) {
    CHECK(captured_);
    CHECK(params.onerec_params() != nullptr);
    // Serialize host-side parameter updates and graph submission while the
    // submitted NPU streams continue to execute independently.
    auto& device_graph_lock =
        DeviceCaptureLock::get_instance().get_lock(device_index_);
    std::lock_guard<std::mutex> lock_guard(device_graph_lock);
    param_.update(model, tokens, positions, params, encoder_output);
    graph_.replay();
    return ModelOutput(param_.hidden_states());
  }

  ModelOutput output() const { return ModelOutput(param_.hidden_states()); }

 private:
  OneRecGraphParam param_;
  c10_npu::NPUGraph graph_;
  std::optional<c10_npu::NPUStream> capture_stream_;
  c10::DeviceIndex device_index_;
  bool captured_ = false;
};

}  // namespace

class OneRecAclGraphExecutor::Impl {
 public:
  Impl(CausalLM* model,
       const ModelArgs& args,
       const torch::Device& device,
       const runtime::Options& options)
      : model_(model), args_(args), device_(device), options_(options) {
    CHECK(model_ != nullptr);
    CHECK(model_->supports_onerec_graph());
  }

  bool is_graph_candidate(const ModelInputParams& params) const {
    const auto* onerec_params = params.onerec_params();
    if (onerec_params == nullptr || onerec_params->is_encoder_forward ||
        params.layer_synchronizer != nullptr) {
      return false;
    }
    if (onerec_params->rec_stage == OneRecModelInputParams::RecStage::PREFILL) {
      return FLAGS_enable_onerec_prefill_acl_graph;
    }
    return params.batch_forward_type.is_decode();
  }

  ModelOutput run(const torch::Tensor& tokens,
                  const torch::Tensor& positions,
                  std::vector<KVCache>& kv_caches,
                  const ModelInputParams& params) {
    const auto* onerec_params = params.onerec_params();
    CHECK(onerec_params != nullptr);
    const bool is_prefill =
        onerec_params->rec_stage == OneRecModelInputParams::RecStage::PREFILL;
    const char* stage_name = is_prefill ? "prefill" : "decode";
    const int64_t output_tokens =
        get_decoder_output_tokens(tokens, params, args_);
    const bool exceeds_prefill_graph_limit =
        is_prefill && FLAGS_max_tokens_for_graph_mode > 0 &&
        output_tokens > FLAGS_max_tokens_for_graph_mode;
    if (exceeds_prefill_graph_limit) {
      VLOG(kGraphExecutorLogVerboseLevel)
          << "OneRec prefill output token count " << output_tokens
          << " exceeds max_tokens_for_graph_mode ("
          << FLAGS_max_tokens_for_graph_mode << "), falling back to eager mode";
    }
    if (output_tokens <= 0 || exceeds_prefill_graph_limit) {
      if (is_prefill && onerec_params->is_first_prefill) {
        last_first_prefill_graph_ready_ = false;
      }
      COUNTER_INC(num_model_execution_total_eager);
      return model_->forward(tokens, positions, kv_caches, params);
    }

    torch::Tensor encoder_output;
    if (is_prefill && onerec_params->has_encoder_output) {
      encoder_output = model_->get_onerec_graph_encoder_output();
      if (!encoder_output.defined()) {
        LOG_FIRST_N(WARNING, 1)
            << "Falling back to eager mode because OneRec encoder output is "
               "not available for decoder prefill ACL graph.";
        if (onerec_params->is_first_prefill) {
          last_first_prefill_graph_ready_ = false;
        }
        COUNTER_INC(num_model_execution_total_eager);
        return model_->forward(tokens, positions, kv_caches, params);
      }
    }

    const std::string graph_key =
        build_graph_key(tokens, positions, params, args_);
    std::unique_lock<std::mutex> graph_lock(graph_mutex_);
    if (is_prefill && !onerec_params->is_first_prefill &&
        !last_first_prefill_graph_ready_) {
      graph_lock.unlock();
      COUNTER_INC(num_model_execution_total_eager);
      return model_->forward(tokens, positions, kv_caches, params);
    }
    auto it = graphs_.find(graph_key);
    if (it != graphs_.end()) {
      VLOG(kGraphExecutorLogVerboseLevel) << "OneRecAclGraphExecutor::run() in "
                                          << stage_name << " replay mode";
      if (is_prefill && onerec_params->is_first_prefill) {
        last_first_prefill_graph_ready_ = true;
      }
      return it->second->replay(
          model_, tokens, positions, params, encoder_output);
    }

    LOG(INFO) << "Lazy capturing OneRec " << stage_name
              << " ACL graph triggered, cached_graphs: " << graphs_.size()
              << ", output_tokens: " << output_tokens;
    auto graph = std::make_unique<OneRecAclGraph>(args_, device_, options_);
    VLOG(kGraphExecutorLogVerboseLevel)
        << "OneRecAclGraphExecutor::run() in " << stage_name << " capture mode";
    const bool capture_success = graph->capture(
        model_, tokens, positions, kv_caches, params, encoder_output);
    if (capture_success) {
      LOG(INFO) << "Lazy capturing OneRec " << stage_name
                << " ACL graph done, key length: " << graph_key.size()
                << ", output_tokens: " << output_tokens
                << ", is_first_prefill: " << onerec_params->is_first_prefill;
      auto result = graph->output();
      graphs_[graph_key] = std::move(graph);
      if (is_prefill && onerec_params->is_first_prefill) {
        last_first_prefill_graph_ready_ = true;
      }
      return result;
    }

    LOG(FATAL) << "Failed to capture OneRec " << stage_name
               << " ACL graph, output_tokens: " << output_tokens;
    return ModelOutput();
  }

  void reset_first_prefill_graph_state_if_needed(
      const ModelInputParams& params) {
    const auto* onerec_params = params.onerec_params();
    if (onerec_params != nullptr &&
        onerec_params->rec_stage == OneRecModelInputParams::RecStage::PREFILL &&
        !onerec_params->is_encoder_forward && onerec_params->is_first_prefill) {
      last_first_prefill_graph_ready_ = false;
    }
  }

 private:
  CausalLM* model_;
  ModelArgs args_;
  torch::Device device_;
  runtime::Options options_;
  absl::flat_hash_map<std::string, std::unique_ptr<OneRecAclGraph>> graphs_;
  std::mutex graph_mutex_;
  bool last_first_prefill_graph_ready_ = false;
};

OneRecAclGraphExecutor::OneRecAclGraphExecutor(CausalLM* model,
                                               const ModelArgs& args,
                                               const torch::Device& device,
                                               const runtime::Options& options)
    : impl_(std::make_unique<Impl>(model, args, device, options)) {}

OneRecAclGraphExecutor::~OneRecAclGraphExecutor() = default;

bool OneRecAclGraphExecutor::is_graph_candidate(
    const ModelInputParams& params) const {
  return impl_->is_graph_candidate(params);
}

ModelOutput OneRecAclGraphExecutor::run(const torch::Tensor& tokens,
                                        const torch::Tensor& positions,
                                        std::vector<KVCache>& kv_caches,
                                        const ModelInputParams& params) {
  return impl_->run(tokens, positions, kv_caches, params);
}

void OneRecAclGraphExecutor::reset_first_prefill_graph_state_if_needed(
    const ModelInputParams& params) {
  impl_->reset_first_prefill_graph_state_if_needed(params);
}

}  // namespace xllm::npu

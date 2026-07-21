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

#include "acl_graph_executor_impl.h"

#include <c10/core/Device.h>
#include <c10/core/TensorOptions.h>
#include <glog/logging.h>
#include <torch/torch.h>
#include <torch_npu/csrc/libs/init_npu.h>
#include <torch_npu/torch_npu.h>

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <sstream>

#include "core/common/global_flags.h"
#ifdef TORCH_HIGHER_THAN_PTA6
#include <torch_npu/csrc/framework/OpCommand.h>
#else
#include <torch_npu/csrc/aten/NPUNativeFunctions.h>
#include <torch_npu/csrc/framework/utils/OpPreparation.h>
#endif
#include "core/common/global_flags.h"
#include "core/common/metrics.h"
#include "core/util/utils.h"
#include "platform/npu/device_capture_lock.h"

// ATB includes
#include <atb/atb_infer.h>
#include <atb/context.h>
#include <atb/operation.h>
#include <customize/custom_paged_attention_function.h>
#include <customize/customize_op_params.h>

#include "pytorch/adapter/utils/utils.h"

namespace xllm::npu {

namespace {
std::pair<torch::Tensor, torch::Tensor> find_attention_plan_kv_cache(
    const std::vector<KVCache>& kv_caches) {
  for (const auto& cache : kv_caches) {
    auto k_cache = cache.get_k_cache();
    auto v_cache = cache.get_v_cache();
    if (k_cache.defined() && v_cache.defined() && k_cache.numel() > 0 &&
        v_cache.numel() > 0) {
      return {std::move(k_cache), std::move(v_cache)};
    }
  }
  return {torch::Tensor(), torch::Tensor()};
}

int64_t get_decode_graph_capacity(const runtime::Options& options) {
  CHECK_GT(options.num_decoding_tokens(), 0)
      << "num_decoding_tokens must be > 0 for graph capacity";
  if (FLAGS_enable_atb_spec_kernel) {
    return options.max_seqs_per_batch();
  }
  return options.max_seqs_per_batch() * options.num_decoding_tokens();
}

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

int64_t get_onerec_decoder_output_tokens(const torch::Tensor& tokens,
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

std::string build_onerec_graph_key(const torch::Tensor& tokens,
                                   const torch::Tensor& positions,
                                   const ModelInputParams& params,
                                   const ModelArgs& args) {
  const auto* onerec_params = params.onerec_params();
  CHECK(onerec_params != nullptr);
  const bool is_decode =
      onerec_params->rec_stage == OneRecModelInputParams::RecStage::DECODE;
  std::ostringstream os;
  os << (is_decode ? "onerec_decode" : "onerec_prefill")
     << "|out_tokens:" << get_onerec_decoder_output_tokens(tokens, params, args)
     << "|is_first:" << onerec_params->is_first_prefill
     << "|has_encoder_output:" << onerec_params->has_encoder_output
     << "|bs:" << onerec_params->bs
     << "|group_width:" << onerec_params->group_width
     << "|seq_len:" << onerec_params->seq_len
     << "|encoder_max_seq_len:" << onerec_params->encoder_max_seq_len
     << "|num_sequences:" << params.num_sequences
     << "|q_max_seq_len:" << params.q_max_seq_len
     << "|kv_max_seq_len:" << params.kv_max_seq_len
     << "|use_xattn:" << (params.onerec_xattention_params() != nullptr)
     << "|use_moe:" << args.use_moe();
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
  append_int_vec_key(os, "encoder_seq_lens", onerec_params->encoder_seq_lens);
  append_int_vec_key(
      os, "cross_kv_cu_vec", onerec_params->cross_attn_kv_cu_seq_lens_vec);
  append_int_vec_key(os, "kv_seq_lens_vec", params.kv_seq_lens_vec);

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

bool is_onerec_decoder_graph_candidate(CausalLM* model,
                                       const ModelInputParams& params) {
  const auto* onerec_params = params.onerec_params();
  if (onerec_params == nullptr || !model->supports_onerec_prefill_graph() ||
      onerec_params->is_encoder_forward ||
      params.layer_synchronizer != nullptr) {
    return false;
  }
  if (onerec_params->rec_stage == OneRecModelInputParams::RecStage::PREFILL) {
    return FLAGS_enable_onerec_prefill_acl_graph;
  }
  return params.batch_forward_type.is_decode();
}
}  // namespace

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

  void bind_model(CausalLM* model, bool bind_cross_kv_caches) {
    model->bind_onerec_prefill_graph_buffers(
        persistent_encoder_output_,
        bind_cross_kv_caches ? cross_k_caches_ : empty_tensor_vec_,
        bind_cross_kv_caches ? cross_v_caches_ : empty_tensor_vec_);
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

  void ensure_hidden_states(const torch::Tensor& tokens,
                            const ModelInputParams& params) {
    const int64_t output_tokens =
        get_onerec_decoder_output_tokens(tokens, params, args_);
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
    (void)kv_caches;
    torch::npu::synchronize();
    auto params_for_capture =
        param_.update(model, tokens, positions, params, encoder_output);
    aclrtStream stream = c10_npu::getCurrentNPUStream(device_index_).stream();
    aclrtSynchronizeStream(stream);

    bool need_restore_stream = false;
    {
      auto& capture_lock =
          ::xllm::npu::DeviceCaptureLock::get_instance().get_lock(
              device_index_);
      std::lock_guard<std::mutex> lock_guard(capture_lock);
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
        c10_npu::setCurrentNPUStream(
            c10_npu::getDefaultNPUStream(device_index_));
      }

      // NPUGraph capture and replay share the device's default generator.
      // Keep the validation replay in the device-wide critical section so a
      // second pipeline cannot start capturing while replay_prologue updates
      // the generator state.
      aclrtSynchronizeStream(stream);
      graph_.replay();
    }
    captured_ = true;
    return true;
  }

  ModelOutput replay(CausalLM* model,
                     const torch::Tensor& tokens,
                     const torch::Tensor& positions,
                     const ModelInputParams& params,
                     const torch::Tensor& encoder_output) {
    CHECK(captured_);
    const auto* onerec_params = params.onerec_params();
    CHECK(onerec_params != nullptr);

    // Each OneRec pipeline owns a separate graph executor, but torch_npu uses
    // one default generator per device for all NPUGraph instances. Serialize
    // the host-side parameter update and graph submission with capture on the
    // same device while allowing the submitted streams to execute in parallel.
    auto& device_graph_lock =
        ::xllm::npu::DeviceCaptureLock::get_instance().get_lock(device_index_);
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

// GraphPersistentParam implementation
GraphPersistentParam::GraphPersistentParam(const ModelArgs& args,
                                           const torch::Device& device,
                                           const runtime::Options& options,
                                           bool need_update_attn_mask)
    : args_(args),
      device_(device),
      options_(options),
      context_for_plan_(nullptr),
      custom_pa_op_for_plan_(nullptr),
      stream_for_plan_(nullptr),
      need_update_attn_mask_(need_update_attn_mask) {
  // Determine whether attention plan needs to be updated based on model type
  // Future logic can be extended here for more complex model-specific behavior
  need_update_attention_plan_ = (args.model_type() != "deepseek_v32" &&
                                 args.model_type() != "glm_moe_dsa");

  // Check if mRoPE is used (for VLM models like qwen2-vl)
  use_mrope_ = !args.rope_scaling_mrope_section().empty();

  // Use max_tokens_per_batch for first dimension size
  // num_decode_tokens
  const int64_t max_tokens_per_batch = FLAGS_max_tokens_per_batch;
  // num_sequences
  const int64_t max_seqs_per_batch = get_decode_graph_capacity(options);
  auto tensor_options = torch::TensorOptions().device(device);

  const int64_t max_seq_len = args_.max_position_embeddings();

  // Create persistent tensors with max_tokens_per_batch as first dimension
  persistent_tokens_ = torch::zeros({max_tokens_per_batch},
                                    torch::dtype(torch::kInt).device(device));
  if (args.rope_scaling_mrope_section().empty()) {
    persistent_positions_ = torch::zeros(
        {max_tokens_per_batch}, torch::dtype(torch::kInt).device(device));
  } else {
    persistent_positions_ = torch::zeros(
        {3, max_tokens_per_batch}, torch::dtype(torch::kInt).device(device));
    use_mrope_ = true;
  }
  persistent_new_cache_slots_ = torch::zeros(
      {max_tokens_per_batch}, torch::dtype(torch::kInt).device(device));

  // Sequence length tensors with max_seqs_per_batch
  q_seq_lens_ = torch::zeros({max_seqs_per_batch},
                             torch::dtype(torch::kInt).device(device));
  kv_seq_lens_ = torch::zeros({max_seqs_per_batch},
                              torch::dtype(torch::kInt).device(device));

  // Block table tensors with maximum possible size
  const auto block_size = options.block_size();
  const int64_t max_block_table_len =
      (max_seq_len + block_size - 1) / block_size + 1;
  persistent_block_tables_ =
      torch::zeros({max_seqs_per_batch, max_block_table_len},
                   torch::dtype(torch::kInt).device(device));

  // Output tensor for hidden states
  torch::Dtype dtype = util::parse_dtype(args.dtype(), device);
  if (args.dtype() == "float" || args.dtype() == "float32") {
    LOG(WARNING)
        << "Acl graph executor init hidden_states compatible with float32 "
           "dtype: float32. This should not happen in production but for test.";
    dtype = torch::kFloat32;
  }
  hidden_states_ = torch::zeros({max_tokens_per_batch, args.hidden_size()},
                                torch::dtype(dtype).device(device));

  // Initialize persistent_mask_ if need_update_attn_mask is true
  if (need_update_attn_mask_) {
    persistent_mask_ = torch::zeros({max_tokens_per_batch, max_seq_len},
                                    torch::dtype(dtype).device(device));
  }

  // Pre-allocate persistent dp/cp ep padding buffers with maximum capacity.
  const int64_t dp_size = options.dp_size();
  const int64_t decode_capacity = get_decode_graph_capacity(options);
  const int64_t padding_buf_capacity = decode_capacity * dp_size;
  auto int_opts = torch::dtype(torch::kInt32).device(device);

  persistent_dp_ep_padding_
      .attn_padding_idx(torch::zeros({padding_buf_capacity}, int_opts))
      .attn_unpadding_idx(torch::zeros({padding_buf_capacity}, int_opts))
      .ffn_padding_idx(torch::zeros({padding_buf_capacity}, int_opts))
      .ffn_unpadding_idx(torch::zeros({padding_buf_capacity}, int_opts))
      .lm_head_skip_padding_token_indices(
          torch::zeros({padding_buf_capacity}, int_opts))
      .gather_prenorm_idx(torch::zeros({padding_buf_capacity}, int_opts))
      .padding_idx(torch::zeros({padding_buf_capacity}, int_opts))
      .un_padding_idx(torch::zeros({padding_buf_capacity}, int_opts))
      .dynamic_ep_idx(torch::zeros({padding_buf_capacity}, int_opts))
      .moe_idx(torch::zeros({padding_buf_capacity}, int_opts))
      .expert_array(torch::zeros({padding_buf_capacity, 1}, int_opts))
      .post_lmhead_gather_indices(
          torch::zeros({padding_buf_capacity}, int_opts));

  persistent_cp_ep_padding_
      .attn_padding_idx(torch::zeros({padding_buf_capacity}, int_opts))
      .attn_unpadding_idx(torch::zeros({padding_buf_capacity}, int_opts))
      .ffn_padding_idx(torch::zeros({padding_buf_capacity}, int_opts))
      .ffn_unpadding_idx(torch::zeros({padding_buf_capacity}, int_opts))
      .lm_head_skip_padding_token_indices(
          torch::zeros({padding_buf_capacity}, int_opts))
      .gather_prenorm_idx(torch::zeros({padding_buf_capacity}, int_opts))
      .padding_idx(torch::zeros({padding_buf_capacity}, int_opts))
      .un_padding_idx(torch::zeros({padding_buf_capacity}, int_opts))
      .dynamic_ep_idx(torch::zeros({padding_buf_capacity}, int_opts))
      .moe_idx(torch::zeros({padding_buf_capacity}, int_opts))
      .expert_array(torch::zeros({padding_buf_capacity, 1}, int_opts));

  // Do not need to create ATB context and custom paged attention operation
  if (args_.head_dim() == 0) {
    return;
  }

  initialize_paged_attention_plan_context(device);
}

namespace {
// Copy src into a pre-allocated persistent buffer. The buffer must already be
// large enough (allocated in GraphPersistentParam constructor).
void copy_into_persistent(torch::Tensor& persistent, const torch::Tensor& src) {
  if (!src.defined() || src.numel() == 0 || !persistent.defined()) {
    return;
  }
  CHECK_LE(src.size(0), persistent.size(0))
      << "dp_ep_padding persistent buffer overflow: src size " << src.size(0)
      << " exceeds pre-allocated capacity " << persistent.size(0);
  persistent.slice(/*dim=*/0, /*start=*/0, /*end=*/src.size(0))
      .copy_(src, /*non_blocking=*/true);
}
}  // namespace

void GraphPersistentParam::update_persistent_dp_ep_padding(
    const DpEpPaddingData& src,
    uint32_t /*padded_tokens*/) {
  // Skip when dp ep padding is not enabled. When enabled, build() always
  // populates attn_padding_idx first, so it is a reliable signal.
  if (!src.attn_padding_idx().defined() ||
      src.attn_padding_idx().numel() == 0) {
    return;
  }
  copy_into_persistent(persistent_dp_ep_padding_.attn_padding_idx(),
                       src.attn_padding_idx());
  copy_into_persistent(persistent_dp_ep_padding_.attn_unpadding_idx(),
                       src.attn_unpadding_idx());
  copy_into_persistent(persistent_dp_ep_padding_.ffn_padding_idx(),
                       src.ffn_padding_idx());
  copy_into_persistent(persistent_dp_ep_padding_.ffn_unpadding_idx(),
                       src.ffn_unpadding_idx());
  copy_into_persistent(
      persistent_dp_ep_padding_.lm_head_skip_padding_token_indices(),
      src.lm_head_skip_padding_token_indices());
  copy_into_persistent(persistent_dp_ep_padding_.gather_prenorm_idx(),
                       src.gather_prenorm_idx());
  copy_into_persistent(persistent_dp_ep_padding_.padding_idx(),
                       src.padding_idx());
  copy_into_persistent(persistent_dp_ep_padding_.un_padding_idx(),
                       src.un_padding_idx());
  copy_into_persistent(persistent_dp_ep_padding_.dynamic_ep_idx(),
                       src.dynamic_ep_idx());
  copy_into_persistent(persistent_dp_ep_padding_.moe_idx(), src.moe_idx());
  copy_into_persistent(persistent_dp_ep_padding_.expert_array(),
                       src.expert_array());
  copy_into_persistent(persistent_dp_ep_padding_.post_lmhead_gather_indices(),
                       src.post_lmhead_gather_indices());
}

void GraphPersistentParam::update_persistent_cp_ep_padding(
    const CpEpPaddingData& src,
    uint32_t /*padded_tokens*/) {
  // Skip when cp ep padding is not enabled. When enabled, build() always
  // populates attn_padding_idx first, so it is a reliable signal.
  if (!src.attn_padding_idx().defined() ||
      src.attn_padding_idx().numel() == 0) {
    return;
  }
  copy_into_persistent(persistent_cp_ep_padding_.attn_padding_idx(),
                       src.attn_padding_idx());
  copy_into_persistent(persistent_cp_ep_padding_.attn_unpadding_idx(),
                       src.attn_unpadding_idx());
  copy_into_persistent(persistent_cp_ep_padding_.ffn_padding_idx(),
                       src.ffn_padding_idx());
  copy_into_persistent(persistent_cp_ep_padding_.ffn_unpadding_idx(),
                       src.ffn_unpadding_idx());
  copy_into_persistent(
      persistent_cp_ep_padding_.lm_head_skip_padding_token_indices(),
      src.lm_head_skip_padding_token_indices());
  copy_into_persistent(persistent_cp_ep_padding_.gather_prenorm_idx(),
                       src.gather_prenorm_idx());
  copy_into_persistent(persistent_cp_ep_padding_.padding_idx(),
                       src.padding_idx());
  copy_into_persistent(persistent_cp_ep_padding_.un_padding_idx(),
                       src.un_padding_idx());
  copy_into_persistent(persistent_cp_ep_padding_.dynamic_ep_idx(),
                       src.dynamic_ep_idx());
  copy_into_persistent(persistent_cp_ep_padding_.moe_idx(), src.moe_idx());
  copy_into_persistent(persistent_cp_ep_padding_.expert_array(),
                       src.expert_array());
}

GraphPersistentParam::~GraphPersistentParam() {
  if (custom_pa_op_for_plan_ != nullptr) {
    atb::DestroyOperation(custom_pa_op_for_plan_);
    custom_pa_op_for_plan_ = nullptr;
  }
  if (stream_for_plan_ != nullptr) {
    aclrtDestroyStream(stream_for_plan_);
    stream_for_plan_ = nullptr;
  }
  if (context_for_plan_ != nullptr) {
    atb::DestroyContext(context_for_plan_);
    context_for_plan_ = nullptr;
  }
}

void GraphPersistentParam::set_aux_hidden_states(const torch::Tensor& value) {
  if (!value.defined()) {
    return;
  }
  const uint32_t result_tokens = value.size(0);
  if (aux_hidden_states_.numel() == 0) {
    // Lazy initialization: create aux_hidden_states tensor if not already
    // created
    const int64_t max_tokens_per_batch = FLAGS_max_tokens_per_batch;
    auto shape = value.sizes().vec();
    shape[0] = max_tokens_per_batch;
    torch::Dtype dtype = util::parse_dtype(args_.dtype(), device_);
    if (args_.dtype() == "float" || args_.dtype() == "float32") {
      dtype = torch::kFloat32;
    }
    aux_hidden_states_ =
        torch::zeros(shape, torch::dtype(dtype).device(device_));
  }
  // Slice to match the actual shape
  auto slice =
      aux_hidden_states_.slice(/*dim=*/0, /*start=*/0, /*end=*/result_tokens);
  // Reshape slice if needed to match value shape
  if (slice.sizes() == value.sizes()) {
    slice.copy_(value, /*non_blocking=*/true);
  }
}

std::optional<ModelInputParams> GraphPersistentParam::update(
    const torch::Tensor& tokens,
    const torch::Tensor& k_cache,
    const torch::Tensor& v_cache,
    const torch::Tensor& positions,
    const ModelInputParams& params,
    uint32_t padded_num_tokens,
    bool return_capture_params) {
  CHECK_GT(padded_num_tokens, 0)
      << "padded_num_tokens must be > 0 when return_capture_params is true";
  const uint32_t actual_num_tokens = tokens.size(0);
  const int64_t actual_batch_size = params.num_sequences;

  // Copy data from input parameters to persistent graph tensors
  if (actual_num_tokens > 0) {
    persistent_tokens_.slice(/*dim=*/0, /*start=*/0, /*end=*/actual_num_tokens)
        .copy_(tokens, /*non_blocking=*/true);
  }
  // mRoPE positions have shape [3, num_tokens], slice on dim 1
  if (actual_num_tokens > 0) {
    if (use_mrope_) {
      persistent_positions_
          .slice(/*dim=*/1, /*start=*/0, /*end=*/actual_num_tokens)
          .copy_(positions, /*non_blocking=*/true);
    } else {
      persistent_positions_
          .slice(/*dim=*/0, /*start=*/0, /*end=*/actual_num_tokens)
          .copy_(positions, /*non_blocking=*/true);
    }
  }
  if (actual_batch_size > 0 && params.q_seq_lens.defined() &&
      params.q_seq_lens.dim() >= 1 && params.q_seq_lens.numel() > 0) {
    const int64_t q_copy_len =
        std::min<int64_t>(actual_batch_size, params.q_seq_lens.size(0));
    if (q_copy_len > 0) {
      q_seq_lens_.slice(/*dim=*/0, /*start=*/0, /*end=*/q_copy_len)
          .copy_(params.q_seq_lens.slice(/*dim=*/0,
                                         /*start=*/0,
                                         /*end=*/q_copy_len),
                 /*non_blocking=*/true);
    }
  }
  if (actual_batch_size > 0 && params.kv_seq_lens.defined() &&
      params.kv_seq_lens.dim() >= 1 && params.kv_seq_lens.numel() > 0) {
    const int64_t kv_copy_len =
        std::min<int64_t>(actual_batch_size, params.kv_seq_lens.size(0));
    if (kv_copy_len > 0) {
      kv_seq_lens_.slice(/*dim=*/0, /*start=*/0, /*end=*/kv_copy_len)
          .copy_(params.kv_seq_lens.slice(/*dim=*/0,
                                          /*start=*/0,
                                          /*end=*/kv_copy_len),
                 /*non_blocking=*/true);
    }
  }
  // Keep padded decode slots valid for empty/local-short DP shards.
  // These tensors are consumed by ATB setup alongside *_seq_lens_vec.
  const int64_t padded_batch_size = static_cast<int64_t>(padded_num_tokens);
  if (padded_batch_size > 0) {
    const int64_t seq_fill_start =
        std::min<int64_t>(actual_batch_size, padded_batch_size);
    if (seq_fill_start < padded_batch_size) {
      q_seq_lens_
          .slice(/*dim=*/0, /*start=*/seq_fill_start, /*end=*/padded_batch_size)
          .fill_(1);
      kv_seq_lens_
          .slice(/*dim=*/0, /*start=*/seq_fill_start, /*end=*/padded_batch_size)
          .fill_(1);
    }
  }

  if (actual_num_tokens > 0 && params.new_cache_slots.defined() &&
      params.new_cache_slots.dim() >= 1 && params.new_cache_slots.numel() > 0) {
    const int64_t slot_copy_len =
        std::min<int64_t>(static_cast<int64_t>(actual_num_tokens),
                          params.new_cache_slots.size(0));
    if (slot_copy_len > 0) {
      persistent_new_cache_slots_
          .slice(/*dim=*/0, /*start=*/0, /*end=*/slot_copy_len)
          .copy_(params.new_cache_slots.slice(/*dim=*/0,
                                              /*start=*/0,
                                              /*end=*/slot_copy_len),
                 /*non_blocking=*/true);
    }
  }
  if (actual_num_tokens < padded_num_tokens) {
    persistent_new_cache_slots_
        .slice(/*dim=*/0,
               /*start=*/actual_num_tokens,
               /*end=*/static_cast<int64_t>(padded_num_tokens))
        .fill_(0);
  }

  // Copy block table data
  if (actual_batch_size > 0 && params.block_tables.defined() &&
      params.block_tables.dim() >= 2 && params.block_tables.numel() > 0) {
    const int64_t block_rows_to_copy =
        std::min<int64_t>(actual_batch_size, params.block_tables.size(0));
    const int64_t actual_block_table_len = params.block_tables.size(1);
    if (block_rows_to_copy > 0 && actual_block_table_len > 0) {
      auto slice_persistent_block_tables =
          persistent_block_tables_
              .slice(/*dim=*/0, /*start=*/0, /*end=*/block_rows_to_copy)
              .slice(/*dim=*/1, /*start=*/0, /*end=*/actual_block_table_len);
      slice_persistent_block_tables.copy_(
          params.block_tables.slice(/*dim=*/0,
                                    /*start=*/0,
                                    /*end=*/block_rows_to_copy),
          /*non_blocking=*/true);
    }
  }
  if (actual_batch_size < padded_batch_size) {
    persistent_block_tables_
        .slice(/*dim=*/0,
               /*start=*/actual_batch_size,
               /*end=*/padded_batch_size)
        .fill_(0);
  }

  // Update persistent embedding from input_embedding if available
  const auto& embedding = params.input_embedding;
  if (embedding.defined() && embedding.dim() >= 2) {
    const int64_t embedding_tokens = embedding.size(0);

    // Initialize persistent_embedding_ if needed and not already initialized
    if (persistent_embedding_.numel() == 0) {
      const int64_t max_tokens_per_batch = FLAGS_max_tokens_per_batch;
      const int64_t embedding_dim = embedding.size(1);
      torch::Dtype dtype = util::parse_dtype(args_.dtype(), device_);
      persistent_embedding_ =
          torch::zeros({max_tokens_per_batch, embedding_dim},
                       torch::dtype(dtype).device(device_));
    }

    // Copy embedding data to persistent buffer
    if (embedding_tokens > 0 && embedding.numel() > 0) {
      persistent_embedding_
          .slice(/*dim=*/0, /*start=*/0, /*end=*/embedding_tokens)
          .copy_(embedding, /*non_blocking=*/true);
    }
  }
  // Update q_cu_seq_lens used by sparse MLA indexer.
  // Empty local DP shards can carry empty q_cu_seq_lens from upper layers;
  // for graph capture we still need a valid non-empty length tensor for padded
  // decode slots.
  if (q_cu_seq_lens_.numel() == 0) {
    const int64_t max_seqs_per_batch = get_decode_graph_capacity(options_);
    q_cu_seq_lens_ = torch::zeros({max_seqs_per_batch + 1},
                                  torch::dtype(torch::kInt).device(device_));
  }
  const bool has_q_cu =
      params.q_cu_seq_lens.defined() && params.q_cu_seq_lens.dim() >= 1;
  const int64_t q_cu_size = (has_q_cu && params.q_cu_seq_lens.numel() > 0)
                                ? params.q_cu_seq_lens.size(0)
                                : 0;
  const int64_t q_cu_copy_len = std::min<int64_t>(actual_batch_size, q_cu_size);
  if (q_cu_copy_len > 0) {
    q_cu_seq_lens_.slice(/*dim=*/0, /*start=*/0, /*end=*/q_cu_copy_len)
        .copy_(params.q_cu_seq_lens.slice(/*dim=*/0,
                                          /*start=*/0,
                                          /*end=*/q_cu_copy_len),
               /*non_blocking=*/true);
  }
  if (padded_batch_size > q_cu_copy_len) {
    auto tail_q_seq_lens = q_seq_lens_.slice(/*dim=*/0,
                                             /*start=*/q_cu_copy_len,
                                             /*end=*/padded_batch_size);
    auto tail_cu = torch::cumsum(tail_q_seq_lens, /*dim=*/0);
    if (q_cu_copy_len > 0) {
      auto last_prefix = q_cu_seq_lens_.slice(/*dim=*/0,
                                              /*start=*/q_cu_copy_len - 1,
                                              /*end=*/q_cu_copy_len);
      tail_cu = tail_cu + last_prefix;
    }
    q_cu_seq_lens_
        .slice(/*dim=*/0,
               /*start=*/q_cu_copy_len,
               /*end=*/padded_batch_size)
        .copy_(tail_cu, /*non_blocking=*/true);
  }

  // Update attention mask only if needed
  if (need_update_attn_mask_) {
    update_attention_mask(params);
  }

  if (tiling_data_.numel() > 0) {
    aclrtStream stream = c10_npu::getCurrentNPUStream().stream();

    if (need_update_attention_plan_ && k_cache.defined() && v_cache.defined() &&
        k_cache.numel() > 0 && v_cache.numel() > 0) {
      plan_paged_attention_tiling(
          tokens, k_cache, v_cache, persistent_block_tables_, params, stream);
    }
  }

  // Update persistent dp/cp ep padding buffers. For capture this ensures
  // stable device addresses are recorded by the graph; for replay this
  // refreshes the data at those same addresses before graph_.replay().
  update_persistent_dp_ep_padding(params.dp_ep_padding_data, padded_num_tokens);
  update_persistent_cp_ep_padding(params.cp_ep_padding_data, padded_num_tokens);

  // Return ModelInputParams with persistent buffer references if requested
  if (return_capture_params) {
    std::optional<ModelInputParams> params_for_capture =
        std::make_optional<ModelInputParams>(params);
    // Set persistent buffers in params_for_capture
    params_for_capture->kv_seq_lens = kv_seq_lens(padded_num_tokens);
    params_for_capture->q_seq_lens = q_seq_lens(padded_num_tokens);
    params_for_capture->kv_seq_lens_vec.resize(padded_num_tokens);
    params_for_capture->q_seq_lens_vec.resize(padded_num_tokens);
    // Copy actual values from original params
    const int64_t kv_vec_size =
        static_cast<int64_t>(params.kv_seq_lens_vec.size());
    const int64_t q_vec_size =
        static_cast<int64_t>(params.q_seq_lens_vec.size());
    const int64_t vec_copy_len =
        std::min<int64_t>(actual_batch_size, std::min(kv_vec_size, q_vec_size));
    for (int64_t i = 0; i < vec_copy_len; i++) {
      params_for_capture->kv_seq_lens_vec[i] = params.kv_seq_lens_vec[i];
      params_for_capture->q_seq_lens_vec[i] = params.q_seq_lens_vec[i];
    }
    // Fill padded positions with default values
    for (int64_t i = vec_copy_len; i < static_cast<int64_t>(padded_num_tokens);
         i++) {
      params_for_capture->kv_seq_lens_vec[i] = 1;
      params_for_capture->q_seq_lens_vec[i] = 1;
    }
    params_for_capture->num_sequences = padded_num_tokens;
    params_for_capture->batch_forward_type = BatchForwardType::DECODE;
    params_for_capture->new_cache_slots =
        persistent_new_cache_slots(padded_num_tokens);
    params_for_capture->block_tables =
        persistent_block_tables(padded_num_tokens);

    // Only set attn_mask if need_update_attn_mask_ is true
    if (need_update_attn_mask_) {
      params_for_capture->graph_buffer.attn_mask =
          persistent_mask(padded_num_tokens);
    }
    params_for_capture->graph_buffer.tiling_data = tiling_data();
    // Set persistent embedding if available
    if (params.input_embedding.defined() && params.input_embedding.dim() >= 2 &&
        persistent_embedding_.defined() && persistent_embedding_.numel() > 0) {
      params_for_capture->input_embedding =
          persistent_embedding(padded_num_tokens);
    }
    // Keep q_cu_seq_lens aligned with padded capture batch.
    if (q_cu_seq_lens_.defined() && q_cu_seq_lens_.numel() > 0) {
      params_for_capture->q_cu_seq_lens =
          q_cu_seq_lens_.slice(/*dim=*/0,
                               /*start=*/0,
                               /*end=*/padded_batch_size);
    }

    // Replace dp/cp ep padding with slices of persistent buffers so that
    // the graph records stable device addresses.  Each slice has the same
    // size as the original tensor but points into the persistent buffer.
    // dp ep and cp ep are mutually exclusive; when neither is enabled the
    // src fields are all undefined and we leave dst untouched so that the
    // captured graph behaves identically to eager mode.
    auto slice_like = [](const torch::Tensor& persistent,
                         const torch::Tensor& src) -> torch::Tensor {
      if (!src.defined() || src.numel() == 0 || !persistent.defined()) {
        return persistent;
      }
      return persistent.slice(/*dim=*/0, /*start=*/0, /*end=*/src.size(0));
    };

    const auto& src_dp = params.dp_ep_padding_data;
    if (src_dp.attn_padding_idx().defined() &&
        src_dp.attn_padding_idx().numel() > 0) {
      auto& dst_dp = params_for_capture->dp_ep_padding_data;
      dst_dp.attn_padding_idx(
          slice_like(persistent_dp_ep_padding_.attn_padding_idx(),
                     src_dp.attn_padding_idx()));
      dst_dp.attn_unpadding_idx(
          slice_like(persistent_dp_ep_padding_.attn_unpadding_idx(),
                     src_dp.attn_unpadding_idx()));
      dst_dp.ffn_padding_idx(
          slice_like(persistent_dp_ep_padding_.ffn_padding_idx(),
                     src_dp.ffn_padding_idx()));
      dst_dp.ffn_unpadding_idx(
          slice_like(persistent_dp_ep_padding_.ffn_unpadding_idx(),
                     src_dp.ffn_unpadding_idx()));
      dst_dp.lm_head_skip_padding_token_indices(slice_like(
          persistent_dp_ep_padding_.lm_head_skip_padding_token_indices(),
          src_dp.lm_head_skip_padding_token_indices()));
      dst_dp.gather_prenorm_idx(
          slice_like(persistent_dp_ep_padding_.gather_prenorm_idx(),
                     src_dp.gather_prenorm_idx()));
      dst_dp.padding_idx(slice_like(persistent_dp_ep_padding_.padding_idx(),
                                    src_dp.padding_idx()));
      dst_dp.un_padding_idx(slice_like(
          persistent_dp_ep_padding_.un_padding_idx(), src_dp.un_padding_idx()));
      dst_dp.dynamic_ep_idx(slice_like(
          persistent_dp_ep_padding_.dynamic_ep_idx(), src_dp.dynamic_ep_idx()));
      dst_dp.moe_idx(
          slice_like(persistent_dp_ep_padding_.moe_idx(), src_dp.moe_idx()));
      dst_dp.expert_array(slice_like(persistent_dp_ep_padding_.expert_array(),
                                     src_dp.expert_array()));
      dst_dp.post_lmhead_gather_indices(
          slice_like(persistent_dp_ep_padding_.post_lmhead_gather_indices(),
                     src_dp.post_lmhead_gather_indices()));
    }

    const auto& src_cp = params.cp_ep_padding_data;
    if (src_cp.attn_padding_idx().defined() &&
        src_cp.attn_padding_idx().numel() > 0) {
      auto& dst_cp = params_for_capture->cp_ep_padding_data;
      dst_cp.attn_padding_idx(
          slice_like(persistent_cp_ep_padding_.attn_padding_idx(),
                     src_cp.attn_padding_idx()));
      dst_cp.attn_unpadding_idx(
          slice_like(persistent_cp_ep_padding_.attn_unpadding_idx(),
                     src_cp.attn_unpadding_idx()));
      dst_cp.ffn_padding_idx(
          slice_like(persistent_cp_ep_padding_.ffn_padding_idx(),
                     src_cp.ffn_padding_idx()));
      dst_cp.ffn_unpadding_idx(
          slice_like(persistent_cp_ep_padding_.ffn_unpadding_idx(),
                     src_cp.ffn_unpadding_idx()));
      dst_cp.lm_head_skip_padding_token_indices(slice_like(
          persistent_cp_ep_padding_.lm_head_skip_padding_token_indices(),
          src_cp.lm_head_skip_padding_token_indices()));
      dst_cp.gather_prenorm_idx(
          slice_like(persistent_cp_ep_padding_.gather_prenorm_idx(),
                     src_cp.gather_prenorm_idx()));
      dst_cp.padding_idx(slice_like(persistent_cp_ep_padding_.padding_idx(),
                                    src_cp.padding_idx()));
      dst_cp.un_padding_idx(slice_like(
          persistent_cp_ep_padding_.un_padding_idx(), src_cp.un_padding_idx()));
      dst_cp.dynamic_ep_idx(slice_like(
          persistent_cp_ep_padding_.dynamic_ep_idx(), src_cp.dynamic_ep_idx()));
      dst_cp.moe_idx(
          slice_like(persistent_cp_ep_padding_.moe_idx(), src_cp.moe_idx()));
      dst_cp.expert_array(slice_like(persistent_cp_ep_padding_.expert_array(),
                                     src_cp.expert_array()));
    }

    return params_for_capture;
  }
  return std::nullopt;
}

void GraphPersistentParam::initialize_paged_attention_plan_context(
    const torch::Device& device) {
  // max paged attention tiling buffer size is 1024 * 256
  constexpr int64_t tiling_buffer_size = 1024 * 256;
  tiling_data_ = torch::zeros({tiling_buffer_size},
                              torch::dtype(torch::kInt32).device(device));

  // Initialize ATB context for paged attention plan
  atb::Status status = atb::customize::CreatePlanContext(&context_for_plan_);
  CHECK_EQ(status, atb::NO_ERROR)
      << "Failed to create ATB context for paged attention plan";

  // Create stream for paged attention plan
  aclError acl_status = aclrtCreateStream(&stream_for_plan_);
  CHECK_EQ(acl_status, ACL_SUCCESS)
      << "Failed to create ACL stream for paged attention plan";
  context_for_plan_->SetExecuteStream(stream_for_plan_);

  // Set launch mode to GRAPH_LAUNCH_MODE
  status = context_for_plan_->SetLaunchMode(atb::LaunchMode::GRAPH_LAUNCH_MODE);
  CHECK_EQ(status, atb::NO_ERROR)
      << "Failed to set launch mode to GRAPH_LAUNCH_MODE";

  // Create custom paged attention operation
  const int dp_local_tp_size = options_.world_size() / options_.dp_size();

  // Cache headNum and head_dim as member variables
  num_head_ = static_cast<int32_t>(args_.n_heads() / dp_local_tp_size);
  head_dim_ = static_cast<int32_t>(args_.head_dim());

  atb::customize::CustomPagedAttentionParam paOpParam;
  // default mask type is UNDEFINED, which means no mask is needed
  if (need_update_attn_mask_) {
    paOpParam.maskType =
        atb::infer::PagedAttentionParam::MaskType::MASK_TYPE_NORM;
  }
  paOpParam.headNum = num_head_;

  std::optional<long int> optionalValue = args_.n_kv_heads();
  paOpParam.kvHeadNum =
      std::max(1,
               static_cast<int32_t>(optionalValue.value_or(args_.n_heads())) /
                   dp_local_tp_size);

  const float head_dim_float = static_cast<float>(head_dim_);
  paOpParam.qkScale = 1.0f / std::sqrt(head_dim_float);

  const bool isBF16 = args_.dtype() == "bfloat16";
  if (isBF16) {
    paOpParam.outDataType = ACL_BF16;
  } else {
    paOpParam.outDataType = ACL_FLOAT16;
  }

  status = atb::CreateOperation(paOpParam, &custom_pa_op_for_plan_);
  CHECK_EQ(status, atb::NO_ERROR)
      << "Failed to create custom paged attention operation";
  CHECK_NE(custom_pa_op_for_plan_, nullptr) << "custom_pa_op_for_plan_ is null";
}

constexpr uint32_t TILING_PARA_SIZE = 17;
constexpr uint32_t TILING_HEAD_SIZE = 44;

namespace {
void parse_pa_host_tiling_buffer(const uint32_t* hostTilingBuffer,
                                 uint64_t tilingBufferSize) {
  VLOG(kGraphExecutorLogVerboseLevel)
      << "hostTilingBuffer.tilingBuffer: " << (void*)hostTilingBuffer;
  VLOG(kGraphExecutorLogVerboseLevel)
      << "hostTilingBuffer.tilingBufferSize: " << tilingBufferSize;
  if (hostTilingBuffer == nullptr || tilingBufferSize == 0) {
    VLOG(kGraphExecutorLogVerboseLevel) << "Invalid host tiling buffer!";
    return;
  }

  uint32_t tilingParamSize = tilingBufferSize / sizeof(uint32_t);
  VLOG(kGraphExecutorLogVerboseLevel)
      << "Total tiling param elements: " << tilingParamSize;

  // Parse header fields (TILING_HEAD_SIZE = 44)
  VLOG(kGraphExecutorLogVerboseLevel) << "\n=== Tiling Header Fields ===";
  VLOG(kGraphExecutorLogVerboseLevel)
      << "TILING_BATCH(tiling_head[0]): " << hostTilingBuffer[0];
  VLOG(kGraphExecutorLogVerboseLevel)
      << "TILING_NUMHEADS(tiling_head[1]): " << hostTilingBuffer[1];
  VLOG(kGraphExecutorLogVerboseLevel)
      << "TILING_HEADDIM(tiling_head[2]): " << hostTilingBuffer[2];
  VLOG(kGraphExecutorLogVerboseLevel)
      << "TILING_NUMBLOKS(tiling_head[3]): " << hostTilingBuffer[3];
  VLOG(kGraphExecutorLogVerboseLevel)
      << "TILING_BLOCKSIZE(tiling_head[4]): " << hostTilingBuffer[4];
  VLOG(kGraphExecutorLogVerboseLevel)
      << "TILING_MAXBLOCKS(tiling_head[5]): " << hostTilingBuffer[5];
  VLOG(kGraphExecutorLogVerboseLevel)
      << "TILING_TOR(tiling_head[6]): " << hostTilingBuffer[6];
  VLOG(kGraphExecutorLogVerboseLevel)
      << "TILING_KVHEADS(tiling_head[7]): " << hostTilingBuffer[7];
  VLOG(kGraphExecutorLogVerboseLevel)
      << "TILING_FORMER_BATCH(tiling_head[8]): " << hostTilingBuffer[8];
  VLOG(kGraphExecutorLogVerboseLevel)
      << "TILING_FORMER_HEAD(tiling_head[9]): " << hostTilingBuffer[9];
  VLOG(kGraphExecutorLogVerboseLevel)
      << "TILING_TAIL_BATCH(tiling_head[10]): " << hostTilingBuffer[10];
  VLOG(kGraphExecutorLogVerboseLevel)
      << "TILING_TAIL_HEAD(tiling_head[11]): " << hostTilingBuffer[11];
  VLOG(kGraphExecutorLogVerboseLevel)
      << "TILING_HEADNUM_MOVE(tiling_head[12]): " << hostTilingBuffer[12];
  VLOG(kGraphExecutorLogVerboseLevel)
      << "TILING_MASK_MAX_LEN(tiling_head[13]): " << hostTilingBuffer[13];
  VLOG(kGraphExecutorLogVerboseLevel)
      << "TILING_BATCH_STRIDE(tiling_head[14]): " << hostTilingBuffer[14];
  VLOG(kGraphExecutorLogVerboseLevel)
      << "TILING_HEAD_STRIDE(tiling_head[15]): " << hostTilingBuffer[15];
  VLOG(kGraphExecutorLogVerboseLevel)
      << "TILING_KEY(tiling_head[16]): " << hostTilingBuffer[16];
  VLOG(kGraphExecutorLogVerboseLevel)
      << "TILING_HEADSIZE(tiling_head[17]): " << hostTilingBuffer[17];
  VLOG(kGraphExecutorLogVerboseLevel)
      << "TILING_PARASIZE(tiling_head[18]): " << hostTilingBuffer[18];
  VLOG(kGraphExecutorLogVerboseLevel)
      << "TILING_GROUPNUM(tiling_head[19]): " << hostTilingBuffer[19];
  VLOG(kGraphExecutorLogVerboseLevel)
      << "TILING_FORMER_GROUP_MOVE(tiling_head[20]): " << hostTilingBuffer[20];
  VLOG(kGraphExecutorLogVerboseLevel)
      << "TILING_TAIL_GROUP_MOVE(tiling_head[21]): " << hostTilingBuffer[21];
  VLOG(kGraphExecutorLogVerboseLevel)
      << "TILING_MAX_KVSEQLEN(tiling_head[22]): " << hostTilingBuffer[22];
  VLOG(kGraphExecutorLogVerboseLevel)
      << "TILING_KVSPLIT(tiling_head[23]): " << hostTilingBuffer[23];
  VLOG(kGraphExecutorLogVerboseLevel)
      << "TILING_KVCORENUM(tiling_head[24]): " << hostTilingBuffer[24];
  VLOG(kGraphExecutorLogVerboseLevel)
      << "TILING_BLOCKSIZE_CALC(tiling_head[25]): " << hostTilingBuffer[25];
  VLOG(kGraphExecutorLogVerboseLevel)
      << "TILING_TOTAL_BLOCK_NUM(tiling_head[26]): " << hostTilingBuffer[26];
  VLOG(kGraphExecutorLogVerboseLevel)
      << "TILING_PREFILL_BS(tiling_head[27]): " << hostTilingBuffer[27];
  VLOG(kGraphExecutorLogVerboseLevel)
      << "TILING_DECODER_BS(tiling_head[28]): " << hostTilingBuffer[28];
  VLOG(kGraphExecutorLogVerboseLevel)
      << "TILING_HEADDIM_V(tiling_head[29]): " << hostTilingBuffer[29];
  VLOG(kGraphExecutorLogVerboseLevel)
      << "TILING_MODCOEF(tiling_head[30]): " << hostTilingBuffer[30];
  VLOG(kGraphExecutorLogVerboseLevel)
      << "TILING_DIVCOEF(tiling_head[31]): " << hostTilingBuffer[31];
  VLOG(kGraphExecutorLogVerboseLevel)
      << "TILING_QHEADORIGINAL(tiling_head[32]): " << hostTilingBuffer[32];
  VLOG(kGraphExecutorLogVerboseLevel)
      << "TILING_COMPRESSHEAD(tiling_head[33]): " << hostTilingBuffer[33];
  VLOG(kGraphExecutorLogVerboseLevel)
      << "TILING_QUANTYPE(tiling_head[34]): " << hostTilingBuffer[34];
  VLOG(kGraphExecutorLogVerboseLevel)
      << "TILING_DATA_SHAPE_TYPE(tiling_head[35]): " << hostTilingBuffer[35];
  VLOG(kGraphExecutorLogVerboseLevel)
      << "TILING_SCALETYPE(tiling_head[36]): " << hostTilingBuffer[36];
  VLOG(kGraphExecutorLogVerboseLevel)
      << "TILING_MASK_TYPE_ND(tiling_head[37]): " << hostTilingBuffer[37];
  VLOG(kGraphExecutorLogVerboseLevel)
      << "TILING_HEADDIM_K_SPLIT(tiling_head[38]): " << hostTilingBuffer[38];
  VLOG(kGraphExecutorLogVerboseLevel)
      << "TILING_HEADDIM_V_SPLIT(tiling_head[39]): " << hostTilingBuffer[39];
  VLOG(kGraphExecutorLogVerboseLevel)
      << "TILING_HEADDIM_V_SPLIT_VECTOR_FORMER(tiling_head[40]): "
      << hostTilingBuffer[40];
  VLOG(kGraphExecutorLogVerboseLevel)
      << "TILING_HEADDIM_V_SPLIT_VECTOR_TAIL(tiling_head[41]): "
      << hostTilingBuffer[41];
  VLOG(kGraphExecutorLogVerboseLevel)
      << "TILING_MTP_HEAD_SPLIT_SIZE(tiling_head[42]): "
      << hostTilingBuffer[42];
  VLOG(kGraphExecutorLogVerboseLevel)
      << "TILING_MTP_HEAD_SPLIT_NUM(tiling_head[43]): " << hostTilingBuffer[43];

  // Parse batch parameters
  if (tilingParamSize > TILING_HEAD_SIZE) {
    uint32_t batchCount = hostTilingBuffer[0];
    VLOG(kGraphExecutorLogVerboseLevel) << "\n=== Batch Parameters ===";
    VLOG(kGraphExecutorLogVerboseLevel) << "Number of batches: " << batchCount;
    batchCount = std::min(batchCount, 20u);

    for (uint32_t batchIdx = 0; batchIdx < batchCount; ++batchIdx) {
      uint32_t offset = TILING_HEAD_SIZE + batchIdx * TILING_PARA_SIZE;
      if (offset + TILING_PARA_SIZE <= tilingParamSize) {
        VLOG(kGraphExecutorLogVerboseLevel)
            << "\n--- Batch " << batchIdx << " ---";
        VLOG(kGraphExecutorLogVerboseLevel)
            << "  qSeqLen(batch_tiling_param[0]): "
            << hostTilingBuffer[offset + 0];
        VLOG(kGraphExecutorLogVerboseLevel)
            << "  kvSeqLen(batch_tiling_param[1]): "
            << hostTilingBuffer[offset + 1];
        VLOG(kGraphExecutorLogVerboseLevel)
            << "  qSBlockTile(batch_tiling_param[2]): "
            << hostTilingBuffer[offset + 2];
        VLOG(kGraphExecutorLogVerboseLevel)
            << "  blockSize(batch_tiling_param[3]): "
            << hostTilingBuffer[offset + 3];
        VLOG(kGraphExecutorLogVerboseLevel)
            << "  addrQSeqOffset[high](batch_tiling_param[4]): "
            << hostTilingBuffer[offset + 4];
        VLOG(kGraphExecutorLogVerboseLevel)
            << "  addrQSeqOffset[low](batch_tiling_param[5]): "
            << hostTilingBuffer[offset + 5];
        VLOG(kGraphExecutorLogVerboseLevel)
            << "  addrOSeqOffset[high](batch_tiling_param[6]): "
            << hostTilingBuffer[offset + 6];
        VLOG(kGraphExecutorLogVerboseLevel)
            << "  addrOSeqOffset[low](batch_tiling_param[7]): "
            << hostTilingBuffer[offset + 7];
        VLOG(kGraphExecutorLogVerboseLevel)
            << "  seqIdx(batch_tiling_param[8]): "
            << hostTilingBuffer[offset + 8];
        VLOG(kGraphExecutorLogVerboseLevel)
            << "  totalQBlkNum(batch_tiling_param[9]): "
            << hostTilingBuffer[offset + 9];
        VLOG(kGraphExecutorLogVerboseLevel)
            << "  maskOffset[high](batch_tiling_param[10]): "
            << hostTilingBuffer[offset + 10];
        VLOG(kGraphExecutorLogVerboseLevel)
            << "  addrLSeqOffset[high](batch_tiling_param[11]): "
            << hostTilingBuffer[offset + 11];
        VLOG(kGraphExecutorLogVerboseLevel)
            << "  addrLSeqOffset[low](batch_tiling_param[12]): "
            << hostTilingBuffer[offset + 12];
        VLOG(kGraphExecutorLogVerboseLevel)
            << "  maskOffset[low](batch_tiling_param[14]): "
            << hostTilingBuffer[offset + 14];
        VLOG(kGraphExecutorLogVerboseLevel)
            << "  addrOFdSeqOffset[high](batch_tiling_param[15]): "
            << hostTilingBuffer[offset + 15];
        VLOG(kGraphExecutorLogVerboseLevel)
            << "  addrOFdSeqOffset[low](batch_tiling_param[16]): "
            << hostTilingBuffer[offset + 16];
      }
    }
  }

  VLOG(kGraphExecutorLogVerboseLevel) << "\n=== End of Tiling Buffer Parse ===";
}
}  // namespace

void GraphPersistentParam::plan_paged_attention_tiling(
    const torch::Tensor& tokens,
    const torch::Tensor& k_cache,
    const torch::Tensor& v_cache,
    const torch::Tensor& block_tables,
    const ModelInputParams& input_params,
    aclrtStream stream) {
  // Convert torch tensors to atb tensors
  atb::Tensor atb_k_cache = atb_speed::Utils::AtTensor2Tensor(k_cache);
  atb::Tensor atb_v_cache = atb_speed::Utils::AtTensor2Tensor(v_cache);
  atb::Tensor atb_block_tables =
      atb_speed::Utils::AtTensor2Tensor(block_tables);
  // Get context_lens from input_params.kv_seq_lens
  atb::Tensor atb_context_lens =
      atb_speed::Utils::AtTensor2Tensor(input_params.kv_seq_lens);
  atb_context_lens.hostData =
      const_cast<int32_t*>(input_params.kv_seq_lens_vec.data());
  atb::Tensor atb_tiling_data = atb_speed::Utils::AtTensor2Tensor(tiling_data_);

  atb_tiling_data.desc.dtype = ACL_UINT32;

  // Construct query atb tensor from tokens: shape [num_tokens, headNum,
  // head_dim] Get number of tokens from tokens tensor
  const int64_t num_tokens = tokens.size(0);

  // Create query atb tensor with only desc (no actual data needed)
  atb::Tensor atb_query;
  // TODO: support quant dtype
  atb_query.desc.dtype = (args_.dtype() == "bfloat16") ? ACL_BF16 : ACL_FLOAT16;
  atb_query.desc.format = ACL_FORMAT_ND;
  atb_query.desc.shape.dimNum = 3;
  atb_query.desc.shape.dims[0] = num_tokens;
  atb_query.desc.shape.dims[1] = num_head_;
  atb_query.desc.shape.dims[2] = head_dim_;
  atb_query.deviceData = atb_k_cache.deviceData;
  atb_query.hostData = nullptr;
  // Calculate dataSize: num_tokens * headNum * head_dim * sizeof(dtype)
  // ACL_FLOAT16 and ACL_BF16 both use 2 bytes per element
  const uint64_t element_size =
      (atb_query.desc.dtype == ACL_BF16 || atb_query.desc.dtype == ACL_FLOAT16)
          ? 2
          : 1;
  atb_query.dataSize = static_cast<uint64_t>(num_tokens) *
                       static_cast<uint64_t>(num_head_) *
                       static_cast<uint64_t>(head_dim_) * element_size;

  atb::VariantPack custom_variantPack;
  // Conditionally include mask based on need_update_attn_mask_
  if (need_update_attn_mask_) {
    atb::Tensor atb_mask = atb_speed::Utils::AtTensor2Tensor(persistent_mask_);
    custom_variantPack.inTensors = {atb_query,
                                    atb_k_cache,
                                    atb_v_cache,
                                    atb_block_tables,
                                    atb_context_lens,
                                    atb_mask,
                                    atb_tiling_data};
  } else {
    // Skip mask when not needed
    custom_variantPack.inTensors = {atb_query,
                                    atb_k_cache,
                                    atb_v_cache,
                                    atb_block_tables,
                                    atb_context_lens,
                                    atb_tiling_data};
  }
  custom_variantPack.outTensors.push_back(atb_query);

  uint64_t custom_workspace_size = 0;
  atb::Status status = custom_pa_op_for_plan_->Setup(
      custom_variantPack, custom_workspace_size, context_for_plan_);
  CHECK_EQ(status, atb::NO_ERROR)
      << "Failed to setup custom paged attention operation for tiling";

  atb::customize::TilingBufferInfo tiling_buffer_info =
      atb::customize::GetHostTilingBufferFromCustomPagedAttentionOperation(
          custom_pa_op_for_plan_);

  CHECK_NE(tiling_buffer_info.tilingBuffer, nullptr)
      << "Tiling buffer is null after setup";
  CHECK_GT(tiling_buffer_info.tilingBufferSize, 0)
      << "Tiling buffer size is zero";

  if (VLOG_IS_ON(kGraphExecutorLogVerboseLevel)) {
    parse_pa_host_tiling_buffer((uint32_t*)tiling_buffer_info.tilingBuffer,
                                tiling_buffer_info.tilingBufferSize);
  }
  aclError acl_status =
      aclrtMemcpyAsync(tiling_data_.data_ptr(),
                       tiling_data_.numel() * sizeof(uint32_t),
                       tiling_buffer_info.tilingBuffer,
                       tiling_buffer_info.tilingBufferSize,
                       ACL_MEMCPY_HOST_TO_DEVICE,
                       stream);
  CHECK_EQ(acl_status, ACL_SUCCESS) << "Failed to copy tiling buffer to device";
}

void GraphPersistentParam::update_attention_mask(
    const ModelInputParams& input_params) {
  torch::Dtype dtype = util::parse_dtype(args_.dtype(), device_);

  // update persistent_mask_ in-place
  const int64_t batch_size = input_params.kv_seq_lens.size(0);
  const int64_t max_seq_len = input_params.kv_max_seq_len > 0
                                  ? input_params.kv_max_seq_len
                                  : args_.max_position_embeddings();

  // persistent_mask_ is already initialized in constructor
  // Check if size is sufficient
  CHECK_LE(max_seq_len, persistent_mask_.size(1))
      << "max_seq_len (" << max_seq_len << ") exceeds max_seq_len ("
      << persistent_mask_.size(1) << ")";

  // Check if q_max_seq_len > 1 (prefill mode, not decode mode)
  bool chunked_prefill = input_params.q_max_seq_len > 1;

  // Calculate num_tokens: in chunked mode, sum of all q_len; in decode mode,
  // batch_size
  int64_t num_tokens = batch_size;  // Default for decode mode
  if (chunked_prefill) {
    CHECK_EQ(input_params.q_seq_lens_vec.size(), batch_size)
        << "q_seq_lens_vec size (" << input_params.q_seq_lens_vec.size()
        << ") != batch_size (" << batch_size << ")";
    num_tokens =
        std::accumulate(input_params.q_seq_lens_vec.begin(),
                        input_params.q_seq_lens_vec.begin() + batch_size,
                        int64_t(0));
  }

  // Check if num_tokens is within bounds
  CHECK_LE(num_tokens, persistent_mask_.size(0))
      << "num_tokens (" << num_tokens << ") exceeds max_tokens_per_batch ("
      << persistent_mask_.size(0) << ")";

  // Get slice for actual num_tokens (compatible with both chunked and
  // non-chunked)
  auto mask_slice =
      persistent_mask_.slice(/*dim=*/0, /*start=*/0, /*end=*/num_tokens)
          .slice(/*dim=*/1, /*start=*/0, /*end=*/max_seq_len);

  // Zero out the slice first
  mask_slice.zero_();

  const float mask_value = (dtype == torch::kFloat16)
                               ? -std::numeric_limits<float>::infinity()
                               : -9984.0f;

  if (chunked_prefill) {
    // Generate mask considering both q_seq_lens and kv_seq_lens
    // For each sequence, generate mask with shape [q_len, kv_len]
    // mask_slice is [num_tokens, max_seq_len], where num_tokens = sum of all
    // q_len

    // Check if kv_seq_lens_vec is available
    CHECK_EQ(input_params.kv_seq_lens_vec.size(), batch_size)
        << "kv_seq_lens_vec size (" << input_params.kv_seq_lens_vec.size()
        << ") != batch_size (" << batch_size << ")";

    int64_t offset = 0;
    for (int64_t i = 0; i < batch_size; i++) {
      const int32_t q_len = input_params.q_seq_lens_vec[i];
      const int32_t kv_len = input_params.kv_seq_lens_vec[i];

      // For chunked mode, slice out q_len rows for this sequence
      // mask_slice is [num_tokens, max_seq_len]
      // Get [q_len, kv_len] slice from mask_slice[offset:offset+q_len, :kv_len]
      auto seq_mask_slice =
          mask_slice.slice(/*dim=*/0, /*start=*/offset, /*end=*/offset + q_len)
              .slice(
                  /*dim=*/1, /*start=*/0, /*end=*/kv_len);  // [q_len, kv_len]

      // Zero out the slice first
      seq_mask_slice.zero_();

      // Generate mask for this sequence: [q_len, kv_len]
      // Use tril to generate lower triangular mask
      int diagonal = kv_len - q_len;
      auto options = torch::TensorOptions().dtype(torch::kBool).device(device_);
      auto bias = torch::tril(torch::ones({q_len, kv_len}, options), diagonal);
      bias = ~bias;  // Invert: True positions need to be masked

      // Fill mask values directly
      seq_mask_slice.masked_fill_(bias, mask_value);

      // Update offset for next sequence
      offset += q_len;
    }
  } else {
    // Original logic: only consider kv_seq_lens (decode mode, q_len = 1 for
    // all)
    auto positions = torch::arange(max_seq_len, torch::kInt32)
                         .to(device_)
                         .unsqueeze(0)
                         .expand({batch_size, max_seq_len});

    auto context_lens_expanded = input_params.kv_seq_lens.to(torch::kInt32)
                                     .unsqueeze(1)
                                     .expand({batch_size, max_seq_len});

    auto mask_condition = positions >= context_lens_expanded;
    mask_slice.masked_fill_(mask_condition, mask_value);
  }
}

bool AclGraph::capture(CausalLM* model,
                       const ModelArgs& args,
                       const runtime::Options& options,
                       const torch::Tensor& tokens,
                       const torch::Tensor& positions,
                       const ModelInputParams& params,
                       std::vector<KVCache>& kv_cache,
                       uint32_t bucket_num_tokens) {
  // Save bucket num_tokens for this graph instance
  num_tokens_ = bucket_num_tokens;

  // Get actual num_tokens from tokens tensor
  // const uint32_t actual_num_tokens = tokens.size(0);

  auto& tensor_options = model->options();

  torch::npu::synchronize();

  // Begin graph capture using NPUGraph mempool for temporary tensor management
  // Get current NPU stream from libtorch NPU API
  aclrtStream stream =
      c10_npu::getCurrentNPUStream(tensor_options.device().index()).stream();

  // For hybrid models (e.g., qwen3_next with mixed GDN/full_attention layers),
  // we need to find the first Full Attention layer to get the correct kv_cache.
  // GDN layers have empty key_cache_/value_cache_ while Full Attention layers
  // have valid kv caches. Using layer 0's cache directly would be incorrect
  // if layer 0 is a GDN layer.
  auto [k_cache, v_cache] = find_attention_plan_kv_cache(kv_cache);
  const uint32_t actual_num_tokens = tokens.size(0);
  CHECK_GE(num_tokens_, actual_num_tokens)
      << "num_tokens_ >= actual_num_tokens";
  auto graph_params = persistent_param_.update(tokens,
                                               k_cache,
                                               v_cache,
                                               positions,
                                               params,
                                               num_tokens_,
                                               /*return_capture_params=*/true);

  // Use the returned ModelInputParams for graph capture
  CHECK(graph_params.has_value())
      << "update() should return ModelInputParams when "
         "return_capture_params=true";

  // Synchronize stream to ensure all data is copied to graph persistent buffers
  aclrtSynchronizeStream(stream);

  // Acquire device-level lock to prevent prepare_work_before_execute from
  // executing simultaneously, which would trigger synchronous operations
  // that conflict with capture mode
  auto device_idx = tensor_options.device().index();

  // Use cached capture stream for graph capture
  // capture_stream_ is initialized in constructor
  bool need_restore_stream = false;

  // capture lock scope
  {
    auto& capture_lock =
        ::xllm::npu::DeviceCaptureLock::get_instance().get_lock(device_idx);
    std::lock_guard<std::mutex> lock_guard(capture_lock);

    if (c10_npu::getCurrentNPUStream(device_idx) ==
        c10_npu::getDefaultNPUStream(device_idx)) {
      c10_npu::setCurrentNPUStream(capture_stream_.value());
      aclrtSynchronizeStream(capture_stream_.value().stream());
      need_restore_stream = true;
    }
    LOG(INFO) << "capture begin, bucket_num_tokens: " << bucket_num_tokens
              << ", actual_num_tokens: " << actual_num_tokens;

    // no mempool id, will create a new one; capture mode is thread local, allow
    // other threads to execute synchronous operations
    graph_.capture_begin(
        {0, 0}, aclmdlRICaptureMode::ACL_MODEL_RI_CAPTURE_MODE_THREAD_LOCAL);
    // Execute forward pass - NPUGraph mempool manages temporary tensors
    auto forward_result =
        model->forward({persistent_param_.persistent_tokens(num_tokens_)},
                       {persistent_param_.persistent_positions(num_tokens_)},
                       kv_cache,
                       {graph_params.value()});

    // Store result in persistent buffer owned by NPUGraph mempool
    persistent_param_.set_hidden_states(forward_result.hidden_states);
    if (options.enable_graph_aux_hidden_states() &&
        forward_result.aux_hidden_states.defined()) {
      persistent_param_.set_aux_hidden_states(forward_result.aux_hidden_states);
    }
    graph_.capture_end();
    // Lock is automatically released here when lock goes out of scope
    if (need_restore_stream) {
      c10_npu::setCurrentNPUStream(
          c10_npu::getDefaultNPUStream(tensor_options.device().index()));
    }
  }
  // Synchronize and test replay to verify graph capture
  aclrtSynchronizeStream(stream);

  graph_.replay();

  // aclrtSynchronizeStream(stream);
  return true;
}

void AclGraph::initialize_capture_stream(c10::DeviceIndex device_index) {
  // Get a secondary stream from high-priority pool for graph capture.
  // This is required because NPUGraph::capture_begin() enforces that capture
  // must be performed on a non-default stream (see
  // torch_npu/csrc/core/npu/NPUGraph.cpp:159).
  capture_stream_ = c10_npu::getStreamFromPool(true, device_index);
  device_index_ = device_index;
  LOG(INFO) << "Initialized capture_stream: " << capture_stream_.value()
            << ", id: " << capture_stream_.value().id()
            << ", device_index: " << device_index;
}

ModelOutput AclGraph::replay(const torch::Tensor& tokens,
                             const torch::Tensor& positions,
                             std::vector<KVCache>& kv_cache,
                             const ModelInputParams& params) {
  const uint32_t actual_num_tokens = tokens.size(0);
  CHECK_LE(actual_num_tokens, num_tokens_)
      << "num_tokens mismatch: expected <= " << num_tokens_ << ", got "
      << actual_num_tokens;

  // Update persistent parameters with new input data
  // Note: tiling_data is updated in update() if needed - for hybrid models
  // (e.g., qwen3_next with mixed GDN/attention layers), tiling should only
  // be updated when Full Attention layers are involved, which is determined
  // by k_cache being valid and non-empty
  auto [k_cache, v_cache] = find_attention_plan_kv_cache(kv_cache);
  persistent_param_.update(tokens,
                           k_cache,
                           v_cache,
                           positions,
                           params,
                           num_tokens_,
                           /*return_capture_params=*/false);

  // Replay captured graph - NPUGraph mempool reuses temporary tensors
  // Get current NPU stream from libtorch NPU API
  aclrtStream stream = c10_npu::getCurrentNPUStream().stream();

  graph_.replay();

  // this is necessary to ensure the graph replay is completed
  // aclError st = aclrtSynchronizeStream(stream);
  // CHECK_EQ(st, ACL_SUCCESS)
  // << "aclrtSynchronizeStream failed, error code: " << st;

  // Return the actual num_tokens portion of ModelOutput
  // Note: aux_hidden_states handling is done in AclGraphExecutorImpl::run()
  // since replay() doesn't have access to options
  return ModelOutput(get_hidden_states(actual_num_tokens));
}

AclGraphExecutorImpl::AclGraphExecutorImpl(CausalLM* model,
                                           const ModelArgs& args,
                                           const torch::Device& device,
                                           const runtime::Options& options)
    : model_(model), args_(args), device_(device), options_(options) {
  // Create single persistent parameter object shared by all AclGraph instances
  persistent_param_ =
      std::make_unique<GraphPersistentParam>(args_, device_, options_);
}

AclGraphExecutorImpl::~AclGraphExecutorImpl() = default;

ForwardInput AclGraphExecutorImpl::prepare_inputs(Batch& batch) {
  // Prepare inputs for workers
  return batch.prepare_forward_input(
      options_.num_decoding_tokens(), 0, args_, options_.cp_size());
}

ModelOutput AclGraphExecutorImpl::run_onerec_graph(
    const torch::Tensor& tokens,
    const torch::Tensor& positions,
    std::vector<KVCache>& kv_caches,
    const ModelInputParams& params) {
  const auto* onerec_params = params.onerec_params();
  CHECK(onerec_params != nullptr);
  const bool is_prefill =
      onerec_params->rec_stage == OneRecModelInputParams::RecStage::PREFILL;
  const char* stage_name = is_prefill ? "prefill" : "decode";
  const int64_t output_tokens =
      get_onerec_decoder_output_tokens(tokens, params, args_);
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
      onerec_last_first_prefill_graph_ready_ = false;
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
        onerec_last_first_prefill_graph_ready_ = false;
      }
      COUNTER_INC(num_model_execution_total_eager);
      return model_->forward(tokens, positions, kv_caches, params);
    }
  }

  const std::string graph_key =
      build_onerec_graph_key(tokens, positions, params, args_);
  std::unique_lock<std::mutex> graph_lock(onerec_graph_mutex_);
  if (is_prefill && !onerec_params->is_first_prefill &&
      !onerec_last_first_prefill_graph_ready_) {
    graph_lock.unlock();
    COUNTER_INC(num_model_execution_total_eager);
    return model_->forward(tokens, positions, kv_caches, params);
  }
  auto it = onerec_graphs_.find(graph_key);
  if (it != onerec_graphs_.end()) {
    VLOG(kGraphExecutorLogVerboseLevel)
        << "AclGraphExecutorImpl::run() in OneRec " << stage_name
        << " replay mode";
    if (is_prefill && onerec_params->is_first_prefill) {
      onerec_last_first_prefill_graph_ready_ = true;
    }
    return it->second->replay(
        model_, tokens, positions, params, encoder_output);
  }

  auto graph = std::make_unique<OneRecAclGraph>(args_, device_, options_);
  VLOG(kGraphExecutorLogVerboseLevel)
      << "AclGraphExecutorImpl::run() in OneRec " << stage_name
      << " capture mode";
  const bool capture_success = graph->capture(
      model_, tokens, positions, kv_caches, params, encoder_output);
  if (capture_success) {
    LOG(INFO) << "Lazy capturing OneRec " << stage_name
              << " ACL graph done, key length: " << graph_key.size()
              << ", output_tokens: " << output_tokens
              << ", is_first_prefill: " << onerec_params->is_first_prefill;
    auto result = graph->output();
    onerec_graphs_[graph_key] = std::move(graph);
    if (is_prefill && onerec_params->is_first_prefill) {
      onerec_last_first_prefill_graph_ready_ = true;
    }
    return result;
  }

  LOG(FATAL) << "Failed to capture OneRec " << stage_name
             << " ACL graph, output_tokens: " << output_tokens;
  return ModelOutput();
}

// Main execution method with graph optimization for decode phase
// tokens: [num_decode_tokens]
// positions: [num_decode_tokens] token pos in the sequence
// returns: [num_decode_tokens, hidden_size]
ModelOutput AclGraphExecutorImpl::run(const torch::Tensor& tokens,
                                      const torch::Tensor& positions,
                                      std::vector<KVCache>& kv_caches,
                                      const ModelInputParams& params) {
  // no mirco batch in decode phase
  const torch::Tensor& tokens_tensor = tokens;
  const torch::Tensor& positions_tensor = positions;
  const ModelInputParams& params_single = params;
  const bool in_decoding_phase = params_single.batch_forward_type.is_decode();
  VLOG(50) << "in_decoding_phase: " << in_decoding_phase
           << " q_max_seq_len: " << params_single.q_max_seq_len
           << " n_layers: " << args_.n_layers();

  if (is_onerec_decoder_graph_candidate(model_, params_single)) {
    return run_onerec_graph(
        tokens_tensor, positions_tensor, kv_caches, params_single);
  }
  if (const auto* onerec_params = params_single.onerec_params();
      onerec_params != nullptr &&
      onerec_params->rec_stage == OneRecModelInputParams::RecStage::PREFILL &&
      !onerec_params->is_encoder_forward && onerec_params->is_first_prefill) {
    onerec_last_first_prefill_graph_ready_ = false;
  }

  // If not in decode phase, use eager mode directly without acl graph
  // TODO: fix mtp model support.
  if (!in_decoding_phase || args_.n_layers() == 1) {
    VLOG(kGraphExecutorLogVerboseLevel)
        << "AclGraphExecutorImpl::run() in eager mode";
    COUNTER_INC(num_model_execution_total_eager);
    return model_->forward(tokens, positions, kv_caches, params);
  }

  // Only use acl graph in decode phase for performance optimization
  // For DP, decode graph bucket should be based on global max tokens across dp
  // groups; local shard can be empty on some ranks.
  uint32_t graph_num_tokens = tokens_tensor.size(/*dim=*/0);
  if (params_single.dp_global_token_nums.size() > 1) {
    graph_num_tokens = util::max(params_single.dp_global_token_nums);
  }
  // Keep actual n_tokens for replay output slicing.
  const uint32_t n_tokens = tokens_tensor.size(/*dim=*/0);
  const uint32_t local_batch_size = n_tokens / options_.num_decoding_tokens();
  const uint32_t global_batch_size =
      graph_num_tokens / options_.num_decoding_tokens();

  // Large decode batches create too many/too large ACL graphs and may OOM.
  // Fall back to eager mode when batch size exceeds the safety threshold.
  // Use global_batch_size so all DP ranks make the same decision and stay in
  // sync on HCCL collectives.
  const uint32_t decode_batch_size_limit =
      std::max(1, FLAGS_acl_graph_decode_batch_size_limit);
  if (global_batch_size > decode_batch_size_limit) {
    LOG_FIRST_N(WARNING, 1)
        << "Falling back to eager mode because decode batch_size (global="
        << global_batch_size << ", local=" << local_batch_size << ") > "
        << decode_batch_size_limit
        << "; ACL graph is disabled for this request size to avoid OOM. "
        << "This message is logged only once. "
        << "Monitor counter 'num_model_execution_total_eager' for frequency.";
    COUNTER_INC(num_model_execution_total_eager);
    return model_->forward(tokens, positions, kv_caches, params);
  }

  const uint32_t bucket_num_tokens = get_bucket_num_tokens(graph_num_tokens);

  // Check if conditions are suitable for graph execution (replay or capture)
  const auto max_seq_len = args_.max_position_embeddings();
  const bool seq_len_supported = params_single.kv_max_seq_len <= max_seq_len;

  // Combined condition for graph capture support
  // ACL graph executor only supports single tensor inputs (no micro-batching)
  const bool capture_supported = seq_len_supported;

  // Early return if conditions are not suitable for graph operations
  if (!capture_supported) {
    LOG_FIRST_N(WARNING, 1)
        << "Falling back to eager mode because kv_max_seq_len ("
        << params_single.kv_max_seq_len << ") > max_seq_len (" << max_seq_len
        << "). This message is logged only once. "
        << "Monitor counter 'num_model_execution_total_eager' for frequency.";
    COUNTER_INC(num_model_execution_total_eager);
    return model_->forward(tokens, positions, kv_caches, params);
  }

  // Check if captured graph exists for this bucket num_tokens
  auto it = graphs_.find(bucket_num_tokens);
  if (it != graphs_.end()) {
    // Replay the existing graph
    VLOG(kGraphExecutorLogVerboseLevel)
        << "AclGraphExecutorImpl::run() in replay mode";
    auto result = it->second->replay(
        tokens_tensor, positions_tensor, kv_caches, params_single);
    // Handle aux_hidden_states based on options
    if (options_.enable_graph_aux_hidden_states()) {
      auto aux_hidden_states = persistent_param_->aux_hidden_states(n_tokens);
      if (aux_hidden_states.defined() && aux_hidden_states.numel() > 0) {
        return ModelOutput(
            result.hidden_states, torch::Tensor(), aux_hidden_states);
      }
    }
    return result;
  }

  // Graph doesn't exist for this bucket num_tokens, try to create it lazily
  auto graph = std::make_unique<AclGraph>(*persistent_param_, device_.index());
  VLOG(kGraphExecutorLogVerboseLevel)
      << "AclGraphExecutorImpl::run() in capture mode";
  bool capture_success = graph->capture(model_,
                                        args_,
                                        options_,
                                        tokens_tensor,
                                        positions_tensor,
                                        params_single,
                                        kv_caches,
                                        bucket_num_tokens);

  if (capture_success) {
    LOG(INFO) << "Lazy capturing ACL graph for bucket num_tokens: "
              << bucket_num_tokens << " (actual num_tokens: " << n_tokens
              << ") done";

    // Save the graph for future reuse
    graphs_[bucket_num_tokens] = std::move(graph);

    // Return the output from capture (no need to replay since capture
    // already executed)
    auto hidden_states =
        graphs_[bucket_num_tokens]->get_hidden_states(n_tokens);
    if (options_.enable_graph_aux_hidden_states()) {
      auto aux_hidden_states = persistent_param_->aux_hidden_states(n_tokens);
      if (aux_hidden_states.defined() && aux_hidden_states.numel() > 0) {
        return ModelOutput(hidden_states, torch::Tensor(), aux_hidden_states);
      }
    }
    return ModelOutput(hidden_states);
  }

  // Fallback to eager mode if capture fails
  LOG(ERROR) << "Failed to capture ACL graph for bucket num_tokens: "
             << bucket_num_tokens;
  COUNTER_INC(num_model_execution_total_eager);
  return model_->forward(tokens, positions, kv_caches, params);
}

void AclGraph::print_graph_tensors() const {
  VLOG(kGraphExecutorLogVerboseLevel)
      << "graph persistent_tokens_: " << persistent_param_.persistent_tokens();
  VLOG(kGraphExecutorLogVerboseLevel)
      << "graph persistent_positions_: "
      << persistent_param_.persistent_positions();
  VLOG(kGraphExecutorLogVerboseLevel)
      << "graph persistent_new_cache_slots_: "
      << persistent_param_.persistent_new_cache_slots();
  VLOG(kGraphExecutorLogVerboseLevel)
      << "graph q_seq_lens_: " << persistent_param_.q_seq_lens();
  VLOG(kGraphExecutorLogVerboseLevel)
      << "graph kv_seq_lens_: " << persistent_param_.kv_seq_lens();
  VLOG(kGraphExecutorLogVerboseLevel)
      << "graph persistent_block_tables_: "
      << persistent_param_.persistent_block_tables();
  VLOG(kGraphExecutorLogVerboseLevel)
      << "graph hidden_states_: " << persistent_param_.hidden_states();
}

// bucket will be [1, 2, 4, 8, 16, 32, 48, 64, ..., max_seqs_per_batch]
uint32_t AclGraphExecutorImpl::get_bucket_num_tokens(
    uint32_t num_tokens) const {
  if (FLAGS_enable_graph_mode_decode_no_padding) {
    return num_tokens;
  }
  if (num_tokens <= 1) {
    return 1;
  } else if (num_tokens <= 2) {
    return 2;
  } else if (num_tokens <= 4) {
    return 4;
  } else if (num_tokens <= 8) {
    return 8;
  } else {
    // For num_tokens > 16, use multiples of 16
    return ((num_tokens + 15) / 16) * 16;
  }
}

}  // namespace xllm::npu

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

#include "core/runtime/acl_graph_persistent_param.h"

#include <glog/logging.h>
#include <torch/torch.h>
#include <torch_npu/torch_npu.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>
#include <utility>
#include <vector>

#include "core/common/constants.h"
#include "core/common/global_flags.h"
#include "core/framework/config/speculative_config.h"
#include "core/util/utils.h"

// ATB includes
#include <atb/atb_infer.h>
#include <atb/context.h>
#include <atb/operation.h>
#include <customize/custom_paged_attention_function.h>
#include <customize/customize_op_params.h>

#include "pytorch/adapter/utils/utils.h"

namespace xllm::npu {

namespace {

int64_t get_decode_graph_capacity(const runtime::Options& options) {
  CHECK_GT(options.num_decoding_tokens(), 0)
      << "num_decoding_tokens must be > 0 for graph capacity";
  return options.max_seqs_per_batch();
}

int64_t get_decode_graph_token_capacity(const runtime::Options& options) {
  CHECK_GT(options.num_decoding_tokens(), 0)
      << "num_decoding_tokens must be > 0 for graph token capacity";
  if (::xllm::SpeculativeConfig::get_instance().enable_atb_spec_kernel()) {
    return options.max_seqs_per_batch();
  }
  if (options.enable_speculative_decode() && !options.is_draft_engine()) {
    return options.max_seqs_per_batch() * options.num_decoding_tokens();
  }
  return options.max_seqs_per_batch();
}

float get_dp_ep_all2all_buffer_factor(int64_t length) {
  const std::vector<std::pair<int64_t, float>> length_thresholds = {
      {1048576, 1.32f},
      {524288, 1.4f},
      {262144, 1.53f},
      {131072, 1.8f},
      {32768, 3.0f},
      {8192, 5.2f},
      {0, 8.0f}};

  for (const auto& threshold : length_thresholds) {
    if (length >= threshold.first) {
      return threshold.second;
    }
  }
  return 8.0f;
}

int64_t get_dp_ep_padding_buffer_capacity(const ModelArgs& args,
                                          const runtime::Options& options) {
  const int64_t dp_size = std::max<int64_t>(options.dp_size(), 1);
  const int64_t graph_capacity = get_decode_graph_token_capacity(options);
  const int64_t topk = std::max<int64_t>(args.num_experts_per_tok(), 1);
  const int64_t base_length = graph_capacity * topk;
  const int64_t global_length = base_length * dp_size;
  const float buffer_factor = get_dp_ep_all2all_buffer_factor(global_length);
  const int64_t moe_buffer_capacity =
      static_cast<int64_t>(std::ceil(base_length * buffer_factor));
  const int64_t aligned_moe_buffer_capacity =
      ((moe_buffer_capacity + dp_size - 1) / dp_size) * dp_size;
  return std::max(graph_capacity * dp_size, aligned_moe_buffer_capacity);
}

int64_t infer_actual_batch_size(const ModelInputParams& params) {
  if (params.meta.num_sequences == 0 && params.meta.actual_num_sequences == 0) {
    return 0;
  }
  if (params.meta.actual_num_sequences > 0) {
    return params.meta.actual_num_sequences;
  }
  if (params.meta.num_sequences > 0) {
    return params.meta.num_sequences;
  }
  if (!params.attention.host.kv_seq_lens.empty()) {
    return static_cast<int64_t>(params.attention.host.kv_seq_lens.size());
  }
  if (!params.attention.host.q_seq_lens.empty()) {
    return static_cast<int64_t>(params.attention.host.q_seq_lens.size());
  }
  if (params.attention.device.kv_seq_lens.defined() &&
      params.attention.device.kv_seq_lens.dim() >= 1) {
    return params.attention.device.kv_seq_lens.size(0);
  }
  if (params.attention.device.q_seq_lens.defined() &&
      params.attention.device.q_seq_lens.dim() >= 1) {
    return params.attention.device.q_seq_lens.size(0);
  }
  if (params.attention.device.block_tables.defined() &&
      params.attention.device.block_tables.dim() >= 2) {
    return params.attention.device.block_tables.size(0);
  }
  for (const auto& block_table : params.multi_block_tables) {
    if (block_table.defined() && block_table.dim() >= 2) {
      return block_table.size(0);
    }
  }
  return 0;
}

}  // namespace

// GraphPersistentParam implementation
GraphPersistentParam::GraphPersistentParam(const ModelArgs& args,
                                           const torch::Device& device,
                                           const runtime::Options& options,
                                           bool need_update_attn_mask,
                                           bool is_hybrid_linear_attention)
    : args_(args),
      device_(device),
      options_(options),
      context_for_plan_(nullptr),
      custom_pa_op_for_plan_(nullptr),
      stream_for_plan_(nullptr),
      need_update_attn_mask_(need_update_attn_mask),
      is_hybrid_linear_attention_(is_hybrid_linear_attention) {
  // Determine whether attention plan needs to be updated based on model type
  // Future logic can be extended here for more complex model-specific behavior
  need_update_attention_plan_ = (args.model_type() != "deepseek_v32" &&
                                 args.model_type() != "deepseek_v4" &&
                                 args.model_type() != "glm_moe_dsa");

  // Check if mRoPE is used (for VLM models like qwen2-vl)
  use_mrope_ = !args.rope_scaling_mrope_section().empty();

  const int64_t max_tokens_per_batch = options.max_tokens_per_batch();
  // Graph-mode token capacity is narrower than max_tokens_per_batch: ACL graph
  // only serves decode / spec-verify batches, so the relevant row upper bound
  // comes from decode graph capacity instead.
  const int64_t max_graph_tokens = get_decode_graph_token_capacity(options);
  // Hybrid spec verify keeps sequence-scoped metadata, while non-hybrid MTP
  // expands every proposal token into its own decode metadata row.
  const int64_t metadata_capacity = is_hybrid_linear_attention
                                        ? get_decode_graph_capacity(options)
                                        : max_graph_tokens;

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
  persistent_new_cache_slots_default_ = torch::zeros(
      {max_tokens_per_batch}, torch::dtype(torch::kInt).device(device));
  persistent_linear_state_indices_ = torch::zeros(
      {metadata_capacity}, torch::dtype(torch::kInt).device(device));
  persistent_num_accepted_tokens_ = torch::ones(
      {metadata_capacity}, torch::dtype(torch::kInt).device(device));

  q_seq_lens_ = torch::zeros({metadata_capacity},
                             torch::dtype(torch::kInt).device(device));
  kv_seq_lens_ = torch::zeros({metadata_capacity},
                              torch::dtype(torch::kInt).device(device));
  q_seq_lens_default_ = torch::ones({metadata_capacity},
                                    torch::dtype(torch::kInt).device(device));
  kv_seq_lens_default_ = torch::ones({metadata_capacity},
                                     torch::dtype(torch::kInt).device(device));
  expanded_kv_seq_lens_ = torch::zeros(
      {max_tokens_per_batch}, torch::dtype(torch::kInt).device(device));

  // Block table tensors with maximum possible size
  const int64_t block_size = options.block_size();
  const int64_t max_block_table_len =
      (max_seq_len + block_size - 1) / block_size + 1;
  persistent_block_tables_ =
      torch::zeros({metadata_capacity, max_block_table_len},
                   torch::dtype(torch::kInt).device(device));
  persistent_block_tables_default_ =
      torch::zeros({metadata_capacity, max_block_table_len},
                   torch::dtype(torch::kInt).device(device));
  persistent_expanded_block_tables_ =
      torch::zeros({max_tokens_per_batch, max_block_table_len},
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

  // Initialize persistent_mask_ only for model types that need to update an
  // explicit attention mask in graph mode. Unlike generic token buffers, the
  // mask is only consumed by decode / spec-verify graphs, so size it by graph
  // token capacity instead of the much larger max_tokens_per_batch prefill
  // budget.
  if (need_update_attn_mask_) {
    persistent_mask_ = torch::zeros({max_graph_tokens, max_seq_len},
                                    torch::dtype(dtype).device(device));
    persistent_mask_zero_template_ = torch::zeros(
        {max_graph_tokens, max_seq_len}, torch::dtype(dtype).device(device));
    const float mask_fill_value = (dtype == torch::kFloat16)
                                      ? -std::numeric_limits<float>::infinity()
                                      : -9984.0f;
    persistent_mask_fill_template_ =
        torch::full({max_graph_tokens, max_seq_len},
                    mask_fill_value,
                    torch::dtype(dtype).device(device));
  }

  q_cu_seq_lens_default_ = torch::zeros(
      {metadata_capacity + 1}, torch::dtype(torch::kInt).device(device));
  q_cu_seq_lens_ = torch::zeros({metadata_capacity + 1},
                                torch::dtype(torch::kInt).device(device));

  // Pre-allocate persistent dp/cp ep padding buffers with maximum capacity.
  const int64_t padding_buf_capacity =
      get_dp_ep_padding_buffer_capacity(args, options);
  torch::TensorOptions int_opts = torch::dtype(torch::kInt32).device(device);

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

  if (!need_update_attention_plan_) {
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

torch::Tensor slice_like_source(const torch::Tensor& persistent,
                                const torch::Tensor& src) {
  if (!src.defined() || src.numel() == 0 || !persistent.defined()) {
    return persistent;
  }
  return persistent.slice(/*dim=*/0, /*start=*/0, /*end=*/src.size(0));
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

void GraphPersistentParam::replace_capture_dp_ep_padding(
    const DpEpPaddingData& src,
    DpEpPaddingData& dst) const {
  if (!src.attn_padding_idx().defined() ||
      src.attn_padding_idx().numel() == 0) {
    return;
  }
  dst.attn_padding_idx(slice_like_source(
      persistent_dp_ep_padding_.attn_padding_idx(), src.attn_padding_idx()));
  dst.attn_unpadding_idx(
      slice_like_source(persistent_dp_ep_padding_.attn_unpadding_idx(),
                        src.attn_unpadding_idx()));
  dst.ffn_padding_idx(slice_like_source(
      persistent_dp_ep_padding_.ffn_padding_idx(), src.ffn_padding_idx()));
  dst.ffn_unpadding_idx(slice_like_source(
      persistent_dp_ep_padding_.ffn_unpadding_idx(), src.ffn_unpadding_idx()));
  dst.lm_head_skip_padding_token_indices(slice_like_source(
      persistent_dp_ep_padding_.lm_head_skip_padding_token_indices(),
      src.lm_head_skip_padding_token_indices()));
  dst.gather_prenorm_idx(
      slice_like_source(persistent_dp_ep_padding_.gather_prenorm_idx(),
                        src.gather_prenorm_idx()));
  dst.padding_idx(slice_like_source(persistent_dp_ep_padding_.padding_idx(),
                                    src.padding_idx()));
  dst.un_padding_idx(slice_like_source(
      persistent_dp_ep_padding_.un_padding_idx(), src.un_padding_idx()));
  dst.dynamic_ep_idx(slice_like_source(
      persistent_dp_ep_padding_.dynamic_ep_idx(), src.dynamic_ep_idx()));
  dst.moe_idx(
      slice_like_source(persistent_dp_ep_padding_.moe_idx(), src.moe_idx()));
  dst.expert_array(slice_like_source(persistent_dp_ep_padding_.expert_array(),
                                     src.expert_array()));
  dst.post_lmhead_gather_indices(
      slice_like_source(persistent_dp_ep_padding_.post_lmhead_gather_indices(),
                        src.post_lmhead_gather_indices()));
}

void GraphPersistentParam::replace_capture_cp_ep_padding(
    const CpEpPaddingData& src,
    CpEpPaddingData& dst) const {
  if (!src.attn_padding_idx().defined() ||
      src.attn_padding_idx().numel() == 0) {
    return;
  }
  dst.attn_padding_idx(slice_like_source(
      persistent_cp_ep_padding_.attn_padding_idx(), src.attn_padding_idx()));
  dst.attn_unpadding_idx(
      slice_like_source(persistent_cp_ep_padding_.attn_unpadding_idx(),
                        src.attn_unpadding_idx()));
  dst.ffn_padding_idx(slice_like_source(
      persistent_cp_ep_padding_.ffn_padding_idx(), src.ffn_padding_idx()));
  dst.ffn_unpadding_idx(slice_like_source(
      persistent_cp_ep_padding_.ffn_unpadding_idx(), src.ffn_unpadding_idx()));
  dst.lm_head_skip_padding_token_indices(slice_like_source(
      persistent_cp_ep_padding_.lm_head_skip_padding_token_indices(),
      src.lm_head_skip_padding_token_indices()));
  dst.gather_prenorm_idx(
      slice_like_source(persistent_cp_ep_padding_.gather_prenorm_idx(),
                        src.gather_prenorm_idx()));
  dst.padding_idx(slice_like_source(persistent_cp_ep_padding_.padding_idx(),
                                    src.padding_idx()));
  dst.un_padding_idx(slice_like_source(
      persistent_cp_ep_padding_.un_padding_idx(), src.un_padding_idx()));
  dst.dynamic_ep_idx(slice_like_source(
      persistent_cp_ep_padding_.dynamic_ep_idx(), src.dynamic_ep_idx()));
  dst.moe_idx(
      slice_like_source(persistent_cp_ep_padding_.moe_idx(), src.moe_idx()));
  dst.expert_array(slice_like_source(persistent_cp_ep_padding_.expert_array(),
                                     src.expert_array()));
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
    const int64_t graph_token_capacity =
        get_decode_graph_token_capacity(options_);
    auto shape = value.sizes().vec();
    shape[0] = graph_token_capacity;
    torch::Dtype dtype = util::parse_dtype(args_.dtype(), device_);
    if (args_.dtype() == "float" || args_.dtype() == "float32") {
      dtype = torch::kFloat32;
    }
    aux_hidden_states_ =
        torch::zeros(shape, torch::dtype(dtype).device(device_));
  }
  CHECK_LE(result_tokens, aux_hidden_states_.size(0))
      << "aux hidden state output exceeds graph token capacity";
  // Slice to match the actual shape
  auto slice =
      aux_hidden_states_.slice(/*dim=*/0, /*start=*/0, /*end=*/result_tokens);
  // Reshape slice if needed to match value shape
  if (slice.sizes() == value.sizes()) {
    slice.copy_(value, /*non_blocking=*/true);
  }
}

namespace {
void zero_tensor_tail(torch::Tensor& tensor,
                      int64_t start,
                      int64_t end,
                      int64_t dim = 0) {
  if (start >= end) {
    return;
  }
  tensor.slice(/*dim=*/dim, /*start=*/start, /*end=*/end).zero_();
}
}  // namespace

std::vector<int32_t>
GraphPersistentParam::update_expanded_spec_decode_attention(
    const ModelInputParams& input_params,
    uint32_t actual_num_tokens,
    uint32_t padded_num_tokens) {
  CHECK(input_params.is_spec_verify)
      << "expanded spec decode attention is only for spec verify";
  CHECK(input_params.meta.batch_forward_type.is_chunked_prefill())
      << "expanded spec decode attention expects chunked prefill";
  CHECK(input_params.graph.use_expanded_decode_for_spec_verify_attention)
      << "MTP worker must prepare expanded spec-verify graph input";
  CHECK(input_params.graph.expanded_kv_seq_lens.defined())
      << "expanded spec-verify kv seq lens must be defined";
  CHECK(input_params.graph.expanded_block_tables.defined())
      << "expanded spec-verify block tables must be defined";
  CHECK_EQ(input_params.graph.expanded_kv_seq_lens_vec.size(),
           static_cast<size_t>(actual_num_tokens))
      << "expanded kv seq lens size must match validate tokens";
  CHECK_EQ(input_params.graph.expanded_kv_seq_lens.numel(),
           static_cast<int64_t>(actual_num_tokens))
      << "expanded kv seq lens tensor size must match validate tokens";
  CHECK_EQ(input_params.graph.expanded_block_tables.dim(), 2)
      << "expanded block tables must be 2D";
  CHECK_EQ(input_params.graph.expanded_block_tables.size(0),
           static_cast<int64_t>(actual_num_tokens))
      << "expanded block table rows must match validate tokens";

  std::vector<int32_t> expanded_kv_seq_lens_vec =
      input_params.graph.expanded_kv_seq_lens_vec;
  expanded_kv_seq_lens_vec.reserve(padded_num_tokens);

  if (padded_num_tokens > actual_num_tokens) {
    const int64_t pad_count = padded_num_tokens - actual_num_tokens;
    for (int64_t i = 0; i < pad_count; ++i) {
      expanded_kv_seq_lens_vec.emplace_back(1);
    }
  }

  torch::Tensor expanded_kv_tensor =
      torch::tensor(expanded_kv_seq_lens_vec, torch::kInt).to(device_);
  expanded_kv_seq_lens_.slice(/*dim=*/0, /*start=*/0, /*end=*/padded_num_tokens)
      .copy_(expanded_kv_tensor, /*non_blocking=*/true);

  const int64_t block_table_len =
      input_params.graph.expanded_block_tables.size(1);
  persistent_expanded_block_tables_
      .slice(/*dim=*/0, /*start=*/0, /*end=*/padded_num_tokens)
      .zero_();
  persistent_expanded_block_tables_
      .slice(/*dim=*/0, /*start=*/0, /*end=*/actual_num_tokens)
      .slice(/*dim=*/1, /*start=*/0, /*end=*/block_table_len)
      .copy_(input_params.graph.expanded_block_tables, /*non_blocking=*/true);
  return expanded_kv_seq_lens_vec;
}

void GraphPersistentParam::update_tokens(const torch::Tensor& tokens,
                                         const ModelInputParams& params,
                                         uint32_t actual_num_tokens,
                                         uint32_t padded_num_tokens) {
  CHECK_GT(padded_num_tokens, 0) << "padded_num_tokens must be > 0";
  const torch::Tensor& graph_tokens =
      params.graph.input_tokens_override.defined()
          ? params.graph.input_tokens_override
          : tokens;
  CHECK(graph_tokens.defined()) << "graph tokens must be defined";
  CHECK_GE(graph_tokens.size(0), static_cast<int64_t>(actual_num_tokens))
      << "graph token override is shorter than actual decode tokens";
  if (actual_num_tokens > 0) {
    persistent_tokens_.slice(/*dim=*/0, /*start=*/0, /*end=*/actual_num_tokens)
        .copy_(graph_tokens.slice(/*dim=*/0,
                                  /*start=*/0,
                                  /*end=*/actual_num_tokens),
               /*non_blocking=*/true);
  }
  if (padded_num_tokens > actual_num_tokens) {
    zero_tensor_tail(persistent_tokens_,
                     actual_num_tokens,
                     static_cast<int64_t>(padded_num_tokens));
  }
}

std::optional<ModelInputParams> GraphPersistentParam::update(
    const torch::Tensor& tokens,
    const torch::Tensor& k_cache,
    const torch::Tensor& v_cache,
    const torch::Tensor& positions,
    const ModelInputParams& params,
    uint32_t padded_num_tokens,
    bool return_capture_params,
    bool skip_token_update) {
  CHECK_GT(padded_num_tokens, 0) << "padded_num_tokens must be > 0";
  const uint32_t actual_num_tokens = tokens.size(0);
  const bool is_decode = params.meta.batch_forward_type.is_decode();
  const bool is_chunked_prefill =
      params.meta.batch_forward_type.is_chunked_prefill();
  const bool is_qwen3_5_spec_verify_chunked_prefill =
      params.is_spec_verify && is_chunked_prefill &&
      is_hybrid_linear_attention_;
  CHECK(is_decode || is_qwen3_5_spec_verify_chunked_prefill)
      << "ACL graph persistent param only supports decode or Qwen3.5 "
         "spec-verify chunked prefill";
  const int64_t decode_tokens =
      is_decode ? std::max<int64_t>(options_.num_decoding_tokens(), 1) : 1;
  int64_t actual_batch_size = infer_actual_batch_size(params);
  if (is_chunked_prefill && params.meta.num_sequences > 0) {
    actual_batch_size = params.meta.num_sequences;
  } else if (is_decode) {
    if (params.meta.num_sequences == 0) {
      actual_batch_size = 0;
    } else {
      actual_batch_size = actual_num_tokens / decode_tokens;
    }
  }
  const int32_t q_max_seq_len = std::max<int32_t>(params.meta.q_max_seq_len, 1);
  const int64_t padded_batch_size =
      is_chunked_prefill
          ? (padded_num_tokens + q_max_seq_len - 1) / q_max_seq_len
          : padded_num_tokens;
  const bool is_empty_dp_decode_rank =
      is_decode && params.meta.num_sequences == 0 && actual_num_tokens > 0 &&
      params.parallel.dp_global_token_nums.size() > 1 &&
      params.attention.host.kv_seq_lens.empty() &&
      params.attention.host.q_seq_lens.empty();
  const int64_t actual_seq_len_rows =
      is_empty_dp_decode_rank
          ? 0
          : (is_chunked_prefill ? actual_batch_size : actual_num_tokens);

  // Copy data from input parameters to persistent graph tensors.
  // Schedule-overlap prepare can defer token copy until replay because tokens
  // are replaced asynchronously from the previous step output.
  if (!skip_token_update) {
    update_tokens(tokens, params, actual_num_tokens, padded_num_tokens);
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
  if (padded_num_tokens > actual_num_tokens) {
    zero_tensor_tail(persistent_positions_,
                     actual_num_tokens,
                     static_cast<int64_t>(padded_num_tokens),
                     use_mrope_ ? 1 : 0);
  }
  if (q_seq_lens_default_.defined() &&
      q_seq_lens_default_.sizes() == q_seq_lens_.sizes()) {
    q_seq_lens_.copy_(q_seq_lens_default_, /*non_blocking=*/true);
  }
  if (actual_seq_len_rows > 0 && params.attention.device.q_seq_lens.defined() &&
      params.attention.device.q_seq_lens.dim() >= 1 &&
      params.attention.device.q_seq_lens.numel() > 0) {
    const int64_t q_copy_len = std::min<int64_t>(
        actual_seq_len_rows, params.attention.device.q_seq_lens.size(0));
    if (q_copy_len > 0) {
      q_seq_lens_.slice(/*dim=*/0, /*start=*/0, /*end=*/q_copy_len)
          .copy_(params.attention.device.q_seq_lens.slice(/*dim=*/0,
                                                          /*start=*/0,
                                                          /*end=*/q_copy_len),
                 /*non_blocking=*/true);
    }
  }
  if (kv_seq_lens_default_.defined() &&
      kv_seq_lens_default_.sizes() == kv_seq_lens_.sizes()) {
    kv_seq_lens_.copy_(kv_seq_lens_default_, /*non_blocking=*/true);
  }
  if (actual_seq_len_rows > 0 &&
      params.attention.device.kv_seq_lens.defined() &&
      params.attention.device.kv_seq_lens.dim() >= 1 &&
      params.attention.device.kv_seq_lens.numel() > 0) {
    const int64_t kv_copy_len = std::min<int64_t>(
        actual_seq_len_rows, params.attention.device.kv_seq_lens.size(0));
    if (kv_copy_len > 0) {
      kv_seq_lens_.slice(/*dim=*/0, /*start=*/0, /*end=*/kv_copy_len)
          .copy_(params.attention.device.kv_seq_lens.slice(/*dim=*/0,
                                                           /*start=*/0,
                                                           /*end=*/kv_copy_len),
                 /*non_blocking=*/true);
    }
  }
  if (padded_batch_size > actual_seq_len_rows) {
    const int32_t padding_q_len = is_chunked_prefill ? q_max_seq_len : 1;
    q_seq_lens_
        .slice(/*dim=*/0,
               /*start=*/actual_seq_len_rows,
               /*end=*/padded_batch_size)
        .fill_(padding_q_len);
    kv_seq_lens_
        .slice(/*dim=*/0,
               /*start=*/actual_seq_len_rows,
               /*end=*/padded_batch_size)
        .fill_(1);
  }

  if (persistent_new_cache_slots_default_.defined() &&
      persistent_new_cache_slots_default_.sizes() ==
          persistent_new_cache_slots_.sizes()) {
    persistent_new_cache_slots_.copy_(persistent_new_cache_slots_default_,
                                      /*non_blocking=*/true);
  }
  if (actual_num_tokens > 0 &&
      params.attention.device.new_cache_slots.defined() &&
      params.attention.device.new_cache_slots.dim() >= 1 &&
      params.attention.device.new_cache_slots.numel() > 0) {
    const int64_t slot_copy_len =
        std::min<int64_t>(static_cast<int64_t>(actual_num_tokens),
                          params.attention.device.new_cache_slots.size(0));
    if (slot_copy_len > 0) {
      persistent_new_cache_slots_
          .slice(/*dim=*/0, /*start=*/0, /*end=*/slot_copy_len)
          .copy_(params.attention.device.new_cache_slots.slice(
                     /*dim=*/0, /*start=*/0, /*end=*/slot_copy_len),
                 /*non_blocking=*/true);
    }
  }
  if (actual_num_tokens < padded_num_tokens) {
    zero_tensor_tail(persistent_new_cache_slots_,
                     actual_num_tokens,
                     static_cast<int64_t>(padded_num_tokens));
  }
  if (!params.embedding.linear_state_ids.empty()) {
    const int64_t linear_copy_len = std::min<int64_t>(
        actual_batch_size,
        static_cast<int64_t>(params.embedding.linear_state_ids.size()));
    if (linear_copy_len > 0) {
      if (params.embedding.linear_state_indices.defined()) {
        persistent_linear_state_indices_
            .slice(/*dim=*/0, /*start=*/0, /*end=*/linear_copy_len)
            .copy_(params.embedding.linear_state_indices.slice(
                       /*dim=*/0, /*start=*/0, /*end=*/linear_copy_len),
                   /*non_blocking=*/true);
      } else {
        persistent_linear_state_indices_
            .slice(/*dim=*/0, /*start=*/0, /*end=*/linear_copy_len)
            .copy_(torch::tensor(params.embedding.linear_state_ids, torch::kInt)
                       .to(device_)
                       .slice(/*dim=*/0, /*start=*/0, /*end=*/linear_copy_len),
                   /*non_blocking=*/true);
      }
    }
    if (padded_batch_size > actual_batch_size) {
      persistent_linear_state_indices_
          .slice(/*dim=*/0,
                 /*start=*/actual_batch_size,
                 /*end=*/padded_batch_size)
          .fill_(kPaddingLinearStateId);
    }
  }
  if (params.num_accepted_tokens.defined()) {
    persistent_num_accepted_tokens_
        .slice(/*dim=*/0, /*start=*/0, /*end=*/actual_batch_size)
        .copy_(params.num_accepted_tokens.slice(
                   /*dim=*/0, /*start=*/0, /*end=*/actual_batch_size),
               /*non_blocking=*/true);
    if (padded_batch_size > actual_batch_size) {
      persistent_num_accepted_tokens_
          .slice(/*dim=*/0,
                 /*start=*/actual_batch_size,
                 /*end=*/padded_batch_size)
          .fill_(1);
    }
  }

  if (persistent_block_tables_default_.defined() &&
      persistent_block_tables_default_.sizes() ==
          persistent_block_tables_.sizes() &&
      padded_batch_size > 0) {
    CHECK_LE(padded_batch_size, persistent_block_tables_.size(0))
        << "padded graph metadata rows exceed persistent block table capacity";
    persistent_block_tables_
        .slice(/*dim=*/0, /*start=*/0, /*end=*/padded_batch_size)
        .copy_(persistent_block_tables_default_.slice(
                   /*dim=*/0, /*start=*/0, /*end=*/padded_batch_size),
               /*non_blocking=*/true);
  }
  if (actual_seq_len_rows > 0 &&
      params.attention.device.block_tables.defined() &&
      params.attention.device.block_tables.dim() >= 2 &&
      params.attention.device.block_tables.numel() > 0) {
    const int64_t block_rows_to_copy = std::min<int64_t>(
        actual_seq_len_rows, params.attention.device.block_tables.size(0));
    const int64_t actual_block_table_len =
        params.attention.device.block_tables.size(1);
    if (block_rows_to_copy > 0 && actual_block_table_len > 0) {
      auto slice_persistent_block_tables =
          persistent_block_tables_
              .slice(/*dim=*/0, /*start=*/0, /*end=*/block_rows_to_copy)
              .slice(/*dim=*/1, /*start=*/0, /*end=*/actual_block_table_len);
      slice_persistent_block_tables.copy_(
          params.attention.device.block_tables.slice(
              /*dim=*/0, /*start=*/0, /*end=*/block_rows_to_copy),
          /*non_blocking=*/true);
    }
  }
  if (actual_seq_len_rows < padded_batch_size) {
    zero_tensor_tail(
        persistent_block_tables_, actual_seq_len_rows, padded_batch_size);
  }

  // Update persistent embedding from input_embedding if available
  const auto& embedding = params.embedding.input_embedding;
  if (embedding.defined()) {
    const int64_t embedding_tokens = embedding.size(0);

    // Initialize persistent_embedding_ if needed and not already initialized
    if (persistent_embedding_.numel() == 0) {
      const int64_t max_tokens_per_batch = options_.max_tokens_per_batch();
      const int64_t embedding_dim = embedding.size(1);
      torch::Dtype dtype = util::parse_dtype(args_.dtype(), device_);
      persistent_embedding_ =
          torch::zeros({max_tokens_per_batch, embedding_dim},
                       torch::dtype(dtype).device(device_));
    }

    // Copy embedding data to persistent buffer
    persistent_embedding_
        .slice(/*dim=*/0, /*start=*/0, /*end=*/embedding_tokens)
        .copy_(embedding, /*non_blocking=*/true);
  }
  if (q_cu_seq_lens_default_.defined() &&
      q_cu_seq_lens_default_.sizes() == q_cu_seq_lens_.sizes()) {
    q_cu_seq_lens_.copy_(q_cu_seq_lens_default_, /*non_blocking=*/true);
  }
  const bool has_q_cu = params.attention.device.q_cu_seq_lens.defined() &&
                        params.attention.device.q_cu_seq_lens.dim() >= 1;
  const int64_t q_cu_size =
      (has_q_cu && params.attention.device.q_cu_seq_lens.numel() > 0)
          ? params.attention.device.q_cu_seq_lens.size(0)
          : 0;
  if (has_q_cu && q_cu_size > 0) {
    const bool use_qwen3_5_query_start_loc = is_hybrid_linear_attention_;
    const bool input_has_leading_zero =
        params.is_spec_verify && use_qwen3_5_query_start_loc;
    const int64_t required_q_cu_seq_lens =
        actual_seq_len_rows + (input_has_leading_zero ? 1 : 0);
    CHECK_GE(params.attention.device.q_cu_seq_lens.numel(),
             required_q_cu_seq_lens)
        << "q_cu_seq_lens does not have enough entries for ACL graph execution";
    if (use_qwen3_5_query_start_loc && !input_has_leading_zero) {
      q_cu_seq_lens_.slice(/*dim=*/0, /*start=*/0, /*end=*/1).zero_();
      q_cu_seq_lens_
          .slice(/*dim=*/0, /*start=*/1, /*end=*/actual_seq_len_rows + 1)
          .copy_(params.attention.device.q_cu_seq_lens.slice(
                     /*dim=*/0, /*start=*/0, /*end=*/actual_seq_len_rows),
                 /*non_blocking=*/true);
    } else {
      q_cu_seq_lens_
          .slice(/*dim=*/0,
                 /*start=*/0,
                 /*end=*/required_q_cu_seq_lens)
          .copy_(params.attention.device.q_cu_seq_lens.slice(
                     /*dim=*/0, /*start=*/0, /*end=*/required_q_cu_seq_lens),
                 /*non_blocking=*/true);
    }
    if (padded_batch_size > actual_seq_len_rows) {
      int32_t offset =
          is_empty_dp_decode_rank ? 0 : static_cast<int32_t>(actual_num_tokens);
      std::vector<int32_t> padded_q_cu_seq_lens;
      padded_q_cu_seq_lens.reserve(padded_batch_size - actual_seq_len_rows);
      const int32_t padding_q_len = is_chunked_prefill ? q_max_seq_len : 1;
      for (int64_t i = actual_seq_len_rows; i < padded_batch_size; ++i) {
        offset += padding_q_len;
        padded_q_cu_seq_lens.emplace_back(offset);
      }
      const int64_t padding_start =
          actual_seq_len_rows + (use_qwen3_5_query_start_loc ? 1 : 0);
      const int64_t padding_end =
          padded_batch_size + (use_qwen3_5_query_start_loc ? 1 : 0);
      q_cu_seq_lens_
          .slice(/*dim=*/0,
                 /*start=*/padding_start,
                 /*end=*/padding_end)
          .copy_(torch::tensor(padded_q_cu_seq_lens, torch::kInt).to(device_),
                 /*non_blocking=*/true);
    }
  }

  // Update attention mask only if needed
  if (need_update_attn_mask_) {
    update_attention_mask(params);
  }

  std::vector<int32_t> padded_kv_seq_lens_vec(
      static_cast<size_t>(padded_batch_size), 1);
  std::vector<int32_t> padded_q_seq_lens_vec(
      static_cast<size_t>(padded_batch_size), 1);
  CHECK_GE(params.attention.host.kv_seq_lens.size(),
           static_cast<size_t>(actual_seq_len_rows))
      << "kv_seq_lens host size is smaller than required graph rows";
  CHECK_GE(params.attention.host.q_seq_lens.size(),
           static_cast<size_t>(actual_seq_len_rows))
      << "q_seq_lens host size is smaller than required graph rows";
  for (int64_t i = 0; i < actual_seq_len_rows; ++i) {
    padded_kv_seq_lens_vec[static_cast<size_t>(i)] =
        params.attention.host.kv_seq_lens[static_cast<size_t>(i)];
    padded_q_seq_lens_vec[static_cast<size_t>(i)] =
        params.attention.host.q_seq_lens[static_cast<size_t>(i)];
  }
  for (int64_t i = actual_seq_len_rows; i < padded_batch_size; ++i) {
    padded_q_seq_lens_vec[static_cast<size_t>(i)] =
        is_chunked_prefill ? q_max_seq_len : 1;
  }
  const bool use_expanded_spec_decode_attention =
      params.graph.use_expanded_decode_for_spec_verify_attention;
  if (is_qwen3_5_spec_verify_chunked_prefill) {
    CHECK(use_expanded_spec_decode_attention)
        << "Qwen3.5 spec-verify ACL graph requires MTP worker expanded "
           "decode attention input";
  }
  std::vector<int32_t> expanded_kv_seq_lens_vec;
  if (use_expanded_spec_decode_attention) {
    expanded_kv_seq_lens_vec = update_expanded_spec_decode_attention(
        params, actual_num_tokens, padded_num_tokens);
  }

  if (uses_paged_attention_tiling()) {
    aclrtStream stream = c10_npu::getCurrentNPUStream().stream();

    if (k_cache.defined() && v_cache.defined() && k_cache.numel() > 0 &&
        v_cache.numel() > 0) {
      ModelInputParams plan_params = params;
      torch::Tensor plan_block_tables;
      if (use_expanded_spec_decode_attention) {
        plan_params.meta.num_sequences =
            static_cast<int32_t>(padded_num_tokens);
        plan_params.attention.device.kv_seq_lens = expanded_kv_seq_lens_.slice(
            /*dim=*/0, /*start=*/0, /*end=*/padded_num_tokens);
        plan_params.attention.device.q_seq_lens = torch::ones(
            {static_cast<int64_t>(padded_num_tokens)},
            torch::TensorOptions().dtype(torch::kInt).device(device_));
        plan_params.attention.host.kv_seq_lens = expanded_kv_seq_lens_vec;
        plan_params.attention.host.q_seq_lens =
            std::vector<int32_t>(static_cast<size_t>(padded_num_tokens), 1);
        plan_block_tables = persistent_expanded_block_tables_.slice(
            /*dim=*/0, /*start=*/0, /*end=*/padded_num_tokens);
        plan_params.attention.device.block_tables = plan_block_tables;
      } else {
        plan_params.meta.num_sequences =
            static_cast<int32_t>(padded_batch_size);
        plan_params.attention.device.kv_seq_lens =
            kv_seq_lens(static_cast<uint32_t>(padded_batch_size));
        plan_params.attention.device.q_seq_lens =
            q_seq_lens(static_cast<uint32_t>(padded_batch_size));
        plan_params.attention.host.kv_seq_lens = padded_kv_seq_lens_vec;
        plan_params.attention.host.q_seq_lens = padded_q_seq_lens_vec;
        plan_block_tables =
            persistent_block_tables(static_cast<uint32_t>(padded_batch_size));
        plan_params.attention.device.block_tables = plan_block_tables;
      }
      plan_paged_attention_tiling(persistent_tokens(padded_num_tokens),
                                  k_cache,
                                  v_cache,
                                  plan_block_tables,
                                  plan_params,
                                  stream);
    }
  }

  // Update persistent dp/cp ep padding buffers. For capture this ensures
  // stable device addresses are recorded by the graph; for replay this
  // refreshes the data at those same addresses before graph_.replay().
  update_persistent_dp_ep_padding(params.parallel.dp_ep_padding_data,
                                  padded_num_tokens);
  update_persistent_cp_ep_padding(params.parallel.cp_ep_padding_data,
                                  padded_num_tokens);

  // Return ModelInputParams with persistent buffer references if requested
  if (return_capture_params) {
    std::optional<ModelInputParams> params_for_capture =
        std::make_optional<ModelInputParams>(params);
    // Set persistent buffers in params_for_capture
    params_for_capture->attention.device.kv_seq_lens =
        kv_seq_lens(static_cast<uint32_t>(padded_batch_size));
    params_for_capture->attention.device.q_seq_lens =
        q_seq_lens(static_cast<uint32_t>(padded_batch_size));
    params_for_capture->meta.actual_num_sequences =
        is_empty_dp_decode_rank ? 0 : static_cast<int32_t>(actual_num_tokens);
    params_for_capture->attention.host.kv_seq_lens = padded_kv_seq_lens_vec;
    params_for_capture->attention.host.q_seq_lens = padded_q_seq_lens_vec;
    params_for_capture->meta.num_sequences =
        static_cast<int32_t>(padded_batch_size);
    params_for_capture->meta.batch_forward_type =
        params.meta.batch_forward_type;
    params_for_capture->enable_graph = true;
    if (params_for_capture->parallel.dp_global_token_nums.size() > 1) {
      params_for_capture->parallel.dp_global_token_nums = std::vector<int32_t>(
          params_for_capture->parallel.dp_global_token_nums.size(),
          static_cast<int32_t>(padded_num_tokens));
    }
    params_for_capture->attention.device.new_cache_slots =
        persistent_new_cache_slots(padded_num_tokens);
    params_for_capture->attention.device.block_tables =
        persistent_block_tables(static_cast<uint32_t>(padded_batch_size));
    if (!params.embedding.linear_state_ids.empty()) {
      params_for_capture->embedding.linear_state_ids =
          params.embedding.linear_state_ids;
      params_for_capture->embedding.linear_state_ids.resize(
          static_cast<size_t>(padded_batch_size), kPaddingLinearStateId);
      params_for_capture->embedding.linear_state_indices =
          persistent_linear_state_indices(
              static_cast<uint32_t>(padded_batch_size));
      params_for_capture->parallel.has_initial_state =
          params.parallel.has_initial_state;
      params_for_capture->parallel.has_initial_state.resize(
          static_cast<size_t>(padded_batch_size), 0);
    }

    // Only set attn_mask if need_update_attn_mask_ is true
    if (need_update_attn_mask_) {
      params_for_capture->graph.attn_mask = persistent_mask(padded_num_tokens);
    }
    if (uses_paged_attention_tiling()) {
      params_for_capture->graph.tiling_data = tiling_data();
    } else {
      params_for_capture->graph.tiling_data = torch::Tensor();
    }
    // Set persistent embedding if available
    if (params.embedding.input_embedding.defined()) {
      params_for_capture->embedding.input_embedding =
          persistent_embedding(padded_num_tokens);
    }
    if (params.num_accepted_tokens.defined()) {
      params_for_capture->num_accepted_tokens = persistent_num_accepted_tokens(
          static_cast<uint32_t>(padded_batch_size));
    }
    if (use_expanded_spec_decode_attention) {
      params_for_capture->graph.use_expanded_decode_for_spec_verify_attention =
          true;
      params_for_capture->graph.expanded_kv_seq_lens =
          expanded_kv_seq_lens_.slice(
              /*dim=*/0, /*start=*/0, /*end=*/padded_num_tokens);
      params_for_capture->graph.expanded_block_tables =
          persistent_expanded_block_tables_.slice(
              /*dim=*/0, /*start=*/0, /*end=*/padded_num_tokens);
      params_for_capture->graph.expanded_tiling_data =
          uses_paged_attention_tiling() ? tiling_data() : torch::Tensor();
      params_for_capture->graph.expanded_kv_seq_lens_vec =
          expanded_kv_seq_lens_vec;
    }
    if (params.attention.device.q_cu_seq_lens.defined()) {
      const bool use_qwen3_5_query_start_loc = is_hybrid_linear_attention_;
      params_for_capture->attention.device.q_cu_seq_lens = q_cu_seq_lens_.slice(
          /*dim=*/0,
          /*start=*/0,
          /*end=*/padded_batch_size + (use_qwen3_5_query_start_loc ? 1 : 0));
    }

    // Replace dp/cp ep padding with slices of persistent buffers so that
    // the graph records stable device addresses.  Each slice has the same
    // size as the original tensor but points into the persistent buffer.
    // dp ep and cp ep are mutually exclusive; when neither is enabled the
    // src fields are all undefined and we leave dst untouched so that the
    // captured graph behaves identically to eager mode.
    replace_capture_dp_ep_padding(
        params.parallel.dp_ep_padding_data,
        params_for_capture->parallel.dp_ep_padding_data);
    replace_capture_cp_ep_padding(
        params.parallel.cp_ep_padding_data,
        params_for_capture->parallel.cp_ep_padding_data);

    auto& qsl = params_for_capture->parallel.query_start_loc;
    qsl.clear();
    qsl.reserve(static_cast<size_t>(padded_batch_size) + 1);
    qsl.emplace_back(0);
    for (int64_t i = 0; i < padded_batch_size; ++i) {
      qsl.emplace_back(qsl.back() +
                       padded_q_seq_lens_vec[static_cast<size_t>(i)]);
    }

    if (!params.parallel.has_initial_state.empty()) {
      auto& his = params_for_capture->parallel.has_initial_state;
      his = params.parallel.has_initial_state;
      if (his.size() > static_cast<size_t>(actual_batch_size)) {
        his.resize(static_cast<size_t>(actual_batch_size));
      }
      his.resize(static_cast<size_t>(padded_batch_size), 0);
    }

    if (params.num_accepted_tokens.defined() &&
        params.num_accepted_tokens.numel() > 0) {
      torch::Tensor nat_host = params.num_accepted_tokens.to(torch::kCPU)
                                   .to(torch::kLong)
                                   .contiguous();
      const int64_t copy_size =
          std::min<int64_t>(actual_batch_size, nat_host.numel());
      const int64_t* data = nat_host.data_ptr<int64_t>();
      params_for_capture->num_accepted_tokens_host.assign(data,
                                                          data + copy_size);
      params_for_capture->num_accepted_tokens_host.resize(
          static_cast<size_t>(padded_batch_size), 1);
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
  const int32_t dp_local_tp_size = options_.world_size() / options_.dp_size();

  // Cache headNum and head_dim as member variables
  num_head_ = static_cast<int32_t>(args_.n_heads() / dp_local_tp_size);
  head_dim_ = static_cast<int32_t>(args_.head_dim());

  atb::customize::CustomPagedAttentionParam pa_op_param;
  // default mask type is UNDEFINED, which means no mask is needed
  if (need_update_attn_mask_) {
    pa_op_param.maskType =
        atb::infer::PagedAttentionParam::MaskType::MASK_TYPE_NORM;
  }
  pa_op_param.headNum = num_head_;

  const int64_t total_kv_heads = args_.n_kv_heads().value_or(args_.n_heads());
  pa_op_param.kvHeadNum = std::max<int32_t>(
      1, static_cast<int32_t>(total_kv_heads) / dp_local_tp_size);

  const float head_dim_float = static_cast<float>(head_dim_);
  pa_op_param.qkScale = 1.0f / std::sqrt(head_dim_float);

  const bool is_bf16 = args_.dtype() == "bfloat16";
  if (is_bf16) {
    pa_op_param.outDataType = ACL_BF16;
  } else {
    pa_op_param.outDataType = ACL_FLOAT16;
  }

  status = atb::CreateOperation(pa_op_param, &custom_pa_op_for_plan_);
  CHECK_EQ(status, atb::NO_ERROR)
      << "Failed to create custom paged attention operation";
  CHECK_NE(custom_pa_op_for_plan_, nullptr) << "custom_pa_op_for_plan_ is null";
}

namespace {
constexpr uint32_t TILING_PARA_SIZE = 17;
constexpr uint32_t TILING_HEAD_SIZE = 44;

void parse_pa_host_tiling_buffer(const uint8_t* host_tiling_buffer,
                                 uint64_t tiling_buffer_size) {
  VLOG(kGraphExecutorLogVerboseLevel)
      << "hostTilingBuffer.tilingBuffer: "
      << static_cast<const void*>(host_tiling_buffer);
  VLOG(kGraphExecutorLogVerboseLevel)
      << "hostTilingBuffer.tilingBufferSize: " << tiling_buffer_size;
  if (host_tiling_buffer == nullptr || tiling_buffer_size == 0) {
    VLOG(kGraphExecutorLogVerboseLevel) << "Invalid host tiling buffer!";
    return;
  }

  uint32_t tilingParamSize = tiling_buffer_size / sizeof(uint32_t);
  std::vector<uint32_t> host_tiling_values(tilingParamSize);
  std::memcpy(host_tiling_values.data(),
              host_tiling_buffer,
              static_cast<size_t>(tilingParamSize) * sizeof(uint32_t));
  const uint32_t* hostTilingBuffer = host_tiling_values.data();
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
  // Get context_lens from input_params.attention.device.kv_seq_lens
  atb::Tensor atb_context_lens = atb_speed::Utils::AtTensor2Tensor(
      input_params.attention.device.kv_seq_lens);
  atb_context_lens.hostData =
      const_cast<int32_t*>(input_params.attention.host.kv_seq_lens.data());
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
    parse_pa_host_tiling_buffer(tiling_buffer_info.tilingBuffer,
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
  // update persistent_mask_ in-place
  const int64_t batch_size = input_params.attention.device.kv_seq_lens.size(0);
  const int64_t max_seq_len = input_params.meta.kv_max_seq_len > 0
                                  ? input_params.meta.kv_max_seq_len
                                  : args_.max_position_embeddings();

  // persistent_mask_ is already initialized in constructor
  // Check if size is sufficient
  CHECK_LE(max_seq_len, persistent_mask_.size(1))
      << "max_seq_len (" << max_seq_len << ") exceeds max_seq_len ("
      << persistent_mask_.size(1) << ")";

  // Check if q_max_seq_len > 1 (prefill mode, not decode mode)
  bool chunked_prefill = input_params.meta.q_max_seq_len > 1;

  // Calculate num_tokens: in chunked mode, sum of all q_len; in decode mode,
  // batch_size
  int64_t num_tokens = batch_size;  // Default for decode mode
  if (chunked_prefill) {
    CHECK_EQ(input_params.attention.host.q_seq_lens.size(), batch_size)
        << "q_seq_lens_vec size ("
        << input_params.attention.host.q_seq_lens.size() << ") != batch_size ("
        << batch_size << ")";
    num_tokens = std::accumulate(
        input_params.attention.host.q_seq_lens.begin(),
        input_params.attention.host.q_seq_lens.begin() + batch_size,
        int64_t(0));
  }

  // Check if num_tokens is within bounds
  CHECK_LE(num_tokens, persistent_mask_.size(0))
      << "num_tokens (" << num_tokens
      << ") exceeds graph attention-mask capacity (" << persistent_mask_.size(0)
      << ")";

  // Get slice for actual num_tokens (compatible with both chunked and
  // non-chunked)
  auto mask_slice =
      persistent_mask_.slice(/*dim=*/0, /*start=*/0, /*end=*/num_tokens)
          .slice(/*dim=*/1, /*start=*/0, /*end=*/max_seq_len);

  CHECK(persistent_mask_zero_template_.defined() &&
        persistent_mask_fill_template_.defined())
      << "persistent mask templates must be initialized";

  if (chunked_prefill) {
    // Generate mask considering both q_seq_lens and kv_seq_lens
    // For each sequence, generate mask with shape [q_len, kv_len]
    // mask_slice is [num_tokens, max_seq_len], where num_tokens = sum of all
    // q_len

    // Check if kv_seq_lens_vec is available
    CHECK_EQ(input_params.attention.host.kv_seq_lens.size(), batch_size)
        << "kv_seq_lens_vec size ("
        << input_params.attention.host.kv_seq_lens.size() << ") != batch_size ("
        << batch_size << ")";

    int64_t offset = 0;
    for (int64_t i = 0; i < batch_size; i++) {
      const int32_t q_len = input_params.attention.host.q_seq_lens[i];
      const int32_t kv_len = input_params.attention.host.kv_seq_lens[i];

      // For chunked mode, slice out q_len rows for this sequence
      // mask_slice is [num_tokens, max_seq_len]
      // Get [q_len, kv_len] slice from mask_slice[offset:offset+q_len, :kv_len]
      auto seq_mask_slice =
          mask_slice.slice(/*dim=*/0, /*start=*/offset, /*end=*/offset + q_len)
              .slice(
                  /*dim=*/1, /*start=*/0, /*end=*/kv_len);  // [q_len, kv_len]
      auto seq_zero_slice = persistent_mask_zero_template_
                                .slice(/*dim=*/0,
                                       /*start=*/offset,
                                       /*end=*/offset + q_len)
                                .slice(/*dim=*/1, /*start=*/0, /*end=*/kv_len);
      auto seq_fill_slice = persistent_mask_fill_template_
                                .slice(/*dim=*/0,
                                       /*start=*/offset,
                                       /*end=*/offset + q_len)
                                .slice(/*dim=*/1, /*start=*/0, /*end=*/kv_len);

      // Generate mask for this sequence: [q_len, kv_len]
      int32_t diagonal = kv_len - q_len;
      auto int_options =
          torch::TensorOptions().dtype(torch::kInt32).device(device_);
      auto row = torch::arange(q_len, int_options).unsqueeze(1);
      auto col = torch::arange(kv_len, int_options).unsqueeze(0);
      auto bias = col > (row + diagonal);  // True positions need to be masked
      seq_mask_slice.copy_(torch::where(bias, seq_fill_slice, seq_zero_slice),
                           /*non_blocking=*/true);

      // Update offset for next sequence
      offset += q_len;
    }
  } else {
    // Original logic: only consider kv_seq_lens (decode mode, q_len = 1 for
    // all)
    auto int_options =
        torch::TensorOptions().dtype(torch::kInt32).device(device_);
    auto positions = torch::arange(max_seq_len, int_options)
                         .unsqueeze(0)
                         .expand({batch_size, max_seq_len});

    auto context_lens_expanded =
        input_params.attention.device.kv_seq_lens.to(torch::kInt32)
            .unsqueeze(1)
            .expand({batch_size, max_seq_len});

    auto mask_condition = positions >= context_lens_expanded;
    auto zero_slice = persistent_mask_zero_template_
                          .slice(/*dim=*/0, /*start=*/0, /*end=*/num_tokens)
                          .slice(/*dim=*/1, /*start=*/0, /*end=*/max_seq_len);
    auto fill_slice = persistent_mask_fill_template_
                          .slice(/*dim=*/0, /*start=*/0, /*end=*/num_tokens)
                          .slice(/*dim=*/1, /*start=*/0, /*end=*/max_seq_len);
    mask_slice.copy_(torch::where(mask_condition, fill_slice, zero_slice),
                     /*non_blocking=*/true);
  }
}

}  // namespace xllm::npu

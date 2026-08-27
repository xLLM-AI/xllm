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

#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/ops/scaled_dot_product_attention.h>
#include <glog/logging.h>
#include <torch_npu/csrc/aten/CustomFunctions.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>

#include <array>
#include <unordered_map>

#include "core/kernels/npu/aclnn/pytorch_npu_helper.hpp"
#include "core/kernels/npu/npu_ops_api.h"
#include "core/kernels/npu/utils.h"

namespace {

constexpr int64_t kSwaIntMax = 2147483647;

using OptionalTensorRef = const std::optional<at::Tensor>&;
using OptionalSymIntArrayRef = c10::OptionalArrayRef<c10::SymInt>;
using FusedInferAttentionOutSignature =
    std::tuple<at::Tensor, at::Tensor>(const at::Tensor&,
                                       const at::Tensor&,
                                       const at::Tensor&,
                                       OptionalTensorRef,
                                       OptionalTensorRef,
                                       OptionalSymIntArrayRef,
                                       OptionalSymIntArrayRef,
                                       OptionalTensorRef,
                                       OptionalTensorRef,
                                       OptionalTensorRef,
                                       OptionalTensorRef,
                                       OptionalTensorRef,
                                       OptionalTensorRef,
                                       OptionalTensorRef,
                                       OptionalTensorRef,
                                       OptionalTensorRef,
                                       OptionalTensorRef,
                                       OptionalTensorRef,
                                       OptionalTensorRef,
                                       OptionalTensorRef,
                                       OptionalTensorRef,
                                       OptionalTensorRef,
                                       OptionalTensorRef,
                                       OptionalSymIntArrayRef,
                                       OptionalTensorRef,
                                       OptionalTensorRef,
                                       OptionalTensorRef,
                                       int64_t,
                                       double,
                                       int64_t,
                                       int64_t,
                                       c10::string_view,
                                       int64_t,
                                       int64_t,
                                       int64_t,
                                       int64_t,
                                       int64_t,
                                       int64_t,
                                       int64_t,
                                       bool,
                                       OptionalTensorRef,
                                       c10::ArrayRef<at::Tensor>);

std::vector<c10::SymInt> to_sym_ints(const std::vector<int64_t>& values) {
  std::vector<c10::SymInt> sym_ints;
  sym_ints.reserve(values.size());
  for (int64_t value : values) {
    sym_ints.emplace_back(value);
  }
  return sym_ints;
}

torch::Tensor ascend950_packed_causal_attention(
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& value,
    const std::vector<int64_t>& actual_seq_lengths,
    const std::vector<int64_t>& actual_seq_lengths_kv,
    int64_t num_heads,
    int64_t num_key_value_heads,
    double scale) {
  CHECK_EQ(actual_seq_lengths.size(), actual_seq_lengths_kv.size())
      << "query and key/value sequence counts must match";

  torch::Tensor output = torch::empty_like(query);
  int64_t query_start = 0;
  int64_t key_value_start = 0;
  for (size_t index = 0; index < actual_seq_lengths.size(); ++index) {
    const int64_t query_end = actual_seq_lengths[index];
    const int64_t key_value_end = actual_seq_lengths_kv[index];
    const int64_t query_length = query_end - query_start;
    const int64_t key_value_length = key_value_end - key_value_start;
    CHECK_EQ(query_length, key_value_length)
        << "Ascend950 torch attention fallback only supports non-chunked "
           "prefill";

    const torch::Tensor query_slice =
        query.narrow(0, query_start, query_length);
    torch::Tensor key_slice = key.narrow(0, key_value_start, key_value_length);
    torch::Tensor value_slice =
        value.narrow(0, key_value_start, key_value_length);
    key_slice = xllm::kernel::npu::expand_kv_heads(
        key_slice, num_heads, num_key_value_heads);
    value_slice = xllm::kernel::npu::expand_kv_heads(
        value_slice, num_heads, num_key_value_heads);

    const torch::Tensor query_4d = query_slice.permute({1, 0, 2}).unsqueeze(0);
    const torch::Tensor key_4d = key_slice.permute({1, 0, 2}).unsqueeze(0);
    const torch::Tensor value_4d = value_slice.permute({1, 0, 2}).unsqueeze(0);
    const torch::Tensor sequence_output =
        torch::scaled_dot_product_attention(query_4d,
                                            key_4d,
                                            value_4d,
                                            /*attn_mask=*/std::nullopt,
                                            /*dropout_p=*/0.0,
                                            /*is_causal=*/true,
                                            /*scale=*/scale)
            .squeeze(0)
            .permute({1, 0, 2});
    output.narrow(0, query_start, query_length).copy_(sequence_output);
    query_start = query_end;
    key_value_start = key_value_end;
  }

  CHECK_EQ(query_start, query.size(0));
  CHECK_EQ(key_value_start, key.size(0));
  return output;
}

torch::Tensor infer_attention_output(
    const torch::Tensor& query,
    const torch::Tensor& value,
    const std::optional<torch::Tensor>& block_table,
    int64_t num_heads,
    const std::string& input_layout) {
  if (input_layout == "TND" || input_layout == "NTD") {
    int64_t value_dim = query.size(-1);
    if (!block_table.has_value() && value.dim() >= 3) {
      value_dim = value.size(-1);
    }
    return torch::empty({query.size(0), num_heads, value_dim}, query.options());
  }

  if (input_layout == "BSH") {
    return torch::empty_like(query);
  }

  if (input_layout == "BNSD") {
    int64_t value_dim = query.size(-1);
    if (!block_table.has_value() && value.dim() >= 4) {
      value_dim = value.size(-1);
    }
    return torch::empty(
        {query.size(0), query.size(1), query.size(2), value_dim},
        query.options());
  }

  LOG(FATAL) << "Unsupported FIA input_layout: " << input_layout;
  return torch::Tensor();
}

torch::Tensor infer_softmax_lse(const torch::Tensor& query,
                                int64_t num_heads,
                                const std::string& input_layout,
                                bool softmax_lse_flag) {
  auto options = query.options().dtype(torch::kFloat32);
  if (!softmax_lse_flag) {
    return torch::empty({0}, options);
  }

  if (input_layout == "TND" || input_layout == "NTD") {
    return torch::empty({query.size(0), num_heads, 1}, options);
  }

  if (input_layout == "BSH") {
    return torch::empty({query.size(0), num_heads, query.size(1), 1}, options);
  }

  if (input_layout == "BNSD") {
    return torch::empty({query.size(0), query.size(1), query.size(2), 1},
                        options);
  }

  LOG(FATAL) << "Unsupported FIA input_layout: " << input_layout;
  return torch::Tensor();
}

std::optional<torch::Tensor> to_optional_tensor(
    const std::optional<torch::Tensor>& tensor_opt) {
  if (tensor_opt.has_value() && tensor_opt.value().defined()) {
    return tensor_opt.value();
  }
  return std::nullopt;
}

class FusedInferAttentionDecodeWorkspaceSignature {
 public:
  c10::DeviceIndex device_index = -1;
  torch::ScalarType query_dtype = torch::kFloat32;
  torch::ScalarType key_dtype = torch::kFloat32;
  torch::ScalarType value_dtype = torch::kFloat32;
  torch::ScalarType block_table_dtype = torch::kInt32;
  torch::ScalarType output_dtype = torch::kFloat32;
  int64_t query_heads = 0;
  int64_t head_dim = 0;
  int64_t key_num_blocks = 0;
  int64_t key_block_size = 0;
  int64_t key_hidden_size = 0;
  int64_t value_num_blocks = 0;
  int64_t value_block_size = 0;
  int64_t value_hidden_size = 0;
  int64_t num_heads = 0;
  int64_t num_key_value_heads = 0;
  int64_t block_size = 0;
  double scale = 0.0;

  bool operator==(const FusedInferAttentionDecodeWorkspaceSignature&) const =
      default;
};

struct FusedInferAttentionDecodeWorkspaceEntry {
  FusedInferAttentionDecodeWorkspaceSignature signature;
  torch::Tensor workspace;
  torch::Tensor softmax_lse;
};

FusedInferAttentionDecodeWorkspaceSignature
make_fused_infer_attention_decode_workspace_signature(
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& value,
    const torch::Tensor& block_table,
    int64_t num_heads,
    int64_t num_key_value_heads,
    double scale,
    int64_t block_size,
    const torch::Tensor& output) {
  CHECK_EQ(query.dim(), 3) << "FIA decode query must use TND layout";
  CHECK_EQ(key.dim(), 3) << "FIA decode key cache must be three-dimensional";
  CHECK_EQ(value.dim(), 3)
      << "FIA decode value cache must be three-dimensional";
  CHECK_EQ(block_table.dim(), 2)
      << "FIA decode block table must be two-dimensional";
  CHECK_EQ(output.dim(), 3) << "FIA decode output must use TND layout";

  return FusedInferAttentionDecodeWorkspaceSignature{
      .device_index = query.device().index(),
      .query_dtype = query.scalar_type(),
      .key_dtype = key.scalar_type(),
      .value_dtype = value.scalar_type(),
      .block_table_dtype = block_table.scalar_type(),
      .output_dtype = output.scalar_type(),
      .query_heads = query.size(1),
      .head_dim = query.size(2),
      .key_num_blocks = key.size(0),
      .key_block_size = key.size(1),
      .key_hidden_size = key.size(2),
      .value_num_blocks = value.size(0),
      .value_block_size = value.size(1),
      .value_hidden_size = value.size(2),
      .num_heads = num_heads,
      .num_key_value_heads = num_key_value_heads,
      .block_size = block_size,
      .scale = scale,
  };
}

class FusedInferAttentionDecodeWorkspaceCache final {
 public:
  FusedInferAttentionDecodeWorkspaceEntry& get(
      const torch::Tensor& query,
      const torch::Tensor& key,
      const torch::Tensor& value,
      const torch::Tensor& block_table,
      const std::vector<int64_t>& actual_seq_lengths,
      const std::vector<int64_t>& actual_seq_lengths_kv,
      int64_t num_heads,
      int64_t num_key_value_heads,
      double scale,
      int64_t block_size,
      const torch::Tensor& output) {
    const c10::DeviceIndex device_index = query.device().index();
    const aclrtStream stream =
        c10_npu::getCurrentNPUStream(device_index).stream();
    const FusedInferAttentionDecodeWorkspaceSignature signature =
        make_fused_infer_attention_decode_workspace_signature(
            query,
            key,
            value,
            block_table,
            num_heads,
            num_key_value_heads,
            scale,
            block_size,
            output);
    std::vector<FusedInferAttentionDecodeWorkspaceEntry>& stream_entries =
        entries_[stream];
    for (FusedInferAttentionDecodeWorkspaceEntry& entry : stream_entries) {
      if (entry.signature == signature) {
        return entry;
      }
    }

    torch::Tensor workspace =
        xllm::kernel::npu::npu_fused_infer_attention_decode_get_max_workspace(
            query,
            key,
            value,
            block_table,
            actual_seq_lengths,
            actual_seq_lengths_kv,
            num_heads,
            num_key_value_heads,
            scale,
            block_size);
    CHECK(workspace.defined()) << "FIA eager workspace must be defined";
    CHECK_EQ(workspace.device(), query.device())
        << "FIA eager workspace must be on the query device";

    stream_entries.emplace_back(FusedInferAttentionDecodeWorkspaceEntry{
        .signature = signature,
        .workspace = std::move(workspace),
        .softmax_lse = torch::empty({0}, query.options()),
    });
    return stream_entries.back();
  }

 private:
  std::unordered_map<aclrtStream,
                     std::vector<FusedInferAttentionDecodeWorkspaceEntry>>
      entries_;
};

FusedInferAttentionDecodeWorkspaceCache&
fused_infer_attention_decode_workspace_cache() {
  thread_local FusedInferAttentionDecodeWorkspaceCache cache;
  return cache;
}

}  // namespace

namespace xllm::kernel::npu {

torch::Tensor npu_fused_infer_attention_decode_get_max_workspace(
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& value,
    const torch::Tensor& block_table,
    const std::vector<int64_t>& actual_seq_lengths,
    const std::vector<int64_t>& actual_seq_lengths_kv,
    int64_t num_heads,
    int64_t num_key_value_heads,
    double scale,
    int64_t block_size) {
  std::vector<c10::SymInt> actual_seq_lengths_sym =
      to_sym_ints(actual_seq_lengths);
  std::vector<c10::SymInt> actual_seq_lengths_kv_sym =
      to_sym_ints(actual_seq_lengths_kv);
  const std::optional<torch::Tensor> none_tensor = std::nullopt;
  const at::OptionalSymIntArrayRef none_int_array = std::nullopt;

  return at_npu::native::custom_ops::
      _npu_fused_infer_attention_score_get_max_workspace(
          query,
          key,
          value,
          none_tensor,
          none_tensor,
          actual_seq_lengths_sym,
          actual_seq_lengths_kv_sym,
          none_tensor,
          none_tensor,
          none_tensor,
          none_tensor,
          none_tensor,
          none_tensor,
          none_tensor,
          none_tensor,
          none_tensor,
          none_tensor,
          none_tensor,
          block_table,
          none_tensor,
          none_tensor,
          none_tensor,
          none_tensor,
          none_int_array,
          none_tensor,
          none_tensor,
          none_tensor,
          num_heads,
          scale,
          kSwaIntMax,
          /*next_tokens=*/0,
          "TND",
          num_key_value_heads,
          /*sparse_mode=*/0,
          /*inner_precise=*/0,
          block_size,
          /*antiquant_mode=*/0,
          /*key_antiquant_mode=*/0,
          /*value_antiquant_mode=*/0,
          /*softmax_lse_flag=*/false);
}

void npu_fused_infer_attention_decode_out(
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& value,
    const torch::Tensor& block_table,
    const std::vector<int64_t>& actual_seq_lengths,
    const std::vector<int64_t>& actual_seq_lengths_kv,
    int64_t num_heads,
    int64_t num_key_value_heads,
    double scale,
    int64_t block_size,
    const torch::Tensor& workspace,
    torch::Tensor& output,
    torch::Tensor& softmax_lse) {
  std::vector<c10::SymInt> actual_seq_lengths_sym =
      to_sym_ints(actual_seq_lengths);
  std::vector<c10::SymInt> actual_seq_lengths_kv_sym =
      to_sym_ints(actual_seq_lengths_kv);
  const std::optional<torch::Tensor> none_tensor = std::nullopt;
  const at::OptionalSymIntArrayRef none_int_array = std::nullopt;
  const std::optional<at::Tensor> workspace_tensor = workspace;
  const std::array<at::Tensor, 2> outputs = {output, softmax_lse};
  static const auto op =
      c10::Dispatcher::singleton()
          .findSchemaOrThrow("npu::npu_fused_infer_attention_score", "out")
          .typed<FusedInferAttentionOutSignature>();

  op.call(query,
          key,
          value,
          none_tensor,
          none_tensor,
          actual_seq_lengths_sym,
          actual_seq_lengths_kv_sym,
          none_tensor,
          none_tensor,
          none_tensor,
          none_tensor,
          none_tensor,
          none_tensor,
          none_tensor,
          none_tensor,
          none_tensor,
          none_tensor,
          none_tensor,
          block_table,
          none_tensor,
          none_tensor,
          none_tensor,
          none_tensor,
          none_int_array,
          none_tensor,
          none_tensor,
          none_tensor,
          num_heads,
          scale,
          kSwaIntMax,
          /*next_tokens=*/0,
          "TND",
          num_key_value_heads,
          /*sparse_mode=*/0,
          /*inner_precise=*/0,
          block_size,
          /*antiquant_mode=*/0,
          /*key_antiquant_mode=*/0,
          /*value_antiquant_mode=*/0,
          /*softmax_lse_flag=*/false,
          workspace_tensor,
          outputs);
}

void npu_fused_infer_attention_decode_out_cached(
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& value,
    const torch::Tensor& block_table,
    const std::vector<int64_t>& actual_seq_lengths,
    const std::vector<int64_t>& actual_seq_lengths_kv,
    int64_t num_heads,
    int64_t num_key_value_heads,
    double scale,
    int64_t block_size,
    torch::Tensor& output) {
  FusedInferAttentionDecodeWorkspaceEntry& cache_entry =
      fused_infer_attention_decode_workspace_cache().get(query,
                                                         key,
                                                         value,
                                                         block_table,
                                                         actual_seq_lengths,
                                                         actual_seq_lengths_kv,
                                                         num_heads,
                                                         num_key_value_heads,
                                                         scale,
                                                         block_size,
                                                         output);
  npu_fused_infer_attention_decode_out(query,
                                       key,
                                       value,
                                       block_table,
                                       actual_seq_lengths,
                                       actual_seq_lengths_kv,
                                       num_heads,
                                       num_key_value_heads,
                                       scale,
                                       block_size,
                                       cache_entry.workspace,
                                       output,
                                       cache_entry.softmax_lse);
}

std::tuple<torch::Tensor, torch::Tensor> npu_fused_infer_attention(
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& value,
    const std::optional<torch::Tensor>& atten_mask,
    const std::optional<torch::Tensor>& block_table,
    const std::vector<int64_t>& actual_seq_lengths,
    const std::vector<int64_t>& actual_seq_lengths_kv,
    int64_t num_heads,
    int64_t num_key_value_heads,
    double scale,
    int64_t block_size,
    int64_t sparse_mode,
    const std::string& input_layout,
    bool softmax_lse_flag,
    bool is_causal) {
  check_tensor(query, "query", "npu_fused_infer_attention");
  check_tensor(key, "key", "npu_fused_infer_attention");
  check_tensor(value, "value", "npu_fused_infer_attention");
  CHECK_GT(num_heads, 0) << "num_heads must be positive";
  CHECK(!actual_seq_lengths.empty()) << "actual_seq_lengths must not be empty";
  CHECK(!actual_seq_lengths_kv.empty())
      << "actual_seq_lengths_kv must not be empty";

  torch::Tensor output = infer_attention_output(
      query, value, block_table, num_heads, input_layout);
  torch::Tensor softmax_lse =
      infer_softmax_lse(query, num_heads, input_layout, softmax_lse_flag);

  if (is_ascend950() && input_layout == "TND" && !block_table.has_value()) {
    CHECK(!softmax_lse_flag)
        << "Ascend950 torch attention fallback does not return softmax_lse";
    output = ascend950_packed_causal_attention(query,
                                               key,
                                               value,
                                               actual_seq_lengths,
                                               actual_seq_lengths_kv,
                                               num_heads,
                                               num_key_value_heads,
                                               scale);
    return {output, softmax_lse};
  }

  std::vector<torch::Tensor> key_tensors_vec{key};
  std::vector<torch::Tensor> value_tensors_vec{value};
  torch::TensorList key_tensors(key_tensors_vec);
  torch::TensorList value_tensors(value_tensors_vec);

  std::optional<torch::Tensor> none_tensor = std::nullopt;
  std::optional<torch::Tensor> atten_mask_tensor =
      to_optional_tensor(atten_mask);
  std::optional<torch::Tensor> block_table_tensor =
      to_optional_tensor(block_table);

  torch::IntArrayRef actual_seq_lengths_ref(actual_seq_lengths);
  torch::IntArrayRef actual_seq_lengths_kv_ref(actual_seq_lengths_kv);
  std::optional<torch::IntArrayRef> actual_seq_lengths_opt =
      actual_seq_lengths_ref;
  std::optional<torch::IntArrayRef> actual_seq_lengths_kv_opt =
      actual_seq_lengths_kv_ref;
  std::optional<torch::IntArrayRef> none_int_array = std::nullopt;

  std::string layout = input_layout;
  char* input_layout_ptr = const_cast<char*>(layout.c_str());
  int64_t pre_tokens = kSwaIntMax;
  int64_t next_tokens = is_causal ? 0 : kSwaIntMax;
  int64_t inner_precise = 0;
  int64_t antiquant_mode = 0;
  int64_t key_antiquant_mode = 0;
  int64_t value_antiquant_mode = 0;

  EXEC_NPU_CMD(aclnnFusedInferAttentionScoreV3,
               query,
               key_tensors,
               value_tensors,
               none_tensor,  // pse_shift
               atten_mask_tensor,
               actual_seq_lengths_opt,
               actual_seq_lengths_kv_opt,
               none_tensor,  // dequant_scale1
               none_tensor,  // quant_scale1
               none_tensor,  // dequant_scale2
               none_tensor,  // quant_scale2
               none_tensor,  // quant_offset2
               none_tensor,  // antiquant_scale
               none_tensor,  // antiquant_offset
               block_table_tensor,
               none_tensor,     // query_padding_size
               none_tensor,     // kv_padding_size
               none_tensor,     // key_antiquant_scale
               none_tensor,     // key_antiquant_offset
               none_tensor,     // value_antiquant_scale
               none_tensor,     // value_antiquant_offset
               none_tensor,     // key_shared_prefix
               none_tensor,     // value_shared_prefix
               none_int_array,  // actual_shared_prefix_len
               none_tensor,     // query_rope
               none_tensor,     // key_rope
               none_tensor,     // key_rope_antiquant_scale
               num_heads,
               scale,
               pre_tokens,
               next_tokens,
               input_layout_ptr,
               num_key_value_heads,
               sparse_mode,
               inner_precise,
               block_size,
               antiquant_mode,
               softmax_lse_flag,
               key_antiquant_mode,
               value_antiquant_mode,
               output,
               softmax_lse);

  return {output, softmax_lse};
}

}  // namespace xllm::kernel::npu

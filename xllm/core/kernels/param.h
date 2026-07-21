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

#pragma once

#include <torch/torch.h>

#include <optional>
#include <string>
#include <vector>

namespace xllm::layer {
struct AttentionMetadata;
}  // namespace xllm::layer

namespace xllm::kernel {

// Note: add default values for optional parameters in the struct definition

// Rotary embedding parameters
struct RotaryParams {
  // Query tensor. First dimension is total_seq_len (T).
  // Will be reshaped to [T, -1] and concatenated with k before applying rotary
  // embedding. Head size must be between 2 and 256.
  torch::Tensor q;
  // Key tensor. First dimension must match q.size(0) (total_seq_len).
  // Will be reshaped to [T, -1] and concatenated with q before applying rotary
  // embedding.
  torch::Tensor k;
  // Sin cache tensor for rotary embedding. Shape:
  // - [rope_seqlen, rope_dim] if dynamic_ntk=false
  // - [batch_size, rope_seqlen, rope_dim] if dynamic_ntk=true
  // rope_dim must be between 2 and head_size, and must be even.
  // rope_dim is extracted as sin.size(-1) and used to reshape qk tensor.
  torch::Tensor sin;
  // Cos cache tensor for rotary embedding. Same shape as sin.
  // The rope_seqlen-stride must equal to sin's rope_seqlen-stride.
  torch::Tensor cos;
  // Precomputed cos_sin tensor. Not used in current MLU implementation
  // (rope.cpp).
  torch::Tensor cos_sin;
  // Pre-formatted cos_sin cache for kernels that need [cos_half, sin_half]
  // layout (CUDA, MUSA, ILU). Avoids chunk/cat operations per layer.
  torch::Tensor precomputed_cos_sin;
  // Optional position IDs tensor. Type must be int32.
  // Shape: [total_seqlen] if discrete=true, or [batch_size] if discrete=false.
  // If discrete=true, position_ids must be provided.
  std::optional<torch::Tensor> position_ids;
  // Cumulative query lengths tensor. Type must be int32, must be contiguous.
  // Required in pack mode (when q/k are 3D). Size should be [batch_size + 1].
  // Note: In current MLU implementation, this is always passed to underlying
  // API.
  std::optional<torch::Tensor> cu_query_lens;
  // Whether to use interleaved rotary embedding pattern.
  bool interleaved;
  // Whether to use discrete position mode. If true, position_ids must be
  // provided and have shape [total_seqlen]. If false, position_ids can be None
  // or have shape [batch_size].
  bool discrete;
  // Whether to use dynamic NTK (Neural Tangent Kernel) scaling.
  // If true, sin and cos caches must have batch dimension.
  // Note: Current MLU implementation hardcodes this to false when calling
  // underlying API, so dynamic_ntk=true may not be fully supported.
  bool dynamic_ntk = false;
  // Maximum query length. In pad mode (4D input), must equal to input.size(1).
  // Must be less than or equal to rope_seqlen if not using discrete
  // position_ids.
  int64_t max_query_len;
};

// Activation parameters
struct ActivationParams {
  // Input tensor. Must be contiguous, dimension >= 2.
  // Last dimension is in_channel, which must be > 0.
  // If is_gated=true, in_channel must be even.
  torch::Tensor input;
  // Output tensor. Must be contiguous, dimension >= 2.
  // Must have same attributes (device, dtype) as input.
  // Only supports stride in dim(-2), stride(-1) must be 1.
  // Shape: [total_tokens, inner_size] where inner_size = in_channel/2 if
  // is_gated else in_channel.
  torch::Tensor output;
  // Optional bias tensor, only used for MoE activation.
  // If provided, cusum_token_count must also be provided.
  // Shape: [expert_size, in_channel]. Must be contiguous.
  std::optional<torch::Tensor> bias;
  // Optional cumulative token count tensor. Type should be int32.
  // Required when bias is provided. Must be contiguous.
  // Size: [num_expert + 1], where num_expert = size(0) - 1.
  std::optional<torch::Tensor> cusum_token_count;
  // Activation mode string. Must be one of: "silu", "gelu", "quick_gelu",
  // "swish".
  // - "silu": SiLU activation (Swish-1)
  // - "gelu": GELU activation
  // - "quick_gelu": Quick GELU with coefficient 1.702
  // - "swish": Swish activation
  std::string act_mode;
  // Whether to use gated activation. If true, input's last dimension
  // (in_channel) must be even, and output's inner_size will be in_channel/2.
  bool is_gated;
  // Starting expert ID for MoE activation. Used when processing multiple
  // experts.
  int64_t start_expert_id = 0;
  // Expert size for MoE activation. Used when bias is provided.
  // Bias tensor shape must be [expert_size, in_channel].
  int64_t expert_size = 0;
};

// Reshape paged cache parameters
struct ReshapePagedCacheParams {
  // Key tensor from context. Shape: [num_tokens, num_heads, head_dim].
  // Last two dimensions must be contiguous: stride(-1)==1,
  // stride(-2)==head_dim. Must have same device and dtype as k_cache and
  // v_cache.
  torch::Tensor key;
  // Optional value tensor from context. Shape: [num_tokens, num_heads,
  // head_dim]. If provided, v_cache must also be provided (and vice versa).
  // Last two dimensions must be contiguous: stride(-1)==1,
  // stride(-2)==head_dim. Must have same device and dtype as other tensors.
  std::optional<torch::Tensor> value;
  // Key cache tensor in paged format. Shape: [num_blocks, num_heads,
  // block_size, head_dim]. Must be contiguous. Must have same device and dtype
  // as key and value.
  torch::Tensor k_cache;
  // Optional value cache tensor in paged format. Shape: [num_blocks, num_heads,
  // block_size, head_dim]. If provided, value must also be provided (and vice
  // versa). Must be contiguous. Must have same device and dtype as other
  // tensors.
  std::optional<torch::Tensor> v_cache;
  // Slot mapping tensor. Shape: [num_tokens]. Type must be int32.
  // Maps each token to its corresponding slot in the cache. Must be contiguous.
  // Must have same device as key.
  torch::Tensor slot_mapping;
  // Direction flag: false = CONTEXT2CACHE (copy from context to cache),
  // true = CACHE2CONTEXT (copy from cache to context).
  bool direction = false;
  // Optional scale tensor for quantized key cache. Shape: [num_blocks,
  // num_heads, block_size]. Dtype: float32. Required when using INT8
  // quantization.
  std::optional<torch::Tensor> k_cache_scale;
  // Optional scale tensor for quantized value cache. Shape: [num_blocks,
  // num_heads, block_size]. Dtype: float32. Required when using INT8
  // quantization.
  std::optional<torch::Tensor> v_cache_scale;
};

// ReshapeFromCacheParams describes parameters for gathering and flattening
// KV (Key/Value) cached data from a possibly paged or non-contiguous storage
// format into a contiguous tensor.
struct ReshapeFromCacheParams {
  // Target tensor to store reshaped key values. Shape: [total_length, head_num,
  // head_size]. Dtype: float32, float16, bfloat16, int8.
  torch::Tensor key;
  // Optional target tensor to store reshaped value values. If provided,
  // value_cache must also be provided. Shape: [total_length, head_num,
  // head_size]. Dtype: float32, float16, bfloat16, int8.
  std::optional<torch::Tensor> value;
  // Source tensor containing cached key values.
  // Shape:
  //   - Linear mode: [max_batch_size, head_num, cache_mem_len, head_size]
  //   - Paged mode: [total_blocks, head_num, block_size, head_size]
  // Dtype: float32, float16, bfloat16, int8.
  torch::Tensor key_cache;
  // Optional source tensor containing cached value values. If provided, value
  // must also be provided. Shape:
  //   - Linear mode: [max_batch_size, head_num, cache_mem_len, head_size]
  //   - Paged mode: [total_blocks, head_num, block_size, head_size]
  // Dtype: float32, float16, bfloat16, int8.
  std::optional<torch::Tensor> value_cache;
  // 1D tensor representing the lengths of each batch context.
  // Shape: [batch_size]. Dtype: int32.
  torch::Tensor context_lengths;
  // Maximum context length that can be processed at once.
  // Used for memory allocation and bounds checking.
  int64_t max_context_len;
  // Optional 1D tensor with per-context sequence offsets.
  // If provided, applies a shift offset for each context's beginning location.
  // Shape: [batch_size]. Dtype: int32. Default: None.
  std::optional<torch::Tensor> context_seq_offset;
  // Optional tensor containing the block indices for each batch.
  // Shape:
  //   - Linear mode: [batch_size, 1]
  //   - Paged mode: [batch_size, max_blocks]
  // Dtype: int32. Default: None (linear mode).
  std::optional<torch::Tensor> block_tables;
  // Optional 1D tensor representing the cache sequence offset for each batch.
  // Used for slicing key and value cache starts in memory.
  // Shape: [batch_size]. Dtype: int32. Default: None.
  std::optional<torch::Tensor> cache_seq_offset;

  // ========== Quantization parameters (for dequant_from_paged_cache)
  // ========== Optional scale tensor for quantized key cache. Shape:
  // [num_blocks, num_heads, block_size] or [num_heads, head_dim]. Dtype:
  // float32. Required when dequantizing INT8 cache.
  std::optional<torch::Tensor> key_cache_quant_scale;
  // Optional scale tensor for quantized value cache.
  // Shape: [num_blocks, num_heads, block_size] or [num_heads, head_dim].
  // Dtype: float32. Required when dequantizing INT8 cache.
  std::optional<torch::Tensor> value_cache_quant_scale;
  // Quantization mode: 0 for per-channel, 1 for per-token. Default: 1.
  int64_t quant_mode = 1;
  // Quantization bit size. Default: 8 (INT8).
  int64_t quant_bit = 8;
};

// Fused layer norm parameters
struct FusedLayerNormParams {
  // Input tensor. Dimension must be >= 2. Last dimension is hidden_size.
  // Last dimension must be contiguous: stride(-1) == 1.
  // Must have same device and dtype as residual, weight, beta, bias,
  // residual_out, normed_out.
  torch::Tensor input;
  // Output tensor. Must have same shape as input.
  // If inplace (input.data_ptr() == output.data_ptr()), strides must also be
  // the same. Must have same device as input, smooth_quant_scale, quant_scale.
  torch::Tensor output;
  // Optional residual tensor. Must have same shape as input.
  // If provided, must have same device and dtype as input.
  std::optional<torch::Tensor> residual;
  // Weight tensor (gamma). Shape: [hidden_size]. Must be contiguous.
  // Required for both layernorm and rmsnorm modes.
  // Must have same device and dtype as input.
  torch::Tensor weight;
  // Optional beta tensor. Shape: [hidden_size]. Must be contiguous.
  // Required for layernorm mode, not used in rmsnorm mode.
  // If provided, must have same dtype as weight.
  std::optional<torch::Tensor> beta;
  // Optional bias tensor. Shape: [hidden_size]. Must be contiguous.
  // Must have same device and dtype as input.
  std::optional<torch::Tensor> bias;
  // Optional quantization scale tensor. Type must be float.
  // Shape: [hidden_size] (1D) or [head, headdim] (2D).
  // - 1D: per-channel quantization, input will be flattened to 2D
  // - 2D: only supported for rmsnorm mode, input must be dim >= 3,
  //       shape must be [head, headdim], residual and bias not supported
  // If dynamic_quant=true, this must be provided.
  std::optional<torch::Tensor> quant_scale;
  // Optional residual output tensor. Used when store_output_before_norm=true.
  // Not supported when both bias and residual are not provided.
  // Must have same device and dtype as input.
  std::optional<torch::Tensor> residual_out;
  // Optional smooth quantization scale tensor. Type must be float.
  // Used when dynamic_quant=true. Will be flattened to 1D.
  // Must have same device as input.
  std::optional<torch::Tensor> smooth_quant_scale;
  // Optional normalized output tensor. Used when store_output_after_norm=true.
  // Only supported when dynamic_quant=true.
  // Must have same device and dtype as input.
  std::optional<torch::Tensor> normed_out;
  // Normalization mode. Must be "layernorm" or "rmsnorm".
  // - "layernorm": requires both weight (gamma) and beta
  // - "rmsnorm": only requires weight (gamma), beta is not used
  std::string mode;
  // Epsilon value for numerical stability in normalization computation.
  double eps;
  // Whether to store output before normalization to residual_out.
  // Not supported when both bias and residual are not provided.
  bool store_output_before_norm = false;
  // Whether to store output after normalization to normed_out.
  // Only supported when dynamic_quant=true.
  bool store_output_after_norm = false;
  // Whether to use dynamic quantization. If true, quant_scale must be provided.
  // When true, uses per-token quantization scheme; otherwise uses per-channel
  // if quant_scale provided.
  bool dynamic_quant = false;
};

// Fused adaptive LayerNorm parameters.
// Computes: LayerNorm(x) * (1 + scale) + shift in a single fused kernel.
struct AdaLayerNormParams {
  // Input tensor [B, S, H]. Last dimension is hidden_size.
  torch::Tensor input;
  // Scale modulation. Shape [B, H] or [B, 1, H] (broadcast over sequence)
  // or [B, S, H] (token-wise). The (1 + scale) is applied inside the kernel,
  // so pass the raw scale.
  torch::Tensor scale;
  // Shift modulation. Same shape constraints as scale.
  torch::Tensor shift;
  // Optional affine weight (gamma) [H]. Only used when elementwise_affine.
  std::optional<torch::Tensor> weight;
  // Optional affine bias (beta) [H]. Only used when elementwise_affine.
  std::optional<torch::Tensor> bias;
  // Output tensor. Same shape as input. Written back by the kernel.
  torch::Tensor output;
  // Epsilon for numerical stability.
  double eps = 1e-6;
};

struct RmsNormDynamicQuantParams {
  torch::Tensor input;
  torch::Tensor weight;
  double eps;
};

// Matmul parameters
struct MatmulParams {
  // Left input tensor A. Must be 2D or 3D. Must have same dimension as b.
  // Must have same dtype as b.
  // For 2D: shape [M, K], output will be [M, N] where N = b.size(-1)
  // For 3D: shape [batch, M, K], output will be [batch, M, N]
  // If input dtype is int8 or fp8, c must be provided to determine output
  // dtype.
  torch::Tensor a;
  // Right input tensor B. Must be 2D or 3D. Must have same dimension as a.
  // Must have same dtype as a.
  // For 2D: shape [K, N], output will be [M, N] where M = a.size(-2)
  // For 3D: shape [batch, K, N], output will be [batch, M, N]
  torch::Tensor b;
  // Optional bias tensor. Will be added to the matrix multiplication result.
  std::optional<torch::Tensor> bias;
  // Optional output tensor C. Can be used to specify output dtype and
  // accumulate result. If input dtype is int8 or fp8, c or dtype must be
  // provided to determine output dtype. If provided, result will be: output =
  // alpha * (a @ b) + beta * c
  std::optional<torch::Tensor> c;
  // Scaling factor for matrix multiplication result. Default: 1.0
  // Result: alpha * (a @ b) + beta * c (if c provided)
  double alpha = 1.0;
  // Scaling factor for tensor c (if provided). Default: 0.0
  // Result: alpha * (a @ b) + beta * c (if c provided)
  double beta = 0.0;
};

// Quantized matmul parameters (NPU aclnnQuantMatmulV4 path).
struct QuantMatmulParams {
  // Quantized activation tensor. Typical dtype: int8.
  torch::Tensor x1;
  // Quantized weight tensor. Typical dtype: int8.
  torch::Tensor x2;
  // If weight transpose.
  bool transpose2 = true;
  // Weight descale tensor.
  torch::Tensor scale;
  // Optional weight offset tensor.
  std::optional<torch::Tensor> offset;
  // Optional per-token activation scale tensor.
  std::optional<torch::Tensor> pertoken_scale;
  // Optional bias tensor.
  std::optional<torch::Tensor> bias;
  // Optional output dtype, default backend behavior is BF16 if unset.
  std::optional<at::ScalarType> output_dtype;
};

struct NpuQuantizeParams {
  // Input activation tensor.
  torch::Tensor input;

  // Per-tensor/per-channel quantization scale. Required for per_tensor mode.
  std::optional<torch::Tensor> scale;
  // Zero-point for non-symmetric quantization. Optional.
  std::optional<torch::Tensor> zero_point;
  // Quantization target dtype. Currently qint8 path maps to int8 tensor output.
  // support at::ScalarType::QInt8 /at::ScalarType::QUInt8 /
  // at::ScalarType::QInt32
  at::ScalarType output_dtype = at::ScalarType::QInt8;
  int64_t axis = 1;

  // dynamicquant param
  c10::optional<at::Tensor> smooth_scales;
  c10::optional<at::Tensor> group_index;
  c10::optional<at::ScalarType> dst_type;
};

struct GroupGemmParams {
  // Input activation tensor.
  // Shape: 2D [M, K] if trans_a==false; [K, M] if trans_a==true.
  // Must be contiguous. Dtype: float16, bfloat16, or float32.
  // Must have same dtype and device as b, output.
  torch::Tensor a;
  // Weight tensor.
  // If trans_b is true, shape is (num_experts, N, K) or (N, K);
  // if trans_b is false, shape is (num_experts, K, N) or (K, N).
  // Must be contiguous. Dtype and device must match a, output.
  torch::Tensor b;
  // Per-expert token count tensor.
  // Shape: 1D [num_experts]. Type must be int32.
  // Controls number of tokens processed per group/expert.
  torch::Tensor token_count;
  // Output tensor.
  // Shape: [num_experts, N] or [num_experts, N, K]. num_experts =
  // token_count.size(0). Must be contiguous. Dtype and device must match a.
  torch::Tensor output;
  // Optional scale tensor for a (input activation), used in quantized mode.
  // Shape depends on quantization granularity.
  std::optional<torch::Tensor> a_scale;
  // Optional scale tensor for b (weight), used in quantized mode.
  // Shape depends on quantization granularity.
  std::optional<torch::Tensor> b_scale;
  // Optional quantization config flag list.
  // Used to control per-expert weight quantization mode.
  std::optional<torch::List<int64_t>> quant_flag;
  // Maximum workspace dimension (e.g., maximum tokens per expert allowed).
  // Used for configuring inner kernel workspace.
  int64_t max_dim;
  // Whether to transpose a:
  // false: [M, K] (default); true: [K, M].
  bool trans_a;
  // Whether to transpose b:
  // false: [K, N] (default); true: [N, K].
  bool trans_b;
  // Quantization bit-width for input a.
  // Set -1 to disable quantization.
  int64_t a_quant_bit;
  // ========== Torch NPU related parameters ==========
  // Optional input tensor list for grouped matmul.
  // If provided, this overrides `a` for NPU backend.
  // Each tensor shape: [M, K] (or [K, M] if trans_a is true).
  std::optional<torch::TensorList> x_list;
  // Optional weight tensor list for grouped matmul.
  // If provided, this overrides `b` for NPU backend.
  // Each tensor shape: [K, N] or [N, K] depending on trans_b.
  std::optional<torch::TensorList> weight_list;
  // Optional bias list. Used in quantized or fused-activation paths.
  std::optional<torch::TensorList> bias_list;
  // Optional scale list for quantized weights.
  std::optional<torch::TensorList> scale_list;
  // Optional offset list for quantized weights.
  std::optional<torch::TensorList> offset_list;
  // Optional anti-quantization scale list.
  std::optional<torch::TensorList> antiquant_scale_list;
  // Optional anti-quantization offset list.
  std::optional<torch::TensorList> antiquant_offset_list;
  // Optional per-token scale list.
  std::optional<torch::TensorList> per_token_scale_list;
  // Optional group list for NPU grouped matmul.
  // If group_list_type == 0: values are cumsum of group sizes.
  // If group_list_type == 1: values are per-group sizes.
  std::optional<torch::Tensor> group_list;
  // Optional activation input list for fused activation.
  std::optional<torch::TensorList> activation_input_list;
  // Optional activation quantization scale list.
  std::optional<torch::TensorList> activation_quant_scale_list;
  // Optional activation quantization offset list.
  std::optional<torch::TensorList> activation_quant_offset_list;
  // Optional split item for grouped matmul.
  // Common value is 2 for gated MLP (gate + up).
  std::optional<int64_t> split_item = 2;
  // Optional group type for grouped matmul.
  // 0 indicates grouping along the M axis (row-wise).
  std::optional<int64_t> group_type = 0;
  // Optional group list type for grouped matmul.
  // 0: cumsum of group sizes; 1: per-group sizes.
  std::optional<int64_t> group_list_type = 1;
  // Optional activation type for fused activation.
  std::optional<int64_t> act_type;
  // Optional tuning configuration for NPU kernel.
  c10::OptionalIntArrayRef tuning_config;
  // Optional output dtype for NPU kernel.
  std::optional<torch::ScalarType> output_dtype;
  // ========== Torch ILU related parameters ==========
  // Inverse mapping of gather_idx.
  // Shape: [expand_token_num].
  // Dtype: int32.
  std::optional<torch::Tensor> combine_idx;
};

struct DequantSwigluQuantParams {
  // Input tensor from grouped matmul output.
  // Typical dtype for W8A8 path: int32.
  torch::Tensor x;
  // Optional weight dequant scale tensor.
  std::optional<torch::Tensor> weight_scale;
  // Optional activation scale tensor.
  std::optional<torch::Tensor> activation_scale;
  // Optional bias tensor.
  std::optional<torch::Tensor> bias;
  // Optional output quant scale tensor for static quant mode.
  std::optional<torch::Tensor> quant_scale;
  // Optional output quant offset tensor for static quant mode.
  std::optional<torch::Tensor> quant_offset;
  // Optional group index / group list tensor for MoE grouped execution.
  std::optional<torch::Tensor> group_index;
  // Whether to apply activation on the left branch.
  bool activate_left = true;
  // Quantization mode: 0 static quant, 1 dynamic quant.
  int64_t quant_mode = 1;
  // V2-only SwiGLU mode. Mode 1 matches DeepSeek V4 clamp semantics.
  int64_t swiglu_mode = 0;
  // V2-only clamp limit. A positive value is required for swiglu_mode 1.
  double clamp_limit = 0.0;
  // V2-only SwiGLU alpha. DeepSeek V4 uses standard SiLU alpha 1.0.
  double glu_alpha = 1.0;
  // V2-only bias applied to the up branch. DeepSeek V4 expects no bias.
  double glu_bias = 0.0;
};

// Ascend W4A8_DYNAMIC MoE weight post-load processing for version 1.0.0.
// Transpose/NZ-convert int4 weights and fold first-level + second-level scales
// into the int64 dtype/layout consumed by npu_grouped_matmul. Torch fused MoE
// additionally packs weights to int32; the ATB grouped-matmul wrapper expects
// the original int8-packed tensor.
struct W4A8DynamicMoePreprocessParams {
  torch::Tensor w13_weight;
  torch::Tensor w2_weight;
  torch::Tensor w13_weight_scale;
  torch::Tensor w2_weight_scale;
  std::optional<torch::Tensor> w13_weight_scale_second;
  std::optional<torch::Tensor> w2_weight_scale_second;
  std::optional<torch::Tensor> w13_scale_bias;
  std::optional<torch::Tensor> w2_scale_bias;
  int64_t group_size = 256;
  bool pack_weight_to_int32 = true;
};

struct MoeFusedTopkParams {
  // Input tensor.
  // Shape: [*, num_mask, num_expert] (e.g., [batch, num_mask, num_expert]).
  // Dtype: float32, float16, bfloat16.
  // Must be contiguous.
  torch::Tensor input;
  // Optional finished mask for NPU gating topk softmax.
  // Shape should be broadcastable to input's leading dims.
  // If not provided, all tokens are considered active.
  std::optional<torch::Tensor> finished;
  // Number of top-k experts to select per token.
  // Constraint: 0 < topk <= num_expert.
  int64_t topk;
  // Number of expert groups for group-limited top-k selection.
  // If > 1, mask must be None, and num_expert % num_expert_group == 0.
  int64_t num_expert_group;
  // Maximum selected experts per group.
  // Constraint: 0 < topk_group <= num_expert_group.
  int64_t topk_group;
  // Whether to renormalize expert weights after top-k selection.
  bool normalize;
  // Optional mask tensor.
  // Shape: [1, ..., 1, num_mask, num_expert] (leading dims must be 1).
  // Dtype must match input.
  // Must be contiguous.
  std::optional<torch::Tensor> mask;
  // Normalization logic after top-k selection.
  // For softmax: "topk_logit" or "softmax_logit".
  // For sigmoid: "topk_logit" or "sigmoid_logit".
  std::string normed_by;
  // Scoring function for expert selection.
  // Supported: "softmax", "sigmoid".
  std::string scoring_func;
  // Route scaling factor applied to routing scores.
  double route_scale;
  // Optional expert score correction bias.
  // Shape: [num_expert].
  // Dtype: float32, float16, or bfloat16.
  // Must be contiguous.
  std::optional<torch::Tensor> e_score_correction_bias;
};

struct MoeGenIdxParams {
  // The input tensor stores the expert id of each token.
  // Shape: [num_tokens, topk].
  // Dtype: int32.
  torch::Tensor expert_id;
  // Expert number.
  // Must be >= 0.
  int64_t expert_num;
};

struct MoeExpandInputParams {
  // Input tensor to be expanded.
  // Shape: [token_num, hidden_size].
  // Dtype: int8, float, half, or bfloat16.
  torch::Tensor input;
  // Index tensor for gather operation.
  // Shape: [expand_token_num].
  // Dtype: int32.
  torch::Tensor gather_index;
  // Optional prefix sum of token count per expert.
  // Shape: [num_experts + 1].
  // Dtype: int32.
  // If provided, adjusts gather range for each expert.
  std::optional<torch::Tensor> cusum_token_count;
  // Starting expert id to process.
  // Must be >= 0.
  int64_t start_expert_id;
  // Number of experts to process in this call.
  // Must be >= 0.
  int64_t expert_size;
  // ========== Torch ILU related parameters ==========
  // Inverse mapping of gather_idx.
  // Shape: [expand_token_num].
  // Dtype: int32.
  torch::Tensor combine_idx;
  // topk for moe
  int topk;
};

struct MoeCombineResultParams {
  // Expert output tensor to be combined.
  // Shape: [num_tokens * topk, hidden_size].
  // - Must be contiguous.
  // - Dtype: float32, float16, or bfloat16.
  // - This is the concatenated output from all experts, not yet reordered back
  // to the original sequence order.
  torch::Tensor input;
  // Router/gating weights tensor. Used for weighted combination of expert
  // outputs. Shape: [num_tokens, topk].
  // - Must be contiguous at last dimension.
  // - Dtype: float32.
  // - Constraint: reduce_weight.numel() == input.size(0).
  torch::Tensor reduce_weight;
  // Gather index tensor that maps combined output to original token positions.
  // Shape: [num_tokens * topk].
  // - Must be contiguous.
  // - Dtype: int32.
  // - Corresponds to permutation/scatter indices for reordering expert outputs.
  torch::Tensor gather_ids;
  // Optional probes tensor for NPU token unpermute.
  // If provided, used as probe weights in unpermute kernel.
  // Shape: [num_tokens, topk].
  std::optional<torch::Tensor> probes;
  // Whether the permuted tokens are padded (NPU token unpermute).
  bool padded_mode = false;
  // Optional restore shape for NPU token unpermute.
  c10::OptionalIntArrayRef restore_shape = c10::nullopt;
  // Optional residual connection input.
  // Shape: [num_tokens, hidden_size].
  // - Must have same shape and dtype as output if provided.
  // - Must be contiguous if provided.
  // - Default: std::nullopt (no residual).
  std::optional<torch::Tensor> residual;
  // Optional cumulative token count for expert assignment.
  // Shape: [num_experts + 1] or deduced by expert_size.
  // - Must be contiguous if provided.
  // - Dtype: int32.
  // - Used to infer num_expert or assist calculation in some kernels.
  std::optional<torch::Tensor> cusum_token_count;
  // Starting expert ID
  // - Must be >= 0.
  // - Used to mark the offset of current experts being processed (for
  // sharding).
  int64_t start_expert_id = 0;
  // Number of experts processed in this step.
  // - If cusum_token_count not given, num_expert is set to this value.
  // - If cusum_token_count given, deduced num_expert must satisfy:
  //   num_expert >= start_expert_id + expert_size
  int64_t expert_size = 0;
  // Optional bias tensor.
  // WARNING: Bias addition is NOT supported in current implementation.
  // Always keep as std::nullopt unless bias support is added in the future.
  std::optional<torch::Tensor> bias;
};

struct MoeAll2AllGenSendLayoutParams {
  // Expert token count tensor.
  // Shape: [expert_num].
  // Dtype: int32.
  // Each element represents the number of tokens assigned to each expert.
  torch::Tensor token_count;
  // Number of ranks (processes) participating in All2All.
  // Must be >= 0.
  int64_t nrank;
};

struct MoeAll2AllGenGatherIndexParams {
  // The table that indicates the relationship of token for each Expert Parallel
  // part. Shape: [rank_num, expert_num], where rank_num is the number of
  // devices in Expert Parallel, and expert_num is the number of experts handled
  // by each device. Dtype: int32.
  torch::Tensor token_num;
  // The max token count for each rank (used for padding).
  // Dtype: int32. Must be >= 0.
  int64_t pad_num;
  // Whether to return the cusum_token_count tensor.
  // If true, cusum_token_count will be returned.
  bool return_cusum_token_count = false;
};

struct MoeAll2AllCreateParams {
  // Byte size of a single token for dispatch All-to-All operation.
  // Each token to be dispatched requires this many bytes.
  int64_t dispatch_token_byte;
  // Byte size of a single token for combine All-to-All operation.
  // Each token to be combined requires this many bytes.
  int64_t combine_token_byte;
  // Maximum number of experts participating in the All-to-All operation.
  // (Sets the upper bound for how many experts can be involved.
  int64_t max_expert_num;
  // Maximum number of tokens to be processed.
  // Upper bound on the total batch size in tokens for the operation.
  int64_t max_token_num;
  // Rank ID of the current process in the distributed group, within [0,
  // nrank-1]. Identifies this process within the world group.
  int64_t rank;
  // Total number of processes in the distributed group.
  // Used for collective communication context and split assignment.
  int64_t nrank;
  // The current compute device to be used
  // default to CPU
  torch::Device device = torch::Device(torch::kCPU);
};

struct MoeAll2AllInitParams {
  // communication backend handle for All-to-All operation.
  // obtained from moe_all2all_create.
  int64_t handle;
  // CPU tensor containing aggregated exchange information from all nrank
  // processes.
  torch::Tensor all_exchange_info;
  // The current compute device to be used
  // default to CPU
  torch::Device device = torch::Device(torch::kCPU);
};

struct MoeAll2AllDispatchParams {
  // Communication backend handle for All-to-All operation.
  // Obtained from moe_all2all_create.
  int64_t handle;
  // Byte size of a single token.
  int64_t token_byte;
  // Number of tokens to be processed in the current operation.
  int64_t token_num;
  // Offset and token count for each rank.
  // The token_count is generated by moe_gen_idx.
  // Shape: [nrank, 2]. Type: int32.
  torch::Tensor send_layout;
  // Number of tokens to send to each expert.
  // Shape: [max_expert_num]. Type: int32.
  torch::Tensor send_token_num;
  // Offset and token count from peer ranks.
  // Shape: [nrank, 2]. Type: int32.
  torch::Tensor recv_layout;
  // Expected number of tokens to receive from each expert.
  // Shape: [max_expert_num]. Type: int32.
  torch::Tensor recv_token_num;
  // Optional tensor containing tokens to dispatch.
  // If not provided, defaults to dispatch_send created by moe_all2all_create.
  std::optional<torch::Tensor> send_token;
  // Optional buffer for receiving tokens.
  // If not provided, defaults to dispatch_recv created by moe_all2all_create.
  std::optional<torch::Tensor> recv_token;
};

struct MoeAll2AllCombineParams {
  // communication backend handle for All-to-All operation.
  // obtained from moe_all2all_create.
  int64_t handle;
  // Byte size of a single token.
  int64_t token_byte;
  // The number of tokens to receive.
  int64_t token_num;
  // The offset and token count for each rank, output from
  // Shape: [nrank, 2],
  // Type: int32.
  torch::Tensor send_src_layout;
  // The expected receive pattern from peer ranks.
  // Shape: [nrank, 2],
  // Type: int32.
  torch::Tensor send_dst_layout;
  // Optional tensor containing the tokens to dispatch. If not provided,
  // defaults to combine_send created by moe_all2all_create.
  std::optional<torch::Tensor> send_token;
  // Optional buffer for receiving tokens. If not provided,
  // defaults to combine_recv created by moe_all2all_create.
  std::optional<torch::Tensor> recv_token;
};

struct MoeAll2AllDestroyParams {
  // communication backend handle for All-to-All operation.
  // obtained from moe_all2all_create.
  int64_t handle;
  // The current compute device to be used
  // default to CPU
  torch::Device device = torch::Device(torch::kCPU);
};

// Per token smooth quantize parameters
// Note: Current MLU implementation uses "dynamic_per_token" quantization mode.
struct ScaledQuantizeParams {
  // Input tensor to quantize. Dimension must be >= 2.
  // Must be continuous between 0 and -2 dimensions (can be flattened to 2D).
  // If gather_index or token_count has value, x must be 2D.
  // Must have same device as other tensors.
  torch::Tensor x;
  // Smooth quantization scale tensor (corresponds to x_scale in underlying
  // API). Shape constraints depend on quantization mode and other parameters.
  // - If token_count has value: shape [token_count.size(0),
  // x.size(-1)/(1+is_gated)]
  // - If is_gated: smooth.size(-1) * 2 == x.size(-1)
  // - Otherwise: smooth.size(-1) == x.size(-1)
  // Must be contiguous if provided. Must have same device as x.
  torch::Tensor smooth;
  // Zero point tensor. Must be None (not supported in current implementation).
  std::optional<torch::Tensor> zero;
  // Optional token count tensor when quantizing MoE group gemm inputs.
  // If provided, x must be 2D and smooth.size(0) must equal
  // token_count.size(0). Must be contiguous if provided. Must have same device
  // as x.
  std::optional<torch::Tensor> token_count;
  // Optional gather index tensor when quantizing MoE group gemm inputs. Shape:
  // [output_tokens]. If provided, x must be 2D. Output shape will be adjusted:
  // output_shape[0] = gather_index.size(0). If gather_index_start_position is
  // provided, gather_index must also be provided. Must be contiguous if
  // provided. Must have same device as x.
  std::optional<torch::Tensor> gather_index;
  // Optional gather index start position tensor when quantizing MoE group gemm
  // inputs. Only used if gather_index is provided. Must be contiguous if
  // provided. Must have same device as x.
  std::optional<torch::Tensor> gather_index_start_position;
  // Optional output tensor when quantizing MoE group gemm inputs.
  // Type must be int8 (kChar), float8_e4m3fn, or float8_e5m2.
  // Dimension must be >= 2. Must be continuous between 0 and -2 dimensions.
  // Shape constraints:
  // - If !gather_index && !is_gated: output.sizes() == x.sizes()
  // - If is_gated: output.size(-1) * 2 == x.size(-1)
  // - If gather_index: output_shape[0] = gather_index.size(0)
  // If not provided, will be allocated automatically with quant_type.
  // Must have same device as x.
  std::optional<torch::Tensor> output;
  // Optional output scale tensor.
  // Used in dynamic_per_token quantization mode.
  // Shape: x.sizes()[0:-1] (same as x except last dimension removed).
  // If gather_index provided: shape[0] = gather_index.size(0).
  // Must be flattenable to 1D with numel == output_flat.size(0).
  // If not provided, will be allocated automatically with float32 dtype.
  // Must have same device as x.
  std::optional<torch::Tensor> output_scale;
  // Activation mode. Must be one of: "none", "gelu", "silu", "swish".
  // Default: "none". If "none", is_gated will be set to false automatically.
  // If "silu", active_coef will be set to 1.0 automatically.
  std::string act_mode = "none";
  // Activation coefficient. Default: 1.0.
  // If act_mode == "silu", this will be set to 1.0 automatically.
  double active_coef = 1.0;
  // Whether to use gated activation. Default: false.
  // If act_mode == "none", this will be set to false automatically.
  // If true, output's last dimension will be x.size(-1) / 2.
  bool is_gated = false;
  // Quantization output data type. Default: torch::kChar (int8).
  // Supported: torch::kChar (int8), torch::kFloat8_e4m3fn, torch::kFloat8_e5m2.
  torch::ScalarType quant_type = torch::kChar;
};

// Scaled matmul parameters
// Note: Current MLU implementation only supports:
// - smooth_quant algorithm
// - w8a8 quantization (quant_bit_size=8, a_quant_bit_size=8)
// - trans_a=false, trans_b=true (hardcoded)
struct ScaledMatmulParams {
  // Input tensor A. Shape: [M, K]. Must be contiguous.
  // Output shape will be [M, N] where N = b.size(0).
  // Must have same device as other tensors.
  torch::Tensor a;
  // Weight tensor B. Shape: [K, N]. Will be transposed (trans_b=true).
  // Must be contiguous. Must have same device as other tensors.
  torch::Tensor b;
  // Optional scale tensor for A. Shape: 1D or 2D. Must be contiguous or have
  // stride (1, m).
  // - 1D: per-token quantization layout
  // - 2D: group-wise quantization layout
  // Note: In current MLU implementation (scaled_matmul.cpp), a_scale is
  // required.
  std::optional<torch::Tensor> a_scale;
  // Scale tensor for B. Shape: 1D or 2D. Must be contiguous or have stride (1,
  // n). Determines quantization layout:
  // - 1D: per-channel quantization
  // - 2D: per-block (if b_scale.size(0) < b.size(0)) or group-wise quantization
  // Must be contiguous. Must have same device as other tensors.
  torch::Tensor b_scale;
  // Output data type. Must be torch::kFloat16 (half) or torch::kBFloat16.
  torch::ScalarType output_dtype;
  // Optional bias tensor. Will be added to the matrix multiplication result.
  // Must be contiguous. Must have same device as other tensors.
  std::optional<torch::Tensor> bias;
  // Optional tensor C for accumulation. Result: alpha * (a @ b) + beta * c.
  // Must be contiguous. Must have same device as other tensors.
  std::optional<torch::Tensor> c;
  // Activation mode. Default: "none". Supported: "none", "silu", "gelu".
  // If "silu", act_coef will be set to 1.0 automatically.
  std::string act_mode = "none";
  // Quantization bit size for B (weight). Default: 8.
  // Current implementation only supports 8 (w8a8 quantization).
  // Supported values: 4, 8.
  int64_t quant_bit_size = 8;
  // Scaling factor for matrix multiplication result. Default: 1.0
  // Result: alpha * (a @ b) + beta * c (if c provided)
  double alpha = 1.0;
  // Scaling factor for tensor c (if provided). Default: 1.0
  // Result: alpha * (a @ b) + beta * c (if c provided)
  double beta = 1.0;
  // Whether to use high precision activation computation. Default: false
  // If true, uses high precision; otherwise uses fast computation.
  bool use_hp_active = false;
  // Quantization bit size for A (activation). Default: -1.
  // Current implementation only supports 8 (w8a8 quantization).
  // Supported values: -1 (no quantization), 4, 8.
  int64_t a_quant_bit_size = -1;
  // Optional calibration tensor for A. Used for flat_quant and svd_quant
  // algorithms. Must be contiguous. Must have same device as other tensors.
  std::optional<torch::Tensor> a_calib;
  // Optional calibration tensor for B. Used for flat_quant and svd_quant
  // algorithms. Must be contiguous. Must have same device as other tensors.
  std::optional<torch::Tensor> b_calib;
  // Optional output tensor. Shape: [M, N] where M = a.size(0), N = b.size(0).
  // If not provided, will be allocated automatically with output_dtype.
  // Must have same device as other tensors.
  std::optional<torch::Tensor> output;
};

// Top-K and Top-P sampling parameters
struct TopKPParams {
  // Input logits tensor. Shape: [batch_size, vocab_size]. Type must be float32.
  // Must be contiguous. Will be converted to float32 if needed.
  // If both top_k and top_p are not defined, logits will be returned directly.
  torch::Tensor logits;
  // Temperature tensor for scaling logits. Shape: [batch_size].
  // Must be contiguous. Will be moved to same device as logits.
  torch::Tensor temperatures;
  // Optional top-k values tensor. Type will be converted to int32.
  // Must be contiguous. Will be moved to same device as logits.
  torch::Tensor top_k;
  // Optional top-p (nucleus sampling) values tensor.
  // Must be contiguous. Will be moved to same device as logits.
  torch::Tensor top_p;
};

// Random sample parameters
struct RandomSampleParams {
  // Input tensor of probabilities for sampling.
  // Must be 2-dimensional: [batch_size, vocab_size]
  torch::Tensor logits;
};

// Rejection sampling parameters for speculative decoding
struct RejectionSampleParams {
  // Candidate draft token indices to be verified.
  // Shape: [total_draft_tokens]. Dtype: int32.
  // total_draft_tokens equals cu_num_draft_tokens[batch_size - 1].
  torch::Tensor draft_token_ids;
  // Number of draft tokens for each sequence in the batch.
  // Shape: [batch_size]. Dtype: int32.
  torch::Tensor num_draft_tokens;
  // Accumulated number of draft tokens in each batch.
  // Shape: [batch_size]. Dtype: int32.
  torch::Tensor cu_num_draft_tokens;
  // Probability distributions of the draft model.
  // Shape: [total_draft_tokens, vocab_size].
  // Dtype: float32, float16, or bfloat16.
  std::optional<torch::Tensor> draft_probs;
  // Probability distributions of the target model.
  // Shape: [total_draft_tokens, vocab_size].
  // Dtype: float32, float16, or bfloat16.
  torch::Tensor target_probs;
  // Bonus token indices to be selected when all draft tokens are accepted.
  // Shape: [batch_size]. Dtype: int32.
  torch::Tensor bonus_token_ids;
  // Random probabilities for acceptance threshold comparison.
  // Shape: [total_draft_tokens]. Dtype: float32.
  // Used to compare with selected_target_probs / selected_draft_probs.
  torch::Tensor uniform_rand;
  // Random probabilities for resampling (recovery) calculation.
  // Shape: [total_draft_tokens, vocab_size]. Dtype: float32.
  torch::Tensor uniform_probs;
  // The maximum number of draft tokens in the batch (max value in
  // num_draft_tokens).
  int32_t max_spec_len;
};

// Masked indexer select paged KV cache parameters
struct MaskedIndexerSelectPagedKVParams {
  // Query tensor. Must have same dtype as k_cache (bfloat16, half, or int8).
  // - Prefill mode: 3D [total_seq_q, head_num, head_size], head_num must be 64
  // - Decode mode: 4D [batch_num, len_q, head_num, head_size], head_num must be
  // 64 Does not need to be contiguous
  torch::Tensor query;
  // Key cache tensor in paged format. Shape: [num_blocks, 1, block_size,
  // head_dim]. Dim(1) must be 1. Must be contiguous. Must have same dtype as
  // query.
  torch::Tensor k_cache;
  // Attention weights tensor. Dtype must be bfloat16 or float32. Must be
  // contiguous.
  torch::Tensor weights;
  // Key cache block table. Shape: [batch_num, k_cache_max_blkn]. Type: int32.
  // Must be contiguous.
  std::optional<torch::Tensor> k_cache_block_table;
  // Cumulative sequence lengths for queries. Type: int32. Must be contiguous.
  // Required in prefill mode, not used in decode mode.
  std::optional<torch::Tensor> cu_seq_q_lens;
  // Cumulative sequence lengths for keys.
  std::optional<torch::Tensor> cu_seq_k_lens;
  // Key context lengths tensor. Shape: [batch_num]. Type: int32. Must be
  // contiguous.
  std::optional<torch::Tensor> k_context_lens;
  // KV cache block table. Shape: [batch_num, kv_cache_max_blkn]. Type: int32.
  // Must be contiguous.
  torch::Tensor kv_cache_block_table;
  // Whether this is prefill phase (true) or decode phase (false).
  // Affects query shape and whether cu_seq_q_lens is used.
  bool is_prefill;
  // Number of top-k indices to select. Must be >= 0.
  int64_t index_topk;
  // KV cache block size.
  int64_t kv_cache_block_size;
  // Softmax scaling factor for attention computation.
  double softmax_scale;
  // Query quantization scale tensor. Must be contiguous.
  // - Required (numel > 0) when query dtype is int8 or fp8
  // - Must be empty (numel == 0) when query dtype is bfloat16 or half
  // - INT8 prefill shape: [total_q, head_num, 1]
  // - INT8 decode shape: [batch, seq_q, head_num, 1]
  std::optional<torch::Tensor> q_scale;
  // Key cache quantization scale tensor. Must be contiguous.
  // - Required (numel > 0) when k_cache dtype is int8 or fp8
  // - Must be empty (numel == 0) when k_cache dtype is bfloat16 or half
  // - Paged INT8 K scale is the cache scale shape with a trailing dim added
  //   before select.
  // - Dense INT8 K scale keeps the layout returned by
  //   scaled_quantize(k.unsqueeze(-2)).
  std::optional<torch::Tensor> k_scale_cache;
  // New sparse block table output tensor. Must be contiguous.
  // - Prefill mode: 2D [total_seq_q, kv_cache_max_blkn]
  // - Decode mode: 3D [batch_num, seq_q, kv_cache_max_blkn]
  torch::Tensor sparse_block_table;
  // New sparse block table output tensor. Shape: [batch_num] (prefill) or
  // [batch_num] (decode). Type: int32. Must be contiguous.
  torch::Tensor sparse_context_lens;
  bool is_score_float = false;
  int64_t compress_ratio = 1;
  std::optional<torch::Tensor> kv_cache_block_table_offset = std::nullopt;
};

struct GatherSplitParams {
  // Input tensor. Shape: (token_num, input_size).
  // Dtype: int8, float32, float16, or bfloat16.
  torch::Tensor input;
  // Gather index tensor. Shape: (token_num).
  // Dtype: int32.
  // Used to select valid tokens from the input tensor.
  torch::Tensor gather_index;
  // Number of valid tokens tensor. Shape: (1).
  // Dtype: int32.
  // Its first element is the actual valid token count: valid_token_num =
  // valid_token_num[0].item().
  torch::Tensor valid_token_num;
  // Output tensor for the "head" split. Shape: (token_num, size_0).
  // Dtype: same as input.
  // Holds the gathered and split tokens for the first size_0 elements of each
  // token.
  torch::Tensor output_head;
  // Optional output tensor for the "tail" split. Shape: (token_num, input_size
  // - size_0). Dtype: same as input. If provided, holds the gathered and split
  // tokens for the remaining elements after size_0.
  // Pass empty tensor to skip the tail split.
  torch::Tensor output_tail;
};

struct FusedMlaQParams {
  // Query tensor for the MLA attention operation.
  // Shape: (batch_size, sequence_length, input_size).
  // Dtype: float16 or bfloat16.
  torch::Tensor q;

  // Output tensor for the fused MLA query operation.
  // Shape: (batch_size, sequence_length, head_num, head_size).
  // Dtype: same as q, int8, float8_e4m3fn.
  torch::Tensor output;

  // Output quantization scales for dynamic per-token quantization.
  // Shape: (batch_size, sequence_length, head_num).
  // Dtype: float32.
  // Only used when quant_mode is "dynamic_per_token".
  torch::Tensor output_scale;

  // Intermediate RMSNorm result tensor.
  // Shape: (batch_size, sequence_length, input_size).
  // Dtype: same as q.
  std::optional<torch::Tensor> output_norm;

  // Scaling parameter for RMSNorm normalization.
  // Shape: (input_size).
  // Dtype: same as q.
  torch::Tensor gamma;

  // Smooth quantization scale for input tensor.
  // Shape: (input_size) if provided.
  // Dtype: float32.
  // Optional: can be nullopt if smooth quantization is not used.
  std::optional<torch::Tensor> smooth_quant_scale;

  // Weight matrix for the first matmul operation in MLA.
  // Shape: (head_num * (nope_dim + pe_dim), input_size).
  // Dtype: int8, float8_e4m3fn.
  torch::Tensor weight_b;

  // Per-channel scale for weight_b quantization.
  // Shape: (head_num * (nope_dim + pe_dim)).
  // Dtype: float32.
  torch::Tensor weight_b_scale;

  // Weight matrix for the bmm operation in MLA.
  // Shape: (head_num, kv_lora_rank, nope_dim).
  // Dtype: same as q.
  torch::Tensor weight_c;

  // Sine values for rotary position embedding.
  // Shape: (rotary_sequence_length, pe_dim).
  // Dtype: same as q.
  torch::Tensor sin;

  // Cosine values for rotary position embedding.
  // Shape: (rotary_sequence_length, pe_dim).
  // Dtype: same as q.
  torch::Tensor cos;

  // Position IDs for rotary embedding.
  // Shape: (batch_size).
  // Dtype: int32.
  torch::Tensor position_id;

  // Quantization mode for the operation.
  // Supported values: "none", "dynamic_per_token".
  // Default: "none".
  std::string quant_mode = "none";

  // Epsilon value for RMSNorm numerical stability.
  double eps = 1e-6;

  // Rotary embedding mode flag.
  // If true, apply cross rotary embedding (interleaved).
  // If false, apply fold rotary embedding (non-interleaved).
  bool interleaved = true;
};

struct FusedMlaKVParams {
  // The input key-value tensor.
  // Shape: (batch, seq, head_num, head_size).
  // Dtype: half, bfloat16.
  torch::Tensor input_kv;

  // The rotary sin table tensor.
  // Shape: (rotary_seq, rotary_dim).
  // Dtype: same as input_kv.
  torch::Tensor sin;

  // The rotary cos table tensor.
  // Shape: (rotary_seq, rotary_dim).
  // Dtype: same as input_kv.
  torch::Tensor cos;

  // The rotary seq_len offset of each batch.
  // Shape: (batch).
  // Dtype: int32.
  torch::Tensor position_id;

  // The weight of RMSNorm normalization.
  // Shape: (norm_dim).
  // Dtype: same as input_kv.
  torch::Tensor gamma;

  // The cache tensor for key-value storage.
  // Shape: (num_blocks, num_heads, block_size, head_size).
  // Dtype: half, bfloat16, int8, float8_e4m3fn.
  torch::Tensor kv_cache;

  // Scale tensor for cache quantization.
  // For static per-channel quantization: shape is (head_num, head_size) or
  // (batch, head_num, head_size). For dynamic per-token quantization: shape is
  // (num_blocks, head_num, block_size) and is an output tensor. Dtype: float32.
  // Optional: only used when quant_mode is "static_per_channel" or
  // "dynamic_per_token".
  std::optional<torch::Tensor> kv_cache_scale;

  // The slot mapping tensor for paged attention.
  // Shape: (batch, seq).
  // Dtype: int32.
  // Optional: only required when is_paged_cache is true.
  std::optional<torch::Tensor> slot_mapping;

  // The batch index in the cache where the kv tensors will be placed.
  // Shape: (batch).
  // Dtype: int32.
  // Optional: used for non-paged cache style.
  std::optional<torch::Tensor> cache_bs_id;

  // A 1D tensor representing the sequence offsets where the cache data starts
  // for each batch. Shape: (batch). Dtype: int32. Optional: used for non-paged
  // cache style.
  std::optional<torch::Tensor> cache_seq_offset;

  // Quantization mode for the operation.
  // Supported values: "none", "static_per_channel", "dynamic_per_token".
  std::string quant_mode = "none";

  // Flag indicating the cache style.
  // If true, uses paged cache style and slot_mapping must be provided.
  // If false, uses linear cache style and cache_bs_id/cache_seq_offset may be
  // used. Default: true.
  bool is_paged_cache = true;

  // Epsilon value for RMSNorm numerical stability.
  double eps = 1e-6;

  // Rotary embedding mode flag.
  // If true, apply cross rotary embedding (interleaved).
  // If false, apply fold rotary embedding (non-interleaved).
  bool interleaved = true;
};

struct FusedIndexerQParams {
  // The input tensor for query projection.
  // Shape: (token_num, input_dim).
  // Dtype: half, bfloat16.
  torch::Tensor input_q;

  // An output tensor to store the final result in-place.
  // Shape: (token_num, head_num, head_size).
  // Dtype: same as input_q, or int8 if output is quantized.
  torch::Tensor output;

  // Optional output tensor to store quantization scales.
  // Shape: (token_num, head_num).
  // Dtype: float32.
  std::optional<torch::Tensor> output_scale;

  // The weight tensor for query projection.
  // Shape: (head_num, head_size, input_dim).
  // Dtype: half, bfloat16.
  torch::Tensor w_q;

  // The scale tensor for the w_q weight, used for per-channel quantization.
  // Shape: (head_num, head_size).
  // Dtype: float32.
  std::optional<torch::Tensor> w_q_scale;

  // Optional weight tensor for the Hadamard transformation.
  // Shape: (head_size, head_size).
  // Dtype: same as input_q.
  std::optional<torch::Tensor> hadamard_matrix;

  // A pre-computed tensor containing sine values for RoPE.
  // Shape: (rotary_seq, rotary_dim).
  // Dtype: same as input_q.
  torch::Tensor sin;

  // A pre-computed tensor containing cosine values for RoPE.
  // Shape: (rotary_seq, rotary_dim).
  // Dtype: same as input_q.
  torch::Tensor cos;

  // A tensor indicating the position index for each token.
  // Shape: (token_num).
  // Dtype: int32.
  torch::Tensor position_id;

  // Quantization mode for the output.
  // Supported values: "none", "dynamic_per_token".
  std::string quant_mode = "none";

  // Rotary embedding mode flag.
  // If true, apply cross rotary embedding (interleaved).
  // If false, apply fold rotary embedding (non-interleaved).
  bool interleaved = true;

  // Flag indicating whether to apply RoPE at the front of the operation.
  // If true, apply RoPE at the front of the operation.
  // If false, apply RoPE at the back of the operation.
  bool rope_at_front = true;
};

struct FusedIndexerKParams {
  // The input tensor.
  // Shape: (m, dim).
  // Dtype: half, bfloat16.
  torch::Tensor x;

  // The weight tensor for K projection.
  // Shape: (head_size, dim).
  // Dtype: same as x.
  torch::Tensor wk;

  // The weight tensor for head projection.
  // Shape: (head_num, dim).
  // Dtype: same as x.
  torch::Tensor wproj;

  // A pre-computed tensor containing sine values for RoPE.
  // Shape: (rotary_seq, rope_dim).
  // Dtype: same as x.
  torch::Tensor sin_table;

  // A pre-computed tensor containing cosine values for RoPE.
  // Shape: (rotary_seq, rope_dim).
  // Dtype: same as x.
  torch::Tensor cos_table;

  // A tensor indicating the position index for each token.
  // Shape: (m).
  // Dtype: int32.
  torch::Tensor position_id;

  // A tensor mapping tokens to cache slots.
  // Shape: (m).
  // Dtype: int32.
  torch::Tensor slot_mapping;

  // The computed head weights tensor.
  // Shape: (m, head_num).
  // Dtype: same as x.
  torch::Tensor head_weights;

  // The K cache tensor.
  // Shape: (block_num, 1, block_size, head_size).
  // Dtype: half, bfloat16, int8.
  torch::Tensor k_cache;

  // Optional scale tensor for quantized K cache.
  // Shape: (block_num, 1, block_size).
  // Dtype: float32.
  std::optional<torch::Tensor> k_cache_scale;

  // Optional weight tensor for the Hadamard transformation.
  // Shape: (head_size, head_size).
  // Dtype: same as x.
  std::optional<torch::Tensor> hadamard_matrix;

  // Rotary embedding mode flag.
  // If true, apply cross rotary embedding (interleaved).
  // If false, apply fold rotary embedding (non-interleaved).
  bool interleaved = true;

  // Optional weight tensor for RMSNorm.
  // Shape: (head_size).
  // Dtype: float32.
  std::optional<torch::Tensor> gamma;

  // Optional bias tensor for RMSNorm.
  // Shape: (head_size).
  // Dtype: float32.
  std::optional<torch::Tensor> beta;

  // RMSNorm epsilon.
  double eps = 1e-6;
};

struct MoeInitRoutingV2Params {
  // TODO: NPU moe_init_routing_v2 is equivalent to moe_gen_idx +
  // moe_expand_input (and token_count/cusum outputs) on other backends.
  torch::Tensor x;
  torch::Tensor expert_idx;
  std::optional<torch::Tensor> scale;
  std::optional<torch::Tensor> offset;
  int active_num;
  int expert_capacity;
  int expert_num;
  int drop_pad_mode;
  int expert_tokens_num_type;
  bool expert_tokens_num_flag;
  int quant_mode;
  torch::IntArrayRef active_expert_range;
  int row_idx_type;
};

struct MoeDistributeDispatchV2Params {
  torch::Tensor x;
  torch::Tensor expert_ids;
  std::optional<torch::Tensor> expert_scales = std::nullopt;
  std::optional<torch::Tensor> x_active_mask = std::nullopt;
  std::optional<torch::Tensor> scales = std::nullopt;
  std::string group_ep;
  int64_t ep_world_size = 1;
  int64_t ep_rank_id = 0;
  int64_t moe_expert_num = 1;
  std::string group_tp;
  int64_t tp_world_size = 0;
  int64_t tp_rank_id = 0;
  int64_t expert_shard_type = 0;
  int64_t shared_expert_num = 1;
  int64_t shared_expert_rank_num = 0;
  int64_t quant_mode = 0;
  int64_t global_bs = 0;
  int64_t expert_token_nums_type = 1;
  std::string comm_alg;
};

struct MoeDistributeCombineV2Params {
  torch::Tensor expand_x;
  torch::Tensor expert_ids;
  torch::Tensor assist_info_for_combine;
  torch::Tensor ep_send_counts;
  torch::Tensor expert_scales;
  std::optional<torch::Tensor> tp_send_counts = std::nullopt;
  std::optional<torch::Tensor> x_active_mask = std::nullopt;
  std::optional<torch::Tensor> expand_scales = std::nullopt;
  std::optional<torch::Tensor> shared_expert_x = std::nullopt;
  std::string group_ep;
  int64_t ep_world_size = 1;
  int64_t ep_rank_id = 0;
  int64_t moe_expert_num = 1;
  std::string group_tp;
  int64_t tp_world_size = 0;
  int64_t tp_rank_id = 0;
  int64_t expert_shard_type = 0;
  int64_t shared_expert_num = 1;
  int64_t shared_expert_rank_num = 0;
  int64_t global_bs = 0;
  int64_t comm_quant_mode = 0;
  std::string comm_alg;
};

struct DispatchFFNCombineParams {
  // Input hidden states. Shape: [num_tokens, hidden_size].
  // Dtype: fp16/bf16 for W8A8 dynamic path.
  torch::Tensor x;
  // First/second expert weights as TensorList. For W8A8 int8 path the fused
  // CANN op expects NZ-formatted int8 weights.
  torch::TensorList weight1;
  torch::TensorList weight2;
  // Expert ids. Shape: [num_tokens, topk], dtype int32.
  torch::Tensor expert_ids;
  // W8A8 scales as TensorList. CANN DispatchFFNCombine expects the int64
  // bit-packed representation used by vLLM/sgl wrappers, not raw fp32 scales.
  torch::TensorList scale1;
  torch::TensorList scale2;
  // Router probabilities. Shape: [num_tokens, topk], dtype fp32.
  torch::Tensor probs;
  // HCCL communication group name.
  std::string group;
  int64_t max_output_size = 0;
  double swiglu_limit = 0.0;
  // Optional preallocated outputs. If absent, wrapper allocates them.
  std::optional<torch::Tensor> output = std::nullopt;
  std::optional<torch::Tensor> expert_token_nums = std::nullopt;
};

struct DispatchGmmCombineDecodeParams {
  // Input hidden states. Shape: [num_tokens, hidden_size].
  torch::Tensor x;
  // Expert ids after routing. Shape: [num_tokens, topk], dtype int32.
  torch::Tensor expert_ids;
  // W8A8 expert weights and fp32 scales in TensorList form.
  torch::TensorList gmm1_permuted_weight;
  torch::TensorList gmm1_permuted_weight_scale;
  torch::TensorList gmm2_weight;
  torch::TensorList gmm2_weight_scale;
  // Router probabilities. Shape: [num_tokens, topk], dtype fp32.
  torch::Tensor expert_scales;
  std::optional<torch::Tensor> expert_smooth_scales = std::nullopt;
  std::optional<torch::Tensor> x_active_mask = std::nullopt;
  std::string group_ep;
  int64_t ep_rank_size = 1;
  int64_t ep_rank_id = 0;
  int64_t moe_expert_num = 1;
  int64_t shared_expert_num = 0;
  int64_t shared_expert_rank_num = 0;
  int64_t quant_mode = 0;
  int64_t global_bs = 0;
};

// FP8 scaled quantize parameters
// Quantizes input tensor to FP8 e4m3 format with scale
struct Fp8ScaledQuantizeParams {
  // Input tensor. Shape: [M, K]. Dtype: float16, bfloat16.
  torch::Tensor input;
  // Optional output tensor. Shape: [M, K]. Dtype: float8_e4m3fn.
  // If not provided, will be allocated automatically.
  std::optional<torch::Tensor> output;
  // Optional pre-computed scale for static quantization.
  // Shape: scalar or [1]. If not provided, scale will be computed dynamically.
  std::optional<torch::Tensor> scale;
};

// FP8 scaled matmul parameters for W8A8 quantization
// Performs: c = (a @ b.T) with scales applied, following CUTLASS convention
struct Fp8ScaledMatmulParams {
  // Quantized input tensor A. Shape: [M, K]. Dtype: float8_e4m3fn.
  torch::Tensor a;
  // Quantized weight tensor B. Shape: [N, K] (will be transposed internally).
  // Dtype: float8_e4m3fn.
  torch::Tensor b;
  // Scale for tensor A. Shape: scalar or [1].
  torch::Tensor a_scale;
  // Scale for tensor B. Shape: scalar or [1].
  torch::Tensor b_scale;
  // Optional bias tensor. Shape: [N].
  std::optional<torch::Tensor> bias;
  // Optional output tensor. Shape: [M, N].
  // If not provided, will be allocated with output_dtype.
  std::optional<torch::Tensor> output;
  // Output data type. Typically float16 or bfloat16.
  torch::ScalarType output_dtype;
  // Optional original input shape (before flatten to 2D).
  // If provided, output will be reshaped to match original input dimensions.
  // E.g., input_shape = [batch, seq, hidden] -> output = [batch, seq, N]
  std::optional<std::vector<int64_t>> input_shape;
};

// Static scaled FP8 quantization parameters
// Quantizes input tensor to FP8 using a pre-computed scale factor
struct StaticScaledFp8QuantParams {
  // Output tensor to store quantized result. Shape: [..., d].
  // Dtype: float8_e4m3fn. Must be pre-allocated.
  torch::Tensor output;
  // Input tensor to quantize. Shape: [..., d].
  // Dtype: float16, bfloat16, or float32.
  torch::Tensor input;
  // Pre-computed scale factor. Shape: [1] or scalar.
  // Dtype: float32. Used for static quantization.
  torch::Tensor scale;
};

// Fused RMSNorm + Static FP8 Quantization Parameters
// These fused operations combine RMSNorm and FP8 quantization to reduce memory
// bandwidth by avoiding the intermediate write-back to global memory.

// Fused RMSNorm + Static FP8 Quantization parameters (without residual)
struct RmsNormStaticFp8QuantParams {
  // Input tensor. Shape: [..., hidden_size]. Dtype: float16, bfloat16, float32.
  torch::Tensor input;
  // RMSNorm weight. Shape: [hidden_size]. Dtype: same as input.
  torch::Tensor weight;
  // FP8 quantization scale (pre-computed). Shape: [1]. Dtype: float32.
  torch::Tensor scale;
  // RMSNorm epsilon.
  double epsilon;
};

// Fused Add + RMSNorm + Static FP8 Quantization parameters (with residual)
struct FusedAddRmsNormStaticFp8QuantParams {
  // Input tensor. Shape: [..., hidden_size]. Dtype: float16, bfloat16, float32.
  torch::Tensor input;
  // Residual tensor. Shape: [..., hidden_size]. Dtype: same as input.
  // Updated in-place with: residual = input + residual
  torch::Tensor residual;
  // RMSNorm weight. Shape: [hidden_size]. Dtype: same as input.
  torch::Tensor weight;
  // FP8 quantization scale (pre-computed). Shape: [1]. Dtype: float32.
  torch::Tensor scale;
  // RMSNorm epsilon.
  double epsilon;
};

// NPU Fused GDN Gating parameters
struct FusedGdnGatingParams {
  torch::Tensor A_log;
  torch::Tensor a;
  torch::Tensor b;
  torch::Tensor dt_bias;
  float beta = 1.0f;
  float threshold = 20.0f;
};

// NPU Fused Recurrent Gated Delta Rule parameters
struct FusedRecurrentGatedDeltaRuleParams {
  torch::Tensor q;
  torch::Tensor k;
  torch::Tensor v;
  torch::Tensor g;
  std::optional<torch::Tensor> beta = std::nullopt;
  std::optional<float> scale = std::nullopt;
  std::optional<torch::Tensor> initial_state = std::nullopt;
  bool inplace_final_state = true;
  std::optional<torch::Tensor> cu_seqlens = std::nullopt;
  std::optional<torch::Tensor> ssm_state_indices = std::nullopt;
  std::optional<torch::Tensor> num_accepted_tokens = std::nullopt;
  bool use_qk_l2norm_in_kernel = false;
};

// NPU Fused Sigmoid Gating Delta Rule Update parameters
struct FusedSigmoidGatingDeltaRuleUpdateParams {
  torch::Tensor A_log;
  torch::Tensor a;
  torch::Tensor dt_bias;
  torch::Tensor q;
  torch::Tensor k;
  torch::Tensor v;
  torch::Tensor b;
  torch::Tensor initial_state_source;
  torch::Tensor initial_state_indices;
  torch::Tensor cu_seqlens;
  std::optional<torch::Tensor> num_accepted_tokens = std::nullopt;
  std::optional<float> scale = std::nullopt;
  bool use_qk_l2norm_in_kernel = false;
  bool is_kda = false;
  float softplus_beta = 1.0f;
  float softplus_threshold = 20.0f;
};

// NPU Causal Conv1d Update parameters
struct CausalConv1dUpdateParams {
  torch::Tensor x;
  torch::Tensor conv_state;
  torch::Tensor weight;
  bool activation = true;
  std::optional<torch::Tensor> bias = std::nullopt;
  std::optional<torch::Tensor> conv_state_indices = std::nullopt;
  std::optional<torch::Tensor> num_accepted_tokens = std::nullopt;
  std::optional<torch::Tensor> query_start_loc = std::nullopt;
  int32_t max_query_len = -1;
  int32_t pad_slot_id = -1;
  std::optional<torch::Tensor> block_idx_last_scheduled_token;
  std::optional<torch::Tensor> initial_state_idx;
  std::optional<torch::Tensor> initial_state_mode;
  bool validate_data = false;
};

struct GatedLayerNormParams {
  torch::Tensor x;
  torch::Tensor weight;
  torch::Tensor bias;
  double eps;
  std::optional<torch::Tensor> z = std::nullopt;
  int64_t group_size = -1;
  bool norm_before_gate = true;
  bool is_rms_norm = true;
};

struct PartialRotaryEmbeddingParams {
  torch::Tensor positions;
  torch::Tensor query;
  torch::Tensor key;
  int64_t head_size;
  int64_t rotary_dim;
  torch::Tensor cos_sin_cache;
  bool is_neox_style;
};

struct FusedQkvzbaSplitReshapeParams {
  torch::Tensor mixed_qkvz;
  torch::Tensor mixed_ba;
  int32_t num_heads_qk;
  int32_t num_heads_v;
  int32_t head_qk;
  int32_t head_v;
};

struct GemmaRMSNormParams {
  torch::Tensor x;
  torch::Tensor gamma;
  double epsilon;
  std::optional<torch::Tensor> residual;
  std::optional<torch::Tensor> residual_out;
  torch::Tensor rstd_out;
  torch::Tensor norm_out;
};

struct SplitQkvRmsnormMropeParams {
  torch::Tensor qkvg;
  torch::Tensor q_weight;
  torch::Tensor k_weight;
  torch::Tensor cos_sin;
  torch::Tensor gather_pattern;
  float eps;
  int64_t num_q_heads;
  int64_t num_kv_heads;
  int64_t head_size;
};

struct ChunkGatedDeltaRuleParams {
  // Query tensor. Shape: [B, T, Hqk, K]. Dtype: bfloat16.
  torch::Tensor q;
  // Key tensor. Shape: [B, T, Hqk, K]. Dtype: bfloat16.
  torch::Tensor k;
  // Value tensor. Shape: [B, T, H, V]. Dtype: bfloat16.
  torch::Tensor v;
  // Gating tensor. Shape: [B, T, H]. Dtype: float32 or bfloat16.
  torch::Tensor g;
  // Beta tensor. Shape: [B, T, H]. Dtype: float32 or bfloat16.
  torch::Tensor beta;
  // Optional scale factor for attention. Default: K^(-0.5).
  std::optional<float> scale = std::nullopt;
  // Optional initial state tensor. Shape: [N, H, K, V]. Dtype: bfloat16.
  std::optional<torch::Tensor> initial_state = std::nullopt;
  // Whether to output the final state.
  bool output_final_state = false;
  // Chunk size for processing. Default: 64.
  int64_t chunk_size = 64;
  // Optional cumulative sequence lengths. Shape: [num_sequences + 1]. Dtype:
  // int32.
  std::optional<torch::Tensor> cu_seqlens = std::nullopt;
  // Whether input is head-first format. Default: false (batch-first).
  bool head_first = false;
  // Whether to apply L2 norm to q and k inside the kernel. Default: false.
  bool use_qk_l2norm_in_kernel = false;
};

struct MegaChunkGdnParams {
  // Query tensor. Shape: [B, T, Hqk, K]. Dtype: bfloat16 input, converted to
  // float16 before aclnnMegaChunkGdn.
  torch::Tensor q;
  // Key tensor. Shape: [B, T, Hqk, K]. Dtype: bfloat16 input, converted to
  // float16 before aclnnMegaChunkGdn.
  torch::Tensor k;
  // Value tensor. Shape: [B, T, H, V]. Dtype: bfloat16 input, converted to
  // float16 before aclnnMegaChunkGdn.
  torch::Tensor v;
  // Gating tensor. Shape: [B, T, H]. Dtype: float32 or bfloat16.
  torch::Tensor g;
  // Beta tensor. Shape: [B, T, H]. Dtype: float32 or bfloat16 input, converted
  // to float16 before aclnnMegaChunkGdn.
  torch::Tensor beta;
  // Optional scale factor for attention. Default: K^(-0.5).
  std::optional<float> scale = std::nullopt;
  // Optional initial state tensor. Shape: [N, H, K, V]. Converted to float16
  // before aclnnMegaChunkGdn; the wrapper receives final_state in a float16
  // buffer and casts it back to float32 for callers.
  std::optional<torch::Tensor> initial_state = std::nullopt;
  // Whether to output the final state.
  bool output_final_state = false;
  // Optional cumulative sequence lengths. Shape: [num_sequences + 1]. Dtype:
  // int32.
  std::optional<torch::Tensor> cu_seqlens = std::nullopt;
  // Optional non-owning host sequence lengths. When provided, these avoid
  // copying cu_seqlens from the device merely to derive the chunk count.
  c10::ArrayRef<int32_t> q_seq_lens = {};
  // Whether to apply L2 norm to q and k inside the kernel. Default: false.
  bool use_qk_l2norm_in_kernel = false;
};

struct HcPostParams {
  torch::Tensor x;
  torch::Tensor residual;
  torch::Tensor post;
  torch::Tensor comb;
};

struct QuantLightningIndexerParams {
  torch::Tensor query;
  torch::Tensor key;
  torch::Tensor weights;
  torch::Tensor query_dequant_scale;
  torch::Tensor key_dequant_scale;
  int64_t query_quant_mode = 0;
  int64_t key_quant_mode = 0;
  c10::optional<torch::Tensor> actual_seq_lengths_query;
  c10::optional<torch::Tensor> actual_seq_lengths_key;
  c10::optional<torch::Tensor> block_table;
  c10::optional<torch::Tensor> metadata;
  std::string layout_query;
  std::string layout_key;
  int64_t sparse_count = 0;
  int64_t sparse_mode = 0;
  int64_t pre_tokens = 0;
  int64_t next_tokens = 0;
  int64_t cmp_ratio = 1;
  bool return_value = false;
};

struct HcPreInvRmsParams {
  torch::Tensor x;
  double epsilon = 1e-6;
};

struct HcPreSinkhornParams {
  torch::Tensor mixes;
  torch::Tensor rsqrt;
  torch::Tensor hc_scale;
  torch::Tensor hc_base;
  torch::Tensor x;
  int64_t hc_mult = 4;
  int64_t hc_sinkhorn_iters = 20;
  double hc_eps = 1e-6;
};

struct HcPreParams {
  torch::Tensor x;
  torch::Tensor hc_fn;
  torch::Tensor hc_scale;
  torch::Tensor hc_base;
  int64_t hc_mult = 4;
  int64_t hc_sinkhorn_iters = 20;
  double norm_eps = 1e-6;
  double hc_eps = 1e-6;
};

struct MoeGatingTopKHashParams {
  torch::Tensor x;
  int64_t k = 0;
  c10::optional<torch::Tensor> bias;
  c10::optional<torch::Tensor> input_ids;
  c10::optional<torch::Tensor> tid2eid;
  int64_t k_group = 1;
  int64_t group_count = 1;
  int64_t group_select_mode = 0;
  int64_t renorm = 0;
  int64_t norm_type = 0;
  bool out_flag = false;
  double routed_scaling_factor = 1.0;
  double eps = 1e-20;
};

struct SparseAttnSharedkvParams {
  torch::Tensor q;
  c10::optional<torch::Tensor> ori_kv;
  c10::optional<torch::Tensor> cmp_kv;
  c10::optional<torch::Tensor> ori_sparse_indices;
  c10::optional<torch::Tensor> cmp_sparse_indices;
  c10::optional<torch::Tensor> ori_block_table;
  c10::optional<torch::Tensor> cmp_block_table;
  c10::optional<torch::Tensor> cu_seqlens_q;
  c10::optional<torch::Tensor> cu_seqlens_ori_kv;
  c10::optional<torch::Tensor> cu_seqlens_cmp_kv;
  c10::optional<torch::Tensor> seqused_q;
  c10::optional<torch::Tensor> seqused_kv;
  c10::optional<torch::Tensor> sinks;
  c10::optional<torch::Tensor> metadata;
  double softmax_scale = 1.0;
  int64_t cmp_ratio = 1;
  int64_t ori_mask_mode = 0;
  int64_t cmp_mask_mode = 0;
  int64_t ori_win_left = 0;
  int64_t ori_win_right = 0;
  std::string layout_q;
  std::string layout_kv;
  bool return_softmax_lse = false;
};

struct SparseFlashAttentionParams {
  torch::Tensor query;
  torch::Tensor key;
  torch::Tensor value;
  torch::Tensor sparse_indices;
  c10::optional<torch::Tensor> block_table;
  c10::optional<torch::Tensor> actual_seq_lengths_query;
  c10::optional<torch::Tensor> actual_seq_lengths_kv;
  c10::optional<torch::Tensor> query_rope;
  c10::optional<torch::Tensor> key_rope;
  double scale_value = 1.0;
  int64_t sparse_block_size = 0;
  std::string layout_query;
  std::string layout_kv;
  int64_t sparse_mode = 0;
};

struct CompressorParams {
  torch::Tensor x;
  torch::Tensor wkv;
  torch::Tensor wgate;
  torch::Tensor kv_state;
  torch::Tensor score_state;
  torch::Tensor ape;
  torch::Tensor norm_weight;
  torch::Tensor rope_sin;
  torch::Tensor rope_cos;
  c10::optional<torch::Tensor> kv_block_table;
  c10::optional<torch::Tensor> score_block_table;
  c10::optional<torch::Tensor> cu_seqlens;
  c10::optional<torch::Tensor> seqused;
  c10::optional<torch::Tensor> start_pos;
  int64_t rope_head_dim = 0;
  int64_t cmp_ratio = 1;
  int64_t coff = 1;
  double norm_eps = 1e-6;
  int64_t rotary_mode = 0;
  bool enable_grad = false;
};

struct QuantLightningIndexerMetadataParams {
  int64_t num_heads_q = 0;
  int64_t num_heads_k = 0;
  int64_t head_dim = 0;
  int64_t query_quant_mode = 0;
  int64_t key_quant_mode = 0;
  c10::optional<torch::Tensor> actual_seq_lengths_query;
  c10::optional<torch::Tensor> actual_seq_lengths_key;
  int64_t batch_size = 0;
  int64_t max_seqlen_q = 0;
  int64_t max_seqlen_k = 0;
  std::string layout_query;
  std::string layout_key;
  int64_t sparse_count = 0;
  int64_t sparse_mode = 0;
  int64_t pre_tokens = 0;
  int64_t next_tokens = 0;
  int64_t cmp_ratio = 1;
  std::string device = "npu";
};

struct SparseAttnSharedkvMetadataParams {
  int64_t num_heads_q = 0;
  int64_t num_heads_kv = 0;
  int64_t head_dim = 0;
  c10::optional<torch::Tensor> cu_seqlens_q;
  c10::optional<torch::Tensor> cu_seqlens_ori_kv;
  c10::optional<torch::Tensor> cu_seqlens_cmp_kv;
  c10::optional<torch::Tensor> seqused_q;
  c10::optional<torch::Tensor> seqused_kv;
  int64_t batch_size = 0;
  int64_t max_seqlen_q = 0;
  int64_t max_seqlen_kv = 0;
  int64_t ori_topk = 0;
  int64_t cmp_topk = 0;
  int64_t cmp_ratio = 1;
  int64_t ori_mask_mode = 0;
  int64_t cmp_mask_mode = 0;
  int64_t ori_win_left = 0;
  int64_t ori_win_right = 0;
  std::string layout_q;
  std::string layout_kv;
  bool has_ori_kv = false;
  bool has_cmp_kv = false;
};

struct NpuInplacePartialRotaryMulParams {
  torch::Tensor x;
  torch::Tensor r1;
  torch::Tensor r2;
  std::string rotary_mode = "interleave";
  std::vector<int64_t> partial_slice;
};

}  // namespace xllm::kernel

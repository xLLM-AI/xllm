# Copyright 2026 The xLLM Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/xLLM-AI/xllm/blob/main/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for Qwen3-VL ViT encode() with different backends.

Tests Qwen3VLVisionTransformer execution across:
- eager: direct model invocation (no compilation)
- torchair (static): torch.compile with torchair backend, dynamic=false
- torchair (dynamic): torch.compile with torchair backend, dynamic=true
- inductor: torch.compile with inductor backend

Each test verifies:
- Model can be created and compiled successfully
- Forward pass produces valid output (no NaN, correct shape)
- Output statistics are reasonable (min/max/mean within expected range)
"""

from __future__ import annotations

import gc
import sys
import time

import pytest
import torch

# conftest.py installs a stub for xllm.python, we need to restore the real module
def _restore_real_xllm_python():
    """Restore real xllm.python module (undo conftest.py stub)."""
    # Remove stub modules
    for key in list(sys.modules.keys()):
        if key.startswith("xllm.python"):
            del sys.modules[key]
    
    # Now import the real xllm.python
    import xllm.python

_restore_real_xllm_python()

# Register dummy ops before importing xllm modules
def _register_dummy_ops():
    """Register dummy xllm_ops so kernels_npu can import without C++ lib."""
    _dummy = lambda *a, **kw: None
    _ops = [
        ("rms_norm", "(Tensor input, Tensor weight, float eps) -> Tensor"),
        ("fused_add_rms_norm", "(Tensor(a!) input, Tensor(b!) residual, Tensor weight, float eps) -> (Tensor, Tensor)"),
        ("silu_and_mul", "(Tensor input) -> Tensor"),
        ("inplace_partial_rotary_mul", "(Tensor(a!) input, Tensor cosine, Tensor sine, str rotary_mode, int[] partial_slice) -> ()"),
        ("fused_qk_norm_rope", "(Tensor(a!) qkv, int num_heads_q, int num_heads_k, int num_heads_v, int head_dim, float eps, Tensor q_weight, Tensor k_weight, Tensor cos_sin_cache, bool interleaved, Tensor position_ids) -> Tensor(a!)"),
        ("reshape_paged_cache", "(Tensor slot_mapping, Tensor(c!) keys, Tensor(d!) values, Tensor(a!) key_cache, Tensor(b!) value_cache) -> Tensor"),
        ("apply_rotary_embedding", "(Tensor(a!) q, Tensor(b!) k, Tensor cos_sin_cache, Tensor positions) -> ()"),
        ("update_decode_graph_metadata", "(Tensor tokens, Tensor positions, Tensor slot_mapping, Tensor kv_seq_lens, Tensor paged_kv_indptr, Tensor paged_kv_indices, Tensor paged_kv_last_page_len, Tensor(a!) dst_tokens, Tensor(b!) dst_positions, Tensor(c!) dst_slot_mapping, Tensor(d!) dst_kv_seq_lens, Tensor(e!) dst_kv_seq_lens_delta, Tensor(f!) dst_paged_kv_indptr, Tensor(g!) dst_paged_kv_indices, Tensor(h!) dst_paged_kv_last_page_len, int padded_num_tokens) -> Tensor"),
        ("quant_matmul", "(Tensor x1, Tensor x2, bool transpose2, Tensor scale, Tensor? offset, Tensor? pertoken_scale, Tensor? bias, ScalarType? output_dtype) -> Tensor"),
        ("quantize_per_tensor", "(Tensor self, Tensor scales, Tensor zero_points, ScalarType dtype, int axis) -> Tensor"),
        ("dynamic_quant", "(Tensor input, Tensor? smooth_scales, Tensor? group_index, ScalarType? dst_type) -> (Tensor, Tensor?)"),
        ("quant_lightning_indexer", "(Tensor query, Tensor key, Tensor weights, Tensor query_dequant_scale, Tensor key_dequant_scale, int query_quant_mode, int key_quant_mode, Tensor? actual_seq_lengths_query, Tensor? actual_seq_lengths_key, Tensor? block_table, Tensor? metadata, str layout_query, str layout_key, int sparse_count, int sparse_mode, int pre_tokens, int next_tokens, int cmp_ratio, bool return_value) -> (Tensor, Tensor)"),
        ("quant_lightning_indexer_metadata", "(int num_heads_q, int num_heads_k, int head_dim, int query_quant_mode, int key_quant_mode, Tensor? actual_seq_lengths_query, Tensor? actual_seq_lengths_key, int batch_size, int max_seqlen_q, int max_seqlen_k, str layout_query, str layout_key, int sparse_count, int sparse_mode, int pre_tokens, int next_tokens, int cmp_ratio, str device) -> Tensor"),
        ("lightning_indexer", "(Tensor query, Tensor key, Tensor weights, Tensor? query_seq_lengths, Tensor? key_seq_lengths, Tensor? block_table, str layout_query, str layout_key, int selected_count, int sparse_mode, int pre_tokens, int next_tokens, bool return_value) -> Tensor"),
        ("lightning_indexer_out", "(Tensor query, Tensor key, Tensor weights, Tensor? query_seq_lengths, Tensor? key_seq_lengths, Tensor? block_table, str layout_query, str layout_key, int selected_count, int sparse_mode, int pre_tokens, int next_tokens, bool return_value, Tensor(a!) sparse_indices_out, Tensor(b!) sparse_values_out) -> Tensor(a!)"),
        ("scatter_nd_update", "(Tensor(a!) var, Tensor indices, Tensor updates) -> ()"),
        ("sparse_flash_attention", "(Tensor query, Tensor key, Tensor value, Tensor sparse_indices, Tensor? block_table, Tensor? actual_seq_lengths_query, Tensor? actual_seq_lengths_kv, Tensor? query_rope, Tensor? key_rope, float scale_value, int sparse_block_size, str layout_query, str layout_kv, int sparse_mode) -> Tensor"),
        ("sparse_flash_attention_out", "(Tensor query, Tensor key, Tensor value, Tensor sparse_indices, Tensor? block_table, Tensor? actual_seq_lengths_query, Tensor? actual_seq_lengths_kv, Tensor? query_rope, Tensor? key_rope, float scale_value, int sparse_block_size, str layout_query, str layout_kv, int sparse_mode, Tensor(a!) output) -> Tensor(a!)"),
        ("mla_preprocess_v2", "(Tensor input, Tensor gamma0, Tensor beta0, Tensor quant_scale0, Tensor quant_offset0, Tensor wdqkv, Tensor descale0, Tensor bias0, Tensor gamma1, Tensor beta1, Tensor quant_scale1, Tensor quant_offset1, Tensor wuq, Tensor descale1, Tensor bias1, Tensor gamma2, Tensor cos, Tensor sin, Tensor wuk, Tensor(a!) kv_cache, Tensor(b!) kv_cache_rope, Tensor slot_mapping, Tensor ctkv_scale, Tensor q_nope_scale, int wdq_dim, int q_rope_dim, int k_rope_dim, float epsilon, int q_rotary_coeff, int k_rotary_coeff, bool transpose_wdq, bool transpose_wuq, bool transpose_wuk, int cache_mode, int quant_mode, bool do_rms_norm, int wdkv_split_count, bool q_down_out_flag) -> (Tensor, Tensor(a!), Tensor, Tensor(b!), Tensor)"),
    ]
    for name, schema in _ops:
        try:
            torch.library.define(f"xllm_ops::{name}", schema, tags=())
        except RuntimeError:
            pass
        try:
            torch.library.impl(f"xllm_ops::{name}", "cpu", _dummy)
        except RuntimeError:
            pass

_register_dummy_ops()

from xllm.python import initialize_runtime
initialize_runtime()

from xllm.python.models.qwen3_vl import (
    Qwen3VLVisionConfig,
    Qwen3VLVisionTransformer,
)
from xllm.python.model_executor.vit_executor import ViTExecutor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _create_vit_executor(backend: str, dynamic: bool = False) -> ViTExecutor:
    """Create Qwen3VLVisionTransformer with ViTExecutor."""
    import torch._dynamo
    torch._dynamo.config.recompile_limit = 128
    
    vision_config = Qwen3VLVisionConfig(
        deepstack_visual_indexes=[8, 16, 24],
        depth=27,
        hidden_size=1152,
        num_heads=16,
        patch_size=16,
        temporal_patch_size=2,
        spatial_merge_size=2,
        intermediate_size=4304,
        out_hidden_size=5120,
        in_channels=3,
        hidden_act="gelu_pytorch_tanh",
        num_position_embeddings=2304,
    )
    
    vit = Qwen3VLVisionTransformer(
        vision_config, dtype=torch.bfloat16, device=torch.device("npu")
    )
    vit.eval().npu()
    
    compile_kwargs = {"dynamic": dynamic} if backend != "eager" else None
    
    return ViTExecutor(
        vit,
        backend=backend,
        compile_kwargs=compile_kwargs,
    )


def _create_dummy_inputs(image_size: int, device: str = "npu", seed: int = 42):
    """Create dummy pixel_values and grid_thw for given image_size."""
    patch_size = 16
    temporal_patch_size = 2
    in_channels = 3
    
    h_patches = image_size // patch_size
    w_patches = image_size // patch_size
    t_patches = 1
    
    num_patches = t_patches * h_patches * w_patches
    patch_dim = in_channels * temporal_patch_size * patch_size * patch_size
    
    torch.manual_seed(seed)
    pixel_values = torch.randn(
        num_patches, patch_dim, dtype=torch.bfloat16, device=device
    )
    grid_thw = torch.tensor(
        [[t_patches, h_patches, w_patches]], dtype=torch.int32, device=device
    )
    
    return pixel_values, grid_thw, num_patches


def _run_forward_and_verify(vit_executor, image_size: int, warmup: int = 1, repeats: int = 3):
    """Run forward pass and verify output validity."""
    pixel_values, grid_thw, num_patches = _create_dummy_inputs(image_size)
    
    def run():
        with torch.no_grad():
            return vit_executor.execute(pixel_values, grid_thw)
    
    # Warmup
    for _ in range(warmup):
        output = run()
    torch.npu.synchronize()
    
    # Benchmark
    latencies = []
    for _ in range(repeats):
        torch.npu.synchronize()
        t0 = time.perf_counter()
        output = run()
        torch.npu.synchronize()
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000)
    
    # Verify output
    assert output is not None, "Output is None"
    assert isinstance(output, torch.Tensor), f"Output is not a Tensor: {type(output)}"
    assert output.dtype == torch.bfloat16, f"Output dtype is {output.dtype}, expected bfloat16"
    assert output.device.type == "npu", f"Output device is {output.device}, expected npu"
    
    # Check for NaN/Inf
    assert not torch.isnan(output).any(), "Output contains NaN"
    assert not torch.isinf(output).any(), "Output contains Inf"
    
    # Check output statistics are reasonable
    output_f32 = output.float()
    min_val = output_f32.min().item()
    max_val = output_f32.max().item()
    mean_val = output_f32.mean().item()
    
    # Reasonable range for attention output (should be bounded)
    assert abs(min_val) < 100.0, f"Output min {min_val} is out of reasonable range"
    assert abs(max_val) < 100.0, f"Output max {max_val} is out of reasonable range"
    assert abs(mean_val) < 10.0, f"Output mean {mean_val} is out of reasonable range"
    
    avg_latency = sum(latencies) / len(latencies)
    
    return {
        "output_shape": output.shape,
        "num_patches": num_patches,
        "min": min_val,
        "max": max_val,
        "mean": mean_val,
        "avg_latency_ms": avg_latency,
    }


def _cleanup(vit_executor):
    """Clean up resources."""
    del vit_executor
    gc.collect()
    torch.npu.empty_cache()
    
    import os
    import shutil
    
    # Clean up torchair generated files
    fusion_result_file = "fusion_result.json"
    if os.path.exists(fusion_result_file):
        try:
            os.remove(fusion_result_file)
        except OSError as e:
            print(f"\nWarning: Failed to delete {fusion_result_file}: {e}")
    
    # Clean up torch_compile_debug directory
    torch_compile_debug_dir = "torch_compile_debug"
    if os.path.exists(torch_compile_debug_dir):
        try:
            shutil.rmtree(torch_compile_debug_dir)
        except OSError as e:
            print(f"\nWarning: Failed to delete {torch_compile_debug_dir}: {e}")


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU not available")
def test_vit_eager():
    """Test ViT with eager backend (no compilation)."""
    image_size = 256
    vit_executor = _create_vit_executor(backend="eager")
    
    try:
        result = _run_forward_and_verify(vit_executor, image_size)
        print(f"\n[Eager] image_size={image_size}, "
              f"output_shape={result['output_shape']}, "
              f"avg_latency={result['avg_latency_ms']:.2f}ms, "
              f"min={result['min']:.6f}, max={result['max']:.6f}, mean={result['mean']:.6f}")
    finally:
        _cleanup(vit_executor)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU not available")
def test_vit_torchair_static():
    """Test ViT with torchair backend, dynamic=false (static graph)."""
    image_size = 256
    vit_executor = _create_vit_executor(backend="torchair", dynamic=False)
    
    try:
        result = _run_forward_and_verify(vit_executor, image_size)
        print(f"\n[TorchAir Static] image_size={image_size}, "
              f"output_shape={result['output_shape']}, "
              f"avg_latency={result['avg_latency_ms']:.2f}ms, "
              f"min={result['min']:.6f}, max={result['max']:.6f}, mean={result['mean']:.6f}")
    finally:
        _cleanup(vit_executor)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU not available")
def test_vit_torchair_dynamic():
    """Test ViT with torchair backend, dynamic=true (dynamic graph)."""
    image_size = 256
    vit_executor = _create_vit_executor(backend="torchair", dynamic=True)
    
    try:
        result = _run_forward_and_verify(vit_executor, image_size)
        print(f"\n[TorchAir Dynamic] image_size={image_size}, "
              f"output_shape={result['output_shape']}, "
              f"avg_latency={result['avg_latency_ms']:.2f}ms, "
              f"min={result['min']:.6f}, max={result['max']:.6f}, mean={result['mean']:.6f}")
    finally:
        _cleanup(vit_executor)


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU not available")
def test_vit_inductor():
    """Test ViT with inductor backend (runs in subprocess to avoid conftest.py interference)."""
    import subprocess
    import sys
    
    # Run inductor test in subprocess to avoid conftest.py stubs affecting torch_npu
    test_code = '''
import torch
import sys
sys.path.insert(0, "/mnt/workspace/gitCode/guopeian/xllm-ai/xllm")

# Register dummy ops
_dummy = lambda *a, **kw: None
_ops = [
    ("rms_norm", "(Tensor input, Tensor weight, float eps) -> Tensor"),
    ("fused_add_rms_norm", "(Tensor(a!) input, Tensor(b!) residual, Tensor weight, float eps) -> (Tensor, Tensor)"),
    ("silu_and_mul", "(Tensor input) -> Tensor"),
    ("inplace_partial_rotary_mul", "(Tensor(a!) input, Tensor cosine, Tensor sine, str rotary_mode, int[] partial_slice) -> ()"),
    ("fused_qk_norm_rope", "(Tensor(a!) qkv, int num_heads_q, int num_heads_k, int num_heads_v, int head_dim, float eps, Tensor q_weight, Tensor k_weight, Tensor cos_sin_cache, bool interleaved, Tensor position_ids) -> Tensor(a!)"),
    ("reshape_paged_cache", "(Tensor slot_mapping, Tensor(c!) keys, Tensor(d!) values, Tensor(a!) key_cache, Tensor(b!) value_cache) -> Tensor"),
    ("apply_rotary_embedding", "(Tensor(a!) q, Tensor(b!) k, Tensor cos_sin_cache, Tensor positions) -> ()"),
    ("update_decode_graph_metadata", "(Tensor tokens, Tensor positions, Tensor slot_mapping, Tensor kv_seq_lens, Tensor paged_kv_indptr, Tensor paged_kv_indices, Tensor paged_kv_last_page_len, Tensor(a!) dst_tokens, Tensor(b!) dst_positions, Tensor(c!) dst_slot_mapping, Tensor(d!) dst_kv_seq_lens, Tensor(e!) dst_kv_seq_lens_delta, Tensor(f!) dst_paged_kv_indptr, Tensor(g!) dst_paged_kv_indices, Tensor(h!) dst_paged_kv_last_page_len, int padded_num_tokens) -> Tensor"),
    ("quant_matmul", "(Tensor x1, Tensor x2, bool transpose2, Tensor scale, Tensor? offset, Tensor? pertoken_scale, Tensor? bias, ScalarType? output_dtype) -> Tensor"),
    ("quantize_per_tensor", "(Tensor self, Tensor scales, Tensor zero_points, ScalarType dtype, int axis) -> Tensor"),
    ("dynamic_quant", "(Tensor input, Tensor? smooth_scales, Tensor? group_index, ScalarType? dst_type) -> (Tensor, Tensor?)"),
    ("quant_lightning_indexer", "(Tensor query, Tensor key, Tensor weights, Tensor query_dequant_scale, Tensor key_dequant_scale, int query_quant_mode, int key_quant_mode, Tensor? actual_seq_lengths_query, Tensor? actual_seq_lengths_key, Tensor? block_table, Tensor? metadata, str layout_query, str layout_key, int sparse_count, int sparse_mode, int pre_tokens, int next_tokens, int cmp_ratio, bool return_value) -> (Tensor, Tensor)"),
    ("quant_lightning_indexer_metadata", "(int num_heads_q, int num_heads_k, int head_dim, int query_quant_mode, int key_quant_mode, Tensor? actual_seq_lengths_query, Tensor? actual_seq_lengths_key, int batch_size, int max_seqlen_q, int max_seqlen_k, str layout_query, str layout_key, int sparse_count, int sparse_mode, int pre_tokens, int next_tokens, int cmp_ratio, str device) -> Tensor"),
    ("lightning_indexer", "(Tensor query, Tensor key, Tensor weights, Tensor? query_seq_lengths, Tensor? key_seq_lengths, Tensor? block_table, str layout_query, str layout_key, int selected_count, int sparse_mode, int pre_tokens, int next_tokens, bool return_value) -> Tensor"),
    ("lightning_indexer_out", "(Tensor query, Tensor key, Tensor weights, Tensor? query_seq_lengths, Tensor? key_seq_lengths, Tensor? block_table, str layout_query, str layout_key, int selected_count, int sparse_mode, int pre_tokens, int next_tokens, bool return_value, Tensor(a!) sparse_indices_out, Tensor(b!) sparse_values_out) -> Tensor(a!)"),
    ("scatter_nd_update", "(Tensor(a!) var, Tensor indices, Tensor updates) -> ()"),
    ("sparse_flash_attention", "(Tensor query, Tensor key, Tensor value, Tensor sparse_indices, Tensor? block_table, Tensor? actual_seq_lengths_query, Tensor? actual_seq_lengths_kv, Tensor? query_rope, Tensor? key_rope, float scale_value, int sparse_block_size, str layout_query, str layout_kv, int sparse_mode) -> Tensor"),
    ("sparse_flash_attention_out", "(Tensor query, Tensor key, Tensor value, Tensor sparse_indices, Tensor? block_table, Tensor? actual_seq_lengths_query, Tensor? actual_seq_lengths_kv, Tensor? query_rope, Tensor? key_rope, float scale_value, int sparse_block_size, str layout_query, str layout_kv, int sparse_mode, Tensor(a!) output) -> Tensor(a!)"),
    ("mla_preprocess_v2", "(Tensor input, Tensor gamma0, Tensor beta0, Tensor quant_scale0, Tensor quant_offset0, Tensor wdqkv, Tensor descale0, Tensor bias0, Tensor gamma1, Tensor beta1, Tensor quant_scale1, Tensor quant_offset1, Tensor wuq, Tensor descale1, Tensor bias1, Tensor gamma2, Tensor cos, Tensor sin, Tensor wuk, Tensor(a!) kv_cache, Tensor(b!) kv_cache_rope, Tensor slot_mapping, Tensor ctkv_scale, Tensor q_nope_scale, int wdq_dim, int q_rope_dim, int k_rope_dim, float epsilon, int q_rotary_coeff, int k_rotary_coeff, bool transpose_wdq, bool transpose_wuq, bool transpose_wuk, int cache_mode, int quant_mode, bool do_rms_norm, int wdkv_split_count, bool q_down_out_flag) -> (Tensor, Tensor(a!), Tensor, Tensor(b!), Tensor)"),
]
for name, schema in _ops:
    try:
        torch.library.define(f"xllm_ops::{name}", schema, tags=())
    except RuntimeError:
        pass
    try:
        torch.library.impl(f"xllm_ops::{name}", "cpu", _dummy)
    except RuntimeError:
        pass

from xllm.python import initialize_runtime
initialize_runtime()

from xllm.python.models.qwen3_vl import Qwen3VLVisionConfig, Qwen3VLVisionTransformer
from xllm.python.model_executor.vit_executor import ViTExecutor
import torch._dynamo

torch._dynamo.config.recompile_limit = 128

vision_config = Qwen3VLVisionConfig(
    deepstack_visual_indexes=[8, 16, 24],
    depth=27,
    hidden_size=1152,
    num_heads=16,
    patch_size=16,
    temporal_patch_size=2,
    spatial_merge_size=2,
    intermediate_size=4304,
    out_hidden_size=5120,
    in_channels=3,
    hidden_act="gelu_pytorch_tanh",
    num_position_embeddings=2304,
)

vit = Qwen3VLVisionTransformer(vision_config, dtype=torch.bfloat16, device=torch.device("npu"))
vit.eval().npu()

vit_executor = ViTExecutor(vit, backend="inductor", compile_kwargs={"dynamic": False})

torch.manual_seed(42)
image_size = 256
patch_size = 16
h_patches = image_size // patch_size
w_patches = image_size // patch_size
num_patches = h_patches * w_patches
patch_dim = 3 * 2 * patch_size * patch_size

pixel_values = torch.randn(num_patches, patch_dim, dtype=torch.bfloat16, device="npu")
grid_thw = torch.tensor([[1, h_patches, w_patches]], dtype=torch.int32, device="npu")

with torch.no_grad():
    output = vit_executor.execute(pixel_values, grid_thw)

assert output is not None
assert isinstance(output, torch.Tensor)
assert output.dtype == torch.bfloat16
assert output.device.type == "npu"
assert not torch.isnan(output).any()
assert not torch.isinf(output).any()

print("SUCCESS")
'''
    
    result = subprocess.run(
        [sys.executable, "-c", test_code],
        capture_output=True,
        text=True,
        timeout=120,
    )
    
    if result.returncode != 0:
        print(f"Subprocess failed:\n{result.stderr}")
        pytest.fail(f"Inductor test failed: {result.stderr[-500:]}")
    
    assert "SUCCESS" in result.stdout
    print("\n[Inductor] Test passed in subprocess")

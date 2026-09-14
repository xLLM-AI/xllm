"""Qwen3-VL ViT benchmark: test encode() with ViTExecutor.

Tests Qwen3VLForConditionalGeneration.encode() across:
- image_size: 256, 512, 800, 1024, 2048
- backend: eager, torchair, inductor (one per run)
- dynamic: false (static graph), true (dynamic graph)

Usage:
    # Run each backend separately (recommended to avoid process pollution)
    python tests/python/bench_qwen3_vl_vit_encode.py --backend eager --dynamic false
    python tests/python/bench_qwen3_vl_vit_encode.py --backend torchair --dynamic false
    python tests/python/bench_qwen3_vl_vit_encode.py --backend inductor --dynamic false
    
    # Profiling mode
    python tests/python/bench_qwen3_vl_vit_encode.py --backend torchair --profile --image-size 256
"""

from __future__ import annotations

import argparse
import gc
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import torch

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

def _import_vit():
    from xllm.python.models.qwen3_vl import (
        Qwen3VLVisionConfig,
        Qwen3VLVisionTransformer,
    )
    from xllm.python.model_executor.vit_executor import ViTExecutor
    return Qwen3VLVisionConfig, Qwen3VLVisionTransformer, ViTExecutor


# Qwen3-VL 32B config (vision tower only, LLM config is minimal)
QWEN3_VL_32B_CONFIG = {
    # Vision config (32B)
    "mm_hidden_size": 1152,
    "mm_num_hidden_layers": 27,
    "mm_num_attention_heads": 16,
    "mm_patch_size": 16,
    "mm_temporal_patch_size": 2,
    "mm_spatial_merge_size": 2,
    "mm_intermediate_size": 4304,
    "mm_projection_dim": 5120,
    "mm_num_channels": 3,
    "mm_hidden_act": "gelu_pytorch_tanh",
    "mm_deepstack_visual_indexes": [8, 16, 24],
    
    # LLM config (minimal, not used in encode)
    "hidden_size": 5120,
    "n_layers": 64,
    "n_heads": 64,
    "n_kv_heads": 8,
    "head_dim": 128,
    "intermediate_size": 25600,
    "vocab_size": 16000,  # Reduced to fit in memory
    "max_position_embeddings": 40960,
    
    # Device and dtype
    "device": "npu",
    "dtype": "bfloat16",
}

IMAGE_SIZES = [256, 512, 800, 1024, 2048]
BACKENDS = ["eager", "torchair", "inductor"]
DYNAMIC_MODES = [False, True]  # dynamic=False: static graph, dynamic=True: dynamic graph


def _create_model(backend: str, dynamic: bool):
    """Create Qwen3VLVisionTransformer with ViTExecutor (ViT only, no LLM)."""
    import torch._dynamo
    Qwen3VLVisionConfig, Qwen3VLVisionTransformer, ViTExecutor = _import_vit()
    
    # Increase recompile limit to avoid recompilation warnings
    # Default is 8, which can cause recompilation warnings for ViT with 27 layers
    torch._dynamo.config.recompile_limit = 128
    
    # Create vision config only
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
    
    # Create ViT only (not the full model with LLM)
    vit = Qwen3VLVisionTransformer(
        vision_config, dtype=torch.bfloat16, device=torch.device("npu")
    )
    vit.eval().npu()

    # Create ViTExecutor
    # For eager backend, dynamic parameter doesn't apply
    # For torchair/inductor, pass dynamic parameter to torch.compile
    if backend != "eager":
        compile_kwargs = {"dynamic": dynamic}
    else:
        compile_kwargs = None
    
    vit_executor = ViTExecutor(
        vit,
        backend=backend,
        compile_kwargs=compile_kwargs,
    )
    
    return vit_executor


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


def _benchmark(vit_executor, image_size: int, warmup: int, repeats: int, seed: int = 42):
    """Benchmark ViTExecutor.execute() for given image_size."""
    pixel_values, grid_thw, num_patches = _create_dummy_inputs(image_size, seed=seed)
    
    def run():
        with torch.no_grad():
            return vit_executor.execute(pixel_values, grid_thw)
    
    # Warmup
    output = None
    for _ in range(warmup):
        output = run()
    torch.npu.synchronize()
    
    # Benchmark
    latencies = []
    for i in range(repeats):
        torch.npu.synchronize()
        t0 = time.perf_counter()
        output = run()
        torch.npu.synchronize()
        t1 = time.perf_counter()
        lat = (t1 - t0) * 1000
        latencies.append(lat)
    
    latencies.sort()
    avg = sum(latencies) / len(latencies)
    p50 = latencies[len(latencies) // 2]
    p99 = latencies[int(len(latencies) * 0.99)]
    
    return avg, p50, p99, min(latencies), max(latencies), num_patches, output


def _profile(vit_executor, image_size: int, warmup: int, profile_steps: int, profile_dir: str):
    """Profile ViTExecutor.execute() for given image_size."""
    import torch_npu
    
    pixel_values, grid_thw, num_patches = _create_dummy_inputs(image_size)
    
    def run():
        with torch.no_grad():
            vit_executor.execute(pixel_values, grid_thw)
    
    # Warmup
    for _ in range(warmup):
        run()
    torch.npu.synchronize()
    
    # Profile
    experimental_config = torch_npu.profiler._ExperimentalConfig(
        export_type=[torch_npu.profiler.ExportType.Text],
        profiler_level=torch_npu.profiler.ProfilerLevel.Level2,
        msprof_tx=False,
        aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
        l2_cache=False,
        op_attr=False,
        data_simplification=False,
        record_op_args=False,
        gc_detect_threshold=None,
    )
    
    output_dir = f"{profile_dir}/img{image_size}"
    with torch_npu.profiler.profile(
        activities=[
            torch_npu.profiler.ProfilerActivity.CPU,
            torch_npu.profiler.ProfilerActivity.NPU,
        ],
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(output_dir),
        record_shapes=True,
        profile_memory=False,
        with_stack=False,
        with_modules=False,
        with_flops=False,
        experimental_config=experimental_config,
    ) as prof:
        for i in range(profile_steps):
            run()
            torch.npu.synchronize()
            prof.step()
    
    print(f"Profiling results saved to: {output_dir}")


def _run_single(backend: str, dynamic: bool, image_size: int, warmup: int, repeats: int, seed: int = 42):
    """Run single benchmark configuration."""
    result = {
        "backend": backend,
        "dynamic": dynamic,
        "image_size": image_size,
        "status": "OK",
    }
    
    try:
        vit_executor = _create_model(backend, dynamic)
        avg, p50, p99, mn, mx, num_patches, output = _benchmark(
            vit_executor, image_size, warmup, repeats, seed=seed
        )
        
        result.update(
            avg=avg, p50=p50, p99=p99, min=mn, max=mx, num_patches=num_patches,
            output=output,
        )
        
        if output is not None:
            print(f"\n--- Model Output (backend={backend}, dynamic={dynamic}, image_size={image_size}) ---")
            if isinstance(output, (tuple, list)):
                for idx_item, item in enumerate(output):
                    if isinstance(item, torch.Tensor):
                        print(f"  output[{idx_item}]: shape={item.shape}, dtype={item.dtype}, "
                              f"min={item.min().item():.6f}, max={item.max().item():.6f}, "
                              f"mean={item.mean().item():.6f}")
                    else:
                        print(f"  output[{idx_item}]: {item}")
            elif isinstance(output, torch.Tensor):
                print(f"  output: shape={output.shape}, dtype={output.dtype}, "
                      f"min={output.min().item():.6f}, max={output.max().item():.6f}, "
                      f"mean={output.mean().item():.6f}")
            else:
                print(f"  output: {output}")
            print("--- End Model Output ---\n")
        
        del vit_executor
        gc.collect()
        torch.npu.empty_cache()
        
    except torch.OutOfMemoryError:
        result["status"] = "OOM"
        torch.npu.empty_cache()
        gc.collect()
    except Exception as e:
        result["status"] = f"ERROR: {type(e).__name__}: {str(e)[:80]}"
        import traceback
        traceback.print_exc()
        torch.npu.empty_cache()
        gc.collect()
    
    return result


def test_torchair_dynamic_multi_shape():
    """Test torchair dynamic=True with multi-shape inputs in the same process.
    
    This reproduces the dimension mismatch error when executing different shapes
    sequentially with torchair dynamic=True backend.
    """
    print("=" * 80)
    print("Test: TorchAir dynamic=True with multi-shape inputs")
    print("=" * 80)
    print()
    
    # Create model with torchair + dynamic=True
    print("Creating ViTExecutor with backend=torchair, dynamic=True...")
    vit_executor = _create_model("torchair", dynamic=True)
    
    # Define two test cases
    test_cases = [
        {
            "name": "Case 1: [320, 1536] with grid_thw=[[1, 16, 20]]",
            "pixel_values_shape": (320, 1536),
            "grid_thw": [[1, 16, 20]],
        },
        {
            "name": "Case 2: [280, 1536] with grid_thw=[[1, 14, 20]]",
            "pixel_values_shape": (280, 1536),
            "grid_thw": [[1, 14, 20]],
        },
    ]
    
    results = []
    
    for i, case in enumerate(test_cases):
        print(f"\n{'=' * 80}")
        print(f"{case['name']}")
        print(f"{'=' * 80}")
        
        # Create inputs
        pixel_values = torch.randn(
            case["pixel_values_shape"],
            dtype=torch.bfloat16,
            device="npu",
        )
        grid_thw = torch.tensor(
            case["grid_thw"],
            dtype=torch.int32,
            device="npu",
        )
        
        print(f"pixel_values.shape: {pixel_values.shape}")
        print(f"grid_thw: {grid_thw.tolist()}")
        
        try:
            # Warmup
            print("\nWarmup...")
            for _ in range(3):
                with torch.no_grad():
                    output = vit_executor.execute(pixel_values, grid_thw)
            
            # Benchmark
            print("Benchmark...")
            latencies = []
            for _ in range(10):
                torch.npu.synchronize()
                start = time.perf_counter()
                with torch.no_grad():
                    output = vit_executor.execute(pixel_values, grid_thw)
                torch.npu.synchronize()
                end = time.perf_counter()
                latencies.append((end - start) * 1000)
            
            avg_latency = sum(latencies) / len(latencies)
            
            print(f"\n✓ SUCCESS")
            print(f"  Output shape: {output.shape}")
            print(f"  Output dtype: {output.dtype}")
            print(f"  Min: {output.min().item():.6f}")
            print(f"  Max: {output.max().item():.6f}")
            print(f"  Mean: {output.mean().item():.6f}")
            print(f"  Avg latency: {avg_latency:.2f} ms")
            
            results.append({
                "case": case["name"],
                "status": "SUCCESS",
                "output_shape": output.shape,
                "avg_latency": avg_latency,
            })
            
        except Exception as e:
            print(f"\n✗ FAILED: {type(e).__name__}")
            print(f"  Error: {str(e)[:200]}")
            import traceback
            traceback.print_exc()
            
            results.append({
                "case": case["name"],
                "status": f"FAILED: {type(e).__name__}",
                "error": str(e),
            })
    
    # Summary
    print(f"\n{'=' * 80}")
    print("Summary")
    print(f"{'=' * 80}")
    for r in results:
        status_icon = "✓" if r["status"] == "SUCCESS" else "✗"
        print(f"{status_icon} {r['case']}: {r['status']}")
        if r["status"] == "SUCCESS":
            print(f"    Output shape: {r['output_shape']}")
            print(f"    Avg latency: {r['avg_latency']:.2f} ms")
        else:
            print(f"    Error: {r.get('error', 'Unknown')[:100]}")
    
    # Cleanup
    del vit_executor
    gc.collect()
    torch.npu.empty_cache()
    
    # Return overall status
    all_success = all(r["status"] == "SUCCESS" for r in results)
    return all_success


def main():
    parser = argparse.ArgumentParser(description="Qwen3-VL ViT Encode Benchmark")
    parser.add_argument(
        "--backend",
        type=str,
        choices=BACKENDS,
        required=True,
        help="Backend to test: eager, torchair, or inductor",
    )
    parser.add_argument(
        "--dynamic",
        type=str,
        choices=["true", "false", "all"],
        default="all",
        help="Dynamic mode: false=static graph, true=dynamic graph (default: all)",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        choices=IMAGE_SIZES + [0],
        default=0,
        help="Single image size to test (0 = all)",
    )
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible inputs")
    parser.add_argument("--compare", action="store_true",
                        help="Compare backend output against eager (runs eager first as reference)")
    parser.add_argument("--profile", action="store_true", help="Enable NPU profiling")
    parser.add_argument("--profile-dir", type=str, default="./profiler", help="Profiling output directory")
    parser.add_argument("--profile-steps", type=int, default=10, help="Number of profiling steps")
    parser.add_argument("--multi-shape", action="store_true",
                        help="Test torchair dynamic=True with multi-shape inputs (320 and 280 patches)")
    args = parser.parse_args()
    
    # Multi-shape test mode
    if args.multi_shape:
        test_torchair_dynamic_multi_shape()
        return
    
    backend = args.backend
    dynamic_modes = (
        DYNAMIC_MODES
        if args.dynamic == "all"
        else [True if args.dynamic == "true" else False]
    )
    image_sizes = IMAGE_SIZES if args.image_size == 0 else [args.image_size]
    
    # Profiling mode
    if args.profile:
        print(f"Running profiling for backend={backend}, {len(image_sizes)} image_size(s) ...")
        print(f"Dynamic Modes: {dynamic_modes}")
        print(f"Image Sizes: {image_sizes}")
        print(f"Profile steps: {args.profile_steps}")
        print(f"Output directory: {args.profile_dir}")
        print()
        
        import os
        for dynamic in dynamic_modes:
            for image_size in image_sizes:
                print(f"Profiling backend={backend}, dynamic={dynamic}, image_size={image_size} ...")
                try:
                    vit_executor = _create_model(backend, dynamic)
                    profile_subdir = f"{args.profile_dir}/{backend}_dynamic{dynamic}"
                    os.makedirs(profile_subdir, exist_ok=True)
                    _profile(vit_executor, image_size, args.warmup, args.profile_steps, profile_subdir)
                    del vit_executor
                    gc.collect()
                    torch.npu.empty_cache()
                except Exception as e:
                    print(f"ERROR: {type(e).__name__}: {str(e)[:80]}")
                    import traceback
                    traceback.print_exc()
                    torch.npu.empty_cache()
                    gc.collect()
        
        return
    
    # Benchmark mode
    total = len(dynamic_modes) * len(image_sizes)
    print(f"Running {total} benchmark configurations ...")
    print(f"Backend: {backend}")
    print(f"Dynamic Modes: {dynamic_modes}")
    print(f"Image Sizes: {image_sizes}")
    print(f"Warmup: {args.warmup}, Repeats: {args.repeats}, Seed: {args.seed}")
    if args.compare:
        print(f"Compare mode: ON (reference = eager)")
    print()
    
    t_total = time.perf_counter()
    results = []
    idx = 0
    eager_outputs = {}
    
    if args.compare and backend != "eager":
        print("=" * 60)
        print("Running eager as reference ...")
        print("=" * 60)
        for dynamic in dynamic_modes:
            for image_size in image_sizes:
                r_eager = _run_single("eager", dynamic, image_size, args.warmup, args.repeats, seed=args.seed)
                if r_eager["status"] == "OK" and r_eager.get("output") is not None:
                    eager_outputs[(dynamic, image_size)] = r_eager["output"].cpu().float()
                    print(f"  eager reference saved: dynamic={dynamic}, image_size={image_size}, "
                          f"shape={r_eager['output'].shape}")
                del r_eager
                gc.collect()
                torch.npu.empty_cache()
        print()
    
    for dynamic in dynamic_modes:
        for image_size in image_sizes:
            idx += 1
            r = _run_single(backend, dynamic, image_size, args.warmup, args.repeats, seed=args.seed)
            results.append(r)
            if r["status"] == "OK":
                print(f"OK  backend={backend}, dynamic={dynamic}, image_size={image_size}\n"
                      f"avg={r['avg']:.2f}ms, p50={r['p50']:.2f}ms, p99={r['p99']:.2f}ms, "
                      f"min={r['min']:.2f}ms, max={r['max']:.2f}ms")
                
                if args.compare and backend != "eager" and (dynamic, image_size) in eager_outputs:
                    ref = eager_outputs[(dynamic, image_size)]
                    out = r.get("output")
                    if out is not None:
                        out_cpu = out.cpu().float()
                        max_diff = (ref - out_cpu).abs().max().item()
                        mean_diff = (ref - out_cpu).abs().mean().item()
                        cos_sim = torch.nn.functional.cosine_similarity(
                            ref.flatten().unsqueeze(0), out_cpu.flatten().unsqueeze(0)
                        ).item()
                        allclose_1e3 = torch.allclose(ref, out_cpu, atol=1e-3, rtol=1e-3)
                        allclose_1e2 = torch.allclose(ref, out_cpu, atol=1e-2, rtol=1e-2)
                        print(f"  [Compare vs eager] max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}, "
                              f"cosine_sim={cos_sim:.8f}")
                        print(f"  [Compare vs eager] allclose(atol=1e-3)={allclose_1e3}, "
                              f"allclose(atol=1e-2)={allclose_1e2}")
                        del out_cpu
            else:
                print(f"FAIL  backend={backend}, dynamic={dynamic}, image_size={image_size}\n"
                      f"status={r['status'][:50]}")
    
    total_elapsed = time.perf_counter() - t_total
    print(f"Total benchmark time: {total_elapsed:.2f}s")


if __name__ == "__main__":
    main()

"""ST benchmark: Qwen3Model (32B) via ModelExecutor with different backends.

Uses ModelExecutor as entry point (simulating C++ calling pattern).
Routes to eager/torchair/inductor via python_graph_backend config.
Uses real NPU attention (npu_fused_infer_attention_score, prefill mode).

Configuration:
  - fullgraph=False, dynamic=False (hardcoded in CompileRunner)
  - Real attention computation via NpuPagedAttentionBackend (prefill mode)
  - KV cache write stubbed (no-op), KV cache tensors allocated

Usage:
  python3 tests/python/bench_qwen3_compile_runner.py
  python3 tests/python/bench_qwen3_compile_runner.py --backend eager
  python3 tests/python/bench_qwen3_compile_runner.py --image-size 256
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from dataclasses import dataclass
from typing import Optional
from unittest.mock import MagicMock

import torch
import torch.nn.functional as F
import torch_npu

_mock_ops = MagicMock()
sys.modules.setdefault("xllm.python.ops", _mock_ops)
sys.modules.setdefault("xllm.python.ops.compute", _mock_ops)

from xllm.python.attention.backend import AttentionMetadata
from xllm.python.model_executor.executor import ModelExecutor
from xllm.python.ops import reshape_paged_cache


# ---------------------------------------------------------------------------
# Op infrastructure: pure PyTorch replacements for xllm_ops C++ kernels.
# ---------------------------------------------------------------------------


def _py_rms_norm(input, weight, eps):
    variance = input.to(torch.float32).pow(2).mean(-1, keepdim=True)
    return (input * torch.rsqrt(variance + eps)).to(input.dtype) * weight


def _py_fused_add_rms_norm(input, residual, weight, eps):
    new_residual = input + residual
    variance = new_residual.to(torch.float32).pow(2).mean(-1, keepdim=True)
    normed = (new_residual * torch.rsqrt(variance + eps)).to(new_residual.dtype) * weight
    return normed, new_residual


def _py_silu_and_mul(input):
    gate, up = input.chunk(2, dim=-1)
    return F.silu(gate) * up


def _py_fused_qk_norm_rope(
    qkv, *, num_heads_q, num_heads_k, num_heads_v, head_dim, eps,
    q_weight, k_weight, cos_sin_cache, position_ids,
    cos=None, sin=None, interleaved=False,
):
    q_size = num_heads_q * head_dim
    kv_size = num_heads_k * head_dim
    qkv_f = qkv.to(torch.float32)
    q = qkv_f[:, :q_size].reshape(-1, num_heads_q, head_dim)
    k = qkv_f[:, q_size:q_size + kv_size].reshape(-1, num_heads_k, head_dim)
    v = qkv_f[:, q_size + kv_size:].to(qkv.dtype).reshape(-1, kv_size)
    q_rms = torch.rsqrt(q.pow(2).mean(-1, keepdim=True) + eps)
    k_rms = torch.rsqrt(k.pow(2).mean(-1, keepdim=True) + eps)
    q = (q * q_rms * q_weight.to(torch.float32)).to(qkv.dtype).reshape(-1, q_size)
    k = (k * k_rms * k_weight.to(torch.float32)).to(qkv.dtype).reshape(-1, kv_size)
    pos = position_ids.long()
    cos_sin = cos_sin_cache[pos]
    half = head_dim // 2
    cos_h = cos_sin[:, :half].unsqueeze(1)
    sin_h = cos_sin[:, half:].unsqueeze(1)
    q_r = q.reshape(-1, num_heads_q, head_dim)
    k_r = k.reshape(-1, num_heads_k, head_dim)
    q1, q2 = q_r[..., :half], q_r[..., half:]
    k1, k2 = k_r[..., :half], k_r[..., half:]
    q_out = torch.cat([q1 * cos_h - q2 * sin_h, q1 * sin_h + q2 * cos_h], dim=-1)
    k_out = torch.cat([k1 * cos_h - k2 * sin_h, k1 * sin_h + k2 * cos_h], dim=-1)
    return q_out.reshape(-1, q_size), k_out.reshape(-1, kv_size), v


def _install_op_infrastructure():
    # 方案1: 提高 recompile_limit，让所有 64 层都能被编译
    # 默认 recompile_limit=8，导致只有前 8 层被编译，其余回退到 eager
    import torch._dynamo.config
    torch._dynamo.config.recompile_limit = 128
    
    from xllm.python.layers.layernorm import RMSNorm
    from xllm.python.models.qwen3 import Qwen3MLP
    import xllm.python.models.qwen3 as qwen3_module
    import xllm.python.ops as ops_module

    RMSNorm.forward = lambda self, x, residual=None: (
        _py_rms_norm(x, self.weight, self.eps) if residual is None
        else _py_fused_add_rms_norm(x, residual, self.weight, self.eps)
    )
    Qwen3MLP.forward = lambda self, x: self.down_proj(
        _py_silu_and_mul(self.gate_up_proj(x))
    )
    ops_module.fused_qk_norm_rope = _py_fused_qk_norm_rope
    ops_module.reshape_paged_cache = lambda *args: None
    qwen3_module.record_layer_event = lambda layer_id: None


# ---------------------------------------------------------------------------
# AttentionMetadata for prefill mode
# ---------------------------------------------------------------------------


@dataclass
class _PrefillMetadata:
    slot_mapping: torch.Tensor
    q_cu_seq_lens: torch.Tensor
    is_prefill: bool = True
    is_chunked_prefill: bool = False
    paged_kv_indptr: Optional[torch.Tensor] = None
    paged_kv_indices: Optional[torch.Tensor] = None
    paged_kv_last_page_len: Optional[torch.Tensor] = None
    qo_indptr: Optional[torch.Tensor] = None
    kv_cu_seq_lens: Optional[torch.Tensor] = None
    kv_seq_lens_host: Optional[torch.Tensor] = None
    paged_kv_indptr_host: Optional[torch.Tensor] = None
    paged_kv_last_page_len_host: Optional[torch.Tensor] = None
    block_table: Optional[torch.Tensor] = None
    kv_seq_lens: Optional[torch.Tensor] = None


# ---------------------------------------------------------------------------
# Qwen3 32B configuration
# ---------------------------------------------------------------------------

QWEN3_32B_CONFIG = {
    "hidden_size": 5120,
    "n_layers": 64,
    "n_heads": 64,
    "n_kv_heads": 8,
    "head_dim": 128,
    "intermediate_size": 25600,
    "vocab_size": 16000,
    "max_position_embeddings": 40960,
    "device": "npu",
    "dtype": "float16",
}

IMAGE_SIZES = [256, 512, 800, 1024, 2048]
BACKENDS = ["eager", "torchair", "inductor"]


# ---------------------------------------------------------------------------
# Benchmark engine
# ---------------------------------------------------------------------------


def _create_model():
    from xllm.python.models.qwen3 import Qwen3ForCausalLM
    model = Qwen3ForCausalLM(QWEN3_32B_CONFIG)
    model.eval()
    return model


def _create_kv_caches(num_layers, device, dtype):
    """Allocate minimal KV cache tensors for prefill mode."""
    caches = []
    for _ in range(num_layers):
        k_cache = torch.zeros(1, 1, 8 * 128, dtype=dtype, device=device)
        v_cache = torch.zeros(1, 1, 8 * 128, dtype=dtype, device=device)
        caches.append((k_cache, v_cache, None))
    return caches


def _create_executor(model, backend_name):
    """Create ModelExecutor with specified backend."""
    cfg = QWEN3_32B_CONFIG.copy()
    
    if backend_name == "eager":
        cfg["python_graph_backend"] = "off"
    else:
        cfg["python_graph_backend"] = backend_name
    
    executor = ModelExecutor(model, cfg, max_seqs_per_batch=1)
    
    kv_caches = _create_kv_caches(cfg["n_layers"], torch.device("npu"), torch.float16)
    executor.bind_kv_caches(kv_caches)
    
    return executor, kv_caches


def _benchmark(executor, seq_len, warmup, repeats):
    device = torch.device("npu")
    vocab_size = QWEN3_32B_CONFIG["vocab_size"]
    input_ids = torch.randint(0, vocab_size, (seq_len,), device=device)
    positions = torch.arange(seq_len, device=device)

    slot_mapping = torch.zeros(seq_len, dtype=torch.int32, device=device)
    q_cu_seq_lens = torch.tensor([0, seq_len], dtype=torch.int32, device=device)
    metadata = _PrefillMetadata(
        slot_mapping=slot_mapping,
        q_cu_seq_lens=q_cu_seq_lens,
    )

    def run():
        with torch.no_grad():
            executor.execute(input_ids, positions, metadata)

    for _ in range(warmup):
        run()
    torch.npu.synchronize()

    latencies = []
    for _ in range(repeats):
        torch.npu.synchronize()
        t0 = time.perf_counter()
        run()
        torch.npu.synchronize()
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000)

    latencies.sort()
    avg = sum(latencies) / len(latencies)
    p50 = latencies[len(latencies) // 2]
    p99 = latencies[int(len(latencies) * 0.99)]
    return avg, p50, p99, min(latencies), max(latencies)


def _run_single(backend_name, image_size, warmup, repeats):
    result = {
        "backend": backend_name,
        "image_size": image_size,
        "status": "OK",
    }
    try:
        model = _create_model()
        executor, kv_caches = _create_executor(model, backend_name)
        avg, p50, p99, mn, mx = _benchmark(executor, image_size, warmup, repeats)
        result.update(avg=avg, p50=p50, p99=p99, min=mn, max=mx)
        del model, executor, kv_caches
        gc.collect()
        torch.npu.empty_cache()
    except torch.OutOfMemoryError:
        result["status"] = "OOM"
        torch.npu.empty_cache()
        gc.collect()
    except Exception as e:
        result["status"] = f"ERROR: {type(e).__name__}: {str(e)[:80]}"
        torch.npu.empty_cache()
        gc.collect()
    return result


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def _print_report(results):
    n_layers = QWEN3_32B_CONFIG["n_layers"]
    hidden = QWEN3_32B_CONFIG["hidden_size"]
    vocab = QWEN3_32B_CONFIG["vocab_size"]

    print()
    print("=" * 100)
    print("Qwen3Model 32B — ModelExecutor Benchmark Report (Real Attention)")
    print("=" * 100)
    print(f"Model:    Qwen3ForCausalLM ({n_layers} layers, hidden={hidden}, vocab={vocab})")
    print(f"Device:   NPU (Ascend 910, 61GB)")
    print(f"Mode:     fullgraph=False, dynamic=False")
    print(f"Entry:    ModelExecutor (simulates C++ calling pattern)")
    print(f"Attention: REAL (npu_fused_infer_attention_score, prefill mode, causal mask)")
    print(f"KV Cache: write stubbed (no-op), tensors allocated via bind_kv_caches")
    print(f"Op infra: pure PyTorch (rms_norm, silu_and_mul, fused_qk_norm_rope)")
    print()

    header = f"{'Backend':<12} {'ImgSize':<8} {'Avg(ms)':<12} {'P50(ms)':<12} {'P99(ms)':<12} {'Min(ms)':<12} {'Max(ms)':<12} {'Status'}"
    print(header)
    print("-" * len(header))
    for r in results:
        if r["status"] == "OK":
            print(
                f"{r['backend']:<12} {r['image_size']:<8} "
                f"{r['avg']:<12.2f} {r['p50']:<12.2f} {r['p99']:<12.2f} "
                f"{r['min']:<12.2f} {r['max']:<12.2f} {r['status']}"
            )
        else:
            print(
                f"{r['backend']:<12} {r['image_size']:<8} "
                f"{'N/A':<12} {'N/A':<12} {'N/A':<12} {'N/A':<12} {'N/A':<12} {r['status'][:40]}"
            )

    print()
    print("Notes:")
    print("  - vocab_size reduced to 16000 (from 151936) to fit 32B model in 61GB NPU.")
    print("  - image_size maps directly to sequence length (token count).")
    print("  - Attention uses real npu_fused_infer_attention_score in prefill mode.")
    print("  - KV cache write (reshape_paged_cache) is stubbed to no-op.")
    print("  - ModelExecutor routes to eager/torchair/inductor via python_graph_backend config.")
    print("=" * 100)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Qwen3Model 32B ModelExecutor benchmark")
    parser.add_argument("--backend", type=str, choices=["eager", "torchair", "inductor", "all"], default="all")
    parser.add_argument("--image-size", type=int, default=None, help="Single image_size to test (default: all)")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=str, default=None, help="Output JSON file path")
    args = parser.parse_args()

    _install_op_infrastructure()

    backends = BACKENDS if args.backend == "all" else [args.backend]
    image_sizes = [args.image_size] if args.image_size else IMAGE_SIZES

    total = len(backends) * len(image_sizes)
    print(f"Running {total} benchmark combinations via ModelExecutor ...")

    results = []
    idx = 0
    for backend in backends:
        for image_size in image_sizes:
            idx += 1
            print(f"[{idx}/{total}] backend={backend}, image_size={image_size} ...", end=" ", flush=True)
            r = _run_single(backend, image_size, args.warmup, args.repeats)
            results.append(r)
            if r["status"] == "OK":
                print(f"OK  avg={r['avg']:.2f}ms")
            else:
                print(f"{r['status'][:50]}")

    _print_report(results)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()

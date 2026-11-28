# Copyright 2026 The xLLM Authors.
# Licensed under the Apache License, Version 2.0.
"""Check against the actual unfused NPU operators, then mutate graph inputs."""

import json
from pathlib import Path

import pytest
import torch
import torch_npu
from torch._subclasses.fake_tensor import FakeTensorMode
from xllm_kernel.ops.normalization import rms_norm

from xllm import xllm_export
from xllm.python import initialize_runtime

initialize_runtime()

from xllm.python import kernels
from xllm.python.kernels_npu.triton.split_qkv_rmsnorm_mrope_opt01 import (
    build_axis_pattern,
    prepare_mrope,
    split_qkv_norm_mrope,
)


def _reference(
    x: torch.Tensor,
    qw: torch.Tensor,
    kw: torch.Tensor,
    positions: torch.Tensor,
    cache: torch.Tensor,
    hq: int,
    hk: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    n = x.shape[0]
    qsize, ksize = hq * 128, hk * 128
    q = rms_norm(x[:, :qsize].reshape(n * hq, 128), qw, 1e-6).reshape(n, qsize)
    k = rms_norm(x[:, qsize : qsize + ksize].reshape(n * hk, 128), kw, 1e-6).reshape(n, ksize)
    q, k = kernels.mrope(
        positions, q, k, cache, 128, mrope_section=[24, 20, 20], rotary_mode="half", cache_mode="interleave"
    )
    return q, k, x[:, qsize + ksize :]


def _check(
    out: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    reference: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> list[float]:
    errors = []
    for actual, expected in zip(out, reference):
        # This specialization is bitwise equal to the native BF16 path on the
        # validated stack; do not mask a regression with a broad tolerance.
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        errors.append(float((actual.float() - expected.float()).abs().max()))
    assert torch.equal(out[2], reference[2])
    return errors


def main() -> None:
    torch.npu.set_device(0)
    torch.manual_seed(101)
    torch.set_num_threads(2)
    dtype, device = torch.bfloat16, torch.device("npu:0")
    freq = torch.outer(
        torch.arange(4096, device=device).float(),
        1.0 / (5000000 ** (torch.arange(0, 128, 2, device=device).float() / 128)),
    )
    cache = torch.cat((freq.cos(), freq.sin()), dim=-1).to(dtype)
    qw = (torch.rand(128, device=device) + 0.5).to(dtype)
    kw = (torch.rand(128, device=device) + 0.5).to(dtype)
    axis = build_axis_pattern([24, 20, 20], True, device)
    results = []
    for n, hq, hk in ((17, 32, 8), (257, 32, 8), (2048, 32, 8), (65, 16, 4), (65, 8, 2), (65, 4, 1)):
        x = torch.randn(n, (hq + 2 * hk) * 128, device=device, dtype=dtype)
        positions = torch.randint(0, 4096, (3, n), device=device)
        table = prepare_mrope(cache, positions, axis)
        output = split_qkv_norm_mrope(x, qw, kw, table, hq, hk, 128, 1e-6)
        errors = _check(output, _reference(x, qw, kw, positions, cache, hq, hk))
        row = {"mode": "eager", "tokens": n, "q_heads": hq, "kv_heads": hk, "max_abs": errors}
        results.append(row)
        print(json.dumps(row), flush=True)
        if n == 257 or hq != 32:

            def forward(
                x: torch.Tensor = x, positions: torch.Tensor = positions, hq: int = hq, hk: int = hk
            ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                return split_qkv_norm_mrope(x, qw, kw, prepare_mrope(cache, positions, axis), hq, hk, 128, 1e-6)

            stream = torch.npu.Stream()
            graph = torch.npu.NPUGraph()
            with torch.npu.stream(stream):
                forward()
            torch.npu.synchronize()
            with torch.npu.graph(graph, stream=stream):
                captured = forward()
            previous = captured[0].clone()
            input_addresses = (x.data_ptr(), positions.data_ptr())
            for mutation in range(2):
                x.copy_(torch.randn_like(x) * (mutation + 1.5))
                positions.copy_(torch.randint(0, 4096, positions.shape, device=device))
                assert (x.data_ptr(), positions.data_ptr()) == input_addresses
                graph.replay()
                torch.npu.synchronize()
                errors = _check(captured, _reference(x, qw, kw, positions, cache, hq, hk))
                assert not torch.equal(captured[0], previous), "Graph read stale input data"
                previous.copy_(captured[0])
                row = {
                    "mode": "graph_replay_mutation",
                    "tokens": n,
                    "q_heads": hq,
                    "kv_heads": hk,
                    "mutation": mutation,
                    "max_abs": errors,
                    "external_input_addresses_unchanged": True,
                }
                results.append(row)
                print(json.dumps(row), flush=True)
            # Destroy the graph while its external input buffers still exist.
            # Later loop iterations replace x and positions with new shapes.
            torch.npu.synchronize()
            graph.reset()
            del captured, graph, stream
        with FakeTensorMode(allow_non_fake_inputs=True):
            fake = split_qkv_norm_mrope(x, qw, kw, prepare_mrope(cache, positions, axis), hq, hk, 128, 1e-6)
            assert tuple(t.shape for t in fake) == ((n, hq * 128), (n, hk * 128), (n, hk * 128))
            assert all(t.dtype == dtype for t in fake)
            assert all(t.device == device for t in fake)
        results.append({"mode": "fake_meta", "tokens": n, "q_heads": hq, "kv_heads": hk})
    Path("kernel_checks.json").write_text(json.dumps({"status": "passed", "results": results}, indent=2))


def test_opt01_kernel_and_graph_replay(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Expose the complete NPU check to the repository's pytest/CMake runner."""
    monkeypatch.chdir(tmp_path)
    with torch.inference_mode():
        main()


if __name__ == "__main__":
    with torch.inference_mode():
        main()

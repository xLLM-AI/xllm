# Copyright 2026 The xLLM Authors.
# Licensed under the Apache License, Version 2.0.
"""Deterministic positions and real model dispatch through the legacy branch."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import torch
import torch_npu
from xllm_kernel.ops.normalization import rms_norm

from xllm import xllm_export
from xllm.python import initialize_runtime

initialize_runtime()

from xllm.python import kernels
from xllm.python.kernels_npu.triton.split_qkv_rmsnorm_mrope_opt01 import prepare_mrope
from xllm.python.models.qwen3 import Qwen3Config
from xllm.python.models.qwen3_vl import Qwen3VLModel


class _PackedProjection(torch.nn.Module):
    def __init__(self, packed: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("packed", packed)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.packed


class _CaptureQkv(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.values: tuple[torch.Tensor, ...] = ()

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        self.values = (q.clone(), k.clone(), v.clone())
        return q


def _run_cases() -> list[dict[str, Any]]:
    torch.npu.set_device(0)
    torch.manual_seed(91001)
    device = torch.device("npu:0")
    results = []
    # 262144 is the actual Qwen3-VL-8B configured position-table capacity.
    cases = [
        ("supported_tp1", torch.bfloat16, 128, [24, 20, 20], 1, True),
        ("supported_tp2_local", torch.bfloat16, 128, [24, 20, 20], 2, True),
        ("supported_tp4_local", torch.bfloat16, 128, [24, 20, 20], 4, True),
        ("supported_tp8_local", torch.bfloat16, 128, [24, 20, 20], 8, True),
        ("fallback_sections", torch.bfloat16, 128, [16, 24, 24], 1, False),
        ("fallback_fp16", torch.float16, 128, [24, 20, 20], 1, False),
    ]
    previous_flag = os.environ.get("XLLM_OPT01_FUSED_MROPE")
    os.environ["XLLM_OPT01_FUSED_MROPE"] = "1"
    try:
        for name, dtype, dim, sections, tp, selected in cases:
            maximum = 262144 if name == "supported_tp1" else 4096
            cfg = Qwen3Config(
                hidden_size=128,
                n_layers=1,
                n_heads=32,
                n_kv_heads=8,
                head_dim=dim,
                intermediate_size=128,
                vocab_size=64,
                max_position_embeddings=maximum,
                tp_size=tp,
                mrope_section=sections,
                rope_scaling={"mrope_interleaved": True},
            )
            model = Qwen3VLModel(cfg, dtype, device)
            assert model._opt01_mrope is selected
            layer = model.layers[0].self_attn
            hq, hk = cfg.head_split()
            n = 17
            x = torch.randn(n, (hq + 2 * hk) * dim, device=device, dtype=dtype)
            layer.q_norm.weight.data.copy_((torch.rand(dim, device=device) + 0.5).to(dtype))
            layer.k_norm.weight.data.copy_((torch.rand(dim, device=device) + 0.5).to(dtype))
            positions = torch.tensor(
                [0, 1, maximum - 1, maximum - 2, 19, 20, 23, 24, 31, 32, 39, 40, 59, 60, 63, 64, maximum // 2],
                device=device,
            )
            positions = torch.stack((positions, positions.roll(5), positions.roll(11))).contiguous()
            layer.qkv_proj = _PackedProjection(x)
            capture = _CaptureQkv()
            layer.attn = capture
            layer.o_proj = torch.nn.Identity()
            cache = model._mrope_cos_sin
            cos = prepare_mrope(cache, positions, model._opt01_axis) if selected else None
            layer(positions, torch.zeros(n, 128, dtype=dtype, device=device), cache, cos, None, sections)
            qsize, ksize = hq * dim, hk * dim
            qref = rms_norm(x[:, :qsize].reshape(n * hq, dim), layer.q_norm.weight, cfg.rms_norm_eps).reshape(n, qsize)
            kref = rms_norm(
                x[:, qsize : qsize + ksize].reshape(n * hk, dim), layer.k_norm.weight, cfg.rms_norm_eps
            ).reshape(n, ksize)
            qref, kref = kernels.mrope(
                positions, qref, kref, cache, dim, mrope_section=sections, rotary_mode="half", cache_mode="interleave"
            )
            reference = (qref, kref, x[:, qsize + ksize :])
            errors = []
            for actual, expected in zip(capture.values, reference):
                # Existing real-operator comparisons are bitwise equal on this
                # supported dtype/layout; enforce that rather than widening a tolerance.
                torch.testing.assert_close(actual, expected, atol=0, rtol=0)
                errors.append(float((actual.float() - expected.float()).abs().max()))
            results.append(
                {
                    "case": name,
                    "fused_selected": selected,
                    "dtype": str(dtype),
                    "head_dim": dim,
                    "local_heads": [hq, hk],
                    "sections": sections,
                    "position_min": int(positions.min()),
                    "position_max": int(positions.max()),
                    "table_capacity": maximum,
                    "max_abs": errors,
                }
            )
            del layer, model, capture, cache, cos, x, reference, qref, kref
        return results
    finally:
        if previous_flag is None:
            os.environ.pop("XLLM_OPT01_FUSED_MROPE", None)
        else:
            os.environ["XLLM_OPT01_FUSED_MROPE"] = previous_flag


def test_position_boundaries_and_fallback_forward(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.chdir(tmp_path)
    with torch.inference_mode():
        results = _run_cases()
    Path("boundary_checks.json").write_text(json.dumps({"status": "passed", "results": results}, indent=2))


if __name__ == "__main__":
    with torch.inference_mode():
        rows = _run_cases()
    Path("boundary_checks.json").write_text(json.dumps({"status": "passed", "results": rows}, indent=2))

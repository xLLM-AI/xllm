# Copyright 2026 The xLLM Authors.
# Licensed under the Apache License, Version 2.0.
"""Check real model initialization selects supported and legacy paths up front."""

import json
import os
from pathlib import Path

import pytest
import torch
import torch_npu
from torch._subclasses.fake_tensor import FakeTensorMode

from xllm import xllm_export
from xllm.python import initialize_runtime

initialize_runtime()

from xllm.python.models.qwen3 import Qwen3Config
from xllm.python.models.qwen3_vl import Qwen3VLModel


def _config(**changes: object) -> Qwen3Config:
    values = dict(
        hidden_size=128,
        n_layers=1,
        n_heads=32,
        n_kv_heads=8,
        head_dim=128,
        intermediate_size=128,
        max_position_embeddings=64,
        vocab_size=64,
        mrope_section=[24, 20, 20],
        rope_scaling={"mrope_interleaved": True},
    )
    values.update(changes)
    return Qwen3Config(**values)


def main() -> None:
    results = []
    cases = [
        ("tp1", {}, torch.bfloat16, "npu", "1", True),
        ("flat_model_args", {"rope_scaling": {}}, torch.bfloat16, "npu", "1", True),
        ("tp2", {"tp_size": 2}, torch.bfloat16, "npu", "1", True),
        ("tp4", {"tp_size": 4}, torch.bfloat16, "npu", "1", True),
        ("tp8", {"tp_size": 8}, torch.bfloat16, "npu", "1", True),
        ("disabled", {}, torch.bfloat16, "npu", "0", False),
        ("fp16", {}, torch.float16, "npu", "1", False),
        ("cpu", {}, torch.bfloat16, "cpu", "1", False),
        ("head_dim64", {"head_dim": 64, "mrope_section": [8, 12, 12]}, torch.bfloat16, "npu", "1", False),
        ("different_heads", {"n_heads": 24}, torch.bfloat16, "npu", "1", False),
        ("different_sections", {"mrope_section": [16, 24, 24]}, torch.bfloat16, "npu", "1", False),
        ("different_layout", {"rope_scaling": {"mrope_interleaved": False}}, torch.bfloat16, "npu", "1", False),
    ]
    with FakeTensorMode():
        for name, changes, dtype, device, flag, expected in cases:
            os.environ["XLLM_OPT01_FUSED_MROPE"] = flag
            model = Qwen3VLModel(_config(**changes), dtype, torch.device(device))
            assert model._opt01_mrope is expected, name
            assert all(layer.self_attn._opt01_mrope is expected for layer in model.layers), name
            assert hasattr(model, "_opt01_axis") is expected, name
            results.append({"case": name, "fused_selected": expected})
    Path("dispatch_checks.json").write_text(json.dumps({"status": "passed", "results": results}, indent=2))


def test_opt01_initialization_dispatch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Run initialization cases through pytest while restoring the process flag."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("XLLM_OPT01_FUSED_MROPE", os.getenv("XLLM_OPT01_FUSED_MROPE", "0"))
    main()


if __name__ == "__main__":
    main()

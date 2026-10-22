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

"""Failure signals and frozen coverage of the full-checkpoint numerical CLI."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import Olmo2Config as HfConfig
from transformers import Olmo2ForCausalLM as HfModel

from tools import validate_olmo2_numerics as validation
from xllm.python.models.olmo2 import Olmo2ForCausalLM
from xllm.python.platform import current_platform

pytestmark = pytest.mark.skipif(not current_platform.is_cuda(), reason="OLMo2 numerical validation targets CUDA")

_LIMITS = {"rmse": 0.01, "max_abs": 0.02, "nrmse_cap": 0.02, "cosine_floor": 0.999}


@pytest.mark.parametrize("fault", ["shape", "empty", "nan", "inf", "overflow", "numeric"])
def test_invalid_numeric_results_cannot_pass(fault: str) -> None:
    actual, reference = torch.ones(2, 4), torch.ones(2, 4)
    if fault == "shape":
        actual = actual[:1]
    elif fault == "empty":
        actual, reference = actual[:0], reference[:0]
    elif fault == "nan":
        actual[0, 0] = float("nan")
    elif fault == "inf":
        reference[0, 0] = float("inf")
    elif fault == "overflow":
        actual.fill_(3e38)
    else:
        actual[0, 0] = 0.9
    result = validation._compare(actual, reference, _LIMITS)
    assert not result["pass"]
    json.dumps(result, allow_nan=False)


def test_equal_zero_and_nonzero_tensors_pass() -> None:
    for value in (torch.zeros(2, 4), torch.ones(2, 4)):
        assert validation._compare(value, value, _LIMITS)["pass"]


def test_frozen_protocol_includes_boundaries_chunking_and_full_budget_sets() -> None:
    fixture = json.loads(validation._FIXTURE.read_text())
    assert len(fixture["node_budgets"]) == 291
    assert len(fixture["logits_budgets"]) == 64
    assert validation._keys(fixture) == fixture["logits_budgets"].keys()
    cases = fixture["cases"]
    assert {127, 128, 129}.issubset({case["length"] for case in cases})
    assert max(case["length"] + case["decode"] for case in cases) == 4096
    assert any(case.get("initial", case["length"]) < case["length"] for case in cases)
    assert fixture["input_ids"] == validation._sequence(fixture, 25)
    assert validation._required_cache_blocks(fixture) == 64


def test_cache_capacity_accounts_for_disjoint_reference_and_mixed_requests() -> None:
    fixture = {
        "input_ids": [1],
        "cases": [{"name": name, "length": 128, "decode": 1} for name in ("a", "b", "c")],
        "batches": [["a", "b", "c"]],
    }
    assert validation._required_cache_blocks(fixture) == 6
    fixture["batches"] = []
    assert validation._required_cache_blocks(fixture) == 4


def test_cache_capacity_rejects_truncated_matrix_before_checkpoint_load(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setattr(
        validation.sys,
        "argv",
        [
            "validator",
            "validate",
            "--model",
            str(tmp_path),
            "--reference",
            str(tmp_path),
            "--output",
            str(tmp_path / "output"),
            "--cache-blocks",
            "63",
        ],
    )
    with pytest.raises(SystemExit) as error:
        validation.main()
    assert error.value.code == 2
    assert not (tmp_path / "output").exists()


def test_missing_reference_node_and_nonfinite_reference_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "reference.pt"
    torch.save({"node": torch.ones(2)}, path)
    with pytest.raises(ValueError, match="tensor keys"):
        validation._load_tensors(path, {"node", "logits"})
    torch.save({"node": torch.tensor([float("nan")])}, path)
    with pytest.raises(ValueError, match="invalid reference tensor"):
        validation._load_tensors(path, {"node"})


def test_tampered_reference_cannot_pass_manifest_check(tmp_path: Path) -> None:
    identity = {"config.json": "frozen config", "model.safetensors": "frozen weights"}
    files = {}
    for kind in ("nodes", "logits"):
        for dtype in ("bf16", "fp32"):
            path = tmp_path / f"{kind}-{dtype}.pt"
            path.write_bytes(b"original evidence")
            files[path.name] = validation._sha(path)
    manifest = {
        "fixture_sha256": validation._sha(validation._FIXTURE),
        "checkpoint": identity,
        "files": files,
        "versions": {"transformers": "5.12.1"},
    }
    validation._save(tmp_path / "manifest.json", manifest)
    validation._reference_manifest(tmp_path, identity)
    (tmp_path / "logits-bf16.pt").write_bytes(b"modified evidence")
    with pytest.raises(ValueError, match="integrity"):
        validation._reference_manifest(tmp_path, identity)


def test_incomplete_matrix_is_rejected_instead_of_silent_success() -> None:
    fixture = {"cases": [{"name": "one", "decode": 0}], "batches": [], "logits_budgets": {"one.0": _LIMITS}}
    with pytest.raises(ValueError, match="output matrix"):
        validation._compare_outputs({"cached": {}, "full": {}, "batch": {}}, {}, fixture)


@pytest.mark.parametrize("passed", [False, True])
def test_main_propagates_numerical_failure_exit_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, passed: bool
) -> None:
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setattr(validation, "_checkpoint", lambda *args: {})
    monkeypatch.setattr(validation, "_validate", lambda *args: passed)
    monkeypatch.setattr(
        validation.sys,
        "argv",
        [
            "validator",
            "validate",
            "--model",
            str(tmp_path),
            "--reference",
            str(tmp_path),
            "--output",
            str(tmp_path / "output"),
        ],
    )
    assert validation.main() == (0 if passed else 1)


def test_output_comparisons_cover_hf_cache_and_tp1_disagreement() -> None:
    fixture: dict[str, Any] = {
        "cases": [{"name": "one", "decode": 0}],
        "batches": [["one"]],
        "logits_budgets": {"one.0": _LIMITS},
    }
    good = torch.ones(1, 4)
    outputs = {"cached": {"one.0": good}, "full": {"one.0": good}, "batch": {"one:one.0": good}}
    baseline = {"cached": {"one.0": good * 0.9}, "full": {"one.0": good}, "batch": {"one:one.0": good}}
    rows = validation._compare_outputs(outputs, {"one.0": good}, fixture, baseline)
    assert len(rows) == 8
    assert [(row["mode"], row["target"]) for row in rows if not row["pass"]] == [("cached", "tp1")]


@pytest.mark.parametrize("tp,rank", [(1, 0), (2, 0), (2, 1)])
def test_weight_audit_accepts_dtype_rounding_and_rejects_corrupt_shards(tmp_path: Path, tp: int, rank: int) -> None:
    cfg = HfConfig(
        hidden_size=16,
        num_hidden_layers=16,
        num_attention_heads=4,
        num_key_value_heads=4,
        intermediate_size=24,
        vocab_size=32,
        max_position_embeddings=32,
        tie_word_embeddings=False,
    )
    torch.manual_seed(19)
    hf = HfModel(cfg).to(torch.bfloat16)
    state = {key: value.contiguous() for key, value in hf.state_dict().items()}
    # BF16 values in FP16's subnormal range undergo legitimate dtype rounding.
    state["model.embed_tokens.weight"][0, :] = 5.8115e-6
    save_file(state, tmp_path / "model.safetensors")
    model = Olmo2ForCausalLM({**cfg.to_dict(), "dtype": "float16", "device": "cpu", "tp_size": tp, "tp_rank": rank})
    with safe_open(str(tmp_path / "model.safetensors"), framework="pt", device="cpu") as handle:
        model.load_weights([validation._State(handle)], rank, tp)
    result = validation._audit_weights(model, tmp_path, rank, tp)
    assert result["checked"] == 131 and result["checkpoint_keys"] == 179 and result["all_equal"]
    assert result["rows"][0]["conversion_changed_elements"] > 0
    with torch.no_grad():
        model.model.layers[5].self_attn.q_norm.weight[0].add_(0.5)
    with pytest.raises(ValueError, match="checkpoint shard mismatch.*layers.5.self_attn.q_norm.weight"):
        validation._audit_weights(model, tmp_path, rank, tp)

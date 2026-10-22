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

"""Reproduce the frozen OLMo2 checkpoint acceptance on real CUDA TP=1/2.

Generate an independent HF reference first, then validate FP16 or BF16 without
calibrating against xLLM outputs. Numerical failures are evidence, not skips.
Use torchrun for TP=2; each rank writes its own report and shares the exit status.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import math
import os
import sys
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
from scripts.logger import logger  # noqa: E402
from tools.olmo2_numerics import _BLOCK, _Capture, _Runner  # noqa: E402

_FIXTURE = _ROOT / "tests/models/fixtures/olmo2/numerics.json"
_LOCAL_NODES = frozenset(
    (
        "q_proj",
        "k_proj",
        "v_proj",
        "q_norm",
        "k_norm",
        "q_rope",
        "k_rope",
        "attention",
        "gate_proj",
        "up_proj",
        "silu_mul",
    )
)


def _save(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def _sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _sequence(fixture: dict[str, Any], length: int, offset: int = 0) -> list[int]:
    ids = fixture["input_ids"]
    return [ids[(i + offset) % len(ids)] for i in range(length)]


def _compare(actual: torch.Tensor, reference: torch.Tensor, limits: dict[str, float]) -> dict[str, Any]:
    """Fail closed on shape/non-finite errors; retain the original v2 criteria."""
    result: dict[str, Any] = {
        "pass": False,
        "actual_shape": list(actual.shape),
        "reference_shape": list(reference.shape),
    }
    if actual.shape != reference.shape or not actual.numel():
        result["reason"] = "shape mismatch or empty tensor"
        return result
    a, b = actual.detach().float().cpu(), reference.detach().float().cpu()
    if not bool(torch.isfinite(a).all() and torch.isfinite(b).all()):
        result["reason"] = "non-finite tensor"
        return result
    diff = a - b
    rmse = float(diff.square().mean().sqrt())
    rms = float(b.square().mean().sqrt())
    product = float(a.norm() * b.norm())
    cosine = float(a.flatten() @ b.flatten()) / product if product else float(torch.equal(a, b))
    maximum = float(diff.abs().max())
    nrmse = rmse / max(rms, 1e-30)
    if not all(math.isfinite(value) for value in (rmse, rms, cosine, maximum, nrmse)):
        result["reason"] = "non-finite error metric"
        return result
    result.update(rmse=rmse, max_abs=maximum, nrmse=nrmse, cosine=cosine, budget=limits)
    result["pass"] = bool(
        rmse <= limits["rmse"]
        and maximum <= limits["max_abs"]
        and nrmse <= limits["nrmse_cap"]
        and cosine >= limits["cosine_floor"]
    )
    if a.shape[-1] == 100352:
        top = b.topk(2, dim=-1).values
        stable = top[..., 0] - top[..., 1] > 2 * limits["max_abs"]
        mismatches = int((stable & (a.argmax(-1) != b.argmax(-1))).sum())
        result.update(stable_top1_positions=int(stable.sum()), stable_top1_mismatches=mismatches)
        result["pass"] = result["pass"] and mismatches == 0
    return result


def _checkpoint(model: Path, fixture: dict[str, Any]) -> dict[str, str]:
    hashes = {name: _sha(model / name) for name in fixture["checkpoint"]}
    if hashes != fixture["checkpoint"]:
        raise ValueError("checkpoint differs from the frozen revision; use the original BF16 config and weights")
    return hashes


def _keys(fixture: dict[str, Any]) -> set[str]:
    return {f"{case['name']}.{step}" for case in fixture["cases"] for step in range(case["decode"] + 1)}


def _load_tensors(path: Path, keys: set[str]) -> dict[str, torch.Tensor]:
    values = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(values, dict) or values.keys() != keys:
        raise ValueError(f"incomplete or unexpected tensor keys: {path}")
    for key, value in values.items():
        if not isinstance(value, torch.Tensor) or not value.numel() or not bool(torch.isfinite(value).all()):
            raise ValueError(f"invalid reference tensor {path}:{key}")
    return values


def _reference_manifest(folder: Path, identity: dict[str, str]) -> dict[str, Any]:
    manifest = json.loads((folder / "manifest.json").read_text())
    if manifest["fixture_sha256"] != _sha(_FIXTURE) or manifest["checkpoint"] != identity:
        raise ValueError("reference provenance differs from the frozen fixture/checkpoint")
    if manifest["versions"]["transformers"] != "5.12.1":
        raise ValueError("the frozen reference requires Transformers 5.12.1")
    expected = {f"{kind}-{dtype}.pt" for kind in ("nodes", "logits") for dtype in ("bf16", "fp32")}
    if manifest["files"].keys() != expected:
        raise ValueError("reference manifest must contain all four HF tensor files")
    for name, digest in manifest["files"].items():
        if _sha(folder / name) != digest:
            raise ValueError(f"reference integrity check failed: {name}")
    return manifest


def _provenance(identity: dict[str, str]) -> dict[str, Any]:
    return {
        "fixture_sha256": _sha(_FIXTURE),
        "checkpoint": identity,
        "versions": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "flashinfer-python")},
        "model_source_sha256": _sha(_ROOT / "xllm/python/models/olmo2.py"),
        "runtime_sources": {
            str(path.relative_to(_ROOT)): _sha(path) for path in sorted((_ROOT / "xllm/python").rglob("*.py"))
        },
        "gpu": torch.cuda.get_device_name(),
        "bf16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
    }


def _generate(model_path: Path, output: Path, fixture: dict[str, Any], identity: dict[str, str]) -> None:
    from transformers import Olmo2ForCausalLM

    if importlib.metadata.version("transformers") != "5.12.1":
        raise ValueError("the frozen HF reference requires transformers==5.12.1")
    for tag, dtype in (("bf16", torch.bfloat16), ("fp32", torch.float32)):
        model = (
            Olmo2ForCausalLM.from_pretrained(
                str(model_path), dtype=dtype, attn_implementation="eager", local_files_only=True
            )
            .to("cuda:0")
            .eval()
        )
        tokens = torch.tensor([fixture["input_ids"]], device="cuda:0")
        capture = _Capture(model, "hf")
        try:
            with torch.inference_mode():
                logits = model(tokens, attention_mask=torch.ones_like(tokens), use_cache=False).logits[0]
            capture.nodes["logits"] = logits.float().cpu()
        finally:
            capture.close()
        if capture.nodes.keys() != fixture["node_budgets"].keys():
            raise ValueError("HF node trace differs from frozen B06")
        torch.save(capture.nodes, output / f"nodes-{tag}.pt")
        del capture, logits
        outputs = {}
        with torch.inference_mode():
            for case in fixture["cases"]:
                for step in range(case["decode"] + 1):
                    ids = _sequence(fixture, case["length"] + step, case["offset"])
                    tokens = torch.tensor([ids], device="cuda:0")
                    outputs[f"{case['name']}.{step}"] = (
                        model(tokens, attention_mask=torch.ones_like(tokens), use_cache=False, logits_to_keep=1)
                        .logits[0]
                        .float()
                        .cpu()
                    )
                logger.info(f"HF {tag}: {case['name']} complete")
        torch.save(outputs, output / f"logits-{tag}.pt")
        del model, outputs, tokens
        gc.collect()
        torch.cuda.empty_cache()
    manifest = _provenance(identity)
    manifest["reference"] = "unmodified HF eager; BF16 acceptance target; FP32 diagnostic only"
    manifest["files"] = {p.name: _sha(p) for p in output.glob("*.pt")}
    _save(output / "manifest.json", manifest)


def _hf_control(args: argparse.Namespace, fixture: dict[str, Any], identity: dict[str, str]) -> bool:
    """Measure the unchanged HF implementation against its own frozen reference."""
    from transformers import Olmo2ForCausalLM

    manifest = _reference_manifest(args.reference, identity)
    if importlib.metadata.version("transformers") != manifest["versions"]["transformers"]:
        raise ValueError("HF control and reference must use the same Transformers version")
    refs = _load_tensors(args.reference / "logits-bf16.pt", _keys(fixture))
    model = (
        Olmo2ForCausalLM.from_pretrained(
            str(args.model), dtype=torch.bfloat16, attn_implementation="eager", local_files_only=True
        )
        .to("cuda:0")
        .eval()
    )
    outputs: dict[str, dict[str, torch.Tensor]] = {"cached": {}, "batch": {}}
    with torch.inference_mode():
        for case in fixture["cases"]:
            n, steps, name = case["length"], case["decode"], case["name"]
            history = _sequence(fixture, n + steps, case["offset"])
            cache = None
            for step in range(steps + 1):
                tokens = torch.tensor([history[:n] if step == 0 else history[n + step - 1 : n + step]], device="cuda:0")
                positions = torch.arange(0 if step == 0 else n + step - 1, n + step, device="cuda:0")[None]
                result = model(
                    tokens,
                    attention_mask=torch.ones((1, n + step), device="cuda:0", dtype=torch.long),
                    position_ids=positions,
                    past_key_values=cache,
                    use_cache=True,
                    logits_to_keep=1,
                )
                cache = result.past_key_values
                outputs["cached"][f"{name}.{step}"] = result.logits[0].float().cpu()
        for names in fixture["batches"]:
            cases = [next(c for c in fixture["cases"] if c["name"] == name) for name in names]
            lengths = [c["length"] for c in cases]
            steps = min(c["decode"] for c in cases)
            histories = [_sequence(fixture, n + steps, c["offset"]) for c, n in zip(cases, lengths, strict=True)]
            width = max(lengths)
            mask = torch.tensor([[0] * (width - n) + [1] * n for n in lengths], device="cuda:0")
            tokens = torch.tensor(
                [[0] * (width - n) + h[:n] for h, n in zip(histories, lengths, strict=True)], device="cuda:0"
            )
            positions = (mask.cumsum(-1) - 1).clamp_min(0)
            cache = None
            for step in range(steps + 1):
                if step:
                    tokens = torch.tensor(
                        [[h[n + step - 1]] for h, n in zip(histories, lengths, strict=True)], device="cuda:0"
                    )
                    positions = torch.tensor([[n + step - 1] for n in lengths], device="cuda:0")
                    mask = torch.cat((mask, torch.ones((len(names), 1), dtype=mask.dtype, device=mask.device)), -1)
                result = model(
                    tokens,
                    attention_mask=mask,
                    position_ids=positions,
                    past_key_values=cache,
                    use_cache=True,
                    logits_to_keep=1,
                )
                cache = result.past_key_values
                for i, name in enumerate(names):
                    outputs["batch"]["+".join(names) + f":{name}.{step}"] = result.logits[i].float().cpu()
    rows = []
    for mode, values in outputs.items():
        for key, value in values.items():
            source = key.split(":")[-1]
            rows.append({"mode": mode, "key": key, **_compare(value, refs[source], fixture["logits_budgets"][source])})
    passed = all(row["pass"] for row in rows)
    torch.save(outputs, args.output / "hf-control.pt")
    _save(
        args.output / "result.json",
        {
            **_provenance(identity),
            "implementation": "unmodified HF eager BF16 cache/batch",
            "status": "PASS" if passed else "FAIL",
            "comparisons": len(rows),
            "failed": [r for r in rows if not r["pass"]],
            "rows": rows,
        },
    )
    (logger.info if passed else logger.error)(
        f"Official HF control: {sum(not row['pass'] for row in rows)}/{len(rows)} failures"
    )
    return passed


class _State:
    def __init__(self, handle: Any) -> None:
        self._handle = handle

    def keys(self) -> list[str]:
        return list(self._handle.keys())

    def has(self, name: str) -> bool:
        return name in self._handle.keys()

    def get_tensor(self, name: str) -> torch.Tensor:
        return self._handle.get_tensor(name)


def _load_model(path: Path, dtype: str, rank: int, tp: int) -> Any:
    from xllm import xllm_export  # noqa: F401
    from xllm.python import initialize_runtime

    initialize_runtime()
    from xllm.python.models.olmo2 import Olmo2ForCausalLM

    config = json.loads((path / "config.json").read_text())
    config.update(dtype=dtype, device=f"cuda:{rank}", tp_size=tp, tp_rank=rank, world_size=tp)
    model = Olmo2ForCausalLM(config).eval()
    with ExitStack() as stack:
        handles = [
            stack.enter_context(safe_open(str(p), framework="pt", device="cpu"))
            for p in sorted(path.glob("*.safetensors"))
        ]
        model.load_weights([_State(handle) for handle in handles], rank, tp)
    return model


def _audit_weights(model: Any, path: Path, rank: int, tp: int) -> dict[str, Any]:
    rows = []
    with safe_open(str(path / "model.safetensors"), framework="pt", device="cpu") as state:

        def _check(actual: torch.Tensor, keys: list[str], dim: int | None) -> None:
            values = [state.get_tensor(key) for key in keys]
            if dim is not None:
                values = [value.chunk(tp, dim=dim)[rank] for value in values]
            expected = torch.cat(values, dim=0) if len(values) > 1 else values[0]
            converted = expected.to(actual.dtype)
            equal = torch.equal(actual.detach().cpu(), converted)
            conversion_error = (converted.float() - expected.float()).abs()
            rows.append(
                {
                    "keys": keys,
                    "shape": list(actual.shape),
                    "shard_dim": dim,
                    "equal": equal,
                    "conversion_changed_elements": int((conversion_error != 0).sum()),
                    "conversion_max_abs": float(conversion_error.max()),
                }
            )
            if not equal:
                raise ValueError(f"checkpoint shard mismatch: {keys}")

        _check(model.model.embed_tokens.weight, ["model.embed_tokens.weight"], 1)
        _check(model.model.norm.weight, ["model.norm.weight"], None)
        _check(model.lm_head.weight, ["lm_head.weight"], 0)
        for i, layer in enumerate(model.model.layers):
            p = f"model.layers.{i}."
            _check(
                layer.self_attn.qkv_proj.weight, [p + f"self_attn.{name}_proj.weight" for name in ("q", "k", "v")], 0
            )
            _check(layer.self_attn.o_proj.weight, [p + "self_attn.o_proj.weight"], 1)
            for name in ("q", "k"):
                _check(getattr(layer.self_attn, name + "_norm").weight, [p + f"self_attn.{name}_norm.weight"], 0)
            for name in ("post_attention_layernorm", "post_feedforward_layernorm"):
                _check(getattr(layer, name).weight, [p + name + ".weight"], None)
            _check(layer.mlp.gate_up_proj.weight, [p + f"mlp.{name}_proj.weight" for name in ("gate", "up")], 0)
            _check(layer.mlp.down_proj.weight, [p + "mlp.down_proj.weight"], 1)
    keys = [key for row in rows for key in row["keys"]]
    if len(rows) != 131 or sorted(keys) != list(model.loaded_checkpoint_keys):
        raise ValueError("weight audit did not cover every loaded tensor")
    return {"checked": len(rows), "checkpoint_keys": 179, "all_equal": all(row["equal"] for row in rows), "rows": rows}


def _matrix(runner: _Runner, fixture: dict[str, Any]) -> dict[str, dict[str, torch.Tensor]]:
    outputs: dict[str, dict[str, torch.Tensor]] = {"cached": {}, "full": {}, "batch": {}}
    for case in fixture["cases"]:
        n, steps, name = case["length"], case["decode"], case["name"]
        history = _sequence(fixture, n + steps, case["offset"])
        pages = runner.pages([n + steps])
        initial = case.get("initial", n)
        actual = runner.run([history[:initial]], [0], pages, "prefill", last_only=True)
        if initial < n:
            actual = runner.run([history[initial:n]], [initial], pages, "chunked", last_only=True)
        for step in range(steps + 1):
            if step:
                actual = runner.run([[history[n + step - 1]]], [n + step - 1], pages, "decode", last_only=True)
            key = f"{name}.{step}"
            outputs["cached"][key] = actual.float().cpu()
            # Recomputed prefill writes disjoint physical pages, preserving decode history.
            full_pages = runner.pages([n + steps, n + steps])[1]
            full = runner.run([history[: n + step]], [0], [full_pages], "prefill", last_only=True)
            outputs["full"][key] = full.float().cpu()
        logger.info(f"xLLM TP{runner.model.cfg.tp_size}: {name} complete")
    for names in fixture["batches"]:
        cases = [next(case for case in fixture["cases"] if case["name"] == name) for name in names]
        lengths = [case["length"] for case in cases]
        steps = min(case["decode"] for case in cases)
        histories = [_sequence(fixture, n + steps, c["offset"]) for n, c in zip(lengths, cases, strict=True)]
        pages = runner.pages([n + steps for n in lengths])
        actual = runner.run(
            [h[:n] for h, n in zip(histories, lengths, strict=True)], [0] * len(names), pages, "prefill", last_only=True
        )
        for step in range(steps + 1):
            if step:
                actual = runner.run(
                    [[h[n + step - 1]] for h, n in zip(histories, lengths, strict=True)],
                    [n + step - 1 for n in lengths],
                    pages,
                    "decode",
                    last_only=True,
                )
            for i, name in enumerate(names):
                outputs["batch"]["+".join(names) + f":{name}.{step}"] = actual[i : i + 1].float().cpu()
    return outputs


def _compare_outputs(
    outputs: dict[str, dict[str, torch.Tensor]],
    reference: dict[str, torch.Tensor],
    fixture: dict[str, Any],
    baseline: dict[str, dict[str, torch.Tensor]] | None = None,
) -> list[dict[str, Any]]:
    _check_output_keys(outputs, fixture)
    if baseline is not None:
        _check_output_keys(baseline, fixture)
    rows = []
    for mode, tensors in outputs.items():
        for key, value in tensors.items():
            source = key.split(":")[-1]
            limits = fixture["logits_budgets"][source]
            rows.append({"mode": mode, "key": key, "target": "hf", **_compare(value, reference[source], limits)})
            if mode != "cached":
                rows.append(
                    {"mode": mode, "key": key, "target": "cached", **_compare(value, outputs["cached"][source], limits)}
                )
            if baseline is not None:
                rows.append({"mode": mode, "key": key, "target": "tp1", **_compare(value, baseline[mode][key], limits)})
    return rows


def _check_output_keys(outputs: dict[str, dict[str, torch.Tensor]], fixture: dict[str, Any]) -> None:
    batch_keys = set()
    for names in fixture["batches"]:
        steps = min(next(c["decode"] for c in fixture["cases"] if c["name"] == name) for name in names)
        batch_keys.update("+".join(names) + f":{name}.{step}" for name in names for step in range(steps + 1))
    expected = {"cached": _keys(fixture), "full": _keys(fixture), "batch": batch_keys}
    if outputs.keys() != expected.keys() or any(outputs[mode].keys() != keys for mode, keys in expected.items()):
        raise ValueError("incomplete or unexpected output matrix")


def _required_cache_blocks(fixture: dict[str, Any]) -> int:
    # Cached and independently recomputed histories need disjoint pages.
    pages = {case["name"]: math.ceil((case["length"] + case["decode"]) / _BLOCK) for case in fixture["cases"]}
    return max(
        math.ceil(len(fixture["input_ids"]) / _BLOCK),
        2 * max(pages.values()),
        max((sum(pages[name] for name in batch) for batch in fixture["batches"]), default=0),
    )


def _validate(args: argparse.Namespace, fixture: dict[str, Any], identity: dict[str, str], rank: int) -> bool:
    _reference_manifest(args.reference, identity)
    nodes = _load_tensors(args.reference / "nodes-bf16.pt", set(fixture["node_budgets"]))
    reference = _load_tensors(args.reference / "logits-bf16.pt", _keys(fixture))
    baseline_nodes, baseline_outputs = None, None
    provenance = _provenance(identity)
    if args.tp == 2:
        if args.baseline is None:
            raise ValueError("TP2 requires --baseline from a matching TP1 validation")
        previous = json.loads((args.baseline / "result-rank0.json").read_text())
        for key in (
            "fixture_sha256",
            "checkpoint",
            "model_source_sha256",
            "runtime_sources",
            "versions",
            "bf16_reduced_precision_reduction",
        ):
            if previous[key] != provenance[key]:
                raise ValueError(f"TP1 baseline differs: {key}")
        if previous["tp"] != 1 or previous["dtype"] != args.dtype:
            raise ValueError("TP1 baseline must have matching dtype and TP=1")
        if previous["reference_manifest_sha256"] != _sha(args.reference / "manifest.json"):
            raise ValueError("TP1 baseline uses a different HF reference")
        if previous["files"].keys() != {"nodes-rank0.pt", "outputs-rank0.pt"}:
            raise ValueError("TP1 baseline must cover both tensor files")
        for name, digest in previous["files"].items():
            if _sha(args.baseline / name) != digest:
                raise ValueError(f"TP1 baseline integrity check failed: {name}")
        baseline_nodes = _load_tensors(args.baseline / "nodes-rank0.pt", set(nodes))
        baseline_outputs = torch.load(args.baseline / "outputs-rank0.pt", weights_only=True)
    model = _load_model(args.model, args.dtype, rank, args.tp)
    from xllm import xllm_export

    provenance["native_extension_sha256"] = _sha(Path(xllm_export.__file__))
    if args.tp == 2 and previous["native_extension_sha256"] != provenance["native_extension_sha256"]:
        raise ValueError("TP1 baseline uses a different native extension")
    if args.tp == 2:
        from xllm.python import distributed

        distributed.init_tp_group(args.tp_host, args.tp_port, rank, args.tp, f"cuda:{rank}", rank, args.tp, 0)
    _save(args.output / f"weights-rank{rank}.json", _audit_weights(model, args.model, rank, args.tp))
    runner = _Runner(model, blocks=args.cache_blocks)
    for cache in runner.caches:
        cache.key.fill_(float("nan"))
        cache.value.fill_(float("nan"))
    capture = _Capture(model, "xllm", runner)
    try:
        logits = runner.run([fixture["input_ids"]], [0], runner.pages([len(fixture["input_ids"])]), "prefill")
        capture.nodes["logits"] = logits.float().cpu()
    finally:
        capture.close()
    if capture.nodes.keys() != nodes.keys():
        raise ValueError("xLLM node trace differs from frozen B06")
    torch.save(capture.nodes, args.output / f"nodes-rank{rank}.pt")
    rows = []
    for key, value in capture.nodes.items():
        sharded = args.tp == 2 and key.split(".")[-1] in _LOCAL_NODES
        expected = nodes[key].chunk(2, -1)[rank] if sharded else nodes[key]
        rows.append({"node": key, "target": "hf", **_compare(value, expected, fixture["node_budgets"][key])})
        if baseline_nodes is not None:
            expected = baseline_nodes[key].chunk(2, -1)[rank] if sharded else baseline_nodes[key]
            rows.append({"node": key, "target": "tp1", **_compare(value, expected, fixture["node_budgets"][key])})
    outputs = _matrix(runner, fixture)
    torch.save(outputs, args.output / f"outputs-rank{rank}.pt")
    matrix_rows = _compare_outputs(outputs, reference, fixture, baseline_outputs)
    rows.extend(matrix_rows)
    passed = all(row["pass"] for row in rows)
    rank_equal = True
    if args.tp == 2:
        # All ranks must receive the same vocabulary after the real all-gather.
        import torch.distributed as dist

        for mode in outputs.values():
            for value in mode.values():
                local = value.to(model.device)
                other = local.clone()
                dist.broadcast(other, src=0)
                rank_equal = rank_equal and torch.equal(local, other)
        flag = torch.tensor(int(passed and rank_equal), device=model.device)
        dist.all_reduce(flag, op=dist.ReduceOp.MIN)
        passed = bool(flag.item())
    result = {
        **provenance,
        "tp": args.tp,
        "rank": rank,
        "dtype": args.dtype,
        "checkpoint_keys": len(model.loaded_checkpoint_keys),
        "cache_blocks": args.cache_blocks,
        "status": "PASS" if passed else "FAIL",
        "rank_logits_equal": rank_equal,
        "reference_manifest_sha256": _sha(args.reference / "manifest.json"),
        "files": {name: _sha(args.output / name) for name in (f"nodes-rank{rank}.pt", f"outputs-rank{rank}.pt")},
        "comparisons": len(rows),
        "failed": [row for row in rows if not row["pass"]],
        "rows": rows,
    }
    _save(args.output / f"result-rank{rank}.json", result)
    (logger.info if passed else logger.error)(
        f"TP{args.tp} rank{rank}: {result['status']}; {len(result['failed'])}/{len(rows)} failures"
    )
    if args.tp == 2:
        from xllm.python.distributed.collectives import _shutdown_process_groups

        _shutdown_process_groups()
    return passed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("reference", "validate", "hf-control"))
    parser.add_argument(
        "--model", type=Path, required=True, help="original checkpoint, including unchanged BF16 config"
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="new evidence directory; never overwrite an earlier run"
    )
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--baseline", type=Path, help="matching TP1 output directory, required for TP2")
    parser.add_argument("--tp", type=int, choices=(1, 2), default=1)
    parser.add_argument(
        "--tp-port", type=int, default=29649, help="xLLM rendezvous port, distinct from torchrun's port"
    )
    parser.add_argument("--tp-host", default="127.0.0.1", help="single-node xLLM rendezvous address")
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    parser.add_argument("--cache-blocks", type=int, default=80, help="Physical KV pages; must cover the full matrix.")
    args = parser.parse_args()
    world, rank = int(os.environ.get("WORLD_SIZE", "1")), int(os.environ.get("LOCAL_RANK", "0"))
    if world != args.tp or (args.mode != "validate" and world != 1):
        parser.error("launch TP2 with torchrun --nproc-per-node=2; reference is single-process")
    if int(os.environ.get("RANK", "0")) != rank or not 0 <= rank < world:
        parser.error("this validator supports one node with local rank equal to global rank")
    if args.mode != "reference" and args.reference is None:
        parser.error("validate/hf-control requires --reference")
    if not __debug__:
        parser.error("validation requires assertions enabled; do not use python -O")
    fixture = json.loads(_FIXTURE.read_text())
    if args.mode == "validate" and args.cache_blocks < _required_cache_blocks(fixture):
        parser.error(f"the complete frozen matrix requires at least {_required_cache_blocks(fixture)} cache blocks")
    identity = _checkpoint(args.model, fixture)
    if rank == 0:
        args.output.mkdir(parents=True, exist_ok=False)
    # Rank1 writes only after xLLM's TP rendezvous, which rank0 reaches after
    # creating the directory. No second process group or filesystem polling.
    torch.cuda.set_device(rank)
    torch.manual_seed(20261020)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    if args.mode == "reference":
        _generate(args.model, args.output, fixture, identity)
        return 0
    if args.mode == "hf-control":
        return 0 if _hf_control(args, fixture, identity) else 1
    return 0 if _validate(args, fixture, identity, rank) else 1


if __name__ == "__main__":
    sys.exit(main())

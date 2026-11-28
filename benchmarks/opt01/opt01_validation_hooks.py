# Copyright 2026 The xLLM Authors.
# Licensed under the Apache License, Version 2.0.
"""Offline-only instrumentation; never installed by the production feature."""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any

import torch
from xllm_kernel.ops.normalization import rms_norm

from xllm.python import kernels


def _write(path: Path, data: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2))
    temporary.replace(path)


class _Measurement:
    def __init__(self, model: torch.nn.Module, root: Path) -> None:
        self.root = root
        self.device = next(model.parameters()).device
        self.phase = ""
        self.rows: dict[str, dict] = {}
        self.current: tuple[Any, ...] | None = None
        model.register_forward_pre_hook(self._before)
        model.register_forward_hook(self._after)
        threading.Thread(target=self._sample, daemon=True).start()

    def _sample(self) -> None:
        torch.npu.set_device(self.device)
        control = self.root / "control.json"
        while True:
            if not control.exists():
                time.sleep(0.05)
                continue
            command = json.loads(control.read_text())
            label = command["phase"]
            changed = False
            if label != self.phase:
                # Client waits for this acknowledgement before sending the wave.
                # Reset is outside any model execution or graph capture.
                torch.npu.synchronize()
                torch.npu.reset_peak_memory_stats()
                free, total = torch.npu.mem_get_info()
                self.rows[label] = {
                    "armed": True,
                    "completed": False,
                    "samples": 0,
                    "allocated_before_bytes": torch.npu.memory_allocated(),
                    "reserved_before_bytes": torch.npu.memory_reserved(),
                    "sampled_device_used_before_bytes": total - free,
                    "sampled_device_used_peak_bytes": total - free,
                    "torch_peak_allocated_bytes": 0,
                    "torch_peak_reserved_bytes": 0,
                }
                self.phase = label
                changed = True
            row = self.rows[label]
            if not row["completed"]:
                free, total = torch.npu.mem_get_info()
                values = {
                    "sampled_device_used_peak_bytes": total - free,
                    "torch_peak_allocated_bytes": torch.npu.max_memory_allocated(),
                    "torch_peak_reserved_bytes": torch.npu.max_memory_reserved(),
                }
                for key, value in values.items():
                    if value > row[key]:
                        row[key] = value
                        changed = True
                row["samples"] += 1
                if command["state"] == "finish":
                    torch.npu.synchronize()
                    row["torch_peak_allocated_bytes"] = torch.npu.max_memory_allocated()
                    row["torch_peak_reserved_bytes"] = torch.npu.max_memory_reserved()
                    row["allocated_after_bytes"] = torch.npu.memory_allocated()
                    row["reserved_after_bytes"] = torch.npu.memory_reserved()
                    row["completed"] = True
                    changed = True
            if changed:
                _write(
                    self.root / "memory.json",
                    {"phases": self.rows, "sample_period_seconds": 0.05, "device_hbm_is_sampled_lower_bound": True},
                )
            time.sleep(0.05)

    def _before(self, module: torch.nn.Module, args: tuple) -> None:
        positions = args[1]
        self.current = None
        if positions.dim() != 2 or not self.phase:
            return
        start, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
        host = time.perf_counter()
        start.record()
        self.current = (self.phase, positions.shape[1], start, end, host)

    def _after(self, module: torch.nn.Module, args: tuple, output: torch.Tensor) -> None:
        if self.current is None:
            return
        phase, tokens, start, end, host = self.current
        end.record()
        end.synchronize()
        row = {
            "phase": phase,
            "tokens": tokens,
            "decoder_prefill_device_stream_ms": start.elapsed_time(end),
            "decoder_prefill_host_synchronized_ms": 1000 * (time.perf_counter() - host),
            "scope": "Qwen3VLModel.forward; vision encoder, LM head, queue/network excluded",
        }
        with (self.root / "prefill.jsonl").open("a") as target:
            target.write(json.dumps(row) + "\n")
        self.current = None


def _install_attention(model: torch.nn.Module, root: Path) -> None:
    for decoder in model.layers:
        layer = decoder.self_attn
        state: dict[str, Any] = {}

        def before(module: torch.nn.Module, args: tuple, state: dict[str, Any] = state) -> None:
            positions, _, cache, _, _, sections = args
            state.clear()
            if positions.dim() == 2 and sections is not None:
                state.update(positions=positions, cache=cache, sections=sections)

        def projected(
            module: torch.nn.Module, args: tuple, output: torch.Tensor, state: dict[str, Any] = state
        ) -> None:
            if state:
                state["qkv"] = output

        def attention(
            module: torch.nn.Module,
            args: tuple,
            output: torch.Tensor,
            state: dict[str, Any] = state,
            layer: Any = layer,
        ) -> torch.Tensor | None:
            if not state:
                return None
            q, k, v = args
            packed = state["qkv"]
            n, dim, hq, hk = packed.shape[0], layer.head_dim, layer.num_heads, layer.num_kv_heads
            qs, ks = hq * dim, hk * dim
            qref = rms_norm(packed[:, :qs].reshape(n * hq, dim), layer.q_norm.weight, layer.q_norm.eps).reshape(n, qs)
            kref = rms_norm(
                packed[:, qs : qs + ks].reshape(n * hk, dim), layer.k_norm.weight, layer.k_norm.eps
            ).reshape(n, ks)
            qref, kref = kernels.mrope(
                state["positions"],
                qref,
                kref,
                state["cache"],
                dim,
                mrope_section=state["sections"],
                rotary_mode="half",
                cache_mode="interleave",
            )
            vref = packed[:, qs + ks :]
            # Backend output may alias a persistent workspace. Preserve actual
            # fused data before invoking the reference on identical metadata/KV slots.
            saved_actual = output.clone()
            reference = module.forward(qref, kref, vref)
            tensors = list(zip((q, k, v, saved_actual), (qref, kref, vref, reference)))
            errors = [float((a.float() - b.float()).abs().max()) for a, b in tensors]
            changed = [int((a != b).sum()) for a, b in tensors]
            with (root / "attention.jsonl").open("a") as target:
                target.write(
                    json.dumps(
                        {
                            "layer": layer.layer_id,
                            "tokens": n,
                            "max_abs_q_k_v_attention": errors,
                            "changed_elements": changed,
                            "dtype": str(saved_actual.dtype),
                            "device": str(saved_actual.device),
                            "attention_shape": list(saved_actual.shape),
                            "kv_and_metadata": "same backend, same context, same layer and cache slots",
                        }
                    )
                    + "\n"
                )
            for actual, expected in tensors:
                torch.testing.assert_close(actual, expected, atol=0, rtol=0)
            # Keep fused results flowing into o_proj; reference data is never substituted.
            return saved_actual

        layer.register_forward_pre_hook(before)
        layer.qkv_proj.register_forward_hook(projected)
        layer.attn.register_forward_hook(attention)


def install(model: torch.nn.Module) -> None:
    root = Path(os.environ["XLLM_OPT01_TEST_DIR"])
    root.mkdir(parents=True, exist_ok=True)
    mode = os.environ["XLLM_OPT01_TEST_MODE"]
    if mode == "attention":
        _install_attention(model, root)
    elif mode == "measure":
        model._opt01_measurement = _Measurement(model, root)
    else:
        raise ValueError(f"Unknown offline instrumentation mode: {mode}")

# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""Versioned RMSNorm cases shared by numerical and performance validation."""

from __future__ import annotations

import argparse
import math
from collections.abc import Callable
from dataclasses import asdict, dataclass
from itertools import product
from typing import TYPE_CHECKING, Any

from xllm_kernel.numerics.suite import OperationSuite, PreparedCase
from xllm_kernel.registry import PreparedKernel

if TYPE_CHECKING:
    import torch


def _report_value(value: float) -> float | str:
    return value if math.isfinite(value) else str(value)


@dataclass(frozen=True)
class NumericsCase:
    rows: int
    width: int
    dtype: str
    seed: int = 5309
    eps: float = 1e-5
    operation: str = "rms_norm"
    contract_version: int = 1
    comparison_version: int = 1

    @property
    def case_id(self) -> str:
        return f"rms_norm/{self.dtype}/{self.rows}x{self.width}/seed{self.seed}"

    @property
    def parameters(self) -> dict[str, Any]:
        return asdict(self)

    def unsupported_reason(self, plan: PreparedKernel) -> str | None:
        if plan.spec.solution == "xlite" and (
            self.dtype == "float32" or self.width % 64 or self.width > 8192 or self.rows * self.width > 2**32 - 1
        ):
            return "outside declared xlite RMSNorm tensor domain"
        return None

    def prepare(self, plan: PreparedKernel) -> PreparedCase:
        import torch_npu

        value, weight = inputs(self)
        return PreparedCase(
            function=lambda: plan.function(value, weight, self.eps),
            validate=lambda function: verify(lambda *_: function(), self, value, weight),
            metadata={
                "input_stride": list(value.stride()),
                "weight_stride": list(weight.stride()),
                "npu_format": torch_npu.get_npu_format(value),
            },
        )


def reference(value: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Independent FP32 CPU mathematical reference, rounded to output dtype."""
    import torch

    value_fp32 = value.cpu().float()
    weight_fp32 = weight.cpu().float()
    return (value_fp32 * torch.rsqrt(value_fp32.square().mean(-1, keepdim=True) + eps) * weight_fp32).to(value.dtype)


def inputs(case: NumericsCase) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate on CPU so providers receive identical logical input values."""
    import torch

    generator = torch.Generator(device="cpu").manual_seed(case.seed)
    dtype = getattr(torch, case.dtype)
    value = torch.randn(case.rows, case.width, generator=generator).to(dtype)
    weight = torch.randn(case.width, generator=generator).to(dtype)
    return value.npu(), weight.npu()


def verify(
    function: Callable[..., torch.Tensor],
    case: NumericsCase,
    value: torch.Tensor,
    weight: torch.Tensor,
) -> dict[str, Any]:
    """Check values, metadata and side effects before any timing is accepted."""
    import torch

    value_before, weight_before = value.cpu().clone(), weight.cpu().clone()
    expected = reference(value_before, weight_before, case.eps)
    output = function(value, weight, case.eps)
    actual = output.cpu()
    tolerance = {"bfloat16": 2e-2, "float16": 2e-3, "float32": 2e-5}[case.dtype]
    error = (actual.float() - expected.float()).abs()
    normalized_error = error / (tolerance + tolerance * expected.float().abs())
    finite = torch.isfinite(actual)
    failures = ~torch.isclose(actual.float(), expected.float(), rtol=tolerance, atol=tolerance, equal_nan=True)
    metadata = output.shape == value.shape and output.dtype == value.dtype and output.device == value.device
    mutation = not torch.equal(value.cpu(), value_before) or not torch.equal(weight.cpu(), weight_before)
    alias = torch._C._is_alias_of(output, value) or torch._C._is_alias_of(output, weight)
    passed = metadata and not mutation and not alias and not bool(failures.any())
    worst = int(error.nan_to_num(posinf=float("inf")).flatten().argmax()) if error.numel() else None
    return {
        "case_id": case.case_id,
        "case": asdict(case),
        "status": "passed" if passed else "failed",
        "reference": "cpu_fp32_rms_norm_v1",
        "tolerance": {"rtol": tolerance, "atol": tolerance},
        "shape_dtype_device_match": metadata,
        "input_mutation": mutation,
        "aliases_inputs": alias,
        "failing_elements": int(failures.sum()),
        "failing_fraction": float(failures.float().mean()) if failures.numel() else 0.0,
        "max_absolute_error": float(error.max()) if error.numel() and bool(torch.isfinite(error).all()) else None,
        "nonfinite_actual": int((~finite).sum()),
        "worst_flat_index": worst,
        "worst_actual": _report_value(float(actual.flatten()[worst])) if worst is not None else None,
        "worst_reference": _report_value(float(expected.flatten()[worst])) if worst is not None else None,
        "max_normalized_error": _report_value(float(normalized_error.max())) if error.numel() else 0.0,
    }


def _add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--rows", type=int, nargs="+", default=[1, 16, 128, 4096])
    parser.add_argument("--widths", type=int, nargs="+", default=[512, 2048, 6144])
    parser.add_argument(
        "--dtypes", nargs="+", choices=["bfloat16", "float16", "float32"], default=["bfloat16", "float16"]
    )


def _cases(args: argparse.Namespace) -> tuple[NumericsCase, ...]:
    if min(args.widths) <= 0 or min(args.rows) < 0:
        raise ValueError("widths must be positive; rows must be nonnegative")
    return tuple(
        NumericsCase(rows, width, dtype) for dtype, width, rows in product(args.dtypes, args.widths, args.rows)
    )


def _prepare(device: str, mode: str, implementation: str) -> PreparedKernel:
    from xllm_kernel.ops.rms_norm import prepare

    return prepare(device, mode, implementation)


SUITE = OperationSuite(
    operation="rms_norm",
    implementations=("npu.aclnn.rms_norm.xllm", "npu.aclnn.rms_norm.torch_npu", "npu.xlite.rms_norm"),
    add_arguments=_add_arguments,
    cases=_cases,
    prepare=_prepare,
)

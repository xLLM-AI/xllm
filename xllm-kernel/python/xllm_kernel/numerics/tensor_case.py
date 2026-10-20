# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""Tensor-tree checks and reset lifecycle shared by operation-owned examples."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any

from xllm_kernel.numerics.suite import PreparedCase
from xllm_kernel.registry import PreparedKernel


def _leaves(value: Any, prefix: str = "") -> Iterator[tuple[str, Any]]:
    import torch

    if isinstance(value, torch.Tensor):
        yield prefix, value
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            yield from _leaves(item, f"{prefix}/{index}")
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from _leaves(item, f"{prefix}/{key}")
    elif value is not None:
        raise TypeError(f"expected a tensor tree, got {type(value).__name__}")


def compare(actual: Any, expected: Any, tolerances: dict[str, tuple[float, float]]) -> dict[str, Any]:
    """Compare structure, metadata and finite values, with explicit per-leaf tolerances."""
    import torch

    got, want = dict(_leaves(actual)), dict(_leaves(expected))
    passed = got.keys() == want.keys()
    details = {}
    for key in want.keys() & got.keys():
        value, reference = got[key], want[key]
        metadata = value.shape == reference.shape and value.dtype == reference.dtype
        default = {torch.bfloat16: (2e-2, 2e-3), torch.float16: (2e-3, 2e-3), torch.float32: (1e-4, 1e-5)}.get(
            reference.dtype, (0.0, 0.0)
        )
        rtol, atol = tolerances.get(key, default)
        finite = bool(torch.isfinite(value).all())
        matches = (
            metadata
            and finite
            and bool(
                torch.isclose(
                    value.detach().cpu().double(),
                    reference.detach().cpu().double(),
                    rtol=rtol,
                    atol=atol,
                ).all()
            )
        )
        details[key] = {"passed": matches, "metadata_match": metadata, "finite": finite, "rtol": rtol, "atol": atol}
        passed = passed and matches
    return {"status": "passed" if passed else "failed", "outputs": details}


@dataclass
class Example:
    """One invocation and an independently computed CPU result.

    Inputs are named so mutation contracts are checked by name. Mutated inputs
    are restored outside device-event intervals and outside graph capture.
    ``observe`` exposes state written by void/out-buffer operations.
    """

    inputs: dict[str, Any]
    invoke: Callable[[Callable[..., Any]], Any]
    expected: Any
    tolerances: dict[str, tuple[float, float]] = field(default_factory=dict)
    observe: Callable[[Any], Any] = lambda result: result
    expected_mutations: dict[str, Any] = field(default_factory=dict)
    canonicalize: Callable[[Any], Any] = lambda result: result
    canonicalize_mutations: Callable[[Any], Any] = lambda result: result
    metadata: dict[str, Any] = field(default_factory=dict)

    def bind(self, plan: PreparedKernel) -> PreparedCase:
        import torch

        inputs = dict(_leaves(self.inputs))
        original = {name: value.detach().clone() for name, value in inputs.items()}
        cpu_original = {name: value.cpu() for name, value in original.items()}
        mutation_names = set(plan.spec.contract.mutates)
        mutated = {name: value for name, value in inputs.items() if name.split("/")[1] in mutation_names}
        expected_mutations = dict(_leaves(self.expected_mutations))
        if mutated.keys() != expected_mutations.keys():
            raise ValueError("example must provide expected contents for every mutated input")

        def reset() -> None:
            for name, value in mutated.items():
                value.copy_(original[name])

        def function() -> Any:
            return self.observe(self.invoke(plan.function))

        def validate(call: Callable[[], Any]) -> dict[str, Any]:
            result = call()
            checked = compare(self.canonicalize(result), self.canonicalize(self.expected), self.tolerances)
            output_tensors = list(_leaves(result))
            device_match = all(value.device.type in ("npu", "privateuseone") for _, value in output_tensors)
            readonly_unchanged = all(
                torch.equal(value.cpu(), cpu_original[name]) for name, value in inputs.items() if name not in mutated
            )
            mutation_check = compare(
                self.canonicalize_mutations(dict(mutated)),
                self.canonicalize_mutations(expected_mutations),
                self.tolerances,
            )
            aliases = any(
                torch._C._is_alias_of(value, source) for _, value in output_tensors for source in inputs.values()
            )
            passed = (
                checked["status"] == "passed"
                and mutation_check["status"] == "passed"
                and readonly_unchanged
                and device_match
                and (plan.spec.contract.aliases_inputs or not aliases)
            )
            return {
                **checked,
                "status": "passed" if passed else "failed",
                "readonly_inputs_unchanged": readonly_unchanged,
                "mutations": mutation_check,
                "output_device_match": device_match,
                "aliases_inputs": aliases,
                "reference": "operation_owned_cpu_reference",
            }

        return PreparedCase(function, validate, self.metadata, reset=reset if mutated else None)


@dataclass(frozen=True)
class TensorCase:
    """Describe an operation's example without importing a device SDK at discovery."""

    operation: str
    dimensions: tuple[int, ...]
    dtype: str
    build: Callable[[tuple[int, ...], str, int], Example]
    seed: int = 5309
    restrict: Callable[[PreparedKernel, tuple[int, ...], str], str | None] = lambda *_: None

    @property
    def case_id(self) -> str:
        return f"{self.operation}/{self.dtype}/{'x'.join(map(str, self.dimensions))}/seed{self.seed}"

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "operation": self.operation,
            "dimensions": self.dimensions,
            "dtype": self.dtype,
            "seed": self.seed,
            "comparison_version": 1,
        }

    def unsupported_reason(self, plan: PreparedKernel) -> str | None:
        return self.restrict(plan, self.dimensions, self.dtype)

    def prepare(self, plan: PreparedKernel) -> PreparedCase:
        return self.build(self.dimensions, self.dtype, self.seed).bind(plan)

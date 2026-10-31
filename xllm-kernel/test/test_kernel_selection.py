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

"""Automatic selection, multi-operation initialization and policy immutability."""

from __future__ import annotations

import os
from dataclasses import replace

import pytest
from xllm_kernel import runtime
from xllm_kernel.catalog import BackendModule, Operation
from xllm_kernel.registry import KernelRegistry, KernelSpec, OpContract
from xllm_kernel.selection import resolve_overrides, select_kernel


@pytest.fixture
def registry(monkeypatch: pytest.MonkeyPatch) -> KernelRegistry:
    result = KernelRegistry()
    monkeypatch.setattr(KernelRegistry, "_instance", result)
    monkeypatch.setattr(runtime, "_prepared", None)
    monkeypatch.setattr(runtime, "_configuration", None)
    # Do not inherit a developer's selection policy into CPU test doubles.
    for key in tuple(os.environ):
        if key.startswith("XLLM_KERNEL_OVERRIDE_"):
            monkeypatch.delenv(key)
    return result


def _contract(name: str) -> OpContract:
    return OpContract(name, 1, (), False, "test")


def _spec(contract: OpContract, solution: str, priority: int = 0) -> KernelSpec:
    return KernelSpec(f"{contract.operation}.{solution}", contract, "cpu", ("eager",), "test", priority, solution)


def test_loads_all_operations_then_selects_and_freezes(
    monkeypatch: pytest.MonkeyPatch, registry: KernelRegistry
) -> None:
    add, multiply = _contract("add"), _contract("multiply")
    operations = (
        Operation(add, (BackendModule("add.backends", "cpu", ()),)),
        Operation(multiply, (BackendModule("multiply.backends", "cpu", ()),)),
    )
    imported = []

    def load(module: str) -> None:
        imported.append(module)
        if module == "add.backends":
            registry.register(_spec(add, "slow"), lambda value: value + 100)
            registry.register(_spec(add, "fast", 10), lambda value: value + 1)
        else:
            registry.register(_spec(multiply, "native"), lambda value: value * 2)

    monkeypatch.setattr(runtime, "load_operations", lambda: operations)
    monkeypatch.setattr(runtime, "import_module", load)
    result = runtime.initialize(device="cpu", execution_mode="eager")
    assert result["add"].function(3) == 4
    assert result["multiply"].function(3) == 6
    assert operations[0].get_prepared() is result["add"]
    assert operations[1].get_prepared() is result["multiply"]
    assert runtime.initialize(device="cpu", execution_mode="eager") is result
    assert imported == ["add.backends", "multiply.backends"]
    with pytest.raises(TypeError):
        result["add"] = result["multiply"]
    with pytest.raises(RuntimeError, match="frozen"):
        registry.register(_spec(add, "late"), lambda value: value)
    monkeypatch.setenv("XLLM_KERNEL_OVERRIDE_ADD", "slow")
    with pytest.raises(RuntimeError, match="restart"):
        runtime.initialize(device="cpu", execution_mode="eager")
    assert result["add"].function(3) == 4


def _one_operation(
    monkeypatch: pytest.MonkeyPatch,
    registry: KernelRegistry,
) -> Operation:
    operation = Operation(_contract("add"), (BackendModule("add.backends", "cpu", ()),))
    monkeypatch.setattr(runtime, "load_operations", lambda: (operation,))
    monkeypatch.setattr(runtime, "import_module", lambda module: None)
    registry.register(_spec(operation.contract, "native", 10), lambda value: value + 1)
    registry.register(_spec(operation.contract, "custom"), lambda value: value + 2)
    return operation


def test_requested_subset_does_not_import_unused_collectives(
    monkeypatch: pytest.MonkeyPatch, registry: KernelRegistry
) -> None:
    add = _one_operation(monkeypatch, registry)
    collective = Operation(_contract("collective"), (BackendModule("requires_process_group", "cpu", ()),))
    monkeypatch.setattr(runtime, "load_operations", lambda: (add, collective))
    imported = []
    monkeypatch.setattr(runtime, "import_module", imported.append)
    result = runtime.initialize(device="cpu", execution_mode="eager", operations=("add",))
    assert tuple(result) == ("add",)
    assert imported == ["add.backends"]
    assert collective.prepared is None
    with pytest.raises(RuntimeError, match="restart"):
        runtime.initialize(device="cpu", execution_mode="eager")
    assert imported == ["add.backends"]


@pytest.mark.parametrize("operations", [(), ("missing",), ("add", "add")])
def test_invalid_operation_set_fails_before_import(
    monkeypatch: pytest.MonkeyPatch, registry: KernelRegistry, operations: tuple[str, ...]
) -> None:
    _one_operation(monkeypatch, registry)
    imported = []
    monkeypatch.setattr(runtime, "import_module", imported.append)
    with pytest.raises(ValueError, match="invalid kernel operation set"):
        runtime.initialize(device="cpu", execution_mode="eager", operations=operations)
    assert not imported


def test_offline_preparation_uses_startup_selection_and_rejects_environment_redirection(
    monkeypatch: pytest.MonkeyPatch, registry: KernelRegistry
) -> None:
    operation = _one_operation(monkeypatch, registry)
    monkeypatch.setenv("XLLM_KERNEL_OVERRIDE_ADD", "custom")
    with pytest.raises(ValueError, match="environment override changed"):
        runtime.prepare_operation("add", "cpu", "eager", "add.native")
    assert operation.get_prepared().spec.name == "add.custom"


def test_missing_optional_parent_does_not_probe_its_child(
    monkeypatch: pytest.MonkeyPatch, registry: KernelRegistry
) -> None:
    operation = _one_operation(monkeypatch, registry)
    operation.backends += (BackendModule("optional.backend", "cpu", ("missing_parent", "missing_parent.extension")),)
    probed = []

    def probe(name: str) -> None:
        probed.append(name)
        assert name == "missing_parent"
        return None

    monkeypatch.setattr(runtime, "find_spec", probe)
    result = runtime.initialize(device="cpu", execution_mode="eager")
    assert probed == ["missing_parent"]
    assert result["add"].excluded[-1] == ("optional.backend", "missing dependencies: missing_parent")


@pytest.mark.parametrize("target", ["custom", "add.custom"])
def test_solution_and_exact_name_overrides_select_lower_priority(
    monkeypatch: pytest.MonkeyPatch, registry: KernelRegistry, target: str
) -> None:
    _one_operation(monkeypatch, registry)
    result = runtime.initialize(device="cpu", execution_mode="eager", overrides={"add": target})
    assert result["add"].spec.name == "add.custom"
    assert result["add"].function(3) == 5


def test_environment_has_precedence_over_mapping(monkeypatch: pytest.MonkeyPatch, registry: KernelRegistry) -> None:
    _one_operation(monkeypatch, registry)
    monkeypatch.setenv("XLLM_KERNEL_OVERRIDE_ADD", "custom")
    result = runtime.initialize(device="cpu", execution_mode="eager", overrides={"add": "native"})
    assert result["add"].spec.solution == "custom"
    assert runtime.initialize(device="cpu", execution_mode="eager") is result


@pytest.mark.parametrize("override", ["missing", "add.wrong_device", "add.wrong_contract", "add.wrong_mode"])
def test_explicit_override_never_bypasses_compatibility_or_falls_back(
    monkeypatch: pytest.MonkeyPatch, registry: KernelRegistry, override: str
) -> None:
    operation = _one_operation(monkeypatch, registry)
    bad = _spec(operation.contract, "wrong_device", 100)
    registry.register(replace(bad, device="npu"), lambda value: value)
    registry.register(replace(bad, name="add.wrong_contract", contract=_contract("other")), lambda value: value)
    registry.register(replace(bad, name="add.wrong_mode", execution_modes=("aclgraph",)), lambda value: value)
    with pytest.raises(ValueError, match="no kernel for add"):
        runtime.initialize(device="cpu", execution_mode="eager", overrides={"add": override})
    assert operation.prepared is None
    assert runtime._prepared is None


def test_optional_missing_backend_is_reported_without_import(
    monkeypatch: pytest.MonkeyPatch, registry: KernelRegistry
) -> None:
    operation = _one_operation(monkeypatch, registry)
    operation.backends += (BackendModule("add.optional", "cpu", ("missing_sdk",)),)
    monkeypatch.setattr(runtime, "find_spec", lambda name: None)
    imported = []
    monkeypatch.setattr(runtime, "import_module", imported.append)
    result = runtime.initialize(device="cpu", execution_mode="eager")
    assert imported == ["add.backends"]
    assert result["add"].spec.solution == "native"
    assert ("add.optional", "missing dependencies: missing_sdk") in result["add"].excluded


def test_broken_installed_backend_propagates_without_default_fallback(
    monkeypatch: pytest.MonkeyPatch, registry: KernelRegistry
) -> None:
    operation = _one_operation(monkeypatch, registry)

    def fail(module: str) -> None:
        raise RuntimeError("launcher ABI mismatch")

    monkeypatch.setattr(runtime, "import_module", fail)
    with pytest.raises(RuntimeError, match="launcher ABI mismatch"):
        runtime.initialize(device="cpu", execution_mode="eager")
    assert operation.prepared is None and runtime._prepared is None


def test_missing_second_operation_does_not_publish_first_binding(
    monkeypatch: pytest.MonkeyPatch, registry: KernelRegistry
) -> None:
    operation = _one_operation(monkeypatch, registry)
    second = Operation(_contract("missing"), (BackendModule("missing.backend", "cpu", ("sdk",)),))
    monkeypatch.setattr(runtime, "load_operations", lambda: (operation, second))
    monkeypatch.setattr(runtime, "find_spec", lambda name: None)
    with pytest.raises(ValueError, match="missing dependencies: sdk"):
        runtime.initialize(device="cpu", execution_mode="eager")
    assert operation.prepared is second.prepared is runtime._prepared is None


def test_unknown_mapping_and_environment_operations_are_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ValueError, match="unknown kernel override operation"):
        resolve_overrides(("add",), {"typo": "native"})
    monkeypatch.setenv("XLLM_KERNEL_OVERRIDE_TYPO", "native")
    with pytest.raises(ValueError, match="unknown kernel override environment"):
        resolve_overrides(("add",), None)


def test_empty_environment_preserves_explicit_mapping(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XLLM_KERNEL_OVERRIDE_ADD", "")
    assert resolve_overrides(("add",), {"add": "custom"}) == {"add": "custom"}
    with pytest.raises(ValueError, match="nonempty"):
        resolve_overrides(("add",), {"add": ""})


def test_exact_name_wins_over_identically_named_solution(registry: KernelRegistry) -> None:
    contract = _contract("add")
    registry.register(_spec(contract, "custom"), lambda value: value + 1)
    registry.register(replace(_spec(contract, "other", 100), solution="add.custom"), lambda value: value + 2)
    registry.freeze()
    selected = select_kernel(contract, device="cpu", execution_mode="eager", override="add.custom")
    assert selected.spec.name == "add.custom"

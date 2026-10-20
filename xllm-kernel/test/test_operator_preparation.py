# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""Preparation selects exactly one provider before device execution."""

import sys
from types import SimpleNamespace

import pytest
import xllm_kernel.operator as preparation
from xllm_kernel.operator import Backend, Operator
from xllm_kernel.registry import OpContract


def test_selection_only_loads_the_requested_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []
    function = lambda value: value

    def imported(name: str) -> SimpleNamespace:
        calls.append(name)
        return SimpleNamespace(load=lambda: function)

    monkeypatch.setattr(preparation, "import_module", imported)
    operation = Operator(
        OpContract("example", 1, (), False, "exact"),
        (
            Backend("first", "first:load", "domain"),
            Backend("second", "second:load", "domain"),
        ),
    )
    plan = operation.prepare("npu", "aclgraph", "npu.second.example")
    assert calls == ["second"]
    assert plan.function is function
    assert plan.spec.contract == operation.contract
    assert plan.spec.solution == "second"


def test_unsupported_selection_fails_before_loading(monkeypatch: pytest.MonkeyPatch) -> None:
    def unexpected(name: str) -> None:
        raise AssertionError(f"must not import {name}")

    monkeypatch.setattr(preparation, "import_module", unexpected)
    operation = Operator(
        OpContract("example", 1, (), False, "exact"), (Backend("only", "only:load", "domain", ("eager",)),)
    )
    for device, mode, implementation in (
        ("cuda", "eager", "npu.only.example"),
        ("npu", "aclgraph", "npu.only.example"),
        ("npu", "eager", "npu.missing.example"),
    ):
        with pytest.raises(ValueError):
            operation.prepare(device, mode, implementation)


def test_variants_share_a_solution_without_loading_each_other(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    def load_xllm() -> object:
        raise AssertionError("the unselected binding must not load")

    def load_torch_npu() -> object:
        calls.append("torch_npu")
        return lambda value: value

    monkeypatch.setattr(
        preparation,
        "import_module",
        lambda name: SimpleNamespace(load_xllm=load_xllm, load_torch_npu=load_torch_npu),
    )
    operation = Operator(
        OpContract("example", 1, (), False, "exact"),
        (
            Backend("aclnn", "example:load_xllm", "domain", variant="xllm", binding="torch.ops.xllm_ops"),
            Backend("aclnn", "example:load_torch_npu", "domain", variant="torch_npu", binding="torch_npu"),
        ),
    )
    plan = operation.prepare("npu", "eager", "npu.aclnn.example.torch_npu")
    assert calls == ["torch_npu"]
    assert plan.function(42) == 42
    assert (plan.spec.solution, plan.spec.variant, plan.spec.binding) == ("aclnn", "torch_npu", "torch_npu")


def test_duplicate_variant_names_fail_before_loading(monkeypatch: pytest.MonkeyPatch) -> None:
    def unexpected(name: str) -> None:
        raise AssertionError(f"must not import {name}")

    monkeypatch.setattr(preparation, "import_module", unexpected)
    operation = Operator(
        OpContract("example", 1, (), False, "exact"),
        (Backend("aclnn", "first:load", "domain"), Backend("aclnn", "second:load", "domain")),
    )
    with pytest.raises(ValueError, match="duplicate implementation names"):
        operation.prepare("npu", "eager", "npu.aclnn.example")


@pytest.mark.parametrize("device_name", ("Ascend910_9391", "Ascend950PR_9599"))
def test_atb_cache_loader_rejects_the_hosts_aclnn_device_path(
    monkeypatch: pytest.MonkeyPatch, device_name: str
) -> None:
    from xllm_kernel.backends import xllm
    from xllm_kernel.ops.reshape_paged_cache import atb

    loaded = []

    def load_operator(name: str) -> object:
        loaded.append(name)
        return lambda: None

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(npu=SimpleNamespace(get_device_name=lambda: device_name)))
    monkeypatch.setitem(sys.modules, "torch_npu", SimpleNamespace())
    monkeypatch.setattr(xllm, "load_operator", load_operator)
    if device_name.startswith("Ascend950"):
        with pytest.raises(ValueError, match="Ascend 950 path uses ACLNN"):
            atb.load()
        assert not loaded
    else:
        assert callable(atb.load())
        assert loaded == ["reshape_paged_cache"]

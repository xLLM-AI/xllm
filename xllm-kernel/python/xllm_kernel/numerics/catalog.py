# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""Explicit discovery of operation-owned suites; loading must stay SDK-free."""

from importlib import import_module

from xllm_kernel.numerics.suite import OperationSuite

# Only this catalog needs a new entry when adding an operation's tool support.
_BUILTINS = ("xllm_kernel.numerics.rms_norm",)


def load_suites() -> tuple[OperationSuite, ...]:
    suites = tuple(import_module(module).SUITE for module in _BUILTINS)
    operations = [suite.operation for suite in suites]
    implementations = [name for suite in suites for name in suite.implementations]
    if len(set(operations)) != len(operations) or len(set(implementations)) != len(implementations):
        raise ValueError("operation suites must have unique operation and implementation names")
    return suites

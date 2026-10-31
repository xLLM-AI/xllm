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

"""Load all operation candidates and publish immutable startup bindings."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from importlib import import_module
from importlib.util import find_spec
from threading import RLock
from types import MappingProxyType

from xllm_kernel.catalog import load_operations
from xllm_kernel.registry import KernelRegistry, PreparedKernel
from xllm_kernel.selection import resolve_overrides, select_kernel

_configuration: tuple[str, str, tuple[str, ...], tuple[tuple[str, str], ...]] | None = None
_prepared: Mapping[str, PreparedKernel] | None = None
_lock = RLock()


def initialize(
    *,
    device: str,
    execution_mode: str = "aclgraph",
    overrides: Mapping[str, str] | None = None,
    operations: tuple[str, ...] | None = None,
) -> Mapping[str, PreparedKernel]:
    """Prepare the requested operation set once and return bindings keyed by name.

    Available backend modules register their implementations on import. Missing
    optional dependencies exclude a module; errors in an installed backend
    propagate. All modules load before the shared registry is frozen. Defaults
    use compatibility and priority; overrides name a solution or implementation.
    Environment overrides take precedence over the mapping. Repeating the same
    effective configuration returns the same read-only mapping. A policy change
    requires a new worker, and forward execution never reselects or falls back.
    By default all catalog operations are required. Models and standalone tools
    can declare a subset so unused collectives do not require a process group.
    """
    global _prepared, _configuration
    with _lock:
        catalog = load_operations()
        names = tuple(op.contract.operation for op in catalog)
        choices = resolve_overrides(names, overrides)
        requested = names if operations is None else operations
        if not requested or len(set(requested)) != len(requested) or set(requested) - set(names):
            raise ValueError(f"invalid kernel operation set: {requested!r}")
        if set(overrides or {}) - set(requested):
            raise ValueError("kernel overrides must target requested operations")
        selected = tuple(op for op in catalog if op.contract.operation in requested)
        choices = {name: value for name, value in choices.items() if name in requested}
        configuration = (device, execution_mode, tuple(sorted(requested)), tuple(sorted(choices.items())))
        if _prepared is not None:
            if configuration != _configuration:
                raise RuntimeError("xllm_kernel is already initialized; restart the worker to change kernel policy")
            return _prepared
        if any(not any(b.device == device for b in op.backends) for op in selected):
            raise ValueError(f"kernel catalog does not provide all required operation adapters for {device!r}")

        unavailable: dict[str, tuple[tuple[str, str], ...]] = {}
        for operation in selected:
            excluded = []
            for backend in operation.backends:
                if backend.device != device:
                    continue
                # Parent packages precede submodules; do not probe a child of
                # an absent optional package (find_spec would raise).
                missing = []
                for dependency in backend.requires:
                    if find_spec(dependency) is None:
                        missing.append(dependency)
                        break
                if missing:
                    excluded.append((backend.module, f"missing dependencies: {', '.join(missing)}"))
                    continue
                import_module(backend.module)
            unavailable[operation.contract.operation] = tuple(excluded)

        KernelRegistry.get().freeze()
        bindings = {}
        for operation in selected:
            name = operation.contract.operation
            try:
                plan = select_kernel(
                    operation.contract,
                    device=device,
                    execution_mode=execution_mode,
                    override=choices.get(name),
                )
            except ValueError as error:
                details = "; ".join(f"{module}: {reason}" for module, reason in unavailable[name])
                raise ValueError(f"{error}; unavailable backends: {details or 'none'}") from error
            bindings[name] = replace(plan, excluded=plan.excluded + unavailable[name])
        # Publish only after every operation has a valid binding.
        for operation in selected:
            operation.prepared = bindings[operation.contract.operation]
        _configuration = configuration
        _prepared = MappingProxyType(bindings)
        return _prepared


def prepare_operation(operation: str, device: str, mode: str, implementation: str) -> PreparedKernel:
    """Use the worker startup path for a single offline validation target."""
    # The model host loads its extension before initialize(). Standalone tools
    # load it only when explicitly requesting one of its existing schemas.
    host_required = any(
        implementation in backend.implementations and "xllm_export" in backend.requires
        for candidate in load_operations()
        if candidate.contract.operation == operation
        for backend in candidate.backends
    )
    if host_required and find_spec("xllm_export") is None:
        from xllm import xllm_export  # noqa: F401

    bindings = initialize(
        device=device,
        execution_mode=mode,
        operations=(operation,),
        overrides={operation: implementation},
    )
    plan = bindings[operation]
    if plan.spec.name != implementation:
        raise ValueError("environment override changed the requested validation implementation")
    return plan

from __future__ import annotations

import importlib
import pkgutil
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

from ...common.spec import (
    DispatchField,
    KernelCompileSpec,
    KernelSpec,
    TilelangKernel,
    is_registered_kernel_class,
)
from ...common.toolchain import prepare_tilelang_import


@dataclass(frozen=True)
class RegisteredKernelFamily:
    module: ModuleType
    kernel_cls: type[TilelangKernel]
    module_name: str
    kernel_name: str
    dispatch_schema: list[DispatchField]
    spec_pairs: list[tuple[KernelCompileSpec, KernelSpec]]


def _load_kernel_module(module_name: str) -> ModuleType:
    prepare_tilelang_import()
    return importlib.import_module(f"{__package__}.kernels.{module_name}")


def _kernels_dir() -> Path:
    return Path(__file__).resolve().parent / "kernels"


def _iter_kernel_module_names() -> list[str]:
    return sorted(
        module.name for module in pkgutil.iter_modules([str(_kernels_dir())]) if not module.name.startswith("_")
    )


def _resolve_registered_kernel_class(
    module_name: str,
) -> tuple[ModuleType, type[TilelangKernel] | None]:
    module = _load_kernel_module(module_name)
    kernel_classes = [
        obj
        for obj in vars(module).values()
        if isinstance(obj, type) and obj.__module__ == module.__name__ and is_registered_kernel_class(obj)
    ]
    if not kernel_classes:
        return module, None
    if len(kernel_classes) > 1:
        kernel_names = ", ".join(sorted(cls.__name__ for cls in kernel_classes))
        raise TypeError(
            f"TileLang kernel module {module_name!r} must define at most one "
            f"@register_kernel class, found: {kernel_names}"
        )

    kernel_cls = kernel_classes[0]
    if not issubclass(kernel_cls, TilelangKernel):
        raise TypeError(f"registered kernel class '{kernel_cls.__name__}' must inherit TilelangKernel")
    return module, kernel_cls


def _load_registered_kernel_family(
    module_name: str,
) -> RegisteredKernelFamily | None:
    module, kernel_cls = _resolve_registered_kernel_class(module_name)
    if kernel_cls is None:
        return None

    generate_source = kernel_cls.__dict__.get("generate_source")
    if not isinstance(generate_source, (staticmethod, classmethod)):
        raise TypeError(
            f"registered kernel class '{kernel_cls.__name__}' must define "
            "callable generate_source(...) as @staticmethod or @classmethod"
        )

    resolved_generate_source = getattr(kernel_cls, "generate_source", None)
    if not callable(resolved_generate_source):
        raise TypeError(f"registered kernel class '{kernel_cls.__name__}' must define callable generate_source(...)")

    resolved_specs = getattr(kernel_cls, "specs", None)
    if not callable(resolved_specs):
        raise TypeError(
            f"registered kernel class '{kernel_cls.__name__}' must define callable specs() -> list[KernelSpec]"
        )
    resolved_dispatch_schema = getattr(kernel_cls, "dispatch_schema", None)
    if not callable(resolved_dispatch_schema):
        raise TypeError(
            f"registered kernel class '{kernel_cls.__name__}' must define "
            "callable dispatch_schema() -> list[DispatchField]"
        )

    try:
        kernel_specs = resolved_specs()
    except NotImplementedError as exc:
        raise TypeError(str(exc)) from exc
    if not isinstance(kernel_specs, list) or not kernel_specs:
        raise TypeError(
            f"registered kernel class '{kernel_cls.__name__}' must return a non-empty list[KernelSpec] from specs()"
        )
    try:
        dispatch_schema = resolved_dispatch_schema()
    except NotImplementedError as exc:
        raise TypeError(str(exc)) from exc
    if not isinstance(dispatch_schema, list) or not dispatch_schema:
        raise TypeError(
            f"registered kernel class '{kernel_cls.__name__}' must return a "
            "non-empty list[DispatchField] from dispatch_schema()"
        )
    for index, field in enumerate(dispatch_schema):
        if not isinstance(field, DispatchField):
            raise TypeError(
                f"registered kernel class '{kernel_cls.__name__}' dispatch_schema()[{index}] must be DispatchField"
            )

    family_kernel_name: str | None = None
    seen_variant_keys: set[str] = set()
    spec_pairs: list[tuple[KernelCompileSpec, KernelSpec]] = []

    for index, kernel_spec in enumerate(kernel_specs):
        if not isinstance(kernel_spec, KernelSpec):
            raise TypeError(f"registered kernel class '{kernel_cls.__name__}' specs()[{index}] must be KernelSpec")

        kernel_spec.validate()
        missing_dispatch_fields = [
            field.name for field in dispatch_schema if field.name not in kernel_spec.specialization
        ]
        if missing_dispatch_fields:
            raise ValueError(
                f"registered kernel class '{kernel_cls.__name__}' specs()[{index}] "
                "is missing DISPATCH_SCHEMA fields: "
                f"{', '.join(missing_dispatch_fields)}"
            )
        compile_spec = kernel_spec.to_compile_spec(
            module_name=module_name,
            dispatch_schema=dispatch_schema,
        )
        if family_kernel_name is None:
            family_kernel_name = compile_spec.kernel_name
        elif compile_spec.kernel_name != family_kernel_name:
            raise ValueError(
                f"registered kernel class '{kernel_cls.__name__}' must return "
                "KernelSpec entries with the same kernel_name"
            )

        if compile_spec.variant_key in seen_variant_keys:
            raise ValueError(
                f"registered kernel class '{kernel_cls.__name__}' has duplicate "
                f"variant_key {compile_spec.variant_key!r}"
            )
        seen_variant_keys.add(compile_spec.variant_key)
        spec_pairs.append((compile_spec, kernel_spec))

    assert family_kernel_name is not None
    return RegisteredKernelFamily(
        module=module,
        kernel_cls=kernel_cls,
        module_name=module_name,
        kernel_name=family_kernel_name,
        dispatch_schema=dispatch_schema,
        spec_pairs=spec_pairs,
    )


def registered_families() -> dict[str, RegisteredKernelFamily]:
    families: dict[str, RegisteredKernelFamily] = {}
    for module_name in _iter_kernel_module_names():
        family = _load_registered_kernel_family(module_name)
        if family is None:
            continue
        if family.kernel_name in families:
            raise ValueError(f"Duplicate Ascend TileLang kernel_name registered: {family.kernel_name}")
        families[family.kernel_name] = family
    return families


def get_default_families(
    kernel_names: list[str] | None = None,
) -> list[RegisteredKernelFamily]:
    families = registered_families()
    if kernel_names is None:
        return list(families.values())
    missing = [name for name in kernel_names if name not in families]
    if missing:
        raise ValueError(f"Unknown Ascend TileLang kernels: {', '.join(missing)}")
    return [families[name] for name in kernel_names]

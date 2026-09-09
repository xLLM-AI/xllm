from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

_REGISTER_KERNEL_ATTR = "__xllm_tilelang_registered_kernel__"
_ENTRY_SYMBOL_CONTEXT_KEY = "entry_symbol"
_C_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_SAFE_VARIANT_KEY_PATTERN = re.compile(r"^[A-Za-z0-9_]+$")
_SUPPORTED_DISPATCH_FIELD_KINDS = frozenset({"int32", "dtype"})
_SPECIALIZATION_CONFIG_KEYS = frozenset(
    {
        "variant_key",
        "specialization",
        "compile_definitions",
        "kernel_name",
        "target",
        "entry_name",
        "source_entry_symbol",
    }
)


@dataclass(frozen=True)
class DispatchField:
    name: str
    kind: str

    def validate(self) -> None:
        if not _C_IDENTIFIER_PATTERN.match(self.name):
            raise ValueError("DispatchField.name must be a valid C/C++ identifier")
        if self.kind not in _SUPPORTED_DISPATCH_FIELD_KINDS:
            supported = ", ".join(sorted(_SUPPORTED_DISPATCH_FIELD_KINDS))
            raise ValueError(f"DispatchField.kind must be one of: {supported}")


@dataclass(frozen=True)
class KernelCompileSpec:
    target: str
    kernel_name: str
    module_name: str
    variant_key: str
    specialization: dict[str, Any] = field(default_factory=dict)
    dispatch_values: dict[str, Any] = field(default_factory=dict)
    entry_name: str | None = None
    source_entry_symbol: str = "call"

    def cache_key_material(self) -> dict[str, Any]:
        return {
            "target": self.target,
            "kernel_name": self.kernel_name,
            "module_name": self.module_name,
            "variant_key": self.variant_key,
            "specialization": self.specialization,
            "dispatch_values": self.dispatch_values,
            "entry_name": self.entry_name,
            "source_entry_symbol": self.source_entry_symbol,
        }


@dataclass(frozen=True)
class KernelSpec:
    variant_key: str
    specialization: dict[str, Any] = field(default_factory=dict)
    compile_definitions: dict[str, str] = field(default_factory=dict)
    kernel_name: str | None = None
    target: str = "ascend"
    entry_name: str | None = None
    source_entry_symbol: str = "call"

    def validate(self) -> None:
        if not self.variant_key:
            raise ValueError("KernelSpec.variant_key must not be empty")
        if not _SAFE_VARIANT_KEY_PATTERN.match(self.variant_key):
            raise ValueError("KernelSpec.variant_key must contain only letters, digits, or underscore")

        if not self.specialization:
            raise ValueError("KernelSpec.specialization must not be empty")

        if self.kernel_name is not None and not _C_IDENTIFIER_PATTERN.match(self.kernel_name):
            raise ValueError("KernelSpec.kernel_name must be a valid C/C++ identifier")

        if self.entry_name is not None and not _C_IDENTIFIER_PATTERN.match(self.entry_name):
            raise ValueError("KernelSpec.entry_name must be a valid C/C++ identifier")

        context_keys = set(self.specialization)
        context_keys.add(_ENTRY_SYMBOL_CONTEXT_KEY)
        uses_entry_symbol = False

        for macro_name, context_key in self.compile_definitions.items():
            if context_key not in context_keys:
                raise KeyError(
                    "KernelSpec.compile_definitions references unknown context "
                    f"key {context_key!r} for macro {macro_name!r}"
                )
            uses_entry_symbol = uses_entry_symbol or context_key == _ENTRY_SYMBOL_CONTEXT_KEY

        if uses_entry_symbol and not self.entry_name:
            raise ValueError("KernelSpec.entry_name is required when compile_definitions references 'entry_symbol'")

    def to_compile_spec(self, *, module_name: str, dispatch_schema: list[DispatchField]) -> KernelCompileSpec:
        dispatch_values = {field.name: self.specialization[field.name] for field in dispatch_schema}
        return KernelCompileSpec(
            target=self.target,
            kernel_name=self.kernel_name or module_name,
            module_name=module_name,
            variant_key=self.variant_key,
            specialization=dict(self.specialization),
            dispatch_values=dispatch_values,
            entry_name=self.entry_name,
            source_entry_symbol=self.source_entry_symbol,
        )

    def render_compile_definitions(self, *, entry_symbol: str) -> list[str]:
        self.validate()
        context = dict(self.specialization)
        context[_ENTRY_SYMBOL_CONTEXT_KEY] = entry_symbol
        definitions: list[str] = []

        for macro_name, context_key in self.compile_definitions.items():
            if context_key not in context:
                raise KeyError(
                    f"compile_definitions references unknown context key {context_key!r} for macro {macro_name!r}"
                )
            definitions.append(f"{macro_name}={context[context_key]}")

        return definitions


class TilelangKernel:
    """Marker base class for TileLang kernel generator classes."""

    TARGET = "ascend"
    KERNEL_NAME: str | None = None
    ENTRY_NAME: str | None = None
    COMPILE_DEFINITIONS: dict[str, str] = {}
    SOURCE_ENTRY_SYMBOL = "call"
    DISPATCH_SCHEMA: list[DispatchField] = []
    SPECIALIZATIONS: list[dict[str, Any] | KernelSpec] = []

    @classmethod
    def dispatch_schema(cls) -> list[DispatchField]:
        if not cls.DISPATCH_SCHEMA:
            raise NotImplementedError(f"{cls.__name__} must define non-empty DISPATCH_SCHEMA")

        normalized: list[DispatchField] = []
        seen_names: set[str] = set()
        for index, field in enumerate(cls.DISPATCH_SCHEMA):
            if not isinstance(field, DispatchField):
                raise TypeError(
                    f"{cls.__name__}.DISPATCH_SCHEMA[{index}] must be DispatchField, got {type(field).__name__}"
                )
            field.validate()
            if field.name in seen_names:
                raise ValueError(f"{cls.__name__}.DISPATCH_SCHEMA contains duplicate field {field.name!r}")
            seen_names.add(field.name)
            normalized.append(field)
        return normalized

    @classmethod
    def specs(cls) -> list[KernelSpec]:
        if not cls.SPECIALIZATIONS:
            raise NotImplementedError(f"{cls.__name__} must define non-empty SPECIALIZATIONS or override specs()")
        return [cls._specialization_to_spec(specialization) for specialization in cls.SPECIALIZATIONS]

    @classmethod
    def _specialization_to_spec(cls, specialization: dict[str, Any] | KernelSpec) -> KernelSpec:
        if isinstance(specialization, KernelSpec):
            return specialization

        if not isinstance(specialization, dict):
            raise TypeError(
                f"{cls.__name__}.SPECIALIZATIONS entries must be dict or KernelSpec, "
                f"got {type(specialization).__name__}"
            )

        specialization_data = dict(specialization)
        specialization_fields = specialization_data.pop("specialization", None)
        if specialization_fields is None:
            specialization_fields = {
                key: value for key, value in specialization_data.items() if key not in _SPECIALIZATION_CONFIG_KEYS
            }
            for key in specialization_fields:
                specialization_data.pop(key)
        elif not isinstance(specialization_fields, dict):
            raise TypeError(
                f"{cls.__name__}.SPECIALIZATIONS specialization must be dict, got "
                f"{type(specialization_fields).__name__}"
            )

        compile_definitions = dict(cls.COMPILE_DEFINITIONS)
        compile_definitions.update(dict(specialization_data.pop("compile_definitions", {})))
        variant_key = specialization_data.pop("variant_key", None)
        if not isinstance(variant_key, str) or not variant_key:
            raise ValueError(f"{cls.__name__}.SPECIALIZATIONS entries must define non-empty 'variant_key'")

        spec = KernelSpec(
            variant_key=variant_key,
            specialization=dict(specialization_fields),
            compile_definitions=compile_definitions,
            kernel_name=specialization_data.pop("kernel_name", cls.KERNEL_NAME),
            target=specialization_data.pop("target", cls.TARGET),
            entry_name=specialization_data.pop("entry_name", cls.ENTRY_NAME),
            source_entry_symbol=specialization_data.pop("source_entry_symbol", cls.SOURCE_ENTRY_SYMBOL),
        )
        if specialization_data:
            unknown_keys = ", ".join(sorted(specialization_data))
            raise KeyError(f"{cls.__name__}.SPECIALIZATIONS contains unsupported config keys: {unknown_keys}")
        return spec


def register_kernel(cls: type[TilelangKernel]) -> type[TilelangKernel]:
    setattr(cls, _REGISTER_KERNEL_ATTR, True)
    return cls


def is_registered_kernel_class(obj: object) -> bool:
    return isinstance(obj, type) and bool(obj.__dict__.get(_REGISTER_KERNEL_ATTR, False))

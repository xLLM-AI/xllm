from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .spec import DispatchField


@dataclass
class KernelAbiParameter:
    cpp_type: str
    name: str


@dataclass
class KernelAbi:
    return_type: str
    parameters: list[KernelAbiParameter] = field(default_factory=list)

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class KernelVariantManifest:
    variant_key: str
    specialization: dict[str, Any]
    generated_source: str
    compiled_binary: str
    entry_symbol: str
    cache_key: str
    dispatch_values: dict[str, Any] = field(default_factory=dict)
    toolchain_options: dict[str, Any] = field(default_factory=dict)
    fingerprint: dict[str, Any] = field(default_factory=dict)
    compile_definitions: list[str] = field(default_factory=list)

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class KernelFamilyManifest:
    target: str
    kernel_name: str
    output_dir: str
    variants_inc: str
    registry_inc: str = ""
    dispatch_schema: list[DispatchField] = field(default_factory=list)
    kernel_abi: KernelAbi | None = None
    variants: list[KernelVariantManifest] = field(default_factory=list)
    schema_version: int = 2

    def to_json_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["dispatch_schema"] = [asdict(field) for field in self.dispatch_schema]
        data["kernel_abi"] = (
            None if self.kernel_abi is None else self.kernel_abi.to_json_dict()
        )
        data["variants"] = [variant.to_json_dict() for variant in self.variants]
        return data

    @property
    def manifest_path(self) -> Path:
        return Path(self.output_dir) / "manifest.json"

    def write(self, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(self.to_json_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def write_if_changed(self, path: str | Path) -> None:
        output = Path(path)
        content = json.dumps(self.to_json_dict(), indent=2, sort_keys=True) + "\n"
        if output.is_file() and output.read_text(encoding="utf-8") == content:
            return
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(content, encoding="utf-8")

    @classmethod
    def read(cls, path: str | Path) -> "KernelFamilyManifest":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        dispatch_schema = [
            DispatchField(**field) for field in data.pop("dispatch_schema", [])
        ]
        kernel_abi_data = data.pop("kernel_abi", None)
        kernel_abi = None
        if kernel_abi_data is not None:
            kernel_abi = KernelAbi(
                return_type=kernel_abi_data["return_type"],
                parameters=[
                    KernelAbiParameter(**param)
                    for param in kernel_abi_data.get("parameters", [])
                ],
            )
        variants = [
            KernelVariantManifest(**variant) for variant in data.pop("variants", [])
        ]
        return cls(
            dispatch_schema=dispatch_schema,
            kernel_abi=kernel_abi,
            variants=variants,
            **data,
        )

    def get_variant(self, variant_key: str) -> KernelVariantManifest | None:
        for variant in self.variants:
            if variant.variant_key == variant_key:
                return variant
        return None

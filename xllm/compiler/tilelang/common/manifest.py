from __future__ import annotations

import json
import os
import stat
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .spec import DispatchField


_DEFAULT_SHARED_FILE_MODE = 0o664


def _write_text_atomically(output: Path, content: str) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        output_mode = stat.S_IMODE(output.stat().st_mode)
    except FileNotFoundError:
        output_mode = _DEFAULT_SHARED_FILE_MODE

    file_descriptor, temporary_path = tempfile.mkstemp(
        dir=output.parent,
        prefix=f".{output.name}.",
    )
    try:
        os.fchmod(file_descriptor, output_mode)
        temporary_file = os.fdopen(file_descriptor, "w", encoding="utf-8")
        file_descriptor = -1
        with temporary_file:
            temporary_file.write(content)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        os.replace(temporary_path, output)
    finally:
        if file_descriptor >= 0:
            os.close(file_descriptor)
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)


def _portable_path(path: str, manifest_dir: Path) -> str:
    artifact_path = Path(path)
    if not artifact_path.is_absolute():
        return artifact_path.as_posix()
    try:
        return artifact_path.relative_to(manifest_dir).as_posix()
    except ValueError:
        return str(artifact_path)


def _resolved_path(path: str, manifest_dir: Path) -> str:
    artifact_path = Path(path)
    if artifact_path.is_absolute():
        return str(artifact_path)
    return str((manifest_dir / artifact_path).resolve())


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
    schema_version: int = 3

    def to_json_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["dispatch_schema"] = [asdict(field) for field in self.dispatch_schema]
        data["kernel_abi"] = (
            None if self.kernel_abi is None else self.kernel_abi.to_json_dict()
        )
        data["variants"] = [variant.to_json_dict() for variant in self.variants]
        return data

    def _to_portable_json_dict(self, manifest_dir: Path) -> dict[str, Any]:
        data = self.to_json_dict()
        data["output_dir"] = _portable_path(data["output_dir"], manifest_dir)
        data["variants_inc"] = _portable_path(data["variants_inc"], manifest_dir)
        if data["registry_inc"]:
            data["registry_inc"] = _portable_path(
                data["registry_inc"],
                manifest_dir,
            )
        for variant in data["variants"]:
            variant["generated_source"] = _portable_path(
                variant["generated_source"],
                manifest_dir,
            )
            variant["compiled_binary"] = _portable_path(
                variant["compiled_binary"],
                manifest_dir,
            )
        return data

    @property
    def manifest_path(self) -> Path:
        return Path(self.output_dir) / "manifest.json"

    def write(self, path: str | Path) -> None:
        output = Path(path)
        _write_text_atomically(
            output,
            json.dumps(
                self._to_portable_json_dict(output.parent.resolve()),
                indent=2,
                sort_keys=True,
            )
            + "\n",
        )

    def write_if_changed(self, path: str | Path) -> None:
        output = Path(path)
        content = (
            json.dumps(
                self._to_portable_json_dict(output.parent.resolve()),
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        if output.is_file() and output.read_text(encoding="utf-8") == content:
            return
        _write_text_atomically(output, content)

    @classmethod
    def read(cls, path: str | Path) -> "KernelFamilyManifest":
        manifest_path = Path(path).resolve()
        manifest_dir = manifest_path.parent
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        data["output_dir"] = _resolved_path(data["output_dir"], manifest_dir)
        data["variants_inc"] = _resolved_path(data["variants_inc"], manifest_dir)
        if data.get("registry_inc"):
            data["registry_inc"] = _resolved_path(
                data["registry_inc"],
                manifest_dir,
            )
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
        variant_data = data.pop("variants", [])
        for variant in variant_data:
            variant["generated_source"] = _resolved_path(
                variant["generated_source"],
                manifest_dir,
            )
            variant["compiled_binary"] = _resolved_path(
                variant["compiled_binary"],
                manifest_dir,
            )
        variants = [KernelVariantManifest(**variant) for variant in variant_data]
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

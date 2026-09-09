from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .manifest import KernelFamilyManifest
from .spec import KernelCompileSpec
from .toolchain import sha256_file


def compute_cache_key(
    spec: KernelCompileSpec,
    fingerprint: dict[str, Any],
    dependency_files: list[str | Path],
) -> str:
    payload = {
        "spec": spec.cache_key_material(),
        "fingerprint": fingerprint,
        "dependencies": {str(Path(path).resolve()): sha256_file(path) for path in dependency_files},
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def is_cache_hit(manifest_path: str | Path, variant_key: str, expected_cache_key: str) -> bool:
    path = Path(manifest_path)
    if not path.is_file():
        return False

    try:
        manifest = KernelFamilyManifest.read(path)
    except Exception:
        return False

    variant = manifest.get_variant(variant_key)
    if variant is None:
        return False

    if variant.cache_key != expected_cache_key:
        return False

    return Path(variant.generated_source).is_file() and Path(variant.compiled_binary).is_file()

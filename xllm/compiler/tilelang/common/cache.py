from __future__ import annotations

import fcntl
import hashlib
import json
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from .spec import KernelCompileSpec
from .toolchain import sha256_file


def _dependency_cache_entry(
    path: str | Path,
    dependency_root: Path,
) -> dict[str, str]:
    resolved_path = Path(path).resolve()
    try:
        cache_path = resolved_path.relative_to(dependency_root).as_posix()
    except ValueError:
        cache_path = f"external/{resolved_path.name}"
    return {
        "path": cache_path,
        "sha256": sha256_file(resolved_path),
    }


def compute_cache_key(
    spec: KernelCompileSpec,
    fingerprint: dict[str, Any],
    dependency_files: list[str | Path],
    dependency_root: str | Path,
) -> str:
    resolved_dependency_root = Path(dependency_root).resolve()
    dependencies = [
        _dependency_cache_entry(path, resolved_dependency_root)
        for path in dependency_files
    ]
    dependencies.sort(key=lambda entry: (entry["path"], entry["sha256"]))
    payload = {
        "spec": spec.cache_key_material(),
        "fingerprint": fingerprint,
        "dependencies": dependencies,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


@contextmanager
def cache_file_lock(lock_path: str | Path) -> Iterator[None]:
    """Serializes writers that publish artifacts into one cache directory."""
    path = Path(lock_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

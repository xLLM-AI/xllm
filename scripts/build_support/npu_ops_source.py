# Copyright 2026 The xLLM Authors.
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

"""Identify and prepare patched operator sources without changing submodules."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

_PREPARED_MARKER = ".xllm_ops_source_identity"
_SOURCE_EXPORT_VERSION = 2
_DEPENDENCY_OUTPUTS = frozenset(("build", "build_out", "output", "dist"))
_COPY_IGNORES = shutil.ignore_patterns(".git", "CMakeFiles", "__pycache__", ".cache", ".pytest_cache")


def _git(source: Path, *args: str) -> bytes:
    return subprocess.check_output(
        ["git", "-c", f"safe.directory={source}", "-C", str(source), *args], stderr=subprocess.PIPE
    )


def _source_revision(source: Path) -> str:
    if not (source / ".git").exists():
        raise RuntimeError(f"Initialize the xllm_ops submodule before building NPU operators: {source}")
    try:
        return _git(source, "rev-parse", "HEAD").decode().strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError(f"Cannot determine the xllm_ops source revision at {source}") from error


def _patches(base_dir: Path) -> list[Path]:
    return sorted((base_dir / "patches" / "xllm_ops").glob("*.patch"))


def xllm_ops_source_identity(base_dir: str | Path) -> str:
    """Retain legacy HEAD markers unless the checkout supplies operator patches."""
    base_dir = Path(base_dir)
    revision = _source_revision(base_dir / "third_party" / "xllm_ops")
    patches = _patches(base_dir)
    if not patches:
        return revision
    patch_ids = [(patch.name, hashlib.sha256(patch.read_bytes()).hexdigest()) for patch in patches]
    digest = hashlib.sha256(json.dumps(patch_ids, separators=(",", ":")).encode()).hexdigest()
    return f"{revision}+patches.{digest}"


def _copy_fetched_dependency(source: Path, destination: Path) -> None:
    def _ignore_artifacts(directory: str, names: list[str]) -> set[str]:
        ignored = _COPY_IGNORES(directory, names)
        if Path(directory) == source:
            # Output names are valid source components too, such as JSON's
            # include/nlohmann/detail/output. Only omit root build products.
            ignored.update(_DEPENDENCY_OUTPUTS.intersection(names))
        return ignored

    # Materialize symlinks: vendor build scripts may patch dependency headers
    # and must never reach the pinned checkout.
    shutil.copytree(source, destination, ignore=_ignore_artifacts)


def _export_source(source: Path, destination: Path, revision: str) -> None:
    """Export pinned files and copy already-fetched build dependencies privately."""
    destination.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryFile() as archive:
        subprocess.run(
            ["git", "-c", f"safe.directory={source}", "-C", str(source), "archive", revision],
            stdout=archive,
            check=True,
        )
        archive.seek(0)
        with tarfile.open(fileobj=archive) as contents:
            # The archive is generated from the pinned repository above, not
            # supplied externally. Keep Git symlinks unchanged on newer Python.
            if hasattr(tarfile, "fully_trusted_filter"):
                contents.extractall(destination, filter="fully_trusted")
            else:
                contents.extractall(destination)
    gitlinks = []
    for entry in _git(source, "ls-tree", "-rz", revision).split(b"\0"):
        if not entry:
            continue
        metadata, relative_path = entry.split(b"\t", 1)
        mode, _, commit = metadata.split()
        if mode == b"160000":
            relative = Path(os.fsdecode(relative_path))
            gitlinks.append(relative)
            _export_source(source / relative, destination / relative, commit.decode())
    dependencies = source / "third_party"
    if dependencies.is_dir():
        for dependency in dependencies.iterdir():
            relative = dependency.relative_to(source)
            if relative in gitlinks or (destination / relative).exists():
                continue
            if dependency.is_dir():
                _copy_fetched_dependency(dependency, destination / relative)


def prepare_xllm_ops_source(base_dir: str | Path, destination: str | Path) -> Path:
    """Reuse a prepared tree while its sources and export format remain unchanged."""
    base_dir = Path(base_dir).resolve()
    source = base_dir / "third_party" / "xllm_ops"
    patches = _patches(base_dir)
    if not patches:
        return source
    identity = xllm_ops_source_identity(base_dir)
    destination = Path(destination).resolve()
    marker = destination / _PREPARED_MARKER
    marker_contents = f"{identity}\nexport-version={_SOURCE_EXPORT_VERSION}\n"
    if marker.is_file() and marker.read_text(encoding="utf-8") == marker_contents:
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".xllm-ops-source-", dir=destination.parent) as temporary:
        prepared = Path(temporary) / "source"
        _export_source(source, prepared, identity.split("+", 1)[0])
        environment = os.environ.copy()
        # The private copy may live below the xLLM checkout's build directory.
        # Apply paths relative to this copy rather than its enclosing repository.
        environment["GIT_CEILING_DIRECTORIES"] = str(prepared.parent)
        for patch in patches:
            subprocess.run(["git", "apply", "--check", str(patch)], cwd=prepared, env=environment, check=True)
            subprocess.run(["git", "apply", str(patch)], cwd=prepared, env=environment, check=True)
        (prepared / _PREPARED_MARKER).write_text(marker_contents, encoding="utf-8")
        if destination.exists():
            shutil.rmtree(destination)
        prepared.rename(destination)
    return destination


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("identity", "prepare"))
    parser.add_argument("--base-dir", type=Path, required=True)
    parser.add_argument("--destination", type=Path)
    args = parser.parse_args()
    if args.command == "identity":
        result = xllm_ops_source_identity(args.base_dir)
    else:
        if args.destination is None:
            parser.error("prepare requires --destination")
        result = str(prepare_xllm_ops_source(args.base_dir, args.destination))
    sys.stdout.write(result + "\n")


if __name__ == "__main__":
    _main()

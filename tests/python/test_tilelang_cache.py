# Copyright 2026 The xLLM Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/jd-opensource/xllm/blob/main/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for portable and concurrent TileLang persistent caches."""

import json
import os
import shutil
import stat
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event

from xllm.compiler.tilelang.common.cache import cache_file_lock, compute_cache_key
from xllm.compiler.tilelang.common.manifest import (
    KernelFamilyManifest,
    KernelVariantManifest,
)
from xllm.compiler.tilelang.common.spec import KernelCompileSpec


_LOCK_WAIT_TIMEOUT_SECONDS = 5.0
_LOCK_CONTENTION_PROBE_SECONDS = 0.1


def _kernel_compile_spec() -> KernelCompileSpec:
    return KernelCompileSpec(
        target="ascend",
        kernel_name="test_kernel",
        module_name="test_module",
        variant_key="default",
    )


def test_cache_key_is_independent_of_checkout_path(tmp_path: Path) -> None:
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    relative_dependency = Path("xllm/compiler/tilelang/kernel.py")
    for checkout_root in (first_root, second_root):
        dependency_path = checkout_root / relative_dependency
        dependency_path.parent.mkdir(parents=True)
        dependency_path.write_text("KERNEL = 1\n", encoding="utf-8")

    first_key = compute_cache_key(
        _kernel_compile_spec(),
        {"device": "a3"},
        [first_root / relative_dependency],
        first_root,
    )
    second_key = compute_cache_key(
        _kernel_compile_spec(),
        {"device": "a3"},
        [second_root / relative_dependency],
        second_root,
    )

    assert first_key == second_key


def test_cache_key_tracks_dependency_content(tmp_path: Path) -> None:
    dependency_path = tmp_path / "kernel.py"
    dependency_path.write_text("KERNEL = 1\n", encoding="utf-8")
    original_key = compute_cache_key(
        _kernel_compile_spec(),
        {"device": "a3"},
        [dependency_path],
        tmp_path,
    )

    dependency_path.write_text("KERNEL = 2\n", encoding="utf-8")
    modified_key = compute_cache_key(
        _kernel_compile_spec(),
        {"device": "a3"},
        [dependency_path],
        tmp_path,
    )

    assert original_key != modified_key


def test_cache_file_lock_serializes_writers(tmp_path: Path) -> None:
    lock_path = tmp_path / "cache" / ".build.lock"
    first_acquired = Event()
    release_first = Event()
    second_started = Event()
    second_acquired = Event()

    def _hold_first_lock() -> None:
        with cache_file_lock(lock_path):
            first_acquired.set()
            assert release_first.wait(timeout=_LOCK_WAIT_TIMEOUT_SECONDS)

    def _acquire_second_lock() -> None:
        second_started.set()
        with cache_file_lock(lock_path):
            second_acquired.set()

    with ThreadPoolExecutor(max_workers=2) as executor:
        first_future = executor.submit(_hold_first_lock)
        assert first_acquired.wait(timeout=_LOCK_WAIT_TIMEOUT_SECONDS)
        second_future = executor.submit(_acquire_second_lock)
        assert second_started.wait(timeout=_LOCK_WAIT_TIMEOUT_SECONDS)
        assert not second_acquired.wait(timeout=_LOCK_CONTENTION_PROBE_SECONDS)
        release_first.set()
        first_future.result(timeout=_LOCK_WAIT_TIMEOUT_SECONDS)
        second_future.result(timeout=_LOCK_WAIT_TIMEOUT_SECONDS)

    assert second_acquired.is_set()


def test_manifest_is_portable_and_preserves_shared_permissions(tmp_path: Path) -> None:
    original_family_dir = tmp_path / "original" / "family"
    variant_dir = original_family_dir / "default"
    variant_dir.mkdir(parents=True)
    generated_source = variant_dir / "kernel.cpp"
    compiled_binary = variant_dir / "kernel.o"
    variants_inc = original_family_dir / "variants.inc"
    registry_inc = original_family_dir / "registry.inc"
    for artifact in (
        generated_source,
        compiled_binary,
        variants_inc,
        registry_inc,
    ):
        artifact.write_text("artifact\n", encoding="utf-8")

    manifest = KernelFamilyManifest(
        target="ascend",
        kernel_name="test_kernel",
        output_dir=str(original_family_dir),
        variants_inc=str(variants_inc),
        registry_inc=str(registry_inc),
        variants=[
            KernelVariantManifest(
                variant_key="default",
                specialization={},
                generated_source=str(generated_source),
                compiled_binary=str(compiled_binary),
                entry_symbol="test_kernel_call",
                cache_key="cache-key",
            )
        ],
    )
    manifest_path = original_family_dir / "manifest.json"
    manifest.write(manifest_path)

    serialized = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert serialized["schema_version"] == 3
    assert serialized["output_dir"] == "."
    assert serialized["variants_inc"] == "variants.inc"
    assert serialized["variants"][0]["generated_source"] == "default/kernel.cpp"
    assert stat.S_IMODE(manifest_path.stat().st_mode) == 0o664

    os.chmod(manifest_path, 0o640)
    manifest.write(manifest_path)
    assert stat.S_IMODE(manifest_path.stat().st_mode) == 0o640

    relocated_family_dir = tmp_path / "relocated" / "family"
    shutil.copytree(original_family_dir, relocated_family_dir)
    relocated_manifest = KernelFamilyManifest.read(
        relocated_family_dir / "manifest.json"
    )
    relocated_variant = relocated_manifest.get_variant("default")
    assert relocated_variant is not None
    assert Path(relocated_variant.generated_source).is_file()
    assert Path(relocated_variant.compiled_binary).is_file()
    assert Path(relocated_variant.generated_source).is_relative_to(
        relocated_family_dir
    )


def test_manifest_reads_legacy_absolute_paths(tmp_path: Path) -> None:
    family_dir = tmp_path / "legacy" / "family"
    variant_dir = family_dir / "default"
    variant_dir.mkdir(parents=True)
    generated_source = variant_dir / "kernel.cpp"
    compiled_binary = variant_dir / "kernel.o"
    variants_inc = family_dir / "variants.inc"
    for artifact in (generated_source, compiled_binary, variants_inc):
        artifact.write_text("artifact\n", encoding="utf-8")

    manifest_path = family_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "target": "ascend",
                "kernel_name": "test_kernel",
                "output_dir": str(family_dir),
                "variants_inc": str(variants_inc),
                "registry_inc": "",
                "dispatch_schema": [],
                "kernel_abi": None,
                "variants": [
                    {
                        "variant_key": "default",
                        "specialization": {},
                        "generated_source": str(generated_source),
                        "compiled_binary": str(compiled_binary),
                        "entry_symbol": "test_kernel_call",
                        "cache_key": "cache-key",
                    }
                ],
                "schema_version": 2,
            }
        ),
        encoding="utf-8",
    )

    manifest = KernelFamilyManifest.read(manifest_path)

    assert manifest.schema_version == 2
    assert manifest.output_dir == str(family_dir)
    assert manifest.variants_inc == str(variants_inc)
    assert manifest.variants[0].generated_source == str(generated_source)
    assert manifest.variants[0].compiled_binary == str(compiled_binary)

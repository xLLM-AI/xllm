# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""Frozen MiniMax-H3 Ref2VA rf_v2 capacity and reorder-record semantics."""

from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from xllm_kernel.ops.attention.rainfusion import plan
from xllm_kernel.ops.attention.rainfusion import tilelang as rainfusion_tilelang
from xllm_kernel.ops.attention.rainfusion.backend import (
    SPARSE_MIDDLE_STEPS,
    MinimaxH3RainfusionBackend,
)
from xllm_kernel.ops.attention.rainfusion.records import build_reorder_records

_PRODUCTION = plan.MinimaxH3RainfusionGeometry(
    total_rows=plan.PRODUCTION_TOTAL_ROWS,
    valid_rows=plan.PRODUCTION_VALID_ROWS,
    video_start=plan.PRODUCTION_VIDEO_START,
    grid_t=plan.PRODUCTION_GRID_T,
    grid_h=plan.PRODUCTION_GRID_H,
    grid_w=plan.PRODUCTION_GRID_W,
    topk_r=plan.PRODUCTION_TOPK_R,
)


def test_production_geometry_matches_the_frozen_native_constants() -> None:
    derived = plan.derive(_PRODUCTION)
    assert derived.total_blocks == plan.PRODUCTION_NUM_BLOCKS
    assert derived.video_end == plan.PRODUCTION_VALID_ROWS
    assert derived.ff_blocks == plan.PRODUCTION_DENSE_PREFIX_BLOCKS
    assert derived.video_full_blocks == plan.PRODUCTION_SPARSE_END_BLOCK
    assert derived.plan_width == 472
    assert plan.record_count(_PRODUCTION) == plan.PRODUCTION_RECORD_COUNT
    assert plan.records_padded(_PRODUCTION) == plan.PRODUCTION_RECORDS_PADDED
    assert plan.logical_fits_capacity(_PRODUCTION, derived)
    assert plan.is_qualified_profile(_PRODUCTION)
    assert plan.unavailability_reason(_PRODUCTION) == ""


def test_capacity_pair_is_the_proven_a3_specialization() -> None:
    capacity = plan.QUALIFIED_CAPACITY
    assert (
        capacity.total_rows,
        capacity.num_blocks,
        capacity.record_count,
        capacity.records_padded,
        capacity.topk_r,
        capacity.plan_width,
        capacity.q_pool_pad,
        capacity.k_pool_pad,
    ) == (61952, 483, 5568, 5568, 97, 488, 480, 496)
    assert plan.SELECTOR_VARIANT_KEY == "cap_ctr61952_ctb483_crc5568_ctk97_abi2"
    assert plan.ATTENTION_VARIANT_KEY == "cap_ctr61952_cnb483_crc5568_crp5568_bf16_abi2"


def test_records_are_a_dense_partition_of_the_valid_video_and_context_rows() -> None:
    records = plan.semantic_records(_PRODUCTION)
    assert len(records) == plan.PRODUCTION_RECORD_COUNT
    destinations = sorted((dst, dst + rows) for _src, dst, rows, _flags in records)
    assert destinations[0][0] == 0
    assert destinations[-1][1] == _PRODUCTION.valid_rows
    for (start, end), (next_start, _) in zip(destinations, destinations[1:]):
        assert end == next_start
    sources = sorted((src, src + rows) for src, _dst, rows, _flags in records)
    assert sources[0][0] == 0
    assert sources[-1][1] == _PRODUCTION.valid_rows


def test_reorder_table_is_worker_major_with_zero_unused_slots() -> None:
    table = build_reorder_records(_PRODUCTION)
    records = plan.semantic_records(_PRODUCTION)
    per_worker = plan.ceildiv(len(records), plan.WORKERS)
    assert tuple(table.shape) == (plan.WORKERS * per_worker, 4)
    used = set()
    for index, expected in enumerate(records):
        slot = (index % plan.WORKERS) * per_worker + index // plan.WORKERS
        used.add(slot)
        assert tuple(int(value) for value in table[slot]) == expected
    assert len(used) == len(records)
    for slot in set(range(table.shape[0])) - used:
        assert not table[slot].any()


def test_oversized_geometry_and_wrong_topk_are_rejected() -> None:
    oversized = plan.MinimaxH3RainfusionGeometry(
        total_rows=plan.QUALIFIED_CAPACITY.total_rows + plan.KV_PACK_ROWS,
        valid_rows=_PRODUCTION.valid_rows,
        video_start=_PRODUCTION.video_start,
        grid_t=_PRODUCTION.grid_t,
        grid_h=_PRODUCTION.grid_h,
        grid_w=_PRODUCTION.grid_w,
        topk_r=_PRODUCTION.topk_r,
    )
    assert not plan.logical_fits_capacity(oversized, plan.derive(oversized))
    wrong_topk = plan.MinimaxH3RainfusionGeometry(
        total_rows=_PRODUCTION.total_rows,
        valid_rows=_PRODUCTION.valid_rows,
        video_start=_PRODUCTION.video_start,
        grid_t=_PRODUCTION.grid_t,
        grid_h=_PRODUCTION.grid_h,
        grid_w=_PRODUCTION.grid_w,
        topk_r=_PRODUCTION.topk_r + 1,
    )
    assert plan.unavailability_reason(wrong_topk) == "unqualified_geometry"
    with pytest.raises(ValueError):
        plan.derive(replace(_PRODUCTION, topk_r=plan.PRODUCTION_NUM_BLOCKS + 1))


def test_backend_admission_is_a_python_plan_check_without_a_device() -> None:
    backend = MinimaxH3RainfusionBackend()
    assert backend.available(
        _PRODUCTION.valid_rows,
        _PRODUCTION.video_start,
        _PRODUCTION.grid_t,
        _PRODUCTION.grid_h,
        _PRODUCTION.grid_w,
    )
    assert (
        backend.unavailability_reason(
            _PRODUCTION.valid_rows,
            _PRODUCTION.video_start,
            _PRODUCTION.grid_t,
            _PRODUCTION.grid_h,
            _PRODUCTION.grid_w,
        )
        == ""
    )

    disabled = MinimaxH3RainfusionBackend(enabled=False)
    assert not disabled.available(
        _PRODUCTION.valid_rows,
        _PRODUCTION.video_start,
        _PRODUCTION.grid_t,
        _PRODUCTION.grid_h,
        _PRODUCTION.grid_w,
    )
    assert (
        disabled.unavailability_reason(
            _PRODUCTION.valid_rows,
            _PRODUCTION.video_start,
            _PRODUCTION.grid_t,
            _PRODUCTION.grid_h,
            _PRODUCTION.grid_w,
        )
        == "disabled"
    )


def test_unsupported_geometry_returns_dense_before_any_device_work() -> None:
    backend = MinimaxH3RainfusionBackend()
    invalid = (0, _PRODUCTION.video_start, _PRODUCTION.grid_t, _PRODUCTION.grid_h, _PRODUCTION.grid_w)
    assert backend.unavailability_reason(*invalid) == "unqualified_geometry"
    assert backend.prepare(*invalid, SPARSE_MIDDLE_STEPS, torch.device("cpu")) is None


def test_runtime_prepare_rejects_non_npu_before_compiling() -> None:
    from xllm_kernel.ops.attention.rainfusion.tilelang import prepare_kernels

    with pytest.raises(ValueError):
        prepare_kernels("cpu", 0)


def test_compile_child_has_no_npu_or_cann_tvm_inheritance(monkeypatch) -> None:
    monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", "3")
    monkeypatch.setenv("TORCH_DEVICE_BACKEND_AUTOLOAD", "1")
    monkeypatch.setenv("TVM_IMPORT_PYTHON_PATH", "/parent/tvm/python")
    monkeypatch.setenv("TVM_LIBRARY_PATH", "/parent/tvm/lib")
    monkeypatch.setenv("ASCEND_HOME_PATH", "/cann/latest")
    monkeypatch.setenv("LD_LIBRARY_PATH", "/cann/lib64")
    monkeypatch.setenv("PYTHONPATH", "/existing/pythonpath")

    env = rainfusion_tilelang._child_env()
    assert env["ASCEND_RT_VISIBLE_DEVICES"] == ""
    assert env["TORCH_DEVICE_BACKEND_AUTOLOAD"] == "0"
    assert "TVM_IMPORT_PYTHON_PATH" not in env
    assert "TVM_LIBRARY_PATH" not in env
    assert env["ASCEND_HOME_PATH"] == "/cann/latest"
    assert env["LD_LIBRARY_PATH"] == "/cann/lib64"
    assert env["PYTHONPATH"].split(os.pathsep) == [str(rainfusion_tilelang._package_root()), "/existing/pythonpath"]
    assert os.environ["ASCEND_RT_VISIBLE_DEVICES"] == "3"
    assert os.environ["TVM_LIBRARY_PATH"] == "/parent/tvm/lib"

    identity_env = rainfusion_tilelang._toolchain_identity()["env"]
    assert identity_env["TVM_IMPORT_PYTHON_PATH"] is None
    assert identity_env["TVM_LIBRARY_PATH"] is None
    assert identity_env["ASCEND_RT_VISIBLE_DEVICES"] == ""
    assert identity_env["TORCH_DEVICE_BACKEND_AUTOLOAD"] == "0"


def test_prepare_for_device_validates_before_npu_normalization(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, int | None]] = []

    def fake_prepare(device_type: str, device_index: int | None) -> None:
        calls.append((device_type, device_index))

    monkeypatch.setattr(rainfusion_tilelang, "prepare_kernels", fake_prepare)
    rainfusion_tilelang.prepare_for_device(torch.device("privateuseone", 2))
    expected_calls = [("npu", 2)]
    if torch._C._get_privateuse1_backend_name() == "npu":
        rainfusion_tilelang.prepare_for_device(torch.device("npu", 1))
        expected_calls.append(("npu", 1))
    with pytest.raises(ValueError):
        rainfusion_tilelang.prepare_for_device(torch.device("cpu"))
    assert calls == expected_calls


def test_isolated_compile_identity_binds_sources_target_and_toolchain() -> None:
    identity = rainfusion_tilelang._cache_identity()
    assert identity["recipe"] == rainfusion_tilelang._RECIPE_VERSION
    assert identity["target"] == "ascendc"
    assert identity["platform"] == "A3"
    assert identity["tilelang"]
    assert set(identity["sources"]) == {"compile", "plan", "selector_dsl", "attention_dsl"}
    assert all(len(digest) == 64 for digest in identity["sources"].values())
    assert rainfusion_tilelang._cache_key(identity) == rainfusion_tilelang._cache_key(identity)
    assert rainfusion_tilelang._cache_key({**identity, "platform": "A5"}) != rainfusion_tilelang._cache_key(identity)


def test_cached_pair_validation_rejects_stale_or_corrupt_entries(tmp_path) -> None:
    identity = {"recipe": "test"}
    key = rainfusion_tilelang._cache_key(identity)
    (tmp_path / rainfusion_tilelang._SELECTOR_LIBRARY).write_bytes(b"selector")
    (tmp_path / rainfusion_tilelang._ATTENTION_LIBRARY).write_bytes(b"attention")
    rainfusion_tilelang._write_manifest(tmp_path, key, identity)

    assert rainfusion_tilelang._validated_pair(tmp_path, key, identity) == (
        tmp_path / rainfusion_tilelang._SELECTOR_LIBRARY,
        tmp_path / rainfusion_tilelang._ATTENTION_LIBRARY,
    )
    assert rainfusion_tilelang._validated_pair(tmp_path, "0" * 64, identity) is None

    (tmp_path / rainfusion_tilelang._SELECTOR_LIBRARY).write_bytes(b"changed")
    assert rainfusion_tilelang._validated_pair(tmp_path, key, identity) is None


def test_isolated_runtime_uses_the_verified_abi_shape() -> None:
    assert len(rainfusion_tilelang._SELECTOR_POINTER_ARGS) == 12
    assert len(rainfusion_tilelang._ATTENTION_POINTER_ARGS) == 13
    assert len(rainfusion_tilelang._SELECTOR_SCALAR_ARGS) == 8
    assert len(rainfusion_tilelang._ATTENTION_SCALAR_ARGS) == 8
    assert rainfusion_tilelang._KERNEL_SYMBOL == "call"


@pytest.mark.parametrize("relative", [False, True])
def test_cache_publish_uses_one_complete_pair(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, relative: bool) -> None:
    identity = {"recipe": "test"}
    key = rainfusion_tilelang._cache_key(identity)
    monkeypatch.setattr(rainfusion_tilelang, "_cache_identity", lambda: identity)
    monkeypatch.chdir(tmp_path)
    cache_root = tmp_path / "tilelang_cache"
    monkeypatch.setenv("TILELANG_CACHE_DIR", cache_root.name if relative else str(cache_root))
    assert rainfusion_tilelang._cache_root() == cache_root / rainfusion_tilelang._CACHE_SUBDIR

    def compile_pair(destination: Path) -> None:
        assert destination.is_absolute()
        (destination / rainfusion_tilelang._SELECTOR_LIBRARY).write_bytes(b"selector")
        (destination / rainfusion_tilelang._ATTENTION_LIBRARY).write_bytes(b"attention")

    def fail_compile(destination: Path) -> None:
        pytest.fail(f"unexpected compile into {destination}")

    monkeypatch.setattr(rainfusion_tilelang, "_run_compiler", compile_pair)
    pair = rainfusion_tilelang._ensure_compiled_pair()
    cache_dir = cache_root / rainfusion_tilelang._CACHE_SUBDIR / key
    assert pair == (
        cache_dir / rainfusion_tilelang._SELECTOR_LIBRARY,
        cache_dir / rainfusion_tilelang._ATTENTION_LIBRARY,
    )
    assert (cache_dir / rainfusion_tilelang._MANIFEST_FILE).is_file()

    monkeypatch.setattr(rainfusion_tilelang, "_run_compiler", fail_compile)
    assert rainfusion_tilelang._ensure_compiled_pair() == pair

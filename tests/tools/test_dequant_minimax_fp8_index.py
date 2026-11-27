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

"""Exercise the index boundary with real, small local safetensors shards."""

import argparse
import json
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from safetensors.torch import load_file, save_file

from tools import dequant_minimax_fp8 as converter

_INDEX = "model.safetensors.index.json"
_WEIGHT = "layer.weight"
_SCALE = "layer.weight_scale_inv"


def _write_checkpoint(root: Path, *, indexed: bool = True, nested: bool = False) -> Path:
    input_dir = root / "input"
    input_dir.mkdir()
    converter.save_json(
        input_dir / "config.json",
        {"quantization_config": {"quant_method": "fp8", "weight_block_size": [2, 2]}},
    )
    weight = torch.tensor([[1, 2, 3], [4, 5, 6]]).to(torch.float8_e4m3fn)
    tensors = {_WEIGHT: weight, _SCALE: torch.tensor([[2.0, 3.0]])}
    bias = torch.tensor([7.0, 8.0, 9.0])
    if indexed:
        first = "part1/model.safetensors" if nested else "model-00001.safetensors"
        second = "part2/model.safetensors" if nested else "model-00002.safetensors"
        (input_dir / first).parent.mkdir(parents=True, exist_ok=True)
        (input_dir / second).parent.mkdir(parents=True, exist_ok=True)
        save_file(tensors, str(input_dir / first), metadata={"source": "test with header metadata"})
        save_file({"bias": bias}, str(input_dir / second))
        converter.save_json(
            input_dir / _INDEX,
            {"metadata": {"total_size": 26}, "weight_map": {_WEIGHT: first, _SCALE: first, "bias": second}},
        )
    else:
        tensors["bias"] = bias
        save_file(tensors, str(input_dir / "model.safetensors"))
    return input_dir


def _run_main(input_dir: Path, output_dir: Path) -> None:
    with patch.object(
        converter, "parse_args", return_value=argparse.Namespace(input_dir=input_dir, output_dir=output_dir)
    ):
        converter.main()


def _assert_preflight_failure(input_dir: Path, output_dir: Path, field: str) -> None:
    with pytest.raises(ValueError, match=field):
        _run_main(input_dir, output_dir)
    assert not output_dir.exists()


@pytest.mark.parametrize("nested", [False, True])
def test_indexed_conversion_preserves_paths_and_recomputes_total_size(tmp_path: Path, nested: bool) -> None:
    input_dir = _write_checkpoint(tmp_path, nested=nested)
    original_index = (input_dir / _INDEX).read_bytes()
    original_config = (input_dir / "config.json").read_bytes()
    input_index = converter.load_json(input_dir / _INDEX)
    output_dir = tmp_path / "output"

    _run_main(input_dir, output_dir)

    output_index = converter.load_json(output_dir / _INDEX)
    assert output_index["weight_map"] == {key: name for key, name in input_index["weight_map"].items() if key != _SCALE}
    # Input: 6 FP8 bytes + 8 scale bytes + 12 bias bytes. Output: 12 BF16 + 12 bias bytes.
    assert output_index["metadata"]["total_size"] == 24
    output_weight = load_file(str(output_dir / input_index["weight_map"][_WEIGHT]))
    assert set(output_weight) == {_WEIGHT}
    torch.testing.assert_close(
        output_weight[_WEIGHT], torch.tensor([[2, 4, 9], [8, 10, 18]], dtype=torch.bfloat16), rtol=0, atol=0
    )
    torch.testing.assert_close(
        load_file(str(output_dir / input_index["weight_map"]["bias"]))["bias"], torch.tensor([7.0, 8.0, 9.0])
    )
    assert (input_dir / _INDEX).read_bytes() == original_index
    assert (input_dir / "config.json").read_bytes() == original_config
    assert not (output_dir / ".model.safetensors.index.json.tmp").exists()


def test_single_shard_without_index_is_preflighted_and_converted(tmp_path: Path) -> None:
    input_dir = _write_checkpoint(tmp_path, indexed=False)
    names, weight_map = converter.discover_safetensor_files(input_dir)
    assert names == ["model.safetensors"]
    assert weight_map == dict.fromkeys([_WEIGHT, _SCALE, "bias"], "model.safetensors")
    _run_main(input_dir, tmp_path / "output")
    index = converter.load_json(tmp_path / "output" / _INDEX)
    assert index["metadata"]["total_size"] == 24
    assert index["weight_map"] == dict.fromkeys([_WEIGHT, "bias"], "model.safetensors")


@pytest.mark.parametrize("metadata", [None, {}, {"source": "test"}])
def test_input_total_size_is_optional(tmp_path: Path, metadata: dict | None) -> None:
    input_dir = _write_checkpoint(tmp_path)
    index = converter.load_json(input_dir / _INDEX)
    if metadata is None:
        del index["metadata"]
    else:
        index["metadata"] = metadata
    converter.save_json(input_dir / _INDEX, index)
    _run_main(input_dir, tmp_path / "output")
    assert converter.load_json(tmp_path / "output" / _INDEX)["metadata"]["total_size"] == 24


@pytest.mark.parametrize("total_size", [0, 24, 27, -1, 26.0, "26", True, None])
def test_invalid_input_total_size_fails_before_output(tmp_path: Path, total_size: object) -> None:
    input_dir = _write_checkpoint(tmp_path)
    index = converter.load_json(input_dir / _INDEX)
    index["metadata"]["total_size"] = total_size
    converter.save_json(input_dir / _INDEX, index)
    _assert_preflight_failure(input_dir, tmp_path / "output", r"metadata\.total_size")


@pytest.mark.parametrize(
    "index",
    [
        [],
        None,
        {},
        {"weight_map": {}},
        {"weight_map": []},
        {"weight_map": {"": "model.safetensors"}},
        {"weight_map": {"weight": ""}},
        {"weight_map": {"weight": 3}},
        {"weight_map": {"weight": ["model.safetensors"]}},
    ],
)
def test_invalid_index_schema_fails_before_output(tmp_path: Path, index: object) -> None:
    input_dir = _write_checkpoint(tmp_path)
    (input_dir / _INDEX).write_text(json.dumps(index))
    _assert_preflight_failure(input_dir, tmp_path / "output", "index|weight_map")


@pytest.mark.parametrize("metadata", [[], "invalid", None])
def test_invalid_metadata_object_is_identified(tmp_path: Path, metadata: object) -> None:
    input_dir = _write_checkpoint(tmp_path)
    index = converter.load_json(input_dir / _INDEX)
    index["metadata"] = metadata
    converter.save_json(input_dir / _INDEX, index)
    _assert_preflight_failure(input_dir, tmp_path / "output", "metadata")


@pytest.mark.parametrize(
    "index_text",
    ['{"weight_map":', '{"weight_map":{"weight":"a.safetensors","weight":"b.safetensors"}}'],
)
def test_malformed_or_duplicate_index_keys_are_rejected(tmp_path: Path, index_text: str) -> None:
    input_dir = _write_checkpoint(tmp_path)
    (input_dir / _INDEX).write_text(index_text)
    _assert_preflight_failure(input_dir, tmp_path / "output", "model.safetensors.index.json")


def test_missing_shard_is_found_before_creating_output(tmp_path: Path) -> None:
    input_dir = _write_checkpoint(tmp_path)
    (input_dir / "model-00002.safetensors").unlink()
    _assert_preflight_failure(input_dir, tmp_path / "output", "model-00002.safetensors.*missing")


@pytest.mark.parametrize("failure", ["missing-key", "wrong-shard", "unlisted-key", "duplicate-key"])
def test_index_and_shard_keys_must_match_exactly(tmp_path: Path, failure: str) -> None:
    input_dir = _write_checkpoint(tmp_path)
    index = converter.load_json(input_dir / _INDEX)
    if failure == "missing-key":
        index["weight_map"]["missing.weight"] = "model-00001.safetensors"
    elif failure == "wrong-shard":
        index["weight_map"][_WEIGHT] = "model-00002.safetensors"
    elif failure == "unlisted-key":
        del index["weight_map"][_SCALE]
    else:
        second = input_dir / "model-00002.safetensors"
        tensors = load_file(str(second))
        tensors[_WEIGHT] = torch.ones((1, 1))
        save_file(tensors, str(second))
    converter.save_json(input_dir / _INDEX, index)
    _assert_preflight_failure(input_dir, tmp_path / "output", "weight_map mismatch.*tensor keys")


@pytest.mark.parametrize(
    "name",
    [
        "../outside.safetensors",
        "/outside.safetensors",
        "sub/../../outside.safetensors",
        "./model.safetensors",
        "sub//model.safetensors",
        "sub\\model.safetensors",
        "config.json",
    ],
)
def test_invalid_shard_paths_are_rejected_before_output(tmp_path: Path, name: str) -> None:
    input_dir = _write_checkpoint(tmp_path)
    converter.save_json(input_dir / _INDEX, {"weight_map": {"weight": name}})
    _assert_preflight_failure(input_dir, tmp_path / "output", "shard path")


@pytest.mark.parametrize("indexed", [False, True])
def test_symlink_to_shard_outside_input_is_rejected(tmp_path: Path, indexed: bool) -> None:
    input_dir = _write_checkpoint(tmp_path, indexed=indexed)
    name = "model-00001.safetensors" if indexed else "model.safetensors"
    original = input_dir / name
    external = tmp_path / "outside.safetensors"
    original.rename(external)
    original.symlink_to(external)
    contents = external.read_bytes()
    _assert_preflight_failure(input_dir, tmp_path / "output", "resolves outside")
    assert external.read_bytes() == contents


def test_symlinked_directory_cannot_escape_input(tmp_path: Path) -> None:
    input_dir = _write_checkpoint(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    save_file({"weight": torch.ones(1)}, str(outside / "model.safetensors"))
    (input_dir / "nested").symlink_to(outside, target_is_directory=True)
    converter.save_json(input_dir / _INDEX, {"weight_map": {"weight": "nested/model.safetensors"}})
    _assert_preflight_failure(input_dir, tmp_path / "output", "resolves outside")


def test_in_directory_shard_symlink_is_supported(tmp_path: Path) -> None:
    input_dir = _write_checkpoint(tmp_path, indexed=False)
    shard = input_dir / "model.safetensors"
    backing = input_dir / "weights.data"
    shard.rename(backing)
    shard.symlink_to(backing)
    files, weight_map = converter.discover_safetensor_files(input_dir)
    assert files == ["model.safetensors"]
    assert set(weight_map) == {_WEIGHT, _SCALE, "bias"}


@pytest.mark.parametrize("kind", ["directory", "corrupt", "empty", "dangling-symlink"])
def test_invalid_referenced_shard_fails_before_output(tmp_path: Path, kind: str) -> None:
    input_dir = _write_checkpoint(tmp_path)
    shard = input_dir / "model-00002.safetensors"
    shard.unlink()
    if kind == "directory":
        shard.mkdir()
    elif kind == "corrupt":
        shard.write_bytes(b"invalid safetensors")
    elif kind == "empty":
        save_file({}, str(shard))
    else:
        shard.symlink_to(input_dir / "absent.safetensors")
    _assert_preflight_failure(input_dir, tmp_path / "output", "model-00002.safetensors")


def test_unreferenced_shards_are_ignored_when_index_exists(tmp_path: Path) -> None:
    input_dir = _write_checkpoint(tmp_path)
    (input_dir / "unused.safetensors").write_bytes(b"unrelated file")
    _run_main(input_dir, tmp_path / "output")
    assert not (tmp_path / "output" / "unused.safetensors").exists()


def test_preflight_reads_headers_without_materializing_tensors(tmp_path: Path) -> None:
    input_dir = _write_checkpoint(tmp_path)
    real_open = converter.safe_open
    contexts = []

    def headers_only(*args: object, **kwargs: object) -> object:
        context = real_open(*args, **kwargs)
        contexts.append(context)

        # Only keys() is exposed; a call to get_tensor() would fail this test.
        class HeaderView:
            def __enter__(self) -> "HeaderView":
                context.__enter__()
                return self

            def keys(self) -> list[str]:
                return context.keys()

            def __exit__(self, *exc: object) -> None:
                context.__exit__(*exc)

        return HeaderView()

    with patch.object(converter, "safe_open", side_effect=headers_only):
        _, weight_map = converter.discover_safetensor_files(input_dir)
    assert len(contexts) == 2
    assert set(weight_map) == {_WEIGHT, _SCALE, "bias"}


def test_later_conversion_failure_never_publishes_output_index(tmp_path: Path) -> None:
    input_dir = _write_checkpoint(tmp_path)
    second = input_dir / "model-00002.safetensors"
    save_file({"bad.weight": torch.ones((2, 2)).to(torch.float8_e4m3fn)}, str(second))
    converter.save_json(
        input_dir / _INDEX,
        {
            "weight_map": {
                _WEIGHT: "model-00001.safetensors",
                _SCALE: "model-00001.safetensors",
                "bad.weight": second.name,
            }
        },
    )
    output_dir = tmp_path / "output"
    with pytest.raises(ValueError, match="[Mm]issing paired scale"):
        _run_main(input_dir, output_dir)
    assert (output_dir / "model-00001.safetensors").exists()
    assert not (output_dir / _INDEX).exists()
    assert not (output_dir / ".model.safetensors.index.json.tmp").exists()


def test_index_write_failure_never_leaves_a_final_index(tmp_path: Path) -> None:
    input_dir = _write_checkpoint(tmp_path)
    output_dir = tmp_path / "output"
    real_save = converter.save_json

    def fail_index_write(path: Path, data: dict) -> None:
        if path.name == ".model.safetensors.index.json.tmp":
            path.write_text('{"weight_map":')
            raise OSError("simulated index write failure")
        real_save(path, data)

    with (
        patch.object(converter, "save_json", side_effect=fail_index_write),
        pytest.raises(OSError, match="index write"),
    ):
        _run_main(input_dir, output_dir)
    assert not (output_dir / _INDEX).exists()
    assert not (output_dir / ".model.safetensors.index.json.tmp").exists()


def test_existing_output_index_is_not_overwritten(tmp_path: Path) -> None:
    input_dir = _write_checkpoint(tmp_path)
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    previous = b'{"existing":"checkpoint"}'
    (output_dir / _INDEX).write_bytes(previous)
    with pytest.raises(ValueError, match="not empty"):
        _run_main(input_dir, output_dir)
    assert (output_dir / _INDEX).read_bytes() == previous


@pytest.mark.parametrize("shard_count", [0, 2])
def test_unindexed_checkpoint_requires_exactly_one_shard(tmp_path: Path, shard_count: int) -> None:
    input_dir = _write_checkpoint(tmp_path, indexed=False)
    if shard_count == 0:
        (input_dir / "model.safetensors").unlink()
    else:
        save_file({"extra": torch.ones(1)}, str(input_dir / "extra.safetensors"))
    _assert_preflight_failure(input_dir, tmp_path / "output", "single .safetensors shard")


def test_file_sizes_including_headers_are_not_valid_total_size(tmp_path: Path) -> None:
    input_dir = _write_checkpoint(tmp_path)
    index = converter.load_json(input_dir / _INDEX)
    index["metadata"]["total_size"] = sum(path.stat().st_size for path in input_dir.glob("*.safetensors"))
    converter.save_json(input_dir / _INDEX, index)
    _assert_preflight_failure(input_dir, tmp_path / "output", "expected 26 input tensor payload bytes")


def test_dangling_index_symlink_is_not_treated_as_an_unindexed_checkpoint(tmp_path: Path) -> None:
    input_dir = _write_checkpoint(tmp_path, indexed=False)
    (input_dir / _INDEX).symlink_to(input_dir / "missing-index.json")
    _assert_preflight_failure(input_dir, tmp_path / "output", "model.safetensors.index.json")


def test_zero_payload_tensor_is_valid_with_total_size_zero(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    save_file({"empty": torch.empty(0)}, str(input_dir / "model.safetensors"))
    converter.save_json(
        input_dir / _INDEX, {"metadata": {"total_size": 0}, "weight_map": {"empty": "model.safetensors"}}
    )
    files, weight_map = converter.discover_safetensor_files(input_dir)
    assert files == ["model.safetensors"]
    assert weight_map == {"empty": "model.safetensors"}


def test_index_rename_failure_does_not_publish_or_leave_temporary_index(tmp_path: Path) -> None:
    input_dir = _write_checkpoint(tmp_path)
    output_dir = tmp_path / "output"
    with (
        patch.object(Path, "replace", side_effect=OSError("simulated rename failure")),
        pytest.raises(OSError, match="rename failure"),
    ):
        _run_main(input_dir, output_dir)
    assert not (output_dir / _INDEX).exists()
    assert not (output_dir / ".model.safetensors.index.json.tmp").exists()

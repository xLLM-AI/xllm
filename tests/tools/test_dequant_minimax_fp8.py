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

"""Validate MiniMax FP8 conversion with small CPU tensors and local shards."""

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file

from tools import dequant_minimax_fp8 as converter

_FP8_DTYPES = [torch.float8_e4m3fn, torch.float8_e4m3fnuz, torch.float8_e5m2, torch.float8_e5m2fnuz]
_SCALE_DTYPES = [torch.float16, torch.bfloat16, torch.float32, torch.float64]
_WEIGHT_NAME = "model.layers.0.mlp.experts.0.w1.weight"
_SCALE_NAME = f"{_WEIGHT_NAME}_scale_inv"


@pytest.mark.parametrize("fp8_dtype", _FP8_DTYPES)
@pytest.mark.parametrize("scale_dtype", _SCALE_DTYPES)
@pytest.mark.parametrize("shape", [(4, 4), (3, 4), (4, 3), (3, 3), (1, 1)])
def test_full_and_partial_blocks_match_reference(
    fp8_dtype: torch.dtype, scale_dtype: torch.dtype, shape: tuple[int, int]
) -> None:
    n, k = shape
    weight = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8], [-1, -2, -3, -4], [-5, -6, -7, -8]])
    weight = weight[:n, :k].to(fp8_dtype)
    scales = torch.tensor([[0.5, 2], [4, 0.25]], dtype=scale_dtype)[: (n + 1) // 2, : (k + 1) // 2]
    expected = torch.tensor(
        [[0.5, 1, 6, 8], [2.5, 3, 14, 16], [-4, -8, -0.75, -1], [-20, -24, -1.75, -2]],
        dtype=torch.bfloat16,
    )[:n, :k]

    result = converter.dequantize_fp8_block_weight(weight, scales, (2, 2))

    torch.testing.assert_close(result, expected, rtol=0, atol=0)
    assert result.device.type == "cpu"


def test_non_square_blocks_and_non_contiguous_inputs() -> None:
    weight = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8], [1, 2, 3, 4]]).to(torch.float8_e4m3fn).T
    scales = torch.tensor([[2.0, 3.0], [4.0, 5.0]]).T
    original_weight = weight.to(torch.float32).clone()
    original_scales = scales.clone()
    expected = torch.tensor([[2, 4, 6, 12], [10, 12, 14, 24], [4, 8, 12, 20]], dtype=torch.bfloat16).T
    assert not weight.is_contiguous()
    assert not scales.is_contiguous()

    result = converter.dequantize_fp8_block_weight(weight, scales, (3, 2))

    torch.testing.assert_close(result, expected, rtol=0, atol=0)
    torch.testing.assert_close(weight.to(torch.float32), original_weight, rtol=0, atol=0)
    torch.testing.assert_close(scales, original_scales, rtol=0, atol=0)


def test_model_block_size_accepts_a_smaller_edge_block() -> None:
    weight = torch.tensor([[1.0, -2.0, 4.0]]).to(torch.float8_e4m3fn)
    result = converter.dequantize_fp8_block_weight(weight, torch.tensor([[0.5]]), (128, 128))
    torch.testing.assert_close(result, torch.tensor([[0.5, -1, 2]], dtype=torch.bfloat16), rtol=0, atol=0)


@pytest.mark.parametrize(
    "block_size",
    [
        None,
        2,
        "2,2",
        (),
        (2,),
        (2, 2, 2),
        (0, 2),
        (2, 0),
        (-1, 2),
        (2, -1),
        (True, 2),
        (2, False),
        (2.0, 2),
        (2, 2.5),
        ("2", 2),
    ],
)
def test_invalid_block_size_is_identified(block_size: object) -> None:
    weight = torch.ones((2, 2)).to(torch.float8_e4m3fn)
    with pytest.raises(ValueError, match="block_size"):
        converter.dequantize_fp8_block_weight(weight, torch.ones((1, 1)), block_size)


@pytest.mark.parametrize("shape", [(), (4,), (1, 2, 2)])
def test_weight_rank_is_identified(shape: tuple[int, ...]) -> None:
    weight = torch.ones(shape).to(torch.float8_e4m3fn)
    with pytest.raises(ValueError, match=r"fp8_weight\.shape"):
        converter.dequantize_fp8_block_weight(weight, torch.ones((1, 1)), (2, 2))


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.int8, torch.bool, torch.complex64])
def test_non_fp8_weight_dtype_is_identified(dtype: torch.dtype) -> None:
    with pytest.raises(ValueError, match=r"fp8_weight\.dtype"):
        converter.dequantize_fp8_block_weight(torch.ones((2, 2), dtype=dtype), torch.ones((1, 1)), (2, 2))


@pytest.mark.parametrize("shape", [(), (4,), (1, 2, 2), (1, 1), (1, 2), (2, 1), (2, 3)])
def test_scale_shape_requires_the_exact_ceil_grid(shape: tuple[int, ...]) -> None:
    weight = torch.ones((3, 3)).to(torch.float8_e4m3fn)
    with pytest.raises(ValueError, match=r"weight_scale_inv\.shape"):
        converter.dequantize_fp8_block_weight(weight, torch.ones(shape), (2, 2))


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64, torch.bool, torch.complex64, torch.float8_e4m3fn])
def test_scale_dtype_is_identified(dtype: torch.dtype) -> None:
    weight = torch.ones((2, 2)).to(torch.float8_e4m3fn)
    scales = torch.ones((1, 1)).to(dtype)
    with pytest.raises(ValueError, match=r"weight_scale_inv\.dtype"):
        converter.dequantize_fp8_block_weight(weight, scales, (2, 2))


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("dtype", _SCALE_DTYPES)
def test_non_finite_scale_is_identified(value: float, dtype: torch.dtype) -> None:
    weight = torch.ones((3, 3)).to(torch.float8_e4m3fn)
    scales = torch.ones((2, 2), dtype=dtype)
    scales[1, 1] = value
    with pytest.raises(ValueError, match="weight_scale_inv.*finite"):
        converter.dequantize_fp8_block_weight(weight, scales, (2, 2))


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("dtype", _FP8_DTYPES)
def test_non_finite_weight_is_identified(value: float, dtype: torch.dtype) -> None:
    weight = torch.ones((3, 3))
    weight[2, 2] = value
    with pytest.raises(ValueError, match="fp8_weight.*finite"):
        converter.dequantize_fp8_block_weight(weight.to(dtype), torch.ones((2, 2)), (2, 2))


def test_scale_that_overflows_bf16_is_identified() -> None:
    weight = torch.ones((2, 2)).to(torch.float8_e4m3fn)
    scales = torch.tensor([[1e100]], dtype=torch.float64)
    with pytest.raises(ValueError, match="weight_scale_inv.*finite.*bfloat16"):
        converter.dequantize_fp8_block_weight(weight, scales, (2, 2))


@pytest.mark.parametrize(
    "block_size", [None, [], [2], [2, 2, 2], [0, 2], [2, -1], [True, 2], [2, False], [2.0, 2], [2, 2.5], ["2", 2]]
)
def test_invalid_config_block_size_fails_before_writing_config(tmp_path: Path, block_size: object) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    output_dir.mkdir()
    config = {"quantization_config": {"quant_method": "fp8", "weight_block_size": block_size}}
    (input_dir / "config.json").write_text(json.dumps(config))

    with pytest.raises(ValueError, match=r"quantization_config\.weight_block_size"):
        converter.update_config(input_dir, output_dir)

    assert not (output_dir / "config.json").exists()
    assert json.loads((input_dir / "config.json").read_text()) == config


@pytest.mark.parametrize("invalid_field", ["shape", "dtype", "finite", "weight", "missing"])
def test_shard_error_names_the_shard_tensor_and_field(tmp_path: Path, invalid_field: str) -> None:
    weight = torch.ones((3, 3)).to(torch.float8_e4m3fn)
    scales = torch.ones((2, 2))
    field = "weight_scale_inv"
    if invalid_field == "shape":
        scales = torch.ones((1, 1))
        field += ".shape"
    elif invalid_field == "dtype":
        scales = scales.to(torch.int32)
        field += ".dtype"
    elif invalid_field == "finite":
        scales[1, 1] = float("nan")
    elif invalid_field == "weight":
        weight = torch.full((3, 3), float("nan")).to(torch.float8_e4m3fn)
        field = "fp8_weight"
    tensors = {_WEIGHT_NAME: weight, _SCALE_NAME: scales}
    if invalid_field == "missing":
        del tensors[_SCALE_NAME]
    source = tmp_path / "model-00001-of-00002.safetensors"
    output = tmp_path / "converted.safetensors"
    save_file(tensors, str(source))

    with pytest.raises(ValueError) as error:
        converter.convert_shard(source, output, (2, 2))

    message = str(error.value)
    assert source.name in message
    assert _WEIGHT_NAME in message
    assert _SCALE_NAME in message
    assert field in message
    assert not output.exists()


def test_cli_converts_a_small_local_checkpoint(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    config = {
        "model_type": "minimax_m2",
        "torch_dtype": "float8_e4m3fn",
        "quantization_config": {"quant_method": "fp8", "weight_block_size": [2, 3]},
    }
    (input_dir / "config.json").write_text(json.dumps(config))
    (input_dir / "tokenizer_config.json").write_text("{}\n")
    weight = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8], [1, 2, 3, 4]]).to(torch.float8_e4m3fn)
    bias = torch.tensor([1.0, 2.0, 3.0])
    save_file(
        {_WEIGHT_NAME: weight, _SCALE_NAME: torch.tensor([[2.0, 3.0], [4.0, 5.0]]), "bias": bias},
        str(input_dir / "model.safetensors"),
    )

    subprocess.run(
        [
            sys.executable,
            str(Path(converter.__file__).resolve()),
            "--input-dir",
            str(input_dir),
            "--output-dir",
            str(output_dir),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )

    converted = load_file(str(output_dir / "model.safetensors"))
    expected = torch.tensor([[2, 4, 6, 12], [10, 12, 14, 24], [4, 8, 12, 20]], dtype=torch.bfloat16)
    assert set(converted) == {_WEIGHT_NAME, "bias"}
    torch.testing.assert_close(converted[_WEIGHT_NAME], expected, rtol=0, atol=0)
    torch.testing.assert_close(converted["bias"], bias, rtol=0, atol=0)
    output_config = json.loads((output_dir / "config.json").read_text())
    assert output_config == {"model_type": "minimax_m2", "torch_dtype": "bfloat16"}
    index = json.loads((output_dir / "model.safetensors.index.json").read_text())
    assert index == {
        "metadata": {"total_size": 3 * 4 * 2 + 3 * 4},
        "weight_map": {_WEIGHT_NAME: "model.safetensors", "bias": "model.safetensors"},
    }
    assert (output_dir / "tokenizer_config.json").read_text() == "{}\n"
    assert json.loads((input_dir / "config.json").read_text()) == config

#!/usr/bin/env python3
"""Offline dequantize a MiniMax-M2.5 FP8 checkpoint into BF16 safetensors.

The current xLLM MiniMax path dequantizes FP8 block weights during
`load_state_dict`. This tool moves that work offline so repeated server
restarts can reuse a dequantized checkpoint directory.

Usage:
    python tools/dequant_minimax_fp8.py \
        --input-dir /models/MiniMax-M2.5 \
        --output-dir /models/MiniMax-M2.5-bf16
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline dequantize MiniMax-M2.5 FP8 safetensors to BF16."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Input MiniMax checkpoint directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for the dequantized checkpoint.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, sort_keys=True)
        f.write("\n")


def is_fp8_dtype(dtype: torch.dtype) -> bool:
    return dtype in {
        torch.float8_e4m3fn,
        torch.float8_e4m3fnuz,
        torch.float8_e5m2,
        torch.float8_e5m2fnuz,
    }


def dequantize_fp8_block_weight(
    fp8_weight: torch.Tensor,
    weight_scale_inv: torch.Tensor,
    block_size: tuple[int, int],
) -> torch.Tensor:
    if fp8_weight.dim() != 2:
        raise ValueError(
            f"Only 2D fp8 weights are supported, got shape {tuple(fp8_weight.shape)}"
        )
    if weight_scale_inv.dim() != 2:
        raise ValueError(
            "FP8 weight scale tensor must be 2D, "
            f"got shape {tuple(weight_scale_inv.shape)}"
        )

    block_n, block_k = block_size
    n, k = fp8_weight.shape
    n_tiles = (n + block_n - 1) // block_n
    k_tiles = (k + block_k - 1) // block_k

    if tuple(weight_scale_inv.shape) != (n_tiles, k_tiles):
        raise ValueError(
            "Unexpected fp8 scale shape "
            f"{tuple(weight_scale_inv.shape)} for weight shape {tuple(fp8_weight.shape)}"
        )

    if n % block_n == 0 and k % block_k == 0:
        weight_bf16 = fp8_weight.to(torch.bfloat16).reshape(
            n_tiles, block_n, k_tiles, block_k
        )
        scale_bf16 = weight_scale_inv.to(torch.bfloat16).reshape(
            n_tiles, 1, k_tiles, 1
        )
        return (weight_bf16 * scale_bf16).reshape(n, k)

    expanded_scale = weight_scale_inv.repeat_interleave(block_n, 0).repeat_interleave(
        block_k, 1
    )
    expanded_scale = expanded_scale[:n, :k].to(torch.bfloat16)
    return fp8_weight.to(torch.bfloat16) * expanded_scale


def discover_safetensor_files(input_dir: Path) -> tuple[list[str], dict[str, str]]:
    index_path = input_dir / "model.safetensors.index.json"
    if index_path.exists():
        index_data = load_json(index_path)
        weight_map = index_data["weight_map"]
        files = sorted(set(weight_map.values()))
        return files, weight_map

    shard_paths = sorted(p.name for p in input_dir.glob("*.safetensors"))
    if len(shard_paths) != 1:
        raise ValueError(
            "Expected model.safetensors.index.json or a single .safetensors shard "
            f"under {input_dir}"
        )
    return shard_paths, {}


def prepare_output_dir(output_dir: Path) -> None:
    if output_dir.exists():
        if any(output_dir.iterdir()):
            raise ValueError(
                f"Output directory {output_dir} already exists and is not empty"
            )
    else:
        output_dir.mkdir(parents=True, exist_ok=True)


def copy_non_safetensors_files(input_dir: Path, output_dir: Path) -> None:
    for path in input_dir.iterdir():
        if path.is_file() and path.suffix != ".safetensors":
            if path.name in {"config.json", "model.safetensors.index.json"}:
                continue
            shutil.copy2(path, output_dir / path.name)


def update_config(input_dir: Path, output_dir: Path) -> tuple[int, int]:
    config_path = input_dir / "config.json"
    if not config_path.exists():
        raise ValueError(f"Missing config.json under {input_dir}")

    config = load_json(config_path)
    quant_config = config.get("quantization_config", {})
    if quant_config.get("quant_method") != "fp8":
        raise ValueError(
            "This converter expects quantization_config.quant_method == 'fp8'"
        )

    block_size = quant_config.get("weight_block_size")
    if not isinstance(block_size, list) or len(block_size) != 2:
        raise ValueError(
            "This converter expects quantization_config.weight_block_size to be a "
            "2-element list"
        )

    config["torch_dtype"] = "bfloat16"
    config.pop("quantization_config", None)
    save_json(output_dir / "config.json", config)
    return int(block_size[0]), int(block_size[1])


def tensor_nbytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def convert_shard(
    input_path: Path, output_path: Path, block_size: tuple[int, int]
) -> tuple[dict[str, str], int, int]:
    converted: dict[str, torch.Tensor] = {}
    shard_weight_map: dict[str, str] = {}
    converted_fp8_tensors = 0
    total_bytes = 0

    with safe_open(input_path, framework="pt") as f:
        keys = list(f.keys())
        key_set = set(keys)
        skipped = set()

        for name in keys:
            if name in skipped:
                continue

            if name.endswith(".weight_scale_inv"):
                paired_weight_name = name[: -len("_scale_inv")]
                if paired_weight_name in key_set:
                    paired_weight = f.get_tensor(paired_weight_name)
                    if is_fp8_dtype(paired_weight.dtype):
                        skipped.add(name)
                        continue

            tensor = f.get_tensor(name)
            if is_fp8_dtype(tensor.dtype):
                scale_name = f"{name}_scale_inv"
                if scale_name not in key_set:
                    raise ValueError(
                        f"Missing paired scale tensor {scale_name} for FP8 weight {name}"
                    )
                scale = f.get_tensor(scale_name)
                tensor = dequantize_fp8_block_weight(tensor, scale, block_size)
                skipped.add(scale_name)
                converted_fp8_tensors += 1

            converted[name] = tensor
            shard_weight_map[name] = output_path.name
            total_bytes += tensor_nbytes(tensor)

    save_file(converted, str(output_path), metadata={"format": "pt"})
    return shard_weight_map, converted_fp8_tensors, total_bytes


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()

    if input_dir == output_dir:
        raise ValueError("--input-dir and --output-dir must differ")

    prepare_output_dir(output_dir)
    copy_non_safetensors_files(input_dir, output_dir)
    block_size = update_config(input_dir, output_dir)

    shard_names, _ = discover_safetensor_files(input_dir)

    total_weight_map: dict[str, str] = {}
    total_size = 0
    total_fp8_tensors = 0
    total_shards = len(shard_names)

    print(
        f"Dequantizing MiniMax FP8 checkpoint from {input_dir} to {output_dir} "
        f"with block_size={list(block_size)}"
    )
    for i, shard_name in enumerate(shard_names, start=1):
        input_path = input_dir / shard_name
        output_path = output_dir / shard_name
        print(f"[{i}/{total_shards}] Processing {shard_name}")
        shard_weight_map, converted_fp8_tensors, shard_bytes = convert_shard(
            input_path, output_path, block_size
        )
        total_weight_map.update(shard_weight_map)
        total_fp8_tensors += converted_fp8_tensors
        total_size += shard_bytes

    index_data = {
        "metadata": {"total_size": total_size},
        "weight_map": total_weight_map,
    }
    save_json(output_dir / "model.safetensors.index.json", index_data)
    print(
        "Done. Converted "
        f"{total_fp8_tensors} fp8 tensors. "
        f"Point MODEL_PATH to {output_dir} to use the bf16 checkpoint cache."
    )


if __name__ == "__main__":
    main()


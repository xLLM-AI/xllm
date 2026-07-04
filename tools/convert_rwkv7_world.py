#!/usr/bin/env python3
"""Convert BlinkDL / ModelScope RWKV-7 World checkpoints to xLLM model layout.

xLLM only loads HuggingFace-style model directories. The official RWKV-7-World
release ships native PyTorch checkpoints instead, so this script bridges the
two formats.

Input format (ModelScope / BlinkDL native)
------------------------------------------
A single source directory, typically downloaded from ModelScope::

  rwkv-7-world/
  ├── README.md
  ├── configuration.json          # {"framework": "Pytorch", "task": "text-generation"}
  ├── RWKV-x070-World-0.1B-v2.8-20241210-ctx4096.pth
  ├── RWKV-x070-World-0.4B-v2.9-20250107-ctx4096.pth
  ├── RWKV-x070-World-1.5B-v3-20250127-ctx4096.pth
  └── RWKV-x070-World-2.9B-v3-20250211-ctx4096.pth

Characteristics of the native format:

- Weights are stored as raw ``.pth`` state dicts (``emb.weight``, ``blocks.*``).
- There is no ``config.json`` with ``model_type``, layer dims, or context length.
- There is no ``model.safetensors``.
- There is no tokenizer (``tokenizer.json`` / ``tokenizer_config.json``).
- Inference with the official ``rwkv`` pip package reads ``.pth`` directly.

Output format (xLLM-ready HuggingFace-style layout)
---------------------------------------------------
One directory per converted checkpoint::

  rwkv-7-world-0.1b-xllm/
  ├── config.json                 # model_type=rwkv7, dims inferred from weights
  ├── model.safetensors           # same tensors as .pth, safetensors format
  ├── rwkv_vocab_v20230424.txt    # official trie vocab for xLLM rwkv tokenizer
  └── tokenizer_config.json       # tokenizer_type=rwkv

``config.json`` fields are inferred from the checkpoint, for example::

  {
    "model_type": "rwkv7",
    "torch_dtype": "float16",
    "vocab_size": 65536,
    "hidden_size": 768,
    "num_hidden_layers": 12,
    "head_size": 64,
    "intermediate_size": 3072,
    "num_attention_heads": 12,
    "max_position_embeddings": 4096,
    ...
  }

Supported sizes
---------------
- 0.1B: L12, hidden=768
- 0.4B: L24, hidden=1024
- 1.5B: L24, hidden=2048
- 2.9B: L32, hidden=2560

Dependencies
------------
- pip install rwkv safetensors torch

Usage
-----
::

  python tools/convert_rwkv7_world.py \\
    --src /path/to/rwkv-7-world \\
    --dst /path/to/output_parent \\
    --sizes all \\
    --dtype float16

Alternatively, set ``RWKV7_WORLD_SRC`` and ``RWKV7_WORLD_DST`` instead of
``--src`` / ``--dst``.

After conversion, start xLLM with::

  ./build/xllm/core/server/xllm \\
    --model=/path/to/output_parent/rwkv-7-world-0.1b-xllm \\
    --port=9977 --device_id=0
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.logger import logger

import torch
from safetensors.torch import save_file

try:
    import rwkv
except ImportError as exc:
    logger.error("Missing dependency 'rwkv'. Install with: pip install rwkv")
    raise SystemExit(1) from exc


ENV_SRC = "RWKV7_WORLD_SRC"
ENV_DST = "RWKV7_WORLD_DST"

# Known ModelScope filenames for rwkv-7-world.
CHECKPOINTS = [
    {
        "pth": "RWKV-x070-World-0.1B-v2.8-20241210-ctx4096.pth",
        "suffix": "0.1b-xllm",
        "ctx": 4096,
    },
    {
        "pth": "RWKV-x070-World-0.4B-v2.9-20250107-ctx4096.pth",
        "suffix": "0.4b-xllm",
        "ctx": 4096,
    },
    {
        "pth": "RWKV-x070-World-1.5B-v3-20250127-ctx4096.pth",
        "suffix": "1.5b-xllm",
        "ctx": 4096,
    },
    {
        "pth": "RWKV-x070-World-2.9B-v3-20250211-ctx4096.pth",
        "suffix": "2.9b-xllm",
        "ctx": 4096,
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert RWKV-7-World native .pth checkpoints to xLLM "
            "HuggingFace-style directories (config.json + model.safetensors "
            "+ tokenizer files)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Input:  ModelScope dir with RWKV-x070-World-*.pth files only.\n"
            "Output: rwkv-7-world-<size>-xllm/ per checkpoint."
        ),
    )
    parser.add_argument(
        "--src",
        help=(
            "Source directory in ModelScope/BlinkDL native format "
            f"(required unless {ENV_SRC} is set)"
        ),
    )
    parser.add_argument(
        "--dst",
        help=(
            "Parent directory for xLLM-ready output folders "
            f"(required unless {ENV_DST} is set)"
        ),
    )
    parser.add_argument(
        "--sizes",
        default="all",
        help="Comma-separated sizes to convert: 0.1b,0.4b,1.5b,2.9b or 'all'",
    )
    parser.add_argument(
        "--dtype",
        default="float16",
        choices=("float16", "bfloat16", "float32"),
        help="torch_dtype written into config.json",
    )
    return parser.parse_args()


def infer_arch(state: dict[str, torch.Tensor]) -> dict:
    emb = state["emb.weight"]
    n_layers = max(
        int(key.split(".")[1])
        for key in state
        if key.startswith("blocks.")
    ) + 1
    head_size = int(state["blocks.0.att.r_k"].shape[-1])
    hidden_size = int(emb.shape[1])
    return {
        "vocab_size": int(emb.shape[0]),
        "hidden_size": hidden_size,
        "num_hidden_layers": n_layers,
        "head_size": head_size,
        "intermediate_size": int(state["blocks.0.ffn.key.weight"].shape[0]),
        "num_attention_heads": hidden_size // head_size,
    }


def build_config(arch: dict, ctx: int, dtype: str) -> dict:
    return {
        "model_type": "rwkv7",
        "torch_dtype": dtype,
        "vocab_size": arch["vocab_size"],
        "hidden_size": arch["hidden_size"],
        "num_hidden_layers": arch["num_hidden_layers"],
        "head_size": arch["head_size"],
        "intermediate_size": arch["intermediate_size"],
        "num_attention_heads": arch["num_attention_heads"],
        "layer_norm_eps": 1e-5,
        "max_position_embeddings": ctx,
        "bos_token_id": 0,
        "eos_token_id": 0,
    }


def export_tokenizer(out_dir: Path) -> None:
    vocab_path = Path(rwkv.__file__).resolve().parent / "rwkv_vocab_v20230424.txt"
    if not vocab_path.exists():
        raise FileNotFoundError(f"RWKV vocab not found: {vocab_path}")

    trie_vocab = out_dir / "rwkv_vocab_v20230424.txt"
    trie_vocab.write_bytes(vocab_path.read_bytes())

    with (out_dir / "tokenizer_config.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "tokenizer_type": "rwkv",
                "tokenizer_class": "RwkvTokenizer",
                "vocab_file": "rwkv_vocab_v20230424.txt",
                "model_max_length": 4096,
                "add_bos_token": False,
                "add_eos_token": False,
            },
            f,
            indent=2,
        )

    # Remove legacy fast-tokenizer export if present.
    legacy_tokenizer_json = out_dir / "tokenizer.json"
    if legacy_tokenizer_json.exists():
        legacy_tokenizer_json.unlink()


def convert_one(src_dir: Path, dst_parent: Path, spec: dict, dtype: str) -> Path:
    pth_path = src_dir / spec["pth"]
    if not pth_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {pth_path}")

    out_dir = dst_parent / f"rwkv-7-world-{spec['suffix']}"
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Converting {pth_path.name} -> {out_dir}")
    state = torch.load(pth_path, map_location="cpu", mmap=True, weights_only=True)
    arch = infer_arch(state)
    config = build_config(arch, spec["ctx"], dtype)

    with (out_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    state = {k: v.contiguous() for k, v in state.items()}
    save_file(state, out_dir / "model.safetensors")
    export_tokenizer(out_dir)

    logger.info(
        f"  layers={arch['num_hidden_layers']} hidden={arch['hidden_size']} "
        f"vocab={arch['vocab_size']} ctx={spec['ctx']}"
    )
    return out_dir


def selected_specs(size_arg: str) -> list[dict]:
    if size_arg.strip().lower() == "all":
        return CHECKPOINTS

    wanted = {s.strip().lower() for s in size_arg.split(",") if s.strip()}
    specs = []
    for spec in CHECKPOINTS:
        m = re.search(r"(\d+(?:\.\d+)?b)", spec["suffix"], re.I)
        label = m.group(1).lower() if m else spec["suffix"]
        if label in wanted or spec["suffix"] in wanted:
            specs.append(spec)
    if not specs:
        raise ValueError(f"No checkpoints matched --sizes={size_arg!r}")
    return specs


def resolve_io_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    src = args.src or os.environ.get(ENV_SRC)
    dst = args.dst or os.environ.get(ENV_DST)
    if not src or not dst:
        raise SystemExit(
            "Both --src and --dst are required "
            f"(or set {ENV_SRC} and {ENV_DST}).\n"
            "Example:\n"
            "  python tools/convert_rwkv7_world.py "
            "--src /path/to/rwkv-7-world --dst /path/to/output"
        )
    return Path(src).resolve(), Path(dst).resolve()


def main() -> int:
    args = parse_args()
    src_dir, dst_parent = resolve_io_paths(args)

    if not src_dir.is_dir():
        logger.error(f"Source directory not found: {src_dir}")
        return 1

    outputs: list[Path] = []
    for spec in selected_specs(args.sizes):
        outputs.append(convert_one(src_dir, dst_parent, spec, args.dtype))

    logger.info("Done. Start xLLM with one of:")
    for out_dir in outputs:
        logger.info(
            f"  ./build/xllm/core/server/xllm "
            f"--model={out_dir} --port=9977 --device_id=0"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

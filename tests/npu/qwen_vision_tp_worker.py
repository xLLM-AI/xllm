# Copyright 2026 The xLLM Authors.
#
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

"""Compare real TP vision towers against TP1; launch with torchrun -m.

Run on explicitly selected idle NPUs after building this checkout, for example:
python -m torch.distributed.run --nproc_per_node=4 --master_port=18753 \
    -m tests.npu.qwen_vision_tp_worker --artifact-dir /tmp/vision-tp
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from scripts.logger import logger


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    args = parser.parse_args()
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    device = torch.device(f"npu:{os.environ['LOCAL_RANK']}")

    from xllm import xllm_export  # noqa: F401
    from xllm.python import distributed, initialize_runtime

    initialize_runtime()
    torch.npu.set_device(device)
    distributed.init_tp_group(
        os.environ["MASTER_ADDR"], int(os.environ["MASTER_PORT"]), rank, world_size, str(device), rank, world_size, 0
    )

    from tests.python.qwen3_5_test_utils import StateDict
    from xllm.python.model_loader import ScopedWeightLoader
    from xllm.python.models.qwen3_vl import Qwen3VLVisionConfig, Qwen3VLVisionTransformer

    results = []
    for head_dim, deepstack in ((72, []), (72, [0]), (64, [])):
        cfg = Qwen3VLVisionConfig(
            depth=2,
            hidden_size=4 * head_dim,
            num_heads=4,
            intermediate_size=640,
            out_hidden_size=64,
            patch_size=2,
            num_position_embeddings=16,
            deepstack_visual_indexes=deepstack,
        )
        reference = Qwen3VLVisionTransformer(cfg, dtype=torch.bfloat16, device=device)
        sharded = Qwen3VLVisionTransformer(cfg, dtype=torch.bfloat16, device=device, tp_size=world_size)
        generator = torch.Generator().manual_seed(37)
        checkpoint = {}
        for name, param in reference.named_parameters():
            value = torch.randn(param.shape, generator=generator) * 0.02
            if "norm" in name and name.endswith("weight"):
                value += 1
            checkpoint[name.replace("patch_embed.", "patch_embed.proj.")] = value.to(torch.bfloat16)
        weights = ScopedWeightLoader([StateDict(checkpoint)])
        reference.load_weights(weights)
        sharded.load_weights(weights, rank)
        grid = torch.tensor([[1, 4, 4], [2, 2, 4]], dtype=torch.int32, device=device)
        pixels = torch.randn(32, 24, generator=generator).to(device=device, dtype=torch.bfloat16)
        hidden = torch.randn(32, cfg.hidden_size, generator=generator).to(device=device, dtype=torch.bfloat16)
        cosine = torch.ones(1, 32, 1, head_dim, device=device, dtype=torch.bfloat16)
        sine = torch.zeros_like(cosine)
        with torch.inference_mode():
            expected_attention = reference.blocks[0].attn(hidden, [16, 24, 32], cosine, sine)
            actual_attention = sharded.blocks[0].attn(hidden, [16, 24, 32], cosine, sine)
            expected_mlp = reference.blocks[0].mlp(hidden)
            actual_mlp = sharded.blocks[0].mlp(hidden)
            expected = reference(pixels, grid)
            actual = sharded(pixels, grid)
        torch.npu.synchronize()
        torch.testing.assert_close(
            actual_attention.cpu().float(), expected_attention.cpu().float(), rtol=0.03, atol=0.01
        )
        torch.testing.assert_close(actual_mlp.cpu().float(), expected_mlp.cpu().float(), rtol=0.03, atol=0.01)
        torch.testing.assert_close(actual.cpu().float(), expected.cpu().float(), rtol=0.03, atol=0.01)
        error = (actual.float() - expected.float()).abs()
        results.append(
            dict(
                head_dim=head_dim,
                local_heads=sharded.blocks[0].attn.num_heads,
                deepstack=deepstack,
                output_shape=list(actual.shape),
                max_abs_error=error.max().item(),
                mean_abs_error=error.mean().item(),
                attention_max_abs_error=(actual_attention.float() - expected_attention.float()).abs().max().item(),
                mlp_max_abs_error=(actual_mlp.float() - expected_mlp.float()).abs().max().item(),
            )
        )
        logger.info("Vision TP rank %d: %s", rank, results[-1])
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    (args.artifact_dir / f"rank-{rank}.json").write_text(
        json.dumps(dict(rank=rank, world_size=world_size, device=str(device), cases=results), indent=2) + "\n"
    )


if __name__ == "__main__":
    _main()

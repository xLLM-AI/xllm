# Copyright 2026 The xLLM Authors. SPDX-License-Identifier: Apache-2.0
"""Run GR00T DROID observations from an NPZ archive and save decoded actions.

Observation archive keys are video.<camera> and state.<group>. Camera tensors
have shape [1,T,H,W,3] and uint8 RGB dtype; state tensors are float32 [1,1,D].
T and camera/group names come from processor_config.json for the embodiment.
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from scripts.logger import logger
from xllm.vla.gr00t import Gr00tDroidPolicy


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--backbone-config", type=Path, required=True)
    parser.add_argument("--processor", type=Path, required=True)
    parser.add_argument("--observation", type=Path, required=True)
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--attention-backend",
        choices=["math", "flash"],
        default="flash",
        help="Attention backend for text, vision and action layers",
    )
    parser.add_argument("--action-execution", choices=["eager", "graph"], default="eager")
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    policy = Gr00tDroidPolicy(
        args.model,
        args.backbone_config,
        args.processor,
        device=args.device,
        dtype=getattr(torch, args.dtype),
        attention_backend=args.attention_backend,
        action_execution=args.action_execution,
    )
    with np.load(args.observation, allow_pickle=False) as data:
        observation = {
            "video": {key: data["video." + key] for key in policy.processor.modalities["video"]["modality_keys"]},
            "state": {key: data["state." + key] for key in policy.processor.modalities["state"]["modality_keys"]},
            "language": {policy.processor.language_key: [[args.instruction]]},
        }
    actions = policy.get_action(observation, seed=args.seed)
    np.savez(args.output, **actions)
    logger.info(
        "Saved decoded DROID actions to %s: %s", args.output, {key: value.shape for key, value in actions.items()}
    )


if __name__ == "__main__":
    main()

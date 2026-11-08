# Copyright 2026 The xLLM Authors.
# SPDX-License-Identifier: Apache-2.0

"""Submit the fixed MiniMax-H3 T2VA baseline with precomputed conditioning."""

import argparse
import base64
import hashlib
import json
import time
import urllib.request
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="minimax-h3")
    parser.add_argument("--conditioning", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    conditioning = torch.load(args.conditioning, map_location="cpu", weights_only=True)
    if "prompt" in conditioning and conditioning["prompt"] != args.prompt:
        parser.error("--prompt must match the text used to generate the conditioning file")
    # The server stacks individual embeddings to create the batch dimension.
    embedding = conditioning["prompt_embed"].squeeze(0).float().contiguous()
    payload = {
        "model": args.model,
        "input": {
            "prompt": args.prompt,
            "prompt_embed": {
                "name": "prompt_embed",
                "datatype": "FP32",
                "shape": list(embedding.shape),
                "contents": {"fp32_contents": embedding.flatten().tolist()},
                "parameters": {"prompt_token_tags": {"int64_tensor_param": conditioning["prompt_token_tags"].tolist()}},
            },
        },
        "parameters": {
            "size": "1344*768",
            "num_inference_steps": 50,
            "seed": 42,
            "num_frames": 124,
            "fps": 24,
            "sampling_rate": 32000,
            "flow_shift": 12.0,
            "guidance_scale": 1.0,
            "num_videos_per_prompt": 1,
            "output_type": "base64",
        },
    }
    request = urllib.request.Request(
        f"{args.url.rstrip('/')}/v1/video/generation",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=3600) as response:
        result = json.load(response)
    elapsed = time.perf_counter() - started
    video = result["output"]["results"][0]["video"]["base64"]
    contents = base64.b64decode(video, validate=True)
    args.output.write_bytes(contents)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "client_seconds": elapsed,
                "mp4_sha256": hashlib.sha256(contents).hexdigest(),
            }
        )
    )


if __name__ == "__main__":
    main()

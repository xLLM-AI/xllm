# Copyright 2026 The xLLM Authors.
# SPDX-License-Identifier: Apache-2.0

"""Generate the fixed H3 reference and export its actual TP2 text conditioning.

Run with the pinned SGLang environment documented in the H3 recipe. Only the
text-stage output is observed; inference uses SGLang's existing pipeline.
"""

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
from typing import Any

_SUPPORTED_SGLANG_VERSION = "0.5.23"


def _capture_conditioning(output: Path) -> dict[str, str]:
    sglang_version = importlib.metadata.version("sglang")
    if sglang_version != _SUPPORTED_SGLANG_VERSION:
        raise RuntimeError(
            f"H3 conditioning export requires SGLang {_SUPPORTED_SGLANG_VERSION}; found {sglang_version}. "
            "Use the pinned environment in the H3 recipe."
        )

    import torch
    import torch_npu  # noqa: F401 -- registers the NPU backend.
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.constants import (
        MINIMAX_H3_TEXT_EMBEDDINGS_EXTRA_KEY,
    )
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.stages import (
        text_encoding,
    )

    forward = text_encoding.MiniMaxH3TextEncodingStage.forward

    def forward_and_export(self: Any, batch: Any, server_args: Any) -> Any:
        result = forward(self, batch, server_args)
        if torch.distributed.get_rank() == 0:
            encoder_tp_group = getattr(self.text_encoder, "_encoder_tp_group", None)
            if encoder_tp_group is None or encoder_tp_group.world_size != 2:
                raise RuntimeError("H3 conditioning export requires the pinned SGLang TP2 text encoder API")
            positive = result.extra[MINIMAX_H3_TEXT_EMBEDDINGS_EXTRA_KEY]["positive"]
            torch.save(
                {
                    "prompt_embed": positive["hidden_states"].detach().cpu().unsqueeze(0),
                    "prompt_token_tags": positive["text_token_tags"].detach().cpu(),
                    "prompt": batch.sampling_params.prompt,
                },
                output / "conditioning.pt",
            )
        return result

    text_encoding.MiniMaxH3TextEncodingStage.forward = forward_and_export
    return {
        "version": sglang_version,
        "text_encoding_sha256": hashlib.sha256(Path(text_encoding.__file__).read_bytes()).hexdigest(),
    }


# SGLang starts fresh Python workers. Install the observer in those workers too.
if os.environ.get("XLLM_H3_REFERENCE_OUTPUT"):
    _capture_conditioning(Path(os.environ["XLLM_H3_REFERENCE_OUTPUT"]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="Native FL2VA checkpoint directory")
    parser.add_argument("--devices", type=int, choices=(4, 8), required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    sglang_source = _capture_conditioning(output)

    from sglang.multimodal_gen.runtime.entrypoints import diffusion_generator

    model = args.model.resolve()
    model_config_sha256 = {
        name: hashlib.sha256((model / name).read_bytes()).hexdigest()
        for name in ("model_index.json", "transformer/config.json")
    }
    output.mkdir(parents=True, exist_ok=False)
    os.environ["XLLM_H3_REFERENCE_OUTPUT"] = str(output)
    runtime_kwargs = dict(
        model_path=str(model),
        model_subfolder=".",
        model_id="MiniMax-H3",
        backend="sglang",
        pipeline_class_name="MiniMaxH3Pipeline",
        num_gpus=args.devices,
        tp_size=2,
        sp_degree=args.devices // 2,
        ulysses_degree=args.devices // 2,
        kv_gather_degree=1,
        encoder_parallel="replicate",
        attention_backend="fa",
        dit_cpu_offload=False,
        component_residency={"text_encoder": "layerwise-offload"},
        disable_conditioning_cache=True,
        disable_autocast=True,
        enable_torch_compile=False,
        warmup_mode="off",
        master_port=30185,
        port=30186,
        scheduler_port=5615,
        output_path=str(output),
    )
    sampling_params = {
        "prompt": args.prompt,
        "seed": 42,
        "num_inference_steps": 50,
        "task": "t2va",
        "conditions": [],
        "target": {"short_edge": 768, "aspect_ratio": "16:9", "duration_seconds": 5.0},
        "quality": "exact",
        "save_output": True,
        "output_path": str(output),
        "output_file_name": "reference.mp4",
        "flow_shift": 12.0,
        "audio_flow_shift": 3.0,
    }
    generator = diffusion_generator.DiffGenerator.from_pretrained(**runtime_kwargs)
    try:
        result = generator.generate(sampling_params_kwargs=sampling_params)
        if result is None:
            raise RuntimeError("SGLang H3 generation failed")
        report = {
            "prompt": args.prompt,
            "tp": 2,
            "sp": args.devices // 2,
            "torch": importlib.metadata.version("torch"),
            "torch_npu": importlib.metadata.version("torch_npu"),
            "sglang": sglang_source["version"],
            "model": {"path": str(model), "config_sha256": model_config_sha256},
            "runtime_kwargs": runtime_kwargs,
            "sampling_params": sampling_params,
            "source_sha256": {
                "generate_minimax_h3_reference.py": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "sglang/text_encoding.py": sglang_source["text_encoding_sha256"],
                "sglang/diffusion_generator.py": hashlib.sha256(
                    Path(diffusion_generator.__file__).read_bytes()
                ).hexdigest(),
            },
            "sha256": {
                name: hashlib.sha256((output / name).read_bytes()).hexdigest()
                for name in ("conditioning.pt", "reference.mp4")
            },
        }
        (output / "reference.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report))
    finally:
        generator.shutdown()


if __name__ == "__main__":
    main()

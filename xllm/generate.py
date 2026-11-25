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
# ==============================================================================

from __future__ import annotations

import argparse
import base64
import json
import socket
import sys
import tempfile
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from scripts.logger import logger
from xllm import launch_server as launcher

_EXTENSIONS = {"image": ".png", "video": ".mp4", "audio": ".wav", "text": ".txt"}
_INPUT_FIELDS = ("prompt", "prompt_2", "negative_prompt", "negative_prompt_2")
_PARAMETER_TYPES = {
    "num_inference_steps": int,
    "guidance_scale": float,
    "guidance_scale_2": float,
    "true_cfg_scale": float,
    "seed": int,
    "num_images_per_prompt": int,
    "num_videos_per_prompt": int,
    "max_sequence_length": int,
    "strength": float,
    "num_frames": int,
    "fps": float,
    "seconds": int,
    "flow_shift": float,
    "sampling_rate": int,
    "audio_duration_frames": int,
    "audio_steps": int,
    "audio_guidance_method": str,
    "max_new_tokens": int,
    "diffusion_steps": int,
    "temperature": float,
    "top_k": int,
    "top_p": float,
    "repetition_penalty": float,
}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="xllm generate",
        usage="%(prog)s [MODEL_PATH] [options]",
        allow_abbrev=False,
        description="Generate DiT media locally, save the result, and exit. Runtime options are forwarded to the engine.",
        add_help=False,
    )
    parser.add_argument("-h", "--help", dest="show_help", action="store_true", help="Show generation options and exit.")
    parser.add_argument("--model", "--model-path", "--model_path")
    parser.add_argument("--nnodes", "--num-gpus", type=int, default=None, help="Number of local devices (default: 1).")
    parser.add_argument(
        "--config-json-file", "--config_json_file", default=None, help="Runtime JSON configuration, as for xllm serve."
    )
    parser.add_argument(
        "--port",
        dest="start_port",
        type=int,
        default=None,
        help="Base port used for rank instance names; generate does not start HTTP serving.",
    )
    parser.add_argument("--log-dir", "--log_dir", default="log", help="Directory for per-rank engine logs.")
    parser.add_argument("--no-log-files", dest="log_dir", action="store_const", const=None)
    parser.add_argument("--binary-path", "--binary_path", help="Path to the built xLLM executable.")
    parser.add_argument("--dry-run", action="store_true", help="Print engine commands without loading a model.")
    parser.set_defaults(machine_rank=None)
    parser.add_argument("--modality", choices=tuple(_EXTENSIONS), default="image")
    parser.add_argument(
        "--request-file",
        "--request_file",
        type=Path,
        help="JSON request using the corresponding /v1/{modality}/generation schema.",
    )
    parser.add_argument(
        "--conditioning",
        type=Path,
        help="PyTorch dictionary of prompt, named prompt embeddings and optional prompt_token_tags.",
    )
    parser.add_argument(
        "--output-path",
        "--output_path",
        type=Path,
        default=Path("outputs"),
        help="Directory for generated files (default: outputs).",
    )
    parser.add_argument(
        "--output-file-name",
        "--output_file_name",
        help="Output filename; multiple results receive _0, _1, ... suffixes.",
    )
    parser.add_argument(
        "--output-file-path",
        "--output_file_path",
        type=Path,
        help="Full output file path, overriding directory and filename.",
    )
    for name in _INPUT_FIELDS:
        flags = ["--" + name.replace("_", "-")]
        if "_" in name:
            flags.append("--" + name)
        parser.add_argument(*flags, dest=name)
    parser.add_argument("--size", help="Width*height, for example 1344*768.")
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    for name, value_type in _PARAMETER_TYPES.items():
        flags = ["--" + name.replace("_", "-")]
        if "_" in name:
            flags.append("--" + name)
        parser.add_argument(*flags, dest=name, type=value_type)
    return parser


def _load_conditioning(path: Path, inputs: dict[str, Any]) -> None:
    import torch

    conditioning = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(conditioning, dict):
        raise ValueError("--conditioning must contain a dictionary")
    prompt = conditioning.get("prompt")
    if prompt is not None:
        if not isinstance(prompt, str):
            raise ValueError("conditioning prompt must be a string")
        if "prompt" in inputs and inputs["prompt"] != prompt:
            raise ValueError("prompt differs from the text used to produce the conditioning")
        inputs["prompt"] = prompt
    tensor_names = ("prompt_embed", "negative_prompt_embed", "pooled_prompt_embed", "negative_pooled_prompt_embed")
    for name in tensor_names:
        if name not in conditioning:
            continue
        tensor = conditioning[name]
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(f"conditioning {name} must be a tensor")
        # HTTP requests carry one sample; the engine adds its batch dimension.
        batched_ndim = 2 if "pooled" in name else 3
        if tensor.ndim == batched_ndim and tensor.shape[0] == 1:
            tensor = tensor.squeeze(0)
        tensor = tensor.detach().contiguous()
        if tensor.dtype in (torch.bfloat16, torch.float16):
            datatype = "BF16" if tensor.dtype == torch.bfloat16 else "FP16"
            contents = {"bytes_contents": base64.b64encode(tensor.view(torch.uint8).numpy().tobytes()).decode()}
        else:
            datatype = "FP32"
            contents = {"fp32_contents": tensor.float().flatten().tolist()}
        entry = {"name": name, "datatype": datatype, "shape": list(tensor.shape), "contents": contents}
        if name == "prompt_embed" and "prompt_token_tags" in conditioning:
            tags = conditioning["prompt_token_tags"]
            if not isinstance(tags, torch.Tensor) or tags.dtype != torch.int64 or tags.ndim != 1:
                raise ValueError("prompt_token_tags must be a one-dimensional int64 tensor")
            entry["parameters"] = {"prompt_token_tags": {"int64_tensor_param": tags.tolist()}}
        inputs[name] = entry


def _build_request(args: argparse.Namespace) -> dict[str, Any]:
    request = json.loads(args.request_file.read_text()) if args.request_file else {}
    if not isinstance(request, dict):
        raise ValueError("--request-file must contain a JSON object")
    inputs = request.setdefault("input", {})
    parameters = request.setdefault("parameters", {})
    if not isinstance(inputs, dict) or not isinstance(parameters, dict):
        raise ValueError("request input and parameters must be JSON objects")
    for name in _INPUT_FIELDS:
        value = getattr(args, name)
        if value is not None:
            inputs[name] = value
    if args.conditioning:
        _load_conditioning(args.conditioning, inputs)
    if not inputs.get("prompt") and "prompt_embed" not in inputs:
        raise ValueError("provide --prompt, --conditioning, or --request-file with generation inputs")
    if (args.width is None) != (args.height is None):
        raise ValueError("--width and --height must be supplied together")
    if args.size and args.width is not None:
        raise ValueError("use --size or --width/--height, not both")
    size = args.size
    if args.width is not None:
        size = f"{args.width}*{args.height}"
    if size is not None:
        dimensions = size.split("*")
        if len(dimensions) != 2 or any(not value.isdecimal() or int(value) <= 0 for value in dimensions):
            raise ValueError("--size must be positive WIDTH*HEIGHT")
        parameters["size"] = size
    for name in _PARAMETER_TYPES:
        value = getattr(args, name)
        if value is not None:
            parameters[name] = value
    request.setdefault("model", args.model)
    return request


def _wait_for_generation(processes: Sequence[launcher.ServerProcess]) -> int:
    while True:
        primary_code = processes[0].process.poll()
        if primary_code is not None:
            return primary_code if primary_code >= 0 else 128 - primary_code
        for worker in processes[1:]:
            code = worker.process.poll()
            if code is not None:
                logger.error("Generation worker rank %s exited early with code %s.", worker.rank, code)
                return code if code > 0 else 1
        time.sleep(0.1)


def generate(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and not argv[0].startswith("-"):
        if any(arg.split("=", 1)[0] in ("--model", "--model-path", "--model_path") for arg in argv[1:]):
            parser.error("specify MODEL_PATH or --model-path, not both")
        argv[0:1] = ["--model", argv[0]]
    args, extra = parser.parse_known_args(argv)
    if args.show_help:
        parser.print_help()
        return 0
    config = launcher._load_config_json(parser, args)
    if args.nnodes is None:
        args.nnodes = launcher._read_json_int(parser, config or {}, "nnodes", 1)
    if args.start_port is None:
        args.start_port = launcher._read_json_int(parser, config or {}, "port", 8010)
    launcher._validate_args(parser, args)
    args.model = args.model or (config or {}).get("model")
    if not args.model:
        parser.error("provide MODEL_PATH or --model-path (or model in --config-json-file)")
    # xLLM's native flags use underscores; accept the same hyphen spelling as
    # the generation options while leaving flag values untouched.
    extra = [
        "--" + arg[2:].split("=", 1)[0].replace("-", "_") + ("=" + arg.split("=", 1)[1] if "=" in arg else "")
        if arg.startswith("--")
        else arg
        for arg in extra
    ]
    launcher._reject_managed_flags(parser, extra)
    if any(arg.split("=", 1)[0] == "--machine_rank" for arg in extra):
        parser.error("generate runs on one machine; use --nnodes for its local devices")
    try:
        request = _build_request(args)
        binary = launcher._resolve_binary_path(args.binary_path)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    suffix = _EXTENSIONS[args.modality]
    output = args.output_file_path or args.output_path / (args.output_file_name or "output" + suffix)
    output = output.expanduser().resolve()
    if output.suffix != suffix:
        parser.error(f"--modality {args.modality} writes {suffix} files; output must use that extension")
    runtime = [f"--model={args.model}", *extra]
    runtime_names = {arg.split("=", 1)[0] for arg in runtime}
    if "--host" not in runtime_names and "host" not in (config or {}):
        runtime.append("--host=127.0.0.1")
    if "--master_node_addr" not in runtime_names and "master_node_addr" not in (config or {}):
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            runtime.append(f"--master_node_addr=127.0.0.1:{sock.getsockname()[1]}")
    launcher._ensure_python_model_path()
    with tempfile.TemporaryDirectory(prefix="xllm-generate-") as temporary:
        request_path = Path(temporary) / "request.json"
        request_path.write_text(json.dumps(request), encoding="utf-8")
        commands = [
            launcher._build_command(
                binary,
                args,
                rank,
                args.start_port + rank,
                ["generate", args.modality, str(request_path), str(output), *runtime],
            )
            for rank in range(args.nnodes)
        ]
        for rank, command in enumerate(commands):
            logger.info("rank %s: %s", rank, launcher._format_command(command))
        if args.dry_run:
            return 0
        output.parent.mkdir(parents=True, exist_ok=True)
        launcher._install_signal_handlers()
        processes: list[launcher.ServerProcess] = []
        try:
            for rank, command in enumerate(commands):
                processes.append(launcher._start_process(command, rank, args.log_dir))
            return_code = _wait_for_generation(processes)
            if return_code == 0:
                logger.info("Generation complete; files saved in %s.", output.parent)
            else:
                logger.error("Generation failed with code %s. Engine logs: %s", return_code, args.log_dir or "console")
            return return_code
        except KeyboardInterrupt:
            logger.warning("Interrupted; stopping generation workers.")
            return 130
        finally:
            launcher._terminate_processes(processes)
            launcher._close_logs(processes)


if __name__ == "__main__":
    raise SystemExit(generate())

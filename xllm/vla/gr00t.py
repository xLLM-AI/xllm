# Copyright 2026 The xLLM Authors. SPDX-License-Identifier: Apache-2.0
"""Offline GR00T inference with an xLLM DCU backbone and eager action head.

Gr00tModel accepts processor tensors and returns normalized actions. The DROID
policy also provides observation processing and physical action decoding.
"""

import copy
import json
import os
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from torch.nn.attention import SDPBackend, sdpa_kernel

from xllm.vla.gr00t_action import Gr00tActionConfig, Gr00tActionHead


class Gr00tModel:
    """Single DCU, batch one, serialized inference with an uncached backbone."""

    def __init__(
        self,
        model_path: str | Path,
        backbone_config: str | Path,
        *,
        device: str = "cuda:0",
        dtype: torch.dtype = torch.bfloat16,
        attention_backend: str = "flash",
        action_execution: str = "eager",
    ) -> None:
        from xllm import xllm_export

        self.device = torch.device(device)
        if self.device.type != "cuda" or torch.version.hip is None:
            raise ValueError("The native GR00T backbone currently requires a DCU (cuda device in HIP PyTorch)")
        if self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        if dtype not in (torch.bfloat16, torch.float16, torch.float32):
            raise ValueError("GR00T dtype must be bfloat16, float16 or float32")
        self.dtype = dtype
        if attention_backend not in ("math", "flash") or action_execution not in ("eager", "graph"):
            raise ValueError("GR00T requires math/flash attention and eager/graph execution")
        if attention_backend == "flash" and dtype != torch.bfloat16:
            raise ValueError("The validated GR00T Flash path requires bfloat16")
        if action_execution == "graph" and attention_backend != "flash":
            raise ValueError("GR00T graph execution requires Flash attention")
        self.attention_backend = attention_backend
        self.action_execution = action_execution
        os.environ["XLLM_DCU_ATTN_IMPL"] = "flash" if attention_backend == "flash" else "torch"
        os.environ["XLLM_DCU_VISION_ATTN_IMPL"] = os.environ["XLLM_DCU_ATTN_IMPL"]
        path = Path(model_path)
        config = json.loads((path / "config.json").read_text())
        if config.get("model_type") != "Gr00tN1d7":
            raise ValueError("Expected a Gr00tN1d7 checkpoint")
        self.action_config = Gr00tActionConfig.from_dict(config)
        self.backbone_config = copy.deepcopy(json.loads(Path(backbone_config).read_text()))
        if self.backbone_config.get("model_type") != "qwen3_vl":
            raise ValueError("GR00T requires a Qwen3-VL backbone config")
        text = self.backbone_config["text_config"]
        if "rope_scaling" not in text and "rope_parameters" in text:
            text["rope_scaling"] = text["rope_parameters"]
        select_layer = config["select_layer"]
        if select_layer >= 0:
            text["num_hidden_layers"] = min(text["num_hidden_layers"], select_layer)
        if text["num_hidden_layers"] <= 0:
            raise ValueError("At least one language decoder layer is required")
        if text["hidden_size"] != self.action_config.backbone_embedding_dim:
            raise ValueError("Backbone hidden_size does not match action feature width")
        self.image_token_id = self.backbone_config["image_token_id"]
        self.deepstack_count = len(self.backbone_config["vision_config"]["deepstack_visual_indexes"])
        with torch.cuda.device(self.device):
            self.backbone = xllm_export.Gr00tBackbone(
                json.dumps(self.backbone_config),
                str(self.device),
                str(dtype).removeprefix("torch."),
            )
            # Construct on meta to avoid an extra initialized multi-GB copy.
            with torch.device("meta"):
                self.action_head = Gr00tActionHead(self.action_config, attention_backend=attention_backend)
            self.action_head.to_empty(device=self.device)
            self.action_head.to(dtype=dtype).eval()
            self._load_weights(path)
        self._action_runner = None
        if attention_backend == "flash":
            from xllm.vla.gr00t_execution import _Gr00tActionRunner

            self._action_runner = _Gr00tActionRunner(self.action_head, action_execution)

    def _load_weights(self, path: Path) -> None:
        from transformers import Qwen3VLConfig, Qwen3VLModel

        index = json.loads((path / "model.safetensors.index.json").read_text())["weight_map"]
        # Validate the complete HF checkpoint schema on meta before passing its
        # shards to xLLM's loaders, which intentionally support partial shards.
        with torch.device("meta"):
            schema = Qwen3VLModel(Qwen3VLConfig.from_dict(self.backbone_config))
        expected_backbone = {
            "backbone.model.model." + key: tuple(value.shape) for key, value in schema.state_dict().items()
        }
        actual_backbone = {key for key in index if key.startswith("backbone.model.model.")}
        if actual_backbone != set(expected_backbone):
            raise ValueError(
                f"Backbone weights differ: missing={set(expected_backbone) - actual_backbone}, "
                f"unexpected={actual_backbone - set(expected_backbone)}"
            )
        expected_action = set(self.action_head.state_dict())
        actual_action = {key.removeprefix("action_head.") for key in index if key.startswith("action_head.")}
        if actual_action != expected_action:
            raise ValueError(
                f"Action weights differ: missing={expected_action - actual_action}, unexpected={actual_action - expected_action}"
            )
        unexpected = [
            key
            for key in index
            if not key.startswith(("backbone.model.model.", "action_head.")) and key != "backbone.model.lm_head.weight"
        ]
        if unexpected:
            raise ValueError(f"Unexpected checkpoint weights: {unexpected}")
        loaded: set[str] = set()
        for shard in sorted(set(index.values())):
            if Path(shard).name != shard:
                raise ValueError("Checkpoint shards must be files in the model directory")
            with safe_open(path / shard, framework="pt", device="cpu") as source:
                keys = set(source.keys())
                requested = {key for key, filename in index.items() if filename == shard}
                if not requested.issubset(keys):
                    raise ValueError(f"Shard {shard} lacks indexed weights: {requested - keys}")
                for key in requested & actual_backbone:
                    if tuple(source.get_slice(key).get_shape()) != expected_backbone[key]:
                        raise ValueError(f"Backbone tensor {key} does not match the supplied backbone config")
                backbone_weights = {
                    key: source.get_tensor(key) for key in requested if key.startswith("backbone.model.model.")
                }
                self.backbone.load_weights(backbone_weights)
                action_weights = {
                    key.removeprefix("action_head."): source.get_tensor(key)
                    for key in requested
                    if key.startswith("action_head.")
                }
                self.action_head.load_state_dict(action_weights, strict=False)
                loaded.update(action_weights)
        if loaded != expected_action:
            raise ValueError("Not all action head weights were loaded")

    @torch.inference_mode()
    def backbone_features(
        self,
        input_ids: torch.Tensor,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return [1,N,D] pre-norm features and [1,N] image mask, without padding."""
        if input_ids.ndim != 2 or input_ids.shape[0] != 1 or input_ids.shape[1] == 0:
            raise ValueError("GR00T native inference requires nonempty input_ids with batch size one")
        if input_ids.dtype not in (torch.int32, torch.int64) or image_grid_thw.dtype not in (torch.int32, torch.int64):
            raise ValueError("Token IDs and image grid dimensions must have integer dtype")
        if image_grid_thw.ndim != 2 or image_grid_thw.shape[1] != 3 or image_grid_thw.shape[0] == 0:
            raise ValueError("image_grid_thw must have shape [num_images,3] with at least one image")
        backend = SDPBackend.FLASH_ATTENTION if self.attention_backend == "flash" else SDPBackend.MATH
        with torch.cuda.device(self.device), sdpa_kernel(backend):
            tokens = input_ids[0].to(device=self.device, dtype=torch.int64)
            image_mask = tokens == self.image_token_id
            visual = self.backbone.encode(pixel_values.to(device=self.device, dtype=self.dtype), image_grid_thw)
            if int(image_mask.sum()) != visual.shape[0]:
                raise ValueError("Number of image tokens differs from encoded image features")
            chunks = visual.chunk(1 + self.deepstack_count, dim=-1)
            embeddings = self.backbone.embed_tokens(tokens)
            embeddings[image_mask] = chunks[0].to(self.dtype)
            deep_stacks = []
            for chunk in chunks[1:]:
                deep = torch.zeros_like(embeddings)
                deep[image_mask] = chunk.to(self.dtype)
                deep_stacks.append(deep)
            positions = self.backbone.positions(tokens.cpu().tolist(), image_grid_thw)
            features = self.backbone.forward(tokens, positions, embeddings, deep_stacks)
        return features.unsqueeze(0), image_mask.unsqueeze(0)

    @torch.inference_mode()
    def predict_processed(
        self,
        inputs: dict[str, Any],
        *,
        initial_noise: torch.Tensor | None = None,
        seed: int | None = None,
    ) -> dict[str, torch.Tensor]:
        """Consume checkpoint processor outputs; return normalized action_pred.

        Expected keys: input_ids, attention_mask, pixel_values, image_grid_thw,
        state and embodiment_id. This path requires an unpadded batch of one.
        """
        if inputs["attention_mask"].shape != inputs["input_ids"].shape or not torch.all(inputs["attention_mask"] == 1):
            raise ValueError("Only unpadded observations are currently supported")
        features, images = self.backbone_features(inputs["input_ids"], inputs["pixel_values"], inputs["image_grid_thw"])
        state = inputs["state"].to(device=self.device, dtype=self.dtype)
        ids = inputs["embodiment_id"].to(device=self.device, dtype=torch.int64)
        generator = None if seed is None else torch.Generator(device=self.device).manual_seed(seed)
        arguments = (features, state, ids, images, torch.ones_like(images))
        if self._action_runner is None:
            actions = self.action_head(*arguments, initial_noise=initial_noise, generator=generator)
        else:
            actions = self._action_runner.run(*arguments, initial_noise=initial_noise, generator=generator)
        return {"action_pred": actions, "backbone_features": features}


class Gr00tDroidPolicy:
    """DROID observation-to-action policy; other embodiments use Gr00tModel."""

    def __init__(
        self,
        model_path: str | Path,
        backbone_config: str | Path,
        processor_path: str | Path,
        *,
        device: str = "cuda:0",
        dtype: torch.dtype = torch.bfloat16,
        attention_backend: str = "flash",
        action_execution: str = "eager",
    ) -> None:
        from xllm.vla.gr00t_processing import Gr00tDroidProcessor

        self.processor = Gr00tDroidProcessor(model_path, processor_path)
        self.model = Gr00tModel(
            model_path,
            backbone_config,
            device=device,
            dtype=dtype,
            attention_backend=attention_backend,
            action_execution=action_execution,
        )

    def get_action(self, observation: dict[str, Any], *, seed: int | None = None) -> dict[str, Any]:
        inputs = self.processor.prepare(observation)
        prediction = self.model.predict_processed(inputs, seed=seed)
        actions = prediction["action_pred"].float().cpu().numpy()
        return self.processor.decode(actions, observation["state"])

    def metadata(self) -> dict[str, Any]:
        return self.processor.metadata()

    def reset(self) -> dict[str, Any]:
        # Observation history is supplied explicitly by the caller.
        return {}

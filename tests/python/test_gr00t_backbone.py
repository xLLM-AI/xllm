# Copyright 2026 The xLLM Authors. SPDX-License-Identifier: Apache-2.0
"""Compare the native backbone with a real HF Qwen3-VL model on DCU."""

import json

import pytest
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import Qwen3VLConfig, Qwen3VLModel

from xllm import xllm_export


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@torch.inference_mode()
@sdpa_kernel(SDPBackend.MATH)
def test_native_pre_norm_features(
    dtype: torch.dtype,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not torch.cuda.is_available() or torch.version.hip is None:
        pytest.skip("Native backbone test requires DCU")
    monkeypatch.setenv("XLLM_DCU_ATTN_IMPL", "torch")
    torch.manual_seed(834)
    config = Qwen3VLConfig(
        text_config={
            "vocab_size": 32,
            "hidden_size": 256,
            "intermediate_size": 512,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 128,
            "max_position_embeddings": 64,
            "rope_scaling": {"rope_type": "default", "mrope_section": [24, 20, 20], "mrope_interleaved": True},
        },
        vision_config={
            "depth": 3,
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_heads": 4,
            "out_hidden_size": 256,
            "patch_size": 16,
            "spatial_merge_size": 2,
            "temporal_patch_size": 2,
            "num_position_embeddings": 36,
            "deepstack_visual_indexes": [0, 1],
        },
        image_token_id=10,
        video_token_id=13,
        vision_start_token_id=11,
        vision_end_token_id=12,
    )
    config.text_config._attn_implementation = "sdpa"
    config.vision_config._attn_implementation = "sdpa"
    reference = Qwen3VLModel(config).to(device="cuda:0", dtype=dtype).eval()
    for module in reference.visual.modules():
        if isinstance(module, torch.nn.Conv3d):
            module.to(memory_format=torch.channels_last_3d)
            module.register_forward_pre_hook(
                lambda _module, args: (args[0].contiguous(memory_format=torch.channels_last_3d), *args[1:])
            )
    native_config = config.to_dict()
    native_config["text_config"]["rope_scaling"] = {
        "rope_type": "default",
        "mrope_section": [24, 20, 20],
        "mrope_interleaved": True,
    }
    native = xllm_export.Gr00tBackbone(json.dumps(native_config), "cuda:0", str(dtype).removeprefix("torch."))
    native.load_weights({"backbone.model.model." + key: value for key, value in reference.state_dict().items()})
    ids = torch.tensor([[1, 11, 10, 10, 10, 10, 12, 2]], device="cuda:0")
    grid = torch.tensor([[1, 4, 4]], device="cuda:0")
    pixels = torch.randn(16, 3 * 2 * 16 * 16, device="cuda:0", dtype=dtype)
    captured = []
    handle = reference.language_model.norm.register_forward_pre_hook(
        lambda _module, args: captured.append(args[0].clone())
    )
    reference(
        input_ids=ids,
        attention_mask=torch.ones_like(ids),
        pixel_values=pixels,
        image_grid_thw=grid,
        mm_token_type_ids=(ids == 10).to(torch.int32),
        use_cache=False,
    )
    handle.remove()
    visual = native.encode(pixels, grid)
    chunks = visual.chunk(3, dim=-1)
    embeddings = native.embed_tokens(ids[0])
    mask = ids[0] == 10
    embeddings[mask] = chunks[0]
    deep_stacks = []
    for chunk in chunks[1:]:
        deep = torch.zeros_like(embeddings)
        deep[mask] = chunk
        deep_stacks.append(deep)
    positions = native.positions(ids[0].tolist(), grid)
    actual = native.forward(ids[0], positions, embeddings, deep_stacks).unsqueeze(0)
    tolerance = 0.025 if dtype == torch.bfloat16 else 3e-4
    torch.testing.assert_close(actual, captured[0], rtol=tolerance, atol=tolerance)

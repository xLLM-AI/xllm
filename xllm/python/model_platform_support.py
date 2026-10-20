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

"""Python model support declared by implementation and platform."""

MODEL_PLATFORM_SUPPORT: dict[str, dict[str, bool]] = {
    "phi3": {"cuda": True, "npu": False},
    "glm5_dspark": {"cuda": False, "npu": True},
    "qwen3": {"cuda": True, "npu": True},
    "qwen3_5": {"cuda": True, "npu": True},
    "qwen3_5_mtp": {"cuda": False, "npu": True},
    "qwen3_5_vl": {"cuda": False, "npu": True},
    "qwen3_dflash": {"cuda": False, "npu": True},
    "qwen3_dflash2": {"cuda": False, "npu": True},
    "qwen3_dspark": {"cuda": False, "npu": True},
    "qwen3_vl": {"cuda": False, "npu": True},
    "deepseek_v32": {"cuda": False, "npu": True},
    "deepseek_v4": {"cuda": False, "npu": True},
    "deepseek_v4_dspark": {"cuda": False, "npu": True},
    "glm5_2": {"cuda": False, "npu": True},
    "glm5_2_mtp": {"cuda": False, "npu": True},
    "glm5_next": {"cuda": False, "npu": True},
    "glm5_next_vl": {"cuda": False, "npu": True},
    "glm5_next_mtp": {"cuda": False, "npu": True},
    "deepseek_v32_mtp": {"cuda": False, "npu": True},
    "joyai_asr": {"cuda": False, "npu": True},
}


def is_deepseek_v4_model(config: dict) -> bool:
    """Whether the executor config describes a DeepSeek-V4 model family.

    Mirrors the C++ util::is_deepseek_v4_model_type set (target, DSpark draft,
    and MTP variants) from xllm/core/util/utils.h.
    """
    model_type = config.get("model_type", "")
    return (
        model_type == "deepseek_v4"
        or model_type == "deepseek_v4_dspark"
        or (model_type.startswith("deepseek_v4") and "mtp" in model_type)
    )


# Block-diffusion draft model types (mirrors
# is_block_draft_model_type / is_dspark_draft_model_type in
# xllm/core/framework/model/model_args.h).
_DSPARK_DRAFT_MODEL_TYPES = ("DSparkDraftModel", "glm5_dspark", "deepseek_v4_dspark")
_BLOCK_DRAFT_MODEL_TYPES = (
    "DFlashDraftModel",
    "DFlash2DraftModel",
) + _DSPARK_DRAFT_MODEL_TYPES


def is_block_draft_model(config: dict) -> bool:
    """Whether the config describes a block-diffusion draft (DFlash/DFlash2/DSpark)."""
    return config.get("model_type", "") in _BLOCK_DRAFT_MODEL_TYPES


def is_dspark_draft_model(config: dict) -> bool:
    """Whether the config describes a DSpark block-diffusion draft."""
    return config.get("model_type", "") in _DSPARK_DRAFT_MODEL_TYPES


# Block-diffusion drafts whose context-KV write goes through the shared
# context_projection_weight + K/V cache layout the ContextKvAclGraphRunner
# captures (DSpark Qwen/GLM5 and DFlash2 — see _independent_block_draft in
# executor.py, upstream #2565). DeepSeek-V4 DSpark writes per-layer SWA caches
# via main_proj instead (see DeepseekV4DSparkModel.write_context_kv), so it
# must stay off that runner; DFlash runs eager.
_CONTEXT_KV_DRAFT_MODEL_TYPES = ("DSparkDraftModel", "glm5_dspark", "DFlash2DraftModel")


def uses_context_kv_projection(config: dict) -> bool:
    """Whether the block-diffusion draft supports the shared context-KV projection."""
    return config.get("model_type", "") in _CONTEXT_KV_DRAFT_MODEL_TYPES

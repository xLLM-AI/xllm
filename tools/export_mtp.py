"""
Export MTP layer for multiple model types (DeepSeek-V3, DeepSeek-V3.2, DeepSeek-R1, DeepSeek-V4, GLM4.5, GLM4.7, Qwen3.5, MiMo etc.).
The exported model can be used for speculative decoding.

Usage:
    # DeepSeek V3
    python3 export_mtp.py --input-dir /path/to/DeepSeek-V3 --output-dir /path/to/DeepSeek-V3-mtp

    # DeepSeek V3.2
    python3 export_mtp.py --input-dir /path/to/DeepSeek-V3.2 --output-dir /path/to/DeepSeek-V3.2-mtp

    # DeepSeek R1
    python3 export_mtp.py --input-dir /path/to/DeepSeek-R1 --output-dir /path/to/DeepSeek-R1-mtp

    # DeepSeek V4
    python3 export_mtp.py --input-dir /path/to/DeepSeek-V4 --output-dir /path/to/DeepSeek-V4-mtp

    # GLM4 MoE
    python3 export_mtp.py --input-dir /path/to/GLM-4.5-Air --output-dir /path/to/GLM-4.5-Air-mtp

    # Qwen3.5
    python3 export_mtp.py --input-dir /path/to/Qwen3.5 --output-dir /path/to/Qwen3.5-mtp

    # MiMo
    python3 export_mtp.py --input-dir /path/to/MiMo-7B-Base --output-dir /path/to/MiMo-7B-Base-mtp
"""
# adapted from https://github.com/sgl-project/sglang/blob/main/scripts/export_deepseek_nextn.py
import argparse
import copy
import json
import os
import shutil
import sys
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoConfig
from scripts.logger import logger


LEGACY_LAYER_MTP_MODEL_TYPES = {"deepseek_v3", "deepseek_v32", "glm4_moe", "glm_moe_dsa"}
MIMO_MTP_MODEL_TYPES = {"mimo"}
QWEN3_5_MODEL_TYPES = {"qwen3_5", "qwen3_5_text"}
QWEN3_5_MOE_MODEL_TYPES = {"qwen3_5_moe", "qwen3_5_moe_text"}
QWEN3_5_EXPORT_MODEL_TYPES = QWEN3_5_MODEL_TYPES | QWEN3_5_MOE_MODEL_TYPES
QWEN3_5_MTP_PREFIXES = ("mtp.", "model.mtp.")
QWEN3_5_EMBEDDING_KEYS = (
    "model.language_model.embed_tokens.weight",
    "language_model.model.embed_tokens.weight",
    "model.embed_tokens.weight",
    "embed_tokens.weight",
)
QWEN3_5_LM_HEAD_KEYS = (
    "lm_head.weight",
    "model.lm_head.weight",
    "language_model.lm_head.weight",
    "model.language_model.lm_head.weight",
)
QWEN3_5_REQUIRED_MTP_KEYS = {
    "pre_fc_norm_embedding.weight",
    "pre_fc_norm_hidden.weight",
    "fc.weight",
    "norm.weight",
}


class ConfigView:
    """Small dict-backed config wrapper matching the AutoConfig API used here."""

    def __init__(self, data: dict[str, Any]) -> None:
        self._data = data

    def __getattr__(self, name: str) -> Any:
        if name in self._data:
            return self._data[name]
        raise AttributeError(name)

    def get(self, name: str, default: Any = None) -> Any:
        return self._data.get(name, default)

    def __contains__(self, name: str) -> bool:
        return name in self._data

    def get_nested(self, path: str, default: Any = None) -> Any:
        current: Any = self._data
        for name in path.split("."):
            if not isinstance(current, dict) or name not in current:
                return default
            current = current[name]
        return current

    def has_nested(self, path: str) -> bool:
        sentinel = object()
        return self.get_nested(path, sentinel) is not sentinel

    def to_dict(self) -> dict[str, Any]:
        return copy.deepcopy(self._data)


def load_config(input_dir: str) -> ConfigView:
    """Load config with AutoConfig first, then fall back to raw config.json."""
    try:
        config = AutoConfig.from_pretrained(input_dir, trust_remote_code=True)
        return ConfigView(config.to_dict())
    except Exception as exc:
        config_path = os.path.join(input_dir, "config.json")
        if not os.path.exists(config_path):
            raise

        logger.info(f"AutoConfig failed ({exc}); falling back to {config_path}")
        with open(config_path) as f:
            return ConfigView(json.load(f))


def _is_positive_config_value(value: Any) -> bool:
    if value is None:
        return False
    try:
        return int(value) > 0
    except (TypeError, ValueError):
        return False


def detect_model_type(config: ConfigView) -> str:
    """Detect model type from config."""
    model_type = str(config.get("model_type", "")).lower()
    text_model_type = str(config.get_nested("text_config.model_type", "")).lower()
    architectures = config.get("architectures", [])
    architecture_names = [str(arch).lower() for arch in architectures]
    model_names = [model_type, text_model_type, *architecture_names]

    if "deepseek" in model_type or any("deepseek" in arch for arch in architecture_names):
        if "v4" in model_type or any("v4" in arch for arch in architecture_names):
            return "deepseek_v4"
        if config.has_nested("index_head_dim") or config.has_nested("index_n_heads") or config.has_nested("index_topk"):
            return "deepseek_v32"
        return "deepseek_v3"

    if "glm4" in model_type or any("glm4" in arch for arch in architecture_names):
        if _is_positive_config_value(config.get("n_routed_experts")):
            return "glm4_moe"
        return "glm4"

    if any("mimo" in name for name in model_names):
        return "mimo"

    if any("qwen3_5" in name for name in model_names):
        if any("moe" in name for name in model_names):
            return "qwen3_5_moe"
        if _is_positive_config_value(config.get_nested("text_config.n_routed_experts")):
            return "qwen3_5_moe"
        if _is_positive_config_value(config.get_nested("text_config.num_experts")):
            return "qwen3_5_moe"
        return "qwen3_5"

    if model_type:
        return model_type

    raise ValueError(f"Unable to detect model type from config. model_type={model_type}, architectures={architectures}")


def get_mtp_layer_id(config: ConfigView, model_type: str) -> int:
    """Get MTP layer ID based on model type."""
    num_hidden_layers = config.get("num_hidden_layers", config.get_nested("text_config.num_hidden_layers"))
    if num_hidden_layers is None:
        raise ValueError("'num_hidden_layers' not found in model config.")

    if model_type in LEGACY_LAYER_MTP_MODEL_TYPES:
        return int(num_hidden_layers)

    raise ValueError(f"Unsupported model type for MTP export: {model_type}")


def get_mtp_model_type(model_type: str) -> str:
    """Get the MTP model type name for the output config."""
    mapping = {
        "deepseek_v3": "deepseek_v3_mtp",
        "deepseek_v32": "deepseek_v32_mtp",
        "deepseek_v4": "deepseek_v4_mtp",
        "glm4_moe": "glm4_moe_mtp",
        "glm_moe_dsa": "glm_moe_dsa_mtp",
        "mimo": "mimo_mtp",
        "qwen3_5": "qwen3_5_mtp",
        "qwen3_5_text": "qwen3_5_mtp",
        "qwen3_5_moe": "qwen3_5_moe_mtp",
        "qwen3_5_moe_text": "qwen3_5_moe_mtp",
    }
    return mapping.get(model_type, f"{model_type}_mtp")


def get_mtp_architecture(model_type: str) -> str:
    """Get the architecture name for the output config."""
    mapping = {
        "deepseek_v3": "DeepseekMTPForCausalLM",
        "deepseek_v32": "DeepseekV32MtpForCausalLM",
        "deepseek_v4": "DeepseekV4MtpForCausalLM",
        "glm4_moe": "Glm4MoeMtpForCausalLM",
        "glm_moe_dsa": "GlmMoeDsaMtpForCausalLM",
        "mimo": "MiMoMtpForCausalLM",
        "qwen3_5": "Qwen3_5MtpForCausalLM",
        "qwen3_5_text": "Qwen3_5MtpForCausalLM",
        "qwen3_5_moe": "Qwen3_5MtpForCausalLM",
        "qwen3_5_moe_text": "Qwen3_5MtpForCausalLM",
    }
    return mapping.get(model_type, "MtpForCausalLM")


def get_mtp_layer_count(config: ConfigView, model_type: str) -> int:
    """Get MTP layer count from model-specific config fields."""
    candidate_paths = [
        "num_nextn_predict_layers",
        "mtp_num_hidden_layers",
        "text_config.num_nextn_predict_layers",
        "text_config.mtp_num_hidden_layers",
    ]
    if model_type in QWEN3_5_EXPORT_MODEL_TYPES:
        candidate_paths = [
            "text_config.mtp_num_hidden_layers",
            "mtp_num_hidden_layers",
            "text_config.num_nextn_predict_layers",
            "num_nextn_predict_layers",
        ]

    for path in candidate_paths:
        value = config.get_nested(path)
        if value is not None:
            mtp_layer_count = int(value)
            if mtp_layer_count <= 0:
                raise ValueError(f"MTP layer count from '{path}' must be positive, but found {mtp_layer_count}.")
            return mtp_layer_count

    raise ValueError(
        "Model does not have MTP layer count fields. Expected one of: "
        f"{', '.join(candidate_paths)}. This model may not support MTP."
    )


def _is_dsa_mtp_model(model_type: str) -> bool:
    return model_type in {"deepseek_v32", "glm_moe_dsa"}


def _has_dsa_indexer(config: ConfigView) -> bool:
    return (
        _is_positive_config_value(config.get("index_n_heads"))
        and _is_positive_config_value(config.get("index_head_dim"))
        and _is_positive_config_value(config.get("index_topk"))
    )


def _update_mtp_dsa_topk_config(
    updates: dict[str, Any],
    config: ConfigView,
    model_type: str,
) -> None:
    if not _is_dsa_mtp_model(model_type) or not _has_dsa_indexer(config):
        return

    if not bool(config.get("index_share_for_mtp_iteration", False)):
        return

    index_topk_freq = int(config.get("index_topk_freq", 1) or 1)
    mtp_topk_updates = {
        "index_share_for_mtp_iteration": True,
        "index_topk_freq": max(index_topk_freq, 2),
        "index_skip_topk_offset": 0,
        "index_topk_pattern": "S",
        "indexer_types": ["full"],
    }
    for key, value in mtp_topk_updates.items():
        if key in config:
            updates[key] = value


def update_and_save_config(config: ConfigView, output_dir: str, model_type: str, mtp_layer_count: int) -> None:
    """Update and save config for MTP model."""
    new_config = config.to_dict()
    mtp_model_type = get_mtp_model_type(model_type)
    mtp_architecture = get_mtp_architecture(model_type)

    updates = {
        "num_hidden_layers": mtp_layer_count,
        "num_nextn_predict_layers": mtp_layer_count,
        "architectures": [mtp_architecture],
        "model_type": mtp_model_type,
        "quantization_config": "",
        "first_k_dense_replace": 0,
    }

    if model_type == "deepseek_v4":
        updates["num_hash_layers"] = 0

    if model_type in QWEN3_5_EXPORT_MODEL_TYPES:
        updates["mtp_num_hidden_layers"] = mtp_layer_count

    _update_mtp_dsa_topk_config(updates, config, model_type)

    new_config.update(updates)

    if model_type in QWEN3_5_EXPORT_MODEL_TYPES and isinstance(new_config.get("text_config"), dict):
        new_config["text_config"]["num_hidden_layers"] = mtp_layer_count
        new_config["text_config"]["mtp_num_hidden_layers"] = mtp_layer_count
        new_config["text_config"]["layer_types"] = ["full_attention"] * mtp_layer_count

    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(new_config, f, indent=2, ensure_ascii=False, sort_keys=True)


def copy_non_safetensors_files(input_dir: str, output_dir: str) -> None:
    for filename in os.listdir(input_dir):
        src_file_path = os.path.join(input_dir, filename)
        if os.path.isfile(src_file_path) and not filename.endswith(".safetensors"):
            dst_file_path = os.path.join(output_dir, filename)
            shutil.copy2(src_file_path, dst_file_path)
    logger.info(f"All non-safetensors files have been copied to {output_dir}")


def get_mtp_weight_prefix(config: ConfigView, model_type: str) -> str:
    if model_type == "deepseek_v4":
        return "mtp.0."
    if model_type in MIMO_MTP_MODEL_TYPES:
        # MiMo checkpoint stores MTP weights at "model.mtp_layers.0." (0-indexed).
        return "model.mtp_layers.0."
    return f"model.layers.{get_mtp_layer_id(config, model_type)}"


def map_mtp_key(key: str, prefix: str, model_type: str) -> str | None:
    if key == "rot.weight":
        return "model.rot.weight"

    if not key.startswith(prefix):
        return None

    if model_type == "deepseek_v4":
        return key.replace(prefix, "model.layers.0.", 1)

    if model_type in MIMO_MTP_MODEL_TYPES:
        # MiMo checkpoint stores MTP weights at "model.mtp_layers.0.*".
        # MiMoMtpModelImpl loads directly from "mtp_layers.0.*", so no
        # key remapping is needed — return the original key unchanged.
        return key

    if any(special in key for special in ["embed_tokens", "shared_head", "enorm", "hnorm", "eh_proj"]):
        return key.replace(prefix, "model", 1)
    return key.replace(prefix, "model.layers.0", 1)


def update_quant_model_description(input_dir: str, output_dir: str, prefix: str, model_type: str) -> None:
    filename = "quant_model_description.json"
    input_path = os.path.join(input_dir, filename)
    output_path = os.path.join(output_dir, filename)

    if not os.path.exists(input_path):
        return

    with open(input_path, encoding="utf-8") as f:
        quant_desc = json.load(f)

    updated_desc = {}
    for key, value in quant_desc.items():
        new_key = map_mtp_key(key, prefix, model_type)
        if new_key is not None:
            updated_desc[new_key] = value

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(updated_desc, f, indent=2, ensure_ascii=False, sort_keys=True)

    logger.info(f"Updated {filename} with {len(updated_desc)} MTP entries")


def save_params_and_index(params: dict[str, torch.Tensor], output_dir: str) -> None:
    """Save exported tensors and write a matching safetensors index."""
    output_path = os.path.join(output_dir, "mtp_layer_parameters.safetensors")
    logger.info(f"Saving {len(params)} parameters to {output_path}")
    save_file(params, output_path)

    index_path = os.path.join(output_dir, "model.safetensors.index.json")
    logger.info(f"Updating safetensors index to {index_path}")
    index_data = {"weight_map": {}}
    for key in params:
        index_data["weight_map"][key] = "mtp_layer_parameters.safetensors"
    with open(index_path, "w") as f:
        json.dump(index_data, f, indent=4)

    logger.info("All done.")


def block_dequant(
    x_q_block: torch.Tensor,
    x_s: torch.Tensor,
    block_size: list[int],
) -> torch.Tensor:
    """This function conducts block-wise dequantization.
    The inputs are block-wise quantization tensor `x_q_block`,
    block-wise quantization scale and the block size.
    The outputs are dequantized tensor.
    """
    block_n, block_k = block_size[0], block_size[1]
    n, k = x_q_block.shape
    n_tiles = (n + block_n - 1) // block_n
    k_tiles = (k + block_k - 1) // block_k
    assert n_tiles == x_s.shape[0]
    assert k_tiles == x_s.shape[1]

    x_dq_block = x_q_block.to(torch.float32)

    for i in range(k_tiles):
        for j in range(n_tiles):
            x_dq_block[
                j * block_n:min((j + 1) * block_n, n),
                i * block_k:min((i + 1) * block_k, k),
            ] *= x_s[j][i]

    return x_dq_block.to(torch.bfloat16)


def export_mtp_layer_parameters(input_dir: str, output_dir: str, config: ConfigView, model_type: str) -> None:
    """Export MTP layer parameters for the specified model type."""
    prefix = get_mtp_weight_prefix(config, model_type)
    params = {}

    for filename in os.listdir(input_dir):
        if not filename.endswith(".safetensors"):
            continue

        file_path = os.path.join(input_dir, filename)
        logger.info(f"Processing: {filename}")

        try:
            with safe_open(file_path, framework="pt") as f:
                matching_keys = [k for k in f.keys() if k == "rot.weight" or k.startswith(prefix)]

                if not matching_keys:
                    logger.info(f"  No parameters starting with '{prefix}' found")
                    continue

                for key in matching_keys:
                    new_key = map_mtp_key(key, prefix, model_type)
                    if new_key is None:
                        continue
                    params[new_key] = f.get_tensor(key)
        except Exception:
            logger.exception(f"  Error processing {filename}")
            raise

    if params:
        new_params = {}
        for key, w_tensor in params.items():
            if "weight_scale_inv" in key and model_type in ["deepseek_v3", "deepseek_v32", "deepseek_v4"]:
                weight_scale = w_tensor
                weight_key = key.replace("weight_scale_inv", "weight")
                if weight_key in params:
                    weight = params[weight_key]
                    weight = block_dequant(weight, weight_scale, [128, 128])
                    new_params[weight_key] = weight
            elif key not in new_params:
                new_params[key] = params[key]
        params = new_params
        save_params_and_index(params, output_dir)
    else:
        logger.info("No matching parameters found.")
        raise ValueError(f"No MTP layer parameters found with prefix '{prefix}'")


def _get_qwen3_5_mtp_relative_key(key: str) -> str | None:
    """Return key relative to the qwen3.5 MTP module when applicable."""
    for prefix in QWEN3_5_MTP_PREFIXES:
        if key.startswith(prefix):
            return key[len(prefix):]
    return None


def export_qwen3_5_mtp_parameters(input_dir: str, output_dir: str, config: ConfigView) -> None:
    """Export Qwen3.5 MTP, shared embedding, and LM head parameters."""
    params: dict[str, torch.Tensor] = {}
    mtp_relative_keys: set[str] = set()
    found_embedding = False
    found_lm_head = False

    for filename in os.listdir(input_dir):
        if not filename.endswith(".safetensors"):
            continue

        file_path = os.path.join(input_dir, filename)
        logger.info(f"Processing: {filename}")
        try:
            with safe_open(file_path, framework="pt") as f:
                for key in f.keys():
                    mtp_relative_key = _get_qwen3_5_mtp_relative_key(key)
                    if mtp_relative_key is not None:
                        params[key] = f.get_tensor(key)
                        mtp_relative_keys.add(mtp_relative_key)
                        continue

                    if key in QWEN3_5_EMBEDDING_KEYS:
                        params[key] = f.get_tensor(key)
                        found_embedding = True
                        continue

                    if key in QWEN3_5_LM_HEAD_KEYS:
                        params[key] = f.get_tensor(key)
                        found_lm_head = True
        except Exception:
            logger.exception(f"  Error processing {filename}")
            raise

    if not found_embedding:
        raise ValueError(
            "No Qwen3.5 shared embedding weights found. Expected one of: "
            f"{', '.join(QWEN3_5_EMBEDDING_KEYS)}"
        )

    if not config.get("tie_word_embeddings", False) and not found_lm_head:
        raise ValueError(
            "No Qwen3.5 lm_head weights found for untied embeddings. Expected one of: "
            f"{', '.join(QWEN3_5_LM_HEAD_KEYS)}"
        )

    missing_mtp_keys = sorted(QWEN3_5_REQUIRED_MTP_KEYS - mtp_relative_keys)
    if missing_mtp_keys:
        raise ValueError(f"Missing Qwen3.5 MTP weights: {missing_mtp_keys}")

    if not any(key.startswith("layers.0.") for key in mtp_relative_keys):
        raise ValueError("No Qwen3.5 MTP decoder layer weights found under mtp.layers.0.*")

    if not params:
        raise ValueError("No Qwen3.5 MTP parameters found.")

    save_params_and_index(params, output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export MTP layer parameters for multiple model types (DeepSeek-V3, DeepSeek-V3.2, DeepSeek-R1, DeepSeek-V4, GLM4, Qwen3.5, etc.)"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Input HuggingFace model directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output MTP model directory.",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default=None,
        help="Model type (deepseek_v3, deepseek_v32, deepseek_v4, glm4_moe, glm_moe_dsa, mimo, qwen3_5, qwen3_5_moe). If not specified, will auto-detect. Note: DeepSeek V3 and R1 use 'deepseek_v3', V3.2 uses 'deepseek_v32', V4 uses 'deepseek_v4'.",
    )
    args = parser.parse_args()

    config = load_config(args.input_dir)

    if args.model_type:
        model_type = args.model_type.lower()
    else:
        model_type = detect_model_type(config)

    logger.info(f"Detected model type: {model_type}")

    mtp_layer_count = get_mtp_layer_count(config, model_type)
    if model_type not in QWEN3_5_EXPORT_MODEL_TYPES and mtp_layer_count != 1:
        raise ValueError(f"Only 1 MTP layer is supported, but found {mtp_layer_count}.")

    logger.info(f"MTP layer count: {mtp_layer_count}")

    os.makedirs(args.output_dir, exist_ok=True)
    copy_non_safetensors_files(args.input_dir, args.output_dir)
    update_and_save_config(config, args.output_dir, model_type, mtp_layer_count)

    if model_type in QWEN3_5_EXPORT_MODEL_TYPES:
        export_qwen3_5_mtp_parameters(args.input_dir, args.output_dir, config)
    else:
        prefix = get_mtp_weight_prefix(config, model_type)
        update_quant_model_description(args.input_dir, args.output_dir, prefix, model_type)
        export_mtp_layer_parameters(args.input_dir, args.output_dir, config, model_type)

    logger.info(f"MTP model exported successfully to: {args.output_dir}")

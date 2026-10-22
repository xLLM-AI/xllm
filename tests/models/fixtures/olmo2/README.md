<!-- Copyright 2026 The xLLM Authors. Licensed under the Apache License, Version 2.0. -->

# OLMo2 configuration fixture

`config.json` is the unmodified configuration of
[`allenai/OLMo-2-0425-1B-Instruct`](https://huggingface.co/allenai/OLMo-2-0425-1B-Instruct/blob/48d788eca847d4d7548f375ad03d3c9312f6139e/config.json),
revision `48d788eca847d4d7548f375ad03d3c9312f6139e`. Tests require no weights,
network access, or device initialization. They exercise the production
`load_hf_model_args` entry point and its registered C++ loader.

| HF field | ModelArgs field | Absent/null behavior |
| --- | --- | --- |
| `model_type` | `model_type` | Production dispatch requires a model type; this loader is registered as `olmo2` |
| `hidden_size` | `hidden_size` | Required positive integer |
| `num_hidden_layers` | `n_layers` | Required positive integer |
| `num_attention_heads` | `n_heads` | Required positive integer |
| `intermediate_size` | `intermediate_size` | Required positive integer |
| `vocab_size` | `vocab_size` | Required positive integer |
| `max_position_embeddings` | `max_position_embeddings` | Required positive integer |
| `num_key_value_heads` | `n_kv_heads` | Defaults to query heads |
| `head_dim` | `head_dim` | Derived as hidden size / query heads |
| `hidden_act` | `hidden_act` | `silu` |
| `rms_norm_eps` | `rms_norm_eps` | `1e-5` |
| `rope_theta` | `rope_theta` | `10000` |
| `rope_parameters.rope_theta` | `rope_theta` | Optional newer format; takes precedence over legacy theta |
| `rope_parameters.type`, `rope_parameters.rope_type` | Unscaled RoPE validation | Legacy `type` fallback, canonical `rope_type` takes precedence; missing: `default`; non-default types or a scaling factor are rejected |
| `rope_scaling` | No scaling fields set | Missing/null means unscaled RoPE; non-null is explicitly rejected |
| `torch_dtype`, `dtype` | `dtype` | Empty means unspecified; non-null `dtype` takes precedence |
| `tie_word_embeddings` | `tie_word_embeddings` | `false` |
| `attention_bias` | `attention_bias`, `qkv_bias` | `false` |
| `attention_dropout` | `attention_dropout` | `0` |
| `bos_token_id` | `bos_token_id` | `-1` (absent) |
| `pad_token_id` | `pad_token_id` | Missing: `1`; explicit null: `-1` |
| `eos_token_id` | `eos_token_id`, optional vector, `stop_token_ids` | Missing: `50279`; null: absent; list: all EOS IDs become stop tokens |

Optional defaults follow the pinned Transformers 5.12.1 `Olmo2Config` reference.
Required checkpoint dimensions deliberately have no architecture defaults.
Non-null values must have their documented JSON types. Hidden size must be
divisible by query heads; KV heads must divide query heads; explicit head
dimension must match the derived, even RoPE dimension. Token IDs must fit the
vocabulary and int32 (`-1` represents an absent scalar token). Norm epsilon and
RoPE theta must be finite and positive; dropout is in `[0, 1)`. Explicit dtypes
are `float16`, `bfloat16`, and `float32`.

RoPE type validation includes the legacy `type` alias even when no `factor` is
present. LongRoPE can derive scaling from context lengths without that field;
accepting it as default RoPE would silently lose the checkpoint's semantics.

`use_cache` is a reference execution setting, not an xLLM cache prohibition.
The loader does not register any runtime factory or establish inference support.

Build `olmo2_args_test` in the configured project build and run:

```bash
cmake --build <build-dir> --target olmo2_args_test -j2
ctest --test-dir <build-dir> -R '^Olmo2ArgsTest\.' --output-on-failure
```

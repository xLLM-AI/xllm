/* Copyright 2026 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/jd-opensource/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#pragma once

#include <cstdint>
#include <string>
#include <unordered_set>
#include <vector>

#include "models/llm/npu/qwen3_eagle3.h"

namespace xllm::npu::model {

class KimiK25Eagle3ForCausalLMImpl : public QWen3Eagle3ForCausalLMImpl {
 public:
  explicit KimiK25Eagle3ForCausalLMImpl(const ModelContext& context)
      : QWen3Eagle3ForCausalLMImpl(context) {}
};
TORCH_MODULE(KimiK25Eagle3ForCausalLM);

REGISTER_CAUSAL_MODEL(kimi_k25_eagle3, KimiK25Eagle3ForCausalLM);

REGISTER_MODEL_ARGS(kimi_k25_eagle3, [&] {
  LOAD_ARG_OR(model_type, "model_type", "kimi_k25_eagle3");
  LOAD_ARG_OR_FUNC(dtype, "torch_dtype", [&] {
    return json.value_or<std::string>("dtype", "");
  });
  LOAD_ARG_OR(draft_vocab_size, "draft_vocab_size", 0);
  LOAD_ARG_OR_FUNC(vocab_size, "vocab_size", [&] {
    return args->draft_vocab_size() > 0
               ? args->draft_vocab_size()
               : json.value_or<int64_t>("text_config.vocab_size", 163840);
  });
  LOAD_ARG_OR_FUNC(hidden_size, "hidden_size", [&] {
    return json.value_or<int64_t>("draft_config.hidden_size", 7168);
  });
  LOAD_ARG_OR_FUNC(hidden_act, "hidden_act", [&] {
    return json.value_or<std::string>("draft_config.hidden_act", "silu");
  });
  LOAD_ARG_OR(n_layers, "num_hidden_layers", 1);
  LOAD_ARG_OR_FUNC(n_heads, "num_attention_heads", [&] {
    return json.value_or<int64_t>("draft_config.num_attention_heads", 64);
  });
  LOAD_ARG_OR_FUNC(n_kv_heads, "num_key_value_heads", [&] {
    return json.value_or<int64_t>("draft_config.num_key_value_heads", 64);
  });
  LOAD_ARG_OR_FUNC(intermediate_size, "intermediate_size", [&] {
    return json.value_or<int64_t>("draft_config.intermediate_size", 12288);
  });
  LOAD_ARG_OR_FUNC(max_position_embeddings, "max_position_embeddings", [&] {
    return json.value_or<int64_t>("text_config.max_position_embeddings",
                                  262144);
  });
  LOAD_ARG_OR_FUNC(rms_norm_eps, "rms_norm_eps", [&] {
    return json.value_or<float>("draft_config.rms_norm_eps", 1e-6f);
  });
  LOAD_ARG_OR_FUNC(eos_token_id, "eos_token_id", [&] {
    return json.value_or<int32_t>("text_config.eos_token_id", 163585);
  });
  LOAD_ARG_OR_FUNC(bos_token_id, "bos_token_id", [&] {
    return json.value_or<int32_t>("text_config.bos_token_id", 163584);
  });
  LOAD_ARG_OR_FUNC(rope_theta, "rope_theta", [&] {
    return json.value_or<float>("text_config.rope_theta", 1000000.0f);
  });
  LOAD_ARG_OR_FUNC(tie_word_embeddings, "tie_word_embeddings", [&] {
    return json.value_or<bool>("draft_config.tie_word_embeddings", false);
  });
  LOAD_ARG_OR_FUNC(use_sliding_window, "use_sliding_window", [&] {
    return json.value_or<bool>("draft_config.use_sliding_window", false);
  });
  LOAD_ARG_OR(max_window_layers, "max_window_layers", 36);
  LOAD_ARG_OR_FUNC(head_dim, "head_dim", [&] {
    return json.value_or<int64_t>("draft_config.head_dim", 128);
  });

  SET_ARG(stop_token_ids, std::unordered_set<int32_t>({163585, 163586}));
});

REGISTER_TOKENIZER_ARGS(kimi_k25_eagle3, [&] {
  SET_ARG(tokenizer_type, "tiktoken");
  SET_ARG(vocab_file, "tiktoken.model");

  const std::vector<SpecialToken> special_tokens(
      {{"[BOS]", 163584},
       {"[EOS]", 163585},
       {"<|im_end|>", 163586},
       {"<|im_user|>", 163587},
       {"<|im_assistant|>", 163588},
       {"<|start_header_id|>", 163590},
       {"<|end_header_id|>", 163591},
       {"[EOT]", 163593},
       {"<|im_system|>", 163594},
       {"<|tool_calls_section_begin|>", 163595},
       {"<|tool_calls_section_end|>", 163596},
       {"<|tool_call_begin|>", 163597},
       {"<|tool_call_argument_begin|>", 163598},
       {"<|tool_call_end|>", 163599},
       {"<|im_middle|>", 163601},
       {"<|media_begin|>", 163602},
       {"<|media_content|>", 163603},
       {"<|media_end|>", 163604},
       {"<|media_pad|>", 163605},
       {"<think>", 163606},
       {"</think>", 163607},
       {"[UNK]", 163838},
       {"[PAD]", 163839}});
  SET_ARG(special_tokens, special_tokens);
});

}  // namespace xllm::npu::model

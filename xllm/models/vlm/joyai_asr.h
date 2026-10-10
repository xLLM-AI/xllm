/* Copyright 2026 The xLLM Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/xLLM-AI/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#pragma once

#include "models/model_registry.h"
#include "processors/audio_processor.h"
#include "processors/image_processor.h"
#include "processors/joyai_asr_audio_processor.h"
#include "processors/joyai_asr_prompt_processor.h"
#include "processors/multimodal_processor.h"
#include "processors/video_processor.h"

namespace xllm {

using JoyaiASRMultimodalProcessor = MultimodalProcessor<JoyaiASRPromptProcessor,
                                                        ImageNoneProcessor,
                                                        VideoNoneProcessor,
                                                        JoyaiASRAudioProcessor>;

REGISTER_MULTIMODAL_PROCESSOR(joyai_asr, JoyaiASRMultimodalProcessor);

REGISTER_MODEL_ARGS(joyai_asr, [&] {
  SET_ARG(model_type, "joyai_asr");
  LOAD_ARG_OR(dtype, "torch_dtype", "");
  LOAD_ARG_OR(vocab_size, "vocab_size", 152064);
  LOAD_ARG_OR(hidden_size, "hidden_size", 3584);
  LOAD_ARG_OR(n_layers, "num_hidden_layers", 28);
  LOAD_ARG_OR(n_heads, "num_attention_heads", 28);
  LOAD_ARG(n_kv_heads, "num_key_value_heads");
  LOAD_ARG_OR(hidden_act, "hidden_act", "silu");
  LOAD_ARG_OR(attention_bias, "attention_bias", true);
  LOAD_ARG_OR(intermediate_size, "intermediate_size", 18944);
  LOAD_ARG_OR(max_position_embeddings, "max_position_embeddings", 32768);
  LOAD_ARG_OR(rms_norm_eps, "rms_norm_eps", 1e-6);
  LOAD_ARG_OR(eos_token_id, "eos_token_id", 151645);
  LOAD_ARG_OR(rope_theta, "rope_theta", 1000000.0f);
  LOAD_ARG_OR(tie_word_embeddings, "tie_word_embeddings", false);
  LOAD_ARG_OR_FUNC(head_dim, "head_dim", [&] {
    return args->hidden_size() / args->n_heads();
  });

  LOAD_ARG_OR(audio_token_id, "audio_token_id", 151647);
  LOAD_ARG_OR(mm_audio_ctc_pad_token_id, "ctc_pad_token_id", 151643);
  LOAD_ARG_OR(mm_audio_idim, "audio_encoder_conf.idim", 80);
  LOAD_ARG_OR(mm_audio_d_model, "audio_encoder_conf.d_model", 1280);
  LOAD_ARG_OR(mm_audio_n_layers, "audio_encoder_conf.n_layers_enc", 16);
  LOAD_ARG_OR(mm_audio_n_head, "audio_encoder_conf.n_head", 20);
  LOAD_ARG_OR(mm_audio_kernel_size, "audio_encoder_conf.kernel_size", 33);
  LOAD_ARG_OR(mm_audio_pe_maxlen, "audio_encoder_conf.pe_maxlen", 5000);
  LOAD_ARG_OR(mm_audio_downsample_rate, "encoder_downsample_rate", 2);

  SET_ARG(stop_token_ids, std::unordered_set<int32_t>({args->eos_token_id()}));
});

}  // namespace xllm

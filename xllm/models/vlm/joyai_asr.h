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

#include <algorithm>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "models/model_registry.h"
#include "models/speech_model.h"
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

class JoyaiAsrSpeechAdapter final : public SpeechModelInterface {
 public:
  bool supports_translation() const override { return false; }

  bool supports_verbose_json() const override { return true; }

  bool supports_segment_timestamps() const override { return false; }

  Status validate_language(const std::string& language) const override {
    if (language.empty()) {
      return Status();
    }
    static const std::vector<std::string> supported = {
        "af", "ar", "hy", "az", "be", "bs", "bg", "ca", "zh", "hr", "cs", "da",
        "nl", "en", "et", "fi", "fr", "gl", "de", "el", "he", "hi", "hu", "is",
        "id", "it", "ja", "kn", "kk", "ko", "lv", "lt", "mk", "ms", "mr", "mi",
        "ne", "no", "fa", "pl", "pt", "ro", "ru", "sr", "sk", "sl", "es", "sw",
        "sv", "tl", "ta", "th", "tr", "uk", "ur", "vi", "cy"};
    if (std::find(supported.begin(), supported.end(), language) ==
        supported.end()) {
      return Status(StatusCode::INVALID_ARGUMENT,
                    "Unsupported language: '" + language +
                        "'. Must be an ISO 639-1 code.");
    }
    return Status();
  }

  std::vector<Message> build_messages(
      const SpeechToTextParams& params) const override {
    static const std::string kTaskInstruction =
        "请转写音频为文字，并给出正确的标点符号。";

    std::string context;
    const std::string_view prompts[] = {params.request_prompt, params.hotwords};
    for (const std::string_view prompt : prompts) {
      if (prompt.empty()) {
        continue;
      }
      std::string text(prompt);
      for (size_t pos = 0;;) {
        const size_t start = text.find("<|", pos);
        if (start == std::string::npos) {
          break;
        }
        const size_t end = text.find("|>", start);
        if (end == std::string::npos) {
          break;
        }
        text.erase(start, end - start + 2);
        pos = start;
      }
      std::string sanitized;
      sanitized.reserve(text.size());
      for (const char c : text) {
        const unsigned char byte = static_cast<unsigned char>(c);
        if (byte <= 0x20 || byte == 0x7F) {
          if (!sanitized.empty() && sanitized.back() != ' ') {
            sanitized.push_back(' ');
          }
          continue;
        }
        sanitized.push_back(c);
      }
      if (!sanitized.empty() && sanitized.back() == ' ') {
        sanitized.pop_back();
      }
      if (!sanitized.empty()) {
        if (!context.empty()) {
          context.push_back('\n');
        }
        context.append(sanitized);
      }
    }

    std::vector<Message> messages;
    messages.reserve(1);

    MMContentVec contents;
    contents.reserve(context.empty() ? 2 : 3);
    AudioURL audio_url;
    audio_url.url = "data:" + params.mime_type + ";binary," +
                    std::to_string(params.audio_bytes);
    audio_url.max_duration_s = params.max_duration_s;
    contents.emplace_back("audio_url", audio_url);
    if (!context.empty()) {
      contents.emplace_back("text", "参考下面的相关内容:\n" + context + "\n");
    }
    contents.emplace_back("text", kTaskInstruction);
    messages.emplace_back("user", std::move(contents));
    return messages;
  }
};

REGISTER_SPEECH_MODEL(joyai_asr, JoyaiAsrSpeechAdapter);

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

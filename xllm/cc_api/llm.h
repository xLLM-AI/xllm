/* Copyright 2025-2026 The xLLM Authors.

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

#include <atomic>
#include <optional>
#include <string>

#include "macros.h"
#include "types.h"

namespace xllm {

// Forward declaration
struct LLMCore;

// A wrapper for loading, initializing, and text generation functions of large
// language models
class XLLM_CAPI_EXPORT LLM {
 public:
  LLM();
  virtual ~LLM();

  LLM(const LLM&) = delete;
  LLM& operator=(const LLM&) = delete;

  LLM(LLM&&) noexcept = delete;
  LLM& operator=(LLM&&) noexcept = delete;

  /**
   * @brief Initialize the model: Load model files and configure runtime
   * environment
   * @param model_path Path to model files
   * @param init_options Advanced initialization options, Provided default
   * configuration
   * @return bool true if initialization succeeds; false if fails
   * @note Must be called before Completions/ChatCompletions, and only needs to
   * be called once. The devices to run on are auto-detected from the visible
   * device mask (e.g. ASCEND_RT_VISIBLE_DEVICES / CUDA_VISIBLE_DEVICES).
   */
  bool Initialize(const std::string& model_path,
                  const XLLM_InitLLMOptions& init_options);

  /**
   * @brief Generate completions for the given prompt
   * @param model_id ID of the loaded model
   * @param prompt Input prompt text
   * @param timeout_ms Timeout in milliseconds
   * @param request_params Request parameters (temperature, max tokens, etc.)
   * @return XLLM_Response Response containing generated text and
   * metadata
   */
  XLLM_Response Completions(const std::string& model_id,
                            const std::string& prompt,
                            uint32_t timeout_ms,
                            const XLLM_RequestParams& request_params);

  /**
   * @brief Generates chat completions based on a sequence of conversation
   * messages
   * @param model_id ID of the loaded model
   * @param messages A list of XLLM_ChatMessage objects representing the
   * conversation history, each message contains a role (user/assistant/system)
   * and content text
   * @param timeout_ms Timeout in milliseconds
   * @param request_params Request parameters (temperature, max tokens, etc.)
   * @return XLLM_Response Response containing generated text and
   * metadata
   */
  XLLM_Response ChatCompletions(const std::string& model_id,
                                const std::vector<XLLM_ChatMessage>& messages,
                                uint32_t timeout_ms,
                                const XLLM_RequestParams& request_params);

 private:
  // Opaque pointer to internal LLM core implementation
  LLMCore* llm_core_ = nullptr;
};
}  // namespace xllm

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

#ifndef XLLM_LLM_API_H
#define XLLM_LLM_API_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "types.h"

/**
 * @brief Opaque handle to an LLM inference instance
 *
 * This handle encapsulates all internal state of an LLM inference runtime,
 * including model weights, device context, and generation cache.
 * The handle MUST be created via xllm_llm_create() and destroyed via
 * xllm_llm_destroy() to prevent memory/device resource leaks.
 */
typedef struct XLLM_LLM_Handler XLLM_LLM_Handler;

/**
 * @brief Create a new LLM inference instance handle
 *
 * Allocates memory and initializes a new LLM handler with default internal
 * state (empty model, uninitialized device context). This is the first function
 * that must be called before using any other LLM APIs.
 *
 * @return Valid XLLM_LLM_Handler* on success; NULL if memory allocation fails
 * @see xllm_llm_destroy
 */
XLLM_CAPI_EXPORT XLLM_LLM_Handler* xllm_llm_create(void);

/**
 * @brief Destroy an LLM instance handle and release all associated resources
 *
 * Frees all memory allocated for the LLM instance, including:
 * - Model weights (host/device memory)
 * - Runtime context (CUDA/NPU streams, compute graphs)
 * - Generation cache and temporary buffers
 * - Device resources (contexts, queues)
 *
 * This function is idempotent—calling with NULL has no effect.
 *
 * @param handler LLM instance handle (NULL = no operation)
 * @note Mandatory: Must be called to avoid memory/device resource leaks
 * @see xllm_llm_create
 */
XLLM_CAPI_EXPORT void xllm_llm_destroy(XLLM_LLM_Handler* handler);

/**
 * @brief Initialize XLLM_InitOptions with canonical default values
 *
 * Populates the XLLM_InitOptions struct with industry-standard default values
 *
 * @param init_options Pointer to XLLM_InitOptions to initialize (NULL = no-op)
 * @see xllm_llm_initialize, XLLM_INIT_LLM_OPTIONS_DEFAULT
 */
XLLM_CAPI_EXPORT void xllm_llm_init_options_default(
    XLLM_InitOptions* init_options);

/**
 * @brief Initialize the LLM model and runtime environment
 *
 * Loads model weights from the specified path, configures target devices,
 * initializes compute contexts, and prepares the inference runtime.
 * Must be called exactly once per handler before using completion/chat APIs.
 * The target devices are auto-detected from the visible device mask
 * (e.g. ASCEND_RT_VISIBLE_DEVICES / CUDA_VISIBLE_DEVICES).
 *
 * If init_options is NULL, this function automatically uses the default values
 * from XLLM_INIT_LLM_OPTIONS_DEFAULT (via xllm_llm_init_options_default()).
 *
 * @param handler Valid LLM instance handle (must not be NULL)
 * @param model_path Null-terminated string of the model directory/file path
 *                   (supports .bin/.pth/.safetensors formats)
 * @param init_options Advanced initialization options (NULL = use defaults)
 *
 * @return true if initialization succeeds; false on failure (see failure causes
 * below)
 *
 * @failure_causes
 * - Invalid handler (NULL or already destroyed)
 * - Invalid model_path (non-existent, corrupted, or unsupported format)
 * - Model load error (mismatched model architecture or weight corruption)
 * - Device initialization failure (out of memory, driver error)
 *
 * @see xllm_llm_init_options_default, XLLM_INIT_LLM_OPTIONS_DEFAULT,
 * xllm_llm_create
 */
XLLM_CAPI_EXPORT bool xllm_llm_initialize(XLLM_LLM_Handler* handler,
                                          const char* model_path,
                                          const XLLM_InitOptions* init_options);

/**
 * @brief Initialize XLLM_RequestParams with canonical generation defaults
 *
 * Populates the XLLM_RequestParams struct with safe default generation values
 *
 * @param request_params Pointer to XLLM_RequestParams to initialize (NULL =
 * no-op)
 * @see xllm_llm_completions, xllm_llm_chat_completions,
 * XLLM_LLM_REQUEST_PARAMS_DEFAULT
 */
XLLM_CAPI_EXPORT void xllm_llm_request_params_default(
    XLLM_RequestParams* request_params);

/**
 * @brief Generate text completions for a single prompt
 *
 * Generates continuation text for the input prompt using the initialized LLM
 * model. Returns a dynamically allocated response struct that MUST be freed
 * with xllm_llm_free_response() to avoid memory leaks.
 *
 * If request_params is NULL, this function automatically uses the default
 * values from XLLM_LLM_REQUEST_PARAMS_DEFAULT (via
 * xllm_llm_request_params_default()).
 *
 * @param handler Valid, initialized LLM instance handle (must not be NULL)
 * @param model_id Null-terminated string of the loaded model ID (must match
 * model_path)
 * @param prompt Null-terminated string of input text to complete (non-empty)
 * @param timeout_ms Timeout in milliseconds (0 = no timeout, wait indefinitely)
 * @param request_params Generation parameters (NULL = use defaults)
 *
 * @return Pointer to XLLM_Response on success; NULL ONLY if memory allocation
 * fails (response->status indicates the actual result status)
 *
 * @response_status_codes
 * - kSuccess: Valid response generated (check response->choices for results)
 * - kNotInitialized: Handler not initialized with xllm_llm_initialize()
 * - kInvalidRequest: Invalid prompt (empty/NULL) or model_id (mismatch)
 * - kTimeout: Generation exceeded timeout_ms (partial results may be available)
 *
 * @warning Mandatory: Call xllm_llm_free_response() to release response memory
 * @see xllm_llm_request_params_default, XLLM_LLM_REQUEST_PARAMS_DEFAULT,
 * xllm_llm_free_response
 */
XLLM_CAPI_EXPORT XLLM_Response* xllm_llm_completions(
    XLLM_LLM_Handler* handler,
    const char* model_id,
    const char* prompt,
    uint32_t timeout_ms,
    const XLLM_RequestParams* request_params);

/**
 * @brief Generate chat completions from a conversation history
 *
 * Generates model responses for a multi-turn conversation using chat-formatted
 * message history (user/assistant/system roles). Returns a dynamically
 * allocated response struct that MUST be freed with xllm_llm_free_response().
 *
 * If request_params is NULL, this function automatically uses the default
 * values from XLLM_LLM_REQUEST_PARAMS_DEFAULT (via
 * xllm_llm_request_params_default()).
 *
 * @param handler Valid, initialized LLM instance handle (must not be NULL)
 * @param model_id Null-terminated string of the loaded model ID
 * @param messages Array of XLLM_ChatMessage structs (conversation history)
 * @param messages_count Number of messages in the messages array (must be ≥ 0)
 * @param timeout_ms Timeout in milliseconds (0 = no timeout)
 * @param request_params Generation parameters (NULL = use defaults)
 *
 * @return Pointer to XLLM_Response on success; NULL ONLY if memory allocation
 * fails (response->status indicates the actual result status)
 *
 * @response_status_codes
 * - kSuccess: Valid chat response generated (check
 * response->choices[0].message)
 * - kNotInitialized: Handler not initialized
 * - kInvalidRequest: Invalid messages (NULL with count>0, empty role/content)
 * - kTimeout: Generation exceeded timeout_ms
 *
 * @warning Mandatory: Call xllm_llm_free_response() to release response memory
 * @see xllm_llm_request_params_default, XLLM_LLM_REQUEST_PARAMS_DEFAULT,
 * xllm_llm_free_response
 */
XLLM_CAPI_EXPORT XLLM_Response* xllm_llm_chat_completions(
    XLLM_LLM_Handler* handler,
    const char* model_id,
    const XLLM_ChatMessage* messages,
    size_t messages_count,
    uint32_t timeout_ms,
    const XLLM_RequestParams* request_params);

/**
 * @brief Free all dynamically allocated memory in an XLLM_Response
 *
 * Releases all heap memory used by the response struct
 *
 * After freeing, all fields are reset to safe defaults (NULL/0) to prevent
 * use-after-free.
 *
 * @param resp Pointer to XLLM_Response to free (NULL = no operation)
 *
 * @note Idempotent: Safe to call multiple times on the same response
 * @warning Mandatory: Must be called after using completions/chat completions
 * responses
 * @see xllm_llm_completions, xllm_llm_chat_completions
 */
XLLM_CAPI_EXPORT void xllm_llm_free_response(XLLM_Response* resp);

#ifdef __cplusplus
}
#endif

#endif  // XLLM_LLM_API_H
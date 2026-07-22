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
#ifndef XLLM_REC_API_H
#define XLLM_REC_API_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "types.h"

/**
 * @brief Opaque handle to a Generative Recommendation (REC) inference instance
 * This handle encapsulates all internal state of a REC-specialized runtime,
 * including:
 * - Generative recommendation model weights (item embedding, ranking head)
 * - Device context (CUDA/NPU streams for batch inference)
 * - Generation cache (user behavior context, item candidate pool)
 * - Runtime config (recommendation-specific decoding strategy)
 * The handle MUST be created via xllm_rec_create() and destroyed via
 * xllm_rec_destroy() to prevent memory/device resource leaks.
 */
typedef struct XLLM_REC_Handler XLLM_REC_Handler;

/**
 * @brief Create a new Generative Recommendation (REC) inference instance handle
 * This is the first function that must be called before using any other REC
 * APIs.
 * @return Valid XLLM_REC_Handler* on success; NULL if memory allocation fails
 * @see xllm_rec_destroy
 */
XLLM_CAPI_EXPORT XLLM_REC_Handler* xllm_rec_create(void);

/**
 * @brief Destroy a Generative Recommendation (REC) inference instance handle
 * and release resources Frees all memory allocated for the REC instance,
 * including:
 * - Model weights (host/device memory for item embedding and ranking head)
 * - Runtime context (CUDA/NPU streams, compute graphs for batch recommendation)
 * - Generation cache (user behavior sequence, item candidate pool, attention
 * cache)
 * - Device resources (contexts, queues, memory pools for batch inference)
 * This function is idempotent—calling with NULL has no effect.
 * @param handler REC inference instance handle (NULL = no operation)
 * @note Mandatory: Must be called to avoid memory/device resource leaks
 * @see xllm_rec_create
 */
XLLM_CAPI_EXPORT void xllm_rec_destroy(XLLM_REC_Handler* handler);

/**
 * @brief Helper to initialize XLLM_InitOptions with REC default values
 * Copies the predefined XLLM_INIT_REC_OPTIONS_DEFAULT values into the target
 * init_options struct. Convenient alternative to manually setting each field,
 * ensuring consistency with REC best practices.
 * @param init_options Pointer to XLLM_InitOptions to initialize (NULL = no-op)
 * @see XLLM_INIT_REC_OPTIONS_DEFAULT, xllm_rec_initialize
 */
XLLM_CAPI_EXPORT void xllm_rec_init_options_default(
    XLLM_InitOptions* init_options);

/**
 * @brief Initialize the Generative Recommendation (REC) model and runtime
 * environment Loads generative recommendation model weights from the specified
 * path, configures target devices, initializes compute contexts, and prepares
 * the recommendation inference runtime. The target devices are auto-detected
 * from the visible device mask (e.g. ASCEND_RT_VISIBLE_DEVICES /
 * CUDA_VISIBLE_DEVICES).
 * @param handler Valid REC inference instance handle (must not be NULL)
 * @param model_path Null-terminated string of the REC model directory/file path
 *                   (supports .bin/.pth/.safetensors formats with ranking head)
 * @param init_options Advanced initialization options (NULL = use REC defaults)
 * @return true if initialization succeeds; false on failure (see failure causes
 * below)
 * @par Failure Causes
 * - Invalid handler (NULL or already destroyed)
 * - Invalid model_path (non-existent, corrupted, or missing ranking head
 * weights)
 * - Model load error (mismatched REC model architecture or embedding table
 * corruption)
 * - Device initialization failure (out of memory, driver error, insufficient
 * batch size)
 * @see xllm_rec_init_options_default, XLLM_INIT_REC_OPTIONS_DEFAULT,
 * xllm_rec_create
 */
XLLM_CAPI_EXPORT bool xllm_rec_initialize(XLLM_REC_Handler* handler,
                                          const char* model_path,
                                          const XLLM_InitOptions* init_options);

/**
 * @brief Helper to initialize XLLM_RequestParams with REC default values
 * Copies the predefined XLLM_REC_REQUEST_PARAMS_DEFAULT values into the target
 * request_params struct.
 * @param request_params Pointer to XLLM_RequestParams to initialize (NULL =
 * no-op)
 * @see XLLM_REC_REQUEST_PARAMS_DEFAULT, xllm_rec_text_completions,
 * xllm_rec_token_completions, xllm_rec_chat_completions
 */
XLLM_CAPI_EXPORT void xllm_rec_request_params_default(
    XLLM_RequestParams* request_params);

/**
 * @brief Generate generative recommendation text completions for a user prompt
 * Generates recommendation-focused continuation text for the input user prompt
 * using the initialized REC model
 * @param handler Valid, initialized REC inference instance handle (must not be
 * NULL)
 * @param model_id Null-terminated string of the loaded REC model ID (must match
 * model_path)
 * @param prompt Null-terminated string of user input prompt (non-empty,
 * recommendation-focused)
 * @param timeout_ms Timeout in milliseconds (0 = no timeout, wait indefinitely)
 * @param request_params Generation parameters (NULL = use REC defaults)
 * @return Pointer to XLLM_Response on success; NULL ONLY if memory allocation
 * fails (response->status indicates the actual result status)
 * @par Response Status Codes
 * - kSuccess: Valid recommendation response generated (check response->choices
 * for item list + explanations)
 * - kNotInitialized: Handler not initialized with xllm_rec_initialize()
 * - kInvalidRequest: Invalid prompt (empty/NULL) or model_id (mismatch)
 * - kTimeout: Generation exceeded timeout_ms (partial recommendation results
 * may be available)
 * @warning Mandatory: Call xllm_rec_free_response() to release response memory
 * @see xllm_rec_request_params_default, XLLM_REC_REQUEST_PARAMS_DEFAULT,
 * xllm_rec_free_response
 */
XLLM_CAPI_EXPORT XLLM_Response* xllm_rec_text_completions(
    XLLM_REC_Handler* handler,
    const char* model_id,
    const char* prompt,
    uint32_t timeout_ms,
    const XLLM_RequestParams* request_params);

/**
 * @brief Generate generative recommendation completions for tokenized input
 * (TOKEN ID INPUT) Generates recommendation results from pre-tokenized user
 * input (bypasses the REC model's tokenizer)
 *
 * @param handler Valid, initialized REC inference instance handle (must not be
 * NULL) Created via xllm_rec_create() and initialized via xllm_rec_initialize()
 * @param model_id Null-terminated string of the loaded REC model ID (must match
 *                the model_path used in xllm_rec_initialize())
 * @param token_ids Pointer to int32_t array of pre-tokenized input IDs (NULL
 * only if token_size = 0) Token IDs must be compatible with the REC model's
 * tokenizer vocabulary (e.g., GPT-2/BERT token IDs for text-based REC models)
 * @param token_size Number of tokens in the token_ids array (must be ≥ 0)
 *                Valid ranges: 1 ≤ token_size ≤ xxx (model-dependent max input
 * length) token_size = 0 will return kInvalidRequest status
 * @param timeout_ms Timeout in milliseconds (0 = no timeout, wait indefinitely)
 * @param request_params Generation parameters (NULL = use REC defaults)
 *
 * @return Pointer to XLLM_Response on success; NULL ONLY if memory allocation
 * fails (response->status indicates the actual result status, even if non-NULL)
 *
 * @par Response Status Codes (XLLM_StatusCode)
 * - kSuccess: Valid recommendation response generated
 *             Check response->choices for recommended item list and explanation
 * text
 * - kNotInitialized: Handler not initialized with xllm_rec_initialize()
 * - kModelNotFound: model_id does not match any loaded REC model
 * - kInvalidRequest:
 *   - token_ids = NULL and token_size > 0 (invalid null pointer with non-zero
 * size)
 *   - token_size = 0 (empty token input)
 *   - token_ids contain invalid IDs (out of vocabulary range)
 *   - model_id is NULL/empty/mismatch
 * - kTimeout: Generation exceeded timeout_ms
 * - kInternalError: Internal REC runtime error (e.g., token embedding failure,
 * item retrieval error)

 * @warning Mandatory: Call xllm_rec_free_response() to release response memory
 * @note 1. Token IDs must be generated using the SAME tokenizer as the REC
 * model (e.g., same vocab.txt)
 *       2. Invalid token IDs (e.g., < 0 or > vocab_size) will trigger
 * kInvalidRequest or kInternalError
 *       3. For token_size > model's max input length, the input will be
 * truncated to max length
 * @see xllm_rec_request_params_default, XLLM_REC_REQUEST_PARAMS_DEFAULT,
 * xllm_rec_free_response
 */
XLLM_CAPI_EXPORT XLLM_Response* xllm_rec_token_completions(
    XLLM_REC_Handler* handler,
    const char* model_id,
    const int32_t* token_ids,
    size_t token_size,
    uint32_t timeout_ms,
    const XLLM_RequestParams* request_params);

/**
 * @brief Generate generative recommendation completions for multimodal input
 * (TOKEN ID + MULTIMODAL DATA INPUT)
 * @details Generates recommendation results from pre-tokenized text input
 * (MANDATORY) supplemented with multimodal data that replaces/augments
 * information for specific tokens in the token_ids array. This API extends
 * xllm_rec_token_completions to support multi-modal recommendation scenarios
 * where partial text tokens are enriched with image/audio/video/embedding
 * features (e.g., replacing product text tokens with image embeddings).
 *
 * @param handler Valid, initialized REC inference instance handle (must not be
 * NULL) Created via xllm_rec_create() and initialized via xllm_rec_initialize()
 * @param model_id Null-terminated string of the loaded REC model ID (must match
 *                the model_path used in xllm_rec_initialize())
 *                Must be a multi-modal REC model (text-only models will return
 * kInvalidRequest)
 * @param token_ids Pointer to int32_t array of pre-tokenized text input IDs
 * (MUST NOT be NULL) Token IDs must be compatible with the REC model's
 * tokenizer vocabulary This is the core input and cannot be empty (token_size >
 * 0 required)
 * @param token_size Number of tokens in the token_ids array (MUST be ≥ 1)
 *                  Valid ranges: 1 ≤ token_size ≤ model-dependent max input
 * length token_size = 0 will return kInvalidRequest status (core text input
 * required)
 * @param mm_data Pointer to multi-modal data container (XLLM_MM_Data) (NULL =
 * no multimodal augmentation) Used to replace/augment information for specific
 * tokens in token_ids (via XLLM_MM_TokenPos) Supports
 * image/audio/video/embedding modalities (see XLLM_MM_Type) Must be valid
 * (mm_data->type_mask != XLLM_MM_TYPE_NONE) if non-NULL, and token positions in
 * mm_data must be within [0, token_size-1] (out-of-range positions trigger
 * kInvalidRequest)
 * @param timeout_ms Timeout in milliseconds (0 = no timeout, wait indefinitely)
 * @param request_params Generation parameters (NULL = use REC defaults)
 *                       See XLLM_RequestParams for configurable options (e.g.,
 * top_k, top_p)
 * @return Pointer to XLLM_Response on success; NULL ONLY if memory allocation
 *         fails (response->status indicates the actual result status, even if
 * non-NULL)
 * @par Response Status Codes (XLLM_StatusCode)
 * - kSuccess: Valid multi-modal recommendation response generated
 *             Check response->choices for recommended item list and explanation
 * text Multimodal data has been applied to augment/replace specified tokens
 * - kNotInitialized: Handler not initialized with xllm_rec_initialize()
 * - kModelNotFound: model_id does not match any loaded REC model
 * - kInvalidRequest:
 *   - token_ids = NULL (core text input is mandatory)
 *   - token_size = 0 (empty core text input)
 *   - token_ids contain invalid IDs (out of vocabulary range)
 *   - model_id is NULL/empty/mismatch or is a text-only model
 *   - mm_data is non-NULL but invalid:
 *     - mm_data->type_mask = XLLM_MM_TYPE_NONE (empty multimodal data)
 *     - token positions in mm_data (XLLM_MM_TokenPos) are out of [0,
 * token_size-1] range
 *     - mismatched tensor types/shape in mm_data (e.g., embedding dim mismatch)
 * - kTimeout: Generation exceeded timeout_ms
 * - kInternalError: Internal REC runtime error (e.g., multimodal embedding
 * fusion failure, token augmentation/replacement error, item retrieval error)
 * @warning Mandatory: Call xllm_rec_free_response() to release response memory
 *          Failing to free will cause memory leaks
 * @note 1. Token IDs must be generated using the SAME tokenizer as the REC
 * model (e.g., same vocab.txt)
 *       2. Invalid token IDs (e.g., < 0 or > vocab_size) will trigger
 * kInvalidRequest or kInternalError
 *       3. For token_size > model's max input length, the input will be
 * truncated to max length
 *       4. mm_data is used to replace/augment specific tokens (via
 * XLLM_MM_TokenPos.offset/length):
 *          - offset: start index of tokens in token_ids to be
 * augmented/replaced
 *          - length: number of consecutive tokens to apply multimodal data to
 *       5. If mm_data is NULL, this API behaves identically to
 * xllm_rec_token_completions (text-only inference)
 *       6. Multimodal data must be aligned with token positions (offset +
 * length ≤ token_size)
 * @see xllm_rec_token_completions, xllm_rec_request_params_default,
 *      XLLM_REC_REQUEST_PARAMS_DEFAULT, xllm_rec_free_response, XLLM_MM_Data,
 * XLLM_MM_TokenPos
 */
XLLM_CAPI_EXPORT XLLM_Response* xllm_rec_multimodal_completions(
    XLLM_REC_Handler* handler,
    const char* model_id,
    const int32_t* token_ids,
    size_t token_size,
    const XLLM_MM_Data* mm_data,
    uint32_t timeout_ms,
    const XLLM_RequestParams* request_params);

/**
 * @brief Generate generative recommendation chat completions from multi-turn
 * conversation history Generates personalized recommendation responses for a
 * multi-turn user-assistant conversation
 * @param handler Valid, initialized REC inference instance handle (must not be
 * NULL)
 * @param model_id Null-terminated string of the loaded REC model ID
 * @param messages Array of XLLM_ChatMessage structs (recommendation-focused
 * conversation history)
 * @param messages_count Number of messages in the messages array (must be ≥ 0)
 * @param timeout_ms Timeout in milliseconds (0 = no timeout, wait indefinitely)
 * @param request_params Generation parameters (NULL = use REC defaults)
 * @return Pointer to XLLM_Response on success; NULL ONLY if memory allocation
 * fails (response->status indicates the actual result status)
 * @par Response Status Codes
 * - kSuccess: Valid chat recommendation response generated (check
 * response->choices[0].message for item list)
 * - kNotInitialized: Handler not initialized with xllm_rec_initialize()
 * - kInvalidRequest: Invalid messages (NULL with count>0, empty role/content,
 * non-recommendation context)
 * - kTimeout: Generation exceeded timeout_ms
 * @warning Mandatory: Call xllm_rec_free_response() to release response memory
 * @see xllm_rec_request_params_default, XLLM_REC_REQUEST_PARAMS_DEFAULT,
 * xllm_rec_free_response
 */
XLLM_CAPI_EXPORT XLLM_Response* xllm_rec_chat_completions(
    XLLM_REC_Handler* handler,
    const char* model_id,
    const XLLM_ChatMessage* messages,
    size_t messages_count,
    uint32_t timeout_ms,
    const XLLM_RequestParams* request_params);

/**
 * @brief Free all dynamically allocated memory in a generative recommendation
 * XLLM_Response Releases all heap memory used by the REC response struct
 * @param resp Pointer to XLLM_Response to free (NULL = no operation)
 * @warning Mandatory: Must be called after using REC completions/chat
 * completions responses
 * @see xllm_rec_text_completions, xllm_rec_token_completions,
 * xllm_rec_chat_completions
 */
XLLM_CAPI_EXPORT void xllm_rec_free_response(XLLM_Response* resp);

#ifdef __cplusplus
}
#endif

#endif  // XLLM_REC_API_H
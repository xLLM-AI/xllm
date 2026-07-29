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

#ifndef XLLM_C_TYPES_H
#define XLLM_C_TYPES_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

// Maximum length for meta string fields (includes '\0' terminator)
#define XLLM_META_STRING_FIELD_MAX_LEN 128

// Major version of the public C ABI. A mismatch requires recompilation.
#define XLLM_C_ABI_VERSION_MAJOR 1

// Export Macro Definition
#ifndef XLLM_CAPI_EXPORT
#define XLLM_CAPI_EXPORT __attribute__((visibility("default")))
#endif

// Core Struct & Enum Definitions

/**
 * @brief Configuration options for initializing an LLM instance
 * @note All string fields are fixed-length arrays. Default values are defined
 * in macros. Empty string indicates disable/use default value.
 */
typedef struct XLLM_CAPI_EXPORT XLLM_InitOptions {
  /** Whether to enable chunked prefill for inference */
  bool enable_chunked_prefill;

  /** Context parallel size. Backend and model support is required. */
  uint32_t cp_size;

  /** Whether to enable prefix cache optimization */
  bool enable_prefix_cache;

  /** Whether to enable disaggregated prefill and decode execution */
  bool enable_disagg_pd;

  /** Whether to enable online-offline co-location in disaggregated PD mode */
  bool enable_pd_ooc;

  /** Whether to enable schedule overlap for parallel execution */
  bool enable_schedule_overlap;

  /** Whether to enable shared memory for model execution */
  bool enable_shm;

  /** Whether to enable graph execution for REC */
  bool enable_graph;

  /** Whether to enable REC fast sampler */
  bool enable_rec_fast_sampler;

  /** Whether to enable prefill piecewise graph for REC */
  bool enable_prefill_piecewise_graph;

  /** Whether to enable xattention one-stage execution for REC */
  bool enable_xattention_one_stage;

  /** Whether to enable graph-mode decode without padding for REC */
  bool enable_graph_mode_decode_no_padding;

  /** Whether to enable block copy kernel */
  bool enable_block_copy_kernel;

  /** Whether to keep REC top-k outputs sorted */
  bool enable_topk_sorted;

  /** Whether to enable rec prefill only */
  bool enable_rec_prefill_only;

  /** KVCache transfer listen port */
  uint32_t transfer_listen_port;

  /** Number of multi-nodes in distributed deployment */
  uint32_t nnodes;

  /** Node rank in distributed deployment */
  uint32_t node_rank;

  /** Data parallel size for MLA attention */
  uint32_t dp_size;

  /** Expert parallel size for MoE model */
  uint32_t ep_size;

  /** Number of slots per kv cache block */
  uint32_t block_size;

  /** Max GPU memory size for kv cache (0 = auto-calculate available memory) */
  uint32_t max_cache_size;

  /** Max number of tokens per batch */
  uint32_t max_tokens_per_batch;

  /** Max number of sequences per batch */
  uint32_t max_seqs_per_batch;

  /** Max number of token per chunk in prefill stage */
  uint32_t max_tokens_per_chunk_for_prefill;

  /** Number of speculative tokens for speculative decoding */
  uint32_t num_speculative_tokens;

  /** Number of threads for handling input requests */
  uint32_t num_request_handling_threads;

  /** Expert parallel degree for MoE model */
  uint32_t expert_parallel_degree;

  /** Index ID for internal server ID (unique for multiple models/versions) */
  uint32_t server_idx;

  /** Beam width for beam search decoding (1 for greedy search) */
  uint32_t beam_width;

  /** Maximum number of decode rounds for each inference request */
  uint32_t max_decode_rounds;

  /** Maximum number of tokens allowed per inference request */
  uint32_t max_token_per_req;

  /** Maximum GPU memory utilization ratio for model inference */
  float max_memory_utilization;

  /** Maximum REC worker pipeline concurrency */
  uint32_t rec_worker_max_concurrency;

  /** FlashInfer attention workspace buffer size in bytes. */
  uint32_t flashinfer_workspace_buffer_size;

  /** Maximum number of requests in the request queue */
  uint32_t request_queue_size;

  /** Model task type (generate/embed) */
  char task[XLLM_META_STRING_FIELD_MAX_LEN];

  /** NPU communication backend (lccl/hccl). Use hccl when dp is enabled */
  char communication_backend[XLLM_META_STRING_FIELD_MAX_LEN];

  /** Instance role (DEFAULT/PREFILL/DECODE/MIX) */
  char instance_role[XLLM_META_STRING_FIELD_MAX_LEN];

  /** Master address for multi-node distributed serving (e.g. 10.18.1.1:9999) */
  char master_node_addr[XLLM_META_STRING_FIELD_MAX_LEN];

  /** XService server address (empty string = disable XService) */
  char xservice_addr[XLLM_META_STRING_FIELD_MAX_LEN];

  /** Unique instance name for identification */
  char instance_name[XLLM_META_STRING_FIELD_MAX_LEN];

  /** KV cache transfer mode (PUSH/PULL) */
  char kv_cache_transfer_mode[XLLM_META_STRING_FIELD_MAX_LEN];

  /** Log directory path (empty string = disable logging) */
  char log_dir[XLLM_META_STRING_FIELD_MAX_LEN];

  /** Draft hf model path (empty string = no draft model) */
  char draft_model[XLLM_META_STRING_FIELD_MAX_LEN];

  // CPU affinity settings, format be like:
  // 1,2,3
  // meaing ith thread attach to logical processor 1,2,3
  // or
  // 1-8
  // meaning ith thread will be attached to first 8 logical processors
  char cpu_affinity[XLLM_META_STRING_FIELD_MAX_LEN];
} XLLM_InitLLMOptions;

/**
 * @brief Chat message structure (for ChatCompletions)
 */
typedef struct XLLM_CAPI_EXPORT XLLM_ChatMessage {
  /** Message role (system/user/assistant) */
  char role[XLLM_META_STRING_FIELD_MAX_LEN];

  /** Message content (NULL for function call messages) */
  char* content;
} XLLM_ChatMessage;

/**
 * @brief Inference request parameters
 * @note All numeric fields are fixed-width integers with value ranges defined
 * in macros;
 */
typedef struct XLLM_CAPI_EXPORT XLLM_RequestParams {
  /** Whether to include original prompt in response */
  bool echo;

  /** Whether it's an offline request */
  bool offline;

  /** Whether to return token log probabilities */
  bool logprobs;

  /** Whether to ignore EOS token */
  bool ignore_eos;

  /** Number of completions to return per prompt */
  uint32_t n;

  /** Maximum number of tokens to generate. Must be <= model context length */
  uint32_t max_tokens;

  /** Number of sequences to generate per prompt for top-n selection */
  uint32_t best_of;

  /** SLO timeout in milliseconds (0 = unlimited) */
  int32_t ttlt_slo_ms;

  int32_t ttft_slo_ms;

  int32_t tpot_slo_ms;

  /** Beam search width (0 = disable beam search) */
  uint32_t beam_width;

  /** Final number of beam search results to return (0 = use beam_width) */
  uint32_t num_return_sequences;

  /** Number of top log probabilities to return */
  int64_t top_logprobs;

  /** Top-K sampling cutoff (-1 = 0xFFFFFFFF means disabled) */
  int64_t top_k;

  /** Top-P sampling cutoff (range: [0.0, 1.0]) */
  float top_p;

  /** Frequency penalty (range: [0.0, 2.0]) */
  float frequency_penalty;

  /** Presence penalty (range: [-2.0, 2.0]) */
  float presence_penalty;

  /** Repetition penalty. >1.0 encourages new tokens, <1.0 encourages repetition
   */
  float repetition_penalty;

  /** Sampling temperature (range: [0.0, 2.0]) */
  float temperature;

  /** Request id */
  char request_id[XLLM_META_STRING_FIELD_MAX_LEN];
} XLLM_RequestParams;

/**
 * @brief API response status codes
 */
typedef enum XLLM_CAPI_EXPORT XLLM_StatusCode {
  /** Request succeeded */
  kSuccess = 0,

  /** LLM instance not initialized */
  kNotInitialized = 1,

  /** Specified model ID not loaded */
  kModelNotFound = 2,

  /** Request timed out */
  kTimeout = 3,

  /** Invalid input parameters */
  kInvalidRequest = 4,

  /** Internal system error */
  kInternalError = 5
} XLLM_StatusCode;

/**
 * @brief Token usage statistics for inference request
 */
typedef struct XLLM_CAPI_EXPORT XLLM_Usage {
  /** Number of tokens in the prompt */
  int32_t prompt_tokens;

  /** Number of tokens in the generated completion */
  int32_t completion_tokens;

  /** Total tokens used (prompt + completion) */
  int32_t total_tokens;

  /** Number of prompt tokens served from prefix cache */
  int32_t cached_tokens;
} XLLM_Usage;

/**
 * @brief Token log probability structure
 */
typedef struct XLLM_CAPI_EXPORT XLLM_LogProb {
  /** Token ID */
  uint32_t token_id;

  /** Log probability of the token */
  float logprob;
} XLLM_LogProb;

/**
 * @brief List of token log probabilities
 */
typedef struct XLLM_CAPI_EXPORT XLLM_LogProbs {
  /** Pointer to array of log probability entries */
  XLLM_LogProb* entries;

  /** Number of entries in the logprobs array */
  size_t entries_size;
} XLLM_LogProbs;

/**
 * @brief Inference result candidate
 */
typedef struct XLLM_CAPI_EXPORT XLLM_Choice {
  /** Index of the generated completion candidate */
  uint32_t index;

  /** Generated text for completions inference (NULL for Chat mode) */
  char* text;

  /** Generated message for chatcompletions inference (NULL for Completion mode)
   */
  XLLM_ChatMessage* message;

  /** Generated token ids */
  int32_t* token_ids;

  /** Generated token ids size */
  size_t token_size;

  /** Token log probabilities */
  XLLM_LogProbs logprobs;

  /** Reason generation stopped (stop/length/function_call) */
  char finish_reason[XLLM_META_STRING_FIELD_MAX_LEN];
} XLLM_Choice;

/**
 * @brief List of inference result candidates
 */
typedef struct XLLM_CAPI_EXPORT XLLM_Choices {
  /** Pointer to array of completion choice entries */
  XLLM_Choice* entries;

  /** Number of entries in the choices array */
  size_t entries_size;
} XLLM_Choices;

/**
 * @brief REC/OneRec specific output extension aligned by choice index
 */
typedef struct XLLM_CAPI_EXPORT XLLM_RecOutput {
  /** Choice index this REC extension belongs to */
  uint32_t index;

  /** Selected REC item ids for this choice */
  int64_t* item_ids;

  /** Number of item ids in the item_ids array */
  size_t item_ids_size;

  /** Token-aligned REC/OneRec logprobs for this choice */
  float* rec_token_logprobs;

  /** Number of entries in rec_token_logprobs */
  size_t rec_token_logprobs_size;
} XLLM_RecOutput;

/**
 * @brief List of REC/OneRec specific output extensions
 */
typedef struct XLLM_CAPI_EXPORT XLLM_RecOutputs {
  /** Pointer to array of REC output entries */
  XLLM_RecOutput* entries;

  /** Number of entries in the REC output array */
  size_t entries_size;
} XLLM_RecOutputs;

#define XLLM_ERROR_INFO_MAX_LEN 512

/**
 * @brief Inference response structure
 */
typedef struct XLLM_CAPI_EXPORT XLLM_Response {
  /** Response status code (0 = success, non-zero = error) */
  XLLM_StatusCode status_code;

  /** Error details (NULL = no error) */
  char error_info[XLLM_ERROR_INFO_MAX_LEN];

  /** Unique ID for the completion request (fixed-length string) */
  char id[XLLM_META_STRING_FIELD_MAX_LEN];

  /** Object type (fixed to "text_completion") */
  char object[XLLM_META_STRING_FIELD_MAX_LEN];

  /** Unix timestamp (seconds) of when the completion was created */
  int64_t created;

  /** Model name used for the completion */
  char model[XLLM_META_STRING_FIELD_MAX_LEN];

  /** List of generated completion candidates */
  XLLM_Choices choices;

  /** Token usage statistics for the request */
  XLLM_Usage usage;

  /** REC/OneRec specific response extensions */
  XLLM_RecOutputs rec_outputs;
} XLLM_Response;

/**
 * @brief Enumeration of tensor data types
 */
typedef enum XLLM_CAPI_EXPORT XLLM_DataType {
  XLLM_DTYPE_UNDEFINED = 0,
  XLLM_DTYPE_FLOAT16 = 1,
  XLLM_DTYPE_FLOAT32 = 2,
  XLLM_DTYPE_FLOAT64 = 3,
  XLLM_DTYPE_BFLOAT16 = 4,
  XLLM_DTYPE_INT8 = 5,
  XLLM_DTYPE_INT16 = 6,
  XLLM_DTYPE_INT32 = 7,
  XLLM_DTYPE_INT64 = 8,
  XLLM_DTYPE_UINT8 = 9,
  XLLM_DTYPE_UINT16 = 10,
  XLLM_DTYPE_UINT32 = 11,
  XLLM_DTYPE_UINT64 = 12,
  XLLM_DTYPE_BOOL = 13,
  XLLM_DTYPE_STRING = 14
} XLLM_DataType;

/**
 * @brief Structure representing tensor dimensions (shape)
 * @note Max supported rank is 8 (matches dim array length)
 */
typedef struct XLLM_CAPI_EXPORT XLLM_Dims {
  /** Number of dimensions (0=scalar, 1=vector, ..., 8) */
  int rank;

  /** Size of each dimension (unused dims must be 0) */
  int dim[8];
} XLLM_Dims;

/**
 * @brief Core tensor structure for numerical computation
 * @warning 1. data pointer is read-only, managed by external caller
 *          2. dtype must match the actual type of data buffer
 *          3. dims.rank must not exceed 8
 */
typedef struct XLLM_CAPI_EXPORT XLLM_Tensor {
  /** Data type of tensor elements */
  XLLM_DataType dtype;

  /** Dimension information (shape) of the tensor */
  XLLM_Dims dims;

  /** Read-only pointer to tensor data buffer */
  const void* data;
} XLLM_Tensor;

/**
 * @brief Dynamic list of tensors (replaces C++ std::vector<XLLM_Tensor>)
 */
typedef struct XLLM_CAPI_EXPORT XLLM_Tensors {
  XLLM_Tensor* entries;
  size_t entries_size;
} XLLM_Tensors;

/**
 * @brief Enumeration of multimodal data types (bitmask compatible)
 * @note Each type is a bit flag (supports multiple types via type_mask)
 */
typedef enum XLLM_CAPI_EXPORT XLLM_MM_Type {
  /** No multimodal type (invalid state) */
  XLLM_MM_TYPE_NONE = 0,

  /** Image modality (JPG/PNG/BMP) */
  XLLM_MM_TYPE_IMAGE = 1 << 0,

  /** Audio modality (WAV/MP3) */
  XLLM_MM_TYPE_AUDIO = 1 << 1,

  /** Video modality (H264/H265) */
  XLLM_MM_TYPE_VIDEO = 1 << 2,

  /** Text modality (tokenized text) */
  XLLM_MM_TYPE_TEXT = 1 << 3,

  /** Embedding modality (token embeddings) */
  XLLM_MM_TYPE_EMBEDDING = 1 << 4
} XLLM_MM_Type;

/**
 * @brief Multimodal value (variant type: single tensor or tensor list)
 */
typedef struct XLLM_CAPI_EXPORT XLLM_MM_Value {
  /** Type flag: true=single tensor, false=tensor list */
  bool is_single_tensor;

  union {
    /** Single tensor (valid if is_single_tensor=true) */
    XLLM_Tensor tensor;

    /** Tensor list (valid if is_single_tensor=false) */
    XLLM_Tensors tensors;
  } data;
} XLLM_MM_Value;

/**
 * @brief Single entry in multimodal dictionary (key-value pair)
 * @note 1. Key is fixed-length string (null-terminated if shorter than max len)
 *       2. Key must be unique within a dictionary
 */
typedef struct XLLM_CAPI_EXPORT XLLM_MM_DictEntry {
  /** Fixed-length key */
  char key[XLLM_META_STRING_FIELD_MAX_LEN];

  /** Value associated with the key */
  XLLM_MM_Value value;
} XLLM_MM_DictEntry;

/**
 * @brief Multimodal dictionary (array of key-value entries)
 * @note 1. entries is a heap-allocated array (must be freed by caller)
 *       2. entries_size = number of valid entries (no empty slots)
 */
typedef struct XLLM_CAPI_EXPORT XLLM_MM_Dict {
  XLLM_MM_DictEntry* entries;
  size_t entries_size;
} XLLM_MM_Dict;

/**
 * @brief Token position information (offset + length) for multimodal data
 * @note Used to map multimodal data to token positions in sequence
 */
typedef struct XLLM_CAPI_EXPORT XLLM_MM_TokenPos {
  /** Start offset of tokens (0-based) */
  uint32_t offset;

  /** Number of tokens (must be >0 for valid position) */
  uint32_t length;
} XLLM_MM_TokenPos;

/**
 * @brief Base struct for multimodal metadata (to be extended by specific
 * modalities)
 * @note This is a placeholder for modality-specific metadata (e.g., image size,
 * audio sample rate) Extend with union for image/audio/video metadata in
 * production use
 */
typedef struct XLLM_CAPI_EXPORT XLLM_MM_Meta {
  // Placeholder for future extension,
  // e.g., XLLM_ImageMeta|XLLM_AudioMeta|XLLM_VideoMeta
} XLLM_MM_Meta;

/**
 * @brief State information for a single multimodal item
 */
typedef struct XLLM_CAPI_EXPORT XLLM_MM_State {
  /** Token position for multimodal data alignment */
  XLLM_MM_TokenPos token_pos;
} XLLM_MM_State;

/**
 * @brief Single multimodal data item (core unit of multimodal data)
 * @note 1. type = modality type (image/audio/video/text/embedding)
 *       2. data = numerical content (tensor/tensor list)
 *       3. meta = modality-specific metadata (empty in base version)
 *       4. state = token position and processing state
 */
typedef struct XLLM_CAPI_EXPORT XLLM_MM_Item {
  /** Modality type (e.g., XLLM_MM_TYPE_EMBEDDING) */
  XLLM_MM_Type type;

  /** Core data (tensor/tensor list) */
  XLLM_MM_Value data;

  /** Modality-specific metadata (extendable) */
  XLLM_MM_Meta meta;

  /** Processing state and token position */
  XLLM_MM_State state;
} XLLM_MM_Item;

/**
 * @brief List of multimodal items (replaces C++ std::vector<XLLM_MM_Item>)
 * @note 1. entries is a heap-allocated array (must be freed by caller)
 *       2. entries_size = number of valid items (no empty slots)
 */
typedef struct XLLM_CAPI_EXPORT XLLM_MM_Items {
  XLLM_MM_Item* entries;
  size_t entries_size;
} XLLM_MM_Items;

/**
 * @brief Core multimodal data container (supports list/dict storage)
 */
typedef struct XLLM_CAPI_EXPORT XLLM_MM_Data {
  /** Bitmask of multimodal types (e.g., IMAGE | EMBEDDING) */
  uint32_t type_mask;

  /** Storage type: true=XLLM_MM_Dict, false=XLLM_MM_Items */
  bool is_dict;
  union {
    /** Dict storage (valid if is_dict=true) */
    XLLM_MM_Dict dict;

    /** List storage (valid if is_dict=false) */
    XLLM_MM_Items items;
  } data;
} XLLM_MM_Data;

#ifdef __cplusplus
}
#endif

#endif  // XLLM_C_TYPES_H

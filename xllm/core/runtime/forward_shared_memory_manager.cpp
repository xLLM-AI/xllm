/* Copyright 2025-2026 The xLLM Authors.
Copyright 2024 The ScaleLLM Authors. All Rights Reserved.
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

#include "forward_shared_memory_manager.h"

#include <gflags/gflags.h>

#include <atomic>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <numeric>
#include <optional>
#include <shared_mutex>
#include <stdexcept>

#include "core/common/global_flags.h"
#include "core/framework/config/eplb_config.h"
#include "core/framework/config/execution_config.h"
#include "core/framework/config/model_config.h"
#include "platform/stream.h"
#if defined(USE_CUDA)
#include <cuda_runtime_api.h>
#endif
#if defined(USE_NPU)
#include "platform/npu/device_capture_lock.h"
#elif defined(USE_CUDA)
#include "platform/cuda/device_capture_lock.h"
#endif
#include "core/util/net.h"
#include "core/util/tensor_helper.h"
#include "util/utils.h"

#if defined(__GNUC__)
static inline bool(likely)(bool x) { return __builtin_expect((x), true); }
static inline bool(unlikely)(bool x) { return __builtin_expect((x), false); }
#else
static inline bool(likely)(bool x) { return x; }
static inline bool(unlikely)(bool x) { return x; }
#endif

namespace xllm {

namespace {
template <typename T>
constexpr size_t type_size = sizeof(T);

constexpr size_t sampling_param_fixed_size() {
  return 5 * type_size<float>       // frequency_penalty, presence_penalty,
                                    // repetition_penalty, temperature, top_p
         + 2 * type_size<int64_t>   // top_k, top_logprobs
         + 3 * type_size<bool>      // logprobs, do_sample, is_embeddings
         + 2 * type_size<int32_t>;  // beam_width, num_return_sequences
}

constexpr size_t swap_block_info_fixed_size() {
  return type_size<int32_t> * 2;  // src_block_id + dst_block_id
}

constexpr std::uintptr_t kCudaZeroCopyAlignment = 16;
constexpr uint64_t kRawInputTensorArenaAlignment =
    static_cast<uint64_t>(kCudaZeroCopyAlignment);

inline uint64_t align_up(uint64_t value, uint64_t alignment) {
  if (alignment == 0) {
    return value;
  }
  return ((value + alignment - 1) / alignment) * alignment;
}

inline uint64_t get_aligned_tensor_arena_bytes(uint64_t data_bytes) {
  return align_up(data_bytes, kRawInputTensorArenaAlignment);
}

inline bool is_aligned_for_cuda_zero_copy(const void* ptr) {
  return reinterpret_cast<std::uintptr_t>(ptr) % kCudaZeroCopyAlignment == 0;
}

struct RawInputLayoutHeader final {
  uint64_t descriptor_bytes = 0;
  uint64_t tensor_arena_offset = 0;
  uint64_t tensor_arena_bytes = 0;
};

struct RawInputSectionCursor final {
  char* ptr = nullptr;
  uint64_t size = 0;
};

struct RawInputSerializeContext final {
  RawInputSectionCursor descriptor;
  RawInputSectionCursor tensor_arena;
};

inline size_t get_string_size(const std::string& str) {
  return type_size<uint64_t> + str.size();
}

inline size_t get_string_vector_size(const std::vector<std::string>& vec) {
  size_t size = type_size<uint64_t>;
  for (const auto& str : vec) {
    size += get_string_size(str);
  }
  return size;
}

template <typename T>
inline size_t get_vector_size(const std::vector<T>& vec) {
  return type_size<uint64_t> + vec.size() * type_size<T>;
}

template <typename T>
size_t get_vector_to_tensor_size(const std::vector<T>& vec) {
  uint64_t size = type_size<uint64_t>;  // ndim
  size += type_size<uint64_t>;          // shape
  size += type_size<int8_t>;            // dtype
  size += type_size<uint64_t>;          // databytes
  size += vec.size() * type_size<T>;    // data
  return size;
}

inline size_t get_tensor_size(const torch::Tensor& tensor) {
  uint64_t size = type_size<uint64_t>;  // ndim
  if (!tensor.defined()) {
    return size;
  }
  size += type_size<uint64_t> * tensor.dim();      // shape
  size += type_size<int8_t>;                       // dtype
  size += type_size<uint64_t>;                     // databytes
  size += tensor.numel() * tensor.element_size();  // data
  return size;
}

template <typename T>
inline size_t get_2d_vector_size(const std::vector<std::vector<T>>& vec2d) {
  size_t size = type_size<uint64_t>;
  for (const auto& vec : vec2d) {
    size += get_vector_size(vec);
  }
  return size;
}

template <typename T>
size_t get_2d_vector_to_tensor_size(const std::vector<std::vector<T>>& vec2d) {
  uint64_t size = type_size<uint64_t>;  // ndim
  if (vec2d.size() == 0 || vec2d[0].size() == 0) {
    return size;
  }
  size += type_size<uint64_t> * 2;                        // shape
  size += type_size<int8_t>;                              // dtype
  size += type_size<uint64_t>;                            // databytes
  size += vec2d.size() * vec2d[0].size() * type_size<T>;  // data
  return size;
}

void normalize_linear_state_ids(std::vector<int32_t>& linear_state_ids,
                                int32_t num_sequences) {
  if (num_sequences <= 0) {
    linear_state_ids.clear();
    return;
  }
  if (linear_state_ids.empty()) {
    linear_state_ids.assign(num_sequences, -1);
    return;
  }
  CHECK_EQ(linear_state_ids.size(), static_cast<size_t>(num_sequences))
      << "linear_state_ids size (" << linear_state_ids.size()
      << ") must match num_sequences (" << num_sequences << ")";
}

inline size_t get_instance_info_size(const InstanceInfo& info) {
  size_t size = get_string_size(info.name) + get_string_size(info.rpc_address) +
                get_string_size(info.type);

  size += type_size<uint64_t> + info.cluster_ids.size() * type_size<uint64_t>;

  size += type_size<uint64_t>;
  for (const auto& addr : info.addrs) {
    size += get_string_size(addr);
  }

  size += type_size<int32_t>    // dp_size
          + type_size<int32_t>  // kv_split_size
          + type_size<uint64_t> +
          info.ttft_profiling_data.size() *
              (type_size<int32_t> + type_size<int64_t>);

  return size;
}

inline size_t get_xtensor_layer_offsets_size(
    const std::vector<XTensorLayerOffsets>& offsets) {
  size_t total = type_size<uint64_t>;  // num_layers
  for (const auto& layer : offsets) {
    total +=
        get_vector_size(layer.k_offsets) + get_vector_size(layer.v_offsets);
  }
  return total;
}

inline size_t get_block_transfer_groups_size(
    const std::vector<KVBlockTransferGroup>& groups) {
  size_t total = type_size<uint64_t>;
  for (const auto& group : groups) {
    total += type_size<int32_t> + get_vector_size(group.local_blocks_ids) +
             get_vector_size(group.remote_blocks_ids);
  }
  return total;
}

inline size_t get_transfer_kv_info_size(const TransferKVInfo& info) {
  return get_string_size(info.request_id) +
         get_vector_size(info.local_blocks_ids) +
         get_vector_size(info.remote_blocks_ids) +
         get_vector_size(info.local_linear_state_ids) +
         get_vector_size(info.remote_linear_state_ids) +
         type_size<int32_t>  // dp_rank
         + get_instance_info_size(info.remote_instance_info) +
         get_xtensor_layer_offsets_size(info.dst_xtensor_layer_offsets) +
         get_block_transfer_groups_size(info.block_transfer_groups);
}

inline size_t get_eplb_info_size(const EplbInfo& info) {
  return type_size<int32_t>  // prepare_layer_id
         + get_vector_size(info.expert_ids) +
         type_size<int32_t>;  // update_layer_id
}

inline size_t get_mm_dict_size(const MMDict& mm_dict) {
  size_t total = 0;
  total += type_size<size_t>;  // mm_dict size
  for (auto& [mm_key, mm_value] : mm_dict) {
    total += get_string_size(mm_key);
    total += type_size<int32_t>;  // num of tensors
    if (std::holds_alternative<torch::Tensor>(mm_value)) {
      total += get_tensor_size(std::get<torch::Tensor>(mm_value));
    } else if (std::holds_alternative<std::vector<torch::Tensor>>(mm_value)) {
      for (const auto& tensor :
           std::get<std::vector<torch::Tensor>>(mm_value)) {
        total += get_tensor_size(tensor);
      }
    }
  }
  return total;
}

inline size_t get_mm_item_size(const MMDataItem& mm_item) {
  size_t total = 0;

  total += type_size<uint32_t>;               // type
  total += get_mm_dict_size(mm_item.data());  // dict

  // token_pos
  total += type_size<int32_t> * 2;
  // mm_token_num
  total += type_size<int32_t>;

  // mm_token_mask
  total += get_tensor_size(mm_item.state().mm_token_mask());

  // seq_index
  total += type_size<int32_t>;

  // schedule_data
  total += XXH3_128BITS_HASH_VALUE_LEN;
  total += type_size<int32_t> * 2;

  return total;
}

size_t get_sampling_params_size(const SamplingParameters& params) {
  size_t total = 0;

  total += get_tensor_size(params.selected_token_idxes);
  total += get_tensor_size(params.frequency_penalties);
  total += get_tensor_size(params.presence_penalties);
  total += get_tensor_size(params.repetition_penalties);
  total += get_tensor_size(params.temperatures);
  total += get_tensor_size(params.top_p);
  total += get_tensor_size(params.top_k);
  total += get_tensor_size(params.unique_token_ids);
  total += get_tensor_size(params.unique_token_counts);
  total += get_tensor_size(params.unique_token_ids_lens);
  total += get_tensor_size(params.sample_idxes);
  total += get_tensor_size(params.do_sample);
  total += get_tensor_size(params.acc_logprob);
  total += type_size<bool> * 5    // all_random_sample + all_greedy_sample +
                                  // logprobs + is_embeddings + use_beam_search
           + type_size<int64_t>   // max_top_logprobs
           + type_size<int32_t>;  // num_return_sequences
  return total;
}

inline size_t get_mm_data_size(const MMData& mm_data) {
  size_t total = 0;
  total += type_size<uint32_t>;  //  mm_type
  if (mm_data.hold<MMItemVec>()) {
    total += type_size<size_t>;  // num of mm_items
    const auto& mm_items = mm_data.items<MMItemVec>();
    for (const auto& mm_item : mm_items) {
      total += get_mm_item_size(mm_item);
    }
  } else if (mm_data.hold<MMDict>()) {
    total += get_mm_dict_size(mm_data.items<MMDict>());
  }
  return total;
}

inline size_t get_mm_batch_data_size(const MMBatchData& mm_data) {
  const auto& vec = mm_data.mm_data_vec();

  size_t total = 0;
  total += type_size<size_t>;   // num of vec
  total += type_size<uint8_t>;  // is_mm_item
  for (const auto& mm_data : vec) {
    total += get_mm_data_size(mm_data);
  }
  return total;
}

inline size_t get_vector_tensor_size(
    const std::vector<torch::Tensor>& tensor_vec) {
  size_t size = type_size<int32_t>;  // vector size
  for (const auto& tensor : tensor_vec) {
    size += get_tensor_size(tensor);
  }
  return size;
}

// calculate dit input size
inline size_t get_dit_generation_params_size(
    const DiTGenerationParams& params) {
  return type_size<int32_t> *
             4  // width, height, num_inference_steps, max_sequence_length
         + type_size<float> *
               4  // true_cfg_scale, guidance_scale, strength, cfg_renorm_min
         + type_size<uint32_t>  // num_images_per_prompt
         + type_size<int64_t>   // seed
         + type_size<bool>      // enable_cfg_renorm
         + type_size<int32_t>   // num_frames
         + type_size<bool>      // force_video_output
         + type_size<double>    // video_fps
         + type_size<float> *   // guidance_scale_2, boundary_ratio, flow_shift
               3 +
         type_size<int32_t>        // seconds
         + type_size<uint32_t>     // num_videos_per_prompt
         + type_size<int32_t> * 2  // max_new_tokens, diffusion_steps
         + type_size<float> * 3    // temperature, top_p, repetition_penalty
         + type_size<int32_t>      // top_k
         + type_size<bool>;        // seed_is_set
}

inline size_t get_dit_forward_input_size(const DiTForwardInput& input) {
  size_t size = type_size<int>;  // batch_size

  // Vector of strings
  size += get_string_vector_size(input.prompts);
  size += get_string_vector_size(input.prompts_2);
  size += get_string_vector_size(input.negative_prompts);
  size += get_string_vector_size(input.negative_prompts_2);

  // Tensors
  size += get_tensor_size(input.images);
  size += get_vector_tensor_size(input.images_list);
  size += get_tensor_size(input.mask_images);
  size += get_tensor_size(input.control_image);
  size += get_tensor_size(input.masked_image_latents);
  size += get_tensor_size(input.prompt_embeds);
  size += get_tensor_size(input.pooled_prompt_embeds);
  size += get_tensor_size(input.negative_prompt_embeds);
  size += get_tensor_size(input.negative_pooled_prompt_embeds);
  size += get_tensor_size(input.latents);
  size += get_tensor_size(input.last_images);
  size += get_tensor_size(input.prompt_audio);
  size += get_string_size(input.audio_prompt_text);

  // Generation params
  size += get_dit_generation_params_size(input.generation_params);

  return size;
}

inline size_t get_dit_forward_output_size(const DiTForwardOutput& output) {
  return get_vector_tensor_size(output.tensors) +
         get_string_vector_size(output.text_output);
}

template <typename T>
inline void write_data(char*& buffer, const T& data) {
  *reinterpret_cast<T*>(buffer) = data;
  buffer += type_size<T>;
}

template <typename T>
inline void write_data(RawInputSectionCursor& cursor, const T& data) {
  if (cursor.ptr != nullptr) {
    *reinterpret_cast<T*>(cursor.ptr) = data;
    cursor.ptr += type_size<T>;
  }
  cursor.size += type_size<T>;
}

inline void write_bytes(RawInputSectionCursor& cursor,
                        const void* data,
                        uint64_t bytes) {
  if (bytes == 0) {
    return;
  }
  if (cursor.ptr != nullptr) {
    std::memcpy(cursor.ptr, data, bytes);
    cursor.ptr += bytes;
  }
  cursor.size += bytes;
}

inline void write_linear_state_cache_ops(
    RawInputSerializeContext& context,
    const std::vector<LinearStateCacheOp>& cache_ops) {
  write_data(context.descriptor, static_cast<uint64_t>(cache_ops.size()));
  for (const LinearStateCacheOp& cache_op : cache_ops) {
    write_data(context.descriptor, cache_op.linear_state_id);
    write_data(context.descriptor, cache_op.restore_requested);
    write_data(context.descriptor, cache_op.restore_src_slot_id);
  }
}

inline void write_padding(RawInputSectionCursor& cursor, uint64_t bytes) {
  if (bytes == 0) {
    return;
  }
  if (cursor.ptr != nullptr) {
    std::memset(cursor.ptr, 0, bytes);
    cursor.ptr += bytes;
  }
  cursor.size += bytes;
}

inline void write_string(char*& buffer, const std::string& str) {
  const uint64_t len = str.size();
  write_data(buffer, len);
  if (len > 0) {
    std::memcpy(buffer, str.data(), len);
    buffer += len;
  }
}

inline void write_string(RawInputSectionCursor& cursor,
                         const std::string& str) {
  const uint64_t len = str.size();
  write_data(cursor, len);
  if (len > 0) {
    write_bytes(cursor, str.data(), len);
  }
}

inline void write_string_vector(char*& buffer,
                                const std::vector<std::string>& vec) {
  const uint64_t size = vec.size();
  write_data(buffer, size);
  for (const auto& str : vec) {
    write_string(buffer, str);
  }
}

inline void write_string_vector(RawInputSectionCursor& cursor,
                                const std::vector<std::string>& vec) {
  const uint64_t size = vec.size();
  write_data(cursor, size);
  for (const auto& str : vec) {
    write_string(cursor, str);
  }
}

inline void write_tensor(char*& buffer, const torch::Tensor& tensor) {
  if (!tensor.defined()) {
    uint64_t ndim = 0;
    write_data(buffer, ndim);
    return;
  }
  auto contig_tensor = tensor.cpu().contiguous();
  // write ndim
  const uint64_t tensor_ndim = contig_tensor.dim();
  write_data(buffer, tensor_ndim);
  // write shape
  for (int64_t i = 0; i < contig_tensor.dim(); ++i) {
    write_data(buffer, static_cast<uint64_t>(contig_tensor.size(i)));
  }
  // write dtype
  const int8_t tensor_dtype = static_cast<int8_t>(contig_tensor.scalar_type());
  write_data(buffer, tensor_dtype);
  // write data_bytes
  const uint64_t tensor_data_bytes =
      contig_tensor.numel() * contig_tensor.element_size();
  write_data(buffer, tensor_data_bytes);

  if (tensor_data_bytes > 0) {
    std::memcpy(buffer, contig_tensor.data_ptr(), tensor_data_bytes);
    buffer += tensor_data_bytes;
  }
}

inline void write_tensor(RawInputSerializeContext& context,
                         const torch::Tensor& tensor) {
  if (!tensor.defined()) {
    uint64_t ndim = 0;
    write_data(context.descriptor, ndim);
    return;
  }

  const uint64_t tensor_ndim = tensor.dim();
  write_data(context.descriptor, tensor_ndim);
  for (int64_t i = 0; i < tensor.dim(); ++i) {
    write_data(context.descriptor, static_cast<uint64_t>(tensor.size(i)));
  }

  const int8_t tensor_dtype = static_cast<int8_t>(tensor.scalar_type());
  write_data(context.descriptor, tensor_dtype);

  const uint64_t tensor_data_bytes = tensor.numel() * tensor.element_size();
  write_data(context.descriptor, tensor_data_bytes);

  if (tensor_data_bytes == 0) {
    return;
  }

  if (context.tensor_arena.ptr != nullptr) {
    torch::Tensor contiguous_tensor = tensor.cpu().contiguous();
    write_bytes(
        context.tensor_arena, contiguous_tensor.data_ptr(), tensor_data_bytes);
  } else {
    context.tensor_arena.size += tensor_data_bytes;
  }

  const uint64_t aligned_bytes =
      get_aligned_tensor_arena_bytes(tensor_data_bytes);
  write_padding(context.tensor_arena, aligned_bytes - tensor_data_bytes);
}

inline void write_sampling_param(char*& buffer,
                                 const RequestSamplingParam& param) {
  char* ptr = buffer;
  *reinterpret_cast<float*>(ptr) = param.frequency_penalty;
  ptr += type_size<float>;
  *reinterpret_cast<float*>(ptr) = param.presence_penalty;
  ptr += type_size<float>;
  *reinterpret_cast<float*>(ptr) = param.repetition_penalty;
  ptr += type_size<float>;
  *reinterpret_cast<float*>(ptr) = param.temperature;
  ptr += type_size<float>;
  *reinterpret_cast<float*>(ptr) = param.top_p;
  ptr += type_size<float>;
  *reinterpret_cast<int64_t*>(ptr) = param.top_k;
  ptr += type_size<int64_t>;
  *reinterpret_cast<bool*>(ptr) = param.logprobs;
  ptr += type_size<bool>;
  *reinterpret_cast<int64_t*>(ptr) = param.top_logprobs;
  ptr += type_size<int64_t>;
  *reinterpret_cast<bool*>(ptr) = param.do_sample;
  ptr += type_size<bool>;
  *reinterpret_cast<bool*>(ptr) = param.is_embeddings;
  ptr += type_size<bool>;
  *reinterpret_cast<int32_t*>(ptr) = param.beam_width;
  ptr += type_size<int32_t>;
  *reinterpret_cast<int32_t*>(ptr) = param.num_return_sequences;
  ptr += type_size<int32_t>;
  buffer = ptr;
}

template <typename T>
inline void write_vector(char*& buffer, const std::vector<T>& vec) {
  const uint64_t size = vec.size();
  write_data(buffer, size);
  if (size > 0) {
    const size_t bytes = size * type_size<T>;
    std::memcpy(buffer, vec.data(), bytes);
    buffer += bytes;
  }
}

template <typename T>
inline void write_vector(RawInputSectionCursor& cursor,
                         const std::vector<T>& vec) {
  const uint64_t size = vec.size();
  write_data(cursor, size);
  if (size > 0) {
    const uint64_t bytes = size * type_size<T>;
    write_bytes(cursor, vec.data(), bytes);
  }
}

template <typename T>
void write_vector_to_tensor(char*& buffer, const std::vector<T>& vec) {
  // write ndim
  uint64_t ndim = 1;
  write_data(buffer, ndim);
  // write shape
  write_data(buffer, vec.size());
  // write dtype
  const int8_t tensor_dtype = static_cast<int8_t>(get_scalar_type<T>());
  write_data(buffer, tensor_dtype);
  // write data_bytes
  const uint64_t data_bytes = vec.size() * type_size<T>;
  write_data(buffer, data_bytes);
  // write vec data
  if (data_bytes > 0) {
    std::memcpy(buffer, vec.data(), data_bytes);
    buffer += data_bytes;
  }
}

template <typename T>
void write_vector_to_tensor(RawInputSerializeContext& context,
                            const std::vector<T>& vec) {
  uint64_t ndim = 1;
  write_data(context.descriptor, ndim);
  write_data(context.descriptor, vec.size());

  const int8_t tensor_dtype = static_cast<int8_t>(get_scalar_type<T>());
  write_data(context.descriptor, tensor_dtype);

  const uint64_t data_bytes = vec.size() * type_size<T>;
  write_data(context.descriptor, data_bytes);
  write_bytes(context.tensor_arena, vec.data(), data_bytes);
  const uint64_t aligned_bytes = get_aligned_tensor_arena_bytes(data_bytes);
  write_padding(context.tensor_arena, aligned_bytes - data_bytes);
}

template <typename T>
inline void write_2d_vector(char*& buffer,
                            const std::vector<std::vector<T>>& vec2d) {
  write_data(buffer, (uint64_t)vec2d.size());
  for (const auto& vec : vec2d) {
    write_vector(buffer, vec);
  }
}

template <typename T>
inline void write_2d_vector(RawInputSectionCursor& cursor,
                            const std::vector<std::vector<T>>& vec2d) {
  write_data(cursor, static_cast<uint64_t>(vec2d.size()));
  for (const auto& vec : vec2d) {
    write_vector(cursor, vec);
  }
}

template <typename T>
void write_2d_vector_to_tensor(char*& buffer,
                               const std::vector<std::vector<T>>& vec2d) {
  // write ndim
  uint64_t ndim;
  if (vec2d.size() == 0 || vec2d[0].size() == 0) {
    ndim = 0;
    write_data(buffer, ndim);
    return;
  }
  ndim = 2;
  write_data(buffer, ndim);
  // write shape
  write_data(buffer, vec2d.size());
  write_data(buffer, vec2d[0].size());
  // write dtype
  const int8_t tensor_dtype = static_cast<int8_t>(get_scalar_type<T>());
  write_data(buffer, tensor_dtype);
  // write data_bytes
  const uint64_t per_data_bytes = vec2d[0].size() * type_size<T>;
  const uint64_t data_bytes = vec2d.size() * per_data_bytes;
  write_data(buffer, data_bytes);
  // write vec data
  for (const auto& vec : vec2d) {
    std::memcpy(buffer, vec.data(), per_data_bytes);
    buffer += per_data_bytes;
  }
}

template <typename T>
void write_2d_vector_to_tensor(RawInputSerializeContext& context,
                               const std::vector<std::vector<T>>& vec2d) {
  uint64_t ndim;
  if (vec2d.empty() || vec2d[0].empty()) {
    ndim = 0;
    write_data(context.descriptor, ndim);
    return;
  }

  ndim = 2;
  write_data(context.descriptor, ndim);
  write_data(context.descriptor, vec2d.size());
  write_data(context.descriptor, vec2d[0].size());

  const int8_t tensor_dtype = static_cast<int8_t>(get_scalar_type<T>());
  write_data(context.descriptor, tensor_dtype);

  const uint64_t per_data_bytes = vec2d[0].size() * type_size<T>;
  const uint64_t data_bytes = vec2d.size() * per_data_bytes;
  write_data(context.descriptor, data_bytes);

  for (const auto& vec : vec2d) {
    write_bytes(context.tensor_arena, vec.data(), per_data_bytes);
  }

  const uint64_t aligned_bytes = get_aligned_tensor_arena_bytes(data_bytes);
  write_padding(context.tensor_arena, aligned_bytes - data_bytes);
}

inline void write_instance_info(char*& buffer, const InstanceInfo& info) {
  write_string(buffer, info.name);
  write_string(buffer, info.rpc_address);
  write_string(buffer, info.type);

  write_vector(buffer, info.cluster_ids);

  write_data(buffer, (uint64_t)info.addrs.size());
  for (const auto& addr : info.addrs) {
    write_string(buffer, addr);
  }

  write_data(buffer, info.dp_size);
  write_data(buffer, info.kv_split_size);

  const uint64_t prof_size = info.ttft_profiling_data.size();
  write_data(buffer, prof_size);
  if (prof_size > 0) {
    std::memcpy(buffer,
                info.ttft_profiling_data.data(),
                prof_size * sizeof(std::pair<int32_t, int64_t>));
    buffer += prof_size * sizeof(std::pair<int32_t, int64_t>);
  }
}

inline void write_instance_info(RawInputSerializeContext& context,
                                const InstanceInfo& info) {
  write_string(context.descriptor, info.name);
  write_string(context.descriptor, info.rpc_address);
  write_string(context.descriptor, info.type);

  write_vector(context.descriptor, info.cluster_ids);

  write_data(context.descriptor, static_cast<uint64_t>(info.addrs.size()));
  for (const auto& addr : info.addrs) {
    write_string(context.descriptor, addr);
  }

  write_data(context.descriptor, info.dp_size);
  write_data(context.descriptor, info.kv_split_size);

  const uint64_t prof_size = info.ttft_profiling_data.size();
  write_data(context.descriptor, prof_size);
  if (prof_size > 0) {
    write_bytes(context.descriptor,
                info.ttft_profiling_data.data(),
                prof_size * sizeof(std::pair<int32_t, int64_t>));
  }
}

inline void write_xtensor_layer_offsets(
    char*& buffer,
    const std::vector<XTensorLayerOffsets>& offsets) {
  write_data(buffer, (uint64_t)offsets.size());
  for (const auto& layer : offsets) {
    write_vector(buffer, layer.k_offsets);
    write_vector(buffer, layer.v_offsets);
  }
}

inline void write_block_transfer_groups(
    char*& buffer,
    const std::vector<KVBlockTransferGroup>& groups) {
  write_data(buffer, static_cast<uint64_t>(groups.size()));
  for (const auto& group : groups) {
    write_data(buffer, group.group_id);
    write_vector(buffer, group.local_blocks_ids);
    write_vector(buffer, group.remote_blocks_ids);
  }
}

inline void write_block_transfer_groups(
    RawInputSerializeContext& context,
    const std::vector<KVBlockTransferGroup>& groups) {
  write_data(context.descriptor, static_cast<uint64_t>(groups.size()));
  for (const auto& group : groups) {
    write_data(context.descriptor, group.group_id);
    write_vector(context.descriptor, group.local_blocks_ids);
    write_vector(context.descriptor, group.remote_blocks_ids);
  }
}

inline void write_xtensor_layer_offsets(
    RawInputSerializeContext& context,
    const std::vector<XTensorLayerOffsets>& offsets) {
  write_data(context.descriptor, static_cast<uint64_t>(offsets.size()));
  for (const auto& layer : offsets) {
    write_vector(context.descriptor, layer.k_offsets);
    write_vector(context.descriptor, layer.v_offsets);
  }
}

inline void write_transfer_kv_info(char*& buffer, const TransferKVInfo& info) {
  write_string(buffer, info.request_id);
  write_vector(buffer, info.local_blocks_ids);
  write_vector(buffer, info.remote_blocks_ids);
  write_vector(buffer, info.local_linear_state_ids);
  write_vector(buffer, info.remote_linear_state_ids);
  write_data(buffer, info.dp_rank);
  write_instance_info(buffer, info.remote_instance_info);
  write_xtensor_layer_offsets(buffer, info.dst_xtensor_layer_offsets);
  write_block_transfer_groups(buffer, info.block_transfer_groups);
}

inline void write_transfer_kv_info(RawInputSerializeContext& context,
                                   const TransferKVInfo& info) {
  write_string(context.descriptor, info.request_id);
  write_vector(context.descriptor, info.local_blocks_ids);
  write_vector(context.descriptor, info.remote_blocks_ids);
  write_vector(context.descriptor, info.local_linear_state_ids);
  write_vector(context.descriptor, info.remote_linear_state_ids);
  write_data(context.descriptor, info.dp_rank);
  write_instance_info(context, info.remote_instance_info);
  write_xtensor_layer_offsets(context, info.dst_xtensor_layer_offsets);
  write_block_transfer_groups(context, info.block_transfer_groups);
}

inline void write_eplb_info(char*& buffer, const EplbInfo& info) {
  write_data(buffer, info.prepare_layer_id);
  write_vector(buffer, info.expert_ids);
  write_data(buffer, info.update_layer_id);
}

inline void write_eplb_info(RawInputSerializeContext& context,
                            const EplbInfo& info) {
  write_data(context.descriptor, info.prepare_layer_id);
  write_vector(context.descriptor, info.expert_ids);
  write_data(context.descriptor, info.update_layer_id);
}

inline void write_swap_blocks(char*& buffer,
                              const std::vector<BlockTransferInfo>& blocks) {
  write_data(buffer, (uint64_t)blocks.size());

  for (const auto& b : blocks) {
    write_data(buffer, b.src_block_id);
    write_data(buffer, b.dst_block_id);
  }
}

inline void write_swap_blocks(RawInputSerializeContext& context,
                              const std::vector<BlockTransferInfo>& blocks) {
  write_data(context.descriptor, static_cast<uint64_t>(blocks.size()));
  for (const auto& b : blocks) {
    write_data(context.descriptor, b.src_block_id);
    write_data(context.descriptor, b.dst_block_id);
  }
}

inline void write_vector_tensor(char*& buffer,
                                const std::vector<torch::Tensor>& tensor_vec) {
  int32_t tensor_num = tensor_vec.size();
  write_data(buffer, tensor_num);
  for (const auto& tensor : tensor_vec) {
    write_tensor(buffer, tensor);
  }
}

inline void write_vector_tensor(RawInputSerializeContext& context,
                                const std::vector<torch::Tensor>& tensor_vec) {
  int32_t tensor_num = tensor_vec.size();
  write_data(context.descriptor, tensor_num);
  for (const auto& tensor : tensor_vec) {
    write_tensor(context, tensor);
  }
}

inline void write_mm_dict(char*& buffer, const MMDict& mm_dict) {
  // size
  size_t size = mm_dict.size();
  write_data(buffer, (size_t)size);
  // tensor num
  int32_t tensor_num = 1;
  for (auto& [mm_key, mm_value] : mm_dict) {
    write_string(buffer, mm_key);
    if (std::holds_alternative<torch::Tensor>(mm_value)) {
      tensor_num = 1;
      write_data(buffer, tensor_num);
      auto& tensor = std::get<torch::Tensor>(mm_value);
      write_tensor(buffer, tensor);
    } else if (std::holds_alternative<std::vector<torch::Tensor>>(mm_value)) {
      auto& tensor_vec = std::get<std::vector<torch::Tensor>>(mm_value);
      tensor_num = tensor_vec.size();
      write_data(buffer, tensor_num);
      for (const auto& tensor : tensor_vec) {
        write_tensor(buffer, tensor);
      }
    }
  }
}

inline void write_mm_dict(RawInputSerializeContext& context,
                          const MMDict& mm_dict) {
  size_t size = mm_dict.size();
  write_data(context.descriptor, size);
  int32_t tensor_num = 1;
  for (auto& [mm_key, mm_value] : mm_dict) {
    write_string(context.descriptor, mm_key);
    if (std::holds_alternative<torch::Tensor>(mm_value)) {
      tensor_num = 1;
      write_data(context.descriptor, tensor_num);
      write_tensor(context, std::get<torch::Tensor>(mm_value));
    } else if (std::holds_alternative<std::vector<torch::Tensor>>(mm_value)) {
      auto& tensor_vec = std::get<std::vector<torch::Tensor>>(mm_value);
      tensor_num = tensor_vec.size();
      write_data(context.descriptor, tensor_num);
      for (const auto& tensor : tensor_vec) {
        write_tensor(context, tensor);
      }
    }
  }
}

inline void write_mm_item(char*& buffer, const MMDataItem& item) {
  write_data(buffer, item.type());
  write_mm_dict(buffer, item.data());

  const auto& state = item.state();
  write_data(buffer, state.seq_index());
  // write token_pos
  write_data(buffer, state.token_pos().offset);
  write_data(buffer, state.token_pos().length);
  write_data(buffer, state.mm_token_num());

  // write mm_token_mask
  write_tensor(buffer, state.mm_token_mask());

  // write schedule_data
  memcpy(buffer, state.schedule_data().key.data, XXH3_128BITS_HASH_VALUE_LEN);
  buffer += XXH3_128BITS_HASH_VALUE_LEN;
  write_data(buffer, state.schedule_data().start_pos);
  write_data(buffer, state.schedule_data().end_pos);
}

inline void write_mm_item(RawInputSerializeContext& context,
                          const MMDataItem& item) {
  write_data(context.descriptor, item.type());
  write_mm_dict(context, item.data());

  const auto& state = item.state();
  write_data(context.descriptor, state.seq_index());
  write_data(context.descriptor, state.token_pos().offset);
  write_data(context.descriptor, state.token_pos().length);
  write_data(context.descriptor, state.mm_token_num());
  write_tensor(context, state.mm_token_mask());
  write_bytes(context.descriptor,
              state.schedule_data().key.data,
              XXH3_128BITS_HASH_VALUE_LEN);
  write_data(context.descriptor, state.schedule_data().start_pos);
  write_data(context.descriptor, state.schedule_data().end_pos);
}

inline void write_mm_data_items(char*& buffer, const MMData& mm_data) {
  const auto& mm_items = mm_data.items<MMItemVec>();
  write_data(buffer, mm_data.type());
  write_data(buffer, mm_items.size());
  for (const auto& mm_item : mm_items) {
    write_mm_item(buffer, mm_item);
  }
}

inline void write_mm_data_items(RawInputSerializeContext& context,
                                const MMData& mm_data) {
  const auto& mm_items = mm_data.items<MMItemVec>();
  write_data(context.descriptor, mm_data.type());
  write_data(context.descriptor, mm_items.size());
  for (const auto& mm_item : mm_items) {
    write_mm_item(context, mm_item);
  }
}

inline void write_mm_data_dict(char*& buffer, const MMData& mm_data) {
  const auto& mm_dict = mm_data.items<MMDict>();
  write_data(buffer, mm_data.type());
  write_mm_dict(buffer, mm_dict);
}

inline void write_mm_data_dict(RawInputSerializeContext& context,
                               const MMData& mm_data) {
  write_data(context.descriptor, mm_data.type());
  write_mm_dict(context, mm_data.items<MMDict>());
}

inline void write_mm_batch_data(char*& buffer, const MMBatchData& mm_data) {
  const auto& vec = mm_data.mm_data_vec();
  write_data(buffer, vec.size());

  uint8_t is_mm_item =
      vec.size() ? static_cast<uint8_t>(vec[0].hold<MMItemVec>()) : 1;
  write_data(buffer, is_mm_item);
  std::function<void(char*&, const MMData&)> write_mm_data =
      is_mm_item
          ? static_cast<void (*)(char*&, const MMData&)>(&write_mm_data_items)
          : static_cast<void (*)(char*&, const MMData&)>(&write_mm_data_dict);
  for (const auto& mm_data : vec) {
    write_mm_data(buffer, mm_data);
  }
}

inline void write_mm_batch_data(RawInputSerializeContext& context,
                                const MMBatchData& mm_data) {
  const auto& vec = mm_data.mm_data_vec();
  write_data(context.descriptor, vec.size());

  uint8_t is_mm_item =
      vec.empty() ? 1 : static_cast<uint8_t>(vec[0].hold<MMItemVec>());
  write_data(context.descriptor, is_mm_item);
  std::function<void(RawInputSerializeContext&, const MMData&)> write_mm_data =
      is_mm_item
          ? static_cast<void (*)(RawInputSerializeContext&, const MMData&)>(
                &write_mm_data_items)
          : static_cast<void (*)(RawInputSerializeContext&, const MMData&)>(
                &write_mm_data_dict);
  for (const auto& current_mm_data : vec) {
    write_mm_data(context, current_mm_data);
  }
}

// write dit data
inline void write_dit_generation_params(char*& buffer,
                                        const DiTGenerationParams& params) {
  write_data(buffer, params.width);
  write_data(buffer, params.height);
  write_data(buffer, params.num_inference_steps);
  write_data(buffer, params.true_cfg_scale);
  write_data(buffer, params.guidance_scale);
  write_data(buffer, params.num_images_per_prompt);
  write_data(buffer, params.seed);
  write_data(buffer, params.max_sequence_length);
  write_data(buffer, params.strength);
  write_data(buffer, params.enable_cfg_renorm);
  write_data(buffer, params.cfg_renorm_min);
  write_data(buffer, params.num_frames);
  write_data(buffer, params.force_video_output);
  write_data(buffer, params.video_fps);
  write_data(buffer, params.guidance_scale_2);
  write_data(buffer, params.seconds);
  write_data(buffer, params.boundary_ratio);
  write_data(buffer, params.flow_shift);
  write_data(buffer, params.num_videos_per_prompt);
  write_data(buffer, params.max_new_tokens);
  write_data(buffer, params.diffusion_steps);
  write_data(buffer, params.temperature);
  write_data(buffer, params.top_k);
  write_data(buffer, params.top_p);
  write_data(buffer, params.repetition_penalty);
  write_data(buffer, params.seed_is_set);
}

inline void write_dit_generation_params(RawInputSerializeContext& context,
                                        const DiTGenerationParams& params) {
  write_data(context.descriptor, params.width);
  write_data(context.descriptor, params.height);
  write_data(context.descriptor, params.num_inference_steps);
  write_data(context.descriptor, params.true_cfg_scale);
  write_data(context.descriptor, params.guidance_scale);
  write_data(context.descriptor, params.num_images_per_prompt);
  write_data(context.descriptor, params.seed);
  write_data(context.descriptor, params.max_sequence_length);
  write_data(context.descriptor, params.strength);
  write_data(context.descriptor, params.enable_cfg_renorm);
  write_data(context.descriptor, params.cfg_renorm_min);
  write_data(context.descriptor, params.num_frames);
  write_data(context.descriptor, params.force_video_output);
  write_data(context.descriptor, params.video_fps);
  write_data(context.descriptor, params.guidance_scale_2);
  write_data(context.descriptor, params.seconds);
  write_data(context.descriptor, params.boundary_ratio);
  write_data(context.descriptor, params.flow_shift);
  write_data(context.descriptor, params.num_videos_per_prompt);
  write_data(context.descriptor, params.max_new_tokens);
  write_data(context.descriptor, params.diffusion_steps);
  write_data(context.descriptor, params.temperature);
  write_data(context.descriptor, params.top_k);
  write_data(context.descriptor, params.top_p);
  write_data(context.descriptor, params.repetition_penalty);
  write_data(context.descriptor, params.seed_is_set);
}

inline void write_dit_forward_input(char*& buffer,
                                    const DiTForwardInput& input) {
  write_data(buffer, input.batch_size);

  write_string_vector(buffer, input.prompts);
  write_string_vector(buffer, input.prompts_2);
  write_string_vector(buffer, input.negative_prompts);
  write_string_vector(buffer, input.negative_prompts_2);

  write_tensor(buffer, input.images);
  write_vector_tensor(buffer, input.images_list);
  write_tensor(buffer, input.mask_images);
  write_tensor(buffer, input.control_image);
  write_tensor(buffer, input.masked_image_latents);
  write_tensor(buffer, input.prompt_embeds);
  write_tensor(buffer, input.pooled_prompt_embeds);
  write_tensor(buffer, input.negative_prompt_embeds);
  write_tensor(buffer, input.negative_pooled_prompt_embeds);
  write_tensor(buffer, input.latents);
  write_tensor(buffer, input.last_images);
  write_tensor(buffer, input.prompt_audio);
  write_string(buffer, input.audio_prompt_text);

  write_dit_generation_params(buffer, input.generation_params);
}

inline void write_dit_forward_input(RawInputSerializeContext& context,
                                    const DiTForwardInput& input) {
  write_data(context.descriptor, input.batch_size);

  write_string_vector(context.descriptor, input.prompts);
  write_string_vector(context.descriptor, input.prompts_2);
  write_string_vector(context.descriptor, input.negative_prompts);
  write_string_vector(context.descriptor, input.negative_prompts_2);

  write_tensor(context, input.images);
  write_vector_tensor(context, input.images_list);
  write_tensor(context, input.mask_images);
  write_tensor(context, input.control_image);
  write_tensor(context, input.masked_image_latents);
  write_tensor(context, input.prompt_embeds);
  write_tensor(context, input.pooled_prompt_embeds);
  write_tensor(context, input.negative_prompt_embeds);
  write_tensor(context, input.negative_pooled_prompt_embeds);
  write_tensor(context, input.latents);
  write_tensor(context, input.last_images);
  write_tensor(context, input.prompt_audio);
  write_string(context.descriptor, input.audio_prompt_text);

  write_dit_generation_params(context, input.generation_params);
}

inline void write_dit_forward_output(char*& buffer,
                                     const DiTForwardOutput& output) {
  write_vector_tensor(buffer, output.tensors);
  write_string_vector(buffer, output.text_output);
}

inline void safe_advance_buffer(const char*& buffer, size_t offset) {
  if (buffer != nullptr) {
    buffer += offset;
  }
}

struct TensorMeta final {
  std::vector<int64_t> shape;
  torch::ScalarType dtype = torch::kFloat;
  uint64_t data_bytes = 0;
};

struct DeviceBufferSession final {
  torch::Tensor owner_buffer;
  const char* device_cursor = nullptr;
  bool active = false;
  bool need_finalize_sync = false;
#if defined(USE_NPU)
  std::optional<std::unique_lock<std::mutex>> capture_lock_guard;
#elif defined(USE_CUDA)
  std::optional<std::shared_lock<std::shared_mutex>> capture_lock_guard;
#endif
};

struct ReadContext final {
  const char* descriptor_cursor;
  const char* tensor_cursor;
  DeviceBufferSession* device_session = nullptr;
};

inline uint64_t get_tensor_payload_bytes(const torch::Tensor& tensor) {
  if (!tensor.defined()) {
    return 0;
  }
  return tensor.numel() * tensor.element_size();
}

inline bool has_device_buffer(const ReadContext& context) {
  return context.device_session != nullptr && context.device_session->active;
}

inline void advance_descriptor_cursor(ReadContext& context, size_t offset) {
  context.descriptor_cursor += offset;
}

inline void advance_tensor_cursors(ReadContext& context, size_t offset) {
  context.tensor_cursor += offset;
  if (has_device_buffer(context)) {
    safe_advance_buffer(context.device_session->device_cursor, offset);
  }
}

template <typename T>
inline void read_data(const char*& buffer, T& data) {
  data = *reinterpret_cast<const T*>(buffer);
  buffer += type_size<T>;
}

template <typename T>
inline void read_data(ReadContext& context, T& data) {
  data = *reinterpret_cast<const T*>(context.descriptor_cursor);
  advance_descriptor_cursor(context, type_size<T>);
}

inline void read_linear_state_cache_ops(
    ReadContext& context,
    std::vector<LinearStateCacheOp>& cache_ops) {
  uint64_t size;
  read_data(context, size);
  cache_ops.resize(size);
  for (LinearStateCacheOp& cache_op : cache_ops) {
    read_data(context, cache_op.linear_state_id);
    read_data(context, cache_op.restore_requested);
    read_data(context, cache_op.restore_src_slot_id);
  }
}

template <typename T>
inline void read_data(const char*& buffer,
                      T& data,
                      const char*& device_buffer) {
  data = *reinterpret_cast<const T*>(buffer);
  buffer += type_size<T>;
  safe_advance_buffer(device_buffer, type_size<T>);
}

inline void read_string(const char*& buffer, std::string& str) {
  uint64_t len;
  read_data(buffer, len);
  if (len > 0) {
    str.assign(buffer, len);
    buffer += len;
  } else {
    str.clear();
  }
}

inline void read_string(const char*& buffer,
                        std::string& str,
                        const char*& device_buffer) {
  uint64_t len;
  read_data(buffer, len, device_buffer);
  if (len > 0) {
    str.assign(buffer, len);
    buffer += len;
    safe_advance_buffer(device_buffer, len);
  } else {
    str.clear();
  }
}

inline void read_string(ReadContext& context, std::string& str) {
  uint64_t len;
  read_data(context, len);
  if (len > 0) {
    str.assign(context.descriptor_cursor, len);
    advance_descriptor_cursor(context, len);
  } else {
    str.clear();
  }
}

inline void read_string_vector(const char*& buffer,
                               std::vector<std::string>& vec) {
  uint64_t size;
  read_data(buffer, size);
  vec.resize(size);
  for (uint64_t i = 0; i < size; ++i) {
    read_string(buffer, vec[i]);
  }
}

inline void read_string_vector(const char*& buffer,
                               std::vector<std::string>& vec,
                               const char*& device_buffer) {
  uint64_t size;
  read_data(buffer, size, device_buffer);
  vec.resize(size);
  for (uint64_t i = 0; i < size; ++i) {
    read_string(buffer, vec[i], device_buffer);
  }
}

inline void read_string_vector(ReadContext& context,
                               std::vector<std::string>& vec) {
  uint64_t size;
  read_data(context, size);
  vec.resize(size);
  for (uint64_t i = 0; i < size; ++i) {
    read_string(context, vec[i]);
  }
}

inline void read_tensor(const char*& buffer, torch::Tensor& tensor) {
  // read ndim
  uint64_t ndim;
  read_data(buffer, ndim);
  if (ndim == 0) {
    return;
  }
  // read shape
  std::vector<int64_t> shape(ndim);
  for (size_t i = 0; i < ndim; ++i) {
    int64_t dim_size;
    read_data(buffer, dim_size);
    shape[i] = static_cast<int64_t>(dim_size);
  }
  // read dtype
  int8_t tensor_dtype;
  read_data(buffer, tensor_dtype);
  auto dtype = static_cast<torch::ScalarType>(tensor_dtype);
  // read data_bytes
  uint64_t data_bytes;
  read_data(buffer, data_bytes);

  tensor = torch::from_blob(const_cast<void*>(static_cast<const void*>(buffer)),
                            shape,
                            torch::TensorOptions()
                                .dtype(dtype)
                                .device(torch::kCPU)
                                .pinned_memory(true));
  buffer += data_bytes;
}

inline TensorMeta read_tensor_meta(ReadContext& context) {
  TensorMeta meta;
  uint64_t ndim;
  read_data(context, ndim);
  if (ndim == 0) {
    return meta;
  }

  meta.shape.resize(ndim);
  for (size_t i = 0; i < ndim; ++i) {
    int64_t dim_size;
    read_data(context, dim_size);
    meta.shape[i] = static_cast<int64_t>(dim_size);
  }

  int8_t tensor_dtype;
  read_data(context, tensor_dtype);
  meta.dtype = static_cast<torch::ScalarType>(tensor_dtype);
  read_data(context, meta.data_bytes);
  return meta;
}

inline torch::Tensor materialize_tensor_from_current_cursor(
    const TensorMeta& meta,
    DeviceBufferSession& session,
    Stream* stream) {
  if (meta.data_bytes == 0) {
    torch::Device device = session.owner_buffer.defined()
                               ? session.owner_buffer.device()
                               : torch::Device(torch::kCPU);
    return torch::empty(
        meta.shape, torch::TensorOptions().dtype(meta.dtype).device(device));
  }
  const char* device_buffer = session.device_cursor;
#if defined(USE_NPU)
  return get_tensor_from_blob(meta.shape, meta.dtype, device_buffer);
#elif defined(USE_CUDA)
  if (session.owner_buffer.defined() &&
      is_aligned_for_cuda_zero_copy(device_buffer)) {
    return get_tensor_from_blob(
        meta.shape, meta.dtype, device_buffer, session.owner_buffer);
  }

  auto options = torch::TensorOptions().dtype(meta.dtype).device(torch::kCUDA);
  auto tensor = torch::empty(meta.shape, options);
  cudaError_t err;
  if (stream != nullptr) {
    err = cudaMemcpyAsync(tensor.data_ptr(),
                          device_buffer,
                          meta.data_bytes,
                          cudaMemcpyDeviceToDevice,
                          stream->get_stream()->stream());
  } else {
    err = cudaMemcpy(tensor.data_ptr(),
                     device_buffer,
                     meta.data_bytes,
                     cudaMemcpyDeviceToDevice);
  }
  CHECK_EQ(err, cudaSuccess)
      << "CUDA device buffer copy failed: " << cudaGetErrorString(err);
  return tensor;
#elif defined(USE_MLU)
  if (session.owner_buffer.defined()) {
    return get_tensor_from_blob(
        meta.shape, meta.dtype, device_buffer, session.owner_buffer);
  }
  return get_tensor_from_blob(meta.shape, meta.dtype, device_buffer);
#else
  LOG(FATAL) << "Unsupported device buffer backend";
#endif
}

inline void read_tensor(ReadContext& context,
                        torch::Tensor& tensor,
                        Stream* stream = nullptr,
                        bool force_host_materialize = false) {
  const TensorMeta meta = read_tensor_meta(context);
  if (meta.shape.empty()) {
    return;
  }

  if (!force_host_materialize && has_device_buffer(context)) {
    tensor = materialize_tensor_from_current_cursor(
        meta, *context.device_session, stream);
  } else {
    tensor = torch::from_blob(
        const_cast<void*>(static_cast<const void*>(context.tensor_cursor)),
        meta.shape,
        torch::TensorOptions()
            .dtype(meta.dtype)
            .device(torch::kCPU)
            .pinned_memory(true));
  }
  advance_tensor_cursors(context,
                         get_aligned_tensor_arena_bytes(meta.data_bytes));
}

inline void read_tensor_and_host(ReadContext& context,
                                 torch::Tensor& tensor,
                                 torch::Tensor& host_tensor,
                                 Stream* stream = nullptr) {
  const TensorMeta meta = read_tensor_meta(context);
  if (meta.shape.empty()) {
    return;
  }

  host_tensor = torch::from_blob(
      const_cast<void*>(static_cast<const void*>(context.tensor_cursor)),
      meta.shape,
      torch::TensorOptions()
          .dtype(meta.dtype)
          .device(torch::kCPU)
          .pinned_memory(true));
  if (has_device_buffer(context)) {
    tensor = materialize_tensor_from_current_cursor(
        meta, *context.device_session, stream);
  } else {
    tensor = host_tensor;
  }
  advance_tensor_cursors(context,
                         get_aligned_tensor_arena_bytes(meta.data_bytes));
}

inline void read_tensor(const char*& buffer,
                        torch::Tensor& tensor,
                        const char*& device_buffer,
                        Stream* stream = nullptr) {
  TensorMeta meta;
  uint64_t ndim;
  read_data(buffer, ndim, device_buffer);
  if (ndim == 0) {
    return;
  }

  meta.shape.resize(ndim);
  for (size_t i = 0; i < ndim; ++i) {
    int64_t dim_size;
    read_data(buffer, dim_size, device_buffer);
    meta.shape[i] = static_cast<int64_t>(dim_size);
  }

  int8_t tensor_dtype;
  read_data(buffer, tensor_dtype, device_buffer);
  meta.dtype = static_cast<torch::ScalarType>(tensor_dtype);
  read_data(buffer, meta.data_bytes, device_buffer);

  if (device_buffer != nullptr) {
    DeviceBufferSession device_session;
    device_session.device_cursor = device_buffer;
    device_session.active = true;
    tensor =
        materialize_tensor_from_current_cursor(meta, device_session, stream);
    safe_advance_buffer(device_buffer, meta.data_bytes);
  } else {
    tensor =
        torch::from_blob(const_cast<void*>(static_cast<const void*>(buffer)),
                         meta.shape,
                         torch::TensorOptions()
                             .dtype(meta.dtype)
                             .device(torch::kCPU)
                             .pinned_memory(true));
  }
  buffer += meta.data_bytes;
}

template <typename T>
inline void read_vector(const char*& buffer, std::vector<T>& vec) {
  uint64_t size;
  read_data(buffer, size);
  vec.resize(size);
  if (size > 0) {
    const size_t bytes = size * type_size<T>;
    std::memcpy(vec.data(), buffer, bytes);
    buffer += bytes;
  }
}

template <typename T>
inline void read_vector(const char*& buffer,
                        std::vector<T>& vec,
                        const char*& device_buffer) {
  uint64_t size;
  read_data(buffer, size, device_buffer);
  vec.resize(size);
  if (size > 0) {
    const size_t bytes = size * type_size<T>;
    std::memcpy(vec.data(), buffer, bytes);
    buffer += bytes;
    safe_advance_buffer(device_buffer, bytes);
  }
}

template <typename T>
inline void read_vector(ReadContext& context, std::vector<T>& vec) {
  uint64_t size;
  read_data(context, size);
  vec.resize(size);
  if (size > 0) {
    const size_t bytes = size * type_size<T>;
    std::memcpy(vec.data(), context.descriptor_cursor, bytes);
    advance_descriptor_cursor(context, bytes);
  }
}

template <typename T>
inline void read_tensor_and_vector(ReadContext& context,
                                   torch::Tensor& tensor,
                                   std::vector<T>& vec,
                                   Stream* stream = nullptr) {
  const TensorMeta meta = read_tensor_meta(context);
  if (meta.shape.empty()) {
    return;
  }

  vec.resize(meta.shape[0]);
  if (has_device_buffer(context)) {
    tensor = materialize_tensor_from_current_cursor(
        meta, *context.device_session, stream);
  } else {
    tensor = torch::from_blob(
        const_cast<void*>(static_cast<const void*>(context.tensor_cursor)),
        meta.shape,
        torch::TensorOptions()
            .dtype(meta.dtype)
            .device(torch::kCPU)
            .pinned_memory(true));
  }
  if (meta.data_bytes > 0) {
    std::memcpy(vec.data(), context.tensor_cursor, meta.data_bytes);
  }
  advance_tensor_cursors(context,
                         get_aligned_tensor_arena_bytes(meta.data_bytes));
}

template <typename T>
inline void read_2d_vector(const char*& buffer,
                           std::vector<std::vector<T>>& vec2d) {
  uint64_t size;
  read_data(buffer, size);
  vec2d.resize(size);
  for (auto& vec : vec2d) {
    read_vector(buffer, vec);
  }
}

inline void read_sampling_param(const char*& buffer,
                                RequestSamplingParam& param) {
  const char* ptr = buffer;
  param.frequency_penalty = *reinterpret_cast<const float*>(ptr);
  ptr += type_size<float>;
  param.presence_penalty = *reinterpret_cast<const float*>(ptr);
  ptr += type_size<float>;
  param.repetition_penalty = *reinterpret_cast<const float*>(ptr);
  ptr += type_size<float>;
  param.temperature = *reinterpret_cast<const float*>(ptr);
  ptr += type_size<float>;
  param.top_p = *reinterpret_cast<const float*>(ptr);
  ptr += type_size<float>;
  param.top_k = *reinterpret_cast<const int64_t*>(ptr);
  ptr += type_size<int64_t>;
  param.logprobs = *reinterpret_cast<const bool*>(ptr);
  ptr += type_size<bool>;
  param.top_logprobs = *reinterpret_cast<const int64_t*>(ptr);
  ptr += type_size<int64_t>;
  param.do_sample = *reinterpret_cast<const bool*>(ptr);
  ptr += type_size<bool>;
  param.is_embeddings = *reinterpret_cast<const bool*>(ptr);
  ptr += type_size<bool>;
  param.beam_width = *reinterpret_cast<const int32_t*>(ptr);
  ptr += type_size<int32_t>;
  param.num_return_sequences = *reinterpret_cast<const int32_t*>(ptr);
  ptr += type_size<int32_t>;
  buffer = ptr;
}

inline void read_instance_info(const char*& buffer, InstanceInfo& info) {
  read_string(buffer, info.name);
  read_string(buffer, info.rpc_address);
  read_string(buffer, info.type);

  read_vector(buffer, info.cluster_ids);

  uint64_t addr_count;
  read_data(buffer, addr_count);
  info.addrs.resize(addr_count);
  for (auto& addr : info.addrs) {
    read_string(buffer, addr);
  }

  read_data(buffer, info.dp_size);
  read_data(buffer, info.kv_split_size);

  uint64_t prof_size;
  read_data(buffer, prof_size);
  info.ttft_profiling_data.resize(prof_size);
  if (prof_size > 0) {
    std::memcpy(info.ttft_profiling_data.data(),
                buffer,
                prof_size * sizeof(std::pair<int32_t, int64_t>));
    buffer += prof_size * sizeof(std::pair<int32_t, int64_t>);
  }
}

inline void read_instance_info(ReadContext& context, InstanceInfo& info) {
  read_string(context, info.name);
  read_string(context, info.rpc_address);
  read_string(context, info.type);

  read_vector(context, info.cluster_ids);

  uint64_t addr_count;
  read_data(context, addr_count);
  info.addrs.resize(addr_count);
  for (auto& addr : info.addrs) {
    read_string(context, addr);
  }

  read_data(context, info.dp_size);
  read_data(context, info.kv_split_size);

  uint64_t prof_size;
  read_data(context, prof_size);
  info.ttft_profiling_data.resize(prof_size);
  if (prof_size > 0) {
    std::memcpy(info.ttft_profiling_data.data(),
                context.descriptor_cursor,
                prof_size * sizeof(std::pair<int32_t, int64_t>));
    advance_descriptor_cursor(context,
                              prof_size * sizeof(std::pair<int32_t, int64_t>));
  }
}

inline void read_xtensor_layer_offsets(
    const char*& buffer,
    std::vector<XTensorLayerOffsets>& offsets) {
  uint64_t num_layers;
  read_data(buffer, num_layers);
  offsets.resize(num_layers);
  for (auto& layer : offsets) {
    read_vector(buffer, layer.k_offsets);
    read_vector(buffer, layer.v_offsets);
  }
}

inline void read_block_transfer_groups(
    const char*& buffer,
    std::vector<KVBlockTransferGroup>& groups) {
  uint64_t group_count;
  read_data(buffer, group_count);
  groups.resize(group_count);
  for (auto& group : groups) {
    read_data(buffer, group.group_id);
    read_vector(buffer, group.local_blocks_ids);
    read_vector(buffer, group.remote_blocks_ids);
  }
}

inline void read_block_transfer_groups(
    ReadContext& context,
    std::vector<KVBlockTransferGroup>& groups) {
  uint64_t group_count;
  read_data(context, group_count);
  groups.resize(group_count);
  for (auto& group : groups) {
    read_data(context, group.group_id);
    read_vector(context, group.local_blocks_ids);
    read_vector(context, group.remote_blocks_ids);
  }
}

inline void read_xtensor_layer_offsets(
    ReadContext& context,
    std::vector<XTensorLayerOffsets>& offsets) {
  uint64_t num_layers;
  read_data(context, num_layers);
  offsets.resize(num_layers);
  for (auto& layer : offsets) {
    read_vector(context, layer.k_offsets);
    read_vector(context, layer.v_offsets);
  }
}

inline void read_transfer_kv_info(const char*& buffer, TransferKVInfo& info) {
  read_string(buffer, info.request_id);
  read_vector(buffer, info.local_blocks_ids);
  read_vector(buffer, info.remote_blocks_ids);
  read_vector(buffer, info.local_linear_state_ids);
  read_vector(buffer, info.remote_linear_state_ids);
  read_data(buffer, info.dp_rank);
  read_instance_info(buffer, info.remote_instance_info);
  read_xtensor_layer_offsets(buffer, info.dst_xtensor_layer_offsets);
  read_block_transfer_groups(buffer, info.block_transfer_groups);
}

inline void read_transfer_kv_info(ReadContext& context, TransferKVInfo& info) {
  read_string(context, info.request_id);
  read_vector(context, info.local_blocks_ids);
  read_vector(context, info.remote_blocks_ids);
  read_vector(context, info.local_linear_state_ids);
  read_vector(context, info.remote_linear_state_ids);
  read_data(context, info.dp_rank);
  read_instance_info(context, info.remote_instance_info);
  read_xtensor_layer_offsets(context, info.dst_xtensor_layer_offsets);
  read_block_transfer_groups(context, info.block_transfer_groups);
}

inline void read_eplb_info(const char*& buffer, EplbInfo& info) {
  read_data(buffer, info.prepare_layer_id);
  read_vector(buffer, info.expert_ids);
  read_data(buffer, info.update_layer_id);
}

inline void read_eplb_info(ReadContext& context, EplbInfo& info) {
  read_data(context, info.prepare_layer_id);
  read_vector(context, info.expert_ids);
  read_data(context, info.update_layer_id);
}

inline void read_swap_blocks(const char*& buffer,
                             std::vector<BlockTransferInfo>& blocks,
                             const char*& device_buffer) {
  uint64_t size;
  read_data(buffer, size, device_buffer);
  blocks.reserve(size);

  int32_t src_block_id;
  int32_t dst_block_id;
  for (int i = 0; i < size; i++) {
    read_data(buffer, src_block_id, device_buffer);
    read_data(buffer, dst_block_id, device_buffer);
    blocks.emplace_back(src_block_id, dst_block_id);
  }
}

inline void read_swap_blocks(ReadContext& context,
                             std::vector<BlockTransferInfo>& blocks) {
  uint64_t size;
  read_data(context, size);
  blocks.reserve(size);

  int32_t src_block_id;
  int32_t dst_block_id;
  for (uint64_t i = 0; i < size; ++i) {
    read_data(context, src_block_id);
    read_data(context, dst_block_id);
    blocks.emplace_back(src_block_id, dst_block_id);
  }
}

inline void read_vector_tensor(const char*& buffer,
                               std::vector<torch::Tensor>& tensor_vec) {
  int32_t tensor_num;
  read_data(buffer, tensor_num);
  tensor_vec.resize(tensor_num);
  for (size_t i = 0; i < tensor_num; ++i) {
    read_tensor(buffer, tensor_vec[i]);
  }
}

inline void read_vector_tensor(ReadContext& context,
                               std::vector<torch::Tensor>& tensor_vec,
                               Stream* stream = nullptr,
                               bool force_host_materialize = false) {
  int32_t tensor_num;
  read_data(context, tensor_num);
  tensor_vec.resize(tensor_num);
  for (size_t i = 0; i < tensor_num; ++i) {
    read_tensor(context, tensor_vec[i], stream, force_host_materialize);
  }
}

inline void read_mm_dict(const char*& buffer,
                         MMDict& mm_dict,
                         const char*& device_buffer) {
  size_t size;
  read_data(buffer, size, device_buffer);
  int32_t tensor_num;
  while (size--) {
    std::string mm_key;
    read_string(buffer, mm_key, device_buffer);
    read_data(buffer, tensor_num, device_buffer);
    if (tensor_num == 1) {
      torch::Tensor tensor;
      read_tensor(buffer, tensor, device_buffer);
      mm_dict[mm_key] = tensor;
    } else {
      std::vector<torch::Tensor> tensor_vec(tensor_num);
      for (size_t i = 0; i < tensor_num; ++i) {
        read_tensor(buffer, tensor_vec[i], device_buffer);
      }
      mm_dict[mm_key] = tensor_vec;
    }
  }
}

inline void read_mm_dict(ReadContext& context, MMDict& mm_dict) {
  size_t size;
  read_data(context, size);
  int32_t tensor_num;
  while (size--) {
    std::string mm_key;
    read_string(context, mm_key);
    read_data(context, tensor_num);
    if (tensor_num == 1) {
      torch::Tensor tensor;
      read_tensor(context, tensor);
      mm_dict[mm_key] = tensor;
    } else {
      std::vector<torch::Tensor> tensor_vec(tensor_num);
      for (int32_t i = 0; i < tensor_num; ++i) {
        read_tensor(context, tensor_vec[i]);
      }
      mm_dict[mm_key] = tensor_vec;
    }
  }
}

inline void read_mm_item(const char*& buffer,
                         MMDataItem& item,
                         const char*& device_buffer) {
  uint32_t type;
  read_data(buffer, type, device_buffer);
  MMDict dict;
  read_mm_dict(buffer, dict, device_buffer);
  auto mm_type_value = static_cast<MMType::Value>(type);
  item = std::move(MMDataItem(mm_type_value, dict));
  auto& state = item.mutable_state();
  read_data(buffer, state.mutable_seq_index(), device_buffer);

  // read token_pos
  read_data(buffer, state.mutable_token_pos().offset, device_buffer);
  read_data(buffer, state.mutable_token_pos().length, device_buffer);
  read_data(buffer, state.mutable_mm_token_num(), device_buffer);

  // read mm_token_mask
  read_tensor(buffer, state.mutable_mm_token_mask(), device_buffer);

  // read schedule_data
  std::memcpy(state.mutable_schedule_data().key.data,
              buffer,
              XXH3_128BITS_HASH_VALUE_LEN);
  buffer += XXH3_128BITS_HASH_VALUE_LEN;
  safe_advance_buffer(device_buffer, XXH3_128BITS_HASH_VALUE_LEN);
  read_data(buffer, state.mutable_schedule_data().start_pos, device_buffer);
  read_data(buffer, state.mutable_schedule_data().end_pos, device_buffer);
}

inline void read_mm_item(ReadContext& context, MMDataItem& item) {
  uint32_t type;
  read_data(context, type);
  MMDict dict;
  read_mm_dict(context, dict);
  auto mm_type_value = static_cast<MMType::Value>(type);
  item = std::move(MMDataItem(mm_type_value, dict));
  auto& state = item.mutable_state();
  read_data(context, state.mutable_seq_index());

  read_data(context, state.mutable_token_pos().offset);
  read_data(context, state.mutable_token_pos().length);
  read_data(context, state.mutable_mm_token_num());

  read_tensor(context, state.mutable_mm_token_mask());

  std::memcpy(state.mutable_schedule_data().key.data,
              context.descriptor_cursor,
              XXH3_128BITS_HASH_VALUE_LEN);
  advance_descriptor_cursor(context, XXH3_128BITS_HASH_VALUE_LEN);
  read_data(context, state.mutable_schedule_data().start_pos);
  read_data(context, state.mutable_schedule_data().end_pos);
}

inline void read_mm_data_dict(const char*& buffer,
                              MMData& mm_data,
                              const char*& device_buffer) {
  uint32_t mm_type;
  read_data(buffer, mm_type, device_buffer);
  MMDict mm_dict;
  read_mm_dict(buffer, mm_dict, device_buffer);
  MMType ty{static_cast<MMType::Value>(mm_type)};
  mm_data = MMData(ty, mm_dict);
}

inline void read_mm_data_dict(ReadContext& context, MMData& mm_data) {
  uint32_t mm_type;
  read_data(context, mm_type);
  MMDict mm_dict;
  read_mm_dict(context, mm_dict);
  MMType ty{static_cast<MMType::Value>(mm_type)};
  mm_data = MMData(ty, mm_dict);
}

inline void read_mm_data_items(const char*& buffer,
                               MMData& mm_data,
                               const char*& device_buffer) {
  uint32_t mm_type;
  read_data(buffer, mm_type, device_buffer);
  size_t mm_items_num;
  read_data(buffer, mm_items_num, device_buffer);
  MMItemVec mm_items;
  mm_items.reserve(mm_items_num);
  for (size_t idx = 0; idx < mm_items_num; ++idx) {
    mm_items.emplace_back(MMType::NONE);
    read_mm_item(buffer, mm_items.back(), device_buffer);
  }
  MMType ty{static_cast<MMType::Value>(mm_type)};
  mm_data = MMData(ty, std::move(mm_items));
}

inline void read_mm_data_items(ReadContext& context, MMData& mm_data) {
  uint32_t mm_type;
  read_data(context, mm_type);
  size_t mm_items_num;
  read_data(context, mm_items_num);
  MMItemVec mm_items;
  mm_items.reserve(mm_items_num);
  for (size_t idx = 0; idx < mm_items_num; ++idx) {
    mm_items.emplace_back(MMType::NONE);
    read_mm_item(context, mm_items.back());
  }
  MMType ty{static_cast<MMType::Value>(mm_type)};
  mm_data = MMData(ty, std::move(mm_items));
}

inline void read_mm_batch_data(const char*& buffer,
                               MMBatchData& batch_mm_data,
                               const char*& device_buffer) {
  using ReadMmDataFn = void (*)(const char*&, MMData&, const char*&);

  std::vector<MMData> vec;

  size_t mm_data_num;
  read_data(buffer, mm_data_num, device_buffer);
  uint8_t is_mm_item;
  read_data(buffer, is_mm_item, device_buffer);
  vec.reserve(mm_data_num);
  ReadMmDataFn read_mm_data =
      is_mm_item ? static_cast<ReadMmDataFn>(&read_mm_data_items)
                 : static_cast<ReadMmDataFn>(&read_mm_data_dict);
  for (size_t i = 0; i < mm_data_num; ++i) {
    vec.emplace_back();
    read_mm_data(buffer, vec.back(), device_buffer);
  }

  batch_mm_data.batch(std::move(vec));
}

inline void read_mm_batch_data(ReadContext& context,
                               MMBatchData& batch_mm_data) {
  using ReadMmDataFn = void (*)(ReadContext&, MMData&);

  std::vector<MMData> vec;

  size_t mm_data_num;
  read_data(context, mm_data_num);
  uint8_t is_mm_item;
  read_data(context, is_mm_item);
  vec.reserve(mm_data_num);
  ReadMmDataFn read_mm_data =
      is_mm_item ? static_cast<ReadMmDataFn>(&read_mm_data_items)
                 : static_cast<ReadMmDataFn>(&read_mm_data_dict);
  for (size_t i = 0; i < mm_data_num; ++i) {
    vec.emplace_back();
    read_mm_data(context, vec.back());
  }

  batch_mm_data.batch(std::move(vec));
}

// read dit data
inline void read_dit_generation_params(const char*& buffer,
                                       DiTGenerationParams& params) {
  read_data(buffer, params.width);
  read_data(buffer, params.height);
  read_data(buffer, params.num_inference_steps);
  read_data(buffer, params.true_cfg_scale);
  read_data(buffer, params.guidance_scale);
  read_data(buffer, params.num_images_per_prompt);
  read_data(buffer, params.seed);
  read_data(buffer, params.max_sequence_length);
  read_data(buffer, params.strength);
  read_data(buffer, params.enable_cfg_renorm);
  read_data(buffer, params.cfg_renorm_min);
  read_data(buffer, params.num_frames);
  read_data(buffer, params.force_video_output);
  read_data(buffer, params.video_fps);
  read_data(buffer, params.guidance_scale_2);
  read_data(buffer, params.seconds);
  read_data(buffer, params.boundary_ratio);
  read_data(buffer, params.flow_shift);
  read_data(buffer, params.num_videos_per_prompt);
  read_data(buffer, params.max_new_tokens);
  read_data(buffer, params.diffusion_steps);
  read_data(buffer, params.temperature);
  read_data(buffer, params.top_k);
  read_data(buffer, params.top_p);
  read_data(buffer, params.repetition_penalty);
  read_data(buffer, params.seed_is_set);
}

inline void read_dit_generation_params(ReadContext& context,
                                       DiTGenerationParams& params) {
  read_data(context, params.width);
  read_data(context, params.height);
  read_data(context, params.num_inference_steps);
  read_data(context, params.true_cfg_scale);
  read_data(context, params.guidance_scale);
  read_data(context, params.num_images_per_prompt);
  read_data(context, params.seed);
  read_data(context, params.max_sequence_length);
  read_data(context, params.strength);
  read_data(context, params.enable_cfg_renorm);
  read_data(context, params.cfg_renorm_min);
  read_data(context, params.num_frames);
  read_data(context, params.force_video_output);
  read_data(context, params.video_fps);
  read_data(context, params.guidance_scale_2);
  read_data(context, params.seconds);
  read_data(context, params.boundary_ratio);
  read_data(context, params.flow_shift);
  read_data(context, params.num_videos_per_prompt);
  read_data(context, params.max_new_tokens);
  read_data(context, params.diffusion_steps);
  read_data(context, params.temperature);
  read_data(context, params.top_k);
  read_data(context, params.top_p);
  read_data(context, params.repetition_penalty);
  read_data(context, params.seed_is_set);
}

inline void clone_tensor_if_defined(torch::Tensor& tensor) {
  if (tensor.defined()) {
    tensor = tensor.clone();
  }
}

inline void clone_vector_tensor_if_defined(
    std::vector<torch::Tensor>& tensors) {
  for (auto& tensor : tensors) {
    clone_tensor_if_defined(tensor);
  }
}

inline void stabilize_dit_forward_input_tensors(DiTForwardInput& input) {
  clone_tensor_if_defined(input.images);
  clone_vector_tensor_if_defined(input.images_list);
  clone_tensor_if_defined(input.mask_images);
  clone_tensor_if_defined(input.control_image);
  clone_tensor_if_defined(input.masked_image_latents);
  clone_tensor_if_defined(input.prompt_embeds);
  clone_tensor_if_defined(input.pooled_prompt_embeds);
  clone_tensor_if_defined(input.negative_prompt_embeds);
  clone_tensor_if_defined(input.negative_pooled_prompt_embeds);
  clone_tensor_if_defined(input.latents);
  clone_tensor_if_defined(input.last_images);
  clone_tensor_if_defined(input.prompt_audio);
}

inline void read_dit_forward_input(const char*& buffer,
                                   DiTForwardInput& input,
                                   bool stabilize_host_tensors = false) {
  read_data(buffer, input.batch_size);

  read_string_vector(buffer, input.prompts);
  read_string_vector(buffer, input.prompts_2);
  read_string_vector(buffer, input.negative_prompts);
  read_string_vector(buffer, input.negative_prompts_2);

  read_tensor(buffer, input.images);
  read_vector_tensor(buffer, input.images_list);
  read_tensor(buffer, input.mask_images);
  read_tensor(buffer, input.control_image);
  read_tensor(buffer, input.masked_image_latents);
  read_tensor(buffer, input.prompt_embeds);
  read_tensor(buffer, input.pooled_prompt_embeds);
  read_tensor(buffer, input.negative_prompt_embeds);
  read_tensor(buffer, input.negative_pooled_prompt_embeds);
  read_tensor(buffer, input.latents);
  read_tensor(buffer, input.last_images);
  read_tensor(buffer, input.prompt_audio);
  read_string(buffer, input.audio_prompt_text);

  read_dit_generation_params(buffer, input.generation_params);
  if (stabilize_host_tensors) {
    stabilize_dit_forward_input_tensors(input);
  }
}

inline void read_dit_forward_input(ReadContext& context,
                                   DiTForwardInput& input,
                                   bool stabilize_host_tensors = false) {
  read_data(context, input.batch_size);

  read_string_vector(context, input.prompts);
  read_string_vector(context, input.prompts_2);
  read_string_vector(context, input.negative_prompts);
  read_string_vector(context, input.negative_prompts_2);

  read_tensor(context,
              input.images,
              /*stream=*/nullptr,
              /*force_host_materialize=*/true);
  read_vector_tensor(context,
                     input.images_list,
                     /*stream=*/nullptr,
                     /*force_host_materialize=*/true);
  read_tensor(context,
              input.mask_images,
              /*stream=*/nullptr,
              /*force_host_materialize=*/true);
  read_tensor(context,
              input.control_image,
              /*stream=*/nullptr,
              /*force_host_materialize=*/true);
  read_tensor(context,
              input.masked_image_latents,
              /*stream=*/nullptr,
              /*force_host_materialize=*/true);
  read_tensor(context,
              input.prompt_embeds,
              /*stream=*/nullptr,
              /*force_host_materialize=*/true);
  read_tensor(context,
              input.pooled_prompt_embeds,
              /*stream=*/nullptr,
              /*force_host_materialize=*/true);
  read_tensor(context,
              input.negative_prompt_embeds,
              /*stream=*/nullptr,
              /*force_host_materialize=*/true);
  read_tensor(context,
              input.negative_pooled_prompt_embeds,
              /*stream=*/nullptr,
              /*force_host_materialize=*/true);
  read_tensor(context,
              input.latents,
              /*stream=*/nullptr,
              /*force_host_materialize=*/true);
  read_tensor(context,
              input.last_images,
              /*stream=*/nullptr,
              /*force_host_materialize=*/true);
  read_tensor(context,
              input.prompt_audio,
              /*stream=*/nullptr,
              /*force_host_materialize=*/true);
  read_string(context, input.audio_prompt_text);

  read_dit_generation_params(context, input.generation_params);
  if (stabilize_host_tensors) {
    stabilize_dit_forward_input_tensors(input);
  }
}

inline void read_dit_forward_output(const char*& buffer,
                                    DiTForwardOutput& output) {
  read_vector_tensor(buffer, output.tensors);
  for (auto& tensor : output.tensors) {
    if (tensor.defined()) {
      tensor = tensor.clone();
    }
  }
  read_string_vector(buffer, output.text_output);
}

inline void initialize_device_buffer_session(ReadContext& context,
                                             ForwardInput& forward_input,
                                             const torch::Device& device,
                                             const char* payload_base,
                                             const uint64_t payload_size,
                                             const uint64_t tensor_arena_offset,
                                             Stream* stream,
                                             bool materialize_device_buffer) {
  if (context.device_session == nullptr) {
    return;
  }

#if defined(USE_NPU) || defined(USE_CUDA) || defined(USE_MLU)
  if (!materialize_device_buffer ||
      !::xllm::ExecutionConfig::get_instance().use_contiguous_input_buffer()) {
    return;
  }

#if defined(USE_CUDA)
  if (device.type() != torch::kCUDA) {
    return;
  }
#elif defined(USE_MLU)
  if (device.type() != torch::kPrivateUse1) {
    return;
  }
#endif

  auto& session = *context.device_session;
  torch::Tensor host_input_buffer =
      torch::from_blob(const_cast<char*>(payload_base),
                       {static_cast<int64_t>(payload_size)},
                       torch::TensorOptions()
                           .dtype(torch::kUInt8)
                           .device(torch::kCPU)
                           .pinned_memory(/*pinned_memory=*/true));

  auto device_options =
      torch::TensorOptions().dtype(torch::kUInt8).device(device);

  if (stream != nullptr) {
#if defined(USE_NPU)
    auto& capture_lock =
        ::xllm::npu::DeviceCaptureLock::get_instance().get_lock(device.index());
    session.capture_lock_guard.emplace(capture_lock);
#elif defined(USE_CUDA)
    if (::xllm::ExecutionConfig::get_instance().enable_graph()) {
      auto& replay_lock =
          ::xllm::cuda::DeviceCaptureLock::get_instance().get_read_lock(
              device.index());
      session.capture_lock_guard.emplace(replay_lock);
    }
#endif
    c10::StreamGuard stream_guard = stream->set_stream_guard();
    forward_input.device_input_buffer =
        safe_to(host_input_buffer, device_options, /*non_blocking=*/true);
  } else {
    forward_input.device_input_buffer =
        safe_to(host_input_buffer, device_options);
  }

  session.owner_buffer = forward_input.device_input_buffer;
  session.device_cursor =
      static_cast<const char*>(forward_input.device_input_buffer.data_ptr()) +
      tensor_arena_offset;
  session.active = session.device_cursor != nullptr;
  session.need_finalize_sync = session.active && stream != nullptr;
#else
  (void)context;
  (void)forward_input;
  (void)device;
  (void)payload_base;
  (void)payload_size;
  (void)tensor_arena_offset;
  (void)stream;
  (void)materialize_device_buffer;
#endif
}

inline void finalize_device_buffer_session(DeviceBufferSession& session,
                                           Stream* stream) {
#if defined(USE_NPU) || defined(USE_CUDA) || defined(USE_MLU)
  if (session.need_finalize_sync && stream != nullptr) {
    stream->synchronize();
  }
#else
  (void)session;
  (void)stream;
#endif
}

inline void deserialize_forward_input_payload(
    const char*& buffer,
    const uint64_t buffer_size,
    ForwardInput& forward_input,
    const torch::Device& device,
    Stream* stream,
    bool materialize_device_buffer = true,
    bool stabilize_dit_host_tensors = false) {
  const char* payload_base = buffer;
  RawInputLayoutHeader layout;
  read_data(buffer, layout.descriptor_bytes);
  read_data(buffer, layout.tensor_arena_offset);
  read_data(buffer, layout.tensor_arena_bytes);
  CHECK_GE(buffer_size, sizeof(RawInputLayoutHeader))
      << "raw input layout header overflow";
  CHECK_GE(layout.tensor_arena_offset,
           sizeof(RawInputLayoutHeader) + layout.descriptor_bytes)
      << "raw input tensor arena overlaps descriptor";
  CHECK_LE(layout.tensor_arena_offset + layout.tensor_arena_bytes, buffer_size)
      << "raw input layout overflow";
  CHECK_EQ(layout.tensor_arena_offset % kRawInputTensorArenaAlignment, 0)
      << "raw input tensor arena offset is not aligned";

  DeviceBufferSession device_session;
  const char* descriptor_base = buffer;
  const char* tensor_arena_base = payload_base + layout.tensor_arena_offset;
  ReadContext context{descriptor_base, tensor_arena_base, &device_session};
  initialize_device_buffer_session(context,
                                   forward_input,
                                   device,
                                   payload_base,
                                   buffer_size,
                                   layout.tensor_arena_offset,
                                   stream,
                                   materialize_device_buffer);

  read_tensor_and_host(
      context, forward_input.token_ids, forward_input.token_ids_host, stream);
  read_tensor_and_host(
      context, forward_input.positions, forward_input.positions_host, stream);

  // input_params
  auto& input_params = forward_input.input_params;
  int32_t batch_forward_type;
  read_data(context, batch_forward_type);
  input_params.meta.batch_forward_type = BatchForwardType(batch_forward_type);
  read_data(context, input_params.meta.num_sequences);
  read_data(context, input_params.meta.kv_max_seq_len);
  read_data(context, input_params.meta.q_max_seq_len);
  read_data(context, input_params.meta.batch_id);
  read_tensor_and_vector(context,
                         input_params.attention.device.q_seq_lens,
                         input_params.attention.host.q_seq_lens,
                         stream);
  read_tensor_and_vector(context,
                         input_params.attention.device.q_cu_seq_lens,
                         input_params.attention.host.q_cu_seq_lens,
                         stream);
  CHECK_EQ(input_params.attention.host.q_seq_lens.empty(),
           input_params.attention.host.q_cu_seq_lens.empty())
      << "q_seq_lens and q_cu_seq_lens must be provided together";
  read_tensor_and_vector(context,
                         input_params.attention.device.kv_seq_lens,
                         input_params.attention.host.kv_seq_lens,
                         stream);
  read_tensor(context, input_params.attention.device.paged_kv_indptr, stream);
  read_tensor(context, input_params.attention.device.paged_kv_indices, stream);
  read_tensor(
      context, input_params.attention.device.paged_kv_last_page_len, stream);
  read_tensor(
      context, input_params.attention.device.new_cache_slot_offsets, stream);
  read_tensor(
      context, input_params.attention.device.kv_cache_start_offsets, stream);
  read_tensor(context, input_params.embedding.input_embedding, stream);
  read_vector(context, input_params.parallel.dp_global_token_nums);
  read_vector(context, input_params.parallel.dp_is_decode);
  read_vector(context, input_params.embedding.embedding_ids);
  read_vector(context, input_params.embedding.linear_state_ids);
  read_linear_state_cache_ops(context, input_params.linear_state_cache_ops);
  normalize_linear_state_ids(input_params.embedding.linear_state_ids,
                             input_params.meta.num_sequences);
  if (!input_params.embedding.linear_state_ids.empty()) {
    input_params.embedding.linear_state_indices =
        torch::tensor(input_params.embedding.linear_state_ids, torch::kInt)
            .to(device, /*non_blocking=*/true);
  }
  read_string_vector(context, input_params.embedding.request_ids);
  read_vector(context, input_params.embedding.extra_token_ids);
  // Keep upstream's root mtp_shifted_token_ids serialization (consumed by
  // non-CP MTP paths + minimax / qwen3-next models). The CP path additionally
  // reads embedding.mtp_shifted_token_ids below. shared_blocks_num was
  // dropped by the CP polish commit (no live worker consumer).
  read_tensor(context, input_params.mtp_shifted_token_ids, stream);
  read_tensor(context, input_params.embedding.mtp_shifted_token_ids, stream);
  read_vector(context, input_params.embedding.mtp_bootstrap_row_idxes);
  read_tensor(context, input_params.embedding.mtp_bootstrap_embeddings, stream);
  read_swap_blocks(context, input_params.block_copy.swap_blocks);
  read_tensor(context, input_params.block_copy.src_block_indices, stream);
  read_tensor(context, input_params.block_copy.dst_block_indices, stream);
  read_tensor(context, input_params.block_copy.cum_sum, stream);
  read_mm_batch_data(context, input_params.multimodal.mm_data);
  read_tensor_and_vector(context,
                         input_params.attention.device.kv_cache_tokens_nums,
                         input_params.attention.host.kv_cache_tokens_nums,
                         stream);

  // sampling_params
  uint64_t selected_token_idxes_size;
  read_data(context, selected_token_idxes_size);
  if (selected_token_idxes_size > 0) {
    auto& sampling_params = forward_input.sampling_params;
    read_tensor(context, sampling_params.selected_token_idxes, stream);
    read_tensor(context, sampling_params.frequency_penalties, stream);
    read_tensor(context, sampling_params.presence_penalties, stream);
    read_tensor(context, sampling_params.repetition_penalties, stream);
    read_tensor(context, sampling_params.temperatures, stream);
    read_tensor(context, sampling_params.top_p, stream);
    read_tensor(context, sampling_params.top_k, stream);
    read_tensor(context, sampling_params.unique_token_ids, stream);
    read_tensor(context, sampling_params.unique_token_counts, stream);
    read_tensor(context, sampling_params.unique_token_ids_lens, stream);
    read_tensor(context, sampling_params.sample_idxes, stream);
    read_tensor(context, sampling_params.do_sample, stream);
    read_data(context, sampling_params.all_random_sample);
    read_data(context, sampling_params.all_greedy_sample);
    read_data(context, sampling_params.logprobs);
    read_data(context, sampling_params.is_embeddings);
    read_data(context, sampling_params.max_top_logprobs);
    read_data(context, sampling_params.num_return_sequences);
    read_data(context, sampling_params.use_beam_search);
  }
  // acc_logprob
  read_tensor(context, forward_input.sampling_params.acc_logprob, stream);

  // Keep transfer/eplb host-materialized, but continue advancing the
  // device cursor when a contiguous device buffer is active.
  uint64_t transfer_count;
  read_data(context, transfer_count);
  forward_input.transfer_kv_infos.resize(transfer_count);
  for (auto& transfer : forward_input.transfer_kv_infos) {
    read_transfer_kv_info(context, transfer);
  }
  read_eplb_info(context, forward_input.input_params.expert.eplb_info);

  read_tensor_and_vector(context,
                         input_params.attention.device.new_cache_slots,
                         input_params.attention.host.new_cache_slots,
                         stream);
  read_tensor_and_host(context,
                       input_params.attention.device.block_tables,
                       input_params.attention.host.block_tables,
                       stream);
  int32_t manager_num = 0;
  read_data(context, manager_num);
  CHECK_GE(manager_num, 0) << "multi_block_tables manager num is invalid.";
  input_params.multi_block_tables.reserve(static_cast<size_t>(manager_num));
  for (int32_t i = 0; i < manager_num; ++i) {
    torch::Tensor manager_table;
    read_tensor(context,
                manager_table,
                /*stream=*/nullptr,
                /*force_host_materialize=*/true);
    // Clone to decouple from shared memory buffer. With schedule_overlap the
    // buffer may be overwritten by the next step while the worker is still
    // reading multi_block_tables (which stay on CPU for DSA metadata).
    input_params.multi_block_tables.emplace_back(manager_table.clone());
  }

  bool has_dit_forward_input = false;
  read_data(context, has_dit_forward_input);
  if (has_dit_forward_input) {
    read_dit_forward_input(
        context, input_params.dit_forward_input, stabilize_dit_host_tensors);
  }

  finalize_device_buffer_session(device_session, stream);
  forward_input.input_host_buffer_has_layout = true;
  if (materialize_device_buffer &&
      forward_input.device_input_buffer.defined()) {
    forward_input.device_tensors_ready = true;
  }
  buffer = payload_base + buffer_size;
}

inline uint64_t get_input_layout_size(const RawInputLayoutHeader& layout) {
  return layout.tensor_arena_offset + layout.tensor_arena_bytes;
}

void packed_proto_to_forward_input_impl(
    const proto::PackedForwardInput& packed_forward_input,
    ForwardInput& forward_input,
    const torch::Device& device,
    Stream* stream) {
  (void)device;
  (void)stream;
  const std::string& payload = packed_forward_input.payload();
  if (payload.empty()) {
    return;
  }

  forward_input.input_host_buffer =
      torch::empty({static_cast<int64_t>(payload.size())},
                   torch::TensorOptions()
                       .dtype(torch::kUInt8)
                       .device(torch::kCPU)
                       .pinned_memory(true));
  std::memcpy(forward_input.input_host_buffer.data_ptr(),
              payload.data(),
              payload.size());
  forward_input.input_host_buffer_has_layout = true;
  forward_input.device_tensors_ready = false;
}

size_t calculate_raw_token_size(const RawToken& token) {
  size_t size = type_size<int64_t>;  // id

  size += type_size<bool>;
  if (token.logprob.has_value()) {
    size += type_size<float>;
  }

  size += type_size<uint64_t> + token.top_tokens.size() * type_size<int64_t>;
  size += type_size<uint64_t> + token.top_logprobs.size() * type_size<float>;
  size += type_size<uint64_t> + token.embeddings.size() * type_size<float>;

  return size;
}

size_t calculate_raw_sample_output_size(const RawSampleOutput& sample) {
  size_t size = type_size<uint64_t>;
  for (const auto& token : sample.tokens) {
    size += calculate_raw_token_size(token);
  }
  return size;
}

size_t calculate_raw_forward_output_size(const RawForwardOutput& output) {
  size_t size = 0;

  size += type_size<uint64_t>;
  for (const auto& sample : output.outputs) {
    size += calculate_raw_sample_output_size(sample);
  }

  size += get_vector_size(output.expert_load_data);
  size += get_vector_size(output.src_seq_idxes);
  size += get_vector_size(output.out_tokens);
  size += get_vector_size(output.out_logprobs);
  size += type_size<int32_t>;  // prepared_layer_id
  // mm_embedding_data
  size += get_vector_tensor_size(output.mm_embeddings);
  const bool has_dit_forward_output =
      !output.dit_forward_output.tensors.empty();
  size += type_size<bool>;
  if (has_dit_forward_output) {
    size += get_dit_forward_output_size(output.dit_forward_output);
  }

  return size;
}

void write_raw_token(char*& buffer, const RawToken& token) {
  write_data(buffer, token.id);

  write_data(buffer, token.logprob.has_value());
  if (token.logprob.has_value()) {
    write_data(buffer, token.logprob.value());
  }

  write_vector(buffer, token.top_tokens);
  write_vector(buffer, token.top_logprobs);
  write_vector(buffer, token.embeddings);
}

void write_raw_sample_output(char*& buffer, const RawSampleOutput& sample) {
  write_data(buffer, static_cast<uint64_t>(sample.tokens.size()));
  for (const auto& token : sample.tokens) {
    write_raw_token(buffer, token);
  }
}

void read_raw_token(const char*& buffer, RawToken& token) {
  read_data(buffer, token.id);

  bool has_logprob;
  read_data(buffer, has_logprob);
  if (has_logprob) {
    float logprob_val;
    read_data(buffer, logprob_val);
    token.logprob = logprob_val;
  } else {
    token.logprob = std::nullopt;
  }

  read_vector(buffer, token.top_tokens);
  read_vector(buffer, token.top_logprobs);
  read_vector(buffer, token.embeddings);
}

void read_raw_sample_output(const char*& buffer, RawSampleOutput& sample) {
  uint64_t token_count;
  read_data(buffer, token_count);
  sample.tokens.resize(token_count);
  for (auto& token : sample.tokens) {
    read_raw_token(buffer, token);
  }
}

void deserialize_raw_forward_output(const char* buffer,
                                    RawForwardOutput& output) {
  uint64_t outputs_count;
  read_data(buffer, outputs_count);
  output.outputs.resize(outputs_count);
  for (auto& sample : output.outputs) {
    read_raw_sample_output(buffer, sample);
  }

  read_vector(buffer, output.expert_load_data);
  read_vector(buffer, output.src_seq_idxes);
  read_vector(buffer, output.out_tokens);
  read_vector(buffer, output.out_logprobs);

  read_data(buffer, output.prepared_layer_id);

  read_vector_tensor(buffer, output.mm_embeddings);

  bool has_dit_forward_output = false;
  read_data(buffer, has_dit_forward_output);
  if (has_dit_forward_output) {
    read_dit_forward_output(buffer, output.dit_forward_output);
  }
}

void serialize_raw_forward_output(const RawForwardOutput& output,
                                  char*& buffer) {
  write_data(buffer, static_cast<uint64_t>(output.outputs.size()));
  for (const auto& sample : output.outputs) {
    write_raw_sample_output(buffer, sample);
  }

  write_vector(buffer, output.expert_load_data);
  write_vector(buffer, output.src_seq_idxes);
  write_vector(buffer, output.out_tokens);
  write_vector(buffer, output.out_logprobs);

  write_data(buffer, output.prepared_layer_id);

  write_vector_tensor(buffer, output.mm_embeddings);
  const bool has_dit_forward_output =
      !output.dit_forward_output.tensors.empty();
  write_data(buffer, has_dit_forward_output);
  if (has_dit_forward_output) {
    write_dit_forward_output(buffer, output.dit_forward_output);
  }
}

template <typename T>
std::vector<T> tensor_to_vector(const torch::Tensor& tensor) {
  if (!tensor.defined() || tensor.numel() == 0) {
    return {};
  }
  torch::Tensor cpu_tensor = tensor.cpu().contiguous();
  if (cpu_tensor.scalar_type() != get_scalar_type<T>()) {
    cpu_tensor = cpu_tensor.to(get_scalar_type<T>());
  }
  const T* data_ptr = cpu_tensor.data_ptr<T>();
  const size_t size = static_cast<size_t>(cpu_tensor.numel());
  return std::vector<T>(data_ptr, data_ptr + size);
}

template <typename T>
std::vector<T> host_vector_or_tensor(const std::vector<T>& host_values,
                                     const torch::Tensor& tensor) {
  if (!host_values.empty()) {
    return host_values;
  }
  return tensor_to_vector<T>(tensor);
}

torch::Tensor cpu_tensor_or_self(const torch::Tensor& tensor) {
  if (!tensor.defined()) {
    return tensor;
  }
  return tensor.device().is_cpu() ? tensor : tensor.cpu();
}

torch::Tensor choose_host_or_device_tensor(const torch::Tensor& host_tensor,
                                           const torch::Tensor& device_tensor) {
  if (host_tensor.defined()) {
    return host_tensor;
  }
  return device_tensor;
}

void write_host_vector_or_tensor(RawInputSerializeContext& context,
                                 const std::vector<int32_t>& host_values,
                                 const torch::Tensor& tensor) {
  if (!host_values.empty()) {
    write_vector_to_tensor(context, host_values);
    return;
  }
  write_tensor(context, tensor);
}

inline void serialize_forward_input_sections(
    const ForwardInput& input,
    RawInputSerializeContext& context) {
  const torch::Tensor host_token_ids =
      cpu_tensor_or_self(input.host_token_ids());
  const torch::Tensor host_positions =
      cpu_tensor_or_self(input.host_positions());
  write_tensor(context, host_token_ids);
  write_tensor(context, host_positions);

  const auto& input_params = input.input_params;
  write_data(context.descriptor, input_params.meta.batch_forward_type.value());
  write_data(context.descriptor, input_params.meta.num_sequences);
  write_data(context.descriptor, input_params.meta.kv_max_seq_len);
  write_data(context.descriptor, input_params.meta.q_max_seq_len);
  write_data(context.descriptor, input_params.meta.batch_id);

  write_host_vector_or_tensor(context,
                              input_params.attention.host.q_seq_lens,
                              input_params.attention.device.q_seq_lens);
  write_host_vector_or_tensor(context,
                              input_params.attention.host.q_cu_seq_lens,
                              input_params.attention.device.q_cu_seq_lens);
  write_host_vector_or_tensor(context,
                              input_params.attention.host.kv_seq_lens,
                              input_params.attention.device.kv_seq_lens);
  write_tensor(context, input_params.attention.device.paged_kv_indptr);
  write_tensor(context, input_params.attention.device.paged_kv_indices);
  write_tensor(context, input_params.attention.device.paged_kv_last_page_len);
  write_tensor(context, input_params.attention.device.new_cache_slot_offsets);
  write_tensor(context, input_params.attention.device.kv_cache_start_offsets);
  write_tensor(context, input_params.embedding.input_embedding);
  write_vector(context.descriptor, input_params.parallel.dp_global_token_nums);
  write_vector(context.descriptor, input_params.parallel.dp_is_decode);
  write_vector(context.descriptor, input_params.embedding.embedding_ids);
  write_vector(context.descriptor, input_params.embedding.linear_state_ids);
  write_linear_state_cache_ops(context, input_params.linear_state_cache_ops);
  write_string_vector(context.descriptor, input_params.embedding.request_ids);
  write_vector(context.descriptor, input_params.embedding.extra_token_ids);
  // Mirror the read_* layout: write root + embedding mtp paths so the
  // deserializer sees both fields. Order MUST match the read_* sequence.
  write_tensor(context, input_params.mtp_shifted_token_ids);
  write_tensor(context, input_params.embedding.mtp_shifted_token_ids);
  write_vector(context.descriptor,
               input_params.embedding.mtp_bootstrap_row_idxes);
  write_tensor(context, input_params.embedding.mtp_bootstrap_embeddings);
  write_swap_blocks(context, input_params.block_copy.swap_blocks);
  write_tensor(context, input_params.block_copy.src_block_indices);
  write_tensor(context, input_params.block_copy.dst_block_indices);
  write_tensor(context, input_params.block_copy.cum_sum);
  write_mm_batch_data(context, input_params.multimodal.mm_data);
  write_host_vector_or_tensor(
      context,
      input_params.attention.host.kv_cache_tokens_nums,
      input_params.attention.device.kv_cache_tokens_nums);

  const SamplingParameters& sampling_params = input.sampling_params;
  const uint64_t selected_token_idxes_size =
      sampling_params.selected_token_idxes.defined()
          ? static_cast<uint64_t>(sampling_params.selected_token_idxes.numel())
          : 0;
  write_data(context.descriptor, selected_token_idxes_size);
  if (selected_token_idxes_size > 0) {
    write_tensor(context, sampling_params.selected_token_idxes);
    write_tensor(context, sampling_params.frequency_penalties);
    write_tensor(context, sampling_params.presence_penalties);
    write_tensor(context, sampling_params.repetition_penalties);
    write_tensor(context, sampling_params.temperatures);
    write_tensor(context, sampling_params.top_p);
    write_tensor(context, sampling_params.top_k);
    write_tensor(context, sampling_params.unique_token_ids);
    write_tensor(context, sampling_params.unique_token_counts);
    write_tensor(context, sampling_params.unique_token_ids_lens);
    write_tensor(context, sampling_params.sample_idxes);
    write_tensor(context, sampling_params.do_sample);
    write_data(context.descriptor, sampling_params.all_random_sample);
    write_data(context.descriptor, sampling_params.all_greedy_sample);
    write_data(context.descriptor, sampling_params.logprobs);
    write_data(context.descriptor, sampling_params.is_embeddings);
    write_data(context.descriptor, sampling_params.max_top_logprobs);
    write_data(context.descriptor, sampling_params.num_return_sequences);
    write_data(context.descriptor, sampling_params.use_beam_search);
  }

  write_tensor(context, sampling_params.acc_logprob);

  write_data(context.descriptor,
             static_cast<uint64_t>(input.transfer_kv_infos.size()));
  for (const auto& transfer : input.transfer_kv_infos) {
    write_transfer_kv_info(context, transfer);
  }
  write_eplb_info(context, input_params.expert.eplb_info);

  write_host_vector_or_tensor(context,
                              input_params.attention.host.new_cache_slots,
                              input_params.attention.device.new_cache_slots);
  write_tensor(
      context,
      choose_host_or_device_tensor(input_params.attention.host.block_tables,
                                   input_params.attention.device.block_tables));
  write_data(context.descriptor,
             static_cast<int32_t>(input_params.multi_block_tables.size()));
  for (const auto& manager_table : input_params.multi_block_tables) {
    write_tensor(context, manager_table);
  }

  const bool has_dit_forward_input = input_params.dit_forward_input.valid();
  write_data(context.descriptor, has_dit_forward_input);
  if (has_dit_forward_input) {
    write_dit_forward_input(context, input_params.dit_forward_input);
  }
}

inline RawInputLayoutHeader calculate_forward_input_layout(
    const ForwardInput& input) {
  RawInputSerializeContext context;
  serialize_forward_input_sections(input, context);
  const uint64_t tensor_arena_offset =
      align_up(sizeof(RawInputLayoutHeader) + context.descriptor.size,
               kRawInputTensorArenaAlignment);
  return RawInputLayoutHeader{
      context.descriptor.size, tensor_arena_offset, context.tensor_arena.size};
}

inline void serialize_forward_input(const ForwardInput& input,
                                    const RawInputLayoutHeader& layout,
                                    char*& buffer) {
  char* payload_base = buffer;
  write_data(buffer, layout.descriptor_bytes);
  write_data(buffer, layout.tensor_arena_offset);
  write_data(buffer, layout.tensor_arena_bytes);

  RawInputSerializeContext context{
      {buffer, 0}, {payload_base + layout.tensor_arena_offset, 0}};
  serialize_forward_input_sections(input, context);

  CHECK_EQ(context.descriptor.size, layout.descriptor_bytes)
      << "forward input descriptor size mismatch";
  CHECK_EQ(context.tensor_arena.size, layout.tensor_arena_bytes)
      << "forward input tensor arena size mismatch";

  const uint64_t padding_bytes =
      layout.tensor_arena_offset -
      (sizeof(RawInputLayoutHeader) + layout.descriptor_bytes);
  if (padding_bytes > 0) {
    std::memset(buffer + layout.descriptor_bytes,
                0,
                static_cast<size_t>(padding_bytes));
  }
  buffer = payload_base + get_input_layout_size(layout);
}

void convert_tensor_to_raw_output(
    const torch::Tensor& next_tokens,
    const torch::Tensor& logprobs,
    const torch::Tensor& top_tokens,
    const torch::Tensor& top_logprobs,
    const torch::Tensor& embeddings,
    const std::vector<torch::Tensor>& mm_embeddings,
    const std::vector<torch::Tensor>& dit_images,
    const std::vector<std::string>& dit_text_output,
    const torch::Tensor& expert_load_data,
    int32_t prepared_layer_id,
    const torch::Tensor& src_seq_idxes,
    const torch::Tensor& out_tokens,
    const torch::Tensor& out_logprobs,
    RawForwardOutput& raw_output) {
  raw_output.prepared_layer_id = prepared_layer_id;

  if (::xllm::EPLBConfig::get_instance().enable_eplb()) {
    torch::Tensor expert_load_data_flattened =
        expert_load_data.view({-1}).contiguous();
    if (expert_load_data_flattened.defined()) {
      const int64_t* data_ptr = expert_load_data_flattened.data_ptr<int64_t>();
      size_t size = static_cast<size_t>(expert_load_data_flattened.size(0));
      raw_output.expert_load_data.assign(data_ptr, data_ptr + size);
    }
  }

  if (src_seq_idxes.defined() && src_seq_idxes.numel() > 0) {
    const int32_t* data_ptr = src_seq_idxes.data_ptr<int32_t>();
    size_t size = static_cast<size_t>(src_seq_idxes.size(0));
    raw_output.src_seq_idxes.assign(data_ptr, data_ptr + size);
  }

  if (out_tokens.defined() && out_tokens.numel() > 0) {
    const int32_t* data_ptr = out_tokens.data_ptr<int32_t>();
    size_t size = static_cast<size_t>(out_tokens.size(0));
    raw_output.out_tokens.assign(data_ptr, data_ptr + size);
  }

  if (out_logprobs.defined() && out_logprobs.numel() > 0) {
    const float* data_ptr = out_logprobs.data_ptr<float>();
    size_t size = static_cast<size_t>(out_logprobs.size(0));
    raw_output.out_logprobs.assign(data_ptr, data_ptr + size);
  }

  int32_t num_seqs =
      next_tokens.defined() ? static_cast<int32_t>(next_tokens.size(0)) : 0;
  if (embeddings.defined() && embeddings.numel() > 0) {
    num_seqs = std::max(num_seqs, static_cast<int32_t>(embeddings.size(0)));
  }

  raw_output.outputs.reserve(num_seqs);
  raw_output.mm_embeddings = mm_embeddings;
  raw_output.dit_forward_output.tensors = dit_images;
  raw_output.dit_forward_output.text_output = dit_text_output;
  for (int32_t output_idx = 0; output_idx < num_seqs; ++output_idx) {
    RawSampleOutput raw_sample_output;

    if (next_tokens.defined() && next_tokens.dim() == 2) {
      const auto curr_idx = output_idx;
      const auto curr_next_tokens = next_tokens[curr_idx];
      const auto curr_logprobs =
          logprobs.defined() ? logprobs[curr_idx] : logprobs;
      const auto curr_top_tokens =
          top_tokens.defined() ? top_tokens[curr_idx] : top_tokens;
      const auto curr_top_logprobs =
          top_logprobs.defined() ? top_logprobs[curr_idx] : top_logprobs;
      const auto curr_embeddings =
          embeddings.defined() ? embeddings[curr_idx] : embeddings;

      int32_t num_tokens = curr_next_tokens.size(0);
      raw_sample_output.tokens.reserve(num_tokens);

      for (int32_t i = 0; i < num_tokens; ++i) {
        const Token token = build_token(i,
                                        curr_next_tokens,
                                        curr_logprobs,
                                        curr_top_tokens,
                                        curr_top_logprobs);
        if (token.id == -1) {
          break;
        }

        RawToken raw_token;
        raw_token.id = token.id;
        raw_token.logprob = token.logprob;
        raw_token.top_tokens = token.top_tokens;
        raw_token.top_logprobs = token.top_logprobs;

        if (curr_embeddings.defined()) {
          const auto token_embeddings = curr_embeddings[i];
          if (token_embeddings.defined()) {
            const float* emb_ptr = token_embeddings.data_ptr<float>();
            size_t emb_size = static_cast<size_t>(token_embeddings.size(0));
            raw_token.embeddings.assign(emb_ptr, emb_ptr + emb_size);
          }
        }

        raw_sample_output.tokens.push_back(std::move(raw_token));
      }
    } else {
      RawToken raw_token;

      if (next_tokens.defined() && next_tokens.numel() > 0) {
        const Token token = build_token(
            output_idx, next_tokens, logprobs, top_tokens, top_logprobs);
        raw_token.id = token.id;
        raw_token.logprob = token.logprob;
        raw_token.top_tokens = std::move(token.top_tokens);
        raw_token.top_logprobs = std::move(token.top_logprobs);
      } else {
        raw_token.id = -1;
        raw_token.logprob = std::nullopt;
      }

      if (embeddings.defined()) {
        const auto token_embeddings = embeddings[output_idx];
        if (token_embeddings.defined()) {
          const float* emb_ptr = token_embeddings.data_ptr<float>();
          size_t emb_size = static_cast<size_t>(token_embeddings.size(0));
          raw_token.embeddings.assign(emb_ptr, emb_ptr + emb_size);
        }
      }

      raw_sample_output.tokens.push_back(std::move(raw_token));
    }
    raw_output.outputs.push_back(std::move(raw_sample_output));
  }
}

}  // namespace

namespace detail {

bool unpack_from_input_host_buffer(const ForwardInput& input,
                                   const torch::Device& device,
                                   torch::ScalarType dtype,
                                   ForwardInput& output,
                                   bool materialize_device_buffer) {
  if (!input.input_host_buffer.defined() ||
      !input.input_host_buffer.device().is_cpu() ||
      input.input_host_buffer.numel() == 0) {
    return false;
  }

  output = input;
  output.device_tensors_ready = false;
  const char* payload_ptr =
      static_cast<const char*>(output.input_host_buffer.data_ptr());
  deserialize_forward_input_payload(
      payload_ptr,
      static_cast<uint64_t>(output.input_host_buffer.numel()),
      output,
      device,
      /*stream=*/nullptr,
      materialize_device_buffer);
  auto normalize_float_param = [&](torch::Tensor& tensor) {
    if (tensor.defined() && tensor.scalar_type() != dtype) {
      tensor = tensor.to(torch::TensorOptions().device(device).dtype(dtype));
    }
  };
  if (materialize_device_buffer) {
    normalize_float_param(output.sampling_params.frequency_penalties);
    normalize_float_param(output.sampling_params.presence_penalties);
    normalize_float_param(output.sampling_params.repetition_penalties);
    normalize_float_param(output.sampling_params.temperatures);
    normalize_float_param(output.sampling_params.top_p);
    normalize_float_param(output.decoder_sampling_params.frequency_penalties);
    normalize_float_param(output.decoder_sampling_params.presence_penalties);
    normalize_float_param(output.decoder_sampling_params.repetition_penalties);
    normalize_float_param(output.decoder_sampling_params.temperatures);
    normalize_float_param(output.decoder_sampling_params.top_p);
    output.positions = detail::normalize_positions_for_device(output.positions);
    return output.device_tensors_ready;
  }

  // For devices without contiguous-input-buffer support, unpack to CPU tensors
  // first and fall back to the regular H2D path in ForwardInput::to().
  output.input_host_buffer_has_layout = false;
  output.device_tensors_ready = false;
  return true;
}

bool unpack_from_input_host_buffer(const ForwardInput& input,
                                   const torch::Device& device,
                                   ForwardInput& output) {
  return unpack_from_input_host_buffer(input,
                                       device,
                                       torch::kFloat32,
                                       output,
                                       /*materialize_device_buffer=*/false);
}

bool try_to_device_from_input_host_buffer(const ForwardInput& input,
                                          const torch::Device& device,
                                          torch::ScalarType dtype,
                                          ForwardInput& output) {
  return unpack_from_input_host_buffer(
      input, device, dtype, output, /*materialize_device_buffer=*/true);
}

}  // namespace detail

bool forward_input_to_packed_proto(
    const ForwardInput& input,
    proto::PackedForwardInput* packed_forward_input) {
  CHECK(packed_forward_input != nullptr);
  const RawInputLayoutHeader layout = calculate_forward_input_layout(input);
  const uint64_t payload_size = get_input_layout_size(layout);
  std::string payload;
  payload.resize(static_cast<size_t>(payload_size));

  char* payload_ptr = &payload[0];
  serialize_forward_input(input, layout, payload_ptr);
  CHECK_EQ(payload_ptr, &payload[0] + payload.size())
      << "packed forward input payload size mismatch";

  packed_forward_input->mutable_payload()->swap(payload);
  return true;
}

void packed_proto_to_forward_input(
    const proto::PackedForwardInput& packed_forward_input,
    ForwardInput& forward_input,
    const torch::Device& device,
    Stream* stream) {
  packed_proto_to_forward_input_impl(
      packed_forward_input, forward_input, device, stream);
}

ForwardSharedMemoryManager::ForwardSharedMemoryManager(const std::string& name,
                                                       size_t size,
                                                       bool& is_creator,
                                                       ForwardType type)
    : SharedMemoryManager(name, size, is_creator), forward_type_(type) {
  control_ptr_ = static_cast<ControlMetadata*>(base_address());
  metadata_addr_ = static_cast<char*>(base_address()) + sizeof(ControlMetadata);
  if (::xllm::ExecutionConfig::get_instance().use_contiguous_input_buffer()) {
    stream_ = std::make_unique<Stream>();
  }
}

ForwardSharedMemoryManager::~ForwardSharedMemoryManager() = default;

/* The shared memory filename may have duplicates when using kill -9 xllm, but
  this doesn't affect usage.*/
std::string ForwardSharedMemoryManager::create_unique_name(
    const std::string& prefix,
    int dp_group,
    ForwardType forward_type,
    int rank) {
  std::string filename = prefix;
  if (forward_type == ForwardType::PB_INPUT ||
      forward_type == ForwardType::RAW_INPUT) {
    filename += "_dpg_" + std::to_string(dp_group) + "_input";
  } else if (forward_type == ForwardType::PB_OUTPUT ||
             forward_type == ForwardType::RAW_OUTPUT) {
    filename += "_rank_" + std::to_string(rank) + "_output";
  } else {
    // TODO: support more type later
  }

  return filename;
}

bool ForwardSharedMemoryManager::input_write(const ForwardInput& input) {
  proto::PackedForwardInput packed_forward_input;
  if (!forward_input_to_packed_proto(input, &packed_forward_input)) {
    LOG(ERROR) << "failed to convert ForwardInput to packed transport payload";
    return false;
  }

  const std::string& payload = packed_forward_input.payload();
  uint64_t total_size = sizeof(ControlMetadata);
  total_size += type_size<uint64_t> + payload.size();
  if (unlikely(total_size > size())) {
    LOG(ERROR) << "forward input size overflow, total_size: " << total_size
               << ", shm size: " << size();
    return false;
  }

  char* data_ptr = static_cast<char*>(base_address()) + sizeof(ControlMetadata);
  write_data(data_ptr, static_cast<uint64_t>(payload.size()));
  if (!payload.empty()) {
    std::memcpy(data_ptr, payload.data(), payload.size());
    data_ptr += payload.size();
  }

  uint64_t real_size =
      static_cast<uint64_t>(data_ptr - static_cast<char*>(base_address()));
  CHECK_EQ(total_size, real_size) << "total_size != real_size.";
  std::atomic_thread_fence(std::memory_order_release);
  control_ptr_->version = ++last_version_;
  return true;
}

void ForwardSharedMemoryManager::input_read(ForwardInput& input,
                                            const torch::Device& device) {
  while (true) {
    if (control_ptr_->version != last_version_) {
      last_version_ = control_ptr_->version;
      std::atomic_thread_fence(std::memory_order_acquire);
      break;
    }
    std::this_thread::sleep_for(std::chrono::nanoseconds(kNumWaitNanoseconds));
  }

  const char* data_ptr =
      static_cast<char*>(base_address()) + sizeof(ControlMetadata);
  uint64_t total_size;
  read_data(data_ptr, total_size);
  bool materialize_device_buffer = false;
#if defined(USE_NPU)
  materialize_device_buffer = true;
#elif defined(USE_CUDA)
  materialize_device_buffer = device.type() == torch::kCUDA;
#elif defined(USE_MLU)
  materialize_device_buffer = device.type() == torch::kPrivateUse1;
#endif
  deserialize_forward_input_payload(data_ptr,
                                    total_size,
                                    input,
                                    device,
                                    stream_.get(),
                                    materialize_device_buffer,
                                    /*stabilize_dit_host_tensors=*/true);

  return;
}

bool ForwardSharedMemoryManager::raw_output_write(
    const torch::Tensor& next_tokens,
    const torch::Tensor& logprobs,
    const torch::Tensor& top_tokens,
    const torch::Tensor& top_logprobs,
    const torch::Tensor& embeddings,
    const std::vector<torch::Tensor>& mm_embeddings,
    const std::vector<torch::Tensor>& dit_images,
    const std::vector<std::string>& dit_text_output,
    const torch::Tensor& expert_load_data,
    int32_t prepared_layer_id,
    const torch::Tensor& src_seq_idxes,
    const torch::Tensor& out_tokens,
    const torch::Tensor& out_logprobs) {
  RawForwardOutput output;
  convert_tensor_to_raw_output(next_tokens,
                               logprobs,
                               top_tokens,
                               top_logprobs,
                               embeddings,
                               mm_embeddings,
                               dit_images,
                               dit_text_output,
                               expert_load_data,
                               prepared_layer_id,
                               src_seq_idxes,
                               out_tokens,
                               out_logprobs,
                               output);
  uint64_t total_size = sizeof(ControlMetadata);
  total_size += calculate_raw_forward_output_size(output);
  if (unlikely(total_size > size())) {
    LOG(ERROR) << "raw output size overflow, total_size: " << total_size
               << ", shm size: " << size();
    return false;
  }

  char* data_ptr = static_cast<char*>(base_address()) + sizeof(ControlMetadata);
  serialize_raw_forward_output(output, data_ptr);
  std::atomic_thread_fence(std::memory_order_release);
  control_ptr_->version = ++last_version_;
  return true;
}

void ForwardSharedMemoryManager::raw_output_read(RawForwardOutput& output) {
  while (true) {
    if (control_ptr_->version != last_version_) {
      last_version_ = control_ptr_->version;
      std::atomic_thread_fence(std::memory_order_acquire);
      break;
    }
    std::this_thread::sleep_for(std::chrono::nanoseconds(kNumWaitNanoseconds));
  }

  const char* data_ptr =
      static_cast<char*>(base_address()) + sizeof(ControlMetadata);
  deserialize_raw_forward_output(data_ptr, output);

  return;
}

void ForwardSharedMemoryManager::clear() {
  std::memset(base_address(), 0, size());
}
}  // namespace xllm

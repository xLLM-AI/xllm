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

#include <unistd.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <memory>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

#include "rec.h"

std::string model_name = "homepage_qwen_06b_6_raw";
std::string model_path = "/export/home/models/homepage_qwen_06b_6_raw";

#define MODEL_WORD_EMBEDDING_SIZE 1024

class XLLM_MM_Data_Wrapper {
 public:
  XLLM_MM_Data_Wrapper() = default;

  ~XLLM_MM_Data_Wrapper() { reset(); }

  XLLM_MM_Data_Wrapper(const XLLM_MM_Data_Wrapper&) = delete;
  XLLM_MM_Data_Wrapper& operator=(const XLLM_MM_Data_Wrapper&) = delete;

  bool build(
      const std::vector<std::pair<uint32_t, uint32_t>>& token_positions) {
    if (is_built_ || token_positions.empty()) {
      fprintf(stderr,
              "build() failed: already built or empty token positions\n");
      return false;
    }

    mm_data_.type_mask = static_cast<uint32_t>(XLLM_MM_TYPE_EMBEDDING);
    mm_data_.is_dict = false;

    for (size_t i = 0; i < token_positions.size(); ++i) {
      const auto& [offset, length] = token_positions[i];

      if (length == 0) {
        fprintf(stderr, "build() skipped item %zu: length cannot be 0\n", i);
        continue;
      }

      items_.emplace_back(create_embedding_item(offset, length));
    }

    if (items_.empty()) {
      fprintf(stderr, "build() failed: no valid embedding items created\n");
      return false;
    }

    mm_data_.data.items.entries_size = items_.size();
    mm_data_.data.items.entries = items_.data();

    is_built_ = true;
    return true;
  }

  void reset() {
    memset(&mm_data_, 0, sizeof(mm_data_));

    items_.clear();
    tensor_buffers_.clear();
    is_built_ = false;
  }

  const XLLM_MM_Data* get_data() const {
    return is_built_ ? &mm_data_ : nullptr;
  }

  void validate() const {
    if (!is_built_) {
      fprintf(stderr,
              "validate() failed: no data available (call build() first)\n");
      return;
    }

    const size_t item_count = mm_data_.data.items.entries_size;
    printf("=== Validating %zu Embedding Items ===\n\n", item_count);

    for (size_t i = 0; i < item_count; ++i) {
      const auto& item = mm_data_.data.items.entries[i];
      printf("=== Embedding Item %zu ===\n", i + 1);
      printf("Token Position: offset=%u, length=%u\n",
             item.state.token_pos.offset,
             item.state.token_pos.length);
      printf("Data Type:  (%d)\n", item.data.data.tensor.dtype);
      printf("Tensor Shape: rank=%d, dim=[%d, %d]\n\n",
             item.data.data.tensor.dims.rank,
             item.data.data.tensor.dims.dim[0],
             item.data.data.tensor.dims.dim[1]);
    }
  }

  bool is_built() const { return is_built_; }

  size_t get_item_count() const {
    return is_built_ ? mm_data_.data.items.entries_size : 0;
  }

 private:
  XLLM_MM_Data mm_data_{};
  std::vector<XLLM_MM_Item> items_;
  std::vector<std::unique_ptr<uint16_t[]>> tensor_buffers_;
  bool is_built_ = false;

  inline uint16_t float_to_bfloat16(float f) {
    union {
      float f32;
      uint32_t u32;
    } u;
    u.f32 = f;
    return static_cast<uint16_t>(u.u32 >> 16);
  }

  XLLM_MM_Item create_embedding_item(uint32_t offset, uint32_t length) {
    XLLM_MM_Item item{};

    item.type = XLLM_MM_TYPE_EMBEDDING;
    item.state.token_pos.offset = offset;
    item.state.token_pos.length = length;

    item.data.is_single_tensor = true;
    item.data.data.tensor.dtype = XLLM_DTYPE_BFLOAT16;
    item.data.data.tensor.dims.rank = 2;
    memset(item.data.data.tensor.dims.dim,
           0,
           sizeof(item.data.data.tensor.dims.dim));
    item.data.data.tensor.dims.dim[0] = static_cast<int>(length);
    item.data.data.tensor.dims.dim[1] = MODEL_WORD_EMBEDDING_SIZE;

    const size_t element_count = length * MODEL_WORD_EMBEDDING_SIZE;
    const size_t buffer_size_bytes = element_count * sizeof(uint16_t);

    auto buffer = std::make_unique<uint16_t[]>(element_count);

    for (size_t i = 0; i < length; ++i) {
      for (size_t j = 0; j < MODEL_WORD_EMBEDDING_SIZE; ++j) {
        float float_val =
            static_cast<float>(i * MODEL_WORD_EMBEDDING_SIZE + j) /
            static_cast<float>(element_count);

        uint16_t bf16_val = float_to_bfloat16(float_val);
        buffer[i * MODEL_WORD_EMBEDDING_SIZE + j] = bf16_val;
      }
    }

    item.data.data.tensor.data = buffer.get();
    tensor_buffers_.push_back(std::move(buffer));

    return item;
  }
};

XLLM_REC_Handler* service_startup_hook() {
  XLLM_REC_Handler* rec_handler = xllm_rec_create();

  // If there is no separate setting, init_options can be passed as nullptr, and
  // the default value(XLLM_INIT_REC_OPTIONS_DEFAULT) will be used
  XLLM_InitOptions init_options;
  xllm_rec_init_options_default(&init_options);
  // init_options.beam_width = 1;
  // init_options.max_decode_rounds = 0;
  snprintf(init_options.log_dir,
           sizeof(init_options.log_dir),
           "/export/home/huheng7/log");

  bool ret =
      xllm_rec_initialize(rec_handler, model_path.c_str(), &init_options);
  if (!ret) {
    std::cout << "REC init failed" << std::endl;
    xllm_rec_destroy(rec_handler);
    return nullptr;
  }

  std::cout << "REC init successfully" << std::endl;

  return rec_handler;
}

void service_stop_hook(XLLM_REC_Handler* rec_handler) {
  xllm_rec_destroy(rec_handler);
  std::cout << "REC stop" << std::endl;
}

int generate_random_int(int min, int max) {
  if (min > max) {
    throw std::invalid_argument("min cannot be greater than max");
  }

  static std::random_device rd;
  static std::mt19937 gen(rd());

  std::uniform_int_distribution<int> dist(min, max);

  return dist(gen);
}

int main(int argc, char** argv) {
  XLLM_REC_Handler* rec_handler = service_startup_hook();
  if (nullptr == rec_handler) {
    return -1;
  }

  // If there is no separate setting, request_params can be passed as nullptr,
  // and the default value(XLLM_REQUEST_PARAMS_DEFAULT) will be used
  XLLM_RequestParams request_params;
  xllm_rec_request_params_default(&request_params);
  // request_params.beam_width = 128;
  request_params.max_tokens = 3;
  request_params.beam_width = 128;
  request_params.logprobs = true;
  // request_params.temperature = 1.0;
  request_params.top_k = 128;
  request_params.top_logprobs = 128;
  // request_params.top_p = 1.0;
  // request_params.repetition_penalty = 1.0;

  std::vector<int32_t> token_ids = {
      151644, 8948,   198,    56568,  101909, 101215, 104799, 101914, 101057,
      3837,   103929, 100032, 44956,  15946,  55338,  45943,  104570, 11622,
      105801, 72881,  64559,  307,    71817,  51463,  3837,   56568,  107618,
      100345, 20002,  104754, 72651,  105565, 45943,  116951, 101034, 67949,
      72651,  109348, 36407,  104538, 20002,  104326, 87267,  72651,  109348,
      1773,   151645, 198,    151644, 872,    198,    20002,  21,     15,
      35727,  31843,  36667,  59879,  20450,  99805,  32044,  72651,  105565,
      45943,  32044,  113507, 153479, 155828, 160439, 11,     153479, 157177,
      160439, 11,     153479, 155828, 160439, 11,     153479, 155828, 160439,
      11,     153479, 155828, 160439, 11,     153479, 155828, 160439, 11,
      155622, 158228, 160337, 11,     152907, 158228, 159858, 11,     153036,
      158228, 160333, 11,     153258, 159797, 160105, 11,     153186, 157627,
      160740, 11,     152907, 158228, 160680, 11,     154562, 157329, 160321,
      11,     153326, 157680, 163928, 11,     153258, 159634, 160105, 11,
      152847, 157129, 162841, 11,     152847, 157399, 162841, 11,     152847,
      158228, 163388, 11,     153036, 159807, 162840, 11,     154562, 157329,
      160321, 11,     154562, 156839, 160321, 11,     154562, 158181, 160321,
      11,     153326, 158534, 163886, 11,     153326, 157177, 163041, 11,
      155622, 158228, 163359, 11,     152569, 155800, 162738, 11,     153390,
      158228, 160357, 11,     152663, 157649, 162738, 11,     155193, 158667,
      162738, 11,     155622, 158228, 160706, 11,     151685, 158473, 162738,
      11,     152907, 158228, 162653, 11,     151876, 158228, 159909, 11,
      152907, 158228, 162407, 11,     152907, 158228, 163551, 11,     151685,
      158473, 162738, 11,     152686, 155927, 162029, 11,     152663, 158228,
      161841, 11,     152686, 155927, 162603, 11,     153516, 157280, 161980,
      11,     153516, 159807, 160708, 11,     153516, 157900, 163856, 11,
      153516, 155967, 161020, 11,     153516, 157280, 160838, 11,     153200,
      157591, 162582, 11,     151924, 158696, 160358, 11,     154562, 159113,
      160860, 11,     153386, 159086, 161519, 11,     154625, 159807, 160781,
      11,     153479, 155828, 160439, 11,     153479, 155828, 160439, 11,
      153479, 157177, 160439, 11,     153479, 155828, 160439, 11,     154213,
      157866, 160523, 11,     153036, 156918, 163610, 11,     153036, 157351,
      160974, 11,     153688, 158228, 160337, 11,     155507, 159807, 162736,
      11,     155370, 159219, 161059, 11,     155002, 158118, 160019, 11,
      155370, 159219, 161059, 11,     153792, 159022, 161003, 11,     155576,
      155927, 161581, 11,     155576, 155927, 163189, 11,     155576, 159630,
      162853, 11,     155576, 159630, 163527, 11,     155576, 159630, 162164,
      11,     155576, 158048, 163339, 11,     155576, 157177, 163339, 11,
      155576, 159630, 163527, 11,     155576, 157177, 163339, 11,     155576,
      157680, 163339, 11,     155576, 159630, 160653, 11,     155576, 159630,
      162153, 11,     155576, 159630, 161747, 11,     155576, 157505, 163339,
      11,     153831, 158228, 160026, 11,     153390, 158228, 161841, 11,
      153831, 156324, 162738, 11,     153390, 158228, 161491, 11,     153390,
      159145, 162738, 11,     155507, 158473, 162738, 11,     153831, 157649,
      162738, 11,     155507, 157770, 162738, 11,     153390, 158228, 161033,
      11,     155507, 158473, 162738, 11,     153390, 158228, 160824, 11,
      153479, 157649, 160439, 11,     153479, 157649, 160439, 11,     153479,
      155828, 160439, 11,     153479, 157649, 160439, 11,     153479, 157649,
      160439, 11,     153479, 157649, 160439, 11,     153849, 159380, 162841,
      11,     152663, 158107, 162738, 11,     152271, 157371, 161110, 11,
      152663, 157176, 160199, 11,     154936, 158966, 162841, 11,     153390,
      158228, 161491, 11,     153036, 158228, 162840, 11,     155646, 158228,
      162408, 11,     152663, 156814, 162738, 11,     152569, 158473, 162738,
      11,     155646, 158228, 161308, 11,     152663, 158228, 163631, 11,
      155370, 159786, 163029, 11,     153534, 159283, 161094, 11,     153534,
      157756, 163778, 11,     151905, 156698, 163573, 11,     151905, 156698,
      161534, 11,     151905, 156698, 162140, 11,     153534, 157931, 161817,
      11,     153534, 157121, 161059, 11,     154826, 158585, 163433, 11,
      154826, 158585, 160756, 11,     154826, 157666, 161504, 11,     154826,
      157351, 161808, 11,     154826, 158585, 161062, 11,     154826, 157666,
      161504, 11,     154826, 156537, 163635, 11,     155370, 159219, 161059,
      11,     155370, 156903, 160381, 11,     155370, 156903, 160381, 11,
      155370, 159219, 162223, 11,     155370, 159330, 162223, 11,     153464,
      159219, 161059, 11,     154809, 156903, 160381, 11,     153464, 156878,
      162223, 11,     154809, 157794, 162010, 11,     154809, 159219, 161059,
      11,     151893, 159807, 162666, 11,     151893, 158534, 160890, 11,
      153326, 157177, 163620, 11,     153326, 159462, 163041, 11,     152663,
      156348, 162738, 11,     152663, 158473, 162736, 11,     152463, 156537,
      160873, 11,     155507, 157176, 162738, 11,     155193, 158473, 162738,
      11,     152663, 157649, 162738, 11,     152663, 158107, 162738, 11,
      152663, 155780, 162738, 11,     152663, 158473, 162738, 11,     152663,
      157649, 162738, 11,     152663, 157649, 162738, 11,     152663, 155828,
      162738, 11,     152663, 158621, 162738, 11,     152663, 157176, 162738,
      11,     155646, 158228, 160017, 11,     155682, 158228, 162859, 67949,
      103969, 72651,  109348, 17714,  155646, 158228, 162234, 1773,   104210,
      67949,  9370,   72651,  45943,  9370,   111450, 37945,  104538, 20002,
      104326, 104309, 72651,  9370,   16,     15,     18947,  45943,  3837,
      11622,  107463, 17992,  71817,  17177,  99859,  1773,   151645, 198,
      151644, 77091,  198};

  size_t token_size = token_ids.size();
  const int32_t* token_ids_ptr = token_ids.data();

  XLLM_MM_Data_Wrapper multimodal_data_wrapper;
  std::vector<std::pair<uint32_t, uint32_t>> positions = {{100, 32}, {300, 64}};
  multimodal_data_wrapper.build(positions);
  multimodal_data_wrapper.validate();
  // multimodal_data_wrapper.get_data(),
  XLLM_Response* resp =
      xllm_rec_multimodal_completions(rec_handler,
                                      model_name.c_str(),
                                      token_ids_ptr,
                                      token_size,
                                      multimodal_data_wrapper.get_data(),
                                      10000,
                                      &request_params);
  if (nullptr == resp) {
    std::cout << "REC completions failed, response is nullptr" << std::endl;
    service_stop_hook(rec_handler);
    return -1;
  }

  if (resp->status_code != XLLM_StatusCode::kSuccess) {
    std::cout << "REC completions failed, status code:" << resp->status_code
              << ", error info:" << resp->error_info << std::endl;
  } else {
    std::cout << "REC completions successfully, size:"
              << resp->choices.entries_size << std::endl;

    if (nullptr != resp->choices.entries) {
      for (int i = 0; i < resp->choices.entries_size; ++i) {
        XLLM_Choice& choice = resp->choices.entries[i];
        std::cout << "token size: " << choice.token_size
                  << ",logprobs size:" << choice.logprobs.entries_size
                  << std::endl;

        for (int j = 0; j < choice.token_size; j++) {
          std::cout << "xllm answer[" << choice.index
                    << "]: token id=" << choice.token_ids[j] << std::endl;
        }

        for (int j = 0; j < choice.logprobs.entries_size; j++) {
          XLLM_LogProb& logprob = choice.logprobs.entries[j];
          std::cout << "xllm answer[" << choice.index
                    << "]: token id=" << logprob.token_id
                    << ", token logprob=" << logprob.logprob << std::endl;
        }
      }
    }
  }

  xllm_rec_free_response(resp);

  service_stop_hook(rec_handler);

  return 0;
}
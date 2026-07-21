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

#include <glog/logging.h>
#include <torch/torch.h>

#include <algorithm>
#include <filesystem>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_set>
#include <vector>

#include "core/common/types.h"
#include "core/framework/config/disagg_pd_config.h"
#include "core/framework/config/parallel_config.h"
#include "core/util/dit_model_discovery.h"
#include "core/util/json_reader.h"
#include "core/util/model_config_utils.h"
#include "models/model_registry.h"
#include "rec.pb.h"
#include "slice.h"
#include "tensor.pb.h"
#include "worker.pb.h"

namespace xllm {
namespace util {

std::pair<int, int> find_ones_indices(std::vector<int>& q_seq_lens);

template <typename T>
void pad_2d_vector(std::vector<std::vector<T>>& vec, T pad_value) {
  size_t max_col_size = 0;
  for (const auto& row : vec) {
    max_col_size = std::max(max_col_size, row.size());
  }

  for (auto& row : vec) {
    row.resize(max_col_size, pad_value);
  }
}

torch::ScalarType parse_dtype(const std::string& dtype_str,
                              const std::optional<torch::Device>& device);

std::optional<std::vector<uint32_t>> parse_batch_sizes(
    const std::string& batch_sizes_str);

template <typename T>
T sum(const std::vector<T>& vec) {
  if (vec.empty()) LOG(FATAL) << "vector is empty.";
  return std::accumulate(vec.begin(), vec.end(), T{});
}

template <typename T>
const T& min(const std::vector<T>& vec) {
  if (vec.empty()) LOG(FATAL) << "vector is empty.";
  return *std::min_element(vec.begin(), vec.end());
}

template <typename T>
const T& max(const std::vector<T>& vec) {
  if (vec.empty()) LOG(FATAL) << "vector is empty.";
  return *std::max_element(vec.begin(), vec.end());
}

template <typename T>
inline std::enable_if_t<std::is_integral_v<T>, T> ceil_div(T value, T divisor) {
  CHECK_GT(divisor, 0) << "divisor must be positive.";
  return value / divisor + static_cast<T>(value % divisor != 0);
}

static inline int64_t align_up(int64_t value, int64_t alignment) {
  if (alignment == 0) {
    return value;
  }
  return ((value + alignment - 1) / alignment) * alignment;
}

// Returns the first token id in `token_ids` that falls outside the valid
// vocabulary range [0, vocab_size); std::nullopt if all ids are in range.
// Callers must skip the scan when vocab_size <= 0 (range unknown).
inline std::optional<int32_t> find_out_of_vocab_token(
    const std::vector<int32_t>& token_ids,
    int64_t vocab_size) {
  for (int32_t token_id : token_ids) {
    if (token_id < 0 || token_id >= vocab_size) {
      return token_id;
    }
  }
  return std::nullopt;
}

bool match_suffix(const Slice<int32_t>& data, const Slice<int32_t>& suffix);

std::vector<uint32_t> cal_vec_split_index(uint32_t vec_size, uint32_t part_num);

torch::Tensor convert_rec_tensor_to_torch(
    const proto::InferInputTensor& input_tensor);

torch::Tensor proto_to_torch(const proto::Tensor& proto_tensor);

bool torch_to_proto(const torch::Tensor& torch_tensor,
                    proto::Tensor* proto_tensor);

int32_t ceil_pow2(int32_t n);

torch::ScalarType datatype_proto_to_torch(const std::string& proto_datatype);

std::string torch_datatype_to_proto(torch::ScalarType torch_dtype);

inline const std::unordered_set<std::string>& mla_model_type_set() {
  static const std::unordered_set<std::string> kMlaModelTypeSet = {
      "deepseek_v2",
      "deepseek_v3",
      "deepseek_v32",
      "deepseek_v3_mtp",
      "deepseek_v32_mtp",
      "kimi_k2",
      "kimi_k25",
      "glm4_moe_lite",
      "glm_moe_dsa",  // glm5 model type
      "glm_moe_dsa_mtp",
      "joyai_llm_flash"};
  return kMlaModelTypeSet;
}

inline bool is_mla_model_type(std::string_view model_type) {
  return mla_model_type_set().contains(std::string(model_type));
}

inline bool has_mtp_model_type_marker(std::string_view model_type) {
  return model_type.find("mtp") != std::string_view::npos;
}

inline bool starts_with_model_type(std::string_view model_type,
                                   std::string_view target_model_type) {
  return model_type.size() >= target_model_type.size() &&
         model_type.compare(0, target_model_type.size(), target_model_type) ==
             0;
}

inline bool is_target_mtp_model_type(std::string_view model_type,
                                     std::string_view target_model_type) {
  return starts_with_model_type(model_type, target_model_type) &&
         has_mtp_model_type_marker(model_type);
}

inline bool is_deepseek_v4_model_type(std::string_view model_type) {
  constexpr std::string_view kTargetModelType = "deepseek_v4";
  return model_type == kTargetModelType ||
         is_target_mtp_model_type(model_type, kTargetModelType);
}

inline bool is_target_model_type(std::string_view model_type,
                                 std::string_view target_model_type,
                                 bool match_mtp) {
  if (model_type == target_model_type) {
    return true;
  }
  return match_mtp && is_target_mtp_model_type(model_type, target_model_type);
}

inline std::string get_model_name(
    const std::filesystem::path& normalized_model_path) {
  std::string model_name;

  if (normalized_model_path.has_filename()) {
    model_name = normalized_model_path.filename().string();
  } else {
    model_name = normalized_model_path.parent_path().filename().string();
  }

  if (model_name.empty()) {
    LOG(FATAL) << "Cannot extract model name from path, as it appears to be a "
                  "root directory: "
               << normalized_model_path.string();
    return "";
  }

  return model_name;
}

inline std::string get_model_backend(const std::filesystem::path& model_path) {
  JsonReader reader;
  std::filesystem::path model_index_json_path = model_path / "model_index.json";

  if (std::filesystem::exists(model_index_json_path)) {
    reader.parse(model_index_json_path);
    if (reader.value<std::string>("_diffusers_version").has_value()) {
      return "dit";
    }
    // DiT models that are not diffusers-based (e.g. Cola-DLM) may have
    // _class_name but no _diffusers_version. Treat them as dit backend.
    if (reader.value<std::string>("_class_name").has_value()) {
      return "dit";
    }
    LOG(FATAL) << "Please check model_index.json file in model path: "
               << model_path << ", it should contain _diffusers_version key.";
  }

  // Component-subdirectory layout (e.g. Cola-DLM with cola_dit/cola_vae).
  if (auto components = discover_dit_components(model_path)) {
    if (!components->empty()) {
      return "dit";
    }
  }

  return ModelRegistry::get_model_backend(get_model_type(model_path));
}

inline bool should_enable_mla(
    const std::filesystem::path& model_path,
    const std::optional<std::string>& backend = std::nullopt) {
  const std::string resolved_backend =
      backend.has_value() ? backend.value() : get_model_backend(model_path);
  if (resolved_backend == "dit") {
    return false;
  }
  return is_mla_model_type(get_model_type(model_path, backend));
}

inline int32_t kv_split_size_effective(void) {
  return ParallelConfig::get_instance().kv_split_size_effective();
}

inline bool enable_kvcache_split(void) { return kv_split_size_effective() > 1; }

}  // namespace util
}  // namespace xllm

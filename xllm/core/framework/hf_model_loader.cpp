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

#include "hf_model_loader.h"

#include <absl/strings/match.h>
#include <absl/strings/str_replace.h>
#include <fcntl.h>
#include <glog/logging.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <torch/torch.h>
#include <unistd.h>

#include <algorithm>
#include <boost/algorithm/string.hpp>
#include <cctype>
#include <filesystem>
#include <limits>
#include <optional>
#include <unordered_map>
#include <vector>

#include "core/common/version_singleton.h"
#include "core/framework/config/model_config.h"
#include "core/framework/config/rec_config.h"
#include "core/framework/state_dict/rec_vocab_dict.h"
#include "core/framework/state_dict/safetensors/safetensors.h"
#include "core/framework/tokenizer/fast_tokenizer.h"
#include "core/framework/tokenizer/rec_tokenizer.h"
#include "core/framework/tokenizer/sentencepiece_tokenizer.h"
#include "core/framework/tokenizer/tiktoken_tokenizer.h"
#include "core/framework/tokenizer/tokenizer_factory.h"
#include "core/platform/device.h"
#include "core/platform/platform.h"
#include "core/util/blocking_counter.h"
#include "core/util/json_reader.h"
#include "core/util/model_config_utils.h"
#include "core/util/rec_model_utils.h"
#include "core/util/scope_guard.h"
#include "core/util/tensor_helper.h"
#include "core/util/utils.h"
#include "models/model_registry.h"

namespace xllm {

namespace {

JsonReader normalize_config_torch_dtype(const JsonReader& reader) {
  auto config = reader.data();
  if (!config.contains("torch_dtype") && config.contains("dtype")) {
    config["torch_dtype"] = config["dtype"];
  }

  JsonReader normalized_reader;
  normalized_reader.parse_text(config.dump());
  return normalized_reader;
}

bool is_compressed_tensors_fp8_scheme(const nlohmann::json& config) {
  auto type_it = config.find("type");
  auto num_bits_it = config.find("num_bits");
  return type_it != config.end() && !type_it->is_null() &&
         num_bits_it != config.end() && !num_bits_it->is_null() &&
         boost::iequals(type_it->get<std::string>(), "float") &&
         num_bits_it->get<int64_t>() == 8;
}

bool is_compressed_tensors_int8_scheme(const nlohmann::json& config,
                                       bool expected_dynamic) {
  auto type_it = config.find("type");
  auto num_bits_it = config.find("num_bits");
  auto dynamic_it = config.find("dynamic");
  const bool dynamic = dynamic_it != config.end() && !dynamic_it->is_null()
                           ? dynamic_it->get<bool>()
                           : false;
  return type_it != config.end() && !type_it->is_null() &&
         num_bits_it != config.end() && !num_bits_it->is_null() &&
         boost::iequals(type_it->get<std::string>(), "int") &&
         num_bits_it->get<int64_t>() == 8 && dynamic == expected_dynamic;
}

bool try_load_compressed_tensors_quant_cfg(const JsonReader& reader,
                                           QuantArgs& quant_args) {
  const auto quant_method =
      reader.value<std::string>("quantization_config.quant_method");
  if (!quant_method.has_value() ||
      !boost::iequals(*quant_method, "compressed-tensors")) {
    return false;
  }

  const auto& data = reader.data();
  auto quant_config_it = data.find("quantization_config");
  if (quant_config_it == data.end() || !quant_config_it->is_object()) {
    LOG(ERROR) << "quantization_config must be an object for "
                  "compressed-tensors quantization.";
    return false;
  }

  auto config_groups_it = quant_config_it->find("config_groups");
  if (config_groups_it == quant_config_it->end() ||
      !config_groups_it->is_object()) {
    LOG(ERROR) << "quantization_config.config_groups must be an object for "
                  "compressed-tensors quantization.";
    return false;
  }

  for (const auto& [group_name, group] : config_groups_it->items()) {
    if (!group.is_object()) {
      continue;
    }

    auto weights_it = group.find("weights");
    auto input_activations_it = group.find("input_activations");
    if (weights_it == group.end() || input_activations_it == group.end() ||
        !weights_it->is_object() || !input_activations_it->is_object()) {
      continue;
    }

    if (!is_compressed_tensors_fp8_scheme(*weights_it) ||
        !is_compressed_tensors_fp8_scheme(*input_activations_it)) {
      // Check for INT8 W8A8 (compressed-tensors int quantized)
      if (Platform::is_dcu() &&
          is_compressed_tensors_int8_scheme(*weights_it,
                                            /*expected_dynamic=*/false) &&
          is_compressed_tensors_int8_scheme(*input_activations_it,
                                            /*expected_dynamic=*/true)) {
        quant_args.bits() = 8;
        quant_args.moe_weight_bits() = 8;
        quant_args.activation_dynamic() = true;
        quant_args.is_compressed_tensors_w8a8_dynamic() = true;
        if (const auto ignore = reader.value<std::vector<std::string>>(
                "quantization_config.ignore");
            ignore.has_value()) {
          quant_args.ignored_modules() = *ignore;
        }
        return true;
      }
      continue;
    }

    quant_args.quant_method() = kQuantMethodFp8;
    quant_args.bits() = 8;
    quant_args.moe_weight_bits() = 8;

    auto dynamic_it = input_activations_it->find("dynamic");
    if (dynamic_it != input_activations_it->end() && !dynamic_it->is_null()) {
      quant_args.activation_dynamic() = dynamic_it->get<bool>();
    }
    if (const auto ignore = reader.value<std::vector<std::string>>(
            "quantization_config.ignore");
        ignore.has_value()) {
      quant_args.ignored_modules() = *ignore;
    }
    return true;
  }

  LOG(ERROR) << "Failed to find an FP8 config_group in "
                "quantization_config.config_groups for compressed-tensors "
                "quantization.";
  return false;
}

bool validate_smoothquant_mixed_w4a8(const JsonReader& reader,
                                     QuantArgs& quant_args,
                                     bool only_expert_per_group) {
  const auto expert_weight_precision =
      reader.value<std::string>("quantization_config.expert_weight_precision");
  const int64_t experts_weight_bits =
      reader.value_or<int64_t>("quantization_config.experts_weight_bits", 0);

  const bool is_w4a8 = (expert_weight_precision.has_value() &&
                        boost::iequals(*expert_weight_precision, "int4")) ||
                       experts_weight_bits == 4;
  if (!is_w4a8) {
    return true;
  }

  quant_args.moe_weight_bits() = 4;

  if (!only_expert_per_group) {
    LOG(ERROR) << "DeepSeek mixed W4A8 requires "
                  "quantization_config.only_expert_per_group=true.";
    return false;
  }
  if (quant_args.quant_method() != "smoothquant") {
    LOG(ERROR) << "DeepSeek mixed W4A8 only supports "
                  "quant_method=smoothquant.";
    return false;
  }
  if (quant_args.bits() != 8) {
    LOG(ERROR) << "DeepSeek mixed W4A8 requires quantization_config.bits=8, "
               << "but got bits=" << quant_args.bits();
    return false;
  }

  const auto weight_precision =
      reader.value<std::string>("quantization_config.weight_precision");
  if (weight_precision.has_value() &&
      !boost::iequals(*weight_precision, "int8")) {
    LOG(ERROR) << "DeepSeek mixed W4A8 requires weight_precision=int8, but got "
               << *weight_precision;
    return false;
  }

  const auto activation_precision =
      reader.value<std::string>("quantization_config.activation_precision");
  if (activation_precision.has_value() &&
      !boost::iequals(*activation_precision, "int8")) {
    LOG(ERROR)
        << "DeepSeek mixed W4A8 requires activation_precision=int8, but got "
        << *activation_precision;
    return false;
  }

  const auto expert_activation_precision = reader.value<std::string>(
      "quantization_config.expert_activation_precision");
  if (expert_activation_precision.has_value() &&
      !boost::iequals(*expert_activation_precision, "int8")) {
    LOG(ERROR) << "DeepSeek mixed W4A8 requires "
                  "expert_activation_precision=int8, but got "
               << *expert_activation_precision;
    return false;
  }
  return true;
}

bool try_parse_layer_id_with_prefix(const std::string& tensor_name,
                                    const std::string& prefix,
                                    int64_t* layer_id) {
  if (!absl::StartsWith(tensor_name, prefix) || layer_id == nullptr) {
    return false;
  }

  const size_t begin = prefix.size();
  if (begin >= tensor_name.size() ||
      !std::isdigit(static_cast<unsigned char>(tensor_name[begin]))) {
    return false;
  }

  int64_t value = 0;
  size_t end = begin;
  while (end < tensor_name.size() &&
         std::isdigit(static_cast<unsigned char>(tensor_name[end]))) {
    const int digit = tensor_name[end] - '0';
    if (value > (std::numeric_limits<int64_t>::max() - digit) / 10) {
      return false;
    }
    value = value * 10 + digit;
    ++end;
  }
  if (end >= tensor_name.size() || tensor_name[end] != '.') {
    return false;
  }

  *layer_id = value;
  return true;
}

bool try_parse_layer_id(const std::string& tensor_name, int64_t* layer_id) {
  static const std::vector<std::string> kLayerPrefixes = {
      "model.layers.", "layers.", "transformer.layers."};
  for (const auto& prefix : kLayerPrefixes) {
    if (try_parse_layer_id_with_prefix(tensor_name, prefix, layer_id)) {
      return true;
    }
  }
  return false;
}

std::string infer_quantize_type_from_desc(bool has_w4a8_dynamic,
                                          bool has_w8a8_dynamic) {
  if (has_w4a8_dynamic) {
    return "w4a8_dynamic";
  }
  if (has_w8a8_dynamic) {
    return "w8a8_dynamic";
  }
  return "";
}

class ScopedMmap {
 public:
  ~ScopedMmap() {
    if (mapped_addr_ != MAP_FAILED) {
      munmap(mapped_addr_, mapped_size_);
    }
    if (fd_ >= 0) {
      close(fd_);
    }
  }

  bool map_read_only(const std::string& file_path) {
    fd_ = open(file_path.c_str(), O_RDONLY);
    if (fd_ < 0) {
      PLOG(ERROR) << "Failed to open safetensors file: " << file_path;
      return false;
    }

    struct stat sb;
    if (fstat(fd_, &sb) != 0) {
      PLOG(ERROR) << "Failed to stat safetensors file: " << file_path;
      return false;
    }
    if (sb.st_size <= 0) {
      LOG(ERROR) << "Safetensors file is empty: " << file_path;
      return false;
    }
    mapped_size_ = static_cast<size_t>(sb.st_size);
    mapped_addr_ = mmap(nullptr, mapped_size_, PROT_READ, MAP_PRIVATE, fd_, 0);
    if (mapped_addr_ == MAP_FAILED) {
      PLOG(ERROR) << "Failed to mmap safetensors file: " << file_path;
      return false;
    }
    return true;
  }

  const uint8_t* data() const {
    return static_cast<const uint8_t*>(mapped_addr_);
  }

  size_t size() const { return mapped_size_; }

 private:
  void* mapped_addr_ = MAP_FAILED;
  size_t mapped_size_ = 0;
  int fd_ = -1;
};

bool try_compute_tensor_nbytes(const View* tensor_view, int64_t* nbytes) {
  if (tensor_view == nullptr || nbytes == nullptr) {
    return false;
  }

  if (tensor_view->stop < tensor_view->start) {
    return false;
  }
  const size_t byte_size = tensor_view->stop - tensor_view->start;
  if (byte_size > static_cast<size_t>(std::numeric_limits<int64_t>::max())) {
    return false;
  }
  *nbytes = static_cast<int64_t>(byte_size);
  return true;
}

bool log_safetensors_error(::Status status,
                           const char* op,
                           const std::string& weights_file,
                           const char* tensor_name = nullptr) {
  if (status == ::Status::Ok) {
    return true;
  }
  if (tensor_name == nullptr) {
    LOG(ERROR) << op << " failed for " << weights_file
               << ", status=" << static_cast<int>(status);
  } else {
    LOG(ERROR) << op << " failed for " << weights_file
               << ", tensor_name=" << tensor_name
               << ", status=" << static_cast<int>(status);
  }
  return false;
}

void check_safetensors_cleanup(::Status status,
                               const char* op,
                               const std::string& weights_file,
                               const char* tensor_name = nullptr) {
  if (tensor_name == nullptr) {
    CHECK(status == ::Status::Ok)
        << op << " cleanup failed for " << weights_file
        << ", status=" << static_cast<int>(status);
  } else {
    CHECK(status == ::Status::Ok)
        << op << " cleanup failed for " << weights_file
        << ", tensor_name=" << tensor_name
        << ", status=" << static_cast<int>(status);
  }
}

}  // namespace

bool load_quant_cfg(const JsonReader& reader, QuantArgs& quant_args) {
  if (!reader.contains("quantization_config")) {
    return true;
  }

  if (auto v = reader.value<std::string>("quantization_config.quant_method")) {
    quant_args.quant_method() = v.value();
  }
  // Only CUDA and DCU currently adapts this compressed-tensors JSON layout.
  // For other backends, skip this special parsing path and continue with the
  // generic quantization config parsing path.
  if ((Platform::is_cuda() || Platform::is_dcu()) &&
      try_load_compressed_tensors_quant_cfg(reader, quant_args)) {
    return true;
  }
  if (auto v = reader.value<int64_t>("quantization_config.bits")) {
    quant_args.bits() = v.value();
    quant_args.moe_weight_bits() = v.value();
  }
  if (auto v = reader.value<int64_t>("quantization_config.group_size")) {
    quant_args.group_size() = v.value();
  }
  if (auto v = reader.value<bool>("quantization_config.desc_act")) {
    quant_args.desc_act() = v.value();
  }
  if (auto v = reader.value<bool>("quantization_config.sym")) {
    quant_args.is_sym() = v.value();
  }
  if (auto v =
          reader.value<std::string>("quantization_config.activation_scheme")) {
    std::string activation_scheme = v.value();
    if (boost::iequals(activation_scheme, "static")) {
      quant_args.activation_dynamic() = false;
    } else if (boost::iequals(activation_scheme, "dynamic")) {
      quant_args.activation_dynamic() = true;
    } else {
      LOG(ERROR) << "quantization_config.activation_scheme only support "
                    "dynamic and static.";
      return false;
    }
  }
  if (auto v = reader.value<std::string>("quantization_config.fmt")) {
    // TODO(liangzhiwei20): check fp8 quantization format
    quant_args.fmt() = v.value();
  }
  if (reader.contains("quantization_config.weight_block_size")) {
    const auto& data = reader.data();
    quant_args.weight_block_size() =
        data["quantization_config"]["weight_block_size"]
            .get<std::vector<int64_t>>();
  }

  bool only_expert_per_group = false;
  if (auto v =
          reader.value<bool>("quantization_config.only_expert_per_group")) {
    only_expert_per_group = v.value();
  }

  return validate_smoothquant_mixed_w4a8(
      reader, quant_args, only_expert_per_group);
}

HFModelLoader::HFModelLoader(const std::string& model_weights_path)
    : model_weights_path_(model_weights_path) {
  CHECK(load_args(model_weights_path))
      << "Failed to load model args from " << model_weights_path;
  // try to load safetensors first
  for (const auto& entry :
       std::filesystem::directory_iterator(model_weights_path)) {
    // load bin or safe tensors
    if (entry.path().extension() == ".safetensors") {
      model_weights_files_.push_back(entry.path().string());
    }
  }
  CHECK(!model_weights_files_.empty())
      << "Failed to find model weights files in " << model_weights_path;
  // sort the model weights files by name
  std::sort(model_weights_files_.begin(), model_weights_files_.end());

  threadpool_ = std::make_unique<ThreadPool>(
      /*num_threads=*/32,
      /*cpu_binding=*/false,
      /*pool_name=*/"HFModelLoader.load_weights");
  if (::xllm::ModelConfig::get_instance().backend() == "rec" &&
      is_onerec_model_type(args_.model_type())) {
    CHECK(load_rec_vocab(model_weights_path))
        << "Failed to load rec content from " << model_weights_path;
  }
}

std::unique_ptr<Tokenizer> HFModelLoader::tokenizer() const {
  return TokenizerFactory::create_tokenizer(model_weights_path_,
                                            tokenizer_args_);
}

std::vector<std::unique_ptr<StateDict>>& HFModelLoader::get_state_dicts() {
  if (state_dicts_.empty()) {
    // load state dict
    state_dicts_.reserve(model_weights_files_.size());
    auto file_cnt = model_weights_files_.size();
    BlockingCounter counter(file_cnt);
    state_dicts_.resize(model_weights_files_.size());
    for (int file_id = 0; file_id < file_cnt; file_id++) {
      threadpool_->schedule([this, file_id, &counter]() mutable {
        LOG(INFO) << "Loading model weights from "
                  << model_weights_files_[file_id];
        state_dicts_[file_id] = std::move(
            StateDictFromSafeTensor::load(model_weights_files_[file_id]));
        counter.decrement_count();
      });
    }
    counter.wait();
  }
  return state_dicts_;
}

int64_t HFModelLoader::get_total_weight_size() const {
  // find the index json file
  std::string index_json_path;
  for (const auto& entry :
       std::filesystem::directory_iterator(model_weights_path_)) {
    if (absl::EndsWith(entry.path().string(), ".index.json")) {
      index_json_path = entry.path().string();
      break;
    }
  }

  if (index_json_path.empty()) {
    LOG(ERROR) << "Failed to find .index.json file in " << model_weights_path_;
    return -1;
  }

  JsonReader reader;
  if (!reader.parse(index_json_path)) {
    LOG(ERROR) << "Failed to parse json file " << index_json_path;
    return -1;
  }

  auto total_size = reader.value<int64_t>("metadata.total_size");
  if (!total_size.has_value()) {
    LOG(ERROR) << "Failed to find metadata.total_size in " << index_json_path;
    return -1;
  }

  int64_t result = total_size.value();

  // When tie_word_embeddings is true, lm_head shares weight with word_embedding
  // in the checkpoint, but we need to allocate memory for both during
  // inference. Add the size of word_embedding weight (vocab_size * hidden_size
  // * bytes_per_elem)
  if (args_.tie_word_embeddings()) {
    auto scalar_type = try_get_scalar_type_from_string(args_.dtype());
    CHECK(scalar_type.has_value()) << "Unsupported dtype: " << args_.dtype();
    int64_t bytes_per_elem = torch::elementSize(*scalar_type);
    int64_t embedding_size =
        args_.vocab_size() * args_.hidden_size() * bytes_per_elem;
    result += embedding_size;
    LOG(INFO) << "tie_word_embeddings is true, adding embedding weight size: "
              << embedding_size << " bytes";
  }

  return result;
}

int64_t HFModelLoader::get_non_decoder_weight_size() const {
  auto scalar_type = try_get_scalar_type_from_string(args_.dtype());
  if (!scalar_type.has_value() ||
      (*scalar_type != torch::kFloat16 && *scalar_type != torch::kBFloat16 &&
       *scalar_type != torch::kFloat32 && *scalar_type != torch::kFloat64 &&
       *scalar_type != torch::kInt8)) {
    LOG(WARNING) << "get_non_decoder_weight_size: unsupported dtype "
                 << args_.dtype() << ", falling back to total_weight_size";
    return get_total_weight_size();
  }
  int64_t bytes_per_elem = torch::elementSize(*scalar_type);

  // embed_tokens: vocab_size * hidden_size
  int64_t embed_size =
      args_.vocab_size() * args_.hidden_size() * bytes_per_elem;

  // final norm: hidden_size
  int64_t norm_size = args_.hidden_size() * bytes_per_elem;

  // lm_head: vocab_size * hidden_size
  // When tie_word_embeddings=true, lm_head shares the checkpoint weight with
  // embed_tokens, but still gets its own device memory allocation at runtime.
  int64_t lm_head_size =
      args_.vocab_size() * args_.hidden_size() * bytes_per_elem;

  int64_t result = embed_size + norm_size + lm_head_size;
  LOG(INFO) << "get_non_decoder_weight_size: embed=" << embed_size
            << " norm=" << norm_size << " lm_head=" << lm_head_size
            << " total=" << result;
  return result;
}

int64_t HFModelLoader::get_max_decoder_layer_weight_size() const {
  constexpr int64_t kInvalidLayerSize = -1;
  if (args_.n_layers() <= 0) {
    LOG(ERROR) << "Invalid n_layers for decoder size estimation: "
               << args_.n_layers();
    return kInvalidLayerSize;
  }

  std::vector<int64_t> layer_sizes(static_cast<size_t>(args_.n_layers()), 0);

  for (const auto& weights_file : model_weights_files_) {
    ScopedMmap mapping;
    if (!mapping.map_read_only(weights_file)) {
      return kInvalidLayerSize;
    }

    Handle* handle = nullptr;
    if (!log_safetensors_error(
            safetensors_deserialize(&handle, mapping.data(), mapping.size()),
            "safetensors_deserialize",
            weights_file)) {
      return kInvalidLayerSize;
    }
    xllm::ScopeGuard handle_guard([&] {
      if (handle != nullptr) {
        check_safetensors_cleanup(
            safetensors_destroy(handle), "safetensors_destroy", weights_file);
        handle = nullptr;
      }
    });

    const char* const* tensor_names = nullptr;
    size_t num_tensors = 0;
    if (!log_safetensors_error(
            safetensors_names(handle, &tensor_names, &num_tensors),
            "safetensors_names",
            weights_file)) {
      return kInvalidLayerSize;
    }
    xllm::ScopeGuard names_guard([&] {
      if (tensor_names != nullptr) {
        check_safetensors_cleanup(
            safetensors_free_names(tensor_names, num_tensors),
            "safetensors_free_names",
            weights_file);
        tensor_names = nullptr;
        num_tensors = 0;
      }
    });

    for (size_t i = 0; i < num_tensors; ++i) {
      const char* tensor_name_cstr = tensor_names[i];
      const std::string tensor_name(tensor_name_cstr);
      int64_t layer_id = -1;
      if (!try_parse_layer_id(tensor_name, &layer_id) || layer_id < 0 ||
          layer_id >= args_.n_layers()) {
        continue;
      }

      View* tensor_view = nullptr;
      if (!log_safetensors_error(
              safetensors_get_tensor(handle, &tensor_view, tensor_name_cstr),
              "safetensors_get_tensor",
              weights_file,
              tensor_name_cstr)) {
        return kInvalidLayerSize;
      }
      xllm::ScopeGuard tensor_guard([&] {
        if (tensor_view != nullptr) {
          check_safetensors_cleanup(safetensors_free_tensor(tensor_view),
                                    "safetensors_free_tensor",
                                    weights_file,
                                    tensor_name_cstr);
          tensor_view = nullptr;
        }
      });

      int64_t tensor_nbytes = 0;
      if (!try_compute_tensor_nbytes(tensor_view, &tensor_nbytes)) {
        LOG(ERROR) << "Failed to compute tensor bytes for tensor_name="
                   << tensor_name << " in " << weights_file;
        return kInvalidLayerSize;
      }

      if (layer_sizes[static_cast<size_t>(layer_id)] >
          std::numeric_limits<int64_t>::max() - tensor_nbytes) {
        LOG(ERROR) << "Decoder layer size overflow while accumulating layer "
                   << layer_id;
        return kInvalidLayerSize;
      }
      layer_sizes[static_cast<size_t>(layer_id)] += tensor_nbytes;
    }
  }

  int64_t max_layer_size = 0;
  int64_t observed_layers = 0;
  for (const int64_t layer_size : layer_sizes) {
    if (layer_size > 0) {
      ++observed_layers;
      max_layer_size = std::max(max_layer_size, layer_size);
    }
  }
  if (max_layer_size <= 0) {
    LOG(ERROR) << "Failed to detect decoder-layer tensor sizes from "
                  "safetensors metadata.";
    return kInvalidLayerSize;
  }

  if (observed_layers != args_.n_layers()) {
    LOG(WARNING) << "Observed decoder layer sizes for " << observed_layers
                 << "/" << args_.n_layers()
                 << " layers while estimating max layer size.";
  }

  LOG(INFO) << "get_max_decoder_layer_weight_size: max_layer_size="
            << max_layer_size << ", observed_layers=" << observed_layers << "/"
            << args_.n_layers();
  return max_layer_size;
}

bool HFModelLoader::load_rec_vocab(const std::string& model_weights_path) {
  if (!tokenizer_args_.vocab_file().empty()) {
    std::filesystem::path path = model_weights_path;
    std::string model_version = path.filename();
    std::string vocab_full_path =
        path.append(tokenizer_args_.vocab_file()).string();

    LOG(INFO) << "Model_version: " << model_version
              << ", vocab_full_path: " << vocab_full_path;

    CHECK(nullptr != VersionSingleton<RecVocabDict>::GetInstance(model_version))
        << "Failed to get vocab dict instance";
    CHECK(VersionSingleton<RecVocabDict>::GetInstance(model_version)
              ->initialize(vocab_full_path))
        << "Failed to initialize vocab dict from " << vocab_full_path;
  } else {
    if (::xllm::RecConfig::get_instance().enable_constrained_decoding()) {
      LOG(ERROR) << "Vocab file is not set for OneRec REC tokenizer under "
                 << model_weights_path
                 << ". Constrained decoding requires `vocab_file` in "
                    "tokenizer_config.json.";
      return false;
    }

    LOG(WARNING) << "Vocab file is not set for OneRec REC tokenizer under "
                 << model_weights_path
                 << ". Skip vocab dict initialization because constrained "
                    "decoding is disabled.";
    return true;
  }

  return true;
}

bool HFModelLoader::load_args(const std::string& model_weights_path) {
  if (!load_model_args(model_weights_path)) {
    LOG(ERROR) << "Failed to load model args from " << model_weights_path;
    return false;
  }

  if (!load_quant_args(model_weights_path)) {
    LOG(ERROR) << "Failed to load quant args from " << model_weights_path;
    return false;
  }

  if (!load_tokenizer_args(model_weights_path)) {
    LOG(ERROR) << "Failed to load tokenizer args from " << model_weights_path;
    return false;
  }

  if (!load_image_preprocessor_args(model_weights_path)) {
    LOG(ERROR) << "Failed to load image preprocess args from "
               << model_weights_path;
    return false;
  }

  if (!load_video_preprocessor_args(model_weights_path)) {
    LOG(ERROR) << "Failed to load video preprocess args from "
               << model_weights_path;
    return false;
  }

  // Some hacky logics to support loading of old models
  // always use float16 for quantization
  // TODO: support quantization for other data types
  if (!quant_args_.quant_method().empty() && args_.dtype() != "bfloat16") {
    LOG(WARNING) << "Overwriting dtype from " << args_.dtype()
                 << " to float16 for quantization";
    args_.dtype() = "bfloat16";
  }

  return true;
}

bool HFModelLoader::load_model_args(const std::string& model_weights_path) {
  JsonReader reader;
  const std::string args_file_path = model_weights_path + "/config.json";
  if (!reader.parse(args_file_path)) {
    LOG(ERROR) << "Failed to parse model args file: " << args_file_path;
    return false;
  }

  const std::string model_type =
      util::get_model_type(reader,
                           std::filesystem::path(model_weights_path),
                           ModelConfig::get_instance().backend());

  std::string resolved_model_type;
  std::string error_message;
  if (!resolve_model_registration_name(
          model_type, &resolved_model_type, &error_message)) {
    LOG(ERROR) << error_message;
    return false;
  }

  auto model_args_loader =
      ModelRegistry::get_model_args_loader(resolved_model_type);
  if (model_args_loader == nullptr) {
    LOG(ERROR) << "Failed to find model args loader for model type "
               << resolved_model_type;
    return false;
  }
  const JsonReader config_reader = normalize_config_torch_dtype(reader);
  model_args_loader(config_reader, &args_);
  args_.enable_mla(
      util::should_enable_mla(std::filesystem::path(model_weights_path),
                              ModelConfig::get_instance().backend()));

  return true;
}

bool HFModelLoader::load_quant_args(const std::string& model_weights_path) {
  JsonReader reader;
  const std::string args_file_path = model_weights_path + "/config.json";
  if (!reader.parse(args_file_path)) {
    LOG(ERROR) << "Failed to parse model args file: " << args_file_path;
    return false;
  }

  const JsonReader config_reader = normalize_config_torch_dtype(reader);

  if (!load_quant_cfg(config_reader, quant_args_)) {
    return false;
  }

  // load quantization args for npu if exists
  const bool has_config_quantize = config_reader.contains("quantize");
  if (has_config_quantize) {
    quant_args_.quantize_type() =
        config_reader.value_or<std::string>("quantize", "");
  }
  if (config_reader.contains("torch_dtype")) {
    quant_args_.torch_dtype() =
        config_reader.value_or<std::string>("torch_dtype", "");
  }
  if (auto v = reader.value<std::string>("quantization_config.version")) {
    quant_args_.quant_version() = v.value();
  }
  if (quant_args_.quantize_type().empty() &&
      boost::iequals(quant_args_.quant_method(), "w4a8_dynamic")) {
    quant_args_.quantize_type() = "w4a8_dynamic";
  }
  bool model_is_w4a8_dynamic =
      boost::iequals(quant_args_.quantize_type(), "w4a8_dynamic");
  if (boost::iequals(quant_args_.quantize_type(), "w4a8_dynamic")) {
    quant_args_.bits() = 8;
    quant_args_.moe_weight_bits() = 4;
    quant_args_.activation_dynamic() = true;
  }

  bool explicit_quant_group_size =
      reader.contains("quantization_config.group_size");

  // Ascend-style quant description file.
  // This file provides per-weight quant types, e.g.
  // "layers.0.attn.wq_a.weight": "W8A8_DYNAMIC".
  JsonReader quant_desc_reader;
  const std::string quant_desc_file_path =
      model_weights_path + "/quant_model_description.json";
  if (quant_desc_reader.parse(quant_desc_file_path)) {
    std::unordered_map<std::string, std::string> quant_descs;
    const auto quant_desc_data = quant_desc_reader.data();
    bool desc_model_quant_is_w4a8_dynamic = false;
    bool desc_has_w4a8 = false;
    bool desc_has_w8a8 = false;
    bool desc_has_w4a8_dynamic = false;
    bool desc_has_w8a8_dynamic = false;
    if (!quant_desc_data.is_object()) {
      LOG(WARNING) << "quant_model_description is not a JSON object: "
                   << quant_desc_file_path;
    } else {
      quant_descs.reserve(quant_desc_data.size());
      for (auto it = quant_desc_data.begin(); it != quant_desc_data.end();
           ++it) {
        if (!it.value().is_string()) {
          continue;
        }
        // Keep only tensor-like keys (contains '.'); skip metadata fields such
        // as model_quant_type/version.
        if (it.key().find('.') == std::string::npos) {
          continue;
        }
        const std::string quant_type = it.value().get<std::string>();
        std::string quant_type_lower = quant_type;
        boost::algorithm::to_lower(quant_type_lower);
        if (boost::algorithm::starts_with(quant_type_lower, "w4a8")) {
          desc_has_w4a8 = true;
        } else if (boost::algorithm::starts_with(quant_type_lower, "w8a8")) {
          desc_has_w8a8 = true;
        }
        if (boost::iequals(quant_type, "w4a8_dynamic")) {
          desc_has_w4a8_dynamic = true;
        } else if (boost::iequals(quant_type, "w8a8_dynamic")) {
          desc_has_w8a8_dynamic = true;
        }
        quant_descs.emplace(it.key(), quant_type);
        std::string mapped_key = it.key();
        const std::string packed_weight = "weight_packed";
        if (const auto pos = mapped_key.find(packed_weight);
            pos != std::string::npos) {
          mapped_key.replace(pos, packed_weight.size(), "weight");
          quant_descs.emplace(std::move(mapped_key), quant_type);
        }
      }
    }
    quant_args_.quant_descs() = std::move(quant_descs);
    if (desc_has_w4a8) {
      quant_args_.quant_method() = kQuantMethodAscendInt4;
    } else if (desc_has_w8a8) {
      quant_args_.quant_method() = kQuantMethodAscendInt8;
    }
    LOG(INFO) << "Loaded quant_model_description from " << quant_desc_file_path
              << ", quant_desc_count=" << quant_args_.quant_descs().size()
              << ", quant_method="
              << (quant_args_.quant_method().empty()
                      ? "<empty>"
                      : quant_args_.quant_method());

    const std::string desc_inferred_quantize_type =
        infer_quantize_type_from_desc(desc_has_w4a8_dynamic,
                                      desc_has_w8a8_dynamic);
    if (!has_config_quantize && !desc_inferred_quantize_type.empty()) {
      quant_args_.quantize_type() = desc_inferred_quantize_type;
    }
    if (quant_args_.quantize_type().empty()) {
      if (auto v = quant_desc_reader.value<std::string>("model_quant_type")) {
        auto quantize_type = v.value();
        boost::algorithm::to_lower(quantize_type);
        quant_args_.quantize_type() = quantize_type;
        LOG(INFO) << "Fallback quantize_type from quant_model_description: "
                  << quant_args_.quantize_type();
      }
    }
    desc_model_quant_is_w4a8_dynamic =
        boost::iequals(quant_args_.quantize_type(), "w4a8_dynamic") ||
        desc_has_w4a8_dynamic;
    model_is_w4a8_dynamic =
        model_is_w4a8_dynamic || desc_model_quant_is_w4a8_dynamic;
    if (desc_model_quant_is_w4a8_dynamic) {
      if (quant_args_.quantize_type().empty()) {
        quant_args_.quantize_type() = "w4a8_dynamic";
      }
      quant_args_.bits() = 8;
      quant_args_.moe_weight_bits() = 4;
      quant_args_.activation_dynamic() = true;
    }
    if (auto v = quant_desc_reader.value<int64_t>("group_size")) {
      quant_args_.group_size() = v.value();
      explicit_quant_group_size = true;
    } else if (desc_model_quant_is_w4a8_dynamic && !explicit_quant_group_size) {
      // Match vllm-ascend W4A8_DYNAMIC's default. A config value of 0 is still
      // respected as the per-channel mode.
      quant_args_.group_size() = 256;
      explicit_quant_group_size = true;
    }
    if (auto v = quant_desc_reader.value<std::string>("version")) {
      quant_args_.quant_version() = v.value();
    }
  }
  if (model_is_w4a8_dynamic && !explicit_quant_group_size) {
    quant_args_.group_size() = 256;
  }
  if (model_is_w4a8_dynamic && quant_args_.quant_version() != "1.0.0") {
    LOG(ERROR) << "W4A8_DYNAMIC only supports quant_version 1.0.0, got "
               << (quant_args_.quant_version().empty()
                       ? "<empty>"
                       : quant_args_.quant_version())
               << ". Please provide quant_model_description.json version "
                  "\"1.0.0\".";
    return false;
  }

  // awq quantization args
  JsonReader awq_reader;
  const std::string quant_args_file_path =
      model_weights_path + "/quant_config.json";
  if (awq_reader.parse(quant_args_file_path)) {
    // hardcode the quant_method to awq if not exists
    quant_args_.quant_method() =
        awq_reader.value_or<std::string>("quant_method", "awq");

    if (auto v = awq_reader.value<int64_t>("w_bit")) {
      quant_args_.bits() = v.value();
    }
    if (auto v = awq_reader.value<int64_t>("q_group_size")) {
      quant_args_.group_size() = v.value();
    }
  }

  // gptq quantization args
  const std::string gptq_args_file_path =
      model_weights_path + "/quantize_config.json";
  JsonReader gptq_reader;
  if (gptq_reader.parse(gptq_args_file_path)) {
    // hardcode the quant_method to gptq if not exists
    quant_args_.quant_method() =
        gptq_reader.value_or<std::string>("quant_method", "gptq");

    if (auto v = gptq_reader.value<int64_t>("bits")) {
      quant_args_.bits() = v.value();
    }
    if (auto v = gptq_reader.value<int64_t>("group_size")) {
      quant_args_.group_size() = v.value();
    }
    if (auto v = gptq_reader.value<bool>("desc_act")) {
      quant_args_.desc_act() = v.value();
    }
    if (auto v = gptq_reader.value<bool>("sym")) {
      quant_args_.is_sym() = v.value();
    }
  }

  return true;
}

namespace {
std::optional<std::string> load_chat_template_file(const std::string& dir) {
  // chat_template.json
  const std::string chat_template_path = dir + "/chat_template.json";
  JsonReader reader;
  if (reader.parse(chat_template_path);
      auto v = reader.value<std::string>("chat_template")) {
    return v;
  }
  // chat_template.jinja
  const std::string raw_chat_template_path = dir + "/chat_template.jinja";
  std::ifstream file(raw_chat_template_path);
  if (file.is_open()) {
    std::ostringstream content;
    content << file.rdbuf();
    file.close();
    return content.str();
  }
  return std::nullopt;
}
}  // namespace

bool HFModelLoader::load_tokenizer_args(const std::string& model_weights_path) {
  // tokenizer args from tokenizer_config.json
  JsonReader tokenizer_reader;
  const std::string tokenizer_args_file_path =
      model_weights_path_ + "/tokenizer_config.json";

  // check if tokenizer.json exists, if exists, set the tokenizer type to fast
  const std::string tokenizer_json_path =
      model_weights_path + "/tokenizer.json";
  if (std::filesystem::exists(tokenizer_json_path)) {
    tokenizer_args_.tokenizer_type() = "fast";
    tokenizer_args_.vocab_file() = tokenizer_json_path;
  }

  if (tokenizer_reader.parse(tokenizer_args_file_path)) {
    // read chat template if exists
    if (auto v = load_chat_template_file(model_weights_path_)) {
      tokenizer_args_.chat_template() = v.value();
    } else if (auto v = tokenizer_reader.value<std::string>("chat_template")) {
      tokenizer_args_.chat_template() = v.value();
    }
    if (auto v = tokenizer_reader.value<bool>("add_bos_token")) {
      tokenizer_args_.add_bos_token() = v.value();
    }
    if (auto v = tokenizer_reader.value<bool>("add_eos_token")) {
      tokenizer_args_.add_eos_token() = v.value();
    }
    if (auto v = tokenizer_reader.value<std::string>("tokenizer_class")) {
      tokenizer_args_.tokenizer_class() = v.value();
    }
    // read bos_token
    if (auto v = tokenizer_reader.value<std::string>("bos_token.content")) {
      tokenizer_args_.bos_token() = v.value();
    } else if (auto v = tokenizer_reader.value<std::string>("bos_token")) {
      tokenizer_args_.bos_token() = v.value();
    }
    // read eos_token
    if (auto v = tokenizer_reader.value<std::string>("eos_token.content")) {
      tokenizer_args_.eos_token() = v.value();
    } else if (auto v = tokenizer_reader.value<std::string>("eos_token")) {
      tokenizer_args_.eos_token() = v.value();
    }
    // read pad_token
    if (auto v = tokenizer_reader.value<std::string>("pad_token.content")) {
      tokenizer_args_.pad_token() = v.value();
    } else if (auto v = tokenizer_reader.value<std::string>("pad_token")) {
      tokenizer_args_.pad_token() = v.value();
    }
  }

  auto tokenizer_args_loader =
      ModelRegistry::get_tokenizer_args_loader(args_.model_type());
  if (tokenizer_args_loader != nullptr) {
    if (!tokenizer_args_loader(tokenizer_reader, &tokenizer_args_)) {
      LOG(ERROR) << "Failed to load tokenizer args from "
                 << tokenizer_args_file_path;
      return false;
    }
  }

  return true;
}

bool HFModelLoader::load_image_preprocessor_args(
    const std::string& model_weights_path) {
  // image preprocessor args
  JsonReader image_preprocess_reader;
  const std::string image_preprocess_file_path =
      model_weights_path + "/preprocessor_config.json";
  if (image_preprocess_reader.parse(image_preprocess_file_path)) {
    LOG(INFO) << "Success to parse image preprocess args file: "
              << image_preprocess_file_path;
    args_.mm_image_do_center_crop() =
        image_preprocess_reader.value_or<bool>("do_center_crop", false);
    args_.mm_image_crop_height_size() =
        image_preprocess_reader.value_or<int>("crop_size.height", 335);
    args_.mm_image_crop_width_size() =
        image_preprocess_reader.value_or<int>("crop_size.width", 335);

    args_.mm_image_do_resize() =
        image_preprocess_reader.value_or<bool>("do_resize", false);
    args_.mm_image_resize_shortest_edge() =
        image_preprocess_reader.value_or<int>("size.shortest_edge", 335);
    args_.mm_image_resample() =
        image_preprocess_reader.value_or<int>("resample", 335);

    args_.mm_image_do_rescale() =
        image_preprocess_reader.value_or<bool>("do_rescale", false);
    args_.mm_image_rescale_factor() =
        image_preprocess_reader.value_or<double>("rescale_factor", 0);

    args_.mm_image_do_normalize() =
        image_preprocess_reader.value_or<bool>("do_normalize", false);

    const auto& image_prerocess_data = image_preprocess_reader.data();
    if (image_preprocess_reader.contains("image_mean")) {
      args_.mm_image_normalize_mean() =
          image_prerocess_data["image_mean"].get<std::vector<double>>();
    }

    if (image_preprocess_reader.contains("image_std")) {
      args_.mm_image_normalize_std() =
          image_prerocess_data["image_std"].get<std::vector<double>>();
    }

    if (image_preprocess_reader.contains("norm_mean")) {
      args_.mm_image_normalize_mean() =
          image_prerocess_data["norm_mean"].get<std::vector<double>>();
    }

    if (image_preprocess_reader.contains("norm_std")) {
      args_.mm_image_normalize_std() =
          image_prerocess_data["norm_std"].get<std::vector<double>>();
    }

    args_.mm_image_shortest_edge() =
        image_preprocess_reader.value_or<int>("size.shortest_edge", 0);

    args_.mm_image_longest_edge() =
        image_preprocess_reader.value_or<int>("size.longest_edge", 0);

    args_.mm_image_min_pixels() =
        image_preprocess_reader.value_or<int>("min_pixels", 0);

    args_.mm_image_max_pixels() =
        image_preprocess_reader.value_or<int>("max_pixels", 0);

    args_.mm_image_patch_size() =
        image_preprocess_reader.value_or<int>("patch_size", 0);

    args_.mm_image_temporal_patch_size() =
        image_preprocess_reader.value_or<int>("temporal_patch_size", 0);

    args_.mm_image_merge_size() =
        image_preprocess_reader.value_or<int>("merge_size", 0);

    args_.mm_image_feature_size() =
        image_preprocess_reader.value_or<int>("image_feature_size", 0);

    args_.mm_scale_resolution() =
        image_preprocess_reader.value_or<int>("scale_resolution", 0);

    args_.mm_slice_mode() =
        image_preprocess_reader.value_or<bool>("slice_mode", false);

    args_.mm_use_image_id() =
        image_preprocess_reader.value_or<bool>("use_image_id", false);
  }

  return true;
}

bool HFModelLoader::load_video_preprocessor_args(
    const std::string& model_weights_path) {
  // video preprocessor args
  JsonReader video_preprocess_reader;
  const std::string video_preprocess_file_path =
      model_weights_path + "/video_preprocessor_config.json";
  if (video_preprocess_reader.parse(video_preprocess_file_path)) {
    LOG(INFO) << "Success to parse video preprocess args file: "
              << video_preprocess_file_path;

    args_.mm_video_shortest_edge() =
        video_preprocess_reader.value_or<int>("size.shortest_edge", 0);

    args_.mm_video_longest_edge() =
        video_preprocess_reader.value_or<int>("size.longest_edge", 0);

    const auto& video_prerocess_data = video_preprocess_reader.data();
    if (video_preprocess_reader.contains("image_mean")) {
      args_.mm_video_normalize_mean() =
          video_prerocess_data["image_mean"].get<std::vector<double>>();
    }

    if (video_preprocess_reader.contains("image_std")) {
      args_.mm_video_normalize_std() =
          video_prerocess_data["image_std"].get<std::vector<double>>();
    }
    args_.mm_video_patch_size() =
        video_preprocess_reader.value_or<int>("patch_size", 0);

    args_.mm_video_temporal_patch_size() =
        video_preprocess_reader.value_or<int>("temporal_patch_size", 0);

    args_.mm_video_merge_size() =
        video_preprocess_reader.value_or<int>("merge_size", 0);

    args_.mm_video_do_rescale() =
        video_preprocess_reader.value_or<bool>("do_rescale", false);
  }

  return true;
}

}  // namespace xllm

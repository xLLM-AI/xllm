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

#include "dit_model_loader.h"

#include <absl/strings/match.h>
#include <absl/strings/str_replace.h>
#include <glog/logging.h>
#include <pybind11/embed.h>
#include <torch/torch.h>

#include <boost/algorithm/string.hpp>
#include <filesystem>
#include <fstream>
#include <vector>

#include "core/framework/tokenizer/tokenizer_args.h"
#include "core/framework/tokenizer/tokenizer_factory.h"
#include "core/util/dit_model_discovery.h"
#include "core/util/json_reader.h"
#include "models/model_registry.h"

namespace xllm {
DiTFolderLoader::DiTFolderLoader(const std::string& folder_path,
                                 const std::string& component_name,
                                 const std::string& model_type)
    : model_weights_path_(folder_path),
      component_name_(component_name),
      model_type_(model_type) {
  CHECK(load_args(folder_path))
      << "Failed to load model args from " << folder_path;
  // try to load safetensors first
  for (const auto& entry : std::filesystem::directory_iterator(folder_path)) {
    // load bin or safe tensors
    if (entry.path().extension() == ".safetensors") {
      model_weights_files_.push_back(entry.path().string());
    }
  }
  if (!model_weights_files_.empty()) {
    // sort the model weights files by name
    std::sort(model_weights_files_.begin(), model_weights_files_.end());
  }
}

std::unique_ptr<Tokenizer> DiTFolderLoader::tokenizer() const {
  // When vocab_file is already an absolute path (e.g. loaded from HF cache),
  // pass empty dir_path so the tokenizer uses it directly without prepending
  // model_weights_path_.
  const std::string& vocab = tokenizer_args_.vocab_file();
  const bool vocab_is_absolute = !vocab.empty() && vocab[0] == '/';
  const std::string dir_path = vocab_is_absolute ? "" : model_weights_path_;
  return TokenizerFactory::create_tokenizer(dir_path,
                                            tokenizer_args_,
                                            /*proxy*/ false);
}

std::vector<std::unique_ptr<StateDict>>& DiTFolderLoader::get_state_dicts() {
  if (state_dicts_.empty()) {
    // load state dict
    state_dicts_.reserve(model_weights_files_.size());
    for (auto& model_weights_file : model_weights_files_) {
      LOG(INFO) << "Loading model weights from " << model_weights_file;
      state_dicts_.emplace_back(
          StateDictFromSafeTensor::load(model_weights_file));
    }
  }
  return state_dicts_;
}

bool DiTFolderLoader::load_args(const std::string& model_weights_path) {
  // model_args must be loaded first: it populates text_encoder_model_ which
  // load_tokenizer_args uses as a fallback when no tokenizer files are present.
  if (!load_model_args(model_weights_path)) {
    LOG(ERROR) << "Failed to load model args from " << model_weights_path;
    return false;
  }

  if (!load_tokenizer_args(model_weights_path)) {
    LOG(ERROR) << "Failed to load tokenizer args from " << model_weights_path;
    return false;
  }

  if (!load_quant_args(model_weights_path)) {
    LOG(WARNING) << "Failed to load quant args from " << model_weights_path;
    // Non-fatal: not all models have quantization configs.
  }

  return true;
}

bool DiTFolderLoader::load_model_args(const std::string& model_weights_path) {
  bool has_safetensors = false;
  std::filesystem::path model_dir(model_weights_path);

  if (std::filesystem::is_directory(model_dir)) {
    for (const auto& entry : std::filesystem::directory_iterator(model_dir)) {
      if (entry.path().extension() == ".safetensors") {
        has_safetensors = true;
        break;
      }
    }
  } else {
    LOG(ERROR) << "Model path is not a valid directory: " << model_weights_path;
    return false;
  }

  auto load_json_config = [&](const std::string& json_filename) -> bool {
    JsonReader reader;
    std::string json_path = model_weights_path + "/" + json_filename;

    if (!std::filesystem::exists(json_path)) {
      LOG(WARNING) << "JSON config file not found: " << json_path;
      return false;
    }

    if (!reader.parse(json_path)) {
      LOG(ERROR) << "Failed to parse JSON config: " << json_path;
      return false;
    }

    auto model_args_loader = ModelRegistry::get_model_args_loader(model_type_);
    if (model_args_loader != nullptr) {
      model_args_loader(reader, &args_);
    } else {
      LOG(WARNING) << "No args loader for model type: " << model_type_;
    }

    return true;
  };

  if (has_safetensors) {
    if (!load_json_config("config.json")) {
      LOG(ERROR) << "Failed to load required config.json for safetensors model";
      return false;
    }
    // Read text_encoder_model for tokenizer fallback resolution.
    const std::string config_json_path = model_weights_path + "/config.json";
    JsonReader cfg_reader;
    if (cfg_reader.parse(config_json_path)) {
      if (auto v = cfg_reader.value<std::string>("text_encoder_model")) {
        text_encoder_model_ = v.value();
      }
    }
    load_image_preprocessor_args(model_weights_path);
  } else {
    std::filesystem::path tokenizer_config_path =
        model_dir / "tokenizer_config.json";
    if (std::filesystem::exists(tokenizer_config_path)) {
      load_image_preprocessor_args(model_weights_path);
      return true;
    }
    std::vector<std::filesystem::path> json_file_paths;
    for (const auto& entry :
         std::filesystem::directory_iterator(model_weights_path)) {
      if (entry.is_regular_file() &&
          entry.path().extension().string() == ".json") {
        json_file_paths.push_back(entry.path());
      }
    }

    if (json_file_paths.empty()) {
      LOG(ERROR) << "No JSON config files found in " << model_weights_path;
      return false;
    }

    bool loaded_any = false;
    for (const auto& json_file : json_file_paths) {
      if (!load_json_config(json_file.filename().string())) {
        LOG(ERROR) << "Failed to parse JSON file: " << json_file;
        continue;
      }
      loaded_any = true;
    }

    if (!loaded_any) {
      LOG(ERROR) << "No valid JSON config files found in "
                 << model_weights_path;
      return false;
    }
  }

  return true;
}

bool DiTFolderLoader::load_image_preprocessor_args(
    const std::string& model_weights_path) {
  JsonReader reader;
  const std::string path = model_weights_path + "/preprocessor_config.json";
  if (!std::filesystem::exists(path)) {
    return true;
  }
  if (!reader.parse(path)) {
    LOG(ERROR) << "Failed to parse preprocessor_config.json at " << path;
    return false;
  }
  LOG(INFO) << "Loaded image preprocessor config from " << path;
  args_.mm_image_min_pixels() =
      reader.value_or<int>("min_pixels", args_.mm_image_min_pixels());
  args_.mm_image_max_pixels() =
      reader.value_or<int>("max_pixels", args_.mm_image_max_pixels());
  args_.mm_image_patch_size() =
      reader.value_or<int>("patch_size", args_.mm_image_patch_size());
  args_.mm_image_temporal_patch_size() = reader.value_or<int>(
      "temporal_patch_size", args_.mm_image_temporal_patch_size());
  args_.mm_image_merge_size() =
      reader.value_or<int>("merge_size", args_.mm_image_merge_size());
  if (reader.contains("image_mean")) {
    args_.mm_image_normalize_mean() =
        reader.data()["image_mean"].get<std::vector<double>>();
  }
  if (reader.contains("image_std")) {
    args_.mm_image_normalize_std() =
        reader.data()["image_std"].get<std::vector<double>>();
  }
  return true;
}

namespace {

// Resolve a tokenizer directory from a HuggingFace repo id or local path,
// following the same lookup order as HuggingFace Hub / mainstream frameworks.
// Returns the resolved directory path, or empty string if not found.
std::string resolve_text_encoder_tokenizer_path(
    const std::string& text_encoder_model) {
  // 1. Local absolute path.
  if (std::filesystem::exists(text_encoder_model) &&
      std::filesystem::is_directory(text_encoder_model)) {
    return text_encoder_model;
  }

  // 2. HuggingFace cache: convert "org/name" -> "models--org--name".
  //    Cache root candidates (matches huggingface_hub default behaviour):
  //      $HF_HOME/hub  >  $HUGGINGFACE_HUB_CACHE  >  ~/.cache/huggingface/hub
  const std::string hf_cache_dir = [&]() -> std::string {
    if (const char* v = std::getenv("HF_HOME")) {
      return std::string(v) + "/hub";
    }
    if (const char* v = std::getenv("HUGGINGFACE_HUB_CACHE")) {
      return std::string(v);
    }
    if (const char* v = std::getenv("HOME")) {
      return std::string(v) + "/.cache/huggingface/hub";
    }
    return "";
  }();

  if (!hf_cache_dir.empty() && std::filesystem::exists(hf_cache_dir)) {
    // "google/umt5-base" -> "models--google--umt5-base"
    std::string cache_key = "models--";
    cache_key += absl::StrReplaceAll(text_encoder_model, {{"/", "--"}});

    const std::filesystem::path model_cache_dir =
        std::filesystem::path(hf_cache_dir) / cache_key;
    const std::filesystem::path snapshots_dir = model_cache_dir / "snapshots";
    if (std::filesystem::exists(snapshots_dir)) {
      // Prefer the snapshot pointed to by refs/main (or refs/master),
      // matching huggingface_hub's standard resolution order.
      for (const auto& ref : {"main", "master"}) {
        const std::filesystem::path ref_file = model_cache_dir / "refs" / ref;
        if (std::filesystem::exists(ref_file)) {
          std::ifstream ifs(ref_file.string());
          std::string commit_hash;
          if (std::getline(ifs, commit_hash) && !commit_hash.empty()) {
            // Trim trailing whitespace/newline
            while (!commit_hash.empty() &&
                   (commit_hash.back() == '\n' || commit_hash.back() == '\r' ||
                    commit_hash.back() == ' ')) {
              commit_hash.pop_back();
            }
            const std::filesystem::path snapshot = snapshots_dir / commit_hash;
            if (std::filesystem::exists(snapshot) &&
                std::filesystem::is_directory(snapshot)) {
              return snapshot.string();
            }
          }
        }
      }
      // Fallback: pick the only snapshot directory if there is exactly one.
      std::filesystem::path only;
      int32_t dir_count = 0;
      for (const auto& entry :
           std::filesystem::directory_iterator(snapshots_dir)) {
        if (entry.is_directory()) {
          only = entry.path();
          ++dir_count;
        }
      }
      if (dir_count == 1) {
        return only.string();
      }
    }
  }

  // 3. Not in local cache — download tokenizer files only via huggingface_hub.
  //    We use snapshot_download with allow_patterns to fetch only tokenizer
  //    files, avoiding downloading multi-GB model weights.
  LOG(INFO) << "Tokenizer for '" << text_encoder_model
            << "' not found in HuggingFace cache, downloading via "
               "huggingface_hub.snapshot_download ...";
  try {
    namespace py = pybind11;
    // The Python interpreter may not be initialized when this function is
    // called from a worker thread (e.g. DiTWorkerImpl::init_model).
    // Initialize it here if needed; it is safe to call Py_InitializeEx(0)
    // multiple times — subsequent calls are no-ops once already initialized.
    if (!Py_IsInitialized()) {
      // init_signal_handlers=0: don't override the process signal handlers
      // (SIGSEGV etc.) that glog/folly have already registered.
      Py_InitializeEx(0);
    }
    py::gil_scoped_acquire gil;
    py::module_ hf_hub = py::module_::import("huggingface_hub");
    py::list allow_patterns;
    for (const auto& p : {"spiece.model",
                          "tokenizer.model",
                          "tokenizer.json",
                          "tokenizer_config.json",
                          "special_tokens_map.json"}) {
      allow_patterns.append(p);
    }
    py::object result = hf_hub.attr("snapshot_download")(
        text_encoder_model, py::arg("allow_patterns") = allow_patterns);
    return result.cast<std::string>();
  } catch (const std::exception& e) {
    LOG(WARNING) << "huggingface_hub.snapshot_download failed for '"
                 << text_encoder_model << "': " << e.what();
  }
  return "";
}
}  // namespace

bool DiTFolderLoader::load_tokenizer_args(
    const std::string& model_weights_path) {
  // tokenizer args from tokenizer_config.json
  std::string path = model_weights_path;
  std::string tokenizer_args_path = path + "/tokenizer_config.json";

  if (!std::filesystem::exists(tokenizer_args_path) &&
      !text_encoder_model_.empty()) {
    LOG(WARNING)
        << "Tokenizer fallback to text_encoder_model '" << text_encoder_model_
        << "' specified in config.json. "
        << "Please download the tokenizer to the model directory or ensure "
        << "the HuggingFace cache is populated (e.g. run the official "
        << "inference script once to cache '" << text_encoder_model_ << "').";

    path = resolve_text_encoder_tokenizer_path(text_encoder_model_);
    if (path.empty()) {
      LOG(ERROR) << "Failed to resolve text encoder model tokenizer path for "
                 << text_encoder_model_;
      return false;
    }
  }

  JsonReader tokenizer_reader;
  tokenizer_args_path = path + "/tokenizer_config.json";

  if (!tokenizer_reader.parse(tokenizer_args_path)) {
    return true;
  }

  bool prefer_sentencepiece = false;
  if (auto v = tokenizer_reader.value<std::string>("tokenizer_class")) {
    const std::string& cls = v.value();
    if (cls == "T5Tokenizer" || cls == "T5TokenizerFast" ||
        cls.find("T5") != std::string::npos) {
      prefer_sentencepiece = true;
    }
  }

  // Check tokenizer.json; but prefer SentencePiece when tokenizer_class is
  // T5Tokenizer (e.g. UMT5), because the fast tokenizer produces different
  // token IDs from the SentencePiece tokenizer used during training.
  const std::string tokenizer_json_path = path + "/tokenizer.json";
  if (std::filesystem::exists(tokenizer_json_path) && !prefer_sentencepiece) {
    tokenizer_args_.tokenizer_type() = "fast";
    tokenizer_args_.vocab_file() = tokenizer_json_path;
  } else if (std::filesystem::exists(path + "/spiece.model")) {
    tokenizer_args_.tokenizer_type() = "sentencepiece";
    tokenizer_args_.vocab_file() = path + "/spiece.model";
  } else if (std::filesystem::exists(path + "/tokenizer.model")) {
    tokenizer_args_.tokenizer_type() = "sentencepiece";
    tokenizer_args_.vocab_file() = path + "/tokenizer.model";
  } else {
    LOG(ERROR) << "No tokenizer files found in directory: " << path;
    return false;
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

  return true;
}

bool DiTFolderLoader::load_quant_args(const std::string& model_weights_path) {
  // Search for a quant description JSON file by fuzzy name match
  // (e.g. "quant_model_description.json", "quant_desc.json", etc.).
  std::string quant_desc_file_path;
  for (const auto& entry :
       std::filesystem::directory_iterator(model_weights_path)) {
    if (!entry.is_regular_file()) {
      continue;
    }
    const auto& path = entry.path();
    if (path.extension() == ".json") {
      std::string stem = path.stem().string();
      boost::algorithm::to_lower(stem);
      if (stem.find("quant") != std::string::npos) {
        quant_desc_file_path = path.string();
        break;
      }
    }
  }
  if (quant_desc_file_path.empty()) {
    return true;  // No quant config — not an error.
  }

  JsonReader quant_desc_reader;
  if (!quant_desc_reader.parse(quant_desc_file_path)) {
    LOG(WARNING) << "Failed to parse quant description file: "
                 << quant_desc_file_path;
    return true;
  }

  const auto quant_desc_data = quant_desc_reader.data();
  if (!quant_desc_data.is_object()) {
    LOG(WARNING) << "Quant description file is not a JSON object: "
                 << quant_desc_file_path;
    return true;
  }

  std::unordered_map<std::string, std::string> quant_descs;
  bool desc_has_w8a8 = false;
  bool desc_has_w4a8 = false;

  for (auto it = quant_desc_data.begin(); it != quant_desc_data.end(); ++it) {
    if (!it.value().is_string()) {
      continue;
    }
    // Keep only tensor-like keys (contain '.') — skip metadata fields.
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
    quant_descs.emplace(it.key(), quant_type);
    // Map "weight_packed" keys to "weight" for compatibility.
    const std::string packed_weight = "weight_packed";
    std::string mapped_key = it.key();
    if (const auto pos = mapped_key.find(packed_weight);
        pos != std::string::npos) {
      mapped_key.replace(pos, packed_weight.size(), "weight");
      quant_descs.emplace(std::move(mapped_key), quant_type);
    }
  }

  quant_args_.quant_descs() = std::move(quant_descs);
  if (desc_has_w4a8) {
    quant_args_.quant_method() = kQuantMethodAscendInt4;
  } else if (desc_has_w8a8) {
    quant_args_.quant_method() = kQuantMethodAscendInt8;
  }

  // Read model-level quant metadata.
  if (auto v = quant_desc_reader.value<std::string>("model_quant_type")) {
    auto quantize_type = v.value();
    boost::algorithm::to_lower(quantize_type);
    quant_args_.quantize_type() = quantize_type;
  }
  if (auto v = quant_desc_reader.value<int64_t>("group_size")) {
    quant_args_.group_size() = v.value();
  }
  if (auto v = quant_desc_reader.value<std::string>("version")) {
    quant_args_.quant_version() = v.value();
  }

  LOG(INFO) << "Loaded quant_model_description from " << quant_desc_file_path
            << ", quant_desc_count=" << quant_args_.quant_descs().size()
            << ", quant_method="
            << (quant_args_.quant_method().empty()
                    ? "<empty>"
                    : quant_args_.quant_method());
  return true;
}

DiTModelLoader::DiTModelLoader(const std::string& model_root_path)
    : model_root_path_(model_root_path) {
  if (!std::filesystem::exists(model_root_path_)) {
    LOG(FATAL) << "Model root path does not exist: " << model_root_path_;
  }

  std::filesystem::path root_path(model_root_path_);
  std::filesystem::path index_file = root_path / "model_index.json";
  const std::string model_index_file = index_file.string();
  if (!std::filesystem::exists(model_index_file)) {
    // Flat layout: no model_index.json. The entire model root is a single
    // component (e.g. AudioDiT ships one model.safetensors at the root).
    // Read model_type from config.json and register the root as "model".
    std::filesystem::path config_json_path = root_path / "config.json";
    if (!std::filesystem::exists(config_json_path)) {
      if (auto layout = util::discover_dit_model_layout(root_path)) {
        for (const auto& component : layout->components) {
          name_to_loader_[component.name] =
              std::make_unique<DiTFolderLoader>(component.path.string(),
                                                component.name,
                                                component.component_type);
          LOG(INFO) << "DiTModelLoader: auto-discovered component '"
                    << component.name << "' with model_type='"
                    << component.component_type << "'";
        }
        set_model_type(layout->pipeline_type);
        LOG(INFO) << "DiTModelLoader: matched registered model type '"
                  << layout->pipeline_type << "' from component layout";
        return;
      }
      LOG(FATAL) << "DiTModelLoader: neither model_index.json nor config.json "
                    "found, and no component subdirectories discovered in: "
                 << model_root_path_;
    }
    JsonReader cfg_reader;
    if (!cfg_reader.parse(config_json_path.string())) {
      LOG(FATAL) << "DiTModelLoader: failed to parse config.json in: "
                 << model_root_path_;
    }
    auto model_type_opt = cfg_reader.value<std::string>("model_type");
    const std::string model_type =
        model_type_opt.has_value() ? model_type_opt.value() : "";
    set_model_type(model_type);
    name_to_loader_["model"] = std::make_unique<DiTFolderLoader>(
        model_root_path_, "model", model_type);
    return;
  }

  JsonReader model_index_reader;
  if (!model_index_reader.parse(model_index_file)) {
    LOG(FATAL) << "Failed to parse model index file: " << model_index_file;
  }

  const nlohmann::json root_json = model_index_reader.data();
  if (!root_json.is_object()) {
    LOG(FATAL) << "DiTModelLoader: model_index.json root is not an object!";
  }

  if (root_json.contains("_class_name")) {
    set_model_type(root_json["_class_name"]);
  } else {
    LOG(WARNING)
        << "model_index.json doesn't contains the _class_name key, xllm may "
        << "not obtain model type for dit model";
  }
  // parse model_index.json & initialize model_loader
  for (const auto& [json_key, json_value] : root_json.items()) {
    if (!json_value.is_array() || json_value.size() != 2) {
      continue;
    }

    if (json_value[1].is_null()) {
      LOG(INFO) << "Skipping null component: " << json_key;
      continue;
    }

    const std::string model_type = json_value[1].get<std::string>();
    const std::string component_name = json_key;

    std::filesystem::path component_folder_path =
        std::filesystem::path(model_root_path_) / component_name;
    const std::string component_folder = component_folder_path.string();
    if (!std::filesystem::exists(component_folder)) {
      LOG(FATAL) << "DiTModelLoader: Component folder not found! "
                 << "ComponentName=" << component_name
                 << ", Folder=" << component_folder;
      continue;
    }
    if (!std::filesystem::is_directory(component_folder)) {
      LOG(FATAL) << "DiTModelLoader: Component path is not a directory! "
                 << "ComponentName=" << component_name
                 << ", Path=" << component_folder;
      continue;
    }

    // create model loader for each Folder
    std::unique_ptr<DiTFolderLoader> loader = std::make_unique<DiTFolderLoader>(
        component_folder, component_name, model_type);
    if (!loader) {
      LOG(FATAL) << "Failed to create loader for: " << component_name;
      continue;
    }

    name_to_loader_[component_name] = std::move(loader);
  }
}

std::unique_ptr<DiTFolderLoader> DiTModelLoader::take_component_loader(
    const std::string& component) {
  auto itor = name_to_loader_.find(component);
  if (itor != name_to_loader_.end()) {
    std::unique_ptr<DiTFolderLoader> loader = std::move(itor->second);
    name_to_loader_.erase(itor);

    return loader;
  } else {
    LOG(FATAL) << "Loader not found, component: " << component;
    return nullptr;
  }
}

bool DiTModelLoader::has_component(const std::string& name) const {
  if (name_to_loader_.find(name) != name_to_loader_.end()) {
    return true;
  } else {
    return false;
  }
}

std::unordered_map<std::string, ModelArgs> DiTModelLoader::get_model_args()
    const {
  std::unordered_map<std::string, ModelArgs> map;
  for (const auto& pair : name_to_loader_) {
    map.insert({pair.first, pair.second->model_args()});
  }

  return map;
}

std::unordered_map<std::string, QuantArgs> DiTModelLoader::get_quant_args()
    const {
  std::unordered_map<std::string, QuantArgs> map;
  for (const auto& pair : name_to_loader_) {
    map.insert({pair.first, pair.second->quant_args()});
  }

  return map;
}

std::string DiTModelLoader::get_torch_dtype() const {
  std::string dtype;
  for (const auto& pair : name_to_loader_) {
    const auto& args = pair.second->model_args();

    const auto& type = args.dtype();
    if (dtype.empty() && !type.empty()) {
      dtype = type;
    } else if (!dtype.empty() && !type.empty() && dtype != type) {
      LOG(WARNING) << " dtype is not equal, dtype=" << dtype
                   << " type:" << type;
    }
  }

  return dtype;
}

}  // namespace xllm

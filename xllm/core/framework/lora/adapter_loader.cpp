/* Copyright 2025 The xLLM Authors. All Rights Reserved.

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

#include "adapter_loader.h"

#include <glog/logging.h>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <string>

#include "framework/state_dict/state_dict.h"

namespace xllm {

namespace {
// Fixed prefix HuggingFace PEFT prepends to every weight in
// adapter_model.safetensors. We strip it and then further rewrite the
// remainder into an xllm-friendly key.
constexpr const char* kHfPrefix = "base_model.model.";
constexpr const char* kLoraATag = ".lora_A.weight";
constexpr const char* kLoraBTag = ".lora_B.weight";

bool contains(const std::vector<std::string>& v, const std::string& item) {
  return std::find(v.begin(), v.end(), item) != v.end();
}
}  // namespace

std::optional<LoRAAdapter> LoRAAdapterLoader::load(const LoRARequest& req) {
  namespace fs = std::filesystem;
  const fs::path root = req.lora_path;
  const fs::path config_path = root / "adapter_config.json";
  const fs::path weights_path = root / "adapter_model.safetensors";

  if (!fs::exists(config_path)) {
    LOG(ERROR) << "[LoRAAdapterLoader] '" << req.lora_name
               << "': missing adapter_config.json at " << config_path;
    return std::nullopt;
  }
  if (!fs::exists(weights_path)) {
    LOG(ERROR) << "[LoRAAdapterLoader] '" << req.lora_name
               << "': missing adapter_model.safetensors at " << weights_path;
    return std::nullopt;
  }

  LoRAAdapter adapter;
  adapter.request = req;

  if (!parse_config(config_path.string(), req.base_model_name, &adapter)) {
    return std::nullopt;
  }
  if (!load_weights(weights_path.string(), &adapter)) {
    return std::nullopt;
  }

  LOG(INFO) << "[LoRAAdapterLoader] loaded '" << req.lora_name
            << "' r=" << adapter.r << " scaling=" << adapter.scaling
            << " target_modules=" << adapter.target_modules.size()
            << " tensors=" << adapter.tensors.size();
  return adapter;
}

bool LoRAAdapterLoader::parse_config(
    const std::string& path,
    const std::string& expected_base_model_name,
    LoRAAdapter* out) const {
  std::ifstream f(path);
  if (!f) {
    LOG(ERROR) << "[LoRAAdapterLoader] cannot open " << path;
    return false;
  }
  nlohmann::json j;
  try {
    f >> j;
  } catch (const std::exception& e) {
    LOG(ERROR) << "[LoRAAdapterLoader] " << path
               << ": parse error: " << e.what();
    return false;
  }

  // Reject non-LoRA PEFT variants explicitly. DoRA and IA3 need extra
  // machinery we don't have yet.
  if (j.value("peft_type", std::string("LORA")) != "LORA") {
    LOG(ERROR) << "[LoRAAdapterLoader] " << path << ": peft_type != LORA (got '"
               << j.value("peft_type", "?") << "'), refusing";
    return false;
  }
  if (j.value("bias", std::string("none")) != "none") {
    LOG(ERROR) << "[LoRAAdapterLoader] " << path << ": bias != none, refusing";
    return false;
  }
  if (j.value("use_dora", false)) {
    LOG(ERROR) << "[LoRAAdapterLoader] " << path << ": use_dora=true, refusing";
    return false;
  }

  const int32_t r = j.value("r", 0);
  const int32_t lora_alpha = j.value("lora_alpha", 0);
  if (r <= 0) {
    LOG(ERROR) << "[LoRAAdapterLoader] " << path << ": r=" << r
               << " must be > 0";
    return false;
  }
  if (r > config_.max_lora_rank) {
    LOG(ERROR) << "[LoRAAdapterLoader] " << path << ": r=" << r
               << " exceeds max_lora_rank=" << config_.max_lora_rank;
    return false;
  }
  out->r = r;
  out->scaling = lora_alpha > 0 ? static_cast<float>(lora_alpha) / r : 1.0f;

  // target_modules whitelist check.
  if (j.contains("target_modules")) {
    for (const auto& m : j["target_modules"]) {
      const std::string mod = m.get<std::string>();
      if (!contains(config_.lora_target_modules, mod)) {
        LOG(ERROR) << "[LoRAAdapterLoader] " << path << ": target_modules "
                   << "contains '" << mod
                   << "' which is not in --lora-target-modules whitelist";
        return false;
      }
      out->target_modules.push_back(mod);
    }
  }

  // Optional base-model check: only enforced if the caller passed one.
  if (!expected_base_model_name.empty() &&
      j.contains("base_model_name_or_path")) {
    const std::string got = j["base_model_name_or_path"].get<std::string>();
    if (got != expected_base_model_name) {
      LOG(ERROR) << "[LoRAAdapterLoader] " << path
                 << ": base_model_name_or_path='" << got
                 << "' does not match expected='" << expected_base_model_name
                 << "'";
      return false;
    }
  }

  return true;
}

std::optional<std::string> LoRAAdapterLoader::canonicalize_weight_name(
    const std::string& raw) const {
  // Strip HuggingFace fixed prefix.
  std::string s;
  if (raw.rfind(kHfPrefix, 0) == 0) {
    s = raw.substr(std::string(kHfPrefix).size());
  } else {
    // Some adapters skip the prefix. Tolerate it silently.
    s = raw;
  }

  // Look for the .lora_A.weight / .lora_B.weight suffix.
  const auto a_pos = s.rfind(kLoraATag);
  const auto b_pos = s.rfind(kLoraBTag);
  const bool is_a = a_pos != std::string::npos &&
                    a_pos + std::string(kLoraATag).size() == s.size();
  const bool is_b = b_pos != std::string::npos &&
                    b_pos + std::string(kLoraBTag).size() == s.size();
  if (!is_a && !is_b) {
    return std::nullopt;  // Not a LoRA weight (could be embedding LoRA which
                          // we skip in v1).
  }

  // Canonical form: "<subkey>#A" or "<subkey>#B" where subkey is the
  // dot-path of the module (e.g. "layers.5.self_attn.q_proj"). Using '#'
  // as the separator keeps subkey addressable via ordinary regex splits.
  const auto pos = is_a ? a_pos : b_pos;
  const std::string subkey = s.substr(0, pos);
  return subkey + (is_a ? "#A" : "#B");
}

bool LoRAAdapterLoader::load_weights(const std::string& safetensors_path,
                                     LoRAAdapter* out) const {
  auto sd = StateDictFromSafeTensor::load(safetensors_path);
  if (!sd) {
    LOG(ERROR) << "[LoRAAdapterLoader] safetensors load failed: "
               << safetensors_path;
    return false;
  }

  int32_t skipped = 0;
  int32_t taken = 0;
  for (const auto& [raw_name, tensor] : *sd) {
    auto canon = canonicalize_weight_name(raw_name);
    if (!canon) {
      ++skipped;
      LOG_EVERY_N(WARNING, 32)
          << "[LoRAAdapterLoader] skip unrecognised tensor '" << raw_name
          << "' in " << safetensors_path;
      continue;
    }
    // Sanity check: A is [r, ?], B is [?, r].
    const bool is_a =
        canon->size() >= 2 && canon->substr(canon->size() - 2) == "#A";
    const auto sizes = tensor.sizes();
    if (sizes.size() != 2) {
      LOG(ERROR) << "[LoRAAdapterLoader] '" << raw_name
                 << "' expected 2D tensor, got sizes.size()=" << sizes.size();
      return false;
    }
    if (is_a && sizes[0] != out->r) {
      LOG(ERROR) << "[LoRAAdapterLoader] '" << raw_name
                 << "' first dim=" << sizes[0] << " != r=" << out->r;
      return false;
    }
    if (!is_a && sizes[1] != out->r) {
      LOG(ERROR) << "[LoRAAdapterLoader] '" << raw_name
                 << "' second dim=" << sizes[1] << " != r=" << out->r;
      return false;
    }
    // MMAP_CLONE_APPLIED: `tensor` is at::from_blob(mmap_ptr,...) — the mmap
    // is owned by `sd` which is a local unique_ptr. Once this function
    // returns and sd destructs, the mmap is unmapped and `tensor` points
    // into freed memory. Materialize a heap-allocated copy right now.
    auto owned = torch::empty(
        tensor.sizes(),
        torch::TensorOptions().dtype(tensor.dtype()).device(torch::kCPU));
    owned.copy_(tensor);
    out->tensors.emplace(*canon, std::move(owned));
    ++taken;
  }

  if (taken == 0) {
    LOG(ERROR) << "[LoRAAdapterLoader] no LoRA weights found in "
               << safetensors_path;
    return false;
  }
  LOG(INFO) << "[LoRAAdapterLoader] " << safetensors_path << " taken=" << taken
            << " skipped=" << skipped;
  return true;
}

}  // namespace xllm

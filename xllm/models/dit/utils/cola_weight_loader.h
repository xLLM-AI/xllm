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

#include <cctype>
#include <string>
#include <unordered_map>

#include "core/framework/dit_model_loader.h"

namespace xllm {

// Translate a checkpoint key to the C++ module key. libtorch register_module
// forbids dots in names, so we register with underscores (e.g. "layers_0")
// while checkpoints use dots (e.g. "layers.0").
inline std::string cola_checkpoint_key_to_cpp_key(const std::string& key) {
  std::string result;
  result.reserve(key.size());
  for (size_t i = 0; i < key.size(); ++i) {
    if (key[i] == '.' && i + 1 < key.size() && std::isdigit(key[i + 1])) {
      result += '_';
    } else {
      result += key[i];
    }
  }
  return result;
}

// Keys present in the Python checkpoint but intentionally not registered in the
// C++ Cola modules (computed at runtime instead of loaded from disk).
inline bool is_cola_ignored_checkpoint_key(const std::string& key) {
  static const std::string kRopeFreqsSuffix = ".rope.rope.freqs";
  return key.size() >= kRopeFreqsSuffix.size() &&
         key.compare(key.size() - kRopeFreqsSuffix.size(),
                     kRopeFreqsSuffix.size(),
                     kRopeFreqsSuffix) == 0;
}

// Load Cola-DLM weights from safetensors into a torch::nn::Module.
// If key_prefix is non-empty, only keys starting with that prefix are loaded,
// with the prefix stripped before matching (e.g. "encoder.", "decoder.").
inline void load_cola_module_from_state_dicts(
    DiTFolderLoader& folder_loader,
    torch::nn::Module* module,
    const std::string& key_prefix = "") {
  auto params = module->named_parameters(/*recurse=*/true);
  auto buffers = module->named_buffers(/*recurse=*/true);

  std::unordered_map<std::string, torch::Tensor*> param_map;
  for (auto& kv : params) {
    param_map[cola_checkpoint_key_to_cpp_key(kv.key())] = &kv.value();
  }
  for (auto& kv : buffers) {
    param_map[cola_checkpoint_key_to_cpp_key(kv.key())] = &kv.value();
  }

  auto strip_prefix_fn = [&](const std::string& raw) -> std::string {
    if (key_prefix.empty()) {
      return raw;
    }
    if (raw.size() < key_prefix.size() ||
        raw.substr(0, key_prefix.size()) != key_prefix) {
      return "";
    }
    return raw.substr(key_prefix.size());
  };

  std::unordered_map<std::string, torch::Tensor> wn_g;
  std::unordered_map<std::string, torch::Tensor> wn_v;
  static const std::string kSuffixG = ".weight_g";
  static const std::string kSuffixV = ".weight_v";
  for (const auto& state_dict_ptr : folder_loader.get_state_dicts()) {
    for (const auto& kv : *state_dict_ptr) {
      std::string key = strip_prefix_fn(kv.first);
      if (key.empty()) {
        continue;
      }
      if (key.size() > kSuffixG.size() &&
          key.substr(key.size() - kSuffixG.size()) == kSuffixG) {
        wn_g[key.substr(0, key.size() - kSuffixG.size())] = kv.second;
      } else if (key.size() > kSuffixV.size() &&
                 key.substr(key.size() - kSuffixV.size()) == kSuffixV) {
        wn_v[key.substr(0, key.size() - kSuffixV.size())] = kv.second;
      }
    }
  }

  for (const auto& state_dict_ptr : folder_loader.get_state_dicts()) {
    for (const auto& kv : *state_dict_ptr) {
      const std::string& raw_key = kv.first;
      const torch::Tensor& src_tensor = kv.second;

      std::string key = raw_key;
      if (!key_prefix.empty()) {
        if (raw_key.size() < key_prefix.size() ||
            raw_key.substr(0, key_prefix.size()) != key_prefix) {
          continue;
        }
        key = raw_key.substr(key_prefix.size());
      }

      if ((key.size() > kSuffixG.size() &&
           key.substr(key.size() - kSuffixG.size()) == kSuffixG) ||
          (key.size() > kSuffixV.size() &&
           key.substr(key.size() - kSuffixV.size()) == kSuffixV)) {
        continue;
      }

      if (is_cola_ignored_checkpoint_key(key)) {
        continue;
      }

      const std::string cpp_key = cola_checkpoint_key_to_cpp_key(key);
      auto it = param_map.find(cpp_key);
      if (it == param_map.end()) {
        if (key_prefix.empty()) {
          LOG(WARNING) << "[Cola-DLM load] Unknown key in checkpoint: " << key;
        }
        continue;
      }
      torch::Tensor& dst = *(it->second);
      if (!dst.defined()) {
        LOG(WARNING) << "[Cola-DLM load] Skipping key with undefined dst: "
                     << key;
        continue;
      }
      torch::NoGradGuard no_grad;
      dst.copy_(src_tensor.to(dst.dtype()).to(dst.device()));
    }
  }

  for (const auto& gkv : wn_g) {
    const std::string& base = gkv.first;
    auto vit = wn_v.find(base);
    if (vit == wn_v.end()) {
      continue;
    }

    const std::string weight_cpp_key =
        cola_checkpoint_key_to_cpp_key(base + ".weight");
    auto it = param_map.find(weight_cpp_key);
    if (it == param_map.end()) {
      LOG(WARNING) << "[Cola-DLM load] weight_norm base not found in module: "
                   << weight_cpp_key;
      continue;
    }

    torch::Tensor g = gkv.second.to(torch::kFloat32);
    torch::Tensor v = vit->second.to(torch::kFloat32);
    int64_t c_out = v.size(0);
    torch::Tensor v_norm = v.view({c_out, -1}).norm(2, 1, /*keepdim=*/true);
    std::vector<int64_t> norm_shape(v.dim(), 1);
    norm_shape[0] = c_out;
    v_norm = v_norm.view(norm_shape);
    torch::Tensor w = g * v / (v_norm + 1e-12f);

    torch::Tensor& dst = *(it->second);
    torch::NoGradGuard no_grad;
    dst.copy_(w.to(dst.dtype()).to(dst.device()));
  }
}

}  // namespace xllm

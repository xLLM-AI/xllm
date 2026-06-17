/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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

#include <string>
#include <unordered_map>

#include "core/framework/config/parallel_config.h"

namespace xllm::dit {

class SequenceParallelPadManager {
 public:
  static SequenceParallelPadManager& getInstance() {
    static SequenceParallelPadManager instance;
    return instance;
  }

  torch::Tensor pad_tensor(const torch::Tensor& ref_tensor,
                           const string& tensor_name,
                           int64_t dim = -1,
                           bool right_pad = true) {
    auto pad_dim = dim;
    if (pad_dim == -1) {
      pad_dim = ref_tensor.dim() - 1;
    }

    if (ref_tensor.defined()) {
      if (ref_tensor.size(dim) %
              ::xllm::ParallelConfig::get_instance().sp_size() !=
          0) {
        int64_t pad_len = ::xllm::ParallelConfig::get_instance().sp_size() -
                          ref_tensor.size(dim) %
                              ::xllm::ParallelConfig::get_instance().sp_size();
        set(tensor_name, pad_len);

        std::vector<int64_t> pad_shape(ref_tensor.dim() * 2);
        int64_t pad_shift = right_pad ? 1 : 0;
        pad_shape[2 * (ref_tensor.dim() - pad_dim - 1) + pad_shift] = pad_len;
        auto pad_tensor = torch::pad(ref_tensor, pad_shape, "constant", 0);
        return pad_tensor;
      }
      set(tensor_name, 0);
    }

    return ref_tensor;
  }

  void unpad_tensor(torch::Tensor& ref_tensor,
                    const string& tensor_name,
                    int64_t dim = -1,
                    bool right_pad = true) {
    if (ref_tensor.defined()) {
      auto pad = get(tensor_name);
      ref_tensor = ref_tensor.narrow(dim, 0, ref_tensor.size(dim) - pad);
    }
  }

  void set(const std::string& key, int64_t length) {
    pad_lengths_[key] = length;
  }

  int64_t get(const std::string& key) const {
    auto it = pad_lengths_.find(key);
    return it != pad_lengths_.end() ? it->second : 0;
  }

  bool has(const std::string& key) const {
    return pad_lengths_.find(key) != pad_lengths_.end();
  }

  void remove(const std::string& key) { pad_lengths_.erase(key); }

  void clear() { pad_lengths_.clear(); }

 private:
  SequenceParallelPadManager() = default;
  ~SequenceParallelPadManager() = default;
  SequenceParallelPadManager(const SequenceParallelPadManager&) = delete;
  SequenceParallelPadManager& operator=(const SequenceParallelPadManager&) =
      delete;

  std::unordered_map<std::string, int64_t> pad_lengths_;
};

}  // namespace xllm::dit

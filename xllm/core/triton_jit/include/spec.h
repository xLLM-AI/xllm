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
#include <vector>

#include "arg.h"
#include "types.h"

namespace xllm::triton_jit {

enum class Kind { PTR, VAL, CONST };

struct ArgSpec {
  Kind kind = Kind::VAL;
  std::string type;  // base Triton type name, e.g. "fp32", "i32" (no "*")
  SpecHint hint = SpecHint::NONE;
  bool specialize = true;
  std::string const_val;
};

using SpecList = std::vector<ArgSpec>;

template <typename T>
inline std::string format_const(T v) {
  using U = std::decay_t<T>;
  if constexpr (std::is_same_v<U, bool>) {
    return v ? "1" : "0";
  } else if constexpr (std::is_floating_point_v<U>) {
    return std::to_string(v);  // e.g. "1.000000" -> float() -> 1.0
  } else {
    return std::to_string(v);  // integers
  }
}

// Deterministic cache key for a (signature, launch cfg, device) triple.
std::string serialize_key(const SpecList& specs,
                          const LaunchCfg& cfg,
                          int32_t device);

// Key for a (signature, device) triple, WITHOUT the launch cfg. Used by the
// autotuner to map a winner LaunchCfg to a specialization across cfg trials.
std::string serialize_key_base(const SpecList& specs, int32_t device);

}  // namespace xllm::triton_jit

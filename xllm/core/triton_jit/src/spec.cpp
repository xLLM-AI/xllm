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

#include "spec.h"

#include <string>

namespace xllm::triton_jit {

namespace {

const char* kind_tag(Kind k) {
  switch (k) {
    case Kind::PTR: {
      return "P";
    }
    case Kind::VAL: {
      return "V";
    }
    case Kind::CONST: {
      return "C";
    }
  }
  return "?";
}

const char* hint_tag(SpecHint h) {
  switch (h) {
    case SpecHint::DIV16: {
      return "16";
    }
    case SpecHint::EQ1: {
      return "1";
    }
    default: {
      return "0";
    }
  }
}
}  // namespace

// Deterministic cache key: each field is type-tagged so there is no parsing
// ambiguity. Also forwarded to Python so triton.compile sees the same
// signature.
std::string serialize_key(const SpecList& specs,
                          const LaunchCfg& cfg,
                          int32_t device) {
  std::string s;
  s.reserve(128);
  for (const auto& a : specs) {
    if (!s.empty()) {
      s += ',';
    }
    s += kind_tag(a.kind);
    s += ':';
    if (a.kind == Kind::PTR) {
      s += '*';
    }
    s += a.type;
    s += ':';
    s += a.specialize ? 's' : 'n';
    s += ':';
    s += hint_tag(a.hint);
    if (a.kind == Kind::CONST) {
      s += ':';
      s += a.const_val;
    }
  }
  s += "|w";
  s += std::to_string(cfg.num_warps);
  s += "|s";
  s += std::to_string(cfg.num_stages);
  s += "|d";
  s += std::to_string(device);
  return s;
}

// Same as serialize_key but without the launch cfg, so the autotuner can key
// the winner LaunchCfg by (signature, device) alone.
std::string serialize_key_base(const SpecList& specs, int32_t device) {
  std::string s;
  s.reserve(128);
  for (const auto& a : specs) {
    if (!s.empty()) {
      s += ',';
    }
    s += kind_tag(a.kind);
    s += ':';
    if (a.kind == Kind::PTR) {
      s += '*';
    }
    s += a.type;
    s += ':';
    s += a.specialize ? 's' : 'n';
    s += ':';
    s += hint_tag(a.hint);
    if (a.kind == Kind::CONST) {
      s += ':';
      s += a.const_val;
    }
  }
  s += "|d";
  s += std::to_string(device);
  return s;
}

}  // namespace xllm::triton_jit

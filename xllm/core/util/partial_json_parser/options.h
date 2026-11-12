/* Copyright 2025-2026 The xLLM Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/xLLM-AI/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#pragma once

#include <cstdint>

namespace partial_json_parser {

enum TypeOptions {
  STR = 1 << 0,        // 1
  NUM = 1 << 1,        // 2
  ARR = 1 << 2,        // 4
  OBJ = 1 << 3,        // 8
  NULL_TYPE = 1 << 4,  // 16 (using NULL_TYPE to avoid conflict with NULL macro)
  BOOL = 1 << 5,       // 32
  NAN_TYPE = 1 << 6,   // 64 (using NAN_TYPE to avoid conflict with NAN macro)
  INFINITY_TYPE =
      1
      << 7,  // 128 (using INFINITY_TYPE to avoid conflict with INFINITY macro)
  NEG_INFINITY = 1 << 8,  // 256

  INF = INFINITY_TYPE | NEG_INFINITY,
  SPECIAL = NULL_TYPE | BOOL | INF | NAN_TYPE,
  ATOM = STR | NUM | SPECIAL,
  COLLECTION = ARR | OBJ,
  ALL = ATOM | COLLECTION
};

}  // namespace partial_json_parser

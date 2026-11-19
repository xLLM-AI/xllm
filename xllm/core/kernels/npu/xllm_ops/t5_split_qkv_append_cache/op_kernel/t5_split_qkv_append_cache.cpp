/* Copyright 2026 The xLLM Authors.

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

#include "op_kernel/t5_copy.h"
extern "C" __global__ __aicore__ void t5_split_qkv_append_cache(
    GM_ADDR qkv,
    GM_ADDR pk,
    GM_ADDR pv,
    GM_ADDR q,
    GM_ADDR k,
    GM_ADDR v,
    GM_ADDR workspace,
    GM_ADDR tiling) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  GET_TILING_DATA(t, tiling);
  AscendC::TPipe pipe;
  T5Copy op;
  op.init(qkv,
          pk,
          pv,
          q,
          k,
          v,
          t.batch,
          t.heads,
          t.length,
          t.dim,
          t.tile_elements,
          t.query_tiles,
          t.cache_tiles,
          t.total_tiles,
          &pipe);
  op.process();
}

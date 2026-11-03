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

#include "common/t5_plan.h"
#include "op_kernel/t5_copy.h"

extern "C" __global__ __aicore__ void t5_direct_kernel(GM_ADDR qkv,
                                                       GM_ADDR pk,
                                                       GM_ADDR pv,
                                                       GM_ADDR q,
                                                       GM_ADDR k,
                                                       GM_ADDR v,
                                                       uint32_t b,
                                                       uint32_t h,
                                                       uint32_t l,
                                                       uint32_t d,
                                                       uint32_t tile,
                                                       uint32_t qt,
                                                       uint32_t kt,
                                                       uint32_t total) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  AscendC::TPipe pipe;
  T5Copy op;
  op.init(qkv, pk, pv, q, k, v, b, h, l, d, tile, qt, kt, total, &pipe);
  op.process();
}

#ifndef ASCENDC_CPU_DEBUG
extern "C" void t5_launch_direct(void* stream,
                                 void* qkv,
                                 void* pk,
                                 void* pv,
                                 void* q,
                                 void* k,
                                 void* v,
                                 uint32_t b,
                                 uint32_t h,
                                 uint32_t l,
                                 uint32_t d) {
  const auto p = t5qkv::make_plan(b, h, l, d);
  t5_direct_kernel<<<p.blocks_, nullptr, stream>>>(
      reinterpret_cast<uint8_t*>(qkv),
      reinterpret_cast<uint8_t*>(pk),
      reinterpret_cast<uint8_t*>(pv),
      reinterpret_cast<uint8_t*>(q),
      reinterpret_cast<uint8_t*>(k),
      reinterpret_cast<uint8_t*>(v),
      b,
      h,
      l,
      d,
      p.tile_elements_,
      p.query_tiles_,
      p.cache_tiles_,
      p.total_tiles_);
}
#endif

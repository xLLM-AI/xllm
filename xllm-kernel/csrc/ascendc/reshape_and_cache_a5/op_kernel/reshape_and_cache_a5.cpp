/* Copyright 2026 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://gitcode.com/xLLM-AI/xllm_ops/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "kernel_operator.h"

namespace {
using namespace AscendC;

template <typename T>
class ReshapeAndCacheA5Kernel {
 public:
  __aicore__ inline void Init(GM_ADDR key,
                              GM_ADDR value,
                              GM_ADDR slot_mapping,
                              GM_ADDR key_cache_out,
                              GM_ADDR value_cache_out,
                              const ReshapeAndCacheA5TilingData* tiling,
                              TPipe* pipe) {
    tiling_ = *tiling;
    key_gm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(key));
    value_gm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(value));
    slot_mapping_gm_.SetGlobalBuffer(
        reinterpret_cast<__gm__ int32_t*>(slot_mapping));
    key_cache_gm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(key_cache_out));
    value_cache_gm_.SetGlobalBuffer(
        reinterpret_cast<__gm__ T*>(value_cache_out));
    pipe_ = pipe;
    pipe_->InitBuffer(copy_buffer_, tiling_.tile_elements * sizeof(T));
  }

  __aicore__ inline void Process() {
    const uint32_t core_id = GetBlockIdx();
    const uint32_t core_count = GetBlockNum();
    const TEventID mte2_to_mte3 =
        GetTPipePtr()->FetchEventID(HardEvent::MTE2_MTE3);
    const TEventID mte3_to_mte2 =
        GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2);
    const TEventID scalar_to_mte2 =
        GetTPipePtr()->FetchEventID(HardEvent::S_MTE2);

    for (uint32_t token = core_id; token < tiling_.num_tokens;
         token += core_count) {
      const int32_t slot = slot_mapping_gm_.GetValue(token);
      SetFlag<HardEvent::S_MTE2>(scalar_to_mte2);
      WaitFlag<HardEvent::S_MTE2>(scalar_to_mte2);
      if (slot < 0 || static_cast<uint32_t>(slot) >= tiling_.total_slots) {
        continue;
      }
      const uint64_t source_offset =
          static_cast<uint64_t>(token) * tiling_.row_elements;
      const uint64_t cache_offset =
          static_cast<uint64_t>(slot) * tiling_.row_elements;
      CopyRow(key_gm_,
              key_cache_gm_,
              source_offset,
              cache_offset,
              mte2_to_mte3,
              mte3_to_mte2);
      CopyRow(value_gm_,
              value_cache_gm_,
              source_offset,
              cache_offset,
              mte2_to_mte3,
              mte3_to_mte2);
    }
  }

 private:
  __aicore__ inline void CopyRow(const GlobalTensor<T>& source,
                                 GlobalTensor<T>& destination,
                                 uint64_t source_offset,
                                 uint64_t destination_offset,
                                 TEventID mte2_to_mte3,
                                 TEventID mte3_to_mte2) {
    LocalTensor<T> local = copy_buffer_.Get<T>();
    uint32_t copied = 0;
    while (copied < tiling_.row_elements) {
      const uint32_t elements =
          min(tiling_.tile_elements, tiling_.row_elements - copied);
      DataCopyExtParams copy_params = {
          1, static_cast<uint32_t>(elements * sizeof(T)), 0, 0, 0};
      DataCopyPadExtParams<T> pad_params = {false, 0, 0, 0};
      DataCopyPad(
          local, source[source_offset + copied], copy_params, pad_params);
      SetFlag<HardEvent::MTE2_MTE3>(mte2_to_mte3);
      WaitFlag<HardEvent::MTE2_MTE3>(mte2_to_mte3);
      DataCopyPad(destination[destination_offset + copied], local, copy_params);
      SetFlag<HardEvent::MTE3_MTE2>(mte3_to_mte2);
      WaitFlag<HardEvent::MTE3_MTE2>(mte3_to_mte2);
      copied += elements;
    }
  }

  ReshapeAndCacheA5TilingData tiling_;
  TPipe* pipe_ = nullptr;
  GlobalTensor<T> key_gm_;
  GlobalTensor<T> value_gm_;
  GlobalTensor<int32_t> slot_mapping_gm_;
  GlobalTensor<T> key_cache_gm_;
  GlobalTensor<T> value_cache_gm_;
  TBuf<TPosition::VECCALC> copy_buffer_;
};
}  // namespace

extern "C" __global__ __aicore__ void reshape_and_cache_a5(
    GM_ADDR key,
    GM_ADDR value,
    GM_ADDR key_cache,
    GM_ADDR value_cache,
    GM_ADDR slot_mapping,
    GM_ADDR key_cache_out,
    GM_ADDR value_cache_out,
    GM_ADDR workspace,
    GM_ADDR tiling) {
  GET_TILING_DATA(tiling_data, tiling);
  AscendC::TPipe pipe;
  ReshapeAndCacheA5Kernel<DTYPE_KEY> kernel;
  kernel.Init(key,
              value,
              slot_mapping,
              key_cache_out,
              value_cache_out,
              &tiling_data,
              &pipe);
  kernel.Process();
}

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

// Build xlite XModelAttnMeta from xllm ModelInputParams.

#pragma once

#include <xlite/xlite.h>

#include <algorithm>
#include <cstdint>
#include <vector>

#include "core/framework/model/model_input_params.h"
#include "core/layers/xlite/xlite_init_utils.h"

namespace xllm::xlite {

class XliteAttnMetaBuilder {
 public:
  static void Build(const ModelInputParams& params,
                    const torch::Tensor& positions,
                    uint32_t block_size,
                    XModelAttnMeta& m,
                    int64_t pad_count = 0) {
    // version=0: xlite recomputes position from cachedLens (framework's
    // positions tensor is off-by-one on decode).
    m.version = 0;
    m.lensCpu.clear();
    m.cachedLensCpu.clear();
    m.blockTablesCpu.clear();

    const auto& host = params.attention.host;
    int n = params.meta.num_sequences;
    uint32_t bs = block_size;

    // block_tables may be undefined in edge cases (DP empty shard).
    const bool has_real_seqs =
        n > 0 && host.block_tables.defined() && host.block_tables.dim() >= 2;
    if (has_real_seqs) {
      auto block_acc = host.block_tables.accessor<int32_t, 2>();
      for (int s = 0; s < n; ++s) {
        int32_t q_len = host.q_seq_lens[s];
        int32_t kv_len = host.kv_seq_lens[s];
        m.lensCpu.push_back(static_cast<uint32_t>(q_len));
        m.cachedLensCpu.push_back(
            static_cast<uint32_t>(std::max(0, kv_len - q_len)));  // clamp >= 0
        int32_t nblocks =
            (kv_len + static_cast<int32_t>(bs) - 1) / static_cast<int32_t>(bs);
        std::vector<uint32_t> row(nblocks);
        for (int32_t b = 0; b < nblocks; ++b) {
          row[b] = static_cast<uint32_t>(block_acc[s][b]);
        }
        m.blockTablesCpu.push_back(std::move(row));
      }
    }

    // DP padding: append dummy seq so sum(lens) aligns across DP groups.
    // Block 0 is the framework-reserved padding block (BlockManagerImpl /
    // XTensorBlockManagerImpl), never owned by real sequences.
    if (pad_count > 0) {
      m.lensCpu.push_back(static_cast<uint32_t>(pad_count));
      m.cachedLensCpu.push_back(0);
      int32_t nblocks =
          (static_cast<int32_t>(pad_count) + static_cast<int32_t>(bs) - 1) /
          static_cast<int32_t>(bs);
      std::vector<uint32_t> row(nblocks, 0);
      m.blockTablesCpu.push_back(std::move(row));
    }
    InitXTensor(m.position, positions);
  }
};

}  // namespace xllm::xlite
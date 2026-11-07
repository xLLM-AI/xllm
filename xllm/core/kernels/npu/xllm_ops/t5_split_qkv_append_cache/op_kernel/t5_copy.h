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

#pragma once
#include "kernel_operator.h"

// All tensors are deliberately interpreted as uint16_t. No FP arithmetic/casts.
constexpr uint32_t kQueueDepth = 2;
// Host validation bounds every tensor to <2^28 elements; all linear indices and
// products below fit uint32. Avoid software 64-bit division in per-tile loops.
class T5Copy final {
 public:
  __aicore__ inline void init(GM_ADDR packed,
                              GM_ADDR past_key,
                              GM_ADDR past_value,
                              GM_ADDR query,
                              GM_ADDR key,
                              GM_ADDR value,
                              uint32_t b,
                              uint32_t h,
                              uint32_t l,
                              uint32_t d,
                              uint32_t tile,
                              uint32_t qt,
                              uint32_t kt,
                              uint32_t total,
                              AscendC::TPipe* pipe) {
    batch_ = b;
    heads_ = h;
    length_ = l;
    dim_ = d;
    tile_ = tile;
    query_tiles_ = qt;
    cache_tiles_ = kt;
    total_ = total;
    q_size_ = static_cast<uint32_t>(b) * h * d;
    cache_size_ = q_size_ * (l + 1);
    packed_.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t*>(packed),
                            3 * q_size_);
    // L=0 never dereferences these possibly null empty-storage pointers.
    pk_.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t*>(past_key),
                        q_size_ * l);
    pv_.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t*>(past_value),
                        q_size_ * l);
    q_.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t*>(query), q_size_);
    k_.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t*>(key), cache_size_);
    v_.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t*>(value), cache_size_);
    // Both views start from the raw pointer: the bypass view already carries
    // an L2 alias and must never be used to initialize the cached view.
    cached_key_.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t*>(key),
                                cache_size_);
    cached_key_tiles_ = 0;
    cached_key_begin_ = 0;
    // Ascend910_9382 has 192 MiB L2. Retain a bounded key suffix alongside
    // the inputs, leaving 16 MiB for unrelated traffic. Other outputs bypass
    // L2. Whole tiles use one policy, including unaligned-storage fallbacks.
    if (cache_size_ >= 16 * 1024 * 1024) {
      k_.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
      v_.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
      constexpr uint32_t kWorkingSetBytes = 176 * 1024 * 1024;
      const uint32_t input_bytes = (3 * q_size_ + 2 * q_size_ * length_) * 2;
      if (input_bytes < kWorkingSetBytes) {
        const uint32_t cached_elements =
            min(cache_size_, (kWorkingSetBytes - input_bytes) / 2);
        cached_key_tiles_ = cached_elements / tile_;
        // Preserve the original number of complete cached tiles. Any final
        // partial tile still bypasses L2, so the NORMAL byte budget is
        // unchanged.
        cached_key_begin_ = cache_size_ / tile_ - cached_key_tiles_;
      }
    }
    // For D and tile divisible by 16, every fragment boundary below is a
    // multiple of 16 uint16 elements. Check the actual NORMAL/BYPASS views
    // once, preserving the original per-fragment fallback for other inputs.
    aligned_fragments_ = false;
    if (cache_size_ >= 16 * 1024 * 1024 && (dim_ & 15) == 0 &&
        (tile_ & 15) == 0) {
      const uint64_t view_addresses =
          reinterpret_cast<uint64_t>(packed_.GetPhyAddr()) |
          reinterpret_cast<uint64_t>(pk_.GetPhyAddr()) |
          reinterpret_cast<uint64_t>(pv_.GetPhyAddr()) |
          reinterpret_cast<uint64_t>(q_.GetPhyAddr()) |
          reinterpret_cast<uint64_t>(k_.GetPhyAddr()) |
          reinterpret_cast<uint64_t>(v_.GetPhyAddr()) |
          reinterpret_cast<uint64_t>(cached_key_.GetPhyAddr());
      aligned_fragments_ = (view_addresses & 31) == 0;
    }
    pipe->InitBuffer(queue_, kQueueDepth, tile_ * sizeof(uint16_t));
  }

  __aicore__ inline void process() {
    for (uint32_t work = AscendC::GetBlockIdx(); work < total_;
         work += AscendC::GetBlockNum()) {
      if (work < query_tiles_) {
        query(work);
      } else if (work < query_tiles_ + cache_tiles_) {
        cache(work - query_tiles_, false);
      } else {
        cache(work - query_tiles_ - cache_tiles_, true);
      }
    }
  }

 private:
  __aicore__ inline uint32_t min(uint32_t a, uint32_t b) {
    return a < b ? a : b;
  }

  __aicore__ inline void copy(AscendC::GlobalTensor<uint16_t>& src,
                              uint32_t src_offset,
                              AscendC::GlobalTensor<uint16_t>& dst,
                              uint32_t dst_offset,
                              uint32_t n) {
    if (!n) {
      return;
    }
    auto local = queue_.AllocTensor<uint16_t>();
    // The plain DMA path requires a complete 32-byte block count. Inspect the
    // actual GM addresses too: contiguous storage-offset views may be
    // unaligned.
    const bool aligned =
        aligned_fragments_ ||
        (((n & 15) == 0) &&
         ((reinterpret_cast<uint64_t>(src.GetPhyAddr() + src_offset) & 31) ==
          0) &&
         ((reinterpret_cast<uint64_t>(dst.GetPhyAddr() + dst_offset) & 31) ==
          0));
    AscendC::DataCopyExtParams params{
        1, n * static_cast<uint32_t>(sizeof(uint16_t)), 0, 0, 0};
    AscendC::DataCopyPadExtParams<uint16_t> padding{false, 0, 0, 0};
    if (aligned) {
      AscendC::DataCopy(local, src[src_offset], n);
    } else {
      AscendC::DataCopyPad(local, src[src_offset], params, padding);
    }
    queue_.EnQue(local);
    local = queue_.DeQue<uint16_t>();
    if (aligned) {
      AscendC::DataCopy(dst[dst_offset], local, n);
    } else {
      AscendC::DataCopyPad(dst[dst_offset], local, params);
    }
    queue_.FreeTensor(local);
  }

  __aicore__ inline void query(uint32_t tile_id) {
    uint32_t pos = static_cast<uint32_t>(tile_id) * tile_,
             end = min(pos + tile_, q_size_);
    const uint32_t hd = static_cast<uint32_t>(heads_) * dim_;
    while (pos < end) {
      const uint32_t b = pos / hd, within = pos % hd;
      const uint32_t n = min(end - pos, hd - within);
      copy(packed_, b * 3 * hd + within, q_, pos, n);
      pos += n;
    }
  }

  __aicore__ inline void cache(uint32_t tile_id, bool is_value) {
    auto& past = is_value ? pv_ : pk_;
    const bool cached = tile_id >= cached_key_begin_ &&
                        tile_id - cached_key_begin_ < cached_key_tiles_;
    auto& out = is_value ? v_ : (cached ? cached_key_ : k_);
    const uint32_t hd = static_cast<uint32_t>(heads_) * dim_;
    const uint32_t old_row = static_cast<uint32_t>(length_) * dim_,
                   row = old_row + dim_;
    uint32_t pos = static_cast<uint32_t>(tile_id) * tile_,
             end = min(pos + tile_, cache_size_);
    if (length_ == 0) {
      // Empty cache is a whole HD segment per batch, not H separate tokens.
      while (pos < end) {
        const uint32_t b = pos / hd, within = pos % hd;
        const uint32_t n = min(end - pos, hd - within);
        copy(packed_,
             b * 3 * hd + (is_value ? 2 : 1) * hd + within,
             out,
             pos,
             n);
        pos += n;
      }
      return;
    }
    // No tile writes outside its own aligned output interval. A tile can cross
    // rows, but every fragment of the same 32B block belongs to one core.
    while (pos < end) {
      const uint32_t head = pos / row, within = pos % row;
      uint32_t n;
      if (within < old_row) {
        n = min(end - pos, old_row - within);
        copy(past, head * old_row + within, out, pos, n);
      } else {
        n = min(end - pos, row - within);
        const uint32_t b = head / heads_, h = head % heads_;
        const uint32_t src =
            b * 3 * hd + (is_value ? 2 : 1) * hd + h * dim_ + within - old_row;
        copy(packed_, src, out, pos, n);
      }
      pos += n;
    }
  }

  uint32_t batch_, heads_, length_, dim_, tile_, query_tiles_, cache_tiles_,
      total_;
  uint32_t q_size_, cache_size_, cached_key_tiles_, cached_key_begin_;
  bool aligned_fragments_;
  AscendC::GlobalTensor<uint16_t> packed_, pk_, pv_, q_, k_, v_, cached_key_;
  AscendC::TQueBind<AscendC::QuePosition::VECIN,
                    AscendC::QuePosition::VECOUT,
                    kQueueDepth>
      queue_;
};

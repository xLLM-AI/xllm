
#include "prefix_cache_with_upload.h"

#include <absl/time/clock.h>
#include <absl/time/time.h>
#include <string.h>

#include <iostream>
#include <thread>

#include "common/metrics.h"
#include "util/hash_util.h"

namespace xllm {

PrefixCacheWithUpload::PrefixCacheWithUpload(uint32_t block_size,
                                             BlockHasherType hasher_type)
    : PrefixCache(block_size, hasher_type) {
  db_kvcache_events_.set_front_value(new KvCacheEvent());
  db_kvcache_events_.set_back_value(new KvCacheEvent());
}

PrefixCacheWithUpload::~PrefixCacheWithUpload() {
  auto back = db_kvcache_events_.get_back_value();
  if (back) {
    delete back;
  }

  auto front = db_kvcache_events_.get_back_value();
  if (front) {
    delete front;
  }
}

size_t PrefixCacheWithUpload::insert(const Slice<int32_t>& token_ids,
                                     std::vector<Block>& blocks,
                                     size_t existed_shared_blocks_num,
                                     const MMData& mm_data) {
  std::vector<XXH3Key> insert_keys;
  auto n_tokens = PrefixCache::insert(
      token_ids, blocks, existed_shared_blocks_num, mm_data, &insert_keys);
  save_event_async(true, insert_keys);
  return n_tokens;
}

size_t PrefixCacheWithUpload::insert(const std::vector<Block>& blocks) {
  Slice<Block> slice(blocks);
  return insert(slice);
}

size_t PrefixCacheWithUpload::insert(Slice<Block>& blocks) {
  std::vector<XXH3Key> insert_keys;
  auto n_tokens = PrefixCache::insert(blocks, &insert_keys);
  save_event_async(true, insert_keys);
  return n_tokens;
}

size_t PrefixCacheWithUpload::evict(size_t n_blocks) {
  std::vector<XXH3Key> evict_keys;
  auto evict_count = PrefixCache::evict(n_blocks, &evict_keys);
  save_event_async(false, evict_keys);
  return evict_count;
}

void PrefixCacheWithUpload::save_event_async(const bool is_insert,
                                             std::vector<XXH3Key>& keys) {
  threadpool_.schedule([this, is_insert = is_insert, keys = std::move(keys)]() {
    std::lock_guard<std::mutex> lock(this->mutex_);
    auto front_ptr = this->db_kvcache_events_.get_front_value();
    if (!front_ptr) {
      LOG(INFO) << "Front DoubleBufferKvCacheEvent is nullptr!";
      return;
    }
    if (!this->exited_.load()) {
      if (is_insert) {
        for (const auto& key : keys) {
          front_ptr->removed_cache.erase(key);
          front_ptr->stored_cache.insert(key);
        }
      } else {
        for (const auto& key : keys) {
          front_ptr->removed_cache.insert(key);
          front_ptr->stored_cache.erase(key);
        }
      }
    }
  });
}

KvCacheEvent* PrefixCacheWithUpload::get_upload_kvcache_events() {
  {
    std::lock_guard<std::mutex> lock(this->mutex_);
    db_kvcache_events_.swap();
  }
  if (!exited_.load()) {
    return db_kvcache_events_.get_back_value();
  } else {
    return nullptr;
  }
}

}  // namespace xllm

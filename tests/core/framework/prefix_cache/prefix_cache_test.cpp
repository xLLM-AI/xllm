#include "prefix_cache.h"

#include <absl/random/random.h>
#include <gtest/gtest.h>
#include <string.h>

#include <iostream>
#include <random>
#include <unordered_map>
#include <vector>

#include "framework/block/block_manager_impl.h"
#include "framework/encoder_cache/encoder_cache.h"
#include "util/hash_util.h"
namespace xllm {

namespace {

// Build the chained block hashes for `tokens`, matching the chain produced by
// xxh3_128bits_hash() inside PrefixCache and Sequence::update_block_hashes().
std::vector<XXH3Key> build_chained_hashes(const std::vector<int32_t>& tokens,
                                          uint32_t block_size) {
  const size_t n_blocks = tokens.size() / block_size;
  std::vector<XXH3Key> hashes;
  hashes.reserve(n_blocks);
  const Slice<int32_t> slice(tokens);
  for (size_t b = 0; b < n_blocks; ++b) {
    XXH3Key key;
    const uint8_t* pre = (b == 0) ? nullptr : hashes.back().data;
    xxh3_128bits_hash(
        pre, slice.slice(b * block_size, (b + 1) * block_size), key.data);
    hashes.emplace_back(key);
  }
  return hashes;
}

std::vector<int32_t> make_random_tokens(size_t count, uint32_t seed) {
  std::mt19937 gen(seed);
  std::uniform_int_distribution<int32_t> dist(0, 65535);
  std::vector<int32_t> tokens(count);
  for (int32_t& t : tokens) {
    t = dist(gen);
  }
  return tokens;
}

}  // namespace

TEST(PrefixCacheTest, EncoderCacheLifecycleIsIndependentFromDecoderCache) {
  const uint32_t block_size = 4;
  const std::vector<int32_t> tokens = {1, 2, 3, 4};
  const Slice<int32_t> token_slice(tokens);
  const torch::Tensor embedding = torch::ones({2, 2}, torch::kFloat32);
  const int64_t embedding_size = static_cast<int64_t>(embedding.nbytes());
  EncoderCache encoder_cache(/*max_size=*/embedding_size);
  PrefixCache decoder_cache(block_size);
  BlockManager::Options options;
  options.num_blocks(2).block_size(block_size);
  BlockManagerImpl block_manager(options);

  const XXH3Key image_key = hash_string("image-a");
  EXPECT_FALSE(encoder_cache.lookup(image_key).has_value());
  EXPECT_TRUE(decoder_cache.match(token_slice).empty());

  encoder_cache.insert(image_key, embedding);
  EXPECT_TRUE(encoder_cache.lookup(image_key).has_value());
  EXPECT_TRUE(decoder_cache.match(token_slice).empty());

  std::vector<Block> token_blocks = block_manager.allocate(/*n_blocks=*/1);
  decoder_cache.insert(token_slice, token_blocks);
  EXPECT_EQ(decoder_cache.match(token_slice).size(), 1);
  token_blocks.clear();

  encoder_cache.insert(hash_string("image-b"), embedding);
  EXPECT_FALSE(encoder_cache.lookup(image_key).has_value());
  EXPECT_EQ(decoder_cache.match(token_slice).size(), 1);
  EXPECT_EQ(decoder_cache.evict(/*n_blocks=*/1), 1);
}

void test_basic_operation(BlockManagerImpl* block_manager,
                          PrefixCache* prefix_cache,
                          uint32_t block_size) {
  EXPECT_EQ(prefix_cache->num_blocks(), 0);

  // token_ids number must be greater than  2 * block_size here
  std::vector<int32_t> token_ids = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  Slice<int32_t> slice_token_ids(token_ids);
  {
    auto block_matched = prefix_cache->match(slice_token_ids);

    EXPECT_EQ(block_matched.size(), 0);
  }

  uint32_t n_blocks = token_ids.size() / block_size;
  {
    std::vector<Block> token_blocks = block_manager->allocate(n_blocks);

    prefix_cache->insert(slice_token_ids, token_blocks);
  }

  EXPECT_EQ(prefix_cache->num_blocks(), n_blocks);

  {
    auto block_matched = prefix_cache->match(slice_token_ids);
    EXPECT_EQ(block_matched.size(), n_blocks);
  }

  {
    auto block_matched =
        prefix_cache->match(slice_token_ids.slice(block_size, 2 * block_size));
    EXPECT_EQ(block_matched.size(), 0);
  }

  EXPECT_EQ(prefix_cache->evict(1), 1);

  EXPECT_EQ(prefix_cache->num_blocks(), n_blocks - 1);

  {
    auto block_matched =
        prefix_cache->match(slice_token_ids.slice(0, block_size));
    EXPECT_EQ(block_matched.size(), 1);
  }

  {
    auto block_matched = prefix_cache->match(slice_token_ids.slice(block_size));
    EXPECT_EQ(block_matched.size(), 0);
  }
}

TEST(PrefixCacheTest, BasicOperation) {
  const uint32_t block_size = 4;
  const uint32_t total_blocks = 5;
  BlockManager::Options options;
  options.num_blocks(total_blocks).block_size(block_size);
  BlockManagerImpl block_manager(options);

  PrefixCache prefix_cache(block_size);

  test_basic_operation(&block_manager, &prefix_cache, block_size);
}

void test_insert_operation(BlockManagerImpl* block_manager,
                           PrefixCache* prefix_cache,
                           uint32_t block_size) {
  EXPECT_EQ(prefix_cache->num_blocks(), 0);

  // insert two-block firstly
  // token_ids number must be greater than  2 * block_size here
  std::vector<int32_t> token_ids = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  Slice<int32_t> slice_token_ids(token_ids);

  {
    auto block_matched = prefix_cache->match(slice_token_ids);

    EXPECT_EQ(block_matched.size(), 0);
  }

  uint32_t n_blocks = token_ids.size() / block_size;

  {
    std::vector<Block> token_blocks = block_manager->allocate(n_blocks);

    prefix_cache->insert(slice_token_ids, token_blocks);

    EXPECT_EQ(prefix_cache->num_blocks(), n_blocks);
  }

  {
    auto block_matched = prefix_cache->match(slice_token_ids);
    EXPECT_EQ(block_matched.size(), n_blocks);
  }

  // insert another two-block
  std::vector<int32_t> token_ids_1 = {9, 10, 11, 12, 13, 14, 15, 16, 17};
  Slice<int32_t> slice_token_ids_1(token_ids_1);

  {
    auto block_matched = prefix_cache->match(slice_token_ids_1);

    EXPECT_EQ(block_matched.size(), 0);
  }

  n_blocks = token_ids_1.size() / block_size;

  {
    std::vector<Block> token_blocks_1 = block_manager->allocate(n_blocks);

    prefix_cache->insert(slice_token_ids_1, token_blocks_1);

    EXPECT_EQ(prefix_cache->num_blocks(), 2 * n_blocks);
  }

  {
    auto block_matched = prefix_cache->match(slice_token_ids_1);
    EXPECT_EQ(block_matched.size(), n_blocks);
  }

  {
    auto block_matched = prefix_cache->match(slice_token_ids);
    EXPECT_EQ(block_matched.size(), n_blocks);
  }

  EXPECT_EQ(prefix_cache->evict(1), 1);
  EXPECT_EQ(prefix_cache->num_blocks(), 2 * n_blocks - 1);

  {
    auto block_matched = prefix_cache->match(slice_token_ids_1);
    EXPECT_EQ(block_matched.size(), n_blocks - 1);
  }

  {
    auto block_matched = prefix_cache->match(slice_token_ids);
    EXPECT_EQ(block_matched.size(), n_blocks);
  }

  EXPECT_EQ(prefix_cache->evict(1), 1);
  EXPECT_EQ(prefix_cache->num_blocks(), 2 * n_blocks - 2);

  {
    auto block_matched = prefix_cache->match(slice_token_ids_1);
    EXPECT_EQ(block_matched.size(), 0);
  }

  {
    auto block_matched = prefix_cache->match(slice_token_ids);
    EXPECT_EQ(block_matched.size(), n_blocks);

    prefix_cache->insert(slice_token_ids, block_matched);
  }

  EXPECT_EQ(prefix_cache->num_blocks(), n_blocks);

  {
    auto block_matched = prefix_cache->match(slice_token_ids);
    EXPECT_EQ(block_matched.size(), n_blocks);
  }

  prefix_cache->evict(1);
  EXPECT_EQ(prefix_cache->num_blocks(), n_blocks - 1);

  {
    auto block_matched = prefix_cache->match(slice_token_ids_1);
    EXPECT_EQ(block_matched.size(), 0);
  }

  {
    auto block_matched = prefix_cache->match(slice_token_ids);
    EXPECT_EQ(block_matched.size(), n_blocks - 1);
  }
}

TEST(PrefixCacheTest, InsertOperation) {
  const uint32_t block_size = 4;
  const uint32_t total_blocks = 5;
  BlockManager::Options options;
  options.num_blocks(total_blocks).block_size(block_size);
  BlockManagerImpl block_manager(options);

  PrefixCache prefix_cache(block_size);

  test_insert_operation(&block_manager, &prefix_cache, block_size);
}

void test_evict_operation(BlockManagerImpl* block_manager,
                          PrefixCache* prefix_cache,
                          uint32_t block_size) {
  EXPECT_EQ(prefix_cache->num_blocks(), 0);

  prefix_cache->evict(1);
  EXPECT_EQ(prefix_cache->num_blocks(), 0);

  // insert two-block firstly
  // token_ids number must be greater than  2 * block_size here
  std::vector<int32_t> token_ids = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  Slice<int32_t> slice_token_ids(token_ids);

  {
    auto block_matched = prefix_cache->match(slice_token_ids);

    EXPECT_EQ(block_matched.size(), 0);
  }

  uint32_t n_blocks = token_ids.size() / block_size;

  {
    std::vector<Block> token_blocks = block_manager->allocate(n_blocks);

    prefix_cache->insert(slice_token_ids, token_blocks);

    EXPECT_EQ(prefix_cache->num_blocks(), n_blocks);
  }

  {
    auto block_matched = prefix_cache->match(slice_token_ids);
    EXPECT_EQ(block_matched.size(), n_blocks);
  }

  EXPECT_EQ(block_manager->num_free_blocks(),
            block_manager->num_total_blocks() - n_blocks);

  EXPECT_EQ(prefix_cache->evict(n_blocks), n_blocks);

  EXPECT_EQ(block_manager->num_free_blocks(),
            block_manager->num_total_blocks());

  {
    auto block_matched = prefix_cache->match(slice_token_ids);
    EXPECT_EQ(block_matched.size(), 0);
  }

  {
    std::vector<Block> token_blocks = block_manager->allocate(n_blocks);

    prefix_cache->insert(slice_token_ids, token_blocks);

    EXPECT_EQ(prefix_cache->num_blocks(), n_blocks);
  }

  {
    auto block_matched = prefix_cache->match(slice_token_ids);
    EXPECT_EQ(block_matched.size(), n_blocks);
  }
}

TEST(PrefixCacheTest, EvictOperation) {
  const uint32_t block_size = 4;
  const uint32_t total_blocks = 5;
  BlockManager::Options options;
  options.num_blocks(total_blocks).block_size(block_size);
  BlockManagerImpl block_manager(options);

  PrefixCache prefix_cache(block_size);

  test_evict_operation(&block_manager, &prefix_cache, block_size);
}

TEST(HashUtilTest, XXHash3) {
  {
    std::vector<int32_t> tokens_1 = {1, 2, 3, 4, 5};
    uint8_t hash_value_1[XXH3_128BITS_HASH_VALUE_LEN];

    std::vector<int32_t> tokens_2 = {1, 2, 3, 4, 5};
    uint8_t hash_value_2[XXH3_128BITS_HASH_VALUE_LEN];

    xxh3_128bits_hash(nullptr, tokens_1, hash_value_1);
    xxh3_128bits_hash(nullptr, tokens_2, hash_value_2);

    EXPECT_EQ(
        std::memcmp(hash_value_1, hash_value_2, XXH3_128BITS_HASH_VALUE_LEN),
        0);
  }

  {
    std::vector<int32_t> tokens_1 = {1, 2, 3, 4, 5};
    uint8_t hash_value_1[XXH3_128BITS_HASH_VALUE_LEN];

    std::vector<int32_t> tokens_2 = {1, 2, 3, 5, 4};
    uint8_t hash_value_2[XXH3_128BITS_HASH_VALUE_LEN];

    xxh3_128bits_hash(nullptr, tokens_1, hash_value_1);
    xxh3_128bits_hash(nullptr, tokens_2, hash_value_2);

    EXPECT_NE(
        std::memcmp(hash_value_1, hash_value_2, XXH3_128BITS_HASH_VALUE_LEN),
        0);
  }

  {
    std::vector<int32_t> tokens_1 = {1, 2, 3, 4, 5};
    uint8_t hash_value_1[XXH3_128BITS_HASH_VALUE_LEN];

    std::vector<int32_t> tokens_2 = {2, 1, 3, 5, 4};
    uint8_t hash_value_2[XXH3_128BITS_HASH_VALUE_LEN];

    xxh3_128bits_hash(nullptr, tokens_1, hash_value_1);
    xxh3_128bits_hash(nullptr, tokens_2, hash_value_2);

    EXPECT_NE(
        std::memcmp(hash_value_1, hash_value_2, XXH3_128BITS_HASH_VALUE_LEN),
        0);
  }

  {
    std::vector<int32_t> tokens_1 = {1, 2, 3, 4, 5};
    uint8_t hash_value_1[XXH3_128BITS_HASH_VALUE_LEN];

    std::vector<int32_t> tokens_2 = {2, 1, 3, 5, 4};
    uint8_t hash_value_2[XXH3_128BITS_HASH_VALUE_LEN];

    xxh3_128bits_hash(nullptr, tokens_1, hash_value_1);
    xxh3_128bits_hash(nullptr, tokens_2, hash_value_2);

    EXPECT_NE(
        std::memcmp(hash_value_1, hash_value_2, XXH3_128BITS_HASH_VALUE_LEN),
        0);
  }

  {
    std::vector<int32_t> tokens_1 = {1, 2, 3, 4, 5};
    uint8_t hash_value_1[XXH3_128BITS_HASH_VALUE_LEN];

    std::vector<int32_t> tokens_2 = {1, 2, 3, 4};
    uint8_t hash_value_2[XXH3_128BITS_HASH_VALUE_LEN];

    xxh3_128bits_hash(nullptr, tokens_1, hash_value_1);
    xxh3_128bits_hash(nullptr, tokens_2, hash_value_2);

    EXPECT_NE(
        std::memcmp(hash_value_1, hash_value_2, XXH3_128BITS_HASH_VALUE_LEN),
        0);
  }

  {
    std::vector<int32_t> tokens_1 = {1, 2, 3, 4, 5};
    uint8_t hash_value_1[XXH3_128BITS_HASH_VALUE_LEN];

    std::vector<int32_t> tokens_2 = {1, 2};
    uint8_t hash_value_2[XXH3_128BITS_HASH_VALUE_LEN];

    xxh3_128bits_hash(nullptr, tokens_1, hash_value_1);
    xxh3_128bits_hash(nullptr, tokens_2, hash_value_2);

    EXPECT_NE(
        std::memcmp(hash_value_1, hash_value_2, XXH3_128BITS_HASH_VALUE_LEN),
        0);
  }

  {
    std::vector<int32_t> tokens_1 = {1, 2, 3, 4, 5};
    uint8_t hash_value_1[XXH3_128BITS_HASH_VALUE_LEN];

    std::vector<int32_t> tokens_2 = {1, 2, 3, 4, 5, 1, 2, 3, 4, 5};
    uint8_t hash_value_2[XXH3_128BITS_HASH_VALUE_LEN];

    xxh3_128bits_hash(nullptr, tokens_1, hash_value_1);
    xxh3_128bits_hash(nullptr, tokens_2, hash_value_2);

    EXPECT_NE(
        std::memcmp(hash_value_1, hash_value_2, XXH3_128BITS_HASH_VALUE_LEN),
        0);
  }

  {
    std::vector<int32_t> tokens_1(256, 1);
    uint8_t hash_value_1[XXH3_128BITS_HASH_VALUE_LEN];
    std::vector<int32_t> tokens_2(256, 2);
    uint8_t hash_value_2[XXH3_128BITS_HASH_VALUE_LEN];
    uint8_t hash_value_3[XXH3_128BITS_HASH_VALUE_LEN];

    xxh3_128bits_hash(nullptr, tokens_1, hash_value_1);
    xxh3_128bits_hash(hash_value_1, tokens_2, hash_value_2);
    xxh3_128bits_hash(hash_value_1, tokens_2, hash_value_3);

    EXPECT_EQ(XXH3Key(hash_value_2), XXH3Key(hash_value_3));
  }
}

// Validates the precompute change: match() must return identical blocks whether
// the chained hash is precomputed (passed in) or computed on the fly. Uses a
// long sequence and both full-hit and partial-hit (after eviction) cases.
TEST(PrefixCacheTest, PrecomputedHashMatchesComputedMatch) {
  const uint32_t block_size = 4;
  const uint32_t n_blocks = 200;
  BlockManager::Options options;
  options.num_blocks(n_blocks + 1).block_size(block_size);
  BlockManagerImpl block_manager(options);
  PrefixCache prefix_cache(block_size);

  std::vector<int32_t> token_ids =
      make_random_tokens(n_blocks * block_size, /*seed=*/2025);
  const Slice<int32_t> tokens(token_ids);
  const std::vector<XXH3Key> hashes =
      build_chained_hashes(token_ids, block_size);
  const Slice<XXH3Key> hash_slice(hashes);

  // Populate the cache with all blocks via the on-the-fly compute path. Scope
  // the allocated blocks so only the prefix cache holds a reference afterwards
  // (otherwise the blocks stay `is_shared()` and cannot be evicted below).
  {
    std::vector<Block> blocks = block_manager.allocate(n_blocks);
    prefix_cache.insert(tokens, blocks);
  }

  // Full hit: precomputed and computed paths must agree. Scope the matched
  // results too, so they do not pin the blocks as shared before eviction.
  {
    auto full_computed = prefix_cache.match(tokens);
    auto full_precomputed =
        prefix_cache.match(tokens, {}, MMData(), hash_slice);
    ASSERT_EQ(full_computed.size(), n_blocks);
    ASSERT_EQ(full_precomputed.size(), full_computed.size());
    for (size_t i = 0; i < full_computed.size(); ++i) {
      EXPECT_EQ(full_computed[i].id(), full_precomputed[i].id());
    }
  }

  // Partial hit: evict part of the cache (suffix-first under LRU), then both
  // paths must still agree (they early-break at the same first miss regardless
  // of hashing strategy).
  prefix_cache.evict(120);
  auto partial_computed = prefix_cache.match(tokens);
  auto partial_precomputed =
      prefix_cache.match(tokens, {}, MMData(), hash_slice);
  EXPECT_LT(partial_computed.size(), n_blocks);
  EXPECT_GT(partial_computed.size(), 0u);
  ASSERT_EQ(partial_precomputed.size(), partial_computed.size());
  for (size_t i = 0; i < partial_computed.size(); ++i) {
    EXPECT_EQ(partial_computed[i].id(), partial_precomputed[i].id());
  }
}

// Validates the precompute change on the insert path: inserting with
// precomputed hashes must populate the cache identically to inserting with
// on-the-fly hashing (same block count, same per-block hash value, same chain).
TEST(PrefixCacheTest, PrecomputedHashInsertMatchesComputedInsert) {
  const uint32_t block_size = 4;
  const uint32_t n_blocks = 50;
  BlockManager::Options options;
  options.num_blocks(2 * n_blocks + 2).block_size(block_size);
  BlockManagerImpl block_manager(options);

  std::vector<int32_t> token_ids =
      make_random_tokens(n_blocks * block_size, /*seed=*/99);
  const Slice<int32_t> tokens(token_ids);
  const std::vector<XXH3Key> hashes =
      build_chained_hashes(token_ids, block_size);
  const Slice<XXH3Key> hash_slice(hashes);

  // Cache A: insert via the on-the-fly compute path.
  PrefixCache cache_compute(block_size);
  std::vector<Block> blocks_a = block_manager.allocate(n_blocks);
  cache_compute.insert(tokens, blocks_a);

  // Cache B: insert via the precomputed path.
  PrefixCache cache_precomputed(block_size);
  std::vector<Block> blocks_b = block_manager.allocate(n_blocks);
  cache_precomputed.insert(
      tokens, blocks_b, /*existed=*/0, MMData(), hash_slice);

  EXPECT_EQ(cache_compute.num_blocks(), n_blocks);
  EXPECT_EQ(cache_precomputed.num_blocks(), n_blocks);

  for (size_t i = 0; i < n_blocks; ++i) {
    // Both insert paths must stamp the same hash onto each block ...
    EXPECT_EQ(std::memcmp(blocks_a[i].get_immutable_hash_value(),
                          blocks_b[i].get_immutable_hash_value(),
                          XXH3_128BITS_HASH_VALUE_LEN),
              0);
    // ... and it must equal the independently computed chain.
    EXPECT_EQ(std::memcmp(blocks_b[i].get_immutable_hash_value(),
                          hashes[i].data,
                          XXH3_128BITS_HASH_VALUE_LEN),
              0);
  }

  // A full match against the precomputed-built cache hits every block.
  auto matched = cache_precomputed.match(tokens, {}, MMData(), hash_slice);
  EXPECT_EQ(matched.size(), n_blocks);
}

// Validates the binary-search boundary equals the linear (early-break) boundary
// over a prefix-closed cache, across hit ratios on a long sequence. This is the
// correctness prerequisite for the linear-vs-binary probe benchmark.
TEST(PrefixCacheTest, BinaryProbeMatchesLinearProbe) {
  using HitMap = std::
      unordered_map<XXH3Key, int32_t, FixedStringKeyHash, FixedStringKeyEqual>;

  const uint32_t block_size = 16;
  const size_t n_blocks = 1024;
  const std::vector<int32_t> token_ids =
      make_random_tokens(n_blocks * block_size, /*seed=*/7);
  const std::vector<XXH3Key> hashes =
      build_chained_hashes(token_ids, block_size);

  auto linear_hit_len = [&](const HitMap& cache) -> size_t {
    size_t k = 0;
    for (; k < hashes.size(); ++k) {
      if (cache.find(hashes[k]) == cache.end()) {
        break;
      }
    }
    return k;
  };
  auto binary_hit_len = [&](const HitMap& cache) -> size_t {
    if (hashes.empty() || cache.find(hashes[0]) == cache.end()) {
      return 0;
    }
    size_t lo = 0;
    size_t hi = hashes.size() - 1;
    while (lo < hi) {
      const size_t mid = (lo + hi + 1) / 2;
      if (cache.find(hashes[mid]) != cache.end()) {
        lo = mid;
      } else {
        hi = mid - 1;
      }
    }
    return lo + 1;
  };

  for (size_t hit_len :
       {size_t{0}, size_t{1}, size_t{7}, size_t{512}, size_t{1023}, n_blocks}) {
    HitMap cache;
    cache.reserve(hit_len * 2 + 1);
    for (size_t i = 0; i < hit_len; ++i) {
      cache.emplace(hashes[i], 1);
    }
    EXPECT_EQ(linear_hit_len(cache), hit_len);
    EXPECT_EQ(binary_hit_len(cache), hit_len);
    EXPECT_EQ(binary_hit_len(cache), linear_hit_len(cache));
  }
}

}  // namespace xllm

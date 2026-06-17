#include "prefix_cache.h"

#include <absl/random/random.h>
#include <gtest/gtest.h>
#include <string.h>

#include <iostream>

#include "framework/block/block_manager_impl.h"
namespace xllm {

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
}

}  // namespace xllm

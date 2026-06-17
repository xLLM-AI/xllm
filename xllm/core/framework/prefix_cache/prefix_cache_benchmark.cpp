/* Copyright 2025 The xLLM Authors. All Rights Reserved.

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

#include <MurmurHash3.h>
#include <benchmark/benchmark.h>
#include <glog/logging.h>
#include <xxHash/xxhash.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <iterator>
#include <random>
#include <vector>

#include "framework/block/block_manager_impl.h"
#include "prefix_cache.h"
using namespace xllm;

// ============================================================================
// Existing prefix-cache benchmark
// ============================================================================

static void BM_HashSearch(benchmark::State& state) {
  const uint32_t block_size = 16;
  const uint32_t total_blocks = state.range(0);
  const uint32_t token_id_count = state.range(1);

  // LOG(INFO) << "total blocks " << total_blocks << ", token_id_count " <<
  // token_id_count;

  assert((token_id_count / block_size) < total_blocks);
  uint32_t n_blocks = token_id_count / block_size;

  state.PauseTiming();
  BlockManager::Options options;
  options.num_blocks(n_blocks + 1).block_size(block_size);
  BlockManagerImpl block_manager(options);

  PrefixCache prefix_cache(block_size);

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<unsigned> dist(0, 65535);

  std::vector<int32_t> token_ids(token_id_count);
  std::generate(
      token_ids.begin(), token_ids.end(), [&]() { return dist(gen); });

  std::vector<Block> token_blocks = block_manager.allocate(n_blocks);
  Slice<int32_t> slice_token_ids(token_ids);
  Slice<int32_t> match_token_ids(token_ids);
  prefix_cache.insert(slice_token_ids, token_blocks);
  state.ResumeTiming();

  size_t count = 0;
  for (auto _ : state) {
    benchmark::DoNotOptimize(prefix_cache.match(match_token_ids));

    ++count;
  }

  state.counters["iter count"] = count;
}

BENCHMARK(BM_HashSearch)
    ->Args({2048, 5000})
    ->Unit(benchmark::TimeUnit::kMillisecond)
    ->UseRealTime()
    ->Iterations(100)
    ->Repetitions(20)
    ->ReportAggregatesOnly(true);

// ============================================================================
// MurmurHash3_x64_128 vs XXH3_128bits_withSeed comparison benchmarks
// ============================================================================
// state.range(0) = data length in bytes
// Each iteration hashes a pre-generated random buffer of that length.

static constexpr uint32_t kMurmurSeed = 42;
static constexpr uint64_t kXXHSeed = 42;

// --------------- MurmurHash3_x64_128 ---------------

static void BM_MurmurHash3_x64_128(benchmark::State& state) {
  const int64_t data_len = state.range(0);

  // Prepare random input data (deterministic seed for reproducibility).
  std::mt19937 gen(12345);
  std::uniform_int_distribution<uint8_t> dist(0, 255);
  std::vector<uint8_t> data(data_len);
  std::generate(data.begin(), data.end(), [&]() { return dist(gen); });

  uint8_t out[16];  // 128-bit output

  for (auto _ : state) {
    MurmurHash3_x64_128(
        data.data(), static_cast<int>(data_len), kMurmurSeed, out);
    benchmark::DoNotOptimize(out);
  }

  state.SetBytesProcessed(state.iterations() * data_len);
}

// --------------- XXH3_128bits_withSeed ---------------

static void BM_XXH3_128bits_withSeed(benchmark::State& state) {
  const int64_t data_len = state.range(0);

  // Prepare random input data (same seed as Murmur benchmark).
  std::mt19937 gen(12345);
  std::uniform_int_distribution<uint8_t> dist(0, 255);
  std::vector<uint8_t> data(data_len);
  std::generate(data.begin(), data.end(), [&]() { return dist(gen); });

  for (auto _ : state) {
    XXH128_hash_t h = XXH3_128bits_withSeed(data.data(), data_len, kXXHSeed);
    benchmark::DoNotOptimize(h);
  }

  state.SetBytesProcessed(state.iterations() * data_len);
}

// --------------- Chain-hash: simulate prefix-cache usage ---------------
// Hash block_size tokens one block at a time, chaining the previous hash
// result as a prefix (same pattern as xxh3_128bits_hash() in prefix_cache.cpp).
// state.range(0) = number of int32_t tokens per block (block_size)
// state.range(1) = total number of blocks to hash

static void BM_MurmurHash3_Chain(benchmark::State& state) {
  const int64_t block_size = state.range(0);
  const int64_t num_blocks = state.range(1);
  const int64_t total_tokens = block_size * num_blocks;

  std::mt19937 gen(12345);
  std::uniform_int_distribution<int32_t> dist(0, 65535);
  std::vector<int32_t> tokens(total_tokens);
  std::generate(tokens.begin(), tokens.end(), [&]() { return dist(gen); });

  uint8_t hash_buf[16] = {};
  uint8_t key_buf[1024];

  for (auto _ : state) {
    std::memset(hash_buf, 0, sizeof(hash_buf));
    for (int64_t b = 0; b < num_blocks; ++b) {
      const int32_t* block_data = tokens.data() + b * block_size;
      const int data_bytes = static_cast<int>(block_size * sizeof(int32_t));
      if (b == 0) {
        MurmurHash3_x64_128(block_data, data_bytes, kMurmurSeed, hash_buf);
      } else {
        std::memcpy(key_buf, hash_buf, 16);
        std::memcpy(key_buf + 16, block_data, data_bytes);
        MurmurHash3_x64_128(key_buf, 16 + data_bytes, kMurmurSeed, hash_buf);
      }
    }
    benchmark::DoNotOptimize(hash_buf);
  }

  state.SetBytesProcessed(state.iterations() * total_tokens *
                          static_cast<int64_t>(sizeof(int32_t)));
}

static void BM_XXH3_128_Chain(benchmark::State& state) {
  const int64_t block_size = state.range(0);
  const int64_t num_blocks = state.range(1);
  const int64_t total_tokens = block_size * num_blocks;

  std::mt19937 gen(12345);
  std::uniform_int_distribution<int32_t> dist(0, 65535);
  std::vector<int32_t> tokens(total_tokens);
  std::generate(tokens.begin(), tokens.end(), [&]() { return dist(gen); });

  uint8_t key_buf[1024];

  for (auto _ : state) {
    XXH128_hash_t hash = {};
    for (int64_t b = 0; b < num_blocks; ++b) {
      const int32_t* block_data = tokens.data() + b * block_size;
      const size_t data_bytes = block_size * sizeof(int32_t);
      if (b == 0) {
        hash = XXH3_128bits_withSeed(block_data, data_bytes, kXXHSeed);
      } else {
        std::memcpy(key_buf, &hash, sizeof(hash));
        std::memcpy(key_buf + sizeof(hash), block_data, data_bytes);
        hash =
            XXH3_128bits_withSeed(key_buf, sizeof(hash) + data_bytes, kXXHSeed);
      }
    }
    benchmark::DoNotOptimize(hash);
  }

  state.SetBytesProcessed(state.iterations() * total_tokens *
                          static_cast<int64_t>(sizeof(int32_t)));
}

// --------------- Register benchmarks ---------------

// Single-shot hash at different data lengths (bytes).
BENCHMARK(BM_MurmurHash3_x64_128)
    ->Arg(64)
    ->Arg(128)
    ->Arg(256)
    ->Arg(512)
    ->Arg(1024)
    ->Unit(benchmark::TimeUnit::kNanosecond);

BENCHMARK(BM_XXH3_128bits_withSeed)
    ->Arg(64)
    ->Arg(128)
    ->Arg(256)
    ->Arg(512)
    ->Arg(1024)
    ->Unit(benchmark::TimeUnit::kNanosecond);

// Chain-hash: simulate prefix-cache block hashing.
// Args: {block_size (tokens), num_blocks}
BENCHMARK(BM_MurmurHash3_Chain)
    ->Args({16, 32})
    ->Args({16, 128})
    ->Args({16, 512})
    ->Unit(benchmark::TimeUnit::kNanosecond);

BENCHMARK(BM_XXH3_128_Chain)
    ->Args({16, 32})
    ->Args({16, 128})
    ->Args({16, 512})
    ->Unit(benchmark::TimeUnit::kNanosecond);

BENCHMARK_MAIN();

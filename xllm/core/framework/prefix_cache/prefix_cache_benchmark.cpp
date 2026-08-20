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
#include <unordered_map>
#include <vector>

#include "framework/block/block_manager_impl.h"
#include "prefix_cache.h"
#include "util/hash_util.h"
using namespace xllm;

// ============================================================================
// Helpers shared by the linear-vs-binary prefix-match probe benchmarks.
//
// `match` already early-breaks on the first miss, so the linear path does
// exactly `hit_len + 1` hash-table probes while binary search does O(log n).
// Both must still collect `hit_len` blocks afterwards, so the ONLY thing that
// differs between the two strategies is the boundary-finding probes isolated
// below. These helpers therefore measure just the probe cost.
// ============================================================================

namespace {

using HitMap = std::
    unordered_map<XXH3Key, int32_t, FixedStringKeyHash, FixedStringKeyEqual>;

// Build the chained block hashes for `tokens`, identical to the chain produced
// by xxh3_128bits_hash() in prefix_cache.cpp.
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

// Populate a cache that holds the first `hit_len` blocks of the chain. This
// models the prefix-closed invariant guaranteed by the LRU (suffix-first)
// eviction: if block i is cached, every block < i is cached too.
HitMap make_cache(const std::vector<XXH3Key>& hashes, size_t hit_len) {
  HitMap cache;
  cache.reserve(hit_len * 2);
  for (size_t i = 0; i < hit_len; ++i) {
    cache.emplace(hashes[i], 1);
  }
  return cache;
}

// Current strategy: probe from the front, stop at the first miss.
size_t find_hit_len_linear(const HitMap& cache,
                           const std::vector<XXH3Key>& hashes) {
  size_t k = 0;
  for (; k < hashes.size(); ++k) {
    if (cache.find(hashes[k]) == cache.end()) {
      break;
    }
  }
  return k;
}

// Binary search for the longest hit, relying on the prefix-closed invariant.
size_t find_hit_len_binary(const HitMap& cache,
                           const std::vector<XXH3Key>& hashes) {
  if (hashes.empty() || cache.find(hashes[0]) == cache.end()) {
    return 0;
  }
  // Invariant: hashes[lo] is present, hashes[hi + 1] is absent.
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
}

}  // namespace

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

// ============================================================================
// Linear vs binary prefix-match probing.
// Args: {n_blocks, hit_percent}  (hit_len = n_blocks * hit_percent / 100)
//
// Reading the numbers:
//   - hit_percent = 100 (full hit): linear does n_blocks probes, binary does
//     ~log2(n_blocks); binary should win and the gap grows with n_blocks.
//   - hit_percent small: linear early-breaks after hit_len+1 probes, so binary
//     (always ~log2(n_blocks)) can be equal or SLOWER. This is the case that
//     shows binary is not a free win.
// ============================================================================

namespace {

struct ProbeFixture {
  std::vector<int32_t> tokens;
  std::vector<XXH3Key> hashes;
  HitMap cache;
};

ProbeFixture make_probe_fixture(size_t n_blocks, int64_t hit_percent) {
  constexpr uint32_t kBlockSize = 16;
  const size_t hit_len = n_blocks * static_cast<size_t>(hit_percent) / 100;

  std::mt19937 gen(12345);
  std::uniform_int_distribution<int32_t> dist(0, 65535);
  ProbeFixture fixture;
  fixture.tokens.resize(n_blocks * kBlockSize);
  std::generate(fixture.tokens.begin(), fixture.tokens.end(), [&]() {
    return dist(gen);
  });
  fixture.hashes = build_chained_hashes(fixture.tokens, kBlockSize);
  fixture.cache = make_cache(fixture.hashes, hit_len);
  return fixture;
}

}  // namespace

static void BM_MatchProbe_Linear(benchmark::State& state) {
  const ProbeFixture fixture =
      make_probe_fixture(state.range(0), state.range(1));
  for (auto _ : state) {
    benchmark::DoNotOptimize(
        find_hit_len_linear(fixture.cache, fixture.hashes));
  }
}

static void BM_MatchProbe_Binary(benchmark::State& state) {
  const ProbeFixture fixture =
      make_probe_fixture(state.range(0), state.range(1));
  for (auto _ : state) {
    benchmark::DoNotOptimize(
        find_hit_len_binary(fixture.cache, fixture.hashes));
  }
}

// Sweep sequence length (n_blocks) at full / half / short / zero hit ratios.
// At block_size=16, n_blocks=1024 ~= 16K-token sequence.
BENCHMARK(BM_MatchProbe_Linear)
    ->Args({64, 100})
    ->Args({256, 100})
    ->Args({1024, 100})
    ->Args({1024, 50})
    ->Args({1024, 10})
    ->Args({1024, 0})
    ->Unit(benchmark::TimeUnit::kNanosecond);

BENCHMARK(BM_MatchProbe_Binary)
    ->Args({64, 100})
    ->Args({256, 100})
    ->Args({1024, 100})
    ->Args({1024, 50})
    ->Args({1024, 10})
    ->Args({1024, 0})
    ->Unit(benchmark::TimeUnit::kNanosecond);

BENCHMARK_MAIN();

/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <vector>

#include "moe_topk.cuh"

namespace xllm::kernel::cuda {
namespace test {

using namespace reduce_topk;

// ============================================================================
// Helper: CPU reference top-K (sort descending by value, ascending by index on
// tie, then take the first K).
// ============================================================================
template <typename T>
void cpuTopK(const std::vector<T>& values,
             const std::vector<int32_t>& indices,
             int k,
             std::vector<T>& outValues,
             std::vector<int32_t>& outIndices) {
  struct Pair {
    T val;
    int32_t idx;
  };
  std::vector<Pair> pairs(values.size());
  for (size_t i = 0; i < values.size(); ++i) {
    pairs[i] = {values[i], indices[i]};
  }
  std::sort(pairs.begin(), pairs.end(), [](const Pair& a, const Pair& b) {
    if (a.val != b.val) return a.val > b.val;
    return a.idx < b.idx;
  });
  outValues.resize(k);
  outIndices.resize(k);
  for (int i = 0; i < k; ++i) {
    outValues[i] = pairs[i].val;
    outIndices[i] = pairs[i].idx;
  }
}

// ============================================================================
// 1. Host-side pack / unpack roundtrip tests
//    makeCmpVal and unpack are __host__ __device__, so we can call them from
//    the CPU when compiling with nvcc.
// ============================================================================

class TopKRedTypePackUnpackTest : public ::testing::Test {};

TEST_F(TopKRedTypePackUnpackTest, FloatRoundtrip) {
  using RedType = TopKRedType<float>;
  const float testValues[] = {
      0.0f, 1.0f, -1.0f, 3.14159f, -100.5f, 1e10f, -1e10f, 0.001f};
  const int32_t testIndices[] = {0, 1, 31, 100, 1000, 65534, 65535};

  for (float val : testValues) {
    for (int32_t idx : testIndices) {
      auto packed = RedType::makeCmpVal(val, idx);
      float unpackedVal;
      int32_t unpackedIdx;
      RedType::unpack(unpackedVal, unpackedIdx, packed);
      EXPECT_EQ(unpackedVal, val)
          << "Value mismatch for val=" << val << " idx=" << idx;
      EXPECT_EQ(unpackedIdx, idx)
          << "Index mismatch for val=" << val << " idx=" << idx;
    }
  }
}

TEST_F(TopKRedTypePackUnpackTest, IntRoundtrip) {
  using RedType = TopKRedType<int>;
  const int testValues[] = {0, 1, -1, 42, -100, INT32_MAX, INT32_MIN};
  const int32_t testIndices[] = {0, 1, 100, 65534, 65535};

  for (int val : testValues) {
    for (int32_t idx : testIndices) {
      auto packed = RedType::makeCmpVal(val, idx);
      int unpackedVal;
      int32_t unpackedIdx;
      RedType::unpack(unpackedVal, unpackedIdx, packed);
      EXPECT_EQ(unpackedVal, val)
          << "Value mismatch for val=" << val << " idx=" << idx;
      EXPECT_EQ(unpackedIdx, idx)
          << "Index mismatch for val=" << val << " idx=" << idx;
    }
  }
}

// Regression: negative index -1 must NOT survive a pack/unpack roundtrip.
// kMaxIdx - (-1) = 65536, truncated to 0 in 16 bits, unpacks to 65535.
TEST_F(TopKRedTypePackUnpackTest, NegativeIndexDoesNotRoundtrip) {
  using RedType = TopKRedType<float>;
  const int32_t badIdx = -1;
  auto packed = RedType::makeCmpVal(1.0f, badIdx);
  float v;
  int32_t idx;
  RedType::unpack(v, idx, packed);
  EXPECT_NE(idx, badIdx) << "Negative index should NOT roundtrip correctly";
  EXPECT_EQ(idx, 65535) << "Expected 65535 after failed roundtrip of -1";
}

// kMaxIdx (65535) as sentinel DOES roundtrip correctly.
TEST_F(TopKRedTypePackUnpackTest, SentinelKMaxIdxRoundtrip) {
  using RedType = TopKRedType<float>;
  const int32_t sentinel = RedType::kMaxIdx;  // 65535
  const float minVal = -std::numeric_limits<float>::infinity();
  auto packed = RedType::makeCmpVal(minVal, sentinel);
  float v;
  int32_t idx;
  RedType::unpack(v, idx, packed);
  EXPECT_EQ(idx, sentinel);
  // -inf should also survive (TwiddleIn/Out handles it).
  EXPECT_EQ(v, minVal);
}

// Zero index roundtrip.
TEST_F(TopKRedTypePackUnpackTest, ZeroIndexRoundtrip) {
  using RedType = TopKRedType<float>;
  auto packed = RedType::makeCmpVal(2.5f, 0);
  float v;
  int32_t idx;
  RedType::unpack(v, idx, packed);
  EXPECT_EQ(idx, 0);
  EXPECT_EQ(v, 2.5f);
}

// ============================================================================
// 2. Host-side comparison ordering tests
// ============================================================================

class TopKRedTypeOrderingTest : public ::testing::Test {};

// Larger value → larger packed representation (same index).
TEST_F(TopKRedTypeOrderingTest, LargerValueHigherPriority) {
  using RedType = TopKRedType<float>;
  const int32_t idx = 42;
  auto p1 = RedType::makeCmpVal(10.0f, idx);
  auto p2 = RedType::makeCmpVal(5.0f, idx);
  EXPECT_GT(p1, p2) << "Larger value should produce larger packed value";
}

// For equal values, smaller index → larger packed representation (higher
// priority).
TEST_F(TopKRedTypeOrderingTest, SmallerIndexHigherPriority) {
  using RedType = TopKRedType<float>;
  auto p1 = RedType::makeCmpVal(7.0f, 10);
  auto p2 = RedType::makeCmpVal(7.0f, 20);
  EXPECT_GT(p1, p2) << "Smaller index should have higher priority";
}

// Negative value ordering: -1 > -10.
TEST_F(TopKRedTypeOrderingTest, NegativeValueOrdering) {
  using RedType = TopKRedType<float>;
  auto p1 = RedType::makeCmpVal(-1.0f, 0);
  auto p2 = RedType::makeCmpVal(-10.0f, 0);
  EXPECT_GT(p1, p2) << "-1 should rank higher than -10";
}

// Integer ordering.
TEST_F(TopKRedTypeOrderingTest, IntOrdering) {
  using RedType = TopKRedType<int>;
  auto p1 = RedType::makeCmpVal(100, 0);
  auto p2 = RedType::makeCmpVal(50, 0);
  EXPECT_GT(p1, p2);
  auto p3 = RedType::makeCmpVal(-1, 0);
  auto p4 = RedType::makeCmpVal(-100, 0);
  EXPECT_GT(p3, p4);
}

// Sentinel (minValue, kMaxIdx) should be the smallest packed value among
// any real candidates, ensuring it always loses in a max-reduction.
TEST_F(TopKRedTypeOrderingTest, SentinelIsSmallest) {
  using RedType = TopKRedType<float>;
  auto sentinel = RedType::makeCmpVal(-std::numeric_limits<float>::infinity(),
                                      RedType::kMaxIdx);
  auto real = RedType::makeCmpVal(0.0f, 0);
  EXPECT_GT(real, sentinel);
}

// Monotone sweep: ascending values with fixed index should produce ascending
// packed values.
TEST_F(TopKRedTypeOrderingTest, MonotoneSweepFloat) {
  using RedType = TopKRedType<float>;
  typename RedType::TypeCmp prev = RedType::makeCmpVal(-1000.0f, 0);
  for (float v = -999.0f; v <= 1000.0f; v += 1.0f) {
    auto cur = RedType::makeCmpVal(v, 0);
    EXPECT_GT(cur, prev);
    prev = cur;
  }
}

// ============================================================================
// 3. Device-side kernel wrappers
// ============================================================================

// Kernel: each of 32 warp lanes holds one (value, index) pair.
// Performs warp-level top-K reduction; lane 0 writes K results.
template <int K, typename Type>
__global__ void testReduceTopKSingleKernel(const Type* __restrict__ values,
                                           const int32_t* __restrict__ indices,
                                           Type* __restrict__ outValues,
                                           int32_t* __restrict__ outIndices,
                                           Type minValue,
                                           int actualK) {
  auto warp = cg::tiled_partition<kWarpSize>(cg::this_thread_block());
  const int lane = threadIdx.x;

  Type val = values[lane];
  int32_t idx = indices[lane];

  Type out[K];
  int32_t outIdx[K];
  reduceTopK<K>(warp, out, outIdx, val, idx, minValue, actualK);

  if (lane == 0) {
    for (int i = 0; i < actualK; ++i) {
      outValues[i] = out[i];
      outIndices[i] = outIdx[i];
    }
  }
}

// Kernel: each of 32 warp lanes holds N (value, index) pairs.
// Input layout: values[lane * N + n], indices[lane * N + n].
template <int K, typename Type, int N>
__global__ void testReduceTopKMultiKernel(const Type* __restrict__ values,
                                          const int32_t* __restrict__ indices,
                                          Type* __restrict__ outValues,
                                          int32_t* __restrict__ outIndices,
                                          Type minValue,
                                          int actualK) {
  auto warp = cg::tiled_partition<kWarpSize>(cg::this_thread_block());
  const int lane = threadIdx.x;

  Type val[N];
  int32_t idx[N];
  for (int n = 0; n < N; ++n) {
    val[n] = values[lane * N + n];
    idx[n] = indices[lane * N + n];
  }

  Type out[K];
  int32_t outIdx[K];
  reduceTopK<K, Type, N>(warp, out, outIdx, val, idx, minValue, actualK);

  if (lane == 0) {
    for (int i = 0; i < actualK; ++i) {
      outValues[i] = out[i];
      outIndices[i] = outIdx[i];
    }
  }
}

// ============================================================================
// RAII wrapper for device memory
// ============================================================================
template <typename T>
struct DevBuf {
  T* ptr = nullptr;
  size_t count = 0;

  explicit DevBuf(size_t n) : count(n) { cudaMalloc(&ptr, n * sizeof(T)); }
  ~DevBuf() {
    if (ptr) cudaFree(ptr);
  }
  void upload(const T* host) {
    cudaMemcpy(ptr, host, count * sizeof(T), cudaMemcpyHostToDevice);
  }
  void download(T* host) const {
    cudaMemcpy(host, ptr, count * sizeof(T), cudaMemcpyDeviceToHost);
  }
  // non-copyable
  DevBuf(const DevBuf&) = delete;
  DevBuf& operator=(const DevBuf&) = delete;
};

// ============================================================================
// 4. Device-side integration tests
// ============================================================================

class ReduceTopKDeviceTest : public ::testing::Test {
 protected:
  void SetUp() override {
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
      GTEST_SKIP() << "CUDA not available, skipping test.";
    }
  }
};

// ---------- Single-value reduceTopK tests ----------

// K=1: find the global maximum across 32 warp lanes.
TEST_F(ReduceTopKDeviceTest, SingleValueTopK1) {
  constexpr int K = 1;
  constexpr int N = kWarpSize;

  std::vector<float> h_vals(N);
  std::vector<int32_t> h_idx(N);
  for (int i = 0; i < N; ++i) {
    h_vals[i] = static_cast<float>(i * 3 - 40);  // [-40 .. 53]
    h_idx[i] = i;
  }

  std::vector<float> refV;
  std::vector<int32_t> refI;
  cpuTopK(h_vals, h_idx, K, refV, refI);

  DevBuf<float> d_vals(N), d_outV(K);
  DevBuf<int32_t> d_idx(N), d_outI(K);
  d_vals.upload(h_vals.data());
  d_idx.upload(h_idx.data());

  testReduceTopKSingleKernel<K>
      <<<1, 32>>>(d_vals.ptr,
                  d_idx.ptr,
                  d_outV.ptr,
                  d_outI.ptr,
                  -std::numeric_limits<float>::infinity(),
                  K);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  float outV[K];
  int32_t outI[K];
  d_outV.download(outV);
  d_outI.download(outI);

  for (int i = 0; i < K; ++i) {
    EXPECT_FLOAT_EQ(outV[i], refV[i]);
    EXPECT_EQ(outI[i], refI[i]);
  }
}

// K=4: find the top-4 across 32 warp lanes with pseudo-random data.
TEST_F(ReduceTopKDeviceTest, SingleValueTopK4) {
  constexpr int K = 4;
  constexpr int N = kWarpSize;

  std::mt19937 rng(42);
  std::vector<float> h_vals(N);
  std::vector<int32_t> h_idx(N);
  for (int i = 0; i < N; ++i) {
    h_vals[i] = static_cast<float>(rng() % 1000) / 10.0f;
    h_idx[i] = i;
  }

  std::vector<float> refV;
  std::vector<int32_t> refI;
  cpuTopK(h_vals, h_idx, K, refV, refI);

  DevBuf<float> d_vals(N), d_outV(K);
  DevBuf<int32_t> d_idx(N), d_outI(K);
  d_vals.upload(h_vals.data());
  d_idx.upload(h_idx.data());

  testReduceTopKSingleKernel<K>
      <<<1, 32>>>(d_vals.ptr,
                  d_idx.ptr,
                  d_outV.ptr,
                  d_outI.ptr,
                  -std::numeric_limits<float>::infinity(),
                  K);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  float outV[K];
  int32_t outI[K];
  d_outV.download(outV);
  d_outI.download(outI);

  for (int i = 0; i < K; ++i) {
    EXPECT_FLOAT_EQ(outV[i], refV[i]);
    EXPECT_EQ(outI[i], refI[i]);
  }
}

// All values identical → top-K should return the K smallest indices.
TEST_F(ReduceTopKDeviceTest, DuplicateValuesPreferSmallerIndex) {
  constexpr int K = 3;
  constexpr int N = kWarpSize;

  std::vector<float> h_vals(N, 42.0f);
  std::vector<int32_t> h_idx(N);
  for (int i = 0; i < N; ++i) h_idx[i] = i;

  DevBuf<float> d_vals(N), d_outV(K);
  DevBuf<int32_t> d_idx(N), d_outI(K);
  d_vals.upload(h_vals.data());
  d_idx.upload(h_idx.data());

  testReduceTopKSingleKernel<K>
      <<<1, 32>>>(d_vals.ptr,
                  d_idx.ptr,
                  d_outV.ptr,
                  d_outI.ptr,
                  -std::numeric_limits<float>::infinity(),
                  K);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  int32_t outI[K];
  d_outI.download(outI);

  for (int i = 0; i < K; ++i) {
    EXPECT_EQ(outI[i], i) << "Should prefer smallest indices when values tie";
  }
}

// actualK < K: only the first actualK results are meaningful.
TEST_F(ReduceTopKDeviceTest, SingleValueActualKLessThanK) {
  constexpr int K = 4;
  constexpr int N = kWarpSize;
  const int actualK = 2;

  std::mt19937 rng(99);
  std::vector<float> h_vals(N);
  std::vector<int32_t> h_idx(N);
  for (int i = 0; i < N; ++i) {
    h_vals[i] = static_cast<float>(rng() % 500) / 5.0f;
    h_idx[i] = i;
  }

  std::vector<float> refV;
  std::vector<int32_t> refI;
  cpuTopK(h_vals, h_idx, actualK, refV, refI);

  DevBuf<float> d_vals(N), d_outV(K);
  DevBuf<int32_t> d_idx(N), d_outI(K);
  d_vals.upload(h_vals.data());
  d_idx.upload(h_idx.data());

  testReduceTopKSingleKernel<K>
      <<<1, 32>>>(d_vals.ptr,
                  d_idx.ptr,
                  d_outV.ptr,
                  d_outI.ptr,
                  -std::numeric_limits<float>::infinity(),
                  actualK);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  float outV[K];
  int32_t outI[K];
  d_outV.download(outV);
  d_outI.download(outI);

  for (int i = 0; i < actualK; ++i) {
    EXPECT_FLOAT_EQ(outV[i], refV[i]);
    EXPECT_EQ(outI[i], refI[i]);
  }
}

// Negative values: top-K should still pick the largest (closest to 0).
TEST_F(ReduceTopKDeviceTest, SingleValueAllNegative) {
  constexpr int K = 2;
  constexpr int N = kWarpSize;

  std::vector<float> h_vals(N);
  std::vector<int32_t> h_idx(N);
  for (int i = 0; i < N; ++i) {
    h_vals[i] = -static_cast<float>(i + 1);  // [-1, -2, ..., -32]
    h_idx[i] = i;
  }

  std::vector<float> refV;
  std::vector<int32_t> refI;
  cpuTopK(h_vals, h_idx, K, refV, refI);

  DevBuf<float> d_vals(N), d_outV(K);
  DevBuf<int32_t> d_idx(N), d_outI(K);
  d_vals.upload(h_vals.data());
  d_idx.upload(h_idx.data());

  testReduceTopKSingleKernel<K>
      <<<1, 32>>>(d_vals.ptr,
                  d_idx.ptr,
                  d_outV.ptr,
                  d_outI.ptr,
                  -std::numeric_limits<float>::infinity(),
                  K);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  float outV[K];
  int32_t outI[K];
  d_outV.download(outV);
  d_outI.download(outI);

  // Top-2 should be -1 (idx 0) and -2 (idx 1).
  for (int i = 0; i < K; ++i) {
    EXPECT_FLOAT_EQ(outV[i], refV[i]);
    EXPECT_EQ(outI[i], refI[i]);
  }
}

// ---------- Multi-value reduceTopK tests (N <= 4 path) ----------

// N=2 per thread, K=2 → total 64 candidates.
TEST_F(ReduceTopKDeviceTest, MultiValueN2K2) {
  constexpr int K = 2;
  constexpr int N_PER_THREAD = 2;
  constexpr int TOTAL = kWarpSize * N_PER_THREAD;

  std::mt19937 rng(123);
  std::vector<float> h_vals(TOTAL);
  std::vector<int32_t> h_idx(TOTAL);
  for (int i = 0; i < TOTAL; ++i) {
    h_vals[i] = static_cast<float>(rng() % 2000) / 10.0f - 100.0f;
    h_idx[i] = i;
  }

  std::vector<float> refV;
  std::vector<int32_t> refI;
  cpuTopK(h_vals, h_idx, K, refV, refI);

  DevBuf<float> d_vals(TOTAL), d_outV(K);
  DevBuf<int32_t> d_idx(TOTAL), d_outI(K);
  d_vals.upload(h_vals.data());
  d_idx.upload(h_idx.data());

  testReduceTopKMultiKernel<K, float, N_PER_THREAD>
      <<<1, 32>>>(d_vals.ptr,
                  d_idx.ptr,
                  d_outV.ptr,
                  d_outI.ptr,
                  -std::numeric_limits<float>::infinity(),
                  K);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  float outV[K];
  int32_t outI[K];
  d_outV.download(outV);
  d_outI.download(outI);

  for (int i = 0; i < K; ++i) {
    EXPECT_FLOAT_EQ(outV[i], refV[i]);
    EXPECT_EQ(outI[i], refI[i]);
  }
}

// N=3 per thread, K=2 → total 96 candidates.
TEST_F(ReduceTopKDeviceTest, MultiValueN3K2) {
  constexpr int K = 2;
  constexpr int N_PER_THREAD = 3;
  constexpr int TOTAL = kWarpSize * N_PER_THREAD;

  std::mt19937 rng(321);
  std::vector<float> h_vals(TOTAL);
  std::vector<int32_t> h_idx(TOTAL);
  for (int i = 0; i < TOTAL; ++i) {
    h_vals[i] = static_cast<float>(rng() % 3000) / 10.0f - 150.0f;
    h_idx[i] = i;
  }

  std::vector<float> refV;
  std::vector<int32_t> refI;
  cpuTopK(h_vals, h_idx, K, refV, refI);

  DevBuf<float> d_vals(TOTAL), d_outV(K);
  DevBuf<int32_t> d_idx(TOTAL), d_outI(K);
  d_vals.upload(h_vals.data());
  d_idx.upload(h_idx.data());

  testReduceTopKMultiKernel<K, float, N_PER_THREAD>
      <<<1, 32>>>(d_vals.ptr,
                  d_idx.ptr,
                  d_outV.ptr,
                  d_outI.ptr,
                  -std::numeric_limits<float>::infinity(),
                  K);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  float outV[K];
  int32_t outI[K];
  d_outV.download(outV);
  d_outI.download(outI);

  for (int i = 0; i < K; ++i) {
    EXPECT_FLOAT_EQ(outV[i], refV[i]);
    EXPECT_EQ(outI[i], refI[i]);
  }
}

// N=4 per thread, K=3 → total 128 candidates (boundary of N<=4 path).
TEST_F(ReduceTopKDeviceTest, MultiValueN4K3) {
  constexpr int K = 3;
  constexpr int N_PER_THREAD = 4;
  constexpr int TOTAL = kWarpSize * N_PER_THREAD;

  std::mt19937 rng(456);
  std::vector<float> h_vals(TOTAL);
  std::vector<int32_t> h_idx(TOTAL);
  for (int i = 0; i < TOTAL; ++i) {
    h_vals[i] = static_cast<float>(rng() % 5000) / 10.0f - 250.0f;
    h_idx[i] = i;
  }

  std::vector<float> refV;
  std::vector<int32_t> refI;
  cpuTopK(h_vals, h_idx, K, refV, refI);

  DevBuf<float> d_vals(TOTAL), d_outV(K);
  DevBuf<int32_t> d_idx(TOTAL), d_outI(K);
  d_vals.upload(h_vals.data());
  d_idx.upload(h_idx.data());

  testReduceTopKMultiKernel<K, float, N_PER_THREAD>
      <<<1, 32>>>(d_vals.ptr,
                  d_idx.ptr,
                  d_outV.ptr,
                  d_outI.ptr,
                  -std::numeric_limits<float>::infinity(),
                  K);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  float outV[K];
  int32_t outI[K];
  d_outV.download(outV);
  d_outI.download(outI);

  for (int i = 0; i < K; ++i) {
    EXPECT_FLOAT_EQ(outV[i], refV[i]);
    EXPECT_EQ(outI[i], refI[i]);
  }
}

// ---------- Multi-value reduceTopK tests (N > 4 path) ----------
// When N > 4 the implementation splits into numLoops rounds of 4 values each,
// accumulates intermediate results into a buffer, then does a final reduction.
// This exercises the code path that previously had the -1 sentinel bug.

// N=8 per thread, K=2 → total 256 candidates.
TEST_F(ReduceTopKDeviceTest, MultiValueN8K2_LargePath) {
  constexpr int K = 2;
  constexpr int N_PER_THREAD = 8;
  constexpr int TOTAL = kWarpSize * N_PER_THREAD;

  std::mt19937 rng(789);
  std::vector<float> h_vals(TOTAL);
  std::vector<int32_t> h_idx(TOTAL);
  for (int i = 0; i < TOTAL; ++i) {
    h_vals[i] = static_cast<float>(rng() % 10000) / 10.0f;
    h_idx[i] = i;
  }

  std::vector<float> refV;
  std::vector<int32_t> refI;
  cpuTopK(h_vals, h_idx, K, refV, refI);

  DevBuf<float> d_vals(TOTAL), d_outV(K);
  DevBuf<int32_t> d_idx(TOTAL), d_outI(K);
  d_vals.upload(h_vals.data());
  d_idx.upload(h_idx.data());

  testReduceTopKMultiKernel<K, float, N_PER_THREAD>
      <<<1, 32>>>(d_vals.ptr,
                  d_idx.ptr,
                  d_outV.ptr,
                  d_outI.ptr,
                  -std::numeric_limits<float>::infinity(),
                  K);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  float outV[K];
  int32_t outI[K];
  d_outV.download(outV);
  d_outI.download(outI);

  for (int i = 0; i < K; ++i) {
    EXPECT_FLOAT_EQ(outV[i], refV[i]);
    EXPECT_EQ(outI[i], refI[i]);
  }
}

// N=8 per thread, K=4 → total 256, wider K.
TEST_F(ReduceTopKDeviceTest, MultiValueN8K4_LargePath) {
  constexpr int K = 4;
  constexpr int N_PER_THREAD = 8;
  constexpr int TOTAL = kWarpSize * N_PER_THREAD;

  std::mt19937 rng(1024);
  std::vector<float> h_vals(TOTAL);
  std::vector<int32_t> h_idx(TOTAL);
  for (int i = 0; i < TOTAL; ++i) {
    h_vals[i] = static_cast<float>(rng() % 8000) / 10.0f - 400.0f;
    h_idx[i] = i;
  }

  std::vector<float> refV;
  std::vector<int32_t> refI;
  cpuTopK(h_vals, h_idx, K, refV, refI);

  DevBuf<float> d_vals(TOTAL), d_outV(K);
  DevBuf<int32_t> d_idx(TOTAL), d_outI(K);
  d_vals.upload(h_vals.data());
  d_idx.upload(h_idx.data());

  testReduceTopKMultiKernel<K, float, N_PER_THREAD>
      <<<1, 32>>>(d_vals.ptr,
                  d_idx.ptr,
                  d_outV.ptr,
                  d_outI.ptr,
                  -std::numeric_limits<float>::infinity(),
                  K);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  float outV[K];
  int32_t outI[K];
  d_outV.download(outV);
  d_outI.download(outI);

  for (int i = 0; i < K; ++i) {
    EXPECT_FLOAT_EQ(outV[i], refV[i]);
    EXPECT_EQ(outI[i], refI[i]);
  }
}

// N=12 per thread, K=3 → total 384, numLoops=3.
TEST_F(ReduceTopKDeviceTest, MultiValueN12K3_LargePath) {
  constexpr int K = 3;
  constexpr int N_PER_THREAD = 12;
  constexpr int TOTAL = kWarpSize * N_PER_THREAD;

  std::mt19937 rng(2048);
  std::vector<float> h_vals(TOTAL);
  std::vector<int32_t> h_idx(TOTAL);
  for (int i = 0; i < TOTAL; ++i) {
    h_vals[i] = static_cast<float>(rng() % 6000) / 10.0f - 300.0f;
    h_idx[i] = i;
  }

  std::vector<float> refV;
  std::vector<int32_t> refI;
  cpuTopK(h_vals, h_idx, K, refV, refI);

  DevBuf<float> d_vals(TOTAL), d_outV(K);
  DevBuf<int32_t> d_idx(TOTAL), d_outI(K);
  d_vals.upload(h_vals.data());
  d_idx.upload(h_idx.data());

  testReduceTopKMultiKernel<K, float, N_PER_THREAD>
      <<<1, 32>>>(d_vals.ptr,
                  d_idx.ptr,
                  d_outV.ptr,
                  d_outI.ptr,
                  -std::numeric_limits<float>::infinity(),
                  K);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  float outV[K];
  int32_t outI[K];
  d_outV.download(outV);
  d_outI.download(outI);

  for (int i = 0; i < K; ++i) {
    EXPECT_FLOAT_EQ(outV[i], refV[i]);
    EXPECT_EQ(outI[i], refI[i]);
  }
}

// N=16 per thread, K=2 → total 512, maximum supported N.
TEST_F(ReduceTopKDeviceTest, MultiValueN16K2_LargePath) {
  constexpr int K = 2;
  constexpr int N_PER_THREAD = 16;
  constexpr int TOTAL = kWarpSize * N_PER_THREAD;

  std::mt19937 rng(4096);
  std::vector<float> h_vals(TOTAL);
  std::vector<int32_t> h_idx(TOTAL);
  for (int i = 0; i < TOTAL; ++i) {
    h_vals[i] = static_cast<float>(rng() % 20000) / 100.0f - 100.0f;
    h_idx[i] = i;
  }

  std::vector<float> refV;
  std::vector<int32_t> refI;
  cpuTopK(h_vals, h_idx, K, refV, refI);

  DevBuf<float> d_vals(TOTAL), d_outV(K);
  DevBuf<int32_t> d_idx(TOTAL), d_outI(K);
  d_vals.upload(h_vals.data());
  d_idx.upload(h_idx.data());

  testReduceTopKMultiKernel<K, float, N_PER_THREAD>
      <<<1, 32>>>(d_vals.ptr,
                  d_idx.ptr,
                  d_outV.ptr,
                  d_outI.ptr,
                  -std::numeric_limits<float>::infinity(),
                  K);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  float outV[K];
  int32_t outI[K];
  d_outV.download(outV);
  d_outI.download(outI);

  for (int i = 0; i < K; ++i) {
    EXPECT_FLOAT_EQ(outV[i], refV[i]);
    EXPECT_EQ(outI[i], refI[i]);
  }
}

// ---------- Edge case: N>4 path with all-duplicate values ----------
// Exercises the sentinel buffer path when no single candidate stands out.
TEST_F(ReduceTopKDeviceTest, MultiValueN8K3_AllDuplicate) {
  constexpr int K = 3;
  constexpr int N_PER_THREAD = 8;
  constexpr int TOTAL = kWarpSize * N_PER_THREAD;

  std::vector<float> h_vals(TOTAL, 7.7f);
  std::vector<int32_t> h_idx(TOTAL);
  for (int i = 0; i < TOTAL; ++i) h_idx[i] = i;

  DevBuf<float> d_vals(TOTAL), d_outV(K);
  DevBuf<int32_t> d_idx(TOTAL), d_outI(K);
  d_vals.upload(h_vals.data());
  d_idx.upload(h_idx.data());

  testReduceTopKMultiKernel<K, float, N_PER_THREAD>
      <<<1, 32>>>(d_vals.ptr,
                  d_idx.ptr,
                  d_outV.ptr,
                  d_outI.ptr,
                  -std::numeric_limits<float>::infinity(),
                  K);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  float outV[K];
  int32_t outI[K];
  d_outV.download(outV);
  d_outI.download(outI);

  // All values are the same, so top-K indices should be the K smallest.
  for (int i = 0; i < K; ++i) {
    EXPECT_FLOAT_EQ(outV[i], 7.7f);
    EXPECT_EQ(outI[i], i)
        << "Duplicate-value top-K should prefer smallest indices";
  }
}

// ---------- Integer type on device ----------

// Single-value reduceTopK with int type, K=2.
TEST_F(ReduceTopKDeviceTest, IntTypeSingleValueTopK2) {
  constexpr int K = 2;
  constexpr int N = kWarpSize;

  std::vector<int> h_vals(N);
  std::vector<int32_t> h_idx(N);
  for (int i = 0; i < N; ++i) {
    h_vals[i] = (i * 7 + 13) % 100 - 50;
    h_idx[i] = i;
  }

  std::vector<int> refV;
  std::vector<int32_t> refI;
  cpuTopK(h_vals, h_idx, K, refV, refI);

  DevBuf<int> d_vals(N), d_outV(K);
  DevBuf<int32_t> d_idx(N), d_outI(K);
  d_vals.upload(h_vals.data());
  d_idx.upload(h_idx.data());

  testReduceTopKSingleKernel<K>
      <<<1, 32>>>(d_vals.ptr, d_idx.ptr, d_outV.ptr, d_outI.ptr, INT32_MIN, K);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  int outV[K];
  int32_t outI[K];
  d_outV.download(outV);
  d_outI.download(outI);

  for (int i = 0; i < K; ++i) {
    EXPECT_EQ(outV[i], refV[i]);
    EXPECT_EQ(outI[i], refI[i]);
  }
}

}  // namespace test
}  // namespace xllm::kernel::cuda

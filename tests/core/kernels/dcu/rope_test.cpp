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
#include <hip/hip_runtime.h>
#include <torch/cuda.h>
#include <torch/torch.h>

#include <functional>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

#include "kernels/cuda/cuda_ops_api.h"

namespace xllm::kernel::dcu {
namespace test {
namespace {

enum class QKLayout {
  kTHD,    // [T, H, D]
  kBSHD,   // [B, S, H, D]
  kTFlat,  // [T, H*D]
  kBSFlat  // [B, S, H*D]
};

torch::Tensor make_cos_sin_cache_cpu(int64_t max_pos, int64_t rot_dim) {
  CHECK(rot_dim % 2 == 0);
  const int64_t half = rot_dim / 2;

  auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
  torch::Tensor cache = torch::empty({max_pos, rot_dim}, opts);

  auto* ptr = cache.data_ptr<float>();
  for (int64_t p = 0; p < max_pos; ++p) {
    for (int64_t i = 0; i < half; ++i) {
      const float theta = static_cast<float>((p + 1) * (i + 1)) * 0.01f;
      ptr[p * rot_dim + i] = std::cos(theta);
      ptr[p * rot_dim + half + i] = std::sin(theta);
    }
  }
  return cache;
}

std::pair<double, double> get_tolerance(torch::ScalarType dtype) {
  if (dtype == torch::kFloat32) {
    return {1e-6, 1e-5};
  }
  if (dtype == torch::kFloat16) {
    return {5e-3, 5e-3};
  }
  if (dtype == torch::kBFloat16) {
    return {1e-2, 1e-2};
  }
  return {1e-5, 1e-5};
}

void expect_throws_c10(const std::function<void()>& fn) {
  try {
    fn();
    FAIL() << "Expected c10::Error, but no exception was thrown.";
  } catch (const c10::Error&) {
    SUCCEED();
  } catch (...) {
    FAIL() << "Expected c10::Error, but got a different exception type.";
  }
}

// CPU canonical reference on [T, H, D]
void apply_rope_reference_cpu(
    torch::Tensor positions_cpu,           // [T], int64
    torch::Tensor query_cpu,               // [T, H, D], float32
    std::optional<torch::Tensor> key_cpu,  // [T, Hkv, D], float32
    torch::Tensor cos_sin_cpu,             // [max_pos, rot_dim], float32
    bool is_neox) {
  CHECK(positions_cpu.device().is_cpu());
  CHECK(query_cpu.device().is_cpu());
  CHECK(cos_sin_cpu.device().is_cpu());
  CHECK(query_cpu.dim() == 3);

  const int64_t num_tokens = positions_cpu.size(0);
  const int64_t num_heads = query_cpu.size(1);
  const int64_t head_size = query_cpu.size(2);
  const int64_t rot_dim = cos_sin_cpu.size(1);
  const int64_t embed_dim = rot_dim / 2;

  CHECK(rot_dim % 2 == 0);
  CHECK(rot_dim <= head_size);

  auto positions = positions_cpu.contiguous();
  auto query = query_cpu.contiguous();
  auto cos_sin = cos_sin_cpu.contiguous();

  auto* pos_ptr = positions.data_ptr<int64_t>();
  auto* q_ptr = query.data_ptr<float>();
  auto* cache_ptr = cos_sin.data_ptr<float>();

  float* k_ptr = nullptr;
  int64_t num_kv_heads = 0;
  torch::Tensor key_contig;
  if (key_cpu.has_value()) {
    CHECK(key_cpu->device().is_cpu());
    CHECK(key_cpu->dim() == 3);
    num_kv_heads = key_cpu->size(1);
    key_contig = key_cpu->contiguous();
    k_ptr = key_contig.data_ptr<float>();
  }

  auto apply_one = [&](float* base_ptr, int64_t heads) {
    for (int64_t t = 0; t < num_tokens; ++t) {
      const int64_t pos = pos_ptr[t];
      const float* cos_ptr = cache_ptr + pos * rot_dim;
      const float* sin_ptr = cos_ptr + embed_dim;

      for (int64_t h = 0; h < heads; ++h) {
        float* arr = base_ptr + (t * heads + h) * head_size;
        for (int64_t off = 0; off < embed_dim; ++off) {
          int64_t x_index, y_index;
          float c, s;
          if (is_neox) {
            x_index = off;
            y_index = embed_dim + off;
            c = cos_ptr[x_index];
            s = sin_ptr[x_index];
          } else {
            x_index = 2 * off;
            y_index = 2 * off + 1;
            c = cos_ptr[off];
            s = sin_ptr[off];
          }
          const float x = arr[x_index];
          const float y = arr[y_index];
          arr[x_index] = x * c - y * s;
          arr[y_index] = y * c + x * s;
        }
      }
    }
  };

  apply_one(q_ptr, num_heads);
  if (k_ptr != nullptr) {
    apply_one(k_ptr, num_kv_heads);
  }

  query_cpu.copy_(query);
  if (key_cpu.has_value()) {
    key_cpu->copy_(key_contig);
  }
}

torch::Tensor make_positions_1d_cpu(int64_t num_tokens, int64_t max_pos) {
  auto opts = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU);
  torch::Tensor pos = torch::empty({num_tokens}, opts);
  auto* p = pos.data_ptr<int64_t>();
  for (int64_t i = 0; i < num_tokens; ++i) {
    p[i] = (i * 3 + 1) % max_pos;
  }
  return pos;
}

torch::Tensor make_positions_for_layout_cpu(int64_t batch_size,
                                            int64_t seq_len,
                                            int64_t max_pos,
                                            bool positions_2d) {
  if (positions_2d) {
    auto pos1d = make_positions_1d_cpu(batch_size * seq_len, max_pos);
    return pos1d.view({batch_size, seq_len}).clone();
  }
  return make_positions_1d_cpu(batch_size * seq_len, max_pos);
}

torch::Tensor to_canonical_3d(torch::Tensor x,
                              QKLayout layout,
                              int64_t batch_size,
                              int64_t seq_len,
                              int64_t heads,
                              int64_t head_size) {
  switch (layout) {
    case QKLayout::kTHD:
      return x.view({batch_size * seq_len, heads, head_size});
    case QKLayout::kBSHD:
      return x.view({batch_size * seq_len, heads, head_size});
    case QKLayout::kTFlat:
      return x.view({batch_size * seq_len, heads, head_size});
    case QKLayout::kBSFlat:
      return x.view({batch_size * seq_len, heads, head_size});
  }
  LOG(FATAL) << "Unknown layout";
  return torch::Tensor();
}

torch::Tensor from_canonical_3d_like(torch::Tensor canonical,
                                     QKLayout layout,
                                     int64_t batch_size,
                                     int64_t seq_len,
                                     int64_t heads,
                                     int64_t head_size) {
  switch (layout) {
    case QKLayout::kTHD:
      return canonical.view({batch_size * seq_len, heads, head_size}).clone();
    case QKLayout::kBSHD:
      return canonical.view({batch_size, seq_len, heads, head_size}).clone();
    case QKLayout::kTFlat:
      return canonical.view({batch_size * seq_len, heads * head_size}).clone();
    case QKLayout::kBSFlat:
      return canonical.view({batch_size, seq_len, heads * head_size}).clone();
  }
  LOG(FATAL) << "Unknown layout";
  return torch::Tensor();
}

torch::Tensor make_q_or_k(torch::ScalarType dtype,
                          torch::Device device,
                          QKLayout layout,
                          int64_t batch_size,
                          int64_t seq_len,
                          int64_t heads,
                          int64_t head_size) {
  auto opts = torch::TensorOptions().dtype(dtype).device(device);
  switch (layout) {
    case QKLayout::kTHD:
      return torch::randn({batch_size * seq_len, heads, head_size}, opts);
    case QKLayout::kBSHD:
      return torch::randn({batch_size, seq_len, heads, head_size}, opts);
    case QKLayout::kTFlat:
      return torch::randn({batch_size * seq_len, heads * head_size}, opts);
    case QKLayout::kBSFlat:
      return torch::randn({batch_size, seq_len, heads * head_size}, opts);
  }
  LOG(FATAL) << "Unknown layout";
  return torch::Tensor();
}

std::string dtype_to_string(torch::ScalarType dtype) {
  switch (dtype) {
    case torch::kFloat32:
      return "fp32";
    case torch::kFloat16:
      return "fp16";
    case torch::kBFloat16:
      return "bf16";
    default:
      return "unknown";
  }
}

std::string layout_to_string(QKLayout layout) {
  switch (layout) {
    case QKLayout::kTHD:
      return "THD";
    case QKLayout::kBSHD:
      return "BSHD";
    case QKLayout::kTFlat:
      return "TFlat";
    case QKLayout::kBSFlat:
      return "BSFlat";
  }
  return "unknown";
}

class RotaryEmbeddingTest : public ::testing::Test {
 protected:
  void SetUp() override {
    if (!torch::cuda::is_available()) {
      GTEST_SKIP() << "CUDA/ROCm device not available";
    }
    torch::manual_seed(2026);
    device_ = torch::Device(torch::kCUDA, 0);
  }

  void synchronize() const {
    auto err = hipDeviceSynchronize();
    ASSERT_EQ(err, hipSuccess) << hipGetErrorString(err);
  }

  void run_case(torch::ScalarType dtype,
                bool is_neox,
                bool with_key,
                bool positions_2d,
                QKLayout q_layout,
                QKLayout k_layout,
                int64_t batch_size,
                int64_t seq_len,
                int64_t num_heads,
                int64_t num_kv_heads,
                int64_t head_size,
                int64_t rot_dim) const {
    ASSERT_TRUE(rot_dim % 2 == 0);
    ASSERT_TRUE(rot_dim == head_size);
    if (with_key) {
      ASSERT_TRUE(num_heads % num_kv_heads == 0);
    }

    const int64_t num_tokens = batch_size * seq_len;
    const int64_t max_pos = std::max<int64_t>(32, num_tokens + 8);

    auto positions_cpu = make_positions_for_layout_cpu(
        batch_size, seq_len, max_pos, positions_2d);
    auto positions_dev = positions_cpu.to(device_, /*non_blocking=*/false);

    auto query = make_q_or_k(
        dtype, device_, q_layout, batch_size, seq_len, num_heads, head_size);
    std::optional<torch::Tensor> key = std::nullopt;
    if (with_key) {
      key = make_q_or_k(dtype,
                        device_,
                        k_layout,
                        batch_size,
                        seq_len,
                        num_kv_heads,
                        head_size);
    }

    // CPU reference always in fp32 canonical [T,H,D]
    auto query_ref = to_canonical_3d(query.cpu().to(torch::kFloat32),
                                     q_layout,
                                     batch_size,
                                     seq_len,
                                     num_heads,
                                     head_size)
                         .clone();

    std::optional<torch::Tensor> key_ref = std::nullopt;
    if (with_key) {
      key_ref = to_canonical_3d(key->cpu().to(torch::kFloat32),
                                k_layout,
                                batch_size,
                                seq_len,
                                num_kv_heads,
                                head_size)
                    .clone();
    }

    auto cache_cpu = make_cos_sin_cache_cpu(max_pos, rot_dim);
    auto cache_dev = cache_cpu.to(device_).to(dtype);

    auto positions_ref_1d = positions_cpu.view({num_tokens}).contiguous();
    apply_rope_reference_cpu(
        positions_ref_1d, query_ref, key_ref, cache_cpu, is_neox);

    xllm::kernel::cuda::rotary_embedding(
        positions_dev, query, key, cache_dev, is_neox);
    synchronize();

    auto query_out_canonical = to_canonical_3d(query.cpu().to(torch::kFloat32),
                                               q_layout,
                                               batch_size,
                                               seq_len,
                                               num_heads,
                                               head_size);

    auto [rtol, atol] = get_tolerance(dtype);
    EXPECT_TRUE(torch::allclose(query_out_canonical, query_ref, rtol, atol))
        << "query mismatch"
        << " dtype=" << dtype_to_string(dtype) << " is_neox=" << is_neox
        << " positions_2d=" << positions_2d
        << " q_layout=" << layout_to_string(q_layout)
        << " head_size=" << head_size << " rot_dim=" << rot_dim
        << " num_heads=" << num_heads << " num_kv_heads=" << num_kv_heads;

    if (with_key) {
      auto key_out_canonical = to_canonical_3d(key->cpu().to(torch::kFloat32),
                                               k_layout,
                                               batch_size,
                                               seq_len,
                                               num_kv_heads,
                                               head_size);
      EXPECT_TRUE(torch::allclose(key_out_canonical, *key_ref, rtol, atol))
          << "key mismatch"
          << " dtype=" << dtype_to_string(dtype) << " is_neox=" << is_neox
          << " positions_2d=" << positions_2d
          << " k_layout=" << layout_to_string(k_layout)
          << " head_size=" << head_size << " rot_dim=" << rot_dim
          << " num_heads=" << num_heads << " num_kv_heads=" << num_kv_heads;
    }
  }

  torch::Device device_ = torch::Device(torch::kCPU);
};

// Basic regression: NeoX with key and full rotary dimension.
TEST_F(RotaryEmbeddingTest, Neox_MHA_WithKey_RotDimEqHeadSize) {
  run_case(torch::kFloat32,
           true,
           true,
           false,
           QKLayout::kTHD,
           QKLayout::kTHD,
           1,
           4,
           2,
           2,
           8,
           8);
}

// GQA: query heads outnumber KV heads.
TEST_F(RotaryEmbeddingTest, Neox_GQA_WithKey_RotDimEqHeadSize) {
  run_case(torch::kFloat32,
           true,
           true,
           false,
           QKLayout::kTHD,
           QKLayout::kTHD,
           1,
           6,
           8,
           2,
           16,
           16);
}

// GPT-J + no key
TEST_F(RotaryEmbeddingTest, GptJ_NoKey) {
  run_case(torch::kFloat32,
           false,
           false,
           false,
           QKLayout::kTHD,
           QKLayout::kTHD,
           1,
           5,
           2,
           2,
           8,
           8);
}

// GPT-J + GQA + key
TEST_F(RotaryEmbeddingTest, GptJ_GQA_WithKey) {
  run_case(torch::kFloat32,
           false,
           true,
           false,
           QKLayout::kTHD,
           QKLayout::kTHD,
           1,
           5,
           6,
           2,
           8,
           8);
}

// 2D positions + 4D Q/K
TEST_F(RotaryEmbeddingTest, Positions2D_BSHD_WithKey) {
  run_case(torch::kFloat32,
           true,
           true,
           true,
           QKLayout::kBSHD,
           QKLayout::kBSHD,
           2,
           3,
           4,
           2,
           8,
           8);
}

// 1D positions + flat Q/K
TEST_F(RotaryEmbeddingTest, Positions1D_Flat_WithKey) {
  run_case(torch::kFloat32,
           true,
           true,
           false,
           QKLayout::kTFlat,
           QKLayout::kTFlat,
           1,
           6,
           4,
           2,
           8,
           8);
}

// 2D positions + flat Q/K
TEST_F(RotaryEmbeddingTest, Positions2D_Flat_WithKey) {
  run_case(torch::kFloat32,
           true,
           true,
           true,
           QKLayout::kBSFlat,
           QKLayout::kBSFlat,
           2,
           3,
           4,
           2,
           8,
           8);
}

// Dtype coverage: fp16.
TEST_F(RotaryEmbeddingTest, DTypeFp16_Neox_GQA) {
  run_case(torch::kFloat16,
           true,
           true,
           false,
           QKLayout::kTHD,
           QKLayout::kTHD,
           1,
           6,
           8,
           2,
           16,
           16);
}

// Dtype coverage: bf16.
TEST_F(RotaryEmbeddingTest, DTypeBf16_Neox_GQA) {
  run_case(torch::kBFloat16,
           true,
           true,
           false,
           QKLayout::kTHD,
           QKLayout::kTHD,
           1,
           6,
           8,
           2,
           16,
           16);
}

// NeoX + no key
TEST_F(RotaryEmbeddingTest, Neox_NoKey) {
  run_case(torch::kFloat32,
           true,
           false,
           false,
           QKLayout::kTHD,
           QKLayout::kTHD,
           1,
           5,
           4,
           4,
           8,
           8);
}

// 2D positions + no key
TEST_F(RotaryEmbeddingTest, Positions2D_BSHD_NoKey) {
  run_case(torch::kFloat32,
           true,
           false,
           true,
           QKLayout::kBSHD,
           QKLayout::kBSHD,
           2,
           3,
           4,
           4,
           8,
           8);
}

// 2D positions + flat + no key
TEST_F(RotaryEmbeddingTest, Positions2D_Flat_NoKey) {
  run_case(torch::kFloat32,
           true,
           false,
           true,
           QKLayout::kBSFlat,
           QKLayout::kBSFlat,
           2,
           3,
           4,
           4,
           8,
           8);
}

// Different Q/K layouts: Q=[B,S,H,D], K=[B,S,H*D].
TEST_F(RotaryEmbeddingTest, DifferentLayout_QBSHD_KBSFlat) {
  run_case(torch::kFloat32,
           true,
           true,
           true,
           QKLayout::kBSHD,
           QKLayout::kBSFlat,
           2,
           3,
           4,
           2,
           8,
           8);
}

// Different Q/K layouts: Q=[B,S,H*D], K=[B,S,H,D].
TEST_F(RotaryEmbeddingTest, DifferentLayout_QBSFlat_KBSHD) {
  run_case(torch::kFloat32,
           true,
           true,
           true,
           QKLayout::kBSFlat,
           QKLayout::kBSHD,
           2,
           3,
           4,
           2,
           8,
           8);
}

// Different Q/K layouts: Q=[T,H,D], K=[T,H*D].
TEST_F(RotaryEmbeddingTest, DifferentLayout_QTHD_KTFlat) {
  run_case(torch::kFloat32,
           false,
           true,
           false,
           QKLayout::kTHD,
           QKLayout::kTFlat,
           1,
           6,
           6,
           2,
           8,
           8);
}

// Minimal boundary case.
TEST_F(RotaryEmbeddingTest, Boundary_MinimalShape) {
  run_case(torch::kFloat32,
           true,
           true,
           false,
           QKLayout::kTHD,
           QKLayout::kTHD,
           1,
           1,
           1,
           1,
           2,
           2);
}

// Single token without key.
TEST_F(RotaryEmbeddingTest, Boundary_SingleToken_NoKey) {
  run_case(torch::kFloat32,
           false,
           false,
           false,
           QKLayout::kTHD,
           QKLayout::kTHD,
           1,
           1,
           2,
           2,
           8,
           8);
}

// ---------- Negative tests ----------

// Invalid positions dimension.
TEST_F(RotaryEmbeddingTest, Invalid_PositionsDim) {
  auto i64_dev = torch::TensorOptions().dtype(torch::kInt64).device(device_);
  auto f32_dev = torch::TensorOptions().dtype(torch::kFloat32).device(device_);

  auto positions = torch::zeros({2, 3, 4}, i64_dev);
  auto query = torch::randn({2, 3, 4, 8}, f32_dev);
  auto cache = make_cos_sin_cache_cpu(16, 8).to(device_);

  ASSERT_DEATH(
      {
        xllm::kernel::cuda::rotary_embedding(
            positions, query, std::nullopt, cache, true);
      },
      "positions must have shape");
}

// Query and positions have mismatched token counts.
TEST_F(RotaryEmbeddingTest, Invalid_QueryPositionsMismatch_1D) {
  auto i64_dev = torch::TensorOptions().dtype(torch::kInt64).device(device_);
  auto f32_dev = torch::TensorOptions().dtype(torch::kFloat32).device(device_);

  auto positions = torch::tensor(std::vector<int64_t>{0, 1, 2, 3}, i64_dev);
  auto query = torch::randn({5, 2, 8}, f32_dev);
  auto cache = make_cos_sin_cache_cpu(16, 8).to(device_);

  ASSERT_DEATH(
      {
        xllm::kernel::cuda::rotary_embedding(
            positions, query, std::nullopt, cache, true);
      },
      "same number of tokens");
}

// Key and positions are inconsistent.
TEST_F(RotaryEmbeddingTest, Invalid_KeyPositionsMismatch_2D) {
  auto i64_dev = torch::TensorOptions().dtype(torch::kInt64).device(device_);
  auto f32_dev = torch::TensorOptions().dtype(torch::kFloat32).device(device_);

  auto positions = torch::zeros({2, 3}, i64_dev);
  auto query = torch::randn({2, 3, 4, 8}, f32_dev);
  auto key = torch::randn({2, 4, 2, 8}, f32_dev);
  auto cache = make_cos_sin_cache_cpu(16, 8).to(device_);

  ASSERT_DEATH(
      {
        xllm::kernel::cuda::rotary_embedding(
            positions, query, key, cache, true);
      },
      "same batch_size and seq_len");
}

// Invalid GQA relation: num_heads % num_kv_heads != 0.
TEST_F(RotaryEmbeddingTest, Invalid_NumHeadsNotDivisibleByNumKvHeads) {
  auto i64_dev = torch::TensorOptions().dtype(torch::kInt64).device(device_);
  auto f32_dev = torch::TensorOptions().dtype(torch::kFloat32).device(device_);

  auto positions = torch::tensor(std::vector<int64_t>{0, 1, 2, 3}, i64_dev);
  auto query = torch::randn({4, 6, 8}, f32_dev);
  auto key = torch::randn({4, 4, 8}, f32_dev);
  auto cache = make_cos_sin_cache_cpu(16, 8).to(device_);

  ASSERT_DEATH(
      {
        xllm::kernel::cuda::rotary_embedding(
            positions, query, key, cache, true);
      },
      "Check failed");
}

// ---------- Smoke cases ----------

// Medium-scale smoke case.
TEST_F(RotaryEmbeddingTest, Smoke_MediumScale_GQA) {
  run_case(torch::kFloat32,
           true,
           true,
           true,
           QKLayout::kBSHD,
           QKLayout::kBSHD,
           4,
           32,
           16,
           4,
           64,
           64);
}

// Large-scale smoke case.
TEST_F(RotaryEmbeddingTest, Smoke_LargeScale_GQA) {
  run_case(torch::kFloat16,
           true,
           true,
           true,
           QKLayout::kBSHD,
           QKLayout::kBSHD,
           8,
           128,
           32,
           8,
           64,
           64);
}

}  // namespace
}  // namespace test
}  // namespace xllm::kernel::dcu

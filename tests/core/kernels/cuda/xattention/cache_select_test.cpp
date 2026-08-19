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

#include <glog/logging.h>
#include <gtest/gtest.h>
#include <torch/torch.h>

#include <limits>
#include <vector>

#include "core/kernels/cuda/xattention/xattention_ops_api.h"

class CacheSelctTest : public ::testing::Test {
 protected:
  void SetUp() override {
    if (!torch::cuda::is_available()) {
      GTEST_SKIP() << "CUDA not available, skipping test.";
    }
    device_ = torch::Device(torch::kCUDA);
    dtype_ = torch::kFloat16;
  }

  torch::Device device_ = torch::kCPU;
  torch::ScalarType dtype_ = torch::kFloat16;
};

void cache_select(
    const torch::Tensor& beam_index,  // [batch * beam, 1] - out_token_index
    std::vector<torch::Tensor>&
        unshared_k_cache,  // per layer: [max_num_request, beam_size,
                           // max_decode_step, kv_heads, head_dim]
    std::vector<torch::Tensor>&
        unshared_v_cache,  // per layer: [max_num_request, beam_size,
                           // max_decode_step, kv_heads, head_dim]
    const torch::Tensor& block_table,  // [batch_size, 1]
    int64_t decode_step,               // current round (step 0, 1, ...)
    int64_t beam_size,                 // beam width
    int64_t layer_num) {               // number of layers

  int64_t batch_size = block_table.size(0);
  int64_t total_beams = beam_index.size(0);
  int64_t kv_heads = unshared_k_cache[0].size(3);
  int64_t head_dim = unshared_k_cache[0].size(4);
  CHECK_EQ(total_beams, batch_size * beam_size) << "beam_index size mismatch";

  if (layer_num > 0) {
    int64_t max_num_request = unshared_k_cache[0].size(0);
    int64_t max_decode_step = unshared_k_cache[0].size(2);
    auto beam_index_reshaped =
        beam_index.reshape({batch_size, beam_size})
            .to(torch::kLong);  // [batch_size, beam_size]
    auto parent_beam = (beam_index_reshaped / beam_size)
                           .to(torch::kLong);  // [batch_size, beam_size]

    auto _block_table = block_table.select(1, 0);

    auto cache_opts = unshared_k_cache[0].options();
    auto ori_k_cache = torch::zeros(
        {layer_num, batch_size, beam_size, max_decode_step, kv_heads, head_dim},
        cache_opts);
    auto ori_v_cache = torch::zeros(
        {layer_num, batch_size, beam_size, max_decode_step, kv_heads, head_dim},
        cache_opts);

    for (int64_t layer = 0; layer < layer_num; ++layer) {
      auto k_cache = unshared_k_cache[layer];
      auto v_cache = unshared_v_cache[layer];
      ori_k_cache[layer].copy_(k_cache.index_select(0, _block_table));
      ori_v_cache[layer].copy_(v_cache.index_select(0, _block_table));
    }

    for (int64_t layer = 0; layer < layer_num; ++layer) {
      auto k_cache = unshared_k_cache[layer];
      auto v_cache = unshared_v_cache[layer];

      for (int64_t b = 0; b < batch_size; ++b) {
        int64_t request_id = block_table[b].item<int64_t>();

        CHECK_GE(request_id, 0) << "Invalid request_id: " << request_id;
        CHECK_LT(request_id, max_num_request)
            << "request_id (" << request_id << ") >= max_num_request ("
            << max_num_request << ")";

        auto parent_beam_batch = parent_beam[b];  // [beam_size]

        for (int64_t new_beam = 0; new_beam < beam_size; ++new_beam) {
          int64_t old_beam = parent_beam_batch[new_beam].item<int64_t>();

          CHECK_GE(old_beam, 0) << "Invalid old_beam: " << old_beam;
          CHECK_LT(old_beam, beam_size)
              << "old_beam (" << old_beam << ") >= beam_size (" << beam_size
              << ")";
          if (new_beam == old_beam) {
            continue;
          }
          for (int64_t step = 0; step <= decode_step; ++step) {
            k_cache[request_id][new_beam][step].copy_(
                ori_k_cache[layer][request_id][old_beam][step]);
            v_cache[request_id][new_beam][step].copy_(
                ori_v_cache[layer][request_id][old_beam][step]);
          }
        }
      }
    }
  }
}
void cache_select_intelligent(
    const torch::Tensor& beam_index,  // [batch * beam, 1] - out_token_index
    std::vector<torch::Tensor>&
        unshared_k_cache,  // per layer: [max_num_request, beam_size,
                           // max_decode_step, kv_heads, head_dim]
    std::vector<torch::Tensor>&
        unshared_v_cache,  // per layer: [max_num_request, beam_size,
                           // max_decode_step, kv_heads, head_dim]
    const torch::Tensor& block_table,  // [batch_size, 1]
    int64_t decode_step,               // current round (step 0, 1, ...)
    int64_t beam_size,                 // beam width
    int64_t layer_num) {
  int64_t batch_size = block_table.size(0);
  int64_t total_beams = beam_index.size(0);
  CHECK_EQ(total_beams, batch_size * beam_size) << "beam_index size mismatch";

  if (layer_num > 0) {
    int64_t max_num_request = unshared_k_cache[0].size(0);
    int64_t max_decode_step = unshared_k_cache[0].size(2);

    CHECK_EQ(unshared_k_cache.size(), static_cast<size_t>(layer_num))
        << "unshared_k_cache size mismatch";
    CHECK_EQ(unshared_v_cache.size(), static_cast<size_t>(layer_num))
        << "unshared_v_cache size mismatch";
    CHECK_LT(decode_step, max_decode_step)
        << "decode_step must be less than max_decode_step";
    auto beam_index_reshaped =
        beam_index.reshape({batch_size, beam_size})
            .to(torch::kLong);  // [batch_size, beam_size]
    auto parent_beam = (beam_index_reshaped / beam_size)
                           .to(torch::kLong);  // [batch_size, beam_size]

    auto block_table_cpu =
        block_table.select(1, 0).to(torch::kCPU);  // [batch_size]
    std::vector<int64_t> dirct_beam(batch_size * beam_size, -1);
    for (int64_t b = 0; b < batch_size; ++b) {
      for (int64_t new_beam = 0; new_beam < beam_size; ++new_beam) {
        int64_t old_beam = parent_beam[b][new_beam].item<int64_t>();
        dirct_beam[b * beam_size + new_beam] = old_beam > new_beam ? 1 : -1;
      }
    }
    for (int64_t layer = 0; layer < layer_num; ++layer) {
      auto& k_cache = unshared_k_cache[layer];
      auto& v_cache = unshared_v_cache[layer];

      for (int64_t b = 0; b < batch_size; ++b) {
        int64_t request_id = block_table_cpu[b].item<int64_t>();

        CHECK_GE(request_id, 0) << "Invalid request_id: " << request_id;
        CHECK_LT(request_id, max_num_request)
            << "request_id (" << request_id << ") >= max_num_request ("
            << max_num_request << ")";

        auto parent_beam_batch = parent_beam[b];  // [beam_size]

        for (int64_t new_beam = 0; new_beam < beam_size; ++new_beam) {
          int64_t old_beam = parent_beam_batch[new_beam].item<int64_t>();

          CHECK_GE(old_beam, 0) << "Invalid old_beam: " << old_beam;
          CHECK_LT(old_beam, beam_size)
              << "old_beam (" << old_beam << ") >= beam_size (" << beam_size
              << ")";

          if (new_beam == old_beam) {
            continue;
          }
          if (dirct_beam[b * beam_size + new_beam] == 1) {
            for (int64_t step = 0; step <= decode_step; ++step) {
              k_cache[request_id][new_beam][step].copy_(
                  k_cache[request_id][old_beam][step]);
              v_cache[request_id][new_beam][step].copy_(
                  v_cache[request_id][old_beam][step]);
            }
          }
        }
        for (int64_t new_beam = beam_size - 1; new_beam > 0; --new_beam) {
          int64_t old_beam = parent_beam_batch[new_beam].item<int64_t>();
          CHECK_GE(old_beam, 0) << "Invalid old_beam: " << old_beam;
          CHECK_LT(old_beam, beam_size)
              << "old_beam (" << old_beam << ") >= beam_size (" << beam_size
              << ")";

          if (new_beam == old_beam) {
            continue;
          }
          if (dirct_beam[b * beam_size + new_beam] == -1) {
            for (int64_t step = 0; step <= decode_step; ++step) {
              k_cache[request_id][new_beam][step].copy_(
                  k_cache[request_id][old_beam][step]);
              v_cache[request_id][new_beam][step].copy_(
                  v_cache[request_id][old_beam][step]);
            }
          }
        }
      }
    }
  }
}

TEST_F(CacheSelctTest, CorrectnessTest) {
  // Small shapes are enough to catch indexing bugs, while keeping the test
  // fast.
  const int64_t batch_size = 1;
  const int64_t beam_size = 2;
  const int64_t top_k = 2;
  int32_t current_step = 1;
  const int64_t kv_heads = 8;
  const int64_t head_dim = 128;
  const int64_t layer_num = 2;
  const int64_t max_num_request = 1;
  const int64_t max_decode_step = 3;
  const auto float_opts = torch::TensorOptions().device(device_).dtype(dtype_);
  const auto int_opts =
      torch::TensorOptions().device(device_).dtype(torch::kInt32);

  torch::Tensor beam_index = torch::randint(
      0, beam_size * top_k, {batch_size * beam_size, 1}, int_opts);
  auto beam_after_sort =
      beam_index.gather(0, beam_index.argsort(static_cast<int64_t>(0), false));
  std::vector<torch::Tensor> base_k_cache;
  std::vector<torch::Tensor> base_v_cache;

  for (int64_t layer = 0; layer < layer_num; ++layer) {
    base_k_cache.push_back(torch::randn(
        {max_num_request, beam_size, max_decode_step, kv_heads, head_dim},
        float_opts));
    base_v_cache.push_back(torch::randn(
        {max_num_request, beam_size, max_decode_step, kv_heads, head_dim},
        float_opts));
  }

  std::vector<torch::Tensor> k_cache_intelligent;
  std::vector<torch::Tensor> v_cache_intelligent;
  std::vector<torch::Tensor> k_cache_normal;
  std::vector<torch::Tensor> v_cache_normal;
  k_cache_intelligent.reserve(layer_num);
  v_cache_intelligent.reserve(layer_num);
  k_cache_normal.reserve(layer_num);
  v_cache_normal.reserve(layer_num);
  for (int64_t layer = 0; layer < layer_num; ++layer) {
    k_cache_intelligent.push_back(base_k_cache[layer].clone());
    v_cache_intelligent.push_back(base_v_cache[layer].clone());
    k_cache_normal.push_back(base_k_cache[layer].clone());
    v_cache_normal.push_back(base_v_cache[layer].clone());
  }

  // CPU reference uses request-level block table: [batch_size, 1].
  torch::Tensor block_table =
      torch::arange(batch_size, int_opts).view({batch_size, 1});

  // CUDA kernel uses sequence-level block table shape [batch_size * beam_size,
  // 1].
  torch::Tensor seq_block_table =
      torch::arange(batch_size * beam_size, int_opts)
          .view({batch_size * beam_size, 1});

  //   use cache intelligent
  cache_select_intelligent(beam_after_sort,
                           k_cache_intelligent,
                           v_cache_intelligent,
                           block_table,
                           current_step,
                           beam_size,
                           layer_num);
  //   use cache normal
  cache_select(beam_after_sort,
               k_cache_normal,
               v_cache_normal,
               block_table,
               current_step,
               beam_size,
               layer_num);

  // CUDA kernel version (single launch over layer axis).
  std::vector<torch::Tensor> k_cache_cuda;
  std::vector<torch::Tensor> v_cache_cuda;
  k_cache_cuda.reserve(layer_num);
  v_cache_cuda.reserve(layer_num);
  for (int64_t layer = 0; layer < layer_num; ++layer) {
    k_cache_cuda.push_back(base_k_cache[layer].clone());
    v_cache_cuda.push_back(base_v_cache[layer].clone());
  }
  xllm::kernel::cuda::cache_select(beam_after_sort,
                                   k_cache_cuda,
                                   v_cache_cuda,
                                   seq_block_table,
                                   current_step,
                                   beam_size,
                                   layer_num);
  for (int64_t layer = 0; layer < layer_num; ++layer) {
    EXPECT_TRUE(torch::allclose(
        k_cache_intelligent[layer], k_cache_normal[layer], 1e-5, 1e-5));
    EXPECT_TRUE(torch::allclose(
        v_cache_intelligent[layer], v_cache_normal[layer], 1e-5, 1e-5));

    EXPECT_TRUE(torch::allclose(
        k_cache_intelligent[layer], k_cache_cuda[layer], 1e-5, 1e-5));
    EXPECT_TRUE(torch::allclose(
        v_cache_intelligent[layer], v_cache_cuda[layer], 1e-5, 1e-5));
  }
}

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

#include "layers/common/qwen2_vision_attention.h"

#include <glog/logging.h>
#include <gtest/gtest.h>
#include <torch/torch.h>

#include "framework/model/model_args.h"
#include "framework/model/model_input_params.h"
#include "framework/parallel_state/parallel_state.h"
#include "framework/state_dict/state_dict.h"
#include "layers/mlu/tests_utils.h"
#include "platform/device.h"

namespace xllm {
namespace layer {

class Qwen2VisionAttentionTest : public ::testing::Test {
 protected:
  void SetUp() override {
    torch::Device device(Device::type_torch(), 0);
    Device xllm_device(device);
    xllm_device.set_seed(42);
    model_args_.model_type() = "qwen2_vl";
    model_args_.mm_hidden_size() = 1280;
    model_args_.mm_head_dim() = 80;
    model_args_.mm_num_attention_heads() = 16;

    options_ = torch::TensorOptions().dtype(torch::kBFloat16).device(device);

    process_group_ = create_process_group(
        0, 1, 1, 3332, false, "localhost", "tp_group", device);
    parallel_args_.tp_group_ = process_group_.get();

    context_ = ModelContext(parallel_args_, model_args_, QuantArgs(), options_);
    InitTestWeights();
  }

  void InitTestWeights() {
    int32_t mm_hidden_size = model_args_.mm_hidden_size();
    int32_t mm_num_heads = model_args_.mm_num_attention_heads();
    int32_t mm_head_dim = model_args_.mm_head_dim();
    int32_t qkv_size = mm_num_heads * mm_head_dim * 3;

    std::unordered_map<std::string, torch::Tensor> weight_map = {
        {"qkv.weight", torch::randn({qkv_size, mm_hidden_size}, options_)},
        {"qkv.bias", torch::randn({qkv_size}, options_)},
        {"proj.weight",
         torch::randn({mm_hidden_size, mm_hidden_size}, options_)},
        {"proj.bias", torch::randn({mm_hidden_size}, options_)},
    };

    for (auto& [name, tensor] : weight_map) {
      tensor = tensor / torch::sqrt(torch::tensor(tensor.size(0), options_));
      weight_dict_["model.visual." + name] = tensor;
    }
  }

  ModelArgs model_args_;
  ModelContext context_;
  ParallelArgs parallel_args_{0, 1, nullptr};
  torch::TensorOptions options_;
  std::unordered_map<std::string, torch::Tensor> weight_dict_;
  std::unique_ptr<ProcessGroup> process_group_ = nullptr;
};

TEST_F(Qwen2VisionAttentionTest, ForwardTest) {
  auto vision_attention = Qwen2VisionAttention(context_);

  std::string prefix = "model.visual.";
  StateDict state_dict(weight_dict_, prefix);
  vision_attention->load_state_dict(state_dict.get_dict_with_prefix(prefix));

  int32_t batch_size = 2;
  int32_t seq_len = 40;
  int32_t mm_hidden_size = model_args_.mm_hidden_size();
  int32_t num_tokens = batch_size * seq_len;

  auto hidden_states =
      test::seeded_tensor("qwen2_vision_attention.hidden_states",
                          {num_tokens, mm_hidden_size},
                          torch::kBFloat16,
                          options_.device());

  // Create cu_seq_len (cumulative sequence lengths)
  std::vector<int32_t> cu_seq_len_vec = {0, seq_len, num_tokens};
  auto cu_seq_len =
      torch::tensor(cu_seq_len_vec, options_.dtype(torch::kInt32));

  // Create rotary embeddings (cos and sin)
  // Shape: (rope_seqlen, rope_dim)
  int32_t mm_head_dim = model_args_.mm_head_dim();
  int32_t rope_dim = mm_head_dim;
  auto m_cos_pos = test::seeded_tensor("qwen2_vision_attention.m_cos_pos",
                                       {num_tokens, rope_dim},
                                       torch::kBFloat16,
                                       options_.device());
  auto m_sin_pos = test::seeded_tensor("qwen2_vision_attention.m_sin_pos",
                                       {num_tokens, rope_dim},
                                       torch::kBFloat16,
                                       options_.device());

  auto output = vision_attention->forward(
      hidden_states, m_cos_pos, m_sin_pos, cu_seq_len, cu_seq_len_vec);
  xllm::Device device(options_.device());
  device.synchronize_default_stream();

  CHECK_EQ(output.sizes(), torch::IntArrayRef({num_tokens, mm_hidden_size}));

  int32_t check_count = 10;
  auto test_output = output.flatten().slice(0, 0, check_count);
  std::vector<float> expected_values = {-0.0864258f,
                                        0.365234f,
                                        0.0898438f,
                                        0.226562f,
                                        0.0834961f,
                                        -0.228516f,
                                        -0.0268555f,
                                        0.126953f,
                                        -0.371094f,
                                        0.108887f};
  test::verify_precision(test_output.unsqueeze(0), expected_values, 1e-3, 1e-3);

  auto output2 = vision_attention->forward(
      hidden_states, m_cos_pos, m_sin_pos, cu_seq_len, cu_seq_len_vec);
  device.synchronize_default_stream();
  ASSERT_TRUE(torch::allclose(output.flatten().to(torch::kFloat32),
                              output2.flatten().to(torch::kFloat32),
                              /*rtol=*/1e-4,
                              /*atol=*/1e-5));
}

}  // namespace layer
}  // namespace xllm

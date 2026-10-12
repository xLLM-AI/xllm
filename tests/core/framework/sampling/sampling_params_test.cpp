/* Copyright 2025-2026 The xLLM Authors.
Copyright 2024 The ScaleLLM Authors. All Rights Reserved.

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

#include "sampling_params.h"

#include <glog/logging.h>
#include <gtest/gtest.h>
#include <torch/torch.h>

namespace xllm {

TEST(SamplingParamsTest, NormalConcat) {
  // construct sampling_parameters_1
  RequestSamplingParam request_1, request_2;
  std::vector<int32_t> selected_token_idxes_1{11, 23};
  std::vector<int32_t> sample_idxes_1{0, 1};
  std::vector<std::vector<int64_t>> unique_token_ids_vec_1{
      std::vector<int64_t>{
          151645, 100022, 104202, 104167, 198, 77091, 872, 220, 151644},
      std::vector<int64_t>{
          151645, 100022, 104202, 104167, 198, 77091, 872, 220, 151644}};
  std::vector<std::vector<int32_t>> unique_token_counts_vec_1{
      std::vector<int32_t>{1, 1, 1, 1, 3, 1, 1, 1, 2},
      std::vector<int32_t>{1, 1, 1, 1, 3, 1, 1, 1, 2}};
  std::vector<int32_t> unique_token_lens_vec_1{9, 9};

  SamplingParameters sampling_parameters_1;
  sampling_parameters_1.init(
      std::vector<const RequestSamplingParam*>{&request_1, &request_2},
      selected_token_idxes_1,
      sample_idxes_1,
      unique_token_ids_vec_1,
      unique_token_counts_vec_1,
      unique_token_lens_vec_1);

  // construct sampling_parameters_2
  RequestSamplingParam request_3, request_4;
  std::vector<int32_t> selected_token_idxes_2{13, 28};
  std::vector<int32_t> sample_idxes_2{0, 1};
  std::vector<std::vector<int64_t>> unique_token_ids_vec_2{
      std::vector<int64_t>{151645,
                           119414,
                           100287,
                           26288,
                           101239,
                           198,
                           77091,
                           106055,
                           872,
                           220,
                           151644},
      std::vector<int64_t>{0,
                           62112,
                           9370,
                           107425,
                           151645,
                           99489,
                           106309,
                           198,
                           77091,
                           71618,
                           872,
                           220,
                           151644}};
  std::vector<std::vector<int32_t>> unique_token_counts_vec_2{
      std::vector<int32_t>{1, 1, 1, 1, 1, 3, 1, 1, 1, 1, 2},
      std::vector<int32_t>{0, 1, 1, 1, 1, 1, 1, 3, 1, 1, 1, 1, 2}};
  std::vector<int32_t> unique_token_lens_vec_2{11, 12};

  SamplingParameters sampling_parameters_2;
  sampling_parameters_2.init(
      std::vector<const RequestSamplingParam*>{&request_3, &request_4},
      selected_token_idxes_2,
      sample_idxes_2,
      unique_token_ids_vec_2,
      unique_token_counts_vec_2,
      unique_token_lens_vec_2);

  // construct expected output
  torch::Tensor result_selected_token_idxes = torch::tensor({11, 23, 37, 52});
  torch::Tensor result_sample_idxes = torch::tensor({0, 1, 2, 3});

  // execute concat
  sampling_parameters_1.concat(sampling_parameters_2);

  // check results
  EXPECT_TRUE(torch::equal(sampling_parameters_1.selected_token_idxes,
                           result_selected_token_idxes));
  EXPECT_TRUE(
      torch::equal(sampling_parameters_1.sample_idxes, result_sample_idxes));
}

TEST(SamplingParamsTest, AbnormalConcat) {
  // construct both of default sampling_parameters
  SamplingParameters sampling_parameters_1, sampling_parameters_2;

  // execute concat
  sampling_parameters_1.concat(sampling_parameters_2);

  // check results
  EXPECT_FALSE(sampling_parameters_1.selected_token_idxes.defined());
  EXPECT_FALSE(sampling_parameters_1.sample_idxes.defined());
}

TEST(SamplingParamsTest, ConcatPadsUnconstrainedRows) {
  RequestSamplingParam constrained_request;
  SamplingParameters constrained;
  constrained.init({&constrained_request},
                   {0},
                   {0},
                   {},
                   {},
                   {},
                   {torch::tensor({0.0F, -1.0e9F})});

  RequestSamplingParam unconstrained_request;
  SamplingParameters unconstrained;
  unconstrained.init({&unconstrained_request}, {1}, {0}, {}, {}, {});

  constrained.concat(unconstrained);

  ASSERT_TRUE(constrained.filter_mask.defined());
  ASSERT_EQ(constrained.filter_mask.sizes(), torch::IntArrayRef({2, 2}));
  EXPECT_EQ(constrained.filter_mask.index({0, 1}).item<float>(), -1.0e9F);
  EXPECT_EQ(constrained.filter_mask.index({1, 0}).item<float>(), 0.0F);
  EXPECT_EQ(constrained.filter_mask.index({1, 1}).item<float>(), 0.0F);
}

namespace {

// Build SamplingParameters from per-request params and return the resulting
// do_sample / all_greedy_sample / all_random_sample classification.
struct SampleClassification {
  torch::Tensor do_sample;
  bool all_greedy_sample;
  bool all_random_sample;
};

SampleClassification classify(
    const std::vector<RequestSamplingParam>& requests) {
  std::vector<const RequestSamplingParam*> req_ptrs;
  req_ptrs.reserve(requests.size());
  for (const auto& req : requests) {
    req_ptrs.push_back(&req);
  }
  std::vector<int32_t> selected_token_idxes;
  std::vector<int32_t> sample_idxes;
  selected_token_idxes.reserve(requests.size());
  sample_idxes.reserve(requests.size());
  for (int32_t i = 0; i < static_cast<int32_t>(requests.size()); ++i) {
    selected_token_idxes.push_back(i);
    sample_idxes.push_back(i);
  }
  std::vector<std::vector<int64_t>> unique_token_ids_vec;
  std::vector<std::vector<int32_t>> unique_token_counts_vec;
  std::vector<int32_t> unique_token_lens_vec;

  SamplingParameters params;
  params.init(req_ptrs,
              selected_token_idxes,
              sample_idxes,
              unique_token_ids_vec,
              unique_token_counts_vec,
              unique_token_lens_vec);
  return SampleClassification{
      params.do_sample, params.all_greedy_sample, params.all_random_sample};
}

}  // namespace

TEST(SamplingParamsTest, GreedyWhenTemperatureZeroEvenWithTopKTopP) {
  // temperature == 0 must stay greedy even when top_k / top_p are set:
  // they are argmax-invariant filters and must not route the request into
  // multinomial sampling.
  RequestSamplingParam request;
  request.temperature = 0.0;
  request.top_k = 40;
  request.top_p = 0.9;

  auto result = classify({request});

  EXPECT_TRUE(torch::equal(
      result.do_sample,
      torch::zeros({1}, torch::TensorOptions().dtype(torch::kBool))));
  EXPECT_TRUE(result.all_greedy_sample);
  EXPECT_FALSE(result.all_random_sample);
}

TEST(SamplingParamsTest, RandomWhenTemperatureNonZero) {
  // temperature > 0 still routes into random sampling.
  RequestSamplingParam request;
  request.temperature = 0.7;

  auto result = classify({request});

  EXPECT_TRUE(torch::equal(
      result.do_sample,
      torch::ones({1}, torch::TensorOptions().dtype(torch::kBool))));
  EXPECT_FALSE(result.all_greedy_sample);
  EXPECT_TRUE(result.all_random_sample);
}

TEST(SamplingParamsTest, MixedGreedyAndRandomClassification) {
  // A temperature==0 request keeps do_sample=false even when batched with a
  // temperature>0 request; the batch is mixed and per-request routing in
  // Sampler::forward handles the split.
  RequestSamplingParam greedy_request;
  greedy_request.temperature = 0.0;
  greedy_request.top_k = 40;

  RequestSamplingParam random_request;
  random_request.temperature = 1.0;

  auto result = classify({greedy_request, random_request});

  EXPECT_TRUE(
      torch::equal(result.do_sample,
                   torch::tensor({false, true},
                                 torch::TensorOptions().dtype(torch::kBool))));
  EXPECT_FALSE(result.all_greedy_sample);
  EXPECT_FALSE(result.all_random_sample);
}

}  // namespace xllm

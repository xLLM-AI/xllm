/* Copyright 2025-2026 The xLLM Authors.
Copyright 2024 The ScaleLLM Authors. All Rights Reserved.

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

namespace {
SamplingParameters make_beam_params(int32_t beam_width) {
  RequestSamplingParam request;
  request.beam_width = beam_width;
  std::vector<int32_t> selected_token_idxes{0};
  std::vector<int32_t> sample_idxes{0};
  std::vector<std::vector<int64_t>> unique_token_ids_vec{
      std::vector<int64_t>{1}};
  std::vector<std::vector<int32_t>> unique_token_counts_vec{
      std::vector<int32_t>{1}};
  std::vector<int32_t> unique_token_lens_vec{1};

  SamplingParameters params;
  params.init(std::vector<const RequestSamplingParam*>{&request},
              selected_token_idxes,
              sample_idxes,
              unique_token_ids_vec,
              unique_token_counts_vec,
              unique_token_lens_vec);
  return params;
}
}  // namespace

TEST(SamplingParamsTest, InitAggregatesBeamWidth) {
  SamplingParameters params = make_beam_params(/*beam_width=*/4);
  EXPECT_EQ(params.beam_width, 4);
  EXPECT_TRUE(params.use_beam_search);
}

TEST(SamplingParamsTest, InitDefaultBeamWidthIsZero) {
  SamplingParameters params = make_beam_params(/*beam_width=*/0);
  EXPECT_EQ(params.beam_width, 0);
  EXPECT_FALSE(params.use_beam_search);
}

TEST(SamplingParamsTest, ToCarriesBeamWidth) {
  SamplingParameters params = make_beam_params(/*beam_width=*/3);
  SamplingParameters device_params = params.to(torch::kCPU, torch::kFloat32);
  EXPECT_EQ(device_params.beam_width, 3);
  EXPECT_TRUE(device_params.use_beam_search);
}

TEST(SamplingParamsTest, ConcatConsistentBeamWidth) {
  SamplingParameters params_1 = make_beam_params(/*beam_width=*/4);
  SamplingParameters params_2 = make_beam_params(/*beam_width=*/4);
  params_1.concat(params_2);
  EXPECT_EQ(params_1.beam_width, 4);
}

TEST(SamplingParamsTest, ConcatCarriesBeamWidthWhenOneSideZero) {
  SamplingParameters params_1 = make_beam_params(/*beam_width=*/0);
  SamplingParameters params_2 = make_beam_params(/*beam_width=*/4);
  params_1.concat(params_2);
  EXPECT_EQ(params_1.beam_width, 4);
}

TEST(SamplingParamsDeathTest, ConcatInconsistentBeamWidth) {
  SamplingParameters params_1 = make_beam_params(/*beam_width=*/4);
  SamplingParameters params_2 = make_beam_params(/*beam_width=*/2);
  EXPECT_DEATH(params_1.concat(params_2), "beam_width");
}

}  // namespace xllm

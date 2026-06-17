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
#pragma once

#include <map>
#include <string>
#include <utility>
#include <vector>

namespace xllm {
namespace layer {

namespace glm4_decoder_constants {
enum DecoderLayerTensorId : int {
  IN_NORM_WEIGHT = 0,      // weight
  IN_NORM_BIAS = 1,        // bias
  IN_NORM_NEW_WEIGHT = 2,  // new weight
  IN_NORM_NEW_BIAS = 3,    // new bias

  IN_Q_WEIGHT = 4,    // weight
  IN_Q_BIAS = 5,      // bias
  IN_Q_DEQSCALE = 6,  // deq_scale
  IN_Q_OFFSET = 7,    // offset
  IN_Q_SCALE = 8,     // scale
  IN_Q_COMPRESS_IDX = 9,

  IN_K_WEIGHT = 10,    // weight
  IN_K_BIAS = 11,      // bias
  IN_K_DEQSCALE = 12,  // deq_scale
  IN_K_OFFSET = 13,    // offset
  IN_K_SCALE = 14,     // scale
  IN_K_COMPRESS_IDX = 15,

  IN_V_WEIGHT = 16,    // weight
  IN_V_BIAS = 17,      // bias
  IN_V_DEQSCALE = 18,  // deq_scale
  IN_V_OFFSET = 19,    // offset
  IN_V_SCALE = 20,     // scale
  IN_V_COMPRESS_IDX = 21,

  IN_ATTENTION_OUT_WEIGHT = 22,    // weight
  IN_ATTENTION_OUT_BIAS = 23,      // bias
  IN_ATTENTION_OUT_DEQSCALE = 24,  // deq_scale
  IN_ATTENTION_OUT_OFFSET = 25,    // offset
  IN_ATTENTION_OUT_SCALE = 26,     // scale
  IN_ATTENTION_OUT_COMPRESS_IDX = 27,

  IN_SELFOUT_NORM_WEIGHT = 28,      // weight
  IN_SELFOUT_NORM_BIAS = 29,        // bias
  IN_SELFOUT_NORM_NEW_WEIGHT = 30,  // new weight
  IN_SELFOUT_NORM_NEW_BIAS = 31,    // new bias

  IN_MLP_GATEUP_WEIGHT = 32,    // weight
  IN_MLP_GATEUP_BIAS = 33,      // bias
  IN_MLP_GATEUP_DEQSCALE = 34,  // deq_scale
  IN_MLP_GATEUP_OFFSET = 35,    // offset
  IN_MLP_GATEUP_SCALE = 36,     // scale
  IN_MLP_GATEUP_COMPRESS_IDX = 37,

  IN_MLP_W1_WEIGHT = 38,    // weight
  IN_MLP_W1_BIAS = 39,      // bias
  IN_MLP_W1_DEQSCALE = 40,  // deq_scale
  IN_MLP_W1_OFFSET = 41,    // offset
  IN_MLP_W1_SCALE = 42,     // scale
  IN_MLP_W1_COMPRESS_IDX = 43,

  IN_MLP_CPROJ_WEIGHT = 44,    // weight
  IN_MLP_CPROJ_BIAS = 45,      // bias
  IN_MLP_CPROJ_DEQSCALE = 46,  // deq_scale
  IN_MLP_CPROJ_OFFSET = 47,    // offset
  IN_MLP_CPROJ_SCALE = 48,     // scale
  IN_MLP_CPROJ_COMPRESS_IDX = 49,

  IN_SELFIN_NORM_WEIGHT = 50,
  IN_MLPOUT_NORM_WEIGHT = 51
};

static std::vector<std::pair<int, std::string>> WEIGHT_MAPPING = {
    {IN_NORM_WEIGHT, "input_layernorm.weight"},
    {IN_Q_WEIGHT, "self_attn.q_proj.weight"},
    {IN_Q_BIAS, "self_attn.q_proj.bias"},
    {IN_K_WEIGHT, "self_attn.k_proj.weight"},
    {IN_K_BIAS, "self_attn.k_proj.bias"},
    {IN_V_WEIGHT, "self_attn.v_proj.weight"},
    {IN_V_BIAS, "self_attn.v_proj.bias"},
    {IN_ATTENTION_OUT_WEIGHT, "self_attn.o_proj.weight"},
    {IN_SELFOUT_NORM_WEIGHT, "post_attention_layernorm.weight"},
    {IN_MLP_GATEUP_WEIGHT, "mlp.gate_up_proj.weight"},
    {IN_MLP_CPROJ_WEIGHT, "mlp.down_proj.weight"},
    {IN_SELFIN_NORM_WEIGHT, "post_self_attn_layernorm.weight"},
    {IN_MLPOUT_NORM_WEIGHT, "post_mlp_layernorm.weight"}};

static std::map<int, int> WEIGHT_SHARD = {{IN_Q_WEIGHT, 0},
                                          {IN_Q_BIAS, 0},
                                          {IN_K_WEIGHT, 0},
                                          {IN_K_BIAS, 0},
                                          {IN_V_WEIGHT, 0},
                                          {IN_V_BIAS, 0},
                                          {IN_ATTENTION_OUT_WEIGHT, 1},
                                          {IN_MLP_CPROJ_WEIGHT, 1}};

}  // namespace glm4_decoder_constants

}  // namespace layer
}  // namespace xllm

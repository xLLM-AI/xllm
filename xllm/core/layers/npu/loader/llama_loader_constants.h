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
#pragma once

#include <map>
#include <string>
#include <unordered_map>
#include <vector>

namespace xllm {
namespace layer {

namespace llama_decoder_constants {
enum DecoderLayerTensorId : int {

  IN_NORM_WEIGHT = 0,  // weight
  IN_NORM_BIAS,        // bias
  IN_NORM_NEW_WEIGHT,  // new weight
  IN_NORM_NEW_BIAS,    // new bias

  IN_Q_WEIGHT,    // weight
  IN_Q_BIAS,      // bias
  IN_Q_DEQSCALE,  // deq_scale
  IN_Q_OFFSET,    // offset
  IN_Q_SCALE,     // scale
  IN_Q_COMPRESS_IDX,

  IN_K_WEIGHT,    // weight
  IN_K_BIAS,      // bias
  IN_K_DEQSCALE,  // deq_scale
  IN_K_OFFSET,    // offset
  IN_K_SCALE,     // scale
  IN_K_COMPRESS_IDX,

  IN_V_WEIGHT,    // weight
  IN_V_BIAS,      // bias
  IN_V_DEQSCALE,  // deq_scale
  IN_V_OFFSET,    // offset
  IN_V_SCALE,     // scale
  IN_V_COMPRESS_IDX,

  IN_ATTENTION_OUT_WEIGHT,    // weight
  IN_ATTENTION_OUT_BIAS,      // bias
  IN_ATTENTION_OUT_DEQSCALE,  // deq_scale
  IN_ATTENTION_OUT_OFFSET,    // offset
  IN_ATTENTION_OUT_SCALE,     // scale
  IN_ATTENTION_OUT_COMPRESS_IDX,

  IN_SELFOUT_NORM_WEIGHT,      // weight
  IN_SELFOUT_NORM_BIAS,        // bias
  IN_SELFOUT_NORM_NEW_WEIGHT,  // new weight
  IN_SELFOUT_NORM_NEW_BIAS,    // new bias

  IN_MLP_W2_WEIGHT,    // weight
  IN_MLP_W2_BIAS,      // bias
  IN_MLP_W2_DEQSCALE,  // deq_scale
  IN_MLP_W2_OFFSET,    // offset
  IN_MLP_W2_SCALE,     // scale
  IN_MLP_W2_COMPRESS_IDX,

  IN_MLP_W1_WEIGHT,    // weight
  IN_MLP_W1_BIAS,      // bias
  IN_MLP_W1_DEQSCALE,  // deq_scale
  IN_MLP_W1_OFFSET,    // offset
  IN_MLP_W1_SCALE,     // scale
  IN_MLP_W1_COMPRESS_IDX,

  IN_MLP_CPROJ_WEIGHT,    // weight
  IN_MLP_CPROJ_BIAS,      // bias
  IN_MLP_CPROJ_DEQSCALE,  // deq_scale
  IN_MLP_CPROJ_OFFSET,    // offset
  IN_MLP_CPROJ_SCALE,     // scale
  IN_MLP_CPROJ_COMPRESS_IDX,
};

static const std::unordered_map<std::string, int> WEIGHT_MAPPING = {
    {"input_layernorm.weight", IN_NORM_WEIGHT},
    {"self_attn.q_proj.weight", IN_Q_WEIGHT},
    {"self_attn.k_proj.weight", IN_K_WEIGHT},
    {"self_attn.v_proj.weight", IN_V_WEIGHT},
    {"self_attn.o_proj.weight", IN_ATTENTION_OUT_WEIGHT},
    {"post_attention_layernorm.weight", IN_SELFOUT_NORM_WEIGHT},
    {"mlp.gate_proj.weight", IN_MLP_W2_WEIGHT},
    {"mlp.up_proj.weight", IN_MLP_W1_WEIGHT},
    {"mlp.down_proj.weight", IN_MLP_CPROJ_WEIGHT},
};

static std::map<int, int> WEIGHT_SHARD = {{IN_Q_WEIGHT, 0},
                                          {IN_K_WEIGHT, 0},
                                          {IN_V_WEIGHT, 0},
                                          {IN_ATTENTION_OUT_WEIGHT, 1},
                                          {IN_MLP_W2_WEIGHT, 0},
                                          {IN_MLP_W1_WEIGHT, 0},
                                          {IN_MLP_CPROJ_WEIGHT, 1}};

}  // namespace llama_decoder_constants

}  // namespace layer
}  // namespace xllm

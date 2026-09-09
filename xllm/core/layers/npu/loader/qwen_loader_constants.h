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
#include <utility>
#include <vector>

namespace xllm {
namespace layer {

namespace qwen2_decoder_constants {
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

  IN_MLP_W2_WEIGHT = 32,    // weight
  IN_MLP_W2_BIAS = 33,      // bias
  IN_MLP_W2_DEQSCALE = 34,  // deq_scale
  IN_MLP_W2_OFFSET = 35,    // offset
  IN_MLP_W2_SCALE = 36,     // scale
  IN_MLP_W2_COMPRESS_IDX = 37,

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
    {IN_MLP_W2_WEIGHT, "mlp.gate_proj.weight"},
    {IN_MLP_W1_WEIGHT, "mlp.up_proj.weight"},
    {IN_MLP_CPROJ_WEIGHT, "mlp.down_proj.weight"}};

static std::vector<std::pair<int, std::string>> WEIGHT_MAPPING_W8A8 = {
    {IN_NORM_WEIGHT, "input_layernorm.weight"},
    {IN_Q_WEIGHT, "self_attn.q_proj.weight"},
    {IN_Q_BIAS, "self_attn.q_proj.quant_bias"},
    {IN_Q_DEQSCALE, "self_attn.q_proj.deq_scale"},
    {IN_Q_OFFSET, "self_attn.q_proj.input_offset"},
    {IN_Q_SCALE, "self_attn.q_proj.input_scale"},
    {IN_K_WEIGHT, "self_attn.k_proj.weight"},
    {IN_K_BIAS, "self_attn.k_proj.quant_bias"},
    {IN_K_DEQSCALE, "self_attn.k_proj.deq_scale"},
    {IN_K_OFFSET, "self_attn.k_proj.input_offset"},
    {IN_K_SCALE, "self_attn.k_proj.input_scale"},
    {IN_V_WEIGHT, "self_attn.v_proj.weight"},
    {IN_V_BIAS, "self_attn.v_proj.quant_bias"},
    {IN_V_DEQSCALE, "self_attn.v_proj.deq_scale"},
    {IN_V_OFFSET, "self_attn.v_proj.input_offset"},
    {IN_V_SCALE, "self_attn.v_proj.input_scale"},
    {IN_ATTENTION_OUT_WEIGHT, "self_attn.o_proj.weight"},
    {IN_ATTENTION_OUT_BIAS, "self_attn.o_proj.quant_bias"},
    {IN_ATTENTION_OUT_DEQSCALE, "self_attn.o_proj.deq_scale"},
    {IN_ATTENTION_OUT_OFFSET, "self_attn.o_proj.input_offset"},
    {IN_ATTENTION_OUT_SCALE, "self_attn.o_proj.input_scale"},
    {IN_SELFOUT_NORM_WEIGHT, "post_attention_layernorm.weight"},
    {IN_MLP_W2_WEIGHT, "mlp.gate_proj.weight"},
    {IN_MLP_W2_BIAS, "mlp.gate_proj.quant_bias"},
    {IN_MLP_W2_DEQSCALE, "mlp.gate_proj.deq_scale"},
    {IN_MLP_W2_OFFSET, "mlp.gate_proj.input_offset"},
    {IN_MLP_W2_SCALE, "mlp.gate_proj.input_scale"},
    {IN_MLP_W1_WEIGHT, "mlp.up_proj.weight"},
    {IN_MLP_W1_BIAS, "mlp.up_proj.quant_bias"},
    {IN_MLP_W1_DEQSCALE, "mlp.up_proj.deq_scale"},
    {IN_MLP_W1_OFFSET, "mlp.up_proj.input_offset"},
    {IN_MLP_W1_SCALE, "mlp.up_proj.input_scale"},
    {IN_MLP_CPROJ_WEIGHT, "mlp.down_proj.weight"}};

static std::map<int, int> WEIGHT_SHARD = {{IN_Q_WEIGHT, 0},
                                          {IN_Q_BIAS, 0},
                                          {IN_K_WEIGHT, 0},
                                          {IN_K_BIAS, 0},
                                          {IN_V_WEIGHT, 0},
                                          {IN_V_BIAS, 0},
                                          {IN_ATTENTION_OUT_WEIGHT, 1},
                                          {IN_MLP_W2_WEIGHT, 0},
                                          {IN_MLP_W1_WEIGHT, 0},
                                          {IN_MLP_CPROJ_WEIGHT, 1}};

static std::map<int, int> WEIGHT_SHARD_W8A8 = {{IN_Q_WEIGHT, 0},
                                               {IN_Q_BIAS, 0},
                                               {IN_Q_DEQSCALE, 0},
                                               {IN_K_WEIGHT, 0},
                                               {IN_K_BIAS, 0},
                                               {IN_K_DEQSCALE, 0},
                                               {IN_V_WEIGHT, 0},
                                               {IN_V_BIAS, 0},
                                               {IN_V_DEQSCALE, 0},
                                               {IN_ATTENTION_OUT_WEIGHT, 1},
                                               {IN_MLP_W2_WEIGHT, 0},
                                               {IN_MLP_W2_BIAS, 0},
                                               {IN_MLP_W2_DEQSCALE, 0},
                                               {IN_MLP_W1_WEIGHT, 0},
                                               {IN_MLP_W1_BIAS, 0},
                                               {IN_MLP_W1_DEQSCALE, 0},
                                               {IN_MLP_CPROJ_WEIGHT, 1}};

}  // namespace qwen2_decoder_constants

namespace qwen3_decoder_constants {
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

  IN_MLP_W2_WEIGHT = 32,    // weight
  IN_MLP_W2_BIAS = 33,      // bias
  IN_MLP_W2_DEQSCALE = 34,  // deq_scale
  IN_MLP_W2_OFFSET = 35,    // offset
  IN_MLP_W2_SCALE = 36,     // scale
  IN_MLP_W2_COMPRESS_IDX = 37,

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

  IN_QKV_SCALE_FILL = 50,
  IN_QKV_OFFSET_FILL = 51,
  IN_MLP_SCALE_FILL = 52,
  IN_MLP_OFFSET_FILL = 53,
  Q_NORM_WEIGHT = 54,
  K_NORM_WEIGHT = 55,
};

static std::vector<std::pair<int, std::string>> WEIGHT_MAPPING = {
    {IN_NORM_WEIGHT, "input_layernorm.weight"},
    {IN_Q_WEIGHT, "self_attn.q_proj.weight"},
    {IN_K_WEIGHT, "self_attn.k_proj.weight"},
    {IN_V_WEIGHT, "self_attn.v_proj.weight"},
    {IN_ATTENTION_OUT_WEIGHT, "self_attn.o_proj.weight"},
    {IN_SELFOUT_NORM_WEIGHT, "post_attention_layernorm.weight"},
    {IN_MLP_W2_WEIGHT, "mlp.gate_proj.weight"},
    {IN_MLP_W1_WEIGHT, "mlp.up_proj.weight"},
    {IN_MLP_CPROJ_WEIGHT, "mlp.down_proj.weight"},
    {Q_NORM_WEIGHT, "self_attn.q_norm.weight"},
    {K_NORM_WEIGHT, "self_attn.k_norm.weight"}};

static std::vector<std::pair<int, std::string>> WEIGHT_MAPPING_W8A8 = {
    {IN_NORM_WEIGHT, "input_layernorm.weight"},
    {IN_Q_WEIGHT, "self_attn.q_proj.weight"},
    {IN_Q_BIAS, "self_attn.q_proj.quant_bias"},
    {IN_Q_DEQSCALE, "self_attn.q_proj.deq_scale"},
    {IN_Q_OFFSET, "self_attn.q_proj.input_offset"},
    {IN_Q_SCALE, "self_attn.q_proj.input_scale"},
    {IN_K_WEIGHT, "self_attn.k_proj.weight"},
    {IN_K_BIAS, "self_attn.k_proj.quant_bias"},
    {IN_K_DEQSCALE, "self_attn.k_proj.deq_scale"},
    {IN_K_OFFSET, "self_attn.k_proj.input_offset"},
    {IN_K_SCALE, "self_attn.k_proj.input_scale"},
    {IN_V_WEIGHT, "self_attn.v_proj.weight"},
    {IN_V_BIAS, "self_attn.v_proj.quant_bias"},
    {IN_V_DEQSCALE, "self_attn.v_proj.deq_scale"},
    {IN_V_OFFSET, "self_attn.v_proj.input_offset"},
    {IN_V_SCALE, "self_attn.v_proj.input_scale"},
    {IN_ATTENTION_OUT_WEIGHT, "self_attn.o_proj.weight"},
    {IN_ATTENTION_OUT_BIAS, "self_attn.o_proj.quant_bias"},
    {IN_ATTENTION_OUT_DEQSCALE, "self_attn.o_proj.deq_scale"},
    {IN_ATTENTION_OUT_OFFSET, "self_attn.o_proj.input_offset"},
    {IN_ATTENTION_OUT_SCALE, "self_attn.o_proj.input_scale"},
    {IN_SELFOUT_NORM_WEIGHT, "post_attention_layernorm.weight"},
    {IN_MLP_W2_WEIGHT, "mlp.gate_proj.weight"},
    {IN_MLP_W2_BIAS, "mlp.gate_proj.quant_bias"},
    {IN_MLP_W2_DEQSCALE, "mlp.gate_proj.deq_scale"},
    {IN_MLP_W2_OFFSET, "mlp.gate_proj.input_offset"},
    {IN_MLP_W2_SCALE, "mlp.gate_proj.input_scale"},
    {IN_MLP_W1_WEIGHT, "mlp.up_proj.weight"},
    {IN_MLP_W1_BIAS, "mlp.up_proj.quant_bias"},
    {IN_MLP_W1_DEQSCALE, "mlp.up_proj.deq_scale"},
    {IN_MLP_W1_OFFSET, "mlp.up_proj.input_offset"},
    {IN_MLP_W1_SCALE, "mlp.up_proj.input_scale"},
    {IN_MLP_CPROJ_WEIGHT, "mlp.down_proj.weight"},
    {IN_MLP_CPROJ_BIAS, "mlp.down_proj.quant_bias"},
    {IN_MLP_CPROJ_DEQSCALE, "mlp.down_proj.deq_scale"},
    {IN_MLP_CPROJ_OFFSET, "mlp.down_proj.input_offset"},
    {IN_MLP_CPROJ_SCALE, "mlp.down_proj.input_scale"},
    {Q_NORM_WEIGHT, "self_attn.q_norm.weight"},
    {K_NORM_WEIGHT, "self_attn.k_norm.weight"}};

static std::map<int, int> WEIGHT_SHARD = {{IN_Q_WEIGHT, 0},
                                          {IN_K_WEIGHT, 0},
                                          {IN_V_WEIGHT, 0},
                                          {IN_ATTENTION_OUT_WEIGHT, 1},
                                          {IN_MLP_W2_WEIGHT, 0},
                                          {IN_MLP_W1_WEIGHT, 0},
                                          {IN_MLP_CPROJ_WEIGHT, 1}};

static std::map<int, int> WEIGHT_SHARD_W8A8 = {{IN_Q_WEIGHT, 0},
                                               {IN_Q_BIAS, 0},
                                               {IN_Q_DEQSCALE, 0},
                                               {IN_K_WEIGHT, 0},
                                               {IN_K_BIAS, 0},
                                               {IN_K_DEQSCALE, 0},
                                               {IN_V_WEIGHT, 0},
                                               {IN_V_BIAS, 0},
                                               {IN_V_DEQSCALE, 0},
                                               {IN_ATTENTION_OUT_WEIGHT, 1},
                                               {IN_MLP_W2_WEIGHT, 0},
                                               {IN_MLP_W2_BIAS, 0},
                                               {IN_MLP_W2_DEQSCALE, 0},
                                               {IN_MLP_W1_WEIGHT, 0},
                                               {IN_MLP_W1_BIAS, 0},
                                               {IN_MLP_W1_DEQSCALE, 0},
                                               {IN_MLP_CPROJ_WEIGHT, 1}};

}  // namespace qwen3_decoder_constants

namespace qwen3_moe_decoder_constants {
enum DecoderLayerTensorId : int {
  IN_INPUT_NORM_WEIGHT = 0,  // [2048]
  IN_INPUT_NORM_BIAS = 1,
  IN_INPUT_NORM_NEW_WEIGHT = 2,
  IN_INPUT_NORM_NEW_BIAS = 3,

  IN_QKV_WEIGHT_0 = 4,  // [4096, 2048]
  IN_QKV_BIAS_0 = 5,
  IN_QKV_DESCALE_0 = 6,
  IN_QKV_OFFSET_0 = 7,
  IN_QKV_SCALE_0 = 8,
  IN_QKV_COMPRESS_IDX_0 = 9,

  IN_QKV_WEIGHT_1 = 10,  // [512, 2048]
  IN_QKV_BIAS_1 = 11,
  IN_QKV_DESCALE_1 = 12,
  IN_QKV_OFFSET_1 = 13,
  IN_QKV_SCALE_1 = 14,
  IN_QKV_COMPRESS_IDX_1 = 15,

  IN_QKV_WEIGHT_2 = 16,  // [512, 2048]
  IN_QKV_BIAS_2 = 17,
  IN_QKV_DESCALE_2 = 18,
  IN_QKV_OFFSET_2 = 19,
  IN_QKV_SCALE_2 = 20,
  IN_QKV_COMPRESS_IDX_2 = 21,

  IN_ATTENTION_OUT_WEIGHT = 22,  // [2048, 4096]
  IN_ATTENTION_OUT_BIAS = 23,
  IN_ATTENTION_OUT_DESCALE = 24,
  IN_ATTENTION_OUT_OFFSET = 25,
  IN_ATTENTION_OUT_SCALE = 26,
  IN_ATTENTION_OUT_COMPRESS_IDX = 27,

  IN_Q_NORM_WEIGHT = 28,  // [128]
  IN_K_NORM_WEIGHT = 29,  // [128]

  IN_SELFATTENTION_OUT_NORM_WEIGHT = 30,  // [2048]
  IN_SELFATTENTION_OUT_NORM_BIAS = 31,
  IN_SELFATTENTION_OUT_NEW_NORM_WEIGHT = 32,
  IN_SELFATTENTION_OUT_NEW_NORM_BIAS = 33,

  IN_BLOCK_SPARSE_MOE_GATE_WEIGHT = 34,  // [128, 2048]
  IN_BLOCK_SPARSE_MOE_GATE_BIAS = 35,
  IN_BLOCK_SPARSE_MOE_GATE_DESCALE = 36,
  IN_BLOCK_SPARSE_MOE_GATE_OFFSET = 37,
  IN_BLOCK_SPARSE_MOE_GATE_SCALE = 38,
  IN_BLOCK_SPARSE_MOE_GATE_COMPRESS_IDX = 39,

  IN_MLP_GATEUP_WEIGHT_EXPERT = 40,
  IN_MLP_GATEUP_BIAS_EXPERT = 41,
  IN_MLP_GATEUP_DESCALE_EXPERT = 42,
  IN_MLP_GATEUP_OFFSET_EXPERT = 43,
  IN_MLP_GATEUP_SCALE_EXPERT = 44,
  IN_MLP_GATEUP_COMPRESS_IDX_EXPERT = 45,

  IN_MLP_DOWN_WEIGHT_EXPERT = 46,  // [2048, 768]
  IN_MLP_DOWN_BIAS_EXPERT = 47,
  IN_MLP_DOWN_DESCALE_EXPERT = 48,
  IN_MLP_DOWN_OFFSET_EXPERT = 49,
  IN_MLP_DOWN_SCALE_EXPERT = 50,
  IN_MLP_DOWN_COMPRESS_IDX_EXPERT = 51,

  IN_MLP_SHARED_GATEUP_WEIGHT = 52,
  IN_MLP_SHARED_DOWN_WEIGHT = 53,
  IN_MLP_SHARED_EXPERT_GATE = 54,
};

static const std::unordered_map<std::string, int> WEIGHT_MAPPING = {
    {"input_layernorm.weight", IN_INPUT_NORM_WEIGHT},

    {"self_attn.q_proj.weight", IN_QKV_WEIGHT_0},

    {"self_attn.k_proj.weight", IN_QKV_WEIGHT_1},

    {"self_attn.v_proj.weight", IN_QKV_WEIGHT_2},

    {"self_attn.o_proj.weight", IN_ATTENTION_OUT_WEIGHT},

    {"self_attn.q_norm.weight", IN_Q_NORM_WEIGHT},
    {"self_attn.k_norm.weight", IN_K_NORM_WEIGHT},

    {"post_attention_layernorm.weight", IN_SELFATTENTION_OUT_NORM_WEIGHT},

    // MoE Gate
    {"mlp.gate.weight", IN_BLOCK_SPARSE_MOE_GATE_WEIGHT},

    // Expert MLP - Gate/Up projections
    {"gate_proj.weight", IN_MLP_GATEUP_WEIGHT_EXPERT},

    {"up_proj.weight", IN_MLP_GATEUP_WEIGHT_EXPERT},

    // Expert MLP - Down projection
    {"down_proj.weight", IN_MLP_DOWN_WEIGHT_EXPERT},

};

static const std::unordered_map<std::string, int> WEIGHT_MAPPING_W8A8 = {
    {"input_layernorm.weight", IN_INPUT_NORM_WEIGHT},
    {"input_layernorm.bias", IN_INPUT_NORM_NEW_BIAS},

    {"self_attn.q_proj.weight", IN_QKV_WEIGHT_0},
    {"self_attn.q_proj.bias", IN_QKV_BIAS_0},
    {"self_attn.q_proj.deq_scale", IN_QKV_DESCALE_0},
    {"self_attn.q_proj.weight_offset", IN_QKV_OFFSET_0},
    {"self_attn.q_proj.weight_scale", IN_QKV_SCALE_0},

    {"self_attn.k_proj.weight", IN_QKV_WEIGHT_1},
    {"self_attn.k_proj.bias", IN_QKV_BIAS_1},
    {"self_attn.k_proj.deq_scale", IN_QKV_DESCALE_1},
    {"self_attn.k_proj.weight_offset", IN_QKV_OFFSET_1},
    {"self_attn.k_proj.weight_scale", IN_QKV_SCALE_1},

    {"self_attn.v_proj.weight", IN_QKV_WEIGHT_2},
    {"self_attn.v_proj.bias", IN_QKV_BIAS_2},
    {"self_attn.v_proj.deq_scale", IN_QKV_DESCALE_2},
    {"self_attn.v_proj.weight_offset", IN_QKV_OFFSET_2},
    {"self_attn.v_proj.weight_scale", IN_QKV_SCALE_2},

    {"self_attn.o_proj.weight", IN_ATTENTION_OUT_WEIGHT},
    {"self_attn.o_proj.quant_bias", IN_ATTENTION_OUT_BIAS},
    {"self_attn.o_proj.deq_scale", IN_ATTENTION_OUT_DESCALE},
    {"self_attn.o_proj.weight_offset", IN_ATTENTION_OUT_OFFSET},
    {"self_attn.o_proj.weight_scale", IN_ATTENTION_OUT_SCALE},

    {"self_attn.q_norm.weight", IN_Q_NORM_WEIGHT},
    {"self_attn.k_norm.weight", IN_K_NORM_WEIGHT},

    {"post_attention_layernorm.weight", IN_SELFATTENTION_OUT_NORM_WEIGHT},
    {"post_attention_layernorm.bias", IN_SELFATTENTION_OUT_NEW_NORM_BIAS},

    // MoE Gate
    {"mlp.gate.weight", IN_BLOCK_SPARSE_MOE_GATE_WEIGHT},

    {"gate_proj.weight", IN_MLP_GATEUP_WEIGHT_EXPERT},
    {"gate_proj.weight_offset", IN_MLP_GATEUP_OFFSET_EXPERT},
    {"gate_proj.weight_scale", IN_MLP_GATEUP_SCALE_EXPERT},
    {"up_proj.weight", IN_MLP_GATEUP_WEIGHT_EXPERT},
    {"up_proj.weight_offset", IN_MLP_GATEUP_OFFSET_EXPERT},
    {"up_proj.weight_scale", IN_MLP_GATEUP_SCALE_EXPERT},

    {"down_proj.weight", IN_MLP_DOWN_WEIGHT_EXPERT},
    {"down_proj.weight_offset", IN_MLP_DOWN_OFFSET_EXPERT},
    {"down_proj.weight_scale", IN_MLP_DOWN_SCALE_EXPERT},
};

static const std::unordered_map<std::string, std::vector<int>>
    SPECIAL_MULTI_ASSIGN_W8A8 = {
        {"input_layernorm.weight",
         {IN_INPUT_NORM_WEIGHT, IN_INPUT_NORM_NEW_WEIGHT}},
        {"post_attention_layernorm.weight",
         {IN_SELFATTENTION_OUT_NORM_WEIGHT,
          IN_SELFATTENTION_OUT_NEW_NORM_WEIGHT}},
};

static const std::map<int, int> WEIGHT_SHARD = {
    {IN_QKV_WEIGHT_0, 0},
    {IN_QKV_WEIGHT_1, 0},
    {IN_QKV_WEIGHT_2, 0},
    {IN_ATTENTION_OUT_WEIGHT, 1},
    {IN_MLP_GATEUP_WEIGHT_EXPERT, 0},
    {IN_MLP_DOWN_WEIGHT_EXPERT, 1},
};

static const std::map<int, int> WEIGHT_SHARD_W8A8 = {
    {IN_QKV_WEIGHT_0, 0},
    {IN_QKV_OFFSET_0, 0},
    {IN_QKV_SCALE_0, 0},
    {IN_QKV_WEIGHT_1, 0},
    {IN_QKV_OFFSET_1, 0},
    {IN_QKV_SCALE_1, 0},
    {IN_QKV_WEIGHT_2, 0},
    {IN_QKV_OFFSET_2, 0},
    {IN_QKV_SCALE_2, 0},
    {IN_ATTENTION_OUT_WEIGHT, 1},
    {IN_MLP_GATEUP_WEIGHT_EXPERT, 0},
    {IN_MLP_GATEUP_OFFSET_EXPERT, 0},
    {IN_MLP_GATEUP_SCALE_EXPERT, 0},
    {IN_MLP_DOWN_WEIGHT_EXPERT, 1},
};

}  // namespace qwen3_moe_decoder_constants

namespace qwen2_vision_encoder_constants {
enum VisionEncoderLayerTensorId : int {
  IN_INPUT_NORM_WEIGHT = 0,
  IN_INPUT_NORM_BIAS,
  IN_POST_NORM_WEIGHT,
  IN_POST_NORM_BIAS,
  IN_QKV_WEIGHT,
  IN_QKV_BIAS,
  IN_WATTENTION_OUT_WEIGHT,
  IN_WATTENTION_OUT_BIAS,
  IN_LINEAR_FC1_WEIGHT,
  IN_LINEAR_FC1_BIAS,
  IN_LINEAR_FC2_WEIGHT,
  IN_LINEAR_FC2_BIAS,
  IN_VISION_Q_WEIGHT,
  IN_VISION_Q_BIAS,
  IN_VISION_K_WEIGHT,
  IN_VISION_K_BIAS,
  IN_VISION_V_WEIGHT,
  IN_VISION_V_BIAS
};

static std::vector<std::pair<int, std::string>> WEIGHT_MAPPING = {
    {IN_INPUT_NORM_WEIGHT, "norm1.weight"},
    {IN_INPUT_NORM_BIAS, "norm1.bias"},
    {IN_POST_NORM_WEIGHT, "norm2.weight"},
    {IN_POST_NORM_BIAS, "norm2.bias"},
    {IN_QKV_WEIGHT, "attn.qkv.weight"},
    {IN_QKV_BIAS, "attn.qkv.bias"},
    {IN_WATTENTION_OUT_WEIGHT, "attn.proj.weight"},
    {IN_WATTENTION_OUT_BIAS, "attn.proj.bias"},
    {IN_LINEAR_FC1_WEIGHT, "mlp.fc1.weight"},
    {IN_LINEAR_FC1_BIAS, "mlp.fc1.bias"},
    {IN_LINEAR_FC2_WEIGHT, "mlp.fc2.weight"},
    {IN_LINEAR_FC2_BIAS, "mlp.fc2.bias"}};

// {weight,dim}
static std::map<int, int> WEIGHT_SHARD = {
    {IN_WATTENTION_OUT_WEIGHT, 1},
    {IN_LINEAR_FC1_WEIGHT, 0},
    {IN_LINEAR_FC1_BIAS, 0},
    {IN_LINEAR_FC2_WEIGHT, 1},
};

}  // namespace qwen2_vision_encoder_constants

namespace qwen2dot5_vision_encoder_constants {
enum VisionEncoderLayerTensorId : int {
  IN_INPUT_NORM_WEIGHT = 0,
  IN_POST_NORM_WEIGHT,
  IN_QKV_WEIGHT,
  IN_QKV_BIAS,
  IN_WATTENTION_OUT_WEIGHT,
  IN_WATTENTION_OUT_BIAS,
  IN_MLP_GATE_WEIGHT,
  IN_MLP_GATE_BIAS,
  IN_MLP_UP_WEIGHT,
  IN_MLP_UP_BIAS,
  IN_MLP_DOWN_WEIGHT,
  IN_MLP_DOWN_BIAS,
  IN_VISION_Q_WEIGHT,
  IN_VISION_Q_BIAS,
  IN_VISION_K_WEIGHT,
  IN_VISION_K_BIAS,
  IN_VISION_V_WEIGHT,
  IN_VISION_V_BIAS
};

static std::vector<std::pair<int, std::string>> WEIGHT_MAPPING = {
    {IN_INPUT_NORM_WEIGHT, "norm1.weight"},
    {IN_POST_NORM_WEIGHT, "norm2.weight"},
    {IN_QKV_WEIGHT, "qkv.weight"},
    {IN_QKV_BIAS, "qkv.bias"},
    {IN_WATTENTION_OUT_WEIGHT, "attn.proj.weight"},
    {IN_WATTENTION_OUT_BIAS, "attn.proj.bias"},
    {IN_MLP_GATE_WEIGHT, "mlp.gate_proj.weight"},
    {IN_MLP_GATE_BIAS, "mlp.gate_proj.bias"},
    {IN_MLP_UP_WEIGHT, "mlp.up_proj.weight"},
    {IN_MLP_UP_BIAS, "mlp.up_proj.bias"},
    {IN_MLP_DOWN_WEIGHT, "mlp.down_proj.weight"},
    {IN_MLP_DOWN_BIAS, "mlp.down_proj.bias"},
};

// {weight,dim}
static std::map<int, int> WEIGHT_SHARD = {
    {IN_WATTENTION_OUT_WEIGHT, 1},
    {IN_MLP_GATE_WEIGHT, 0},
    {IN_MLP_GATE_BIAS, 0},
    {IN_MLP_UP_WEIGHT, 0},
    {IN_MLP_UP_BIAS, 0},
    {IN_MLP_DOWN_WEIGHT, 1},
};

}  // namespace qwen2dot5_vision_encoder_constants

namespace qwen3_vision_encoder_constants {
enum VisionEncoderLayerTensorId : int {
  IN_INPUT_NORM_WEIGHT = 0,
  IN_INPUT_NORM_BIAS,
  IN_POST_NORM_WEIGHT,
  IN_POST_NORM_BIAS,
  IN_QKV_WEIGHT,
  IN_QKV_BIAS,
  IN_WATTENTION_OUT_WEIGHT,
  IN_WATTENTION_OUT_BIAS,
  IN_LINEAR_FC1_WEIGHT,
  IN_LINEAR_FC1_BIAS,
  IN_LINEAR_FC2_WEIGHT,
  IN_LINEAR_FC2_BIAS,
  IN_VISION_Q_WEIGHT,
  IN_VISION_Q_BIAS,
  IN_VISION_K_WEIGHT,
  IN_VISION_K_BIAS,
  IN_VISION_V_WEIGHT,
  IN_VISION_V_BIAS
};

static std::vector<std::pair<int, std::string>> WEIGHT_MAPPING = {
    {IN_INPUT_NORM_WEIGHT, "norm1.weight"},
    {IN_INPUT_NORM_BIAS, "norm1.bias"},
    {IN_POST_NORM_WEIGHT, "norm2.weight"},
    {IN_POST_NORM_BIAS, "norm2.bias"},
    {IN_QKV_WEIGHT, "attn.qkv.weight"},
    {IN_QKV_BIAS, "attn.qkv.bias"},
    {IN_WATTENTION_OUT_WEIGHT, "attn.proj.weight"},
    {IN_WATTENTION_OUT_BIAS, "attn.proj.bias"},
    {IN_LINEAR_FC1_WEIGHT, "mlp.linear_fc1.weight"},
    {IN_LINEAR_FC1_BIAS, "mlp.linear_fc1.bias"},
    {IN_LINEAR_FC2_WEIGHT, "mlp.linear_fc2.weight"},
    {IN_LINEAR_FC2_BIAS, "mlp.linear_fc2.bias"}};

// {weight,dim}
static std::map<int, int> WEIGHT_SHARD = {
    {IN_WATTENTION_OUT_WEIGHT, 1},
    {IN_LINEAR_FC1_WEIGHT, 0},
    {IN_LINEAR_FC1_BIAS, 0},
    {IN_LINEAR_FC2_WEIGHT, 1},
};

}  // namespace qwen3_vision_encoder_constants

}  // namespace layer
}  // namespace xllm

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

namespace glm4_moe_decoder_constants {

enum DecoderLayerTensorId : int {
  IN_INPUT_NORM_WEIGHT = 0,
  IN_INPUT_NORM_BIAS = 1,
  IN_INPUT_NORM_NEW_WEIGHT = 2,
  IN_INPUT_NORM_NEW_BIAS = 3,

  IN_QKV_WEIGHT_0 = 4,
  IN_QKV_BIAS_0 = 5,
  IN_QKV_DESCALE_0 = 6,
  IN_QKV_OFFSET_0 = 7,
  IN_QKV_SCALE_0 = 8,
  IN_QKV_COMPRESS_IDX_0 = 9,

  IN_QKV_WEIGHT_1 = 10,
  IN_QKV_BIAS_1 = 11,
  IN_QKV_DESCALE_1 = 12,
  IN_QKV_OFFSET_1 = 13,
  IN_QKV_SCALE_1 = 14,
  IN_QKV_COMPRESS_IDX_1 = 15,

  IN_QKV_WEIGHT_2 = 16,
  IN_QKV_BIAS_2 = 17,
  IN_QKV_DESCALE_2 = 18,
  IN_QKV_OFFSET_2 = 19,
  IN_QKV_SCALE_2 = 20,
  IN_QKV_COMPRESS_IDX_2 = 21,

  IN_QKV_DENSE_WEIGHT = 22,
  IN_QKV_DENSE_BIAS = 23,
  IN_QKV_DENSE_DESCALE = 24,
  IN_QKV_DENSE_OFFSET = 25,
  IN_QKV_DENSE_SCALE = 26,
  IN_QKV_DENSE_COMPRESS_IDX = 27,

  IN_POST_ATTN_NORM_WEIGHT = 28,
  IN_POST_ATTN_NORM_BIAS = 29,
  IN_POST_ATTN_NORM_NEW_WEIGHT = 30,
  IN_POST_ATTN_NORM_NEW_BIAS = 31,

  IN_MLP_GATEUP_WEIGHT_SHARED_EXPERT = 32,
  IN_MLP_GATEUP_BIAS_SHARED_EXPERT = 33,
  IN_MLP_GATEUP_DESCALE_SHARED_EXPERT = 34,
  IN_MLP_GATEUP_OFFSET_SHARED_EXPERT = 35,
  IN_MLP_GATEUP_SCALE_SHARED_EXPERT = 36,
  IN_MLP_GATEUP_COMPRESS_IDX_SHARED_EXPERT = 37,

  IN_MLP_DOWN_WEIGHT_SHARED_EXPERT = 38,
  IN_MLP_DOWN_BIAS_SHARED_EXPERT = 39,
  IN_MLP_DOWN_DESCALE_SHARED_EXPERT = 40,
  IN_MLP_DOWN_OFFSET_SHARED_EXPERT = 41,
  IN_MLP_DOWN_SCALE_SHARED_EXPERT = 42,
  IN_MLP_DOWN_COMPRESS_IDX_SHARED_EXPERT = 43,

  IN_SHARED_EXPERT_GATE_WEIGHT = 44,
  IN_SHARED_EXPERT_GATE_BIAS = 45,
  IN_SHARED_EXPERT_GATE_DESCALE = 46,
  IN_SHARED_EXPERT_GATE_OFFSET = 47,
  IN_SHARED_EXPERT_GATE_SCALE = 48,
  IN_SHARED_EXPERT_GATE_COMPRESS_IDX = 49,

  BLOCK_SPARSE_MOE_GATE_WEIGHT = 50,
  BLOCK_SPARSE_MOE_GATE_BIAS = 51,
  BLOCK_SPARSE_MOE_GATE_DESCALE = 52,
  BLOCK_SPARSE_MOE_GATE_OFFSET = 53,
  BLOCK_SPARSE_MOE_GATE_SCALE = 54,
  BLOCK_SPARSE_MOE_GATE_COMPRESS_IDX = 55,

  IN_MLP_GATEUP_WEIGHT = 56,
  IN_MLP_GATEUP_BIAS = 57,
  IN_MLP_GATEUP_DESCALE = 58,
  IN_MLP_GATEUP_OFFSET = 59,
  IN_MLP_GATEUP_SCALE = 60,
  IN_MLP_GATEUP_COMPRESS_IDX = 61,

  IN_MLP_DOWN_WEIGHT = 62,
  IN_MLP_DOWN_BIAS = 63,
  IN_MLP_DOWN_DESCALE = 64,
  IN_MLP_DOWN_OFFSET = 65,
  IN_MLP_DOWN_SCALE = 66,
  IN_MLP_DOWN_COMPRESS_IDX = 67,

  Q_NORM_WEIGHT = 68,
  K_NORM_WEIGHT = 69
};

inline const std::unordered_map<std::string, int> WEIGHT_MAPPING = {
    {"input_layernorm.weight", IN_INPUT_NORM_WEIGHT},

    {"self_attn.q_proj.weight", IN_QKV_WEIGHT_0},
    {"self_attn.q_proj.bias", IN_QKV_BIAS_0},

    {"self_attn.k_proj.weight", IN_QKV_WEIGHT_1},
    {"self_attn.k_proj.bias", IN_QKV_BIAS_1},

    {"self_attn.v_proj.weight", IN_QKV_WEIGHT_2},
    {"self_attn.v_proj.bias", IN_QKV_BIAS_2},

    {"self_attn.o_proj.weight", IN_QKV_DENSE_WEIGHT},

    {"post_attention_layernorm.weight", IN_POST_ATTN_NORM_WEIGHT},

    {"mlp.gate_proj.weight", IN_MLP_GATEUP_WEIGHT_SHARED_EXPERT},
    {"mlp.up_proj.weight", IN_MLP_GATEUP_WEIGHT_SHARED_EXPERT},
    {"mlp.down_proj.weight", IN_MLP_DOWN_WEIGHT_SHARED_EXPERT},

    {"mlp.shared_experts.gate_proj.weight", IN_MLP_GATEUP_WEIGHT_SHARED_EXPERT},
    {"mlp.shared_experts.up_proj.weight", IN_MLP_GATEUP_WEIGHT_SHARED_EXPERT},
    {"mlp.shared_experts.down_proj.weight", IN_MLP_DOWN_WEIGHT_SHARED_EXPERT},

    {"mlp.gate.weight", BLOCK_SPARSE_MOE_GATE_WEIGHT},
    {"mlp.gate.e_score_correction_bias", BLOCK_SPARSE_MOE_GATE_BIAS},

    {"gate_proj.weight", IN_MLP_GATEUP_WEIGHT},
    {"up_proj.weight", IN_MLP_GATEUP_WEIGHT},
    {"down_proj.weight", IN_MLP_DOWN_WEIGHT},
};

inline const std::unordered_map<std::string, int> WEIGHT_MAPPING_W8A8 = {
    {"input_layernorm.weight", IN_INPUT_NORM_WEIGHT},
    {"input_layernorm.bias", IN_INPUT_NORM_NEW_BIAS},

    {"self_attn.q_proj.weight", IN_QKV_WEIGHT_0},
    {"self_attn.q_proj.deq_scale", IN_QKV_DESCALE_0},
    {"self_attn.q_proj.quant_bias", IN_QKV_BIAS_0},
    {"self_attn.q_proj.input_offset", IN_QKV_OFFSET_0},
    {"self_attn.q_proj.input_scale", IN_QKV_SCALE_0},

    {"self_attn.k_proj.weight", IN_QKV_WEIGHT_1},
    {"self_attn.k_proj.deq_scale", IN_QKV_DESCALE_1},
    {"self_attn.k_proj.quant_bias", IN_QKV_BIAS_1},

    {"self_attn.v_proj.weight", IN_QKV_WEIGHT_2},
    {"self_attn.v_proj.deq_scale", IN_QKV_DESCALE_2},
    {"self_attn.v_proj.quant_bias", IN_QKV_BIAS_2},

    {"self_attn.o_proj.weight", IN_QKV_DENSE_WEIGHT},
    {"self_attn.o_proj.quant_bias", IN_QKV_DENSE_BIAS},
    {"self_attn.o_proj.deq_scale", IN_QKV_DENSE_DESCALE},
    {"self_attn.o_proj.weight_offset", IN_QKV_DENSE_OFFSET},
    {"self_attn.o_proj.weight_scale", IN_QKV_DENSE_SCALE},

    {"post_attention_layernorm.weight", IN_POST_ATTN_NORM_WEIGHT},
    {"post_attention_layernorm.bias", IN_POST_ATTN_NORM_NEW_BIAS},

    {"mlp.gate_proj.weight", IN_MLP_GATEUP_WEIGHT_SHARED_EXPERT},
    {"mlp.gate_proj.weight_offset", IN_MLP_GATEUP_OFFSET_SHARED_EXPERT},
    {"mlp.gate_proj.weight_scale", IN_MLP_GATEUP_SCALE_SHARED_EXPERT},

    {"mlp.up_proj.weight", IN_MLP_GATEUP_WEIGHT_SHARED_EXPERT},
    {"mlp.up_proj.weight_offset", IN_MLP_GATEUP_OFFSET_SHARED_EXPERT},
    {"mlp.up_proj.weight_scale", IN_MLP_GATEUP_SCALE_SHARED_EXPERT},

    {"mlp.down_proj.weight", IN_MLP_DOWN_WEIGHT_SHARED_EXPERT},
    {"mlp.down_proj.weight_offset", IN_MLP_DOWN_OFFSET_SHARED_EXPERT},
    {"mlp.down_proj.weight_scale", IN_MLP_DOWN_SCALE_SHARED_EXPERT},

    {"mlp.shared_experts.gate_proj.weight", IN_MLP_GATEUP_WEIGHT_SHARED_EXPERT},
    {"mlp.shared_experts.gate_proj.weight_offset",
     IN_MLP_GATEUP_OFFSET_SHARED_EXPERT},
    {"mlp.shared_experts.gate_proj.weight_scale",
     IN_MLP_GATEUP_SCALE_SHARED_EXPERT},

    {"mlp.shared_experts.up_proj.weight", IN_MLP_GATEUP_WEIGHT_SHARED_EXPERT},
    {"mlp.shared_experts.up_proj.weight_offset",
     IN_MLP_GATEUP_OFFSET_SHARED_EXPERT},
    {"mlp.shared_experts.up_proj.weight_scale",
     IN_MLP_GATEUP_SCALE_SHARED_EXPERT},

    {"mlp.shared_experts.down_proj.weight", IN_MLP_DOWN_WEIGHT_SHARED_EXPERT},
    {"mlp.shared_experts.down_proj.weight_offset",
     IN_MLP_DOWN_OFFSET_SHARED_EXPERT},
    {"mlp.shared_experts.down_proj.weight_scale",
     IN_MLP_DOWN_SCALE_SHARED_EXPERT},

    {"mlp.gate.weight", BLOCK_SPARSE_MOE_GATE_WEIGHT},
    {"mlp.gate.e_score_correction_bias", BLOCK_SPARSE_MOE_GATE_BIAS},

    {"gate_proj.weight", IN_MLP_GATEUP_WEIGHT},
    {"gate_proj.weight_offset", IN_MLP_GATEUP_OFFSET},
    {"gate_proj.weight_scale", IN_MLP_GATEUP_SCALE},
    {"up_proj.weight", IN_MLP_GATEUP_WEIGHT},
    {"up_proj.weight_offset", IN_MLP_GATEUP_OFFSET},
    {"up_proj.weight_scale", IN_MLP_GATEUP_SCALE},

    {"down_proj.weight", IN_MLP_DOWN_WEIGHT},
    {"down_proj.weight_offset", IN_MLP_DOWN_OFFSET},
    {"down_proj.weight_scale", IN_MLP_DOWN_SCALE},
};

inline const std::unordered_map<std::string, std::vector<int>>
    SPECIAL_MULTI_ASSIGN_W8A8 = {
        {"input_layernorm.weight",
         {IN_INPUT_NORM_WEIGHT, IN_INPUT_NORM_NEW_WEIGHT}},
        {"post_attention_layernorm.weight",
         {IN_POST_ATTN_NORM_WEIGHT, IN_POST_ATTN_NORM_NEW_WEIGHT}},
};

inline const std::map<int, int> WEIGHT_SHARD = {
    {IN_QKV_WEIGHT_0, 0},
    {IN_QKV_BIAS_0, 0},
    {IN_QKV_WEIGHT_1, 0},
    {IN_QKV_BIAS_1, 0},
    {IN_QKV_WEIGHT_2, 0},
    {IN_QKV_BIAS_2, 0},
    {IN_QKV_DENSE_WEIGHT, 1},
    {IN_MLP_GATEUP_WEIGHT_SHARED_EXPERT, 0},
    {IN_MLP_DOWN_WEIGHT_SHARED_EXPERT, 1},
    {IN_MLP_GATEUP_WEIGHT, 0},
    {IN_MLP_DOWN_WEIGHT, 1},
};

inline const std::map<int, int> WEIGHT_SHARD_W8A8 = {
    {IN_QKV_WEIGHT_0, 0},
    {IN_QKV_BIAS_0, 0},
    {IN_QKV_DESCALE_0, 0},
    {IN_QKV_WEIGHT_1, 0},
    {IN_QKV_BIAS_1, 0},
    {IN_QKV_DESCALE_1, 0},
    {IN_QKV_WEIGHT_2, 0},
    {IN_QKV_BIAS_2, 0},
    {IN_QKV_DESCALE_2, 0},
    {IN_QKV_DENSE_WEIGHT, 1},
    {IN_MLP_GATEUP_WEIGHT_SHARED_EXPERT, 0},
    {IN_MLP_GATEUP_OFFSET_SHARED_EXPERT, 0},
    {IN_MLP_GATEUP_SCALE_SHARED_EXPERT, 0},
    {IN_MLP_DOWN_WEIGHT_SHARED_EXPERT, 1},
    {IN_MLP_GATEUP_WEIGHT, 0},
    {IN_MLP_GATEUP_OFFSET, 0},
    {IN_MLP_GATEUP_SCALE, 0},
    {IN_MLP_DOWN_WEIGHT, 1},
};

}  // namespace glm4_moe_decoder_constants

namespace glm4_moe_lite_decoder_constants {

enum DecoderLayerTensorId : int {
  IN_INPUT_NORM_WEIGHT = 0,
  IN_INPUT_NORM_BIAS = 1,
  IN_INPUT_NORM_NEW_WEIGHT = 2,
  IN_INPUT_NORM_NEW_BIAS = 3,

  IN_Q_A_WEIGHT = 4,
  IN_Q_A_BIAS = 5,
  IN_Q_A_DESCALE = 6,
  IN_Q_A_OFFSET = 7,
  IN_Q_A_SCALE = 8,
  IN_Q_A_COMPRESS_IDX = 9,

  IN_Q_B_WEIGHT = 10,
  IN_Q_B_BIAS = 11,
  IN_Q_B_DESCALE = 12,
  IN_Q_B_OFFSET = 13,
  IN_Q_B_SCALE = 14,
  IN_Q_B_COMPRESS_IDX = 15,

  IN_KV_A_WEIGHT = 16,
  IN_KV_A_BIAS = 17,
  IN_KV_A_DESCALE = 18,
  IN_KV_A_OFFSET = 19,
  IN_KV_A_SCALE = 20,
  IN_KV_A_COMPRESS_IDX = 21,

  IN_K_B_WEIGHT = 22,
  IN_K_B_BIAS = 23,
  IN_K_B_DESCALE = 24,
  IN_K_B_OFFSET = 25,
  IN_K_B_SCALE = 26,
  IN_K_B_COMPRESS_IDX = 27,

  IN_V_B_WEIGHT = 28,
  IN_V_B_BIAS = 29,
  IN_V_B_DESCALE = 30,
  IN_V_B_OFFSET = 31,
  IN_V_B_SCALE = 32,
  IN_V_B_COMPRESS_IDX = 33,

  IN_QKV_DENSE_WEIGHT = 34,
  IN_QKV_DENSE_BIAS = 35,
  IN_QKV_DENSE_DESCALE = 36,
  IN_QKV_DENSE_OFFSET = 37,
  IN_QKV_DENSE_SCALE = 38,
  IN_QKV_DENSE_COMPRESS_IDX = 39,

  IN_POST_ATTN_NORM_WEIGHT = 40,
  IN_POST_ATTN_NORM_BIAS = 41,
  IN_POST_ATTN_NORM_NEW_WEIGHT = 42,
  IN_POST_ATTN_NORM_NEW_BIAS = 43,

  IN_MLP_GATEUP_WEIGHT_SHARED_EXPERT = 44,
  IN_MLP_GATEUP_BIAS_SHARED_EXPERT = 45,
  IN_MLP_GATEUP_DESCALE_SHARED_EXPERT = 46,
  IN_MLP_GATEUP_OFFSET_SHARED_EXPERT = 47,
  IN_MLP_GATEUP_SCALE_SHARED_EXPERT = 48,
  IN_MLP_GATEUP_COMPRESS_IDX_SHARED_EXPERT = 49,

  IN_MLP_DOWN_WEIGHT_SHARED_EXPERT = 50,
  IN_MLP_DOWN_BIAS_SHARED_EXPERT = 51,
  IN_MLP_DOWN_DESCALE_SHARED_EXPERT = 52,
  IN_MLP_DOWN_OFFSET_SHARED_EXPERT = 53,
  IN_MLP_DOWN_SCALE_SHARED_EXPERT = 54,
  IN_MLP_DOWN_COMPRESS_IDX_SHARED_EXPERT = 55,

  IN_SHARED_EXPERT_GATE_WEIGHT = 56,
  IN_SHARED_EXPERT_GATE_BIAS = 57,
  IN_SHARED_EXPERT_GATE_DESCALE = 58,
  IN_SHARED_EXPERT_GATE_OFFSET = 59,
  IN_SHARED_EXPERT_GATE_SCALE = 60,
  IN_SHARED_EXPERT_GATE_COMPRESS_IDX = 61,

  BLOCK_SPARSE_MOE_GATE_WEIGHT = 62,
  BLOCK_SPARSE_MOE_GATE_BIAS = 63,
  BLOCK_SPARSE_MOE_GATE_DESCALE = 64,
  BLOCK_SPARSE_MOE_GATE_OFFSET = 65,
  BLOCK_SPARSE_MOE_GATE_SCALE = 66,
  BLOCK_SPARSE_MOE_GATE_COMPRESS_IDX = 67,

  IN_MLP_GATEUP_WEIGHT = 68,
  IN_MLP_GATEUP_BIAS = 69,
  IN_MLP_GATEUP_DESCALE = 70,
  IN_MLP_GATEUP_OFFSET = 71,
  IN_MLP_GATEUP_SCALE = 72,
  IN_MLP_GATEUP_COMPRESS_IDX = 73,

  IN_MLP_DOWN_WEIGHT = 74,
  IN_MLP_DOWN_BIAS = 75,
  IN_MLP_DOWN_DESCALE = 76,
  IN_MLP_DOWN_OFFSET = 77,
  IN_MLP_DOWN_SCALE = 78,
  IN_MLP_DOWN_COMPRESS_IDX = 79,

  Q_NORM_WEIGHT = 80,
  Q_NORM_BIAS = 81,
  KV_NORM_WEIGHT = 82,
  KV_NORM_BIAS = 83
};

inline const std::unordered_map<std::string, int> WEIGHT_MAPPING = {
    {"input_layernorm.weight", IN_INPUT_NORM_WEIGHT},
    {"post_attention_layernorm.weight", IN_POST_ATTN_NORM_WEIGHT},

    {"self_attn.q_a_proj.weight", IN_Q_A_WEIGHT},
    {"self_attn.q_a_proj.bias", IN_Q_A_BIAS},
    {"self_attn.kv_a_proj_with_mqa.weight", IN_KV_A_WEIGHT},
    {"self_attn.kv_a_proj_with_mqa.bias", IN_KV_A_BIAS},

    {"self_attn.q_b_proj.weight", IN_Q_B_WEIGHT},
    {"self_attn.q_b_proj.bias", IN_Q_B_BIAS},
    {"self_attn.kv_b_proj.weight", IN_K_B_WEIGHT},
    {"self_attn.kv_b_proj.bias", IN_K_B_BIAS},

    {"self_attn.q_a_layernorm.weight", Q_NORM_WEIGHT},
    {"self_attn.q_a_layernorm.bias", Q_NORM_BIAS},
    {"self_attn.kv_a_layernorm.weight", KV_NORM_WEIGHT},
    {"self_attn.kv_a_layernorm.bias", KV_NORM_BIAS},

    {"self_attn.o_proj.weight", IN_QKV_DENSE_WEIGHT},
    {"self_attn.o_proj.bias", IN_QKV_DENSE_BIAS},

    {"mlp.gate_proj.weight", IN_MLP_GATEUP_WEIGHT_SHARED_EXPERT},
    {"mlp.up_proj.weight", IN_MLP_GATEUP_WEIGHT_SHARED_EXPERT},
    {"mlp.down_proj.weight", IN_MLP_DOWN_WEIGHT_SHARED_EXPERT},
    {"mlp.shared_experts.gate_proj.weight", IN_MLP_GATEUP_WEIGHT_SHARED_EXPERT},
    {"mlp.shared_experts.up_proj.weight", IN_MLP_GATEUP_WEIGHT_SHARED_EXPERT},
    {"mlp.shared_experts.down_proj.weight", IN_MLP_DOWN_WEIGHT_SHARED_EXPERT},

    {"mlp.gate.weight", BLOCK_SPARSE_MOE_GATE_WEIGHT},
    {"mlp.gate.e_score_correction_bias", BLOCK_SPARSE_MOE_GATE_BIAS},

    {"gate_proj.weight", IN_MLP_GATEUP_WEIGHT},
    {"up_proj.weight", IN_MLP_GATEUP_WEIGHT},
    {"down_proj.weight", IN_MLP_DOWN_WEIGHT},
};

inline const std::unordered_map<std::string, int> WEIGHT_MAPPING_W8A8 = {
    {"input_layernorm.weight", IN_INPUT_NORM_WEIGHT},
    {"input_layernorm.bias", IN_INPUT_NORM_NEW_BIAS},

    {"self_attn.q_a_proj.weight", IN_Q_A_WEIGHT},
    {"self_attn.q_a_proj.quant_bias", IN_Q_A_BIAS},
    {"self_attn.q_a_proj.deq_scale", IN_Q_A_DESCALE},
    {"self_attn.q_a_proj.input_offset", IN_Q_A_OFFSET},
    {"self_attn.q_a_proj.input_scale", IN_Q_A_SCALE},

    {"self_attn.q_a_layernorm.weight", Q_NORM_WEIGHT},
    {"self_attn.q_a_layernorm.bias", Q_NORM_BIAS},

    {"self_attn.q_proj.weight", IN_Q_B_WEIGHT},
    {"self_attn.q_b_proj.weight", IN_Q_B_WEIGHT},
    {"self_attn.q_b_proj.quant_bias", IN_Q_B_BIAS},
    {"self_attn.q_b_proj.input_scale", IN_Q_B_SCALE},
    {"self_attn.q_b_proj.deq_scale", IN_Q_B_DESCALE},
    {"self_attn.q_b_proj.input_offset", IN_Q_B_OFFSET},

    {"self_attn.kv_a_proj_with_mqa.weight", IN_KV_A_WEIGHT},
    {"self_attn.kv_a_proj_with_mqa.quant_bias", IN_KV_A_BIAS},
    {"self_attn.kv_a_proj_with_mqa.deq_scale", IN_KV_A_DESCALE},
    {"self_attn.kv_a_proj_with_mqa.input_offset", IN_KV_A_OFFSET},
    {"self_attn.kv_a_proj_with_mqa.input_scale", IN_KV_A_SCALE},

    {"self_attn.kv_a_layernorm.weight", KV_NORM_WEIGHT},
    {"self_attn.kv_a_layernorm.bias", KV_NORM_BIAS},

    {"self_attn.kv_b_proj.weight", IN_K_B_WEIGHT},
    {"self_attn.kv_b_proj.bias", IN_K_B_BIAS},

    {"self_attn.o_proj.weight", IN_QKV_DENSE_WEIGHT},
    {"self_attn.o_proj.quant_bias", IN_QKV_DENSE_BIAS},
    {"self_attn.o_proj.deq_scale", IN_QKV_DENSE_DESCALE},
    {"self_attn.o_proj.weight_offset", IN_QKV_DENSE_OFFSET},
    {"self_attn.o_proj.weight_scale", IN_QKV_DENSE_SCALE},

    {"post_attention_layernorm.weight", IN_POST_ATTN_NORM_WEIGHT},
    {"post_attention_layernorm.bias", IN_POST_ATTN_NORM_NEW_BIAS},

    {"mlp.gate_proj.weight", IN_MLP_GATEUP_WEIGHT_SHARED_EXPERT},
    {"mlp.gate_proj.weight_offset", IN_MLP_GATEUP_OFFSET_SHARED_EXPERT},
    {"mlp.gate_proj.weight_scale", IN_MLP_GATEUP_SCALE_SHARED_EXPERT},

    {"mlp.up_proj.weight", IN_MLP_GATEUP_WEIGHT_SHARED_EXPERT},
    {"mlp.up_proj.weight_offset", IN_MLP_GATEUP_OFFSET_SHARED_EXPERT},
    {"mlp.up_proj.weight_scale", IN_MLP_GATEUP_SCALE_SHARED_EXPERT},

    {"mlp.down_proj.weight", IN_MLP_DOWN_WEIGHT_SHARED_EXPERT},
    {"mlp.down_proj.weight_offset", IN_MLP_DOWN_OFFSET_SHARED_EXPERT},
    {"mlp.down_proj.weight_scale", IN_MLP_DOWN_SCALE_SHARED_EXPERT},

    {"mlp.shared_experts.gate_proj.weight", IN_MLP_GATEUP_WEIGHT_SHARED_EXPERT},
    {"mlp.shared_experts.gate_proj.weight_offset",
     IN_MLP_GATEUP_OFFSET_SHARED_EXPERT},
    {"mlp.shared_experts.gate_proj.weight_scale",
     IN_MLP_GATEUP_SCALE_SHARED_EXPERT},

    {"mlp.shared_experts.up_proj.weight", IN_MLP_GATEUP_WEIGHT_SHARED_EXPERT},
    {"mlp.shared_experts.up_proj.weight_offset",
     IN_MLP_GATEUP_OFFSET_SHARED_EXPERT},
    {"mlp.shared_experts.up_proj.weight_scale",
     IN_MLP_GATEUP_SCALE_SHARED_EXPERT},

    {"mlp.shared_experts.down_proj.weight", IN_MLP_DOWN_WEIGHT_SHARED_EXPERT},
    {"mlp.shared_experts.down_proj.weight_offset",
     IN_MLP_DOWN_OFFSET_SHARED_EXPERT},
    {"mlp.shared_experts.down_proj.weight_scale",
     IN_MLP_DOWN_SCALE_SHARED_EXPERT},

    {"mlp.gate.weight", BLOCK_SPARSE_MOE_GATE_WEIGHT},
    {"mlp.gate.e_score_correction_bias", BLOCK_SPARSE_MOE_GATE_BIAS},

    {"gate_proj.weight", IN_MLP_GATEUP_WEIGHT},
    {"gate_proj.weight_offset", IN_MLP_GATEUP_OFFSET},
    {"gate_proj.weight_scale", IN_MLP_GATEUP_SCALE},
    {"up_proj.weight", IN_MLP_GATEUP_WEIGHT},
    {"up_proj.weight_offset", IN_MLP_GATEUP_OFFSET},
    {"up_proj.weight_scale", IN_MLP_GATEUP_SCALE},

    {"down_proj.weight", IN_MLP_DOWN_WEIGHT},
    {"down_proj.weight_offset", IN_MLP_DOWN_OFFSET},
    {"down_proj.weight_scale", IN_MLP_DOWN_SCALE},
};

inline const std::unordered_map<std::string, std::vector<int>>
    SPECIAL_MULTI_ASSIGN_W8A8 = {
        {"input_layernorm.weight",
         {IN_INPUT_NORM_WEIGHT, IN_INPUT_NORM_NEW_WEIGHT}},
        {"post_attention_layernorm.weight",
         {IN_POST_ATTN_NORM_WEIGHT, IN_POST_ATTN_NORM_NEW_WEIGHT}},
};

inline const std::map<int, int> WEIGHT_SHARD = {
    {IN_Q_B_WEIGHT, 0},
    {IN_Q_B_BIAS, 0},
    {IN_K_B_WEIGHT, 0},
    {IN_K_B_BIAS, 0},
    {IN_QKV_DENSE_WEIGHT, 1},

    {IN_MLP_GATEUP_WEIGHT_SHARED_EXPERT, 0},
    {IN_MLP_DOWN_WEIGHT_SHARED_EXPERT, 1},
    {IN_MLP_GATEUP_WEIGHT, 0},
    {IN_MLP_DOWN_WEIGHT, 1},
};

inline const std::map<int, int> WEIGHT_SHARD_W8A8 = {
    {IN_Q_B_WEIGHT, 0},
    {IN_Q_B_BIAS, 0},
    {IN_Q_B_SCALE, 0},
    {IN_Q_B_DESCALE, 0},
    {IN_Q_B_OFFSET, 0},
    {IN_K_B_WEIGHT, 0},
    {IN_K_B_BIAS, 0},
    {IN_K_B_SCALE, 0},
    {IN_K_B_DESCALE, 0},
    {IN_K_B_OFFSET, 0},
    {IN_QKV_DENSE_WEIGHT, 1},

    {IN_MLP_GATEUP_WEIGHT_SHARED_EXPERT, 0},
    {IN_MLP_GATEUP_OFFSET_SHARED_EXPERT, 0},
    {IN_MLP_GATEUP_SCALE_SHARED_EXPERT, 0},
    {IN_MLP_DOWN_WEIGHT_SHARED_EXPERT, 1},
    {IN_MLP_GATEUP_WEIGHT, 0},
    {IN_MLP_GATEUP_OFFSET, 0},
    {IN_MLP_GATEUP_SCALE, 0},
    {IN_MLP_DOWN_WEIGHT, 1},
};

}  // namespace glm4_moe_lite_decoder_constants

}  // namespace layer
}  // namespace xllm

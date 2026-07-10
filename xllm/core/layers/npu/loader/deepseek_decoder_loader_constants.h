/* Copyright 2025-2026 The xLLM Authors.

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
#include <unordered_map>
#include <utility>
#include <vector>

namespace xllm {
namespace layer {
namespace deepseek_v2_decoder_constants {

enum DecoderLayerTensorId : int {
  IN_INPUT_NORM_WEIGHT = 0,
  IN_INPUT_NORM_BIAS = 1,
  IN_INPUT_NORM_NEW_WEIGHT = 2,
  IN_INPUT_NORM_NEW_BIAS = 3,

  IN_Q_PROJ_A_WEIGHT = 4,
  IN_Q_PROJ_A_BIAS = 5,
  IN_Q_PROJ_A_DESCALE = 6,
  IN_Q_PROJ_A_OFFSET = 7,
  IN_Q_PROJ_A_SCALE = 8,
  IN_Q_PROJ_A_COMPRESS_IDX = 9,
  IN_Q_PROJ_A_LAYERNORM_WEIGHT = 10,
  IN_Q_PROJ_A_LAYERNORM_BIAS = 11,

  IN_Q_PROJ_B_WEIGHT = 12,
  IN_Q_PROJ_B_BIAS = 13,
  IN_Q_PROJ_B_DESCALE = 14,
  IN_Q_PROJ_B_OFFSET = 15,
  IN_Q_PROJ_B_SCALE = 16,
  IN_Q_PROJ_B_COMPRESS_IDX = 17,

  IN_KV_PROJ_WITH_MQA_WEIGHT = 18,
  IN_KV_PROJ_WITH_MQA_BIAS = 19,
  IN_KV_PROJ_WITH_MQA_DESCALE = 20,
  IN_KV_PROJ_WITH_MQA_OFFSET = 21,
  IN_KV_PROJ_WITH_MQA_SCALE = 22,
  IN_KV_PROJ_WITH_MQA_COMPRESS_IDX = 23,

  IN_KV_PROJ_A_LAYERNORM_WEIGHT = 24,
  IN_KV_PROJ_A_LAYERNORM_BIAS = 25,

  IN_K_PROJ_B_FOR_Q_WEIGHT = 26,
  IN_K_PROJ_B_FOR_Q_BIAS = 27,
  IN_K_PROJ_B_FOR_Q_DESCALE = 28,
  IN_K_PROJ_B_FOR_Q_OFFSET = 29,
  IN_K_PROJ_B_FOR_Q_SCALE = 30,
  IN_K_PROJ_B_FOR_Q_COMPRESS_IDX = 31,

  IN_V_PROJ_B_FOR_O_WEIGHT = 32,
  IN_V_PROJ_B_FOR_O_BIAS = 33,
  IN_V_PROJ_B_FOR_O_DESCALE = 34,
  IN_V_PROJ_B_FOR_O_OFFSET = 35,
  IN_V_PROJ_B_FOR_O_SCALE = 36,
  IN_V_PROJ_B_FOR_O_COMPRESS_IDX = 37,

  IN_ATTENTION_OUT_WEIGHT = 38,
  IN_ATTENTION_OUT_BIAS = 39,
  IN_ATTENTION_OUT_DESCALE = 40,
  IN_ATTENTION_OUT_OFFSET = 41,
  IN_ATTENTION_OUT_SCALE = 42,
  IN_ATTENTION_OUT_COMPRESS_IDX = 43,

  IN_SELFATTENTION_OUT_NORM_WEIGHT = 44,
  IN_SELFATTENTION_OUT_NORM_BIAS = 45,
  IN_SELFATTENTION_OUT_NEW_NORM_WEIGHT = 46,
  IN_SELFATTENTION_OUT_NEW_NORM_BIAS = 47,

  IN_MLP_GATEUP_WEIGHT_SHARED_EXPERT = 48,
  IN_MLP_GATEUP_BIAS_SHARED_EXPERT = 49,
  IN_MLP_GATEUP_DESCALE_SHARED_EXPERT = 50,
  IN_MLP_GATEUP_OFFSET_SHARED_EXPERT = 51,
  IN_MLP_GATEUP_SCALE_SHARED_EXPERT = 52,
  IN_MLP_GATEUP_COMPRESS_IDX_SHARED_EXPERT = 53,

  IN_MLP_DOWN_WEIGHT_SHARED_EXPERT = 54,
  IN_MLP_DOWN_BIAS_SHARED_EXPERT = 55,
  IN_MLP_DOWN_DESCALE_SHARED_EXPERT = 56,
  IN_MLP_DOWN_OFFSET_SHARED_EXPERT = 57,
  IN_MLP_DOWN_SCALE_SHARED_EXPERT = 58,
  IN_MLP_DOWN_COMPRESS_IDX_SHARED_EXPERT = 59,

  IN_SHARED_EXPERT_GATE_WEIGHT = 60,
  IN_SHARED_EXPERT_GATE_BIAS = 61,
  IN_SHARED_EXPERT_GATE_DESCALE = 62,
  IN_SHARED_EXPERT_GATE_OFFSET = 63,
  IN_SHARED_EXPERT_GATE_SCALE = 64,
  IN_SHARED_EXPERT_GATE_COMPRESS_IDX = 65,

  IN_BLOCK_SPARSE_MOE_GATE_WEIGHT = 66,
  IN_BLOCK_SPARSE_MOE_GATE_BIAS = 67,
  IN_BLOCK_SPARSE_MOE_GATE_DESCALE = 68,
  IN_BLOCK_SPARSE_MOE_GATE_OFFSET = 69,
  IN_BLOCK_SPARSE_MOE_GATE_SCALE = 70,
  IN_BLOCK_SPARSE_MOE_GATE_COMPRESS_IDX = 71,

  IN_MLP_GATEUP_WEIGHT_EXPERT = 72,
  IN_MLP_GATEUP_BIAS_EXPERT = 73,
  IN_MLP_GATEUP_DESCALE_EXPERT = 74,
  IN_MLP_GATEUP_OFFSET_EXPERT = 75,
  IN_MLP_GATEUP_SCALE_EXPERT = 76,
  IN_MLP_GATEUP_COMPRESS_IDX_EXPERT = 77,

  IN_MLP_DOWN_WEIGHT_EXPERT = 78,
  IN_MLP_DOWN_BIAS_EXPERT = 79,
  IN_MLP_DOWN_DESCALE_EXPERT = 80,
  IN_MLP_DOWN_OFFSET_EXPERT = 81,
  IN_MLP_DOWN_SCALE_EXPERT = 82,
  IN_MLP_DOWN_COMPRESS_IDX_EXPERT = 83,
};

inline const std::vector<std::pair<int, std::string>> WEIGHT_MAPPING = {};

inline const std::unordered_map<std::string, int> WEIGHT_MAPPING_W8A8 = {
    {"input_layernorm.weight", IN_INPUT_NORM_WEIGHT},
    {"input_layernorm.bias", IN_INPUT_NORM_BIAS},

    {"self_attn.q_a_proj.weight", IN_Q_PROJ_A_WEIGHT},
    {"self_attn.q_a_proj.quant_bias", IN_Q_PROJ_A_BIAS},
    {"self_attn.q_a_proj.deq_scale", IN_Q_PROJ_A_DESCALE},
    {"self_attn.q_a_proj.input_offset", IN_Q_PROJ_A_OFFSET},
    {"self_attn.q_a_proj.input_scale", IN_Q_PROJ_A_SCALE},
    {"self_attn.q_a_layernorm.weight", IN_Q_PROJ_A_LAYERNORM_WEIGHT},
    {"self_attn.q_a_layernorm.bias", IN_Q_PROJ_A_LAYERNORM_BIAS},

    {"self_attn.q_proj.weight", IN_Q_PROJ_B_WEIGHT},
    {"self_attn.q_b_proj.weight", IN_Q_PROJ_B_WEIGHT},
    {"self_attn.q_b_proj.quant_bias", IN_Q_PROJ_B_BIAS},
    {"self_attn.q_b_proj.input_scale", IN_Q_PROJ_B_SCALE},
    {"self_attn.q_b_proj.deq_scale", IN_Q_PROJ_B_DESCALE},
    {"self_attn.q_b_proj.input_offset", IN_Q_PROJ_B_OFFSET},

    {"self_attn.kv_a_proj_with_mqa.weight", IN_KV_PROJ_WITH_MQA_WEIGHT},
    {"self_attn.kv_a_proj_with_mqa.quant_bias", IN_KV_PROJ_WITH_MQA_BIAS},
    {"self_attn.kv_a_proj_with_mqa.deq_scale", IN_KV_PROJ_WITH_MQA_DESCALE},
    {"self_attn.kv_a_proj_with_mqa.input_offset", IN_KV_PROJ_WITH_MQA_OFFSET},
    {"self_attn.kv_a_proj_with_mqa.input_scale", IN_KV_PROJ_WITH_MQA_SCALE},

    {"self_attn.kv_a_layernorm.weight", IN_KV_PROJ_A_LAYERNORM_WEIGHT},
    {"self_attn.kv_a_layernorm.bias", IN_KV_PROJ_A_LAYERNORM_BIAS},

    {"self_attn.kv_b_proj.weight", IN_K_PROJ_B_FOR_Q_WEIGHT},  // merge
    // {"self_attn.kv_b_proj.weight", IN_V_PROJ_B_FOR_O_WEIGHT},  // merge

    {"self_attn.o_proj.weight", IN_ATTENTION_OUT_WEIGHT},
    {"self_attn.o_proj.quant_bias", IN_ATTENTION_OUT_BIAS},
    {"self_attn.o_proj.deq_scale", IN_ATTENTION_OUT_DESCALE},
    {"self_attn.o_proj.input_offset", IN_ATTENTION_OUT_OFFSET},
    {"self_attn.o_proj.input_scale", IN_ATTENTION_OUT_SCALE},

    {"post_attention_layernorm.weight", IN_SELFATTENTION_OUT_NORM_WEIGHT},
    {"post_attention_layernorm.bias", IN_SELFATTENTION_OUT_NORM_BIAS},

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

    {"mlp.gate.weight", IN_BLOCK_SPARSE_MOE_GATE_WEIGHT},
    {"mlp.gate.e_score_correction_bias", IN_BLOCK_SPARSE_MOE_GATE_BIAS},

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

inline const std::map<int, int> WEIGHT_SHARD = {};

inline const std::map<int, int> WEIGHT_SHARD_W8A8 = {
    {IN_Q_PROJ_B_WEIGHT, 0},
    {IN_Q_PROJ_B_BIAS, 0},
    {IN_Q_PROJ_B_DESCALE, 0},
    {IN_K_PROJ_B_FOR_Q_WEIGHT, 0},
    {IN_V_PROJ_B_FOR_O_WEIGHT, 0},
    {IN_ATTENTION_OUT_WEIGHT, 1},
    {IN_MLP_GATEUP_WEIGHT_SHARED_EXPERT, 0},
    {IN_MLP_GATEUP_OFFSET_SHARED_EXPERT, 0},
    {IN_MLP_GATEUP_SCALE_SHARED_EXPERT, 0},
    {IN_MLP_DOWN_WEIGHT_SHARED_EXPERT, 1},
    {IN_MLP_GATEUP_WEIGHT_EXPERT, 0},
    {IN_MLP_GATEUP_OFFSET_EXPERT, 0},
    {IN_MLP_GATEUP_SCALE_EXPERT, 0},
    {IN_MLP_DOWN_WEIGHT_EXPERT, 1},
};

inline const std::vector<int> SQUEEZE_WEIGHT_VEC = {
    IN_MLP_GATEUP_OFFSET_SHARED_EXPERT,
    IN_MLP_GATEUP_SCALE_SHARED_EXPERT,
    IN_MLP_DOWN_OFFSET_SHARED_EXPERT,
    IN_MLP_DOWN_SCALE_SHARED_EXPERT};

inline const std::vector<std::string> LINEAR_FOR_ROPE = {
    "self_attn.q_b_proj.weight",
    "self_attn.q_b_proj.quant_bias",
    "self_attn.q_b_proj.deq_scale",
    "self_attn.kv_a_proj_with_mqa.weight",
    "self_attn.kv_a_proj_with_mqa.quant_bias",
    "self_attn.kv_a_proj_with_mqa.deq_scale",
};

}  // namespace deepseek_v2_decoder_constants

namespace deepseek_v32_decoder_constants {

enum DecoderLayerTensorId : int {
  IN_INPUT_NORM_WEIGHT = 0,
  IN_INPUT_NORM_BIAS = 1,
  IN_INPUT_NORM_NEW_WEIGHT = 2,
  IN_INPUT_NORM_NEW_BIAS = 3,

  IN_Q_PROJ_A_WEIGHT = 4,
  IN_Q_PROJ_A_BIAS = 5,
  IN_Q_PROJ_A_DESCALE = 6,
  IN_Q_PROJ_A_OFFSET = 7,
  IN_Q_PROJ_A_SCALE = 8,
  IN_Q_PROJ_A_COMPRESS_IDX = 9,
  IN_Q_PROJ_A_LAYERNORM_WEIGHT = 10,
  IN_Q_PROJ_A_LAYERNORM_BIAS = 11,

  IN_Q_PROJ_B_WEIGHT = 12,
  IN_Q_PROJ_B_BIAS = 13,
  IN_Q_PROJ_B_DESCALE = 14,
  IN_Q_PROJ_B_OFFSET = 15,
  IN_Q_PROJ_B_SCALE = 16,
  IN_Q_PROJ_B_COMPRESS_IDX = 17,

  IN_KV_PROJ_WITH_MQA_WEIGHT = 18,
  IN_KV_PROJ_WITH_MQA_BIAS = 19,
  IN_KV_PROJ_WITH_MQA_DESCALE = 20,
  IN_KV_PROJ_WITH_MQA_OFFSET = 21,
  IN_KV_PROJ_WITH_MQA_SCALE = 22,
  IN_KV_PROJ_WITH_MQA_COMPRESS_IDX = 23,

  IN_KV_PROJ_A_LAYERNORM_WEIGHT = 24,
  IN_KV_PROJ_A_LAYERNORM_BIAS = 25,

  IN_K_PROJ_B_FOR_Q_WEIGHT = 26,
  IN_K_PROJ_B_FOR_Q_BIAS = 27,
  IN_K_PROJ_B_FOR_Q_DESCALE = 28,
  IN_K_PROJ_B_FOR_Q_OFFSET = 29,
  IN_K_PROJ_B_FOR_Q_SCALE = 30,
  IN_K_PROJ_B_FOR_Q_COMPRESS_IDX = 31,

  IN_V_PROJ_B_FOR_O_WEIGHT = 32,
  IN_V_PROJ_B_FOR_O_BIAS = 33,
  IN_V_PROJ_B_FOR_O_DESCALE = 34,
  IN_V_PROJ_B_FOR_O_OFFSET = 35,
  IN_V_PROJ_B_FOR_O_SCALE = 36,
  IN_V_PROJ_B_FOR_O_COMPRESS_IDX = 37,

  IN_ATTENTION_OUT_WEIGHT = 38,
  IN_ATTENTION_OUT_BIAS = 39,
  IN_ATTENTION_OUT_DESCALE = 40,
  IN_ATTENTION_OUT_OFFSET = 41,
  IN_ATTENTION_OUT_SCALE = 42,
  IN_ATTENTION_OUT_COMPRESS_IDX = 43,

  IN_SELFATTENTION_OUT_NORM_WEIGHT = 44,
  IN_SELFATTENTION_OUT_NORM_BIAS = 45,
  IN_SELFATTENTION_OUT_NEW_NORM_WEIGHT = 46,
  IN_SELFATTENTION_OUT_NEW_NORM_BIAS = 47,

  IN_MLP_GATEUP_WEIGHT_SHARED_EXPERT = 48,
  IN_MLP_GATEUP_BIAS_SHARED_EXPERT = 49,
  IN_MLP_GATEUP_DESCALE_SHARED_EXPERT = 50,
  IN_MLP_GATEUP_OFFSET_SHARED_EXPERT = 51,
  IN_MLP_GATEUP_SCALE_SHARED_EXPERT = 52,
  IN_MLP_GATEUP_COMPRESS_IDX_SHARED_EXPERT = 53,

  IN_MLP_DOWN_WEIGHT_SHARED_EXPERT = 54,
  IN_MLP_DOWN_BIAS_SHARED_EXPERT = 55,
  IN_MLP_DOWN_DESCALE_SHARED_EXPERT = 56,
  IN_MLP_DOWN_OFFSET_SHARED_EXPERT = 57,
  IN_MLP_DOWN_SCALE_SHARED_EXPERT = 58,
  IN_MLP_DOWN_COMPRESS_IDX_SHARED_EXPERT = 59,

  IN_SHARED_EXPERT_GATE_WEIGHT = 60,
  IN_SHARED_EXPERT_GATE_BIAS = 61,
  IN_SHARED_EXPERT_GATE_DESCALE = 62,
  IN_SHARED_EXPERT_GATE_OFFSET = 63,
  IN_SHARED_EXPERT_GATE_SCALE = 64,
  IN_SHARED_EXPERT_GATE_COMPRESS_IDX = 65,

  IN_BLOCK_SPARSE_MOE_GATE_WEIGHT = 66,
  IN_BLOCK_SPARSE_MOE_GATE_BIAS = 67,
  IN_BLOCK_SPARSE_MOE_GATE_DESCALE = 68,
  IN_BLOCK_SPARSE_MOE_GATE_OFFSET = 69,
  IN_BLOCK_SPARSE_MOE_GATE_SCALE = 70,
  IN_BLOCK_SPARSE_MOE_GATE_COMPRESS_IDX = 71,

  IN_MLP_GATEUP_WEIGHT_EXPERT = 72,
  IN_MLP_GATEUP_BIAS_EXPERT = 73,
  IN_MLP_GATEUP_DESCALE_EXPERT = 74,
  IN_MLP_GATEUP_OFFSET_EXPERT = 75,
  IN_MLP_GATEUP_SCALE_EXPERT = 76,
  IN_MLP_GATEUP_COMPRESS_IDX_EXPERT = 77,

  IN_MLP_DOWN_WEIGHT_EXPERT = 78,
  IN_MLP_DOWN_BIAS_EXPERT = 79,
  IN_MLP_DOWN_DESCALE_EXPERT = 80,
  IN_MLP_DOWN_OFFSET_EXPERT = 81,
  IN_MLP_DOWN_SCALE_EXPERT = 82,
  IN_MLP_DOWN_COMPRESS_IDX_EXPERT = 83,

  IN_INDEXER_WQ_B_WEIGHT = 84,
  IN_INDEXER_WQ_B_BIAS = 85,
  IN_INDEXER_WQ_B_DESCALE = 86,
  IN_INDEXER_WQ_B_OFFSET = 87,
  IN_INDEXER_WQ_B_SCALE = 88,
  IN_INDEXER_WQ_B_COMPRESS_IDX = 89,

  IN_INDEXER_WK_WEIGHT = 90,
  IN_INDEXER_WK_BIAS = 91,
  IN_INDEXER_WK_DESCALE = 92,
  IN_INDEXER_WK_OFFSET = 93,
  IN_INDEXER_WK_SCALE = 94,
  IN_INDEXER_WK_COMPRESS_IDX = 95,

  IN_INDEXER_K_NORM_WEIGHT = 96,
  IN_INDEXER_K_NORM_BIAS = 97,

  IN_INDEXER_PROJ_WEIGHT = 98,
  IN_INDEXER_PROJ_BIAS = 99,
  IN_INDEXER_PROJ_DESCALE = 100,
  IN_INDEXER_PROJ_OFFSET = 101,
  IN_INDEXER_PROJ_SCALE = 102,
  IN_INDEXER_PROJ_COMPRESS_IDX = 103,
  IN_Q_PROJ_A_RECOMPUTE_WEIGHT = 104,
  IN_Q_PROJ_A_RECOMPUTE_BIAS = 105,
  IN_Q_PROJ_A_RECOMPUTE_DESCALE = 106,
  IN_Q_PROJ_A_RECOMPUTE_OFFSET = 107,
  IN_Q_PROJ_A_RECOMPUTE_SCALE = 108,
  IN_Q_PROJ_A_RECOMPUTE_COMPRESS_IDX = 109,
};

inline const std::vector<std::pair<int, std::string>> WEIGHT_MAPPING = {};

inline const std::unordered_map<std::string, int> WEIGHT_MAPPING_W8A8 = {
    {"input_layernorm.weight", IN_INPUT_NORM_WEIGHT},
    {"input_layernorm.bias", IN_INPUT_NORM_BIAS},

    {"self_attn.q_a_proj.weight", IN_Q_PROJ_A_WEIGHT},
    {"self_attn.q_a_proj.quant_bias", IN_Q_PROJ_A_BIAS},
    {"self_attn.q_a_proj.deq_scale", IN_Q_PROJ_A_DESCALE},
    {"self_attn.q_a_proj.input_offset", IN_Q_PROJ_A_OFFSET},
    {"self_attn.q_a_proj.input_scale", IN_Q_PROJ_A_SCALE},
    {"self_attn.q_a_proj.weight_offset", IN_Q_PROJ_A_OFFSET},
    {"self_attn.q_a_proj.weight_scale", IN_Q_PROJ_A_SCALE},
    {"self_attn.q_a_layernorm.weight", IN_Q_PROJ_A_LAYERNORM_WEIGHT},
    {"self_attn.q_a_layernorm.bias", IN_Q_PROJ_A_LAYERNORM_BIAS},

    {"self_attn.q_proj.weight", IN_Q_PROJ_B_WEIGHT},
    {"self_attn.q_b_proj.weight", IN_Q_PROJ_B_WEIGHT},
    {"self_attn.q_b_proj.quant_bias", IN_Q_PROJ_B_BIAS},
    {"self_attn.q_b_proj.input_scale", IN_Q_PROJ_B_SCALE},
    {"self_attn.q_b_proj.deq_scale", IN_Q_PROJ_B_DESCALE},
    {"self_attn.q_b_proj.input_offset", IN_Q_PROJ_B_OFFSET},
    {"self_attn.q_b_proj.weight_offset", IN_Q_PROJ_B_OFFSET},
    {"self_attn.q_b_proj.weight_scale", IN_Q_PROJ_B_SCALE},

    {"self_attn.kv_a_proj_with_mqa.weight", IN_KV_PROJ_WITH_MQA_WEIGHT},
    {"self_attn.kv_a_proj_with_mqa.quant_bias", IN_KV_PROJ_WITH_MQA_BIAS},
    {"self_attn.kv_a_proj_with_mqa.deq_scale", IN_KV_PROJ_WITH_MQA_DESCALE},
    {"self_attn.kv_a_proj_with_mqa.input_offset", IN_KV_PROJ_WITH_MQA_OFFSET},
    {"self_attn.kv_a_proj_with_mqa.input_scale", IN_KV_PROJ_WITH_MQA_SCALE},
    {"self_attn.kv_a_proj_with_mqa.weight_offset", IN_KV_PROJ_WITH_MQA_OFFSET},
    {"self_attn.kv_a_proj_with_mqa.weight_scale", IN_KV_PROJ_WITH_MQA_SCALE},

    {"self_attn.kv_a_layernorm.weight", IN_KV_PROJ_A_LAYERNORM_WEIGHT},
    {"self_attn.kv_a_layernorm.bias", IN_KV_PROJ_A_LAYERNORM_BIAS},

    {"self_attn.kv_b_proj.weight", IN_K_PROJ_B_FOR_Q_WEIGHT},  // merge
    // {"self_attn.kv_b_proj.weight", IN_V_PROJ_B_FOR_O_WEIGHT},  // merge

    {"self_attn.o_proj.weight", IN_ATTENTION_OUT_WEIGHT},
    {"self_attn.o_proj.quant_bias", IN_ATTENTION_OUT_BIAS},
    {"self_attn.o_proj.deq_scale", IN_ATTENTION_OUT_DESCALE},
    {"self_attn.o_proj.input_offset", IN_ATTENTION_OUT_OFFSET},
    {"self_attn.o_proj.input_scale", IN_ATTENTION_OUT_SCALE},
    {"self_attn.o_proj.weight_offset", IN_ATTENTION_OUT_OFFSET},
    {"self_attn.o_proj.weight_scale", IN_ATTENTION_OUT_SCALE},

    {"self_attn.indexer.wq_b.weight", IN_INDEXER_WQ_B_WEIGHT},
    {"self_attn.indexer.wq_b.quant_bias", IN_INDEXER_WQ_B_BIAS},
    {"self_attn.indexer.wq_b.deq_scale", IN_INDEXER_WQ_B_DESCALE},
    {"self_attn.indexer.wq_b.input_offset", IN_INDEXER_WQ_B_OFFSET},
    {"self_attn.indexer.wq_b.input_scale", IN_INDEXER_WQ_B_SCALE},
    {"self_attn.indexer.wq_b.weight_offset", IN_INDEXER_WQ_B_OFFSET},
    {"self_attn.indexer.wq_b.weight_scale", IN_INDEXER_WQ_B_SCALE},
    {"self_attn.indexer.wk.weight", IN_INDEXER_WK_WEIGHT},
    {"self_attn.indexer.k_norm.weight", IN_INDEXER_K_NORM_WEIGHT},
    {"self_attn.indexer.k_norm.bias", IN_INDEXER_K_NORM_BIAS},
    {"self_attn.indexer.weights_proj.weight", IN_INDEXER_PROJ_WEIGHT},

    {"post_attention_layernorm.weight", IN_SELFATTENTION_OUT_NORM_WEIGHT},
    {"post_attention_layernorm.bias", IN_SELFATTENTION_OUT_NORM_BIAS},

    {"mlp.gate_proj.weight", IN_MLP_GATEUP_WEIGHT_SHARED_EXPERT},
    {"mlp.gate_proj.weight_offset", IN_MLP_GATEUP_OFFSET_SHARED_EXPERT},
    {"mlp.gate_proj.weight_scale", IN_MLP_GATEUP_SCALE_SHARED_EXPERT},
    {"mlp.gate_proj.scale_bias", IN_MLP_GATEUP_BIAS_SHARED_EXPERT},

    {"mlp.up_proj.weight", IN_MLP_GATEUP_WEIGHT_SHARED_EXPERT},
    {"mlp.up_proj.weight_offset", IN_MLP_GATEUP_OFFSET_SHARED_EXPERT},
    {"mlp.up_proj.weight_scale", IN_MLP_GATEUP_SCALE_SHARED_EXPERT},
    {"mlp.up_proj.scale_bias", IN_MLP_GATEUP_BIAS_SHARED_EXPERT},

    {"mlp.down_proj.weight", IN_MLP_DOWN_WEIGHT_SHARED_EXPERT},
    {"mlp.down_proj.weight_offset", IN_MLP_DOWN_OFFSET_SHARED_EXPERT},
    {"mlp.down_proj.weight_scale", IN_MLP_DOWN_SCALE_SHARED_EXPERT},
    {"mlp.down_proj.scale_bias", IN_MLP_DOWN_BIAS_SHARED_EXPERT},

    {"mlp.shared_experts.gate_proj.weight", IN_MLP_GATEUP_WEIGHT_SHARED_EXPERT},
    {"mlp.shared_experts.gate_proj.weight_offset",
     IN_MLP_GATEUP_OFFSET_SHARED_EXPERT},
    {"mlp.shared_experts.gate_proj.weight_scale",
     IN_MLP_GATEUP_SCALE_SHARED_EXPERT},
    {"mlp.shared_experts.gate_proj.scale_bias",
     IN_MLP_GATEUP_BIAS_SHARED_EXPERT},

    {"mlp.shared_experts.up_proj.weight", IN_MLP_GATEUP_WEIGHT_SHARED_EXPERT},
    {"mlp.shared_experts.up_proj.weight_offset",
     IN_MLP_GATEUP_OFFSET_SHARED_EXPERT},
    {"mlp.shared_experts.up_proj.weight_scale",
     IN_MLP_GATEUP_SCALE_SHARED_EXPERT},
    {"mlp.shared_experts.up_proj.scale_bias", IN_MLP_GATEUP_BIAS_SHARED_EXPERT},

    {"mlp.shared_experts.down_proj.weight", IN_MLP_DOWN_WEIGHT_SHARED_EXPERT},
    {"mlp.shared_experts.down_proj.weight_offset",
     IN_MLP_DOWN_OFFSET_SHARED_EXPERT},
    {"mlp.shared_experts.down_proj.weight_scale",
     IN_MLP_DOWN_SCALE_SHARED_EXPERT},
    {"mlp.shared_experts.down_proj.scale_bias", IN_MLP_DOWN_BIAS_SHARED_EXPERT},

    {"mlp.gate.weight", IN_BLOCK_SPARSE_MOE_GATE_WEIGHT},
    {"mlp.gate.e_score_correction_bias", IN_BLOCK_SPARSE_MOE_GATE_BIAS},

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

inline const std::unordered_map<std::string, int>
    WEIGHT_MAPPING_W8A8_RECOMPUTE = {
        {"self_attn.q_a_proj.weight", IN_Q_PROJ_A_RECOMPUTE_WEIGHT},
        {"self_attn.q_a_proj.quant_bias", IN_Q_PROJ_A_RECOMPUTE_BIAS},
        {"self_attn.q_a_proj.deq_scale", IN_Q_PROJ_A_RECOMPUTE_DESCALE},
        {"self_attn.q_a_proj.input_offset", IN_Q_PROJ_A_RECOMPUTE_OFFSET},
        {"self_attn.q_a_proj.input_scale", IN_Q_PROJ_A_RECOMPUTE_SCALE},
        {"self_attn.q_a_proj.weight_offset", IN_Q_PROJ_A_RECOMPUTE_OFFSET},
        {"self_attn.q_a_proj.weight_scale", IN_Q_PROJ_A_RECOMPUTE_SCALE},
};

inline const std::map<int, int> WEIGHT_SHARD = {};

inline const std::map<int, int> WEIGHT_SHARD_W8A8 = {
    {IN_Q_PROJ_B_WEIGHT, 0},
    {IN_Q_PROJ_B_BIAS, 0},
    {IN_Q_PROJ_B_DESCALE, 0},
    {IN_K_PROJ_B_FOR_Q_WEIGHT, 0},
    {IN_V_PROJ_B_FOR_O_WEIGHT, 0},
    {IN_ATTENTION_OUT_WEIGHT, 1},
    {IN_MLP_GATEUP_WEIGHT_SHARED_EXPERT, 0},
    {IN_MLP_GATEUP_OFFSET_SHARED_EXPERT, 0},
    {IN_MLP_GATEUP_SCALE_SHARED_EXPERT, 0},
    {IN_MLP_DOWN_WEIGHT_SHARED_EXPERT, 1},
    {IN_MLP_GATEUP_WEIGHT_EXPERT, 0},
    {IN_MLP_GATEUP_OFFSET_EXPERT, 0},
    {IN_MLP_GATEUP_SCALE_EXPERT, 0},
    {IN_MLP_DOWN_WEIGHT_EXPERT, 1},
};

inline const std::vector<int> SQUEEZE_WEIGHT_VEC = {
    IN_MLP_GATEUP_OFFSET_SHARED_EXPERT,
    IN_MLP_GATEUP_SCALE_SHARED_EXPERT,
    IN_MLP_DOWN_OFFSET_SHARED_EXPERT,
    IN_MLP_DOWN_SCALE_SHARED_EXPERT};

inline const std::vector<std::string> LINEAR_FOR_ROPE = {
    "self_attn.q_b_proj.weight",
    "self_attn.q_b_proj.quant_bias",
    "self_attn.q_b_proj.deq_scale",
    "self_attn.kv_a_proj_with_mqa.weight",
    "self_attn.kv_a_proj_with_mqa.quant_bias",
    "self_attn.kv_a_proj_with_mqa.deq_scale",
};

}  // namespace deepseek_v32_decoder_constants

}  // namespace layer
}  // namespace xllm

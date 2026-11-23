/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 * MindIE is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan
 * PSL v2. You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY
 * KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
 * NON-INFRINGEMENT, MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE. See the
 * Mulan PSL v2 for more details.
 */
#ifndef __MATMUL_CONST_BF16_H__
#define __MATMUL_CONST_BF16_H__

constexpr int AICFLAGID = 0;
constexpr int AIVFLAGID = 1;
constexpr int AIC2AIVFLAGID = 2;
constexpr int AIV2AICFLAGID = 3;
constexpr int CUBE_PV_READY_FLAG = 4;
constexpr int VECTOR_OUTPUT_READY_FLAG = 5;

using T_OUTPUT = float;
constexpr int32_t L0AB_PINGPONG_BUFFER_LEN = 16384;  // 32 KB
constexpr int32_t BLOCK_SIZE = 16;
constexpr int32_t CUBE_MATRIX_SIZE = 256;           // 16 * 16
constexpr int64_t L1_PINGPONG_BUFFER_LEN = 16384;   // 32 KB
constexpr int64_t L0C_PINGPONG_BUFFER_LEN = 16384;  // 64 KB

constexpr int32_t BASE_BLOCK_SIZE = 16384;    // BASE_BLOCK shape ：[128 * 128]
constexpr int32_t BASE_BLOCK_SIDE_LEN = 128;  // BASE_BLOCK  row and column size

constexpr int32_t B16_SIZE = 2;
constexpr int32_t B32_SIZE = 4;
constexpr int32_t CUBE2_LENGTH_M = 128;
constexpr int32_t CUBE2_LENGTH_K = 128;
constexpr int32_t CUBE2_LENGTH_N = 128;
constexpr int32_t MAX_SWITCH_TIME =
    8;  // 一个core最多处理16个基本块，因此最多只能有16段 17

constexpr int32_t BASE_BLOCK_LENGTH = 128;
// 基本块是方阵，长和宽
constexpr int BASE_BLOCK_SIZE_DOUBLE = BASE_BLOCK_SIDE_LEN * 2;
constexpr int HEAD_DIM = 128;  // head的维度
constexpr int BASE_BLOCK_DATA_NUM =
    BASE_BLOCK_SIDE_LEN * BASE_BLOCK_SIDE_LEN;  // 基本块含有数据量
constexpr int MAX_LENG_PER_UB_PROC = 16384;  // UB一次处理的最大长度 (单个ping)
constexpr int MAX_BLOCK_PER_ONE_PROC =
    MAX_LENG_PER_UB_PROC / BASE_BLOCK_SIDE_LEN;
constexpr int BLOCK_NUM_FOR_VMAX = 16;  // 折半计算的基准block数量
constexpr int SHORT_SEQ_THRESHOLD = 8192;
constexpr int MDDIUM_SEQ_THRESHOLD = 32768;

// backward vector
constexpr int BASE_BLOCK_SIZE_LEN_BACKWARD =
    128;  // 基本块是方阵，长和宽        // head的维度
// 基本块含有数据量
constexpr int BASE_BLOCK_DATA_NUM_BACKWARD =
    BASE_BLOCK_SIZE_LEN_BACKWARD * BASE_BLOCK_SIZE_LEN_BACKWARD;
constexpr int MAX_LENG_PER_UB_PROC_BACKWARD =
    4096;  // UB一次处理的最大长度 (单个ping)
constexpr int MAX_BLOCK_PER_ONE_PROC_BACKWARD =
    MAX_LENG_PER_UB_PROC_BACKWARD / BASE_BLOCK_SIZE_LEN_BACKWARD;  // 行数
constexpr int BASIC_GAP_BACKWARD =
    BASE_BLOCK_DATA_NUM_BACKWARD - BASE_BLOCK_SIZE_LEN_BACKWARD;

constexpr int BASIC_GAP = BASE_BLOCK_DATA_NUM - BASE_BLOCK_SIDE_LEN;

constexpr float PADDING_FOR_MAX =
    -1e30;  // 非2的幂长度，折半计算vmax时，需要padding
constexpr half PADDING_FOR_MAX2 = -65504;
constexpr int PADDING_TYPE_ROWMAX = 0;  // 填充PADDING_FOR_MAX最小值用于求vmax
constexpr int PADDING_TYPE_ROWSUM = 1;  // 填充0用于求vadd

constexpr int TRI_MATRIX_NONE = 0;
constexpr int TRI_MATRIX_TAIL = 1;
constexpr int TRI_MATRIX_HEAD = 2;
constexpr int TRI_MATRIX_HEAD_AND_TAIL = 3;

template <typename TYPE>
struct PhyAddr {
  __gm__ TYPE* left;
  __gm__ TYPE* right;
  __gm__ TYPE* out;
  int32_t k = 0;
};

template <typename TYPE>
struct PhyAddrCube2 {
  __gm__ TYPE* left;
  __gm__ TYPE* right;
  __gm__ float* out;
  int32_t k = 0;
};

// template<typename TYPE = __bf16>
template <typename TYPE, typename WORKSPACE_TYPE>
struct PhyAddrCube3 {
  __gm__ WORKSPACE_TYPE* left;
  __gm__ TYPE* right;
  __gm__ T_OUTPUT* out;
  int32_t k = 0;
};

struct Addr {
  int32_t b;
  int32_t n;
  int32_t i_r;
  int32_t i_c;
  int32_t k;
};

template <typename TYPE, typename WORKSPACE_TYPE>
struct PhyAddrCube1 {
  __gm__ TYPE* left;
  __gm__ TYPE* right;
  __gm__ WORKSPACE_TYPE* out;
  int32_t k = 0;
};

template <typename TYPE, typename WORKSPACE_TYPE>
struct PhyAddrCube2Rowsum {
  __gm__ WORKSPACE_TYPE* left;
  __gm__ TYPE* right;
  __gm__ float* out;
  __gm__ float* rowsum_out;
  int32_t k = 0;
};

#endif

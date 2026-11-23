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
#ifndef __CUBEFORWARD_H__
#define __CUBEFORWARD_H__

#include <cstdint>
#include <limits>

// clang-format off
#include "matmul_const.h"
#include "AddressMappingForwardOnline.h"
#include "kernel_operator.h"
// clang-format on

#ifdef __DAV_C220_CUBE__
namespace CUBE_FORWARD_ONLINE {

constexpr int32_t SIZE_16 = 16;
constexpr int32_t SIZE_32 = 32;
constexpr int32_t SIZE_256 = 256;
constexpr int32_t SIZE_128 = 128;

struct Addr_Matmul_Fp32 {
  __gm__ float* left;
  __gm__ float* right;
  __gm__ float* out;
};

template <typename TYPE, bool IF_BF16, typename WORKSPACE_TYPE>
class CubeForward {
 public:
  __aicore__ inline CubeForward(){};
  __aicore__ inline void Init(__gm__ uint8_t* __restrict__ gm_Q,
                              __gm__ uint8_t* __restrict__ gm_K,
                              __gm__ uint8_t* __restrict__ gm_V,
                              __gm__ uint8_t* __restrict__ gm_S,
                              __gm__ uint8_t* __restrict__ gm_P,
                              __gm__ float* __restrict__ gm_O,
                              __gm__ float* __restrict__ gm_rowsum_diag,
                              __gm__ float* __restrict__ gm_rowmax_diag,
                              __gm__ float* __restrict__ gm_rowsum,
                              int32_t Y,
                              int32_t F,
                              int32_t B,
                              int32_t N,
                              int32_t S1,
                              int32_t S2,
                              int32_t D,
                              int32_t nG,
                              int32_t qk_triangle,
                              int32_t sparseMode,
                              int32_t window_length);
  __aicore__ inline void Run();
  __aicore__ inline void PresetFlag();
  __aicore__ inline void ClearFlag();

 private:
  // CUBE1
  template <typename T_LEFT, typename T_RIGHT, typename T_OUTPUT>
  __aicore__ __inline__ void cube1_matmul_op(
      const Address::PhyAddrForwardCube1Online<T_LEFT, T_RIGHT, T_OUTPUT>* src,
      int64_t src_len);

  __aicore__ __inline__ void cube1_base_matmul(
      __cbuf__ TYPE* l1_a,
      __cbuf__ TYPE* l1_b,
      __gm__ WORKSPACE_TYPE* gm_out,
      int32_t ky,
      int32_t out_put_matrix_line_strid,
      bool upper_right_flag);

  __aicore__ __inline__ void matmul_fp32(Addr_Matmul_Fp32 addr,
                                         int32_t ping_flag,
                                         bool copy_out,
                                         int32_t ky_idx);

  __aicore__ __inline__ void diag_matmul(__gm__ float* gm_base,
                                         __gm__ float* gm_diag,
                                         int32_t ky,
                                         bool copy_out);

  template <typename T_LEFT, typename T_RIGHT, typename T_OUTPUT>
  __aicore__ __inline__ void cube2_matmul_op(
      const Address::PhyAddrForwardCube2Online<T_LEFT, T_RIGHT, T_OUTPUT>* src,
      int64_t src_len,
      int64_t vcore_num_per_head,
      int64_t roundId);
  // CUBE2

 private:
  __gm__ TYPE* __restrict__ gm_Q;  // gm_a_cube1;
  __gm__ TYPE* __restrict__ gm_K;  // gm_b_cube1;
  __gm__ TYPE* __restrict__ gm_V;  // gm_b_cube2;
  __gm__ TYPE* __restrict__ gm_P;
  __gm__ WORKSPACE_TYPE* __restrict__ gm_S;  // gm_c_cube1;
  __gm__ float* __restrict__ gm_O;
  __gm__ TYPE* __restrict__ gm_ones;

  __gm__ float* __restrict__ rowsum_gm;
  __gm__ float* __restrict__ rowsum_diag;
  __gm__ float* __restrict__ rowmax_diag;
  // L1_base for CUBE1 + CUBE2
  __cbuf__ TYPE* l1_base_b_cube1 =
      reinterpret_cast<__cbuf__ TYPE*>((uintptr_t)(128 * 1024));  // 128 KB
  __cbuf__ TYPE* l1_base_a_cube2 =
      reinterpret_cast<__cbuf__ TYPE*>((uintptr_t)(256 * 1024));  // 26 KB
  __cbuf__ TYPE* l1_base_b_cube2 =
      reinterpret_cast<__cbuf__ TYPE*>((uintptr_t)(384 * 1024));  // 208 KB

  __cbuf__ TYPE* l1_a_ping_ =
      reinterpret_cast<__cbuf__ TYPE*>((uintptr_t)0);  // 64 KB
  __cbuf__ TYPE* l1_a_pong_ =
      reinterpret_cast<__cbuf__ TYPE*>((uintptr_t)(64 * 1024));  // 64 KB
  __cbuf__ TYPE* l1_b_ping_ =
      reinterpret_cast<__cbuf__ TYPE*>((uintptr_t)128 * 1024);  // 64 KB
  __cbuf__ TYPE* l1_b_pong_ =
      reinterpret_cast<__cbuf__ TYPE*>((uintptr_t)(192 * 1024));  // 64 KB

  __cbuf__ float* l1_local_diag_ping_ =
      reinterpret_cast<__cbuf__ float*>(256 * 1024);  // 64 KB
  __cbuf__ float* l1_local_diag_pong_ =
      reinterpret_cast<__cbuf__ float*>(320 * 1024);  // 64 KB
  __cbuf__ float* l1_local_attention_ping_ =
      reinterpret_cast<__cbuf__ float*>(384 * 1024);  // 64 KB
  __cbuf__ float* l1_local_attention_pong_ =
      reinterpret_cast<__cbuf__ float*>(448 * 1024);  // 64 KB

  uint32_t l1_a_ping_pong_flag_ = 0;
  uint32_t l1_b_ping_pong_flag_ = 0;

  __ca__ TYPE* l0_a_ping_ = reinterpret_cast<__ca__ TYPE*>((uintptr_t)0);
  __ca__ TYPE* l0_a_pong_ =
      reinterpret_cast<__ca__ TYPE*>((uintptr_t)(32 * 1024));
  __cb__ TYPE* l0_b_ping_ = reinterpret_cast<__cb__ TYPE*>((uintptr_t)0);
  __cb__ TYPE* l0_b_pong_ =
      reinterpret_cast<__cb__ TYPE*>((uintptr_t)(32 * 1024));
  __cc__ float* l0_c_ping_ = reinterpret_cast<__cc__ float*>((uintptr_t)0);
  __cc__ float* l0_c_pong_ =
      reinterpret_cast<__cc__ float*>((uintptr_t)(64 * 1024));

  __ca__ float* l0a_diag = reinterpret_cast<__ca__ float*>((uintptr_t)0);
  __cb__ float* l0b_out = reinterpret_cast<__cb__ float*>((uintptr_t)0);
  __cc__ float* l0c_base = reinterpret_cast<__cc__ float*>((uintptr_t)0);

  uint32_t l0_a_ping_pong_flag_ = 0;
  uint32_t l0_b_ping_pong_flag_ = 0;
  uint32_t l0_c_ping_pong_flag_ = 0;

  uint32_t ping_pong_flag_l1_a_ = 0;
  uint32_t ping_pong_flag_l1_b_ = 0;
  uint32_t ping_pong_flag_l0_a_ = 0;
  uint32_t ping_pong_flag_l0_b_ = 0;
  uint32_t ping_pong_flag_l0_c_ = 0;
  // L0A L0B for CUBE1 + CUBE2   公用
  __ca__ TYPE* l0a_base = reinterpret_cast<__ca__ TYPE*>((uintptr_t)0);
  __cb__ TYPE* l0b_base = reinterpret_cast<__cb__ TYPE*>((uintptr_t)0);
  __cc__ float* l0c_buf = reinterpret_cast<__cc__ float*>((uintptr_t)0);

  // Y个core处理一个完整行，所有core分成 F 组， block_dim = F * Y
  int32_t Y;
  int32_t F;

  // Q K V shape : [B,N,S,D] 其中 D 固定为128
  int32_t B;
  int32_t N;
  int32_t S1;  // 256 - 128k (256的倍数) 方阵的行
  int32_t S2;  // 256 - 128k (256的倍数) 方阵的行
  int32_t D;
  int32_t nG;
  int32_t G;              // GQA 场景  k v shape [B,G,S,D]
  int32_t qk_triangle;    // GQA 场景  k v shape [B,G,S,D]
  int32_t sparseMode;     // sparseMode: 0:dense, 1:sparse
  int32_t window_length;  // sparse场景下，滑动窗口的长度

  int32_t H1;
  int32_t H2;
  int32_t L;  // 负责均衡
  int32_t W;
  int32_t Cols;
  int32_t Remain;

  int32_t cur_core_index;
  int32_t local_block_index;  // 组内 core id [0, Y-1], 组内第几个 0
  int32_t core_group_index;   // [0, F-1], 第几组

  int32_t row_per_batch;
  int32_t column_per_core;
  int32_t column_remain;

  bool isHighPrecision = true;
  bool is_syn = false;
  int32_t last_idx = 0;

  // 注入前向寻址模块
  Address::AddressMappingForwardOnline<TYPE> address;
};

template <typename TYPE, bool IF_BF16, typename WORKSPACE_TYPE>
template <typename T_LEFT, typename T_RIGHT, typename T_OUTPUT>
__aicore__ __inline__ void
CubeForward<TYPE, IF_BF16, WORKSPACE_TYPE>::cube2_matmul_op(
    const Address::PhyAddrForwardCube2Online<T_LEFT, T_RIGHT, T_OUTPUT>* src,
    int64_t src_len,
    int64_t vcore_num_per_head,
    int64_t roundId) {
  int64_t l1_m = CUBE2_LENGTH_M;               // 128
  int64_t l1_k = CUBE2_LENGTH_K;               // 128
  int64_t l1_n = CUBE2_LENGTH_N;               // 128
  int64_t l0_k = BASE_BLOCK_LENGTH;            // 128
  int64_t l0_m = BASE_BLOCK_LENGTH;            // 128
  int64_t l0_n = BASE_BLOCK_LENGTH;            // 128
  int64_t l0_k_block_num = l0_k / BLOCK_SIZE;  // 16
  int64_t l0_n_block_num = l0_n / BLOCK_SIZE;  // 16

  int64_t SIZE_256 = 256;
  int64_t SIZE_128 = 128;
  int64_t SIZE_16 = 16;
  auto n_ = 128;
  auto n0_ = 128;
  auto m0_ = 128;
  auto k0_ = 128;

  for (int32_t idx = 0; idx < src_len; ++idx) {
    // 寻址的参数
    auto left_start_addr = src[idx].left;
    auto right_start_addr = src[idx].right;
    auto result_gm = src[idx].out;
    int32_t Ky = src[idx].ky;
    int32_t Kx = src[idx].kx;
    int32_t line_stride = src[idx].lineStride;
    bool upper_right = src[idx].upperRight;
    bool lower_left = false;

    auto curr_rowmax = rowmax_diag + ((roundId % 2) * MAX_SWITCH_TIME + idx) *
                                         BASE_BLOCK_SIZE * Ky;
    auto curr_rowsum = rowsum_diag + ((roundId % 2) * MAX_SWITCH_TIME + idx) *
                                         BASE_BLOCK_SIZE * Ky;

    if constexpr (!IF_BF16) {
      if (!src[idx].onStartSection) {
        diag_matmul(result_gm, curr_rowmax, 2, false);
      }
    }

    // 外循环按x方向遍历
    for (int32_t i = 0; i < Kx; i++) {
      bool l1_skip_flag = (upper_right && i == Kx - 1);
      bool last_k = (i >= Kx - 1);
      int l0_c_init_flag =
          (i == 0 && (IF_BF16 || src[idx].onStartSection)) ? 1 : 0;
      // pingpong设置
      auto l1_a_buf = ping_pong_flag_l1_a_ ? l1_a_pong_ : l1_a_ping_;
      auto l1_b_buf = ping_pong_flag_l1_b_ ? l1_b_pong_ : l1_b_ping_;

      // 左矩阵按列的方向一次搬运
      wait_flag(PIPE_MTE1, PIPE_MTE2, ping_pong_flag_l1_b_);
      wait_flag(PIPE_MTE1, PIPE_MTE2, ping_pong_flag_l1_a_ + 2);

      if (!l1_skip_flag) {
        copy_gm_to_cbuf_multi_nd2nz_b16(
            l1_a_buf,
            (__gm__ TYPE*)(left_start_addr +
                           i * (IF_BF16 ? l1_k : l1_m * l1_k)),
            0,                                   // sid
            vcore_num_per_head,                  // ndNum
            l1_m / vcore_num_per_head,           // nValue   实际拷贝的行数
            l1_k,                                // dValue   实际拷贝的列数
            128 / vcore_num_per_head * 128 * 2,  // srcNdMatrixStride, unused
            (IF_BF16 ? Kx * l1_k : l1_k),        // srcDValue 大矩阵的列数
            l1_m,                                // dstNzC0Stride 目标行数
            1,  // dstNzNStride  目标行之间间隔（1为连续）
            128 * BLOCK_SIZE / vcore_num_per_head  // dstNzMatrixStride, unused
        );
      }

      copy_gm_to_cbuf_multi_nd2nz_b16(
          l1_a_buf + SIZE_128 * SIZE_128,
          (__gm__ TYPE*)(left_start_addr + i * (IF_BF16 ? l1_k : l1_k * l1_m) +
                         line_stride),
          0,                          // sid
          vcore_num_per_head,         // ndNum
          l1_m / vcore_num_per_head,  // nValue   实际拷贝的行数
          l1_k,                       // dValue   实际拷贝的列数
          128 / vcore_num_per_head * 128 *
              2,                         // srcNdMatrixStride, 拷贝两块之间间隔
          (IF_BF16 ? Kx * l1_k : l1_k),  // srcDValue 大矩阵的列数
          l1_m,                          // dstNzC0Stride 目标行数
          1,  // dstNzNStride  目标行之间间隔（1为连续）
          128 * BLOCK_SIZE / vcore_num_per_head  // dstNzMatrixStride, unused
      );

      // 右矩阵搬运搬运一块
      copy_gm_to_cbuf_multi_nd2nz_b16(
          l1_b_buf,
          right_start_addr + i * l1_k * n_,
          0,     // sid
          1,     // ndNum
          l1_k,  // nValue   实际拷贝的行数
          l1_n,  // dValue   实际拷贝的列数
          0,     // srcNdMatrixStride, unused
          n_,    // srcDValue 大矩阵的列数
          l1_k,  // dstNzC0Stride 目标行数
          1,     // dstNzNStride  目标行之间间隔（1为连续）
          0      // dstNzMatrixStride, unused
      );
      set_flag(PIPE_MTE2, PIPE_MTE1, ping_pong_flag_l1_b_);
      wait_flag(PIPE_MTE2, PIPE_MTE1, ping_pong_flag_l1_b_);
      set_flag(PIPE_MTE2, PIPE_MTE1, ping_pong_flag_l1_a_ + 2);
      wait_flag(PIPE_MTE2, PIPE_MTE1, ping_pong_flag_l1_a_ + 2);

      for (int32_t n_offset = 0; n_offset < l1_n; n_offset += SIZE_128) {
        auto l0_b_buf = ping_pong_flag_l0_b_ ? l0_b_pong_ : l0_b_ping_;
        // 右矩阵L0B常驻：
        wait_flag(PIPE_M, PIPE_MTE1, ping_pong_flag_l0_b_ + 2);
        for (int nn = 0; nn < k0_ / SIZE_16; nn++) {
          load_cbuf_to_cb(l0_b_buf + nn * n0_ * SIZE_16,
                          l1_b_buf + n_offset * l1_k + nn * SIZE_256,
                          0,               // baseIdx
                          n0_ / SIZE_16,   // repeat
                          l1_k / SIZE_16,  // srcStride  连续读为1
                          0,               // dstStride  连续写为0
                          0,               // sid
                          true,            // transpose
                          inc              // addr_cal_mode_t
          );
        }
        set_flag(PIPE_MTE1, PIPE_M, ping_pong_flag_l0_b_ + 2);
        wait_flag(PIPE_MTE1, PIPE_M, ping_pong_flag_l0_b_ + 2);

        // 内循环按y方向遍历
        for (int32_t j = 0; j < Ky; j++) {
          bool l0_skip_flag = (l1_skip_flag && j == 0);
          bool last_k =
              (i == Kx - 1 && j != 0) ||
              (i == Kx - 1 && j == 0 && !upper_right) ||
              (upper_right && i == Kx - 2 && j == 0);  // 控制搬出的标志
          auto l0_a_buf = ping_pong_flag_l0_a_ ? l0_a_pong_ : l0_a_ping_;
          auto l0_c_buf =
              j == 0 ? l0_c_pong_ : l0_c_ping_;  // ping_pong_flag_l0_c_

          // 左矩阵pingpong
          wait_flag(PIPE_M, PIPE_MTE1, ping_pong_flag_l0_a_);
          if (!l0_skip_flag) {
            for (int32_t jj = 0; jj < m0_ / SIZE_16; jj++) {
              load_cbuf_to_ca(
                  l0_a_buf + jj * k0_ * SIZE_16,
                  l1_a_buf + j * SIZE_128 * SIZE_128 + jj * SIZE_256,
                  0,               // baseIdx
                  k0_ / SIZE_16,   // repeat
                  l1_m / SIZE_16,  // srcStride
                  0,               // dstStride
                  0,               // sid
                  false,           // transpose
                  inc              // addr_cal_mode_t
              );
            }
          }
          set_flag(PIPE_MTE1, PIPE_M, ping_pong_flag_l0_a_);
          // matmual
          wait_flag(PIPE_MTE1, PIPE_M, ping_pong_flag_l0_a_);
          mad(l0_c_buf,
              l0_a_buf,
              l0_b_buf,
              m0_,              // m
              k0_,              // k
              n0_,              // n
              last_k ? 3 : 2,   // unitFlag
              0,                // kDirectionAlign  fp32矩阵乘相关配置
              0,                // cmatrixSource   l0c地址模式，默认1
              l0_c_init_flag);  // cmatrixInitVal  1 l0c初始为0

          if (last_k) {
            // 无法判断那个核先算完，只能将GM初始为0，全部做atomic add
            copy_matrix_cc_to_gm((IF_BF16 ? curr_rowmax : result_gm) +
                                     j * SIZE_128 * n_ + n_offset,
                                 l0_c_buf,
                                 0,        // sid
                                 n0_,      // NSize  结果矩阵的列
                                 m0_,      // MSize  结果矩阵的行
                                 n_,       // dstStride_dst_D  结果大矩阵的列
                                 m0_,      // srcStride
                                 3,        // UnitFlagMode
                                 NoQuant,  // QuantPRE
                                 0,        // ReLUPRE
                                 false,    // channelSplit
                                 true      // NZ2ND_EN
            );
            set_flag(PIPE_M, PIPE_MTE2, EVENT_ID7);
            wait_flag(PIPE_M, PIPE_MTE2, EVENT_ID7);
          }

          set_flag(PIPE_M, PIPE_MTE1, ping_pong_flag_l0_a_);
          ping_pong_flag_l0_a_ = 1 - ping_pong_flag_l0_a_;
          ping_pong_flag_l0_c_ = 1 - ping_pong_flag_l0_c_;
        }
        set_flag(PIPE_M, PIPE_MTE1, ping_pong_flag_l0_b_ + 2);
        ping_pong_flag_l0_b_ = 1 - ping_pong_flag_l0_b_;
      }
      set_flag(PIPE_MTE1, PIPE_MTE2, ping_pong_flag_l1_b_);
      set_flag(PIPE_MTE1, PIPE_MTE2, ping_pong_flag_l1_a_ + 2);

      ping_pong_flag_l1_a_ = 1 - ping_pong_flag_l1_a_;
      ping_pong_flag_l1_b_ = 1 - ping_pong_flag_l1_b_;
    }
    if constexpr (!IF_BF16) {
      if (src[idx].onEndSection) {
        diag_matmul(src[idx].out, curr_rowsum, src[idx].ky, true);
      }
    }
    last_idx = src_len - 1;
  }
}

template <typename TYPE, bool IF_BF16, typename WORKSPACE_TYPE>
__aicore__ __inline__ void
CubeForward<TYPE, IF_BF16, WORKSPACE_TYPE>::matmul_fp32(Addr_Matmul_Fp32 addr,
                                                        int32_t ping_flag,
                                                        bool copy_out,
                                                        int32_t ky_idx) {
  auto l0_c_buf = ky_idx == 0 ? l0_c_pong_ : l0_c_ping_;
  __cbuf__ float* l1a_buf =
      ping_flag ? l1_local_diag_ping_ : l1_local_diag_pong_;
  __cbuf__ float* l1b_buf =
      ping_flag ? l1_local_attention_ping_ : l1_local_attention_pong_;

  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID7);
  wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID7);

  copy_gm_to_cbuf_multi_nd2nz_b32s(
      l1a_buf, addr.left, 0, 1, 128, 128, 0, 128, 128, 1, 0);
  constexpr int R0 = 16;
  copy_gm_to_cbuf_multi_nd2nz_b32s(l1b_buf,
                                   addr.right,
                                   0,
                                   128 / 16,
                                   R0,
                                   128,
                                   R0 * 128,
                                   128,
                                   R0,
                                   1,
                                   R0 * 128);

  int32_t ca_loops = 128 / 16;
  set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID7);
  wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID7);
  set_flag(PIPE_M, PIPE_MTE1, EVENT_ID7);
  wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID7);

  for (int32_t row = 0; row < ca_loops; row++) {
    load_cbuf_to_ca(l0a_diag + row * 16 * 16 * 8,
                    l1a_buf + row * 16 * 8,
                    0,
                    16,
                    8,
                    0,
                    0,
                    false,
                    inc);
  }

  for (int32_t row = 0; row < 8; row++) {
    load_cbuf_to_cb_transpose(l0b_out + row * 16 * 16 * 8,  // dst,
                              l1b_buf + row * 16 * 16 * 8,  // src,
                              0,                            // indexID,
                              8,                            // repeat,
                              1,                            // srcStride,
                              0,                            // dstStride,
                              inc,                          // addrmode,
                              7                             // dstFracStride
    );
  }
  set_flag(PIPE_MTE1, PIPE_M, EVENT_ID7);
  wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID7);

  mad(l0_c_buf,
      l0a_diag,
      l0b_out,
      128,
      128,
      128,
      copy_out ? 0b11 : 0b10,
      0,
      0,
      true  // y == 0 ? true : false
  );

  if (copy_out) {
    copy_matrix_cc_to_gm(addr.out,
                         l0_c_buf,
                         0,
                         128,
                         128,
                         128,
                         128,
                         0b11,
                         NoQuant,
                         0,
                         false,
                         true);
  }

  set_flag(PIPE_M, PIPE_MTE1, EVENT_ID7);
  wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID7);
}

template <typename TYPE, bool IF_BF16, typename WORKSPACE_TYPE>
__aicore__ __inline__ void
CubeForward<TYPE, IF_BF16, WORKSPACE_TYPE>::diag_matmul(__gm__ float* gm_base,
                                                        __gm__ float* gm_diag,
                                                        int32_t ky,
                                                        bool copy_out) {
  int32_t ping_flag = 1;
  set_flag(PIPE_M, PIPE_MTE2, EVENT_ID7);
  wait_flag(PIPE_M, PIPE_MTE2, EVENT_ID7);

  set_flag(PIPE_FIX, PIPE_MTE2, EVENT_ID7);
  wait_flag(PIPE_FIX, PIPE_MTE2, EVENT_ID7);

  for (int32_t y = 0; y < 2; y++) {
    Addr_Matmul_Fp32 addr;
    int32_t offset = y * BASE_BLOCK_SIZE;
    addr.left = gm_diag + offset;
    addr.right = gm_base + offset;
    addr.out = gm_base + offset;
    matmul_fp32(addr, ping_flag, copy_out, y);
    ping_flag = 1 - ping_flag;
  }
}

template <typename TYPE, bool IF_BF16, typename WORKSPACE_TYPE>
template <typename T_LEFT, typename T_RIGHT, typename T_OUTPUT>
__aicore__ __inline__ void
CubeForward<TYPE, IF_BF16, WORKSPACE_TYPE>::cube1_matmul_op(
    const Address::PhyAddrForwardCube1Online<T_LEFT, T_RIGHT, T_OUTPUT>* src,
    int64_t src_len) {
  for (int32_t idx = 0; idx < src_len; ++idx) {
    auto kx = src[idx].kx;
    auto ky = src[idx].ky;
    auto line_stride = src[idx].lineStride;
    int32_t m_loop = 1;   // 512/ 512 = 1
    int32_t n_loop = kx;  // 512/128 = 4
    auto l1_m_size_ = ky * 128;
    auto l1_k_size_ = 128;
    auto l1_n_size_ = 128;
    auto k_ = 128;
    auto n_ = 128;

    auto gm_a = src[idx].left;
    auto gm_b = src[idx].right;
    auto gm_c = src[idx].out;
    bool upper_right = src[idx].upperRight;
    bool lower_left = src[idx].lowerLeft;

    for (int m_index = 0; m_index < m_loop; m_index++) {
      // Load A to L1
      auto l1_a = ping_pong_flag_l1_a_ ? l1_a_pong_ : l1_a_ping_;
      wait_flag(PIPE_MTE1, PIPE_MTE2, ping_pong_flag_l1_a_ + 2);
      copy_gm_to_cbuf_multi_nd2nz_b16(
          l1_a,
          gm_a,
          0,           // sid
          1,           // ndNum
          l1_m_size_,  // nValue   实际拷贝的行数 256
          l1_k_size_,  // dValue   实际拷贝的列数 128
          0,           // srcNdMatrixStride, unused
          l1_k_size_,  // srcDValue 大矩阵的列数
          l1_m_size_,  // dstNzC0Stride 目标行数
          1,           // dstNzNStride  目标行之间间隔（1为连续）
          0            // dstNzMatrixStride, unused
      );
      set_flag(PIPE_MTE2, PIPE_MTE1, ping_pong_flag_l1_a_ + 2);
      wait_flag(PIPE_MTE2, PIPE_MTE1, ping_pong_flag_l1_a_ + 2);

      for (int n_index = 0; n_index < n_loop; n_index++) {
        bool upper_right_flag = (upper_right && n_index == n_loop - 1);
        auto l1_b = ping_pong_flag_l1_b_ ? l1_b_pong_ : l1_b_ping_;

        wait_flag(PIPE_MTE1, PIPE_MTE2, ping_pong_flag_l1_b_);

        // Load B to L1
        copy_gm_to_cbuf_multi_nd2nz_b16(
            l1_b,
            gm_b + n_index * l1_n_size_ * k_,
            0,           // sid
            1,           // ndNum
            l1_k_size_,  // nValue   实际拷贝的行数
            l1_n_size_,  // dValue   实际拷贝的列数
            0,           // srcNdMatrixStride, unused
            l1_n_size_,  // srcDValue 大矩阵的列数
            l1_k_size_,  // dstNzC0Stride 目标行数
            1,           // dstNzNStride  目标行之间间隔（1为连续）
            0            // dstNzMatrixStride, unused
        );

        set_flag(PIPE_MTE2, PIPE_MTE1, ping_pong_flag_l1_b_);
        wait_flag(PIPE_MTE2, PIPE_MTE1, ping_pong_flag_l1_b_);

        cube1_base_matmul(l1_a, l1_b, gm_c, ky, line_stride, upper_right_flag);

        set_flag(PIPE_MTE1, PIPE_MTE2, ping_pong_flag_l1_b_);
        ping_pong_flag_l1_b_ = 1 - ping_pong_flag_l1_b_;
        gm_c += IF_BF16 ? SIZE_128 : SIZE_128 * SIZE_128;
        // gm_b +=
      }

      set_flag(PIPE_MTE1, PIPE_MTE2, ping_pong_flag_l1_a_ + 2);
      ping_pong_flag_l1_a_ = 1 - ping_pong_flag_l1_a_;
    }
  }
}

template <typename TYPE, bool IF_BF16, typename WORKSPACE_TYPE>
__aicore__ __inline__ void
CubeForward<TYPE, IF_BF16, WORKSPACE_TYPE>::cube1_base_matmul(
    __cbuf__ TYPE* l1_a,
    __cbuf__ TYPE* l1_b,
    __gm__ WORKSPACE_TYPE* gm_out,
    int32_t ky,
    int32_t out_put_matrix_line_strid,
    bool upper_right_flag) {
  auto l1_n_size_ = 128;
  auto l1_m_size_ = ky * 128;
  auto k0_ = 128;
  auto n0_ = 128;
  auto m0_ = 128;

  auto n_ = IF_BF16 ? out_put_matrix_line_strid / 128 : 128;

  for (int n_offset = 0; n_offset < l1_n_size_; n_offset += SIZE_128) {
    // Load b to L0
    auto l0_b = ping_pong_flag_l0_b_ ? l0_b_pong_ : l0_b_ping_;
    wait_flag(PIPE_M, PIPE_MTE1, ping_pong_flag_l0_b_ + 2);
    if (l1_n_size_ == SIZE_128) {
      load_cbuf_to_cb(l0_b,
                      l1_b,
                      0,                     // baseIdx
                      k0_ * n0_ / SIZE_256,  // repeat
                      1,                     // srcStride  连续读为1
                      0,                     // dstStride  连续写为0
                      0,                     // sid
                      false,                 // transpose
                      inc                    // addr_cal_mode_t
      );
    } else {
      for (int i = 0; i < k0_ / SIZE_16; i++) {
        load_cbuf_to_cb(l0_b + i * n0_ * SIZE_16,
                        l1_b + i * l1_n_size_ * SIZE_16 +
                            n_offset * SIZE_16,  // l1_b + i * l1_n_size_ *
                                                 // SIZE_16 + n_offset * SIZE_16
                        0,                       // baseIdx
                        n0_ / SIZE_16,           // repeat
                        1,                       // srcStride  连续读为1
                        0,                       // dstStride  连续写为0
                        0,                       // sid
                        false,                   // transpose
                        inc                      // addr_cal_mode_t
        );
      }
    }

    set_flag(PIPE_MTE1, PIPE_M, ping_pong_flag_l0_b_ + 2);
    wait_flag(PIPE_MTE1, PIPE_M, ping_pong_flag_l0_b_ + 2);

    for (int m_offset = 0; m_offset < l1_m_size_; m_offset += SIZE_128) {
      bool l0_skip_flag = (upper_right_flag && m_offset == 0);
      // Load a to L0
      auto l0_a = ping_pong_flag_l0_a_ ? l0_a_pong_ : l0_a_ping_;
      auto l0_c = ping_pong_flag_l0_c_ ? l0_c_pong_ : l0_c_ping_;

      wait_flag(PIPE_M, PIPE_MTE1, ping_pong_flag_l0_a_);
      if (!l0_skip_flag) {
        for (int32_t i = 0; i < m0_ / SIZE_16; i++) {
          load_cbuf_to_ca(
              l0_a + i * k0_ * SIZE_16,
              l1_a + m_offset * SIZE_16 +
                  i * SIZE_256,      // l1_a + m_offset * SIZE_16 + i * SIZE_256
              0,                     // baseIdx
              k0_ / SIZE_16,         // repeat
              l1_m_size_ / SIZE_16,  // srcStride
              0,                     // dstStride
              0,                     // sid
              false,                 // transpose
              inc                    // addr_cal_mode_t
          );
        }
      }
      set_flag(PIPE_MTE1, PIPE_M, ping_pong_flag_l0_a_);

      wait_flag(PIPE_MTE1, PIPE_M, ping_pong_flag_l0_a_);
      if (!l0_skip_flag) {
        mad(l0_c,
            l0_a,
            l0_b,
            m0_,  // m
            k0_,  // k
            n0_,  // n
            3,    // unitFlag
            0,    // kDirectionAlign  fp32矩阵乘相关配置
            0,    // cmatrixSource   l0c地址模式，默认1
            1);   // cmatrixInitVal  1 l0c初始为0
      }
      set_flag(PIPE_M, PIPE_MTE1, ping_pong_flag_l0_a_);

      auto out_offset = (m_offset / 128) * out_put_matrix_line_strid + n_offset;
      auto trans_dtype = IF_BF16 ? F322BF16 : F322F16;
      if (!l0_skip_flag) {
        copy_matrix_cc_to_gm(
            gm_out + out_offset,  // m_offset * n_ + n_offset
            l0_c,
            0,                            // sid
            n0_,                          // NSize  结果矩阵的列
            m0_,                          // MSize  结果矩阵的行
            n_,                           // dstStride_dst_D  结果大矩阵的列
            m0_,                          // srcStride
            3,                            // UnitFlagMode
            IF_BF16 ? NoQuant : F322F16,  // BF16 scores retain FP32.
            0,                            // ReLUPRE
            false,                        // channelSplit
            true                          // NZ2ND_EN
        );
      }
      ping_pong_flag_l0_c_ = 1 - ping_pong_flag_l0_c_;
      ping_pong_flag_l0_a_ = 1 - ping_pong_flag_l0_a_;
    }

    set_flag(PIPE_M, PIPE_MTE1, ping_pong_flag_l0_b_ + 2);
    ping_pong_flag_l0_b_ = 1 - ping_pong_flag_l0_b_;
  }
}

template <typename TYPE, bool IF_BF16, typename WORKSPACE_TYPE>
__aicore__ inline void CubeForward<TYPE, IF_BF16, WORKSPACE_TYPE>::Run() {
  set_padding(0);
  uint64_t config = 0x1;
  set_nd_para(config);

  uint64_t mode;
  uint64_t sync_config;
  is_syn = true;
  auto Z = address.get_total_rounds();
  PresetFlag();

  if (Z == 1) {
    for (int64_t roundId = 0; roundId < Z; roundId++) {
      if (address.is_running(roundId)) {
        Address::PhyAddrForwardCube1Online<TYPE, TYPE, WORKSPACE_TYPE> src[16];
        int64_t src_len = 0;
        address.addrMapping_cube1(gm_Q, gm_K, gm_S, src, src_len, roundId);
        cube1_matmul_op(src, src_len);
      }
      if (is_syn) {
        mode = 2;
        sync_config = 1 | (mode << 4) | (AIC2AIVFLAGID << 8);
        ffts_cross_core_sync(PIPE_FIX, sync_config);

        wait_flag_dev(AIV2AICFLAGID);
      }

      if (address.is_running(roundId)) {
        Address::PhyAddrForwardCube2Online<TYPE, TYPE, float> src[16];
        int64_t src_len = 0;
        address.addrMapping_cube2(gm_P, gm_V, gm_O, src, src_len, roundId);
        cube2_matmul_op(src, src_len, 1, roundId);
      }
      if constexpr (IF_BF16) {
        ffts_cross_core_sync(PIPE_FIX,
                             1 | (2 << 4) | (CUBE_PV_READY_FLAG << 8));
      }
    }
  } else {
    for (int64_t roundId = 0; roundId < 2; roundId++) {
      if (address.is_running(roundId)) {
        Address::PhyAddrForwardCube1Online<TYPE, TYPE, WORKSPACE_TYPE> src[16];
        int64_t src_len = 0;
        address.addrMapping_cube1(gm_Q, gm_K, gm_S, src, src_len, roundId);
        cube1_matmul_op(src, src_len);
      }

      if (is_syn) {
        if (roundId == 0) {
          mode = 2;  // inner-group aic/aiv sync
          sync_config = 1 | (mode << 4) | (AIC2AIVFLAGID << 8);
          ffts_cross_core_sync(PIPE_FIX, sync_config);
        }
      }
    }
    // /**** cube2 + cube3 + cube1 ****/
    for (int64_t roundId = 2; roundId < Z; roundId++) {
      if (is_syn) {
        wait_flag_dev(AIV2AICFLAGID);
        mode = 2;  // inner-group aic/aiv sync
        sync_config = 1 | (mode << 4) | (AIC2AIVFLAGID << 8);
        ffts_cross_core_sync(PIPE_FIX, sync_config);
      }

      if constexpr (IF_BF16) {
        // Reuse a PV slot only after the vector consumed it two rounds
        // ago.
        if (roundId > 3) {
          wait_flag_dev(VECTOR_OUTPUT_READY_FLAG);
        }
      }
      if (address.is_running(roundId - 2)) {
        Address::PhyAddrForwardCube2Online<TYPE, TYPE, float> src[16];
        int64_t src_len = 0;
        address.addrMapping_cube2(gm_P, gm_V, gm_O, src, src_len, roundId - 2);
        cube2_matmul_op(src, src_len, 1, roundId - 2);
      }
      if constexpr (IF_BF16) {
        ffts_cross_core_sync(PIPE_FIX,
                             1 | (2 << 4) | (CUBE_PV_READY_FLAG << 8));
      }

      if (address.is_running(roundId)) {
        Address::PhyAddrForwardCube1Online<TYPE, TYPE, WORKSPACE_TYPE> src[16];
        int64_t src_len = 0;
        address.addrMapping_cube1(gm_Q, gm_K, gm_S, src, src_len, roundId);
        cube1_matmul_op(src, src_len);
      }
    }

    /**** (cube2 + cube3) * 2 ****/
    for (int64_t roundId = 0; roundId < 2; roundId++) {
      if (is_syn) {
        wait_flag_dev(AIV2AICFLAGID);
        if (roundId == 0) {
          mode = 2;  // inner-group aic/aiv sync
          sync_config = 1 | (mode << 4) | (AIC2AIVFLAGID << 8);
          ffts_cross_core_sync(PIPE_FIX, sync_config);
        }
      }

      if constexpr (IF_BF16) {
        if (roundId + Z > 3) {
          wait_flag_dev(VECTOR_OUTPUT_READY_FLAG);
        }
      }
      if (address.is_running(roundId + Z - 2)) {
        Address::PhyAddrForwardCube2Online<TYPE, TYPE, float> src[16];
        int64_t src_len = 0;
        address.addrMapping_cube2(
            gm_P, gm_V, gm_O, src, src_len, roundId + Z - 2);
        cube2_matmul_op(src, src_len, 1, roundId + Z - 2);
      }
      if constexpr (IF_BF16) {
        ffts_cross_core_sync(PIPE_FIX,
                             1 | (2 << 4) | (CUBE_PV_READY_FLAG << 8));
      }
    }
  }

  if constexpr (IF_BF16) {
    wait_flag_dev(VECTOR_OUTPUT_READY_FLAG);
    if (Z > 1) {
      wait_flag_dev(VECTOR_OUTPUT_READY_FLAG);
    }
  }
  ClearFlag();
}

template <typename TYPE, bool IF_BF16, typename WORKSPACE_TYPE>
__aicore__ inline void CubeForward<TYPE, IF_BF16, WORKSPACE_TYPE>::Init(
    __gm__ uint8_t* __restrict__ gm_Q,
    __gm__ uint8_t* __restrict__ gm_K,
    __gm__ uint8_t* __restrict__ gm_V,
    __gm__ uint8_t* __restrict__ gm_S,
    __gm__ uint8_t* __restrict__ gm_P,
    __gm__ float* __restrict__ gm_O,
    __gm__ float* __restrict__ gm_rowsum_diag,
    __gm__ float* __restrict__ gm_rowmax_diag,
    __gm__ float* __restrict__ gm_rowsum,
    int32_t Y,
    int32_t F,
    int32_t B,
    int32_t N,
    int32_t S1,
    int32_t S2,
    int32_t D,
    int32_t nG,
    int32_t qk_triangle,
    int32_t sparseMode,
    int32_t window_length) {
  this->gm_Q = (__gm__ TYPE* __restrict__)gm_Q;
  this->gm_K = (__gm__ TYPE* __restrict__)gm_K;
  this->gm_V = (__gm__ TYPE* __restrict__)gm_V;
  this->gm_S = (__gm__ WORKSPACE_TYPE* __restrict__)gm_S;
  this->gm_P = (__gm__ TYPE* __restrict__)gm_P;
  this->gm_O = gm_O;

  this->Y = Y;
  if (this->Y == 0) {
    return;
  }
  this->F = F;
  this->B = B;
  this->N = N;
  this->S1 = S1;
  this->S2 = S2;
  this->D = D;
  this->nG = nG;
  if (this->nG == 0) {
    return;
  }
  this->G = N / nG;

  this->qk_triangle = qk_triangle;
  this->sparseMode = sparseMode;
  this->window_length = window_length;

  // 无mask
  if (qk_triangle == 0) {
    H1 = S1 / 128;
    H2 = S2 / 128;
    L = H1;  // 负责均衡
    column_per_core = H2 / Y;
    column_remain = H2 % Y;
  }
  // 有mask
  else {
    H1 = S1 / 128;
    H2 = S2 / 128;
    L = H1 / 2;  // 负责均衡
    column_per_core = (H2 + 1) / Y;
    column_remain = (H2 + 1) % Y;

    // sparse场景：行数、列数以及尾块需要重新设置 update
    if (this->sparseMode == 1) {
      H1 = S1 / 128;
      H2 = S2 / 128;
      W = this->window_length / 128;
      L = H1 - W / 2;
      column_per_core = (W + 1) / Y;
      column_remain = (W + 1) % Y;
    }
  }

  cur_core_index = get_block_idx();
  core_group_index = cur_core_index / Y;
  local_block_index = cur_core_index % Y;
  if (N < 0 || L < 0) {
    return;
  }
  if (N > std::numeric_limits<int32_t>::max() / L) {
    return;
  }
  row_per_batch = N * L;
  this->rowsum_gm = gm_rowsum;
  this->rowsum_diag = gm_rowsum_diag + cur_core_index * BASE_BLOCK_SIZE * 2 *
                                           MAX_SWITCH_TIME * 2;
  this->rowmax_diag = gm_rowmax_diag + cur_core_index * BASE_BLOCK_SIZE * 2 *
                                           MAX_SWITCH_TIME * 2;

  // 寻址模块的初始化
  address.init(this->B,
               this->N,
               this->S1,
               this->S2,
               128,
               this->G,
               this->qk_triangle,
               this->sparseMode,
               this->window_length);
  // 设置寻址模块核组信息
  address.set_tiling(
      this->Y * this->F, cur_core_index, IF_BF16 ? 8 : 64, 2);  // A3
}

template <typename TYPE, bool IF_BF16, typename WORKSPACE_TYPE>
__aicore__ inline void
CubeForward<TYPE, IF_BF16, WORKSPACE_TYPE>::PresetFlag() {
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID3);

  set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
  set_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
  set_flag(PIPE_M, PIPE_MTE1, EVENT_ID2);
  set_flag(PIPE_M, PIPE_MTE1, EVENT_ID3);
}

template <typename TYPE, bool IF_BF16, typename WORKSPACE_TYPE>
__aicore__ inline void CubeForward<TYPE, IF_BF16, WORKSPACE_TYPE>::ClearFlag() {
  wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
  wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
  wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
  wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID3);

  wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
  wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
  wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID2);
  wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID3);
}

}  // namespace CUBE_FORWARD_ONLINE

#endif

#endif  // __CUBEFORWARD_H__

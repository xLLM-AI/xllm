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
/*
 * AddressMapping头文件： 用于寻址的前反向功能，与cube的寻址解耦开来
 */
#ifndef __ADDRESS_MAPPING_FORWARD_ONLINE_H__
#define __ADDRESS_MAPPING_FORWARD_ONLINE_H__

#include <cstdint>

#include "address_const.h"

namespace Address {
template <typename TYPE>
class AddressMappingForwardOnline {
 public:
  // B N S D的格式
  int64_t batchSize_;            // 批次
  int64_t headNum_;              // head数量
  int64_t querySequenceLen_;     // query序列长度
  int64_t keyValueSequenceLen_;  // key序列长度
  int64_t headDim_;              // headDim
  int64_t gqaGroupNum_;          // GQA的组数

  // 负载均衡前的信息
  int64_t isOdd_;
  int64_t blockRows_;  // 行数
  int64_t blockCols_;  // 列数

  // 负责均衡的信息
  bool isTriangle_;       // 三角形的标志
  int64_t sparseMode_;    // sparse模式
  int64_t windowLength_;  // 滑动窗口
  int64_t windowSize_;    // 滑动窗口

  // 核组信息
  int64_t coreNum_;         // 正向的核数
  int64_t localCoreIndex_;  // 正向当前核的index

  // 轮次、基本块相关信息
  int64_t blockNumPerCore_;    // 正向每个轮次计算的基本块数量
  int64_t ky_;                 // 正向按y方向变量的基本块数量
  int64_t kx_;                 // 正向按x方向变量的基本块数量
  int64_t blockNumPerRow_;     // 正向每行的基本块数量
  int64_t blockNumPerColumn_;  // 正向每列的基本块数量
  int64_t blockNumPerHead_;    // 正向每个head的基本块数量
  int64_t blockNumPerBatch_;   // 正向每个batch的基本块数量
  int64_t totalBlocks_;        // 正向计算的总块数
  int64_t totalRounds_;        // 正向总共的轮次
  int64_t totalLines_;         // 正向总共的行数

 public:
  /**
   * 预处理：提前分好段
   * @param addr
   * @param addr_len
   * @param round_id
   */
  __aicore__ __inline__ void forward_addrMapping_pre(ForWardAddrOnline* addr,
                                                     int64_t& addr_len,
                                                     int64_t round_id) {
    if (this->coreNum_ == 0) {
      return;
    }
    if (this->ky_ == 0) {
      return;
    }
    if (this->blockNumPerRow_ == 0) {
      return;
    }
    if (this->blockNumPerColumn_ == 0) {
      return;
    }
    if (this->blockNumPerHead_ == 0) {
      return;
    }
    if (this->blockNumPerBatch_ == 0) {
      return;
    }
    if (this->headNum_ == 0) {
      return;
    }
    // 当前核计算起始块的索引
    int64_t skip_block = this->coreNum_ * this->blockNumPerRow_ * this->ky_;
    int64_t outer_row =
        (round_id * this->kx_) / this->blockNumPerRow_ * skip_block;
    int64_t inner_row =
        this->localCoreIndex_ * this->ky_ * this->blockNumPerRow_;
    int64_t inner_col = (round_id * this->kx_) % this->blockNumPerRow_;
    int64_t cur_block_id = outer_row + inner_row + inner_col;

    int64_t row_num_per_round = this->ky_;
    int64_t col_num_per_round = this->kx_;
    int64_t cur_core_totalBlocks_ =
        this->blockNumPerRow_ * this->ky_ *
        (this->totalLines_ / this->ky_ / this->coreNum_);
    int64_t remain_block_num =
        (this->totalLines_ % (this->coreNum_ * this->ky_)) / this->ky_;
    if (this->localCoreIndex_ < remain_block_num) {
      cur_core_totalBlocks_ += this->ky_ * this->blockNumPerRow_;
    }

    // 最后轮次的尾块处理：
    int64_t remain = this->blockNumPerCore_;
    if ((round_id + 1) * blockNumPerCore_ > cur_core_totalBlocks_) {
      remain = blockNumPerCore_ -
               ((round_id + 1) * blockNumPerCore_ - cur_core_totalBlocks_);
    }

    // 设置x，y方向的基本块数量
    int64_t Ky = this->ky_;
    int64_t Kx = remain / Ky;

    // 当前轮次的(b,n,ir_begin,ic_begin)
    int64_t b = cur_block_id / this->blockNumPerBatch_;
    int64_t n = cur_block_id % this->blockNumPerBatch_ / this->blockNumPerHead_;
    int64_t block_row = cur_block_id % this->blockNumPerHead_ /
                        (row_num_per_round * this->blockNumPerRow_);
    int64_t ir = (outer_row / this->blockNumPerRow_ +
                  this->localCoreIndex_ * this->ky_) %
                 this->blockNumPerColumn_;
    int64_t ic = inner_col;

    // 处理边界：
    addr[0].b = b;
    addr[0].n = n;
    addr[0].iR = ir;
    addr[0].iC = ic;
    addr[0].kx = Kx;
    addr[0].ky = Ky;
    addr[0].k = remain;

    int index = 0;
    for (; remain > 0;) {
      if (addr[index].iC + addr[index].kx > this->blockNumPerRow_) {  // 换行
        addr[index].kx = this->blockNumPerRow_ - addr[index].iC;
        addr[index].k = addr[index].kx * addr[index].ky;

        addr[index + 1].b = addr[index].b;
        addr[index + 1].n = addr[index].n;
        addr[index + 1].iR = addr[index].iR + addr[index].ky * coreNum_;
        addr[index + 1].iC = 0;
        addr[index + 1].k = remain - addr[index].k;
        addr[index + 1].ky = addr[index].ky;
        addr[index + 1].kx = addr[index + 1].k / addr[index + 1].ky;
        if (addr[index + 1].iR >= this->blockNumPerColumn_) {  // 换head
          int64_t skip_head = addr[index + 1].iR / this->blockNumPerColumn_;
          addr[index + 1].n = addr[index].n + skip_head;
          addr[index + 1].iR = addr[index + 1].iR % this->blockNumPerColumn_;
          int64_t skip_batch = addr[index + 1].n / this->headNum_;
          if (addr[index + 1].n >= this->headNum_) {  // 换batch
            addr[index + 1].b = addr[index].b + skip_batch;
            addr[index + 1].n = addr[index + 1].n % this->headNum_;
          }
        }
      }
      remain -= addr[index].k;
      ++index;
    }
    addr_len = index;
  }

  /**
   * no-mask场景的cube1寻址
   * @tparam T_LEFT
   * @tparam T_RIGHT
   * @tparam T_OUTPUT
   * @param left
   * @param right
   * @param out
   * @param addr
   * @param src
   * @param src_len
   * @param round_id
   */
  template <typename T_LEFT, typename T_RIGHT, typename T_OUTPUT>
  __aicore__ __inline__ void addrMapping_rectangular_cube1(
      __gm__ T_LEFT* __restrict__ left,
      __gm__ T_RIGHT* __restrict__ right,
      __gm__ T_OUTPUT* __restrict__ out,
      const ForWardAddrOnline* addr,
      PhyAddrForwardCube1Online<T_LEFT, T_RIGHT, T_OUTPUT>* src,
      int64_t src_len,
      int64_t round_id) {
    // 开启work space
    auto out_offset_round_even = out + this->localCoreIndex_ *
                                           this->blockNumPerCore_ *
                                           ATTENTION_SCORE_BLOCK_SIZE;
    auto out_offset_round_odd =
        out +
        this->coreNum_ * this->blockNumPerCore_ * ATTENTION_SCORE_BLOCK_SIZE +
        this->localCoreIndex_ * this->blockNumPerCore_ *
            ATTENTION_SCORE_BLOCK_SIZE;

    if (this->gqaGroupNum_ == 0) {
      return;
    }
    if (this->headNum_ == 0) {
      return;
    }
    for (int64_t i = 0; i < src_len; ++i) {
      int64_t b = addr[i].b;
      int64_t n = addr[i].n;
      int64_t ir = addr[i].iR;
      int64_t ic = addr[i].iC;
      int64_t Kx = addr[i].kx;
      int64_t Ky = addr[i].ky;
      int64_t k = addr[i].k;

      // 设置bn偏移量
      auto bn_left_offset = left + (b * this->headNum_ + n) *
                                       this->querySequenceLen_ * this->headDim_;
      auto bn_right_offset = right + (b * this->headNum_ + n) *
                                         this->keyValueSequenceLen_ *
                                         this->headDim_;
      int64_t g_index = n / (this->headNum_ / this->gqaGroupNum_);
      auto bn_right_offset_gqa = right + (b * this->gqaGroupNum_ + g_index) *
                                             this->keyValueSequenceLen_ *
                                             this->headDim_;

      src[i].left = bn_left_offset + ir * ATTENTION_SCORE_BLOCK_SIZE;
      src[i].right = bn_right_offset_gqa + ic * ATTENTION_SCORE_BLOCK_SIZE;
      src[i].out =
          ((round_id + 1) % 2) ? out_offset_round_even : out_offset_round_odd;
      src[i].kx = Kx;
      src[i].ky = Ky;
      src[i].k = k;
      src[i].lineStride = Kx * ATTENTION_SCORE_BLOCK_SIZE;
      src[i].lowerLeft = false;
      src[i].upperRight = false;
      src[i].onStartSection = ic == 0 ? true : false;
      src[i].onEndSection =
          (ic + Kx >= this->blockNumPerRow_ - 1) ? true : false;

      // 多段时，work space的偏移量更新
      out_offset_round_even += k * ATTENTION_SCORE_BLOCK_SIZE;
      out_offset_round_odd += k * ATTENTION_SCORE_BLOCK_SIZE;
    }
  }

  /**
   * mask场景的cube1寻址
   * @tparam T_LEFT
   * @tparam T_RIGHT
   * @tparam T_OUTPUT
   * @param left
   * @param right
   * @param out
   * @param addr
   * @param src
   * @param src_len
   * @param round_id
   */
  template <typename T_LEFT, typename T_RIGHT, typename T_OUTPUT>
  __aicore__ __inline__ void addrMapping_triangular_cube1(
      __gm__ T_LEFT* __restrict__ left,
      __gm__ T_RIGHT* __restrict__ right,
      __gm__ T_OUTPUT* __restrict__ out,
      const ForWardAddrOnline* addr,
      PhyAddrForwardCube1Online<T_LEFT, T_RIGHT, T_OUTPUT>* src,
      int64_t& src_len,
      int64_t round_id) {
    int64_t index = 0;
    int64_t tri_block_num_per_column =
        this->blockNumPerColumn_ - 2 * this->isOdd_;
    if (this->gqaGroupNum_ == 0) {
      return;
    }
    if (this->headNum_ == 0) {
      return;
    }
    for (int64_t i = 0; i < src_len && index < src_len; ++i) {
      int64_t iR = addr[i].iR;
      int64_t iC = addr[i].iC;
      int64_t kx = addr[i].kx;
      int64_t ky = addr[i].ky;
      int64_t k = addr[i].k;

      // 倒三角和非倒三角这三个变量不一样
      int64_t switch_index = tri_block_num_per_column + iR + (iR + 1) % 2;
      int64_t row_offset = (iR + 1) % 2 == 1 ? -1 : 1;
      int64_t row_index_left_section =
          tri_block_num_per_column + iR;  // 倒三角 ：非倒三角
      int64_t row_index_right_section =
          tri_block_num_per_column - 1 - iR + row_offset;

      int64_t col_index_left_section = iC;
      int64_t col_index_right_section = iC - switch_index - 1;

      // GQA设置：在cube1中K（右矩阵）进行修改
      int64_t g_index = addr[i].n / (this->headNum_ / this->gqaGroupNum_);
      int64_t bn_offset_gqa_right_matrix =
          (addr[i].b * this->gqaGroupNum_ + g_index) *
          this->keyValueSequenceLen_ * this->headDim_;  // for gqa mode
      int64_t bn_offset_right_matrix =
          (addr[i].b * this->headNum_ + addr[i].n) *
          this->keyValueSequenceLen_ * this->headDim_;
      int64_t bn_offset_left_matrix = (addr[i].b * this->headNum_ + addr[i].n) *
                                      this->querySequenceLen_ * this->headDim_;

      int64_t q_left_offset_section =
          row_index_left_section * ATTENTION_SCORE_BLOCK_SIZE;
      int64_t q_right_offset_section =
          row_index_right_section * ATTENTION_SCORE_BLOCK_SIZE;
      int64_t k_left_offset_section =
          col_index_left_section * ATTENTION_SCORE_BLOCK_SIZE;
      int64_t k_right_offset_section =
          col_index_right_section * ATTENTION_SCORE_BLOCK_SIZE;

      // sparse场景：is_tri == true 以及 sparse_mode == 1
      bool sparse_flag = false;
      int64_t window_block_size = this->windowLength_ / 128;
      if (this->isTriangle_ && this->sparseMode_ == 1) {
        sparse_flag = iR > ((window_block_size - 1) / 2) ? true : false;
        switch_index = (window_block_size / 2) + iR;
        row_index_left_section = (window_block_size / 2) + iR;
        row_index_right_section = (window_block_size / 2) - 1 - iR;
        col_index_left_section = iC;
        col_index_right_section = iC - switch_index - 1;
        q_left_offset_section =
            row_index_left_section * ATTENTION_SCORE_BLOCK_SIZE;
        q_right_offset_section =
            row_index_right_section * ATTENTION_SCORE_BLOCK_SIZE;
        k_left_offset_section =
            col_index_left_section * ATTENTION_SCORE_BLOCK_SIZE;
        k_right_offset_section =
            col_index_right_section * ATTENTION_SCORE_BLOCK_SIZE;
      }
      int64_t row_index_sparse_section = iR + (window_block_size / 2);
      int64_t col_index_sparse_section = iR + iC - (window_block_size / 2);
      int64_t q_sparse_offset_section =
          row_index_sparse_section * ATTENTION_SCORE_BLOCK_SIZE;
      int64_t k_sparse_offset_section =
          col_index_sparse_section * ATTENTION_SCORE_BLOCK_SIZE;

      int64_t out_offset =
          ((addr[i].b * this->headNum_ + addr[i].n) * this->blockNumPerRow_ *
               this->blockNumPerColumn_ +
           (iR * this->blockNumPerRow_)) *
          ATTENTION_SCORE_BLOCK_SIZE;

      int64_t db_offset =
          (round_id % 2) * (this->coreNum_ * this->blockNumPerCore_ *
                            ATTENTION_SCORE_BLOCK_SIZE);
      if (index == 0) {
        src[index].out = out + db_offset +
                         this->localCoreIndex_ * this->blockNumPerCore_ *
                             ATTENTION_SCORE_BLOCK_SIZE;
      } else {
        src[index].out =
            src[index - 1].out + src[index - 1].k * ATTENTION_SCORE_BLOCK_SIZE;
      }

      if (!sparse_flag && switch_index < iC) {
        if (index >= src_len) {
          break;
        }
        src[index].left = left + bn_offset_left_matrix + q_right_offset_section;
        src[index].right =
            right + bn_offset_gqa_right_matrix + k_right_offset_section;
        src[index].kx = kx;
        src[index].ky = ky;
        src[index].k = k;
        src[index].upperRight =
            (iC + src[index].kx >= blockNumPerRow_ - 1) ? true : false;
        src[index].lowerLeft = false;
        src[index].lineStride = src[index].kx * ATTENTION_SCORE_BLOCK_SIZE;
        src[index].onEndSection = false;
        src[index].onStartSection = false;

        src[index].onStartSection = iC == 0 ? true : false;
        src[index].onEndSection = src[index].upperRight;

        index++;
      } else if (!sparse_flag && iC <= switch_index &&
                 iC + kx - 1 > switch_index) {
        if (index + 1 >= src_len) {
          break;
        }
        src[index].left = left + bn_offset_left_matrix + q_left_offset_section;
        src[index].right =
            right + bn_offset_gqa_right_matrix + k_left_offset_section;
        src[index].kx = switch_index - iC + 1;
        src[index].ky = ky;
        src[index].k = src[index].kx * src[index].ky;
        src[index].upperRight = true;
        src[index].lowerLeft = false;
        src[index].lineStride = src[index].kx * ATTENTION_SCORE_BLOCK_SIZE;
        src[index].onEndSection = true;
        src[index].onStartSection = iC == 0 ? true : false;

        src[index + 1].left =
            left + bn_offset_left_matrix + q_right_offset_section;
        src[index + 1].right = right + bn_offset_gqa_right_matrix;
        src[index + 1].out =
            src[index].out + src[index].k * ATTENTION_SCORE_BLOCK_SIZE;
        src[index + 1].kx = kx - src[index].kx;
        src[index + 1].ky = ky;
        src[index + 1].k = src[index + 1].kx * src[index + 1].ky;
        src[index + 1].upperRight =
            switch_index + src[index + 1].kx >= this->blockNumPerRow_ - 1
                ? true
                : false;
        src[index + 1].lowerLeft = false;
        src[index + 1].lineStride =
            src[index + 1].kx * ATTENTION_SCORE_BLOCK_SIZE;
        src[index + 1].onStartSection = true;
        src[index + 1].onEndSection = src[index + 1].upperRight;

        index += 2;
      } else if (!sparse_flag && iC <= switch_index &&
                 iC + kx - 1 <= switch_index) {
        if (index >= src_len) {
          break;
        }
        src[index].left = left + bn_offset_left_matrix + q_left_offset_section;
        src[index].right =
            right + bn_offset_gqa_right_matrix + k_left_offset_section;
        src[index].kx = kx;
        src[index].ky = ky;
        src[index].k = k;
        src[index].upperRight =
            iC + src[index].kx - 1 >= switch_index ? true : false;
        src[index].lowerLeft = false;
        src[index].lineStride = src[index].kx * ATTENTION_SCORE_BLOCK_SIZE;
        src[index].onStartSection = iC == 0 ? true : false;
        src[index].onEndSection = src[index].upperRight;

        index++;

      } else {
        if (index >= src_len) {
          break;
        }
        src[index].left =
            left + bn_offset_left_matrix + q_sparse_offset_section;
        src[index].right =
            right + bn_offset_gqa_right_matrix + k_sparse_offset_section;
        src[index].k = k;
        index++;
      }
    }
    src_len = index;
  }

  /**
   * no-mask场景的cube2寻址
   * @tparam T_LEFT
   * @tparam T_RIGHT
   * @tparam T_OUTPUT
   * @param left
   * @param right
   * @param out
   * @param addr
   * @param src
   * @param src_len
   * @param round_id
   */
  template <typename T_LEFT, typename T_RIGHT, typename T_OUTPUT>
  __aicore__ __inline__ void addrMapping_rectangular_cube2(
      __gm__ T_LEFT* __restrict__ left,
      __gm__ T_RIGHT* __restrict__ right,
      __gm__ T_OUTPUT* __restrict__ out,
      const ForWardAddrOnline* addr,
      PhyAddrForwardCube2Online<T_LEFT, T_RIGHT, T_OUTPUT>* src,
      int64_t src_len,
      int64_t round_id) {
    // 开启work space
    auto left_offset_round_even = left + this->localCoreIndex_ *
                                             this->blockNumPerCore_ *
                                             ATTENTION_SCORE_BLOCK_SIZE;
    auto left_offset_round_odd =
        left +
        this->coreNum_ * this->blockNumPerCore_ * ATTENTION_SCORE_BLOCK_SIZE +
        this->localCoreIndex_ * this->blockNumPerCore_ *
            ATTENTION_SCORE_BLOCK_SIZE;

    if (this->gqaGroupNum_ == 0) {
      return;
    }
    if (this->headNum_ == 0) {
      return;
    }
    for (int64_t i = 0; i < src_len; ++i) {
      auto b = addr[i].b;
      auto n = addr[i].n;
      auto ir = addr[i].iR;
      auto ic = addr[i].iC;
      auto Kx = addr[i].kx;
      auto Ky = addr[i].ky;
      auto k = addr[i].k;

      // 设置偏移量
      auto bn_right_offset = right + (b * this->headNum_ + n) *
                                         this->keyValueSequenceLen_ *
                                         this->headDim_;
      int64_t g_index = n / (this->headNum_ / this->gqaGroupNum_);
      auto bn_right_offset_gqa = right + (b * this->gqaGroupNum_ + g_index) *
                                             this->keyValueSequenceLen_ *
                                             this->headDim_;
      auto bn_out_offset = out + (b * this->headNum_ + n) *
                                     this->querySequenceLen_ * this->headDim_;

      src[i].left =
          ((round_id + 1) % 2) ? left_offset_round_even : left_offset_round_odd;
      src[i].right = bn_right_offset_gqa + ic * ATTENTION_SCORE_BLOCK_SIZE;
      src[i].out = bn_out_offset + ir * ATTENTION_SCORE_BLOCK_SIZE;
      src[i].kx = Kx;
      src[i].ky = Ky;
      src[i].k = k;
      src[i].lineStride = Kx * ATTENTION_SCORE_BLOCK_SIZE;
      src[i].lowerLeft = false;
      src[i].upperRight = false;
      src[i].onStartSection = ic == 0 ? true : false;
      src[i].onEndSection =
          (ic + Kx >= this->blockNumPerRow_ - 1) ? true : false;

      // 多段时，work space偏移量更新
      left_offset_round_even += k * ATTENTION_SCORE_BLOCK_SIZE;
      left_offset_round_odd += k * ATTENTION_SCORE_BLOCK_SIZE;
    }
  }

  /**
   * mask场景饿cube2寻址
   * @tparam T_LEFT
   * @tparam T_RIGHT
   * @tparam T_OUTPUT
   * @param left
   * @param right
   * @param out
   * @param addr
   * @param src
   * @param src_len
   * @param round_id
   */
  template <typename T_LEFT, typename T_RIGHT, typename T_OUTPUT>
  __aicore__ __inline__ void addrMapping_triangular_cube2(
      __gm__ T_LEFT* __restrict__ left,
      __gm__ T_RIGHT* __restrict__ right,
      __gm__ T_OUTPUT* __restrict__ out,
      const ForWardAddrOnline* addr,
      PhyAddrForwardCube2Online<T_LEFT, T_RIGHT, T_OUTPUT>* src,
      int64_t& src_len,
      int64_t round_id) {
    // 负载均衡的地址偏移
    int64_t index = 0;
    int64_t tri_blocks_per_column = this->blockNumPerColumn_ - 2 * this->isOdd_;
    if (this->gqaGroupNum_ == 0) {
      return;
    }
    if (this->headNum_ == 0) {
      return;
    }
    for (int64_t i = 0; i < src_len && index < src_len; ++i) {
      // left、right、out的地址配置
      int64_t iR = addr[i].iR;
      int64_t iC = addr[i].iC;
      int64_t kx = addr[i].kx;
      int64_t ky = addr[i].ky;
      int64_t k = addr[i].k;
      // 倒三角和非倒三角这三个变量不一样

      int64_t switch_index = tri_blocks_per_column + iR + (iR + 1) % 2;
      int64_t row_index_left_section =
          tri_blocks_per_column + iR;  // 倒三角 ：非倒三角
      int64_t row_offset = (iR + 1) % 2 == 1 ? -1 : 1;
      int64_t row_index_right_section =
          tri_blocks_per_column - 1 - iR + row_offset;
      int64_t col_index_left_section = iC;
      int64_t col_index_right_section = iC - switch_index - 1;
      // GQA设置：在cube1中K（右矩阵）进行修改
      int64_t g_index = addr[i].n / (this->headNum_ / this->gqaGroupNum_);
      int64_t bn_offset_gqa_right_matrix =
          (addr[i].b * this->gqaGroupNum_ + g_index) *
          this->keyValueSequenceLen_ * this->headDim_;
      int64_t bn_offset_left_matrix =
          ((addr[i].b * this->headNum_ + addr[i].n) * this->blockNumPerRow_ *
           this->blockNumPerColumn_) *
          ATTENTION_SCORE_BLOCK_SIZE;  // 当前b,n下的偏移量
      int64_t bn_offset_right_matrix =
          (addr[i].b * this->headNum_ + addr[i].n) *
          this->keyValueSequenceLen_ * this->headDim_;  // 当前b,n下的偏移量
      int64_t bn_offset_out = (addr[i].b * this->headNum_ + addr[i].n) *
                              this->querySequenceLen_ *
                              this->headDim_;  // 当前b,n下的偏移量

      // sparse场景：is_tri == true 以及 sparse_mode == 1
      bool sparse_flag = false;
      int64_t window_block_size = this->windowLength_ / 128;
      if (this->isTriangle_ && this->sparseMode_ == 1) {
        sparse_flag = iR > ((window_block_size - 1) / 2) ? true : false;
        switch_index = (window_block_size / 2) + iR;
        row_index_left_section = (window_block_size / 2) + iR;
        row_index_right_section = (window_block_size / 2) - 1 - iR;
        col_index_left_section = iC;
        col_index_right_section = iC - switch_index - 1;
      }
      int64_t row_index_sparse_section = iR + (window_block_size / 2);
      int64_t col_index_sparse_section = iR + iC - (window_block_size / 2);

      // 开启double buffer
      int64_t db_offset =
          (round_id % 2) * (this->coreNum_ * this->blockNumPerCore_ *
                            ATTENTION_SCORE_BLOCK_SIZE);
      if (index == 0) {
        src[index].left = left + db_offset +
                          this->localCoreIndex_ * this->blockNumPerCore_ *
                              ATTENTION_SCORE_BLOCK_SIZE;
      } else {
        src[index].left =
            src[index - 1].left + src[index - 1].k * ATTENTION_SCORE_BLOCK_SIZE;
      }

      if (!sparse_flag && switch_index < iC) {
        if (index >= src_len) {
          break;
        }
        src[index].right = right + bn_offset_gqa_right_matrix +
                           col_index_right_section * ATTENTION_SCORE_BLOCK_SIZE;
        src[index].out = out + bn_offset_out +
                         row_index_right_section * ATTENTION_SCORE_BLOCK_SIZE;
        src[index].kx = kx;
        src[index].ky = ky;
        src[index].k = k;
        src[index].upperRight =
            (iC + src[index].kx >= blockNumPerRow_ - 1) ? true : false;
        src[index].lowerLeft = false;
        src[index].lineStride = src[index].kx * ATTENTION_SCORE_BLOCK_SIZE;

        src[index].onEndSection = false;
        src[index].onStartSection = (iC == switch_index + 1) ? true : false;
        if (src[index].upperRight) {
          src[index].onEndSection = true;
        }
        if (iC == 0) {
          src[index].onStartSection = true;
        }
        ++index;
      } else if (!sparse_flag && iC <= switch_index &&
                 iC + kx - 1 > switch_index) {
        if (index + 1 >= src_len) {
          break;
        }
        src[index].right = right + bn_offset_gqa_right_matrix +
                           col_index_left_section * ATTENTION_SCORE_BLOCK_SIZE;
        src[index].out = out + bn_offset_out +
                         row_index_left_section * ATTENTION_SCORE_BLOCK_SIZE;
        src[index].kx = switch_index - iC + 1;
        src[index].ky = ky;
        src[index].k = src[index].kx * src[index].ky;
        src[index].upperRight = true;
        src[index].lowerLeft = false;
        src[index].lineStride = src[index].kx * ATTENTION_SCORE_BLOCK_SIZE;
        src[index].onStartSection = iC == 0 ? true : false;
        src[index].onEndSection = true;

        src[index + 1].left =
            src[index].left + src[index].k * ATTENTION_SCORE_BLOCK_SIZE;
        src[index + 1].right = right + bn_offset_gqa_right_matrix;
        src[index + 1].out =
            out + bn_offset_out +
            row_index_right_section * ATTENTION_SCORE_BLOCK_SIZE;
        src[index + 1].kx = kx - src[index].kx;
        src[index + 1].ky = ky;
        src[index + 1].k = src[index + 1].kx * src[index + 1].ky;
        src[index + 1].upperRight =
            switch_index + src[index + 1].kx >= blockNumPerRow_ - 1 ? true
                                                                    : false;
        src[index + 1].lowerLeft = false;
        src[index + 1].lineStride =
            src[index + 1].kx * ATTENTION_SCORE_BLOCK_SIZE;
        src[index + 1].onStartSection = true;
        src[index + 1].onEndSection = src[index + 1].upperRight;

        index += 2;
      } else if (!sparse_flag && iC <= switch_index &&
                 iC + kx - 1 <= switch_index) {
        if (index >= src_len) {
          break;
        }
        src[index].right = right + bn_offset_gqa_right_matrix +
                           col_index_left_section * ATTENTION_SCORE_BLOCK_SIZE;
        src[index].out = out + bn_offset_out +
                         row_index_left_section * ATTENTION_SCORE_BLOCK_SIZE;
        src[index].kx = kx;
        src[index].ky = ky;
        src[index].k = k;
        src[index].lowerLeft = false;
        src[index].upperRight =
            iC + src[index].kx - 1 >= switch_index ? true : false;
        src[index].lineStride = src[index].kx * ATTENTION_SCORE_BLOCK_SIZE;
        src[index].onStartSection = iC == 0 ? true : false;
        src[index].onEndSection = src[index].upperRight;

        index++;
      } else {
        if (index >= src_len) {
          break;
        }
        src[index].right =
            right + bn_offset_gqa_right_matrix +
            col_index_sparse_section * ATTENTION_SCORE_BLOCK_SIZE;
        src[index].out = out + bn_offset_out +
                         row_index_sparse_section * ATTENTION_SCORE_BLOCK_SIZE;
        src[index].k = k;
        ++index;
      }
    }
    src_len = index;
  }

 public:
  /**
   * 类的初始化
   * @param batch_size
   * @param head_num
   * @param query_sequence_len
   * @param key_value_sequence_len
   * @param head_dim
   * @param gqa_group_num
   * @param is_triangle
   * @param sparse_mode
   * @param window_length
   */
  __aicore__ __inline__ void init(int64_t batch_size,
                                  int64_t head_num,
                                  int64_t query_sequence_len,
                                  int64_t key_value_sequence_len,
                                  int64_t head_dim,
                                  int64_t gqa_group_num,
                                  bool is_triangle,
                                  int64_t sparse_mode,
                                  int64_t window_length) {
    // B N S D初始化
    this->batchSize_ = batch_size;
    this->headNum_ = head_num;
    this->querySequenceLen_ = query_sequence_len;
    this->keyValueSequenceLen_ = key_value_sequence_len;
    this->headDim_ = head_dim;
    this->gqaGroupNum_ = gqa_group_num;
    this->isOdd_ =
        this->querySequenceLen_ / BASE_BLOCK_LENGTH / 2 % 2;  // 2 is index

    // 负责均衡前信息的初始化
    this->blockRows_ = this->querySequenceLen_ / BASE_BLOCK_LENGTH;
    this->blockCols_ = this->keyValueSequenceLen_ / BASE_BLOCK_LENGTH;

    // 初始化负载均衡的信息
    this->isTriangle_ = is_triangle;
    this->sparseMode_ = sparse_mode;
    this->windowLength_ = window_length;
    this->windowSize_ = this->windowLength_ / SIZE_128;
  }

  /**
   * 总轮次
   * @return
   */
  __aicore__ __inline__ int64_t get_total_rounds() {
    return this->totalRounds_;
  }

  /**
   * 判断此轮次是否参与计算
   * @param round_id
   * @return
   */
  __aicore__ __inline__ bool is_running(int64_t round_id) {
    if (this->blockNumPerRow_ == 0) {
      return false;
    }
    int64_t skip_block = this->coreNum_ * this->blockNumPerRow_ * this->ky_;
    int64_t outer_row =
        (round_id * this->kx_) / this->blockNumPerRow_ * skip_block;
    int64_t inner_row =
        this->localCoreIndex_ * this->ky_ * this->blockNumPerRow_;
    int64_t inner_col = (round_id * this->kx_) % this->blockNumPerRow_;
    int64_t cur_block_id = outer_row + inner_row + inner_col;

    return (cur_block_id < this->totalBlocks_);
  }

  /**
   * 设置tiling信息
   * @param core_num
   * @param local_core_index
   * @param block_num_per_core
   * @param ky
   */
  __aicore__ __inline__ void set_tiling(int64_t core_num,
                                        int64_t local_core_index,
                                        int64_t block_num_per_core,
                                        int64_t ky) {
    if (this->ky_ == 0) {
      return;
    }
    if (this->blockNumPerCore_ == 0) {
      return;
    }
    // 初始化
    this->coreNum_ = core_num;
    this->localCoreIndex_ = local_core_index;
    this->blockNumPerCore_ = block_num_per_core;
    this->ky_ = ky;
    this->kx_ = this->blockNumPerCore_ / this->ky_;

    // 根据负载均衡信息来计算基本块
    this->blockNumPerColumn_ = this->blockRows_;
    this->blockNumPerRow_ = this->blockCols_;
    if (this->isTriangle_) {
      this->blockNumPerColumn_ =
          this->blockRows_ / 2 + this->isOdd_;  // 2 is index
      this->blockNumPerRow_ =
          this->blockCols_ + 2 * (1 - this->isOdd_);  // 2 is index
      if (this->sparseMode_ == 1) {
        this->blockNumPerColumn_ =
            this->blockRows_ - this->windowSize_ / 2;  // 2 is value
        this->blockNumPerRow_ = this->windowSize_ + 1;
      }
    }

    // 初始化基本块数量
    this->blockNumPerHead_ = this->blockNumPerColumn_ * this->blockNumPerRow_;
    this->blockNumPerBatch_ = this->headNum_ * this->blockNumPerHead_;
    this->totalBlocks_ = this->batchSize_ * this->blockNumPerBatch_;
    this->totalLines_ =
        this->blockNumPerColumn_ * this->batchSize_ * this->headNum_;

    // 轮次的计算
    int64_t segment_line_per_round =
        this->ky_ * this->coreNum_;  // 在ky方向：每一轮次处理的行数
    int64_t totalRounds_segment_line =
        (this->totalLines_ + segment_line_per_round - 1) /
        segment_line_per_round;  // 处理完所有行数所需要的次数
    int64_t total_block_num = totalRounds_segment_line * this->ky_ *
                              this->blockNumPerRow_;  // 总共的基本块
    this->totalRounds_ =
        (total_block_num + this->blockNumPerCore_ - 1) / this->blockNumPerCore_;
  }

  /**
   * cube1的接口
   * @tparam T_LEFT
   * @tparam T_RIGHT
   * @tparam T_OUTPUT
   * @param left
   * @param right
   * @param out
   * @param src
   * @param src_len
   * @param round_id
   */
  template <typename T_LEFT, typename T_RIGHT, typename T_OUTPUT>
  __aicore__ __inline__ void addrMapping_cube1(
      __gm__ T_LEFT* __restrict__ left,
      __gm__ T_RIGHT* __restrict__ right,
      __gm__ T_OUTPUT* __restrict__ out,
      PhyAddrForwardCube1Online<T_LEFT, T_RIGHT, T_OUTPUT>* src,
      int64_t& src_len,
      int64_t round_id) {
    // 寻址的预处理
    ForWardAddrOnline forward_addr[MAX_SWITCH_TIME];
    forward_addrMapping_pre(forward_addr, src_len, round_id);

    // cube1地址偏移的计算
    if (this->isTriangle_) {
      addrMapping_triangular_cube1(
          left, right, out, forward_addr, src, src_len, round_id);
    } else {  // no-mask
      addrMapping_rectangular_cube1(
          left, right, out, forward_addr, src, src_len, round_id);
    }
  }

  /**
   * cube2的接口
   * @tparam T_LEFT
   * @tparam T_RIGHT
   * @tparam T_OUTPUT
   * @param left
   * @param right
   * @param out
   * @param src
   * @param src_len
   * @param round_id
   */
  template <typename T_LEFT, typename T_RIGHT, typename T_OUTPUT>
  __aicore__ __inline__ void addrMapping_cube2(
      __gm__ T_LEFT* __restrict__ left,
      __gm__ T_RIGHT* __restrict__ right,
      __gm__ T_OUTPUT* __restrict__ out,
      PhyAddrForwardCube2Online<T_LEFT, T_RIGHT, T_OUTPUT>* src,
      int64_t& src_len,
      int64_t round_id) {
    // 寻址的预处理
    ForWardAddrOnline forward_addr[MAX_SWITCH_TIME];
    forward_addrMapping_pre(forward_addr, src_len, round_id);

    // cube1地址偏移的计算
    if (this->isTriangle_) {
      addrMapping_triangular_cube2(
          left, right, out, forward_addr, src, src_len, round_id);
    } else {
      addrMapping_rectangular_cube2(
          left, right, out, forward_addr, src, src_len, round_id);
    }
  }
};
}  // namespace Address
#endif

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
 * 该头文件用于前向vector_online的寻址模块
 */
#ifndef ADDRESS_MODULE_ADDRESSMAPPING_VECTOR_FORWARD_ONLINE_H
#define ADDRESS_MODULE_ADDRESSMAPPING_VECTOR_FORWARD_ONLINE_H

#include <cstdint>

#include "address_const.h"

namespace Address {
class AddressMappingVectorForwardOnline {
 public:
  // B N S D 格式的基础信息
  int64_t batchSize_;            // batch批次的大小
  int64_t headNum_;              // head数量
  int64_t querySequenceLen_;     // query序列长度
  int64_t keyValueSequenceLen_;  // key、value序列长度
  int64_t maskSequenceLen_;      // 传入mask的序列长度

  // 核数、核组的信息
  int64_t coreNum_;      // core数量
  int64_t vectorNum_;    // vector数量
  int64_t coreIndex_;    // 当前核心的序号
  int64_t vectorIndex_;  // vector的序号：0、1

  // 偏移相关信息
  bool isTriangle_;            // 倒三角mask的标志
  int64_t sparseMode_;         // sparse mode: 1 表示开启，0 表示关闭
  int64_t windowSize_;         // 滑动窗口的基本块数量
  int64_t isOdd_;              // 序列长度是256的奇偶数倍
  int64_t blockNumPerCol_;     // 负载均衡后attention的行数
  int64_t blockNumPerRow_;     // 负载均衡后attention的列数
  int64_t blockNumPerHead_;    // 前向每个head的基本块数量
  int64_t blockNumPerBatch_;   // 前向每个batch的基本块数量
  int64_t blockRowsPerHead_;   // 前向每个head的基本行块数量
  int64_t blockRowsPerBatch_;  // 前向每个batch的基本行块数量
  int64_t totalRows_;          // 前向计算的总行块数
  int64_t totalBlocks_;        // 前向计算的总块数
  int64_t totalRounds_;        // 前向总共的轮次
  int64_t blockNumPerCore_;    // 每一个核心处理基本块的数量
  int64_t kx_;                 // kx方向处理基本块的数量
  int64_t ky_;                 // ky方向处理基本块的数量

  // vector偏移相关信息
  int64_t processLineNum_;     // 前向vector处理的行数
  int64_t coreOffset_;         // 前向当前核在work space中的偏移量
  int64_t startLine_;          // 前向vector处理的起始行
  int64_t startLineOffset_;    // 前向vector起始行的偏移量
  int64_t vectorStartOffset_;  // 前向vector起始行的偏移量
  int64_t maskOffset_;         // mask的偏移量
  int64_t totalLines_;         // 总共的行数
  int64_t normProcessLine_;    // vector归一化处理的行数
  int64_t normLineOffset_;     // vector归一化处理的起始行

 public:
  /**
   * 预处理：提前分好段: vector分段略微和cube不同，遇到跳变点也会提前切分
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
    if (this->blockNumPerCol_ == 0) {
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
    int64_t inner_row = this->coreIndex_ * this->ky_ * this->blockNumPerRow_;
    int64_t inner_col = (round_id * this->kx_) % this->blockNumPerRow_;
    int64_t cur_block_id = outer_row + inner_row + inner_col;

    int64_t row_num_per_round = this->ky_;
    int64_t col_num_per_round = this->kx_;
    int64_t cur_core_total_blocks =
        this->blockNumPerRow_ * this->ky_ *
        (this->totalRows_ / this->ky_ / this->coreNum_);
    int64_t remain_block_num =
        (this->totalRows_ % (this->coreNum_ * this->ky_)) / this->ky_;
    if (this->coreIndex_ < remain_block_num) {
      cur_core_total_blocks += this->ky_ * this->blockNumPerRow_;
    }

    // 最后轮次的尾块处理：
    int64_t remain = this->blockNumPerCore_;
    if ((round_id + 1) * blockNumPerCore_ > cur_core_total_blocks) {
      remain =
          this->blockNumPerCore_ -
          ((round_id + 1) * this->blockNumPerCore_ - cur_core_total_blocks);
    }

    // 设置x，y方向的基本块数量
    int64_t Ky = this->ky_;
    int64_t Kx = remain / Ky;

    // 当前轮次的(b,n,ir_begin,ic_begin)
    int64_t b = cur_block_id / this->blockNumPerBatch_;
    int64_t n = cur_block_id % this->blockNumPerBatch_ / this->blockNumPerHead_;
    int64_t ir =
        (outer_row / this->blockNumPerRow_ + this->coreIndex_ * this->ky_) %
        this->blockNumPerCol_;
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
      // 跳变点的位置
      int64_t switch_index =
          addr[index].iR + this->blockNumPerCol_ + (addr[index].iR + 1) % 2;
      switch_index = this->isTriangle_ ? switch_index - 2 * this->isOdd_
                                       : switch_index;  // 2 is value
      if (this->isTriangle_ && (addr[index].iC <= switch_index) &&
          (addr[index].iC + addr[index].kx - 1 >
           switch_index)) {  // 沿着跳变点切分
        addr[index].kx = switch_index - addr[index].iC + 1;
        addr[index].k = addr[index].kx * addr[index].ky;

        addr[index + 1].b = addr[index].b;
        addr[index + 1].n = addr[index].n;
        addr[index + 1].iR = addr[index].iR;
        addr[index + 1].iC = switch_index + 1;
        addr[index + 1].k = remain - addr[index].k;
        addr[index + 1].ky = addr[index].ky;
        addr[index + 1].kx = addr[index + 1].k / addr[index + 1].ky;
      }
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
        if (addr[index + 1].iR >= this->blockNumPerCol_) {  // 换head
          int64_t skip_head = addr[index + 1].iR / this->blockNumPerCol_;
          addr[index + 1].n = addr[index].n + skip_head;
          addr[index + 1].iR = addr[index + 1].iR % this->blockNumPerCol_;
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

    // 处理空的段
    int64_t pos = 0;
    for (size_t i = 0; i < index; ++i) {
      if (addr[i].k == 0) {
        continue;
      }
      addr[pos++] = addr[i];
    }
    addr_len = pos;
  }

  /**
   * 设置全局的信息：轮次、总块数等
   * @return
   */
  __aicore__ __inline__ void set_global_info() {
    // no-mask场景
    this->blockNumPerCol_ = this->querySequenceLen_ / SIZE_128;
    this->blockNumPerRow_ = this->keyValueSequenceLen_ / SIZE_128;

    // 倒三角mask场景：
    if (this->isTriangle_) {
      this->blockNumPerCol_ =
          this->querySequenceLen_ / SIZE_128 / 2 + this->isOdd_;  // 2 is value
      this->blockNumPerRow_ = this->keyValueSequenceLen_ / SIZE_128 +
                              2 * (1 - this->isOdd_);  // 2 is
    }

    // 计算head、batch的基本块
    this->blockNumPerHead_ = this->blockNumPerCol_ * this->blockNumPerRow_;
    this->blockNumPerBatch_ = this->blockNumPerHead_ * this->headNum_;
    this->totalBlocks_ = this->blockNumPerBatch_ * this->batchSize_;
    this->blockRowsPerHead_ = this->blockNumPerCol_;
    this->blockRowsPerBatch_ = this->blockRowsPerHead_ * this->headNum_;
    this->totalRows_ = this->blockRowsPerBatch_ * this->batchSize_;

    // 轮次:
    int64_t segment_line_per_round =
        this->ky_ * this->coreNum_;  // 在ky方向：每一轮次处理的行数
    int64_t totalRounds_segment_line =
        (this->totalRows_ + segment_line_per_round - 1) /
        segment_line_per_round;  // 处理完所有行数所需要的次数
    int64_t total_block_num = totalRounds_segment_line * this->ky_ *
                              this->blockNumPerRow_;  // 总共的基本块
    this->totalRounds_ =
        (total_block_num + this->blockNumPerCore_ - 1) / this->blockNumPerCore_;
  }

  /**
   * 设置vector本地的信息，当前vector的序号等
   * @param vector_index
   * @return
   */
  __aicore__ __inline__ void set_local_info() {
    if (this->vectorNum_ == 0) {
      return;
    }
    this->processLineNum_ = SIZE_128 / 2;  // 2 is value
    this->totalLines_ =
        this->batchSize_ * this->headNum_ * this->querySequenceLen_;

    // 前面的vector处理多余的行
    int64_t vector_id = this->coreIndex_ * 2 + this->vectorIndex_;
    this->normProcessLine_ = this->totalLines_ / this->vectorNum_;
    this->normLineOffset_ = this->normProcessLine_ * vector_id;
    int64_t rows_remain = this->totalLines_ % this->vectorNum_;
    if (rows_remain > 0 && vector_id < rows_remain) {
      this->normProcessLine_ += 1;
    }
    this->normLineOffset_ += vector_id < rows_remain ? vector_id : rows_remain;
  }

  /**
   * 基本偏移量的设置
   */
  __aicore__ __inline__ void set_init_offset() {
    // 计算核组、处理行的偏移量
    this->coreOffset_ =
        this->coreIndex_ * this->blockNumPerCore_ * ATTENTION_SCORE_BLOCK_SIZE;
    this->startLine_ = this->processLineNum_ * this->vectorIndex_;
    this->startLineOffset_ = this->startLine_ * SIZE_128;

    this->vectorStartOffset_ = this->coreOffset_ + this->startLineOffset_;
    this->maskOffset_ = this->startLine_ * this->maskSequenceLen_;
  }

  /**
   * nomask场景的偏移量设置
   * @param round_id
   * @param section
   */
  __aicore__ __inline__ void addrMapping_nomask(const ForWardAddrOnline* addr,
                                                int64_t& src_len,
                                                int64_t round_id,
                                                FORWARD_SECTION_INFO& section) {
    int64_t diag_out_offset = this->coreIndex_ * this->ky_ *
                              ATTENTION_SCORE_BLOCK_SIZE * MAX_SWITCH_TIME * 2;

    for (int64_t i = 0; i < src_len; ++i) {
      int64_t b = addr[i].b;
      int64_t n = addr[i].n;
      int64_t ir = addr[i].iR;
      int64_t ic = addr[i].iC;
      int64_t Kx = addr[i].kx;

      // row_max的偏移量
      int64_t row_max_bn_offset =
          (b * this->headNum_ + n) * this->querySequenceLen_;
      int64_t row_max_inner_offset =
          ir * SIZE_128 + SIZE_128 / 2 * this->vectorIndex_;
      section.rowmaxOffset[2 * i] =
          row_max_bn_offset + row_max_inner_offset;  // 2 is index
      section.rowmaxOffset[2 * i + 1] =
          section.rowmaxOffset[2 * i] + SIZE_128;  // ky = 2, 向下偏移一行

      // section的偏移量
      section.sectionBlockNums[2 * i] = Kx;  // 2 is index
      section.sectionBlockOffset[2 * i] =    // 2 is index
          i == 0 ? this->vectorStartOffset_
                 : section.sectionBlockOffset[2 * (i - 1) + 1] +    // 2 is
                       section.sectionBlockNums[2 * (i - 1) + 1] *  // 2 is
                           ATTENTION_SCORE_BLOCK_SIZE;
      section.sectionBlockNums[2 * i + 1] = Kx;  // 2 is index
      section.sectionBlockOffset[2 * i + 1] =
          section.sectionBlockOffset[2 * i] +  // 2 is index
          section.sectionBlockNums[2 * i] *    // 2 is index
              ATTENTION_SCORE_BLOCK_SIZE;
      // 对角阵的偏移量
      section.diagOffset[2 * i] =
          diag_out_offset +  // 2 is index
          ((round_id % 2) * MAX_SWITCH_TIME + i) * 2 *
              ATTENTION_SCORE_BLOCK_SIZE +                      // 2 is index
          this->vectorIndex_ * ATTENTION_SCORE_BLOCK_SIZE / 2;  // 2 is index
      section.diagOffset[2 * i + 1] =
          section.diagOffset[2 * i] + ATTENTION_SCORE_BLOCK_SIZE;  // 2 is

      // 头尾的判断
      section.isHeadSection[2 * i] = ic == 0 ? true : false;  // 2 is index
      section.isTailSection[2 * i] =
          (ic + Kx >= this->blockNumPerRow_ - 1) ? true : false;  // 2 is index
      section.isHeadSection[2 * i + 1] = ic == 0 ? true : false;  // 2 is index
      section.isTailSection[2 * i + 1] =
          (ic + Kx >= this->blockNumPerRow_ - 1) ? true : false;  // 2 is
    }
    section.sectionNum = src_len * 2;  // 2 is  value
    section.maskNum = 0;
    section.matrixMaskOffset = this->maskOffset_;
    section.processLineNum = this->processLineNum_;
    section.sparseFlag = false;
    section.isTriangle = false;
    section.attentionScoreOffset = (round_id % 2) *  // 2 is index
                                   this->coreNum_ * this->blockNumPerCore_ *
                                   ATTENTION_SCORE_BLOCK_SIZE;
  }

  /**
   * mask场景的偏移量设置
   * @param round_id
   * @param section
   */
  __aicore__ __inline__ void addrMapping_mask(const ForWardAddrOnline* addr,
                                              int64_t& src_len,
                                              int64_t round_id,
                                              FORWARD_SECTION_INFO& section) {
    int64_t index = 0;
    int64_t diag_out_offset = this->coreIndex_ * this->ky_ *
                              ATTENTION_SCORE_BLOCK_SIZE * MAX_SWITCH_TIME * 2;
    int64_t tri_block_num_per_column = this->blockNumPerCol_ - 2 * this->isOdd_;
    if (this->vectorIndex_ == 0) {
      return;
    }
    for (int64_t i = 0; i < src_len; ++i) {
      int64_t b = addr[i].b;
      int64_t n = addr[i].n;
      int64_t i_r = addr[i].iR;
      int64_t i_c = addr[i].iC;
      int64_t kx = addr[i].kx;
      int64_t ky = addr[i].ky;
      int64_t k = addr[i].k;

      // 倒三角跳变点的设置
      int64_t switch_index = tri_block_num_per_column + i_r + (i_r + 1) % 2;
      int64_t row_offset = (i_r + 1) % 2 == 1 ? -1 : 1;
      int64_t row_index_left_section =
          tri_block_num_per_column + i_r;  // 倒三角 ：非倒三角
      int64_t row_index_right_section =
          tri_block_num_per_column - 1 - i_r + row_offset;

      int64_t col_index_left_section = i_c;
      int64_t col_index_right_section = i_c - switch_index - 1;

      // row_max在bn维度上的偏移量
      int64_t row_max_bn_offset =
          (b * this->headNum_ + n) * this->querySequenceLen_;

      if (switch_index < i_c) {
        // row_max的偏移量
        int64_t row_max_inner_offset = row_index_right_section * SIZE_128 +
                                       SIZE_128 / 2 * this->vectorIndex_;
        section.rowmaxOffset[index] = row_max_bn_offset + row_max_inner_offset;
        section.rowmaxOffset[index + 1] =
            section.rowmaxOffset[index] + SIZE_128;  // ky = 2, 向下偏移一行

        // section的偏移量
        section.sectionBlockNums[index] =
            (i_c + kx >= this->blockNumPerRow_ - 1) ? kx - 1 : kx;
        section.sectionBlockOffset[index] =
            index == 0 ? this->vectorStartOffset_
                       : section.sectionBlockOffset[index - 1] +
                             section.sectionBlockNums[index - 1] *
                                 ATTENTION_SCORE_BLOCK_SIZE;
        section.sectionBlockNums[index + 1] = kx;
        section.sectionBlockOffset[index + 1] =
            section.sectionBlockOffset[index] + kx * ATTENTION_SCORE_BLOCK_SIZE;

        // 头尾判断
        section.isHeadSection[index] = (i_c == switch_index + 1) ? true : false;
        section.isHeadSection[index + 1] = section.isHeadSection[index];
        section.isTailSection[index] =
            (i_c + kx >= this->blockNumPerRow_ - 1) ? true : false;
        section.isTailSection[index + 1] = section.isTailSection[index];

        // 对角阵的偏移量
        section.diagOffset[index] =
            diag_out_offset +
            ((round_id % 2) * MAX_SWITCH_TIME * 2 + index) *
                ATTENTION_SCORE_BLOCK_SIZE +                      // 2 is index
            this->vectorIndex_ * ATTENTION_SCORE_BLOCK_SIZE / 2;  // 2 is index
        section.diagOffset[index + 1] =
            section.diagOffset[index] + ATTENTION_SCORE_BLOCK_SIZE;

        index += 2;  // 2 is offset
      } else {
        // row_max的偏移量
        int64_t row_max_inner_offset = row_index_left_section * SIZE_128 +
                                       SIZE_128 / 2 * this->vectorIndex_;
        section.rowmaxOffset[index] = row_max_bn_offset + row_max_inner_offset;
        section.rowmaxOffset[index + 1] =
            section.rowmaxOffset[index] + SIZE_128;  // ky = 2, 向下偏移一行

        // section的偏移量
        section.sectionBlockNums[index] =
            (i_c + kx >= switch_index) ? kx - 1 : kx;
        section.sectionBlockNums[index + 1] = kx;
        section.sectionBlockOffset[index] =
            index == 0 ? this->vectorStartOffset_
                       : section.sectionBlockOffset[index - 1] +
                             section.sectionBlockNums[index - 1] *
                                 ATTENTION_SCORE_BLOCK_SIZE;
        section.sectionBlockOffset[index + 1] =
            section.sectionBlockOffset[index] + kx * ATTENTION_SCORE_BLOCK_SIZE;

        // 头尾判断
        section.isHeadSection[index] = (i_c == 0) ? true : false;
        section.isHeadSection[index + 1] = section.isHeadSection[index];
        section.isTailSection[index] =
            (i_c + kx - 1 >= switch_index) ? true : false;
        section.isTailSection[index + 1] = section.isTailSection[index];

        // 对角阵的偏移量
        section.diagOffset[index] =
            diag_out_offset +
            ((round_id % 2) * MAX_SWITCH_TIME * 2 + index) *
                ATTENTION_SCORE_BLOCK_SIZE +                      // 2 is index
            this->vectorIndex_ * ATTENTION_SCORE_BLOCK_SIZE / 2;  // 2 is index
        section.diagOffset[index + 1] =
            section.diagOffset[index] + ATTENTION_SCORE_BLOCK_SIZE;

        index += 2;  // 2 is index
      }
    }

    // 清空section_num = 0的section
    int64_t pos = 0;
    for (int64_t i = 0; i < index; ++i) {
      if (section.sectionBlockNums[i] == 0) {
        continue;
      }
      section.sectionBlockNums[pos] = section.sectionBlockNums[i];
      section.sectionBlockOffset[pos] = section.sectionBlockOffset[i];
      section.rowmaxOffset[pos] = section.rowmaxOffset[i];
      section.isHeadSection[pos] = section.isHeadSection[i];
      section.isTailSection[pos] = section.isTailSection[i];

      ++pos;
    }

    // 设置全局的信息
    section.sectionNum = pos;
    section.maskNum = pos;
    section.matrixMaskOffset = this->maskOffset_;
    section.isTriangle = true;
    section.sparseFlag = false;
    section.processLineNum = this->processLineNum_;
    section.attentionScoreOffset = (round_id % 2) * this->coreNum_ *
                                   this->blockNumPerCore_ *
                                   ATTENTION_SCORE_BLOCK_SIZE;  // 2 is
  }

 public:  // 向外暴露的接口
  /**
   * 类的初始化
   * @param batch_size
   * @param head_num
   * @param query_sequence_len
   * @param key_value_sequence_len
   * @param mask_sequence_len
   * @param is_triangle
   * @param window_size
   * @param sparse_mode
   * @param block_num_per_core
   * @param ky
   */
  __aicore__ __inline__ void init(int64_t batch_size,
                                  int64_t head_num,
                                  int64_t query_sequence_len,
                                  int64_t key_value_sequence_len,
                                  int64_t mask_sequence_len,
                                  bool is_triangle,
                                  int64_t window_size,
                                  int64_t sparse_mode,
                                  int64_t block_num_per_core,
                                  int64_t ky) {
    this->batchSize_ = batch_size;
    this->headNum_ = head_num;
    this->querySequenceLen_ = query_sequence_len;
    this->keyValueSequenceLen_ = key_value_sequence_len;
    this->maskSequenceLen_ = mask_sequence_len;
    this->isTriangle_ = is_triangle;
    this->windowSize_ = window_size;
    this->sparseMode_ = sparse_mode;
    this->blockNumPerCore_ = block_num_per_core;
    this->ky_ = ky;
    if (ky == 0) {
      return;
    }
    this->kx_ = (ky != 0) ? (block_num_per_core / ky) : 0;
    this->isOdd_ =
        this->querySequenceLen_ / BASE_BLOCK_LENGTH / 2 % 2;  // 2 is index
  }

  /**
   * 设置核组信息
   * @param core_num
   * @param cur_core_index
   * @param vector_index
   */
  __aicore__ __inline__ void set_core_info(int64_t core_num,
                                           int64_t cur_core_index,
                                           int64_t vector_index) {
    // core
    this->coreNum_ = core_num;
    this->coreIndex_ = cur_core_index;

    // vector
    this->vectorNum_ = this->coreNum_ * 2;  // 2 is index
    this->vectorIndex_ = vector_index;
  }

  /**
   * 前向vector寻址启动，计算一些基本的偏移量
   * @return
   */
  __aicore__ __inline__ void start() {
    set_global_info();
    set_local_info();
    set_init_offset();
  }

  /**
   * 总轮次
   * @return
   */
  __aicore__ __inline__ int64_t get_total_round() { return this->totalRounds_; }

  /**
   * 归一化时需要处理的行数
   * @return
   */
  __aicore__ __inline__ int64_t get_norm_process_lines() {
    return this->normProcessLine_;
  }

  /**
   * 归一化的偏移量
   * @return
   */
  __aicore__ __inline__ int64_t get_norm_offset() {
    return this->normLineOffset_;
  }

  /**
   * 判断当前轮次、当前核是否要计算
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
    int64_t inner_row = this->coreIndex_ * this->ky_ * this->blockNumPerRow_;
    int64_t inner_col = (round_id * this->kx_) % this->blockNumPerRow_;
    int64_t cur_block_id = outer_row + inner_row + inner_col;

    return (cur_block_id < this->totalBlocks_);
  }

  /**
   * 获取当前轮次的section信息
   * @param round_id
   * @param section
   */
  __aicore__ __inline__ void get_section_info(int64_t round_id,
                                              FORWARD_SECTION_INFO& section) {
    // 寻址的预处理
    int64_t src_len = 0;
    ForWardAddrOnline forward_addr[MAX_SWITCH_TIME];
    forward_addrMapping_pre(forward_addr, src_len, round_id);

    // 地址偏移的计算
    if (this->isTriangle_) {
      return addrMapping_mask(forward_addr, src_len, round_id, section);
    }
    return addrMapping_nomask(forward_addr, src_len, round_id, section);
  }
};
}  // namespace Address

#endif

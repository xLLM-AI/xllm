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

#ifndef __ADDRESS_CONST_H__
#define __ADDRESS_CONST_H__

namespace Address {
// 定义常量
const int SIZE_128 = 128;
const int SIZE_256 = 256;

// 基本块的length
const int BASE_BLOCK_LENGTH = SIZE_128;

// 读取query,key,value的block size
const int QUERY_BLOCK_SIZE = SIZE_128 * BASE_BLOCK_LENGTH;
const int KEY_BLOCK_SIZE = SIZE_128 * BASE_BLOCK_LENGTH;
const int VALUE_BLOCK_SIZE = SIZE_128 * BASE_BLOCK_LENGTH;
const int ATTENTION_SCORE_BLOCK_SIZE = BASE_BLOCK_LENGTH * BASE_BLOCK_LENGTH;
const int ROWSUM_BLOCK_SIZE = SIZE_128;
const int OUTPUT_BLOCK_SIZE = SIZE_128 * BASE_BLOCK_LENGTH;

// 反向最长的分段数
const int MAX_LENGTH = 16;

// 反向寻址预处理的结构体
struct BackWardAddr {
  int32_t b;   // batch索引
  int32_t n;   // head索引
  int32_t iR;  // 行索引
  int32_t iC;  // 列索引
  int32_t kx;  // 行数
  int32_t ky;  // 列数
  int32_t k;   // 基本块的数量
};

// 反向cube1寻址模块
template <typename T_LEFT, typename T_RIGHT, typename T_OUTPUT>
struct PhyAddrBackwardCube1 {
  __gm__ T_LEFT* left;     // left起始位置
  __gm__ T_RIGHT* right;   // right起始位置
  __gm__ T_OUTPUT* out;    // out起始位置
  int32_t kx = 0;          // x方向的长度
  int32_t ky = 0;          // y方向的长度
  int32_t k = 0;           // 总共的块数
  int32_t lineStride = 0;  // 行和行的间隔
  bool
      lowerLeft;  // 左下角是否不需要计算：true代表不需要计算，false代表需要计算
  bool upperRight;  // 右上角是否不需要计算：
                    // true代表不需要计算，false代表需要计算
};

// 反向cube2寻址模块
template <typename T_LEFT, typename T_RIGHT, typename T_OUTPUT>
struct PhyAddrBackwardCube2 {
  __gm__ T_LEFT* left;     // left起始位置
  __gm__ T_RIGHT* right;   // right起始位置
  __gm__ T_OUTPUT* out;    // out起始位置
  int32_t kx = 0;          // x方向的长度
  int32_t ky = 0;          // y方向的长度
  int32_t k = 0;           // 总共的块数
  int32_t lineStride = 0;  // 行和行的间隔
  bool
      lowerLeft;  // 左下角是否不需要计算：true代表不需要计算，false代表需要计算
  bool upperRight;  // 右上角是否不需要计算：
                    // true代表不需要计算，false代表需要计算
};

// 反向cube3寻址模块
template <typename T_LEFT, typename T_RIGHT, typename T_OUTPUT>
struct PhyAddrBackwardCube3 {
  __gm__ T_LEFT* left;     // left起始位置
  __gm__ T_RIGHT* right;   // right起始位置
  __gm__ T_OUTPUT* out;    // out起始位置
  int32_t kx = 0;          // x方向的长度
  int32_t ky = 0;          // y方向的长度
  int32_t k = 0;           // 总共的块数
  int32_t lineStride = 0;  // 行和行的间隔
  bool
      lowerLeft;  // 左下角是否不需要计算：true代表不需要计算，false代表需要计算
  bool upperRight;  // 右上角是否不需要计算：
                    // true代表不需要计算，false代表需要计算
};

// 反向vector寻址预处理的结构体
struct VectorAddr {
  int32_t b = 0;   // batch索引
  int32_t n = 0;   // head索引
  int32_t iR = 0;  // 行索引
  int32_t iC = 0;  // 列索引
  int32_t kx = 0;  // 行数
  int32_t ky = 0;  // 列数
  int32_t k = 0;   // 基本块的数量
};

struct ForWardAddrOnline {
  int32_t b;   // batch索引
  int32_t n;   // head索引
  int32_t iR;  // 行索引
  int32_t iC;  // 列索引
  int32_t kx;  // 行数
  int32_t ky;  // 列数
  int32_t k;   // 基本块的数量
};

// vector的全局信息
struct GLOBAL_INFO {
  int64_t cubeNum = 0;           // cube数量
  int64_t blockNumPerCube = 0;   // 每个cube处理block的数量
  int64_t headNum = 0;           // head数量
  int64_t batchNum = 0;          // batch数量
  int64_t seqLenQ = 0;           // query的序列长度
  int64_t seqLenK = 0;           // key、value的序列长度
  int64_t headDim = 0;           // head-dim = 128;
  bool triMatix = false;         // 是否是三角阵
  int32_t blockRows = 0;         // 负载均衡前的行数
  int32_t blockCols = 0;         // 负载均衡前的列数
  int64_t blockNumPerRow = 0;    // 一行有几个block（三角阵和非三角阵不同）
  int64_t blockNumPerCol = 0;    // 一列有几个block（也就是一个head有几个行）
  int64_t blockNumPerLoop = 0;   // 一个loop处理block的数量
  int64_t blockNumPerHead = 0;   // 一个head包含的block数量
  int64_t blockNumPerBatch = 0;  // 一个batch包含的block数量
  int64_t loopTimes = 0;         // 大循环次数
  int64_t tailBlockNum = 0;      // 最后一次循环处理的block数量
  int64_t isSparse = 0;          // 是否为sparse 魔道e
  int64_t windowLength = 0;      // 滑动窗口的length
  int64_t windowsBlockNum = 0;   // 滑窗大小, sparse使用
};

// vector的当前信息
struct LOCAL_INFO {
  int64_t cubeIndex = 0;  // 所属的cube分组（第几个cube很重要）
  int64_t vectorIdx = 0;  // 当前vector在cube内的编号（0或1）
  int64_t startLineInBaseBlock =
      0;                  // 处理基本块的起始行（都是2个vector处理一个基本块）
  bool procTail = false;  // 是否参与尾块处理
  int64_t procTailBlockNum = 0;  // 处理尾块中的block数量
};

// vector的分段信息
const int MAX_SWITCH_TIME = 8;
struct SECTION_INFO {
  int64_t sectionNum = 0;  // 当前处理的block条，包含几个块
  int64_t sectionStartBlock[MAX_SWITCH_TIME] = {0};  // 起始block编号
  int64_t headSkipBlock[MAX_SWITCH_TIME] = {
      0};  // 当前section的头尾是否要跳过计算
  int64_t tailSkipBlock[MAX_SWITCH_TIME] = {
      0};  // 当前section的头尾是否要跳过计算
  int64_t globalLinesInHeads[MAX_SWITCH_TIME] = {0};  // 在所有heads中的起始行
  int64_t len[MAX_SWITCH_TIME] = {0};                 // 每一段的长度
  bool headApplyMask[MAX_SWITCH_TIME] = {false};
  bool tailApplyMask[MAX_SWITCH_TIME] = {false};
  int64_t oDOOffset[MAX_SWITCH_TIME] = {0};  // 矩阵O dO的偏移量
  int64_t sDPOffset[MAX_SWITCH_TIME] = {0};  // 矩阵S dP的偏移量
  int64_t processLines = {0};                // 处理的行数
};

// 前向vector的分段信息
const int FORWARD_MAX_SWITCH_TIME = 8;
struct FORWARD_SECTION_INFO {
  int64_t sectionNum = 0;  // section的数量
  int64_t sectionBlockNums[FORWARD_MAX_SWITCH_TIME] = {
      0};  // 每段section计算的块数
  int64_t sectionBlockOffset[FORWARD_MAX_SWITCH_TIME] = {
      0};                                               // 每段section的偏移量
  int64_t rowmaxOffset[FORWARD_MAX_SWITCH_TIME] = {0};  // rowmax的偏移量
  int64_t maskNum;                                      // mask的数量
  int64_t matrixMaskOffset;                             // mask矩阵的偏移量
  int64_t processLineNum;                               // 处理的行数
  bool sparseFlag = false;                              // sparse的标志
  bool isTriangle = false;                              // 倒三角的标志
  int64_t attentionScoreOffset;                       // attention_score的偏移量
  int64_t diagOffset[FORWARD_MAX_SWITCH_TIME] = {0};  // 对角阵偏移量
  bool isHeadSection[FORWARD_MAX_SWITCH_TIME] = {false};  // 是否更新
  bool isTailSection[FORWARD_MAX_SWITCH_TIME] = {false};  // 是否是末尾
};

// 前向cube1寻址的结构体
template <typename T_LEFT, typename T_RIGHT, typename T_OUTPUT>
struct PhyAddrForwardCube1 {
  __gm__ T_LEFT* left;    // 左矩阵的起始地址，基本块为 128 * 192
  __gm__ T_RIGHT* right;  // 右矩阵的起始地址， 基本块为 128 * 192
  __gm__ T_OUTPUT* out;   // 输出的起始地址， 基本块为 128 * 128
  int32_t k = 0;          // 连续计算的基本块数量
};

// 前向cube2寻址的结构体
template <typename T_LEFT, typename T_RIGHT, typename T_OUTPUT>
struct PhyAddrForwardCube2Rowsum {
  __gm__ T_LEFT* left;          // 左矩阵的起始地址，基本块为 128 * 128
  __gm__ T_RIGHT* right;        // 右矩阵的起始地址， 基本块为 128 * 192
  __gm__ T_OUTPUT* out;         // cube2输出的起始地址， 基本块为 128 * 192
  __gm__ T_OUTPUT* rowsum_out;  // row_sum输出的起始地址， 基本块为 128 * 1
  int32_t k = 0;                // 连续计算的基本块数量
};

// 前向online优化：cube1寻址模块
template <typename T_LEFT, typename T_RIGHT, typename T_OUTPUT>
struct PhyAddrForwardCube1Online {
  __gm__ T_LEFT* left;     // left起始位置
  __gm__ T_RIGHT* right;   // right起始位置
  __gm__ T_OUTPUT* out;    // out起始位置
  int32_t kx = 0;          // x方向的长度
  int32_t ky = 0;          // y方向的长度
  int32_t k = 0;           // 总共的块数
  int32_t lineStride = 0;  // 行和行的间隔
  bool
      lowerLeft;  // 左下角是否不需要计算：true代表不需要计算，false代表需要计算
  bool upperRight;      // 右上角是否不需要计算：
                        // true代表不需要计算，false代表需要计算
  bool onStartSection;  // 是否处在一行的的起始
  bool onEndSection;    // 是否处在一行的末尾
};

// 前向online优化：：cube2寻址模块
template <typename T_LEFT, typename T_RIGHT, typename T_OUTPUT>
struct PhyAddrForwardCube2Online {
  __gm__ T_LEFT* left;     // left起始位置
  __gm__ T_RIGHT* right;   // right起始位置
  __gm__ T_OUTPUT* out;    // out起始位置
  int32_t kx = 0;          // x方向的长度
  int32_t ky = 0;          // y方向的长度
  int32_t k = 0;           // 总共的块数
  int32_t lineStride = 0;  // 行和行的间隔
  bool
      lowerLeft;  // 左下角是否不需要计算：true代表不需要计算，false代表需要计算
  bool upperRight;      // 右上角是否不需要计算：
                        // true代表不需要计算，false代表需要计算
  bool onStartSection;  // 是否处在一行的的起始
  bool onEndSection;    // 是否处在一行的末尾
};
}  // namespace Address
#endif

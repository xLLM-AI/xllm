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
#ifndef __VECTORFORWARD_H__
#define __VECTORFORWARD_H__

#include "AddressMappingVectorForwardOnline.h"
#include "matmul_const.h"

#ifdef __DAV_C220_VEC__
#define ROUND_UP_8(x) (((x) + 7) / 8 * 8)

constexpr size_t MASK_BASE = 128;
constexpr size_t MASK_HALF_BASE = 64;

template <typename INPUT_T, bool IF_BF16, typename WORKSPACE_T>
class VectorForward {
 public:
  __aicore__ inline VectorForward(){};
  __aicore__ inline void Init(__gm__ uint8_t* __restrict__ a_cube1,  // Q
                              __gm__ uint8_t* __restrict__ b_cube1,  // K
                              __gm__ uint8_t* __restrict__ b_cube2,  // V
                              __gm__ uint8_t* __restrict__ mask_gm,
                              __gm__ uint8_t* __restrict__ score_gm,
                              __gm__ uint8_t* __restrict__ probability_gm,
                              __gm__ float* __restrict__ c_cube2,
                              __gm__ float* __restrict__ log_sum_gm,
                              __gm__ float* __restrict__ diag_rowsum_gm,
                              __gm__ float* __restrict__ d_rowmax_gm,
                              int32_t qSeqLength,
                              int32_t kSeqLength,
                              int32_t H,
                              int32_t B,
                              int32_t Y,
                              int32_t qk,
                              int32_t windows_block_num,
                              int32_t maskSeqLength,
                              float scale,
                              int32_t windowLen
                              // INPUT_T scale
  );
  __aicore__ __inline__ void __set_mask(int32_t len) {
    if (len >= 128) {  // 128 is len
      set_vector_mask((uint64_t)-1, (uint64_t)-1);
      return;
    }
    int32_t highMask = len - 64 > 0 ? len - 64 : 0;
    int32_t lowMask = len - 64 >= 0 ? 64 : len;
    if (len < MASK_HALF_BASE) {  // 64 is len
      set_vector_mask(0x0, ((uint64_t)1 << lowMask) - 1);
    } else {
      set_vector_mask(((uint64_t)1 << highMask) - 1, 0xffffffffffffffff);
    }
  }

  __aicore__ __inline__ void __set_reverse_mask(int32_t len) {
    if (len >= MASK_BASE) {
      set_vector_mask((uint64_t)-1, (uint64_t)-1);
      return;
    }
    int32_t lowMask = len - 64 > 0 ? len - 64 : 0;
    int32_t highMask = len - 64 >= 0 ? 64 : len;
    if (len < MASK_HALF_BASE) {
      set_vector_mask(0xffffffffffffffff, ~(((uint64_t)1 << highMask) - 1));
    } else {
      set_vector_mask(~(uint64_t)1 << (lowMask - 1), 0x0);
    }
  }

  __aicore__ inline void Run();
  __aicore__ inline void SetHighPrecision(bool isHighPrecision) {
    this->isHighPrecision = isHighPrecision;
  };

  struct UB_FOR_SHORT_LEN_ATTN_SCORE {
    __ubuf__ float* cur_buf_for_vbrcb_rowmax_fp32;
    __ubuf__ INPUT_T* buf_for_load_attn_score_fp16;
    __ubuf__ INPUT_T* buf_for_subMaxValueResult_fp16;
    __ubuf__ float* buf_for_diag_fp32;
    __ubuf__ INPUT_T* buf_for_load_one_block_tri_mask_fp16;

    __ubuf__ float* buf_for_cacl_final_rowmax_fp32;
    __ubuf__ INPUT_T* buf_for_cacl_final_rowmax_fp16;
    __ubuf__ float* buf_for_cacl_rowmax_fp32;
    __ubuf__ INPUT_T* buf_for_cacl_rowmax_fp16;
    __ubuf__ float* buf_for_vbrcb_rowmax_fp32;
    __ubuf__ INPUT_T* buf_for_vbrcb_rowmax_fp16;
    __ubuf__ float* buf_for_record_rowmax_fp32;
    __ubuf__ INPUT_T* buf_for_record_rowmax_fp16;
    // ping-pong buffer
    // Score
    __ubuf__ INPUT_T* pp_buf_for_attn_score_fp16[2];

    __ubuf__ INPUT_T* pp_buf_for_load_one_block_tri_mask_fp16[2];

    __ubuf__ INPUT_T* maxJVectorUbAddr;
    __ubuf__ INPUT_T* lastMaxJVectorUbAddr;
    __ubuf__ INPUT_T* subMaxValueResultVectorUbAddr;
    __ubuf__ float* subMaxValueResultVectorUbAddr_fp32;
    __ubuf__ INPUT_T* ljVectorUbAddr;
    __ubuf__ float* ljVectorUbAddr_fp32;
    __ubuf__ float* lastLjVectorUbAddr;
    __ubuf__ float* diagExpMaxJMatPingUbAddr;
    __ubuf__ float* diagExpMaxJMatPongUbAddr;
  };

  struct UB_FOR_NORMALIZE {
    __ubuf__ float* buf_for_load_O_fp32;       // 装载计算好的O
    __ubuf__ float* buf_for_load_rowsum_fp32;  // 装载计算好的rowsum
    __ubuf__ float* buf_for_brcb_rowsum_fp32;  // 32字节对齐的rowsum

    int32_t o_ping_pong_interval;
    int32_t rowsum_ping_pong_interval;
    int32_t rowsum_brcb_ping_pong_interval;
  };  // UB空间划分，求归一化

  struct PARAM_MEDIUM_SEQ_EXP {
    int32_t
        block_num_per_step;  // 当前处理的块数量（对齐MAX_BLOCK_PER_ONE_PROC）
    int32_t
        block_num_for_last;  // 最后一次ping-pong（不满MAX_BLOCK_PER_ONE_PROC时）
    int32_t last_padding_block_num;     // 当前需要padding的块数量
    int32_t section_start_line_offset;  // 记录当前行的起始地址
    int32_t section_mask_offset;
    int32_t total_frag_num;  // 总分段数量
    int32_t cur_frag;        // 当前分段id
    bool tail_block;
    int32_t tri_matrix_num;  // 三角阵的数量
                             // 0-非三角阵；1-三角阵，非unirow；2-三角阵，unirow
    int32_t apply_tri_mask;
    int32_t buf_offset;  // 存score * scal的偏移
    int32_t record_rowmax_offset;
  };

  struct PARAM_SHORT_SEQ_MAX {
    int32_t section_block_num;
    int32_t section_padding_block_num;
    int32_t section_start_line_offset;  //  attn score 起始地址
    int32_t section_mask_offset;        //  mask的地址 （两个secion的相同）
    int32_t record_rowmax_offset;       // 记录rowmax的偏移
    int32_t apply_tri_mask;
    bool is_head_section;
    bool is_tail_section;
  };

  struct PARAM_LONG_SEQ_EXP {
    int32_t section_block_num;          // 当前处理的块数量
    int32_t section_padding_block_num;  // 当前需要padding的块数量

    int32_t section_start_line_offset;  //  attn score 起始地址
    int32_t section_mask_offset;        //  mask的地址 （两个secion的相同）

    int32_t total_frag_num;  // 总分段数量
    int32_t cur_frag;        // 当前分段id

    int32_t tri_matrix_num;  // 三角阵的数量
                             // 0-非三角阵；1-三角阵，非unirow；2-三角阵，unirow
    int32_t apply_tri_mask;
  };

  struct UB_FOR_LN_ROWSUM {
    __ubuf__ float* ub_buf_for_rowsum;
    __ubuf__ float* ub_buf_for_rowsum_res;
  };

 private:
  __aicore__ __inline__ void initWorkSpace();
  __aicore__ __inline__ void allocate_ubuf_for_norm(UB_FOR_NORMALIZE* ub_norm);
  __aicore__ inline void get_sub_seq_length_per_proc(
      int32_t k_seq_len,
      int32_t block_num_per_full_line,
      int32_t* sub_seq_length_per_proc);
  __aicore__ __inline__ void get_padding_info_for_row_max(
      int32_t total_block_num,  // 当前总共需要处理的block数量
      int32_t* padding_block_num);
  __aicore__ __inline__ void padding_for_row_max_or_rowsum(
      int32_t total_block_num,
      int32_t padding_block_num,
      int32_t ping_pong_flag,
      int32_t paddingType,
      __ubuf__ float* pp_buf_for_attn_score_fp16[]);
  __aicore__ __inline__ void padding_for_row_max_or_rowsum2(
      int32_t total_block_num,
      int32_t padding_block_num,
      int32_t ping_pong_flag,
      int32_t paddingType,
      __ubuf__ INPUT_T* pp_buf_for_attn_score_fp16[]);
  __aicore__ __inline__ void process_cacl_max(
      int32_t basic_block_num,
      int32_t padding_block_num,
      bool pp_first_section,
      __ubuf__ INPUT_T* cur_buf_for_attn_score,
      __ubuf__ INPUT_T* cur_buf_for_rowmax,
      __ubuf__ INPUT_T* buf_for_cacl_final_rowmax_fp16);
  __aicore__ __inline__ void cacl_max(__ubuf__ INPUT_T* buf_for_cacl,
                                      int32_t _block_num);

  __aicore__ __inline__ void process_calc_sum(
      int32_t qk_triangle,
      PARAM_SHORT_SEQ_MAX param,
      int32_t ping_pong_flag,
      bool first_line,
      __gm__ WORKSPACE_T* attn_score_gm,
      __gm__ INPUT_T* attn_mask_gm,
      UB_FOR_SHORT_LEN_ATTN_SCORE ub_attn,
      bool sparse_flag,
      int32_t lines);

  __aicore__ __inline__ void process_line_phase_one_for_short_seq_max(
      bool is_head_section,
      bool is_tail_section,
      int32_t qk_triangle,
      PARAM_SHORT_SEQ_MAX param,
      int32_t ping_pong_flag,
      bool first_line,
      __gm__ WORKSPACE_T* attn_score_gm,
      __gm__ INPUT_T* attn_mask_gm,
      UB_FOR_SHORT_LEN_ATTN_SCORE ub_attn,
      bool sparse_flag,
      int32_t lines,
      __ubuf__ INPUT_T* maxJVectorUbAddr,
      __ubuf__ INPUT_T* lastMaxJVectorUbAddr,
      __ubuf__ INPUT_T* subMaxValueResultVectorUbAddr,
      int rowsPerDiag,
      int rowmax_build_step);
  __aicore__ __inline__ void
  process_line_phase_one_for_short_seq_exp_and_rowsum(
      int32_t qk_triangle,
      PARAM_SHORT_SEQ_MAX param,
      int32_t ping_pong_flag,
      __gm__ WORKSPACE_T* attn_score_gm,
      __gm__ INPUT_T* attn_mask_gm,
      UB_FOR_SHORT_LEN_ATTN_SCORE ub_attn,
      int32_t offset,
      int32_t lines,
      bool sparse_flag);
  __aicore__ __inline__ void attention_score_short_double_line_one(
      int32_t sectionLoop,
      int32_t sectionNum,
      int32_t qk_triangle,
      int32_t section_block_nums,
      int32_t tri_matrix_mask_offset,  // 128*128的三角阵中取第n行
      int32_t each_vector_proc_line_num,
      int32_t local_section_start_line_offset,
      bool isa_head_section,
      bool is_tail_section,
      int32_t diag_offset,
      __gm__ WORKSPACE_T* __restrict__ attn_score_gm,
      __gm__ INPUT_T* __restrict__ attn_mask_gm,
      UB_FOR_SHORT_LEN_ATTN_SCORE ub_attn,
      bool sparse_flag);
  __aicore__ __inline__ void backupMaxandSum(
      __ubuf__ float* ljVectorUbAddr,
      __ubuf__ float* lastLjVectorUbAddr,
      __ubuf__ INPUT_T* maxJVectorUbAddr,
      __ubuf__ INPUT_T* lastMaxJVectorUbAddr);
  __aicore__ __inline__ void updateRowsum(
      __ubuf__ float* ljVectorUbAddr,
      __ubuf__ float* lastLjVectorUbAddr,
      __ubuf__ float* subMaxValueResultVectorUbAddr,
      __ubuf__ float* tempUbAddr);

  __aicore__ __inline__ void cacl_wrap(__ubuf__ INPUT_T* buf_for_cacl,
                                       int32_t _block_num,
                                       int32_t optcode);
  __aicore__ __inline__ void add_max_wrap(__ubuf__ INPUT_T* dest_addr,
                                          __ubuf__ INPUT_T* src_addr,
                                          __ubuf__ INPUT_T* src_addr2,
                                          int32_t rpt,
                                          int32_t dst_stride,
                                          int32_t src_stride,
                                          int32_t src_stride2,
                                          int32_t dst_rpt_stride,
                                          int32_t src_rpt_stride,
                                          int32_t src_rpt_stride2,
                                          int32_t optcode);
  __aicore__ __inline__ void process_cacl_wrap(
      int32_t basic_block_num,
      int32_t padding_block_num,
      bool pp_first_section,
      __ubuf__ INPUT_T* cur_buf_for_attn_score,
      __ubuf__ INPUT_T* cur_buf_for_rowmax,
      __ubuf__ INPUT_T* buf_for_cacl_final_rowmax_fp16,
      int32_t optcode);
  __aicore__ __inline__ void get_uni_rowmax_seq_info_per_proc(
      int32_t block_num_per_full_line,
      int32_t sub_seq_length_per_proc,  // 一次处理处理的长度
      int32_t* ping_block_offset_num,   // ping起始块
      int32_t* pong_block_offset_num,   // pong起始块
      int32_t* tail_block_offset_num,   // tail起始块
      int32_t* tail_block_num,          // tail的块数
      int32_t* ping_pong_times          // pingpang的循环次数
  );
  __aicore__ __inline__ void allocate_ubuf_for_short_seq_attn_score(
      UB_FOR_SHORT_LEN_ATTN_SCORE* ub_attn_score);

  __aicore__ __inline__ void form_diagmat(
      __ubuf__ float* diagmataddr,
      __ubuf__ float* vecaddr,
      __gm__ float* gm_addr,
      int step,
      int rowsPerDiag,
      int32_t local_section_start_line_offset,
      int32_t ping_pong_flag,
      int type);

 private:
  __gm__ INPUT_T* __restrict__ gm_a_cube1;
  __gm__ INPUT_T* __restrict__ gm_b_cube1;
  __gm__ INPUT_T* __restrict__ gm_b_cube2;
  __gm__ INPUT_T* __restrict__ attn_mask_gm;
  __gm__ WORKSPACE_T* __restrict__ attn_score_gm;
  __gm__ INPUT_T* __restrict__ attn_probability_gm;
  __aicore__ inline void RunBfloat16();
  __gm__ float* __restrict__ gm_c_cube2;
  __gm__ float* __restrict__ log_sum_max_gm;

  // Tag：传入循环中；
  __gm__ float* diag_rowsum_gm;
  __gm__ float* diag_rowmax_gm;
  __ubuf__ float* diagExpMaxJMatUbAddr;
  __ubuf__ float* tempUb;

  __ubuf__ float* __restrict__ ub_for_softsync_flags;
  __ubuf__ float* __restrict__ ub_for_softsync_check;

  // int32_t seq_len;
  int32_t q_seq_len;
  int32_t k_seq_len;
  int32_t head_num;
  int32_t batch_num;
  int32_t y_cube_num_per_line;
  int32_t qk_triangle;
  int32_t V_num;
  int32_t softSyncTimesCount = 0;
  bool use_soft_sync = false;
  bool isHighPrecision = true;
  int32_t maskSeqLength;
  int32_t rowPerTime = 1;
  float SCALE;
  // INPUT_T SCALE;
  INPUT_T FP16_SMALL_NUM = 5.9604644775390625e-8;
  int32_t block_per_core = IF_BF16 ? 8 : 64;
  int32_t ky = 2;
  int32_t SIZE_FP32 = 4;
  int32_t SIZE_FP16 = 2;
  int32_t BLOCK_NUM_8 = 8;
  int32_t BYTES_PER_BLOCK = 32;
  int32_t data_num_per_blk = BYTES_PER_BLOCK / SIZE_FP16;
  int32_t windowLen;

  // 注入寻址模块
  Address::AddressMappingVectorForwardOnline address;
};

template <typename INPUT_T, bool IF_BF16, typename WORKSPACE_T>
__aicore__ inline void VectorForward<INPUT_T, IF_BF16, WORKSPACE_T>::Init(
    __gm__ uint8_t* __restrict__ a_cube1,
    __gm__ uint8_t* __restrict__ b_cube1,
    __gm__ uint8_t* __restrict__ b_cube2,
    __gm__ uint8_t* __restrict__ mask_gm,
    __gm__ uint8_t* __restrict__ score_gm,
    __gm__ uint8_t* __restrict__ probability_gm,
    __gm__ float* __restrict__ c_cube2,
    __gm__ float* __restrict__ log_sum_gm,
    __gm__ float* __restrict__ d_rowsum_gm,
    __gm__ float* __restrict__ d_rowmax_gm,
    // int32_t S,
    int32_t qSeqLength,
    int32_t kSeqLength,
    int32_t H,
    int32_t B,
    int32_t Y,
    int32_t qk,
    int32_t windows_block_num,
    int32_t maskSeqLength,
    float scale,
    int32_t windowLen
    // INPUT_T scale
) {
  gm_a_cube1 = (__gm__ INPUT_T* __restrict__)a_cube1;
  gm_b_cube1 = (__gm__ INPUT_T* __restrict__)b_cube1;
  gm_b_cube2 = (__gm__ INPUT_T* __restrict__)b_cube2;
  attn_mask_gm = (__gm__ INPUT_T* __restrict__)mask_gm;
  attn_score_gm = (__gm__ WORKSPACE_T* __restrict__)score_gm;
  attn_probability_gm = (__gm__ INPUT_T*)probability_gm;
  gm_c_cube2 = c_cube2;
  log_sum_max_gm = log_sum_gm;

  q_seq_len = qSeqLength;
  k_seq_len = kSeqLength;
  head_num = H;
  batch_num = B;
  y_cube_num_per_line = Y;
  qk_triangle = qk;
  V_num = windows_block_num;
  this->maskSeqLength = maskSeqLength;
  this->SCALE = scale;
  this->windowLen = windowLen;

  // 寻址模块的初始化
  int32_t sparse_mode = this->V_num == 0 ? 0 : 1;
  address.init(this->batch_num,
               this->head_num,
               this->q_seq_len,
               this->k_seq_len,
               this->maskSeqLength,
               this->qk_triangle,
               this->V_num,
               sparse_mode,
               this->block_per_core,
               this->ky);

  // 寻址模块核组的设置
  int32_t core_num = get_block_num();
  int32_t cur_core_index = get_block_idx();
  int32_t vector_id = get_subblockid();  // 0, 1
  address.set_core_info(core_num, cur_core_index, vector_id);

  diag_rowsum_gm = d_rowsum_gm;
  diag_rowmax_gm = d_rowmax_gm;

  // 寻址模块启动
  address.start();
}

template <typename INPUT_T, bool IF_BF16, typename WORKSPACE_T>
__aicore__ inline void
VectorForward<INPUT_T, IF_BF16, WORKSPACE_T>::RunBfloat16() {
  set_atomic_none();
  set_mask_norm();
  __set_mask(64);
  // 512-key blocks and FP32 vector updates match the dense BF16 arithmetic.
  // Keep the running output in UB; only the final normalized result reaches GM.
  // UB buffers occupy 192 KiB; pending row metadata is double-buffered.
  constexpr int32_t rows = 32;
  constexpr int32_t columns = 512;
  // A finite sentinel can exceed valid BF16 logits and poison online softmax.
  constexpr int32_t kNegativeInfinity = static_cast<int32_t>(0xff800000u);
  auto scores = reinterpret_cast<__ubuf__ float*>(512);
  auto reduced = reinterpret_cast<__ubuf__ float*>(68 * 1024);
  auto temporary = reinterpret_cast<__ubuf__ float*>(78 * 1024);
  auto maxima = reinterpret_cast<__ubuf__ float*>(80 * 1024);
  auto sums = reinterpret_cast<__ubuf__ float*>(81 * 1024);
  auto factors = reinterpret_cast<__ubuf__ float*>(82 * 1024);
  auto broadcast = reinterpret_cast<__ubuf__ float*>(83 * 1024);
  auto previous_max = reinterpret_cast<__ubuf__ float*>(85 * 1024);
  auto previous_sum = reinterpret_cast<__ubuf__ float*>(86 * 1024);
  auto probabilities = reinterpret_cast<__ubuf__ bfloat16_t*>(96 * 1024);
  auto accumulated_output = reinterpret_cast<__ubuf__ float*>(128 * 1024);
  auto pending_scales = reinterpret_cast<__ubuf__ float*>(88 * 1024);
  auto pending_sums = reinterpret_cast<__ubuf__ float*>(90 * 1024);
  set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
  for (int32_t round = 0; round <= address.get_total_round(); ++round) {
    Address::FORWARD_SECTION_INFO section;
    if (round < address.get_total_round()) {
      wait_flag_dev(AIC2AIVFLAGID);
      address.get_section_info(round, section);
      if (!address.is_running(round)) {
        section.sectionNum = 0;
      }
      for (int32_t index = 0; index < section.sectionNum; ++index) {
        const int32_t blocks = section.sectionBlockNums[index];
        const int32_t count = blocks * 128;
        const bool first = section.isHeadSection[index];
        const bool last = section.isTailSection[index];
        for (int32_t row = 0; row < section.processLineNum; row += rows) {
          const int32_t state_offset = (index % 2) * 64 + row;
          const int64_t offset =
              section.attentionScoreOffset + section.sectionBlockOffset[index] +
              get_subblockid() * 64 * (count - 128) + row * count;
          if (count < columns) {
            for (int32_t half = 0; half < 2; ++half) {
              vector_dup(
                  reinterpret_cast<__ubuf__ int32_t*>(scores + half * 8192),
                  kNegativeInfinity,
                  128,
                  1,
                  1,
                  8,
                  1);
            }
            set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
            wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
          }
          set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
          wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
          // BF16 scores use contiguous rows within each 512-key section.
          copy_gm_to_ubuf(scores,
                          attn_score_gm + offset,
                          0,
                          rows,
                          count / 8,
                          0,
                          (columns - count) / 8);
          set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
          wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
          for (int32_t half = 0; half < 2; ++half) {
            vmuls(scores + half * 8192,
                  scores + half * 8192,
                  SCALE,
                  128,
                  1,
                  1,
                  8,
                  8);
          }
          pipe_barrier(PIPE_V);
          if (last && windowLen != 0) {
            const int32_t repeats = windowLen / 64;
            const int32_t remainder = windowLen % 64;
            const int32_t start = count - repeats * 64;
            for (int32_t col = start; col < count; col += 64) {
              vector_dup(reinterpret_cast<__ubuf__ int32_t*>(scores + col),
                         kNegativeInfinity,
                         rows,
                         1,
                         1,
                         columns / 8,
                         1);
            }
            if (remainder != 0) {
              set_vector_mask(0, ((uint64_t)-1) << (64 - remainder));
              vector_dup(
                  reinterpret_cast<__ubuf__ int32_t*>(scores + start - 64),
                  kNegativeInfinity,
                  rows,
                  1,
                  1,
                  columns / 8,
                  1);
              __set_mask(64);
            }
            pipe_barrier(PIPE_V);
          }
          for (int32_t half = 0; half < 2; ++half) {
            vcgmax(reduced + half * 1024, scores + half * 8192, 128, 1, 1, 8);
          }
          pipe_barrier(PIPE_V);
          vcgmax(temporary, reduced, rows, 1, 1, 8);
          pipe_barrier(PIPE_V);
          vcgmax(maxima, temporary, rows / 8, 1, 1, 8);
          pipe_barrier(PIPE_V);
          __set_mask(rows);
          if (!first) {
            vmax(maxima,
                 maxima,
                 previous_max + state_offset,
                 1,
                 1,
                 1,
                 1,
                 8,
                 8,
                 8);
            pipe_barrier(PIPE_V);
            vsub(factors,
                 previous_max + state_offset,
                 maxima,
                 1,
                 1,
                 1,
                 1,
                 8,
                 8,
                 8);
            pipe_barrier(PIPE_V);
            vexp(factors, factors, 1, 1, 1, 8, 8);
            pipe_barrier(PIPE_V);
          }
          vadds(previous_max + state_offset, maxima, 0.0f, 1, 1, 1, 8, 8);
          pipe_barrier(PIPE_V);
          set_vector_mask((uint64_t)-1, (uint64_t)-1);
          vbrcb((__ubuf__ uint32_t*)broadcast,
                (__ubuf__ uint32_t*)maxima,
                1,
                8,
                rows / 8);
          pipe_barrier(PIPE_V);
          __set_mask(64);
          for (int32_t col = 0; col < columns; col += 64) {
            vsub(scores + col,
                 scores + col,
                 broadcast,
                 rows,
                 1,
                 1,
                 0,
                 columns / 8,
                 columns / 8,
                 1);
          }
          pipe_barrier(PIPE_V);
          for (int32_t half = 0; half < 2; ++half) {
            vexp(scores + half * 8192, scores + half * 8192, 128, 1, 1, 8, 8);
          }
          pipe_barrier(PIPE_V);
          const int32_t live_columns = count - (last ? windowLen : 0);
          if (live_columns == columns) {
            for (int32_t half = 0; half < 2; ++half) {
              vcgadd(reduced + half * 1024, scores + half * 8192, 128, 1, 1, 8);
            }
            pipe_barrier(PIPE_V);
            vcgadd(temporary, reduced, rows, 1, 1, 8);
            pipe_barrier(PIPE_V);
            vcgadd(sums, temporary, rows / 8, 1, 1, 8);
            pipe_barrier(PIPE_V);
          } else {
            // Preserve the reference's tail reduction order: sum each
            // 64-key group before adding the groups in sequence.
            vector_dup(sums, 0.0f, 1, 1, 1, 8, 1);
            pipe_barrier(PIPE_V);
            for (int32_t col = 0; col < live_columns; col += 64) {
              int32_t width = live_columns - col;
              if (width > 64) {
                width = 64;
              }
              __set_mask(width);
              vcgadd(reduced, scores + col, rows, 1, 1, columns / 8);
              pipe_barrier(PIPE_V);
              const uint64_t block_mask =
                  ((uint64_t)1 << ((width + 7) / 8)) - 1;
              const uint64_t repeated_mask = block_mask * 0x0101010101010101;
              set_vector_mask(0, repeated_mask);
              vcgadd(temporary, reduced, rows / 8, 1, 1, 8);
              pipe_barrier(PIPE_V);
              __set_mask(rows);
              vadd(sums, sums, temporary, 1, 1, 1, 1, 8, 8, 8);
              pipe_barrier(PIPE_V);
            }
            __set_mask(64);
          }
          wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
          for (int32_t half = 0; half < 2; ++half) {
            vconv_f322bf16r(probabilities + half * 8192,
                            scores + half * 8192,
                            128,
                            1,
                            1,
                            4,
                            8);
          }
          set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
          wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
          copy_ubuf_to_gm(attn_probability_gm + offset,
                          probabilities,
                          0,
                          rows,
                          count / 16,
                          (columns - count) / 16,
                          0);
          __set_mask(rows);
          if (!first) {
            vmul(temporary,
                 previous_sum + state_offset,
                 factors,
                 1,
                 1,
                 1,
                 1,
                 8,
                 8,
                 8);
            pipe_barrier(PIPE_V);
            vadd(sums, sums, temporary, 1, 1, 1, 1, 8, 8, 8);
            pipe_barrier(PIPE_V);
            vadds(pending_scales + (round % 2) * 128 + index * 64 + row,
                  factors,
                  0.0f,
                  1,
                  1,
                  1,
                  8,
                  8);
          }
          vadds(previous_sum + state_offset, sums, 0.0f, 1, 1, 1, 8, 8);
          pipe_barrier(PIPE_V);
          if (last) {
            vadds(pending_sums + (round % 2) * 128 + index * 64 + row,
                  sums,
                  0.0f,
                  1,
                  1,
                  1,
                  8,
                  8);
          }
          set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
          __set_mask(64);
        }
      }
      set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
      wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
      // PV writes one FP32 partial result per section. Update O with vector
      // operations instead of multiplying by dense previous_output matrices.
      ffts_cross_core_sync(PIPE_MTE3, 1 | (2 << 4) | (AIV2AICFLAGID << 8));
    }
    if (round == 0) {
      continue;
    }
    wait_flag_dev(CUBE_PV_READY_FLAG);
    address.get_section_info(round - 1, section);
    if (!address.is_running(round - 1)) {
      section.sectionNum = 0;
    }
    for (int32_t index = 0; index < section.sectionNum; ++index) {
      const bool first = section.isHeadSection[index];
      const bool last = section.isTailSection[index];
      constexpr int32_t output_rows = 64;
      for (int32_t row = 0; row < section.processLineNum; row += output_rows) {
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
        auto out = gm_c_cube2 + (section.rowmaxOffset[index] + row) * 128;
        auto previous_output =
            accumulated_output + ((index % 2) * 64 + row) * 128;
        copy_gm_to_ubuf(scores,
                        diag_rowmax_gm + section.diagOffset[index] + row * 128,
                        0,
                        1,
                        output_rows * 128 / 8,
                        0,
                        0);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        if (!first) {
          set_vector_mask((uint64_t)-1, (uint64_t)-1);
          vbrcb((__ubuf__ uint32_t*)broadcast,
                (__ubuf__ uint32_t*)(pending_scales + ((round - 1) % 2) * 128 +
                                     index * 64 + row),
                1,
                8,
                output_rows / 8);
          pipe_barrier(PIPE_V);
          __set_mask(64);
          vmul(previous_output,
               previous_output,
               broadcast,
               output_rows,
               1,
               1,
               0,
               16,
               16,
               1);
          vmul(previous_output + 64,
               previous_output + 64,
               broadcast,
               output_rows,
               1,
               1,
               0,
               16,
               16,
               1);
          pipe_barrier(PIPE_V);
          vadd(previous_output,
               scores,
               previous_output,
               output_rows * 2,
               1,
               1,
               1,
               8,
               8,
               8);
          pipe_barrier(PIPE_V);
        }
        if (first) {
          vadds(previous_output, scores, 0.0f, output_rows * 2, 1, 1, 8, 8);
          pipe_barrier(PIPE_V);
        }
        if (last) {
          set_vector_mask((uint64_t)-1, (uint64_t)-1);
          vbrcb((__ubuf__ uint32_t*)broadcast,
                (__ubuf__ uint32_t*)(pending_sums + ((round - 1) % 2) * 128 +
                                     index * 64 + row),
                1,
                8,
                output_rows / 8);
          pipe_barrier(PIPE_V);
          __set_mask(64);
          vdiv(previous_output,
               previous_output,
               broadcast,
               output_rows,
               1,
               1,
               0,
               16,
               16,
               1);
          vdiv(previous_output + 64,
               previous_output + 64,
               broadcast,
               output_rows,
               1,
               1,
               0,
               16,
               16,
               1);
          pipe_barrier(PIPE_V);
          set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
          wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
          copy_ubuf_to_gm(
              out, previous_output, 0, 1, output_rows * 128 / 8, 0, 0);
          set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
          wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
        }
      }
    }
    ffts_cross_core_sync(PIPE_MTE3,
                         1 | (2 << 4) | (VECTOR_OUTPUT_READY_FLAG << 8));
  }
  wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
  set_vector_mask((uint64_t)-1, (uint64_t)-1);
}
template <typename INPUT_T, bool IF_BF16, typename WORKSPACE_T>
__aicore__ inline void VectorForward<INPUT_T, IF_BF16, WORKSPACE_T>::Run() {
  if constexpr (IF_BF16) {
    RunBfloat16();
  } else {
    set_atomic_none();
    set_mask_norm();
    set_vector_mask((uint64_t)-1, (uint64_t)-1);

    bool unirow_mode = false;  // true: 三角阵，一行统一计算 (还未支持末尾tri
                               // mask)；false：一行分两个section计算

    UB_FOR_NORMALIZE ub_norm;
    allocate_ubuf_for_norm(
        &ub_norm);  //  归一化在EXP计算完成之后，这里重新分配UB地址，不会影响rowmax计算
    UB_FOR_SHORT_LEN_ATTN_SCORE
    ub_short_seq_attn;                   // attention score （取代UB_FOR_EXP）
    int32_t tri_matrix_offset[2] = {0};  // 三角块的offset

    // 大循环
    for (int times_sync_cube = 0; times_sync_cube < address.get_total_round();
         times_sync_cube++) {
      wait_flag_dev(AIC2AIVFLAGID);
      // 开始计算
      int32_t sectionLoopTimes = qk_triangle == 0 ? 1 : 2;
      __ubuf__ float* fp32_test =
          reinterpret_cast<__ubuf__ float*>((uintptr_t)192 * 1024 - 128 * 4);
      int32_t rowmax_ub_offset = 0;
      int32_t record_rowmax_offset = 0;

      Address::FORWARD_SECTION_INFO section;
      address.get_section_info(times_sync_cube, section);
      // 最简单no_mask  example: section.sectionNum=2
      if (!address.is_running(times_sync_cube)) section.sectionNum = 0;
      // 1head  section_num : 2    16head:  section_num : 4
      for (int32_t sectionLoop = 0; sectionLoop < section.sectionNum;
           sectionLoop++) {
        int32_t sectionBlockNum = section.sectionBlockNums[sectionLoop];
        int32_t maxRow = MAX_LENG_PER_UB_PROC / sectionBlockNum / 128;
        if (maxRow >= 32) {
          rowPerTime = 32;
        } else if (maxRow >= 16) {
          rowPerTime = 16;
        } else if (maxRow >= 8) {
          rowPerTime = 8;
        } else if (maxRow >= 4) {
          rowPerTime = 4;
        } else if (maxRow >= 2) {
          rowPerTime = 2;
        } else {
          rowPerTime = 1;
        }

        while ((128 / (2 * y_cube_num_per_line) / 2 / rowPerTime) ==
               0) {  // 长序列
          rowPerTime /= 2;
        }
        if (section.sparseFlag && rowPerTime > 16) {
          rowPerTime = 16;
        }

        rowPerTime = 4;
        allocate_ubuf_for_short_seq_attn_score(&ub_short_seq_attn);

        diagExpMaxJMatUbAddr = ub_short_seq_attn.diagExpMaxJMatPingUbAddr;

        attention_score_short_double_line_one(
            sectionLoop,
            section.sectionNum,
            section.isTriangle,
            section.sectionBlockNums[sectionLoop],
            section.matrixMaskOffset,
            section.processLineNum,
            section.sectionBlockOffset[sectionLoop],
            section.isHeadSection[sectionLoop],
            section.isTailSection[sectionLoop],
            section.diagOffset[sectionLoop],
            attn_score_gm + section.attentionScoreOffset,
            attn_mask_gm,
            ub_short_seq_attn,
            section.sparseFlag);

        if (section.isTailSection[sectionLoop]) {
          // ~~~~~~~~~~~~~~~~ ln_rowsum ~~~~~~~~~~~~~~
          __ubuf__ float* ljVectorUbAddr =
              ub_short_seq_attn.ljVectorUbAddr_fp32;
          __ubuf__ float* buf_for_diag_fp32 =
              ub_short_seq_attn.buf_for_diag_fp32;
          __ubuf__ float* tempUb = ub_short_seq_attn.buf_for_record_rowmax_fp32;

          __set_mask(64);
          // tempUb
          vector_dup(tempUb,    // dst
                     (float)1,  // scalar
                     1,         // repeat  fp16 每次是128数
                     1,         // dstBlockStride
                     1,         // srcBlockStride
                     8,         // dstRepeatStride
                     1);        // srcRepeatStride
          pipe_barrier(PIPE_V);

          vdiv(buf_for_diag_fp32, tempUb, ljVectorUbAddr, 1, 1, 1, 1, 8, 8, 8);
          pipe_barrier(PIPE_V);
          set_vector_mask((uint64_t)-1, (uint64_t)-1);

          set_flag(PIPE_V, PIPE_S, EVENT_ID0);

          wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
          int32_t vector_id = get_subblockid();

          form_diagmat(diagExpMaxJMatUbAddr,
                       buf_for_diag_fp32,
                       diag_rowsum_gm + section.diagOffset[sectionLoop],
                       0,
                       64,
                       vector_id,
                       0,
                       1);
          set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
          wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
        }
      }
      uint64_t mode4 = 2;
      uint64_t config4 = 1 | (mode4 << 4) | (AIV2AICFLAGID << 8);
      ffts_cross_core_sync(PIPE_MTE3, config4);
    }
  }
}

template <typename INPUT_T, bool IF_BF16, typename WORKSPACE_T>
__aicore__ __inline__ void VectorForward<INPUT_T, IF_BF16, WORKSPACE_T>::
    allocate_ubuf_for_short_seq_attn_score(
        UB_FOR_SHORT_LEN_ATTN_SCORE* ub_attn_score) {
  int32_t offset = 512;
  int32_t section_num = 8;

  ub_attn_score->buf_for_cacl_rowmax_fp16 =
      reinterpret_cast<__ubuf__ INPUT_T*>((uintptr_t)offset);  //  32K
  offset += MAX_LENG_PER_UB_PROC * SIZE_FP16;                  // 8k*4

  // 2k *4;
  ub_attn_score->buf_for_vbrcb_rowmax_fp16 =
      reinterpret_cast<__ubuf__ INPUT_T*>((uintptr_t)offset);  //  1K
  offset += BASE_BLOCK_SIDE_LEN * SIZE_FP16 * rowPerTime * 2;

  ub_attn_score->buf_for_cacl_final_rowmax_fp16 =
      reinterpret_cast<__ubuf__ INPUT_T*>((uintptr_t)offset);
  offset += BASE_BLOCK_SIDE_LEN * 2 * rowPerTime * SIZE_FP16 * 2;

  // ub_attn_score -> buf_for_cacl_final_rowmax_fp32 = reinterpret_cast<__ubuf__
  // float *>((uintptr_t)offset);

  ub_attn_score->buf_for_record_rowmax_fp32 =
      reinterpret_cast<__ubuf__ float*>((uintptr_t)offset);
  offset += BASE_BLOCK_SIDE_LEN / 2 * ROUND_UP_8(rowPerTime) * SIZE_FP32 * 2;

  //  32K 无pingpong
  ub_attn_score->cur_buf_for_vbrcb_rowmax_fp32 =
      reinterpret_cast<__ubuf__ float*>((uintptr_t)offset);
  offset += BASE_BLOCK_SIDE_LEN * rowPerTime * SIZE_FP32;  // 精修

  offset = 512 + MAX_LENG_PER_UB_PROC * SIZE_FP32;
  //  fp16 attention score 32K
  ub_attn_score->buf_for_load_attn_score_fp16 =
      reinterpret_cast<__ubuf__ INPUT_T*>((uintptr_t)offset);
  offset += MAX_LENG_PER_UB_PROC * SIZE_FP16 * 2;

  ub_attn_score->buf_for_diag_fp32 =
      reinterpret_cast<__ubuf__ float*>((uintptr_t)offset);
  offset += BASE_BLOCK_SIDE_LEN / 2 * SIZE_FP32 * 2;

  //  1K
  ub_attn_score->buf_for_load_one_block_tri_mask_fp16 =
      reinterpret_cast<__ubuf__ INPUT_T*>((uintptr_t)offset);
  offset += BASE_BLOCK_SIDE_LEN * rowPerTime * 2 * SIZE_FP16;  // 精修

  // Mj
  ub_attn_score->maxJVectorUbAddr =
      reinterpret_cast<__ubuf__ INPUT_T*>((uintptr_t)offset);
  offset += BASE_BLOCK_SIDE_LEN / 2 * 4 * SIZE_FP16;  //  2K
  ub_attn_score->lastMaxJVectorUbAddr =
      reinterpret_cast<__ubuf__ INPUT_T*>((uintptr_t)offset);
  offset += BASE_BLOCK_SIDE_LEN / 2 * section_num * 4 * SIZE_FP16;
  ub_attn_score->subMaxValueResultVectorUbAddr =
      reinterpret_cast<__ubuf__ INPUT_T*>((uintptr_t)offset);
  offset += BASE_BLOCK_SIDE_LEN / 2 * 4 * SIZE_FP16;
  ub_attn_score->subMaxValueResultVectorUbAddr_fp32 =
      reinterpret_cast<__ubuf__ float*>((uintptr_t)offset);
  offset += BASE_BLOCK_SIDE_LEN / 2 * 4 * SIZE_FP32;
  ub_attn_score->ljVectorUbAddr =
      reinterpret_cast<__ubuf__ INPUT_T*>((uintptr_t)offset);
  offset += BASE_BLOCK_SIDE_LEN * SIZE_FP16;
  ub_attn_score->ljVectorUbAddr_fp32 =
      reinterpret_cast<__ubuf__ float*>((uintptr_t)offset);
  offset += BASE_BLOCK_SIDE_LEN * SIZE_FP32;
  ub_attn_score->lastLjVectorUbAddr =
      reinterpret_cast<__ubuf__ float*>((uintptr_t)offset);
  offset += BASE_BLOCK_SIDE_LEN / 2 * section_num * SIZE_FP16;
  ub_attn_score->diagExpMaxJMatPingUbAddr =
      reinterpret_cast<__ubuf__ float*>((uintptr_t)offset);  // 32k
  offset += BASE_BLOCK_SIDE_LEN / 2 * BASE_BLOCK_SIDE_LEN * SIZE_FP32;

  ub_attn_score->pp_buf_for_attn_score_fp16[1] =
      ub_attn_score->buf_for_load_attn_score_fp16 + MAX_LENG_PER_UB_PROC;
  ub_attn_score->pp_buf_for_attn_score_fp16[0] =
      ub_attn_score->buf_for_load_attn_score_fp16;

  ub_attn_score->pp_buf_for_load_one_block_tri_mask_fp16[1] =
      ub_attn_score->buf_for_load_one_block_tri_mask_fp16 +
      BASE_BLOCK_SIDE_LEN * rowPerTime;
  ub_attn_score->pp_buf_for_load_one_block_tri_mask_fp16[0] =
      ub_attn_score->buf_for_load_one_block_tri_mask_fp16;
}

template <typename INPUT_T, bool IF_BF16, typename WORKSPACE_T>
__aicore__ __inline__ void
VectorForward<INPUT_T, IF_BF16, WORKSPACE_T>::allocate_ubuf_for_norm(
    UB_FOR_NORMALIZE* ub_norm) {
  int32_t offset = 512;

  int32_t sizeof_buf_for_load_O_fp32 =
      MAX_LENG_PER_UB_PROC * 4 * 2;  // ~ 64k        // 8192个FP32 + ping-pong
  int32_t sizeof_buf_for_load_rowsum_fp32 =
      MAX_LENG_PER_UB_PROC / HEAD_DIM * 4 * 2;  // ~0.5K 128个O对应1个rowsum
  int32_t sizeof_buf_for_brcb_rowsum_fp32 =
      MAX_LENG_PER_UB_PROC / HEAD_DIM * 2 * 4 * (32 / 4);  // 展开成32字节对齐

  ub_norm->buf_for_load_O_fp32 =
      reinterpret_cast<__ubuf__ float*>((uintptr_t)offset);
  offset += sizeof_buf_for_load_O_fp32;

  ub_norm->buf_for_load_rowsum_fp32 =
      reinterpret_cast<__ubuf__ float*>((uintptr_t)offset);
  offset += sizeof_buf_for_load_rowsum_fp32;

  ub_norm->buf_for_brcb_rowsum_fp32 =
      reinterpret_cast<__ubuf__ float*>((uintptr_t)offset);
  offset += sizeof_buf_for_brcb_rowsum_fp32;

  ub_norm->o_ping_pong_interval = MAX_LENG_PER_UB_PROC;
  ub_norm->rowsum_ping_pong_interval = MAX_LENG_PER_UB_PROC / HEAD_DIM;
  ub_norm->rowsum_brcb_ping_pong_interval =
      ub_norm->rowsum_ping_pong_interval * (32 / 4);
}

/**
UB每次处理信号的长度
三角阵拼接后，即使按左右两个section计算，也按最长的预留空间
**/
template <typename INPUT_T, bool IF_BF16, typename WORKSPACE_T>
__aicore__ inline void
VectorForward<INPUT_T, IF_BF16, WORKSPACE_T>::get_sub_seq_length_per_proc(
    int32_t k_seq_len,
    int32_t block_num_per_full_line,
    int32_t* sub_seq_length_per_proc) {
  // 分两个section处理，不会总是2的幂；求max时用折半的方法，就不行
  *sub_seq_length_per_proc =
      k_seq_len > MAX_LENG_PER_UB_PROC * 1 ? MAX_LENG_PER_UB_PROC : k_seq_len;

  // 序列小于MAX_LENG_PER_UB_PROC, 需要减半以支持ping-pong
  if (*sub_seq_length_per_proc < MAX_LENG_PER_UB_PROC &&
      block_num_per_full_line > 1) {
    *sub_seq_length_per_proc = *sub_seq_length_per_proc / 2;  // (256)
  }
}

/***
非2的幂次长度，为了折半求vmax，需要进行padding
*/
template <typename INPUT_T, bool IF_BF16, typename WORKSPACE_T>
__aicore__ __inline__ void
VectorForward<INPUT_T, IF_BF16, WORKSPACE_T>::get_padding_info_for_row_max(
    int32_t total_block_num,  // 当前总共需要处理的block数量
    int32_t* padding_block_num) {
  auto tail_num = total_block_num %
                  BLOCK_NUM_FOR_VMAX;  // 满足最大长度倍数的部分不需要padding

  if (tail_num == 0) {
    *padding_block_num = 0;
    return;
  }

  int32_t total_block = 2;

  while (total_block < BLOCK_NUM_FOR_VMAX) {
    if (tail_num <= total_block) {
      break;
    }
    total_block *= 2;
  }

  *padding_block_num = total_block - tail_num;
}

template <typename INPUT_T, bool IF_BF16, typename WORKSPACE_T>
__aicore__ __inline__ void
VectorForward<INPUT_T, IF_BF16, WORKSPACE_T>::padding_for_row_max_or_rowsum(
    int32_t total_block_num,
    int32_t padding_block_num,
    int32_t ping_pong_flag,
    int32_t paddingType,
    __ubuf__ float* pp_buf_for_attn_score_fp16[]) {
  if (padding_block_num == 0) {
    return;
  }

  auto tail_num = total_block_num % MAX_BLOCK_PER_ONE_PROC;
  __ubuf__ float* cur_buf_for_attn_score =
      pp_buf_for_attn_score_fp16[ping_pong_flag];

  if (paddingType == 0)
    vector_dup(
        cur_buf_for_attn_score + tail_num * BASE_BLOCK_SIDE_LEN * rowPerTime,
        float(PADDING_FOR_MAX),
        padding_block_num * rowPerTime * 2,
        1,
        1,
        8,
        8);
  else
    vector_dup(
        cur_buf_for_attn_score + tail_num * BASE_BLOCK_SIDE_LEN * rowPerTime,
        float(0),
        padding_block_num * rowPerTime * 2,
        1,
        1,
        8,
        8);

  pipe_barrier(PIPE_V);
}

template <typename INPUT_T, bool IF_BF16, typename WORKSPACE_T>
__aicore__ __inline__ void
VectorForward<INPUT_T, IF_BF16, WORKSPACE_T>::padding_for_row_max_or_rowsum2(
    int32_t total_block_num,
    int32_t padding_block_num,
    int32_t ping_pong_flag,
    int32_t paddingType,
    __ubuf__ INPUT_T* pp_buf_for_attn_score_fp16[]) {
  if (padding_block_num == 0) {
    return;
  }

  auto tail_num = total_block_num % MAX_BLOCK_PER_ONE_PROC;
  __ubuf__ INPUT_T* cur_buf_for_attn_score =
      pp_buf_for_attn_score_fp16[ping_pong_flag];

  if (paddingType == 0)
    vector_dup(
        cur_buf_for_attn_score + tail_num * BASE_BLOCK_SIDE_LEN * rowPerTime,
        INPUT_T(PADDING_FOR_MAX2),
        padding_block_num * rowPerTime,
        1,
        1,
        8,
        8);
  else
    vector_dup(
        cur_buf_for_attn_score + tail_num * BASE_BLOCK_SIDE_LEN * rowPerTime,
        INPUT_T(0),
        padding_block_num * rowPerTime,
        1,
        1,
        8,
        8);

  pipe_barrier(PIPE_V);
}

template <typename INPUT_T, bool IF_BF16, typename WORKSPACE_T>
__aicore__ __inline__ void
VectorForward<INPUT_T, IF_BF16, WORKSPACE_T>::cacl_max(
    __ubuf__ INPUT_T* buf_for_cacl,
    int32_t _block_num) {
  // Tag：cur_block_num实际含义是次数，表面对折cur_block_num次后为2个block长度，即2
  // * 128
  auto cur_block_num = _block_num;

  while (cur_block_num > 1) {
    vmax(buf_for_cacl,
         buf_for_cacl,
         buf_for_cacl + BASE_BLOCK_SIDE_LEN * cur_block_num * rowPerTime,
         BASE_BLOCK_SIDE_LEN * cur_block_num * rowPerTime / BLOCK_NUM_8 /
             data_num_per_blk,
         1,
         1,
         1,
         8,
         8,
         8);  // ~~ fp32
    pipe_barrier(PIPE_V);

    cur_block_num = cur_block_num / 2;
  }

  vmax(buf_for_cacl,
       buf_for_cacl,
       buf_for_cacl + BASE_BLOCK_SIDE_LEN * rowPerTime,
       BASE_BLOCK_SIDE_LEN * 1 * rowPerTime / BLOCK_NUM_8 / data_num_per_blk,
       1,
       1,
       1,
       8,
       8,
       8);  // ~~ fp32
  pipe_barrier(PIPE_V);
}

// modi
/**
 * 64个基本块以下求max
 */
template <typename INPUT_T, bool IF_BF16, typename WORKSPACE_T>
__aicore__ __inline__ void
VectorForward<INPUT_T, IF_BF16, WORKSPACE_T>::process_cacl_max(
    int32_t basic_block_num,
    int32_t padding_block_num,
    bool pp_first_section,
    __ubuf__ INPUT_T* cur_buf_for_attn_score,
    __ubuf__ INPUT_T* cur_buf_for_rowmax,
    __ubuf__ INPUT_T* buf_for_cacl_final_rowmax_fp16) {
  // vmax:2k;  1k, 1k -->1k;  512, 512; ... ;64个数-->vcmax
  // vadd:                                            vcadd
  int32_t all_block_num = basic_block_num + padding_block_num;
  int32_t tail_block_num = all_block_num % 16;
  int32_t done_block_num = all_block_num / 16 * 16;
  bool from_buf_for_attn_score = false;

  if (all_block_num == 64)  // 写死就好
  {                         // 64 blk to 32 blk
    vmax(cur_buf_for_rowmax,
         cur_buf_for_attn_score,
         cur_buf_for_attn_score + BASE_BLOCK_SIDE_LEN * 32,
         32 * BASE_BLOCK_SIDE_LEN / BLOCK_NUM_8 / data_num_per_blk,  // repeat
         1,
         1,
         1,
         8,
         8,
         8);

    pipe_barrier(PIPE_V);
    cacl_max(cur_buf_for_rowmax, 16);
  } else if (all_block_num >= 48)  // 48(0)\50(2)\52(4)\56(8)
  {                                // first 32 blk to 16 blk
    vmax(cur_buf_for_rowmax,
         cur_buf_for_attn_score,
         cur_buf_for_attn_score + BASE_BLOCK_SIDE_LEN * 16,
         16 * BASE_BLOCK_SIDE_LEN / BLOCK_NUM_8 / data_num_per_blk,  // repeat
         1,
         1,
         1,
         8,
         8,
         8);  // ~~ fp32
    pipe_barrier(PIPE_V);

    // blks between 32 and 48 compare with the previous 16 blk
    vmax(cur_buf_for_rowmax,
         cur_buf_for_rowmax,
         cur_buf_for_attn_score + BASE_BLOCK_SIDE_LEN * 32,
         16 * BASE_BLOCK_SIDE_LEN / BLOCK_NUM_8 / data_num_per_blk,
         1,
         1,
         1,
         8,
         8,
         8);  // ~~ fp32
    pipe_barrier(PIPE_V);

    // comparing the last 8 blks with previous 16 blk
    vmax(cur_buf_for_rowmax,
         cur_buf_for_rowmax,
         cur_buf_for_rowmax + BASE_BLOCK_SIDE_LEN * 8,
         8 * BASE_BLOCK_SIDE_LEN / BLOCK_NUM_8 / data_num_per_blk,
         1,
         1,
         1,
         8,
         8,
         8);  // ~~ fp32
    pipe_barrier(PIPE_V);

    cacl_max(cur_buf_for_rowmax, 4);
  } else if (all_block_num >= 32)  // 32(0)\34(2)\36(4)\40(8)
  {
    // 32 blk to 16 blk
    vmax(cur_buf_for_rowmax,
         cur_buf_for_attn_score,
         cur_buf_for_attn_score + BASE_BLOCK_SIDE_LEN * 16 * rowPerTime,
         BASE_BLOCK_SIDE_LEN * 16 * rowPerTime / BLOCK_NUM_8 / data_num_per_blk,
         1,
         1,
         1,
         8,
         8,
         8);  // ~~ fp32
    pipe_barrier(PIPE_V);
    cacl_max(cur_buf_for_rowmax, 8);
  } else if (all_block_num >= 16)  // 16(0)\18(2)\20(4)\24(8)
  {
    // 16 blk to 8 blk
    vmax(cur_buf_for_rowmax,
         cur_buf_for_attn_score,
         cur_buf_for_attn_score + BASE_BLOCK_SIDE_LEN * 8 * rowPerTime,
         BASE_BLOCK_SIDE_LEN * 8 * rowPerTime / BLOCK_NUM_8 /
             data_num_per_blk,  // repeat
         1,
         1,
         1,
         8,
         8,
         8);  // ~~ fp32
    pipe_barrier(PIPE_V);
    cacl_max(cur_buf_for_rowmax, 4);
  }

  if (tail_block_num == 8) {
    if (all_block_num < 16) {
      vmax(
          cur_buf_for_rowmax,
          cur_buf_for_attn_score +
              BASE_BLOCK_SIDE_LEN * done_block_num * rowPerTime,
          cur_buf_for_attn_score +
              BASE_BLOCK_SIDE_LEN * (done_block_num + 4) * rowPerTime,
          BASE_BLOCK_SIDE_LEN * 4 * rowPerTime / BLOCK_NUM_8 / data_num_per_blk,
          1,
          1,
          1,
          8,
          8,
          8);  // ~~ fp32
      pipe_barrier(PIPE_V);

      vmax(
          cur_buf_for_rowmax,
          cur_buf_for_rowmax,
          cur_buf_for_rowmax + BASE_BLOCK_SIDE_LEN * 2 * rowPerTime,
          BASE_BLOCK_SIDE_LEN * 2 * rowPerTime / BLOCK_NUM_8 / data_num_per_blk,
          1,
          1,
          1,
          8,
          8,
          8);  // ~~ fp32
      pipe_barrier(PIPE_V);
    } else {
      vmax(
          cur_buf_for_rowmax + BASE_BLOCK_SIDE_LEN * rowPerTime,
          cur_buf_for_attn_score +
              BASE_BLOCK_SIDE_LEN * done_block_num * rowPerTime,
          cur_buf_for_attn_score +
              BASE_BLOCK_SIDE_LEN * (done_block_num + 4) * rowPerTime,
          BASE_BLOCK_SIDE_LEN * 4 * rowPerTime / BLOCK_NUM_8 / data_num_per_blk,
          1,
          1,
          1,
          8,
          8,
          8);  // ~~ fp32
      pipe_barrier(PIPE_V);

      vmax(
          cur_buf_for_rowmax + BASE_BLOCK_SIDE_LEN * rowPerTime,
          cur_buf_for_rowmax + BASE_BLOCK_SIDE_LEN * rowPerTime,
          cur_buf_for_rowmax + BASE_BLOCK_SIDE_LEN * 3 * rowPerTime,
          BASE_BLOCK_SIDE_LEN * 2 * rowPerTime / BLOCK_NUM_8 / data_num_per_blk,
          1,
          1,
          1,
          8,
          8,
          8);  // ~~ fp32
      pipe_barrier(PIPE_V);

      vmax(
          cur_buf_for_rowmax + BASE_BLOCK_SIDE_LEN * rowPerTime,
          cur_buf_for_rowmax + BASE_BLOCK_SIDE_LEN * rowPerTime,
          cur_buf_for_rowmax + BASE_BLOCK_SIDE_LEN * 2 * rowPerTime,
          BASE_BLOCK_SIDE_LEN * 1 * rowPerTime / BLOCK_NUM_8 / data_num_per_blk,
          1,
          1,
          1,
          8,
          8,
          8);  // ~~ fp32
      pipe_barrier(PIPE_V);
    }

    vmax(cur_buf_for_rowmax,
         cur_buf_for_rowmax,
         cur_buf_for_rowmax + BASE_BLOCK_SIDE_LEN * rowPerTime,
         BASE_BLOCK_SIDE_LEN * 1 * rowPerTime / BLOCK_NUM_8 / data_num_per_blk,
         1,
         1,
         1,
         8,
         8,
         8);  // ~~ fp32  剩下128个FP32, 2组
    pipe_barrier(PIPE_V);
  } else if (tail_block_num == 4) {
    if (all_block_num < 16) {
      vmax(
          cur_buf_for_rowmax,
          cur_buf_for_attn_score +
              BASE_BLOCK_SIDE_LEN * done_block_num * rowPerTime,
          cur_buf_for_attn_score +
              BASE_BLOCK_SIDE_LEN * (done_block_num + 2) * rowPerTime,
          BASE_BLOCK_SIDE_LEN * 2 * rowPerTime / BLOCK_NUM_8 / data_num_per_blk,
          1,
          1,
          1,
          8,
          8,
          8);  // ~~ fp32
      pipe_barrier(PIPE_V);
    } else {
      vmax(
          cur_buf_for_rowmax + BASE_BLOCK_SIDE_LEN * rowPerTime,
          cur_buf_for_attn_score +
              BASE_BLOCK_SIDE_LEN * done_block_num * rowPerTime,
          cur_buf_for_attn_score +
              BASE_BLOCK_SIDE_LEN * (done_block_num + 2) * rowPerTime,
          BASE_BLOCK_SIDE_LEN * 2 * rowPerTime / BLOCK_NUM_8 / data_num_per_blk,
          1,
          1,
          1,
          8,
          8,
          8);  // ~~ fp32
      pipe_barrier(PIPE_V);

      vmax(
          cur_buf_for_rowmax + BASE_BLOCK_SIDE_LEN * rowPerTime,
          cur_buf_for_rowmax + BASE_BLOCK_SIDE_LEN * rowPerTime,
          cur_buf_for_rowmax + BASE_BLOCK_SIDE_LEN * 2 * rowPerTime,
          BASE_BLOCK_SIDE_LEN * 1 * rowPerTime / BLOCK_NUM_8 / data_num_per_blk,
          1,
          1,
          1,
          8,
          8,
          8);  // ~~ fp32
      pipe_barrier(PIPE_V);
    }

    vmax(cur_buf_for_rowmax,
         cur_buf_for_rowmax,
         cur_buf_for_rowmax + BASE_BLOCK_SIDE_LEN * rowPerTime,
         BASE_BLOCK_SIDE_LEN * 1 * rowPerTime / BLOCK_NUM_8 / data_num_per_blk,
         1,
         1,
         1,
         8,
         8,
         8);  // ~~ fp32
    pipe_barrier(PIPE_V);
  } else if (tail_block_num == 2) {
    if (all_block_num < 16) {
      vmax(
          cur_buf_for_rowmax,
          cur_buf_for_attn_score +
              BASE_BLOCK_SIDE_LEN * done_block_num * rowPerTime,
          cur_buf_for_attn_score +
              BASE_BLOCK_SIDE_LEN * (done_block_num + 1) * rowPerTime,
          BASE_BLOCK_SIDE_LEN * 1 * rowPerTime / BLOCK_NUM_8 / data_num_per_blk,
          1,
          1,
          1,
          8,
          8,
          8);  // ~~ fp32
      pipe_barrier(PIPE_V);
    } else {
      vmax(
          cur_buf_for_rowmax + BASE_BLOCK_SIDE_LEN * rowPerTime,
          cur_buf_for_attn_score +
              BASE_BLOCK_SIDE_LEN * done_block_num * rowPerTime,
          cur_buf_for_attn_score +
              BASE_BLOCK_SIDE_LEN * (done_block_num + 1) * rowPerTime,
          BASE_BLOCK_SIDE_LEN * 1 * rowPerTime / BLOCK_NUM_8 / data_num_per_blk,
          1,
          1,
          1,
          8,
          8,
          8);  // ~~ fp32
      pipe_barrier(PIPE_V);

      vmax(
          cur_buf_for_rowmax,
          cur_buf_for_rowmax,
          cur_buf_for_rowmax + BASE_BLOCK_SIDE_LEN * rowPerTime,
          BASE_BLOCK_SIDE_LEN * 1 * rowPerTime / BLOCK_NUM_8 / data_num_per_blk,
          1,
          1,
          1,
          8,
          8,
          8);  // ~~ fp32
      pipe_barrier(PIPE_V);
    }

  }  // 没有其他分支了

  auto src_buf =
      from_buf_for_attn_score ? cur_buf_for_attn_score : cur_buf_for_rowmax;

  if (pp_first_section) {
    copy_ubuf_to_ubuf(buf_for_cacl_final_rowmax_fp16,
                      src_buf,
                      0,           // sid
                      rowPerTime,  // nBurst
                      128 / 16,    // lenBurst
                      0,           // srcStride
                      0);          // dstStride
    pipe_barrier(PIPE_V);
  }
}

template <typename INPUT_T, bool IF_BF16, typename WORKSPACE_T>
__aicore__ __inline__ void
VectorForward<INPUT_T, IF_BF16, WORKSPACE_T>::process_calc_sum(
    int32_t qk_triangle,
    PARAM_SHORT_SEQ_MAX param,
    int32_t ping_pong_flag,
    bool first_line,
    __gm__ WORKSPACE_T* attn_score_gm,
    __gm__ INPUT_T* attn_mask_gm,
    UB_FOR_SHORT_LEN_ATTN_SCORE ub_attn,
    bool sparse_flag,
    int32_t lines)
// 折半求
{
  // 8192段序列，一定会遇到末尾三角阵
  auto event_id = (ping_pong_flag == 0 ? EVENT_ID0 : EVENT_ID1);

  // 原始atten score;
  __ubuf__ INPUT_T* cur_buf_for_attn_score =
      ub_attn.pp_buf_for_attn_score_fp16[ping_pong_flag];

  // 这里可以复用，最后的结果存在 buf_for_cacl_short_(second)_final_rowmax_fp16
  __ubuf__ INPUT_T* cur_buf_for_rowmax = ub_attn.buf_for_cacl_rowmax_fp16;

  // 得到最后128个最大值 (section one & tow 会连着存放)
  __ubuf__ INPUT_T* cur_buf_for_final_rowmax =
      ub_attn.buf_for_cacl_final_rowmax_fp16;

  // 存储最大值，最终需要返回到GM上
  __ubuf__ INPUT_T* cur_buf_for_record_rowmax =
      ub_attn.buf_for_record_rowmax_fp16;  // 16ok; 8ok
  __ubuf__ INPUT_T* ljVectorUbAddr = ub_attn.ljVectorUbAddr;

  // 计算section one
  if (param.section_block_num > 0) {
    padding_for_row_max_or_rowsum2(param.section_block_num,
                                   param.section_padding_block_num,
                                   ping_pong_flag,
                                   PADDING_TYPE_ROWSUM,
                                   ub_attn.pp_buf_for_attn_score_fp16);
  }

  // 输出rowPertime*64
  process_cacl_wrap(param.section_block_num,
                    param.section_padding_block_num,
                    true,
                    cur_buf_for_attn_score,
                    cur_buf_for_rowmax,
                    cur_buf_for_final_rowmax,
                    1);  // 最后的结果在 cur_buf_for_final_rowmax
  // 保存UB位置offset, 跟lines相关；
  vcadd(ljVectorUbAddr + lines,
        cur_buf_for_final_rowmax,
        rowPerTime,
        1,
        1,
        8,
        false);
  pipe_barrier(PIPE_V);
}

/**
 * 8192长度以下的序列计算max
 */
template <typename INPUT_T, bool IF_BF16, typename WORKSPACE_T>
__aicore__ __inline__ void VectorForward<INPUT_T, IF_BF16, WORKSPACE_T>::
    process_line_phase_one_for_short_seq_max(
        bool is_head_section,
        bool is_tail_section,
        int32_t qk_triangle,
        PARAM_SHORT_SEQ_MAX param,
        int32_t ping_pong_flag,
        bool first_line,
        __gm__ WORKSPACE_T* attn_score_gm,
        __gm__ INPUT_T* attn_mask_gm,
        UB_FOR_SHORT_LEN_ATTN_SCORE ub_attn,
        bool sparse_flag,
        int32_t lines,
        __ubuf__ INPUT_T* maxJVectorUbAddr,
        __ubuf__ INPUT_T* lastMaxJVectorUbAddr,
        __ubuf__ INPUT_T* subMaxValueResultVectorUbAddr,
        int rowsPerDiag,
        int rowmax_build_step)
// 折半求
{
  event_t event_ids[] = {
      EVENT_ID1, EVENT_ID2, EVENT_ID3, EVENT_ID4, EVENT_ID5, EVENT_ID6};
  // 8192段序列，一定会遇到末尾三角阵
  auto event_id = (ping_pong_flag == 0 ? EVENT_ID0 : EVENT_ID1);

  // 原始atten score;
  __ubuf__ INPUT_T* cur_buf_for_attn_score_fp16 =
      ub_attn.pp_buf_for_attn_score_fp16[ping_pong_flag];

  // 这里可以复用，最后的结果存在 buf_for_cacl_short_(second)_final_rowmax_fp16
  __ubuf__ INPUT_T* cur_buf_for_rowmax = ub_attn.buf_for_cacl_rowmax_fp16;

  // 得到最后128个最大值 (section one & tow 会连着存放)
  __ubuf__ INPUT_T* cur_buf_for_final_rowmax =
      ub_attn.buf_for_cacl_final_rowmax_fp16;

  // 32B对齐的最大值
  __ubuf__ INPUT_T* cur_buf_for_vbrcb_rowmax_fp16 =
      ub_attn.buf_for_vbrcb_rowmax_fp16;

  // mask
  __ubuf__ INPUT_T* cur_buf_for_mask_fp32 =
      ub_attn.pp_buf_for_load_one_block_tri_mask_fp16[ping_pong_flag];
  // Tag:
  __ubuf__ INPUT_T* cur_buf_for_mask_fp16 =
      (ub_attn.pp_buf_for_load_one_block_tri_mask_fp16[ping_pong_flag] +
       64 * rowPerTime);

  __ubuf__ INPUT_T* cur_buf_for_head_mask =
      ub_attn.pp_buf_for_load_one_block_tri_mask_fp16[ping_pong_flag] +
      256 * rowPerTime;
  __ubuf__ INPUT_T* cur_buf_for_head_mask_fp16 =
      (__ubuf__ INPUT_T*)(cur_buf_for_head_mask + 64 * rowPerTime);

  // 存储最大值，最终需要返回到GM上
  __ubuf__ float* cur_buf_for_record_rowmax =
      ub_attn.buf_for_record_rowmax_fp32;  // 16ok; 8ok
  __ubuf__ float* cur_buf_for_vbrcb_rowmax_fp32 =
      ub_attn.cur_buf_for_vbrcb_rowmax_fp32;

  __ubuf__ float* cur_buf_for_score_fp32 =
      (__ubuf__ float*)ub_attn.buf_for_cacl_rowmax_fp16;
  // 读取Score;

  copy_gm_to_ubuf(cur_buf_for_attn_score_fp16,
                  attn_score_gm + param.section_start_line_offset,
                  0,                        // sid 一般0
                  param.section_block_num,  // 整行两个section全搬运进来
                  BASE_BLOCK_SIDE_LEN * rowPerTime / 16,  // burst_len 32B unit
                  (BASE_BLOCK_DATA_NUM - rowPerTime * BASE_BLOCK_SIDE_LEN) /
                      16,  // src stride 即burst stride   128*127/16
                  0);

  int32_t cur_core_index = get_block_idx();
  int32_t vector_id = get_subblockid();

  bool sparse_tail_mask_flag =
      sparse_flag && (param.apply_tri_mask == TRI_MATRIX_TAIL ||
                      param.apply_tri_mask == TRI_MATRIX_HEAD_AND_TAIL);
  bool sparse_head_mask_flag =
      sparse_flag && (param.apply_tri_mask == TRI_MATRIX_HEAD ||
                      param.apply_tri_mask == TRI_MATRIX_HEAD_AND_TAIL);

  // 一个tri matrix
  int32_t srcGap = rowPerTime == 1 ? 0 : maskSeqLength - BASE_BLOCK_SIDE_LEN;
  if ((qk_triangle == 1 && is_tail_section) || sparse_tail_mask_flag) {
    copy_gm_to_ubuf(cur_buf_for_mask_fp16,
                    attn_mask_gm + param.section_mask_offset,
                    0,
                    // sid 一般0
                    rowPerTime,
                    BASE_BLOCK_SIDE_LEN / 16,  // burst_len 32B unit
                    srcGap / 16,               // src stride 即burst stride
                    0);
  }
  if (sparse_head_mask_flag) {
    copy_gm_to_ubuf(
        cur_buf_for_head_mask_fp16,
        attn_mask_gm + param.section_mask_offset + 1,
        0,                         // sid 一般0
        rowPerTime,                // repeat times
        BASE_BLOCK_SIDE_LEN / 16,  // burst_len 32B unit  ~~ fp16: 128 * 2 / 32
        srcGap / 16,  // src stride 即burst stride // src gap // gm中连续存储，
        0);           // dst stride
  }
  set_flag(PIPE_MTE2, PIPE_V, event_id);
  wait_flag(PIPE_MTE2, PIPE_V, event_id);

  vmuls(cur_buf_for_attn_score_fp16,
        cur_buf_for_attn_score_fp16,
        (INPUT_T)SCALE,
        param.section_block_num * rowPerTime,
        1,
        1,
        8,
        8);
  pipe_barrier(PIPE_V);

  if ((is_tail_section) && (windowLen != 0)) {
    int align_size = 256;
    int tailLen = (align_size - windowLen) % BASE_BLOCK_SIDE_LEN;
    int pad_block_num = 1;
    half PADDING_VAL = -60000;
    if (windowLen >= 128) {
      vector_dup(
          cur_buf_for_attn_score_fp16 +
              rowPerTime * (param.section_block_num - 1) * BASE_BLOCK_SIDE_LEN,
          (INPUT_T)PADDING_VAL,
          rowPerTime,
          1,
          1,
          8,
          1);
      pipe_barrier(PIPE_V);
      pad_block_num = 2;
    }
    if (tailLen != 0) {
      __set_reverse_mask(tailLen);
      vector_dup(cur_buf_for_attn_score_fp16 +
                     rowPerTime * (param.section_block_num - pad_block_num) *
                         BASE_BLOCK_SIDE_LEN,
                 (INPUT_T)PADDING_VAL,
                 rowPerTime,
                 1,
                 1,
                 8,
                 1);
    }
    pipe_barrier(PIPE_V);
    set_vector_mask((uint64_t)-1, (uint64_t)-1);
  }

  // 计算section one
  if (param.section_block_num > 0) {
    padding_for_row_max_or_rowsum2(param.section_block_num,
                                   param.section_padding_block_num,
                                   ping_pong_flag,
                                   PADDING_TYPE_ROWMAX,
                                   ub_attn.pp_buf_for_attn_score_fp16);
    pipe_barrier(PIPE_V);
  }

  process_cacl_max(
      param.section_block_num,
      param.section_padding_block_num,
      true,
      cur_buf_for_attn_score_fp16,
      cur_buf_for_rowmax,
      cur_buf_for_final_rowmax);  // 最后的结果在 cur_buf_for_final_rowmax

  pipe_barrier(PIPE_V);

  // 4个max;
  vcmax(maxJVectorUbAddr,
        cur_buf_for_final_rowmax,
        rowPerTime,
        1,
        1,
        8,
        (Order_t)0b10);
  pipe_barrier(PIPE_V);

  if (!is_head_section) {
    // 更新rowmax：
    int BLOCK_SIZE_64 = 64;
    int BLOCK_SIZE_8 = 8;
    event_t event_ids[] = {
        EVENT_ID1, EVENT_ID2, EVENT_ID3, EVENT_ID4, EVENT_ID5, EVENT_ID6};

    __set_mask(rowPerTime);
    vmax(maxJVectorUbAddr,
         maxJVectorUbAddr,
         lastMaxJVectorUbAddr,
         1,  // repeat
         1,
         1,
         1,
         BLOCK_SIZE_64 / BLOCK_SIZE_8,   // dstRepeatStride
         BLOCK_SIZE_64 / BLOCK_SIZE_8,   // src0RepeatStride
         BLOCK_SIZE_64 / BLOCK_SIZE_8);  // src1RepeatStride)
    pipe_barrier(PIPE_V);

    vsub(subMaxValueResultVectorUbAddr,  // dm =  m_j-1 - m_j
         lastMaxJVectorUbAddr,           // mj-1
         maxJVectorUbAddr,               // mj
         1,                              // repeat
         1,                              //
         1,
         1,
         8,
         8,
         8);
    pipe_barrier(PIPE_V);

    vexp(subMaxValueResultVectorUbAddr,  // e^(m_j-1 - m_j)
         subMaxValueResultVectorUbAddr,
         1,
         1,
         1,
         8,
         8);
    pipe_barrier(PIPE_V);

    vadds(subMaxValueResultVectorUbAddr,
          subMaxValueResultVectorUbAddr,
          FP16_SMALL_NUM,
          1,
          1,
          1,
          8,
          8);
    pipe_barrier(PIPE_V);

    set_vector_mask((uint64_t)-1, (uint64_t)-1);
    set_flag(PIPE_V, PIPE_S, event_ids[rowmax_build_step]);
  }

  vconv_f162f32(cur_buf_for_record_rowmax, maxJVectorUbAddr, 1, 1, 1, 8, 4);
  pipe_barrier(PIPE_V);

  int32_t vbrcbRepeatTimes = (rowPerTime + 7) / 8;
  __set_mask(rowPerTime);
  // Tag：注意vector指令的src 地址都需要32位对齐！      不然 会报错
  vbrcb((__ubuf__ uint32_t*)cur_buf_for_vbrcb_rowmax_fp32,
        (__ubuf__ uint32_t*)cur_buf_for_record_rowmax,
        2,
        8,
        vbrcbRepeatTimes);
  pipe_barrier(PIPE_V);

  vbrcb((__ubuf__ uint32_t*)cur_buf_for_vbrcb_rowmax_fp32 + 8,
        (__ubuf__ uint32_t*)cur_buf_for_record_rowmax,
        2,
        8,
        vbrcbRepeatTimes);
  pipe_barrier(PIPE_V);

  set_vector_mask((uint64_t)-1, (uint64_t)-1);

  vconv_f322f16(cur_buf_for_vbrcb_rowmax_fp16,
                cur_buf_for_vbrcb_rowmax_fp32,
                1,
                1,
                1,
                4,
                8);
  pipe_barrier(PIPE_V);
}

template <typename INPUT_T, bool IF_BF16, typename WORKSPACE_T>
__aicore__ __inline__ void VectorForward<INPUT_T, IF_BF16, WORKSPACE_T>::
    process_line_phase_one_for_short_seq_exp_and_rowsum(
        int32_t qk_triangle,
        PARAM_SHORT_SEQ_MAX param,
        int32_t ping_pong_flag,
        __gm__ WORKSPACE_T* attn_score_gm,
        __gm__ INPUT_T* attn_mask_gm,
        UB_FOR_SHORT_LEN_ATTN_SCORE ub_attn,
        int32_t offset,
        int32_t lines,
        bool sparse_flag) {
  auto event_id = (ping_pong_flag == 0 ? EVENT_ID0 : EVENT_ID1);

  __ubuf__ INPUT_T* cur_buf_for_attn_score_fp16 =
      ub_attn.pp_buf_for_attn_score_fp16[ping_pong_flag];

  __ubuf__ INPUT_T* cur_buf_for_vbrcb_rowmax_fp16 =
      ub_attn.buf_for_vbrcb_rowmax_fp16;

  // 减去最大值
  //    S-max
  if (rowPerTime < 16) {
    if (rowPerTime == 1) {
      vsub(cur_buf_for_attn_score_fp16,
           cur_buf_for_attn_score_fp16,
           cur_buf_for_vbrcb_rowmax_fp16,
           param.section_block_num * 2,
           1,
           1,
           0,
           8,
           8,
           0);
      pipe_barrier(PIPE_V);
    } else {
      for (int32_t i = 0; i < rowPerTime;
           i++) {  // 0-128: 0-64 65-128---------  0-7       129-256
        vsub(cur_buf_for_attn_score_fp16 + i * 128,
             cur_buf_for_attn_score_fp16 + i * 128,
             cur_buf_for_vbrcb_rowmax_fp16 + i * 16,
             param.section_block_num,
             1,
             1,
             0,
             rowPerTime * 8,
             rowPerTime * 8,
             0);
        pipe_barrier(PIPE_V);
      }
    }
  } else {
    for (int32_t i = 0; i < param.section_block_num; i++) {
      vsub(cur_buf_for_attn_score_fp16 + i * 128 * rowPerTime,
           cur_buf_for_attn_score_fp16 + i * 128 * rowPerTime,
           cur_buf_for_vbrcb_rowmax_fp16,
           rowPerTime,
           1,
           1,
           0,
           16,
           16,
           1);
      pipe_barrier(PIPE_V);
      vsub(cur_buf_for_attn_score_fp16 + i * 128 * rowPerTime + 64,
           cur_buf_for_attn_score_fp16 + i * 128 * rowPerTime + 64,
           cur_buf_for_vbrcb_rowmax_fp16,
           rowPerTime,
           1,
           1,
           0,
           16,
           16,
           1);
      pipe_barrier(PIPE_V);
    }
  }
  //  128/64=2
  vexp(cur_buf_for_attn_score_fp16,
       cur_buf_for_attn_score_fp16,
       param.section_block_num * rowPerTime * BASE_BLOCK_SIDE_LEN /
           BLOCK_NUM_8 / data_num_per_blk,  // 5 * 128 * 8   / 64    5 * 2 * 8
       1,
       1,
       8,
       8);
  pipe_barrier(PIPE_V);

  set_flag(PIPE_V, PIPE_MTE3, event_id);
  wait_flag(PIPE_V, PIPE_MTE3, event_id);

  //  一开始total_offset_fp32=0;
  int32_t total_offset_fp16 = param.section_start_line_offset;
  copy_ubuf_to_gm(
      ((__gm__ half*)(attn_score_gm + total_offset_fp16)),  // 高精度
      (__ubuf__ half*)cur_buf_for_attn_score_fp16,
      0,
      param.section_block_num,
      BASE_BLOCK_SIDE_LEN * rowPerTime / 16,
      0,
      ((BASE_BLOCK_DATA_NUM - rowPerTime * BASE_BLOCK_SIDE_LEN) + 0) / 16);
  process_calc_sum(qk_triangle,
                   param,
                   ping_pong_flag,
                   lines == 0,
                   attn_score_gm,
                   attn_mask_gm,
                   ub_attn,
                   sparse_flag,
                   lines);
}
template <typename INPUT_T, bool IF_BF16, typename WORKSPACE_T>
__aicore__ __inline__ void
VectorForward<INPUT_T, IF_BF16, WORKSPACE_T>::add_max_wrap(
    __ubuf__ INPUT_T* dest_addr,
    __ubuf__ INPUT_T* src_addr,
    __ubuf__ INPUT_T* src_addr2,
    int32_t rpt,
    int32_t dst_stride,
    int32_t src_stride,
    int32_t src_stride2,
    int32_t dst_rpt_stride,
    int32_t src_rpt_stride,
    int32_t src_rpt_stride2,
    int32_t optcode) {
  if (optcode == 2) {
    vmax(dest_addr,
         src_addr,
         src_addr2,
         rpt,
         dst_stride,
         src_stride,
         src_stride2,
         dst_rpt_stride,
         src_rpt_stride,
         src_rpt_stride2);
  } else if (optcode == 1) {
    vadd(dest_addr,
         src_addr,
         src_addr2,
         rpt,
         dst_stride,
         src_stride,
         src_stride2,
         dst_rpt_stride,
         src_rpt_stride,
         src_rpt_stride2);
  }
}
template <typename INPUT_T, bool IF_BF16, typename WORKSPACE_T>
__aicore__ __inline__ void
VectorForward<INPUT_T, IF_BF16, WORKSPACE_T>::cacl_wrap(
    __ubuf__ INPUT_T* buf_for_cacl,
    int32_t _block_num,
    int32_t optcode) {
  auto cur_block_num = _block_num;
  auto repeat_factor = BASE_BLOCK_SIDE_LEN / (256 / SIZE_FP16);
  while (cur_block_num > 1) {
    add_max_wrap(
        buf_for_cacl,
        buf_for_cacl,
        buf_for_cacl + BASE_BLOCK_SIDE_LEN * cur_block_num * rowPerTime,
        cur_block_num * repeat_factor * rowPerTime,
        1,
        1,
        1,
        8,
        8,
        8,
        optcode);  // ~~ fp32
    pipe_barrier(PIPE_V);

    cur_block_num = cur_block_num / 2;
  }

  add_max_wrap(buf_for_cacl,
               buf_for_cacl,
               buf_for_cacl + BASE_BLOCK_SIDE_LEN * rowPerTime,
               repeat_factor * rowPerTime,
               1,
               1,
               1,
               8,
               8,
               8,
               optcode);  // ~~ fp32
  pipe_barrier(PIPE_V);
}
template <typename INPUT_T, bool IF_BF16, typename WORKSPACE_T>
__aicore__ __inline__ void
VectorForward<INPUT_T, IF_BF16, WORKSPACE_T>::process_cacl_wrap(
    int32_t basic_block_num,
    int32_t padding_block_num,
    bool pp_first_section,
    __ubuf__ INPUT_T* cur_buf_for_attn_score,
    __ubuf__ INPUT_T* cur_buf_for_rowmax,
    __ubuf__ INPUT_T* buf_for_cacl_final_rowmax_fp16,
    int32_t optcode) {
  int32_t all_block_num = basic_block_num + padding_block_num;
  int32_t tail_block_num = all_block_num % 16;
  int32_t done_block_num = all_block_num / 16 * 16;
  bool from_buf_for_attn_score = false;
  int32_t threshold = 64;
  int32_t sumrep = 0;
  int32_t rowfactor = all_block_num < 48 ? rowPerTime : 1;
  if (all_block_num >= 16) {
    while (threshold > all_block_num) {
      threshold /= 2;
    }
    threshold /= 2;
    sumrep = BASE_BLOCK_SIDE_LEN * threshold / (256 / SIZE_FP16);
    add_max_wrap(
        cur_buf_for_rowmax,
        cur_buf_for_attn_score,
        cur_buf_for_attn_score + BASE_BLOCK_SIDE_LEN * threshold * rowfactor,
        sumrep * rowfactor,
        1,
        1,
        1,
        8,
        8,
        8,
        optcode);
    pipe_barrier(PIPE_V);
    if (all_block_num >= 48 && all_block_num < 64) {
      add_max_wrap(cur_buf_for_rowmax,
                   cur_buf_for_rowmax,
                   cur_buf_for_attn_score + BASE_BLOCK_SIDE_LEN * 32,
                   sumrep,
                   1,
                   1,
                   1,
                   8,
                   8,
                   8,
                   optcode);
      pipe_barrier(PIPE_V);
      threshold = 8;
      add_max_wrap(cur_buf_for_rowmax,
                   cur_buf_for_rowmax,
                   cur_buf_for_rowmax + BASE_BLOCK_SIDE_LEN * threshold,
                   16,
                   1,
                   1,
                   1,
                   8,
                   8,
                   8,
                   optcode);
      pipe_barrier(PIPE_V);
    }
    cacl_wrap(cur_buf_for_rowmax, threshold / 2, optcode);
  }

  rowfactor = rowPerTime;
  if (tail_block_num > 0) {
    int32_t tail_threshold = 8;
    while (tail_threshold >= 2) {
      if (tail_threshold == tail_block_num) {
        break;
      }
      tail_threshold /= 2;
    }
    if (tail_threshold == 1) {
      return;
    }
    tail_threshold /= 2;
    sumrep = BASE_BLOCK_SIDE_LEN * tail_threshold / (256 / SIZE_FP16);
    auto bufstart = cur_buf_for_rowmax + BASE_BLOCK_SIDE_LEN *
                                             (int32_t)(all_block_num >= 16) *
                                             rowfactor;
    auto tailstart = cur_buf_for_attn_score +
                     BASE_BLOCK_SIDE_LEN * done_block_num * rowfactor;
    add_max_wrap(bufstart,
                 tailstart,
                 tailstart + BASE_BLOCK_SIDE_LEN * tail_threshold * rowfactor,
                 sumrep * rowfactor,
                 1,
                 1,
                 1,
                 8,
                 8,
                 8,
                 optcode);
    pipe_barrier(PIPE_V);
    tail_threshold /= 2;
    if (tail_threshold >= 1) {
      cacl_wrap(bufstart, tail_threshold, optcode);
    }
    pipe_barrier(PIPE_V);
    if (all_block_num >= 16) {
      add_max_wrap(cur_buf_for_rowmax,
                   cur_buf_for_rowmax,
                   bufstart,
                   2 * rowfactor,
                   1,
                   1,
                   1,
                   8,
                   8,
                   8,
                   optcode);
      pipe_barrier(PIPE_V);
    }
  }

  auto src_buf =
      from_buf_for_attn_score ? cur_buf_for_attn_score : cur_buf_for_rowmax;

  copy_ubuf_to_ubuf(buf_for_cacl_final_rowmax_fp16,
                    src_buf,
                    0,          // sid
                    rowfactor,  // nBurst
                    128 / 16,   // lenBurst
                    0,          // srcStride
                    0);         // dstStride
  pipe_barrier(PIPE_V);
}
template <typename INPUT_T, bool IF_BF16, typename WORKSPACE_T>
__aicore__ __inline__ void
VectorForward<INPUT_T, IF_BF16, WORKSPACE_T>::form_diagmat(
    __ubuf__ float* diagmataddr,
    __ubuf__ float* vecaddr,
    __gm__ float* gm_addr,
    int step,
    int rowsPerDiag,
    int32_t vector_id,
    int32_t ping_pong_flag,
    int type) {
  event_t event_ids[] = {
      EVENT_ID1, EVENT_ID2, EVENT_ID3, EVENT_ID4, EVENT_ID5, EVENT_ID6};
  int stepshift = rowsPerDiag * BASE_BLOCK_LENGTH * (step);  // 16*128 32*128

  int init =
      step * rowsPerDiag + vector_id * BASE_BLOCK_LENGTH / 2;  // 对角线上偏移

  // rowmax
  if (type == 0) {
    // rowmax diag
    wait_flag(PIPE_MTE3, PIPE_S, event_ids[step]);
    for (int line = 0; line < rowsPerDiag; line++) {
      int lineshift = init + (BASE_BLOCK_LENGTH)*line;
      (diagmataddr + stepshift)[lineshift + line] = vecaddr[line];
    }
  } else {
    // rowsum diag
    for (int line = 0; line < rowsPerDiag; line++) {
      int lineshift = init + (BASE_BLOCK_LENGTH)*line;
      (diagmataddr + stepshift)[lineshift + line] = vecaddr[line];
    }
  }

  set_flag(PIPE_S, PIPE_MTE3, event_ids[step]);
  wait_flag(PIPE_S, PIPE_MTE3, event_ids[step]);

  copy_ubuf_to_gm(gm_addr + stepshift,
                  diagmataddr + stepshift,
                  0,
                  1,
                  BASE_BLOCK_LENGTH * rowsPerDiag * SIZE_FP32 / 32,
                  0,
                  0);
  if (type == 0) {
    set_flag(PIPE_MTE3, PIPE_S, event_ids[step + 1]);
  }
}

template <typename INPUT_T, bool IF_BF16, typename WORKSPACE_T>
__aicore__ __inline__ void
VectorForward<INPUT_T, IF_BF16, WORKSPACE_T>::updateRowsum(
    __ubuf__ float* ljVectorUbAddr,
    __ubuf__ float* lastLjVectorUbAddr,
    __ubuf__ float* subMaxValueResultVectorUbAddr,
    __ubuf__ float* tempUbAddr) {
  vmul((__ubuf__ float*)tempUbAddr,  // l_j
       subMaxValueResultVectorUbAddr,
       lastLjVectorUbAddr,  // l_j-1
       1,                   // repeat
       1,
       1,
       1,
       8,
       8,
       8);
  pipe_barrier(PIPE_V);

  vadd(ljVectorUbAddr,
       ljVectorUbAddr,  // row_sum
       tempUbAddr,
       1,  // repeat
       1,
       1,
       1,
       8,
       8,
       8);
  pipe_barrier(PIPE_V);
}
template <typename INPUT_T, bool IF_BF16, typename WORKSPACE_T>
__aicore__ __inline__ void
VectorForward<INPUT_T, IF_BF16, WORKSPACE_T>::backupMaxandSum(
    __ubuf__ float* ljVectorUbAddr,
    __ubuf__ float* lastLjVectorUbAddr,
    __ubuf__ INPUT_T* maxJVectorUbAddr,
    __ubuf__ INPUT_T* lastMaxJVectorUbAddr) {
  copy_ubuf_to_ubuf(lastLjVectorUbAddr,
                    ljVectorUbAddr,
                    0,       // sid
                    1,       // nBurst
                    64 / 8,  // lenBurst
                    0,       // srcStride
                    0);      // dstStride
  pipe_barrier(PIPE_V);

  copy_ubuf_to_ubuf(lastMaxJVectorUbAddr,
                    maxJVectorUbAddr,
                    0,         // sid
                    1,         // nBurst
                    256 / 16,  // lenBurst //8nums;
                    0,         // srcStride
                    0);        // dstStride
  pipe_barrier(PIPE_V);
}
/**
 * 短序列一次拷贝
 */
template <typename INPUT_T, bool IF_BF16, typename WORKSPACE_T>
__aicore__ __inline__ void VectorForward<INPUT_T, IF_BF16, WORKSPACE_T>::
    attention_score_short_double_line_one(
        int32_t sectionLoop,
        int32_t sectionNum,
        int32_t qk_triangle,  // 是否倒三角
        int32_t section_block_nums,
        int32_t tri_matrix_mask_offset,  // 128*128的三角阵中取第n行
        int32_t each_vector_proc_line_num,
        int32_t local_section_start_line_offset,
        bool is_head_section,
        bool is_tail_section,
        int32_t diag_offset,
        __gm__ WORKSPACE_T* __restrict__ attn_score_gm,
        __gm__ INPUT_T* __restrict__ attn_mask_gm,
        UB_FOR_SHORT_LEN_ATTN_SCORE ub_attn,
        bool sparse_flag) {
  event_t event_ids[] = {
      EVENT_ID1, EVENT_ID2, EVENT_ID3, EVENT_ID4, EVENT_ID5, EVENT_ID6};

  // 只能支持8192以下的序列，否则没法一次拷贝一个完整行
  PARAM_SHORT_SEQ_MAX param_ping_pong[2] = {0};
  param_ping_pong[0].section_start_line_offset =
      local_section_start_line_offset;
  param_ping_pong[0].section_block_num = section_block_nums;
  param_ping_pong[0].record_rowmax_offset = 0;

  param_ping_pong[1].section_start_line_offset =
      local_section_start_line_offset +
      BASE_BLOCK_SIDE_LEN * rowPerTime;  // pong和ping相差rowPerTime行
  param_ping_pong[1].section_block_num = section_block_nums;
  param_ping_pong[1].record_rowmax_offset = 0;

  get_padding_info_for_row_max(param_ping_pong[0].section_block_num,
                               &param_ping_pong[0].section_padding_block_num);
  param_ping_pong[1].section_padding_block_num =
      param_ping_pong[0].section_padding_block_num;
  param_ping_pong[0].section_mask_offset = tri_matrix_mask_offset;
  param_ping_pong[1].section_mask_offset =
      tri_matrix_mask_offset + maskSeqLength * rowPerTime;
  if (qk_triangle == 1) {
    param_ping_pong[0].section_mask_offset = tri_matrix_mask_offset;
    param_ping_pong[1].section_mask_offset =
        tri_matrix_mask_offset + maskSeqLength * rowPerTime;
  }

  int32_t ping_pong_flag = 0;

  set_flag(PIPE_MTE3,
           PIPE_MTE2,
           EVENT_ID0);  // 是不是应该放到lines循环上面？ -- 可以最后再看看
  set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
  //  Tag：这里都是同个section;
  int rowsPerDiag = 16;
  int rowmax_build_step = 0;  // 构造对角阵的迭代次数
  int rowsum_build_step = 0;  // 构造对角阵的迭代次数
  set_flag(PIPE_MTE3, PIPE_S, event_ids[rowmax_build_step]);

  // <64
  int32_t vector_id = get_subblockid();

  __ubuf__ float* lastLjVectorUbAddr =
      ub_attn.lastLjVectorUbAddr + (sectionLoop % 2) * 64;
  __ubuf__ INPUT_T* ljVectorUbAddr = ub_attn.ljVectorUbAddr;
  __ubuf__ float* ljVectorUbAddr_fp32 = ub_attn.ljVectorUbAddr_fp32;
  __ubuf__ INPUT_T* subMaxValueResultVectorUbAddr =
      ub_attn.subMaxValueResultVectorUbAddr;
  __ubuf__ float* tempUbAddr = ub_attn.buf_for_record_rowmax_fp32;
  __ubuf__ INPUT_T* maxJVectorUbAddr = ub_attn.maxJVectorUbAddr;
  __ubuf__ INPUT_T* lastMaxJVectorUbAddr =
      ub_attn.lastMaxJVectorUbAddr + (sectionLoop % 2) * 128 * 2;
  __ubuf__ float* subMaxValueResultVectorUbAddr_fp32 =
      ub_attn.subMaxValueResultVectorUbAddr_fp32;

  // UB清零 每次256/4=64数；
  vector_dup(maxJVectorUbAddr, 0, 2, 1, 1, 8, 1);
  pipe_barrier(PIPE_V);
  vector_dup(ljVectorUbAddr, 0, 1, 1, 1, 8, 1);
  pipe_barrier(PIPE_V);
  vector_dup(diagExpMaxJMatUbAddr,
             0,
             BASE_BLOCK_SIDE_LEN / 2 * BASE_BLOCK_SIDE_LEN / 64,
             1,
             1,
             8,
             1);
  pipe_barrier(PIPE_V);

  for (int32_t lines = 0; lines < each_vector_proc_line_num;
       lines += rowPerTime) {
    auto event_id = (ping_pong_flag == 0 ? EVENT_ID0 : EVENT_ID1);
    param_ping_pong[0].apply_tri_mask = TRI_MATRIX_NONE;  // TRI_MATRIX_NONE = 0
    param_ping_pong[1].apply_tri_mask = TRI_MATRIX_NONE;
    if (qk_triangle)  // 非三角阵不用
    {
      param_ping_pong[ping_pong_flag].apply_tri_mask = TRI_MATRIX_TAIL;
    }
    if (sparse_flag) {
      param_ping_pong[ping_pong_flag].apply_tri_mask = TRI_MATRIX_HEAD_AND_TAIL;
    }
    // 输出4个max； 输入4*2k;  maxJVectorUbAddr(64行；) 地址偏移 去保存；
    // 抽取: process_score()
    // step1: xzj ok

    int linesRound = lines / rowPerTime * 16;
    wait_flag(PIPE_MTE3, PIPE_MTE2, event_id);
    process_line_phase_one_for_short_seq_max(
        is_head_section,
        is_tail_section,
        qk_triangle,
        param_ping_pong[ping_pong_flag],
        ping_pong_flag,
        lines == 0,
        attn_score_gm,
        attn_mask_gm,
        ub_attn,
        sparse_flag,
        lines,
        maxJVectorUbAddr + linesRound,  // 后期维护偏移
        lastMaxJVectorUbAddr + linesRound,
        subMaxValueResultVectorUbAddr + linesRound,
        rowsPerDiag,
        rowmax_build_step);

    int32_t offset = lines * 128;
    // //Step3:算P值and Rowsum
    process_line_phase_one_for_short_seq_exp_and_rowsum(
        qk_triangle,
        param_ping_pong[ping_pong_flag],
        ping_pong_flag,
        attn_score_gm,
        attn_mask_gm,
        ub_attn,
        offset,
        lines,
        sparse_flag);

    set_flag(PIPE_MTE3, PIPE_MTE2, event_id);

    if (!is_head_section) {
      wait_flag(PIPE_V, PIPE_S, event_ids[rowmax_build_step]);

      for (int i = 0; i < rowPerTime; i++) {
        (subMaxValueResultVectorUbAddr + lines)[i] =
            (subMaxValueResultVectorUbAddr + linesRound)[i];
      }
      set_flag(PIPE_S, PIPE_V, event_ids[rowmax_build_step]);
      wait_flag(PIPE_S, PIPE_V, event_ids[rowmax_build_step]);
    }

    // Step2,&& 构造max对角阵;
    if ((lines + rowPerTime) % rowsPerDiag == 0 && !is_head_section) {
      // Tag:exp(mj-mj_1); Tag：第一次迭代没有max对角阵,因为没有mj-1
      int offset = ((lines + rowPerTime) - rowsPerDiag);

      __set_mask(rowsPerDiag);
      vconv_f162f32(subMaxValueResultVectorUbAddr_fp32 + offset,
                    subMaxValueResultVectorUbAddr + offset,
                    1,
                    1,
                    1,
                    8,
                    4);
      set_vector_mask((uint64_t)-1, (uint64_t)-1);

      set_flag(PIPE_V, PIPE_S, event_ids[rowmax_build_step]);
      wait_flag(PIPE_V, PIPE_S, event_ids[rowmax_build_step]);

      form_diagmat(diagExpMaxJMatUbAddr,
                   subMaxValueResultVectorUbAddr_fp32 + offset,
                   diag_rowmax_gm + diag_offset,
                   rowmax_build_step,
                   rowsPerDiag,
                   vector_id,
                   ping_pong_flag,
                   0);
      rowmax_build_step += 1;
    }

    param_ping_pong[ping_pong_flag].section_start_line_offset +=
        BASE_BLOCK_SIZE_DOUBLE * rowPerTime;
    param_ping_pong[ping_pong_flag].section_mask_offset +=
        maskSeqLength * 2 * rowPerTime;
    ping_pong_flag = 1 - ping_pong_flag;
  }
  wait_flag(PIPE_MTE3, PIPE_S, event_ids[rowmax_build_step]);
  __set_mask(64);
  vconv_f162f32(ljVectorUbAddr_fp32, ljVectorUbAddr, 1, 1, 1, 8, 4);
  set_vector_mask((uint64_t)-1, (uint64_t)-1);
  pipe_barrier(PIPE_V);

  if (!is_head_section) {
    updateRowsum(ljVectorUbAddr_fp32,
                 lastLjVectorUbAddr,
                 subMaxValueResultVectorUbAddr_fp32,
                 tempUbAddr);
  }

  // Step6:备份(此时是完整64数；max, sum)

  if (!is_tail_section) {
    backupMaxandSum(ljVectorUbAddr_fp32,
                    lastLjVectorUbAddr,
                    maxJVectorUbAddr,
                    lastMaxJVectorUbAddr);
  }

  wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
  wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
}

template <typename INPUT_T, bool IF_BF16, typename WORKSPACE_T>
__aicore__ __inline__ void
VectorForward<INPUT_T, IF_BF16, WORKSPACE_T>::get_uni_rowmax_seq_info_per_proc(
    int32_t block_num_per_full_line,
    int32_t sub_seq_length_per_proc,  // 一次处理处理的长度
    int32_t* ping_block_offset_num,   // ping起始块
    int32_t* pong_block_offset_num,   // pong起始块
    int32_t* tail_block_offset_num,   // tail起始块
    int32_t* tail_block_num,          // tail的块数
    int32_t* ping_pong_times          // pingpang的循环次数
) {
  *tail_block_num =
      block_num_per_full_line % 2;  // 完整行块的剩余尾块：pingpong需要偶数
  *tail_block_offset_num =
      block_num_per_full_line -
      *tail_block_num;  // (4)    --  如果y_num_cube_per_line = 3

  *ping_block_offset_num = 0;                             // (0)
  *pong_block_offset_num = (*tail_block_offset_num) / 2;  // 一定可以整除 (2)

  auto _total_size = *pong_block_offset_num * BASE_BLOCK_SIDE_LEN;

  *ping_pong_times = _total_size / sub_seq_length_per_proc *
                     2;  // ping和pong各算一次循环 （2）

  // 分2个Section处理时，不是2的幂，未必能整除
  if (_total_size % sub_seq_length_per_proc > 0) {
    *ping_pong_times += 2;  // 并非是尾块  (尾块只有一个block )
  }
}

#endif

#endif  // __VECTORFORWARD_H__

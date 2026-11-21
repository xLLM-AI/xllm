#include "kernel_operator.h"
using namespace AscendC;

class RowColStatsKernel {
 public:
  __aicore__ inline void Init(GM_ADDR x, GM_ADDR partialRows, GM_ADDR partialCols,
                             uint32_t rows, uint32_t cols) {
    rows_ = rows; cols_ = cols; colTiles_ = (cols + 255) / 256;
    x_.SetGlobalBuffer((__gm__ half*)x);
    pr_.SetGlobalBuffer((__gm__ float*)partialRows);
    pc_.SetGlobalBuffer((__gm__ float*)partialCols);
    pipe_.InitBuffer(input_, 64 * 256 * sizeof(half));
    pipe_.InitBuffer(values_, 64 * 256 * sizeof(float));
    pipe_.InitBuffer(rowMax_, 64 * sizeof(float));
    pipe_.InitBuffer(colMax_, 256 * sizeof(float));
    pipe_.InitBuffer(scratch_, 256 * sizeof(float));
    pipe_.InitBuffer(scalar_, 256);
  }
  __aicore__ inline void Process() {
    uint32_t jobs = ((rows_ + 63) / 64) * colTiles_;
    auto input = input_.Get<half>();
    auto values = values_.Get<float>();
    auto rowMax = rowMax_.Get<float>();
    auto colMax = colMax_.Get<float>();
    auto scratch = scratch_.Get<float>();
    auto scalar = scalar_.Get<float>();
    for (uint32_t job = GetBlockIdx(); job < jobs; job += GetBlockNum()) {
      uint32_t row = job / colTiles_ * 64;
      uint32_t col = job % colTiles_ * 256;
      uint32_t nr = rows_ - row < 64 ? rows_ - row : 64;
      uint32_t nc = cols_ - col < 256 ? cols_ - col : 256;
      if (nr != 64 || nc != 256) Duplicate(input, static_cast<half>(0), 64 * 256);
      Duplicate(rowMax, 0.0f, 64);
      Duplicate(colMax, 0.0f, 256);
      PipeBarrier<PIPE_ALL>();
      DataCopyExtParams load{static_cast<uint16_t>(nr), nc * 2,
                             (cols_ - nc) * 2, (256 - (nc + 15) / 16 * 16) / 16, 0};
      DataCopyPadExtParams<half> padding{false, 0, 0, 0};
      DataCopyPad(input, x_[static_cast<uint64_t>(row) * cols_ + col], load, padding);
      PipeBarrier<PIPE_ALL>();
      Cast(values, input, RoundMode::CAST_NONE, 64 * 256);
      PipeBarrier<PIPE_ALL>();
      Abs(values, values, 64 * 256);
      PipeBarrier<PIPE_ALL>();
      Duplicate(scratch, 0.0f, 256);
      PipeBarrier<PIPE_ALL>();
      // Each vector repeat reduces one row's 64-column group. The 256-column
      // tile needs at most four groups, then four vector max operations.
      for (uint32_t offset = 0; offset < nc; offset += 64) {
        uint64_t mask = nc - offset < 64 ? nc - offset : 64;
        WholeReduceMax(scratch[offset], values[offset], mask, static_cast<uint8_t>(nr),
                       1, 1, 32, ReduceOrder::ORDER_ONLY_VALUE);
      }
      PipeBarrier<PIPE_ALL>();
      for (uint32_t offset = 0; offset < nc; offset += 64) {
        Max(rowMax, rowMax, scratch[offset], 64);
        PipeBarrier<PIPE_ALL>();
      }
      // Binary tree over all 64 rows; missing rows were initialized to zero.
      for (uint32_t step = 32; step; step /= 2) {
        Max(values, values, values[step * 256], step * 256);
        PipeBarrier<PIPE_ALL>();
      }
      DataCopy(colMax, values, 256);
      PipeBarrier<PIPE_ALL>();
      DataCopy(pr_[static_cast<uint64_t>(job) * 64], rowMax, 64);
      DataCopy(pc_[static_cast<uint64_t>(job) * 256], colMax, 256);
      PipeBarrier<PIPE_ALL>();
    }
  }
 private:
  TPipe pipe_;
  TBuf<TPosition::VECCALC> input_, values_, rowMax_, colMax_, scratch_, scalar_;
  GlobalTensor<half> x_;
  GlobalTensor<float> pr_, pc_;
  uint32_t rows_, cols_, colTiles_;
};


#include "kernel_operator.h"
using namespace AscendC;

__aicore__ inline void RowColMerge(
    GM_ADDR partialRows, GM_ADDR partialCols, GM_ADDR rowAbsmax,
    GM_ADDR colAbsmax, uint32_t rows, uint32_t cols) {
  TPipe pipe;
  TBuf<TPosition::VECCALC> accBuffer, inputBuffer;
  pipe.InitBuffer(accBuffer, 256 * sizeof(float));
  pipe.InitBuffer(inputBuffer, 256 * sizeof(float));
  auto acc = accBuffer.Get<float>();
  auto input = inputBuffer.Get<float>();
  GlobalTensor<float> pr, pc, ro, co;
  pr.SetGlobalBuffer((__gm__ float*)partialRows);
  pc.SetGlobalBuffer((__gm__ float*)partialCols);
  ro.SetGlobalBuffer((__gm__ float*)rowAbsmax);
  co.SetGlobalBuffer((__gm__ float*)colAbsmax);
  uint32_t rt = (rows + 63) / 64, ct = (cols + 255) / 256;
  for (uint32_t job = GetBlockIdx(); job < rt + ct; job += GetBlockNum()) {
    Duplicate(acc, 0.0f, 256);
    PipeBarrier<PIPE_ALL>();
    if (job < rt) {
      for (uint32_t c = 0; c < ct; ++c) {
        DataCopy(input, pr[(static_cast<uint64_t>(job) * ct + c) * 64], 64);
        PipeBarrier<PIPE_ALL>();
        Max(acc, acc, input, 64);
        PipeBarrier<PIPE_ALL>();
      }
      uint32_t count = rows - job * 64 < 64 ? rows - job * 64 : 64;
      DataCopyPad(ro[job * 64], acc, DataCopyExtParams{1, count * 4, 0, 0, 0});
    } else {
      uint32_t c = job - rt;
      for (uint32_t r = 0; r < rt; ++r) {
        DataCopy(input, pc[(static_cast<uint64_t>(r) * ct + c) * 256], 256);
        PipeBarrier<PIPE_ALL>();
        Max(acc, acc, input, 256);
        PipeBarrier<PIPE_ALL>();
      }
      uint32_t count = cols - c * 256 < 256 ? cols - c * 256 : 256;
      DataCopyPad(co[c * 256], acc, DataCopyExtParams{1, count * 4, 0, 0, 0});
    }
    PipeBarrier<PIPE_ALL>();
  }
}

#include "kernel_operator.h"
using namespace AscendC;

class RowColQuantKernel {
 public:
  __aicore__ inline void Init(GM_ADDR x, GM_ADDR rows, GM_ADDR cols,
      GM_ADDR qr, GM_ADDR qc, uint32_t m, uint32_t n) {
    m_ = m; n_ = n; ct_ = (n + 255) / 256;
    x_.SetGlobalBuffer((__gm__ half*)x);
    r_.SetGlobalBuffer((__gm__ float*)rows);
    c_.SetGlobalBuffer((__gm__ float*)cols);
    qr_.SetGlobalBuffer((__gm__ int8_t*)qr);
    qc_.SetGlobalBuffer((__gm__ int8_t*)qc);
    pipe_.InitBuffer(input_, 32 * 256 * sizeof(half));
    pipe_.InitBuffer(value_, 32 * 256 * sizeof(float));
    pipe_.InitBuffer(rowOut_, 32 * 256);
    pipe_.InitBuffer(colOut_, 32 * 256);
    pipe_.InitBuffer(rowScale_, 32 * sizeof(float));
    pipe_.InitBuffer(colScale_, 256 * sizeof(float));
    pipe_.InitBuffer(temp_, 256 * sizeof(float));
    pipe_.InitBuffer(intTemp_, 32 * 256 * sizeof(int32_t));
    pipe_.InitBuffer(halfTemp_, 32 * 256 * sizeof(half));
  }
  __aicore__ inline void Process() {
    auto input = input_.Get<half>();
    auto value = value_.Get<float>();
    auto rout = rowOut_.Get<int8_t>();
    auto cout = colOut_.Get<int8_t>();
    auto rs = rowScale_.Get<float>();
    auto cs = colScale_.Get<float>();
    auto temp = temp_.Get<float>();
    for (uint32_t job = GetBlockIdx(); job < ((m_ + 31) / 32) * ct_; job += GetBlockNum()) {
      uint32_t row = job / ct_ * 32, col = job % ct_ * 256;
      uint32_t nr = m_ - row < 32 ? m_ - row : 32;
      uint32_t nc = n_ - col < 256 ? n_ - col : 256;
      if (nr != 32 || nc != 256) Duplicate(input, static_cast<half>(0), 32 * 256);
      PipeBarrier<PIPE_ALL>();
      DataCopyPad(input, x_[static_cast<uint64_t>(row) * n_ + col],
          DataCopyExtParams{static_cast<uint16_t>(nr), nc * 2,
                           (n_ - nc) * 2, (256 - (nc + 15) / 16 * 16) / 16, 0},
          DataCopyPadExtParams<half>{false, 0, 0, 0});
      DataCopyPad(rs, r_[row], DataCopyExtParams{1, nr * 4, 0, 0, 0},
                  DataCopyPadExtParams<float>{false, 0, 0, 0});
      DataCopyPad(cs, c_[col], DataCopyExtParams{1, nc * 4, 0, 0, 0},
                  DataCopyPadExtParams<float>{false, 0, 0, 0});
      PipeBarrier<PIPE_ALL>();
      Cast(value, input, RoundMode::CAST_NONE, nr * 256);
      Maxs(cs, cs, 1.0e-20f, nc);
      Duplicate(temp, 127.0f, nc);
      PipeBarrier<PIPE_ALL>();
      Div(cs, temp, cs, nc);
      PipeBarrier<PIPE_ALL>();
      for (uint32_t r = 0; r < nr; ++r) {
        float scale = rs.GetValue(r);
        Muls(value[r * 256], value[r * 256], scale == 0 ? 0.0f : 127.0f / scale, nc);
      }
      PipeBarrier<PIPE_ALL>();
      RoundAndStore(rout, value, nr * 256);
      // The FP16 tile is unchanged by row quantization; reuse it for columns
      // instead of reading the same tile from HBM a second time.
      Cast(value, input, RoundMode::CAST_NONE, nr * 256);
      PipeBarrier<PIPE_ALL>();
      // Broadcast each 64-element column-scale group to every row using a
      // zero source repeat stride, while rows advance by 256 FP32 elements.
      for (uint32_t offset = 0; offset < nc; offset += 64) {
        uint64_t mask = nc - offset < 64 ? nc - offset : 64;
        Mul(value[offset], value[offset], cs[offset], mask, static_cast<uint8_t>(nr),
            BinaryRepeatParams{1, 1, 1, 32, 32, 0});
      }
      PipeBarrier<PIPE_ALL>();
      RoundAndStore(cout, value, nr * 256);
      DataCopyExtParams store{static_cast<uint16_t>(nr), nc,
                               (256 - (nc + 31) / 32 * 32) / 32, n_ - nc, 0};
      DataCopyPad(qr_[static_cast<uint64_t>(row) * n_ + col], rout, store);
      DataCopyPad(qc_[static_cast<uint64_t>(row) * n_ + col], cout, store);
      PipeBarrier<PIPE_ALL>();
    }
  }
 private:
  __aicore__ inline void RoundAndStore(LocalTensor<int8_t> dst,
      LocalTensor<float> src, uint32_t count) {
    auto ints = intTemp_.Get<int32_t>();
    auto halves = halfTemp_.Get<half>();
    Maxs(src, src, -127.0f, count);
    PipeBarrier<PIPE_ALL>();
    Mins(src, src, 127.0f, count);
    PipeBarrier<PIPE_ALL>();
    // Round in FP32 before narrowing; FP16 rounding of the original product
    // could otherwise move values across an INT8 half-integer boundary.
    Cast(ints, src, RoundMode::CAST_RINT, count);
    PipeBarrier<PIPE_ALL>();
    Cast(src, ints, RoundMode::CAST_NONE, count);
    PipeBarrier<PIPE_ALL>();
    Cast(halves, src, RoundMode::CAST_RINT, count);
    PipeBarrier<PIPE_ALL>();
    Cast(dst, halves, RoundMode::CAST_TRUNC, count);
    PipeBarrier<PIPE_ALL>();
  }
  TPipe pipe_;
  TBuf<TPosition::VECCALC> input_, value_, rowOut_, colOut_, rowScale_,
      colScale_, temp_, intTemp_, halfTemp_;
  GlobalTensor<half> x_;
  GlobalTensor<float> r_, c_;
  GlobalTensor<int8_t> qr_, qc_;
  uint32_t m_, n_, ct_;
};


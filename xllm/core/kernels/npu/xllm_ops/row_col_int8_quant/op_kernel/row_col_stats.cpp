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

extern "C" __global__ __aicore__ void row_col_stats(
    GM_ADDR x, GM_ADDR pr, GM_ADDR pc, uint32_t rows, uint32_t cols) {
  RowColStatsKernel k;
  k.Init(x, pr, pc, rows, cols);
  k.Process();
}
extern "C" void row_col_stats_launch(uint32_t blocks, void* stream,
    uint8_t* x, uint8_t* pr, uint8_t* pc, uint32_t rows, uint32_t cols) {
  row_col_stats<<<blocks, nullptr, stream>>>(x, pr, pc, rows, cols);
}

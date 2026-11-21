#include "kernel_operator.h"
using namespace AscendC;

extern "C" __global__ __aicore__ void row_col_merge(
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
extern "C" void row_col_merge_launch(uint32_t blocks, void* stream,
    uint8_t* pr, uint8_t* pc, uint8_t* r, uint8_t* c, uint32_t rows, uint32_t cols) {
  row_col_merge<<<blocks, nullptr, stream>>>(pr, pc, r, c, rows, cols);
}

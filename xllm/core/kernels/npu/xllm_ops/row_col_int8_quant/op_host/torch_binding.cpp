#include <ATen/ATen.h>
#include <torch/library.h>
#include "torch_npu/csrc/framework/OpCommand.h"
#include "torch_npu/csrc/core/npu/NPUGuard.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/core/npu/NPUCachingAllocator.h"
#include "api.h"
#include <algorithm>
#include <cmath>
#include <vector>

static uint8_t* ptr(const at::Tensor& t) { return static_cast<uint8_t*>(t.data_ptr()); }
static void record(const std::vector<at::Tensor>& ts, c10_npu::NPUStream stream) {
  for (const auto& t : ts) if (t.numel())
    c10_npu::NPUCachingAllocator::recordStream(t.storage().data_ptr(), stream);
}
static std::vector<at::Tensor> run_native(const std::vector<at::Tensor>& in, double eps) {
  TORCH_CHECK(in.size() == 1, "wrong input count");
  const auto& x = in[0];
  TORCH_CHECK(x.device().type() == c10::DeviceType::PrivateUse1, "NPU inputs required");
  for (const auto& t : in)
    TORCH_CHECK(t.device() == x.device() && t.scalar_type() == x.scalar_type() && t.is_contiguous(),
                "inputs must be contiguous and share device and dtype");
  TORCH_CHECK(x.scalar_type() == at::kHalf || false,
              "unsupported dtype");
  c10_npu::NPUGuard guard(x.device());
  auto stream = c10_npu::getCurrentNPUStream(x.device().index());
  auto raw_stream = stream.stream(false);
  std::vector<at::Tensor> out, workspace;
  TORCH_CHECK(x.dim()==2,"x must be a matrix");
  auto m=x.size(0),n=x.size(1);
  TORCH_CHECK(m>=1 && m<=4096 && n>=2 && n<=4096 && n%2==0,"shape outside contract");
  auto rt=(m+63)/64,ct=(n+255)/256;
  out={at::empty(x.sizes(),x.options().dtype(at::kChar)),at::empty(x.sizes(),x.options().dtype(at::kChar)),
       at::empty({m},x.options().dtype(at::kFloat)),at::empty({n},x.options().dtype(at::kFloat))};
  workspace={at::empty({rt*ct*64},x.options().dtype(at::kFloat)),
             at::empty({rt*ct*256},x.options().dtype(at::kFloat))};
  // Queue raw pointers like the native backend. The NPU caching allocator
  // tracks use on this stream; retaining Tensor objects in the host queue
  // would prevent same-stream reuse and inflate memory/latency for large data.
  std::vector<uint8_t*> pointers;
  for (const auto& tensor : in) pointers.push_back(ptr(tensor));
  for (const auto& tensor : out) pointers.push_back(ptr(tensor));
  for (const auto& tensor : workspace) pointers.push_back(ptr(tensor));
  // Outputs and scratch are allocated on the current launch stream.
 // Its FIFO ordering already protects reuse. recordStream is required for
 // inputs from other streams; recording fresh outputs forces needless events.
 record(in,stream);
  auto submit = [=]() -> int {
    row_col_stats_launch(std::min<int64_t>(48,rt*ct),raw_stream,pointers[0],pointers[5],pointers[6],m,n);
    row_col_merge_launch(std::min<int64_t>(48,rt+ct),raw_stream,pointers[5],pointers[6],pointers[3],pointers[4],m,n);
    row_col_quant_launch(std::min<int64_t>(48,rt*ct),raw_stream,pointers[0],pointers[3],pointers[4],pointers[1],pointers[2],m,n);
    return 0;
  };
  at_npu::native::OpCommand::RunOpApiV2("RowColInt8Quant", submit);

  return out;
}
TORCH_LIBRARY_FRAGMENT(cann_community, m) {
  m.def("row_col_int8_quant(Tensor[] inputs, float epsilon=1e-5) -> Tensor[]");
}
TORCH_LIBRARY_IMPL(cann_community, PrivateUse1, m) {
  m.impl("row_col_int8_quant", TORCH_FN(run_native));
}

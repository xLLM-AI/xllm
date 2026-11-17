// Copyright 2026 The xLLM Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://github.com/xLLM-AI/xllm/blob/main/LICENSE
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Host-side helpers for the packed page-indices operator.
//
// The device compute stays in the Triton Ascend kernels.  This extension owns
// the host-side scheduling pieces that must not be re-derived in Python on the
// hot path: tensor metadata, output allocation on the current NPU device,
// construction/refresh of the -1-prefixed mapping table, and the native NPU
// Gather dispatch used by the large-output path.
#include <torch/extension.h>

#include <c10/core/DeviceType.h>

#include <limits>
#include <vector>

namespace {

void check_common(const at::Tensor& X, const at::Tensor& M, const at::Tensor& cu) {
  TORCH_CHECK(X.defined() && M.defined() && cu.defined(), "inputs must be defined");
  TORCH_CHECK(X.scalar_type() == at::kInt && M.scalar_type() == at::kInt &&
                  cu.scalar_type() == at::kInt,
              "all inputs must be int32");
  TORCH_CHECK(X.is_contiguous() && M.is_contiguous() && cu.is_contiguous(),
              "all inputs must be contiguous");
  TORCH_CHECK(X.device() == M.device() && X.device() == cu.device(),
              "all inputs must be on the same device");
  TORCH_CHECK(X.numel() <= std::numeric_limits<int32_t>::max() &&
                  M.numel() <= std::numeric_limits<int32_t>::max() &&
                  cu.numel() <= std::numeric_limits<int32_t>::max(),
              "input indexing exceeds int32");
}

}  // namespace

// Metadata used to schedule the kernels.  Called once from prepare(), never
// from prepared.run().
std::vector<int64_t> packed_meta(const at::Tensor& X, const at::Tensor& M,
                                 const at::Tensor& cu) {
  check_common(X, M, cu);
  TORCH_CHECK(X.dim() == 2, "X must be [R, C]");
  TORCH_CHECK(M.dim() == 1 || M.dim() == 2, "M must be [B] or [L, B]");
  TORCH_CHECK(M.size(-1) >= 1, "B must be >= 1");
  TORCH_CHECK(M.dim() == 1 || M.size(0) >= 1, "L must be >= 1");
  TORCH_CHECK(cu.dim() == 1 && cu.size(0) == X.size(0) + 1,
              "cu must be [R + 1]");
  const int64_t R = X.size(0);
  const int64_t C = X.size(1);
  const int64_t L = M.dim() == 1 ? 1 : M.size(0);
  const int64_t B = M.size(-1);
  const int64_t P = cu.size(0) > 0 ? cu[cu.size(0) - 1].item<int64_t>() : 0;
  return {R, C, L, B, P, M.dim() == 1 ? 1 : 0};
}

// Independent output allocation on the input device.
at::Tensor packed_empty(const at::Tensor& X, int64_t layers, int64_t total,
                        bool single) {
  auto options = X.options().dtype(at::kInt);
  if (single) {
    return at::empty({total}, options);
  }
  return at::empty({layers, total}, options);
}

// M_aug[:, 0] = -1, M_aug[:, 1:] = M.  Index 0 encodes a negative logical page;
// index x + 1 encodes a valid page x.  Built once in prepare().
at::Tensor packed_make_aug(const at::Tensor& M) {
  auto options = M.options().dtype(at::kInt);
  if (M.dim() == 1) {
    auto out = at::empty({M.size(0) + 1}, options);
    out.narrow(0, 0, 1).fill_(-1);
    out.narrow(0, 1, M.size(0)).copy_(M);
    return out;
  }
  auto out = at::empty({M.size(0), M.size(1) + 1}, options);
  out.narrow(1, 0, 1).fill_(-1);
  out.narrow(1, 1, M.size(1)).copy_(M);
  return out;
}

// Refresh the mapping values after an in-place M update or a set_ rebind.
void packed_refresh_aug(at::Tensor& aug, const at::Tensor& M) {
  if (M.dim() == 1) {
    aug.narrow(0, 1, M.size(0)).copy_(M);
  } else {
    aug.narrow(1, 1, M.size(1)).copy_(M);
  }
}

// Native NPU Gather from the augmented mapping table.
at::Tensor packed_gather(const at::Tensor& m_aug, const at::Tensor& idx,
                         bool single) {
  return at::index_select(m_aug, single ? 0 : 1, idx);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("packed_meta", &packed_meta, "packed page indices metadata");
  m.def("packed_empty", &packed_empty, "independent int32 output allocation");
  m.def("packed_make_aug", &packed_make_aug, "build the -1-prefixed mapping");
  m.def("packed_refresh_aug", &packed_refresh_aug, "refresh the mapping values");
  m.def("packed_gather", &packed_gather, "native NPU gather");
}

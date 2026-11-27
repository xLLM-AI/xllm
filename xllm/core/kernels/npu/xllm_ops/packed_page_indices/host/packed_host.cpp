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

// Host-side planning for the PackedPageIndices operator.
//
// The extension owns every value-dependent decision: cu validation, page
// range checks, the total packed length P, the per-layer mapping shape, and
// the int32 element limit. It returns the frozen launch plan to Python, which
// constructs the Triton plan and launches the device kernel. Keeping this
// logic on the host avoids one device read per validated prefix and keeps the
// timed run() free of preparation work.

#include <torch/extension.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <tuple>

namespace {

int64_t require_nonnegative_product(int64_t left, int64_t right, const char* name) {
  TORCH_CHECK(left >= 0 && right >= 0, name, " must be non-negative");
  TORCH_CHECK(
      left == 0 || right <= std::numeric_limits<int64_t>::max() / left,
      name,
      " overflows int64");
  return left * right;
}

}  // namespace

std::tuple<int64_t, int64_t, int64_t, bool> plan(
    torch::Tensor x,
    torch::Tensor m,
    torch::Tensor cu) {
  TORCH_CHECK(x.dim() == 2, "X must be 2-D, got ", x.dim(), "-D");
  TORCH_CHECK(cu.dim() == 1, "cu must be 1-D, got ", cu.dim(), "-D");
  TORCH_CHECK(m.dim() == 1 || m.dim() == 2, "M must be 1-D or 2-D, got ", m.dim(), "-D");
  TORCH_CHECK(x.scalar_type() == torch::kInt32, "X must be int32");
  TORCH_CHECK(m.scalar_type() == torch::kInt32, "M must be int32");
  TORCH_CHECK(cu.scalar_type() == torch::kInt32, "cu must be int32");
  TORCH_CHECK(x.device().is_cpu() && m.device().is_cpu() && cu.device().is_cpu(), "host planning takes CPU tensors");

  auto x_cpu = x.contiguous();
  auto m_cpu = m.contiguous();
  auto cu_cpu = cu.contiguous();

  const int64_t rows = x_cpu.size(0);
  const int64_t columns = x_cpu.size(1);
  TORCH_CHECK(cu_cpu.numel() == rows + 1, "cu must have R + 1 = ", rows + 1, " entries, got ", cu_cpu.numel());

  const bool single = m_cpu.dim() == 1;
  const int64_t layers = single ? 1 : m_cpu.size(0);
  const int64_t blocks = single ? m_cpu.size(0) : m_cpu.size(1);
  TORCH_CHECK(layers >= 1 && blocks >= 1, "M must have at least one layer and one block");

  auto cu_values = cu_cpu.accessor<int32_t, 1>();
  TORCH_CHECK(cu_values[0] == 0, "cu[0] must be 0, got ", cu_values[0]);
  int64_t total = 0;
  for (int64_t row = 0; row < rows; ++row) {
    const int64_t start = cu_values[row];
    const int64_t end = cu_values[row + 1];
    TORCH_CHECK(end >= start, "cu must be non-decreasing, got cu[", row + 1, "] = ", end, " after ", start);
    const int64_t count = end - start;
    TORCH_CHECK(count <= columns, "request ", row, " has ", count, " pages but X only has ", columns, " columns");
    total = end;
  }

  auto mapping = m_cpu.reshape({layers, blocks});
  auto mapping_values = mapping.accessor<int32_t, 2>();
  for (int64_t layer = 0; layer < layers; ++layer) {
    for (int64_t block = 0; block < blocks; ++block) {
      TORCH_CHECK(mapping_values[layer][block] >= 0, "M[", layer, "][", block, "] must be non-negative");
    }
  }

  auto x_values = x_cpu.accessor<int32_t, 2>();
  for (int64_t row = 0; row < rows; ++row) {
    const int64_t count = cu_values[row + 1] - cu_values[row];
    for (int64_t column = 0; column < count; ++column) {
      const int32_t page = x_values[row][column];
      TORCH_CHECK(page < blocks, "X[", row, "][", column, "] = ", page, " does not fit in ", blocks, " logical pages");
    }
  }

  const int64_t elements = std::max(
      {require_nonnegative_product(total, layers, "output elements"),
       require_nonnegative_product(rows, columns, "X elements"),
       require_nonnegative_product(layers, blocks, "M elements")});
  TORCH_CHECK(elements <= 2147483647, elements, " elements exceed the int32 index limit");

  return {total, layers, blocks, single};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "plan",
      &plan,
      "Validate PackedPageIndices inputs and return (total, layers, blocks, single)");
}

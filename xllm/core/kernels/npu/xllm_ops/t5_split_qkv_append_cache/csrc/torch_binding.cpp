/* Copyright 2026 The xLLM Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/xLLM-AI/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <c10/core/DeviceGuard.h>
#include <torch/extension.h>

#include <algorithm>
#include <array>
#include <cstdlib>
#include <memory>
#include <tuple>
#include <vector>

#include "acl/acl.h"
#include "aclnn_t5_split_qkv_append_cache.h"
#include "common/t5_plan.h"
#include "torch_npu/csrc/aten/CustomFunctions.h"
#include "torch_npu/csrc/core/npu/NPUCachingAllocator.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"

extern "C" void t5_launch_direct(void*,
                                 void*,
                                 void*,
                                 void*,
                                 void*,
                                 void*,
                                 void*,
                                 uint32_t,
                                 uint32_t,
                                 uint32_t,
                                 uint32_t);
namespace {
using Tensors = std::array<torch::Tensor, 3>;
using Shape = std::array<int64_t, 4>;
struct Arguments {
  std::vector<torch::Tensor> inputs_;
  int64_t heads_ = -2, dim_ = -2;
};
Arguments parse_arguments(pybind11::handle inputs,
                          pybind11::handle attributes) {
  if ((!PyTuple_Check(inputs.ptr()) && !PyList_Check(inputs.ptr())) ||
      PySequence_Size(inputs.ptr()) != 3) {
    throw pybind11::value_error(
        "inputs must be a tuple/list of exactly three tensors");
  }
  if (!PyDict_Check(attributes.ptr())) {
    throw pybind11::type_error("attributes must be a dictionary");
  }
  Arguments result;
  Py_ssize_t position = 0;
  PyObject *key, *value;
  while (PyDict_Next(attributes.ptr(), &position, &key, &value)) {
    const bool heads = PyUnicode_Check(key) &&
                       PyUnicode_CompareWithASCIIString(key, "headNum") == 0;
    const bool dim = PyUnicode_Check(key) &&
                     PyUnicode_CompareWithASCIIString(key, "headDim") == 0;
    if (!heads && !dim) {
      throw pybind11::value_error("unknown attributes");
    }
    // Exact int excludes bool and int subclasses, matching the public contract.
    if (!PyLong_CheckExact(value)) {
      throw pybind11::value_error("attributes must be integers");
    }
    const int64_t number = PyLong_AsLongLong(value);
    if (PyErr_Occurred()) {
      throw pybind11::error_already_set();
    }
    if (number < 1 || number > (heads ? 128 : 256)) {
      throw pybind11::value_error("attribute out of range");
    }
    (heads ? result.heads_ : result.dim_) = number;
  }
  result.inputs_ = pybind11::cast<std::vector<torch::Tensor>>(inputs);
  return result;
}
Shape validate(const std::vector<torch::Tensor>& xs,
               int64_t head_num,
               int64_t head_dim) {
  TORCH_CHECK(xs.size() == 3, "expected qkv, pastKey, pastValue");
  const auto& q = xs[0];
  const auto& k = xs[1];
  const auto& v = xs[2];
  TORCH_CHECK(q.device().type() == c10::DeviceType::PrivateUse1,
              "inputs must be on NPU");
  TORCH_CHECK(
      q.scalar_type() == torch::kHalf || q.scalar_type() == torch::kBFloat16,
      "FP16 or BF16 required");
  for (const auto& x : xs) {
    TORCH_CHECK(x.device() == q.device(),
                "all inputs must use the same device");
    TORCH_CHECK(x.scalar_type() == q.scalar_type(), "input dtype mismatch");
    TORCH_CHECK(x.layout() == c10::kStrided && x.is_contiguous(),
                "contiguous ND input required");
    const int64_t format = at_npu::native::custom_ops::get_npu_format(x);
    // Base NCHW/NCDHW descriptors have the same contiguous physical bytes.
    // Reject padded/blocked internal formats instead of pretending they are ND.
    TORCH_CHECK(format == ACL_FORMAT_ND || format == ACL_FORMAT_NCHW ||
                    format == ACL_FORMAT_NCDHW,
                "contiguous ND/base storage required; internal blocked format "
                "is unsupported");
  }
  TORCH_CHECK(q.dim() == 3 && k.dim() == 4 && v.dim() == 4,
              "invalid input rank");
  TORCH_CHECK(k.sizes() == v.sizes(), "cache shape mismatch");
  const int64_t b = k.size(0), h = k.size(1), l = k.size(2), d = k.size(3);
  TORCH_CHECK(t5qkv::valid_shape(b, h, l, d),
              "shape range/output element limit exceeded");
  TORCH_CHECK(q.size(0) == b && q.size(1) == 1 && q.size(2) == 3 * h * d,
              "invalid packed QKV shape");
  TORCH_CHECK(
      (head_num == -2 || head_num == h) && (head_dim == -2 || head_dim == d),
      "attribute/shape mismatch");
  return {b, h, l, d};
}

class TensorDeleter final {
 public:
  void operator()(aclTensor* x) const {
    if (x) {
      aclDestroyTensor(x);
    }
  }
};
using Descriptor = std::unique_ptr<aclTensor, TensorDeleter>;
Descriptor describe(const torch::Tensor& x) {
  // data_ptr() already includes storage_offset(). Describe this contiguous span
  // with offset zero, instead of applying the PyTorch offset a second time.
  const int64_t storage_size = x.numel();
  aclDataType type = x.scalar_type() == torch::kHalf ? ACL_FLOAT16 : ACL_BF16;
  auto* p = aclCreateTensor(x.sizes().data(),
                            x.dim(),
                            type,
                            x.strides().data(),
                            0,
                            ACL_FORMAT_ND,
                            &storage_size,
                            1,
                            x.data_ptr());
  TORCH_CHECK(p, "aclCreateTensor failed");
  return Descriptor(p);
}

using Descriptors = std::array<Descriptor, 6>;
Descriptors describe_all(const std::vector<torch::Tensor>& xs,
                         const Tensors& out) {
  return {describe(xs[0]),
          describe(xs[1]),
          describe(xs[2]),
          describe(out[0]),
          describe(out[1]),
          describe(out[2])};
}
aclOpExecutor* build_executor(const Descriptors& desc, const Shape& shape) {
  uint64_t workspace_size = 0;
  aclOpExecutor* executor = nullptr;
  aclError status = aclnnT5SplitQkvAppendCacheGetWorkspaceSize(desc[0].get(),
                                                               desc[1].get(),
                                                               desc[2].get(),
                                                               shape[1],
                                                               shape[3],
                                                               desc[3].get(),
                                                               desc[4].get(),
                                                               desc[5].get(),
                                                               &workspace_size,
                                                               &executor);
  TORCH_CHECK(status == ACL_SUCCESS, "GetWorkspaceSize failed: ", status);
  TORCH_CHECK(workspace_size == 0,
              "zero-workspace contract violated: ",
              workspace_size);
  return executor;
}
void launch_aclnn(aclOpExecutor* executor, aclrtStream stream) {
  const aclError status =
      aclnnT5SplitQkvAppendCache(nullptr, 0, executor, stream);
  TORCH_CHECK(
      status == ACL_SUCCESS, "aclnnT5SplitQkvAppendCache failed: ", status);
}

class ExecutorDeleter final {
 public:
  void operator()(aclOpExecutor* p) const {
    if (p) {
      aclDestroyAclOpExecutor(p);
    }
  }
};
class AclPlan final {
 public:
  Shape shape_;
  torch::ScalarType dtype_;
  c10::Device device_;
  aclrtStream stream_;
  std::array<std::vector<int64_t>, 3> strides_;
  Descriptors desc_;
  // Declared last: destroy the executor before its descriptor resources.
  std::unique_ptr<aclOpExecutor, ExecutorDeleter> executor_;
  AclPlan(const std::vector<torch::Tensor>& xs,
          const Tensors& out,
          const Shape& s,
          aclrtStream st)
      : shape_(s),
        dtype_(xs[0].scalar_type()),
        device_(xs[0].device()),
        stream_(st),
        strides_{xs[0].strides().vec(),
                 xs[1].strides().vec(),
                 xs[2].strides().vec()},
        desc_(describe_all(xs, out)) {}
  bool matches(const std::vector<torch::Tensor>& xs,
               const Shape& s,
               aclrtStream st) const {
    if (shape_ != s || dtype_ != xs[0].scalar_type() ||
        device_ != xs[0].device() || stream_ != st) {
      return false;
    }
    for (size_t i = 0; i < xs.size(); ++i) {
      if (xs[i].strides().size() != strides_[i].size() ||
          !std::equal(xs[i].strides().begin(),
                      xs[i].strides().end(),
                      strides_[i].begin())) {
        return false;
      }
    }
    return true;
  }
};

std::unique_ptr<AclPlan>& cached_plan() {
  static thread_local std::unique_ptr<AclPlan> plan;
  return plan;
}

void submit_aclnn(const std::vector<torch::Tensor>& xs,
                  const Tensors& out,
                  const Shape& shape,
                  aclrtStream stream) {
  // One metadata-only entry per calling thread. It owns no ATen tensors or
  // device workspace, and is replaced on metadata/device/stream changes.
  auto& plan = cached_plan();
  static thread_local bool reuse_supported = true;
  // Explicit same-build control for experiments; default behavior is enabled.
  static const bool reuse_enabled = [] {
    const char* setting = std::getenv("T5_ACLNN_EXECUTOR_REUSE");
    return setting == nullptr || std::string(setting) != "0";
  }();
  bool aligned = true;
  for (const auto& x : xs) {
    aligned &= (reinterpret_cast<uintptr_t>(x.data_ptr()) & 31) == 0;
  }
  if (!aligned || !reuse_supported || !reuse_enabled) {
    auto desc = describe_all(xs, out);
    launch_aclnn(build_executor(desc, shape), stream);
    return;
  }
  if (!plan || !plan->matches(xs, shape, stream)) {
    auto next = std::make_unique<AclPlan>(xs, out, shape, stream);
    auto* executor = build_executor(next->desc_, shape);
    const aclError status = aclSetAclOpExecutorRepeatable(executor);
    if (status != ACL_SUCCESS) {
      // The ordinary second phase releases a non-repeatable executor.
      reuse_supported = false;
      TORCH_WARN("ACLNN executor reuse unavailable: ",
                 status,
                 "; using normal submission");
      launch_aclnn(executor, stream);
      return;
    }
    next->executor_.reset(executor);
    plan = std::move(next);
  }
  // Refresh every live address, including freshly allocated outputs. Empty
  // input descriptors remain empty; their original null address is unchanged.
  for (size_t i = 0; i < 6; ++i) {
    const auto& tensor = i < 3 ? xs[i] : out[i - 3];
    if (!tensor.numel()) {
      continue;
    }
    const aclError status = aclSetTensorAddr(
        plan->executor_.get(), i, plan->desc_[i].get(), tensor.data_ptr());
    TORCH_CHECK(status == ACL_SUCCESS,
                "aclSetTensorAddr failed at tensor ",
                i,
                ": ",
                status);
  }
  launch_aclnn(plan->executor_.get(), stream);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> execute(
    const std::vector<torch::Tensor>& xs,
    int64_t head_num,
    int64_t head_dim,
    bool direct) {
  const auto shape = validate(xs, head_num, head_dim);
  const auto [b, h, l, d] = shape;
  c10::DeviceGuard device_guard(xs[0].device());
  const auto options = xs[0].options().requires_grad(false);
  Tensors out{torch::empty({b, h, 1, d}, options),
              torch::empty({b, h, l + 1, d}, options),
              torch::empty({b, h, l + 1, d}, options)};
  auto stream = c10_npu::getCurrentNPUStream(xs[0].device().index());
  // Official stream() drains pending host submissions, not device work. This
  // preserves ordering with preceding PyTorch operations without another host
  // callback/captured-output queue. Kernel execution remains asynchronous.
  const aclrtStream raw_stream = stream.stream();
  TORCH_CHECK(raw_stream, "failed to obtain ordered current NPU stream");
  auto call = [&]() -> int32_t {
    for (const auto& x : xs) {
      if (x.numel()) {
        c10_npu::NPUCachingAllocator::recordStream(x.storage().data_ptr(),
                                                   stream);
      }
    }
    if (direct) {
      t5_launch_direct(raw_stream,
                       xs[0].data_ptr(),
                       xs[1].data_ptr(),
                       xs[2].data_ptr(),
                       out[0].data_ptr(),
                       out[1].data_ptr(),
                       out[2].data_ptr(),
                       shape[0],
                       shape[1],
                       shape[2],
                       shape[3]);
      return 0;
    }
    submit_aclnn(xs, out, shape, raw_stream);
    return 0;
  };
  call();
  return {std::move(out[0]), std::move(out[1]), std::move(out[2])};
}
}  // namespace
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // Register after torch_npu import, so the main thread releases CANN metadata
  // before torch_npu's earlier atexit handler finalizes its runtime.
  m.def("_clear_cached_plan", []() { cached_plan().reset(); });
  pybind11::module_::import("atexit").attr("register")(
      m.attr("_clear_cached_plan"));
  m.def("validate", [](pybind11::handle inputs, pybind11::handle attributes) {
    auto args = parse_arguments(inputs, attributes);
    pybind11::gil_scoped_release release;
    return validate(args.inputs_, args.heads_, args.dim_);
  });
  m.def("execute",
        [](pybind11::handle inputs, pybind11::handle attributes, bool direct) {
          auto args = parse_arguments(inputs, attributes);
          pybind11::gil_scoped_release release;
          return execute(args.inputs_, args.heads_, args.dim_, direct);
        });
}

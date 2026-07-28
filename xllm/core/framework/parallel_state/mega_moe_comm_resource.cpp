/* Copyright 2025-2026 The xLLM Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/jd-opensource/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "framework/parallel_state/mega_moe_comm_resource.h"

#include <ATen/ATen.h>
#include <dlfcn.h>
#include <glog/logging.h>

#include <array>
#include <cstring>
#include <utility>
#include <vector>

namespace xllm {
namespace {

constexpr uint32_t kHcclMaxRankSize = 1024;
constexpr uint8_t kCommEngineAiv = 4;
constexpr uint8_t kAllToAllOpType = 8;
constexpr char kHcclLibrary[] = "libhccl.so";
constexpr char kHcclFrameworkLibrary[] = "libhccl_fwk.so";
constexpr char kAllToAllAlgorithm[] =
    "AlltoAll=level0:fullmesh;level1:pairwise";

struct MegaMoeCommContext {
  uint32_t ep_rank_id = 0;
  uint32_t rank_size_per_server = 0;
  uint64_t kfc_context_addr = 0;
  std::array<uint64_t, kHcclMaxRankSize> ep_hccl_buffer = {};
  std::array<uint64_t, kHcclMaxRankSize> hcomm_handle = {};
};

static_assert(sizeof(MegaMoeCommContext) % sizeof(int32_t) == 0);

using HcclKfcAllocOpArgs = HcclResult (*)(void**);
using HcclKfcFreeOpArgs = HcclResult (*)(void*);
using HcclKfcOpArgsSetAlgConfig = HcclResult (*)(void*, char*);
using HcclKfcOpArgsSetCommEngine = HcclResult (*)(void*, uint8_t);
using HcclCreateOpResCtx = HcclResult (*)(HcclComm, uint8_t, void*, void**);
using HcclGetRankSize = HcclResult (*)(HcclComm, uint32_t*);
using HcclGetRankId = HcclResult (*)(HcclComm, uint32_t*);
using HcclGetHcclBuffer = HcclResult (*)(HcclComm, void**, uint64_t*);
using HcclGetRemoteIpcHcclBuf = HcclResult (*)(HcclComm,
                                               uint64_t,
                                               void**,
                                               uint64_t*);
using HcclCommGetHandleWithName = HcclResult (*)(const char*, HcclComm*);

template <typename Function>
Function load_symbol(void* library, const char* symbol) {
  return reinterpret_cast<Function>(dlsym(library, symbol));
}

class MegaMoeCommApis final {
 public:
  MegaMoeCommApis() {
    hccl_library_ = dlopen(kHcclLibrary, RTLD_NOW | RTLD_LOCAL);
    if (hccl_library_ == nullptr) {
      missing_symbol_ = kHcclLibrary;
      return;
    }
    hccl_framework_library_ =
        dlopen(kHcclFrameworkLibrary, RTLD_NOW | RTLD_LOCAL);
    if (hccl_framework_library_ == nullptr) {
      missing_symbol_ = kHcclFrameworkLibrary;
      return;
    }
    if (!load_hccl_symbol(kfc_alloc_op_args_, "HcclKfcAllocOpArgs") ||
        !load_hccl_symbol(kfc_free_op_args_, "HcclKfcFreeOpArgs") ||
        !load_hccl_symbol(kfc_set_alg_config_, "HcclKfcOpArgsSetAlgConfig") ||
        !load_hccl_symbol(kfc_set_comm_engine_, "HcclKfcOpArgsSetCommEngine") ||
        !load_hccl_symbol(create_op_res_ctx_, "HcclCreateOpResCtx") ||
        !load_hccl_symbol(get_rank_size_, "HcclGetRankSize") ||
        !load_hccl_symbol(get_rank_id_, "HcclGetRankId") ||
        !load_hccl_symbol(get_hccl_buffer_, "HcclGetHcclBuffer") ||
        !load_framework_symbol(get_remote_ipc_hccl_buffer_,
                               "HcclGetRemoteIpcHcclBuf") ||
        !load_framework_symbol(get_handle_with_name_,
                               "HcclCommGetHandleWithName")) {
      return;
    }
    available_ = true;
  }

  bool available() const { return available_; }
  const std::string& missing_symbol() const { return missing_symbol_; }

  HcclKfcAllocOpArgs kfc_alloc_op_args() const { return kfc_alloc_op_args_; }
  HcclKfcFreeOpArgs kfc_free_op_args() const { return kfc_free_op_args_; }
  HcclKfcOpArgsSetAlgConfig kfc_set_alg_config() const {
    return kfc_set_alg_config_;
  }
  HcclKfcOpArgsSetCommEngine kfc_set_comm_engine() const {
    return kfc_set_comm_engine_;
  }
  HcclCreateOpResCtx create_op_res_ctx() const { return create_op_res_ctx_; }
  HcclGetRankSize get_rank_size() const { return get_rank_size_; }
  HcclGetRankId get_rank_id() const { return get_rank_id_; }
  HcclGetHcclBuffer get_hccl_buffer() const { return get_hccl_buffer_; }
  HcclGetRemoteIpcHcclBuf get_remote_ipc_hccl_buffer() const {
    return get_remote_ipc_hccl_buffer_;
  }
  HcclCommGetHandleWithName get_handle_with_name() const {
    return get_handle_with_name_;
  }

 private:
  template <typename Function>
  bool load_hccl_symbol(Function& function, const char* symbol) {
    function = load_symbol<Function>(hccl_library_, symbol);
    return record_missing_symbol(function, symbol);
  }
  template <typename Function>
  bool load_framework_symbol(Function& function, const char* symbol) {
    function = load_symbol<Function>(hccl_framework_library_, symbol);
    return record_missing_symbol(function, symbol);
  }
  template <typename Function>
  bool record_missing_symbol(Function function, const char* symbol) {
    if (function != nullptr) return true;
    missing_symbol_ = symbol;
    return false;
  }

  void* hccl_library_ = nullptr;
  void* hccl_framework_library_ = nullptr;
  bool available_ = false;
  std::string missing_symbol_;
  HcclKfcAllocOpArgs kfc_alloc_op_args_ = nullptr;
  HcclKfcFreeOpArgs kfc_free_op_args_ = nullptr;
  HcclKfcOpArgsSetAlgConfig kfc_set_alg_config_ = nullptr;
  HcclKfcOpArgsSetCommEngine kfc_set_comm_engine_ = nullptr;
  HcclCreateOpResCtx create_op_res_ctx_ = nullptr;
  HcclGetRankSize get_rank_size_ = nullptr;
  HcclGetRankId get_rank_id_ = nullptr;
  HcclGetHcclBuffer get_hccl_buffer_ = nullptr;
  HcclGetRemoteIpcHcclBuf get_remote_ipc_hccl_buffer_ = nullptr;
  HcclCommGetHandleWithName get_handle_with_name_ = nullptr;
};

MegaMoeCommApis& comm_apis() {
  static MegaMoeCommApis apis;
  return apis;
}

class OpArgsGuard final {
 public:
  OpArgsGuard(void* op_args, HcclKfcFreeOpArgs free_op_args)
      : op_args_(op_args), free_op_args_(free_op_args) {}
  ~OpArgsGuard() {
    if (op_args_ == nullptr) return;
    const HcclResult result = free_op_args_(op_args_);
    if (result != HCCL_SUCCESS) {
      LOG(ERROR) << "HcclKfcFreeOpArgs failed during cleanup, result="
                 << static_cast<int32_t>(result);
    }
  }
  OpArgsGuard(const OpArgsGuard&) = delete;
  OpArgsGuard& operator=(const OpArgsGuard&) = delete;
  void release() { op_args_ = nullptr; }

 private:
  void* op_args_ = nullptr;
  HcclKfcFreeOpArgs free_op_args_ = nullptr;
};

void check_hccl_result(HcclResult result, const char* operation) {
  CHECK_EQ(result, HCCL_SUCCESS)
      << operation << " failed, result=" << static_cast<int32_t>(result);
}

HcclComm resolve_hccl_comm(const MegaMoeCommSpec& spec) {
  if (spec.hccl_comm != nullptr) {
    return spec.hccl_comm;
  }
  MegaMoeCommApis& apis = comm_apis();
  CHECK(apis.available()) << "missing MegaMoe communication symbol: "
                          << apis.missing_symbol();
  HcclComm comm = nullptr;
  check_hccl_result(apis.get_handle_with_name()(spec.group_name.c_str(), &comm),
                    "HcclCommGetHandleWithName");
  CHECK(comm != nullptr)
      << "HcclCommGetHandleWithName returned null for group '"
      << spec.group_name << "'";
  return comm;
}

MegaMoeCommContext build_context(const MegaMoeCommSpec& spec,
                                 int64_t& ccl_buffer_size) {
  MegaMoeCommApis& apis = comm_apis();
  CHECK(apis.available()) << "missing MegaMoe communication symbol: "
                          << apis.missing_symbol();

  HcclComm comm = resolve_hccl_comm(spec);

  void* op_args = nullptr;
  check_hccl_result(apis.kfc_alloc_op_args()(&op_args), "HcclKfcAllocOpArgs");
  CHECK(op_args != nullptr) << "HcclKfcAllocOpArgs returned null";
  OpArgsGuard op_args_guard(op_args, apis.kfc_free_op_args());

  check_hccl_result(apis.kfc_set_comm_engine()(op_args, kCommEngineAiv),
                    "HcclKfcOpArgsSetCommEngine");
  check_hccl_result(
      apis.kfc_set_alg_config()(op_args, const_cast<char*>(kAllToAllAlgorithm)),
      "HcclKfcOpArgsSetAlgConfig");

  MegaMoeCommContext context;
  void* op_resource_context = nullptr;
  check_hccl_result(apis.create_op_res_ctx()(
                        comm, kAllToAllOpType, op_args, &op_resource_context),
                    "HcclCreateOpResCtx");
  CHECK(op_resource_context != nullptr) << "HcclCreateOpResCtx returned null";
  context.kfc_context_addr = reinterpret_cast<uint64_t>(op_resource_context);

  const HcclResult free_result = apis.kfc_free_op_args()(op_args);
  op_args_guard.release();
  check_hccl_result(free_result, "HcclKfcFreeOpArgs");

  uint32_t rank_size = 0;
  uint32_t rank_id = 0;
  check_hccl_result(apis.get_rank_size()(comm, &rank_size), "HcclGetRankSize");
  check_hccl_result(apis.get_rank_id()(comm, &rank_id), "HcclGetRankId");
  CHECK_EQ(rank_size, static_cast<uint32_t>(spec.ep_world_size));
  CHECK_LT(rank_id, rank_size);
  context.ep_rank_id = rank_id;

  std::vector<void*> rank_buffer_addresses(rank_size, nullptr);
  std::vector<uint64_t> rank_accessible_spans(rank_size, 0);
  for (uint32_t remote_rank_id = 0; remote_rank_id < rank_size;
       ++remote_rank_id) {
    void* remote_address = nullptr;
    uint64_t remote_size = 0;
    const char* buffer_query = nullptr;
    HcclResult result = HCCL_E_INTERNAL;
    if (remote_rank_id == rank_id) {
      result = apis.get_hccl_buffer()(comm, &remote_address, &remote_size);
      ccl_buffer_size = static_cast<int64_t>(remote_size);
      buffer_query = "HcclGetHcclBuffer";
    } else {
      result = apis.get_remote_ipc_hccl_buffer()(
          comm, remote_rank_id, &remote_address, &remote_size);
      buffer_query = "HcclGetRemoteIpcHcclBuf";
    }
    check_hccl_result(result, buffer_query);
    CHECK(remote_address != nullptr);
    rank_buffer_addresses[remote_rank_id] = remote_address;
    rank_accessible_spans[remote_rank_id] = remote_size;
  }

  const auto span_validation = validate_mega_moe_buffer_accessible_spans(
      static_cast<uint64_t>(ccl_buffer_size), rank_accessible_spans);
  CHECK(span_validation.valid)
      << "MegaMoe HCCL accessible span is smaller than the local payload at "
         "rank "
      << span_validation.mismatched_rank
      << ": required_payload=" << span_validation.required_payload_size
      << ", accessible_span=" << span_validation.accessible_span;

  for (uint32_t remote_rank_id = 0; remote_rank_id < rank_size;
       ++remote_rank_id) {
    context.ep_hccl_buffer[remote_rank_id] =
        reinterpret_cast<uint64_t>(rank_buffer_addresses[remote_rank_id]);
  }
  return context;
}

}  // namespace

MegaMoeBufferSpanValidation validate_mega_moe_buffer_accessible_spans(
    uint64_t local_payload_size,
    const std::vector<uint64_t>& rank_accessible_spans) {
  if (local_payload_size == 0) {
    return {false, -1, local_payload_size, 0};
  }
  if (rank_accessible_spans.empty()) {
    return {false, -1, local_payload_size, 0};
  }
  for (size_t rank = 0; rank < rank_accessible_spans.size(); ++rank) {
    const uint64_t accessible_span = rank_accessible_spans[rank];
    if (accessible_span < local_payload_size) {
      return {false,
              static_cast<int32_t>(rank),
              local_payload_size,
              accessible_span};
    }
  }
  return {true, -1, local_payload_size, 0};
}

class MegaMoeCommResource::Impl final {
 public:
  explicit Impl(const MegaMoeCommSpec& spec)
      : device_index_(spec.device_index),
        max_num_tokens_per_rank_(spec.max_num_tokens_per_rank) {
    const MegaMoeCommContext context = build_context(spec, ccl_buffer_size_);
    const int64_t context_elements =
        static_cast<int64_t>(sizeof(context) / sizeof(int32_t));
    host_context_tensor_ =
        at::empty({context_elements}, at::TensorOptions().dtype(at::kInt));
    std::memcpy(
        host_context_tensor_.data_ptr<int32_t>(), &context, sizeof(context));
    context_tensor_ = host_context_tensor_.to(
        at::Device(at::kPrivateUse1, spec.device_index), at::kInt, false, true);
  }

  const at::Tensor& context_tensor() const { return context_tensor_; }
  int64_t ccl_buffer_size() const { return ccl_buffer_size_; }
  int64_t max_num_tokens_per_rank() const { return max_num_tokens_per_rank_; }

 private:
  at::Tensor host_context_tensor_;
  at::Tensor context_tensor_;
  int32_t device_index_ = -1;
  int64_t ccl_buffer_size_ = 0;
  int64_t max_num_tokens_per_rank_ = 0;
};

MegaMoeCommResource::MegaMoeCommResource(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}

MegaMoeCommResource::~MegaMoeCommResource() = default;

std::unique_ptr<MegaMoeCommResource> MegaMoeCommResource::create(
    const MegaMoeCommSpec& spec) {
  CHECK(spec.ep_world_size > 0) << "MegaMoe requires ep_world_size > 0";
  CHECK(spec.device_index >= 0) << "MegaMoe requires device_index >= 0";
  CHECK(spec.max_num_tokens_per_rank > 0)
      << "MegaMoe requires max_num_tokens_per_rank > 0";
  CHECK(comm_apis().available())
      << "MegaMoe communication symbols are unavailable: "
      << comm_apis().missing_symbol();
  return std::unique_ptr<MegaMoeCommResource>(
      new MegaMoeCommResource(std::make_unique<Impl>(spec)));
}

const at::Tensor& MegaMoeCommResource::context_tensor() const {
  return impl_->context_tensor();
}

int64_t MegaMoeCommResource::ccl_buffer_size() const {
  return impl_->ccl_buffer_size();
}

int64_t MegaMoeCommResource::max_num_tokens_per_rank() const {
  return impl_->max_num_tokens_per_rank();
}

bool MegaMoeCommResourceSlot::same_key(const MegaMoeCommSpec& lhs,
                                       const MegaMoeCommSpec& rhs) {
  return lhs.hccl_comm == rhs.hccl_comm && lhs.group_name == rhs.group_name &&
         lhs.ep_world_size == rhs.ep_world_size &&
         lhs.device_index == rhs.device_index &&
         lhs.max_num_tokens_per_rank == rhs.max_num_tokens_per_rank;
}

std::shared_ptr<MegaMoeCommResource> MegaMoeCommResourceSlot::acquire(
    const MegaMoeCommSpec& spec) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (resource_ != nullptr && cached_spec_.has_value() &&
      same_key(cached_spec_.value(), spec)) {
    return resource_;
  }
  std::shared_ptr<MegaMoeCommResource> candidate =
      std::shared_ptr<MegaMoeCommResource>(MegaMoeCommResource::create(spec));
  CHECK(candidate != nullptr)
      << "MegaMoe communication resource factory returned null.";
  cached_spec_ = spec;
  resource_ = std::move(candidate);
  return resource_;
}

void MegaMoeCommResourceSlot::reset() {
  std::shared_ptr<MegaMoeCommResource> released;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    cached_spec_.reset();
    released = std::move(resource_);
  }
}

}  // namespace xllm

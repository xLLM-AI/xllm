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

#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "acl/acl.h"
#include "aclnn_t5_split_qkv_append_cache.h"

namespace {
void check(aclError code, const char* call) {
  if (code != ACL_SUCCESS) {
    throw std::runtime_error(std::string(call) + ": " + std::to_string(code));
  }
}
class Runtime final {
 public:
  aclrtStream stream_{};
  Runtime() {
    check(aclInit(nullptr), "aclInit");
    check(aclrtSetDevice(0), "setDevice");
    check(aclrtCreateStream(&stream_), "createStream");
  }
  Runtime(const Runtime&) = delete;
  Runtime& operator=(const Runtime&) = delete;
  ~Runtime() {
    if (stream_) {
      aclrtSynchronizeStream(stream_);
      aclrtDestroyStream(stream_);
    }
    aclrtResetDevice(0);
    aclFinalize();
  }
};
class Buffer final {
 public:
  void* ptr_{};
  size_t bytes_;
  explicit Buffer(size_t n) : bytes_(n * 2) {
    check(aclrtMalloc(
              &ptr_, std::max<size_t>(bytes_, 32), ACL_MEM_MALLOC_NORMAL_ONLY),
          "malloc");
  }
  ~Buffer() {
    if (ptr_) {
      aclrtFree(ptr_);
    }
  }
  Buffer(const Buffer&) = delete;
  Buffer& operator=(const Buffer&) = delete;
};
class Tensor final {
 public:
  aclTensor* ptr_{};
  Tensor(Buffer& b, std::vector<int64_t> shape, aclDataType type) {
    std::vector<int64_t> strides(shape.size(), 1);
    for (int32_t i = static_cast<int32_t>(shape.size()) - 2; i >= 0; --i) {
      strides[i] = strides[i + 1] * shape[i + 1];
    }
    int64_t storage_elements = b.bytes_ / 2;
    ptr_ = aclCreateTensor(shape.data(),
                           shape.size(),
                           type,
                           strides.data(),
                           0,
                           ACL_FORMAT_ND,
                           &storage_elements,
                           1,
                           b.ptr_);
    if (!ptr_) {
      throw std::runtime_error("aclCreateTensor failed");
    }
  }
  Tensor(const Tensor&) = delete;
  Tensor& operator=(const Tensor&) = delete;
  ~Tensor() {
    if (ptr_) {
      aclDestroyTensor(ptr_);
    }
  }
};
}  // namespace

int main(int argc, char** argv) {
  try {
    const bool negative = argc > 1 && std::string(argv[1]) == "--negative";
    const bool bf16 = argc > 1 && std::string(argv[1]) == "--bf16";
    const aclDataType type = bf16 ? ACL_BF16 : ACL_FLOAT16;
    Runtime runtime;
    constexpr int64_t kBatch = 2, kHeads = 3, kLength = 2, kDim = 5,
                      kElements = kBatch * kHeads * kDim;
    Buffer packed(3 * kElements), past_key(kElements * kLength),
        past_value(kElements * kLength), query(kElements),
        key(kElements * (kLength + 1)), value(kElements * (kLength + 1));
    std::vector<uint16_t> x(3 * kElements), pk(kElements * kLength),
        pv(kElements * kLength);
    for (size_t i = 0; i < x.size(); ++i) {
      x[i] = static_cast<uint16_t>(i * 997 + 0x7c01);
    }
    for (size_t i = 0; i < pk.size(); ++i) {
      pk[i] = static_cast<uint16_t>(i * 719 + 0x8000);
      pv[i] = static_cast<uint16_t>(i * 113 + 0xfc00);
    }
    check(aclrtMemcpy(packed.ptr_,
                      packed.bytes_,
                      x.data(),
                      packed.bytes_,
                      ACL_MEMCPY_HOST_TO_DEVICE),
          "copy qkv");
    check(aclrtMemcpy(past_key.ptr_,
                      past_key.bytes_,
                      pk.data(),
                      past_key.bytes_,
                      ACL_MEMCPY_HOST_TO_DEVICE),
          "copy pk");
    check(aclrtMemcpy(past_value.ptr_,
                      past_value.bytes_,
                      pv.data(),
                      past_value.bytes_,
                      ACL_MEMCPY_HOST_TO_DEVICE),
          "copy pv");
    Tensor tx(packed, {kBatch, 1, 3 * kHeads * kDim}, type),
        tpk(past_key, {kBatch, kHeads, kLength, kDim}, type),
        tpv(past_value, {kBatch, kHeads, kLength, kDim}, type);
    Tensor tq(query, {kBatch, kHeads, 1, kDim}, type),
        tk(key, {kBatch, kHeads, kLength + 1, kDim}, type),
        tv(value, {kBatch, kHeads, kLength + 1, kDim}, type);
    if (negative) {
      // Five independent invalid metadata requests. Never submit a kernel.
      for (int32_t scenario = 0; scenario < 5; ++scenario) {
        auto shape = std::vector<int64_t>{kBatch, kHeads, 1, kDim};
        if (scenario == 0) {
          shape = {kBatch, kHeads, kDim};
        }
        if (scenario == 1) {
          shape[3] = kDim - 1;
        }
        if (scenario == 2) {
          shape[0] = kBatch - 1;
        }
        Tensor bad(query, shape, scenario == 3 ? ACL_INT16 : type);
        uint64_t ws = 0;
        aclOpExecutor* ex = nullptr;
        aclError code = aclnnT5SplitQkvAppendCacheGetWorkspaceSize(
            tx.ptr_,
            tpk.ptr_,
            tpv.ptr_,
            scenario == 4 ? kHeads + 1 : kHeads,
            kDim,
            bad.ptr_,
            tk.ptr_,
            tv.ptr_,
            &ws,
            &ex);
        if (code == ACL_SUCCESS) {
          throw std::runtime_error("invalid metadata accepted: " +
                                   std::to_string(scenario));
        }
        std::cout << "{\"metadata_scenario\":" << scenario
                  << ",\"status\":\"passed\",\"acl_status\":" << code << "}\n";
      }
      return 0;
    }
    uint64_t ws = 0;
    aclOpExecutor* executor = nullptr;
    check(aclnnT5SplitQkvAppendCacheGetWorkspaceSize(tx.ptr_,
                                                     tpk.ptr_,
                                                     tpv.ptr_,
                                                     kHeads,
                                                     kDim,
                                                     tq.ptr_,
                                                     tk.ptr_,
                                                     tv.ptr_,
                                                     &ws,
                                                     &executor),
          "GetWorkspaceSize");
    if (ws) {
      throw std::runtime_error("nonzero explicit workspace");
    }
    check(aclnnT5SplitQkvAppendCache(nullptr, 0, executor, runtime.stream_),
          "aclnn launch");
    check(aclrtSynchronizeStream(runtime.stream_), "synchronize");
    std::array<std::vector<uint16_t>, 3> result{
        std::vector<uint16_t>(kElements),
        std::vector<uint16_t>(kElements * (kLength + 1)),
        std::vector<uint16_t>(kElements * (kLength + 1))};
    std::array<Buffer*, 3> buffers{&query, &key, &value};
    for (int32_t i = 0; i < 3; ++i) {
      check(aclrtMemcpy(result[i].data(),
                        buffers[i]->bytes_,
                        buffers[i]->ptr_,
                        buffers[i]->bytes_,
                        ACL_MEMCPY_DEVICE_TO_HOST),
            "copy output");
    }
    for (int64_t b = 0; b < kBatch; ++b) {
      for (int64_t h = 0; h < kHeads; ++h) {
        for (int64_t d = 0; d < kDim; ++d) {
          const int64_t qpos = (b * kHeads + h) * kDim + d;
          if (result[0][qpos] != x[b * 3 * kHeads * kDim + h * kDim + d]) {
            throw std::runtime_error("query bits differ");
          }
          for (int64_t l = 0; l <= kLength; ++l) {
            const int64_t dst =
                ((b * kHeads + h) * (kLength + 1) + l) * kDim + d;
            for (int32_t s = 1; s <= 2; ++s) {
              const uint16_t expected =
                  l == kLength
                      ? x[b * 3 * kHeads * kDim + s * kHeads * kDim + h * kDim +
                          d]
                      : (s == 1
                             ? pk
                             : pv)[((b * kHeads + h) * kLength + l) * kDim + d];
              if (result[s][dst] != expected) {
                throw std::runtime_error("cache bits differ");
              }
            }
          }
        }
      }
    }
    std::cout << "{\"status\":\"passed\",\"workspace_bytes\":0,\"bit_"
                 "mismatches\":0}\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}

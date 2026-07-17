/* Copyright 2025-2026 The xLLM Authors.
Copyright 2024 The ScaleLLM Authors. All Rights Reserved.

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

#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "common.pb.h"
#include "nlohmann/json.hpp"
#include "util/slice.h"

namespace xllm {

class EngineType {
 public:
  enum Value : int8_t {
    LLM = 0,
    SSM = 1,
    VLM = 2,
    DIT = 3,
    REC = 4,
    INVALID = -1,
  };

  constexpr EngineType(Value v) : value_(v) {}
  EngineType(const std::string& str) {
    if (str == "LLM" || str == "llm") {
      value_ = LLM;
    } else if (str == "SSM" || str == "ssm") {
      value_ = SSM;
    } else if (str == "VLM" || str == "vlm") {
      value_ = VLM;
    } else if (str == "DIT" || str == "dit") {
      value_ = DIT;
    } else if (str == "REC" || str == "rec") {
      value_ = REC;
    } else {
      value_ = INVALID;
    }
  }

  EngineType() = delete;

  constexpr operator Value() const { return value_; }
  explicit operator bool() = delete;

  bool operator==(EngineType rhs) const { return value_ == rhs.value_; }
  bool operator!=(EngineType rhs) const { return value_ != rhs.value_; }
  bool operator==(Value rhs) const { return value_ == rhs; }
  bool operator!=(Value rhs) const { return value_ != rhs; }

  constexpr const char* to_string() const {
    if (this->value_ == LLM) {
      return "LLM";
    } else if (this->value_ == SSM) {
      return "SSM";
    } else if (this->value_ == VLM) {
      return "VLM";
    } else if (this->value_ == DIT) {
      return "DIT";
    } else if (this->value_ == REC) {
      return "REC";
    } else {
      return "INVALID";
    }
  }

 private:
  Value value_;
};

enum class StatusCode : uint8_t {
  OK = 0,
  // request was cancelled.
  CANCELLED = 1,
  // Unknown error.
  UNKNOWN = 2,
  // invalid argument.
  INVALID_ARGUMENT = 3,
  // timeout.
  DEADLINE_EXCEEDED = 4,
  // resource exhausted.
  RESOURCE_EXHAUSTED = 5,
  // service unavailable (e.g., model is sleeping).
  UNAVAILABLE = 6,
};

class Status final {
 public:
  Status() = default;

  Status(StatusCode code) : code_(code) {}

  Status(StatusCode code, std::string msg)
      : code_(code), msg_(std::move(msg)) {}

  StatusCode code() const { return code_; }

  const std::string& message() const { return msg_; }

  bool ok() const { return code_ == StatusCode::OK; }

 private:
  StatusCode code_ = StatusCode::OK;
  std::string msg_;
};

class InstanceRole {
 public:
  enum RoleType : int8_t {
    DEFAULT = 0,
    PREFILL = 1,
    DECODE = 2,
    MIX = 3,
    INVALID = -1,
  };

  InstanceRole() : role_(DEFAULT) {}
  InstanceRole(const std::string& role) {
    if (role == "DEFAULT") {
      role_ = DEFAULT;
    } else if (role == "PREFILL") {
      role_ = PREFILL;
    } else if (role == "DECODE") {
      role_ = DECODE;
    } else if (role == "MIX") {
      role_ = MIX;
    } else {
      role_ = INVALID;
    }
  }

  InstanceRole(RoleType role) : role_(role) {}
  operator RoleType() const { return role_; }
  explicit operator bool() const = delete;

  bool operator==(InstanceRole rhs) const { return role_ == rhs.role_; }
  bool operator!=(InstanceRole rhs) const { return role_ != rhs.role_; }
  bool operator==(RoleType role) const { return role_ == role; }
  bool operator!=(RoleType role) const { return role_ != role; }

  std::string to_string() const {
    if (this->role_ == DEFAULT) {
      return "DEFAULT";
    } else if (this->role_ == PREFILL) {
      return "PREFILL";
    } else if (this->role_ == DECODE) {
      return "DECODE";
    } else if (this->role_ == MIX) {
      return "MIX";
    } else {
      return "INVALID";
    }
  }

 private:
  RoleType role_;
};

struct Token {
  explicit Token(int64_t id) : id(id) {}

  int64_t id = 0;
  std::optional<float> logprob;
  Slice<int64_t> top_tokens;
  Slice<float> top_logprobs;
};

struct RemoteToken {
  int64_t token_id;
  std::optional<float> token_logprob;
  std::vector<int64_t> token_top_tokens;
  std::vector<float> token_top_logprobs;
};

struct RawToken {
  int64_t id = 0;
  std::optional<float> logprob;
  std::vector<int64_t> top_tokens;
  std::vector<float> top_logprobs;
  std::vector<float> embeddings;  // hidden states
};

struct RecItemInfo {
  // The decoded recommendation item id.
  int64_t item_id = 0;
  // The DID associated with the recommendation item.
  std::string did;
  // The business type associated with the recommendation item.
  std::string type;
};

// Weight segment info for P2P transfer (supports non-contiguous allocation)
// Forward declaration needed by InstanceInfo
struct WeightSegment {
  uint64_t offset;  // Offset from GlobalXTensor base address
  uint64_t size;    // Segment size in bytes
};

struct InstanceInfo {
  std::string name = "";
  std::string rpc_address = "";
  std::string incarnation_id = "";
  uint64_t register_ts_ms = 0;
  // DEFAULT/PREFILL/DECODE/MIX
  std::string type = "";
  // remote kv cache info
  std::vector<uint64_t> cluster_ids;
  std::vector<std::string> addrs;
  int32_t dp_size = 1;
  int32_t kv_split_size = 1;
  // transfer listen ports
  std::vector<uint16_t> ports;
  // ttft profiling data
  std::vector<std::pair<int32_t, double>> ttft_profiling_data;
  // tpot profiling data
  std::vector<std::tuple<int32_t, int32_t, double>> tpot_profiling_data;

  // XTensor mode: per-worker free physical pages
  std::vector<size_t> worker_free_phy_pages;
  // XTensor mode: total physical pages per worker (all workers have the same
  // total)
  size_t total_phy_pages = 0;
  // XTensor mode: model weight segments in GlobalXTensor (dp group 0 only)
  // key: model_id, value: list of {offset, size} segments
  std::unordered_map<std::string, std::vector<WeightSegment>>
      model_weight_segments;

  nlohmann::json serialize_to_json() const {
    nlohmann::json json_val;
    json_val["name"] = name;
    json_val["rpc_address"] = rpc_address;
    json_val["incarnation_id"] = incarnation_id;
    json_val["register_ts_ms"] = register_ts_ms;
    if (InstanceRole(type) == InstanceRole::DEFAULT) {
      json_val["type"] = 0;
    } else if (InstanceRole(type) == InstanceRole::PREFILL) {
      json_val["type"] = 1;
    } else if (InstanceRole(type) == InstanceRole::DECODE) {
      json_val["type"] = 2;
    } else if (InstanceRole(type) == InstanceRole::MIX) {
      json_val["type"] = 3;
    } else {
      LOG(ERROR) << "Unsupported instance type: " << type;
      return json_val;
    }
    json_val["cluster_ids"] = cluster_ids;
    json_val["addrs"] = addrs;
    json_val["dp_size"] = dp_size;
    json_val["kv_split_size"] = kv_split_size;
    json_val["ports"] = ports;
    json_val["ttft_profiling_data"] = ttft_profiling_data;
    json_val["tpot_profiling_data"] = tpot_profiling_data;
    // XTensor mode info
    json_val["worker_free_phy_pages"] = worker_free_phy_pages;
    json_val["total_phy_pages"] = total_phy_pages;
    // Serialize model_weight_segments: {model_id -> [{offset, size}, ...]}
    nlohmann::json segments_json;
    for (const auto& [model_id, segments] : model_weight_segments) {
      nlohmann::json seg_array = nlohmann::json::array();
      for (const auto& seg : segments) {
        seg_array.push_back({{"offset", seg.offset}, {"size", seg.size}});
      }
      segments_json[model_id] = seg_array;
    }
    json_val["model_weight_segments"] = segments_json;
    return json_val;
  }
};

// XTensor mode: per-layer offsets for KV cache transfer
struct XTensorLayerOffsets {
  std::vector<uint64_t> k_offsets;  // K cache offsets in GlobalXTensor
  std::vector<uint64_t> v_offsets;  // V cache offsets in GlobalXTensor
};

// Block mapping for one cache group. group_id is an opaque, stable identifier
// shared by the P/D cache layouts; transfer code does not interpret model
// semantics such as compression ratios.
struct KVBlockTransferGroup {
  int32_t group_id = 0;
  std::vector<uint64_t> local_blocks_ids;
  std::vector<uint64_t> remote_blocks_ids;
};

struct TransferKVInfo {
  std::string request_id;
  // Before batch build this may carry the full logical block table length as a
  // hint. BatchInputBuilder overwrites it with the current step local blocks.
  std::vector<uint64_t> local_blocks_ids;
  std::vector<uint64_t> remote_blocks_ids;
  std::vector<uint64_t> local_linear_state_ids;
  std::vector<uint64_t> remote_linear_state_ids;
  int32_t dp_rank;
  InstanceInfo remote_instance_info;

  // XTensor mode: destination offsets from D-node (per-layer)
  // Only populated when KVCacheConfig::enable_xtensor is true.
  std::vector<XTensorLayerOffsets> dst_xtensor_layer_offsets;

  // Group-aware transfer mappings, one entry per exported block manager.
  std::vector<KVBlockTransferGroup> block_transfer_groups;
};

// in bytes
struct DeviceStats {
  int64_t total_memory = 0;
  int64_t weights_memory = 0;
  int64_t total_kv_cache_memory = 0;
  int64_t total_activation_memory = 0;
  int64_t active_activation_memory = 0;
  // TODO: add more device stats
};

// Function call related types
struct JsonFunction {
  std::string name;
  std::string description;
  nlohmann::json parameters;

  JsonFunction() = default;
  JsonFunction(const std::string& func_name,
               const std::string& desc,
               const nlohmann::json& params)
      : name(func_name), description(desc), parameters(params) {}
};

struct JsonTool {
  std::string type;  // "function"
  JsonFunction function;

  JsonTool() : type("function") {}
  JsonTool(const std::string& tool_type, const JsonFunction& func)
      : type(tool_type), function(func) {}
};
// Experts update the required information
struct EplbInfo {
  // Target layer ID for new expert weight pre-loading (-1 = no pending load)
  // Values >=0 indicate the layer ID that should start loading new expert
  // weights
  int32_t prepare_layer_id = -1;
  // Expert IDs requiring updates, ordered by device shard assignment
  // Contains per-device expert indices for distributed weight updates
  std::vector<int32_t> expert_ids;
  // Layer ID ready for expert weight activation (-1 = no pending update)
  // Values >=0 indicate the layer ID whose pre-loaded weights are ready for
  // deployment
  int32_t update_layer_id = -1;
};

class MasterStatus {
 public:
  enum Value : int32_t {
    WAKEUP = 0,
    LIGHT_SLEEP = 1,
    DEEP_SLEEP = 2,
  };

  constexpr MasterStatus(Value v) : value_(v) {}

  explicit MasterStatus(const xllm::proto::MasterStatus& status)
      : value_(WAKEUP) {
    switch (status) {
      case xllm::proto::MasterStatus::WAKEUP:
        value_ = WAKEUP;
        break;
      case xllm::proto::MasterStatus::LIGHT_SLEEP:
        value_ = LIGHT_SLEEP;
        break;
      case xllm::proto::MasterStatus::DEEP_SLEEP:
        value_ = DEEP_SLEEP;
        break;
      default:
        LOG(FATAL) << "Unsupported master status: " << status;
        break;
    }
  }

  MasterStatus() = delete;

  constexpr operator Value() const { return value_; }
  explicit operator bool() = delete;

  bool operator==(MasterStatus rhs) const { return value_ == rhs.value_; }
  bool operator!=(MasterStatus rhs) const { return value_ != rhs.value_; }
  bool operator==(Value rhs) const { return value_ == rhs; }
  bool operator!=(Value rhs) const { return value_ != rhs; }

  xllm::proto::MasterStatus to_proto() const {
    switch (value_) {
      case WAKEUP:
        return xllm::proto::MasterStatus::WAKEUP;
      case LIGHT_SLEEP:
        return xllm::proto::MasterStatus::LIGHT_SLEEP;
      case DEEP_SLEEP:
        return xllm::proto::MasterStatus::DEEP_SLEEP;
      default:
        LOG(FATAL) << "Unsupported master status: " << value_;
        return xllm::proto::MasterStatus::WAKEUP;  // unreachable, suppress
                                                   // warning
    }
  }

 private:
  Value value_;
};

// Options for remote weight wakeup
struct WakeupOptions {
  MasterStatus master_status = MasterStatus::WAKEUP;
  std::vector<std::string> remote_addrs;
  // Each remote_addr has a list of weight segments to pull
  // Segments are ordered and should be concatenated at destination
  std::vector<std::vector<WeightSegment>> src_weight_segments;
};

inline constexpr int32_t REC_TOKEN_SIZE = 3;
using RecTokenTriple = std::array<int32_t, REC_TOKEN_SIZE>;
inline constexpr const char* LLM_REC_INPUT_TOKENS = "llm_rec_input_tokens";
inline constexpr const char* LLM_REC_INPUT_INDICES = "llm_rec_input_indices";
inline constexpr const char* LLM_REC_INPUT_EMBEDDING =
    "llm_rec_input_embedding";

}  // namespace xllm

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

#include "framework/kv_cache_transfer/pd_topology_guard.h"

#include <glog/logging.h>

#include <cstddef>
#include <limits>

namespace xllm {

namespace {

bool fail_topo(const std::string& msg, std::string* reason) {
  if (reason != nullptr) {
    *reason = msg;
  }
  return false;
}

PdTopoResult check_hetero_pd_req(const PdTopo& prefill_topo,
                                 const PdTopo& decode_topo,
                                 const std::string& kv_mode,
                                 bool enable_mla) {
  if (kv_mode != "PUSH") {
    return PdTopoResult{PdTopoStatus::DENY_HETERO,
                        "hetero pd requires kv_mode=PUSH"};
  }

  if (enable_mla) {
    return PdTopoResult{PdTopoStatus::ALLOW_HETERO, ""};
  }

  if (prefill_topo.dp_size != decode_topo.dp_size) {
    return PdTopoResult{PdTopoStatus::DENY_HETERO,
                        "non-mla hetero pd requires equal dp_size"};
  }

  if (prefill_topo.tp_size < decode_topo.tp_size ||
      prefill_topo.tp_size % decode_topo.tp_size != 0) {
    return PdTopoResult{
        PdTopoStatus::DENY_HETERO,
        "non-mla hetero pd requires prefill tp_size divisible by decode "
        "tp_size"};
  }
  return PdTopoResult{PdTopoStatus::ALLOW_HETERO, ""};
}

}  // namespace

bool try_get_pd_topo(const InstanceInfo& info,
                     PdTopo* topo,
                     std::string* reason) {
  if (topo == nullptr) {
    return fail_topo("topo must not be null", reason);
  }
  if (info.dp_size <= 0) {
    return fail_topo("dp_size must be greater than 0", reason);
  }

  const size_t cluster_num = info.cluster_ids.size();
  if (cluster_num == static_cast<size_t>(0)) {
    return fail_topo("cluster_ids must not be empty", reason);
  }

  const size_t dp_size = static_cast<size_t>(info.dp_size);
  if (cluster_num % dp_size != 0) {
    return fail_topo("cluster_ids.size() must be divisible by dp_size", reason);
  }
  if (cluster_num > static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
    return fail_topo("cluster_ids.size() exceeds int32_t range", reason);
  }

  topo->dp_size = info.dp_size;
  topo->tp_size = static_cast<int32_t>(cluster_num / dp_size);
  if (reason != nullptr) {
    reason->clear();
  }
  return true;
}

PdTopo get_pd_topo(const InstanceInfo& info) {
  PdTopo topo;
  std::string reason;
  CHECK(try_get_pd_topo(info, &topo, &reason)) << reason;
  return topo;
}

PdTopoResult check_pd_topo(const InstanceInfo& local,
                           const InstanceInfo& remote,
                           const std::string& kv_mode,
                           bool enable_mla) {
  PdTopo local_topo;
  std::string reason;
  if (!try_get_pd_topo(local, &local_topo, &reason)) {
    return PdTopoResult{PdTopoStatus::INVALID_LOCAL,
                        "invalid local pd topo: " + reason};
  }

  PdTopo remote_topo;
  if (!try_get_pd_topo(remote, &remote_topo, &reason)) {
    return PdTopoResult{PdTopoStatus::INVALID_REMOTE,
                        "invalid remote pd topo: " + reason};
  }

  const bool same_dp = local_topo.dp_size == remote_topo.dp_size;
  const bool same_tp = local_topo.tp_size == remote_topo.tp_size;
  if (same_dp && same_tp) {
    return PdTopoResult{PdTopoStatus::ALLOW_HOMO, ""};
  }

  return check_hetero_pd_req(local_topo, remote_topo, kv_mode, enable_mla);
}

}  // namespace xllm

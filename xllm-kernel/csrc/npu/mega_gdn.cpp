// Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.

#include <acl/acl.h>
#include <c10/util/Logging.h>
#include <torch/library.h>
#include <torch/types.h>

#include <array>
#include <cstdint>
#include <mutex>
#include <string_view>
#include <unordered_map>
#include <utility>

#include "csrc/npu/ascendc_api.h"
#include "op_plugin/utils/op_api_common_base.h"

// The runtime exports this ABI for device-local AIC/AIV handshakes.
extern "C" int32_t rtGetC2cCtrlAddr(uint64_t* address, uint32_t* length);

namespace xllm_kernel {
namespace {

constexpr const char* kPrefill = "aclnnXllmKernelMegaGdnPrefillOp";
constexpr const char* kDecode = "aclnnXllmKernelMegaGdnDecode";
constexpr const char* kMtp = "aclnnXllmKernelMegaGdnMtpDecode";
constexpr const char* kVarlen = "aclnnXllmKernelMegaGdnMtpDecodeVarlen";

const auto& mega_gdn_apis() {
  static const std::array apis{
      std::pair{
          kPrefill,
          load_ascendc_api(kPrefill,
                           "aclnnXllmKernelMegaGdnPrefillOpGetWorkspaceSize")},
      std::pair{kDecode,
                load_ascendc_api(
                    kDecode, "aclnnXllmKernelMegaGdnDecodeGetWorkspaceSize")},
      std::pair{kMtp,
                load_ascendc_api(
                    kMtp, "aclnnXllmKernelMegaGdnMtpDecodeGetWorkspaceSize")},
      std::pair{kVarlen,
                find_ascendc_api(
                    kVarlen,
                    "aclnnXllmKernelMegaGdnMtpDecodeVarlenGetWorkspaceSize")},
  };
  return apis;
}

void GetApiFunc(const char* name,
                const char* /*workspace_name*/,
                void*& launch,
                void*& workspace) {
  for (const auto& [api_name, api] : mega_gdn_apis()) {
    if (std::string_view(name) == api_name) {
      CHECK(api.launch && api.workspace)
          << name << " is unavailable in xllm-kernel OPP";
      launch = api.launch;
      workspace = api.workspace;
      return;
    }
  }
  LOG(FATAL) << "Unknown MegaGDN ACLNN entry point: " << name;
}

bool prepare_ascendc_mega_gdn() {
  const auto& varlen = mega_gdn_apis().back().second;
  return varlen.launch && varlen.workspace;
}

int64_t get_ffts_addr(const torch::Device& device) {
  static std::unordered_map<int32_t, int64_t> addresses;
  static std::mutex mutex;
  std::lock_guard<std::mutex> lock(mutex);
  const int32_t device_index = static_cast<int32_t>(device.index());
  const auto entry = addresses.find(device_index);
  if (entry != addresses.end()) {
    return entry->second;
  }

  // A2/A3 use a device-local FFTS address. Ascend950 does not use FFTS.
  uint64_t address = 0;
  const char* soc = aclrtGetSocName();
  CHECK(soc);
  if (std::string_view(soc).find("Ascend950") != 0) {
    uint32_t length = 0;
    CHECK((rtGetC2cCtrlAddr(&address, &length)) == 0);
    CHECK(address && length);
  }
  const int64_t result = static_cast<int64_t>(address);
  addresses.emplace(device_index, result);
  return result;
}

torch::Tensor mega_gdn_prefill(const torch::Tensor& mixed_qkv,
                               const torch::Tensor& b,
                               const torch::Tensor& a,
                               const torch::Tensor& z,
                               const torch::Tensor& conv_weight,
                               torch::Tensor& conv_state,
                               const torch::Tensor& a_log,
                               const torch::Tensor& dt_bias,
                               const torch::Tensor& conv_state_read_indices,
                               const torch::Tensor& conv_state_write_indices,
                               const torch::Tensor& ssm_state_read_indices,
                               const torch::Tensor& ssm_state_write_indices,
                               torch::Tensor& ssm_cache,
                               const torch::Tensor& mask_lower,
                               const torch::Tensor& mask_full,
                               const torch::Tensor& minus_identity,
                               const torch::Tensor& cu_seqlens,
                               const torch::Tensor& norm_weight,
                               int64_t num_matrices) {
  const torch::DeviceGuard guard(mixed_qkv.device());

  const int64_t total_tokens = z.size(0);
  const int64_t num_value_heads = z.size(1);
  const int64_t head_dim = z.size(2);

  auto norm_output =
      torch::empty({total_tokens, num_value_heads, head_dim},
                   torch::TensorOptions(z.device()).dtype(torch::kBFloat16));
  torch::Tensor& conv_state_out = conv_state;
  torch::Tensor& ssm_cache_out = ssm_cache;

  const int64_t ffts_addr = get_ffts_addr(mixed_qkv.device());

  EXEC_NPU_CMD_EXT(aclnnXllmKernelMegaGdnPrefillOp,
                   mixed_qkv,
                   b,
                   a,
                   z,
                   conv_weight,
                   conv_state,
                   a_log,
                   dt_bias,
                   conv_state_read_indices,
                   conv_state_write_indices,
                   ssm_state_read_indices,
                   ssm_state_write_indices,
                   ssm_cache,
                   mask_lower,
                   mask_full,
                   minus_identity,
                   cu_seqlens,
                   norm_weight,
                   ffts_addr,
                   num_matrices,
                   norm_output,
                   conv_state_out,
                   ssm_cache_out);

  return norm_output;
}

torch::Tensor mega_gdn_decode(const torch::Tensor& qkv,
                              const torch::Tensor& z,
                              const torch::Tensor& b,
                              const torch::Tensor& a,
                              const torch::Tensor& conv_weight,
                              torch::Tensor& conv_state,
                              const torch::Tensor& a_log,
                              const torch::Tensor& dt_bias,
                              torch::Tensor& ssm_state,
                              const torch::Tensor& read_state_indices,
                              const torch::Tensor& write_state_indices,
                              const torch::Tensor& norm_weight,
                              bool fla_ssm_state_layout) {
  const torch::DeviceGuard guard(qkv.device());
  torch::Tensor conv_out = torch::empty_like(qkv);
  torch::Tensor out = torch::empty_like(z);
  torch::Tensor& conv_state_out = conv_state;
  torch::Tensor& ssm_state_out = ssm_state;

  EXEC_NPU_CMD_EXT(aclnnXllmKernelMegaGdnDecode,
                   qkv,
                   z,
                   b,
                   a,
                   conv_weight,
                   conv_state,
                   a_log,
                   dt_bias,
                   ssm_state,
                   read_state_indices,
                   write_state_indices,
                   norm_weight,
                   fla_ssm_state_layout,
                   conv_out,
                   conv_state_out,
                   ssm_state_out,
                   out);

  return out;
}

torch::Tensor mega_gdn_mtp_decode(const torch::Tensor& qkv,
                                  const torch::Tensor& z,
                                  const torch::Tensor& b,
                                  const torch::Tensor& a,
                                  const torch::Tensor& conv_weight,
                                  torch::Tensor& conv_state,
                                  const torch::Tensor& a_log,
                                  const torch::Tensor& dt_bias,
                                  torch::Tensor& ssm_state,
                                  const torch::Tensor& read_state_indices,
                                  const torch::Tensor& write_state_indices,
                                  const torch::Tensor& num_accepted_tokens,
                                  const torch::Tensor& norm_weight,
                                  bool fla_ssm_state_layout) {
  const torch::DeviceGuard guard(qkv.device());
  torch::Tensor conv_out = torch::empty_like(qkv);
  torch::Tensor out = torch::empty_like(z);
  torch::Tensor& conv_state_out = conv_state;
  torch::Tensor& ssm_state_out = ssm_state;

  EXEC_NPU_CMD_EXT(aclnnXllmKernelMegaGdnMtpDecode,
                   qkv,
                   z,
                   b,
                   a,
                   conv_weight,
                   conv_state,
                   a_log,
                   dt_bias,
                   ssm_state,
                   read_state_indices,
                   write_state_indices,
                   num_accepted_tokens,
                   norm_weight,
                   fla_ssm_state_layout,
                   conv_out,
                   conv_state_out,
                   ssm_state_out,
                   out);

  return out;
}

void mega_gdn_mtp_decode_varlen(const torch::Tensor& qkv,
                                const torch::Tensor& z,
                                const torch::Tensor& b,
                                const torch::Tensor& a,
                                const torch::Tensor& conv_weight,
                                torch::Tensor& conv_state,
                                const torch::Tensor& a_log,
                                const torch::Tensor& dt_bias,
                                torch::Tensor& ssm_state,
                                const torch::Tensor& state_indices,
                                const torch::Tensor& num_accepted_tokens,
                                const torch::Tensor& norm_weight,
                                const torch::Tensor& cu_seqlens,
                                torch::Tensor& out,
                                bool fla_ssm_state_layout) {
  // Different request chunks write disjoint packed rows of the same
  // zero-initialized output. Untouched capacity rows must remain zero.
  const torch::DeviceGuard guard(qkv.device());
  torch::Tensor conv_out = torch::empty_like(qkv);
  EXEC_NPU_CMD_EXT(aclnnXllmKernelMegaGdnMtpDecodeVarlen,
                   qkv,
                   z,
                   b,
                   a,
                   conv_weight,
                   conv_state,
                   a_log,
                   dt_bias,
                   ssm_state,
                   state_indices,
                   num_accepted_tokens,
                   norm_weight,
                   cu_seqlens,
                   fla_ssm_state_layout,
                   conv_out,
                   conv_state,
                   ssm_state,
                   out);
}

TORCH_LIBRARY_FRAGMENT(xllm_kernel, m) {
  m.def("prepare_ascendc_mega_gdn() -> bool",
        TORCH_FN(prepare_ascendc_mega_gdn));
  m.def(
      "mega_gdn_prefill(Tensor mixed_qkv, Tensor b, Tensor a, Tensor z, "
      "Tensor conv_weight, Tensor(a!) conv_state, Tensor a_log, Tensor "
      "dt_bias, Tensor conv_state_read_indices, Tensor "
      "conv_state_write_indices, Tensor ssm_state_read_indices, Tensor "
      "ssm_state_write_indices, Tensor(b!) ssm_cache, Tensor mask_lower, "
      "Tensor mask_full, Tensor minus_identity, Tensor cu_seqlens, "
      "Tensor norm_weight, int num_matrices) -> Tensor");
  m.def(
      "mega_gdn_decode(Tensor qkv, Tensor z, Tensor b, Tensor a, Tensor "
      "conv_weight, Tensor(a!) conv_state, Tensor a_log, Tensor dt_bias, "
      "Tensor(b!) ssm_state, Tensor read_state_indices, Tensor "
      "write_state_indices, Tensor norm_weight, bool fla_ssm_state_layout) "
      "-> Tensor");
  m.def(
      "mega_gdn_mtp_decode(Tensor qkv, Tensor z, Tensor b, Tensor a, Tensor "
      "conv_weight, Tensor(a!) conv_state, Tensor a_log, Tensor dt_bias, "
      "Tensor(b!) ssm_state, Tensor read_state_indices, Tensor "
      "write_state_indices, Tensor num_accepted_tokens, Tensor norm_weight, "
      "bool fla_ssm_state_layout) -> Tensor");
  m.def(
      "mega_gdn_mtp_decode_varlen(Tensor qkv, Tensor z, Tensor b, Tensor a, "
      "Tensor conv_weight, Tensor(a!) conv_state, Tensor a_log, Tensor "
      "dt_bias, Tensor(b!) ssm_state, Tensor state_indices, "
      "Tensor num_accepted_tokens, Tensor norm_weight, Tensor cu_seqlens, "
      "Tensor(c!) out, bool fla_ssm_state_layout) -> ()");
}

TORCH_LIBRARY_IMPL(xllm_kernel, PrivateUse1, m) {
  m.impl("mega_gdn_prefill", TORCH_FN(mega_gdn_prefill));
  m.impl("mega_gdn_decode", TORCH_FN(mega_gdn_decode));
  m.impl("mega_gdn_mtp_decode", TORCH_FN(mega_gdn_mtp_decode));
  m.impl("mega_gdn_mtp_decode_varlen", TORCH_FN(mega_gdn_mtp_decode_varlen));
}

}  // namespace
}  // namespace xllm_kernel

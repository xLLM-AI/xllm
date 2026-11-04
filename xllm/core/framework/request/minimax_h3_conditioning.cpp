/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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

#include "core/framework/request/minimax_h3_conditioning.h"

#include <glog/logging.h>
#include <openssl/sha.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "tensor.pb.h"

namespace xllm {
namespace {

constexpr std::string_view kPrefix = "minimax_h3.";
constexpr int64_t kHiddenSize = 5120;
constexpr int64_t kMaxTokens = 60160;
constexpr std::array<std::string_view, 10> kMetadataKeys = {
    "minimax_h3.schema",
    "minimax_h3.cache_abi_version",
    "minimax_h3.source_backend",
    "minimax_h3.decoder_layer_index",
    "minimax_h3.hidden_state_slot",
    "minimax_h3.token_count",
    "minimax_h3.hidden_digest",
    "minimax_h3.token_tags_digest",
    "minimax_h3.condition_cache_key",
    "minimax_h3.reference_pixels_digest"};

struct ConditionMetadata {
  int64_t token_count = 0;
  const std::vector<int64_t>* token_tags = nullptr;
  const std::string* hidden_digest = nullptr;
  const std::string* token_tags_digest = nullptr;
  const std::string* reference_pixels_digest = nullptr;
};

Status invalid(std::string detail) {
  return Status(StatusCode::INVALID_ARGUMENT,
                "invalid MiniMax-H3 conditioning: " + detail);
}

bool is_digest(const std::string* value) {
  return value != nullptr && value->size() == SHA256_DIGEST_LENGTH * 2 &&
         std::all_of(value->begin(), value->end(), [](char ch) {
           return (ch >= '0' && ch <= '9') || (ch >= 'a' && ch <= 'f');
         });
}

Status parse_metadata(const TensorParameters& parameters,
                      ConditionMetadata& metadata) {
  for (const auto& [name, value] : parameters) {
    if (name.starts_with(kPrefix) &&
        std::find(kMetadataKeys.begin(), kMetadataKeys.end(), name) ==
            kMetadataKeys.end()) {
      return invalid("unknown metadata key: " + name);
    }
  }
  const std::string* schema =
      get_tensor_parameter<std::string>(parameters, "minimax_h3.schema");
  const int64_t* abi =
      get_tensor_parameter<int64_t>(parameters, "minimax_h3.cache_abi_version");
  const std::string* backend = get_tensor_parameter<std::string>(
      parameters, "minimax_h3.source_backend");
  const int64_t* layer = get_tensor_parameter<int64_t>(
      parameters, "minimax_h3.decoder_layer_index");
  const int64_t* slot =
      get_tensor_parameter<int64_t>(parameters, "minimax_h3.hidden_state_slot");
  const int64_t* count =
      get_tensor_parameter<int64_t>(parameters, "minimax_h3.token_count");
  if (schema == nullptr || *schema != "xllm.minimax_h3.text_conditioning/v1" ||
      abi == nullptr || *abi != 2 || backend == nullptr ||
      (*backend != "official_hf" && *backend != "xllm_native") ||
      layer == nullptr || *layer != 49 || slot == nullptr || *slot != 50) {
    return invalid("missing or incompatible capture metadata");
  }
  if (count == nullptr || *count < 1 || *count > kMaxTokens) {
    return invalid("token_count must be in [1,60160]");
  }
  metadata.token_count = *count;
  metadata.token_tags = get_tensor_parameter<std::vector<int64_t>>(
      parameters, "prompt_token_tags");
  if (metadata.token_tags == nullptr ||
      metadata.token_tags->size() != static_cast<size_t>(*count) ||
      !std::all_of(metadata.token_tags->begin(),
                   metadata.token_tags->end(),
                   [](int64_t tag) { return tag == 0 || tag == 1; })) {
    return invalid("prompt_token_tags must contain N values in {0,1}");
  }
  metadata.hidden_digest =
      get_tensor_parameter<std::string>(parameters, "minimax_h3.hidden_digest");
  metadata.token_tags_digest = get_tensor_parameter<std::string>(
      parameters, "minimax_h3.token_tags_digest");
  metadata.reference_pixels_digest = get_tensor_parameter<std::string>(
      parameters, "minimax_h3.reference_pixels_digest");
  const std::string* cache_key = get_tensor_parameter<std::string>(
      parameters, "minimax_h3.condition_cache_key");
  if (!is_digest(metadata.hidden_digest) ||
      !is_digest(metadata.token_tags_digest) ||
      !is_digest(metadata.reference_pixels_digest) || !is_digest(cache_key)) {
    return invalid("digests and condition_cache_key must be lowercase SHA256");
  }
  return Status();
}

int32_t parameter_value_count(const proto::Parameter& parameter) {
  return static_cast<int32_t>(parameter.has_bool_param()) +
         static_cast<int32_t>(parameter.has_int64_param()) +
         static_cast<int32_t>(parameter.has_string_param()) +
         static_cast<int32_t>(parameter.has_double_param()) +
         static_cast<int32_t>(parameter.has_uint64_param()) +
         static_cast<int32_t>(parameter.int64_tensor_param_size() > 0);
}

bool has_nonbyte_contents(const proto::TensorContents& contents) {
  return contents.bool_contents_size() != 0 ||
         contents.int_contents_size() != 0 ||
         contents.int64_contents_size() != 0 ||
         contents.uint_contents_size() != 0 ||
         contents.uint64_contents_size() != 0 ||
         contents.fp32_contents_size() != 0 ||
         contents.fp64_contents_size() != 0 ||
         contents.string_contents_size() != 0;
}

std::string bytes_sha256(const void* bytes, size_t size) {
  std::array<uint8_t, SHA256_DIGEST_LENGTH> digest{};
  SHA256(static_cast<const unsigned char*>(bytes), size, digest.data());
  constexpr std::string_view kHex = "0123456789abcdef";
  std::string result(SHA256_DIGEST_LENGTH * 2, '0');
  for (size_t index = 0; index < digest.size(); ++index) {
    result[index * 2] = kHex[digest[index] >> 4];
    result[index * 2 + 1] = kHex[digest[index] & 0xf];
  }
  return result;
}

}  // namespace

bool has_minimax_h3_condition_metadata(const proto::Tensor& tensor) {
  return std::any_of(
      tensor.parameters().begin(),
      tensor.parameters().end(),
      [](const auto& item) { return item.first.starts_with(kPrefix); });
}

bool has_minimax_h3_condition_metadata(const TensorParameters& parameters) {
  return std::any_of(
      parameters.begin(), parameters.end(), [](const auto& item) {
        return item.first.starts_with(kPrefix);
      });
}

Status validate_minimax_h3_condition_wire(const proto::Tensor& tensor,
                                          const BinaryPayload& payload) {
  const auto parameters = proto_to_tensor_parameters(tensor);
  if (!parameters.has_value()) {
    return invalid("invalid tensor parameters");
  }
  ConditionMetadata metadata;
  const Status metadata_status = parse_metadata(*parameters, metadata);
  if (!metadata_status.ok()) {
    return metadata_status;
  }
  if (tensor.name() != "prompt_embed" || tensor.datatype() != "BF16" ||
      tensor.shape_size() != 2 || tensor.shape(0) != metadata.token_count ||
      tensor.shape(1) != kHiddenSize) {
    return invalid(
        "hidden must be named prompt_embed and have BF16 [N,5120] shape");
  }
  const uint64_t byte_count =
      static_cast<uint64_t>(metadata.token_count) * kHiddenSize * 2;
  const auto& transport = tensor.parameters();
  const auto binary = transport.find("is_binary");
  bool is_binary = false;
  if (binary != transport.end()) {
    if (!binary->second.has_bool_param() ||
        parameter_value_count(binary->second) != 1) {
      return invalid("is_binary must contain only a bool value");
    }
    is_binary = binary->second.bool_param();
  }
  const auto offset = transport.find("offset");
  const auto length = transport.find("len");
  if (has_nonbyte_contents(tensor.contents())) {
    return invalid("BF16 contents must use bytes_contents only");
  }
  if (!is_binary) {
    if (offset != transport.end() || length != transport.end() ||
        tensor.contents().bytes_contents().size() != byte_count) {
      return invalid(
          "inline BF16 byte length mismatch or stray binary offset/len");
    }
    return Status();
  }
  if (offset == transport.end() || length == transport.end() ||
      !offset->second.has_int64_param() || !length->second.has_int64_param() ||
      parameter_value_count(offset->second) != 1 ||
      parameter_value_count(length->second) != 1 ||
      offset->second.int64_param() < 0 || length->second.int64_param() < 0 ||
      static_cast<uint64_t>(length->second.int64_param()) != byte_count ||
      !tensor.contents().bytes_contents().empty()) {
    return invalid("binary BF16 requires INT64 offset/len and empty contents");
  }
  const uint64_t byte_offset =
      static_cast<uint64_t>(offset->second.int64_param());
  if (byte_offset > payload.size() ||
      byte_count > payload.size() - byte_offset) {
    return invalid("binary BF16 range exceeds payload");
  }
  return Status();
}

std::string minimax_h3_tensor_sha256(const torch::Tensor& tensor) {
  CHECK(tensor.defined() && tensor.device().is_cpu() && tensor.is_contiguous());
  return bytes_sha256(
      tensor.const_data_ptr(),
      static_cast<size_t>(tensor.numel()) * tensor.element_size());
}

Status validate_minimax_h3_condition(const NamedTensor& source,
                                     const DiTMediaSources& media_sources) {
  ConditionMetadata metadata;
  const Status metadata_status = parse_metadata(source.parameters, metadata);
  if (!metadata_status.ok()) {
    return metadata_status;
  }
  const torch::Tensor& hidden = source.tensor;
  if (source.name != "prompt_embed" || !hidden.defined() ||
      !hidden.device().is_cpu() || hidden.scalar_type() != torch::kBFloat16 ||
      !hidden.is_contiguous() || hidden.dim() != 2 ||
      hidden.size(0) != metadata.token_count || hidden.size(1) != kHiddenSize ||
      !torch::isfinite(hidden).all().item<bool>()) {
    return invalid(
        "decoded hidden must be finite contiguous CPU BF16 [N,5120]");
  }
  if (minimax_h3_tensor_sha256(hidden) != *metadata.hidden_digest) {
    return invalid("hidden_digest mismatch");
  }
  // Tags are 0/1; encode explicitly as little-endian INT64 for the Oracle ABI.
  std::vector<uint8_t> tag_bytes(metadata.token_tags->size() * sizeof(int64_t),
                                 0);
  for (size_t index = 0; index < metadata.token_tags->size(); ++index) {
    tag_bytes[index * sizeof(int64_t)] =
        static_cast<uint8_t>((*metadata.token_tags)[index]);
  }
  if (bytes_sha256(tag_bytes.data(), tag_bytes.size()) !=
      *metadata.token_tags_digest) {
    return invalid("token_tags_digest mismatch");
  }
  if (media_sources.size() != 1) {
    return invalid("exactly one reference image is required");
  }
  const MediaNamedTensor& reference = media_sources.at(0);
  const torch::Tensor& pixels = reference.tensor;
  if (reference.name != "image" || reference.modality != "image" ||
      !pixels.defined() || !pixels.device().is_cpu() ||
      pixels.scalar_type() != torch::kUInt8 || pixels.dim() != 3 ||
      pixels.size(0) != 3 || pixels.size(1) <= 0 || pixels.size(2) <= 0) {
    return invalid(
        "reference must be named image and have CPU UINT8 [3,H,W] shape");
  }
  if (minimax_h3_tensor_sha256(pixels.contiguous()) !=
      *metadata.reference_pixels_digest) {
    return invalid("reference_pixels_digest mismatch");
  }
  return Status();
}

}  // namespace xllm

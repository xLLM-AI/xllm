/* Copyright 2025-2026 The xLLM Authors.

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

#include "embedding_output_builder.h"

namespace xllm {

TensorProtoBuilder::TensorProtoBuilder(bool use_binary_encoding)
    : use_binary_encoding_(use_binary_encoding) {};

bool TensorProtoBuilder::build_repeated_tensor(
    const std::vector<torch::Tensor>& in_tensors,
    google::protobuf::RepeatedPtrField<xllm::proto::Tensor>& out_tensors,
    std::string& binary_payload) {
  for (const auto& in_tensor : in_tensors) {
    CHECK(in_tensor.is_contiguous())
        << "Internal Error: only support contiguous mm_embedding";

    xllm::proto::Tensor* out_tensor = out_tensors.Add();

    if (!build_tensor(in_tensor, *out_tensor, binary_payload)) {
      return false;
    }
  }
  return true;
}

bool TensorProtoBuilder::build_tensor(const torch::Tensor& in_tensor,
                                      xllm::proto::Tensor& out_tensor,
                                      std::string& binary_payload) {
  if (use_binary_encoding_) {
    // build tensor in binary format
    out_tensor.set_datatype(
        util::torch_datatype_to_proto(in_tensor.scalar_type()));

    for (auto dim : in_tensor.sizes()) {
      out_tensor.add_shape(static_cast<int32_t>(dim));
    }

    auto numel = in_tensor.numel();
    auto byte_len = numel * in_tensor.element_size();
    size_t offset = binary_payload.size();
    auto* params = out_tensor.mutable_parameters();
    (*params)["offset"].set_int64_param(offset);
    (*params)["len"].set_int64_param(byte_len);
    (*params)["is_binary"].set_bool_param(true);
    binary_payload.append(reinterpret_cast<const char*>(in_tensor.data_ptr()),
                          byte_len);
  } else {
    // build tensor in json format
    util::torch_to_proto(in_tensor, &out_tensor);
  }
  return true;
}

bool TensorProtoBuilder::build_tensor(const xllm::proto::Tensor& in_tensor,
                                      const std::string& binary_payload,
                                      torch::Tensor& out_tensor) {
  const auto& params = in_tensor.parameters();
  auto it = params.find("is_binary");
  bool is_binary = (it != params.end() && it->second.bool_param());

  if (!is_binary) {
    out_tensor = util::proto_to_torch(in_tensor);
    return true;
  }

  auto offset_it = params.find("offset");
  auto len_it = params.find("len");

  if (offset_it == params.end() || len_it == params.end()) {
    return false;
  }

  const int64_t offset = offset_it->second.int64_param();
  const int64_t byte_len = len_it->second.int64_param();

  if (offset < 0 || byte_len <= 0 ||
      static_cast<size_t>(offset + byte_len) > binary_payload.size()) {
    return false;
  }

  auto dtype = util::datatype_proto_to_torch(in_tensor.datatype());

  std::vector<int64_t> sizes;
  sizes.reserve(in_tensor.shape_size());
  for (auto dim : in_tensor.shape()) {
    sizes.push_back(dim);
  }

  const char* src = binary_payload.data() + offset;

  out_tensor = torch::from_blob(
      const_cast<char*>(src), sizes, torch::TensorOptions().dtype(dtype));

  out_tensor = out_tensor.clone();

  return true;
}

EmbeddingOutputBuilder::EmbeddingOutputBuilder(
    bool embedding_use_binary_encoding,
    bool metadata_use_binary_encoding)
    : embedding_use_binary_encoding_(embedding_use_binary_encoding),
      metadata_use_binary_encoding_(metadata_use_binary_encoding) {};

EmbeddingOutputBuilder::~EmbeddingOutputBuilder() {};

bool EmbeddingOutputBuilder::build_repeated_embedding_output(
    const std::vector<EmbeddingOutput>& in_embeddings,
    google::protobuf::RepeatedPtrField<xllm::proto::Embedding>& out_embeddings,
    std::string& binary_payload) {
  for (const auto& in_embedding : in_embeddings) {
    xllm::proto::Embedding* out_embedding = out_embeddings.Add();
    if (!build_embedding_output(in_embedding, *out_embedding, binary_payload)) {
      return false;
    }
  }
  return true;
}

bool EmbeddingOutputBuilder::build_embedding_output(
    const EmbeddingOutput& in_embedding,
    xllm::proto::Embedding& out_embedding,
    std::string& binary_payload) {
  TensorProtoBuilder embedding_output_builder(embedding_use_binary_encoding_);
  embedding_output_builder.build_tensor(in_embedding.embedding,
                                        *out_embedding.mutable_embedding(),
                                        binary_payload);

  auto* meta_map = out_embedding.mutable_metadata();
  TensorProtoBuilder meta_output_builder(metadata_use_binary_encoding_);
  for (const auto& [key, value] : in_embedding.metadata) {
    xllm::proto::Tensor metadata_tensor;
    meta_output_builder.build_tensor(
        in_embedding.metadata.at(key), metadata_tensor, binary_payload);
    (*meta_map)[key] = std::move(metadata_tensor);
  }
  return true;
};

bool EmbeddingOutputBuilder::build_embedding_output(
    const xllm::proto::Embedding& in_embedding,
    std::string& binary_payload,
    EmbeddingOutput& out_embedding) {
  TensorProtoBuilder embedding_tensor_builder(embedding_use_binary_encoding_);
  if (!embedding_tensor_builder.build_tensor(
          in_embedding.embedding(), binary_payload, out_embedding.embedding)) {
    return false;
  }

  out_embedding.metadata.clear();

  TensorProtoBuilder meta_builder(metadata_use_binary_encoding_);
  for (const auto& [key, proto_tensor] : in_embedding.metadata()) {
    torch::Tensor tensor;

    if (!meta_builder.build_tensor(proto_tensor, binary_payload, tensor)) {
      return false;
    }

    out_embedding.metadata.emplace(key, std::move(tensor));
  }

  return true;
}
};  // namespace xllm

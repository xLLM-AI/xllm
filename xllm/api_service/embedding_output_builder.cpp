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
    return util::torch_to_proto(in_tensor, &out_tensor, binary_payload);
  }
  return util::torch_to_proto(in_tensor, &out_tensor);
}

bool TensorProtoBuilder::build_tensor(const xllm::proto::Tensor& in_tensor,
                                      const std::string& binary_payload,
                                      torch::Tensor& out_tensor) {
  out_tensor = util::proto_to_torch(in_tensor, binary_payload);
  return out_tensor.defined();
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

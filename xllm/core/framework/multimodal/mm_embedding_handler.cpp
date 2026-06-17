/* Copyright 2025 The xLLM Authors. All Rights Reserved.

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

#include "mm_embedding_handler.h"

#include "core/framework/multimodal/embedding_output.h"
#include "core/util/utils.h"
#include "mm_input.h"

namespace xllm {

namespace {

bool parse_tensor(const xllm::proto::Tensor& in_tensor,
                  MMPayload& payload,
                  torch::Tensor& out_tensor) {
  const auto& params = in_tensor.parameters();
  auto it = params.find("is_binary");
  bool is_binary = (it != params.end() && it->second.bool_param());

  if (!is_binary) {
    out_tensor = util::proto_to_torch(in_tensor);
    return true;
  }

  auto len_it = params.find("len");

  if (len_it == params.end()) {
    return false;
  }

  const int64_t offset = 0;
  const int64_t byte_len = len_it->second.int64_param();

  std::string binary_payload;
  if (!payload.get(binary_payload, byte_len)) {
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

bool parse_embedding_output(const xllm::proto::Embedding& in_embedding_output,
                            MMPayload& payload,
                            EmbeddingOutput& out_embedding_output) {
  if (!parse_tensor(in_embedding_output.embedding(),
                    payload,
                    out_embedding_output.embedding)) {
    return false;
  }

  out_embedding_output.metadata.clear();

  for (const auto& [key, proto_tensor] : in_embedding_output.metadata()) {
    torch::Tensor tensor;

    if (!parse_tensor(proto_tensor, payload, tensor)) {
      return false;
    }

    out_embedding_output.metadata.emplace(key, std::move(tensor));
  }

  return true;
}

}  // namespace

MMEmbeddingHandler::MMEmbeddingHandler(MMType::Value mm_type)
    : mm_type_(mm_type) {};

MMErrCode MMEmbeddingHandler::load(const MMContent& content,
                                   MMInputItem& input,
                                   MMPayload& payload) {
  input.type = mm_type_;
  if (!parse_embedding_output(content.embedding, payload, input.embedding)) {
    LOG(ERROR) << "parse embedding failed";
    return MMErrCode::PARSE_EMB_ERR;
  }

  return MMErrCode::SUCCESS;
}

MMErrCode MMEmbeddingHandler::decode(MMInputItem& input) {
  return MMErrCode::SUCCESS;
}

}  // namespace xllm
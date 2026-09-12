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

#pragma once

#include <google/protobuf/message.h>
#include <google/protobuf/struct.pb.h>

#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "api_service/chat_struct_spans.h"
#include "core/common/types.h"

namespace xllm {

// Decoding of an OpenAI chat body into ChatRequest / MMChatRequest.
//
// protobuf's own JSON parser (JsonStringToMessage) is the reference for what a
// chat body means, but it costs ~12us of fixed overhead plus ~3us per message,
// five to eight times what brpc's json2pb needs for the same body. json2pb
// cannot be used directly because two members of the schema are
// google.protobuf.Struct -- chat_template_kwargs and
// tools[].function.parameters
// -- a well-known type it does not map canonically.
//
// The decoder therefore parses with json2pb after carving those members out:
// it locates their bytes with a structural scan, converts them to Struct with
// brpc's bundled rapidjson, hands json2pb a copy of the body with the members
// blanked to null, and attaches the Structs afterwards.
//
// The reference parser stays authoritative. The fast path is taken only for a
// body in canonical shape, and the decoder falls back to JsonStringToMessage on
// the original bytes whenever the scan cannot account for the body, rapidjson
// refuses a Struct member, json2pb rejects the body, or json2pb had to drop a
// mismatched optional scalar that protobuf's parser would have coerced or
// rejected. Every rejection a client sees is therefore still produced by the
// reference parser.
//
// Two differences remain for bodies both parsers accept and are deliberately
// not guarded: json2pb keeps the first of duplicate keys where protobuf's
// parser keeps the last, and nested lowerCamelCase field aliases
// (protobuf's json_name) are ignored by json2pb. Root-level camelCase keys are
// detected and sent down the fallback. The completion endpoint has parsed with
// json2pb, and lived with both, all along.

// Struct members decoded alongside the request, for the caller to attach.
struct ChatStructMembers {
  std::optional<google::protobuf::Struct> chat_template_kwargs;
  std::vector<std::pair<int32_t, google::protobuf::Struct>> tool_parameters;
};

// Fills `request` from `body`. On the fast path json2pb parses the body with
// its Struct members blanked and the decoded members are returned through
// `members` for the caller to attach; on the fallback path the reference parser
// fills the request completely and `members` is left empty.
Status decode_chat_request_core(std::string body,
                                google::protobuf::Message* request,
                                ChatStructMembers* members);

// Decodes `body` into a ChatRequest or MMChatRequest.
template <typename RequestT>
Status decode_chat_request(std::string body, RequestT* request) {
  ChatStructMembers members;
  const Status status =
      decode_chat_request_core(std::move(body), request, &members);
  if (!status.ok()) {
    return status;
  }
  if (members.chat_template_kwargs.has_value()) {
    request->mutable_chat_template_kwargs()->Swap(
        &members.chat_template_kwargs.value());
  }
  for (auto& [index, parameters] : members.tool_parameters) {
    // json2pb adds one tool per array element and the scan admits only object
    // elements, so the indices line up; guard the invariant all the same.
    if (index >= request->tools_size()) {
      return Status(StatusCode::INVALID_ARGUMENT,
                    "tools decoded inconsistently with the request body");
    }
    request->mutable_tools(index)
        ->mutable_function()
        ->mutable_parameters()
        ->Swap(&parameters);
  }
  return Status();
}

}  // namespace xllm

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

#include "api_service/chat_request_decoder.h"

#include <butil/third_party/rapidjson/document.h>
#include <google/protobuf/util/json_util.h>
#include <json2pb/json_to_pb.h>

#include <algorithm>

namespace xllm {
namespace {

namespace rapidjson = BUTIL_RAPIDJSON_NAMESPACE;

// Container nesting beyond which a Struct member is left to the reference
// parser, which enforces its own recursion limit.
constexpr int32_t kMaxStructDepth = 100;

// What json2pb sees in place of a carved-out Struct member. It skips a null
// value, leaving the field unset for the decoded Struct to be attached.
constexpr std::string_view kBlankedMember = "null";

// `depth` counts container nesting; the two functions recurse into each other
// and only the container hops increment it.
bool value_to_proto(const rapidjson::Value& value,
                    google::protobuf::Value* out,
                    int32_t depth);

bool object_to_struct(const rapidjson::Value& object,
                      google::protobuf::Struct* out,
                      int32_t depth) {
  if (depth > kMaxStructDepth) {
    return false;
  }
  auto* fields = out->mutable_fields();
  for (auto it = object.MemberBegin(); it != object.MemberEnd(); ++it) {
    // A repeated key overwrites the earlier entry, as it does in the map the
    // reference parser fills.
    google::protobuf::Value& slot = (*fields)[std::string(
        it->name.GetString(), it->name.GetStringLength())];
    if (!value_to_proto(it->value, &slot, depth)) {
      return false;
    }
  }
  return true;
}

bool value_to_proto(const rapidjson::Value& value,
                    google::protobuf::Value* out,
                    int32_t depth) {
  if (value.IsNull()) {
    out->set_null_value(google::protobuf::NULL_VALUE);
    return true;
  }
  if (value.IsBool()) {
    out->set_bool_value(value.GetBool());
    return true;
  }
  if (value.IsNumber()) {
    // Struct carries every number as a double, as the reference parser does.
    out->set_number_value(value.GetDouble());
    return true;
  }
  if (value.IsString()) {
    out->set_string_value(value.GetString(), value.GetStringLength());
    return true;
  }
  if (value.IsArray()) {
    if (depth >= kMaxStructDepth) {
      return false;
    }
    auto* list = out->mutable_list_value();
    list->mutable_values()->Reserve(static_cast<int32_t>(value.Size()));
    for (auto it = value.Begin(); it != value.End(); ++it) {
      if (!value_to_proto(*it, list->add_values(), depth + 1)) {
        return false;
      }
    }
    return true;
  }
  return object_to_struct(value, out->mutable_struct_value(), depth + 1);
}

// Parses one carved-out member into a Struct. Returns false when rapidjson
// refuses the bytes, so the reference parser can decide the whole body.
bool parse_struct(std::string_view json, google::protobuf::Struct* out) {
  // rapidjson parses a terminated buffer; the Struct-typed members are small.
  const std::string buffer(json);
  rapidjson::Document document;
  document.Parse<rapidjson::kParseValidateEncodingFlag |
                 rapidjson::kParseIterativeFlag |
                 rapidjson::kParseFullPrecisionFlag>(buffer.c_str());
  if (document.HasParseError() || !document.IsObject()) {
    return false;
  }
  return object_to_struct(document, out, /*depth=*/0);
}

bool decode_struct_members(std::string_view body,
                           const ChatStructSpans& spans,
                           ChatStructMembers* members) {
  if (spans.chat_template_kwargs.has_value()) {
    const JsonSpan& span = spans.chat_template_kwargs.value();
    google::protobuf::Struct kwargs;
    if (!parse_struct(body.substr(span.begin, span.end - span.begin),
                      &kwargs)) {
      return false;
    }
    members->chat_template_kwargs = std::move(kwargs);
  }
  members->tool_parameters.reserve(spans.tool_parameters.size());
  for (const auto& [index, span] : spans.tool_parameters) {
    google::protobuf::Struct parameters;
    if (!parse_struct(body.substr(span.begin, span.end - span.begin),
                      &parameters)) {
      return false;
    }
    members->tool_parameters.emplace_back(static_cast<int32_t>(index),
                                          std::move(parameters));
  }
  return true;
}

// Copies the body with every Struct member replaced by null.
std::string blank_struct_members(std::string_view body,
                                 const ChatStructSpans& spans) {
  std::vector<JsonSpan> ordered;
  ordered.reserve(spans.tool_parameters.size() + 1);
  if (spans.chat_template_kwargs.has_value()) {
    ordered.emplace_back(spans.chat_template_kwargs.value());
  }
  for (const auto& [index, span] : spans.tool_parameters) {
    ordered.emplace_back(span);
  }
  std::sort(ordered.begin(),
            ordered.end(),
            [](const JsonSpan& lhs, const JsonSpan& rhs) {
              return lhs.begin < rhs.begin;
            });

  std::string blanked;
  blanked.reserve(body.size());
  size_t copied = 0;
  for (const JsonSpan& span : ordered) {
    blanked.append(body.data() + copied, span.begin - copied);
    blanked.append(kBlankedMember);
    copied = span.end;
  }
  blanked.append(body.data() + copied, body.size() - copied);
  return blanked;
}

// The reference decode: protobuf's JSON parser, as the chat handler has always
// used it.
Status decode_with_reference_parser(const std::string& body,
                                    google::protobuf::Message* request) {
  google::protobuf::util::JsonParseOptions options;
  options.ignore_unknown_fields = true;
  const auto status =
      google::protobuf::util::JsonStringToMessage(body, request, options);
  if (!status.ok()) {
    return Status(StatusCode::INVALID_ARGUMENT, status.ToString());
  }
  return Status();
}

}  // namespace

Status decode_chat_request_core(std::string body,
                                google::protobuf::Message* request,
                                ChatStructMembers* members) {
  const std::optional<ChatStructSpans> spans = locate_chat_struct_members(body);
  if (!spans.has_value()) {
    return decode_with_reference_parser(body, request);
  }

  ChatStructMembers decoded;
  if (!decode_struct_members(body, spans.value(), &decoded)) {
    return decode_with_reference_parser(body, request);
  }

  // The common body carries neither Struct member and goes to json2pb as is.
  const bool has_struct_members = spans->chat_template_kwargs.has_value() ||
                                  !spans->tool_parameters.empty();
  const std::string blanked = has_struct_members
                                  ? blank_struct_members(body, spans.value())
                                  : std::string();
  const std::string& json2pb_input = has_struct_members ? blanked : body;

  std::string error;
  json2pb::Json2PbOptions options;
  const bool ok =
      json2pb::JsonToProtoMessage(json2pb_input, request, options, &error);
  // json2pb drops an optional scalar whose JSON type does not match instead
  // of rejecting it, and reports that through `error` while still returning
  // true. The reference parser would have coerced or rejected the value, and
  // its wording is what a rejected client has always seen, so it decides both
  // that case and outright rejections.
  if (!ok || !error.empty()) {
    request->Clear();
    return decode_with_reference_parser(body, request);
  }
  *members = std::move(decoded);
  return Status();
}

}  // namespace xllm

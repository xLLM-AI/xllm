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

#pragma once

#include <brpc/controller.h>

#include <string>

#include "core/common/types.h"

namespace xllm {
namespace api_service {

// Map an internal StatusCode to the corresponding HTTP status code.
int32_t status_code_to_http_status(StatusCode code);

// Map an internal StatusCode to an OpenAI-style error "type" string
// (e.g. "invalid_request_error", "rate_limit_error").
const char* status_code_to_openai_error_type(StatusCode code);

// Map an internal StatusCode to an Anthropic-style error "type" string.
const char* status_code_to_anthropic_error_type(StatusCode code);

// Stable machine-readable short code for a StatusCode (e.g.
// "invalid_argument").
const char* status_code_to_string(StatusCode code);

// Build an OpenAI-compatible error body:
//   {"error":{"message":...,"type":...,"code":...,"param":null}}
std::string make_openai_error_json(StatusCode code, const std::string& message);

// Build an Anthropic-compatible error body:
//   {"type":"error","error":{"type":...,"message":...}}
std::string make_anthropic_error_json(StatusCode code,
                                      const std::string& message);

// Extract the client-supplied x-request-id from an HTTP controller, looking up
// "x-request-id" then "x-ms-client-request-id". Returns an empty string when
// neither header is present.
std::string get_header_x_request_id(const brpc::Controller* controller);

// Generate a server-side request-scoped x-request-id, formatted like the
// engine's internal request ids ("req-<instance-hash>-<short-uuid>") so it is
// globally unique and visually consistent with them.
std::string generate_x_request_id();

// Resolve the x-request-id for a request: the client-supplied value when
// present, otherwise a freshly generated one. Guarantees a non-empty result so
// early failures (before a Call exists) can still be correlated.
std::string ensure_x_request_id(const brpc::Controller* controller);

// Log a request failure together with its x-request-id, without touching the
// response. Used by streaming paths that have already committed a 200 header
// and can no longer change the status code or body envelope.
void log_request_error(StatusCode code,
                       const std::string& message,
                       const std::string& x_request_id);

// Finish a non-stream HTTP request with a structured OpenAI-style error: sets
// the HTTP status code, writes the JSON error body, and logs the failure with
// the x-request-id. Does NOT call SetFailed, which would discard the custom
// body and force a generic 500 status.
void write_openai_error(brpc::Controller* controller,
                        StatusCode code,
                        const std::string& message,
                        const std::string& x_request_id);

// Same as write_openai_error but emits the Anthropic error envelope.
void write_anthropic_error(brpc::Controller* controller,
                           StatusCode code,
                           const std::string& message,
                           const std::string& x_request_id);

}  // namespace api_service
}  // namespace xllm

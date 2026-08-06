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
#include <cstdint>
#include <string>
#include <vector>

namespace xllm {
namespace net {

std::string get_local_ip_addr();
std::string get_route_ip(const std::string& remote_addr);
int get_local_free_port();
uint64_t convert_ip_port_to_uint64(const std::string& ip, uint16_t port);
std::pair<std::string, uint16_t> convert_uint64_to_ip_port(uint64_t input);
void parse_host_port_from_addr(const std::string& addr,
                               std::string& host,
                               int& port);

std::string extract_ip(const std::string& input);
std::string extract_port(const std::string& input);
}  // namespace net
}  // namespace xllm

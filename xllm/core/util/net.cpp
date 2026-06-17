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

#include "net.h"

#include <arpa/inet.h>
#include <glog/logging.h>
#include <netdb.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <memory>
#include <mutex>
#include <sstream>
#include <unordered_set>

namespace xllm {
namespace net {

namespace {

std::mutex g_port_mutex;
std::unordered_set<int> g_allocated_port_map;

}  // namespace

// TODO: return private ip
std::string get_local_ip_addr() {
  char ip[INET_ADDRSTRLEN]{'\0'};
  char hostname[256];
  int ret = gethostname(hostname, sizeof(hostname));
  if (ret != 0) {
    LOG(ERROR) << "gethostname failed";
    return "";
  }
  struct addrinfo* info = nullptr;
  struct addrinfo hints;
  memset(&hints, 0, sizeof(hints));
  hints.ai_family = AF_INET;
  hints.ai_socktype = SOCK_STREAM;
  ret = getaddrinfo(hostname, nullptr, &hints, &info);
  if (ret != 0) {
    LOG(ERROR) << "getaddrinfo failed";
    return "";
  }
  std::unique_ptr<struct addrinfo, decltype(&freeaddrinfo)> guard(info,
                                                                  freeaddrinfo);
  const sockaddr_in* addr = reinterpret_cast<const sockaddr_in*>(info->ai_addr);
  const char* result =
      inet_ntop(addr->sin_family, &addr->sin_addr, ip, sizeof(ip));
  if (result == nullptr) {
    LOG(ERROR) << "inet_ntop failed";
    return "";
  }

  return std::string(ip);
}

int get_local_free_port() {
  std::lock_guard<std::mutex> lock(g_port_mutex);
  int port = 0;
  do {
    port = 0;
    struct sockaddr_in addr;
    const int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) {
      return -1;
    }
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons(port);
    if (bind(fd, reinterpret_cast<struct sockaddr*>(&addr), sizeof(addr)) < 0) {
      ::close(fd);
      return -1;
    }
    socklen_t len = sizeof(addr);
    if (getsockname(fd, reinterpret_cast<struct sockaddr*>(&addr), &len) ==
        -1) {
      ::close(fd);
      return -1;
    }
    port = ntohs(addr.sin_port);

    ::close(fd);
  } while (g_allocated_port_map.find(port) != g_allocated_port_map.end());

  g_allocated_port_map.insert(port);

  return port;
}

uint64_t convert_ip_port_to_uint64(const std::string& ip, uint16_t port) {
  in_addr ip_addr;
  CHECK(inet_pton(AF_INET, ip.c_str(), &ip_addr) == 1)
      << "Invalid IPv4 address format : " << ip;

  uint32_t ip_network = ip_addr.s_addr;
  return (static_cast<uint64_t>(ip_network) << 32) | port;
}

std::pair<std::string, uint16_t> convert_uint64_to_ip_port(uint64_t input) {
  uint16_t port = static_cast<uint16_t>(input & 0xFFFF);
  uint32_t ip_network = static_cast<uint32_t>(input >> 32);

  in_addr ip_addr;
  ip_addr.s_addr = ip_network;

  char ip_str[INET_ADDRSTRLEN];
  const char* result = inet_ntop(AF_INET, &ip_addr, ip_str, INET_ADDRSTRLEN);
  CHECK(result != nullptr) << "Failed to convert IP address from uint64: "
                           << input;

  return {std::string(ip_str), port};
}

// input example: 127.0.0.1:18889
std::string extract_ip(const std::string& input) {
  std::istringstream stream(input);
  std::string ip;

  std::getline(stream, ip, ':');
  if (ip == "127.0.0.1" || ip == "0.0.0.0" || ip == "localhost") {
    ip = get_local_ip_addr();
  }
  return ip;
}

std::string extract_port(const std::string& input) {
  std::istringstream stream(input);
  std::string ip;
  std::string port;

  std::getline(stream, ip, ':');
  std::getline(stream, port, ':');

  return port;
}

void parse_host_port_from_addr(const std::string& addr,
                               std::string& host,
                               int& port) {
  CHECK(!addr.empty()) << "Address is empty";

  const std::size_t colon_pos = addr.find(':');
  CHECK_NE(colon_pos, std::string::npos) << "Invalid address format: " << addr;

  host = addr.substr(0, colon_pos);
  port = std::stoi(addr.substr(colon_pos + 1));
}

}  // namespace net
}  // namespace xllm

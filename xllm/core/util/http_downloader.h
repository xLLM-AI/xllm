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

#pragma once

#include <brpc/channel.h>
#include <brpc/controller.h>
#include <bthread/rwlock.h>

#include <memory>
#include <string>
#include <unordered_map>
#include <utility>

namespace xllm {

class HttpDownloader {
 public:
  HttpDownloader() = default;
  virtual ~HttpDownloader() = default;

  bool fetch_data(
      const std::string& url,
      std::string& data,
      const std::unordered_map<std::string, std::string>& headers = {});

 protected:
  bool parse_url(const std::string& url, std::string& host);
  virtual bool download(
      const std::string& host,
      const std::string& url,
      std::string& data,
      const std::unordered_map<std::string, std::string>& headers = {}) = 0;
};

class BRpcDownloader final : public HttpDownloader {
 public:
  explicit BRpcDownloader(
      std::unordered_map<std::string, std::string> default_headers = {})
      : default_headers_(std::move(default_headers)) {}
  ~BRpcDownloader() override = default;

  bool download(const std::string& host,
                const std::string& url,
                std::string& data,
                const std::unordered_map<std::string, std::string>& headers =
                    {}) override;

 private:
  std::shared_ptr<brpc::Channel> get_channel(const std::string& host);

 private:
  const std::unordered_map<std::string, std::string> default_headers_;
  inline static bthread::RWLock rw_lock_;
  inline static std::unordered_map<std::string, std::shared_ptr<brpc::Channel>>
      channels_;
};

// Parse a JSON string of headers into a key-value map.
// Returns empty map on empty input or parse failure.
// Exposed for testing.
std::unordered_map<std::string, std::string> parse_headers_json(
    const std::string& raw);

}  // namespace xllm

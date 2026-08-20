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

#include "http_downloader.h"

#include <glog/logging.h>

#include <nlohmann/json.hpp>

#include "core/framework/config/model_config.h"

namespace xllm {

std::unordered_map<std::string, std::string> parse_headers_json(
    const std::string& raw) {
  std::unordered_map<std::string, std::string> result;
  if (raw.empty()) return result;

  try {
    auto j = nlohmann::json::parse(raw);
    for (auto& [k, v] : j.items()) {
      result[k] = v.get<std::string>();
    }
  } catch (const std::exception& e) {
    LOG(ERROR) << "Failed to parse mm_download_headers JSON: " << e.what();
  }
  return result;
}

std::unordered_map<std::string, std::string> parse_global_headers() {
  static const std::unordered_map<std::string, std::string> cached =
      parse_headers_json(ModelConfig::get_instance().mm_download_headers());
  return cached;
}

bool HttpDownloader::fetch_data(
    const std::string& url,
    std::string& data,
    const std::unordered_map<std::string, std::string>& headers) {
  std::string host;
  if (!parse_url(url, host)) {
    return false;
  }

  return download(host, url, data, headers);
}

bool HttpDownloader::parse_url(const std::string& url, std::string& host) {
  size_t pos = url.find("://");
  if (pos == std::string::npos) {
    LOG(ERROR) << " Invalid URL, missing protocol, url is" << url;
    return false;
  }

  size_t host_start = pos + 3;
  pos = url.find('/', host_start);
  if (pos == std::string::npos) {
    LOG(ERROR) << "Invalid URL, no path is found, url is" << url;
    return false;
  }

  host = url.substr(host_start, pos - host_start);
  return true;
}

std::shared_ptr<brpc::Channel> BRpcDownloader::get_channel(
    const std::string& host) {
  {
    bthread::RWLockRdGuard rd_guard(rw_lock_);
    auto it = channels_.find(host);
    if (it != channels_.end()) {
      return it->second;
    }
  }

  bthread::RWLockWrGuard wr_guard(rw_lock_);
  auto it = channels_.find(host);
  if (it != channels_.end()) {
    return it->second;
  }

  brpc::ChannelOptions option;
  option.protocol = brpc::PROTOCOL_HTTP;
  option.connection_type = brpc::CONNECTION_TYPE_POOLED;
  option.max_retry = 3;

  auto channel = std::make_shared<brpc::Channel>();
  if (channel->Init(host.c_str(), &option) != 0) {
    LOG(ERROR) << "fail to init channel for " << host;
    return nullptr;
  }

  channels_[host] = channel;
  return channel;
}

bool BRpcDownloader::download(
    const std::string& host,
    const std::string& url,
    std::string& data,
    const std::unordered_map<std::string, std::string>& headers) {
  brpc::Controller cntl;
  cntl.http_request().uri() = url;
  // 1) global defaults (lowest priority)
  for (const auto& [k, v] : parse_global_headers()) {
    cntl.http_request().SetHeader(k, v);
  }
  // 2) per-request headers (override global)
  for (const auto& [k, v] : headers) {
    cntl.http_request().SetHeader(k, v);
  }
  cntl.set_timeout_ms(2000);
  auto channel = get_channel(host);
  if (!channel) {
    LOG(ERROR) << "Channel is null";
    return false;
  }

  channel->CallMethod(nullptr, &cntl, nullptr, nullptr, nullptr);
  if (cntl.Failed()) {
    LOG(ERROR) << "Request failed: " << cntl.ErrorText();
    return false;
  }

  if (cntl.http_response().status_code() != 200) {
    LOG(ERROR) << "HTTP error: " << cntl.http_response().status_code();
    return false;
  }

  const butil::IOBuf& io = cntl.response_attachment();
  data = std::move(io.to_string());
  return true;
}

}  // namespace xllm

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

#include "tests/api_service/http_protocol_test_fixture.h"

#include <netinet/in.h>
#include <spawn.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>

#include <chrono>

#include "core/framework/config/service_config.h"

extern char** environ;

namespace xllm::api_service {

bool AdmissionTestService::wait_held() {
  std::unique_lock<std::mutex> lock(mutex_);
  return changed_.wait_for(lock, std::chrono::seconds(5), [this] {
    return held_request_ != nullptr;
  });
}

void AdmissionTestService::release(bool cancel) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (cancel && held_request_ != nullptr) {
    held_request_->set_cancel();
  }
  released_ = true;
  changed_.notify_all();
}

void AdmissionTestService::stop() {
  std::lock_guard<std::mutex> lock(mutex_);
  stopping_ = true;
  if (held_request_ != nullptr) {
    held_request_->set_cancel();
  }
  changed_.notify_all();
}

bool AdmissionTestService::wait_released() {
  std::unique_lock<std::mutex> lock(mutex_);
  return changed_.wait_for(lock, std::chrono::seconds(5), [this] {
    return held_request_ == nullptr &&
           rate_limiter_.get_num_concurrent_requests() == 0;
  });
}

bool AdmissionTestService::observe_disconnect() {
  std::lock_guard<std::mutex> lock(mutex_);
  if (held_request_ == nullptr) {
    return false;
  }
  held_request_->update_connection_status();
  return held_request_->cancelled();
}

void HttpProtocolTestFixture::start_service(AdmissionTestService& service,
                                            const char* routes) {
  service_ = &service;
  previous_limit_ = ServiceConfig::get_instance().max_concurrent_requests();
  ServiceConfig::get_instance().max_concurrent_requests(1);
  ASSERT_EQ(
      server_.AddService(service_, brpc::SERVER_DOESNT_OWN_SERVICE, routes), 0);
  ASSERT_EQ(server_.Start(/*port=*/0, /*options=*/nullptr), 0);
  brpc::ChannelOptions options;
  options.protocol = brpc::PROTOCOL_HTTP;
  options.timeout_ms = 5000;
  options.max_retry = 0;
  ASSERT_EQ(channel_.Init(server_.listen_address(), &options), 0);
}

void HttpProtocolTestFixture::TearDown() {
  service_->stop();
  server_.Stop(/*timeout_ms=*/0);
  server_.Join();
  EXPECT_EQ(service_->rate_limiter().get_num_concurrent_requests(), 0);
  ServiceConfig::get_instance().max_concurrent_requests(previous_limit_);
}

void HttpProtocolTestFixture::post(const char* path,
                                   const std::string& body,
                                   brpc::Controller& controller,
                                   const char* content_type) {
  controller.http_request().uri() = path;
  controller.http_request().set_method(brpc::HTTP_METHOD_POST);
  if (content_type != nullptr) {
    controller.http_request().set_content_type(content_type);
  }
  controller.request_attachment().append(body);
  channel_.CallMethod(nullptr, &controller, nullptr, nullptr, nullptr);
}

int32_t HttpProtocolTestFixture::open_socket(const char* path,
                                             const std::string& body) {
  const int32_t fd = socket(AF_INET, SOCK_STREAM, 0);
  if (fd < 0) {
    return fd;
  }
  sockaddr_in address{};
  address.sin_family = AF_INET;
  address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  address.sin_port = htons(server_.listen_address().port);
  if (connect(fd, reinterpret_cast<sockaddr*>(&address), sizeof(address)) !=
      0) {
    close(fd);
    return -1;
  }
  const std::string wire = std::string("POST ") + path +
                           " HTTP/1.1\r\n"
                           "Host: localhost\r\nContent-Type: "
                           "application/json\r\nContent-Length: " +
                           std::to_string(body.size()) + "\r\n\r\n" + body;
  if (send(fd, wire.data(), wire.size(), MSG_NOSIGNAL) !=
      static_cast<ssize_t>(wire.size())) {
    close(fd);
    return -1;
  }
  return fd;
}

std::string HttpProtocolTestFixture::base_url() const {
  return "http://127.0.0.1:" + std::to_string(server_.listen_address().port);
}

void HttpProtocolTestFixture::run_sdk(const char* script,
                                      const std::string& url,
                                      const std::string& phase) {
  std::string script_path(script);
  std::string target_url(url);
  std::string selected_phase(phase);
  char* args[] = {
      script_path.data(), target_url.data(), selected_phase.data(), nullptr};
  pid_t pid = -1;
  ASSERT_EQ(
      posix_spawn(&pid, script_path.c_str(), nullptr, nullptr, args, environ),
      0);
  int32_t status = 0;
  ASSERT_EQ(waitpid(pid, &status, 0), pid);
  ASSERT_TRUE(WIFEXITED(status));
  EXPECT_EQ(WEXITSTATUS(status), 0);
}

}  // namespace xllm::api_service

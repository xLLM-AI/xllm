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

#include "xllm_server.h"

#include <brpc/server.h>
#include <butil/at_exit.h>
#include <unistd.h>

#include <array>
#include <csignal>

#include "core/common/global_flags.h"
#include "core/framework/config/disagg_pd_config.h"
#include "core/framework/config/distributed_config.h"
#include "core/framework/config/kv_cache_config.h"
#include "core/framework/config/service_config.h"
#include "health_reporter.h"

namespace xllm {

namespace {
volatile std::sig_atomic_t g_quit_flag = 0;

void quit_signal_handler(int /*signum*/) { g_quit_flag = 1; }

constexpr const char* kApiServiceRoutes =
    "v1/completions => CompletionsHttp,"
    "v1/sample => SampleHttp,"
    "v1/chat/completions => ChatCompletionsHttp,"
    "v1/embeddings => EmbeddingsHttp,"
    "v1/models => ModelsHttp,"
    "v1/image/generation => ImageGenerationHttp,"
    "v1/audio/generation => AudioGenerationHttp,"
    "v1/text/generation => TextGenerationHttp,"
    "v1/video/generation => VideoGenerationHttp,"
    "v1/rerank => RerankHttp,"
    "v1/messages => AnthropicMessagesHttp,"
    "v2/repository/index => ModelVersionsHttp,"
    "fork_master => ForkMasterHttp,"
    "sleep => SleepHttp,"
    "wakeup => WakeupHttp,"
    "start_profile => StartProfileHttp,"
    "stop_profile => StopProfileHttp,"
    "pause => PauseHttp,"
    "resume => ResumeHttp,"
    "link_p2p => LinkP2PHttp,"
    "unlink_p2p => UnlinkP2PHttp";

constexpr const char* kForkOnlyRoute = "fork_master => ForkMasterHttp";

struct ApiRouteBinding {
  const char* name;
  bool (*enabled)();
  const char* routes;
};

bool is_master_node() {
  return ::xllm::DistributedConfig::get_instance().node_rank() == 0;
}

bool is_xtensor_node() {
  return ::xllm::DistributedConfig::get_instance().node_rank() != 0 &&
         ::xllm::KVCacheConfig::get_instance().enable_xtensor();
}

const char* get_api_service_routes_for_current_mode() {
  static constexpr std::array<ApiRouteBinding, 2> kBindings = {{
      {"master_node", &is_master_node, kApiServiceRoutes},
      {"xtensor_node", &is_xtensor_node, kForkOnlyRoute},
  }};
  for (const auto& binding : kBindings) {
    if (binding.enabled()) {
      LOG(INFO) << "Use API route mode: " << binding.name;
      return binding.routes;
    }
  }
  return nullptr;
}

void install_quit_signal_handler() {
  g_quit_flag = 0;
  struct sigaction sa = {};
  sa.sa_handler = quit_signal_handler;
  sigemptyset(&sa.sa_mask);
  sigaction(SIGINT, &sa, nullptr);
  sigaction(SIGTERM, &sa, nullptr);
}

void wait_for_quit_signal() {
  while (!g_quit_flag) {
    sleep(1);
  }
}

bool configure_generic_server(brpc::Server* server,
                              google::protobuf::Service* service,
                              const std::string& server_name) {
  if (server->AddService(service, brpc::SERVER_DOESNT_OWN_SERVICE) != 0) {
    LOG(ERROR) << "Fail to add " << server_name << " service";
    return false;
  }
  return true;
}
}  // namespace

XllmServer::XllmServer() { butil::AtExitManager exit_manager; }

XllmServer::~XllmServer() {
  stop();

  if (running_thread_ && running_thread_->joinable()) {
    running_thread_->join();
  }
}

bool XllmServer::start(std::unique_ptr<APIService> service) {
  server_ = std::make_unique<brpc::Server>();
  if (const char* routes = get_api_service_routes_for_current_mode();
      routes != nullptr) {
    if (server_->AddService(
            service.get(), brpc::SERVER_DOESNT_OWN_SERVICE, routes) != 0) {
      LOG(ERROR) << "Fail to add api service";
      return false;
    }
  } else {
    LOG(INFO) << "No API routes enabled on current node mode.";
  }

  brpc::ServerOptions options;
  // TODO: enable arean message factory later.
  // options.rpc_pb_message_factory =
  //    brpc::GetArenaRpcPBMessageFactory<1024 * 1024, 1024 * 1024 * 128>();
  options.idle_timeout_sec =
      ::xllm::ServiceConfig::get_instance().rpc_idle_timeout_s();
  options.num_threads = ::xllm::ServiceConfig::get_instance().num_threads();
  // Use custom health reporter for /health endpoint
  options.health_reporter = &HealthReporter::instance();
  if (server_->Start(::xllm::ServiceConfig::get_instance().port(), &options) !=
      0) {
    LOG(ERROR) << "Failed to start server on port "
               << ::xllm::ServiceConfig::get_instance().port();
    return false;
  }
  LOG(INFO) << "Brpc Server started on port "
            << ::xllm::ServiceConfig::get_instance().port()
            << ", idle_timeout_s: "
            << ::xllm::ServiceConfig::get_instance().rpc_idle_timeout_s()
            << ", num_threads: "
            << ::xllm::ServiceConfig::get_instance().num_threads();

  listen_address_ =
      std::string(butil::endpoint2str(server_->listen_address()).c_str());
  listen_port_ = ::xllm::ServiceConfig::get_instance().port();
  has_initialized_ = true;

  auto pid = getpid();
  LOG(INFO) << "     Started server process [" << pid << "]";
  LOG(INFO) << "     Waiting for application startup.";
  LOG(INFO) << "     Application startup complete.";

  install_quit_signal_handler();
  wait_for_quit_signal();

  LOG(INFO) << "     Shutting down";
  LOG(INFO) << "     Waiting for application shutdown.";

  stop();

  LOG(INFO) << "     Application shutdown complete.";
  LOG(INFO) << "     Finished server process [" << pid << "]";

  return true;
}

bool XllmServer::start(std::unique_ptr<DisaggPDService> service) {
  std::string addr("");
  if (!::xllm::ServiceConfig::get_instance().host().empty()) {
    addr =
        ::xllm::ServiceConfig::get_instance().host() + ":" +
        std::to_string(::xllm::DisaggPDConfig::get_instance().disagg_pd_port());
  }
  if (!create_server((google::protobuf::Service*)(service.get()),
                     addr,
                     ::xllm::DisaggPDConfig::get_instance().disagg_pd_port(),
                     "Disagg PD")) {
    return false;
  }

  has_initialized_ = true;
  // Wait until Ctrl-C is pressed, then Stop() and Join() the server.
  server_->Join();
  return true;
}

bool XllmServer::start(std::unique_ptr<PDOOCService> service) {
  std::string addr("");
  if (!::xllm::ServiceConfig::get_instance().host().empty()) {
    addr =
        ::xllm::ServiceConfig::get_instance().host() + ":" +
        std::to_string(::xllm::DisaggPDConfig::get_instance().disagg_pd_port());
  }
  if (!create_server((google::protobuf::Service*)(service.get()),
                     addr,
                     ::xllm::DisaggPDConfig::get_instance().disagg_pd_port(),
                     "PD OOC")) {
    return false;
  }

  has_initialized_ = true;
  // Wait until Ctrl-C is pressed, then Stop() and Join() the server.
  server_->Join();
  return true;
}

bool XllmServer::start(std::shared_ptr<CollectiveService> service,
                       const std::string& addr,
                       const std::string& server_name) {
  if (!create_server(
          (google::protobuf::Service*)(service.get()), addr, -1, server_name)) {
    return false;
  }

  running_thread_ =
      std::make_unique<std::thread>([this, service = std::move(service)]() {
        has_initialized_ = true;
        server_->Join();
      });

  return true;
}

bool XllmServer::start(std::shared_ptr<WorkerService> service,
                       const std::string& addr) {
  return create_server(static_cast<google::protobuf::Service*>(service.get()),
                       addr,
                       -1,
                       "DistributeWorker");
}

bool XllmServer::start(std::shared_ptr<XTensorDistService> service,
                       const std::string& addr) {
  return create_server(static_cast<google::protobuf::Service*>(service.get()),
                       addr,
                       -1,
                       "XTensorDist");
}

bool XllmServer::create_server(google::protobuf::Service* service,
                               const std::string& addr,
                               int port,
                               const std::string& server_name) {
  server_ = std::make_unique<brpc::Server>();
  if (!configure_generic_server(server_.get(), service, server_name)) {
    return false;
  }

  brpc::ServerOptions options;
  options.idle_timeout_sec =
      ::xllm::ServiceConfig::get_instance().rpc_idle_timeout_s();
  options.num_threads = ::xllm::ServiceConfig::get_instance().num_threads();
  butil::EndPoint endpoint;
  if (!addr.empty()) {
    listen_address_ = addr;
    if (butil::str2endpoint(listen_address_.c_str(), &endpoint) < 0) {
      LOG(FATAL) << "Convert listen_address_ to endpoint failed: "
                 << listen_address_;
      return false;
    }
  } else {
    endpoint = butil::EndPoint(butil::IP_ANY, port);
  }

  if (server_->Start(endpoint, &options) != 0) {
    LOG(ERROR) << "Failed to start " << server_name
               << " server on address: " << endpoint;
    return false;
  }

  listen_address_ =
      std::string(butil::endpoint2str(server_->listen_address()).c_str());
  listen_port_ = server_->listen_address().port;

  LOG(INFO) << server_name << " server started on address "
            << server_->listen_address() << ", idle_timeout_sec: "
            << ::xllm::ServiceConfig::get_instance().rpc_idle_timeout_s()
            << ", num_threads: "
            << ::xllm::ServiceConfig::get_instance().num_threads();

  return true;
}

void XllmServer::run() {
  if (has_initialized_) {
    return;
  }

  has_initialized_ = true;
  server_->Join();
}

void XllmServer::stop() {
  if (!server_) {
    return;
  }
  server_->Stop(0);
  server_->Join();
}

}  // namespace xllm

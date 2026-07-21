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

#include <brpc/controller.h>
#include <brpc/server.h>
#include <butil/endpoint.h>
#include <butil/logging.h>
#include <gflags/gflags.h>

#include <cctype>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "llm.h"
#include "rec.h"
#include "utils.h"
#include "xllm_test.pb.h"

namespace xllm_capi_test {

namespace {

std::unique_ptr<char[]> CopyContent(const std::string& s) {
  if (s.empty()) {
    return nullptr;
  }
  auto p = std::make_unique<char[]>(s.size() + 1);
  std::memcpy(p.get(), s.data(), s.size());
  p[s.size()] = '\0';
  return p;
}

void SetErrorResponse(c_api_test::XLLM_Response* res,
                      c_api_test::XLLM_StatusCode code,
                      const std::string& msg) {
  res->Clear();
  res->set_status_code(code);
  res->set_error_info(msg);
}

}  // namespace

class XllmRecCapiServiceImpl : public c_api_test::XllmRecCapiService {
 public:
  XllmRecCapiServiceImpl(XLLM_REC_Handler* rec_handler,
                         XLLM_LLM_Handler* llm_handler)
      : rec_handler_(rec_handler), llm_handler_(llm_handler) {}

  void Inference(google::protobuf::RpcController* cntl_base,
                 const c_api_test::XLLM_Request* request,
                 c_api_test::XLLM_Response* response,
                 google::protobuf::Closure* done) override {
    brpc::ClosureGuard done_guard(done);
    (void)cntl_base;

    std::lock_guard<std::mutex> lock(mu_);

    const std::string& fn = request->call_function();
    const bool is_llm_fn =
        (fn == "xllm_llm_completions" || fn == "xllm_llm_chat_completions");
    const bool is_rec_fn = (fn == "xllm_rec_text_completions" ||
                            fn == "xllm_rec_token_completions" ||
                            fn == "xllm_rec_multimodal_completions" ||
                            fn == "xllm_rec_chat_completions");

    if (is_llm_fn && !llm_handler_) {
      SetErrorResponse(response,
                       c_api_test::XLLM_STATUS_INTERNAL_ERROR,
                       "LLM handler is null");
      return;
    }
    if (is_rec_fn && !rec_handler_) {
      SetErrorResponse(response,
                       c_api_test::XLLM_STATUS_INTERNAL_ERROR,
                       "REC handler is null");
      return;
    }

    XLLM_RequestParams params{};
    if (is_llm_fn) {
      xllm_llm_request_params_default(&params);
    } else {
      xllm_rec_request_params_default(&params);
    }
    if (request->params().ByteSizeLong() > 0) {
      PbToXllmRequestParams(request->params(), &params);
    }

    const char* model_id =
        request->model_id().empty() ? "" : request->model_id().c_str();
    const uint32_t timeout_ms = request->timeout_ms();

    XLLM_Response* raw = nullptr;

    if (fn == "xllm_llm_completions") {
      raw = xllm_llm_completions(llm_handler_,
                                 model_id,
                                 request->prompt().c_str(),
                                 timeout_ms,
                                 &params);
    } else if (fn == "xllm_llm_chat_completions") {
      std::vector<XLLM_ChatMessage> cms;
      std::vector<std::unique_ptr<char[]>> contents;
      cms.reserve(request->messages_size());
      for (int i = 0; i < request->messages_size(); ++i) {
        const auto& m = request->messages(i);
        XLLM_ChatMessage cm{};
        std::memset(cm.role, 0, sizeof(cm.role));
        std::strncpy(
            cm.role, m.role().c_str(), XLLM_META_STRING_FIELD_MAX_LEN - 1);
        cm.role[XLLM_META_STRING_FIELD_MAX_LEN - 1] = '\0';
        contents.push_back(CopyContent(m.content()));
        cm.content = contents.back() ? contents.back().get() : nullptr;
        cms.push_back(cm);
      }
      raw = xllm_llm_chat_completions(llm_handler_,
                                      model_id,
                                      cms.empty() ? nullptr : cms.data(),
                                      cms.size(),
                                      timeout_ms,
                                      &params);
    } else if (fn == "xllm_rec_text_completions") {
      raw = xllm_rec_text_completions(rec_handler_,
                                      model_id,
                                      request->prompt().c_str(),
                                      timeout_ms,
                                      &params);
    } else if (fn == "xllm_rec_token_completions") {
      std::vector<int32_t> token_ids;
      token_ids.reserve(request->token_ids_size());
      for (int i = 0; i < request->token_ids_size(); ++i) {
        token_ids.push_back(request->token_ids(i));
      }
      raw = xllm_rec_token_completions(
          rec_handler_,
          model_id,
          token_ids.empty() ? nullptr : token_ids.data(),
          token_ids.size(),
          timeout_ms,
          &params);
    } else if (fn == "xllm_rec_multimodal_completions") {
      std::vector<int32_t> token_ids;
      token_ids.reserve(request->token_ids_size());
      for (int i = 0; i < request->token_ids_size(); ++i) {
        token_ids.push_back(request->token_ids(i));
      }
      XLLM_MM_Data mm{};
      MmDataOwned mm_owned;
      if (!PbToXllmMmData(request->mm_data(), &mm, &mm_owned)) {
        SetErrorResponse(response,
                         c_api_test::XLLM_STATUS_INVALID_REQUEST,
                         "invalid or empty mm_data");
        return;
      }
      raw = xllm_rec_multimodal_completions(
          rec_handler_,
          model_id,
          token_ids.empty() ? nullptr : token_ids.data(),
          token_ids.size(),
          &mm,
          timeout_ms,
          &params);
    } else if (fn == "xllm_rec_chat_completions") {
      std::vector<XLLM_ChatMessage> cms;
      std::vector<std::unique_ptr<char[]>> contents;
      cms.reserve(request->messages_size());
      for (int i = 0; i < request->messages_size(); ++i) {
        const auto& m = request->messages(i);
        XLLM_ChatMessage cm{};
        std::memset(cm.role, 0, sizeof(cm.role));
        std::strncpy(
            cm.role, m.role().c_str(), XLLM_META_STRING_FIELD_MAX_LEN - 1);
        cm.role[XLLM_META_STRING_FIELD_MAX_LEN - 1] = '\0';
        contents.push_back(CopyContent(m.content()));
        cm.content = contents.back() ? contents.back().get() : nullptr;
        cms.push_back(cm);
      }
      raw = xllm_rec_chat_completions(rec_handler_,
                                      model_id,
                                      cms.empty() ? nullptr : cms.data(),
                                      cms.size(),
                                      timeout_ms,
                                      &params);
    } else {
      SetErrorResponse(response,
                       c_api_test::XLLM_STATUS_INVALID_REQUEST,
                       "unsupported call_function: " + fn);
      return;
    }

    if (raw == nullptr) {
      SetErrorResponse(response,
                       c_api_test::XLLM_STATUS_INTERNAL_ERROR,
                       "C API returned null response");
      return;
    }

    XllmResponseToPb(raw, response);
    if (is_llm_fn) {
      xllm_llm_free_response(raw);
    } else {
      xllm_rec_free_response(raw);
    }
  }

 private:
  XLLM_REC_Handler* rec_handler_;
  XLLM_LLM_Handler* llm_handler_;
  std::mutex mu_;
};

}  // namespace xllm_capi_test

int main(int argc, char* argv[]) {
  GFLAGS_NAMESPACE::ParseCommandLineFlags(&argc, &argv, true);

  if (FLAGS_model_path.empty()) {
    LOG(ERROR) << "Missing --model_path (set in gflags file or command line)";
    return -1;
  }

  std::string backend = FLAGS_backend;
  for (char& c : backend) {
    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  }
  if (backend != "llm" && backend != "rec") {
    LOG(ERROR) << "Invalid --backend=\"" << FLAGS_backend
               << "\" (expected llm or rec)";
    return -2;
  }

  XLLM_REC_Handler* rec_ptr = nullptr;
  XLLM_LLM_Handler* llm_ptr = nullptr;
  std::unique_ptr<XLLM_REC_Handler, void (*)(XLLM_REC_Handler*)> rec_holder(
      nullptr, xllm_rec_destroy);
  std::unique_ptr<XLLM_LLM_Handler, void (*)(XLLM_LLM_Handler*)> llm_holder(
      nullptr, xllm_llm_destroy);

  if (backend == "rec") {
    rec_holder.reset(xllm_rec_create());
    if (!rec_holder) {
      LOG(ERROR) << "xllm_rec_create failed";
      return -3;
    }
    XLLM_InitOptions init{};
    xllm_rec_init_options_default(&init);
    xllm_capi_test::ApplyGflagsToXllmInitOptions(&init);
    if (!xllm_rec_initialize(
            rec_holder.get(), FLAGS_model_path.c_str(), &init)) {
      LOG(ERROR) << "xllm_rec_initialize failed model_path="
                 << FLAGS_model_path;
      return -4;
    }
    rec_ptr = rec_holder.get();
  } else {
    llm_holder.reset(xllm_llm_create());
    if (!llm_holder) {
      LOG(ERROR) << "xllm_llm_create failed";
      return -5;
    }
    XLLM_InitOptions init{};
    xllm_llm_init_options_default(&init);
    xllm_capi_test::ApplyGflagsToXllmInitOptions(&init);
    if (!xllm_llm_initialize(
            llm_holder.get(), FLAGS_model_path.c_str(), &init)) {
      LOG(ERROR) << "xllm_llm_initialize failed model_path="
                 << FLAGS_model_path;
      return -6;
    }
    llm_ptr = llm_holder.get();
  }

  xllm_capi_test::XllmRecCapiServiceImpl svc(rec_ptr, llm_ptr);
  brpc::Server server;
  if (server.AddService(&svc, brpc::SERVER_DOESNT_OWN_SERVICE) != 0) {
    LOG(ERROR) << "Fail to add XllmRecCapiService";
    return -7;
  }

  butil::EndPoint point;
  if (!FLAGS_listen_addr.empty()) {
    if (butil::str2endpoint(FLAGS_listen_addr.c_str(), &point) < 0) {
      LOG(ERROR) << "Invalid --listen_addr=" << FLAGS_listen_addr;
      return -8;
    }
  } else {
    point = butil::EndPoint(butil::IP_ANY, FLAGS_port);
  }

  brpc::ServerOptions options;
  options.idle_timeout_sec = FLAGS_idle_timeout_s;

  if (server.Start(point, &options) != 0) {
    LOG(ERROR) << "Fail to start brpc server";
    return -9;
  }

  LOG(INFO) << "xllm_test C API brpc server backend=" << backend
            << " listening on " << butil::endpoint2str(point).c_str();
  server.RunUntilAskedToQuit();
  return 0;
}

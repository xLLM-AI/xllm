// Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.

#pragma once

#include <dlfcn.h>
#include <glog/logging.h>

#include <filesystem>

namespace xllm_kernel {

struct AscendcApi {
  void* launch;
  void* workspace;
};

inline AscendcApi find_ascendc_api(const char* launch_name,
                                   const char* workspace_name) {
  static void* const library = [] {
    Dl_info binding{};
    CHECK(dladdr(reinterpret_cast<const void*>(&find_ascendc_api), &binding));
    const auto path = std::filesystem::path(binding.dli_fname).parent_path() /
                      "ascendc_opp/xllm_kernel/op_api/lib/libcust_opapi.so";
    // Keep the package API and its CANN registrations alive for the process.
    void* handle = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
    CHECK(handle) << path << ": " << dlerror();
    return handle;
  }();
  return {dlsym(library, launch_name), dlsym(library, workspace_name)};
}

inline AscendcApi load_ascendc_api(const char* launch_name,
                                   const char* workspace_name) {
  const auto api = find_ascendc_api(launch_name, workspace_name);
  CHECK(api.launch && api.workspace) << launch_name << " in xllm-kernel OPP";
  return api;
}

}  // namespace xllm_kernel

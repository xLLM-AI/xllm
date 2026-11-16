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

#include "xllm_kernel/native/runtime.h"

#include <dlfcn.h>

#include <algorithm>
#include <cerrno>
#include <cstdlib>
#include <string>
#include <system_error>
#include <vector>

namespace xllm_kernel::native {
std::filesystem::path resource_root() {
  Dl_info location{};
  if (::dladdr(reinterpret_cast<const void*>(&resource_root), &location) == 0 ||
      location.dli_fname == nullptr) {
    throw std::system_error(
        std::make_error_code(std::errc::no_such_file_or_directory),
        "cannot locate xllm-kernel native library");
  }
  return std::filesystem::canonical(location.dli_fname).parent_path() / "opp";
}
void configure_resources() {
  const std::filesystem::path vendors = resource_root() / "vendors";
  if (!std::filesystem::is_directory(vendors)) {
    return;
  }
  std::vector<std::string> roots;
  for (const auto& vendor : std::filesystem::directory_iterator(vendors)) {
    if (vendor.is_directory()) {
      roots.emplace_back(vendor.path().string());
    }
  }
  std::sort(roots.begin(), roots.end());
  const char* existing = std::getenv("ASCEND_CUSTOM_OPP_PATH");
  const std::string paths = existing == nullptr ? "" : existing;
  size_t start = 0;
  while (start < paths.size()) {
    const size_t end = paths.find(':', start);
    const std::string path = paths.substr(start, end - start);
    if (!path.empty() &&
        std::find(roots.begin(), roots.end(), path) == roots.end()) {
      roots.emplace_back(path);
    }
    if (end == std::string::npos) {
      break;
    }
    start = end + 1;
  }
  std::string configured;
  for (const auto& root : roots) {
    if (!configured.empty()) {
      configured += ':';
    }
    configured += root;
  }
  if (::setenv("ASCEND_CUSTOM_OPP_PATH", configured.c_str(), /*overwrite=*/1) !=
      0) {
    throw std::system_error(errno,
                            std::generic_category(),
                            "cannot register xllm-kernel resources");
  }
}
}  // namespace xllm_kernel::native

extern "C" int32_t xllm_kernel_native_abi_version() { return 1; }

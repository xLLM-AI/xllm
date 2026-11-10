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

#include "core/common/native_library.h"

#include <dlfcn.h>
#include <gtest/gtest.h>

#include <cstdlib>
#include <string>

namespace xllm {
namespace {

TEST(NativeLibraryTest, NoConfiguredLibrariesPreservesStartup) {
  EXPECT_EXIT(
      {
        unsetenv("XLLM_NATIVE_LIBRARIES");
        load_native_libraries();
        std::exit(0);
      },
      testing::ExitedWithCode(0),
      "");
}

TEST(NativeLibraryTest, LoadsCompatibleLibraryAndKeepsItResident) {
  EXPECT_EXIT(
      {
        setenv("XLLM_NATIVE_LIBRARIES", NATIVE_VALID_FIXTURE, 1);
        load_native_libraries();
        load_native_libraries();
        void* handle = dlopen(NATIVE_VALID_FIXTURE, RTLD_NOLOAD | RTLD_NOW);
        if (handle == nullptr) {
          std::exit(1);
        }
        using ValueGetter = int32_t (*)();
        auto getter = reinterpret_cast<ValueGetter>(
            dlsym(handle, "xllm_native_library_fixture_value"));
        const bool valid = getter != nullptr && getter() == 42;
        dlclose(handle);
        std::exit(valid ? 0 : 2);
      },
      testing::ExitedWithCode(0),
      "");
}

TEST(NativeLibraryTest, AcceptsRepeatedCanonicalPaths) {
  EXPECT_EXIT(
      {
        const std::string paths =
            std::string(NATIVE_VALID_FIXTURE) + ":" + NATIVE_VALID_FIXTURE;
        setenv("XLLM_NATIVE_LIBRARIES", paths.c_str(), 1);
        load_native_libraries();
        std::exit(0);
      },
      testing::ExitedWithCode(0),
      "");
}

TEST(NativeLibraryTest, RejectsMissingLibraries) {
  EXPECT_DEATH(
      {
        setenv("XLLM_NATIVE_LIBRARIES", "/missing/xllm/native-library.so", 1);
        load_native_libraries();
      },
      "Cannot resolve native library");
}

TEST(NativeLibraryTest, RejectsRelativePaths) {
  EXPECT_DEATH(
      {
        setenv("XLLM_NATIVE_LIBRARIES", "relative-native-library.so", 1);
        load_native_libraries();
      },
      "requires absolute, nonempty paths");
}

TEST(NativeLibraryTest, RejectsMissingBuildDescriptors) {
  EXPECT_DEATH(
      {
        setenv("XLLM_NATIVE_LIBRARIES", NATIVE_NO_DESCRIPTOR_FIXTURE, 1);
        load_native_libraries();
      },
      "does not define its own build descriptor");
}

TEST(NativeLibraryTest, RejectsIncompatibleBuildDescriptors) {
  EXPECT_DEATH(
      {
        setenv("XLLM_NATIVE_LIBRARIES", NATIVE_MISMATCH_FIXTURE, 1);
        load_native_libraries();
      },
      "incompatible source revision");
}

TEST(NativeLibraryTest, RejectsEmptyListEntries) {
  EXPECT_DEATH(
      {
        const std::string paths = std::string(NATIVE_VALID_FIXTURE) + ":";
        setenv("XLLM_NATIVE_LIBRARIES", paths.c_str(), 1);
        load_native_libraries();
      },
      "empty trailing path");
}

TEST(NativeLibraryTest, RejectsIncompatibleDescriptorVersions) {
  EXPECT_DEATH(
      {
        setenv("XLLM_NATIVE_LIBRARIES", NATIVE_VERSION_MISMATCH_FIXTURE, 1);
        load_native_libraries();
      },
      "incompatible descriptor version");
}

}  // namespace
}  // namespace xllm

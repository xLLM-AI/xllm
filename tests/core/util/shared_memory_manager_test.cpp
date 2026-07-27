/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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

#include "core/util/shared_memory_manager.h"

#include <fcntl.h>
#include <gtest/gtest.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

#include <array>
#include <atomic>
#include <cerrno>
#include <csignal>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <thread>

namespace xllm {
namespace {

TEST(SharedMemoryManagerTest, AttachingPreservesExistingContents) {
  const std::string name =
      "xllm_shm_attach_test_" + std::to_string(static_cast<int64_t>(getpid()));
  constexpr size_t kSize = 4096;

  bool writer_is_creator = false;
  SharedMemoryManager writer(name, kSize, writer_is_creator);
  ASSERT_TRUE(writer_is_creator);
  auto* writer_bytes = static_cast<uint8_t*>(writer.base_address());
  writer_bytes[0] = 0x5A;
  writer_bytes[kSize - 1] = 0xA5;

  bool reader_is_creator = true;
  SharedMemoryManager reader(name, kSize, reader_is_creator);
  ASSERT_FALSE(reader_is_creator);
  const auto* reader_bytes = static_cast<const uint8_t*>(reader.base_address());
  EXPECT_EQ(reader_bytes[0], 0x5A);
  EXPECT_EQ(reader_bytes[kSize - 1], 0xA5);
}

TEST(SharedMemoryManagerTest, ReclaimsStaleSegmentWithoutActiveOwner) {
  const std::string name =
      "xllm_shm_stale_test_" + std::to_string(static_cast<int64_t>(getpid()));
  constexpr size_t kSize = 4096;
  shm_unlink(name.c_str());

  const pid_t child = fork();
  ASSERT_NE(child, -1);
  if (child == 0) {
    bool child_is_creator = false;
    SharedMemoryManager manager(name, kSize, child_is_creator);
    if (!child_is_creator) {
      _exit(2);
    }
    std::memset(manager.base_address(), 0xA5, kSize);
    _exit(0);
  }
  int32_t child_status = 0;
  ASSERT_EQ(waitpid(child, &child_status, 0), child);
  ASSERT_TRUE(WIFEXITED(child_status));
  ASSERT_EQ(WEXITSTATUS(child_status), 0);

  bool is_creator = false;
  SharedMemoryManager manager(name, kSize, is_creator);
  EXPECT_TRUE(is_creator);
  const auto* bytes = static_cast<const uint8_t*>(manager.base_address());
  EXPECT_EQ(bytes[0], 0);
  EXPECT_EQ(bytes[kSize - 1], 0);
}

TEST(SharedMemoryManagerTest, RejectsLegacyLayoutWithoutChangingIt) {
  const std::string name =
      "xllm_shm_legacy_test_" + std::to_string(static_cast<int64_t>(getpid()));
  constexpr size_t kSize = 4096;
  shm_unlink(name.c_str());

  const int32_t legacy_fd = shm_open(name.c_str(), O_CREAT | O_RDWR, 0600);
  ASSERT_NE(legacy_fd, -1);
  ASSERT_EQ(ftruncate(legacy_fd, kSize), 0);
  auto* legacy_address = static_cast<uint8_t*>(
      mmap(nullptr, kSize, PROT_READ | PROT_WRITE, MAP_SHARED, legacy_fd, 0));
  ASSERT_NE(legacy_address, MAP_FAILED);
  legacy_address[0] = 0x5A;

  EXPECT_EXIT(
      {
        bool is_creator = false;
        SharedMemoryManager manager(name, kSize, is_creator);
      },
      testing::KilledBySignal(SIGABRT),
      "incompatible shared memory layout");

  struct stat legacy_stat;
  ASSERT_EQ(fstat(legacy_fd, &legacy_stat), 0);
  EXPECT_EQ(legacy_stat.st_size, static_cast<off_t>(kSize));
  EXPECT_EQ(legacy_address[0], 0x5A);
  ASSERT_EQ(munmap(legacy_address, kSize), 0);
  ASSERT_EQ(close(legacy_fd), 0);
  ASSERT_EQ(shm_unlink(name.c_str()), 0);
}

TEST(SharedMemoryManagerTest, DoesNotCreatePerSegmentLockObjects) {
  const std::string name =
      "xllm_shm_lock_test_" + std::to_string(static_cast<int64_t>(getpid()));
  constexpr size_t kSize = 4096;
  shm_unlink(name.c_str());
  shm_unlink((name + ".lock").c_str());

  bool is_creator = false;
  {
    SharedMemoryManager manager(name, kSize, is_creator);
    ASSERT_TRUE(is_creator);
  }

  errno = 0;
  const int32_t lock_fd = shm_open((name + ".lock").c_str(), O_RDONLY, 0600);
  EXPECT_EQ(lock_fd, -1);
  EXPECT_EQ(errno, ENOENT);
}

TEST(SharedMemoryManagerTest, ConcurrentLastOwnerHandoffKeepsOneGeneration) {
  const std::string name =
      "xllm_shm_handoff_test_" + std::to_string(static_cast<int64_t>(getpid()));
  constexpr size_t kSize = 4096;
  constexpr int32_t kThreads = 8;
  constexpr int32_t kIterations = 128;

  shm_unlink(name.c_str());
  for (int32_t iteration = 0; iteration < kIterations; ++iteration) {
    bool owner_is_creator = false;
    auto owner =
        std::make_unique<SharedMemoryManager>(name, kSize, owner_is_creator);
    ASSERT_TRUE(owner_is_creator);

    std::array<std::unique_ptr<SharedMemoryManager>, kThreads> managers;
    std::array<bool, kThreads> creator_flags{};
    std::array<std::thread, kThreads> threads;
    std::atomic<int32_t> ready{0};
    std::atomic<bool> start{false};
    for (int32_t thread_index = 0; thread_index < kThreads; ++thread_index) {
      threads[thread_index] = std::thread([&, thread_index] {
        ready.fetch_add(1, std::memory_order_release);
        while (!start.load(std::memory_order_acquire)) {
          std::this_thread::yield();
        }
        managers[thread_index] = std::make_unique<SharedMemoryManager>(
            name, kSize, creator_flags[thread_index]);
      });
    }
    while (ready.load(std::memory_order_acquire) != kThreads) {
      std::this_thread::yield();
    }
    start.store(true, std::memory_order_release);
    owner.reset();
    for (std::thread& thread : threads) {
      thread.join();
    }

    auto* first = static_cast<uint8_t*>(managers[0]->base_address());
    first[0] = static_cast<uint8_t>(iteration);
    for (const auto& manager : managers) {
      const auto* bytes = static_cast<const uint8_t*>(manager->base_address());
      EXPECT_EQ(bytes[0], static_cast<uint8_t>(iteration));
    }
  }
}

}  // namespace
}  // namespace xllm

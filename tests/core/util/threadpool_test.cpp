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

#include "threadpool.h"

#include <absl/synchronization/notification.h>
#include <absl/time/clock.h>
#include <gtest/gtest.h>
#include <pthread.h>
#include <sched.h>

namespace xllm {

TEST(ThreadPoolTest, ScheduleEmptyTask) {
  ThreadPool threadpool(1);
  absl::Notification notification;
  threadpool.schedule(nullptr);
}

TEST(ThreadPoolTest, ScheduleTask) {
  ThreadPool threadpool(1);
  absl::Notification notification;
  bool called = false;
  threadpool.schedule([&called, &notification]() {
    called = true;
    notification.Notify();
  });
  notification.WaitForNotification();
  EXPECT_TRUE(called);
}

TEST(ThreadPoolTest, ScheduleMultipleTasks) {
  ThreadPool threadpool(1);
  std::vector<std::string> completed_tasks;
  absl::Notification notification;

  // run frist task
  threadpool.schedule([&completed_tasks, &notification]() {
    completed_tasks.emplace_back("first");
    if (completed_tasks.size() == 2) {
      absl::SleepFor(absl::Milliseconds(100));
      notification.Notify();
    }
  });

  // run second task
  threadpool.schedule([&completed_tasks, &notification]() {
    completed_tasks.emplace_back("second");
    if (completed_tasks.size() == 2) {
      notification.Notify();
    }
  });

  notification.WaitForNotificationWithTimeout(absl::Milliseconds(200));
  EXPECT_EQ(completed_tasks.size(), 2);
  EXPECT_EQ(completed_tasks[0], "first");
  EXPECT_EQ(completed_tasks[1], "second");
}

TEST(ThreadPoolTest, MultipleThreads) {
  ThreadPool threadpool(4);
  std::atomic_uint32_t counter = 0;
  absl::Notification notification;

  for (int i = 0; i < 10; ++i) {
    threadpool.schedule([&counter, &notification]() {
      absl::SleepFor(absl::Milliseconds(100));
      counter++;
      if (counter == 10) {
        notification.Notify();
      }
    });
  }

  EXPECT_TRUE(
      notification.WaitForNotificationWithTimeout(absl::Milliseconds(400)));
  EXPECT_EQ(counter, 10);
}

TEST(ThreadPoolTest, CpuCoreBindingConstructor) {
  // Construct with cpu_cores binding — should not crash even if binding fails
  // (e.g., in containers with restricted affinity).
  std::vector<int32_t> cpu_cores = {0, 0};  // bind both threads to core 0
  ThreadPool threadpool(2, nullptr, cpu_cores);
  EXPECT_EQ(threadpool.size(), 2);

  std::atomic<int> counter{0};
  absl::Notification notification;
  for (int i = 0; i < 2; ++i) {
    threadpool.schedule([&counter, &notification]() {
      if (++counter == 2) {
        notification.Notify();
      }
    });
  }
  EXPECT_TRUE(
      notification.WaitForNotificationWithTimeout(absl::Milliseconds(500)));
  EXPECT_EQ(counter, 2);
}

TEST(ThreadPoolTest, CpuCoreBindingWithInitFunc) {
  std::vector<int32_t> cpu_cores = {0};
  std::atomic<bool> init_called{false};
  absl::Notification init_done;
  ThreadPool threadpool(
      1,
      [&init_called, &init_done]() {
        init_called = true;
        init_done.Notify();
      },
      cpu_cores);
  EXPECT_TRUE(
      init_done.WaitForNotificationWithTimeout(absl::Milliseconds(500)));
  EXPECT_TRUE(init_called);
}

TEST(ThreadPoolTest, CpuCoreBindingMismatchFallback) {
  // Mismatched cpu_cores size — should fall back to no binding gracefully.
  std::vector<int32_t> cpu_cores = {0, 1};  // 2 cores but 4 threads
  ThreadPool threadpool(4, nullptr, cpu_cores);
  EXPECT_EQ(threadpool.size(), 4);

  std::atomic<int> counter{0};
  absl::Notification notification;
  for (int i = 0; i < 4; ++i) {
    threadpool.schedule([&counter, &notification]() {
      if (++counter == 4) {
        notification.Notify();
      }
    });
  }
  EXPECT_TRUE(
      notification.WaitForNotificationWithTimeout(absl::Milliseconds(500)));
  EXPECT_EQ(counter, 4);
}

TEST(ThreadPoolTest, CpuCoreBindingVerifyAffinity) {
  // Verify that after construction the thread is actually bound to the
  // requested core (if the system allows it).
  const int32_t target_core = 0;
  std::vector<int32_t> cpu_cores = {target_core};

  absl::Notification done;
  std::atomic<bool> affinity_ok{false};

  ThreadPool threadpool(1, nullptr, cpu_cores);
  threadpool.schedule([&done, &affinity_ok, target_core]() {
    cpu_set_t cpu_set;
    CPU_ZERO(&cpu_set);
    if (pthread_getaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpu_set) ==
        0) {
      affinity_ok = CPU_ISSET(target_core, &cpu_set);
    }
    done.Notify();
  });
  EXPECT_TRUE(done.WaitForNotificationWithTimeout(absl::Milliseconds(500)));
  EXPECT_TRUE(affinity_ok);
}

}  // namespace xllm

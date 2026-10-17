/* Copyright 2026 The xLLM Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <future>
#include <limits>
#include <vector>

#include "core/scheduler/scheduler_notifier_test_peer.h"

namespace xllm {

namespace {

constexpr auto kWaitTimeout = std::chrono::seconds(5);
constexpr auto kWakeDeadline = std::chrono::seconds(1);

TEST(SchedulerNotifierTest, BusySchedulerPreservesEventsWithoutNotification) {
  SchedulerNotifier notifier;
  const auto initial = notifier.snapshot();

  constexpr uint64_t kEventCount = 16;
  for (uint64_t i = 0; i < kEventCount; ++i) {
    EXPECT_FALSE(notifier.notify());
  }

  const auto current = notifier.snapshot();
  EXPECT_EQ(current.generation, initial.generation + kEventCount);
  EXPECT_FALSE(current.stop_requested);
}

TEST(SchedulerNotifierTest, BusyPathsCompleteWhileWaitMutexIsHeld) {
  SchedulerNotifier notifier;
  auto wait_lock = SchedulerNotifierTestPeer::lock_wait_mutex(notifier);
  auto producer = std::async(std::launch::async, [&] {
    const auto observed = notifier.snapshot();
    const bool notified = notifier.notify();
    const auto current = notifier.snapshot();
    const bool stopped = notifier.wait_for(observed.generation, kWaitTimeout);
    return !notified && !stopped &&
           current.generation == observed.generation + 1;
  });

  const std::future_status status = producer.wait_for(kWakeDeadline);
  wait_lock.unlock();

  EXPECT_EQ(status, std::future_status::ready);
  EXPECT_TRUE(producer.get());
}

TEST(SchedulerNotifierTest, StoppedWaitCompletesWhileWaitMutexIsHeld) {
  SchedulerNotifier notifier;
  notifier.request_stop();
  const auto observed = notifier.snapshot();
  auto wait_lock = SchedulerNotifierTestPeer::lock_wait_mutex(notifier);
  auto waiter = std::async(std::launch::async, [&] {
    return notifier.wait_for(observed.generation, kWaitTimeout);
  });

  const std::future_status status = waiter.wait_for(kWakeDeadline);
  wait_lock.unlock();

  EXPECT_EQ(status, std::future_status::ready);
  EXPECT_TRUE(waiter.get());
}

TEST(SchedulerNotifierTest, EventBeforeWaitPreventsSleeping) {
  SchedulerNotifier notifier;
  const auto observed = notifier.snapshot();
  EXPECT_FALSE(notifier.notify());

  auto waiter = std::async(std::launch::async, [&] {
    return notifier.wait_for(observed.generation, kWaitTimeout);
  });
  EXPECT_EQ(waiter.wait_for(kWakeDeadline), std::future_status::ready);
  notifier.request_stop();
  EXPECT_FALSE(waiter.get());
}

TEST(SchedulerNotifierTest, NotificationWakesBlockedWaiter) {
  SchedulerNotifier notifier;
  const auto observed = notifier.snapshot();
  auto waiter = std::async(std::launch::async, [&] {
    return notifier.wait_for(observed.generation, kWaitTimeout);
  });

  EXPECT_TRUE(SchedulerNotifierTestPeer::wait_until_armed(notifier));
  EXPECT_TRUE(notifier.notify());
  EXPECT_EQ(waiter.wait_for(kWakeDeadline), std::future_status::ready);
  notifier.request_stop();
  EXPECT_FALSE(waiter.get());
}

TEST(SchedulerNotifierTest, WaitRegistrationPreservesPublicGeneration) {
  SchedulerNotifier notifier;
  const auto observed = notifier.snapshot();
  auto waiter = std::async(std::launch::async, [&] {
    return notifier.wait_for(observed.generation, kWaitTimeout);
  });
  EXPECT_TRUE(SchedulerNotifierTestPeer::wait_until_armed(notifier));

  const auto registered = notifier.snapshot();
  EXPECT_EQ(registered.generation, observed.generation);
  EXPECT_FALSE(registered.stop_requested);
  notifier.request_stop();
  EXPECT_TRUE(waiter.get());
}

TEST(SchedulerNotifierTest, EventIsPublishedBeforeSlowPathMutexHandshake) {
  SchedulerNotifier notifier;
  const auto observed = notifier.snapshot();
  auto waiter = std::async(std::launch::async, [&] {
    return notifier.wait_for(observed.generation, kWaitTimeout);
  });
  EXPECT_TRUE(SchedulerNotifierTestPeer::wait_until_armed(notifier));

  auto wait_lock = SchedulerNotifierTestPeer::lock_wait_mutex(notifier);
  EXPECT_TRUE(SchedulerNotifierTestPeer::is_armed(notifier));
  auto producer =
      std::async(std::launch::async, [&] { return notifier.notify(); });
  const bool published =
      SchedulerNotifierTestPeer::wait_until_generation_changes(
          notifier, observed.generation);
  const std::future_status before_unlock =
      producer.wait_for(std::chrono::nanoseconds::zero());
  wait_lock.unlock();

  EXPECT_TRUE(published);
  EXPECT_EQ(before_unlock, std::future_status::timeout);
  EXPECT_EQ(producer.wait_for(kWakeDeadline), std::future_status::ready);
  EXPECT_TRUE(producer.get());
  EXPECT_EQ(waiter.wait_for(kWakeDeadline), std::future_status::ready);
  notifier.request_stop();
  EXPECT_FALSE(waiter.get());
}

TEST(SchedulerNotifierTest, StopIsPublishedBeforeSlowPathMutexHandshake) {
  SchedulerNotifier notifier;
  const auto observed = notifier.snapshot();
  auto waiter = std::async(std::launch::async, [&] {
    return notifier.wait_for(observed.generation, kWaitTimeout);
  });
  EXPECT_TRUE(SchedulerNotifierTestPeer::wait_until_armed(notifier));

  auto wait_lock = SchedulerNotifierTestPeer::lock_wait_mutex(notifier);
  EXPECT_TRUE(SchedulerNotifierTestPeer::is_armed(notifier));
  auto stopper =
      std::async(std::launch::async, [&] { notifier.request_stop(); });
  const bool published =
      SchedulerNotifierTestPeer::wait_until_stopped(notifier);
  const std::future_status before_unlock =
      stopper.wait_for(std::chrono::nanoseconds::zero());
  wait_lock.unlock();

  EXPECT_TRUE(published);
  EXPECT_EQ(before_unlock, std::future_status::timeout);
  EXPECT_EQ(stopper.wait_for(kWakeDeadline), std::future_status::ready);
  stopper.get();
  EXPECT_EQ(waiter.wait_for(kWakeDeadline), std::future_status::ready);
  EXPECT_TRUE(waiter.get());
  EXPECT_TRUE(notifier.snapshot().stop_requested);
}

TEST(SchedulerNotifierTest, ConcurrentEventsShareBlockedWaiterNotification) {
  SchedulerNotifier notifier;
  const auto observed = notifier.snapshot();
  auto waiter = std::async(std::launch::async, [&] {
    return notifier.wait_for(observed.generation, kWaitTimeout);
  });
  EXPECT_TRUE(SchedulerNotifierTestPeer::wait_until_armed(notifier));

  constexpr uint64_t kProducerCount = 8;
  std::promise<void> start;
  const std::shared_future<void> start_signal = start.get_future().share();
  std::vector<std::future<bool>> producers;
  producers.reserve(kProducerCount);
  for (uint64_t i = 0; i < kProducerCount; ++i) {
    producers.emplace_back(std::async(std::launch::async, [&] {
      start_signal.wait();
      return notifier.notify();
    }));
  }
  start.set_value();

  uint64_t notification_count = 0;
  for (auto& producer : producers) {
    notification_count += static_cast<uint64_t>(producer.get());
  }
  EXPECT_EQ(notification_count, 1);
  EXPECT_EQ(notifier.snapshot().generation,
            observed.generation + kProducerCount);
  EXPECT_EQ(waiter.wait_for(kWakeDeadline), std::future_status::ready);
  notifier.request_stop();
  EXPECT_FALSE(waiter.get());
}

TEST(SchedulerNotifierTest, WaitRegistrationIsRearmedAfterEachWakeup) {
  SchedulerNotifier notifier;
  constexpr uint64_t kWaitCount = 8;
  for (uint64_t i = 0; i < kWaitCount; ++i) {
    const auto observed = notifier.snapshot();
    auto waiter = std::async(std::launch::async, [&] {
      return notifier.wait_for(observed.generation, kWaitTimeout);
    });
    EXPECT_TRUE(SchedulerNotifierTestPeer::wait_until_armed(notifier));
    EXPECT_TRUE(notifier.notify());
    const std::future_status status = waiter.wait_for(kWakeDeadline);
    EXPECT_EQ(status, std::future_status::ready);
    if (status != std::future_status::ready) {
      notifier.request_stop();
    }
    EXPECT_FALSE(waiter.get());
    EXPECT_FALSE(notifier.notify());
  }
}

TEST(SchedulerNotifierTest, TimeoutClearsWaitRegistration) {
  SchedulerNotifier notifier;
  const auto observed = notifier.snapshot();

  const auto before_wait = std::chrono::steady_clock::now();
  EXPECT_FALSE(
      notifier.wait_for(observed.generation, std::chrono::milliseconds(1)));
  EXPECT_GE(std::chrono::steady_clock::now() - before_wait,
            std::chrono::milliseconds(1));
  EXPECT_FALSE(notifier.notify());
  EXPECT_EQ(notifier.snapshot().generation, observed.generation + 1);
}

TEST(SchedulerNotifierTest, TimedOutWaitCanBeRegisteredAgain) {
  SchedulerNotifier notifier;
  const auto observed = notifier.snapshot();
  EXPECT_FALSE(
      notifier.wait_for(observed.generation, std::chrono::milliseconds(1)));

  auto waiter = std::async(std::launch::async, [&] {
    return notifier.wait_for(observed.generation, kWaitTimeout);
  });
  EXPECT_TRUE(SchedulerNotifierTestPeer::wait_until_armed(notifier));
  EXPECT_TRUE(notifier.notify());
  EXPECT_EQ(waiter.wait_for(kWakeDeadline), std::future_status::ready);
  notifier.request_stop();
  EXPECT_FALSE(waiter.get());
}

TEST(SchedulerNotifierTest, ShutdownWakesBlockedWaiter) {
  SchedulerNotifier notifier;
  const auto observed = notifier.snapshot();
  auto waiter = std::async(std::launch::async, [&] {
    return notifier.wait_for(observed.generation, kWaitTimeout);
  });
  EXPECT_TRUE(SchedulerNotifierTestPeer::wait_until_armed(notifier));

  notifier.request_stop();

  EXPECT_EQ(waiter.wait_for(kWakeDeadline), std::future_status::ready);
  EXPECT_TRUE(waiter.get());
  EXPECT_TRUE(notifier.snapshot().stop_requested);
  EXPECT_FALSE(notifier.notify());
}

TEST(SchedulerNotifierTest, ShutdownBeforeWaitPreventsSleeping) {
  SchedulerNotifier notifier;
  const auto observed = notifier.snapshot();
  notifier.request_stop();

  auto waiter = std::async(std::launch::async, [&] {
    return notifier.wait_for(observed.generation, kWaitTimeout);
  });

  EXPECT_EQ(waiter.wait_for(kWakeDeadline), std::future_status::ready);
  EXPECT_TRUE(waiter.get());
}

TEST(SchedulerNotifierTest, NotificationsPreserveStopRequest) {
  SchedulerNotifier notifier;
  notifier.request_stop();
  const auto stopped = notifier.snapshot();

  constexpr uint64_t kEventCount = 8;
  for (uint64_t i = 0; i < kEventCount; ++i) {
    EXPECT_FALSE(notifier.notify());
  }

  const auto current = notifier.snapshot();
  EXPECT_TRUE(current.stop_requested);
  EXPECT_EQ(current.generation, stopped.generation + kEventCount);
  EXPECT_TRUE(notifier.wait_for(current.generation, kWaitTimeout));
}

TEST(SchedulerNotifierTest, ConcurrentStopAndNotificationsPreserveEvents) {
  SchedulerNotifier notifier;
  const auto observed = notifier.snapshot();
  std::promise<void> start;
  const std::shared_future<void> start_signal = start.get_future().share();
  auto stopper = std::async(std::launch::async, [&] {
    start_signal.wait();
    notifier.request_stop();
  });
  constexpr uint64_t kProducerCount = 8;
  std::vector<std::future<bool>> producers;
  producers.reserve(kProducerCount);
  for (uint64_t i = 0; i < kProducerCount; ++i) {
    producers.emplace_back(std::async(std::launch::async, [&] {
      start_signal.wait();
      return notifier.notify();
    }));
  }
  start.set_value();

  for (auto& producer : producers) {
    EXPECT_FALSE(producer.get());
  }
  stopper.get();

  const auto current = notifier.snapshot();
  EXPECT_TRUE(current.stop_requested);
  EXPECT_EQ(current.generation, observed.generation + kProducerCount + 1);
}

TEST(SchedulerNotifierTest, GenerationWraparoundWakesBlockedWaiter) {
  SchedulerNotifier notifier;
  SchedulerNotifierTestPeer::prepare_generation_wraparound(notifier);
  const auto observed = notifier.snapshot();
  EXPECT_EQ(observed.generation, std::numeric_limits<uint64_t>::max() / 4);
  auto waiter = std::async(std::launch::async, [&] {
    return notifier.wait_for(observed.generation, kWaitTimeout);
  });
  EXPECT_TRUE(SchedulerNotifierTestPeer::wait_until_armed(notifier));

  EXPECT_TRUE(notifier.notify());

  EXPECT_EQ(waiter.wait_for(kWakeDeadline), std::future_status::ready);
  EXPECT_EQ(notifier.snapshot().generation, 0);
  EXPECT_FALSE(notifier.snapshot().stop_requested);
  notifier.request_stop();
  EXPECT_FALSE(waiter.get());
}

TEST(SchedulerNotifierTest, GenerationWraparoundPreservesStopRequest) {
  SchedulerNotifier notifier;
  SchedulerNotifierTestPeer::prepare_generation_wraparound(notifier);

  notifier.request_stop();

  const auto stopped = notifier.snapshot();
  EXPECT_EQ(stopped.generation, 0);
  EXPECT_TRUE(stopped.stop_requested);
  EXPECT_FALSE(notifier.notify());
  const auto notified = notifier.snapshot();
  EXPECT_EQ(notified.generation, 1);
  EXPECT_TRUE(notified.stop_requested);
}

TEST(SchedulerNotifierTest, NotificationRacingWithWaitDoesNotLoseEvent) {
  SchedulerNotifier notifier;
  constexpr uint64_t kAttemptCount = 64;
  for (uint64_t i = 0; i < kAttemptCount; ++i) {
    const auto observed = notifier.snapshot();
    auto waiter = std::async(std::launch::async, [&] {
      return notifier.wait_for(observed.generation, kWaitTimeout);
    });
    notifier.notify();

    const std::future_status status = waiter.wait_for(kWakeDeadline);
    EXPECT_EQ(status, std::future_status::ready);
    if (status != std::future_status::ready) {
      notifier.request_stop();
    }
    EXPECT_FALSE(waiter.get());
  }
}

}  // namespace
}  // namespace xllm

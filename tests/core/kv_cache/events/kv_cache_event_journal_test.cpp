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

#include "core/kv_cache/events/kv_cache_event_journal.h"

#include <glog/logging.h>
#include <gtest/gtest.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
#include <zmq.h>

#include <array>
#include <atomic>
#include <chrono>
#include <memory>
#include <optional>
#include <thread>

#include "core/kv_cache/events/kv_cache_event_publisher.h"

extern "C" int __real_zmq_msg_recv(zmq_msg_t* message, void* socket, int flags);

namespace xllm {
namespace {
const std::string kFirstHash(32, '1');
const std::string kSecondHash(32, '2');
const KVCacheEventScope kDevice{"model", 0, "gpu"};
const KVCacheEventScope kHost{"model", 0, "cpu"};
std::shared_ptr<KVCacheEventJournal> replay_update_journal;

struct EventFrame {
  uint64_t sequence = 0;
  nlohmann::json payload;
};

class SubscriberSocket final {
 public:
  SubscriberSocket(void* context, int32_t type, const std::string& endpoint)
      : socket_(zmq_socket(context, type), &zmq_close) {
    CHECK(socket_);
    const int32_t linger = 0;
    const int32_t timeout = 2000;
    CHECK_EQ(zmq_setsockopt(socket_.get(), ZMQ_LINGER, &linger, sizeof(linger)),
             0);
    CHECK_EQ(
        zmq_setsockopt(socket_.get(), ZMQ_RCVTIMEO, &timeout, sizeof(timeout)),
        0);
    if (type == ZMQ_SUB) {
      CHECK_EQ(zmq_setsockopt(socket_.get(), ZMQ_SUBSCRIBE, "", 0), 0);
    }
    CHECK_EQ(zmq_connect(socket_.get(), endpoint.c_str()), 0);
  }

  bool request_replay() {
    const std::array<uint8_t, 8> start{};
    return zmq_send(socket_.get(), nullptr, 0, ZMQ_SNDMORE) == 0 &&
           zmq_send(socket_.get(), start.data(), start.size(), 0) ==
               static_cast<int32_t>(start.size());
  }

  std::optional<EventFrame> receive() {
    std::array<uint8_t, 4096> buffer{};
    EventFrame result;
    for (int32_t frame = 0; frame < 3; ++frame) {
      const int32_t size =
          zmq_recv(socket_.get(), buffer.data(), buffer.size(), 0);
      if (size < 0 || static_cast<size_t>(size) > buffer.size()) {
        return std::nullopt;
      }
      int32_t more = 0;
      size_t length = sizeof(more);
      if (zmq_getsockopt(socket_.get(), ZMQ_RCVMORE, &more, &length) != 0 ||
          (more != 0) != (frame < 2)) {
        return std::nullopt;
      }
      if (frame == 1) {
        if (size != static_cast<int32_t>(sizeof(uint64_t))) {
          return std::nullopt;
        }
        for (int32_t index = 0; index < size; ++index) {
          result.sequence = (result.sequence << 8) | buffer[index];
        }
      } else if (frame == 2 && size > 0) {
        result.payload =
            nlohmann::json::from_msgpack(buffer.begin(), buffer.begin() + size);
      }
    }
    return result;
  }

 private:
  std::unique_ptr<void, decltype(&zmq_close)> socket_;
};

class BoundPort final {
 public:
  BoundPort() : fd_(socket(AF_INET, SOCK_STREAM, 0)) {
    CHECK_GE(fd_, 0) << "test socket: " << errno;
    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    CHECK_EQ(bind(fd_, reinterpret_cast<sockaddr*>(&address), sizeof(address)),
             0)
        << "test bind: " << errno;
    socklen_t length = sizeof(address);
    CHECK_EQ(getsockname(fd_, reinterpret_cast<sockaddr*>(&address), &length),
             0)
        << "test port: " << errno;
    endpoint_ = "tcp://127.0.0.1:" + std::to_string(ntohs(address.sin_port));
  }
  ~BoundPort() { release(); }
  BoundPort(const BoundPort&) = delete;
  BoundPort& operator=(const BoundPort&) = delete;
  const std::string& endpoint() const { return endpoint_; }
  void release() {
    if (fd_ >= 0) {
      close(fd_);
      fd_ = -1;
    }
  }

 private:
  int32_t fd_;
  std::string endpoint_;
};
}  // namespace

// This test binary alone wraps replay receive. Inject a producer update after
// the live publication boundary and immediately before the snapshot callback.
extern "C" int __wrap_zmq_msg_recv(zmq_msg_t* message,
                                   void* socket,
                                   int flags) {
  const int32_t result = __real_zmq_msg_recv(message, socket, flags);
  if (result < 0) {
    return result;
  }
  int32_t type = 0;
  size_t length = sizeof(type);
  if (zmq_getsockopt(socket, ZMQ_TYPE, &type, &length) != 0 ||
      type != ZMQ_ROUTER) {
    return result;
  }
  const auto journal = std::atomic_exchange(
      &replay_update_journal, std::shared_ptr<KVCacheEventJournal>{});
  if (journal) {
    journal->store(kDevice, kSecondHash, kFirstHash, {3, 4}, /*block_index=*/1);
    journal->store_copy(kHost, kSecondHash);
  }
  return result;
}

TEST(KVCacheEventPublisherTest, ReplaySnapshotMatchesItsLivePublication) {
  BoundPort engine_port;
  BoundPort pool_port;
  BoundPort replay_port;
  auto journal = std::make_shared<KVCacheEventJournal>();
  journal->store(kDevice, kFirstHash, "", {1, 2}, /*block_index=*/0);
  journal->store_copy(kHost, kFirstHash);
  std::unique_ptr<void, decltype(&zmq_ctx_term)> context(zmq_ctx_new(),
                                                         &zmq_ctx_term);
  ASSERT_TRUE(context);
  engine_port.release();
  pool_port.release();
  replay_port.release();
  KVCacheEventPublisher publisher(
      journal,
      {{"endpoint", engine_port.endpoint()},
       {"motor_pool_endpoint", pool_port.endpoint()},
       {"replay_endpoint", replay_port.endpoint()},
       {"snapshot_interval_seconds", 60}},
      "model",
      /*dp_size=*/1);
  ASSERT_TRUE(publisher.healthy());
  SubscriberSocket engine(context.get(), ZMQ_SUB, engine_port.endpoint());
  SubscriberSocket pool(context.get(), ZMQ_SUB, pool_port.endpoint());
  SubscriberSocket replay(context.get(), ZMQ_DEALER, replay_port.endpoint());
  ASSERT_TRUE(engine.receive());
  ASSERT_TRUE(pool.receive());
  // Let any coalesced subscription reset finish; the injection then exercises
  // an ordinary incremental iteration, rather than a pre-existing reset page.
  std::this_thread::sleep_for(std::chrono::milliseconds(1200));
  std::atomic_store(&replay_update_journal, journal);
  ASSERT_TRUE(replay.request_replay());
  const auto recovered = replay.receive();
  ASSERT_TRUE(recovered);
  const auto end = replay.receive();
  ASSERT_TRUE(end);
  EXPECT_EQ(end->sequence, UINT64_MAX);
  EXPECT_TRUE(end->payload.is_null());
  EXPECT_FALSE(std::atomic_load(&replay_update_journal));
  ASSERT_EQ(recovered->payload[1].size(), 5);
  EXPECT_EQ(recovered->payload[1][0]["type"], "AllBlocksCleared");

  std::optional<EventFrame> live;
  do {
    live = engine.receive();
    ASSERT_TRUE(live);
  } while (live->sequence < recovered->sequence);
  EXPECT_EQ(live->sequence, recovered->sequence);
  EXPECT_EQ(live->payload[1], recovered->payload[1]);

  const auto snapshot = journal->read("", 0);
  EXPECT_EQ(snapshot["events"].size(), 4);
  journal->remove(kDevice, kSecondHash);
  std::optional<EventFrame> following;
  do {
    following = engine.receive();
    ASSERT_TRUE(following);
    for (const auto& event : following->payload[1]) {
      if (event["type"] == "BlockStored" && event["medium"] == "gpu") {
        EXPECT_NE(
            event["block_hashes"],
            nlohmann::json({KVCacheEventPublisher::wire_hash(kSecondHash)}))
            << "A replay-covered update was published again as a live delta";
      }
    }
  } while (following->payload[1][0]["type"] != "BlockRemoved");
  EXPECT_GT(following->sequence, recovered->sequence);
  ASSERT_EQ(following->payload[1].size(), 1);
  EXPECT_EQ(following->payload[1][0]["block_hashes"],
            nlohmann::json({KVCacheEventPublisher::wire_hash(kSecondHash)}));
  EXPECT_TRUE(publisher.healthy());
}

TEST(KVCacheEventJournalTest, HostCopySurvivesDeviceEviction) {
  KVCacheEventJournal journal;
  journal.store(kDevice, kFirstHash, "", {11, 12}, /*block_index=*/0);
  journal.store_copy(kHost, kFirstHash);
  journal.remove(kDevice, kFirstHash);
  const nlohmann::json state = journal.read("", 0);
  ASSERT_EQ(state["events"].size(), 1);
  EXPECT_EQ(state["events"][0]["medium"], "cpu");
  EXPECT_EQ(state["events"][0]["token_ids"], nlohmann::json({11, 12}));
  journal.store_copy(kDevice, kFirstHash);
  EXPECT_EQ(journal.read("", 0)["events"].size(), 2);
  journal.remove(kHost, kFirstHash);
  EXPECT_EQ(journal.read("", 0)["events"][0]["medium"], "gpu");
}

TEST(KVCacheEventJournalTest, DuplicateStoreDoesNotProduceAnotherEvent) {
  KVCacheEventJournal journal;
  const nlohmann::json initial = journal.read("", 0);
  journal.store(kDevice, kFirstHash, "", {1, 2}, /*block_index=*/0);
  journal.store(kDevice, kFirstHash, "", {1, 2}, /*block_index=*/0);
  const nlohmann::json page = journal.read(initial["epoch"], 0);
  ASSERT_FALSE(page["reset"].get<bool>());
  EXPECT_EQ(page["events"].size(), 1);
  EXPECT_EQ(page["cursor"], 1);
}

TEST(KVCacheEventJournalTest, OverflowAndRestartReturnCurrentSnapshot) {
  KVCacheEventJournal journal(/*capacity=*/2);
  const nlohmann::json initial = journal.read("", 0);
  journal.store(kDevice, kFirstHash, "", {1, 2}, /*block_index=*/0);
  journal.store(kDevice, kSecondHash, kFirstHash, {3, 4}, /*block_index=*/1);
  journal.remove(kDevice, kFirstHash);
  const nlohmann::json page = journal.read(initial["epoch"], 0);
  ASSERT_TRUE(page["reset"].get<bool>());
  ASSERT_EQ(page["events"].size(), 1);
  EXPECT_EQ(page["events"][0]["block_hash"], kSecondHash);
  EXPECT_EQ(page["events"][0]["parent_hash"], kFirstHash);
  EXPECT_EQ(page["cursor"], 3);
  EXPECT_TRUE(journal.read("previous-process", 3)["reset"].get<bool>());
  EXPECT_TRUE(journal.read(initial["epoch"], 100)["reset"].get<bool>());
}

TEST(KVCacheEventJournalTest, RankAndModelResidenciesAreIndependent) {
  KVCacheEventJournal journal;
  journal.store(kDevice, kFirstHash, "", {1, 2}, /*block_index=*/0);
  journal.store({"model", 1, "gpu"}, kFirstHash, "", {1, 2}, /*block_index=*/0);
  journal.store({"other", 0, "gpu"}, kFirstHash, "", {1, 2}, /*block_index=*/0);
  journal.remove(kDevice, kFirstHash);
  EXPECT_EQ(journal.read("", 0)["events"].size(), 2);
}

TEST(KVCacheEventJournalTest, IncrementalPaginationDoesNotSkipEvents) {
  KVCacheEventJournal journal;
  const nlohmann::json initial = journal.read("", 0);
  journal.store(kDevice, kFirstHash, "", {1, 2}, /*block_index=*/0);
  journal.store_copy(kHost, kFirstHash);
  journal.remove(kDevice, kFirstHash);
  const nlohmann::json first = journal.read(initial["epoch"], 0, /*limit=*/1);
  const nlohmann::json rest = journal.read(first["epoch"], first["cursor"]);
  EXPECT_EQ(first["cursor"], 1);
  EXPECT_EQ(rest["events"].size(), 2);
  EXPECT_EQ(rest["events"][1]["type"], "removed");
}

TEST(KVCacheEventJournalTest, HostSnapshotTracksEvictionAndDeviceOnlyUpdates) {
  KVCacheEventJournal journal;
  journal.store(kDevice, kFirstHash, "", {1, 2}, /*block_index=*/0);
  journal.store_copy(kHost, kFirstHash);
  const nlohmann::json host =
      journal.read("", 0, /*limit=*/1024, /*medium=*/"cpu");
  ASSERT_EQ(host["events"].size(), 1);
  EXPECT_EQ(host["events"][0]["medium"], "cpu");
  EXPECT_EQ(host["events"][0]["token_ids"], nlohmann::json({1, 2}));

  journal.remove(kHost, kFirstHash);
  const nlohmann::json removed = journal.read(
      host["epoch"], host["cursor"], /*limit=*/1, /*medium=*/"cpu");
  ASSERT_EQ(removed["events"].size(), 1);
  EXPECT_EQ(removed["events"][0]["type"], "removed");
  EXPECT_TRUE(
      journal.read("", 0, /*limit=*/1024, /*medium=*/"cpu")["events"].empty());
  EXPECT_EQ(journal.read("", 0)["events"].size(), 1);

  journal.store(kDevice, kSecondHash, kFirstHash, {3, 4}, /*block_index=*/1);
  const nlohmann::json filtered = journal.read(
      removed["epoch"], removed["cursor"], /*limit=*/1, /*medium=*/"cpu");
  EXPECT_TRUE(filtered["events"].empty());
  EXPECT_GT(filtered["cursor"], removed["cursor"]);
}

TEST(KVCacheEventJournalTest, PendingOverflowClearsClaimsAndStopsCollection) {
  KVCacheEventJournal journal(/*capacity=*/4096, /*pending_capacity=*/2);
  const nlohmann::json initial = journal.read("", 0);
  journal.store(kDevice, kFirstHash, "", {1, 2}, /*block_index=*/0);
  journal.store(kDevice, kSecondHash, kFirstHash, {3, 4}, /*block_index=*/1);
  journal.remove(kDevice, kFirstHash);
  EXPECT_FALSE(journal.enabled());
  const nlohmann::json reset = journal.read(initial["epoch"], 0);
  EXPECT_TRUE(reset["reset"].get<bool>());
  EXPECT_NE(reset["epoch"], initial["epoch"]);
  EXPECT_TRUE(reset["events"].empty());
  journal.store(kDevice, kFirstHash, "", {1, 2}, /*block_index=*/0);
  EXPECT_TRUE(journal.read("", 0)["events"].empty());
}

TEST(KVCacheEventJournalTest, InvalidMetadataClearsPreviouslyPublishedClaims) {
  KVCacheEventJournal journal;
  journal.store(kDevice, kFirstHash, "", {1, 2}, /*block_index=*/0);
  const nlohmann::json initial = journal.read("", 0);
  ASSERT_EQ(initial["events"].size(), 1);
  journal.store(kDevice, "invalid-hash", "", {3, 4}, /*block_index=*/0);
  const nlohmann::json reset =
      journal.read(initial["epoch"], initial["cursor"]);
  EXPECT_FALSE(journal.enabled());
  EXPECT_TRUE(reset["reset"].get<bool>());
  EXPECT_TRUE(reset["events"].empty());
}

TEST(KVCacheEventJournalTest,
     ConcurrentProducersPreserveResidencyDuringSnapshots) {
  KVCacheEventJournal journal;
  std::atomic<bool> done{false};
  std::thread reader([&]() {
    while (!done.load()) {
      journal.read("", 0);
    }
  });
  std::vector<std::thread> writers;
  writers.reserve(4);
  for (int32_t rank = 0; rank < 4; ++rank) {
    writers.emplace_back([&, rank]() {
      const KVCacheEventScope scope{"model", rank, "gpu"};
      for (int32_t index = 0; index < 500; ++index) {
        const std::string suffix = std::to_string(index);
        const std::string hash = std::string(32 - suffix.size(), '0') + suffix;
        journal.store(scope, hash, "", {index, index + 1}, /*block_index=*/0);
      }
    });
  }
  for (auto& writer : writers) {
    writer.join();
  }
  done.store(true);
  reader.join();
  EXPECT_TRUE(journal.enabled());
  EXPECT_EQ(journal.read("", 0)["events"].size(), 2000);
}

TEST(KVCacheEventPublisherTest,
     InvalidConfigTypeDisablesExporterWithoutTerminatingCaller) {
  auto journal = std::make_shared<KVCacheEventJournal>();
  KVCacheEventPublisher publisher(
      journal, {{"endpoint", 123}}, "model", /*dp_size=*/1);
  EXPECT_FALSE(publisher.healthy());
  EXPECT_FALSE(journal->enabled());
}

TEST(KVCacheEventPublisherTest,
     EmptyEndpointDisablesExporterWithoutTerminatingCaller) {
  auto journal = std::make_shared<KVCacheEventJournal>();
  KVCacheEventPublisher publisher(
      journal, {{"endpoint", ""}}, "model", /*dp_size=*/1);
  EXPECT_FALSE(publisher.healthy());
  EXPECT_FALSE(journal->enabled());
}

TEST(KVCacheEventPublisherTest,
     BindFailureDisablesExporterWithoutTerminatingCaller) {
  BoundPort occupied;
  auto journal = std::make_shared<KVCacheEventJournal>();
  KVCacheEventPublisher publisher(
      journal, {{"endpoint", occupied.endpoint()}}, "model", /*dp_size=*/1);
  EXPECT_FALSE(publisher.healthy());
  EXPECT_FALSE(journal->enabled());
  journal->store(kDevice, kFirstHash, "", {1, 2}, /*block_index=*/0);
  EXPECT_TRUE(journal->read("", 0)["events"].empty());
}

TEST(KVCacheEventPublisherTest,
     BackgroundMetadataFailureDoesNotTerminateCaller) {
  BoundPort reserved;
  const std::string endpoint = reserved.endpoint();
  reserved.release();
  auto journal = std::make_shared<KVCacheEventJournal>();
  KVCacheEventPublisher publisher(
      journal, {{"endpoint", endpoint}}, "model", /*dp_size=*/1);
  ASSERT_TRUE(publisher.healthy());
  journal->store(kDevice, kFirstHash, "", {1, 2}, /*block_index=*/0);
  journal->store(kDevice, kFirstHash, "", {3, 4}, /*block_index=*/0);
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(2);
  while (journal->enabled() && std::chrono::steady_clock::now() < deadline) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  EXPECT_FALSE(journal->enabled());
  EXPECT_FALSE(publisher.healthy());
}

TEST(KVCacheEventPublisherTest, StoredRemovedAndParentUseSameWireIdentity) {
  KVCacheEventJournal journal;
  const nlohmann::json initial = journal.read("", 0);
  journal.store(kDevice, kFirstHash, "", {1, 2}, /*block_index=*/0);
  journal.store(kDevice, kSecondHash, kFirstHash, {3, 4}, /*block_index=*/1);
  journal.remove(kDevice, kFirstHash);
  const auto events = journal.read(initial["epoch"], 0)["events"];
  const auto first = KVCacheEventPublisher::engine_event(events[0]);
  const auto second = KVCacheEventPublisher::engine_event(events[1]);
  const auto removed = KVCacheEventPublisher::engine_event(events[2]);
  EXPECT_EQ(first["block_hashes"], removed["block_hashes"]);
  EXPECT_EQ(first["block_hashes"][0], second["parent_block_hash"]);
  EXPECT_TRUE(first["parent_block_hash"].is_null());
  EXPECT_EQ(second["token_ids"], nlohmann::json({3, 4}));
  EXPECT_EQ(KVCacheEventPublisher::pool_event(events[1])["seq_hashes"],
            second["block_hashes"]);
  EXPECT_EQ(KVCacheEventPublisher::rank_endpoint("tcp://*:5557", 2),
            "tcp://*:5559");
}

}  // namespace xllm

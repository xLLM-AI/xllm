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

#include "core/util/shared_memory_manager.h"

#include <fcntl.h>
#include <glog/logging.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>
#include <csignal>
#include <cstring>
#include <mutex>
#include <new>

namespace xllm {
namespace {

constexpr char kGenerationLockName[] = "/xllm_shm_generation_lock_v2";
constexpr uint64_t kSharedMemoryMagic = 0x584C4C4D53484D32ULL;
constexpr uint32_t kSharedMemoryLayoutVersion = 2;

class alignas(64) SharedMemoryLayoutHeader final {
 public:
  uint64_t magic = kSharedMemoryMagic;
  uint32_t version = kSharedMemoryLayoutVersion;
  uint32_t header_size = sizeof(SharedMemoryLayoutHeader);
  uint64_t payload_size = 0;
  uint64_t generation = 1;
  uint64_t reserved[4] = {};
};

static_assert(sizeof(SharedMemoryLayoutHeader) == 64);

}  // namespace

SharedMemoryManager::SharedMemoryManager(const std::string& name,
                                         size_t size,
                                         bool& is_creator)
    : shm_name_(name),
      size_(static_cast<int64_t>(size)),
      mapping_size_(static_cast<int64_t>(sizeof(SharedMemoryLayoutHeader)) +
                    static_cast<int64_t>(size)) {
  // Register cleanup handlers for signals (once per process)
  static std::once_flag flag;
  std::call_once(flag, [] {
    signal(SIGINT, cleanup_handler);
    signal(SIGTERM, cleanup_handler);
    // signal(SIGSEGV, cleanup_handler);
  });

  CHECK_GT(size_, 0) << "Shared memory payload size must be positive.";
  CHECK_GT(mapping_size_, size_) << "Shared memory mapping size overflow.";

  // A single persistent lock serializes name generation transitions for every
  // xLLM SHM object. Its namespace is bounded and it prevents
  // open/unlink/create races without leaking one lock object per service port
  // or test name.
  lock_fd_ = shm_open(kGenerationLockName, O_CREAT | O_RDWR, 0666);
  if (lock_fd_ == -1) {
    LOG(FATAL) << "shm_open lock failed: " << strerror(errno);
  }
  if (flock(lock_fd_, LOCK_EX) == -1) {
    close(lock_fd_);
    lock_fd_ = -1;
    LOG(FATAL) << "flock generation lock failed: " << strerror(errno);
  }

  bool created_exclusively = false;
  fd_ = shm_open(name.c_str(), O_CREAT | O_RDWR | O_EXCL, 0666);
  if (fd_ != -1) {
    created_exclusively = true;
  } else if (errno == EEXIST) {
    fd_ = shm_open(name.c_str(), O_RDWR, 0666);
    if (fd_ == -1) {
      LOG(FATAL) << "shm_open failed: " << strerror(errno);
    }
  } else {
    LOG(FATAL) << "shm_open create failed: " << strerror(errno);
  }

  if (created_exclusively) {
    if (flock(fd_, LOCK_EX) == -1) {
      close(fd_);
      LOG(FATAL) << "flock(LOCK_EX) failed: " << strerror(errno);
    }
    if (ftruncate(fd_, mapping_size_) == -1) {
      close(fd_);
      LOG(FATAL) << "ftruncate failed: " << strerror(errno);
    }
    is_creator_ = true;
  } else {
    struct stat st;
    if (fstat(fd_, &st) == -1) {
      close(fd_);
      LOG(FATAL) << "fstat failed: " << strerror(errno);
    }
    if (st.st_size != static_cast<off_t>(mapping_size_)) {
      close(fd_);
      LOG(FATAL) << "incompatible shared memory layout for " << name
                 << ": expected mapped size " << mapping_size_ << ", got "
                 << st.st_size
                 << ". Stop all old xLLM processes and remove the stale SHM "
                    "object before upgrading.";
    }
    is_creator_ = (flock(fd_, LOCK_EX | LOCK_NB) == 0);
    if (!is_creator_ && errno != EWOULDBLOCK && errno != EAGAIN) {
      close(fd_);
      LOG(FATAL) << "flock(LOCK_EX) failed: " << strerror(errno);
    }
    if (!is_creator_ && flock(fd_, LOCK_SH) == -1) {
      close(fd_);
      LOG(FATAL) << "flock(LOCK_SH) failed: " << strerror(errno);
    }
  }
  is_creator = is_creator_;

  addr_ =
      mmap(nullptr, mapping_size_, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0);
  if (addr_ == MAP_FAILED) {
    close(fd_);
    LOG(FATAL) << "mmap failed: " << strerror(errno);
  }
  auto* header = static_cast<SharedMemoryLayoutHeader*>(addr_);
  payload_addr_ = static_cast<char*>(addr_) + sizeof(SharedMemoryLayoutHeader);

  if (created_exclusively) {
    new (header) SharedMemoryLayoutHeader();
    header->payload_size = static_cast<uint64_t>(size_);
  } else {
    CHECK_EQ(header->magic, kSharedMemoryMagic)
        << "incompatible shared memory layout magic for " << name;
    CHECK_EQ(header->version, kSharedMemoryLayoutVersion)
        << "incompatible shared memory layout version for " << name;
    CHECK_EQ(header->header_size, sizeof(SharedMemoryLayoutHeader))
        << "incompatible shared memory layout header size for " << name;
    CHECK_EQ(header->payload_size, static_cast<uint64_t>(size_))
        << "incompatible shared memory payload size for " << name;
    if (is_creator_) {
      ++header->generation;
    }
  }

  if (is_creator_) {
    std::memset(payload_addr_, 0, size_);
    if (flock(fd_, LOCK_SH) == -1) {
      munmap(addr_, mapping_size_);
      close(fd_);
      LOG(FATAL) << "flock(LOCK_SH) failed: " << strerror(errno);
    }
  }

  if (flock(lock_fd_, LOCK_UN) == -1) {
    munmap(addr_, mapping_size_);
    close(fd_);
    close(lock_fd_);
    fd_ = -1;
    lock_fd_ = -1;
    LOG(FATAL) << "flock generation unlock failed: " << strerror(errno);
  }
}

SharedMemoryManager::~SharedMemoryManager() {
  LOG(INFO) << "Delete ~SharedMemoryManager";
  if (addr_ != MAP_FAILED) {
    munmap(addr_, mapping_size_);
  }

  if (fd_ == -1 || lock_fd_ == -1) {
    return;
  }

  if (flock(lock_fd_, LOCK_EX) == -1) {
    PLOG(ERROR) << "flock generation lock failed while releasing " << shm_name_;
    close(fd_);
    close(lock_fd_);
    return;
  }

  if (flock(fd_, LOCK_UN) == -1) {
    PLOG(ERROR) << "flock(LOCK_UN) failed while releasing " << shm_name_;
  } else if (flock(fd_, LOCK_EX | LOCK_NB) == 0) {
    shm_unlink(shm_name_.c_str());
  } else if (errno != EWOULDBLOCK && errno != EAGAIN) {
    PLOG(ERROR) << "flock(LOCK_EX) failed while releasing " << shm_name_;
  }

  close(fd_);
  if (flock(lock_fd_, LOCK_UN) == -1) {
    PLOG(ERROR) << "flock generation unlock failed while releasing "
                << shm_name_;
  }
  close(lock_fd_);
}

void SharedMemoryManager::cleanup_handler(int sig) {
  // Avoid non-async-signal-safe operations (mutex, logging, shm_unlink, exit).
  // Just restore default handler and re-raise to terminate normally.
  // TODO: support cleaning up shared memory properly when singal is received.
  signal(sig, SIG_DFL);
  kill(getpid(), sig);
}

}  // namespace xllm

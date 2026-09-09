/* Copyright 2025-2026 The xLLM Authors.

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

#pragma once

#define EASY_SMP_LOCK "lock;"
#define easy_atomic_set(v, i) ((v) = (i))

#if defined(__x86_64)
#define cpu_relax() asm volatile("pause\n" : : : "memory")
#else
#define cpu_relax() asm volatile("yield\n" : : : "memory")
#endif

using easy_atomic_t = volatile int64_t;
static __inline__ void easy_atomic_add(easy_atomic_t* v, int64_t i) {
#if defined(__x86_64__)
  __asm__ __volatile__(EASY_SMP_LOCK "addq %1,%0"
                       : "=m"((*v))
                       : "r"(i), "m"((*v)));
#else
  __atomic_add_fetch(v, i, __ATOMIC_SEQ_CST);
#endif
}
static __inline__ int64_t easy_atomic_add_return(easy_atomic_t* value,
                                                 int64_t i) {
  int64_t __i = i;
#if defined(__x86_64__)
  __asm__ __volatile__(EASY_SMP_LOCK "xaddq %0, %1;"
                       : "=r"(i)
                       : "m"(*value), "0"(i));
#else
  i = __atomic_fetch_add(value, i, __ATOMIC_SEQ_CST);
#endif
  return i + __i;
}
static __inline__ int64_t easy_atomic_cmp_set(easy_atomic_t* lock,
                                              int64_t old,
                                              int64_t set) {
  uint8_t res;
#if defined(__x86_64__)
  __asm__ volatile(EASY_SMP_LOCK "cmpxchgq %3, %1; sete %0"
                   : "=a"(res)
                   : "m"(*lock), "a"(old), "r"(set)
                   : "cc", "memory");
#else
  res = __atomic_compare_exchange_n(
      lock, &old, set, true, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST);
#endif
  return res;
}
static __inline__ void easy_atomic_inc(easy_atomic_t* v) {
#if defined(__x86_64__)
  __asm__ __volatile__(EASY_SMP_LOCK "incq %0" : "=m"(*v) : "m"(*v));
#else
  __atomic_add_fetch(v, 1, __ATOMIC_SEQ_CST);
#endif
}
static __inline__ void easy_atomic_dec(easy_atomic_t* v) {
#if defined(__x86_64__)
  __asm__ __volatile__(EASY_SMP_LOCK "decq %0" : "=m"(*v) : "m"(*v));
#else
  __atomic_sub_fetch(v, 1, __ATOMIC_SEQ_CST);
#endif
}

#define EASY_OK 0
#define EASY_ERROR (-1)
#define EASY_ABORT (-2)
#define EASY_ASYNC (-3)
#define EASY_BREAK (-4)
#define EASY_ENCODE (-5)
#define EASY_QUEUE_FULL (-6)
#define EASY_AGAIN (-EAGAIN)

struct easy_spinrwlock_t {
  easy_atomic_t ref_cnt;
  easy_atomic_t wait_write;
};
#define EASY_SPINRWLOCK_INITIALIZER {0, 0}
static __inline__ int easy_spinrwlock_rdlock(easy_spinrwlock_t* lock) {
  int ret = EASY_OK;

  if (lock == nullptr) {
    ret = EASY_ERROR;
  } else {
    int cond = 1;

    while (cond) {
      int loop = 1;

      do {
        easy_atomic_t oldv = lock->ref_cnt;

        if (0 <= oldv && 0 == lock->wait_write) {
          if (easy_atomic_cmp_set(&lock->ref_cnt, oldv, oldv + 1)) {
            return ret;
          }
        }

        cpu_relax();
        loop <<= 1;
      } while (loop < 1024);

      sched_yield();
    }
  }

  return ret;
}
static __inline__ int easy_spinrwlock_wrlock(easy_spinrwlock_t* lock) {
  int ret = EASY_OK;

  if (lock == nullptr) {
    ret = EASY_ERROR;
  } else {
    int cond = 1;
    easy_atomic_inc(&lock->wait_write);

    while (cond) {
      int loop = 1;

      do {
        easy_atomic_t oldv = lock->ref_cnt;

        if (0 == oldv) {
          if (easy_atomic_cmp_set(&lock->ref_cnt, oldv, -1)) {
            cond = 0;
            break;
          }
        }

        cpu_relax();
        loop <<= 1;
      } while (loop < 1024);

      if (cond) sched_yield();
    }

    easy_atomic_dec(&lock->wait_write);
  }

  return ret;
}
static __inline__ int easy_spinrwlock_try_rdlock(easy_spinrwlock_t* lock) {
  int ret = EASY_OK;

  if (lock == nullptr) {
    ret = EASY_ERROR;
  } else {
    ret = EASY_AGAIN;
    easy_atomic_t oldv = lock->ref_cnt;

    if (0 <= oldv && 0 == lock->wait_write) {
      easy_atomic_t newv = oldv + 1;

      if (easy_atomic_cmp_set(&lock->ref_cnt, oldv, newv)) {
        ret = EASY_OK;
      }
    }
  }

  return ret;
}
static __inline__ int easy_spinrwlock_try_wrlock(easy_spinrwlock_t* lock) {
  int ret = EASY_OK;

  if (lock == nullptr) {
    ret = EASY_ERROR;
  } else {
    ret = EASY_AGAIN;
    easy_atomic_t oldv = lock->ref_cnt;

    if (0 == oldv) {
      easy_atomic_t newv = -1;

      if (easy_atomic_cmp_set(&lock->ref_cnt, oldv, newv)) {
        ret = EASY_OK;
      }
    }
  }

  return ret;
}
static __inline__ int easy_spinrwlock_unlock(easy_spinrwlock_t* lock) {
  int ret = EASY_OK;

  if (lock == nullptr) {
    ret = EASY_ERROR;
  } else {
    while (1) {
      easy_atomic_t oldv = lock->ref_cnt;

      if (-1 == oldv) {
        easy_atomic_t newv = 0;

        if (easy_atomic_cmp_set(&lock->ref_cnt, oldv, newv)) {
          break;
        }
      } else if (0 < oldv) {
        easy_atomic_t newv = oldv - 1;

        if (easy_atomic_cmp_set(&lock->ref_cnt, oldv, newv)) {
          break;
        }
      } else {
        ret = EASY_ERROR;
        break;
      }
    }
  }

  return ret;
}
namespace xllm {

class spin_rd_lock {
 public:
  using lock_type = easy_spinrwlock_t;

  explicit spin_rd_lock(lock_type* lock) : lock_(lock) {
    easy_spinrwlock_rdlock(lock_);
  }
  explicit spin_rd_lock(lock_type& lock) : lock_(&lock) {
    easy_spinrwlock_rdlock(lock_);
  }
  ~spin_rd_lock() { easy_spinrwlock_unlock(lock_); }

 private:
  lock_type* lock_;
};

class spin_wr_lock {
 public:
  using lock_type = easy_spinrwlock_t;

  explicit spin_wr_lock(lock_type* lock) : lock_(lock) {
    easy_spinrwlock_wrlock(lock_);
  }
  explicit spin_wr_lock(lock_type& lock) : lock_(&lock) {
    easy_spinrwlock_wrlock(lock_);
  }
  ~spin_wr_lock() { easy_spinrwlock_unlock(lock_); }

 private:
  lock_type* lock_;
};

}  // namespace xllm

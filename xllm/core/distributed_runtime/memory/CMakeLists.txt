include(cc_library)

cc_library(
  NAME
    worker_page_budget
  HDRS
    worker_page_budget.h
  SRCS
    worker_page_budget.cpp
  DEPS
    :common
    glog::glog
)

include(cc_library)

cc_library(
  NAME
    disaggregation
  HDRS
    pd_scheduler.h
    pd_service.h
    pd_service_impl.h
    disagg_pd_request_converter.h
    disagg_pd_response.h
  SRCS
    pd_scheduler.cpp
    pd_service.cpp
    pd_service_impl.cpp
    disagg_pd_request_converter.cpp
    disagg_pd_response.cpp
  DEPS
    :scheduler
    :distributed_runtime
)
target_link_libraries(disaggregation PRIVATE
  :xllm_server
  :distributed_vmm_coordinator
  :api_response
  :pd_topology_guard)

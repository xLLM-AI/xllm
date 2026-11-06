include(cc_library)

cc_library(
  NAME
    disaggregation
  HDRS
    disagg_pd_scheduler.h
    disagg_pd_service.h
    disagg_pd_service_impl.h
    disagg_pd_request_converter.h
    disagg_pd_response.h
  SRCS
    disagg_pd_scheduler.cpp
    disagg_pd_service.cpp
    disagg_pd_service_impl.cpp
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

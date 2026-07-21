if(NOT BUILD_TESTING)
  return()
endif()

add_subdirectory(api_service)
add_subdirectory(core)
add_subdirectory(function_call)
add_subdirectory(models)

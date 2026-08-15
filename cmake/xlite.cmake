# xlite link helper: find_package(xlite) when USE_NPU AND USE_XLITE are ON.

function(xllm_link_xlite target)
  if(NOT USE_NPU OR NOT USE_XLITE)
    return()
  endif()

  if(NOT TARGET xlite::xlite)
    execute_process(
      COMMAND ${Python_EXECUTABLE} -c "import xlite; print(xlite.cmake_prefix_path)"
      OUTPUT_VARIABLE _XLITE_CMAKE_PREFIX
      OUTPUT_STRIP_TRAILING_WHITESPACE
      RESULT_VARIABLE _XLITE_IMPORT_RESULT)
    if(NOT _XLITE_IMPORT_RESULT EQUAL 0 OR _XLITE_CMAKE_PREFIX STREQUAL "")
      message(FATAL_ERROR "USE_XLITE is ON but xlite not found. Install xlite or pass -DUSE_XLITE=OFF.")
    endif()

    find_package(xlite REQUIRED CONFIG PATHS "${_XLITE_CMAKE_PREFIX}" NO_DEFAULT_PATH)
    message(STATUS "xlite::xlite found via find_package (${_XLITE_CMAKE_PREFIX})")
  endif()

  target_link_libraries(${target} PRIVATE xlite::xlite)
endfunction()
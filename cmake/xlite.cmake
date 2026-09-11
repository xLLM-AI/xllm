# xlite: auto pip install the submodule into site-packages at configure time,
# then find_package + link. Same git-HEAD marker pattern as xllm_ops.

if(USE_XLITE)
  execute_process(
    COMMAND git -c "safe.directory=${CMAKE_SOURCE_DIR}/third_party/GVirt"
             -C "${CMAKE_SOURCE_DIR}/third_party/GVirt" rev-parse HEAD
    OUTPUT_VARIABLE XLITE_GIT_HEAD
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
  )

  execute_process(
    COMMAND ${Python3_EXECUTABLE} -c
            "import sysconfig; print(sysconfig.get_paths()['platlib'])"
    OUTPUT_VARIABLE _XLITE_PLATLIB
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  set(XLITE_MARKER_PATH "${_XLITE_PLATLIB}/xlite/.xlite_git_head")

  if(DEFINED ENV{XLITE_GIT_HEAD_CACHED})
    set(XLITE_GIT_HEAD_CACHED "$ENV{XLITE_GIT_HEAD_CACHED}")
  endif()

  if(NOT DEFINED XLITE_GIT_HEAD_CACHED
     OR NOT XLITE_GIT_HEAD STREQUAL XLITE_GIT_HEAD_CACHED
     OR NOT EXISTS "${XLITE_MARKER_PATH}")
    message(STATUS "xlite git HEAD changed; pip installing from submodule into ${_XLITE_PLATLIB}")
    execute_process(
      COMMAND ${Python3_EXECUTABLE} -m pip install --no-deps --no-build-isolation
              --force-reinstall ${CMAKE_SOURCE_DIR}/third_party/GVirt/xlite
      RESULT_VARIABLE XLITE_PIP_RESULT
    )
    if(NOT XLITE_PIP_RESULT EQUAL 0)
      message(FATAL_ERROR
        "Failed to precompile xlite, error code: ${XLITE_PIP_RESULT}.\n"
        "This precompile ran because the xlite submodule git HEAD changed "
        "(e.g. after switching commits/branches or updating the submodule).\n"
        "A common cause of failure here is a stale incremental build cache "
        "left by a previous version of the xlite sources.\n"
        "Fix by removing the stale xlite build directory and re-running the "
        "build. CMake re-triggers the xlite precompile automatically (the "
        "HEAD cache is only updated after a successful precompile), and the "
        "main C++ incremental build is unaffected:\n"
        "  rm -rf ${CMAKE_SOURCE_DIR}/third_party/GVirt/xlite/cmake_build")
    endif()

    set(XLITE_GIT_HEAD_CACHED "${XLITE_GIT_HEAD}" CACHE INTERNAL "" FORCE)
    get_filename_component(XLITE_MARKER_DIR "${XLITE_MARKER_PATH}" DIRECTORY)
    file(MAKE_DIRECTORY "${XLITE_MARKER_DIR}")
    file(WRITE "${XLITE_MARKER_PATH}" "${XLITE_GIT_HEAD}\n")
    message(STATUS "xlite installed to ${_XLITE_PLATLIB}/xlite (HEAD ${XLITE_GIT_HEAD})")
  else()
    message(STATUS "xlite git HEAD unchanged; skipping pip install")
  endif()
endif()

function(xllm_link_xlite target)
  if(NOT USE_XLITE)
    return()
  endif()

  if(NOT TARGET xlite::xlite)
    execute_process(
      COMMAND ${Python3_EXECUTABLE} -c "import xlite; print(xlite.cmake_prefix_path)"
      OUTPUT_VARIABLE _XLITE_CMAKE_PREFIX
      OUTPUT_STRIP_TRAILING_WHITESPACE
      RESULT_VARIABLE _XLITE_IMPORT_RESULT)
    if(NOT _XLITE_IMPORT_RESULT EQUAL 0 OR _XLITE_CMAKE_PREFIX STREQUAL "")
      message(FATAL_ERROR
        "USE_XLITE is ON but xlite is not installed. "
        "Build and install it from the submodule first:\n"
        "  pip install --no-deps --no-build-isolation "
        "${CMAKE_SOURCE_DIR}/third_party/GVirt/xlite\n"
        "or pass -DUSE_XLITE=OFF.")
    endif()

    find_package(xlite REQUIRED CONFIG PATHS "${_XLITE_CMAKE_PREFIX}" NO_DEFAULT_PATH)
    message(STATUS "xlite::xlite found via find_package (${_XLITE_CMAKE_PREFIX})")
  endif()

  target_link_libraries(${target} PRIVATE xlite::xlite)
endfunction()
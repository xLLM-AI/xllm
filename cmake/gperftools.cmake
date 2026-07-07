# Build shared libtcmalloc.so.4 from third_party/gperftools (pinned in parent repo).

include(ExternalProject)
include(ProcessorCount)

ProcessorCount(GPERFTOOLS_BUILD_JOBS)
if(NOT GPERFTOOLS_BUILD_JOBS)
  set(GPERFTOOLS_BUILD_JOBS 4)
endif()

if(DEFINED XLLM_SOURCE_DIR)
  set(_XLLM_REPO_ROOT "${XLLM_SOURCE_DIR}")
else()
  set(_XLLM_REPO_ROOT "${CMAKE_SOURCE_DIR}")
endif()

set(GPERFTOOLS_SOURCE_DIR "${_XLLM_REPO_ROOT}/third_party/gperftools")
set(GPERFTOOLS_INSTALL_DIR "${CMAKE_BINARY_DIR}/third_party/gperftools/install")
set(GPERFTOOLS_LIB_DIR "${GPERFTOOLS_INSTALL_DIR}/lib")
set(GPERFTOOLS_LIB_FILE "${GPERFTOOLS_LIB_DIR}/libtcmalloc.so")
set(GPERFTOOLS_EXPECTED_VERSION "2.10")

execute_process(
  COMMAND bash "${_XLLM_REPO_ROOT}/scripts/ensure_gperftools_submodule.sh"
  WORKING_DIRECTORY "${_XLLM_REPO_ROOT}"
  RESULT_VARIABLE _gperftools_fetch_result
  COMMAND_ECHO STDOUT
)
if(NOT _gperftools_fetch_result EQUAL 0)
  message(FATAL_ERROR
    "Failed to initialize gperftools at ${GPERFTOOLS_SOURCE_DIR}")
endif()

if(EXISTS "${GPERFTOOLS_SOURCE_DIR}/configure.ac")
  file(READ "${GPERFTOOLS_SOURCE_DIR}/configure.ac" _gperftools_configure_ac)
  if(NOT _gperftools_configure_ac MATCHES
      "AC_INIT\\(\\[gperftools\\],\\[${GPERFTOOLS_EXPECTED_VERSION}\\]")
    message(FATAL_ERROR
      "third_party/gperftools must be ${GPERFTOOLS_EXPECTED_VERSION} "
      "(found other version in configure.ac). "
      "Run: git submodule update --init third_party/gperftools")
  endif()
endif()

if(NOT EXISTS "${GPERFTOOLS_SOURCE_DIR}/configure")
  message(FATAL_ERROR
    "gperftools configure script missing at ${GPERFTOOLS_SOURCE_DIR}")
endif()

# Drop install tree produced by a different gperftools release (e.g. 2.18 -> .so.4.6.5).
file(GLOB _stale_tcmalloc_libs
  "${GPERFTOOLS_LIB_DIR}/libtcmalloc.so.4.[6-9]*"
  "${GPERFTOOLS_LIB_DIR}/libtcmalloc.so.4.[1-9][0-9]*")
if(_stale_tcmalloc_libs)
  message(STATUS
    "Removing stale gperftools install (wrong libtcmalloc SONAME): "
    "${GPERFTOOLS_INSTALL_DIR}")
  file(REMOVE_RECURSE "${GPERFTOOLS_INSTALL_DIR}")
  file(REMOVE_RECURSE "${CMAKE_BINARY_DIR}/third_party/gperftools/ep-prefix")
endif()

ExternalProject_Add(
  gperftools_ep
  SOURCE_DIR "${GPERFTOOLS_SOURCE_DIR}"
  PREFIX "${CMAKE_BINARY_DIR}/third_party/gperftools/ep-prefix"
  CONFIGURE_COMMAND
    ${CMAKE_COMMAND} -E env
    "CFLAGS=-O2 -U_ISOC23_SOURCE"
    "CXXFLAGS=-O2 -U_ISOC23_SOURCE"
    "${GPERFTOOLS_SOURCE_DIR}/configure"
    --prefix=${GPERFTOOLS_INSTALL_DIR}
    --enable-shared
    --disable-static
  BUILD_COMMAND make -j${GPERFTOOLS_BUILD_JOBS}
  INSTALL_COMMAND make install
  BUILD_IN_SOURCE 1
  INSTALL_DIR "${GPERFTOOLS_INSTALL_DIR}"
  UPDATE_COMMAND ""
  BUILD_BYPRODUCTS
    "${GPERFTOOLS_LIB_FILE}"
    "${GPERFTOOLS_LIB_DIR}/libtcmalloc.so.4"
    "${GPERFTOOLS_LIB_DIR}/libtcmalloc.so.4.5.10"
)

# leveldb::leveldb (vcpkg) exports INTERFACE_LINK_LIBRARIES=tcmalloc, so CMake
# validates tcmalloc's interface paths at configure time. Use source headers
# (present after submodule init); install/include only exists after gperftools_ep.
file(MAKE_DIRECTORY "${GPERFTOOLS_INSTALL_DIR}/include")

add_library(tcmalloc SHARED IMPORTED GLOBAL)
set_target_properties(
  tcmalloc
  PROPERTIES
    IMPORTED_LOCATION "${GPERFTOOLS_LIB_FILE}"
    INTERFACE_INCLUDE_DIRECTORIES "${GPERFTOOLS_SOURCE_DIR}/src"
    IMPORTED_NO_SONAME TRUE
)
add_dependencies(tcmalloc gperftools_ep)

set(XLLM_GPERFTOOLS_INSTALL_DIR "${GPERFTOOLS_INSTALL_DIR}" CACHE INTERNAL
    "Install prefix of bundled gperftools shared libraries")
set(XLLM_TCMALLOC_LIB_DIR "${GPERFTOOLS_LIB_DIR}" CACHE INTERNAL
    "Directory containing bundled libtcmalloc.so*")

install(
  DIRECTORY "${GPERFTOOLS_LIB_DIR}/"
  DESTINATION lib
  FILES_MATCHING
  PATTERN "libtcmalloc.so*"
)

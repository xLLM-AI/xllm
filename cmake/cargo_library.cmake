include(CMakeParseArguments)

# inspired by https://github.com/abseil/abseil-cpp
# cc_library()
# CMake function to imitate Bazel's cc_library rule.
function(cargo_library)
  cmake_parse_arguments(
      CARGO # prefix
      "" # options
      "NAME" # one value args
      "HDRS" # multi value args
      ${ARGN}
  )

  string(REPLACE "-" "_" LIB_NAME ${CARGO_NAME})
  # set(CARGO_TARGET_DIR ${CMAKE_CURRENT_BINARY_DIR})

  # figure out the target triple
  if(WIN32)
    if(CMAKE_SIZEOF_VOID_P EQUAL 8)
      set(LIB_TARGET "x86_64-pc-windows-msvc")
    else()
      set(LIB_TARGET "i686-pc-windows-msvc")
    endif()
  elseif(ANDROID)
    if(ANDROID_SYSROOT_ABI STREQUAL "x86")
      set(LIB_TARGET "i686-linux-android")
    elseif(ANDROID_SYSROOT_ABI STREQUAL "x86_64")
      set(LIB_TARGET "x86_64-linux-android")
    elseif(ANDROID_SYSROOT_ABI STREQUAL "arm")
      set(LIB_TARGET "arm-linux-androideabi")
    elseif(ANDROID_SYSROOT_ABI STREQUAL "arm64")
      set(LIB_TARGET "aarch64-linux-android")
    endif()
  elseif(IOS)
    set(LIB_TARGET "universal")
  elseif(CMAKE_SYSTEM_NAME STREQUAL Darwin)
    set(LIB_TARGET "x86_64-apple-darwin")
  else()
    if(CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64")
        set(LIB_TARGET "aarch64-unknown-linux-gnu") 
    elseif(CMAKE_SIZEOF_VOID_P EQUAL 8)
        set(LIB_TARGET "x86_64-unknown-linux-gnu")
    else()
        set(LIB_TARGET "i686-unknown-linux-gnu")
    endif()
  endif()

  if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(LIB_BUILD_TYPE "debug")
  else()
    set(LIB_BUILD_TYPE "release")
  endif()

  if(IOS)
    set(CARGO_ARGS "lipo")
  else()
    set(CARGO_ARGS "build")
    list(APPEND CARGO_ARGS "--target" ${LIB_TARGET})
  endif()

  if(${LIB_BUILD_TYPE} STREQUAL "release")
    list(APPEND CARGO_ARGS "--release")
  endif()

  file(GLOB_RECURSE LIB_SOURCES CONFIGURE_DEPENDS "*.rs")
  set(CARGO_INPUTS ${LIB_SOURCES} "${CMAKE_CURRENT_SOURCE_DIR}/Cargo.toml")
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/Cargo.lock")
    list(APPEND CARGO_INPUTS "${CMAKE_CURRENT_SOURCE_DIR}/Cargo.lock")
  endif()

  if(DEFINED ENV{XLLM_CARGO_TARGET_ROOT} AND
     NOT "$ENV{XLLM_CARGO_TARGET_ROOT}" STREQUAL "")
    get_filename_component(
      CARGO_TARGET_ROOT "$ENV{XLLM_CARGO_TARGET_ROOT}" ABSOLUTE
    )
    set(CARGO_TARGET_DIR "${CARGO_TARGET_ROOT}/${CARGO_NAME}")
  else()
    set(CARGO_TARGET_DIR "${CMAKE_CURRENT_BINARY_DIR}")
  endif()

  set(CARGO_ENV_COMMAND ${CMAKE_COMMAND} -E env "CARGO_TARGET_DIR=${CARGO_TARGET_DIR}")

  # build the library target with cargo
  set(STATIC_LIB_NAME
    "${CMAKE_STATIC_LIBRARY_PREFIX}${LIB_NAME}${CMAKE_STATIC_LIBRARY_SUFFIX}")
  set(CARGO_LIB_FILE
    "${CARGO_TARGET_DIR}/${LIB_TARGET}/${LIB_BUILD_TYPE}/${STATIC_LIB_NAME}")
  set(CARGO_LOCAL_ARTIFACT_DIR "${CMAKE_CURRENT_BINARY_DIR}/cargo-artifacts")
  set(LIB_FILE "${CARGO_LOCAL_ARTIFACT_DIR}/${STATIC_LIB_NAME}")

  message(STATUS
    "running: ${CARGO_ENV_COMMAND} ${CARGO_EXECUTABLE} ${CARGO_ARGS}")

  # Always enter Cargo so its content- and toolchain-aware fingerprint can
  # validate a persistent target directory. Copy the result into this CMake
  # build tree so Ninja clean never removes the shared Cargo cache artifact.
  add_custom_target(${CARGO_NAME}_target ALL
      COMMAND ${CARGO_ENV_COMMAND} ${CARGO_EXECUTABLE} ${CARGO_ARGS}
      COMMAND ${CMAKE_COMMAND} -E make_directory "${CARGO_LOCAL_ARTIFACT_DIR}"
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
              "${CARGO_LIB_FILE}" "${LIB_FILE}"
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
      BYPRODUCTS "${LIB_FILE}"
      DEPENDS ${CARGO_INPUTS}
      COMMENT "Building cargo library ${LIB_FILE}"
      VERBATIM
  )

  # add the library target
  add_library(${CARGO_NAME} STATIC IMPORTED GLOBAL)
  add_dependencies(${CARGO_NAME} ${CARGO_NAME}_target)
  set_target_properties(${CARGO_NAME} PROPERTIES 
    IMPORTED_LOCATION ${LIB_FILE}
  )
  target_sources(${CARGO_NAME} INTERFACE ${CARGO_HDRS})
endfunction()

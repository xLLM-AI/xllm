# Copyright 2026 The xLLM Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0.

function(xllm_configure_datasystem_allocator)
  if(NOT USE_NPU OR NOT USE_YUANRONG_DATASYSTEM OR
     NOT DATASYSTEM_AVOID_TCMALLOC)
    return()
  endif()
  if(NOT TARGET leveldb::leveldb)
    message(FATAL_ERROR "DataSystem allocator setup requires leveldb::leveldb")
  endif()
  get_target_property(_leveldb_links leveldb::leveldb INTERFACE_LINK_LIBRARIES)
  if(NOT _leveldb_links)
    return()
  endif()
  # LevelDB's optional allocator interposes allocations across the SDK DSO.
  # Preserve every dependency except the exact allocator target observed
  # in the October repair. Do not rewrite imported package files.
  list(FIND _leveldb_links "tcmalloc" _tcmalloc_index)
  if(NOT _tcmalloc_index EQUAL -1)
    list(REMOVE_ITEM _leveldb_links "tcmalloc")
    set_property(TARGET leveldb::leveldb PROPERTY
      INTERFACE_LINK_LIBRARIES "${_leveldb_links}")
    message(STATUS "DataSystem: removed LevelDB's optional tcmalloc link")
  endif()
endfunction()

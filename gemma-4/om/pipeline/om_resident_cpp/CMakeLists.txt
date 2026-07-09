cmake_minimum_required(VERSION 3.10)
project(om_resident_daemon CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT ASCEND_INCLUDE)
  if(DEFINED ENV{ASCEND_INCLUDE})
    set(ASCEND_INCLUDE $ENV{ASCEND_INCLUDE})
  elseif(DEFINED ENV{DDK_PATH})
    set(ASCEND_INCLUDE "$ENV{DDK_PATH}/acllib/include")
  else()
    set(ASCEND_INCLUDE "/usr/local/Ascend/ascend-toolkit/latest/acllib/include")
  endif()
endif()

if(NOT ASCEND_LIB)
  if(DEFINED ENV{ASCEND_LIB})
    set(ASCEND_LIB $ENV{ASCEND_LIB})
  elseif(DEFINED ENV{NPU_HOST_LIB})
    set(ASCEND_LIB $ENV{NPU_HOST_LIB})
  else()
    set(ASCEND_LIB "/usr/local/Ascend/ascend-toolkit/latest/acllib/lib64/stub")
  endif()
endif()

message(STATUS "ASCEND_INCLUDE=${ASCEND_INCLUDE}")
message(STATUS "ASCEND_LIB=${ASCEND_LIB}")

if(NOT EXISTS "${ASCEND_INCLUDE}/acl/acl.h")
  message(FATAL_ERROR "acl/acl.h not found under ${ASCEND_INCLUDE}")
endif()

include_directories(${ASCEND_INCLUDE})
link_directories(${ASCEND_LIB})

add_executable(om_resident_daemon
  src/main.cpp
  src/acl_model.cpp
  src/job_queue.cpp
)

target_link_libraries(om_resident_daemon PRIVATE ascendcl pthread)

set_target_properties(om_resident_daemon PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/out"
  BUILD_RPATH "${ASCEND_LIB}"
)

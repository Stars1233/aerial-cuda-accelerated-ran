# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

cmake_minimum_required (VERSION 3.25)

project (cuphyoam LANGUAGES CXX)

option(ENABLE_YANG_PARSER "Enable YANG parser with libyang-cpp dependency" ON)

add_library(cuphyoamlib SHARED src/cuphyoam.cpp)
set_property(TARGET cuphyoamlib PROPERTY POSITION_INDEPENDENT_CODE ON)

target_sources(cuphyoamlib PRIVATE
    src/common_svc.cpp
    src/cus_conn_mgr.cpp
    src/push_notification_svc.cpp
    src/scf_fapi_oam_services.cpp
    src/yang_model_manager.cpp
    src/utils.cpp
)

find_package(PkgConfig REQUIRED)

# Use system libyang and libyang-cpp (conditional)
if(ENABLE_YANG_PARSER)
    pkg_check_modules(LIBYANG REQUIRED libyang>=2.1.88  IMPORTED_TARGET)
    message(STATUS "LIBYANG version = ${LIBYANG_VERSION}")
    message(STATUS "LIBYANG dir= ${LIBYANG_LIBDIR}")

    pkg_check_modules(LIBYANG-CPP REQUIRED libyang-cpp>=1.1.0  IMPORTED_TARGET)
    message(STATUS "LIBYANG cpp version = ${LIBYANG-CPP_VERSION}")
    message(STATUS "LIBYANG cpp dir= ${LIBYANG-CPP_LIBDIR}")
    
    target_compile_definitions(cuphyoamlib PRIVATE ENABLE_YANG_PARSER=1)
endif()

# Link with internal targets and conditional libyang dependencies
target_link_libraries(cuphyoamlib PUBLIC nvipc nvlog app_config)

# Add conditional libyang dependencies
if(ENABLE_YANG_PARSER)
    target_link_libraries(cuphyoamlib PUBLIC PkgConfig::LIBYANG PkgConfig::LIBYANG-CPP)
endif()
target_include_directories(cuphyoamlib PUBLIC ${DPDK_INCLUDE_DIRS})

# Try modern config-based Protobuf first, fall back to legacy FindProtobuf.cmake
find_package(Protobuf CONFIG QUIET)
if(NOT Protobuf_FOUND)
    find_package(Protobuf REQUIRED)
endif()

# Try modern config-based gRPC first, fall back to legacy FindgRPC.cmake
find_package(gRPC CONFIG QUIET)
if(NOT gRPC_FOUND)
    find_package(gRPC REQUIRED)
endif()


target_include_directories(cuphyoamlib PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/inc>)
target_include_directories(cuphyoamlib PUBLIC ${CMAKE_CURRENT_BINARY_DIR})

# Use system protoc and grpc plugins
find_program(_PROTOBUF_PROTOC protoc REQUIRED)
find_program(_GRPC_CPP_PLUGIN_EXECUTABLE grpc_cpp_plugin REQUIRED)
find_program(_GRPC_PYTHON_PLUGIN_EXECUTABLE grpc_python_plugin REQUIRED)

message(STATUS "Using system protoc: ${_PROTOBUF_PROTOC}")
message(STATUS "Using system grpc_cpp_plugin: ${_GRPC_CPP_PLUGIN_EXECUTABLE}")
message(STATUS "Using system grpc_python_plugin: ${_GRPC_PYTHON_PLUGIN_EXECUTABLE}")


foreach(_proto
  aerial_common
  aerial_push_notification
  scf_fapi_p9
)
  get_filename_component(cur_proto "proto/${_proto}.proto" ABSOLUTE)
  get_filename_component(cur_proto_path "${cur_proto}" PATH)

  # Proto file
  get_filename_component(cur_proto "proto/${_proto}.proto" ABSOLUTE)
  get_filename_component(cur_proto_path "${cur_proto}" PATH)

  # Generated sources
  set(cur_proto_srcs "${CMAKE_CURRENT_BINARY_DIR}/${_proto}.pb.cc")
  set(cur_proto_hdrs "${CMAKE_CURRENT_BINARY_DIR}/${_proto}.pb.h")
  set(cur_grpc_srcs "${CMAKE_CURRENT_BINARY_DIR}/${_proto}.grpc.pb.cc")
  set(cur_grpc_hdrs "${CMAKE_CURRENT_BINARY_DIR}/${_proto}.grpc.pb.h")
  set(cur_proto_py_srcs "${CMAKE_CURRENT_BINARY_DIR}/${_proto}_pb2.py")
  set(cur_grpc_py_srcs "${CMAKE_CURRENT_BINARY_DIR}/${_proto}_pb2_grpc.py")
  add_custom_command(
        OUTPUT "${cur_proto_srcs}" "${cur_proto_hdrs}" "${cur_grpc_srcs}" "${cur_grpc_hdrs}" "${cur_proto_py_srcs}" "${cur_grpc_py_srcs}"
        COMMAND ${_PROTOBUF_PROTOC}
        ARGS --grpc_out "${CMAKE_CURRENT_BINARY_DIR}"
          --cpp_out "${CMAKE_CURRENT_BINARY_DIR}"
          --experimental_allow_proto3_optional
          -I "${cur_proto_path}"
          --plugin=protoc-gen-grpc="${_GRPC_CPP_PLUGIN_EXECUTABLE}"
          "${cur_proto}"
        COMMAND ${_PROTOBUF_PROTOC}
        ARGS --grpc_out "${CMAKE_CURRENT_BINARY_DIR}"
          --python_out "${CMAKE_CURRENT_BINARY_DIR}"
          --experimental_allow_proto3_optional
          -I "${cur_proto_path}"
          --plugin=protoc-gen-grpc="${_GRPC_PYTHON_PLUGIN_EXECUTABLE}"
          "${cur_proto}"
        DEPENDS "${cur_proto}")

  target_sources(cuphyoamlib PRIVATE
      ${cur_proto_srcs}
      ${cur_grpc_srcs})
endforeach()

target_link_libraries(cuphyoamlib PRIVATE gRPC::grpc++ protobuf gRPC::grpc++_reflection)

function(link_dpdk _target)
	# Link to DPDK shared libraries
	target_compile_options(${_target} PRIVATE ${DPDK_CFLAGS})
	target_include_directories(${_target} PRIVATE ${DPDK_INCLUDE_DIRS})
	target_link_directories(${_target} PRIVATE ${DPDK_LIBRARY_DIRS})
	target_link_libraries(${_target} PRIVATE -Wl,--no-as-needed)
	target_link_libraries(${_target} PRIVATE ${DPDK_LIBRARIES})
endfunction()

add_executable(test_cuphyoam examples/test_cuphyoam.cpp)
add_executable(test_basic_dpdk examples/test_basic_dpdk.cpp)
add_executable(test_time examples/test_time.cpp)
add_executable(test_grpc_stream examples/test_grpc_stream.cpp)
add_executable(test_grpc_cpp_client examples/test_grpc_cpp_client.cpp)
add_executable(test_grpc_push_notification_client examples/test_grpc_push_notification_client.cpp)
add_executable(p9_msg_client_grpc_test examples/p9_msg_client_grpc_test.cpp)

if(ENABLE_YANG_PARSER)
    add_executable(libyang_test examples/libyang_test.cpp)
endif()

add_executable(cus_conn_mgr examples/cus_conn_mgr.cpp)


target_link_libraries(test_cuphyoam PRIVATE cuphyoamlib)
target_link_libraries(test_basic_dpdk PRIVATE cuphyoamlib)
target_link_libraries(test_time PRIVATE cuphyoamlib)
target_link_libraries(test_grpc_stream PRIVATE cuphyoamlib)
target_link_libraries(test_grpc_cpp_client PRIVATE gRPC::grpc++ cuphyoamlib)
target_link_libraries(test_grpc_push_notification_client PRIVATE gRPC::grpc++ cuphyoamlib)
target_link_libraries(p9_msg_client_grpc_test PRIVATE gRPC::grpc++ cuphyoamlib)

if(ENABLE_YANG_PARSER)
    target_link_libraries(libyang_test PRIVATE PkgConfig::LIBYANG PkgConfig::LIBYANG-CPP nvlog)
    target_compile_features(libyang_test PRIVATE cxx_std_20)
endif()

target_link_libraries(cus_conn_mgr PRIVATE gRPC::grpc++ cuphyoamlib)

link_dpdk(test_time)

# DPDK import techniques

if (0)
   IF(NOT TARGET dpdk_target)
      SET(DPDK_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../../gpu-dpdk)
      INCLUDE(${CMAKE_CURRENT_SOURCE_DIR}/../../cmake/dpdk.cmake)
   ENDIF()

   set_target_properties(cuphyoamlib PROPERTIES COMPILE_FLAGS ${DPDK_BUILD_FLAGS})
   target_link_libraries(test_cuphyoam ${DPDK_LIBS})
   set_target_properties(test_cuphyoam PROPERTIES COMPILE_FLAGS ${DPDK_BUILD_FLAGS})
   message(WARNING "DPDK_BUILD_FLAGS: ${DPDK_BUILD_FLAGS}")
   message(WARNING "DPDK_LIBS: ${DPDK_LIBS}")
endif()

if (0)
   set(DPDK_TARGET_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../../gpu-dpdk/build)
   SET(ENV{PKG_CONFIG_PATH} "${DPDK_TARGET_PATH}/meson-private")
   EXECUTE_PROCESS(OUTPUT_VARIABLE DPDK_BUILD_FLAGS
      OUTPUT_STRIP_TRAILING_WHITESPACE
      WORKING_DIRECTORY ${DPDK_TARGET_PATH}/meson-private
   	  COMMAND pkg-config --cflags libdpdk.pc)

   EXECUTE_PROCESS(OUTPUT_VARIABLE DPDK_LINK_FLAGS
      OUTPUT_STRIP_TRAILING_WHITESPACE
      WORKING_DIRECTORY ${DPDK_TARGET_PATH}/meson-private
      COMMAND pkg-config --static --libs libdpdk.pc)

   STRING(REPLACE "-Wl,--no-whole-archive" "-Wl,--whole-archive" DPDK_LINK_FLAGS ${DPDK_LINK_FLAGS})
   STRING(REPLACE "-lpthread" "" DPDK_LINK_FLAGS ${DPDK_LINK_FLAGS})
   SET(DPDK_LIBS "${DPDK_LINK_FLAGS} -Wl,--no-whole-archive -pthread -lnuma -ldl")

   set_target_properties(cuphyoamlib PROPERTIES COMPILE_FLAGS ${DPDK_BUILD_FLAGS})
   target_link_libraries(test_cuphyoam ${DPDK_LIBS})
   target_link_libraries(test_basic_dpdk ${DPDK_LIBS})
   target_link_libraries(test_time ${DPDK_LIBS})
   set_target_properties(test_cuphyoam PROPERTIES COMPILE_FLAGS ${DPDK_BUILD_FLAGS})
   set_target_properties(test_basic_dpdk PROPERTIES COMPILE_FLAGS ${DPDK_BUILD_FLAGS})
   set_target_properties(test_time PROPERTIES COMPILE_FLAGS ${DPDK_BUILD_FLAGS})
endif()

if (0)
   set(DPDK_LIBS "")
   foreach(x IN LISTS DPDK_STATIC_LDFLAGS)
      STRING(REPLACE "-Wl,--no-whole-archive" "-Wl,--whole-archive" TMP1 "${x}")
      STRING(REPLACE "-lpthread" "" TMP1 ${TMP1})
      list(APPEND DPDK_LIBS ${TMP1})
   endforeach()

   list(APPEND DPDK_LIBS "-Wl,--no-whole-archive -pthread -lnuma -ldl")

   target_include_directories(cuphyoamlib PRIVATE ${DPDK_INCLUDE_DIRS})
   #target_link_libraries(cuphyoamlib "${DPDK_LIBS}")
   target_compile_options(cuphyoamlib PRIVATE ${DPDK_CFLAGS_OTHER})

   target_include_directories(test_basic_dpdk PRIVATE ${DPDK_INCLUDE_DIRS})
   target_link_libraries(test_basic_dpdk "${DPDK_LIBS}")
   target_compile_options(test_basic_dpdk PRIVATE ${DPDK_CFLAGS_OTHER})

   target_include_directories(test_time PRIVATE ${DPDK_INCLUDE_DIRS})
   target_link_libraries(test_time "${DPDK_LIBS}")
   target_compile_options(test_time PRIVATE ${DPDK_CFLAGS_OTHER})

   target_include_directories(test_grpc_stream PRIVATE ${DPDK_INCLUDE_DIRS})
   target_link_libraries(test_grpc_stream "${DPDK_LIBS}")
   target_compile_options(test_grpc_stream PRIVATE ${DPDK_CFLAGS_OTHER})

   target_include_directories(test_cuphyoam PRIVATE ${DPDK_INCLUDE_DIRS})
   target_link_libraries(test_cuphyoam "${DPDK_LIBS}")
   target_compile_options(test_cuphyoam PRIVATE ${DPDK_CFLAGS_OTHER})
endif()

if (0)
   SET(ENV{PKG_CONFIG_PATH} "${DPDK_TARGET_PATH}/meson-private")
   find_package(PkgConfig REQUIRED)
   pkg_check_modules(DPDK REQUIRED IMPORTED_TARGET libdpdk)

   #message(WARNING "DPDK_FOUND: ${DPDK_FOUND}")
   #message(WARNING "DPDK_INCLUDE_DIRS: ${DPDK_INCLUDE_DIRS}")
   #message(WARNING "DPDK_LDFLAGS: ${DPDK_LDFLAGS}")
   target_link_libraries(cuphyoamlib PkgConfig::DPDK)
   target_link_libraries(test_cuphyoam cuphyoamlib PkgConfig::DPDK)
endif()


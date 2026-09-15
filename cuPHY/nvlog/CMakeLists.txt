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

include(GNUInstallDirs)


# We do not need aerial_sdk_version for standalone nvipc
# The entire vanilla build is built with NVIPC_FMTLOG_ENABLE=ON
# but standalone one is built with NVIPC_FMTLOG_ENABLE=OFF
# which will not build it and will not add it to any deps below.
if(DEFINED NVIPC_FMTLOG_ENABLE AND NVIPC_FMTLOG_ENABLE)
    # YAML SDK version check header-only library
    add_library(aerial_sdk_version INTERFACE)

    # Add header files to help IDEs recognize them
    target_sources(aerial_sdk_version INTERFACE
        ${CMAKE_CURRENT_LIST_DIR}/include/yaml_sdk_version.hpp
        ${CMAKE_CURRENT_LIST_DIR}/include/yaml_sdk_version.tpp
    )

    target_include_directories(aerial_sdk_version INTERFACE ${CMAKE_CURRENT_LIST_DIR}/include)
    find_package(fmt REQUIRED)
    target_link_libraries(aerial_sdk_version INTERFACE yaml yaml-cpp fmt::fmt)
    target_compile_definitions(aerial_sdk_version INTERFACE AERIAL_SDK_VERSION=\"${AERIAL_SDK_VERSION}\")
endif () # NVIPC_FMTLOG_ENABLE

# NVLOG SHARED library
set(SOURCES src/nv_utils.c
            src/nvlog.c
            src/stat_log.c
            src/aerial_event_code.c
            src/exit_handler.cpp
            )

if(NVIPC_FMTLOG_ENABLE)
    list(APPEND SOURCES src/nvlog.cpp)
endif()

######## libnvlog.so ###################
add_library(nvlog SHARED ${SOURCES})
set_property(TARGET nvlog PROPERTY POSITION_INDEPENDENT_CODE ON)
target_include_directories(nvlog PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(nvlog PUBLIC rt -pthread yaml yaml-cpp)
if(NVIPC_FMTLOG_ENABLE)
    target_link_libraries(nvlog PUBLIC aerial_sdk_version PUBLIC fmtlog-shared fmt::fmt)
endif()
target_compile_options(nvlog PRIVATE ${AERIAL_ARCH_TUNE_FLAGS})

set_target_properties(nvlog PROPERTIES PUBLIC_HEADER
   $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)

install(TARGETS nvlog
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/../)

######## libnvlog_static.a #############
add_library(nvlog_static STATIC ${SOURCES})
set_property(TARGET nvlog_static PROPERTY POSITION_INDEPENDENT_CODE ON)
target_include_directories(nvlog_static PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
target_include_directories(nvlog_static INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(nvlog_static PUBLIC rt -pthread yaml yaml-cpp)
if(NVIPC_FMTLOG_ENABLE)
    target_link_libraries(nvlog_static PUBLIC aerial_sdk_version fmtlog-static fmt::fmt)
endif()
target_compile_options(nvlog_static PRIVATE ${AERIAL_ARCH_TUNE_FLAGS})

if(NVIPC_FMTLOG_ENABLE AND ENABLE_TESTS)
    add_subdirectory(test)
endif()

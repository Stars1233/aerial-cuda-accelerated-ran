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

cmake_minimum_required(VERSION 3.25 FATAL_ERROR)

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(TARGET app_utils)
set(SOURCES app_utils.cpp
            app_utils.hpp
            ptp_service_status_checking.cpp
            ptp_service_status_checking.hpp
            rhocp_ptp_event_consumer.cpp
            rhocp_ptp_event_consumer.hpp)

add_library(${TARGET} SHARED ${SOURCES})

find_package(CURL REQUIRED)
find_package(httplib REQUIRED)
find_package(OpenSSL REQUIRED)

target_compile_definitions(${TARGET} PUBLIC CPPHTTPLIB_OPENSSL_SUPPORT)

target_include_directories(${TARGET} PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
)

target_link_libraries(${TARGET} PUBLIC nvlog CURL::libcurl httplib::httplib OpenSSL::SSL OpenSSL::Crypto)

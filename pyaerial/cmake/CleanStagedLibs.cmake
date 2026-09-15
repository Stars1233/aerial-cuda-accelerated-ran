# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

cmake_minimum_required(VERSION 3.25)

if(NOT PYAERIAL_PYCUPHY_PKG_DIR)
    message(FATAL_ERROR "PYAERIAL_PYCUPHY_PKG_DIR is required")
endif()

file(GLOB _pyaerial_staged_libs "${PYAERIAL_PYCUPHY_PKG_DIR}/*.so*")
foreach(_pyaerial_lib IN LISTS _pyaerial_staged_libs)
    get_filename_component(_pyaerial_lib_name "${_pyaerial_lib}" NAME)
    if(NOT _pyaerial_lib_name MATCHES "^_pycuphy.*\\.so")
        file(REMOVE "${_pyaerial_lib}")
    endif()
endforeach()

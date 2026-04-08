# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# By default, "${CMAKE_SOURCE_DIR}/aerial-sdk-version" includes "26-1-cubb"
# During packaging, this string will be replaced with a specific version.
# Below cmake code we will read it and populate AERIAL_SDK_VERSION
# The value of AERIAL_SDK_VERSION will be compared against distributed YAML files with
# a released version. This way we make sure that both the software built/running and the
# YAML files came out from the same distribution.
# If users mix old YAMLs (for example) cuphydriver and similar executables will bail out saying that
# the tags that the SW is compiled with, and the YAML files' version attribute - these are having a
# mismatch.
# AERIAL_SDK_VERSION is passed via compile time with -DAERIAL_SDK_VERSION="..." and is used during
# runtime to check the YAML version attribute.

function(read_aerial_sdk_version_file aerial_sdk_version_dir_location)
    if (NOT AERIAL_SDK_VERSION)
        set(AERIAL_SDK_VERSION_FILE "${aerial_sdk_version_dir_location}/aerial-sdk-version")
        if (NOT AERIAL_SDK_VERSION_FILE)
            message(FATAL_ERROR "Error cannot find aerial-sdk-version file in directory ${aerial_sdk_version_dir_location}, halting")
        endif()
        file(READ ${AERIAL_SDK_VERSION_FILE} AERIAL_SDK_VERSION)
        string(STRIP "${AERIAL_SDK_VERSION}" AERIAL_SDK_VERSION) # strip \n EOL
        message(STATUS "The content of file: ${AERIAL_SDK_VERSION_FILE} is: ${AERIAL_SDK_VERSION}, assigned to AERIAL_SDK_VERSION")
        set(AERIAL_SDK_VERSION ${AERIAL_SDK_VERSION} PARENT_SCOPE)
    endif () # NOT AERIAL_SDK_VERSION
endfunction()

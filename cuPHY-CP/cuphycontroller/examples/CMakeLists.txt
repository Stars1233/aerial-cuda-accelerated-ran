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

set(TARGET_SCF cuphycontroller_scf)

set(SOURCES cuphycontroller_scf.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/../src/yamlparser.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/../src/cuphydriver.cpp
)

find_package(OpenSSL REQUIRED)

add_executable(${TARGET_SCF} ${SOURCES})
add_dependencies(${TARGET_SCF} cuphydriver scf_5g_fapi nvphy cuphyoamlib app_config)
if (AERIAL_METRICS)
    add_dependencies(${TARGET_SCF} aerial_metrics)
endif()

target_include_directories(${TARGET_SCF} PUBLIC /usr/local/include)
target_include_directories(${TARGET_SCF} PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>)
target_include_directories(${TARGET_SCF} PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../lib/cuphydriver/include/>)

target_compile_options(${TARGET_SCF} PRIVATE ${AERIAL_ARCH_TUNE_FLAGS})

# required in case of nvtx
target_link_directories(${TARGET_SCF} PRIVATE "${CUDA_TOOLKIT_ROOT_DIR}/lib64")

target_link_libraries(${TARGET_SCF}
    yaml
    cuphydriver
    cuda_driver_utils
    # scf_5g_fapi and nvphy have circular static-archive refs; rescan resolves both.
    "$<LINK_GROUP:RESCAN,scf_5g_fapi,nvphy>"
    nvlog
    cuphyoamlib
    mimalloc
    app_config
    aerial_sdk_version
    app_utils
    CUDA::cudart)

if (AERIAL_METRICS)
    target_link_libraries(${TARGET_SCF} aerial_metrics)
endif()

#Add path to DOCA GPU DPDK libraries
target_link_directories(${TARGET_SCF} PRIVATE ${DPDK_LIBRARY_DIRS})
target_link_dpdk_pmd_directories(${TARGET_SCF})

# To aviod /usr/lib/aarch64-linux-gnu/libcrypto.so and /usr/local/lib/libcrypto.so coexist conflict
target_link_libraries(${TARGET_SCF} OpenSSL::Crypto OpenSSL::SSL)

install(PROGRAMS ${CMAKE_CURRENT_BINARY_DIR}/cuphycontroller DESTINATION .)
install(PROGRAMS ${PROJECT_SOURCE_DIR}/config/cuphycontroller.yaml
    ${PROJECT_SOURCE_DIR}/config/launch_pattern.yaml
    ${PROJECT_SOURCE_DIR}/config/l2_adapter_config.yaml DESTINATION .)

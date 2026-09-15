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

# Set name of excutable 
set(TARGET cubb_gpu_test_bench)

# Opt-in: build the cuMAC AI-RAN trtEngine inference subcontext into the cuMAC
# worker so the test bench can run cuPHY + cuMAC trtEngine inference GPU-sharing
# tests (cumac_options.airan.infer=1). Pulls in the ml trtEngine sources and
# TensorRT. Off by default so existing builds are unaffected. Requires ENABLE_CUMAC.
option(ENABLE_CUMAC_AIRAN "Enable cuMAC AI-RAN trtEngine inference subcontext in the test bench" OFF)

# set sources
set(CUPHY_EXAMPLE_DIR ../../cuPHY/examples)

set(SOURCES ${CUPHY_EXAMPLE_DIR}/common/util.cu ${CUPHY_EXAMPLE_DIR}/pusch_rx_multi_pipe/pusch_rx_test.cpp testbench_common.cpp cuphy_testWrkr.cpp cubb_gpu_test_bench.cpp ${CUPHY_EXAMPLE_DIR}/common/datasets.cpp)

# extra include dirs / libraries appended for the optional AI-RAN inference build
set(CUMAC_AIRAN_INCLUDE_DIRS "")
set(CUMAC_AIRAN_LIBS "")

# Conditionally add cuMAC sources and set compile definition
if(ENABLE_CUMAC)
    add_compile_definitions(AERIAL_CUMAC_ENABLE)
    set(SOURCES ${SOURCES} cumac_testWrkr.cpp)
    set(CUMAC_DIR ../../cuMAC)

    if(ENABLE_CUMAC_AIRAN)
        add_compile_definitions(AERIAL_CUMAC_AIRAN_ENABLE)
        find_library(NVINFER nvinfer)
        find_library(NVONNXPARSER nvonnxparser)
        if(NOT NVINFER OR NOT NVONNXPARSER)
            message(FATAL_ERROR "ENABLE_CUMAC_AIRAN requires TensorRT (libnvinfer, libnvonnxparser) but libraries were not found")
        endif()
        set(CUMAC_ML_SRC_DIR ${CUMAC_DIR}/examples/ml)
        # trtEngine + AI-RAN inference subcontext + TV helper sources
        set(SOURCES ${SOURCES}
            ${CUMAC_ML_SRC_DIR}/cumacAiranSubcontext.cpp
            ${CUMAC_ML_SRC_DIR}/airanTvUtil.cpp
            ${CUMAC_ML_SRC_DIR}/trtEngine.cpp)
        set(CUMAC_AIRAN_INCLUDE_DIRS ${CUMAC_ML_SRC_DIR})
        set(CUMAC_AIRAN_LIBS ${NVINFER} ${NVONNXPARSER})
        message(STATUS "cubb_gpu_test_bench: AI-RAN trtEngine inference enabled (TensorRT: ${NVINFER}, ${NVONNXPARSER})")
    endif()
endif()

# add executable
add_executable(${TARGET} ${SOURCES})

# set include directories
if(ENABLE_CUMAC)
    target_include_directories(${TARGET} PRIVATE ${CUPHY_EXAMPLE_DIR}/pusch_rx_multi_pipe ${CUPHY_EXAMPLE_DIR}/pdsch_tx ${CUPHY_EXAMPLE_DIR}/common ${CUMAC_DIR}/src ${CUMAC_DIR}/lib/cumac_msg ${CUMAC_DIR}/examples ${CUMAC_DIR}/examples/tools ${HDF5_CXX_INCLUDE_DIRS} ${CUMAC_DIR}/examples/tvLoadingTest ${CUMAC_AIRAN_INCLUDE_DIRS})
    # set libraries with cuMAC                                           
    target_link_libraries(${TARGET} PRIVATE cuphy_channels cuphy_hdf5 cuphy Threads::Threads yaml cumac ${HDF5_CXX_LIBRARIES} ${CUMAC_AIRAN_LIBS})
else()
    target_include_directories(${TARGET} PRIVATE ${CUPHY_EXAMPLE_DIR}/pusch_rx_multi_pipe ${CUPHY_EXAMPLE_DIR}/pdsch_tx ${CUPHY_EXAMPLE_DIR}/common ${HDF5_CXX_INCLUDE_DIRS})
    # set libraries without cuMAC
    target_link_libraries(${TARGET} PRIVATE cuphy_channels cuphy_hdf5 cuphy Threads::Threads yaml ${HDF5_CXX_LIBRARIES})
endif()

# add CMakeLists.release.txt
install(FILES CMakeLists.release.txt DESTINATION testBenches/cubb_gpu_test_bench RENAME CMakeLists.txt)

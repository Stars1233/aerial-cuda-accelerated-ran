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

include_directories(common)
find_package(gsl-lite REQUIRED)
add_library(cuphy_channels STATIC pusch_rx.cpp pdsch_tx.cpp pdcch_tx.cpp pucch_rx.cpp csirs_tx.cpp csirs_rx.cpp prach_rx.cpp ssb_tx.cpp bfw_tx.cpp srs_rx.cpp srs_tx.cpp)
set_source_files_properties(pucch_rx.cpp PROPERTIES LANGUAGE CUDA)
set_property(TARGET cuphy_channels PROPERTY POSITION_INDEPENDENT_CODE ON)

target_include_directories(cuphy_channels PRIVATE ../../examples/common)
target_include_directories(cuphy_channels PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
                                             $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
                                             ${HDF5_INCLUDE_DIRS})

target_compile_options(cuphy_channels PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                       ${CUPHY_NVCC_FLAGS}
                       >)

target_compile_options(cuphy_channels PRIVATE ${AERIAL_ARCH_TUNE_FLAGS})

target_link_libraries(cuphy_channels PUBLIC cuphy_hdf5 cuphy gsl-lite::gsl-lite ${HDF5_C_LIBRARIES})

install(TARGETS cuphy_channels DESTINATION lib)
install(FILES pusch_rx.hpp pdsch_tx.hpp pdcch_tx.hpp pucch_rx.hpp csirs_tx.hpp csirs_rx.hpp prach_rx.hpp ssb_tx.hpp bfw_tx.hpp srs_rx.hpp srs_tx.hpp DESTINATION include)

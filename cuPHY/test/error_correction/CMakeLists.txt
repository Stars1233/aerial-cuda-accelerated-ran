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

########################################################################
# test_ldpc_internal_app_addr
set(SRCS test_ldpc_internal_app_addr.cu
         ${PROJECT_SOURCE_DIR}/src/cuphy/error_correction/ldpc2_bg_desc_8.cpp
         ${PROJECT_SOURCE_DIR}/src/cuphy/error_correction/ldpc2_bg_desc_16.cpp
         ${PROJECT_SOURCE_DIR}/src/cuphy/error_correction/ldpc2_bg_desc_nzs_16.cpp
         ${PROJECT_SOURCE_DIR}/src/cuphy/error_correction/ldpc2_bg_adj_desc_8.cpp
         ${PROJECT_SOURCE_DIR}/src/cuphy/error_correction/ldpc2_bg_adj_desc_16.cpp
         ${PROJECT_SOURCE_DIR}/src/cuphy/error_correction/ldpc2_bg_adj_desc_32.cpp
         ${PROJECT_SOURCE_DIR}/src/cuphy/error_correction/ldpc2_bg_adj_desc_nzs_16.cpp
         ${PROJECT_SOURCE_DIR}/src/cuphy/error_correction/ldpc2_bg_adj_desc_nzs_32.cpp)
add_executable(test_ldpc_internal_app_addr ${SRCS})
target_include_directories(test_ldpc_internal_app_addr PRIVATE ${PROJECT_SOURCE_DIR}/examples/common)
target_link_libraries(test_ldpc_internal_app_addr cuphy)
target_link_libraries(test_ldpc_internal_app_addr gtest gtest_main)
target_compile_options(test_ldpc_internal_app_addr PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                       -lineinfo
                       >)
add_test(NAME test_ldpc_internal_app_addr COMMAND test_ldpc_internal_app_addr)
#add_test(NAME test_ldpc_internal COMMAND test_ldpc_internal WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/test/error_correction)

########################################################################
# test_ldpc_internal_rc
add_executable(test_ldpc_internal_rc test_ldpc_internal_rc.cu)
target_include_directories(test_ldpc_internal_rc PRIVATE ${PROJECT_SOURCE_DIR}/examples/common)
target_link_libraries(test_ldpc_internal_rc cuphy)
target_link_libraries(test_ldpc_internal_rc gtest gtest_main)
target_compile_options(test_ldpc_internal_rc PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                       -lineinfo
                       >)
add_test(NAME test_ldpc_internal_rc COMMAND test_ldpc_internal_rc)

########################################################################
# test_ldpc_internal_loader
add_executable(test_ldpc_internal_loader test_ldpc_internal_loader.cu)
target_include_directories(test_ldpc_internal_loader PRIVATE ${PROJECT_SOURCE_DIR}/examples/common)
target_link_libraries(test_ldpc_internal_loader cuphy)
target_link_libraries(test_ldpc_internal_loader gtest gtest_main)
target_compile_options(test_ldpc_internal_loader PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                       -lineinfo
                       >)
add_test(NAME test_ldpc_internal_loader COMMAND test_ldpc_internal_loader)

########################################################################
# test_ldpc_clamp_validation
add_executable(test_ldpc_clamp_validation test_ldpc_clamp_validation.cpp)
target_include_directories(test_ldpc_clamp_validation PRIVATE
                           ${PROJECT_SOURCE_DIR}/examples/error_correction)
target_link_libraries(test_ldpc_clamp_validation cuphy gtest gtest_main)
add_test(NAME test_ldpc_clamp_validation COMMAND test_ldpc_clamp_validation)

########################################################################
# test_ldpc_band_decoder_gates
add_executable(test_ldpc_band_decoder_gates test_ldpc_band_decoder_gates.cpp)
target_include_directories(test_ldpc_band_decoder_gates PRIVATE
                           ${PROJECT_SOURCE_DIR}/src/cuphy)
target_link_libraries(test_ldpc_band_decoder_gates cuphy CUDA::cudart gtest gtest_main)
add_test(NAME test_ldpc_band_decoder_gates COMMAND test_ldpc_band_decoder_gates)

# test_ldpc_cb_kernel_spi
add_executable(test_ldpc_cb_kernel_spi test_ldpc_cb_kernel_spi.cpp)
target_include_directories(test_ldpc_cb_kernel_spi PRIVATE
                           ${PROJECT_SOURCE_DIR}/src/cuphy
                           ${PROJECT_SOURCE_DIR}/examples/common)
target_link_libraries(test_ldpc_cb_kernel_spi cuphy CUDA::cudart gtest gtest_main)
add_test(NAME test_ldpc_cb_kernel_spi COMMAND test_ldpc_cb_kernel_spi)

########################################################################
# test_ldpc_cb_prepare_launch
add_executable(test_ldpc_cb_prepare_launch test_ldpc_cb_prepare_launch.cpp)
target_include_directories(test_ldpc_cb_prepare_launch PRIVATE
                           ${PROJECT_SOURCE_DIR}/src/cuphy
                           ${PROJECT_SOURCE_DIR}/examples/common)
target_link_libraries(test_ldpc_cb_prepare_launch cuphy CUDA::cudart gtest gtest_main)
add_test(NAME test_ldpc_cb_prepare_launch COMMAND test_ldpc_cb_prepare_launch)

#                        --keep
#                       --cudart static

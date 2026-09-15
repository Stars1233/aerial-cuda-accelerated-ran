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

#set(CMAKE_CUDA_FLAGS "-Xcompiler -mlzcnt ${CMAKE_CUDA_FLAGS} -arch=sm_70 -Xptxas=-v")
set(CMAKE_CUDA_FLAGS_DEBUG " -g -G")

# add_executable (oran.x oran.cu)
# target_link_libraries (oran.x comp_decomp_lib)

# add_executable(packets.x packets.cu)
# target_link_libraries(packets.x comp_decomp_lib)

# add_executable(barsync.x barsync.cu)
# target_link_libraries(barsync.x comp_decomp_lib)

add_executable(qam_test QAM_test.cu)
target_link_libraries(qam_test PRIVATE comp_decomp_lib cuda_driver_utils CUDA::curand)

add_executable(qam_param_test QAM_param_test.cu)
target_link_libraries(qam_param_test PRIVATE comp_decomp_lib)

find_package(HDF5 1.10 REQUIRED COMPONENTS C)

add_executable(modcomp_test modcomp_test.cu)
target_include_directories(modcomp_test PUBLIC ../comp_decomp_lib/include ${HDF5_INCLUDE_DIRS} 
                      ${CMAKE_CURRENT_SOURCE_DIR}/../../../cuPHY/examples/common 
                      ${CMAKE_CURRENT_SOURCE_DIR}/../../aerial-fh-driver/include/aerial-fh-driver)

target_link_libraries(modcomp_test PRIVATE comp_decomp_lib cuda_driver_utils nvlog cuphy_hdf5 ${HDF5_C_LIBRARIES})


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

set(SOURCES cuphy_ex_ldpc.cpp
            ldpc_decode_test_vec.hpp
            ldpc_decode_test_vec.cpp
            ldpc_decode_test_vec_file.hpp
            ldpc_decode_test_vec_file.cpp
            ldpc_decode_test_vec_gen.hpp
            ldpc_decode_test_vec_gen.cpp
            ldpc_decode_test_vec_gen_kernels.hpp
            ldpc_decode_test_vec_gen_kernels.cu
            ldpc_decode_test_vec_pybind.hpp
            ldpc_decode_test_vec_pybind.cpp
            ldpc_interm_check.hpp
            ldpc_interm_check.cpp
            dummy.cu)
add_executable(cuphy_ex_ldpc ${SOURCES})
target_link_libraries(cuphy_ex_ldpc cuphy_hdf5 cuphy curand CLI11::CLI11)
target_compile_definitions(cuphy_ex_ldpc PRIVATE
                           CUPHY_LDPC_SPLIT_DUMP_KERNELS=$<BOOL:${CUPHY_LDPC_SPLIT_DUMP_KERNELS}>)

add_executable(cuphy_ex_ldpc_rm
               cuphy_ex_ldpc_rm.cpp
               ldpc_interm_check.hpp
               ldpc_interm_check.cpp
               cuphy_ex_ldpc_util.hpp
               cuphy_ex_ldpc_util.cu
               dummy.cu)
target_link_libraries(cuphy_ex_ldpc_rm cuphy_hdf5 cuphy curand CLI11::CLI11)
target_compile_definitions(cuphy_ex_ldpc_rm PRIVATE
                           CUPHY_LDPC_SPLIT_DUMP_KERNELS=$<BOOL:${CUPHY_LDPC_SPLIT_DUMP_KERNELS}>)

# Codeblock-centric LDPC decoder example
set(CB_SOURCES cuphy_ex_ldpc_cb.cpp
               ldpc_decode_test_vec.hpp
               ldpc_decode_test_vec.cpp
               ldpc_decode_test_vec_file.hpp
               ldpc_decode_test_vec_file.cpp
               ldpc_decode_test_vec_gen.hpp
               ldpc_decode_test_vec_gen.cpp
               ldpc_decode_test_vec_gen_kernels.hpp
               ldpc_decode_test_vec_gen_kernels.cu
               ldpc_decode_test_vec_pusch.hpp
               ldpc_decode_test_vec_pusch.cpp
               ldpc_decode_test_vec_pybind.hpp
               ldpc_decode_test_vec_pybind.cpp
               dummy.cu)
add_executable(cuphy_ex_ldpc_cb ${CB_SOURCES})
target_link_libraries(cuphy_ex_ldpc_cb cuphy_hdf5 cuphy curand CLI11::CLI11)

add_executable(cuphy_ldpc_find_dispatch cuphy_ldpc_find_dispatch.cpp)
# Driver API (cuInit/cuFuncGetName/...) is called directly; the Runtime API
# (cudaMalloc/cudaFree/...) comes in via header-inlined cuphy::tensor_device.
target_link_libraries(cuphy_ldpc_find_dispatch PRIVATE cuphy CUDA::cuda_driver CUDA::cudart CLI11::CLI11)

install(FILES CMakeLists.release.txt DESTINATION examples/error_correction RENAME CMakeLists.txt)

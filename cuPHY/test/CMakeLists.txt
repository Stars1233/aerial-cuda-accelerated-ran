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

find_package(GTest REQUIRED)
find_package(CUDAToolkit REQUIRED COMPONENTS cudart curand)

include_directories("${PROJECT_SOURCE_DIR}/src/cuphy/crc")
include_directories("${PROJECT_SOURCE_DIR}/src/cuphy/descrambling")
# include_directories("${PROJECT_SOURCE_DIR}/src/cuphy/rate_matching")
include_directories("${PROJECT_SOURCE_DIR}/src/cuphy/dl_rate_matching")
include_directories("${PROJECT_SOURCE_DIR}/src/cuphy/rate_matching")
include_directories("${PROJECT_SOURCE_DIR}/src/cuphy/modulation_mapper")
include_directories("${PROJECT_BINARY_DIR}/src/cuphy/LUTS")
include_directories("${PROJECT_SOURCE_DIR}/src/cuphy_hdf5")
include_directories("${PROJECT_SOURCE_DIR}/src/cuphy/error_correction")
include_directories("${PROJECT_SOURCE_DIR}/src/cuphy/pucch_receiver")
include_directories("${PROJECT_SOURCE_DIR}/src/cuphy/soft_demapper")
include_directories("${PROJECT_SOURCE_DIR}/examples/common")
include_directories("${PROJECT_SOURCE_DIR}/examples/pusch_rx_multi_pipe")
add_subdirectory(crc)
add_subdirectory(descrambling)
# add_subdirectory(rate_matching)
add_subdirectory(derate_match_modulo)
add_subdirectory(dl_rate_matching)
add_subdirectory(hdf5)
add_subdirectory(modulation_mapper)
add_subdirectory(error_correction)
add_subdirectory(pucch)
add_subdirectory(soft_demapper)
add_subdirectory(rng)
add_subdirectory(fill)
add_subdirectory(tile)
add_subdirectory(elementwise)
add_subdirectory(reduction)
add_subdirectory(convert)
add_subdirectory(kernelDescr)
add_subdirectory(criticalPathCudaApi)
add_subdirectory(bfw_compress)
add_subdirectory(cuphy_pti_ut)
add_subdirectory(cfo_ta_est)
add_subdirectory(tensor_desc_tests)
add_subdirectory(cuphy)
add_subdirectory(cuphy_api)
add_subdirectory(uciOnPusch_csi2Ctrl)
add_subdirectory(prach_receiver)
add_subdirectory(pucch_F2_front_end)
add_subdirectory(pucch_F3_segLLRs)
add_subdirectory(pusch_noise_intf_est)
add_subdirectory(pucch_F1_receiver)
add_subdirectory(pucch_F0_receiver)
add_subdirectory(polar_decoder)
add_subdirectory(polar_encoder)
add_subdirectory(bfc)
add_subdirectory(pusch_rssi)
add_subdirectory(embed_pdcch_tf_signal)
add_subdirectory(ss)
add_subdirectory(rate_matching)

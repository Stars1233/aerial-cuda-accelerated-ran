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

add_library(cuphy_cpp_header_only INTERFACE)
target_include_directories(cuphy_cpp_header_only INTERFACE common/)

include_directories(../nvlog/include)
include_directories(common)
add_subdirectory(common)
add_subdirectory(test_example)
add_subdirectory(delay_kernel_bench)
add_subdirectory(bfc)
add_subdirectory(channel_estimation)
add_subdirectory(ch_est)
add_subdirectory(cfo_ta_est)
add_subdirectory(channel_eq)
add_subdirectory(csirs_tx_multi_cell)
add_subdirectory(csirs_rx_multi_cell)
add_subdirectory(pusch_rx_multi_pipe)
add_subdirectory(error_correction)
add_subdirectory(ldpc_encode)
add_subdirectory(pdsch_tx)
add_subdirectory(pdsch_tx_multi_cell)
add_subdirectory(ssb_tx_multi_cell)
add_subdirectory(graphs)
add_subdirectory(pucch_F0_receiver)
add_subdirectory(pucch_F1_receiver)
add_subdirectory(pucch_F2_front_end)
add_subdirectory(pucch_F3_front_end)
add_subdirectory(pucch_F3_segLLRs)
add_subdirectory(pucch_F234_uci_seg)
add_subdirectory(simplex_decoder)
add_subdirectory(pucch_receiver)
add_subdirectory(prach_receiver_multi_cell)
add_subdirectory(prach_test)
add_subdirectory(dl_rate_matching)
add_subdirectory(modulation_mapper)
add_subdirectory(pdcch)
add_subdirectory(pdcch_tx_multi_cell)
add_subdirectory(pdsch_dmrs)
add_subdirectory(ss)
add_subdirectory(polar_encoder)
add_subdirectory(sym_demod)
add_subdirectory(pusch_crc)
add_subdirectory(comp_cwTreeTypes)
add_subdirectory(uci_on_pusch_deRm)
add_subdirectory(pucch_rx_pipeline)
add_subdirectory(csi_rs)
add_subdirectory(polar_seg_deRm_deItl)
add_subdirectory(polar_decoder)
add_subdirectory(pusch_rateMatch)
add_subdirectory(ReedMuller_decoder)
add_subdirectory(uciOnPusch_deSeg_part2)
add_subdirectory(uciOnPusch_deSeg_part0)
add_subdirectory(uciOnPusch_csi2_ctrl)
add_subdirectory(srs_chEst)
add_subdirectory(srs_rx_pipeline)
add_subdirectory(srs_tx)
add_subdirectory(gen_pusch_bler_curve)
add_subdirectory(gen_pucch_perf_curve)
add_subdirectory(cuphy_pti_example)
add_subdirectory(cupti_example)
add_subdirectory(srs_bfw)

install(FILES CMakeLists.release.txt DESTINATION examples RENAME CMakeLists.txt)

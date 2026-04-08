/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <cstdint>
#include <ctime>
#include <string>
#include <stdexcept>
#include <yaml-cpp/yaml.h>
#include "cumac_muUeGrp.h"
#include "cumac_msg.h"

constexpr uint16_t MU_TEST_SLOTS_PER_FRAME = 20;
constexpr uint16_t MU_TEST_MAX_SFN = 1024;

struct l1_cumac_message_t {
    uint32_t real_buff_idx;
    uint16_t srs_info_idx;
    uint16_t cell_idx;
};

struct l2_l1_message_t {
    uint16_t nSrsUes; // total number of SRS UEs for the current slot in all cells
    uint32_t arr_usage[0]; // usage flags of the SRS UEs
    uint16_t arr_cell_idx[0]; // cell indices of the SRS UEs
    uint16_t arr_rnti[0]; // RNTIs of the SRS UEs
    uint16_t arr_buffer_Idx[0]; // SRS buffer indices of the SRS UEs
    uint16_t arr_srs_info_idx[0]; // SRS info indices of the SRS UEs
};

struct sys_param_t {
    bool enable_l1_l2_mem_sharing{false}; // enable L1/L2 memory sharing
    bool print_ue_pairing_solution{false}; // print UE pairing solution
    int num_time_slots; // number of time slots
    int cumac_main_thread_core; // cuMAC main thread core
    int l2_main_thread_core; // L2 main thread core
    int l1_main_thread_core; // L1 main thread core
    int l2_cumac_recv_thread_core; // L2 cuMAC receiver thread core
    int l1_l2_recv_thread_core; // L1 L2 receiver thread core
    int cumac_l2_recv_thread_core; // cuMAC L2 receiver thread core
    int cumac_l1_recv_thread_core; // cuMAC L1 receiver thread core

    uint32_t cuda_device_id{0}; // CUDA device ID

    std::string TDD_pattern; // TDD pattern
    int num_cell; // number of cells
    int num_ue_ant_port; // number of antenna ports per UE
    int num_bs_ant_port; // number of antenna ports per RU
    int num_srs_ue_per_cell; // conneccted SRS UEs in each cell
    int num_subband; // number of subbands
    int num_prg_samp_per_subband; // number of per-PRG SRS channel estimate samples per subband
    int num_prg_per_cell; // number of PRGs per cell
    int num_srs_ue_per_slot; // number of SRS UEs scheduled per S-slot
    int max_num_ue_schd_per_cell_tti; // maximum number of UEs scheduled per cell per TTI
    int max_num_ue_for_grp_per_cell; // maximum number of UEs considered for MU-MIMO UE grouping per cell per TTI

    float scs; // subcarrier spacing
    uint64_t slot_interval_ns; // slot interval in nanoseconds

    float srs_chan_est_coeff_var; // * not a system configuration parameter. Used for generating SRS channel estimates.
    
    // CUDA kernel config
    int num_blocks_per_row_chanOrtMat; // cuMAC CUDA kernel configuration parameter

    uint8_t kernel_launch_flags; // bit flags for CUDA kernel launch mode
    // 0x01: whether the channel correlation computation kernel is to be launched.
    // 0x02: whether the UE pairing algorithm kernel is to be launched.

    sys_param_t(const char* yaml_path)
    {
        // load parameters from YAML file
        YAML::Node config = YAML::LoadFile(yaml_path);
        if (config["ENABLE_L1_L2_MEM_SHARING"]) {
            enable_l1_l2_mem_sharing = config["ENABLE_L1_L2_MEM_SHARING"].as<bool>();
        }
        if (config["PRINT_UE_PAIRING_SOLUTION"]) {
            print_ue_pairing_solution = config["PRINT_UE_PAIRING_SOLUTION"].as<bool>();
        }
        if (config["KERNEL_LAUNCH_FLAGS"]) {
            kernel_launch_flags = config["KERNEL_LAUNCH_FLAGS"].as<uint8_t>();
        }
        if (config["TDD_PATTERN"]) {
            TDD_pattern = config["TDD_PATTERN"].as<std::string>();
        }
        if (config["NUM_TIME_SLOTS"]) {
            num_time_slots = config["NUM_TIME_SLOTS"].as<int>();
        }
        if (config["cuMAC_MAIN_THREAD_CORE"]) {
            cumac_main_thread_core = config["cuMAC_MAIN_THREAD_CORE"].as<int>();
        }
        if (config["L2_MAIN_THREAD_CORE"]) {
            l2_main_thread_core = config["L2_MAIN_THREAD_CORE"].as<int>();
        }
        if (config["L1_MAIN_THREAD_CORE"]) {
            l1_main_thread_core = config["L1_MAIN_THREAD_CORE"].as<int>();
        }
        if (config["L2_CUMAC_RECV_THREAD_CORE"]) {
            l2_cumac_recv_thread_core = config["L2_CUMAC_RECV_THREAD_CORE"].as<int>();
        }
        if (config["L1_L2_RECV_THREAD_CORE"]) {
            l1_l2_recv_thread_core = config["L1_L2_RECV_THREAD_CORE"].as<int>();
        }
        if (config["CUMAC_L2_RECV_THREAD_CORE"]) {
            cumac_l2_recv_thread_core = config["CUMAC_L2_RECV_THREAD_CORE"].as<int>();
        }
        if (config["CUMAC_L1_RECV_THREAD_CORE"]) {
            cumac_l1_recv_thread_core = config["CUMAC_L1_RECV_THREAD_CORE"].as<int>();
        }
        if (config["CUDA_DEVICE_ID"]) {
            cuda_device_id = config["CUDA_DEVICE_ID"].as<uint32_t>();
        }
        if (config["NUM_CELL"]) {
            num_cell = config["NUM_CELL"].as<int>();
        }
        if (config["NUM_UE_ANT_PORT"]) {
            num_ue_ant_port = config["NUM_UE_ANT_PORT"].as<int>();
        }
        if (config["NUM_BS_ANT_PORT"]) {
            num_bs_ant_port = config["NUM_BS_ANT_PORT"].as<int>();
        }
        if (config["NUM_SRS_UE_PER_CELL"]) {
            num_srs_ue_per_cell = config["NUM_SRS_UE_PER_CELL"].as<int>();
        }
        if (config["NUM_SUBBAND"]) {
            num_subband = config["NUM_SUBBAND"].as<int>();
        }
        if (config["NUM_PRG_SAMP_PER_SUBBAND"]) {
            num_prg_samp_per_subband = config["NUM_PRG_SAMP_PER_SUBBAND"].as<int>();
        }
        if (config["NUM_PRG_PER_CELL"]) {
            num_prg_per_cell = config["NUM_PRG_PER_CELL"].as<int>();
        }
        if (config["NUM_SRS_UE_PER_SLOT"]) {
            num_srs_ue_per_slot = config["NUM_SRS_UE_PER_SLOT"].as<int>();
        }
        if (config["MAX_NUM_UE_SCHEDULED_PER_CELL_TTI"]) {
            max_num_ue_schd_per_cell_tti = config["MAX_NUM_UE_SCHEDULED_PER_CELL_TTI"].as<int>();
        }
        if (config["MAX_NUM_UE_FOR_GRP_PER_CELL"]) {
            max_num_ue_for_grp_per_cell = config["MAX_NUM_UE_FOR_GRP_PER_CELL"].as<int>();
        }
        if (config["SCS"]) {
            scs = config["SCS"].as<float>();
        }
        if (config["SRS_CHAN_EST_COEFF_VAR"]) {
            srs_chan_est_coeff_var = config["SRS_CHAN_EST_COEFF_VAR"].as<float>();
        }
        if (config["NUM_BLOCKS_PER_ROW_CHAN_OR_MAT"]) {
            num_blocks_per_row_chanOrtMat = config["NUM_BLOCKS_PER_ROW_CHAN_OR_MAT"].as<int>();
        }
        if (config["SLOT_INTERVAL_NS"]) {
            slot_interval_ns = config["SLOT_INTERVAL_NS"].as<uint64_t>();
        }

        // validate parameters
        if (num_cell > MAX_NUM_CELL) {
            throw std::runtime_error("num_cell > MAX_NUM_CELL");
        }
        if (num_srs_ue_per_slot > MAX_NUM_UE_SRS_INFO_PER_SLOT) {
            throw std::runtime_error("num_srs_ue_per_slot > MAX_NUM_UE_SRS_INFO_PER_SLOT");
        }
        if (num_subband > MAX_NUM_SUBBAND) {
            throw std::runtime_error("num_subband > MAX_NUM_SUBBAND");
        }
        if (num_prg_samp_per_subband > MAX_NUM_PRG_SAMP_PER_SUBBAND) {
            throw std::runtime_error("num_prg_samp_per_subband > MAX_NUM_PRG_SAMP_PER_SUBBAND");
        }
        if (num_prg_per_cell > MAX_NUM_PRG) {
            throw std::runtime_error("num_prg_per_cell > MAX_NUM_PRG");
        }
        if (num_srs_ue_per_cell > MAX_NUM_SRS_UE_PER_CELL) {
            throw std::runtime_error("num_srs_ue_per_cell > MAX_NUM_SRS_UE_PER_CELL");
        }
        if (num_ue_ant_port > MAX_NUM_UE_ANT_PORT) {
            throw std::runtime_error("num_ue_ant_port > MAX_NUM_UE_ANT_PORT");
        }
        if (num_bs_ant_port > MAX_NUM_BS_ANT_PORT) {
            throw std::runtime_error("num_bs_ant_port > MAX_NUM_BS_ANT_PORT");
        }
    }
};

inline void get_next_slot_timespec(struct timespec* ts, uint64_t interval_nsec)
{
    ts->tv_nsec += interval_nsec;
    while (ts->tv_nsec >= 1000000000L)
    {
        ts->tv_nsec -= 1000000000L;
        ts->tv_sec++;
    }
}

inline void advance_sfn_slot(uint16_t& sfn, uint16_t& slot)
{
    slot++;
    if (slot >= MU_TEST_SLOTS_PER_FRAME) {
        slot = 0;
        sfn++;
        if (sfn >= MU_TEST_MAX_SFN) {
            sfn = 0;
        }
    }
}

inline const char* get_cumac_msg_name(int msg_id)
{
    switch(msg_id)
    {
    case CUMAC_PARAM_REQUEST:
        return "PARAM.req";
    case CUMAC_PARAM_RESPONSE:
        return "PARAM.resp";

    case CUMAC_CONFIG_REQUEST:
        return "CONFIG.req";
    case CUMAC_CONFIG_RESPONSE:
        return "CONFIG.resp";

    case CUMAC_START_REQUEST:
        return "START.req";
    case CUMAC_START_RESPONSE:
        return "START.resp";

    case CUMAC_STOP_REQUEST:
        return "STOP.req";
    case CUMAC_STOP_RESPONSE:
        return "STOP.resp";

    case CUMAC_ERROR_INDICATION:
        return "ERR.ind";

    case CUMAC_TTI_ERROR_INDICATION:
        return "TTI_ERR.ind";
    case CUMAC_DL_TTI_REQUEST:
        return "DL_TTI.req";
    case CUMAC_UL_TTI_REQUEST:
        return "UL_TTI.req";

    case CUMAC_SCH_TTI_REQUEST:
        return "SCH_TTI.req";
    case CUMAC_SCH_TTI_RESPONSE:
        return "SCH_TTI.resp";

    case CUMAC_TTI_END:
        return "TTI_END.req";

    default:
        return "UNKNOWN_CUMAC_MSG";
    }
}

inline void print_ue_pairing_sol(const char* tag, uint16_t sfn, uint16_t slot, const uint8_t* out_buf, int num_cell, int cell_id_start = 0)
{
    printf("\n========== [%s] UE Pairing Solution  SFN=%u  Slot=%u  (%d cell%s) ==========\n",
           tag, sfn, slot, num_cell, num_cell > 1 ? "s" : "");

    for (int cellId = 0; cellId < num_cell; cellId++) {
        const cumac_muUeGrp_resp_info_t* resp =
            reinterpret_cast<const cumac_muUeGrp_resp_info_t*>(out_buf + cellId * sizeof(cumac_muUeGrp_resp_info_t));

        printf("  ---- Cell %d : %u scheduled UEG(s) ----\n", cell_id_start + cellId, resp->numSchdUeg);

        for (uint32_t uegId = 0; uegId < resp->numSchdUeg; uegId++) {
            const auto& ueg = resp->schdUegInfo[uegId];
            const char* ueg_type = (ueg.numUeInGrp > 1) ? "MU-MIMO" : "SU-MIMO";

            printf("    UEG %u [%s]  PRG=[%d, %d)  UEs=%u  flags=0x%02X\n",
                   uegId, ueg_type, ueg.allocPrgStart, ueg.allocPrgEnd,
                   ueg.numUeInGrp, ueg.flags);

            for (uint8_t ueId = 0; ueId < ueg.numUeInGrp; ueId++) {
                const auto& ue = ueg.ueInfo[ueId];
                printf("      UE %u : rnti=%5u  id=%3u  layerSel=0x%02X  order=%u  nSCID=%u  flags=0x%02X\n",
                       ueId, ue.rnti, ue.id, ue.layerSel, ue.ueOrderInGrp, ue.nSCID, ue.flags);
            }
        }
    }

    printf("========== [%s] End of UE Pairing Solution ==========\n\n", tag);
}

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

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <ctime>
#include <dirent.h>
#include <regex>
#include <string>
#include <utility>
#include <vector>
#include <stdexcept>
#include <yaml-cpp/yaml.h>
#include <H5Cpp.h>
#include "cumac_muUeGrp.h"
#include "cumac_msg.h"

#ifndef CHECK_CUDA_ERR
#define CHECK_CUDA_ERR(stmt)                                                                                                                                     \
    do                                                                                                                                                           \
    {                                                                                                                                                            \
        cudaError_t result1 = (stmt);                                                                                                                            \
        if (cudaSuccess != result1)                                                                                                                              \
        {                                                                                                                                                        \
            NVLOGW(MU_TEST_TAG, "[%s:%d] cuda failed with result1 %s", __FILE__, __LINE__, cudaGetErrorString(result1));                                            \
            cudaError_t result2 = cudaGetLastError();                                                                                                            \
            if (cudaSuccess != result2)                                                                                                                          \
            {                                                                                                                                                    \
                NVLOGW(MU_TEST_TAG, "[%s:%d] cuda failed with result2 %s result1 %s", __FILE__, __LINE__, cudaGetErrorString(result2), cudaGetErrorString(result1)); \
                cudaError_t result3 = cudaGetLastError(); /*check for stickiness*/                                                                               \
                if (cudaSuccess != result3)                                                                                                                      \
                {                                                                                                                                                \
                    NVLOGE(MU_TEST_TAG, AERIAL_CUDA_API_EVENT, "[%s:%d] cuda failed with result3 %s result2 %s result1 %s",                                          \
                               __FILE__,                                                                                                                         \
                               __LINE__,                                                                                                                         \
                               cudaGetErrorString(result3),                                                                                                      \
                               cudaGetErrorString(result2),                                                                                                      \
                               cudaGetErrorString(result1));                                                                                                     \
                }                                                                                                                                                \
            }                                                                                                                                                    \
        }                                                                                                                                                        \
    } while (0)
#endif

// Log TAG configured in nvlog
constexpr int MU_TEST_TAG = (NVLOG_TAG_BASE_NVIPC + 0);

constexpr uint16_t MU_TEST_SLOTS_PER_FRAME = 20;
constexpr uint16_t MU_TEST_MAX_SFN = 1024;


//! One cubb-side SRS-buffer dump file discovered on disk.
//! Files are produced by cuphydriver `SrsIpcManager::dump_h5` when run
//! with `export DUMP_SRS_SLOT_NUM=N` (N>0), with names of the form
//! `cubb_srs_buffers_<dump_idx>_SFN_<sfn>.<slot>.h5`.
struct CubbTvFile {
    int         index;  ///< Dump order index (0-based), from filename
    int         sfn;
    int         slot;
    std::string path;
};

//! Scan `dir` for `cubb_srs_buffers_<index>_SFN_<sfn>.<slot>.h5` files and
//! return them sorted ascending by dump index. Returns an empty vector if the
//! directory cannot be opened or contains no matching files.
inline std::vector<CubbTvFile> scan_cubb_tv_dir(const std::string& dir)
{
    std::vector<CubbTvFile> out;
    if (dir.empty()) {
        return out;
    }
    DIR* d = opendir(dir.c_str());
    if (d == nullptr) {
        return out;
    }
    const std::regex re(R"(^cubb_srs_buffers_(\d+)_SFN_(\d+)\.(\d+)\.h5$)");
    struct dirent* ent;
    while ((ent = readdir(d)) != nullptr) {
        std::string name(ent->d_name);
        std::smatch m;
        if (!std::regex_match(name, m, re)) {
            continue;
        }
        CubbTvFile f;
        f.index = std::stoi(m[1].str());
        f.sfn   = std::stoi(m[2].str());
        f.slot  = std::stoi(m[3].str());
        f.path  = dir + "/" + name;
        out.push_back(std::move(f));
    }
    closedir(d);
    std::sort(out.begin(), out.end(), [](const CubbTvFile& a, const CubbTvFile& b) {
        return a.index < b.index;
    });
    return out;
}

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
    bool enable_tv_test_mode{false}; // enable HDF5 TV save/load round-trip test mode
    bool tv_save_all_slots{false}; // true: save TVs for all slots; false: save TVs for last slot only
    int num_time_slots; // number of time slots

    std::vector<int> schedule_slot_ids;  // slot IDs that are scheduled (modulo schedule_slot_period)
    int schedule_slot_period{0};         // schedule pattern period in slots

    // CUBB TV input configurations
    bool        enable_cubb_tv_input{false};
    std::string cubb_tv_input_dir;
    int         cubb_cumac_srs_slot_lag{0};
    std::vector<CubbTvFile> cubb_tv_files;

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
        if (config["ENABLE_TV_TEST_MODE"]) {
            enable_tv_test_mode = config["ENABLE_TV_TEST_MODE"].as<bool>();
        }
        if (config["TV_SAVE_ALL_SLOTS"]) {
            tv_save_all_slots = config["TV_SAVE_ALL_SLOTS"].as<bool>();
        }
        if (config["SCHEDULE_SLOT_IDS"]) {
            schedule_slot_ids = config["SCHEDULE_SLOT_IDS"].as<std::vector<int>>();
        }
        if (config["SCHEDULE_SLOT_PERIOD"]) {
            schedule_slot_period = config["SCHEDULE_SLOT_PERIOD"].as<int>();
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
        if (config["CUMAC_MAIN_THREAD_CORE"]) {
            cumac_main_thread_core = config["CUMAC_MAIN_THREAD_CORE"].as<int>();
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

        if (config["ENABLE_CUBB_TV_INPUT"]) {
            enable_cubb_tv_input = config["ENABLE_CUBB_TV_INPUT"].as<bool>();
        }
        if (config["CUBB_TV_INPUT_DIR"]) {
            cubb_tv_input_dir = config["CUBB_TV_INPUT_DIR"].as<std::string>();
        }
        if (config["CUBB_CUMAC_SRS_SLOT_LAG"]) {
            cubb_cumac_srs_slot_lag = config["CUBB_CUMAC_SRS_SLOT_LAG"].as<int>();
        }

        if (enable_cubb_tv_input) {
            if (cubb_tv_input_dir.empty()) {
                throw std::runtime_error(
                    "ENABLE_CUBB_TV_INPUT=true but CUBB_TV_INPUT_DIR is empty");
            }
            cubb_tv_files = scan_cubb_tv_dir(cubb_tv_input_dir);
            if (cubb_tv_files.empty()) {
                throw std::runtime_error(
                    "CUBB_TV_INPUT_DIR='" + cubb_tv_input_dir +
                    "' contains no cubb_srs_buffers_<index>_SFN_*.<slot>.h5 files");
            }
            num_time_slots = static_cast<int>(cubb_tv_files.size());
            schedule_slot_ids.clear();
            schedule_slot_ids.reserve(cubb_tv_files.size());
            for (const auto& f : cubb_tv_files) {
                int abs_slot = f.sfn * static_cast<int>(MU_TEST_SLOTS_PER_FRAME)
                             + f.slot
                             + cubb_cumac_srs_slot_lag;
                // Normalize into [0, schedule_slot_period) so the slot IDs stay
                // within the period configured in YAML (e.g. 40). This avoids
                // the out-of-range validation below and writes the correct
                // schedule_slot_period attribute to every TV H5 file.
                if (schedule_slot_period > 0) {
                    abs_slot = ((abs_slot % schedule_slot_period) + schedule_slot_period) % schedule_slot_period;
                }
                schedule_slot_ids.push_back(abs_slot);
            }
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
        // Validate schedule_slot_ids against schedule_slot_period (when set).
        if (!schedule_slot_ids.empty() && schedule_slot_period > 0) {
            for (int s : schedule_slot_ids) {
                if (s < 0 || s >= schedule_slot_period) {
                    throw std::runtime_error(
                        "SCHEDULE_SLOT_IDS contains entry " + std::to_string(s) +
                        " out of range [0, SCHEDULE_SLOT_PERIOD=" + std::to_string(schedule_slot_period) + ")");
                }
            }
        }
    }

    // Returns true if `slot` is a scheduled slot (TV should be generated for it).
    bool is_scheduled_slot(uint16_t slot) const
    {
        if (schedule_slot_ids.empty()) {
            return true;
        }
        const int slot_in_period = (schedule_slot_period > 0)
            ? (static_cast<int>(slot) % schedule_slot_period)
            : static_cast<int>(slot);
        return std::find(schedule_slot_ids.begin(),
                         schedule_slot_ids.end(),
                         slot_in_period) != schedule_slot_ids.end();
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

#ifdef __CUDACC__
// HDF5 TV creation: saves per-cell HDF5 test vector files for standalone muMimoUserPairing testing
inline void create_h5_tv(const std::string&     tvNameBase,
                         const uint16_t         sfn,
                         const uint16_t         slot,
                         const sys_param_t&     sys_param,
                         cumac::muUePairTask*   muMimoUserPairingTask,
                         uint8_t*               srs_chan_est_buf_base_addr,
                         uint32_t               srs_chan_est_buf_size,
                         float*                 srs_snr_buf_base_addr,
                         uint32_t               srs_snr_buf_size,
                         float*                 chan_orth_mat_buf_base_addr,
                         uint32_t               chan_orth_mat_buf_size,
                         __half2*               cubb_srs_gpu_buf_base_addr,
                         size_t                 cubb_srs_gpu_buf_size,
                         bool                   is_first_slot = false)
{
    const int  num_cell       = sys_param.num_cell;
    const bool is_mem_sharing = muMimoUserPairingTask->is_mem_sharing;

    const uint32_t task_in_per_cell = is_mem_sharing
        ? static_cast<uint32_t>(sizeof(cumac_muUeGrp_req_info_t)
            + sizeof(cumac_muUeGrp_req_srs_info_msh_t) * MAX_NUM_UE_SRS_INFO_PER_SLOT
            + sizeof(cumac_muUeGrp_req_ue_info_t)      * MAX_NUM_SRS_UE_PER_CELL)
        : static_cast<uint32_t>(sizeof(cumac_muUeGrp_req_info_t)
            + sizeof(cumac_muUeGrp_req_srs_info_t)     * MAX_NUM_UE_SRS_INFO_PER_SLOT
            + sizeof(cumac_muUeGrp_req_ue_info_t)      * MAX_NUM_SRS_UE_PER_CELL);

    const uint32_t snr_total_elems  = srs_snr_buf_size / sizeof(float);
    const uint32_t orth_total_elems = chan_orth_mat_buf_size / sizeof(float);

    std::vector<uint8_t> h_task_in(task_in_per_cell * num_cell);
    CHECK_CUDA_ERR(cudaMemcpy(h_task_in.data(), muMimoUserPairingTask->task_in_buf,
               h_task_in.size(), cudaMemcpyDeviceToHost));

    // Big GPU buffers are saved only in cell 0's TV to reduce total TV size
    std::vector<uint8_t> h_chan_est(srs_chan_est_buf_size);
    std::vector<float>   h_snr(snr_total_elems);
    std::vector<float>   h_orth(orth_total_elems);
    CHECK_CUDA_ERR(cudaMemcpy(h_chan_est.data(), srs_chan_est_buf_base_addr,
               srs_chan_est_buf_size, cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERR(cudaMemcpy(h_snr.data(), srs_snr_buf_base_addr,
               srs_snr_buf_size, cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERR(cudaMemcpy(h_orth.data(), chan_orth_mat_buf_base_addr,
               chan_orth_mat_buf_size, cudaMemcpyDeviceToHost));

    std::vector<uint8_t> h_cubb;
    if (is_mem_sharing && cubb_srs_gpu_buf_size > 0) {
        h_cubb.resize(cubb_srs_gpu_buf_size);
        CHECK_CUDA_ERR(cudaMemcpy(h_cubb.data(), cubb_srs_gpu_buf_base_addr,
                   cubb_srs_gpu_buf_size, cudaMemcpyDeviceToHost));
    }

    H5::DataSpace scalarSpace(H5S_SCALAR);

    for (int c = 0; c < num_cell; c++) {
        std::string fileName = tvNameBase + "_cell" + std::to_string(c) + ".h5";
        H5::H5File file(fileName, H5F_ACC_TRUNC);

        auto attrU8  = [&](const char* n, uint8_t  v) { file.createAttribute(n, H5::PredType::NATIVE_UINT8,  scalarSpace).write(H5::PredType::NATIVE_UINT8,  &v); };
        auto attrU16 = [&](const char* n, uint16_t v) { file.createAttribute(n, H5::PredType::NATIVE_UINT16, scalarSpace).write(H5::PredType::NATIVE_UINT16, &v); };
        auto attrI32 = [&](const char* n, int32_t  v) { file.createAttribute(n, H5::PredType::NATIVE_INT32,  scalarSpace).write(H5::PredType::NATIVE_INT32,  &v); };
        auto attrU32 = [&](const char* n, uint32_t v) { file.createAttribute(n, H5::PredType::NATIVE_UINT32, scalarSpace).write(H5::PredType::NATIVE_UINT32, &v); };

        attrU16("sfn",                          sfn);
        attrU16("slot",                         slot);
        attrU16("cell_idx",                     static_cast<uint16_t>(c));
        attrI32("num_cell",                     sys_param.num_cell);
        attrI32("num_bs_ant_port",              sys_param.num_bs_ant_port);
        attrI32("num_ue_ant_port",              sys_param.num_ue_ant_port);
        attrI32("num_subband",                  sys_param.num_subband);
        attrI32("num_prg_samp_per_subband",     sys_param.num_prg_samp_per_subband);
        attrI32("num_prg_per_cell",             sys_param.num_prg_per_cell);
        attrI32("num_srs_ue_per_cell",          sys_param.num_srs_ue_per_cell);
        attrI32("num_srs_ue_per_slot",          sys_param.num_srs_ue_per_slot);
        attrI32("max_num_ue_schd_per_cell_tti", sys_param.max_num_ue_schd_per_cell_tti);
        attrI32("max_num_ue_for_grp_per_cell",  sys_param.max_num_ue_for_grp_per_cell);
        attrI32("num_blocks_per_row_chanOrtMat", sys_param.num_blocks_per_row_chanOrtMat);
        attrU8 ("kernel_launch_flags",          sys_param.kernel_launch_flags);
        attrU8 ("is_mem_sharing",               is_mem_sharing ? 1 : 0);
        attrU16("num_srs_ue_per_slot_cell",     muMimoUserPairingTask->num_srs_ue_per_slot_cell);
        attrU32("task_in_buf_len_per_cell",     task_in_per_cell);
        attrU8 ("first_slot",                   is_first_slot ? 1 : 0);
        attrI32("schedule_slot_period",         sys_param.schedule_slot_period);

        const cumac_muUeGrp_req_info_t* req_info =
            reinterpret_cast<const cumac_muUeGrp_req_info_t*>(
                h_task_in.data() + c * task_in_per_cell);
        attrU16("req_numUeInfo",            req_info->numUeInfo);
        attrU16("req_numSrsInfo",           req_info->numSrsInfo);
        attrU16("req_numSubband",           req_info->numSubband);
        attrU16("req_numPrgSampPerSubband", req_info->numPrgSampPerSubband);
        attrU16("req_numUeForGrpPerCell",   req_info->numUeForGrpPerCell);
        attrU16("req_nPrbGrp",             req_info->nPrbGrp);
        attrU8 ("req_nBsAnt",              req_info->nBsAnt);
        attrU8 ("req_nMaxUeSchdPerCellTTI", req_info->nMaxUeSchdPerCellTTI);
        attrU8 ("req_nMaxUePerGrp",         req_info->nMaxUePerGrp);
        attrU8 ("req_nMaxLayerPerGrp",      req_info->nMaxLayerPerGrp);
        attrU8 ("req_nMaxUegPerCell",       req_info->nMaxUegPerCell);
        attrU8 ("req_allocType",            req_info->allocType);
        attrU32("req_betaCoeff",            req_info->betaCoeff);
        attrU32("req_muCoeff",              req_info->muCoeff);
        attrU32("req_chanCorrThr",          req_info->chanCorrThr);
        attrU32("req_srsSnrThr",            req_info->srsSnrThr);

        {
            hsize_t dim = task_in_per_cell;
            H5::DataSpace ds(1, &dim);
            file.createDataSet("task_in_buf", H5::PredType::NATIVE_UINT8, ds)
                .write(h_task_in.data() + c * task_in_per_cell, H5::PredType::NATIVE_UINT8);
        }

        // Save big GPU buffers only in cell 0's TV (they are common across all cells)
        if (c == 0) {
            {
                hsize_t dim = srs_chan_est_buf_size;
                H5::DataSpace ds(1, &dim);
                file.createDataSet("srs_chan_est_buf", H5::PredType::NATIVE_UINT8, ds)
                    .write(h_chan_est.data(), H5::PredType::NATIVE_UINT8);
            }

            {
                hsize_t dim = snr_total_elems;
                H5::DataSpace ds(1, &dim);
                file.createDataSet("srs_snr_buf", H5::PredType::NATIVE_FLOAT, ds)
                    .write(h_snr.data(), H5::PredType::NATIVE_FLOAT);
            }

            {
                hsize_t dim = orth_total_elems;
                H5::DataSpace ds(1, &dim);
                file.createDataSet("chan_orth_mat_buf", H5::PredType::NATIVE_FLOAT, ds)
                    .write(h_orth.data(), H5::PredType::NATIVE_FLOAT);
            }

            if (is_mem_sharing && cubb_srs_gpu_buf_size > 0) {
                hsize_t dim = cubb_srs_gpu_buf_size;
                H5::DataSpace ds(1, &dim);
                file.createDataSet("cubb_srs_gpu_buf", H5::PredType::NATIVE_UINT8, ds)
                    .write(h_cubb.data(), H5::PredType::NATIVE_UINT8);
            }
        }

        printf("create_h5_tv: wrote %s  (SFN=%u, slot=%u, cell=%d)\n",
               fileName.c_str(), sfn, slot, c);
    }
}

inline void create_h5_solution_tv(const std::string& tvNameBase,
                                  uint16_t           sfn,
                                  uint16_t           slot,
                                  int                num_cell,
                                  const uint8_t*     solution_buf)
{
    std::string fileName = tvNameBase + "_solution.h5";
    H5::H5File file(fileName, H5F_ACC_TRUNC);

    H5::DataSpace scalarSpace(H5S_SCALAR);
    auto attrU16 = [&](const char* n, uint16_t v) { file.createAttribute(n, H5::PredType::NATIVE_UINT16, scalarSpace).write(H5::PredType::NATIVE_UINT16, &v); };
    auto attrI32 = [&](const char* n, int32_t  v) { file.createAttribute(n, H5::PredType::NATIVE_INT32,  scalarSpace).write(H5::PredType::NATIVE_INT32,  &v); };

    attrU16("sfn",      sfn);
    attrU16("slot",     slot);
    attrI32("num_cell", num_cell);

    uint32_t sol_size = static_cast<uint32_t>(sizeof(cumac_muUeGrp_resp_info_t)) * num_cell;
    hsize_t dim = sol_size;
    H5::DataSpace ds(1, &dim);
    file.createDataSet("solution", H5::PredType::NATIVE_UINT8, ds)
        .write(solution_buf, H5::PredType::NATIVE_UINT8);

    printf("create_h5_solution_tv: wrote %s  (SFN=%u, slot=%u, %d cell(s), %u bytes)\n",
           fileName.c_str(), sfn, slot, num_cell, sol_size);
}

// HDF5 TV loading: reads per-cell HDF5 test vector files and populates GPU buffers.
// Big GPU buffers (srs_chan_est, srs_snr, chan_orth_mat, cubb_srs) are stored only
// in cell 0's TV and loaded as full multi-cell buffers from there.
inline void load_h5_tv(const std::string&     tvNameBase,
                       int                    num_cell,
                       uint16_t&              sfn,
                       uint16_t&              slot,
                       cumac::muUePairTask*   muMimoUserPairingTask,
                       uint8_t*               srs_chan_est_buf_base_addr,
                       uint32_t               srs_chan_est_buf_size,
                       float*                 srs_snr_buf_base_addr,
                       uint32_t               srs_snr_buf_size,
                       float*                 chan_orth_mat_buf_base_addr,
                       uint32_t               chan_orth_mat_buf_size,
                       __half2*               cubb_srs_gpu_buf_base_addr,
                       size_t                 cubb_srs_gpu_buf_size)
{
    for (int c = 0; c < num_cell; c++) {
        std::string fileName = tvNameBase + "_cell" + std::to_string(c) + ".h5";
        H5::H5File file(fileName, H5F_ACC_RDONLY);

        if (c == 0) {
            file.openAttribute("sfn").read(H5::PredType::NATIVE_UINT16, &sfn);
            file.openAttribute("slot").read(H5::PredType::NATIVE_UINT16, &slot);

            uint16_t srs_ue_per_slot_cell;
            file.openAttribute("num_srs_ue_per_slot_cell")
                .read(H5::PredType::NATIVE_UINT16, &srs_ue_per_slot_cell);
            muMimoUserPairingTask->num_srs_ue_per_slot_cell = srs_ue_per_slot_cell;

            uint8_t flags;
            file.openAttribute("kernel_launch_flags")
                .read(H5::PredType::NATIVE_UINT8, &flags);
            muMimoUserPairingTask->kernel_launch_flags = flags;

            int32_t blocks_per_row;
            file.openAttribute("num_blocks_per_row_chanOrtMat")
                .read(H5::PredType::NATIVE_INT32, &blocks_per_row);
            muMimoUserPairingTask->num_blocks_per_row_chanOrtMat =
                static_cast<uint16_t>(blocks_per_row);

            uint8_t mem_sharing;
            file.openAttribute("is_mem_sharing")
                .read(H5::PredType::NATIVE_UINT8, &mem_sharing);
            muMimoUserPairingTask->is_mem_sharing = (mem_sharing != 0);

            // Load full multi-cell GPU buffers from cell 0's TV
            {
                H5::DataSet ds = file.openDataSet("srs_chan_est_buf");
                hsize_t dim;
                ds.getSpace().getSimpleExtentDims(&dim);
                std::vector<uint8_t> buf(dim);
                ds.read(buf.data(), H5::PredType::NATIVE_UINT8);
                CHECK_CUDA_ERR(cudaMemcpy(srs_chan_est_buf_base_addr,
                           buf.data(), dim, cudaMemcpyHostToDevice));
            }

            {
                H5::DataSet ds = file.openDataSet("srs_snr_buf");
                hsize_t dim;
                ds.getSpace().getSimpleExtentDims(&dim);
                std::vector<float> buf(dim);
                ds.read(buf.data(), H5::PredType::NATIVE_FLOAT);
                CHECK_CUDA_ERR(cudaMemcpy(srs_snr_buf_base_addr,
                           buf.data(), dim * sizeof(float), cudaMemcpyHostToDevice));
            }

            {
                H5::DataSet ds = file.openDataSet("chan_orth_mat_buf");
                hsize_t dim;
                ds.getSpace().getSimpleExtentDims(&dim);
                std::vector<float> buf(dim);
                ds.read(buf.data(), H5::PredType::NATIVE_FLOAT);
                CHECK_CUDA_ERR(cudaMemcpy(chan_orth_mat_buf_base_addr,
                           buf.data(), dim * sizeof(float), cudaMemcpyHostToDevice));
            }

            if (cubb_srs_gpu_buf_size > 0 &&
                H5Lexists(file.getId(), "cubb_srs_gpu_buf", H5P_DEFAULT) > 0) {
                H5::DataSet ds = file.openDataSet("cubb_srs_gpu_buf");
                hsize_t dim;
                ds.getSpace().getSimpleExtentDims(&dim);
                std::vector<uint8_t> buf(dim);
                ds.read(buf.data(), H5::PredType::NATIVE_UINT8);
                CHECK_CUDA_ERR(cudaMemcpy(cubb_srs_gpu_buf_base_addr,
                           buf.data(), dim, cudaMemcpyHostToDevice));
            }
        }

        {
            H5::DataSet ds = file.openDataSet("task_in_buf");
            hsize_t dim;
            ds.getSpace().getSimpleExtentDims(&dim);
            std::vector<uint8_t> buf(dim);
            ds.read(buf.data(), H5::PredType::NATIVE_UINT8);
            CHECK_CUDA_ERR(cudaMemcpy(muMimoUserPairingTask->task_in_buf + c * dim,
                       buf.data(), dim, cudaMemcpyHostToDevice));
        }

        printf("load_h5_tv: read %s  (cell=%d)\n", fileName.c_str(), c);
    }

    printf("load_h5_tv: loaded %d cell(s), SFN=%u, slot=%u\n",
           num_cell, sfn, slot);
}

// HDF5 TV loading for L2 emulator.
//
// Reads per-cell HDF5 test vector files and populates caller-provided
// per-cell host buffers (req_info_bufs[0..num_cell-1]) with the
// reconstructed cumac_muUeGrp_req_info_t structs (header + srsInfo/ueInfo
// payload, internal pointers fixed up).  The L2 emulator can then send
// these buffers to cuMAC-CP via NVIPC as request messages.
// req_info_data_len[c] receives the actual byte length of each cell's
// buffer so the NVIPC data_len can be set correctly.
//
inline void load_h5_tv_l2emu(
    const std::string&     tvNameBase,
    int                    num_cell,
    uint16_t&              sfn,
    uint16_t&              slot,
    uint8_t*               req_info_bufs[],       // [num_cell] pre-allocated host buffers
    uint32_t               req_info_data_len[])    // [num_cell] actual data length per cell (out)
{
    bool is_mem_sharing = false;

    for (int c = 0; c < num_cell; c++) {
        std::string fileName = tvNameBase + "_cell" + std::to_string(c) + ".h5";
        H5::H5File file(fileName, H5F_ACC_RDONLY);

        if (c == 0) {
            file.openAttribute("sfn").read(H5::PredType::NATIVE_UINT16, &sfn);
            file.openAttribute("slot").read(H5::PredType::NATIVE_UINT16, &slot);

            uint8_t mem_sharing;
            file.openAttribute("is_mem_sharing")
                .read(H5::PredType::NATIVE_UINT8, &mem_sharing);
            is_mem_sharing = (mem_sharing != 0);
        }

        H5::DataSet ds = file.openDataSet("task_in_buf");
        hsize_t dim;
        ds.getSpace().getSimpleExtentDims(&dim);
        ds.read(req_info_bufs[c], H5::PredType::NATIVE_UINT8);
        req_info_data_len[c] = static_cast<uint32_t>(dim);

        cumac_muUeGrp_req_info_t* req_info =
            reinterpret_cast<cumac_muUeGrp_req_info_t*>(req_info_bufs[c]);

        if (is_mem_sharing) {
            req_info->srsInfoMsh =
                reinterpret_cast<cumac_muUeGrp_req_srs_info_msh_t*>(req_info->payload);
            req_info->ueInfo =
                reinterpret_cast<cumac_muUeGrp_req_ue_info_t*>(
                    req_info->payload
                    + sizeof(cumac_muUeGrp_req_srs_info_msh_t) * MAX_NUM_UE_SRS_INFO_PER_SLOT);
        } else {
            req_info->srsInfo =
                reinterpret_cast<cumac_muUeGrp_req_srs_info_t*>(req_info->payload);
            req_info->ueInfo =
                reinterpret_cast<cumac_muUeGrp_req_ue_info_t*>(
                    req_info->payload
                    + sizeof(cumac_muUeGrp_req_srs_info_t) * MAX_NUM_UE_SRS_INFO_PER_SLOT);
        }

        printf("load_h5_tv_l2emu: cell %d req_info: numUeInfo=%u, numSrsInfo=%u, "
               "nPrbGrp=%u, nBsAnt=%u, numSubband=%u, nMaxUeSchdPerCellTTI=%u, "
               "data_len=%u\n",
               c, req_info->numUeInfo, req_info->numSrsInfo,
               req_info->nPrbGrp, req_info->nBsAnt, req_info->numSubband,
               req_info->nMaxUeSchdPerCellTTI, req_info_data_len[c]);
    }

    printf("load_h5_tv_l2emu: loaded %d cell(s), SFN=%u, slot=%u\n",
           num_cell, sfn, slot);
}

// HDF5 TV loading for cuMAC-CP.
//
// Reads per-cell HDF5 test vector files, populates the pre-allocated GPU
// data buffers (srs_chan_est, srs_snr, chan_orth_mat, cubb_srs) that the
// GPU UE pairing module reads during execution, and sets muUePairTask
// parameters (kernel_launch_flags, num_srs_ue_per_slot_cell, etc.).
// Big GPU buffers are stored only in cell 0's TV and loaded as full
// multi-cell buffers from there.
// The task_in_buf GPU copy is NOT done here -- cuMAC-CP performs that
// copy after it receives the request messages from L2 via NVIPC.
//
inline void load_h5_tv_cumacCp(
    const std::string&     tvNameBase,
    int                    num_cell,
    uint16_t&              sfn,
    uint16_t&              slot,
    cumac::muUePairTask*   muMimoUserPairingTask,
    uint8_t*               srs_chan_est_buf_base_addr,
    uint32_t               srs_chan_est_buf_size,
    float*                 srs_snr_buf_base_addr,
    uint32_t               srs_snr_buf_size,
    float*                 chan_orth_mat_buf_base_addr,
    uint32_t               chan_orth_mat_buf_size,
    __half2*               cubb_srs_gpu_buf_base_addr,
    size_t                 cubb_srs_gpu_buf_size)
{
    for (int c = 0; c < num_cell; c++) {
        std::string fileName = tvNameBase + "_cell" + std::to_string(c) + ".h5";
        H5::H5File file(fileName, H5F_ACC_RDONLY);

        if (c == 0) {
            file.openAttribute("sfn").read(H5::PredType::NATIVE_UINT16, &sfn);
            file.openAttribute("slot").read(H5::PredType::NATIVE_UINT16, &slot);

            uint16_t srs_ue_per_slot_cell;
            file.openAttribute("num_srs_ue_per_slot_cell")
                .read(H5::PredType::NATIVE_UINT16, &srs_ue_per_slot_cell);
            muMimoUserPairingTask->num_srs_ue_per_slot_cell = srs_ue_per_slot_cell;

            uint8_t flags;
            file.openAttribute("kernel_launch_flags")
                .read(H5::PredType::NATIVE_UINT8, &flags);
            muMimoUserPairingTask->kernel_launch_flags = flags;

            int32_t blocks_per_row;
            file.openAttribute("num_blocks_per_row_chanOrtMat")
                .read(H5::PredType::NATIVE_INT32, &blocks_per_row);
            muMimoUserPairingTask->num_blocks_per_row_chanOrtMat =
                static_cast<uint16_t>(blocks_per_row);

            uint8_t mem_sharing;
            file.openAttribute("is_mem_sharing")
                .read(H5::PredType::NATIVE_UINT8, &mem_sharing);
            muMimoUserPairingTask->is_mem_sharing = (mem_sharing != 0);

            // Load full multi-cell GPU buffers from cell 0's TV
            {
                H5::DataSet ds = file.openDataSet("srs_chan_est_buf");
                hsize_t dim;
                ds.getSpace().getSimpleExtentDims(&dim);
                std::vector<uint8_t> buf(dim);
                ds.read(buf.data(), H5::PredType::NATIVE_UINT8);
                CHECK_CUDA_ERR(cudaMemcpy(srs_chan_est_buf_base_addr,
                           buf.data(), dim, cudaMemcpyHostToDevice));
            }

            {
                H5::DataSet ds = file.openDataSet("srs_snr_buf");
                hsize_t dim;
                ds.getSpace().getSimpleExtentDims(&dim);
                std::vector<float> buf(dim);
                ds.read(buf.data(), H5::PredType::NATIVE_FLOAT);
                CHECK_CUDA_ERR(cudaMemcpy(srs_snr_buf_base_addr,
                           buf.data(), dim * sizeof(float), cudaMemcpyHostToDevice));
            }

            {
                H5::DataSet ds = file.openDataSet("chan_orth_mat_buf");
                hsize_t dim;
                ds.getSpace().getSimpleExtentDims(&dim);
                std::vector<float> buf(dim);
                ds.read(buf.data(), H5::PredType::NATIVE_FLOAT);
                CHECK_CUDA_ERR(cudaMemcpy(chan_orth_mat_buf_base_addr,
                           buf.data(), dim * sizeof(float), cudaMemcpyHostToDevice));
            }

            if (cubb_srs_gpu_buf_size > 0 &&
                H5Lexists(file.getId(), "cubb_srs_gpu_buf", H5P_DEFAULT) > 0) {
                H5::DataSet ds = file.openDataSet("cubb_srs_gpu_buf");
                hsize_t dim;
                ds.getSpace().getSimpleExtentDims(&dim);
                std::vector<uint8_t> buf(dim);
                ds.read(buf.data(), H5::PredType::NATIVE_UINT8);
                CHECK_CUDA_ERR(cudaMemcpy(cubb_srs_gpu_buf_base_addr,
                           buf.data(), dim, cudaMemcpyHostToDevice));
            }
        }

        printf("load_h5_tv_cumacCp: read %s  (cell=%d)\n", fileName.c_str(), c);
    }

    printf("load_h5_tv_cumacCp: loaded %d cell(s), SFN=%u, slot=%u\n",
           num_cell, sfn, slot);
}

inline void load_h5_solution_tv(const std::string& tvNameBase,
                                int                num_cell,
                                uint16_t&          sfn,
                                uint16_t&          slot,
                                uint8_t*           solution_buf)
{
    std::string fileName = tvNameBase + "_solution.h5";
    H5::H5File file(fileName, H5F_ACC_RDONLY);

    file.openAttribute("sfn").read(H5::PredType::NATIVE_UINT16, &sfn);
    file.openAttribute("slot").read(H5::PredType::NATIVE_UINT16, &slot);

    H5::DataSet ds = file.openDataSet("solution");
    hsize_t dim;
    ds.getSpace().getSimpleExtentDims(&dim);
    ds.read(solution_buf, H5::PredType::NATIVE_UINT8);

    printf("load_h5_solution_tv: read %s  (SFN=%u, slot=%u, %d cell(s), %llu bytes)\n",
           fileName.c_str(), sfn, slot, num_cell, (unsigned long long)dim);
}
#endif // __CUDACC__

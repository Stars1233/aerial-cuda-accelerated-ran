/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "cumac.h"
#include "muMimoUserPairing/muMimoUserPairing.cuh"
#include "simple_srs_memory_bank.hpp"
#include "l2_muUeGrp_test.h"
#include "l1_muUeGrp_test.h"
#include "nv_utils.h"
#include "nv_ipc.h"
#include "nv_ipc_utils.h"
#include "nv_lockfree.hpp"
#include <yaml-cpp/yaml.h>
#include <vector>
#include <thread>
#include <chrono>

constexpr const char* YAML_CUMAC_L2_NVIPC_CONFIG_PATH = "./cuMAC/examples/muMimoUeGrpL2Integration/yamlConfigFiles/cumac_l2_nvipc.yaml"; // path to the cuMAC/L2 NVIPC configuration YAML file
constexpr const char* YAML_CUMAC_L1_NVIPC_CONFIG_PATH = "./cuMAC/examples/muMimoUeGrpL2Integration/yamlConfigFiles/cumac_l1_nvipc.yaml"; // path to the cuMAC/L1 NVIPC configuration YAML file

#define CUMAC_L1_SECONDARY_PROCESS 0

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

struct test_task_t {
    uint32_t        sfn;
    uint32_t        slot;
    uint8_t*        task_in_buf; // storing data for all cells in contiguous GPU memory
    uint32_t        task_in_buf_len_per_cell; // length of the allocated memory for each cell
    uint16_t        num_cell; // number of cells for this task
    uint16_t        num_srs_ue; // number of SRS UEs for this task
    bool            is_mem_sharing;
    nv_ipc_msg_t    recv_msg[MAX_NUM_CELL];
    cudaStream_t    strm;

    void alloc_mem(bool is_mem_sharing_in, int num_cell_in) {
        is_mem_sharing = is_mem_sharing_in;
        num_cell = num_cell_in;
        uint32_t allocMemLength;
        if (is_mem_sharing) {
            allocMemLength = sizeof(cumac_muUeGrp_req_info_t) + sizeof(cumac_muUeGrp_req_srs_info_msh_t)*MAX_NUM_UE_SRS_INFO_PER_SLOT + sizeof(cumac_muUeGrp_req_ue_info_t)*MAX_NUM_SRS_UE_PER_CELL;
        } else {
            allocMemLength = sizeof(cumac_muUeGrp_req_info_t) + sizeof(cumac_muUeGrp_req_srs_info_t)*MAX_NUM_UE_SRS_INFO_PER_SLOT + sizeof(cumac_muUeGrp_req_ue_info_t)*MAX_NUM_SRS_UE_PER_CELL;
        }
        CHECK_CUDA_ERR(cudaMalloc(&task_in_buf, allocMemLength*num_cell));
        task_in_buf_len_per_cell = allocMemLength;
    }
};

inline void update_req_srs_info_msh(CVSrsChestBuff_contMemAlloc* arr_cv_srs_chest_buff_base_addr, l1_cumac_message_t* arr_l1_cumac_msg, nv_ipc_msg_t* recv_msg, uint16_t num_srs_ue) {
    for (int srs_ue_idx = 0; srs_ue_idx < num_srs_ue; srs_ue_idx++) {
        uint16_t cIdx = arr_l1_cumac_msg[srs_ue_idx].cell_idx;
        uint16_t srs_info_idx = arr_l1_cumac_msg[srs_ue_idx].srs_info_idx;
        uint32_t real_buff_idx = arr_l1_cumac_msg[srs_ue_idx].real_buff_idx;
        
        cumac_muUeGrp_req_info_t* req_ptr = (cumac_muUeGrp_req_info_t*) recv_msg[cIdx].data_buf;
        req_ptr->srsInfoMsh[srs_info_idx].realBuffIdx = real_buff_idx;

        CVSrsChestBuff_contMemAlloc* ue_buffer = arr_cv_srs_chest_buff_base_addr + real_buff_idx;
        uint16_t srsStartPrg;
        uint16_t srsStartValidPrg;
        uint16_t srsNValidPrg;
        uint8_t  srsPrgSize;
        ue_buffer->getSrsPrgInfo(&srsPrgSize, &srsStartPrg, &srsStartValidPrg, &srsNValidPrg);
        req_ptr->srsInfoMsh[srs_info_idx].srsStartPrg = srsStartPrg;
        req_ptr->srsInfoMsh[srs_info_idx].srsStartValidPrg = srsStartValidPrg;
        req_ptr->srsInfoMsh[srs_info_idx].srsNValidPrg = srsNValidPrg;
        req_ptr->srsInfoMsh[srs_info_idx].flags = 0x01; // valid
    }
}

inline bool compare_gpu_cpu_results(uint8_t* gpu_out_buf, uint8_t* cpu_out_buf, int num_cell) 
{
    bool match = true;

    for (uint16_t cellId = 0; cellId < num_cell; cellId++) {
        cumac_muUeGrp_resp_info_t* gpu_out_ptr = (cumac_muUeGrp_resp_info_t*) (gpu_out_buf + cellId*sizeof(cumac_muUeGrp_resp_info_t));
        cumac_muUeGrp_resp_info_t* cpu_out_ptr = (cumac_muUeGrp_resp_info_t*) (cpu_out_buf + cellId*sizeof(cumac_muUeGrp_resp_info_t));

        if (gpu_out_ptr->numSchdUeg != cpu_out_ptr->numSchdUeg) {
            NVLOGE(MU_TEST_TAG, AERIAL_NVIPC_API_EVENT,
                "GPU/CPU mismatch: cell %u numSchdUeg GPU=%u CPU=%u",
                cellId, gpu_out_ptr->numSchdUeg, cpu_out_ptr->numSchdUeg);
            match = false;
            continue;
        }

        for (uint16_t uegId = 0; uegId < gpu_out_ptr->numSchdUeg; uegId++) {
            const auto& gpu_ueg = gpu_out_ptr->schdUegInfo[uegId];
            const auto& cpu_ueg = cpu_out_ptr->schdUegInfo[uegId];

            if (gpu_ueg.numUeInGrp != cpu_ueg.numUeInGrp) {
                NVLOGE(MU_TEST_TAG, AERIAL_NVIPC_API_EVENT,
                    "GPU/CPU mismatch: cell %u ueg %u numUeInGrp GPU=%u CPU=%u",
                    cellId, uegId, gpu_ueg.numUeInGrp, cpu_ueg.numUeInGrp);
                match = false;
            }

            if (gpu_ueg.allocPrgStart != cpu_ueg.allocPrgStart) {
                NVLOGE(MU_TEST_TAG, AERIAL_NVIPC_API_EVENT,
                    "GPU/CPU mismatch: cell %u ueg %u allocPrgStart GPU=%d CPU=%d",
                    cellId, uegId, gpu_ueg.allocPrgStart, cpu_ueg.allocPrgStart);
                match = false;
            }

            if (gpu_ueg.allocPrgEnd != cpu_ueg.allocPrgEnd) {
                NVLOGE(MU_TEST_TAG, AERIAL_NVIPC_API_EVENT,
                    "GPU/CPU mismatch: cell %u ueg %u allocPrgEnd GPU=%d CPU=%d",
                    cellId, uegId, gpu_ueg.allocPrgEnd, cpu_ueg.allocPrgEnd);
                match = false;
            }

            if (gpu_ueg.flags != cpu_ueg.flags) {
                NVLOGE(MU_TEST_TAG, AERIAL_NVIPC_API_EVENT,
                    "GPU/CPU mismatch: cell %u ueg %u flags GPU=0x%02x CPU=0x%02x",
                    cellId, uegId, gpu_ueg.flags, cpu_ueg.flags);
                match = false;
            }

            uint8_t ueCount = (gpu_ueg.numUeInGrp < cpu_ueg.numUeInGrp)
                            ? gpu_ueg.numUeInGrp : cpu_ueg.numUeInGrp;
            for (uint16_t ueId = 0; ueId < ueCount; ueId++) {
                const auto& gpu_ue = gpu_ueg.ueInfo[ueId];
                const auto& cpu_ue = cpu_ueg.ueInfo[ueId];

                if (gpu_ue.rnti != cpu_ue.rnti) {
                    NVLOGE(MU_TEST_TAG, AERIAL_NVIPC_API_EVENT,
                        "GPU/CPU mismatch: cell %u ueg %u ue %u rnti GPU=%u CPU=%u",
                        cellId, uegId, ueId, gpu_ue.rnti, cpu_ue.rnti);
                    match = false;
                }

                if (gpu_ue.id != cpu_ue.id) {
                    NVLOGE(MU_TEST_TAG, AERIAL_NVIPC_API_EVENT,
                        "GPU/CPU mismatch: cell %u ueg %u ue %u id GPU=%u CPU=%u",
                        cellId, uegId, ueId, gpu_ue.id, cpu_ue.id);
                    match = false;
                }

                if (gpu_ue.layerSel != cpu_ue.layerSel) {
                    NVLOGE(MU_TEST_TAG, AERIAL_NVIPC_API_EVENT,
                        "GPU/CPU mismatch: cell %u ueg %u ue %u layerSel GPU=0x%02x CPU=0x%02x",
                        cellId, uegId, ueId, gpu_ue.layerSel, cpu_ue.layerSel);
                    match = false;
                }

                if (gpu_ue.ueOrderInGrp != cpu_ue.ueOrderInGrp) {
                    NVLOGE(MU_TEST_TAG, AERIAL_NVIPC_API_EVENT,
                        "GPU/CPU mismatch: cell %u ueg %u ue %u ueOrderInGrp GPU=%u CPU=%u",
                        cellId, uegId, ueId, gpu_ue.ueOrderInGrp, cpu_ue.ueOrderInGrp);
                    match = false;
                }

                if (gpu_ue.nSCID != cpu_ue.nSCID) {
                    NVLOGE(MU_TEST_TAG, AERIAL_NVIPC_API_EVENT,
                        "GPU/CPU mismatch: cell %u ueg %u ue %u nSCID GPU=%u CPU=%u",
                        cellId, uegId, ueId, gpu_ue.nSCID, cpu_ue.nSCID);
                    match = false;
                }

                if (gpu_ue.flags != cpu_ue.flags) {
                    NVLOGE(MU_TEST_TAG, AERIAL_NVIPC_API_EVENT,
                        "GPU/CPU mismatch: cell %u ueg %u ue %u flags GPU=0x%02x CPU=0x%02x",
                        cellId, uegId, ueId, gpu_ue.flags, cpu_ue.flags);
                    match = false;
                }
            }
        }
    }

    return match;
}
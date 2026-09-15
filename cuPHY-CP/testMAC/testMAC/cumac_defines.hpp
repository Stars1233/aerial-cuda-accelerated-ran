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

#ifndef _CUMAC_DEFINES_HPP_
#define _CUMAC_DEFINES_HPP_

#include <stdint.h>
#include <stddef.h>
#include <string.h>

#include <map>
#include <vector>
#include <atomic>
#include <iostream>
#include <unordered_map>

#include "cuphy.h"
#include "common_defines.hpp"
#include "cumac_msg.h"
// #include "api.h"

static constexpr uint32_t TESTMAC_CUMAC_TASK_MASK_4T4R{(static_cast<uint32_t>(1U) << CUMAC_TASK_PFM_SORT) - 1U};

// Buffer pointers for SCH_TTI.req
typedef struct cumac_tti_req_buf_ptrs
{
    uint32_t taskBitMask; // Indicate which cuMAC tasks to be scheduled. Each bit represent 1 task type defined in cumac_task_type_t
    uint16_t cellID;      // cell ID
    uint8_t ULDLSch;      // Indication for UL/DL scheduling. Value - 0: UL scheduling, 1: DL scheduling
    uint16_t nActiveUe;   // total number of active UEs in the cell
    uint16_t nSrsUe;      // the number of UEs in the cell that have refreshed SRS channel estimates
    uint16_t nPrbGrp;     // the number of PRGs that can be allocated for the current TTI, excluding the PRGs that need to be reserved for HARQ re-tx's
    uint8_t nBsAnt;       // number of BS antenna ports
    uint8_t nUeAnt;       // number of UE antenna ports
    float sigmaSqrd;      // noise variance

    // data buffer pointers
    uint16_t*   CRNTI; // C-RNTIs of all active UEs in the cell
    uint16_t*   srsCRNTI; // C-RNTIs of the UEs that have refreshed SRS channel estimates in the cell.
    uint8_t*    prgMsk; // Bit map for the availability of each PRG for allocation
    float*      postEqSinr; // array of the per-PRG per-layer post-equalizer SINRs of all active UEs in the cell
    float*      wbSinr; // array of wideband per-layer post-equalizer SINRs of all active UEs in the cell
    cuComplex*  estH_fr; // For FP32. array of the subband (per-PRG) SRS channel estimate coefficients for all active UEs in the cell
    cuComplex*  estH_fr_half; // For FP16. array of the subband (per-PRG) SRS channel estimate coefficients for all active UEs in the cell
    cuComplex*  prdMat; // array of the precoder/beamforming weights for all active UEs in the cell
    cuComplex*  detMat; // array of the detector/beamforming weights for all active UEs in the cell
    float*      sinVal; // array of the per-UE, per-PRG, per-layer singular values obtained from the SVD of the channel matrix
    float*      avgRatesActUe; // array of the long-term average data rates of all active UEs in the cell
    uint16_t*   prioWeightActUe; // For priority-based UE selection. Priority weights of all active UEs in the cell
    int8_t*     tbErrLastActUe; // TB decoding error indicators of all active UEs in the cell
    int8_t*     newDataActUe; // Indicators of initial transmission/retransmission for all active UEs in the cell
    int16_t*    allocSolLastTxActUe; // The PRG allocation solution for the last transmissions of all active UEs in the cell
    int16_t*    mcsSelSolLastTxActUe; // MCS selection solution for the last transmissions of all active UEs in the cell
    uint8_t*    layerSelSolLastTxActUe; // Layer selection solution for the last transmissions of all active UEs in the cell
    int8_t*     cqiActUe; // CQI values of all active UEs in the cell
    cumac_pfm_cell_info_t* pfmCellInfo; // PFM sorting input buffer

    //! MU-MIMO UE pairing / muUeGrp request from HDF5 task_in_buf (per cell); nullptr if not loaded
    cumac_muUeGrp_req_info_t* muUeGrpInfo; // cumac_muUeGrp_req_info_t
    size_t                    muUeGrpReqDataLen;
    bool                      muUeGrpIsMemSharing;

} cumac_tti_req_tv_t;

// Buffer pointers for SCH_TTI.resp
typedef struct
{
    uint16_t*   setSchdUePerCellTTI; // Set of IDs of the selected UEs for the cell
    int16_t*    allocSol; // PRB group allocation solution for all active UEs in the cell
    uint8_t*    layerSelSol; // Layer selection solution for all active UEs in the cell
    int16_t*    mcsSelSol; // MCS selection solution for all active UEs in the cell
    cumac_pfm_output_cell_info_t* pfmSortSol; // PFM sorting output buffer
    cumac_muUeGrp_resp_info_t* muUeGrpSol;     // Expected muUeGrp output for this cell (from *_solution.h5)
} cumac_tti_resp_tv_t;

typedef cumac_config_req_payload_t cumac_cell_configs_t;

typedef struct
{
    cumac_tti_req_tv_t  req;
    cumac_tti_resp_tv_t resp;
} cumac_test_vector_t;

typedef enum
{
    CUMAC_SCH_TTI_REQ = 0,
    CUMAC_REQ_SIZE = 1
} cumac_group_t;

static inline const char* get_task_name(int task_type)
{
    switch(task_type)
    {
    case CUMAC_TASK_UE_SELECTION:
        return "UE_SEL";
    case CUMAC_TASK_PRB_ALLOCATION:
        return "PRB_ALLOC";
    case CUMAC_TASK_LAYER_SELECTION:
        return "LAYER_SEL";
    case CUMAC_TASK_MCS_SELECTION:
        return "MCS_SEL";
    case CUMAC_TASK_PFM_SORT:
        return "PFM_SORT";
    case CUMAC_TASK_MU_UE_GRP:
        return "MU_UE_GRP";
    default:
        return "INVALID";
    }
}

static inline cumac_task_type_t get_task_type(const char* channel_name)
{
    for(int ch = 0; ch < cumac_task_type_t::CUMAC_TASK_TOTAL_NUM; ch++)
    {
        const char* name = get_task_name(ch);
        if(strncmp(name, channel_name, strlen(name)) == 0)
        {
            return cumac_task_type_t(ch);
        }
    }
    return cumac_task_type_t::CUMAC_TASK_TOTAL_NUM;
}

typedef struct
{
    int              cell_idx;
    int              slot_idx;
    std::string      tv_file;
    cumac_test_vector_t*   tv_data;
} cumac_req_t;

class cumac_thrput_t {
public:
    cumac_thrput_t() {
        reset();
    }

    cumac_thrput_t(const cumac_thrput_t& obj) {
        reset();
    }

    void reset() {
        cumac_slots = 0;
        for (int i = 0; i < CUMAC_TASK_TOTAL_NUM; i++) {
            task_slots[i] = 0;
        }
        error = 0;
        invalid = 0;
    }

    std::atomic<uint32_t> cumac_slots;
    std::atomic<uint32_t> task_slots[CUMAC_TASK_TOTAL_NUM]; //!< Individual counters for each task type

    // ERROR.indication counter
    std::atomic<uint32_t> error;

    // Validation failure counter
    std::atomic<uint32_t> invalid;
}; // cumac_thrput_t;

#endif /* _CUMAC_DEFINES_HPP_ */

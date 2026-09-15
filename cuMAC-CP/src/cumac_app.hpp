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

#include <stdint.h>
#include <stddef.h>

#include "cumac_msg.h"

#ifndef _CUMAC_APP_
#define _CUMAC_APP_

//! Maximum path length for file paths
#define MAX_PATH_LEN 1024

/**
 * Number of parent directories to traverse to reach cuBB_SDK root
 *
 * Example: 2 means "../../" from current process directory
 */
#define CONFIG_CUBB_ROOT_DIR_RELATIVE_NUM 2

//! Path to cuMAC CP configuration YAML file
#define CONFIG_CUMAC_CONFIG_YAML "cuMAC-CP/config/cumac_cp.yaml"
//! Path to multi-cell launch pattern files
#define CONFIG_LAUNCH_PATTERN_PATH "testVectors/multi-cell/"
//! Path to test vector files
#define CONFIG_TEST_VECTOR_PATH "testVectors/"

/**
 * Extract bit field from integer value
 *
 * @param var Source variable
 * @param start Starting bit position
 * @param width Number of bits to extract
 */
#define INTEGER_GET_BITS(var, start, width) (((var) >> (start)) & ((1 << (width)) - 1))

/**
 * Set bit field in integer value
 *
 * @param var Target variable
 * @param start Starting bit position
 * @param width Number of bits to set
 * @param val Value to set in bit field
 */
#define INTEGER_SET_BITS(var, start, width, val) ((var & ~(((1 << width) - 1) << start)) | (val << start))

//! NR numerology (0 ~ 4): 1 means 30kHz subcarrier spacing
inline constexpr uint32_t NR_NUMEROLOGY = 1;
//! Slot interval duration in nanoseconds
inline constexpr uint64_t SLOT_INTERVAL = (1000L * 1000 / (1 << NR_NUMEROLOGY));
//! Number of slots per second
inline constexpr uint64_t SLOTS_PER_SECOND = (1000L * 1000 * 1000 / SLOT_INTERVAL);
//! Number of slots per frame (10ms)
inline constexpr uint64_t SLOT_NUM_PER_FRAME = (1000L * 1000 * 10 / SLOT_INTERVAL);

//! Maximum number of SFNs
inline constexpr uint64_t SFN_NUM_MAX = 1024;
//! Maximum number of SFNs * slots per frame
inline constexpr uint64_t SFN_SLOT_NUM_MAX = SFN_NUM_MAX * SLOT_NUM_PER_FRAME;

//! cuMAC-CP task mask constants
inline constexpr uint32_t CUMAC_CP_TASK_MASK_4T4R = (static_cast<uint32_t>(1U) << CUMAC_TASK_PFM_SORT) - 1U;
inline constexpr uint32_t CUMAC_CP_TASK_MASK_PFM_SORT = static_cast<uint32_t>(1U) << CUMAC_TASK_PFM_SORT;
inline constexpr uint32_t CUMAC_CP_TASK_MASK_MU_UE_GRP = static_cast<uint32_t>(1U) << CUMAC_TASK_MU_UE_GRP;
inline constexpr uint32_t CUMAC_CP_TASK_MASK_DEFAULT = 0xFFFFU;

#define CHECK_CUDA_ERR(stmt)                                                                                                                                     \
    do                                                                                                                                                           \
    {                                                                                                                                                            \
        cudaError_t result1 = (stmt);                                                                                                                            \
        if (cudaSuccess != result1)                                                                                                                              \
        {                                                                                                                                                        \
            NVLOGW_FMT(TAG, "[{}:{}] cuda failed with result1 {} ", __FILE__, __LINE__, cudaGetErrorString(result1));                                            \
            cudaError_t result2 = cudaGetLastError();                                                                                                            \
            if (cudaSuccess != result2)                                                                                                                          \
            {                                                                                                                                                    \
                NVLOGW_FMT(TAG, "[{}:{}] cuda failed with result2 {} result1 {}", __FILE__, __LINE__, cudaGetErrorString(result2), cudaGetErrorString(result1)); \
                cudaError_t result3 = cudaGetLastError(); /*check for stickiness*/                                                                               \
                if (cudaSuccess != result3)                                                                                                                      \
                {                                                                                                                                                \
                    NVLOGF_FMT(TAG, AERIAL_CUDA_API_EVENT, "[{}:{}] cuda failed with result3 {} result2 {} result1 {}",                                          \
                               __FILE__,                                                                                                                         \
                               __LINE__,                                                                                                                         \
                               cudaGetErrorString(result3),                                                                                                      \
                               cudaGetErrorString(result2),                                                                                                      \
                               cudaGetErrorString(result1));                                                                                                     \
                }                                                                                                                                                \
            }                                                                                                                                                    \
        }                                                                                                                                                        \
    } while (0)

/**
 * cuMAC buffer element counts structure
 *
 * Stores the number of elements (NOT bytes) for each buffer type.
 * These counts are used for buffer allocation and indexing.
 */
typedef struct
{
    uint32_t cellId; //!< Cell ID array: nCell elements
    uint32_t prgMsk; //!< PRB group mask: nPrbGrp elements
    uint32_t wbSinr; //!< Wideband SINR: float[nActiveUe * nUeAnt]
    uint32_t avgRatesActUe; //!< Average rates for active UEs: float[nActiveUe]
    uint32_t avgRates; //!< Average rates for all UEs: float[nUe]

    uint32_t setSchdUePerCellTTI; //!< Scheduled UEs per cell per TTI: uint16_t[nCell * numUeSchdPerCellTTI]
    uint32_t postEqSinr; //!< Post-equalization SINR: float[nActiveUe * nPrbGrp * nUeAnt]
    uint32_t cellAssoc; //!< Cell association for all UEs: uint8_t[nCell * nUe]
    uint32_t cellAssocActUe; //!< Cell association for active UEs: uint8_t[nCell * nActiveUe]
    uint32_t blerTargetActUe; //!< BLER target for active UEs: float[nActiveUe]

    uint32_t sinVal; //!< Singular values: float[nUe * nPrbGrp * nUeAnt]
    uint32_t prdMat; //!< Product matrix: complex[nUe * nPrbGrp * nBsAnt * nBsAnt]
    uint32_t detMat; //!< Determinant matrix: complex[nUe * nPrbGrp * nBsAnt * nBsAnt]
    uint32_t estH_fr; //!< Estimated channel frequency domain: complex[nPrbGrp * nUe * nCell * nBsAnt * nUeAnt]
    uint32_t allocSol; //!< Resource allocation solution: int16_t array

    uint32_t tbErrLastActUe; //!< Transport block error for active UEs: int8_t array
    uint32_t tbErrLast; //!< Transport block error for all UEs: int8_t array
    uint32_t pfMetricArr; //!< Proportional fair metric array
    uint32_t pfIdArr; //!< Proportional fair ID array
    uint32_t newDataActUe; //!< New data indicator for active UEs

    uint32_t layerSelSol; //!< Layer selection solution: uint8_t array
    uint32_t mcsSelSol; //!< MCS selection solution: int16_t array
    uint32_t pfmCellInfo; //!< PFM sorting input: cumac::PFM_CELL_INFO array
    uint32_t pfmSortSol; //!< PFM sorting output: cumac::PFM_OUTPUT_CELL_INFO array
    uint32_t muUeGrpInfo; //!< MU-MIMO UE grouping request: total uint8_t slots (nCell * per TV layout or max stride)

    uint32_t muUeGrpSol;     //!< MU-MIMO UE grouping output: cumac_muUeGrp_resp_info_t per cell
} cumac_buf_num_t;

#endif /* _CUMAC_APP_ */

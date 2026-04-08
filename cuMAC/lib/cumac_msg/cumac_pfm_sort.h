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

#ifndef _CUMAC_PFM_SORT_DATA_TYPE_
#define _CUMAC_PFM_SORT_DATA_TYPE_

#include <stdint.h>
#include <stddef.h>

#if defined(__cplusplus)
extern "C" {
#endif

// ************************************************** PFM sorting constants **************************************************
/// @brief Time slot interval in seconds
#define CUMAC_PFM_SLOT_DURATION (0.0005F) // 0.5 ms

/// @brief Maximum number of cells
#define CUMAC_PFM_MAX_NUM_CELL (40U)

/// @brief Maximum number of UEs for PFM sorting per cell
/// @note a maximum of 512 UEs per cell can be supported
#define CUMAC_PFM_MAX_NUM_UE_PER_CELL (512U)

/// @brief Number of QoS types for UL
/// @note 0 - ul_gbr_critical, 1 - ul_gbr_non_critical, 2 - ul_ngbr_critical, 3 - ul_ngbr_non_critical, 4 - ul_mbr_non_critical
#define CUMAC_PFM_NUM_QOS_TYPES_UL (5U)

/// @brief Number of QoS types for DL
/// @note 0 - dl_gbr_critical, 1 - dl_gbr_non_critical, 2 - dl_ngbr_critical, 3 - dl_ngbr_non_critical, 4 - dl_mbr_non_critical
#define CUMAC_PFM_NUM_QOS_TYPES_DL (5U)

/// @brief Maximum number of LCGs per UE (for UL)
#define CUMAC_PFM_MAX_NUM_LCG_PER_UE (4U)

/// @brief Maximum number of LCs per UE (for DL)
#define CUMAC_PFM_MAX_NUM_LC_PER_UE (4U)

/// @brief Maximum number of output sorted LCs per DL QoS type per cell
#define CUMAC_PFM_MAX_NUM_SORTED_LC_PER_QOS (CUMAC_PFM_MAX_NUM_UE_PER_CELL * CUMAC_PFM_MAX_NUM_LC_PER_UE)

/// @brief Maximum number of output sorted LCGs per UL QoS type per cell
#define CUMAC_PFM_MAX_NUM_SORTED_LCG_PER_QOS (CUMAC_PFM_MAX_NUM_UE_PER_CELL * CUMAC_PFM_MAX_NUM_LCG_PER_UE)

/// @brief Maximum number of scheduled UEs per cell per time slot for both DL and UL
#define CUMAC_PFM_MAX_NUM_SCHEDULED_UE_PER_CELL (16U)

/// @brief IIR alpha for ravg calculation
#define CUMAC_PFM_IIR_ALPHA (0.001F)

#define CUMAC_INVALID_RNTI (0U)

// ************************************************** PFM sorting request data structures (sent from L2 stack host to cuMAC-CP) **************************************************
/// @brief Downlink LC info structure
typedef struct {
    uint32_t            tbs_scheduled; // TBS scheduled last time for this LC
    uint8_t             flags; // a collection of flags: flags & 0x01 - is_valid, flags & 0x02 - reset ravg to 1.0
    uint8_t             qos_type; // 0 - dl_gbr_critical, 1 - dl_gbr_non_critical, 2 - dl_ngbr_critical, 3 - dl_ngbr_non_critical, 4 - dl_mbr_non_critical
} cumac_pfm_dl_lc_info_t;

/// @brief Uplink LCG info structure
typedef struct {
    uint32_t            tbs_scheduled; // TBS scheduled last time for this LCG
    uint8_t             flags; // a collection of flags: flags & 0x01 - is_valid, flags & 0x02 - reset ravg to 1.0
    uint8_t             qos_type; // 0 - ul_gbr_critical, 1 - ul_gbr_non_critical, 2 - ul_ngbr_critical, 3 - ul_ngbr_non_critical, 4 - ul_mbr_non_critical 
} cumac_pfm_ul_lcg_info_t;

/// @brief Per-UE info structure
typedef struct {
    uint32_t                rcurrent_dl; // current instantaneous rate of this UE for DL
    uint32_t                rcurrent_ul; // current instantaneous rate of this UE for UL
    uint16_t                rnti; // RNTI of this UE
    uint16_t                id; // cuMAC 0-based UE ID of this UE
    uint8_t                 num_layers_dl; // number of layers scheduled for this UE last time for DL
    uint8_t                 num_layers_ul; // number of layers scheduled for this UE last time for UL
    uint8_t                 flags; // a collection of flags: flags & 0x01 - is_scheduled_dl, flags & 0x02 - is_scheduled_ul
    cumac_pfm_dl_lc_info_t  dl_lc_info[CUMAC_PFM_MAX_NUM_LC_PER_UE]; // LCs configured for this UE for DL
    cumac_pfm_ul_lcg_info_t ul_lcg_info[CUMAC_PFM_MAX_NUM_LCG_PER_UE]; // LCGs configured for this UE for UL
} cumac_pfm_ue_info_t;

/// @brief PFM sorting request data structure per cell
typedef struct {
    uint16_t                num_ue; // number of UEs for the PFM sorting in the current slot for this cell
    uint8_t                 num_lc_per_ue; // number of LCs per UE for the PFM sorting in the current slot for this cell
    uint8_t                 num_lcg_per_ue; // number of LCGs per UE for the PFM sorting in the current slot for this cell
    uint16_t                num_output_sorted_lc[CUMAC_PFM_NUM_QOS_TYPES_UL + CUMAC_PFM_NUM_QOS_TYPES_DL]; // array of number of output sorted LCs/LCGs per DL/UL QoS type
    cumac_pfm_ue_info_t     ue_info[CUMAC_PFM_MAX_NUM_UE_PER_CELL]; // UE info for the PFM sorting in the current slot for this cell
} cumac_pfm_cell_info_t;

// ************************************************** PFM sorting device output data structures **************************************************
/// @brief DL output info structure for the GPU PFM sorting
typedef struct {
    uint16_t            rnti; // UE RNTI of this DL info struct
    uint8_t             lc_id; // LC ID of this DL info struct
} cumac_pfm_dl_output_info_t;

/// @brief UL output info structure for the GPU PFM sorting
typedef struct {
    uint16_t            rnti; // UE RNTI of this UL info struct
    uint8_t             lcg_id; // LCG ID of this UL info struct
} cumac_pfm_ul_output_info_t;

/// @brief Output info structure for the GPU PFM sorting per cell
typedef struct {
    cumac_pfm_dl_output_info_t  dl_gbr_critical[CUMAC_PFM_MAX_NUM_SORTED_LC_PER_QOS]; // array of sorted LCs for the GBR critical QoS type 
    cumac_pfm_dl_output_info_t  dl_gbr_non_critical[CUMAC_PFM_MAX_NUM_SORTED_LC_PER_QOS]; // array of sorted LCs for the GBR non-critical QoS type 
    cumac_pfm_dl_output_info_t  dl_ngbr_critical[CUMAC_PFM_MAX_NUM_SORTED_LC_PER_QOS]; // array of sorted LCs for the NGBR critical QoS type 
    cumac_pfm_dl_output_info_t  dl_ngbr_non_critical[CUMAC_PFM_MAX_NUM_SORTED_LC_PER_QOS]; // array of sorted LCs for the NGBR non-critical QoS type 
    cumac_pfm_dl_output_info_t  dl_mbr_non_critical[CUMAC_PFM_MAX_NUM_SORTED_LC_PER_QOS]; // array of sorted LCs for the MBR non-critical QoS type 

    cumac_pfm_ul_output_info_t  ul_gbr_critical[CUMAC_PFM_MAX_NUM_SORTED_LCG_PER_QOS]; // array of sorted LCGs for the GBR critical QoS type 
    cumac_pfm_ul_output_info_t  ul_gbr_non_critical[CUMAC_PFM_MAX_NUM_SORTED_LCG_PER_QOS]; // array of sorted LCGs for the GBR non-critical QoS type 
    cumac_pfm_ul_output_info_t  ul_ngbr_critical[CUMAC_PFM_MAX_NUM_SORTED_LCG_PER_QOS]; // array of sorted LCGs for the NGBR critical QoS type 
    cumac_pfm_ul_output_info_t  ul_ngbr_non_critical[CUMAC_PFM_MAX_NUM_SORTED_LCG_PER_QOS]; // array of sorted LCGs for the NGBR non-critical QoS type 
    cumac_pfm_ul_output_info_t  ul_mbr_non_critical[CUMAC_PFM_MAX_NUM_SORTED_LCG_PER_QOS]; // array of sorted LCGs for the MBR non-critical QoS type 
} cumac_pfm_output_cell_info_t;


#if defined(__cplusplus)
} /* extern "C" */
#endif

#endif /* _CUMAC_PFM_SORT_DATA_TYPE_ */

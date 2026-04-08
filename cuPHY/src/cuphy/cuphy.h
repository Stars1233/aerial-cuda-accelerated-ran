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

/** \file cuphy.h
 *  \brief PHY Layer library header file
 *
 *  Header file for the cuPHY API
 */

#if !defined(CUPHY_H_INCLUDED_)
#define CUPHY_H_INCLUDED_

#include <cuda_runtime.h>
#include <stdint.h>
#include "cuComplex.h"
#include "cuda_fp16.h"
#include <cuda.h>

#ifndef CUPHYWINAPI
#ifdef _WIN32
#define CUPHYWINAPI __stdcall
#else
#define CUPHYWINAPI
#endif
#endif

/// @cond TODO none of these have doxygen comments
//#define _HUGE_TB_TEST_ 1
#if defined(__cplusplus)
extern "C" {
#endif /* defined(__cplusplus) */
#define MAX_N_DMRSSYMS_SUPPORTED 4
#define MAX_ENCODED_CODE_BLOCK_BIT_SIZE 25344 // Do not modify. Used in DL channels.
#define MAX_DECODED_CODE_BLOCK_BIT_SIZE 8448
#define MAX_N_BBU_LAYERS_SUPPORTED 32
#define MAX_N_BBU_LAYERS_PUCCH_SUPPORTED 16
#define MAX_N_BBU_LAYERS_PUSCH_SUPPORTED 16
#define MAX_N_LAYERS_PUSCH 8 // maximum number of layers per PUSCH UE group
#define MAX_N_ANTENNAS_SUPPORTED 64
#define MAX_N_CARRIERS_SUPPORTED 10
#define MAX_NF_SUPPORTED (3276 * MAX_N_CARRIERS_SUPPORTED)
#define MAX_ND_SUPPORTED 14
#define OFDM_SYMBOLS_PER_SLOT 14
#define N_MAX_DMRS_SYMS 4
#define MAX_N_ADDLN_POS 4
#ifdef ENABLE_64C
#define MAX_N_TBS_PER_CELL_GROUP_SUPPORTED 256                              // maximum number of transport blocks supported for an entire cell-group
#define MAX_N_TBS_SUPPORTED (MAX_N_TBS_PER_CELL_GROUP_SUPPORTED)            // maximum number of transport blocks supported
#define MAX_N_USER_GROUPS_SUPPORTED 256                                     // maximum number of FDM user groups supported per cell
#else
#define MAX_N_TBS_PER_CELL_GROUP_SUPPORTED 192                              // maximum number of transport blocks supported for an entire cell-group
#define MAX_N_TBS_SUPPORTED (MAX_N_TBS_PER_CELL_GROUP_SUPPORTED)            // maximum number of transport blocks supported
#define MAX_N_USER_GROUPS_SUPPORTED 192                                     // maximum number of FDM user groups supported per cell
#endif
#define MAX_N_PRBS_SUPPORTED 273
#define CUPHY_PUSCH_RX_MAX_N_TIME_CH_EST (4)                               // maximum number of channel estimates in time (max of 3 additional positions)
#define CUPHY_PUSCH_RX_MAX_N_TIME_CH_EQ (CUPHY_PUSCH_RX_MAX_N_TIME_CH_EST) // maximum number of channel equalizer coefficient sets
#define CUPHY_PUSCH_RX_MAX_N_LAYERS_PER_UE_GROUP (8)
#define CUPHY_PUSCH_RX_MAX_N_UE_PER_UE_GROUP (CUPHY_PUSCH_RX_MAX_N_LAYERS_PER_UE_GROUP) // 8 UEs (1 layer each)
#define PUSCH_STREAM_PRIORITY (-2)            // Default priority used for PUSCH CUDA streams (e.g., in cuphyPuschStatPrms_t) in PUSCH examples
#ifdef ENABLE_64C
#define MAX_CELLS_PER_SLOT (40)
#else
#define MAX_CELLS_PER_SLOT (20)
#endif
#define DL_MAX_CELLS_PER_SLOT MAX_CELLS_PER_SLOT
#define UL_MAX_CELLS_PER_SLOT MAX_CELLS_PER_SLOT

// macro for early-HARQ processing
#define CUPHY_PUSCH_RX_SOFT_DEMAPPER_EARLY_HARQ_SYMBOL_BITMASK (0xF)  //0xF
#define CUPHY_PUSCH_RX_SOFT_DEMAPPER_EARLY_HARQ_SYMBOL_BITMASK_MMIMO_EXTRA_DMRS (0x1F)  //0x1F Extra DMRS symbol from maxLen = 2 to support increased UL layer count in 64TR mMIMO. Thus HARQ bits will be configured in symbol 4 instead of symbol 3 
#define CUPHY_PUSCH_RX_SOFT_DEMAPPER_EARLY_HARQ_SYMBOL_UPPER_BOUND (3)
#define EARLY_HARQ_SYM_IDX_UB (4)

#define CUPHY_PUSCH_RX_SOFT_DEMAPPER_FULL_SLOT_SYMBOL_BITMASK (0x3FFF) //0x3FFF

// macro to enable sub-slot processing
#define CUPHY_ENABLE_SUB_SLOT_PROCESSING (0)
#define N_MAX_SUB_SLOT_STAGES (N_MAX_DMRS_SYMS+1)

//WIP limits for PDSCH multiple cells in a cell group. TODO Could use some runtime limits from static parameters
#define PDSCH_MAX_UES_PER_CELL 64
#define PDSCH_MAX_HET_LDPC_CONFIGS_SUPPORTED 64 // Maximum number of heterogeneous LDPC encoder configs supported by PDSCH for a single cell group
#if 1
#define PDSCH_MAX_CELLS_PER_CELL_GROUP 64 // Maximum number of cells within cell-group for a single PDSCH object
#ifdef ENABLE_64C
#define PDSCH_MAX_UES_PER_CELL_GROUP 256   // Used for overprovisioned allocations etc.
#else
#define PDSCH_MAX_UES_PER_CELL_GROUP 192   // Used for overprovisioned allocations etc.
#endif
#else
//If you want a min. single cell config, use:
#define PDSCH_MAX_CELLS_PER_CELL_GROUP 1                         // Maximum number of cells within cell-group for a single PDSCH object
#define PDSCH_MAX_UES_PER_CELL_GROUP (PDSCH_MAX_UES_PER_CELL) // Used for overprovisioned allocations
#endif
#ifdef ENABLE_64C
#define PDSCH_MAX_UE_GROUPS_PER_CELL_GROUP 256
#else
#define PDSCH_MAX_UE_GROUPS_PER_CELL_GROUP 128
#endif
#define PDSCH_MAX_CWS_PER_CELL_GROUP (PDSCH_MAX_UES_PER_CELL_GROUP) // Assuming one codeword (CW) per UE for now. At most 2x.
#define PDSCH_STREAM_PRIORITY (-5)                                  // Default priority used for PDSCH CUDA streams (e.g., in cuphyPdschStatPrms_t) in PDSCH examples
#define PDSCH_MAX_GPU_BUFFS 20 //This value is used for circulating multiple arrays used for PDSCH H2D TB copy. Value is set to match the number of slots in a radio frame
#define MAX_N_CBS_PER_TB_SUPPORTED 152                 // maximum number of code blocks per transport block supported
#define MAX_TOTAL_N_CBS_SUPPORTED ((MAX_N_TBS_SUPPORTED) * (MAX_N_CBS_PER_TB_SUPPORTED))
//#define PDSCH_MAX_TOTAL_N_CBS_SUPPORTED ((PDSCH_MAX_UES_PER_CELL) * (MAX_N_CBS_PER_TB_SUPPORTED)) // max CBs per cell (not currently used)
#define PDSCH_MAX_ER_PER_CB_BITS 256000 // maximum number or rate-matched bits for the r-th coded block of a TB supported by cuPHY PDSCH; only possible in adaptive retransmission cases

#define PDSCH_ABOVE_16_LAYERS_DMRSPORTS_BIT_LOC 15

#define PUSCH_MAX_ER_PER_CB_BITS 256000 // maximum number or rate-matched bits for the r-th coded block of a TB supported by cuPHY PUSCH; only possible in adaptive retransmission cases
// Assuming single codeword (i.e., max number of layers/UE = 4) and thus using 4 layers, 273 PRBs, MCS=27, 14 symbols w/ 1 DMRS symbol.
// + 24 accounts for per-TB CRC.
#define MAX_BYTES_PER_TRANSPORT_BLOCK (159749 + 24)
#define MAX_WORDS_PER_TRANSPORT_BLOCK (MAX_BYTES_PER_TRANSPORT_BLOCK / sizeof(uint32_t))
#define MAX_N_RM_LLRS_PER_CB 26112;

#define CUPHY_LDPC_MAX_LIFTING_SIZE (384)
#define CUPHY_LDPC_MIN_PARITY_NODES (4) // Mininum permitted parity node count irrespective of BG (base graph)
#define CUPHY_LDPC_MAX_BG1_PARITY_NODES (46)
#define CUPHY_LDPC_MAX_BG2_PARITY_NODES (42)
#define CUPHY_LDPC_MAX_BG1_VAR_NODES (68)
#define CUPHY_LDPC_MAX_BG2_VAR_NODES (52)
#define CUPHY_LDPC_BG1_INFO_NODES (22)
#define CUPHY_LDPC_MAX_BG2_INFO_NODES (10)
#define CUPHY_LDPC_NUM_PUNCTURED_NODES (2)
#define CUPHY_LDPC_MAX_BG1_UNPUNCTURED_VAR_NODES (CUPHY_LDPC_MAX_BG1_VAR_NODES - CUPHY_LDPC_NUM_PUNCTURED_NODES)
#define CUPHY_LDPC_MAX_BG2_UNPUNCTURED_VAR_NODES (CUPHY_LDPC_MAX_BG2_VAR_NODES - CUPHY_LDPC_NUM_PUNCTURED_NODES)
#define CUPHY_LDPC_DECODE_DESC_MAX_TB (32)

#define QAM_STRIDE 8 // Stride for UE QAM symbols in the equalizer output

#define CUPHY_PUSCH_RX_CH_EST_N_HOM_CFG (1)
// Maximum number of heterogenous channel estimation configs supported for legacy ChEst algorithm
#define CUPHY_PUSCH_RX_CH_EST_LEGACY_MMSE_N_MAX_HET_CFGS (16)
// Maximum number of heterogenous channel estimation configs supported for multistage ChEst algorithm
#define CUPHY_PUSCH_RX_CH_EST_MULTISTAGE_MMSE_N_MAX_HET_CFGS (4)
// Maximum number of heterogenous channel estimation configs supported for all ChEst algorithms
static constexpr int CUPHY_PUSCH_RX_CH_EST_ALL_ALGS_N_MAX_HET_CFGS = std::max(
    CUPHY_PUSCH_RX_CH_EST_LEGACY_MMSE_N_MAX_HET_CFGS, CUPHY_PUSCH_RX_CH_EST_MULTISTAGE_MMSE_N_MAX_HET_CFGS);
#define NUM_RKHS_ZP (10) // number of RKHS zero-padding sizes


// Number of PRBs per cluster used to compute shift-1 auto-correlations when using
// the PUSCH_CH_EST_ALGO_TYPE_MULTISTAGE_MMSE_WITH_DELAY_EST channel estimation algorithm
#define CUPHY_PUSCH_RX_CH_EST_NON_PRG_DELAY_EST_PRB_CLUSTER_SIZE (4)
#define CUPHY_PUSCH_RX_CH_EST_PRG_DELAY_EST_PRB_CLUSTER_SIZE (12)
#define CUPHY_PUSCH_RX_CH_EQ_N_HOM_CFG (1)
#define CUPHY_PUSCH_RX_CH_EQ_N_MAX_HET_CFGS (8) // Maximum number of heterogenous channel equalization configs supported

#define FFT128  (128)
#define FFT256  (256)
#define FFT512  (512)
#define FFT1024 (1024)
#define FFT2048 (2048)
#define FFT4096 (4096)
#define FFT8192 (8192)

#define CUPHY_PUSCH_RX_NOISE_INTF_EST_N_MAX_HET_CFGS (1) // Maximum number of heterogenous noise-interference estimation configs supported
#define CUPHY_PUSCH_RX_CFO_EST_N_MAX_HET_CFGS (1) // Maximum number of CFO estimation configs supported

// cuPHY Front-End Tensor object dimensions
#define CUPHY_PUSCH_RX_FE_N_DIM_DATA_RX (3)
#define CUPHY_PUSCH_RX_FE_N_DIM_H_EST (4)
#define CUPHY_PUSCH_RX_FE_N_DIM_START_PRB (1)
#define CUPHY_PUSCH_RX_FE_N_DIM_NUM_PRB (1)
#define CUPHY_PUSCH_RX_FE_N_DIM_CFO_EST (2)
#define CUPHY_PUSCH_RX_FE_N_DIM_TA_EST (1)
#define CUPHY_PUSCH_RX_FE_N_DIM_BATCH_CFG (2)

#define CUPHY_PUSCH_RX_CH_EST_N_DIM_FREQ_INTERP_COEFS (3)
#define CUPHY_PUSCH_RX_CH_EST_N_DIM_SHIFT_SEQ (2)
#define CUPHY_PUSCH_RX_CH_EST_N_DIM_UNSHIFT_SEQ (1)
#define CUPHY_PUSCH_RX_CH_EST_N_DIM_DMRS_SCID (1)
#define CUPHY_PUSCH_RX_CH_EST_N_DIM_DBG (4)

#define CUPHY_PUSCH_RX_CH_EQ_N_DIM_NOISE_PWR (4)
#define CUPHY_PUSCH_RX_CH_EQ_N_DIM_COEF (4)
#define CUPHY_PUSCH_RX_CH_EQ_N_DIM_REE_DIAG (3)
#define CUPHY_PUSCH_RX_CH_EQ_N_DIM_DATA_SYM_LOC (1)
#define CUPHY_PUSCH_RX_CH_EQ_N_DIM_QAM_INFO (3)
#define CUPHY_PUSCH_RX_CH_EQ_N_DIM_DATA_EQ (3)
#define CUPHY_PUSCH_RX_CH_EQ_N_DIM_LLR (4)
#define CUPHY_PUSCH_RX_CH_EQ_N_DIM_DBG (4)

// SRS parameters
#define CUPHY_SRS_MAX_N_USERS (512)
#define CUPHY_SRS_CH_EST_N_HOM_CFG (1)
#define CUPHY_SRS_CH_EST_N_HET_CFG (1)
#define CUPHY_SRS_MAX_FULL_BAND_SRS_ANT_PORTS_SLOT_PER_CELL (16)
#define CUPHY_SRS_MAX_N_PRGS_SUPPORTED (273)
#define CUPHY_SRS_MAX_FULL_BAND_CHEST_PER_TTI (8)
#define MAX_SRS_SYMBOLS_PER_SLOT 4
#define CUPHY_SRS_RKHS_WORKSPACE_SIZE_PER_UE (1048576)
#define CUPHY_SRS_RKHS_WORKSPACE_SIZE_PER_CELL (CUPHY_SRS_RKHS_WORKSPACE_SIZE_PER_UE * 16)

#define CUPHY_SRS_CH_EST_N_DIM_DATA_RX (3)
#define CUPHY_SRS_CH_EST_N_DIM_FREQ_INTERP_COEFS (3)
#define CUPHY_SRS_CH_EST_N_DIM_H_EST (4)
#define CUPHY_SRS_CH_EST_N_DIM_DBG (4)
#define NUM_RKHS_GRIDS (3)  // number of RKHS grid sizes
//#define CUPHY_SRS_DISABLE_RKHS (1)

// BFW parameters
#define CUPHY_ENABLE_FLEXIBLE_BFW_PRB_GRPS (1)
#define CUPHY_BFW_COEF_COMP_N_MAX_HET_CFGS (8)
#ifdef ENABLE_32DL
#define CUPHY_BFW_COEF_COMP_N_MAX_LAYERS_PER_USER_GRP (32)
#else
#define CUPHY_BFW_COEF_COMP_N_MAX_LAYERS_PER_USER_GRP (16) // Maximum number of layers beamformed
#endif
#define CUPHY_BFW_COEF_COMP_N_MAX_USER_GRPS (72) // 24 UeGrps/cell * 3 cells
#define CUPHY_BFW_MIN_PRB_GRP_SIZE 2
#define CUPHY_BFW_N_MAX_PRB_GRPS ((MAX_N_PRBS_SUPPORTED+1)/CUPHY_BFW_MIN_PRB_GRP_SIZE)

// CFO/TA estimation
#define CUPHY_PUSCH_RX_CFO_N_DIM_PHASE_ROT (3)
#define CUPHY_PUSCH_RX_TA_N_DIM_PHASE_ROT (3)
#define CUPHY_PUSCH_RX_CFO_TA_N_DIM_INTER_CTA_SYNC (1)
#define CUPHY_PUSCH_RX_CFO_N_DIM_DBG (4)
#define CUPHY_PUSCH_RX_CFO_CHECK_THRESHOLD (2.0)
#define CUPHY_PUSCH_RX_TO_CHECK_THRESHOLD (0.01)

// RSSI
#define CUPHY_PUSCH_RX_RSSI_N_MAX_HET_CFGS (1)

#define CUPHY_PUSCH_RSSI_N_DIM_MEAS_FULL (3)
#define CUPHY_PUSCH_RSSI_N_DIM_MEAS (1)
#define CUPHY_PUSCH_RSSI_N_DIM_INTER_CTA_SYNC (1)

// RSRP
#define CUPHY_PUSCH_RX_RSRP_N_MAX_HET_CFGS (1)

// PDCCH
#define CUPHY_PDCCH_N_CRC_BITS (24)
#define CUPHY_PDCCH_MAX_DCIS_PER_CORESET (91)                                           // Increased to support new TVs; previous max. value was 32
#define CUPHY_PDCCH_N_MAX_CORESETS_PER_CELL (40)
#define CUPHY_PDCCH_MAX_DCI_PAYLOAD_BYTES (20)                                          //picked byte size after 140 bits divisible by 4 TODO Can change
#define CUPHY_PDCCH_MAX_DCI_PAYLOAD_BYTES_W_CRC (CUPHY_PDCCH_MAX_DCI_PAYLOAD_BYTES + 4) // CRC is 24-bits, here rounded up so size is divisible by 32 TODO Can change
#define CUPHY_PDCCH_MAX_AGGREGATION_LEVEL (16)
#define CUPHY_PDCCH_MAX_TX_BITS_PER_DCI (2 * 9 * 6 * CUPHY_PDCCH_MAX_AGGREGATION_LEVEL) // Needs to be divisible by 32

// SSB (Signal Synchronization Block)
#define CUPHY_SSB_N_MIB_BITS (24) // MIB (master information block) bit length before payload generation
#define CUPHY_SSB_N_PBCH_PAYLOAD_BITS (32) // length of payload sequence for PBCH
#define CUPHY_SSB_N_PBCH_SEQ_W_CRC_BITS (56) // length of PBCH sequence with CRC in bits
#define CUPHY_SSB_N_PBCH_POLAR_ENCODED_BITS (512)    // length of polar encoded sequence for PBCH

#define CUPHY_SSB_N_PBCH_SCRAMBLING_SEQ_BITS (864)    // length of scrambl. sequence for PBCH in bits
#define CUPHY_SSB_N_DMRS_SEQ_BITS (288)    // length of gold sequence for DMRS in bits
#define CUPHY_SSB_N_SS_SEQ_BITS (127)    // length of PSS/SSS sequence in bits
#define CUPHY_SSB_NF (240) // number of freq. domain subcarriers for SSB
#define CUPHY_SSB_NT (4) // number of time domain symbols for SSB
#define CUPHY_SSB_MAX_SSBS_PER_CELL_PER_SLOT (3) // maximum number of SSBs per cell in a slot

// PUCCH
#define CUPHY_PUCCH_F0_MAX_GRPS (MAX_CELLS_PER_SLOT * 10) // supporting 10 UCI groups per cell
#define CUPHY_PUCCH_F0_MAX_UCI_PER_GRP (12)
#define CUPHY_PUCCH_F1_MAX_GRPS (MAX_CELLS_PER_SLOT * 24) // supporting 24 UCI groups per cell
#define CUPHY_PUCCH_F1_MAX_UCI_PER_GRP (45)
#define CUPHY_PUCCH_F2_MAX_UCI (MAX_CELLS_PER_SLOT * 24)  // supporting 24 UCIs per cell
#define CUPHY_PUCCH_F2_MAX_PRBS (16)
#define CUPHY_PUCCH_F3_MAX_UCI (MAX_CELLS_PER_SLOT * 24)  // supporting 24 UCIs per cell
#define CUPHY_PUCCH_F3_MAX_PRBS (16)
        // TODO Likely deprecated
#define CUPHY_PUCCH_FORMAT1 1
#define NUM_PAPR_SEQUENCES 30
#define MAX_UE_CNT 42 // 6 * OFDM_SYMBOLS_PER_SLOT / 2



#define CUPHY_DEFAULT_EXT_DTX_THRESHOLD (-100.0)

// Polar Decoder
#define CUPHY_POLAR_DECODER_MAX_BITS (1024)

// UCI
#define CUPHY_MAX_N_UCI_ON_PUSCH (MAX_N_TBS_PER_CELL_GROUP_SUPPORTED)
#define CUPHY_MAX_N_PUSCH_CSI2 (MAX_N_TBS_PER_CELL_GROUP_SUPPORTED)
// Each PUSCH TB can have up to three UCI payloads "riding along" with it: HARQ UCI, CSI-P1 UCI, and CSI-P2 UCI.
// We decode HARQ UCI and CSI-P1 UCI in parallel and need parameters for both. Thus, CUPHY_MAX_N_SPX_CWS,
// CUPHY_MAX_N_POL_UCI_SEGS and CUPHY_MAX_N_POL_CWS are defined with 2x since HARQ UCI and CSI-P1 UCI are run in parallel
#define CUPHY_MAX_N_POL_UCI_SEGS (2*MAX_N_TBS_PER_CELL_GROUP_SUPPORTED)
#define CUPHY_MAX_N_SPX_CWS (2*MAX_N_TBS_PER_CELL_GROUP_SUPPORTED)
#define CUPHY_MAX_N_POL_CWS (CUPHY_MAX_N_POL_UCI_SEGS*2) // should be 2 x CUPHY_MAX_N_POL_UCI_SEGS
#define CUPHY_MAX_N_POL_UCI_SEGS_CSI2 (MAX_N_TBS_PER_CELL_GROUP_SUPPORTED)
#define CUPHY_MAX_N_POL_CWS_CSI2 (2*CUPHY_MAX_N_POL_UCI_SEGS_CSI2)
#define CUPHY_MAX_N_CSI2_WORDS (54)
#define CUPHY_N_MAX_UCI_BITS_SIMPLEX (2)
#define CUPHY_N_MAX_UCI_BITS_RM (11)
#define CUPHY_POLAR_DECODER_LIST_SIZE (8)
#define CUPHY_POLAR_ENC_MAX_INFO_BITS (164)
#define CUPHY_POLAR_ENC_MAX_CODED_BITS (512)
#define CUPHY_POLAR_ENC_MAX_TX_BITS (8192)
#define CUPHY_MAX_N_CSI2_REPORTS_PER_UE (16)
#define CUPHY_MAX_N_CSI1_PRMS (4)
#define CUPHY_CSI2_SIZE_MAP_BUFFER_SIZE_PER_CELL (4096)
#define CUPHY_MAX_NUM_CSI2_SIZE_MAPS_PER_CELL (16)

// CSI-RS
#define CUPHY_CSIRS_MAX_ANTENNA_PORTS (32)
#define CUPHY_CSIRS_MAX_KBAR_LBAR_LENGTH (16)
#define CUPHY_CSIRS_SYMBOL_LOCATION_TABLE_LENGTH (18)
#define CUPHY_CSIRS_MAX_KI_INDEX_LENGTH (6)
#define CUPHY_CSIRS_MAX_SEQ_INDEX_COUNT (8)
#define CUPHY_CSIRS_MAX_NUM_PARAMS (32) // Maximum number of CSI-RS parameters per cell for CSI-RS and PDSCH channels. Cell group maximum is scaled by number of cells.
                                        // The cell group limit is what is enforced.
#define CUPHY_CSIRS_MAX_NUM_UES (1024)  // Maximum number of UEs to process CSI-RS in the cell group.

// DTX
#define CUPHY_DTX_EN  (0x0F)
#define CUPHY_DTX_THRESHOLD_ADJ_SIMPLEX_DECODER (0.2f)
#define CUPHY_DTX_THRESHOLD_ADJ_RM_DECODER      (0.2f)

// Detection Status and Val
#define CUPHY_DET_EN           (0x30)
#define CUPHY_PUCCH_DET_EN     (0xC0)
#define CUPHY_FAPI_CRC_PASS    (0x1)
#define CUPHY_FAPI_CRC_FAILURE (0x2)
#define CUPHY_FAPI_DTX         (0x3)
#define CUPHY_FAPI_NO_DTX      (0x4)

// PUSCH noise regularizer
#define CUPHY_NOISE_REGULARIZER       (0.00000001f)
//#define CUPHY_NOISE_RATIO_LEGACYMMSE  (1.1f)
#define MAX_UINT16 0xFFFF
/// @endcond

/**
 * cuPHY error codes
 */
typedef enum
{
    CUPHY_STATUS_SUCCESS               = 0,  /*!< The API call returned with no errors.                                    */
    CUPHY_STATUS_INTERNAL_ERROR        = 1,  /*!< An unexpected, internal error occurred.                                  */
    CUPHY_STATUS_NOT_SUPPORTED         = 2,  /*!< The requested function is not currently supported.                       */
    CUPHY_STATUS_INVALID_ARGUMENT      = 3,  /*!< One or more of the arguments provided to the function was invalid.       */
    CUPHY_STATUS_ARCH_MISMATCH         = 4,  /*!< The requested operation is not supported on the current architecture.    */
    CUPHY_STATUS_ALLOC_FAILED          = 5,  /*!< A memory allocation failed.                                              */
    CUPHY_STATUS_SIZE_MISMATCH         = 6,  /*!< The size of the operands provided to the function do not match.          */
    CUPHY_STATUS_MEMCPY_ERROR          = 7,  /*!< An error occurred during a memcpy operation.                             */
    CUPHY_STATUS_INVALID_CONVERSION    = 8,  /*!< An invalid conversion operation was requested.                           */
    CUPHY_STATUS_UNSUPPORTED_TYPE      = 9,  /*!< An operation was requested on an unsupported type.                       */
    CUPHY_STATUS_UNSUPPORTED_LAYOUT    = 10, /*!< An operation was requested on an unsupported layout.                     */
    CUPHY_STATUS_UNSUPPORTED_RANK      = 11, /*!< An operation was requested on an unsupported rank.                       */
    CUPHY_STATUS_UNSUPPORTED_CONFIG    = 12, /*!< An operation was requested on an unsupported configuration.              */
    CUPHY_STATUS_UNSUPPORTED_ALIGNMENT = 13, /*!< One or more API arguments don't have the required alignment.             */
    CUPHY_STATUS_VALUE_OUT_OF_RANGE    = 14, /*!< Data conversion could not occur because an input value was out of range. */
    CUPHY_STATUS_REF_MISMATCH          = 15, /*!< Mismatch found when comparing to TV                                      */
    CUPHY_N_STATUS_CONFIGS             = 16  /*!< Number of unique cuphyStatus_t values.                                   */
} cuphyStatus_t;

// required for fmt:: lib
inline uint32_t format_as(const cuphyStatus_t status) { return static_cast<uint32_t>(status); }

/**
 * PUSCH Status Types
 */
typedef enum _cuphyPuschStatusType
{
    CUPHY_PUSCH_STATUS_SUCCESS_OR_UNTRACKED_ISSUE                        = CUPHY_STATUS_SUCCESS, /*!< Default value; reset on every cuPHY PUSCH setup */
    CUPHY_PUSCH_STATUS_UNSUPPORTED_MAX_ER_PER_CB                         = 1,                  /*!< Set if at least one UE in the cell group has per CB Er > PUSCH_MAX_ER_PER_CB_BITS. Special handling in cuPHY-CP. */
    CUPHY_PUSCH_STATUS_TBSIZE_MISMATCH                                   = 2,
    CUPHY_PUSCH_STATUS_NCBS_PERTB_OUT_OF_RANGE                           = 3,
    CUPHY_PUSCH_STATUS_NCBS_PERCELLGROUP_OUT_OF_RANGE                    = 4,
    CUPHY_PUSCH_STATUS_NTBS_PERCELLGROUP_OUT_OF_RANGE                    = 5,
    CUPHY_PUSCH_STATUS_NUEGRPS_PERCELLGROUP_OUT_OF_RANGE                 = 6,
    CUPHY_PUSCH_STATUS_NPRBS_OUT_OF_RANGE                                = 7,
    CUPHY_PUSCH_STATUS_HARQ_BUFFER_SIZE_NULLPTR                          = 8,
    CUPHY_PUSCH_STATUS_CHEST_SETUP_ERROR                                 = 9,
    CUPHY_PUSCH_STATUS_CFO_TA_SETUP_ERROR                                = 10,
    CUPHY_PUSCH_STATUS_CHEQ_COEF_COMP_SETUP_ERROR                        = 11,
    CUPHY_PUSCH_STATUS_SOFT_DEMAP_SETUP_ERROR                            = 12,
    CUPHY_PUSCH_STATUS_SOFT_DEMAP_EARLY_HARQ_SETUP_ERROR                 = 13,
    CUPHY_PUSCH_STATUS_IDFT_SETUP_ERROR                                  = 14,
    CUPHY_PUSCH_STATUS_SOFT_DEMAP_AFTER_DFT_SETUP_ERROR                  = 15,
    CUPHY_PUSCH_STATUS_IDFT_EARLY_HARQ_SETUP_ERROR                       = 16,
    CUPHY_PUSCH_STATUS_SOFT_DEMAP_AFTER_DFT_EARLY_HARQ_SETUP_ERROR       = 17,
    CUPHY_PUSCH_STATUS_RATE_MATCH_SETUP_ERROR                            = 18,
    CUPHY_PUSCH_STATUS_CRC_DECODE_SETUP_ERROR                            = 19,
    CUPHY_PUSCH_STATUS_RSSI_SETUP_ERROR                                  = 20,
    CUPHY_PUSCH_STATUS_RSRP_SETUP_ERROR                                  = 21,
    CUPHY_PUSCH_STATUS_NOISE_INTF_EST_SETUP_ERROR                        = 22,
    CUPHY_PUSCH_STATUS_NOISE_INTF_EST_EARLY_HARQ_SETUP_ERROR             = 23,
    CUPHY_PUSCH_STATUS_UCI_SEG_LLR0_SETUP_ERROR                          = 24,
    CUPHY_PUSCH_STATUS_SIMPLEX_DECODE_SETUP_ERROR                        = 25,
    CUPHY_PUSCH_STATUS_RM_DECODE_SETUP_ERROR                             = 26,
    CUPHY_PUSCH_STATUS_COMP_CW_TREE_TYPE_SETUP_ERROR                     = 27,
    CUPHY_PUSCH_STATUS_POLAR_SEG_RATE_MATCH_SETUP_ERROR                  = 28,
    CUPHY_PUSCH_STATUS_POLAR_DECODE_SETUP_ERROR                          = 29,
    CUPHY_PUSCH_STATUS_UCI_CSI2_CTRL_SETUP_ERROR                         = 30,
    CUPHY_PUSCH_STATUS_UCI_SEG_LLR2_SETUP_ERROR                          = 31,
    CUPHY_PUSCH_STATUS_SIMPLEX_DECODE_CSI2_SETUP_ERROR                   = 32,
    CUPHY_PUSCH_STATUS_RM_DECODE_CSI2_SETUP_ERROR                        = 33,
    CUPHY_PUSCH_STATUS_COMP_CW_TREE_TYPE_CSI2_SETUP_ERROR                = 34,
    CUPHY_PUSCH_STATUS_POLAR_SEG_RATE_MATCH_CSI2_SETUP_ERROR             = 35,
    CUPHY_PUSCH_STATUS_POLAR_DECODE_CSI2_SETUP_ERROR                     = 36,
    CUPHY_PUSCH_STATUS_SINR_EARLY_HARQ_SETUP_ERROR                       = 37,
    CUPHY_PUSCH_STATUS_RSSI_EARLY_HARQ_SETUP_ERROR                       = 38,
    // More types can be added as needed
    CUPHY_MAX_PUSCH_STATUS_TYPES

} cuphyPuschStatusType_t;

/**
 * PUSCH Output Status
 */
typedef struct _cuphyPuschStatusOut
{
    cuphyPuschStatusType_t status; /* cuPHY PUSCH status after setup call. Currently used to highlight if CUPHY_PUSCH_STATUS_UNSUPPORTED_MAX_ER_PER_CB occured. */
    uint16_t cellPrmStatIdx;
    uint16_t ueIdx;
} cuphyPuschStatusOut_t;

/**
 * PUSCH Full-slot Proc Type
 */
typedef enum _cuphyPuschFullSlotProcMode
{
  PUSCH_LEGACY_FULL_SLOT_PROC             = 0,
  PUSCH_FRONT_LOADED_DMRS_FULL_SLOT_PROC  = 1,
  CUPHY_MAX_PUSCH_FULL_SLOT_PROC_MODES
} cuphyPuschFullSlotProcMode_t;

/**
 * PUSCH Channel Estimation Algorithm Selection
 */
typedef enum _cuphyPuschChEstAlgoType
{
  PUSCH_CH_EST_ALGO_TYPE_LEGACY_MMSE                    = 0,
  PUSCH_CH_EST_ALGO_TYPE_MULTISTAGE_MMSE_WITH_DELAY_EST = 1,
  PUSCH_CH_EST_ALGO_TYPE_RKHS                           = 2,
  PUSCH_CH_EST_ALGO_TYPE_LS_ONLY                        = 3
} cuphyPuschChEstAlgoType_t;


/**
 * SRS Channel Estimation Algorithm Selection
 */
typedef enum _cuphySrsChEstAlgoType
{
  SRS_CH_EST_ALGO_TYPE_MMSE = 0,
  SRS_CH_EST_ALGO_TYPE_RKHS = 1
} cuphySrsChEstAlgoType_t;

/**
 * PUSCH Wait Kernel Status
 */
typedef enum _cuphyPuschWaitKernelStatus
{
  PUSCH_RX_WAIT_KERNEL_STATUS_DONE    = 0,
  PUSCH_RX_WAIT_KERNEL_STATUS_TIMEOUT = 1
} cuphyPuschWaitKernelStatus_t;

/**
 * PUSCH Execution Path: full-slot/sub-slot
 */
typedef enum _cuphyPuschExecutionPath
{
    CUPHY_PUSCH_FULL_SLOT_PATH = 0, /*!< Used for kernels in full-slot processing pipeline*/
    CUPHY_PUSCH_SUB_SLOT_PATH  = 1, /*!< Used for kernels in sub-slor processing, such as in early-HARP processing */
    CUPHY_MAX_PUSCH_EXECUTION_PATHS
} cuphyPuschExecutionPath_t;

/**
 * PUSCH Noise Estimation DMRS Symbol Index
 */
typedef enum _cuphyPuschNoiseIntfEstDmrsSymbolIdx
{
    CUPHY_PUSCH_NOISE_EST_DMRS_ADDITIONAL_POS_0 = 0, /*!< To process the first DMRS symbol for noise estimation*/
    CUPHY_PUSCH_NOISE_EST_DMRS_ADDITIONAL_POS_1 = 1, /*!< To process the second DMRS symbol for noise estimation*/
    CUPHY_PUSCH_NOISE_EST_DMRS_ADDITIONAL_POS_2 = 2, /*!< To process the third DMRS symbol for noise estimation*/
    CUPHY_PUSCH_NOISE_EST_DMRS_ADDITIONAL_POS_3 = 3, /*!< To process the fourth DMRS symbol for noise estimation*/
    CUPHY_PUSCH_NOISE_EST_DMRS_FULL_SLOT /*!< To process all the DMRS symbols for noise estimation*/
} cuphyPuschNoiseIntfEstDmrsSymbolIdx_t;

/**
 * PUSCH RSRP Estimation DMRS Symbol Index
 */
typedef enum _cuphyPuschRsrpEstDmrsSymbolIdx
{
    CUPHY_PUSCH_RSRP_EST_FIRST_DMRS                        = 0, /*!< To process the first DMRS symbol*/
    CUPHY_PUSCH_RSRP_EST_FULL_SLOT_DMRS_WITHOUT_FIRST_DMRS = 1, /*!< To process all the DMRS symbols except the firstm DMRS symbol*/
    CUPHY_PUSCH_RSRP_EST_FULL_SLOT_DMRS                         /*!< To process all the DMRS symbols*/
} cuphyPuschRsrpEstDmrsSymbolIdx_t;

/**
 * PUSCH RSSI Estimation DMRS Symbol Index
 */
typedef enum _cuphyPuschRssiEstDmrsSymbolIdx
{
    CUPHY_PUSCH_RSSI_EST_FIRST_DMRS                        = 0, /*!< To process the first DMRS symbol*/
    CUPHY_PUSCH_RSSI_EST_FULL_SLOT_DMRS_WITHOUT_FIRST_DMRS = 1, /*!< To process all the DMRS symbols except the firstm DMRS symbol*/
    CUPHY_PUSCH_RSSI_EST_FULL_SLOT_DMRS                         /*!< To process all the DMRS symbols*/
} cuphyPuschRssiEstDmrsSymbolIdx_t;

/**
 * PUCCH Status Types
 */
typedef enum _cuphyPucchStatusType
{
    CUPHY_PUCCH_STATUS_SUCCESS_OR_UNTRACKED_ISSUE        = CUPHY_STATUS_SUCCESS,
    CUPHY_PUCCH_STATUS_F0_SETUP_ERROR                    = 0x1,
    CUPHY_PUCCH_STATUS_F1_SETUP_ERROR                    = 0x2,
    CUPHY_PUCCH_STATUS_F2_SETUP_ERROR                    = 0x3,
    CUPHY_PUCCH_STATUS_F3_SETUP_ERROR                    = 0x4,
    CUPHY_PUCCH_STATUS_F234_UCI_SEG_SETUP_ERROR          = 0x5,
    CUPHY_PUCCH_STATUS_RM_DECODE_SETUP_ERROR             = 0x6,
    CUPHY_PUCCH_STATUS_COMP_CW_TREE_TYPE_SETUP_ERROR     = 0x7,
    CUPHY_PUCCH_STATUS_POLAR_SEG_RATE_MATCH_SETUP_ERROR  = 0x8,
    CUPHY_PUCCH_STATUS_POLAR_DECODE_SETUP_ERROR          = 0x9,

    CUPHY_MAX_PUCCH_STATUS_TYPES

} cuphyPucchStatusType_t;

/**
 * PUCCH Output Status
 */
typedef struct _cuphyPucchStatusOut
{
    cuphyPucchStatusType_t status;
    uint16_t cellPrmStatIdx;
    uint16_t ueIdx;
} cuphyPucchStatusOut_t;

/**
 * PRACH Status Types
 */
typedef enum _cuphyPrachStatusType
{
    CUPHY_PRACH_STATUS_SUCCESS_OR_UNTRACKED_ISSUE        = CUPHY_STATUS_SUCCESS,
    CUPHY_PRACH_STATUS_GRAPH_UPDATE_ERROR                = 0x1,

    CUPHY_MAX_PRACH_STATUS_TYPES

} cuphyPrachStatusType_t;

/**
 * PRACH Output Status
 */
typedef struct _cuphyPrachStatusOut
{
    cuphyPrachStatusType_t status;
    uint16_t cellPrmStatIdx;
    uint16_t ueIdx;
} cuphyPrachStatusOut_t;

/**
 * SRS Status Types
 */
typedef enum _cuphySrsStatusType
{
    CUPHY_SRS_STATUS_SUCCESS_OR_UNTRACKED_ISSUE        = CUPHY_STATUS_SUCCESS,
    CUPHY_SRS_STATUS_CHEST_SETUP_ERROR                 = 0x1,

    CUPHY_MAX_SRS_STATUS_TYPES

} cuphySrsStatusType_t;

/**
 * SRS Output Status
 */
typedef struct _cuphySrsStatusOut
{
    cuphySrsStatusType_t status;
    uint16_t cellPrmStatIdx;
    uint16_t ueIdx;
} cuphySrsStatusOut_t;

/**
 * BFW Status Types
 */
typedef enum _cuphyBfwStatusType
{
    CUPHY_BFW_STATUS_SUCCESS_OR_UNTRACKED_ISSUE        = CUPHY_STATUS_SUCCESS,
    CUPHY_BFW_STATUS_COEF_COMP_SETUP_ERROR             = 0x1,

    CUPHY_MAX_BFW_STATUS_TYPES

} cuphyBfwStatusType_t;

/**
 * BFW Output Status
 */
typedef struct _cuphyBfwStatusOut
{
    cuphyBfwStatusType_t status;
    uint16_t cellPrmStatIdx;
    uint16_t ueIdx;
} cuphyBfwStatusOut_t;

/// Transport block parameters
typedef struct tb_pars
{
    // MIMO
    uint32_t numLayers;
    uint64_t layerMap; //[MAXLAYERS];    This field is a bit map for now

    // Resource allocation
    uint32_t startPrb;
    uint32_t numPrb;
    uint32_t startSym;
    uint32_t numSym;
    uint32_t dataScramId;
    // Back-end parameters
    // FIXME Cannot deprecate mcsTableIndex and mcsIndex; used in cuPHY-CP
    uint32_t mcsTableIndex; // deprecated, to be removed after 22-1 ED1
    uint32_t mcsIndex; // deprecated, to be removed after 22-1 ED1
    uint32_t rv;

    // Parameters to enable MCS greater than 28
    uint16_t targetCodeRate;//Assuming the code rate is x/1024.0 where x contains a single digit after decimal point, then targetCodeRate = static_cast<uint16_t>(x * 10) = static_cast<uint16_t>(codeRate * 1024 * 10)
    uint8_t  qamModOrder;//Value: 2,4,6,8 if transform precoding is disabled; 1,2,4,6,8 if transform precoding is enabled

    // DMRS parameters
    uint32_t dmrsType;
    uint32_t dmrsAddlPosition;
    uint32_t dmrsMaxLength;
    uint32_t dmrsScramId;
    uint32_t dmrsEnergy;

    uint32_t dmrsCfg;
    uint32_t nRnti;

    uint32_t nPortIndex; //up to 8 layers encoded in an uint32_t, in groups of 4 bits
    uint32_t nSCID;
    uint32_t userGroupIndex;
    uint32_t nBBULayers;
} tb_pars;

/**
 * CSI Part 2 map parameters
 */
typedef struct _cuphyCsi2MapPrm
{
    uint16_t  csi2MapSize;      /*!< size of this csi2 map */
    uint32_t  csi2MapStartIdx;  /*!< location of this csi2 map within the large csi2Map buffer */
}cuphyCsi2MapPrm_t;

/**
 * PUSCH Cell static parameters
 */
typedef struct _cuphyPuschCellStatPrm
{
    uint8_t nCsirsPorts;      /*!< number of CSI RS ports (Candidate values = 2,4,8,12,16,24,32) */
    uint8_t N1;               /*!< number of antenna ports in first dimension (horizontal) */
    uint8_t N2;               /*!< number of antenna ports in second dimension (vertical) */
    uint8_t csiReportingBand; /*!< 0: wideband, 1: subband, 2: both */
    uint8_t codebookType;     /*!< 0: type1SinglePanel, 1: type1MultiPanel, 2: type2, 3: type2PortSelection */
    uint8_t codebookMode;     /*!< 0 (N/A) or 1 or 2 (non zero value only needed for type 1 codebook) */
    uint8_t isCqi;            /*!< whether CSI-P2 contains CQI (wideband or subband): 0 or 1 */
    uint8_t isLi;             /*!< whether CSI-P2 contains LI : 0 or 1 */

    // FAPIv3 method for computing CSI-P2 size:
    uint16_t            nCsi2Maps;         /*!< number of maps used to compute CSI-P2 size */
    uint16_t*           pCsi2MapBuffer;    /*!< pointer to GPU buffer containing all CSI-P2 maps */
    cuphyCsi2MapPrm_t*  pCsi2MapPrm;       /*!< pointer to GPU buffer contaiing CSI-P2 map parameters */
} cuphyPuschCellStatPrm_t;

typedef struct gnb_pars
{
    uint32_t fc;
    uint32_t mu;
    uint32_t nRx;
    uint32_t nPrb;
    uint32_t cellId;
    uint32_t slotNumber; // @todo: slotNumber to be removed since its a slot level parameter
    uint32_t Nf;
    uint32_t Nt;
    uint32_t df;
    uint32_t dt;
    uint32_t numBsAnt;
    uint32_t numBbuLayers;
    uint32_t numTb;
    uint32_t ldpcEarlyTermination;
    uint32_t ldpcAlgoIndex;
    uint32_t ldpcFlags;
    uint32_t ldpcUseHalf;
    uint32_t ldpcKernelLaunch;
    uint32_t slotType; // 0 UL, 1 DL
    uint32_t nUserGroups;
} gnb_pars;


// Per UCI PUCCH parameters
/**
 * PUCCH CSI-P1 & P2 parameters
 * SCF FAPI Table 3-95, UCI Part1 to Part2 correspondence
 */
typedef struct _cuphyPucchUciP1P2Crpd
{
    uint16_t numPart2s;/*!< Max number of UCI part2 that could be included in the CSI report.
                            Value: 0 -> 100. Currently assume numPart2s <= 1 */

    // to do: other fields for each part2 in Table 3-95
} cuphyPucchUciP1P2Crpd_t;

/**
 * PUCCH Per UCI PUCCH parameters
 */
typedef struct _cuphyPucchUciPrm
{
    uint16_t cellPrmDynIdx;  /*!< Index into parent cell-dynamic parameters (pCellPrms in cuphyPucchCellGrpDynPrm_t) of the UE transmitting the UCI */
    uint16_t cellPrmStatIdx; /*!< Index into parent cell-static parameters (pCellPrms in cuphyPucchCellGrpDynPrm_t) of the UE transmitting the UCI */
    uint16_t uciOutputIdx;
    uint8_t  formatType;
    uint16_t rnti;
    uint16_t bwpStart;              /*!< Bandwidth part size [3GPP TS 38.213 [4], sec12].
                                         Number of contiguous PRBs allocated to the BWP \n
                                         Value: 1->275*/
    uint8_t  multiSlotTxIndicator;
    uint8_t  pi2Bpsk;
    uint16_t startPrb;
    uint8_t  prbSize;
    uint8_t  startSym;
    uint8_t  nSym;
    uint8_t  freqHopFlag;
    uint16_t secondHopPrb;
    uint8_t  groupHopFlag;
    uint8_t  sequenceHopFlag;
    uint16_t initialCyclicShift;
    uint8_t  timeDomainOccIdx;
    uint8_t  srFlag;
    uint16_t bitLenSr;
    uint16_t bitLenHarq;
    uint16_t bitLenCsiPart1;
    uint8_t  AddDmrsFlag;
    uint16_t dataScramblingId;
    uint16_t DmrsScramblingId;
    uint8_t  maxCodeRate;

    uint16_t nBitsCsi2;
    uint8_t  rankBitOffset;
    uint8_t  nRanksBits;
    // External DTX detection threshold
    float DTXthreshold;

    // Format 3 UCI Part1 to Part2 correspondence
    // Refer to SCF FAPI Table 3-95
    cuphyPucchUciP1P2Crpd_t uciP1P2Crpd_t;

    uint16_t nUplinkStreams;
} cuphyPucchUciPrm_t;

/**
 * PUCCH per-cell static parameters, which shall remain constant throughout the pipeline lifecycle
 */
typedef struct _cuphyPucchCellStatPrm
{
    uint8_t nCsirsPorts;      /*!< number of CSI RS ports (Candidate values = 2,4,8,12,16,24,32) */
    uint8_t N1;               /*!< number of antenna ports in first dimension (horizontal) */
    uint8_t N2;               /*!< number of antenna ports in second dimension (vertical) */
    uint8_t csiReportingBand; /*!< 0: wideband, 1: subband, 2: both */
    uint8_t codebookType;     /*!< 0: type1SinglePanel, 1: type1MultiPanel, 2: type2, 3: type2PortSelection */
    uint8_t codebookMode;     /*!< 0 (N/A) or 1 or 2 (non zero value only needed for type 1 codebook) */
    uint8_t isCqi;            /*!< whether CSI-P2 contains CQI (wideband or subband): 0 or 1 */
    uint8_t isLi;             /*!< whether CSI-P2 contains LI : 0 or 1 */
} cuphyPucchCellStatPrm_t;

/**
 * UCI output structure for PUCCH formats 0 and 1
 */
typedef struct _cuphyPucchF0F1UciOut
{
    uint8_t SRindication; /*! Indicates if an SR was detected */

    /*! SRconfidenceLevel in SCF FAPI, Table "SR PDU for format 0 or 1", 0 stands for "Good" and 1 stands for "Bad"
    * When detection result is DTX, SRconfidenceLevel is always set to 1 (bad)
    * When SR is detected, SRconfidenceLevel is 0 (good) or 1 (bad) based on detection result reliability
    */
    uint8_t SRconfidenceLevel;

    uint8_t NumHarq; /*! Number of HARQ bits present in UCI */

    /*! HarqconfidenceLevel in SCF FAPI, Table "HARQ PDU for format 0 or 1", 0 stands for "Good" and 1 stands for "Bad"
    * When detection result is DTX, HarqconfidenceLevel is always set to 1 (bad)
    * When HARQ-ACK bits are detected, HarqconfidenceLevel is 0 (good) or 1 (bad) based on detection result reliability
    */
    uint8_t HarqconfidenceLevel;

    uint8_t HarqValues[2]; /*! Indicates result on HARQ data */

    float   taEstMicroSec; /*! Timing advance measurement */

    float   SinrDB; /*! SINR metric in dB value for channel quality */
    float   InterfDB; /*! Interference plus noise power in dB value */

    float   RSSI; /*! RSSI reported in dB */
    float   RSRP; /*! RSRP reported in dB */
} cuphyPucchF0F1UciOut_t;


/**
 * parameters for computing CSI2 payload size
 */
typedef struct _cuphyCalcCsi2SizePrm
{
    uint8_t   nPart1Prms;                          // number of parameters in CSI-P1 payload. 1-4
    uint8_t   prmSizes[CUPHY_MAX_N_CSI1_PRMS];     // size of parameters
    uint16_t  prmOffsets[CUPHY_MAX_N_CSI1_PRMS];   // offset of parameters in CSI-P1 payload
    uint16_t  csi2sizeMapIdx;                      // index of map from CSI-P1 parameters to CSI-P2 size

}cuphyCalcCsi2SizePrm_t;

/**
 * Offsets for locating PUSCH outputs
 */
typedef struct _cuphyUciOnPuschOutOffsets
{
    uint8_t  isEarlyHarq;               /*! Flag when set indicates that the HarqDetectionStatusOffset, harqPayloadByteOffset, harqCrcFlagOffset
                                            may be used to read early-HARQ results from result buffers cuphyPuschDataOut_t.HarqDetectionStatus,
                                            cuphyPuschDataOut_t.pUciPayloads, cuphyPuschDataOut_t.pUciCrcFlags respectively */
    uint16_t HarqDetectionStatusOffset;
    uint16_t CsiP1DetectionStatusOffset;
    uint16_t CsiP2DetectionStatusOffset;

    uint32_t harqPayloadByteOffset;
    uint16_t harqCrcFlagOffset;

    uint32_t csi1PayloadByteOffset;
    uint16_t csi1CrcFlagOffset;

    uint16_t numCsi2BitsOffset;
    uint32_t csi2PayloadByteOffset;
    uint16_t csi2CrcFlagOffset;
} cuphyUciOnPuschOutOffsets_t;

/**
 * Offsets for locating PUCCH outputs
 */
typedef struct _cuphyPucchF234OutOffsets
{
    uint16_t dtxFlagOffset;
    uint16_t dtxF2RMFlagOffset;
    uint16_t RSSIoffset;
    uint16_t snrOffset;
    uint16_t RSRPoffset;
    uint16_t InterfOffset;
    uint16_t taEstOffset;

    uint16_t HarqDetectionStatusOffset;
    uint16_t CsiP1DetectionStatusOffset;
    uint16_t CsiP2DetectionStatusOffset;

    uint32_t uciSeg1PayloadByteOffset;
    uint16_t uciSeg1CrcFlagOffset;

    uint32_t srPayloadByteOffset;
    uint16_t srCrcFlagOffset;

    uint32_t harqPayloadByteOffset;
    uint16_t harqCrcFlagOffset;

    uint32_t csi1PayloadByteOffset;
    uint16_t csi1CrcFlagOffset;

    uint16_t numCsi2BitsOffset;
    uint32_t csi2PayloadByteOffset;
    uint16_t csi2CrcFlagOffset;
} cuphyPucchF234OutOffsets_t;


///  Parameters for polar encoded UCI segment
typedef struct _cuphyPolarUciSegPrm
{
    __half* pUciSegLLRs;  /// pointer to LLRs of rate-matched bits
    uint8_t exitFlag;     /// option to exit Kernel. 0 or 1. (for CSI-P2 support)

    uint8_t nCbs;           /// number of codeblocks UCI segment split into. 1 or 2.
#ifdef ENABLE_64C
    uint16_t childCbIdxs[2]; /// indicies of child codeblocks
#else
    uint8_t childCbIdxs[2]; /// indicies of child codeblocks
#endif
    uint8_t zeroInsertFlag; /// Flag, indicates if zero inserted at start of first cb.

    uint32_t E_seg; /// number of rate matched bits for segment

    uint8_t  nCrcBits; /// number of crc bits per codeword. 6 or 12.
    uint32_t E_cw;     /// number of rate matched bits per codeword
    uint16_t K_cw;     /// number of input bits per codeword (info + CRC + possible zero insertion)
    uint16_t N_cw;     /// codeword size
    uint8_t  n_cw;     /// n_cb = log2(N_cb)
} cuphyPolarUciSegPrm_t;

/// Parameters for polar codewords
typedef struct _cuphyPolarCwPrm
{
    __half*   pCwTreeLLRs;
    __half*   pCwLLRs;
    uint32_t* pCbEst;

    uint8_t exitFlag;        /// option to exit Kernel. 0 or 1. (for CSI-P2 support)

    uint16_t N_cw;           /// codeword size
    uint8_t  nCrcBits;       /// number of crc bits per codeword. 6 or 12.
    uint16_t A_cw;           /// number of payload bits per codeword. (does not include CRC).
    uint8_t* pCwTreeTypes;   /// pointer to cw Tree types
    uint8_t* pCrcStatus;     /// pointer to CRC status
    uint8_t* pCrcStatus1;    /// pointer to CRC status extra placeholder
    uint8_t  en_CrcStatus;   /// enable CRC Status in polar_decoder.cu

    uint8_t   nCbsInUciSeg;      /// number of codeblocks in parent UCI seg. 1 or 2
    uint8_t   cbIdxWithinUciSeg; /// index of clodeblock within parent UCI seg. 0 or 1
    uint8_t   zeroInsertFlag;    /// flag, indicates if zero inserted at start of first codeblock. 0 or 1
    uint32_t* pUciSegEst;        /// pointer to estimated UCI segment (GPU).
} cuphyPolarCwPrm_t;

/// Parameters for simplex codewords
typedef struct _cuphySimplexCwPrm
{
    uint8_t   exitFlag;    /// option to exit Kernel. 0 or 1. (for CSI-P2 support)
    uint8_t   K;           /// number of information bits. 1 or 2.
    uint8_t   nBitsPerQam; /// number of bits per qam.
    uint32_t  E;           /// number of rate-matched bits.
    __half*   d_LLRs;      /// pointer to LLRs of rate-match bits
    float*    d_noiseVar;  /// pointer to noise var
    float     DTXthreshold;/// DTX threshold
    uint32_t* d_cbEst;     /// pointer to codeblock estimate
    uint8_t*  d_DTXStatus; /// pointer to DTX detection status
    uint8_t   en_DTXest;   /// enable DTXest in simplex_decoder.cu.
} cuphySimplexCwPrm_t;

/// Parameters for Reed Muller codewords
typedef struct _cuphyRmCwPrm
{
    uint8_t   exitFlag;    /// option to exit Kernel. 0 or 1. (for CSI-P2 support)
    uint8_t   K;           /// number of information bits. 1 or 2.
    uint32_t  E;           /// number of rate-matched bits.
    uint32_t  Qm;          /// modulation order per TB: [2, 4, 6, 8]
    __half*   d_LLRs;      /// pointer to LLRs of rate-match bits
    float*    d_noiseVar;  /// pointer to noise var
    float     DTXthreshold;/// DTX threshold
    uint32_t* d_cbEst;     /// pointer to codeblock estimate
    uint8_t*  d_DTXStatus; /// pointer to DTX detection status
    uint8_t*  d_DTXStatus1;/// pointer to DTX detection status extra placeholder
    uint8_t*  d_DTXStatus2;/// pointer to DTX detection status extra placeholder
    uint8_t   en_DTXest;   /// enable DTXest in rm_decoder.cu.
} cuphyRmCwPrm_t;

// Note: CUDA_R/C values are defined in CUDA library_types.h header

/**
 * cuPHY data types
 */
typedef enum
{
    CUPHY_VOID  = -1,         /*!< uninitialized type                       */
    CUPHY_BIT   = 20,         /*!< 1-bit value                              */
    CUPHY_R_8I  = CUDA_R_8I,  /*!< 8-bit signed integer real values         */
    CUPHY_C_8I  = CUDA_C_8I,  /*!< 8-bit signed integer complex values      */
    CUPHY_R_8U  = CUDA_R_8U,  /*!< 8-bit unsigned integer real values       */
    CUPHY_C_8U  = CUDA_C_8U,  /*!< 8-bit unsigned integer complex values    */
    CUPHY_R_16I = 21,         /*!< 16-bit signed integer real values        */
    CUPHY_C_16I = 22,         /*!< 16-bit signed integer complex values     */
    CUPHY_R_16U = 23,         /*!< 16-bit unsigned integer real values      */
    CUPHY_C_16U = 24,         /*!< 16-bit unsigned integer complex values   */
    CUPHY_R_32I = CUDA_R_32I, /*!< 32-bit signed integer real values        */
    CUPHY_C_32I = CUDA_C_32I, /*!< 32-bit signed integer complex values     */
    CUPHY_R_32U = CUDA_R_32U, /*!< 32-bit unsigned integer real values      */
    CUPHY_C_32U = CUDA_C_32U, /*!< 32-bit unsigned integer complex values   */
    CUPHY_R_16F = CUDA_R_16F, /*!< half precision (16-bit) real values      */
    CUPHY_C_16F = CUDA_C_16F, /*!< half precision (16-bit) complex values   */
    CUPHY_R_32F = CUDA_R_32F, /*!< single precision (32-bit) real values    */
    CUPHY_C_32F = CUDA_C_32F, /*!< single precision (32-bit) complex values */
    CUPHY_R_64F = CUDA_R_64F, /*!< double precision (64-bit) real values    */
    CUPHY_C_64F = CUDA_C_64F  /*!< double precision (64-bit) complex values */
} cuphyDataType_t;

/**
 * cuPHY implementation of variant type
 */
typedef struct
{
    int type;
    union
    {
        unsigned int    b1;   /*!< CUPHY_BIT   (1-bit value)                              */
        signed char     r8i;  /*!< CUPHY_R_8I  (8-bit signed integer real values)         */
        char2           c8i;  /*!< CUPHY_C_8I  (8-bit signed integer complex values)      */
        unsigned char   r8u;  /*!< CUPHY_R_8U  (8-bit unsigned integer real values)       */
        uchar2          c8u;  /*!< CUPHY_C_8U  (8-bit unsigned integer complex values)    */
        short           r16i; /*!< CUPHY_R_16I (16-bit signed integer real values)        */
        short2          c16i; /*!< CUPHY_C_16I (16-bit signed integer complex values)     */
        unsigned short  r16u; /*!< CUPHY_R_16U (16-bit unsigned integer real values)      */
        ushort2         c16u; /*!< CUPHY_C_16U (16-bit unsigned integer complex values)   */
        int             r32i; /*!< CUPHY_R_32I (32-bit signed integer real values)        */
        int2            c32i; /*!< CUPHY_C_32I (32-bit signed integer complex values)     */
        unsigned int    r32u; /*!< CUPHY_R_32U (32-bit unsigned integer real values)      */
        uint2           c32u; /*!< CUPHY_C_32U (32-bit unsigned integer complex values)   */
        __half_raw      r16f; /*!< CUPHY_R_16F (half precision (16-bit) real values)      */
        __half2_raw     c16f; /*!< CUPHY_C_16F (half precision (16-bit) complex values)   */
        float           r32f; /*!< CUPHY_R_32F (single precision (32-bit) real values)    */
        cuComplex       c32f; /*!< CUPHY_C_32F (single precision (32-bit) complex values) */
        double          r64f; /*!< CUPHY_R_64F (double precision (64-bit) real values)    */
        cuDoubleComplex c64f; /*!< CUPHY_C_64F (double precision (64-bit) complex values) */
    } value;
} cuphyVariant_t;

/**
 * cuPHY element-wise operations
 */
typedef enum
{
    CUPHY_ELEMWISE_ADD,    /*!< Add elements                                   */
    CUPHY_ELEMWISE_MUL,    /*!< Multiply elements                              */
    CUPHY_ELEMWISE_MIN,    /*!< Select the minimum of two elements             */
    CUPHY_ELEMWISE_MAX,    /*!< Select the maximum of two elements             */
    CUPHY_ELEMWISE_ABS,    /*!< Determine the absolute value of a single input */
    CUPHY_ELEMWISE_BIT_XOR /*!< Perform bitwise XOR (CUPHY_BIT tensors only)   */
} cuphyElementWiseOp_t;

/**
 * cuPHY reduction operations
 */
typedef enum
{
    CUPHY_REDUCTION_SUM, /*!< Add elements               */
    CUPHY_REDUCTION_MIN, /*!< Select the minimum element */
    CUPHY_REDUCTION_MAX, /*!< Select the maximum element */
} cuphyReductionOp_t;

/**
 * \defgroup CUPHY_ERROR Error Handling
 *
 * This section describes the error handling functions of the cuPHY
 * application programming interface.
 *
 * @{
 */

/******************************************************************/ /**
 * \brief Returns the description string for an error code
 *
 * Returns the description string for an error code.  If the error
 * code is not recognized, "Unknown status code" is returned.
 *
 * \param status - Status code for desired string
 *
 * \return
 * \p char* pointer to a NULL-terminated string
 *
 * \sa ::cuphyGetErrorName, ::cuphyStatus_t
 */
const char* CUPHYWINAPI cuphyGetErrorString(cuphyStatus_t status);

/******************************************************************/ /**
 * \brief Returns a string version of an error code enumeration value
 *
 * Returns a string version of an error code.  If the error
 * code is not recognized, "CUPHY_UNKNOWN_STATUS" is returned.
 *
 * \param status - Status code for desired string
 *
 * \return
 * \p char* pointer to a NULL-terminated string
 *
 * \sa ::cuphyGetErrorString, ::cuphyStatus_t
 */
const char* CUPHYWINAPI cuphyGetErrorName(cuphyStatus_t status);

/** @} */ /* END CUPHY_ERROR */

/// @cond TODO bypassing doxygen on macro'd constants
/**
 * Maximum supported number of tensor dimensions
 */
#define CUPHY_TENSOR_N_DIM_1 (1)
#define CUPHY_TENSOR_N_DIM_2 (2)
#define CUPHY_TENSOR_N_DIM_3 (3)
#define CUPHY_TENSOR_N_DIM_4 (4)
#define CUPHY_TENSOR_N_DIM_5 (5)
#define CUPHY_DIM_MAX CUPHY_TENSOR_N_DIM_5


/* cuphySetTensorDescriptor() flags */
/* Use strides if provided, otherwise TIGHT                */
#define CUPHY_TENSOR_ALIGN_DEFAULT 0x00
/* Pack tightly, regardless of stride values               */
#define CUPHY_TENSOR_ALIGN_TIGHT 0x01
/* Align 2nd dimension for coalesced I/O (ignore strides)  */
#define CUPHY_TENSOR_ALIGN_COALESCE 0x02

/* QAM levels - set value set to log2(QAM) value */
#define CUPHY_QAM_2 (1)
#define CUPHY_QAM_4 (2)
#define CUPHY_QAM_16 (4)
#define CUPHY_QAM_64 (6)
#define CUPHY_QAM_256 (8)

/* # of subcarriers/tones per PRB */
#define CUPHY_N_TONES_PER_PRB (12)

/* DMRS configurations supported */
#define CUPHY_DMRS_CFG0 (0) // 1 layer : DMRS grid 0   ; fOCC = [+1, +1]          ; 1 DMRS symbol
#define CUPHY_DMRS_CFG1 (1) // 2 layers: DMRS grids 0,1; fOCC = [+1, +1]          ; 1 DMRS symbol
#define CUPHY_DMRS_CFG2 (2) // 4 layers: DMRS grids 0,1; fOCC = [+1, +1], [+1, -1]; 1 DMRS symbol
#define CUPHY_DMRS_CFG3 (3) // 8 layers: DMRS grids 0,1; fOCC/tOCC = [+1, +1], [+1, -1]; 4 DMRS symbols

/* Maximum number of downlink layers per transport block (TB) */
#define MAX_DL_LAYERS_PER_TB 4

/* Maximum number of downlink layers */
#define MAX_DL_LAYERS 32

/* Maximum number of downlink ports */
#define MAX_DL_PORTS 32

/* Maximum size of Resource Block Mask for Resource Allocation Type 0 (with 8 RBs per byte) */
#define MAX_RBMASK_UINT32_ELEMENTS 9 // ceil(273 RBs/32)
#define MAX_RBMASK_BYTE_SIZE 36 // ceil(273 RBs/32 bits)*4 bytes per uint32_t

/* LDPC decoder flags */
/* Default operation flag                        */
#define CUPHY_LDPC_DECODE_DEFAULT (0)
/* Use early termination (currently unsupported) */
#define CUPHY_LDPC_DECODE_EARLY_TERM (0x01)
/* When possible, choose an LDPC decoder algorithm that optimizes      *
 * throughput instead of latency.                                      *
 * For some LDPC decoder configurations, and on some GPU               *
 * architectures, kernel implementations exist that have longer        *
 * latency but increased overall throughput (by, for example, decoding *
 * two codewords at a time). When the algorithm index is zero and this *
 * flag is NOT set, the chosen kernel will minimize latency. When the  *
 * algorithm index is 0 and this flag IS set, a higher throughput      *
 * kernel will be chosen (if one is available).                        */
#define CUPHY_LDPC_DECODE_CHOOSE_THROUGHPUT (0x2)
/* Write soft output values following the last iteration */
#define CUPHY_LDPC_DECODE_WRITE_SOFT_OUTPUTS (0x04)
/// @endcond

/**
 * \defgroup CUPHY_CONTEXT Library context
 *
 * This section describes the context functions of the cuPHY application
 * programming interface.
 *
 * @{
 */

struct cuphyContext;
/**
 * cuPHY context
 */
typedef struct cuphyContext* cuphyContext_t;

/******************************************************************/ /**
 * \brief Allocates and initializes a cuPHY context
 *
 * Allocates a cuPHY library context and returns a handle in the
 * address provided by the caller.
 *
 * Returns ::CUPHY_STATUS_INVALID_ARGUMENT if \p pcontext is NULL.
 *
 * Returns ::CUPHY_STATUS_ALLOC_FAILED if a context cannot be allocated
 * on the host.
 *
 * Returns ::CUPHY_STATUS_SUCCESS if allocation and initialization were
 * successful.
 *
 * \param pcontext - Address to return the new ::cuphyContext_t instance
 * \param flags - Creation flags (currently unused)
 *
 * \return
 * ::CUPHY_STATUS_SUCCESS,
 * ::CUPHY_STATUS_ALLOC_FAILED,
 * ::CUPHY_STATUS_INVALID_ARGUMENT
 *
 * \sa ::cuphyStatus_t,::cuphyGetErrorName,::cuphyGetErrorString,::cuphyDestroyContext
 */
cuphyStatus_t CUPHYWINAPI cuphyCreateContext(cuphyContext_t* pcontext,
                                             unsigned int    flags);

/******************************************************************/ /**
 * \brief Destroys a cuPHY context
 *
 * Destroys a cuPHY context object that was previously created by a call
 * to ::cuphyCreateContext. The handle provided to this function should
 * not be used for any operations after this function returns.
 *
 * Returns ::CUPHY_STATUS_INVALID_ARGUMENT if \p decoder is NULL.
 *
 * Returns ::CUPHY_STATUS_SUCCESS if destruction was successful.
 *
 * \param ctx - previously allocated ::cuphyContext_t instance
 *
 * \return
 * ::CUPHY_STATUS_SUCCESS,
 * ::CUPHY_STATUS_INVALID_ARGUMENT
 *
 * \sa ::cuphyStatus_t,::cuphyCreateContext
 */
cuphyStatus_t CUPHYWINAPI cuphyDestroyContext(cuphyContext_t ctx);

/** @} */ /* END CUPHY_CONTEXT */

/**
 * \defgroup CUPHY_TENSOR_DESC Tensor Descriptors
 *
 * This section describes the tensor descriptor functions of the cuPHY
 * application programming interface.
 *
 * @{
 */

struct cuphyTensorDescriptor;
/**
 * cuPHY Tensor Descriptor handle
 */
typedef struct cuphyTensorDescriptor* cuphyTensorDescriptor_t;

/**
 * cuPHY Tensor parameters
 */
typedef struct _cuphyTensorPrm
{
    cuphyTensorDescriptor_t desc;
    void*                   pAddr;
} cuphyTensorPrm_t;

/**
 * cuPHY Tensor information
 */
/// 1-D Tensor
typedef struct
{
    void*           pAddr;
    cuphyDataType_t elemType;
    int32_t         strides[CUPHY_TENSOR_N_DIM_1];
} cuphyTensorInfo1_t;

/// 2-D Tensor
typedef struct
{
    void*           pAddr;
    cuphyDataType_t elemType;
    int32_t         strides[CUPHY_TENSOR_N_DIM_2];
} cuphyTensorInfo2_t;

/// 3-D Tensor
typedef struct
{
    void*           pAddr;
    cuphyDataType_t elemType;
    int32_t         strides[CUPHY_TENSOR_N_DIM_3];
} cuphyTensorInfo3_t;

/// 4-D Tensor
typedef struct
{
    void*           pAddr;
    cuphyDataType_t elemType;
    int32_t         strides[CUPHY_TENSOR_N_DIM_4];
} cuphyTensorInfo4_t;

/// 5-D Tensor
typedef struct
{
    void*           pAddr;
    cuphyDataType_t elemType;
    int32_t         strides[CUPHY_TENSOR_N_DIM_5];
} cuphyTensorInfo5_t;

/// 6-D Tensor only for ML
typedef struct
{
    void*           pAddr;
    cuphyDataType_t elemType;
    int32_t         strides[6];
} cuphyTensorInfo6_t;


/******************************************************************/ /**
 * \brief Allocates and initializes a cuPHY tensor descriptor
 *
 * Allocates a cuPHY tensor descriptor and returns a handle in the address
 * provided by the caller.
 *
 * The allocated descriptor will have type ::CUPHY_VOID, and (in most
 * cases) cannot be used for operations until the tensor state has been
 * initialized by calling ::cuphySetTensorDescriptor.
 *
 * Upon successful return the tensor descriptor will have a rank of 0.
 *
 * Returns ::CUPHY_STATUS_INVALID_ARGUMENT if \p ptensorDesc is NULL.
 *
 * Returns ::CUPHY_STATUS_ALLOC_FAILED if a tensor descriptor cannot be
 * allocated on the host.
 *
 * Returns ::CUPHY_STATUS_SUCCESS if allocation and initialization were
 * successful.
 *
 * \param ptensorDesc - Address for the new ::cuphyTensorDescriptor_t
 * instance
 *
 * \return
 * ::CUPHY_STATUS_SUCCESS,
 * ::CUPHY_STATUS_ALLOC_FAILED,
 * ::CUPHY_STATUS_INVALID_ARGUMENT
 *
 * \sa ::cuphyStatus_t,::cuphyGetErrorName,::cuphyGetErrorString
 */
cuphyStatus_t CUPHYWINAPI cuphyCreateTensorDescriptor(cuphyTensorDescriptor_t* ptensorDesc);

/******************************************************************/ /**
 * \brief Destroys a cuPHY tensor descriptor
 *
 * Destroys a cuPHY tensor descriptor that was previously allocated by
 * a call to ::cuphyCreateTensorDescriptor. The handle provided to this
 * function should not be used for any operations after this function
 * returns.
 *
 * Returns ::CUPHY_STATUS_INVALID_ARGUMENT if \p tensorDesc is NULL.
 *
 * Returns ::CUPHY_STATUS_SUCCESS if destruction was successful.
 *
 * \param tensorDesc - previously allocated ::cuphyTensorDescriptor_t
 * instance
 *
 * \return
 * ::CUPHY_STATUS_SUCCESS,
 * ::CUPHY_STATUS_INVALID_ARGUMENT
 *
 * \sa ::cuphyStatus_t,::cuphyCreateTensorDescriptor
 */
cuphyStatus_t CUPHYWINAPI cuphyDestroyTensorDescriptor(cuphyTensorDescriptor_t tensorDesc);

/******************************************************************/ /**
 * \brief Provide values for the internal state of a cuPHY tensor descriptor
 *
 * Sets the internal state of a tensor descriptor that was created via
 * the ::cuphyCreateTensorDescriptor function.
 *
 * Note that a tensor descriptor is not associated with a specific memory
 * allocation or address. A tensor descriptor provides the cuPHY library with
 * values that can be used "interpret" a range of memory as a tensor with
 * the specified properties. A tensor descriptor can be used with multiple
 * different addresses, and an address can be accessed with multiple different
 * tensor descriptors.
 *
 * \param tensorDesc - previously allocated ::cuphyTensorDescriptor_t
 * instance
 * \param type - ::cuphyDataType_t enumeration with the desired tensor
 * element type
 * \param numDimensions - the desired tensor rank
 * \param dimensions - an array of dimensions for the tensor descriptor
 * \param strides - an array of strides (may be NULL)
 * \param flags - tensor descriptor flags
 *
 * Returns ::CUPHY_STATUS_INVALID_ARGUMENT if:
 * <ul>
 *   <li>\p tensorDesc is NULL.</li>
 *   <li>\p dimensions is NULL.</li>
 *   <li>\p numDimensions <= 0.</li>
 *   <li>\p numDimensions > CUPHY_DIM_MAX.</li>
 *   <li>\p type is ::CUPHY_VOID.</li>
 *   <li>Any element of the dimensions array is less than equal to 0.</li>
 * </ul>
 *
 * Returns ::CUPHY_STATUS_SUCCESS if the state update was successful.
 *
 * The stride of a given dimension describes the distance between two
 * elements that differ by 1 in that dimension. For example, a 2-dimensional,
 * (10 x 8) matrix with no padding would have a stride[0] = 1 and stride[1] =
 * 10.
 *
 * There is no requirement that strides be in ascending order.
 *
 * The \p flags argument can be used to request that the cuPHY library
 * automatically calculate values for the tensor strides, as a
 * convenience. The values allowed for \p flags are:
 * <ul>
 *   <li>CUPHY_TENSOR_ALIGN_DEFAULT: If strides are provided, they will
 *       be used. Otherwise, set the strides for tight packing.</li>
 *   <li>CUPHY_TENSOR_ALIGN_TIGHT: Set the strides so that no padding is
 *       present. stride[0] = 1, and stride[i] = dimensions[i - 1] * strides[i - 1]</li>
 *   <li>CUPHY_TENSOR_ALIGN_COALESCE: Set the strides for the first dimension
 *       based on the element type, so that the stride (in bytes) will be a
 *       multiple of 128.</li>
 * </ul>
 *
 * \return
 * ::CUPHY_STATUS_SUCCESS,
 * ::CUPHY_STATUS_INVALID_ARGUMENT
 *
 * \sa ::cuphyStatus_t,::cuphyCreateTensorDescriptor
 */
cuphyStatus_t CUPHYWINAPI cuphySetTensorDescriptor(cuphyTensorDescriptor_t tensorDesc,
                                                   cuphyDataType_t         type,
                                                   int                     numDimensions,
                                                   const int               dimensions[],
                                                   const int               strides[],
                                                   unsigned int            flags);

/******************************************************************/ /**
 * \brief Query values for the internal state of a cuPHY tensor descriptor
 *
 * Retrieves the internal state of a tensor descriptor that was created via
 * the ::cuphyCreateTensorDescriptor function and initialized with the
 * ::cuphySetTensorDescriptor function
 *
 * \param tensorDesc - previously allocated ::cuphyTensorDescriptor_t
 * instance
 * \param numDimsRequested - the size of the array provided by the \p dimensions
 *        parameter, and the \p strides parameter (if non-NULL)
 * \param dataType - address for the returned ::cuphyDataType_t (may be NULL)
 * \param numDims - output address for the rank of the tensor descriptor (may be NULL)
 * \param dimensions - output location for dimensions for the tensor descriptor
 * \param strides - output location for tensor strides (may be NULL)
 *
 * Returns ::CUPHY_STATUS_INVALID_ARGUMENT if \p tensorDesc is NULL, or if
 * \p numDimsRequested > 0 and dimensions is NULL.
 *
 * Returns ::CUPHY_STATUS_SUCCESS if the state query was successful.
 *
 * \return
 * ::CUPHY_STATUS_SUCCESS,
 * ::CUPHY_STATUS_INVALID_ARGUMENT
 *
 * \sa ::cuphyStatus_t,::cuphyCreateTensorDescriptor,::cuphySetTensorDescriptor
 */
cuphyStatus_t CUPHYWINAPI cuphyGetTensorDescriptor(cuphyTensorDescriptor_t tensorDesc,
                                                   int                     numDimsRequested,
                                                   cuphyDataType_t*        dataType,
                                                   int*                    numDims,
                                                   int                     dimensions[],
                                                   int                     strides[]);

/******************************************************************/ /**
 * \brief Returns the size of an allocation for a tensor descriptor
 *
 * Calculates the size (in bytes) of an allocation that would be required
 * to represent a tensor described by the given descriptor.
 *
 * \param tensorDesc - previously allocated ::cuphyTensorDescriptor_t
 * instance
 * \param psz - address to hold the calculated size output
 *
 * Returns ::CUPHY_STATUS_INVALID_ARGUMENT if \p tensorDesc is NULL, or if
 * \p psz is NULL.
 *
 * Returns ::CUPHY_STATUS_SUCCESS if the size calculation was successful.
 *
 * \return
 * ::CUPHY_STATUS_SUCCESS,
 * ::CUPHY_STATUS_INVALID_ARGUMENT
 *
 * \sa ::cuphyStatus_t,::cuphyCreateTensorDescriptor,::cuphySetTensorDescriptor
 */
cuphyStatus_t CUPHYWINAPI cuphyGetTensorSizeInBytes(cuphyTensorDescriptor_t tensorDesc,
                                                    size_t*                 psz);

/******************************************************************/ /**
 * \brief Returns a string value for a given data type
 *
 * Returns a string for the given ::cuphyDataType_t, or "UNKNOWN_TYPE"
 * if the type is unknown.
 *
 * \param type - data type (::cuphyDataType_t)
 *
 * \return
 * \p char* pointer to a NULL-terminated string
 *
 * \sa ::cuphyDataType_t
 */
const char* CUPHYWINAPI cuphyGetDataTypeString(cuphyDataType_t type);

/** @} */ /* END CUPHY_TENSOR_DESC */

/**
 * \defgroup CUPHY_TENSOR_OPS Tensor Operations
 *
 * This section describes the tensor operation functions of the cuPHY
 * application programming interface.
 *
 * @{
 */

/******************************************************************/ /**
 * \brief Converts a source tensor to a different type or layout
 *
 * Converts an input tensor (described by an address and a tensor
 * descriptor) to an output tensor, possibly changing layout and/or
 * data type in the process.
 * The input and output tensors must have the same dimensions.
 *
 * Tensors with identical data types, dimensions, and strides may be
 * converted internally using a memory copy operation.
 *
 * The following conversions are currently supported:
 * <ul>
 *   <li>Conversion of all types to tensors with the same dimensions but different
 *   strides</li>
 *   <li>Widening conversions (e.g. conversion of a signed, unsigned, or
 *   floating point fundamental type to the same fundamental type with a
 *   larger range (e.g. CUPHY_R_8I to CUPHY_R_32I)</li>
 * </ul>
 *
 * Other conversions are possible and may be added in the future.
 *
 * \param tensorDescDst - previously allocated ::cuphyTensorDescriptor_t for
 * the destination (output)
 * \param dstAddr - tensor address for output data
 * \param tensorDescSrc - previously allocated ::cuphyTensorDescriptor_t for
 * source data
 * \param srcAddr - tensor address for input data
 * \param strm - CUDA stream for memory copy
 *
 * Returns ::CUPHY_STATUS_INVALID_ARGUMENT if any of \p tensorDescDst, \p dstAddr,
 * \p tensorDescSrc, or \p srcAddr is NULL, or if the data type of either
 * \p tensorDescDst or \p tensorDescSrc is CUPHY_VOID.
 *
 * Returns ::CUPHY_STATUS_SIZE_MISMATCH if all dimensions of tensor descriptors
 * \p tensorDescDst and \p tensorDescSrc do not match.
 *
 * Returns ::CUPHY_STATUS_MEMCPY_ERROR if an error occurred performing a memory
 * copy from the source to the destination.
 *
 * Returns ::CUPHY_STATUS_SUCCESS if the conversion operation was submitted to
 * the given stream successfully.
 *
 * \return
 * ::CUPHY_STATUS_SUCCESS,
 * ::CUPHY_STATUS_SIZE_MISMATCH
 * ::CUPHY_STATUS_INVALID_ARGUMENT
 *
 * \sa ::cuphyStatus_t,::cuphyCreateTensorDescriptor,::cuphySetTensorDescriptor
 */
cuphyStatus_t CUPHYWINAPI cuphyConvertTensor(cuphyTensorDescriptor_t tensorDescDst,
                                             void*                   dstAddr,
                                             cuphyTensorDescriptor_t tensorDescSrc,
                                             const void*             srcAddr,
                                             cudaStream_t            strm);
/** @} */ /* END CUPHY_TENSOR_OPS */


/**
 * Per PUCCH common cell parameters
 */
typedef struct _cuphyPucchCellPrm
{
    uint16_t           nRxAnt;
    uint16_t           slotNum;
    uint16_t           pucchHoppingId;
    uint8_t            mu;
    cuphyTensorInfo3_t tDataRx;
} cuphyPucchCellPrm_t;

/**
 * \defgroup CUPHY_PUSCH_RX_FRONT_END (Channel Estimation + Equalization + SoftDemap)
 *
 * This section describes the common PUSCH receiver front-end parameters of the cuPHY
 * application programming interface.
 *
 * @{
 */

#define OFDM_SYMBOLS_PER_SLOT 14
#define MAX_SRS_SYMBOLS_PER_SLOT 4
typedef enum _cuphyPuschEqCoefAlgoType
{
    PUSCH_EQ_ALGO_TYPE_RZF             = 0,
    PUSCH_EQ_ALGO_TYPE_NOISE_DIAG_MMSE      = 1,
    PUSCH_EQ_ALGO_TYPE_MMSE_IRC             = 2,
    PUSCH_EQ_ALGO_TYPE_MMSE_IRC_SHRINK_RBLW = 3,
    PUSCH_EQ_ALGO_TYPE_MMSE_IRC_SHRINK_OAS  = 4,
    PUSCH_EQ_ALGO_MAX_TYPES
} cuphyPuschEqCoefAlgoType_t;


/// Container which captures information needed for processing a UE group
typedef struct _puschRxUeGrpPrms
{
    //--------------------------------------------------------------------------------
    /// Common parameters
    uint16_t statCellIdx;
    uint16_t nRxAnt;  /// Total number of receiving antennas @todo: read from cell static database?
    uint16_t nLayers; /// Total number of layers for this UE group
    uint16_t slotNum; /// @todo: read from cell-dynamic database?

    /// DFT-s-OFDM parameters
    uint8_t  enableDftSOfdm;
    uint8_t  enableTfPrcd;
    uint8_t  optionalDftSOfdm;
    uint32_t puschIdentity;
    uint8_t  groupOrSequenceHopping;
    uint8_t  N_symb_slot;
    uint8_t  N_slot_frame;
    uint8_t  lowPaprGroupNumber;
    uint16_t lowPaprSequenceNumber;

    /// Frequency domain resource allocation
    uint16_t startPrb; /// starting PRB locations for UE groups
    uint16_t nPrb;     /// number of PRBs
    uint8_t  mu;       /// subcarrier spacing parameter (spacing is 15 kHz * 2^mu)

    /// Time domain resource allocation
    uint8_t puschStartSym;
    uint8_t nPuschSym;

    /// Used in CFO/TA estimation, RSRP measurement
    uint8_t  nUes;
    uint16_t ueIdxs[CUPHY_PUSCH_RX_MAX_N_UE_PER_UE_GROUP]; /// UE indices used for cuPHY PUSCH input/output interfaces
    uint8_t  ueGrpLayerToUeIdx[CUPHY_PUSCH_RX_MAX_N_UE_PER_UE_GROUP];
    
    /// Weighted average CFO estimation
    float   foForgetCoeff[CUPHY_PUSCH_RX_MAX_N_UE_PER_UE_GROUP]; 

    cuphyTensorInfo3_t tInfoDataRx;        /// Slot data tensor information
    cuphyTensorInfo4_t tInfoHEst;          /// Estimated channel tensor information
    cuphyTensorInfo1_t tInfoDmrsAccum;     /// DMRS accumulation buffer used to compute delay mean
    cuphyTensorInfo1_t tInfoDmrsDelayMean; /// DMRS delay mean used for time shifts
    cuphyTensorInfo4_t tInfoDmrsLSEst;     /// DMRS least squares intermediate information
    cuphyTensorInfo2_t tInfoCfoEst;        /// CFO estimate tensor information
    cuphyTensorInfo1_t tInfoNoiseVarPreEq; /// Pre-equalizer Noise-interference power tensor information
    cuphyTensorInfo1_t tInfoPerPrbNoiseVar;/// Pre-equalizer per-PRB Noise-interference power tensor information (in sub-slot processing)
    cuphyTensorInfo1_t tInfoNoiseVarPostEq;/// Post-equalizer Noise-interference power tensor information
    cuphyTensorInfo4_t tInfoLwInv;         /// Inverse Cholesky factor of noise-interference tensor information
    float              noiseVarForDtx;     /// noise var used for DTX detection

    //--------------------------------------------------------------------------------
    /// Channel Estimation parameters
    uint16_t nDmrsSyms;
    uint16_t dmrsSymLocBmsk;
    uint16_t dmrsScrmId;
    uint8_t  dmrsMaxLen;
    uint8_t  dmrsAddlnPos;
    uint8_t  dmrsSymLoc[N_MAX_DMRS_SYMS]; // @todo: optimize
    uint8_t  dmrsCnt;                     // valid entries in dmrsSymLoc
    uint8_t  scid;
    uint8_t  nDmrsCdmGrpsNoData;
    uint8_t  dmrsActiveAccumBuf;
    uint8_t  frontLoadedDmrsUpperBound;
    uint16_t prgSize;
    uint8_t  enablePerPrgChEst;

    int8_t  nDmrsGridsPerPrb;
    uint8_t activeDMRSGridBmsk;
    uint8_t activeTOCCBmsk[2];
    uint8_t activeFOCCBmsk[2];
    uint8_t OCCIdx[MAX_N_LAYERS_PUSCH];
    cuphyTensorInfo4_t tInfoChEstDbg; /// Channel estimation debug tensor information

    //--------------------------------------------------------------------------------
    /// Noise Estimation parameters
    uint8_t dmrsPortIdxs[MAX_N_LAYERS_PUSCH]; /// Layer to port map
    cuphyTensorInfo1_t tInfoNoiseIntfEstInterCtaSyncCnt; /// Noise estimation intermediate workspace buffer tensor parameters (tensor must be pre-initialized with zeros)

    //--------------------------------------------------------------------------------
    /// Channel equalization parameters
    /// Common to both coef comp and soft demap
    uint8_t nTimeChEsts;

    cuphyTensorInfo5_t tInfoEqCoef;     /// Channel equalizer coefficient tensor information
    cuphyTensorInfo4_t tInfoReeDiagInv; /// Channel equalizer residual error tensor information

    /// Equalizer Coef compute
    cuphyTensorInfo4_t tInfoNoisePwrInv; /// Noise power inverse (used in Channel equalizer) tensor information
    cuphyTensorInfo4_t tInfoChEqDbg;     /// Channel equalizer debug tensor information
    float              invNoiseVarLin;   /// inverse noise variance (linear)

    /// Equalizer application and soft demap
    uint8_t nDataSym; // note: this is Nd
    uint8_t enableCfoCorrection;
    uint8_t enableWeightedAverageCfo;
    uint8_t enableToEstimation;
    uint8_t enablePuschTdi;
    uint8_t dataSymLoc[OFDM_SYMBOLS_PER_SLOT];             // @todo: optimize
    uint8_t dataCnt;                                       // valid entries in dataSymLoc
    uint8_t qam[CUPHY_PUSCH_RX_MAX_N_LAYERS_PER_UE_GROUP]; // @todo: point to UE structure ?
    cuphyPuschEqCoefAlgoType_t eqCoeffAlgo;

    cuphyTensorInfo3_t tInfoDataEq;          /// Equalized channel data tensor information
    cuphyTensorInfo1_t tInfoDataEqDft;       /// Equalized channel data tensor information for DFT-s-OFDM (One UE per UEGRP)
    cuphyTensorInfo4_t tInfoLLR;             /// Soft demapped LLR tensor information
    cuphyTensorInfo4_t tInfoLLRCdm1;         /// Soft demapped LLR tensor information
    cuphyTensorInfo4_t tInfoChEqSoftDempDbg; /// Equalizer application and soft demapper debug tensor information

    //--------------------------------------------------------------------------------
    /// CFO/TA Estimation
    uint8_t  nUeLayers[CUPHY_PUSCH_RX_MAX_N_UE_PER_UE_GROUP];
    uint32_t scsKHz; /// subcarrier spacing in KHz

    cuphyTensorInfo3_t tInfoCfoPhaseRot;             /// Time domain phase rotation tensor information
    cuphyTensorInfo3_t tInfoTaPhaseRot;              /// Frequency domain phase rotation tensor information
    cuphyTensorInfo1_t tInfoTaEst;                   /// Estimated Timing Advance/Offset tensor information
    cuphyTensorInfo1_t tInfoCfoTaEstInterCtaSyncCnt; /// CFO/TA intermediate workspace buffer tensor information (tensor must be pre-initialized with zeros)
    cuphyTensorInfo1_t tInfoCfoEstInterCtaSyncCnt;   /// used in sub-slot processing: CFO intermediate workspace buffer tensor information (tensor must be pre-initialized with zeros)
    cuphyTensorInfo1_t tInfoTaEstInterCtaSyncCnt;    /// used in sub-slot processing: TA intermediate workspace buffer tensor information (tensor must be pre-initialized with zeros)
    cuphyTensorInfo1_t tInfoCfoHz;                   /// Measured CFO(Hz) tensor information (per UE)

    //--------------------------------------------------------------------------------
    /// RSSI measurement
    uint16_t           rssiSymPosBmsk;
    cuphyTensorInfo3_t tInfoRssiFull;            /// Measured RSSI (per symbol, per antenna, per UE group)
    cuphyTensorInfo1_t tInfoRssi, tInfoRssiEhq;  /// Measured RSSI tensor information (per UE group)
    cuphyTensorInfo1_t tInfoRssiInterCtaSyncCnt; /// RSSI intermediate workspace buffer tensor parameters (tensor must be pre-initialized with zeros)

    /// RSRP, noise variance, SINR measurement
    cuphyTensorInfo1_t tInfoRsrp, tInfoRsrpEhq;                /// Measured RSRP tensor information (per UE)
    cuphyTensorInfo1_t tInfoRsrpInterCtaSyncCnt; /// RSSI intermediate workspace buffer tensor parameters (tensor must be pre-initialized with zeros)

    cuphyTensorInfo1_t tInfoSinrPreEq;  /// SINR computed from pre-equalizer noise estimate
    cuphyTensorInfo1_t tInfoSinrPostEq; /// SINR computed from post-equalizer residual error covariance
} cuphyPuschRxUeGrpPrms_t;

 // Per PRB RKHS paramaters
typedef struct _rkhsPrbPrms
{
    uint8_t           zpIdx;                // zero padding index, used to load pZpDmrsScEigenVec and pZpInterpVec
    __half            sumEigValues;         // sum of prb eignValues
    cuphyTensorPrm_t  corr_half_nZpDmrsSc;  // correlation between eignevectors seperated by nZpDmrsSc / 2 intervals. Dim: nEigs x nEigs
    cuphyTensorPrm_t  corr;                 // correlation between eigenvectors seperated by 0 to (nCpInt - 1) intervals. Dim: nEigs x nEigs x nCpInt
    cuphyTensorPrm_t  eigVecCob;            // change of basis between prb and zp-prb eigenvectors. Dim: nEigs x nZpDmrsScEigs
    cuphyTensorPrm_t  eigVal;               // Prb eigenvalues. Dim: nEigs x 1
    cuphyTensorPrm_t  interpCob;            // change of basis between prb and interpolation eigenvectors. Dim: nIterpEigs x nEigs
} rkhsPrbPrms_t;

// Per zero-padding RKHS paramaters
typedef struct _rkhsZpPrms
{
    cuphyTensorPrm_t zpDmrsScEigenVec;          // zero-padded eigenvector tensor paramaters
    cuphyTensorPrm_t zpInterpVec;               // interpolation eigenvector tensor paramaters
    cuphyTensorPrm_t secondStageTwiddleFactors; // twiddle factors for second stage of FFT
    cuphyTensorPrm_t secondStageFourierPerm;    // input permutation for input to second stage Fourier transform
}rkhsZpPrms_t;

// RKHS paramaters
typedef struct _cuphyPuschRkhsPrms
{
    uint8_t           nZpSizes;          // number of zero-padding sizes
    uint16_t          nPrbSizes;         // number of prb sizes
    rkhsPrbPrms_t*    pPerPrbRkhsPrms;   // pointer to per PRB rkhs paramaters
    rkhsZpPrms_t*     pPerZpRkhsPrms;    // pointer to per ZP rkhs paramaters
}cuphyPuschRkhsPrms_t;


// RKHS grid paramaters
typedef struct _cuphyRkhsGridPrms
{
    uint8_t     nEigs;      // number of eigenvectors/eigenvalues
    uint16_t    gridSize;   // grid size
    uint16_t    zpGridSize; // zero padded grid size

    cuphyTensorPrm_t eigenVecs;   // eigen vectors. Dim: gridSize x nEigs
    cuphyTensorPrm_t eigenValues; // eigen values. Dim: nEigs x 1
    cuphyTensorPrm_t eigenCorr;   // correlation of modulated eigenvectors.

    cuphyTensorPrm_t secondStageTwiddleFactors; // twiddle factors for second stage of FFT
    cuphyTensorPrm_t secondStageFourierPerm;    // input permutation for input to second stage Fourier transform
}cuphyRkhsGridPrms_t;


// SRS RKHS paramaters
typedef struct _cuphySrsRkhsPrms
{
    uint8_t              nGridSizes;     // number of srs RKHS grids
    cuphyRkhsGridPrms_t* pRkhsGridPrms;  // pointer RKHS grid paramaters
}cuphySrsRkhsPrms_t;

/**
 * cuPHY PUSCH Receiver front-end graph node creation parameters
 */
typedef struct _cuphyPuschRxFeCreateGraphNodePrms
{
    cudaGraph_t*     pGraph;
    cudaGraphNode_t* pNode;
    cudaGraphNode_t* pDependencies;
    size_t           nDependencies;
} cuphyPuschRxFeCreateGraphNodePrms_t;

/**
 * cuPHY PUSCH Receiver front-end graph node update parameters
 */
typedef struct _cuphyPuschRxFeUpdateGraphNodePrms
{
    cudaGraphExec_t* pGraphExec;
    cudaGraphNode_t* pNode;
} cuphyPuschRxFeUpdateGraphNodePrms_t;

/**
 * cuPHY PUSCH Receiver front-end graph node creation/update time parameters
 */
typedef struct _cuphyPuschRxFeGraphNodePrms
{
    cuphyPuschRxFeCreateGraphNodePrms_t* pCreatePrms;
    cuphyPuschRxFeUpdateGraphNodePrms_t* pUpdatePrms;
    uint8_t*                             pSuccess;
} cuphyPuschRxFeGraphNodePrms_t;

/** @} */ /* END CUPHY_PUSCH_RX_FRONT_END */

/**
 * \defgroup CUPHY_SUB_SLOT_PROCESSING sub-slot processing
 *
 * This section describes the functions in sub-slot processing in
 * the cuPHY application programming interface.
 *
 * @{
 */

typedef struct
{
    CUDA_KERNEL_NODE_PARAMS kernelNodeParamsDriver;
    void*                   kernelArgs[2];
} cuphyPuschRxWaitLaunchCfg_t;

typedef struct
{
    CUDA_KERNEL_NODE_PARAMS kernelNodeParamsDriver;
    void*                   kernelArgs[2];
} cuphyPuschRxDglLaunchCfg_t;

/** @} */ /* END CUPHY_SUB_SLOT_PROCESSING */

/**
 * \defgroup CUPHY_CHANNEL_ESTIMATION Channel Estimation
 *
 * This section describes the channel estimation functions of the cuPHY
 * application programming interface.
 *
 * @{
 */

struct cuphyPuschRxChEst;
/**
 * cuPHY PUSCH Receiver channel estimation handle
 */
typedef struct cuphyPuschRxChEst* cuphyPuschRxChEstHndl_t;

/**
 * cuPHY PUSCH Receiver channel estimation launch configuration
 */
typedef struct
{
    // For DMRS kernels that estimate the delay mean, we call two kernels. For modes that do
    // not currently support delay mean estimation (e.g. DFT-s-OFDM), we only use the first
    // of the following kernel node parameters.
    CUDA_KERNEL_NODE_PARAMS kernelNodeParamsDriver;
    CUDA_KERNEL_NODE_PARAMS kernelNodeParamsDriverSecond;
    void*                   kernelArgs[2];
    uint16_t                chEst1DmrsSymLocBmsk;
} cuphyPuschRxChEstLaunchCfg_t;

typedef struct
{
    uint32_t                     nCfgs;
    cuphyPuschRxChEstLaunchCfg_t cfgs[CUPHY_PUSCH_RX_CH_EST_ALL_ALGS_N_MAX_HET_CFGS];
} cuphyPuschRxChEstLaunchCfgs_t;

/******************************************************************/ /**
 * \brief Helper to compute cuPHY channel estimation descriptor buffer sizes and alignments
 *
 * Computes cuPHY PUSCH channel estimation descriptor buffer sizes and alignments. To be used by the caller to
 * allocate these buffers (in CPU and GPU memories) and provide them to other PuschRxChEst APIs
 *
 * Returns ::CUPHY_STATUS_INVALID_ARGUMENT if \p pStatDescrSizeBytes and/or \p pStatDescrAlignBytes and/or
 * \p pDynDescrSizeBytes and/or \p pDynDescrAlignBytes is NULL
 *
 * Returns ::CUPHY_STATUS_SUCCESS if allocation and initialization were
 * successful.
 *
 * \param pStatDescrSizeBytes  - Size in bytes of static descriptor
 * \param pStatDescrAlignBytes - Alignment in bytes of static descriptor
 * \param pDynDescrSizeBytes   - Size in bytes of dynamic descriptor
 * \param pDynDescrAlignBytes  - Alignment in bytes of dynamic descriptor
 *
 * \return
 * ::CUPHY_STATUS_SUCCESS,
 * ::CUPHY_STATUS_INVALID_ARGUMENT
 *
 * \sa ::cuphyStatus_t,::cuphyGetErrorName,::cuphyGetErrorString
 */
cuphyStatus_t CUPHYWINAPI cuphyPuschRxChEstGetDescrInfo(size_t* pStatDescrSizeBytes,
                                                        size_t* pStatDescrAlignBytes,
                                                        size_t* pDynDescrSizeBytes,
                                                        size_t* pDynDescrAlignBytes);

/******************************************************************/ /**
 * \brief Performs 1-D time/frequency channel estimation
 *
 * Performs MMSE channel estimation using 1-D interpolation in the time
 * and frequency dimensions
 *
 * \param tensorDescDst - tensor descriptor for output
 * \param dstAddr - address for tensor output
 * \param tensorDescSymbols - tensor descriptor for input symbol data
 * \param symbolsAddr - address for input symbol data
 * \param tensorDescFreqFilters - tensor descriptor for input frequency filters
 * \param freqFiltersAddr - address for input frequency filters
 * \param tensorDescTimeFilters - tensor descriptor for input time filters
 * \param timeFiltersAddr - address for input time filters
 * \param tensorDescFreqIndices - tensor descriptor for pilot symbol frequency indices
 * \param freqIndicesAddr - address for pilot symbol frequency indices
 * \param tensorDescTimeIndices - tensor descriptor for pilot symbol time indices
 * \param timeIndicesAddr - address for pilot symbol time indices
 * \param strm - CUDA stream for kernel launch
 *
 * Returns ::CUPHY_STATUS_INVALID_ARGUMENT if any of the tensor descriptors or
 * address values are NULL.
 *
 * Returns ::CUPHY_STATUS_SUCCESS if submission of the kernel was successful
 *
 * \return
 * ::CUPHY_STATUS_SUCCESS,
 * ::CUPHY_STATUS_INVALID_ARGUMENT
 *
 * \sa ::cuphyStatus_t
 */
cuphyStatus_t CUPHYWINAPI cuphyChannelEst1DTimeFrequency(cuphyTensorDescriptor_t tensorDescDst,
                                                         void*                   dstAddr,
                                                         cuphyTensorDescriptor_t tensorDescSymbols,
                                                         const void*             symbolsAddr,
                                                         cuphyTensorDescriptor_t tensorDescFreqFilters,
                                                         const void*             freqFiltersAddr,
                                                         cuphyTensorDescriptor_t tensorDescTimeFilters,
                                                         const void*             timeFiltersAddr,
                                                         cuphyTensorDescriptor_t tensorDescFreqIndices,
                                                         const void*             freqIndicesAddr,
                                                         cuphyTensorDescriptor_t tensorDescTimeIndices,
                                                         const void*             timeIndicesAddr,
                                                         cudaStream_t            strm);

/** @} */ /* END CUPHY_CHANNEL_ESTIMATION */


/**
 * \defgroup CUPHY_TRT_ENGINE AI/ML TensorRT engine
 *
 * This section describes the TensorRT engine functions of the cuPHY
 * application programming interface.
 *
 * @{
 */

struct cuphyTrtEngine;
/**
 * cuPHY TensorRT engine handle
 */
typedef struct cuphyTrtEngine* cuphyTrtEngineHndl_t;


/**
 * cuPHY TensorRT tensor parameters. This structure captures the parameters of a single
 * TensorRT input/output tensor.
 */
typedef struct _cuphyTrtTensorPrms
{
    const char*      name;
    cuphyDataType_t  dataType;
    uint8_t          nDims;
    int              dims[6];
} cuphyTrtTensorPrms_t;


/******************************************************************/ /**
 * \brief Allocate and initialize a cuPHY TensorRT engine object
 *
 * Allocates a cuPHY TensorRT engine object and returns a handle in the
 * address provided by the caller.
 *
 * Returns ::CUPHY_STATUS_INVALID_ARGUMENT if \p pTrtEngineHndl is NULL, or if the
 * model file is not valid.
 *
 * Returns ::CUPHY_STATUS_INTERNAL_ERROR if a the TensorRT engine can not be allocated.
 *
 * Returns ::CUPHY_STATUS_SUCCESS if allocation and initialization were successful.
 *
 * \param pTrtEngineHndl                - Address to return the new TensoprRT engine instance.
 * \param modelFile                     - TRT model filename.
 * \param maxBatchSize                  - Maximum batch size. Memory is allocated based on this value.
 * \param inputTensorPrms               - Input tensor parameters.
 * \param numInputs                     - Number of input tensors.
 * \param outputTensorPrms              - Output tensor parameters.
 * \param numOutputs                    - Number of output tensors.
 * \param cuStream                      - CUDA stream for kernel launch.
 *
 * \return
 * ::CUPHY_STATUS_SUCCESS,
 * ::CUPHY_STATUS_INTERNAL_ERROR,
 * ::CUPHY_STATUS_INVALID_ARGUMENT
 *
 * \sa ::cuphyStatus_t,::cuphyGetErrorName,::cuphyGetErrorString,::cuphySetupTrtEngine,::cuphyDestroyTrtEngine,::cuphyRunTrtEngine
 */
cuphyStatus_t CUPHYWINAPI cuphyCreateTrtEngine(cuphyTrtEngineHndl_t*     pTrtEngineHndl,
                                               const char*               modelFile,
                                               const uint32_t            maxBatchSize,
                                               cuphyTrtTensorPrms_t*     inputTensorPrms,
                                               uint8_t                   numInputs,
                                               cuphyTrtTensorPrms_t*     outputTensorPrms,
                                               uint8_t                   numOutputs,
                                               cudaStream_t              cuStream);

/******************************************************************/ /**
 * \brief Destroys a cuPHY TensorRT engine object
 *
 * Destroys a cuPHY TensorRT engine object that was previously
 * created by ::cuphyCreateTrtEngine. The handle provided to this function
 * should not be used for any operations after this function returns.
 *
 * Returns ::CUPHY_STATUS_INVALID_ARGUMENT if \p trtEngineHndl is NULL.
 *
 * Returns ::CUPHY_STATUS_SUCCESS if destruction was successful.
 *
 * \param trtEngineHndl - Handle to previously allocated TrtEngine instance.
 *
 * \return
 * ::CUPHY_STATUS_SUCCESS,
 * ::CUPHY_STATUS_INVALID_ARGUMENT
 *
 * \sa ::cuphyStatus_t,::cuphyGetErrorName,::cuphyGetErrorString,::cuphySetupTrtEngine,::cuphyRunTrtEngine,::cuphyCreateTrtEngine
 */
cuphyStatus_t CUPHYWINAPI cuphyDestroyTrtEngine(cuphyTrtEngineHndl_t trtEngineHndl);

/******************************************************************/ /**
 * \brief Setup cuPHY TensorRT engine
 *
 * Setup cuPHY TensorRT in preparation towards slot execution. Set
 * tensor addresses to TRT.
 *
 * Returns ::CUPHY_STATUS_SUCCESS if setup is successful.
 *
 * Returns ::CUPHY_STATUS_INVALID_ARGUMENT if the tensor names or device memory addresses are invalid.
 *
 * Returns ::CUPHY_STATUS_INTERNAL_ERROR if not all input dimensions are specified in the TRT model.
 *
 * \param trtEngineHndl                 - Handle to previously created TrtEngine instance
 * \param ppInputDeviceBuf              - Input tensors to TRT engine in device memory.
 * \param numInputs                     - Number of input tensors.
 * \param ppOutputDeviceBuf             - Output tensors in device memory.
 * \param numOutputs                    - Number of output tensors.
 * \param batchSize                     - Actual batch size. If not given, maximum batch size is used.
 *
 * \return
 * ::CUPHY_STATUS_SUCCESS,
 * ::CUPHY_STATUS_INVALID_ARGUMENT
 * ::CUPHY_STATUS_INTERNAL_ERROR
 *
 * \sa ::cuphyStatus_t,::cuphyGetErrorName,::cuphyGetErrorString,::cuphyCreateTrtEngine,::cuphyDestroyTrtEngine,::cuphyRunTrtEngine
 */
cuphyStatus_t CUPHYWINAPI
cuphySetupTrtEngine(cuphyTrtEngineHndl_t        trtEngineHndl,
                    void**                      ppInputDeviceBuf,
                    uint8_t                     numInputs,
                    void**                      ppOutputDeviceBuf,
                    uint8_t                     numOutputs,
                    uint32_t                    batchSize = 0);

/******************************************************************/ /**
 * \brief Run cuPHY TensorRT engine
 *
 * Run the TensorRT engine.
 *
 * Returns ::CUPHY_STATUS_SUCCESS if the run is successful.
 *
 * Returns ::CUPHY_STATUS_INTERNAL_ERROR if the TRT enqueue operation fails.
 *
 * \param trtEngineHndl                   - Handle to previously created TrtEngine instance
 * \param strm                            - CUDA stream for running everything
 *
 * \return
 * ::CUPHY_STATUS_SUCCESS,
 * ::CUPHY_STATUS_INVALID_ARGUMENT
 *
 * \sa ::cuphyStatus_t,::cuphyGetErrorName,::cuphyGetErrorString,::cuphyCreateTrtEngine,::cuphyDestroyTrtEngine,::cuphySetupTrtEngine
 */
cuphyStatus_t CUPHYWINAPI cuphyRunTrtEngine(cuphyTrtEngineHndl_t       trtEngineHndl,
                                            cudaStream_t               strm);

/** @} */ /* END CUPHY_TRT_ENGINE */



/**
 * \defgroup CUPHY_PUSCH_NOISE_INTERFERENCE_ESTIMATION Noise-interference Estimation
 *
 * This section describes the PUSCH noise-interference estimation functions of the cuPHY
 * application programming interface.
 *
 * @{
 */

struct cuphyPuschRxNoiseIntfEst;
/**
 * cuPHY PUSCH Receiver noise-interference estimation handle
 */
typedef struct cuphyPuschRxNoiseIntfEst* cuphyPuschRxNoiseIntfEstHndl_t;

/**
 * cuPHY PUSCH Receiver noise-interference estimation launch configuration
 */
typedef struct
{
    CUDA_KERNEL_NODE_PARAMS kernelNodeParamsDriver;
    void*                   kernelArgs[1];
} cuphyPuschRxNoiseIntfEstLaunchCfg_t;

typedef struct
{
    uint32_t                     nCfgs;
    cuphyPuschRxNoiseIntfEstLaunchCfg_t cfgs[CUPHY_PUSCH_RX_NOISE_INTF_EST_N_MAX_HET_CFGS * N_MAX_SUB_SLOT_STAGES];
} cuphyPuschRxNoiseIntfEstLaunchCfgs_t;

/******************************************************************/ /**
 * \brief Helper to compute cuPHY PUSCH noise-interference estimation descriptor buffer sizes and alignments
 *
 * Computes cuPHY PUSCH noise-interference estimation descriptor buffer sizes and alignments. To be used by the caller to
 * allocate these buffers (in CPU and GPU memories) and provide them to other PuschRxNoiseIntfEst APIs
 *
 * Returns ::CUPHY_STATUS_INVALID_ARGUMENT if \p pDynDescrSizeBytes and/or \p pDynDescrAlignBytes is NULL
 *
 * Returns ::CUPHY_STATUS_SUCCESS if allocation and initialization were
 * successful.
 *
 * \param pDynDescrSizeBytes   - Size in bytes of dynamic descriptor
 * \param pDynDescrAlignBytes  - Alignment in bytes of dynamic descriptor
 *
 * \return
 * ::CUPHY_STATUS_SUCCESS,
 * ::CUPHY_STATUS_INVALID_ARGUMENT
 *
 * \sa ::cuphyStatus_t,::cuphyGetErrorName,::cuphyGetErrorString,::cuphyCreatePuschRxNoiseIntfEst,::cuphyDestroyPuschRxNoiseIntfEst
 */
cuphyStatus_t CUPHYWINAPI cuphyPuschRxNoiseIntfEstGetDescrInfo(size_t* pDynDescrSizeBytes,
                                                               size_t* pDynDescrAlignBytes);

/******************************************************************/ /**
 * \brief Allocate and initialize a cuPHY PuschRx noise-interference estimation object
 *
 * Allocates a cuPHY PUSCH noise-interference estimation object and returns a handle in the
 * address provided by the caller.
 *
 * Returns ::CUPHY_STATUS_INVALID_ARGUMENT if \p pPuschRxNoiseIntfEstH is NULL.
 *
 * Returns ::CUPHY_STATUS_ALLOC_FAILED if a PuschRxNoiseIntfEst object cannot be allocated
 *
 * Returns ::CUPHY_STATUS_SUCCESS if allocation and initialization were successful
 *
 * \param pPuschRxNoiseIntfEstHndl    - Address to return the new PuschRxNoiseIntfEst instance
 *
 * \return
 * ::CUPHY_STATUS_SUCCESS,
 * ::CUPHY_STATUS_ALLOC_FAILED,
 * ::CUPHY_STATUS_INVALID_ARGUMENT
 *
 * \sa ::cuphyStatus_t,::cuphyGetErrorName,::cuphyGetErrorString,::cuphyPuschRxNoiseIntfEstGetDescrInfo,::cuphySetupPuschRxNoiseIntfEst,::cuphyDestroyPuschRxNoiseIntfEst
 */
cuphyStatus_t CUPHYWINAPI cuphyCreatePuschRxNoiseIntfEst(cuphyPuschRxNoiseIntfEstHndl_t* pPuschRxNoiseIntfEstHndl);

/******************************************************************/ /**
 * \brief Setup cuPHY noise-interference estimation for slot processing
 *
 * Setup cuPHY PUSCH noise-interference estimation in preparation towards slot execution
 *
 * Returns ::CUPHY_STATUS_SUCCESS if setup is successful.
 *
 * Returns ::CUPHY_STATUS_INVALID_ARGUMENT if \p pPuschRxNoiseIntfEstHndl and/or
 * and/or \p ppDynDescrsCpu and/or \p ppDynDescrsGpu is NULL.
 *
 * \param puschRxNoiseIntfEstHndl     - Handle to previously created PuschRxNoiseIntfEst instance
 * \param pDrvdUeGrpPrmsCpu           - Pointer to derived UE groups parameters in CPU memory
 * \param pDrvdUeGrpPrmsGpu           - Pointer to derived UE groups parameters in GPU memory
 * \param nUeGrps                     - number of UE groups to be processed
 * \param nMaxPrb                     - maximum number of PRBs across UE groups
 * \param enableDftSOfdm              - Flag when set support DFT-s-OFDM
 * \param dmrsSymbolIdx               - Index of DMRS symbols to be processed
 * \param enableCpuToGpuDescrAsyncCpy - Flag when set enables async copy of CPU descriptor into GPU
 * \param pDynDescrsCpu               - Pointer to dynamic descriptors in CPU memory
 * \param pDynDescrsGpu               - Pointer to dynamic descriptors in GPU memory
 * \param pLaunchCfgs                 - Pointer to noise-interference estimation launch configurations
 * \param strm                        - CUDA stream for descriptor copy operation
 * \param subSlotStageIdx             - Index used in sub-slot processing, default index of 0 used in full-slot processing mode
 *
 * \return
 * ::CUPHY_STATUS_SUCCESS,
 * ::CUPHY_STATUS_INVALID_ARGUMENT
 *
 * \sa ::cuphyStatus_t,::cuphyGetErrorName,::cuphyGetErrorString,::cuphyPuschRxNoiseIntfEstGetDescrInfo,::cuphyCreatePuschRxNoiseIntfEst,::cuphyDestroyPuschRxNoiseIntfEst
 */
cuphyStatus_t CUPHYWINAPI
cuphySetupPuschRxNoiseIntfEst(cuphyPuschRxNoiseIntfEstHndl_t        puschRxNoiseIntfEstHndl,
                              cuphyPuschRxUeGrpPrms_t*              pDrvdUeGrpPrmsCpu,
                              cuphyPuschRxUeGrpPrms_t*              pDrvdUeGrpPrmsGpu,
                              uint16_t                              nUeGrps,
                              uint16_t                              nMaxPrb,
                              uint8_t                               enableDftSOfdm,
                              uint8_t                               dmrsSymbolIdx,
                              uint8_t                               enableCpuToGpuDescrAsyncCpy,
                              void*                                 pDynDescrsCpu,
                              void*                                 pDynDescrsGpu,
                              cuphyPuschRxNoiseIntfEstLaunchCfgs_t* pLaunchCfgs,
                              cudaStream_t                          strm,
                              uint8_t                               subSlotStageIdx = 0);

/******************************************************************/ /**
 * \brief Destroys a cuPHY PUSCH noise-interference estimation object
 *
 * Destroys a cuPHY PUSCH noise-interference estimation object that was previously
 * created by ::cuphyCreatePuschRxNoiseIntfEst. The handle provided to this function
 * should not be used for any operations after this function returns.
 *
 * Returns ::CUPHY_STATUS_INVALID_ARGUMENT if \p puschRxNoiseIntfEstHndl is NULL.
 *
 * Returns ::CUPHY_STATUS_SUCCESS if destruction was successful.
 *
 * \param puschRxNoiseIntfEstHndl - handle to previously allocated PuschRxNoiseIntfEst instance
 *
 * \return
 * ::CUPHY_STATUS_SUCCESS,
 * ::CUPHY_STATUS_INVALID_ARGUMENT
 *
 * \sa ::cuphyStatus_t,::cuphyGetErrorName,::cuphyGetErrorString,::cuphyPuschRxNoiseIntfEstGetDescrInfo,::cuphyCreatePuschRxNoiseIntfEst,::cuphySetupPuschRxNoiseIntfEst
 */
cuphyStatus_t CUPHYWINAPI cuphyDestroyPuschRxNoiseIntfEst(cuphyPuschRxNoiseIntfEstHndl_t puschRxNoiseIntfEstHndl);

/** @} */ /* END CUPHY_PUSCH_NOISE_INTERFERENCE_ESTIMATION */

/**
 * \defgroup CUPHY_CFO_TA_ESTIMATION Carrier Frequency Offset and Timing Advance Estimation
 *
 * This section describes the carrier frequency offset and timing advance estimation functions of the cuPHY
 * application programming interface.
 *
 * @{
 */

struct cuphyPuschRxCfoTaEst;
/**
 * cuPHY PUSCH Receiver carrier frequency offset estimation handle
 */
typedef struct cuphyPuschRxCfoTaEst* cuphyPuschRxCfoTaEstHndl_t;

/**
 * cuPHY PUSCH Receiver carrier frequency offset and timing advance estimation launch configuration
 */
typedef struct
{
    CUDA_KERNEL_NODE_PARAMS kernelNodeParamsDriver;
    void*                   kernelArgs[2];
} cuphyPuschRxCfoTaEstLaunchCfg_t;

typedef struct
{
    uint32_t                        nCfgs;
    cuphyPuschRxCfoTaEstLaunchCfg_t cfgs[CUPHY_PUSCH_RX_CFO_EST_N_MAX_HET_CFGS];
} cuphyPuschRxCfoTaEstLaunchCfgs_t;

/******************************************************************/ /**
 * \brief Helper to compute cuPHY CFO and TA estimation descriptor buffer sizes and alignments
 *
 * Computes cuPHY PUSCH carrier frequency offset and timing advance estimation descriptor buffer sizes and alignments.
 * To be used by the caller to allocate these buffers (in CPU and GPU memories) and provide them to other PuschRxCfoTaEst APIs
 *
 * Returns ::CUPHY_STATUS_INVALID_ARGUMENT if \p pStatDescrSizeBytes and/or \p pStatDescrAlignBytes and/or
 * \p pDynDescrSizeBytes and/or \p pDynDescrAlignBytes is NULL
 *
 * Returns ::CUPHY_STATUS_SUCCESS if allocation and initialization were
 * successful.
 *
 * \param pStatDescrSizeBytes  - Size in bytes of static descriptor
 * \param pStatDescrAlignBytes - Alignment in bytes of static descriptor
 * \param pDynDescrSizeBytes   - Size in bytes of dynamic descriptor
 * \param pDynDescrAlignBytes  - Alignment in bytes of dynamic descriptor
 *
 * \return
 * ::CUPHY_STATUS_SUCCESS,
 * ::CUPHY_STATUS_INVALID_ARGUMENT
 *
 * \sa ::cuphyStatus_t,::cuphyGetErrorName,::cuphyGetErrorString,::cuphyCreatePuschRxCfoTaEst,::cuphyDestroyPuschRxCfoTaEst
 */
cuphyStatus_t CUPHYWINAPI cuphyPuschRxCfoTaEstGetDescrInfo(size_t* pStatDescrSizeBytes,
                                                           size_t* pStatDescrAlignBytes,
                                                           size_t* pDynDescrSizeBytes,
                                                           size_t* pDynDescrAlignBytes);

/******************************************************************/ /**
 * \brief Allocate and initialize a cuPHY PuschRx CFO and TA estimation object
 *
 * Allocates a cuPHY carrier frequency offset and timing advance estimation object and returns a handle in the
 * address provided by the caller.
 *
 * Returns ::CUPHY_STATUS_INVALID_ARGUMENT if \p pPuschRxCfoTaEstHndl and/or \p pStatDescrCpu and/or \p pStatDescrGpu is NULL.
 *
 * Returns ::CUPHY_STATUS_ALLOC_FAILED if a PuschRxCfoTaEst object cannot be allocated
 *
 * Returns ::CUPHY_STATUS_SUCCESS if allocation and initialization were successful
 *
 * \param pPuschRxCfoTaEstHndl        - Address to return the new PuschRxCfoTaEst instance
 * \param enableCpuToGpuDescrAsyncCpy - flag if non-zero enables async copy of CPU descriptor into GPU
 * \param pStatDescrCpu               - Pointer to static descriptor in CPU memory
 * \param pStatDescrGpu               - Pointer to static descriptor in GPU memory
 * \param strm                        - CUDA stream for descriptor copy operation
 *
 * \return
 * ::CUPHY_STATUS_SUCCESS,
 * ::CUPHY_STATUS_ALLOC_FAILED,
 * ::CUPHY_STATUS_INVALID_ARGUMENT
 *
 * \sa ::cuphyStatus_t,::cuphyGetErrorName,::cuphyGetErrorString,::cuphyPuschRxCfoTaEstGetDescrInfo,::cuphySetupPuschRxCfoTaEst,::cuphyDestroyPuschRxCfoTaEst
 */
cuphyStatus_t CUPHYWINAPI cuphyCreatePuschRxCfoTaEst(cuphyPuschRxCfoTaEstHndl_t* pPuschRxCfoTaEstHndl,
                                                     uint8_t                     enableCpuToGpuDescrAsyncCpy,
                                                     void*                       pStatDescrCpu,
                                                     void*                       pStatDescrGpu,
                                                     cudaStream_t                strm);

/******************************************************************/ /**
 * \brief Setup cuPHY PuschRx CFO and TA estimation for slot processing
 *
 * Setup cuPHY PUSCH carrier frequency offset and timing advance estimation in preparation towards slot execution
 *
 * Returns ::CUPHY_STATUS_SUCCESS if setup is successful.
 *
 * Returns ::CUPHY_STATUS_INVALID_ARGUMENT if \p puschRxCfoTaEstHndl and/or \p pDynDescrsCpu and/or \p pDynDescrsGpu and/or \p pLaunchCfgs is NULL.
 *
 * \param puschRxCfoTaEstHndl         - Handle to previously created PuschRxCfoTaEst instance
 * \param pDrvdUeGrpPrmsCpu           - Pointer to derived UE group parameters on CPU
 * \param pDrvdUeGrpPrmsGpu           - Pointer to derived UE group parameters on GPU
 * \param pFoCompensationBuffers      - Pointers to the CFO compensation buffers on GPU
 * \param nUeGrps                     - number of UE groups to be processed
 * \param nMaxPrb                     - maximum number of PRBs across UE groups
 * \param pDbg                        - Pointer to debug tensor parameters (0 if no debug info is desired)
 * \param enableCpuToGpuDescrAsyncCpy - Flag when set enables async copy of CPU descriptor into GPU
 * \param pDynDescrsCpu               - Pointer to dynamic descriptor in CPU memory
 * \param pDynDescrsGpu               - Pointer to dynamic descriptor in GPU memory
 * \param pLaunchCfgs                 - Pointer to channel estimation launch configurations
 * \param strm                        - CUDA stream for descriptor copy operation
 *
 * \return
 * ::CUPHY_STATUS_SUCCESS,
 * ::CUPHY_STATUS_INVALID_ARGUMENT
 *
 * \sa ::cuphyStatus_t,::cuphyGetErrorName,::cuphyGetErrorString,::cuphyPuschRxCfoTaEstGetDescrInfo,::cuphyCreatePuschRxCfoTaEst,::cuphyDestroyPuschRxCfoTaEst
 */

cuphyStatus_t CUPHYWINAPI
cuphySetupPuschRxCfoTaEst(cuphyPuschRxCfoTaEstHndl_t        puschRxCfoTaEstHndl,
                          cuphyPuschRxUeGrpPrms_t*          pDrvdUeGrpPrmsCpu,
                          cuphyPuschRxUeGrpPrms_t*          pDrvdUeGrpPrmsGpu,
                          float**                           pFoCompensationBuffers,
                          uint16_t                          nUeGrps,
                          uint32_t                          nMaxPrb,
                          cuphyTensorPrm_t*                 pDbg,
                          uint8_t                           enableCpuToGpuDescrAsyncCpy,
                          void*                             pDynDescrsCpu,
                          void*                             pDynDescrsGpu,
                          cuphyPuschRxCfoTaEstLaunchCfgs_t* pLaunchCfgs,
                          cudaStream_t                      strm);

/******************************************************************/ /**
 * \brief Destroys a cuPHY PUSCH CFO estimation object
 *
 * Destroys a cuPHY PUSCH carrier frequency estimation object that was previously
 * created by ::cuphyCreatePuschRxCfoTaEst. The handle provided to this function
 * should not be used for any operations after this function returns.
 *
 * Returns ::CUPHY_STATUS_INVALID_ARGUMENT if \p puschRxCfoTaEstHndl is NULL.
 *
 * Returns ::CUPHY_STATUS_SUCCESS if destruction was successful.
 *
 * \param puschRxCfoTaEstHndl - handle to previously allocated PuschRxCfoTaEst instance
 *
 * \return
 * ::CUPHY_STATUS_SUCCESS,
 * ::CUPHY_STATUS_INVALID_ARGUMENT
 *
 * \sa ::cuphyStatus_t,::cuphyGetErrorName,::cuphyGetErrorString,::cuphyPuschRxCfoTaEstGetDescrInfo,::cuphyCreatePuschRxCfoTaEst,::cuphySetupPuschRxCfoTaEst
 */
cuphyStatus_t CUPHYWINAPI cuphyDestroyPuschRxCfoTaEst(cuphyPuschRxCfoTaEstHndl_t puschRxCfoTaEstHndl);

/** @} */ /* END CUPHY_CFO_TA_ESTIMATION */

/**
 * \defgroup CUPHY_CHANNEL_EQUALIZATION Channel Equalization
 *
 * This section describes the channel equalization functions of the cuPHY
 * application programming interface.
 *
 * @{
 */

struct cuphyPuschRxChEq;
/**
 * cuPHY PUSCH Receiver channel equalization handle
 */
typedef struct cuphyPuschRxChEq* cuphyPuschRxChEqHndl_t;

/**
 * cuPHY PUSCH Receiver channel equalization launch configuration
 */
typedef struct
{
    CUDA_KERNEL_NODE_PARAMS kernelNodeParamsDriver;
    void*                   kernelArgs[2];
} cuphyPuschRxChEqLaunchCfg_t;

typedef struct
{
    uint32_t                    nCfgs;
    cuphyPuschRxChEqLaunchCfg_t cfgs[CUPHY_PUSCH_RX_CH_EQ_N_MAX_HET_CFGS];
} cuphyPuschRxChEqLaunchCfgs_t;

/******************************************************************/ /**
 * \brief Helper to compute cuPHY channel equalization descriptor buffer sizes and alignments
 *
 * Computes cuPHY PUSCH channel equalization descriptor buffer sizes and alignments. To be used by the caller
 * to allocate these buffers (in CPU and GPU memories) and provide them to other PuschRxChEq APIs
 *
 * Returns ::CUPHY_STATUS_INVALID_ARGUMENT if \p pStatDescrSizeBytes and/or \p pStatDescrAlignBytes and/or
 * \p pCoefCompDynDescrSizeBytes and/or \p pCoefCompDynDescrAlignBytes and/or \p pSoftDemapDynDescrSizeBytes
 * and/or \p pSoftDemapDynDescrAlignBytes
 *
 * Returns ::CUPHY_STATUS_SUCCESS if allocation and initialization were
 * successful.
 *
 * \param pStatDescrSizeBytes          - Size in bytes of equalizer common static descriptor
 * \param pStatDescrAlignBytes         - Alignment in bytes of equalizer common static descriptor
 * \param pIdftStatDescrSizeBytes      - Size in bytes of IDFT static descriptor
 * \param pIdftStatDescrAlignBytes     - Alignment in bytes of IDFT static descriptor
 * \param pCoefCompDynDescrSizeBytes   - Size in bytes of coefficient compute dynamic descriptor
 * \param pCoefCompDynDescrAlignBytes  - Alignment in bytes of coefficient compute dynamic descriptor
 * \param pSoftDemapDynDescrSizeBytes  - Size in bytes of soft demap dynamic descriptor
 * \param pSoftDemapDynDescrAlignBytes - Alignment in bytes of soft demap dynamic descriptor
 *
 * \return
 * ::CUPHY_STATUS_SUCCESS,
 * ::CUPHY_STATUS_INVALID_ARGUMENT
 *
 * \sa ::cuphyStatus_t,::cuphyGetErrorName,::cuphyGetErrorString,::cuphyCreatePuschRxChEq,::cuphySetupPuschRxChEqCoefCompute,::cuphySetupPuschRxChEqSoftDemap,::cuphySetupPuschRxChEqSoftDemapAfterDft,::cuphyDestroyPuschRxChEq
 */
cuphyStatus_t CUPHYWINAPI cuphyPuschRxChEqGetDescrInfo(size_t* pStatDescrSizeBytes,
                                                       size_t* pStatDescrAlignBytes,
                                                       size_t* pIdftStatDescrSizeBytes,
                                                       size_t* pIdftStatDescrAlignBytes,
                                                       size_t* pCoefCompDynDescrSizeBytes,
                                                       size_t* pCoefCompDynDescrAlignBytes,
                                                       size_t* pSoftDemapDynDescrSizeBytes,
                                                       size_t* pSoftDemapDynDescrAlignBytes);

/******************************************************************/ /**
 * \brief Allocate and initialize a cuPHY PuschRx channel equalization object
 *
 * Allocates a cuPHY channel equalization object and returns a handle in the
 * address provided by the caller.
 *
 * Returns ::CUPHY_STATUS_INVALID_ARGUMENT if \p pPuschRxChEqHndl and/or \p ppStatDescrCpu and/or
 * \p ppStatDescrGpu and/or \p ppIdftStatDescrCpu and/or \p ppIdftStatDescrGpu is NULL.
 *
 * Returns ::CUPHY_STATUS_ALLOC_FAILED if a PuschRxChEq object cannot be allocated
 *
 * Returns ::CUPHY_STATUS_SUCCESS if allocation and initialization were successful
 *
 * \param ctx                                - cuPHY context
 * \param pPuschRxChEqHndl                   - Address to return the new PuschRxChEq instance
 * \param tInfoDftBluesteinWorkspaceTime     - tensor info for Bluestein Workspace in the time domain
 * \param tInfoDftBluesteinWorkspaceFreq     - tensor info for Bluestein Workspace in the frequency domain
 * \param cudaDeviceArch                     - GPU device architecture
 * \param enableDftSOfdm                     - flag if DFT-s-OFDM is enabled
 * \param enableDebugEqOutput                - flag if post-eq output should be saved for debug/test purposes
 * \param enableCpuToGpuDescrAsyncCpy        - flag if non-zero enables async copy of CPU descriptor into GPU
 * \param ppStatDescrCpu                     - Pointer to array of static descriptors in CPU memory
 * \param ppStatDescrGpu                     - Pointer to array of static descriptors in GPU memory
 * \param ppIdftStatDescrCpu                 - Pointer to array of static descriptors for IDFT in CPU memory
 * \param ppIdftStatDescrGpu                 - Pointer to array of static descriptors for IDFT in GPU memory
 * \param strm                               - CUDA stream for descriptor copy operation
 *
 * \return
 * ::CUPHY_STATUS_SUCCESS,
 * ::CUPHY_STATUS_ALLOC_FAILED,
 * ::CUPHY_STATUS_INVALID_ARGUMENT
 *
 * \sa ::cuphyStatus_t,::cuphyGetErrorName,::cuphyGetErrorString,::cuphyPuschRxChEqGetDescrInfo,::cuphySetupPuschRxChEqCoefCompute,::cuphySetupPuschRxChEqSoftDemap,::cuphySetupPuschRxChEqSoftDemapAfterDft,::cuphyDestroyPuschRxChEq
 */
cuphyStatus_t CUPHYWINAPI cuphyCreatePuschRxChEq(cuphyContext_t          ctx,
                                                 cuphyPuschRxChEqHndl_t* pPuschRxChEqHndl,
                                                 cuphyTensorInfo2_t&     tInfoDftBluesteinWorkspaceTime,
                                                 cuphyTensorInfo2_t&     tInfoDftBluesteinWorkspaceFreq,
                                                 uint                    cudaDeviceArch,
                                                 uint8_t                 enableDftSOfdm,
                                                 uint8_t                 enableDebugEqOutput,
                                                 uint8_t                 enableCpuToGpuDescrAsyncCpy,
                                                 void**                  ppStatDescrCpu,
                                                 void**                  ppStatDescrGpu,
                                                 void**                  ppIdftStatDescrCpu,
                                                 void**                  ppIdftStatDescrGpu,
                                                 cudaStream_t            strm);

/******************************************************************/ /**
 * \brief Setup cuPHY channel equalization coefficient compute for slot processing
 *
 * Setup cuPHY PUSCH channel equalization coefficient compute in preparation towards slot execution
 *
 * Returns ::CUPHY_STATUS_SUCCESS if setup is successful.
 *
 * Returns ::CUPHY_STATUS_INVALID_ARGUMENT if \p puschRxChEqHndl and/or \p pDynDescrsCpu and/or \p pDynDescrsGpu and/or \p pLaunchCfgs is NULL.
 *
 * \param puschRxChEqHndl             - Handle to previously created PuschRxChEq instance
 * \param pDrvdUeGrpPrmsCpu           - Pointer to derived UE groups parameters in CPU memory
 * \param pDrvdUeGrpPrmsGpu           - Pointer to derived UE groups parameters in GPU memory
 * \param nUeGrps                     - total number of UE groups to be processed
 * \param nMaxPrb                     - maximum number of data PRBs across all UE groups
 * \param enableCfoCorrection         - enable application of CFO correction
 * \param enablePuschTdi              - enable time domain interpolation on equalizer coefficients
 * \param enableCpuToGpuDescrAsyncCpy - Flag when set enables async copy of CPU descriptor into GPU
 * \param pDynDescrsCpu               - Pointer to dynamic descriptor in CPU memory
 * \param pDynDescrsGpu               - Pointer to dynamic descriptor in GPU memory
 * \param pLaunchCfgs                 - Pointer to channel estimation launch configurations
 * \param strm                        - CUDA stream for descriptor copy operation
 *
 * \return
 * ::CUPHY_STATUS_SUCCESS,
 * ::CUPHY_STATUS_INVALID_ARGUMENT
 *
 * \sa ::cuphyStatus_t,::cuphyGetErrorName,::cuphyGetErrorString,::cuphyCreatePuschRxChEq,::cuphyPuschRxChEqGetDescrInfo,::cuphySetupPuschRxChEqSoftDemap,::cuphySetupPuschRxChEqSoftDemapAfterDft,::cuphyDestroyPuschRxChEq
 */
cuphyStatus_t CUPHYWINAPI
cuphySetupPuschRxChEqCoefCompute(cuphyPuschRxChEqHndl_t        puschRxChEqHndl,
                                 cuphyPuschRxUeGrpPrms_t*      pDrvdUeGrpPrmsCpu,
                                 cuphyPuschRxUeGrpPrms_t*      pDrvdUeGrpPrmsGpu,
                                 uint16_t                      nUeGrps,
                                 uint16_t                      nMaxPrb,
                                 uint8_t                       enableCfoCorrection,
                                 uint8_t                       enablePuschTdi,
                                 uint8_t                       enableCpuToGpuDescrAsyncCpy,
                                 void**                        pDynDescrsCpu,
                                 void**                        pDynDescrsGpu,
                                 cuphyPuschRxChEqLaunchCfgs_t* pLaunchCfgs,
                                 cudaStream_t                  strm);

/******************************************************************/ /**
 * \brief Setup cuPHY channel equalization soft demap for slot processing
 *
 * Setup cuPHY PUSCH channel equalization soft demap in preparation towards slot execution
 *
 * Returns ::CUPHY_STATUS_SUCCESS if setup is successful.
 *
 * Returns ::CUPHY_STATUS_INVALID_ARGUMENT if \p puschRxChEqHndl and/or \p pDynDescrsCpu and/or \p pDynDescrsGpu and/or \p pLaunchCfgs is NULL.
 *
 * \param puschRxChEqHndl             - Handle to previously created PuschRxChEq instance
 * \param pDrvdUeGrpPrmsCpu           - Pointer to derived UE groups parameters in CPU memory
 * \param pDrvdUeGrpPrmsGpu           - Pointer to derived UE groups parameters in GPU memory
 * \param nUeGrps                     - total number of UE groups to be processed
 * \param nMaxPrb                     - maximum number of data PRBs across all UE groups
 * \param enableCfoCorrection         - enable application of CFO correction
 * \param enablePuschTdi              - enable time domain interpolation on equalizer coefficients
 * \param symbolBitmask               - bitmask for data symbols to execute soft demapper
 * \param enableCpuToGpuDescrAsyncCpy - Flag when set enables async copy of CPU descriptor into GPU
 * \param pDynDescrsCpu               - Pointer to dynamic descriptor in CPU memory
 * \param pDynDescrsGpu               - Pointer to dynamic descriptor in GPU memory
 * \param pLaunchCfgs                 - Pointer to channel estimation launch configurations
 * \param strm                        - CUDA stream for descriptor copy operation
 *
 * \return
 * ::CUPHY_STATUS_SUCCESS,
 * ::CUPHY_STATUS_INVALID_ARGUMENT
 *
 * \sa ::cuphyStatus_t,::cuphyGetErrorName,::cuphyGetErrorString,::cuphyCreatePuschRxChEq,::cuphyPuschRxChEqGetDescrInfo,::cuphySetupPuschRxChEqCoefCompute,::cuphyDestroyPuschRxChEq
 */
cuphyStatus_t CUPHYWINAPI
cuphySetupPuschRxChEqSoftDemap(cuphyPuschRxChEqHndl_t        puschRxChEqHndl,
                               cuphyPuschRxUeGrpPrms_t*      pDrvdUeGrpPrmsCpu,
                               cuphyPuschRxUeGrpPrms_t*      pDrvdUeGrpPrmsGpu,
                               uint16_t                      nUeGrps,
                               uint16_t                      nMaxPrb,
                               uint8_t                       enableCfoCorrection,
                               uint8_t                       enablePuschTdi,
                               uint16_t                      symbolBitmask,
                               uint8_t                       enableCpuToGpuDescrAsyncCpy,
                               void*                         pDynDescrsCpu,
                               void*                         pDynDescrsGpu,
                               cuphyPuschRxChEqLaunchCfgs_t* pLaunchCfgs,
                               cudaStream_t                  strm);

/******************************************************************/ /**
 * \brief Setup cuPHY channel equalization soft demap for IDFT
 *
 * Setup cuPHY PUSCH channel equalization soft demap in preparation towards slot execution
 *
 * Returns ::CUPHY_STATUS_SUCCESS if setup is successful.
 *
 * Returns ::CUPHY_STATUS_INVALID_ARGUMENT if \p puschRxChEqHndl and/or \p pDynDescrsCpu and/or \p pDynDescrsGpu and/or \p pLaunchCfgs is NULL.
 *
 * \param puschRxChEqHndl             - Handle to previously created PuschRxChEq instance
 * \param pDrvdUeGrpPrmsCpu           - Pointer to derived UE groups parameters in CPU memory
 * \param pDrvdUeGrpPrmsGpu           - Pointer to derived UE groups parameters in GPU memory
 * \param nUeGrps                     - total number of UE groups to be processed
 * \param nMaxPrb                     - maximum number of data PRBs across all UE groups
 * \param cudaDeviceArch              - GPU device architecture
 * \param enableCfoCorrection         - enable application of CFO correction
 * \param enablePuschTdi              - enable time domain interpolation on equalizer coefficients
 * \param symbolBitmask               - bitmask for data symbols to execute soft demapper
 * \param enableCpuToGpuDescrAsyncCpy - Flag when set enables async copy of CPU descriptor into GPU
 * \param pDynDescrsCpu               - Pointer to dynamic descriptor in CPU memory
 * \param pDynDescrsGpu               - Pointer to dynamic descriptor in GPU memory
 * \param pLaunchCfgs                 - Pointer to IDFT1 launch configurations
 * \param strm                        - CUDA stream for descriptor copy operation
 *
 * \return
 * ::CUPHY_STATUS_SUCCESS,
 * ::CUPHY_STATUS_INVALID_ARGUMENT
 *
 * \sa ::cuphyStatus_t,::cuphyGetErrorName,::cuphyGetErrorString,::cuphyCreatePuschRxChEq,::cuphyPuschRxChEqGetDescrInfo,::cuphySetupPuschRxChEqCoefCompute,::cuphySetupPuschRxChEqSoftDemap,::cuphyDestroyPuschRxChEq
 */
cuphyStatus_t CUPHYWINAPI
cuphySetupPuschRxChEqSoftDemapIdft(cuphyPuschRxChEqHndl_t        puschRxChEqHndl,
                                   cuphyPuschRxUeGrpPrms_t*      pDrvdUeGrpPrmsCpu,
                                   cuphyPuschRxUeGrpPrms_t*      pDrvdUeGrpPrmsGpu,
                                   uint16_t                      nUeGrps,
                                   uint16_t                      nMaxPrb,
                                   uint                          cudaDeviceArch,
                                   uint8_t                       enableCfoCorrection,
                                   uint8_t                       enablePuschTdi,
                                   uint16_t                      symbolBitmask,
                                   uint8_t                       enableCpuToGpuDescrAsyncCpy,
                                   void*                         pDynDescrsCpu,
                                   void*                         pDynDescrsGpu,
                                   cuphyPuschRxChEqLaunchCfgs_t* pLaunchCfgs,
                                   cudaStream_t                  strm);
/******************************************************************/ /**
 * \brief Setup cuPHY channel equalization soft demap after DFT-s-OFDM slot processing
 *
 * Setup cuPHY PUSCH channel equalization soft demap in preparation towards slot execution
 *
 * Returns ::CUPHY_STATUS_SUCCESS if setup is successful.
 *
 * Returns ::CUPHY_STATUS_INVALID_ARGUMENT if \p puschRxChEqHndl and/or \p pDynDescrsCpu and/or \p pDynDescrsGpu and/or \p pLaunchCfgs is NULL.
 *
 * \param puschRxChEqHndl             - Handle to previously created PuschRxChEq instance
 * \param pDrvdUeGrpPrmsCpu           - Pointer to derived UE groups parameters in CPU memory
 * \param pDrvdUeGrpPrmsGpu           - Pointer to derived UE groups parameters in GPU memory
 * \param nUeGrps                     - total number of UE groups to be processed
 * \param nMaxPrb                     - maximum number of data PRBs across all UE groups
 * \param enableCfoCorrection         - enable application of CFO correction
 * \param enablePuschTdi              - enable time domain interpolation on equalizer coefficients
 * \param symbolBitmask               - bitmask for data symbols to execute soft demapper
 * \param enableCpuToGpuDescrAsyncCpy - Flag when set enables async copy of CPU descriptor into GPU
 * \param pDynDescrsCpu               - Pointer to dynamic descriptor in CPU memory
 * \param pDynDescrsGpu               - Pointer to dynamic descriptor in GPU memory
 * \param pLaunchCfgs                 - Pointer to channel estimation launch configurations
 * \param strm                        - CUDA stream for descriptor copy operation
 *
 * \return
 * ::CUPHY_STATUS_SUCCESS,
 * ::CUPHY_STATUS_INVALID_ARGUMENT
 *
 * \sa ::cuphyStatus_t,::cuphyGetErrorName,::cuphyGetErrorString,::cuphyCreatePuschRxChEq,::cuphyPuschRxChEqGetDescrInfo,::cuphySetupPuschRxChEqCoefCompute,::cuphySetupPuschRxChEqSoftDemap,::cuphyDestroyPuschRxChEq
 */
cuphyStatus_t CUPHYWINAPI
cuphySetupPuschRxChEqSoftDemapAfterDft(cuphyPuschRxChEqHndl_t        puschRxChEqHndl,
                                       cuphyPuschRxUeGrpPrms_t*      pDrvdUeGrpPrmsCpu,
                                       cuphyPuschRxUeGrpPrms_t*      pDrvdUeGrpPrmsGpu,
                                       uint16_t                      nUeGrps,
                                       uint16_t                      nMaxPrb,
                                       uint8_t                       enableCfoCorrection,
                                       uint8_t                       enablePuschTdi,
                                       uint16_t                      symbolBitmask,
                                       uint8_t                       enableCpuToGpuDescrAsyncCpy,
                                       void*                         pDynDescrsCpu,
                                       void*                         pDynDescrsGpu,
                                       cuphyPuschRxChEqLaunchCfgs_t* pLaunchCfgs,
                                       cudaStream_t                  strm);

/******************************************************************/ /**
 * \brief Perform symbol modulation
 *
 * Perform symbol modulation, generating symbol values for an input
 * sequence of bits
 *
 * Returns ::CUPHY_STATUS_SUCCESS if modulation is launched successfully
 *
 * Returns ::CUPHY_STATUS_INVALID_ARGUMENT if \p tSym, \p pSym, \p tBits, or \p pBits are NULL,
 *         or if \p log2_QAM does not represent a supported modulation
 *         value (1, 2, 4, 6, or 8)
 * Returns ::CUPHY_STATUS_UNSUPPORTED_TYPE is \p tSym is not of type
 *         CUPHY_C_32F or CUPHY_C_16F, or if \p tBits is not of type
 *         CUPHY_BIT
 * Returns ::CUPHY_STATUS_SIZE_MISMATCH if \p tBits is not a multiple
 *         of \p log2_QAM, or if the first dimension of \p tSym is not
 *         equal to first dimension of \p tBits divided by \p log2_QAM
 *
 * \param tSym          - tensor descriptor for complex symbol values
 * \param pSym          - address of output symbol values
 * \param tBits         - tensor descriptor for input bit
 * \param pBits         - address of input bit values
 * \param log2_QAM      - log2(QAM), describing the quadrature amplitude that the
 *                        symbols were modulated with. This is the number of bits
 *                        represented by each symbol. Value values are 1 (BPSK),
 *                        2 (QPSK), 4 (QAM16), 6 (QAM64), and 8 (QAM256)
 * \param strm          - CUDA stream for kernel launch
 *
 * \return
 * ::CUPHY_STATUS_SUCCESS,
 * ::CUPHY_STATUS_INVALID_ARGUMENT
 *
 * \sa ::cuphyStatus_t,::cuphyGetErrorName,::cuphyGetErrorString,::cuphyDemodulateSymbol
 */
cuphyStatus_t cuphyModulateSymbol(cuphyTensorDescriptor_t tSym,
                                  void*                   pSym,
                                  cuphyTensorDescriptor_t tBits,
                                  const void*             pBits,
                                  int                     log2_QAM,
                                  cudaStream_t            strm);

/******************************************************************/ /**
 * \brief Perform symbol demodulation
 *
 * Perform symbol demodulation, generating log-likelihood values (LLRs)
 * for each bit
 *
 * Returns ::CUPHY_STATUS_SUCCESS if demodulation is launched successfully
 *
 * Returns ::CUPHY_STATUS_INVALID_ARGUMENT if \p tSym, \p pSym, \p tLLR, or \p pLLR are NULL.
 *
 * \param context       - cuPHY context
 * \param tLLR          - tensor descriptor for output log-likelihood values
 * \param pLLR          - address of output log-likelihood values
 * \param tSym          - tensor descriptor for symbol values
 * \param pSym          - address of symbol tensor data
 * \param log2_QAM      - log2(QAM), describing the quadrature amplitude that the
 *                        symbols were modulated with
 * \param noiseVariance - QAM noise variance
 * \param strm          - CUDA stream for kernel launch
 *
 * \return
 * ::CUPHY_STATUS_SUCCESS,
 * ::CUPHY_STATUS_INVALID_ARGUMENT
 *
 * \sa ::cuphyStatus_t,::cuphyGetErrorName,::cuphyGetErrorString,::cuphyModulateSymbol
 */
cuphyStatus_t cuphyDemodulateSymbol(cuphyContext_t          context,
                                    cuphyTensorDescriptor_t tLLR,
                                    void*                   pLLR,
                                    cuphyTensorDescriptor_t tSym,
                                    const void*             pSym,
                                    int                     log2_QAM,
                                    float                   noiseVariance,
                                    cudaStream_t            strm);

/******************************************************************/ /**
 * \brief Destroys a cuPHY PUSCH channel equalization object
 *
 * Destroys a cuPHY PUSCH channel equalization object that was previously
 * created by ::cuphyCreatePuschRxChEq. The handle provided to this function
 * should not be used for any operations after this function returns.
 *
 * Returns ::CUPHY_STATUS_INVALID_ARGUMENT if \p puschRxChEqHndl is NULL.
 *
 * Returns ::CUPHY_STATUS_SUCCESS if destruction was successful.
 *
 * \param puschRxChEqHndl - handle to previously allocated PuschRxChEq instance
 *
 * \return
 * ::CUPHY_STATUS_SUCCESS,
 * ::CUPHY_STATUS_INVALID_ARGUMENT
 *
 * \sa ::cuphyStatus_t,::cuphyGetErrorName,::cuphyGetErrorString,::cuphyCreatePuschRxChEq,::cuphyPuschRxChEqGetDescrInfo,::cuphySetupPuschRxChEqCoefCompute,::cuphySetupPuschRxChEqSoftDemap
 */
cuphyStatus_t CUPHYWINAPI cuphyDestroyPuschRxChEq(cuphyPuschRxChEqHndl_t puschRxChEqHndl);

/** @} */ /* END CUPHY_CHANNEL_EQUALIZATION */

/**
 * \defgroup CUPHY_PUSCH_RSSI Pusch receiver RSSI measurement
 *
 * This section describes Pusch receiver Received Signal Strength Indicator
 * measurement functions of the cuPHY application programming interface.
 *
 * @{
 */

struct cuphyPuschRxRssi;
/**
 * cuPHY PUSCH Receiver RSSI metric handle
 */
typedef struct cuphyPuschRxRssi* cuphyPuschRxRssiHndl_t;

/**
 * cuPHY PUSCH Receiver RSSI metric launch configuration
 */
typedef struct
{
    CUDA_KERNEL_NODE_PARAMS kernelNodeParamsDriver;
    void*                   kernelArgs[1];
} cuphyPuschRxRssiLaunchCfg_t;

typedef struct
{
    uint32_t                    nCfgs;
    cuphyPuschRxRssiLaunchCfg_t cfgs[CUPHY_PUSCH_RX_RSSI_N_MAX_HET_CFGS];
} cuphyPuschRxRssiLaunchCfgs_t;

/**
 * cuPHY PUSCH Receiver RSRP metric launch configuration
 */
typedef struct
{
    CUDA_KERNEL_NODE_PARAMS kernelNodeParamsDriver;
    void*                   kernelArgs[1];
} cuphyPuschRxRsrpLaunchCfg_t;

typedef struct
{
    uint32_t                    nCfgs;
    cuphyPuschRxRsrpLaunchCfg_t cfgs[CUPHY_PUSCH_RX_RSRP_N_MAX_HET_CFGS];
} cuphyPuschRxRsrpLaunchCfgs_t;

/******************************************************************/ /**
 * \brief Helper to compute cuPHY RSSI, RSRP measurement descriptor buffer sizes and alignments
 *
 * Computes cuPHY PUSCH RSSI (Received Signal Strength Indicator) and RSRP (Reference Signal Received Power)
 * descriptor buffer sizes and alignments.
 * To be used by the caller to allocate these buffers (in CPU and GPU memories) and provide them to
 * other PuschRxRssi APIs
 *
 * Returns ::CUPHY_STATUS_INVALID_ARGUMENT if \p pRssiDynDescrSizeBytes and/or \p pRssiDynDescrAlignBytes and/or \p pRsrpDynDescrSizeBytes and/or \p pRsrpDynDescrAlignBytes is NULL
 *
 * Returns ::CUPHY_STATUS_SUCCESS if allocation and initialization were
 * successful.
 *
 * \param pRssiDynDescrSizeBytes   - Size in bytes of RSSI dynamic descriptor
 * \param pRssiDynDescrAlignBytes  - Alignment in bytes of RSSI dynamic descriptor
 * \param pRsrpDynDescrSizeBytes   - Size in bytes of RSRP dynamic descriptor
 * \param pRsrpDynDescrAlignBytes  - Alignment in bytes of RSRP dynamic descriptor
 *
 * \return
 * ::CUPHY_STATUS_SUCCESS,
 * ::CUPHY_STATUS_INVALID_ARGUMENT
 *
 * \sa ::cuphyStatus_t,::cuphyGetErrorName,::cuphyGetErrorString,::cuphyCreatePuschRxRssi,::cuphyDestroyPuschRxRssi
 */
cuphyStatus_t CUPHYWINAPI cuphyPuschRxRssiGetDescrInfo(size_t* pRssiDynDescrSizeBytes,
                                                       size_t* pRssiDynDescrAlignBytes,
                                                       size_t* pRsrpDynDescrSizeBytes,
                                                       size_t* pRsrpDynDescrAlignBytes);

/******************************************************************/ /**
 * \brief Allocate and initialize a cuPHY PuschRx RSSI, RSRP estimation object
 *
 * Allocates a cuPHY RSSI (Received Signal Strength Indicator) and RSRP (Reference Signal Received Power)  object and returns a handle in the
 * address provided by the caller.
 *
 * Returns ::CUPHY_STATUS_INVALID_ARGUMENT if \p pPuschRxRssiHndl is NULL.
 *
 * Returns ::CUPHY_STATUS_ALLOC_FAILED if a PuschRxRssi object cannot be allocated
 *
 * Returns ::CUPHY_STATUS_SUCCESS if allocation and initialization were successful
 *
 * \param pPuschRxRssiHndl - Address to return the new PuschRxRssi instance
 *
 * \return
 * ::CUPHY_STATUS_SUCCESS,
 * ::CUPHY_STATUS_ALLOC_FAILED,
 * ::CUPHY_STATUS_INVALID_ARGUMENT
 *
 * \sa ::cuphyStatus_t,::cuphyGetErrorName,::cuphyGetErrorString,::cuphyPuschRxRssiGetDescrInfo,::cuphySetupPuschRxRssi,::cuphyDestroyPuschRxRssi
 */
cuphyStatus_t CUPHYWINAPI cuphyCreatePuschRxRssi(cuphyPuschRxRssiHndl_t* pPuschRxRssiHndl);

/******************************************************************/ /**
 * \brief Setup cuPHY PuschRx RSSI for slot processing
 *
 * Setup cuPHY PUSCH Received Signal Strength Indicator in preparation towards slot execution
 *
 * Returns ::CUPHY_STATUS_SUCCESS if setup is successful.
 *
 * Returns ::CUPHY_STATUS_INVALID_ARGUMENT if \p puschRxRssiHndl
 * and/or \p pDynDescrsCpu and/or \p pDynDescrsGpu and/or \p pLaunchCfgs is NULL.
 *
 * \param puschRxRssiHndl             - Handle to previously created PuschRxRssi instance
 * \param pDrvdUeGrpPrmsCpu           - Pointer to derived UE groups parameters in CPU memory
 * \param pDrvdUeGrpPrmsGpu           - Pointer to derived UE groups parameters in GPU memory
 * \param nUeGrps                     - number of UE groups to be processed
 * \param nMaxPrb                     - maximum number of PRBs across UE groups
 * \param dmrsSymbolIdx               - Index of DMRS symbols to be processed
 * \param enableCpuToGpuDescrAsyncCpy - Flag when set enables async copy of CPU descriptor into GPU
 * \param pDynDescrsCpu               - Pointer to dynamic descriptor in CPU memory
 * \param pDynDescrsGpu               - Pointer to dynamic descriptor in GPU memory
 * \param pLaunchCfgs                 - Pointer to channel estimation launch configurations
 * \param strm                        - CUDA stream for descriptor copy operation
 *
 * \return
 * ::CUPHY_STATUS_SUCCESS,
 * ::CUPHY_STATUS_INVALID_ARGUMENT
 *
 * \sa ::cuphyStatus_t,::cuphyGetErrorName,::cuphyGetErrorString,::cuphyPuschRxRssiGetDescrInfo,::cuphyCreatePuschRxRssi,,::cuphySetupPuschRxRsrp,::cuphyDestroyPuschRxRssi
 */
cuphyStatus_t CUPHYWINAPI
cuphySetupPuschRxRssi(cuphyPuschRxRssiHndl_t        puschRxRssiHndl,
                      cuphyPuschRxUeGrpPrms_t*      pDrvdUeGrpPrmsCpu,
                      cuphyPuschRxUeGrpPrms_t*      pDrvdUeGrpPrmsGpu,
                      uint16_t                      nUeGrps,
                      uint32_t                      nMaxPrb,
                      uint8_t                       dmrsSymbolIdx,
                      uint8_t                       enableCpuToGpuDescrAsyncCpy,
                      void*                         pDynDescrsCpu,
                      void*                         pDynDescrsGpu,
                      cuphyPuschRxRssiLaunchCfgs_t* pLaunchCfgs,
                      cudaStream_t                  strm);

/******************************************************************/ /**
 * \brief Setup cuPHY PuschRx RSRP for slot processing
 *
 * Setup cuPHY PUSCH Reference Signal Received Power in preparation towards slot execution
 *
 * Returns ::CUPHY_STATUS_SUCCESS if setup is successful.
 *
 * Returns ::CUPHY_STATUS_INVALID_ARGUMENT if \p puschRxRssiHndl
 * and/or \p pDynDescrsCpu and/or \p pDynDescrsGpu and/or \p pLaunchCfgs is NULL.
 *
 * \param puschRxRssiHndl             - Handle to previously created PuschRxRssi instance
 * \param pDrvdUeGrpPrmsCpu           - Pointer to derived UE groups parameters in CPU memory
 * \param pDrvdUeGrpPrmsGpu           - Pointer to derived UE groups parameters in GPU memory
 * \param nUeGrps                     - number of UE groups to be processed
 * \param nMaxPrb                     - maximum number of PRBs across UE groups
 * \param dmrsSymbolIdx               - Index of DMRS symbols to be processed
 * \param enableCpuToGpuDescrAsyncCpy - Flag when set enables async copy of CPU descriptor into GPU
 * \param pDynDescrsCpu               - Pointer to dynamic descriptor in CPU memory
 * \param pDynDescrsGpu               - Pointer to dynamic descriptor in GPU memory
 * \param pLaunchCfgs                 - Pointer to channel estimation launch configurations
 * \param strm                        - CUDA stream for descriptor copy operation
 *
 * \return
 * ::CUPHY_STATUS_SUCCESS,
 * ::CUPHY_STATUS_INVALID_ARGUMENT
 *
 * \sa ::cuphyStatus_t,::cuphyGetErrorName,::cuphyGetErrorString,::cuphyPuschRxRssiGetDescrInfo,::cuphyCreatePuschRxRssi,::cuphySetupPuschRxRssi,::cuphyDestroyPuschRxRssi
 */
cuphyStatus_t CUPHYWINAPI
cuphySetupPuschRxRsrp(cuphyPuschRxRssiHndl_t        puschRxRssiHndl,
                      cuphyPuschRxUeGrpPrms_t*      pDrvdUeGrpPrmsCpu,
                      cuphyPuschRxUeGrpPrms_t*      pDrvdUeGrpPrmsGpu,
                      uint16_t                      nUeGrps,
                      uint32_t                      nMaxPrb,
                      uint8_t                       dmrsSymbolIdx,
                      uint8_t                       enableCpuToGpuDescrAsyncCpy,
                      void*                         pDynDescrsCpu,
                      void*                         pDynDescrsGpu,
                      cuphyPuschRxRsrpLaunchCfgs_t* pLaunchCfgs,
                      cudaStream_t                  strm);

/******************************************************************/ /**
 * \brief Destroys a cuPHY PUSCH RSSI estimation object
 *
 * Destroys a cuPHY PUSCH RSSI (Received Signal Strength Indicator) and RSRP object that was previously
 * created by ::cuphyCreatePuschRxRssi. The handle provided to this function
 * should not be used for any operations after this function returns.
 *
 * Returns ::CUPHY_STATUS_INVALID_ARGUMENT if \p puschRxRssiHndl is NULL.
 *
 * Returns ::CUPHY_STATUS_SUCCESS if destruction was successful.
 *
 * \param puschRxRssiHndl - handle to previously allocated PuschRxRssi instance
 *
 * \return
 * ::CUPHY_STATUS_SUCCESS,
 * ::CUPHY_STATUS_INVALID_ARGUMENT
 *
 * \sa ::cuphyStatus_t,::cuphyGetErrorName,::cuphyGetErrorString,::cuphyPuschRxRssiGetDescrInfo,::cuphyCreatePuschRxRssi,::cuphySetupPuschRxRssi
 */
cuphyStatus_t CUPHYWINAPI cuphyDestroyPuschRxRssi(cuphyPuschRxRssiHndl_t puschRxRssiHndl);

/** @} */ /* END CUPHY_PUSCH_RSSI */

/**
 * \defgroup CUPHY_CRC CRC Computation
 *
 * This section describes the CRC computation functions of the cuPHY
 * application programming interface.
 *
 * @{
 */
struct PerTbParams;
struct PdschPerTbParams;

// CRC Launch descriptor

typedef struct
{
    dim3               cbKernelBlockDim;
    dim3               cbKernelGridDim;
    dim3               tbKernelBlockDim;
    dim3               tbKernelGridDim;
    uint32_t*          outputCBCRCs;
    uint8_t*           outputTBs;
    const uint32_t*    inputCodeBlocks;
    uint32_t*          outputTBCRCs;
    const PerTbParams* tbPrmsArray;
    uint8_t            reverseBytes;

} crcLaunchDescriptor;

// Graphs API
#define N_CRC_DECODE_GRAPH_NODES 2

void createCRCDecodeNodes(cudaGraphNode_t crcNodes[N_CRC_DECODE_GRAPH_NODES], cudaGraph_t graph, const cudaGraphNode_t* dependencies, uint32_t nDependencies, const crcLaunchDescriptor* crcDesc);
void updateCRCDecodeNodes(cudaGraphNode_t crcNodes[N_CRC_DECODE_GRAPH_NODES], cudaGraphExec_t graphExec, const crcLaunchDescriptor* crcDesc);

// populates crcLaunchDescriptor
cuphyStatus_t cuphyCRCDecodeLaunchSetup(
    uint32_t             nTBs,          // total number of input transport blocks
    uint32_t             maxNCBsPerTB,  // Maximum number of code blocks per transport block for current launch
    uint32_t             maxTBByteSize, // Maximum size in bytes of transport block for current launch
    crcLaunchDescriptor* crcDecodeDesc);

cuphyStatus_t cuphyCRCDecode(
    /* DEVICE MEMORY*/
    uint32_t* d_outputCBCRCs, // output buffer containing result of CRC check for each input code block (one uint32_t value per code block): 0 if the CRC check passed, a value different than zero otherwise
    uint32_t* d_outputTBCRCs, // output buffer containing result of CRC check for each input transport block (one uint32_t value per transport block): 0 if the CRC check passed, a value different than zero otherwise

    uint8_t*           d_outputTransportBlocks, // output buffer containing the information bytes of each input transport block
    const uint32_t*    d_inputCodeBlocks,       // input buffer containing the input code blocks
    const PerTbParams* d_tbPrmsArray,           // array of PerTbParams structs describing each input transport block
    /* END DEVICE MEMORY*/
    uint32_t     nTBs,           // total number of input transport blocks
    uint32_t     maxNCBsPerTB,   // Maximum number of code blocks per transport block for current launch
    uint32_t     maxTBByteSize,  // Maximum size in bytes of transport block for current launch
    int          reverseBytes,   // reverse order of bytes in each word before computing the CRC
    int          timeIt,         // run NRUNS times and report average running time
    uint32_t     NRUNS,          // number of iterations used to compute average running time
    uint32_t     codeBlocksOnly, // Only compute CRC of code blocks. Skip transport block CRC computation
    cudaStream_t strm);

struct cuphyPrepareCrcEncodeLaunchConfig
{
    CUDA_KERNEL_NODE_PARAMS m_kernelNodeParams;
    void*                   m_desc;
    void*                   m_kernelArgs[1];
};
typedef struct cuphyPrepareCrcEncodeLaunchConfig* cuphyPrepareCrcEncodeLaunchConfig_t;

/** @brief: Prepare CRC Encoder
 *
 * @param[in] prepareCrcEncodeLaunchConfig:  Launch config for CRC encoder
 * @param[in] d_inputOrigTBs: Array containing input
 * @param[in] d_inputTBs:     Array containing input after preparation
 * @param[in] d_inputTBsTM:   Array containing input after preparation for testing mode
 * @param[in] d_tbPrmsArray:  Array of PdschPerTbParams structs describing each input transport block
 * @param[in] nTBs:           Total number of input transport blocks
 * @param[in] maxNCBsPerTB:   Maximum number of code blocks per transport block for current launch
 * @param[in] maxTbSizeBytes  Maximum number TB size in bytes
 * @param[in] cpu_desc:       Descriptor in CPU memory
 * @param[in] gpu_desc:       Descriptor in GPU memory
 * @param[in] enable_desc_async_copy: Flag to specify asynchronous copying of descriptor
 * @param[in] strm:           CUDA stream on which to run kernel
 * @return CUPHY_STATUS_SUCCESS or CUPHY_STATUS_INVALID_ARGUMENT
 */
cuphyStatus_t cuphySetupPrepareCRCEncode(
    cuphyPrepareCrcEncodeLaunchConfig_t prepareCrcEncodeLaunchConfig,
    const uint32_t* d_inputOrigTBs,
    uint32_t*       d_inputTBs,
    uint32_t*       d_inputTBsTM,

    const PdschPerTbParams* d_tbPrmsArray,
    uint32_t           nTBs,
    uint32_t           maxNCBsPerTB,
    uint32_t           maxTbSizeBytes,
    void*              cpu_desc,
    void*              gpu_desc,
    uint8_t            enable_desc_async_copy,
    cudaStream_t       strm);

/**
 * CRC Encoder kernel launch configs
 */
struct cuphyCrcEncodeLaunchConfig
{
    CUDA_KERNEL_NODE_PARAMS m_kernelNodeParams[2];
    void*                   m_desc;
    void*                   m_kernelArgs[1];
};
typedef struct cuphyCrcEncodeLaunchConfig* cuphyCrcEncodeLaunchConfig_t;

/** @brief: Compute descriptor size and alignment for CRC Encoder.
 *
 * @param[in,out] pDescrSizeBytes:  Size in bytes of descriptor
 * @param[in,out] pDescrAlignBytes: Alignment in bytes of descriptor
 * @return CUPHY_STATUS_SUCCESS or CUPHY_STATUS_INVALID_ARGUMENT
 */
cuphyStatus_t CUPHYWINAPI cuphyCrcEncodeGetDescrInfo(size_t* pDescrSizeBytes,
                                                     size_t* pDescrAlignBytes);

/**
 * @brief Compute descriptor size and alignment for CRC Encoder Preparation
 * @param[in,out] pDescrSizeBytes:  Size in bytes of descriptor
 * @param[in,out] pDescrAlignBytes: Alignment in bytes of descriptor
 * @return CUPHY_STATUS_SUCCESS or CUPHY_STATUS_INVALID_ARGUMENT
 */
cuphyStatus_t CUPHYWINAPI cuphyPrepareCrcEncodeGetDescrInfo(size_t* pDescrSizeBytes,
                                                            size_t* pDescrAlignBytes);

/** @brief: Setup CRC encoder component.
 *
 *  @param[in] crcEncodeLaunchConfig: Pointer to cuphyCrcEncodeLaunchConfig.
 *  @param[in, out] d_cbCRCs: if not nullptr, output buffer with per-CB CRCs across all TBs for debugging
 *  @param[in, out] d_tbCRCs: output buffer containing per-TB CRCs across all TBS (needed by CB kernel)
 *  @param[in] d_inputTransportBlocks: input buffer; currently prepared via cuphyPrepareCRCEncode
 *  @param[out] d_codeBlocks: CRC output
 *  @param[in,out] d_tbPrmsArray: array of PdschPerTbParams structs describing each input transport block.
 *  @param[in] nTBs: number of TBs handled in a kernel launch
 *  @param[in] maxNCBsPerTB: maximum number of code blocks per transport block for current launch
 *  @param[in] maxTBByteSize: maximum size in bytes of transport block for current launch
 *  @param[in] reverseBytes: reverse order of bytes in each word before computing CRC
 *  @param[in] codeBlocksOnly: only compute CRC of code blocks (CBs); skip transport block CRC computation.
 *  @param[in] cpu_desc: Pointer to descriptor in CPU memory
 *  @param[in] gpu_desc: Pointer to descriptor in GPU memory
 *  @param[in] enable_desc_async_copy: async copy CPU descriptor into GPU if set.
 *  @param[in] strm: CUDA stream for async copy
 *
 *  @return CUPHY_STATUS_SUCCESS or CUPHY_STATUS_INVALID_ARGUMENT
 */
cuphyStatus_t CUPHYWINAPI
cuphySetupCrcEncode(cuphyCrcEncodeLaunchConfig_t crcEncodeLaunchConfig,
                    uint32_t*                    d_cbCRCs,
                    uint32_t*                    d_tbCRCs,
                    const uint32_t*              d_inputTransportBlocks,
                    uint8_t*                     d_codeBlocks,
                    const PdschPerTbParams*      d_tbPrmsArray,
                    uint32_t                     nTBs,
                    uint32_t                     maxNCBsPerTB,
                    uint32_t                     maxTBByteSize,
                    uint8_t                      reverseBytes,
                    uint8_t                      codeBlocksOnly,
                    void*                        cpu_desc,
                    void*                        gpu_desc,
                    uint8_t                      enable_desc_async_copy,
                    cudaStream_t                 strm);

/** @} */ /* END CUPHY_CRC */

/**
 * \defgroup CUPHY_CRC_DECODE CRC Decode
 *
 * This section describes the puschRx CRC deocder (transport block + code block + segmentation)
 * functions of the cuPHY application programming interface.
 *
 * @{
 */

struct cuphyPuschRxCrcDecode
{
    // Add a reserved member to empty struct
    // so size is the same for C and C++
    uint8_t reserved;
};

/**
 * cuPHY Pusch Rate Match handle
 */

typedef struct cuphyPuschRxCrcDecode* cuphyPuschRxCrcDecodeHndl_t;

typedef struct
{
    CUDA_KERNEL_NODE_PARAMS kernelNodeParamsDriver;
    void*                   desc;
    void*                   kernelArgs[1];
} cuphyPuschRxCrcDecodeLaunchCfg_t;

/******************************************************************/ /**
 * \brief Helper to compute cuPHY crc decoder descriptor buffer sizes and alignments
 *
 * Computes cuPHY PUSCH crc decoder descriptor buffer sizes and alignments. To be used by the caller to
 * allocate these buffers (in CPU and GPU memories) and provide them to other PuschRxCrcDecode APIs
 *
 * Returns ::CUPHY_STATUS_INVALID_ARGUMENT if \p pDescrSizeBytes and/or \p pDescrAlignBytes
 *
 * Returns ::CUPHY_STATUS_SUCCESS if allocation and initialization were
 * successful.
 *
 * \param pDescrSizeBytes  - Size in bytes descriptor
 * \param pDescrAlignBytes - Alignment of descriptor
 *
 * \return
 * ::CUPHY_STATUS_SUCCESS,
 * ::CUPHY_STATUS_INVALID_ARGUMENT
 *
 * \sa ::cuphyStatus_t,::cuphyGetErrorName,::cuphyGetErrorString,::cuphyCreatePuschRxRateMatch,::cuphyDestroyPuschRxRateMatch
 */

cuphyStatus_t CUPHYWINAPI cuphyPuschRxCrcDecodeGetDescrInfo(size_t* pDescrSizeBytes,
                                                            size_t* pDescrAlignBytes);

/******************************************************************/ /**
 * \brief Allocate and initialize a cuPHY PuschRx crc decode object
 *
 * Allocates a cuPHY pusch crc decode object and returns a handle in the
 * address provided by the caller.
 *
 * Returns ::CUPHY_STATUS_INVALID_ARGUMENT if \p puschRxCrcDecodeHndl and/or \p FPconfig is NULL.
 *
 * Returns ::CUPHY_STATUS_ALLOC_FAILED if a PuschRxCrcDecode object cannot be allocated
 *
 * Returns ::CUPHY_STATUS_SUCCESS if allocation and initialization were successful
 *
 * \param puschRxCrcDecodeHndl     - Address to return the new PuschRxRateMatch instance
 * \param reverseBytes             - 0 or 1. Option to reverse bytes during crc.

 *
 * \return
 * ::CUPHY_STATUS_SUCCESS,
 * ::CUPHY_STATUS_ALLOC_FAILED,
 * ::CUPHY_STATUS_INVALID_ARGUMENT
 *
 * \sa ::cuphyStatus_t,::cuphyGetErrorName,::cuphyGetErrorString,::cuphyPuschRxCrcDecodeGetDescrInfo,::cuphySetupPuschRxCrcDecode,::cuphyDestroyPuschRxCrcDecode
 */

cuphyStatus_t CUPHYWINAPI cuphyCreatePuschRxCrcDecode(cuphyPuschRxCrcDecodeHndl_t* puschRxCrcDecodeHndl,
                                                      int                          reverseBytes);

/******************************************************************/ /**
 * \brief Setup cuPHY crc decode for slot processing
 *
 * Setup cuPHY PUSCH crc decode in preparation towards slot execution
 *
 * Returns ::CUPHY_STATUS_SUCCESS if setup is successful.
 *
 * Returns ::CUPHY_STATUS_INVALID_ARGUMENT if any inputs NULL.
 *
 * \param puschRxCrcDecodeHndl        - Address to return the PuschRxCrcDecode instance
 * \param nSchUes                     - number of users with sch data
 * \param pSchUserIdxsCpu             - Address of sch user indicies
 * \param pOutputCBCRCs               - Address of where to strore CB crc results
 * \param pOutputTBs                  - Address of where to store estimated transport blocks (w/h crc removed)
 * \param pInputCodeBlocks            - Address of input codeblocks (output of LDPC)
 * \param pOutputTBCRCs               - Address of where to stroe TB crc results
 * \param pTbPrmsCpu                  - Address of tb parameters in CPU
 * \param pTbPrmsGpu                  - Address of tb parameters in GPU
 * \param pCpuDesc                    - Address of descriptor in CPU
 * \param pGpuDesc                    - Address of descriptor in GPU
 * \param enableCpuToGpuDescrAsyncCpy - Option to copy desc from CPU to GPU
 * \param pCbCrcLaunchCfg             - Address of CB CRC decoder launch configuration
 * \param pTbCrcLaunchCfg             - Address of TB CRC decoder launch configuration
 * \param strm                        - stream to perform copy
 *
 * \return
 * ::CUPHY_STATUS_SUCCESS,
 * ::CUPHY_STATUS_INVALID_ARGUMENT
 *
 * \sa ::cuphyStatus_t,::cuphyGetErrorName,::cuphyGetErrorString,::cuphyPuschRxCrcDecodeGetDescrInfo,::cuphyCreatePuschRxCrcDecode,::cuphyDestroyPuschRxCrcDecode
 */

cuphyStatus_t CUPHYWINAPI cuphySetupPuschRxCrcDecode(cuphyPuschRxCrcDecodeHndl_t       puschRxCrcDecodeHndl,
                                                     uint16_t                          nSchUes,
                                                     uint16_t*                         pSchUserIdxsCpu,
                                                     uint32_t*                         pOutputCBCRCs,
                                                     uint8_t*                          pOutputTBs,
                                                     const uint32_t*                   pInputCodeBlocks,
                                                     uint32_t*                         pOutputTBCRCs,
                                                     const PerTbParams*                pTbPrmsCpu,
                                                     const PerTbParams*                pTbPrmsGpu,
                                                     void*                             pCpuDesc,
                                                     void*                             pGpuDesc,
                                                     uint8_t                           enableCpuToGpuDescrAsyncCpy,
                                                     cuphyPuschRxCrcDecodeLaunchCfg_t* pCbCrcLaunchCfg,
                                                     cuphyPuschRxCrcDecodeLaunchCfg_t* pTbCrcLaunchCfg,
                                                     cudaStream_t                      strm);

/******************************************************************/ /**
 * \brief Destroys a cuPHY PUSCH crc decode object
 *
 * Destroys a cuPHY PUSCH crc decode object that was previously
 * created by ::cuphyCreatePuschRxCrcDecode. The handle provided to this function
 * should not be used for any operations after this function returns.
 *
 * Returns ::CUPHY_STATUS_INVALID_ARGUMENT if \p puschRxCrcDecodeHndl is NULL.
 *
 * Returns ::CUPHY_STATUS_SUCCESS if destruction was successful.
 *
 * \param puschRxCrcDecodeHndl - handle to previously allocated PuschRxRateMatch instance
 *
 * \return
 * ::CUPHY_STATUS_SUCCESS,
 * ::CUPHY_STATUS_INVALID_ARGUMENT
 *
 * \sa ::cuphyStatus_t,::cuphyGetErrorName,::cuphyGetErrorString,::cuphyPuschRxCrcDecodeGetDescrInfo,::cuphyCreatePuschRxCrcDecode,::cuphySetupPuschRxCrcDecode
 */

cuphyStatus_t CUPHYWINAPI cuphyDestroyPuschRxCrcDecode(cuphyPuschRxCrcDecodeHndl_t puschRxCrcDecodeHndl);

/** @} */ /* END CUPHY_CRC_DECODE */

/**
 * \defgroup CUPHY_SCRAMBLE Scrambling/Descrambling
 *
 * This section describes the scrambling/descrambling functions of the cuPHY
 * application programming interface.
 *
 * @{
 */

// FIXME: this is a stand-alone implementation of descrambling used as initial reference.
// It will be removed eventually
void cuphyDescrambleInit(void** descrambleEnv);

void cuphyDescrambleCleanUp(void** descrambleEnv);

cuphyStatus_t cuphyDescrambleLoadParams(void**          descrambleEnv,
                                        uint32_t        nTBs,
                                        uint32_t        maxNCodeBlocks,
                                        const uint32_t* tbBoundaryArray,
                                        const uint32_t* cinitArray);

cuphyStatus_t cuphyDescrambleLoadInput(void** descrambleEnv,
                                       float* llrs);

cuphyStatus_t cuphyDescramble(void**       descrambleEnv,
                              float*       d_llrs,
                              bool         timeIt,
                              uint32_t     NRUNS,
                              cudaStream_t strm);

cuphyStatus_t cuphyDescrambleStoreOutput(void** descrambleEnv,
                                         float* llrs);

cuphyStatus_t cuphyDescrambleAllParams(float*          llrs,
                                       const uint32_t* tbBoundaryArray,
                                       const uint32_t* cinitArray,
                                       uint32_t        nTBs,
                                       uint32_t        maxNCodeBlocks,
                                       int             timeIt,
                                       uint32_t        NRUNS,
                                       cudaStream_t    stream);
/** @} */ /* END CUPHY_SCRAMBLE */

/**
 * \defgroup CUPHY_RATE_MATCHING Rate Matching
 *
 * This section describes the rate matching functions of the cuPHY
 * application programming interface.
 *
 * @{
 */

struct cuphyPuschRxRateMatch
{
    // Add a reserved member to empty struct
    // so size is the same for C and C++
    uint8_t reserved;
};

/**
 * cuPHY Pusch Rate Match handle
 */

typedef struct cuphyPuschRxRateMatch* cuphyPuschRxRateMatchHndl_t;

typedef struct
{
    CUDA_KERNEL_NODE_PARAMS kernelNodeParamsDriver;          // Main de_rate_matching_global2 kernel
    CUDA_KERNEL_NODE_PARAMS resetKernelNodeParamsDriver;     // Reset buffer kernel (launched first)
    CUDA_KERNEL_NODE_PARAMS clampKernelNodeParamsDriver;     // Clamp buffer kernel (launched last)
    void*                   desc;
    void*                   kernelArgs[1];
} cuphyPuschRxRateMatchLaunchCfg_t;

/******************************************************************/ /**
 * \brief Helper to compute cuPHY rate match descriptor buffer sizes and alignments
 *
 * Computes cuPHY PUSCH rate match descriptor buffer sizes and alignments. To be used by the caller to
 * allocate these buffers (in CPU and GPU memories) and provide them to other PuschRxRateMatch APIs
 *
 * Returns ::CUPHY_STATUS_INVALID_ARGUMENT if \p pDescrSizeBytes and/or \p pDescrAlignBytes
 *
 * Returns ::CUPHY_STATUS_SUCCESS if allocation and initialization were
 * successful.
 *
 * \param pDescrSizeBytes  - Size in bytes descriptor
 * \param pDescrAlignBytes - Alignment of descriptor
 *
 * \return
 * ::CUPHY_STATUS_SUCCESS,
 * ::CUPHY_STATUS_INVALID_ARGUMENT
 *
 * \sa ::cuphyStatus_t,::cuphyGetErrorName,::cuphyGetErrorString,::cuphyCreatePuschRxRateMatch,::cuphyDestroyPuschRxRateMatch
 */

cuphyStatus_t CUPHYWINAPI cuphyPuschRxRateMatchGetDescrInfo(size_t* pDescrSizeBytes,
                                                            size_t* pDescrAlignBytes);

/******************************************************************/ /**
 * \brief Allocate and initialize a cuPHY PuschRx rate match object
 *
 * Allocates a cuPHY pusch rate match object and returns a handle in the
 * address provided by the caller.
 *
 * Returns ::CUPHY_STATUS_INVALID_ARGUMENT if \p puschRxRateMatchHndl is NULL.
 *
 * Returns ::CUPHY_STATUS_ALLOC_FAILED if a PuschRxRateMatch object cannot be allocated
 *
 * Returns ::CUPHY_STATUS_SUCCESS if allocation and initialization were successful
 *
 * \param puschRxRateMatchHndl     - Address to return the new PuschRxRateMatch instance
 * \param FPconfig                 -0: FP32 in, FP32 out; 1: FP16 in, FP32 out; 2: FP32 in, FP16 out; 3: FP16 in, FP16 out; other values: invalid
 * \param descramblingOn           - enable/disable descrambling

 *
 * \return
 * ::CUPHY_STATUS_SUCCESS,
 * ::CUPHY_STATUS_ALLOC_FAILED,
 * ::CUPHY_STATUS_INVALID_ARGUMENT
 *
 * \sa ::cuphyStatus_t,::cuphyGetErrorName,::cuphyGetErrorString,::cuphyPuschRxRateMatchGetDescrInfo,::cuphySetupPuschRxRateMatch,::cuphyDestroyPuschRxRateMatch
 */

cuphyStatus_t CUPHYWINAPI cuphyCreatePuschRxRateMatch(cuphyPuschRxRateMatchHndl_t* puschRxRateMatchHndl,
                                                      int                          FPconfig,
                                                      int                          descramblingOn);

/******************************************************************/ /**
 * \brief Setup cuPHY rate match for slot processing
 *
 * Setup cuPHY PUSCH rate match in preparation towards slot execution
 *
 * Returns ::CUPHY_STATUS_SUCCESS if setup is successful.
 *
 * Returns ::CUPHY_STATUS_INVALID_ARGUMENT if \p puschRxRateMatchHndl and/or \p pSchUserIdxsCpu and/or \p pTbPrmsCpu and/or
 * \p pTbPrmsGpu and/or \p pTPrmRmIn and/or \p pTPrmCdm1RmIn and/or \p ppRmOut and/or \p pCpuDesc and/or \p pGpuDesc and/or \p pLaunchCfg is NULL.
 *
 * \param puschRxRateMatchHndl             - handle to rate-matching class
 * \param nSchUes                          - number of users with sch data
 * \param pSchUserIdxsCpu                  - Address of sch user indicies
 * \param pTbPrmsCpu                       - starting adress of transport block paramters (CPU)
 * \param pTbPrmsGpu                       - starting adress of transport block paramters (GPU)
 * \param pTPrmRmIn                        - starting adress of input LLR tensor parameters
 * \param pTPrmCdm1RmIn                    - starting adress of input LLR tensor (for CDM=1) parameters
 * \param ppRmOut                          - array of rm outputs, one per transport block (GPU)
 * \param pCpuDesc                         - pointer to descriptor in cpu
 * \param pGpuDesc                         - pointer to descriptor in gpu
 * \param enableCpuToGpuDescrAsyncCpy      - option to copy cpu descriptors from cpu to gpu
 * \param pLaunchCfg                       - pointer to rate matching launch configuration
 * \param strm                             - stream to perform copy

 *
 * \return
 * ::CUPHY_STATUS_SUCCESS,
 * ::CUPHY_STATUS_INVALID_ARGUMENT
 *
 * \sa ::cuphyStatus_t,::cuphyGetErrorName,::cuphyGetErrorString,::cuphyPuschRxRateMatchGetDescrInfo,::cuphyCreatePuschRxRateMatch,::cuphyDestroyPuschRxRateMatch
 */

cuphyStatus_t CUPHYWINAPI cuphySetupPuschRxRateMatch(cuphyPuschRxRateMatchHndl_t       puschRxRateMatchHndl,
                                                     uint16_t                          nSchUes,
                                                     uint16_t*                         pSchUserIdxsCpu,
                                                     const PerTbParams*                pTbPrmsCpu,
                                                     const PerTbParams*                pTbPrmsGpu,
                                                     cuphyTensorPrm_t*                 pTPrmRmIn,
                                                     cuphyTensorPrm_t*                 pTPrmCdm1RmIn,
                                                     void**                            ppRmOut,
                                                     void*                             pCpuDesc,
                                                     void*                             pGpuDesc,
                                                     uint8_t                           enableCpuToGpuDescrAsyncCpy,
                                                     cuphyPuschRxRateMatchLaunchCfg_t* pLaunchCfg,
                                                     cudaStream_t                      strm);

/******************************************************************/ /**
 * \brief Destroys a cuPHY PUSCH rate match object
 *
 * Destroys a cuPHY PUSCH rate match object that was previously
 * created by ::cuphyCreatePuschRxRateMatch. The handle provided to this function
 * should not be used for any operations after this function returns.
 *
 * Returns ::CUPHY_STATUS_INVALID_ARGUMENT if \p puschRxRateMatchHndl is NULL.
 *
 * Returns ::CUPHY_STATUS_SUCCESS if destruction was successful.
 *
 * \param puschRxRateMatchHndl - handle to previously allocated PuschRxRateMatch instance
 *
 * \return
 * ::CUPHY_STATUS_SUCCESS,
 * ::CUPHY_STATUS_INVALID_ARGUMENT
 *
 * \sa ::cuphyStatus_t,::cuphyGetErrorName,::cuphyGetErrorString,::cuphyPuschRxRateMatchGetDescrInfo,::cuphyCreatePuschRxRateMatch,::cuphySetupPuschRxRateMatch
 */

cuphyStatus_t CUPHYWINAPI cuphyDestroyPuschRxRateMatch(cuphyPuschRxRateMatchHndl_t puschRxRateMatchHndl);

/** @} */ /* END CUPHY_RATE_MATCHING */

/**
 * \defgroup DL_CUPHY_RATE_MATCHING DL Rate Matching
 *
 * This section describes the downlink rate matching functions of the cuPHY
 * application programming interface.
 *
 * @{
 */

/**
 * @brief Struct that tracks configuration information at a per TB (Transport Block) granularity
 */
struct PerTbParams
{
    uint32_t ndi;     /*!< Indicates if this is new data or a retransmission, 0=retransmission, 1=new data; is updated in setupCmnPhase2() after HARQ buffer allocation instead of setupCmnPhase1() */
    uint32_t rv;      /*!< redundancy version per TB; [0, 3] */
    uint32_t Qm;      /*!< modulation order per TB: [2, 4, 6, 8] */
    uint32_t bg;      /*!< base graph per TB; options are 1 or 2 */
    uint32_t Nl;      /*!< number of transmission layers per TB; [1, MAX_DL_LAYERS_PER_TB] in DL */
    uint32_t num_CBs; /*!< number of code blocks (CBs) per TB */
    uint32_t Zc;      /*!< lifting factor per TB */

    uint32_t N;            /*!< # bits in code block per TB */
    uint32_t Ncb;          /*!< same as N for now */
    uint32_t Ncb_padded;   /*!< Ncb w/ padding for LDPC decoder alignment requirements */
    uint32_t G;            /*!< number of rate-matched bits available for TB transmission */
    uint32_t K;            /*!< non punctured systematic bits */
    uint32_t F;            /*!< filler bits */
    uint32_t cinit;        /*!< used to generate scrambling sequence; seed2 arg. of gold32 */
    uint32_t nDataBytes;   /*!< number of data bytes in transport block (no CRCs) */
    uint32_t nZpBitsPerCb; /*!< number of zero padded encoded bits per codeblock (input to LDPC decoder) */
    uint32_t k0;           /*!< k0 from TS 38.212 rate matching, depends on {bg, rv, Zc, Ncb} */

    uint8_t  ldpcEarlyTerminationPerUe;
    uint8_t  ldpcMaxNumItrPerUe;


    uint32_t firstCodeBlockIndex;                         /// for symbol-by-symbol processing
    uint32_t encodedSize;                                 /// Size in bytes of encoded Tb
    uint32_t layer_map_array[MAX_N_LAYERS_PUSCH];         /*!< first Nl elements of array specify the layer(s) this TB maps to. TODO potentially convert to bitmap. */
    uint32_t userGroupIndex;                              /// user group/cell index
    uint32_t nBBULayers;                                  /// number of BBU layers for current user group/cell
    uint32_t startLLR;                                    /// start LLR index for transport block

    /// uci parameters
    uint32_t mScUciSum;      /// Total number of REs available for UCI transmission
    uint32_t codedBitsSum;   /// Summation of K_r, r=0,..,(C_ULSCH-1), the denominator of first term in Q'_ACK [Ref. TS 38.212 Sec. 6.3.2.4.1.1]
    uint8_t  isDataPresent;  /// Bit0 = 1 in pduBitmap,if data is present
    uint8_t  betaOffsetCsi2; /// Beta offset of CSI Part 2 [TS 38.213 Table 9.3-2] ==> FAPI parameter
    uint32_t qPrimeAck;      /// Rate matched output sequence length for HARQ-ACK payload
    float    codeRate;
    float    alpha;
    uint32_t qPrimeCsi1;

    uint8_t  isEarlyHarq;    /// indicates if HARQ UCI to be decoded early
    uint8_t  uciOnPuschFlag; /// indicates if uci on pusch
    uint8_t  csi2Flag;       /// indicates if CSI2 present
    uint32_t G_schAndCsi2;   /// number of SCH + CSI2 rate matched bits
    uint32_t G_harq;         /// number of harq rate matched bits
    uint32_t G_csi1;         /// number of csi part 1 rate matched bits
    uint32_t G_csi2;         /// number of csi part 2 rate matched bits
    uint32_t G_harq_rvd;     /// number of harq reserved bits
    uint32_t nBitsHarq;
    uint16_t nBitsCsi2;
    uint8_t  nCsiReports;
    uint8_t  rankBitOffset;
    uint8_t  nRanksBits;
    __half*  d_schAndCsi2LLRs;
    __half*  d_csi1LLRs;
    __half*  d_harqLLrs;
    uint32_t tbSize;

    // FAPIv3 method for computing CSI2 size:
    uint16_t               nCsi2Reports;
    cuphyCalcCsi2SizePrm_t calcCsi2SizePrms[CUPHY_MAX_N_CSI2_REPORTS_PER_UE];

    uint8_t enableTfPrcd;
    uint8_t nDmrsCdmGrpsNoData;

    /// Debugging
    uint32_t* debug_d_derateCbsIndices;
};

/**
 * @brief Struct that tracks configuration information at a per TB (Transport Block) granularity
 *        for the downlink shared channel (PDSCH).
 */

struct PdschPerTbParams
{
    // For prepareCRC
    const uint8_t* tbStartAddr;
    uint32_t tbStartOffset;
    uint32_t tbSize;                   /*!< TB size in bytes */
    uint32_t cumulativeTbSizePadding;
    uint8_t  testModel; /*!< TB's cell not in testing mode if 0; 1 otherwise. When 1, tbSize holds the length of PN23 (pseudorandom sequence). */

    uint8_t rv;      /*!< redundancy version per TB; [0, 3] */
    uint8_t Qm;      /*!< modulation order per TB: [2, 4, 6, 8] */
    uint8_t bg;      /*!< base graph per TB; options are 1 or 2 */
    uint8_t Nl;      /*!< number of transmission layers per TB; [1, MAX_DL_LAYERS_PER_TB] in DL */
    uint32_t num_CBs; /*!< number of code blocks (CBs) per TB */
    uint32_t Zc;      /*!< lifting factor per TB */

    uint32_t N;            /*!< # bits in code block per TB */
    uint32_t Ncb;          /*!< same as N for now */
    uint32_t G;            /*!< number of rate-matched bits available for TB transmission without accounting punctured REs due to CSI-RS (max_G) */
    uint32_t max_REs;      /*!< number of REs for TB transmission without accounting punctured REs due to CSI-RS. It's G /(Qm * Nl) */
    uint32_t K;            /*!< non punctured systematic bits */
    uint32_t F;            /*!< filler bits */
    uint32_t cinit;        /*!< used to generate scrambling sequence; seed2 arg. of gold32 */

    uint32_t firstCodeBlockIndex;                         // for symbol-by-symbol processing (not currently used in PDSCH, always set to 0)
};


/**
 * @brief Update PdschPerTbParams struct that tracks configuration information at per TB
 *        granularity. Check that configuration values are valid.
 * @param[in,out] tb_params_struct: pointer to a PerTbParams configuration struct
 * @param[in] cfg_rv: redundancy version
 * @param[in] cfg_Qm: modulation order
 * @param[in] cfg_bg: base graph
 * @param[in] cfg_Nl: number of layers per Tb (at most MAX_DL_LAYERS_PER_TB for downlink)
 * @param[in] cfg_num_CBs: number of code blocks
 * @param[in] cfg_Zc: lifting factor
 * @param[in] cfg_G: number of rated matched bits available for TB transmission
 * @param[in] cfg_F: number of filler bits
 * @param[in] cfg_cinit: seed used for scrambling sequence
 * @param[in] cfg_Nref: used to determine Ncb if smaller than N
 * @return CUPHY_STATUS_SUCCESS or CUPHY_STATUS_INVALID_ARGUMENT.
 */
cuphyStatus_t cuphySetTBParams(PdschPerTbParams* tb_params_struct,
                               uint32_t     cfg_rv,
                               uint32_t     cfg_Qm,
                               uint32_t     cfg_bg,
                               uint32_t     cfg_Nl,
                               uint32_t     cfg_num_CBs,
                               uint32_t     cfg_Zc,
                               uint32_t     cfg_G,
                               uint32_t     cfg_F,
                               uint32_t     cfg_cinit,
                               uint32_t     cfg_Nref);

/** @brief: Return workspace size, in bytes, needed for all configuration parameters
 *          of the rate matching component. Does not allocate any space.
 *  @param[in] num_TBs: number of Transport blocks (TBs) to be processed within a kernel launch
 *  @return workspace size in bytes
 */
size_t cuphyDlRateMatchingWorkspaceSize(int num_TBs);

/**
 * DL Rate matching kernel launch configs
 */
struct cuphyDlRateMatchingLaunchConfig
{
    CUDA_KERNEL_NODE_PARAMS m_kernelNodeParams[2]; // rate-matching and restructuring kernels
    void*                   m_desc;
    void*                   m_kernelArgs[1];
};
typedef struct cuphyDlRateMatchingLaunchConfig* cuphyDlRateMatchingLaunchConfig_t;

/** @brief: Compute descriptor buffer size and alignment for rate matching.
 *
 * @param[in,out] pDescrSizeBytes:  Size in bytes of descriptor
 * @param[in,out] pDescrAlignBytes: Alignment in bytes of descriptor
 *
 * @return CUPHY_STATUS_SUCCESS or CUPHY_STATUS_INVALID_ARGUMENT
 */
cuphyStatus_t CUPHYWINAPI cuphyDlRateMatchingGetDescrInfo(size_t* pDescrSizeBytes,
                                                          size_t* pDescrAlignBytes);
struct PdschDmrsParams;
struct PdschUeGrpParams;

/**
 * PDSCH Status Types
 */
typedef enum _cuphyPdschStatusType
{
    CUPHY_PDSCH_STATUS_SUCCESS_OR_UNTRACKED_ISSUE = 0x0, /*!< Default value; reset on every cuPHY PDSCH setup */
    CUPHY_PDSCH_STATUS_UNSUPPORTED_MAX_ER_PER_CB  = 0x1, /*!< Set if at least one UE in the cell group has per CB Er > PDSCH_MAX_ER_PER_CB_BITS. Special handling in cuPHY-CP. */
    // More types can be added as needed
    CUPHY_MAX_PDSCH_STATUS_TYPES

} cuphyPdschStatusType_t;


/**
 * PDSCH Output Status
 */
typedef struct _cuphyPdschStatusOut
{
    cuphyPdschStatusType_t status; /* cuPHY PDSCH status after setup call. Currently used to highlight if CUPHY_PDSCH_STATUS_UNSUPPORTED_MAX_ER_PER_CB occured. */
    uint16_t cellPrmStatIdx;       /*!< cuphyPdschCellDynPrm_t.cellPrmStatiDX of first cell in the cell group that caused status != 0. Invalid value (MAX_UINT16) if status == 0 */
    uint16_t ueIdx;                /*!< index of first UE in the cell group that cause status != 0. Index is UE position in pUePrms or pCwPrms for that cell group. Invalid value (MAX_UINT16) if status == 0. */
} cuphyPdschStatusOut_t;

/** @brief: Setup rate matching component incl. kernel node params for rate-matching
 *          (incl. scrambling and layer mapping) and rate-matching output restructuring (if enabled).
 *          If enable_modulation is set, this component also performs modulation too.
 *
 *  @param[in] dlRateMatchingLaunchConfig: Pointer to cuphyDlRateMatchingLaunchConfig.
 *  @param[out] status: pointer to cuphyPdschStatusOut_t struct; updated if CB_Er > PDSCH_MAX_ER_PER_CB_BITS for any UE in cell group.
 *  @param[in] d_rate_matching_input: LDPC encoder's output; device buffer, previously allocated.
 *  @param[out] d_rate_matching_output: rate-matching output, with scrambling and layer-mapping, if enabled; device pointer, preallocated.
 *  @param[out] d_restructure_rate_matching_output: d_rate_matching_output restructured for modulation. There are Er bits per code block.
 *                                             Each layer starts at an uint32_t aligned boundary.
 *  @param[out] d_modulation_output: pointer to output tensor (preallocated)
 *                                 Each symbol is a complex number using half-precision for
 *                                 the real and imaginary parts. Update: no longer used; the cell_output_tensor_addr field of PdschDmrsParams is
 *                                 used instead.
 *  @param[in] d_xtf_re_map: RE (resource element) map array, relevant when CSI-RS symbols overlap with TB allocations.
 *                           Can set to nullptr if there is no such overlap.
 *  @param[in] max_PRB_BWP: maximum number of downlink PRBs for all cells whose TBs are processed here. Used to index into the d_xtf_re_map array.
 *  @param[in] num_TBs: number of TBs handled in a kernel launch
 *  @param[in] num_layers: number of layers
 *  @param[in] enable_scrambling: enable scrambling when 1, no scrambling when 0
 *  @param[in] enable_layer_mapping: enable layer mapping when 1, no layer mapping when 0
 *  @param[in] enable_modulation: run a fused rate matching and modulation kernel when 1; used in PDSCH pipeline.
 *  @param[in] precoding: 1 if any TB has precoding enabled; 0 otherwise.
 *  @param[in] restructure_kernel: set-up kernel node params for restructure kernel when 1.
 *  @param[in] batching: when enabled the TBs from this kernel launch can belong to different cells
 *  @param[in] h_workspace: pinned host memory for temporary buffers
 *  @param[in] d_workspace: device memory for h_workspace. The H2D copy from h_workspace to d_workspace happens within cuphySetupDlRateMatching if enable_desc_async_copy is set.
 *  @param[in] h_params: pointer to # TBs PdschPerTbParams struct; pinned host memory
 *  @param[in] d_params: pointer to device memory for h_params. The H2D copy from h_params to d_params happens *outside* cuphySetupDlRateMatching.
 *  @param[in] d_dmrs_params: pointer to PdschDmrs parameters on the device.
 *  @param[in] d_ue_grp_params: pointer to PdschUeGrpParams parameters on the device.
 *  @param[in] cpu_desc: Pointer to descriptor in CPU memory
 *  @param[in] gpu_desc: Pointer to descriptor in GPU memory
 *  @param[in] enable_desc_async_copy: async copy CPU descriptor into GPU if set.
 *  @param[in] strm: CUDA stream for async copy
 *
 *  @return CUPHY_STATUS_SUCCESS or CUPHY_STATUS_INVALID_ARGUMENT
 */
cuphyStatus_t CUPHYWINAPI
cuphySetupDlRateMatching(cuphyDlRateMatchingLaunchConfig_t dlRateMatchingLaunchConfig,
                         cuphyPdschStatusOut_t*            status,
                         const uint32_t*                   d_rate_matching_input,
                         uint32_t*                         d_rate_matching_output,
                         uint32_t*                         d_restructure_rate_matching_output,
                         void*                             d_modulation_output,
                         void*                             d_xtf_re_map,
                         uint16_t                          max_PRB_BWP,
                         int                               num_TBs,
                         int                               num_layers,
                         uint8_t                           enable_scrambling,
                         uint8_t                           enable_layer_mapping,
                         uint8_t                           enable_modulation,
                         uint8_t                           precoding,
                         uint8_t                           restructure_kernel,
                         uint8_t                           batching,
                         uint32_t*                         h_workspace,
                         uint32_t*                         d_workspace,
                         PdschPerTbParams*                 h_params,
                         PdschPerTbParams*                 d_params,
                         PdschDmrsParams*                  d_dmrs_params,
                         PdschUeGrpParams*                 d_ue_grp_params,
                         void*                             cpu_desc,
                         void*                             gpu_desc,
                         uint8_t                           enable_desc_async_copy,
                         cudaStream_t                      strm);

/** @} */ /* END DL_CUPHY_RATE_MATCHING */

/**
 * LDPC Codeword Results
 */
typedef struct
{
    unsigned char numIterations;
    unsigned char checkErrorCount;
} cuphyLDPCResults_t;

/**
 * LDPC Decode Iteration selection
 */
typedef enum _cuphyLdpcMaxNumItrAlgoType
{
    LDPC_MAX_NUM_ITR_ALGO_TYPE_FIXED   = 0,
    LDPC_MAX_NUM_ITR_ALGO_TYPE_LUT     = 1,
    LDPC_MAX_NUM_ITR_ALGO_TYPE_PER_UE  = 2
} cuphyLdpcMaxItrAlgoType_t;

/**
 * LDPC Encoder launch config
 */
struct cuphyLDPCEncodeLaunchConfig
{
    CUDA_KERNEL_NODE_PARAMS m_kernelNodeParams;
    void*                   m_desc;
    void*                   m_kernelArgs[1];
};
typedef struct cuphyLDPCEncodeLaunchConfig* cuphyLDPCEncodeLaunchConfig_t;

/** @brief: Compute descriptor size and alignment for LDPC Encoder.
 *
 * @param[in,out] pDescrSizeBytes:  Size in bytes of descriptor
 * @param[in,out] pDescrAlignBytes: Alignment in bytes of descriptor
 * @param[in]     maxUes: Maximum number of UEs processed with this workspace. Can use PDSCH_MAX_UES_PER_CELL_GROUP as max.
 * @param[in,out] pWorkspaceBytes: Number of workspace bytes; it's a function of maxUes (allocated by caller)
 * @return CUPHY_STATUS_SUCCESS or CUPHY_STATUS_INVALID_ARGUMENT
 */
cuphyStatus_t CUPHYWINAPI cuphyLDPCEncodeGetDescrInfo(size_t*  pDescrSizeBytes,
                                                      size_t*  pDescrAlignBytes,
                                                      uint16_t maxUes,
                                                      size_t*  pWorkspaceBytes);

/** @brief: Setup LDPC encoder.
 *
 *  @param[in] ldpcEncodeLaunchConfig: Pointer to LDPC encoder launch config.
 *  @param[in] inDesc: tensor descriptor for LDPC encoder's input
 *  @param[in] inAddr: address for LDPC encoder's input, only used if batching is disabled
 *  @param[in] outDesc: tensor descirptor for LDPC encoder's output
 *  @param[in] outAddr: address for LDPC encoder's output, only used if batching is disabled
 *  @param[in] BG: base graph type; supported values 1 or 2.
 *  @param[in] Z: lifting size
 *  @param[in] puncture: puncture nodes if set to 1; no puncturing if 0.
 *  @param[in] maxParityNodes: maximum number of parity nodes to compute; set to 0 if unknown or if no optimization is needed.
 *  @param[in] max_rv: redundancy version, the max. in the batch
 *  @param[in] batching: when enabled, the input and output addresses used are the first batched_TBs elements inBatchedAddr and outBatchedAddr respectively,
 *                       and not inAddr or outAddr. The TBs batched can also belong to different cells.
 *  @param[in] batched_TBs: number of transport blocks (TBs) processed in a single kernel launch
 *  @param[in] inBatchedAddr: array of per-TB input addresses; first batched_TBs elements are valid if batching is 1
 *  @param[in] outBatchedAddr: array of per-TB output addresses; first batched_TBs elements are valid if batching is 1
 *  @param[in] h_workspace: pre-allocated host buffer used internally in LDPC
 *  @param[in] d_workspace: device memory for h_workspace. The H2D copy from h_workspace to d_workspace happens within cuphySetupDLDPCEncode if enable_desc_async_copy is set.
 *  @param[in] cpu_desc: Pointer to descriptor in CPU memory
 *  @param[in] gpu_desc: Pointer to descriptor in GPU memory
 *  @param[in] enable_desc_async_copy: async copy CPU descriptor into GPU if set.
 *  @param[in] strm: CUDA stream for async copy
 *
 *  @return CUPHY_STATUS_SUCCESS or CUPHY_STATUS_INVALID_ARGUMENT
 */
cuphyStatus_t CUPHYWINAPI cuphySetupLDPCEncode(cuphyLDPCEncodeLaunchConfig_t ldpcEncodeLaunchConfig,
                                               cuphyTensorDescriptor_t       inDesc,
                                               void*                         inAddr,
                                               cuphyTensorDescriptor_t       outDesc,
                                               void*                         outAddr,
                                               int                           BG,
                                               int                           Z,
                                               uint8_t                       puncture,
                                               int                           maxParityNodes,
                                               int                           max_rv,
                                               uint8_t                       batching,
                                               int                           batched_TBs,
                                               void**                        inBatchedAddr,
                                               void**                        outBatchedAddr,
                                               void*                         h_workspace,
                                               void*                         d_workspace,
                                               void*                         cpu_desc,
                                               void*                         gpu_desc,
                                               uint8_t                       enable_desc_async_copy,
                                               cudaStream_t                  strm);

/**
 * \defgroup CUPHY_ERROR_CORRECTION Error Correction
 *
 * This section describes the error correction functions of the cuPHY
 * application programming interface.
 *
 * @{
 */

struct cuphyLDPCDecoder;
/**
 * cuPHY LDPC decoder handle
 */
typedef struct cuphyLDPCDecoder* cuphyLDPCDecoder_t;

/******************************************************************/ /**
 * \brief Allocates and initializes a cuPHY LDPC decoder instance
 *
 * Allocates a cuPHY decoder instance and returns a handle in the
 * address provided by the caller.
 *
 * Returns ::CUPHY_STATUS_INVALID_ARGUMENT if \p pdecoder is NULL.
 *
 * Returns ::CUPHY_STATUS_ALLOC_FAILED if an LDPC decoder cannot be
 * allocated on the host.
 *
 * Returns ::CUPHY_STATUS_SUCCESS if allocation and initialization were
 * successful.
 *
 * \param context - cuPHY context
 * \param pdecoder - Address for the new ::cuphyLDPCDecoder_t instance
 * \param flags - Creation flags (currently unused)
 *
 * \return
 * ::CUPHY_STATUS_SUCCESS,
 * ::CUPHY_STATUS_ALLOC_FAILED,
 * ::CUPHY_STATUS_INVALID_ARGUMENT
 *
 * \sa ::cuphyStatus_t,::cuphyGetErrorName,::cuphyGetErrorString,::cuphyCreateContext,::cuphyDestroyLDPCDecoder
 */
cuphyStatus_t CUPHYWINAPI cuphyCreateLDPCDecoder(cuphyContext_t      context,
                                                 cuphyLDPCDecoder_t* pdecoder,
                                                 unsigned int        flags);

/******************************************************************/ /**
 * \brief Destroys a cuPHY LDPC decoder object
 *
 * Destroys a cuPHY LDPC decoder object that was previously created by
 * a call to ::cuphyCreateLDPCDecoder. The handle provided to this
 * function should not be used for any operations after this function
 * returns.
 *
 * Returns ::CUPHY_STATUS_INVALID_ARGUMENT if \p decoder is NULL.
 *
 * Returns ::CUPHY_STATUS_SUCCESS if destruction was successful.
 *
 * \param decoder - previously allocated ::cuphyLDPCDecoder_t instance
 *
 * \return
 * ::CUPHY_STATUS_SUCCESS,
 * ::CUPHY_STATUS_INVALID_ARGUMENT
 *
 * \sa ::cuphyStatus_t,::cuphyCreateLDPCDecoder
 */
cuphyStatus_t CUPHYWINAPI cuphyDestroyLDPCDecoder(cuphyLDPCDecoder_t decoder);

/**
 * LDPC LLR buffer
 */
typedef struct
{
    void*   addr;
    int32_t stride_elements;
    int32_t num_codewords;
} cuphyTransportBlockLLRDesc_t;

/**
 * LDPC Output buffer
 */
typedef struct
{
    uint32_t* addr;
    int32_t   stride_words;
    int32_t   num_codewords;
} cuphyTransportBlockDataDesc_t;

/**
 * LDPC Decoder normalization parameter
 */
typedef union
{
    float       f32;
    __half2_raw f16x2;
} cuphyLDPCNormalization_t;

/**
 *  LDPC Decoder configuration descriptor
 */
typedef struct
{
    cuphyDataType_t          llr_type;         /*!< Type of LLR input data (CUPHY_R_16F or CUPHY_R_32F) */
    int16_t                  num_parity_nodes; /*!< Number of parity nodes */
    int16_t                  Z;                /*!< Lifting size */
    int16_t                  max_iterations;   /*!< Maximum number of iterations */
    int16_t                  Kb;               /*!< Number of "information" variable nodes */
    cuphyLDPCNormalization_t norm;             /*!< Normalization (for normalized min-sum) */
    uint32_t                 flags;            /*!< Flags */
    int16_t                  BG;               /*!< Base graph (1 or 2) */
    int16_t                  algo;             /*!< Algorithm (0 for automatic choice) */
    float                    clamp_value;      /*!< Clamp value */
    void*                    workspace;        /*!< Workspace area */
} cuphyLDPCDecodeConfigDesc_t;

/**
 * LDPC decode descriptor
 */
typedef struct
{
    cuphyLDPCDecodeConfigDesc_t   config;                                    /*!< Common decoder configuration */
    int32_t                       num_tbs;                                   /*!< Number of valid TB descriptors */
    cuphyTransportBlockLLRDesc_t  llr_input[CUPHY_LDPC_DECODE_DESC_MAX_TB];  /*!< Input LLR buffers */
    cuphyTransportBlockDataDesc_t tb_output[CUPHY_LDPC_DECODE_DESC_MAX_TB];  /*!< Output bit/data buffers */
    cuphyTransportBlockLLRDesc_t  llr_output[CUPHY_LDPC_DECODE_DESC_MAX_TB]; /*!< Output LLR buffers (optional) */
} cuphyLDPCDecodeDesc_t;

/**
 * Launch config for LDPC Decoder
 */
typedef struct
{
    CUDA_KERNEL_NODE_PARAMS kernel_node_params_driver;
    void*                   kernel_args[2];
    cuphyLDPCDecodeDesc_t   decode_desc;
} cuphyLDPCDecodeLaunchConfig_t;

/******************************************************************/ /**
 * \brief Perform a bulk LDPC decode operation on a tensor of soft input values
 *
 * Performs a bulk LDPC decode operation on an input tensor of "soft"
 * log likelihood ratio (LLR) values.
 *
 * \param decoder - cuPHY LDPC decoder instance
 * \param tensorDescDst - tensor descriptor for LDPC output
 * \param dstAddr - address for LDPC output
 * \param tensorDescLLR - tensor descriptor for soft input LLR values
 * \param LLRAddr - address for soft input LLR values
 * \param tensorDescSoftOutputs - tensor descriptor for soft output LLR values (can be NULL)
 * \param softOutputsAddr - address for soft output LLR values (can be NULL)
 * \param config - LDPC configuration structure
 * \param strm - CUDA stream for LDPC execution
 *
 * If the value of \p algoIndex is zero, the library will choose the "best"
 * algorithm for the given LDPC configuration.
 *
 * The type of input tensor descriptor \p tensorDescLLR must be either ::CUPHY_R_32F or
 * ::CUPHY_R_16F, and the rank must be 2.
 *
 * The type of output tensor descriptor \p tensorDescDst must be ::CUPHY_BIT, and the
 * rank must be 2.
 *
 * For input LLR tensors of type CUPHY_R_16F, loads occur as multiples of 8 elements
 * (i.e. 16 bytes). Therefore, memory allocation should be performed such that
 * the number of LLR elements that can be read is a multiple of 8 for each codeword.
 * This can be done by specifying a stride that is multiple of 8 for the second
 * dimension, or by using the CUPHY_TENSOR_ALIGN_COALESCE flag when allocating
 * the tensor. Values read from this padded memory will not be used, and do not
 * need to be zeroed or cleared.
 *
 * For input LLR tensors of type CUPHY_R_32F, loads occur as multiples of 4 elements
 * (i.e. 16 bytes). Therefore, memory allocation should be performed such that
 * the number of LLR elements that can be read is a multiple of 4 for each codeword.
 * This can be done by specifying a stride that is multiple of 4 for the second
 * dimension, or by using the CUPHY_TENSOR_ALIGN_COALESCE flag when allocating
 * the tensor. Values read from this padded memory will not be used, and do not
 * need to be zeroed or cleared.
 *
 * Soft output values will be written when \p softOutputsAddr and \p tensorDescSoftOutputs
 * are both non-NULL. Soft output values are always written as fp16 values, regardless
 * of the input LLR data type. Soft output stores will use 32-bit instructions. Therefore,
 * \p softOutputsAddr must be 4-byte aligned, strides[0] must be 1, and strides[1] must
 * be a multiple of 2.
 *
 * The union member of the normalization value in the configuration \p config must
 * match the LLR type in \p config. In other words, if the LLR type is CUPHY_R_32F,
 * the normalization value should be populated using the f32 union member, and if
 * the LLR type is CUPHY_R_16F, both halves of the f16x2 union member should be set
 * with the same normalization value in fp16 format. The CUDA __float2half2_rn()
 * function can be used to convert a float value to a pair of fp16 values.
 * Alternatively, if the ::cuphyErrorCorrectionLDPCDecodeSetNormalization() function
 * is used, the correct union member will be set automatically by that function.
 *
 * Returns ::CUPHY_STATUS_INVALID_ARGUMENT if:
 * <ul>
 *   <li>\p decoder is NULL</li>
 *   <li>\p BG, \p Kb, \p mb, and \p Z do not represent a valid LDPC configuration</li>
 *   <li>\p maxNumIterations <= 0</li>
 *   <li>\p tensorDescDst is NULL</li>
 *   <li>\p tensorDescLLR is NULL</li>
 *   <li>\p dstAddr NULL</li>
 *   <li>\p LLRAddr is NULL</li>
 *   <li>\p the data type of \p tensorDescDst and llr_type in \p config do not match</li>
 *   <li>the data type of \p tensorDescDst and llr_type in \p config do not match</li>
 *   <li>\p tensorDescSoftOutputs and \p softOutputsAddr are not both NULL or both non-NULL</li>
 *   <li>\p softOutputsAddr is non-NULL and the type of \p tensorDescSoftOutputs is not CUPHY_R_16F</li>
 * </ul>
 *
 * Returns ::CUPHY_STATUS_UNSUPPORTED_CONFIG if the combination of the LDPC configuration
 * (\p BG, \p Kb, \p mb, and \p Z) is not supported for a given LLR tensor and/or
 * algorithm index (\p algoIndex).
 *
 * Returns ::CUPHY_STATUS_UNSUPPORTED_RANK if either the input tensor descriptor (\p tensorDescLLR)
 * or output tensor descriptor (\p tensorDescDst) do not have a rank of 2.
 *
 * Returns ::CUPHY_STATUS_UNSUPPORTED_TYPE if the output tensor descriptor (\p tensorDescLLR)
 * is not of type ::CUPHY_BIT, or if the input tensor descriptor is not one of (::CUPHY_R_32F or ::CUPHY_R_16F)
 *
 * Returns ::CUPHY_STATUS_SUCCESS if the decode operation was submitted to the stream
 * successfully.
 *
 * \return
 * ::CUPHY_STATUS_SUCCESS,
 * ::CUPHY_STATUS_INVALID_ARGUMENT
 * ::CUPHY_STATUS_UNSUPPORTED_RANK
 * ::CUPHY_STATUS_UNSUPPORTED_TYPE
 * ::CUPHY_STATUS_UNSUPPORTED_CONFIG
 *
 * \sa ::cuphyStatus_t,::cuphyCreateLDPCDecoder,::cuphyDestroyLDPCDecoder,::cuphyErrorCorrectionLDPCDecodeGetWorkspaceSize
 */
cuphyStatus_t CUPHYWINAPI cuphyErrorCorrectionLDPCDecode(cuphyLDPCDecoder_t                 decoder,
                                                         cuphyTensorDescriptor_t            tensorDescDst,
                                                         void*                              dstAddr,
                                                         cuphyTensorDescriptor_t            tensorDescLLR,
                                                         const void*                        LLRAddr,
                                                         cuphyTensorDescriptor_t            tensorDescSoftOutputs,
                                                         void*                              softOutputsAddr,
                                                         const cuphyLDPCDecodeConfigDesc_t* config,
                                                         cudaStream_t                       strm);

/******************************************************************/ /**
 * \brief Perform a bulk LDPC decode operation on a tensor of soft input values
 *
 * Performs a bulk LDPC decode operation on "soft" log likelihood ratio
 * (LLR) values for one or more transport blocks
 *
 * \param decoder - cuPHY LDPC decoder instance
 * \param decodeDesc - LDPC decode descriptor
 * \param strm - CUDA stream for LDPC execution
 *
 * If the value of algo field of the descriptor \p decodeDesc is zero, the
 * library will choose the "best" algorithm for the given LDPC configuration.
 *
 * The llr_type field of the \p decodeDesc must be either ::CUPHY_R_32F or
 * ::CUPHY_R_16F.
 *
 * For input LLR buffers of type CUPHY_R_16F, loads occur as multiples of 8 elements
 * (i.e. 16 bytes). Therefore, memory allocation should be performed such that
 * the number of LLR elements that can be read is a multiple of 8 for each codeword.
 * The memory need only be addressable. (For a multi-codeword case, the memory
 * can lie in the next codeword.) Values read from padded memory will not be used,
 * and do not need to be zeroed or cleared.
 *
 * For input LLR tensors of type CUPHY_R_32F, loads occur as multiples of 4 elements
 * (i.e. 16 bytes). Therefore, memory allocation should be performed such that
 * the number of LLR elements that can be read is a multiple of 4 for each codeword.
 * Values read from padded memory will not be used, and do not need to be zeroed
 * or cleared.
 *
 * The union member of the normalization value in the configuration \p config must
 * match the LLR type in the decode descriptor configuration. In other words, if the
 * LLR type is CUPHY_R_32F, the normalization value should be populated using the
 * f32 union member, and if the LLR type is CUPHY_R_16F, both halves of the f16x2
 * union member should be set with the same normalization value in fp16 format. The
 * CUDA __float2half2_rn() function can be used to convert a float value to a pair
 * of fp16 values. Alternatively, if the
 * ::cuphyErrorCorrectionLDPCDecodeSetNormalization() function is used, the correct
 * union member will be set automatically by that function.
 *
 * Returns ::CUPHY_STATUS_INVALID_ARGUMENT if:
 * <ul>
 *   <li>\p decoder is NULL</li>
 *   <li>\p BG, \p Kb, \p mb, and \p Z do not represent a valid LDPC configuration</li>
 *   <li>\p maxNumIterations <= 0</li>
 * </ul>
 *
 * Returns ::CUPHY_STATUS_UNSUPPORTED_CONFIG if the combination of the LDPC configuration
 * (\p BG, \p Kb, \p mb, and \p Z) is not supported for a given LLR tensor and/or
 * algorithm index.
 *
 * Returns ::CUPHY_STATUS_SUCCESS if the decode operation was submitted to the stream
 * successfully.
 *
 * \return
 * ::CUPHY_STATUS_SUCCESS,
 * ::CUPHY_STATUS_INVALID_ARGUMENT
 * ::CUPHY_STATUS_UNSUPPORTED_CONFIG
 *
 * \sa ::cuphyStatus_t,::cuphyCreateLDPCDecoder,::cuphyDestroyLDPCDecoder
 */
cuphyStatus_t CUPHYWINAPI cuphyErrorCorrectionLDPCTransportBlockDecode(cuphyLDPCDecoder_t           decoder,
                                                                       const cuphyLDPCDecodeDesc_t* decodeDesc,
                                                                       cudaStream_t                 strm);

/******************************************************************/ /**
 * \brief Returns the workspace size for and LDPC decode operation
 *
 * Calculates the workspace size (in bytes) required to perform an LDPC
 * decode operation for the given LDPC configuration.
 *
 * If the \p algoIndex parameter is -1, the function will return the
 * maximum workspace size for all numbers of parity nodes less than or
 * equal to the value of the \p mb parameter (for the given lifting
 * size \p Z). This is useful for determining the maximum workspace
 * size across different code rates.
 *
 * \param decoder - decoder object created by ::cuphyCreateLDPCDecoder
 * \param config - LDPC decoder configuration
 * \param numCodeWords - number of codewords to decode simultaneously
 * \param sizeInBytes - output address for calculated workspace size
 *
 * Different LDPC decoding algorithms may have different workspace
 * requirements. If the value of \p algoIndex is zero, the library will
 * choose the "best" algorithm for the given LDPC configuration.
 *
 * Returns ::CUPHY_STATUS_INVALID_ARGUMENT if:
 * <ul>
 *   <li>\p BG, \p Kb, \p mb, and \p Z do not represent a valid LDPC configuration</li>
 *   <li>\p numCodeWords <= 0</li>
 *   <li>\p sizeInBytes is NULL</li>
 * </ul>
 *
 * Returns ::CUPHY_STATUS_UNSUPPORTED_CONFIG if the combination of the LDPC configuration
 * (\p BG, \p Kb, \p mb, and \p Z) is not supported for a given \p LLRtype and/or
 * algorithm index (\p algoIndex).
 *
 *
 * Returns ::CUPHY_STATUS_SUCCESS if the size calculation was successful.
 *
 * \return
 * ::CUPHY_STATUS_SUCCESS,
 * ::CUPHY_STATUS_INVALID_ARGUMENT
 * ::CUPHY_STATUS_UNSUPPORTED_CONFIG
 *
 * \sa ::cuphyStatus_t,::cuphyCreateLDPCDecoder,::cuphyErrorCorrectionLDPCDecode,::cuphyDestroyLDPCDecoder
 */
cuphyStatus_t CUPHYWINAPI cuphyErrorCorrectionLDPCDecodeGetWorkspaceSize(cuphyLDPCDecoder_t                 decoder,
                                                                         const cuphyLDPCDecodeConfigDesc_t* config,
                                                                         int                                numCodeWords,
                                                                         size_t*                            sizeInBytes);

/******************************************************************/ /**
 * \brief Sets the min-sum normalization constant for a given LDPC configuration
 *
 * Determines an appropriate LDPC decoder min-sum normalization constant,
 * given the LLR type and num_parity_nodes fields of the input configuration.
 * Note that if the llr_type field of the configuration is CUPHY_R_16F, the
 * field will be set to a pair of __half values (as is expected by the
 * LDPC decoder kernel).
 *
 * \param decoder - decoder object created by ::cuphyCreateLDPCDecoder
 * \param decodeDesc - decode descriptor with valid llr_type and num_parity_nodes fields
 *
 * Returns ::CUPHY_STATUS_INVALID_ARGUMENT if:
 * <ul>
 *   <li>\p llr_type or num_parity_nodes fields do not represent a valid LDPC configuration</li>
 * </ul>
 *
 * Returns ::CUPHY_STATUS_UNSUPPORTED_CONFIG if the combination of the LDPC configuration
 * (\p BG, \p Kb, \p mb, and \p Z) is not supported for a given \p LLRtype and/or
 * algorithm index (\p algo).
 *
 *
 * Returns ::CUPHY_STATUS_SUCCESS if the constant was set successfully.
 *
 * \return
 * ::CUPHY_STATUS_SUCCESS,
 * ::CUPHY_STATUS_INVALID_ARGUMENT
 *
 * \sa ::cuphyStatus_t,::cuphyCreateLDPCDecoder,::cuphyDestroyLDPCDecoder
 */
cuphyStatus_t CUPHYWINAPI cuphyErrorCorrectionLDPCDecodeSetNormalization(cuphyLDPCDecoder_t           decoder,
                                                                         cuphyLDPCDecodeConfigDesc_t* decodeDesc);

/******************************************************************/ /**
 * \brief Populates a launch configuration for the LDPC decoder
 *
 *
 * \param decoder - decoder object created by ::cuphyCreateLDPCDecoder
 * \param launchConfig - launch structure with a populated config (see ::cuphyLDPCDecodeConfigDesc_t)
 *
 * Returns ::CUPHY_STATUS_INVALID_ARGUMENT if:
 * <ul>
 *   <li>\p decoder is not a valid ::cuphyLDPCDecoder_t instance</li>
 *   <li>\p launchConfig is NULL</li>
 * </ul>
 *
 * Returns ::CUPHY_STATUS_UNSUPPORTED_CONFIG if the combination of the LDPC configuration
 * (\p BG, \p Kb, \p mb, and \p Z) is not supported for a given \p LLRtype and/or
 * algorithm index (\p algo).
 *
 *
 * Returns ::CUPHY_STATUS_SUCCESS if the launch configuration was populated successfully
 *
 * \return
 * ::CUPHY_STATUS_SUCCESS,
 * ::CUPHY_STATUS_INVALID_ARGUMENT
 *
 * \sa ::cuphyStatus_t,::cuphyCreateLDPCDecoder,::cuphyDestroyLDPCDecoder
 */
cuphyStatus_t CUPHYWINAPI cuphyErrorCorrectionLDPCDecodeGetLaunchDescriptor(cuphyLDPCDecoder_t             decoder,
                                                                            cuphyLDPCDecodeLaunchConfig_t* launchConfig);

/** @} */ /* END CUPHY_ERROR_CORRECTION */

/**
 * \defgroup CUPHY_MODULATION_MAPPER  Modulation Mapper
 *
 * This section describes the modulation function(s) of the cuPHY
 * application programming interface.
 *
 * @{
 */

struct PdschDmrsParams;

/**
 * Modulation launch config
 */
struct cuphyModulationLaunchConfig
{
    CUDA_KERNEL_NODE_PARAMS m_kernelNodeParams;
    void*                   m_desc;
    void*                   m_kernelArgs[1];
};
typedef struct cuphyModulationLaunchConfig* cuphyModulationLaunchConfig_t;

cuphyStatus_t CUPHYWINAPI cuphySetEmptyKernelNodeParams(CUDA_KERNEL_NODE_PARAMS* pNodeParams);
cuphyStatus_t CUPHYWINAPI cuphySetGenericEmptyKernelNodeParams(CUDA_KERNEL_NODE_PARAMS* pNodeParams, int ptrArgsCnt, void** pKernelParams);
cuphyStatus_t CUPHYWINAPI cuphySetGenericEmptyKernelNodeGridConstantParams(CUDA_KERNEL_NODE_PARAMS* pNodeParams, void** pKernelParams, int ptrArgsCnt, uint16_t descr_size);
cuphyStatus_t CUPHYWINAPI cuphySetWorkCancelKernelNodeParams(CUDA_KERNEL_NODE_PARAMS* pNodeParams, void** pKernelPARAMS, uint8_t device_graph_launch);

void CUPHYWINAPI cuphySetD2HMemcpyNodeParams(CUDA_MEMCPY3D *memcpyParams, void* src_d, void* dst_h, size_t size_in_bytes);

/** @brief: Compute descriptor size and alignment for modulation mapper.
 *
 * @param[in,out] pDescrSizeBytes:  Size in bytes of descriptor
 * @param[in,out] pDescrAlignBytes: Alignment in bytes of descriptor
 * @return CUPHY_STATUS_SUCCESS or CUPHY_STATUS_INVALID_ARGUMENT
 */
cuphyStatus_t CUPHYWINAPI cuphyModulationGetDescrInfo(size_t* pDescrSizeBytes,
                                                      size_t* pDescrAlignBytes);

/** @brief: Setup modulation mapper component.
 *  @param[in] modulationLaunchConfig: Pointer to cuphyModulationLaunchConfig.
 *  @param[in] d_params: Pointer to PdschDmrsParams on the device.
 *                       If nullptr, then symbols are allocated contiguously, starting from
 *                       zero in modulation_output. If not, symbols are allocated
 *                       in the appropriate Rbs, start position, in the {273*12, 14, 16}
 *                       modulation_output tensor.
 *  @param[in] input_desc: input tensor descriptor; dimension ceil(num_bits/32.0). Not used.
 *  @param[in] modulation_input: pointer to input tensor data
 *                               Data is expected to be contiguously allocated for every layer without
 *                               any gaps. Each layer should start at a uint32_t aligned boundary.
 *  @param[in] max_num_symbols: maximum number of symbols across all TBs.
 *  @param[in] max_bits_per_layer: maximum number of bits per layer across all TBs in modulation_input.
 *  @param[in] num_TBs:  number of Transport Blocks contained in modulation_input
 *  @param[in] workspace: pointer to # TBs PerTBParams struct on the device. Only fields G and Qm are used.
 *  @param[in] output_desc: output tensor descriptor; dimension (num_bits / modulation_order)
 *                          if d_params=nullptr or {273*12, 14, 16} otherwise. Not used.
 *  @param[in,out] modulation_output: pointer to output tensor (preallocated)
 *                                    Each symbol is a complex number using half-precision for
 *                                    the real and imaginary parts.
 *  @param[in] cpu_desc: Pointer to descriptor in CPU memory
 *  @param[in] gpu_desc: Pointer to descriptor in GPU memory
 *  @param[in] enable_desc_async_copy: async copy CPU descriptor into GPU if set.
 *  @param[in] strm: CUDA stream for async copy
 *  @return CUPHY_STATUS_SUCCESS or CUPHY_STATUS_INVALID_ARGUMENT
 */
cuphyStatus_t CUPHYWINAPI cuphySetupModulation(cuphyModulationLaunchConfig_t modulationLaunchConfig,
                                               PdschDmrsParams*              d_params,
                                               cuphyTensorDescriptor_t       input_desc, /* not used */
                                               const void*                   modulation_input,
                                               int                           max_num_symbols,
                                               int                           max_bits_per_layer,
                                               int                           num_TBs,
                                               PdschPerTbParams*             workspace,   /* num_TBs entries; the only fields used are G (# rate matched bits) and Qm (modulation_order) */
                                               cuphyTensorDescriptor_t       output_desc, /* not used */
                                               void*                         modulation_output,
                                               void*                         cpu_desc,
                                               void*                         gpu_desc,
                                               uint8_t                       enable_desc_async_copy,
                                               cudaStream_t                  strm);

/** @} */ /* END CUPHY_MODULATION_MAPPER */

/**
 * \defgroup UL_CUPHY_PUCCH_RECEIVER PUCCH Receiver
 *
 * This section describes the structs and functions of the uplink
 * cuPHY control channel receiver. Currently only PUCCH Format 1 is supported.
 *
 * @{
 */

/**
 * @brief Struct that tracks user equipment (UE) specific PUCCH parameters.
 */
struct PucchUeCellParams
{
    uint32_t time_cover_code_index;   /*!< time cover code index; used to remove user's code */
    uint32_t num_bits;                /*!< number of transmitted bits: 1 or 2 */
    uint32_t init_cyclic_shift_index; /*!< initial cyclic shift; used in cyclic shift index computation*/
};

/**
 * @brief Struct that tracks all necessary parameters for PUCCH receiver processing.
 *        It also includes a PucchUeCellParams struct per UE.
 */
struct PucchParams
{
    uint32_t format;       /*!< PUCCH format. Should be CUPHY_PUCCH_FORMAT1 for now. */
    uint32_t num_pucch_ue; /*!< number of user equipment (UEs) in PUCCH */

    float Wf[CUPHY_N_TONES_PER_PRB * CUPHY_N_TONES_PER_PRB];          /*!< frequency channel estimation filter */
    float Wt_cell[OFDM_SYMBOLS_PER_SLOT * OFDM_SYMBOLS_PER_SLOT / 4]; /*!< time channel estimation filter; overprovisioned */

    uint32_t start_symbol;       /*!< start symbol (in time dimension of input signal) */
    uint32_t num_symbols;        /*!< number of symbols [4, 14] */
    uint32_t PRB_index;          /*!< index of physical resource allocation */
    uint32_t low_PAPR_seq_index; /*!< sequence of low-PAPR (Peak-to-Average Power ratio) */
    uint32_t num_dmrs_symbols;   /*!< number of DMRS symbols (derived parameter); ceil(num_symbols*1.0/2) in PUCCH Format 1 */
    uint32_t num_data_symbols;   /*!< number of data symbols (derived parameters); num_symbols - num_dmrs_symbols */
    uint32_t num_bs_antennas;    /*!< number of base station antennas */
    uint32_t mu;                 /*!< numerology */
    uint32_t slot_number;        /*!< slot number */
    uint32_t hopping_id;         /*!< hopping Id */

    PucchUeCellParams cell_params[MAX_UE_CNT]; /*!< PucchUeCellParams structs; overprovisioned (first num_pucch_ue elements valid) */
};

/** @brief: Return workspace size, in bytes, needed for all configuration parameters
 *          and intermediate computations of the PUCCH receiver. Does not allocate any space.
 *  @param[in] num_ues: number of User Equipement (UEs)
 *  @param[in] num_bs_antennas: number of Base Station (BS) antennas
 *  @param[in] num_symbols: number of symbols; sum of DMRS and data symbols.
 *  @param[in] pucch_complex_data_type: PUCCH receiver data type identifier: CUPHY_C_32F or CUPHY_C_16F
 *  @return workspace size in bytes
 */
size_t cuphyPucchReceiverWorkspaceSize(int             num_ues,
                                       int             num_bs_antennas,
                                       int             num_symbols,
                                       cuphyDataType_t pucch_complex_data_type);

/** @brief: Copy PUCCH params from the CPU to the allocated PUCCH receiver workspace.
 *          The location of the struct in the workspace is implementation dependent.
 *  @param[in] h_pucch_params: pointer to PUCCH configuration parameters on the host.
 *  @param[in] pucch_workspace: pointer to the pre-allocated pucch receiver's workspace on the device.
 *  @param[in] pucch_complex_data_type: PUCCH receiver data type identifier: CUPHY_C_32F or CUPHY_C_16F
 */
void cuphyCopyPucchParamsToWorkspace(const PucchParams* h_pucch_params,
                                     void*              pucch_workspace,
                                     cuphyDataType_t    pucch_complex_data_type);

/** @brief: Launch PUCCH receiver kernels that do processing at receive end of PUCCH
 *          (Physical Uplink Control Channel).
 *  @param[in] data_rx_desc: input tensor descriptor; dimensions: Nf x Nt x L_BS
 *  @param[in] data_rx_addr: pointer to input tensor data (i.e., base station received signal); each
 *                           tensor element is a complex number
 *  @param[in] bit_estimates_desc: output tensor descriptor; dimensions nUe_pucch x 2
 *  @param[in, out] bit_estimates_addr: pre-allocated device buffer with bit estimates
 *  @param[in] pucch_format: PUCCH format; currently only format 1 is supported.
 *  @param[in, out] pucch_params: pointer to PUCCH config params.
 *  @param[in] strm: CUDA stream for kernel launch.
 *  @param[in, out] pucch_workspace: address of user allocated workspace
 *                  pucch params should have been already copied there via a
 *                  cuphyCopyPucchParamsToWorkspace() call.
 *  @param[in] allocated_workspace_size: size of pucch_workspace
 *  @param[in] pucch_complex_data_type: PUCCH receiver data type identifier: CUPHY_C_32F or CUPHY_C_16F
 */
void cuphyPucchReceiver(cuphyTensorDescriptor_t data_rx_desc,
                        const void*             data_rx_addr,
                        cuphyTensorDescriptor_t bit_estimates_desc,
                        void*                   bit_estimates_addr,
                        const uint32_t          pucch_format,
                        const PucchParams*      pucch_params,
                        cudaStream_t            strm,
                        void*                   pucch_workspace,
                        size_t                  allocated_workspace_size,
                        cuphyDataType_t         pucch_complex_data_type);

/** @} */ /* END UL_CUPHY_PUCCH_RECEIVER */

/**
 * \defgroup DL_CUPHY_PDCCH  PDCCH
 *
 * This section describes the PDCCH functions of the cuPHY
 * application programming interface.
 *
 * @{
 */

/**
 * PDCCH encoder rate matching launch configs for multi-DCI kernel
 */
typedef struct
{
    CUDA_KERNEL_NODE_PARAMS kernelNodeParamsDriver;
    void*                   kernelArgs[5];
} cuphyEncoderRateMatchMultiDCILaunchCfg_t;

/**
 * PDCCH encoder rate matching launch configs for multi-SSB kernel
 */
typedef struct
{
    CUDA_KERNEL_NODE_PARAMS kernelNodeParamsDriver;
    void*                   kernelArgs[3];
} cuphyEncoderRateMatchMultiSSBLaunchCfg_t;

/**
 * PDCCH SSB Mapper launch configs
 */
typedef struct
{
    CUDA_KERNEL_NODE_PARAMS kernelNodeParamsDriver;
    void*                   kernelArgs[5];
} cuphySsbMapperLaunchCfg_t;

/**
 * PDCCH Kernel launch configs to generate scrambling sequence in PDCCH
 */
typedef struct
{
    CUDA_KERNEL_NODE_PARAMS kernelNodeParamsDriver;
    void*                   kernelArgs[2];
} cuphyGenScramblingSeqLaunchCfg_t;

/**
 * Launch configs to generate TF signals in PDCCH
 */
typedef struct
{
    CUDA_KERNEL_NODE_PARAMS kernelNodeParamsDriver;
    void*                   kernelArgs[6];
} cuphyGenPdcchTfSgnlLaunchCfg_t;

/**
 * @brief Struct that tracks all necessary parameters for PDCCH computation.
 *        It contains information common across all DCIs, as well as
 *        as per-DCI specific configuration parameters.
 */
struct PdcchParams
{
    uint32_t n_f;         /*!< number of subcarriers in full BW */
    uint32_t slot_number; /*!< slot number */
    uint32_t start_rb;    /*!< starting RB */
    uint32_t start_sym;   /*!< starting OFDM symbol number */
    uint32_t n_sym;       /*!< number of PDCCH OFDM symbols (1-3) */

    uint32_t shift_index;      /*!< shift index */
    uint32_t bundle_size;      /*!< bundle size for PDCCH. It is in REGs. Can be 2, 3, or 6.*/
    uint16_t interleaver_size; /*!< Interleaving happens at the bundle granularity. Can be 2, 3, or 6. */
    uint8_t  interleaved;      /*!< 1 for interleaved mode, 0 otherwise */
    uint8_t  testModel;        /*!< 1 if coreset belongs to cell in testing mode; 0 otherwise */
    uint64_t coreset_map;      /*!< Derived. Used as bitmask. Shifted version of freq_domain_resource */
    uint32_t n_CCE;            /*!< Derived. It is the number of set bits in coreset_map (or freq_domain_resource) multiplied by n_sym */
    uint32_t rb_coreset;       /*!< Derived. Indicates the number of bits in coreset_map to be considered. It is # RBs divided by 6. */
    uint32_t num_dl_dci;       /*!< # DCIs included in this CORESET */
    uint32_t coreset_type;     /*!< Coreset type: 0 or 1 */

    uint64_t freq_domain_resource;                         /*!< Bitmask. Used to compute coreset_map, n_CCE, rb_coreset */

    uint32_t slotBufferIdx;
    uint32_t dciStartIdx;
    void*    slotBufferAddr; /*!< Pointer to buffer address.  Note: different coresets could have the same buffer */
};

struct _cuphyPdcchDciDynPrm;
typedef struct _cuphyPdcchDciDynPrm cuphyPdcchDciPrm_t;

struct _cuphyPmWOneLayer_t;
/// Precoding matrix for PDCCH
typedef struct _cuphyPmWOneLayer_t cuphyPdcchPmWOneLayer_t;

/**
 * @brief: Prepare for PDCCH TX pipeline.
 * @param[in, out] h_x_crc_addr: pointer to the payload after CRC was added and bit order reveresed. Every DCI payload sarts at a CUPHY_PDCCH_MAX_DCI_PAYLOAD_BYTES_W_CRC byte offset.
 * @param[in] h_x_crc_desc: descriptor for above payload. Not currently used.
 * @param[in]  h_xin_addr: pointer to the PDCCH input payload sequence, spanning multiple DCIs.
 *                      Each DCI payload starts at CUPHY_PDCCH_MAX_DCI_PAYLOAD_BYTES byte offset.
 * @param[in] h_xin_desc: descriptor for PDCCH input payload. Currently unused.
 * @param[in] num_coresets: number of coresets to be processed
 * @param[in] num_DCIs: cumulative number of DCIs over all num_coresets coresets
 * @param[in] h_params: pointer to PdcchParams struct
 * @param[in] h_dci_params: pointer to cuphyPdcchDciPrm_t struct
 * @param[in] h_dci_tm_info: pointer to DCI specific testing mode information
 * @param[in] pEncdRMLaunchCfg: pointer to launch configs to encode rate match with multiple DCIs in PDCCH.
 * @param[in] pScrmSeqLaunchCfg: pointer to launch configs to generate scrambling sequence in PDCCH.
 * @param[in] pTfSignalLaunchCfg: pointer to launch configs to generate TF signals in PDCCH.
 * @param[in] stream:   CUDA stream (currently not used)
 * @return CUPHY_STATUS_SUCCESS or CUPHY_STATUS_INVALID_ARGUMENT.
 */

cuphyStatus_t cuphyPdcchPipelinePrepare(void*                                     h_x_crc_addr,
                                        cuphyTensorDescriptor_t                   h_x_crc_desc,
                                        const void*                               h_xin_addr,
                                        cuphyTensorDescriptor_t                   h_xin_desc,
                                        int                                       num_coresets,
                                        int                                       num_DCIs,
                                        PdcchParams*                              h_params,
                                        cuphyPdcchDciPrm_t*                       h_dci_params,
                                        uint8_t*                                  h_dci_tm_info,
                                        cuphyEncoderRateMatchMultiDCILaunchCfg_t* pEncdRMLaunchCfg,
                                        cuphyGenScramblingSeqLaunchCfg_t*         pScrmSeqLaunchCfg,
                                        cuphyGenPdcchTfSgnlLaunchCfg_t*           pTfSignalLaunchCfg,
                                        cudaStream_t                              stream);

/** @} */ /* END DL_CUPHY_PDCCH */

/**
 * \defgroup DL_CUPHY_CSIRS  CSIRS
 *
 * This section describes the CSIRS functions of the cuPHY
 * application programming interface.
 *
 * @{
 */

/**
 * @brief CSIType enum
 */
typedef enum _cuphyCsiType
{
    TRS        = 0,
    NZP_CSI_RS = 1,
    ZP_CSI_RS  = 2
} cuphyCsiType_t;

/**
 * @brief CDM type
 */
typedef enum _cuphyCdmType
{
    NO_CDM       = 0,
    CDM2_FD      = 1,
    CDM4_FD2_TD2 = 2,
    CDM8_FD2_TD4 = 3,
    MAX_CDM_TYPE
} cuphyCdmType_t;

/**
 * @brief UCI DTX type
 */
typedef enum _cuphyUciDtxTypes
{
    UCI_HARQ_DTX  = 0,
    UCI_CSI1_DTX  = 1,
    UCI_CSI2_DTX  = 2,
    N_UCI_DTX     = 3
}cuphyUciDtxTypes_t;

/**
 * Launch configuration for CSI-RS scrambling sequence generation
 */
typedef struct
{
    CUDA_KERNEL_NODE_PARAMS kernelNodeParamsDriver;
    void*                   kernelArgs[3];
} cuphyGenScramblingLaunchCfg_t;

/**
 * Launch configuration for CSI-RS computation
 */
typedef struct
{
    CUDA_KERNEL_NODE_PARAMS kernelNodeParamsDriver;
    void*                   kernelArgs[8];
    uint32_t                totalNumThreadsLB;
} cuphyGenCsirsTfSignalLaunchCfg_t;

typedef struct
{
    CUDA_KERNEL_NODE_PARAMS kernelNodeParamsDriver;
    void*                   kernelArgs[4];
} cuphyCsirsRxChEstLaunchCfg_t;

/**
 * @brief CSI-RS resource mapping location row
 */
typedef struct _CsirsSymbLocRow
{
    uint8_t numPorts;                                        /*!< Number of ports */
    uint8_t lenKBarLBar;                                     /*!< (K-Bar, L-Bar) values length */
    uint8_t lenKPrime;                                       /*!< K' values length */
    uint8_t lenLPrime;                                       /*!< L' values length */
    uint8_t kIndices[CUPHY_CSIRS_MAX_KBAR_LBAR_LENGTH];      /*!< KBar indices */
    uint8_t kOffsets[CUPHY_CSIRS_MAX_KBAR_LBAR_LENGTH];      /*!< KBar offsets */
    uint8_t lIndices[CUPHY_CSIRS_MAX_KBAR_LBAR_LENGTH];      /*!< LBar indices */
    uint8_t lOffsets[CUPHY_CSIRS_MAX_KBAR_LBAR_LENGTH];      /*!< LBar offsets */
    uint8_t cdmGroupIndex[CUPHY_CSIRS_MAX_KBAR_LBAR_LENGTH]; /*!< CDM group index */
} CsirsSymbLocRow;

/**
 * @brief Tables used in CSI-RS signal generation algorithm
 */
typedef struct _CsirsTables
{
    CsirsSymbLocRow rowData[CUPHY_CSIRS_SYMBOL_LOCATION_TABLE_LENGTH];             /*!< resource mapping table */
    int8_t          seqTable[MAX_CDM_TYPE][CUPHY_CSIRS_MAX_SEQ_INDEX_COUNT][2][4]; /*!< wf/wt seq table layout: 2- Wf,Wt; 4 max(maxkprimelen, maxlprimelen) */
} CsirsTables;

/**
 * @brief Precoding matrix for CSI-RS
 */
typedef struct _cuphyPmWOneLayer_t cuphyCsirsPmWOneLayer_t;

/**
 * @brief: CUDA kernel setup for CSI-RS pipeline.
 * @param[in, out] pGenCsirsScramblingLaunchCfg: pointer to launch configuration for scrambling sequence generation.
 * @param[in, out] pGenCsirsTfSignalLaunchCfg: pointer to launch configuration for CSI-RS computation.
 * @param[in]  numParams: number of CSI-RS parameters in the batch across all cells (or in a cell group).
 * @return CUPHY_STATUS_SUCCESS or CUPHY_STATUS_INVALID_ARGUMENT.
 */
cuphyStatus_t cuphyCsirsKernelSelect(cuphyGenScramblingLaunchCfg_t*    pGenCsirsScramblingLaunchCfg,
                                     cuphyGenCsirsTfSignalLaunchCfg_t* pGenCsirsTfSignalLaunchCfg,
                                     uint32_t                          numParams);

/**
 * @brief: CUDA kernel setup for CSI-RS RX pipeline.
 * @param[in, out] pCsirsRxChEstLaunchCfg: pointer to launch configuration for CSI-RS RX channel estimation computation.
 * @param[in]      numUes: the total number of UEs.
 * @param[in]      maxNumRrcParamPerUe: the maximum number of CSI-RS parameters per UE.
 * @param[in]      maxNumPrbPerUe: the maximum number of PRBs per UE.
 * @return CUPHY_STATUS_SUCCESS or CUPHY_STATUS_INVALID_ARGUMENT.
 */
cuphyStatus_t cuphyCsirsRxKernelSelect(cuphyCsirsRxChEstLaunchCfg_t* pCsirsRxChEstLaunchCfg, int numUes, int maxNumRrcParamPerUe, int maxNumPrbPerUe);

/** @} */ /* END DL_CUPHY_CSIRS */


typedef struct _cuphyCsirsRxChEstBuffInfo
{
    cuphyTensorPrm_t* tChEstBuffer;   // Tensor parameters for CSI-RS channel estimation buffer. The tensor dimension is sizePrb x nPorts x nRxAnts.
                                      // Tensors reside in the GPU memory.
    uint16_t*         startPrb;       // The first PRB in ChEst buffer.
    uint16_t*         sizePrb;        // The number of PRBs which is the first dimension of tChEstBuffer tensor.
    uint8_t           nCsirs;         // The number of CSI-RS co-scheduled for this cell/UE. Maximum allowed: CUPHY_CSIRS_MAX_NUM_PARAMS.
} cuphyCsirsRxChEstBuffInfo_t;


typedef struct
{
    CUDA_KERNEL_NODE_PARAMS kernelNodeParamsDriver;
    void*                   kernelArgs[2];
} cuphyGenSrsTxLaunchCfg_t;

/**
 * @brief: CUDA kernel setup for SRS TX pipeline.
 * @param[in, out] pGenSrsTxLaunchCfg: pointer to launch configuration for SRS TX generation.
 * @param[in]  numParams: number of SRS TX parameters in the batch across all cells (or in a cell group).
 * @return CUPHY_STATUS_SUCCESS or CUPHY_STATUS_INVALID_ARGUMENT.
 */
cuphyStatus_t cuphySrsTxKernelSelect(cuphyGenSrsTxLaunchCfg_t* pGenSrsTxLaunchCfg, uint32_t numParams);

/**
 * \defgroup DL_CUPHY_PDSCH_DMRS  PDSCH DMRS
 *
 * This section describes the PDSCH (Physical DOwnlink Shared Channel) DMRS functions of the cuPHY
 * application programming interface.
 *
 * @{
 */

/**
 * @brief Struct that tracks parameters needed for rate-matching/modulation
 *        when CSI-RS parameters are present.
 */
struct PdschUeGrpParams
{
    // uint16_t suffices; made uint32_t for atomics
    uint32_t cumulative_skipped_REs[OFDM_SYMBOLS_PER_SLOT]; /*!< number of REs skipped up to including this current data symbol
                                                                 for this TB. Only first num_data_symbols are valid. */
    uint8_t tb_idx;                                         /*!< TB identifier; one of the TBs of this UE group */
};

/**
 * @brief Struct that tracks all necessary parameters for PDSCH DMRS computation.
 *        This struct is also used in PDSCH modulation.
 *        There is one PdschDmrsParams struct per TB.
 */
struct PdschDmrsParams
{
    void* cell_output_tensor_addr; /*!< output address for the cell this TB belong to.
                                              NB: replicates information across all TBs in the same cell. Could
                                              alternatively add a field with the cell index in cell group, and maintain
                                              a separate array of cell indices to be used both by DMRS and modulation. */

    uint64_t data_sym_loc; /*!< Starting from least significant bit the first 4 * num_data_symbols bits are valid and
                                    specify the location of the data symbols. 4 bits are used for each position. */

    uint32_t slot_number;      /*!< slotNumber */
    uint32_t cell_id;          /*!< cellId */
    uint8_t num_dmrs_symbols; /*!< number of DMRS symbols */
    uint8_t num_data_symbols; /*!< number of data symbols */
    uint8_t symbol_number;    /*!< index of initial symbol (0-based). */

    uint8_t  resourceAlloc; /*!< Resource Allocation type */
    uint32_t rbBitmap[MAX_RBMASK_UINT32_ELEMENTS]; /*!< bitmap indicating allocated RBs per FAPI Table 3-70 */
    uint16_t num_BWP_PRBs; /*!< number of PRBs in this bandwidth part. */
    uint16_t num_Rbs;      /*!< number of allocated RBs (Resource Blocks), at most 273.  0=Don't format modulator output */
    uint16_t start_Rb;     /*!< initial RB (0 indexing) */
    float    beta_dmrs;    /*!< DMRS amplitude scaling */
    float    beta_qam;     /*!< QAM amplitude scaling */
    uint8_t num_layers;   /*!< number of layers */

    uint8_t port_ids[MAX_DL_LAYERS_PER_TB]; /*!< at most 8 ports supported for DMRS configuration type 1 per UE, but this is per TB;
                                                  only the first num_layers values are valid; actual port is +1000 */
    uint32_t n_scid;                         /*!< scrambling Id used  */
    uint32_t dmrs_scid;                      /*!< DMRS scrambling Id */
    uint16_t dmrs_sym_loc; /*!< Starting from least significant bit the first 4 * num_drms_symbols bits are valid and
                                                  specify the location of the DMRS symbols. 4 bits are used for each position. */

    uint16_t BWP_start_PRB;                                 /*!< start PRB for this bandwidth part. Used only if ref_point is 1. */

    uint8_t cell_index_in_cell_group; /*!< Different than cell_id. */

    uint8_t ref_point;                                /*!< DMRS reference point: 0 or 1 */
    uint8_t enablePrcdBf;                             /*!< is pre-coding enabled */
    uint8_t Np;                                       /*!< number of antenna ports for this UE when precoding is enabled (enablePrcdBf true); 0 otherwise. */
    __half2 pmW[MAX_DL_LAYERS_PER_TB * MAX_DL_PORTS]; /*!< pre-coding matrix to be used only if enablePrcdBf is true with
                                                           num_layers rows and Np columns */
    uint8_t ueGrp_idx;                                /*!< UE group identifier associated with this TB */
    uint8_t dmrsCdmGrpsNoData1;                       /*!< is PDSCH CDM groups without data set to 1 for this TB */
    uint8_t nlAbove16{0};
};

/**
 * DMRS kernel launch config
 */
struct cuphyPdschDmrsLaunchConfig
{
    CUDA_KERNEL_NODE_PARAMS m_kernelNodeParams;
    void*                   m_desc;
    void*                   m_kernelArgs[1];
};
typedef struct cuphyPdschDmrsLaunchConfig* cuphyPdschDmrsLaunchConfig_t;

/** @brief: Compute descriptor size and alignment for PDSCH DMRS.
 *
 * @param[in,out] pDescrSizeBytes:  Size in bytes of descriptor
 * @param[in,out] pDescrAlignBytes: Alignment in bytes of descriptor
 * @return CUPHY_STATUS_SUCCESS or CUPHY_STATUS_INVALID_ARGUMENT
 */
cuphyStatus_t CUPHYWINAPI cuphyPdschDmrsGetDescrInfo(size_t* pDescrSizeBytes,
                                                     size_t* pDescrAlignBytes);

/**
 * @brief: Setup PDSCH DMRS component.
 * @param[in] pdschDmrsLaunchConfig: Pointer to DMRS launch config.
 * @param[in] dmrs_params: DMRS config. parameters struct array on the device, with # TBs entries.
 * @param[in] num_TBs: number of TBs.
 * @param[in] enable_precoding: Enabling pre-coding. Set to true if this batch has any UE with pre-coding enabled.
 * @param[in] dmrs_output_desc: output tensor descriptor; dimensions {273*12, 14, 16} tensor.
 * @param[in] dmrs_output_addr: pointer to output tensor data; each element is a complex number (half-precision).
 * @param[in] cpu_desc: Pointer to descriptor in CPU memory
 * @param[in] gpu_desc: Pointer to descriptor in GPU memory
 * @param[in] enable_desc_async_copy: async copy CPU descriptor into GPU if set.
 * @param[in] strm: CUDA stream for async copy
 * @return CUPHY_STATUS_SUCCESS or CUPHY_STATUS_INVALID_ARGUMENT or CUPHY_STATUS_MEMCPY_ERROR
 *         or CUPHY_STATUS_INTERNAL_ERROR
 */
cuphyStatus_t CUPHYWINAPI cuphySetupPdschDmrs(cuphyPdschDmrsLaunchConfig_t pdschDmrsLaunchConfig,
                                              PdschDmrsParams*             dmrs_params,
                                              int                          num_TBs,
                                              uint8_t                      enable_precoding,
                                              cuphyTensorDescriptor_t      dmrs_output_desc,
                                              void*                        dmrs_output_addr,
                                              void*                        cpu_desc,
                                              void*                        gpu_desc,
                                              uint8_t                      enable_desc_async_copy,
                                              cudaStream_t                 strm);

/** @} */ /* END DL_CUPHY_PDSCH_DMRS */

/**
 * PDSCH + CSI-RS Prepare kernel launch config
 */
struct cuphyPdschCsirsPrepLaunchConfig
{
    CUDA_KERNEL_NODE_PARAMS m_kernelNodeParams[3]; // 3rd kernel executes first
    void*                   m_desc;
    void*                   m_kernelArgs[1];
};
typedef struct cuphyPdschCsirsPrepLaunchConfig* cuphyPdschCsirsPrepLaunchConfig_t;


/** @brief: Compute descriptor size and alignment for PDSCH CSIRS Preprocessing work.
 *
 * @param[in,out] pDescrSizeBytes:  Size in bytes of descriptor
 * @param[in,out] pDescrAlignBytes: Alignment in bytes of descriptor
 * @return CUPHY_STATUS_SUCCESS or CUPHY_STATUS_INVALID_ARGUMENT
 */
cuphyStatus_t CUPHYWINAPI cuphyPdschCsirsPrepGetDescrInfo(size_t* pDescrSizeBytes,
                                                          size_t* pDescrAlignBytes);

/** @brief: Polar encoding and rate matching for control channel processing
 *  @param[in]  nInfoBits  : Number of information bits, range [1,164]
 *  @param[in]  nTxBits    : Number of rate-matched transmit bits, range [1, 8192]
 *  @param[in]  pInfoBits  : Pointer to GPU memory containing information bit stream packed in
 *                           a uint8_t array (with at least 32b alignment), size ceiling(nInfoBits/8), up to 21 bytes (164 bits)
 *  @param[in]  pNCodedBits: Pointer to CPU memory to store store the encoded bit length (valid values: 32,64,128,256,512)
 *  @param[out] pCodedBits : Pointer to GPU memory to store polar encoded bit stream packed in
 *                           a uint8_t array (with atleast 32b alignment), size ceiling(nMaxCodedBits/8) = 64 bytes
 *  @param[out] pTxBits    : Pointer to device memory for storing polar rate-matched transmit bit stream
 *                           packed in a uint8_t array (with atleast 32b alignment), size must be a multiple
 *                           of 4 bytes (padded to nearest 32b boundary) with max size being ceiling(nTxBits/8), upto 1024 bytes
 *  @param[in] procModeBmsk: Bit mask indicating DL (default) or UL Encoding
 * @param[in]   strm       : CUDA stream for kernel launch.
 * @return CUPHY_STATUS_SUCCESS or CUPHY_STATUS_INVALID_ARGUMENT or CUPHY_STATUS_UNSUPPORTED_ALIGNMENT
 *         or CUPHY_STATUS_INTERNAL_ERROR
 */
cuphyStatus_t CUPHYWINAPI cuphyPolarEncRateMatch(uint32_t       nInfoBits,
                                                 uint32_t       nTxBits,
                                                 uint8_t const* pInfoBits,
                                                 uint32_t*      pNCodedBits,
                                                 uint8_t*       pCodedBits,
                                                 uint8_t*       pTxBits,
                                                 uint32_t       procModeBmsk,
                                                 cudaStream_t   strm);

/**
 * @brief: CUDA kernel setup for SSB pipeline.
 * @param[in, out] pEncdRMLaunchCfg: pointer to the launch configuration for encoder rate match
 * @param[in, out] pSsbMapperLaunchCfg: pointer to the launch configuration for SSB mapper kernel
 * @param[in]  nSSBs: number of SSBs
 * @return CUPHY_STATUS_SUCCESS or CUPHY_STATUS_INVALID_ARGUMENT.
 */
cuphyStatus_t CUPHYWINAPI cuphySSBsKernelSelect(cuphyEncoderRateMatchMultiSSBLaunchCfg_t* pEncdRMLaunchCfg,
                                                cuphySsbMapperLaunchCfg_t*                pSsbMapperLaunchCfg,
                                                uint16_t                                  nSSBs);

/**
 * @brief: Run Polar encoding and rate matching for control channel processing
 * @param pEncdRmSSBCfg : pointer to rate matching launch config
 *  @param  pInfoBits  : Pointer to GPU memory containing information bit stream packed in
 *                           a uint8_t array (with at least 32b alignment), size ceiling(nInfoBits/8), up to 21 bytes (164 bits)
 *  @param  pCodedBits : Pointer to GPU memory to store polar encoded bit stream packed in
 *                           a uint8_t array (with atleast 32b alignment), size ceiling(nMaxCodedBits/8) = 64 bytes
 *  @param  pTxBits    : Pointer to device memory for storing polar rate-matched transmit bit stream
 *                           packed in a uint8_t array (with atleast 32b alignment), size must be a multiple
 *                           of 4 bytes (padded to nearest 32b boundary) with max size being ceiling(nTxBits/8), upto 1024 bytes
 * @param   nSSBs      : number of SSBs
 * @param   strm       : CUDA stream for kernel launch.
 * @return CUPHY_STATUS_SUCCESS or CUPHY_STATUS_INVALID_ARGUMENT.
 */
cuphyStatus_t CUPHYWINAPI cuphyRunPolarEncRateMatchSSBs(
    cuphyEncoderRateMatchMultiSSBLaunchCfg_t* pEncdRmSSBCfg,
    uint8_t const*                            pInfoBits,
    uint8_t*                                  pCodedBits,
    uint8_t*                                  pTxBits,
    uint16_t                                  nSSBs,
    cudaStream_t                              strm);

/**
 * \defgroup CUPHY_RAND Random number generation for cuPHY tensors
 *
 * This section describes the functions for populating cuPHY tensors with
 * random data.
 *
 * @{
 */
struct cuphyRNG;
/**
 *  cuPHY random number generator handle
 */
typedef struct cuphyRNG* cuphyRNG_t;

/******************************************************************/ /**
 * \brief Allocates and initializes a cuPHY random number generator
 *
 * Allocates a cuPHY random number generator and returns a handle in the
 * address provided by the caller.
 *
 * Returns ::CUPHY_STATUS_INVALID_ARGUMENT if \p pRNG is NULL.
 *
 * Returns ::CUPHY_STATUS_ALLOC_FAILED if a context cannot be allocated
 * on the host.
 *
 * Returns ::CUPHY_STATUS_SUCCESS if allocation and initialization were
 * successful.
 *
 * \param pRNG - Address to return the new ::cuphyRNG_t instance
 * \param seed - Random number generator seed
 * \param flags - Creation flags (currently unused)
 * \param strm - CUDA stream for initialization
 *
 * \return
 * ::CUPHY_STATUS_SUCCESS,
 * ::CUPHY_STATUS_ALLOC_FAILED,
 * ::CUPHY_STATUS_INVALID_ARGUMENT
 *
 * \sa ::cuphyStatus_t,::cuphyGetErrorName,::cuphyGetErrorString,::cuphyDestroyContext,::cuphyDestroyRandomNumberGenerator
 */
cuphyStatus_t CUPHYWINAPI cuphyCreateRandomNumberGenerator(cuphyRNG_t*        pRNG,
                                                           unsigned long long seed,
                                                           unsigned int       flags,
                                                           cudaStream_t       strm);

/******************************************************************/ /**
 * \brief Destroys a cuPHY random number generator
 *
 * Destroys a previously created cuPHY random number generator instance
 *
 * Returns ::CUPHY_STATUS_INVALID_ARGUMENT if \p rng is NULL.
 *
 * Returns ::CUPHY_STATUS_SUCCESS if destruction was successful.
 *
 * \param rng - Existing ::cuphyRNG_t instance
 *
 * \return
 * ::CUPHY_STATUS_SUCCESS,
 * ::CUPHY_STATUS_INVALID_ARGUMENT
 *
 * \sa ::cuphyStatus_t,::cuphyGetErrorName,::cuphyGetErrorString,::cuphyDestroyContext,::cuphyCreateRandomNumberGenerator
 */
cuphyStatus_t CUPHYWINAPI cuphyDestroyRandomNumberGenerator(cuphyRNG_t rng);

/******************************************************************/ /**
 * \brief Populate a cuPHY tensor with uniformly distributed random data
 *
 * Populates a cuPHY tensor with random data that has a uniform distribution,
 * using the given min/max range.
 * The minimum and maximum values are ignored for tensors of type CUPHY_BIT.
 * For CUPHY_BIT tensors with a first dimension that is not a multiple of
 * 32, high-order bits in the end of the last word will be set to zero.
 *
 * Returns ::CUPHY_STATUS_INVALID_ARGUMENT if \p rng is NULL.
 * Returns ::CUPHY_STATUS_UNSUPPORTED_TYPE if the type of the input
 *         tensor is complex.
 *
 * Returns ::CUPHY_STATUS_SUCCESS if kernel launch was successful.
 *
 * \param rng - Existing ::cuphyRNG_t instance
 * \param tDst - Descriptor for output tensor
 * \param pDst - Address of output tensor
 * \param minValue - Minimum value
 * \param maxValue - Maximum value
 * \param strm - CUDA stream for kernel launch
 *
 * \return
 * ::CUPHY_STATUS_SUCCESS,
 * ::CUPHY_STATUS_INVALID_ARGUMENT
 * ::CUPHY_STATUS_UNSUPPORTED_TYPE
 *
 * \sa ::cuphyStatus_t,::cuphyGetErrorName,::cuphyGetErrorString,::cuphyDestroyContext,::cuphyCreateRandomNumberGenerator,::cuphyDestroyRandomNumberGenerator
 */
cuphyStatus_t CUPHYWINAPI cuphyRandomUniform(cuphyRNG_t              rng,
                                             cuphyTensorDescriptor_t tDst,
                                             void*                   pDst,
                                             const cuphyVariant_t*   minValue,
                                             const cuphyVariant_t*   maxValue,
                                             cudaStream_t            strm);

/******************************************************************/ /**
 * \brief Populate a cuPHY tensor with random data with a normal distribution
 *
 * Populates a cuPHY tensor with random data that has a normal (Gaussian) distribution
 *
 * Returns ::CUPHY_STATUS_INVALID_ARGUMENT if \p rng is NULL.
 *
 * Returns ::CUPHY_STATUS_SUCCESS if kernel launch was successful.
 *
 * \param rng - Existing ::cuphyRNG_t instance
 * \param tDst - Descriptor for output tensor
 * \param pDst - Address of output tensor
 * \param mean - Mean value
 * \param stddev - Standard deviation
 * \param strm - CUDA stream for kernel launch
 *
 * \return
 * ::CUPHY_STATUS_SUCCESS,
 * ::CUPHY_STATUS_INVALID_ARGUMENT
 *
 * \sa ::cuphyStatus_t,::cuphyGetErrorName,::cuphyGetErrorString,::cuphyDestroyContext,::cuphyCreateRandomNumberGenerator,::cuphyDestroyRandomNumberGenerator
 */
cuphyStatus_t CUPHYWINAPI cuphyRandomNormal(cuphyRNG_t              rng,
                                            cuphyTensorDescriptor_t tDst,
                                            void*                   pDst,
                                            const cuphyVariant_t*   mean,
                                            const cuphyVariant_t*   stddev,
                                            cudaStream_t            strm);

/** @} */ /* END CUPHY_RAND */

/******************************************************************/ /**
 * \brief Convert an input variant to a given type
 *
 * Attempts to convert the given variant to a value of the specified cuPHY data type.
 * Integer conversions to a destination type that cannot represent the
 * source value will return ::CUPHY_STATUS_VALUE_OUT_OF_RANGE. For
 * floating point types, Inf values will be generated without an error.
 *
 * Returns ::CUPHY_STATUS_INVALID_ARGUMENT if \p v is NULL or \p t is CUPHY_VOID
 * Returns ::CUPHY_STATUS_INVALID_CONVERSION if conversion to the destination type is not supported
 * Returns ::CUPHY_STATUS_VALUE_OUT_OF_RANGE if the destination type cannot represent the source value
 * Returns ::CUPHY_STATUS_SUCCESS if conversion was successful
 *
 * \param v - address of variant to convert
 * \param t - destination data type
 *
 * \return
 * ::CUPHY_STATUS_SUCCESS,
 * ::CUPHY_STATUS_INVALID_ARGUMENT
 * ::CUPHY_STATUS_INVALID_CONVERSION
 * ::CUPHY_STATUS_VALUE_OUT_OF_RANGE
 *
 * \sa ::cuphyDataType_t,::cuphyStatus_t,::cuphyGetErrorName,::cuphyGetErrorString
 */
cuphyStatus_t CUPHYWINAPI cuphyConvertVariant(cuphyVariant_t* v,
                                              cuphyDataType_t t);

/******************************************************************/ /**
 * \brief Fill tensor memory with a specific value
 *
 * Populates tensor memory described by the given descriptor with a
 * single value.
 *
 * Returns ::CUPHY_STATUS_INVALID_ARGUMENT if \p tDst, \p pDst, or v is
 *         NULL, or if the type of the input variable \p v is CUPHY_VOID
 * Returns ::CUPHY_STATUS_INVALID_CONVERSION if conversion to the destination type is not supported
 * Returns ::CUPHY_STATUS_VALUE_OUT_OF_RANGE if the destination type cannot represent the source value
 * Returns ::CUPHY_STATUS_SUCCESS if the conversion process was initiated
 *
 * \param tDst - descriptor for output tensor
 * \param pDst - address of output tensor memory
 * \param v - address of variant to populate tensor with
 * \param strm - CUDA stream for invocation of fill operation
 *
 * \return
 * ::CUPHY_STATUS_SUCCESS,
 * ::CUPHY_STATUS_INVALID_ARGUMENT
 * ::CUPHY_STATUS_INVALID_CONVERSION
 * ::CUPHY_STATUS_VALUE_OUT_OF_RANGE
 *
 * \sa ::cuphyDataType_t,::cuphyStatus_t,::cuphyGetErrorName,::cuphyGetErrorString
 */
cuphyStatus_t CUPHYWINAPI cuphyFillTensor(cuphyTensorDescriptor_t tDst,
                                          void*                   pDst,
                                          const cuphyVariant_t*   v,
                                          cudaStream_t            strm);

/******************************************************************/ /**
 * \brief Duplicate an input tensor into a tiled output
 *
 * Populates tensor memory by replicating the input tensor using
 * one or more copies in each dimension.
 *
 * Returns ::CUPHY_STATUS_INVALID_ARGUMENT if \p tDst, \p pDst, \p tSrc,
 *         \p pSrc, or \p tileExtents is NULL, or if \p tileRank is 0
 * Returns ::CUPHY_STATUS_UNSUPPORTED_TYPE if the tensor type of \p tDst and
 *         \p tSrc do not match
 * Returns ::CUPHY_STATUS_SIZE_MISMATCH if the destination tensor
 *         dimensions are not equal to the product of the source dimensions
 *         and the tile extents
 * Returns ::CUPHY_STATUS_SUCCESS if the tiling operation was initiated
 *
 * \param tDst - descriptor for output tensor
 * \param pDst - address of output tensor memory
 * \param tSrc - descriptor for input tensor
 * \param pSrc - address of input tensor memory
 * \param tileRank - number of tile dimensions
 * \param tileExtents - array with the number of tiles in each dimension
 * \param strm - CUDA stream for invocation of tile operation
 *
 * \return
 * ::CUPHY_STATUS_SUCCESS,
 * ::CUPHY_STATUS_INVALID_ARGUMENT
 * ::CUPHY_STATUS_INVALID_CONVERSION
 * ::CUPHY_STATUS_VALUE_OUT_OF_RANGE
 *
 * \sa ::cuphyDataType_t,::cuphyStatus_t,::cuphyGetErrorName,::cuphyGetErrorString
 */
cuphyStatus_t CUPHYWINAPI cuphyTileTensor(cuphyTensorDescriptor_t tDst,
                                          void*                   pDst,
                                          cuphyTensorDescriptor_t tSrc,
                                          const void*             pSrc,
                                          int                     tileRank,
                                          const int*              tileExtents,
                                          cudaStream_t            strm);

/******************************************************************/ /**
 * \brief Perform an element-wise operation on one or more tensors
 *
 * Populates an output tensor by performing an element-wise operation
 * on input tensors.
 *
 * Returns ::CUPHY_STATUS_INVALID_ARGUMENT if \p tDst, \p pDst, \p tSrcA,
 *         \p pSrcA, \p tSrcB, or \p pSrcB is NULL
 * Returns ::CUPHY_STATUS_UNSUPPORTED_TYPE if the tensor types of \p tDst,
 *         \p tSrcA, and \p tSrcB do not match the requirements for \p elemOp
 * Returns ::CUPHY_STATUS_SIZE_MISMATCH if the dimensions of the source
 *         and destination tensors do not match.
 * Returns ::CUPHY_STATUS_INVALID_CONVERSION if the value for \p alpha or
 *         \p beta cannot be converted to the arithmetic/output type
 * Returns ::CUPHY_STATUS_VALUE_OUT_OF_RANGE if the value for \p alpha or
 *         \p beta is out of range for the arithmetic/output data type
 * Returns ::CUPHY_STATUS_SUCCESS if the element-wise operation was initiated
 *
 * Tensor Input Requirements:
 * <ul>
 *   <li>::CUPHY_ELEMWISE_ADD
 *       <ul>
 *          <li>destination and source A must be non-NULL</li>
 *          <li>destination and source A data types must be the same</li>
 *          <li>if source B is non-NULL, its data type must match destination and A</li>
 *       </ul>
 *   </li>
 *   <li>::CUPHY_ELEMWISE_MUL (currently unimplemented)
 *       <ul>
 *          <li>destination and source A must be non-NULL</li>
 *          <li>destination and source A data types must be the same</li>
 *          <li>if source B is non-NULL, its data type must match destination and A</li>
 *       </ul>
 *   </li>
 *   <li>::CUPHY_ELEMWISE_MIN (currently unimplemented)
 *       <ul>
 *          <li>destination, source A, and source B must be non-NULL</li>
 *          <li>destination, source A, and source B data types must be the same</li>
 *       </ul>
 *   </li>
 *   <li>::CUPHY_ELEMWISE_MAX (currently unimplemented)
 *       <ul>
 *          <li>destination, source A, and source B must be non-NULL</li>
 *          <li>destination, source A, and source B data types must be the same</li>
 *       </ul>
 *   </li>
 *   <li>::CUPHY_ELEMWISE_ABS (currently unimplemented)
 *       <ul>
 *          <li>destination and source A must be non-NULL</li>
 *          <li>source B must be NULL<li>
 *          <li>destination and source A data types must be the same</li>
 *       </ul>
 *   </li>
 *   <li>::CUPHY_ELEMWISE_BIT_XOR
 *       <ul>
 *          <li>destination, source A, and source B must be non-NULL</li>
 *          <li>destination, source A, and source B data types must be ::CUPHY_BIT</li>
 *       </ul>
 *   </li>
 * </ul>
 *
 * \param tDst - descriptor for output tensor
 * \param pDst - address of output tensor memory
 * \param tSrcA - descriptor for input tensor A
 * \param pSrcA - address of input tensor memory A
 * \param alpha - scaling value for input A
 * \param tSrcB - descriptor for input tensor B
 * \param pSrcB - address of input tensor memory B
 * \param beta - scaling value for input B
 * \param elemOp - ::cuphyElementWiseOp_t describing the operation to perform
 * \param strm - CUDA stream for invocation of tile operation
 *
 * \return
 * ::CUPHY_STATUS_SUCCESS,
 * ::CUPHY_STATUS_INVALID_ARGUMENT
 * ::CUPHY_STATUS_UNSUPPORTED_TYPE
 * ::CUPHY_STATUS_INVALID_CONVERSION
 * ::CUPHY_STATUS_VALUE_OUT_OF_RANGE
 *
 * \sa ::cuphyDataType_t,::cuphyStatus_t,::cuphyGetErrorName,::cuphyGetErrorString,::cuphyElementWiseOp_t
 */
cuphyStatus_t CUPHYWINAPI cuphyTensorElementWiseOperation(cuphyTensorDescriptor_t tDst,
                                                          void*                   pDst,
                                                          cuphyTensorDescriptor_t tSrcA,
                                                          const void*             pSrcA,
                                                          const cuphyVariant_t*   alpha,
                                                          cuphyTensorDescriptor_t tSrcB,
                                                          const void*             pSrcB,
                                                          const cuphyVariant_t*   beta,
                                                          cuphyElementWiseOp_t    elemOp,
                                                          cudaStream_t            strm);

/******************************************************************/ /**
 * \brief Perform a reduction operation on a tensor
 *
 * Populates an output tensor by performing an user-specified reduction
 * operation on an input tensor.
 *
 * The size of dimension \p reductionDim of the destination tensor
 * \p tDst should be equal to 1. All other dimensions of \p tDst should
 * be the same as the corresponding dimension in \p tSrc.
 *
 * Returns ::CUPHY_STATUS_INVALID_ARGUMENT if \p tDst, \p pDst, \p tSrc,
 *         or \p pSrc is NULL, or if \p reductionDim is less than 0 or
 *         greater than CUPHY_DIM_MAX
 * Returns ::CUPHY_STATUS_UNSUPPORTED_TYPE if the tensor types of \p tDst,
 *         and \p tSrc do not match the requirements for \p redOp
 * Returns ::CUPHY_STATUS_SIZE_MISMATCH if the dimensions of the source
 *         and destination tensors do not match the requirements for a
 *         reduction
 * Returns ::CUPHY_STATUS_SUCCESS if the reduction operation was initiated
 *
 * Tensor Input Requirements:
 * <ul>
 *   <li>::CUPHY_ELEMWISE_ADD
 *       <ul>
 *          <li>The following type pairs are supported:
 *              <ul>
 *                  <li>source = CUPHY_R_32F, destination = CUPHY_R_32F</li>
 *                  <li>source = CUPHY_BIT, destination = CUPHY_R_32U (count of bits in a column)</li>
 *              </ul>
 *          </li>
 *       </ul>
 *   </li>
 *   <li>::CUPHY_ELEMWISE_MIN (currently unimplemented)
 *       <ul>
 *          <li>destination and source tensors must be the same type</li>
 *       </ul>
 *   </li>
 *   <li>::CUPHY_ELEMWISE_MAX (currently unimplemented)
 *       <ul>
 *          <li>destination and source tensors must be the same type</li>
 *       </ul>
 *   </li>
 *</ul>
 *
 * \param tDst - descriptor for output tensor
 * \param pDst - address of output tensor memory
 * \param tSrc - descriptor for input tensor
 * \param pSrc - address of input tensor memory
 * \param redOp - ::cuphyReductionOp_t describing the operation to perform
 * \param reductionDim - dimension to reduce across
 * \param workspaceSize - size of workspace buffer (currently ignored)
 * \param workspace - address of workspace buffer (currently ignored)
 * \param strm - CUDA stream for invocation of tile operation
 *
 * \return
 * ::CUPHY_STATUS_SUCCESS,
 * ::CUPHY_STATUS_INVALID_ARGUMENT
 * ::CUPHY_STATUS_SIZE_MISMATCH
 * ::CUPHY_STATUS_UNSUPPORTED_TYPE
 *
 * \sa ::cuphyDataType_t,::cuphyStatus_t,::cuphyGetErrorName,::cuphyGetErrorString,::cuphyReductionOp_t
 */
cuphyStatus_t CUPHYWINAPI cuphyTensorReduction(cuphyTensorDescriptor_t tDst,
                                               void*                   pDst,
                                               cuphyTensorDescriptor_t tSrc,
                                               const void*             pSrc,
                                               cuphyReductionOp_t      redOp,
                                               int                     reductionDim,
                                               size_t                  workspaceSize,
                                               void*                   workspace,
                                               cudaStream_t            strm);

//------------------------------------------------------------------------------
// Legacy rate-match

// Graphs API

typedef struct
{
    dim3               blockDim;
    dim3               gridDim;
    const void**       llr_vec_in;
    void*              out;
    const PerTbParams* tbPrmsArray;
    int                descramblingOn;
} rmLaunchDescriptor;

cuphyStatus_t rmLaunchSetup(
    uint32_t            nTBs,
    uint32_t            CMax,
    uint32_t            EMax,
    rmLaunchDescriptor* rmDesc);

void createRMNode(cudaGraphNode_t* rmNode, cudaGraph_t graph, const cudaGraphNode_t* dependencies, uint32_t nDependencies, const rmLaunchDescriptor* rmDesc, int FP16orFP32);
void updateRMNode(cudaGraphNode_t* rmNode, cudaGraphExec_t graphExec, const rmLaunchDescriptor* rmDesc);

void rate_matching(
    uint32_t           CMax,           // maximum number of code blocks per transport blocks
    uint32_t           EMax,           // maximum input code block size in "soft-bits"
    uint32_t           nTb,            // number of input transport blocks
    const PerTbParams* d_tbPrmsArray,  // array of PerTbParams structs describing each input transport block
    const void**       in,             // array of input LLR arrays
    void*              out,            // rate-dematched, descrambled and layer de-mapped output LLRs
    int                FPconfig,       // 0: FP32 in, FP32 out; 1: FP16 in, FP32 out; 2: FP32 in, FP16 out; 3: FP16 in, FP16 out; other values: don't run
    int                descramblingOn, // enable/disable descrambling
    cudaStream_t       strm);

// struct cuphyRmDecoder;
typedef struct cuphyRmDecoder* cuphyRmDecoderHndl_t;

typedef struct
{
    CUDA_KERNEL_NODE_PARAMS kernelNodeParamsDriver;
    void*                   kernelArgs[2];
} cuphyRmDecoderLaunchCfg_t;

cuphyStatus_t CUPHYWINAPI cuphyRmDecoderGetDescrInfo(size_t* pDynDescrSizeBytes,
                                                     size_t* pDynDescrAlignBytes);

cuphyStatus_t CUPHYWINAPI cuphyCreateRmDecoder(cuphyContext_t        context,
                                               cuphyRmDecoderHndl_t* pHndl,
                                               unsigned int          flags,
                                               void*                 pMemoryFootprint);

cuphyStatus_t CUPHYWINAPI cuphySetupRmDecoder(cuphyRmDecoderHndl_t       hndl,
                                              uint16_t                   nCws,
                                              cuphyRmCwPrm_t*            pCwPrmsGpu,
                                              uint8_t                    enableCpuToGpuDescrAsyncCpy, // option to copy descriptors from CPU to GPU
                                              void*                      pCpuDynDesc,                 // pointer to descriptor in cpu
                                              void*                      pGpuDynDesc,                 // pointer to descriptor in gpu
                                              cuphyRmDecoderLaunchCfg_t* pLaunchCfg,                  // pointer to launch configuration
                                              cudaStream_t               strm);                                     // stream to perform copy

cuphyStatus_t CUPHYWINAPI cuphyDestroyRmDecoder(cuphyRmDecoderHndl_t hndl);

/**
 * \defgroup CUPHY_SIMPLEX_DECODER Simplex Decoder
 *
 * This section describes the Simplex decoder functions of the cuPHY
 * application programming interface.
 *
 * @{
 */

struct cuphySimplexDecoder;
typedef struct cuphySimplexDecoder* cuphySimplexDecoderHndl_t;

typedef struct
{
    CUDA_KERNEL_NODE_PARAMS kernelNodeParamsDriver;
    void*                   kernelArgs[2];
} cuphySimplexDecoderLaunchCfg_t;

cuphyStatus_t CUPHYWINAPI cuphySimplexDecoderGetDescrInfo(size_t* pDynDescrSizeBytes,
                                                          size_t* pDynDescrAlignBytes);

cuphyStatus_t CUPHYWINAPI cuphyCreateSimplexDecoder(cuphySimplexDecoderHndl_t* pHndl);

cuphyStatus_t CUPHYWINAPI cuphySetupSimplexDecoder(cuphySimplexDecoderHndl_t       simplexDecoderHndl,
                                                   uint16_t                        nCws,
                                                   cuphySimplexCwPrm_t*            pCwPrmsCpu,
                                                   cuphySimplexCwPrm_t*            pCwPrmsGpu,
                                                   uint8_t                         enableCpuToGpuDescrAsyncCpy, // option to copy descriptors from CPU to GPU
                                                   void*                           pCpuDynDesc,                 // pointer to descriptor in cpu
                                                   void*                           pGpuDynDesc,                 // pointer to descriptor in gpu
                                                   cuphySimplexDecoderLaunchCfg_t* pLaunchCfg,                  // pointer to launch configuration
                                                   cudaStream_t                    strm);                                          // stream to perform copy

cuphyStatus_t CUPHYWINAPI cuphyDestroySimplexDecoder(cuphySimplexDecoderHndl_t simplexDecoderHndl);

/** @} */ /* END CUPHY_SIMPLEX_DECODER */

/**
 * \defgroup CUPHY_PUCCH_F0_RECEIVER PUCCH F0 receiver
 *
 * This section describes the PUCCH F0 receiver functions of the cuPHY
 * application programming interface.
 *
 * @{
 */

struct cuphyPucchF0Rx;
/**
 * cuPHY PUCCH F0 receiver handle
 */
typedef struct cuphyPucchF0Rx* cuphyPucchF0RxHndl_t;

/**
 * cuPHY PUCCH F0 receiver launch configuration
 */
typedef struct
{
    CUDA_KERNEL_NODE_PARAMS kernelNodeParamsDriver;
    void*                   kernelArgs[2];
} cuphyPucchF0RxLaunchCfg_t;

/******************************************************************/ /**
 * \brief Helper to compute cuPHY PUCCH F0 receiver descriptor buffer sizes and alignments
 *
 * Computes cuPHY PUSCH PUCCH F0 receiver descriptor buffer sizes and alignments. To be used by the caller to
 * allocate these buffers (in CPU and GPU memories) and provide them to other PucchF0Rx APIs
 *
 * Returns ::CUPHY_STATUS_INVALID_ARGUMENT if \p pDynDescrSizeBytes and/or \p pDynDescrAlignBytes is NULL
 *
 * Returns ::CUPHY_STATUS_SUCCESS if allocation and initialization were
 * successful.
 *
 * \param pDynDescrSizeBytes   - Size in bytes of dynamic descriptor
 * \param pDynDescrAlignBytes  - Alignment in bytes of dynamic descriptor
 *
 * \return
 * ::CUPHY_STATUS_SUCCESS,
 * ::CUPHY_STATUS_INVALID_ARGUMENT
 *
 * \sa ::cuphyStatus_t,::cuphyGetErrorName,::cuphyGetErrorString
 */
cuphyStatus_t CUPHYWINAPI cuphyPucchF0RxGetDescrInfo(size_t* pDynDescrSizeBytes,
                                                     size_t* pDynDescrAlignBytes);

/******************************************************************/ /**
 * \brief Allocate and initialize a cuPHY PucchF0Rx object
 *
 * Allocates a cuPHY pucch F0 receiver object and returns a handle in the
 * address provided by the caller.
 *
 * Returns ::CUPHY_STATUS_INVALID_ARGUMENT if \p pPucchF0RxHndl  is NULL.
 *
 * Returns ::CUPHY_STATUS_ALLOC_FAILED if a pucchF0Rx object cannot be allocated
 *
 * Returns ::CUPHY_STATUS_SUCCESS if allocation and initialization were successful
 *
 * \param pPucchF0RxHndl     - Address to return the new pucchF0Rx instance
 * \param strm               - CUDA stream for async copies
 *
 * \return
 * ::CUPHY_STATUS_SUCCESS,
 * ::CUPHY_STATUS_ALLOC_FAILED,
 * ::CUPHY_STATUS_INVALID_ARGUMENT
 *
::cuphyStatus_t,::cuphyGetErrorName,::cuphyGetErrorString,::cuphyPucchF0RxGetDescrInfo,::cuphySetupPucchF0Rx */

cuphyStatus_t CUPHYWINAPI cuphyCreatePucchF0Rx(cuphyPucchF0RxHndl_t* pPucchF0RxHndl, cudaStream_t strm);

/******************************************************************/ /**
 * \brief Setup cuPHY PucchF0Rx for slot processing
 *
 * Setup cuPHY PUCCH F0 receiver in preparation towards slot execution
 *
 * Returns ::CUPHY_STATUS_SUCCESS if setup is successful.
 *
 * Returns ::CUPHY_STATUS_INVALID_ARGUMENT if \p pucchF0RxHndl and/or \p pDataRx and/or \p pF0UcisOut and/or
 * \p pF0UciPrms  and/or \p pCpuDynDesc  and/or \p pDynDescrsGpu is NULL.
 *
 * \param pucchF0RxHndl                 - Handle to previously created PucchF0Rx instance
 * \param pDataRx                       - Pointer to received data tensor parameters
 * \param pF0UcisOut                    - Pointer to F0 uci output buffer
 * \param nCells                        - Number of cells
 * \param nF0Ucis                       - Number of F0 ucis to process
 * \param enableUlRxBf                  - UL RX Beamforning enabling flag
 * \param pF0UciPrms                    - Pointer to F0 uci parameters
 * \param pCmnCellPrms                  - Common cell parameters: number of gNB receive antennas, current slot number, gNB hopping ID
 * \param enableCpuToGpuDescrAsyncCpy   - Flag when set enables async copy of CPU descriptor into GPU
 * \param pCpuDynDesc                   - Pointer to dynamic descriptor in CPU memory
 * \param pGpuDynDesc                   - Pointer to dynamic descriptor in GPU memory
 * \param pLaunchCfg                    - Pointer to channel estimation launch configurations
 * \param strm                          - CUDA stream for descriptor copy operation
 *
 * \return
 * ::CUPHY_STATUS_SUCCESS,
 * ::CUPHY_STATUS_INVALID_ARGUMENT
 *
 * \sa ::cuphyStatus_t,::cuphyGetErrorName,::cuphyGetErrorString,::cuphyPucchF0RxGetDescrInfo,::cuphyDestroyPucchF0Rx
 */

cuphyStatus_t CUPHYWINAPI
cuphySetupPucchF0Rx(cuphyPucchF0RxHndl_t       pucchF0RxHndl,
                    cuphyTensorPrm_t*          pDataRx,
                    cuphyPucchF0F1UciOut_t*    pF0UcisOut,
                    uint16_t                   nCells,
                    uint16_t                   nF0Ucis,
                    uint8_t                    enableUlRxBf,
                    cuphyPucchUciPrm_t*        pF0UciPrms,
                    cuphyPucchCellPrm_t*       pCmnCellPrms,
                    uint8_t                    enableCpuToGpuDescrAsyncCpy,
                    void*                      pCpuDynDesc,
                    void*                      pGpuDynDesc,
                    cuphyPucchF0RxLaunchCfg_t* pLaunchCfg,
                    cudaStream_t               strm);

/******************************************************************/ /**
 * \brief Destroys a cuPHY PUCCH F0 receiver object
 *
 * Destroys a cuPHY PUCCH F0 receiver object that was previously
 * created by ::cuphyCreatePucchF0Rx. The handle provided to this function
 * should not be used for any operations after this function returns.
 *
 * Returns ::CUPHY_STATUS_INVALID_ARGUMENT if \p pucchF0RxHndl is NULL.
 *
 * Returns ::CUPHY_STATUS_SUCCESS if destruction was successful.
 *
 * \param pucchF0RxHndl - handle to previously allocated PuschRxChEst instance
 *
 * \return
 * ::CUPHY_STATUS_SUCCESS,
 * ::CUPHY_STATUS_INVALID_ARGUMENT
 *
 * \sa ::cuphyStatus_t,::cuphyGetErrorName,::cuphyGetErrorString,::cuphyPucchF0RxGetDescrInfo,::cuphyCreatePucchF0Rx,::cuphySetupPucchF0Rx
 */
cuphyStatus_t CUPHYWINAPI cuphyDestroyPucchF0Rx(cuphyPucchF0RxHndl_t pucchF0RxHndl);

/** @} */ /* END CUPHY_PUCCH_F0_RECEIVER */

////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////

/**
 * \defgroup CUPHY_PUCCH_F1_RECEIVER PUCCH F1 receiver
 *
 * This section describes the PUCCH F1 receiver functions of the cuPHY
 * application programming interface.
 *
 * @{
 */

struct cuphyPucchF1Rx;
/**
 * cuPHY PUCCH F1 receiver handle
 */
typedef struct cuphyPucchF1Rx* cuphyPucchF1RxHndl_t;

/**
 * cuPHY PUCCH F1 receiver launch configuration
 */
typedef struct
{
    CUDA_KERNEL_NODE_PARAMS kernelNodeParamsDriver;
    void*                   kernelArgs[2];
} cuphyPucchF1RxLaunchCfg_t;

/******************************************************************/ /**
 * \brief Helper to compute cuPHY PUCCH F1 receiver descriptor buffer sizes and alignments
 *
 * Computes cuPHY PUSCH PUCCH F1 receiver descriptor buffer sizes and alignments. To be used by the caller to
 * allocate these buffers (in CPU and GPU memories) and provide them to other PucchF1Rx APIs
 *
 * Returns ::CUPHY_STATUS_INVALID_ARGUMENT if \p pDynDescrSizeBytes and/or \p pDynDescrAlignBytes is NULL
 *
 * Returns ::CUPHY_STATUS_SUCCESS if allocation and initialization were
 * successful.
 *
 * \param pDynDescrSizeBytes   - Size in bytes of dynamic descriptor
 * \param pDynDescrAlignBytes  - Alignment in bytes of dynamic descriptor
 *
 * \return
 * ::CUPHY_STATUS_SUCCESS,
 * ::CUPHY_STATUS_INVALID_ARGUMENT
 *
 * \sa ::cuphyStatus_t,::cuphyGetErrorName,::cuphyGetErrorString,::cuphyCreatePucchF1Rx,::cuphyDestroyPucchF1Rx
 */
cuphyStatus_t CUPHYWINAPI cuphyPucchF1RxGetDescrInfo(size_t* pDynDescrSizeBytes,
                                                     size_t* pDynDescrAlignBytes);

/******************************************************************/ /**
 * \brief Allocate and initialize a cuPHY PucchF1Rx object
 *
 * Allocates a cuPHY pucch F1 receiver object and returns a handle in the
 * address provided by the caller.
 *
 * Returns ::CUPHY_STATUS_INVALID_ARGUMENT if \p pPucchF1RxHndl  is NULL.
 *
 * Returns ::CUPHY_STATUS_ALLOC_FAILED if a pucchF1Rx object cannot be allocated
 *
 * Returns ::CUPHY_STATUS_SUCCESS if allocation and initialization were successful
 *
 * \param pPucchF1RxHndl     - Address to return the new pucchF1Rx instance
 * \param strm               - CUDA stream for async copies
 *
 * \return
 * ::CUPHY_STATUS_SUCCESS,
 * ::CUPHY_STATUS_ALLOC_FAILED,
 * ::CUPHY_STATUS_INVALID_ARGUMENT
 *
::cuphyStatus_t,::cuphyGetErrorName,::cuphyGetErrorString,::cuphyPucchF1RxGetDescrInfo,::cuphySetupPucchF1Rx */

cuphyStatus_t CUPHYWINAPI cuphyCreatePucchF1Rx(cuphyPucchF1RxHndl_t* pPucchF1RxHndl, cudaStream_t strm);

/******************************************************************/ /**
 * \brief Setup cuPHY PucchF1Rx for slot processing
 *
 * Setup cuPHY PUCCH F1 receiver in preparation towards slot execution
 *
 * Returns ::CUPHY_STATUS_SUCCESS if setup is successful.
 *
 * Returns ::CUPHY_STATUS_INVALID_ARGUMENT if \p pucchF1RxHndl and/or \p pDataRx and/or \p pF1UcisOut and/or
 * \p pF1UciPrms  and/or \p pCpuDynDesc  and/or \p pDynDescrsGpu is NULL.
 *
 * \param pucchF1RxHndl                 - Handle to previously created PucchF1Rx instance
 * \param pDataRx                       - Pointer to received data tensor parameters
 * \param pF1UcisOut                    - Pointer to F1 uci output buffer
 * \param nCells                        - Number of cells
 * \param nF1Ucis                       - Number of F1 ucis to process
 * \param enableUlRxBf                  - UL RX Beamforning enabling flag
 * \param pF1UciPrms                    - Pointer to F1 uci parameters
 * \param pCmnCellPrms                  - Common cell parameters: number of gNB receive antennas, current slot number, gNB hopping ID
 * \param enableCpuToGpuDescrAsyncCpy   - Flag when set enables async copy of CPU descriptor into GPU
 * \param pCpuDynDesc                   - Pointer to dynamic descriptor in CPU memory
 * \param pGpuDynDesc                   - Pointer to dynamic descriptor in GPU memory
 * \param pLaunchCfg                    - Pointer to channel estimation launch configurations
 * \param strm                          - CUDA stream for descriptor copy operation
 *
 * \return
 * ::CUPHY_STATUS_SUCCESS,
 * ::CUPHY_STATUS_INVALID_ARGUMENT
 *
 * \sa ::cuphyStatus_t,::cuphyGetErrorName,::cuphyGetErrorString,::cuphyPucchF1RxGetDescrInfo,::cuphyDestroyPucchF1Rx
 */

cuphyStatus_t CUPHYWINAPI
cuphySetupPucchF1Rx(cuphyPucchF1RxHndl_t       pucchF1RxHndl,
                    cuphyTensorPrm_t*          pDataRx,
                    cuphyPucchF0F1UciOut_t*    pF1UcisOut,
                    uint16_t                   nCells,
                    uint16_t                   nF1Ucis,
                    uint8_t                    enableUlRxBf,
                    cuphyPucchUciPrm_t*        pF1UciPrms,
                    cuphyPucchCellPrm_t*       pCmnCellPrms,
                    uint8_t                    enableCpuToGpuDescrAsyncCpy,
                    void*                      pCpuDynDesc,
                    void*                      pGpuDynDesc,
                    cuphyPucchF1RxLaunchCfg_t* pLaunchCfg,
                    cudaStream_t               strm);

/******************************************************************/ /**
 * \brief Destroys a cuPHY PUCCH F1 receiver object
 *
 * Destroys a cuPHY PUCCH F1 receiver object that was previously
 * created by ::cuphyCreatePucchF1Rx. The handle provided to this function
 * should not be used for any operations after this function returns.
 *
 * Returns ::CUPHY_STATUS_INVALID_ARGUMENT if \p pucchF1RxHndl is NULL.
 *
 * Returns ::CUPHY_STATUS_SUCCESS if destruction was successful.
 *
 * \param pucchF1RxHndl - handle to previously allocated pucchF1Rx instance
 *
 * \return
 * ::CUPHY_STATUS_SUCCESS,
 * ::CUPHY_STATUS_INVALID_ARGUMENT
 *
 * \sa ::cuphyStatus_t,::cuphyGetErrorName,::cuphyGetErrorString,::cuphyPucchF1RxGetDescrInfo,::cuphyCreatePucchF1Rx,::cuphySetupPucchF1Rx
 */
cuphyStatus_t CUPHYWINAPI cuphyDestroyPucchF1Rx(cuphyPucchF1RxHndl_t pucchF1RxHndl);

/** @} */ /* END CUPHY_PUCCH_F1_RECEIVER */

////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////

/**
 * \defgroup CUPHY_PUCCH_F2_RECEIVER PUCCH F2 receiver
 *
 * This section describes the PUCCH F2 receiver functions of the cuPHY
 * application programming interface.
 *
 * @{
 */

struct cuphyPucchF2Rx;
/**
 * cuPHY PUCCH F2 receiver handle
 */
typedef struct cuphyPucchF2Rx* cuphyPucchF2RxHndl_t;

/**
 * cuPHY PUCCH F2 receiver launch configuration
 */
typedef struct
{
    CUDA_KERNEL_NODE_PARAMS kernelNodeParamsDriver;
    void*                   kernelArgs[2];
} cuphyPucchF2RxLaunchCfg_t;

/******************************************************************/ /**
 * \brief Helper to compute cuPHY PUCCH F2 receiver descriptor buffer sizes and alignments
 *
 * Computes cuPHY PUSCH PUCCH F2 receiver descriptor buffer sizes and alignments. To be used by the caller to
 * allocate these buffers (in CPU and GPU memories) and provide them to other PucchF2Rx APIs
 *
 * Returns ::CUPHY_STATUS_INVALID_ARGUMENT if \p pDynDescrSizeBytes and/or \p pDynDescrAlignBytes is NULL
 *
 * Returns ::CUPHY_STATUS_SUCCESS if allocation and initialization were
 * successful.
 *
 * \param pDynDescrSizeBytes   - Size in bytes of dynamic descriptor
 * \param pDynDescrAlignBytes  - Alignment in bytes of dynamic descriptor
 *
 * \return
 * ::CUPHY_STATUS_SUCCESS,
 * ::CUPHY_STATUS_INVALID_ARGUMENT
 *
 * \sa ::cuphyStatus_t,::cuphyGetErrorName,::cuphyGetErrorString,::cuphyCreatePucchF2Rx,::cuphyDestroyPucchF2Rx
 */
cuphyStatus_t CUPHYWINAPI cuphyPucchF2RxGetDescrInfo(size_t* pDynDescrSizeBytes,
                                                     size_t* pDynDescrAlignBytes);

/******************************************************************/ /**
 * \brief Allocate and initialize a cuPHY PucchF2Rx object
 *
 * Allocates a cuPHY pucch F2 receiver object and returns a handle in the
 * address provided by the caller.
 *
 * Returns ::CUPHY_STATUS_INVALID_ARGUMENT if \p pPucchF2RxHndl  is NULL.
 *
 * Returns ::CUPHY_STATUS_ALLOC_FAILED if a pucchF2Rx object cannot be allocated
 *
 * Returns ::CUPHY_STATUS_SUCCESS if allocation and initialization were successful
 *
 * \param pPucchF2RxHndl     - Address to return the new pucchF2Rx instance
 * \param strm               - CUDA stream for async copies
*
 * \return
 * ::CUPHY_STATUS_SUCCESS,
 * ::CUPHY_STATUS_ALLOC_FAILED,
 * ::CUPHY_STATUS_INVALID_ARGUMENT
 *
::cuphyStatus_t,::cuphyGetErrorName,::cuphyGetErrorString,::cuphyPucchF2RxGetDescrInfo,::cuphySetupPucchF2Rx */

cuphyStatus_t CUPHYWINAPI cuphyCreatePucchF2Rx(cuphyPucchF2RxHndl_t* pPucchF2RxHndl, cudaStream_t strm);

/******************************************************************/ /**
 * \brief Setup cuPHY PucchF2Rx for slot processing
 *
 * Setup cuPHY PUCCH F2 receiver in preparation towards slot execution
 *
 * Returns ::CUPHY_STATUS_SUCCESS if setup is successful.
 *
 * Returns ::CUPHY_STATUS_INVALID_ARGUMENT if \p pucchF2RxHndl and/or \p pDataRx and/or
 * \p pF2UciPrms  and/or \p pCpuDynDesc  and/or \p pDynDescrsGpu is NULL.
 *
 * \param pucchF2RxHndl                 - Handle to previously created PucchF2Rx instance
 * \param pDataRx                       - Pointer to received data tensor parameters
 * \param pDescramLLRaddrs              - pointer to descrambled segment 1 LLR addresses
 * \param pDTXflags                     - pointer to DTX flag buffer
 * \param pSinr                         - pointer to SINR buffer
 * \param pRssi                         - pointer to RSSI buffer
 * \param pRsrp                         - pointer to RSRP buffer
 * \param pInterf                       - pointer to interference level buffer
 * \param pNoiseVar                     - pointer to Noise Var buffer
 * \param pTaEst                        - pointer to timing advance buffer
 * \param nCells                        - Number of cells
 * \param nF2Ucis                       - Number of F2 ucis to process
 * \param enableUlRxBf                  - UL RX Beamforning enabling flag
 * \param pF2UciPrms                    - Pointer to F2 uci parameters
 * \param pCmnCellPrms                  - Common cell parameters: number of gNB receive antennas, current slot number, gNB hopping ID
 * \param enableCpuToGpuDescrAsyncCpy   - Flag when set enables async copy of CPU descriptor into GPU
 * \param pCpuDynDesc                   - Pointer to dynamic descriptor in CPU memory
 * \param pGpuDynDesc                   - Pointer to dynamic descriptor in GPU memory
 * \param pLaunchCfg                    - Pointer to channel estimation launch configurations
 * \param strm                          - CUDA stream for descriptor copy operation
 *
 * \return
 * ::CUPHY_STATUS_SUCCESS,
 * ::CUPHY_STATUS_INVALID_ARGUMENT
 *
 * \sa ::cuphyStatus_t,::cuphyGetErrorName,::cuphyGetErrorString,::cuphyPucchF2RxGetDescrInfo,::cuphyDestroyPucchF2Rx
 */

cuphyStatus_t CUPHYWINAPI
cuphySetupPucchF2Rx(cuphyPucchF2RxHndl_t       pucchF2RxHndl,
                    cuphyTensorPrm_t*          pDataRx,
                    __half**                   pDescramLLRaddrs,
                    uint8_t*                   pDTXflags,
                    float*                     pSinr,
                    float*                     pRssi,
                    float*                     pRsrp,
                    float*                     pInterf,
                    float*                     pNoiseVar,
                    float*                     pTaEst,
                    uint16_t                   nCells,
                    uint16_t                   nF2Ucis,
                    uint8_t                    enableUlRxBf,
                    cuphyPucchUciPrm_t*        pF2UciPrms,
                    cuphyPucchCellPrm_t*       pCmnCellPrms,
                    uint8_t                    enableCpuToGpuDescrAsyncCpy,
                    void*                      pCpuDynDesc,
                    void*                      pGpuDynDesc,
                    cuphyPucchF2RxLaunchCfg_t* pLaunchCfg,
                    cudaStream_t               strm);

/******************************************************************/ /**
 * \brief Destroys a cuPHY PUCCH F2 receiver object
 *
 * Destroys a cuPHY PUCCH F2 receiver object that was previously
 * created by ::cuphyCreatePucchF2Rx. The handle provided to this function
 * should not be used for any operations after this function returns.
 *
 * Returns ::CUPHY_STATUS_INVALID_ARGUMENT if \p pucchF2RxHndl is NULL.
 *
 * Returns ::CUPHY_STATUS_SUCCESS if destruction was successful.
 *
 * \param pucchF2RxHndl - handle to previously allocated pucchF2Rx instance
 *
 * \return
 * ::CUPHY_STATUS_SUCCESS,
 * ::CUPHY_STATUS_INVALID_ARGUMENT
 *
 * \sa ::cuphyStatus_t,::cuphyGetErrorName,::cuphyGetErrorString,::cuphyPucchF2RxGetDescrInfo,::cuphyCreatePucchF2Rx,::cuphySetupPucchF2Rx
 */
cuphyStatus_t CUPHYWINAPI cuphyDestroyPucchF2Rx(cuphyPucchF2RxHndl_t pucchF2RxHndl);

/** @} */ /* END CUPHY_PUCCH_F2_RECEIVER */

////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////

/**
 * \defgroup CUPHY_PUCCH_F3_RECEIVER PUCCH F3 receiver
 *
 * This section describes the PUCCH F3 receiver functions of the cuPHY
 * application programming interface.
 *
 * @{
 */

struct cuphyPucchF3Rx;
/**
 * cuPHY PUCCH F3 receiver handle
 */
typedef struct cuphyPucchF3Rx* cuphyPucchF3RxHndl_t;

/**
 * cuPHY PUCCH F3 receiver launch configuration
 */
typedef struct
{
    CUDA_KERNEL_NODE_PARAMS kernelNodeParamsDriver;
    void*                   kernelArgs[2];
} cuphyPucchF3RxLaunchCfg_t;

/******************************************************************/ /**
 * \brief Helper to compute cuPHY PUCCH F3 receiver descriptor buffer sizes and alignments
 *
 * Computes cuPHY PUSCH PUCCH F3 receiver descriptor buffer sizes and alignments. To be used by the caller to
 * allocate these buffers (in CPU and GPU memories) and provide them to other PucchF3Rx APIs
 *
 * Returns ::CUPHY_STATUS_INVALID_ARGUMENT if \p pDynDescrSizeBytes and/or \p pDynDescrAlignBytes is NULL
 *
 * Returns ::CUPHY_STATUS_SUCCESS if allocation and initialization were
 * successful.
 *
 * \param pDynDescrSizeBytes   - Size in bytes of dynamic descriptor
 * \param pDynDescrAlignBytes  - Alignment in bytes of dynamic descriptor
 *
 * \return
 * ::CUPHY_STATUS_SUCCESS,
 * ::CUPHY_STATUS_INVALID_ARGUMENT
 *
 * \sa ::cuphyStatus_t,::cuphyGetErrorName,::cuphyGetErrorString,::cuphyCreatePucchF3Rx,::cuphyDestroyPucchF3Rx
 */
cuphyStatus_t CUPHYWINAPI cuphyPucchF3RxGetDescrInfo(size_t* pDynDescrSizeBytes,
                                                     size_t* pDynDescrAlignBytes);

/******************************************************************/ /**
 * \brief Allocate and initialize a cuPHY PucchF3Rx object
 *
 * Allocates a cuPHY pucch F3 receiver object and returns a handle in the
 * address provided by the caller.
 *
 * Returns ::CUPHY_STATUS_INVALID_ARGUMENT if \p pPucchF3RxHndl  is NULL.
 *
 * Returns ::CUPHY_STATUS_ALLOC_FAILED if a pucchF3Rx object cannot be allocated
 *
 * Returns ::CUPHY_STATUS_SUCCESS if allocation and initialization were successful
 *
 * \param pPucchF3RxHndl     - Address to return the new pucchF3Rx instance
 * \param strm               - CUDA stream for async copies
 *
 * \return
 * ::CUPHY_STATUS_SUCCESS,
 * ::CUPHY_STATUS_ALLOC_FAILED,
 * ::CUPHY_STATUS_INVALID_ARGUMENT
 *
::cuphyStatus_t,::cuphyGetErrorName,::cuphyGetErrorString,::cuphyPucchF3RxGetDescrInfo,::cuphySetupPucchF3Rx */

cuphyStatus_t CUPHYWINAPI cuphyCreatePucchF3Rx(cuphyPucchF3RxHndl_t* pPucchF3RxHndl, cudaStream_t strm);

/******************************************************************/ /**
 * \brief Setup cuPHY PucchF3Rx for slot processing
 *
 * Setup cuPHY PUCCH F3 receiver in preparation towards slot execution
 *
 * Returns ::CUPHY_STATUS_SUCCESS if setup is successful.
 *
 * Returns ::CUPHY_STATUS_INVALID_ARGUMENT if \p pucchF3RxHndl and/or \p pDataRx and/or \p pF3UcisOut and/or
 * \p pF3UciPrms  and/or \p pCpuDynDesc  and/or \p pDynDescrsGpu is NULL.
 *
 * \param pucchF3RxHndl                 - Handle to previously created PucchF3Rx instance
 * \param pDataRx                       - Pointer to received data tensor parameters
 * \param pDescramLLRaddrs              - pointer to descrambled segment 1 LLR addresses
 * \param pDTXflags                     - pointer to DTX flag buffer
 * \param pSinr                         - pointer to SINR buffer
 * \param pRssi                         - pointer to RSSI buffer
 * \param pRsrp                         - pointer to RSRP buffer
 * \param pInterf                       - pointer to interference level buffer
 * \param pNoiseVar                     - pointer to Noise Var buffer
 * \param pTaEst                        - pointer to timing advance buffer
 * \param nCells                        - Number of cells
 * \param nF3Ucis                       - Number of F3 ucis to process
 * \param enableUlRxBf                  - UL RX Beamforning enabling flag
 * \param pF3UciPrms                    - Pointer to F3 uci parameters
 * \param pCmnCellPrms                  - Common cell parameters: number of gNB receive antennas, current slot number, gNB hopping ID
 * \param enableCpuToGpuDescrAsyncCpy   - Flag when set enables async copy of CPU descriptor into GPU
 * \param pCpuDynDesc                   - Pointer to dynamic descriptor in CPU memory
 * \param pGpuDynDesc                   - Pointer to dynamic descriptor in GPU memory
 * \param pLaunchCfg                    - Pointer to channel estimation launch configurations
 * \param strm                          - CUDA stream for descriptor copy operation
 *
 * \return
 * ::CUPHY_STATUS_SUCCESS,
 * ::CUPHY_STATUS_INVALID_ARGUMENT
 *
 * \sa ::cuphyStatus_t,::cuphyGetErrorName,::cuphyGetErrorString,::cuphyPucchF3RxGetDescrInfo,::cuphyDestroyPucchF3Rx
 */

cuphyStatus_t CUPHYWINAPI
cuphySetupPucchF3Rx(cuphyPucchF3RxHndl_t       pucchF3RxHndl,
                    cuphyTensorPrm_t*          pDataRx,
                    __half**                   pDescramLLRaddrs,
                    uint8_t*                   pDTXflags,
                    float*                     pSinr,
                    float*                     pRssi,
                    float*                     pRsrp,
                    float*                     pInterf,
                    float*                     pNoiseVar,
                    float*                     pTaEst,
                    uint16_t                   nCells,
                    uint16_t                   nF3Ucis,
                    uint8_t                    enableUlRxBf,
                    cuphyPucchUciPrm_t*        pF3UciPrms,
                    cuphyPucchCellPrm_t*       pCmnCellPrms,
                    uint8_t                    enableCpuToGpuDescrAsyncCpy,
                    void*                      pCpuDynDesc,
                    void*                      pGpuDynDesc,
                    cuphyPucchF3RxLaunchCfg_t* pLaunchCfg,
                    cudaStream_t               strm);

/******************************************************************/ /**
 * \brief Destroys a cuPHY PUCCH F3 receiver object
 *
 * Destroys a cuPHY PUCCH F3 receiver object that was previously
 * created by ::cuphyCreatePucchF3Rx. The handle provided to this function
 * should not be used for any operations after this function returns.
 *
 * Returns ::CUPHY_STATUS_INVALID_ARGUMENT if \p pucchF3RxHndl is NULL.
 *
 * Returns ::CUPHY_STATUS_SUCCESS if destruction was successful.
 *
 * \param pucchF3RxHndl - handle to previously allocated pucchF3Rx instance
 *
 * \return
 * ::CUPHY_STATUS_SUCCESS,
 * ::CUPHY_STATUS_INVALID_ARGUMENT
 *
 * \sa ::cuphyStatus_t,::cuphyGetErrorName,::cuphyGetErrorString,::cuphyPucchF3RxGetDescrInfo,::cuphyCreatePucchF3Rx,::cuphySetupPucchF3Rx
 */
cuphyStatus_t CUPHYWINAPI cuphyDestroyPucchF3Rx(cuphyPucchF3RxHndl_t pucchF3RxHndl);

/** @} */ /* END CUPHY_PUCCH_F3_RECEIVER */

////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////

/**
 * \defgroup CUPHY_PUCCH_F3_CSI2_CTRL : computes number of CSI-P2 bits for PUCCH format 3. Setups CSI-P2 backend.
 *
 * This section describes application programming interface for the PUCCH format 3 CSI-P2 control kernel
 *
 * @{
 */

struct cuphyPucchF3Csi2Ctrl;
/**
 * cuPHY pucchF3Csi2Ctrl handle
 */
typedef struct cuphyPucchF3Csi2Ctrl* cuphyPucchF3Csi2CtrlHndl_t;

/**
 * cuPHY PUCCH format 3 Csi2 control, launch configuration
 */
typedef struct
{
    CUDA_KERNEL_NODE_PARAMS kernelNodeParamsDriver;
    void*                   kernelArgs[2];
} cuphyPucchF3Csi2CtrlLaunchCfg_t;

/******************************************************************/ /**
 * \brief Helper to compute pucchF3Csi2Ctrl descriptor buffer sizes and alignments
 *
 * Computes pucchF3Csi2Ctrl descriptor buffer sizes and alignments. To be used by the caller to
 * allocate these buffers (in CPU and GPU memories) and provide them to other pucchF3Csi2Ctrl APIs
 *
 * Returns ::CUPHY_STATUS_INVALID_ARGUMENT if \p pDynDescrSizeBytes and/or \p pDynDescrAlignBytes is NULL
 *
 * Returns ::CUPHY_STATUS_SUCCESS if allocation and initialization were
 * successful.
 *
 * \param pDynDescrSizeBytes   - Size in bytes of dynamic descriptor
 * \param pDynDescrAlignBytes  - Alignment in bytes of dynamic descriptor
 *
 * \return
 * ::CUPHY_STATUS_SUCCESS,
 * ::CUPHY_STATUS_INVALID_ARGUMENT
 *
 * \sa ::cuphyStatus_t,::cuphyGetErrorName,::cuphyGetErrorString
 */
cuphyStatus_t CUPHYWINAPI cuphyPucchF3Csi2CtrlGetDescrInfo(size_t* pDynDescrSizeBytes,
                                                           size_t* pDynDescrAlignBytes);

/******************************************************************/ /**
 * \brief Allocate and initialize a cuPHY pucchF3Csi2Ctrl object
 *
 * Allocates a pucchF3Csi2Ctrl object and returns a handle in the
 * address provided by the caller.
 *
 * Returns ::CUPHY_STATUS_INVALID_ARGUMENT if \p pPucchF3Csi2CtrlHndl  is NULL.
 *
 * Returns ::CUPHY_STATUS_ALLOC_FAILED if a pucchF3Csi2Ctrl object cannot be allocated
 *
 * Returns ::CUPHY_STATUS_SUCCESS if allocation and initialization were successful
 *
 * \param pPucchF3Csi2CtrlHndl   - Address to return the new pucchF3Csi2Ctrl instance
 *
 * \return
 * ::CUPHY_STATUS_SUCCESS,
 * ::CUPHY_STATUS_ALLOC_FAILED,
 * ::CUPHY_STATUS_INVALID_ARGUMENT
 *
::cuphyStatus_t,::cuphyGetErrorName,::cuphyGetErrorString */

cuphyStatus_t CUPHYWINAPI cuphyCreatePucchF3Csi2Ctrl(cuphyPucchF3Csi2CtrlHndl_t* pPucchF3Csi2CtrlHndl);

/******************************************************************/ /**
 * \brief Setup cuPHY pucchF3Csi2Ctrl for slot processing
 *
 * Setup cuPHY pucchF3Csi2Ctrl in preparation towards slot execution
 *
 * Returns ::CUPHY_STATUS_SUCCESS if setup is successful.
 *
 * \param pucchF3Csi2CtrlHndl         - Handle for Pucch F3 CSI part 2 component instance
 * \param nCsi2Ucis                   - number of UCIs bearing CSI part2 payload
 * \param pCsi2UciIdxsCpu             - indices of CSI part2 payload bearing UCIs in CPU memory (index to resolve UCI from set of all UCIs being processed by PUCCH format 3)
 * \param pUciPrmsCpu                 - address of UCI parameters in CPU memory
 * \param pUciPrmsGpu                 - address of UCI parameters in GPU memory
 * \param pCellStatPrmsGpu            - cell static parameters specific to PUCCH pipeline
 * \param pPucchF3OutOffsetsCpu       - pointer to any array of structures containing per UCI offsets for locating PUCCH F3 outputs
 * \param pUciPayloadsGpu             - pointer to UCI payloads in GPU
 * \param pNumCsi2BitsGpu             - pointer to array containing number of CSI part2 payload bits
 * \param pCsi2PolarSegPrmsGpu        - pointer to parameters for polar encoded UCI segment
 * \param pCsi2PolarCwPrmsGpu         - pointer to parameters for polar code words
 * \param pCsi2RmCwPrmsGpu            - Reed-muller decoder code word parameters in GPU memory
 * \param pCsi2SpxCwPrmsGpu           - simplex decoder code word parameters in GPU memory
 * \param pCpuDynDesc                 - pointer to dynamic descriptor in CPU memory
 * \param pGpuDynDesc                 - pointer to dynamic descriptor in GPU memory
 * \param enableCpuToGpuDescrAsyncCpy - Flag when set enables async copy of CPU descriptor into GPU
 * \param pLaunchCfg                  - Pointer to launch configurations
 * \param strm                        - CUDA stream for descriptor copy operation
 * \return
 * ::CUPHY_STATUS_SUCCESS,
 * ::CUPHY_STATUS_INVALID_ARGUMENT
 *
 * \sa ::cuphyStatus_t,::cuphyGetErrorName,::cuphyGetErrorString
 */

cuphyStatus_t CUPHYWINAPI cuphySetupPucchF3Csi2Ctrl(cuphyPucchF3Csi2CtrlHndl_t           pucchF3Csi2CtrlHndl,
                                                    uint16_t                             nCsi2Ucis,
                                                    uint16_t*                            pCsi2UciIdxsCpu,
                                                    cuphyPucchUciPrm_t*                  pUciPrmsCpu,
                                                    cuphyPucchUciPrm_t*                  pUciPrmsGpu,
                                                    cuphyPucchCellStatPrm_t*             pCellStatPrmsGpu,
                                                    cuphyPucchF234OutOffsets_t*          pPucchF3OutOffsetsCpu,
                                                    uint8_t*                             pUciPayloadsGpu,
                                                    uint16_t*                            pNumCsi2BitsGpu,
                                                    cuphyPolarUciSegPrm_t*               pCsi2PolarSegPrmsGpu,
                                                    cuphyPolarCwPrm_t*                   pCsi2PolarCwPrmsGpu,
                                                    cuphyRmCwPrm_t*                      pCsi2RmCwPrmsGpu,
                                                    cuphySimplexCwPrm_t*                 pCsi2SpxCwPrmsGpu,
                                                    void*                                pCpuDynDesc,
                                                    void*                                pGpuDynDesc,
                                                    bool                                 enableCpuToGpuDescrAsyncCpy,
                                                    cuphyPucchF3Csi2CtrlLaunchCfg_t*     pLaunchCfg,
                                                    cudaStream_t                         strm);

/******************************************************************/ /**
 * \brief Destroys a cuPHY pucchF3Csi2Ctrl object
 *
 * Destroys a cuPHY pucchF3Csi2Ctrl object that was previously
 * created by ::cuphyCreatePucchF3Csi2Ctrl. The handle provided to this function
 * should not be used for any operations after this function returns.
 *
 * Returns ::CUPHY_STATUS_INVALID_ARGUMENT if \p pucchF3Csi2CtrlHndl is NULL.
 *
 * Returns ::CUPHY_STATUS_SUCCESS if destruction was successful.
 *
 * \param pucchF3Csi2CtrlHndl - handle to previously allocated instance
 *
 * \return
 * ::CUPHY_STATUS_SUCCESS,
 * ::CUPHY_STATUS_INVALID_ARGUMENT
 *
 * \sa ::cuphyStatus_t,::cuphyGetErrorName,::cuphyGetErrorString
 */
cuphyStatus_t CUPHYWINAPI cuphyDestroyPucchF3Csi2Ctrl(cuphyPucchF3Csi2CtrlHndl_t pucchF3Csi2CtrlHndl);

/** @} */ /* END CUPHY_PUCCH_F3_CSI2_CTRL */

////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////

/**
 * \defgroup CUPHY_PUCCH_F3_SEG_LLRs : segment descrambled LLR array into two LLR sub-arrays for HARQ/SR/CSIp1 and CSIp2 for PUCCH format 3.
 *
 * This section describes application programming interface for the PUCCH format 3 LLR segmentation kernel
 *
 * @{
 */

struct cuphyPucchF3SegLLRs;
/**
 * cuPHY pucchF3SegLLRs handle
 */
typedef struct cuphyPucchF3SegLLRs* cuphyPucchF3SegLLRsHndl_t;

/**
 * cuPHY PUCCH format 3 LLR segmentation, launch configuration
 */
typedef struct
{
    CUDA_KERNEL_NODE_PARAMS kernelNodeParamsDriver;
    void*                   kernelArgs[2];
} cuphyPucchF3SegLLRsLaunchCfg_t;

/******************************************************************/ /**
 * \brief Helper to compute pucchF3SegLLRs descriptor buffer sizes and alignments
 *
 * Computes pucchF3SegLLRs descriptor buffer sizes and alignments. To be used by the caller to
 * allocate these buffers (in CPU and GPU memories) and provide them to other pucchF3SegLLRs APIs
 *
 * Returns ::CUPHY_STATUS_INVALID_ARGUMENT if \p pDynDescrSizeBytes and/or \p pDynDescrAlignBytes is NULL
 *
 * Returns ::CUPHY_STATUS_SUCCESS if allocation and initialization were
 * successful.
 *
 * \param pDynDescrSizeBytes   - Size in bytes of dynamic descriptor
 * \param pDynDescrAlignBytes  - Alignment in bytes of dynamic descriptor
 *
 * \return
 * ::CUPHY_STATUS_SUCCESS,
 * ::CUPHY_STATUS_INVALID_ARGUMENT
 *
 * \sa ::cuphyStatus_t,::cuphyGetErrorName,::cuphyGetErrorString
 */
cuphyStatus_t CUPHYWINAPI cuphyPucchF3SegLLRsGetDescrInfo(size_t* pDynDescrSizeBytes,
                                                          size_t* pDynDescrAlignBytes);

/******************************************************************/ /**
 * \brief Allocate and initialize a cuPHY pucchF3SegLLRs object
 *
 * Allocates a pucchF3SegLLRs object and returns a handle in the
 * address provided by the caller.
 *
 * Returns ::CUPHY_STATUS_INVALID_ARGUMENT if \p pPucchF3SegLLRsHndl  is NULL.
 *
 * Returns ::CUPHY_STATUS_ALLOC_FAILED if a pucchF3SegLLRs object cannot be allocated
 *
 * Returns ::CUPHY_STATUS_SUCCESS if allocation and initialization were successful
 *
 * \param pPucchF3SegLLRsHndl   - Address to return the new pucchF3SegLLRs instance
 *
 * \return
 * ::CUPHY_STATUS_SUCCESS,
 * ::CUPHY_STATUS_ALLOC_FAILED,
 * ::CUPHY_STATUS_INVALID_ARGUMENT
 *
::cuphyStatus_t,::cuphyGetErrorName,::cuphyGetErrorString */

cuphyStatus_t CUPHYWINAPI cuphyCreatePucchF3SegLLRs(cuphyPucchF3SegLLRsHndl_t* pPucchF3SegLLRsHndl);

/******************************************************************/ /**
 * \brief Setup cuPHY pucchF3SegLLRs for slot processing
 *
 * Setup cuPHY pucchF3SegLLRs in preparation towards slot execution
 *
 * Returns ::CUPHY_STATUS_SUCCESS if setup is successful.
 *
 * \param pucchF3SegLLRsHndl          - Handle for Pucch F3 LLR array segmentation component instance
 * \param nF3Ucis                     - number of PF3 UCIs
 * \param pF3UciPrms                  - address of UCI parameters in CPU memory
 * \param pDescramLLRaddrs            - address of descrambled LLR arrays
 * \param pCpuDynDesc                 - pointer to dynamic descriptor in CPU memory
 * \param pGpuDynDesc                 - pointer to dynamic descriptor in GPU memory
 * \param enableCpuToGpuDescrAsyncCpy - Flag when set enables async copy of CPU descriptor into GPU
 * \param pLaunchCfg                  - Pointer to launch configurations
 * \param strm                        - CUDA stream for descriptor copy operation
 * \return
 * ::CUPHY_STATUS_SUCCESS,
 * ::CUPHY_STATUS_INVALID_ARGUMENT
 *
 * \sa ::cuphyStatus_t,::cuphyGetErrorName,::cuphyGetErrorString
 */

cuphyStatus_t CUPHYWINAPI cuphySetupPucchF3SegLLRs(cuphyPucchF3SegLLRsHndl_t            pucchF3SegLLRsHndl,
                                                   uint16_t                             nF3Ucis,
                                                   cuphyPucchUciPrm_t*                  pF3UciPrms,
                                                   __half**                             pDescramLLRaddrs,
                                                   void*                                pCpuDynDesc,
                                                   void*                                pGpuDynDesc,
                                                   bool                                 enableCpuToGpuDescrAsyncCpy,
                                                   cuphyPucchF3SegLLRsLaunchCfg_t*      pLaunchCfg,
                                                   cudaStream_t                         strm);

/******************************************************************/ /**
 * \brief Destroys a cuPHY pucchF3SegLLRs object
 *
 * Destroys a cuPHY pucchF3SegLLRs object that was previously
 * created by ::cuphyCreatePucchF3SegLLRs. The handle provided to this function
 * should not be used for any operations after this function returns.
 *
 * Returns ::CUPHY_STATUS_INVALID_ARGUMENT if \p pucchF3SegLLRsHndl is NULL.
 *
 * Returns ::CUPHY_STATUS_SUCCESS if destruction was successful.
 *
 * \param pucchF3SegLLRsHndl - handle to previously allocated instance
 *
 * \return
 * ::CUPHY_STATUS_SUCCESS,
 * ::CUPHY_STATUS_INVALID_ARGUMENT
 *
 * \sa ::cuphyStatus_t,::cuphyGetErrorName,::cuphyGetErrorString
 */
cuphyStatus_t CUPHYWINAPI cuphyDestroyPucchF3SegLLRs(cuphyPucchF3SegLLRsHndl_t pucchF3SegLLRsHndl);

/** @} */ /* END CUPHY_PUCCH_F3_SEG_LLRs */

////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////

/**
 * \defgroup CUPHY_PUCCH_F234_UCI_SEG : segment UCI into HARQ, SR and CSI part 1 for PUCCH format 2, 3, 4.
 *
 * This section describes application programming interface for the PUCCH format 2, 3, 4 UCI segmentation kernel
 *
 * @{
 */

struct cuphyPucchF234UciSeg;
/**
 * cuPHY pucchF234UciSeg handle
 */
typedef struct cuphyPucchF234UciSeg* cuphyPucchF234UciSegHndl_t;

/**
 * cuPHY PUCCH format 2, 3, 4 UCI segmentation, launch configuration
 */
typedef struct
{
    CUDA_KERNEL_NODE_PARAMS kernelNodeParamsDriver;
    void*                   kernelArgs[2];
} cuphyPucchF234UciSegLaunchCfg_t;

/******************************************************************/ /**
 * \brief Helper to compute pucchF234UciSeg descriptor buffer sizes and alignments
 *
 * Computes pucchF234UciSeg descriptor buffer sizes and alignments. To be used by the caller to
 * allocate these buffers (in CPU and GPU memories) and provide them to other pucchF234UciSeg APIs
 *
 * Returns ::CUPHY_STATUS_INVALID_ARGUMENT if \p pDynDescrSizeBytes and/or \p pDynDescrAlignBytes is NULL
 *
 * Returns ::CUPHY_STATUS_SUCCESS if allocation and initialization were
 * successful.
 *
 * \param pDynDescrSizeBytes   - Size in bytes of dynamic descriptor
 * \param pDynDescrAlignBytes  - Alignment in bytes of dynamic descriptor
 *
 * \return
 * ::CUPHY_STATUS_SUCCESS,
 * ::CUPHY_STATUS_INVALID_ARGUMENT
 *
 * \sa ::cuphyStatus_t,::cuphyGetErrorName,::cuphyGetErrorString
 */
cuphyStatus_t CUPHYWINAPI cuphyPucchF234UciSegGetDescrInfo(size_t* pDynDescrSizeBytes,
                                                           size_t* pDynDescrAlignBytes);

/******************************************************************/ /**
 * \brief Allocate and initialize a cuPHY pucchF234UciSeg object
 *
 * Allocates a pucchF234UciSeg object and returns a handle in the
 * address provided by the caller.
 *
 * Returns ::CUPHY_STATUS_INVALID_ARGUMENT if \p pPucchF234UciSegHndl  is NULL.
 *
 * Returns ::CUPHY_STATUS_ALLOC_FAILED if a pucchF234UciSeg object cannot be allocated
 *
 * Returns ::CUPHY_STATUS_SUCCESS if allocation and initialization were successful
 *
 * \param pPucchF234UciSegHndl   - Address to return the new pucchF234UciSeg instance
 *
 * \return
 * ::CUPHY_STATUS_SUCCESS,
 * ::CUPHY_STATUS_ALLOC_FAILED,
 * ::CUPHY_STATUS_INVALID_ARGUMENT
 *
::cuphyStatus_t,::cuphyGetErrorName,::cuphyGetErrorString */

cuphyStatus_t CUPHYWINAPI cuphyCreatePucchF234UciSeg(cuphyPucchF234UciSegHndl_t* pPucchF234UciSegHndl);

/******************************************************************/ /**
 * \brief Setup cuPHY pucchF234UciSeg for slot processing
 *
 * Setup cuPHY pucchF234UciSeg in preparation towards slot execution
 *
 * Returns ::CUPHY_STATUS_SUCCESS if setup is successful.
 *
 * \param pucchF234UciSegHndl         - Handle for Pucch F2/F3/F4 UCI segmentation component instance
 * \param nF2Ucis                     - number of PF2 UCIs
 * \param nF3Ucis                     - number of PF3 UCIs
 * \param pF2UciPrms                  - address of PF2 UCI parameters in CPU memory
 * \param pF3UciPrms                  - address of PF3 UCI parameters in CPU memory
 * \param pF2OutOffsetsCpu            - address of PF2 output offset parameters in CPU memory
 * \param pF3OutOffsetsCpu            - address of PF3 output offset parameters in CPU memory
 * \param uciPayloadsGpu              - address of decoded UCI payloads array
 * \param pCpuDynDesc                 - pointer to dynamic descriptor in CPU memory
 * \param pGpuDynDesc                 - pointer to dynamic descriptor in GPU memory
 * \param enableCpuToGpuDescrAsyncCpy - Flag when set enables async copy of CPU descriptor into GPU
 * \param pLaunchCfg                  - Pointer to launch configurations
 * \param strm                        - CUDA stream for descriptor copy operation
 * \return
 * ::CUPHY_STATUS_SUCCESS,
 * ::CUPHY_STATUS_INVALID_ARGUMENT
 *
 * \sa ::cuphyStatus_t,::cuphyGetErrorName,::cuphyGetErrorString
 */

cuphyStatus_t CUPHYWINAPI cuphySetupPucchF234UciSeg(cuphyPucchF234UciSegHndl_t       pucchF234UciSegHndl,
                                                    uint16_t                         nF2Ucis,
                                                    uint16_t                         nF3Ucis,
                                                    cuphyPucchUciPrm_t*              pF2UciPrms,
                                                    cuphyPucchUciPrm_t*              pF3UciPrms,
                                                    cuphyPucchF234OutOffsets_t*&     pF2OutOffsetsCpu,
                                                    cuphyPucchF234OutOffsets_t*&     pF3OutOffsetsCpu,
                                                    uint8_t*                         uciPayloadsGpu,
                                                    void*                            pCpuDynDesc,
                                                    void*                            pGpuDynDesc,
                                                    bool                             enableCpuToGpuDescrAsyncCpy,
                                                    cuphyPucchF234UciSegLaunchCfg_t* pLaunchCfg,
                                                    cudaStream_t                     strm);

/******************************************************************/ /**
 * \brief Destroys a cuPHY pucchF234UciSeg object
 *
 * Destroys a cuPHY pucchF234UciSeg object that was previously
 * created by ::cuphyCreatePucchF234UciSeg. The handle provided to this function
 * should not be used for any operations after this function returns.
 *
 * Returns ::CUPHY_STATUS_INVALID_ARGUMENT if \p pPucchF234UciSegHndl is NULL.
 *
 * Returns ::CUPHY_STATUS_SUCCESS if destruction was successful.
 *
 * \param pPucchF234UciSegHndl - handle to previously allocated instance
 *
 * \return
 * ::CUPHY_STATUS_SUCCESS,
 * ::CUPHY_STATUS_INVALID_ARGUMENT
 *
 * \sa ::cuphyStatus_t,::cuphyGetErrorName,::cuphyGetErrorString
 */
cuphyStatus_t CUPHYWINAPI cuphyDestroyPucchF234UciSeg(cuphyPucchF234UciSegHndl_t pPucchF234UciSegHndl);

/** @} */ /* END CUPHY_PUCCH_F234_UCI_SEG */

////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////

/**
 * \defgroup CUPHY_COMP_POL_TREE_CW_TYPES Codeword types for polar tree
 *
 * This section describes application programming interface for computing codeword types (rate-0, rate-1, neither) for codewords within a polar codeword tree.
 *
 * @{
 */

struct cuphyCompCwTreeTypes;
/**
 * cuPHY compCwTreeTypes handle
 */
typedef struct cuphyCompCwTreeTypes* cuphyCompCwTreeTypesHndl_t;

/**
 * cuPHY compute polar codeword tree types, launch configuration
 */
typedef struct
{
    CUDA_KERNEL_NODE_PARAMS kernelNodeParamsDriver;
    void*                   kernelArgs[2];
} cuphyCompCwTreeTypes_t;

typedef struct
{
    CUDA_KERNEL_NODE_PARAMS kernelNodeParamsDriver;
    void*                   kernelArgs[2];
} cuphyCompCwTreeTypesLaunchCfg_t;

/******************************************************************/ /**
 * \brief Helper to compute compCwTreeTypes descriptor buffer sizes and alignments
 *
 * Computes compCwTreeTypes descriptor buffer sizes and alignments. To be used by the caller to
 * allocate these buffers (in CPU and GPU memories) and provide them to other compCwTreeTypes APIs
 *
 * Returns ::CUPHY_STATUS_INVALID_ARGUMENT if \p pDynDescrSizeBytes and/or \p pDynDescrAlignBytes is NULL
 *
 * Returns ::CUPHY_STATUS_SUCCESS if allocation and initialization were
 * successful.
 *
 * \param pDynDescrSizeBytes   - Size in bytes of dynamic descriptor
 * \param pDynDescrAlignBytes  - Alignment in bytes of dynamic descriptor
 *
 * \return
 * ::CUPHY_STATUS_SUCCESS,
 * ::CUPHY_STATUS_INVALID_ARGUMENT
 *
 * \sa ::cuphyStatus_t,::cuphyGetErrorName,::cuphyGetErrorString,::cuphyCreatePucchF1Rx,::cuphyDestroyPucchF1Rx
 */
cuphyStatus_t CUPHYWINAPI cuphyCompCwTreeTypesGetDescrInfo(size_t* pDynDescrSizeBytes,
                                                           size_t* pDynDescrAlignBytes);

/******************************************************************/ /**
 * \brief Allocate and initialize a cuPHY compCwTreeTypes object
 *
 * Allocates a compCwTreeTypes object and returns a handle in the
 * address provided by the caller.
 *
 * Returns ::CUPHY_STATUS_INVALID_ARGUMENT if \p pPucchF0RxHndl  is NULL.
 *
 * Returns ::CUPHY_STATUS_ALLOC_FAILED if a compCwTreeTypes object cannot be allocated
 *
 * Returns ::CUPHY_STATUS_SUCCESS if allocation and initialization were successful
 *
 * \param pCompCwTreeTypes     - Address to return the new compCwTreeTypes instance
 *
 * \return
 * ::CUPHY_STATUS_SUCCESS,
 * ::CUPHY_STATUS_ALLOC_FAILED,
 * ::CUPHY_STATUS_INVALID_ARGUMENT
 *
::cuphyStatus_t,::cuphyGetErrorName,::cuphyGetErrorString,::cuphyPucchF0RxGetDescrInfo,::cuphySetupPucchF0Rx */

cuphyStatus_t CUPHYWINAPI cuphyCreateCompCwTreeTypes(cuphyCompCwTreeTypesHndl_t* pCompCwTreeTypes);

/******************************************************************/ /**
 * \brief Setup cuPHY compCwTreeTypes for slot processing
 *
 * Setup cuPHY compCwTreeTypes in preparation towards slot execution
 *
 * Returns ::CUPHY_STATUS_SUCCESS if setup is successful.
 *
 * Returns ::CUPHY_STATUS_INVALID_ARGUMENT if \p compCwTreeTypesHndl and/or \p pPolUciSegPrmsCpu and/or \p pPolUciSegPrmsGpu and/or
 * \p pCpuDynDescCompTree  and/or \p pGpuDynDescCompTree  and/or \p pLaunchCfg is NULL.
 *
 * \param compCwTreeTypesHndl           - Handle to previously created compCwTreeTypes instance
 * \param nPolUciSegs                   - number of polar UCI segments
 * \param pPolUciSegPrmsCpu             - starting address of polar UCI segment parameters (CPU)
 * \param pPolUciSegPrmsGpu             - starting address of polar UCI segment parameters (GPU)
 * \param pCwTreeTypesAddrs             - pointer to cwTreeTypes addresses
 * \param pCpuDynDescCompTree           - pointer to compTree descriptor in cpu
 * \param pGpuDynDescCompTree           - pointer to comptTree descriptor in gpu
 * \param pCpuDynDescCompTreeAddrs      - pointer to compTreeAddrs descriptor in cpu
 * \param enableCpuToGpuDescrAsyncCpy   - option to copy cpu descriptors from cpu to gpu
 * \param pLaunchCfg                    - pointer to rate matching launch configuration
 * \param strm                          - stream to perform copy
 *
 * \return
 * ::CUPHY_STATUS_SUCCESS,
 * ::CUPHY_STATUS_INVALID_ARGUMENT
 *
 * \sa ::cuphyStatus_t,::cuphyGetErrorName,::cuphyGetErrorString,::cuphyPucchF0RxGetDescrInfo,::cuphyDestroyPucchF0Rx
 */
cuphyStatus_t CUPHYWINAPI
cuphySetupCompCwTreeTypes(cuphyCompCwTreeTypesHndl_t       compCwTreeTypesHndl,
                          uint16_t                         nPolUciSegs,
                          const cuphyPolarUciSegPrm_t*     pPolUciSegPrmsCpu,
                          const cuphyPolarUciSegPrm_t*     pPolUciSegPrmsGpu,
                          uint8_t**                        pCwTreeTypesAddrs,
                          void*                            pCpuDynDescCompTree,
                          void*                            pGpuDynDescCompTree,
                          void*                            pCpuDynDescCompTreeAddrs,
                          uint8_t                          enableCpuToGpuDescrAsyncCpy,
                          cuphyCompCwTreeTypesLaunchCfg_t* pLaunchCfg,
                          cudaStream_t                     strm);

/******************************************************************/ /**
 * \brief Destroys a cuPHY compCwTreeTypes object
 *
 * Destroys a cuPHY compCwTreeTypes object that was previously
 * created by ::cuphyCreateCompCwTreeTypes. The handle provided to this function
 * should not be used for any operations after this function returns.
 *
 * Returns ::CUPHY_STATUS_INVALID_ARGUMENT if \p compCwTreeTypesHndl is NULL.
 *
 * Returns ::CUPHY_STATUS_SUCCESS if destruction was successful.
 *
 * \param compCwTreeTypesHndl - handle to previously allocated compCwTreeTypes instance
 *
 * \return
 * ::CUPHY_STATUS_SUCCESS,
 * ::CUPHY_STATUS_INVALID_ARGUMENT
 *
 * \sa ::cuphyStatus_t,::cuphyGetErrorName,::cuphyGetErrorString,::cuphyPucchF0RxGetDescrInfo,::cuphyCreatePucchF0Rx,::cuphySetupPucchF0Rx
 */
cuphyStatus_t CUPHYWINAPI cuphyDestroyCompCwTreeTypes(cuphyCompCwTreeTypesHndl_t compCwTreeTypesHndl);

/** @} */ /* END CUPHY_COMP_POL_TREE_CW_TYPES */

////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////

/**
 * \defgroup CUPHY_POL_SEG_DERM_DITL Polar segmentation, de-rate matching and de-interleaving
 *
 * This section describes application programming interface for polar codeword LLR segmentation + deInterleaving + deRateMatching
 *
 * @{
 */

struct cuphyPolSegDeItlDeRm;
/**
 * cuPHY polSegDeRmDeItl handle
 */
typedef struct cuphyPolSegDeRmDeItl* cuphyPolSegDeRmDeItlHndl_t;

/**
 * cuPHY polar codeword LLR segmentation + deInterleaving + deRateMatching, launch configuration
 */
typedef struct
{
    CUDA_KERNEL_NODE_PARAMS kernelNodeParamsDriver;
    void*                   kernelArgs[2];
} cuphyPolSegDeRmDeItlLaunchCfg_t;

/******************************************************************/ /**
 * \brief Helper to compute PolSegDeRmDeItl descriptor buffer sizes and alignments
 *
 * Computes PolSegDeRmDeItl descriptor buffer sizes and alignments. To be used by the caller to
 * allocate these buffers (in CPU and GPU memories) and provide them to other PolSegDeRmDeItl APIs
 *
 * Returns ::CUPHY_STATUS_INVALID_ARGUMENT if \p pDynDescrSizeBytes and/or \p pDynDescrAlignBytes is NULL
 *
 * Returns ::CUPHY_STATUS_SUCCESS if allocation and initialization were
 * successful.
 *
 * \param pDynDescrSizeBytes   - Size in bytes of dynamic descriptor
 * \param pDynDescrAlignBytes  - Alignment in bytes of dynamic descriptor
 *
 * \return
 * ::CUPHY_STATUS_SUCCESS,
 * ::CUPHY_STATUS_INVALID_ARGUMENT
 *
 * \sa ::cuphyStatus_t,::cuphyGetErrorName,::cuphyGetErrorString,::cuphyCreatePolSegDeRmDeItl,::cuphyDestroyPolSegDeRmDeItl
 */
cuphyStatus_t CUPHYWINAPI cuphyPolSegDeRmDeItlGetDescrInfo(size_t* pDynDescrSizeBytes,
                                                           size_t* pDynDescrAlignBytes);

/******************************************************************/ /**
 * \brief Allocate and initialize a cuPHY polSegDeRmDeItl object
 *
 * Allocates a polSegDeRmDeItl object and returns a handle in the
 * address provided by the caller.
 *
 * Returns ::CUPHY_STATUS_INVALID_ARGUMENT if \p pPolSegDeRmDeItl  is NULL.
 *
 * Returns ::CUPHY_STATUS_ALLOC_FAILED if a compCwTreeTypes object cannot be allocated
 *
 * Returns ::CUPHY_STATUS_SUCCESS if allocation and initialization were successful
 *
 * \param pPolSegDeRmDeItlHndl   - Address to return the new polSegDeRmDeItl instance
 *
 * \return
 * ::CUPHY_STATUS_SUCCESS,
 * ::CUPHY_STATUS_ALLOC_FAILED,
 * ::CUPHY_STATUS_INVALID_ARGUMENT
 *
::cuphyStatus_t,::cuphyGetErrorName,::cuphyGetErrorString,::cuphyPolSegDeRmDeItlGetDescrInfo,::cuphySetupPolSegDeRmDeItl */

cuphyStatus_t CUPHYWINAPI cuphyCreatePolSegDeRmDeItl(cuphyPolSegDeRmDeItlHndl_t* pPolSegDeRmDeItlHndl);

/******************************************************************/ /**
 * \brief Setup cuPHY polSegDeRmDeItl for slot processing
 *
 * Setup cuPHY polSegDeRmDeItl in preparation towards slot execution
 *
 * Returns ::CUPHY_STATUS_SUCCESS if setup is successful.
 *
 * Returns ::CUPHY_STATUS_INVALID_ARGUMENT if \p polSegDeRmDeItlHndl and/or \p pPolUciSegPrmsCpu and/or \p pPolUciSegPrmsGpu and/or
 * \p pUciSegLLRsAddrs  and/or \p pCwLLRsAddrs  and/or \p pCpuDynDesc and/or \p pGpuDynDesc is NULL.
 *
 * \param polSegDeRmDeItlHndl           - Handle to previously created polSegDeRmDeItl instance
 * \param nPolUciSegs                   - number of polar UCI segments
 * \param nPolCws                       - number of polar codewords
 * \param pPolUciSegPrmsCpu             - starting address of polar UCI segment parameters (CPU)
 * \param pPolUciSegPrmsGpu             - starting address of polar UCI segment parameters (GPU)
 * \param pPolCwPrmsCpu                 - starting address of polar codeword parameters (CPU)
 * \param pPolCwPrmsGpu                 - starting address of polar codeword parameters (GPU)
 * \param pUciSegLLRsAddrs              - pointer to uci segment LLR addresses
 * \param pCwLLRsAddrs                  - pointer to cw LLR addresses
 * \param pCpuDynDescDrDi               - pointer to polSegDeRmDeItlDynDescr descriptor in cpu
 * \param pGpuDynDescDrDi               - pointer to polSegDeRmDeItlDynDescr descriptor in gpu
 * \param pCpuDynDescDrDiCwAddrs        - pointer to cw LLR addresses in polSegDeRmDeItlDynDescr descriptor
 * \param pCpuDynDescDrDiUciAddrs       - pointer to UCI Seg LLR addresses in polSegDeRmDeItlDynDescr descriptor
 * \param enableCpuToGpuDescrAsyncCpy   - option to copy cpu descriptors from cpu to gpu
 * \param pLaunchCfg                    - pointer to rate matching launch configuration
 * \param strm                          - stream to perform copy
 *
 * \return
 * ::CUPHY_STATUS_SUCCESS,
 * ::CUPHY_STATUS_INVALID_ARGUMENT
 *
 * \sa ::cuphyStatus_t,::cuphyGetErrorName,::cuphyGetErrorString,::cuphyPolSegDeRmDeItlGetDescrInfo,::cuphyDestroyPolSegDeRmDeItl
 */
cuphyStatus_t CUPHYWINAPI
cuphySetupPolSegDeRmDeItl(cuphyPolSegDeRmDeItlHndl_t       polSegDeRmDeItlHndl,
                          uint16_t                         nPolUciSegs,
                          uint16_t                         nPolCws,
                          const cuphyPolarUciSegPrm_t*     pPolUciSegPrmsCpu,
                          const cuphyPolarUciSegPrm_t*     pPolUciSegPrmsGpu,
                          const cuphyPolarCwPrm_t*         pPolCwPrmsCpu,
                          const cuphyPolarCwPrm_t*         pPolCwPrmsGpu,
                          __half**                         pUciSegLLRsAddrs,
                          __half**                         pCwLLRsAddrs,
                          void*                            pCpuDynDescDrDi,
                          void*                            pGpuDynDescDrDi,
                          void*                            pCpuDynDescDrDiCwAddrs,
                          void*                            pCpuDynDescDrDiUciAddrs,
                          uint8_t                          enableCpuToGpuDescrAsyncCpy,
                          cuphyPolSegDeRmDeItlLaunchCfg_t* pLaunchCfg,
                          cudaStream_t                     strm);

/******************************************************************/ /**
 * \brief Destroys a cuPHY polSegDeRmDeItl object
 *
 * Destroys a cuPHY polSegDeRmDeItl object that was previously
 * created by ::cuphyCreatePolSegDeRmDeItl. The handle provided to this function
 * should not be used for any operations after this function returns.
 *
 * Returns ::CUPHY_STATUS_INVALID_ARGUMENT if \p polSegDeRmDeItlHndl is NULL.
 *
 * Returns ::CUPHY_STATUS_SUCCESS if destruction was successful.
 *
 * \param polSegDeRmDeItlHndl - handle to previously allocated compCwTreeTypes instance
 *
 * \return
 * ::CUPHY_STATUS_SUCCESS,
 * ::CUPHY_STATUS_INVALID_ARGUMENT
 *
 * \sa ::cuphyStatus_t,::cuphyGetErrorName,::cuphyGetErrorString,::cuphyPucchF0RxGetDescrInfo,::cuphyCreatePucchF0Rx,::cuphySetupPucchF0Rx
 */
cuphyStatus_t CUPHYWINAPI cuphyDestroyPolSegDeRmDeItl(cuphyPolSegDeRmDeItlHndl_t polSegDeRmDeItlHndl);

/** @} */ /* END CUPHY_POL_SEG_DERM_DITL */

////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////

/**
 * \defgroup CUPHY_UCI_ON_PUSCH_SEG_LLRS_1 uci on pusch: LLRs segmentation part 1. De-scrambles and segments SCH, HARQ, and CSI-PART1 LLRs
 *
 * This section describes application programming interface for part 1 UCI on PUSCH deSegmentation
 *
 * @{
 */

struct cuphyUciOnPuschSegLLRs1;
/**
 * cuPHY uciOnPuschSegLLRs1 handle
 */
typedef struct cuphyUciOnPuschSegLLRs1* cuphyUciOnPuschSegLLRs1Hndl_t;

/**
 * cuPHY polar uci on pusch LLR segmentation part 1, launch configuration
 */
typedef struct
{
    CUDA_KERNEL_NODE_PARAMS kernelNodeParamsDriver;
    void*                   kernelArgs[2];
} cuphyUciOnPuschSegLLRs1LaunchCfg_t;

/******************************************************************/ /**
 * \brief Helper to compute uciOnPuschSegLLRs1 descriptor buffer sizes and alignments
 *
 * Computes uciOnPuschSegLLRs1 descriptor buffer sizes and alignments. To be used by the caller to
 * allocate these buffers (in CPU and GPU memories) and provide them to other uciOnPuschSegLLRs1 APIs
 *
 * Returns ::CUPHY_STATUS_INVALID_ARGUMENT if \p pDynDescrSizeBytes and/or \p pDynDescrAlignBytes is NULL
 *
 * Returns ::CUPHY_STATUS_SUCCESS if allocation and initialization were
 * successful.
 *
 * \param pDynDescrSizeBytes   - Size in bytes of dynamic descriptor
 * \param pDynDescrAlignBytes  - Alignment in bytes of dynamic descriptor
 *
 * \return
 * ::CUPHY_STATUS_SUCCESS,
 * ::CUPHY_STATUS_INVALID_ARGUMENT
 *
 * \sa ::cuphyStatus_t,::cuphyGetErrorName,::cuphyGetErrorString,::cuphyCreateUciOnPuschSegLLRs1,::cuphyDestroyUciOnPuschSegLLRs1
 */
cuphyStatus_t CUPHYWINAPI cuphyUciOnPuschSegLLRs1GetDescrInfo(size_t* pDynDescrSizeBytes,
                                                              size_t* pDynDescrAlignBytes);

/******************************************************************/ /**
 * \brief Allocate and initialize a cuPHY uciOnPuschSegLLRs1 object
 *
 * Allocates a uciOnPuschSegLLRs1 object and returns a handle in the
 * address provided by the caller.
 *
 * Returns ::CUPHY_STATUS_INVALID_ARGUMENT if \p pUciOnPuschSegLLRs1Hndl is NULL.
 *
 * Returns ::CUPHY_STATUS_ALLOC_FAILED if a uciOnPuschSegLLRs1 object cannot be allocated
 *
 * Returns ::CUPHY_STATUS_SUCCESS if allocation and initialization were successful
 *
 * \param pUciOnPuschSegLLRs1Hndl   - Address to return the new uciOnPuschSegLLRs1 instance
 *
 * \return
 * ::CUPHY_STATUS_SUCCESS,
 * ::CUPHY_STATUS_ALLOC_FAILED,
 * ::CUPHY_STATUS_INVALID_ARGUMENT
 *
::cuphyStatus_t,::cuphyGetErrorName,::cuphyGetErrorString,::cuphyUciOnPuschSegLLRs1GetDescrInfo,::cuphySetupUciOnPuschSegLLRs1 */

cuphyStatus_t CUPHYWINAPI cuphyCreateUciOnPuschSegLLRs1(cuphyUciOnPuschSegLLRs1Hndl_t* pUciOnPuschSegLLRs1Hndl);

/******************************************************************/ /**
 * \brief Setup cuPHY uciOnPuschSegLLRs1 for slot processing
 *
 * Setup cuPHY uciOnPuschSegLLRs1 in preparation towards slot execution
 *
 * Returns ::CUPHY_STATUS_SUCCESS if setup is successful.
 *
 * \param uciOnPuschSegLLRs1Hndl      - handle of uciOnPuschSegLLRs1 instance
 * \param nUciUes                     - number of UEs bearing Uplink Control Information (UCI)
 * \param pUciUserIdxs                - indices of UCI bearing UEs (index to resolve UE from set of all UEs being processed by PUSCH)
 * \param pTbPrmsCpu                  - address of Transport block parameters in CPU memory
 * \param pTbPrmsGpu                  - address of Transport block parameters in GPU memory
 * \param nUeGrps                     - number of UE groups to be processed
 * \param pTensorPrmsEqOutLLRs        - tensor parameters for equalizer output LLRs
 * \param pNumPrbs                    - number of allocated PRBs
 * \param startSym                    - first symbol of PUSCH
 * \param nPuschSym                   - total number of PUSCH symbols
 * \param nPuschDataSym               - number of PUSCH data symbols
 * \param pDataSymIdxs                - symbol indices of PUSCH data symbols
 * \param nPuschDmrsSym               - number of PUSCH DMRS symbols
 * \param pDmrsSymIdxs                - symbol indices of PUSCH DMRS symbols
 * \param pCpuDynDesc                 - Pointer to dynamic descriptor in CPU memory
 * \param pGpuDynDesc                 - Pointer to dynamic descriptor in GPU memory
 * \param enableCpuToGpuDescrAsyncCpy - Flag when set enables async copy of CPU descriptor into GPU
 * \param pLaunchCfg                  - Pointer to channel estimation launch configurations
 * \param strm                        - CUDA stream for descriptor copy operation
 * \return
 * ::CUPHY_STATUS_SUCCESS,
 * ::CUPHY_STATUS_INVALID_ARGUMENT
 *
 * \sa ::cuphyStatus_t,::cuphyGetErrorName,::cuphyGetErrorString,::cuphyUciOnPuschSegLLRs1GetDescrInfo,::cuphyDestroyUciOnPuschSegLLRs1
 */

cuphyStatus_t CUPHYWINAPI cuphySetupUciOnPuschSegLLRs1(cuphyUciOnPuschSegLLRs1Hndl_t       uciOnPuschSegLLRs1Hndl,
                                                       uint16_t                            nUciUes,
                                                       uint16_t*                           pUciUserIdxs,
                                                       PerTbParams*                        pTbPrmsCpu,
                                                       PerTbParams*                        pTbPrmsGpu,
                                                       uint16_t                            nUeGrps,
                                                       cuphyTensorPrm_t*                   pTensorPrmsEqOutLLRs,
                                                       uint16_t*                           pNumPrbs,
                                                       uint8_t                             startSym,
                                                       uint8_t                             nPuschSym,
                                                       uint8_t                             nPuschDataSym,
                                                       uint8_t*                            pDataSymIdxs,
                                                       uint8_t                             nPuschDmrsSym,
                                                       uint8_t*                            pDmrsSymIdxs,
                                                       void*                               pCpuDynDesc,
                                                       void*                               pGpuDynDesc,
                                                       uint8_t                             enableCpuToGpuDescrAsyncCpy,
                                                       cuphyUciOnPuschSegLLRs1LaunchCfg_t* pLaunchCfg,
                                                       cudaStream_t                        strm);

/******************************************************************/ /**
 * \brief Destroys a cuPHY uciOnPuschSegLLRs1 object
 *
 * Destroys a cuPHY uciOnPuschSegLLRs1 object that was previously
 * created by ::cuphyCreateUciOnPuschSegLLRs1. The handle provided to this function
 * should not be used for any operations after this function returns.
 *
 * Returns ::CUPHY_STATUS_INVALID_ARGUMENT if \p uciOnPuschSegLLRs1Hndl is NULL.
 *
 * Returns ::CUPHY_STATUS_SUCCESS if destruction was successful.
 *
 * \param uciOnPuschSegLLRs1Hndl - handle to previously allocated compCwTreeTypes instance
 *
 * \return
 * ::CUPHY_STATUS_SUCCESS,q
 * ::CUPHY_STATUS_INVALID_ARGUMENT
 *
 * \sa ::cuphyStatus_t,::cuphyGetErrorName,::cuphyGetErrorString,::cuphyUciOnPuschSegLLRs1GetDescrInfo,::cuphyCreateUciOnPuschSegLLRs1,::cuphySetupUciOnPuschSegLLRs1
 */
cuphyStatus_t CUPHYWINAPI cuphyDestroyUciOnPuschSegLLRs1(cuphyUciOnPuschSegLLRs1Hndl_t uciOnPuschSegLLRs1Hndl);

/** @} */ /* END CUPHY_UCI_ON_PUSCH_SEG_LLRS_1 */

////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////

typedef enum _cuphyUciToSeg
{
    SEG_ONLY_EARLY_UCI    = 0,
    SEG_ALL_UCI           = 1,
} cuphyUciToSeg_t;



struct cuphyUciOnPuschSegLLRs0;
/**
 * cuPHY uciOnPuschSegLLRs0 handle
 */
typedef struct cuphyUciOnPuschSegLLRs0* cuphyUciOnPuschSegLLRs0Hndl_t;

/**
 * cuPHY polar uci on pusch LLR segmentation part 0, launch configuration
 */
typedef struct
{
    CUDA_KERNEL_NODE_PARAMS kernelNodeParamsDriver;
    void*                   kernelArgs[2];
} cuphyUciOnPuschSegLLRs0LaunchCfg_t;



cuphyStatus_t CUPHYWINAPI cuphyUciOnPuschSegLLRs0GetDescrInfo(size_t* pDynDescrSizeBytes,
                                                              size_t* pDynDescrAlignBytes);



cuphyStatus_t CUPHYWINAPI cuphyCreateUciOnPuschSegLLRs0(cuphyUciOnPuschSegLLRs0Hndl_t* pUciOnPuschSegLLRs0Hndl);


cuphyStatus_t CUPHYWINAPI cuphySetupUciOnPuschSegLLRs0(cuphyUciOnPuschSegLLRs0Hndl_t       uciOnPuschSegLLRs0Hndl,
                                                       uint16_t                            nUciUes,
                                                       uint16_t*                           pUciUeIdxs,
                                                       PerTbParams*                        pTbPrmsCpu,
                                                       PerTbParams*                        pTbPrmsGpu,
                                                       uint16_t                            nUeGrps,
                                                       cuphyTensorPrm_t*                   pTensorPrmsEqOutLLRs,
                                                       cuphyPuschRxUeGrpPrms_t*            pUeGrpPrmsCpu,
                                                       cuphyPuschRxUeGrpPrms_t*            pUeGrpPrmsGpu,
                                                       cuphyUciToSeg_t                     uciToSeg,
                                                       void*                               pCpuDynDesc,
                                                       void*                               pGpuDynDesc,
                                                       uint8_t                             enableCpuToGpuDescrAsyncCpy,
                                                       cuphyUciOnPuschSegLLRs0LaunchCfg_t* pLaunchCfg,
                                                       cudaStream_t                        strm);

cuphyStatus_t CUPHYWINAPI cuphyDestroyUciOnPuschSegLLRs0(cuphyUciOnPuschSegLLRs0Hndl_t uciOnPuschSegLLRs0Hndl);


////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////

/**
 * \defgroup CUPHY_UCI_ON_PUSCH_SEG_LLRS_2 uci on pusch: LLRs segmentation part 2. De-scrambles and segments SCH, CSI-PART2 LLRs
 *
 * This section describes application programming interface for part 1 UCI on PUSCH deSegmentation
 *
 * @{
 */

struct cuphyUciOnPuschSegLLRs2;
/**
 * cuPHY uciOnPuschSegLLRs1 handle
 */
typedef struct cuphyUciOnPuschSegLLRs2* cuphyUciOnPuschSegLLRs2Hndl_t;

/**
 * cuPHY polar uci on pusch LLR segmentation part 1, launch configuration
 */
typedef struct
{
    CUDA_KERNEL_NODE_PARAMS kernelNodeParamsDriver;
    void*                   kernelArgs[2];
} cuphyUciOnPuschSegLLRs2LaunchCfg_t;

/******************************************************************/ /**
 * \brief Helper to compute uciOnPuschSegLLRs2 descriptor buffer sizes and alignments
 *
 * Computes uciOnPuschSegLLRs2 descriptor buffer sizes and alignments. To be used by the caller to
 * allocate these buffers (in CPU and GPU memories) and provide them to other uciOnPuschSegLLRs2 APIs
 *
 * Returns ::CUPHY_STATUS_INVALID_ARGUMENT if \p pDynDescrSizeBytes and/or \p pDynDescrAlignBytes is NULL
 *
 * Returns ::CUPHY_STATUS_SUCCESS if allocation and initialization were
 * successful.
 *
 * \param pDynDescrSizeBytes   - Size in bytes of dynamic descriptor
 * \param pDynDescrAlignBytes  - Alignment in bytes of dynamic descriptor
 *
 * \return
 * ::CUPHY_STATUS_SUCCESS,
 * ::CUPHY_STATUS_INVALID_ARGUMENT
 *
 * \sa ::cuphyStatus_t,::cuphyGetErrorName,::cuphyGetErrorString,::cuphyCreateUciOnPuschSegLLRs1,::cuphyDestroyUciOnPuschSegLLRs1
 */
cuphyStatus_t CUPHYWINAPI cuphyUciOnPuschSegLLRs2GetDescrInfo(size_t* pDynDescrSizeBytes,
                                                              size_t* pDynDescrAlignBytes);

/******************************************************************/ /**
 * \brief Allocate and initialize a cuPHY uciOnPuschSegLLRs2 object
 *
 * Allocates a uciOnPuschSegLLRs2 object and returns a handle in the
 * address provided by the caller.
 *
 * Returns ::CUPHY_STATUS_INVALID_ARGUMENT if \p pUciOnPuschSegLLRs2Hndl is NULL.
 *
 * Returns ::CUPHY_STATUS_ALLOC_FAILED if a uciOnPuschSegLLRs2 object cannot be allocated
 *
 * Returns ::CUPHY_STATUS_SUCCESS if allocation and initialization were successful
 *
 * \param pUciOnPuschSegLLRs2Hndl   - Address to return the new uciOnPuschSegLLRs2 instance
 *
 * \return
 * ::CUPHY_STATUS_SUCCESS,
 * ::CUPHY_STATUS_ALLOC_FAILED,
 * ::CUPHY_STATUS_INVALID_ARGUMENT
 *
::cuphyStatus_t,::cuphyGetErrorName,::cuphyGetErrorString,::cuphyUciOnPuschSegLLRs2GetDescrInfo,::cuphySetupUciOnPuschSegLLRs2 */

cuphyStatus_t CUPHYWINAPI cuphyCreateUciOnPuschSegLLRs2(cuphyUciOnPuschSegLLRs2Hndl_t* pUciOnPuschSegLLRs2Hndl);

/******************************************************************/ /**
 * \brief Setup cuPHY uciOnPuschSegLLRs2 for slot processing
 *
 * Setup cuPHY uciOnPuschSegLLRs2 in preparation towards slot execution
 *
 * Returns ::CUPHY_STATUS_SUCCESS if setup is successful.
 *
 * \param uciOnPuschSegLLRs2Hndl       - handle of uciOnPuschSegLLRs2 instance
 * \param nCsi2Ues                     - number of UES bearing CSI part2 payload data
 * \param pCsi2UeIdxs                  - indices of CSI part2 payload bearing UEs (index to resolve UE from set of all UEs being processed by PUSCH)
 * \param pTbPrmsCpu                   - address of Transport block parameters in CPU memory
 * \param pTbPrmsGpu                   - address of Transport block parameters in GPU memory
 * \param nUeGrps                      - number of UE groups to be processed
 * \param pTensorPrmsEqOutLLRs         - tensor parameters for equalizer output LLRs
 * \param pUeGrpPrmsCpu                - UE group parameters in CPU memory
 * \param pUeGrpPrmsGpu                - UE group parameters in GPU memory
 * \param pCpuDynDesc                  - pointer to dynamic descriptor in CPU memory
 * \param pGpuDynDesc                  - pointer to dynamic descriptor in GPU memory
 * \param enableCpuToGpuDescrAsyncCpy  - Flag when set enables async copy of CPU descriptor into GPU
 * \param pLaunchCfg                   - Pointer to channel estimation launch configurations
 * \param strm                         - CUDA stream for descriptor copy operation
 * \return
 * ::CUPHY_STATUS_SUCCESS,
 * ::CUPHY_STATUS_INVALID_ARGUMENT
 *
 * \sa ::cuphyStatus_t,::cuphyGetErrorName,::cuphyGetErrorString,::cuphyUciOnPuschSegLLRs2GetDescrInfo,::cuphyDestroyUciOnPuschSegLLRs2
 */

cuphyStatus_t CUPHYWINAPI cuphySetupUciOnPuschSegLLRs2(cuphyUciOnPuschSegLLRs2Hndl_t       uciOnPuschSegLLRs2Hndl,
                                                       uint16_t                            nCsi2Ues,
                                                       uint16_t*                           pCsi2UeIdxs,
                                                       PerTbParams*                        pTbPrmsCpu,
                                                       PerTbParams*                        pTbPrmsGpu,
                                                       uint16_t                            nUeGrps,
                                                       cuphyTensorPrm_t*                   pTensorPrmsEqOutLLRs,
                                                       cuphyPuschRxUeGrpPrms_t*            pUeGrpPrmsCpu,
                                                       cuphyPuschRxUeGrpPrms_t*            pUeGrpPrmsGpu,
                                                       void*                               pCpuDynDesc,
                                                       void*                               pGpuDynDesc,
                                                       uint8_t                             enableCpuToGpuDescrAsyncCpy,
                                                       cuphyUciOnPuschSegLLRs2LaunchCfg_t* pLaunchCfg,
                                                       cudaStream_t                        strm);

/******************************************************************/ /**
 * \brief Destroys a cuPHY uciOnPuschSegLLRs2 object
 *
 * Destroys a cuPHY uciOnPuschSegLLRs2 object that was previously
 * created by ::cuphyCreateUciOnPuschSegLLRs2. The handle provided to this function
 * should not be used for any operations after this function returns.
 *
 * Returns ::CUPHY_STATUS_INVALID_ARGUMENT if \p uciOnPuschSegLLRs2Hndl is NULL.
 *
 * Returns ::CUPHY_STATUS_SUCCESS if destruction was successful.
 *
 * \param uciOnPuschSegLLRs2Hndl - handle to previously allocated  instance
 *
 * \return
 * ::CUPHY_STATUS_SUCCESS,q
 * ::CUPHY_STATUS_INVALID_ARGUMENT
 *
 * \sa ::cuphyStatus_t,::cuphyGetErrorName,::cuphyGetErrorString,::cuphyUciOnPuschSegLLRs2GetDescrInfo,::cuphyCreateUciOnPuschSegLLRs2,::cuphySetupUciOnPuschSegLLRs2
 */
cuphyStatus_t CUPHYWINAPI cuphyDestroyUciOnPuschSegLLRs2(cuphyUciOnPuschSegLLRs2Hndl_t uciOnPuschSegLLRs2Hndl);

/** @} */ /* END CUPHY_UCI_ON_PUSCH_SEG_LLRS_2 */

////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////

/**
 * \defgroup CUPHY_UCI_ON_PUSCH_CSI2_CTRL uci on pusch: computes number of CSI-P2 bits. Setups CSI-P2 backend.
 *
 * This section describes application programming interface for the uciOnPusch CSI-P2 control kernel
 *
 * @{
 */

struct cuphyUciOnPuschCsi2Ctrl;
/**
 * cuPHY uciOnPuschCsi2Ctrl handle
 */
typedef struct cuphyUciOnPuschCsi2Ctrl* cuphyUciOnPuschCsi2CtrlHndl_t;

/**
 * cuPHY polar uci on pusch Csi2 control, launch configuration
 */
typedef struct
{
    CUDA_KERNEL_NODE_PARAMS kernelNodeParamsDriver;
    void*                   kernelArgs[2];
} cuphyUciOnPuschCsi2CtrlLaunchCfg_t;

/******************************************************************/ /**
 * \brief Helper to compute uciOnPuschCsi2Ctrl descriptor buffer sizes and alignments
 *
 * Computes uciOnPuschCsi2Ctrl descriptor buffer sizes and alignments. To be used by the caller to
 * allocate these buffers (in CPU and GPU memories) and provide them to other uciOnPuschCsi2Ctrl APIs
 *
 * Returns ::CUPHY_STATUS_INVALID_ARGUMENT if \p pDynDescrSizeBytes and/or \p pDynDescrAlignBytes is NULL
 *
 * Returns ::CUPHY_STATUS_SUCCESS if allocation and initialization were
 * successful.
 *
 * \param pDynDescrSizeBytes   - Size in bytes of dynamic descriptor
 * \param pDynDescrAlignBytes  - Alignment in bytes of dynamic descriptor
 *
 * \return
 * ::CUPHY_STATUS_SUCCESS,
 * ::CUPHY_STATUS_INVALID_ARGUMENT
 *
 * \sa ::cuphyStatus_t,::cuphyGetErrorName,::cuphyGetErrorString,::cuphyCreateUciOnPuschSegLLRs1,::cuphyDestroyUciOnPuschSegLLRs1
 */
cuphyStatus_t CUPHYWINAPI cuphyUciOnPuschCsi2CtrlGetDescrInfo(size_t* pDynDescrSizeBytes,
                                                              size_t* pDynDescrAlignBytes);

/******************************************************************/ /**
 * \brief Allocate and initialize a cuPHY uciOnPuschCsi2Ctrl object
 *
 * Allocates a uciOnPuschCsi2Ctrl object and returns a handle in the
 * address provided by the caller.
 *
 * Returns ::CUPHY_STATUS_INVALID_ARGUMENT if \p pUciOnPuschCsi2CtrlHndl  is NULL.
 *
 * Returns ::CUPHY_STATUS_ALLOC_FAILED if a uciOnPuschCsi2Ctrl object cannot be allocated
 *
 * Returns ::CUPHY_STATUS_SUCCESS if allocation and initialization were successful
 *
 * \param pUciOnPuschCsi2CtrlHndl   - Address to return the new uciOnPuschCsi2Ctrl instance
 *
 * \return
 * ::CUPHY_STATUS_SUCCESS,
 * ::CUPHY_STATUS_ALLOC_FAILED,
 * ::CUPHY_STATUS_INVALID_ARGUMENT
 *
::cuphyStatus_t,::cuphyGetErrorName,::cuphyGetErrorString,::cuphyUciOnPuschSegLLRs2GetDescrInfo,::cuphySetupUciOnPuschSegLLRs2 */

cuphyStatus_t CUPHYWINAPI cuphyCreateUciOnPuschCsi2Ctrl(cuphyUciOnPuschCsi2CtrlHndl_t* pUciOnPuschCsi2CtrlHndl);

/******************************************************************/ /**
 * \brief Setup cuPHY uciOnPuschCsi2Ctrl for slot processing
 *
 * Setup cuPHY uciOnPuschCsi2Ctrl in preparation towards slot execution
 *
 * Returns ::CUPHY_STATUS_SUCCESS if setup is successful.
 *
 * \param uciOnPuschCsi2CtrlHndl      - Handle for PUSCH CSI part 2 component instance
 * \param nCsi2Ues                    - number of UES bearing CSI part2 payload
 * \param pCsi2UeIdxsCpu              - indices of CSI part2 payload bearing UEs in CPU memory (index to resolve UE from set of all UEs being processed by PUSCH)
 * \param pTbPrmsCpu                  - address of Transport block parameters in CPU memory
 * \param pTbPrmsGpu                  - address of Transport block parameters in GPU memory
 * \param pUeGrpPrmsCpu               - UE group parameters in CPU memory
 * \param pCellStatPrmsGpu            - cell static parameters specific to PUSCH pipeline
 * \param pUciOnPuschOutOffsetsCpu    - pointer to any array of structures containing per UE offsets for locating PUSCH outputs
 * \param pUciPayloadsGpu             - pointer to UCI payloads in GPU
 * \param pNumCsi2BitsGpu             - pointer to array containing number of CSI part2 payload bits
 * \param pCsi2PolarSegPrmsGpu        - pointer to parameters for polar encoded UCI segment
 * \param pCsi2PolarCwPrmsGpu         - pointer to parameters for polar code words
 * \param pCsi2RmCwPrmsGpu            - Reed-muller decoder code word parameters in GPU memory
 * \param pCsi2SpxCwPrmsGpu           - simplex decoder code word parameters in GPU memory
 * \param forcedNumCsi2Bits           - Debug feature. if > 0 cuPHY assumes all csi2 UCIs have forcedNumCsi2Bits bits
 * \param enableCsiP2Fapiv3           - flag to enable FAPIv3 method for computing CSI-P2 size
 * \param pCpuDynDesc                 - pointer to dynamic descriptor in CPU memory
 * \param pGpuDynDesc                 - pointer to dynamic descriptor in GPU memory
 * \param enableCpuToGpuDescrAsyncCpy - Flag when set enables async copy of CPU descriptor into GPU
 * \param pLaunchCfg                  - Pointer to channel estimation launch configurations
 * \param strm                        - CUDA stream for descriptor copy operation
 * \return
 * ::CUPHY_STATUS_SUCCESS,
 * ::CUPHY_STATUS_INVALID_ARGUMENT
 *
 * \sa ::cuphyStatus_t,::cuphyGetErrorName,::cuphyGetErrorString,::cuphyUciOnPuschCsi2CtrlGetDescrInfo,::cuphyDestroyUciOnPuschCsi2Ctrl
 */

cuphyStatus_t CUPHYWINAPI cuphySetupUciOnPuschCsi2Ctrl(cuphyUciOnPuschCsi2CtrlHndl_t       uciOnPuschCsi2CtrlHndl,
                                                       uint16_t                            nCsi2Ues,
                                                       uint16_t*                           pCsi2UeIdxsCpu,
                                                       PerTbParams*                        pTbPrmsCpu,
                                                       PerTbParams*                        pTbPrmsGpu,
                                                       cuphyPuschRxUeGrpPrms_t*            pUeGrpPrmsCpu,
                                                       cuphyPuschCellStatPrm_t*            pCellStatPrmsGpu,
                                                       cuphyUciOnPuschOutOffsets_t*        pUciOnPuschOutOffsetsCpu,
                                                       uint8_t*                            pUciPayloadsGpu,
                                                       uint16_t*                           pNumCsi2BitsGpu,
                                                       cuphyPolarUciSegPrm_t*              pCsi2PolarSegPrmsGpu,
                                                       cuphyPolarCwPrm_t*                  pCsi2PolarCwPrmsGpu,
                                                       cuphyRmCwPrm_t*                     pCsi2RmCwPrmsGpu,
                                                       cuphySimplexCwPrm_t*                pCsi2SpxCwPrmsGpu,
                                                       uint16_t                            forcedNumCsi2Bits,
                                                       uint8_t                             enableCsiP2Fapiv3,
                                                       void*                               pCpuDynDesc,
                                                       void*                               pGpuDynDesc,
                                                       uint8_t                             enableCpuToGpuDescrAsyncCpy,
                                                       cuphyUciOnPuschCsi2CtrlLaunchCfg_t* pLaunchCfg,
                                                       cudaStream_t                        strm);

/******************************************************************/ /**
 * \brief Destroys a cuPHY uciOnPuschCsi2Ctrl object
 *
 * Destroys a cuPHY uciOnPuschCsi2Ctrl object that was previously
 * created by ::cuphyCreateUciOnPuschCsi2Ctrl. The handle provided to this function
 * should not be used for any operations after this function returns.
 *
 * Returns ::CUPHY_STATUS_INVALID_ARGUMENT if \p uciOnPuschCsi2CtrlHndl is NULL.
 *
 * Returns ::CUPHY_STATUS_SUCCESS if destruction was successful.
 *
 * \param uciOnPuschCsi2CtrlHndl - handle to previously allocated  instance
 *
 * \return
 * ::CUPHY_STATUS_SUCCESS,q
 * ::CUPHY_STATUS_INVALID_ARGUMENT
 *
 * \sa ::cuphyStatus_t,::cuphyGetErrorName,::cuphyGetErrorString,::cuphyUciOnPuschCsi2CtrlGetDescrInfo,::cuphyCreateUciOnPuschCsi2Ctrl,::cuphySetupUciOnPuschCsi2Ctrl
 */
cuphyStatus_t CUPHYWINAPI cuphyDestroyUciOnPuschCsi2Ctrl(cuphyUciOnPuschCsi2CtrlHndl_t uciOnPuschCsi2CtrlHndl);

/** @} */ /* END CUPHY_UCI_ON_PUSCH_CSI2_CTRL */

////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////

/**
 * \defgroup CUPHY_POLAR_DECODER Polar Decoder
 *
 * This section describes application programming interface for polar decoder
 *
 * @{
 */

struct cuphyPolarDecoder;
/**
 * cuPHY  uciPolDecoder handle
 */
typedef struct cuphyPolarDecoder* cuphyPolarDecoderHndl_t;

/**
 * cuPHY polarDecoder launch configuration
 */
typedef struct
{
    CUDA_KERNEL_NODE_PARAMS kernelNodeParamsDriver;
    void*                   kernelArgs[2];
} cuphyPolarDecoderLaunchCfg_t;

/******************************************************************/ /**
 * \brief Helper to compute polarDecoder descriptor buffer sizes and alignments
 *
 * Computes polarDecoder descriptor buffer sizes and alignments. To be used by the caller to
 * allocate these buffers (in CPU and GPU memories) and provide them to other uciPolDecoder APIs
 *
 * Returns ::CUPHY_STATUS_INVALID_ARGUMENT if \p pDynDescrSizeBytes and/or \p pDynDescrAlignBytes is NULL
 *
 * Returns ::CUPHY_STATUS_SUCCESS if allocation and initialization were
 * successful.
 *
 * \param pDynDescrSizeBytes   - Size in bytes of dynamic descriptor
 * \param pDynDescrAlignBytes  - Alignment in bytes of dynamic descriptor
 *
 * \return
 * ::CUPHY_STATUS_SUCCESS,
 * ::CUPHY_STATUS_INVALID_ARGUMENT
 *
 * \sa ::cuphyStatus_t,::cuphyGetErrorName,::cuphyGetErrorString,::cuphyCreateUciOnPuschSegLLRs1,::cuphyDestroyUciOnPuschSegLLRs1
 */
cuphyStatus_t CUPHYWINAPI cuphyPolarDecoderGetDescrInfo(size_t* pDynDescrSizeBytes,
                                                        size_t* pDynDescrAlignBytes);

/******************************************************************/ /**
 * \brief Allocate and initialize a cuPHY polarDecoder object
 *
 * Allocates a polarDecoder object and returns a handle in the
 * address provided by the caller.
 *
 * Returns ::CUPHY_STATUS_INVALID_ARGUMENT if \p pPolarDecoderHndl  is NULL.
 *
 * Returns ::CUPHY_STATUS_ALLOC_FAILED if a polarDecoder object cannot be allocated
 *
 * Returns ::CUPHY_STATUS_SUCCESS if allocation and initialization were successful
 *
 * \param pPolarDecoderHndl   - Address to return the new polarDecoder instance
 *
 * \return
 * ::CUPHY_STATUS_SUCCESS,
 * ::CUPHY_STATUS_ALLOC_FAILED,
 * ::CUPHY_STATUS_INVALID_ARGUMENT
 *
::cuphyStatus_t,::cuphyGetErrorName,::cuphyGetErrorString,::cuphyPolarDecoderGetDescrInfo,::cuphySetupPolarDecoder */

cuphyStatus_t CUPHYWINAPI cuphyCreatePolarDecoder(cuphyPolarDecoderHndl_t* pPolarDecoderHndl);

/******************************************************************/ /**
 * \brief Setup cuPHY polar decoder for slot processing
 *
 * Setup cuPHY polar decoder in preparation towards slot execution
 *
 * Returns ::CUPHY_STATUS_SUCCESS if setup is successful.
 *
 * \param polarDecoderHndl                  - polar decoder component handle
 * \param nPolCws                           - number of polar codewords
 * \param pCwTreeLLRsAddrs                  - pointer to codeword tree LLR addresses
 * \param pCwPrmsGpu                        - pointer to codeword parameters in GPU
 * \param pCwPrmsCpu                        - pointer to codeword parameters in CPU
 * \param pPolCbEstAddrs                    - pointer to estimated codeblock addresses
 * \param pListPolScratchAddrs              - pointer to scratch buffer used in list polar decoder
 * \param nPolarList                        - list size for polar decoder
 * \param pPolCrcErrorFlags                 - pointer to buffer storing CRC error flags
 * \param enableCpuToGpuDescrAsyncCpy       - option to copy descriptors from CPU to GPU
 * \param pCpuDynDescPolar                  - pointer to polarDecoderDynDescr descriptor in cpu
 * \param pGpuDynDescPolar                  - pointer to polarDecoderDynDescr descriptor in gpu
 * \param pCpuDynDescPolarLLRAddrs          - pointer to cwTreeLLRsAddrs in polarDecoderDynDescr descriptor in cpu
 * \param pCpuDynDescPolarCBAddrs           - pointer to polCbEstAddrs in polarDecoderDynDescr descriptor in cpu
 * \param pCpuDynDescListPolarScratchAddrs  - pointer to listPolScratchAddrs in polarDecoderDynDescr descriptor in cpu
 * \param pLaunchCfg                        - pointer to launch configuration
 * \param strm                              - stream to perform copy
 *
 * \return
 * ::CUPHY_STATUS_SUCCESS,
 * ::CUPHY_STATUS_INVALID_ARGUMENT
 *
 * \sa ::cuphyStatus_t,::cuphyGetErrorName,::cuphyGetErrorString,::cuphyPolarDecoderGetDescrInfo,::cuphyDestroyPolarDecoder
 */

cuphyStatus_t CUPHYWINAPI cuphySetupPolarDecoder(cuphyPolarDecoderHndl_t       polarDecoderHndl,
                                                 uint16_t                      nPolCws,
                                                 __half**                      pCwTreeLLRsAddrs,
                                                 cuphyPolarCwPrm_t*            pCwPrmsGpu,
                                                 cuphyPolarCwPrm_t*            pCwPrmsCpu,
                                                 uint32_t**                    pPolCbEstAddrs,
                                                 bool**                        pListPolScratchAddrs,
                                                 uint8_t                       nPolarList,
                                                 uint8_t*                      pPolCrcErrorFlags,
                                                 bool                          enableCpuToGpuDescrAsyncCpy,
                                                 void*                         pCpuDynDescPolar,
                                                 void*                         pGpuDynDescPolar,
                                                 void*                         pCpuDynDescPolarLLRAddrs,
                                                 void*                         pCpuDynDescPolarCBAddrs,
                                                 void*                         pCpuDynDescListPolarScratchAddrs,
                                                 cuphyPolarDecoderLaunchCfg_t* pLaunchCfg,
                                                 cudaStream_t                  strm);

/******************************************************************/ /**
 * \brief Destroys a cuPHY polarDecoder object
 *
 * Destroys a cuPHY polarDecoder object that was previously
 * created by ::cuphyCreatePolarDecoder. The handle provided to this function
 * should not be used for any operations after this function returns.
 *
 * Returns ::CUPHY_STATUS_INVALID_ARGUMENT if \p polarDecoderHndl is NULL.
 *
 * Returns ::CUPHY_STATUS_SUCCESS if destruction was successful.
 *
 * \param polarDecoderHndl - handle to previously allocated compCwTreeTypes instance
 *
 * \return
 * ::CUPHY_STATUS_SUCCESS,
 * ::CUPHY_STATUS_INVALID_ARGUMENT
 *
 * \sa ::cuphyStatus_t,::cuphyGetErrorName,::cuphyGetErrorString,::cuphyPolarDecoderGetDescrInfo,::cuphyCreatePolarDecoder,::cuphySetupPolarDecoder
 */
cuphyStatus_t CUPHYWINAPI cuphyDestroyPolarDecoder(cuphyPolarDecoderHndl_t polarDecoderHndl);


/** @} */ /* END CUPHY_POLAR_DECODER */

////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////



/**
 * \defgroup CUPHY_SRS_CHEST Srs channel estimator
 *
 * This section describes application programming interface for srs channel estimator
 *
 * @{
 */

// Parameters for SRS
typedef struct _cuphyUeSrsPrm
{
        uint16_t cellIdx;                  // index of cell user belongs to
        uint8_t  nAntPorts;                // number of SRS antenna ports. 1,2, or 4
        uint8_t  nSyms;                    // number of SRS symbols. 1,2, or 4
        uint8_t  nRepetitions;             // number of repititions. 1,2, or 4
        uint8_t  combSize;                 // SRS comb size. 2 or 4
        uint8_t  startSym;                 // starting SRS symbol. 0-13
        uint16_t sequenceId;               // SRS sequence id. 0-1023
        uint8_t  configIdx;                // SRS bandwidth cfg idx. 0-63
        uint8_t  bandwidthIdx;             // SRS bandwidth index. 0-3
        uint8_t  combOffset;               // SRS comb offset. 0-3
        uint8_t  cyclicShift;              // cyclic shift. 0-11
        uint8_t  frequencyPosition;        // frequency domain position. 0-67
        uint16_t frequencyShift;           // frequency domain shift. 0-268
        uint8_t  frequencyHopping;         // freuqnecy hopping options. 0-3
        uint8_t  resourceType;             // Type of SRS allocation. 0: aperiodic. 1: semi-persistent. 2: periodic
        uint16_t Tsrs;                     // SRS periodicity in slots. 0,2,3,5,8,10,16,20,32,40,64,80,160,320,640,1280,2560
        uint16_t Toffset;                  // slot offset value. 0-2569
        uint8_t  groupOrSequenceHopping;   // Hopping configuration. 0: no hopping. 1: groupHopping. 2: sequenceHopping
        uint16_t chEstBuffIdx;             // index of which chEstBuff to store SRS ChEsts into
        uint8_t  srsAntPortToUeAntMap[4];  // mapping between SRS antenna ports and UE antennas in ChEst buffer:
                                           // store ChEst for srsAntPort_i in srsAntPortToUeAntMap[i]
        uint16_t rnti;                     // user rnti (not used by cuPHY L1)
        uint32_t handle;                   // user handle (not used by cuPHY L1)
        uint16_t prgSize;                  // number of PRBs per PRBG
        uint16_t RNTI;                     // RNTI of the UE
        uint32_t usage;
        uint16_t srsStartPrg;
        uint16_t srsChestBufferIndexL2; 
        uint16_t startValidPrg;
        uint16_t nValidPrg;

} cuphyUeSrsPrm_t;

// Parameters for SRS TX
typedef struct _cuphyUeSrsTxPrm
{
        uint8_t  nAntPorts;                // number of SRS antenna ports. 1,2, or 4
        uint8_t  nSyms;                    // number of SRS symbols. 1,2, or 4
        uint8_t  nRepetitions;             // number of repititions. 1,2, or 4
        uint8_t  combSize;                 // SRS comb size. 2 or 4
        uint8_t  startSym;                 // starting SRS symbol. 0-13
        uint16_t sequenceId;               // SRS sequence id. 0-1023
        uint8_t  configIdx;                // SRS bandwidth cfg idx. 0-63
        uint8_t  bandwidthIdx;             // SRS bandwidth index. 0-3
        uint8_t  combOffset;               // SRS comb offset. 0-3
        uint8_t  cyclicShift;              // cyclic shift. 0-11
        uint8_t  frequencyPosition;        // frequency domain position. 0-67
        uint16_t frequencyShift;           // frequency domain shift. 0-268
        uint8_t  frequencyHopping;         // freuqnecy hopping options. 0-3
        uint8_t  resourceType;             // Type of SRS allocation. 0: aperiodic. 1: semi-persistent. 2: periodic
        uint16_t Tsrs;                     // SRS periodicity in slots. 0,2,3,5,8,10,16,20,32,40,64,80,160,320,640,1280,2560
        uint16_t Toffset;                  // slot offset value. 0-2569
        uint8_t  groupOrSequenceHopping;   // Hopping configuration. 0: no hopping. 1: groupHopping. 2: sequenceHopping
        uint16_t idxSlotInFrame;
        uint16_t idxFrame;
} cuphyUeSrsTxPrm_t;

// SRS cell parameters
typedef struct _cuphySrsCellPrms
{
    uint16_t slotNum;
    uint16_t frameNum;
    uint8_t  srsStartSym;
    uint8_t  nSrsSym;
    uint16_t nRxAntSrs;
    uint8_t  mu;
}cuphySrsCellPrms_t;

// SRS output structure
typedef struct _cuphySrsReport
{
    float   toEstMicroSec;
    float   widebandSnr;
    float   widebandNoiseEnergy;    // need to be initialized to 0
    float   widebandSignalEnergy;   // need to be initialized to 0
    __half2 widebandScCorr;         // need to be initialized to {0, 0}
    float   widebandCsCorrRatioDb;  // need to be initialized to 0
    float   widebandCsCorrUse;      // need to be initialized to 0
    float   widebandCsCorrNotUse;   // need to be initialized to 0
    uint8_t highDensityAntPortFlag; // Flag indicates if high density of cover-codes muxed on srs comb. Means SNR measurments unreliable.
} cuphySrsReport_t;


// Srs ChEst to L2
typedef struct _cuphySrsChEstToL2
{
    uint8_t*              pChEstCpuBuff; // Pointer to CPU ChEst buffer.
    uint16_t              prbGrpSize;    // Prb group size
    uint16_t              nPrbGrps;      // number of Prb groups
}cuphySrsChEstToL2_t;

// SRS filter parameters
typedef struct _cuphySrsFilterPrms
{
    cuphyTensorPrm_t tPrmFocc_table;
    cuphyTensorPrm_t tPrmFocc_comb2_table;
    cuphyTensorPrm_t tPrmFocc_comb4_table;

    cuphyTensorPrm_t tPrmW_comb2_nPorts1_wide;  // CUPHY_C_16F. Dim: 24 x 24
    cuphyTensorPrm_t tPrmW_comb2_nPorts2_wide;  // CUPHY_C_16F. Dim: 24 x 24
    cuphyTensorPrm_t tPrmW_comb2_nPorts4_wide;  // CUPHY_C_16F. Dim: 24 x 24
    cuphyTensorPrm_t tPrmW_comb2_nPorts8_wide;  // CUPHY_C_16F. Dim: 24 x 24

    cuphyTensorPrm_t tPrmW_comb4_nPorts1_wide;   // CUPHY_C_16F. Dim: 12 x 12
    cuphyTensorPrm_t tPrmW_comb4_nPorts2_wide;   // CUPHY_C_16F. Dim: 12 x 12
    cuphyTensorPrm_t tPrmW_comb4_nPorts4_wide;   // CUPHY_C_16F. Dim: 12 x 12
    cuphyTensorPrm_t tPrmW_comb4_nPorts6_wide;   // CUPHY_C_16F. Dim: 12 x 12
    cuphyTensorPrm_t tPrmW_comb4_nPorts12_wide;  // CUPHY_C_16F. Dim: 12 x 12

    cuphyTensorPrm_t tPrmW_comb2_nPorts1_narrow;  // CUPHY_C_16F. Dim: 24 x 24
    cuphyTensorPrm_t tPrmW_comb2_nPorts2_narrow;  // CUPHY_C_16F. Dim: 24 x 24
    cuphyTensorPrm_t tPrmW_comb2_nPorts4_narrow;  // CUPHY_C_16F. Dim: 24 x 24
    cuphyTensorPrm_t tPrmW_comb2_nPorts8_narrow;  // CUPHY_C_16F. Dim: 24 x 24

    cuphyTensorPrm_t tPrmW_comb4_nPorts1_narrow;  // CUPHY_C_16F. Dim: 12 x 12
    cuphyTensorPrm_t tPrmW_comb4_nPorts2_narrow;  // CUPHY_C_16F. Dim: 12 x 12
    cuphyTensorPrm_t tPrmW_comb4_nPorts4_narrow;  // CUPHY_C_16F. Dim: 12 x 12
    cuphyTensorPrm_t tPrmW_comb4_nPorts6_narrow;  // CUPHY_C_16F. Dim: 12 x 12
    cuphyTensorPrm_t tPrmW_comb4_nPorts12_narrow; // CUPHY_C_16F. Dim: 12 x 12

    float noisEstDebias_comb2_nPorts1;
    float noisEstDebias_comb2_nPorts2;
    float noisEstDebias_comb2_nPorts4;
    float noisEstDebias_comb2_nPorts8;

    float noisEstDebias_comb4_nPorts1;
    float noisEstDebias_comb4_nPorts2;
    float noisEstDebias_comb4_nPorts4;
    float noisEstDebias_comb4_nPorts6;
    float noisEstDebias_comb4_nPorts12;
} cuphySrsFilterPrms_t;



typedef struct _cuphySrsChEstBuffInfo
{
    cuphyTensorPrm_t tChEstBuffer;     // Tensor parameters for SRS channel estimation buffer. Dim: nPrbGrpEsts x nGnbAnts x nUeAnts
                                       // Tensors reside in GPU memory
    uint16_t         startPrbGrp;      // The first prb group in ChEst buffer.
    uint16_t         srsPrbGrpSize;    // number of prbs per PRB group
    uint16_t         startValidPrg;
    uint16_t         nValidPrg;
} cuphySrsChEstBuffInfo_t;


struct cuphySrsChEst;
/**
 * cuPHY  srsChEst handle
 */
typedef struct cuphySrsChEst* cuphySrsChEstHndl_t;

/**
 * cuPHY srs chEst launch configuration
 */
typedef struct
{
    CUDA_KERNEL_NODE_PARAMS kernelNodeParamsDriver;
    void*                   kernelArgs[2];
} cuphySrsChEstLaunchCfg_t;


typedef struct
{
    CUDA_KERNEL_NODE_PARAMS kernelNodeParamsDriver;
    void*                   kernelArgs[2];
} cuphySrsChEstNormalizationLaunchCfg_t;


/******************************************************************
 * \brief Helper to compute cuPHY SRS channel estimation descriptor buffer sizes and alignments
 *
 * Computes cuPHY SRS channel estimation descriptor buffer sizes and alignments. To be used by the caller to
 * allocate these buffers (in CPU and GPU memories) and provide them to other SrsChEst APIs
 *
 * Returns ::CUPHY_STATUS_INVALID_ARGUMENT if \p pStatDescrSizeBytes and/or \p pStatDescrAlignBytes and/or
 * \p pDynDescrSizeBytes and/or \p pDynDescrAlignBytes is NULL
 *
 * Returns ::CUPHY_STATUS_SUCCESS if allocation and initialization were
 * successful.
 *
 * \param pStatDescrSizeBytes  - Size in bytes of static descriptor
 * \param pStatDescrAlignBytes - Alignment in bytes of static descriptor
 * \param pDynDescrSizeBytes   - Size in bytes of dynamic descriptor
 * \param pDynDescrAlignBytes  - Alignment in bytes of dynamic descriptor
 *
 * \return
 * ::CUPHY_STATUS_SUCCESS,
 * ::CUPHY_STATUS_INVALID_ARGUMENT
 *
 * \sa ::cuphyStatus_t,::cuphyGetErrorName,::cuphyGetErrorString,::cuphyCreateSrsChEst,::cuphyDestroySrsChE
*/
cuphyStatus_t CUPHYWINAPI cuphySrsChEstGetDescrInfo(size_t* pStatDescrSizeBytes,
                                                     size_t* pStatDescrAlignBytes,
                                                     size_t* pDynDescrSizeBytes,
                                                     size_t* pDynDescrAlignBytes);

/******************************************************************
 * \brief Allocate and initialize a cuPHY SRS channel estimation object
 *
 * Allocates a cuPHY channel estimation object and returns a handle in the
 * address provided by the caller.
 *
 * Returns ::CUPHY_STATUS_INVALID_ARGUMENT if \p pSrsChEstHndl and/or \p pSrsFilterPrms and/or \p pShiftSeq
 * and/or \p pUnShiftSeq and/or \p pCpuStatDesc and/or \p pGpuStatDesc is NULL.
 *
 * Returns ::CUPHY_STATUS_ALLOC_FAILED if a SrsChEst object cannot be allocated
 *
 * Returns ::CUPHY_STATUS_SUCCESS if allocation and initialization were successful
 *
 * \param pSrsChEstHndl               - Address to return the new SrsChEst instance
 * \param pSrsFilterPrms              - Pointer to SRS filter parameters
 * \param pRkhsPrms                   - pointer to SRS SRKHS paramaters
 * \param chEstAlgo                   - SRS channel estimation algorithm
 * \param chEstToL2NormalizationAlgo  - Normalization algorithm
 * \param chEstToL2ConstantScaler     - Constant scaling factor
 * \param enableDelayOffsetCorrection - Flag to enable delay offset correction
 * \param enableCpuToGpuDescrAsyncCpy - Flag if non-zero enables async copy of CPU descriptor into GPU
 * \param pCpuStatDesc                - Pointer to static descriptor in CPU memory
 * \param pGpuStatDesc                - Pointer to static descriptor in GPU memory
 * \param strm                        - CUDA stream for descriptor copy operation
 *
 * \return
 * ::CUPHY_STATUS_SUCCESS,
 * ::CUPHY_STATUS_ALLOC_FAILED,
 * ::CUPHY_STATUS_INVALID_ARGUMENT
 *
 * \sa ::cuphyStatus_t,::cuphyGetErrorName,::cuphyGetErrorString,::cuphySrsChEstGetDescrInfo,::cuphySetupSrsChEst,::cuphyDestroySrsChEst
 */

cuphyStatus_t CUPHYWINAPI cuphyCreateSrsChEst(  cuphySrsChEstHndl_t*    pSrsChEstHndl,
                                                cuphySrsFilterPrms_t*   pSrsFilterPrms,
                                                cuphySrsRkhsPrms_t*     pRkhsPrms,
                                                cuphySrsChEstAlgoType_t chEstAlgo,
                                                uint8_t                 chEstToL2NormalizationAlgo,
                                                float                   chEstToL2ConstantScaler,
                                                uint8_t                 enableDelayOffsetCorrection,
                                                uint8_t                 enableCpuToGpuDescrAsyncCpy,
                                                void*                   pCpuStatDesc,
                                                void*                   pGpuStatDesc,
                                                cudaStream_t            strm);

/******************************************************************
 * \brief Setup cuPHY SRS channel estimation for slot processing
 *
 * Setup cuPHY SRS channel estimation in preparation towards slot execution
 *
 * Returns ::CUPHY_STATUS_SUCCESS if setup is successful.
 *
 * Returns ::CUPHY_STATUS_INVALID_ARGUMENT if \p srsChEstHndl and/or \p h_srsUePrms and/or \p h_srsCellPrms and/or
 * \p d_rbSnrBuff and/or \p h_rbSnrBuffOffsets and/or \p d_pSrsReports and/or \p d_addrsChEstToL2Buff and/or \p h_chEstToL2 and/or \p d_pSrsReports and.or \p h_chEstBuffInfo and/or \p pCpuDynDesc and/or \p pGpuDynDesc and/or \p pLaunchCfg is NULL.
 *
 * \param srsChEstHndl                - Handle to previously allocated SrsChEst instance
 * \param nSrsUes                     - Number of SRS UEs
 * \param h_srsUePrms                 - Pointer to SRS UE parameters
 * \param nCells                      - Number of cells
 * \param pTDataRx                    - Pointer to received data tensor parameters
 * \param h_srsCellPrms               - Host pointer to estimated channel tensor parameters
 * \param d_rbSnrBuff                 - Device pointer to debug tensor parameters
 * \param h_rbSnrBuffOffsets          - Host pointer to rb SNR buffer offsets
 * \param h_chEstBuffInfo             - Host pointer to channel estimation buffer information
 * \param d_pSrsReports               - Device pointer to SRS reports
 * \param d_addrsChEstToL2InnerBuff   - Device pointer to channel estimation to L2 buffer
 * \param d_addrsChEstToL2Buff        - Device pointer to channel estimation to L2 buffer
 * \param h_chEstToL2                 - Host pointer to channel estimation to L2 buffer
 * \param enableCpuToGpuDescrAsyncCpy - Flag when set enables async copy of CPU descriptor into GPU
 * \param pCpuDynDesc                 - Pointer to dynamic descriptor in CPU memory
 * \param pGpuDynDesc                 - Pointer to dynamic descriptor in GPU memory
 * \param pLaunchCfg                  - Pointer to launch configuration
 * \parma pNormalizationLaunchCfg     - Pointer to launch configuration for the normalization kernel
 * \param strm                        - CUDA stream for descriptor copy operation
 * \return
 * ::CUPHY_STATUS_SUCCESS,
 * ::CUPHY_STATUS_INVALID_ARGUMENT
 *
 * \sa ::cuphyStatus_t,::cuphyGetErrorName,::cuphyGetErrorString,::cuphySrsChEstGetDescrInfo,::cuphyCreateSrsChEst,::cuphyDestroySrsChEst
 */
cuphyStatus_t CUPHYWINAPI cuphySetupSrsChEst(   cuphySrsChEstHndl_t          srsChEstHndl,
                                                 uint16_t                      nSrsUes,
                                                 cuphyUeSrsPrm_t*              h_srsUePrms,
                                                 uint16_t                      nCells,
                                                 cuphyTensorPrm_t*             pTDataRx,
                                                 cuphySrsCellPrms_t*           h_srsCellPrms,
                                                 float*                        d_rbSnrBuff,
                                                 uint32_t*                     h_rbSnrBuffOffsets,
                                                 cuphySrsReport_t*             d_pSrsReports,
                                                 cuphySrsChEstBuffInfo_t*      h_chEstBuffInfo,
                                                 void**                        d_addrsChEstToL2InnerBuff,
                                                 void**                        d_addrsChEstToL2Buff,
                                                 cuphySrsChEstToL2_t*          h_chEstToL2,
                                                 void*                         d_workspace,
                                                 uint8_t                       enableCpuToGpuDescrAsyncCpy,
                                                 void*                         pCpuDynDesc,
                                                 void*                         pGpuDynDesc,
                                                 cuphySrsChEstLaunchCfg_t*     pLaunchCfg,
                                                 cuphySrsChEstNormalizationLaunchCfg_t* pNormalizationLaunchCfg,
                                                 cudaStream_t                  strm);

/******************************************************************
 * \brief Destroys a cuPHY SRS channel estimation object
 *
 * Destroys a cuPHY SRS channel estimation object that was previously
 * created by ::cuphyCreateSrsChEst. The handle provided to this function
 * should not be used for any operations after this function returns.
 *
 * Returns ::CUPHY_STATUS_INVALID_ARGUMENT if \p srsChEstHndl is NULL.
 *
 * Returns ::CUPHY_STATUS_SUCCESS if destruction was successful.
 *
 * \param srsChEstHndl - handle to previously allocated SrsChEst instance
 *
 * \return
 * ::CUPHY_STATUS_SUCCESS,
 * ::CUPHY_STATUS_INVALID_ARGUMENT
 *
 * \sa ::cuphyStatus_t,::cuphyGetErrorName,::cuphyGetErrorString,::cuphySrsChEstGetDescrInfo,::cuphyCreateSrsChEst,::cuphySetupSrsChEst
 */
cuphyStatus_t CUPHYWINAPI cuphyDestroySrsChEst(cuphySrsChEstHndl_t srsChEstHndl);


/** @} */ /* END CUPHY_SRS_CHEST */

/**
 * \defgroup CUPHY_BEAMFORMING_WEIGHT_COMPUTE Beamforming Weight computation
 *
 * This section describes beamforming coefficient computation functions of the cuPHY
 * application programming interface.
 *
 * @{
 */

typedef struct _cuphyBfwLayerPrm
{
    uint16_t         chEstInfoBufIdx; // index into input SRS channel estimation information buffer of UE layer
    uint8_t          ueLayerIndex;    // specific layer of the UE used to index into SRS channel estimate bank in GPU memory
} cuphyBfwLayerPrm_t;

typedef struct _cuphyBfwUeGrpPrm
{
    uint16_t startPrbGrp; // start frequency index. NOTE: TO BE DEPREICATED

    uint16_t            nRxAnt;      // number of gNB receiving antennas (todo: rename nRxAnt to nGnbAnt)
    uint16_t            nRxAntSrs;   // number of receiving antenna ports for SRS. SRS UL/DL beamforming also assumes this many gNB ports
    uint8_t             nBfLayers;   // number of layers being beamformed
    cuphyBfwLayerPrm_t* pBfLayerPrm; // pointer to an array of length nLayers containing per layer information
    uint16_t            coefBufIdx;  // index into output beamforming coefficient tensor for UE group

    uint16_t startPrb;      // start prb for ueGrp freuquency allocation
    uint16_t bfwPrbGrpSize; // beamforming prbGrpSize. That is: beamforming weights are constant on PRB blocks of size prbGrpSize.
    uint16_t nPrbGrp;       // number of beamforming PRB groups in ueGrp frequency allocation.

    int16_t beamIdOffset; //starting offset for dynamic beam IDs

} cuphyBfwUeGrpPrm_t;

// @todo: Pending refactoring
cuphyStatus_t CUPHYWINAPI
cuphyBfcCoefCompute(unsigned int            nBSAnts,
                    unsigned int            nLayers,
                    unsigned int            Nprb,
                    cuphyTensorDescriptor_t tDescH,
                    const void*             HAddr,
                    cuphyTensorDescriptor_t tDescLambda,
                    const void*             lambdaAddr,
                    cuphyTensorDescriptor_t tDescCoef,
                    void*                   coefAddr,
                    cuphyTensorDescriptor_t tDescDbg,
                    void*                   dbgAddr,
                    cudaStream_t            strm);

struct cuphyBfwCoefComp;
/**
 * cuPHY Beamforming weight compute handle
 */
typedef struct cuphyBfwCoefComp* cuphyBfwCoefCompHndl_t;

/**
 * cuPHY Beamforming weight compute launch configuration
 */
typedef struct
{
    CUDA_KERNEL_NODE_PARAMS kernelNodeParamsDriver;
    void*                   kernelArgs[2];
} cuphyBfwCoefCompLaunchCfg_t;

typedef struct
{
    uint32_t                    nCfgs;
    cuphyBfwCoefCompLaunchCfg_t cfgs[CUPHY_BFW_COEF_COMP_N_MAX_HET_CFGS];
} cuphyBfwCoefCompLaunchCfgs_t;

/******************************************************************/ /**
 * \brief Helper to compute cuPHY beamforming coefficient compute descriptor buffer sizes and alignments
 *
 * Computes cuPHY beamforming coefficient compute descriptor buffer sizes and alignments. To be used by the caller
 * to allocate these buffers (in CPU and GPU memories) and provide them to other BfwCoefComp APIs
 *
 * Returns ::CUPHY_STATUS_INVALID_ARGUMENT if \p pStatDescrSizeBytes and/or \p pStatDescrAlignBytes and/or
 * \p pDynDescrSizeBytes and/or \p pDynDescrAlignBytes and/or \p pHetCfgUeGrpMapSizeBytes and/or \p pHetCfgUeGrpMapAlignBytes
 * and/or \p pUeGrpPrmsSizeBytes and/or \p pUeGrpPrmsAlignBytes and/or \p pBfLayerPrmsSizeBytes and/or \p pBfLayerPrmsAlignBytes
 * is NULL
 *
 * Returns ::CUPHY_STATUS_SUCCESS otherwise
 *
 * \param nMaxUeGrps                 - Max total number of UE groups to be processed in a single API invocation
 * \param nMaxTotalLayers            - Maximum total beamformed layers (i.e. sum of layer count across all UE groups)
 *                                     to be processed in a single API invocation
 * \param pStatDescrSizeBytes        - Size in bytes of beamforming coefficient compute static descriptor
 * \param pStatDescrAlignBytes       - Alignment in bytes of beamforming coefficient compute static descriptor
 * \param pDynDescrSizeBytes         - Size in bytes of beamforming coefficient compute dynamic descriptor
 * \param pDynDescrAlignBytes        - Alignment in bytes of beamforming coefficient compute dynamic descriptor
 * \param pHetCfgUeGrpMapSizeBytes   - Size in bytes of hetergenous config to UE group map descriptor
 * \param pHetCfgUeGrpMapAlignBytes  - Alignment in bytes of hetergenous config to UE group map descriptor
 * \param pUeGrpPrmsSizeBytes        - Size in bytes of UE group parameter descriptor
 * \param pUeGrpPrmsAlignBytes       - Alignment in bytes of UE group parameter descriptor
 * \param pBfLayerPrmsSizeBytes      - Size in bytes of beamforming layer descriptor
 * \param pBfLayerPrmsAlignBytes     - Alignment in bytes of beamforming layer descriptor
 *
 * \return
 * ::CUPHY_STATUS_SUCCESS,
 * ::CUPHY_STATUS_INVALID_ARGUMENT
 *
 * \sa ::cuphyStatus_t,::cuphyGetErrorName,::cuphyGetErrorString,::cuphyCreateBfwCoefComp,::cuphySetupBfwCoefComp,::cuphyDestroyBfwCoefComp
 */
cuphyStatus_t CUPHYWINAPI cuphyGetDescrInfoBfwCoefComp(uint16_t nMaxUeGrps,
                                                       uint16_t nMaxTotalLayers,
                                                       size_t*  pStatDescrSizeBytes,
                                                       size_t*  pStatDescrAlignBytes,
                                                       size_t*  pDynDescrSizeBytes,
                                                       size_t*  pDynDescrAlignBytes,
                                                       size_t*  pHetCfgUeGrpMapSizeBytes,
                                                       size_t*  pHetCfgUeGrpMapAlignBytes,
                                                       size_t*  pUeGrpPrmsSizeBytes,
                                                       size_t*  pUeGrpPrmsAlignBytes,
                                                       size_t*  pBfLayerPrmsSizeBytes,
                                                       size_t*  pBfLayerPrmsAlignBytes);

/******************************************************************/ /**
 * \brief Allocate and initialize a cuPHY beamforming coefficient compute object
 *
 * Allocates a cuPHY beamforming coefficient compute object and returns a handle in the
 * address provided by the caller.
 *
 * Returns ::CUPHY_STATUS_INVALID_ARGUMENT if \p pStatDescrCpu and/or \p pStatDescrGpu and/or
 * \p pDynDescrsCpu and/or \p pDynDescrsGpu and/or \p pHetCfgUeGrpMapCpu and/or \p pHetCfgUeGrpMapGpu
 * and/or \p pUeGrpPrmsCpu and/or \p pUeGrpPrmsGpu and/or \p pBfLayerPrmsCpu and/or \p pBfLayerPrmsGpu
 * is NULL
 *
 * Returns ::CUPHY_STATUS_SUCCESS if allocation and initialization were successful.
 *
 * Returns ::CUPHY_STATUS_ALLOC_FAILED if a BfwCoefComp object cannot be allocated
 *
 * Returns ::CUPHY_STATUS_SUCCESS if allocation and initialization were successful
 *
 * \param pBfwCoefCompHndl            - Address to return the new  instance
 * \param enableCpuToGpuDescrAsyncCpy - flag if non-zero enables async copy of CPU descriptor into GPU
 * \param compressBitwidth            - Number of bits to use for block floating point compression of weights (0 is uncompressed)
 * \param nMaxUeGrps                  - Max total number of UE groups to be processed in a single API invocation
 * \param nMaxTotalLayers             - Maximum total beamformed layers (i.e. sum of layer count across all UE groups)
 *                                      to be processed in a single API invocation
 * \param beta                        - prescalar for BFP compression of BFW values
 * \param lambda                      - regularization constant
 * \param bfwPowerNormAlg_selector    - selector of power normalization algorithm
 * \param enableBatchedMemcpy         - flag to enable batched H2D memcpy in bfwCoefComp::setupCoefComp
 * \param pStatDescrCpu               - Pointer to static descriptor in CPU memory
 * \param pStatDescrGpu               - Pointer to static descriptor in GPU memory
 * \param pDynDescrsCpu               - Pointer to dynamic descriptors in CPU memory
 * \param pDynDescrsGpu               - Pointer to dynamic descriptor in GPU memory
 * \param pHetCfgUeGrpMapCpu          - Pointer to heterogenous config to UE group map descriptor in CPU memory
 * \param pHetCfgUeGrpMapGpu          - Pointer to heterogenous config to UE group map descriptor in GPU memory
 * \param pUeGrpPrmsCpu               - Pointer to UE group parameter descriptor in CPU memory
 * \param pUeGrpPrmsGpu               - Pointer to UE group parameter descriptor in GPU memory
 * \param pBfLayerPrmsCpu             - Pointer to beamforming layer parameter descriptor in CPU memory
 * \param pBfLayerPrmsGpu             - Pointer to beamforming layer parameter descriptor in GPU memory
 * \param strm                        - CUDA stream for descriptor copy operation
 *
 * \return
 * ::CUPHY_STATUS_SUCCESS,
 * ::CUPHY_STATUS_ALLOC_FAILED,
 * ::CUPHY_STATUS_INVALID_ARGUMENT
 *
 * \sa ::cuphyStatus_t,::cuphyGetErrorName,::cuphyGetErrorString,::cuphyGetDescrInfoBfwCoefComp,::cuphySetupBfwCoefComp,::cuphyDestroyBfwCoefComp
 */
cuphyStatus_t CUPHYWINAPI cuphyCreateBfwCoefComp(cuphyBfwCoefCompHndl_t* pBfwCoefCompHndl,
                                                 uint8_t                 enableCpuToGpuDescrAsyncCpy,
                                                 uint8_t                 compressBitwidth,
                                                 uint16_t                nMaxUeGrps,
                                                 uint16_t                nMaxTotalLayers,
                                                 float                   beta,
                                                 float                   lambda,
                                                 uint8_t                 bfwPowerNormAlg_selector,
                                                 uint8_t                 enableBatchedMemcpy,
                                                 void*                   pStatDescrCpu,
                                                 void*                   pStatDescrGpu,
                                                 void*                   pDynDescrsCpu,
                                                 void*                   pDynDescrsGpu,
                                                 void*                   pHetCfgUeGrpMapCpu,
                                                 void*                   pHetCfgUeGrpMapGpu,
                                                 void*                   pUeGrpPrmsCpu,
                                                 void*                   pUeGrpPrmsGpu,
                                                 void*                   pBfLayerPrmsCpu,
                                                 void*                   pBfLayerPrmsGpu,
                                                 cudaStream_t            strm);

/******************************************************************/ /**
 * \brief Destroys a cuPHY beamforming coefficient compute object
 *
 * Destroys a cuPHY Pbeamforming coefficient compute object that was previously
 * created by ::cuphyCreateBfwCoefComp. The handle provided to this function
 * should not be used for any operations after this function returns.
 *
 * Returns ::CUPHY_STATUS_INVALID_ARGUMENT if \p puschRxChEstHndl is NULL.
 *
 * Returns ::CUPHY_STATUS_SUCCESS if destruction was successful.
 *
 * \param bfwCoefCompHndl - handle to previously allocated BfwCoefComp instance
 *
 * \return
 * ::CUPHY_STATUS_SUCCESS,
 * ::CUPHY_STATUS_INVALID_ARGUMENT
 *
 * \sa ::cuphyStatus_t,::cuphyGetErrorName,::cuphyGetErrorString,::cuphyGetDescrInfoBfwCoefComp,::cuphyCreateBfwCoefComp,::cuphySetupBfwCoefComp
 */
cuphyStatus_t CUPHYWINAPI cuphyDestroyBfwCoefComp(cuphyBfwCoefCompHndl_t bfwCoefCompHndl);

/******************************************************************/ /**
 * \brief Setup cuPHY beamforming coefficient compute object for calculation
 *
 * Setup cuPHY beamforming coefficient compute object in preparation towards execution for generating coefficients
 *
 * Returns ::CUPHY_STATUS_SUCCESS if setup is successful.
 *
 * Returns ::CUPHY_STATUS_INVALID_ARGUMENT if \p bfwCoefCompHndl and/or \p pUeGrpPrms
 * and/or \p pChEstBufInfo and/or \p pBfwCompCoef and/or \p pLaunchCfgs is NULL.
 *
 * \param bfwCoefCompHndl             - Handle to previously created BfwCoefComp instance
 * \param nUeGrps                     - total number of UE groups to be processed
 * \param pUeGrpPrms                  - Pointer to array of UE group parameters
 * \param enableCpuToGpuDescrAsyncCpy - Flag when set enables async copy of CPU descriptor into GPU
 * \param pChEstBufInfo               - Pointer to array of SRS channel estimation information buffers
 * \param pBfwCompCoef                - Pointer to array of compressed beamforming weights
 * \param pLaunchCfgs                 - Pointer to beamforming coefficient compute launch configurations
 * \param strm                        - CUDA stream for descriptor copy operation
 *
 * \return
 * ::CUPHY_STATUS_SUCCESS,
 * ::CUPHY_STATUS_INVALID_ARGUMENT
 *
 * \sa ::cuphyStatus_t,::cuphyGetErrorName,::cuphyGetErrorString,::cuphyGetDescrInfoBfwCoefComp,::cuphyCreateBfwCoefComp,::cuphyDestroyBfwCoefComp
 */
cuphyStatus_t CUPHYWINAPI cuphySetupBfwCoefComp(cuphyBfwCoefCompHndl_t        bfwCoefCompHndl,
                                                uint16_t                      nUeGrps,
                                                cuphyBfwUeGrpPrm_t const*     pUeGrpPrms,
                                                uint8_t                       enableCpuToGpuDescrAsyncCpy,
                                                cuphySrsChEstBuffInfo_t*      pChEstBufInfo,
                                                uint8_t**                     pBfwCompCoef,
                                                cuphyBfwCoefCompLaunchCfgs_t* pLaunchCfgs,
                                                cudaStream_t                  strm);

/** @} */ /* END CUPHY_BEAMFORMING_WEIGHT_COMPUTE */

/**
 * Launches a device<->host kernel-based copy
 *
 * Enqueues an internal CUDA copy kernel on the provided stream to transfer data
 * between a device buffer and a pinned, device-accessible host buffer. This API
 * does not call cudaMemcpy and does not perform the copy itself; it only
 * launches the kernel. Completion follows the stream's semantics.
 *
 * @param[in] devicePtr   Device pointer (source or destination depending on direction)
 * @param[in] pinnedHostPtr Pinned host pointer (source or destination depending on direction)
 * @param[in] numBytes    Number of bytes to copy
 * @param[in] direction   Copy direction: cudaMemcpyDeviceToHost or cudaMemcpyHostToDevice
 * @param[in] stream      CUDA stream on which to launch the copy kernel
 *
 * @return CUPHY_STATUS_SUCCESS on successful kernel launch
 * @return CUPHY_STATUS_INTERNAL_ERROR on CUDA errors or unsupported direction
 */
 cuphyStatus_t cuphylaunchKernelCopy(void* devicePtr,
    void* pinnedHostPtr,
    std::size_t numBytes,
    cudaMemcpyKind direction,
    cudaStream_t stream);


#if defined(__cplusplus)
} /* extern "C" */
#endif /* defined(__cplusplus) */

#endif /* !defined(CUPHY_H_INCLUDED_) */

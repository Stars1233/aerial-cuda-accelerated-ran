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

#pragma once

#include "cuphy_api.h"
#include "ldpc/ldpc_params.hpp"
#include "nvlog.hpp"
#include "common_utils.hpp"

#define TAG_UTILS 931

// 38.211 Table 7.4.1.5.3-1. "row" in 3GPP is 1-based: from 1 to 18. Access the following CSIRS tables by index = row - 1.
static constexpr uint8_t csirsRowDataNumPorts[CUPHY_CSIRS_SYMBOL_LOCATION_TABLE_LENGTH] =
                                                   {1, 1,    /* rows 1, 2 */
                                                    2,       /* row 3 */
                                                    4, 4,    /* rows 4, 5 */
                                                    8, 8, 8, /* rows 6, 7, 8 */
                                                    12, 12,  /* rows 9, 10 */
                                                    16, 16,  /* rows 11, 12 */
                                                    24, 24, 24, /* rows 13, 14, 15 */
                                                    32, 32, 32}; /* rows 16, 17, 18 */
static constexpr bool csirsRowDataBothL0L1[CUPHY_CSIRS_SYMBOL_LOCATION_TABLE_LENGTH] =
    {false, false, false, false, false, false, // row1..6
     false, false, false, false, false, false, // row7..12
     true, true, false, true, true, false}; // row13..18
// "l0, l0+1" counted as 2 symbols; the pattern is the same when both l0 and l1 are present.
static constexpr uint8_t csirsRowDataNumSymPerLGroup[CUPHY_CSIRS_SYMBOL_LOCATION_TABLE_LENGTH] =
    {1, 1, 1, 1, 2, 1, // row1..6
     2, 1, 1, 1, 2, 1, // row7..12
     2, 1, 1, 2, 1, 1}; // row13..18
// note that l_prime column is always {0..num_l_prime - 1}.
// note that when num_l_prime > 1, num_extra_sym_per_l is always 0, but not vice versa.
static constexpr uint8_t csirsRowDataNumLPrime[CUPHY_CSIRS_SYMBOL_LOCATION_TABLE_LENGTH] =
     {1, 1, 1, 1, 1, 1, // row1..6
      1, 2, 1, 2, 1, 2, // row7..12
      1, 2, 4, 1, 2, 4}; // row13..18

/**
 * @brief Return base graph (BG) number based on code rate and transport block size.
 * @param[in] code_rate: code rate
 * @param[in] transport_block_size: Transport Block (TB) size in bits
 * @return base graph number
 */
inline int get_base_graph(float code_rate, uint32_t transport_block_size) {

    int base_graph_number = ((transport_block_size <= 292) || (code_rate <= 0.25f) ||
        ((transport_block_size <= 3824) && (code_rate <= 0.67f))) ? 2 : 1;
    return base_graph_number;
}

/**
 * @brief Return TB CRC size in bits based on transport block size.
 * @param[in] transport_block_size: Transport Block (TB) size in bits before TB-CRC attachment.
 * @return TB CRC size (24 or 16)
 */
inline uint32_t compute_TB_CRC(uint32_t transport_block_size) {
    return (transport_block_size > 3824) ? 24 : 16;
}


/**
 * @brief Compute TB size, in bits, and number of Code Blocks (CBs) for this TB.
 * @param[in] num_symbols: number of data symbols for this TB not inluding DMRS symbols
 * @param[in] num_prbs: number of allocated PRBs (Physical Resource Blocks) for this TB
 * @param[in] num_layers: number of layers this TB maps to
 * @param[in] code_rate: code rate
 * @param[in] Qm: modulation order
 * @param[in, out] num_CBs: numbers of CBs.
 * @param[in, out] tb_size: transport block size in bits.
 * @param[in] num_dmrs_cdmGrpsNoData1_symbols: number of DMRS symbols when dmrsCdmGrpsNoData == 1 for this TB; 0 otherwise
 */
inline void get_TB_size_and_num_CBs(int num_symbols, int num_prbs, int num_layers, float code_rate, uint32_t Qm, uint32_t & num_CBs, uint32_t & tb_size, int num_dmrs_cdmGrpsNoData1_symbols) {

     uint32_t TBS_table[93] = {24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128, 136, 144, 152, 160, 168, 176, 184, 192, 208, 224, 240, 256, 272, 288, 304, 320, 336, 352, 368, 384, 408, 432, 456, 480, 504, 528, 552, 576, 608, 640, 672, 704, 736, 768, 808, 848, 888, 928, 984, 1032, 1064, 1128, 1160, 1192, 1224, 1256, 1288, 1320, 1352, 1416, 1480, 1544, 1608, 1672, 1736, 1800, 1864, 1928, 2024, 2088, 2152, 2216, 2280, 2408, 2472, 2536, 2600, 2664, 2728, 2792, 2856, 2976, 3104, 3240, 3368, 3496, 3624, 3752, 3824};

    // Compute number of REs
    // If cdmGrpNoData is set to 1 for this TB, then these DMRS symbols contribute half the REs per PRB, as if they were half a data symbol.
    int Nre_prime = CUPHY_N_TONES_PER_PRB * num_symbols + (CUPHY_N_TONES_PER_PRB / 2) * num_dmrs_cdmGrpsNoData1_symbols;
    uint32_t Nre = std::min(156, Nre_prime) * num_prbs;

    // Compute number of info bits
    float Ninfo = Nre * code_rate * Qm * num_layers;
    uint32_t Ninfo_prime;
    tb_size = 0; // needed for the TBS_table case below

    if (Ninfo <= 3824) {
        // For "small" sizes, look up TBS in a table. First round the
        // number of information bits.
        uint32_t n  = std::max(3, int(floor(log2(Ninfo))) - 6);
        Ninfo_prime = std::max(24, int(pow(2, n) * floor(Ninfo / pow(2, n))));

        // Pick the smallest TB from TBS_table which is not smaller than Ninfo_prime
        for (int j = 0; ((j < 93) && (tb_size == 0)); j++) {
            if (TBS_table[j] >= Ninfo_prime) {
                tb_size = TBS_table[j];
            }
        }
        num_CBs = 1;
    } else {

        // For "large" sizes, compute TBS. First round the number of
        // information bits to a power of two.
        uint32_t n  = floor(log2(Ninfo - 24)) - 5;
        Ninfo_prime = std::max(3840, int(pow(2, n) * round((double(Ninfo - 24.0) / pow(2, n)))));

        // Next, compute the number of code words. For large code rates,
        // use base-graph 1. For small code rate use base-graph 2.
        if (code_rate <= 0.25) {
            num_CBs = ceil((Ninfo_prime + 24) * 1.0f / 3816.0);
        } else {
            num_CBs = (Ninfo_prime > 8424) ?  ceil((Ninfo_prime + 24) * 1.0f/ 8424.0) : 1;
        }
        tb_size = 8 * num_CBs * ceil((Ninfo_prime + 24) * 1.0f / (8.0 * num_CBs)) - 24;
    }

    //NVLOGC_FMT(TAG_UTILS, "maxLayer {}, maxQm {}, n_PRB_LBRM {}, Ninfo {:.2f} vs. TBsize {}", num_layers, Qm, num_prbs, Ninfo, tb_size);
}


/**
 * @brief Compute number of information nodes based on base graph and transport block size.
 * @param[in] transport_block_size: TB size in bits before TB-CRC attachment.
 * @param[in] base_graph: base graph number
 * @return number of information nodes (Kb)
 */
inline uint32_t get_Kb(uint32_t transport_block_size, uint32_t base_graph) {
    uint32_t transport_block_size_w_CRC = transport_block_size + compute_TB_CRC(transport_block_size);
    return cuphy::ldpc::derive_Kb(static_cast<int>(base_graph), transport_block_size_w_CRC);
}

/**
 * @brief Compute number of per-CB bits K_prime, which includes per-CB CRC bits but not filler bits,
 *        based on base graph and transport block size.
 * @param[in] transport_block_size: TB size in bits before TB-CRC attachment
 * @param[in] base_graph: base graph number
 * @param[in,out] num_CBs: number of code blocks (extra check if num_CBs != 0)
 * @return K_prime
 */
inline uint32_t get_K_prime(uint32_t transport_block_size, uint32_t base_graph, uint32_t& num_CBs) {
    uint32_t  K_cb = (base_graph == 1) ? 8448 : 3840;
    uint32_t transport_block_size_w_CRC = transport_block_size + compute_TB_CRC(transport_block_size);
    uint32_t B_prime = transport_block_size_w_CRC;
    uint32_t C = 1;

    if (transport_block_size_w_CRC > K_cb) { // The TB will be segmented into multiple CBs.
        uint32_t L = 24;
        C = ceil((transport_block_size_w_CRC * 1.0f) / (K_cb - L));
        B_prime = transport_block_size_w_CRC + C * L;
    }
    if (num_CBs == 0) {
        num_CBs = C;
    } else  {
        // extra check to ensure C is the same as num_CBs computed in get_TB_size_and_num_CBs.
        if (C != num_CBs) {
            throw std::runtime_error("Mismatch in number of CBs computation.");
        }
    }
    return (B_prime / C);
}

/**
 * @brief Return lifting size (Zc).
 * @param[in] B: transport block size in bits before TB-CRC attachment
 * @param[in] base_graph: base graph number
 * @param[in] K_prime: number of per-CB bits including CRC but not filler bits.
 * @return lifting size
 */
inline uint32_t get_lifting_size(uint32_t B, uint32_t base_graph, uint32_t K_prime) {
    uint32_t K_b = get_Kb(B, base_graph); // This function considers the TB-CRC attachment internally

    uint32_t Z[51] = {2, 4, 8, 16, 32, 64, 128, 256,
                      3, 6, 12, 24, 48, 96, 192, 384,
                      5, 10, 20, 40, 80, 160, 320,
                      7, 14, 28, 56, 112, 224,
                      9, 18, 36, 72, 144, 288,
                      11, 22, 44, 88, 176, 352,
                      13, 26, 52, 104, 208,
                      15, 30, 60, 120, 240};

    // Derive ZcArray (from derive_lifting.m)
    // find smallest Zc such that Zc*K_b >= K_prime:
    uint32_t Zc = 384; // max possible
    for (int j = 0; j < 51; j++) {
        uint32_t temp = Z[j] * K_b;
        if ((temp >= K_prime) && (Z[j] < Zc)) {
            Zc = Z[j];
        }
    }
    return Zc;
}

inline uint32_t get_per_UE_CSI_RS_CNT(uint16_t* xtf_re_map,
                                      int BWP_PRBs,
                                      int start_symbol,
                                      int num_symbols, /* this is total # symbols incl. DMRS, but it's OK because xtf_re_map will never skip any REs for DMRS symbols */
                                      int start_prb, int num_prbs)
{
    uint32_t CSI_RS_RE_count = 0;
    for (int symbol_cnt = 0; symbol_cnt < num_symbols; symbol_cnt++) {
        int symbol_id = start_symbol + symbol_cnt;
        for (int re_cnt = 0; re_cnt < num_prbs * CUPHY_N_TONES_PER_PRB; re_cnt++) {
            if (xtf_re_map[symbol_id * BWP_PRBs * CUPHY_N_TONES_PER_PRB + \
                            re_cnt + start_prb * CUPHY_N_TONES_PER_PRB] == 1) {
                CSI_RS_RE_count += 1;
                //NVLOGC_FMT(TAG_UTILS, "CSI-RS symbol {}, PRB {}, RE {}", symbol_id, re_cnt / 12 + start_prb, re_cnt % 12);
            }
        }
    }
    return CSI_RS_RE_count;
}

inline __host__ bool pdsch_dl_rm_mod_fast_path_enabled(const PdschPerTbParams& tb_params,
                                                       const int               num_ports,
                                                       const int               resource_alloc,
                                                       const bool              dmrs_carry_data,
                                                       const bool              su_mimo)
{
    bool ok = (tb_params.Qm == CUPHY_QAM_256) && (tb_params.Nl == 4) && (num_ports == 4);
    ok      = ok && su_mimo;
    ok      = ok && (tb_params.testModel == 0);
    ok      = ok && (resource_alloc == 1);
    ok      = ok && !dmrs_carry_data;
    ok      = ok && (tb_params.rv == 0);
    return ok;
}

/**
 * @brief Update PdschPerTbParams structs that track configuration information at per TB
 *        granularity from cell group dynamic and cell static parameters.
 * @param[in,out] tb_params_struct: pointer to a PdschPerTbParams configuration struct
 * @param[in] cell_grp_dyn_params: pointer to dynamic parameters of a single cell group
 * @param[in] stat_params: array of cuphyPdschStatPrms_t structs
 * @return CUPHY_STATUS_SUCCESS or CUPHY_STATUS_INVALID_ARGUMENT.
 */
inline cuphyStatus_t cuphySetTBParamsFromStructs(PdschPerTbParams * tb_params_struct,
                                                 cuphyPdschCellGrpDynPrm_t const * cell_grp_dyn_params,
                                                 const cuphyPdschStatPrms_t* stat_params) {
    int num_TBs = cell_grp_dyn_params->nCws; // or nUes?
    bool check_TB_size = (stat_params->pDbg->checkTbSize != 0); // recompute TB size, if possible

    for (int TB_id = 0; TB_id < num_TBs; TB_id++) {
        //Pointers to CW, UE, UE group and cell
        cuphyPdschCwPrm_t* cw = &cell_grp_dyn_params->pCwPrms[TB_id];
        cuphyPdschUePrm_t* ue = cw->pUePrm;
        cuphyPdschUeGrpPrm_t* ue_group = ue->pUeGrpPrm;
        cuphyPdschCellDynPrm_t* cell = ue_group->pCellPrm;

        tb_params_struct[TB_id].firstCodeBlockIndex = 0; //Is this always zero?
        tb_params_struct[TB_id].testModel = cell->testModel; // 0 or 1 possible values

        tb_params_struct[TB_id].rv = cw->rv; //uint8_t to int
        if (tb_params_struct[TB_id].rv > 3) {
            NVLOGE_FMT(TAG_UTILS, AERIAL_CUPHY_EVENT, "tb_pars rv {} has to be <= 3.", tb_params_struct[TB_id].rv);
            return CUPHY_STATUS_INVALID_ARGUMENT;
        }
        // Note we were previously using cellId instead of dataScramId. stat_params is only
        // useful in that case.
        tb_params_struct[TB_id].cinit = (ue->rnti << 15) + /*(q << 14)*/ + ue->dataScramId;
        //tb_params_struct[TB_id].cinit = (ue->rnti << 15) + /*(q << 14)*/ + stat_params->pCellStatPrms[cell->cellPrmStatIdx].phyCellId;

        // Modulation order is no longer derived using MCS table and index, but provided directly. Doing so allows us to support MCS index > 28 too.
        tb_params_struct[TB_id].Qm = cw->qamModOrder;

        if ((tb_params_struct[TB_id].Qm != CUPHY_QAM_4) && (tb_params_struct[TB_id].Qm != CUPHY_QAM_16) &&
            (tb_params_struct[TB_id].Qm != CUPHY_QAM_64) && (tb_params_struct[TB_id].Qm != CUPHY_QAM_256)) {
            NVLOGE_FMT(TAG_UTILS, AERIAL_CUPHY_EVENT, "tb_pars Qm {} is invalid!", tb_params_struct[TB_id].Qm);
            return CUPHY_STATUS_INVALID_ARGUMENT;
        }

        //TODO: In the future rework this so processing is top-down rather than bottom-up.
        uint8_t num_PDSCH_symbols = (cell->nPdschSym == 0) ? ue_group->nPdschSym : cell->nPdschSym;
        uint8_t pdsch_start_sym = (cell->nPdschSym == 0) ? ue_group->pdschStartSym : cell->pdschStartSym;
        uint16_t dmrs_bitmask = (cell->dmrsSymLocBmsk == 0) ? ue_group->dmrsSymLocBmsk : cell->dmrsSymLocBmsk;
        int num_dmrs_symbols =  __builtin_popcount(dmrs_bitmask); // Assumes that all set DMRS bits are after the start symbol.
        int first_set_DMRS_bit = __builtin_ctz(dmrs_bitmask); // Count number of trailing 0-bits starting from the least significant bit of the bitmask
        if (first_set_DMRS_bit <  pdsch_start_sym) {
            NVLOGE_FMT(TAG_UTILS, AERIAL_CUPHY_EVENT, "First set bit in DMRS bit mask {} is before PDSCH start symbol {}", first_set_DMRS_bit, pdsch_start_sym);
            return CUPHY_STATUS_INVALID_ARGUMENT;
        }
        uint32_t num_symbols = num_PDSCH_symbols -  num_dmrs_symbols;
        int num_dmrs_CdmGrpsNoData1_symbols = (ue_group->pDmrsDynPrm->nDmrsCdmGrpsNoData == 1) ? num_dmrs_symbols : 0;
        // DMRS symbols when CdmGrpsNoData is 1 also contribute to max_REs below but only half the REs in a PRB (i.e., 6).
        tb_params_struct[TB_id].max_REs = (num_symbols * CUPHY_N_TONES_PER_PRB  + num_dmrs_CdmGrpsNoData1_symbols * (CUPHY_N_TONES_PER_PRB / 2)) * ue_group->nPrb; // max_REs because it doesn't include the punctured ones
        //tb_params_struct[TB_id].G = (num_symbols * 12 * ue_group->nPrb - csi_rs_RE_cnt) * tb_params_struct[TB_id].Qm * ue->nUeLayers; // possibly not known here
        tb_params_struct[TB_id].G = tb_params_struct[TB_id].max_REs * tb_params_struct[TB_id].Qm * ue->nUeLayers; // in reality max_G

        // code rate is not longer derived from MCS table and index, but computed from the provided target code rate. Doing so allows us to support MCS index > 28 too.
        if (cw->targetCodeRate == 0) {
            NVLOGE_FMT(TAG_UTILS, AERIAL_CUPHY_EVENT, "targetCodeRate for TB {} cannot be 0.", TB_id);
            return CUPHY_STATUS_INVALID_ARGUMENT;
        }
        float code_rate = cw->targetCodeRate / (1024.0f * 10);
        //return CUPHY_STATUS_INVALID_ARGUMENT; // Uncomment if you want to test proof of concept error handling patch

        uint32_t transport_block_size = 0;
        uint32_t num_CBs = 0;

        // Reminder: For MCS table 0 and 2, MCS indices <= 28 are OK, while for MCS table 1 it's indices < 28.
        bool recompute_and_check_TB_size = check_TB_size && ((cw->mcsIndex < 28) || ((cw->mcsIndex == 28) && cw->mcsTableIndex != 1));

        uint32_t K_prime = 0;
        if (recompute_and_check_TB_size) {
            get_TB_size_and_num_CBs(num_symbols, ue_group->nPrb, ue->nUeLayers, code_rate, tb_params_struct[TB_id].Qm, num_CBs, transport_block_size, num_dmrs_CdmGrpsNoData1_symbols);
            tb_params_struct[TB_id].bg = get_base_graph(code_rate, transport_block_size);
            K_prime = get_K_prime(transport_block_size, tb_params_struct[TB_id].bg, num_CBs); // num_CBs will not be 0, so num_CBs will not be updated
        } else {
            // Exceptionally skip TB size computation as the provided target code rate config. may not be valid for the TB size computation (in case of a retransmission)
            // or if the checkTbSize static parameters is set to 0.
            transport_block_size = cw->tbSize * 8; //tbSize is in bytes
            tb_params_struct[TB_id].bg = get_base_graph(code_rate, transport_block_size); //assumes target code rate is valid for base graph computation
            K_prime = get_K_prime(transport_block_size, tb_params_struct[TB_id].bg, num_CBs); // num_CBs will be 0, so its value will be updated
        }
        tb_params_struct[TB_id].num_CBs = num_CBs;

        uint32_t Zc = get_lifting_size(transport_block_size, tb_params_struct[TB_id].bg, K_prime);

        tb_params_struct[TB_id].Zc = Zc;

        tb_params_struct[TB_id].N = (tb_params_struct[TB_id].bg == 1) ? CUPHY_LDPC_MAX_BG1_UNPUNCTURED_VAR_NODES * Zc : CUPHY_LDPC_MAX_BG2_UNPUNCTURED_VAR_NODES * Zc;

        // Compute LBRM (limited block rate-matching) transport block size and Nref.
        uint32_t LBRM_transport_block_size = 0, LBRM_num_CBs = 0;
        float max_code_rate = 948.0f / 1024;

        if ((cw->maxLayers < 1) || (cw->maxLayers > 4)) {
            NVLOGE_FMT(TAG_UTILS, AERIAL_CUPHY_EVENT, "maxLayers {} is invalid! Should be in [1, 4].", cw->maxLayers);
            return CUPHY_STATUS_INVALID_ARGUMENT;
        }
        if ((cw->maxQm != CUPHY_QAM_64) && (cw->maxQm != CUPHY_QAM_256)){
            NVLOGE_FMT(TAG_UTILS, AERIAL_CUPHY_EVENT, "max Qm {} is invalid! Should be 6 or 8.", cw->maxQm);
            return CUPHY_STATUS_INVALID_ARGUMENT;
        }

        if ((cw->n_PRB_LBRM < 32) || (cw->n_PRB_LBRM > 273)) {
            // Only specific values in that range are permitted. Checking only for range here to reduce overhead.
            NVLOGE_FMT(TAG_UTILS, AERIAL_CUPHY_EVENT, "n_PRB_LBRM {} is in invalid range! Should be within [32, 273].", cw->n_PRB_LBRM);
            return CUPHY_STATUS_INVALID_ARGUMENT;
        }

        get_TB_size_and_num_CBs(13 /* num_symbols. Force Nre = 12 * num_symbols to 156*/,
                                cw->n_PRB_LBRM,
                                cw->maxLayers, max_code_rate, cw->maxQm, LBRM_num_CBs, LBRM_transport_block_size, 0);
        // Nref computation below is equivalent to floor(LBRM_transport_block_size/(num_CBs * LBRM_code_rate)) with LBRM_code_rate = 2.0f/3


        uint32_t Nref = LBRM_transport_block_size * 3/(num_CBs * 2);
        tb_params_struct[TB_id].Ncb = std::min(tb_params_struct[TB_id].N, Nref);

        tb_params_struct[TB_id].K = (tb_params_struct[TB_id].bg == 1) ? CUPHY_LDPC_BG1_INFO_NODES * Zc : CUPHY_LDPC_MAX_BG2_INFO_NODES * Zc;
        tb_params_struct[TB_id].F = tb_params_struct[TB_id].K - K_prime; // number of filler bits
        //Set number of layers. Layer map updated in cuphyUpdatePdschDmsrsParams()
        tb_params_struct[TB_id].Nl = ue->nUeLayers;
        if ((tb_params_struct[TB_id].Nl < 1) || (tb_params_struct[TB_id].Nl > MAX_DL_LAYERS_PER_TB)) {
            NVLOGE_FMT(TAG_UTILS, AERIAL_CUPHY_EVENT, "tb_pars Nl {} has to be in [1, {}].", tb_params_struct[TB_id].Nl, MAX_DL_LAYERS_PER_TB);
            return CUPHY_STATUS_INVALID_ARGUMENT;
        }

        // pdsch tx RM fast_path: host sets flag, kernel reads it
        auto&      tb_params       = tb_params_struct[TB_id];
        const int  num_ports       = (ue->enablePrcdBf != 0) ? (int)cell_grp_dyn_params->pPmwPrms[ue->pmwPrmIdx].nPorts : 0;
        const bool dmrs_carry_data = (ue_group->pDmrsDynPrm->nDmrsCdmGrpsNoData == 1);
        const bool su_mimo         = (ue_group->nUes == 1);
        tb_params.fast_path = pdsch_dl_rm_mod_fast_path_enabled(tb_params,
                                                                num_ports,
                                                                (int)ue_group->resourceAlloc,
                                                                dmrs_carry_data,
                                                                su_mimo);

        // If this TB is for a cell in testing mode, then overwrite N, num_CBs fields
        // so we still respect the MAX_ENCODED_CODE_BLOCK_BIT_SIZE constraint.
        if (tb_params_struct[TB_id].testModel != 0) {
            int tb_size_in_bits = cw->tbSize*8; // PN23 sequence bits stored in TB payload buffer

            // Sanity check that confirms tbSize is what is expected for TM mode. max_REs accounts for REs in presence of CMD_GRM_NO_DATA=1 too
            // No punctured REs are considered since they will not be present for TM cells.
            // Because cw->tbSize is in bytes, the check should happen at byte granularity to avoid false mismatches when computed TB size is not evently divisible by 8.
            int expected_tb_size_in_bits = tb_params_struct[TB_id].max_REs * tb_params_struct[TB_id].Qm * tb_params_struct[TB_id].Nl;
            int expected_tb_size_in_bytes = ((expected_tb_size_in_bits + 7) / 8); //round up to next number of bytes
            if (check_TB_size && (cw->tbSize != expected_tb_size_in_bytes)) {
                NVLOGE_FMT(TAG_UTILS, AERIAL_CUPHY_EVENT, "TB {} in TM has TB size in bits {} ({} B) but expected is {} ({} B)",
                           TB_id, tb_size_in_bits, cw->tbSize, expected_tb_size_in_bits, expected_tb_size_in_bytes);
                return CUPHY_STATUS_INVALID_ARGUMENT;
            }

            if (expected_tb_size_in_bits <= MAX_ENCODED_CODE_BLOCK_BIT_SIZE) { // <= 25344
                tb_params_struct[TB_id].num_CBs = 1;
                tb_params_struct[TB_id].N = expected_tb_size_in_bits;
            } else {
                tb_params_struct[TB_id].num_CBs = (expected_tb_size_in_bits + MAX_ENCODED_CODE_BLOCK_BIT_SIZE -1) / MAX_ENCODED_CODE_BLOCK_BIT_SIZE;
                tb_params_struct[TB_id].N = MAX_ENCODED_CODE_BLOCK_BIT_SIZE; //the last CB would have fewer bits; this should be considered in the rm block
            }
            /*int last_CB_N = ((tb_params_struct[TB_id].testModel != 0) && (tb_params_struct[TB_id].num_CBs != 1)) ? \
                            (expected_tb_size_in_bits % tb_params_struct[TB_id].N) : tb_params_struct[TB_id].N;
            NVLOGC_FMT(TAG_UTILS, "TB {} (whose cell is in TM) with tb_size_in_bits {}: updated num_CBs {} and N {}. Last CB is {}",
                   TB_id, expected_tb_size_in_bits, tb_params_struct[TB_id].num_CBs, tb_params_struct[TB_id].N, last_CB_N);*/
        }
    }
    return CUPHY_STATUS_SUCCESS;
}

inline void init_CSIRS_tables(CsirsTables* h_csirs_tables) {

    // Initialize CSI-RS tables
    // Fields: numPorts, lenKBarLBar lenKPrime lenLPrime kIndices kOffsets lIndices lOffsets cdmGroupIndex

    if (h_csirs_tables == nullptr) {
        throw std::runtime_error("CSIRS tables pointer is nullptr!");
    }
    h_csirs_tables->rowData[0] = {1, 3, 1, 1,
                                  {0, 0, 0},
                                  {0, 4, 8},
                                  {0, 0, 0},
                                  {0, 0, 0},
                                  {0, 0, 0}};
    h_csirs_tables->rowData[1] = {1, 1, 1, 1,
                                  {0},
                                  {0},
                                  {0},
                                  {0},
                                  {0}};
    h_csirs_tables->rowData[2] = {2, 1, 2, 1,
                                  {0},
                                  {0},
                                  {0},
                                  {0},
                                  {0}};
    h_csirs_tables->rowData[3] = {4, 2, 2, 1,
                                  {0, 0},
                                  {0, 2},
                                  {0, 0},
                                  {0, 0},
                                  {0, 1}};
    h_csirs_tables->rowData[4] = {4, 2, 2, 1,
                                  {0, 0},
                                  {0, 0},
                                  {0, 0},
                                  {0, 1},
                                  {0, 1}};
    h_csirs_tables->rowData[5] = {8, 4, 2, 1,
                                  {0, 1, 2, 3},
                                  {0, 0, 0, 0},
                                  {0, 0, 0, 0},
                                  {0, 0, 0, 0},
                                  {0, 1, 2, 3}};
    h_csirs_tables->rowData[6] = {8, 4, 2, 1,
                                  {0, 1, 0, 1},
                                  {0, 0, 0, 0},
                                  {0, 0, 0, 0},
                                  {0, 0, 1, 1},
                                  {0, 1, 2, 3}};
    h_csirs_tables->rowData[7] = {8, 2, 2, 2,
                                  {0, 1},
                                  {0, 0},
                                  {0, 0},
                                  {0, 0},
                                  {0, 1}};
    h_csirs_tables->rowData[8] = {12, 6, 2, 1,
                                  {0, 1, 2, 3, 4, 5},
                                  {0, 0, 0, 0, 0},
                                  {0, 0, 0, 0, 0},
                                  {0, 0, 0, 0, 0},
                                  {0, 1, 2, 3, 4, 5}};
    h_csirs_tables->rowData[9] = {12, 3, 2, 2,
                                  {0, 1, 2},
                                  {0, 0, 0},
                                  {0, 0, 0},
                                  {0, 0, 0},
                                  {0, 1, 2}};
    h_csirs_tables->rowData[10] = {16, 8, 2, 1,
                                   {0, 1, 2, 3, 0, 1, 2, 3},
                                   {0, 0, 0, 0, 0, 0, 0, 0},
                                   {0, 0, 0, 0, 0, 0, 0, 0},
                                   {0, 0, 0, 0, 1, 1, 1, 1},
                                   {0, 1, 2, 3, 4, 5, 6, 7}};
    h_csirs_tables->rowData[11] = {16, 4, 2, 2,
                                   {0, 1, 2, 3},
                                   {0, 0, 0, 0},
                                   {0, 0, 0, 0},
                                   {0, 0, 0, 0},
                                   {0, 1, 2, 3}};
    h_csirs_tables->rowData[12] = {24, 12, 2, 1,
                                   {0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2},
                                   {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                                   {0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1},
                                   {0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1},
                                   {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}};
    h_csirs_tables->rowData[13] = {24, 6, 2, 2,
                                   {0, 1, 2, 0, 1, 2},
                                   {0, 0, 0, 0, 0, 0},
                                   {0, 0, 0, 1, 1, 1},
                                   {0, 0, 0, 0, 0, 0},
                                   {0, 1, 2, 3, 4, 5}};
    h_csirs_tables->rowData[14] = {24, 3, 2, 4,
                                   {0, 1, 2},
                                   {0, 0, 0},
                                   {0, 0, 0},
                                   {0, 0, 0},
                                   {0, 1, 2}};
    h_csirs_tables->rowData[15] = {32, 16, 2, 1,
                                   {0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3},
                                   {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                                   {0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1},
                                   {0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1},
                                   {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}};
    h_csirs_tables->rowData[16] = {32, 8, 2, 2,
                                   {0, 1, 2, 3, 0, 1, 2, 3},
                                   {0, 0, 0, 0, 0, 0, 0, 0},
                                   {0, 0, 0, 0, 1, 1, 1, 1},
                                   {0, 0, 0, 0, 0, 0, 0, 0},
                                   {0, 1, 2, 3, 4, 5, 6, 7}};
    h_csirs_tables->rowData[17] = {32, 4, 2, 4,
                                   {0, 1, 2, 3},
                                   {0, 0, 0, 0},
                                   {0, 0, 0, 0},
                                   {0, 0, 0, 0},
                                   {0, 1, 2, 3}};
}

/**
 * @brief Structure containing CUDA device architecture information and library support flags.
 */
struct cuphyDeviceArchInfo {
    unsigned int computeCapability;  ///< Compute capability (e.g., 800 for 8.0, 750 for 7.5)
    bool cuFFTDxSupported;          ///< True if cuFFTDx is supported on this architecture
    bool tensorCoreSupported;       ///< True if Tensor Cores are supported
    bool cuPHYSupported;            ///< True if cuPHY is supported on this architecture
    int deviceId;                   ///< CUDA device ID
};

static cuphyDeviceArchInfo get_cuda_device_arch() {
    int device;
    CUDA_CHECK(cudaGetDevice(&device));

    int major = 0;
    int minor = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device));
    CUDA_CHECK(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device));

    cuphyDeviceArchInfo archInfo;
    archInfo.computeCapability = static_cast<unsigned>(major) * 100 + static_cast<unsigned>(minor) * 10;
    archInfo.deviceId = device;
    
    // Determine library support based on compute capability
    // cuFFTDx requires compute capability 7.0 or higher
    archInfo.cuFFTDxSupported = (major >= 7);
    
    // Tensor Cores are available on Volta (7.0) and newer architectures
    archInfo.tensorCoreSupported = (major >= 7);

    switch(major)
    {
        case 8:
        case 9:
        case 10:
            archInfo.cuPHYSupported = true;
            break;
        case 12:
            archInfo.cuPHYSupported = (minor == 1);
            break;
        default:
            archInfo.cuPHYSupported = false;
    }

    return archInfo;
}

inline __device__ float2 gen_pusch_dftsofdm_descrcode(
    uint16_t M_ZC, uint16_t rIdx, int u, int v, uint16_t nPrb,
    const int8_t (&d_phi_6)[30][6],
    const int8_t (&d_phi_12)[30][12],
    const int8_t (&d_phi_18)[30][18],
    const int8_t (&d_phi_24)[30][24],
    const uint16_t* d_primeNums)
{
    float2 descrCode = {0.0f, 0.0f};
    if (M_ZC < 36) 
    {
        if (rIdx < M_ZC)
        {
            switch (M_ZC)
            {
                case 6:
                    descrCode.x = (float)cos(M_PI * d_phi_6[u][rIdx] / 4.0f);
                    descrCode.y = (float)sin(M_PI * d_phi_6[u][rIdx] / 4.0f);
                    break;
                case 12:
                    descrCode.x = (float)cos(M_PI * d_phi_12[u][rIdx] / 4.0f);
                    descrCode.y = (float)sin(M_PI * d_phi_12[u][rIdx] / 4.0f);
                    break;
                case 18:
                    descrCode.x = (float)cos(M_PI * d_phi_18[u][rIdx] / 4.0f);
                    descrCode.y = (float)sin(M_PI * d_phi_18[u][rIdx] / 4.0f);
                    break;
                case 24:
                    descrCode.x = (float)cos(M_PI * d_phi_24[u][rIdx] / 4.0f);
                    descrCode.y = (float)sin(M_PI * d_phi_24[u][rIdx] / 4.0f);
                    break;
                case 30:
                    descrCode.x = (float)cos(M_PI * (u + 1) * (rIdx + 1) * (rIdx + 2) / 31.0f);
                    descrCode.y = (float)(-sin(M_PI * (u + 1) * (rIdx + 1) * (rIdx + 2) / 31.0f));
                    break;
                default:
                    // Per TS 38.211, section 5.2.2.2, only M_ZC multiples of 6 are defined for M_ZC < 36.  We will return 0 otherwise
                    break;
            }  
        }
    }
    else 
    {   
        uint16_t d_primeNum = d_primeNums[nPrb-1];
        float qbar = d_primeNum * (u + 1) / 31.0f;
        float q    = (int)(qbar + 0.5f) + (v * (((int)(2 * qbar) & 1) * -2 + 1));
        uint32_t m = rIdx % d_primeNum;
        descrCode.x = (float)cos(M_PI * q * m * (m + 1) / d_primeNum);
        descrCode.y = (float)(-sin(M_PI * q * m * (m + 1) / d_primeNum));
    }
    return descrCode;
}

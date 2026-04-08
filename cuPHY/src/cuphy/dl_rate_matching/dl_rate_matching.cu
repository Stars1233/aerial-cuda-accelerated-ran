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

#include "cuphy.h"
#include "cuphy.hpp"
#include "cuphy_internal.h"
#include <vector>
#include <iostream>

#include "dl_rate_matching.hpp"
#include "dl_rate_matching.cuh"
#include "descrambling.cuh"
#include "tensor_desc.hpp"

#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>

using namespace cuphy_i;
using namespace descrambling; // for gold32

#define FLOAT_COMP 1
#define DMRS_CDM_GRPS_NO_DATA_NOT_1_FLAG 0xFFFF //used in place of the 16-bit dmrs symbol location mask to indicate dmrsGrpsNoData != 1

__device__ __constant__ float rev_qam_16_long[8] = {
    0.316227766,
    -0.316227766,
    0.316227766,
    -0.316227766,
    0.948683298,
    -0.948683298,
    0.948683298,
    -0.94868329
};

/* Indexed as {bit 4, bit 2, bit 0}, thus the reverse in the name. */
__device__ __constant__ float rev_qam_64[8] = {
    0.462910049886276,
    -0.462910049886276,
    0.77151674981046,
    -0.77151674981046,
    0.154303349962092,
    -0.154303349962092,
    1.08012344973464,
    -1.08012344973464
};

__device__ __constant__ float rev_qam_256[16] = {
    0.383482494,
    -0.383482494,
    0.843661488,
    -0.843661488,
    0.230089497,
    -0.230089497,
    0.997054486,
    -0.997054486,
    0.536875492,
    -0.536875492,
    0.69026849,
    -0.69026849,
    0.076696499,
    -0.076696499,
    1.150447483,
    -1.150447483
};

__device__ __inline__ uint32_t map_index_8bits(uint32_t index) {
#if 0
    uint32_t masked_index = (index & 0x1) | ((index & 0x4) >> 1) |
                   ((index & 0x10) >> 2)  | ((index & 0x40) >> 3);
    return masked_index;
#else
    uint32_t masked_index = (index & 0x55U);
    uint32_t tmp = (masked_index | (masked_index >> 1));
    return (tmp & 0x3) | ((tmp & 0x30) >> 2);
#endif
}


__device__ __inline__ uint32_t map_index_6bits(uint32_t index) {
    uint32_t masked_index = (index & 0x1) | ((index & 0x4) >> 1) |
                   ((index & 0x10) >> 2);
    return masked_index;
}

__device__ __inline__ __half2 mult_float2(float2 a, float2 b) {
    return make_complex<__half2>::create((__half)(a.x*b.x - a.y*b.y),(__half)( a.x*b.y + a.y*b.x));
}

__device__ __inline__ __half2 mult_half2(__half2 a, __half2 b) {
#if 1
    return make_complex<__half2>::create(a.x*b.x - a.y*b.y, a.x*b.y + a.y*b.x);
#else
    return make_complex<__half2>::create(__hadd(__hmul(a.x, b.x), __hneg(__hmul(a.y, b.y))),
                                         __hadd(__hmul(a.x, b.y), __hmul(a.y, b.x)));
#endif
}

/**
 * @brief Update PdschPerTbParams struct that tracks configuration information at per TB
 *        granularity. Check that configuration values are valid.
 * @param[in] cfg_rv: redundancy version
 * @param[in] cfg_Qm: modulation order
 * @param[in] cfg_bg: base graph
 * @param[in] cfg_Nl: number of layers per Tb (at most MAX_DL_LAYERS_PER_TB for downlink)
 * @param[in] cfg_num_CBs: number of code blocks
 * @param[in] cfg_Zc: lifting factor
 * @param[in] cfg_G: number of rated matched bits available for TB transmission
 * @param[in] cfg_F: number of filler bits
 * @param[in] cfg_cinit: seed used for scrambling sequence
 * @param[in] cfg_Nref: used to determine Ncb if smaller than N.
 * @return CUPHY_STATUS_SUCCESS or CUPHY_STATUS_INVALID_ARGUMENT.
 */
cuphyStatus_t cuphySetTBParams(PdschPerTbParams * tb_params_struct, uint32_t cfg_rv, uint32_t cfg_Qm,
		               uint32_t cfg_bg, uint32_t cfg_Nl, uint32_t cfg_num_CBs,
		               uint32_t cfg_Zc, uint32_t cfg_G, uint32_t cfg_F, uint32_t cfg_cinit, uint32_t cfg_Nref) {


    if (cfg_rv > 3) {
        NVLOGE_FMT(NVLOG_PDSCH, AERIAL_CUPHY_EVENT, "{}: cfg_rv {} has to be <= 3", __FUNCTION__, cfg_rv);
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }

    if ((cfg_Qm != CUPHY_QAM_4) && (cfg_Qm != CUPHY_QAM_16) && (cfg_Qm != CUPHY_QAM_64) && (cfg_Qm != CUPHY_QAM_256)) {
        NVLOGE_FMT(NVLOG_PDSCH, AERIAL_CUPHY_EVENT, "{}: cfg_Qm {} is invalid", __FUNCTION__, cfg_Qm);
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }

    if ((cfg_bg != 1) && (cfg_bg != 2)) {
        NVLOGE_FMT(NVLOG_PDSCH, AERIAL_CUPHY_EVENT, "{}: cfg_bg {} can either be 1 or 2", __FUNCTION__, cfg_bg);
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }

    if ((cfg_Nl < 1) || (cfg_Nl > MAX_DL_LAYERS_PER_TB)) {
        NVLOGE_FMT(NVLOG_PDSCH, AERIAL_CUPHY_EVENT, "{}: cfg_Nl {} has to be in [1, {}]", __FUNCTION__, cfg_Nl, MAX_DL_LAYERS_PER_TB);
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }

    tb_params_struct->rv = cfg_rv;
    tb_params_struct->Qm = cfg_Qm;
    tb_params_struct->bg = cfg_bg;
    tb_params_struct->Nl = cfg_Nl;
    tb_params_struct->num_CBs = cfg_num_CBs;
    tb_params_struct->Zc = cfg_Zc;

    const uint32_t NUM_UNPUNCTURED_VAR_NODES = (cfg_bg == 1)                            ?
                                               CUPHY_LDPC_MAX_BG1_UNPUNCTURED_VAR_NODES :
                                               CUPHY_LDPC_MAX_BG2_UNPUNCTURED_VAR_NODES;

    tb_params_struct->N = NUM_UNPUNCTURED_VAR_NODES * cfg_Zc;
    tb_params_struct->Ncb = std::min(tb_params_struct->N, cfg_Nref);
    tb_params_struct->K = (cfg_bg == 1) ? CUPHY_LDPC_BG1_INFO_NODES * cfg_Zc : CUPHY_LDPC_MAX_BG2_INFO_NODES * cfg_Zc;

    tb_params_struct->G = cfg_G;
    tb_params_struct->F = cfg_F;
    tb_params_struct->cinit = cfg_cinit;

    return CUPHY_STATUS_SUCCESS;
}


/**
 * @brief Compute starting position, ko, for redundancy version rv as per
 *        Table 5.4.2.1-2 from the 3GPP reference.
 * @param[in] rv: redundancy version; [0, 3] valid range
 * @param[in] bg_num: base graph number; 1 or 2.
 * @param[in] Ncb: circular buffer length for LDPC encoder's output
 * @param[in] Zc: lifting size
 * @return starting position k0
 */
int compute_k0(int rv, int bg_num, int Ncb, int Zc) {
    if (rv == 0) return 0;
    int denom = (bg_num == 1) ? CUPHY_LDPC_MAX_BG1_UNPUNCTURED_VAR_NODES : CUPHY_LDPC_MAX_BG2_UNPUNCTURED_VAR_NODES;
    int enumerator[2][4] = {{0, 17, 33, 56}, {0, 13, 25, 43}};
    return floorf(enumerator[bg_num - 1][rv] * Ncb * 1.0f / (denom * Zc)) * Zc;
}


/**
 * @brief Compute rate matching sequence length, in bits, for all C code blocks in a TB.
 @param[in, out] Er[]: will hold 2 uint32_t elements per-TB. First is the id of the CB_split for Er (num_CBs if no split), and second is
                       the smallest supported Er. The Er for CBs from CB_split and beyond is Er + Nl * Qm.
 * @param[in] C: number of code blocks
 * @param[in] Qm: modulation order
 * @param[in] Nl: number of transmission layers
 * @param[in] G: number of coded bits available for TB's transmission.
 * @param[in, out] Emax: maximum Er across all TBs processed in a kernel; divisible by word_size
 * @param[in, out] updated_Emax: true if EMax was updated this call
 * @param[in] no_rate_matching: should be set if TB belongs to cell in TM; Er values differ then
 * @param[in] word_size: element size, in bits, for rate matching's input and output;
 *                       default = sizeof(uint32_t)*8 = 32
 */
void compute_rate_matching_length(uint32_t Er[], int C, int Qm, int Nl, int G, uint32_t & Emax, bool& updated_Emax, bool no_rate_matching, int word_size) {
    // Assumption: C, Qm, Nl, G are the same same across all CBs in a TB.
#if 0
    int j = 0;
    Er[0] = C;
    for (int code_block_id = 0; code_block_id < C; code_block_id++) {
        /*if (false) {
            //FIXME: Clarify the "r-th coded block not scheduled for transmission" case from the guide
            //Either set appropriate condition or remove if statement.
            Er[code_block_id] = 0;
        } else {*/
            uint32_t Er_val = 0;
            if (j <= C - ((G / (Nl * Qm)) % C) - 1) {
                Er_val =  Nl * Qm * floorf(G * 1.0f / (Nl * Qm * C));
            } else {
                Er_val = Nl * Qm * ceilf(G * 1.0f / (Nl * Qm * C));
            }
            if (code_block_id == 0) {
                Er[1] = Er_val;
            } else if ((Er[0] == C) && (Er_val != Er[1])) {
                Er[0] = code_block_id; //Update id for CB split
            }
	    //printf("Er[%d] = %d\n", code_block_id, Er[code_block_id]);
            j += 1;
            Emax = std::max(Emax, Er_val);
       //}
    }

    if (Emax % word_size != 0) {
        Emax = (Emax / word_size + 1) * word_size;
    }
#else
    // See Section 5.4.2.1 in 3GPP TS 38.212 v16.2.0 for details on Er computation. Please note that G, and thus Er,
    // does not account for any REs punctured due to CSI-RS. The Er, Emax computed here are upper bounds in that case.
    int modulo_C = no_rate_matching ? 0 : (G / (Nl * Qm)) % C; //split, if any, at C - modulo_C CB
    Er[0] = C - modulo_C;
    Er[1] = no_rate_matching ? G : Nl * Qm * floorf(G * 1.0f / (Nl * Qm * C)); // When !no_rate_matching G is a misnomer, it is actually N
    uint32_t max_TB_Er = Er[1] + ((modulo_C == 0) ? 0 : Nl * Qm);
    if (max_TB_Er > Emax) {
        Emax =  max_TB_Er;
        updated_Emax = true;
    } else {
        updated_Emax = false;
    }
    if (Emax % word_size != 0) {
        Emax = (Emax / word_size + 1) * word_size;
    }
#endif
    //printf("Er[0] %d, Er[1] %d, Emax %d\n", Er[0], Er[1], Emax);
}


size_t cuphyDlRateMatchingWorkspaceSize(int num_TBs) {
    //return num_TBs * sizeof(PdschPerTbParams) + num_TBs * sizeof(uint32_t) * (2 + 2);
    return num_TBs * sizeof(uint32_t) * (2 + 2);
}

// FIXME initial impl. exercised when cmdGrpNoData==1;
// Return value's bits [31:16] specify re_within_PRB_alloc, while the 4 least significant bits from [15:0] specify the symbol from [0, 13]
/* Alternatively, instead of sequentially looping over symbols, we could divide by  (REs_per_OFDM_symbol/2), if no CSI-RS is present, and check a pre-filled
   look up table e.g., in shared memory, that specifies which OFDM symbol this corresponds to.
   So for example, if the allocation was /-/data/dmrs/data/.../ (made up example)  and assuming the table has 2*OFDM_SYMBOLS_PER_SLOT entries,
   indexing with 0 or 1 would give symbol 1
   indexing with 2 should give symbol 2
   indexing with 3 or 4 should give symbol 2
*/
template<bool csi_rs=false>
__device__ __forceinline__ uint32_t find_symbol_pos_x(int qam_in_num_qams_per_layer_per_TB, int REs_per_OFDM_symbol,
                                                      uint64_t data_sym_loc, uint16_t dmrs_sym_loc, uint8_t first_symbol,
                                                      const PdschUeGrpParams* ue_grp_params) {
    uint32_t symbol_pos = 0;
    bool found = false;
    int dmrs_index = 0, data_index = 0;
    uint32_t x = 0;
    int skipped_REs = 0;
    int last_data_index = -1;
    bool is_dmrs = 0;
    int32_t re_within_PRB_alloc = 0;

    for (int32_t i = first_symbol; i < OFDM_SYMBOLS_PER_SLOT && !found; i++) {
        bool check = false;
        if (((data_sym_loc >> (4*data_index)) & 0x0fu) == i) {
            last_data_index = (data_index - 1); // yes, it'll be -1 for data_index 0
            skipped_REs = (csi_rs) ? ue_grp_params->cumulative_skipped_REs[data_index] : 0; // skipped REs is only for data symbols.
            x += 2;
            data_index += 1;
            is_dmrs = false;
            check = true;
        } else if ((dmrs_sym_loc >> (4*dmrs_index) & 0xfu) == i) {
            x += 1;
            dmrs_index += 1;
            is_dmrs = true;
            check = true;
        }
        if (csi_rs) {
            if (check && (qam_in_num_qams_per_layer_per_TB < ((x  * (REs_per_OFDM_symbol >> 1)) - skipped_REs))) {
                found = true;
                symbol_pos = i;
                // is_dmrs code path is in case there is a CSIRS symbol immediately before a DMRS symbol when cdmGrpNoData=1
                // Note that if the first symbol of this UE group is a CSIRS one and then there's DMRS, the last_data_index will still be -1
                // for the DMRS one, so the is_dmrs check should happen first.
                re_within_PRB_alloc = is_dmrs ? skipped_REs : ((last_data_index == -1) ? 0 : ue_grp_params->cumulative_skipped_REs[last_data_index]);

            }
        } else {
            if (check && (qam_in_num_qams_per_layer_per_TB < (x  * (REs_per_OFDM_symbol >> 1)))) {
                found = true;
                symbol_pos = i;
            }
        }
    }
    if (found) {
        if (is_dmrs) {
            // Only exercised if cdmGrpWithNoData=1 and for that the 3GPP spec says that only ports 0 or 1 can be used in that case, which implies delta for DMRS will always be 1
            // (i.g., even REs in PRB) and thus the data will always be at odd REs in PRB for the data written, thus the + 1 below
            // Please note that the (qam_in_num_qams_per_layer_per_TB - (x-1) * REs_per_OFDM_symbol/2)) operand may be negative, but the end result of re_within_PRB_alloc
            // should always be positive.
            re_within_PRB_alloc = (re_within_PRB_alloc + (qam_in_num_qams_per_layer_per_TB - (x-1) * REs_per_OFDM_symbol/2))*2 + 1;
        } else {
           re_within_PRB_alloc += (qam_in_num_qams_per_layer_per_TB - (x-2) * REs_per_OFDM_symbol/2);
        }
    }
    return ((re_within_PRB_alloc << 16) | symbol_pos);
}


__device__ __forceinline__ uint16_t find_freq_idx(int re_within_PRB_alloc, const uint32_t* rbBitmap)
{
            int desired_rb_num = re_within_PRB_alloc / CUPHY_N_TONES_PER_PRB; // Nth RB to be assigned
            // Find the index of the desired rb
            uint16_t rb_assign_cnt = 0;
            uint16_t rb_bitmap_idx = 0;
            uint32_t rbBitmap_element;
            // Find the element with desired RB set
            do {
                rbBitmap_element = rbBitmap[rb_bitmap_idx];
                rb_assign_cnt += __popc(rbBitmap_element);
                ++rb_bitmap_idx;
            } while((rb_assign_cnt <= desired_rb_num) && (rb_bitmap_idx < MAX_RBMASK_UINT32_ELEMENTS));

            // Back out last word bit totals and perform binary search to find bit index matching desired
            rb_bitmap_idx = (rb_bitmap_idx-1) << 5;
            rb_assign_cnt -= __popc(rbBitmap_element);
            uint16_t bits_in_element = desired_rb_num - rb_assign_cnt;
            uint32_t mask = 0xFFFF;
            uint16_t size = 16;
            while(size > 0) {
                const uint32_t count = __popc(rbBitmap_element & mask);
                if(bits_in_element >= count) {
                    rb_bitmap_idx += size;
                    size >>= 1;
                    mask |= mask << size;
                } else {
                    size >>= 1;
                    mask >>= size;
                }
            }
            int freq_idx = rb_bitmap_idx * CUPHY_N_TONES_PER_PRB + (re_within_PRB_alloc-desired_rb_num*CUPHY_N_TONES_PER_PRB);
            return freq_idx;
}

// No precoding for this specific UE. It is possible that other UEs in the cell have precoding enabled.
template<bool csi_rs=false>
__device__ __forceinline__ uint32_t output_index_no_precoding(int start_Rb, int qam_in_num_qams_per_layer_per_TB,
                                                         int REs_per_OFDM_symbol, const PdschDmrsParams* dmrs_params, const PdschUeGrpParams* ue_grp_params,
                                                         uint64_t data_symbol_loc, int all_Rbs_symbols, uint16_t* d_xtf_re_map,
                                                         int layer_id, uint16_t modified_dmrs_symbol_loc) {

    if (modified_dmrs_symbol_loc == DMRS_CDM_GRPS_NO_DATA_NOT_1_FLAG) {
        /* Compute potential symbol, per layer, i.e., which of the data_symbols data symbols (not incl. DMRS symbols)
           these bits correspond to.
           The symbol position in the [0, OFDM_SYMBOLS_PER_SLOT) will be computed in symbol_pos later.
           With csi_rs, potential_symbol will be the min symbol to map to; it could be a later one too.
       */

        int potential_symbol = qam_in_num_qams_per_layer_per_TB / REs_per_OFDM_symbol;
        uint32_t output_index = (start_Rb * CUPHY_N_TONES_PER_PRB);
        if (csi_rs) {
            /* potential_symbol is the min. symbol when we have skipped REs. Worst case, forward search from potential_symbol and on.
               Reminder this is only for data symbols and not DMRS ones. I could have used the # data symbols as boundary condition instead of
               OFDM_SYMBOLS_PER_SLOT, but the second condition will fail first.
                      */
            while ((potential_symbol + 1 < OFDM_SYMBOLS_PER_SLOT) && \
                   (qam_in_num_qams_per_layer_per_TB >= ((potential_symbol + 1) *  REs_per_OFDM_symbol  - ue_grp_params->cumulative_skipped_REs[potential_symbol]))) {
                potential_symbol += 1;
            }
            int symbol_pos = (data_symbol_loc >> (4*potential_symbol)) & 0xF;
            int re_within_PRB_alloc = qam_in_num_qams_per_layer_per_TB - ((potential_symbol == 0) ? 0 : (potential_symbol * REs_per_OFDM_symbol - ue_grp_params->cumulative_skipped_REs[potential_symbol-1]));
            if(dmrs_params->resourceAlloc) {
                output_index += all_Rbs_symbols * (layer_id * OFDM_SYMBOLS_PER_SLOT + symbol_pos) \
                             + re_within_PRB_alloc + d_xtf_re_map[symbol_pos * all_Rbs_symbols + re_within_PRB_alloc + start_Rb * CUPHY_N_TONES_PER_PRB];
            } else {
                int freq_idx = find_freq_idx(re_within_PRB_alloc, dmrs_params->rbBitmap) + CUPHY_N_TONES_PER_PRB * dmrs_params->BWP_start_PRB;
                output_index = all_Rbs_symbols * (layer_id * OFDM_SYMBOLS_PER_SLOT + symbol_pos) + freq_idx
                             + d_xtf_re_map[symbol_pos * all_Rbs_symbols + freq_idx];
            }
        } else {
            int symbol_pos = (data_symbol_loc >> (4*potential_symbol)) & 0xF;
            int re_within_PRB_alloc = qam_in_num_qams_per_layer_per_TB - potential_symbol * REs_per_OFDM_symbol;
            if(dmrs_params->resourceAlloc) {
                output_index += all_Rbs_symbols * (layer_id * OFDM_SYMBOLS_PER_SLOT + symbol_pos) + re_within_PRB_alloc;
            } else { // Resource Allocation Type 0
                output_index = all_Rbs_symbols * (layer_id * OFDM_SYMBOLS_PER_SLOT + symbol_pos) \
                             + find_freq_idx(re_within_PRB_alloc,dmrs_params->rbBitmap) + CUPHY_N_TONES_PER_PRB * dmrs_params->BWP_start_PRB;
            }
        }
        return output_index;
    }


    // only exercised if this UE has no dmrsCdmGrpNoData==1
    uint32_t output_index = (start_Rb * CUPHY_N_TONES_PER_PRB);
    uint32_t re_within_PRB_alloc_and_symbol_pos = find_symbol_pos_x<csi_rs>(qam_in_num_qams_per_layer_per_TB, REs_per_OFDM_symbol,  data_symbol_loc,  modified_dmrs_symbol_loc, dmrs_params->symbol_number, ue_grp_params);
    int re_within_PRB_alloc = (re_within_PRB_alloc_and_symbol_pos >> 16);
    if(dmrs_params->resourceAlloc == 0) {
        re_within_PRB_alloc = find_freq_idx(re_within_PRB_alloc, dmrs_params->rbBitmap) + (dmrs_params->BWP_start_PRB - start_Rb) * CUPHY_N_TONES_PER_PRB;
    }
    int symbol_pos = (re_within_PRB_alloc_and_symbol_pos & 0xf);
    output_index += (all_Rbs_symbols * symbol_pos + re_within_PRB_alloc);

    if (csi_rs) {
        output_index +=  d_xtf_re_map[output_index]; // output_index for d_xtf_re_map should not contain layer_id info
    }
    output_index +=  all_Rbs_symbols * (layer_id * OFDM_SYMBOLS_PER_SLOT);
    return output_index;
}

template<bool csi_rs=false>
__device__ __forceinline__ uint32_t partial_output_index(int start_Rb, int qam_in_num_qams_per_layer_per_TB,
                                                    int REs_per_OFDM_symbol, const PdschDmrsParams* __restrict__ dmrs_params, const PdschUeGrpParams* __restrict__ ue_grp_params,
                                                    uint64_t data_symbol_loc, int all_Rbs_symbols, uint16_t* __restrict__ d_xtf_re_map, uint16_t modified_dmrs_symbol_loc) {

    if (modified_dmrs_symbol_loc == DMRS_CDM_GRPS_NO_DATA_NOT_1_FLAG) {
        /* Compute potential symbol, per layer, i.e., which of the data_symbols data symbols (not incl. DMRS symbols)
           these bits correspond to.
           The symbol position in the [0, OFDM_SYMBOLS_PER_SLOT) will be computed in symbol_pos later.
           With csi_rs, potential_symbol will be the min symbol to map to; it could be a later one too.
       */

        int potential_symbol = qam_in_num_qams_per_layer_per_TB / REs_per_OFDM_symbol;
        uint32_t output_index = (start_Rb * CUPHY_N_TONES_PER_PRB);
        if (csi_rs) {
            /* potential_symbol is the min. symbol when we have skipped REs. Worst case, forward search from potential_symbol and on.
               Reminder this is only for data symbols and not DMRS ones. I could have used the # data symbols are boundary condition instead of
               OFDM_SYMBOLS_PER_SLOT, but the second condition will fail first.
                      */
            while ((potential_symbol + 1 < OFDM_SYMBOLS_PER_SLOT) && \
                   (qam_in_num_qams_per_layer_per_TB >= ((potential_symbol + 1) *  REs_per_OFDM_symbol  - ue_grp_params->cumulative_skipped_REs[potential_symbol]))) {
                potential_symbol += 1;
            }
            int symbol_pos = (data_symbol_loc >> (4*potential_symbol)) & 0xF;
            int re_within_PRB_alloc = qam_in_num_qams_per_layer_per_TB - ((potential_symbol == 0) ? 0 : (potential_symbol * REs_per_OFDM_symbol - ue_grp_params->cumulative_skipped_REs[potential_symbol-1]));
            if(dmrs_params->resourceAlloc) {
            output_index += all_Rbs_symbols * symbol_pos \
                            + re_within_PRB_alloc + d_xtf_re_map[symbol_pos * all_Rbs_symbols + re_within_PRB_alloc + start_Rb * CUPHY_N_TONES_PER_PRB];
            } else { // Resource Allocation Type 0
                // re_map is referenced to CRB0 whereas rbBitmap is BWP start so add it before re_map look-up
                int freq_idx = find_freq_idx(re_within_PRB_alloc,dmrs_params->rbBitmap) + CUPHY_N_TONES_PER_PRB * dmrs_params->BWP_start_PRB;
                output_index = all_Rbs_symbols * symbol_pos + freq_idx + d_xtf_re_map[symbol_pos * all_Rbs_symbols + freq_idx];
            }
        } else {
            int symbol_pos = (data_symbol_loc >> (4*potential_symbol)) & 0xF;
            int re_within_PRB_alloc = qam_in_num_qams_per_layer_per_TB - potential_symbol * REs_per_OFDM_symbol;
            if(dmrs_params->resourceAlloc) {
                output_index += all_Rbs_symbols * symbol_pos + re_within_PRB_alloc;
            } else { // Resource Allocation Type 0
                output_index = all_Rbs_symbols * symbol_pos + find_freq_idx(re_within_PRB_alloc,dmrs_params->rbBitmap) \
                             + CUPHY_N_TONES_PER_PRB * dmrs_params->BWP_start_PRB;
            }
        }
    return output_index;
    }

    // only exercised if the UE has dmrs_cdm_grps == 1
    uint32_t output_index = (start_Rb * CUPHY_N_TONES_PER_PRB);
    uint32_t re_within_PRB_alloc_and_symbol_pos = find_symbol_pos_x<csi_rs>(qam_in_num_qams_per_layer_per_TB, REs_per_OFDM_symbol,  data_symbol_loc,  modified_dmrs_symbol_loc, dmrs_params->symbol_number, ue_grp_params);
    int re_within_PRB_alloc = (re_within_PRB_alloc_and_symbol_pos >> 16);
    if(dmrs_params->resourceAlloc == 0) {
        re_within_PRB_alloc = find_freq_idx(re_within_PRB_alloc, dmrs_params->rbBitmap) - (start_Rb * CUPHY_N_TONES_PER_PRB);
    }
    int symbol_pos = (re_within_PRB_alloc_and_symbol_pos & 0xf);
    output_index += (all_Rbs_symbols * symbol_pos + re_within_PRB_alloc);
    if (csi_rs) {
        output_index += d_xtf_re_map[output_index];
    }
    return output_index;
}


template<uint8_t Tnl, bool csi_rs=false, bool precoding=false, uint8_t shift_bits=Tnl/2>
__device__ void QAM4_work_all_Nl_but_3(uint32_t CB_start_qam_per_layer, int EdivQm_bits,
                                       const PdschDmrsParams* __restrict__ dmrs_params, const PdschUeGrpParams* __restrict__ ue_grp_params, int rounded_Er_elements,
                                       __half2* __restrict__ modulation_output,
                                       uint16_t* __restrict__ d_xtf_re_map) {

     extern __shared__ uint32_t dl_rm_shmem[];
     int REs_per_OFDM_symbol =  dmrs_params->num_Rbs * CUPHY_N_TONES_PER_PRB;

     int all_Rbs_symbols = dmrs_params->num_BWP_PRBs * CUPHY_N_TONES_PER_PRB;
     int start_Rb = dmrs_params->start_Rb;
     uint32_t num_antenna_ports = dmrs_params->Np;

#if FLOAT_COMP
    float reciprocal_sqrt2 = 0.707106781186547f * dmrs_params->beta_qam;
#else
    __half reciprocal_sqrt2 = __hmul(hrsqrt(2), __float2half(dmrs_params->beta_qam)); //0.707106781186547;
#endif
    uint64_t data_symbol_loc = dmrs_params->data_sym_loc;
    uint16_t modified_dmrs_sym_loc = dmrs_params->dmrsCdmGrpsNoData1 ? dmrs_params->dmrs_sym_loc : DMRS_CDM_GRPS_NO_DATA_NOT_1_FLAG;

    if (dmrs_params->enablePrcdBf == 0) {

         for (int i = threadIdx.x; i < rounded_Er_elements; i+= blockDim.x) {
             uint32_t read_val = dl_rm_shmem[i];
             for (int intra_index = 0; intra_index < 16; intra_index++) {
                uint32_t qam_value = (read_val >> (intra_index * 2)) & 0x03U;
                int tmp_layer_id = intra_index & (Tnl - 1);
                int qam_offset = 16 * i + intra_index;
                int qam_in_num_qams_per_layer_per_TB = CB_start_qam_per_layer + (qam_offset >> shift_bits);
                if (qam_offset < EdivQm_bits) {
                   int layer_id = calculate_layer_id(dmrs_params->port_ids[tmp_layer_id], dmrs_params->n_scid, dmrs_params->nlAbove16);
                   uint32_t output_index = output_index_no_precoding<csi_rs>(start_Rb, qam_in_num_qams_per_layer_per_TB,
                                                                             REs_per_OFDM_symbol, dmrs_params, ue_grp_params,
                                                                             data_symbol_loc, all_Rbs_symbols, d_xtf_re_map, layer_id,
                                                                             modified_dmrs_sym_loc);
                    __half2 tmp_val;
                    tmp_val.x = ((qam_value & 0x1) == 0) ? reciprocal_sqrt2 : -reciprocal_sqrt2;
                    tmp_val.y = ((qam_value & 0x2) == 0) ? reciprocal_sqrt2 : -reciprocal_sqrt2;
                    if (precoding) {
                        atomicAdd(&modulation_output[output_index], tmp_val);
                    } else { // No TB in this cell has precoding enabled, so no need for atomic updates
                        modulation_output[output_index] = tmp_val;
                    }

                }
             }
         }

    } else { // Precoding for this UE

         for (int i = threadIdx.x; i < rounded_Er_elements; i+= blockDim.x) {
             uint32_t read_val = dl_rm_shmem[i];

             __half2 local_qam_output[MAX_DL_PORTS];

             for (int index = 0; index < 16/Tnl; index++) {

                 for (int tmp_layer_id = 0; tmp_layer_id < Tnl; tmp_layer_id++) {
                     int intra_index = index * Tnl + tmp_layer_id;
                     uint32_t qam_value = (read_val >> (intra_index * 2)) & 0x03U;

                     int qam_offset = 16 * i + intra_index;
    #if FLOAT_COMP
                     float2 tmp_val;
    #else
                     __half2 tmp_val;
    #endif

                     if (qam_offset < EdivQm_bits) {
                         tmp_val.x = ((qam_value & 0x1) == 0) ? reciprocal_sqrt2 : -reciprocal_sqrt2;
                         tmp_val.y = ((qam_value & 0x2) == 0) ? reciprocal_sqrt2 : -reciprocal_sqrt2;
                     } else { // Needed because I do the update in the end no matter what. So value should be 0.
                         tmp_val.x = 0;
                         tmp_val.y = 0;
                     }

                     for (int antenna_port = 0; antenna_port < num_antenna_ports; antenna_port++) {
                         __half2 matCoeff = dmrs_params->pmW[tmp_layer_id * num_antenna_ports + antenna_port];
#if FLOAT_COMP
                         float2 matCoeff2;
                         matCoeff2.x = matCoeff.x;
                         matCoeff2.y = matCoeff.y;
                         __half2 new_tmp_val = mult_float2(tmp_val, matCoeff2);
#else
                         __half2 new_tmp_val = mult_half2(tmp_val, matCoeff);
#endif
                         if (tmp_layer_id == 0) {
                             local_qam_output[antenna_port] = new_tmp_val;
                         } else {
                             local_qam_output[antenna_port] += new_tmp_val;
                         }
                      }
                  }

                  int qam_in_num_qams_per_layer_per_TB = CB_start_qam_per_layer + ((16*i + index *Tnl) >> shift_bits);

                  uint32_t output_index = partial_output_index<csi_rs>(start_Rb, qam_in_num_qams_per_layer_per_TB,
                                                                       REs_per_OFDM_symbol, dmrs_params, ue_grp_params,
                                                                       data_symbol_loc, all_Rbs_symbols, d_xtf_re_map,
                                                                       modified_dmrs_sym_loc);

                  for (int antenna_port = 0; antenna_port < num_antenna_ports; antenna_port++) {
                        uint32_t output_index_port_offset = all_Rbs_symbols * antenna_port * OFDM_SYMBOLS_PER_SLOT;
                        atomicAdd(&modulation_output[output_index + output_index_port_offset], local_qam_output[antenna_port]);
                  }
             }
         }
    }
}

template<bool csi_rs=false, bool precoding=false>
__device__ void QAM4_work_Nl_3(uint32_t CB_start_qam_per_layer, int EdivQm_bits,
                               const PdschDmrsParams* __restrict__ dmrs_params, const PdschUeGrpParams* __restrict__ ue_grp_params, int rounded_Er_elements,
                               __half2* __restrict__ modulation_output,
                               uint16_t* __restrict__ d_xtf_re_map) {

     extern __shared__ uint32_t dl_rm_shmem[];
     int Tnl = 3;
     int REs_per_OFDM_symbol =  dmrs_params->num_Rbs * CUPHY_N_TONES_PER_PRB;

     int all_Rbs_symbols = dmrs_params->num_BWP_PRBs * CUPHY_N_TONES_PER_PRB;
     int start_Rb = dmrs_params->start_Rb;
     uint32_t num_antenna_ports = dmrs_params->Np;

    //__half reciprocal_sqrt2 = __hmul(hrsqrt(2), __float2half(dmrs_params->beta_qam)); //0.707106781186547;
    float reciprocal_sqrt2 = 0.707106781186547f * dmrs_params->beta_qam;
    uint64_t data_symbol_loc = dmrs_params->data_sym_loc;
    uint16_t modified_dmrs_sym_loc = dmrs_params->dmrsCdmGrpsNoData1 ? dmrs_params->dmrs_sym_loc : DMRS_CDM_GRPS_NO_DATA_NOT_1_FLAG;

    if (dmrs_params->enablePrcdBf == 0) {

         for (int i = threadIdx.x; i < rounded_Er_elements; i+= blockDim.x) {
             uint32_t read_val = dl_rm_shmem[i];
             int i_mod_3 = i % 3;

             for (int intra_index = 0; intra_index < 16; intra_index++) {
                uint32_t qam_value = (read_val >> (intra_index * 2)) & 0x03U;
                int tmp_layer_id = (16 * i_mod_3 + intra_index) % Tnl;
                int qams_offset = (i / 3) * 16  + (16 * i_mod_3 + intra_index) / Tnl;
                int qam_in_num_qams_per_layer_per_TB = CB_start_qam_per_layer + qams_offset;
                if (qams_offset * Tnl  < EdivQm_bits) {
                   int layer_id = calculate_layer_id(dmrs_params->port_ids[tmp_layer_id], dmrs_params->n_scid, dmrs_params->nlAbove16);
                   uint32_t output_index = output_index_no_precoding<csi_rs>(start_Rb, qam_in_num_qams_per_layer_per_TB,
                                                                             REs_per_OFDM_symbol, dmrs_params, ue_grp_params,
                                                                             data_symbol_loc, all_Rbs_symbols, d_xtf_re_map, layer_id,
                                                                             modified_dmrs_sym_loc);

                    __half2 tmp_val;
                    tmp_val.x = ((qam_value & 0x1) == 0) ? reciprocal_sqrt2 : -reciprocal_sqrt2;
                    tmp_val.y = ((qam_value & 0x2) == 0) ? reciprocal_sqrt2 : -reciprocal_sqrt2;
                    if (precoding) {
                        atomicAdd(&modulation_output[output_index], tmp_val);
                    } else { // No TB in this cell has precoding enabled, so no need for atomic updates
                        modulation_output[output_index] = tmp_val;
                    }

                }
             }
         }
    } else { // Separate just for initial proof of concept example
         for (int i = threadIdx.x; i < rounded_Er_elements; i+= blockDim.x) {
             uint32_t read_val = dl_rm_shmem[i];
             int i_mod_3 = i % 3;

             __half2 local_qam_output[MAX_DL_PORTS];

             for (int index = 0; index < 6; index++) {

                 int start_layer_id = (index == 0) ? i_mod_3 : 0;
                 int end_layer_id = (index == 5) ? i_mod_3 + 1: Tnl;
                 for (int tmp_layer_id = start_layer_id; tmp_layer_id < end_layer_id; tmp_layer_id++) {

                     int intra_index = index * Tnl + tmp_layer_id - i_mod_3;
                     uint32_t qam_value = (read_val >> (intra_index * 2)) & 0x03U;
                     int qam_offset = (i / 3) * 16  + index + i_mod_3*5;

                     __half2 tmp_val;
                     if (qam_offset * Tnl  < EdivQm_bits) {
                         tmp_val.x = ((qam_value & 0x1) == 0) ? reciprocal_sqrt2 : -reciprocal_sqrt2;
                         tmp_val.y = ((qam_value & 0x2) == 0) ? reciprocal_sqrt2 : -reciprocal_sqrt2;
                     } else { // Needed because I do the update in the end no matter what. So value should be 0.
                         tmp_val.x = 0;
                         tmp_val.y = 0;
                     }

                     for (int antenna_port = 0; antenna_port < num_antenna_ports; antenna_port++) {
                         __half2 matCoeff = dmrs_params->pmW[tmp_layer_id * num_antenna_ports + antenna_port];
                         __half2 new_tmp_val = mult_half2(tmp_val, matCoeff);
                         if (tmp_layer_id == start_layer_id) {
                             local_qam_output[antenna_port] = new_tmp_val;
                         } else {
                             local_qam_output[antenna_port] += new_tmp_val;
                         }
                      }
                 }

                 int qam_in_num_qams_per_layer_per_TB = CB_start_qam_per_layer + (i / 3) * 16  + index + i_mod_3*5;
                 uint32_t output_index = partial_output_index<csi_rs>(start_Rb, qam_in_num_qams_per_layer_per_TB,
                                                                      REs_per_OFDM_symbol, dmrs_params, ue_grp_params,
                                                                      data_symbol_loc, all_Rbs_symbols, d_xtf_re_map,
                                                                      modified_dmrs_sym_loc);

                 for (int antenna_port = 0; antenna_port < num_antenna_ports; antenna_port++) {
                       uint32_t output_index_port_offset = all_Rbs_symbols * antenna_port * OFDM_SYMBOLS_PER_SLOT;
                       atomicAdd(&modulation_output[output_index + output_index_port_offset], local_qam_output[antenna_port]);
                 }

             }
         }
    }
}


template<uint8_t Tnl, bool csi_rs=false, bool precoding=false, uint8_t shift_bits=Tnl/2>
__device__ void QAM16_work_all_Nl_but_3(uint32_t CB_start_qam_per_layer, int EdivQm_bits,
                                        const PdschDmrsParams* __restrict__ dmrs_params, const PdschUeGrpParams* __restrict__ ue_grp_params, int rounded_Er_elements,
                                        __half2* __restrict__ modulation_output,
                                        uint16_t* __restrict__ d_xtf_re_map) {

     uint32_t num_antenna_ports = dmrs_params->Np;

     extern __shared__ uint32_t dl_rm_shmem[];
     int REs_per_OFDM_symbol =  dmrs_params->num_Rbs * CUPHY_N_TONES_PER_PRB;

     int all_Rbs_symbols = dmrs_params->num_BWP_PRBs * CUPHY_N_TONES_PER_PRB;
     int start_Rb = dmrs_params->start_Rb;

     __shared__ __half shmem_qam_16[8];
     if (threadIdx.x < 8) {
         shmem_qam_16[threadIdx.x] = (__half) (rev_qam_16_long[threadIdx.x] * dmrs_params->beta_qam);
     }
     __syncthreads();

    uint64_t data_symbol_loc = dmrs_params->data_sym_loc;
    uint16_t modified_dmrs_sym_loc = dmrs_params->dmrsCdmGrpsNoData1 ? dmrs_params->dmrs_sym_loc : DMRS_CDM_GRPS_NO_DATA_NOT_1_FLAG;

    if (dmrs_params->enablePrcdBf == 0) {

         for (int i = threadIdx.x; i < rounded_Er_elements; i+= blockDim.x) {
             uint32_t read_val = dl_rm_shmem[i];

             for (int intra_index = 0; intra_index < 8; intra_index++) {
                uint32_t qam_value = (read_val >> (intra_index * 4)) & 0x0FU;
                int tmp_layer_id = intra_index & (Tnl - 1); // modulo Tnl which is 1, 2 or 4
                int qam_offset = 8 * i + intra_index;
                int qam_in_num_qams_per_layer_per_TB = CB_start_qam_per_layer + (qam_offset >> shift_bits);
                if (qam_offset < EdivQm_bits) {
                    int layer_id = calculate_layer_id(dmrs_params->port_ids[tmp_layer_id], dmrs_params->n_scid, dmrs_params->nlAbove16);
                    uint32_t output_index = output_index_no_precoding<csi_rs>(start_Rb, qam_in_num_qams_per_layer_per_TB,
                                                                              REs_per_OFDM_symbol, dmrs_params, ue_grp_params,
                                                                              data_symbol_loc, all_Rbs_symbols, d_xtf_re_map, layer_id,
                                                                              modified_dmrs_sym_loc);

                    if (precoding) {
                        atomicAdd(&modulation_output[output_index], make_complex<__half2>::create(shmem_qam_16[qam_value & 0x05],
                                                                                          shmem_qam_16[(qam_value >> 1) & 0x05]));
                    } else { // No TB in this cell has precoding enabled, so no need for atomic updates
                        modulation_output[output_index] = make_complex<__half2>::create(shmem_qam_16[qam_value & 0x05],
                                                                                        shmem_qam_16[(qam_value >> 1) & 0x05]);
                    }
                }
             }
        }
    } else {
         for (int i = threadIdx.x; i < rounded_Er_elements; i+= blockDim.x) {
             uint32_t read_val = dl_rm_shmem[i];

             __half2 local_qam_output[MAX_DL_PORTS];
             for (int index = 0; index < 8/Tnl; index++) {

                 for (int tmp_layer_id = 0; tmp_layer_id < Tnl; tmp_layer_id++) {
                     int intra_index = index * Tnl + tmp_layer_id;
                     uint32_t qam_value = (read_val >> (intra_index * 4)) & 0x0FU;

                     int qam_offset = 8 * i + intra_index;
                     __half2 tmp_val  = (qam_offset < EdivQm_bits) ? \
                                         make_complex<__half2>::create(shmem_qam_16[qam_value & 0x05],
                                                                       shmem_qam_16[(qam_value >> 1) & 0x05]) :
                                         make_complex<__half2>::create(0, 0);

                     for (int antenna_port = 0; antenna_port < num_antenna_ports; antenna_port++) {
                         __half2 matCoeff = dmrs_params->pmW[tmp_layer_id * num_antenna_ports + antenna_port];
                         __half2 new_tmp_val = mult_half2(tmp_val, matCoeff);
                         if (tmp_layer_id == 0) {
                             local_qam_output[antenna_port] = new_tmp_val;
                         } else {
                             local_qam_output[antenna_port] += new_tmp_val;
                         }
                      }
                  }

                  int qam_in_num_qams_per_layer_per_TB = CB_start_qam_per_layer + ((8*i + index *Tnl) >> shift_bits);

                  uint32_t output_index = partial_output_index<csi_rs>(start_Rb, qam_in_num_qams_per_layer_per_TB,
                                                                       REs_per_OFDM_symbol, dmrs_params, ue_grp_params,
                                                                       data_symbol_loc, all_Rbs_symbols, d_xtf_re_map,
                                                                       modified_dmrs_sym_loc);

                  for (int antenna_port = 0; antenna_port < num_antenna_ports; antenna_port++) {
                        uint32_t output_index_port_offset = all_Rbs_symbols * antenna_port * OFDM_SYMBOLS_PER_SLOT;
                        atomicAdd(&modulation_output[output_index + output_index_port_offset], local_qam_output[antenna_port]);
                  }

              }
         }
    }
}

template<bool csi_rs=false, bool precoding=false>
__device__ void QAM16_work_Nl_3(uint32_t CB_start_qam_per_layer, int EdivQm_bits,
                                const PdschDmrsParams* __restrict__ dmrs_params, const PdschUeGrpParams* __restrict__ ue_grp_params, int rounded_Er_elements,
                                __half2* __restrict__ modulation_output,
                                uint16_t* __restrict__ d_xtf_re_map) {

     extern __shared__ uint32_t dl_rm_shmem[];
     int Tnl = 3;
     int REs_per_OFDM_symbol =  dmrs_params->num_Rbs * CUPHY_N_TONES_PER_PRB;

     int all_Rbs_symbols = dmrs_params->num_BWP_PRBs * CUPHY_N_TONES_PER_PRB;
     int start_Rb = dmrs_params->start_Rb;

     uint32_t num_antenna_ports = dmrs_params->Np;

     __shared__ __half shmem_qam_16[8];
     if (threadIdx.x < 8) {
         shmem_qam_16[threadIdx.x] = (__half) (rev_qam_16_long[threadIdx.x] * dmrs_params->beta_qam);
     }
     __syncthreads();
    uint64_t data_symbol_loc = dmrs_params->data_sym_loc;
    uint16_t modified_dmrs_sym_loc = dmrs_params->dmrsCdmGrpsNoData1 ? dmrs_params->dmrs_sym_loc : DMRS_CDM_GRPS_NO_DATA_NOT_1_FLAG;

    if (dmrs_params->enablePrcdBf == 0) {

         for (int i = threadIdx.x; i < rounded_Er_elements; i+= blockDim.x) {
             uint32_t read_val = dl_rm_shmem[i];
             int i_mod_3 = i % 3;

             for (int intra_index = 0; intra_index < 8; intra_index++) {
                uint32_t qam_value = (read_val >> (intra_index * 4)) & 0x0FU;
                int tmp_layer_id = (8 * i_mod_3 + intra_index) % Tnl;
                int qams_offset = (i / 3) * 8 + (8 * i_mod_3 + intra_index) / Tnl;
                int qam_in_num_qams_per_layer_per_TB = CB_start_qam_per_layer + qams_offset;
                if (qams_offset * Tnl <  EdivQm_bits) {
                    int layer_id = calculate_layer_id(dmrs_params->port_ids[tmp_layer_id], dmrs_params->n_scid, dmrs_params->nlAbove16);
                    uint32_t output_index = output_index_no_precoding<csi_rs>(start_Rb, qam_in_num_qams_per_layer_per_TB,
                                                                              REs_per_OFDM_symbol, dmrs_params, ue_grp_params,
                                                                              data_symbol_loc, all_Rbs_symbols, d_xtf_re_map, layer_id,
                                                                              modified_dmrs_sym_loc);

                    if (precoding) {
                        atomicAdd(&modulation_output[output_index],  make_complex<__half2>::create(shmem_qam_16[qam_value & 0x05],
                                                                        shmem_qam_16[(qam_value >> 1) & 0x05]));
                    } else { // No TB in this cell has precoding enabled, so no need for atomic updates
                        modulation_output[output_index] = make_complex<__half2>::create(shmem_qam_16[qam_value & 0x05],
                                                                           shmem_qam_16[(qam_value >> 1) & 0x05]);
                    }
                }
             }
         }
    } else {
         uint8_t start_layer[3] = {0, 2, 1};
         uint8_t end_layer[3] = {2, 1, 3};

         for (int i = threadIdx.x; i < rounded_Er_elements; i+= blockDim.x) {
             uint32_t read_val = dl_rm_shmem[i];
             int i_mod_3 = i % 3;
             int end_index = (i_mod_3 == 1) ? 4 : 3;

             __half2 local_qam_output[MAX_DL_PORTS];

             for (int index = 0; index < end_index; index++) {

                 int start_layer_id = (index == 0) ? start_layer[i_mod_3] : 0;
                 int end_layer_id = (index == end_index-1) ? end_layer[i_mod_3] : Tnl;
                 for (int tmp_layer_id = start_layer_id; tmp_layer_id < end_layer_id; tmp_layer_id++) {

                     int intra_index = index * Tnl + tmp_layer_id - start_layer[i_mod_3];
                     uint32_t qam_value = (read_val >> (intra_index * 4)) & 0x0FU;
                     int qam_offset = (i / 3) * 8  + index + i_mod_3*2;
                     if (i_mod_3 == 2) qam_offset += 1;

                     __half2 tmp_val;
                     if (qam_offset * Tnl  < EdivQm_bits) {
                         tmp_val = make_complex<__half2>::create(shmem_qam_16[qam_value & 0x05],
                                                                 shmem_qam_16[(qam_value >> 1) & 0x05]);
                     } else { // Needed because I do the update in the end no matter what. So value should be 0.
                         tmp_val.x = 0;
                         tmp_val.y = 0;
                     }

                     for (int antenna_port = 0; antenna_port < num_antenna_ports; antenna_port++) {
                         __half2 matCoeff = dmrs_params->pmW[tmp_layer_id * num_antenna_ports + antenna_port];
                         __half2 new_tmp_val = mult_half2(tmp_val, matCoeff);
                         if (tmp_layer_id == start_layer_id) {
                             local_qam_output[antenna_port] = new_tmp_val;
                         } else {
                             local_qam_output[antenna_port] += new_tmp_val;
                         }
                      }
                 }

                  int qam_in_num_qams_per_layer_per_TB = CB_start_qam_per_layer + (i / 3) * 8  + index + i_mod_3*2;
                  if (i_mod_3 == 2) qam_in_num_qams_per_layer_per_TB += 1;

                  uint32_t output_index = partial_output_index<csi_rs>(start_Rb, qam_in_num_qams_per_layer_per_TB,
                                                                       REs_per_OFDM_symbol, dmrs_params, ue_grp_params,
                                                                       data_symbol_loc, all_Rbs_symbols, d_xtf_re_map,
                                                                       modified_dmrs_sym_loc);

                  for (int antenna_port = 0; antenna_port < num_antenna_ports; antenna_port++) {
                        uint32_t output_index_port_offset = all_Rbs_symbols * antenna_port * OFDM_SYMBOLS_PER_SLOT;
                        atomicAdd(&modulation_output[output_index + output_index_port_offset], local_qam_output[antenna_port]);
                  }

             }
         }
    }
}


template<uint8_t Tnl, bool csi_rs=false, bool precoding=false>
__device__ void QAM64_work_all_Nl_but_3(uint32_t CB_start_qam_per_layer, int EdivQm_bits,
                                        const PdschDmrsParams* __restrict__ dmrs_params, const PdschUeGrpParams* __restrict__ ue_grp_params, int rounded_Er_elements,
                                        __half2* __restrict__ modulation_output,
                                        uint16_t* __restrict__ d_xtf_re_map) {

     extern __shared__ uint32_t dl_rm_shmem[];
     int REs_per_OFDM_symbol =  dmrs_params->num_Rbs * CUPHY_N_TONES_PER_PRB;

     int all_Rbs_symbols = dmrs_params->num_BWP_PRBs * CUPHY_N_TONES_PER_PRB;
     int start_Rb = dmrs_params->start_Rb;
     uint32_t num_antenna_ports = dmrs_params->Np;

     __shared__ __half shmem_qam_64[8];
     if (threadIdx.x < 8) {
        shmem_qam_64[threadIdx.x] = (__half) (rev_qam_64[threadIdx.x] * dmrs_params->beta_qam);
     }
     __syncthreads();

     uint8_t offset[3] = {0, 6, 11};
    uint64_t data_symbol_loc = dmrs_params->data_sym_loc;
    uint16_t modified_dmrs_sym_loc = dmrs_params->dmrsCdmGrpsNoData1 ? dmrs_params->dmrs_sym_loc : DMRS_CDM_GRPS_NO_DATA_NOT_1_FLAG;

    if (dmrs_params->enablePrcdBf == 0) {

         for (int i = threadIdx.x; i < rounded_Er_elements; i+= blockDim.x) {
             uint32_t read_val = dl_rm_shmem[i];
             int i_mod_3 = i % 3;
             int max_qams = (i_mod_3 == 0) ? 6 : 5;

             if (i_mod_3 == 1) {
                 read_val >>= 4;
                 // Read 2 bits from the next element
                 read_val |= ((dl_rm_shmem[i+1] & 0x03U) << 28);
             }
             if (i_mod_3 == 2) {
                 read_val >>= 2;
             }

             for (int intra_index = 0; intra_index < max_qams; intra_index++) { // Can read at most 6 QAMs
                uint32_t qam_value = (read_val >> (intra_index * 6)) & 0x3FU;
                if ((i_mod_3 == 0) && (intra_index == 5)) {
                    // Read 4 bits from the next element
                    qam_value &= 0x03U;
                    qam_value |= ((dl_rm_shmem[i + 1] & 0x0FU) << 2);
                }
                int tmp_layer_id = ((offset[i_mod_3] + intra_index) % Tnl);
                int qams_offset = (i / 3) * (16 / Tnl) + (offset[i_mod_3] + intra_index)/ Tnl;

                int qam_in_num_qams_per_layer_per_TB = CB_start_qam_per_layer + qams_offset;
                if (qams_offset * Tnl <  EdivQm_bits) {
                    int layer_id = calculate_layer_id(dmrs_params->port_ids[tmp_layer_id], dmrs_params->n_scid, dmrs_params->nlAbove16);
                    uint32_t output_index = output_index_no_precoding<csi_rs>(start_Rb, qam_in_num_qams_per_layer_per_TB,
                                                                              REs_per_OFDM_symbol, dmrs_params, ue_grp_params,
                                                                              data_symbol_loc, all_Rbs_symbols, d_xtf_re_map, layer_id,
                                                                              modified_dmrs_sym_loc);
                    int x_index = map_index_6bits(qam_value);
                    int y_index = map_index_6bits(qam_value >> 1);

                    if (precoding) {
                        atomicAdd(&modulation_output[output_index], make_complex<__half2>::create(shmem_qam_64[x_index],
                                                                       shmem_qam_64[y_index]));
                    } else { // No TB in this cell has precoding enabled, so no need for atomic updates
                        modulation_output[output_index] = make_complex<__half2>::create(shmem_qam_64[x_index],
                                                                                        shmem_qam_64[y_index]);
                    }

                }
             }
         }
    } else {
         for (int i = threadIdx.x; i < rounded_Er_elements; i+= blockDim.x) {
             uint32_t read_val = dl_rm_shmem[i];
             int i_mod_3 = i % 3;
             int max_qams = (i_mod_3 == 0) ? 6 : 5;

             if (i_mod_3 == 1) {
                 read_val >>= 4;
                 // Read 2 bits from the next element
                 read_val |= ((dl_rm_shmem[i+1] & 0x03U) << 28);
             }
             if (i_mod_3 == 2) {
                 read_val >>= 2;
             }

             for (int intra_index = 0; intra_index < max_qams; intra_index++) { // Can read at most 6 QAMs
                uint32_t qam_value = (read_val >> (intra_index * 6)) & 0x3FU;
                if ((i_mod_3 == 0) && (intra_index == 5)) {
                    // Read 4 bits from the next element
                    qam_value &= 0x03U;
                    qam_value |= ((dl_rm_shmem[i + 1] & 0x0FU) << 2);
                }
                int tmp_layer_id = ((offset[i_mod_3] + intra_index) % Tnl);
                int qams_offset = (i / 3) * (16 / Tnl) + (offset[i_mod_3] + intra_index)/ Tnl;

                int qam_in_num_qams_per_layer_per_TB = CB_start_qam_per_layer + qams_offset;
                if (qams_offset * Tnl <  EdivQm_bits) {

                  uint32_t output_index = partial_output_index<csi_rs>(start_Rb, qam_in_num_qams_per_layer_per_TB,
                                                                       REs_per_OFDM_symbol, dmrs_params, ue_grp_params,
                                                                       data_symbol_loc, all_Rbs_symbols, d_xtf_re_map,
                                                                       modified_dmrs_sym_loc);
                    int x_index = map_index_6bits(qam_value);
                    int y_index = map_index_6bits(qam_value >> 1);

                    __half2 tmp_val = make_complex<__half2>::create(shmem_qam_64[x_index],
                                                                    shmem_qam_64[y_index]);

                    for (int antenna_port = 0; antenna_port < num_antenna_ports; antenna_port++) {
                        __half2 matCoeff = dmrs_params->pmW[tmp_layer_id * num_antenna_ports + antenna_port];
                        uint32_t output_index_port_offset = all_Rbs_symbols * antenna_port * OFDM_SYMBOLS_PER_SLOT;
                        __half2 new_tmp_val = mult_half2(tmp_val, matCoeff);
                        atomicAdd(&modulation_output[output_index + output_index_port_offset], new_tmp_val); // FIXME inefficient. Initial proof of concept.
                    }

                }
             }
         }
    }
}


template<bool csi_rs=false, bool precoding=false>
__device__ void QAM64_work_Nl_3(uint32_t CB_start_qam_per_layer, int EdivQm_bits,
                                const PdschDmrsParams* __restrict__ dmrs_params, const PdschUeGrpParams* __restrict__ ue_grp_params, int rounded_Er_elements,
                                __half2* __restrict__ modulation_output,
                                uint16_t* __restrict__ d_xtf_re_map) {

     extern __shared__ uint32_t dl_rm_shmem[];
     int Tnl = 3;
     int REs_per_OFDM_symbol =  dmrs_params->num_Rbs * CUPHY_N_TONES_PER_PRB;

     int all_Rbs_symbols = dmrs_params->num_BWP_PRBs * CUPHY_N_TONES_PER_PRB;
     int start_Rb = dmrs_params->start_Rb;
     uint32_t num_antenna_ports = dmrs_params->Np;

     __shared__ __half shmem_qam_64[8];
     if (threadIdx.x < 8) {
        shmem_qam_64[threadIdx.x] = (__half) (rev_qam_64[threadIdx.x] * dmrs_params->beta_qam);
     }
     __syncthreads();

    if (dmrs_params->enablePrcdBf == 0) {

         uint8_t offset[9] = {0, 6, 11,
                              16, 22, 27,
                              32, 38, 43};
        uint64_t data_symbol_loc = dmrs_params->data_sym_loc;
        uint16_t modified_dmrs_sym_loc = dmrs_params->dmrsCdmGrpsNoData1 ? dmrs_params->dmrs_sym_loc : DMRS_CDM_GRPS_NO_DATA_NOT_1_FLAG;

         for (int i = threadIdx.x; i < rounded_Er_elements; i+= blockDim.x) {
             uint32_t read_val = dl_rm_shmem[i];
             int i_mod_3 = i % 3;
             int i_mod_9 = i % 9;
             int max_qams = (i_mod_3 == 0) ? 6 : 5;

             if (i_mod_3 == 1) {
                 read_val >>= 4;
                 // Read 2 bits from the next element
                 read_val |= ((dl_rm_shmem[i+1] & 0x03U) << 28);
             }
             if (i_mod_3 == 2) {
                 read_val >>= 2;
             }

             for (int intra_index = 0; intra_index < max_qams; intra_index++) { // Can read at most 6 QAMs
                uint32_t qam_value = (read_val >> (intra_index * 6)) & 0x3FU;
                if ((i_mod_3 == 0) && (intra_index == 5)) {
                    // Read 4 bits from the next element
                    qam_value &= 0x03U;
                    qam_value |= ((dl_rm_shmem[i + 1] & 0x0FU) << 2);
                }
                int tmp_layer_id = ((offset[i_mod_9] + intra_index) % Tnl);
                int qams_offset = (i / 9) * 16 + (offset[i_mod_9] + intra_index)/ Tnl;

                int qam_in_num_qams_per_layer_per_TB = CB_start_qam_per_layer + qams_offset;
                if (qams_offset * Tnl <  EdivQm_bits) {
                    int layer_id = calculate_layer_id(dmrs_params->port_ids[tmp_layer_id], dmrs_params->n_scid, dmrs_params->nlAbove16);
                    uint32_t output_index = output_index_no_precoding<csi_rs>(start_Rb, qam_in_num_qams_per_layer_per_TB,
                                                                              REs_per_OFDM_symbol, dmrs_params, ue_grp_params,
                                                                              data_symbol_loc, all_Rbs_symbols, d_xtf_re_map, layer_id,
                                                                              modified_dmrs_sym_loc);

                    int x_index = map_index_6bits(qam_value);
                    int y_index = map_index_6bits(qam_value >> 1);

                    if (precoding) {
                        atomicAdd(&modulation_output[output_index], make_complex<__half2>::create(shmem_qam_64[x_index],
                                                                        shmem_qam_64[y_index]));
                    } else { // No TB in this cell has precoding enabled, so no need for atomic updates
                        modulation_output[output_index] = make_complex<__half2>::create(shmem_qam_64[x_index],
                                                                           shmem_qam_64[y_index]);
                    }
                }
             }
         }
    } else {
         uint8_t offset[9] = {0, 6, 11,
                              16, 22, 27,
                              32, 38, 43};
        uint64_t data_symbol_loc = dmrs_params->data_sym_loc;
        uint16_t modified_dmrs_sym_loc = dmrs_params->dmrsCdmGrpsNoData1 ? dmrs_params->dmrs_sym_loc : DMRS_CDM_GRPS_NO_DATA_NOT_1_FLAG;

         for (int i = threadIdx.x; i < rounded_Er_elements; i+= blockDim.x) {
             uint32_t read_val = dl_rm_shmem[i];
             int i_mod_3 = i % 3;
             int i_mod_9 = i % 9;
             int max_qams = (i_mod_3 == 0) ? 6 : 5;

             if (i_mod_3 == 1) {
                 read_val >>= 4;
                 // Read 2 bits from the next element
                 read_val |= ((dl_rm_shmem[i+1] & 0x03U) << 28);
             }
             if (i_mod_3 == 2) {
                 read_val >>= 2;
             }

             for (int intra_index = 0; intra_index < max_qams; intra_index++) { // Can read at most 6 QAMs
                uint32_t qam_value = (read_val >> (intra_index * 6)) & 0x3FU;
                if ((i_mod_3 == 0) && (intra_index == 5)) {
                    // Read 4 bits from the next element
                    qam_value &= 0x03U;
                    qam_value |= ((dl_rm_shmem[i + 1] & 0x0FU) << 2);
                }
                int tmp_layer_id = ((offset[i_mod_9] + intra_index) % Tnl);
                int qams_offset = (i / 9) * 16 + (offset[i_mod_9] + intra_index)/ Tnl;

                int qam_in_num_qams_per_layer_per_TB = CB_start_qam_per_layer + qams_offset;
                if (qams_offset * Tnl <  EdivQm_bits) {
                    uint32_t output_index = partial_output_index<csi_rs>(start_Rb, qam_in_num_qams_per_layer_per_TB,
                                                                         REs_per_OFDM_symbol, dmrs_params, ue_grp_params,
                                                                         data_symbol_loc, all_Rbs_symbols, d_xtf_re_map,
                                                                         modified_dmrs_sym_loc);
                    int x_index = map_index_6bits(qam_value);
                    int y_index = map_index_6bits(qam_value >> 1);

                    __half2 tmp_val = make_complex<__half2>::create(shmem_qam_64[x_index],
                                                                    shmem_qam_64[y_index]);
                    for (int antenna_port = 0; antenna_port < num_antenna_ports; antenna_port++) {
                        __half2 matCoeff = dmrs_params->pmW[tmp_layer_id * num_antenna_ports + antenna_port];
                        uint32_t output_index_port_offset = all_Rbs_symbols * antenna_port * OFDM_SYMBOLS_PER_SLOT;
                        __half2 new_tmp_val = mult_half2(tmp_val, matCoeff);
                        atomicAdd(&modulation_output[output_index + output_index_port_offset], new_tmp_val); // FIXME inefficient. Initial proof of concept.
                    }

                }
             }
         }
    }
}


template<uint8_t Tnl, bool csi_rs=false, bool precoding=false, uint8_t shift_bits=Tnl/2>
__device__ void QAM256_work_all_Nl_but_3(uint32_t CB_start_qam_per_layer, int EdivQm_bits,
                                         const PdschDmrsParams* __restrict__ dmrs_params, const PdschUeGrpParams* __restrict__ ue_grp_params, int rounded_Er_elements,
                                         __half2* __restrict__ modulation_output,
                                         uint16_t* __restrict__ d_xtf_re_map, __half2* shmem_matrix) {

     int32_t num_antenna_ports = dmrs_params->Np;
     __builtin_assume(num_antenna_ports <= MAX_DL_PORTS);

     extern __shared__ uint32_t dl_rm_shmem[];
     int REs_per_OFDM_symbol =  dmrs_params->num_Rbs * CUPHY_N_TONES_PER_PRB;

     int all_Rbs_symbols = dmrs_params->num_BWP_PRBs * CUPHY_N_TONES_PER_PRB;
     int start_Rb = dmrs_params->start_Rb;

    __shared__ __half  shmem_qam_256[16];
    if (threadIdx.x < 16) {
        shmem_qam_256[threadIdx.x] = (__half) (rev_qam_256[threadIdx.x] * dmrs_params->beta_qam);
    }
    uint64_t data_symbol_loc = dmrs_params->data_sym_loc;
    uint16_t modified_dmrs_sym_loc = dmrs_params->dmrsCdmGrpsNoData1 ? dmrs_params->dmrs_sym_loc : DMRS_CDM_GRPS_NO_DATA_NOT_1_FLAG;

    __pipeline_wait_prior(0);
    __syncthreads();

    if (dmrs_params->enablePrcdBf == 0) {

         for (int i = threadIdx.x; i < rounded_Er_elements; i+= blockDim.x) {
             uint32_t read_val = dl_rm_shmem[i];
             for (int intra_index = 0; intra_index < 4; intra_index++) {
                uint32_t qam_value = (read_val >> (intra_index * 8)) & 0x0FFU;
                int tmp_layer_id = intra_index & (Tnl - 1);
                int qams_offset = i * 4 + intra_index;
                int qam_in_num_qams_per_layer_per_TB = CB_start_qam_per_layer + (qams_offset >> shift_bits);
                if (qams_offset < EdivQm_bits) {
                    int layer_id = calculate_layer_id(dmrs_params->port_ids[tmp_layer_id], dmrs_params->n_scid, dmrs_params->nlAbove16);
                    uint32_t output_index = output_index_no_precoding<csi_rs>(start_Rb, qam_in_num_qams_per_layer_per_TB,
                                                                              REs_per_OFDM_symbol, dmrs_params, ue_grp_params,
                                                                              data_symbol_loc, all_Rbs_symbols, d_xtf_re_map, layer_id,
                                                                              modified_dmrs_sym_loc);

                    int x_index = map_index_8bits(qam_value);
                    int y_index = map_index_8bits(qam_value >> 1);

                    if (precoding) {
                        atomicAdd(&modulation_output[output_index],  make_complex<__half2>::create(shmem_qam_256[x_index],
                                                                        shmem_qam_256[y_index]));
                    } else { // No TB in this cell has precoding enabled, so no need for atomic updates
                        modulation_output[output_index] = make_complex<__half2>::create(shmem_qam_256[x_index],
                                                                        shmem_qam_256[y_index]);
                    }

                }
             }
         }
    } else {
        if (num_antenna_ports == 4) { // Code avoids use of local memory for local_qam_output as in else clause

                 for (int i = threadIdx.x; i < rounded_Er_elements; i+= blockDim.x) {
                     uint32_t read_val = dl_rm_shmem[i];

                     for (int index = 0; index < 4/Tnl; index++) {
                         u128_half2x4 local_qam_output;
                         local_qam_output.u128 = 0;

                         for (int tmp_layer_id = 0; tmp_layer_id < Tnl; tmp_layer_id++) {
                             int intra_index = index * Tnl + tmp_layer_id;
                             uint32_t qam_value = (read_val >> (intra_index * 8)) & 0x0FFU;
                             int qams_offset = i * 4 + intra_index;

                             int x_index = map_index_8bits(qam_value);
                             int y_index = map_index_8bits(qam_value >> 1);

                             __half2 tmp_val = (qams_offset < EdivQm_bits) ? \
                                                make_complex<__half2>::create(shmem_qam_256[x_index],
                                                                              shmem_qam_256[y_index]) :
                                                make_complex<__half2>::create(0, 0);

                             // Since num_antenna_ports is 4, manually unroll the loop over num_antenna_ports
        #if 1
                             __half2 matCoeff = shmem_matrix[tmp_layer_id * 4 + 0];
                             local_qam_output.h2[0] = __hcmadd(tmp_val, matCoeff,  local_qam_output.h2[0]);

                             matCoeff = shmem_matrix[tmp_layer_id * 4 + 1];
                             local_qam_output.h2[1] = __hcmadd(tmp_val, matCoeff, local_qam_output.h2[1]);

                             matCoeff = shmem_matrix[tmp_layer_id * 4 + 2];
                             local_qam_output.h2[2] = __hcmadd(tmp_val, matCoeff, local_qam_output.h2[2]);

                             matCoeff = shmem_matrix[tmp_layer_id * 4 + 3];
                             local_qam_output.h2[3] = __hcmadd(tmp_val, matCoeff, local_qam_output.h2[3]);
        #else
                             // compiler does the single 128-bit load from shared memory, so this else is not strictly needed
                             uint4_half2x4 matCoeffs;
                             matCoeffs.u4 = shmem_matrix[tmp_layer_id * 4];
                             __half2 new_tmp_val = mult_half2(tmp_val, matCoeffs.h2[0]);
                             local_qam_output.h2[0] += new_tmp_val;

                             new_tmp_val = mult_half2(tmp_val, matCoeff.h2[1]);
                             local_qam_output.h2[1] += new_tmp_val;

                             new_tmp_val = mult_half2(tmp_val, matCoeff.h2[2]);
                             local_qam_output.h2[2] += new_tmp_val;

                             new_tmp_val = mult_half2(tmp_val, matCoeff.h2[3]);
                             local_qam_output.h2[3] += new_tmp_val;
        #endif
                         }

                         int qam_in_num_qams_per_layer_per_TB = CB_start_qam_per_layer + ((4*i + index *Tnl) >> shift_bits);
                         uint32_t output_index = partial_output_index<csi_rs>(start_Rb, qam_in_num_qams_per_layer_per_TB,
                                                                              REs_per_OFDM_symbol, dmrs_params, ue_grp_params,
                                                                              data_symbol_loc, all_Rbs_symbols, d_xtf_re_map,
                                                                              modified_dmrs_sym_loc);

                         // Manually unrolled loop over the four num_antenna_ports
                         atomicAdd(&modulation_output[output_index + all_Rbs_symbols * OFDM_SYMBOLS_PER_SLOT * 0], local_qam_output.h2[0]);
                         atomicAdd(&modulation_output[output_index + all_Rbs_symbols * OFDM_SYMBOLS_PER_SLOT * 1], local_qam_output.h2[1]);
                         atomicAdd(&modulation_output[output_index + all_Rbs_symbols * OFDM_SYMBOLS_PER_SLOT * 2], local_qam_output.h2[2]);
                         atomicAdd(&modulation_output[output_index + all_Rbs_symbols * OFDM_SYMBOLS_PER_SLOT * 3], local_qam_output.h2[3]);

                     }
                 } // end of loop over rounded_Er_elements
        } else { // number of antenna ports != 4; using local memory
                 for (int i = threadIdx.x; i < rounded_Er_elements; i+= blockDim.x) {
                     uint32_t read_val = dl_rm_shmem[i];
                     __half2 local_qam_output[MAX_DL_PORTS];

                     for (int index = 0; index < 4/Tnl; index++) {

                         for (int tmp_layer_id = 0; tmp_layer_id < Tnl; tmp_layer_id++) {
                             int intra_index = index * Tnl + tmp_layer_id;
                             uint32_t qam_value = (read_val >> (intra_index * 8)) & 0x0FFU;
                             int qams_offset = i * 4 + intra_index;

                             int x_index = map_index_8bits(qam_value);
                             int y_index = map_index_8bits(qam_value >> 1);

                             __half2 tmp_val = (qams_offset < EdivQm_bits) ? \
                                                make_complex<__half2>::create(shmem_qam_256[x_index],
                                                                              shmem_qam_256[y_index]) :
                                                make_complex<__half2>::create(0, 0);

                             for (int antenna_port = 0; antenna_port < num_antenna_ports; antenna_port++) {
                                 //Currently loading the precoding matrix from shared memory instead global
                                 //__half2 matCoeff = dmrs_params->pmW[tmp_layer_id * num_antenna_ports + antenna_port];
                                 __half2 matCoeff = shmem_matrix[tmp_layer_id * num_antenna_ports + antenna_port];
                                 __half2 new_tmp_val = mult_half2(tmp_val, matCoeff);
                                 if (tmp_layer_id == 0) {
                                     local_qam_output[antenna_port] = new_tmp_val;
                                 } else {
                                     local_qam_output[antenna_port] += new_tmp_val;
                                 }
                             }
                         }

                         int qam_in_num_qams_per_layer_per_TB = CB_start_qam_per_layer + ((4*i + index *Tnl) >> shift_bits);
                         uint32_t output_index = partial_output_index<csi_rs>(start_Rb, qam_in_num_qams_per_layer_per_TB,
                                                                              REs_per_OFDM_symbol, dmrs_params, ue_grp_params,
                                                                              data_symbol_loc, all_Rbs_symbols, d_xtf_re_map,
                                                                              modified_dmrs_sym_loc);

                         for (int antenna_port = 0; antenna_port < num_antenna_ports; antenna_port++) {
                             uint32_t output_index_port_offset = all_Rbs_symbols * antenna_port * OFDM_SYMBOLS_PER_SLOT;
                             atomicAdd(&modulation_output[output_index + output_index_port_offset], local_qam_output[antenna_port]);
                         }
                     }
                 }
        } // end of else clause for num_antenna_ports
    }
}



template<bool csi_rs=false, bool precoding=false>
__device__ void QAM256_work_Nl_3(uint32_t CB_start_qam_per_layer, int EdivQm_bits,
                                 const PdschDmrsParams* __restrict__ dmrs_params, const PdschUeGrpParams* __restrict__ ue_grp_params, int rounded_Er_elements,
                                 __half2* __restrict__ modulation_output,
                                uint16_t* __restrict__ d_xtf_re_map) {

    extern __shared__ uint32_t dl_rm_shmem[];
    const int Tnl = 3;
    int REs_per_OFDM_symbol =  dmrs_params->num_Rbs * CUPHY_N_TONES_PER_PRB;

    int all_Rbs_symbols = dmrs_params->num_BWP_PRBs * CUPHY_N_TONES_PER_PRB;
    int start_Rb = dmrs_params->start_Rb;

    uint32_t num_antenna_ports = dmrs_params->Np;

    __shared__ __half  shmem_qam_256[16];
    if (threadIdx.x < 16) {
        shmem_qam_256[threadIdx.x] = (__half) (rev_qam_256[threadIdx.x] * dmrs_params->beta_qam);
    }
    __syncthreads();
    uint64_t data_symbol_loc = dmrs_params->data_sym_loc;
    uint16_t modified_dmrs_sym_loc = dmrs_params->dmrsCdmGrpsNoData1 ? dmrs_params->dmrs_sym_loc : DMRS_CDM_GRPS_NO_DATA_NOT_1_FLAG;

    if (dmrs_params->enablePrcdBf == 0) {

        for (int i = threadIdx.x; i < rounded_Er_elements; i+= blockDim.x) {
            uint32_t read_val = dl_rm_shmem[i];
            int i_mod_3 = i % 3;

            for (int intra_index = 0; intra_index < 4; intra_index++) {
                uint32_t qam_value = (read_val >> (intra_index * 8)) & 0x0FFU;
                int tmp_layer_id = (4 * i_mod_3 + intra_index) % Tnl;
                int qams_offset = (i / 3)* 4  + (4 * i_mod_3 + intra_index)/ Tnl;
                int qam_in_num_qams_per_layer_per_TB = CB_start_qam_per_layer + qams_offset;
                if (qams_offset * Tnl <  EdivQm_bits) {
                    int layer_id = calculate_layer_id(dmrs_params->port_ids[tmp_layer_id], dmrs_params->n_scid, dmrs_params->nlAbove16);
                    uint32_t output_index = output_index_no_precoding<csi_rs>(start_Rb, qam_in_num_qams_per_layer_per_TB,
                                                                              REs_per_OFDM_symbol, dmrs_params, ue_grp_params,
                                                                              data_symbol_loc, all_Rbs_symbols, d_xtf_re_map, layer_id,
                                                                              modified_dmrs_sym_loc);
                    int x_index = map_index_8bits(qam_value);
                    int y_index = map_index_8bits(qam_value >> 1);

                    if (precoding) {
                       atomicAdd(&modulation_output[output_index], make_complex<__half2>::create(shmem_qam_256[x_index],
                                                                        shmem_qam_256[y_index]));
                    } else { // No TB in this cell has precoding enabled, so no need for atomic updates
                       modulation_output[output_index] = make_complex<__half2>::create(shmem_qam_256[x_index],
                                                                        shmem_qam_256[y_index]);
                    }
                }
            }
        }
    } else {

         for (int i = threadIdx.x; i < rounded_Er_elements; i+= blockDim.x) {
             uint32_t read_val = dl_rm_shmem[i];
             int i_mod_3 = i % 3;

             __half2 local_qam_output[MAX_DL_PORTS];

             for (int index = 0; index < 2; index++) {

                 int start_layer_id = (index == 0) ? i_mod_3 : 0;
                 int end_layer_id = (index == 1) ? i_mod_3 + 1: Tnl;
                 for (int tmp_layer_id = start_layer_id; tmp_layer_id < end_layer_id; tmp_layer_id++) {

                     int intra_index = index * Tnl + tmp_layer_id - i_mod_3;
                     uint32_t qam_value = (read_val >> (intra_index * 8)) & 0x0FFU;
                     int qam_offset = (i / 3) * 4 + index + i_mod_3;

                     __half2 tmp_val;
                     if (qam_offset * Tnl  < EdivQm_bits) {
                         int x_index = map_index_8bits(qam_value);
                         int y_index = map_index_8bits(qam_value >> 1);
                         tmp_val = make_complex<__half2>::create(shmem_qam_256[x_index],
                                                                 shmem_qam_256[y_index]);

                     } else { // Needed because I do the update in the end no matter what. So value should be 0.
                         tmp_val.x = 0;
                         tmp_val.y = 0;
                     }

                     for (int antenna_port = 0; antenna_port < num_antenna_ports; antenna_port++) {
                         __half2 matCoeff = dmrs_params->pmW[tmp_layer_id * num_antenna_ports + antenna_port];
                         __half2 new_tmp_val = mult_half2(tmp_val, matCoeff);
                         if (tmp_layer_id == start_layer_id) {
                             local_qam_output[antenna_port] = new_tmp_val;
                         } else {
                             local_qam_output[antenna_port] += new_tmp_val;
                         }
                      }
                 }

                 int qam_in_num_qams_per_layer_per_TB = CB_start_qam_per_layer + (i / 3) * 4  + index + i_mod_3;

                 uint32_t output_index = partial_output_index<csi_rs>(start_Rb, qam_in_num_qams_per_layer_per_TB,
                                                                      REs_per_OFDM_symbol, dmrs_params, ue_grp_params,
                                                                      data_symbol_loc, all_Rbs_symbols, d_xtf_re_map,
                                                                       modified_dmrs_sym_loc);

                 for (int antenna_port = 0; antenna_port < num_antenna_ports; antenna_port++) {
                     uint32_t output_index_port_offset = all_Rbs_symbols * antenna_port * OFDM_SYMBOLS_PER_SLOT;
                     atomicAdd(&modulation_output[output_index + output_index_port_offset], local_qam_output[antenna_port]);
                 }

             }
         }
    }
}


template<bool precoding>
__global__ void
__launch_bounds__(288, 4) /* previously used 3 blocks because 4 was not possible if we used -DCUPHY_GENCODE_ARCH_LIST="75". Since we're not using this, switched to 4. */
fused_dl_rm_and_modulation(dlRateMatchingDescr_t* p_desc) {

    dlRateMatchingDescr_t& desc = *p_desc;
    const PdschPerTbParams* __restrict__ cfg_workspace =  desc.cfg_workspace;

    const uint32_t TB_id = blockIdx.y;
    const uint32_t CB_id = blockIdx.x;

    const PdschPerTbParams * TB_params = &cfg_workspace[TB_id];
    const uint8_t test_model = TB_params->testModel;
    const uint32_t num_CBs = TB_params->num_CBs;
    if (CB_id >= num_CBs) { // Exit early if code block does not exist for TB_id TB
        return;
    }
    //const uint32_t* __restrict__ ldpc_encoder_output = desc.d_rate_matching_input;
    //const uint32_t* __restrict__ Er_array = desc.d_Er_array;
    const uint32_t* __restrict__ k0_array = desc.d_k0_array;
    const bool enable_scrambling = desc.enable_scrambling;
    const bool enable_layer_mapping = desc.enable_layer_mapping;
    const uint32_t cmax = desc.cmax;
    const uint32_t emax = desc.emax;

    const int ELEMENT_SIZE = 32; // sizeof(uint32_t) * 8; // in bits
    const int ELEMENT_BITS = 5; // log2(ELEMENT_SIZE)
    const int ELEMENT_MASK = ELEMENT_SIZE - 1;

    const uint32_t TB_start = desc.d_TB_start_offset_array[TB_id]; //w.r.t ldpc_encoder_output
    //if ((threadIdx.x == 0) && (CB_id == 0)) printf("TB id %d has TB_start %d\n", TB_id, TB_start);
    //const uint32_t rv = TB_params->rv;
    const uint32_t Qm_val = TB_params->Qm;
    //const uint32_t tmp_Qm_val = (test_model != 0) ? 1 : TB_params->Qm;
    const uint32_t num_layers = TB_params->Nl; // Reminder can be [1, MAX_DL_LAYERS_PER_TB=4]

    const uint32_t F_val = TB_params->F;
    const int k0 = k0_array[TB_id]; // Maybe set it to 0 if rv == 0?
    //const int Kd =  TB_params->K - (TB_params->Zc << 1) - F_val;
    uint32_t rounded_N = round_up_to_next((int)TB_params->N, ELEMENT_SIZE); // N should be divisible by 32 for LDPC encoder's output.
    uint32_t CB_start_input = (TB_start + CB_id * rounded_N) >> ELEMENT_BITS;
    const uint32_t* __restrict__ CB_ldpc_encoder_output = desc.d_rate_matching_input + CB_start_input;

    /* Er_array has 2 elements per TB.
       - Er_array[TB_id * 2 + 0] holds the CB id of the Er split point (this TB's num_CBs is none)
       - Er_array[TB_id * 2 + 1] holds the min Er in bits for that TB. The max Er, for all CBs >= split point
         is min Er + num_layers * modulation order bits.
    */
#if 0
    int CB_split_id = Er_array[TB_id * 2];
    bool before_CB_split_point = (CB_id < CB_split_id);
    const int CB_Er = Er_array[TB_id * 2 + 1] + ((before_CB_split_point) ? 0 : num_layers * Qm_val);
#else
    int ue_grp_idx = desc.d_params[TB_id].ueGrp_idx;
    int num_data_symbols = desc.d_params[TB_id].num_data_symbols;

    int csirs_rs_RE_cnt = desc.d_ue_grp_params[ue_grp_idx].cumulative_skipped_REs[num_data_symbols - 1]; // cumulative_skipped_REs computed as part of postProcessCsirsReMap kernel from cuphyRunPdschCsirsPreprocessing during PDSCH setup
    int per_UE_REs = TB_params->max_REs - csirs_rs_RE_cnt;
    int G = per_UE_REs * num_layers * Qm_val; // for TBs in TM this is almost equivalent to tbSize*8 (G/8 rounded up is tbSize); reminder no CSI-RS present for these cells.
    int quotient_C = floorf(per_UE_REs * (1.0f/num_CBs));
    int modulo_C = (test_model == 0) ? (per_UE_REs - quotient_C * num_CBs) : 0; //split, if any, at C - modulo_C CB
    int CB_split_id = num_CBs - modulo_C;
    int after_CB_split_point = (CB_id >= CB_split_id) ? 1 : 0;
    //const int CB_Er = (test_model == 0) ? ((quotient_C + after_CB_split_point) * num_layers * Qm_val): ((CB_id == num_CBs-1) ? (tbSize*8 - CB_id *rounded_N) : TB_params->N);
    const int CB_Er = (test_model == 0) ? ((quotient_C + after_CB_split_point) * num_layers * Qm_val): ((CB_id == num_CBs-1) ? (G - CB_id*MAX_ENCODED_CODE_BLOCK_BIT_SIZE) : MAX_ENCODED_CODE_BLOCK_BIT_SIZE);
#endif
    // don't care for Kd if I bypass selection/interleaving
    //const int Kd =  (test_model != 0) ? CB_Er : TB_params->K - (TB_params->Zc << 1) - F_val;
    const int Kd =  TB_params->K - (TB_params->Zc << 1) - F_val;

    //if (threadIdx.x == 0) printf("TB %d, CB %d, CB_split_id %d, after_CB_split_point %d, per_UE_REs %d, G %d, num_CBs %d, CB_Er %d\n", TB_id, CB_id, CB_split_id, after_CB_split_point, per_UE_REs, G, num_CBs, CB_Er);

    const int rounded_Er_elements = emax >> ELEMENT_BITS;
    int EdivQm_bits = 0;
    int maxNdivQm_bits = 0;
    // For TBs in cells in testing mode, CB_Er will be MAX_ENCODED_CODE_BLOCK_BIT_SIZE=25344 for TB_id in [0, num_CBs-2] and G - (num_CBs-1)*25344 for TB_id=num_CBs-1.
    // For those TBs, no bit selection/ interleaving shall take place. This is currently achieved by enforcing EdivQm_bits == CB_Er, i.e., as if Qm_val was 1.
    if (Qm_val == CUPHY_QAM_4) {
        EdivQm_bits = CB_Er >> 1;
        maxNdivQm_bits = MAX_ENCODED_CODE_BLOCK_BIT_SIZE >> 1;
    } else if (Qm_val ==  CUPHY_QAM_16) {
        EdivQm_bits = CB_Er >> 2;
        maxNdivQm_bits = MAX_ENCODED_CODE_BLOCK_BIT_SIZE >> 2;
    } else if (Qm_val ==  CUPHY_QAM_64) {
        EdivQm_bits = CB_Er / 6;
        maxNdivQm_bits = MAX_ENCODED_CODE_BLOCK_BIT_SIZE / 6;
    } else if (Qm_val ==  CUPHY_QAM_256) {
        EdivQm_bits = CB_Er >> 3;
        maxNdivQm_bits = MAX_ENCODED_CODE_BLOCK_BIT_SIZE >> 3;
    }
    int orig_EdivQm_bits = EdivQm_bits;

    if (test_model != 0) EdivQm_bits = CB_Er;

    /* dynamic shared memory organized as follows:
       - The first rounded_Er_elements (if no layer mapping) or MAX_DL_LAYERS_PER_TB * rounded_Er_element (layer mapping)
         are used to minimize the overhead of atomicOr operations
       - The following rounded_Er_elements + 1 (if scrambling) are used to keep track of the scrambling sequence values.*/
    extern __shared__ uint32_t dl_rm_shmem[];
    __shared__ __half2 shmem_matrix[MAX_DL_LAYERS_PER_TB*MAX_DL_PORTS]; //shared memory for the precoding matrix; using max possible dimensions

    const int scale = (enable_layer_mapping) ? MAX_DL_LAYERS_PER_TB : 1;
    uint32_t * CB_scrambling_vals = (uint32_t*)&dl_rm_shmem[rounded_Er_elements * scale];

    uint32_t gold32_CB_start_output = 0; // CB start (rate-matched within a TB)

#if 0
    for (int i = threadIdx.x; i < scale * rounded_Er_elements; i += blockDim.x) {
        dl_rm_shmem[i] = 0;
    }
#else
    const int limit = ((int)scale * rounded_Er_elements + 3) >> 2;
    int4* dl_rm_shmem_int4 = (int4*)dl_rm_shmem;
    for (int i = threadIdx.x; i < limit; i += blockDim.x) {
        dl_rm_shmem_int4[i] = {0, 0, 0, 0};
    }
#endif

    // Copying precoding matrix from global to shared memory.
    // For this TB, the precoding matrix, if one exists has size num_layers * desc.d_params[TB_id].Np.
    // If this TB has no precoding enabled (i.e., desc.d_params[TB_id].enablePrcdBf == 0), the shared memory contents will be invalid
    // (but still no illegal memory access).
    for (int i = threadIdx.x; i < MAX_DL_PORTS * MAX_DL_LAYERS_PER_TB; i += blockDim.x) {
        __half2* src_ptr = (desc.d_params[TB_id].pmW);
        __pipeline_memcpy_async(shmem_matrix+i, src_ptr+i, sizeof(__half2));
    }
    __pipeline_commit();


    //Only if scrambling/layer mapping
    uint32_t CB_start_qam = 0;
    if (after_CB_split_point == 0) {
        // For TBs in cells in testing mode, all CBs will be before the split point.
        // Cannot use CB_Er, as it will have a different value than 25344 for last CB. Instead of using rounded_N, use 25344 directly
        if (test_model == 0) {
            gold32_CB_start_output = CB_id * CB_Er;
            CB_start_qam = CB_id * orig_EdivQm_bits; //using EdivQm_bits is ok too
        } else {
            gold32_CB_start_output = CB_id * MAX_ENCODED_CODE_BLOCK_BIT_SIZE;
            CB_start_qam = CB_id * maxNdivQm_bits;
        }
    } else { // after the split
        //gold32_CB_start_output = TB_params->G - ((num_CBs - CB_id) * CB_Er);
        gold32_CB_start_output = G - ((num_CBs - CB_id) * CB_Er);
        CB_start_qam = CB_id * EdivQm_bits - CB_split_id * num_layers;
    }

    if (enable_scrambling) {
        const uint32_t cinit_val = TB_params->cinit;
        for (int i = threadIdx.x; i <= rounded_Er_elements; i += blockDim.x) {
	       // Heads up: if 2nd gold32 argument isn't divisible by 32, the gold32 returns the sequence 2nd_arg // 32. Offset it in tmp_final_index below.
	       // For the same reason, CB_scrambling_vals needs to have (rounded_Er_elements + 1) elements
	       CB_scrambling_vals[i] = gold32(cinit_val, gold32_CB_start_output + (i << ELEMENT_BITS));
        }
    }
    __syncthreads();

    if (test_model == 0) {

#if 1
        const bool buffer_case = (Kd < k0);
        const int first_pass_bits =  TB_params ->Ncb - k0 - ((Kd < k0) ? 0 : F_val);
        bool only_first_pass = (CB_Er <= first_pass_bits);
        bool CB_Er_alignment = (((CB_Er >> 3) & 0x3) == 0); // only relevant for 256-QAM for now
        bool special_case = (only_first_pass && (k0 == 0) && CB_Er_alignment);
#else
        //const uint32_t N_minus_F = TB_params->N - F_val;
        int buffer_case = 0;
        int first_pass_bits = 0;
        // It is not clear if buffer_case = 2 is possible. For N != Ncb, it could only be possible
        // for rv = 1 and BG=1 if F > 3*Zc. TBD if this is permitted by the spec.

        if (Kd >= k0) {
            buffer_case = 0;
            //if (test_model == 0) {
                first_pass_bits = TB_params->Ncb - (F_val + k0);
            //} else {
            //    first_pass_bits = EdivQm_bits; // FIXME k0 is 0, could force F_val to 0 and ensure Ncb = EdivQm_bits?
            //}
        } /*else if ((k0 >= Kd + F_val)) {
            buffer_case = 1;
            first_pass_bits = TB_params->N - k0;
        } else {
            buffer_case = 2;
            first_pass_bits = TB_params->N - (F_val + Kd);
        }*/
        else {
            buffer_case = 1;
            first_pass_bits = TB_params->Ncb - k0;
        }
#endif

        // Each thread block is working on a CB. Each thread on Er/(32 * blkDimX) distinct elements
        for (int i = threadIdx.x; i < rounded_Er_elements; i += blockDim.x) {

            int index = i << ELEMENT_BITS;
            int read_index = index + k0;

            uint32_t element_read = (index >= first_pass_bits) ? 0 :  CB_ldpc_encoder_output[(read_index >> ELEMENT_BITS)]; // FIXME

            int EdivQm_block_id = index / EdivQm_bits;
            int EdivQm_bit_id = index - (EdivQm_bits * EdivQm_block_id);

            if ((Qm_val == CUPHY_QAM_256) && special_case) {
                // Special handling for a subset of 256-QAM cases for perf. reasons. When alignment and other conditions are favorable, avoid doing bit-level atomicOr to shared memory
                // The special_case conditions are: redundancy version 0 (k0==0), no wrap around during bit selection, and CB_Er in bytes is divisible by 4
                // TODO this same principle can be extended to all modulation orders (Qm). Handling of cases without the 32-bit alignment would require additional work.

                for (int bit_id = 0; bit_id < ELEMENT_SIZE; bit_id+= 4) {
                    int new_bit_index = bit_id;
                    if (index < CB_Er) {
                        if ((index < first_pass_bits) && (buffer_case == 0) && (index + k0 < Kd)) {
                            uint32_t four_bits_read = (element_read >> new_bit_index);
                            // Bit interleaving
                            int final_index = (EdivQm_bit_id * Qm_val) + EdivQm_block_id; // bit_index within CB_Er block.
                            uint32_t val = 0;
                            val |= (four_bits_read & 0x1ULL) <<  (final_index & ELEMENT_MASK);
                            val |= ((four_bits_read >> 1)& 0x1ULL) <<  ((final_index + 1*Qm_val) & ELEMENT_MASK);
                            val |= ((four_bits_read >> 2)& 0x1ULL) <<  ((final_index + 2*Qm_val) & ELEMENT_MASK);
                            val |= ((four_bits_read >> 3)& 0x1ULL) <<  ((final_index + 3*Qm_val) & ELEMENT_MASK);
                            atomicOr(&dl_rm_shmem[(final_index >> ELEMENT_BITS)], val);
                        }
                        else if ((index < first_pass_bits) && (buffer_case == 0) && (index + k0 >= Kd)) {

                            read_index = index + k0 + F_val;
                            element_read = CB_ldpc_encoder_output[(read_index >> ELEMENT_BITS)]; // not optimal
                            new_bit_index = read_index & ELEMENT_MASK;
                            uint32_t four_bits_read = (element_read >> new_bit_index);
                            int final_index = (EdivQm_bit_id * Qm_val) + EdivQm_block_id; // bit_index within CB_Er block.

                            uint32_t val = 0;
                            val |= (four_bits_read & 0x1ULL) <<  (final_index & ELEMENT_MASK);
                            val |= ((four_bits_read >> 1)& 0x1ULL) <<  ((final_index + Qm_val) & ELEMENT_MASK);
                            val |= ((four_bits_read >> 2)& 0x1ULL) <<  ((final_index + 2*Qm_val) & ELEMENT_MASK);
                            val |= ((four_bits_read >> 3)& 0x1ULL) <<  ((final_index + 3*Qm_val) & ELEMENT_MASK);
                            atomicOr(&dl_rm_shmem[(final_index >> ELEMENT_BITS)], val);
                        }
                    }
                    read_index += 4;
                    index += 4;
                    if (EdivQm_bit_id == EdivQm_bits - 4) {
                        EdivQm_bit_id = 0;
                        EdivQm_block_id += 1;
                    } else {
                        EdivQm_bit_id += 4;
                    }
                }
            } else {
                // Fallback for all Qm order other than 256-QAM and for 256-QAM !special_case cases
                #pragma unroll 8
                for (int bit_id = 0; bit_id < ELEMENT_SIZE; bit_id++) {

                    if (index < CB_Er) {

                        // Bit selection
                        int new_bit_index = bit_id;
                        if (index < first_pass_bits) {
                            if ((buffer_case == 0) && (index + k0 >= Kd)) { // Needed to skip filler bits. Only possible for rv=0 or rv=1
                                read_index = index + k0 + F_val;
                                element_read = CB_ldpc_encoder_output[(read_index >> ELEMENT_BITS)]; // not optimal
                                new_bit_index = read_index & ELEMENT_MASK;
                            }
                            else if ((k0 & ELEMENT_MASK) != 0) { // N/A for rv=0
                                read_index = index + k0;
                                element_read = CB_ldpc_encoder_output[(read_index >> ELEMENT_BITS)]; // not optimal
                                new_bit_index = read_index & ELEMENT_MASK;
                            }
                        } else { // This is applicable in all cases
                            read_index = (index - first_pass_bits) % (TB_params->Ncb - F_val);
                            if (read_index >= Kd) read_index += F_val;
                            element_read = CB_ldpc_encoder_output[(read_index >> ELEMENT_BITS)]; // not optimal
                            new_bit_index = read_index & ELEMENT_MASK;
                        }
                        uint32_t bit_read = (element_read >> new_bit_index);

                        // Bit interleaving
                        //int final_index = (EdivQm_bit_id * tmp_Qm_val) + EdivQm_block_id; // bit_index within CB_Er block.
                        int final_index = (EdivQm_bit_id * Qm_val) + EdivQm_block_id; // bit_index within CB_Er block.
                        bit_read = (bit_read & 0x1ULL) <<  (final_index & ELEMENT_MASK);
                        atomicOr(&dl_rm_shmem[(final_index >> ELEMENT_BITS)], bit_read); // not optimal - todo
                    }
                    read_index += 1;
                    index += 1;
                    if (EdivQm_bit_id == EdivQm_bits - 1) {
                        EdivQm_bit_id = 0;
                        EdivQm_block_id += 1;
                    } else {
                        EdivQm_bit_id += 1;
                    }
                }
            } // end of generic else clause
        }  // end of for loop over rounded_Er_elements
    } else {
        // For TBs in TM, we can avoid bit selection and interleaving instead of doing it at 1-bit granularity
        for (int i = threadIdx.x; i < ((CB_Er+31) >> 5); i += blockDim.x) {
            dl_rm_shmem[i] = CB_ldpc_encoder_output[i];
        }
    }
    __syncthreads();
#if 0
if (threadIdx.x == 0) {
    for (int k = 0; k < ((CB_Er + 31) >> 5); k++)
    printf("TB %d, CB %d, dl_rm[%d] = %x\n", TB_id, CB_id, k, dl_rm_shmem[k]);
}
#endif
    uint32_t* __restrict__ rm_output = desc.d_rate_matching_output;
    const uint32_t G_start = TB_id * cmax * emax; // start of TB in output buffer in bits
    uint32_t CB_start_output = (G_start + CB_id * emax) >> ELEMENT_BITS;

    uint32_t CB_remainder = gold32_CB_start_output & ELEMENT_MASK;
    uint32_t scrambling_val = 0;
    for (int i = threadIdx.x; i < rounded_Er_elements; i+= blockDim.x) {
        if (enable_scrambling) {
            if (CB_remainder == 0) {
                 scrambling_val  = CB_scrambling_vals[i];
            } else {
                 scrambling_val  = ((CB_scrambling_vals[i] >> CB_remainder) | ((CB_scrambling_vals[i+1]) << (32 - CB_remainder)));
            }
        }
        if (!enable_layer_mapping) {
            rm_output[CB_start_output + i] = (dl_rm_shmem[i] ^ scrambling_val);
        } else {
            dl_rm_shmem[i] = dl_rm_shmem[i] ^ scrambling_val;
        }
    }
    if (!enable_layer_mapping) return;
    // There's a syncthreads in the QAM*_work* device functions below


    // At this point we have scrambled but not layer mapped values in shmem
    // We know each CB_Er is evenly divisible by Qam and also Nl
    // Go from symbol within a CB to (layer, symbol, RE)
    // Every CB contributes either a  or (a+1) symbols per layer, where a = CB_Er_0/ (Qm_val * Nl_val)
    // Can find start symbol each CB contributes to.

    uint32_t CB_start_qam_per_layer = 0;
/*
    if ((threadIdx.x == 0) && (blockIdx.x == 0) && (blockIdx.y == 0)) {
        for (int k = 42835; k < 42864; k++) printf("xtf_re_map[%d] = %d\n", k, desc.temp_xtf_re_map[k]);
    }
*/
    //int num_data_symbols = desc.d_params[TB_id].num_data_symbols;
    //bool csi_rs = (desc.d_params[TB_id].cumulative_skipped_REs[num_data_symbols - 1] != 0);
    //int ue_grp_idx = desc.d_params[TB_id].ueGrp_idx;
    bool csi_rs = (desc.d_ue_grp_params[ue_grp_idx].cumulative_skipped_REs[num_data_symbols - 1] != 0);
    __half2* modulation_output = (__half2*)desc.d_params[TB_id].cell_output_tensor_addr;

    // RE map recently extended by OFDM_SYMBOLS_PER_SLOT
    uint16_t* cell_xtf_re_map = (uint16_t*)desc.temp_xtf_re_map + desc.d_params[TB_id].cell_index_in_cell_group * (desc.max_PRB_BWP * OFDM_SYMBOLS_PER_SLOT * CUPHY_N_TONES_PER_PRB + OFDM_SYMBOLS_PER_SLOT);

    if (num_layers == 1) {
        CB_start_qam_per_layer = CB_start_qam;
        //if (threadIdx.x == 0) printf("TB %d CB %d, EdivQm_bits %d, rounded_Er_elements %d, CB_start_qam_per_layer %d, rounded_Er_elements %d\n", TB_id, CB_id, EdivQm_bits, rounded_Er_elements, CB_start_qam_per_layer, rounded_Er_elements);
        if (Qm_val == CUPHY_QAM_4) {
            if (csi_rs) {
                QAM4_work_all_Nl_but_3<1, true, precoding>(CB_start_qam_per_layer, EdivQm_bits,
                                                &desc.d_params[TB_id], &desc.d_ue_grp_params[ue_grp_idx], rounded_Er_elements, modulation_output, cell_xtf_re_map);
            } else {
                QAM4_work_all_Nl_but_3<1, false, precoding>(CB_start_qam_per_layer, orig_EdivQm_bits,
                                                 &desc.d_params[TB_id], &desc.d_ue_grp_params[ue_grp_idx], rounded_Er_elements, modulation_output, cell_xtf_re_map);
            }
        } else if (Qm_val == CUPHY_QAM_16) {
            if (csi_rs) {
                QAM16_work_all_Nl_but_3<1, true, precoding>(CB_start_qam_per_layer, EdivQm_bits,
                                                 &desc.d_params[TB_id], &desc.d_ue_grp_params[ue_grp_idx], rounded_Er_elements, modulation_output, cell_xtf_re_map);
            } else {
                QAM16_work_all_Nl_but_3<1, false, precoding>(CB_start_qam_per_layer, orig_EdivQm_bits,
                                                  &desc.d_params[TB_id], &desc.d_ue_grp_params[ue_grp_idx], rounded_Er_elements, modulation_output, cell_xtf_re_map);
            }
        } else if (Qm_val == CUPHY_QAM_64) {
            if (csi_rs) {
                QAM64_work_all_Nl_but_3<1, true, precoding>(CB_start_qam_per_layer, EdivQm_bits,
                                                 &desc.d_params[TB_id], &desc.d_ue_grp_params[ue_grp_idx], rounded_Er_elements, modulation_output, cell_xtf_re_map);
            } else {
                QAM64_work_all_Nl_but_3<1, false, precoding>(CB_start_qam_per_layer, orig_EdivQm_bits,
                                                  &desc.d_params[TB_id], &desc.d_ue_grp_params[ue_grp_idx], rounded_Er_elements, modulation_output, cell_xtf_re_map);
            }
        } else if (Qm_val == CUPHY_QAM_256) {
            if (csi_rs) {
                QAM256_work_all_Nl_but_3<1, true, precoding>(CB_start_qam_per_layer, EdivQm_bits,
                                                  &desc.d_params[TB_id], &desc.d_ue_grp_params[ue_grp_idx], rounded_Er_elements, modulation_output, cell_xtf_re_map, shmem_matrix);
            } else {
                QAM256_work_all_Nl_but_3<1, false, precoding>(CB_start_qam_per_layer, orig_EdivQm_bits,
                                                   &desc.d_params[TB_id], &desc.d_ue_grp_params[ue_grp_idx], rounded_Er_elements, modulation_output, cell_xtf_re_map, shmem_matrix);
            }
        }
    } else if (num_layers == 2) {
        CB_start_qam_per_layer = CB_start_qam >> 1;
        if (Qm_val == CUPHY_QAM_4) {
            if (csi_rs) {
                QAM4_work_all_Nl_but_3<2, true, precoding>(CB_start_qam_per_layer, EdivQm_bits,
                                                  &desc.d_params[TB_id], &desc.d_ue_grp_params[ue_grp_idx], rounded_Er_elements, modulation_output, cell_xtf_re_map);
            } else {
                QAM4_work_all_Nl_but_3<2, false, precoding>(CB_start_qam_per_layer, orig_EdivQm_bits,
                                                 &desc.d_params[TB_id], &desc.d_ue_grp_params[ue_grp_idx], rounded_Er_elements, modulation_output, cell_xtf_re_map);
            }
        } else if (Qm_val == CUPHY_QAM_16) {
            if (csi_rs) {
                QAM16_work_all_Nl_but_3<2, true, precoding>(CB_start_qam_per_layer, EdivQm_bits,
                                                 &desc.d_params[TB_id], &desc.d_ue_grp_params[ue_grp_idx], rounded_Er_elements, modulation_output, cell_xtf_re_map);
            } else {
                QAM16_work_all_Nl_but_3<2, false, precoding>(CB_start_qam_per_layer, orig_EdivQm_bits,
                                                  &desc.d_params[TB_id], &desc.d_ue_grp_params[ue_grp_idx], rounded_Er_elements, modulation_output, cell_xtf_re_map);
            }
        } else if (Qm_val == CUPHY_QAM_64) {
            if (csi_rs) {
                QAM64_work_all_Nl_but_3<2, true, precoding>(CB_start_qam_per_layer, EdivQm_bits,
                                                 &desc.d_params[TB_id], &desc.d_ue_grp_params[ue_grp_idx], rounded_Er_elements, modulation_output, cell_xtf_re_map);
            } else {
                QAM64_work_all_Nl_but_3<2, false, precoding>(CB_start_qam_per_layer, orig_EdivQm_bits,
                                                  &desc.d_params[TB_id], &desc.d_ue_grp_params[ue_grp_idx], rounded_Er_elements, modulation_output, cell_xtf_re_map);
            }
        } else if (Qm_val == CUPHY_QAM_256) {
            if (csi_rs) {
                QAM256_work_all_Nl_but_3<2, true, precoding>(CB_start_qam_per_layer, EdivQm_bits,
                                                  &desc.d_params[TB_id], &desc.d_ue_grp_params[ue_grp_idx], rounded_Er_elements, modulation_output, cell_xtf_re_map, shmem_matrix);
            } else {
                QAM256_work_all_Nl_but_3<2, false, precoding>(CB_start_qam_per_layer, orig_EdivQm_bits,
                                                   &desc.d_params[TB_id], &desc.d_ue_grp_params[ue_grp_idx],  rounded_Er_elements, modulation_output, cell_xtf_re_map, shmem_matrix);
            }
        }
    } else if (num_layers == 3) {
        CB_start_qam_per_layer = CB_start_qam / 3;
        if (Qm_val == CUPHY_QAM_4) {
            if (csi_rs) {
                QAM4_work_Nl_3<true, precoding>(CB_start_qam_per_layer, EdivQm_bits,
                                     &desc.d_params[TB_id], &desc.d_ue_grp_params[ue_grp_idx], rounded_Er_elements, modulation_output, cell_xtf_re_map);
            } else {
                QAM4_work_Nl_3<false, precoding>(CB_start_qam_per_layer, EdivQm_bits,
                                      &desc.d_params[TB_id], &desc.d_ue_grp_params[ue_grp_idx], rounded_Er_elements, modulation_output, cell_xtf_re_map);
            }
        } else if (Qm_val == CUPHY_QAM_16) {
            if (csi_rs) {
                QAM16_work_Nl_3<true, precoding>(CB_start_qam_per_layer, EdivQm_bits,
                                     &desc.d_params[TB_id], &desc.d_ue_grp_params[ue_grp_idx], rounded_Er_elements, modulation_output, cell_xtf_re_map);
            } else {
                QAM16_work_Nl_3<false, precoding>(CB_start_qam_per_layer, EdivQm_bits,
                                     &desc.d_params[TB_id], &desc.d_ue_grp_params[ue_grp_idx], rounded_Er_elements, modulation_output, cell_xtf_re_map);
            }
        } else if (Qm_val == CUPHY_QAM_64) {
            if (csi_rs) {
                QAM64_work_Nl_3<true, precoding>(CB_start_qam_per_layer, EdivQm_bits,
                                     &desc.d_params[TB_id], &desc.d_ue_grp_params[ue_grp_idx], rounded_Er_elements, modulation_output, cell_xtf_re_map);
            } else {
                QAM64_work_Nl_3<false, precoding>(CB_start_qam_per_layer, EdivQm_bits,
                                     &desc.d_params[TB_id], &desc.d_ue_grp_params[ue_grp_idx], rounded_Er_elements, modulation_output, cell_xtf_re_map);
            }
        } else if (Qm_val == CUPHY_QAM_256) {
            if (csi_rs) {
                QAM256_work_Nl_3<true, precoding>(CB_start_qam_per_layer, EdivQm_bits,
                                     &desc.d_params[TB_id], &desc.d_ue_grp_params[ue_grp_idx], rounded_Er_elements, modulation_output, cell_xtf_re_map);
            } else {
                QAM256_work_Nl_3<false, precoding>(CB_start_qam_per_layer, EdivQm_bits,
                                     &desc.d_params[TB_id], &desc.d_ue_grp_params[ue_grp_idx], rounded_Er_elements, modulation_output, cell_xtf_re_map);
            }
        }
    } else if (num_layers == 4) {
        CB_start_qam_per_layer = CB_start_qam >> 2;
        if (Qm_val == CUPHY_QAM_4) {
            if (csi_rs) {
                QAM4_work_all_Nl_but_3<4, true, precoding>(CB_start_qam_per_layer, EdivQm_bits,
                                     &desc.d_params[TB_id], &desc.d_ue_grp_params[ue_grp_idx], rounded_Er_elements, modulation_output, cell_xtf_re_map);
            } else {
                QAM4_work_all_Nl_but_3<4, false, precoding>(CB_start_qam_per_layer, EdivQm_bits,
                                     &desc.d_params[TB_id], &desc.d_ue_grp_params[ue_grp_idx], rounded_Er_elements, modulation_output, cell_xtf_re_map);
            }
        } else if (Qm_val == CUPHY_QAM_16) {
            if (csi_rs) {
                QAM16_work_all_Nl_but_3<4, true, precoding>(CB_start_qam_per_layer, EdivQm_bits,
                                     &desc.d_params[TB_id], &desc.d_ue_grp_params[ue_grp_idx], rounded_Er_elements, modulation_output, cell_xtf_re_map);
            } else {
                QAM16_work_all_Nl_but_3<4, false, precoding>(CB_start_qam_per_layer, EdivQm_bits,
                                     &desc.d_params[TB_id], &desc.d_ue_grp_params[ue_grp_idx], rounded_Er_elements, modulation_output, cell_xtf_re_map);
            }
        } else if (Qm_val == CUPHY_QAM_64) {
            if (csi_rs) {
                 QAM64_work_all_Nl_but_3<4, true, precoding>(CB_start_qam_per_layer, EdivQm_bits,
                                     &desc.d_params[TB_id], &desc.d_ue_grp_params[ue_grp_idx], rounded_Er_elements, modulation_output, cell_xtf_re_map);
            } else {
                 QAM64_work_all_Nl_but_3<4, false, precoding>(CB_start_qam_per_layer, EdivQm_bits,
                                     &desc.d_params[TB_id], &desc.d_ue_grp_params[ue_grp_idx], rounded_Er_elements, modulation_output, cell_xtf_re_map);
            }
        } else if (Qm_val == CUPHY_QAM_256) {
            if (csi_rs) {
                QAM256_work_all_Nl_but_3<4, true, precoding>(CB_start_qam_per_layer, EdivQm_bits,
                                     &desc.d_params[TB_id], &desc.d_ue_grp_params[ue_grp_idx], rounded_Er_elements, modulation_output, cell_xtf_re_map, shmem_matrix);
            } else {
                QAM256_work_all_Nl_but_3<4, false, precoding>(CB_start_qam_per_layer, EdivQm_bits,
                                     &desc.d_params[TB_id], &desc.d_ue_grp_params[ue_grp_idx], rounded_Er_elements, modulation_output, cell_xtf_re_map, shmem_matrix);
            }
        }
    }
}


/* Kernel that processes num_TBs transport blocks: rate_matching, scrambling and layer_mapping */
__global__ void dl_rate_matching(dlRateMatchingDescr_t* p_desc) {

    dlRateMatchingDescr_t& desc = *p_desc;
    const PdschPerTbParams* __restrict__ cfg_workspace =  desc.cfg_workspace;

    const uint32_t TB_id = blockIdx.y;
    const uint32_t CB_id = blockIdx.x;

    const PdschPerTbParams * TB_params = &cfg_workspace[TB_id];
    const uint8_t test_model = TB_params->testModel;
    const uint32_t num_CBs = TB_params->num_CBs;
    if (CB_id >= num_CBs) { // Exit early if code block does not exist for TB_id TB
        return;
    }

    const uint32_t* __restrict__ ldpc_encoder_output = desc.d_rate_matching_input;
    uint32_t* rate_matching_output = desc.d_rate_matching_output;
    const uint32_t num_TBs = desc.num_TBs;
    const uint32_t* __restrict__ Er_array = desc.d_Er_array;
    const uint32_t* __restrict__ k0_array = desc.d_k0_array;
    const uint32_t* __restrict__ TB_start_offset_array = desc.d_TB_start_offset_array;
    bool enable_scrambling = desc.enable_scrambling;
    bool enable_layer_mapping = desc.enable_layer_mapping;
    const uint32_t emax = desc.emax;
    const uint32_t cmax = desc.cmax;

    const int ELEMENT_SIZE = 32; // sizeof(uint32_t) * 8; // in bits
    const int ELEMENT_BITS = 5; // log2(ELEMENT_SIZE)
    const int ELEMENT_MASK = ELEMENT_SIZE - 1;

    const uint32_t emax_uints = emax >> ELEMENT_BITS;

    uint32_t TB_start = TB_start_offset_array[TB_id]; //w.r.t ldpc_encoder_output
    uint32_t rounded_N = round_up_to_next((int)TB_params->N, ELEMENT_SIZE); // N should be divisible by 32 for LDPC encoder's output.
    uint32_t CB_start_input = TB_start + CB_id * rounded_N; // in bits
    uint32_t Qm_val = TB_params->Qm;
    uint32_t Nl_val = TB_params->Nl;
    bool before_CB_split_point = (CB_id < Er_array[TB_id * 2]);
    const int Er = Er_array[TB_id * 2 + 1] + ((before_CB_split_point) ? 0 : Nl_val * Qm_val);

    uint32_t F_val = TB_params->F;
    const int k0 = k0_array[TB_id]; // Maybe set it to 0 if rv == 0?
    int Kd =  (test_model != 0) ? Er : TB_params->K - (TB_params->Zc << 1) - F_val;
    CB_start_input = CB_start_input >> ELEMENT_BITS;
    uint32_t G_start = TB_id * cmax * emax; // start of TB in output buffer in bits
    //uint32_t rv = TB_params->rv;
    uint32_t cinit_val = TB_params->cinit;
    float rev_Nl_val = (1.0f / (Nl_val & 0x1f));

    int rounded_Er_elements = emax_uints;
    int EdivQm_bits = 0;
    if (Qm_val == CUPHY_QAM_4) {
        EdivQm_bits = Er >> 1;
    } else if (Qm_val ==  CUPHY_QAM_16) {
        EdivQm_bits = Er >> 2;
    } else if (Qm_val ==  CUPHY_QAM_64) {
        EdivQm_bits = Er / 6;
    } else if (Qm_val ==  CUPHY_QAM_256) {
        EdivQm_bits = Er >> 3;
    }

    if (test_model != 0) EdivQm_bits = (CB_id == num_CBs-1) ? TB_params->G - CB_id*rounded_N: Er;
    int tmp_Er = (test_model != 0) ? EdivQm_bits : Er;
    //if ((blockIdx.x == 0) && (threadIdx.x == 0)) printf("TB %d, emax %d, cmax %d, TB_start %d, rounded_N %d, Er %d, emax_uints %d, EdivQm_bits %d \n", TB_id, emax,  cmax, TB_start, rounded_N, Er, emax_uints, EdivQm_bits);

    //if (threadIdx.x == 0) printf("TB %d, CB %d: rv %d, k0 %d, kd %d, N %d, Er %d\n", TB_id, CB_id, rv, k0, Kd, (int)TB_params->N, Er);

    const uint8_t * this_TB_port_ids_array =  desc.d_params[TB_id].port_ids;
    const uint8_t  n_scid = desc.d_params[TB_id].n_scid;
    const uint8_t  nlAbove16 = desc.d_params[TB_id].nlAbove16;

    /* dynamic shared memory organized as follows:
       - The first rounded_Er_elements (if no layer mapping) or MAX_DL_LAYERS_PER_TB * rounded_Er_element (layer mapping)
         are used to minimize the overhead of atomicOr operations
       - The following rounded_Er_elements + 1 (if scrambling) are used to keep track of the scrambling sequence values.*/
    extern __shared__ uint32_t dl_rm_shmem[];

    const int scale = (enable_layer_mapping) ? MAX_DL_LAYERS_PER_TB : 1;
    uint32_t * CB_scrambling_vals = (uint32_t*)&dl_rm_shmem[rounded_Er_elements * scale];

        uint32_t CB_start_output = (G_start + CB_id * emax) >> ELEMENT_BITS;
        uint32_t gold32_CB_start_output = 0; // CB start (rate-matched within a TB)
#if 0
        if (enable_layer_mapping) {
            for (int i = threadIdx.x; i < rounded_Er_elements * MAX_DL_LAYERS_PER_TB; i += blockDim.x) {
	           dl_rm_shmem[i] = 0;
            }
        } else {
            for (int i = threadIdx.x; i < rounded_Er_elements; i += blockDim.x) {
	           dl_rm_shmem[i] = 0;
            }
        }
#else
        for (int i = threadIdx.x; i < scale * rounded_Er_elements; i += blockDim.x) {
	           dl_rm_shmem[i] = 0;
        }
#endif

        if (enable_scrambling) {
            if (before_CB_split_point) {
                gold32_CB_start_output = CB_id * Er;
            } else { // after the split
                gold32_CB_start_output = TB_params->G - ((TB_params->num_CBs - CB_id) * Er);
            }

            for (int i = threadIdx.x; i <= rounded_Er_elements; i += blockDim.x) {
	       // Heads up: if 2nd gold32 argument isn't divisible by 32, the gold32 returns the sequence 2nd_arg // 32. Offset it in tmp_final_index below.
	       // For the same reason, CB_scrambling_vals needs to have (rounded_Er_elements + 1) elements
	       CB_scrambling_vals[i] = gold32(cinit_val, gold32_CB_start_output + (i << ELEMENT_BITS));
            }
        }
        __syncthreads();

        //const uint32_t N_minus_F = TB_params->N - F_val;

#if 0
There are three categories when it comes to buffer accesses w.r.t k0 (affected by rv) and  Kd.
(a) kd >= k0

<---------------------- N bits ----------------------------------->
 -------------------------|---------------|-----------------------
|                         |               |                       |
|                         |               |                       |
 -----------|-------------|---------------|-----------------------
            k0           kd <----- F ----->

Wrap around condition for this CB: N - (F + k0) < Er
First pass: Access [k0, kd) bits i.e., (kd - k0) bits and
            [kd + F, N) bits, i.e., N - (kd + F) bits

(b) k0 >= (kd + F)

<---------------------- N bits ----------------------------------->
 -------------------------|---------------|-----------------------
|                         |               |                       |
|                         |               |                       |
 -------------------------|---------------|-------------|---------
                         kd <----- F ----->             k0

Wrap around condition for this CB: N - k0 < Er
First pass: Access [k0, N) bits i.e., (N - k0) bits. Nothing to skip in this pass.

(c) k0 < (kd + F) && (k0 > kd)

<---------------------- N bits ----------------------------------->
                                     k0
 -------------------------|----------|----|-----------------------
|                         |               |                       |
|                         |               |                       |
 -------------------------|---------------|-----------------------
                         kd <----- F ----->

Wrap around condition for this CB: N - (kd + F) < Er
First pass: Access [kd+F, N) bits i.e., N - (kd + F) bits. Skip only a few bits

TODO: It is still TBD if (c) is possible given the spec. Current code supports (a) and (b).

NB: In all three cases, if a wrap around is possible, the following bits are accessed for each
subsequent wrap around pass:
  Access [0, kd) bits, i.e., Kd bits and
  [kd + F, N) bits, i.e., N - (kd + F) bits

#endif

        int buffer_case = 0;
        int first_pass_bits = 0;

        if (Kd >= k0) {
            buffer_case = 0;
            if (test_model == 0) {
                first_pass_bits = TB_params->Ncb - (F_val + k0);
            } else {
                first_pass_bits = EdivQm_bits;
            }
        } /*else if ((k0 >= Kd + F_val)) {
            buffer_case = 1;
            first_pass_bits = TB_params->N - k0;
        } else {
            buffer_case = 2;
            first_pass_bits = TB_params->N - (F_val + Kd);
        }*/
        else {
            buffer_case = 1;
            first_pass_bits = TB_params->Ncb - k0;
        }

        // Each thread block is working on a CB. Each thread on Er/(32 * blkDimX) distinct elements
        for (int i = threadIdx.x; i < rounded_Er_elements; i += blockDim.x) {

            int index = i << ELEMENT_BITS;
            int read_index = index + k0;
            uint32_t element_read = (index >= first_pass_bits) ? 0 :  ldpc_encoder_output[CB_start_input + (read_index >> ELEMENT_BITS)]; // FIXME

            int EdivQm_block_id = index / EdivQm_bits;
            int EdivQm_bit_id = index - (EdivQm_bits * EdivQm_block_id);

            #pragma unroll 8
            for (int bit_id = 0; bit_id < ELEMENT_SIZE; bit_id++) {

                if (index < tmp_Er) {

                    // Bit selection
                    int new_bit_index = bit_id;
#if 0
                    if (index < first_pass_bits) {
                        if (buffer_case == 1) {
                            read_index = index + k0;
                        } else if (buffer_case == 2) {
                            //read_index = index + k0 + Kd + F_val - k0;
                            read_index = index + Kd + F_val;
                        } else if (buffer_case == 0) {
                            read_index = index + k0;
                            if (read_index >= Kd) read_index += F_val;
                        }
                        element_read = ldpc_encoder_output[CB_start_input + (read_index >> ELEMENT_BITS)]; // not optimal
                        new_bit_index = read_index & ELEMENT_MASK;
                    } else { // This is applicable in all cases
                        read_index = (index - first_pass_bits) % (TB_params->N - F_val);
                        if (read_index >= Kd) read_index += F_val;
                        element_read = ldpc_encoder_output[CB_start_input + (read_index >> ELEMENT_BITS)]; // not optimal
                        new_bit_index = read_index & ELEMENT_MASK;
                    }
#else
                    if (index < first_pass_bits) {
                        if ((buffer_case == 0) && (index + k0 >= Kd)) { // Needed to skip filler bits. Only possible for rv=0 or rv=1
                            read_index = index + k0 + F_val;
                            element_read = ldpc_encoder_output[CB_start_input + (read_index >> ELEMENT_BITS)]; // not optimal
                            new_bit_index = read_index & ELEMENT_MASK;
                        }
                        else if ((k0 & ELEMENT_MASK) != 0) { // N/A for rv=0
                            read_index = index + k0;
                            element_read = ldpc_encoder_output[CB_start_input + (read_index >> ELEMENT_BITS)]; // not optimal
                            new_bit_index = read_index & ELEMENT_MASK;
                        }
                    } else { // This is applicable in all cases
                        read_index = (index - first_pass_bits) % (TB_params->Ncb - F_val);
                        if (read_index >= Kd) read_index += F_val;
                        element_read = ldpc_encoder_output[CB_start_input + (read_index >> ELEMENT_BITS)]; // not optimal
                        new_bit_index = read_index & ELEMENT_MASK;
                    }
#endif
                    uint32_t bit_read = (element_read >> new_bit_index);

                    // Bit interleaving
                    int final_index = (test_model == 0)? (EdivQm_bit_id * Qm_val) + EdivQm_block_id : EdivQm_bit_id; // bit_index within CB_Er block.

                    if (enable_scrambling) {
                        int tmp_final_index = final_index + (gold32_CB_start_output & ELEMENT_MASK); // offset it by # bits off
                        int CB_scrambling_index =  tmp_final_index >> ELEMENT_BITS; // index within shmem CB_scrambling_vals block
                        int CB_scrambling_bit = tmp_final_index & ELEMENT_MASK;

                        uint32_t scrambling_val = CB_scrambling_vals[CB_scrambling_index];
                        uint32_t scrambling_bit = (scrambling_val >> CB_scrambling_bit);
                        bit_read = bit_read ^ scrambling_bit; // bit not masked - will do so later
                    }
                    uint32_t layered_bit_read = bit_read;

                    bit_read = (bit_read & 0x1ULL) <<  (final_index & ELEMENT_MASK);
                    if (!enable_layer_mapping) {
                        atomicOr(&dl_rm_shmem[(final_index >> ELEMENT_BITS)], bit_read);
		    } else { // if layer mapping enabled
                       int tmp_div = EdivQm_bit_id * rev_Nl_val;
                       uint32_t tmp_layer = EdivQm_bit_id - tmp_div * Nl_val; // Don't care about actual layer value at this point.
                       int index_in_layer = (test_model == 0) ? tmp_div * Qm_val + EdivQm_block_id : tmp_div; // in bits
                       if (test_model != 0) { //FIXME ugly
                           tmp_div = tmp_div / Qm_val;
                           int modulo_tmp_div = EdivQm_bit_id - tmp_div * Qm_val * Nl_val;
                           tmp_layer = (modulo_tmp_div < Qm_val) ? 0 : 1; // at most 2 layers supported in TM (testing mode);
                           index_in_layer = tmp_div * Qm_val + ((modulo_tmp_div  < Qm_val) ? modulo_tmp_div : modulo_tmp_div - Qm_val);
                       }

                       int final_layer_index = tmp_layer * emax + index_in_layer;
                       layered_bit_read = (layered_bit_read & 0x1ULL) << (final_layer_index & ELEMENT_MASK);

                       atomicOr(&dl_rm_shmem[(final_layer_index >> ELEMENT_BITS)], layered_bit_read);
                       //if ((threadIdx.x < 32) && (blockIdx.x == 0) && (blockIdx.y == 0)) printf("bit_id %d, shmem index %d\n", bit_id, final_layer_index >> ELEMENT_BITS);
                    }
                }
                read_index += 1;
                index += 1;
                if (EdivQm_bit_id == EdivQm_bits - 1) {
                    EdivQm_bit_id = 0;
                    EdivQm_block_id += 1;
                } else {
                    EdivQm_bit_id += 1;
                }
            }
        }

        __syncthreads();
        if (!enable_layer_mapping) {
            for (int i = threadIdx.x; i < rounded_Er_elements; i+= blockDim.x) {
                rate_matching_output[CB_start_output + i] = dl_rm_shmem[i];
            }
        } else {
            int elements_per_layer = (rounded_Er_elements + Nl_val - 1)/ Nl_val;  //element here is uint32_t
            for (int i = threadIdx.x; i < elements_per_layer * Nl_val; i+= blockDim.x) {
                int layer = i / elements_per_layer;
                uint8_t current_layer = calculate_layer_id(this_TB_port_ids_array[layer], n_scid, nlAbove16);
                int layer_mapped_index = (cmax * num_TBs * current_layer + TB_id * cmax + CB_id) * rounded_Er_elements + (i % elements_per_layer);
                rate_matching_output[layer_mapped_index] = dl_rm_shmem[layer * emax_uints +  (i % elements_per_layer)];
            }
        }
}


#if 0

    // d_rate_matching_input will point to an array organized as follows.
    //  _________ ____________ ______________ ___________ ________________
    // |  TB 0   |  TB 1      |      TB 2    |  ...      | TB (num_TBs-1) |
    // | ________|____________|______________|___________|________________|
    //
    //  where each TBi contains different number of code blocks C of size N.
    //  Code block size is the same across all CBs in a TB but can vary across TBs.
    //  Start of ith TB needs to be computed sequentially (i.e., w.r.t (i-1) TB)

#endif


//Restructuring needed for modulation.
__global__ void restructure_rate_matching_output(dlRateMatchingDescr_t* p_desc) {

    dlRateMatchingDescr_t& desc = *p_desc;
    const uint32_t* __restrict__ orig_rm_output = desc.d_rate_matching_output;
    uint32_t* new_rm_output = desc.d_restructure_rate_matching_output;
    const uint32_t num_TBs = desc.num_TBs;
    const uint32_t* __restrict__ Er_array = desc.d_Er_array;
    const uint32_t cmax = desc.cmax;
    const uint32_t emax = desc.emax;
    const uint32_t max_bits_per_layer = desc.max_bits_per_layer;
    const uint32_t total_layers = desc.num_layers;
    const PdschPerTbParams* __restrict__ workspace =  desc.cfg_workspace;

    uint32_t TB_id = blockIdx.y;
    uint32_t CB_id = blockIdx.x;

    int emax_elements = emax >> 5; //emax is divisible by 32
    int num_layers = workspace[TB_id].Nl;
    int num_CBs = workspace[TB_id].num_CBs;
    uint32_t current_bits_per_layer = workspace[TB_id].G / num_layers;
    uint32_t padded_bits_per_layer = div_round_up<uint32_t>(max_bits_per_layer, 32) * 32; //Have each layer in the restructured output start at uint32_t aligned boundary

    if (CB_id >= num_CBs) return;

    // Update starting offset in bits for current CB; this is a global offset, not layer specific.
    uint32_t layer_restructure_rm_shmem = TB_id * padded_bits_per_layer; //Use max padded bits per layer
    bool before_CB_split_point = (CB_id < Er_array[TB_id * 2]);
    const int CB_Er = Er_array[TB_id * 2 + 1] + ((before_CB_split_point) ? 0 : num_layers * workspace[TB_id].Qm);
    int per_layer_Er = (CB_Er / num_layers); // in bits
    if (before_CB_split_point) {
        layer_restructure_rm_shmem += (CB_id * per_layer_Er);
    } else { // CB after the split point
        layer_restructure_rm_shmem += (current_bits_per_layer - ((num_CBs - CB_id) * per_layer_Er));
    }
    const uint8_t* port_ids_array = desc.d_params[TB_id].port_ids;
    const uint8_t n_scid = desc.d_params[TB_id].n_scid;
    const uint8_t nlAbove16 = desc.d_params[TB_id].nlAbove16;

    for (int potential_layer_id = 0; potential_layer_id < num_layers; potential_layer_id++) {
        int layer_id = calculate_layer_id(port_ids_array[potential_layer_id], n_scid, nlAbove16); // layer_id is the actual layer


        // Update starting offset in bits for current CB; this is a global offset, not layer specific.
        uint32_t restructure_rm_shmem = (layer_id * num_TBs) * padded_bits_per_layer + layer_restructure_rm_shmem;
        uint32_t * output_addr = (&new_rm_output[0]);
        const uint32_t * input_addr = (uint32_t*)&orig_rm_output[(layer_id * cmax * num_TBs + cmax * TB_id + CB_id) * emax_elements];

        //Each CB will need to populate the bit range: [output_start_offset_in_bits, output_end_offset_in_bits)
        uint32_t output_start_offset_in_bits = restructure_rm_shmem;
        uint32_t output_start_offset_in_elements = (output_start_offset_in_bits >> 5);
        uint32_t tmp_rem = (output_start_offset_in_bits & 0x1F);
        uint32_t start_rem = (tmp_rem == 0) ? 0 : 32 - tmp_rem;

        uint32_t output_end_offset_in_bits = restructure_rm_shmem + per_layer_Er;
        uint32_t output_end_offset_in_elements = output_end_offset_in_bits >> 5;
        uint32_t end_rem = (output_end_offset_in_bits & 0x1F);

        //Each thread will process an uint32_t element
        if (end_rem != 0) output_end_offset_in_elements += 1;
        int elements = output_end_offset_in_elements - output_start_offset_in_elements;

        for (int element = threadIdx.x; element < elements; element += blockDim.x) { //uint32_t elements for a CB
            uint32_t read_val = 0;

            if ((element == (elements - 1)) && (end_rem != 0)) {
                if (start_rem != 0) {
                    // Partially update last element of this CB and first of next one
                    read_val = (input_addr[element - 1] >> start_rem);
                    read_val |= ((input_addr[element] & ((1 << start_rem)-1)) << tmp_rem);
                    read_val = (read_val & ((1 << end_rem) -1));
                } else {
                    read_val = input_addr[element];
                }

                // Include partial update of next element, to avoid atomicOr
                uint32_t output_index = (layer_id * num_TBs + TB_id) * cmax + (CB_id + 1);
                if (CB_id == (cmax - 1)) {
                    output_index = (layer_id == (total_layers - 1)) ? 0 : ((layer_id + 1) * num_TBs + TB_id) *  cmax;
                }
                if (output_index != 0) {
                    const uint32_t * next_CB_input_addr = (uint32_t*)&orig_rm_output[output_index * emax_elements];
                    read_val |= ((next_CB_input_addr[0] & ((1 << (32 - end_rem))-1)) << end_rem);
                }
                output_addr[output_start_offset_in_elements + element] = read_val;

            } else if ((element != 0) || (start_rem == 0)) {
                if (start_rem != 0) {
                    read_val = (input_addr[element - 1] >> start_rem);
                    read_val |= ((input_addr[element] & ((1 << start_rem)-1)) << tmp_rem);
                } else {
                    read_val = input_addr[element];
                }
                output_addr[output_start_offset_in_elements + element] = read_val;
            }
        }
    }
}


cuphyStatus_t CUPHYWINAPI cuphySetupDlRateMatching(cuphyDlRateMatchingLaunchConfig_t dlRateMatchingLaunchConfig,
                                                   cuphyPdschStatusOut_t* status,
                                                   const uint32_t* d_rate_matching_input,
                                                   uint32_t* d_rate_matching_output,
                                                   uint32_t* d_restructure_rate_matching_output,
                                                   void* d_modulation_output,
                                                   void* d_xtf_re_map,
                                                   uint16_t max_PRB_BWP,
                                                   int num_TBs,
                                                   int num_layers,
                                                   uint8_t enable_scrambling,
                                                   uint8_t enable_layer_mapping,
                                                   uint8_t enable_modulation,
                                                   uint8_t precoding,
                                                   uint8_t restructure_kernel,
                                                   uint8_t batching,
                                                   uint32_t* h_workspace,
                                                   uint32_t* d_workspace, //H2D copy from h_workspace to d_workspace happens within setup if enable_desc_async_copy
                                                   PdschPerTbParams* h_params,
                                                   PdschPerTbParams* d_params, //H2D copy from h_params to d_params should happen outside setup
                                                   PdschDmrsParams* d_dmrs_params,
                                                   PdschUeGrpParams* d_ue_grp_params,
                                                   void* cpu_desc,
                                                   void* gpu_desc,
                                                   uint8_t enable_desc_async_copy,
                                                   cudaStream_t strm)
{

    if (d_rate_matching_input == nullptr) {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    if ((!enable_layer_mapping) && (d_rate_matching_output == nullptr)) {
        // The d_rate_matching_output is only used in AAS mode which implies no layer mapping
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    if (restructure_kernel && ((d_restructure_rate_matching_output == nullptr) || (num_layers == 0))) {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    if (enable_modulation && ((d_dmrs_params == nullptr) || (d_modulation_output == nullptr))) {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    if (!h_workspace || !d_workspace || !h_params || !d_params || !cpu_desc || !gpu_desc) {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    if ((max_PRB_BWP > MAX_N_PRBS_SUPPORTED) || (max_PRB_BWP == 0)) {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }

    dlRateMatchingLaunchConfig->m_kernelArgs[0] = &(dlRateMatchingLaunchConfig->m_desc);
    dlRateMatchingLaunchConfig->m_kernelNodeParams[0].extra = nullptr;
    dlRateMatchingLaunchConfig->m_kernelNodeParams[0].kernelParams = &(dlRateMatchingLaunchConfig->m_kernelArgs[0]);

    dlRateMatchingLaunchConfig->m_kernelNodeParams[1].extra = nullptr;
    dlRateMatchingLaunchConfig->m_kernelNodeParams[1].kernelParams = &(dlRateMatchingLaunchConfig->m_kernelArgs[0]);

    //Set up CPU descriptor.
    dlRateMatchingDescr_t& desc = *(static_cast<dlRateMatchingDescr_t*>(cpu_desc));
    desc.d_rate_matching_input = d_rate_matching_input;
    desc.d_rate_matching_output = d_rate_matching_output;
    desc.d_restructure_rate_matching_output = d_restructure_rate_matching_output;
    desc.max_PRB_BWP = max_PRB_BWP;
    desc.num_TBs = num_TBs;
    desc.d_params = d_dmrs_params;
    desc.d_ue_grp_params = d_ue_grp_params;
    desc.temp_xtf_re_map = (uint16_t*)d_xtf_re_map;

    // Let workspace be as follows k0 first, TB_start_offset and then Er.
    // Populate h_workspace first
    uint32_t* h_k0_array = (uint32_t*) h_workspace;
    uint32_t* h_TB_start_offset_array = (uint32_t*)h_workspace + num_TBs;
    uint32_t* h_Er_array = h_TB_start_offset_array + num_TBs;

    // Compute max number of code blocks across all num_TBs transport blocks.
    // Needed for allocation of Er array.
    uint32_t c_max = 0;
    uint32_t n_max = 0;
    uint32_t max_bits_per_layer = 0;
    for (int i = 0; i < num_TBs; i++) {
        //printf("TB %d TM %d tbSize %d\n", i, h_params[i].testModel, h_params[i].tbSize);
        c_max = max(c_max, h_params[i].num_CBs);
        n_max = max(n_max, round_up_to_next((int)h_params[i].N, 32));
        // Even when testModel == 1, h_params[i].G == h_params[i].tbsize*8;
        max_bits_per_layer = std::max(max_bits_per_layer, h_params[i].G / h_params[i].Nl);
        //printf("TB %d, CBs %d, N %d\n", i, h_params[i].num_CBs, h_params[i].N);
    }
    //printf("cmax %d n_max %d\n", c_max, n_max);

    // Populate internal arrays
    uint32_t er_max = 0; // updated in compute_rate_matching_length function call
    bool     updated_er_max = false;
    int      TB_idx_with_er_max = -1;
    for (int i = 0; i < num_TBs; i++) {
        if (h_params[i].testModel == 0) {
            compute_rate_matching_length(&(h_Er_array[i * 2]), h_params[i].num_CBs,
                                         h_params[i].Qm, h_params[i].Nl, h_params[i].G, er_max, updated_er_max);
            h_k0_array[i] = compute_k0(h_params[i].rv, h_params[i].bg, h_params[i].Ncb, h_params[i].Zc);
        } else {
            compute_rate_matching_length(&(h_Er_array[i * 2]), h_params[i].num_CBs,
                                         h_params[i].Qm, h_params[i].Nl, h_params[i].N, er_max, updated_er_max, true);
            h_k0_array[i] = 0;
        }
        if (updated_er_max) TB_idx_with_er_max = i;
        if (batching) {
            h_TB_start_offset_array[i] = (i == 0) ? 0 : h_TB_start_offset_array[i-1] + ((h_params[i-1].testModel == 0) ? (round_up_to_next((int) h_params[i-1].N, 32) * h_params[i-1].num_CBs) : round_up_to_next((int)h_params[i-1].tbSize*8,32)); //to support different TB configs
        } else {
            h_TB_start_offset_array[i] = i * c_max * n_max; //to support different TB configs
        }
       //printf("i %d, prev offset %d, N %d, CBs %d\n", i,  (i == 0) ? 0 : h_TB_start_offset_array[i-1], (i == 0) ? 0 : h_params[i-1].N, (i == 0) ? 0 : h_params[i-1].num_CBs);
    }
    er_max = div_round_up<uint32_t>(er_max, 32)*32;
    uint32_t rounded_emax = div_round_up<uint32_t>(er_max, 32);
    if (er_max > PDSCH_MAX_ER_PER_CB_BITS) {
        status->status = cuphyPdschStatusType_t::CUPHY_PDSCH_STATUS_UNSUPPORTED_MAX_ER_PER_CB;
        status->ueIdx = TB_idx_with_er_max;
        // status->cellPrmStatIdx will be updated in PDSCH setup to get the mapping of this UE to cell
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }

    desc.d_k0_array = (uint32_t*)d_workspace;
    desc.d_TB_start_offset_array = (uint32_t*)d_workspace + num_TBs;
    desc.d_Er_array = desc.d_TB_start_offset_array + num_TBs;

    size_t workspace_size = num_TBs * sizeof(uint32_t) * (2 + 2);

    desc.enable_scrambling = enable_scrambling;
    desc.enable_layer_mapping = enable_layer_mapping;
    desc.cmax = c_max;
    desc.emax = er_max;
    desc.cfg_workspace = d_params;
    desc.max_bits_per_layer = max_bits_per_layer;
    desc.num_layers = num_layers;
    //printf("emax %d, c_max %d, max_bits_per_layer %d\n", er_max, c_max, max_bits_per_layer);

    // Optional descriptor copy to GPU memory
    // When running as part of a pipeline, it's better to do a single copy of all descriptors in the pipeline.
    if (enable_desc_async_copy) {
        CUDA_CHECK(cudaMemcpyAsync(d_workspace, h_workspace, workspace_size, cudaMemcpyHostToDevice, strm));
        CUDA_CHECK(cudaMemcpyAsync(gpu_desc, cpu_desc, sizeof(dlRateMatchingDescr_t), cudaMemcpyHostToDevice, strm));
    }
    dlRateMatchingLaunchConfig->m_desc = static_cast<dlRateMatchingDescr_t*>(gpu_desc);
    const uint32_t num_threads = 288;

    size_t scrambling_shmem_elements = enable_scrambling ? rounded_emax + 1: 0;
    size_t layer_mapping_shmem_elements = (enable_layer_mapping) ? rounded_emax * MAX_DL_LAYERS_PER_TB : rounded_emax;
    size_t shmem_size = (scrambling_shmem_elements + layer_mapping_shmem_elements) * sizeof(uint32_t);

    if (enable_modulation) {
        // Always update the kernel function, as the presence of precoding for the cells in a cell group
        // can change throughout the cuPHY PDSCH object's lifetime.
        // This is also the case for standalone components.
        cudaFunction_t device_function;
        if (precoding == 0) {
            {MemtraceDisableScope md; CUDA_CHECK(cudaGetFuncBySymbol(&device_function, reinterpret_cast<void*>(fused_dl_rm_and_modulation<false>)));}
            // In some cases, shmem_size may exceed 48KB, which requires an explicit opt-in.
            // For example, a single UE with an allocation of 273 PRBs, 13 data symbols, 4 layers, 4 layers,
            // and mcsTable=2 and mcsIndex=0 (i.e., lowest code-rate) would require 70984 of shared memory.
            // Note if shmem_size is greater than the max. supported option on that GPU, the call below would fail. FIXME TBD if this is possible in case of retransmission
            // See cuFuncSetAttribute call below
            //CUDA_CHECK(cudaFuncSetAttribute(fused_dl_rm_and_modulation<false>, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_size));
#if 0
            /*cudaFuncAttributes attr;
            CUDA_CHECK(cudaFuncGetAttributes(&attr, fused_dl_rm_and_modulation<false>));
            if (attr.maxDynamicSharedSizeBytes < shmem_size) {
                printf("attr.maxDynamicSharedSizeBytes %d but shmem_size %d \n", attr.maxDynamicSharedSizeBytes, shmem_size);
                CUDA_CHECK(cudaFuncSetAttribute(fused_dl_rm_and_modulation<false>, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_size));
            }*/
#endif
        } else {
            {MemtraceDisableScope md; CUDA_CHECK(cudaGetFuncBySymbol(&device_function, reinterpret_cast<void*>(fused_dl_rm_and_modulation<true>)));}
            // In some cases, shmem_size may exceed 48KB, which requires an explicit opt-in. See earlier comment too. See cuFuncSetAttribute call below
            //CUDA_CHECK(cudaFuncSetAttribute(fused_dl_rm_and_modulation<true>, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_size));
        }
        CU_CHECK(cuFuncSetAttribute(device_function, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, shmem_size));
        dlRateMatchingLaunchConfig->m_kernelNodeParams[0].func = device_function;

        // Fused rate-matching + modulation kernel.
        dlRateMatchingLaunchConfig->m_kernelNodeParams[0].blockDimX = num_threads;
        dlRateMatchingLaunchConfig->m_kernelNodeParams[0].blockDimY = 1;
        dlRateMatchingLaunchConfig->m_kernelNodeParams[0].blockDimZ = 1;
        dlRateMatchingLaunchConfig->m_kernelNodeParams[0].gridDimX  = desc.cmax;
        dlRateMatchingLaunchConfig->m_kernelNodeParams[0].gridDimY  = num_TBs;
        dlRateMatchingLaunchConfig->m_kernelNodeParams[0].gridDimZ  = 1;
        dlRateMatchingLaunchConfig->m_kernelNodeParams[0].sharedMemBytes = shmem_size;
    } else {
        // Separate rate-matching + restructuring kernels
        cudaFunction_t rm_device_function, restructuring_device_function;

        // Always update the kernel functions, as the presence of precoding for the cells in a cell group
        // can change throughout the cuPHY PDSCH object's lifetime.
        // This is also the case for standalone components.
        {MemtraceDisableScope md; CUDA_CHECK(cudaGetFuncBySymbol(&rm_device_function, reinterpret_cast<void*>(dl_rate_matching)));}
        {MemtraceDisableScope md; CUDA_CHECK(cudaGetFuncBySymbol(&restructuring_device_function, reinterpret_cast<void*>(restructure_rate_matching_output)));}
        CUDA_CHECK(cudaFuncSetAttribute(dl_rate_matching, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_size));

        dlRateMatchingLaunchConfig->m_kernelNodeParams[0].func = rm_device_function;
        dlRateMatchingLaunchConfig->m_kernelNodeParams[1].func = restructuring_device_function;

        // For rate matching kernel
        dlRateMatchingLaunchConfig->m_kernelNodeParams[0].blockDimX = num_threads;
        dlRateMatchingLaunchConfig->m_kernelNodeParams[0].blockDimY = 1;
        dlRateMatchingLaunchConfig->m_kernelNodeParams[0].blockDimZ = 1;
        dlRateMatchingLaunchConfig->m_kernelNodeParams[0].gridDimX  = desc.cmax;
        dlRateMatchingLaunchConfig->m_kernelNodeParams[0].gridDimY  = num_TBs;
        dlRateMatchingLaunchConfig->m_kernelNodeParams[0].gridDimZ  = 1;
        dlRateMatchingLaunchConfig->m_kernelNodeParams[0].sharedMemBytes = shmem_size;

        // For restructuring kernel
        dlRateMatchingLaunchConfig->m_kernelNodeParams[1].blockDimX = 256;
        dlRateMatchingLaunchConfig->m_kernelNodeParams[1].blockDimY = 1;
        dlRateMatchingLaunchConfig->m_kernelNodeParams[1].blockDimZ = 1;
        dlRateMatchingLaunchConfig->m_kernelNodeParams[1].gridDimX  = desc.cmax;
        dlRateMatchingLaunchConfig->m_kernelNodeParams[1].gridDimY  = num_TBs;
        dlRateMatchingLaunchConfig->m_kernelNodeParams[1].gridDimZ  = 1;
        dlRateMatchingLaunchConfig->m_kernelNodeParams[1].sharedMemBytes = 0;
    }

    return CUPHY_STATUS_SUCCESS;
}

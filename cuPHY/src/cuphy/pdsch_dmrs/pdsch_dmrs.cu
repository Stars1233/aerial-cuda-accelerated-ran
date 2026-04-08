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

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include "cuphy.h"
#include "cuphy_api.h"
#include "cuphy_internal.h"
#include "descrambling.hpp" // for POLY_* masks etc.
#include "descrambling.cuh"
#include "crc.hpp"
#include "csirs.cuh"
#include "tensor_desc.hpp"
#include "math_utils.cuh"
#include "pdsch_dmrs.hpp"

#include "cuphy.hpp"

using namespace cuphy_i;
using namespace descrambling; // for POLY_ etc.
namespace cg = cooperative_groups;

#define PRINT_TIMING 0

#if PRINT_TIMING

#include <chrono>

using t_ns = std::chrono::nanoseconds;
using t_us = std::chrono::microseconds;
using t_ms = std::chrono::milliseconds;
using t_tp = std::chrono::time_point<std::chrono::system_clock>;

class Time {
public:
    Time();
    ~Time();

    static t_ns nowNs()
    {
        return std::chrono::system_clock::now().time_since_epoch();
    }
    static t_us NsToUs(t_ns time)
    {
    return std::chrono::duration_cast<t_us>(time);
   }
};
#endif


cuphyStatus_t CUPHYWINAPI cuphyUpdatePdschDmrsParams(PdschDmrsParams * h_dmrs_params, cuphyPdschDynPrms_t* dyn_params,
                                                     const cuphyPdschStatPrms_t* static_params,
                                                     PdschUeGrpParams* pdsch_ue_group_params) {

    if ((h_dmrs_params == nullptr) || (dyn_params == nullptr) || \
        (static_params == nullptr)) {
        NVLOGE_FMT(NVLOG_PDSCH, AERIAL_CUPHY_EVENT, "cuphyUpdatePdschDmrsParam nullptr print: h_dmrs_params {:p}, dyn_params {:p}, static_params {:p}",
                   (void*)h_dmrs_params, (void*)dyn_params, (void*)static_params);
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }

    if (pdsch_ue_group_params) {
        //Reset all skipped REs for all symbols to zero for all UE groups FIXME won't work now that this is moved to Run
        memset(pdsch_ue_group_params, 0, sizeof(PdschUeGrpParams) * dyn_params->pCellGrpDynPrm->nUeGrps);
    }

    // Get the 3rd dimension of the output tensor per cell. Will later check if number of antenna ports
    // for each UE when precoding is enabled is valid.
    // Note: tensor_ports is only used for error checking. Could potentially make the check optional.
    uint16_t tensor_ports[PDSCH_MAX_CELLS_PER_CELL_GROUP]; // overprovisioned
    int dims[3];
    int num_dims = 0;
    for (int dyn_cell_idx = 0; dyn_cell_idx < dyn_params->pCellGrpDynPrm->nCells; dyn_cell_idx++)
    {
            cuphyStatus_t tensor_status = cuphyGetTensorDescriptor(dyn_params->pDataOut->pTDataTx[dyn_cell_idx].desc, 3, nullptr, &num_dims, dims, nullptr);
            if (tensor_status != CUPHY_STATUS_SUCCESS)
            {
                NVLOGE_FMT(NVLOG_PDSCH, AERIAL_CUPHY_EVENT, "cuphyUpdatePdschDmrsParam invalid argument when calling cuphyGetTensorDescriptor for dyn_cell_idx {}", dyn_cell_idx);
                return CUPHY_STATUS_INVALID_ARGUMENT;
            }
            tensor_ports[dyn_cell_idx]= dims[2];
    }

#if PRINT_TIMING
    t_ns start_time, end_time;
    start_time = Time::nowNs();
#endif

    // The CSI-RS re-map, if CSI-RS parameters, are present, should only be updated once per UE-group, and not by each UE in that UE group separately.
    for (int UE_group_idx = 0; UE_group_idx < dyn_params->pCellGrpDynPrm->nUeGrps; UE_group_idx++) {
        cuphyPdschUeGrpPrm_t* ue_group = &dyn_params->pCellGrpDynPrm->pUeGrpPrms[UE_group_idx];

        cuphyPdschCellDynPrm_t* cell   = ue_group->pCellPrm;
        uint16_t dyn_cell = cell->cellPrmDynIdx;
        cuphyCellStatPrm_t *static_cell_params = &static_params->pCellStatPrms[cell->cellPrmStatIdx];

        //Use ue_group->pDmrsDynPrm->nDmrsCdmGrpsNoData for a table lookup
        //(Table 4.1-1 from ETSI 138.214)
        int dmrs_cdm_grps = ue_group->pDmrsDynPrm->nDmrsCdmGrpsNoData;
        if (dmrs_cdm_grps == 3) {
            NVLOGE_FMT(NVLOG_PDSCH, AERIAL_CUPHY_EVENT, "3 DM-RS CDM groups without data not supported for type-I DMRS!");
            return CUPHY_STATUS_INVALID_ARGUMENT;
        }

        uint16_t dmrs_bitmask = (cell->dmrsSymLocBmsk == 0) ? ue_group->dmrsSymLocBmsk : cell->dmrsSymLocBmsk;
        uint8_t num_pdsch_symbols = (cell->nPdschSym == 0) ? ue_group->nPdschSym : cell->nPdschSym;
        uint8_t num_dmrs_symbols  =  __builtin_popcount(dmrs_bitmask); // Assumes that all set DMRS bits are after the start symbol.
        uint8_t num_data_symbols  =  num_pdsch_symbols - num_dmrs_symbols;
        uint16_t start_symbol = (cell->nPdschSym == 0) ? ue_group->pdschStartSym : cell->pdschStartSym;

        // Compute CSI-RS related fields

        // Data symbol positions
        uint64_t data_sym_loc = 0, dmrs_sym_loc = 0;
        uint8_t current_symbol_index = start_symbol;
        uint32_t current_symbol_offset = current_symbol_index * static_cell_params->nPrbDlBwp * CUPHY_N_TONES_PER_PRB + ue_group->startPrb * CUPHY_N_TONES_PER_PRB;
        int data_symbol = 0, dmrs_symbol = 0;

        int first_TB_id = ue_group->pUePrmIdxs[0];

        // Set tb_idx field
        if (pdsch_ue_group_params) {
            pdsch_ue_group_params[UE_group_idx].tb_idx = first_TB_id;
        }

        while (data_symbol < num_data_symbols) {
            if (((dmrs_bitmask >> current_symbol_index) & 1) == 0) {
                data_sym_loc |= ((uint64_t)(current_symbol_index & 0xF) << (data_symbol * 4));
                data_symbol += 1;
            } else {
                dmrs_sym_loc |= ((uint16_t)(current_symbol_index & 0xF) << (dmrs_symbol * 4));
                dmrs_symbol += 1;
            }
            current_symbol_index += 1;
            current_symbol_offset +=  static_cell_params->nPrbDlBwp * CUPHY_N_TONES_PER_PRB;
        }

        while (dmrs_symbol < num_dmrs_symbols) {
            if (((dmrs_bitmask >> current_symbol_index) & 1) == 1) {
                dmrs_sym_loc |= ((uint16_t)(current_symbol_index & 0xF) << (dmrs_symbol * 4));
                dmrs_symbol += 1;
            }
            current_symbol_index += 1;
        }


        for (int UE_idx = 0; UE_idx < ue_group->nUes; UE_idx++) {
            int TB_id = ue_group->pUePrmIdxs[UE_idx];
            h_dmrs_params[TB_id].ueGrp_idx = UE_group_idx;

            cuphyPdschCwPrm_t* cw = &dyn_params->pCellGrpDynPrm->pCwPrms[TB_id];
            cuphyPdschUePrm_t* ue = cw->pUePrm;

            h_dmrs_params[TB_id].data_sym_loc = data_sym_loc;
            h_dmrs_params[TB_id].dmrs_sym_loc = dmrs_sym_loc;
            h_dmrs_params[TB_id].num_dmrs_symbols = num_dmrs_symbols;
            h_dmrs_params[TB_id].num_data_symbols = num_pdsch_symbols - num_dmrs_symbols;

            //NVLOGC_FMT(NVLOG_PDSCH, "num DMRS symbols {:d}, DMRS loc {:#x}", h_dmrs_params[TB_id].num_dmrs_symbols, h_dmrs_params[TB_id].dmrs_sym_loc);
            //NVLOGC_FMT(NVLOG_PDSCH, "num data symbols {:d}. loc in hex is {:#x}", h_dmrs_params[TB_id].num_data_symbols, h_dmrs_params[TB_id].data_sym_loc);

            // Set pointer to cell's output tensor. FIXME potentially revisit
            h_dmrs_params[TB_id].cell_output_tensor_addr = dyn_params->pDataOut->pTDataTx[dyn_cell].pAddr;
            //NVLOGC_FMT(NVLOG_PDSCH, "TB {:d} dyn cell {:d} addr {:p}\n", TB_id, dyn_cell,  h_dmrs_params[TB_id].cell_output_tensor_addr);

            int slot_number = cell->slotNum;
            int cell_id = static_cell_params->phyCellId;

            h_dmrs_params[TB_id].slot_number = slot_number;
            h_dmrs_params[TB_id].cell_id = cell_id;

            h_dmrs_params[TB_id].beta_dmrs = sqrtf(dmrs_cdm_grps) * ue->beta_dmrs;
            h_dmrs_params[TB_id].beta_qam  = ue->beta_qam;
            h_dmrs_params[TB_id].num_BWP_PRBs = static_cell_params->nPrbDlBwp; // Superfluous

            h_dmrs_params[TB_id].symbol_number = cell->pdschStartSym;
            h_dmrs_params[TB_id].num_layers = ue->nUeLayers;
            h_dmrs_params[TB_id].resourceAlloc = ue_group->resourceAlloc;
            memcpy(h_dmrs_params[TB_id].rbBitmap,ue_group->rbBitmap,MAX_RBMASK_BYTE_SIZE);

            h_dmrs_params[TB_id].start_Rb = ue_group->startPrb;
            h_dmrs_params[TB_id].num_Rbs = ue_group->nPrb;
            if (h_dmrs_params[TB_id].num_Rbs == 0) {
                NVLOGE_FMT(NVLOG_PDSCH, AERIAL_CUPHY_EVENT, "Zero PRBs allocated for DMRS!");
                return CUPHY_STATUS_INVALID_ARGUMENT;
            }
            h_dmrs_params[TB_id].num_BWP_PRBs = static_cell_params->nPrbDlBwp; // Superfluous
            if (h_dmrs_params[TB_id].num_Rbs > h_dmrs_params[TB_id].num_BWP_PRBs) {
                NVLOGE_FMT(NVLOG_PDSCH, AERIAL_CUPHY_EVENT, "Allocated PRBs {} are more than the PRBs {} in this BWP (bandwidth part)!",
                           h_dmrs_params[TB_id].num_BWP_PRBs, static_cell_params->nPrbDlBwp);
                return CUPHY_STATUS_INVALID_ARGUMENT;
            }
            h_dmrs_params[TB_id].cell_index_in_cell_group = dyn_cell; //Used to find this cell's RE map

#if 0
            // Up to 8 layers are encoded in an uint32_t, 4 bits at a time.
            uint32_t port_index = ue->nPortIndex;
            for (int i = 0; i < h_dmrs_params[TB_id].num_layers; i++) {
                h_dmrs_params[TB_id].port_ids[i] = (port_index >> (28 - 4 * i)) & 0x0FU; // not adding the 1000 offset to all
            }
#else
            uint32_t dmrs_ports_bitmask = ue->dmrsPortBmsk;
            for (int i = 0; i < h_dmrs_params[TB_id].num_layers; i++) {
                h_dmrs_params[TB_id].port_ids[i] = __builtin_ctz(dmrs_ports_bitmask);
                dmrs_ports_bitmask ^= (1 << h_dmrs_params[TB_id].port_ids[i]);
            }
#endif

            h_dmrs_params[TB_id].n_scid = ue->scid;
            h_dmrs_params[TB_id].dmrs_scid = ue->dmrsScrmId;
            h_dmrs_params[TB_id].nlAbove16 = ue->nlAbove16;
            h_dmrs_params[TB_id].ref_point = ue->refPoint;
            h_dmrs_params[TB_id].BWP_start_PRB = ue->BWPStart;

            h_dmrs_params[TB_id].enablePrcdBf = ue->enablePrcdBf;
            h_dmrs_params[TB_id].dmrsCdmGrpsNoData1 = (dmrs_cdm_grps == 1);
            if(ue->enablePrcdBf)
            {
                // Set the number of antenna ports for this UE which is also the number of columns of the pre-coding matrix.
                h_dmrs_params[TB_id].Np = dyn_params->pCellGrpDynPrm->pPmwPrms[ue->pmwPrmIdx].nPorts;

                // Check that the number of antenna ports is not greater than the 3rd dimension of the output tensor
                if (h_dmrs_params[TB_id].Np > tensor_ports[dyn_cell])
                {
                    NVLOGE_FMT(NVLOG_PDSCH, AERIAL_CUPHY_EVENT, "UE with precoding enabled has more antenna ports than the 3rd dimension of the output PDSCH TX tensor!");
                    return CUPHY_STATUS_INVALID_ARGUMENT;
                }

                // copy pre-coding matrix
                memcpy(h_dmrs_params[TB_id].pmW, dyn_params->pCellGrpDynPrm->pPmwPrms[ue->pmwPrmIdx].matrix, sizeof(__half2) * ue->nUeLayers * h_dmrs_params[TB_id].Np);
            }
            else
            {
                h_dmrs_params[TB_id].Np = 0; // Number of ports not used if precoding not enabled.
            }
        }
    }
#if PRINT_TIMING
    end_time = Time::nowNs();
    NVLOGC_FMT(NVLOG_PDSCH, "updateDMRS timing: {} us", Time::NsToUs(end_time - start_time).count());
#endif
    return CUPHY_STATUS_SUCCESS;
}

template<bool enablePrecoding, typename Tcomplex, typename Tscalar=typename scalar_from_complex<Tcomplex>::type>
__global__ void fused_dmrs(pdschDmrsDescr_t* p_desc) {

    pdschDmrsDescr_t& desc = *p_desc;
    const PdschDmrsParams* params = desc.dmrs_params;
    //const int num_TBs = desc.num_TBs; // The grid's y dim is num_TBs * MAX_DL_LAYERS_PER_TB. Reading from desc. not currently needed.

    // Let each blockIdx.y be: TB_id * num_layers + layer_id;
    const int TB_id = blockIdx.y;

    Tcomplex* dmrs_output = (Tcomplex*)params[TB_id].cell_output_tensor_addr;

    const int slot_number = params[TB_id].slot_number;
    const int num_dmrs_symbols = params[TB_id].num_dmrs_symbols;
    __builtin_assume((num_dmrs_symbols > 0) && (num_dmrs_symbols <= 4));

    Tscalar positive_scramble_seq = 0.707106781186547f * params[TB_id].beta_dmrs;
    __shared__ uint32_t shmem_gold_seqs[64];  // overprovisioned. Should be > (blockDim.x * 2 / 32) * 4; // Last * 4 is due to max four DMRS symbols (might be superfluous)

    // Compute (blockDim.x / 32) gold_seq. elements per block; each gold32 call computes 32 bits
    __builtin_assume(blockDim.x == 256);
    const int gold_elements_one_dmrs = 256 >> 4; //blockDim.x >> 4; // blockDim.x * 2 / 32; Multiplied by 2 because each thread will read 2 bits

    // Reminder: according to the spec (Table 7.4.1.1.2-4 in 38.211) only addln DMRS positions 0 or 1 exist for double symbol DMRS, so
    // max 4 DMRS symbols are possible.

    // Currently using a host-computed dmrs_sym_loc, an uint16_t to keep track of the max 4 possible symbols
    const uint16_t bitmask = params[TB_id].dmrs_sym_loc;
    const uint8_t symbol_loc[4] = {(uint8_t)(bitmask & 0xfu),
                                   (uint8_t)((bitmask >> 4)& 0xfu),
                                   (uint8_t)((bitmask >> 8) & 0xfu),
                                   (uint8_t)((bitmask >> 12) & 0xfu)};

    for (int i = threadIdx.x; i < num_dmrs_symbols * gold_elements_one_dmrs; i += blockDim.x) {

        int dmrs_symbol = i >> 4; // As gold_elements_one_dmrs is 16 for block size of 256.
        int offset = i - dmrs_symbol * gold_elements_one_dmrs;
        uint32_t gold_index = blockIdx.x * gold_elements_one_dmrs + offset; // Indexing not affected by # dmrs symbols, just the seed is.

        uint32_t double_nid = (params[TB_id].dmrs_scid << 1);
        // Reminder the seed init. value uses DRMS symbol location (0-based indexing) + 1.
        uint32_t c_init = ((1 << 17) * (slot_number * OFDM_SYMBOLS_PER_SLOT + symbol_loc[dmrs_symbol] + 1) * (double_nid + 1) + double_nid + params[TB_id].n_scid) & 0x7FFFFFFFU;
        shmem_gold_seqs[i] = gold32(c_init, gold_index << 5); // gold_index mult. by 32, the # of bits
    }
    __syncthreads();

    // Build scrambling sequence
    int tid_x = threadIdx.x + blockIdx.x * blockDim.x;
    uint16_t start_Rb = params[TB_id].start_Rb;

    //Every thread will read the value from dmrs_addr
    const int scrambling_seq_start_tone = ((params[TB_id].ref_point == 1) ? (start_Rb - params[TB_id].BWP_start_PRB) : start_Rb) * (CUPHY_N_TONES_PER_PRB >> 1);
    const int new_tidx = tid_x - scrambling_seq_start_tone;
    const int BWP_PRBs = params[TB_id].num_BWP_PRBs;
    // Only use threads that cover used RBs
    bool tid_enable = (new_tidx < (params[TB_id].num_Rbs * (CUPHY_N_TONES_PER_PRB >> 1))) && (new_tidx >= 0); // Resource Allocation Type 1
    if(params[TB_id].resourceAlloc == 0){
        //const int tid_rb = ((params[TB_id].ref_point == 1) ? params[TB_id].BWP_start_PRB : 0) + tid_x/(CUPHY_N_TONES_PER_PRB >> 1);
        const int tid_rb = ((params[TB_id].ref_point == 1) ? 0 : -params[TB_id].BWP_start_PRB) + tid_x/(CUPHY_N_TONES_PER_PRB >> 1);
        if((tid_rb < (MAX_RBMASK_BYTE_SIZE*8))&&(0<=tid_rb)) {
            tid_enable = (0!=(params[TB_id].rbBitmap[tid_rb>>5] & (1L << (tid_rb & 0x1F))));
        }
    }
    if(tid_enable){

        for (int symbol_id = 0; symbol_id < num_dmrs_symbols; symbol_id++) {
            int base_shmem_index = symbol_id * gold_elements_one_dmrs;
            int shmem_index = base_shmem_index + (threadIdx.x >> 4);

            /*if (shmem_index >= num_dmrs_symbols * gold_elements_one_dmrs) {
                printf("threadIdx %d, blockIdx.x %d, blockIdx.y %d, shmem_index %d, gold_elements %d\n",
                        threadIdx.x, blockIdx.x, blockIdx.y, shmem_index, num_dmrs_symbols * gold_elements_one_dmrs);
            }*/

            int shmem_bit_offset = ((threadIdx.x & 0xFU) << 1); // Shift by 1 because each thread reads two bits
            uint32_t gold_value = ((shmem_gold_seqs[shmem_index] >> shmem_bit_offset) & 0x3U);
            //uint32_t gold_value = (shmem_gold_seqs[shmem_index] >> shmem_bit_offset);

#if 1
            Tcomplex scrambled_val;
            if (gold_value == 0) {
                scrambled_val = make_complex<Tcomplex>::create(positive_scramble_seq, positive_scramble_seq);
            } else if (gold_value == 1) {
                scrambled_val = make_complex<Tcomplex>::create(-positive_scramble_seq, positive_scramble_seq);
            } else if (gold_value == 2) {
                scrambled_val = make_complex<Tcomplex>::create(positive_scramble_seq, -positive_scramble_seq);
            } else if (gold_value == 3) {
                scrambled_val = make_complex<Tcomplex>::create(-positive_scramble_seq, -positive_scramble_seq);
            }
            Tcomplex dmrs_temp_out1[MAX_DL_PORTS] = {{0, 0}};
            Tcomplex dmrs_temp_out2[MAX_DL_PORTS] = {{0, 0}};

            bool cdm_grps_no_data_1 = params[TB_id].dmrsCdmGrpsNoData1;

            for (int layer_id = 0; layer_id < params[TB_id].num_layers; ++layer_id) {
                Tcomplex symbol_to_read = scrambled_val;
    #else
                int mult_0 = (gold_value & 0x2) ? -1 : 1;
                int mult_1 = (gold_value & 0x1) ? -1 : 1;
                Tcomplex symbol_to_read = make_complex<Tcomplex>::create(mult_1 * positive_scramble_seq, mult_0 * positive_scramble_seq);
    #endif

                // Do remap dmrs part
                const int port_idx = params[TB_id].port_ids[layer_id];
                const int delta =  (port_idx >> 1) & 0x1U; // DRMS config. type 1 only; valid options 0 or 1
                const int fOCC_flag = (port_idx & 0x1U); // even port_idx, i.e., fOCC_flag == 0, means Wf(k') is all +1; odd port means Wf(k') is +1, -1, alternating
                const int tOCC_flag = (port_idx >> 2) & 0x1U; // port_idx < 4, i.e., tOCC_flag == 0, means Wt(l') is all +1; port_idx >= 4 means Wt(l') is +1, -1

                if ((fOCC_flag == 1) && ((tid_x & 0x1U) == 0x1U)){
                    symbol_to_read =  make_complex<Tcomplex>::create(-symbol_to_read.x, -symbol_to_read.y);
                }
                if (((symbol_id & 0x1) == 1) && (tOCC_flag == 1)) {
                    symbol_to_read =  make_complex<Tcomplex>::create(-symbol_to_read.x, -symbol_to_read.y);
                }

                //Each thread writes 2 consecutive symbols, one zero-ed out, if cdm_grps_no_data != 1
                Tcomplex symbol_to_write_0, symbol_to_write_1;
                if (delta == 0) {  // {symbol_read, 0}
                    symbol_to_write_0 = symbol_to_read;
                    symbol_to_write_1 = make_complex<Tcomplex>::create(0, 0);
                } else { // (delta == 1) {0, symbol_read}
                    symbol_to_write_0 = make_complex<Tcomplex>::create(0, 0);
                    symbol_to_write_1 = symbol_to_read;
                }
                if(!enablePrecoding)
                {
                    const int layer = calculate_layer_id(port_idx, params[TB_id].n_scid, params[TB_id].nlAbove16);
                    uint32_t output_index = (CUPHY_N_TONES_PER_PRB * BWP_PRBs * (OFDM_SYMBOLS_PER_SLOT * layer + symbol_loc[symbol_id])) +
                                            (CUPHY_N_TONES_PER_PRB * start_Rb) + (new_tidx << 1);
                    if (!cdm_grps_no_data_1)
                    {
                        dmrs_output[output_index] =  symbol_to_write_0;
                        dmrs_output[output_index + 1] = symbol_to_write_1;
                    }
                    else
                    {
                        // The other RE will be written by the fused rate-matching and modulation kernel
                        dmrs_output[output_index + delta] =  symbol_to_read;
                    }
                }
                else
                {
                    if(!params[TB_id].enablePrcdBf)
                    {
                        const int layer = calculate_layer_id(port_idx, params[TB_id].n_scid, params[TB_id].nlAbove16);
                        uint32_t output_index = (CUPHY_N_TONES_PER_PRB * BWP_PRBs * (OFDM_SYMBOLS_PER_SLOT * layer + symbol_loc[symbol_id])) +
                                                (CUPHY_N_TONES_PER_PRB * start_Rb) + (new_tidx << 1);
                        // No need to perform atomicAdd with 0 for the alternate RE
                        atomicAdd(&dmrs_output[output_index + delta], symbol_to_read);
                    }
                    else
                    {
                        #pragma loop unroll
                        for(int out_port = 0; out_port < params[TB_id].Np; ++out_port) {
                            __half2 matCoeff = params[TB_id].pmW[layer_id *  params[TB_id].Np + out_port];
                            dmrs_temp_out1[out_port] = __hcmadd(symbol_to_write_0, matCoeff, dmrs_temp_out1[out_port]);
                            dmrs_temp_out2[out_port] = __hcmadd(symbol_to_write_1, matCoeff, dmrs_temp_out2[out_port]);
                        }
                    }
                }
            }

            if(enablePrecoding && params[TB_id].enablePrcdBf)
            {
                #pragma loop unroll
                for(int out_port = 0; out_port < params[TB_id].Np; ++out_port) {
                    uint32_t output_index = (CUPHY_N_TONES_PER_PRB * BWP_PRBs * (OFDM_SYMBOLS_PER_SLOT * out_port + symbol_loc[symbol_id])) +
                                            (CUPHY_N_TONES_PER_PRB * start_Rb) + (new_tidx << 1);
                    atomicAdd(&dmrs_output[output_index], dmrs_temp_out1[out_port]);
                    atomicAdd(&dmrs_output[output_index + 1], dmrs_temp_out2[out_port]);
                }
            }
        }
    }
}


cuphyStatus_t CUPHYWINAPI cuphySetupPdschDmrs(cuphyPdschDmrsLaunchConfig_t pdschDmrsLaunchConfig,
                                              PdschDmrsParams * dmrs_params,
                                              int num_TBs,
                                              uint8_t enable_precoding,
                                              cuphyTensorDescriptor_t dmrs_output_desc,
                                              void*                   dmrs_output_addr,
                                              void* cpu_desc,
                                              void* gpu_desc,
                                              uint8_t enable_desc_async_copy,
                                              cudaStream_t            strm)
{

    if ((dmrs_params == nullptr) || (cpu_desc == nullptr) || (gpu_desc == nullptr)) {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }

    pdschDmrsLaunchConfig->m_kernelArgs[0] = &(pdschDmrsLaunchConfig->m_desc);
    pdschDmrsLaunchConfig->m_kernelNodeParams.extra = nullptr;
    pdschDmrsLaunchConfig->m_kernelNodeParams.kernelParams = &(pdschDmrsLaunchConfig->m_kernelArgs[0]);

    //Set up CPU descriptor. Assumes it has been pre-allocated. Easier for a pipeline, than for a standalone component setting.
    pdschDmrsDescr_t& desc = *(static_cast<pdschDmrsDescr_t*>(cpu_desc));
    desc.dmrs_params = dmrs_params; // device pointer
    desc.num_TBs = num_TBs;

    // Optional descriptor copy to GPU memory
    // When running as part of a pipeline, it's better to do a single copy of all descriptors in the pipeline.
    if (enable_desc_async_copy) {
        cudaError_t res = cudaMemcpyAsync(gpu_desc, cpu_desc, sizeof(pdschDmrsDescr_t), cudaMemcpyHostToDevice, strm);
        if (res != cudaSuccess) {
            return CUPHY_STATUS_MEMCPY_ERROR;
        }
    }

    pdschDmrsLaunchConfig->m_desc = static_cast<pdschDmrsDescr_t*>(gpu_desc);

    const uint32_t threads = 256; //FIXME need to update shared mem. size per group accordingly
    int max_Rbs = 273; // FIXME potentially adjust based on numerology
    int max_Nf = max_Rbs * CUPHY_N_TONES_PER_PRB;
    int re_blocks_x = div_round_up(max_Nf / 2, (int) threads);

    // Always update the DMRS kernel function, as the presence of precoding for the cells in a cell group
    // can change throughout the cuPHY PDSCH object's lifetime.
    // This is also the case for standalone components.
    cudaFunction_t dmrs_device_function;
    cudaError_t res = cudaSuccess;
    if(enable_precoding != 0)
    {
        MemtraceDisableScope md;
        res = cudaGetFuncBySymbol(&dmrs_device_function, reinterpret_cast<void*>(fused_dmrs<true, __half2>));
    }
    else
    {
        MemtraceDisableScope md;
        res = cudaGetFuncBySymbol(&dmrs_device_function, reinterpret_cast<void*>(fused_dmrs<false, __half2>));
    }
    if (res != cudaSuccess) { // Currently cudaGetFuncBySymbol only returns cudaSuccess.
        return CUPHY_STATUS_INTERNAL_ERROR;
    }
    pdschDmrsLaunchConfig->m_kernelNodeParams.func = dmrs_device_function;

    pdschDmrsLaunchConfig->m_kernelNodeParams.blockDimX = threads;
    pdschDmrsLaunchConfig->m_kernelNodeParams.blockDimY = 1;
    pdschDmrsLaunchConfig->m_kernelNodeParams.blockDimZ = 1;
    pdschDmrsLaunchConfig->m_kernelNodeParams.gridDimX = re_blocks_x;
    pdschDmrsLaunchConfig->m_kernelNodeParams.gridDimY = num_TBs;
    pdschDmrsLaunchConfig->m_kernelNodeParams.gridDimZ = 1;
    pdschDmrsLaunchConfig->m_kernelNodeParams.sharedMemBytes = 0;

    return CUPHY_STATUS_SUCCESS;
}

__global__ void genCsirsReMap(pdschCsirsPrepDescr_t* p_desc)
{
    pdschCsirsPrepDescr_t& desc = *p_desc;
    uint16_t*       reMapArray  = desc.reMapArray;
    cuphyCsirsRrcDynPrm_t * csirsParams = desc.csirsParams;
    uint32_t* offsets    = desc.offsets;
    const uint32_t total_offsets = desc.totalOffsets;
    uint32_t* cellIndexArray = desc.cellIndexArray;
    uint16_t  max_BWP = desc.maxBWP;

    int globalIndex = blockIdx.x * blockDim.x + threadIdx.x;
    int paramNum = 0;

    // Offset array is of size CUPHY_CSIRS_MAX_NUM_PARAMS(32) + 1
    if(globalIndex >= total_offsets)
    {
        return;
    }

    // find the CSI-RS parameter number for this thread using offset array
    // ToDo: Consider removing dependency on offset array by repeating offset calculation
    // in kernel code.
    while(globalIndex >= offsets[paramNum + 1])
    {
        ++paramNum;
    }

    // local index for this thread within threads spawned for a CSI-RS parameter set
    int localIndex = globalIndex - offsets[paramNum];

    // get the parameter from array
    //ZpCsirsParams& params = csirsParams[paramNum];
    cuphyCsirsRrcDynPrm_t& params = csirsParams[paramNum];
    int rowNum = params.row;

    uint32_t cell_index_and_BWP = cellIndexArray[paramNum];
    int cell_index = (cell_index_and_BWP >> 16) & 0xFFFF;
    int cell_BWP = cell_index_and_BWP & 0xFFFF;
    size_t per_cell_xtf_re_map_elements = max_BWP * CUPHY_N_TONES_PER_PRB * OFDM_SYMBOLS_PER_SLOT + OFDM_SYMBOLS_PER_SLOT;
    uint16_t* reMap = reMapArray +  cell_index * per_cell_xtf_re_map_elements;

    // rowNum is 1 index, the table is 0-indexed
    CsirsSymbLocRow& rowData = constRowDataCsirs[rowNum - 1];

    uint nRB = params.nRb;
    uint lenKBarLBar = rowData.lenKBarLBar;
    uint lenKPrime = rowData.lenKPrime;
    uint lenLPrime = rowData.lenLPrime;

    // needs to be done as number of threads for a parameter are warp size aligned
    // to avoid divergence
    if(localIndex >= nRB * lenKBarLBar * lenKPrime * lenLPrime)
    {
        return;
    }

    // get index values for kBar-LBar array, lPrime and kPrime
    uint idxRB =  localIndex / (lenKBarLBar * lenKPrime * lenLPrime);
    uint lenLKBarPrime = localIndex - idxRB * lenKBarLBar * lenKPrime * lenLPrime;

    idxRB += params.startRb;
    bool gen_even_rb = (params.freqDensity == 0) ? 1 : 0;
    bool isEvenRB = (idxRB & 1) == 0;
    if (params.freqDensity <= 1)
    {
        if ((gen_even_rb && !isEvenRB) || (!gen_even_rb && isEvenRB))
            return;
    }

    uint idxKBarLBar = lenLKBarPrime / (lenKPrime * lenLPrime);
    uint lenLKPrime = lenLKBarPrime - idxKBarLBar * lenKPrime * lenLPrime;
    const uint lenKPrime_minus_one = ((lenKPrime - 1) & 0x1);
    const uint lPrime = lenLKPrime >> lenKPrime_minus_one;
    const uint kPrime = lenLKPrime & lenKPrime_minus_one;

    uint ki_index = rowData.kIndices[idxKBarLBar];
    __builtin_assume(ki_index <  CUPHY_CSIRS_MAX_KI_INDEX_LENGTH); // at most 6
    uint16_t freqDomain = params.freqDomain;
    uint ki_multiplier = 2;
    if (params.row == 4) ki_multiplier = 4;
    else if ((params.row == 1) || (params.row == 2)) ki_multiplier = 1;

    int rightmost_set_index = 0;
    for (int i = 0; i <= ki_index; i++) { // Loop at most 6 times; ki_index < CUPHY_CSIRS_MAX_KI_INDEX_LENGTH
        rightmost_set_index = __ffs(freqDomain);
        freqDomain >>= rightmost_set_index;
        freqDomain <<= rightmost_set_index;
    }
    uint ki_val = (rightmost_set_index - 1) * ki_multiplier;
    uint kBar = ki_val + rowData.kOffsets[idxKBarLBar];
    uint lBar = ((rowData.lIndices[idxKBarLBar] == 0) ? params.symbL0 : params.symbL1) + rowData.lOffsets[idxKBarLBar];

    uint k = kBar + kPrime + idxRB * CUPHY_N_TONES_PER_PRB;
    uint l = lBar + lPrime;

    reMap[k + l *  cell_BWP * CUPHY_N_TONES_PER_PRB] = 1;

    uint16_t* re_map_symbols = reMap + max_BWP*CUPHY_N_TONES_PER_PRB * OFDM_SYMBOLS_PER_SLOT;
    re_map_symbols[l] = 1;
    //printf("threadIdx.x %d, blockIdx.x %d, RE map @ symbol l %d, RE %d = 1\n", threadIdx.x, blockIdx.x, l, k);
    /*if ((blockIdx.x == 0) && (threadIdx.x < 4))
    printf("threadIdx.x %d, blockIdx.x %d, idxKBarLBar %d, kBar %d, kPrime %d, k %d\n",
            threadIdx.x, blockIdx.x, idxKBarLBar, kBar, kPrime, k);
    */
}

__global__ void postProcessCsirsReMap(pdschCsirsPrepDescr_t* p_desc)
{
    pdschCsirsPrepDescr_t& desc = *p_desc;
    uint16_t*       reMapArray  = desc.reMapArray;
    PdschUeGrpParams*  d_ue_grp_params = desc.ueGrpParams;
    PdschDmrsParams*   d_dmrs_params = desc.dmrsParams;
    uint16_t  max_BWP = desc.maxBWP;
    cg::thread_block_tile<32> tile = cg::tiled_partition<32>(cg::this_thread_block());

    __shared__ uint32_t warp_start_pos_wr[2*OFDM_SYMBOLS_PER_SLOT];
    uint32_t* warp_start_pos_rd = &warp_start_pos_wr[OFDM_SYMBOLS_PER_SLOT];
    if (threadIdx.x < 2*OFDM_SYMBOLS_PER_SLOT) {
        warp_start_pos_wr[threadIdx.x] = 0;
    }

    int ue_group_idx= blockIdx.x;
    int symbol_idx = threadIdx.x >> 5; // A warp in the block processes one symbol_idx

    // get one TB associated with this UE group
    int TB_idx = d_ue_grp_params[ue_group_idx].tb_idx;

    // FIXME Need to reset skipped_REs somewhere otherwise back to back runs without setup can result in illegal mem. accesses
    for (int i = threadIdx.x; i < OFDM_SYMBOLS_PER_SLOT; i += blockDim.x) {
       d_ue_grp_params[ue_group_idx].cumulative_skipped_REs[i] = 0;
    }

    int cell_index = d_dmrs_params[TB_idx].cell_index_in_cell_group;
    size_t per_cell_xtf_re_map_elements = max_BWP * CUPHY_N_TONES_PER_PRB * OFDM_SYMBOLS_PER_SLOT + OFDM_SYMBOLS_PER_SLOT;
    uint16_t* d_cell_xtf_re_map = reMapArray +  cell_index * per_cell_xtf_re_map_elements;

    uint16_t* re_map_symbols = d_cell_xtf_re_map + max_BWP*CUPHY_N_TONES_PER_PRB * OFDM_SYMBOLS_PER_SLOT;
    if (re_map_symbols[symbol_idx] == 0) return;

    //Potential future opt. option. Not implemented; Is symbol_idx a data symbol for this TB?  Check the data_sym_loc of that TB.
    uint64_t data_sym_loc = d_dmrs_params[TB_idx].data_sym_loc;
    bool found = false;
    int data_symbol = 0;
    for (int i = 0; !found && i < d_dmrs_params[TB_idx].num_data_symbols; i++) {
        if (symbol_idx == ((data_sym_loc >> (4*i)) & 0xf)) {
            data_symbol = i;
            found = true;
        }
    }
    if (!found) return; // not a data symbol

    const int lane_idx = threadIdx.x & 0x1F;
    const int all_Rbs_symbols = d_dmrs_params[TB_idx].num_BWP_PRBs * CUPHY_N_TONES_PER_PRB;

    int skipped_symbol_REs = 0;
    if  (d_dmrs_params[TB_idx].resourceAlloc) { // Resource Allocation Type 1
        uint32_t start_Rb = d_dmrs_params[TB_idx].start_Rb;
        uint32_t num_Rbs  = d_dmrs_params[TB_idx].num_Rbs;

        int num_warp_iters_needed = (num_Rbs*CUPHY_N_TONES_PER_PRB + 31)/32;
        uint32_t start_re_map_index = start_Rb * CUPHY_N_TONES_PER_PRB + symbol_idx * all_Rbs_symbols;

        __syncthreads(); // synchronize before accessing warp_start_pos_wr array; can't init. shared memory here using threadIdx.x < OFDM_SYMBOLS_PER_SLOT because threads  may not be active

        // At most 103 iterations needed if a single UE group has all 273 PRBs allocated to it
        for (int warp_iter = 0; warp_iter < num_warp_iters_needed; warp_iter++) {

            int tid = lane_idx + warp_iter * 32;

            uint32_t re_map_index = start_re_map_index + tid;
            uint16_t read_val = (tid < num_Rbs * CUPHY_N_TONES_PER_PRB) ? d_cell_xtf_re_map[re_map_index] : 1;
            uint32_t outcome = __match_any_sync(0xffffffff, read_val);
            __syncwarp();
            uint32_t mask_of_1_bits = __brev((read_val == 0) ? ~outcome : outcome); // with brev, most significant bit is for RE 0

            // Find number of zeros to left of current position
            int bits_set_to_zero_to_the_left = lane_idx - __popc(mask_of_1_bits >> (32 -  lane_idx));
            int position_to_write_to = bits_set_to_zero_to_the_left + warp_start_pos_wr[symbol_idx] + start_re_map_index; //offset from some global one TBD
            int value_to_write = tid - bits_set_to_zero_to_the_left - warp_start_pos_wr[symbol_idx]; // only write if you had read 0
            if (read_val == 0) {
                d_cell_xtf_re_map[position_to_write_to] = value_to_write;
                //if (blockIdx.y == 0)
                //printf("threadIdx.x %d, warp_iter %d, ue_group %d, RB %d, RE %d, re_map[%d] = %d\n", threadIdx.x, warp_iter, blockIdx.y, position_to_write_to/12, position_to_write_to % 12, position_to_write_to, value_to_write);
            }
            if (lane_idx == 31) warp_start_pos_wr[symbol_idx] += (bits_set_to_zero_to_the_left + (read_val == 0 ? 1 : 0));
            __syncwarp();
        }

        skipped_symbol_REs = num_Rbs * CUPHY_N_TONES_PER_PRB - warp_start_pos_wr[symbol_idx];
    } else { // Resource Allocation Type 0
        // Find min and max allocated RBs to bound remapping
        const int BWP_start = d_dmrs_params[TB_idx].BWP_start_PRB;
        uint32_t rbmap_element = 0;
        if(lane_idx < MAX_RBMASK_UINT32_ELEMENTS)
        {
            rbmap_element = d_dmrs_params[TB_idx].rbBitmap[lane_idx];
        }
        uint16_t total_alloc_REs = cg::reduce(tile, __popc(rbmap_element), cg::plus<int>())*CUPHY_N_TONES_PER_PRB;
        uint32_t rb_elem_mask = __match_any_sync(0xFFFFFFFF,(rbmap_element!=0));
        rb_elem_mask = (rbmap_element!=0) ? rb_elem_mask : ~rb_elem_mask;
        uint8_t rb_elem_idx_low = __ffs(rb_elem_mask)-1;
        uint8_t rb_elem_idx_hi  = 31 - __clz(rb_elem_mask);
        uint16_t min_rb = (rb_elem_idx_low << 5) + __ffs(d_dmrs_params[TB_idx].rbBitmap[rb_elem_idx_low])-1;
        uint16_t max_rb = (rb_elem_idx_hi << 5)  + (31 - __clz(d_dmrs_params[TB_idx].rbBitmap[rb_elem_idx_hi]));

        uint32_t start_re_map_index = min_rb * CUPHY_N_TONES_PER_PRB + symbol_idx * all_Rbs_symbols;

        __syncthreads(); // synchronize before accessing shared memory (warp_start_pos_wr/rd array)

        uint8_t rb_bit_rd = 0;
        uint8_t rb_bit_wr = 0;
        const uint16_t max_re_offset = ((max_rb-min_rb+1) * CUPHY_N_TONES_PER_PRB)-1;
        const uint8_t  lane_shift = 32 - lane_idx;
        uint8_t read_step = 0;
        int16_t value_to_write = 0;
        int32_t position_to_write_to = 0;
        uint16_t write_cnt = 0;

        // This function has a look-ahead and write indices to determine which REs get mapped to where based on avoiding
        // unallocated RBs and REs reserved for CSI-RS.  The output should generate a map only occupying the REs which
        // correspond to active rbBitmap REs for a UE group.  The offsets contained in that remapping should also only map to
        // active elements of that UE group's rbBitmap.  For example, if RE's 0-11 and 24-35 are active for UE group 0,
        // remap[x] will only have values such that (remap[x] + x) will also be within RE's 0-11 or 24-35

        // Possible optimization: The remapping currently increments over all REs within the range of allocated REs but
        //      unallocated REs can be skipped CUPHY_N_TONES_PER_PRB at a time

        for (int i = lane_idx; i - lane_idx - warp_start_pos_rd[symbol_idx] < max_re_offset; i+=read_step) {
            uint32_t re_map_index = start_re_map_index + i;
            {
                value_to_write = 0;
                position_to_write_to = 0;

                // Leading index - remap to (read)
                int16_t freq_idx_rd = (re_map_index - warp_start_pos_rd[symbol_idx] - symbol_idx*all_Rbs_symbols)/CUPHY_N_TONES_PER_PRB;
                rb_bit_rd = ((d_dmrs_params[TB_idx].rbBitmap[freq_idx_rd >> 5] >> (freq_idx_rd & 0x1F)) & 0x1);
                // Lagging index - remap from (write)
                int16_t freq_idx_wr = (re_map_index - warp_start_pos_wr[symbol_idx] - symbol_idx*all_Rbs_symbols)/CUPHY_N_TONES_PER_PRB;
                rb_bit_wr = (d_dmrs_params[TB_idx].rbBitmap[freq_idx_wr >> 5] >> (freq_idx_wr & 0x1F)) & 0x1;
                // Advance the read delta when the remap to offset isn't valid
                uint16_t read_val = d_cell_xtf_re_map[re_map_index - warp_start_pos_rd[symbol_idx] + BWP_start*CUPHY_N_TONES_PER_PRB];
                bool remap_idx_skp = !(rb_bit_rd && read_val==0); // Skip remap idx
                uint32_t outcome_skp = __match_any_sync(0xffffffff, remap_idx_skp);
                // Advance the write delta when the remap RB isn't active
                uint32_t outcome_wr = __match_any_sync(0xffffffff, rb_bit_wr);
                __syncwarp();
                // flip value based on current lane so each thread has the same value
                // and reverse so most significant bit is for RE 0 and set if rb_bit_wr is false
                uint32_t mask_of_bits_wr = __brev((-rb_bit_wr) ^ ~outcome_wr);
                uint8_t  rd_bits_set_to_the_left = lane_idx - __popc(mask_of_bits_wr >> (lane_shift));
                // Count the number of lanes that match lane 0 (i.e. number of values to be written or skipped this cycle)
                read_step = __clz((-((mask_of_bits_wr & 0x80000000)!=0)) ^ mask_of_bits_wr);

                uint32_t mask_of_bits_skp = __brev((-remap_idx_skp) ^ outcome_skp);
                // Find number of set bits to left of current position
                uint8_t  wr_bits_set_to_the_left = lane_idx - __popc((mask_of_bits_skp & mask_of_bits_wr) >> (lane_shift));
                // count all the values of valid remap to values until the 1st disable RE for a remap from index
                int write_step = __popc(~(mask_of_bits_skp) & __brev((1ULL << __clz(~mask_of_bits_wr)) - 1));

                if(rb_bit_wr && rb_bit_rd && read_val==0 && (mask_of_bits_wr & 0x80000000)) {
                    position_to_write_to = re_map_index - (warp_start_pos_wr[symbol_idx] + wr_bits_set_to_the_left + (remap_idx_skp && rb_bit_wr)) + BWP_start*CUPHY_N_TONES_PER_PRB;
                    value_to_write = (warp_start_pos_wr[symbol_idx] + wr_bits_set_to_the_left + (remap_idx_skp && rb_bit_wr)) - (warp_start_pos_rd[symbol_idx] + rd_bits_set_to_the_left + !rb_bit_wr);
                    d_cell_xtf_re_map[position_to_write_to] = value_to_write;
                    write_cnt++;
                }
                if (lane_idx == 31 ) {
                    warp_start_pos_wr[symbol_idx] += write_step;
                    if(!(mask_of_bits_wr & 0x80000000)) {
                        warp_start_pos_rd[symbol_idx] += read_step;
                    }
                }
                __syncwarp();
            }
        }
        skipped_symbol_REs = total_alloc_REs - cg::reduce(tile, write_cnt, cg::plus<uint16_t>());
    }
    // Update the per UE group cumulative parameters for this data symbol and the ones after it.
    // Only valid data symbols (blockIdx.x) exist at this point.
    int data_symbol_cnt =  d_dmrs_params[TB_idx].num_data_symbols - data_symbol;
    if (lane_idx < data_symbol_cnt) {
        atomicAdd(&d_ue_grp_params[ue_group_idx].cumulative_skipped_REs[lane_idx + data_symbol], skipped_symbol_REs);
    }
}

__global__ void zero_memset_kernel(pdschCsirsPrepDescr_t* p_desc)
{
    pdschCsirsPrepDescr_t& desc = *p_desc;
    void* d_buffer = desc.reMapArray;
    const int d_buffer_size = desc.bufferSizeInBytes;

    const uint4 val = {0, 0, 0, 0};
    // assumption that d_buffer is uint4_t aligned

    uint4* d_buffer_addr = (uint4*)d_buffer;
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < (d_buffer_size >> 4)) { // 4 is to divide by sizeof(uint4) as d_buffer_size is in bytes
        d_buffer_addr[tid] = val;
    }

    //Handle leftover bytes
    int leftover_bytes = d_buffer_size & 0xF; // modulo sizeof(uint4) = 16
    if (tid < leftover_bytes) {
        uint8_t* d_buffer_byte_addr = (uint8_t*)d_buffer + (d_buffer_size - leftover_bytes);
        d_buffer_byte_addr[tid] = 0;
    }
}


cuphyStatus_t CUPHYWINAPI cuphySetupPdschCsirsPreprocessing(cuphyPdschCsirsPrepLaunchConfig_t pdschCsirsPrepLaunchConfig,
                                              void*                   re_map_array_addr,
                                              cuphyCsirsRrcDynPrm_t*  d_params,
                                              size_t                  numParams,
                                              uint32_t                total_offsets,
                                              uint32_t*               d_offsets,
                                              uint32_t*               d_cellIndex,
                                              uint16_t                num_ue_groups,
                                              PdschUeGrpParams*       d_ue_grp_params,
                                              PdschDmrsParams*        d_dmrs_params,
                                              uint16_t                max_BWP,
                                              uint16_t                num_cells,
                                              void*                   cpu_desc,
                                              void*                   gpu_desc,
                                              uint8_t                 enable_desc_async_copy,
                                              cudaStream_t            stream)
{
    if ((!re_map_array_addr) || (!d_params) \
        || (!d_cellIndex) || (!d_ue_grp_params) || (!d_dmrs_params)
        || (total_offsets == 0) || (numParams == 0) || (!d_offsets) || (num_ue_groups == 0) || (!cpu_desc) || (!gpu_desc))
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }

    pdschCsirsPrepLaunchConfig->m_kernelArgs[0] = &(pdschCsirsPrepLaunchConfig->m_desc);
    pdschCsirsPrepLaunchConfig->m_kernelNodeParams[0].extra = nullptr;
    pdschCsirsPrepLaunchConfig->m_kernelNodeParams[0].kernelParams = &(pdschCsirsPrepLaunchConfig->m_kernelArgs[0]);

    pdschCsirsPrepLaunchConfig->m_kernelNodeParams[1].extra = nullptr;
    pdschCsirsPrepLaunchConfig->m_kernelNodeParams[1].kernelParams = &(pdschCsirsPrepLaunchConfig->m_kernelArgs[0]);

    pdschCsirsPrepLaunchConfig->m_kernelNodeParams[2].extra = nullptr;
    pdschCsirsPrepLaunchConfig->m_kernelNodeParams[2].kernelParams = &(pdschCsirsPrepLaunchConfig->m_kernelArgs[0]);

    int per_cell_xtf_re_map_elements = max_BWP * CUPHY_N_TONES_PER_PRB * OFDM_SYMBOLS_PER_SLOT + OFDM_SYMBOLS_PER_SLOT;
    if((reinterpret_cast<uintptr_t>(re_map_array_addr) & 0xF) != 0) { // Ensure re_map_array_addr has uint4 alignment
        return CUPHY_STATUS_INVALID_ARGUMENT;
        //return CUPHY_STATUS_UNSUPPORTED_ALIGNMENT;
    }
    int buffer_size_in_bytes = per_cell_xtf_re_map_elements * num_cells * sizeof(uint16_t);


    //Set up CPU descriptor. Assumes it has been pre-allocated.
    pdschCsirsPrepDescr_t& desc = *(static_cast<pdschCsirsPrepDescr_t*>(cpu_desc));
    desc.reMapArray             = (__uint16_t*)re_map_array_addr;
    desc.bufferSizeInBytes      = buffer_size_in_bytes;
    desc.csirsParams            = d_params;
    desc.numParams              = numParams;
    desc.offsets                = d_offsets;
    desc.totalOffsets           = total_offsets;
    desc.cellIndexArray         = d_cellIndex;
    desc.ueGrpParams            = d_ue_grp_params;
    desc.dmrsParams             = d_dmrs_params;
    desc.maxBWP                 = max_BWP;


    // Optional descriptor copy to GPU memory
    // When running as part of a pipeline, it's better to do a single copy of all descriptors in the pipeline.
    if (enable_desc_async_copy) {
        cudaError_t res = cudaMemcpyAsync(gpu_desc, cpu_desc, sizeof(pdschCsirsPrepDescr_t), cudaMemcpyHostToDevice, stream);
        if (res != cudaSuccess) {
            return CUPHY_STATUS_MEMCPY_ERROR;
        }
    }

    pdschCsirsPrepLaunchConfig->m_desc = static_cast<pdschCsirsPrepDescr_t*>(gpu_desc);

    if (pdschCsirsPrepLaunchConfig->m_kernelNodeParams[0].func == nullptr) {
        cudaFunction_t csirs_kernel1_device_function;
        MemtraceDisableScope md;
        cudaError_t res = cudaGetFuncBySymbol(&csirs_kernel1_device_function, reinterpret_cast<void*>(genCsirsReMap));
        if (res != cudaSuccess) { // Currently cudaGetFuncBySymbol only returns cudaSuccess.
            return CUPHY_STATUS_INTERNAL_ERROR;
        }
        pdschCsirsPrepLaunchConfig->m_kernelNodeParams[0].func = csirs_kernel1_device_function;
    }

    const int block_size = 128;
    pdschCsirsPrepLaunchConfig->m_kernelNodeParams[0].blockDimX = block_size;
    pdschCsirsPrepLaunchConfig->m_kernelNodeParams[0].blockDimY = 1;
    pdschCsirsPrepLaunchConfig->m_kernelNodeParams[0].blockDimZ = 1;
    pdschCsirsPrepLaunchConfig->m_kernelNodeParams[0].gridDimX = (total_offsets + block_size - 1) / block_size;
    pdschCsirsPrepLaunchConfig->m_kernelNodeParams[0].gridDimY = 1,
    pdschCsirsPrepLaunchConfig->m_kernelNodeParams[0].gridDimZ = 1;
    pdschCsirsPrepLaunchConfig->m_kernelNodeParams[0].sharedMemBytes = 0;

    if (pdschCsirsPrepLaunchConfig->m_kernelNodeParams[1].func == nullptr) {
        cudaFunction_t csirs_kernel2_device_function;
        MemtraceDisableScope md;
        cudaError_t res = cudaGetFuncBySymbol(&csirs_kernel2_device_function, reinterpret_cast<void*>(postProcessCsirsReMap));
        if (res != cudaSuccess) { // Currently cudaGetFuncBySymbol only returns cudaSuccess.
            return CUPHY_STATUS_INTERNAL_ERROR;
        }
        pdschCsirsPrepLaunchConfig->m_kernelNodeParams[1].func = csirs_kernel2_device_function;
    }

    pdschCsirsPrepLaunchConfig->m_kernelNodeParams[1].blockDimX = 32 * OFDM_SYMBOLS_PER_SLOT;
    pdschCsirsPrepLaunchConfig->m_kernelNodeParams[1].blockDimY = 1;
    pdschCsirsPrepLaunchConfig->m_kernelNodeParams[1].blockDimZ = 1;
    pdschCsirsPrepLaunchConfig->m_kernelNodeParams[1].gridDimX = num_ue_groups;
    pdschCsirsPrepLaunchConfig->m_kernelNodeParams[1].gridDimY = 1,
    pdschCsirsPrepLaunchConfig->m_kernelNodeParams[1].gridDimZ = 1;
    pdschCsirsPrepLaunchConfig->m_kernelNodeParams[1].sharedMemBytes = 0;

    const int num_threads = 1024;
    int blocks = (buffer_size_in_bytes + sizeof(uint4)*num_threads -1) / (sizeof(uint4)*num_threads);

    if (pdschCsirsPrepLaunchConfig->m_kernelNodeParams[2].func == nullptr) {
        cudaFunction_t csirs_kernel3_device_function;
        MemtraceDisableScope md;
        cudaError_t res = cudaGetFuncBySymbol(&csirs_kernel3_device_function, reinterpret_cast<void*>(zero_memset_kernel));
        if (res != cudaSuccess) { // Currently cudaGetFuncBySymbol only returns cudaSuccess.
            return CUPHY_STATUS_INTERNAL_ERROR;
        }
        pdschCsirsPrepLaunchConfig->m_kernelNodeParams[2].func = csirs_kernel3_device_function;
    }

    pdschCsirsPrepLaunchConfig->m_kernelNodeParams[2].blockDimX = num_threads;
    pdschCsirsPrepLaunchConfig->m_kernelNodeParams[2].blockDimY = 1;
    pdschCsirsPrepLaunchConfig->m_kernelNodeParams[2].blockDimZ = 1;
    pdschCsirsPrepLaunchConfig->m_kernelNodeParams[2].gridDimX = blocks;
    pdschCsirsPrepLaunchConfig->m_kernelNodeParams[2].gridDimY = 1,
    pdschCsirsPrepLaunchConfig->m_kernelNodeParams[2].gridDimZ = 1;
    pdschCsirsPrepLaunchConfig->m_kernelNodeParams[2].sharedMemBytes = 0;

    return CUPHY_STATUS_SUCCESS;
 }

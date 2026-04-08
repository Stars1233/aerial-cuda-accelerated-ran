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

#include "pdsch_tx.hpp"
#include "utils.cuh"
#include "cuphy_internal.h"

#define DBG_PDSCH_CRC 0
#define PRINT_CONFIG 0
#define PRINT_CSV_FRIENDLY_CONFIG 0 // Printed at the end of the setup phase. TODO Set to 1, if needed.

// Flags to explore impact on setup latency
#define USE_INPUT_ASYNC_COPY 1  // Default for current code is 1. If set to 0, the input buffer (host-pinned)
                                // will be accessed by the prepare_crc_buffers kernel directly.
                                // Ignored when inter-cell batching is enabled for multiple cells in cell group case; that mode always accesses host pinned memory.

#define AVOID_STREAM_CREATION_IN_SETUP 1 // Current default is 1. Performance impact only relevant if PdschTx constructed with identical_LDPC_configs=false
                                         // or if TBs in cell have non identical LDPC configs.
                                         // When set to 1, a pool of (MAX_UES_PER_CELL_GROUP - 1) CUDA streams is created in PdschTx's constructor.
                                         // When set to 0, the required number of CUDA streams will be created, if needed, during setup.

#define REVERT_TO_ASYNC_COPIES 1 // Revert to async copies in case of inter cell batching if 1. Changed so it is 1 by default
#define OLD_GRAPH 0
#define MOVE_DMRS_EARLY 1

#define FUSED_DL_RM_AFTER_DMRS 0 // When 1, the fused_dl_rm_and_modulation kernel waits both for the LDPC encoder and the fused_dmrs kernel to finish.
                                 // When 0, the fused_dl_rm_and_modulation kernel only depends on LDPC encoder and the CSIRS post-processing kernel (if enabled)
                                 // and it can execute in parallel with the fused_dmrs kernel, without being gated by it, as there is no data dependency.


using namespace cuphy;

void print_TB_config_params(PdschPerTbParams* kernel_params, PdschDmrsParams* dmrs_params, int TB_id, int num_TBs, bool layer_mapping, bool scrambling)
{
    PdschPerTbParams TB_params = kernel_params[TB_id];

    // Current codebase expects these config. params to be the same across TBs.
    if(TB_id == 0)
    {
        NVLOGC_FMT(NVLOG_PDSCH, "Config. Parameters shared across all {} TB(s):", num_TBs);
        NVLOGC_FMT(NVLOG_PDSCH, "* layer_mapping is {}", layer_mapping);
        NVLOGC_FMT(NVLOG_PDSCH, "* scrambling is {}", scrambling);
    }

    // Config. parameters that vary across TBs.
    NVLOGC_FMT(NVLOG_PDSCH, "");
    NVLOGC_FMT(NVLOG_PDSCH, "Config. Parameters specific to TB {}: ", TB_id);

    NVLOGC_FMT(NVLOG_PDSCH, "* rv = {}", (int)TB_params.rv);
    NVLOGC_FMT(NVLOG_PDSCH, "* Qm = {}", (int)TB_params.Qm);
    NVLOGC_FMT(NVLOG_PDSCH, "* bg = {}", (int)TB_params.bg);
    NVLOGC_FMT(NVLOG_PDSCH, "* Nl = {}", (int)TB_params.Nl);
    NVLOGC_FMT(NVLOG_PDSCH, "* num_CBs = {}", TB_params.num_CBs);
    NVLOGC_FMT(NVLOG_PDSCH, "* Zc = {}", TB_params.Zc);

    NVLOGC_FMT(NVLOG_PDSCH, "* N = {}", TB_params.N);
    NVLOGC_FMT(NVLOG_PDSCH, "* max G = {}", TB_params.G); //FIXME this is max G
    NVLOGC_FMT(NVLOG_PDSCH, "* max REs = {}", TB_params.max_REs);
    NVLOGC_FMT(NVLOG_PDSCH, "* K = {}", TB_params.K);
    NVLOGC_FMT(NVLOG_PDSCH, "* F = {}", TB_params.F);

    NVLOGC_FMT(NVLOG_PDSCH, "* cinit = {}", TB_params.cinit);
    int TB_layers = TB_params.Nl;
    std::stringstream layers_strm;
    layers_strm << "* layer_map[" << TB_layers << "] = {";
    for(int layer_cnt = 0; layer_cnt < TB_layers; layer_cnt++)
    {
        layers_strm << std::to_string(calculate_layer_id(dmrs_params[TB_id].port_ids[layer_cnt], dmrs_params[TB_id].n_scid, dmrs_params[TB_id].nlAbove16));
        if(layer_cnt != TB_layers - 1)
        {
            layers_strm << ", ";
        }
        else
        {
            layers_strm << "}";
        }
    }
    NVLOGC_FMT(NVLOG_PDSCH, "{}", layers_strm.str().c_str());
}


// Only used in pdsch_tx_multi_cell  TODO remove from here and ideally have that phase-2 example use datasets too similar to phase-3
void cumulative_read_pdsch_static_pars_from_file(cuphyPdschStatPrms_t& pdsch_static_params, hdf5hpp::hdf5_file& input_file, const char* filename, bool ref_check, bool first_call)
{
    hdf5hpp::hdf5_dataset cell_static_dataset = input_file.open_dataset("cellStat_pars");
    int                   num_cells           = cell_static_dataset.get_dataspace().get_dimensions()[0];

    if (first_call) {
        pdsch_static_params.nCells              = 0;
        pdsch_static_params.nMaxCellsPerSlot    = 0;
        pdsch_static_params.nMaxUesPerCellGroup = 0;
        pdsch_static_params.nMaxCBsPerTB        = 0;
        pdsch_static_params.nMaxPrb             = 0;
        pdsch_static_params.stream_priority     = PDSCH_STREAM_PRIORITY;
        //Pre-allocate max cell and dbg arrays
        pdsch_static_params.pCellStatPrms = new cuphyCellStatPrm_t[PDSCH_MAX_CELLS_PER_CELL_GROUP];
        pdsch_static_params.pDbg          = new cuphyPdschDbgPrms_t[PDSCH_MAX_CELLS_PER_CELL_GROUP];
        pdsch_static_params.enableBatchedMemcpy = 1; //FIXME find appropriate default
    }
    cuphyCellStatPrm_t* cell_static_params = &pdsch_static_params.pCellStatPrms[pdsch_static_params.nCells];
    cuphyPdschDbgPrms_t* dbg_params = &pdsch_static_params.pDbg[pdsch_static_params.nCells];
    int cells_base = pdsch_static_params.nCells;
    pdsch_static_params.nCells        += num_cells;
    if (pdsch_static_params.nCells > PDSCH_MAX_CELLS_PER_CELL_GROUP) { NVLOGE_FMT(NVLOG_PDSCH, AERIAL_CUPHY_EVENT, "error! not supported max cells!"); }
    read_cell_static_pars_from_file(cell_static_params, cell_static_dataset, num_cells, cells_base, pdsch_static_params.nMaxPrb);

    for (int i = 0; i < num_cells; i++) {
        dbg_params[i] = {filename, 1 /* TB size check */, ref_check}; // last field not set will be deprecated
    }

    // Update max. values
    pdsch_static_params.nMaxCellsPerSlot = pdsch_static_params.nCells;

    hdf5hpp::hdf5_dataset_elem cell_grp_dyn_config = input_file.open_dataset("cellGrpDyn_pars")[0];
    int16_t new_UEs                                = cell_grp_dyn_config["nUes"].as<uint16_t>();
    pdsch_static_params.nMaxUesPerCellGroup       += new_UEs;

    hdf5hpp::hdf5_dataset_elem  cell_dyn_config = input_file.open_dataset("cellDyn_pars")[0];
    uint8_t cell_testing_mode = 0;
    try {
        cell_testing_mode                   = cell_dyn_config["testModel"].as<uint8_t>();
    } catch(...) {
        cell_testing_mode                   = 0;
    }

    for (int UE_idx = 0; UE_idx < new_UEs; UE_idx++) {
        if (cell_testing_mode == 0) {
            std::string CBs_dataset_name      = "tb" + std::to_string(UE_idx) + "_cbs";
            hdf5hpp::hdf5_dataset CBs_dataset = input_file.open_dataset(CBs_dataset_name.c_str());
            uint16_t num_CBs_for_TB           = CBs_dataset.get_dataspace().get_dimensions()[0];
            pdsch_static_params.nMaxCBsPerTB  = std::max(pdsch_static_params.nMaxCBsPerTB, num_CBs_for_TB);
        } else {
            // If the cell this UE belongs to is in testing mode, then use TB size (in bits) / 25344, rounded up, to determine number of CBs.
            std::string tb_dataset_name      = "tb" + std::to_string(UE_idx) + "_inputdata";
            hdf5hpp::hdf5_dataset tb_dataset  = input_file.open_dataset(tb_dataset_name.c_str());
            uint16_t num_CBs_for_TB           = div_round_up<int>(tb_dataset.get_dataspace().get_dimensions()[1], MAX_ENCODED_CODE_BLOCK_BIT_SIZE);
            pdsch_static_params.nMaxCBsPerTB  = std::max(pdsch_static_params.nMaxCBsPerTB, num_CBs_for_TB);
        }
    }
}

void PdschTx::updateCsirsWorkspaceOffsets(int cells, int params)
{
    // workspace for CSI-RS
    int max_alignment = sizeof(_cuphyCsirsRrcDynPrm);

    workspace_offsets[1] = workspace_offsets[0] + round_up_to_next<int>(params * sizeof(uint32_t), max_alignment); // cell index and BWP per cell for each CSIRS parameter
    workspace_offsets[2] = workspace_offsets[1] + round_up_to_next<int>(params * sizeof(_cuphyCsirsRrcDynPrm), max_alignment);
    workspace_offsets[3] = workspace_offsets[2] + round_up_to_next<int>((params + 1)* sizeof(uint32_t), max_alignment);

    /* Currently max_alignment = 28B
       max workspace in bytes, assuming 8 cells and max 32 CSI-RS params per cell, is: 1036 + 7168 + 1036 = 9240B
       Actual for 8 cells with 1 CSI-RS param. per cell is is 56 + 224 + 56 = 336B
     */
}

void PdschTx::updateCsirsWorkspacePtrs()
{
    // Workspace layout as follows: h_csirs_cell_BWP_index, h_csirs_params, h_offsets

    h_csirs_params = (_cuphyCsirsRrcDynPrm*)(h_workspace.get() + workspace_offsets[1]);
    d_csirs_params = (_cuphyCsirsRrcDynPrm*)(d_workspace.get() + workspace_offsets[1]);

    h_offsets = (uint32_t*)(h_workspace.get() + workspace_offsets[2]);
    d_offsets = (uint32_t*)(d_workspace.get() + workspace_offsets[2]);

    /* NVLOGC_FMT(NVLOG_PDSCH, "h_workspace {:p}, d_workspace {:p}", (void*)h_workspace.get(), (void*)d_workspace.get());
    for (int i = 0; i <= N_CSIRS_COMPONENTS; i++) {
        NVLOGC_FMT(NVLOG_PDSCH, "offset[{}] = {}", i, workspace_offsets[i]);
    }*/
}


cuphyStatus_t PdschTx::d_compute_re_maps()
{
    int cell_grp_csirs_params = dynamic_params->pCellGrpDynPrm[0].nCsiRsPrms;
    if (cell_grp_csirs_params == 0) { // if no CSI-RS parameters are present avoid all CSI-RS computations
        return CUPHY_STATUS_SUCCESS;
    }

    if(cell_grp_csirs_params > (CUPHY_CSIRS_MAX_NUM_PARAMS * max_cells))
    {
        NVLOGE_FMT(NVLOG_PDSCH, AERIAL_CUPHY_EVENT, "Number of CSIRS parameters passed in PDSCH setup ({}) is more than {}.", cell_grp_csirs_params, CUPHY_CSIRS_MAX_NUM_PARAMS * max_cells);
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }

    // Need to reset the Xtf re-maps on every slot
    //CUDA_CHECK(cudaMemsetAsync(d_xtf_re_maps_v2.get(), 0, num_cells * per_cell_xtf_re_map_elements * sizeof(uint16_t), dynamic_params->cuStream));

    _cuphyCsirsRrcDynPrm* csirs_params = dynamic_params->pCellGrpDynPrm[0].pCsiRsPrms; // single cell group

    updateCsirsWorkspaceOffsets(num_cells, cell_grp_csirs_params);
    updateCsirsWorkspacePtrs();

    // To enable bulk H2D copy
    memcpy(h_csirs_params, csirs_params, cell_grp_csirs_params*sizeof(_cuphyCsirsRrcDynPrm));

    // create offset array used to define starting thread for each parameter set
    int offset = 0;
    for (int cell = 0; cell < num_cells; cell++)
    {
        uint16_t  csirs_params_offset = dynamic_params->pCellGrpDynPrm->pCellPrms[cell].csiRsPrmsOffset;
        uint16_t  num_csirs_params = dynamic_params->pCellGrpDynPrm->pCellPrms[cell].nCsiRsPrms;
        uint16_t cell_index = dynamic_params->pCellGrpDynPrm->pCellPrms[cell].cellPrmDynIdx;
        uint16_t static_cell_index = dynamic_params->pCellGrpDynPrm->pCellPrms[cell].cellPrmStatIdx;
        uint16_t cell_BWP = static_params->pCellStatPrms[static_cell_index].nPrbDlBwp;
        for (int i = 0; i < num_csirs_params; i++) {
            int j = csirs_params_offset + i;
            h_csirs_cell_index[j] = (cell_index << 16) | cell_BWP; // could also do this before the updateCsirsWorkspaceOffsets and Ptrs calls as h_csirs_cell_index starts at beginning of workspace
            h_offsets[j] = offset;

            _cuphyCsirsRrcDynPrm* h_param = &(h_csirs_params[j]);
            int numElements = h_param->nRb * ((h_param->row == 1) ? 3 : csirsRowDataNumPorts[h_param->row - 1]);
            // Max. numElements per parameter is 8736
            offset +=  (numElements + 31) & ~31; // aligned by warp size, so that two parameter sets are never part of same warp
        }
    }
    // initialize last parameter with total threads which are needed on GPU
    h_offsets[cell_grp_csirs_params] = offset;

    // Could potentially pull that into the bulk async copy, but then we'd need to copy the overprovisioned workspace. Since it's one copy per cell group, leaving it separately for now.
    CUDA_CHECK(cudaMemcpyAsync(d_workspace.get(), h_workspace.get(), workspace_offsets[N_CSIRS_COMPONENTS], cudaMemcpyHostToDevice, dynamic_params->cuStream));



    cuphyStatus_t status = cuphySetupPdschCsirsPreprocessing(m_csirs_prep_launch_cfg.get(),
                                                             d_xtf_re_maps_v2.get(), //version out of the bulk desc.
                                                           d_csirs_params,
                                                           cell_grp_csirs_params,
                                                           offset,
                                                           d_offsets,
                                                           d_csirs_cell_index,
                                                           dynamic_params->pCellGrpDynPrm->nUeGrps,
                                                           d_ue_grp_params,
                                                           d_dmrs_params,
                                                           max_PRB_BWP,
                                                           num_cells,
                                                           static_cast<void*>(m_component_descrs.getCpuStartAddrs()[PDSCH_CSIRS_PREP]), //CPU desc
                                                           static_cast<void*>(m_component_descrs.getGpuStartAddrs()[PDSCH_CSIRS_PREP]), //GPU desc
                                                           (!bulk_desc_async_copy),
                                                           dynamic_params->cuStream);
    if(status != CUPHY_STATUS_SUCCESS)
    {
        NVLOGE_FMT(NVLOG_PDSCH, AERIAL_CUPHY_EVENT, "CSI-RS pre-processing setup failure! {}", cuphyGetErrorString(status));
    }
    return status;
}


void PdschTx::runPdschCsirsPrepMultipleCells(cudaStream_t cuda_strm)
{
    if (dynamic_params->pCellGrpDynPrm[0].nCsiRsPrms == 0) return;

    CUresult status_memset = launch_kernel(m_csirs_prep_launch_cfg.get()->m_kernelNodeParams[2], cuda_strm);
    CUresult status_k1 = launch_kernel(m_csirs_prep_launch_cfg.get()->m_kernelNodeParams[0], cuda_strm);
    CUresult status_k2 = launch_kernel(m_csirs_prep_launch_cfg.get()->m_kernelNodeParams[1], cuda_strm);
    if((status_k1 != CUDA_SUCCESS) ||
       (status_k2 != CUDA_SUCCESS) || (status_memset != CUDA_SUCCESS))
    {
       throw std::runtime_error("PDSCH CSIRS preprocessing error(s)");
    }

#if 0
    CUDA_CHECK(cudaDeviceSynchronize());
#if 0
    uint16_t h_cell_0_re_map[273*12*14 + 14];
    CUDA_CHECK(cudaMemcpyAsync(h_cell_0_re_map, d_xtf_re_maps_v2.get(), sizeof(uint16_t)*(273*12*14 + 14), cudaMemcpyDeviceToHost, cuda_strm));
    CUDA_CHECK(cudaDeviceSynchronize());
    for (int RB = 0; RB < 273; RB++) {
       for (int re = 0; re < 12; re++) {
           NVLOGC_FMT(NVLOG_PDSCH, "symbol 5, RB {}, RE {}, value {}", RB, re, h_cell_0_re_map[5*273*12 + 12*RB + re]);
       }
    }
#endif

    CUDA_CHECK(cudaMemcpyAsync(h_ue_grp_params, d_ue_grp_params,  dynamic_params->pCellGrpDynPrm->nUeGrps * sizeof(PdschUeGrpParams), cudaMemcpyDeviceToHost, cuda_strm));
    CUDA_CHECK(cudaStreamSynchronize(cuda_strm));

    for(int cell_TB_id = 0; cell_TB_id < per_cell_num_TBs[0]; cell_TB_id++)
    {
      int TB_id = per_cell_TB_params_offset[0] + cell_TB_id;
      PdschUeGrpParams tmp_ue_grp_params      = h_ue_grp_params[h_dmrs_params[TB_id].ueGrp_idx];
      int csi_rs_RE_cnt = tmp_ue_grp_params.cumulative_skipped_REs[h_dmrs_params[TB_id].num_data_symbols - 1];
      NVLOGC_FMT(NVLOG_PDSCH, "cell 0, TB {} has csi_rs_RE_cnt {}", TB_id, csi_rs_RE_cnt);
    }
#endif
}


// Unused - could delete
void PdschTx::h_compute_re_map(CsirsTables* h_csirs_tables, uint16_t* computed_re_map,  _cuphyCsirsRrcDynPrm* csi_rs_params, uint16_t csirs_params_offset, int num_params, uint16_t cell) {

#if 0 // Commenting out function body but leaving it in as it could be useful for debugging
    int      num_BWP_PRBs    = static_params->pCellStatPrms[cell].nPrbDlBwp;
    std::memset(computed_re_map, 0, num_BWP_PRBs*CUPHY_N_TONES_PER_PRB * OFDM_SYMBOLS_PER_SLOT * sizeof(uint16_t)); // Partial memset based on BWP

    // The last OFDM_SYMBOLS_PER_SLOT of the max. RE map allocation (max. alloc is per_cell_xtf_re_map_elements) should be reset to 0.
    // They will be set to 1 if there is at least one CSI-RS RE for that OFDM symbol.
    uint16_t* re_map_symbols = computed_re_map + per_cell_xtf_re_map_elements - OFDM_SYMBOLS_PER_SLOT;
    std::memset(re_map_symbols,  0, OFDM_SYMBOLS_PER_SLOT * sizeof(uint16_t));

    for(int j = 0; j < num_params; j++)
    {
        uint16_t csirs_idx = csirs_params_offset + j;
        //NVLOGC_FMT(NVLOG_PDSCH, "Cell {}: CSI-RS parameter {} of {}, offset {}, csirs_idx {}", cell, j, num_params, csirs_params_offset, csirs_idx);

        if(csi_rs_params[csirs_idx].row > CUPHY_CSIRS_SYMBOL_LOCATION_TABLE_LENGTH)
        {
            std::string err = "Row in CSI-RS parameter is more than " + std::to_string(CUPHY_CSIRS_SYMBOL_LOCATION_TABLE_LENGTH);
            throw std::runtime_error(err);
        }

        uint8_t genEvenRb = (csi_rs_params[csirs_idx].freqDensity == 0) ? 1 : 0;
        float rho_vals[4] = {0.5f, 0.5f, 1, 3};
        float rho = rho_vals[csi_rs_params[csirs_idx].freqDensity & 0x3];

        uint8_t row = csi_rs_params[csirs_idx].row; // 1-based indexing
        uint8_t num_ports = h_csirs_tables->rowData[row - 1].numPorts;
        float alpha = (num_ports == 1) ? rho : 2 * rho;

        uint8_t ki_ref_loc[CUPHY_CSIRS_MAX_KI_INDEX_LENGTH]; /*!< reference location of CSI-RS in frequency domain (k0,k1,k2,k3,k4,k5) */

        uint8_t li[2] = {csi_rs_params[csirs_idx].symbL0, csi_rs_params[csirs_idx].symbL1};

        // fill ki_ref_loc
        uint16_t freqDomain = csi_rs_params[csirs_idx].freqDomain;
        for(int i = 0; i < CUPHY_CSIRS_MAX_KI_INDEX_LENGTH && freqDomain > 0; ++i)
        {
            // rightmost(least significant) bit
            uint8_t ki = log2((freqDomain & (freqDomain - 1)) ^ freqDomain);
            switch (row)
            {
                case 1:
                case 2:
                    ki_ref_loc[i] = ki;
                    break;
                case 4:
                    ki_ref_loc[i] = 4 * ki;
                    break;
                default:
                    ki_ref_loc[i] = 2 * ki;
             }
             freqDomain = freqDomain & (freqDomain - 1);
        }

        // row is 1 index, the table is 0-indexed
        CsirsSymbLocRow& rowData = h_csirs_tables->rowData[row - 1];

        uint16_t nRB = csi_rs_params[csirs_idx].nRb;
        uint8_t lenKBarLBar = rowData.lenKBarLBar;
        uint8_t lenKPrime = rowData.lenKPrime;
        uint8_t lenLPrime = rowData.lenLPrime;

        uint16_t startRb = csi_rs_params[csirs_idx].startRb;

        for (int idxRb = startRb; idxRb <  startRb + csi_rs_params[csirs_idx].nRb; idxRb++) {
            bool isEvenRb = ((idxRb & 1) == 0);
            if (rho == 0.5f) {
                if ((genEvenRb && !isEvenRb) || (!genEvenRb && isEvenRb))
                    continue;
            }

             for (int idxKBarLBar = 0; idxKBarLBar < lenKBarLBar; idxKBarLBar++) {

                 uint8_t kBar = ki_ref_loc[rowData.kIndices[idxKBarLBar]] + rowData.kOffsets[idxKBarLBar];
                 uint8_t lBar = li[rowData.lIndices[idxKBarLBar]] + rowData.lOffsets[idxKBarLBar];

                 for (int idxLPrime = 0; idxLPrime < lenLPrime; idxLPrime++) {
                     uint8_t l = lBar + idxLPrime;
                     re_map_symbols[l] = 1; // Can update here as lenKPrime > 0
                     for (int idxKPrime = 0; idxKPrime < lenKPrime; idxKPrime++) {

                         uint16_t k = kBar + idxKPrime + idxRb * CUPHY_N_TONES_PER_PRB;
                         //NVLOGC_FMT(NVLOG_PDSCH, "k is {} and l is {}", k, l);
                         computed_re_map[k + l * num_BWP_PRBs * CUPHY_N_TONES_PER_PRB] = 1;
                         //NVLOGC_FMT(NVLOG_PDSCH, "Cell {}, index {}", cell, k + l * num_BWP_PRBs * CUPHY_N_TONES_PER_PRB);
                         //NVLOGC_FMT(NVLOG_PDSCH, "Cell {}, symbol l {} re {} set 1", cell, l, k);
                         /*if (idxRb == startRb) {
                             NVLOGC_FMT(NVLOG_PDSCH, "idxRb {}, isEvenRb {}, rho {:f}, freqDomain {}, nRB {}, startRb {}, idxKBarLBar {}, kBar {}, idxKPrime {}, k {}, lenKBarLBar {}, lenKPrime {}, lenLPrime {}",
                                    idxRb, isEvenRb, rho, csi_rs_params[csirs_idx].freqDomain, nRB, startRb, idxKBarLBar, kBar, idxKPrime, k, lenKBarLBar, lenKPrime, lenLPrime);

                         }
*/
                     }
                 }
             }
        }
    }
#endif
}


// Only used in pdsch_tx_multi_cell  TODO remove from here and ideally have that phase-2 example use datasets too similar to phase-3
void cumulative_read_cell_group_dynamic_pars_from_file(std::vector<cuphyPdschCellGrpDynPrm_t>& cell_grp_dyn_params, hdf5hpp::hdf5_file& input_file, bool first_call)
{
    hdf5hpp::hdf5_dataset cell_grp_dyn_pars_dataset = input_file.open_dataset("cellGrpDyn_pars");
    int                   num_cell_groups           = cell_grp_dyn_pars_dataset.get_dataspace().get_dimensions()[0];

    hdf5hpp::hdf5_dataset csirs_dyn_pars_dataset    = input_file.open_dataset("csirs_pars");
    int                   num_csirs                 = csirs_dyn_pars_dataset.get_dataspace().get_dimensions()[0];

    if(num_cell_groups != 1)
    {
        throw std::runtime_error("Only a single cell group is supported per pipeline!");
    }

    for(int cell_group_id = 0; cell_group_id < num_cell_groups; cell_group_id++)
    {
        cuphy::cuphyHDF5_struct cell_grp_dyn_config = cuphy::get_HDF5_struct_index(cell_grp_dyn_pars_dataset, cell_group_id);

        uint16_t num_cells                        = cell_grp_dyn_config.get_value_as<uint16_t>("nCells");
        uint16_t num_ue_groups                     = cell_grp_dyn_config.get_value_as<uint16_t>("nUeGrps");
        uint16_t num_ues                        = cell_grp_dyn_config.get_value_as<uint16_t>("nUes");
        uint16_t num_cws                        = cell_grp_dyn_config.get_value_as<uint16_t>("nCws");

        if (first_call) {
            cell_grp_dyn_params[cell_group_id].nCells = 0;
            cell_grp_dyn_params[cell_group_id].nUeGrps = 0;
            cell_grp_dyn_params[cell_group_id].nUes = 0;
            cell_grp_dyn_params[cell_group_id].nCws = 0;
            cell_grp_dyn_params[cell_group_id].nCsiRsPrms = 0;
            cell_grp_dyn_params[cell_group_id].nPrecodingMatrices = 0;

            //Allocate overprovisioned arrays for now for max cells, max UE groups, max UEs, max CWs.
            cell_grp_dyn_params[cell_group_id].pCellPrms  = new cuphyPdschCellDynPrm_t[PDSCH_MAX_CELLS_PER_CELL_GROUP];
            cell_grp_dyn_params[cell_group_id].pCellMetrics  = new cuphyPdschCellAerialMetrics_t[PDSCH_MAX_CELLS_PER_CELL_GROUP];
            //cell_grp_dyn_params[cell_group_id].pCellMetrics  = nullptr;
            cell_grp_dyn_params[cell_group_id].pUeGrpPrms = new cuphyPdschUeGrpPrm_t[PDSCH_MAX_UE_GROUPS_PER_CELL_GROUP];
            cell_grp_dyn_params[cell_group_id].pUePrms    = new cuphyPdschUePrm_t[PDSCH_MAX_UES_PER_CELL_GROUP];
            cell_grp_dyn_params[cell_group_id].pCwPrms    = new cuphyPdschCwPrm_t[PDSCH_MAX_CWS_PER_CELL_GROUP];
            cell_grp_dyn_params[cell_group_id].pCsiRsPrms = new _cuphyCsirsRrcDynPrm[CUPHY_CSIRS_MAX_NUM_PARAMS * PDSCH_MAX_CELLS_PER_CELL_GROUP];
            cell_grp_dyn_params[cell_group_id].pPmwPrms   = new cuphyPmW_t[PDSCH_MAX_UES_PER_CELL_GROUP];
        }

        cuphyPdschCellDynPrm_t* cell_dynamic_params = &cell_grp_dyn_params[cell_group_id].pCellPrms[cell_grp_dyn_params[cell_group_id].nCells];
        cuphyPdschUeGrpPrm_t* ue_group_dynamic_params = &cell_grp_dyn_params[cell_group_id].pUeGrpPrms[cell_grp_dyn_params[cell_group_id].nUeGrps];
        cuphyPdschUePrm_t* ue_dynamic_params = &cell_grp_dyn_params[cell_group_id].pUePrms[cell_grp_dyn_params[cell_group_id].nUes];
        cuphyPdschCwPrm_t* cw_dynamic_params = &cell_grp_dyn_params[cell_group_id].pCwPrms[cell_grp_dyn_params[cell_group_id].nCws];
        _cuphyCsirsRrcDynPrm* csirs_dynamic_params = &cell_grp_dyn_params[cell_group_id].pCsiRsPrms[cell_grp_dyn_params[cell_group_id].nCsiRsPrms];

        int cells_base = cell_grp_dyn_params[cell_group_id].nCells;
        int ue_groups_base = cell_grp_dyn_params[cell_group_id].nUeGrps;
        int ues_base = cell_grp_dyn_params[cell_group_id].nUes;
        int cws_base = cell_grp_dyn_params[cell_group_id].nCws;
        int csirs_base = cell_grp_dyn_params[cell_group_id].nCsiRsPrms;

        cell_grp_dyn_params[cell_group_id].nCells += num_cells;
        cell_grp_dyn_params[cell_group_id].nUeGrps += num_ue_groups;
        cell_grp_dyn_params[cell_group_id].nUes += num_ues;
        cell_grp_dyn_params[cell_group_id].nCws += num_cws;
        cell_grp_dyn_params[cell_group_id].nCsiRsPrms += num_csirs;

        // Check values less than max ones.
        if ((cell_grp_dyn_params[cell_group_id].nCells > PDSCH_MAX_CELLS_PER_CELL_GROUP) ||
            (cell_grp_dyn_params[cell_group_id].nUeGrps > PDSCH_MAX_UE_GROUPS_PER_CELL_GROUP) ||
            (cell_grp_dyn_params[cell_group_id].nUes > PDSCH_MAX_UES_PER_CELL_GROUP) ||
            (cell_grp_dyn_params[cell_group_id].nCws > PDSCH_MAX_CWS_PER_CELL_GROUP) ||
            (cell_grp_dyn_params[cell_group_id].nCsiRsPrms > PDSCH_MAX_CELLS_PER_CELL_GROUP*CUPHY_CSIRS_MAX_NUM_PARAMS))
        {
            // The check is cumulative per TV so the left hand side argument is an in-flux value and not the final one.
            // Also, allocations in cuPHY PDSCH will use the max cells configured in static parameters and so the max. values used there will be adjusted accordingly.
            NVLOGE_FMT(NVLOG_PDSCH, AERIAL_CUPHY_EVENT, "Error while stitching together dataset for multi-cell cell group. One or more of the following max limits has been exceeded.");
            NVLOGF_FMT(NVLOG_PDSCH, AERIAL_CUPHY_EVENT, "Current number of cells {} and max {}, number of UE groups {} and max {}, number of UEs {} and max {}, number of CWs {} and max {}, number of CSI-RS parameters {} and max {}",
                       cell_grp_dyn_params[cell_group_id].nCells, PDSCH_MAX_CELLS_PER_CELL_GROUP,
                       cell_grp_dyn_params[cell_group_id].nUeGrps, PDSCH_MAX_UE_GROUPS_PER_CELL_GROUP,
                       cell_grp_dyn_params[cell_group_id].nUes, PDSCH_MAX_UES_PER_CELL_GROUP,
                       cell_grp_dyn_params[cell_group_id].nCws, PDSCH_MAX_CWS_PER_CELL_GROUP,
                       cell_grp_dyn_params[cell_group_id].nCsiRsPrms, PDSCH_MAX_CELLS_PER_CELL_GROUP*CUPHY_CSIRS_MAX_NUM_PARAMS);

            //throw std::runtime_error("Invalid arg!");
        }

        // Populate arrays. Includes setting pointer to parent/children structs etc., so it should
        // happen *after* all previous mem. allocations.
        read_cell_dynamic_pars_from_file(cell_dynamic_params, input_file, cells_base, csirs_base);

        std::vector<cuphyPdschDmrsPrm_t> pdsch_dmrs_pars;
        read_dmrs_pars_from_file(pdsch_dmrs_pars, input_file, ue_groups_base);

        read_ue_groups_pars_from_file(ue_group_dynamic_params, input_file, cell_dynamic_params, pdsch_dmrs_pars, ue_groups_base, ues_base);
        read_ue_pars_from_file(ue_dynamic_params, input_file, ue_group_dynamic_params, cell_grp_dyn_params[cell_group_id], ues_base);
        read_cw_pars_from_file(cw_dynamic_params, input_file, ue_dynamic_params, cws_base);

        read_pdsch_csirs_pars_from_file(csirs_dynamic_params, num_csirs, csirs_dyn_pars_dataset);
    }
}


cuphyStatus_t CUPHYWINAPI cuphyCreatePdschTx(cuphyPdschTxHndl_t* pPdschTxHndl, cuphyPdschStatPrms_t const* pStatPrms)
{
    if((pPdschTxHndl == nullptr) || (pStatPrms == nullptr) || (pStatPrms->pDbg == nullptr))
    {
        NVLOGE_FMT(NVLOG_PDSCH, AERIAL_CUPHY_EVENT, "cuphyCreatePdschTx was called with cuphyPdschTxHndl_t ptr ({:p}) or cuphyPdschStatPrms_t ptr ({:p}) or pDbg ({:p}) nullptr.",
                   static_cast<void*>(pPdschTxHndl), const_cast<void*>(static_cast<void const*>(pStatPrms)),
                   (pStatPrms == nullptr) ? nullptr: static_cast<void*>(pStatPrms->pDbg));
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }

    return cuphy::tryCallableAndCatch([&]
    {
        PdschTx* new_pipeline = new PdschTx(pStatPrms, 0, pStatPrms->read_TB_CRC, !(pStatPrms->full_slot_processing) /*aas mode*/);

        if(new_pipeline == nullptr)
        {
            return CUPHY_STATUS_ALLOC_FAILED;
        }

        *pPdschTxHndl = new_pipeline;
        //cuphy::print_pdsch_static((cuphyPdschStatPrms_t*)new_pipeline->static_params);
        return CUPHY_STATUS_SUCCESS;
    });
}


size_t cuphyGetGpuMemoryFootprintPdschTx(cuphyPdschTxHndl_t pdschTxHndl)
{
    if(pdschTxHndl == nullptr)
    {
        return 0;
    }
    PdschTx* pipeline_ptr  = static_cast<PdschTx*>(pdschTxHndl);
    return pipeline_ptr->getGpuBufferSize();
}

#if 0
const void* cuphyGetMemoryFootprintTrackerPdschTx(cuphyPdschTxHndl_t pdschTxHndl)
{
    if(pdschTxHndl == nullptr)
    {
        return nullptr;
    }
    PdschTx* pipeline_ptr  = static_cast<PdschTx*>(pdschTxHndl);
    return pipeline_ptr->getMemoryTracker();
}
#endif


cuphyStatus_t CUPHYWINAPI cuphySetupPdschTx(cuphyPdschTxHndl_t pdschTxHndl, cuphyPdschDynPrms_t* pDynPrms, cuphyPdschBatchPrmHndl_t const batchPrmHndl)
{
    if((pDynPrms == nullptr) || (pdschTxHndl == nullptr))
    { // TODO Expand to include batchPrmHndl when batching enabled.
        NVLOGE_FMT(NVLOG_PDSCH, AERIAL_CUPHY_EVENT, "cuphySetupPdschTx called with nullptr for cuphyPdschTxHndl_t ({:p}) or for cuphyPdschDynPrms_t ptr ({:p}).",
                   static_cast<void*>(pdschTxHndl), static_cast<void*>(pDynPrms));
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }

    cuphyStatus_t status = cuphy::tryCallableAndCatch([&]
    {
        //auto start_time = Timer::now();
        PUSH_RANGE("cuphySetupPdschTx", 1);
        PdschTx* pipeline_ptr  = static_cast<PdschTx*>(pdschTxHndl);
        //Will ignore batchPrmHndl for now.
        pipeline_ptr->dynamic_params = pDynPrms; // Needed to retrieve CUDA stream etc.
        pipeline_ptr->setProcMode(pDynPrms->procModeBmsk & PDSCH_PROC_MODE_GRAPHS);

        //Reset status for next call
        pDynPrms->pStatusInfo->status = cuphyPdschStatusType_t::CUPHY_PDSCH_STATUS_SUCCESS_OR_UNTRACKED_ISSUE;
        pDynPrms->pStatusInfo->ueIdx           = MAX_UINT16;
        pDynPrms->pStatusInfo->cellPrmStatIdx  = MAX_UINT16;

        //NVLOGC_FMT(NVLOG_PDSCH, "{} pBufferType {}",__func__, pDynPrms->pDataIn->pBufferType);
        cuphyStatus_t status = pipeline_ptr->expandParametersMultipleCells(pDynPrms, pDynPrms->cuStream);
        POP_RANGE
        return status;

        //auto end_time = Timer::now();
        //NVLOGC_FMT(NVLOG_PDSCH, "updateSetup timing: {:f} us", Timer::elapsedTime(start_time, end_time));
    }, CUPHY_STATUS_INVALID_ARGUMENT);
    if ((status != CUPHY_STATUS_SUCCESS) && (pDynPrms->pStatusInfo->status == cuphyPdschStatusType_t::CUPHY_PDSCH_STATUS_UNSUPPORTED_MAX_ER_PER_CB)) {
        int ue_idx = pDynPrms->pStatusInfo->ueIdx; // across all UEs in that group
        int static_cell_idx = pDynPrms->pCellGrpDynPrm->pUePrms[ue_idx].pUeGrpPrm->pCellPrm->cellPrmStatIdx;
        pDynPrms->pStatusInfo->cellPrmStatIdx =  static_cell_idx;
        //NVLOGC_FMT(NVLOG_PDSCH, "ue_idx {}, static_cell_idx {}", ue_idx, static_cell_idx);
    }
    return status;
}

cuphyStatus_t CUPHYWINAPI cuphyFallbackBufferSetupPdschTx(cuphyPdschTxHndl_t pdschTxHndl, void* pAddr, cudaStream_t cuda_strm)
{
    if((pAddr == nullptr) || (pdschTxHndl == nullptr))
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    return cuphy::tryCallableAndCatch([&]
    {
        //Only works for single cell.
        PdschTx* pipeline_ptr  = static_cast<PdschTx*>(pdschTxHndl);
        pipeline_ptr->dynamic_params->pDataOut->pTDataTx[0].pAddr  = pAddr;
        pipeline_ptr->updateOutputTensorBuffer(cuda_strm);
    });
}

cuphyStatus_t CUPHYWINAPI cuphyFallbackBuffersSetupPdschTx(cuphyPdschTxHndl_t pdschTxHndl, void** pAddr, int fallback_cells, cudaStream_t cuda_strm)
{
    if((pAddr == nullptr) || (pdschTxHndl == nullptr) || (fallback_cells <= 0))
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }

    return cuphy::tryCallableAndCatch([&]
    {
        PdschTx* pipeline_ptr  = static_cast<PdschTx*>(pdschTxHndl);
        for (int i = 0; i < fallback_cells; i++) {
            if (pAddr[i] == nullptr)
            {
                return CUPHY_STATUS_INVALID_ARGUMENT;
            }
            // Assumes that the order of cell addresses in pAddr matches the order of cells in pTDataTx during its initial non-fallback setup.
            pipeline_ptr->dynamic_params->pDataOut->pTDataTx[i].pAddr  = pAddr[i];
            //NVLOGC_FMT(NVLOG_PDSCH, "cuphyFallbackBuffersSetupPdschTx cell {} with addr {:p}", i, (void*)pAddr[i]);
        }
        if (fallback_cells == 1) {
            pipeline_ptr->updateOutputTensorBuffer(cuda_strm);
        }
        else
        {
            pipeline_ptr->updateOutputTensorBuffers(fallback_cells, cuda_strm);
        }
        return CUPHY_STATUS_SUCCESS;
    });
}


void updateRefCheck(cuphyPdschTxHndl_t pdschTxHndl, bool ref_check)
{
    PdschTx* pipeline_ptr  = static_cast<PdschTx*>(pdschTxHndl);
    pipeline_ptr->static_params->pDbg->refCheck = ref_check;
}

void updateRefCheckMultipleCells(cuphyPdschTxHndl_t pdschTxHndl, bool ref_check)
{
    PdschTx* pipeline_ptr  = static_cast<PdschTx*>(pdschTxHndl);
    for (int cell = 0; cell < PDSCH_MAX_CELLS_PER_CELL_GROUP; cell++) {
        pipeline_ptr->static_params->pDbg[cell].refCheck = ref_check;
    }
}

void updateFileName(cuphyPdschTxHndl_t pdschTxHndl, const char* file_name)
{
    PdschTx* pipeline_ptr  = static_cast<PdschTx*>(pdschTxHndl);
    pipeline_ptr->static_params->pDbg->pCfgFileName = file_name;
}

void updateFileNameMultipleCells(cuphyPdschTxHndl_t pdschTxHndl, uint32_t cell_id, const char* file_name)
{
    PdschTx* pipeline_ptr  = static_cast<PdschTx*>(pdschTxHndl);
    pipeline_ptr->static_params->pDbg[cell_id].pCfgFileName = file_name;
}

/* A cuphyPdschTxHndl_t corresponds to a single pipeline */
cuphyStatus_t CUPHYWINAPI cuphyRunPdschTx(cuphyPdschTxHndl_t pdschTxHndl, uint64_t procModeBmsk)
{
    if(pdschTxHndl == nullptr)
    { //TODO Expand to check for valid processing mode as needed.
        NVLOGE_FMT(NVLOG_PDSCH, AERIAL_CUPHY_EVENT, "cuphyRunPdschTx error: cuphyPdschTxHndl_t is nullptr!");
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }

    return cuphy::tryCallableAndCatch([&]
    {
        PdschTx* pipeline_ptr  = static_cast<PdschTx*>(pdschTxHndl);

        // Treat bitmask as the PDSCH_PROC_MODE for now.
        //Check we don't accidentally switch graph modes between Setup and Run.
        // The PDSCH_INTER_CELL_BATCHING field of the procModeBmsk is ignored; always treated as 1.
        bool new_graph_mode = ((procModeBmsk & PDSCH_PROC_MODE_GRAPHS) == 1);
        if (pipeline_ptr->getGraphMode() != new_graph_mode)
        {
            NVLOGE_FMT(NVLOG_PDSCH, AERIAL_CUPHY_EVENT, "PDSCH: Cannot switch graph mode between setup and run!");
            return CUPHY_STATUS_INVALID_ARGUMENT;
        }
        PUSH_RANGE("cuphyRunPdschTx", 1);

        pipeline_ptr->setProcMode(new_graph_mode);

        cuphyStatus_t status = pipeline_ptr->RunMultipleCells(pipeline_ptr->dynamic_params->cuStream, pipeline_ptr->static_params->pDbg->refCheck);
        POP_RANGE
        return status; // could be mismatch found, but we use CUPHY_STATUS_INTERNAL_ERROR to cover general case instead of CUPHY_STATUS_REF_MISMATCH
    });
}

cuphyStatus_t CUPHYWINAPI cuphyDestroyPdschTx(cuphyPdschTxHndl_t pdschTxHndl)
{
    if(pdschTxHndl == nullptr)
    {
        NVLOGE_FMT(NVLOG_PDSCH, AERIAL_CUPHY_EVENT, "cuphyDestroyPdschTx error: cuphyPdschTxHndl_t is nullptr!");
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    PdschTx* pipeline_ptr  = static_cast<PdschTx*>(pdschTxHndl);
    delete pipeline_ptr;
    return CUPHY_STATUS_SUCCESS;
}


PdschTx::PdschTx(const cuphyPdschStatPrms_t* cfg_static_params, cudaStream_t cfg_strm, bool read_TB_CRC, bool cfg_aas_mode) :
    static_params(cfg_static_params),
    hdf5_filename(""),
    num_TBs(1),
    num_cells(1),
    strm(cfg_strm),
    scrambling(true),
    layer_mapping(true),
    read_TB_CRC(read_TB_CRC),
    inter_cell_batching_mode(true),
    max_CBs_per_TB((cfg_static_params->nMaxCBsPerTB == 0) ? MAX_N_CBS_PER_TB_SUPPORTED : cfg_static_params->nMaxCBsPerTB),
    max_cells((cfg_static_params->nMaxCellsPerSlot == 0) ? PDSCH_MAX_CELLS_PER_CELL_GROUP : cfg_static_params->nMaxCellsPerSlot),
    per_cell_group_max_TBs((cfg_static_params->nMaxUesPerCellGroup == 0) ? PDSCH_MAX_UES_PER_CELL_GROUP : cfg_static_params->nMaxUesPerCellGroup),
    max_PRB_BWP((cfg_static_params->nMaxPrb == 0) ? MAX_N_PRBS_SUPPORTED : cfg_static_params->nMaxPrb),
    pdsch_comp_tol(__float2half(0.00098f)), //TODO FIXME check Matlab and cuPHY impl. Changed temporarily from 0.0001 for 3259  TV
    m_component_descrs("PdschDescr"),
    bulk_desc_async_copy(true),
#if AVOID_STREAM_CREATION_IN_SETUP
    ldpc_stream_pool(per_cell_group_max_TBs - 1, cfg_static_params->stream_priority),
#else
    ldpc_stream_pool(0, cfg_static_params->stream_priority),
#endif
    m_batchedMemcpyHelper(max_cells, batchedMemcpySrcHint::srcIsHost, batchedMemcpyDstHint::dstIsDevice, (PDSCH_USE_BATCHED_MEMCPY == 1) && (cfg_static_params->enableBatchedMemcpy == 1))
{
    cfg_static_params->pOutInfo->pMemoryFootprint = &memory_footprint; // update  static parameter field that points to the cuphyMemoryFootprintTracker object for this channel

    /* Note on stream priority:  There is no runtime check that cfg_static_params->stream_priority is within the [greatest_priority, least_priority] of the GPU device
       as returned by cudaDeviceGetStreamPriorityRange(&least_priority, &greatest_priority) call. Any priority outside that range will be clamped to their respective
       min or max values. Can add an extra error check, if needed. */

    /*NVLOGD_FMT(NVLOG_PDSCH, "max_cells {} nMaxCellsPerSlot {} and PDSCH_MAX_CELLS_PER_CELL_GROUP {}", max_cells, static_params->nMaxCellsPerSlot, PDSCH_MAX_CELLS_PER_CELL_GROUP);
    NVLOGD_FMT(NVLOG_PDSCH, "max_UEs {} nMaxUesPerCellGroup {} and PDSCH_MAX_UES_PER_CELL_GROUP {}", per_cell_group_max_TBs, static_params->nMaxUesPerCellGroup, PDSCH_MAX_UES_PER_CELL_GROUP);
    NVLOGD_FMT(NVLOG_PDSCH, "max CBs per TB {} nMaxCBsPerTB {} and MAX_N_CBS_PER_TB_SUPPORTED {}", max_CBs_per_TB, static_params->nMaxCBsPerTB, MAX_N_CBS_PER_TB_SUPPORTED);
    NVLOGD_FMT(NVLOG_PDSCH, "max PRB_BWP {}",max_PRB_BWP);*/

    if (max_cells > PDSCH_MAX_CELLS_PER_CELL_GROUP)
    {
        NVLOGE_FMT(NVLOG_PDSCH, AERIAL_CUPHY_EVENT, "PDSCH: max cells is {} but PDSCH_MAX_CELLS_PER_CELL_GROUP is {}.", max_cells, PDSCH_MAX_CELLS_PER_CELL_GROUP);
    }
    if (per_cell_group_max_TBs > PDSCH_MAX_UES_PER_CELL_GROUP)
    {
        NVLOGE_FMT(NVLOG_PDSCH, AERIAL_CUPHY_EVENT, "PDSCH: max TBs per cell group is {} but PDSCH_MAX_UES_PER_CELL_GROUP is {}", per_cell_group_max_TBs, PDSCH_MAX_UES_PER_CELL_GROUP);
    }
    if (per_cell_group_max_TBs > 256) {
        NVLOGE_FMT(NVLOG_PDSCH, AERIAL_CUPHY_EVENT, "Currently up to 256 TBs in cell group supported.");
        //Should change TB_idxs FROM LDPC_batch_info data type to uint16_t from uint8_t.
    }
    if (max_CBs_per_TB > MAX_N_CBS_PER_TB_SUPPORTED)
    {
        NVLOGE_FMT(NVLOG_PDSCH, AERIAL_CUPHY_EVENT, "PDSCH: max CBs per TB {} but MAX_N_CBS_PER_TB_SUPPORTED is {}", max_CBs_per_TB, MAX_N_CBS_PER_TB_SUPPORTED);
    }
    if (max_PRB_BWP > MAX_N_PRBS_SUPPORTED)
    {
        NVLOGE_FMT(NVLOG_PDSCH, AERIAL_CUPHY_EVENT, "PDSCH: max PRB BWP (bandwidth part) {} but max supported is {}", max_PRB_BWP,  MAX_N_PRBS_SUPPORTED);
    }

    // Number of elements in per cell RE map used when CSI-RS  parameters are present.
    // The last OFDM_SYMBOLS_PER_SLOT elements of each map's max allocation indicate if there are any punctured REs for that OFDM symbol (value of 1)
    // or no (value of 0). These filters help reduce the overhead of cuphyUpdatePdschDmrsParams on the host when only a small number of symbols has CSI-RS allocations.
    // The RE-map computation and post-processing should be revisited as we scale to more cells.
    per_cell_xtf_re_map_elements = max_PRB_BWP * CUPHY_N_TONES_PER_PRB * OFDM_SYMBOLS_PER_SLOT + OFDM_SYMBOLS_PER_SLOT;


    // Resize all vectors that were previously dimensioned on PDSCH_MAX_CELLS_PER_CELL_GROUP
    per_cell_hdf5_filename.resize(max_cells);
    per_cell_pipeline_input_bytes.resize(max_cells);
    h_per_cell_pipeline_input_tb_crcs.resize(max_cells);
    per_cell_pipeline_input_size_bytes.resize(max_cells);
    per_cell_pipeline_input_padding_size_bytes.resize(max_cells);
    per_cell_num_TBs.resize(max_cells);
    per_cell_TB_params_offset.resize(max_cells);
    per_cell_CRC_output_bytes.resize(max_cells);
    per_cell_max_CBs.resize(max_cells);
    per_cell_CRC_max_TB_padded_bytes.resize(max_cells);
    per_cell_max_tb_size_bytes.resize(max_cells);
    per_cell_crc_h_in_tensor_bytes.resize(max_cells);
    per_cell_crc_h_in_tensor_bytes_offset.resize(max_cells);
    per_cell_LDPC_K_max.resize(max_cells);
    per_cell_input_file.resize(max_cells);
    d_per_cell_ldpc_out_tensor_ref.resize(max_cells);
    m_crcEncodeNodesMultipleCells.resize(2); // 2 CRC kernels per cell-group
    m_rmNodesMultipleCells.resize(2);
    m_dmrsNodeMultipleCells.resize(1);
    m_csirsPrepNodesMultipleCells.resize(3); // 2 CSI-RS prep. kernel when CSI-RS parameters are present; FIXME could be 3 if I do memset in run too; added it last
    m_prepareCRCEncodeNodeMultipleCells.resize(1);

    //Resize all vectors that were previously dimensioned on PDSCH_MAX_UES_PER_CELL_GROUP
    cell_idx_for_TB.resize(per_cell_group_max_TBs);
    m_ldpcEncodeNodesMultipleCells.resize(per_cell_group_max_TBs);

    // Temp. call added as high overhead was observed in first cudaGetFuncBySymbol call in CRC setup in cuPHY-CP under MPS.
    // TODO Delete once we root cause this.
    CUDA_KERNEL_NODE_PARAMS temp;
    CUPHY_CHECK(cuphySetEmptyKernelNodeParams(&temp));

    aas_mode               = cfg_aas_mode;

    if(aas_mode) layer_mapping = false;
    allocateBuffers();

    // Allocate launch config structs for all pipeline components.
    // Set kernel functions of all components but LDPC to nullptr to ensure call to cudaFuncGetSymbol
    // happens only once.

    m_ldpc_encode_launch_cfg   = std::make_unique<cuphyLDPCEncodeLaunchConfig[]>(PDSCH_MAX_HET_LDPC_CONFIGS_SUPPORTED);

    m_crc_encode_launch_cfg = std::make_unique<cuphyCrcEncodeLaunchConfig>();
    m_crc_encode_launch_cfg.get()->m_kernelNodeParams[0].func    = nullptr;
    m_crc_encode_launch_cfg.get()->m_kernelNodeParams[1].func    = nullptr;

    m_dmrs_launch_cfg       = std::make_unique<cuphyPdschDmrsLaunchConfig>();
    m_dmrs_launch_cfg.get()->m_kernelNodeParams.func             = nullptr;

    m_rate_matching_launch_cfg = std::make_unique<cuphyDlRateMatchingLaunchConfig>();
    m_rate_matching_launch_cfg.get()->m_kernelNodeParams[0].func = nullptr;
    m_rate_matching_launch_cfg.get()->m_kernelNodeParams[1].func = nullptr;

    m_csirs_prep_launch_cfg = std::make_unique<cuphyPdschCsirsPrepLaunchConfig>();
    m_csirs_prep_launch_cfg.get()->m_kernelNodeParams[0].func = nullptr;
    m_csirs_prep_launch_cfg.get()->m_kernelNodeParams[1].func = nullptr;
    m_csirs_prep_launch_cfg.get()->m_kernelNodeParams[2].func = nullptr;

    m_prepare_crc_encode_launch_cfg = std::make_unique<cuphyPrepareCrcEncodeLaunchConfig>();
    m_prepare_crc_encode_launch_cfg.get()->m_kernelNodeParams.func = nullptr;

    for (int cell = 0; cell < max_cells; cell++) {
       per_cell_input_file[cell] = nullptr;
       per_cell_hdf5_filename[cell] = "";
    }

    allocateDescr(); //Allocate Descriptors.

    createAndInstantiateGraph(); // performed unconditionally with empty kernel nodes to avoid high initial overhead
    graph_mode = false;

    if (PRINT_GPU_MEMORY_CUPHY_CHANNEL == 1) 
    {
      memory_footprint.printMemoryFootprint(this, "PDSCH");
    }
}


void PdschTx::print_csv_friendly_config()
{
    //NB: Should be called after setup for all the values to be meaningful.
    // Prepend header w/ CSV_HEADER and every line with CSV_BODY so it's easier for scripts to grep if enabled

    std::cout << "CSV_HEADERTV_name,TV_regex,TB_id,#TBs,#total used layers,TB_size,num_CBs,K,N,G,rv,Qm,bg,Nl,Zc,F,";
    std::cout << "layer_map,Er_min,Er_max(if applicable),CB_split (if applicable),Xtf dims(TVs),start PRB,num_PRBs,start Sym,nSym,targetCodeRate,qamModOrder,DMRS_max_length,";
    std::cout << "# cells, dyn.cell Id, total UE groups,total UEs,total CWs,# UEs (for this group),#CWs (for this UE)\n";

    using tensor_pinned_C_64F = typed_tensor<CUPHY_C_64F, pinned_alloc>;
    tensor_pinned_C_64F pdsch_tx_ref_output = typed_tensor_from_dataset<CUPHY_C_64F, pinned_alloc>((*per_cell_input_file[0]).open_dataset("Xtf"));
    uint32_t total_layers    = pdsch_tx_ref_output.layout().dimensions()[2]; // data_tx_tensor is overprovisioned in terms of # layers
    int      num_REs   = pdsch_tx_ref_output.layout().dimensions()[0];

    for (int TB_id = 0; TB_id < num_TBs; TB_id++) {
        std::cout << "CSV_BODY";
        std::cout << hdf5_filename << ",," << TB_id << "," << num_TBs << ","<< "not computed" << ",";
        std::cout << TB_sizes[TB_id] << "," << kernel_params[TB_id].num_CBs << ",";
        std::cout << kernel_params[TB_id].K << "," <<  kernel_params[TB_id].N << "," <<  kernel_params[TB_id].G << ",";
        std::cout << kernel_params[TB_id].rv << "," << kernel_params[TB_id].Qm << "," << kernel_params[TB_id].bg << ",";
        std::cout << kernel_params[TB_id].Nl << "," << kernel_params[TB_id].Zc << "," << kernel_params[TB_id].F << ",";

        // Backpointers to parent UE, UE group, cell.
        cuphyPdschUePrm_t* p_parent_UE = dynamic_params->pCellGrpDynPrm->pCwPrms[TB_id].pUePrm;
        cuphyPdschUeGrpPrm_t* p_parent_UE_grp = p_parent_UE->pUeGrpPrm;
        cuphyPdschCellDynPrm_t* p_parent_cell = p_parent_UE_grp->pCellPrm;

        std::cout << "{";
        int TB_layers = kernel_params[TB_id].Nl;
        for(int layer_cnt = 0; layer_cnt < TB_layers; layer_cnt++) {
            std::cout << calculate_layer_id(h_dmrs_params[TB_id].port_ids[layer_cnt], h_dmrs_params[TB_id].n_scid, h_dmrs_params[TB_id].nlAbove16);
            if(layer_cnt != TB_layers - 1)
            {
                std::cout << "-";
            }
            else
            {
                std::cout << "},";
            }
        }

        //Er min, Er_max, CB split - Reminder Er values do not take into consideration punctured REs due to CSI-RS parameters.
        uint32_t* Er = h_rm_workspace + 2 * num_TBs;
        uint32_t CB_split = Er[TB_id * 2]; // if applicable
        uint32_t Er_min = Er[TB_id * 2 + 1];
        uint32_t Er_max = Er_min + kernel_params[TB_id].Nl * kernel_params[TB_id].Qm;
        if (CB_split == kernel_params[TB_id].num_CBs) { // no split
            std::cout << Er_min << ",,,";
        } else if (CB_split == 0) {
            std::cout << "," << Er_max << "," << CB_split << ",";
        } else {
            std::cout << Er_min << "," << Er_max << "," << CB_split << ",";
        }

        // TV's Xtf dataset dims in {#Res, # OFDM symbols, # layers} order
        std::cout << "{" << num_REs << "x" << OFDM_SYMBOLS_PER_SLOT << "x" << total_layers << "},";

        //start PRB, num_PRBs, start_sym, nSym
        std::cout << h_dmrs_params[TB_id].start_Rb << "," << h_dmrs_params[TB_id].num_Rbs << ",";
        std::cout << h_dmrs_params[TB_id].symbol_number << ",";
        int num_sym = p_parent_cell->nPdschSym; // number of symbols is a cell property
        std::cout << num_sym << ",";

        // targetCodeRate, qamModOrder
        uint16_t targetCodeRate = dynamic_params->pCellGrpDynPrm->pCwPrms[TB_id].targetCodeRate;
        uint8_t qamModOrder = dynamic_params->pCellGrpDynPrm->pCwPrms[TB_id].qamModOrder;
        std::cout << targetCodeRate << "," << qamModOrder << ",";

        //DMRS max length
        std::cout << h_dmrs_params[TB_id].num_dmrs_symbols << ","; // For now assumes dmrsAddlnPos is 0

        int total_cells = dynamic_params->pCellGrpDynPrm->nCells;
        int total_UE_grps = dynamic_params->pCellGrpDynPrm->nUeGrps;
        int total_UEs = dynamic_params->pCellGrpDynPrm->nUes;
        int total_Cws = dynamic_params->pCellGrpDynPrm->nCws;

        //int static_cell_id = p_parent_cell->cellPrmStatIdx;
        int dynamic_cell_id = p_parent_cell->cellPrmDynIdx;
        //int UE_grp_id = p_parent_UE->ueGrpIdx; //TODO missing from PDSCH API
        int n_UEs = p_parent_UE_grp->nUes;
        int n_Cws = p_parent_UE->nCw;

        //TODO Ideally, it'd be possible to specify the UE group, UE, CW ids for each TB.
        std::cout << total_cells << "," << dynamic_cell_id << ",";
        std::cout << total_UE_grps << "," << total_UEs << "," << total_Cws << ",";
        std::cout << n_UEs << "," << n_Cws;
        std::cout << std::endl;
    }
}


bool PdschTx::getGraphMode()
{
    return graph_mode;
}


void PdschTx::setProcMode(bool cfg_graph_mode, bool cfg_inter_cell_batching_mode)
{
    graph_mode = cfg_graph_mode;
}

void PdschTx::setReverseBits(bool value)
{
    rev_bit_order = value;
}

void PdschTx::setMaxVals()
{ //TODO Update the default values as needed.
    per_cell_max_TBs = PDSCH_MAX_UES_PER_CELL;

    max_K_per_CB = CUPHY_LDPC_BG1_INFO_NODES * CUPHY_LDPC_MAX_LIFTING_SIZE;
    max_N_per_CB = MAX_ENCODED_CODE_BLOCK_BIT_SIZE;
    max_Emax     = PDSCH_MAX_ER_PER_CB_BITS;
    max_layers   = MAX_N_BBU_LAYERS_SUPPORTED;
}

void PdschTx::setMaxVals(int cfg_cell_max_TBs, int cfg_max_K_per_CB, int cfg_max_N_per_CB, int cfg_max_Emax, int cfg_max_layers)
{
    per_cell_max_TBs = cfg_cell_max_TBs;
    max_K_per_CB   = cfg_max_K_per_CB;
    max_N_per_CB   = cfg_max_N_per_CB;
    max_Emax       = cfg_max_Emax;
    max_layers     = cfg_max_layers;
}

size_t PdschTx::getGpuBufferSize()
{
    return memory_footprint.getGpuRegularSize();
}

const void* PdschTx::getMemoryTracker()
{
    return &memory_footprint;
}


void PdschTx::allocateBuffers()
{
    // Reminder: overprovisioned buffer allocation happens only once, in the constructor.

#if 0
    setMaxVals(); // TODO Use setMaxVals with specific max values, if default not appropriate.
#else
    // TODO Adjust as needed
    setMaxVals(PDSCH_MAX_UES_PER_CELL /* max # TBs, currently 16 */,
               CUPHY_LDPC_BG1_INFO_NODES * CUPHY_LDPC_MAX_LIFTING_SIZE /* max_K_per_CB */,
               MAX_ENCODED_CODE_BLOCK_BIT_SIZE /* max N per CB */,
               PDSCH_MAX_ER_PER_CB_BITS /* max Emax */,
               MAX_N_BBU_LAYERS_SUPPORTED /* max layers */);
#endif

    // CSI-RS specific params
    // Allocate workspace buffers and update relevant host/device pointers for CSI-RS (temp)
    workspace_offsets[0] = 0; // unchanged
    updateCsirsWorkspaceOffsets(max_cells, CUPHY_CSIRS_MAX_NUM_PARAMS * max_cells);
    h_workspace = make_unique_pinned<uint8_t>(workspace_offsets[N_CSIRS_COMPONENTS]);
    d_workspace = make_unique_device<uint8_t>(workspace_offsets[N_CSIRS_COMPONENTS], &memory_footprint);
    h_csirs_cell_index = (uint32_t*)(h_workspace.get());
    d_csirs_cell_index = (uint32_t*)(d_workspace.get());
    updateCsirsWorkspacePtrs();

    h_xtf_re_maps_v2 = make_unique_pinned<uint16_t>(max_cells * per_cell_xtf_re_map_elements);
    d_xtf_re_maps_v2 = make_unique_device<uint16_t>(max_cells * per_cell_xtf_re_map_elements, &memory_footprint);

    ldpc_streams.resize(per_cell_group_max_TBs, 0);
    ldpc_complete_events.resize(per_cell_group_max_TBs);
    TB_sizes.resize(per_cell_group_max_TBs); // Set in expandParametersMultipleCells()
    max_ldpc_parity_nodes.resize(per_cell_group_max_TBs); // Set in prepareRateMatching*()
    TB_padded_byte_sizes.resize(per_cell_group_max_TBs);  // Set in prepareCRC*()
    single_TB_d_ldpc_out_tensor.resize(per_cell_group_max_TBs);
    single_TB_d_ldpc_in_tensor_desc.resize(per_cell_group_max_TBs);
    single_TB_d_ldpc_out_tensor_desc.resize(per_cell_group_max_TBs);

    CRC_dst_offset.resize(per_cell_group_max_TBs);
    LDPC_dst_offset.resize(per_cell_group_max_TBs);
    LDPC_batches.resize(per_cell_group_max_TBs);
    LDPC_input_addr.resize(per_cell_group_max_TBs);
    LDPC_output_addr.resize(per_cell_group_max_TBs);

    // Resize vectors used only in prepareCRC or prepareCRCMultipleCells.
    // In the single cell per cell group case, only N_MAX_TBS_SUPPORTED will be used.
    total_CB_byte_sizes.resize(per_cell_group_max_TBs);
    CB_data_byte_sizes.resize(per_cell_group_max_TBs);
    tb_size.resize(per_cell_group_max_TBs);
    padding_bytes.resize(per_cell_group_max_TBs);

    for(int i = 0; i < per_cell_group_max_TBs; i++)
    {
        CUDA_CHECK(cudaEventCreateWithFlags(&ldpc_complete_events[i], cudaEventDisableTiming));
    }
    ldpc_stream_elements = 0; // Also reset on every setup
    created_ldpc_streams = 0; // Only reset once

    d_TB_CRCs = make_unique_device<uint32_t>(per_cell_group_max_TBs, &memory_footprint);
// For CRC
    max_per_cell_CB_CRCs_elements = per_cell_max_TBs * max_CBs_per_TB;
#if DBG_PDSCH_CRC
    d_CB_CRCs = make_unique_device<uint32_t>(max_cells * max_per_cell_CB_CRCs_elements, &memory_footprint);
#else
    //Avoid writing the per-TB and per-CB CRCs separately.
    d_CB_CRCs = nullptr;
#endif
    max_per_cell_code_blocks_bytes = per_cell_max_TBs * max_CBs_per_TB * div_round_up<uint32_t>(max_K_per_CB, 8);
    d_code_blocks                 = make_unique_device<uint8_t>(max_cells * max_per_cell_code_blocks_bytes, &memory_footprint);
    if(static_params->pDbg->refCheck) {
        // Memset needed when ref. checks are enabled to avoid compute-sanitizer's initcheck error flagged for
        // the D2H memory copy in getHostOutputCRCMultipleCells() call if there is at least one cell in testing mode (testModel=1).
        // That function proactively copies the entire d_code_blocks buffer, but ref. checks in refCheckCRCMultipleCells() are skipped for any cells in testing mode.
        CUDA_CHECK(cudaMemset(d_code_blocks.get(), 0xff, max_cells * max_per_cell_code_blocks_bytes * sizeof(uint8_t)));
    }

    max_per_cell_crc_workspace_elements = 2 * div_round_up<uint32_t>(max_per_cell_code_blocks_bytes, sizeof(uint32_t)); //double as temp fix.
    // PdschPerTbParams allocated in allocateDescr()

    d_crc_workspace = make_unique_device<uint32_t>(max_cells * max_per_cell_crc_workspace_elements, &memory_footprint);

    if (REVERT_TO_ASYNC_COPIES == 1) { // Allocate buffer used to copy PDSCH TB input
        d_prepare_crc_input_buffer = make_unique_device<uint8_t>(max_cells * max_per_cell_code_blocks_bytes, &memory_footprint);
    }


    // For LDPC
    max_per_TB_LDPC_workspace_size = div_round_up<uint32_t>(max_N_per_CB, 8) * max_CBs_per_TB; //FIXME div_round_up to uint32_t?
    size_t max_LDPC_workspace_size = max_per_TB_LDPC_workspace_size * per_cell_group_max_TBs;

    d_ldpc_workspace = make_unique_device<uint32_t>(div_round_up<uint32_t>(max_LDPC_workspace_size, sizeof(uint32_t)), &memory_footprint);
    if(static_params->pDbg->refCheck) {
        // Memset needed when ref. checks are enabled to avoid compute-sanitizer's initcheck
        CUDA_CHECK(cudaMemset(d_ldpc_workspace.get(), 0xff, round_up_to_next<uint32_t>(max_LDPC_workspace_size, sizeof(uint32_t))));
    }

    // For Rate Matching
    // Workspace buffer allocated in allocateDescr()

    if (aas_mode) {
        // Note: this buffer is only used in AAS mode. Allocated conditionally now that AAS mode is a static parameter passed to the PdschTx constructor
        // Buffer increased a lot for FDM
        uint32_t max_rm_output_elements = div_round_up<uint32_t>(per_cell_max_TBs * max_layers * max_CBs_per_TB * max_Emax, 32);
        d_rate_matching_output          = tensor_device(CUPHY_R_32U, max_rm_output_elements, cuphy::tensor_flags::align_tight);
    }

    // RE maps allocated in allocateDescr()

    // LDPC workspace buffers d_ldpc_w_ptr and h_ldpc_w_ptr are allocated in allocateDescr

    // For Modulation: nothing needed

    // For DMRS
    // PDSCH dmrs params allocated in allocateDescr()
}

void PdschTx::allocateDescr()
{
    std::array<size_t, N_PDSCH_COMPONENTS> dynDescrSizeBytes{};
    std::array<size_t, N_PDSCH_COMPONENTS> dynDescrAlignBytes{};
    size_t* pDynDescrSizeBytes  = dynDescrSizeBytes.data();
    size_t* pDynDescrAlignBytes = dynDescrAlignBytes.data();
    ldpc_workspace_bytes = 0; // it doesn't matter it's overwritten in every for loop iteration below as all values are identical

    cuphyStatus_t status;

    status = cuphyLDPCEncodeGetDescrInfo(&pDynDescrSizeBytes[PDSCH_LDPC], &pDynDescrAlignBytes[PDSCH_LDPC], per_cell_group_max_TBs, &ldpc_workspace_bytes);
    if(status != CUPHY_STATUS_SUCCESS)
    {
        NVLOGE_FMT(NVLOG_PDSCH, AERIAL_CUPHY_EVENT, "Cell group cuphyLDPCEncodeGetDescrInfo error {:d}", status);
        throw cuphy::cuphy_exception(status);
    }

    status = cuphyCrcEncodeGetDescrInfo(&pDynDescrSizeBytes[PDSCH_CRC], &pDynDescrAlignBytes[PDSCH_CRC]);

    if(status != CUPHY_STATUS_SUCCESS)
    {
        NVLOGE_FMT(NVLOG_PDSCH, AERIAL_CUPHY_EVENT, "Cell group cuphyCrcEncodeGetDescrInfo error {}", status);
        throw cuphy::cuphy_exception(status);
    }

    status = cuphyDlRateMatchingGetDescrInfo(&pDynDescrSizeBytes[PDSCH_RM], &pDynDescrAlignBytes[PDSCH_RM]);
    if(status != CUPHY_STATUS_SUCCESS)
    {
        NVLOGE_FMT(NVLOG_PDSCH, AERIAL_CUPHY_EVENT, "Cell group cuphyDlRateMatchingGetDescrInfo error {}", status);
        throw cuphy::cuphy_exception(status);
    }

    status = cuphyPdschDmrsGetDescrInfo(&pDynDescrSizeBytes[PDSCH_DMRS], &pDynDescrAlignBytes[PDSCH_DMRS]);
    if(status != CUPHY_STATUS_SUCCESS)
    {
        NVLOGE_FMT(NVLOG_PDSCH, AERIAL_CUPHY_EVENT, "Cell group cuphyPdschDmrsGetDescrInfo error {}", status);
        throw cuphy::cuphy_exception(status);
    }

    status = cuphyModulationGetDescrInfo(&pDynDescrSizeBytes[PDSCH_MODULATION], &pDynDescrAlignBytes[PDSCH_MODULATION]);
    if(status != CUPHY_STATUS_SUCCESS)
    {
        NVLOGE_FMT(NVLOG_PDSCH, AERIAL_CUPHY_EVENT, "cuphyModulationGetDescrInfo error {}", status);
        throw cuphy::cuphy_exception(status);
    }

    status = cuphyPdschCsirsPrepGetDescrInfo(&pDynDescrSizeBytes[PDSCH_CSIRS_PREP], &pDynDescrAlignBytes[PDSCH_CSIRS_PREP]);
    if(status != CUPHY_STATUS_SUCCESS)
    {
        NVLOGE_FMT(NVLOG_PDSCH, AERIAL_CUPHY_EVENT, "cuphyPdschCsirsPrepGetDescrInfo error {}", status);
        throw cuphy::cuphy_exception(status);
    }

    status = cuphyPrepareCrcEncodeGetDescrInfo(&pDynDescrSizeBytes[PDSCH_CRC_PREP], &pDynDescrAlignBytes[PDSCH_CRC_PREP]);
    if(status != CUPHY_STATUS_SUCCESS)
    {
        NVLOGE_FMT(NVLOG_PDSCH, AERIAL_CUPHY_EVENT, "cuphyPdschPrepareCRCEncodeGetDescrInfo error {}", status);
        throw cuphy::cuphy_exception(status);
    }

    // Allocate workspace memory for some components to take advantage of the bulk async copy of the descriptors.

    // Allocate LDPC workspace buffers
    pDynDescrSizeBytes[PDSCH_LDPC_WORKSPACE]  = ldpc_workspace_bytes;
    pDynDescrAlignBytes[PDSCH_LDPC_WORKSPACE] = alignof(uint8_t);
    ldpc_workspace_offset = ldpc_workspace_bytes / per_cell_group_max_TBs; // per-TB offset; not dividing by 2 on purpose

    //Allocate memory for PdschPerTbParams
    pDynDescrSizeBytes[PDSCH_PER_TB_PARAMS]  = sizeof(PdschPerTbParams) * per_cell_group_max_TBs;
    pDynDescrAlignBytes[PDSCH_PER_TB_PARAMS] = alignof(PdschPerTbParams);

    //Allocate memory for RM workspace
    rm_allocated_workspace_size             = cuphyDlRateMatchingWorkspaceSize(per_cell_group_max_TBs);
    pDynDescrSizeBytes[PDSCH_RM_WORKSPACE]  = rm_allocated_workspace_size;
    pDynDescrAlignBytes[PDSCH_RM_WORKSPACE] = 256; //FIXME

    //Allocate memory for DMRS params
    pDynDescrSizeBytes[PDSCH_DMRS_PARAMS]  = sizeof(PdschDmrsParams) * per_cell_group_max_TBs;
    pDynDescrAlignBytes[PDSCH_DMRS_PARAMS] = alignof(PdschDmrsParams);

    //Allocate memory for UE group workspace
    pDynDescrSizeBytes[PDSCH_UE_GRP_WORKSPACE]  = sizeof(PdschUeGrpParams) * PDSCH_MAX_UE_GROUPS_PER_CELL_GROUP; //FIXME  add dyn. params runtime check for UE groups
    pDynDescrAlignBytes[PDSCH_UE_GRP_WORKSPACE] = alignof(PdschUeGrpParams);


    //Allocate memory for Xtf RE maps
    //pDynDescrSizeBytes[PDSCH_XTF_RE_MAPS]  = sizeof(uint16_t) * max_cells * per_cell_xtf_re_map_elements;
    //pDynDescrAlignBytes[PDSCH_XTF_RE_MAPS] = alignof(uint32_t);

    //Allocate memory for the per-TB CRCs, max_TBs TBs, to avoid the Memset during CRC setup
    //Caveat: the associated copy for this allocation will happen -as part of the bulk async. copy-
    //even if the per-TB CRCs are provided by the caller.
    pDynDescrSizeBytes[PDSCH_TB_CRCS]  = sizeof(uint32_t) * per_cell_group_max_TBs;
    pDynDescrAlignBytes[PDSCH_TB_CRCS] = alignof(uint32_t);

    m_component_descrs.alloc(dynDescrSizeBytes, dynDescrAlignBytes, &memory_footprint);
    //m_component_descrs.displayDescrSizes();

    ldpc_descr_offset = pDynDescrSizeBytes[PDSCH_LDPC] / PDSCH_MAX_HET_LDPC_CONFIGS_SUPPORTED; // PDSCH_MAX_HET_LDPC_CONFIGS_SUPPORTED descriptors overprovisioned; see ldpc.hpp

    // Added for convenience
    kernel_params = (PdschPerTbParams*)m_component_descrs.getCpuStartAddrs()[PDSCH_PER_TB_PARAMS];
    d_tbPrmsArray = (PdschPerTbParams*)m_component_descrs.getGpuStartAddrs()[PDSCH_PER_TB_PARAMS];

    h_rm_workspace = (uint32_t*)m_component_descrs.getCpuStartAddrs()[PDSCH_RM_WORKSPACE];
    d_rm_workspace = (uint32_t*)m_component_descrs.getGpuStartAddrs()[PDSCH_RM_WORKSPACE];

    h_dmrs_params = (PdschDmrsParams*)m_component_descrs.getCpuStartAddrs()[PDSCH_DMRS_PARAMS];
    d_dmrs_params = (PdschDmrsParams*)m_component_descrs.getGpuStartAddrs()[PDSCH_DMRS_PARAMS];

    h_ue_grp_params = (PdschUeGrpParams*)m_component_descrs.getCpuStartAddrs()[PDSCH_UE_GRP_WORKSPACE];
    d_ue_grp_params = (PdschUeGrpParams*)m_component_descrs.getGpuStartAddrs()[PDSCH_UE_GRP_WORKSPACE];

    h_ldpc_w_ptr = (uint8_t*)m_component_descrs.getCpuStartAddrs()[PDSCH_LDPC_WORKSPACE];
    d_ldpc_w_ptr = (uint8_t*)m_component_descrs.getGpuStartAddrs()[PDSCH_LDPC_WORKSPACE];

    //h_xtf_re_maps = (uint16_t*)m_component_descrs.getCpuStartAddrs()[PDSCH_XTF_RE_MAPS];
    //d_xtf_re_maps = (uint16_t*)m_component_descrs.getGpuStartAddrs()[PDSCH_XTF_RE_MAPS];
    //Set memory to 0 on the host-side once; will be copied to the device on every setup as part of the bulk async. copy
    memset(m_component_descrs.getCpuStartAddrs()[PDSCH_TB_CRCS], 0, pDynDescrSizeBytes[PDSCH_TB_CRCS]);
}

cuphyStatus_t PdschTx::expandParametersMultipleCells(cuphyPdschDynPrms_t* dyn_params,
                                            cudaStream_t         cuda_strm)
{
    if(dyn_params == nullptr)
    {
        NVLOGE_FMT(NVLOG_PDSCH, AERIAL_CUPHY_EVENT, "expandParametersMultipleCells() got cuphyPdschDynPrms_t nullptr!");
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    //cuphy::print_pdsch_dynamic_cell_group((cuphyPdschCellGrpDynPrm_t*)dyn_params->pCellGrpDynPrm);
    num_cells = dyn_params->pCellGrpDynPrm->nCells; // originally set to 1 in the constructor
    if(num_cells > max_cells)
    {
        NVLOGE_FMT(NVLOG_PDSCH, AERIAL_CUPHY_EVENT, "Number of cells in cell group is {} but current max. specified for PDSCH is {}", num_cells, max_cells);
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    cell_group_Emax    = 0; // set in prepareRateMatchingMultipleCells()

    // Check that stream priority passed via static parameters matches stream priority of cuda_strm
    int cuda_strm_prio = 0;
    CUDA_CHECK(cudaStreamGetPriority(cuda_strm, &cuda_strm_prio));
    if (cuda_strm_prio != static_params->stream_priority) {
        NVLOGD_FMT(NVLOG_PDSCH,"FYI CUDA stream priority mismatch! CUDA stream priority in static parameters is {} while priority of dynamic parameters' CUDA stream is {}",
                   static_params->stream_priority, cuda_strm_prio);
    }

    failed_ref_checks = 0; //reset # failed reference checks for subsequent pipeline run
    setReverseBits(false);

    num_TBs = dyn_params->pCellGrpDynPrm->nCws; // originally set to 1 in the constructor
    if (num_TBs > per_cell_group_max_TBs) {
        NVLOGE_FMT(NVLOG_PDSCH, AERIAL_CUPHY_EVENT, "Number of TBs in cell group is {} but current max. specified for PDSCH is {}", num_TBs, per_cell_group_max_TBs);
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    /*if(per_cell_max_TBs < num_TBs)
    {
        NVLOGE_FMT(NVLOG_PDSCH, AERIAL_CUPHY_EVENT, "Original buffer allocation was for {} TBs < {}", per_cell_max_TBs, num_TBs);
        throw std::runtime_error("Buffer allocation was for fewer TBs. Update PdschTx::setMaxValues");
    }*/


    if(dyn_params->pDataIn->pTbInput == nullptr)
    {
        NVLOGE_FMT(NVLOG_PDSCH, AERIAL_CUPHY_EVENT, "expandParametersMultipleCells() got empty array of pipeline input buffers!");
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }

    pdsch_data_in_is_gpu_buffer = (dyn_params->pDataIn->pBufferType == cuphyPdschDataIn_t::GPU_BUFFER);

    if(static_params->read_TB_CRC)
    {
        if(dyn_params->pTbCRCDataIn->pTbInput == nullptr)
        {
           NVLOGE_FMT(NVLOG_PDSCH, AERIAL_CUPHY_EVENT, "expandParametersMultipleCells() got empty array of TB CRC input buffers!");
           return CUPHY_STATUS_INVALID_ARGUMENT;
        }
    }

    failed_ref_checks = 0; // reset # failed reference checks for subsequent pipeline run
    for (int cell = 0; cell < num_cells; cell++) {
        per_cell_num_TBs[cell] = 0;
    }

    int num_UE_groups = dyn_params->pCellGrpDynPrm->nUeGrps;
    if (num_UE_groups > PDSCH_MAX_UE_GROUPS_PER_CELL_GROUP) {
        NVLOGE_FMT(NVLOG_PDSCH, AERIAL_CUPHY_EVENT, "Number of UE groups in cell group is {} but current max. specified for PDSCH is {}", num_UE_groups, PDSCH_MAX_UE_GROUPS_PER_CELL_GROUP);
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    for (int UE_group = 0; UE_group < num_UE_groups; UE_group++) {
        cuphyPdschUeGrpPrm_t* UE_group_ptr = &dyn_params->pCellGrpDynPrm->pUeGrpPrms[UE_group];
        int num_UEs = UE_group_ptr->nUes; // assume 1 CW per UE for now
        cuphyPdschCellDynPrm_t* cell_ptr = UE_group_ptr->pCellPrm;
        per_cell_num_TBs[cell_ptr->cellPrmDynIdx] += num_UEs; // assume 1 CW per UE
        //FIXME Also assuming all dynamic cells are used. That's OK
    }

    //PUSH_RANGE("RE_map_etc_loops", 3);
    //Compute RE map per cell before calling cuphyUpdatePdschDmrsParams, as it updates it in place
    _cuphyCsirsRrcDynPrm* csirs_params = dyn_params->pCellGrpDynPrm[0].pCsiRsPrms; // single cell group

    for (int cell = 0; cell < num_cells; cell++)
    {
        per_cell_TB_params_offset[cell]  = (cell == 0) ? 0 : (per_cell_num_TBs[cell - 1] + per_cell_TB_params_offset[cell - 1]);

        if(dyn_params->pDataIn->pTbInput[cell] == nullptr)
        {
            NVLOGE_FMT(NVLOG_PDSCH, AERIAL_CUPHY_EVENT, "expandParametersMultipleCells() got empty pipeline_input buffer!");
            return CUPHY_STATUS_INVALID_ARGUMENT;
        }
        per_cell_pipeline_input_bytes[cell] = dyn_params->pDataIn->pTbInput[cell];      // pointer to uint8_t

        if(static_params->read_TB_CRC)
        {
            if(dyn_params->pTbCRCDataIn->pTbInput[cell] == nullptr)
            {
                NVLOGE_FMT(NVLOG_PDSCH, AERIAL_CUPHY_EVENT, "expandParametersMultipleCells() got empty TB CRC input buffer!");
                return CUPHY_STATUS_INVALID_ARGUMENT;
            }
            h_per_cell_pipeline_input_tb_crcs[cell] =  dyn_params->pTbCRCDataIn->pTbInput[cell]; // pointer to uint8_t
        } else {
            h_per_cell_pipeline_input_tb_crcs[cell] =  nullptr;
        }

        per_cell_pipeline_input_size_bytes[cell] = 0;
        per_cell_pipeline_input_padding_size_bytes[cell] = 0;
        int cw_offset = per_cell_TB_params_offset[cell];
        for (int cw = 0; cw < per_cell_num_TBs[cell]; cw++) {
            //printf("cw %d size %d offset %d\n", cw, dyn_params->pCellGrpDynPrm->pCwPrms[cw_offset + cw].tbSize, dyn_params->pCellGrpDynPrm->pCwPrms[cw_offset + cw].tbStartOffset);
            // Since the pTbInput[cell] buffer can have padding > 1 byte between TBs, cannot just add up tbSize to determine per_cell_pipeline_input_size_bytes[cell].

            TB_sizes[cw_offset + cw] = dyn_params->pCellGrpDynPrm->pCwPrms[cw_offset + cw].tbSize * 8;
            cell_idx_for_TB[cw_offset + cw] = cell;
            // Assume no padding in last TB of each cell (can't infer if single TB either)
            per_cell_pipeline_input_padding_size_bytes[cell] += ((cw < per_cell_num_TBs[cell] -1) ? dyn_params->pCellGrpDynPrm->pCwPrms[cw_offset + cw+1].tbStartOffset - dyn_params->pCellGrpDynPrm->pCwPrms[cw_offset + cw].tbStartOffset - dyn_params->pCellGrpDynPrm->pCwPrms[cw_offset + cw].tbSize : 0); // round it up to be divisible by sizeof(uint32_t) later
        }
        //FIXME Making the assumption that last TB in each cell has the max. tbStartOffset, i.e., TBs are listed with ascening tbStartOffset. Otherwise would need to find the max of all TBs in cell.
        //Also, note the assumption that everything in between (i.e., from tbStartOffset = 0 of the first TB in the cell till then last TB) will be valid data, so can do a single H2D copy of the entire per_cell_pipeline_input_size_bytes[cell] bytes.
        per_cell_pipeline_input_size_bytes[cell] = dyn_params->pCellGrpDynPrm->pCwPrms[cw_offset + per_cell_num_TBs[cell]-1].tbStartOffset +
                                                   dyn_params->pCellGrpDynPrm->pCwPrms[cw_offset + per_cell_num_TBs[cell]-1].tbSize;

    }
    //POP_RANGE

    total_LDPC_kernel_configs = 0;

    // Update PDSCH DMRS parameters first before calling cuphySetTBParamsFromStructs.
    //PUSH_RANGE("updateDmrsPars", 3);
    cuphyStatus_t dmrs_params_status = cuphyUpdatePdschDmrsParams(h_dmrs_params, dyn_params, static_params, h_ue_grp_params);
    //POP_RANGE
    if (dmrs_params_status != CUPHY_STATUS_SUCCESS)
    {
        NVLOGE_FMT(NVLOG_PDSCH, AERIAL_CUPHY_EVENT, "Error when setting TB config parameters! cuphyUpdatePdschDmrsParams returned {}", cuphyGetErrorString(dmrs_params_status));
        return dmrs_params_status;
    }

    //PUSH_RANGE("updateTBParams", 3);
    cuphyStatus_t params_status = cuphySetTBParamsFromStructs(kernel_params, dyn_params->pCellGrpDynPrm, static_params);
    //POP_RANGE
    if(params_status != CUPHY_STATUS_SUCCESS)
    {
        NVLOGE_FMT(NVLOG_PDSCH, AERIAL_CUPHY_EVENT, "Error when setting TB config parameters! cuphySetTBParamsFromStructs returned {}", cuphyGetErrorString(params_status));
        return params_status;
    }
#if 0 // DBG print info
    for(int i = 0; i < num_TBs; i++)
    {
        print_TB_config_params(kernel_params, h_dmrs_params, i, num_TBs, true, true);
    }
#endif

    // Populate buffers etc for all pipeline components
    //PUSH_RANGE("prepareBuffers", 3);
    cuphyStatus_t prep_buffers_status = prepareBuffersMultipleCells(cuda_strm);
    //POP_RANGE

    if(prep_buffers_status != CUPHY_STATUS_SUCCESS)
    {
        return prep_buffers_status;
    }

    // Bulk copy all descriptors
    if(bulk_desc_async_copy)
    {
        m_component_descrs.asyncCpuToGpuCpy(cuda_strm);
    }

    // Grap update can take some time; ensure bulk async. copy has been issued beforehand
    if(graph_mode)
    {
        // Graph has been instantiated in the constructor
        updateNodeParamsMultipleCells(); // in exec_graph
        //CU_CHECK_EXCEPTION(cuGraphUpload(exec_graph, cuda_strm));
    }

    for (int cell = 0; cell < num_cells; cell += 1) {
        uint16_t static_cell = dyn_params->pCellGrpDynPrm->pCellPrms[cell].cellPrmStatIdx;
        //This has to be *after* prepareBuffers; different code path for prepareCRC depending on whether input_file is nullptr or not
        if(static_params->pDbg[static_cell].refCheck)
        {
            per_cell_hdf5_filename[cell] = static_params->pDbg[static_cell].pCfgFileName; // calls new() so avoid unless refCheck enables
            //NVLOGC_FMT(NVLOG_PDSCH, "cell {} has filename {}", cell, per_cell_hdf5_filename[cell].c_str());
            if(!per_cell_hdf5_filename[cell].empty())
            {
                per_cell_input_file[cell] = std::unique_ptr<hdf5hpp::hdf5_file>(new hdf5hpp::hdf5_file(hdf5hpp::hdf5_file::open(per_cell_hdf5_filename[cell].c_str())));
            }
        }
    }

    #if PRINT_CSV_FRIENDLY_CONFIG
    print_csv_friendly_config();
    #endif
    //Uncomment to print per cell aerial metrics
    //cuphy::print_pdsch_cell_aerial_metrics((cuphyPdschCellGrpDynPrm_t*)dyn_params->pCellGrpDynPrm);
    return CUPHY_STATUS_SUCCESS;
}

PdschTx::~PdschTx()
{
    CUresult r;
    //Reminder: A graph is now created and instantiated in PdschTx() unconditionally
    r = cuGraphExecDestroy(exec_graph);
    if(r != CUDA_SUCCESS)
    {
        NVLOGE_FMT(NVLOG_PDSCH, AERIAL_CUPHY_EVENT, "cuGraphExecDestroy error");
    }
    r = cuGraphDestroy(m_graph);
    if(r != CUDA_SUCCESS)
    {
        NVLOGE_FMT(NVLOG_PDSCH, AERIAL_CUPHY_EVENT, "cuGraphDestroy error");
    }

#if AVOID_STREAM_CREATION_IN_SETUP == 0
    // Destroy all ldpc_streams except for the first one, the stream
    // with which PdschTx class was called. It's the responsibility of the caller
    // to destroy that one.
    for(int TB_id = 1; TB_id <= created_ldpc_streams; TB_id++)
    {
        if(ldpc_streams[TB_id] != 0) // Check not strictly needed when iterating over created streams
        {
            CUDA_CHECK_NO_THROW(cudaStreamDestroy(ldpc_streams[TB_id]));
        }
    }
#endif
}

cuphyStatus_t PdschTx::prepareCRC_step2MultipleCells(cudaStream_t cuda_strm)
{
    // Called outside cuphySetupCrcEncode
    cuphyStatus_t status = cuphySetupPrepareCRCEncode(
            m_prepare_crc_encode_launch_cfg.get(),
            pdsch_data_in_is_gpu_buffer ? nullptr : ((REVERT_TO_ASYNC_COPIES == 1) ? (uint32_t*) d_prepare_crc_input_buffer.get() : nullptr), // if nullptr the tbStartAddr from d_tbPrmsArray will be used
            (uint32_t*)cell_group_crc_d_in_tensor_ref.addr(),
            (uint32_t*)d_ldpc_workspace.get(),
            d_tbPrmsArray,
            num_TBs, // total number of TBs across all cells
            cell_max_CBs, // max # CBs per TB across all cells
            max_tb_size_bytes, //max TB size in bytes across all cells
            static_cast<void*>(m_component_descrs.getCpuStartAddrs()[PDSCH_CRC_PREP]), //CPU desc
            static_cast<void*>(m_component_descrs.getGpuStartAddrs()[PDSCH_CRC_PREP]), //GPU desc
            (!bulk_desc_async_copy),
            cuda_strm);
    return status;
}

cuphyStatus_t PdschTx::prepareCRCMultipleCells(cudaStream_t cuda_strm)
{
    if(dynamic_params == nullptr)
    {
        NVLOGE_FMT(NVLOG_PDSCH, AERIAL_CUPHY_EVENT, "dynamic parameters nullptr or input file not nullptr");
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    has_cells_in_TM = false; // reset on every setup

    uint32_t per_cell_CRC_input_bytes[PDSCH_MAX_CELLS_PER_CELL_GROUP];
    uint32_t per_cell_total_padding_bytes[PDSCH_MAX_CELLS_PER_CELL_GROUP];
    int max_num_CBs[PDSCH_MAX_CELLS_PER_CELL_GROUP];

    std::memset(per_cell_CRC_output_bytes.data(), 0, num_cells * sizeof(uint32_t));
    std::memset(per_cell_CRC_input_bytes, 0, num_cells * sizeof(uint32_t));
    std::memset(per_cell_total_padding_bytes, 0, num_cells * sizeof(uint32_t));
    std::memset(per_cell_max_tb_size_bytes.data(), 0, num_cells * sizeof(uint32_t));
    std::memset(max_num_CBs, 0, num_cells * sizeof(int));
    std::memset(per_cell_CRC_max_TB_padded_bytes.data(), 0, num_cells * sizeof(uint32_t));
    cell_group_CRC_output_bytes = 0;
    cell_CRC_max_TB_padded_bytes = 0;
    max_tb_size_bytes = 0;

    for(int i = 0; i < num_TBs; i++)
    {
        uint16_t dyn_cell = cell_idx_for_TB[i];
        int num_CBs = kernel_params[i].num_CBs;

        uint32_t per_CB_crc_byte_size = (num_CBs == 1) ? 0 : 3; // per-CB
        uint32_t per_TB_crc_byte_size = (TB_sizes[i] > 3824) ? 3 : 2;

        /*if((kernel_params[i].K % 8) != 0)
        {
            NVLOGE_FMT(NVLOG_PDSCH, AERIAL_CUPHY_EVENT, "CRC preprocessing heads-up! K {} not divisible by 8!", kernel_params[i].K);
        }
        if((kernel_params[i].F % 8) != 0)
        {
            NVLOGE_FMT(NVLOG_PDSCH, AERIAL_CUPHY_EVENT, "CRC preprocessing heads-up! F {} not divisible by 8!", kernel_params[i].F);
        }*/
        uint32_t total_CB_byte_size = div_round_up<uint32_t>(kernel_params[i].K, 8); // K in bytes (includes CRC and filler bits)
        CB_data_byte_sizes[i] = total_CB_byte_size - per_CB_crc_byte_size - div_round_up<uint32_t>(kernel_params[i].F, 8);
        // LDPC expects each K to start at a 32-bit aligned boundary
        total_CB_byte_sizes[i] = div_round_up<uint32_t>(kernel_params[i].K, BITS_PER_U32) * sizeof(uint32_t);

        per_cell_CRC_output_bytes[dyn_cell] += (total_CB_byte_sizes[i] * kernel_params[i].num_CBs); // K * num_CBs in bytes

        TB_padded_byte_sizes[i] = CB_data_byte_sizes[i] * kernel_params[i].num_CBs; // includes per-TB CRC
        if (kernel_params[i].testModel == 0)
        {
            // TB's cell is not in testing mode
            per_cell_CRC_input_bytes[dyn_cell] += TB_padded_byte_sizes[i];
            tb_size[i] = TB_padded_byte_sizes[i];

            per_cell_CRC_input_bytes[dyn_cell] -= per_TB_crc_byte_size;
            tb_size[i] -= per_TB_crc_byte_size;
        } 
        else
        {
            // TB's cell is in testing mode and the TB payload buffer holds bits from the PN23 (pseudo random sequence)
            // instead of a TB. Size in this case cannot be computed from from K and CBs.
            tb_size[i] = dynamic_params->pCellGrpDynPrm->pCwPrms[i].tbSize;
            per_cell_CRC_input_bytes[dyn_cell] += tb_size[i];
            has_cells_in_TM = true; 
        }

        if(kernel_params[i].num_CBs > max_num_CBs[dyn_cell])
        {
            max_num_CBs[dyn_cell] = kernel_params[i].num_CBs;
        }
        padding_bytes[i] = div_round_up<uint32_t>(tb_size[i], sizeof(uint32_t)) * sizeof(uint32_t) - tb_size[i];
        padding_bytes[i] += (padding_bytes[i] <= 2) ? sizeof(uint32_t) : 0;
        per_cell_total_padding_bytes[dyn_cell] += padding_bytes[i];

        // Update tbStartOffset and tbSize (both in Bytes) in PdschPerTbParams
        kernel_params[i].tbStartOffset = dynamic_params->pCellGrpDynPrm->pCwPrms[i].tbStartOffset;
        kernel_params[i].tbSize        = dynamic_params->pCellGrpDynPrm->pCwPrms[i].tbSize;
        /* tbStartAddr contains the starting address for that TB in pinned host memory. When inter_cell_batching is enabled with multiple cells,
           the CRC kernel launched during setup will access that memory directly.
        */
        kernel_params[i].tbStartAddr   = per_cell_pipeline_input_bytes[dyn_cell] + dynamic_params->pCellGrpDynPrm->pCwPrms[i].tbStartOffset;

        //kernel_params[i].paddingBytes = padding_bytes[i];
        kernel_params[i].cumulativeTbSizePadding = ((i == 0) ? 0 : kernel_params[i-1].cumulativeTbSizePadding) +  kernel_params[i].tbSize + padding_bytes[i];
        per_cell_max_tb_size_bytes[dyn_cell]            = std::max(per_cell_max_tb_size_bytes[dyn_cell], kernel_params[i].tbSize);
        
        //int dbg_case = kernel_params[i].cumulativeTbSizePadding - ((i == 0) ? 0 : kernel_params[i-1].cumulativeTbSizePadding);
        //NVLOGC_FMT(NVLOG_PDSCH, "TB {} dbg-case {} vs. tb size {} = {}", i, dbg_case, kernel_params[i].tbSize, dbg_case - kernel_params[i].tbSize);

        tb_size[i] *= 8; // convert to bits

        per_cell_CRC_max_TB_padded_bytes[dyn_cell] = std::max(per_cell_CRC_max_TB_padded_bytes[dyn_cell], TB_padded_byte_sizes[i]);
        //print_TB_config_params(kernel_params, h_dmrs_params, i, num_TBs,  layer_mapping, scrambling);

        uint32_t TB_ldpc_signature = LDPC_signature(kernel_params[i].bg, kernel_params[i].Zc, kernel_params[i].num_CBs); // Having grouping include #CBs simplifies LDPC work iff configs grouped are not contiguous.
        //NVLOGC_FMT(NVLOG_PDSCH, "TB {} has signature {:#x}, BG {}, Zc {}, num_CBs {}", i, TB_ldpc_signature, kernel_params[i].bg, kernel_params[i].Zc, kernel_params[i].num_CBs); //WIP intention is to keep on grouping, across cells too, as long as the signature is identical.
        bool found = (kernel_params[i].testModel != 0); // if TB is in testing mode,  set found variable to true so this TB doesn't contribute to an LDPC kernel config
        
        for (int cfg = 0; (cfg < total_LDPC_kernel_configs) && !found; cfg++) {
            if (LDPC_batches[cfg].signature== TB_ldpc_signature) {
                LDPC_batches[cfg].TB_idxs[LDPC_batches[cfg].num_TBs] = i; // TB_idxs contains the TB identifier not on a per-cell basis but across all cells in the group
                LDPC_batches[cfg].num_TBs += 1;
                found = true;
            }
        }
        if (!found) {
            LDPC_batches[total_LDPC_kernel_configs].signature = TB_ldpc_signature;
            LDPC_batches[total_LDPC_kernel_configs].num_TBs = 1;
            LDPC_batches[total_LDPC_kernel_configs].TB_idxs[0] = i;
            total_LDPC_kernel_configs += 1;

        }

        CRC_dst_offset[i] = (i == 0) ? 0 : (CRC_dst_offset[i-1] + (((kernel_params[i-1].K + 31) >> 5)* sizeof(uint32_t) * kernel_params[i-1].num_CBs)); // in bytes for now.
        //NB: the LDPC_dst_offset should match what the fused rate matching + modulation kernel expects
        uint32_t additional_TM_offset = ((kernel_params[i-1].tbSize*8 + 31) >> 5)*sizeof(uint32_t); //in bytes for now or divide by BITS_PER_U32 instead

        LDPC_dst_offset[i] = (i == 0) ? 0 : (LDPC_dst_offset[i-1] + \
                                            ((kernel_params[i-1].testModel == 0) ? (((kernel_params[i-1].N + 31) >> 5)* sizeof(uint32_t) * kernel_params[i-1].num_CBs) : \
                                            additional_TM_offset)); //in bytes for now or divide by BITS_PER_U32 instead

        // Need to populate an offset array for TM TBs, since the PN sequence payload should be copied during the prepare CRC step
        // to a different destination buffer (LDPC's output/rate-matching+modulation input) than the TB payload of non TM TBs.
        if (kernel_params[i].testModel != 0) {
            uint32_t* tmp = (uint32_t*)m_component_descrs.getCpuStartAddrs()[PDSCH_CRC_PREP]; //assumes offset array is first field of the descriptor struct
            tmp[i] = LDPC_dst_offset[i]; // offset in bytes of every TB. 
            //NVLOGC_FMT(NVLOG_PDSCH, "TB %d has offset of bytes {} = {} and tbSize {} B", i, tmp[i], LDPC_dst_offset[i], kernel_params[i].tbSize);
        }

           
    }
    if (total_LDPC_kernel_configs > PDSCH_MAX_HET_LDPC_CONFIGS_SUPPORTED) // See ldpcEncodeDescr_t_array for size. TODO change as needed here and in graphs
    {
        NVLOGE_FMT(NVLOG_PDSCH, AERIAL_CUPHY_EVENT, "Number of LDPC batched configs ({}) is greater than the current max. supported ({})!", total_LDPC_kernel_configs, PDSCH_MAX_HET_LDPC_CONFIGS_SUPPORTED);
        return CUPHY_STATUS_UNSUPPORTED_CONFIG;
    }
#if 0
    for (int cfg = 0; cfg < total_LDPC_kernel_configs; cfg++) {
        std::stringstream ldpc_batch_tbs_info;
        for (int idx = 0; idx < LDPC_batches[cfg].num_TBs; idx++) {
            ldpc_batch_tbs_info << std::to_string(LDPC_batches[cfg].TB_idxs[idx]) << " ";
        }
        NVLOGC_FMT(NVLOG_PDSCH, "LDPC cfg {} has signature {}, TBs {} and TB_idxs = {{{}}}", cfg, LDPC_batches[cfg].signature, LDPC_batches[cfg].num_TBs, ldpc_batch_tbs_info.str().c_str());
    }
#endif

    // Update per cell metrics if relevant pointer is provided
    if(dynamic_params->pCellGrpDynPrm->pCellMetrics == nullptr)
    {
        NVLOGW_FMT(NVLOG_PDSCH, "pCellMetrics pointer for PDSCH cell group is nullptr. No per cell metrics will be updated");
    }
    else
    {
        for (int cell = 0; cell < num_cells; cell++)
        {
            dynamic_params->pCellGrpDynPrm->pCellMetrics[cell].nTBs = per_cell_num_TBs[cell];
            dynamic_params->pCellGrpDynPrm->pCellMetrics[cell].tbSize = per_cell_CRC_input_bytes[cell];  // in bytes
        }
    }

    cell_max_CBs = 0;
    cell_group_crc_h_in_tensor_bytes = 0;
    m_batchedMemcpyHelper.reset();
    for (int cell = 0; cell < num_cells; cell++) {
        per_cell_max_CBs[cell] = max_num_CBs[cell];
        cell_max_CBs = std::max(cell_max_CBs, per_cell_max_CBs[cell]);
        max_tb_size_bytes = std::max(per_cell_max_tb_size_bytes[cell], max_tb_size_bytes);

        if (max_num_CBs[cell] > max_CBs_per_TB) {
            NVLOGE_FMT(NVLOG_PDSCH, AERIAL_CUPHY_EVENT, "Max. number of CBs for cell {} in TB is {} but current max. specified for PDSCH is {}", cell, max_num_CBs[cell], max_CBs_per_TB);
            return CUPHY_STATUS_INVALID_ARGUMENT;
        }

        cell_CRC_max_TB_padded_bytes = std::max( cell_CRC_max_TB_padded_bytes,  per_cell_CRC_max_TB_padded_bytes[cell]);
        cell_group_CRC_output_bytes += per_cell_CRC_output_bytes[cell];

        int rounded_up_per_cell_pipeline_input_padding_size_bytes = round_up_to_next<int>(per_cell_pipeline_input_padding_size_bytes[cell], sizeof(uint32_t));
        per_cell_crc_h_in_tensor_bytes[cell] = per_cell_CRC_input_bytes[cell] + per_cell_total_padding_bytes[cell] + rounded_up_per_cell_pipeline_input_padding_size_bytes;
        per_cell_crc_h_in_tensor_bytes_offset[cell] = (cell == 0) ? 0 : (per_cell_crc_h_in_tensor_bytes[cell - 1] + per_cell_crc_h_in_tensor_bytes_offset[cell - 1]);
        //NVLOGC_FMT(NVLOG_PDSCH, "cell {} has per_cell_CRC_input_bytes {}, padding {} extra padding (without final round up){}, offset {}",
        //                       cell, per_cell_CRC_input_bytes[cell], per_cell_total_padding_bytes[cell], per_cell_pipeline_input_padding_size_bytes[cell], per_cell_crc_h_in_tensor_bytes_offset[cell]);
 

        cell_group_crc_h_in_tensor_bytes += per_cell_crc_h_in_tensor_bytes[cell];

        // Check that input buffer is as expected. per_cell_CRC_input_bytes only accounts for the TBs (no extra padding) and per_cell_pipeline_input_padding_size_byte adds the extra padding.
        // per_cell_pipeline_input_size_bytes is what will be used for the size of the H2D copy in case of a CPU_BUFFER buffer.
        if(per_cell_CRC_input_bytes[cell] + per_cell_pipeline_input_padding_size_bytes[cell] < per_cell_pipeline_input_size_bytes[cell])
        {
            NVLOGE_FMT(NVLOG_PDSCH, AERIAL_CUPHY_EVENT, "Error! Input buffer for cell {} has more elements than expected ({} vs. {}). Some will be removed!",
                       cell, per_cell_pipeline_input_size_bytes[cell], per_cell_CRC_input_bytes[cell] + per_cell_pipeline_input_padding_size_bytes[cell]);
        }
        else if(per_cell_CRC_input_bytes[cell] + per_cell_pipeline_input_padding_size_bytes[cell] > per_cell_pipeline_input_size_bytes[cell])
        {
            NVLOGE_FMT(NVLOG_PDSCH, AERIAL_CUPHY_EVENT, "Cell {} expected # CRC input bytes {} > # Pipeline input buffer bytes {}",
                       cell, per_cell_CRC_input_bytes[cell] + per_cell_pipeline_input_padding_size_bytes[cell], per_cell_pipeline_input_size_bytes[cell]);
            //throw std::runtime_error("Pipeline input buffer has fewer bytes than expected CRC input.");
            return CUPHY_STATUS_INVALID_ARGUMENT;
        }

        if(h_per_cell_pipeline_input_tb_crcs[cell]) {
            CUDA_CHECK(cudaMemcpyAsync(d_TB_CRCs.get() + per_cell_TB_params_offset[cell],
                                       h_per_cell_pipeline_input_tb_crcs[cell],
                                       per_cell_num_TBs[cell] * sizeof(uint32_t),
                                       cudaMemcpyHostToDevice,
                                       cuda_strm));
        }


        if (!pdsch_data_in_is_gpu_buffer) {
             if (REVERT_TO_ASYNC_COPIES == 1) {
                 /*NVLOGC_FMT(NVLOG_PDSCH, "REVERT_TO_ASYNC_COPIES cell {}: copy from {:p} {} B to dst offset {}",
                        cell, (void*)per_cell_pipeline_input_bytes[cell], per_cell_pipeline_input_size_bytes[cell],
                        per_cell_crc_h_in_tensor_bytes_offset[cell]);*/

                  m_batchedMemcpyHelper.updateMemcpy((uint32_t*)d_prepare_crc_input_buffer.get() + (per_cell_crc_h_in_tensor_bytes_offset[cell] / sizeof(uint32_t)),
                                                     (void*)per_cell_pipeline_input_bytes[cell],
                                                     per_cell_pipeline_input_size_bytes[cell],
                                                     cudaMemcpyHostToDevice,
                                                     cuda_strm);
             } else {
                 // Double check that memory is host-pinned. Only relevant when running with the control plane too. For standalone cuPHY this is always the case.
                 // In case of REVERT_TO_ASYNC_COPIES=1, no error will be triggered but the memory copy will be slower. It's also OK to always enable this check.
                 cudaPointerAttributes input_attributes;
                 CUDA_CHECK(cudaPointerGetAttributes(&input_attributes, per_cell_pipeline_input_bytes[cell]));
                 if  (input_attributes.type == cudaMemoryTypeUnregistered )
                 {
                     NVLOGE_FMT(NVLOG_PDSCH, AERIAL_CUPHY_EVENT, "Error! PDSCH data input pointer not accessible by device; should be pinned memory!");
                     return CUPHY_STATUS_INVALID_ARGUMENT;
                 }
             }
        }
    }

    if ((!pdsch_data_in_is_gpu_buffer) && (REVERT_TO_ASYNC_COPIES == 1))
    {
        cuphyStatus_t batched_memcpy_status = m_batchedMemcpyHelper.launchBatchedMemcpy(cuda_strm); // If an error occured, a CUPHY_STATUS_INVALID_ARGUMENT will be returned.
        if (batched_memcpy_status != CUPHY_STATUS_SUCCESS)
        {
            NVLOGE_FMT(NVLOG_PDSCH, AERIAL_CUPHY_EVENT, "Error! Launching the PDSCH data input buffer(s) batched memory copy returned an error!");
            return CUPHY_STATUS_INVALID_ARGUMENT;
        }
    }

    // Temp code just to see if having H2D copies instead of a kernel accessing pinned memory directly avoids dropped packets for UL.
    if (REVERT_TO_ASYNC_COPIES == 1) {
        for(int i = 0; i < num_TBs; i++)
        {
            uint16_t dyn_cell = cell_idx_for_TB[i];
            kernel_params[i].tbStartOffset += per_cell_crc_h_in_tensor_bytes_offset[dyn_cell];
        }
    }


    cell_group_crc_d_in_tensor_ref.desc().set(CUPHY_BIT,
                                              2 * cell_group_crc_h_in_tensor_bytes * 8,
                                              cuphy::tensor_flags::align_tight);
    cell_group_crc_d_in_tensor_ref.set_addr(d_crc_workspace.get());

    //NB: The call to cuphyPrepareCRCEncode happens in expandParametersMultipleCells(),
    //as it requires the descr. bulk async copy to happen first.
    cuphyStatus_t status = cuphySetupCrcEncode(m_crc_encode_launch_cfg.get(),
#if DBG_PDSCH_CRC
                                               d_CB_CRCs.get(), // output CB CRCs
#else
                                               nullptr,
#endif
                                               /* Hardcode cell 0 to see if TB CRCs are computed by cuPHY or read. Assuming similar across all cells*/
                                               h_per_cell_pipeline_input_tb_crcs[0] ? d_TB_CRCs.get() : \
                                                          (uint32_t*)m_component_descrs.getGpuStartAddrs()[PDSCH_TB_CRCS], // output TB CRCs
                                               (const uint32_t*)cell_group_crc_d_in_tensor_ref.addr(),
                                               (uint8_t*)d_code_blocks.get(), // output code blocks
                                               d_tbPrmsArray,
                                               num_TBs, //per_cell_num_TBs[0],
                                               cell_max_CBs, // max # CBs per TB across all cells
                                               cell_CRC_max_TB_padded_bytes, //max TB bytes
                                               1, //reverse bytes
                                               read_TB_CRC,  // if true compute CB-CRC only, i.e., skip TB CRC computation
                                               static_cast<void*>(m_component_descrs.getCpuStartAddrs()[PDSCH_CRC]), //CPU desc
                                               static_cast<void*>(m_component_descrs.getGpuStartAddrs()[PDSCH_CRC]), //GPU desc
                                               (!bulk_desc_async_copy),
                                               cuda_strm);

    if(status != CUPHY_STATUS_SUCCESS)
    {
        NVLOGE_FMT(NVLOG_PDSCH, AERIAL_CUPHY_EVENT, "Invalid argument(s) for cuphySetupCrcEncode");
    }
    return status;
}


void PdschTx::runCRCMultipleCells(cudaStream_t cuda_strm, bool ref_check)
{
    // Run prepare-crc kernel
    CUresult status_prepare = launch_kernel(m_prepare_crc_encode_launch_cfg.get()->m_kernelNodeParams, cuda_strm);
    if(status_prepare != CUDA_SUCCESS)
    {
       throw std::runtime_error("Prepare CRC Encode error(s)");
    }

    // CRC has 2 kernels: one for per-TB CRC computation and one for per-CB CRC computation
    CUresult status_k1 = (read_TB_CRC) ? CUDA_SUCCESS : launch_kernel(m_crc_encode_launch_cfg.get()->m_kernelNodeParams[0], cuda_strm);
    CUresult status_k2 = launch_kernel(m_crc_encode_launch_cfg.get()->m_kernelNodeParams[1], cuda_strm);
    if((status_k1 != CUDA_SUCCESS) ||
       (status_k2 != CUDA_SUCCESS))
    {
       throw std::runtime_error("CRC Encode error(s)");
    }

    crc_event.record(cuda_strm);


    if(ref_check)
    {
        failed_ref_checks += (refCheckCRCMultipleCells(cuda_strm) != 0) ? 1 : 0;
    }
}


std::vector<uint32_t> PdschTx::getHostOutputCRCMultipleCells(uint16_t cell, cudaStream_t cuda_strm)
{
    if(!ran_pipeline)
    {
        throw std::runtime_error("Cannot get CRC's output without calling Run first.");
    }
    size_t bytes = cell_group_CRC_output_bytes;
    std::vector<uint32_t> h_CRC_output((bytes + sizeof(uint32_t) - 1) / sizeof(uint32_t));
    // This copy can trigger an initcheck error (compute-sanitizer) during ref. checks if there are cells in testing mode, even though that part of the buffer is ignored.
    // Solution is to proactively memset the entire d_code_blocks buffer if ref. checks are enabed in PdschTx constructor.
    CUDA_CHECK(cudaMemcpyAsync(h_CRC_output.data(), d_code_blocks.get() + cell * max_per_cell_code_blocks_bytes, bytes, cudaMemcpyDeviceToHost, cuda_strm));

    return h_CRC_output;
}


const uint8_t* PdschTx::getGPUOutputCRC()
{
    return d_code_blocks.get();
}

int PdschTx::refCheckCRCMultipleCells(cudaStream_t cuda_strm)
{
    using tensor_pinned_R_8U = typed_tensor<CUPHY_R_8U, pinned_alloc>;
    int total_error_cnt = 0;
    std::vector<uint32_t> h_CRC_output;
    h_CRC_output = getHostOutputCRCMultipleCells(0, cuda_strm);
    CUDA_CHECK(cudaStreamSynchronize(cuda_strm));

    uint32_t per_TB_offset = 0;
    for (int cell = 0; cell < num_cells; cell++) {

        if(per_cell_input_file[cell] == nullptr)
        {
            NVLOGE_FMT(NVLOG_PDSCH, AERIAL_CUPHY_EVENT, "Cannot do a reference check without a valid input file!");
            return -1;
        }

        // Populate host buffer with CRC output and reset per_TB_offset if there's no inter-cell batching
    #if DBG_PDSCH_CRC
        // Get per-TB CRCs for debugging purposes.
        std::vector<uint32_t> h_TB_CRCs(per_cell_num_TBs[cell]);
        CUDA_CHECK(cudaMemcpyAsync(h_TB_CRCs.data(),
                                   h_per_cell_pipeline_input_tb_crcs[cell] ? d_TB_CRCs.get() + per_cell_TB_params_offset[cell]: \
                                                              (uint32_t*)m_component_descrs.getGpuStartAddrs()[PDSCH_TB_CRCS] + per_cell_TB_params_offset[cell],
                                   per_cell_num_TBs[cell] * sizeof(uint32_t), cudaMemcpyDeviceToHost, cuda_strm));
        CUDA_CHECK(cudaStreamSynchronize(cuda_strm));
        //NVLOGC_FMT(NVLOG_PDSCH, "cell {}, per_cell_num_TBs[cell] {} per_cell_TB_params_offset[cell] {}", cell, per_cell_num_TBs[cell], per_cell_TB_params_offset[cell]);
        //for (int i = 0; i < h_TB_CRCs.size(); i++) NVLOGC_FMT(NVLOG_PDSCH, "cell {} TB {} has TB-CRC {}", cell, i, h_TB_CRCs[i]);
    #endif

        uint32_t error_cnt = 0;
        for(int TB_id = 0; TB_id < per_cell_num_TBs[cell]; TB_id++)
        {
            int overall_TB_id = per_cell_TB_params_offset[cell] + TB_id;
            bool cell_not_in_testing_mode = (kernel_params[overall_TB_id].testModel == 0);

            std::string         ref_dataset_name = "tb" + std::to_string(TB_id) + "_cbs";
            tensor_pinned_R_8U crc_ref_output; // only read if cell is not in testing mode; otherwise, dataset would be empty and next line would throw an exception
            if (cell_not_in_testing_mode) crc_ref_output   = typed_tensor_from_dataset<CUPHY_R_8U, pinned_alloc>((*per_cell_input_file[cell]).open_dataset(ref_dataset_name.c_str()));

            int K       = kernel_params[overall_TB_id].K;
            int rounded_up_K = round_up_to_next(K, BITS_PER_U32);
            int num_CBs = kernel_params[overall_TB_id].num_CBs;

            uint32_t K_mask = (1 << (K % BITS_PER_U32)) - 1; // Mask to apply if K not evenly divisible by BITS_PER_U32

            //Compare code_blocks w/ tb_cbs reference input.
            uint32_t per_TB_error_cnt = 0;
            for(int CB = 0; ((CB < num_CBs) && cell_not_in_testing_mode); CB += 1)
            {
                for(int k_element_start = 0; k_element_start < K; k_element_start += BITS_PER_U32)
                { // In bits
                    uint32_t ref_bits = 0;
                    for(int offset = 0; offset < BITS_PER_U32; offset++)
                    {
                        if(k_element_start + offset < K)
                        {
                            uint32_t bit = crc_ref_output(k_element_start + offset, CB);
                            // 1st element of h5 file's sourceData datatset will map to the
                            // least significant bit of a tensor element
                            ref_bits |= (bit << offset);
                        }
                    }
                    int      GPU_index  = per_TB_offset + (rounded_up_K / BITS_PER_U32) * CB + (k_element_start / BITS_PER_U32);
                    uint32_t GPU_bits   = h_CRC_output[GPU_index];
                    int      element_id = k_element_start / BITS_PER_U32;
                    if(k_element_start + BITS_PER_U32 > K)
                    {
                        GPU_bits = GPU_bits & K_mask;
                    }
                    if(ref_bits != GPU_bits)
                    {
                        per_TB_error_cnt += 1;
                        /*if (per_TB_error_cnt < 10)
                        NVLOGC_FMT(NVLOG_PDSCH, "Cell {}: CRC mismatch for TB {}, CB {}, element {}, GPU index {}: Expected reference {:#x} and got {:#x}",
                               cell, TB_id, CB, element_id, GPU_index, ref_bits, GPU_bits);*/
                    }
                }
            }
            error_cnt += per_TB_error_cnt;
            per_TB_offset += ((num_CBs * rounded_up_K) / BITS_PER_U32); // should be incremented even for TBs whose cells are in testing mode,
                                                                        // as a mix of cells in testing and regual mode is possible
        }

        NVLOGC_FMT(NVLOG_PDSCH, "");
        NVLOGC_FMT(NVLOG_PDSCH, "CRC Error Count: {}; GPU output compared w/ reference dataset(s) <tb*_cbs> from <{}>", error_cnt, per_cell_hdf5_filename[cell]);

        total_error_cnt += error_cnt;
    }

    return total_error_cnt;
}


uint32_t PdschTx::LDPC_signature(uint32_t BG, uint32_t Zc, uint32_t num_CBs, uint32_t F)
{

    uint32_t signature = (BG == 1) ? 0 : 1;
    signature |= (Zc << 1);
    signature |= (F << 10);
    signature |= (num_CBs << 24);

    return signature;
}


cuphyStatus_t PdschTx::prepareLDPCBatching(cudaStream_t cuda_strm)
{
    //NVLOGC_FMT(NVLOG_PDSCH, "total_LDPC_kernel_configs {}", total_LDPC_kernel_configs);

    bool puncture_bits = true;
    uint32_t cumulative_TBs = 0;

    // Reset pool index so we pick a stream from the beginning of the pool
    // Without this, if you are running a config. w/ 5 LDPC kernels per cell group, you'd use
    // streams [0, 3] for the first setup (one kernel will use cuda_strm)
    // and streams [4, 7] for the second setup (again one kernel will use cuda_strm), assuming your pool size > 7.
    // With the advance_to_start() option, you'll only use streams [0, 3] in both setup calls.
    ldpc_stream_pool.advance_to_start();

    //Future TODO If it is just one config, it's special case and we don't need to worry about reordering, addr. offsets etc.
    for (int LDPC_cfg = 0; LDPC_cfg < total_LDPC_kernel_configs; LDPC_cfg++) {
        int first_TB_idx = LDPC_batches[LDPC_cfg].TB_idxs[0];
        int TBs_in_batch = LDPC_batches[LDPC_cfg].num_TBs;
        int CBs_in_batch = kernel_params[first_TB_idx].num_CBs; //all TBs have same # CBs
        //NVLOGC_FMT(NVLOG_PDSCH, "LDPC kernel batch {}: first_TB_idx {}, TBs_in_batch {}, CBs_in_batch {}", LDPC_cfg, first_TB_idx, TBs_in_batch, CBs_in_batch);

        if(LDPC_cfg == 0)
        {
            ldpc_streams[0]      = cuda_strm;
            ldpc_stream_elements = 1;
        }
        else {
             // TODO Potentially modify to try out a different number of CUDA streams for LDPC.
    #if AVOID_STREAM_CREATION_IN_SETUP == 0
             if (LDPC_cfg > created_ldpc_streams) {
                 CUDA_CHECK(cudaStreamCreateWithFlags(&ldpc_streams[LDPC_cfg], cudaStreamNonBlocking));
                 created_ldpc_streams += 1;
             }
    #else
             ldpc_streams[LDPC_cfg] = ldpc_stream_pool.current_stream().handle();
             ldpc_stream_pool.advance(); // Necessary to ensure calls to current_stream() get different CUDA streams
    #endif
             ldpc_stream_elements += 1;
        }

        //Populate input/output addresses
        int max_rv = 0;
        for (int cnt = 0; cnt < TBs_in_batch; cnt++) {
            LDPC_input_addr[cumulative_TBs + cnt] = d_code_blocks.get() + CRC_dst_offset[LDPC_batches[LDPC_cfg].TB_idxs[cnt]];
            LDPC_output_addr[cumulative_TBs + cnt] = (uint8_t*)d_ldpc_workspace.get() + LDPC_dst_offset[LDPC_batches[LDPC_cfg].TB_idxs[cnt]];
            /* NVLOGC_FMT(NVLOG_PDSCH, "LDPC kernel batch {}, global TB idx {}, parity nodes {}",
                       LDPC_cfg, LDPC_batches[LDPC_cfg].TB_idxs[cnt],
                       max_ldpc_parity_nodes[LDPC_batches[LDPC_cfg].TB_idxs[cnt]]);*/
            max_rv = std::max(max_rv, (int) kernel_params[LDPC_batches[LDPC_cfg].TB_idxs[cnt]].rv);
        }

        int per_TB_ldpc_in_dims[2]             = {round_up_to_next((int)kernel_params[first_TB_idx].K, BITS_PER_U32), CBs_in_batch /** TBs_in_batch*/};
        int per_TB_ldpc_out_dims[2]            = {round_up_to_next((int)kernel_params[first_TB_idx].N, BITS_PER_U32), CBs_in_batch /** TBs_in_batch*/};
        single_TB_d_ldpc_in_tensor_desc[first_TB_idx].set(CUPHY_BIT, per_TB_ldpc_in_dims);
        single_TB_d_ldpc_out_tensor_desc[first_TB_idx].set(CUPHY_BIT, per_TB_ldpc_out_dims);

        if (LDPC_cfg== 0) { //FIXME needed?
            int N_max = div_round_up<uint32_t>(kernel_params[first_TB_idx].N, BITS_PER_U32) * BITS_PER_U32;
            d_ldpc_out_tensor_ref.desc().set(CUPHY_BIT,
                                             N_max,
                                             CBs_in_batch * TBs_in_batch,
                                             cuphy::tensor_flags::align_tight);
            d_ldpc_out_tensor_ref.set_addr(d_ldpc_workspace.get());
        }

        //Call setup for this batch
        cuphyStatus_t status = cuphySetupLDPCEncode(
                                                    &m_ldpc_encode_launch_cfg.get()[LDPC_cfg],
                                                    single_TB_d_ldpc_in_tensor_desc[first_TB_idx].handle(),
                                                    d_code_blocks.get(),
                                                    single_TB_d_ldpc_out_tensor_desc[first_TB_idx].handle(),
                                                    d_ldpc_workspace.get(),
                                                    kernel_params[first_TB_idx].bg,
                                                    kernel_params[first_TB_idx].Zc,
                                                    puncture_bits,
                                                    max_ldpc_parity_nodes[first_TB_idx], /* the parity nodes for all TBs in an LDPC cfg was updated in prepareRateMatchingMultipleCells()*/
                                                    max_rv, //kernel_params[first_TB_idx].rv, //currently set to non-zero if at least one TB in batch has non zero rv
                                                    1, /* batching enabled */
                                                    TBs_in_batch,
                                                    LDPC_input_addr.data() + cumulative_TBs,
                                                    LDPC_output_addr.data() + cumulative_TBs,
                                                    h_ldpc_w_ptr + cumulative_TBs * ldpc_workspace_offset,
                                                    d_ldpc_w_ptr + cumulative_TBs * ldpc_workspace_offset,
                                                    static_cast<void*>(m_component_descrs.getCpuStartAddrs()[PDSCH_LDPC] + LDPC_cfg * ldpc_descr_offset), //CPU desc
                                                    static_cast<void*>(m_component_descrs.getGpuStartAddrs()[PDSCH_LDPC] + LDPC_cfg * ldpc_descr_offset), //GPU desc
                                                    (!bulk_desc_async_copy),
                                                    ldpc_streams[LDPC_cfg]);

        if(status != CUPHY_STATUS_SUCCESS)
        {
            NVLOGE_FMT(NVLOG_PDSCH, AERIAL_CUPHY_EVENT, "Invalid argument(s) for cuphySetupLDPCEncode; {}", cuphyGetErrorString(status));
            return CUPHY_STATUS_INVALID_ARGUMENT;
        }

        cumulative_TBs += TBs_in_batch;
    }
    return CUPHY_STATUS_SUCCESS;

}


int PdschTx::refCheckLDPCMultipleCells(cudaStream_t cuda_strm)
{
    int total_error_cnt = 0;

    for (int cell = 0; cell < num_cells; cell++) {

        if(per_cell_input_file[cell] == nullptr)
        {
            NVLOGE_FMT(NVLOG_PDSCH, AERIAL_CUPHY_EVENT, "Cannot do a reference check without a valid input file!");
            return -1;
        }

        uint32_t error_cnt = 0;
        const bool cell_not_in_testing_mode = (dynamic_params->pCellGrpDynPrm->pCellPrms[cell].testModel == 0);

        for(int cell_TB_id = 0; ((cell_TB_id < per_cell_num_TBs[cell]) && cell_not_in_testing_mode); cell_TB_id++)
        {
            int num_CBs = 0;
            int N = 0;

            const std::string         ref_dataset_name = "tb" + std::to_string(cell_TB_id) + "_codedcbs";
            // performs 2 cudaHostAlloc calls one from typed_tensor call and one from read_HDF5_dataset call
            auto ldpc_ref_output  = typed_tensor_from_dataset<CUPHY_R_8U, pinned_alloc>((*per_cell_input_file[cell]).open_dataset(ref_dataset_name.c_str()),  cuphy::tensor_flags::align_default, cuda_strm);

            // Separate copies per TB needed for easier indexing. d_ldpc_out_tensor is overprovisioned for max N and max CBs.

            // In batching mode, after runLDPCBatching cuda_strm will wait for all LDPC batched kernels to complete.
            // So using cuda_strm here suffices.
            typed_tensor<CUPHY_BIT, pinned_alloc> single_TB_h_ldpc_out_tensor = getHostOutputLDPCTBPerCell(cell, cell_TB_id, &num_CBs, &N, cuda_strm);
            CUDA_CHECK(cudaStreamSynchronize(cuda_strm)); //superfluous after adding stream synchronization in getHostOutputLDPCTBPerCell

            const uint32_t N_mask  = (1L << (N % BITS_PER_U32)) - 1; // Mask to apply if N not evenly divisible by BITS_PER_U32

            for(int CB = 0; CB < num_CBs; CB += 1)
            {
                for(int element_start = 0; element_start < N; element_start += BITS_PER_U32)
                {
                    uint32_t ref_bits = 0;
                    for(int offset = 0; offset < BITS_PER_U32; offset++)
                    {
                        if(element_start + offset < N)
                        {
                            // Note some bits in reference input are -1. Treat them as 0s.
                            uint32_t bit = ldpc_ref_output(element_start + offset, CB);
                            ref_bits |= (bit << offset);
                        }
                    }
                    uint32_t GPU_bits = single_TB_h_ldpc_out_tensor(element_start / BITS_PER_U32, CB);
                    if(element_start + BITS_PER_U32 > N)
                    {
                        GPU_bits = GPU_bits & N_mask;
                    }
                    if(ref_bits != GPU_bits)
                    {
                        error_cnt += 1;
                        // Leaving log entry uncommented for dbg purposes for now.
                        NVLOGC_FMT(NVLOG_PDSCH, "LDPC mismatch for cell {} cell TB {} CB {}, element id {}: Expected reference {:#x} and got {:#x}",
                               cell, cell_TB_id,
                               CB,
                               element_start / BITS_PER_U32,
                               ref_bits,
                               GPU_bits);

                    }
                }
            }
        }

        total_error_cnt += error_cnt;

        NVLOGC_FMT(NVLOG_PDSCH, "");
        NVLOGC_FMT(NVLOG_PDSCH, "LDPC Error Count: {}; GPU output compared w/ reference dataset <tb*_codedcbs> from <{}>",
                   error_cnt, per_cell_hdf5_filename[cell]);
    }

    return total_error_cnt;
}


typed_tensor<CUPHY_BIT, pinned_alloc> PdschTx::getHostOutputLDPCTBPerCell(uint16_t cell, int cell_TB_id, int* updated_num_CBs, int* updated_N, cudaStream_t cuda_strm)
{
    if(!ran_pipeline)
    {
        throw std::runtime_error("Cannot get LDPC's output without calling Run first.");
    }

    if (!inter_cell_batching_mode)
    {
        throw std::runtime_error("Please call with inter-cell batching mode enabled!");
    }

    if (cell_TB_id >= per_cell_num_TBs[cell])
    {
        throw std::runtime_error("Invalid TB index for this cell!");
    }


    int TB_id = per_cell_TB_params_offset[cell] + cell_TB_id;
    int N = kernel_params[TB_id].N;

    if((max_ldpc_parity_nodes[TB_id] != 0) && (kernel_params[TB_id].rv == 0))
    {
       // Need to compute updated N.
        int Zc = kernel_params[TB_id].Zc;
        int Kb = get_Kb(TB_sizes[TB_id], kernel_params[TB_id].bg);
        N      = (Kb + max_ldpc_parity_nodes[TB_id] - CUPHY_LDPC_NUM_PUNCTURED_NODES) * Zc; // Assumes LDPC is also called with the punctured_bits set option
    }

    int num_CBs = kernel_params[TB_id].num_CBs;
    *updated_num_CBs = num_CBs;
    *updated_N = N;

    // Separate copies per TB needed for easier indexing. d_ldpc_out_tensor is overprovisioned for max N and max CBs.
    int temp_TB_id = TB_id;

    int per_TB_ldpc_out_dims[2]            = {round_up_to_next((int)kernel_params[TB_id].N, BITS_PER_U32), (int)kernel_params[TB_id].num_CBs};
    cuphy::tensor_desc tmp_tensor_desc = cuphy::tensor_desc(CUPHY_BIT, per_TB_ldpc_out_dims);

    tensor_device temp_tensor = tensor_device((uint8_t*)d_ldpc_out_tensor_ref.addr() + LDPC_dst_offset[temp_TB_id],
                                         CUPHY_BIT,
                                         per_TB_ldpc_out_dims[0],
                                         per_TB_ldpc_out_dims[1],
                                         cuphy::tensor_flags::align_tight);
    CUDA_CHECK(cudaStreamSynchronize(cuda_strm));

    typed_tensor<CUPHY_BIT, pinned_alloc> single_TB_h_ldpc_out_tensor(temp_tensor.layout());
    CUPHY_CHECK(cuphyConvertTensor(single_TB_h_ldpc_out_tensor.desc().handle(),
                                   single_TB_h_ldpc_out_tensor.addr(),
                                   temp_tensor.desc().handle(),
                                   temp_tensor.addr(),
                                   cuda_strm));

    CUDA_CHECK(cudaStreamSynchronize(cuda_strm));

    return single_TB_h_ldpc_out_tensor;
}


typed_tensor<CUPHY_BIT, pinned_alloc> PdschTx::getHostOutputLDPCMultipleCells(uint16_t cell, cudaStream_t cuda_strm)
{
    if(!ran_pipeline)
    {
        throw std::runtime_error("Cannot get LDPC's output without calling Run first.");
    }

    typed_tensor<CUPHY_BIT, pinned_alloc> h_LDPC_out_tensor(d_per_cell_ldpc_out_tensor_ref[cell].desc().get_info().layout());
    h_LDPC_out_tensor.convert(d_per_cell_ldpc_out_tensor_ref[cell], cuda_strm);

    CUDA_CHECK(cudaStreamSynchronize(cuda_strm));
    return h_LDPC_out_tensor;
}

cuphy::tensor_ref const& PdschTx::getGPUOutputLDPC()
{
    return d_ldpc_out_tensor_ref;
}

void PdschTx::runLDPCBatching(cudaStream_t cuda_strm, bool ref_check)
{
    for (int LDPC_cfg = 0; LDPC_cfg < total_LDPC_kernel_configs; LDPC_cfg++) {
        // Wait for CRC kernel to complete
        CUDA_CHECK(cudaStreamWaitEvent(ldpc_streams[LDPC_cfg], crc_event.handle(), 0));

        // Run the LDPC encoder
        CUresult encode_status =
                                 launch_kernel(m_ldpc_encode_launch_cfg.get()[LDPC_cfg].m_kernelNodeParams, ldpc_streams[LDPC_cfg]);

        if(encode_status != CUDA_SUCCESS)
        {
            throw std::runtime_error("LDPC kernel failure!");
        }

        //Have components that follow wait for all LDPC batches to complete
        if(LDPC_cfg != 0)
        {
            CUDA_CHECK(cudaEventRecord(ldpc_complete_events[LDPC_cfg], ldpc_streams[LDPC_cfg]));
            CUDA_CHECK(cudaStreamWaitEvent(ldpc_streams[0], ldpc_complete_events[LDPC_cfg], 0));
        }
    }
}

std::vector<uint32_t> PdschTx::getHostOutputRateMatching(cudaStream_t cuda_strm)
{
    if(!ran_pipeline)
    {
        throw std::runtime_error("Cannot get Rate Matching's output without calling Run first.");
    }

    //Note Rate Matching's output requires some reordering! Some of the copied memory, that is not used, might not be initialized.
    std::vector<uint32_t> h_rate_matching_output(rm_output_elements);
    if(aas_mode)
    {
        CUDA_CHECK(cudaMemcpyAsync(h_rate_matching_output.data(), d_rate_matching_output.addr(), rm_output_elements * sizeof(uint32_t), cudaMemcpyDeviceToHost, cuda_strm));
    }
    else
    {
        NVLOGD_FMT(NVLOG_PDSCH, "Rate Matching Output is empty in non AAS mode");
    }
    return h_rate_matching_output;
}

const uint32_t* PdschTx::getGPUOutputRateMatching()
{
    return (aas_mode) ? (uint32_t*)d_rate_matching_output.addr() :  nullptr;
}

int PdschTx::refCheckRateMatchingMultipleCells(cudaStream_t cuda_strm)
{
    if (layer_mapping)
    {
        NVLOGE_FMT(NVLOG_PDSCH, AERIAL_CUPHY_EVENT, "AAS mode is incompatible with layer mapping!");
        return -1;
    }

    int total_error_cnt = 0;
    if (num_cells != 1) {
        NVLOGE_FMT(NVLOG_PDSCH, AERIAL_CUPHY_EVENT, "AAS mode is currently not supported for multi-cell cell groups!");
        return -1;
    }


    uint32_t* Er = h_rm_workspace + 2*num_TBs;
    std::string dataset = scrambling ? "_scramcbs" : "_ratematcbs";

    for (int cell = 0; cell < num_cells; cell++)
    {
        if(per_cell_input_file[cell] == nullptr)
        {
            NVLOGE_FMT(NVLOG_PDSCH, AERIAL_CUPHY_EVENT, "Cannot do a reference check without a valid input file!");
            return -1;
        }

        // Get output of rate matching (with or without scrambling)
        std::vector<uint32_t> h_rate_matching_output = getHostOutputRateMatching(cuda_strm);
        CUDA_CHECK(cudaMemcpyAsync(h_ue_grp_params, d_ue_grp_params,  dynamic_params->pCellGrpDynPrm->nUeGrps * sizeof(PdschUeGrpParams), cudaMemcpyDeviceToHost, cuda_strm));
        CUDA_CHECK(cudaStreamSynchronize(cuda_strm));

        unsigned long long error_cnt = 0;
        uint32_t total_ref_bits = 0;

        for(int cell_TB_id = 0; cell_TB_id < per_cell_num_TBs[cell]; cell_TB_id++)
        {

            int TB_id = per_cell_TB_params_offset[cell] + cell_TB_id;

            // Special handling needed as the Er array is populated with the max_G and does not account for punctured REs. Compute and fill in Er values in place.
            PdschUeGrpParams tmp_ue_grp_params      = h_ue_grp_params[h_dmrs_params[TB_id].ueGrp_idx];
            int csi_rs_RE_cnt = tmp_ue_grp_params.cumulative_skipped_REs[h_dmrs_params[TB_id].num_data_symbols - 1];
            int G = (kernel_params[TB_id].max_REs - csi_rs_RE_cnt) * kernel_params[TB_id].Nl * kernel_params[TB_id].Qm;
            int modulo_C = (kernel_params[TB_id].max_REs - csi_rs_RE_cnt) %  kernel_params[TB_id].num_CBs; //split, if any, at C - modulo_C CB
            Er[TB_id * 2] = (kernel_params[TB_id].testModel == 0) ?  kernel_params[TB_id].num_CBs - modulo_C : kernel_params[TB_id].num_CBs;
            Er[TB_id * 2 + 1] =  (kernel_params[TB_id].testModel == 0) ? kernel_params[TB_id].Nl *  kernel_params[TB_id].Qm* floorf(G * 1.0f / (kernel_params[TB_id].Nl *  kernel_params[TB_id].Qm * kernel_params[TB_id].num_CBs)) : kernel_params[TB_id].N;


            int ref_bit       = 0;
            std::string         ref_dataset_name = "tb" + std::to_string(cell_TB_id) + dataset;
            auto dataset_info = get_HDF5_dataset_info((*per_cell_input_file[cell]).open_dataset(ref_dataset_name.c_str()));
            auto ref_data         = typed_tensor_from_dataset<CUPHY_R_8U, pinned_alloc>((*per_cell_input_file[cell]).open_dataset(ref_dataset_name.c_str()));
            bool not_in_test_mode = (kernel_params[TB_id].testModel == 0);
            int check_CBs = kernel_params[TB_id].num_CBs;
            for(int CB = 0; CB < check_CBs; CB++)
            {
                uint32_t Er_CB = Er[TB_id * 2 + 1] + ((CB < Er[TB_id * 2]) ? 0 : kernel_params[TB_id].Nl * kernel_params[TB_id].Qm);
                if ((kernel_params[TB_id].testModel != 0) && (CB == kernel_params[TB_id].num_CBs-1)) {
                    Er_CB = kernel_params[TB_id].G - CB*kernel_params[TB_id].N; // Should not use tbSize*8 instead of G as TB size in bits may not be evenly divisible by 8
                }
                //NVLOGD_FMT(NVLOG_PDSCH, "TB {}, CB {}, Er_CB {}, cell_max_CBs {}, cell_group_Emax {}", cell_TB_id, CB, Er_CB, cell_max_CBs, cell_group_Emax);
                for(int Er_bit = 0; Er_bit < Er_CB;  Er_bit++)
                {
                    uint32_t ref_value = ref_data(ref_bit, 0);

                    int out_index = (TB_id * cell_max_CBs  + CB) * cell_group_Emax + Er_bit;

                    int      out_word       = out_index / BITS_PER_U32;
                    int      out_bits       = out_index % BITS_PER_U32;
                    uint32_t computed_value = (h_rate_matching_output[out_word] >> out_bits) & 0x1;
                    if(ref_value != computed_value)
                    {
                        error_cnt += 1;
                        /*NVLOGD_FMT(NVLOG_PDSCH, "GPU vs. reference output mismatch!");
                        NVLOGD_FMT(NVLOG_PDSCH, "TB {}, CB {}, Er_bit {}: computed_value {} vs. reference {}",
                                   TB_id, CB, Er_bit, computed_value, ref_value);*/
                    }
                    ref_bit += 1;
                    total_ref_bits += 1;
                }
            }
        }

        NVLOGC_FMT(NVLOG_PDSCH, "");
        NVLOGC_FMT(NVLOG_PDSCH, "Rate Matching Error Count: {} out of {}; GPU output compared w/ reference dataset <tb*{}> from <{}>",
                                error_cnt, total_ref_bits, dataset, per_cell_hdf5_filename[cell]);

        total_error_cnt += error_cnt;
    }
    return total_error_cnt;
}

cuphyStatus_t PdschTx::prepareRateMatchingMultipleCells(cudaStream_t cuda_strm)
{
    int kernel_launches = 1;
    int cell = 0;

    LDPC_N_max = 0;
    for (int TB_id = 0; TB_id < num_TBs; TB_id++)
    {
        LDPC_N_max = std::max(LDPC_N_max, kernel_params[TB_id].N);
    }
    int N_max = div_round_up<uint32_t>(LDPC_N_max, BITS_PER_U32) * BITS_PER_U32;

    uint32_t* d_per_cell_ldpc_workspace = d_ldpc_workspace.get();

    // Only used in if !inter_cell_batching_mode
    d_per_cell_ldpc_out_tensor_ref[0].desc().set(CUPHY_BIT,
                                                 N_max,
                                                 per_cell_max_CBs[0],
                                                 per_cell_num_TBs[0],
                                                 cuphy::tensor_flags::align_tight);
    d_per_cell_ldpc_out_tensor_ref[0].set_addr(d_per_cell_ldpc_workspace);


    uint32_t rm_workspace_offset = per_cell_TB_params_offset[0] * (rm_allocated_workspace_size/per_cell_group_max_TBs) / sizeof(uint32_t);
    cuphyStatus_t status = cuphySetupDlRateMatching(m_rate_matching_launch_cfg.get(),
                                                    dynamic_params->pStatusInfo,
                                                    d_ldpc_workspace.get(),
                                                    (aas_mode) ? (uint32_t*)d_rate_matching_output.addr() : nullptr,
                                                    nullptr,
                                                    dynamic_params->pDataOut->pTDataTx[0].pAddr, //tx_tensor.addr() prev. TODO remove
                                                    d_xtf_re_maps_v2.get(), // Pointer to first cell's RE map. The rest are accessed with fixed offset from there
                                                                   // using d_dmrs_params.cell_index_in_cell_group field
                                                    max_PRB_BWP,
                                                    num_TBs,
                                                    0, //per_cell_total_num_layers[cell], //FIXME - I don't think it's used
                                                    scrambling,
                                                    layer_mapping,
                                                    true, // use single fused modulation kernel
                                                    dynamic_params->pCellGrpDynPrm->nPrecodingMatrices > 0 ? 1 : 0,
                                                    false,
                                                    true, //inter_cell_batching_mode
                                                    h_rm_workspace + rm_workspace_offset, //m_component_descrs.getCpuStartAddrs()[PDSCH_RM_WORKSPACE], CPU workspace desc
                                                    d_rm_workspace + rm_workspace_offset, //m_component_descrs.getGpuStartAddrs()[PDSCH_RM_WORKSPACE], GPU workspace desc
                                                    kernel_params + per_cell_TB_params_offset[0],  //m_component_descrs.getCpuStartAddrs()[PDSCH_PER_TB_PARAMS]
                                                    d_tbPrmsArray + per_cell_TB_params_offset[0],  //m_component_descrs.getGpuStartAddrs()[PDSCH_PER_TB_PARAMS]
                                                    d_dmrs_params + per_cell_TB_params_offset[0],
                                                    d_ue_grp_params, //FIXME provide offset if no batching
                                                    static_cast<void*>(m_component_descrs.getCpuStartAddrs()[PDSCH_RM]), //CPU desc
                                                    static_cast<void*>(m_component_descrs.getGpuStartAddrs()[PDSCH_RM]), //GPU desc
                                                    (!bulk_desc_async_copy),
                                                    cuda_strm);

    if(status != CUPHY_STATUS_SUCCESS)
    {
        return status;
    }

    uint32_t* Er = h_rm_workspace + rm_workspace_offset + 2 * num_TBs; //FIXME at this point Er is computed with max_G in mind

    // Outer loop is LDPC cfgs
    int num_cfgs = total_LDPC_kernel_configs;
    for (int cfg = 0; cfg < num_cfgs; cfg++) {
        uint32_t per_cell_Emax        = 0;
        int TBs_in_cfg = LDPC_batches[cfg].num_TBs;
        uint32_t max_F = 0;
        for (int cnt = 0; cnt < TBs_in_cfg; cnt++) {
            int i = LDPC_batches[cfg].TB_idxs[cnt];
           //FIXME this was already computed in setup, could just copy back that value directly
            uint32_t Er_CB = Er[i * 2 + 1] + (((kernel_params[per_cell_TB_params_offset[cell] + i].num_CBs - 1) < Er[i * 2]) ? 0 : kernel_params[per_cell_TB_params_offset[cell] + i].Nl * kernel_params[per_cell_TB_params_offset[cell] + i].Qm);
            per_cell_Emax           = std::max(per_cell_Emax, Er_CB);
            max_F                   = std::max(max_F, kernel_params[i].F); // it's possible that different TBs are batched that have identical BG, Zc but different filler bits. In this case, the one twith the largest number of filler bits should influence the max number of parity nodes computed
        }
        per_cell_Emax = round_up_to_next((int) per_cell_Emax, BITS_PER_U32);
        cell_group_Emax = std::max(cell_group_Emax, (int)per_cell_Emax);

        for (int cnt = 0; cnt < TBs_in_cfg; cnt++) {
            int i = LDPC_batches[cfg].TB_idxs[cnt];
            i += per_cell_TB_params_offset[0];
            /* NB The TB_idxs, when inter-cell batching is enabled, are with respect to all TBs in the cell group.
               It's OK to add per_cell_TB_params_offset[cell] in that case too, as cell variable in that mode is at most 0
               and the respective offset is 0 too. The cell is a bit of a misnomer in that scenario. */

            // NB: Needs to be computed before calling prepareLDPC
            //max_ldpc_parity_nodes[i] = ceil((per_cell_Emax - kernel_params[i].K + kernel_params[i].F + 2 * kernel_params[i].Zc) * 1.0f / kernel_params[i].Zc);
            max_ldpc_parity_nodes[i] = ceil((per_cell_Emax - kernel_params[i].K + max_F + 2 * kernel_params[i].Zc) * 1.0f / kernel_params[i].Zc);
            //NVLOGD_FMT(NVLOG_PDSCH, "Emax {}, K {}, F {} but max_F {}, Zc {}, ldpc parity nodes {}", per_cell_Emax, kernel_params[i].K, kernel_params[i].F, max_F, kernel_params[i].Zc, max_ldpc_parity_nodes[i]);

            //For BG1 parity nodes should be in [4, 46], for BG2 in [4, 42]
            //Cap number of parity nodes to respective max, if needed.
            uint32_t max_parity_nodes = (kernel_params[i].bg == 1) ? CUPHY_LDPC_MAX_BG1_PARITY_NODES : CUPHY_LDPC_MAX_BG2_PARITY_NODES;
            max_ldpc_parity_nodes[i] = std::clamp(max_ldpc_parity_nodes[i], static_cast<uint32_t>(CUPHY_LDPC_MIN_PARITY_NODES), max_parity_nodes);
            //max_ldpc_parity_nodes[i] = 0; // Uncomment to compute all parity nodes.
        }

        if(aas_mode && (per_cell_Emax > max_Emax))
        {
            NVLOGE_FMT(NVLOG_PDSCH, AERIAL_CUPHY_EVENT, "Cell group Emax {} but supported max Emax is {}", per_cell_Emax, max_Emax);
            //throw std::runtime_error("Emax exceeds max supported! Update PdschTx::setMaxValues.");
            return CUPHY_STATUS_UNSUPPORTED_CONFIG;
        }

        /*if(per_cell_total_num_layers[0] > max_layers)
        {
            NVLOGE_FMT(NVLOG_PDSCH, AERIAL_CUPHY_EVENT, "Cell group has total_num_layers {} but max is {}", per_cell_total_num_layers[0], max_layers);
            throw std::runtime_error("total layers exceed max supported! Update PdschTx::setMaxValues.");
        }*/
    }
    // Used in refCheckRateMatchingMultipleCells() for AAS mode and in getHostOutputRateMatching()
    if (has_cells_in_TM) cell_group_Emax = MAX_ENCODED_CODE_BLOCK_BIT_SIZE; // NB: if all TBs are for cells in TM, then cell_group_Emax without this update, would have been 0.
    rm_output_elements =  div_round_up<uint32_t>(num_TBs * cell_max_CBs * cell_group_Emax, BITS_PER_U32); // set before getHostOutputRateMatching is called

    return CUPHY_STATUS_SUCCESS;
}


void PdschTx::runRateMatchingMultipleCells(cudaStream_t cuda_strm, bool ref_check)
{
    /*NVLOGC_FMT(NVLOG_PDSCH, "grid {{{}, {}, {}}}, block {{{}, {}, {}}}, sharedMemBytes {}",
                                m_rate_matching_launch_cfg.get()->m_kernelNodeParams[0].gridDimX,
                                m_rate_matching_launch_cfg.get()->m_kernelNodeParams[0].gridDimY,
                                m_rate_matching_launch_cfg.get()->m_kernelNodeParams[0].gridDimZ,
                                m_rate_matching_launch_cfg.get()->m_kernelNodeParams[0].blockDimX,
                                m_rate_matching_launch_cfg.get()->m_kernelNodeParams[0].blockDimY,
                                m_rate_matching_launch_cfg.get()->m_kernelNodeParams[0].blockDimZ,
                                m_rate_matching_launch_cfg.get()->m_kernelNodeParams[0].sharedMemBytes);*/
    CU_CHECK_EXCEPTION(launch_kernel(m_rate_matching_launch_cfg.get()->m_kernelNodeParams[0], cuda_strm));
#if 0
    CUresult rm_status = launch_kernel(m_rate_matching_launch_cfg.get()->m_kernelNodeParams[0], cuda_strm);
    if(rm_status != CUDA_SUCCESS)
    {
        throw std::runtime_error("Invalid argument(s) for Rate Matching");
    }
#endif

    if(ref_check)
    {
        if(aas_mode)
            failed_ref_checks += (refCheckRateMatchingMultipleCells(cuda_strm) != 0) ? 1 : 0;
        else
        {
            failed_ref_checks += (refCheckModulationMultipleCells(cuda_strm) != 0) ? 1 : 0;
        }
    }
}


int PdschTx::refCheckModulationMultipleCells(cudaStream_t cuda_strm)
{
    if (dynamic_params->pCellGrpDynPrm->nPrecodingMatrices > 0)
    {
        // Cannot use the tb*_qams dataset for comparison as that dataset is before precoding. With precoding
        // a single {RE, symbol, antenna_port} element can have contributions from multiple TBs.
        // Will perform the ref. check for the entire output tensor in the next step with refCheckTxDataMultipleCells().
        NVLOGC_FMT(NVLOG_PDSCH, "Fused Rate Matching and Modulation Mapper: No comparison because at least one UE has precoding enabled.");
        return 0;
    }

    int total_error_cnt = 0;

    for (int cell = 0; cell < num_cells; cell++) {
        if(per_cell_input_file[cell] == nullptr)
        {
            NVLOGE_FMT(NVLOG_PDSCH, AERIAL_CUPHY_EVENT, "Cannot do a reference check without a valid input file!");
            return -1;
        }

        // Also copy: xtf_re_map
        bool non_zero_csirs_parameters = (dynamic_params->pCellGrpDynPrm[0].nCsiRsPrms != 0);
        if (non_zero_csirs_parameters) {
            CUDA_CHECK(cudaMemcpyAsync(h_xtf_re_maps_v2.get(), d_xtf_re_maps_v2.get(), num_cells * per_cell_xtf_re_map_elements * sizeof(uint16_t), cudaMemcpyDeviceToHost, cuda_strm));
        }

        // All UE group parameters
        CUDA_CHECK(cudaMemcpyAsync(h_ue_grp_params, d_ue_grp_params,  dynamic_params->pCellGrpDynPrm->nUeGrps * sizeof(PdschUeGrpParams), cudaMemcpyDeviceToHost, cuda_strm));

        using tensor_pinned_C_64F               = typed_tensor<CUPHY_C_64F, pinned_alloc>;
        std::string         ref_dataset_name    = "Xtf";
        tensor_pinned_C_64F pdsch_tx_ref_output = typed_tensor_from_dataset<CUPHY_C_64F, pinned_alloc>((*per_cell_input_file[cell]).open_dataset(ref_dataset_name.c_str()));

        #if 0 // In multi-cell test benches it's possible tensor allocation is max and so can't rely on tensor descriptor dims.
        int num_dims;
        int dims[3];
        CUPHY_CHECK(cuphyGetTensorDescriptor(dynamic_params->pDataOut->pTDataTx[cell].desc, 3, nullptr, &num_dims, dims, nullptr));

        tensor_device cell_data_tx_tensor = tensor_device(dynamic_params->pDataOut->pTDataTx[cell].pAddr,
                                                 CUPHY_C_16F,
                                                 dims[0],
                                                 dims[1],
                                                 dims[2],
                                                 cuphy::tensor_flags::align_tight);
        #else
        tensor_device cell_data_tx_tensor = tensor_device(dynamic_params->pDataOut->pTDataTx[cell].pAddr,
                                                 CUPHY_C_16F,
                                                 pdsch_tx_ref_output.layout().dimensions()[0],
                                                 pdsch_tx_ref_output.layout().dimensions()[1],
                                                 pdsch_tx_ref_output.layout().dimensions()[2],
                                                 cuphy::tensor_flags::align_tight);
        #endif


        typed_tensor<CUPHY_C_16F, pinned_alloc> h_pdsch_out_tensor(cell_data_tx_tensor.layout());
        h_pdsch_out_tensor.convert(cell_data_tx_tensor, cuda_strm);
        CUDA_CHECK(cudaStreamSynchronize(cuda_strm));

        uint16_t static_cell = dynamic_params->pCellGrpDynPrm->pCellPrms[cell].cellPrmStatIdx;
        int      num_BWP_PRBs    = static_params->pCellStatPrms[static_cell].nPrbDlBwp; // cell vs. static_cell
        using tensor_pinned_C_64F = typed_tensor<CUPHY_C_64F, pinned_alloc>;

        uint32_t gpu_mismatch    = 0;
        uint32_t symbols_checked = 0;

        for(int cell_TB_id = 0; cell_TB_id < per_cell_num_TBs[cell]; cell_TB_id++)
        {
            int TB_id = per_cell_TB_params_offset[cell] + cell_TB_id;

            int TB_num_layers     = kernel_params[TB_id].Nl;
            int modulation_order  = kernel_params[TB_id].Qm;
#if 0
            // G not valid so use ref_QAM_elements as qam_elements for now // FIXME
            int rate_matched_bits = kernel_params[TB_id].G; //FIXME ensure it's valid or G was updated at some point
            int qam_elements      = ceil(rate_matched_bits * 1.0f / modulation_order);
#endif

            std::string         modulation_dataset_name = "tb" + std::to_string(cell_TB_id) + "_qams";
            tensor_pinned_C_64F output_data             = typed_tensor_from_dataset<CUPHY_C_64F, pinned_alloc>((*per_cell_input_file[cell]).open_dataset(modulation_dataset_name.c_str()));
            const int           ref_qam_elements        = output_data.layout().dimensions()[0];
            int qam_elements      = ref_qam_elements;

            PdschDmrsParams tmp_h_dmrs_params       = h_dmrs_params[TB_id];
            PdschUeGrpParams tmp_ue_grp_params      = h_ue_grp_params[h_dmrs_params[TB_id].ueGrp_idx];
            int             start_freq_idx          = CUPHY_N_TONES_PER_PRB * tmp_h_dmrs_params.start_Rb;
            int             cutoff_freq_idx         = start_freq_idx + (CUPHY_N_TONES_PER_PRB * tmp_h_dmrs_params.num_Rbs);

            // Find first and last RB for Resource allocation Type 0 and set start and stop bounds accordingly
            if(tmp_h_dmrs_params.resourceAlloc == 0) {
                int rb_idx = 0;
                uint32_t rbBitmap_dword = tmp_h_dmrs_params.rbBitmap[rb_idx];

                while((0 == rbBitmap_dword) && (rb_idx < MAX_RBMASK_UINT32_ELEMENTS)) {
                    ++rb_idx;
                    // Extra check added to ensure there is no out of bounds read with rb_idx from rbBitmap below.
                    // This would only be possible if rbBitmask[0] to [8] inclusive was all 0s, which would be an invalid config.
                    if (rb_idx == MAX_RBMASK_UINT32_ELEMENTS) {
                        NVLOGE_FMT(NVLOG_PDSCH, AERIAL_CUPHY_EVENT, "refCheckModulationMultipleCells error: rbBitmap cannot have all 0s in resource allocation type 0!");
                        return -1;
                    }
                    rbBitmap_dword = tmp_h_dmrs_params.rbBitmap[rb_idx];
                }
                rb_idx = rb_idx << 5;
                start_freq_idx = (rb_idx + ffs(rbBitmap_dword) - 1 + tmp_h_dmrs_params.BWP_start_PRB) * CUPHY_N_TONES_PER_PRB;
                rb_idx = MAX_RBMASK_UINT32_ELEMENTS - 1;
                rbBitmap_dword = tmp_h_dmrs_params.rbBitmap[rb_idx];
                while((0 == rbBitmap_dword) && (rb_idx >= start_freq_idx/(CUPHY_N_TONES_PER_PRB*BITS_PER_U32))) {
                    --rb_idx;
                    rbBitmap_dword = tmp_h_dmrs_params.rbBitmap[rb_idx];
                }
                int last_bits = 0;
                if(rbBitmap_dword) last_bits = __builtin_clz(rbBitmap_dword);
                cutoff_freq_idx = (((rb_idx + 1) << 5) - last_bits + tmp_h_dmrs_params.BWP_start_PRB) * CUPHY_N_TONES_PER_PRB;
            }

            int             freq_idx                = start_freq_idx;
            int             per_layer_symbol_offset = 0;

            uint64_t allocated_data_or_dmrs_w_data_symbol_loc = 0ULL;
            int allocated_data_or_dmrs_w_data_symbols = tmp_h_dmrs_params.num_data_symbols;
            uint32_t dmrs_bitmask = 0;
            if (tmp_h_dmrs_params.dmrsCdmGrpsNoData1) {
                int data_index = 0, dmrs_index = 0, index = 0;
                for (uint64_t i = 0; i < OFDM_SYMBOLS_PER_SLOT; i++) {
                    if (((tmp_h_dmrs_params.data_sym_loc >> (4 * data_index)) & 0xF) == i) {
                        allocated_data_or_dmrs_w_data_symbol_loc |= (i << (4*index));
                        data_index += 1;
                        index += 1;
                    } else if (((tmp_h_dmrs_params.dmrs_sym_loc >> (4 * dmrs_index)) & 0xF) == i) {
                        allocated_data_or_dmrs_w_data_symbol_loc |= (i << (4*index));
                        dmrs_index += 1;
                        index += 1;
                        dmrs_bitmask |= (1 << i);
                    }
                }
                allocated_data_or_dmrs_w_data_symbols += tmp_h_dmrs_params.num_dmrs_symbols;
            } else {
                allocated_data_or_dmrs_w_data_symbol_loc = tmp_h_dmrs_params.data_sym_loc;
            }
            /*NVLOGD_FMT(NVLOG_PDSCH, "data symbol loc {:#x} for {:d} symbols, dmrs symbol loc {:#x} for {:d} symbols, data_or_dmrs {:#x} for {:d} symbols",
                    tmp_h_dmrs_params.data_sym_loc, tmp_h_dmrs_params.num_data_symbols,
                    tmp_h_dmrs_params.dmrs_sym_loc, tmp_h_dmrs_params.num_dmrs_symbols,
                    allocated_data_or_dmrs_w_data_symbol_loc,
                    allocated_data_or_dmrs_w_data_symbols);*/

            int seen_dmrs_symbols = 0; // cumulative_skipped_REs indexing doesn't account for them

            // symbol_id here corresponds to a QAM element not an OFDM symbol
            for(int symbol_id = 0; symbol_id < qam_elements; symbol_id += 1)
            {
                __half2 ref_symbol;
                ref_symbol.x = (half)output_data(symbol_id).x;
                ref_symbol.y = (half)output_data(symbol_id).y;

                int layer_cnt = symbol_id / (qam_elements / TB_num_layers);
                int layer_id  = calculate_layer_id(tmp_h_dmrs_params.port_ids[layer_cnt], tmp_h_dmrs_params.n_scid, tmp_h_dmrs_params.nlAbove16);

                int symbol_pos = (allocated_data_or_dmrs_w_data_symbol_loc >> (4 * per_layer_symbol_offset)) & 0xF; // data or dmrs w/ data symbol
                bool symbol_is_dmrs = (dmrs_bitmask >> symbol_pos) & 0x1;

                if (symbol_is_dmrs && (freq_idx == start_freq_idx)) {
                    freq_idx += (((tmp_h_dmrs_params.port_ids[layer_cnt] >> 1) & 0x1) ^ 0x1U);
                }

                // freq offset is due to skipped REs for CSI-RS
                uint16_t* h_cell_xtf_re_map_ptr =  h_xtf_re_maps_v2.get() + cell * per_cell_xtf_re_map_elements;
                uint16_t freq_offset = non_zero_csirs_parameters ? h_cell_xtf_re_map_ptr[symbol_pos * num_BWP_PRBs * CUPHY_N_TONES_PER_PRB + freq_idx] : 0;
                __half2 gpu_symbol = h_pdsch_out_tensor(freq_idx + freq_offset, symbol_pos, layer_id);

                if(!complex_approx_equal<__half2, __half>(gpu_symbol, ref_symbol, pdsch_comp_tol))
                {
                    /*NVLOGD_FMT(NVLOG_PDSCH, "Error! Cell {:d} Overall TB_id {:d}: TB {:d}, Layer cnt {:d}, Mismatch for QAM symbol {:d} ({{{:d}, {:d}, {:d}}}) - expected={:f} + i {:f} vs. gpu={:f} + i {:f}",
                           cell, TB_id,  cell_TB_id, layer_cnt, symbol_id,
                           freq_idx + freq_offset, symbol_pos, layer_id,
                           (float) ref_symbol.x, (float) ref_symbol.y,
                           (float) gpu_symbol.x, (float) gpu_symbol.y);*/
                    gpu_mismatch += 1;
                }
                symbols_checked += 1;

                freq_idx += ((symbol_is_dmrs) ? 2 : 1);
                // For Resource Allocation Type 0, skip RBs not enabled
                if(tmp_h_dmrs_params.resourceAlloc == 0) {
                    int rb_idx = freq_idx / CUPHY_N_TONES_PER_PRB - tmp_h_dmrs_params.BWP_start_PRB;
                    while(0 == ((tmp_h_dmrs_params.rbBitmap[rb_idx>>5] >> (rb_idx & 0x1F)) & 0x1) && (freq_idx < cutoff_freq_idx))
                    {
                        ++rb_idx;
                        freq_idx += CUPHY_N_TONES_PER_PRB;
                    }
                }

                // skipped_REs_per_symbol, after the subtraction, specifies how many REs are punctured in this specific symbol. This is only relevant for data symbols.
                // Reminder: cumulative_skipped_REs has valid data only for the first num_data_symbols; DMRS symbols that may carry data in case of cdmGrpNoData=1 are
                // not included as there will be no csi-rs on a DMRS symbol, thus the need to skip them.
                uint16_t skipped_REs_per_symbol = (symbol_is_dmrs) ? 0 : tmp_ue_grp_params.cumulative_skipped_REs[per_layer_symbol_offset - seen_dmrs_symbols];
                if ((!symbol_is_dmrs) && (per_layer_symbol_offset > 0) && ((per_layer_symbol_offset - 1 - seen_dmrs_symbols) >= 0)) {
                    skipped_REs_per_symbol -= tmp_ue_grp_params.cumulative_skipped_REs[per_layer_symbol_offset - 1 - seen_dmrs_symbols];
                }

                if(freq_idx >= (cutoff_freq_idx - skipped_REs_per_symbol))
                {
                    freq_idx = start_freq_idx;
                    per_layer_symbol_offset += 1;
                    if (symbol_is_dmrs)  {
                        seen_dmrs_symbols += 1;
                    }
                    if(per_layer_symbol_offset >= allocated_data_or_dmrs_w_data_symbols)
                    {
                        per_layer_symbol_offset = 0;
                        seen_dmrs_symbols = 0;
                    }
                }
            }
        }

        total_error_cnt += gpu_mismatch;

        NVLOGC_FMT(NVLOG_PDSCH, "");
        NVLOGC_FMT(NVLOG_PDSCH, "Fused Rate Matching and Modulation Mapper: Found {} mismatched QAM symbols out of {}", gpu_mismatch, symbols_checked);
        NVLOGC_FMT(NVLOG_PDSCH, "GPU output compared w/ reference dataset <tb*_qams> from <{}>", per_cell_hdf5_filename[cell]);
    }

    return total_error_cnt;
}


void PdschTx::updateOutputTensorBuffer(cudaStream_t cuda_strm)
{
    if (num_cells != 1)
    {
        throw std::runtime_error("Invalid number of cells for cuphyFallbackBufferSetupPdschTx");
    }

    // Update cell_output_tensor_addr field of PdschDmrsParams struct for all TBs.
    // Do an extra H2D copy for that array. No need to update the launch descr.
    // Precondition: all TBs belong to the same first cell.
    for (int i = 0; i < num_TBs; i++) {
        h_dmrs_params[i].cell_output_tensor_addr = dynamic_params->pDataOut->pTDataTx[0].pAddr;
    }
    cudaError_t res = cudaMemcpyAsync(d_dmrs_params, h_dmrs_params, num_TBs * sizeof(PdschDmrsParams), cudaMemcpyHostToDevice, strm);
    if (res != cudaSuccess) {
        throw std::runtime_error("Error for fallback setup");
    }
}


void PdschTx::updateOutputTensorBuffers(int fallback_cells, cudaStream_t cuda_strm)
{
    // Ensure number of cells didn't change from the initial full setup call.
    if (fallback_cells != num_cells)
    {
        throw std::runtime_error("Invalid number of cells for cuphyFallbackBuffersSetupPdschTx");
    }

    // Update cell_output_tensor_addr field of PdschDmrsParams struct for all TBs.
    // Do an extra H2D copy for that array. No need to update the launch descr.

    // Precondition: TBs in h_dmrs_params array are allocated in order of dynamic cells
    // and all TBs belonging to the same cell are consecutive in that array.
    int dyn_cell = 0;
    int TBs_in_dyn_cell = 0;
    for (int i = 0; i < num_TBs; i++) {
        if (TBs_in_dyn_cell >= per_cell_num_TBs[dyn_cell])
        {
            TBs_in_dyn_cell = 0;
            dyn_cell += 1;
        }
        TBs_in_dyn_cell += 1;
        h_dmrs_params[i].cell_output_tensor_addr = dynamic_params->pDataOut->pTDataTx[dyn_cell].pAddr;
        //NVLOGC_FMT(NVLOG_PDSCH, "TB {} with dyn. cell {} has addr {:p}", i, dyn_cell, (void*)h_dmrs_params[i].cell_output_tensor_addr);
    }

    cudaError_t res = cudaMemcpyAsync(d_dmrs_params, h_dmrs_params, num_TBs * sizeof(PdschDmrsParams), cudaMemcpyHostToDevice, strm);
    if (res != cudaSuccess) {
        throw std::runtime_error("Error for fallback setup");
    }
}


cuphyStatus_t PdschTx::prepareDmrsMultipleCells(cudaStream_t cuda_strm)
{

    /* NB precoding: if any of the cells in the cell group have precoding for any of their UEs, then the precoding template
       specialization will be used for all DMRS kernels. Even for cells that do not have any UE with precoding enabled.
       TODO Consider an alternative later, if needed. There is no correctness impact, only a potential performance one. */
    cuphyStatus_t status = cuphySetupPdschDmrs(m_dmrs_launch_cfg.get(),
                                               d_dmrs_params,
                                               num_TBs,
                                               dynamic_params->pCellGrpDynPrm->nPrecodingMatrices > 0, // See note for precoding above
                                               dynamic_params->pDataOut->pTDataTx[0].desc, // not used
                                               dynamic_params->pDataOut->pTDataTx[0].pAddr, //FIXME not used after adding output addr. in PdschDmrsParams struct. Potentially revisit
                                               static_cast<void*>(m_component_descrs.getCpuStartAddrs()[PDSCH_DMRS]), //CPU desc
                                               static_cast<void*>(m_component_descrs.getGpuStartAddrs()[PDSCH_DMRS]), //GPU desc
                                               (!bulk_desc_async_copy),
                                               cuda_strm);
    return status;
}


void PdschTx::runDmrsMultipleCells(cudaStream_t cuda_strm, bool ref_check)
{
   CUresult dmrs_status = launch_kernel(m_dmrs_launch_cfg.get()->m_kernelNodeParams, cuda_strm);

   if(dmrs_status != CUDA_SUCCESS)
   {
        throw std::runtime_error("Invalid argument(s) for cuphyRunPdschDmrs");
   }
}


int PdschTx::refCheckDmrsMultipleCells(cudaStream_t cuda_strm)
{
    int total_error_cnt = 0;

    for (int cell = 0; cell < num_cells; cell++) {

        if(per_cell_input_file[cell] == nullptr)
        {
            NVLOGE_FMT(NVLOG_PDSCH, AERIAL_CUPHY_EVENT, "Cannot do a reference check without a valid input file!");
            return -1;
        }

        using tensor_pinned_C_64F               = typed_tensor<CUPHY_C_64F, pinned_alloc>;
        std::string         ref_dataset_name    = "Xtf";
        tensor_pinned_C_64F pdsch_tx_ref_output = typed_tensor_from_dataset<CUPHY_C_64F, pinned_alloc>((*per_cell_input_file[cell]).open_dataset(ref_dataset_name.c_str()));

        int num_dims;
        int dims[3];
        CUPHY_CHECK(cuphyGetTensorDescriptor(dynamic_params->pDataOut->pTDataTx[cell].desc, 3, nullptr, &num_dims, dims, nullptr));

        tensor_device cell_data_tx_tensor = tensor_device(dynamic_params->pDataOut->pTDataTx[cell].pAddr,
                                                 CUPHY_C_16F,
                                                 pdsch_tx_ref_output.layout().dimensions()[0],
                                                 dims[1],
                                                 pdsch_tx_ref_output.layout().dimensions()[2],
                                                 cuphy::tensor_flags::align_tight);

        typed_tensor<CUPHY_C_16F, pinned_alloc> h_pdsch_out_tensor(cell_data_tx_tensor.layout());
        h_pdsch_out_tensor.convert(cell_data_tx_tensor, cuda_strm);
        CUDA_CHECK(cudaStreamSynchronize(cuda_strm));

        uint32_t gpu_mismatch    = 0;
        uint32_t checked_symbols = 0;
        uint32_t total_layers    = pdsch_tx_ref_output.layout().dimensions()[2]; // data_tx_tensor is overprovisioned in terms of # layers
        int      num_hdf5_PRBs   = pdsch_tx_ref_output.layout().dimensions()[0] / CUPHY_N_TONES_PER_PRB;
        uint16_t static_cell     = dynamic_params->pCellGrpDynPrm->pCellPrms[cell].cellPrmStatIdx;
        int      num_BWP_PRBs    = static_params->pCellStatPrms[static_cell].nPrbDlBwp;
        if (num_hdf5_PRBs != num_BWP_PRBs)
        {
            NVLOGE_FMT(NVLOG_PDSCH, AERIAL_CUPHY_EVENT, "Error! # PRBs in Xtf dataset frequency dimensions are {} different than number of PRBs {}",
                       num_hdf5_PRBs, num_BWP_PRBs);
            return -1;
        }

        for (int TB_id = 0; TB_id < per_cell_num_TBs[cell]; TB_id++) {
            int overall_TB_id = per_cell_TB_params_offset[cell] + TB_id;
            int start_Rb = h_dmrs_params[overall_TB_id].start_Rb;
            int num_Rbs = h_dmrs_params[overall_TB_id].num_Rbs;

            std::vector<int> dmrs_positions(h_dmrs_params[overall_TB_id].num_dmrs_symbols); // Holds DMRS positions 0-based indexing
            uint16_t bitmask = h_dmrs_params[overall_TB_id].dmrs_sym_loc;
            for (int j = 0; j < dmrs_positions.size(); j++) {
                dmrs_positions[j] = (bitmask >> (4 * j)) & 0xF;
                //NVLOGD_FMT(NVLOG_PDSCH, "DMRS position[{}] ={}", j, dmrs_positions[j]);
            }

            int numLayers = h_dmrs_params[overall_TB_id].enablePrcdBf ? h_dmrs_params[overall_TB_id].Np : h_dmrs_params[overall_TB_id].num_layers;

            //FIXME Only checks allocated PRBs have valid values. Could also ensure it doesn't overwrite anything else.
            for (int freq_idx = start_Rb * CUPHY_N_TONES_PER_PRB; freq_idx < CUPHY_N_TONES_PER_PRB * (start_Rb + num_Rbs); freq_idx++) {
                for (int dmrs_symbol = 0; dmrs_symbol < h_dmrs_params[overall_TB_id].num_dmrs_symbols; dmrs_symbol++)  {
                    int symbol_id = dmrs_positions[dmrs_symbol];
                    for (int tmp_layer = 0; tmp_layer < numLayers; tmp_layer++) {
                        int layer_id = tmp_layer;
                        if(!h_dmrs_params[overall_TB_id].enablePrcdBf)
                        {
                            layer_id = calculate_layer_id(h_dmrs_params[overall_TB_id].port_ids[tmp_layer], h_dmrs_params[overall_TB_id].n_scid, h_dmrs_params[overall_TB_id].nlAbove16);
                        }
                        int delta =  (h_dmrs_params[overall_TB_id].port_ids[tmp_layer] >> 1) & 0x1U;

                        // If dmrsCmdGrpsNoData is set to 1 for this TB only check half the REs per PRB for this DMRS only example, as the rest will contain data symbols.
                        if (((h_dmrs_params[overall_TB_id].dmrsCdmGrpsNoData1 == 1) && (freq_idx % 2 == delta)) ||
                            (h_dmrs_params[overall_TB_id].dmrsCdmGrpsNoData1 == 0))
                        {
                            __half2 gpu_symbol = h_pdsch_out_tensor(freq_idx, symbol_id, layer_id);
                            __half2 ref_symbol;
                            /* The reference HDF5 dataset should contain all the number of PRBs for this BWP. Any unallocated PRB should be empty. */
                            ref_symbol.x = (half)pdsch_tx_ref_output(freq_idx, symbol_id, layer_id).x;
                            ref_symbol.y = (half)pdsch_tx_ref_output(freq_idx, symbol_id, layer_id).y;
                            checked_symbols += 1;
                            if(!complex_approx_equal<__half2, __half>(gpu_symbol, ref_symbol, pdsch_comp_tol))
                            {
                                /*NVLOGC_FMT(NVLOG_PDSCH, "Error! Mismatch for symbol {{freq_bin {}, symbol {}, layer {}}}:  expected={:f} + i {:f} vs. gpu={:f} + i {:f}",
                                    freq_idx, symbol_id, layer_id,
                                   (float) ref_symbol.x, (float) ref_symbol.y,
                                   (float) gpu_symbol.x, (float) gpu_symbol.y);*/
                                gpu_mismatch += 1;
                           }
                       }
                    }
                }
            }
        }

        total_error_cnt += gpu_mismatch;

        NVLOGC_FMT(NVLOG_PDSCH, "");
        NVLOGC_FMT(NVLOG_PDSCH, "PDSCH DMRS: Found {} mismatched symbols out of {} in {{{}, {}, {}}} output tensor.",
                   gpu_mismatch, checked_symbols, num_BWP_PRBs * CUPHY_N_TONES_PER_PRB, OFDM_SYMBOLS_PER_SLOT, total_layers);
        NVLOGC_FMT(NVLOG_PDSCH, "GPU output compared w/ reference dataset <{}> from <{}>", ref_dataset_name, per_cell_hdf5_filename[cell]);
    }

    return total_error_cnt;
}


int PdschTx::refCheckTxDataMultipleCells(cudaStream_t cuda_strm)
{
    //return refCheckDmrsMultipleCells(cuda_strm); // use to only check DMRS symbols

    int total_error_cnt = 0;

    for (int cell = 0; cell < num_cells; cell++) {

        if(per_cell_input_file[cell] == nullptr)
        {
            NVLOGE_FMT(NVLOG_PDSCH, AERIAL_CUPHY_EVENT, "Cannot do a reference check without a valid input file!");
            return -1;
        }

        using tensor_pinned_C_64F               = typed_tensor<CUPHY_C_64F, pinned_alloc>;
        std::string         ref_dataset_name    = "Xtf";
        tensor_pinned_C_64F pdsch_tx_ref_output = typed_tensor_from_dataset<CUPHY_C_64F, pinned_alloc>((*per_cell_input_file[cell]).open_dataset(ref_dataset_name.c_str()));

        /*int num_dims;
        int dims[3];
        CUPHY_CHECK(cuphyGetTensorDescriptor(dynamic_params->pDataOut->pTDataTx[cell].desc, 3, nullptr, &num_dims, dims, nullptr));
        */

        tensor_device cell_data_tx_tensor = tensor_device(dynamic_params->pDataOut->pTDataTx[cell].pAddr,
                                                 CUPHY_C_16F,
                                                 pdsch_tx_ref_output.layout().dimensions()[0],
                                                 pdsch_tx_ref_output.layout().dimensions()[1],
                                                 pdsch_tx_ref_output.layout().dimensions()[2],
                                                 cuphy::tensor_flags::align_tight);

        typed_tensor<CUPHY_C_16F, pinned_alloc> h_pdsch_out_tensor(cell_data_tx_tensor.layout());
        h_pdsch_out_tensor.convert(cell_data_tx_tensor, cuda_strm);
        CUDA_CHECK(cudaStreamSynchronize(cuda_strm));

        uint32_t gpu_mismatch    = 0;
        uint32_t checked_symbols = 0;
        uint32_t total_layers    = pdsch_tx_ref_output.layout().dimensions()[2]; // data_tx_tensor is overprovisioned in terms of # layers
        int      num_hdf5_PRBs   = pdsch_tx_ref_output.layout().dimensions()[0] / CUPHY_N_TONES_PER_PRB;
        uint16_t static_cell     = dynamic_params->pCellGrpDynPrm->pCellPrms[cell].cellPrmStatIdx;
        int      num_BWP_PRBs    = static_params->pCellStatPrms[static_cell].nPrbDlBwp;
        if (num_hdf5_PRBs != num_BWP_PRBs)
        {
            NVLOGE_FMT(NVLOG_PDSCH, AERIAL_CUPHY_EVENT, "Error! # PRBs in Xtf dataset frequency dimensions are {} different than number of PRBs {}",
                       num_hdf5_PRBs, num_BWP_PRBs);
            return -1;
        }

        for(int layer_id = 0; layer_id < total_layers; layer_id++)
        {
            for(int symbol_id = 0; symbol_id < OFDM_SYMBOLS_PER_SLOT; symbol_id++)
            {
                for(int freq_idx = 0; freq_idx < num_BWP_PRBs * CUPHY_N_TONES_PER_PRB; freq_idx++)
                {
                    __half2 gpu_symbol = h_pdsch_out_tensor(freq_idx, symbol_id, layer_id);
                    __half2 ref_symbol;
                    /* The reference HDF5 dataset should contain all the number of PRBs for this BWP. Any unallocated PRB should be empty. */
                    ref_symbol.x = (half)pdsch_tx_ref_output(freq_idx, symbol_id, layer_id).x;
                    ref_symbol.y = (half)pdsch_tx_ref_output(freq_idx, symbol_id, layer_id).y;
                    checked_symbols += 1;
                    if(!complex_approx_equal<__half2, __half>(gpu_symbol, ref_symbol, pdsch_comp_tol))
                    {
                        /*NVLOGD_FMT(NVLOG_PDSCH, "Error! Mismatch for symbol {{freq_bin {:d}, symbol {:d}, layer {:d}}}: expected={: 5.3f} + i {: 5.3f} vs. gpu={: 5.3f} + i {: 5.3f}",
                                freq_idx, symbol_id, layer_id,
                               (float) ref_symbol.x, (float) ref_symbol.y,
                               (float) gpu_symbol.x, (float) gpu_symbol.y);*/
                        gpu_mismatch += 1;
                    }
                }
            }
        }

        total_error_cnt += gpu_mismatch;

        NVLOGC_FMT(NVLOG_PDSCH, "");
        NVLOGC_FMT(NVLOG_PDSCH, "PDSCH: Found {} mismatched symbols out of {} in {{{}, {}, {}}} output tensor.",
                   gpu_mismatch, checked_symbols, num_BWP_PRBs * CUPHY_N_TONES_PER_PRB, OFDM_SYMBOLS_PER_SLOT, total_layers);
        NVLOGC_FMT(NVLOG_PDSCH, "GPU output compared w/ reference dataset <{}> from <{}>", ref_dataset_name, per_cell_hdf5_filename[cell]);
    }

    return total_error_cnt;
}

cuphyStatus_t PdschTx::prepareBuffersMultipleCells(cudaStream_t cuda_strm)
{
    cuphyStatus_t status = CUPHY_STATUS_SUCCESS;
    ldpc_stream_elements = 0;
    //PUSH_RANGE("prepareCRC", 1);
    status = prepareCRCMultipleCells(cuda_strm);
    cuphyStatus_t status_step2 = prepareCRC_step2MultipleCells(cuda_strm);
    //POP_RANGE
    if(status != CUPHY_STATUS_SUCCESS)
    {
        return status;
    }
    if(status_step2 != CUPHY_STATUS_SUCCESS)
    {
        return status;
    }

    // prepareRM moved before prepareLDPC.
    //PUSH_RANGE("prepareRM", 3);
    status = prepareRateMatchingMultipleCells(cuda_strm);
    //POP_RANGE

    if(status != CUPHY_STATUS_SUCCESS)
    {
        return status;
    }

    //PUSH_RANGE("prepareLDPC", 2);

    status = prepareLDPCBatching(cuda_strm); // deals with batching under the hood;
    if(status != CUPHY_STATUS_SUCCESS)
    {
        return status;
    }
    //POP_RANGE

    //PUSH_RANGE("prepareDMRS", 4);
    if(!aas_mode)
    {
        status = prepareDmrsMultipleCells(cuda_strm); // should be called before runModulation is called
        if(status != CUPHY_STATUS_SUCCESS)
        {
            NVLOGE_FMT(NVLOG_PDSCH, AERIAL_CUPHY_EVENT, "Invalid argument(s) for cuphySetupPdschDmrs");
            return status;
        }
    }
    //POP_RANGE

    status = d_compute_re_maps(); // CSI-RS pre-processing setup, if CSI-RS params present
    return status; // assuming d_compute_re_maps is the last call
}

cuphyStatus_t PdschTx::RunMultipleCells(const cudaStream_t& cuda_strm, bool ref_check)
{
    // Check if we are in the fallback mode (i.e., calling back to back runs for the same TV with a single setup but still require functional correctness)
    // If yes, we need to reset all buffers managed by PdschTx that involve atomic updates.
    // Note that the output tensor buffer should still be reset by the caller.
    if ((dynamic_params->procModeBmsk & PDSCH_PROC_MODE_SETUP_ONCE_FALLBACK) != 0)
    {
        // Reset per-TB CRC buffers unless the TB-CRCs are provided by the caller. Check 1st cell, and assume similar setting across all
        if (h_per_cell_pipeline_input_tb_crcs[0] == nullptr)
        {
            CUDA_CHECK(cudaMemsetAsync((void*)((uint32_t*)m_component_descrs.getGpuStartAddrs()[PDSCH_TB_CRCS]),
                          0, sizeof(uint32_t) * num_TBs, cuda_strm));
        }
    }

    ldpc_streams[0] = cuda_strm;
    ran_pipeline    = true;

    if(graph_mode)
    {
        {
        MemtraceDisableScope md; //FIXME temporary
        CUresult e = cuGraphLaunch(exec_graph, cuda_strm);
        if (e != CUDA_SUCCESS) {
            const char* pErrStr;
            CUresult str_e = cuGetErrorString(e, &pErrStr);
            NVLOGE_FMT(NVLOG_PDSCH, AERIAL_CUPHY_EVENT, "Invalid graph launch for PDSCH ({}, {}).", e, (str_e == CUDA_SUCCESS) ? pErrStr : "CUDA error");
            return CUPHY_STATUS_INTERNAL_ERROR;
        }
        }

        if(ref_check)
        {
            failed_ref_checks += (refCheckCRCMultipleCells(cuda_strm) != 0) ? 1 : 0;
            failed_ref_checks += (refCheckLDPCMultipleCells(cuda_strm) != 0) ? 1 : 0;

            if(aas_mode)
            {
                failed_ref_checks += (refCheckRateMatchingMultipleCells(cuda_strm) != 0) ? 1 : 0;
            }
            else
            {
                failed_ref_checks += (refCheckModulationMultipleCells(cuda_strm) != 0) ? 1 : 0; // Fused Rate Matching + Modulation
                failed_ref_checks += (refCheckTxDataMultipleCells(cuda_strm) != 0) ? 1 : 0;
            }
        }

        return (failed_ref_checks == 0) ? CUPHY_STATUS_SUCCESS : CUPHY_STATUS_INTERNAL_ERROR;
    }

    runPdschCsirsPrepMultipleCells(cuda_strm); //TODO need not be before CRC; it only needs to complete before rate-matching
    runCRCMultipleCells(cuda_strm, ref_check);

    runLDPCBatching(cuda_strm, ref_check);
    if(ref_check)
    {
        failed_ref_checks += (refCheckLDPCMultipleCells(cuda_strm) != 0) ? 1 : 0;
    }
    runRateMatchingMultipleCells(cuda_strm, ref_check); // Fused Rate Matching and Modulation

    if(!aas_mode)
    {
        runDmrsMultipleCells(cuda_strm, ref_check);

        if(ref_check)
        {
            failed_ref_checks += (refCheckTxDataMultipleCells(cuda_strm) != 0) ? 1 : 0;
        }
    }

    return (failed_ref_checks == 0) ? CUPHY_STATUS_SUCCESS : CUPHY_STATUS_INTERNAL_ERROR;
}

void PdschTx::createAndInstantiateGraph()
{
    CU_CHECK_EXCEPTION(cuGraphCreate(&m_graph, 0));
    // Set empty graph kernel node with a single argument (a pointer) to avoid dynamic
    // memory allocation during graph kernel node update. All PDSCH kernels have a single kernel argument.
    // Use of CUPHY_CHECK(cuphySetEmptyKernelNodeParams(&m_emptyNode0paramDriver)) would only be applicable if
    // kernel had no args.
    void* arg;
    void* kernelParams[1] = {&arg};
    CUPHY_CHECK(cuphySetGenericEmptyKernelNodeParams(&m_emptyNodeParamsDriver, 1, &kernelParams[0]));
    total_LDPC_kernel_configs = 1; // hardcoded assumes ones LDPC kernel launched; will be updated properly in first setup

    //TODO: last argument to cuphySetGenericEmptyKernelNodeGridConstantParams function depends on the size of the kernel's (__grid_constant__) descriptor
    void* kernelParams_48B[1] = {&arg_48B};
    CUPHY_CHECK(cuphySetGenericEmptyKernelNodeGridConstantParams(&m_emptyNodeParamsDriver_48B, &kernelParams_48B[0], 0, 48));

    void* kernelParams_32B[1] = {&arg_32B};
    CUPHY_CHECK(cuphySetGenericEmptyKernelNodeGridConstantParams(&m_emptyNodeParamsDriver_32B, &kernelParams_32B[0], 0, 32));

    if(!read_TB_CRC)
    {
        CU_CHECK_EXCEPTION(cuGraphAddKernelNode(&m_crcEncodeNodesMultipleCells[0], m_graph, nullptr, 0, &m_emptyNodeParamsDriver_48B));
    }
    CU_CHECK_EXCEPTION(cuGraphAddKernelNode(&m_crcEncodeNodesMultipleCells[1], m_graph, nullptr, 0, &m_emptyNodeParamsDriver_48B));

    // Add fused rate matching and modulation mapper node
    CU_CHECK_EXCEPTION(cuGraphAddKernelNode(&m_rmNodesMultipleCells[0], m_graph, nullptr, 0, &m_emptyNodeParamsDriver));
    if(!aas_mode)
    {
        CU_CHECK_EXCEPTION(cuGraphAddKernelNode(&m_dmrsNodeMultipleCells[0], m_graph, nullptr, 0, &m_emptyNodeParamsDriver));
    }
    int first_unused_ldpc_node = total_LDPC_kernel_configs;
    prev_first_unused_ldpc_node = first_unused_ldpc_node;

    // There is a runtime check in prepareCrcMultipleCells that verifies that total # LDPC configs is less than PDSCH_MAX_HET_LDPC_CONFIGS_SUPPORTED.
    for(int LDPC_cfg = 0; LDPC_cfg < std::min(per_cell_group_max_TBs, PDSCH_MAX_HET_LDPC_CONFIGS_SUPPORTED); LDPC_cfg++)
    {
        CU_CHECK_EXCEPTION(cuGraphAddKernelNode(&m_ldpcEncodeNodesMultipleCells[LDPC_cfg], m_graph, nullptr, 0, &m_emptyNodeParamsDriver_32B));
    }

    CU_CHECK_EXCEPTION(cuGraphAddKernelNode(&m_csirsPrepNodesMultipleCells[0], m_graph, nullptr, 0, &m_emptyNodeParamsDriver));
    CU_CHECK_EXCEPTION(cuGraphAddKernelNode(&m_csirsPrepNodesMultipleCells[1], m_graph, nullptr, 0, &m_emptyNodeParamsDriver));
    CU_CHECK_EXCEPTION(cuGraphAddKernelNode(&m_csirsPrepNodesMultipleCells[2], m_graph, nullptr, 0, &m_emptyNodeParamsDriver));

    CU_CHECK_EXCEPTION(cuGraphAddKernelNode(&m_prepareCRCEncodeNodeMultipleCells[0], m_graph, nullptr, 0, &m_emptyNodeParamsDriver));

    addDependenciesMultipleCells();

#if 0
    // Generate a ps file. Can convert to pdf via $dot -Tpdf graph.ps -o graph.pdf
    if (max_cells == 1) {
        CU_CHECK_EXCEPTION(cuGraphDebugDotPrint(m_graph, "pdsch_1cell_dot_graph", cudaGraphDebugDotFlagsVerbose));
    }
#endif

#if CUDA_VERSION >= 12000
    CU_CHECK_EXCEPTION(cuGraphInstantiate(&exec_graph, m_graph, 0));
#else
    CU_CHECK_EXCEPTION(cuGraphInstantiate(&exec_graph, m_graph, 0, 0, 0));
#endif

    disableLDPCNodesMultipleCells();
    disableCsirsPrepNodesMultipleCells();
    prev_had_CSIRS_params = false;
    total_LDPC_kernel_configs = 0; // reset to 0 before first setup is called
}



void PdschTx::disableLDPCNodesMultipleCells()
{
    int first_unused_ldpc_node = total_LDPC_kernel_configs;

    // Update remaining nodes to point to an empty kernel function and minimal launch config.
    for(int LDPC_cfg = first_unused_ldpc_node; LDPC_cfg < std::min(per_cell_group_max_TBs, PDSCH_MAX_HET_LDPC_CONFIGS_SUPPORTED); LDPC_cfg++)
    {
        CU_CHECK_EXCEPTION(cuGraphNodeSetEnabled(exec_graph, m_ldpcEncodeNodesMultipleCells[LDPC_cfg], 0));
    }
}

void PdschTx::disableCsirsPrepNodesMultipleCells()
{
    CU_CHECK_EXCEPTION(cuGraphNodeSetEnabled(exec_graph, m_csirsPrepNodesMultipleCells[0], 0));
    CU_CHECK_EXCEPTION(cuGraphNodeSetEnabled(exec_graph, m_csirsPrepNodesMultipleCells[1], 0));
    CU_CHECK_EXCEPTION(cuGraphNodeSetEnabled(exec_graph, m_csirsPrepNodesMultipleCells[2], 0));
}


void PdschTx::updateNodeParamsMultipleCells()
{
    if (dynamic_params->pCellGrpDynPrm[0].nCsiRsPrms == 0) {
        // No action needed if there are no CSI-RS parameters in this graph launch, as was the case in the last one as nodes are already disabled.
        // If we we previously had CSI-RS parameters, then all CSI-RS kernel nodes have to be disabled.
        if (prev_had_CSIRS_params)
        {
            CU_CHECK_EXCEPTION(cuGraphNodeSetEnabled(exec_graph, m_csirsPrepNodesMultipleCells[0], 0));
            CU_CHECK_EXCEPTION(cuGraphNodeSetEnabled(exec_graph, m_csirsPrepNodesMultipleCells[1], 0));
            CU_CHECK_EXCEPTION(cuGraphNodeSetEnabled(exec_graph, m_csirsPrepNodesMultipleCells[2], 0));
        }

    } else {
        if (!prev_had_CSIRS_params)
        {
            // Only re-enable nodes if they were previously disabled, i.e., if there were no CSI-RS parameters in previous graph update
            CU_CHECK_EXCEPTION(cuGraphNodeSetEnabled(exec_graph, m_csirsPrepNodesMultipleCells[0], 1));
            CU_CHECK_EXCEPTION(cuGraphNodeSetEnabled(exec_graph, m_csirsPrepNodesMultipleCells[1], 1));
            CU_CHECK_EXCEPTION(cuGraphNodeSetEnabled(exec_graph, m_csirsPrepNodesMultipleCells[2], 1));
        }
        CU_CHECK_EXCEPTION_W_TAG(NVLOG_PDSCH, cuGraphExecKernelNodeSetParams(exec_graph, m_csirsPrepNodesMultipleCells[0], &(m_csirs_prep_launch_cfg.get()->m_kernelNodeParams[0])));
        CU_CHECK_EXCEPTION_W_TAG(NVLOG_PDSCH, cuGraphExecKernelNodeSetParams(exec_graph, m_csirsPrepNodesMultipleCells[1], &(m_csirs_prep_launch_cfg.get()->m_kernelNodeParams[1])));
        CU_CHECK_EXCEPTION_W_TAG(NVLOG_PDSCH, cuGraphExecKernelNodeSetParams(exec_graph, m_csirsPrepNodesMultipleCells[2], &(m_csirs_prep_launch_cfg.get()->m_kernelNodeParams[2])));
    }
    // Update prev_had_CSIRS_params for next graph update
    prev_had_CSIRS_params = (dynamic_params->pCellGrpDynPrm[0].nCsiRsPrms != 0);

    // Update CRC nodes
    CU_CHECK_EXCEPTION_W_TAG(NVLOG_PDSCH, cuGraphExecKernelNodeSetParams(exec_graph, m_prepareCRCEncodeNodeMultipleCells[0], &(m_prepare_crc_encode_launch_cfg.get()->m_kernelNodeParams)));

    if(!read_TB_CRC)
    {
        CU_CHECK_EXCEPTION_W_TAG(NVLOG_PDSCH, cuGraphExecKernelNodeSetParams(exec_graph, m_crcEncodeNodesMultipleCells[0], &(m_crc_encode_launch_cfg.get()->m_kernelNodeParams[0])));
    }
    {
        CU_CHECK_EXCEPTION_W_TAG(NVLOG_PDSCH, cuGraphExecKernelNodeSetParams(exec_graph, m_crcEncodeNodesMultipleCells[1], &(m_crc_encode_launch_cfg.get()->m_kernelNodeParams[1])));
    }

    // Update LDPC nodes
    // NB: The CUDA graph kernel node updates can fail for CUDA < 11.2 if they involve a change in the kernel function.
    int first_unused_ldpc_node = total_LDPC_kernel_configs;

    for(int LDPC_cfg = 0; LDPC_cfg < first_unused_ldpc_node; LDPC_cfg++)
    {
        // Only re-enable nodes that were previously disabled
        if (LDPC_cfg >= prev_first_unused_ldpc_node) {
            CU_CHECK_EXCEPTION(cuGraphNodeSetEnabled(exec_graph, m_ldpcEncodeNodesMultipleCells[LDPC_cfg], 1));
        }
	CU_CHECK_EXCEPTION_W_TAG(NVLOG_PDSCH, cuGraphExecKernelNodeSetParams(exec_graph, m_ldpcEncodeNodesMultipleCells[LDPC_cfg], &(m_ldpc_encode_launch_cfg.get()[LDPC_cfg].m_kernelNodeParams)));
    }


    // Update remaining nodes to point to an empty kernel function and minimal launch config or disable them.
    // Only do that for nodes that were previously enabled, i.e., from within [first_unused_ldpc_node, prev_first_unused_ldpc_node) if non empty.
    // This assumes that all std::min(per_cell_group_max_TBs, PDSCH_MAX_HET_LDPC_CONFIGS_SUPPORTED) LDPC graph nodes were originally disabled.
    for(int LDPC_cfg = first_unused_ldpc_node; LDPC_cfg < prev_first_unused_ldpc_node; LDPC_cfg++)
    {
        CU_CHECK_EXCEPTION(cuGraphNodeSetEnabled(exec_graph, m_ldpcEncodeNodesMultipleCells[LDPC_cfg], 0));
    }
    // Update prev_first_unused_ldpc_node for next graph update
    prev_first_unused_ldpc_node = first_unused_ldpc_node;

    // Update fused rate matching and modulation mapper node
    CU_CHECK_EXCEPTION_W_TAG(NVLOG_PDSCH, cuGraphExecKernelNodeSetParams(exec_graph, m_rmNodesMultipleCells[0], &(m_rate_matching_launch_cfg.get()->m_kernelNodeParams[0])));

    if(!aas_mode)
    {
        // Update DMRS node
        CU_CHECK_EXCEPTION_W_TAG(NVLOG_PDSCH, cuGraphExecKernelNodeSetParams(exec_graph, m_dmrsNodeMultipleCells[0], &(m_dmrs_launch_cfg.get()->m_kernelNodeParams)));
    }
}

//Dependencies helper which assumes nullptr edge data (necessary with CUDA 13)
CUresult PdschTx::addDependenciesHelper(CUgraph hGraph, const CUgraphNode* from, const CUgraphNode* to, size_t numDependencies)
{
#if CUDA_VERSION >= 13000
    return cuGraphAddDependencies(hGraph, from, to, nullptr /* edgeData*/, numDependencies);
#else
    return cuGraphAddDependencies(hGraph, from, to, numDependencies);
#endif
}

// Add dependencies.
void PdschTx::addDependenciesMultipleCells()
{
#if OLD_GRAPH
    CU_CHECK_EXCEPTION(addDependenciesHelper(m_graph, &m_prepareCRCEncodeNodeMultipleCells[0], &m_csirsPrepNodesMultipleCells[2], 1));
    CU_CHECK_EXCEPTION(addDependenciesHelper(m_graph, &m_csirsPrepNodesMultipleCells[2], &m_csirsPrepNodesMultipleCells[0], 1));
    CU_CHECK_EXCEPTION(addDependenciesHelper(m_graph, &m_csirsPrepNodesMultipleCells[0], &m_csirsPrepNodesMultipleCells[1], 1));

    if(!read_TB_CRC)
    {
        CU_CHECK_EXCEPTION(addDependenciesHelper(m_graph, &m_csirsPrepNodesMultipleCells[1], &m_crcEncodeNodesMultipleCells[0], 1));
        // m_crcEncodeNodes[0] -> m_crcEncodeNodes[1]
        CU_CHECK_EXCEPTION(addDependenciesHelper(m_graph, &m_crcEncodeNodesMultipleCells[0], &m_crcEncodeNodesMultipleCells[1], 1));
    }
    else
    {
        CU_CHECK_EXCEPTION(addDependenciesHelper(m_graph, &m_csirsPrepNodesMultipleCells[1], &m_crcEncodeNodesMultipleCells[1], 1));
    }
    // LDPC nodes and fused rate matching and modulation mapper node
    for(int LDPC_cfg = 0; LDPC_cfg < std::min(per_cell_group_max_TBs, PDSCH_MAX_HET_LDPC_CONFIGS_SUPPORTED); LDPC_cfg++)
    {
        CU_CHECK_EXCEPTION(addDependenciesHelper(m_graph, &m_crcEncodeNodesMultipleCells[1], &m_ldpcEncodeNodesMultipleCells[LDPC_cfg], 1));
        CU_CHECK_EXCEPTION(addDependenciesHelper(m_graph, &m_ldpcEncodeNodesMultipleCells[LDPC_cfg], &m_rmNodesMultipleCells[0], 1));
    }
    if(!aas_mode)
    {
        // m_rmNode -> m_dmrsNode is not a true dependency. The DMRS node can execute
        // alongside any other node.  TODO Explore other options
        CU_CHECK_EXCEPTION(addDependenciesHelper(m_graph, &m_rmNodesMultipleCells[0], &m_dmrsNodeMultipleCells[0], 1));
    }
#else

#if MOVE_DMRS_EARLY
    CU_CHECK_EXCEPTION(addDependenciesHelper(m_graph, &m_prepareCRCEncodeNodeMultipleCells[0], &m_csirsPrepNodesMultipleCells[2], 1));
    CU_CHECK_EXCEPTION(addDependenciesHelper(m_graph, &m_csirsPrepNodesMultipleCells[2], &m_csirsPrepNodesMultipleCells[0], 1));
    CU_CHECK_EXCEPTION(addDependenciesHelper(m_graph, &m_csirsPrepNodesMultipleCells[0], &m_csirsPrepNodesMultipleCells[1], 1));
    if(!aas_mode)
    {
#if FUSED_DL_RM_AFTER_DMRS
        CU_CHECK_EXCEPTION(addDependenciesHelper(m_graph, &m_csirsPrepNodesMultipleCells[1], &m_dmrsNodeMultipleCells[0], 1));
        CU_CHECK_EXCEPTION(addDependenciesHelper(m_graph, &m_dmrsNodeMultipleCells[0], &m_rmNodesMultipleCells[0], 1));
#else
        CU_CHECK_EXCEPTION(addDependenciesHelper(m_graph, &m_csirsPrepNodesMultipleCells[1], &m_rmNodesMultipleCells[0], 1));
        CU_CHECK_EXCEPTION(addDependenciesHelper(m_graph, &m_csirsPrepNodesMultipleCells[1], &m_dmrsNodeMultipleCells[0], 1));
#endif
    }
    else
    {
        CU_CHECK_EXCEPTION(addDependenciesHelper(m_graph, &m_csirsPrepNodesMultipleCells[1], &m_rmNodesMultipleCells[0], 1));
    }

    if(!read_TB_CRC)
    {
        CU_CHECK_EXCEPTION(addDependenciesHelper(m_graph, &m_prepareCRCEncodeNodeMultipleCells[0], &m_crcEncodeNodesMultipleCells[0], 1));
        // m_crcEncodeNodes[0] -> m_crcEncodeNodes[1]
        CU_CHECK_EXCEPTION(addDependenciesHelper(m_graph, &m_crcEncodeNodesMultipleCells[0], &m_crcEncodeNodesMultipleCells[1], 1));
    }
    else
    {
        CU_CHECK_EXCEPTION(addDependenciesHelper(m_graph, &m_csirsPrepNodesMultipleCells[1], &m_crcEncodeNodesMultipleCells[1], 1));
    }
    // LDPC nodes and fused rate matching and modulation mapper node
    for(int LDPC_cfg = 0; LDPC_cfg < std::min(per_cell_group_max_TBs, PDSCH_MAX_HET_LDPC_CONFIGS_SUPPORTED); LDPC_cfg++)
    {
        CU_CHECK_EXCEPTION(addDependenciesHelper(m_graph, &m_crcEncodeNodesMultipleCells[1], &m_ldpcEncodeNodesMultipleCells[LDPC_cfg], 1));
        CU_CHECK_EXCEPTION(addDependenciesHelper(m_graph, &m_ldpcEncodeNodesMultipleCells[LDPC_cfg], &m_rmNodesMultipleCells[0], 1));
    }
#else
    CU_CHECK_EXCEPTION(addDependenciesHelper(m_graph, &m_prepareCRCEncodeNodeMultipleCells[0], &m_csirsPrepNodesMultipleCells[2], 1));
    CU_CHECK_EXCEPTION(addDependenciesHelper(m_graph, &m_csirsPrepNodesMultipleCells[2], &m_csirsPrepNodesMultipleCells[0], 1));
    CU_CHECK_EXCEPTION(addDependenciesHelper(m_graph, &m_csirsPrepNodesMultipleCells[0], &m_csirsPrepNodesMultipleCells[1], 1));
    CU_CHECK_EXCEPTION(addDependenciesHelper(m_graph, &m_csirsPrepNodesMultipleCells[1], &m_rmNodesMultipleCells[0], 1));

    if(!read_TB_CRC)
    {
        CU_CHECK_EXCEPTION(addDependenciesHelper(m_graph, &m_prepareCRCEncodeNodeMultipleCells[0], &m_crcEncodeNodesMultipleCells[0], 1));
        // m_crcEncodeNodes[0] -> m_crcEncodeNodes[1]
        CU_CHECK_EXCEPTION(addDependenciesHelper(m_graph, &m_crcEncodeNodesMultipleCells[0], &m_crcEncodeNodesMultipleCells[1], 1));
    }
    else
    {
        CU_CHECK_EXCEPTION(addDependenciesHelper(m_graph, &m_csirsPrepNodesMultipleCells[1], &m_crcEncodeNodesMultipleCells[1], 1));
    }
    // LDPC nodes and fused rate matching and modulation mapper node
    for(int LDPC_cfg = 0; LDPC_cfg < std::min(per_cell_group_max_TBs, PDSCH_MAX_HET_LDPC_CONFIGS_SUPPORTED); LDPC_cfg++)
    {
        CU_CHECK_EXCEPTION(addDependenciesHelper(m_graph, &m_crcEncodeNodesMultipleCells[1], &m_ldpcEncodeNodesMultipleCells[LDPC_cfg], 1));
        CU_CHECK_EXCEPTION(addDependenciesHelper(m_graph, &m_ldpcEncodeNodesMultipleCells[LDPC_cfg], &m_rmNodesMultipleCells[0], 1));
    }
    if(!aas_mode)
    {
        // m_rmNode -> m_dmrsNode is not a true dependency. The DMRS node can execute
        // alongside any other node.  TODO Explore other options
        CU_CHECK_EXCEPTION(addDependenciesHelper(m_graph, &m_rmNodesMultipleCells[0], &m_dmrsNodeMultipleCells[0], 1));
    }
#endif
#endif

}

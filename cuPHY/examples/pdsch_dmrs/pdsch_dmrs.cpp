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

#include <cstdlib>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include "cuphy.h"
#include "cuphy.hpp"
#include "cuphy_hdf5.hpp"
#include "hdf5hpp.hpp"
#include "util.hpp"
#include "cuphy_api.h"
#include "cuphy_internal.h"

using namespace std;
using namespace cuphy;

void usage() {

    std::cout << "pdsch_dmrs [options]" << std::endl;
    std::cout << "  Options:" << std::endl;
    std::cout << "     -h                              (Display usage information)" << std::endl;
    std::cout << "     input_filename                  (Input HDF5 filename)" << std::endl;


    std::cout << std::endl;
    std::cout << "  Examples:" << std::endl;
    std::cout << "      ./pdsch_dmrs ~/input_file.h5" << std::endl;
    std::cout << "      ./pdsch_dmrs -h" << std::endl;
}


int main(int argc, char* argv[]) {

    using tensor_pinned_C_32F = typed_tensor<CUPHY_C_32F, pinned_alloc>;
    using tensor_pinned_C_64F = typed_tensor<CUPHY_C_64F, pinned_alloc>;
    cuphyNvlogFmtHelper nvlog_fmt("pdsch_dmrs.log");

    const int ELEMENT_SIZE = sizeof(uint32_t) * 8; // 32 bits
    cudaStream_t strm = 0;

    if ((argc != 2) || ((argc == 2) && (argv[1][0] == '-') && (argv[1][1] == 'h'))) {
        usage();
        exit(1);
    }

    // Open HDF5 input file
    std::unique_ptr<hdf5hpp::hdf5_file> input_file(new hdf5hpp::hdf5_file(hdf5hpp::hdf5_file::open(argv[1])));

    // Parse PdschDmrsParams - Different code path depending on which TV API is used
    std::vector<PdschDmrsParams> h_dmrs_params;

    bool use_new_api = input_file->is_valid_dataset("ue_pars");
    if (!use_new_api) {
        NVLOGE_FMT(NVLOG_PDSCH, AERIAL_CUPHY_EVENT,  "PDSCH DMRS component example no longer supports old API TVs");
        exit(1);
    }

    //Reference comparison after resource element mapping of DMRS. Will only check the DMRS QAMs.
    tensor_pinned_C_64F re_mapped_ref_data = typed_tensor_from_dataset<CUPHY_C_64F, pinned_alloc>((*input_file).open_dataset("Xtf"));
    int num_REs = re_mapped_ref_data.layout().dimensions()[0];
    int num_output_layers = re_mapped_ref_data.layout().dimensions()[2];

    // Allocate device buffer for final result
    tensor_device re_mapped_dmrs(CUPHY_C_16F,
                                 num_REs, OFDM_SYMBOLS_PER_SLOT, num_output_layers,
                                 cuphy::tensor_flags::align_tight);

    int output_bytes = re_mapped_dmrs.desc().get_size_in_bytes();
    // Clear output buffer.
    CUDA_CHECK(cudaMemset(re_mapped_dmrs.addr(), 0, output_bytes));

    cuphyPdschStatPrms_t static_params;
    cuphy::read_pdsch_static_pars_from_file(static_params, *input_file, argv[1], true, false);

    cuphyTensorPrm_t output_tensor_prm;
    output_tensor_prm.desc  = re_mapped_dmrs.desc().handle();
    output_tensor_prm.pAddr = re_mapped_dmrs.addr();
    cuphyPdschDataOut_t output_data = {&output_tensor_prm}; // single cell

    std::vector<cuphyPdschCellGrpDynPrm_t> pdsch_cell_grp_dyn_params(1); //hardcoded to 1 cell group
    cuphy::read_cell_group_dynamic_pars_from_file(pdsch_cell_grp_dyn_params, *input_file);
    cuphyPdschDynPrms_t dyn_params = {0, 0, pdsch_cell_grp_dyn_params.data(), nullptr, nullptr, nullptr, &output_data, nullptr};
/*
    cuphyPdschDataOut_t output_data = {new cuphyTensorPrm_t[1]};
    cuphyPdschDynPrms_t dyn_params = {0, 0, pdsch_cell_grp_dyn_params.data(), nullptr, nullptr, &output_data};
    //Update output addressas it is used in cuphyUpdatePdschDmrsParams()
    dyn_params.pDataOut->pTDataTx[0].pAddr = re_mapped_dmrs.addr();
*/

    h_dmrs_params.resize(dyn_params.pCellGrpDynPrm->nCws);
    cuphyStatus_t status = cuphyUpdatePdschDmrsParams(h_dmrs_params.data(), &dyn_params, &static_params, nullptr);
    if (status != CUPHY_STATUS_SUCCESS) {
        printf("cuphyUpdatePdschDmrsParams error!\n");
        throw cuphy::cuphy_exception(status);
    }
    int num_BWP_PRBs = static_params.pCellStatPrms[0].nPrbDlBwp;
    cuphy::pdsch_params_cleanup(static_params, pdsch_cell_grp_dyn_params);

    int num_TBs = h_dmrs_params.size();
    unique_device_ptr<PdschDmrsParams> d_params = make_unique_device<PdschDmrsParams>(num_TBs);
    CUDA_CHECK(cudaMemcpy(d_params.get(), &h_dmrs_params[0], num_TBs * sizeof(PdschDmrsParams), cudaMemcpyHostToDevice));

    // Allocate launch config struct.
    std::unique_ptr<cuphyPdschDmrsLaunchConfig> dmrs_hndl = make_unique<cuphyPdschDmrsLaunchConfig>();

    // Allocate descriptors and setup PDSCH DMRS component
    uint8_t desc_async_copy = 1; // Copy descriptor to the GPU during setup

    size_t desc_size=0, alloc_size=0;
    status = cuphyPdschDmrsGetDescrInfo(&desc_size, &alloc_size);
    if (status != CUPHY_STATUS_SUCCESS) {
        printf("cuphyPdschDmrsGetDescrInfo error!\n");
        throw cuphy::cuphy_exception(status);
    }
    cuphy::unique_device_ptr<uint8_t> d_dmrs_desc = make_unique_device<uint8_t>(desc_size);
    cuphy::unique_pinned_ptr<uint8_t> h_dmrs_desc = make_unique_pinned<uint8_t>(desc_size);

    status = cuphySetupPdschDmrs(dmrs_hndl.get(),
                                 d_params.get(),
                                 num_TBs,
                                 dyn_params.pCellGrpDynPrm->nPrecodingMatrices > 0,
                                 re_mapped_dmrs.desc().handle(),
                                 re_mapped_dmrs.addr(),
                                 h_dmrs_desc.get(),
                                 d_dmrs_desc.get(),
                                 desc_async_copy,
                                 strm);
    if (status != CUPHY_STATUS_SUCCESS) {
        throw std::runtime_error("Invalid argument(s) for cuphySetupPdschDmrs");
    }

    // Run PDSCH DRMS
    CUresult r = launch_kernel(dmrs_hndl.get()->m_kernelNodeParams, strm);
    if (r != CUDA_SUCCESS) {
        throw std::runtime_error("Invalid argument(s) for PDSCH DMRS kernel");
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    typed_tensor<CUPHY_C_16F, pinned_alloc> h_re_mapped_dmrs_tensor(re_mapped_dmrs.layout());
    h_re_mapped_dmrs_tensor = re_mapped_dmrs;
    if ((num_REs / CUPHY_N_TONES_PER_PRB) != num_BWP_PRBs) {
        NVLOGE_FMT(NVLOG_PDSCH, AERIAL_CUPHY_EVENT,  "Error! # REs in Xtf dataset frequency dimensions are {} but BWP PRBs {}", num_REs, num_BWP_PRBs);
        exit(1);
    }

    uint32_t re_mapped_gpu_mismatch = 0;
    uint32_t checked_elements = 0;
    const __half tolerance = __float2half(0.0001f); //update comparison tolerance as needed.

    for (int TB_id = 0; TB_id < num_TBs; TB_id++) {
        int start_Rb = h_dmrs_params[TB_id].start_Rb;
        int num_Rbs = h_dmrs_params[TB_id].num_Rbs;

        std::vector<int> dmrs_positions(h_dmrs_params[TB_id].num_dmrs_symbols); // Holds DMRS positions 0-based indexing
        uint16_t bitmask = h_dmrs_params[TB_id].dmrs_sym_loc;
        for (int j = 0; j < dmrs_positions.size(); j++) {
            dmrs_positions[j] = (bitmask >> (4 * j)) & 0xF;
            //printf("DMRS position[%d] =%d\n", j, dmrs_positions[j]);
        }

        int numLayers = h_dmrs_params[TB_id].enablePrcdBf ? h_dmrs_params[TB_id].Np : h_dmrs_params[TB_id].num_layers;

        //FIXME Only checks allocated PRBs have valid values. Could also ensure it doesn't overwrite anything else.
        for (int freq_idx = start_Rb * CUPHY_N_TONES_PER_PRB; freq_idx < CUPHY_N_TONES_PER_PRB * (start_Rb + num_Rbs); freq_idx++) {
            for (int dmrs_symbol = 0; dmrs_symbol < h_dmrs_params[TB_id].num_dmrs_symbols; dmrs_symbol++)  {
                int symbol_id = dmrs_positions[dmrs_symbol];
                for (int tmp_layer = 0; tmp_layer < numLayers; tmp_layer++) {
                    int layer_id = tmp_layer;
                    if(!h_dmrs_params[TB_id].enablePrcdBf)
                    {                        
                        layer_id = h_dmrs_params[TB_id].port_ids[tmp_layer] + 8 * h_dmrs_params[TB_id].n_scid;
                    }
                    int delta =  (h_dmrs_params[TB_id].port_ids[tmp_layer] >> 1) & 0x1U;
                    // If dmrsCmdGrpsNoData is set to 1 for this TB, only check half the REs per PRB for this DMRS only example, as the rest will contain data symbols.
                    // We know that the delta can only be 0 in that case (port can be only 0 or 1) but computing it regardless.
                    if (((h_dmrs_params[TB_id].dmrsCdmGrpsNoData1 == 1) && (freq_idx % 2 == delta)) ||
                        (h_dmrs_params[TB_id].dmrsCdmGrpsNoData1 == 0))
                    {
                        __half2 gpu_symbol = h_re_mapped_dmrs_tensor(freq_idx, symbol_id, layer_id);
                        __half2 ref_symbol;
                        ref_symbol.x = (half) re_mapped_ref_data(freq_idx, symbol_id, layer_id).x;
                        ref_symbol.y = (half) re_mapped_ref_data(freq_idx, symbol_id, layer_id).y;
                        checked_elements += 1;

                        if (!complex_approx_equal<__half2, __half>(gpu_symbol, ref_symbol, tolerance)) {
                            /*printf("Error! TB %d, start_PRB %d, Mismatch for (freq. bin %d, symbol %d, layer_id %d) - expected=%f + i %f vs. gpu=%f + i %f\n",
                                TB_id, start_Rb,
                                freq_idx, symbol_id, layer_id,
                                (float) ref_symbol.x, (float) ref_symbol.y,
                                (float) gpu_symbol.x, (float) gpu_symbol.y);*/
                            re_mapped_gpu_mismatch += 1;
                        }
                    }
                }
            }
        }
    }

    std::cout << "Found " << re_mapped_gpu_mismatch << " mismatched RE mapped DMRS symbols out of " << checked_elements << std::endl;

    if (re_mapped_gpu_mismatch != 0) {
        NVLOGE_FMT(NVLOG_PDSCH, AERIAL_CUPHY_EVENT,  "Reference check failed!");
        exit(1);
    }

    return 0;
}

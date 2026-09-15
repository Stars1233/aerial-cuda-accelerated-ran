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
#include "cuphy_api.h"
#include "cuphy.hpp"
#include "cuphy_hdf5.hpp"
#include "hdf5hpp.hpp"
#include "util.hpp"
#include "utils.cuh"
#include "cuphy_internal.h"

using namespace std;
using namespace cuphy;


void usage() {

    std::cout << "dl_rate_matching [options]" << std::endl;
    std::cout << "  Options:" << std::endl;
    std::cout << "     -h                              (Display usage information)" << std::endl;
    std::cout << "     input_filename  num_iterations  (Input HDF5 filename, Number of iterations)" << std::endl;


    std::cout << std::endl;
    std::cout << "  Examples:" << std::endl;
    std::cout << "      ./dl_rate_matching ~/input_file.h5 20" << std::endl;
    std::cout << "      ./dl_rate_matching -h" << std::endl;
}

void print_TB_config_params(std::vector<PdschPerTbParams> & kernel_params, PdschDmrsParams* dmrs_params, int TB_id,
                            bool layer_mapping, bool scrambling) {

    int num_TBs = kernel_params.size();

    // Current codebase expects these config. params to be the same across TBs.
    if (TB_id == 0) {
        std::cout << "Config. Parameters shared across all " <<  num_TBs << " TB(s):" << std::endl;

        std::cout << "* layer_mapping is " << layer_mapping << std::endl;
        std::cout << "* scrambling is " << scrambling << std::endl;
    }

    // Config. parameters that vary across TBs.
    std::cout << std::endl;
    std::cout << "Config. Parameters specific to TB " << TB_id << ": " << std::endl;

    std::cout << "* rv = " << (int)kernel_params[TB_id].rv << std::endl;
    std::cout << "* Qm = " << (int)kernel_params[TB_id].Qm << std::endl;
    std::cout << "* bg = " << (int)kernel_params[TB_id].bg << std::endl;
    std::cout << "* Nl = " << (int)kernel_params[TB_id].Nl << std::endl;
    std::cout << "* num_CBs = " << kernel_params[TB_id].num_CBs << std::endl;
    std::cout << "* Zc = " << kernel_params[TB_id].Zc << std::endl;

    std::cout << "* N = " << kernel_params[TB_id].N << std::endl;
    //std::cout << "Ncb = " << kernel_params[TB_id].Ncb << std::endl;
    std::cout << "* G = " << kernel_params[TB_id].G << std::endl;
    std::cout << "* max REs = " << kernel_params[TB_id].max_REs << std::endl;
    std::cout << "* K = " << kernel_params[TB_id].K << std::endl;
    std::cout << "* F = " << kernel_params[TB_id].F << std::endl;

    std::cout << "* cinit = " << kernel_params[TB_id].cinit << std::endl;
    int TB_layers = kernel_params[TB_id].Nl;
    std::cout << "* layer_map[" << TB_layers << "] = {";
    for (int layer_cnt = 0; layer_cnt < TB_layers; layer_cnt++) {
        std::cout << dmrs_params[TB_id].port_ids[layer_cnt] + 8 * dmrs_params[TB_id].n_scid;
        if (layer_cnt != TB_layers - 1) {
            std::cout << ", ";
        } else {
            std::cout << "}" << std::endl;
        }
    }
}


int main(int argc, char* argv[]) {

    using tensor_pinned_R_64F = typed_tensor<CUPHY_R_64F, pinned_alloc>;

    const int ELEMENT_SIZE = sizeof(uint32_t) * 8; // 32 bits
    cudaStream_t strm = 0; // update as needed
    
    cuphyNvlogFmtHelper nvlog_fmt("dl_rate_matching.log");

    if ((argc != 3) || ((argc == 2) && (argv[1][0] == '-') && (argv[1][1] == 'h'))) {
        usage();
        exit(1);
    }

    int num_iterations = stoi(argv[2]);
    if (num_iterations <= 0) {
        NVLOGE_FMT(NVLOG_PDSCH, AERIAL_CUPHY_EVENT,  "Invalid number of iterations: {}. Should be > 0", num_iterations);
        exit(1);
    }

    // Read from HDF5 input file.
    std::string hdf5_filename = argv[1];

    hdf5hpp::hdf5_file input_file = hdf5hpp::hdf5_file::open(hdf5_filename.c_str());

    bool use_new_api = input_file.is_valid_dataset("ue_pars");
    if (!use_new_api) {
        NVLOGE_FMT(NVLOG_PDSCH, AERIAL_CUPHY_EVENT,  "DL Rate Matching component example no longer supports old API TVs");
        exit(1);
    }

    std::vector<PdschPerTbParams> kernel_params;
    int num_TBs = 1, num_layers = 0; // will be updated later
    cuphyStatus_t params_status;

    cuphyPdschStatPrms_t static_params;
    cuphy::read_pdsch_static_pars_from_file(static_params, input_file, argv[1], true, false);

    std::vector<cuphyPdschCellGrpDynPrm_t> pdsch_cell_grp_dyn_params(1); //hardcoded to 1 cell group
    cuphy::read_cell_group_dynamic_pars_from_file(pdsch_cell_grp_dyn_params, input_file);
    cuphyPdschDynPrms_t dyn_params = {0, 0, pdsch_cell_grp_dyn_params.data(), nullptr, nullptr};

    num_TBs = dyn_params.pCellGrpDynPrm->nCws;
    kernel_params.resize(num_TBs);

    cuphy::unique_pinned_ptr<PdschDmrsParams> h_dmrs_params = make_unique_pinned<PdschDmrsParams>(num_TBs);

    // One option would be to compute Xtf_re_map from all CSI-RS params, call cuphyUpdatePdschDmrsParams, and then call cuphySetTBParamsFromStructs to compute the correct G value per TB.
    // Alternatively, use the dataset size to update the G field, in case CSI-RS is present.
    // Currently doing the latter, as this standalone example is not meant to be used with CSI-RS.

    // If this TB belongs to a cell in TM, then cuphySetTBParamsFromStructs also overwrites num_CBs and N, so we still respect the MAX_ENCODED_CODE_BLOCK_BIT_SIZE constraint
    params_status = cuphySetTBParamsFromStructs(kernel_params.data(), dyn_params.pCellGrpDynPrm, &static_params);
    cuphyPdschUePrm_t* UE_params = dyn_params.pCellGrpDynPrm->pUePrms;
    for (int TB_id = 0; TB_id < num_TBs; TB_id++) {
        std::string dataset_name = "tb" + std::to_string(TB_id) + "_ratematcbs";
        auto dset = input_file.open_dataset(dataset_name.c_str());
        auto info = cuphy::get_HDF5_dataset_info(dset);
        kernel_params[TB_id].G = info.layout().dimensions()[0];

        // Only populate necessary fields for PdschDmrsParams: i.e., port_ids and n_scid
        uint32_t dmrs_ports_bitmask = UE_params[TB_id].dmrsPortBmsk;
        h_dmrs_params.get()[TB_id].n_scid =  UE_params[TB_id].scid;
        for (int i = 0; i < UE_params[TB_id].nUeLayers; i++) {
            h_dmrs_params.get()[TB_id].port_ids[i] = __builtin_ctz(dmrs_ports_bitmask);
            dmrs_ports_bitmask ^= (1 << h_dmrs_params.get()[TB_id].port_ids[i]);
            printf("TB %d has scid %d and port %d\n", TB_id,  UE_params[TB_id].scid, h_dmrs_params.get()[TB_id].port_ids[i]);
        }
        // Initialize tbSize as it will be needed in case of TM (testing mode)
        kernel_params[TB_id].tbSize = dyn_params.pCellGrpDynPrm->pCwPrms[TB_id].tbSize; // in bytes

        // If this TB belongs to a cell in TM, then cuphySetTBParamsFromStructs overwrote num_CBs and N, so we still respect the MAX_ENCODED_CODE_BLOCK_BIT_SIZE constraint
    }

    cuphy::pdsch_params_cleanup(static_params, pdsch_cell_grp_dyn_params);

    if (params_status != CUPHY_STATUS_SUCCESS) {
        throw std::runtime_error("Error when setting TB config parameters!");
    }

    int max_Emax = PDSCH_MAX_ER_PER_CB_BITS; //(kernel_params[0].testModel == 0) ? 24736 : MAX_ENCODED_CODE_BLOCK_BIT_SIZE;
    int N_max = kernel_params[0].N; // OK even in TM since N has been updated above
    uint32_t Cmax = kernel_params[0].num_CBs; // Could alternatively overprovision w/ MAX_N_CBS_PER_TB_SUPPORTED
    for (int i = 1; i < num_TBs; i++) {
        N_max = std::max(N_max, (int) kernel_params[i].N);
        Cmax = std::max(Cmax, kernel_params[i].num_CBs);
    }
    bool scrambling = true; // Set to true or false as needed.
    bool layer_mapping = true; // Set to true or false as needed. If layer_mapping is true, then scrambling also has to be true.

    uint32_t Emax = 0;

    // Allocate workspace and copy config params
    size_t allocated_workspace_size = cuphyDlRateMatchingWorkspaceSize(num_TBs);
    unique_device_ptr<uint32_t> config_workspace = make_unique_device<uint32_t>(div_round_up<uint32_t>(allocated_workspace_size, sizeof(uint32_t)));
    unique_pinned_ptr<uint32_t> h_workspace = make_unique_pinned<uint32_t>((2 + 2) * num_TBs);

    cuphy::unique_device_ptr<PdschPerTbParams> d_tbPrmsArray = make_unique_device<PdschPerTbParams>(num_TBs);
    CUDA_CHECK(cudaMemcpyAsync(d_tbPrmsArray.get(), kernel_params.data(), sizeof(PdschPerTbParams) * num_TBs, cudaMemcpyHostToDevice, strm));

    // Allocate input and overpovisioned output buffers.
    tensor_device d_in_tensor = tensor_device(CUPHY_BIT, N_max, Cmax, num_TBs);

    // Use MAX_N_BBU_LAYERS_SUPPORTED in the buffer allocation instead of num_layers. It is possible to have one TB mapping to layers {1, 2, 3},
    // which would make num_layers be 3 even though we'd need to allocate at least 4 layers.
    size_t max_output_elements = (layer_mapping) ? div_round_up<uint32_t>(num_TBs * MAX_N_BBU_LAYERS_SUPPORTED * Cmax * max_Emax, ELEMENT_SIZE) : div_round_up<uint32_t>(num_TBs * Cmax * max_Emax, ELEMENT_SIZE);
    std::vector<uint32_t> h_rate_matching_output(max_output_elements);
    unique_device_ptr<uint32_t> d_rate_matching_output = make_unique_device<uint32_t>(max_output_elements);

    // Allocate launch config struct.
    std::unique_ptr<cuphyDlRateMatchingLaunchConfig> rm_hndl = make_unique<cuphyDlRateMatchingLaunchConfig>();

    // Allocate descriptors and setup rate matching component
    uint8_t desc_async_copy = 1; // Copy descriptor to the GPU during setup

    size_t desc_size=0, alloc_size=0;
    cuphyStatus_t status = cuphyDlRateMatchingGetDescrInfo(&desc_size, &alloc_size);
    if (status != CUPHY_STATUS_SUCCESS) {
        printf("cuphyDlRateMatchingGetDescrInfo error %d\n", status);
    }
    cuphy::unique_device_ptr<uint8_t> d_rm_desc = make_unique_device<uint8_t>(desc_size);
    cuphy::unique_pinned_ptr<uint8_t> h_rm_desc = make_unique_pinned<uint8_t>(desc_size);

    cuphy::unique_device_ptr<PdschDmrsParams> d_dmrs_params = make_unique_device<PdschDmrsParams>(num_TBs);
    CUDA_CHECK(cudaMemcpyAsync(d_dmrs_params.get(), h_dmrs_params.get(), num_TBs * sizeof(PdschDmrsParams), cudaMemcpyHostToDevice, strm));

    // Used in PDSCH channel
    cuphyPdschStatusOut_t output_status = {cuphyPdschStatusType_t::CUPHY_PDSCH_STATUS_SUCCESS_OR_UNTRACKED_ISSUE, MAX_UINT16, MAX_UINT16};

    status = cuphySetupDlRateMatching(rm_hndl.get(),
                                      &output_status,
                                      (const uint32_t*)d_in_tensor.addr(),
		                      d_rate_matching_output.get(),
                                      nullptr,
                                      nullptr,
                                      nullptr, /* Xtf RE map */
                                      273, /* max PRBs for Xtf RE map */
                                      num_TBs,
                                      0,
                                      scrambling,
                                      layer_mapping,
                                      false,
                                      0, // no precoding
                                      false,
                                      false,
                                      false,
                                      h_workspace.get(), // h_workspace
                                      config_workspace.get(), // d_workspace - Explicit H2D copy as part of setup
                                      kernel_params.data(), // h_params
                                      d_tbPrmsArray.get(), // d_params - No H2D copy. FIXME make this configurable?
                                      d_dmrs_params.get(),
                                      nullptr,
                                      h_rm_desc.get(),
                                      d_rm_desc.get(),
                                      desc_async_copy,
                                      strm);

    if (status != CUPHY_STATUS_SUCCESS) {
        throw std::runtime_error("Invalid argument(s) for cuphySetupDlRateMatching");
    }

    uint32_t* Er = (uint32_t*)h_workspace.get() + 2 * num_TBs;
    for (int i = 0; i < num_TBs; i++) { //FIXME this was already computed in setup, could just copy back that value directly
         uint32_t Er_CB = (kernel_params[i].testModel == 0) ? Er[i * 2 + 1] + (((kernel_params[i].num_CBs - 1) < Er[i * 2]) ? 0 : kernel_params[i].Nl * kernel_params[i].Qm) : Er[i*2+1];
         Emax = std::max(Emax, Er_CB);
    }
    Emax = div_round_up<uint32_t>(Emax, ELEMENT_SIZE) * ELEMENT_SIZE;
    if (Emax > max_Emax) {
        NVLOGE_FMT(NVLOG_PDSCH, AERIAL_CUPHY_EVENT,  "Emax {} but supported max Emax is {}", Emax, max_Emax);
        throw std::runtime_error("Emax exceeds max supported! Update max_Emax in this example.");
    }
    size_t output_elements = (layer_mapping) ? div_round_up<uint32_t>(num_TBs * MAX_N_BBU_LAYERS_SUPPORTED * Cmax * Emax, ELEMENT_SIZE) : div_round_up<uint32_t>(num_TBs * Cmax * Emax, ELEMENT_SIZE);

    // Read and prep input. Currently each HDF5 element in tb*_codedbcbs is a double.
    // Rate matching expects bits packed into uint32_t elements
    //tensor_device d_in_tensor = (tensor_info(CUPHY_BIT, {N_max, (int) Cmax, num_TBs}));
    std::vector<tensor_device> single_TB_d_in_tensor(num_TBs);
    int single_TB_in_tensor_bytes = d_in_tensor.desc().get_size_in_bytes() / num_TBs;

    for (int TB_id = 0; TB_id < num_TBs; TB_id++) {
        int num_CBs = kernel_params[TB_id].num_CBs; // OK even in TM as num_CBs per TB has been properly updated
        uint32_t in_TB_offset = TB_id * single_TB_in_tensor_bytes;
        int N = kernel_params[TB_id].N;
        int last_CB_N = ((kernel_params[TB_id].testModel != 0) && (num_CBs != 1)) ?  (kernel_params[TB_id].G % N) : N;

        int rounded_N = round_up_to_next(N, 32);
        single_TB_d_in_tensor[TB_id] = tensor_device((void*)((uint8_t*)d_in_tensor.addr() + in_TB_offset),
                                                     CUPHY_BIT, rounded_N, num_CBs);

        std::string input_dataset_name = "tb" + std::to_string(TB_id) + ((kernel_params[TB_id].testModel == 0) ? "_codedcbs" : "_inputdata");
        auto input_data = typed_tensor_from_dataset<CUPHY_R_8U, pinned_alloc>(input_file.open_dataset(input_dataset_name.c_str()));
        typed_tensor<CUPHY_BIT, pinned_alloc> single_TB_h_in_tensor(single_TB_d_in_tensor[TB_id].layout());
        for (int CB = 0; CB < num_CBs; CB++) {
            int TM_offset = (kernel_params[TB_id].testModel == 0) ? 0 : CB * N;
            if ((kernel_params[TB_id].testModel != 0) && (num_CBs != 1) && (CB == num_CBs - 1)) { // N for last CB in TB should be updated
                N = last_CB_N;
            }
            for (int element_start = 0; element_start < N; element_start += ELEMENT_SIZE)  {
                uint32_t bits = 0;
                for (int offset = 0; (offset < ELEMENT_SIZE) && (element_start + offset < N); offset++) {
                    uint32_t bit = ((kernel_params[TB_id].testModel == 0) ? (input_data(element_start + offset, CB) == 1) : (input_data(element_start+offset+TM_offset, 0) == 1))? 1 : 0;
		    bits |= (bit << offset);
                 }
                 single_TB_h_in_tensor(element_start / ELEMENT_SIZE, CB) = bits;
            }
        }
        CUDA_CHECK(cudaMemcpy(single_TB_d_in_tensor[TB_id].addr(), single_TB_h_in_tensor.addr(),
                              single_TB_h_in_tensor.desc().get_size_in_bytes(), cudaMemcpyHostToDevice));

        print_TB_config_params(kernel_params, h_dmrs_params.get(), TB_id, layer_mapping, scrambling);
    }

    CUDA_CHECK(cudaDeviceSynchronize());


    // Launch rate matching kernel
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float time1 = 0.0;
    CUresult res = CUDA_SUCCESS;
    cudaEventRecord(start);

    for (int iter = 0; ((iter < num_iterations) && (res == CUDA_SUCCESS)); iter++) {

        res = launch_kernel_ex(rm_hndl.get()->m_kernelNodeParams[0], strm, true);
    }

    if (res != CUDA_SUCCESS) {
        NVLOGE_FMT(NVLOG_PDSCH, AERIAL_CUPHY_EVENT,  "Launching kernel returned {}", res);
        exit(1);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time1, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    time1 /= num_iterations;
    printf("\nDL Rate Matching Kernel: %.2f us (avg. over %d iterations)\n", time1 * 1000, num_iterations);

    CUDA_CHECK(cudaMemcpy(h_rate_matching_output.data(), d_rate_matching_output.get(), output_elements * sizeof(uint32_t), cudaMemcpyDeviceToHost));

    // Compare rate-matching's output w/ reference output
    int ref_bit = 0;
    unsigned long long error_cnt = 0;
    std::string dataset = (!layer_mapping) ? (scrambling ? "_scramcbs" : "_ratematcbs") : "_layer_mapped";

    uint32_t total_ref_bits = 0;
    for (int TB_id = 0; TB_id < num_TBs; TB_id++) {
        int TB_num_layers = (!layer_mapping) ? 1 : kernel_params[TB_id].Nl;
        int ref_bit = 0;

        std::string ref_dataset_name = "tb" + std::to_string(TB_id) + dataset;
        auto ref_data = typed_tensor_from_dataset<CUPHY_R_8U, pinned_alloc>(input_file.open_dataset(ref_dataset_name.c_str()));


        for (int layer_cnt = 0; layer_cnt < TB_num_layers; layer_cnt++) {
            int layer_or_TB_id = layer_mapping ? h_dmrs_params.get()[TB_id].port_ids[layer_cnt] + 8 * h_dmrs_params.get()[TB_id].n_scid : TB_id;
            for (int CB = 0; CB < kernel_params[TB_id].num_CBs; CB++) {
                uint32_t Er_CB = Er[TB_id * 2 + 1] + ((CB < Er[TB_id * 2]) ? 0 : kernel_params[TB_id].Nl * kernel_params[TB_id].Qm);
                if ((kernel_params[TB_id].testModel != 0) && (CB == kernel_params[TB_id].num_CBs-1)) {
                    Er_CB = kernel_params[TB_id].G - CB*kernel_params[TB_id].N; // Should not use tbSize*8 instead of G as TB size in bits may not be evenly divisible by 8
                }
                for (int Er_bit = 0; Er_bit < Er_CB/TB_num_layers; Er_bit++) {
                    uint32_t ref_value = ref_data(ref_bit, 0);

                    int out_index = layer_or_TB_id * Cmax *Emax + CB * Emax + Er_bit;
                    if (layer_mapping) {
                        out_index = layer_or_TB_id * num_TBs * Cmax *Emax + TB_id * Cmax * Emax + CB * Emax + Er_bit;
                    }
                    int out_word = out_index / ELEMENT_SIZE;
                    int out_bits = out_index % ELEMENT_SIZE;
                    uint32_t computed_value = (h_rate_matching_output[out_word] >> out_bits) & 0x1;
                    if (ref_value != computed_value) {
                        error_cnt += 1;
                        //NVLOGE_FMT(NVLOG_PDSCH, AERIAL_CUPHY_EVENT,  "ERROR: GPU vs. reference output mismatch");
                        //NVLOGE_FMT(NVLOG_PDSCH, AERIAL_CUPHY_EVENT,  "TB {}, Layer {}, CB {}, Er bit {}: computed value {} vs. reference {}", TB_id, layer_or_TB_id, CB, Er_bit, computed_value, ref_value);
                    }
                    ref_bit += 1;
                    total_ref_bits += 1;
                }
            }
        }
    }

    std::cout << std::endl << "Rate Matching Error Count: " << error_cnt << " bits out of " << total_ref_bits;
    std::cout << "; GPU output compared w/ reference dataset <tb*" << dataset << "> from <" << hdf5_filename << ">" << std::endl;

    if (error_cnt != 0) {
        NVLOGE_FMT(NVLOG_PDSCH, AERIAL_CUPHY_EVENT,  "Reference check failed!");
        exit(1);
    }

    return 0;
}

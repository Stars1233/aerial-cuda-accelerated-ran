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
//#define _READ_TB_CRC_ 1

#include "memtrace.h"
#include "nvlog.hpp"


using namespace cuphy;
#define SPECIAL_SLOT 0 // TODO: If set to 1, update the TV to be used during the special slot accordingly. See special_slot_file_name. For now it's the same TV.
#define REF_CHECKS_DURING_TIMING 0
#define TEST_SETUP_ONCE_FALLBACK 0 //TODO: Set to 1, to test setup-once fallback for back to back runs of the same TV. Do not change anything else.
#define PDSCH_TB_INPUT_ON_GPU 0 //FIXME set to 1 if you want to test the TB buffers in GPU mode for cuPHY standalone


/**
 *  @brief Print usage information for the DL pipeline example.
 */
void usage()
{
    std::cout << "DEPRECATION NOTICE: pdsch_tx example is deprecated and will be removed in a future release. Please use the pdsch_tx_multi_cell example instead, a superset of pdsch_tx, which now supports both single-cell and multi-cell tests" << std::endl;

    std::cout << "cuphy_ex_pdsch_tx [options]" << std::endl;
    std::cout << "  Options:" << std::endl;
    std::cout << "     -h                Display usage information" << std::endl;
    std::cout << "     input_filename  num_iterations  AAS_mode graphs_mode (Input HDF5 filename, Number of iterations, Enable AAS mode, Enable graphs mode, TB byte alignment)" << std::endl;

    std::cout << std::endl;
    std::cout << "  Examples:" << std::endl;
    std::cout << "      ./cuphy_ex_pdsch_tx -h" << std::endl;
    std::cout << "      ./cuphy_ex_pdsch_tx ~/dl_pipeline.h5 20 0 1" << std::endl;
    std::cout << "      ./cuphy_ex_pdsch_tx ~/dl_pipeline.h5 20 1 0" << std::endl;
}

int main(int argc, char* argv[]) {

    char nvlog_yaml_file[1024];
    // Relative path from binary to default nvlog_config.yaml
    std::string relative_path = std::string("../../../../").append(NVLOG_DEFAULT_CONFIG_FILE);
    nv_get_absolute_path(nvlog_yaml_file, relative_path.c_str());
    pthread_t log_thread_id = nvlog_fmtlog_init(nvlog_yaml_file, "pdsch_tx.log",NULL);
    nvlog_fmtlog_thread_init();

    if (((argc != 6) && (argc != 5) && (argc != 4)) || ((argc == 2) && (argv[1][0] == '-') && (argv[1][1] == 'h'))) {
        usage();
        exit(1);
    }

    int num_iterations = std::stoi(argv[2]);
    if(num_iterations <= 0)
    {
        NVLOGF_FMT(NVLOG_PDSCH, AERIAL_CUPHY_EVENT, "Invalid number of iterations: {}. Should be > 0.", num_iterations);
    }

    int cfg_aas_mode = std::stoi(argv[3]);
    if(cfg_aas_mode < 0)
    {
        NVLOGC_FMT(NVLOG_PDSCH, "Negative AAS mode value treated as 0.");
    }
    else if(cfg_aas_mode > 1)
    {
        NVLOGC_FMT(NVLOG_PDSCH, "AAS mode value > 1 treated as 1.");
    }

    int cfg_graphs_mode = 0;
    if (argc == 5) {
        cfg_graphs_mode = std::stoi(argv[4]);
        if (cfg_graphs_mode < 0) {
            NVLOGC_FMT(NVLOG_PDSCH, "Negative graphs mode value treated as 0.");
            cfg_graphs_mode = 0;
        } else if (cfg_graphs_mode > 1) {
            NVLOGC_FMT(NVLOG_PDSCH, "graphs mode value > 1 treated as 1.");
            cfg_graphs_mode = 1;
        }
    }
    bool graphs_mode = (cfg_graphs_mode == 1)? true : false;
    std::string graphs_streams_mode_string = (graphs_mode) ? "Graphs" : "Streams";

    int forced_TB_byte_alignment = 1; // 1 byte alignment by default
    if (argc == 6) {
        forced_TB_byte_alignment = std::stoi(argv[5]);
        if((forced_TB_byte_alignment <= 0) || (forced_TB_byte_alignment > 32) || ((forced_TB_byte_alignment & (forced_TB_byte_alignment -1)) != 0)) {
            NVLOGF_FMT(NVLOG_PDSCH, AERIAL_CUPHY_EVENT,  "ERROR: Invalid TB byte alignment {}. Supported values are 1, 2, 4, 8, 16, 32", forced_TB_byte_alignment);
        }
    }

    // pdsch_tx example does not support the new, experimental, PDSCH_POST_FEC_PROCESSING mode. To use that, please use pdsch_tx_multi_cell example instead. See deprecation notice below:
    NVLOGC_FMT(NVLOG_PDSCH, "DEPRECATION NOTICE: pdsch_tx example is deprecated and will be removed in a future release. Please use the pdsch_tx_multi_cell example instead, a superset of pdsch_tx, which now supports both single-cell and multi-cell tests");

    CUDA_CHECK(cudaSetDevice(0));
#if 0
    //FIXME tmp code to test green contexts
    CUdevice current_device;
    CU_CHECK(cuDeviceGet(&current_device, 0));

    CUdevResource initial_device_GPU_resources = {};
    CUdevResourceType default_resource_type = CU_DEV_RESOURCE_TYPE_SM; // other alternative is CU_DEV_RESOURCE_TYPE_INVALID
    CUdevResource split_result[2] = {{}, {}};

    CU_CHECK(cuDeviceGetDevResource(current_device, &initial_device_GPU_resources, default_resource_type));
    unsigned int split_groups = 1;
    CU_CHECK(cuDevSmResourceSplitByCount(&split_result[0], &split_groups, &initial_device_GPU_resources, &split_result[1], 0, 102 /* hardcoded */));
    cuphy::cudaGreenContext pdsch_cuda_green_ctx;
    pdsch_cuda_green_ctx.create(0, &split_result[0]);
    pdsch_cuda_green_ctx.bind();

#endif
    memtrace_set_config(MI_MEMTRACE_CONFIG_ENABLE | MI_MEMTRACE_CONFIG_EXIT_AFTER_BACKTRACE);
    bool ref_check = (memtrace_get_config() == 0); // ref. checks should be disabled if memtracing is ongoing as there are allocations during those functions
    memtrace_set_config(0); // disable memory allocation tracing beyond this point

    bool aas_mode  = (cfg_aas_mode >= 1);
    cuphyPdschProcMode_t pdsch_proc_mode = (graphs_mode) ? PDSCH_PROC_MODE_GRAPHS : PDSCH_PROC_MODE_NO_GRAPHS;
#if TEST_SETUP_ONCE_FALLBACK
    // Setting this flag (bitwise or with PDSCH_PROC_MODE_SETUP_ONCE_FALLBACK) is key when running back to back runs of the same test vector with only a single setup.
    // It tells the pipeline to reset all its internal buffers that are involved in atomic operations. Doing so is needed for correctness.
    // In this example, the single setup is the one happing during the initial run with reference checks (i.e., before the timing loop).
    pdsch_proc_mode = static_cast<cuphyPdschProcMode_t>((uint32_t) pdsch_proc_mode | (uint32_t)PDSCH_PROC_MODE_SETUP_ONCE_FALLBACK);
#endif
    std::string pipeline_mode = (aas_mode) ? "AAS" : "non AAS";
    bool identical_LDPC_configs = true; // A runtime check resets LDPC configs to non-identical if they are not.

    // Downlink pipeline includes: (a) CRC, (b) LDPC  encoder, (c) Rate-Matching,
    // (d) Modulation Mapper and (e) DMRS components.
    stream       strm(cudaStreamNonBlocking, PDSCH_STREAM_PRIORITY); // Creating CUDA stream with the same priority as the pool in PdschTx
    cudaStream_t strm_handle = strm.handle();

    std::unique_ptr<hdf5hpp::hdf5_file> input_file = std::make_unique<hdf5hpp::hdf5_file>(hdf5hpp::hdf5_file(hdf5hpp::hdf5_file::open(argv[1])));
#if SPECIAL_SLOT
    // NB: If you choose to change the special slot TV from the default, you need to ensure that the two TVs used (i.e., argv[1] and special_slot_file_name)
    // have the same static parameters (cuphyPdschStatPrms_t). Static parameters are only read during PdschTx pipeline creation.
    // Using a special slot TV with different static parameters will result in reference check mismatches and an internal error.
    std::string                         special_slot_file_name  = argv[1];
    std::unique_ptr<hdf5hpp::hdf5_file> special_slot_input_file = std::make_unique<hdf5hpp::hdf5_file>(hdf5hpp::hdf5_file(hdf5hpp::hdf5_file::open(special_slot_file_name.c_str())));
#endif

    // Large buffer added as cuPHY tools needs a buffer with a power of 2 size.
    // Note: this example assumes a single cell group with a single cell.
    constexpr int                               large_buffer_bytes = 1 << 23;
    unique_device_ptr<cuFloatComplex> large_buffer       = make_unique_device<cuFloatComplex>(large_buffer_bytes / sizeof(cuFloatComplex));
    int num_REs = cuphy::get_HDF5_dataset_info((*input_file).open_dataset("Xtf")).layout().dimensions()[0];
    int num_ports = cuphy::get_HDF5_dataset_info((*input_file).open_dataset("Xtf")).layout().dimensions()[2];
    tensor_device                     data_tx_tensor     = tensor_device(large_buffer.get(), CUPHY_C_16F, num_REs, OFDM_SYMBOLS_PER_SLOT, num_ports, cuphy::tensor_flags::align_tight);


    int data_tx_tensor_bytes = data_tx_tensor.desc().get_size_in_bytes();
    if(data_tx_tensor_bytes > large_buffer_bytes)
    {
        NVLOGF_FMT(NVLOG_PDSCH, AERIAL_CUPHY_EVENT, "Buffer ({} bytes) is smaller than data_tx_tensor ({})", large_buffer_bytes, data_tx_tensor_bytes);
    }
    // Reset output buffer
    CUDA_CHECK(cudaMemsetAsync(data_tx_tensor.addr(), 0, data_tx_tensor_bytes, strm_handle));

    // Confirm that new API TVs are used. Dataset ue_pars only exists in the new API TVs.
    bool use_new_api = input_file->is_valid_dataset("ue_pars");
    if(!use_new_api)
    {
        NVLOGF_FMT(NVLOG_PDSCH, AERIAL_CUPHY_EVENT, "PDSCH examples no longer support old API TVs.");
    }

    std::unique_ptr<cuphyPdschTxHndl_t> pdsch_handle = std::make_unique<cuphyPdschTxHndl_t>();

    // Read static params
    cuphyPdschStatPrms_t pdsch_static_params;
    cuphyTracker_t       pdsch_tracker = {nullptr};
    read_pdsch_static_pars_from_file(pdsch_static_params, *input_file, argv[1], ref_check, identical_LDPC_configs);
    pdsch_static_params.pOutInfo = &pdsch_tracker;

#if SPECIAL_SLOT
    // special_slot_pdsch_static_params not used
    cuphyPdschStatPrms_t special_slot_pdsch_static_params;
    read_pdsch_static_pars_from_file(special_slot_pdsch_static_params, *special_slot_input_file, special_slot_file_name.c_str(), ref_check, identical_LDPC_configs);
#endif

    // Read dynamic group params
    std::vector<cuphyPdschCellGrpDynPrm_t> pdsch_cell_grp_dyn_params(1); //hardcoded to 1 cell group
    read_cell_group_dynamic_pars_from_file(pdsch_cell_grp_dyn_params, *input_file);

    //At this point input parameters have been parsed for this cell. Update tbStartOffset based on alignment
    int num_CWs = pdsch_cell_grp_dyn_params[0].nCws;
    for (int tmp_cw = 0; tmp_cw < num_CWs; tmp_cw++) {
        if (tmp_cw !=  0) {
            pdsch_cell_grp_dyn_params[0].pCwPrms[tmp_cw].tbStartOffset = round_up_to_next<int>(pdsch_cell_grp_dyn_params[0].pCwPrms[tmp_cw-1].tbStartOffset + pdsch_cell_grp_dyn_params[0].pCwPrms[tmp_cw-1].tbSize, forced_TB_byte_alignment);
        }
    }
    int total_cell_TB_buffer_size = (pdsch_cell_grp_dyn_params[0].pCwPrms[num_CWs-1].tbStartOffset + pdsch_cell_grp_dyn_params[0].pCwPrms[num_CWs-1].tbSize);

#if SPECIAL_SLOT
    std::vector<cuphyPdschCellGrpDynPrm_t> special_slot_pdsch_cell_grp_dyn_params(1); //hardcoded to 1 cell group
    read_cell_group_dynamic_pars_from_file(special_slot_pdsch_cell_grp_dyn_params, *special_slot_input_file);
#endif

    int num_cells = pdsch_cell_grp_dyn_params[0].nCells;
    if(num_cells != 1)
    {
        NVLOGF_FMT(NVLOG_PDSCH, AERIAL_CUPHY_EVENT, "Error! # cells = {}, but this example only supports a single cell per pipeline", num_cells);
    }

    //Parse input dataset
    hdf5hpp::hdf5_dataset crc_dataset = (*input_file).open_dataset("InputData");
    using tensor_pinned_R_8U          = typed_tensor<CUPHY_R_8U, pinned_alloc>;
    using tensor_device_R_8U          = typed_tensor<CUPHY_R_8U, device_alloc>;
    using tensor_pinned_R_32U         = typed_tensor<CUPHY_R_32U, pinned_alloc>;
    auto data_in_ptr                  = std::make_unique<uint8_t*[]>(1); // 1 cell
    pdsch_static_params.pipeline_processing_mode = (aas_mode) ? cuphyPdschPipelineMode_t::PDSCH_AAS_PROCESSING : cuphyPdschPipelineMode_t::PDSCH_FULL_PROCESSING;
#if _READ_TB_CRC_
    auto tb_crc_data_in_ptr              = std::make_unique<uint8_t*[]>(1); // 1 cell
    pdsch_static_params.read_TB_CRC      = true;
    hdf5hpp::hdf5_dataset tb_crc_dataset = (*input_file).open_dataset("tbCrcBuffer");
    tensor_pinned_R_32U   tb_crc_data    = typed_tensor_from_dataset<CUPHY_R_32U, pinned_alloc>(tb_crc_dataset, cuphy::tensor_flags::align_default, strm_handle);
    tb_crc_data_in_ptr[0]                = (uint8_t*)tb_crc_data.addr();
    cuphyPdschDataIn_t    tb_crc_data_in = {&tb_crc_data_in_ptr[0], cuphyPdschDataIn_t::CPU_BUFFER};
#else
    pdsch_static_params.read_TB_CRC   = false;
    cuphyPdschDataIn_t tb_crc_data_in = {nullptr, cuphyPdschDataIn_t::CPU_BUFFER};
#endif

#if PDSCH_TB_INPUT_ON_GPU
    tensor_device_R_8U crc_input_data = typed_tensor_from_dataset<CUPHY_R_8U, device_alloc>(crc_dataset, cuphy::tensor_flags::align_default, strm_handle);
    cuphy::unique_device_ptr<uint8_t> padded_crc_input_data = make_unique_device<uint8_t>(total_cell_TB_buffer_size);

    data_in_ptr[0]                    = padded_crc_input_data.get();
    cuphyPdschDataIn_t data_in        = {&data_in_ptr[0], cuphyPdschDataIn_t::GPU_BUFFER};
#else
    tensor_pinned_R_8U crc_input_data = typed_tensor_from_dataset<CUPHY_R_8U, pinned_alloc>(crc_dataset, cuphy::tensor_flags::align_default, strm_handle);
    cuphy::unique_pinned_ptr<uint8_t> padded_crc_input_data = make_unique_pinned<uint8_t>(total_cell_TB_buffer_size);
    data_in_ptr[0]                    = padded_crc_input_data.get();
    cuphyPdschDataIn_t data_in        = {&data_in_ptr[0], cuphyPdschDataIn_t::CPU_BUFFER};
#endif

    // Populate input CPU or GPU buffers based on required padding. Since TVs currently have a 1Byte alignment padding, do separate per-TB memory copies.
    // Padding bytes are all memset to 0.
    int cumulative_offset = 0;
    for (int tmp_cw = 0; tmp_cw < num_CWs; tmp_cw++) {
#if PDSCH_TB_INPUT_ON_GPU
        CUDA_CHECK(cudaMemcpyAsync((uint8_t*)padded_crc_input_data.get() + pdsch_cell_grp_dyn_params[0].pCwPrms[tmp_cw].tbStartOffset, crc_input_data.addr() + cumulative_offset, pdsch_cell_grp_dyn_params[0].pCwPrms[tmp_cw].tbSize, cudaMemcpyDeviceToDevice, strm.handle()));
#else
        CUDA_CHECK(cudaMemcpyAsync((uint8_t*)padded_crc_input_data.get() + pdsch_cell_grp_dyn_params[0].pCwPrms[tmp_cw].tbStartOffset, crc_input_data.addr() + cumulative_offset, pdsch_cell_grp_dyn_params[0].pCwPrms[tmp_cw].tbSize, cudaMemcpyHostToHost, strm.handle()));
#endif
        int padded_bytes = ((tmp_cw == num_CWs-1) ?  total_cell_TB_buffer_size : pdsch_cell_grp_dyn_params[0].pCwPrms[tmp_cw+1].tbStartOffset) - pdsch_cell_grp_dyn_params[0].pCwPrms[tmp_cw].tbStartOffset - pdsch_cell_grp_dyn_params[0].pCwPrms[tmp_cw].tbSize;
        CUDA_CHECK(cudaMemsetAsync((uint8_t*)(padded_crc_input_data.get() + pdsch_cell_grp_dyn_params[0].pCwPrms[tmp_cw].tbStartOffset + pdsch_cell_grp_dyn_params[0].pCwPrms[tmp_cw].tbSize), 0,  padded_bytes, strm.handle()));
        cumulative_offset += pdsch_cell_grp_dyn_params[0].pCwPrms[tmp_cw].tbSize;
    }

    input_file.reset(); // Close the file; will reopen it later

#if SPECIAL_SLOT
    auto special_slot_data_in_ptr                     = std::make_unique<uint8_t*[]>(1);
    hdf5hpp::hdf5_dataset special_slot_crc_dataset    = (*special_slot_input_file).open_dataset("InputData");
    tensor_pinned_R_8U    special_slot_crc_input_data = typed_tensor_from_dataset<CUPHY_R_8U, pinned_alloc>(special_slot_crc_dataset, cuphy::tensor_flags::align_default, strm_handle);
    special_slot_data_in_ptr[0]                       = special_slot_crc_input_data.addr();
    cuphyPdschDataIn_t    special_slot_data_in        = {special_slot_data_in_ptr.get(), cuphyPdschDataIn_t::CPU_BUFFER};
    special_slot_input_file.reset(); // Close the file; will reopen it later
#endif

    cuphyPdschStatusOut_t status_info = {cuphyPdschStatusType_t::CUPHY_PDSCH_STATUS_SUCCESS_OR_UNTRACKED_ISSUE, MAX_UINT16, MAX_UINT16};
    cuphyPdschDataOut_t output_data = {new cuphyTensorPrm_t[num_cells]};

    cuphyPdschDynPrms_t pdsch_dyn_params         = {strm_handle, pdsch_proc_mode, pdsch_cell_grp_dyn_params.data(), &data_in, &tb_crc_data_in, nullptr, &output_data, &status_info};
    pdsch_dyn_params.pDataOut->pTDataTx[0].desc  = data_tx_tensor.desc().handle();
    pdsch_dyn_params.pDataOut->pTDataTx[0].pAddr = data_tx_tensor.addr();

#if SPECIAL_SLOT
    cuphyPdschStatusOut_t special_slot_status_info = {cuphyPdschStatusType_t::CUPHY_PDSCH_STATUS_SUCCESS_OR_UNTRACKED_ISSUE, MAX_UINT16, MAX_UINT16};
    cuphyPdschDynPrms_t special_slot_pdsch_dyn_params         = {strm_handle, pdsch_proc_mode, special_slot_pdsch_cell_grp_dyn_params.data(), &special_slot_data_in, &tb_crc_data_in, nullptr, &output_data, &special_slot_status_info};
    special_slot_pdsch_dyn_params.pDataOut->pTDataTx[0].desc  = data_tx_tensor.desc().handle();
    special_slot_pdsch_dyn_params.pDataOut->pTDataTx[0].pAddr = data_tx_tensor.addr();
#endif

    cuphyStatus_t status = cuphyCreatePdschTx(pdsch_handle.get(), &pdsch_static_params); // Currently calling PdschTx constructor with empty filename
    if(status != CUPHY_STATUS_SUCCESS)
    {
        NVLOGF_FMT(NVLOG_PDSCH, AERIAL_CUPHY_EVENT, "Error! cuphyCreatePdschTx(): {}", cuphyGetErrorString(status));
    }
    const cuphyMemoryFootprint* channel_memory_footprint = reinterpret_cast<const cuphyMemoryFootprint*>(pdsch_static_params.pOutInfo->pMemoryFootprint);
    channel_memory_footprint->printMemoryFootprint(*pdsch_handle.get(), "PDSCH");

    // Enable dynamic memory allocation tracing in real-time code path; only applicable when running with LD_PRELOAD=<PATH to libmimalloc.so>
    memtrace_set_config(MI_MEMTRACE_CONFIG_ENABLE | MI_MEMTRACE_CONFIG_EXIT_AFTER_BACKTRACE);

    status = cuphySetupPdschTx(*pdsch_handle, &pdsch_dyn_params, nullptr);
    if(status != CUPHY_STATUS_SUCCESS)
    {
        if (pdsch_dyn_params.pStatusInfo->status == cuphyPdschStatusType_t::CUPHY_PDSCH_STATUS_UNSUPPORTED_MAX_ER_PER_CB) {
            NVLOGE_FMT(NVLOG_PDSCH, AERIAL_CUPHY_EVENT, "CUPHY_PDSCH_STATUS_UNSUPPORTED_MAX_ER_PER_CB error in cuphySetupPdschTx(): {}. Triggered by TB {} in cell group and cellPrmStatIdx {}",
                       cuphyGetErrorString(status), pdsch_dyn_params.pStatusInfo->ueIdx, pdsch_dyn_params.pStatusInfo->cellPrmStatIdx);
        } else {
            NVLOGE_FMT(NVLOG_PDSCH, AERIAL_CUPHY_EVENT, "Error! cuphySetupPdschTx(): {}", cuphyGetErrorString(status));
        }

        NVLOGF_FMT(NVLOG_PDSCH, AERIAL_CUPHY_EVENT, "Error! cuphySetupPdschTx(): {}", cuphyGetErrorString(status));
    }

    {
        MemtraceDisableScope md; // disable memory allocation
        NVLOGC_FMT(NVLOG_PDSCH, "Running DL pipeline once w/ reference checks enabled in {} and {} mode.", pipeline_mode, graphs_streams_mode_string);
    }

    status = cuphyRunPdschTx(*pdsch_handle, pdsch_proc_mode);
    if(status != CUPHY_STATUS_SUCCESS)
    {
        NVLOGF_FMT(NVLOG_PDSCH, AERIAL_CUPHY_EVENT, "Error! cuphyRunPdschTx(): {}", cuphyGetErrorString(status));
    }
    memtrace_set_config(0); // disable memory allocation tracing beyond this point

#if 0
    typed_tensor<CUPHY_C_16F, pinned_alloc> h_pdsch_out_tensors(data_tx_tensors.layout());
    h_pdsch_out_tensors.convert(data_tx_tensors, strm_handle);
    strm.synchronize();

    // Example to show the layout of the output tensor
    int layer_id = 1;
    int symbol_id = 3;
    int freq_bin = 0;
    NVLOGC_FMT(NVLOG_PDSCH, "Symbol at (freq. bin {}, symbol {}, layer {}) =  ({:f} + i {:f})\n", freq_bin, symbol_id, layer_id,
          (float) h_pdsch_out_tensors(freq_bin, symbol_id, layer_id, 0).x,
          (float) h_pdsch_out_tensors(freq_bin, symbol_id, layer_id, 0).y);
#endif

    strm.synchronize();

    // Time pipeline. Does not time allocations etc.
#if !REF_CHECKS_DURING_TIMING
    ref_check = false;
    updateRefCheck(*pdsch_handle, ref_check);
#endif
#if TEST_SETUP_ONCE_FALLBACK
    // Enable reference checks to confirm correctness.
    updateRefCheck(*pdsch_handle, true);
#endif

    NVLOGC_FMT(NVLOG_PDSCH, "");
    NVLOGC_FMT(NVLOG_PDSCH, "Timing the DL pipeline in {} and {} mode.", pipeline_mode, graphs_streams_mode_string);
    NVLOGC_FMT(NVLOG_PDSCH, "- NB: Allocations not included. Ref. checks will fail!");
    NVLOGC_FMT(NVLOG_PDSCH, "");

    // Time pipeline
    event_timer cuphy_timer;
    cuphy_timer.record_begin(strm);

    for(int iter = 0; iter < num_iterations; iter++)
    {
#if REF_CHECKS_DURING_TIMING || TEST_SETUP_ONCE_FALLBACK
        // Resetting the PdschTx output tensor is the caller's responsibility.
        // It is needed even if only a single PDSCH pipeline is running (and no other channels),
        // as TVs with precoding enabled also perform atomic updates to the output tensor.
        //NVLOGC_FMT(NVLOG_PDSCH, "--------------------- Iter {} ---------------------------", iter);
        CUDA_CHECK(cudaMemsetAsync(data_tx_tensor.addr(), 0, data_tx_tensor_bytes, strm_handle));
#endif
#if SPECIAL_SLOT
        if((iter != 0) && (iter % 3 == 0))
        { // TODO Update how often the special slot is run as needed
            updateFileName(*pdsch_handle, special_slot_file_name.c_str());
#if REF_CHECKS_DURING_TIMING
            NVLOGC_FMT(NVLOG_PDSCH, "--------------------- SPECIAL SLOT  ---------------------------");
            CUPHY_CHECK(cuphySetupPdschTx(*pdsch_handle, &special_slot_pdsch_dyn_params, nullptr));
#endif
            CUPHY_CHECK(cuphyRunPdschTx(*pdsch_handle, pdsch_proc_mode));
        }
        else
        {
            updateFileName(*pdsch_handle, argv[1]);
#if REF_CHECKS_DURING_TIMING
            CUPHY_CHECK(cuphySetupPdschTx(*pdsch_handle, &pdsch_dyn_params, nullptr));
#endif
            CUPHY_CHECK(cuphyRunPdschTx(*pdsch_handle, pdsch_proc_mode));
        }
#else // No special slot
#if REF_CHECKS_DURING_TIMING
        CUPHY_CHECK(cuphySetupPdschTx(*pdsch_handle, &pdsch_dyn_params, nullptr));
#endif
        CUPHY_CHECK(cuphyRunPdschTx(*pdsch_handle, pdsch_proc_mode));
#endif
    }

    cuphy_timer.record_end(strm_handle);
    cuphy_timer.synchronize();
    float time1 = cuphy_timer.elapsed_time_ms();
    time1 /= num_iterations;
    NVLOGC_FMT(NVLOG_PDSCH, "DL pipeline: {:.2f} us (avg. over {} iterations)", time1 * 1000, num_iterations);

    NVLOGC_FMT(NVLOG_PDSCH, "cuPHY PDSCH footprint: {:} bytes.", cuphyGetGpuMemoryFootprintPdschTx(*pdsch_handle));

    // Cleanup
    pdsch_params_cleanup(pdsch_static_params, pdsch_cell_grp_dyn_params);
#if SPECIAL_SLOT
    pdsch_params_cleanup(special_slot_pdsch_static_params, special_slot_pdsch_cell_grp_dyn_params);
#endif
    status = cuphyDestroyPdschTx(*pdsch_handle);
    if(status != CUPHY_STATUS_SUCCESS)
    {
        NVLOGF_FMT(NVLOG_PDSCH, AERIAL_CUPHY_EVENT, "Error! cuphyDestroyPdschTx(): {}", cuphyGetErrorString(status));
    }

    delete[] output_data.pTDataTx;
    nvlog_fmtlog_close(log_thread_id);

    return 0;
}

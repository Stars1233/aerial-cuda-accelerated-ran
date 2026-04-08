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

#include "order_kernel_test_bench.hpp"
#include <fstream>

using namespace order_kernel_tb;

void* order_kernel_tb::allocate_memory(size_t size)
{
  void* ptr= new char[size];
  return ptr;
}

int order_kernel_tb::free_memory(void* ptr)
{
  delete static_cast<char*>(ptr);
  return 0;
}

OrderKernelTestBench::OrderKernelTestBench(
    std::string config_file,
    std::array<std::string, UL_MAX_CELLS_PER_SLOT> &binary_file, std::string launch_pattern_file,
    std::string output_file, uint32_t start_slot, uint32_t num_slots,
    uint32_t num_cells, uint8_t same_slot, uint32_t mps_sm_count, uint32_t gc_sm_count, uint8_t mimo, uint8_t srs_enabled)
    : yaml_parser_(config_file) {
    for(int index=0;index<num_cells;index++)
    {
        binary_file_[index]=binary_file[index];
    }
    config_file_ = config_file;
    output_file_ = output_file;
    launch_pattern_file_ = launch_pattern_file;
    start_test_slot = start_slot;
    num_test_slots = num_slots;
    num_test_cells = num_cells;
    same_test_slot = same_slot;
    num_mps_sms = mps_sm_count;
    num_gc_sms = gc_sm_count;
    gpuId = 0; // Assume GPU device ID is 0
    enable_mimo=mimo;
    enable_srs=srs_enabled;
    if(enable_mimo)
    {
        max_rx_ant=MAX_RX_ANT_PUSCH_PUCCH_PRACH_64T64R;
        num_ant_ports=4; //TODO : Read from config file
        num_ant_ports_prach=2; //TODO : Read from config file
    }
    else
    {
        max_rx_ant=MAX_RX_ANT_4T4R;
    }
    
    // Initialize the dynamic arrays in ok_tb_config_file_params with max_rx_ant size
    for(int cell_idx = 0; cell_idx < UL_MAX_CELLS_PER_SLOT; cell_idx++)
    {
        for(int slot_idx = 0; slot_idx < MAX_UL_SLOTS_OK_TB; slot_idx++)
        {
            ok_tb_config_file_params[cell_idx].pusch_eAxC_map[slot_idx].resize(max_rx_ant, 0);
            ok_tb_config_file_params[cell_idx].prach_eAxC_map[slot_idx].resize(max_rx_ant, 0);
            ok_tb_config_file_params[cell_idx].srs_eAxC_map[slot_idx].resize(MAX_RX_ANT_SRS_64T64R, 0);
        }
    }
    
    slot_info = new slotInfo_t[sizeof(slotInfo_t)*UL_MAX_CELLS_PER_SLOT];
    add_gpu_comm_ready_flags();
    CUDA_CHECK(cudaMallocHost((void **)&ok_tb_config_params, sizeof(orderKernelTbConfigParams_t)));
    initStatus = initialize();
}

int OrderKernelTestBench::initialize()
{
    CUresult cuStatus;

    CUDA_CHECK(cudaSetDevice(gpuId));
    CU_CHECK(cuDeviceGet(&cuDev, gpuId));

    int actualDevSmCount = 0;
    CU_CHECK(cuDeviceGetAttribute(&actualDevSmCount, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, cuDev));

    int32_t gpuMaxSmCount = 0;
    CU_CHECK(cuDeviceGetAttribute(&gpuMaxSmCount, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, cuDev));

    if (num_mps_sms > 0) // using MPS
    {
      //Check num_gc_sms value is in valid range
      if (num_mps_sms > (uint32_t)gpuMaxSmCount)
      {
        NVLOGE_FMT(TAG_ORDER_TB_INIT, AERIAL_TESTBENCH_EVENT,  "Error: Invalid --num_mps_sms argument {}. It is greater than {} (GPU's max SMs).", num_mps_sms, gpuMaxSmCount);
        exit(1);
      }
      // Create CUDA context with SM affinity
      CUexecAffinityParam affinityPrm;
      affinityPrm.type = CU_EXEC_AFFINITY_TYPE_SM_COUNT;
      affinityPrm.param.smCount.val = num_mps_sms;
#if CUDA_VERSION >= 13000
      CUctxCreateParams ctxParams{};
      ctxParams.execAffinityParams = &affinityPrm;
      ctxParams.numExecAffinityParams = 1;
      ctxParams.cigParams = nullptr;
      cuStatus = cuCtxCreate(&cuCtx_oktb, &ctxParams, CU_CTX_SCHED_SPIN | CU_CTX_MAP_HOST, cuDev);
#else
      cuStatus = cuCtxCreate_v3(&cuCtx_oktb, &affinityPrm, 1, CU_CTX_SCHED_SPIN | CU_CTX_MAP_HOST, cuDev);
#endif
      if (cuStatus != CUDA_SUCCESS) {
        NVLOGE_FMT(TAG_ORDER_TB_INIT, AERIAL_TESTBENCH_EVENT, "Error: Failed to create CUDA MPS context!");
        NVLOGW_FMT(TAG_ORDER_TB_INIT, "Please ensure the MPS daemon is running by executing:");
        NVLOGW_FMT(TAG_ORDER_TB_INIT, "  export CUDA_MPS_PIPE_DIRECTORY=/var");
        NVLOGW_FMT(TAG_ORDER_TB_INIT, "  export CUDA_MPS_LOG_DIRECTORY=/var");
        NVLOGW_FMT(TAG_ORDER_TB_INIT, "  sudo -E nvidia-cuda-mps-control -d");
        NVLOGW_FMT(TAG_ORDER_TB_INIT, "  sudo -E echo start_server -uid 0 | sudo -E nvidia-cuda-mps-control");
        return 1;
      }

      CUexecAffinityParam appliedAffinityPrm;
      CU_CHECK(cuCtxGetExecAffinity(&appliedAffinityPrm, CU_EXEC_AFFINITY_TYPE_SM_COUNT));
      int applied_MPS_SMs = appliedAffinityPrm.param.smCount.val;

      NVLOGC_FMT(TAG_ORDER_TB_INIT, "Running with MPS context, utilizing {} SMs, ({} SMs requested).",
                 applied_MPS_SMs, num_mps_sms);
    }
    else if (num_gc_sms > 0) // using Green Context
    {
#if CUDA_VERSION >= 12040
      CUdevResource initial_device_GPU_resources = {};
      CUdevResourceType default_resource_type = CU_DEV_RESOURCE_TYPE_SM; // other alternative is CU_DEV_RESOURCE_TYPE_INVALID
      CUdevResource split_result[2] = {{}, {}};
      unsigned int split_groups = 1;

      // Best to ensure that MPS service is not running
      int mpsEnabled = 0;
      CU_CHECK(cuDeviceGetAttribute(&mpsEnabled, CU_DEVICE_ATTRIBUTE_MPS_ENABLED, cuDev));
      if (mpsEnabled == 1) {
        NVLOGE_FMT(TAG_ORDER_TB_INIT, AERIAL_TESTBENCH_EVENT,  "MPS is enabled. Heads-up that currently using green contexts with MPS enabled can have unintended side effects. Will run regardless.");
        //exit(1);
      } else {
        NVLOGC_FMT(TAG_ORDER_TB_INIT, "MPS service is not running.");
      }

      //Check num_gc_sms value is in valid range
      if (num_gc_sms > (uint32_t)gpuMaxSmCount)
      {
        NVLOGE_FMT(TAG_ORDER_TB_INIT, AERIAL_TESTBENCH_EVENT,  "Error: Invalid --num_gc_sms argument {}. It is greater than {} (GPU's max SMs).", num_gc_sms, gpuMaxSmCount);
        exit(1);
      }

      CU_CHECK(cuDeviceGetDevResource(cuDev, &initial_device_GPU_resources, default_resource_type));
      CU_CHECK(cuDevSmResourceSplitByCount(&split_result[0], &split_groups, &initial_device_GPU_resources, &split_result[1], 0, num_gc_sms));
      greenCtx_oktb.create(gpuId, &split_result[0]);
      //greenCtx_oktb.bind();
      cuCtx_oktb = greenCtx_oktb.primary_handle();
      NVLOGC_FMT(TAG_ORDER_TB_INIT, "Running with green context, utilizing {} SMs, ({} SMs requested).",
          greenCtx_oktb.getSmCount(), num_gc_sms);
#else
      NVLOGE_FMT(TAG_ORDER_TB_INIT, AERIAL_TESTBENCH_EVENT, "Error: Green Context is not supported in CUDA version {}", CUDA_VERSION);
      return 1;
#endif
    }
    else // using all SMs on the device
    {
#if CUDA_VERSION >= 13000
      CUctxCreateParams ctxParams{};
      cuStatus = cuCtxCreate(&cuCtx_oktb, &ctxParams, 0, cuDev); // zero-init ctxParams and flags = 0 for default behavior
#else
      cuStatus = cuCtxCreate(&cuCtx_oktb, 0, cuDev); // Flags = 0 for default behavior
#endif
      if (cuStatus != CUDA_SUCCESS) {
        NVLOGE_FMT(TAG_ORDER_TB_INIT, AERIAL_TESTBENCH_EVENT, "Error: Failed to create CUDA context!");
        return 1;
      }
      NVLOGC_FMT(TAG_ORDER_TB_INIT, "No limits on the number of GPU SMs used to run the test bench.");
    }

    CU_CHECK(cuCtxSetCurrent(cuCtx_oktb));
    CUDA_CHECK(cudaStreamCreateWithPriority(&stream_oktb, cudaStreamNonBlocking, -5));
    CUDA_CHECK(cudaEventCreate(&start_ok_tb_process));
    CUDA_CHECK(cudaEventCreate(&end_ok_tb_process));
    return 0;
}

void OrderKernelTestBench::setup_config_params()
{
    for (int cell_idx=0;cell_idx<UL_MAX_CELLS_PER_SLOT;cell_idx++)
    {
        CUDA_CHECK(cudaMalloc((void**)&fh_buf_ok_tb[cell_idx],MAX_PKTS_PER_SLOT_OK_TB*ok_tb_max_packet_size*MAX_UL_SLOTS_OK_TB));
    }
    //TODO: Read and set params from config file
    // uint32_t      pusch_prb_stride[UL_MAX_CELLS_PER_SLOT]={273};
    // int      prach_prb_stride[UL_MAX_CELLS_PER_SLOT]={12};
    // uint16_t sem_order_num[UL_MAX_CELLS_PER_SLOT]={0};
    // int ru_type[UL_MAX_CELLS_PER_SLOT] = {OTHER_MODE};
    // int comp_meth[UL_MAX_CELLS_PER_SLOT] = {1};
    // int bit_width[UL_MAX_CELLS_PER_SLOT] = {BFP_COMPRESSION_9_BITS};

    for(int cell_count=0;cell_count<num_test_cells;cell_count++)
    {
        ok_tb_config_params->sem_order_num[cell_count]=0;
        ok_tb_config_params->ru_type[cell_count]=OTHER_MODE;
        ok_tb_config_params->comp_meth[cell_count]=1;
        ok_tb_config_params->bit_width[cell_count]=BFP_COMPRESSION_9_BITS;
        ok_tb_config_params->beta[cell_count]=0.0078086853;//TODO:Remove hardcoding for BFP9 //1.0/float(1UL << (bit_width[cell_count] - 5));
        ok_tb_config_params->last_sem_idx_order_h[cell_count]=ok_tb_input_params.last_sem_idx_order_h[cell_count];

        ok_tb_config_params->early_rx_packets[cell_count]=ok_tb_input_params.early_rx_packets[cell_count];
        ok_tb_config_params->on_time_rx_packets[cell_count]=ok_tb_input_params.on_time_rx_packets[cell_count];
        ok_tb_config_params->late_rx_packets[cell_count]=ok_tb_input_params.late_rx_packets[cell_count];
        ok_tb_config_params->next_slot_early_rx_packets[cell_count]=ok_tb_input_params.next_slot_early_rx_packets[cell_count];
        ok_tb_config_params->next_slot_on_time_rx_packets[cell_count]=ok_tb_input_params.next_slot_on_time_rx_packets[cell_count];
        ok_tb_config_params->next_slot_late_rx_packets[cell_count]=ok_tb_input_params.next_slot_late_rx_packets[cell_count];
        ok_tb_config_params->rx_packets_dropped_count[cell_count]=ok_tb_input_params.rx_packets_dropped_count[cell_count];
        ok_tb_config_params->start_cuphy_d[cell_count]=ok_tb_input_params.start_cuphy_d[cell_count];

        if(enable_srs)
        {
            ok_tb_config_params->srs_buffer[cell_count]=ok_tb_input_params.srs_buffer[cell_count];
            ok_tb_config_params->srs_eAxC_map[cell_count]=ok_tb_input_params.srs_eAxC_map[cell_count];
            ok_tb_config_params->srs_ordered_prbs[cell_count]=ok_tb_input_params.srs_ordered_prbs[cell_count];
            ok_tb_config_params->srs_start_sym[cell_count]=12; //TODO : Remove hardcoding, read from config file
            ok_tb_config_params->srs_prb_stride[cell_count]=273;
        }
        else
        {
            ok_tb_config_params->pusch_buffer[cell_count]=ok_tb_input_params.pusch_buffer[cell_count];
            ok_tb_config_params->pusch_eAxC_map[cell_count]=ok_tb_input_params.pusch_eAxC_map[cell_count];
            ok_tb_config_params->pusch_prb_x_port_x_symbol[cell_count]=273;
            ok_tb_config_params->pusch_ordered_prbs[cell_count]=ok_tb_input_params.pusch_ordered_prbs[cell_count];
            ok_tb_config_params->prach_eAxC_map[cell_count]=ok_tb_input_params.prach_eAxC_map[cell_count];
            ok_tb_config_params->prach_buffer_0[cell_count]=ok_tb_input_params.prach_buffer_0[cell_count];
            ok_tb_config_params->prach_buffer_1[cell_count]=ok_tb_input_params.prach_buffer_1[cell_count];
            ok_tb_config_params->prach_buffer_2[cell_count]=ok_tb_input_params.prach_buffer_2[cell_count];
            ok_tb_config_params->prach_buffer_3[cell_count]=ok_tb_input_params.prach_buffer_3[cell_count];
            ok_tb_config_params->prach_prb_x_port_x_symbol[cell_count]=12;
            ok_tb_config_params->prach_ordered_prbs[cell_count]=ok_tb_input_params.prach_ordered_prbs[cell_count];
        }
        ok_tb_config_params->order_kernel_last_timeout_error_time[cell_count]=ok_tb_input_params.order_kernel_last_timeout_error_time[cell_count];
        ok_tb_config_params->last_sem_idx_rx_h[cell_count]=ok_tb_input_params.last_sem_idx_rx_h[cell_count];
        ok_tb_config_params->sem_gpu[cell_count]=ok_tb_input_params.sem_gpu[cell_count];
        ok_tb_config_params->sem_gpu_aerial_fh[cell_count]=ok_tb_input_params.sem_gpu_aerial_fh[cell_count];
        ok_tb_config_params->doca_rxq[cell_count]=ok_tb_input_params.doca_rxq[cell_count];
    }
        ok_tb_config_params->prb_size=DEFAULT_PRB_STRIDE;
        ok_tb_config_params->max_pkt_size=ok_tb_max_packet_size;

    if(!enable_srs)
    {
        ok_tb_config_params->sym_ord_done_sig_arr=ok_tb_input_params.sym_ord_done_sig_arr;
        ok_tb_config_params->sym_ord_done_mask_arr=ok_tb_input_params.sym_ord_done_mask_arr;
        ok_tb_config_params->pusch_prb_symbol_map=ok_tb_input_params.pusch_prb_symbol_map;
        ok_tb_config_params->num_order_cells_sym_mask_arr=ok_tb_input_params.num_order_cells_sym_mask_arr;
        ok_tb_config_params->pusch_symbols_x_slot=ORAN_PUSCH_SYMBOLS_X_SLOT;

        ok_tb_config_params->prach_symbols_x_slot=ORAN_PRACH_B4_SYMBOLS_X_SLOT;
        ok_tb_config_params->prach_section_id_0=2048;
        ok_tb_config_params->prach_section_id_1=ok_tb_config_params->prach_section_id_0+1;
        ok_tb_config_params->prach_section_id_2=ok_tb_config_params->prach_section_id_0+2;
        ok_tb_config_params->prach_section_id_3=ok_tb_config_params->prach_section_id_0+3;
    }

        ok_tb_config_params->num_order_cells=num_test_cells;
        ok_tb_config_params->cell_health = ok_tb_input_params.cell_health;
}

OrderKernelTestBench::~OrderKernelTestBench()
{
    if (initStatus) return;
    for (int cell_count=0;cell_count<UL_MAX_CELLS_PER_SLOT;cell_count++){
        cudaFree(fh_buf_ok_tb[cell_count]);
        cudaFreeHost(ok_tb_input_params.exit_cond_d[cell_count]);
        cudaFree(ok_tb_input_params.last_sem_idx_order_h[cell_count]);
        cudaFree(ok_tb_input_params.early_rx_packets[cell_count]);
        cudaFree(ok_tb_input_params.on_time_rx_packets[cell_count]);
        cudaFree(ok_tb_input_params.late_rx_packets[cell_count]);
        cudaFree(ok_tb_input_params.next_slot_early_rx_packets[cell_count]);
        cudaFree(ok_tb_input_params.next_slot_on_time_rx_packets[cell_count]);
        cudaFree(ok_tb_input_params.next_slot_late_rx_packets[cell_count]);
        cudaFree(ok_tb_input_params.rx_packets_dropped_count[cell_count]);
        cudaFree(ok_tb_input_params.start_cuphy_d[cell_count]);
        cudaFree(ok_tb_input_params.pusch_buffer[cell_count]);
        cudaFreeHost(ok_tb_input_params.pusch_buffer_h[cell_count]);
        cudaFree(ok_tb_input_params.pusch_eAxC_map[cell_count]);
        cudaFree(ok_tb_input_params.pusch_ordered_prbs[cell_count]);
        cudaFree(ok_tb_input_params.prach_eAxC_map[cell_count]);
        cudaFree(ok_tb_input_params.prach_buffer_0[cell_count]);
        cudaFree(ok_tb_input_params.prach_buffer_1[cell_count]);
        cudaFree(ok_tb_input_params.prach_buffer_2[cell_count]);
        cudaFree(ok_tb_input_params.prach_buffer_3[cell_count]);
        cudaFreeHost(ok_tb_input_params.prach_buffer_0_h[cell_count]);
        cudaFreeHost(ok_tb_input_params.prach_buffer_1_h[cell_count]);
        cudaFreeHost(ok_tb_input_params.prach_buffer_2_h[cell_count]);
        cudaFreeHost(ok_tb_input_params.prach_buffer_3_h[cell_count]);
        cudaFree(ok_tb_input_params.prach_ordered_prbs[cell_count]);
        cudaFree(ok_tb_input_params.order_kernel_last_timeout_error_time[cell_count]);
        cudaFree(ok_tb_input_params.last_sem_idx_rx_h[cell_count]);
        cudaFreeHost(ok_tb_input_params.pkt_info[cell_count]);
        cudaFreeHost(ok_tb_input_params.sem_gpu[cell_count]);
        cudaFreeHost(ok_tb_input_params.sem_gpu_aerial_fh[cell_count]);
        cudaFree(ok_tb_input_params.doca_rxq[cell_count]);            
    }
    cudaFree(ok_tb_input_params.sym_ord_done_sig_arr);
    cudaFree(ok_tb_input_params.sym_ord_done_mask_arr);
    cudaFree(ok_tb_input_params.pusch_prb_symbol_map);
    cudaFree(ok_tb_input_params.num_order_cells_sym_mask_arr);
    cudaFree(ok_tb_input_params.cell_health);

    cudaStreamDestroy(stream_oktb);
    cuCtxSynchronize();
    cuCtxDestroy(cuCtx_oktb);
    cudaEventDestroy(start_ok_tb_process);
    cudaEventDestroy(end_ok_tb_process);
    delete slot_info;
}

void OrderKernelTestBench::add_gpu_comm_ready_flags()
{
    // auto ok_Info              = const_cast<OrderKernelInfo*>(&yaml_parser_.get_order_kernel_info());
    // for(auto cuda_device: ok_Info->cuda_device_ids)
    // {
    //     NVLOGC_FMT(TAG_ORDER_TB_BASE, "Creating GPU instance with GDRCopy for Device ID {}", cuda_device);
    //     gpus_.push_back(std::make_unique<GpuDevice>(cuda_device, true));
    //     buffer_ready_gdr.push_back(gpus_.back()->newGDRbuf(1 * sizeof(uint32_t)));
    //     ((uint32_t*)buffer_ready_gdr.back()->addrh())[0] = 1;
    // }
  
}

void OrderKernelTestBench::OrderKernelSetupDocaParams(uint32_t cell_idx)
{
    doca_rxq_info[cell_idx].cqe_mask=DOCA_GPUNETIO_CQE_CI_MASK;
    doca_rxq_info[cell_idx].wqe_mask=DOCA_GPUNETIO_CQE_CI_MASK;
    doca_rxq_info[cell_idx].cqe_ci=0;
    
}

void OrderKernelTestBench::setup_input_params()
{
    for(int cell_count=0;cell_count<UL_MAX_CELLS_PER_SLOT;cell_count++)
    {
        CUDA_CHECK(cudaMallocHost((void**)&ok_tb_input_params.exit_cond_d[cell_count],sizeof(uint32_t)));
        *ok_tb_input_params.exit_cond_d[cell_count]=ORDER_KERNEL_RUNNING;
        CUDA_CHECK(cudaMalloc((void**)&ok_tb_input_params.last_sem_idx_order_h[cell_count],sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc((void**)&ok_tb_input_params.early_rx_packets[cell_count],sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc((void**)&ok_tb_input_params.on_time_rx_packets[cell_count],sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc((void**)&ok_tb_input_params.late_rx_packets[cell_count],sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc((void**)&ok_tb_input_params.next_slot_early_rx_packets[cell_count],sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc((void**)&ok_tb_input_params.next_slot_on_time_rx_packets[cell_count],sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc((void**)&ok_tb_input_params.next_slot_late_rx_packets[cell_count],sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc((void**)&ok_tb_input_params.rx_packets_dropped_count[cell_count],sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc((void**)&ok_tb_input_params.start_cuphy_d[cell_count],sizeof(uint32_t)));
        CUDA_CHECK(cudaMemset((void*)ok_tb_input_params.start_cuphy_d[cell_count],0,sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc((void**)&ok_tb_input_params.pusch_buffer[cell_count],UL_ST1_AP_BUF_SIZE*max_rx_ant));
        CUDA_CHECK(cudaMallocHost((void**)&ok_tb_input_params.pusch_buffer_h[cell_count],UL_ST1_AP_BUF_SIZE*max_rx_ant));
        CUDA_CHECK(cudaMalloc((void**)&ok_tb_input_params.pusch_eAxC_map[cell_count],max_rx_ant*sizeof(uint16_t)));
        CUDA_CHECK(cudaMalloc((void**)&ok_tb_input_params.pusch_ordered_prbs[cell_count],sizeof(uint32_t)));
        CUDA_CHECK(cudaMemset((void*)ok_tb_input_params.pusch_ordered_prbs[cell_count],0,sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc((void**)&ok_tb_input_params.prach_eAxC_map[cell_count],max_rx_ant*sizeof(uint16_t)));
        CUDA_CHECK(cudaMalloc((void**)&ok_tb_input_params.prach_buffer_0[cell_count],UL_ST3_AP_BUF_SIZE*max_rx_ant));
        CUDA_CHECK(cudaMalloc((void**)&ok_tb_input_params.prach_buffer_1[cell_count],UL_ST3_AP_BUF_SIZE*max_rx_ant));
        CUDA_CHECK(cudaMalloc((void**)&ok_tb_input_params.prach_buffer_2[cell_count],UL_ST3_AP_BUF_SIZE*max_rx_ant));
        CUDA_CHECK(cudaMalloc((void**)&ok_tb_input_params.prach_buffer_3[cell_count],UL_ST3_AP_BUF_SIZE*max_rx_ant));
        CUDA_CHECK(cudaMallocHost((void**)&ok_tb_input_params.prach_buffer_0_h[cell_count],UL_ST3_AP_BUF_SIZE*max_rx_ant));
        CUDA_CHECK(cudaMallocHost((void**)&ok_tb_input_params.prach_buffer_1_h[cell_count],UL_ST3_AP_BUF_SIZE*max_rx_ant));
        CUDA_CHECK(cudaMallocHost((void**)&ok_tb_input_params.prach_buffer_2_h[cell_count],UL_ST3_AP_BUF_SIZE*max_rx_ant));
        CUDA_CHECK(cudaMallocHost((void**)&ok_tb_input_params.prach_buffer_3_h[cell_count],UL_ST3_AP_BUF_SIZE*max_rx_ant));
        CUDA_CHECK(cudaMalloc((void**)&ok_tb_input_params.prach_ordered_prbs[cell_count],sizeof(uint32_t)));
        CUDA_CHECK(cudaMemset((void*)ok_tb_input_params.prach_ordered_prbs[cell_count],0,sizeof(uint32_t)));

        /*SRS*/
        CUDA_CHECK(cudaMalloc((void**)&ok_tb_input_params.srs_eAxC_map[cell_count],MAX_RX_ANT_SRS_64T64R*sizeof(uint16_t)));
        CUDA_CHECK(cudaMalloc((void**)&ok_tb_input_params.srs_buffer[cell_count],UL_ST2_AP_BUF_SIZE*MAX_RX_ANT_SRS_64T64R));
        CUDA_CHECK(cudaMallocHost((void**)&ok_tb_input_params.srs_buffer_h[cell_count],UL_ST2_AP_BUF_SIZE*MAX_RX_ANT_SRS_64T64R));
        CUDA_CHECK(cudaMalloc((void**)&ok_tb_input_params.srs_ordered_prbs[cell_count],sizeof(uint32_t)));
        CUDA_CHECK(cudaMemset((void*)ok_tb_input_params.srs_ordered_prbs[cell_count],0,sizeof(uint32_t)));

        CUDA_CHECK(cudaMalloc((void**)&ok_tb_input_params.order_kernel_last_timeout_error_time[cell_count],sizeof(uint64_t)));
        CUDA_CHECK(cudaMemset((void*)ok_tb_input_params.order_kernel_last_timeout_error_time[cell_count],0,sizeof(uint64_t)));        
        CUDA_CHECK(cudaMalloc((void**)&ok_tb_input_params.last_sem_idx_rx_h[cell_count],sizeof(uint32_t)));
        CUDA_CHECK(cudaMemset((void*)ok_tb_input_params.last_sem_idx_rx_h[cell_count],0,sizeof(uint32_t)));        
        CUDA_CHECK(cudaMallocHost((void**)&ok_tb_input_params.sem_gpu[cell_count],sizeof(doca_gpu_semaphore_gpu_t)));
        CUDA_CHECK(cudaMallocHost((void**)&ok_tb_input_params.pkt_info[cell_count],sizeof(struct doca_gpu_semaphore_packet)*4096));
        ok_tb_input_params.sem_gpu[cell_count]->pkt_info_gpu=(struct doca_gpu_semaphore_packet *)ok_tb_input_params.pkt_info[cell_count];
        CUDA_CHECK(cudaMallocHost((void**)&ok_tb_input_params.sem_gpu_aerial_fh[cell_count],sizeof(struct aerial_fh_gpu_semaphore_gpu)));
        CUDA_CHECK(cudaMalloc((void**)&ok_tb_input_params.cq_db_rec[cell_count],sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc((void**)&ok_tb_input_params.rq_db_rec[cell_count],sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc((void**)&ok_tb_input_params.doca_rxq[cell_count],sizeof(struct doca_gpu_eth_rxq)));
        CUDA_CHECK(cudaMalloc((void**)&ok_tb_input_params.cqe_addr[cell_count],sizeof(struct mlx5_cqe)*(DOCA_GPUNETIO_CQE_CI_MASK+1)));
        CUDA_CHECK(cudaMemset((void*)ok_tb_input_params.cqe_addr[cell_count],0,sizeof(struct mlx5_cqe)));  
    }
    CUDA_CHECK(cudaMalloc((void**)&ok_tb_input_params.cell_health,UL_MAX_CELLS_PER_SLOT*sizeof(bool)));
    CUDA_CHECK(cudaMemset((void*)ok_tb_input_params.cell_health,0x1,UL_MAX_CELLS_PER_SLOT*sizeof(bool)));
    CUDA_CHECK(cudaMalloc((void**)&ok_tb_input_params.sym_ord_done_sig_arr,ORAN_PUSCH_SYMBOLS_X_SLOT*sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc((void**)&ok_tb_input_params.sym_ord_done_mask_arr,ORAN_PUSCH_SYMBOLS_X_SLOT*sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc((void**)&ok_tb_input_params.pusch_prb_symbol_map,UL_MAX_CELLS_PER_SLOT*ORAN_PUSCH_SYMBOLS_X_SLOT*sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc((void**)&ok_tb_input_params.num_order_cells_sym_mask_arr,ORAN_PUSCH_SYMBOLS_X_SLOT*sizeof(uint32_t)));
}

void OrderKernelTestBench::read_ok_tb_config_file_params()
{
    if(enable_srs)
    {
        std::ifstream inFile(config_file_);
        if (!inFile.is_open())
        {
        throw std::runtime_error("Failed to open configuration file: " + config_file_);
        }
        std::string line;
        std::string temp;
        uint32_t frameId,subframeId,slotId;
        // FixMe: make the read file more robust
        std::getline(inFile, line);
        std::istringstream numValidSlotsStrm(line);
        numValidSlotsStrm>>temp>>temp>>temp;
        ok_tb_num_valid_slots=std::stoi(temp);
        std::getline(inFile, line);
        std::istringstream maxPacketSizeStrm(line);
        maxPacketSizeStrm>>temp>>temp>>temp;
        ok_tb_max_packet_size=std::stoi(temp);        
        for(uint32_t slot_idx=0;slot_idx<ok_tb_num_valid_slots;slot_idx++)
        {
            std::string slot_name = "slot_"+std::to_string(slot_idx);
            bool slot_found = false;

            while (std::getline(inFile, line)) // Ensure we do not loop endlessly
            {
                if(line.find(slot_name) != std::string::npos)
                {
                    slot_found = true;
                    std::getline(inFile, line);
                    std::istringstream frameStrm(line);
                    frameStrm>>temp>>temp>>temp;
                    frameId=std::stoi(temp);
                    std::getline(inFile, line);
                    frameStrm.clear();
                    frameStrm.str(line);
                    frameStrm>>temp>>temp>>temp;
                    subframeId=std::stoi(temp);
                    std::getline(inFile, line);
                    frameStrm.clear();
                    frameStrm.str(line);
                    frameStrm>>temp>>temp>>temp;
                    slotId=std::stoi(temp);

                    for(uint32_t cell_idx=0;cell_idx<num_test_cells;cell_idx++)
                    {
                        std::getline(inFile, line);
                        std::string cell_name = "cell_"+std::to_string(cell_idx);
                        if(line.find(cell_name) != std::string::npos)
                        {
                            ok_tb_config_file_params[cell_idx].frameId[slot_idx]=frameId;
                            ok_tb_config_file_params[cell_idx].subframeId[slot_idx]=subframeId;
                            ok_tb_config_file_params[cell_idx].slotId[slot_idx]=slotId;

                            std::getline(inFile, line);
                            std::istringstream cidStrm(line);
                            cidStrm>>temp>>temp>>temp;
                            ok_tb_config_file_params[cell_idx].cell_id[slot_idx]=std::stoi(temp);

                            std::getline(inFile, line);
                            std::istringstream pktStrm(line);
                            pktStrm>>temp>>temp>>temp;
                            ok_tb_config_file_params[cell_idx].num_rx_packets[slot_idx]=std::stoi(temp);

                            std::getline(inFile, line);
                            std::istringstream puschPrbStrm(line);
                            puschPrbStrm>>temp>>temp>>temp;
                            ok_tb_config_file_params[cell_idx].num_srs_prbs[slot_idx]=std::stoi(temp);

                            std::getline(inFile, line);
                            std::istringstream eAxCStrm(line);       
                            eAxCStrm>>temp>>temp>>temp;
                            ok_tb_config_file_params[cell_idx].srs_eAxC_num[slot_idx]=std::stoi(temp);             
                            std::getline(inFile, line);
                            eAxCStrm.clear();
                            eAxCStrm.str(line);       
                            eAxCStrm>>temp>>temp; //skip first two strings
                            for(int tmp=0;tmp<ok_tb_config_file_params[cell_idx].srs_eAxC_num[slot_idx];tmp++)
                            {
                                eAxCStrm>>temp;
                                ok_tb_config_file_params[cell_idx].srs_eAxC_map[slot_idx][tmp]=std::stoi(temp);
                            }
                        }
                    }
                    break;
                }
            }
            if (!slot_found)
            {
            throw std::runtime_error("Slot name not found in configuration file: " + slot_name);
            }
        }
        inFile.close();
        parse_launch_pattern(launch_pattern_file_);
        load_srs_tvs();
    }
    else
    {
        std::ifstream inFile(config_file_);
        if (!inFile.is_open())
        {
        throw std::runtime_error("Failed to open configuration file: " + config_file_);
        }
        std::string line;
        std::string temp;
        uint32_t frameId,subframeId,slotId,num_order_cells_sym_mask_temp[ORAN_PUSCH_SYMBOLS_X_SLOT];
        // FixMe: make the read file more robust
        std::getline(inFile, line);
        std::istringstream numValidSlotsStrm(line);
        numValidSlotsStrm>>temp>>temp>>temp;
        ok_tb_num_valid_slots=std::stoi(temp);
        std::getline(inFile, line);
        std::istringstream maxPacketSizeStrm(line);
        maxPacketSizeStrm>>temp>>temp>>temp;
        ok_tb_max_packet_size=std::stoi(temp);        
        for(uint32_t slot_idx=0;slot_idx<ok_tb_num_valid_slots;slot_idx++)
        {
            std::string slot_name = "slot_"+std::to_string(slot_idx);
            bool slot_found = false;

            while (std::getline(inFile, line)) // Ensure we do not loop endlessly
            {
                if(line.find(slot_name) != std::string::npos)
                {
                    slot_found = true;
                    std::getline(inFile, line);
                    std::istringstream frameStrm(line);
                    frameStrm>>temp>>temp>>temp;
                    frameId=std::stoi(temp);
                    std::getline(inFile, line);
                    frameStrm.clear();
                    frameStrm.str(line);
                    frameStrm>>temp>>temp>>temp;
                    subframeId=std::stoi(temp);
                    std::getline(inFile, line);
                    frameStrm.clear();
                    frameStrm.str(line);
                    frameStrm>>temp>>temp>>temp;
                    slotId=std::stoi(temp);
                    std::getline(inFile, line);
                    std::istringstream cellSymMaskStrm(line);
                    cellSymMaskStrm>>temp>>temp; //skip first two strings
                    for(int tmp=0;tmp<ORAN_PUSCH_SYMBOLS_X_SLOT;tmp++)
                    {
                        cellSymMaskStrm>>temp;
                        num_order_cells_sym_mask_temp[tmp]=std::stoi(temp);
                        //Override based on actual number of cells being tested
                        num_order_cells_sym_mask_temp[tmp]&=((0x1<<num_test_cells)-1);
                    }            

                    for(uint32_t cell_idx=0;cell_idx<num_test_cells;cell_idx++)
                    {
                        memcpy(ok_tb_config_file_params[cell_idx].num_order_cells_sym_mask[slot_idx],num_order_cells_sym_mask_temp,ORAN_PUSCH_SYMBOLS_X_SLOT*sizeof(uint32_t));
                        std::getline(inFile, line);
                        std::string cell_name = "cell_"+std::to_string(cell_idx);
                        if(line.find(cell_name) != std::string::npos)
                        {
                            ok_tb_config_file_params[cell_idx].frameId[slot_idx]=frameId;
                            ok_tb_config_file_params[cell_idx].subframeId[slot_idx]=subframeId;
                            ok_tb_config_file_params[cell_idx].slotId[slot_idx]=slotId;

                            std::getline(inFile, line);
                            std::istringstream cidStrm(line);
                            cidStrm>>temp>>temp>>temp;
                            ok_tb_config_file_params[cell_idx].cell_id[slot_idx]=std::stoi(temp);

                            std::getline(inFile, line);
                            std::istringstream pktStrm(line);
                            pktStrm>>temp>>temp>>temp;
                            ok_tb_config_file_params[cell_idx].num_rx_packets[slot_idx]=std::stoi(temp);

                            std::getline(inFile, line);
                            std::istringstream puschPrbStrm(line);
                            puschPrbStrm>>temp>>temp>>temp;
                            ok_tb_config_file_params[cell_idx].num_pusch_prbs[slot_idx]=std::stoi(temp);

                            std::getline(inFile, line);
                            std::istringstream prachPrbStrm(line);
                            prachPrbStrm>>temp>>temp>>temp;
                            ok_tb_config_file_params[cell_idx].num_prach_prbs[slot_idx]=std::stoi(temp);                                                            

                            std::getline(inFile, line);
                            std::istringstream eAxCStrm(line);       
                            eAxCStrm>>temp>>temp>>temp;
                            ok_tb_config_file_params[cell_idx].pusch_eAxC_num[slot_idx]=std::stoi(temp);             
                            std::getline(inFile, line);
                            eAxCStrm.clear();
                            eAxCStrm.str(line);       
                            eAxCStrm>>temp>>temp; //skip first two strings
                            for(int tmp=0;tmp<ok_tb_config_file_params[cell_idx].pusch_eAxC_num[slot_idx];tmp++)
                            {
                                eAxCStrm>>temp;
                                ok_tb_config_file_params[cell_idx].pusch_eAxC_map[slot_idx][tmp]=std::stoi(temp);
                            }

                            std::getline(inFile, line);
                            eAxCStrm.clear();
                            eAxCStrm.str(line);       
                            eAxCStrm>>temp>>temp>>temp;
                            ok_tb_config_file_params[cell_idx].prach_eAxC_num[slot_idx]=std::stoi(temp);             
                            std::getline(inFile, line);
                            eAxCStrm.clear();
                            eAxCStrm.str(line);       
                            eAxCStrm>>temp>>temp; //skip first two strings
                            for(int tmp=0;tmp<ok_tb_config_file_params[cell_idx].prach_eAxC_num[slot_idx];tmp++)
                            {
                                eAxCStrm>>temp;
                                ok_tb_config_file_params[cell_idx].prach_eAxC_map[slot_idx][tmp]=std::stoi(temp);
                            }
                            
                            std::getline(inFile, line);
                            std::istringstream prbSymMapStrm(line);
                            prbSymMapStrm>>temp>>temp; //skip first two strings
                            for(int tmp=0;tmp<ORAN_PUSCH_SYMBOLS_X_SLOT;tmp++)
                            {
                                prbSymMapStrm>>temp;
                                ok_tb_config_file_params[cell_idx].pusch_prb_symbol_map[slot_idx][tmp]=std::stoi(temp);
                            }                                                                
                        }
                    }
                    break;
                }
            }
            if (!slot_found)
            {
            throw std::runtime_error("Slot name not found in configuration file: " + slot_name);
            }
        }
        inFile.close();
        parse_launch_pattern(launch_pattern_file_);
        load_pusch_tvs();
        load_prach_tvs();        
    }
}

void OrderKernelTestBench::launch_pattern_v2_tv_pre_processing(yaml::node& root)
{
   int slot_num;
   try
   {
       for(int slot_idx = 0; slot_idx < root.length(); ++slot_idx)
       {
           yaml::node config_node = root[slot_idx][YAML_LP_CONFIG];
           std::string node_type = config_node.type_string();
           bool ul_ch_slot=false;
           if(node_type == "YAML_SCALAR_NODE")
           {
               continue;
           }
           slot_num=static_cast<int>(root[slot_idx][YAML_LP_SLOT]);

           ul_ch_slot=false;
           for(int cell_idx = 0; cell_idx < (int)config_node.length(); ++cell_idx)
           {
               yaml::node cell_node = config_node[cell_idx];
               yaml::node channels_node = cell_node[YAML_LP_CHANNELS];
               yaml::node type_node = cell_node[YAML_LP_CHANNEL_TYPE];
               for(int node_idx=0;node_idx<type_node.length();node_idx++)
               {
                    if((0==static_cast<std::string>(type_node[node_idx]).compare("PUSCH"))||(0==static_cast<std::string>(type_node[node_idx]).compare("PUCCH"))||(0==static_cast<std::string>(type_node[node_idx]).compare("PRACH"))||(0==static_cast<std::string>(type_node[node_idx]).compare("SRS")))
                    {
                        ul_ch_slot=true;
                    }
               }
               if(!ul_ch_slot)
               {
                    NVLOGI_FMT(TAG_ORDER_TB_BASE,"Uplink channels not present in the slot index {} of launch pattern file",slot_idx);
                    break;
               }

               std::string node_type = channels_node.type_string();
               if(node_type == "YAML_SCALAR_NODE")
               {
                   continue;
               }

               for(int ch_idx = 0; ch_idx < channels_node.length(); ++ch_idx)
               {

                    std::string tv_name = static_cast<std::string>(channels_node[ch_idx]);

                    std::string key=tv_name;
                    bool key_exists=false;
                    auto range = ok_tb_tv_params.tv_to_slot_map[cell_idx].equal_range(key);
                    for (auto it = range.first; it != range.second; ++it)
                    {
                        if (it->second == slot_num) {
                            key_exists=true;
                            break;
                        }
                    }
                    if(!key_exists)
                    {
                        ok_tb_tv_params.tv_to_slot_map[cell_idx].insert({tv_name,slot_num});
                    }

                    if(ok_tb_tv_params.tv_to_channel_map.find(tv_name) != ok_tb_tv_params.tv_to_channel_map.end())
                    {
                        continue;
                    }
                    char tv_full_path[MAX_PATH_LEN];
                    get_full_path_file(tv_full_path, CONFIG_TEST_VECTOR_PATH, tv_name.c_str(), CONFIG_CUBB_ROOT_DIR_RELATIVE_NUM);
                    std::string tv_path = std::string(tv_full_path);

                    if (access(tv_full_path, F_OK) != 0)
                    {
                        NVLOGE_FMT(TAG_ORDER_TB_BASE, AERIAL_INVALID_PARAM_EVENT, "File {} does not exist", tv_path);
                        exit(1);
                    }

                    hdf5hpp::hdf5_file hdf5file = hdf5hpp::hdf5_file::open(tv_path.c_str());
                    int count = 1;
                    std::string pdu = "PDU";
                    while(1)
                    {
                       std::string dset_string = pdu + std::to_string(count);
                       if(!hdf5file.is_valid_dataset(dset_string.c_str()))
                       {
                            break;
                       }
                       hdf5hpp::hdf5_dataset dset_PDU  = hdf5file.open_dataset(dset_string.c_str());
                       hdf5hpp::hdf5_dataset_elem pdu_pars = dset_PDU[0];

                       uint8_t channel_type = pdu_pars["type"].as<uint8_t>();
                       std::string channel_string = "NONE";

                       switch (channel_type)
                       {
                            case nrsim_tv_type::PRACH:
                                channel_string = "PRACH";
                                break;
                            case nrsim_tv_type::PUCCH:
                                channel_string = "PUCCH";
                                break;
                            case nrsim_tv_type::PUSCH:
                                channel_string = "PUSCH";
                                break;
                            case nrsim_tv_type::SRS:
                                channel_string = "SRS";
                                break;
                       }

                       if(ok_tb_tv_params.channel_to_tv_map[channel_string].find(tv_name) == ok_tb_tv_params.channel_to_tv_map[channel_string].end())
                       {
                          int size = ok_tb_tv_params.channel_to_tv_map[channel_string].size();
                          std::string short_name = "TV" + std::to_string(size+1);
                          ok_tb_tv_params.channel_to_tv_map[channel_string].insert({tv_name, short_name});
                       }

                       ok_tb_tv_params.tv_to_channel_map[tv_name].insert(channel_string);
                       count++;
                   }
               }
           }
       }
   }
   catch(const std::exception& e)
   {
       NVLOGC_FMT(TAG_ORDER_TB_BASE,"{} Detected Launch pattern v2 not compatible, using v1 parsing",e.what());
   }
   //Iterate over the built multimap
#if 0
    for (auto it = ok_tb_tv_params.tv_to_slot_map[0].begin(); it !=ok_tb_tv_params.tv_to_slot_map[0].end();)
    {
        std::string key = it->first;
        std::cout << "Key: " << key << ", Values: ";
        std::cout << std::endl;

        auto range =    ok_tb_tv_params.tv_to_slot_map[0].equal_range(key);
        for (auto it2 = range.first; it2 != range.second; ++it2) {
            std::cout << it2->second << " ";
        }
        std::cout << std::endl;

        it = range.second; // Move to the next unique key
    }
#endif
}

void OrderKernelTestBench::yaml_assign_launch_pattern_tv(yaml::node root, std::string key, std::vector<std::string>& tvs, std::unordered_map<std::string, int>& tv_map)
{
    try
    {
        std::string tv_short_name;
        std::string tv_path;
        NVLOGI_FMT(TAG_ORDER_TB_BASE,"launch pattern v2 TV list {} {}", key.c_str(), ok_tb_tv_params.channel_to_tv_map[key].size());
        for(auto tv_p: ok_tb_tv_params.channel_to_tv_map[key])
        {
            tv_short_name = tv_p.second;
            tv_path = tv_p.first;

            char tv_full_path[MAX_PATH_LEN];
            get_full_path_file(tv_full_path, CONFIG_TEST_VECTOR_PATH, tv_path.c_str(), CONFIG_CUBB_ROOT_DIR_RELATIVE_NUM);
            tv_path = std::string(tv_full_path);

            tv_map[tv_short_name] = tvs.size();
            tvs.push_back(tv_path);
        }
    }
    catch(const std::exception& e)
    {
        NVLOGI_FMT(TAG_ORDER_TB_BASE,"{}", e.what());
        NVLOGI_FMT(TAG_ORDER_TB_BASE,"Exception parsing TVs for {}, assuming no TVs found for channel {}", key.c_str(), key.c_str());
        return;
    }
}

void OrderKernelTestBench::parse_launch_pattern_channel(yaml::node& root, std::string key, tv_object& tv_object)
{
    yaml::node sched = root[YAML_LP_SCHED];
    yaml_assign_launch_pattern_tv(root, key, tv_object.tv_names, tv_object.tv_map);
}

void OrderKernelTestBench::parse_launch_pattern(std::string& yaml_file)
{

    yaml::file_parser fp(yaml_file.c_str());
    yaml::document doc = fp.next_document();
    yaml::node root = doc.root();

    ok_tb_tv_params.num_cells = root[YAML_LP_CELL_CCONFIGS].length();
    yaml::node sched = root[YAML_LP_SCHED];
    ok_tb_tv_params.launch_pattern_version = root.has_key(YAML_LP_TV) ? 1 : 2;


    launch_pattern_v2_tv_pre_processing(sched);

    if(enable_srs)
    {
        parse_launch_pattern_channel(root, YAML_LP_SRS, ok_tb_tv_params.srs_object);
    }
    else 
    {
        parse_launch_pattern_channel(root, YAML_LP_PUSCH, ok_tb_tv_params.pusch_object);
        parse_launch_pattern_channel(root, YAML_LP_PRACH, ok_tb_tv_params.prach_object);
        parse_launch_pattern_channel(root, YAML_LP_PUCCH, ok_tb_tv_params.pucch_object);
    }
}


void OrderKernelTestBench::read_cell_cfg_from_tv(hdf5hpp::hdf5_file & hdf5file, struct tv_info & tv_info, std::string & tv_name)
{
    // Parse CellConfig dataset to retrieve CellConfig.ulGridSize value
    std::string dset_string = "Cell_Config";
    if(!hdf5file.is_valid_dataset(dset_string.c_str()))
    {
        NVLOGW_FMT(TAG_ORDER_TB_BASE,"ERROR No Cell_Config dataset found in TV {}",tv_name);
    }
    hdf5hpp::hdf5_dataset dset_Cell_Config = hdf5file.open_dataset(dset_string.c_str());
    hdf5hpp::hdf5_dataset_elem cell_config_pars = dset_Cell_Config[0];

    tv_info.nPrbUlBwp = cell_config_pars["ulGridSize"].as<uint16_t>(); // FAPI parameter is uint32_t
    tv_info.nPrbDlBwp = cell_config_pars["dlGridSize"].as<uint16_t>(); // FAPI parameter is uint32_t
    tv_info.numGnbAnt = cell_config_pars["numTxAnt"].as<uint16_t>();
}


int OrderKernelTestBench::load_num_antenna_from_nr_tv(hdf5hpp::hdf5_file& hdf5file)
{
    hdf5hpp::hdf5_dataset dset = hdf5file.open_dataset("X_tf_fp16");

    if(dset.get_dataspace().get_dimensions().size() < 3)
    {
        return 1;
    }
    return dset.get_dataspace().get_dimensions()[0];
}

Dataset OrderKernelTestBench::load_tv_datasets_single(hdf5hpp::hdf5_file& hdf5file, std::string const& file, std::string const dataset)
{
    Dataset d;
    hdf5hpp::hdf5_dataset dset = hdf5file.open_dataset(dataset.c_str());
    d.size = dset.get_buffer_size_bytes();
    NVLOGI_FMT(TAG_ORDER_TB_BASE,"Opened {}-byte dataset {}", d.size, dataset.c_str());
    auto pg_sz = sysconf(_SC_PAGESIZE);
    if(pg_sz == -1)
    {
        NVLOGC_FMT(TAG_ORDER_TB_BASE,"failed to get page size");
    }

    void * fh_mem = allocate_memory(d.size);
    if(fh_mem == nullptr)
    {
        NVLOGC_FMT(TAG_ORDER_TB_BASE,"order_kernel_tb::allocate_memory failure ");
    }

    d.data.reset(memset(fh_mem, 0, d.size));

    if (d.data.get() == nullptr)
        NVLOGC_FMT(TAG_ORDER_TB_BASE," malloc testvector data failed");
    NVLOGI_FMT(TAG_ORDER_TB_BASE,"Reading {}-byte dataset {}", d.size, dataset.c_str());
    dset.read(d.data.get());
    NVLOGI_FMT(TAG_ORDER_TB_BASE,"Read {}-byte dataset {}", d.size, dataset.c_str());

    return std::move(d);
}

Slot OrderKernelTestBench::dataset_to_slot_prach(Dataset d_1,Dataset d_2,Dataset d_3,Dataset d_4, size_t num_ante, size_t start_symbol, size_t num_symbols, size_t prb_sz, size_t tv_prbs_per_symbol, size_t start_prb)
{
    Slot ret(num_ante);
    Dataset* d;


    for(int buf_idx=0;buf_idx<PRACH_MAX_NUM_SEC;buf_idx++)
    {
        if(buf_idx==1)
            d=&d_2;
        else if(buf_idx==2)
            d=&d_3;
        else if(buf_idx==3)
            d=&d_4;
        else
            d=&d_1;

        ret.data_sz = d->size;

        ret.antenna_sz = d->size / num_ante;
        ret.symbol_sz = ret.antenna_sz / num_symbols;

        ret.prbs_per_symbol = ret.symbol_sz / prb_sz;
        ret.prbs_per_symbol = tv_prbs_per_symbol;
        ret.prbs_per_slot = ret.prbs_per_symbol * num_symbols * num_ante;

        NVLOGI_FMT(TAG_ORDER_TB_BASE,"[dataset_to_slot_prach] num_ante {} start_symbol {} num_symbols {} prb_sz {} tv_prbs_per_symbol {} start_prb {} ret.data_sz {} ret.antenna_sz {} ret.symbol_sz {} ret.prbs_per_symbol {} ret.prbs_per_slot {}",num_ante,start_symbol,num_symbols,prb_sz,tv_prbs_per_symbol,start_prb,ret.data_sz,ret.antenna_sz,ret.symbol_sz,ret.prbs_per_symbol,ret.prbs_per_slot);


        char *base_ptr = (char *)d->data.get();
        for (size_t symbol_idx = start_symbol; symbol_idx < num_symbols + start_symbol; ++symbol_idx)
        {
            for (size_t antenna_idx = 0; antenna_idx < num_ante; ++antenna_idx)
            {
                for (size_t prb_idx = start_prb; prb_idx < ret.prbs_per_symbol + start_prb; ++prb_idx)
                {
                    ret.ptrs_prach.at(buf_idx).at(antenna_idx).at(symbol_idx).at(prb_idx) =
                        (void *)(base_ptr + antenna_idx * ret.antenna_sz + (symbol_idx - start_symbol) * ret.symbol_sz + (prb_idx - start_prb) * prb_sz);
                }
            }
        }
    }
    ret.raw_data_prach_0 = std::move(d_1);
    ret.raw_data_prach_1 = std::move(d_2);
    ret.raw_data_prach_2 = std::move(d_3);
    ret.raw_data_prach_3 = std::move(d_4);
    return ret;
}

Slot OrderKernelTestBench::dataset_to_slot(Dataset d, size_t num_ante, size_t start_symbol, size_t num_symbols, size_t prb_sz, size_t tv_prbs_per_symbol, size_t start_prb)
{
    Slot ret(num_ante);

    ret.data_sz = d.size;
    if (ret.data_sz % num_symbols)
    {
        NVLOGW_FMT(TAG_ORDER_TB_BASE,"Slot size {} doesn't divide into the number of symbols {}",ret.data_sz,num_symbols);
    }

    ret.antenna_sz = d.size / num_ante;
    ret.symbol_sz = ret.antenna_sz / num_symbols;
    if (ret.symbol_sz % prb_sz)
    {
        NVLOGW_FMT(TAG_ORDER_TB_BASE,"Symbol size {} doesn't divide into the size of a PRB {}",ret.symbol_sz,prb_sz);
    }

    ret.prbs_per_symbol = ret.symbol_sz / prb_sz;
    if (ret.prbs_per_symbol > MAX_N_PRBS_SUPPORTED)
    {
        NVLOGW_FMT(TAG_ORDER_TB_BASE,"Resulting number of PRBs per symbol {} is higher than MAX_N_PRBS_SUPPORTED",ret.prbs_per_symbol);
    }
    ret.prbs_per_symbol = tv_prbs_per_symbol;
    ret.prbs_per_slot = ret.prbs_per_symbol * num_symbols * num_ante;

    char *base_ptr = (char *)d.data.get();
    for (size_t symbol_idx = start_symbol; symbol_idx < num_symbols + start_symbol; ++symbol_idx)
    {
        for (size_t antenna_idx = 0; antenna_idx < num_ante; ++antenna_idx)
        {
            for (size_t prb_idx = start_prb; prb_idx < ret.prbs_per_symbol + start_prb; ++prb_idx)
            {
                ret.ptrs.at(antenna_idx).at(symbol_idx).at(prb_idx) =
                    (void *)(base_ptr + antenna_idx * ret.antenna_sz + (symbol_idx - start_symbol) * ret.symbol_sz + (prb_idx - start_prb) * prb_sz);
#if 0
                if(symbol_idx==0 && antenna_idx==0 && prb_idx==start_prb)
                {
                    printf("start_prb=%d tv_prbs_per_symbol=%d\n",start_prb,tv_prbs_per_symbol);
                    uint8_t* tmp = static_cast<uint8_t*>(ret.ptrs.at(antenna_idx).at(symbol_idx).at(prb_idx));
                    __half*  tmp_half;
                    float tmp_float;
                    for(int i=0;i<prb_sz;i+=2)
                    {
                        tmp_half=reinterpret_cast<__half*>(tmp);
                        tmp_float=__half2float(*tmp_half);
                        printf("%f ",tmp_float);
                        tmp+=2;
                    }
                    printf("\n");
                }
#endif
            }
        }
    }

#if 0
    NVLOGI_FMT(TAG_ORDER_TB_BASE,"data_sz {}", data_sz);
    NVLOGI_FMT(TAG_ORDER_TB_BASE,"antenna_sz {}", antenna_sz);
    NVLOGI_FMT(TAG_ORDER_TB_BASE,"symbol_sz {}", symbol_sz);
    NVLOGI_FMT(TAG_ORDER_TB_BASE,"prb_sz {}", prb_sz);
    NVLOGI_FMT(TAG_ORDER_TB_BASE,"prbs_per_symbol {}", prbs_per_symbol);
#endif

    ret.raw_data = std::move(d);
    return ret;
}

void OrderKernelTestBench::load_ul_qams(hdf5hpp::hdf5_file& hdf5file, ul_tv_object& tv_object, ul_tv_info& tv_info, std::string& tv_name)
{
    if(hdf5file.is_valid_dataset("X_tf_fp16"))
    {
        int slot_index=tv_object.slots[BFP_NO_COMPRESSION].size();
        //Update the slot num to slot index map
        for(int cell_idx=0;cell_idx<num_test_cells;cell_idx++)
        {
            for (auto it = ok_tb_tv_params.tv_to_slot_map[cell_idx].begin(); it !=ok_tb_tv_params.tv_to_slot_map[cell_idx].end();)
            {
                std::string key = it->first;
                std::string key_full_path;
                char tv_full_path[MAX_PATH_LEN];
                get_full_path_file(tv_full_path, CONFIG_TEST_VECTOR_PATH, key.c_str(), CONFIG_CUBB_ROOT_DIR_RELATIVE_NUM);
                key_full_path = std::string(tv_full_path);
                auto range =    ok_tb_tv_params.tv_to_slot_map[cell_idx].equal_range(key);
                //NVLOGC_FMT(TAG_ORDER_TB_BASE,"[load_ul_qams] key {} tv_name {}",key,tv_name);
                if(0==key_full_path.compare(tv_name))
                {
                    NVLOGI_FMT(TAG_ORDER_TB_BASE,"[load_ul_qams] key {} tv_name {} inside compare",key,tv_name);
                    for (auto it2 = range.first; it2 != range.second; ++it2)
                    {
                        ok_tb_tv_params.slot_num_to_slot_idx_map[cell_idx].insert({it2->second,slot_index});
                        //NVLOGC_FMT(TAG_ORDER_TB_BASE,"[load_ul_qams] After Inserting TV {} against slot num {} with slot index {}",tv_name,it->second,slot_index);
                    }
                    break;
                }
                else
                {
                    it = range.second; // Move to the next unique key
                }
            }
        }
        Dataset d = std::move(load_tv_datasets_single(hdf5file, " ", "X_tf_fp16"));
        tv_object.slots[BFP_NO_COMPRESSION].emplace_back(dataset_to_slot(std::move(d), tv_info.numFlows, 0, SLOT_NUM_SYMS, PRB_SIZE_16F, tv_info.nPrbUlBwp, 0));
    }
    else
    {
        NVLOGC_FMT(TAG_ORDER_TB_BASE,"HDF File missing X_tf_fp16");
    }
}

void OrderKernelTestBench::load_ul_qams_prach(hdf5hpp::hdf5_file& hdf5file, ul_tv_object& tv_object, ul_tv_info& tv_info, std::string& tv_name)
{
    if(hdf5file.is_valid_dataset("X_tf_prach_1_fp16"))
    {
        int slot_index=tv_object.slots[BFP_NO_COMPRESSION].size();
        //Update the slot num to slot index map
        for(int cell_idx=0;cell_idx<num_test_cells;cell_idx++)
        {
            for (auto it = ok_tb_tv_params.tv_to_slot_map[cell_idx].begin(); it !=ok_tb_tv_params.tv_to_slot_map[cell_idx].end();)
            {
                std::string key = it->first;
                std::string key_full_path;
                char tv_full_path[MAX_PATH_LEN];
                get_full_path_file(tv_full_path, CONFIG_TEST_VECTOR_PATH, key.c_str(), CONFIG_CUBB_ROOT_DIR_RELATIVE_NUM);
                key_full_path = std::string(tv_full_path);
                auto range =    ok_tb_tv_params.tv_to_slot_map[cell_idx].equal_range(key);
                //NVLOGC_FMT(TAG_ORDER_TB_BASE,"[load_ul_qams] key {} tv_name {}",key,tv_name);
                if(0==key_full_path.compare(tv_name))
                {
                    NVLOGI_FMT(TAG_ORDER_TB_BASE,"[load_ul_qams] key {} tv_name {} inside compare",key,tv_name);
                    for (auto it2 = range.first; it2 != range.second; ++it2)
                    {
                        ok_tb_tv_params.prach_slot_num_to_slot_idx_map[cell_idx].insert({it2->second,slot_index});
                        NVLOGI_FMT(TAG_ORDER_TB_BASE,"[load_ul_qams_prach] After Inserting TV {} against slot num {} with slot index {}",tv_name,it2->second,slot_index);
                    }
                    break;
                }
                else
                {
                    it = range.second; // Move to the next unique key
                }
            }
        }
        Dataset d_1 = std::move(load_tv_datasets_single(hdf5file, " ", "X_tf_prach_1_fp16"));
        Dataset d_2 = std::move(load_tv_datasets_single(hdf5file, " ", "X_tf_prach_2_fp16"));
        Dataset d_3 = std::move(load_tv_datasets_single(hdf5file, " ", "X_tf_prach_3_fp16"));
        Dataset d_4 = std::move(load_tv_datasets_single(hdf5file, " ", "X_tf_prach_3_fp16"));

        tv_object.slots[BFP_NO_COMPRESSION].emplace_back(dataset_to_slot_prach(std::move(d_1),std::move(d_2),std::move(d_3),std::move(d_4),tv_info.numFlows, tv_info.startSym, tv_info.numSym, PRB_SIZE_16F, tv_info.numPrb, tv_info.startPrb));
    }
    else
    {
        NVLOGC_FMT(TAG_ORDER_TB_BASE,"HDF File missing X_tf_prach_1_fp16");
    }
}

void OrderKernelTestBench::load_pusch_tvs()
{

    // Load the requested test vectors
    // uplink_tv_flow_count = std::vector<uint16_t>(pusch_tvs.size());
    for (int i = 0; i < ok_tb_tv_params.pusch_object.tv_names.size(); ++i)
    {
        struct ul_tv_info ul_tv_info;
        std::string dataset_name;

        hdf5hpp::hdf5_file hdf5file = hdf5hpp::hdf5_file::open(ok_tb_tv_params.pusch_object.tv_names[i].c_str());
        read_cell_cfg_from_tv(hdf5file, ul_tv_info, ok_tb_tv_params.pusch_object.tv_names[i]);

        int count = 1;
        std::string pdu = "PDU";
        bool pusch_found = false;
        ul_tv_info.tb_size = 0;
        ul_tv_info.numPrb = 0;
        while(1)
        {
            std::string dset_string = pdu + std::to_string(count);

            if(!hdf5file.is_valid_dataset(dset_string.c_str()))
            {
                if(pusch_found)
                {
                    break;
                }
                NVLOGW_FMT(TAG_ORDER_TB_BASE,"ERROR No PUSCH PDU found in TV {}",ok_tb_tv_params.pusch_object.tv_names[i]);
            }
            hdf5hpp::hdf5_dataset dset_PDU  = hdf5file.open_dataset(dset_string.c_str());
            hdf5hpp::hdf5_dataset_elem pdu_pars = dset_PDU[0];
            if(pdu_pars["type"].as<uint8_t>() != nrsim_tv_type::PUSCH)
            {
                count++;
                continue;
            }
            pusch_found = true;

            pdu_info pdu_info;
            pdu_info.startSym = pdu_pars["StartSymbolIndex"].as<uint8_t>();
            pdu_info.numSym = pdu_pars["NrOfSymbols"].as<uint8_t>();
            pdu_info.startPrb = pdu_pars["rbStart"].as<uint16_t>() + pdu_pars["BWPStart"].as<uint16_t>();
            pdu_info.numPrb = pdu_pars["rbSize"].as<uint16_t>();
            pdu_info.numFlows = load_num_antenna_from_nr_tv(hdf5file);
            pdu_info.tb_size = pdu_pars["TBSize"].as<uint32_t>();
            pdu_info.dmrsPorts = pdu_pars["dmrsPorts"].as<uint8_t>();
            pdu_info.scid = pdu_pars["SCID"].as<uint8_t>();
            ul_tv_info.tb_size += pdu_info.tb_size;

            if (enable_mimo) //TODO: mMIMO to bbe supported
            {
                //get_ul_ports(pdu_pars, pdu_info);
                //ul_tv_info.numPrb += pdu_info.numPrb * pdu_info.numSym * pdu_info.flow_indices.size();
            }

            auto found = [&pdu_info] (const struct pdu_info& pdu) {
                return (pdu.startSym == pdu_info.startSym && pdu.numSym == pdu_info.numSym && pdu.startPrb == pdu_info.startPrb && pdu.numPrb ==  pdu_info.numPrb);
            };

            auto iter = std::find_if(ul_tv_info.pdu_infos.begin(), ul_tv_info.pdu_infos.end(), found);
            if(iter == ul_tv_info.pdu_infos.end())
            {
                ++ul_tv_info.numSections;
                /* Currently the numPrb calculation is based on the assuption that the
                 * symbol range and prb range are the same for each PDU. Need to revisit
                 * this implementation once we have different test case.
                 */

                /* Taking out flow multiplier because a 4 antenna TV could be configured for 2
                 * antennas on the FH if there is no traffic on other antennas. Adding the
                 * cell_configs num_ul_flow multiplier at the completion check
                 */

                if (!enable_mimo)
                {
                    ul_tv_info.numPrb += pdu_info.numPrb * pdu_info.numSym; // * pdu_info.numFlows;}
                }
            }

            ul_tv_info.pdu_infos.emplace_back(std::move(pdu_info));
            count++;
        }
        ul_tv_info.numFlows = load_num_antenna_from_nr_tv(hdf5file);
        NVLOGI_FMT(TAG_ORDER_TB_BASE,"Loading PUSCH data from TV {}",ok_tb_tv_params.pusch_object.tv_names[i].c_str());
        load_ul_qams(hdf5file, ok_tb_tv_params.pusch_object, ul_tv_info,ok_tb_tv_params.pusch_object.tv_names[i]);
        ok_tb_tv_params.pusch_object.tv_info.emplace_back(ul_tv_info);
    }
}

void OrderKernelTestBench::load_prach_tvs()
{
    // Load the requested test vectors
    // uplink_tv_flow_count = std::vector<uint16_t>(pusch_tvs.size());
    for (int i = 0; i < ok_tb_tv_params.prach_object.tv_names.size(); ++i)
    {
        struct ul_tv_info ul_tv_info;
        std::string base_x_tf_dataset_name;

        hdf5hpp::hdf5_file hdf5file = hdf5hpp::hdf5_file::open(ok_tb_tv_params.prach_object.tv_names[i].c_str());
        read_cell_cfg_from_tv(hdf5file, ul_tv_info, ok_tb_tv_params.prach_object.tv_names[i]);

        int count = 1;
        std::string pdu = "PDU";
        std::string ro_config = "RO_Config_";
        bool prach_found = false;
        uint8_t numFlows = 0;

        while(1)
        {
            std::string dset_string = pdu + std::to_string(count);
            std::string ro_dset_string = ro_config + std::to_string(count);
            if(!hdf5file.is_valid_dataset(dset_string.c_str()))
            {
                if(!prach_found)
                {
                    NVLOGW_FMT(TAG_ORDER_TB_BASE,"ERROR No PRACH PDU found in TV {}",ok_tb_tv_params.prach_object.tv_names[i]);
                }
                break;
            }

            hdf5hpp::hdf5_dataset dset_PDU  = hdf5file.open_dataset(dset_string.c_str());
            hdf5hpp::hdf5_dataset_elem pdu_pars = dset_PDU[0];

            if(pdu_pars["type"].as<uint8_t>() != nrsim_tv_type::PRACH)
            {
                count++;
                continue;
            }
            prach_found = true;

            pdu_info pdu_info;
            if(hdf5file.is_valid_dataset(ro_dset_string.c_str()))
            {
                hdf5hpp::hdf5_dataset dset_ro_config  = hdf5file.open_dataset(ro_dset_string.c_str());
                hdf5hpp::hdf5_dataset_elem ro_pars = dset_ro_config[0];
                pdu_info.startPrb = ro_pars["k1"].as<uint16_t>();
            }

            base_x_tf_dataset_name = "X_tf_prach_" + std::to_string(count);
            pdu_info.startSym = pdu_pars["prachStartSymbol"].as<uint8_t>();;
            pdu_info.numSym = 12;
            pdu_info.numPrb = 12;
            pdu_info.numFlows = load_num_antenna_from_nr_prach_tv(hdf5file, base_x_tf_dataset_name);
            numFlows = pdu_info.numFlows;

            ul_tv_info.pdu_infos.emplace_back(std::move(pdu_info));
            count++;
        }
        ul_tv_info.startPrb = 0;
        ul_tv_info.numPrb = 12;
        ul_tv_info.startSym = 0;
        ul_tv_info.numSym = 12;
        ul_tv_info.numFlows = numFlows;
        NVLOGC_FMT(TAG_ORDER_TB_BASE, "Loading PRACH data from TV {}",ok_tb_tv_params.prach_object.tv_names[i].c_str());
        load_ul_qams_prach(hdf5file, ok_tb_tv_params.prach_object, ul_tv_info,ok_tb_tv_params.prach_object.tv_names[i]);
        ok_tb_tv_params.prach_object.tv_info.emplace_back(ul_tv_info);
    }
}

int OrderKernelTestBench::load_num_antenna_from_nr_tv_srs(hdf5hpp::hdf5_file& hdf5file)
{
    hdf5hpp::hdf5_dataset dset = hdf5file.open_dataset("X_tf_srs_fp16");

    if(dset.get_dataspace().get_dimensions().size() < 3)
    {
        return 1;
    }
    return dset.get_dataspace().get_dimensions()[0];
}

int OrderKernelTestBench::load_num_antenna_from_nr_prach_tv(hdf5hpp::hdf5_file& hdf5file, std::string dset)
{
    hdf5hpp::hdf5_dataset ds = hdf5file.open_dataset(dset.c_str());

    if(ds.get_dataspace().get_dimensions().size() < 2)
    {
        return 1;
    }
    return ds.get_dataspace().get_dimensions()[0];
}

void OrderKernelTestBench::load_srs_tvs()
{
    for (int i = 0; i < ok_tb_tv_params.srs_object.tv_names.size(); ++i)
    {
        struct ul_tv_info ul_tv_info;
        std::string dataset_name;
        hdf5hpp::hdf5_file hdf5file = hdf5hpp::hdf5_file::open(ok_tb_tv_params.srs_object.tv_names[i].c_str());
        read_cell_cfg_from_tv(hdf5file, ul_tv_info, ok_tb_tv_params.srs_object.tv_names[i]);

        int count = 1;
        std::string pdu = "PDU";
        bool srs_found = false;
        ul_tv_info.tb_size = 0;
        ul_tv_info.endPrb = 0;
        ul_tv_info.startPrb = 273;
        ul_tv_info.numPrb = 0;

        while(1)
        {
            std::string dset_string = pdu + std::to_string(count);
            if(!hdf5file.is_valid_dataset(dset_string.c_str()))
            {
                if(srs_found)
                {
                    break;
                }
                NVLOGW_FMT(TAG_ORDER_TB_BASE,"ERROR No SRS PDU found in TV {}",ok_tb_tv_params.srs_object.tv_names[i]);
            }
            hdf5hpp::hdf5_dataset dset_PDU  = hdf5file.open_dataset(dset_string.c_str());
            hdf5hpp::hdf5_dataset_elem pdu_pars = dset_PDU[0];
            if(pdu_pars["type"].as<uint8_t>() != nrsim_tv_type::SRS)
            {
                count++;
                continue;
            }
            srs_found = true;
            count++;
        }        

        ul_tv_info.numFlows = load_num_antenna_from_nr_tv_srs(hdf5file);
        NVLOGC_FMT(TAG_ORDER_TB_BASE,"LOADING SRS TV {}", ok_tb_tv_params.srs_object.tv_names[i].c_str());
        load_ul_qams(hdf5file, ok_tb_tv_params.srs_object, ul_tv_info,ok_tb_tv_params.srs_object.tv_names[i]);
        ok_tb_tv_params.srs_object.tv_info.emplace_back(ul_tv_info);
        hdf5file.close();    
    }
}

void OrderKernelTestBench::write_output_file()
{
    std::ofstream file(output_file_);
    float average_process_run_time=0;
    if(enable_srs)
    {
        file << "Num_cells,Slot_count,frameId,subframeId,slotId,srs_mismatch_count(IQ),Duration(us)\n";
        for(int slot_idx=0;slot_idx<num_test_slots;slot_idx++)
        {
            file<<num_test_cells<<","<<slot_idx<<","<<slot_info[slot_idx].frameId<<","<<slot_info[slot_idx].subframeId<<","<<slot_info[slot_idx].slotId<<","<<slot_info[slot_idx].srs_mismatch_count<<","<<process_dur_us[slot_idx]<<"\n";
            if(slot_idx>0)
            {
                average_process_run_time+=process_dur_us[slot_idx];
            }
        }        
    }
    else
    {
        file << "Num_cells,Slot_count,frameId,subframeId,slotId,pusch_pucch_mismatch_count(IQ),prach_mismatch_count(IQ),Duration(us)\n";
        for(int slot_idx=0;slot_idx<num_test_slots;slot_idx++)
        {
            file<<num_test_cells<<","<<slot_idx<<","<<slot_info[slot_idx].frameId<<","<<slot_info[slot_idx].subframeId<<","<<slot_info[slot_idx].slotId<<","<<slot_info[slot_idx].pusch_pucch_mismatch_count<<","<<slot_info[slot_idx].prach_mismatch_count<<","<<process_dur_us[slot_idx]<<"\n";
            if(slot_idx>0)
            {
                average_process_run_time+=process_dur_us[slot_idx];
            }
        }        
    }

    if(num_test_slots>1)
    {
        average_process_run_time=average_process_run_time/(num_test_slots-1);
    }
    file << "Average Process kernel run duration(us)," << average_process_run_time <<"\n";
    file.close();
}

void OrderKernelTestBench::get_process_kernel_run_duration(int slot_count)
{
    float ms = 0;

    CUDA_CHECK(cudaEventQuery(start_ok_tb_process));
    CUDA_CHECK(cudaEventQuery(end_ok_tb_process));
    CUDA_CHECK(cudaEventElapsedTime(&ms, start_ok_tb_process, end_ok_tb_process));

    process_dur_us[slot_count]=ms*1000;
    NVLOGC_FMT(TAG_ORDER_TB_BASE,"Slot count {} Process kernel duration {} us",slot_count,process_dur_us[slot_count]);
}

void OrderKernelTestBench::save_process_kernel_run_info(int slot_count,slotInfo_t& slot_info_curr)
{
  slot_info[slot_count]=slot_info_curr;
}

int OrderKernelTestBench::run_test()
{
    if (initStatus) {
      NVLOGE_FMT(TAG_ORDER_TB_RUN, AERIAL_TESTBENCH_EVENT, "Exiting with error ... ");
      return 1;
    }

    NVLOGC_FMT(TAG_ORDER_TB_RUN, "Run launch_process_kernel_for_test_bench");
    //cudaFree(0);
    //cudaSetDevice(0);

    setup_input_params();
    read_ok_tb_config_file_params();
    setup_config_params();

    size_t max_pkts_size_per_cell = MAX_PKTS_PER_SLOT_OK_TB*ok_tb_max_packet_size*MAX_UL_SLOTS_OK_TB;
    
    //Read FH packets from binary file and copy into device memory before trigerring the process kernel
    std::array<uint8_t*,UL_MAX_CELLS_PER_SLOT> buf_ok_tb;
    std::ifstream file;
    file.clear();
    for(int cell_idx=0;cell_idx<num_test_cells;cell_idx++)
    {
        file.open(binary_file_[cell_idx], std::ios::binary);
        if (!file) {
            NVLOGE_FMT(TAG_ORDER_TB_RUN, AERIAL_TESTBENCH_EVENT, "Error reading binary file for cell idx {}", cell_idx);
            return 1;
        }
        buf_ok_tb[cell_idx]=new uint8_t[max_pkts_size_per_cell];
        file.read(reinterpret_cast<char*>(buf_ok_tb[cell_idx]), max_pkts_size_per_cell);
        CUDA_CHECK(cudaMemcpy(fh_buf_ok_tb[cell_idx], buf_ok_tb[cell_idx], max_pkts_size_per_cell, cudaMemcpyHostToDevice));
        file.close();
        file.clear();
    }

    int start_slot_idx=start_test_slot%ok_tb_num_valid_slots;
    for(int slot_count=0;slot_count<num_test_slots;slot_count++)
    {
        for(int cell_idx=0;cell_idx<num_test_cells;cell_idx++)
        {
            std::vector<uint16_t> eAxC_map_pusch(max_rx_ant, 0);
            std::vector<uint16_t> eAxC_map_prach(max_rx_ant, 0);
            std::vector<uint16_t> eAxC_map_srs(MAX_RX_ANT_SRS_64T64R, 0);
            ok_tb_config_params->cell_id[cell_idx]=ok_tb_config_file_params[cell_idx].cell_id[start_slot_idx];
            if(enable_srs)
            {
                ok_tb_config_params->srs_prb_x_slot[cell_idx]=ok_tb_config_file_params[cell_idx].num_srs_prbs[start_slot_idx];
                CUDA_CHECK(cudaMemset((void*)ok_tb_input_params.srs_ordered_prbs[cell_idx],0,sizeof(uint32_t)));
                ok_tb_config_params->srs_eAxC_num[cell_idx]=ok_tb_config_file_params[cell_idx].srs_eAxC_num[start_slot_idx];
                for(int tmp=0;tmp<ok_tb_config_file_params[cell_idx].srs_eAxC_num[start_slot_idx];tmp++)
                {
                    eAxC_map_srs[tmp]=ok_tb_config_file_params[cell_idx].srs_eAxC_map[start_slot_idx][tmp];
                }
                CUDA_CHECK(cudaMemcpy(ok_tb_input_params.srs_eAxC_map[cell_idx],eAxC_map_srs.data(),MAX_RX_ANT_SRS_64T64R*sizeof(uint16_t),cudaMemcpyHostToDevice));
            }
            else
            {
                ok_tb_config_params->pusch_prb_x_slot[cell_idx]=ok_tb_config_file_params[cell_idx].num_pusch_prbs[start_slot_idx];
                ok_tb_config_params->prach_prb_x_slot[cell_idx]=ok_tb_config_file_params[cell_idx].num_prach_prbs[start_slot_idx];
                CUDA_CHECK(cudaMemset((void*)ok_tb_input_params.pusch_ordered_prbs[cell_idx],0,sizeof(uint32_t)));
                CUDA_CHECK(cudaMemset((void*)ok_tb_input_params.prach_ordered_prbs[cell_idx],0,sizeof(uint32_t)));      
                ok_tb_config_params->pusch_eAxC_num[cell_idx]=ok_tb_config_file_params[cell_idx].pusch_eAxC_num[start_slot_idx];
                ok_tb_config_params->prach_eAxC_num[cell_idx]=ok_tb_config_file_params[cell_idx].prach_eAxC_num[start_slot_idx];
                for(int tmp=0;tmp<ok_tb_config_file_params[cell_idx].pusch_eAxC_num[start_slot_idx];tmp++)
                {
                    eAxC_map_pusch[tmp]=ok_tb_config_file_params[cell_idx].pusch_eAxC_map[start_slot_idx][tmp];
                }
                CUDA_CHECK(cudaMemcpy(ok_tb_input_params.pusch_eAxC_map[cell_idx],eAxC_map_pusch.data(),max_rx_ant*sizeof(uint16_t),cudaMemcpyHostToDevice));
                for(int tmp=0;tmp<ok_tb_config_file_params[cell_idx].prach_eAxC_num[start_slot_idx];tmp++)
                {
                    eAxC_map_prach[tmp]=ok_tb_config_file_params[cell_idx].prach_eAxC_map[start_slot_idx][tmp];
                }
                CUDA_CHECK(cudaMemcpy(ok_tb_input_params.prach_eAxC_map[cell_idx],eAxC_map_prach.data(),max_rx_ant*sizeof(uint16_t),cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaMemcpy((ok_tb_input_params.pusch_prb_symbol_map+cell_idx*ORAN_PUSCH_SYMBOLS_X_SLOT),ok_tb_config_file_params[cell_idx].pusch_prb_symbol_map[start_slot_idx],ORAN_PUSCH_SYMBOLS_X_SLOT*sizeof(uint32_t),cudaMemcpyHostToDevice));                      
            }
            ok_tb_config_params->rx_pkt_num_slot[cell_idx]=ok_tb_config_file_params[cell_idx].num_rx_packets[start_slot_idx];
            ok_tb_config_params->tb_fh_buf[cell_idx]=fh_buf_ok_tb[cell_idx]+(start_slot_idx*MAX_PKTS_PER_SLOT_OK_TB*ok_tb_max_packet_size);
            *ok_tb_input_params.exit_cond_d[cell_idx]=ORDER_KERNEL_RUNNING;
            ok_tb_config_params->exit_cond_d[cell_idx]=ok_tb_input_params.exit_cond_d[cell_idx];
            ok_tb_config_params->sem_gpu[cell_idx]->pkt_info_gpu[0].status=DOCA_GPU_SEMAPHORE_STATUS_READY;
            ok_tb_config_params->sem_gpu_aerial_fh[cell_idx]->pkt_info_gpu[slot_count%MAX_SEM_ITEMS].status=AERIAL_FH_GPU_SEMAPHORE_STATUS_READY;


            OrderKernelSetupDocaParams(cell_idx);            
            cudaMemcpy(ok_tb_input_params.doca_rxq[cell_idx],&doca_rxq_info[cell_idx],sizeof(struct doca_gpu_eth_rxq),cudaMemcpyHostToDevice); 
            cudaMemcpy(&ok_tb_input_params.doca_rxq[cell_idx]->cqe_addr,&ok_tb_input_params.cqe_addr[cell_idx],sizeof(struct mlx5_cqe*),cudaMemcpyHostToDevice);
            cudaMemcpy(&ok_tb_input_params.doca_rxq[cell_idx]->cq_db_rec,&ok_tb_input_params.cq_db_rec[cell_idx],sizeof(uint32_t*),cudaMemcpyHostToDevice);
            cudaMemcpy(&ok_tb_input_params.doca_rxq[cell_idx]->rq_db_rec,&ok_tb_input_params.rq_db_rec[cell_idx],sizeof(uint32_t*),cudaMemcpyHostToDevice);
        }
        ok_tb_config_params->frameId = ok_tb_config_file_params[0].frameId[start_slot_idx];
        ok_tb_config_params->subframeId = ok_tb_config_file_params[0].subframeId[start_slot_idx];
        ok_tb_config_params->slotId = ok_tb_config_file_params[0].slotId[start_slot_idx];
        ok_tb_config_params->timeout_no_pkt_ns=3000000;
        ok_tb_config_params->timeout_first_pkt_ns=ok_tb_config_params->timeout_no_pkt_ns/2;
        ok_tb_config_params->timeout_log_interval_ns=0;
        ok_tb_config_params->timeout_log_enable=1;
        ok_tb_config_params->commViaCpu=false;
        ok_tb_config_params->max_rx_pkts=512;
        ok_tb_config_params->rx_pkts_timeout_ns=200000;
        ok_tb_config_params->ul_rx_pkt_tracing_level=0;
        ok_tb_config_params->ul_order_kernel_mode=0;//Dual CTA mode
        if(!enable_srs)
        {
            CUDA_CHECK(cudaMemcpy(ok_tb_input_params.num_order_cells_sym_mask_arr,ok_tb_config_file_params[0].num_order_cells_sym_mask[start_slot_idx],ORAN_PUSCH_SYMBOLS_X_SLOT*sizeof(uint32_t),cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemset((void*)ok_tb_input_params.sym_ord_done_sig_arr,SYM_RX_NOT_DONE,ORAN_PUSCH_SYMBOLS_X_SLOT*sizeof(uint32_t)));
            CUDA_CHECK(cudaMemset((void*)ok_tb_input_params.sym_ord_done_mask_arr,0,ORAN_PUSCH_SYMBOLS_X_SLOT*sizeof(uint32_t)));
        }

        CUDA_CHECK(cudaEventRecord(start_ok_tb_process, stream_oktb));
        //printf("Before launching process kernel ok_tb_input_params.pusch_buffer[cell_idx] = 0x%p\n",(void*)ok_tb_input_params.pusch_buffer[0]);
        launch_receive_process_kernel_for_test_bench(
            stream_oktb,
            /* Cell */
            ok_tb_config_params->cell_id,
            ok_tb_config_params->exit_cond_d,
            ok_tb_config_params->sem_order_num,
            ok_tb_config_params->ru_type,
            /* ORAN */
            ok_tb_config_params->frameId,
            ok_tb_config_params->subframeId,
            ok_tb_config_params->slotId,

            ok_tb_config_params->prb_size,
            ok_tb_config_params->comp_meth,
            ok_tb_config_params->bit_width,
            ok_tb_config_params->beta,
            ok_tb_config_params->last_sem_idx_order_h,

            ok_tb_config_params->rx_pkt_num_slot,
            ok_tb_config_params->tb_fh_buf,
            ok_tb_config_params->max_pkt_size,

            ok_tb_config_params->early_rx_packets,
            ok_tb_config_params->on_time_rx_packets,
            ok_tb_config_params->late_rx_packets,
            ok_tb_config_params->next_slot_early_rx_packets,
            ok_tb_config_params->next_slot_on_time_rx_packets,
            ok_tb_config_params->next_slot_late_rx_packets,
            ok_tb_config_params->rx_packets_dropped_count,
            ok_tb_config_params->cell_health,
            ok_tb_config_params->start_cuphy_d,

            /* Sub-slot processing*/
            ok_tb_config_params->sym_ord_done_sig_arr,
            ok_tb_config_params->sym_ord_done_mask_arr,
            ok_tb_config_params->pusch_prb_symbol_map,
            ok_tb_config_params->num_order_cells_sym_mask_arr,

            /*PUSCH*/
            ok_tb_config_params->pusch_buffer,
            ok_tb_config_params->pusch_eAxC_map,
            ok_tb_config_params->pusch_eAxC_num,
            ok_tb_config_params->pusch_symbols_x_slot,
            ok_tb_config_params->pusch_prb_x_port_x_symbol,
            ok_tb_config_params->pusch_ordered_prbs,
            ok_tb_config_params->pusch_prb_x_slot,

            /*PRACH*/
            ok_tb_config_params->prach_eAxC_map,
            ok_tb_config_params->prach_eAxC_num,
            ok_tb_config_params->prach_buffer_0,
            ok_tb_config_params->prach_buffer_1,
            ok_tb_config_params->prach_buffer_2,
            ok_tb_config_params->prach_buffer_3,
            ok_tb_config_params->prach_prb_x_slot,
            ok_tb_config_params->prach_symbols_x_slot,
            ok_tb_config_params->prach_prb_x_port_x_symbol,
            ok_tb_config_params->prach_ordered_prbs,
            ok_tb_config_params->prach_section_id_0,
            ok_tb_config_params->prach_section_id_1,
            ok_tb_config_params->prach_section_id_2,
            ok_tb_config_params->prach_section_id_3,
            ok_tb_config_params->num_order_cells,

            /*SRS*/
            ok_tb_config_params->srs_eAxC_map,
            ok_tb_config_params->srs_eAxC_num,
            ok_tb_config_params->srs_buffer,
            ok_tb_config_params->srs_prb_x_slot,
            ok_tb_config_params->srs_prb_stride,
            ok_tb_config_params->srs_ordered_prbs,
            ok_tb_config_params->srs_start_sym,

            /*Receive CTA params*/            
            ok_tb_config_params->timeout_no_pkt_ns,
            ok_tb_config_params->timeout_first_pkt_ns,
            ok_tb_config_params->timeout_log_interval_ns,
            ok_tb_config_params->timeout_log_enable,
            ok_tb_config_params->order_kernel_last_timeout_error_time,
            ok_tb_config_params->last_sem_idx_rx_h,
            ok_tb_config_params->commViaCpu,
            ok_tb_config_params->doca_rxq,
            ok_tb_config_params->max_rx_pkts,
            ok_tb_config_params->rx_pkts_timeout_ns,
            ok_tb_config_params->sem_gpu,
            ok_tb_config_params->sem_gpu_aerial_fh,
            ok_tb_config_params->slot_start,
            ok_tb_config_params->ta4_min_ns,
            ok_tb_config_params->ta4_max_ns,
            ok_tb_config_params->slot_duration,
            ok_tb_config_params->ul_rx_pkt_tracing_level,
            ok_tb_config_params->ul_order_kernel_mode,
            enable_srs
        );
        CUDA_CHECK(cudaEventRecord(end_ok_tb_process, stream_oktb));
        for(int cell_idx=0;cell_idx<num_test_cells;cell_idx++){
            if(enable_srs)
            {
                CUDA_CHECK(cudaMemcpyAsync(ok_tb_input_params.srs_buffer_h[cell_idx],ok_tb_input_params.srs_buffer[cell_idx],UL_ST2_AP_BUF_SIZE*MAX_RX_ANT_SRS_64T64R,cudaMemcpyDeviceToHost,stream_oktb));
            }
            else {
                CUDA_CHECK(cudaMemcpyAsync(ok_tb_input_params.pusch_buffer_h[cell_idx],ok_tb_input_params.pusch_buffer[cell_idx],UL_ST1_AP_BUF_SIZE*max_rx_ant,cudaMemcpyDeviceToHost,stream_oktb));
                if(ok_tb_config_params->prach_prb_x_slot[cell_idx]>0)
                {
                    CUDA_CHECK(cudaMemcpyAsync(ok_tb_input_params.prach_buffer_0_h[cell_idx],ok_tb_input_params.prach_buffer_0[cell_idx],UL_ST3_AP_BUF_SIZE*max_rx_ant,cudaMemcpyDeviceToHost,stream_oktb));
                    CUDA_CHECK(cudaMemcpyAsync(ok_tb_input_params.prach_buffer_1_h[cell_idx],ok_tb_input_params.prach_buffer_1[cell_idx],UL_ST3_AP_BUF_SIZE*max_rx_ant,cudaMemcpyDeviceToHost,stream_oktb));
                    CUDA_CHECK(cudaMemcpyAsync(ok_tb_input_params.prach_buffer_2_h[cell_idx],ok_tb_input_params.prach_buffer_2[cell_idx],UL_ST3_AP_BUF_SIZE*max_rx_ant,cudaMemcpyDeviceToHost,stream_oktb));
                }            
            }
        }
        cudaStreamSynchronize(stream_oktb);
        get_process_kernel_run_duration(slot_count);
        std::array<uint32_t,UL_MAX_CELLS_PER_SLOT> mis_match_counter;
        slotInfo_t slot_info_curr;
        int pusch_pucch_mm_counter=0;
        int prach_mm_counter=0;
        int srs_mm_counter=0;
        mis_match_counter.fill(0);
        if(!enable_srs)
        {
            //PUSCH/PUCCH buffer comparison
            if(0==(compare_iq(start_slot_idx,ok_tb_config_params->frameId,ok_tb_config_params->subframeId,ok_tb_config_params->slotId,mis_match_counter,false)))
            {
                NVLOGC_FMT(TAG_ORDER_TB_BASE, "[PUSCH/PUCCH]IQ samples match for F{}S{}S{}!!!",ok_tb_config_params->frameId,ok_tb_config_params->subframeId,ok_tb_config_params->slotId);
            }
            else
            {
                for(int cell_idx=0;cell_idx<num_test_cells;cell_idx++)
                {
                    NVLOGC_FMT(TAG_ORDER_TB_BASE, "[PUSCH/PUCCH]IQ samples mis-match for F{}S{}S{}!!! Cell wise break up as follows",ok_tb_config_params->frameId,ok_tb_config_params->subframeId,ok_tb_config_params->slotId);
                    if(mis_match_counter[cell_idx]>0)
                    {
                        NVLOGC_FMT(TAG_ORDER_TB_BASE, "[PUSCH/PUCCH]IQ samples mis-match for Cell Idx {} F{}S{}S{}!!! mis_match_count {}",cell_idx,ok_tb_config_params->frameId,ok_tb_config_params->subframeId,ok_tb_config_params->slotId,mis_match_counter[cell_idx]);
                    }
                    else
                    {
                        NVLOGC_FMT(TAG_ORDER_TB_BASE, "[PUSCH/PUCCH]IQ samples match for Cell Idx {} F{}S{}S{}!!!",cell_idx,ok_tb_config_params->frameId,ok_tb_config_params->subframeId,ok_tb_config_params->slotId);
                    }
                }
            }
            pusch_pucch_mm_counter=std::accumulate(mis_match_counter.begin(),mis_match_counter.end(),0);
            //PRACH buffer comparison
            if(ok_tb_config_params->prach_prb_x_slot[0]>0) //Assume PRACH is present in all cells if cell_idx=0 satisfies condition
            {
                mis_match_counter.fill(0);
                if(0==(compare_iq(start_slot_idx,ok_tb_config_params->frameId,ok_tb_config_params->subframeId,ok_tb_config_params->slotId,mis_match_counter,true)))
                {
                    NVLOGC_FMT(TAG_ORDER_TB_BASE, "[PRACH]IQ samples match for F{}S{}S{}!!!",ok_tb_config_params->frameId,ok_tb_config_params->subframeId,ok_tb_config_params->slotId);
                }
                else
                {
                    for(int cell_idx=0;cell_idx<num_test_cells;cell_idx++)
                    {
                        NVLOGC_FMT(TAG_ORDER_TB_BASE, "[PRACH]IQ samples mis-match for F{}S{}S{}!!! Cell wise break up as follows",ok_tb_config_params->frameId,ok_tb_config_params->subframeId,ok_tb_config_params->slotId);
                        if(mis_match_counter[cell_idx]>0)
                        {
                            NVLOGC_FMT(TAG_ORDER_TB_BASE, "[PRACH]IQ samples mis-match for Cell Idx {} F{}S{}S{}!!! mis_match_count {}",cell_idx,ok_tb_config_params->frameId,ok_tb_config_params->subframeId,ok_tb_config_params->slotId,mis_match_counter[cell_idx]);
                        }
                        else
                        {
                            NVLOGC_FMT(TAG_ORDER_TB_BASE, "[PRACH]IQ samples match for Cell Idx {} F{}S{}S{}!!!",cell_idx,ok_tb_config_params->frameId,ok_tb_config_params->subframeId,ok_tb_config_params->slotId);
                        }
                    }
                }
                prach_mm_counter=std::accumulate(mis_match_counter.begin(),mis_match_counter.end(),0);
            }
        }
        else
        {
            //SRS buffer comparison
            if(0==(compare_iq(start_slot_idx,ok_tb_config_params->frameId,ok_tb_config_params->subframeId,ok_tb_config_params->slotId,mis_match_counter,false)))
            {
                NVLOGC_FMT(TAG_ORDER_TB_BASE, "[SRS]IQ samples match for F{}S{}S{}!!!",ok_tb_config_params->frameId,ok_tb_config_params->subframeId,ok_tb_config_params->slotId);
            }
            else
            {
                for(int cell_idx=0;cell_idx<num_test_cells;cell_idx++)
                {
                    NVLOGC_FMT(TAG_ORDER_TB_BASE, "[SRS]IQ samples mis-match for F{}S{}S{}!!! Cell wise break up as follows",ok_tb_config_params->frameId,ok_tb_config_params->subframeId,ok_tb_config_params->slotId);
                    if(mis_match_counter[cell_idx]>0)
                    {
                        NVLOGC_FMT(TAG_ORDER_TB_BASE, "[SRS]IQ samples mis-match for Cell Idx {} F{}S{}S{}!!! mis_match_count {}",cell_idx,ok_tb_config_params->frameId,ok_tb_config_params->subframeId,ok_tb_config_params->slotId,mis_match_counter[cell_idx]);
                    }
                    else
                    {
                        NVLOGC_FMT(TAG_ORDER_TB_BASE, "[SRS]IQ samples match for Cell Idx {} F{}S{}S{}!!!",cell_idx,ok_tb_config_params->frameId,ok_tb_config_params->subframeId,ok_tb_config_params->slotId);
                    }
                }
            }
            srs_mm_counter=std::accumulate(mis_match_counter.begin(),mis_match_counter.end(),0);            
        }

        slot_info_curr.frameId = ok_tb_config_params->frameId;
        slot_info_curr.subframeId = ok_tb_config_params->subframeId;
        slot_info_curr.slotId = ok_tb_config_params->slotId;
        slot_info_curr.pusch_pucch_mismatch_count = pusch_pucch_mm_counter;
        slot_info_curr.prach_mismatch_count = prach_mm_counter;
        slot_info_curr.srs_mismatch_count = srs_mm_counter;
        save_process_kernel_run_info(slot_count,slot_info_curr);

        if(same_test_slot==0){
            start_slot_idx=(start_slot_idx+1)%ok_tb_num_valid_slots;
        }
#if 0
        uint8_t* tmp = ok_tb_input_params.pusch_buffer_h[0];
        __half*  tmp_half;
        float tmp_float;
        for(int i=0;i<48;i+=2)
        {
            tmp_half=reinterpret_cast<__half*>(tmp);
            tmp_float=__half2float(*tmp_half);
            printf("[OK]%f ",tmp_float);
            tmp+=2;
        }
        printf("\n");
#endif
    }
    NVLOGI_FMT(TAG_ORDER_TB_RUN, "Writing output files for launch_process_kernel_for_test_bench");
    write_output_file();
    return 0;
}

int OrderKernelTestBench::compare_iq(int test_slot_idx,int frameId,int subframeId,int slotId,std::array<uint32_t,UL_MAX_CELLS_PER_SLOT>& mis_match_counter,bool is_prach)
{
    int mis_match=0;
    int slot_num = 20*frameId+2*subframeId+slotId;
    int num_ante,prbs_per_symbol,prb_sz=PRB_SIZE_16F;
    uint8_t* temp_ptr_tv;
    uint8_t* temp_ptr_ok;
    int prbs_to_compare_pusch,prbs_to_compare_prach,prbs_to_compare_srs;
    std::array<std::string,2> IQ_type_arr ={"I","Q"};

    if(enable_mimo||enable_srs)
    {
        slot_num=slot_num%NUM_SLOTS_LP_40;
    }
    else {
        slot_num=slot_num%NUM_SLOTS_LP_80;
    }

    if(enable_srs)
    {
            uint32_t num_srs_syms=2;
            uint32_t start_srs_sym=12;
            for(int cell_idx=0;cell_idx<num_test_cells;cell_idx++)
            {
                uint8_t* ok_buf = ok_tb_input_params.srs_buffer_h[cell_idx];
                std::unordered_map<int,int>& temp_map =  ok_tb_tv_params.slot_num_to_slot_idx_map[cell_idx];
                int slot_idx=temp_map[slot_num];
                Slot& tv_slot = ok_tb_tv_params.srs_object.slots[BFP_NO_COMPRESSION].at(slot_idx);
                num_ante = ok_tb_config_file_params[cell_idx].srs_eAxC_num[test_slot_idx];
                //prb_stride_pusch=ok_tb_config_params->pusch_prb_x_port_x_symbol[cell_idx];   
                prbs_per_symbol=ok_tb_config_params->srs_prb_stride[cell_idx];                     
                for (size_t antenna_idx = 0; antenna_idx < num_ante; ++antenna_idx)
                {
                    for (size_t symbol_idx = start_srs_sym; symbol_idx < (start_srs_sym+num_srs_syms); ++symbol_idx)
                    {
                        prbs_to_compare_srs = ok_tb_config_params->srs_prb_x_slot[cell_idx]/(num_ante*num_srs_syms);
                        //NVLOGC_FMT(TAG_ORDER_TB_BASE,"Comparing {} PRBs for cell index {} antenna_index {} symbol index {} slot index {}",prbs_to_compare_srs,cell_idx,antenna_idx,symbol_idx,test_slot_idx);                    
                        for (size_t prb_idx = 0; prb_idx < prbs_to_compare_srs; ++prb_idx)
                        {
                            temp_ptr_tv = static_cast<uint8_t*>(tv_slot.ptrs.at(antenna_idx).at(symbol_idx).at(prb_idx));
                            temp_ptr_ok = ok_buf + (antenna_idx*ORAN_MAX_SRS_SYMBOLS*prbs_per_symbol*prb_sz) + ((symbol_idx-start_srs_sym)*prbs_per_symbol*prb_sz) + (prb_idx*prb_sz);
                            for(int i=0;i<prb_sz;i+=2,temp_ptr_tv+=2,temp_ptr_ok+=2)
                            {
                                __half* temp_ptr_half_tv = reinterpret_cast<__half*>(temp_ptr_tv);
                                __half* temp_ptr_half_ok = reinterpret_cast<__half*>(temp_ptr_ok);
                                float temp_tv,temp_ok;
                                if(!compare_approx(*temp_ptr_half_tv,*temp_ptr_half_ok,tolf_iq_comp))
                                {
                                    mis_match_counter[cell_idx]++;
                                    mis_match=1;
                                    temp_tv=__half2float(*temp_ptr_half_tv);
                                    temp_ok=__half2float(*temp_ptr_half_ok);
                                    NVLOGC_FMT(TAG_ORDER_TB_BASE,"[SRS]IQ sample mismatch Cell idx {} F{}S{}S{} antenna_idx {} symbol_idx {} prb_idx {} IQ Type {} RE index {} TV {} OK {}",cell_idx,frameId,subframeId,slotId,antenna_idx,symbol_idx,prb_idx,(((i/2)%2)==0)?IQ_type_arr[0]:IQ_type_arr[1],(i/4),temp_tv,temp_ok);
                                }                            
                            }
                        }
                    }
                }
            }        
    }
    else
    {
        if(is_prach)
        {
            for(int cell_idx=0;cell_idx<num_test_cells;cell_idx++)
            {
                uint8_t* ok_buf = ok_tb_input_params.prach_buffer_0_h[cell_idx];
                std::unordered_map<int,int>& temp_map =  ok_tb_tv_params.prach_slot_num_to_slot_idx_map[cell_idx];
                int slot_idx=temp_map[slot_num];
                Slot& tv_slot = ok_tb_tv_params.prach_object.slots[BFP_NO_COMPRESSION].at(slot_idx);
                if(enable_mimo)
                {
                    num_ante = num_ant_ports_prach;
                }
                else
                {
                    num_ante = ok_tb_config_file_params[cell_idx].prach_eAxC_num[test_slot_idx];
                }
                prbs_per_symbol = ok_tb_config_params->prach_prb_x_port_x_symbol[cell_idx];
                for(int buf_idx=0;buf_idx<(PRACH_MAX_NUM_SEC-1);buf_idx++)
                {
                    if(buf_idx==1)
                        ok_buf = ok_tb_input_params.prach_buffer_1_h[cell_idx];
                    else if(buf_idx==2)
                        ok_buf = ok_tb_input_params.prach_buffer_2_h[cell_idx];
                    else if(buf_idx==3)
                        ok_buf = ok_tb_input_params.prach_buffer_3_h[cell_idx];
                    for (size_t antenna_idx = 0; antenna_idx < num_ante; ++antenna_idx)
                    {
                        for (size_t symbol_idx = 0; symbol_idx < 12; ++symbol_idx)
                        {
                            for (size_t prb_idx = 0; prb_idx < prbs_per_symbol; ++prb_idx)
                            {
                                temp_ptr_tv = static_cast<uint8_t*>(tv_slot.ptrs_prach.at(buf_idx).at(antenna_idx).at(symbol_idx).at(prb_idx));
                                temp_ptr_ok = ok_buf + (antenna_idx*12*prbs_per_symbol*prb_sz) + (symbol_idx*prbs_per_symbol*prb_sz) + (prb_idx*prb_sz);
                                for(int i=0;i<prb_sz;i+=2,temp_ptr_tv+=2,temp_ptr_ok+=2)
                                {
                                    __half* temp_ptr_half_tv = reinterpret_cast<__half*>(temp_ptr_tv);
                                    __half* temp_ptr_half_ok = reinterpret_cast<__half*>(temp_ptr_ok);
                                    float temp_tv,temp_ok;
                                    if(!compare_approx(*temp_ptr_half_tv,*temp_ptr_half_ok,tolf_iq_comp))
                                    {
                                        mis_match_counter[cell_idx]++;
                                        mis_match=1;
                                        temp_tv=__half2float(*temp_ptr_half_tv);
                                        temp_ok=__half2float(*temp_ptr_half_ok);
                                        NVLOGC_FMT(TAG_ORDER_TB_BASE,"[PRACH]IQ sample mismatch Cell idx {} F{}S{}S{} antenna_idx {} symbol_idx {} prb_idx {} buf_idx {} IQ Type {} RE index {} TV {} OK {}",cell_idx,frameId,subframeId,slotId,antenna_idx,symbol_idx,prb_idx,buf_idx,(((i/2)%2)==0)?IQ_type_arr[0]:IQ_type_arr[1],(i/4),temp_tv,temp_ok);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        else
        {
            for(int cell_idx=0;cell_idx<num_test_cells;cell_idx++)
            {
                uint8_t* ok_buf = ok_tb_input_params.pusch_buffer_h[cell_idx];
                std::unordered_map<int,int>& temp_map =  ok_tb_tv_params.slot_num_to_slot_idx_map[cell_idx];
                int slot_idx=temp_map[slot_num];
                Slot& tv_slot = ok_tb_tv_params.pusch_object.slots[BFP_NO_COMPRESSION].at(slot_idx);
                if(enable_mimo)
                {
                    num_ante = num_ant_ports;
                }
                else
                {
                    num_ante = ok_tb_config_file_params[cell_idx].pusch_eAxC_num[test_slot_idx];
                }
                //prb_stride_pusch=ok_tb_config_params->pusch_prb_x_port_x_symbol[cell_idx];   
                prbs_per_symbol=ok_tb_config_params->pusch_prb_x_port_x_symbol[cell_idx];                     
                for (size_t antenna_idx = 0; antenna_idx < num_ante; ++antenna_idx)
                {
                    for (size_t symbol_idx = 0; symbol_idx < SLOT_NUM_SYMS; ++symbol_idx)
                    {
                        prbs_to_compare_pusch = ok_tb_config_file_params[cell_idx].pusch_prb_symbol_map[test_slot_idx][symbol_idx]/num_ante;
                        //NVLOGC_FMT(TAG_ORDER_TB_BASE,"Comparing {} PRBs for cell index {} antenna_index {} symbol index {} slot index {}",prbs_to_compare_pusch,cell_idx,antenna_idx,symbol_idx,test_slot_idx);                    
                        for (size_t prb_idx = 0; prb_idx < prbs_to_compare_pusch; ++prb_idx)
                        {
                            temp_ptr_tv = static_cast<uint8_t*>(tv_slot.ptrs.at(antenna_idx).at(symbol_idx).at(prb_idx));
                            temp_ptr_ok = ok_buf + (antenna_idx*SLOT_NUM_SYMS*prbs_per_symbol*prb_sz) + (symbol_idx*prbs_per_symbol*prb_sz) + (prb_idx*prb_sz);
                            for(int i=0;i<prb_sz;i+=2,temp_ptr_tv+=2,temp_ptr_ok+=2)
                            {
                                __half* temp_ptr_half_tv = reinterpret_cast<__half*>(temp_ptr_tv);
                                __half* temp_ptr_half_ok = reinterpret_cast<__half*>(temp_ptr_ok);
                                float temp_tv,temp_ok;
                                if(!compare_approx(*temp_ptr_half_tv,*temp_ptr_half_ok,tolf_iq_comp))
                                {
                                    mis_match_counter[cell_idx]++;
                                    mis_match=1;
                                    temp_tv=__half2float(*temp_ptr_half_tv);
                                    temp_ok=__half2float(*temp_ptr_half_ok);
                                    NVLOGC_FMT(TAG_ORDER_TB_BASE,"[PUSCH/PUCCH]IQ sample mismatch Cell idx {} F{}S{}S{} antenna_idx {} symbol_idx {} prb_idx {} IQ Type {} RE index {} TV {} OK {}",cell_idx,frameId,subframeId,slotId,antenna_idx,symbol_idx,prb_idx,(((i/2)%2)==0)?IQ_type_arr[0]:IQ_type_arr[1],(i/4),temp_tv,temp_ok);
                                }                            
                            }
                        }
                    }
                }
            }
        }        
    }
    return mis_match;
}

bool OrderKernelTestBench::compare_approx(__half& a,__half& b,float tolf)
{
    const __half tolerance = __float2half(tolf);
    __half diff = __habs(a - b);
    return (diff <= tolerance);
}

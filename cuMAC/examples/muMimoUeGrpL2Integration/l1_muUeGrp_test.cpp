/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "l1_muUeGrp_test.h"
#include <csignal>

//#define SRS_MEMORY_BANK_TEST

// global variables
int NUM_TIME_SLOTS;
int NUM_CELL; // number of cells
int L1_MAIN_THREAD_CORE; // L1 main thread core
int L1_L2_RECV_THREAD_CORE; // L1 L2 receiver thread core

volatile sig_atomic_t g_shutdown = 0;

static void signal_handler(int signum)
{
    g_shutdown = 1;
}

nv_ipc_t* ipc_l1_l2 = NULL; // NVIPC interface to the L2 stack

sem_t l2_recv_msg_sem;

uint8_t* recv_msg_buff = nullptr;

// *****************************************************
// L1 receiver thread to receive the SRS messages from L2
void* l1_l2_blocking_recv_task(void* arg)
{
    // Set thread name, max string length < 16
    pthread_setname_np(pthread_self(), "l1_l2_recv");
    // Set thread schedule policy to SCHED_FIFO and set priority to 80
    nv_set_sched_fifo_priority(80);
    // Set the thread CPU core
    nv_assign_thread_cpu_core(L1_L2_RECV_THREAD_CORE);

    int num_slot = 0;

    while (num_slot < NUM_TIME_SLOTS && !g_shutdown) {
        NVLOGI(MU_TEST_TAG, "%s: wait for incoming messages notification ...", __func__);

        // Wait for incoming messages notification from L2
        ipc_l1_l2->rx_tti_sem_wait(ipc_l1_l2);
        if (g_shutdown) break;
        num_slot++;

        nv_ipc_msg_t recv_msg;

        while (ipc_l1_l2->rx_recv_msg(ipc_l1_l2, &recv_msg) >= 0) {
            std::memcpy(recv_msg_buff, recv_msg.data_buf, recv_msg.data_len);

            NVLOGC(MU_TEST_TAG, "L1-L2 RECV: NVIPC message data length = %d", recv_msg.data_len);
            
            // Release the NVIPC message buffer
            ipc_l1_l2->rx_release(ipc_l1_l2, &recv_msg);
        }

        sem_post(&l2_recv_msg_sem);

        NVLOGC(MU_TEST_TAG, "L1-L2 RECV: time slot %d, received SRS message from L2", num_slot-1);
    }

    NVLOGC(MU_TEST_TAG, "L1-L2 RECV: test completed successfully");
    return NULL;    
}

void srs_chan_est(SimpleCvSrsChestMemoryBank* memory_bank, l1_cumac_message_t* arr_l1_cumac_msg, const TestConfig& config, const sys_param_t& sys_param)
{
    l2_l1_message_t* l2_l1_msg = (l2_l1_message_t*) recv_msg_buff;

    uint16_t nSrsUes = l2_l1_msg->nSrsUes;

    NVLOGC(MU_TEST_TAG, "L1-L2 RECV: nSrsUes = %d", nSrsUes);

    uint16_t* arr_cell_idx = (uint16_t*) (l2_l1_msg->arr_usage + nSrsUes);
    uint16_t* arr_rnti = (uint16_t*) (arr_cell_idx + nSrsUes);
    uint16_t* arr_buffer_Idx = (uint16_t*) (arr_rnti + nSrsUes);
    uint16_t* arr_srs_info_idx = (uint16_t*) (arr_buffer_Idx + nSrsUes);

    size_t buffer_size = static_cast<size_t>(config.num_prg * config.num_gnb_ant * config.num_ue_layer);

    std::vector<__half2> srs_ch_est_buff_cpu;
    srs_ch_est_buff_cpu.resize(buffer_size);

    // initialize random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    // Normal (Gaussian) distribution with mean 0 and stddev 1
    std::normal_distribution<> normal_distrib(0.0, 1.0);

    for (int srs_ue_idx = 0; srs_ue_idx < nSrsUes; srs_ue_idx++) {
        const uint16_t cell_idx = arr_cell_idx[srs_ue_idx];
        const uint32_t usage = l2_l1_msg->arr_usage[srs_ue_idx];
        const uint16_t rnti = arr_rnti[srs_ue_idx];
        const uint16_t buffer_Idx = arr_buffer_Idx[srs_ue_idx];
        const uint16_t srs_info_idx = arr_srs_info_idx[srs_ue_idx];
        CVSrsChestBuff_contMemAlloc* ue_buffer = nullptr;
        uint32_t real_buff_idx;
        memory_bank->preAllocateBuffer(cell_idx, rnti, buffer_Idx, usage, &ue_buffer, &real_buff_idx);
        arr_l1_cumac_msg[srs_ue_idx].real_buff_idx = real_buff_idx;
        arr_l1_cumac_msg[srs_ue_idx].srs_info_idx = srs_info_idx;
        arr_l1_cumac_msg[srs_ue_idx].cell_idx = cell_idx;

        for (size_t i = 0; i < buffer_size; i++) {
            srs_ch_est_buff_cpu[i] = __half2(normal_distrib(gen)*sqrt(0.5*sys_param.srs_chan_est_coeff_var), 
                                             normal_distrib(gen)*sqrt(0.5*sys_param.srs_chan_est_coeff_var));
        }

        ue_buffer->configSrsInfo(config.num_prg, config.num_gnb_ant, config.num_ue_layer, 
                                 config.srs_prg_size, config.srs_start_prg, config.srs_start_valid_prg, 
                                 config.srs_n_valid_prg);

        cudaMemcpy(ue_buffer->getAddr(), srs_ch_est_buff_cpu.data(), buffer_size*sizeof(__half2), cudaMemcpyHostToDevice);

        memory_bank->updateSrsChestBufferState(cell_idx, buffer_Idx, slot_command_api::SRS_CHEST_BUFF_READY);
    }
}

int main(int argc, char** argv)
{
    try {
        struct sigaction sa = {};
        sa.sa_handler = signal_handler;
        sigemptyset(&sa.sa_mask);
        sigaction(SIGINT, &sa, nullptr);
        sigaction(SIGTERM, &sa, nullptr);

        NVLOGC(MU_TEST_TAG, "L1-MAIN: parameter config YAML file: %s", YAML_PARAM_CONFIG_PATH);

        // system configuration parameters
        sys_param_t sys_param(YAML_PARAM_CONFIG_PATH);
        L1_MAIN_THREAD_CORE = sys_param.l1_main_thread_core;
        L1_L2_RECV_THREAD_CORE = sys_param.l1_l2_recv_thread_core;

        NUM_TIME_SLOTS = sys_param.num_time_slots;
        NUM_CELL = sys_param.num_cell;

        // Allocate memory for the received message buffer
        recv_msg_buff = (uint8_t*) malloc(sizeof(l2_l1_message_t) + (sizeof(uint32_t) + 4 * sizeof(uint16_t)) * MAX_NUM_SRS_UE_PER_CELL * NUM_CELL);
        if (recv_msg_buff == nullptr) {
            NVLOGE(MU_TEST_TAG, AERIAL_MEMORY_API_EVENT, "L1-MAIN: %s: failed to allocate memory for the L1 received message buffer", __func__);
            return -1;
        }

        // SRS Shared Memory Bank
        NVLOGC(MU_TEST_TAG, "L1-MAIN: SRS Memory Bank: Loading configuration from %s...", YAML_PARAM_CONFIG_PATH);
        TestConfig config;
        if (!config.loadFromYaml(YAML_PARAM_CONFIG_PATH)) {
            NVLOGE(MU_TEST_TAG, AERIAL_NVIPC_API_EVENT, "L1-MAIN: %s: Warning: Could not load config from %s, using default values", __func__, YAML_PARAM_CONFIG_PATH);
            return -1;
        }

        uint32_t total_num_buffers = std::min(config.num_srs_buffers, static_cast<uint32_t>(slot_command_api::MAX_SRS_CHEST_BUFFERS));
        uint32_t buffer_size = config.num_prg * config.num_gnb_ant * config.num_ue_layer * sizeof(uint32_t);

        // Initialize semaphore for L1 SRS update finish notification
        nv_ipc_sem_t* l1_cumac_sem = nv_ipc_sem_open(L1_CUMAC_PRIMARY_PROCESS, L1_SEM_NAME);

        // Create shared memory pool in GPU
        nv_ipc_mempool_t* cpu_mem_pool = nv_ipc_mempool_open(L1_CUMAC_PRIMARY_PROCESS, L1_CPU_MEM_POOL_NAME, (sizeof(CVSrsChestBuff_contMemAlloc)*total_num_buffers + sizeof(l1_cumac_message_t)*MAX_NUM_UE_SRS_INFO_PER_SLOT*NUM_CELL), 1, NV_IPC_MEMPOOL_NO_CUDA_DEV); // CPU memory pool to be shared between L1 and cuMAC
        nv_ipc_mempool_t* gpu_mem_pool = nv_ipc_mempool_open(L1_CUMAC_PRIMARY_PROCESS, L1_GPU_MEM_POOL_NAME, buffer_size, total_num_buffers, config.cuda_device_id); // GPU memory pool to be shared between L1 and cuMAC
        
        // Get CPU/GPU shared memory pool start address
        void* cpu_pool_start_addr = cpu_mem_pool->get_addr(cpu_mem_pool, 0); 
        void* gpu_pool_start_addr = gpu_mem_pool->get_addr(gpu_mem_pool, 0);

        if (!is_aligned_for_type<CVSrsChestBuff_contMemAlloc>(cpu_pool_start_addr)) {
            NVLOGE(MU_TEST_TAG, AERIAL_NVIPC_API_EVENT, "L1-MAIN: %s: CPU memory base address is not aligned for CVSrsChestBuff_contMemAlloc", __func__);
            return -1;
        }
        CVSrsChestBuff_contMemAlloc* arr_cv_srs_chest_buff_base_addr = reinterpret_cast<CVSrsChestBuff_contMemAlloc*>(cpu_pool_start_addr);

        if (!is_aligned_for_type<__half2>(gpu_pool_start_addr)) {
            NVLOGE(MU_TEST_TAG, AERIAL_NVIPC_API_EVENT, "L1-MAIN: %s: GPU memory base address is not aligned for __half2", __func__);
            return -1;
        }

        // ------------------------------------------------------------
        // Initialize CUDA device
        cudaError_t err = cudaSetDevice(config.cuda_device_id);
        if (err != cudaSuccess) {
            NVLOGE(MU_TEST_TAG, AERIAL_NVIPC_API_EVENT, "L1-MAIN: %s: Failed to set CUDA device: %s", __func__, cudaGetErrorString(err));
            return -1;
        }
        
        // Get and print device properties
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, config.cuda_device_id);

        // Create simplified SRS memory bank (uses CVSrsChestBuff_contMemAlloc class)
        // This version directly uses CUDA APIs without requiring GpuDevice/PhyDriverCtx
        NVLOGC(MU_TEST_TAG, "L1-MAIN: SRS Memory Bank Test: Creating SimpleCvSrsChestMemoryBank, contiguous GPU memory: %s", (config.is_contiguous_gpu_mem ? "true" : "false"));
        SimpleCvSrsChestMemoryBank* memory_bank = new SimpleCvSrsChestMemoryBank(config, cpu_pool_start_addr, gpu_pool_start_addr);
        
        // Check if buffers are contiguous in GPU memory
        NVLOGC(MU_TEST_TAG, "L1-MAIN: SRS Memory Bank Test: Checking buffer contiguity...");
        const bool contiguous = memory_bank->areBuffersContiguous();

        l1_cumac_message_t* arr_l1_cumac_msg = reinterpret_cast<l1_cumac_message_t*>(arr_cv_srs_chest_buff_base_addr + total_num_buffers);
        
#ifdef SRS_MEMORY_BANK_TEST
        ret = test_srs_memory_bank(memory_bank, config);
        if (ret != 0) {
            NVLOGE(MU_TEST_TAG, AERIAL_NVIPC_API_EVENT, "L1-MAIN: %s: SRS Memory Bank Test: Test FAILED", __func__);
            return 1;
        }
#endif

        // Initialize SRS memory bank
        for (uint32_t cell_idx = 0; cell_idx < NUM_CELL; cell_idx++) {
            memory_bank->memPoolAllocatePerCell(config.alloc_request, cell_idx, config.num_srs_buffers_per_cell);
        }

        // Set up L1 L2 NVIPC interface
        // Load nvipc configuration from YAML file
        nv_ipc_config_t l1_l2_config;
        load_nv_ipc_yaml_config(&l1_l2_config, YAML_L1_L2_NVIPC_CONFIG_PATH, NV_IPC_MODULE_SECONDARY);

        // Initialize NVIPC interface and connect to L2
        if ((ipc_l1_l2 = create_nv_ipc_interface(&l1_l2_config)) == NULL) {
            NVLOGE(MU_TEST_TAG, AERIAL_NVIPC_API_EVENT, "L1-MAIN: %s: create L1/L2 secondary IPC interface failed", __func__);
            return -1;
        }

        // Sleep 1 seconds for NVIPC connection
        usleep(1000000);

        // Configure L1 main thread
        pthread_setname_np(pthread_self(), "l1_main");
        nv_set_sched_fifo_priority(80);
        nv_assign_thread_cpu_core(L1_MAIN_THREAD_CORE);

        // Initialize semaphore for L2 SRS message reception notification
        sem_init(&l2_recv_msg_sem, 0, 0);

        // Create L1/L2 receiver thread
        pthread_t thread_id;
        int ret = pthread_create(&thread_id, NULL, l1_l2_blocking_recv_task, NULL);
        if(ret != 0) {
            NVLOGE(MU_TEST_TAG, AERIAL_NVIPC_API_EVENT, "%s failed, ret=%d", __func__, ret);
            return -1;
        }

        // Initial SFN/SLOT = 0.0
        uint16_t sfn = 0, slot = 0;

        int slot_count = 0;      
        while(slot_count < NUM_TIME_SLOTS && !g_shutdown) {
            // Wait for L2 SRS message reception notification by semaphore l2_recv_msg_sem
            sem_wait(&l2_recv_msg_sem);
            if (g_shutdown) break;

            srs_chan_est(memory_bank, arr_l1_cumac_msg, config, sys_param);

            // Post L1 semaphore signal to notify cuMAC that the SRS channel estimation is done
            NVLOGC(MU_TEST_TAG, "L1-MAIN: posting cuMAC semaphore signal...");
            l1_cumac_sem->sem_post(l1_cumac_sem);

            // Wait for cuMAC UE pairing completion notification by semaphore 
            NVLOGC(MU_TEST_TAG, "L1-MAIN: waiting for cuMAC semaphore signal...");
            l1_cumac_sem->sem_wait(l1_cumac_sem);
            
            // Update SFN/SLOT for next slot
            advance_sfn_slot(sfn, slot);

            slot_count++;
        }

        if (g_shutdown) {
            NVLOGC(MU_TEST_TAG, "L1-MAIN: received shutdown signal, cleaning up...");
        }

        // clean up memory bank before closing NVIPC shared memory pools
        delete memory_bank;

        cpu_mem_pool->close(cpu_mem_pool);
        gpu_mem_pool->close(gpu_mem_pool);
        l1_cumac_sem->close(l1_cumac_sem);

        free(recv_msg_buff);

        ipc_l1_l2->ipc_destroy(ipc_l1_l2);

    } catch (const std::exception& e) {
        NVLOGE(MU_TEST_TAG, AERIAL_NVIPC_API_EVENT, "L1-MAIN: %s: test failed with exception: %s", __func__, e.what());
        return 1;
    }

    NVLOGC(MU_TEST_TAG, "L1-MAIN: test completed successfully");
    
    return 0;
}
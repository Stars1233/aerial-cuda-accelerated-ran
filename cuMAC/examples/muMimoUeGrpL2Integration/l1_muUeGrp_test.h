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

#pragma once

// Simplified header for l1_muUeGrp_test
// Only includes what's needed for the simplified SRS memory bank test
#include "simple_srs_memory_bank.hpp"
#include "l2_muUeGrp_test.h"
#include "common_utils.h"
#include "cumac.h"
#include <iostream>
#include <memory>
#include <vector>
#include <thread>
#include <chrono>
#include <cuda_runtime.h>
#include "nv_utils.h"
#include "nv_ipc.h"
#include "nv_ipc_utils.h"
#include "nv_ipc_sem.h"
#include "nv_ipc_cudapool.h" 
#include "nv_ipc_mempool.h" 

constexpr const char* YAML_L1_CUMAC_NVIPC_CONFIG_PATH = "./cuMAC/examples/muMimoUeGrpL2Integration/yamlConfigFiles/l1_cumac_nvipc.yaml"; // path to the L1 NVIPC configuration parameters YAML file
constexpr const char* YAML_L1_L2_NVIPC_CONFIG_PATH = "./cuMAC/examples/muMimoUeGrpL2Integration/yamlConfigFiles/l1_l2_nvipc.yaml"; // path to the L1/L2 NVIPC configuration parameters YAML file

#define L1_CPU_MEM_POOL_NAME "l1_cpu_sh_pool"
#define L1_GPU_MEM_POOL_NAME "l1_gpu_sh_pool"
#define L1_SEM_NAME "l1_sem"
#define L1_CUMAC_PRIMARY_PROCESS 1

// NVIPC interface
nv_ipc_t* l1_cumac_ipc_interface = NULL;

struct CudaIpcHandles {
    cudaIpcEventHandle_t event_handle;
    uint32_t buffer_size;
    cudaIpcMemHandle_t mem_handle;
};

int test_srs_memory_bank(SimpleCvSrsChestMemoryBank* memory_bank, const TestConfig& config)
{
    try {
        // Test accessing the arr_cv_srs_chest_buff array
        std::cout << "\nSRS Memory Bank Test: Testing array access..." << std::endl;
        std::cout << "Allocated buffers: " << memory_bank->getNumBuffers() << std::endl;
        
        // Allocate memory pools for multiple test cells
        std::cout << "\nSRS Memory Bank Test: Allocating memory pools for " << config.num_cell << " cells..." << std::endl;
        const uint32_t buffers_per_cell = config.num_srs_buffers / config.num_cell;
        const uint32_t request_type = config.alloc_request;
        
        std::cout << "Allocating " << buffers_per_cell << " buffers per cell" << std::endl;
        
        for (uint32_t cell_idx = 0; cell_idx < config.num_cell; cell_idx++) {
            const uint32_t cell_id = config.cell_id_base + cell_idx;
            const bool alloc_success = memory_bank->memPoolAllocatePerCell(request_type, cell_id, buffers_per_cell);
            if (!alloc_success) {
                throw std::runtime_error("Failed to allocate memory pool for cell " + std::to_string(cell_id));
            }
            std::cout << "  Cell " << cell_id << ": Successfully allocated " << buffers_per_cell << " buffers" << std::endl;
        }
        std::cout << "All cell memory pools allocated successfully\n" << std::endl;
        
        // Test pre-allocating buffers for UEs across multiple cells
        std::cout << "\nSRS Memory Bank Test: Testing buffer pre-allocation for " << config.num_srs_ues_per_cell 
                  << " UE(s) per cell across " << config.num_cell << " cells..." << std::endl;
        const uint32_t buffer_size = config.num_prg * config.num_gnb_ant * config.num_ue_layer * sizeof(uint32_t);
        
        uint32_t total_ues_allocated = 0;
        for (uint32_t cell_idx = 0; cell_idx < config.num_cell; cell_idx++) {
            const uint32_t cell_id = config.cell_id_base + cell_idx;
            std::cout << "\nCell " << cell_id << ":" << std::endl;
            
            for (uint32_t ue_idx = 0; ue_idx < config.num_srs_ues_per_cell; ue_idx++) {
                const uint32_t rnti = config.rnti_base + (cell_idx * config.num_srs_ues_per_cell) + ue_idx;
                const uint16_t buffer_idx = ue_idx;
                const uint32_t usage = 1;  // Initial usage count
                CVSrsChestBuff_contMemAlloc* ue_buffer = nullptr;
                
                const int ret = memory_bank->preAllocateBuffer(cell_id, rnti, buffer_idx, usage, &ue_buffer);
                if (ret == 0 && ue_buffer) {
                    std::cout << "  UE " << ue_idx << " (RNTI=" << rnti << "): Allocated buffer[" << buffer_idx << "]" << std::endl;
                    std::cout << "    GPU addr: " << static_cast<void*>(ue_buffer->getAddr()) << std::endl;
                    std::cout << "    Size: " << buffer_size << " bytes" << std::endl;
                    std::cout << "    Cell ID: " << ue_buffer->getCellId() << std::endl;
                    std::cout << "    Buffer Index: " << ue_buffer->getBufferIdx() << std::endl;
                    
                    // Set SFN/Slot unique per UE
                    ue_buffer->setSfnSlot(100 + total_ues_allocated, 5 + ue_idx);
                    std::cout << "    SFN: " << ue_buffer->getSfn() << ", Slot: " << ue_buffer->getSlot() << std::endl;
                    total_ues_allocated++;
                } else {
                    std::cerr << "  Failed to pre-allocate buffer for UE " << ue_idx << " (RNTI=" << rnti << ")" << std::endl;
                }
            }
        }
        std::cout << "\nTotal UEs allocated: " << total_ues_allocated << std::endl;
        
        // Test buffer state management across multiple cells
        std::cout << "\nSRS Memory Bank Test: Testing buffer state management across cells..." << std::endl;
        for (uint32_t cell_idx = 0; cell_idx < config.num_cell; cell_idx++) {
            const uint32_t cell_id = config.cell_id_base + cell_idx;
            const uint16_t test_buffer_idx = 0;
            
            memory_bank->updateSrsChestBufferState(cell_id, test_buffer_idx, slot_command_api::SRS_CHEST_BUFF_REQUESTED);
            slot_command_api::srsChestBuffState state_requested = memory_bank->getSrsChestBufferState(cell_id, test_buffer_idx);
            
            memory_bank->updateSrsChestBufferState(cell_id, test_buffer_idx, slot_command_api::SRS_CHEST_BUFF_READY);
            slot_command_api::srsChestBuffState state_ready = memory_bank->getSrsChestBufferState(cell_id, test_buffer_idx);
            
            std::cout << "Cell " << cell_id << ": buffer[" << test_buffer_idx << "] state transitions: "
                      << "REQUESTED(" << static_cast<int>(state_requested) << ") -> "
                      << "READY(" << static_cast<int>(state_ready) << ")" << std::endl;
        }
        
        // Test buffer retrieval across multiple cells
        std::cout << "\nSRS Memory Bank Test: Testing buffer retrieval across cells..." << std::endl;
        for (uint32_t cell_idx = 0; cell_idx < config.num_cell; cell_idx++) {
            const uint32_t cell_id = config.cell_id_base + cell_idx;
            const uint32_t test_rnti = config.rnti_base + (cell_idx * config.num_srs_ues_per_cell);
            const uint16_t buffer_idx = 0;
            
            CVSrsChestBuff_contMemAlloc* retrieved_buffer = nullptr;
            const int retrieve_ret = memory_bank->retrieveBuffer(cell_id, test_rnti, buffer_idx, &retrieved_buffer);
            if (retrieve_ret == 0 && retrieved_buffer) {
                std::cout << "Cell " << cell_id << ", RNTI " << test_rnti << ": Retrieved buffer successfully" << std::endl;
                std::cout << "  Verified Cell ID: " << retrieved_buffer->getCellId() 
                          << ", Buffer Index: " << retrieved_buffer->getBufferIdx()
                          << ", RNTI: " << retrieved_buffer->getRnti() << std::endl;
            } else {
                std::cerr << "Cell " << cell_id << ": Failed to retrieve buffer (ret=" << retrieve_ret << ")" << std::endl;
            }
        }
        
        // Test usage counter across cells
        std::cout << "\nSRS Memory Bank Test: Testing usage counter across cells..." << std::endl;
        for (uint32_t cell_idx = 0; cell_idx < config.num_cell; cell_idx++) {
            const uint32_t cell_id = config.cell_id_base + cell_idx;
            const uint32_t test_rnti = config.rnti_base + (cell_idx * config.num_srs_ues_per_cell);
            const uint16_t buffer_idx = 0;
            const uint32_t new_usage = 5 + cell_idx;
            
            memory_bank->updateSrsChestBufferUsage(cell_id, test_rnti, buffer_idx, new_usage);
            const uint32_t usage = memory_bank->getSrsChestBufferUsage(cell_id, test_rnti, buffer_idx);
            std::cout << "Cell " << cell_id << ", buffer[" << buffer_idx << "]: usage count = " << usage << std::endl;
        }
        
        // Summary
        std::cout << "\n=== Test Summary ===" << std::endl;
        std::cout << "✓ Using CVSrsChestBuff_contMemAlloc class from simple_srs_memory_bank.hpp" << std::endl;
        std::cout << "✓ Configuration loaded from: " << YAML_PARAM_CONFIG_PATH << std::endl;
        std::cout << "✓ CUDA device: " << config.cuda_device_id << std::endl;
        std::cout << "✓ Successfully allocated " << memory_bank->getNumBuffers() << " GPU buffers" << std::endl;
        std::cout << "✓ Tested " << config.num_cell << " cells with " << buffers_per_cell << " buffers per cell" << std::endl;
        std::cout << "✓ Pre-allocated and configured " << total_ues_allocated << " UE buffers across all cells" << std::endl;        
        std::cout << "\n=== Test PASSED ===" << std::endl;

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "\nSRS Memory Bank Test: Test FAILED ===" << std::endl;
        std::cerr << "Exception: " << e.what() << std::endl;
        return 1;
    }
}

int l1_test_cuda_ipc_api(SimpleCvSrsChestMemoryBank* memory_bank, const TestConfig& config)
{
    // Set CUDA device
    cudaError_t err = cudaSetDevice(config.cuda_device_id);
    if (err != cudaSuccess) {
        NVLOGE(MU_TEST_TAG, AERIAL_NVIPC_API_EVENT, "L1-MAIN: %s: cudaSetDevice(%u) failed: %s", __func__, config.cuda_device_id, cudaGetErrorString(err));
        return 1;
    }

    // Load NVIPC configuration from YAML file
    nv_ipc_config_t l1_cumac_ipc_config{};
    load_nv_ipc_yaml_config(&l1_cumac_ipc_config, YAML_L1_CUMAC_NVIPC_CONFIG_PATH, NV_IPC_MODULE_PRIMARY);
        
    if ((l1_cumac_ipc_interface = create_nv_ipc_interface(&l1_cumac_ipc_config)) == NULL) {
        NVLOGE(MU_TEST_TAG, AERIAL_NVIPC_API_EVENT, "L1-MAIN: %s: create L1/cuMAC primary NVIPC interface failed", __func__);
        return 1;
    }
    
    // Allocate device buffer
    void* d_buf = nullptr;
    const uint32_t buffer_size = 1024;
    
    err = cudaMalloc(&d_buf, buffer_size);
    if (err != cudaSuccess) {
        NVLOGE(MU_TEST_TAG, AERIAL_NVIPC_API_EVENT, "L1-MAIN: %s: cudaMalloc(%zu) failed: %s", __func__,
        buffer_size, cudaGetErrorString(err));
        return 1;
    }
    
    // Get IPC memory handle
    cudaIpcMemHandle_t mem_handle{};
    err = cudaIpcGetMemHandle(&mem_handle, d_buf);
    if (err != cudaSuccess) {
        NVLOGE(MU_TEST_TAG, AERIAL_NVIPC_API_EVENT, "L1-MAIN: %s: cudaIpcGetMemHandle(%p) failed: %s", __func__, d_buf, cudaGetErrorString(err));
        return 1;
    }
    
    // Create interprocess event
    cudaEvent_t ready_event;
    err = cudaEventCreateWithFlags(&ready_event, cudaEventDisableTiming | cudaEventInterprocess);
    if (err != cudaSuccess) {
        NVLOGE(MU_TEST_TAG, AERIAL_NVIPC_API_EVENT, "L1-MAIN: %s: cudaEventCreateWithFlags() failed: %s", __func__, cudaGetErrorString(err));
        return 1;
    }
    
    // Get IPC event handle
    cudaIpcEventHandle_t event_handle{};
    err = cudaIpcGetEventHandle(&event_handle, ready_event);
    if (err != cudaSuccess) {
        NVLOGE(MU_TEST_TAG, AERIAL_NVIPC_API_EVENT, "L1-MAIN: %s: cudaIpcGetEventHandle(%p) failed: %s", __func__, ready_event, cudaGetErrorString(err));
        return 1;
    }
    
    // Record event to signal data is ready (use default stream)
    err = cudaEventRecord(ready_event, 0);
    if (err != cudaSuccess) {
        NVLOGE(MU_TEST_TAG, AERIAL_NVIPC_API_EVENT, "L1-MAIN: %s: cudaEventRecord() failed: %s", __func__, cudaGetErrorString(err));
        return 1;
    }
    
    // Prepare IPC handles message
    CudaIpcHandles ipc_handles{};
    ipc_handles.mem_handle = mem_handle;
    ipc_handles.event_handle = event_handle;
    ipc_handles.buffer_size = buffer_size;
    
    // Allocate and send IPC handles via NVIPC
    nv_ipc_msg_t send_msg{};
    send_msg.msg_id = 0x1000;  // Custom message ID for CUDA IPC handles
    send_msg.msg_len = sizeof(CudaIpcHandles);
    send_msg.data_len = 0;  // No separate data buffer needed
    send_msg.data_pool = NV_IPC_MEMPOOL_CPU_MSG;
        
    // Allocate buffer for TX message
    if (l1_cumac_ipc_interface->tx_allocate(l1_cumac_ipc_interface, &send_msg, 0) != 0) {
        NVLOGE(MU_TEST_TAG, AERIAL_NVIPC_API_EVENT, "L1-MAIN: %s: Failed to allocate TX buffer", __func__);
        cudaEventDestroy(ready_event);
        cudaFree(d_buf);
        l1_cumac_ipc_interface->ipc_destroy(l1_cumac_ipc_interface);
        return 1;
    }
        
    // Copy IPC handles to message buffer
    memcpy(send_msg.msg_buf, &ipc_handles, sizeof(CudaIpcHandles));
        
    // Send the message
    if (l1_cumac_ipc_interface->tx_send_msg(l1_cumac_ipc_interface, &send_msg) != 0) {
        NVLOGE(MU_TEST_TAG, AERIAL_NVIPC_API_EVENT, "L1-MAIN: %s: failed to send IPC handles via NVIPC", __func__);
        l1_cumac_ipc_interface->tx_release(l1_cumac_ipc_interface, &send_msg);
        cudaEventDestroy(ready_event);
        err = cudaFree(d_buf);
        if (err != cudaSuccess) {
            NVLOGE(MU_TEST_TAG, AERIAL_NVIPC_API_EVENT, "L1-MAIN: %s: cudaFree(%p) failed: %s", __func__, d_buf, cudaGetErrorString(err));
            return 1;
        }
        l1_cumac_ipc_interface->ipc_destroy(l1_cumac_ipc_interface);
        return 1;
    }
    
    // Post the NVIPC TX notification
    while (l1_cumac_ipc_interface->tx_tti_sem_post(l1_cumac_ipc_interface) != 0) {
        usleep(1000);
    }
    
    usleep(10000000);
    
    // Cleanup
    cudaEventDestroy(ready_event);
    cudaFree(d_buf);
    // l1_cumac_ipc_interface->ipc_destroy(l1_cumac_ipc_interface);
    
    NVLOGC(MU_TEST_TAG, "L1-MAIN: CUDA IPC API test passed");
    
    return 0;
}



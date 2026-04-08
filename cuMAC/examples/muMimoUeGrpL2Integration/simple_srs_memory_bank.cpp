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

#include "simple_srs_memory_bank.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <cuda_runtime.h>

SimpleCvSrsChestMemoryBank::SimpleCvSrsChestMemoryBank(const TestConfig& _config)
    : config(_config)
    , cuda_device_id(_config.cuda_device_id)
    , total_num_buffers(std::min(_config.num_srs_buffers, static_cast<uint32_t>(slot_command_api::MAX_SRS_CHEST_BUFFERS)))
    , buffer_size(_config.num_prg * _config.num_gnb_ant * _config.num_ue_layer * sizeof(uint32_t))
    , is_shared_memory(false)
{
    std::cout << "\n=== Simplified CvSrsChestMemoryBank (using CVSrsChestBuff_contMemAlloc buffer type) ===" << std::endl;
    std::cout << "Using L1 standalone/non-shared memory" << std::endl;
    std::cout << "CUDA device: " << cuda_device_id << std::endl;
    std::cout << "Allocating " << total_num_buffers << " SRS channel estimate buffers" << std::endl;
    
    // Set active GPU device directly
    cudaError_t err = cudaSetDevice(cuda_device_id);
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("Failed to set CUDA device: ") + cudaGetErrorString(err));
    }
    
    // Calculate size of each buffer (same as original CvSrsChestMemoryBank)
    // Each CV buffer is a 3-dim buffer with dimensions: (nPrbG * nGnbAnt * nUeLayers)
    // Using complex float (2 floats = 8 bytes per complex number)
    
    std::cout << "Each buffer size: " << buffer_size << " bytes" << std::endl;
    std::cout << "  Dimensions: " << config.num_prg << " PRBs x " 
              << config.num_gnb_ant << " antennas x " 
              << config.num_ue_layer << " layers" << std::endl;
    std::cout << "Total GPU memory: " << (buffer_size * total_num_buffers) << " bytes ("
              << (buffer_size * total_num_buffers / 1024.0 / 1024.0) << " MB)" << std::endl;

    std::cout << "\nAllocating individual GPU buffers (each via separate cudaMalloc):" << std::endl;
    // Allocate GPU memory for each CV buffer
    // IMPORTANT: Each buffer gets its own cudaMalloc call, so they are NOT contiguous
    for (uint32_t idx = 0; idx < total_num_buffers; idx++) {
        // Allocate GPU device buffer
        // Pass nullptr for GpuDevice* since IOBuf doesn't actually use it (only stores it)
        // The actual allocation is done by device_alloc::allocate() which just calls cudaMalloc
        dev_buf* device_buffer = new dev_buf(buffer_size, nullptr);
        
        // Create CVSrsChestBuff_contMemAlloc wrapper (contiguous memory allocation version)
        CVSrsChestBuff_contMemAlloc* cv_buffer = new CVSrsChestBuff_contMemAlloc(device_buffer);
        cv_buffer->setSrsChestBuffState(slot_command_api::SRS_CHEST_BUFF_NONE);
        
        arr_cv_srs_chest_buff[idx] = cv_buffer;
        
        // Add to free pool
        memIndexPool.push(idx);
    }

    std::cout << "\nAll " << total_num_buffers << " GPU buffers allocated successfully!" << std::endl;
    std::cout << "Free buffer pool initialized with " << memIndexPool.size() << " buffers" << std::endl;
}

SimpleCvSrsChestMemoryBank::SimpleCvSrsChestMemoryBank(const TestConfig& _config, void* cpu_buf_start_addr, void* gpu_buf_start_addr)
    : config(_config)
    , cuda_device_id(_config.cuda_device_id)
    , total_num_buffers(std::min(_config.num_srs_buffers, static_cast<uint32_t>(slot_command_api::MAX_SRS_CHEST_BUFFERS)))
    , buffer_size(_config.num_prg * _config.num_gnb_ant * _config.num_ue_layer * sizeof(uint32_t))
    , is_shared_memory(true)
{
    std::cout << "\n=== Simplified CvSrsChestMemoryBank (using CVSrsChestBuff_contMemAlloc buffer type) ===" << std::endl;
    std::cout << "Using shared/contiguous GPU and GPU memory pools" << std::endl;
    std::cout << "CUDA device: " << cuda_device_id << std::endl;
    std::cout << "Allocating " << total_num_buffers << " SRS channel estimate buffers" << std::endl;
    
    // Set active GPU device directly
    cudaError_t err = cudaSetDevice(cuda_device_id);
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("Failed to set CUDA device: ") + cudaGetErrorString(err));
    }
    
    // Calculate size of each buffer (same as original CvSrsChestMemoryBank)
    // Each CV buffer is a 3-dim buffer with dimensions: (nPrbG * nGnbAnt * nUeLayers)
    // Using complex float (2 floats = 8 bytes per complex number)
    
    std::cout << "Each buffer size: " << buffer_size << " bytes" << std::endl;
    std::cout << "  Dimensions: " << config.num_prg << " PRBs x " 
              << config.num_gnb_ant << " antennas x " 
              << config.num_ue_layer << " layers" << std::endl;
    std::cout << "Total GPU memory: " << (buffer_size * total_num_buffers) << " bytes ("
              << (buffer_size * total_num_buffers / 1024.0 / 1024.0) << " MB)" << std::endl;

    // Validate shared buffer pointer
    if (cpu_buf_start_addr == nullptr || gpu_buf_start_addr == nullptr) {
        throw std::runtime_error("Shared buffer pointers are null");
    }

    // get the GPU memory base address of the contiguous memory region
    if (!is_aligned_for_type<__half2>(gpu_buf_start_addr)) {
        throw std::runtime_error("GPU memory base address is not aligned for __half2");
    }
    gpu_buff_base_addr = reinterpret_cast<__half2*>(gpu_buf_start_addr);

    // get the CPU memory base address of the contiguous memory region
    if (!is_aligned_for_type<CVSrsChestBuff_contMemAlloc>(cpu_buf_start_addr)) {
        throw std::runtime_error("CPU memory base address is not aligned for CVSrsChestBuff_contMemAlloc");
    }
    arr_cv_srs_chest_buff_base_addr = reinterpret_cast<CVSrsChestBuff_contMemAlloc*>(cpu_buf_start_addr);

    // Create CVSrsChestBuff_contMemAlloc at (arr_cv_srs_chest_buff_base_addr + idx) using placement new
    for (uint32_t idx = 0; idx < total_num_buffers; idx++) {
        uint8_t* gpu_buff_addr = reinterpret_cast<uint8_t*>(gpu_buff_base_addr + idx * _config.num_prg * _config.num_gnb_ant * _config.num_ue_layer);
        dev_buf_view* buffer_view = new dev_buf_view(gpu_buff_addr, buffer_size);

        // Placement new: construct at the pre-allocated slot in the shared CPU pool; store pointer for view
        CVSrsChestBuff_contMemAlloc* cv_buffer = new (arr_cv_srs_chest_buff_base_addr + idx) CVSrsChestBuff_contMemAlloc(buffer_view);
        cv_buffer->setSrsChestBuffState(slot_command_api::SRS_CHEST_BUFF_NONE);
        arr_cv_srs_chest_buff[idx] = cv_buffer;

        // Add to free pool
        memIndexPool.push(idx);
    }

    std::cout << "Free buffer pool initialized with " << memIndexPool.size() << " buffers" << std::endl;
}

SimpleCvSrsChestMemoryBank::~SimpleCvSrsChestMemoryBank()
{
    if (!is_shared_memory) {
        for (uint32_t idx = 0; idx < total_num_buffers; idx++) {
            delete arr_cv_srs_chest_buff[idx];
            arr_cv_srs_chest_buff[idx] = nullptr;
        }
        std::cout << "All GPU buffers freed." << std::endl;
    } else {
        // Shared path: objects were placement-new'd at arr_cv_srs_chest_buff_base_addr + idx; call destructors only (do not delete)
        for (uint32_t idx = 0; idx < total_num_buffers; idx++) {
            CVSrsChestBuff_contMemAlloc* p = arr_cv_srs_chest_buff[idx];
            if (p) {
                p->~CVSrsChestBuff_contMemAlloc();
                arr_cv_srs_chest_buff[idx] = nullptr;
            }
        }
    }
}

void SimpleCvSrsChestMemoryBank::printBufferInfo() const
{
    std::cout << "\n=== Buffer Information ===" << std::endl;
    std::cout << "Total buffers: " << total_num_buffers << std::endl;

    for (uint32_t idx = 0; idx < total_num_buffers; idx++) {
        CVSrsChestBuff_contMemAlloc* buffer = arr_cv_srs_chest_buff[idx];
        if (buffer) {
            std::cout << "Buffer[" << idx << "]: "
                      << "GPU addr = " << static_cast<const void*>(buffer->getAddr())
                      << ", size = " << buffer_size << " bytes"
                      << ", state = " << buffer->getSrsChestBuffState()
                      << ", RNTI = " << buffer->getRnti() << std::endl;
        }
    }
}

bool SimpleCvSrsChestMemoryBank::areBuffersContiguous() const
{
    std::cout << "\n=== Checking Buffer Contiguity ===" << std::endl;
    
    if (total_num_buffers < 2) {
        std::cout << "Need at least 2 buffers to check contiguity" << std::endl;
        return false;
    }
    
    bool all_contiguous = true;
    
    
    for (uint32_t idx = 0; idx < total_num_buffers - 1; idx++)
    {
        CVSrsChestBuff_contMemAlloc* curr = arr_cv_srs_chest_buff[idx];
        CVSrsChestBuff_contMemAlloc* next = arr_cv_srs_chest_buff[idx + 1];
        
        if (curr && next) {
            const uint8_t* curr_addr = curr->getAddr();
            const uint8_t* next_addr = next->getAddr();
            
            // Check if next buffer immediately follows current buffer
            const uint8_t* expected_next_addr = curr_addr + buffer_size;
            const bool is_contiguous = (expected_next_addr == next_addr);
            
            if (!is_contiguous) {
                const ptrdiff_t gap = next_addr - expected_next_addr;
                std::cout << "NOT CONTIGUOUS (gap: " << gap << " bytes)" << std::endl;
                all_contiguous = false;
            }
        }
    }

    std::cout << "\nResult: Buffers are " << (all_contiguous ? "CONTIGUOUS" : "NOT CONTIGUOUS") << std::endl;

    return all_contiguous;
}

int SimpleCvSrsChestMemoryBank::preAllocateBuffer(uint32_t cell_id, uint32_t rnti, uint16_t buffer_idx, uint32_t usage, CVSrsChestBuff_contMemAlloc** ptr, uint32_t* realBuffIdx_out)
{
    // Validate input arguments
    if (ptr == nullptr || usage == 0 || rnti >= CV_INVALID_RNTI) {
        std::cerr << "preAllocateBuffer: Invalid input argument - "
                  << "ptr=" << static_cast<void*>(ptr) 
                  << ", rnti=" << rnti 
                  << ", usage=" << usage << std::endl;
        return -1;
    }
    
    // Check if cell has been allocated a memory pool
    if (srsChEstBuffIndexMap.find(cell_id) == srsChEstBuffIndexMap.end()) {
        std::cerr << "preAllocateBuffer: Cell ID " << cell_id << " doesn't exist!" << std::endl;
        return -1;
    }
    
    // Validate buffer_idx is within cell's allocated pool
    if (buffer_idx >= srsChEstBuffIndexMap[cell_id].mempoolSize) {
        std::cerr << "preAllocateBuffer: Invalid buffer_idx " << buffer_idx 
                  << " for cellId " << cell_id 
                  << " with mempoolsize " << srsChEstBuffIndexMap[cell_id].mempoolSize 
                  << " rnti=" << rnti << std::endl;
        return -1;
    }
    
    // Translate FAPI buffer index to real global buffer index
    const uint32_t realBuffIndex = srsChEstBuffIndexMap[cell_id].indexMap[buffer_idx];
    if (realBuffIdx_out != nullptr) {
        *realBuffIdx_out = realBuffIndex;
    }

    // Get the actual buffer from the global array
    *ptr = arr_cv_srs_chest_buff[realBuffIndex];
    CVSrsChestBuff_contMemAlloc* buffer = *ptr;
    
    if (!buffer) {
        std::cerr << "preAllocateBuffer: Buffer at realBuffIndex " << realBuffIndex << " is null" << std::endl;
        return -1;
    }
    
    // Check current buffer state - must be NONE or READY to allow pre-allocation
    const slot_command_api::srsChestBuffState currSrsChestBuffState = buffer->getSrsChestBuffState();
    if (currSrsChestBuffState != slot_command_api::SRS_CHEST_BUFF_NONE && 
        currSrsChestBuffState != slot_command_api::SRS_CHEST_BUFF_READY) {
        std::cerr << "preAllocateBuffer: SRS Chest Buffer in Use - "
                  << "cell_id=" << cell_id 
                  << " rnti=" << rnti 
                  << " usage=" << usage 
                  << " buffer_idx=" << buffer_idx 
                  << " SrsChestBuffState=" << static_cast<int>(currSrsChestBuffState) << std::endl;
        return -1;
    }
    
    // Initialize buffer with UE and cell information
    buffer->init(rnti, buffer_idx, cell_id, usage);
    
    // Configure SRS info with config values
    buffer->configSrsInfo(
        config.num_prg,
        config.num_gnb_ant,
        config.num_ue_layer,
        config.srs_prg_size,
        config.srs_start_prg,
        config.srs_start_valid_prg,
        config.srs_n_valid_prg
    );
    
    /*
    const slot_command_api::srsChestBuffState newState = buffer->getSrsChestBuffState();
    std::cout << "preAllocateBuffer: SRS Chest Buffer Pointer=" << static_cast<void*>(buffer)
              << ", newState=" << static_cast<int>(newState) << std::endl;
    */
    
    return 0;
}

int SimpleCvSrsChestMemoryBank::retrieveBuffer(uint32_t cell_id, uint32_t rnti, uint16_t buffer_idx, CVSrsChestBuff_contMemAlloc** ptr)
{
    // Validate input arguments  
    if (ptr == nullptr || rnti >= CV_INVALID_RNTI) {
        std::cerr << "retrieveBuffer: Invalid input argument - "
                  << "ptr=" << static_cast<void*>(ptr) 
                  << ", rnti=" << rnti << std::endl;
        return -1;
    }
    
    // Check if cell has been allocated a memory pool
    if (srsChEstBuffIndexMap.find(cell_id) == srsChEstBuffIndexMap.end()) {
        std::cerr << "retrieveBuffer: Cell ID " << cell_id << " doesn't exist!" << std::endl;
        return -1;
    }
    
    // Validate buffer_idx is within cell's allocated pool
    if (buffer_idx >= srsChEstBuffIndexMap[cell_id].mempoolSize) {
        std::cerr << "retrieveBuffer: Invalid buffer_idx " << buffer_idx 
                  << " for cellId " << cell_id 
                  << " with mempoolsize " << srsChEstBuffIndexMap[cell_id].mempoolSize 
                  << " rnti=" << rnti << std::endl;
        return -1;
    }
    
    // Translate FAPI buffer index to real global buffer index
    const uint32_t realBuffIndex = srsChEstBuffIndexMap[cell_id].indexMap[buffer_idx];
    
    std::cout << "retrieveBuffer: cell_id=" << cell_id 
              << " rnti=" << rnti 
              << " FAPI buffer_idx=" << buffer_idx 
              << " realBuffIndex=" << realBuffIndex << std::endl;
    
    // Get the actual buffer from the global array
    *ptr = arr_cv_srs_chest_buff[realBuffIndex];
    CVSrsChestBuff_contMemAlloc* buffer = *ptr;
    
    if (!buffer) {
        std::cerr << "retrieveBuffer: Buffer at realBuffIndex " << realBuffIndex << " is null" << std::endl;
        return -1;
    }
    
    // Check buffer state - must be READY for retrieval
    const slot_command_api::srsChestBuffState currSrsChestBuffState = buffer->getSrsChestBuffState();
    if (currSrsChestBuffState != slot_command_api::SRS_CHEST_BUFF_READY) {
        std::cerr << "retrieveBuffer: SRS Chest Buffer not ready - "
                  << "cell_id=" << cell_id 
                  << " rnti=" << rnti 
                  << " SrsChestBuffState=" << static_cast<int>(currSrsChestBuffState) 
                  << " buffer_idx=" << buffer_idx << std::endl;
        return -1;
    }
    
    return 0;
}

void SimpleCvSrsChestMemoryBank::updateSrsChestBufferState(uint32_t cell_id, uint16_t buffer_idx, slot_command_api::srsChestBuffState srs_chest_buff_state)
{
    // Check if cell exists
    if (srsChEstBuffIndexMap.find(cell_id) == srsChEstBuffIndexMap.end()) {
        std::cerr << "updateSrsChestBufferState: Cell ID " << cell_id << " doesn't exist!" << std::endl;
        return;
    }
    
    // Validate buffer_idx
    if (buffer_idx >= srsChEstBuffIndexMap[cell_id].mempoolSize) {
        std::cerr << "updateSrsChestBufferState: Invalid buffer_idx " << buffer_idx 
                  << " for cellId " << cell_id 
                  << " with mempoolsize " << srsChEstBuffIndexMap[cell_id].mempoolSize << std::endl;
        return;
    }
    
    // Get real buffer index and update state
    const uint32_t realBuffIndex = srsChEstBuffIndexMap[cell_id].indexMap[buffer_idx];
    CVSrsChestBuff_contMemAlloc* buffer = arr_cv_srs_chest_buff[realBuffIndex];
    if (buffer) {
        buffer->setSrsChestBuffState(srs_chest_buff_state);
    }
}

slot_command_api::srsChestBuffState SimpleCvSrsChestMemoryBank::getSrsChestBufferState(uint32_t cell_id, uint16_t buffer_idx)
{
    // Check if cell exists
    if (srsChEstBuffIndexMap.find(cell_id) == srsChEstBuffIndexMap.end()) {
        std::cerr << "getSrsChestBufferState: Cell ID " << cell_id << " doesn't exist!" << std::endl;
        return slot_command_api::SRS_CHEST_BUFF_NONE;
    }
    
    // Validate buffer_idx
    if (buffer_idx >= srsChEstBuffIndexMap[cell_id].mempoolSize) {
        std::cerr << "getSrsChestBufferState: Invalid buffer_idx " << buffer_idx 
                  << " for cellId " << cell_id 
                  << " with mempoolsize " << srsChEstBuffIndexMap[cell_id].mempoolSize << std::endl;
        return slot_command_api::SRS_CHEST_BUFF_NONE;
    }
    
    // Get real buffer index and return state
    const uint32_t realBuffIndex = srsChEstBuffIndexMap[cell_id].indexMap[buffer_idx];
    CVSrsChestBuff_contMemAlloc* buffer = arr_cv_srs_chest_buff[realBuffIndex];
    if (buffer) {
        return buffer->getSrsChestBuffState();
    }
    return slot_command_api::SRS_CHEST_BUFF_NONE;
}

void SimpleCvSrsChestMemoryBank::updateSrsChestBufferUsage(uint32_t cell_id, uint32_t rnti, uint16_t buffer_idx, uint32_t usage)
{
    // Check if cell exists
    if (srsChEstBuffIndexMap.find(cell_id) == srsChEstBuffIndexMap.end()) {
        std::cerr << "updateSrsChestBufferUsage: Cell ID " << cell_id << " doesn't exist!" << std::endl;
        return;
    }
    
    // Validate buffer_idx
    if (buffer_idx >= srsChEstBuffIndexMap[cell_id].mempoolSize) {
        std::cerr << "updateSrsChestBufferUsage: Invalid buffer_idx " << buffer_idx 
                  << " for cellId " << cell_id 
                  << " with mempoolsize " << srsChEstBuffIndexMap[cell_id].mempoolSize << std::endl;
        return;
    }
    
    // Get real buffer index and update usage
    const uint32_t realBuffIndex = srsChEstBuffIndexMap[cell_id].indexMap[buffer_idx];
    CVSrsChestBuff_contMemAlloc* buffer = arr_cv_srs_chest_buff[realBuffIndex];
    if (buffer) {
        buffer->setSrsChestBuffUsage(usage);
    }
}

uint32_t SimpleCvSrsChestMemoryBank::getSrsChestBufferUsage(uint32_t cell_id, uint32_t rnti, uint16_t buffer_idx)
{
    // Check if cell exists
    if (srsChEstBuffIndexMap.find(cell_id) == srsChEstBuffIndexMap.end()) {
        std::cerr << "getSrsChestBufferUsage: Cell ID " << cell_id << " doesn't exist!" << std::endl;
        return 0;
    }
    
    // Validate buffer_idx
    if (buffer_idx >= srsChEstBuffIndexMap[cell_id].mempoolSize) {
        std::cerr << "getSrsChestBufferUsage: Invalid buffer_idx " << buffer_idx 
                  << " for cellId " << cell_id 
                  << " with mempoolsize " << srsChEstBuffIndexMap[cell_id].mempoolSize << std::endl;
        return 0;
    }
    
    // Get real buffer index and return usage
    const uint32_t realBuffIndex = srsChEstBuffIndexMap[cell_id].indexMap[buffer_idx];
    CVSrsChestBuff_contMemAlloc* buffer = arr_cv_srs_chest_buff[realBuffIndex];
    if (buffer) {
        return buffer->getSrsChestBuffUsage();
    }
    return 0;
}

bool SimpleCvSrsChestMemoryBank::memPoolAllocatePerCell(uint32_t requestedBy, uint16_t cell_id, uint32_t mempoolSize)
{
    // Validate inputs
    if (memIndexPool.size() < mempoolSize) {
        std::cerr << "memPoolAllocatePerCell: Insufficient buffers (requested=" << mempoolSize 
                  << ", available=" << memIndexPool.size() << ")" << std::endl;
        return false;
    }
    
    // Check if cell already has allocation
    if (srsChEstBuffIndexMap.find(cell_id) != srsChEstBuffIndexMap.end()) {
        std::cout << "memPoolAllocatePerCell: Cell " << cell_id 
                  << " already exists in srsChEstBuffIndexMap (requestedBy=" << requestedBy 
                  << ", stored=" << srsChEstBuffIndexMap[cell_id].requestedBy << ")" << std::endl;
        // For testing, allow re-allocation from same source
        if (srsChEstBuffIndexMap[cell_id].requestedBy == requestedBy) {
            std::cout << "  Ignoring duplicate allocation request" << std::endl;
            return true;
        } else {
            std::cerr << "  ERROR: Different requestedBy" << std::endl;
            return false;
        }
    }
    
    // Create new entry in map and populate it directly (automatically increases map size by 1)
    srsChEstBuffIndexMap[cell_id].requestedBy = requestedBy;
    srsChEstBuffIndexMap[cell_id].mempoolSize = mempoolSize;
    srsChEstBuffIndexMap[cell_id].indexMap.resize(mempoolSize);
    
    for (uint32_t idx = 0; idx < mempoolSize; idx++) {
        const uint32_t realBuffIndex = memIndexPool.front();
        srsChEstBuffIndexMap[cell_id].indexMap[idx] = realBuffIndex;
        memIndexPool.pop();
    }

    return true;
}

bool SimpleCvSrsChestMemoryBank::memPoolDeAllocatePerCell(uint16_t cell_id)
{
    bool retVal = true;
    
    if (srsChEstBuffIndexMap.find(cell_id) != srsChEstBuffIndexMap.end()) {
        const uint32_t mempoolSize = srsChEstBuffIndexMap[cell_id].mempoolSize;
        
        // Return buffers to free pool
        for (uint32_t idx = 0; idx < mempoolSize; idx++) {
            const uint32_t realBuffIndex = srsChEstBuffIndexMap[cell_id].indexMap[idx];
            memIndexPool.push(realBuffIndex);
        }
        
        // Remove cell entry from map (decreases map size by 1)
        srsChEstBuffIndexMap.erase(cell_id);
    } else {
        retVal = false;
        std::cerr << "memPoolDeAllocatePerCell: Cell id " << cell_id 
                  << " doesn't exist in srsChEstBuffIndexMap" << std::endl;
    }
    
    return retVal;
}

void CVSrsChestBuff_contMemAlloc::getSrsPrgInfo(uint8_t* pSrsPrgSize_out, uint16_t* pSrsStartPrg_out, uint16_t* pSrsStartValidPrg_out, uint16_t* pSrsNValidPrg_out)
{
    *pSrsPrgSize_out    = srsPrgSize;
    *pSrsStartPrg_out   = srsStartPrg;
    *pSrsStartValidPrg_out = srsStartValidPrg;
    *pSrsNValidPrg_out     = srsNValidPrg;
}

void CVSrsChestBuff_contMemAlloc::setSrsChestBuffState(slot_command_api::srsChestBuffState  _srs_chest_buff_state)
{
    srs_chest_buff_state = _srs_chest_buff_state;
}

void CVSrsChestBuff_contMemAlloc::setSrsChestBuffUsage(uint32_t  _srs_chest_buff_usage)
{
    srs_chest_buff_usage = _srs_chest_buff_usage;
}

void CVSrsChestBuff_contMemAlloc::configSrsInfo(uint16_t nPrg, uint8_t nAnt, uint8_t nLayer, uint8_t srsPrgSize_in, uint16_t srsStartPrg_in, uint16_t startValidPrg_in, uint16_t nValidPrg_in)
{
    srsPrgSize           = srsPrgSize_in;
    srsStartPrg          = srsStartPrg_in;
    srsStartValidPrg     = startValidPrg_in;
    srsNValidPrg         = nValidPrg_in;
    srs_chest_buff_state = slot_command_api::SRS_CHEST_BUFF_REQUESTED;

    // Set descriptor
    std::array<int, 3> dims = {nPrg, nAnt, nLayer};
    cuphyStatus_t setupStatus = cuphySetTensorDescriptor(buffDesc.handle(),
                                                        CUPHY_C_16F,
                                                        dims.size(),
                                                        dims.data(),
                                                        nullptr,
                                                        static_cast<int>(cuphy::tensor_flags::align_tight));

}

void CVSrsChestBuff_contMemAlloc::setSfnSlot(uint16_t _sfn, uint16_t _slot){
    sfn = _sfn;
    slot = _slot;
}

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

#include "cv_memory_bank_srs_chest.hpp"
#include <iostream>
#include <queue>
#include <unordered_map>
#include <yaml-cpp/yaml.h>
#include <string>
#include <span>

/**
 * Configuration structure for SRS memory bank test
 * Loaded from YAML configuration file
 */
struct TestConfig {
    // CUDA device configuration
    uint32_t cuda_device_id{0};        ///< CUDA device ID to use for GPU operations
    
    bool is_contiguous_gpu_mem{true}; ///< True if the buffers are contiguous in GPU memory

    uint32_t num_time_slots{0};       ///< Number of time slots in the test scenario

    uint64_t slot_interval_ns{0};     ///< Slot interval in nanoseconds

    uint32_t num_cell{1};       ///< Number of cells in the test scenario

    // Buffer dimensions
    uint32_t num_ue_layer{4};      ///< Number of UE antenna layers (MAX_UE_SRS_ANT_PORTS)
    uint32_t num_gnb_ant{64};      ///< Number of gNodeB antennas (MAX_AP_PER_SLOT_SRS)
    uint32_t num_prg{136};        ///< Number of Physical Resource Block Groups (ORAN_MAX_PRB)
    
    // Test scenario parameters
    uint32_t max_num_srs_ue_per_slot{32}; ///< Number of SRS UEs scheduled per S-slot
    uint32_t num_srs_buffers_per_cell{1024};     ///< Total number of SRS buffers to allocate per cell
    uint32_t num_srs_buffers{1024};     ///< Total number of SRS buffers to allocate
    
    uint32_t num_srs_ues_per_cell{2}; ///< Number of SRS UEs per cell
    uint32_t rnti_base{100};     ///< Base RNTI for test UEs
    uint32_t cell_id_base{0};    ///< Base cell ID for multi-cell tests
    
    // SRS PRB configuration
    uint8_t  srs_prg_size{2};         ///< Size of each PRB group for SRS
    uint16_t srs_start_prg{0};        ///< Starting PRB group index
    uint16_t srs_start_valid_prg{0};  ///< First valid PRB group
    uint16_t srs_n_valid_prg{136};    ///< Number of valid PRB groups
    
    // FAPI allocation request type
    uint32_t alloc_request{0x02}; ///< Test allocation request type (SCF_FAPI_CONFIG_REQUEST)
    
    /**
     * Load configuration from YAML file
     * 
     * @param yaml_path Path to YAML configuration file
     * @return true on success, false on failure
     */
    bool loadFromYaml(const char* yaml_path) {
        try {
            YAML::Node config = YAML::LoadFile(yaml_path);

            if (config["NUM_TIME_SLOTS"]) {
                num_time_slots = config["NUM_TIME_SLOTS"].as<uint32_t>();
            }

            if (config["SLOT_INTERVAL_NS"]) {
                slot_interval_ns = config["SLOT_INTERVAL_NS"].as<uint64_t>();
            }

            if (config["NUM_CELL"]) {
                num_cell = config["NUM_CELL"].as<uint32_t>();
            }

            if (config["CUDA_DEVICE_ID"]) {
                cuda_device_id = config["CUDA_DEVICE_ID"].as<uint32_t>();
            }

            if (config["NUM_UE_ANT_PORT"]) {
                num_ue_layer = config["NUM_UE_ANT_PORT"].as<uint32_t>();
            }

            if (config["NUM_BS_ANT_PORT"]) {
                num_gnb_ant = config["NUM_BS_ANT_PORT"].as<uint32_t>();
            }

            if (config["NUM_PRG_PER_CELL"]) {
                num_prg = config["NUM_PRG_PER_CELL"].as<uint32_t>();
            }

            if (config["PRG_SIZE"]) {
                srs_prg_size = config["PRG_SIZE"].as<uint8_t>();
            }

            if (config["NUM_SRS_UE_PER_SLOT"]) {
                max_num_srs_ue_per_slot = config["NUM_SRS_UE_PER_SLOT"].as<uint32_t>();
            }

            if (config["NUM_SRS_UE_PER_CELL"]) {
                num_srs_ues_per_cell = config["NUM_SRS_UE_PER_CELL"].as<uint32_t>();
            }
            
            if (!config["srs_mem_bank_config"]) {
                std::cerr << "Warning: 'srs_mem_bank_config' section not found in YAML, using defaults" << std::endl;
                return false;
            }
            
            YAML::Node srs_mem_bank_config = config["srs_mem_bank_config"];
            
            // Load is_contiguous_gpu_mem configuration
            if (srs_mem_bank_config["is_contiguous_gpu_mem"]) is_contiguous_gpu_mem = srs_mem_bank_config["is_contiguous_gpu_mem"].as<bool>();
            if (srs_mem_bank_config["num_srs_buffers_per_cell"]) num_srs_buffers_per_cell = srs_mem_bank_config["num_srs_buffers_per_cell"].as<uint32_t>();

            num_srs_buffers = num_srs_buffers_per_cell*num_cell;
            // Load test scenario parameters
            if (srs_mem_bank_config["rnti_base"]) rnti_base = srs_mem_bank_config["rnti_base"].as<uint32_t>();
            if (srs_mem_bank_config["cell_id_base"]) cell_id_base = srs_mem_bank_config["cell_id_base"].as<uint32_t>();
            
            // Load SRS PRB configuration
            if (srs_mem_bank_config["srs_start_prg"]) srs_start_prg = srs_mem_bank_config["srs_start_prg"].as<uint16_t>();
            if (srs_mem_bank_config["srs_start_valid_prg"]) srs_start_valid_prg = srs_mem_bank_config["srs_start_valid_prg"].as<uint16_t>();
            if (srs_mem_bank_config["srs_n_valid_prg"]) srs_n_valid_prg = srs_mem_bank_config["srs_n_valid_prg"].as<uint16_t>();
            
            // Load FAPI allocation request type
            if (srs_mem_bank_config["alloc_request"]) alloc_request = srs_mem_bank_config["alloc_request"].as<uint32_t>();
            
            return true;
        } catch (const YAML::Exception& e) {
            std::cerr << "Error loading YAML config: " << e.what() << std::endl;
            return false;
        }
    }
};

template<typename T>
inline bool is_aligned_for_type(void* p) {
    return (reinterpret_cast<std::uintptr_t>(p) % alignof(T)) == 0;
}

/**
 * @brief Lightweight buffer class - NO ownership, NO cudaFree
 * 
 * This class is used to view a buffer as a contiguous memory region.
 * It is used to avoid the overhead of creating a new buffer object.
 */
 class dev_buf_view {
    public:
        dev_buf_view(uint8_t* addr, size_t size) : _addr(addr), _size(size) {}
        
        uint8_t* addr() { return _addr; }
        size_t size() const { return _size; }
        void clear() { cudaMemset(_addr, 0, _size); }
        
    private:
        uint8_t* _addr;  // Non-owning pointer
        size_t _size;
};

/**
 * @brief SRS Channel Estimate Buffer for contiguous memory allocation
 * 
 * Stores channel estimates from Sounding Reference Signals (SRS) for a single UE.
 * Each buffer contains the frequency-domain channel response across all gNodeB antennas
 * and UE antenna ports, organized by Physical Resource Block Groups (PRBs).
 */
 typedef struct _CVSrsChestBuff_contMemAlloc
 {
     public:
         /**
          * @brief Construct SRS channel estimate buffer - takes ownership of the GPU buffer
          * 
          * @param bdev  GPU device buffer pointer
          */
          _CVSrsChestBuff_contMemAlloc(dev_buf * bdev) {
             buffer.reset(bdev);
             buffer->clear();
             is_contiguous_gpu_mem = false;
         }

         /**
          * @brief Construct SRS channel estimate buffer - takes ownership of the GPU buffer view
          * 
          * @param bdev_view  GPU device buffer view pointer
          */
         _CVSrsChestBuff_contMemAlloc(dev_buf_view * bdev_view) {
            buffer_view.reset(bdev_view);
            buffer_view->clear();
            is_contiguous_gpu_mem = true;
         }

         /**
          * @brief Initialize buffer with UE and cell information
          * 
          * @param _rnti        Radio Network Temporary Identifier (UE ID)
          * @param _buffer_idx  Buffer index in memory pool
          * @param _cell_id     Cell ID this buffer belongs to
          * @param _usage       Usage counter/reference count
          * @return int         0 on success
          */
         int init(uint32_t _rnti, uint32_t _buffer_idx, uint32_t _cell_id, uint32_t _usage) {
             srs_chest_buff_state = slot_command_api::SRS_CHEST_BUFF_INIT;
             buffer_idx = _buffer_idx;
             rnti = _rnti;
             cell_id = _cell_id;
             srs_chest_buff_usage = _usage;
             return 0;
         }
 
         /**
          * @brief Configure SRS PRB allocation information
          * 
          * @param nPrg              Total number of PRB groups
          * @param nAnt              Number of gNodeB antennas
          * @param nLayer            Number of UE antenna layers/ports
          * @param srsPrgSize_in     Size of each PRB group
          * @param srsStartPrg_in    Starting PRB group index
          * @param startValidPrg_in  First valid PRB group index
          * @param nValidPrg_in      Number of valid PRB groups
          */
         void configSrsInfo(uint16_t nPrg, uint8_t nAnt, uint8_t nLayer, uint8_t srsPrgSize_in, uint16_t srsStartPrg_in, uint16_t startValidPrg_in, uint16_t nValidPrg_in);
         
         /**
          * @brief Set SFN and slot for this channel estimate
          * 
          * @param _sfn   System Frame Number (0-1023)
          * @param _slot  Slot number within frame
          */
         void setSfnSlot(uint16_t _sfn, uint16_t _slot);
         
         /**
          * @brief Get GPU buffer address
          * 
          * @return uint8_t*  Pointer to GPU device memory
          */
         uint8_t * getAddr() const { return is_contiguous_gpu_mem ? buffer_view->addr() : buffer->addr();};
         
         /**
          * @brief Get SRS PRB allocation information
          * 
          * @param[out] pSrsPrgSize_out       PRB group size
          * @param[out] pSrsStartPrg_out      Starting PRB group index
          * @param[out] pSrsStartValidPrg_out First valid PRB group index
          * @param[out] pSrsNValidPrg_out     Number of valid PRB groups
          */
         void getSrsPrgInfo(uint8_t* pSrsPrgSize_out, uint16_t* pSrsStartPrg_out, uint16_t* pSrsStartValidPrg_out, uint16_t* pSrsNValidPrg_out);
         
         /**
          * @brief Get RNTI (UE identifier)
          * 
          * @return uint32_t  Radio Network Temporary Identifier
          */
         uint32_t getRnti() const { return rnti;};
         
         /**
          * @brief Get buffer index in memory pool
          * 
          * @return uint32_t  Buffer index
          */
         uint32_t getBufferIdx() const { return buffer_idx;};
         
         /**
          * @brief Get cuPHY tensor descriptor for this buffer
          * 
          * @return cuphyTensorDescriptor_t  Tensor descriptor handle for cuPHY API
          */
         cuphyTensorDescriptor_t getSrsDescr() const { return buffDesc.handle();};
         
         /**
          * @brief Get cell ID
          * 
          * @return uint32_t  Cell identifier
          */
         uint32_t getCellId() const { return cell_id;};
         
         /**
          * @brief Get System Frame Number
          * 
          * @return uint16_t  SFN (0-1023)
          */
         uint16_t getSfn(){ return sfn;};
         
         /**
          * @brief Get slot number
          * 
          * @return uint16_t  Slot number within frame
          */
         uint16_t getSlot(){ return slot;};
         
         /**
          * @brief Set buffer state
          * 
          * @param _srs_chest_buff_state  New state (INIT, REQUESTED, READY, NONE)
          */
         void setSrsChestBuffState(slot_command_api::srsChestBuffState _srs_chest_buff_state);
         
         /**
          * @brief Get buffer state
          * 
          * @return slot_command_api::srsChestBuffState  Current state
          */
         slot_command_api::srsChestBuffState getSrsChestBuffState() const {return srs_chest_buff_state;};
         
         /**
          * @brief Get buffer usage counter
          * 
          * @return uint32_t  Reference/usage count
          */
         uint32_t getSrsChestBuffUsage() const {return srs_chest_buff_usage;};
         
         /**
          * @brief Set buffer usage counter
          * 
          * @param _srs_chest_buff_usage  New usage count
          */
         void setSrsChestBuffUsage(uint32_t _srs_chest_buff_usage);
 
         /**
          * @brief Clear/reset buffer to invalid state
          */
         void clear() {
             srs_chest_buff_state = slot_command_api::SRS_CHEST_BUFF_NONE;
             rnti = CV_INVALID_RNTI;
             cell_id = 0;
             buffer_idx = CV_INVALID_CESHT_BUF_INDEX;
             srs_chest_buff_usage = 0;
             sfn = 0xFFFF;
             slot = 0xFFFF;
         }
         
     private:
         std::unique_ptr<dev_buf> buffer;                           ///< GPU device memory buffer for channel estimates
         std::unique_ptr<dev_buf_view> buffer_view;                 ///< View of the GPU buffer as a contiguous memory region
         cuphy::tensor_desc buffDesc;                               ///< cuPHY tensor descriptor for this buffer
         slot_command_api::srsChestBuffState srs_chest_buff_state;  ///< Buffer state (INIT, REQUESTED, READY, NONE)
         uint32_t rnti;                                             ///< Radio Network Temporary Identifier (UE ID)
         uint32_t buffer_idx;                                       ///< Buffer index in memory pool
         uint32_t cell_id;                                          ///< Cell ID this buffer belongs to
         uint32_t srs_chest_buff_usage;                             ///< Buffer reference/usage counter
         uint16_t sfn;                                              ///< System Frame Number (0-1023) when channel estimate was captured
         uint16_t slot;                                             ///< Slot number within frame when channel estimate was captured
         uint16_t srsStartPrg;                                      ///< Starting PRB group index for SRS allocation
         uint16_t srsStartValidPrg;                                 ///< First valid PRB group index
         uint16_t srsNValidPrg;                                     ///< Number of valid PRB groups
         uint8_t  srsPrgSize;                                       ///< Size of each PRB group
         bool is_contiguous_gpu_mem{true};                          ///< True if the buffer is contiguous in GPU memory
 } CVSrsChestBuff_contMemAlloc; // alignment size is 8 bytes

/**
 * Simplified SRS Channel Estimate Memory Bank for Testing
 * 
 * Uses the _CVSrsChestBuff_contMemAlloc and CellIdtoSrsBuffIndexMap classes
 * Provides simplified construction without PhyDriverCtx/FhProxy dependencies.
 * This version focuses only on GPU memory allocation testing.
 * 
 * NOTE: This simplified version does NOT use GpuDevice to avoid PhyDriverCtx requirements.
 * It allocates buffers directly using CUDA APIs.
 */
class SimpleCvSrsChestMemoryBank
{
public:
    /**
     * Construct simplified SRS memory bank
     * 
     * All configuration including CUDA device ID is loaded from the TestConfig.
     * 
     * @param config Test configuration containing all parameters including CUDA device ID
     */
    SimpleCvSrsChestMemoryBank(const TestConfig& _config);

    /**
     * Construct simplified SRS memory bank with externally allocated shared CPU and GPU memory pools
     * 
     * All configuration including CUDA device ID is loaded from the TestConfig.
     * 
     * @param config Test configuration containing all parameters including CUDA device ID
     * @param cpu_buf_start_addr Shared CPU memory buffer start address
     * @param gpu_buf_start_addr Shared GPU memory buffer start address
     */
    SimpleCvSrsChestMemoryBank(const TestConfig& _config, void* cpu_buf_start_addr, void* gpu_buf_start_addr);
    
    /**
     * Destructor - frees all GPU buffers
     */
    ~SimpleCvSrsChestMemoryBank();
    
    /**
     * Get total number of allocated buffers
     * 
     * @return Number of buffers
     */
    uint32_t getNumBuffers() const { return total_num_buffers; }
    
    /**
     * Get buffer at specific index
     * 
     * @param idx Buffer index
     * @return Pointer to buffer, or nullptr if invalid index
     */
     CVSrsChestBuff_contMemAlloc* getBuffer(uint32_t idx) const
    {
        if (idx < total_num_buffers) {
            return arr_cv_srs_chest_buff[idx];
        }
        return nullptr;
    }
    
    /**
     * Pre-allocate a buffer for future use
     * 
     * @param cell_id Cell ID requesting the buffer
     * @param rnti UE RNTI (Radio Network Temporary Identifier)
     * @param buffer_idx FAPI buffer index (within cell's pool, 0 to mempoolSize-1)
     * @param usage Buffer usage/reference count (must be > 0)
     * @param ptr Output pointer to allocated buffer
     * @param realBuffIdx_out Output pointer to real buffer index (optional)
     * @return int 0 on success, negative on failure
     */
    int preAllocateBuffer(uint32_t cell_id, uint32_t rnti, uint16_t buffer_idx, uint32_t usage, CVSrsChestBuff_contMemAlloc** ptr, uint32_t* realBuffIdx_out = nullptr);
    
    /**
     * Retrieve an existing buffer
     * 
     * @param cell_id Cell ID
     * @param rnti UE RNTI
     * @param buffer_idx Buffer index to retrieve
     * @param ptr Output pointer to retrieved buffer
     * @return int 0 on success, negative on failure
     */
    int retrieveBuffer(uint32_t cell_id, uint32_t rnti, uint16_t buffer_idx, CVSrsChestBuff_contMemAlloc** ptr);
    
    /**
     * Update buffer state
     * 
     * @param cell_id Cell ID
     * @param buffer_idx Buffer index
     * @param srs_chest_buff_state New state to set
     */
    void updateSrsChestBufferState(uint32_t cell_id, uint16_t buffer_idx, slot_command_api::srsChestBuffState srs_chest_buff_state);
    
    /**
     * Get buffer state
     * 
     * @param cell_id Cell ID
     * @param buffer_idx Buffer index
     * @return slot_command_api::srsChestBuffState Current buffer state
     */
    slot_command_api::srsChestBuffState getSrsChestBufferState(uint32_t cell_id, uint16_t buffer_idx);
    
    /**
     * Update buffer usage counter
     * 
     * @param cell_id Cell ID
     * @param rnti UE RNTI
     * @param buffer_idx Buffer index
     * @param usage New usage count
     */
    void updateSrsChestBufferUsage(uint32_t cell_id, uint32_t rnti, uint16_t buffer_idx, uint32_t usage);
    
    /**
     * Get buffer usage counter
     * 
     * @param cell_id Cell ID
     * @param rnti UE RNTI
     * @param buffer_idx Buffer index
     * @return uint32_t Current usage count
     */
    uint32_t getSrsChestBufferUsage(uint32_t cell_id, uint32_t rnti, uint16_t buffer_idx);
    
    /**
     * Allocate a memory pool partition for a specific cell
     * 
     * @param requestedBy Source of the allocation request (e.g., SCF_FAPI_CONFIG_REQUEST)
     * @param cell_id Cell ID to allocate pool for
     * @param mempoolSize Number of buffers to allocate to this cell
     * @return bool true on success, false if insufficient buffers available
     */
    bool memPoolAllocatePerCell(uint32_t requestedBy, uint16_t cell_id, uint32_t mempoolSize);
    
    /**
     * Deallocate a cell's memory pool partition
     * 
     * @param cell_id Cell ID to deallocate
     * @return bool true on success, false on failure
     */
    bool memPoolDeAllocatePerCell(uint16_t cell_id);
    
    /**
     * Print buffer information
     */
    void printBufferInfo() const;
    
    /**
     * Check if buffers are contiguous in GPU memory
     * 
     * @return false (buffers are NOT contiguous - each has separate cudaMalloc)
     */
    bool areBuffersContiguous() const;

private:
    TestConfig config;                                                            ///< Test configuration
    bool is_shared_memory{false};                                                 ///< True if buffers are shared in CPU memory
    uint32_t cuda_device_id {0};                                                   ///< CUDA device ID being used
    uint32_t total_num_buffers;                                                   ///< Total number of allocated buffers
    uint32_t buffer_size;
    std::array<CVSrsChestBuff_contMemAlloc*, slot_command_api::MAX_SRS_CHEST_BUFFERS> arr_cv_srs_chest_buff; ///< Array of CVSrsChestBuff_contMemAlloc pointers
    CVSrsChestBuff_contMemAlloc* arr_cv_srs_chest_buff_base_addr{nullptr};          ///< Base address of the CVSrsChestBuff_contMemAlloc array
    __half2* gpu_buff_base_addr {nullptr};                                         ///< Base address of GPU memory for contiguous memory allocation
    std::queue<uint32_t> memIndexPool;                                            ///< Queue of free buffer indices
    std::unordered_map<uint32_t, CellIdtoSrsBuffIndexMap> srsChEstBuffIndexMap;   ///< Map from cell ID to buffer indices
};


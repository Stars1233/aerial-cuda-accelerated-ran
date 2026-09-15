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
#include "nv_lockfree.hpp"
#include "gpudevice.hpp"
#include "common_utils.h"
#include "srs_ipc_manager.hpp"
#include <H5Cpp.h>
#include <cuda_runtime.h>
#include <cstring>
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
 * Simplified SRS Channel Estimate Memory Bank for Testing
 *
 * Uses the CVSrsChestBuff and CellIdtoSrsBuffIndexMap classes
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
     * @param gpu_mem_pool       Non-owning shared GPU IPC memory pool; the caller
     *                           must keep it alive until this memory bank is destroyed
     */
    SimpleCvSrsChestMemoryBank(const TestConfig& _config, void* cpu_buf_start_addr, nv::lock_free_mem_pool<uint8_t>* gpu_mem_pool);
    
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
     CVSrsChestBuff* getBuffer(uint32_t idx) const
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
    int preAllocateBuffer(uint32_t cell_id, uint32_t rnti, uint16_t buffer_idx, uint32_t usage, CVSrsChestBuff** ptr, uint32_t* realBuffIdx_out = nullptr);
    
    /**
     * Retrieve an existing buffer
     * 
     * @param cell_id Cell ID
     * @param rnti UE RNTI
     * @param buffer_idx Buffer index to retrieve
     * @param ptr Output pointer to retrieved buffer
     * @return int 0 on success, negative on failure
     */
    int retrieveBuffer(uint32_t cell_id, uint32_t rnti, uint16_t buffer_idx, CVSrsChestBuff** ptr);
    
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
    std::array<CVSrsChestBuff*, slot_command_api::MAX_SRS_CHEST_BUFFERS> arr_cv_srs_chest_buff; ///< Array of CVSrsChestBuff pointers
    CVSrsChestBuff* arr_cv_srs_chest_buff_base_addr{nullptr};          ///< Base address of the CVSrsChestBuff array
    __half2* gpu_buff_base_addr {nullptr};                                         ///< Base address of GPU memory for contiguous memory allocation
    std::queue<uint32_t> memIndexPool;                                            ///< Queue of free buffer indices
    std::unordered_map<uint32_t, CellIdtoSrsBuffIndexMap> srsChEstBuffIndexMap;   ///< Map from cell ID to buffer indices

    nv::lock_free_mem_pool<uint8_t>* gpu_mem_pool{nullptr};
};

//! Geometry attributes saved by cuphydriver `SrsIpcManager::dump_h5` in the
//! root group of every `cubb_srs_buffers_<dump_idx>_SFN_<sfn>.<slot>.h5` dump.
struct CubbTvAttrs {
    uint32_t gpu_pool_len{0};
    uint32_t gpu_buf_size{0};
    uint32_t num_prg{0};
    uint32_t num_gnb_ant{0};
    uint32_t num_ue_layer{0};
    uint32_t cell_num{0};
};

//! Validate that the geometry attributes saved in a cubb-side
//! `cubb_srs_buffers_SFN_<sfn>.<slot>.h5` dump match the expected values
//! derived from the cuMAC TV-generation YAML config. The cubb dump and
//! cuMAC pools must agree on every attribute exactly; any divergence
//! means realBuffIdx-based addressing would land at the wrong offset
//! (silent data corruption rather than a kernel error). On any mismatch
//! this logs every diverging field and returns -1 so callers can print
//! a full expected-vs-actual table from `*out_attrs` and exit before any
//! pool bytes are touched. `out_attrs` is populated whenever the file
//! was readable, regardless of whether validation passed.
//!
//! Attributes checked: gpu_pool_len, gpu_buf_size, num_prg, num_gnb_ant,
//! num_ue_layer, cell_num.
inline int validate_cubb_tv_attrs(const std::string& path,
                                  uint32_t           expected_gpu_pool_len,
                                  uint32_t           expected_gpu_buf_size,
                                  uint32_t           expected_num_prg,
                                  uint32_t           expected_num_gnb_ant,
                                  uint32_t           expected_num_ue_layer,
                                  uint32_t           expected_cell_num,
                                  CubbTvAttrs*       out_attrs = nullptr)
{
    CubbTvAttrs attrs{};
    try {
        H5::H5File file(path, H5F_ACC_RDONLY);

        auto read_u32 = [&](const char* name, uint32_t& out) {
            file.openAttribute(name).read(H5::PredType::NATIVE_UINT32, &out);
        };

        read_u32("gpu_pool_len",  attrs.gpu_pool_len);
        read_u32("gpu_buf_size",  attrs.gpu_buf_size);
        read_u32("num_prg",       attrs.num_prg);
        read_u32("num_gnb_ant",   attrs.num_gnb_ant);
        read_u32("num_ue_layer",  attrs.num_ue_layer);
        read_u32("cell_num",      attrs.cell_num);
        if (out_attrs) *out_attrs = attrs;

        bool ok = true;
        auto check = [&](const char* name, uint32_t got, uint32_t want) {
            if (got != want) {
                NVLOGE(MU_TEST_TAG, AERIAL_NVIPC_API_EVENT,
                       "validate_cubb_tv_attrs: %s mismatch in %s "
                       "(cubb=%u, cuMAC YAML=%u)",
                       name, path.c_str(), got, want);
                ok = false;
            }
        };
        check("gpu_pool_len",  attrs.gpu_pool_len,  expected_gpu_pool_len);
        check("gpu_buf_size",  attrs.gpu_buf_size,  expected_gpu_buf_size);
        check("num_prg",       attrs.num_prg,       expected_num_prg);
        check("num_gnb_ant",   attrs.num_gnb_ant,   expected_num_gnb_ant);
        check("num_ue_layer",  attrs.num_ue_layer,  expected_num_ue_layer);
        check("cell_num",      attrs.cell_num,      expected_cell_num);
        if (!ok) {
            return -1;
        }

        NVLOGC(MU_TEST_TAG,
               "validate_cubb_tv_attrs: %s matches YAML "
               "(gpu_pool_len=%u, gpu_buf_size=%u, num_prg=%u, "
               "num_gnb_ant=%u, num_ue_layer=%u, cell_num=%u)",
               path.c_str(), attrs.gpu_pool_len, attrs.gpu_buf_size,
               attrs.num_prg, attrs.num_gnb_ant, attrs.num_ue_layer, attrs.cell_num);
        return 0;
    } catch (const H5::Exception& e) {
        NVLOGE(MU_TEST_TAG, AERIAL_NVIPC_API_EVENT,
               "validate_cubb_tv_attrs: H5 exception reading %s: %s",
               path.c_str(), e.getCDetailMsg());
        if (out_attrs) *out_attrs = attrs;
        return -1;
    }
}

//! Overwrite the three shared SRS pools with the contents of one cuphydriver
//! `cubb_srs_buffers_<dump_idx>_SFN_<sfn>.<slot>.h5` dump file (produced by
//! `SrsIpcManager::dump_h5` when DUMP_SRS_SLOT_NUM>0).
//!
//! Per dataset:
//!   - `ipc_gpu_pool`   -> H2D copy into `gpu_pool->get_buf_addr(0)`
//!   - `chest_buf_pool` -> memcpy into `chest_pool->get_buf_addr(0)`
//!   - `srs_info_pool`  -> memcpy into `msg_pool->get_buf_addr(0)`
//!
//! Sizes must match the L1-side pool configuration exactly; on mismatch the
//! function logs an error and returns -1 without touching any pool. Returns 0
//! on success.
inline int load_cubb_pools_from_h5(
    const std::string& path,
    nv::lock_free_mem_pool<uint8_t>* gpu_pool,
    nv::lock_free_mem_pool<CVSrsChestBuff>* chest_pool,
    nv::lock_free_mem_pool<SrsInfoUpdate>* msg_pool)
{
    if (gpu_pool == nullptr || chest_pool == nullptr || msg_pool == nullptr) {
        NVLOGE(MU_TEST_TAG, AERIAL_NVIPC_API_EVENT,
               "load_cubb_pools_from_h5: null pool pointer (path=%s)", path.c_str());
        return -1;
    }

    const size_t gpu_pool_bytes   = static_cast<size_t>(gpu_pool->get_pool_len())
                                    * static_cast<size_t>(gpu_pool->get_buf_size());
    const size_t chest_pool_bytes = static_cast<size_t>(chest_pool->get_pool_len())
                                    * sizeof(CVSrsChestBuff);
    const size_t msg_pool_bytes   = static_cast<size_t>(msg_pool->get_pool_len())
                                    * sizeof(SrsInfoUpdate);

    void* gpu_base   = gpu_pool->get_buf_addr(0);
    void* chest_base = chest_pool->get_buf_addr(0);
    void* msg_base   = msg_pool->get_buf_addr(0);
    if (gpu_base == nullptr || chest_base == nullptr || msg_base == nullptr) {
        NVLOGE(MU_TEST_TAG, AERIAL_NVIPC_API_EVENT,
               "load_cubb_pools_from_h5: null pool base addr (gpu=%p chest=%p msg=%p)",
               gpu_base, chest_base, msg_base);
        return -1;
    }

    try {
        H5::H5File file(path, H5F_ACC_RDONLY);

        auto check_dataset_size = [&](const char* ds_name, size_t expected) -> int {
            H5::DataSet ds = file.openDataSet(ds_name);
            hsize_t dim = 0;
            ds.getSpace().getSimpleExtentDims(&dim, nullptr);
            if (static_cast<size_t>(dim) != expected) {
                NVLOGE(MU_TEST_TAG, AERIAL_NVIPC_API_EVENT,
                       "load_cubb_pools_from_h5: %s size mismatch in %s "
                       "(file=%llu pool=%zu)",
                       ds_name, path.c_str(),
                       static_cast<unsigned long long>(dim), expected);
                return -1;
            }
            return 0;
        };

        if (check_dataset_size("ipc_gpu_pool",   gpu_pool_bytes)   != 0) return -1;
        if (check_dataset_size("chest_buf_pool", chest_pool_bytes) != 0) return -1;
        if (check_dataset_size("srs_info_pool",  msg_pool_bytes)   != 0) return -1;

        if (gpu_pool_bytes > 0) {
            std::vector<uint8_t> host_buf(gpu_pool_bytes);
            file.openDataSet("ipc_gpu_pool").read(host_buf.data(), H5::PredType::NATIVE_UINT8);
            cudaError_t cerr = cudaMemcpy(gpu_base, host_buf.data(), gpu_pool_bytes,
                                          cudaMemcpyHostToDevice);
            if (cerr != cudaSuccess) {
                NVLOGE(MU_TEST_TAG, AERIAL_CUDA_API_EVENT,
                       "load_cubb_pools_from_h5: cudaMemcpy H2D failed (%zu B): %s",
                       gpu_pool_bytes, cudaGetErrorString(cerr));
                return -1;
            }
        }

        if (chest_pool_bytes > 0) {
            // Read into a temp byte buffer; do NOT blast raw bytes onto live pool
            // objects because each CVSrsChestBuff embeds an
            // ipc_dev_buf unique_ptr and a cuphy::tensor_desc that hold live
            // IPC-backed heap pointers for this process.  Overwriting those with
            // cuBB-capture-process pointers corrupts L1's tensor-descriptor state.
            // Copy only the plain scalar data fields via copyDataFrom().
            std::vector<uint8_t> chest_tmp(chest_pool_bytes);
            file.openDataSet("chest_buf_pool").read(chest_tmp.data(), H5::PredType::NATIVE_UINT8);

            const uint32_t n_entries = chest_pool->get_pool_len();
            for (uint32_t i = 0; i < n_entries; ++i) {
                CVSrsChestBuff* live = chest_pool->get_buf_addr(i);
                const CVSrsChestBuff* src =
                    reinterpret_cast<const CVSrsChestBuff*>(
                        chest_tmp.data() + i * sizeof(CVSrsChestBuff));
                live->copyDataFrom(*src);
            }
        }

        if (msg_pool_bytes > 0) {
            file.openDataSet("srs_info_pool")
                .read(msg_base, H5::PredType::NATIVE_UINT8);
        }
    } catch (const H5::Exception& e) {
        NVLOGE(MU_TEST_TAG, AERIAL_NVIPC_API_EVENT,
               "load_cubb_pools_from_h5: H5 exception reading %s: %s",
               path.c_str(), e.getCDetailMsg());
        return -1;
    }

    NVLOGC(MU_TEST_TAG,
           "load_cubb_pools_from_h5: loaded %s (gpu=%zu B, chest=%zu B, info=%zu B)",
           path.c_str(), gpu_pool_bytes, chest_pool_bytes, msg_pool_bytes);
    return 0;
}


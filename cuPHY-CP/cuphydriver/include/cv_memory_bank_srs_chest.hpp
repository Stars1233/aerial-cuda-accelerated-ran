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

#ifndef CV_MEMORY_BANK_SRS_CHEST_H
#define CV_MEMORY_BANK_SRS_CHEST_H

#include <unordered_map>
#include <queue>
#include <iostream>
#include <typeinfo>
#include <atomic>
#include "gpudevice.hpp"
#include "constant.hpp"
#include "cuphydriver_api.hpp"
#include "fh.hpp"
#include "phychannel.hpp"


#define CV_NUM_UE_LAYER MAX_UE_SRS_ANT_PORTS           ///< Number of UE layers for SRS channel estimates
#define CV_NUM_GNB_ANT MAX_AP_PER_SLOT_SRS            ///< Number of gNodeB antennas for SRS processing per slot
#define CV_NUM_PRBG ORAN_MAX_PRB                      ///< Number of Physical Resource Block Groups for channel estimates
#define CV_INVALID_RNTI 65535                         ///< Invalid RNTI value
#define CV_INVALID_CESHT_BUF_INDEX 65535              ///< Invalid channel estimate buffer index

/**
 * @brief SRS Channel Estimate Buffer
 * 
 * Stores channel estimates from Sounding Reference Signals (SRS) for a single UE.
 * Each buffer contains the frequency-domain channel response across all gNodeB antennas
 * and UE antenna ports, organized by Physical Resource Block Groups (PRBs).
 */
typedef struct _CVSrsChestBuff
{
    public:
        /**
         * @brief Construct SRS channel estimate buffer
         * 
         * @param bdev  GPU device buffer pointer
         */
        _CVSrsChestBuff(dev_buf* bdev) :
            srs_chest_buff_state(slot_command_api::SRS_CHEST_BUFF_NONE),
            rnti(CV_INVALID_RNTI),
            buffer_idx(CV_INVALID_CESHT_BUF_INDEX),
            cell_id(0),
            srs_chest_buff_usage(0),
            sfn(0xFFFF),
            slot(0xFFFF),
            srsPrgSize(0),
            srsStartPrg(0),
            srsStartValidPrg(0),
            srsNValidPrg(0)
        {
            buffer.reset(bdev);
            buffer->clear();
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
        uint8_t * getAddr() const { return buffer->addr();};
        
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
        slot_command_api::srsChestBuffState srs_chest_buff_state;  ///< Buffer state (INIT, REQUESTED, READY, NONE)
        std::unique_ptr<dev_buf> buffer;                           ///< GPU device memory buffer for channel estimates
        cuphy::tensor_desc buffDesc;                               ///< cuPHY tensor descriptor for this buffer
        uint32_t rnti;                                             ///< Radio Network Temporary Identifier (UE ID)
        uint32_t buffer_idx;                                       ///< Buffer index in memory pool
        uint32_t cell_id;                                          ///< Cell ID this buffer belongs to
        uint32_t srs_chest_buff_usage;                             ///< Buffer reference/usage counter
        uint16_t sfn;                                              ///< System Frame Number (0-1023) when channel estimate was captured
        uint16_t slot;                                             ///< Slot number within frame when channel estimate was captured
        uint8_t  srsPrgSize;                                       ///< Size of each PRB group
        uint16_t srsStartPrg;                                      ///< Starting PRB group index for SRS allocation
        uint16_t srsStartValidPrg;                                 ///< First valid PRB group index
        uint16_t srsNValidPrg;                                     ///< Number of valid PRB groups
} CVSrsChestBuff;

/**
 * @brief Per-cell buffer index mapping for SRS channel estimate memory allocation
 * 
 * Maps a cell ID to its allocated pool of SRS buffer indices. Each cell can request
 * a specific number of buffers from the global pool for its exclusive use.
 */
typedef struct _cellIdtoSrsBuffIndexMap
{
    uint32_t                requestedBy;   ///< FAPI message type that requested this allocation (SCF_FAPI_CONFIG_REQUEST, CV_MEM_BANK_CONFIG_REQUEST, or SCF_FAPI_START_REQUEST)
    uint32_t                mempoolSize;   ///< Number of buffers allocated to this cell
    std::vector<uint32_t>   indexMap;      ///< Vector of buffer indices allocated from global pool
}CellIdtoSrsBuffIndexMap;

/**
 * @brief SRS Channel Estimate Memory Bank Manager
 * 
 * Manages a global pool of SRS channel estimate buffers across all cells and UEs.
 * Provides allocation, retrieval, and state management for channel estimate storage.
 * Supports per-cell memory pool partitioning for isolation and resource management.
 */
class CvSrsChestMemoryBank 
{
    public:
        /**
         * @brief Construct SRS channel estimate memory bank
         * 
         * @param _pdh                          Physical layer driver handle
         * @param _gDev                         GPU device pointer
         * @param _total_num_srs_chest_buffers  Total number of buffers in global pool
         */
        CvSrsChestMemoryBank(phydriver_handle _pdh, GpuDevice* _gDev, uint32_t _total_num_srs_chest_buffers);
        
        /**
         * @brief Destructor - releases all buffers
         */
        ~CvSrsChestMemoryBank();

        /**
         * @brief Pre-allocate a buffer for future use
         * 
         * @param cell_id     Cell ID requesting the buffer
         * @param rnti        UE RNTI (Radio Network Temporary Identifier)
         * @param buffer_idx  Desired buffer index
         * @param reportType  Report type (periodic/aperiodic/etc.)
         * @param ptr         Output pointer to allocated buffer
         * @return int        0 on success, negative on failure
         */
        int preAllocateBuffer(uint32_t cell_id, uint32_t rnti, uint16_t buffer_idx, uint32_t reportType, CVSrsChestBuff** ptr);
        
        /**
         * @brief Retrieve an existing buffer
         * 
         * @param cell_id     Cell ID
         * @param rnti        UE RNTI
         * @param buffer_idx  Buffer index to retrieve
         * @param reportType  Report type
         * @param ptr         Output pointer to retrieved buffer
         * @return int        0 on success, negative on failure
         */
        int retrieveBuffer(uint32_t cell_id, uint32_t rnti, uint16_t buffer_idx, uint32_t reportType, CVSrsChestBuff** ptr);
        
        /**
         * @brief Update buffer state
         * 
         * @param cell_id               Cell ID
         * @param buffer_idx            Buffer index
         * @param srs_chest_buff_state  New state to set
         */
        void updateSrsChestBufferState(uint32_t cell_id, uint16_t buffer_idx, slot_command_api::srsChestBuffState srs_chest_buff_state);
        
        /**
         * @brief Get buffer state
         * 
         * @param cell_id     Cell ID
         * @param buffer_idx  Buffer index
         * @return slot_command_api::srsChestBuffState  Current buffer state
         */
        slot_command_api::srsChestBuffState getSrsChestBufferState(uint32_t cell_id, uint16_t buffer_idx);
        
        /**
         * @brief Update buffer usage counter
         * 
         * @param cell_id     Cell ID
         * @param rnti        UE RNTI
         * @param buffer_idx  Buffer index
         * @param usage       New usage count
         */
        void updateSrsChestBufferUsage(uint32_t cell_id, uint32_t rnti, uint16_t buffer_idx, uint32_t usage);
        
        /**
         * @brief Get buffer usage counter
         * 
         * @param cell_id     Cell ID
         * @param rnti        UE RNTI
         * @param buffer_idx  Buffer index
         * @return uint32_t   Current usage count
         */
        uint32_t getSrsChestBufferUsage(uint32_t cell_id, uint32_t rnti, uint16_t buffer_idx);
        
        /**
         * @brief Allocate a memory pool partition for a specific cell
         * 
         * @param requestedBy  FAPI message type requesting allocation (SCF_FAPI_CONFIG_REQUEST=0x02, CV_MEM_BANK_CONFIG_REQUEST=0x92, or SCF_FAPI_START_REQUEST=0x04)
         * @param cell_id      Cell ID to allocate pool for
         * @param mempoolSize  Number of buffers to allocate to this cell
         * @return bool        true on success, false if insufficient buffers available or invalid request source
         */
        bool memPoolAllocatePerCell(uint32_t requestedBy, uint16_t cell_id, uint32_t mempoolSize);
        
        /**
         * @brief Deallocate a cell's memory pool partition
         * 
         * @param cell_id  Cell ID to deallocate
         * @return bool    true on success, false on failure
         */
        bool memPoolDeAllocatePerCell(uint16_t cell_id);
        
        MemFoot             mf;  ///< Memory footprint tracking for this memory bank

    private:
        phydriver_handle                                                        pdh;                     ///< Physical layer driver handle
        GpuDevice*                                                              gDev;                    ///< GPU device pointer for memory allocation
        uint32_t                                                                total_num_srs_chest_buffers;  ///< Total number of buffers in global pool
        std::array<CVSrsChestBuff *, slot_command_api::MAX_SRS_CHEST_BUFFERS>   arr_cv_srs_chest_buff;  ///< Array of all SRS channel estimate buffer pointers
        std::queue<uint32_t>                                                    memIndexPool;            ///< Queue of free buffer indices available for allocation
        std::unordered_map<uint32_t, CellIdtoSrsBuffIndexMap>                   srsChEstBuffIndexMap;    ///< Map from cell ID to its allocated buffer index pool
};
#endif

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

#define TAG (NVLOG_TAG_BASE_CUPHY_DRIVER + 32) // "DRV.CV_MEM_BNK"

#include "cv_memory_bank_srs_chest.hpp"
#include "context.hpp"
#include "nvlog.hpp"
#include "exceptions.hpp"

//////////////////////////////////////////////////////////////////////////////////////////////////////////////
//// Channel Vector Memory Bank
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
CvSrsChestMemoryBank::CvSrsChestMemoryBank(phydriver_handle _pdh, GpuDevice* _gDev, uint32_t _total_num_srs_chest_buffers):
    pdh(_pdh),
    gDev(_gDev)
{
    PhyDriverCtx * pdctx = StaticConversion<PhyDriverCtx>(pdh).get();
    FhProxy * fhproxy = pdctx->getFhProxy();

    total_num_srs_chest_buffers = std::min(_total_num_srs_chest_buffers, slot_command_api::MAX_SRS_CHEST_BUFFERS);

    mf.init(_pdh, std::string("CvSrsChestMemoryBank"), sizeof(CVSrsChestBuff));

    gDev->setDevice();

    // Allocate memory for CV buffers
    //Each CV buffer is a 3-dim buffer of fp32 with dimensions - (nPrbG * nGnbAnt * nUeLayers)
    uint32_t  size_of_each_buffer = CV_NUM_PRBG * CV_NUM_GNB_ANT * CV_NUM_UE_LAYER * sizeof(uint32_t);
    for(uint32_t idx = 0; idx < total_num_srs_chest_buffers ; idx++)
    {
        dev_buf* buffer_dev = new dev_buf(size_of_each_buffer, gDev);
        mf.addGpuRegularSize(buffer_dev->size_alloc);
        CVSrsChestBuff* buffer = new CVSrsChestBuff(buffer_dev);
        buffer->setSrsChestBuffState(slot_command_api::SRS_CHEST_BUFF_NONE);
        arr_cv_srs_chest_buff[idx] = buffer;
        memIndexPool.push(idx);
        NVLOGD_FMT(TAG,"realIdx={} buffer={}", idx, static_cast<void *>(buffer));
    }
    NVLOGI_FMT(TAG, "CvSrsChestMemoryBank for {} free CV buffers created srsChEstBuffIndexMap size={}", total_num_srs_chest_buffers, srsChEstBuffIndexMap.size());
}

CvSrsChestMemoryBank::~CvSrsChestMemoryBank()
{
    for(int i = 0; i < total_num_srs_chest_buffers ; i++)
    {
        delete arr_cv_srs_chest_buff[i];
    }
}

int CvSrsChestMemoryBank::preAllocateBuffer(uint32_t cell_id, uint32_t rnti, uint16_t buffer_idx, uint32_t usage, CVSrsChestBuff** ptr)
{
    if((ptr == nullptr) || (usage == 0) || (rnti >= CV_INVALID_RNTI))
    {
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "Invalid input argument in CvSrsChestMemoryBank::{} rnti={} usage={}", __func__, rnti, usage);
        return -1;
    }

    if(srsChEstBuffIndexMap.find(cell_id) == srsChEstBuffIndexMap.end())
    {
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "Cell Id {} doesn't exist !!", cell_id);
        return -1;
    }

    if(buffer_idx >= srsChEstBuffIndexMap[cell_id].mempoolSize)
    {
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "Invalid buffer_idx {} for cellId {} with mempoolsize {} rnti={} ", buffer_idx, cell_id, srsChEstBuffIndexMap[cell_id].mempoolSize, rnti);
        return -1;
    }

    uint32_t realBuffIndex = srsChEstBuffIndexMap[cell_id].indexMap[buffer_idx];

    NVLOGD_FMT(TAG, "preAllocateBuffer: cell_id {} rnti {} FAPI buffer_idx {} realBuffIndex {}", cell_id, rnti, buffer_idx, realBuffIndex);

    *ptr = arr_cv_srs_chest_buff[realBuffIndex];

    int retVal = 0;

    CVSrsChestBuff* buffer = *ptr;
    uint8_t currSrsChestBuffState = buffer->getSrsChestBuffState();
    if((currSrsChestBuffState != slot_command_api::SRS_CHEST_BUFF_NONE) && (currSrsChestBuffState != slot_command_api::SRS_CHEST_BUFF_READY))
    {
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "SRS Chest Buffer in Use CvSrsChestMemoryBank::{} cell_id={} rnti={} usage={} buffer_idx={} SrsChestBuffState={}",
                        __func__, cell_id, rnti, usage, buffer_idx, currSrsChestBuffState);
        return -1;
    }
    buffer->init(rnti, buffer_idx, cell_id, usage);
    currSrsChestBuffState = buffer->getSrsChestBuffState();
    NVLOGD_FMT(TAG, "{} SRS Chest Buffer Pointer = {}, currSrsChestBuffState = {}", __func__, static_cast<void *>(arr_cv_srs_chest_buff[realBuffIndex]), currSrsChestBuffState);
    return retVal;
}

int CvSrsChestMemoryBank::retrieveBuffer(uint32_t cell_id, uint32_t rnti, uint16_t buffer_idx, uint32_t reportType, CVSrsChestBuff** ptr)
{
    if((ptr == nullptr) || (reportType > 1) || (rnti >= CV_INVALID_RNTI))
    {
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "Invalid input argument in CvSrsChestMemoryBank::allocateBuffer");
        return -1;
    }

    if(srsChEstBuffIndexMap.find(cell_id) == srsChEstBuffIndexMap.end())
    {
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "Cell Id {} doesn't exist !!", cell_id);
        return -1;
    }

    //Range check
    if(buffer_idx >= srsChEstBuffIndexMap[cell_id].mempoolSize)
    {
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "Invalid buffer_idx {} for cellId {} with mempoolsize {} rnti={} ", buffer_idx, cell_id, srsChEstBuffIndexMap[cell_id].mempoolSize, rnti);
        return -1;
    }

    uint32_t realBuffIndex = srsChEstBuffIndexMap[cell_id].indexMap[buffer_idx];

    NVLOGD_FMT(TAG, "retrieveBuffer: cell_id {} rnti {} FAPI buffer_idx {} realBuffIndex {}", cell_id, rnti, buffer_idx, realBuffIndex);

    *ptr = arr_cv_srs_chest_buff[realBuffIndex];

    NVLOGD_FMT(TAG, "{} SRS Chest Buffer Pointer = {}", __func__, static_cast<void *>(arr_cv_srs_chest_buff[realBuffIndex]));
    
    CVSrsChestBuff* buffer = *ptr;
    uint8_t currSrsChestBuffState = buffer->getSrsChestBuffState();

    if(currSrsChestBuffState != slot_command_api::SRS_CHEST_BUFF_READY)
    {
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "SRS Chest Buffer in free or SRS Chest not ready CvSrsChestMemoryBank::{} cell_id={} rnti={} reportType={} SrsChestBuffState={} buffer_idx={}",
                        __func__, cell_id, rnti, reportType, currSrsChestBuffState, buffer_idx);
        return -1;
    }

    if(*ptr == NULL)
        return -1;
    else
        return 0;
}

void CvSrsChestMemoryBank::updateSrsChestBufferState(uint32_t cell_id, uint16_t buffer_idx, slot_command_api::srsChestBuffState srs_chest_buff_state)
{
    if(srsChEstBuffIndexMap.find(cell_id) == srsChEstBuffIndexMap.end())
    {
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "Cell Id {} doesn't exist !!", cell_id);
        return;
    }

    //Range check
    if(buffer_idx >= srsChEstBuffIndexMap[cell_id].mempoolSize)
    {
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "updateSrsChestBufferState: Invalid buffer_idx {} for cellId {} with mempoolsize {} ", buffer_idx, cell_id, srsChEstBuffIndexMap[cell_id].mempoolSize);
        return;
    }

    uint32_t realBuffIndex = srsChEstBuffIndexMap[cell_id].indexMap[buffer_idx];

    CVSrsChestBuff* buffer = arr_cv_srs_chest_buff[realBuffIndex];
    buffer->setSrsChestBuffState(srs_chest_buff_state);
    
}

slot_command_api::srsChestBuffState CvSrsChestMemoryBank::getSrsChestBufferState(uint32_t cell_id, uint16_t buffer_idx)
{
    if(srsChEstBuffIndexMap.find(cell_id) == srsChEstBuffIndexMap.end())
    {
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "Cell Id {} doesn't exist !!", cell_id);
        return slot_command_api::SRS_CHEST_BUFF_NONE;
    }

    //Range check
    if(buffer_idx >= srsChEstBuffIndexMap[cell_id].mempoolSize)
    {
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "getSrsChestBufferState: Invalid buffer_idx {} for cellId {} with mempoolsize {} ", buffer_idx, cell_id, srsChEstBuffIndexMap[cell_id].mempoolSize);
        return slot_command_api::SRS_CHEST_BUFF_NONE;
    }

    uint32_t realBuffIndex = srsChEstBuffIndexMap[cell_id].indexMap.at(buffer_idx);

    CVSrsChestBuff* buffer = arr_cv_srs_chest_buff[realBuffIndex];
    return (buffer->getSrsChestBuffState());
}

uint32_t CvSrsChestMemoryBank::getSrsChestBufferUsage(uint32_t cell_id, uint32_t rnti, uint16_t buffer_idx)
{
    if(srsChEstBuffIndexMap.find(cell_id) == srsChEstBuffIndexMap.end())
    {
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "Cell Id {} doesn't exist !!", cell_id);
        return slot_command_api::SRS_CHEST_BUFF_NONE;
    }

    //Range check
    if(buffer_idx >= srsChEstBuffIndexMap[cell_id].mempoolSize)
    {
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "getSrsChestBufferUsage: Invalid buffer_idx {} for cellId {} with mempoolsize {} rnti={} ", buffer_idx, cell_id, srsChEstBuffIndexMap[cell_id].mempoolSize, rnti);
        return slot_command_api::SRS_CHEST_BUFF_NONE;
    }

    uint32_t realBuffIndex = srsChEstBuffIndexMap[cell_id].indexMap[buffer_idx];

    CVSrsChestBuff* buffer = arr_cv_srs_chest_buff[realBuffIndex];
    return (buffer->getSrsChestBuffUsage());
}

void CvSrsChestMemoryBank::updateSrsChestBufferUsage(uint32_t cell_id, uint32_t rnti, uint16_t buffer_idx, uint32_t usage)
{
    if(srsChEstBuffIndexMap.find(cell_id) == srsChEstBuffIndexMap.end())
    {
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "Cell Id {} doesn't exist !!", cell_id);
        return;
    }

    //Range check
    if(buffer_idx >= srsChEstBuffIndexMap[cell_id].mempoolSize)
    {
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "updateSrsChestBufferUsage: Invalid buffer_idx {} for cellId {} with mempoolsize {} rnti={} ", buffer_idx, cell_id, srsChEstBuffIndexMap[cell_id].mempoolSize, rnti);
        return;
    }

    uint32_t realBuffIndex = srsChEstBuffIndexMap[cell_id].indexMap[buffer_idx];

    CVSrsChestBuff* buffer = arr_cv_srs_chest_buff[realBuffIndex];
    buffer->setSrsChestBuffUsage(usage);
}

bool CvSrsChestMemoryBank::memPoolAllocatePerCell(uint32_t requestedBy, uint16_t cell_id, uint32_t mempoolSize)
{
    bool retVal = true;
    if((mempoolSize > memIndexPool.size()) || 
       ((requestedBy != SCF_FAPI_CONFIG_REQUEST) && (requestedBy != CV_MEM_BANK_CONFIG_REQUEST) && (requestedBy != SCF_FAPI_START_REQUEST)))
    {
        if(srsChEstBuffIndexMap.find(cell_id) != srsChEstBuffIndexMap.end())
        {
            retVal = true;
            NVLOGI_FMT(TAG, "Cell id {} already exists in srsChEstBuffIndexMap", cell_id);
        }
        else
        {
            retVal = false;
            NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "Cell id {} unable to config mempool for mempoolSize {} memIndexPool.size {} requestedBy {}", cell_id, mempoolSize, memIndexPool.size(), requestedBy);
        }
    }
    else
    {
        if(srsChEstBuffIndexMap.find(cell_id) != srsChEstBuffIndexMap.end())
        {
            /* Cell id already exists */
            switch(requestedBy)
            {
                case SCF_FAPI_CONFIG_REQUEST:
                    {
                        switch(srsChEstBuffIndexMap[cell_id].requestedBy)
                        {
                            case SCF_FAPI_CONFIG_REQUEST:
                                retVal = false;
                                NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "Cell id {} already exists in srsChEstBuffIndexMap", cell_id);
                                break;
                            case CV_MEM_BANK_CONFIG_REQUEST:
                                NVLOGD_FMT(TAG, "Cell id {} already exists in srsChEstBuffIndexMap..ignoring it", cell_id);
                                break;
                            default:
                                retVal = false;
                                NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "Unsupported condition requestedBy {} stored {}!!!", requestedBy,srsChEstBuffIndexMap[cell_id].requestedBy);
                                break;
                        }
                    }
                    break;
                case SCF_FAPI_START_REQUEST:
                    {
                        switch(srsChEstBuffIndexMap[cell_id].requestedBy)
                        {
                            case SCF_FAPI_CONFIG_REQUEST:
                            case CV_MEM_BANK_CONFIG_REQUEST:
                                NVLOGD_FMT(TAG, "Cell id {} already exists in srsChEstBuffIndexMap..ignoring it", cell_id);
                                break;
                            default:
                                retVal = false;
                                NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "Unsupported condition requestedBy {} stored {}!!!", requestedBy,srsChEstBuffIndexMap[cell_id].requestedBy);
                                break;
                        }
                    }
                    break;
                case CV_MEM_BANK_CONFIG_REQUEST:
                    {
                        switch(srsChEstBuffIndexMap[cell_id].requestedBy)
                        {
                            case SCF_FAPI_CONFIG_REQUEST:
                            case CV_MEM_BANK_CONFIG_REQUEST:
                                NVLOGD_FMT(TAG, "Cell id {} already exists in srsChEstBuffIndexMap..ignoring it", cell_id);
                                break;
                            default:
                                retVal = false;
                                NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "Unsupported condition requestedBy {} stored {}!!!", requestedBy,srsChEstBuffIndexMap[cell_id].requestedBy);
                                break;
                        }
                    }
                    break;
                default:
                    retVal = false;
                    NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "Cell id {} already exists in srsChEstBuffIndexMap and unsupported api {} reuested it", cell_id, requestedBy);
                    break;
            }
        }
        else
        {
            MemtraceDisableScope md;
            srsChEstBuffIndexMap[cell_id].requestedBy = requestedBy;
            srsChEstBuffIndexMap[cell_id].mempoolSize = mempoolSize;
            srsChEstBuffIndexMap[cell_id].indexMap.resize(mempoolSize);
            for(uint32_t idx = 0; idx < mempoolSize; idx++)
            {
                uint32_t realBuffIndex = memIndexPool.front();
                srsChEstBuffIndexMap[cell_id].indexMap[idx] = realBuffIndex;
                NVLOGD_FMT(TAG, "cell_id={}, idx={}, realBuffIndex={}", cell_id, idx, realBuffIndex);
                memIndexPool.pop();
            }
            NVLOGD_FMT(TAG, "memPoolAllocatePerCell: cell_id {} mempoolSize={}", cell_id, mempoolSize);
        }
    }
    return retVal;
}

bool CvSrsChestMemoryBank::memPoolDeAllocatePerCell(uint16_t cell_id)
{
    bool retVal = true;
    if(srsChEstBuffIndexMap.find(cell_id) != srsChEstBuffIndexMap.end())
    {
        MemtraceDisableScope md;
        uint32_t mempoolSize = srsChEstBuffIndexMap[cell_id].mempoolSize;
        for(uint32_t idx = 0; idx < mempoolSize; idx++)
        {
            uint32_t realBuffIndex = srsChEstBuffIndexMap[cell_id].indexMap[idx];
            memIndexPool.push(realBuffIndex);
        }
        NVLOGD_FMT(TAG, "memPoolDeAllocatePerCell: cell_id {} mempoolSize={}", cell_id, mempoolSize);
        srsChEstBuffIndexMap.erase(cell_id);
    }
    else
    {
        retVal = false;
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "Cell id {} doesn't exist in srsChEstBuffIndexMap", cell_id);
    }
    return retVal;
}

void CVSrsChestBuff::getSrsPrgInfo(uint8_t* pSrsPrgSize_out, uint16_t* pSrsStartPrg_out, uint16_t* pSrsStartValidPrg_out, uint16_t* pSrsNValidPrg_out)
{
    *pSrsPrgSize_out    = srsPrgSize;
    *pSrsStartPrg_out   = srsStartPrg;
    *pSrsStartValidPrg_out = srsStartValidPrg;
    *pSrsNValidPrg_out     = srsNValidPrg;
}

void CVSrsChestBuff::setSrsChestBuffState(slot_command_api::srsChestBuffState  _srs_chest_buff_state)
{
    srs_chest_buff_state = _srs_chest_buff_state;
}

void CVSrsChestBuff::setSrsChestBuffUsage(uint32_t  _srs_chest_buff_usage)
{
    srs_chest_buff_usage = _srs_chest_buff_usage;
}

void CVSrsChestBuff::configSrsInfo(uint16_t nPrg, uint8_t nAnt, uint8_t nLayer, uint8_t srsPrgSize_in, uint16_t srsStartPrg_in, uint16_t startValidPrg_in, uint16_t nValidPrg_in)
{
    srsPrgSize          = srsPrgSize_in;
    srsStartPrg         = srsStartPrg_in;
    srsStartValidPrg       = startValidPrg_in;
    srsNValidPrg           = nValidPrg_in;
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

void CVSrsChestBuff::setSfnSlot(uint16_t _sfn, uint16_t _slot){
    sfn = _sfn;
    slot = _slot;
}

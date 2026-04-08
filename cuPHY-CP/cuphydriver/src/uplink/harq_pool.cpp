/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#define TAG (NVLOG_TAG_BASE_CUPHY_DRIVER + 18) // "DRV.HARQ_POOL"

#include "harq_pool.hpp"
#include "context.hpp"
#include "nvlog.hpp"
#include "exceptions.hpp"

//////////////////////////////////////////////////////////////////////////////////////////////////////////////
//// Pool of free HARQ buffers with fixed size
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
HarqPool::HarqPool(size_t _size,size_t _sizePool, GpuDevice* _gDev, FronthaulHandle _fhi_handle)
    :
    size(_size), sizePool(_sizePool), gpuMemSize(0), gDev(_gDev)
{
    char * ring_name;

    if(size == 0)
        PHYDRIVER_THROW_EXCEPTIONS(EINVAL, "HarqPool bad input size");

    gDev->setDevice();

    ring_name = (char*) calloc(256, sizeof(char));
    snprintf(ring_name, 256, "HARQ_POOL_%zd", size);
    ring_info.count = sizePool;
    ring_info.socket_id = AERIAL_SOCKET_ID_ANY;
    ring_info.multi_producer = true;
    ring_info.multi_consumer = true;

    ring_info.name = (const char *)ring_name;
    // Create (DPDK) lockless ring to maintain the pool
    // Assume sizePool is a power of 2
    if(aerial_fh::ring_create(_fhi_handle, &ring_info, &hb_ring))
        PHYDRIVER_THROW_EXCEPTIONS(EINVAL, "aerial_fh::ring_create error");

    totRingElems = (int)aerial_fh::ring_free_count(hb_ring);

    // Create HARQ set of free buffers
    for(int i = 0; i < sizePool-1; i++)
    {
        dev_buf* _dev_buf = new dev_buf(size, gDev);
        gpuMemSize += _dev_buf->size_alloc;
        hb_list.push_back(std::unique_ptr<HarqBuffer>(new HarqBuffer(_dev_buf, size)));

        // Push all the HARQ buffers in the ring
        if(aerial_fh::ring_enqueue(hb_ring, (void*)hb_list[i].get()))
        {
            NVLOGE_FMT(TAG, AERIAL_ORAN_FH_EVENT, "Error inserting obj {}", i);
            PHYDRIVER_THROW_EXCEPTIONS(EINVAL, "aerial_fh::ring_enqueue error");
        }
        // else
          //   NVLOGI_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "Free item in the ring {}", (int)aerial_fh::ring_free_count(hb_ring));
    }

    NVLOGI_FMT_EVT(TAG, AERIAL_CUPHYDRV_API_EVENT, "HarqPool created with {} elements of size {}", countElements(), size);
}

HarqPool::~HarqPool() {
    for(auto& i:hb_list) {
        i.reset(nullptr);
    }

    hb_list.clear();
    if(aerial_fh::ring_destroy(hb_ring))
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "Error destroying hb_ring");
}

size_t HarqPool::getSize() const {
    return size;
}

size_t HarqPool::countElements() const {
    return sizePool - ((int)aerial_fh::ring_free_count(hb_ring));
}

size_t HarqPool::getGpuMemSize() const {
    return gpuMemSize;
}

// Allocate a buffer from FH rte_ring
HarqBuffer * HarqPool::pullBuffer() {

    HarqBuffer * buf = nullptr;

    if(aerial_fh::ring_dequeue(hb_ring, (void**)&buf))
    {
        NVLOGE_FMT(TAG, AERIAL_ORAN_FH_EVENT, "aerial_fh::ring_dequeue");
        return nullptr;
    }

    if (buf == nullptr)
    {
        NVLOGE_FMT(TAG, AERIAL_ORAN_FH_EVENT, "pullBuffer: nullptr");
        return nullptr;
    }

    buf->setInUse();

#ifdef DEBUG_HARQ_POOL
    if (buf->getSize() != size)
    {
        NVLOGE_FMT(TAG, AERIAL_ORAN_FH_EVENT, "pullBuffer: Wrong HarqPool: size: {}-{} {}: rnti {} hpid{} cell_id {} TxCount={}",
                buf->getSize(), size, reinterpret_cast<void*>(buf), buf->getRnti(), buf->getHarqPid(), buf->getCellId(), buf->getTxCount());
    }

    if(buf->refAdd() != 0)
    {
        NVLOGE_FMT(TAG, AERIAL_ORAN_FH_EVENT, "Dequeued used buf {}: rnti {} hpid{} cell_id {} TxCount={}",
                reinterpret_cast<void*>(buf), buf->getRnti(), buf->getHarqPid(), buf->getCellId(), buf->getTxCount());
    }
#endif

    return buf;
}

// Release the buffer back to FH rte_ring
int HarqPool::pushBuffer(HarqBuffer * buf) {

    if(!buf)
        return EINVAL;

    if (buf->getSize() != size)
    {
        NVLOGE_FMT(TAG, AERIAL_ORAN_FH_EVENT, "Wrong HarqPool: size: {}-{} {}: rnti {} hpid{} cell_id {} TxCount={}",
                buf->getSize(), size, reinterpret_cast<void*>(buf), buf->getRnti(), buf->getHarqPid(), buf->getCellId(), buf->getTxCount());
    }

#ifdef DEBUG_HARQ_POOL
    int ref = buf->refSub();
    if (ref > 1)
    {
        NVLOGE_FMT(TAG, AERIAL_ORAN_FH_EVENT, "Double enqueue buf {}: rnti {} hpid{} cell_id {} TxCount={}",
                reinterpret_cast<void*>(buf), buf->getRnti(), buf->getHarqPid(), buf->getCellId(), buf->getTxCount());
        return 0;
    }
    else if (ref < 1)
    {
        NVLOGE_FMT(TAG, AERIAL_ORAN_FH_EVENT, "Double enqueue buf {}: SFN {}.{} rnti {} hpid{} cell_id {} TxCount={}",
                reinterpret_cast<void*>(buf), buf->getSfn(), buf->getSlot(), buf->getRnti(), buf->getHarqPid(), buf->getCellId(), buf->getTxCount());
        return 0;
    }
    NVLOGD_FMT(TAG, "clear buffer: {} SFN {}.{} rnti {} hPID {} ndi {} cell_id {}",
            reinterpret_cast<void*>(buf), buf->getSfn(), buf->getSlot(), buf->getRnti(), buf->getHarqPid(), buf->getCellId());
#endif

    buf->clear();

    if(aerial_fh::ring_enqueue(hb_ring, (void*)buf))
    {
        NVLOGE_FMT(TAG, AERIAL_ORAN_FH_EVENT, "aerial_fh::ring_enqueue {}", reinterpret_cast<void*>(buf));
        return -1;
    }

    return 0;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////
//// HARQ Bucket
//////////////////////////////////////////////////////////////////////////////////////////////////////////////

int HarqBucket::getBucketRnti() const {
    return rnti;
}

// Allocate a buffer, add to bucket_map
int HarqBucket::pushBuffer(HarqBuffer * hb_ptr, uint32_t rnti, uint32_t harq_pid, uint64_t cell_id) {
    bool found = false;
    int r_index = (rnti%256)/MAX_BUCKET_ENTRY;
    if(harq_pid >= MAX_HARQ_PID)
        return EINVAL;

    bucket_map[r_index][harq_pid].mlock.lock();

    for(int k=0; k<MAX_BUCKET_MAP_ENTRIES; k++)
    {
        if(bucket_map[r_index][harq_pid].hb_rnti_v[k] == nullptr)
        {
            bucket_map[r_index][harq_pid].hb_rnti_v[k] = hb_ptr;
            if(hb_ptr->init(rnti, harq_pid, cell_id))
            {
                NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "rnti {} harq_pid {} setBucketInfo error", rnti, harq_pid);
            }
            found = true;
            break;
        }
    }

    bucket_map[r_index][harq_pid].mlock.unlock();

    return found ? 0 : -1;
}

int HarqBucket::getBuffer(HarqBuffer ** hb_ptr, uint32_t rnti, uint32_t harq_pid, uint64_t cell_id) {
    int r_index = (rnti%256)/MAX_BUCKET_ENTRY;
    bool found = false;

    (*hb_ptr) = nullptr;

    if(harq_pid >= MAX_HARQ_PID)
    {
        NVLOGE_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "Invalid harq_pid: bucket {} rnti {} harq_pid {}", r_index, rnti, harq_pid);
        return EINVAL;
    }

    (*hb_ptr) = nullptr;

    NVLOGD_FMT(TAG, "Looking for HARQ buffer in bucket {} rnti {} harq_pid {}", r_index, rnti, harq_pid);

    bucket_map[r_index][harq_pid].mlock.lock();

    for(int k=0; k<MAX_BUCKET_MAP_ENTRIES; k++)
    {
        if(
            bucket_map[r_index][harq_pid].hb_rnti_v[k] != nullptr               &&
            bucket_map[r_index][harq_pid].hb_rnti_v[k]->getRnti() == rnti       &&
            bucket_map[r_index][harq_pid].hb_rnti_v[k]->getCellId() == cell_id
        )
        {
            (*hb_ptr) = bucket_map[r_index][harq_pid].hb_rnti_v[k];
            found = true;
        }

        if(found)
            break;
    }

    // auto it = std::find_if(
    //     bucket_map[r_index][harq_pid].hb_rnti_v.begin(), bucket_map[r_index][harq_pid].hb_rnti_v.end(),
    //     [&rnti, &cell_id](HarqBuffer * hb_ptr) {
    //                 return ((hb_ptr != nullptr) && (hb_ptr->getCellId() == cell_id) && (hb_ptr->getRnti() == rnti));
    //             }
    // );

    // if(it != bucket_map[r_index][harq_pid].hb_rnti_v.end()){
    //     (*hb_ptr) = (*it);
    // }

    bucket_map[r_index][harq_pid].mlock.unlock();

    return 0;
}

// Release a buffer, clean from bucket_map
int HarqBucket::pullBuffer(uint32_t rnti, uint32_t harq_pid, uint64_t cell_id) {
    int r_index = (rnti%256)/MAX_BUCKET_ENTRY;
    bool found = false;

    if(harq_pid >= MAX_HARQ_PID)
        return EINVAL;

    bucket_map[r_index][harq_pid].mlock.lock();

    for(int k=0; k<MAX_BUCKET_MAP_ENTRIES; k++)
    {
        if(
            bucket_map[r_index][harq_pid].hb_rnti_v[k] != nullptr            &&
            bucket_map[r_index][harq_pid].hb_rnti_v[k]->getRnti() == rnti    &&// >= ?
            bucket_map[r_index][harq_pid].hb_rnti_v[k]->getCellId() == cell_id
        )
        {
            bucket_map[r_index][harq_pid].hb_rnti_v[k] = nullptr;
            found = true;
            break;
        }
    }

    bucket_map[r_index][harq_pid].mlock.unlock();

    if(found)
        return 0;
    return -1;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////
//// HARQ Pool Manager
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
HarqPoolManager::HarqPoolManager(phydriver_handle _pdh, GpuDevice* _gDev) :
    pdh(_pdh),
    gDev(_gDev)
{
    PhyDriverCtx * pdctx = StaticConversion<PhyDriverCtx>(pdh).get();
    FhProxy * fhproxy = pdctx->getFhProxy();
    id = Time::nowNs().count();

    mf.init(_pdh, std::string("HarqPoolManager"), sizeof(HarqPoolManager));

    // Create HARQ buffer pools for each size
    for(int i = 0; i < MAX_HARQ_POOLS; i++)
    {
        HarqPool* hb_pool = new HarqPool(HARQ_POOL_SIZE[i], pdctx->getMaxHarqPools(), gDev, fhproxy->getFhInstance());
        mf.addGpuRegularSize(hb_pool->getGpuMemSize());
        hb_pool_list.push_back(std::unique_ptr<HarqPool>(hb_pool));
    }

    // Prepare MAX_BUCKET_ENTRY buckets
    for(int i = 0; i < MAX_BUCKET_ENTRY; i++)
    {
        hb_bucket_list.push_back(std::unique_ptr<HarqBucket>(new HarqBucket(i)));
        mf.addCpuRegularSize(sizeof(HarqBucket));
    }
}

HarqPoolManager::~HarqPoolManager()
{
    for(auto& i:hb_pool_list) {
        i.reset(nullptr);
    }
    hb_pool_list.clear();

    for(auto& i:hb_bucket_list) {
        i.reset(nullptr);
    }
    hb_bucket_list.clear();
}

// Allocate a HARQ pool based on size
HarqPool * HarqPoolManager::poolAlloc(size_t buf_size) {
    for(auto it = hb_pool_list.begin(); it != hb_pool_list.end(); ++it)
    {
        if((*it)->getSize() >= buf_size && (*it)->countElements() > 0)
        {
            return (*it).get();
        }
    }
    NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "HARQ buffer pool is empty. buf_size={}", (int)buf_size);
    return nullptr;
}

// Find the right HARQ pool based on size
HarqPool * HarqPoolManager::poolFind(size_t buf_size) {
    for(auto it = hb_pool_list.begin(); it != hb_pool_list.end(); ++it)
    {
        if((*it)->getSize() >= buf_size)
        {
            return (*it).get();
        }
    }
    NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "HARQ buffer pool is not found. buf_size={}", buf_size);
    return nullptr;
}

void HarqPoolManager::checkPoolDepletion(ReleasedHarqBufferInfo& released_harq_buffer_info)
{
    
    static uint32_t hystersis = 0;
    PhyDriverCtx * pdctx = StaticConversion<PhyDriverCtx>(pdh).get();

    if(hb_pool_list[MAX_HARQ_POOLS-1]->countElements()*100/(pdctx->getMaxHarqPools()) < 30)
    {
        NVLOGW_FMT(TAG, "Largest sized HARQ pool at less than 30 percent availability. Free all HARQ buffers");
        freeAllHarqBuffers(released_harq_buffer_info);
    }
    else
    {
        if( (hb_pool_list[0]->countElements()*100/pdctx->getMaxHarqPools() < 30) ||
            (hb_pool_list[1]->countElements()*100/pdctx->getMaxHarqPools() < 30) )
        {
            hystersis++;
            if(HARQ_CLEANUP_HYSTERSIS == hystersis)
            {
                NVLOGW_FMT(TAG, "HARQ pool[0/1] at less than 30 percent availability. Entering massive cleanup ");
                hystersis = 0;
                cleanupHarqBuckets(released_harq_buffer_info);
            }
            else
                NVLOGW_FMT(TAG, "HARQ pool[0/1] at less than 30 percent availability. Hystersis={}",hystersis);
        }
    }
}

int HarqPoolManager::bucketFindBuffer(HarqBuffer ** hb_ptr, uint32_t rnti, uint32_t harq_pid, uint64_t cell_id) {
    if(hb_bucket_list[rnti%MAX_BUCKET_ENTRY]->getBuffer(hb_ptr, rnti, harq_pid, cell_id))
    {
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "No HB found in bucket for rnti {} harq pid {}", rnti, harq_pid);
        return -1;
    }

    return 0;
}

int HarqPoolManager::bucketAllocateBuffer(HarqBuffer ** hb_ptr, size_t buf_size, uint32_t rnti, uint32_t harq_pid, uint8_t* ndi, uint64_t cell_id, bool is_bundled_pdu, uint8_t pusch_aggr_factor, uint16_t sfn, uint16_t slot) {
    HarqPool * hb_pool = nullptr;
    t_ns now = Time::nowNs();
    PhyDriverCtx * pdctx = StaticConversion<PhyDriverCtx>(pdh).get();
    if(buf_size == 0)
        return EINVAL;

    while(1) {
        if(hb_bucket_list[rnti%MAX_BUCKET_ENTRY]->getBuffer(hb_ptr, rnti, harq_pid,cell_id))
        {
            NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "getBuffer error size {}", buf_size);
            return -1;
        }

        // FIXME: in_use is not locked, caller need to make sure there's no multi-thread competition.
        if(
            (*hb_ptr) != nullptr                                                &&
            (*hb_ptr)->isInUse() == false                                       &&
            (
                (*hb_ptr)->getTxCount() >= (*hb_ptr)->getMaxTxCount()                    /* ||
                (now - (*hb_ptr)->getTimestampLastUsed()).count() > MAX_HARQB_NS_BUCKET*/
            )
        )
        {
            NVLOGD_FMT(TAG, "Release expired buffer: {} SFN {}.{} rnti {} hPID {} ndi {} cell_id {}",
                    reinterpret_cast<void*>(*hb_ptr), (*hb_ptr)->getSfn(), (*hb_ptr)->getSlot(), (*hb_ptr)->getRnti(), (*hb_ptr)->getHarqPid(),*ndi, (*hb_ptr)->getCellId());
            bucketReleaseBuffer((*hb_ptr));
            (*hb_ptr) = nullptr;
        }
        else
            break;
    }

    if((*hb_ptr) == nullptr)
    {
        //NVLOGD_FMT(TAG, "hb_bucket_list[{}]->getBuffer(rnti {}, harq_pid {}) ndi == {}", rnti%MAX_BUCKET_ENTRY, rnti, harq_pid, *ndi);
        // Buffer not found in the bucket, need a new buffer
        hb_pool = poolAlloc(buf_size);
        if(hb_pool == nullptr)
        {
            NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "hb_pool is null");
            return -1;
        }

        (*hb_ptr) = hb_pool->pullBuffer();
        if((*hb_ptr) == nullptr)
        {
            NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "pullBuffer error size {}", buf_size);
            return -1;
        }

        if(hb_bucket_list[rnti%MAX_BUCKET_ENTRY]->pushBuffer((*hb_ptr), rnti, harq_pid, cell_id))
        {
            NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "pushBuffer error size {}", buf_size);
            return -1;
        }

        if(*ndi == 0)
        {
            //PHY uses ndi to reset the HARQ buffer to 0s
            *ndi = 1;
            (*hb_ptr)->resetTxCount();

            NVLOGI_FMT_EVT(TAG, AERIAL_CUPHYDRV_API_EVENT, "new buffer resetting the txCount to 0 and NDI=1 for NDI == 0 , hqbuf rnti {} hpid {} cell_id {}",
                        rnti, harq_pid, cell_id);
        }
        (*hb_ptr)->refReset();
        if(is_bundled_pdu)
        {
            (*hb_ptr)->setMaxTxCount( pdctx->getMaxHarqTxCountBundled());
        }
        else
        {
            (*hb_ptr)->setMaxTxCount( pdctx->getMaxHarqTxCountNonBundled());
        }
        //Reset the number of TTIs that this HARQ buffer is bundled with and start fresh count in this bundling window.
        (*hb_ptr)->resetNumTtiBundled();
        (*hb_ptr)->incrementNumTtiBundled(1);
    }
    else
    {
        NVLOGD_FMT(TAG, "hb_bucket_list[{}] found (rnti {}, harq_pid {}) ndi == {}, tx_count = {}",
            rnti%MAX_BUCKET_ENTRY, rnti, harq_pid, *ndi,(*hb_ptr)->getTxCount());

        /* If the buffer was found in bucket and is still in use - means MAC reusing harq buffer while UL processing still in flight. Cannot honor
           this request as it will lead to race condition. Return failure. */
        if((*hb_ptr)->isInUse() == true)
        {
            if(is_bundled_pdu && (*ndi == 0) && ((*hb_ptr)->getSize() >= buf_size))
            {
                uint16_t Slot_diff = ((sfn * 20 + slot) - ((*hb_ptr)->getSfn() * 20  + (*hb_ptr)->getSlot()));
                if((*hb_ptr)->getNumTtiBundled() + Slot_diff <= pusch_aggr_factor)
                {
                    (*hb_ptr)->incrementNumTtiBundled(Slot_diff);
                }
                else
                {
                    NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "Error: MAC reusing hqbuf for SFN {}.{} cur sfn {}.{} rnti {} hpid {} cell_id {} ndi {}  Slot_diff {} num_tti_bundled {} pusch_aggr_factor while UL processing in flight outside bundling window. Return failure",
                            (*hb_ptr)->getSfn(), (*hb_ptr)->getSlot(), sfn, slot, rnti, harq_pid, cell_id, *ndi, Slot_diff, (*hb_ptr)->getNumTtiBundled(), pusch_aggr_factor);
                    return -1;
                }
            }
            else
            {
                NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "Error: MAC reusing hqbuf for SFN {}.{} cur sfn {}.{} rnti {} hpid {} cell_id {} ndi {}  existing buff size {} new buff size {} is_bundled_pdu {} while UL processing in flight. Return failure",
                    (*hb_ptr)->getSfn(), (*hb_ptr)->getSlot(), sfn, slot, rnti, harq_pid, cell_id, *ndi, (*hb_ptr)->getSize(), buf_size, is_bundled_pdu);
                return -1;    
            }
        }
        
        /* If the buffer was found in bucket and is NOT in use and NDI is 1, means MAC trying to reuse the buffer without waiting for harq retx to
           complete. Log a warning and allow reuse. It is not mandated by 3GPP that a HARQ retx has to be completed if there is a CRC error on a TB. */
        if(*ndi == 1)
        {
            if((*hb_ptr)->getSize() < buf_size)
            {
                NVLOGI_FMT_EVT(TAG, AERIAL_CUPHYDRV_API_EVENT, "SFN {}.{}: MAC reusing hqbuf rnti {} hpid {} cell_id {} with NDI=1 and the requested buffer size {}, release the existing HARQ buffer and reallocate a new HARQ buffer", (*hb_ptr)->getSfn(), (*hb_ptr)->getSlot(), rnti, harq_pid, cell_id, buf_size);
                
                // release the existing HARQ buffer
                bucketReleaseBuffer((*hb_ptr));
                (*hb_ptr) = nullptr;
                
                // reallocate a new HARQ buffer
                if(bucketAllocateBuffer(hb_ptr, buf_size, rnti, harq_pid, ndi, cell_id, is_bundled_pdu, pusch_aggr_factor, sfn, slot))
                {
                    NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "bucketAllocateBuffer reallocation returned error");
                    return -1;
                }
            }
            else
            {
                NVLOGI_FMT_EVT(TAG, AERIAL_CUPHYDRV_API_EVENT, "SFN {}.{} Warning: reused buffer resetting the txCount to 0 for NDI == 1, hqbuf rnti {} hpid {} cell_id {}", (*hb_ptr)->getSfn(), (*hb_ptr)->getSlot(), rnti, harq_pid, cell_id);
            }
            (*hb_ptr)->resetTxCount();
            (*hb_ptr)->refReset();
            (*hb_ptr)->resetNumTtiBundled();
            if(is_bundled_pdu)
            {
                (*hb_ptr)->setMaxTxCount( pdctx->getMaxHarqTxCountBundled());
            }
            else
            {
                (*hb_ptr)->setMaxTxCount( pdctx->getMaxHarqTxCountNonBundled());
            }
            (*hb_ptr)->incrementNumTtiBundled(1);
        }
        else if(*ndi == 0) 
        {
            if((*hb_ptr)->getSize() < buf_size) //this is for L2 misconfiguration!!!
            {
                *ndi = 1; 
                bucketReleaseBuffer((*hb_ptr));
                (*hb_ptr) = nullptr;
                
                // reallocate a new HARQ buffer
                if(bucketAllocateBuffer(hb_ptr, buf_size, rnti, harq_pid, ndi, cell_id, is_bundled_pdu, pusch_aggr_factor, sfn, slot))
                {
                    NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "bucketAllocateBuffer reallocation returned error");
                    return -1;
                }
                (*hb_ptr)->resetTxCount();
                (*hb_ptr)->refReset();
                (*hb_ptr)->resetNumTtiBundled();
                (*hb_ptr)->incrementNumTtiBundled(1);
                if(is_bundled_pdu)
                {
                    (*hb_ptr)->setMaxTxCount( pdctx->getMaxHarqTxCountBundled());
                }
                else
                {
                    (*hb_ptr)->setMaxTxCount( pdctx->getMaxHarqTxCountNonBundled());
                }
                NVLOGW_FMT(TAG, "SFN {}.{} Warining: hqbuf rnti {} hpid {} cell_id {} with NDI=0 for retransmission but the requested buffer size {} is larger than the allocated buffer size {} by L2 misconfiguration, new buffer resetting the txCount to 0 and NDI=1 for NDI == 0!", (*hb_ptr)->getSfn(), (*hb_ptr)->getSlot(), rnti, harq_pid, cell_id, buf_size, (*hb_ptr)->getSize());
            }
        }
    }
    
    (*hb_ptr)->increaseTxCount();
    (*hb_ptr)->setTimestampLastUsed();
    (*hb_ptr)->refAdd();
    (*hb_ptr)->setSfnSlot(sfn, slot);
    (*hb_ptr)->setInUse(); //set in use after every successful allocation
    return 0;
}

int HarqPoolManager::bucketReleaseBuffer(HarqBuffer * hb_ptr) {

    HarqPool * hb_pool = nullptr;
    HarqBucket * hb_bucket = nullptr;

    if(hb_ptr == nullptr)
        return EINVAL;

    // Remove buffer from bucket
    if(hb_bucket_list[hb_ptr->getRnti()%MAX_BUCKET_ENTRY]->pullBuffer(hb_ptr->getRnti(), hb_ptr->getHarqPid(),hb_ptr->getCellId()))
    {
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "SFN {}.{} buffer {} not found for rnti {} hpid {} cell_id {}",
            hb_ptr->getSfn(), hb_ptr->getSlot(), reinterpret_cast<void*>(hb_ptr), hb_ptr->getRnti(), hb_ptr->getHarqPid(), hb_ptr->getCellId());
        return -1;
    }
    // Restore buffer into original pool
    hb_pool = poolFind(hb_ptr->getSize());
    if(hb_pool == nullptr)
    {
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "poolFind size {}", hb_ptr->getSize());
        return -1;
    }

    if(hb_pool->pushBuffer(hb_ptr))
    {
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "pushBuffer error");
        return -1;
    }

    return 0;
}

int HarqPoolManager::unsetInUse(HarqBuffer * hb_ptr) {
    if(hb_ptr == nullptr)
        return EINVAL;

    hb_ptr->unsetInUse();

    return 0;
}

int HarqPoolManager::cleanupHarqBuckets(ReleasedHarqBufferInfo& released_harq_buffer_info) {
    HarqBuffer * hb_ptr;
    t_ns now = Time::nowNs();
    uint32_t count = 0;
    for(int hb_idx = 0; hb_idx < MAX_BUCKET_ENTRY; hb_idx++) {
        for(int i=0; i<MAX_BUCKET_ENTRY; i++ ) {
            for(int j=0; j<MAX_HARQ_PID; j++ ) {
                for(int k=0; k<MAX_BUCKET_MAP_ENTRIES; k++) {
                    hb_ptr = hb_bucket_list[hb_idx]->bucket_map[i][j].hb_rnti_v[k];
                    if(hb_ptr != nullptr)
                    {
                        // FIXME: in_use is not locked, caller need to make sure there's no multi-thread competition.
                        if(hb_ptr->isInUse() == false &&
                            (
                                hb_ptr->getTxCount() >= MAX_HARQ_RE_TX                           ||
                                (now - hb_ptr->getTimestampLastUsed()).count() > MAX_HARQB_NS_BUCKET
                            )
                        )
                        {
                            /*NVLOGD_FMT(TAG, "Clean HARQ buffer rnti {} hpid{} cell_id {} TxCount={}",
                                    hb_ptr->getRnti(), hb_ptr->getHarqPid(), hb_ptr->getCellId(), hb_ptr->getTxCount());*/
                            if(0 == bucketReleaseBuffer(hb_ptr))
                            {
                                ReleasedHarqBuffer released_harq_buffer;
                                released_harq_buffer.rnti = hb_ptr->getRnti();
                                released_harq_buffer.harq_pid = hb_ptr->getHarqPid();
                                released_harq_buffer.sfn = hb_ptr->getSfn();
                                released_harq_buffer.slot = hb_ptr->getSlot();
                                released_harq_buffer.cell_id = hb_ptr->getCellId();
                                released_harq_buffer_info.released_harq_buffer_list.push_back(released_harq_buffer);
                                released_harq_buffer_info.num_released_harq_buffers++;
                            }
                            count++;
                            hb_ptr = nullptr;
                        }
                        /*else
                            NVLOGD_FMT(TAG, "Clean HARQ buffer rnti {} hpid{} cell_id {} TxCount={} time={}",
                                    hb_ptr->getRnti(), hb_ptr->getHarqPid(), hb_ptr->getCellId(), hb_ptr->getTxCount(),
                                    (now - hb_ptr->getTimestampLastUsed()).count());*/

                    }
                }
            }
        }
    }

    NVLOGW_FMT(TAG, " Tried to free {} stale buffers, but only freed {} buffers",count, released_harq_buffer_info.num_released_harq_buffers);
    for(int i = 0; i < MAX_HARQ_POOLS; i++)
        NVLOGW_FMT(TAG, "HARQ pool[{}] has {} buffers",i,hb_pool_list[i]->countElements());
    return 0;
}

void HarqPoolManager::freeAllHarqBuffers(ReleasedHarqBufferInfo& released_harq_buffer_info) {
    HarqBuffer * hb_ptr;
    uint32_t count = 0;

    for(int hb_idx = 0; hb_idx < MAX_BUCKET_ENTRY; hb_idx++) {
        for(int i=0; i<MAX_BUCKET_ENTRY; i++ ) {
            for(int j=0; j<MAX_HARQ_PID; j++ ) {
                for(int k=0; k<MAX_BUCKET_MAP_ENTRIES; k++) {
                    hb_ptr = hb_bucket_list[hb_idx]->bucket_map[i][j].hb_rnti_v[k];
                    if(hb_ptr != nullptr)
                    {
                        // FIXME: in_use is not locked, caller need to make sure there's no multi-thread competition.
                        if(hb_ptr->isInUse() == false)
                        {
                            /*NVLOGD_FMT(TAG, "Clean HARQ buffer rnti {} hpid{} cell_id {} TxCount={}",
                                    hb_ptr->getRnti(), hb_ptr->getHarqPid(), hb_ptr->getCellId(), hb_ptr->getTxCount());*/
                            if(0 == bucketReleaseBuffer(hb_ptr))
                            {
                                ReleasedHarqBuffer released_harq_buffer;
                                released_harq_buffer.rnti = hb_ptr->getRnti();
                                released_harq_buffer.harq_pid = hb_ptr->getHarqPid();
                                released_harq_buffer.cell_id = hb_ptr->getCellId();
                                released_harq_buffer.sfn = hb_ptr->getSfn();
                                released_harq_buffer.slot = hb_ptr->getSlot();
                                released_harq_buffer_info.released_harq_buffer_list.push_back(released_harq_buffer);
                                released_harq_buffer_info.num_released_harq_buffers++;
                            }
                            count++;
                            hb_ptr = nullptr;
                        }
                    }
                }
            }
        }
    }
    NVLOGW_FMT(TAG, " Tried to free {} buffers, but only freed {} buffers ",count, released_harq_buffer_info.num_released_harq_buffers);
    for(int i = 0; i < MAX_HARQ_POOLS; i++)
        NVLOGW_FMT(TAG, "HARQ pool[{}] has {} buffers",i,hb_pool_list[i]->countElements());
}

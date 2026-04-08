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

#define TAG (NVLOG_TAG_BASE_CUPHY_DRIVER + 26) // "DRV.ULBUF"

#include "ulbuffer.hpp"
#include "cuphydriver_api.hpp"
#include "context.hpp"
#include "nvlog.hpp"
#include "exceptions.hpp"
#include "cell.hpp"
#include <typeinfo>

ULInputBuffer::ULInputBuffer(phydriver_handle _pdh, GpuDevice* _gDev, cell_id_t _cell_id, size_t _size) :
    pdh(_pdh),
    gDev(_gDev),
    cell_id(_cell_id),
    addr_sz(_size)
{
    mf.init(_pdh, std::string("ULInputBuffer"), sizeof(ULInputBuffer));
    PhyDriverCtx* pdctx = StaticConversion<PhyDriverCtx>(pdh).get();

    // Cell * cell_ptr = pdctx->getCellById(cell_id);
    // if(cell_ptr == nullptr)
    //     PHYDRIVER_THROW_EXCEPTIONS(-1, "No valid cell associated to UL Buffer obj");

    gDev->setDevice();

    // addr_sz = ORAN_MAX_PRB * ORAN_MAX_SYMBOLS * ORAN_RE * cell_ptr->geteAxCNum() /* MAX_AP_PER_SLOT */ * sizeof(uint32_t);

    addr_d.reset(new dev_buf(addr_sz * sizeof(uint8_t), gDev));
    addr_d->clear();
    mf.addGpuRegularSize(addr_d->size_alloc);

    addr_h.reset(new host_buf(addr_sz * sizeof(uint8_t), gDev));
    addr_h->clear();
    mf.addCpuPinnedSize(addr_sz);

    active            = false;
    id = Time::nowNs().count();
}

ULInputBuffer::~ULInputBuffer()
{
}

uint64_t ULInputBuffer::getId() const {
    return id;
}

cell_id_t ULInputBuffer::getCellId() const {
    return cell_id;
}

int ULInputBuffer::reserve()
{
    int ret = 0;

    mlock.lock();
    if(active == true)
        ret = -1;
    else
        active   = true;
    mlock.unlock();

    return ret;
}

void ULInputBuffer::release()
{
    mlock.lock();
    active   = false;
    mlock.unlock();
}

void ULInputBuffer::cleanup(cudaStream_t stream)
{
    /*
     * Can't be moved to buffer release because FH, at the end of the DL task,
     * still has to send the content of the buffer (bug: DPDK callback to give an ACK is not working yet)
     */
    CUDA_CHECK_PHYDRIVER(cudaMemsetAsync(getBufD(), 0, getSize(), stream));
}

size_t ULInputBuffer::getSize() const
{
    return addr_sz;
}

uint8_t* ULInputBuffer::getBufD() const
{
    return addr_d->addr();
}

uint8_t* ULInputBuffer::getBufH() const
{
    return addr_h->addr();
}

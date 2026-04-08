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

#define TAG (NVLOG_TAG_BASE_CUPHY_DRIVER + 11) // "DRV.DLBUF"

#include "dlbuffer.hpp"
#include "cuphydriver_api.hpp"
#include "context.hpp"
#include "nvlog.hpp"
#include "exceptions.hpp"
#include "cuda_events.hpp"
#include "cell.hpp"
#include <typeinfo>
#include <slot_command/slot_command.hpp>

#ifdef ENABLE_32DL
#define MAX_PDSCH_DL_LAYERS 32
#else
#define MAX_PDSCH_DL_LAYERS 16
#endif

DLOutputBuffer::DLOutputBuffer(phydriver_handle _pdh, GpuDevice* _gDev, cell_id_t _cell_id) :
    pdh(_pdh),
    gDev(_gDev),
    cell_id(_cell_id)
{
    PhyDriverCtx* pdctx = StaticConversion<PhyDriverCtx>(pdh).get();
    Cell * cell_ptr = pdctx->getCellById(cell_id);
    if(cell_ptr == nullptr)
        PHYDRIVER_THROW_EXCEPTIONS(-1, "No valid cell associated to DL Buffer obj");

    mf.init(_pdh, std::string("DLOutputBuffer"), sizeof(DLOutputBuffer));

    sz_mr = DL_OUTPUT_BUFFER_SIZE; //CUPHY_N_TONES_PER_PRB * 273 * OFDM_SYMBOLS_PER_SLOT * cell_ptr->geteAxCNum() * sizeof(uint32_t);

    large_buffer = cuphy::make_unique_device<cuFloatComplex>(sz_mr / sizeof(cuFloatComplex));
    mf.addGpuRegularSize(sz_mr);

    tx_tensor    = cuphy::tensor_device(large_buffer.get(),
                                        CUPHY_C_16F,
                                        CUPHY_N_TONES_PER_PRB * ORAN_MAX_PRB,
                                        OFDM_SYMBOLS_PER_SLOT,
                                        MAX_PDSCH_DL_LAYERS,
                                        cuphy::tensor_flags::align_tight
                                    );

    sz_tx = tx_tensor.desc().get_size_in_bytes();
    if(sz_tx >= sz_mr)
    {
        std::cerr << "Buffer (" << sz_mr << " bytes) is smaller than tx_tensor (" << sz_tx << ")" << std::endl;
        PHYDRIVER_THROW_EXCEPTIONS(-1, "DLOutputBuffer size is too small");
    }
    CUDA_CHECK_PHYDRIVER(cudaMemset(tx_tensor.addr(), 0, sz_tx));
    addr_d = (uint8_t*)tx_tensor.addr();

    addr_h.reset(new host_buf(sz_mr * sizeof(uint8_t), gDev));
    addr_h->clear();
    mf.addCpuPinnedSize(sz_mr);

    umsg_tx_list.umsg_info_symbol_antenna = (fhproxy_umsg_tx*)calloc(MAX_UPLANE_MSGS_PER_SLOT, sizeof(fhproxy_umsg_tx)); //Overallocation
    if(umsg_tx_list.umsg_info_symbol_antenna == NULL)
        PHYDRIVER_THROW_EXCEPTIONS(-1, "Couldn't allocate fhproxy_umsg_tx");

    mf.addCpuRegularSize(MAX_UPLANE_MSGS_PER_SLOT * sizeof(fhproxy_umsg_tx*));

    active            = false;
    id = Time::nowNs().count();

    //Processing State Variables
    compression_is_queued = false;
    buffer_ready_gdr = gDev->newGDRbuf(1 * sizeof(uint32_t));
    mf.addGpuPinnedSize(buffer_ready_gdr->size_alloc);
    resetProcessingState();

    //Reserving variables
    last_used = Time::zeroNs();

    size_t prb_num_ptrs = ORAN_MAX_PRB * OFDM_SYMBOLS_PER_SLOT * API_MAX_ANTENNAS;
    prb_ptrs = cuphy::make_unique_device<uint8_t*>(prb_num_ptrs);
    mf.addGpuRegularSize(prb_num_ptrs * sizeof(uint8_t*));

    mod_comp_params_per_cell = nullptr;
    mod_comp_config_temp = nullptr;
    uint16_t dl_comp_method = cell_ptr->getDLCompMeth();

    if(dl_comp_method == static_cast<int>(aerial_fh::UserDataCompressionMethod::MODULATION_COMPRESSION))
    {
        //Initialize compression config desc
        mod_comp_config_temp = cuphy::make_unique_pinned<struct mod_compression_params>(1);
        memset(mod_comp_config_temp.get(), 0, sizeof(struct mod_compression_params));
        CUDA_CHECK_PHYDRIVER(cudaMalloc((void**)&mod_comp_params_per_cell, sizeof(mod_compression_params)));
        mf.addGpuRegularSize(sizeof(mod_compression_params));
    }
    else
    {
        mod_comp_params_per_cell = nullptr;
    }
    if (pdctx->gpuCommDlEnabled()) {
        pdctx->setGpuCommsCtx();
        NVLOGI_FMT(TAG, "Setting GPU Comms context and initializing events!");
        CUDA_CHECK_PHYDRIVER(cudaEventCreate(&prepare_start_evt));
        CUDA_CHECK_PHYDRIVER(cudaEventCreate(&prepare_copy_evt));
        if(pdctx->enablePrepareTracing()) {
            CUDA_CHECK_PHYDRIVER(cudaEventCreate(&prepare_stop_evt));
            CUDA_CHECK_PHYDRIVER(cudaEventCreate(&pre_prepare_stop_evt));
        } else {
            CUDA_CHECK_PHYDRIVER(cudaEventCreateWithFlags(&prepare_stop_evt, cudaEventDisableTiming));
            CUDA_CHECK_PHYDRIVER(cudaEventCreateWithFlags(&pre_prepare_stop_evt, cudaEventDisableTiming));
        }

        cudaEventCreate(&tx_end_evt);        
        pdctx->setDlCtx();
    }
    cudaEventCreate(&all_channels_done_evt);
    cudaEventCreate(&compression_start_evt);
    cudaEventCreate(&compression_stop_evt);

    CUDA_CHECK_PHYDRIVER(cudaEventCreateWithFlags(&ev_cleanup, cudaEventDisableTiming));
}

DLOutputBuffer::~DLOutputBuffer()
{
    // Should not call cudaFree on addr_d, as this is using the large_buffer allocation
    // on which the device_deleter will be called automatically.

    delete buffer_ready_gdr;
    free(umsg_tx_list.umsg_info_symbol_antenna);
    PhyDriverCtx* pdctx = StaticConversion<PhyDriverCtx>(pdh).get();
    Cell * cell_ptr = pdctx->getCellById(cell_id);
    if(cell_ptr)
    {
        if(cell_ptr->getDLCompMeth() == static_cast<int>(aerial_fh::UserDataCompressionMethod::MODULATION_COMPRESSION))
        {
            cudaFree(mod_comp_params_per_cell);
        }
    }

}

uint64_t DLOutputBuffer::getId() const {
    return id;
}

void DLOutputBuffer::resetProcessingState()
{
    //Reset state of compression
    compression_is_queued = false;

    //Reset state of gdr copy
    ACCESS_ONCE(((uint32_t*)buffer_ready_gdr->addrh())[0]) = 0;

}

int DLOutputBuffer::reserve()
{
    int ret = 0;
    PhyDriverCtx* pdctx = StaticConversion<PhyDriverCtx>(pdh).get();

    mlock.lock();

    if(active == true && (pdctx->gpuCommDlEnabled() && (Time::nowNs() - last_used).count() < DL_OUTPUT_BUFFER_BUSY_NS)) {
        //Unable to reserve
        ret = -1;
    } else {
        //Able to reserve - make sure state of the buffer is reset
        resetProcessingState();
        active = true;
        last_used = Time::nowNs();
        
    }

    mlock.unlock();

    return ret;
}

//NB: never called in case of GPU-init communications
void DLOutputBuffer::release()
{
    mlock.lock();
    active = false;
    mlock.unlock();
}

struct umsg_fh_tx_msg& DLOutputBuffer::getTxMsgContainer() {
    return umsg_tx_list;
}

void DLOutputBuffer::cleanup(cudaStream_t stream, MpsCtx * mpsCtx)
{
    /*
     * Can't be moved to buffer release because FH, at the end of the DL task,
     * still has to send the content of the buffer (bug: DPDK callback to give an ACK is not working yet)
     */

    mpsCtx->setCtx();
    CUDA_CHECK_PHYDRIVER(cudaMemsetAsync(getBufD(), 0, getSize(), stream));
    CUDA_CHECK_PHYDRIVER(cudaEventRecord(ev_cleanup, stream));
}

cudaEvent_t* DLOutputBuffer::cleanupEventRecord(cudaStream_t stream, MpsCtx * mpsCtx)
{

    mpsCtx->setCtx();
    CUDA_CHECK_PHYDRIVER(cudaEventRecord(ev_cleanup, stream));
    return &ev_cleanup;
}

void DLOutputBuffer::waitCleanup(cudaStream_t stream, MpsCtx * mpsCtx)
{
    /*
     * Can't be moved to buffer release because FH, at the end of the DL task,
     * still has to send the content of the buffer (bug: DPDK callback to give an ACK is not working yet)
     */
    mpsCtx->setCtx();
    CUDA_CHECK_PHYDRIVER(cudaStreamWaitEvent(stream, ev_cleanup, 0));
}

size_t DLOutputBuffer::getSizeFh() const
{
    return sz_mr;
}

size_t DLOutputBuffer::getSize() const
{
    return sz_tx;
}

uint8_t* DLOutputBuffer::getBufD() const
{
    return addr_d;
}

uint8_t* DLOutputBuffer::getBufH() const
{
    return addr_h->addr();
}

cuphy::tensor_device* DLOutputBuffer::getTensor()
{
    return &(tx_tensor);
}

/////////////////////////////////////////////////////////////////////////////////////////
//// COMPRESSION
/////////////////////////////////////////////////////////////////////////////////////////

int DLOutputBuffer::runCompression(const std::array<compression_params, NUM_USER_DATA_COMPRESSION_METHODS>& cparams_array, MpsCtx * mpsCtx, cudaStream_t stream)
{
    mpsCtx->setCtx();

    {
        MemtraceDisableScope md;
        CUDA_CHECK_PHYDRIVER(cudaEventRecord(compression_start_evt, stream));
    }
    

    launch_kernel_compression(stream, cparams_array);
   
    {
        MemtraceDisableScope md;
        CUDA_CHECK_PHYDRIVER(cudaEventRecord(compression_stop_evt, stream));
    }

    compression_is_queued = true;

    return 0;
}

//Non-blocking wait on externally specified cuda event
// Returns 1 if run completion event has been triggered
int DLOutputBuffer::waitEventNonBlocking(cudaEvent_t event) {
    cudaError_t temp = cudaEventQuery(event);

    //While waiting for completion, call will return cudaErrorNotReady
    if(temp == cudaErrorNotReady) {
        return 0;
    }

    //Throw exception on non "cudaSuccess" value
    CUDA_CHECK_PHYDRIVER(temp);

    //Result must have been cudaSuccess
    return 1;
}


int DLOutputBuffer::waitCompression(cudaEvent_t event, bool for_compression_start)
{

    //Make sure we have a valid cell to retrieve slot parameters
    PhyDriverCtx* pdctx = StaticConversion<PhyDriverCtx>(pdh).get();
    Cell * cell_ptr = pdctx->getCellById(cell_id);
    if(cell_ptr == nullptr)
    {
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "No valid cell associated to DL Output Buffer obj");
        return -1;
    }

    //Make sure this DL buffer object has been reserved
    if(active.load() == false)
        return -1;

    //Set threshold and reference time
    t_ns threshold_t, start_t;
    threshold_t = t_ns((cell_ptr->getTtiNsFromMu(cell_ptr->getMu()) * cell_ptr->getSlotAhead()));
    start_t     = Time::nowNs();

    //Wait until compression has been submitted
    while(!compression_is_queued) {
        if(Time::nowNs() - start_t > threshold_t)
        {
            NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "ERROR compression: DLBuffer Object Cell {} waiting for compression to be queued for more than {} ns",
                cell_ptr->getPhyId(), cell_ptr->getTtiNsFromMu(cell_ptr->getMu()) * cell_ptr->getSlotAhead());
            return -1;
        }
    }

    //Wait for event to occur
    while(waitEventNonBlocking(event) == 0)
    {
        if(Time::nowNs() - start_t > threshold_t)
        {
            NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "ERROR compression: DLBuffer Object Cell {} waiting for compression {} for more than {} ns",
                cell_ptr->getPhyId(), for_compression_start ? "start" : "completion", cell_ptr->getTtiNsFromMu(cell_ptr->getMu()) * cell_ptr->getSlotAhead());
            return -1;
        }
    }

    return 0;
}

int DLOutputBuffer::waitPrePrepareStop()
{

    //Make sure we have a valid cell to retrieve slot parameters
    PhyDriverCtx* pdctx = StaticConversion<PhyDriverCtx>(pdh).get();
    Cell * cell_ptr = pdctx->getCellById(cell_id);
    if(cell_ptr == nullptr)
    {
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "No valid cell associated to DL Output Buffer obj");
        return -1;
    }

    //Make sure this DL buffer object has been reserved
    if(active.load() == false)
        return -1;

    //Set threshold and reference time
    t_ns threshold_t, start_t;
    threshold_t = t_ns((cell_ptr->getTtiNsFromMu(cell_ptr->getMu()) * cell_ptr->getSlotAhead()));
    start_t     = Time::nowNs();

    //Wait for event to occur
    while(waitEventNonBlocking(pre_prepare_stop_evt) == 0)
    {
        if(Time::nowNs() - start_t > threshold_t)
        {
            NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "ERROR PrePrepare: DLBuffer Object Cell {} waiting for PrePrepare completion more than {} ns",
                cell_ptr->getPhyId(), cell_ptr->getTtiNsFromMu(cell_ptr->getMu()) * cell_ptr->getSlotAhead());
            return -1;
        }
    }

    return 0;
}

int DLOutputBuffer::waitCompressionStart() {
    return waitCompression(compression_start_evt, true);
}

int DLOutputBuffer::waitCompressionStop() {
    int return_val = waitCompression(compression_stop_evt, false);
    compression_is_queued = false;
    return return_val;
}

uint32_t* DLOutputBuffer::getReadyFlag()
{
    return (uint32_t*)buffer_ready_gdr->addrd();
}

int DLOutputBuffer::setReadyFlag(cudaStream_t stream)
{
    MemtraceDisableScope md;
    launch_kernel_write(stream, (uint32_t*)buffer_ready_gdr->addrd(), (uint32_t)1);
    return 0;
}

void DLOutputBuffer::getPrepareExecutionTimes(float& time1, float& time2, float& time3) {
    PhyDriverCtx* pdctx = StaticConversion<PhyDriverCtx>(pdh).get();

    // Set all values to 0
    time1 = 0.0;
    time2 = 0.0;
    time3 = 0.0;
    if(pdctx->gpuCommDlEnabled()) {
        time1 = GetCudaEventElapsedTime(prepare_start_evt, prepare_copy_evt, __func__);
        time2 = GetCudaEventElapsedTime(prepare_copy_evt, pre_prepare_stop_evt, __func__);
        time3 = GetCudaEventElapsedTime(pre_prepare_stop_evt, prepare_stop_evt, __func__);
    }
}

float DLOutputBuffer::getPrepareExecutionTime1() {
    //Returns the duration of the prepare kernel
    PhyDriverCtx* pdctx = StaticConversion<PhyDriverCtx>(pdh).get();

    if(pdctx->gpuCommDlEnabled()) {
        return 1000.0f * GetCudaEventElapsedTime(prepare_start_evt, prepare_copy_evt, __func__);
    }

    return 0.0f;
}

float DLOutputBuffer::getPrepareExecutionTime2() {
    //Returns the duration of the prepare kernel
    PhyDriverCtx* pdctx = StaticConversion<PhyDriverCtx>(pdh).get();

    if(pdctx->gpuCommDlEnabled()) {
        return 1000.0f * GetCudaEventElapsedTime(prepare_copy_evt, pre_prepare_stop_evt, __func__);
    }

    return 0.0f;
}

float DLOutputBuffer::getPrepareExecutionTime3() {
    //Returns the duration of the prepare kernel
    PhyDriverCtx* pdctx = StaticConversion<PhyDriverCtx>(pdh).get();

    if(pdctx->gpuCommDlEnabled()) {
        return 1000.0f * GetCudaEventElapsedTime(pre_prepare_stop_evt, prepare_stop_evt, __func__);
    }

    return 0.0f;
}


float DLOutputBuffer::getChannelToCompressionGap() {
    //Returns the time between end of GPU channel execution (for all channels) and start of compression (in usec)
    return 1000.0f * GetCudaEventElapsedTime(all_channels_done_evt, compression_start_evt, __func__);
}

float DLOutputBuffer::getCompressionExecutionTime() {
    //Returns the time between start and end of GPU compression execution (in usec)
    return 1000.0f * GetCudaEventElapsedTime(compression_start_evt, compression_stop_evt, __func__);
}

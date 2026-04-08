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

#define TAG (NVLOG_TAG_BASE_CUPHY_DRIVER + 8) // "DRV.PHYCH"

#include "phychannel.hpp"
#include "cuphydriver_api.hpp"
#include "context.hpp"
#include "nvlog.hpp"
#include "exceptions.hpp"
#include "cuda_events.hpp"
#include <typeinfo>

/////////////////////////////////////////////////////////////
//// Generic PhyChannel
/////////////////////////////////////////////////////////////

PhyChannel::PhyChannel(phydriver_handle _pdh, GpuDevice* _gDev, cell_id_t _cell_id, cudaStream_t _s_channel, MpsCtx * _mpsCtx) :
    pdh(_pdh),
    gDev(_gDev),
    cell_id(_cell_id),
    s_channel(_s_channel),
    mpsCtx(_mpsCtx)
{
    active              = false;
    setup_status     = CH_SETUP_NOT_DONE;
    run_status     = CH_RUN_NOT_DONE;
    id                  = get_ns();
    current_slot_params = nullptr;
    buf_sz              = 0;
    buf_d               = nullptr;
    buf_h               = nullptr;
    pCuphyTracker       = nullptr;
    cuphy_tracker.pMemoryFootprint = nullptr;

    // For debug: assign a different this_id for each instance
    static int instance_id = 0;
    this_id = instance_id++;

    PhyDriverCtx * pdctx = StaticConversion<PhyDriverCtx>(pdh).get();

    setCtx();

    // Initialize memory footprint tracker for base class, name can be overridden by derived classes
    mf.init(pdh, std::string("PhyChannelBase"), 0);

    channel_complete_h.reset(new host_buf(1 * sizeof(uint32_t), gDev));
    channel_complete_h->clear();
    mf.addCpuPinnedSize(1 * sizeof(uint32_t));

    channel_complete_gdr.reset(gDev->newGDRbuf(1 * sizeof(uint32_t)));
    ((uint32_t*)channel_complete_gdr->addrh())[0] = 0;
    mf.addGpuPinnedSize(channel_complete_gdr->size_alloc);

    CUDA_CHECK_PHYDRIVER(cudaEventCreate(&start_run));
    CUDA_CHECK_PHYDRIVER(cudaEventCreate(&end_run));
    CUDA_CHECK_PHYDRIVER(cudaEventCreate(&start_setup));
    CUDA_CHECK_PHYDRIVER(cudaEventCreate(&end_setup));

    CUDA_CHECK_PHYDRIVER(cudaEventCreateWithFlags(&run_completion, cudaEventDisableTiming));

    cellStatPrm.phyCellId  = 0;
    cellStatPrm.nRxAnt     = 0;
    cellStatPrm.nRxAntSrs  = 0;
    cellStatPrm.nTxAnt     = 0;
    cellStatPrm.nPrbUlBwp  = 0;
    cellStatPrm.nPrbDlBwp  = 0;
    cellStatPrm.mu         = 0;
    // cellStatPrm.beta_dmrs  = 0;
    // cellStatPrm.beta_qam   = 0;

    channel_type = slot_command_api::channel_type::NONE;

    cnt_used = 0;

};

PhyChannel::~PhyChannel()
{
    CUDA_CHECK_PHYDRIVER(cudaEventDestroy(start_run));
    CUDA_CHECK_PHYDRIVER(cudaEventDestroy(end_run));
    CUDA_CHECK_PHYDRIVER(cudaEventDestroy(start_setup));
    CUDA_CHECK_PHYDRIVER(cudaEventDestroy(end_setup));

    CUDA_CHECK_PHYDRIVER(cudaEventDestroy(run_completion));

    active = false;
    setup_status     = CH_SETUP_NOT_DONE;
    run_status     = CH_RUN_NOT_DONE;

};

phydriver_handle PhyChannel::getPhyDriverHandler(void) const
{
    return pdh;
}

uint64_t PhyChannel::getId() const
{
    return id;
}

void PhyChannel::setActive()
{
    active = true;
}

void PhyChannel::setInactive()
{
    active = false;
}

bool PhyChannel::isActive()
{
    return active.load();
}

void PhyChannel::setSetupStatus(ch_setup_status_t status)
{
    setup_status = status;
}

void PhyChannel::setRunStatus(ch_run_status_t status)
{
    run_status = status;
}

ch_setup_status_t PhyChannel::getSetupStatus()
{
    return setup_status;
}

ch_run_status_t PhyChannel::getRunStatus()
{
    return run_status;
}


static int get_channel_count(slot_params* params)
{
    int channels = 0;

    if(params->slot_phy_prms.pusch != nullptr)
        channels++;
    if(params->slot_phy_prms.pdsch != nullptr)
        channels++;
    if(params->slot_phy_prms.pdcch_ul != nullptr)
        channels++;
    if(params->slot_phy_prms.pdcch_dl != nullptr)
        channels++;
    if(params->slot_phy_prms.pbch != nullptr)
        channels++;
    if(params->slot_phy_prms.csi_rs != nullptr)
        channels++;
    if(params->slot_phy_prms.pucch != nullptr)
        channels++;
    if(params->slot_phy_prms.prach != nullptr)
        channels++;

    return channels;
}


int PhyChannel::setDynParams(slot_params* curr_slot_params) //struct slot_command_api::slot_indication si, slot_command_api::phy_slot_params& slot_phy_prms)
{
    NVLOGD_FMT(TAG, "{}: {} count={} current_slot_params {} ==> {}", __FUNCTION__, channel_name.c_str(), get_channel_count(curr_slot_params), reinterpret_cast<void*>(current_slot_params), reinterpret_cast<void*>(curr_slot_params));
    current_slot_params = curr_slot_params; //new slot_params(si, &slot_phy_prms);
    return 0;
}

int PhyChannel::setDynAggrParams(slot_params_aggr* _aggr_slot_params) //struct slot_command_api::slot_indication si, slot_command_api::phy_slot_params& slot_phy_prms)
{
    NVLOGI_FMT(TAG, "phychannel{}: setDynAggrParams: _aggr_slot_params={}", this_id, reinterpret_cast<void*>(_aggr_slot_params));
    aggr_slot_params = _aggr_slot_params;
    return 0;
}

void PhyChannel::cleanupDynParams()
{
    NVLOGD_FMT(TAG, "{}: {} count={} current_slot_params={}", __FUNCTION__, channel_name.c_str(), get_channel_count(current_slot_params), reinterpret_cast<void*>(current_slot_params));
    if(current_slot_params)
    {
        switch(channel_type)
        {
        case slot_command_api::channel_type::PUSCH:
            current_slot_params->slot_phy_prms.pusch = nullptr;
            break;
        case slot_command_api::channel_type::PDSCH:
            current_slot_params->slot_phy_prms.pdsch = nullptr;
            break;
        case slot_command_api::channel_type::PDCCH_DL:
            current_slot_params->slot_phy_prms.pdcch_dl = nullptr;
            break;
        case slot_command_api::channel_type::PDCCH_UL:
            current_slot_params->slot_phy_prms.pdcch_ul = nullptr;
            break;
        case slot_command_api::channel_type::PBCH:
            current_slot_params->slot_phy_prms.pbch = nullptr;
            break;
        case slot_command_api::channel_type::CSI_RS:
            current_slot_params->slot_phy_prms.csi_rs = nullptr;
            break;
        case slot_command_api::channel_type::PUCCH:
            current_slot_params->slot_phy_prms.pucch = nullptr;
            break;
        case slot_command_api::channel_type::PRACH:
            current_slot_params->slot_phy_prms.prach = nullptr;
            break;
        default:
            break;
        }

        if(get_channel_count(current_slot_params) == 0)
        {
            delete current_slot_params;
        }
        current_slot_params = nullptr;
    }
}

cell_id_t PhyChannel::getCellId()
{
    return cell_id;
}

slot_command_api::oran_slot_ind PhyChannel::getOranSlotIndication()
{
    // slot_command_api::oran_slot_ind oran_ind = slot_command_api::to_oran_slot_format(current_slot_params.si);
    // return std::move(oran_ind);
    return slot_command_api::to_oran_slot_format(current_slot_params->si);
}

slot_command_api::oran_slot_ind PhyChannel::getOranAggrSlotIndication()
{
    // slot_command_api::oran_slot_ind oran_ind = slot_command_api::to_oran_slot_format(current_slot_params.si);
    // return std::move(oran_ind);
    return slot_command_api::to_oran_slot_format(*(aggr_slot_params->si));
}

const slot_command_api::slot_info_t& PhyChannel::getOranSlotInfo()
{
    return *current_slot_params->slot_phy_prms.sym_prb_info.get();
}

struct slot_command_api::cell_group_command * PhyChannel::getCellGroupCommand()
{
    return aggr_slot_params->cgcmd;
}

void PhyChannel::configureCtx(MpsCtx * _mpsCtx)
{
    mpsCtx = _mpsCtx;
}

MpsCtx * PhyChannel::getCtx()
{
    return mpsCtx;
}

void PhyChannel::setCtx()
{
    mpsCtx->setCtx();
}

uint8_t * PhyChannel::getBufD() const {
    return buf_d;
}

uint8_t * PhyChannel::getBufH() const {
    return buf_h;
}

size_t PhyChannel::getBufSize() const {
    return buf_sz;
}

cudaStream_t PhyChannel::getStream() const {
    return s_channel;
}

int PhyChannel::cleanup()
{
    PhyDriverCtx* pdctx = StaticConversion<PhyDriverCtx>(pdh).get();

    ACCESS_ONCE(*((uint32_t*)channel_complete_h->addr())) = 0;
    ACCESS_ONCE(*((uint32_t*)channel_complete_gdr->addrh())) = 0;

    return 0;
}

int PhyChannel::waitToStartCPU(uint32_t * wait_addr_h) {
    setCtx();
    launch_kernel_wait_eq(s_channel, wait_addr_h, 1);
    return 0;
}

int PhyChannel::waitToStartGPU(uint32_t * wait_addr_d) {
    setCtx();
    launch_kernel_wait_eq(s_channel, wait_addr_d, 1);
    return 0;
}

int PhyChannel::waitToStartGPU(uint32_t * wait_addr_d, cudaStream_t stream_) {
    setCtx();
    launch_kernel_wait_eq(stream_, wait_addr_d, 1);
    return 0;
}

int PhyChannel::waitToStartGPUEvent(cudaEvent_t event) {
    setCtx();
    CUDA_CHECK_PHYDRIVER(cudaStreamWaitEvent(s_channel, event, 0));
    return 0;
}

int PhyChannel::waitToStartGPUEvent(cudaEvent_t event, cudaStream_t stream_) {
    setCtx();
    CUDA_CHECK_PHYDRIVER(cudaStreamWaitEvent(stream_, event, 0));
    return 0;
}

//Blocking wait on externally specified cuda event
int PhyChannel::waitEvent(cudaEvent_t event) {
    CUDA_CHECK_PHYDRIVER(cudaEventSynchronize(event));
    return 0;
}

//Non-blocking wait on externally specified cuda event
// Returns 1 if run completion event has been triggered
int PhyChannel::waitEventNonBlocking(cudaEvent_t event) {
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

int PhyChannel::waitStartRunEvent()
{
    return waitEvent(start_run);
}

//Returns 1 if run completion event has been triggered
int PhyChannel::waitStartRunEventNonBlocking()
{
    return waitEventNonBlocking(start_run);
}

//Signals the completion of the phychannel run through both a host buffer write
// and a device buffer write
int PhyChannel::signalRunCompletion()
{
    setCtx();

    launch_kernel_write(s_channel, (uint32_t*)channel_complete_h->addr(), 1);
    launch_kernel_write(s_channel, (uint32_t*)channel_complete_gdr->addrd(), 1);

    return 0;
}

//Signals the completion of the phychannel run through an event
// Optionally also signals via host buffer write
int PhyChannel::signalRunCompletionEvent(bool trigger_write_kernel)
{
    setCtx();

    if(trigger_write_kernel){
        launch_kernel_write(s_channel, (uint32_t*)channel_complete_h->addr(), 1);
    }

    CUDA_CHECK_PHYDRIVER(cudaEventRecord(run_completion, s_channel));

    return 0;
}

//Signals the completion of the phychannel run through an event
// Optionally also signals via host buffer write
//Allows external specification of stream
int PhyChannel::signalRunCompletionEvent(cudaStream_t stream_, bool trigger_write_kernel)
{
    setCtx();

    if(trigger_write_kernel){
        launch_kernel_write(stream_, (uint32_t*)channel_complete_h->addr(), 1);
    }

    CUDA_CHECK_PHYDRIVER(cudaEventRecord(run_completion, stream_));

    return 0;
}

//Block a CPU thread until run completion
int PhyChannel::waitRunCompletion(int wait_ns)
{
    t_ns          threshold_t(wait_ns), start_t = Time::nowNs();
    PhyDriverCtx* pdctx = StaticConversion<PhyDriverCtx>(pdh).get();

    if(!isActive())
        return -1;

    while(ACCESS_ONCE(*((uint32_t*)channel_complete_h->addr())) == 0)
    {
        if(Time::nowNs() - start_t > threshold_t)
        {
            NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "ERROR: {} Object {} Cell {} waiting for GPU task more than {} ns",
                channel_name.c_str(), getId(), cell_id, wait_ns);
            return -1;
        }
    }

    return 0;
}

int PhyChannel::waitRunCompletionEvent()
{
    return waitEvent(run_completion);
}

//Returns 1 if run completion event has been triggered
int PhyChannel::waitRunCompletionEventNonBlocking()
{
    return waitEventNonBlocking(run_completion);
}

//Block a GPU stream until run completion (using wait kernel)
int PhyChannel::waitRunCompletionGPU(cudaStream_t stream_, MpsCtx * mpsCtx_)
{
    mpsCtx_->setCtx();
    launch_kernel_wait_eq(stream_, (uint32_t*)channel_complete_gdr->addrd(), 1);
    return 0;
}

//Block a GPU stream until run completion (using event)
int PhyChannel::waitRunCompletionGPUEvent(cudaStream_t stream_, MpsCtx * mpsCtx_)
{
    mpsCtx_->setCtx();
    MemtraceDisableScope md;
    CUDA_CHECK_PHYDRIVER(cudaStreamWaitEvent(stream_, run_completion, 0));
    return 0;
}

int PhyChannel::reserve(uint8_t * _buf_d, uint8_t * _buf_h, size_t _buf_sz)
{
    PhyDriverCtx* pdctx = StaticConversion<PhyDriverCtx>(pdh).get();
    int ret = 0;

    if(_buf_sz == 0 || _buf_d == nullptr) //_buf_h is just for debug purposes
    {
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "Error reserving obj {}: invalid params", getId());
        ret = EINVAL;
        goto exit;
    }

    // Atomically check if already active and set to active if not
    // exchange() returns the old value and sets the new value atomically
    if(active.exchange(true) == true)
    {
        ret = -1;  // Was already active, cannot reserve
    }

    if(ret == 0)
    {
        buf_d = _buf_d;
        buf_h = _buf_h;
        buf_sz = _buf_sz;
    }
exit:
    return ret;
}

int PhyChannel::reserve(uint8_t * _buf_d, size_t _buf_sz, cuphy::tensor_device* _tx_tensor)
{
    PhyDriverCtx* pdctx = StaticConversion<PhyDriverCtx>(pdh).get();
    int           ret   = 0;

    if(_buf_sz == 0 || _buf_d == nullptr || _tx_tensor == nullptr)
    {
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "Error reserving obj {}: invalid params", getId());
        ret = EINVAL;
        goto exit;
    }

    // Atomically check if already active and set to active if not
    // exchange() returns the old value and sets the new value atomically
    if(active.exchange(true) == true)
    {
        ret = -1;  // Was already active, cannot reserve
    }

    if(ret == 0)
    {
        buf_sz      = _buf_sz;
        buf_d       = _buf_d;
        tx_tensor   = _tx_tensor;
    }

exit:
    return ret;
}

int PhyChannel::reserveCellGroup()
{
    PhyDriverCtx* pdctx = StaticConversion<PhyDriverCtx>(pdh).get();
    int ret = 0;

    // Atomically check if already active and set to active if not
    // exchange() returns the old value and sets the new value atomically
    if(active.exchange(true) == true)
    {
        ret = -1;  // Was already active, cannot reserve
    }

exit:
    return ret;
}


int PhyChannel::release()
{
    active = false;
    setup_status = CH_SETUP_NOT_DONE;
    run_status = CH_RUN_NOT_DONE;
    return 0;
}

float PhyChannel::getGPUSetupTime() {
    return 1000.0f * GetCudaEventElapsedTime(start_setup, end_setup, __func__, getId());
}

float PhyChannel::getGPURunTime() {
    return 1000.0f * GetCudaEventElapsedTime(start_run, end_run, __func__, getId());
}


void PhyChannel::printGpuMemoryFootprint() {
    if (pCuphyTracker) {
        // TODO
        // cuphyMf.printGpuMemoryFootprint();
        //pCuphyTracker->printMemoryFootprint(); // for stdout
        // Can expand as needed
    }
}

void PhyChannel::updateMemoryTracker() {
    if (pCuphyTracker) {
        // TODO
        // cuphyMf.addGpuRegularSize(pCuphyTracker->getGpuRegularSize());
        //FIXME can easily expand to track host pinned memory etc.
    }
}

size_t PhyChannel::getGpuMemoryFootprint() {
    return (pCuphyTracker) ? pCuphyTracker->getGpuRegularSize() : 0;
}

void PhyChannel::checkPhyChannelObjCreationError(cuphyStatus_t errorStatus,std::string& phyChannelName)
{    
    MemtraceDisableScope md;
    std::string error_msg;
    if(errorStatus != CUPHY_STATUS_SUCCESS)
    {
        switch(errorStatus)
        {
            case CUPHY_STATUS_INTERNAL_ERROR:
                error_msg = phyChannelName + " returned CUPHY_STATUS_INTERNAL_ERROR";
                PHYDRIVER_THROW_EXCEPTIONS(-1, error_msg);
                break;
            case CUPHY_STATUS_NOT_SUPPORTED:
                error_msg = phyChannelName + " returned CUPHY_STATUS_NOT_SUPPORTED";
                PHYDRIVER_THROW_EXCEPTIONS(-1, error_msg);
                break;
            case CUPHY_STATUS_INVALID_ARGUMENT:
                error_msg = phyChannelName + " returned CUPHY_STATUS_INVALID_ARGUMENT";
                PHYDRIVER_THROW_EXCEPTIONS(-1, error_msg);
                break;
            case CUPHY_STATUS_ARCH_MISMATCH:
                error_msg = phyChannelName + " returned CUPHY_STATUS_ARCH_MISMATCH";
                PHYDRIVER_THROW_EXCEPTIONS(-1, error_msg);
                break;
            case CUPHY_STATUS_ALLOC_FAILED:
                error_msg = phyChannelName + " returned CUPHY_STATUS_ALLOC_FAILED";
                PHYDRIVER_THROW_EXCEPTIONS(-1, error_msg);
                break;
            case CUPHY_STATUS_SIZE_MISMATCH:
                error_msg = phyChannelName + " returned CUPHY_STATUS_SIZE_MISMATCH";
                PHYDRIVER_THROW_EXCEPTIONS(-1, error_msg);
                break;
            case CUPHY_STATUS_MEMCPY_ERROR:
                error_msg = phyChannelName + " returned CUPHY_STATUS_MEMCPY_ERROR";
                PHYDRIVER_THROW_EXCEPTIONS(-1, error_msg);
                break;
            case CUPHY_STATUS_INVALID_CONVERSION:
                error_msg = phyChannelName + " returned CUPHY_STATUS_INVALID_CONVERSION";
                PHYDRIVER_THROW_EXCEPTIONS(-1, error_msg);
                break;
            case CUPHY_STATUS_UNSUPPORTED_TYPE:
                error_msg = phyChannelName + " returned CUPHY_STATUS_UNSUPPORTED_TYPE";
                PHYDRIVER_THROW_EXCEPTIONS(-1, error_msg);
                break;
            case CUPHY_STATUS_UNSUPPORTED_LAYOUT:
                error_msg = phyChannelName + " returned CUPHY_STATUS_UNSUPPORTED_LAYOUT";
                PHYDRIVER_THROW_EXCEPTIONS(-1, error_msg);
                break;
            case CUPHY_STATUS_UNSUPPORTED_CONFIG:
                error_msg = phyChannelName + " returned CUPHY_STATUS_UNSUPPORTED_CONFIG";
                PHYDRIVER_THROW_EXCEPTIONS(-1, error_msg);
                break;
            case CUPHY_STATUS_UNSUPPORTED_ALIGNMENT:
                error_msg = phyChannelName + " returned CUPHY_STATUS_UNSUPPORTED_ALIGNMENT";
                PHYDRIVER_THROW_EXCEPTIONS(-1, error_msg);
                break;
            case CUPHY_STATUS_VALUE_OUT_OF_RANGE:
                error_msg = phyChannelName + " returned CUPHY_STATUS_VALUE_OUT_OF_RANGE";
                PHYDRIVER_THROW_EXCEPTIONS(-1, error_msg);
                break;
            case CUPHY_STATUS_UNSUPPORTED_RANK:
                error_msg = phyChannelName + " returned CUPHY_STATUS_UNSUPPORTED_RANK";
                PHYDRIVER_THROW_EXCEPTIONS(-1, error_msg);
                break;
            default:
                error_msg = phyChannelName + " returned CUPHY_STATUS_UNKNOWN";
                PHYDRIVER_THROW_EXCEPTIONS(-1, error_msg);
                break;
       }       
    }    
}

//redundant change
slot_command_api::pm_group* PhyChannel::getPmGroup()
{
    return aggr_slot_params->cgcmd->get_pm_group();
}

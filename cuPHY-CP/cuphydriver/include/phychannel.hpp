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

#ifndef PHY_CHANNEL_H
#define PHY_CHANNEL_H

#include <unordered_map>
#include <iostream>
#include <sstream>
#include <algorithm>
#include <atomic>
#include "gpudevice.hpp"
#include "time.hpp"
#include "constant.hpp"
#include "fh.hpp"
#include "mps.hpp"
#include <slot_command/slot_command.hpp>
#include "cuphydriver_api.hpp"
#include "cuphy_api.h"
#include "cuphy_hdf5.hpp"
#include "cuphy.hpp"
#include "hdf5hpp.hpp"
#include "dlbuffer.hpp"
#include "ulbuffer.hpp"
#include "order_entity.hpp"
#include "harq_pool.hpp"
#include "wavgcfo_pool.hpp"

using fp16_complex_t = __half2;
#if 0
//Need information from cuPHY
struct SSTxParams
{
    float    beta_pss;   /*!< scaling factor for PSS (primary synchronization signal) */
    float    beta_sss;   /*!< scaling factor for SSS (secondary synchronization signal), PBCH data and DMRS */
    uint16_t NID;        /*!< Physical cell id */
    uint16_t nHF;        /*!< Half frame index (0 or 1) */
    uint16_t Lmax;       /*!< Max number of ss blocks in pbch period (4,8,or 64) */
    uint16_t blockIndex; /*!< SS block index (0 - L_max) */
    uint16_t f0;         /*!< Index of initial ss subcarrier */
    uint16_t t0;         /*!< Index of initial ss ofdm symbol */
    uint16_t SFN;        /*!< frame index */
    uint16_t k_SSB;      /*!< SSB subcarrier offset */
    uint16_t nF;         /*!< number of subcarriers for one slot */
    uint16_t nT;         /*!< number of symbols for one slot */
};
#endif
/**
 * @brief Slot parameters container with ownership semantics
 *
 * Holds slot indication and PHY layer parameters for a single cell/channel mode (currently not used).
 */
struct slot_params
{
    struct slot_command_api::slot_indication si;               ///< Slot indication (SFN, slot number, tick)
    slot_command_api::phy_slot_params        slot_phy_prms;    ///< PHY layer parameters for the slot

    slot_params();

    explicit slot_params(slot_command_api::slot_indication& other,
                         slot_command_api::phy_slot_params* other_phy_params) :
        si(other),
        slot_phy_prms(std::move(*other_phy_params))
    {
    }
    slot_params(const slot_params&) = delete;
    slot_params& operator=(const slot_params&) = delete;

    slot_params(slot_params&& other) :
        si(std::move(other.si)),
        slot_phy_prms(std::move(other.slot_phy_prms))
    {
    }

    ~slot_params()
    {
        slot_phy_prms.reset();
    }
};

/**
 * @brief Slot parameters container with pointer semantics
 *
 * Holds pointers to externally-owned slot indication and cell group command (no ownership).
 * It is used for multi-cell, multi-channel processing.
 */
struct slot_params_aggr
{
    struct slot_command_api::slot_indication* si;              ///< Pointer to slot indication (SFN, slot number, tick)
    struct slot_command_api::cell_group_command * cgcmd;       ///< Pointer to cell group command containing all channel parameters

    void populate(slot_command_api::slot_indication* _si,
                struct slot_command_api::cell_group_command * _cgcmd)
    {
        si = _si;
        cgcmd = _cgcmd;
    }

    void cleanup() {
        si = nullptr;
        cgcmd = nullptr;
    }
};

/**
 * @brief Channel setup status indicators
 */
typedef enum ch_setup_status{
    CH_SETUP_NOT_DONE=0,        ///< Setup phase not yet started or in progress
    CH_SETUP_DONE_NO_ERROR,     ///< Setup phase completed successfully
    CH_SETUP_DONE_ERROR         ///< Setup phase completed with errors
}ch_setup_status_t;

/**
 * @brief Channel run status indicators
 */
typedef enum ch_run_status{
    CH_RUN_NOT_DONE=0,          ///< Run phase not yet started or in progress
    CH_RUN_DONE_NO_ERROR,       ///< Run phase completed successfully
    CH_RUN_DONE_ERROR           ///< Run phase completed with errors
}ch_run_status_t;

/**
 * @brief Base class for PHY channel processing
 *
 * Provides common infrastructure for all physical layer channel types (PUSCH, PDSCH, PRACH, etc.).
 * Manages GPU resources, CUDA streams, synchronization, slot parameters, and completion signaling.
 */
class PhyChannel {
public:
    PhyChannel(phydriver_handle _pdh, GpuDevice* _gDev, cell_id_t _cell_id, cudaStream_t _s_channel, MpsCtx * _mpsCtx);
    virtual ~PhyChannel();

    phydriver_handle    getPhyDriverHandler() const;   ///< Get handle to parent cuphydriver object
    uint64_t            getId() const;                  ///< Get unique channel instance identifier
    void                setActive();                    ///< Mark channel as active for processing
    void                setInactive();                  ///< Mark channel as inactive
    bool                isActive();                     ///< Check if channel is active

    /////////////////////////////////////////////////////////////
    //// Slot Info
    /////////////////////////////////////////////////////////////
    cell_id_t                       getCellId();                                    ///< Get cell identifier for this channel
    slot_command_api::oran_slot_ind getOranSlotIndication();                       ///< Get ORAN slot indication (SFN, slot, symbol timing)
    slot_command_api::oran_slot_ind getOranAggrSlotIndication();                   ///< Get ORAN slot indication for aggregated processing
    const slot_command_api::slot_info_t& getOranSlotInfo();                        ///< Get ORAN slot timing and configuration information
    int                             setDynParams(slot_params* curr_slot_params);    ///< Set dynamic slot parameters for non-aggregated processing
    int                             setDynAggrParams(slot_params_aggr* _aggr_slot_params);  ///< Set dynamic slot parameters for aggregated processing
    void cleanupDynParams();                                                        ///< Clean up and release slot parameters
    struct slot_command_api::cell_group_command * getCellGroupCommand();            ///< Get cell group command containing all channel parameters
    /////////////////////////////////////////////////////////////
    //// GPU interaction
    /////////////////////////////////////////////////////////////
    void        setCtx();                               ///< Set CUDA context for this channel
    void        configureCtx(MpsCtx * _mpsCtx);         ///< Configure CUDA context with specified MPS/green context
    MpsCtx *    getCtx();                               ///< Get MPS/green context for this channel
    uint8_t *   getBufD() const;                        ///< Get GPU device buffer pointer
    uint8_t *   getBufH() const;                        ///< Get host buffer pointer
    size_t      getBufSize() const;                     ///< Get buffer size in bytes
    float       getGPUSetupTime();                      ///< Get GPU setup phase execution time in milliseconds
    float       getGPURunTime();                        ///< Get GPU run phase execution time in milliseconds
    void        printGpuMemoryFootprint();              ///< Print GPU memory footprint to log
    void        updateMemoryTracker();                  ///< Update internal memory tracking counters
    size_t      getGpuMemoryFootprint();                ///< Get total GPU memory footprint in bytes
    slot_command_api::pm_group* getPmGroup();           ///< Get performance metrics group
    /////////////////////////////////////////////////////////////
    //// Generic methods
    /////////////////////////////////////////////////////////////

    //Block the phy stream based on host/device writes and events
    // (optional - can externally specify which stream to block for cases of multiple
    //  streams per phy context)
    int                 waitToStartCPU(uint32_t * wait_addr_h);                        ///< CPU polling wait on host memory flag
    int                 waitToStartGPU(uint32_t * wait_addr_d);                        ///< Insert GPU stream wait on device memory flag (channel's stream)
    int                 waitToStartGPU(uint32_t * wait_addr_d, cudaStream_t stream_);  ///< Insert GPU stream wait on device memory flag (specified stream)
    int                 waitToStartGPUEvent(cudaEvent_t event);                        ///< Insert GPU stream wait on CUDA event (channel's stream)
    int                 waitToStartGPUEvent(cudaEvent_t event, cudaStream_t stream_);  ///< Insert GPU stream wait on CUDA event (specified stream)

    //Completion signaling
    int                 signalRunCompletion();                                              ///< Signal run completion via host pinned buffer and GDR flag (optionally with host buffer write)
    int                 signalRunCompletionEvent(bool trigger_write_kernel);                ///< Signal run completion via CUDA event (optionally with host buffer write)
    int                 signalRunCompletionEvent(cudaStream_t stream_, bool trigger_write_kernel);  ///< Signal run completion via CUDA event (optionally with host buffer write)
    cudaEvent_t         getRunCompletionEvent() {return run_completion;};                   ///< Get CUDA event for run completion synchronization

    //Generic event waiting functions
    int                 waitEvent(cudaEvent_t event);                               ///< CPU blocking wait on CUDA event
    int                 waitEventNonBlocking(cudaEvent_t event);                    ///< CPU non-blocking query on CUDA event

    //CPU waiting function specific to start_run event
    int                 waitStartRunEvent();                                        ///< CPU blocking wait for start_run event
    int                 waitStartRunEventNonBlocking();                             ///< CPU non-blocking query for start_run event

    //CPU/GPU waiting functions specific to run completion
    int                 waitRunCompletion(int wait_ns);                             ///< CPU polling wait for run completion flag with timeout
    int                 waitRunCompletionEvent();                                   ///< CPU blocking wait for run_completion event
    int                 waitRunCompletionEventNonBlocking();                        ///< CPU non-blocking query for run_completion event
    int                 waitRunCompletionGPU(cudaStream_t stream_, MpsCtx * mpsCtx_);      ///< Insert GPU stream wait for run completion GDR flag
    int                 waitRunCompletionGPUEvent(cudaStream_t stream_, MpsCtx * mpsCtx_); ///< Insert GPU stream wait for run_completion event

    cudaStream_t        getStream() const;                                          ///< Get CUDA stream for this channel
    int                 reserve(uint8_t * _buf_d, uint8_t * _buf_h, size_t _buf_sz);   ///< Reserve GPU/host buffers for channel processing
    int                 reserve(uint8_t * _buf_d, size_t _buf_sz, cuphy::tensor_device* _tx_tensor); ///< Reserve GPU buffer with tensor for TX processing
    int                 reserveCellGroup();                                         ///< Reserve resources for cell group processing

    /////////////////////////////////////////////////////////////
    //// Virtual methods
    /////////////////////////////////////////////////////////////
    virtual int         cleanup();                                                      ///< Clean up channel resources
    virtual int         release();                                                      ///< Release channel resources

    void setSetupStatus(ch_setup_status_t status);                                      ///< Set setup phase status
    void setRunStatus(ch_run_status_t status);                                          ///< Set run phase status
    ch_setup_status_t getSetupStatus();                                                 ///< Get setup phase status
    ch_run_status_t getRunStatus();                                                     ///< Get run phase status
    void checkPhyChannelObjCreationError(cuphyStatus_t errorStatus,std::string& phyChannelName);  ///< Check and handle cuPHY object creation errors

    MemFoot              mf;                                                            ///< Memory footprint tracker for cuphydriver allocations
    MemFoot              cuphyMf;                                                       ///< Memory footprint tracker for cuPHY library allocations
    const cuphyMemoryFootprint* pCuphyTracker;                                          ///< Pointer to cuPHY memory footprint tracker

protected:
    phydriver_handle               pdh;                        ///< Handle to parent cuphydriver context
    uint64_t                       id;                         ///< Unique channel instance identifier (timestamp-based)
    slot_command_api::channel_type channel_type;               ///< Channel type (PUSCH, PDSCH, PRACH, PUCCH, etc.)
    std::string                    channel_name;               ///< Human-readable channel name
    std::atomic<bool>              active;                     ///< Atomic flag indicating if channel is active
    ch_setup_status_t                setup_status;             ///< Setup phase status (not done, done no error, done with error)
    ch_run_status_t                  run_status;               ///< Run phase status (not done, done no error, done with error)
    GpuDevice*                     gDev;                       ///< Pointer to GPU device manager
    cell_id_t                      cell_id;                    ///< Cell identifier for this channel
    struct slot_params *           current_slot_params;        ///< Current slot parameters (non-aggregated mode)
    struct slot_params_aggr *      aggr_slot_params;           ///< Aggregated slot parameters (aggregated mode)
    size_t                         buf_sz;                     ///< Buffer size in bytes
    uint8_t *                      buf_d;                      ///< GPU device buffer pointer
    uint8_t *                      buf_h;                      ///< Host buffer pointer
    cuphyCellStatPrm_t             cellStatPrm;                ///< cuPHY cell static parameters
    hdf5hpp::hdf5_file             fInput;                     ///< HDF5 file handle for input data (testing/debugging)
    MpsCtx*                        mpsCtx;                     ///< MPS or green context for GPU resource partitioning
    std::unique_ptr<host_buf>      channel_complete_h;         ///< Host pinned buffer for completion signaling
    std::unique_ptr<struct gpinned_buffer> channel_complete_gdr;  ///< GDR buffer for GPU-CPU completion signaling
    cudaStream_t                   s_channel;                  ///< CUDA stream for this channel
    cudaEvent_t                    start_setup;                ///< CUDA event marking setup phase start
    cudaEvent_t                    end_setup;                  ///< CUDA event marking setup phase end
    cudaEvent_t                    start_run;                  ///< CUDA event marking run phase start
    cudaEvent_t                    end_run;                    ///< CUDA event marking run phase end
    cudaEvent_t                    run_completion;             ///< CUDA event for run completion synchronization (not for timing)
    cuphy::tensor_device*          tx_tensor;                  ///< cuPHY device tensor for TX processing
    cuphyTracker_t                 cuphy_tracker;              ///< cuPHY performance tracker

    int cnt_used;                                              ///< Debug counter tracking number of times channel has been processed (wraps at 65536)

    int this_id;                                               ///< Instance ID for debugging/tracking purposes.
};


#endif

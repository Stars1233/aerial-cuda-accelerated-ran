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

#ifndef DLBUFFER_H
#define DLBUFFER_H

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
#include "hdf5hpp.hpp"
#include "pusch_rx.hpp"
#include "pdsch_tx.hpp"
#include "aerial-fh-driver/oran.hpp"

/**
 * @brief Downlink output buffer for storing and managing DL channel data
 *
 * General-purpose GPU memory buffer that stores frequency-domain IQ data from downlink
 * physical channels (PDSCH, PDCCH, PBCH, etc.) before fronthaul transmission.
 * Manages the entire DL pipeline including compression, packet preparation,
 * and transmission scheduling. Supports GPU-direct communication and modulation
 * compression for efficient fronthaul data transfer.
 */
class DLOutputBuffer {
public:
    /**
     * @brief Construct a DL output buffer
     *
     * @param _pdh     - Physical layer driver handle
     * @param _gDev    - GPU device for memory allocation
     * @param _cell_id - Cell identifier this buffer belongs to
     */
    DLOutputBuffer(phydriver_handle _pdh, GpuDevice* _gDev, cell_id_t _cell_id);
    ~DLOutputBuffer();

    uint64_t                getId() const;                                                             ///< Get unique buffer identifier
    void                    resetProcessingState();                                                    ///< Reset buffer state for reuse (clear active flags, reset timestamps)
    int                     reserve();                                                                 ///< Reserve buffer for exclusive use (atomic operation, returns 0 on success)
    void                    release();                                                                 ///< Release buffer for reuse by marking inactive
    void                    cleanup(cudaStream_t stream, MpsCtx * mpsCtx);                             ///< Cleanup buffer resources on specified stream
    cudaEvent_t*            cleanupEventRecord(cudaStream_t stream, MpsCtx * mpsCtx);                  ///< Record cleanup completion event and return pointer to event
    void                    waitCleanup(cudaStream_t stream, MpsCtx * mpsCtx);                         ///< Wait for cleanup event to complete on specified stream
    size_t                  getSizeFh() const;                                                         ///< Get fronthaul buffer size (for transmission)
    size_t                  getSize() const;                                                           ///< Get total buffer size in bytes
    uint8_t*                getBufD() const;                                                           ///< Get device (GPU) buffer pointer
    uint8_t*                getBufH() const;                                                           ///< Get host (CPU) buffer pointer (pinned memory)
    cuphy::tensor_device*   getTensor();                                                               ///< Get cuPHY tensor descriptor for this buffer
    struct umsg_fh_tx_msg&  getTxMsgContainer();                                                       ///< Get fronthaul TX message container for U-plane packets
    int                     txUplaneSlot();                                                            ///< Transmit U-plane data for this slot (returns 0 on success)
    int                     runCompression(const std::array<compression_params, NUM_USER_DATA_COMPRESSION_METHODS>& cparams_array, MpsCtx * mpsCtx, cudaStream_t stream); ///< Run compression kernel on GPU for all compression methods (returns 0 on success)
    mod_compression_params* getModCompressionConfig(){return mod_comp_params_per_cell;};              ///< Get modulation compression configuration
    mod_compression_params* getModCompressionTempConfig(){return mod_comp_config_temp.get();};        ///< Get temporary modulation compression configuration
    int                     waitEventNonBlocking(cudaEvent_t event);                                   ///< Non-blocking wait for CUDA event (returns 0 if complete, -1 if not ready)
    int                     waitCompression(cudaEvent_t event, bool for_compression_start=true);       ///< Wait for compression start or stop event (returns 0 on success)
    int                     waitCompressionStart();                                                    ///< Wait for compression to start (returns 0 on success)
    int                     waitCompressionStop();                                                     ///< Wait for compression to complete (returns 0 on success)
    int                     waitPrePrepareStop();                                                      ///< Wait for pre-preparation stage to complete (returns 0 on success)
    uint32_t*               getReadyFlag();                                                            ///< Get pointer to GPU-accessible ready flag for synchronization
    int                     setReadyFlag(cudaStream_t stream);                                         ///< Set ready flag on GPU stream (for GPU-CPU synchronization)
    float                   getPrepareExecutionTime1();                                                ///< GPU time between prepare start and prepare copy events in milliseconds (only active when GPU direct comm is enabled.)
    float                   getPrepareExecutionTime2();                                                ///< GPU time between prepare copy and pre-prepare stop events in milliseconds (only active when GPU direct comm is enabled.)
    float                   getPrepareExecutionTime3();                                                ///< GPU time between pre-prepare stop and prepare stop events in milliseconds (only active when GPU direct comm is enabled.)
    void                    getPrepareExecutionTimes(float& time1, float& time2, float& time3);        ///< Get all preparation stage execution times in milliseconds (only active when GPU direct comm is enabled.)
    float                   getChannelToCompressionGap();                                              ///< Get time gap between channel processing and compression start (ms)
    float                   getPrepareToCompressionGap();                                              ///< Get time gap between preparation and compression start (ms)
    float                   getCompressionExecutionTime();                                             ///< Get compression execution time in milliseconds

    uint8_t**               getPrbPtrs() const { return prb_ptrs.get(); };                            ///< Get array of PRB (Physical Resource Block) pointers for packet assembly
    cudaEvent_t             getPrepareStartEvt(void) const { return prepare_start_evt; }              ///< Get CUDA event marking packet preparation start
    cudaEvent_t             getPrepareCopyEvt(void) const { return prepare_copy_evt; }                ///< Get CUDA event marking memory copy during preparation
    cudaEvent_t             getPrepareStopEvt(void) const { return prepare_stop_evt; }                ///< Get CUDA event marking packet preparation completion
    cudaEvent_t             getPrePrepareStopEvt(void) const { return pre_prepare_stop_evt; }         ///< Get CUDA event marking pre-preparation stage completion
    cudaEvent_t             getAllChannelsDoneEvt(void) const { return all_channels_done_evt; };      ///< Get CUDA event marking all DL channels processing completion
    cudaEvent_t             getCompressionStartEvt(void) const { return compression_start_evt; };     ///< Get CUDA event marking compression start
    cudaEvent_t             getCompressionStopEvt(void) const { return compression_stop_evt; };       ///< Get CUDA event marking compression completion
    cudaEvent_t             getTxEndEvt(void) const { return tx_end_evt; }                            ///< Get CUDA event marking fronthaul transmission completion

    MemFoot                 mf;                                                                        ///< Memory footprint tracker for this buffer
    cell_id_t               cell_id;                                                                   ///< Cell identifier this buffer belongs to
protected:
    uint64_t                                 id;                                ///< Unique buffer identifier (timestamp-based)
    size_t                                   sz_tx;                             ///< Transmission buffer size (for fronthaul TX)
    size_t                                   sz_mr;                             ///< Memory region size allocated on GPU
    uint8_t  *                               addr_d;                            ///< Device (GPU) memory address for IQ data
    std::unique_ptr<host_buf>                addr_h;                            ///< Host (CPU) pinned memory buffer
    cuphy::tensor_device                     tx_tensor;                         ///< cuPHY tensor descriptor for the IQ data.
    cuphy::unique_device_ptr<cuFloatComplex> large_buffer;                      ///< Raw GPU buffer to hold max IQ sample data for a cell.
    std::atomic<bool>                        active;                            ///< Atomic flag indicating buffer is in use
    Mutex                                    mlock;                             ///< Mutex for thread-safe buffer operations
    phydriver_handle                         pdh;                               ///< Physical layer driver handle
    GpuDevice*                               gDev;                              ///< GPU device pointer for memory operations
    MpsCtx *                                 mpsCtx;                            ///< MPS (Multi-Process Service) context for GPU resource partitioning
    cudaEvent_t                              ev_cleanup;                        ///< CUDA event for cleanup synchronization
    struct gpinned_buffer*                   buffer_ready_gdr;                  ///< GPU Direct RDMA buffer for ready flag synchronization (required for GPU communication)
    t_ns                                     last_used;                         ///< Timestamp of last buffer usage (required for GPU communication scheduling)
    cuphy::unique_device_ptr<uint8_t*>       prb_ptrs;                          ///< Device memory array of PRB pointers for packet assembly
    cuphy::unique_pinned_ptr<mod_compression_params> mod_comp_config_temp;      ///< Temporary pinned memory for modulation compression configuration updates
    mod_compression_params*                  mod_comp_params_per_cell;          ///< Per-cell modulation compression parameters
    void*                                    mod_comp_config_device_mem;        ///< Device memory for modulation compression configuration

    cudaEvent_t                              prepare_start_evt;                 ///< CUDA event marking start of packet preparation stage
    cudaEvent_t                              prepare_copy_evt;                  ///< CUDA event marking memory copy completion during preparation
    cudaEvent_t                              prepare_stop_evt;                  ///< CUDA event marking completion of packet preparation stage
    cudaEvent_t                              pre_prepare_stop_evt;              ///< CUDA event marking completion of pre-preparation stage (before main prepare)
    cudaEvent_t                              all_channels_done_evt;             ///< CUDA event marking completion of all DL channel processing
    cudaEvent_t                              compression_start_evt;             ///< CUDA event marking start of compression
    cudaEvent_t                              compression_stop_evt;              ///< CUDA event marking completion of compression
    std::atomic<bool>                        compression_is_queued;             ///< Atomic flag indicating compression kernel is queued for execution
    cudaEvent_t                              tx_end_evt;                        ///< CUDA event marking completion of fronthaul transmission

    /**
     * List of U-plane packet items to be transmitted in the current slot.
     * Maximum size: Max eAxC (extended Antenna-Carrier) IDs * OFDM Symbols per slot
     */
    struct umsg_fh_tx_msg umsg_tx_list;                                         ///< Fronthaul transmission message list containing all U-plane packets for this slot
};

#endif

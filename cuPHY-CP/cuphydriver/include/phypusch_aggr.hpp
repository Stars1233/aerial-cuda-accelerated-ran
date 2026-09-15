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

#ifndef PHY_PUSCH_AGGR_CHANNEL_H
#define PHY_PUSCH_AGGR_CHANNEL_H

#include "phychannel.hpp"
#include "cell.hpp"
#include "data_lake.hpp"
#include <optional>

/**
 * @class PhyPuschAggr
 * @brief Handles PUSCH (Physical Uplink Shared Channel) reception and decoding processing.
 *
 * This class implements the uplink PUSCH data reception pipeline using the cuPHY library.
 * PUSCH carries uplink user data (UL-SCH transport blocks) and multiplexed UCI (HARQ-ACK, CSI).
 * Supports aggregated multi-cell processing with two-phase setup/run, sub-slot (early HARQ) processing,
 * CUDA graph/stream execution modes, and HARQ buffer pool management.
 */
class PhyPuschAggr : public PhyChannel {
public:

    /**
     * @brief Constructs a PhyPuschAggr object.
     *
     * Initializes the PUSCH channel with the given cuphydriver handle, GPU device, CUDA streams,
     * and MPS context. Sets up memory footprint tracking, allocates output buffers for decoded
     * transport blocks, CRCs, UCI payloads, quality metrics, and initializes HARQ pool management.
     *
     * @param _pdh cuphydriver handle
     * @param _gDev GPU device struct pointer
     * @param _s_channels Array of CUDA streams (uses first stream) for asynchronous GPU operations
     * @param _mpsCtx MPS context for GPU resource partitioning
     */
    PhyPuschAggr(
        phydriver_handle _pdh,
        GpuDevice*       _gDev,
        cudaStream_t*    _s_channels,
        MpsCtx *        _mpsCtx
    );

    /**
     * @brief Destructor for PhyPuschAggr.
     *
     * Frees allocated buffers, destroys CUDA events, and destroys the cuPHY PUSCH RX handle.
     */
    ~PhyPuschAggr();

    ////////////////////////////////////////////////////////////
    /// Module generic
    ////////////////////////////////////////////////////////////
    /**
     * @brief Configures the PUSCH reception for the current slot (two-phase setup).
     *
     * Phase 1: Sets up UE parameters and queries HARQ buffer sizes from cuPHY.
     * Phase 2: Allocates and configures HARQ buffers after sizes are known.
     * Sets up input buffer pointers from UL order kernel (st1 buffers) and output buffer configuration.
     * Calls cuphySetupPuschRx() for both phases.
     *
     * @param aggr_cell_list List of Cell objects being processed in this slot
     * @param aggr_ulbuf_st1 List of UL input buffers (st1/o1) containing ordered PUSCH samples
     * @param phase1_stream CUDA stream for phase 1 setup operations
     * @param phase2_stream CUDA stream for phase 2 setup operations
     * @return 0 on success, -1 on failure (setup error)
     */
    int setup(const std::vector<Cell *>& aggr_cell_list, const std::vector<ULInputBuffer *>& aggr_ulbuf_st1, cudaStream_t phase1_stream, cudaStream_t phase2_stream);
    
    /**
     * @brief Executes the PUSCH decoding processing on GPU (phase-based).
     *
     * Invokes cuphyRunPuschRx() with the specified run phase to perform channel estimation,
     * equalization, demodulation, and LDPC decoding. Supports sub-slot (early HARQ after symbols 0-3)
     * and full-slot processing. Uses either CUDA graphs or streams based on configuration.
     *
     * @param runPhase PUSCH_RUN_SUB_SLOT (early HARQ), PUSCH_RUN_POST_SUB_SLOT (remaining symbols), or PUSCH_RUN_FULL_SLOT
     * @return 0 on success, -1 on failure (run error)
     */
    int run(cuphyPuschRunPhase_t runPhase);
    
    /**
     * @brief Gets GPU execution time for sub-slot (early HARQ) phase.
     *
     * Returns elapsed time between waitCompletedSubSlotEvent and subSlotCompletedEvent.
     * These events are recorded internally by cuPHY during sub-slot processing.
     *
     * @return Elapsed time in milliseconds for sub-slot phase
     */
    float getGPURunSubSlotTime();
    
    /**
     * @brief Gets GPU execution time for post-sub-slot (remaining symbols) phase.
     *
     * Returns elapsed time between waitCompletedFullSlotEvent and end_run_ph1.
     * waitCompletedFullSlotEvent is recorded internally by cuPHY.
     *
     * @return Elapsed time in milliseconds for post-sub-slot phase
     */
    float getGPURunPostSubSlotTime();
    
    /**
     * @brief Gets GPU idle time gap between sub-slot and post-sub-slot phases.
     *
     * Returns elapsed time between end_run_ph1 and start_run_ph2 CUDA events.
     *
     * @return Elapsed time in milliseconds for the gap between phases
     */
    float getGPURunGapTime();
    
    /**
     * @brief Gets GPU execution time for a specific PUSCH run phase.
     *
     * Returns elapsed time for the specified run phase based on CUDA event measurements:
     * - PUSCH_RUN_SUB_SLOT_PROC: elapsed time from start_run_ph1 to end_run_ph1
     * - PUSCH_RUN_FULL_SLOT_COPY: elapsed time from start_run_ph2 to end_run_ph2
     * - PUSCH_RUN_ALL_PHASES: elapsed time from start_run to end_run
     *
     * @param runPhase PUSCH_RUN_SUB_SLOT_PROC, PUSCH_RUN_FULL_SLOT_COPY, or PUSCH_RUN_ALL_PHASES
     * @return Elapsed time in milliseconds for the specified phase, or 0 for unrecognized phases
     */
    float getGPUPhaseRunTime(cuphyPuschRunPhase_t runPhase);
    
    /**
     * @brief Checks if early HARQ-ACK is present in the current slot.
     *
     * Inline function that returns whether early HARQ-ACK bits were detected on symbols 0-3.
     *
     * @return true if early HARQ is present, false otherwise
     */
    bool isEarlyHarqPresent() {return getPuschDynParams()->pDataOut->isEarlyHarqPresent == 1;};
    
    /**
     * @brief Checks if front-loaded DMRS is present in the current slot.
     *
     * Inline function that returns whether front-loaded DMRS symbols were configured.
     *
     * @return true if front-loaded DMRS is present, false otherwise
     */
    bool isFrontLoadedDmrsPresent() {return getPuschDynParams()->pDataOut->isFrontLoadedDmrsPresent == 1;};
    
    /**
     * @brief Cleans up PUSCH resources after slot processing.
     *
     * Returns HARQ buffers to the pool for reuse. Called at the end of each slot.
     *
     * @return 0 on success
     */
    int cleanup();
    
    /**
     * @brief Validates PUSCH output against reference data (for testing/debugging).
     *
     * Compares decoded transport blocks and CRCs against expected values loaded from test vectors.
     * Handles cell timeouts and GPU early HARQ timeouts. Optionally captures PCAP data for debugging.
     *
     * @param cell_timeout_list Array indicating which cells timed out (1=timeout, 0=ok)
     * @param gpu_early_harq_timeout true if GPU early HARQ wait kernel timed out
     * @param aggr_ulbuf_pcap_capture List of buffers for PCAP capture of received data
     * @param aggr_ulbuf_pcap_capture_ts List of buffers for PCAP capture with timestamps
     * @return 0 on success, error code otherwise
     */
    int validate(std::array<uint8_t,UL_MAX_CELLS_PER_SLOT>& cell_timeout_list,bool gpu_early_harq_timeout, const std::vector<ULInputBuffer *>& aggr_ulbuf_pcap_capture,  const std::vector<ULInputBuffer *>& aggr_ulbuf_pcap_capture_ts);
    
    /**
     * @brief Post-processing callback after PUSCH decoding completes.
     *
     * Invokes registered UL callback (data_cb_fn) to report decoded transport blocks, CRCs, UCI,
     * and quality metrics to upper layers (MAC). Handles cell timeouts and updates per-cell metrics.
     *
     * @param cell_timeout_list Array indicating which cells timed out (1=timeout, 0=ok)
     * @param gpu_early_harq_timeout true if GPU early HARQ wait kernel timed out
     * @return 0 on success, error code otherwise
     */
    int callback(std::array<uint8_t,UL_MAX_CELLS_PER_SLOT>& cell_timeout_list,bool gpu_early_harq_timeout);
    
    /**
     * @brief Updates physical cell ID in static parameters.
     *
     * Used during cell reconfiguration to update the PHY cell ID without recreating
     * the entire cuPHY object.
     *
     * @param phyCellId_old Old physical cell ID to replace
     * @param phyCellId_new New physical cell ID
     */
    void updatePhyCellId(uint16_t,uint16_t);
    
    /**
     * @brief Sets the work cancellation flag to abort ongoing GPU processing.
     *
     * Signals GPU kernels to exit early via the bWorkCancelInfo buffer. Used for
     * graceful shutdown or timeout handling.
     *
     * @param flag_value true to cancel work, false to clear cancellation flag (default: true)
     */
    void setWorkCancelFlag(bool flag_value=true);

    ////////////////////////////////////////////////////////////
    /// Module specific
    ////////////////////////////////////////////////////////////
    
    /**
     * @brief Validates PUSCH output (stub implementation).
     *
     * Currently returns false immediately. Provided for API consistency.
     *
     * @return false (unimplemented)
     */
    bool                            validateOutput();
    
    /**
     * @brief Gets GPU execution time for setup phase 1.
     *
     * Returns elapsed time between start_setup_ph1 and end_setup_ph1 CUDA events.
     *
     * @return Elapsed time in milliseconds for setup phase 1
     */
    float                           getGPUSetupPh1Time();
    
    /**
     * @brief Gets GPU execution time for setup phase 2.
     *
     * Returns elapsed time between start_setup_ph2 and end_setup_ph2 CUDA events.
     *
     * @return Elapsed time in milliseconds for setup phase 2
     */
    float                           getGPUSetupPh2Time();
    
    /**
     * @brief Gets GPU execution time for CRC calculation.
     *
     * Returns elapsed time between start_crc and end_crc CUDA events.
     *
     * @return Elapsed time in milliseconds for CRC calculation
     */
    float                           getGPUCrcTime();
    
    /**
     * @brief Gets minimum PRB index (unimplemented).
     *
     * Function declared but not implemented for PUSCH. Returns 0 if called.
     *
     * @return 0 (unimplemented)
     */
    uint16_t                        getMinPrb();
    
    /**
     * @brief Gets maximum PRB index (unimplemented).
     *
     * Function declared but not implemented for PUSCH. Returns 0 if called.
     *
     * @return 0 (unimplemented)
     */
    uint16_t                        getMaxPrb();
    
    /**
     * @brief Creates the cuPHY PUSCH RX object.
     *
     * Initializes the cuPHY PUSCH reception handle, builds the cell list with static
     * parameters (antennas, PRBs, numerology, scrambling sequences). Called during driver initialization.
     *
     * @return 0 on success, non-zero error code on failure
     */
    int                             createPhyObj();
    
    /**
     * @brief Retrieves the PUSCH dynamic parameters for the current slot.
     *
     * Extracts the PUSCH parameters (UE configs, TBs, modulation) from the aggregated slot
     * command structure for use in setup().
     *
     * @return Pointer to the PUSCH parameters for the current slot
     */
    slot_command_api::pusch_params* getDynParams();
    
    /**
     * @brief Retrieves the cuPHY PUSCH dynamic parameters structure.
     *
     * Returns pointer to the cuPHY-specific dynamic parameters used in cuphyRunPuschRx().
     *
     * @return Pointer to cuphyPuschDynPrms_t structure
     */
    cuphyPuschDynPrms_t *           getPuschDynParams();
    
    /**
     * @brief Retrieves the cuPHY PUSCH static parameters structure.
     *
     * Returns pointer to the cuPHY-specific static parameters used in cuphyCreatePuschRx().
     *
     * @return Pointer to cuphyPuschStatPrms_t structure
     */
    cuphyPuschStatPrms_t*           getPuschStatParams();
    
    /**
     * @brief Loads static parameters from HDF5 test vector file.
     *
     * Debug utility function that loads PUSCH static parameters from test vectors.
     *
     * @param tv_h5 Path to HDF5 test vector file
     * @param cell_idx Cell index within the test vector
     */
    void                            tvStatPrms(const char* tv_h5, int cell_idx);
    
    /**
     * @brief Retrieves the HARQ buffer pool manager.
     *
     * Returns pointer to the HARQ pool manager for accessing HARQ buffers.
     *
     * @return Pointer to HarqPoolManager
     */
    HarqPoolManager *               getHarqPoolManager();
    
    /**
     * @brief Gets GPU-side completion flag pointer for symbol ordering.
     *
     * Returns GPU memory address of the completion flag for the specified OFDM symbol index.
     * Used by UL order kernel to signal symbol availability for sub-slot processing.
     *
     * @param sym_idx OFDM symbol index (0-13)
     * @return Pointer to GPU flag (uint32_t*)
     */
    uint32_t*                       getSymOrderSigDoneGpuFlag(int sym_idx);
    
    /**
     * @brief Gets CPU-side completion flag pointer for symbol ordering.
     *
     * Returns host memory address of the completion flag for the specified OFDM symbol index.
     * Used by CPU to poll symbol availability for sub-slot processing.
     *
     * @param sym_idx OFDM symbol index (0-13)
     * @return Pointer to host flag (uint32_t*)
     */
    uint32_t*                       getSymOrderSigDoneCpuFlag(int sym_idx);
    
    /**
     * @brief Gets pre-early HARQ wait kernel status.
     *
     * Returns the status byte indicating whether the wait kernel before early HARQ processing
     * completed successfully or timed out. Set by cuPHY GPU kernel.
     *
     * @return 0=done (PUSCH_RX_WAIT_KERNEL_STATUS_DONE), 1=timeout (PUSCH_RX_WAIT_KERNEL_STATUS_TIMEOUT)
     */
    uint8_t                         getPreEarlyHarqWaitKernelStatus();
    
    /**
     * @brief Gets post-early HARQ wait kernel status.
     *
     * Returns the status byte indicating whether the wait kernel after early HARQ processing
     * completed successfully or timed out. Set by cuPHY GPU kernel.
     *
     * @return 0=done (PUSCH_RX_WAIT_KERNEL_STATUS_DONE), 1=timeout (PUSCH_RX_WAIT_KERNEL_STATUS_TIMEOUT)
     */
    uint8_t                         getPostEarlyHarqWaitKernelStatus();
    
    /**
     * @brief Clears UCI detection status flags to default values.
     *
     * Resets HARQ, CSI Part1, CSI Part2 detection status flags to 2 (not detected),
     * and optionally clears UCI CRC flags. Called before each slot processing.
     *
     * @param harq_status_only true to clear only HARQ status, false to clear all UCI flags
     * @return 0 on success
     */
    int                             clearUciFlags(bool harq_status_only);

#ifdef AERIAL_METRICS
    /**
     * @brief Per-cell PUSCH metrics for monitoring (when AERIAL_METRICS is enabled).
     *
     * Tracks number of transport blocks, TB sizes, and CRC results per cell.
     */
    typedef struct _cuphyPuschCellAerialMetrics final
    {
         uint32_t   nTBs{};      ///< Number of transport blocks processed
         uint32_t   tbSize{};    ///< Total transport block size in bytes
         uint32_t   nTbCrc{};    ///< Number of TB CRC passes
    } cuphyPuschCellAerialMetrics_t;
#endif

protected:
    ///< Tensor descriptors for PUSCH received sample data per cell
    cuphy::tensor_desc pusch_data_rx_desc[UL_MAX_CELLS_PER_SLOT];

    ////////////////////////////////////////////
    //// cuPHY aggregate
    ////////////////////////////////////////////
    ///< List of cell IDs being processed in the current slot
    std::vector<cell_id_t> cell_id_list;
    
    ///< Static parameters for cuphyCreatePuschRx() (cell list, antenna configuration)
    cuphyPuschStatPrms_t static_params{};
    
    ///< Vector of per-cell static parameters (physical cell ID, antennas, numerology)
    std::vector<cuphyCellStatPrm_t> static_params_cell;
    
    ///< Dynamic parameters for cuphyRunPuschRx() (stream, input/output pointers, per-slot configs)
    cuphyPuschDynPrms_t dyn_params{};

    ////////////////////////////////////////////
    //// cuPHY specific
    ////////////////////////////////////////////
    ///< Vector of transport block parameters for the current slot (per-UE TB configs)
    std::vector<tb_pars>               currentTbsPrmsArray;
    
    ///< gNodeB parameters (cell configuration)
    gnb_pars                           BBUPrms;
    
    ///< Current slot number for processing
    uint32_t                           slotNumber;
    
    /*!< Processing mode bitmask:
     *   - Bit 0: PUSCH_PROC_MODE_FULL_SLOT_GRAPHS (0x1) for CUDA graphs, or 0x0 for streams
     *   - Bit 1: PUSCH_PROC_MODE_SUB_SLOT (0x2) for early HARQ sub-slot processing
     *   - Combined: (PUSCH_PROC_MODE_FULL_SLOT_GRAPHS | PUSCH_PROC_MODE_SUB_SLOT) = 0x3
     */
    uint64_t                           procModeBmsk;
    
    ///< cuPHY PUSCH reception handle (created via cuphyCreatePuschRx())
    cuphyPuschRxHndl_t                 puschRxHndl;
    
    ///< Tensor parameter for DFT frequency domain weights (standard size)
    cuphyTensorPrm_t                   tPrmWFreq;
    
    ///< Tensor parameter for DFT frequency domain weights
    cuphyTensorPrm_t                   tPrmWFreq4;
    
    ///< Tensor parameter for DFT frequency domain weights (small size)
    cuphyTensorPrm_t                   tPrmWFreqSmall;
    
    ///< Tensor parameter for cyclic shift sequence
    cuphyTensorPrm_t                   tPrmShiftSeq;
    
    ///< Tensor parameter for unshift sequence
    cuphyTensorPrm_t                   tPrmUnShiftSeq;
    
    ///< Tensor parameter for cyclic shift sequence
    cuphyTensorPrm_t                   tPrmShiftSeq4;
    
    ///< Tensor parameter for unshift sequence
    cuphyTensorPrm_t                   tPrmUnShiftSeq4;
    
    ///< cuPHY input data structure (tensor descriptors and pointers to received samples)
    cuphyPuschDataIn_t                 DataIn;
    
    ///< cuPHY output data structure (pointers to TBs, CRCs, UCI, quality metrics)
    cuphyPuschDataOut_t                DataOut;
    
    ///< cuPHY input/output data structure (HARQ buffer pointers)
    cuphyPuschDataInOut_t              DataInOut;
    
    ///< Cell group dynamic parameters (UE groups, per-UE configs for all cells)
    cuphyPuschCellGrpDynPrm_t          cellGrpDynPrm;
    
    ///< Tensor parameter for received data (used for setup)
    cuphyTensorPrm_t                   tDataRx;
    
    ///< cuPHY PUSCH batch parameter handle (unused)
    cuphyPuschBatchPrmHndl_t           batchPrmHndl;
    
    ///< Vector of per-cell static PUSCH parameters (antennas, PRBs, numerology)
    std::vector<cuphyPuschCellStatPrm_t> puschCellStatPrmsVec;
    
    ///< Status output from cuphyRunPuschRx() (success/failure, cell/UE index on error)
    cuphyPuschStatusOut_t              statusOut;

    ///< Vector of UE group parameters (per-cell UE grouping for batching)
    std::vector<cuphyPuschUeGrpPrm_t>  ueGrpPrmVec;
    
    ///< Vector of per-UE parameters (modulation, layers, PRB allocation, TBS)
    std::vector<cuphyPuschUePrm_t>     uePrmsVec;
    
    ///< Vector of UE index buffers per cell (for UE group organization)
    std::vector<std::vector<uint16_t>> ueIdxBuffer;
    
    ///< GPU tensor for DFT frequency domain weights
    cuphy::tensor_device               tWFreq;
    
    ///< GPU tensor for cyclic shift sequence
    cuphy::tensor_device               tShiftSeq;
    
    ///< GPU tensor for unshift sequence
    cuphy::tensor_device               tUnShiftSeq;
    
    ///< GPU tensor for DFT frequency domain weights
    cuphy::tensor_device               tWFreq4;
    
    ///< GPU tensor for DFT frequency domain weights (small size)
    cuphy::tensor_device               tWFreqSmall;
    
    ///< GPU tensor for cyclic shift sequence
    cuphy::tensor_device               tShiftSeq4;
    
    ///< GPU tensor for unshift sequence
    cuphy::tensor_device               tUnShiftSeq4;
    
    ///< Static debug parameters (descrambling, HDF5 logging, forced CSI-2 bits)
    cuphyPuschStatDbgPrms_t            static_params_dbg;
    
    ///< Dynamic debug parameters (HDF5 output file)
    cuphyPuschDynDbgPrms_t             dyn_params_dbg;
    
    ///< HDF5 file handle for debug output (test vector validation)
    std::unique_ptr<hdf5hpp::hdf5_file> debugFileH;
    
    ///< Host pinned buffer for code block CRC start offsets (per-TB CB boundaries)
    cuphy::buffer<uint32_t, cuphy::pinned_alloc>            bStartOffsetsCbCrc;
    
    ///< Host pinned buffer for transport block CRC start offsets
    cuphy::buffer<uint32_t, cuphy::pinned_alloc>            bStartOffsetsTbCrc;
    
    ///< Host pinned buffer for transport block payload start offsets
    cuphy::buffer<uint32_t, cuphy::pinned_alloc>            bStartOffsetsTbPayload;
    
    ///< Total number of code block CRCs allocated (across all TBs and cells)
    uint32_t                                                totNumCbCrc;
    
    ///< Total number of transport block CRCs allocated (across all UEs and cells)
    uint32_t                                                totNumTbCrc;
    
    ///< Total number of transport block payload bytes allocated
    uint32_t                                                totNumTbByte;
    
    ///< Host pinned buffer for received sample data (full size when datalake enabled, size 0 otherwise)
    cuphy::buffer<__half2, cuphy::pinned_alloc>             bDataRx;
    
    ///< Host pinned buffer for code block CRC results (0=pass, 1=fail per CB)
    cuphy::buffer<uint32_t, cuphy::pinned_alloc>            bCbCrcs;
    
    ///< Host pinned buffer for transport block CRC results (0=pass, 1=fail per TB)
    cuphy::buffer<uint32_t, cuphy::pinned_alloc>            bTbCrcs;
    
    ///< Host pinned buffer for decoded transport block payloads
    cuphy::buffer<uint8_t, cuphy::pinned_alloc>             bTbPayloads;
    
    ///< Host pinned buffer for Timing Advance (TA) estimates per TB
    cuphy::buffer<float, cuphy::pinned_alloc>               bTaEst;
    
    ///< Host pinned tensor for transport block bytes (test vector reference data, comparison code disabled)
    cuphy::tensor_pinned                                    tTbBytes;
    
    ///< Host pinned buffer for decoded UCI on PUSCH payloads (HARQ-ACK, CSI bits)
    cuphy::buffer<uint8_t, cuphy::pinned_alloc>             bUciPayloads;
    
    ///< Host pinned buffer for UCI CRC status flags (0=pass, 1=fail)
    cuphy::buffer<uint8_t, cuphy::pinned_alloc>            bUciCrcFlags;
    
    ///< Host pinned buffer for number of CSI Part2 bits per UCI segment
    cuphy::buffer<uint16_t, cuphy::pinned_alloc>            bNumCsi2Bits;
    
    ///< Host pinned buffer for UCI on PUSCH output offsets (payload start positions)
    cuphy::buffer<cuphyUciOnPuschOutOffsets_t, cuphy::pinned_alloc> bUciOnPuschOutOffsets;
    
    ///< Host pinned buffer for Received Signal Strength Indicator (RSSI) per TB
    cuphy::buffer<float, cuphy::pinned_alloc>               bRssi;
    
    ///< Host pinned buffer for Reference Signal Received Power (RSRP) per TB
    cuphy::buffer<float, cuphy::pinned_alloc>               bRsrp;
    
    ///< Host pinned buffer for Signal-to-Interference-plus-Noise Ratio (SINR) per TB
    cuphy::buffer<float, cuphy::pinned_alloc>               bSinr;
    
    ///< Host pinned buffer for Carrier Frequency Offset (CFO) estimates in Hz per TB
    cuphy::buffer<float, cuphy::pinned_alloc>               bCfo;
    
    ///< Host pinned buffer for noise/interference variance per TB
    cuphy::buffer<float, cuphy::pinned_alloc>               bNoiseIntfVar;
    
    ///< Host pinned buffer for HARQ-ACK detection status per UCI (0=detected, 1=DTX, 2=not detected)
    cuphy::buffer<uint8_t, cuphy::pinned_alloc>             bHarqDetectionStatus;
    
    ///< Host pinned buffer for CSI Part1 detection status per UCI (0=detected, 1=DTX, 2=not detected)
    cuphy::buffer<uint8_t, cuphy::pinned_alloc>             bCsiP1DetectionStatus;
    
    ///< Host pinned buffer for CSI Part2 detection status per UCI (0=detected, 1=DTX, 2=not detected)
    cuphy::buffer<uint8_t, cuphy::pinned_alloc>             bCsiP2DetectionStatus;

    // Channel estimates buffers - Currently only first UE group
    cuphy::buffer<float2, cuphy::pinned_alloc>              bChannelEsts;      // Using float2 to match cuPHY tensor type
    cuphy::buffer<uint32_t, cuphy::pinned_alloc>            bChannelEstSizes;  // Size in elements for each UE group

    ///< Host pinned buffer for work cancellation flag (signals GPU kernels to abort)
    cuphy::buffer<uint8_t, cuphy::pinned_alloc>             bWorkCancelInfo;    

    ///< Size of UCI payload buffer in bytes
    int nUciPayloadBytes;
    
    ///< Number of UCI segments (for segmented UCI decoding)
    int nUciSegs;
    
    ///< Number of UEs with UCI on PUSCH
    int nUciUes;
    
    ///< Number of CSI Part2 bits
    int nCsi2Bits;
 
    ///< Start timestamp for PUSCH reception processing (nanoseconds)
    t_ns start_t_rx;
    
    ///< End timestamp for PUSCH reception processing (nanoseconds)
    t_ns end_t_rx;
    
    ///< UL input buffer pointer (st1 buffer containing ordered PUSCH samples)
    ULInputBuffer *                                         ulbuf;
    
    ///< HARQ buffer pool manager (manages HARQ buffer allocation and recycling)
    HarqPoolManager *                                       hb_pool_m;
    
    ///< WAvgCfo buffer pool manager (manages WAvgCfo buffer allocation and recycling)
    WAvgCfoPoolManager *                                    wavgcfo_pool_m;
    
    ///< Vector of HARQ buffer sizes in bytes (per-TB HARQ buffer requirements)
    std::vector<uint32_t>                                   bHarqBufferSizeInBytes;
    
    ///< Array of HARQ buffer pointers for the current slot (per-TB HARQ buffers)
    std::array<HarqBuffer *,MAX_N_TBS_PER_CELL_GROUP_SUPPORTED>    hb_slot;
    
    ///< HARQ buffer counter (tracks number of HARQ buffers used in current slot)
    int                                                     hq_buffer_counter;

    ///< GPU/host pinned buffer array for symbol ordering completion flags (sub-slot processing)
    std::unique_ptr<gpinned_buffer>                         sym_ord_done_sig_arr;
    
    ////////////////////////////////////////////
    //// Wait Kernel Status
    ////////////////////////////////////////////
    ///< Pointer to pre-early HARQ wait kernel status flag (0=done, 1=timeout)
    uint8_t* pPreEarlyHarqWaitKernelStatus;
    
    ///< Pointer to post-early HARQ wait kernel status flag (0=done, 1=timeout)
    uint8_t* pPostEarlyHarqWaitKernelStatus;

    ////////////////////////////////////////////
    //// CUDA Events
    ////////////////////////////////////////////
    ///< CUDA event marking start of setup phase 1
    cudaEvent_t start_setup_ph1;
    
    ///< CUDA event marking end of setup phase 1
    cudaEvent_t end_setup_ph1;
    
    ///< CUDA event marking start of setup phase 2
    cudaEvent_t start_setup_ph2;
    
    ///< CUDA event marking end of setup phase 2
    cudaEvent_t end_setup_ph2;
    
    ///< CUDA event marking start of CRC calculation
    cudaEvent_t start_crc;
    
    ///< CUDA event marking end of CRC calculation
    cudaEvent_t end_crc;
    
    ///< CUDA event recorded by cuPHY when sub-slot processing completes
    cudaEvent_t subSlotCompletedEvent;
    
    ///< CUDA event recorded by cuPHY at start of sub-slot wait (passed to cuPHY in static_params)
    cudaEvent_t waitCompletedSubSlotEvent;
    
    ///< CUDA event recorded by cuPHY at start of full-slot wait (passed to cuPHY in static_params)
    cudaEvent_t waitCompletedFullSlotEvent;
    
    ///< CUDA event recorded by cuPHY at the completion of processing UCI-on-PUSCH
    cudaEvent_t uciOnPuschCompletedEvent;
    
    ///< CUDA event marking start of run phase 1 (sub-slot/early HARQ)
    cudaEvent_t start_run_ph1;
    
    ///< CUDA event marking end of run phase 1 (sub-slot/early HARQ)
    cudaEvent_t end_run_ph1;
    
    ///< CUDA event marking start of run phase 2 (post-sub-slot/remaining symbols)
    cudaEvent_t start_run_ph2;
    
    ///< CUDA event marking end of run phase 2 (post-sub-slot/remaining symbols)
    cudaEvent_t end_run_ph2;

    ////////////////////////////////////////////
    //// Validation
    ////////////////////////////////////////////
    ///< Number of code block CRC errors (for test vector validation)
    uint32_t nCbCrcErrors;
    
    ///< Number of transport block CRC errors (for test vector validation)
    uint32_t nTbCrcErrors;
    
    ///< Number of transport block payload byte errors (for test vector validation)
    uint32_t nTbByteErrors;
    
    ///< Timeout status flag (true if processing timeout occurred)
    bool     okTimeout;

    ///< Flag indicating whether test vectors are being read for validation
    bool read_tv;

    ////////////////////////////////////////////
    //// Debug
    ////////////////////////////////////////////
    ///< Flag indicating whether HDF5 debug dump has been performed
    bool h5dumped = false;

    ////////////////////////////////////////////
    //// CSI2 Maps
    ////////////////////////////////////////////
    ///< GPU buffer for CSI Part2 mapping table
    cuphy::unique_device_ptr<uint16_t>                      csi2MapGpuBuffer;
    
    ///< GPU buffer for CSI Part2 mapping parameters
    cuphy::unique_device_ptr<cuphyCsi2MapPrm_t>             csi2MapParamsGpuBuffer;

    ReleasedHarqBufferInfo released_harq_buffer_info; ///< Tracking information for released HARQ buffers

    ////////////////////////////////////////////
    //// Zero-copy TB buffer support (GT-11677)
    ////////////////////////////////////////////
    ///< Pre-allocated nvIPC message buffer from L2 adapter (used as D2H destination)
    std::optional<slot_command_api::ul_output_msg_buffer> m_extMsgBuffer{};
    ///< Saved original pTbPayloads pointer for restoration after callback
    uint8_t* m_origTbPayloadsPtr{nullptr};

    /**
     * @brief Attempts to acquire an external buffer from L2 for zero-copy D2H.
     *
     * Calls the registered alloc_fn callback to get a pre-allocated nvIPC data
     * buffer. If successful and the buffer is large enough, overrides
     * DataOut.pTbPayloads so that cudaMemcpyAsync writes directly into the IPC
     * buffer, eliminating the subsequent std::copy in send_rx_data_indication.
     *
     * Falls back silently to the internal bTbPayloads buffer on failure.
     */
    void acquireExternalTbBuffer();

    /**
     * @brief Restores DataOut.pTbPayloads to the internal buffer after callback.
     */
    void releaseExternalTbBuffer();

    /**
     * @brief RAII scope guard that releases the external TB buffer on scope exit
     *        unless explicitly dismissed (e.g. on the success path).
     */
    class ExtTbBufferGuard
    {
    public:
        explicit ExtTbBufferGuard(PhyPuschAggr& owner) : m_owner(owner) {}
        ~ExtTbBufferGuard() { if (!m_dismissed) m_owner.releaseExternalTbBuffer(); }

        void dismiss() noexcept { m_dismissed = true; }

        ExtTbBufferGuard(const ExtTbBufferGuard&) = delete;
        ExtTbBufferGuard& operator=(const ExtTbBufferGuard&) = delete;

    private:
        PhyPuschAggr& m_owner;
        bool m_dismissed{false};
    };

#ifdef AERIAL_METRICS
    ////////////////////////////////////////////
    //// Cell Metric
    ////////////////////////////////////////////
    ///< Per-cell PUSCH metrics information (TB counts, sizes, CRC results)
    cuphyPuschCellAerialMetrics_t  cell_metrics_info[ slot_command_api::MAX_CELLS_PER_CELL_GROUP];
#endif
};

#endif

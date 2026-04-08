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

#ifndef PHY_PDSCH_AGGR_CHANNEL_H
#define PHY_PDSCH_AGGR_CHANNEL_H

#include "phychannel.hpp"
#include "cell.hpp"

/**
 * @class PhyPdschAggr
 * @brief Handles PDSCH (Physical Downlink Shared Channel) transmission processing.
 *
 * This class implements the downlink PDSCH data transmission pipeline using the cuPHY library.
 * PDSCH carries user data (transport blocks) for downlink transmissions to UEs. Supports
 * aggregated multi-cell processing with CUDA graph and stream-based execution modes.
 */
class PhyPdschAggr : public PhyChannel {
public:
    /**
     * @brief Constructs a PhyPdschAggr object.
     *
     * Initializes the PDSCH channel with the given cuphydriver handle, GPU device, CUDA stream,
     * and MPS context. Sets up memory footprint tracking, processing mode (graphs vs. streams,
     * fallback mode), and allocates input/output buffers for transport blocks.
     *
     * @param _pdh cuphydriver handle
     * @param _gDev GPU device struct pointer
     * @param _s_channel CUDA stream for asynchronous GPU operations
     * @param _mpsCtx MPS context for GPU resource partitioning
     */
    PhyPdschAggr(
            phydriver_handle _pdh,
            GpuDevice*       _gDev,
            cudaStream_t     _s_channel,
            MpsCtx *        _mpsCtx
        );
    
    /**
     * @brief Destructor for PhyPdschAggr.
     *
     * Frees allocated transport block buffers and destroys the cuPHY PDSCH TX handle.
     */
    ~PhyPdschAggr();

    ////////////////////////////////////////////////////////////
    /// Module generic
    ////////////////////////////////////////////////////////////
    /**
     * @brief Configures the PDSCH transmission for the current slot.
     *
     * Sets up dynamic parameters including transport block data pointers, modulation/coding schemes,
     * resource allocation, DMRS configuration, and output buffer pointers. Handles preponed H2D copy 
     * synchronization when enabled. Calls cuphySetupPdschTx() to configure the PDSCH transmission.
     *
     * @param aggr_cell_list List of Cell objects being processed in this slot
     * @param aggr_dlbuf List of downlink output buffers (one per cell) for PDSCH.
     * @return 0 on success, -1 on failure (setup error)
     */
    int setup(
        const std::vector<Cell *> &aggr_cell_list,
        const std::vector<DLOutputBuffer *> &aggr_dlbuf
    );

    /**
     * @brief Executes the PDSCH transmission processing on GPU.
     *
     * Invokes cuphyRunPdschTx() to perform LDPC encoding, rate matching, modulation, layer mapping,
     * precoding. Only runs if setup completed successfully.
     * Uses either CUDA graphs or streams based on configuration.
     *
     * @return 0 on success, -1 on failure (run error)
     */
    int          run();
    
    /**
     * @brief Post-processing callback after PDSCH transmission completes.
     *
     * Invokes registered L2 DL callback to notify upper layers of PDSCH processing completion.
     * Updates per-cell metrics (TB count, bytes, UEs, processing time) when AERIAL_METRICS enabled.
     *
     * @param si Slot indication (SFN, slot) for the completed transmission
     * @return 0 on success, error code otherwise
     */
    int          callback(struct slot_command_api::slot_indication si);
    
    /**
     * @brief Cleanup after PDSCH processing completes.
     *
     * Currently a no-op (returns 0 immediately). Provided for API consistency.
     *
     * @return Always returns 0
     */
    int          cleanup();

    ////////////////////////////////////////////////////////////
    /// Module specific
    ////////////////////////////////////////////////////////////
    /**
     * @brief Retrieves the PDSCH dynamic parameters for the current slot.
     *
     * Extracts the PDSCH parameters (TB data, MCS, resource allocation) from the aggregated
     * slot command structure for use in setup().
     *
     * @return Pointer to the PDSCH parameters for the current slot
     */
    slot_command_api::pdsch_params* getDynParams();
    
    /**
     * @brief Creates the cuPHY PDSCH TX object.
     *
     * Initializes the cuPHY PDSCH transmission handle, builds the cell list with static
     * parameters (antennas, PRBs, numerology), configures max UEs, and sets up processing mode.
     * Called during driver initialization.
     *
     * @return 0 on success, non-zero error code on failure
     */
    int createPhyObj();
    
    /**
     * @brief Updates physical cell ID in static parameters.
     *
     * Used during cell reconfiguration to update the PHY cell ID without recreating the
     * entire cuPHY object.
     *
     * @param phyCellId_old Old physical cell ID to replace
     * @param phyCellId_new New physical cell ID
     */
    void updatePhyCellId(uint16_t,uint16_t);
    
    /**
     * @brief Waits for preponed H2D copy CUDA event to complete.
     *
     * When preponed H2D copy is enabled, this function waits for the H2D copy thread to record
     * the completion event for the current slot's transport block data. Has a timeout of 8ms
     * (2x GENERIC_WAIT_THRESHOLD_NS). Only used when both enable_prepone_h2d_cpy and
     * h2d_copy_thread_enable are true.
     *
     * @return 0 on success, -1 on timeout
     */
    int waitH2dCopyCudaEventRec();

    /**
     * @brief Get PDSCH H2D copy duration in microseconds for given slot.
     *
     * Calculates the elapsed time between the H2D copy start and completion CUDA events
     * for the specified slot. Returns 0 if preponed H2D copy is not enabled.
     *
     * @param[in] slot Slot number to get the H2D copy time for
     * @return H2D copy duration in microseconds, or 0 if preponed H2D copy is disabled
     */
    float getPdschH2DCopyTime(uint8_t slot);

protected:
    cuphyPdschTxHndl_t                  handle;         ///< cuPHY PDSCH transmission handle (opaque handle to cuPHY PDSCH TX object)
    uint64_t                            procModeBmsk;   /*!< Processing mode bitmask [B2 B1 B0]: 
                                                         *   B0: 0=streams (PDSCH_PROC_MODE_NO_GRAPHS), 1=graphs (PDSCH_PROC_MODE_GRAPHS)
                                                         *   B1: 1=setup-once fallback (debugging only - not used in production, PDSCH_PROC_MODE_SETUP_ONCE_FALLBACK)
                                                         *   B2: 1=inter-cell batching (PDSCH_INTER_CELL_BATCHING, deprecated - no effect) */
    cuphyPdschDataIn_t                  tb_crc_data_in; ///< Input data structure for TB CRC (currently unused: read_TB_CRC=false)
    uint64_t                            tb_bytes;       ///< Total transport block bytes in current slot (for metrics)
    uint16_t                            tb_count;       ///< Total transport block count in current slot (for metrics)
    uint16_t                            nUes;           ///< Number of UEs in current slot (for metrics)
    std::vector<cell_id_t>              cell_id_list;   ///< List of physical cell IDs being processed (built during createPhyObj)
    cuphyPdschStatPrms_t                static_params;  ///< cuPHY static parameters for PDSCH TX (setup-time configuration)
    std::vector<cuphyCellStatPrm_t>     static_params_cell;  ///< Per-cell static parameters (antennas, PRBs, numerology)
    cuphyPdschDynPrms_t                 dyn_params;     ///< cuPHY dynamic parameters for PDSCH TX (per-slot configuration)
    cuphyPdschDataIn_t                  DataIn;         ///< cuPHY input data structure (contains TB data pointers)
    cuphyPdschDataOut_t                 DataOut;        ///< cuPHY output data structure (contains PDSCH data output buffer pointers)
    cuphyPdschStatusOut_t               statusOut;      ///< cuPHY status output (error tracking for TB processing)

    // Fallback mode members
    uint8_t * fbOutBuf[PDSCH_MAX_CELLS_PER_CELL_GROUP];  ///< Fallback output buffer pointers (used in setup-once fallback mode)
    uint8_t first_slot;                                   ///< Slot counter for fallback mode (full setup only on first 1-2 slots)
};

#endif

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

#ifndef PHY_PRACH_AGGR_CHANNEL_H
#define PHY_PRACH_AGGR_CHANNEL_H

#include "phychannel.hpp"
#include "cell.hpp"
#include "prach_stage_error.hpp"

#include <cstdint>
#include <optional>

#include <tl/expected.hpp>

#define PRACH_USE_BATCHED_MEMCPY 1 ///< Enable batched memcpy for the 4 D2H memcpy calls after PRACH run

struct cell_phy_info;

/**
 * @class PhyPrachAggr
 * @brief Handles PRACH (Physical Random Access Channel) reception and preamble detection processing.
 *
 * This class implements the uplink PRACH reception pipeline using the cuPHY library.
 * PRACH is used for random access procedure, where UEs transmit preambles to establish
 * initial connection with the network. Supports aggregated multi-cell/multi-occasion processing
 * with dynamic reconfiguration capabilities.
 */
class PhyPrachAggr : public PhyChannel {
public:
    /**
     * @brief Constructs a PhyPrachAggr object.
     *
     * Initializes the PRACH channel with the given cuphydriver handle, GPU device, CUDA streams,
     * and MPS context. Sets up memory footprint tracking, allocates input/output buffers for
     * preamble detection results, and initializes batched memcpy helper.
     *
     * @param _pdh cuphydriver handle
     * @param _gDev GPU device struct pointer
     * @param _s_channels Array of CUDA streams (uses first stream) for asynchronous GPU operations
     * @param _mpsCtx MPS context for GPU resource partitioning
     */
    PhyPrachAggr(
            phydriver_handle _pdh,
            GpuDevice*       _gDev,
            cudaStream_t*     _s_channels,
            MpsCtx * _mpsCtx);
    
    /**
     * @brief Destructor for PhyPrachAggr.
     *
     * Frees allocated buffers and destroys the cuPHY PRACH RX handle.
     */
    ~PhyPrachAggr();

    ////////////////////////////////////////////////////////////
    /// Module generic
    ////////////////////////////////////////////////////////////
    /**
     * @brief Configures the PRACH reception for the current slot.
     *
     * Sets up dynamic parameters including PRACH occasions, force thresholds, input buffer
     * pointers from UL order kernel (st3 buffers), and prepares output buffers for detected
     * preambles. Calls cuphySetupPrachRx().
     *
     * @param aggr_cell_list List of Cell objects being processed in this slot
     * @param ulbuf_st3_v List of UL input buffers (st3/o3) containing ordered PRACH samples
     * @param stream CUDA stream for setup operations
     * @return 0 on success, -1 on failure (setup error)
     */
    int          setup(const std::vector<Cell *>& aggr_cell_list, const std::vector<ULInputBuffer *>& ulbuf_st3_v, cudaStream_t stream);
    
    /**
     * @brief Executes the PRACH preamble detection processing on GPU.
     *
     * Invokes cuphyRunPrachRx() to perform preamble correlation, peak detection, and timing
     * estimation. After GPU processing, copies detection results (preamble indices, timing
     * delays, power estimates, RSSI) from GPU to host using batched memcpy when enabled.
     *
     * @return 0 on success, -1 on failure (run error)
     */
    int          run();
    
    /**
     * @brief Validates PRACH output against reference data.
     *
     * Compares detected preambles against MATLAB reference data when validation mode is enabled.
     * Used for testing/debugging purposes.
     *
     * @return 0 on success, error code otherwise
     */
    int          validate();
    
    /**
     * @brief Post-processing callback after PRACH detection completes.
     *
     * Invokes registered UL callback to report detected preambles (indices, timing, power)
     * to upper layers (MAC). Updates per-cell PRACH metrics when AERIAL_METRICS enabled.
     *
     * @return 0 on success, error code otherwise
     */
    int          callback();
    
    /**
     * @brief Cleanup after PRACH processing completes.
     *
     * Calls parent class cleanup and returns.
     *
     * @return Always returns 0
     */
    int          cleanup();
    
    /**
     * @brief Reserves UL input buffers for PRACH data.
     *
     * Stores references to UL buffers (st3) for use during setup(). Called before processing begins.
     *
     * @param ulbuf Vector of UL input buffer pointers to reserve
     * @return 0 on success
     */
    int          reserve(std::vector<ULInputBuffer *>& ulbuf);
    
    /**
     * @brief Creates the cuPHY PRACH RX object.
     *
     * Initializes the cuPHY PRACH reception handle, builds the cell and occasion lists with
     * static parameters (configuration index, root sequences, zero-correlation zones, etc.).
     * Called during driver initialization.
     *
     * @return 0 on success, non-zero error code on failure
     */
    int          createPhyObj();
    
    /**
     * @brief Creates a new temporary cuPHY PRACH RX object for reconfiguration.
     *
     * Creates a new PRACH handle with updated configuration (new occasions/cells) without
     * disrupting the currently active handle. Used during dynamic reconfiguration.
     *
     * @return 0 on success, -1 on failure
     */
    int          createNewPhyObj();
    
    /**
     * @brief Deletes the temporary cuPHY PRACH RX object.
     *
     * Destroys the temporary handle created by createNewPhyObj(). Called after reconfiguration
     * is complete or cancelled.
     *
     * @return Always returns 0
     */
    int          deleteTempPhyObj();

    /**
     * @brief Swaps the active and temporary PRACH handles.
     *
     * Makes the temporary handle (with new configuration) the active handle, and demotes
     * the old active handle to temporary. Used to atomically switch configurations during
     * dynamic reconfiguration.
     */
    void         changePhyObj();

    // ---- Offload (nonslot_lp) CONFIG reconfiguration ----
    // Additive PRACH-specific staging used by PrachOffloadReconfig. The legacy
    // getNextPrachAggr() path does not use these.

    /// Result of stageConfig(): on success the new occasion start index (nullopt on the
    /// in-place path); PrachStageError::StageRejected on failure. The failure cause (cell
    /// not present here, or invalid PRACH params) is also logged at the source in
    /// applyPrachUpdate; the typed error lets callers name the failure instead of a bare
    /// false. The tl::expected separates failure from the success-side optional.
    using StageConfigResult = tl::expected<std::optional<std::uint16_t>, PrachStageError>;

    /**
     * Stages a new PRACH configuration without altering live state.
     *
     * @param[in] cell_id    Cell being reconfigured.
     * @param[in] cell_pinfo New cell PHY configuration.
     * @return On success, the new occasion start index (nullopt when the in-place path
     *         leaves it unchanged); tl::unexpected(PrachStageError::StageRejected) on failure.
     */
    [[nodiscard]] StageConfigResult stageConfig(std::uint16_t cell_id, const cell_phy_info& cell_pinfo);

    /**
     * Creates the temporary (PONG) PRACH object from the staged config.
     *
     * @return Empty on success; tl::unexpected(PrachStageError::TempCreateFailed) if
     *         cuphyCreatePrachRx failed to build the temp handle.
     */
    [[nodiscard]] tl::expected<void, PrachStageError> createNewPhyObjFromStagedConfig();

    /** Commits the staged PRACH config into live state. */
    void               commitStagedConfig();

    /**
     * Discards the staged PRACH config (pre-arm rollback).
     *
     * Idempotent: safe to call on an aggregator that was never staged (no-op),
     * since the orchestrator's failure path discards every aggregator regardless
     * of how far staging progressed.
     */
    void               discardStagedConfig();
    
    /**
     * @brief Updates PRACH configuration for a specific cell.
     *
     * Dynamically updates PRACH occasions (FDM occasions, root sequences, etc.) for the
     * specified cell. Rebuilds the occasion list and updates static parameters. Used when
     * cell configuration changes without recreating the entire pipeline.
     *
     * @param cell_id Physical cell ID to update
     * @param cell_pinfo New cell PHY info containing updated PRACH configuration
     * @return 0 on success, error code otherwise
     */
    int          updateConfig(cell_id_t cell_id, cell_phy_info& cell_pinfo);

    ////////////////////////////////////////////////////////////
    /// Module specific
    ////////////////////////////////////////////////////////////
    /**
     * @brief Gets the size of the cell static parameter vector.
     *
     * Returns the number of cells configured in this PRACH aggregation instance.
     *
     * @return Number of cells in prachCellStatVec
     */
    uint32_t     getCellStatVecSize();
    
    /**
     * @brief Retrieves the PRACH dynamic parameters for the current slot.
     *
     * Extracts the PRACH parameters (occasions, thresholds) from the aggregated slot
     * command structure for use in setup().
     *
     * @return Pointer to the PRACH parameters for the current slot
     */
    slot_command_api::prach_params* getDynParams();
    
    /**
     * @brief Gets the GPU to host copy time for PRACH results.
     *
     * Returns the elapsed time between start_copy and end_copy events, which measures
     * the D2H transfer time for preamble detection results.
     *
     * @return GPU copy time in milliseconds
     */
    float                           getGPUCopyTime();
    
    /**
     * @brief Converts PRACH resource allocation parameters to number of PRBs.
     *
     * Static utility function that calculates the number of PRBs occupied by PRACH based on
     * preamble length (L_RA), PRACH subcarrier spacing (delta_f_RA), and data SCS (delta_f).
     * Formula: ceil((L_RA/12) * (delta_f_RA / delta_f))
     *
     * @param L_RA Preamble length (839 for long, 139 for short formats)
     * @param delta_f_RA PRACH subcarrier spacing in Hz (1.25kHz or 5kHz)
     * @param delta_f Data subcarrier spacing in Hz (15kHz, 30kHz, etc.)
     * @return Number of PRBs occupied by PRACH
     */
    static inline uint16_t          numPrbcConversionTable(uint32_t L_RA, uint32_t delta_f_RA, uint32_t delta_f);

protected:
    uint8_t*                    originalInputData_d;            ///< Original PRACH input data on device (unused)
    uint8_t*                    originalInputData_h;            ///< Original PRACH input data on host (unused)
    std::unique_ptr<host_buf>   prach_crc_errors_h;            ///< Host buffer for CRC error count
    std::vector<ULInputBuffer *> ulbuf_st3;                    ///< Reserved UL input buffers (st3/o3) for PRACH samples
    std::unique_ptr<hdf5hpp::hdf5_file> debugFileH;            ///< HDF5 file handle for debug output

    ////////////////////////////////////////////
    //// cuPHY specific
    ////////////////////////////////////////////
    cuphy::tensor_desc prach_data_rx_desc[PRACH_MAX_OCCASIONS_AGGR];  ///< Tensor descriptors for PRACH RX data (one per occasion)
    cuphyTensorPrm_t tDataRx;                                  ///< Tensor parameter for PRACH RX data
    cuphyTensorPrm_t dataIn;                                   ///< Tensor parameter for input data
    cuphy::tensor_device y_u_ref;                              ///< Reference tensor for validation
    cuphyPrachStatPrms_t prach_params_static;                  ///< cuPHY static parameters for PRACH RX (setup-time configuration)
    std::vector<cuphyPrachOccaStatPrms_t> prach_occa_stat_params;  ///< Temporary vector for occasion static parameters during configuration
    cuphyPrachOccaDynPrms_t* prach_dyn_occa_params;            ///< Pointer to dynamic occasion parameters (points to slot command data)
    cuphyPrachDynDbgPrms_t  dyn_dbg_params;                    ///< Dynamic debug parameters for runtime output (HDF5 API logging control)
    cuphyPrachStatDbgPrms_t stat_dbg_params;                   ///< Static debug parameters (setup-time HDF5 dump control)
    std::vector<cuphyPrachCellStatPrms_t> prachCellStatVec;    ///< Per-cell static parameters (configuration, antennas, occasions)
    std::vector<cuphyPrachOccaStatPrms_t> prachOccaStatVec;    ///< Per-occasion static parameters (root sequences, zero-correlation zones)
    std::vector<cell_id_t> cell_id_list;                       ///< List of physical cell IDs being processed (built during createPhyObj)
    std::unordered_map<cell_id_t,uint32_t>prachCellStatIndex; ///< Map from cell_id to index in prachCellStatVec (for fast lookup)

    /**
     * Genuinely staged PRACH static config for the offload reconfig handover.
     *
     * Holds copies of the live cell/occasion vectors with the requested update
     * applied, so the new (PONG) handle can be built and validated without
     * mutating live state until the handover commits.
     */
    struct PrachAggrStagedConfig final
    {
        std::vector<cuphyPrachCellStatPrms_t> cell_stat;     //!< Staged copy of prachCellStatVec.
        std::vector<cuphyPrachOccaStatPrms_t> occa_stat;     //!< Staged copy of prachOccaStatVec.
        bool                                  active{false}; //!< True between stageConfig and commit/discard.

        void reset()
        {
            cell_stat.clear();
            occa_stat.clear();
            active = false;
        }
    };
    PrachAggrStagedConfig staged_;  ///< Offload-only staged PRACH config (single-threaded: nonslot_lp worker).

    /**
     * Applies the @p cell_pinfo PRACH update to the given cell/occasion vectors.
     *
     * Shared core of the legacy in-place updateConfig() and the offload staged
     * stageConfig(): validates the incoming params, then either updates occasions
     * in place (count unchanged/shrunk) or appends the grown occasion set. Does NOT
     * touch live Cell state or prach_params_static; the caller commits those.
     *
     * @param[in]     cell_id        Cell being reconfigured.
     * @param[in]     cell_pinfo     New cell PHY configuration.
     * @param[in,out] cell_stat      Cell-stat vector to update (live or staged copy).
     * @param[in,out] occa_stat      Occasion-stat vector to update (live or staged copy).
     * @param[out]    occa_start_idx Set to the cell's new occasion start index when the
     *                               occasion set grew (append path); left unset otherwise.
     * @return 0 on success, -1 on validation/lookup failure.
     */
    [[nodiscard]] int applyPrachUpdate(cell_id_t                              cell_id,
                                       const cell_phy_info&                   cell_pinfo,
                                       std::vector<cuphyPrachCellStatPrms_t>& cell_stat,
                                       std::vector<cuphyPrachOccaStatPrms_t>& occa_stat,
                                       std::optional<std::uint16_t>&          occa_start_idx);
    size_t prach_workspace_size;                               ///< Size of cuPHY workspace buffer in bytes
    cuphy::buffer<float, cuphy::device_alloc> prach_workspace_buffer;  ///< cuPHY workspace buffer for intermediate calculations
    cuphy::tensor_device gpu_num_detectedPrmb;                 ///< GPU tensor: number of detected preambles per occasion
    cuphy::tensor_pinned cpu_num_detectedPrmb;                 ///< CPU tensor: number of detected preambles per occasion (copied from GPU)
    cuphy::tensor_device gpu_prmbIndex_estimates;              ///< GPU tensor: detected preamble indices
    cuphy::tensor_pinned cpu_prmbIndex_estimates;              ///< CPU tensor: detected preamble indices (copied from GPU)
    cuphy::tensor_device gpu_prmbDelay_estimates;              ///< GPU tensor: timing delay estimates for detected preambles
    cuphy::tensor_pinned cpu_prmbDelay_estimates;              ///< CPU tensor: timing delay estimates (copied from GPU)
    cuphy::tensor_device gpu_prmbPower_estimates;              ///< GPU tensor: power estimates for detected preambles
    cuphy::tensor_pinned cpu_prmbPower_estimates;              ///< CPU tensor: power estimates (copied from GPU)
    cuphy::tensor_pinned ant_rssi;                             ///< Per-antenna RSSI (Received Signal Strength Indicator)
    cuphy::tensor_pinned rssi;                                 ///< Combined RSSI across all antennas
    cuphy::tensor_pinned interference;                         ///< Interference power estimate
    // Validation: read results from matlab test vector
    using tensor_pinned_R_32U = cuphy::typed_tensor<CUPHY_R_32U, cuphy::pinned_alloc>;  ///< Type alias for pinned 32-bit unsigned tensor
    using tensor_pinned_R_32F = cuphy::typed_tensor<CUPHY_R_32F, cuphy::pinned_alloc>;  ///< Type alias for pinned 32-bit float tensor
    tensor_pinned_R_32U matlab_prmbIndex_estimates;            ///< Reference preamble indices from MATLAB (for validation)
    tensor_pinned_R_32F matlab_prmbDelay_estimates;            ///< Reference timing delays from MATLAB (for validation)
    tensor_pinned_R_32F matlab_prmbPower_estimates;            ///< Reference power estimates from MATLAB (for validation)
    
    /**
     * @brief Compares PRACH detection results against reference data.
     *
     * Internal validation function that compares detected preambles against MATLAB reference.
     *
     * @return 0 on match, error code otherwise
     */
    int reference_comparison();

    t_ns start_t_rx;                                           ///< Reception start timestamp
    t_ns end_t_rx;                                             ///< Reception end timestamp

    ////////////////////////////////////////////
    //// CUDA Events
    ////////////////////////////////////////////
    cudaEvent_t start_copy;                                    ///< CUDA event for start of D2H copy
    cudaEvent_t end_copy;                                      ///< CUDA event for end of D2H copy

    int rach_occasion;                                         ///< RACH occasion index (unused/legacy)
    cuphyPrachRxHndl_t handle;                                 ///< cuPHY PRACH RX handle (active handle)
    cuphyPrachRxHndl_t handle_temp;                            ///< Temporary cuPHY PRACH RX handle (for reconfiguration)
    uint32_t           num_new_handles;                        ///< Number of new handles during reconfiguration (unused)
    cuphyTensorPrm_t* pTensors;                                ///< Pointer to tensor parameters array (unused/legacy)
    cuphyPrachDataIn_t  pInput;                                ///< cuPHY input data structure (contains RX sample pointers)
    cuphyPrachDataOut_t pOutput;                               ///< cuPHY output data structure (contains detection result pointers)
    cuphyPrachDynPrms_t pDynParams;                            ///< cuPHY dynamic parameters (per-slot configuration)

    cuphyPrachStatusOut_t statusOut;                           ///< cuPHY status output (error tracking for preamble detection)

    cuphyBatchedMemcpyHelper m_batchedMemcpyHelper;            ///< Helper for batched D2H memcpy of detection results
private:
    // Cell buffer information: pairs of {cell_id, {buffer_idx, num_occasions}}
    std::array<std::pair<int, std::pair<int, size_t>>, MAX_CELLS_PER_SLOT> cell_buffer_info{};
    size_t cell_buffer_info_size{};  //!< Number of valid entries in cell_buffer_info
};

#endif

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

 #pragma once

 #include "api.h"
 #include "cumac.h"

 // cuMAC namespace
namespace cumac {
    /// @brief Task structure for the MU-MIMO user pairing algorithm
    struct muUePairTask {
        // GPU buffer
        uint8_t*        task_in_buf; // GPU buffer for the task input data for the current slot for all cells. Buffer size = m_task_in_buf_len_per_cell * num_cell
        uint8_t*        task_out_buf; // GPU buffer for the task solution output data for the current slot for all cells. Buffer size = m_task_out_buf_len_per_cell * num_cell

        // CUDA stream
        cudaStream_t    strm; 

        // parameters
        uint16_t        num_srs_ue_per_slot_cell; // number of updated SRS UEs per slot per cell. Should be set to the maximum value across all cells for the current slot.
        uint16_t        num_blocks_per_row_chanOrtMat; // number of thread blocks per row for the channel correlation matrix
        uint8_t         kernel_launch_flags; // bit flags for CUDA kernel launch mode
        // 0x01: whether the channel correlation computation kernel is to be launched.
        // 0x02: whether the UE pairing algorithm kernel is to be launched.
        bool            is_mem_sharing; // whether to use shared cuBB SRS memory bank to obtain channel estimates
    };

    // ************************************************** Dynamic descriptor for the MU-MIMO user pairing CUDA kernel **************************************************
    /// @brief Dynamic descriptor for the MU-MIMO user pairing CUDA kernel
    struct muUePairChanCorrDynDescr {
        // GPU buffers
        uint8_t*    srs_chan_est_buf; // SRS channel estimates GPU buffer
        float*      srs_snr_buf; // SRS SNRs GPU buffer
        uint8_t*    task_in_buf; // Task input GPU buffer
        float*      chan_orth_mat_buf; // Channel correlation matrix GPU buffer
        __half2*    cubb_srs_gpu_buf; // cuBB SRS memory bank GPU buffer

        // parameters
        uint32_t    task_in_buf_len_per_cell; // length of the allocated task input GPU buffer for each cell in bytes
        uint16_t    num_cell; // number of cells
        uint16_t    num_bs_ant_port; // number of BS antenna ports
        uint16_t    num_prg; // number of PRGs on the channel (with per-PRG-level SRS channel estimates)
        uint16_t    num_subband; // number of subbands
        uint16_t    num_prg_samp_per_subband; // number of PRG samples per subband
        uint16_t    num_blocks_per_prg; // number of thread blocks per PRG
        uint16_t    num_blocks_per_cell; // number of thread blocks per cell
        uint16_t    num_blocks_per_row_chanOrtMat; // number of thread blocks per row for the channel correlation matrix
    };

    struct muUePairAlgDynDescr {
        // GPU buffers
        float*      srs_snr_buf; // SRS SNRs GPU buffer
        uint8_t*    task_in_buf; // Task input GPU buffer
        uint8_t*    task_out_buf; // Task output GPU buffer
        float*      chan_orth_mat_buf; // Channel correlation matrix GPU buffer

        // parameters
        uint16_t    num_cell; // number of cells
        uint16_t    num_bs_ant_port; // number of BS antenna ports
        uint16_t    num_subband; // number of subbands
        uint16_t    num_prg_samp_per_subband; // number of PRG samples per subband
        uint32_t    task_in_buf_len_per_cell; // length of the allocated task input GPU buffer for each cell in bytes
        uint32_t    task_out_buf_len_per_cell; // length of the allocated task output GPU buffer for each cell in bytes
    };

    // ************************************************** MU-MIMO user pairing class **************************************************
    /** 
     * @brief This module performs the MU-MIMO user pairing algorithm on the GPU 
     * for the connected UEs across a number of cells controlled by a common DU.
    */
    class muMimoUserPairing {
    public:
        // default constructor
        muMimoUserPairing(uint8_t*      srs_chan_est_buf_base_addr,
                          float*        srs_snr_buf_base_addr,
                          float*        chan_orth_mat_buf_base_addr,
                          __half2*      cubb_srs_gpu_buf_base_addr,
                          uint16_t      num_cell,
                          uint16_t      num_prg,
                          uint16_t      num_subband,
                          uint16_t      num_prg_samp_per_subband,
                          uint16_t      num_bs_ant_port);

        ~muMimoUserPairing();

        muMimoUserPairing(muMimoUserPairing const&)            = delete;
        muMimoUserPairing& operator=(muMimoUserPairing const&) = delete;

        /**
        * @brief Sets up the MU-MIMO user pairing operation on GPU for both the UE pairing algorithm and the channel correlation computation.
        *
        * @param[in] task     Pointer to the task structure.
        *
        * @note               This function does not perform internal stream synchronization to allow for
        *                     asynchronous operation and overlapping of GPU work.
        *
        */
        void setup(muUePairTask* task);

        /**
        * @brief Executes the MU-MIMO user pairing operation on GPU and stores results in host memory
        *
        * @param[out] hSolAddr Pointer to host memory where the MU-MIMO user pairing solution will be written.
        *                      Must be a valid host-accessible memory address with sufficient space
        *                      to hold the complete solution structure. Memory should be pre-allocated
        *                      by the caller.
        *
        * @note                This function does not perform internal stream synchronization to allow for
        *                      asynchronous operation and overlapping of GPU work. External device synchronization 
        *                      (e.g., cudaStreamSynchronize or cudaDeviceSynchronize) is required after this call 
        *                      and before accessing the solution data at hSolAddr or sending it over the NVIPC interface.
        *
        */
        void run(uint8_t* hSolAddr);

        /**
        * @brief Executes the MU-MIMO user pairing operation on CPU
        */
        void run_cpu();

    private:
        /// @brief CUDA stream for the MU-MIMO user pairing operation
        cudaStream_t                    m_strm{nullptr};

        /// @brief bit flags for CUDA kernel launch mode
        uint8_t                         m_kernel_launch_flags{0x00};
        // 0x01: whether the channel correlation computation kernel is to be launched.
        // 0x02: whether the UE pairing algorithm kernel is to be launched.

        /// @brief Whether to use shared cuBB SRS memory bank to obtain channel estimates
        bool                            m_is_mem_sharing{false};

        /// @brief Number of cells
        uint16_t                        m_num_cell{0};

        /// @brief Number of PRGs per cell
        uint16_t                        m_num_prg{0};

        /// @brief Number of subbands
        uint16_t                        m_num_subband{0};

        /// @brief Number of PRG samples per subband
        uint16_t                        m_num_prg_samp_per_subband{0};

        /// @brief Number of BS antenna ports
        uint16_t                        m_num_bs_ant_port{0};

        /// @brief Length of the allocated task-in GPU buffer for each cell in bytes
        uint32_t                        m_task_in_buf_len_per_cell{0};

        /// @brief Length of the allocated task-out GPU buffer for each cell in bytes
        uint32_t                        m_task_out_buf_len_per_cell{0};

        /// @brief SRS channel estimates GPU buffer
        // device memory should be allocated by the caller externally.
        // minimum buffer size = sizeof(__half2) * m_num_cell * m_num_subband * m_num_prg_samp_per_subband * MAX_NUM_SRS_UE_PER_CELL * MAX_NUM_UE_ANT_PORT * m_num_bs_ant_port
        uint8_t*                        m_srs_chan_est_buf{nullptr};
        

        /// @brief SRS SNRs GPU buffer
        // device memory should be allocated by the caller externally.
        // minimum buffer size = sizeof(float) * m_num_cell * MAX_NUM_SRS_UE_PER_CELL
        float*                          m_srs_snr_buf{nullptr};

        /// @brief Channel correlation matrix GPU buffer
        // device memory should be allocated by the caller externally.
        // minimum buffer size = sizeof(float) * m_num_cell * m_num_subband * m_num_prg_samp_per_subband * MAX_NUM_SRS_UE_PER_CELL * MAX_NUM_UE_ANT_PORT * (MAX_NUM_SRS_UE_PER_CELL * MAX_NUM_UE_ANT_PORT + 1) / 2
        float*                          m_chan_orth_mat_buf{nullptr};

        /// @brief cuBB SRS memory bank GPU buffer
        // device memory should be allocated by cuPHY driver in cuBB
        __half2*                        m_cubb_srs_gpu_buf{nullptr};

        /// @brief UE pairing solution output GPU buffer pointer
        // device memory should be allocated by the caller externally.
        uint8_t*                        m_solution_out_buf{nullptr};
        
        /// @brief Dynamic descriptor for the channel correlation computation on the GPU
        muUePairChanCorrDynDescr*       m_gpuDynDescrChanCorr{nullptr};

        /// @brief Dynamic descriptor for the UE pairing algorithm on the GPU
        muUePairAlgDynDescr*            m_gpuDynDescrUePair{nullptr};

        /// @brief Dynamic descriptor for the channel correlation computation on the CPU
        muUePairChanCorrDynDescr*       m_cpuDynDescrChanCorr{nullptr};

        /// @brief Dynamic descriptor for the UE pairing algorithm on the CPU
        muUePairAlgDynDescr*            m_cpuDynDescrUePair{nullptr};
        
        /// @brief CUDA kernel dimensions
        uint16_t                        m_numThrdPerBlkChanCorr{0};
        uint16_t                        m_numBlocksPerPrgChanCorr{0};
        uint16_t                        m_numBlocksPerCellChanCorr{0};
        uint16_t                        m_numBlocksPerRowChanCorr{0};
        uint16_t                        m_numThrdBlkChanCorr{0};
        dim3                            m_blockDimChanCorr;
        dim3                            m_gridDimChanCorr;
        
        uint16_t                        m_numThrdBlkUePair{0};
        uint16_t                        m_numThrdPerBlkUePair{0};
        dim3                            m_blockDimUePair;
        dim3                            m_gridDimUePair;

        /// @brief CUDA kernel launch configurations
        std::unique_ptr<launchCfg_t>    m_launchCfgChanCorr;
        std::unique_ptr<launchCfg_t>    m_launchCfgUePair;

        /// @brief CUDA kernel selection for the MU-MIMO channel correlation computation/user pairing algorithms
        void chanCorrKernelSelect();
        void uePairKernelSelect();
    };

    /// @brief MU-MIMO user pairing CUDA kernels
    static __global__ void muUePairChanCorrKernel(muUePairChanCorrDynDescr* pDynDescr);
    static __global__ void muUePairChanCorrKernel_memSharing(muUePairChanCorrDynDescr* pDynDescr);
    static __global__ void muUePairAlgKernel(muUePairAlgDynDescr* pDynDescr);
    static __global__ void muUePairAlgKernel_memSharing(muUePairAlgDynDescr* pDynDescr);

    /// @brief CPU version functions of the CUDA kernels for debugging purposes
    typedef std::pair<float, uint16_t> pfMetricUePair;

    void muUePairChanCorrKernel_cpu(muUePairChanCorrDynDescr* pDynDescr);
    void muUePairChanCorrKernel_memSharing_cpu(muUePairChanCorrDynDescr* pDynDescr);
    void muUePairAlgKernel_cpu(muUePairAlgDynDescr* pDynDescr);
    void muUePairAlgKernel_memSharing_cpu(muUePairAlgDynDescr* pDynDescr);
}
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

 #pragma once

 #include "api.h"
 #include "cumac.h"

 // cuMAC namespace
namespace cumac {
  
    /// @brief Per-UE info structure for managing the UE info in device main memory
    struct PFM_UE_INFO_MANAGE {
        uint32_t            ravg_dl_lc[CUMAC_PFM_MAX_NUM_LC_PER_UE]; // array of average rates of LCs configured for this UE for DL
        uint32_t            ravg_ul_lcg[CUMAC_PFM_MAX_NUM_LCG_PER_UE]; // array of average rates of LCGs configured for this UE for UL
    };

    struct PFM_CELL_INFO_MANAGE {
        PFM_UE_INFO_MANAGE  ue_info_manage[CUMAC_PFM_MAX_NUM_UE_PER_CELL]; // array of managed UE info for the PFM sorting in the current slot for this cell
    };

    /// @brief PFM sorting task structure for setup() function
    struct pfmSortTask {
        uint16_t        num_cell; // Number of cells.
        cudaStream_t    strm; // CUDA stream.
        uint8_t*        gpu_buf; // GPU buffer for the data in the current slot for all cells. GPU buffer size = sizeof(cumac_pfm_cell_info_t) * num_cell
    };

    // ************************************************** Dynamic descriptor for the PFM sorting CUDA kernel **************************************************
    /// @brief Dynamic descriptor for the PFM sorting CUDA kernel
    struct pfmSortDynDescr {
        uint8_t*    task_in_buf; // GPU buffer for the task input data for each cell
        uint8_t*    gpu_main_in_buf; // GPU main buffer for maintaining PFM sorting data for each cell
        uint8_t*    gpu_out_buf; // GPU output buffer for storing the PFM sorting results for each cell
    };

    // ************************************************** PFM sorting class **************************************************
    /** 
     * @brief This module performs the proportional-fairness metric (PFM) sorting algorithm on the GPU 
     * for the connected UE LCs/LCGs across a number of cells controlled by a common DU.
    */
    class pfmSort {
    public:
        pfmSort();
        ~pfmSort();

        pfmSort(pfmSort const&)            = delete;
        pfmSort& operator=(pfmSort const&) = delete;

        /**
         * @brief Sets up the PFM sort operation on GPU
         *
         * @param[in] task      Pointer to the PFM sort task structure.
         *
         * @note                This function does not perform internal stream synchronization to allow for
         *                      asynchronous operation and overlapping of GPU work.
         *
         */
        void setup(pfmSortTask* task);

        /**
        * @brief Executes the PFM sort operation on GPU and stores results in host memory
        *
        * @param[out] hSolAddr Pointer to host memory where the PFM sort solution will be written.
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

    private:
        /// @brief Number of cells
        uint16_t            num_cell;

        /// @brief CUDA stream for the PFM sorting operation
        cudaStream_t        strm;

        /// @brief GPU main buffer for maintaining PFM sorting data for each cell
        uint8_t*            gpu_main_in_buf;

        /// @brief GPU output buffer for storing the PFM sorting results for each cell
        uint8_t*            gpu_out_buf;

        /// @brief Dynamic descriptor on the host
        std::unique_ptr<pfmSortDynDescr> pCpuDynDesc;

        /// @brief Dynamic descriptor on the device
        pfmSortDynDescr*    pGpuDynDesc;
        
        /// @brief CUDA kernel block dimensions
        uint16_t            numThrdBlk;
        uint16_t            numThrdPerBlk;
    
        /// @brief CUDA kernel grid dimensions
        dim3                gridDim;
        /// @brief CUDA kernel block dimensions
        dim3                blockDim;

        /// @brief Launch configuration structure
        std::unique_ptr<launchCfg_t> pLaunchCfg;

        /// @brief Selects the appropriate kernel for the PFM sorting algorithm
        void kernelSelect();
    };

    /// @brief PFM sorting CUDA kernel
    static __global__ void pfmSortKernel(pfmSortDynDescr* pDynDescr);
}
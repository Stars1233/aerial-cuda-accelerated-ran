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

#if !defined(PRACH_RX_HPP_INCLUDED_)
#define PRACH_RX_HPP_INCLUDED_

#include "cuphy.h"
#include "cuphy_api.h"
#include "cuphy.hpp"
#include "cuphy_hdf5.hpp"
#include "prach_receiver/prach_receiver.hpp"
#include "util.hpp"
#include "utils.cuh"
#include <array>
#include <iostream>
#include <cstdlib>
#include <string>

struct cuphyPrachRx
{};

class PrachRx : public cuphyPrachRx {

public:
    
    // Internal structure definition for grouping debug output parameters
    struct OutputParams
    {
        // debug parameters
        bool                        debugOutputFlag;
        hdf5hpp::hdf5_file          outHdf5File;
    };

    /**
     * @brief: Construct PrachRx class.
     */
     PrachRx(cuphyPrachStatPrms_t const* pStatPrms, cuphyStatus_t* status);

    /**
     * @brief: PrachRx cleanup.
     */
    ~PrachRx();

    /**
     * @brief: PrachRx setup
     * @param[in] dyn_params: input parameters to PRACH.
     */
    cuphyStatus_t expandParameters(cuphyPrachDynPrms_t* pDynPrms);

    /**
     * @brief: Run PRACH
     */
    cuphyStatus_t Run();

    /**
     * @brief Write debug data out to h5 file
     * 
     * @param cuStream 
     */
    void writeDbgBufSynch(cudaStream_t cuStream);


    const void* getMemoryTracker();

    /**
     * @brief Print Static API Parameters
     * 
     */
    template <fmtlog::LogLevel log_level=fmtlog::DBG>
    static void printStatApiPrms(cuphyPrachStatPrms_t const* pStatPrms);
    /**
     * @brief Print Dynamic API Parameters
     * 
     */
    template <fmtlog::LogLevel log_level=fmtlog::DBG>
    static void printDynApiPrms(cuphyPrachDynPrms_t* pDynPrm);

    /**
     * @brief Validate static parameters without instantiating the class
     * 
     * @param[in] pStatPrms Static parameters to validate
     * @return cuphyStatus_t Validation result
     */
    static cuphyStatus_t validateStaticParams(cuphyPrachStatPrms_t const* pStatPrms);

private:
    cuphyMemoryFootprint m_memoryFootprint;

    // number of cells passed during create pipeline
    uint16_t nMaxCells;

    // total number of occasion across all cells passed during create pipeline
    uint16_t nTotCellOcca;

    // max number of occasion that will be processed in a single pipeline processing
    uint16_t nMaxOccasions;

    // number of occasion to be processed, passed during setup pipeline
    uint16_t nOccaProc;

    // processing mode - graph (1) or stream(0)
    uint64_t procModeBmsk = 0;

    uint16_t maxAntenna = 0;
    uint max_l_oran_ant = 0;
    uint max_ant_u = 0;
    uint max_nfft = 0;
    int max_zoneSizeExt = 0;
    const cuphyDeviceArchInfo m_cudaDeviceArchInfo;

    std::vector<PrachInternalStaticParamPerOcca> staticParam;

    // Vector of size nMaxOccasions containing state of each occasion
    // value at each index is 0 or 1
    // 1 - occasion is active, 0 - not active
    std::vector<char> activeOccasions;
    // same as above - active occasions in previous step
    std::vector<char> prevActiveOccasions;

    cuphy::buffer<PrachDeviceInternalStaticParamPerOcca, cuphy::pinned_alloc> h_staticParam;
    cuphy::buffer<PrachDeviceInternalStaticParamPerOcca, cuphy::device_alloc> d_staticParam;
    cuphy::buffer<PrachInternalDynParamPerOcca, cuphy::device_alloc> d_dynParam;
    cuphy::buffer<PrachInternalDynParamPerOcca, cuphy::pinned_alloc> h_dynParam;

    cuphy::buffer<cuFloatComplex *, cuphy::pinned_alloc> h_fftPointers;
    std::array<FftInfo, PRACH_NUM_SUPPORTED_FFT_SIZES> h_fftInfo;

    cuphyTensorPrm_t numDetectedPrmb;
    cuphyTensorPrm_t prmbIndexEstimates;
    cuphyTensorPrm_t prmbDelayEstimates;
    cuphyTensorPrm_t prmbPowerEstimates;
    cuphyTensorPrm_t antRssi;
    cuphyTensorPrm_t rssi;
    cuphyTensorPrm_t interference;

    OutputParams     m_outputPrms;

    // dynamic parameters used in previous step
    // Need to maintain to see if CUDA graph requires updating
    uint32_t* prev_numDetectedPrmb = nullptr;
    uint32_t* prev_prmbIndexEstimates = nullptr;
    float* prev_prmbDelayEstimates = nullptr;
    float* prev_prmbPowerEstimates = nullptr;
    float* prev_antRssi = nullptr;
    float* prev_rssi = nullptr;
    float* prev_interference = nullptr;
    uint16_t nPrevOccaProc;

    cudaStream_t cuStream;

    cudaGraph_t graph;
    cudaGraphExec_t graphInstance;
    std::vector<cudaGraphNode_t> nodes;
};

#endif // !defined(PRACH_RX_HPP_INCLUDED_)

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

#include "cuphy.h"
#include "cuphy_api.h"
#include "tensor_desc.hpp"
#include <mutex>



#pragma once

static constexpr uint32_t F1_CG_SIZE              = 32;
static constexpr uint32_t F1_NUM_UCI_GRPS_PER_BLK = 2;
static constexpr uint32_t MAX_DATA_SYMS_F1        = 7;
static constexpr uint32_t MAX_DMRS_SYMS_F1        = 7;
static constexpr uint32_t F1_MAX_SYMS             = 14;
static constexpr uint32_t F1_MAX_RX_ANTENNA       = 4;

static constexpr float confidenceThrF1 = 0.1; // threshold for determining confidence levels of SR and HARQ values

// Structure to track max UCI shape parameters for dynamic shared memory allocation
struct F1UciShape
{
    uint8_t nAnt     = 0;  // max antennas across all groups
    uint8_t nSymData = 0;  // max data symbols across all groups
    uint8_t nSymDmrs = 0;  // max DMRS symbols across all groups
};
typedef struct F1UciShape F1UciShape_t;

// Implementation of the PUCCH F1 reciever interface exposed as an opaque data type to abstract out implementation
// details (PUCCH F1 reciever C++ class). The PUCCH F1 reciever is implemented as a C++ class which inherits
// from this interface structure defiend as an empty shell (opaque type is a struct since the interface is C
// compatible). Pointer to the opaque type is also exposed in the interface as a handle to the underlying
// implementation
struct cuphyPucchF1Rx
{};


// Parameters for a single PUCCH F1 Group
struct pucchF1UciGrpPrms
{
    uint8_t nUciInGrp;

    // parameters shared by all UCI in group:
    uint8_t  freqHopFlag;
    uint8_t  startSym;
    uint16_t startCrb;
    uint8_t  nSym;
    uint8_t  groupHopFlag;
    uint16_t secondHopCrb;
    uint8_t  u[2];
    uint16_t csCommon[F1_MAX_SYMS];
    uint8_t  nSym_data;
    uint8_t  nSym_dmrs;
    uint8_t  nSymDataFirstHop;
    uint8_t  nSymFirstHop;
    uint8_t  nSymDMRSFirstHop;
    uint8_t  nSymDataSecondHop;
    uint8_t  nSymDMRSSecondHop;
    uint16_t cellIdx;

    // UCI specific parameters:
    uint8_t     bitLenHarq       [CUPHY_PUCCH_F1_MAX_UCI_PER_GRP];
    uint8_t     srFlag           [CUPHY_PUCCH_F1_MAX_UCI_PER_GRP];
    uint8_t     cs0              [CUPHY_PUCCH_F1_MAX_UCI_PER_GRP];
    uint16_t    uciOutputIdx     [CUPHY_PUCCH_F1_MAX_UCI_PER_GRP];
    uint8_t     timeDomainOccIdx [CUPHY_PUCCH_F1_MAX_UCI_PER_GRP];
    __half      DTXthreshold  [CUPHY_PUCCH_F1_MAX_UCI_PER_GRP];
    
    uint16_t nUplinkStreams;
};
typedef struct pucchF1UciGrpPrms pucchF1UciGrpPrms_t;

// Pucch F1 reciever dynamic descriptor
struct pucchF1RxDynDescr
{
    pucchF1UciGrpPrms_t               uciGrpPrms[CUPHY_PUCCH_F1_MAX_GRPS]; // parameters
    cuphyPucchF0F1UciOut_t*           pF1UcisOut;                          // output uci buffer
    cuphyPucchCellPrm_t*              pCellPrms;                           // RX Antennas, slot num , hopping id and input slot buffer
    uint16_t                          numUciGrps;
    uint8_t                           enableUlRxBf;
    // Max UCI shape for dynamic shared memory offsets (computed at setup)
    uint8_t                           maxNAnt;                             // max antennas across all groups
    uint8_t                           maxNSymData;                         // max data symbols across all groups
    uint8_t                           maxNSymDmrs;                         // max DMRS symbols across all groups
};
typedef struct pucchF1RxDynDescr pucchF1RxDynDescr_t;

// PUCCH format 1 kernel arguments (supplied via descriptors)
typedef struct 
{
    pucchF1RxDynDescr_t*  pDynDescr;  // pointer to an array of CUPHY_PUSCH_RX_RSSI_MEAS_N_HOM_CFG dynamic descriptors
} pucchF1KernelArgs_t;


// Class implementation of PUCCH F1 reciver 
// class puschRxChEst : public cuphyPuschRxChEst 
class pucchF1Rx : public cuphyPucchF1Rx 
{
public:
    pucchF1Rx(cudaStream_t strm);
    ~pucchF1Rx()                           = default;
    pucchF1Rx(pucchF1Rx const&)            = delete;
    pucchF1Rx& operator=(pucchF1Rx const&) = delete;

    void InitConstantMem(cudaStream_t strm);

    // setup object state and dynamic component descriptor in prepration towards execution
    void setup(cuphyTensorPrm_t*          pDataRx,
               cuphyPucchF0F1UciOut_t*    pF1UcisOut,
               uint16_t                   nCells,
               uint16_t                   nF1Ucis,
               uint8_t                    enableUlRxBf,
               cuphyPucchUciPrm_t*        pF1UciPrms,
               cuphyPucchCellPrm_t*       pCmnCellPrms,                    // number of antennas, slot number and hopping idx
               bool                       enableCpuToGpuDescrAsyncCpy,
               pucchF1RxDynDescr_t*       pCpuDynDesc,                     // pointer to descriptor in cpu
               void*                      pGpuDynDesc,                     // pointer to descriptor in gpu
               cuphyPucchF1RxLaunchCfg_t* pLaunchCfg,                      // pointer to launch configuration
               cudaStream_t               strm);                           // stream to perform copy


    void kernelSelect(uint32_t                   nUciGrps,
                      pucchF1RxDynDescr_t*       pCpuDynDesc, 
                      F1UciShape_t&              maxShape,
                      cuphyPucchF1RxLaunchCfg_t* pLaunchCfg);


    static void getDescrInfo(size_t& dynDescrSizeBytes, size_t& dynDescrAlignBytes);
    pucchF1KernelArgs_t m_kernelArgs;

    private:
      static bool isConstMemInited;
      static std::mutex m_mutexConstMemInit;
};


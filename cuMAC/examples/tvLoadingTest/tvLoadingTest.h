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
#include "parameters.h"
#include "h5TvCreate.h"
#include "h5TvLoad.h"
#include <chrono>
#include <ctime>
#include <iomanip> // for std::setfill and std::setw
#include <thread>  // for std::this_thread

using namespace cumac;

bool compareCpuGpuAllocSol(cumac::cumacSubcontext* subcontextGpu, cumac::cumacSubcontext* subcontextCpu, uint8_t* modulesCalled)
{
    bool matchUeSel     = true;
    bool matchPrg       = true;
    bool matchLayerSel  = true;
    bool matchMcsSel    = true;

    if (modulesCalled[0] == 1) { // UE selection, compare UE selection solutions for each cell. Order of selected UEs in each cell can be different.
        uint16_t nUe = subcontextGpu->cellGrpPrmsGpu->nUe;
        uint16_t nCell = subcontextGpu->cellGrpPrmsGpu->nCell;
        uint8_t  nUePerCell = subcontextGpu->cellGrpPrmsGpu->numUeSchdPerCellTTI;

        uint16_t* setSchdUePerCellTTI = new uint16_t[nUe];
        CUDA_CHECK_ERR(cudaMemcpy(setSchdUePerCellTTI, subcontextGpu->schdSolGpu->setSchdUePerCellTTI, sizeof(uint16_t)*nUe, cudaMemcpyDeviceToHost));

        for (int cIdx = 0; cIdx < nCell; cIdx++) {
            int offset = cIdx * nUePerCell;

            std::vector<uint16_t> cpuSet(subcontextCpu->schdSolCpu->setSchdUePerCellTTI + offset,
                                         subcontextCpu->schdSolCpu->setSchdUePerCellTTI + offset + nUePerCell);
            std::vector<uint16_t> gpuSet(setSchdUePerCellTTI + offset,
                                         setSchdUePerCellTTI + offset + nUePerCell);

            std::sort(cpuSet.begin(), cpuSet.end());
            std::sort(gpuSet.begin(), gpuSet.end());

            if (cpuSet != gpuSet) {
                matchUeSel = false;
                printf("Failure: CPU and GPU UE selection mismatch at cell %d\n", cIdx);
                for (int uIdx = 0; uIdx < nUePerCell; uIdx++) {
                    printf("  cell %d, uIdx %d: CPU = %d, GPU = %d\n", cIdx, uIdx, cpuSet[uIdx], gpuSet[uIdx]);
                }
                break;
            }
        }

        if (matchUeSel) {
            printf("Success: CPU and GPU UE selection solutions match\n");
        } else {
            printf("Failure: CPU and GPU UE selection solutions do not match\n");
        }
        delete[] setSchdUePerCellTTI;
    }

    if (modulesCalled[1] == 1) { // PRG allocation
        int16_t* allocSol;
        uint32_t gpuAllocSolSize;
        if (subcontextGpu->cellGrpPrmsGpu->allocType == 1) { // type-1 PRB allocation
            gpuAllocSolSize = 2*subcontextGpu->cellGrpPrmsGpu->nUe;
            
        } else { // type-0 PRB allocation
            gpuAllocSolSize = subcontextGpu->cellGrpPrmsGpu->nCell*subcontextGpu->cellGrpPrmsGpu->nPrbGrp;
        }

        allocSol = new int16_t[gpuAllocSolSize];
        CUDA_CHECK_ERR(cudaMemcpy(allocSol, subcontextGpu->schdSolGpu->allocSol, sizeof(int16_t)*gpuAllocSolSize, cudaMemcpyDeviceToHost)); 

        for (int idx = 0; idx < gpuAllocSolSize; idx++) {
            if (subcontextCpu->schdSolCpu->allocSol[idx] != allocSol[idx]) {
                matchPrg = false;
                break;
            }
        }

        if (matchPrg) {
            printf("Success: CPU and GPU PRG allocation solutions match\n");
        } else {
            printf("Failure: CPU and GPU PRG allocation solutions do not match\n");
        }
        delete[] allocSol;
    }

    if (modulesCalled[2] == 1) { // layer selection
        uint8_t* layerSelSol = new uint8_t[subcontextGpu->cellGrpPrmsGpu->nUe];
        CUDA_CHECK_ERR(cudaMemcpy(layerSelSol, subcontextGpu->schdSolGpu->layerSelSol, sizeof(uint8_t)*subcontextGpu->cellGrpPrmsGpu->nUe, cudaMemcpyDeviceToHost)); 

        for (int uIdx = 0; uIdx<subcontextGpu->cellGrpPrmsGpu->nUe; uIdx++) {
            if (subcontextCpu->schdSolCpu->layerSelSol[uIdx] != layerSelSol[uIdx]) {
                matchLayerSel = false;
                break;
            }
        }

        if (matchLayerSel) {
            printf("Success: CPU and GPU layer selection solutions match\n");
        } else {
            printf("Failure: CPU and GPU layer selection solutions do not match\n");
        }
        delete[] layerSelSol;
    }

    if (modulesCalled[3] == 1) { // MCS selection
        int16_t* mcsSelSol = new int16_t[subcontextGpu->cellGrpPrmsGpu->nUe];
        CUDA_CHECK_ERR(cudaMemcpy(mcsSelSol, subcontextGpu->schdSolGpu->mcsSelSol, sizeof(int16_t)*subcontextGpu->cellGrpPrmsGpu->nUe, cudaMemcpyDeviceToHost)); 

        for (int uIdx = 0; uIdx<subcontextGpu->cellGrpPrmsGpu->nUe; uIdx++) {
            if (subcontextCpu->schdSolCpu->mcsSelSol[uIdx] != mcsSelSol[uIdx]) {
                matchMcsSel = false;
                break;
            }
        }

        if (matchMcsSel) {
            printf("Success: CPU and GPU MCS selection solutions match\n");
        } else {
            printf("Failure: CPU and GPU MCS selection solutions do not match\n");
        }
        delete[] mcsSelSol;
    }

    return matchUeSel & matchPrg & matchLayerSel & matchMcsSel ;
}

template <typename T>
void saveVectorToCSV(const std::vector<T>& vec, const std::string& filename) {
    std::ofstream outputFile(filename);

    if (!outputFile.is_open()) {
        std::cerr << "Error: Unable to open file " << filename << " for writing." << std::endl;
        return;
    }

    // Write vector elements to the file, separated by commas
    for (size_t i = 0; i < vec.size(); ++i) {
        if (i > 0) {
            outputFile << ",";
        }
        outputFile << vec[i];
    }

    // Close the file
    outputFile.close();
}

/**
 * @brief run cuMAC subcontext for nItr times per intervalUs microseconds
 * 
 * @param subcontext cumacSubcontext to run scheduler
 * @param nItr number of iterations
 * @param intervalUs interval length in microseconds
 * @param runSlotPattern slot run pattern for 10 slots, 1: run, 0 skip
 * @param printTimeStamp whether print the time stamp for each run: 1 enable; 0 disable
 * @param strm CUDA stream to run cuMAC scheduler
 * 
 * @note: TODO: only called when running CPU scheduler（-g 0) or GPU scheduler (-g 1), not call wjhen running both CPU and GPU schedulers (-g 2)
 */
void runCumacIter(cumac::cumacSubcontext * subcontext, int nItr, int intervalUs, uint8_t* runSlotPattern, bool printTimeStamp, cudaStream_t strm);

/**
 * @brief Get the TDD slot run pattern
 * 
 * @param DL DL (1) or UL (0)
 * @param tddPatternIdx 0: full slot; 1: DDDSUUDDDD; 2: DDDSU; 
 * @param runSlotPattern a pointer with next 10 elements for slot run pattern: 1: run; 0: skip;
 */
inline void getTddRunSlotPattern(uint8_t & DL, uint8_t & tddPatternIdx, uint8_t* & runSlotPattern);

// predefined slot run pattern for DL and UL: 1: run, 0: skip
uint8_t runSlotPatternPool_DL[3][10]=
{
  {1, 1, 1, 1, 1, 1, 1, 1, 1, 1},  // full slot
  {1, 1, 1, 0, 0, 0, 1, 1, 1, 1},  // DDDSUUDDDD
  {1, 1, 1, 0, 0, 1, 1, 1, 0, 0},  // DDDSU
};

uint8_t runSlotPatternPool_UL[3][10]=
{
  {1, 1, 1, 1, 1, 1, 1, 1, 1, 1},  // full slot
  {0, 0, 0, 0, 1, 1, 0, 0, 0, 0},  // DDDSUUDDDD
  {0, 0, 0, 0, 1, 0, 0, 0, 0, 1},  // DDDSU
};
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

#include "cumac.h"

// cuMAC namespace
namespace cumac {

cellAssociationCpu::cellAssociationCpu()
{
    // allocate buffer for CPU descriptor
    m_pCpuDynDesc = new caDynDescr_t;
}

cellAssociationCpu::~cellAssociationCpu()
{
    delete m_pCpuDynDesc;
}

void cellAssociationCpu::setup(cumacCellGrpPrms* cellGrpPrms, cumacSimParam* simParam)
{
    m_pCpuDynDesc -> estH_fr    = cellGrpPrms -> estH_fr;
    m_pCpuDynDesc -> cellAssoc  = cellGrpPrms -> cellAssoc;
    m_pCpuDynDesc -> nUe        = cellGrpPrms -> nUe;
    // FIXME: currently assuming nCell = totNumCell
    // cellGrpPrms -> nCell       :   number of cooperate cells
    // simParam    -> totNumCell  :   number of total cells (cooperate + interference)
    m_pCpuDynDesc -> totNumCell = simParam -> totNumCell;
    m_pCpuDynDesc -> nPrbGrp    = cellGrpPrms -> nPrbGrp;
    m_pCpuDynDesc -> nBsAnt     = cellGrpPrms -> nBsAnt;
    m_pCpuDynDesc -> nUeAnt     = cellGrpPrms -> nUeAnt;
}

void cellAssociationCpu::run()
{
    cuComplex * estH_fr   =  m_pCpuDynDesc -> estH_fr;
    uint8_t  * cellAssoc  =  m_pCpuDynDesc -> cellAssoc;
    uint16_t & nUe        =  m_pCpuDynDesc -> nUe;
    uint16_t & totNumCell =  m_pCpuDynDesc -> totNumCell;
    uint16_t & nPrbGrp    =  m_pCpuDynDesc -> nPrbGrp;
    uint8_t  & nBsAnt     =  m_pCpuDynDesc -> nBsAnt;
    uint8_t  & nUeAnt     =  m_pCpuDynDesc -> nUeAnt;
    
    #ifdef CELLASSOCIATION_KERNEL_TIME_MEASURE_
    auto start = std::chrono::steady_clock::now();
    for (int idx = 0; idx < numRunSchKnlTimeMsr; idx++) {
    #endif
        // clear all cell association, later can change to change a specific UE
        for (int init_index = 0; init_index < totNumCell*nUe; init_index++ )
        {
            cellAssoc [init_index] = 0;
        }
        
        // calculate the mean gain  per UE to each BS
        for (int ueIdx = 0; ueIdx < nUe; ueIdx++) 
        {
            float bestMetric = 0.0f;
            uint16_t bestCellIdx = 0.0f;
            for (int cellIdx = 0; cellIdx < totNumCell; cellIdx++) 
            {
                float assocMetric = 0.0f;
                for (int prbIdx = 0; prbIdx < nPrbGrp; prbIdx++) 
                {
                    // generate channel coefficients per antenna pair
                    for (int txAntIdx = 0; txAntIdx < nBsAnt; txAntIdx++) 
                    {
                        for (int rxAntIdx = 0; rxAntIdx < nUeAnt; rxAntIdx++) 
                        {
                            int index = prbIdx*nUe*totNumCell*nBsAnt*nUeAnt;
                            index += ueIdx*totNumCell*nBsAnt*nUeAnt;
                            index += cellIdx*nBsAnt*nUeAnt;
                            index += txAntIdx*nUeAnt;
                            index += rxAntIdx;

                            cuComplex current_chl = estH_fr[index];
                            float temp_gain = pow(current_chl.x,2) + pow(current_chl.y,2) ;
                            //calculate the expected data rate per cell
                            // float temp_gain = log2(1+(pow(current_chl.x,2)+pow(current_chl.y,2))/noise_sigma2);
                            //calculate the average channel gain
                            assocMetric += temp_gain;

                            // if(ueIdx == 0 ) 
                            // {
                            //     printf("current_chl.x = %f, current_chl.y = %f, temp_gain = %.4f, index = %d \n", float(current_chl.x), float(current_chl.y), temp_gain, index); 
                            // }
                        }
                    }
                }
                // update best cell, use the best ave data rate for cell association
                if(bestMetric <= assocMetric)
                {
                    bestMetric  = assocMetric;
                    bestCellIdx = cellIdx;
                }
            }
            // assign ue to the best cell idx
            cellAssoc[bestCellIdx*nUe + ueIdx] = 1;
        }

        //printf("Cell Association done!");
    #ifdef CELLASSOCIATION_KERNEL_TIME_MEASURE_
    }
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed_seconds = end-start;
    printf("cell association on CPU elapsed time: %f ms\n", 1000.0*elapsed_seconds.count()/static_cast<float>(numRunSchKnlTimeMsr));
    #endif
}
}
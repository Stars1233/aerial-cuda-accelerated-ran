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

 // debug parameters
 // #define OUTPUT_SOLUTION_
 // #define LIMIT_NUM_SM_TIME_MEASURE_
 // #define MCSCHEDULER_DEBUG_
 // #define SCSCHEDULER_DEBUG_
 // #define CHANN_INPUT_DEBUG_
 // #define CELLASSOCIATION_PRINT_SAMPLE_

 // GPU index
 #define gpuDeviceIdx           0 // index of GPU device to use

 // simulation duration
 #define numSimChnRlz           2000 //total number of simulated TTIs (e.g., 15000 for 1200 active UEs per cell; 5000 for 500 active UEs per cell)
 
 // randomness
 #define seedConst              0 // randomness seed

 // system parameters
 //#define mu                     0 // OFDM numerology: 0, 1, 2, 3, 4
 #define slotDurationConst      0.5e-3 // 1.0e-3, 0.5e-3, 0.25e-3, 0.125e-3, 0.0625e-3
 #define scsConst               30000.0 // 15000.0, 30000.0, 60000.0, 120000.0, 240000.0 corresponding to OFDM numerology: 0, 1, 2, 3, 4
 #define numMcsLevels           28
 #define cellRadiusConst        1000
 #define numCellConst           20 // total number of cells in the network, including coordinated cells and interfering cells
 #define numCoorCellConst       numCellConst // currently support max 21 coordinated cells
 #define numUePerCellConst      16 // number of UEs scheduled per time slot per cell
 #define numUeForGrpConst       32 // number of UEs considered for MU-MIMO UE grouping per TTI per cell
 // assumption's that numUePerCellConst <= numUeForGrpConst
 #define numActiveUePerCellConst 500 // 100, 500, 1200. should be <= 2048
 #define totNumUesConst         numCellConst*numUePerCellConst // total number of scheduled UEs per TTI that are associated with the coordinated cells   
 #define totNumActiveUesConst   numCellConst*numActiveUePerCellConst // total number of active UEs associated with the coordinated cells   
 // antenna configurations
 // *AntSize, *AntSpacing, *AntPolarAngles, *AntPattern, vDirection are only used in CDL channel model and n*Ant must be equal to prod(*AntSize)
 // for other channel models, *AntSize, *AntSpacing, *AntPolarAngles, *AntPattern, vDirection are not used
 // by default, UE uses isotropic antennas, BS uses directional antennas
 #define nBsAntConst            4 
#define bsAntSizeConst          {1,1,1,2,2} // {M_g,N_g,M,N,P} 3GPP TR 38.901 Section 7.3
#define bsAntSpacingConst       {1.0f, 1.0f, 0.5f, 0.5f} // BS antenna spacing [d_g_h, d_g_v, d_h, d_v] in wavelengths
 #define bsAntPolarAnglesConst  {45.0f, -45.0f} // BS antenna polarization angles
 #define bsAntPatternConst      1 // 0: isotropic; 1: 38.901
 #define nUeAntConst            4 // assumption's that nUeAntConst <= nBsAntConst; nUeAntConst is also equal to the maximum number of layers
#define ueAntSizeConst          {1,1,2,2,1} // {M_g,N_g,M,N,P} 3GPP TR 38.901 Section 7.3
#define ueAntSpacingConst       {1.0f, 1.0f, 0.5f, 0.5f} // UE antenna spacing [d_g_h, d_g_v, d_h, d_v] in wavelengths
 #define ueAntPolarAnglesConst  {0.0f, 90.0f} // UE antenna polarization angles
 #define ueAntPatternConst      0  // 0: isotropic; 1: 38.901
 #define vDirectionConst        {90, 0} // moving direction, [RxA; RxZ] — RxA and RxZ specify the azimuth and zenith of the direction of travel of the moving UE; moving speed is converted to maxDopplerShift in cdlCfg
 #define nPrbsPerGrpConst       4
 #define nPrbGrpsConst          68
 #define WConst                 12.0*scsConst*nPrbsPerGrpConst
 #define totWConst              WConst*nPrbGrpsConst
 #define PtConst                79.4328 // Macrocell - 49.0 dBm (79.4328 W), Microcell - 23 dBm (0.1995 W)
 #define PtRbgConst             PtConst/nPrbGrpsConst
 #define PtRbgAntConst          PtRbgConst/nBsAntConst
 #define bandwidthRBConst       12*scsConst
 #define bandwidthRBGConst      nPrbsPerGrpConst*bandwidthRBConst
 #define noiseFigureConst       9 // dB
 // For testing need to adjust noise variance based on channel gain
 #define sigmaSqrdDBmConst      -174 + noiseFigureConst+ 10*log10(bandwidthRBGConst)
 #define sigmaSqrdConst         pow(10.0, ((sigmaSqrdDBmConst - 30.0)/10.0))
 #define gpuAllocTypeConst      1 // 0 - non-consecutive type 0 allocate, 1 - consecutive type 1 allocate
 #define cpuAllocTypeConst      1 // 0 - non-consecutive type 0 allocate, 1 - consecutive type 1 allocate
 #define prdSchemeConst         0 // 0 - no precoding, 1 - SVD precoding
 #define rxSchemeConst          1 // 1 - MMSE-IRC
 #define heteroUeSelCellsConst  0 // 0 - homogeneous UE selection config. across cells, 1 - heterogeneous UE selection config. across cells
 // heterogeneous UE selection config. currently not supported for performance benchmarking vs. RR scheduler

 // max dimentions
 #define maxNumCoorCellConst    21
 #define maxNumBsAntConst       16
 #define maxNumUeAntConst       16
 #define maxNumPrbGrpsConst     100

 // buffer size
 #define estHfrSizeCOnst        nPrbGrpsConst*totNumUesConst*numCoorCellConst*nBsAntConst*nUeAntConst
 
 // PDSCH parameters
 #define pdschNrOfSymbols       12
 #define pdschNrOfDmrsSymb      1
 #define pdschNrOfDataSymb      pdschNrOfSymbols-pdschNrOfDmrsSymb
 #define pdschNrOfLayers        1

 // PF scheduling
 #define initAvgRateConst       1.0
 #define pfAvgRateUpdConst      0.001
 #define betaCoeffConst         1.0
 #define sinValThrConst         0.1
 #define prioWeightStepConst    100
 // power scaling
 #define AFTER_SCALING_SIGMA_CONST 1.0 // noise std after scaling to improve precision
 // 1.0 for 49.0 dBm BS Tx power

 #define cpuGpuPerfGapPerUeConst 0.005
 #define cpuGpuPerfGapSumRConst 0.01
 // interference control
 #define toleranceConst         0.4

 // SVD precoder parameters
 #define svdToleranceConst      1.e-7
 #define svdMaxSweeps           15

 // Normalized channel coefficients for __half range
 #define amplifyCoeConst        1

 // output file
 #define mcOutputFile           "output.txt"     
 #define mcOutputFileShort      "output_short.txt"

#define targetChanCoeRangeConst 0.1f * nPrbGrpsConst * totNumUesConst // target channel coefficients range for precision issue
#define MinNoiseRangeConst      0.001f // minimum noise figure for stability issues 

// 64TR MU-MIMO parameters
#define nMaxUeSchdPerCellTTIConst 16

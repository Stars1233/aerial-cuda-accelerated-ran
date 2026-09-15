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

#include <cstdint>
#include <string>
#include <vector>

// debug parameters (kept as compile-time toggles)
// #define OUTPUT_SOLUTION_
// #define LIMIT_NUM_SM_TIME_MEASURE_
// #define MCSCHEDULER_DEBUG_
// #define SCSCHEDULER_DEBUG_
// #define CHANN_INPUT_DEBUG_
// #define CELLASSOCIATION_PRINT_SAMPLE_

namespace cumac {

// Runtime-loaded simulation parameters. Loaded once at program start from
// parameters.yaml; downstream code accesses values through the `Const`-suffixed
// macros defined below (back-compat shims over the global g_params instance).
struct Params {
    int gpuDeviceIdx;                       //!< CUDA device index used by examples.

    int numSimChnRlz;                       //!< Number of simulated channel realizations (TTIs).

    int seedConst;                          //!< Seed for std::srand-based randomness.

    double slotDurationConst;               //!< Slot duration, seconds.
    double scsConst;                        //!< Subcarrier spacing, Hz.
    int    numMcsLevels;                    //!< Number of MCS levels modelled.
    int    cellRadiusConst;                 //!< Cell radius, metres.
    int    numCellConst;                    //!< Number of cells in the network.
    int    numUePerCellConst;               //!< UEs scheduled per cell per TTI.
    int    numUeForGrpConst;                //!< UE group size for grouping heuristics.
    int    numActiveUePerCellConst;         //!< Total active UEs per cell (population size).

    int                    nBsAntConst;            //!< BS antennas per cell.
    std::vector<uint16_t>  bsAntSizeConst;         //!< BS antenna shape, 5 elements {M_g, N_g, M, N, P}.
    std::vector<float>     bsAntSpacingConst;      //!< BS antenna spacing (wavelengths), 4 elements {dg_H, dg_V, d_H, d_V}.
    std::vector<float>     bsAntPolarAnglesConst;  //!< BS polarization angles in degrees, P entries (typically 2).
    int                    bsAntPatternConst;      //!< BS element pattern id (0=isotropic, 1=3GPP TR38.901, ...).
    int                    nUeAntConst;            //!< UE antennas.
    std::vector<uint16_t>  ueAntSizeConst;         //!< UE antenna shape, 5 elements {M_g, N_g, M, N, P}.
    std::vector<float>     ueAntSpacingConst;      //!< UE antenna spacing (wavelengths), 4 elements {dg_H, dg_V, d_H, d_V}.
    std::vector<float>     ueAntPolarAnglesConst;  //!< UE polarization angles in degrees, P entries (typically 2).
    int                    ueAntPatternConst;      //!< UE element pattern id.
    std::vector<float>     vDirectionConst;        //!< UE velocity direction {azimuth, zenith} in degrees, 2 elements.

    int nPrbsPerGrpConst;                   //!< PRBs per PRB-group (RBG). Must be > 0.
    int nPrbGrpsConst;                      //!< Number of PRB-groups. Must be > 0.

    double PtConst;                         //!< Per-cell total transmit power, linear scale.

    double noiseFigureConst;                //!< Receiver noise figure, dB.

    int gpuAllocTypeConst;                  //!< GPU UE-selection allocator variant.
    int cpuAllocTypeConst;                  //!< CPU UE-selection allocator variant.
    int prdSchemeConst;                     //!< Precoder scheme id.
    int rxSchemeConst;                      //!< Receiver scheme id.
    int heteroUeSelCellsConst;              //!< Enable heterogeneous per-cell UE selection (0/1).

    int maxNumCoorCellConst;                //!< Upper bound on coordinated cells (sizing constant).
    int maxNumBsAntConst;                   //!< Upper bound on BS antennas (sizing constant).
    int maxNumUeAntConst;                   //!< Upper bound on UE antennas (sizing constant).
    int maxNumPrbGrpsConst;                 //!< Upper bound on PRB-groups (sizing constant).

    int pdschNrOfSymbols;                   //!< OFDM symbols in a PDSCH allocation.
    int pdschNrOfDmrsSymb;                  //!< DMRS symbols inside that allocation.
    int pdschNrOfLayers;                    //!< Spatial layers per PDSCH.

    double initAvgRateConst;                //!< Initial PF average rate.
    double pfAvgRateUpdConst;               //!< PF average-rate update step (EWMA alpha).
    double betaCoeffConst;                  //!< PF fairness exponent.
    double sinValThrConst;                  //!< SVD singular-value threshold for layer drop.
    int    prioWeightStepConst;             //!< Integer priority-weight quantization step.

    double AFTER_SCALING_SIGMA_CONST;       //!< Post-power-scaling noise variance.

    double cpuGpuPerfGapPerUeConst;         //!< Per-UE CPU vs GPU runtime gap heuristic.
    double cpuGpuPerfGapSumRConst;          //!< Sum-rate CPU vs GPU gap heuristic.

    double toleranceConst;                  //!< Inter-UE interference tolerance.

    double svdToleranceConst;               //!< SVD convergence tolerance.
    int    svdMaxSweeps;                    //!< SVD maximum Jacobi sweeps.

    int amplifyCoeConst;                    //!< Channel-coefficient amplification flag.

    std::string mcOutputFile;               //!< Long-form per-TTI result output path.
    std::string mcOutputFileShort;          //!< Summary result output path.

    double targetChanCoeRangeScale;         //!< Scale factor for derived targetChanCoeRangeConst.
    float  MinNoiseRangeConst;              //!< Minimum normalized noise variance.

    int nMaxUeSchdPerCellTTIConst;          //!< Cap on UEs scheduled per cell per TTI (64TR MU-MIMO).

    //
    // Derived values, computed in loadParameters() after the above are read.
    //
    int    numCoorCellConst;        //!< Coordinated cells in current configuration; = numCellConst.
    int    totNumUesConst;          //!< Total UEs across the network; = numCellConst * numUePerCellConst.
    int    totNumActiveUesConst;    //!< Total active UEs; = numCellConst * numActiveUePerCellConst.
    double WConst;                  //!< RBG bandwidth in Hz; = 12 * scsConst * nPrbsPerGrpConst.
    double totWConst;               //!< Total bandwidth in Hz; = WConst * nPrbGrpsConst.
    double PtRbgConst;              //!< Per-RBG transmit power; = PtConst / nPrbGrpsConst.
    double PtRbgAntConst;           //!< Per-RBG per-antenna power; = PtRbgConst / nBsAntConst.
    double bandwidthRBConst;        //!< PRB bandwidth in Hz; = 12 * scsConst.
    double bandwidthRBGConst;       //!< RBG bandwidth in Hz; = nPrbsPerGrpConst * bandwidthRBConst.
    double sigmaSqrdDBmConst;       //!< Noise power, dBm; = -174 + nf + 10*log10(bandwidthRBGConst).
    double sigmaSqrdConst;          //!< Noise power, linear; = 10^((sigmaSqrdDBmConst - 30)/10).
    int    estHfrSizeCOnst;         //!< Estimated H-frame element count for sizing buffers.
    int    pdschNrOfDataSymb;       //!< Data symbols in PDSCH; = pdschNrOfSymbols - pdschNrOfDmrsSymb.
    float  targetChanCoeRangeConst; //!< Target channel-coefficient range; = scale * nPrbGrpsConst * totNumUesConst.
};

// Single global parameter instance. Populated by loadParameters().
extern Params g_params;

/**
 * Load runtime parameters from a YAML file into ::cumac::g_params.
 *
 * Resolution order for the YAML path (first existing file wins):
 *   1. explicit `yamlPath` argument (if non-empty)
 *   2. CUMAC_PARAMS_YAML environment variable
 *   3. "parameters.yaml" in the current working directory
 *   4. "<exe-dir>/parameters.yaml"
 *   5. "<exe-dir>/../examples/parameters.yaml"
 *
 * If no file is found, or the file is malformed, or individual keys have the
 * wrong type, a warning is logged and the compiled-in defaults are used
 * (byte-identical to the legacy parameters.h #define values). The call never
 * throws on bad input; it is safe to invoke once at program start.
 *
 * Safe to call multiple times; later calls fully replace g_params.
 *
 * @param[in] yamlPath Optional override path. Empty means "auto-discover".
 */
void loadParameters(const std::string& yamlPath = "");

} // namespace cumac

// ---------------------------------------------------------------------------
// Back-compat shims: existing source code uses bare identifiers like
// `numCellConst` or `gpuDeviceIdx`. These macros redirect to g_params so we
// don't have to rewrite hundreds of call sites.
//
// parameters.cpp defines CUMAC_PARAMETERS_NO_MACROS before including this
// header so it can manipulate the struct members by their real names.
// ---------------------------------------------------------------------------
#ifndef CUMAC_PARAMETERS_NO_MACROS
#define gpuDeviceIdx                  (::cumac::g_params.gpuDeviceIdx)
#define numSimChnRlz                  (::cumac::g_params.numSimChnRlz)
#define seedConst                     (::cumac::g_params.seedConst)
#define slotDurationConst             (::cumac::g_params.slotDurationConst)
#define scsConst                      (::cumac::g_params.scsConst)
#define numMcsLevels                  (::cumac::g_params.numMcsLevels)
#define cellRadiusConst               (::cumac::g_params.cellRadiusConst)
#define numCellConst                  (::cumac::g_params.numCellConst)
#define numUePerCellConst             (::cumac::g_params.numUePerCellConst)
#define numUeForGrpConst              (::cumac::g_params.numUeForGrpConst)
#define numActiveUePerCellConst       (::cumac::g_params.numActiveUePerCellConst)
#define nBsAntConst                   (::cumac::g_params.nBsAntConst)
#define bsAntSizeConst                (::cumac::g_params.bsAntSizeConst)
#define bsAntSpacingConst             (::cumac::g_params.bsAntSpacingConst)
#define bsAntPolarAnglesConst         (::cumac::g_params.bsAntPolarAnglesConst)
#define bsAntPatternConst             (::cumac::g_params.bsAntPatternConst)
#define nUeAntConst                   (::cumac::g_params.nUeAntConst)
#define ueAntSizeConst                (::cumac::g_params.ueAntSizeConst)
#define ueAntSpacingConst             (::cumac::g_params.ueAntSpacingConst)
#define ueAntPolarAnglesConst         (::cumac::g_params.ueAntPolarAnglesConst)
#define ueAntPatternConst             (::cumac::g_params.ueAntPatternConst)
#define vDirectionConst               (::cumac::g_params.vDirectionConst)
#define nPrbsPerGrpConst              (::cumac::g_params.nPrbsPerGrpConst)
#define nPrbGrpsConst                 (::cumac::g_params.nPrbGrpsConst)
#define PtConst                       (::cumac::g_params.PtConst)
#define noiseFigureConst              (::cumac::g_params.noiseFigureConst)
#define gpuAllocTypeConst             (::cumac::g_params.gpuAllocTypeConst)
#define cpuAllocTypeConst             (::cumac::g_params.cpuAllocTypeConst)
#define prdSchemeConst                (::cumac::g_params.prdSchemeConst)
#define rxSchemeConst                 (::cumac::g_params.rxSchemeConst)
#define heteroUeSelCellsConst         (::cumac::g_params.heteroUeSelCellsConst)
#define maxNumCoorCellConst           (::cumac::g_params.maxNumCoorCellConst)
#define maxNumBsAntConst              (::cumac::g_params.maxNumBsAntConst)
#define maxNumUeAntConst              (::cumac::g_params.maxNumUeAntConst)
#define maxNumPrbGrpsConst            (::cumac::g_params.maxNumPrbGrpsConst)
#define pdschNrOfSymbols              (::cumac::g_params.pdschNrOfSymbols)
#define pdschNrOfDmrsSymb             (::cumac::g_params.pdschNrOfDmrsSymb)
#define pdschNrOfLayers               (::cumac::g_params.pdschNrOfLayers)
#define initAvgRateConst              (::cumac::g_params.initAvgRateConst)
#define pfAvgRateUpdConst             (::cumac::g_params.pfAvgRateUpdConst)
#define betaCoeffConst                (::cumac::g_params.betaCoeffConst)
#define sinValThrConst                (::cumac::g_params.sinValThrConst)
#define prioWeightStepConst           (::cumac::g_params.prioWeightStepConst)
#define AFTER_SCALING_SIGMA_CONST     (::cumac::g_params.AFTER_SCALING_SIGMA_CONST)
#define cpuGpuPerfGapPerUeConst       (::cumac::g_params.cpuGpuPerfGapPerUeConst)
#define cpuGpuPerfGapSumRConst        (::cumac::g_params.cpuGpuPerfGapSumRConst)
#define toleranceConst                (::cumac::g_params.toleranceConst)
#define svdToleranceConst             (::cumac::g_params.svdToleranceConst)
#define svdMaxSweeps                  (::cumac::g_params.svdMaxSweeps)
#define amplifyCoeConst               (::cumac::g_params.amplifyCoeConst)
#define mcOutputFile                  (::cumac::g_params.mcOutputFile)
#define mcOutputFileShort             (::cumac::g_params.mcOutputFileShort)
#define targetChanCoeRangeScale       (::cumac::g_params.targetChanCoeRangeScale)
#define MinNoiseRangeConst            (::cumac::g_params.MinNoiseRangeConst)
#define nMaxUeSchdPerCellTTIConst     (::cumac::g_params.nMaxUeSchdPerCellTTIConst)

// derived
#define numCoorCellConst              (::cumac::g_params.numCoorCellConst)
#define totNumUesConst                (::cumac::g_params.totNumUesConst)
#define totNumActiveUesConst          (::cumac::g_params.totNumActiveUesConst)
#define WConst                        (::cumac::g_params.WConst)
#define totWConst                     (::cumac::g_params.totWConst)
#define PtRbgConst                    (::cumac::g_params.PtRbgConst)
#define PtRbgAntConst                 (::cumac::g_params.PtRbgAntConst)
#define bandwidthRBConst              (::cumac::g_params.bandwidthRBConst)
#define bandwidthRBGConst             (::cumac::g_params.bandwidthRBGConst)
#define sigmaSqrdDBmConst             (::cumac::g_params.sigmaSqrdDBmConst)
#define sigmaSqrdConst                (::cumac::g_params.sigmaSqrdConst)
#define estHfrSizeCOnst               (::cumac::g_params.estHfrSizeCOnst)
#define pdschNrOfDataSymb             (::cumac::g_params.pdschNrOfDataSymb)
#define targetChanCoeRangeConst       (::cumac::g_params.targetChanCoeRangeConst)

#endif // CUMAC_PARAMETERS_NO_MACROS

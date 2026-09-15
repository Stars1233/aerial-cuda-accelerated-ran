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

// Suppress the back-compat macros in parameters.h so we can access struct
// members by their actual names inside this translation unit.
#define CUMAC_PARAMETERS_NO_MACROS 1
#include "parameters.h"

#include <yaml-cpp/yaml.h>

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace cumac {

Params g_params{};

namespace {

template <typename T>
T optional_or(const YAML::Node& root, const char* key, T fallback)
{
    if (!root[key]) {
        return fallback;
    }
    try {
        return root[key].as<T>();
    } catch (const std::exception& e) {
        std::cerr << "[parameters] WARNING: key '" << key
                  << "' has wrong type (" << e.what()
                  << ") — using compiled-in default\n";
        return fallback;
    }
}

// Fill g_params with the original parameters.h #define values. Called
// unconditionally before YAML parsing so missing keys (or a missing YAML
// file entirely) fall back to the legacy compiled-in behaviour.
void applyHardcodedDefaults(Params& p)
{
    p.gpuDeviceIdx              = 0;
    p.numSimChnRlz              = 2000;
    p.seedConst                 = 0;
    p.slotDurationConst         = 0.5e-3;
    p.scsConst                  = 30000.0;
    p.numMcsLevels              = 28;
    p.cellRadiusConst           = 1000;
    p.numCellConst              = 20;
    p.numUePerCellConst         = 16;
    p.numUeForGrpConst          = 32;
    p.numActiveUePerCellConst   = 500;

    p.nBsAntConst               = 4;
    p.bsAntSizeConst            = {1, 1, 1, 2, 2};
    p.bsAntSpacingConst         = {1.0f, 1.0f, 0.5f, 0.5f};
    p.bsAntPolarAnglesConst     = {45.0f, -45.0f};
    p.bsAntPatternConst         = 1;
    p.nUeAntConst               = 4;
    p.ueAntSizeConst            = {1, 1, 2, 2, 1};
    p.ueAntSpacingConst         = {1.0f, 1.0f, 0.5f, 0.5f};
    p.ueAntPolarAnglesConst     = {0.0f, 90.0f};
    p.ueAntPatternConst         = 0;
    p.vDirectionConst           = {90.0f, 0.0f};

    p.nPrbsPerGrpConst          = 4;
    p.nPrbGrpsConst             = 68;
    p.PtConst                   = 79.4328;
    p.noiseFigureConst          = 9.0;

    p.gpuAllocTypeConst         = 1;
    p.cpuAllocTypeConst         = 1;
    p.prdSchemeConst            = 0;
    p.rxSchemeConst             = 1;
    p.heteroUeSelCellsConst     = 0;

    p.maxNumCoorCellConst       = 21;
    p.maxNumBsAntConst          = 16;
    p.maxNumUeAntConst          = 16;
    p.maxNumPrbGrpsConst        = 100;

    p.pdschNrOfSymbols          = 12;
    p.pdschNrOfDmrsSymb         = 1;
    p.pdschNrOfLayers           = 1;

    p.initAvgRateConst          = 1.0;
    p.pfAvgRateUpdConst         = 0.001;
    p.betaCoeffConst            = 1.0;
    p.sinValThrConst            = 0.1;
    p.prioWeightStepConst       = 100;

    p.AFTER_SCALING_SIGMA_CONST = 1.0;

    p.cpuGpuPerfGapPerUeConst   = 0.005;
    p.cpuGpuPerfGapSumRConst    = 0.01;

    p.toleranceConst            = 0.4;

    p.svdToleranceConst         = 1.e-7;
    p.svdMaxSweeps              = 15;

    p.amplifyCoeConst           = 1;

    p.mcOutputFile              = "output.txt";
    p.mcOutputFileShort         = "output_short.txt";

    p.targetChanCoeRangeScale   = 0.1;
    p.MinNoiseRangeConst        = 0.001f;

    p.nMaxUeSchdPerCellTTIConst = 16;
}

// Verify the inputs that computeDerived() divides by or feeds into log10().
// Returns true on success; on failure logs which field is bad and returns false
// so the caller can revert to compiled-in defaults.
bool validateForComputeDerived(const Params& p)
{
    auto badPositive = [](const char* name, auto v) {
        if (v <= 0) {
            std::cerr << "[parameters] WARNING: '" << name << "' must be > 0 (got "
                      << v << ") — using compiled-in defaults\n";
            return true;
        }
        return false;
    };
    return !badPositive("nPrbGrpsConst",   p.nPrbGrpsConst)
        && !badPositive("nPrbsPerGrpConst", p.nPrbsPerGrpConst)
        && !badPositive("nBsAntConst",      p.nBsAntConst)
        && !badPositive("nUeAntConst",      p.nUeAntConst)
        && !badPositive("scsConst",         p.scsConst);
}

// Antenna/direction vectors are consumed positionally by the channel model and
// must have a fixed element count: antenna-shape {M_g,N_g,M,N,P} = 5, spacing
// {dg_H,dg_V,d_H,d_V} = 4, polarization angles = 2, velocity direction = 2.
// A wrong-length array from the YAML would risk out-of-bounds access, so revert
// any mis-sized vector to its compiled-in default with a warning.
void validateVectorLengths(Params& p)
{
    Params d{};
    applyHardcodedDefaults(d);
    auto checkVec = [](const char* name, auto& v, const auto& def, size_t expected) {
        if (v.size() != expected) {
            std::cerr << "[parameters] WARNING: '" << name << "' must have " << expected
                      << " elements (got " << v.size() << ") — using compiled-in default\n";
            v = def;
        }
    };
    checkVec("bsAntSizeConst",        p.bsAntSizeConst,        d.bsAntSizeConst,        5);
    checkVec("bsAntSpacingConst",     p.bsAntSpacingConst,     d.bsAntSpacingConst,     4);
    checkVec("bsAntPolarAnglesConst", p.bsAntPolarAnglesConst, d.bsAntPolarAnglesConst, 2);
    checkVec("ueAntSizeConst",        p.ueAntSizeConst,        d.ueAntSizeConst,        5);
    checkVec("ueAntSpacingConst",     p.ueAntSpacingConst,     d.ueAntSpacingConst,     4);
    checkVec("ueAntPolarAnglesConst", p.ueAntPolarAnglesConst, d.ueAntPolarAnglesConst, 2);
    checkVec("vDirectionConst",       p.vDirectionConst,       d.vDirectionConst,       2);
}

void computeDerived(Params& p)
{
    p.numCoorCellConst     = p.numCellConst;
    p.totNumUesConst       = p.numCellConst * p.numUePerCellConst;
    p.totNumActiveUesConst = p.numCellConst * p.numActiveUePerCellConst;
    p.WConst               = 12.0 * p.scsConst * p.nPrbsPerGrpConst;
    p.totWConst            = p.WConst * p.nPrbGrpsConst;
    p.PtRbgConst           = p.PtConst / p.nPrbGrpsConst;
    p.PtRbgAntConst        = p.PtRbgConst / p.nBsAntConst;
    p.bandwidthRBConst     = 12.0 * p.scsConst;
    p.bandwidthRBGConst    = p.nPrbsPerGrpConst * p.bandwidthRBConst;
    p.sigmaSqrdDBmConst    = -174.0 + p.noiseFigureConst + 10.0 * std::log10(p.bandwidthRBGConst);
    p.sigmaSqrdConst       = std::pow(10.0, (p.sigmaSqrdDBmConst - 30.0) / 10.0);
    p.estHfrSizeCOnst      = p.nPrbGrpsConst * p.totNumUesConst * p.numCoorCellConst
                             * p.nBsAntConst * p.nUeAntConst;
    p.pdschNrOfDataSymb    = p.pdschNrOfSymbols - p.pdschNrOfDmrsSymb;
    p.targetChanCoeRangeConst = static_cast<float>(p.targetChanCoeRangeScale)
                                * p.nPrbGrpsConst * p.totNumUesConst;
}

// Resolve a YAML path WITHOUT throwing on miss. Empty string means
// "no file was found"; the caller then runs with hard-coded defaults.
std::string findYamlPath(const std::string& explicitPath)
{
    namespace fs = std::filesystem;

    auto tryPath = [](const fs::path& p) -> std::string {
        std::error_code ec;
        if (!p.empty() && fs::exists(p, ec) && !ec) {
            return fs::absolute(p, ec).string();
        }
        return {};
    };

    if (!explicitPath.empty()) {
        if (auto r = tryPath(explicitPath); !r.empty()) return r;
        std::cerr << "[parameters] WARNING: requested YAML file not found: "
                  << explicitPath << " — falling back to lower-priority locations\n";
    }

    if (const char* env = std::getenv("CUMAC_PARAMS_YAML")) {
        if (auto r = tryPath(env); !r.empty()) return r;
        std::cerr << "[parameters] WARNING: CUMAC_PARAMS_YAML points to missing file: "
                  << env << " — falling back to lower-priority locations\n";
    }

    if (auto r = tryPath("parameters.yaml"); !r.empty()) return r;

    std::error_code ec;
    fs::path self = fs::read_symlink("/proc/self/exe", ec);
    if (!ec) {
        fs::path exeDir = self.parent_path();
        if (auto r = tryPath(exeDir / "parameters.yaml"); !r.empty()) return r;
        if (auto r = tryPath(exeDir / ".." / "examples" / "parameters.yaml"); !r.empty()) return r;
        if (auto r = tryPath(exeDir / ".." / ".." / "examples" / "parameters.yaml"); !r.empty()) return r;
    }

    return {};
}

} // namespace

void loadParameters(const std::string& yamlPath)
{
    Params p{};
    applyHardcodedDefaults(p);

    const std::string path = findYamlPath(yamlPath);
    if (path.empty()) {
        std::cout << "[parameters] no YAML file found; using compiled-in defaults" << std::endl;
        computeDerived(p);
        g_params = std::move(p);
        return;
    }

    std::cout << "[parameters] loading: " << path << std::endl;
    YAML::Node y;
    try {
        y = YAML::LoadFile(path);
    } catch (const std::exception& e) {
        std::cerr << "[parameters] WARNING: failed to parse " << path << " (" << e.what()
                  << ") — using compiled-in defaults\n";
        computeDerived(p);
        g_params = std::move(p);
        return;
    }

    // Every key is optional; missing keys keep the default value.
    p.gpuDeviceIdx              = optional_or<int>(y, "gpuDeviceIdx",              p.gpuDeviceIdx);
    p.numSimChnRlz              = optional_or<int>(y, "numSimChnRlz",              p.numSimChnRlz);
    p.seedConst                 = optional_or<int>(y, "seedConst",                 p.seedConst);
    p.slotDurationConst         = optional_or<double>(y, "slotDurationConst",      p.slotDurationConst);
    p.scsConst                  = optional_or<double>(y, "scsConst",               p.scsConst);
    p.numMcsLevels              = optional_or<int>(y, "numMcsLevels",              p.numMcsLevels);
    p.cellRadiusConst           = optional_or<int>(y, "cellRadiusConst",           p.cellRadiusConst);
    p.numCellConst              = optional_or<int>(y, "numCellConst",              p.numCellConst);
    p.numUePerCellConst         = optional_or<int>(y, "numUePerCellConst",         p.numUePerCellConst);
    p.numUeForGrpConst          = optional_or<int>(y, "numUeForGrpConst",          p.numUeForGrpConst);
    p.numActiveUePerCellConst   = optional_or<int>(y, "numActiveUePerCellConst",   p.numActiveUePerCellConst);

    p.nBsAntConst               = optional_or<int>(y, "nBsAntConst",               p.nBsAntConst);
    p.bsAntSizeConst            = optional_or<std::vector<uint16_t>>(y, "bsAntSizeConst",        p.bsAntSizeConst);
    p.bsAntSpacingConst         = optional_or<std::vector<float>>(y,    "bsAntSpacingConst",     p.bsAntSpacingConst);
    p.bsAntPolarAnglesConst     = optional_or<std::vector<float>>(y,    "bsAntPolarAnglesConst", p.bsAntPolarAnglesConst);
    p.bsAntPatternConst         = optional_or<int>(y, "bsAntPatternConst",         p.bsAntPatternConst);
    p.nUeAntConst               = optional_or<int>(y, "nUeAntConst",               p.nUeAntConst);
    p.ueAntSizeConst            = optional_or<std::vector<uint16_t>>(y, "ueAntSizeConst",        p.ueAntSizeConst);
    p.ueAntSpacingConst         = optional_or<std::vector<float>>(y,    "ueAntSpacingConst",     p.ueAntSpacingConst);
    p.ueAntPolarAnglesConst     = optional_or<std::vector<float>>(y,    "ueAntPolarAnglesConst", p.ueAntPolarAnglesConst);
    p.ueAntPatternConst         = optional_or<int>(y, "ueAntPatternConst",         p.ueAntPatternConst);
    p.vDirectionConst           = optional_or<std::vector<float>>(y,    "vDirectionConst",       p.vDirectionConst);

    p.nPrbsPerGrpConst          = optional_or<int>(y, "nPrbsPerGrpConst",          p.nPrbsPerGrpConst);
    p.nPrbGrpsConst             = optional_or<int>(y, "nPrbGrpsConst",             p.nPrbGrpsConst);
    p.PtConst                   = optional_or<double>(y, "PtConst",                p.PtConst);
    p.noiseFigureConst          = optional_or<double>(y, "noiseFigureConst",       p.noiseFigureConst);

    p.gpuAllocTypeConst         = optional_or<int>(y, "gpuAllocTypeConst",         p.gpuAllocTypeConst);
    p.cpuAllocTypeConst         = optional_or<int>(y, "cpuAllocTypeConst",         p.cpuAllocTypeConst);
    p.prdSchemeConst            = optional_or<int>(y, "prdSchemeConst",            p.prdSchemeConst);
    p.rxSchemeConst             = optional_or<int>(y, "rxSchemeConst",             p.rxSchemeConst);
    p.heteroUeSelCellsConst     = optional_or<int>(y, "heteroUeSelCellsConst",     p.heteroUeSelCellsConst);

    p.maxNumCoorCellConst       = optional_or<int>(y, "maxNumCoorCellConst",       p.maxNumCoorCellConst);
    p.maxNumBsAntConst          = optional_or<int>(y, "maxNumBsAntConst",          p.maxNumBsAntConst);
    p.maxNumUeAntConst          = optional_or<int>(y, "maxNumUeAntConst",          p.maxNumUeAntConst);
    p.maxNumPrbGrpsConst        = optional_or<int>(y, "maxNumPrbGrpsConst",        p.maxNumPrbGrpsConst);

    p.pdschNrOfSymbols          = optional_or<int>(y, "pdschNrOfSymbols",          p.pdschNrOfSymbols);
    p.pdschNrOfDmrsSymb         = optional_or<int>(y, "pdschNrOfDmrsSymb",         p.pdschNrOfDmrsSymb);
    p.pdschNrOfLayers           = optional_or<int>(y, "pdschNrOfLayers",           p.pdschNrOfLayers);

    p.initAvgRateConst          = optional_or<double>(y, "initAvgRateConst",       p.initAvgRateConst);
    p.pfAvgRateUpdConst         = optional_or<double>(y, "pfAvgRateUpdConst",      p.pfAvgRateUpdConst);
    p.betaCoeffConst            = optional_or<double>(y, "betaCoeffConst",         p.betaCoeffConst);
    p.sinValThrConst            = optional_or<double>(y, "sinValThrConst",         p.sinValThrConst);
    p.prioWeightStepConst       = optional_or<int>(y, "prioWeightStepConst",       p.prioWeightStepConst);

    p.AFTER_SCALING_SIGMA_CONST = optional_or<double>(y, "AFTER_SCALING_SIGMA_CONST", p.AFTER_SCALING_SIGMA_CONST);

    p.cpuGpuPerfGapPerUeConst   = optional_or<double>(y, "cpuGpuPerfGapPerUeConst", p.cpuGpuPerfGapPerUeConst);
    p.cpuGpuPerfGapSumRConst    = optional_or<double>(y, "cpuGpuPerfGapSumRConst",  p.cpuGpuPerfGapSumRConst);

    p.toleranceConst            = optional_or<double>(y, "toleranceConst",         p.toleranceConst);

    p.svdToleranceConst         = optional_or<double>(y, "svdToleranceConst",      p.svdToleranceConst);
    p.svdMaxSweeps              = optional_or<int>(y, "svdMaxSweeps",              p.svdMaxSweeps);

    p.amplifyCoeConst           = optional_or<int>(y, "amplifyCoeConst",           p.amplifyCoeConst);

    p.mcOutputFile              = optional_or<std::string>(y, "mcOutputFile",      p.mcOutputFile);
    p.mcOutputFileShort         = optional_or<std::string>(y, "mcOutputFileShort", p.mcOutputFileShort);

    p.targetChanCoeRangeScale   = optional_or<double>(y, "targetChanCoeRangeScale", p.targetChanCoeRangeScale);
    p.MinNoiseRangeConst        = optional_or<float>(y, "MinNoiseRangeConst",      p.MinNoiseRangeConst);

    p.nMaxUeSchdPerCellTTIConst = optional_or<int>(y, "nMaxUeSchdPerCellTTIConst", p.nMaxUeSchdPerCellTTIConst);

    if (!validateForComputeDerived(p)) {
        Params defaults{};
        applyHardcodedDefaults(defaults);
        p = std::move(defaults);
    }

    validateVectorLengths(p);

    computeDerived(p);
    g_params = std::move(p);

    std::cout << "[parameters] loaded: numCellConst=" << g_params.numCellConst
              << " numUePerCellConst=" << g_params.numUePerCellConst
              << " numSimChnRlz=" << g_params.numSimChnRlz
              << " nBsAntConst=" << g_params.nBsAntConst
              << " nUeAntConst=" << g_params.nUeAntConst
              << std::endl;
}

} // namespace cumac

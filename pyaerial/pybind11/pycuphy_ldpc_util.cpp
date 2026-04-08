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

#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <vector>
#include <stdexcept>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "cuphy.h"
#include "utils.cuh"
#include "pusch_utils.hpp"
#include "pycuphy_ldpc.hpp"

namespace py = pybind11;

namespace pycuphy {

void setPdschPerTbParams(PdschPerTbParams& tbParams,
                         uint8_t* tbStartAddr,
                         uint32_t tbStartOffset,
                         uint32_t tbSize,
                         uint32_t cumulativeTbSizePadding,
                         float codeRate,
                         uint32_t rateMatchLen,
                         uint8_t qamMod,
                         uint32_t numCodedBits,
                         uint8_t rv,
                         uint8_t numLayers,
                         uint32_t cinit) {

    tbParams.tbStartAddr = tbStartAddr;
    tbParams.tbStartOffset = tbStartOffset;
    tbParams.tbSize = div_round_up<uint32_t>(tbSize, CHAR_BIT);
    tbParams.cumulativeTbSizePadding = cumulativeTbSizePadding;
    tbParams.testModel =  0;  // No test model in pycuphy/pyAerial.

    tbParams.bg = get_base_graph(codeRate, tbSize);
    tbParams.num_CBs = 0;  // Get this as an output in the following.
    uint16_t Kprime = get_K_prime(tbSize, tbParams.bg, tbParams.num_CBs);
    tbParams.Zc = get_lifting_size(tbSize, tbParams.bg, Kprime);
    tbParams.K = (tbParams.bg == 1) ? CUPHY_LDPC_BG1_INFO_NODES * tbParams.Zc : CUPHY_LDPC_MAX_BG2_INFO_NODES * tbParams.Zc;
    tbParams.F = tbParams.K - Kprime;
    tbParams.N = (tbParams.bg == 1) ? CUPHY_LDPC_MAX_BG1_UNPUNCTURED_VAR_NODES * tbParams.Zc : CUPHY_LDPC_MAX_BG2_UNPUNCTURED_VAR_NODES * tbParams.Zc;
    if(numCodedBits != 0 && tbParams.N != numCodedBits) {  // Just a check as these should match.
        throw std::runtime_error("Invalid number of coded bits!");
    }

    tbParams.Ncb = tbParams.N;
    tbParams.G = rateMatchLen;
    tbParams.Qm = qamMod;
    tbParams.rv = rv;
    tbParams.Nl = numLayers;
    tbParams.max_REs = rateMatchLen / (qamMod * numLayers);
    tbParams.cinit = cinit;
    tbParams.firstCodeBlockIndex = 0;  // Always zero?
}


void setPerTbParams(PerTbParams& tbParams,
                    cuphyLDPCParams& ldpcParams,
                    const uint32_t tbSize,
                    const float codeRate,
                    const uint8_t qamMod,
                    const uint32_t ndi,
                    const uint32_t rv,
                    const uint32_t rateMatchLen,
                    const uint32_t cinit,
                    const uint32_t userGroupIdx,
                    const uint8_t numLayers,
                    const uint8_t numUeGrpLayers,
                    const std::vector<uint32_t>& layerMapArray,
                    const uint8_t nDmrsCdmGrpsNoData) {
    tbParams.tbSize = tbSize;  // In bits.
    tbParams.codeRate = codeRate;
    tbParams.Qm = qamMod;
    tbParams.ndi = ndi;
    tbParams.rv = rv;
    tbParams.bg = get_base_graph(codeRate, tbSize);
    tbParams.Nl = numLayers;
    tbParams.num_CBs = 0;
    uint32_t Kprime = get_K_prime(tbSize, tbParams.bg, tbParams.num_CBs);
    tbParams.Zc = get_lifting_size(tbSize, tbParams.bg, Kprime);
    tbParams.N = (tbParams.bg == 1) ? CUPHY_LDPC_MAX_BG1_UNPUNCTURED_VAR_NODES * tbParams.Zc : CUPHY_LDPC_MAX_BG2_UNPUNCTURED_VAR_NODES * tbParams.Zc;
    tbParams.Ncb = tbParams.N;          // Same as N for now
    tbParams.Ncb_padded = (tbParams.N + 2 * tbParams.Zc + 7) / 8;
    tbParams.Ncb_padded *= 8;
    tbParams.G = rateMatchLen;
    tbParams.K = (tbParams.bg == 1) ? CUPHY_LDPC_BG1_INFO_NODES * tbParams.Zc : CUPHY_LDPC_MAX_BG2_INFO_NODES * tbParams.Zc;
    tbParams.F = tbParams.K - Kprime;
    tbParams.cinit = cinit;
    tbParams.nDataBytes = tbSize / 8;
    tbParams.firstCodeBlockIndex = 0;
    tbParams.encodedSize = tbParams.G;
    for(int i = 0; i < numLayers; i++)
        tbParams.layer_map_array[i] = layerMapArray[i];
    tbParams.userGroupIndex = userGroupIdx;
    tbParams.nBBULayers = numUeGrpLayers;
    tbParams.startLLR = 0;

    uint32_t Kd = tbParams.K - tbParams.F - 2 * tbParams.Zc;

    uint32_t numParityNodes{};
    uint32_t Zc = tbParams.Zc;
    uint32_t Ncb = tbParams.Ncb;
    uint32_t k0{};
    if(tbParams.bg == 1) {

        if(rv == 0) {
            k0 = 0;
        }
        else if(rv == 1) {
            k0 = (17 * Ncb / (66 * Zc)) * Zc;
        }
        else if(rv == 2) {
            k0 = (33 * Ncb / (66 * Zc)) * Zc;
        }
        else if(rv == 3) {
            k0 = (56 * Ncb / (66 * Zc)) * Zc;
        }
        uint32_t NcbForParity = std::min<uint32_t>((tbParams.encodedSize) / tbParams.num_CBs + k0, Ncb);
        numParityNodes = (NcbForParity - Kd + Zc - 1) / Zc;
        numParityNodes = std::max<uint32_t>(4, std::min<uint32_t>(CUPHY_LDPC_MAX_BG1_PARITY_NODES, numParityNodes));
    }
    else {
        if(rv == 0) {
            k0 = 0;
        }
        else if(rv == 1) {
            k0 = (13 * Ncb / (50 * Zc)) * Zc;
        }
        else if(rv == 2) {
            k0 = (25 * Ncb / (50 * Zc)) * Zc;
        }
        else if(rv == 3) {
            k0 = (43 * Ncb / (50 * Zc)) * Zc;
        }
        uint32_t NcbForParity = std::min<uint32_t>((tbParams.encodedSize) / tbParams.num_CBs + k0, Ncb);
        numParityNodes = (NcbForParity - Kd + Zc - 1) / Zc;
        numParityNodes = std::max<uint32_t>(4, std::min<uint32_t>(CUPHY_LDPC_MAX_BG2_PARITY_NODES, numParityNodes));
    }

    ldpcParams.parityNodesArray.push_back(numParityNodes);
    uint32_t Kb = get_Kb(tbSize, tbParams.bg);
    ldpcParams.KbArray.push_back(Kb);

    tbParams.k0 = k0;
    tbParams.nZpBitsPerCb = (numParityNodes * Zc) + tbParams.K;
    tbParams.mScUciSum = 0;
    tbParams.isDataPresent = 1;
    tbParams.uciOnPuschFlag = 0;
    tbParams.csi2Flag = 0;
    tbParams.debug_d_derateCbsIndices = nullptr;
    tbParams.enableTfPrcd = 0;
    tbParams.nDmrsCdmGrpsNoData = nDmrsCdmGrpsNoData;
}


void readDmrsParams(const std::vector<py::object>& pdschConfigs,
                    const uint32_t slot,
                    const uint16_t cellId,
                    const uint16_t nPrbDlBwp,
                    __half2* cellOutputTensorAddr,
                    uint32_t& numTbs,
                    cuphy::buffer<PdschDmrsParams, cuphy::pinned_alloc>& dmrsPrms) {

    numTbs = 0;
    for(int ueGrpIdx = 0; const auto& pdschConfig : pdschConfigs) {

        // Read PDSCH UE group parameters.
        const auto numDmrsCdmGrps = pdschConfig.attr("num_dmrs_cdm_grps_no_data").cast<uint8_t>();
        if(numDmrsCdmGrps == 3) {
            NVLOGE_FMT(NVLOG_PYAERIAL, AERIAL_PYAERIAL_EVENT, "3 DM-RS CDM groups without data not supported for type-I DMRS!");
            throw std::runtime_error("PyPdschDmrsTx::readDmrsParams: Invalid parameters");
        }
        const auto resourceAlloc = pdschConfig.attr("resource_alloc").cast<uint8_t>();
        const py::array& rbBitmap = pdschConfig.attr("prb_bitmap");
        const auto startRb = pdschConfig.attr("start_prb").cast<uint16_t>();
        const auto numRbs = pdschConfig.attr("num_prbs").cast<uint16_t>();
        const auto dmrsSyms = pdschConfig.attr("dmrs_syms");
        const auto startSym = pdschConfig.attr("start_sym").cast<uint8_t>();
        const auto numSyms = pdschConfig.attr("num_symbols").cast<uint8_t>();

        uint16_t dmrsSymLoc{};
        uint64_t dataSymLoc{};
        uint8_t numDmrsSyms{}, numDataSyms{};

        expandDmrsBitmap(dmrsSyms, startSym, numSyms, dmrsSymLoc, dataSymLoc, numDmrsSyms, numDataSyms);

        const py::list& pdschUeConfigs = pdschConfig.attr("ue_configs");
        const uint32_t numUes = pdschUeConfigs.size();
        for(int ueIdx = 0; ueIdx < numUes; ++ueIdx) {
            const auto& pdschUeConfig = pdschUeConfigs[ueIdx];

            // Read PDSCH UE parameters.
            const auto scid = pdschUeConfig.attr("scid").cast<uint32_t>();
            const auto dmrsScramblingId = pdschUeConfig.attr("dmrs_scrm_id").cast<uint32_t>();
            const auto layers = pdschUeConfig.attr("layers").cast<uint8_t>();
            const auto bwpStart = pdschUeConfig.attr("bwp_start").cast<uint16_t>();
            const auto refPoint = pdschUeConfig.attr("ref_point").cast<uint8_t>();
            const auto betaQam = pdschUeConfig.attr("beta_qam").cast<float>();
            const auto betaDmrs = pdschUeConfig.attr("beta_dmrs").cast<float>();

            // Check sufficient allocation.
            if(numTbs >= dmrsPrms.size()) {
                NVLOGE_FMT(NVLOG_PYAERIAL, AERIAL_PYAERIAL_EVENT, "Insufficient allocation for DM-RS parameters!");
                throw std::runtime_error("readDmrsParams: Insufficient allocation for DM-RS parameters");
            }

            PdschDmrsParams& dmrsPrm = dmrsPrms[numTbs];
            dmrsPrm.cell_output_tensor_addr = cellOutputTensorAddr;
            dmrsPrm.data_sym_loc = dataSymLoc;
            dmrsPrm.slot_number = slot;
            dmrsPrm.cell_id = cellId;
            dmrsPrm.num_dmrs_symbols = numDmrsSyms;
            dmrsPrm.num_data_symbols = numDataSyms;
            dmrsPrm.symbol_number = startSym;
            dmrsPrm.resourceAlloc = resourceAlloc;
            readRbBitmap(rbBitmap, dmrsPrm.rbBitmap);
            dmrsPrm.num_BWP_PRBs = nPrbDlBwp;
            dmrsPrm.num_Rbs = numRbs;
            dmrsPrm.start_Rb = startRb;
            dmrsPrm.beta_dmrs = betaDmrs;
            dmrsPrm.beta_qam = betaQam;
            dmrsPrm.num_layers = layers;
            dmrsPrm.nlAbove16 = 0;  // Initialize for 16-layer mode (32-layer not supported in pyaerial yet)

            auto dmrsPortsBmsk = pdschUeConfig.attr("dmrs_ports").cast<uint16_t>();
            for (int i = 0; i < layers; i++) {
                dmrsPrm.port_ids[i] = __builtin_ctz(dmrsPortsBmsk);
                dmrsPortsBmsk ^= (1 << dmrsPrm.port_ids[i]);
            }

            const py::object& precMatrix = pdschUeConfig.attr("precoding_matrix");
            if (not (py::str(precMatrix).cast<std::string>() == "None")) {
                dmrsPrm.enablePrcdBf = 1;
                const py::array_t<std::complex<float>>& pmwArray = precMatrix;
                readPrecodingMatrix(pmwArray, dmrsPrm.pmW, dmrsPrm.Np);
            }
            else {
                dmrsPrm.enablePrcdBf = 0;
                dmrsPrm.Np = 0;
            }

            dmrsPrm.n_scid = scid;
            dmrsPrm.dmrs_scid = dmrsScramblingId;
            dmrsPrm.dmrs_sym_loc = dmrsSymLoc;
            dmrsPrm.BWP_start_PRB = bwpStart;
            dmrsPrm.cell_index_in_cell_group = 0;
            dmrsPrm.ref_point = refPoint;
            dmrsPrm.ueGrp_idx = ueGrpIdx;
            dmrsPrm.dmrsCdmGrpsNoData1 = (numDmrsCdmGrps == 1);

            ++numTbs;
        }

        ++ueGrpIdx;
    }
}


void readTbParams(const std::vector<py::object>& pdschConfigs,
                  uint8_t* tbInputAddr,
                  uint32_t& numTbs,
                  cuphy::buffer<PdschPerTbParams, cuphy::pinned_alloc>& tbPrms) {

    uint32_t tbStartOffset = 0;
    uint32_t cumulativeTbSizePadding = 0;
    numTbs = 0;

    for(const auto& pdschConfig : pdschConfigs) {

        // Read PDSCH UE group parameters.
        const auto numDmrsCdmGrps = pdschConfig.attr("num_dmrs_cdm_grps_no_data").cast<uint8_t>();
        if(numDmrsCdmGrps == 3) {
            NVLOGE_FMT(NVLOG_PYAERIAL, AERIAL_PYAERIAL_EVENT, "3 DM-RS CDM groups without data not supported for type-I DMRS!");
            throw std::runtime_error("PyPdschDmrsTx::readDmrsParams: Invalid parameters");
        }

        const auto startRb = pdschConfig.attr("start_prb").cast<uint16_t>();
        const auto numRbs = pdschConfig.attr("num_prbs").cast<uint16_t>();
        const auto dmrsSyms = pdschConfig.attr("dmrs_syms");
        const auto startSym = pdschConfig.attr("start_sym").cast<uint8_t>();
        const auto numSyms = pdschConfig.attr("num_symbols").cast<uint8_t>();

        uint16_t dmrsSymLoc{};
        uint64_t dataSymLoc{};
        uint8_t numDmrsSyms{}, numDataSyms{};
        expandDmrsBitmap(dmrsSyms, startSym, numSyms, dmrsSymLoc, dataSymLoc, numDmrsSyms, numDataSyms);

        const auto numDmrsCdmGrpsNoData1Sym = (numDmrsCdmGrps == 1) ? numDmrsSyms : 0;

        const auto maxREs = (numDataSyms * CUPHY_N_TONES_PER_PRB  + numDmrsCdmGrpsNoData1Sym * (CUPHY_N_TONES_PER_PRB / 2)) * numRbs;

        const py::list& pdschUeConfigs = pdschConfig.attr("ue_configs");
        const uint32_t numUes = pdschUeConfigs.size();
        for(int ueIdx = 0; ueIdx < numUes; ++ueIdx) {
            const auto& pdschUeConfig = pdschUeConfigs[ueIdx];

            const auto numLayers = pdschUeConfig.attr("layers").cast<uint8_t>();
            const auto rnti = pdschUeConfig.attr("rnti").cast<uint32_t>();
            const auto dataScramblingId = pdschUeConfig.attr("data_scid").cast<uint32_t>();
            const auto cinit = (rnti << 15) + dataScramblingId;

            const py::list& pdschCwConfigs = pdschUeConfig.attr("cw_configs");
            const uint32_t numCws = pdschCwConfigs.size();
            for(int cwIdx = 0; cwIdx < numCws; ++cwIdx) {
                const auto& pdschCwConfig = pdschCwConfigs[cwIdx];

                const auto rv = pdschCwConfig.attr("rv").cast<uint8_t>();
                const auto modOrder = pdschCwConfig.attr("mod_order").cast<uint8_t>();
                const auto codeRate = pdschCwConfig.attr("code_rate").cast<float>() / 1024.0f / 10;
                const auto rateMatchLen = maxREs * modOrder * numLayers;

                // Get number of code blocks and TB size.
                uint32_t numCodeBlocks, tbSize;
                get_TB_size_and_num_CBs(numDataSyms,
                                        numRbs,
                                        numLayers,
                                        codeRate,
                                        modOrder,
                                        numCodeBlocks,
                                        tbSize,
                                        numDmrsCdmGrpsNoData1Sym);

                const auto tbSizeBytes = div_round_up<uint32_t>(tbSize, CHAR_BIT);
                auto paddingBytes = div_round_up<uint32_t>(tbSize, sizeof(uint32_t)) * sizeof(uint32_t) - tbSizeBytes;
                paddingBytes += ((paddingBytes <= 2) ? sizeof(uint32_t) : 0);
                cumulativeTbSizePadding += (tbSizeBytes + paddingBytes);

                // Check sufficient allocation.
                if(numTbs >= tbPrms.size()) {
                    NVLOGE_FMT(NVLOG_PYAERIAL, AERIAL_PYAERIAL_EVENT, "Insufficient allocation for TB parameters!");
                    throw std::runtime_error("readTbParams: Insufficient allocation for TB parameters");
                }

                // Set TB parameters.
                PdschPerTbParams& tbPrm = tbPrms[numTbs];
                setPdschPerTbParams(tbPrm,
                                    tbInputAddr,
                                    tbStartOffset,
                                    tbSize,
                                    cumulativeTbSizePadding,
                                    codeRate,
                                    rateMatchLen,
                                    modOrder,
                                    0,  // numCodedBits, computed within the function.
                                    rv,
                                    numLayers,
                                    cinit);

                tbStartOffset += tbSizeBytes;
                ++numTbs;
            }
        }
    }
}


} // namespace pycuphy

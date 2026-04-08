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


#include <gtest/gtest.h>
#include <vector>
#include <cstdint>
#include <cstring>
#include <memory>

#include "cuphy.h"
#include "cuphy_api.h"
#include "cuphy.hpp"
#include <cuda_fp16.h>
#include "cuphy/pucch_F3_front_end/pucch_F3_segLLRs.hpp"

// Synthetic host-side reference implementation mirroring pucchF3SegLLRsKernel
namespace {

// Device constant tables reproduced for host reference
static const uint8_t uciSymInd_ref[17][12] = {{0, 2, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                                              {1, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                                              {1, 2, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                                              {0, 2, 3, 5, 0, 0, 0, 0, 0, 0, 0, 0},
                                              {0, 2, 3, 5, 6, 0, 0, 0, 0, 0, 0, 0},
                                              {0, 2, 3, 4, 6, 7, 0, 0, 0, 0, 0, 0},
                                              {0, 2, 3, 4, 5, 7, 8, 0, 0, 0, 0, 0},
                                              {0, 1, 3, 4, 5, 6, 8, 9, 0, 0, 0, 0},
                                              {0, 2, 4, 5, 7, 9, 0, 0, 0, 0, 0, 0},
                                              {0, 1, 3, 4, 5, 6, 8, 9, 10, 0, 0, 0},
                                              {0, 2, 4, 5, 7, 8, 10, 0, 0, 0, 0, 0},
                                              {0, 1, 3, 4, 5, 6, 7, 9, 10, 11, 0, 0},
                                              {0, 2, 3, 5, 6, 8, 9, 11, 0, 0, 0, 0},
                                              {0, 1, 3, 4, 5, 6, 7, 8, 10, 11, 12, 0},
                                              {0, 2, 3, 5, 6, 8, 9, 10, 12, 0, 0, 0},
                                              {0, 1, 2, 4, 5, 6, 7, 8, 9, 11, 12, 13},
                                              {0, 2, 3, 4, 6, 7, 9, 10, 11, 13, 0, 0}};

static const uint8_t SiUci_1_ref[17][8] = {{0, 2, 0, 0, 0, 0, 0, 0},
                                           {1, 3, 0, 0, 0, 0, 0, 0},
                                           {1, 2, 4, 0, 0, 0, 0, 0},
                                           {0, 2, 3, 5, 0, 0, 0, 0},
                                           {0, 2, 3, 5, 0, 0, 0, 0},
                                           {0, 2, 4, 6, 0, 0, 0, 0},
                                           {0, 2, 5, 7, 0, 0, 0, 0},
                                           {1, 3, 6, 8, 0, 0, 0, 0},
                                           {0, 2, 4, 5, 7, 9, 0, 0},
                                           {1, 3, 6, 8, 0, 0, 0, 0},
                                           {0, 2, 4, 5, 7, 8, 10, 0},
                                           {1, 3, 7, 9, 0, 0, 0, 0},
                                           {0, 2, 3, 5, 6, 8, 9, 11},
                                           {1, 3, 8, 10, 0, 0, 0, 0},
                                           {0, 2, 3, 5, 6, 8, 10, 12},
                                           {2, 4, 9, 11, 0, 0, 0, 0},
                                           {0, 2, 4, 6, 7, 9, 11, 13}};

static const uint8_t SiUci_2_ref[11][4] = {{3, 0, 0, 0},
                                           {6, 0, 0, 0},
                                           {3, 7, 0, 0},
                                           {3, 4, 8, 0},
                                           {0, 4, 5, 9},
                                           {0, 4, 5, 9},
                                           {0, 4, 6, 10},
                                           {0, 4, 7, 11},
                                           {9, 0, 0, 0},
                                           {1, 5, 8, 12},
                                           {3, 10, 0, 0}};

static const uint8_t SiUci_3_ref[4][4]  = {{10, 0, 0, 0},
                                           {5, 11, 0, 0},
                                           {5, 6, 12, 0},
                                           {0, 6, 7, 13}};

struct CaseParams {
    uint8_t  nSym;
    uint8_t  nSym_dmrs;
    uint8_t  Qm;           // 1 or 2
    uint16_t prbSize;      // 1..CUPHY_PUCCH_F3_MAX_PRBS
    uint32_t E_seg1_units; // in units of LLRs (not bytes)
};

struct DerivParams {
    int uciSymInd_row;
    int SiUci_1_row;
    int SiUci_2_row;
    int SiUci_3_row;
    int NiUci_1;
    int NiUci_2;
    int NiUci_3;
    int sumNiUci[3];
    uint8_t nSym_data;
};

static DerivParams derive_params(uint8_t nSym, uint8_t nSym_dmrs)
{
    DerivParams d{};
    d.uciSymInd_row = -1;
    d.SiUci_1_row = -1;
    d.SiUci_2_row = -1;
    d.SiUci_3_row = -1;
    d.NiUci_1 = d.NiUci_2 = d.NiUci_3 = 0;
    d.sumNiUci[0] = d.sumNiUci[1] = d.sumNiUci[2] = 0;

    switch(nSym) {
        case 4:
            if(nSym_dmrs == 1) { d.uciSymInd_row = 0; d.SiUci_1_row = 0; d.SiUci_2_row = 0; d.NiUci_1 = 2; d.NiUci_2 = 1; d.sumNiUci[0]=2; d.sumNiUci[1]=3; }
            else { d.uciSymInd_row = 1; d.SiUci_1_row = 1; d.NiUci_1 = 2; d.sumNiUci[0]=2; }
            break;
        case 5:
            d.uciSymInd_row = 2; d.SiUci_1_row = 2; d.NiUci_1 = 3; d.sumNiUci[0]=3; break;
        case 6:
            d.uciSymInd_row = 3; d.SiUci_1_row = 3; d.NiUci_1 = 4; d.sumNiUci[0]=4; break;
        case 7:
            d.uciSymInd_row = 4; d.SiUci_1_row = 4; d.SiUci_2_row = 1; d.NiUci_1 = 4; d.NiUci_2 = 1; d.sumNiUci[0]=4; d.sumNiUci[1]=5; break;
        case 8:
            d.uciSymInd_row = 5; d.SiUci_1_row = 5; d.SiUci_2_row = 2; d.NiUci_1 = 4; d.NiUci_2 = 2; d.sumNiUci[0]=4; d.sumNiUci[1]=6; break;
        case 9:
            d.uciSymInd_row = 6; d.SiUci_1_row = 6; d.SiUci_2_row = 3; d.NiUci_1 = 4; d.NiUci_2 = 3; d.sumNiUci[0]=4; d.sumNiUci[1]=7; break;
        case 10:
            if(nSym_dmrs == 2) { d.uciSymInd_row = 7; d.SiUci_1_row = 7; d.SiUci_2_row = 4; d.NiUci_1=4; d.NiUci_2=4; d.sumNiUci[0]=4; d.sumNiUci[1]=8; }
            else { d.uciSymInd_row = 8; d.SiUci_1_row = 8; d.NiUci_1=6; d.sumNiUci[0]=6; }
            break;
        case 11:
            if(nSym_dmrs == 2) { d.uciSymInd_row = 9; d.SiUci_1_row=9; d.SiUci_2_row=5; d.SiUci_3_row=0; d.NiUci_1=4; d.NiUci_2=4; d.NiUci_3=1; d.sumNiUci[0]=4; d.sumNiUci[1]=8; d.sumNiUci[2]=9; }
            else { d.uciSymInd_row = 10; d.SiUci_1_row=10; d.NiUci_1=7; d.sumNiUci[0]=7; }
            break;
        case 12:
            if(nSym_dmrs == 2) { d.uciSymInd_row = 11; d.SiUci_1_row=11; d.SiUci_2_row=6; d.SiUci_3_row=1; d.NiUci_1=4; d.NiUci_2=4; d.NiUci_3=2; d.sumNiUci[0]=4; d.sumNiUci[1]=8; d.sumNiUci[2]=10; }
            else { d.uciSymInd_row = 12; d.SiUci_1_row=12; d.NiUci_1=8; d.sumNiUci[0]=8; }
            break;
        case 13:
            if(nSym_dmrs == 2) { d.uciSymInd_row = 13; d.SiUci_1_row=13; d.SiUci_2_row=7; d.SiUci_3_row=2; d.NiUci_1=4; d.NiUci_2=4; d.NiUci_3=3; d.sumNiUci[0]=4; d.sumNiUci[1]=8; d.sumNiUci[2]=11; }
            else { d.uciSymInd_row = 14; d.SiUci_1_row=14; d.SiUci_2_row=8; d.NiUci_1=8; d.NiUci_2=1; d.sumNiUci[0]=8; d.sumNiUci[1]=9; }
            break;
        case 14:
            if(nSym_dmrs == 2) { d.uciSymInd_row = 15; d.SiUci_1_row=15; d.SiUci_2_row=9; d.SiUci_3_row=3; d.NiUci_1=4; d.NiUci_2=4; d.NiUci_3=4; d.sumNiUci[0]=4; d.sumNiUci[1]=8; d.sumNiUci[2]=12; }
            else { d.uciSymInd_row = 16; d.SiUci_1_row=16; d.SiUci_2_row=10; d.NiUci_1=8; d.NiUci_2=2; d.sumNiUci[0]=8; d.sumNiUci[1]=10; }
            break;
    }

    d.nSym_data = static_cast<uint8_t>(d.NiUci_1 + d.NiUci_2 + d.NiUci_3);
    return d;
}

static void cpu_segment_llrs(const CaseParams& cp,
                             uint16_t nSymUci,
                             const std::vector<__half>& in,
                             std::vector<__half>& out)
{
    const DerivParams d = derive_params(cp.nSym, cp.nSym_dmrs);

    const uint32_t E_seg1 = cp.E_seg1_units;
    const uint32_t E_tot  = static_cast<uint32_t>(cp.Qm) * nSymUci * d.nSym_data;
    const uint32_t E_seg2 = E_tot - E_seg1;

    // j selection with guard
    uint8_t j = 0;
    const int maxJ = (d.NiUci_3 ? 2 : (d.NiUci_2 ? 1 : 0));
    while(j < maxJ && d.sumNiUci[j] * nSymUci * cp.Qm < E_seg1) { ++j; }

    // build comSiUciLessj
    uint8_t comSiUciLessj[10] = {0};
    int     comSize = 0;
    if(j > 0) {
        for(int i = 0; i < d.NiUci_1; ++i) { comSiUciLessj[i] = SiUci_1_ref[d.SiUci_1_row][i]; }
        comSize = d.NiUci_1;
        if(j == 2) {
            for(int i = 0; i < d.NiUci_2; ++i) { comSiUciLessj[i + d.NiUci_1] = SiUci_2_ref[d.SiUci_2_row][i]; }
            comSize += d.NiUci_2;
        }
    }

    uint16_t nBarSymUci = 0;
    int      M = 0;
    if(j == 1) {
        const uint32_t temp = d.sumNiUci[0] * nSymUci * cp.Qm;
        nBarSymUci = static_cast<uint16_t>((E_seg1 - temp) / (d.NiUci_2 * cp.Qm));
        M = static_cast<int>(((E_seg1 - temp) / cp.Qm) % d.NiUci_2);
    } else if(j == 2) {
        const uint32_t temp = d.sumNiUci[1] * nSymUci * cp.Qm;
        nBarSymUci = static_cast<uint16_t>((E_seg1 - temp) / (d.NiUci_3 * cp.Qm));
        M = static_cast<int>(((E_seg1 - temp) / cp.Qm) % d.NiUci_3);
    } else {
        nBarSymUci = static_cast<uint16_t>(E_seg1 / (d.NiUci_1 * cp.Qm));
        M = static_cast<int>((E_seg1 / cp.Qm) % d.NiUci_1);
    }

    std::vector<__half> seq1(E_tot);
    std::vector<__half> seq2(E_tot);
    uint32_t n1 = 0, n2 = 0;

    for(uint8_t l = 0; l < d.nSym_data; ++l) {
        const uint8_t sl = uciSymInd_ref[d.uciSymInd_row][l];
        bool in_com = false;
        for(int c = 0; c < comSize; ++c) if(sl == comSiUciLessj[c]) { in_com = true; break; }
        if(in_com) {
            for(int sc = 0; sc < nSymUci; ++sc) {
                for(int v = 0; v < cp.Qm; ++v) seq1[n1 + sc * cp.Qm + v] = in[(l * nSymUci * cp.Qm) + sc * cp.Qm + v];
            }
            n1 += nSymUci * cp.Qm;
        } else {
            bool in_Si = false;
            if(j == 0) {
                for(int c = 0; c < d.NiUci_1; ++c) if(sl == SiUci_1_ref[d.SiUci_1_row][c]) { in_Si = true; break; }
            } else if(j == 1) {
                for(int c = 0; c < d.NiUci_2; ++c) if(sl == SiUci_2_ref[d.SiUci_2_row][c]) { in_Si = true; break; }
            } else {
                for(int c = 0; c < d.NiUci_3; ++c) if(sl == SiUci_3_ref[d.SiUci_3_row][c]) { in_Si = true; break; }
            }

            if(in_Si) {
                uint8_t gamma = (M > 0) ? 1 : 0;
                --M; // mirror kernel behavior
                for(int sc = 0; sc < nSymUci; ++sc) {
                    if(sc < nBarSymUci + gamma) {
                        for(int v = 0; v < cp.Qm; ++v) seq1[n1 + sc * cp.Qm + v] = in[(l * nSymUci * cp.Qm) + sc * cp.Qm + v];
                    } else {
                        for(int v = 0; v < cp.Qm; ++v) seq2[n2 + (sc - nBarSymUci - gamma) * cp.Qm + v] = in[(l * nSymUci * cp.Qm) + sc * cp.Qm + v];
                    }
                }
                n1 += (nBarSymUci + gamma) * cp.Qm;
                n2 += (nSymUci - nBarSymUci - gamma) * cp.Qm;
            } else {
                for(int sc = 0; sc < nSymUci; ++sc) {
                    for(int v = 0; v < cp.Qm; ++v) seq2[n2 + sc * cp.Qm + v] = in[(l * nSymUci * cp.Qm) + sc * cp.Qm + v];
                }
                n2 += nSymUci * cp.Qm;
            }
        }
    }

    out.resize(E_tot);
    // Emulate kernel's final copy rounds
    uint16_t round = static_cast<uint16_t>(E_seg1 / F3_SEG_LLR_THREAD_PER_UCI);
    if (E_seg1 - round * F3_SEG_LLR_THREAD_PER_UCI) { ++round; }
    for (int r = 0; r < round; ++r) {
        for (uint16_t subcarrIdx = 0; subcarrIdx < F3_SEG_LLR_THREAD_PER_UCI; ++subcarrIdx) {
            uint32_t index = static_cast<uint32_t>(r) * F3_SEG_LLR_THREAD_PER_UCI + subcarrIdx;
            if (index < E_seg1) {
                out[index] = seq1[index];
            }
        }
    }

    round = static_cast<uint16_t>(E_seg2 / F3_SEG_LLR_THREAD_PER_UCI);
    if (E_seg2 - round * F3_SEG_LLR_THREAD_PER_UCI) { ++round; }
    for (int r = 0; r < round; ++r) {
        for (uint16_t subcarrIdx = 0; subcarrIdx < F3_SEG_LLR_THREAD_PER_UCI; ++subcarrIdx) {
            uint32_t index = static_cast<uint32_t>(r) * F3_SEG_LLR_THREAD_PER_UCI + subcarrIdx;
            if (index < E_seg2) {
                out[E_seg1 + index] = seq2[index];
            }
        }
    }
}

static void fill_input_llrs(uint8_t Qm, uint16_t nSymUci, uint8_t nSym_data, std::vector<__half>& in)
{
    const uint32_t E_tot = static_cast<uint32_t>(Qm) * nSymUci * nSym_data;
    in.resize(E_tot);
    // Small integer patterns to remain exact in FP16
    uint32_t idx = 0;
    for(uint8_t l = 0; l < nSym_data; ++l) {
        for(uint16_t sc = 0; sc < nSymUci; ++sc) {
            for(uint8_t v = 0; v < Qm; ++v) {
                const uint16_t val = static_cast<uint16_t>((static_cast<uint16_t>(l) * 16u + sc) * 2u + v);
                in[idx++] = __float2half(static_cast<float>(val));
            }
        }
    }
}

// Compute segment SNR in dB between reference and measured LLRs
static double compute_snr_db(const std::vector<__half>& ref,
                             const std::vector<__half>& meas,
                             uint32_t start,
                             uint32_t len)
{
    double signalEnergy = 0.0;
    double errorEnergy  = 0.0;
    for(uint32_t i = 0; i < len; ++i) {
        const double r = static_cast<double>(__half2float(ref[start + i]));
        const double m = static_cast<double>(__half2float(meas[start + i]));
        signalEnergy += (r * r);
        const double e = (r - m);
        errorEnergy  += (e * e);
    }
    if(errorEnergy == 0.0) return 1e9; // perfect match
    return 10.0 * log10(signalEnergy / errorEnergy);
}

// Validate both LLR segments using an SNR threshold (default 30 dB)
static bool validate_llr_segments_snr(const std::vector<__half>& ref,
                                      const std::vector<__half>& meas,
                                      uint32_t E_seg1,
                                      uint32_t E_seg2,
                                      double threshold_db = 30.0)
{
    if(E_seg1 > 0) {
        const double snr1 = compute_snr_db(ref, meas, 0, E_seg1);
        if(snr1 < threshold_db) return false;
    }
    if(E_seg2 > 0) {
        const double snr2 = compute_snr_db(ref, meas, E_seg1, E_seg2);
        if(snr2 < threshold_db) return false;
    }
    return true;
}

class PucchF3SegLLRsTest : public ::testing::Test {
protected:
    cuphy::stream cuStrmMain;
    std::unique_ptr<cuphy::linear_alloc<128, cuphy::device_alloc>> pLinearAlloc;

    void SetUp() override {
        pLinearAlloc = std::make_unique<cuphy::linear_alloc<128, cuphy::device_alloc>>(16 * 1024 * 1024);
    }
};

enum class ValidationMode { SNR, Exact };

static bool run_one_case(const CaseParams& cp, ValidationMode mode = ValidationMode::SNR, bool enableCpuToGpuDescrAsyncCpy = false)
{
    // Derive params and construct inputs
    DerivParams d = derive_params(cp.nSym, cp.nSym_dmrs);
    const uint16_t nSymUci = static_cast<uint16_t>(12 * cp.prbSize);
    std::vector<__half> inHost;
    fill_input_llrs(cp.Qm, nSymUci, d.nSym_data, inHost);

    const uint32_t E_tot  = static_cast<uint32_t>(cp.Qm) * nSymUci * d.nSym_data;
    const uint32_t E_seg1 = std::min(cp.E_seg1_units, E_tot);
    const uint32_t E_seg2 = E_tot - E_seg1;

    // GPU buffers for one UCI
    __half* d_in = nullptr;
    cudaError_t err = cudaMalloc(&d_in, E_tot * sizeof(__half));
    if(err != cudaSuccess) return false;
    err = cudaMemcpy(d_in, inHost.data(), E_tot * sizeof(__half), cudaMemcpyHostToDevice);
    if(err != cudaSuccess) { cudaFree(d_in); return false; }

    // Descriptor buffers
    size_t dynDescrSizeBytes = 0, dynDescrAlignBytes = 0;
    if(cuphyPucchF3SegLLRsGetDescrInfo(&dynDescrSizeBytes, &dynDescrAlignBytes) != CUPHY_STATUS_SUCCESS) { cudaFree(d_in); return false; }
    cuphy::buffer<uint8_t, cuphy::pinned_alloc> dynDescrBufCpu(dynDescrSizeBytes);
    cuphy::buffer<uint8_t, cuphy::device_alloc> dynDescrBufGpu(dynDescrSizeBytes);

    // Create kernel object and launch cfg
    cuphyPucchF3SegLLRsHndl_t hndl;
    if(cuphyCreatePucchF3SegLLRs(&hndl) != CUPHY_STATUS_SUCCESS) { cudaFree(d_in); return false; }

    cuphyPucchF3SegLLRsLaunchCfg_t launchCfg{};

    // Prepare minimal F3 UCI params for setup
    uint16_t nF3Ucis = 1;
    cuphyPucchUciPrm_t f3UciPrms{};
    f3UciPrms.nSym     = cp.nSym;
    f3UciPrms.pi2Bpsk  = (cp.Qm == 1) ? 1 : 0;
    f3UciPrms.prbSize  = static_cast<uint8_t>(cp.prbSize);

    __half* pDescramLLRaddrs[1] = { d_in };

    // Use setup to populate descriptor skeleton and obtain kernel params
    if(cuphySetupPucchF3SegLLRs(hndl,
                                nF3Ucis,
                                &f3UciPrms,
                                pDescramLLRaddrs,
                                dynDescrBufCpu.addr(),
                                dynDescrBufGpu.addr(),
                                enableCpuToGpuDescrAsyncCpy ? 1 : 0,
                                &launchCfg,
                                0) != CUPHY_STATUS_SUCCESS) {
        cuphyDestroyPucchF3SegLLRs(hndl);
        cudaFree(d_in);
        return false;
    }

    // Override descriptor fields with test-specific values
    auto* pCpuDesc = reinterpret_cast<pucchF3SegLLRsDynDescr_t*>(dynDescrBufCpu.addr());
    pCpuDesc->numUcis = 1;
    pCpuDesc->pInLLRaddrs[0] = d_in;
    pCpuDesc->perUciPrmsArray[0].nSym       = cp.nSym;
    pCpuDesc->perUciPrmsArray[0].nSym_data  = d.nSym_data;
    pCpuDesc->perUciPrmsArray[0].nSym_dmrs  = cp.nSym_dmrs;
    pCpuDesc->perUciPrmsArray[0].Qm         = cp.Qm;
    pCpuDesc->perUciPrmsArray[0].nSymUci    = nSymUci;
    pCpuDesc->perUciPrmsArray[0].E_seg1     = static_cast<uint16_t>(E_seg1);
    pCpuDesc->perUciPrmsArray[0].E_seg2     = static_cast<uint16_t>(E_seg2);

    // Copy descriptor to GPU
    err = cudaMemcpy(dynDescrBufGpu.addr(), dynDescrBufCpu.addr(), dynDescrSizeBytes, cudaMemcpyHostToDevice);
    if(err != cudaSuccess) {
        cuphyDestroyPucchF3SegLLRs(hndl);
        cudaFree(d_in);
        return false;
    }

    // Launch kernel
    const CUDA_KERNEL_NODE_PARAMS& k = launchCfg.kernelNodeParamsDriver;
    CUresult kres = cuLaunchKernel(k.func,
                                   k.gridDimX, k.gridDimY, k.gridDimZ,
                                   k.blockDimX, k.blockDimY, k.blockDimZ,
                                   k.sharedMemBytes,
                                   0,
                                   k.kernelParams,
                                   k.extra);
    if(kres != CUDA_SUCCESS) {
        cuphyDestroyPucchF3SegLLRs(hndl);
        cudaFree(d_in);
        return false;
    }
    cudaDeviceSynchronize();

    // Copy back and compare with CPU reference
    std::vector<__half> outGpu(E_tot);
    err = cudaMemcpy(outGpu.data(), d_in, E_tot * sizeof(__half), cudaMemcpyDeviceToHost);
    if(err != cudaSuccess) {
        cuphyDestroyPucchF3SegLLRs(hndl);
        cudaFree(d_in);
        return false;
    }

    std::vector<__half> outCpu;
    cpu_segment_llrs(cp, nSymUci, inHost, outCpu);

    bool ok = true;
    if(mode == ValidationMode::Exact) {
        for(uint32_t i = 0; i < E_tot; ++i) {
            const float a = __half2float(outGpu[i]);
            const float b = __half2float(outCpu[i]);
            if(a != b) { ok = false; break; }
        }
    } else {
        ok = validate_llr_segments_snr(outCpu, outGpu, E_seg1, E_seg2, 30.0);
    }

    cuphyDestroyPucchF3SegLLRs(hndl);
    cudaFree(d_in);
    return ok;
}

// Run with custom launch UCI count to hit early-exit path (uciIdx >= numUcis)
static bool run_case_with_launch_ucis(const CaseParams& cp,
                                      uint16_t launchUcis,
                                      uint16_t effectiveNumUcis,
                                      ValidationMode mode = ValidationMode::SNR,
                                      bool enableCpuToGpuDescrAsyncCpy = false)
{
    // Derive params and construct inputs for first UCI only
    DerivParams d = derive_params(cp.nSym, cp.nSym_dmrs);
    const uint16_t nSymUci = static_cast<uint16_t>(12 * cp.prbSize);
    std::vector<__half> inHost;
    fill_input_llrs(cp.Qm, nSymUci, d.nSym_data, inHost);

    const uint32_t E_tot  = static_cast<uint32_t>(cp.Qm) * nSymUci * d.nSym_data;
    const uint32_t E_seg1 = std::min(cp.E_seg1_units, E_tot);
    const uint32_t E_seg2 = E_tot - E_seg1;

    __half* d_in = nullptr;
    if(cudaMalloc(&d_in, E_tot * sizeof(__half)) != cudaSuccess) return false;
    if(cudaMemcpy(d_in, inHost.data(), E_tot * sizeof(__half), cudaMemcpyHostToDevice) != cudaSuccess) { cudaFree(d_in); return false; }

    size_t dynDescrSizeBytes = 0, dynDescrAlignBytes = 0;
    if(cuphyPucchF3SegLLRsGetDescrInfo(&dynDescrSizeBytes, &dynDescrAlignBytes) != CUPHY_STATUS_SUCCESS) { cudaFree(d_in); return false; }
    cuphy::buffer<uint8_t, cuphy::pinned_alloc> dynDescrBufCpu(dynDescrSizeBytes);
    cuphy::buffer<uint8_t, cuphy::device_alloc> dynDescrBufGpu(dynDescrSizeBytes);

    cuphyPucchF3SegLLRsHndl_t hndl;
    if(cuphyCreatePucchF3SegLLRs(&hndl) != CUPHY_STATUS_SUCCESS) { cudaFree(d_in); return false; }

    cuphyPucchF3SegLLRsLaunchCfg_t launchCfg{};

    cuphyPucchUciPrm_t f3UciPrms{};
    f3UciPrms.nSym     = cp.nSym;
    f3UciPrms.pi2Bpsk  = (cp.Qm == 1) ? 1 : 0;
    f3UciPrms.prbSize  = static_cast<uint8_t>(cp.prbSize);

    __half* pDescramLLRaddrs[1] = { d_in };

    if(cuphySetupPucchF3SegLLRs(hndl,
                                launchUcis,
                                &f3UciPrms,
                                pDescramLLRaddrs,
                                dynDescrBufCpu.addr(),
                                dynDescrBufGpu.addr(),
                                enableCpuToGpuDescrAsyncCpy ? 1 : 0,
                                &launchCfg,
                                0) != CUPHY_STATUS_SUCCESS) {
        cuphyDestroyPucchF3SegLLRs(hndl);
        cudaFree(d_in);
        return false;
    }

    auto* pCpuDesc = reinterpret_cast<pucchF3SegLLRsDynDescr_t*>(dynDescrBufCpu.addr());
    pCpuDesc->numUcis = effectiveNumUcis;
    pCpuDesc->pInLLRaddrs[0] = d_in;
    pCpuDesc->perUciPrmsArray[0].nSym       = cp.nSym;
    pCpuDesc->perUciPrmsArray[0].nSym_data  = d.nSym_data;
    pCpuDesc->perUciPrmsArray[0].nSym_dmrs  = cp.nSym_dmrs;
    pCpuDesc->perUciPrmsArray[0].Qm         = cp.Qm;
    pCpuDesc->perUciPrmsArray[0].nSymUci    = nSymUci;
    pCpuDesc->perUciPrmsArray[0].E_seg1     = static_cast<uint16_t>(E_seg1);
    pCpuDesc->perUciPrmsArray[0].E_seg2     = static_cast<uint16_t>(E_seg2);

    if(cudaMemcpy(dynDescrBufGpu.addr(), dynDescrBufCpu.addr(), dynDescrSizeBytes, cudaMemcpyHostToDevice) != cudaSuccess) {
        cuphyDestroyPucchF3SegLLRs(hndl);
        cudaFree(d_in);
        return false;
    }

    const CUDA_KERNEL_NODE_PARAMS& k = launchCfg.kernelNodeParamsDriver;
    CUresult kres = cuLaunchKernel(k.func,
                                   k.gridDimX, k.gridDimY, k.gridDimZ,
                                   k.blockDimX, k.blockDimY, k.blockDimZ,
                                   k.sharedMemBytes,
                                   0,
                                   k.kernelParams,
                                   k.extra);
    if(kres != CUDA_SUCCESS) {
        cuphyDestroyPucchF3SegLLRs(hndl);
        cudaFree(d_in);
        return false;
    }
    cudaDeviceSynchronize();

    std::vector<__half> outGpu(E_tot);
    if(cudaMemcpy(outGpu.data(), d_in, E_tot * sizeof(__half), cudaMemcpyDeviceToHost) != cudaSuccess) {
        cuphyDestroyPucchF3SegLLRs(hndl);
        cudaFree(d_in);
        return false;
    }

    std::vector<__half> outCpu;
    cpu_segment_llrs(cp, nSymUci, inHost, outCpu);

    bool ok = (mode == ValidationMode::Exact)
        ? [&](){ for(uint32_t i=0;i<E_tot;++i){ if(__half2float(outGpu[i])!=__half2float(outCpu[i])) return false;} return true; }()
        : validate_llr_segments_snr(outCpu, outGpu, E_seg1, E_seg2, 30.0);

    cuphyDestroyPucchF3SegLLRs(hndl);
    cudaFree(d_in);
    return ok;
}

// Helper: pick a mid-band E_seg1 for target j, respecting symbol granularity
static uint32_t pick_E_seg1_mid_band(const DerivParams& d,
                                     uint16_t nSymUci,
                                     uint8_t Qm,
                                     int targetJ)
{
    int bandStart = (targetJ == 0) ? 0 : d.sumNiUci[targetJ - 1];
    int bandEnd   = d.sumNiUci[targetJ];
    int span      = bandEnd - bandStart;
    int midGroups = bandStart + std::max(1, span / 2);
    return static_cast<uint32_t>(midGroups) * nSymUci * Qm;
}

// Helper: build and run a case by target j selection
static bool run_case_for_j(uint8_t nSym,
                           uint8_t nSym_dmrs,
                           uint8_t Qm,
                           uint16_t prbSize,
                           int targetJ,
                           ValidationMode mode)
{
    DerivParams d = derive_params(nSym, nSym_dmrs);
    const uint16_t nSymUci = static_cast<uint16_t>(12 * prbSize);
    // Guard invalid targetJ
    if((targetJ == 2 && d.NiUci_3 == 0) || (targetJ == 1 && d.NiUci_2 == 0)) {
        return false;
    }
    uint32_t E_seg1 = pick_E_seg1_mid_band(d, nSymUci, Qm, targetJ);
    CaseParams cp{nSym, nSym_dmrs, Qm, prbSize, E_seg1};
    return run_one_case(cp, mode);
}

} // anonymous namespace

//-----------------------------------------------------------------------------
// Test cases covering different branches (j = 0,1,2), Qm = 1/2, DMRS variants
//-----------------------------------------------------------------------------

TEST(PucchF3SegLLRsTest, JEqualsZero_QPSK_nSym7)
{
    // nSym=7 -> NiUci_1=4, NiUci_2=1, total data sym=5. Choose E_seg1 < 4*nSymUci*Qm
    CaseParams cp{7, 2, 2, 1, static_cast<uint32_t>(3 * 12 * 2 + 2)}; // prb=1; E_seg1 within group 1
    ASSERT_TRUE(run_one_case(cp));
}

TEST(PucchF3SegLLRsTest, JEqualsOne_BPSK_nSym7)
{
    // E_seg1 between 4 and 5 groups to force j=1
    CaseParams cp{7, 2, 1, 1, static_cast<uint32_t>(4 * 12 * 1 + 3)};
    ASSERT_TRUE(run_one_case(cp));
}

TEST(PucchF3SegLLRsTest, JEqualsTwo_QPSK_nSym12_dmrs2)
{
    // nSym=12, dmrs=2 -> sumNiUci = [4,8,10]; pick between 8 and 10 groups to force j=2
    CaseParams cp{12, 2, 2, 1, static_cast<uint32_t>(9 * 12 * 2 + 4)};
    ASSERT_TRUE(run_one_case(cp));
}

TEST(PucchF3SegLLRsTest, DMRS4_QPSK_nSym14_jEqualsOne)
{
    // nSym=14, dmrs=4 -> sumNiUci=[8,10]; pick E_seg1 > 8 groups but <=10 to force j=1
    // Use an E_seg1 near the upper bound to avoid borderline per-subcarrier effects
    CaseParams cp{14, 4, 2, 1, static_cast<uint32_t>(10 * 12 * 2 - 1)};
    // Use exact validation for this tricky case to avoid SNR sensitivity
    ASSERT_TRUE(run_one_case(cp, ValidationMode::Exact));
}

TEST(PucchF3SegLLRsTest, Small_nSym4_dmrs1_QPSK)
{
    // nSym=4, dmrs=1 -> sumNiUci=[2,3]; choose E_seg1 < 2 groups to stay j=0
    CaseParams cp{4, 1, 2, 1, static_cast<uint32_t>(1 * 12 * 2)};
    ASSERT_TRUE(run_one_case(cp));
}

// Additional cases to improve kernel branch coverage

TEST(PucchF3SegLLRsTest, QPSK_nSym5_j0)
{
    CaseParams cp{5, 2, 2, 1, static_cast<uint32_t>(2 * 12 * 2)}; // < 3 groups
    ASSERT_TRUE(run_one_case(cp));
}

TEST(PucchF3SegLLRsTest, QPSK_nSym6_j0)
{
    CaseParams cp{6, 2, 2, 1, static_cast<uint32_t>(3 * 12 * 2)}; // < 4 groups
    ASSERT_TRUE(run_one_case(cp));
}

TEST(PucchF3SegLLRsTest, QPSK_nSym8_j1)
{
    ASSERT_TRUE(run_case_for_j(8, 2, 2, 1, 1, ValidationMode::Exact));
}

TEST(PucchF3SegLLRsTest, QPSK_nSym9_j1)
{
    ASSERT_TRUE(run_case_for_j(9, 2, 2, 1, 1, ValidationMode::Exact));
}

TEST(PucchF3SegLLRsTest, QPSK_nSym10_dmrs2_j1)
{
    ASSERT_TRUE(run_case_for_j(10, 2, 2, 1, 1, ValidationMode::Exact));
}

TEST(PucchF3SegLLRsTest, QPSK_nSym10_dmrs4_j0)
{
    ASSERT_TRUE(run_case_for_j(10, 4, 2, 1, 0, ValidationMode::Exact));
}

TEST(PucchF3SegLLRsTest, QPSK_nSym11_dmrs2_j2)
{
    ASSERT_TRUE(run_case_for_j(11, 2, 2, 1, 2, ValidationMode::Exact));
}

TEST(PucchF3SegLLRsTest, QPSK_nSym11_dmrs4_j0)
{
    ASSERT_TRUE(run_case_for_j(11, 4, 2, 1, 0, ValidationMode::Exact));
}

TEST(PucchF3SegLLRsTest, QPSK_nSym13_dmrs2_j2)
{
    ASSERT_TRUE(run_case_for_j(13, 2, 2, 1, 2, ValidationMode::Exact));
}

TEST(PucchF3SegLLRsTest, QPSK_nSym13_dmrs4_j1)
{
    ASSERT_TRUE(run_case_for_j(13, 4, 2, 1, 1, ValidationMode::Exact));
}

TEST(PucchF3SegLLRsTest, EarlyExitBranch_CoversUciIdxGE)
{
    // Launch 2 UCIs but only 1 effective to trigger (uciIdx >= numUcis) path
    CaseParams cp{7, 2, 2, 1, static_cast<uint32_t>(3 * 12 * 2)};
    ASSERT_TRUE(run_case_with_launch_ucis(cp, 2, 1));
}

TEST(PucchF3SegLLRsTest, RoundingLoops_Eseg1Multiple_Eseg2NonMultiple)
{
    // Use prbSize=2 to get E_tot large enough. For 14 dmrs4, E_tot=2*24*10*Qm/2? With Qm=2 -> 480.
    // Set E_seg1 exact multiple of 192 and E_seg2 non-multiple to hit both round branches.
    CaseParams cp{14, 4, 2, 2, static_cast<uint32_t>(192)};
    ASSERT_TRUE(run_one_case(cp));
}

// Exercise optional descriptor async copy path within setup
TEST(PucchF3SegLLRsTest, DescriptorAsyncCopy_Enabled)
{
    // Simple stable case to validate the async copy flag path
    CaseParams cp{6, 2, 2, 1, static_cast<uint32_t>(2 * 12 * 2)};
    ASSERT_TRUE(run_one_case(cp, ValidationMode::Exact, true));
}

// -----------------------------------------------------------------------------
// Cover uncovered branch for nSym=14 with nSym_dmrs=2 (uciSymInd_row=15 ...)
// Exercise all j bands using the common helper

TEST(PucchF3SegLLRsTest, QPSK_nSym14_dmrs2_j0)
{
    ASSERT_TRUE(run_case_for_j(14, 2, 2, 1, 0, ValidationMode::SNR));
}

// -----------------------------------------------------------------------------
// Cover uncovered branch for nSym=12 with nSym_dmrs=4 (uciSymInd_row=12,...)
// Only j=0 is valid here (NiUci_2=NiUci_3=0). Exercise both QPSK and BPSK.

TEST(PucchF3SegLLRsTest, QPSK_nSym12_dmrs4_j0)
{
    ASSERT_TRUE(run_case_for_j(12, 4, 2, 1, 0, ValidationMode::Exact));
}

TEST(PucchF3SegLLRsTest, BPSK_nSym12_dmrs4_j0)
{
    ASSERT_TRUE(run_case_for_j(12, 4, 1, 1, 0, ValidationMode::SNR));
}

// -----------------------------------------------------------------------------
// Cover uncovered branch for nSym=4 (dmrs==1 and dmrs==2)

TEST(PucchF3SegLLRsTest, QPSK_nSym4_dmrs1_j0)
{
    ASSERT_TRUE(run_case_for_j(4, 1, 2, 1, 0, ValidationMode::Exact));
}

TEST(PucchF3SegLLRsTest, QPSK_nSym4_dmrs1_j1)
{
    ASSERT_TRUE(run_case_for_j(4, 1, 2, 1, 1, ValidationMode::Exact));
}

TEST(PucchF3SegLLRsTest, QPSK_nSym4_dmrs2_j0)
{
    ASSERT_TRUE(run_case_for_j(4, 2, 2, 1, 0, ValidationMode::Exact));
}

TEST(PucchF3SegLLRsTest, QPSK_nSym14_dmrs2_j1)
{
    ASSERT_TRUE(run_case_for_j(14, 2, 2, 1, 1, ValidationMode::SNR));
}

TEST(PucchF3SegLLRsTest, QPSK_nSym14_dmrs2_j2)
{
    ASSERT_TRUE(run_case_for_j(14, 2, 2, 1, 2, ValidationMode::SNR));
}

int main(int argc, char* argv[])
{
    // Initialize Google Test
    ::testing::InitGoogleTest(&argc, argv);

    // Run all tests
    int result = RUN_ALL_TESTS();

    return result;
}

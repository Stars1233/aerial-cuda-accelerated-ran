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
#include <cuda_runtime.h>
#include <stdint.h>
#include <memory>
#include <vector>
#include <iostream>
#include <algorithm>
#include <stdexcept>
#include <tuple>

#include "cuphy.h"
#include "cuphy.hpp"
#include "cuphy_internal.h"
#include "ss.hpp"


using cuphy::stream;
using cuphy::tensor_device;
using cuphy::make_unique_device;

namespace
{

// Utility: CUDA check for tests
static void AssertCudaSuccess(cudaError_t e)
{
    ASSERT_EQ(e, cudaSuccess) << "CUDA error: " << cudaGetErrorString(e);
}

// Simple SSB static parameters initialization (without HDF5 dependencies)
struct SimpleSSBStatic
{
    cuphySsbStatPrms_t ssbStatPrms;
    cuphyTracker_t     ssbTracker;

    SimpleSSBStatic(int max_cells_per_slot = 1)
    {
        ssbStatPrms.nMaxCellsPerSlot = max_cells_per_slot;
        ssbTracker.pMemoryFootprint  = nullptr;
        ssbStatPrms.pOutInfo         = &ssbTracker;
        ssbStatPrms.pDbgPrms         = nullptr;
    }
};

// Common test configuration structure
struct SSBTestConfig
{
    uint16_t nF;
    uint16_t nSym;
    uint16_t nPorts;
    uint16_t num_ssbs;
    uint64_t proc_mode;

    SSBTestConfig(uint16_t ssb_count = 1, uint64_t mode = SSB_PROC_MODE_STREAMS) :
        nF(273 * CUPHY_N_TONES_PER_PRB), nSym(OFDM_SYMBOLS_PER_SLOT), nPorts(1), num_ssbs(ssb_count), proc_mode(mode) {}
};

// -------------------------------------------------------------------------
// Host-side LUT copies (mirrors the __device__ __constant__ arrays in ss.cu)
// Must be declared before SSBTestHelper so verifyOutputQuantitative can call
// compute_pss_reference / compute_sss_reference.
// -------------------------------------------------------------------------

// SSB_PSS_X_EXT: first 127 elements are the M-sequence; next 86 repeat the
// start so that accesses [n + 43*NID2] (NID2 ∈ {0,1,2}) never need modulo.
static const uint8_t HOST_PSS_X_EXT[213] = {
    0,1,1,0,1,1,1,1,0,0,1,1,1,0,0,1,0,1,0,1,1,0,0,1,1,0,0,0,0,0,1,1,
    0,1,1,0,1,0,1,1,1,0,1,0,0,0,1,1,0,0,1,0,0,0,1,0,0,0,0,0,0,1,0,0,
    1,0,0,1,1,0,1,0,0,1,1,1,1,0,1,1,1,0,0,0,0,1,1,1,1,1,1,1,0,0,0,1,
    1,1,0,1,1,0,0,0,1,0,1,0,0,1,0,1,1,1,1,1,0,1,0,1,0,1,0,0,0,0,1,
    // repeated prefix (86 elements = 43*2)
    0,1,1,0,1,1,1,1,0,0,1,1,1,0,0,1,0,1,0,1,1,0,0,1,1,0,0,0,0,0,1,1,
    0,1,1,0,1,0,1,1,1,0,1,0,0,0,1,1,0,0,1,0,0,0,1,0,0,0,0,0,0,1,0,0,
    1,0,0,1,1,0,1,0,0,1,1,1,1,0,1,1,1,0,0,0,0,1
};

// SSS X0/X1 m-sequences (127 elements each, from 3GPP TS 38.211 §7.4.2.3)
static const uint8_t HOST_SSS_X0[127] = {
    1,0,0,0,0,0,0,1,0,0,1,0,0,1,1,0,1,0,0,1,1,1,1,0,1,1,1,0,0,0,0,1,
    1,1,1,1,1,1,0,0,0,1,1,1,0,1,1,0,0,0,1,0,1,0,0,1,0,1,1,1,1,1,0,1,
    0,1,0,1,0,0,0,0,1,0,1,1,0,1,1,1,1,0,0,1,1,1,0,0,1,0,1,0,1,1,0,0,
    1,1,0,0,0,0,0,1,1,0,1,1,0,1,0,1,1,1,0,1,0,0,0,1,1,0,0,1,0,0,0
};

static const uint8_t HOST_SSS_X1[127] = {
    1,0,0,0,0,0,0,1,0,0,0,0,0,1,1,0,0,0,0,1,0,1,0,0,0,1,1,1,1,0,0,1,
    0,0,0,1,0,1,1,0,0,1,1,1,0,1,0,1,0,0,1,1,1,1,1,0,1,0,0,0,0,1,1,1,
    0,0,0,1,0,0,1,0,0,1,1,0,1,1,0,1,0,1,1,0,1,1,1,1,0,1,1,0,0,0,1,1,
    0,1,0,0,1,0,1,1,1,0,1,1,1,0,0,1,1,0,0,1,0,1,0,1,0,1,1,1,1,1,1
};

// Compute the host-side PSS reference for a given NID.
// Returns a 127-element vector of real-valued BPSK samples scaled by beta_pss.
// Formula (3GPP TS 38.211 §7.4.2.2):
//   d_PSS(n) = beta_pss * (1 - 2 * PSS_X_EXT[n + 43*(NID%3)])  for n=0..126
static std::vector<float> compute_pss_reference(uint16_t NID, float beta_pss)
{
    const uint16_t     NID2 = NID % 3;
    std::vector<float> ref(127);
    for(int n = 0; n < 127; ++n)
        ref[n] = beta_pss * (1.0f - 2.0f * HOST_PSS_X_EXT[n + 43 * NID2]);
    return ref;
}

// Compute the host-side SSS reference for a given NID.
// Returns a 127-element vector of real-valued BPSK-product samples scaled by beta_sss.
// Formula (3GPP TS 38.211 §7.4.2.3):
//   d_SSS(n) = beta_sss * (1-2*x0[(n+m0)%127]) * (1-2*x1[(n+m1)%127])
//   where NID1=NID/3, NID2=NID%3,
//         m0 = (NID1/112)*15 + 5*NID2,   m1 = NID1 % 112
static std::vector<float> compute_sss_reference(uint16_t NID, float beta_sss)
{
    const uint16_t     NID1 = NID / 3;
    const uint16_t     NID2 = NID % 3;
    const int          m0   = (NID1 / 112) * 15 + 5 * NID2;
    const int          m1   = NID1 % 112;
    std::vector<float> ref(127);
    for(int n = 0; n < 127; ++n)
    {
        int lh = 1 - 2 * HOST_SSS_X0[(n + m0) % 127];
        int uh = 1 - 2 * HOST_SSS_X1[(n + m1) % 127];
        ref[n] = beta_sss * static_cast<float>(lh * uh);
    }
    return ref;
}

// Common SSB test setup helper
class SSBTestHelper {
public:
    stream                                strm;
    cudaStream_t                          strm_handle;
    SimpleSSBStatic                       static_params;
    std::unique_ptr<cuphySsbTxHndl_t>     ssb_handle;
    cuphy::unique_device_ptr<__half2>     d_output;
    std::unique_ptr<cuphy::tensor_device> output_tensor;
    cuphyStatus_t                         last_destroy_status = CUPHY_STATUS_SUCCESS;

    SSBTestHelper() :
        strm(cudaStreamNonBlocking), strm_handle(strm.handle()), static_params(1), ssb_handle(std::make_unique<cuphySsbTxHndl_t>())
    {
        cuphyStatus_t status = cuphyCreateSsbTx(ssb_handle.get(), &static_params.ssbStatPrms);
        if(status != CUPHY_STATUS_SUCCESS)
        {
            throw std::runtime_error("Failed to create SSB TX: " + std::string(cuphyGetErrorString(status)));
        }
    }

    ~SSBTestHelper()
    {
        (void)destroySsbHandle();
    }

    cuphyStatus_t destroySsbHandle()
    {
        if(!ssb_handle)
        {
            return last_destroy_status;
        }

        last_destroy_status = cuphyDestroySsbTx(*ssb_handle);
        if(last_destroy_status != CUPHY_STATUS_SUCCESS)
        {
            std::cerr << "Failed to destroy SSB TX: " << cuphyGetErrorString(last_destroy_status) << std::endl;
        }
        ssb_handle.reset();
        return last_destroy_status;
    }

    void setupOutputBuffer(const SSBTestConfig& config)
    {
        size_t output_size = static_cast<size_t>(config.nF) * config.nSym * config.nPorts;
        d_output           = cuphy::make_unique_device<__half2>(output_size);
        AssertCudaSuccess(cudaMemsetAsync(d_output.get(), 0, output_size * sizeof(__half2), strm_handle));

        output_tensor = std::make_unique<cuphy::tensor_device>(
            d_output.get(), CUPHY_C_16F, config.nF, config.nSym, config.nPorts, cuphy::tensor_flags::align_tight);
    }

    void resetOutputBuffer(const SSBTestConfig& config)
    {
        size_t output_size = static_cast<size_t>(config.nF) * config.nSym * config.nPorts;
        AssertCudaSuccess(cudaMemsetAsync(d_output.get(), 0, output_size * sizeof(__half2), strm_handle));
    }

    void runSSBTest(cuphySsbDynPrms_t& dyn_params)
    {
        cuphyStatus_t status = cuphySetupSsbTx(*ssb_handle, &dyn_params);
        ASSERT_EQ(status, CUPHY_STATUS_SUCCESS) << "Failed to setup SSB TX: " << cuphyGetErrorString(status);

        status = cuphyRunSsbTx(*ssb_handle, (dyn_params.procModeBmsk == SSB_PROC_MODE_GRAPHS) ? 1 : 0);
        ASSERT_EQ(status, CUPHY_STATUS_SUCCESS) << "Failed to run SSB TX: " << cuphyGetErrorString(status);
    }

    struct SSBValidationResult
    {
        bool                     passed;
        int                      non_zero_count;
        int                      pss_samples;
        int                      sss_samples;
        int                      pbch_samples;
        int                      dmrs_samples;
        float                    max_magnitude;
        float                    pss_power_avg;
        float                    sss_power_avg;
        float                    pbch_power_avg;
        std::vector<std::string> errors;

        SSBValidationResult() :
            passed(false), non_zero_count(0), pss_samples(0), sss_samples(0), pbch_samples(0), dmrs_samples(0), max_magnitude(0.0f), pss_power_avg(0.0f), sss_power_avg(0.0f), pbch_power_avg(0.0f) {}
    };

    SSBValidationResult verifyOutputQuantitative(const SSBTestConfig&            config,
                                                 const cuphyPerCellSsbDynPrms_t& cell_params,
                                                 const cuphyPerSsBlockDynPrms_t& ssb_params,
                                                 bool                            verbose = false)
    {
        SSBValidationResult result;

        // Get full output buffer for comprehensive analysis
        size_t               output_size = static_cast<size_t>(config.nF) * config.nSym * config.nPorts;
        std::vector<__half2> host_output(output_size);

        AssertCudaSuccess(cudaMemcpyAsync(host_output.data(), d_output.get(), host_output.size() * sizeof(__half2), cudaMemcpyDeviceToHost, strm_handle));
        AssertCudaSuccess(cudaStreamSynchronize(strm_handle));

        // Signal type categorization and validation
        float             pss_power_sum = 0.0f, sss_power_sum = 0.0f, pbch_power_sum = 0.0f;
        std::vector<bool> found_pss_locations(127, false);
        std::vector<bool> found_sss_locations(127, false);
        int               pss_power_outliers = 0;
        int               sss_power_outliers = 0;
        int               pbch_power_outliers = 0;

        // Extract SSB positioning parameters
        uint16_t       f0       = ssb_params.f0;
        uint8_t        t0       = ssb_params.t0;
        float          beta_pss = ssb_params.beta_pss;
        float          beta_sss = ssb_params.beta_sss;
        const uint16_t NID      = cell_params.NID;

        // Analyze each sample
        for(size_t i = 0; i < host_output.size(); ++i)
        {
            float real_val     = static_cast<float>(host_output[i].x);
            float imag_val     = static_cast<float>(host_output[i].y);
            float magnitude_sq = real_val * real_val + imag_val * imag_val;
            float magnitude    = std::sqrt(magnitude_sq);

            if(magnitude_sq > 1e-10f)
            {
                result.non_zero_count++;
                result.max_magnitude = std::max(result.max_magnitude, magnitude);

                // Determine signal type based on position
                size_t freq_idx = i % config.nF;
                size_t sym_idx  = (i / config.nF) % config.nSym;
                // PSS validation (symbol t0, frequencies f0+56 to f0+182)
                if(sym_idx == t0 && freq_idx >= (f0 + 56) && freq_idx <= (f0 + 182))
                {
                    result.pss_samples++;
                    pss_power_sum += magnitude_sq;
                    size_t pss_idx = freq_idx - (f0 + 56);
                    if(pss_idx < 127)
                    {
                        found_pss_locations[pss_idx] = true;
                    }

                    // Validate PSS power level (should be ~beta_pss)
                    float expected_pss_power = beta_pss * beta_pss;
                    if(std::abs(magnitude_sq - expected_pss_power) > expected_pss_power * 0.1f)
                    {
                        // Track outliers instead of silently ignoring them.
                        pss_power_outliers++;
                    }
                }
                // SSS validation (symbol t0+2, frequencies f0+56 to f0+182)
                else if(sym_idx == (t0 + 2) && freq_idx >= (f0 + 56) && freq_idx <= (f0 + 182))
                {
                    result.sss_samples++;
                    sss_power_sum += magnitude_sq;
                    size_t sss_idx = freq_idx - (f0 + 56);
                    if(sss_idx < 127)
                    {
                        found_sss_locations[sss_idx] = true;
                    }

                    // Validate SSS power level (should be ~beta_sss)
                    float expected_sss_power = beta_sss * beta_sss;
                    if(std::abs(magnitude_sq - expected_sss_power) > expected_sss_power * 0.1f)
                    {
                        sss_power_outliers++;
                    }
                }
                // PBCH validation (symbols t0+1 and t0+3, plus partial t0+2)
                else if((sym_idx == (t0 + 1) || sym_idx == (t0 + 3)) ||
                        (sym_idx == (t0 + 2) && (freq_idx < (f0 + 56) || freq_idx > (f0 + 182))))
                {
                    result.pbch_samples++;
                    pbch_power_sum += magnitude_sq;

                    // Validate PBCH QPSK constellation (should be ±beta_sss_factor)
                    float beta_sss_factor     = beta_sss * 0.70710678f;
                    float expected_pbch_power = 2.0f * beta_sss_factor * beta_sss_factor; // I²+Q²
                    if(std::abs(magnitude_sq - expected_pbch_power) > expected_pbch_power * 0.15f)
                    {
                        pbch_power_outliers++;
                    }
                }
                // DMRS validation (interspersed with PBCH)
                else
                {
                    result.dmrs_samples++;
                }
            }
        }

        // Calculate average power levels
        if(result.pss_samples > 0) result.pss_power_avg = pss_power_sum / result.pss_samples;
        if(result.sss_samples > 0) result.sss_power_avg = sss_power_sum / result.sss_samples;
        if(result.pbch_samples > 0) result.pbch_power_avg = pbch_power_sum / result.pbch_samples;

        // Quantitative validation checks
        result.passed = true;

        // Check 1: Expected PSS sample count (127 samples expected)
        if(result.pss_samples < 120 || result.pss_samples > 127)
        {
            result.errors.push_back("PSS sample count out of range: " + std::to_string(result.pss_samples) + " (expected ~127)");
            result.passed = false;
        }

        // Check 2: Expected SSS sample count (127 samples expected)
        if(result.sss_samples < 120 || result.sss_samples > 127)
        {
            result.errors.push_back("SSS sample count out of range: " + std::to_string(result.sss_samples) + " (expected ~127)");
            result.passed = false;
        }

        // Check 3: PBCH+DMRS sample count should be significant (576 REs expected total)
        // Note: DMRS samples are interspersed with PBCH, so total count should be ~576
        int total_pbch_dmrs = result.pbch_samples + result.dmrs_samples;
        if(total_pbch_dmrs < 550 || total_pbch_dmrs > 600)
        {
            result.errors.push_back("PBCH+DMRS sample count out of range: " + std::to_string(total_pbch_dmrs) + " (expected ~576)");
            result.passed = false;
        }

        // Check 4: Power level validation
        float expected_pss_power = beta_pss * beta_pss;
        if(std::abs(result.pss_power_avg - expected_pss_power) > expected_pss_power * 0.2f)
        {
            result.errors.push_back("PSS power deviation: " + std::to_string(result.pss_power_avg) +
                                    " vs expected " + std::to_string(expected_pss_power));
            result.passed = false;
        }

        float expected_sss_power = beta_sss * beta_sss;
        if(std::abs(result.sss_power_avg - expected_sss_power) > expected_sss_power * 0.2f)
        {
            result.errors.push_back("SSS power deviation: " + std::to_string(result.sss_power_avg) +
                                    " vs expected " + std::to_string(expected_sss_power));
            result.passed = false;
        }

        // Check 5: Signal coverage validation
        int pss_coverage = std::count(found_pss_locations.begin(), found_pss_locations.end(), true);
        int sss_coverage = std::count(found_sss_locations.begin(), found_sss_locations.end(), true);

        if(pss_coverage < 120)
        {
            result.errors.push_back("Insufficient PSS frequency coverage: " + std::to_string(pss_coverage) + "/127");
            result.passed = false;
        }

        if(sss_coverage < 120)
        {
            result.errors.push_back("Insufficient SSS frequency coverage: " + std::to_string(sss_coverage) + "/127");
            result.passed = false;
        }

        // Check 6: Per-sample power consistency checks
        if(result.pss_samples > 0 && pss_power_outliers > 2)
        {
            result.errors.push_back("PSS per-sample power outliers exceed tolerance: " + std::to_string(pss_power_outliers) +
                                    "/" + std::to_string(result.pss_samples));
            result.passed = false;
        }

        if(result.sss_samples > 0 && sss_power_outliers > 2)
        {
            result.errors.push_back("SSS per-sample power outliers exceed tolerance: " + std::to_string(sss_power_outliers) +
                                    "/" + std::to_string(result.sss_samples));
            result.passed = false;
        }

        if(result.pbch_samples > 0 && pbch_power_outliers > 8)
        {
            result.errors.push_back("PBCH per-sample power outliers exceed tolerance: " + std::to_string(pbch_power_outliers) +
                                    "/" + std::to_string(result.pbch_samples));
            result.passed = false;
        }

        // Check 7: Total non-zero samples should be reasonable
        int expected_total = result.pss_samples + result.sss_samples + result.pbch_samples + result.dmrs_samples;
        if(result.non_zero_count < expected_total * 0.9f)
        {
            result.errors.push_back("Total non-zero count too low: " + std::to_string(result.non_zero_count));
            result.passed = false;
        }

        // Check 8: Golden-sequence correctness (PSS and SSS IQ values vs. reference).
        // Skipped when precoding is enabled because the precoding matrix transforms
        // the real-valued BPSK samples into complex port-specific values.
        // FP16 quantisation tolerance: ±0.01 for a unit-amplitude (beta=1) signal.
        if(!ssb_params.enablePrcdBf)
        {
            // PSS: symbol t0, subcarriers [f0+56 .. f0+182] — real-valued BPSK
            auto ref_pss          = compute_pss_reference(NID, beta_pss);
            int  pss_seq_mismatches = 0;
            for(int n = 0; n < 127; ++n)
            {
                size_t idx       = static_cast<size_t>(t0) * config.nF + (f0 + 56 + n);
                float  gpu_val   = (idx < host_output.size()) ? static_cast<float>(host_output[idx].x) : 0.0f;
                float  tolerance = 0.01f * std::fabs(beta_pss) + 1e-4f;
                if(std::fabs(gpu_val - ref_pss[n]) > tolerance)
                    ++pss_seq_mismatches;
            }
            if(pss_seq_mismatches > 0)
            {
                result.errors.push_back("PSS sequence mismatch: " + std::to_string(pss_seq_mismatches) +
                                        "/127 samples wrong for NID=" + std::to_string(NID));
                result.passed = false;
            }

            // SSS: symbol t0+2, subcarriers [f0+56 .. f0+182] — real-valued BPSK product
            auto ref_sss          = compute_sss_reference(NID, beta_sss);
            int  sss_seq_mismatches = 0;
            for(int n = 0; n < 127; ++n)
            {
                size_t idx       = static_cast<size_t>(t0 + 2) * config.nF + (f0 + 56 + n);
                float  gpu_val   = (idx < host_output.size()) ? static_cast<float>(host_output[idx].x) : 0.0f;
                float  tolerance = 0.01f * std::fabs(beta_sss) + 1e-4f;
                if(std::fabs(gpu_val - ref_sss[n]) > tolerance)
                    ++sss_seq_mismatches;
            }
            if(sss_seq_mismatches > 0)
            {
                result.errors.push_back("SSS sequence mismatch: " + std::to_string(sss_seq_mismatches) +
                                        "/127 samples wrong for NID=" + std::to_string(NID));
                result.passed = false;
            }
        }

        if(verbose || !result.passed)
        {
            std::cout << "=== Quantitative SSB Validation Results ===" << std::endl;
            std::cout << "Status: " << (result.passed ? "PASS" : "FAIL") << std::endl;
            std::cout << "Total samples: " << output_size << std::endl;
            std::cout << "Non-zero samples: " << result.non_zero_count << std::endl;
            std::cout << "PSS samples: " << result.pss_samples << " (coverage: " << pss_coverage << "/127)" << std::endl;
            std::cout << "SSS samples: " << result.sss_samples << " (coverage: " << sss_coverage << "/127)" << std::endl;
            std::cout << "PBCH samples: " << result.pbch_samples << std::endl;
            std::cout << "DMRS samples: " << result.dmrs_samples << std::endl;

            // Coverage analysis for DMRS
            if(result.dmrs_samples == 0)
            {
                std::cout << "NOTE: DMRS=0 indicates thread range 576-719 not executing or" << std::endl;
                std::cout << "      DMRS samples are classified as PBCH due to overlapping locations." << std::endl;
                std::cout << "      Total PBCH+DMRS should be 576 REs (432 PBCH + 144 DMRS)." << std::endl;
            }

            std::cout << "Max magnitude: " << result.max_magnitude << std::endl;
            std::cout << "PSS avg power: " << result.pss_power_avg << " (expected: " << expected_pss_power << ")" << std::endl;
            std::cout << "SSS avg power: " << result.sss_power_avg << " (expected: " << expected_sss_power << ")" << std::endl;
            std::cout << "PBCH avg power: " << result.pbch_power_avg << std::endl;
            std::cout << "PSS outliers: " << pss_power_outliers << std::endl;
            std::cout << "SSS outliers: " << sss_power_outliers << std::endl;
            std::cout << "PBCH outliers: " << pbch_power_outliers << std::endl;

            if(!result.errors.empty())
            {
                std::cout << "Validation Errors:" << std::endl;
                for(const auto& error : result.errors)
                {
                    std::cout << "  - " << error << std::endl;
                }
            }
            std::cout << "===========================================" << std::endl;
        }

        return result;
    }
};

// Common function to create SSB dynamic parameters
cuphySsbDynPrms_t createSSBDynamicParams(
    const SSBTestConfig&                   config,
    cudaStream_t                           strm_handle,
    std::vector<cuphyPerCellSsbDynPrms_t>& per_cell_params,
    std::vector<cuphyPerSsBlockDynPrms_t>& per_ssb_params,
    std::vector<uint32_t>&                 mib_input,
    std::vector<cuphyTensorPrm_t>&         output_tensor_params,
    cuphySsbDataIn_t&                      data_in,
    cuphySsbDataOut_t&                     data_out)
{
    // Setup data structures
    data_in.pMibInput   = mib_input.data();
    data_in.pBufferType = cuphySsbDataIn_t::CPU_BUFFER;
    data_out.pTDataTx   = output_tensor_params.data();

    // Assemble dynamic parameters
    cuphySsbDynPrms_t dyn_params{};
    dyn_params.cuStream             = strm_handle;
    dyn_params.procModeBmsk         = config.proc_mode;
    dyn_params.nCells               = 1;
    dyn_params.pPerCellSsbDynParams = per_cell_params.data();
    dyn_params.nSSBlocks            = config.num_ssbs;
    dyn_params.pPerSsBlockParams    = per_ssb_params.data();
    dyn_params.nPrecodingMatrices   = 0;
    dyn_params.pPmwParams           = nullptr;
    dyn_params.pDataIn              = &data_in;
    dyn_params.pDataOut             = &data_out;

    return dyn_params;
}

// Common function to fill basic cell parameters
void fillBasicCellParams(cuphyPerCellSsbDynPrms_t& cell_params, uint16_t nid, uint16_t nF)
{
    cell_params.NID           = nid;
    cell_params.nHF           = 0;
    cell_params.Lmax          = 4;
    cell_params.SFN           = 42;
    cell_params.k_SSB         = 0;
    cell_params.nF            = nF;
    cell_params.slotBufferIdx = 0;
}

// Common function to fill basic SSB parameters
void fillBasicSSBParams(cuphyPerSsBlockDynPrms_t& ssb_params, uint16_t block_index)
{
    ssb_params.f0           = 56;
    ssb_params.t0           = 2;
    ssb_params.blockIndex   = block_index;
    ssb_params.beta_pss     = 1.0f;
    ssb_params.beta_sss     = 1.0f;
    ssb_params.cell_index   = 0;
    ssb_params.enablePrcdBf = 0;
    ssb_params.pmwPrmIdx    = 0;
}

struct SSBCaseData
{
    std::vector<cuphyPerCellSsbDynPrms_t> per_cell_params;
    std::vector<cuphyPerSsBlockDynPrms_t> per_ssb_params;
    std::vector<uint32_t>                 mib_input;
    std::vector<cuphyTensorPrm_t>         output_tensor_params;
    cuphySsbDataIn_t                      data_in{};
    cuphySsbDataOut_t                     data_out{};

    explicit SSBCaseData(uint16_t num_ssbs = 1) :
        per_cell_params(1), per_ssb_params(num_ssbs), mib_input(num_ssbs), output_tensor_params(1)
    {
    }

    void bindOutputTensor(SSBTestHelper& helper)
    {
        output_tensor_params[0].desc  = helper.output_tensor->desc().handle();
        output_tensor_params[0].pAddr = helper.output_tensor->addr();
    }

    cuphySsbDynPrms_t makeDynParams(const SSBTestConfig& config, cudaStream_t stream_handle)
    {
        return createSSBDynamicParams(
            config, stream_handle, per_cell_params, per_ssb_params, mib_input, output_tensor_params, data_in, data_out);
    }
};

class SSBTestFixture : public ::testing::Test
{
protected:
    SSBTestHelper helper;
    SSBTestConfig config{1, SSB_PROC_MODE_STREAMS};

    void TearDown() override
    {
        cuphyStatus_t status = helper.destroySsbHandle();
        EXPECT_EQ(status, CUPHY_STATUS_SUCCESS) << "Failed to destroy SSB TX: " << cuphyGetErrorString(status);
    }

    void setupConfig(uint16_t ssb_count = 1, uint64_t mode = SSB_PROC_MODE_STREAMS)
    {
        config = SSBTestConfig(ssb_count, mode);
        helper.setupOutputBuffer(config);
    }

    SSBCaseData makeBasicCase(uint16_t nid = 123, uint16_t block_index = 0, uint32_t mib_payload = 0x123456, uint16_t num_ssbs = 1)
    {
        SSBCaseData data(num_ssbs);
        fillBasicCellParams(data.per_cell_params[0], nid, config.nF);
        for(uint16_t i = 0; i < num_ssbs; ++i)
        {
            fillBasicSSBParams(data.per_ssb_params[i], block_index + i);
            data.mib_input[i] = mib_payload + i;
        }
        data.bindOutputTensor(helper);
        return data;
    }
};

} // namespace

// Test: Basic SSB quantitative validation with manual parameters

TEST_F(SSBTestFixture, BasicQuantitativeValidation)
{
    setupConfig();
    auto data = makeBasicCase(123, 0, 0x123456);
    auto dyn_params = data.makeDynParams(config, helper.strm_handle);

    // Run streams mode test with quantitative validation
    helper.runSSBTest(dyn_params);
    auto validation_result = helper.verifyOutputQuantitative(config, data.per_cell_params[0], data.per_ssb_params[0], true);

    EXPECT_TRUE(validation_result.passed) << "SSB quantitative validation failed";
    EXPECT_GT(validation_result.non_zero_count, 500) << "SSB output has too few non-zero samples";
    EXPECT_GT(validation_result.pss_samples, 120) << "Insufficient PSS samples";
    EXPECT_GT(validation_result.sss_samples, 120) << "Insufficient SSS samples";
    EXPECT_GT(validation_result.pbch_samples + validation_result.dmrs_samples, 550) << "Insufficient PBCH+DMRS samples";

    // Test graphs mode with same quantitative validation
    dyn_params.procModeBmsk = SSB_PROC_MODE_GRAPHS;
    helper.resetOutputBuffer(config);
    helper.runSSBTest(dyn_params);
    auto graph_validation_result = helper.verifyOutputQuantitative(config, data.per_cell_params[0], data.per_ssb_params[0], false);
    EXPECT_TRUE(graph_validation_result.passed) << "SSB graphs mode quantitative validation failed";
}

// Test: Multiple NID values to exercise compute_qam_index_v2 with different index arrays
TEST_F(SSBTestFixture, MultipleNIDsForQamIndexCoverage)
{
    setupConfig();

    // Test with different NID values to exercise different index arrays in compute_qam_index_v2
    // NID & 0x3 determines which index array is used (0, 1, 2, 3)
    std::vector<uint16_t> test_nids = {0, 1, 2, 3, 127, 255, 511, 1007};

    for(uint16_t nid : test_nids)
    {
        SCOPED_TRACE("NID=" + std::to_string(nid));
        helper.resetOutputBuffer(config);
        auto data = makeBasicCase(nid, 0, 0xFEDCBA);
        auto dyn_params = data.makeDynParams(config, helper.strm_handle);
        EXPECT_NO_FATAL_FAILURE(helper.runSSBTest(dyn_params));
        auto result = helper.verifyOutputQuantitative(config, data.per_cell_params[0], data.per_ssb_params[0], false);
        EXPECT_TRUE(result.passed) << "Quantitative validation failed for NID " << nid;
    }

    std::cout << "Tested " << test_nids.size() << " different NID values for compute_qam_index_v2 coverage" << std::endl;
}

// Test: Multiple SSB blocks with varied parameters for comprehensive coverage
TEST_F(SSBTestFixture, MultipleSSBsComprehensiveCoverage)
{
    setupConfig(3, SSB_PROC_MODE_STREAMS); // 3 SSBs
    auto data = makeBasicCase(42, 0, 0xABCDE0, config.num_ssbs);

    // Fill per-cell parameters with different values for variety
    data.per_cell_params[0].NID           = 42;
    data.per_cell_params[0].nHF           = 1;
    data.per_cell_params[0].Lmax          = 8;
    data.per_cell_params[0].SFN           = 100;
    data.per_cell_params[0].k_SSB         = 4;
    data.per_cell_params[0].nF            = config.nF;
    data.per_cell_params[0].slotBufferIdx = 0;

    // Fill per-SS block parameters with different values to trigger different code paths
    for(uint16_t i = 0; i < config.num_ssbs; ++i)
    {
        data.per_ssb_params[i].f0       = 56 + i * 16;
        data.per_ssb_params[i].t0       = 2 + i;
        data.per_ssb_params[i].beta_pss = 1.0f + i * 0.1f;
        data.per_ssb_params[i].beta_sss = 1.0f + i * 0.1f;

        data.mib_input[i] = 0xABCDE0 + i * 0x1000; // Different MIB payloads
    }

    // Create and run streams mode test
    auto dyn_params = data.makeDynParams(config, helper.strm_handle);

    helper.runSSBTest(dyn_params);
    auto multi_ssb_result = helper.verifyOutputQuantitative(config, data.per_cell_params[0], data.per_ssb_params[0], true);
    std::cout << "Multi-SSB Test Results:" << std::endl;
    std::cout << "  Non-zero samples: " << multi_ssb_result.non_zero_count << std::endl;
    std::cout << "  PSS samples: " << multi_ssb_result.pss_samples << std::endl;
    std::cout << "  SSS samples: " << multi_ssb_result.sss_samples << std::endl;
    std::cout << "  PBCH+DMRS samples: " << (multi_ssb_result.pbch_samples + multi_ssb_result.dmrs_samples) << std::endl;
    // Relaxed assertions for multi-SSB runs (validate presence, not exact per-SSB totals)
    EXPECT_GE(multi_ssb_result.non_zero_count, 800) << "Should produce substantial output";
    EXPECT_GE(multi_ssb_result.pss_samples, 127) << "At least one SSB PSS should be present";
    EXPECT_GE(multi_ssb_result.sss_samples, 127) << "At least one SSB SSS should be present";
    EXPECT_GE(multi_ssb_result.pbch_samples + multi_ssb_result.dmrs_samples, 576) << "At least one SSB PBCH+DMRS should be present";

    // Test graphs mode
    dyn_params.procModeBmsk = SSB_PROC_MODE_GRAPHS;
    helper.resetOutputBuffer(config);
    helper.runSSBTest(dyn_params);
    auto multi_ssb_graph_result = helper.verifyOutputQuantitative(config, data.per_cell_params[0], data.per_ssb_params[0], false);
    std::cout << "Multi-SSB Graphs Results:" << std::endl;
    std::cout << "  Non-zero samples: " << multi_ssb_graph_result.non_zero_count << std::endl;
    std::cout << "  PSS samples: " << multi_ssb_graph_result.pss_samples << std::endl;
    std::cout << "  SSS samples: " << multi_ssb_graph_result.sss_samples << std::endl;
    std::cout << "  PBCH+DMRS samples: " << (multi_ssb_graph_result.pbch_samples + multi_ssb_graph_result.dmrs_samples) << std::endl;
    EXPECT_GE(multi_ssb_graph_result.non_zero_count, 800) << "Graphs mode should produce substantial output";
    EXPECT_GE(multi_ssb_graph_result.pss_samples, 127) << "Graphs mode should include PSS";
    EXPECT_GE(multi_ssb_graph_result.sss_samples, 127) << "Graphs mode should include SSS";
    EXPECT_GE(multi_ssb_graph_result.pbch_samples + multi_ssb_graph_result.dmrs_samples, 576) << "Graphs mode should include PBCH+DMRS";
}

// Test: DMRS processing investigation with varied SSB configurations
TEST_F(SSBTestFixture, DMRSProcessingInvestigation)
{
    setupConfig();

    // Try different parameter combinations that might trigger DMRS processing
    std::vector<std::tuple<uint16_t, uint16_t, uint16_t, uint8_t, std::string>> dmrs_test_cases = {
        // {NID, f0, t0, Lmax, description}
        {0, 0, 0, 4, "Minimal case"},
        {1, 56, 2, 4, "Standard SSB location"},
        {42, 100, 4, 8, "Different offsets, Lmax=8"},
        {123, 200, 6, 8, "Higher offsets"},
        {511, 0, 0, 64, "Lmax=64 case"},
    };

    for(auto [nid, f0, t0, lmax, desc] : dmrs_test_cases)
    {
        SCOPED_TRACE(desc);
        helper.resetOutputBuffer(config);
        auto data = makeBasicCase(nid, 0, static_cast<uint32_t>(0x123000 + nid));
        data.per_cell_params[0].Lmax = lmax;
        data.per_cell_params[0].SFN  = 0;
        data.per_ssb_params[0].f0    = f0;
        data.per_ssb_params[0].t0    = t0;

        // Create and run test
        auto dyn_params = data.makeDynParams(config, helper.strm_handle);

        helper.runSSBTest(dyn_params);

        // Validate this configuration with quantitative analysis
        auto dmrs_result = helper.verifyOutputQuantitative(config, data.per_cell_params[0], data.per_ssb_params[0], false);
        EXPECT_GT(dmrs_result.non_zero_count, 0) << "Output should not be empty";

        std::cout << "DMRS investigation: " << desc << " (NID=" << nid << ", f0=" << f0
                  << ", t0=" << t0 << ", Lmax=" << lmax << ")" << std::endl;
        std::cout << "  Non-zero samples: " << dmrs_result.non_zero_count
                  << ", DMRS: " << dmrs_result.dmrs_samples << std::endl;
    }
}

// Test: Enhanced SSB signal analysis with quantitative validation
TEST_F(SSBTestFixture, EnhancedSignalAnalysisAndValidation)
{
    setupConfig();
    auto data = makeBasicCase(1, 0, 0x000001);
    auto dyn_params = data.makeDynParams(config, helper.strm_handle);

    // Run and analyze detailed output with quantitative validation
    helper.runSSBTest(dyn_params);

    // Use the new quantitative validation function
    auto analysis_result = helper.verifyOutputQuantitative(config, data.per_cell_params[0], data.per_ssb_params[0], true);

    std::cout << "=== Enhanced SSB Output Analysis ===" << std::endl;
    std::cout << "Validation Status: " << (analysis_result.passed ? "PASS" : "FAIL") << std::endl;
    std::cout << "Total non-zero samples: " << analysis_result.non_zero_count << std::endl;
    std::cout << "PSS samples: " << analysis_result.pss_samples << " (avg power: " << analysis_result.pss_power_avg << ")" << std::endl;
    std::cout << "SSS samples: " << analysis_result.sss_samples << " (avg power: " << analysis_result.sss_power_avg << ")" << std::endl;
    std::cout << "PBCH samples: " << analysis_result.pbch_samples << " (avg power: " << analysis_result.pbch_power_avg << ")" << std::endl;
    std::cout << "DMRS samples: " << analysis_result.dmrs_samples << std::endl;

    // Note: DMRS coverage issue - this helps document the problem
    if(analysis_result.dmrs_samples == 0)
    {
        std::cout << "*** NO DMRS SAMPLES FOUND ***" << std::endl;
        std::cout << "This explains why compute_dmrs_index_v2 has 0% coverage." << std::endl;
        std::cout << "Either DMRS thread range (576-719) is not executing, or" << std::endl;
        std::cout << "DMRS samples are being classified as PBCH due to overlapping locations." << std::endl;
        std::cout << "Total PBCH+DMRS = " << (analysis_result.pbch_samples + analysis_result.dmrs_samples) << " matches expected 576." << std::endl;
    }

    // Enhanced validation assertions
    EXPECT_GT(analysis_result.non_zero_count, 0) << "SSB should produce some output";
    EXPECT_GT(analysis_result.pss_samples, 100) << "Should have significant PSS output";
    EXPECT_GT(analysis_result.sss_samples, 100) << "Should have significant SSS output";
    EXPECT_GT(analysis_result.pbch_samples + analysis_result.dmrs_samples, 500) << "Should have significant PBCH+DMRS output";

    // Power level validation
    EXPECT_GT(analysis_result.pss_power_avg, 0.5f) << "PSS power should be reasonable";
    EXPECT_GT(analysis_result.sss_power_avg, 0.5f) << "SSS power should be reasonable";
}

// Test: Different Lmax values for scrambling sequence generation coverage
TEST_F(SSBTestFixture, LmaxVariationsForScramblingCoverage)
{
    setupConfig();

    // Test different Lmax values which affect the scrambling sequence generation
    std::vector<std::tuple<uint8_t, std::string>> lmax_cases = {
        {4, "Lmax=4 (typical)"},
        {8, "Lmax=8"},
        {64, "Lmax=64 (maximum)"}};

    for(auto [lmax, desc] : lmax_cases)
    {
        SCOPED_TRACE(desc);
        helper.resetOutputBuffer(config);
        auto data = makeBasicCase(123, 0, 0x555555);
        data.per_cell_params[0].Lmax = lmax;
        auto dyn_params = data.makeDynParams(config, helper.strm_handle);

        helper.runSSBTest(dyn_params);
        auto result = helper.verifyOutputQuantitative(config, data.per_cell_params[0], data.per_ssb_params[0], false);
        EXPECT_TRUE(result.passed) << "Lmax case failed: " << desc;

        std::cout << "Tested " << desc << std::endl;
    }
}

// Test: Edge cases and boundary conditions
TEST_F(SSBTestFixture, EdgeCasesAndBoundaryConditions)
{
    setupConfig();

    // Test edge cases for various parameters
    std::vector<std::tuple<uint16_t, uint16_t, uint16_t, uint8_t, std::string>> edge_cases = {
        // {NID, f0, t0, blockIndex, description}
        {0, 0, 0, 0, "All zeros"},
        {1023, 272, 10, 7, "Maximum values"},
        {511, 136, 5, 3, "Mid-range values"},
        {3, 240, 8, 1, "Boundary values"},
    };

    for(auto [nid, f0, t0, block_idx, desc] : edge_cases)
    {
        SCOPED_TRACE(desc);
        helper.resetOutputBuffer(config);
        auto data = makeBasicCase(nid, block_idx, 0xFFFFFF);

        // Override with edge case values
        data.per_ssb_params[0].f0 = f0;
        data.per_ssb_params[0].t0 = t0;

        auto dyn_params = data.makeDynParams(config, helper.strm_handle);
        EXPECT_NO_FATAL_FAILURE(helper.runSSBTest(dyn_params));
        auto result = helper.verifyOutputQuantitative(config, data.per_cell_params[0], data.per_ssb_params[0], false);
        EXPECT_GT(result.non_zero_count, 0) << "Edge case should produce non-empty output";
        std::cout << "Edge case test passed: " << desc << std::endl;
    }
}

// Test: Precoding enabled with quantitative validation
TEST_F(SSBTestFixture, PrecodingEnabledQuantitativeValidation)
{
    setupConfig();
    // Precoding matrix below uses 2 ports, so allocate a 2-port output buffer.
    config.nPorts = 2;
    helper.setupOutputBuffer(config);
    auto data = makeBasicCase(42, 0, 0xABCDEF);
    std::vector<cuphyPmWOneLayer_t>       precoding_matrix(1);

    // Fill SSB parameters with precoding enabled
    data.per_ssb_params[0].enablePrcdBf = 1; // Enable precoding
    data.per_ssb_params[0].pmwPrmIdx    = 0; // Use first precoding matrix

    // Setup simple precoding matrix (2 ports)
    precoding_matrix[0].nPorts      = 2;
    precoding_matrix[0].matrix[0].x = 1.0f; // Port 0: (1+0j)
    precoding_matrix[0].matrix[0].y = 0.0f;
    precoding_matrix[0].matrix[1].x = 0.0f; // Port 1: (0+1j)
    precoding_matrix[0].matrix[1].y = 1.0f;

    // Keep matrix ports aligned with output tensor ports to avoid OOB writes.
    ASSERT_EQ(config.nPorts, precoding_matrix[0].nPorts);

    // Create dynamic parameters with precoding
    auto dyn_params = data.makeDynParams(config, helper.strm_handle);

    // Override precoding parameters
    dyn_params.nPrecodingMatrices = 1;
    dyn_params.pPmwParams         = precoding_matrix.data();

    // Run test with quantitative validation
    helper.runSSBTest(dyn_params);

    auto precoding_result = helper.verifyOutputQuantitative(config, data.per_cell_params[0], data.per_ssb_params[0], true);
    std::cout << "Precoding test validation: " << (precoding_result.passed ? "PASS" : "FAIL") << std::endl;
    std::cout << "Non-zero samples: " << precoding_result.non_zero_count << std::endl;

    // Precoding should still produce valid SSB signals
    EXPECT_GT(precoding_result.non_zero_count, 0) << "Precoding test should produce non-empty output";
    EXPECT_GT(precoding_result.pss_samples, 120) << "Precoding should not affect PSS generation";
    EXPECT_GT(precoding_result.sss_samples, 120) << "Precoding should not affect SSS generation";
    EXPECT_GT(precoding_result.pbch_samples + precoding_result.dmrs_samples, 550) << "Precoding should not affect PBCH+DMRS generation";

    // Test graphs mode as well
    dyn_params.procModeBmsk = SSB_PROC_MODE_GRAPHS;
    helper.resetOutputBuffer(config);
    helper.runSSBTest(dyn_params);
    auto precoding_graph_result = helper.verifyOutputQuantitative(config, data.per_cell_params[0], data.per_ssb_params[0], false);
    EXPECT_GT(precoding_graph_result.non_zero_count, 0) << "Precoding graphs mode should produce non-empty output";
    EXPECT_GT(precoding_graph_result.pss_samples, 120) << "Precoding graphs mode should include PSS";
    EXPECT_GT(precoding_graph_result.sss_samples, 120) << "Precoding graphs mode should include SSS";
    EXPECT_GT(precoding_graph_result.pbch_samples + precoding_graph_result.dmrs_samples, 550)
        << "Precoding graphs mode should include PBCH+DMRS";
}

// Test: Comprehensive quantitative validation with cross-validation
TEST_F(SSBTestFixture, ComprehensiveQuantitativeValidation)
{
    setupConfig();
    auto data = makeBasicCase(100, 0, 0x555AAA);

    // Use standard 5G NR parameters
    data.per_ssb_params[0].beta_pss = 2.0f; // Higher PSS power
    data.per_ssb_params[0].beta_sss = 1.5f; // Moderate SSS power
    auto dyn_params = data.makeDynParams(config, helper.strm_handle);

    // Run comprehensive validation test
    helper.runSSBTest(dyn_params);
    auto result = helper.verifyOutputQuantitative(config, data.per_cell_params[0], data.per_ssb_params[0], true);

    std::cout << "=== Comprehensive Quantitative Validation ===" << std::endl;

    // Comprehensive assertions for quantitative correctness
    ASSERT_TRUE(result.passed) << "Comprehensive SSB test must pass all validation checks";

    // Signal count validation (based on 5G NR spec)
    EXPECT_EQ(result.pss_samples, 127) << "PSS should have exactly 127 samples";
    EXPECT_EQ(result.sss_samples, 127) << "SSS should have exactly 127 samples";
    EXPECT_GE(result.pbch_samples + result.dmrs_samples, 576) << "PBCH+DMRS should have 576 samples total";

    // Power level validation with tight tolerances
    float expected_pss_power = data.per_ssb_params[0].beta_pss * data.per_ssb_params[0].beta_pss; // 4.0
    float expected_sss_power = data.per_ssb_params[0].beta_sss * data.per_ssb_params[0].beta_sss; // 2.25

    EXPECT_NEAR(result.pss_power_avg, expected_pss_power, expected_pss_power * 0.1f)
        << "PSS power should match beta_pss^2";
    EXPECT_NEAR(result.sss_power_avg, expected_sss_power, expected_sss_power * 0.1f)
        << "SSS power should match beta_sss^2";

    // Cross-validation: Test with different parameters should give different results
    data.per_ssb_params[0].beta_pss = 1.0f; // Lower PSS power
    data.per_ssb_params[0].beta_sss = 3.0f; // Higher SSS power
    data.per_cell_params[0].NID     = 200;  // Different NID

    auto dyn_params2 = data.makeDynParams(config, helper.strm_handle);

    helper.resetOutputBuffer(config);
    helper.runSSBTest(dyn_params2);
    auto result2 = helper.verifyOutputQuantitative(config, data.per_cell_params[0], data.per_ssb_params[0], false);

    EXPECT_TRUE(result2.passed) << "Cross-validation test should also pass";
    EXPECT_NE(result.pss_power_avg, result2.pss_power_avg) << "Different beta_pss should give different PSS power";
    EXPECT_NE(result.sss_power_avg, result2.sss_power_avg) << "Different beta_sss should give different SSS power";

    std::cout << "Cross-validation: PSS power " << result.pss_power_avg << " -> " << result2.pss_power_avg << std::endl;
    std::cout << "Cross-validation: SSS power " << result.sss_power_avg << " -> " << result2.sss_power_avg << std::endl;
    std::cout << "=============================================" << std::endl;
}

// Test: Error path coverage for cuphySSBsKernelSelect - Polar encoder invalid argument
TEST(SSBTest, SSBSKernelSelect_PolarEncoderInvalidArg)
{
    // Line 610: return when polar_encoder::kernelSelectEncodeRateMatchMultiSSBs fails
    // Trigger by passing nullptr for the encoder launch config
    cuphySsbMapperLaunchCfg_t mapperCfg{}; // valid dummy mapper cfg
    cuphyStatus_t             status = cuphySSBsKernelSelect(nullptr, &mapperCfg, 1);
    EXPECT_EQ(status, CUPHY_STATUS_INVALID_ARGUMENT);
}

// Test: Error path coverage for cuphySSBsKernelSelect - Mapper invalid argument
TEST(SSBTest, SSBSKernelSelect_MapperInvalidArg)
{
    // Line 615: return when kernelSelectSsbMapper fails
    // Trigger by passing nullptr for the mapper launch config
    cuphyEncoderRateMatchMultiSSBLaunchCfg_t encdrCfg{}; // zero-init; function will populate
    cuphyStatus_t                            status = cuphySSBsKernelSelect(&encdrCfg, nullptr, 1);
    EXPECT_EQ(status, CUPHY_STATUS_INVALID_ARGUMENT);
}

// Test: Error path coverage for cuphyRunSsbMapper - Invalid arguments
TEST(SSBTest, CuphyRunSsbMapper_InvalidArgs)
{
    // Pass nullptrs so the early guard returns CUPHY_STATUS_INVALID_ARGUMENT
    cuphyStatus_t status = cuphyRunSsbMapper(
        /*d_x_tx*/ nullptr,
        /*d_tfSignal*/ nullptr,
        /*d_ssb_params*/ nullptr,
        /*d_per_cell_params*/ nullptr,
        /*d_pmw_params*/ nullptr,
        /*num_SSBs*/ 0,
        /*num_cells*/ 0,
        /*stream*/ nullptr,
        /*pSsbMapperCfg*/ nullptr);
    EXPECT_EQ(status, CUPHY_STATUS_INVALID_ARGUMENT);
}

// Test: Error path coverage for cuphyRunSsbMapper - Internal error branch
// Force the post-launch branch (internal error) in cuphyRunSsbMapper by providing
// valid non-null pointers but an invalid launch configuration (null kernel func)
TEST(SSBTest, CuphyRunSsbMapper_InternalErrorBranch)
{
    // Minimal device buffers just to satisfy non-null checks
    uint8_t* d_x_tx   = nullptr;
    __half2* d_tf_buf = nullptr;
    AssertCudaSuccess(cudaMalloc(&d_x_tx, 1));
    AssertCudaSuccess(cudaMalloc(&d_tf_buf, sizeof(__half2)));

    // Host array of device pointers (function only checks non-null)
    __half2* tf_array_host = d_tf_buf;

    // Minimal parameter structs (on host; not dereferenced by this function)
    cuphyPerSsBlockDynPrms_t ssb_params{};
    ssb_params.blockIndex = 0;
    cuphyPerCellSsbDynPrms_t cell_params{};
    cell_params.NID = 0;
    cuphyPmWOneLayer_t pmw{};

    // Invalid launch cfg: func remains nullptr so launch_kernel should fail
    cuphySsbMapperLaunchCfg_t mapperCfg{}; // zero-init

    // NOTE: d_ssb_params/d_per_cell_params/d_pmw_params are documented as device pointers.
    // This test intentionally passes host stack addresses because this error-path only
    // requires non-null arguments before launch fails on the invalid mapperCfg (func == nullptr).
    // If pointer-attribute validation is added in the future, this test may need device allocations.
    cuphyStatus_t status = cuphyRunSsbMapper(
        d_x_tx,
        &tf_array_host,
        &ssb_params,
        &cell_params,
        &pmw,
        1,
        1,
        /*stream*/ nullptr,
        &mapperCfg);

    EXPECT_EQ(status, CUPHY_STATUS_INTERNAL_ERROR);

    // Cleanup
    AssertCudaSuccess(cudaFree(d_x_tx));
    AssertCudaSuccess(cudaFree(d_tf_buf));
}

// =========================================================================
// Golden-reference sequence tests
//
// These tests close the NID-identity gap identified in the functional review:
// because PSS and SSS are constant-envelope (|x[n]|=1), power checks alone
// cannot distinguish between sequences generated for different NID values.
// The tests below extract each IQ sample from the GPU output and compare
// its REAL part (PSS/SSS are BPSK real-valued) against the CPU reference
// derived directly from the 3GPP TS 38.211 §7.4.2.2/§7.4.2.3 formulas.
// =========================================================================

// Helper: extract 127 PSS real values from the full GPU output tensor.
// PSS lives at symbol t0, subcarriers [f0+56 .. f0+182].
static std::vector<float> extract_pss_from_output(const std::vector<__half2>& buf,
                                                   uint16_t nF, uint8_t t0, uint16_t f0)
{
    std::vector<float> out(127);
    for(int n = 0; n < 127; ++n)
    {
        size_t idx = static_cast<size_t>(t0) * nF + (f0 + 56 + n);
        out[n]     = static_cast<float>(buf[idx].x);
    }
    return out;
}

// Helper: extract 127 SSS real values from the full GPU output tensor.
// SSS lives at symbol t0+2, subcarriers [f0+56 .. f0+182].
static std::vector<float> extract_sss_from_output(const std::vector<__half2>& buf,
                                                   uint16_t nF, uint8_t t0, uint16_t f0)
{
    std::vector<float> out(127);
    for(int n = 0; n < 127; ++n)
    {
        size_t idx = static_cast<size_t>(t0 + 2) * nF + (f0 + 56 + n);
        out[n]     = static_cast<float>(buf[idx].x);
    }
    return out;
}

// Test: PSS golden-sequence validation for all three NID2 values (0, 1, 2).
// For each NID, the GPU-generated PSS real samples must match the host
// reference to within the FP16 quantisation tolerance (±0.01 for beta=1).
TEST_F(SSBTestFixture, PssSequenceExactValidation)
{
    setupConfig();
    // Test NID values that cover all three NID2 cases: NID%3 ∈ {0,1,2}
    const std::vector<uint16_t> test_nids = {0 /*NID2=0*/, 1 /*NID2=1*/, 2 /*NID2=2*/,
                                             123 /*NID2=0*/, 124 /*NID2=1*/, 125 /*NID2=2*/};
    const float                 beta_pss  = 1.0f;
    // FP16 has ~3 decimal digits of precision; allow ±0.01 for a unit-amplitude signal.
    const float tolerance = 0.01f;

    for(uint16_t nid : test_nids)
    {
        SCOPED_TRACE("NID=" + std::to_string(nid));
        helper.resetOutputBuffer(config);
        auto data = makeBasicCase(nid, 0, 0xABCDEF);
        data.per_ssb_params[0].beta_pss = beta_pss;
        data.per_ssb_params[0].beta_sss = beta_pss;

        auto dyn_params = data.makeDynParams(config, helper.strm_handle);
        helper.runSSBTest(dyn_params);

        // Copy full output to host
        const size_t         output_size = static_cast<size_t>(config.nF) * config.nSym * config.nPorts;
        std::vector<__half2> host_buf(output_size);
        AssertCudaSuccess(cudaMemcpyAsync(host_buf.data(), helper.d_output.get(),
                                         output_size * sizeof(__half2),
                                         cudaMemcpyDeviceToHost, helper.strm_handle));
        AssertCudaSuccess(cudaStreamSynchronize(helper.strm_handle));

        const uint16_t nF = config.nF;
        const uint8_t  t0 = data.per_ssb_params[0].t0;
        const uint16_t f0 = data.per_ssb_params[0].f0;

        auto gpu_pss = extract_pss_from_output(host_buf, nF, t0, f0);
        auto ref_pss = compute_pss_reference(nid, beta_pss);

        ASSERT_EQ(gpu_pss.size(), 127u);
        int mismatches = 0;
        for(int n = 0; n < 127; ++n)
        {
            if(std::fabs(gpu_pss[n] - ref_pss[n]) > tolerance)
            {
                ++mismatches;
                ADD_FAILURE() << "PSS[" << n << "] NID=" << nid
                              << ": gpu=" << gpu_pss[n]
                              << " ref=" << ref_pss[n]
                              << " diff=" << std::fabs(gpu_pss[n] - ref_pss[n]);
            }
        }
        EXPECT_EQ(mismatches, 0) << "PSS sequence mismatch for NID=" << nid
                                 << " (NID2=" << (nid % 3) << ")";
    }
}

// Test: SSS golden-sequence validation for a representative set of NID values.
// Validates the two-m-sequence product formula with NID-specific m0/m1 offsets.
TEST_F(SSBTestFixture, SssSequenceExactValidation)
{
    setupConfig();
    // Cover different (NID1, NID2) combos: small/mid/large NID1, all three NID2
    const std::vector<uint16_t> test_nids = {
        0,    //  NID1=0,   NID2=0, m0=0,  m1=0
        1,    //  NID1=0,   NID2=1, m0=5,  m1=0
        2,    //  NID1=0,   NID2=2, m0=10, m1=0
        336,  //  NID1=112, NID2=0, m0=15, m1=0
        503,  //  NID1=167, NID2=2, m0=25, m1=55
        1007  //  NID1=335, NID2=2, m0=55, m1=111 (near max)
    };
    const float beta_sss  = 1.0f;
    const float tolerance = 0.01f;

    for(uint16_t nid : test_nids)
    {
        SCOPED_TRACE("NID=" + std::to_string(nid));
        helper.resetOutputBuffer(config);
        auto data = makeBasicCase(nid, 0, 0x123456);
        data.per_ssb_params[0].beta_pss = beta_sss;
        data.per_ssb_params[0].beta_sss = beta_sss;

        auto dyn_params = data.makeDynParams(config, helper.strm_handle);
        helper.runSSBTest(dyn_params);

        const size_t         output_size = static_cast<size_t>(config.nF) * config.nSym * config.nPorts;
        std::vector<__half2> host_buf(output_size);
        AssertCudaSuccess(cudaMemcpyAsync(host_buf.data(), helper.d_output.get(),
                                         output_size * sizeof(__half2),
                                         cudaMemcpyDeviceToHost, helper.strm_handle));
        AssertCudaSuccess(cudaStreamSynchronize(helper.strm_handle));

        const uint16_t nF = config.nF;
        const uint8_t  t0 = data.per_ssb_params[0].t0;
        const uint16_t f0 = data.per_ssb_params[0].f0;

        auto gpu_sss = extract_sss_from_output(host_buf, nF, t0, f0);
        auto ref_sss = compute_sss_reference(nid, beta_sss);

        ASSERT_EQ(gpu_sss.size(), 127u);
        int mismatches = 0;
        for(int n = 0; n < 127; ++n)
        {
            if(std::fabs(gpu_sss[n] - ref_sss[n]) > tolerance)
            {
                ++mismatches;
                ADD_FAILURE() << "SSS[" << n << "] NID=" << nid
                              << ": gpu=" << gpu_sss[n]
                              << " ref=" << ref_sss[n]
                              << " diff=" << std::fabs(gpu_sss[n] - ref_sss[n]);
            }
        }
        EXPECT_EQ(mismatches, 0) << "SSS sequence mismatch for NID=" << nid
                                 << " (NID1=" << (nid / 3) << ", NID2=" << (nid % 3) << ")";
    }
}

// Main function for the test executable
int main(int argc, char** argv)
{
    // Initialize Google Test
    ::testing::InitGoogleTest(&argc, argv);

    // Print test information
    std::cout << "=== Enhanced SSB Test Suite ===" << std::endl;
    std::cout << "Testing SSB (Synchronization Signal Block) functionality" << std::endl;
    std::cout << "Target: src/cuphy/ss/ss.cu" << std::endl;
    std::cout << "Features:" << std::endl;
    std::cout << "  - Quantitative correctness validation (similar to examples/testSS.cpp)" << std::endl;
    std::cout << "  - Signal power and constellation validation" << std::endl;
    std::cout << "  - PSS/SSS/PBCH/DMRS pattern verification" << std::endl;
    std::cout << "  - Cross-validation between parameter sets" << std::endl;
    std::cout << "  - Manual parameter initialization without HDF5 dependency" << std::endl;
    std::cout << "Coverage goals: compute_qam_index_v2, compute_dmrs_index_v2" << std::endl;
    std::cout << "NOTE: DMRS coverage may be 0% if thread range 576-719 doesn't execute" << std::endl;
    std::cout << "      or DMRS REs are classified differently than expected." << std::endl;
    std::cout << "================================" << std::endl;

    // Run all tests
    int result = RUN_ALL_TESTS();

    // Print summary
    std::cout << "=== Test Summary ===" << std::endl;
    if(result == 0)
    {
        std::cout << "All SSB tests passed!" << std::endl;
    }
    else
    {
        std::cout << "Some SSB tests failed. Check output above." << std::endl;
    }
    std::cout << "===================" << std::endl;

    return result;
}

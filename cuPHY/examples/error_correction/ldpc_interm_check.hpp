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

// Infrastructure for --llr_check / --llr_diff: per-iteration APP comparison against the MATLAB TV.
// Mirrors the HDF5 /params compound dataset written by genTV_ldpc.m.
// If genTV_ldpc.m adds or renames a field, update LdpcTvParams and read().
//
// LIMITATIONS (read before trusting a PASS):
//  - Validates per-iteration APP "numeric fidelity" vs a reference decoder.
//  - Compares only the CORE columns (Kb+4)*Z (systematic + 4 core parity).
//    Extension parity is never checked (it is not updated per iteration).
//  - STORE-ELISION BLIND SPOT: some decoders elide core-column APP stores in
//    their TIMED kernel; llr_check/llr_diff run the dumpApp twin (which stores
//    everything), so they CANNOT validate those decoders' product store
//    behaviour -- those need a BLER + low-SNR waterfall check.

#pragma once
#include <bitset>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>
#include <cuda_fp16.h>
#include "cuphy.h"
#include "cuphy.hpp"
#include "hdf5hpp.hpp"
#include <fmt/core.h>
#include "cuphy_hdf5.hpp"

struct LdpcTvParams
{
    // config params (match genTV_ldpc.m field order)
    int32_t rng_seed{};
    float   dec_param_fp16_clamp{};
    float   dec_param_fp8_clamp{};
    int32_t dec_param_fp8_c2v_simple{};
    int32_t nRB{};
    int32_t nLayers{};
    int32_t mcsIndex{};
    int32_t mcsTable{};
    float   SNR_dB{};
    int32_t maxItr{};
    // derived params
    int32_t BGN{};
    int32_t Zc{};
    int32_t nV_parity{};
    int32_t i_LS{};
    int32_t A{};
    int32_t n_tb_crc{};
    int32_t n_cb_crc{};
    int32_t C{};
    int32_t K{};
    int32_t K_prime{};
    int32_t F{};
    int32_t modOrd{};
    int32_t G{};

    // LLR clamp for the decoder. Only fp16/fp32 are wired (both use the fp16 clamp);
    // the llr_type arg is a hook for future fp8.
    [[nodiscard]] float clamp_for(cuphyDataType_t llr_type) const
    {
        (void)llr_type;
        return dec_param_fp16_clamp;
    }

    void read(hdf5hpp::hdf5_file& f)
    {
        // Fields are matched by name, so struct layout and /params order are independent.
        cuphy::cuphyHDF5_struct s = cuphy::get_HDF5_struct(f, "params");
        rng_seed                 = s.get_value_as<int32_t>("rng_seed");
        dec_param_fp16_clamp     = s.get_value_as<float>("dec_param_fp16_clamp");
        dec_param_fp8_clamp      = s.get_value_as<float>("dec_param_fp8_clamp");
        dec_param_fp8_c2v_simple = s.get_value_as<int32_t>("dec_param_fp8_c2v_simple");
        nRB                      = s.get_value_as<int32_t>("nRB");
        nLayers                  = s.get_value_as<int32_t>("nLayers");
        mcsIndex                 = s.get_value_as<int32_t>("mcsIndex");
        mcsTable                 = s.get_value_as<int32_t>("mcsTable");
        SNR_dB                   = s.get_value_as<float>("SNR_dB");
        maxItr                   = s.get_value_as<int32_t>("maxItr");
        BGN                      = s.get_value_as<int32_t>("BGN");
        Zc                       = s.get_value_as<int32_t>("Zc");
        nV_parity                = s.get_value_as<int32_t>("nV_parity");
        i_LS                     = s.get_value_as<int32_t>("i_LS");
        A                        = s.get_value_as<int32_t>("A");
        n_tb_crc                 = s.get_value_as<int32_t>("n_tb_crc");
        n_cb_crc                 = s.get_value_as<int32_t>("n_cb_crc");
        C                        = s.get_value_as<int32_t>("C");
        K                        = s.get_value_as<int32_t>("K");
        K_prime                  = s.get_value_as<int32_t>("K_prime");
        F                        = s.get_value_as<int32_t>("F");
        modOrd                   = s.get_value_as<int32_t>("modOrd");
        G                        = s.get_value_as<int32_t>("G");
    }

    void print() const
    {
        fmt::print("LdpcTvParams Config: rng_seed={}  fp16_clamp={:.1f}  fp8_clamp={:.1f}  fp8_c2v_simple={}  nRB={}  nLayers={}  mcsIndex={}  mcsTable={}  SNR={:.1f} dB  maxItr={}\n",
                   rng_seed, dec_param_fp16_clamp, dec_param_fp8_clamp, dec_param_fp8_c2v_simple,
                   nRB, nLayers, mcsIndex, mcsTable, SNR_dB, maxItr);
        fmt::print("Derived: BGN={}  Zc={}  nV_parity={}  i_LS={}  C={}  modOrd={}\n",
                   BGN, Zc, nV_parity, i_LS, C, modOrd);
        fmt::print("         A={}  n_tb_crc={}  n_cb_crc={}\n",
                   A, n_tb_crc, n_cb_crc);
        fmt::print("         K={}  K_prime={}  F={}  G={}\n",
                   K, K_prime, F, G);
    }

    // Returns a bitmask: bit N set = check N failed. 1 bit per check, max 32.
    [[nodiscard]] uint32_t sanity_check() const
    {
        if(BGN != 1 && BGN != 2) { return (1u << 31); }
        uint32_t      err           = 0;
        int           cnt           = 0;
        const int32_t Kb            = (BGN == 1) ? 22 : 10;
        const int32_t max_nV_parity = (BGN == 1) ? 46 : 42;
        const int32_t QmNl          = modOrd * nLayers;
        bool ok;
        ok = (K == Zc * Kb);
        err |= (ok || cnt >= 32) ? 0u : (1u << cnt);
        ++cnt;
        ok = (nV_parity >= 4 && nV_parity <= max_nV_parity);
        err |= (ok || cnt >= 32) ? 0u : (1u << cnt);
        ++cnt;
        ok = (K_prime == K - F);
        err |= (ok || cnt >= 32) ? 0u : (1u << cnt);
        ++cnt;
        ok = (G > 0);
        err |= (ok || cnt >= 32) ? 0u : (1u << cnt);
        ++cnt;
        ok = (maxItr >= 1 && maxItr <= 20);
        err |= (ok || cnt >= 32) ? 0u : (1u << cnt);
        ++cnt;
        ok = (C > 0);
        err |= (ok || cnt >= 32) ? 0u : (1u << cnt);
        ++cnt;
        ok = (QmNl > 0);
        err |= (ok || cnt >= 32) ? 0u : (1u << cnt);
        ++cnt;
        ok = (QmNl > 0 && G % QmNl == 0);
        err |= (ok || cnt >= 32) ? 0u : (1u << cnt);
        ++cnt;
        ok = (n_tb_crc == 16 || n_tb_crc == 24)         &&
             ((A > 3824) == (n_tb_crc == 24))            &&
             (n_cb_crc == 0 || n_cb_crc == 24)           &&
             ((C == 1) == (n_cb_crc == 0))               &&
             (A + n_tb_crc + C * n_cb_crc == C * K_prime);
        err |= (ok || cnt >= 32) ? 0u : (1u << cnt);
        ++cnt;
        return err;
    }
};  // end struct LdpcTvParams

// Decode once (one algo + llr_type) on a device LLR input, capturing its APP history.
//   Input : llr_desc/llr_addr -- the on-device input LLRs (layout descriptor + device
//           pointer). Both the generator and the TV loader produce this pair, and
//           geometry (BGN, Zc, ...) is passed as scalars, so random-codeword
//           inputs work too -- not just TV files.
//   Output: runs with CUPHY_LDPC_DECODE_DUMP_INTERM (ET disabled -> all NUM_ITER iterations run);
//           per-iteration APP -> h_APP_out (fp32, [C, NUM_ITER, NUM_VAR]); h_dec_out (if != null) the
//           packed decoded bits, for BER.
void capture_app_device(cuphy::context&           ctx,
                        const cuphy::tensor_desc& llr_desc,
                        void*                     llr_addr,
                        int                       BGN,
                        int                       Zc,
                        int                       nV_parity,
                        int                       K,
                        int                       C,
                        int                       maxItr,
                        float                     clamp,
                        cuphyDataType_t           llr_type,
                        int                       algo,
                        int                       max_iter,
                        std::vector<float>&       h_APP_out,
                        std::vector<uint8_t>*     h_dec_out = nullptr);

// Default per-iteration APP pass gates -- the SINGLE source of truth. Both
// cuphy_ex_ldpc and cuphy_ex_ldpc_rm use these as their --p99err_rms_threshold /
// --snr_threshold defaults, and check_APP prints them, so the Python sweeps read
// the gate back from the binary output instead of keeping their own copies.
constexpr float LLR_P99ERR_RMS_THRESHOLD_DEFAULT = 0.07f;  // pctile99|ref-dut| / rms_ref
constexpr float LLR_SNR_THRESHOLD_DEFAULT       = 30.0f;  // dB

// True if a captured APP buffer is entirely zero -> the decoder has no dump path
// compiled (used to report N/A instead of comparing against zeros).
[[nodiscard]] bool APP_history_is_empty(const std::vector<float>& h_APP);

// Per-iteration APP comparison of two host buffers (ref vs dut). Pass if every
// (cw, iter) row passes all three gates: p99err_rms <= p99err_rms_threshold,
// SNR(dB) >= snr_threshold, and no sign flips in the last 3 iterations. Compares
// only the first N_var_checked columns.
[[nodiscard]] bool check_APP(const float* h_APP_ref, const float* h_APP_dut,
               int C, int NUM_ITER, int NUM_VAR, int N_var_checked,
               float p99err_rms_threshold, float snr_threshold);

// Run decoder with interm_results dump, write ldpc_dut.h5, then compare against
// MATLAB /APP_history reference loaded from the TV.
[[nodiscard]] bool run_llr_check(const std::string& tv_path, cuphyDataType_t llr_type, int algo,
                   int max_iter,
                   float p99err_rms_threshold, float snr_threshold);

// DUT-vs-DUT: decode the same TV with algoA and algoB; per-algo dumps go to
// ldpc_dut_a<algo>.h5; compares the two DUT APP_history dumps in memory.
[[nodiscard]] bool run_llr_diff(const std::string& tv_path, cuphyDataType_t llr_type,
                  int algoA, int algoB,
                  int max_iter,
                  float p99err_rms_threshold, float snr_threshold);

// Compare REF vs DUT HDF5 files (REF from genTV_ldpc.m, DUT from run_llr_check).
// Only CUPHY_R_16F and CUPHY_R_32F are accepted; check_llr_type rejects FP8.
[[nodiscard]] bool check_interm_results(const std::string& tv_path, const std::string& dut_path,
                          cuphyDataType_t llr_type,
                          int max_iter,
                          float p99err_rms_threshold, float snr_threshold);

// Write DUT intermediate results to HDF5 for plotting / MATLAB comparison. Datasets:
//  - /inputLLR [C, N]
//  - /APP_history [C, maxItr, N]
//  - /llr_type: scalar string ("fp16" / "fp32")
//  - /BGN /Zc /Kb /mb /F: structural scalars (written only if BGN>0)
void write_dut_h5(
    const char*     filename,
    const float*    llr_init,   // [C, N]
    int             C,
    int             N,
    const float*    app_hist,   // [C, maxItr, N]
    int             maxItr,
    cuphyDataType_t llr_type,
    int             BGN = 0,   // structural metadata for plotter regions (0 = unknown)
    int             Zc  = 0,
    int             Kb  = 0,
    int             mb  = 0,
    int             F   = 0);

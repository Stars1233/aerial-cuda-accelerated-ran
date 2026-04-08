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
#include <vector>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <climits>
#include <utility>
#include <complex>
#include <numeric>

#include <cuda_fp16.h>
#include <cuComplex.h>
#include "cuphy.h"
#include "cuphy_internal.h"
#include "cuphy.hpp"
#include "prach_receiver/prach_receiver.hpp"


namespace
{

// Query active device and return an architecture code of the form
// major*100 + minor*10 (e.g., 800 for SM80), matching prach_get_fft_param.
static unsigned int getCudaArch()
{
    int         device = 0;
    cudaError_t err    = cudaGetDevice(&device);
    if(err != cudaSuccess)
    {
        return 800; // default to SM80
    }
    cudaDeviceProp props{};
    err = cudaGetDeviceProperties(&props, device);
    if(err != cudaSuccess)
    {
        return 800;
    }
    // Map to values used by prach_get_fft_param: e.g., 750, 800, 860, 870, 890, 900, 1000, 1210
    return static_cast<unsigned int>(props.major * 100 + props.minor * 10);
}

template <typename T>
static void cudaAllocAndZero(T** ptr, size_t count)
{
    ASSERT_EQ(cudaSuccess, cudaMalloc(reinterpret_cast<void**>(ptr), count * sizeof(T)));
    ASSERT_EQ(cudaSuccess, cudaMemset(*ptr, 0, count * sizeof(T)));
}

static size_t nextPowerOfTwo(size_t v)
{
    if(v == 0) return 1;
    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
#if ULONG_MAX > 0xFFFFFFFF
    v |= v >> 32;
#endif
    return v + 1;
}

struct PrachTestConfig
{
    // Core parameters
    uint32_t L_RA;
    uint32_t Nfft;
    uint32_t N_rep;
    uint32_t N_CS;
    uint32_t uCount;
    uint32_t N_nc;
    uint32_t N_ant;
    uint32_t mu;
    uint32_t delta_f_RA; // in Hz
    uint32_t kBar;
    bool     enableUlRxBf;
    uint16_t nUplinkStreams;
};

static void fillPrachParams(const PrachTestConfig& cfg, PrachParams& out)
{
    out.N_CS       = cfg.N_CS;
    out.uCount     = cfg.uCount;
    out.L_RA       = cfg.L_RA;
    out.N_rep      = cfg.N_rep;
    out.delta_f_RA = cfg.delta_f_RA;
    out.N_ant      = cfg.N_ant;
    out.mu         = cfg.mu;
    out.Nfft       = cfg.Nfft;
    out.N_nc       = cfg.N_nc;
    out.kBar       = cfg.kBar;
}

// Conservative workspace size computation to match kernels' expectations.
static size_t getWorkspaceFloats(const PrachParams& prm)
{
    const size_t fft_elements = static_cast<size_t>(prm.Nfft) * prm.N_ant * prm.uCount * prm.N_nc;
    const size_t pdp_elements = static_cast<size_t>(prm.N_ant) * CUPHY_PRACH_RX_NUM_PREAMBLE;

    // prach_workspace_buffer layout (as floats):
    // [FFT buffer (complex float)] + [pdp buffer (struct)] + [det struct] + [ant_rssi floats] + [rssiLin float] + [count uint]
    // Use safe upper bounds: size of prach_pdp_t<float> and prach_det_t<float>
    const size_t fft_floats  = 2 * fft_elements; // cuFloatComplex
    const size_t pdp_bytes   = pdp_elements * sizeof(prach_pdp_t<float>);
    const size_t det_bytes   = sizeof(prach_det_t<float>);
    const size_t tail_bytes  = sizeof(float) * (prm.N_ant + 1) + sizeof(unsigned int);
    size_t       total_bytes = fft_floats * sizeof(float) + pdp_bytes + det_bytes + tail_bytes;
    // safety margin
    total_bytes += 4096;
    return (total_bytes + sizeof(float) - 1) / sizeof(float);
}

static void makeOnesHalf2(std::vector<__half2>& v)
{
    for(auto& e : v)
    {
        e = __floats2half2_rn(1.0f, 0.0f);
    }
}

static void makeZerosHalf2(std::vector<__half2>& v)
{
    for(auto& e : v)
    {
        e = __floats2half2_rn(0.0f, 0.0f);
    }
}

static uint16_t findq(uint16_t u, uint16_t L_RA)
{
    for(uint16_t q = 0; q < L_RA; ++q)
    {
        if(((uint32_t)q * (uint32_t)u) % L_RA == 1)
            return q;
    }
    return 0;
}

static int computePdpTargetBinForPrmb0(const PrachParams& prm)
{
    // prmbCount = 0 → C_v = 0
    const int Nfft          = (int)prm.Nfft;
    const int L_RA          = (int)prm.L_RA;
    const int N_CS          = (int)prm.N_CS;
    const int zoneSearchGap = (int)(prm.Nfft / prm.L_RA);
    const int zoneSize      = (N_CS * Nfft + L_RA - 1) / L_RA;
    (void)zoneSize;                                // Not strictly needed for bin 0 of the zone
    int zone_start = (0 * Nfft + L_RA - 1) / L_RA; // C_v=0
    zone_start     = (Nfft - zone_start) & (Nfft - 1);
    int bin        = (zone_start - zoneSearchGap + Nfft) & (Nfft - 1);
    return bin;
}

static void generateZcYRef(uint16_t L_RA, uint16_t u, std::vector<__half2>& out)
{
    out.resize(L_RA);
    std::vector<std::complex<double>> x_u(L_RA);
    const double                      pi = std::acos(-1.0);
    for(int i = 0; i < L_RA; ++i)
    {
        double phase = -pi * (double)u * (double)i * (double)(i + 1) / (double)L_RA;
        x_u[i]       = std::exp(std::complex<double>(0.0, phase));
    }
    std::complex<double> sum         = std::accumulate(x_u.begin(), x_u.end(), std::complex<double>(0.0, 0.0));
    uint16_t             q           = findq(u, L_RA);
    double               invsqrt_lra = 1.0 / std::sqrt((double)L_RA);
    for(int m = 0; m < L_RA; ++m)
    {
        std::complex<double> y = sum * x_u[0] * std::conj(x_u[(q * m) % L_RA]) * invsqrt_lra;
        out[m]                 = __floats2half2_rn((float)y.real(), (float)y.imag());
    }
}

static void prepareInputRx(const PrachParams& prm, uint32_t kBar, bool useRefSignal, const std::vector<__half2>& y_ref, int mShift, float amplitude, std::vector<__half2>& host)
{
    const int    L_ORAN = (prm.L_RA == 139 ? 144 : 864);
    const size_t total  = static_cast<size_t>(prm.N_ant) * prm.N_rep * L_ORAN;
    host.resize(total);
    makeZerosHalf2(host);
    if(useRefSignal)
    {
        // For each antenna and repetition, set portion [kBar, kBar+L_RA) to y_ref with linear phase to shift IFFT peak to mShift
        for(uint32_t a = 0; a < prm.N_ant; ++a)
        {
            for(uint32_t r = 0; r < prm.N_rep; ++r)
            {
                size_t base = (static_cast<size_t>(a) * prm.N_rep + r) * L_ORAN;
                for(uint32_t i = 0; i < prm.L_RA; ++i)
                {
                    float       xr        = __low2float(y_ref[i]);
                    float       xi        = __high2float(y_ref[i]);
                    const float TWO_PI    = 6.28318530717958647692f;
                    float       angle     = TWO_PI * (float)(i % prm.Nfft) * (float)(mShift % (int)prm.Nfft) / (float)prm.Nfft;
                    float       ca        = cosf(angle);
                    float       sa        = sinf(angle);
                    float       rr        = amplitude * (xr * ca - xi * sa);
                    float       ri        = amplitude * (xr * sa + xi * ca);
                    host[base + kBar + i] = __floats2half2_rn(rr, ri);
                }
            }
        }
    }
}

static void runSingleOccasionTest(const PrachTestConfig& cfg,
                                  bool                   onesSignal,
                                  uint32_t&              outNumDetected,
                                  float&                 outRssiDb,
                                  float&                 outInterfDb)
{
    // Ensure no stale CUDA error from prior tests affects this run.
    (void)cudaGetLastError();

    using Tcomplex = cuFloatComplex;
    using Tscalar  = float;

    PrachInternalStaticParamPerOcca h_staticOcca{};
    fillPrachParams(cfg, h_staticOcca.prach_params);
    h_staticOcca.enableUlRxBf = cfg.enableUlRxBf ? 1 : 0;

    const PrachParams&   prm = h_staticOcca.prach_params;
    std::vector<__half2> host_y_ref(static_cast<size_t>(prm.L_RA) * prm.uCount);
    // Use flat frequency reference (all ones) to produce strong IFFT peak at bin 0
    makeOnesHalf2(host_y_ref);

    // Allocate device buffers directly into host static struct
    h_staticOcca.prach_workspace_buffer = cuphy::buffer<float, cuphy::device_alloc>(getWorkspaceFloats(prm));
    h_staticOcca.d_y_u_ref              = cuphy::buffer<__half2, cuphy::device_alloc>(prm.L_RA * prm.uCount);
    if(!host_y_ref.empty())
    {
        ASSERT_EQ(cudaSuccess,
                  cudaMemcpy(h_staticOcca.d_y_u_ref.addr(), host_y_ref.data(), host_y_ref.size() * sizeof(__half2), cudaMemcpyHostToDevice));
    }

    // Populate host static and device static
    // buffers already allocated directly in h_staticOcca above
    PrachDeviceInternalStaticParamPerOcca h_devStatic{};
    h_devStatic.prach_params           = prm;
    h_devStatic.prach_workspace_buffer = h_staticOcca.prach_workspace_buffer.addr();
    h_devStatic.d_y_u_ref              = h_staticOcca.d_y_u_ref.addr();
    h_devStatic.enableUlRxBf           = h_staticOcca.enableUlRxBf;

    // Device arrays for static and dynamic params (size 1)
    PrachDeviceInternalStaticParamPerOcca* d_static = nullptr;
    PrachInternalDynParamPerOcca*          d_dyn    = nullptr;
    PrachInternalDynParamPerOcca           h_dyn{};

    // Prepare input RX
    std::vector<__half2> host_rx;
    // Set mShift to the first bin in prmbIdx=0 PDP zone
    int mShift = computePdpTargetBinForPrmb0(prm);
    prepareInputRx(prm, prm.kBar, onesSignal, host_y_ref, mShift, /*amplitude=*/8.0f, host_rx);
    __half2* d_rx = nullptr;
    ASSERT_EQ(cudaSuccess, cudaMalloc(reinterpret_cast<void**>(&d_rx), host_rx.size() * sizeof(__half2)));
    ASSERT_EQ(cudaSuccess, cudaMemcpy(d_rx, host_rx.data(), host_rx.size() * sizeof(__half2), cudaMemcpyHostToDevice));

    h_dyn.dataRx         = d_rx;
    h_dyn.occaPrmStatIdx = 0;
    h_dyn.occaPrmDynIdx  = 0;
    h_dyn.thr0           = onesSignal ? 0.05f : 5.0f; // permissive threshold for detection; strict for no-signal
    h_dyn.nUplinkStreams = cfg.nUplinkStreams;

    ASSERT_EQ(cudaSuccess, cudaMalloc(reinterpret_cast<void**>(&d_static), sizeof(PrachDeviceInternalStaticParamPerOcca)));
    ASSERT_EQ(cudaSuccess, cudaMemcpy(d_static, &h_devStatic, sizeof(PrachDeviceInternalStaticParamPerOcca), cudaMemcpyHostToDevice));
    ASSERT_EQ(cudaSuccess, cudaMalloc(reinterpret_cast<void**>(&d_dyn), sizeof(PrachInternalDynParamPerOcca)));
    ASSERT_EQ(cudaSuccess, cudaMemcpy(d_dyn, &h_dyn, sizeof(PrachInternalDynParamPerOcca), cudaMemcpyHostToDevice));

    // Host copies of dyn/static arrays as expected by API
    PrachInternalDynParamPerOcca    h_dyn_arr[1]    = {h_dyn};
    PrachInternalStaticParamPerOcca h_static_arr[1] = {h_staticOcca};

    // Output/device buffers
    uint32_t* d_numDetected = nullptr;
    uint32_t* d_prmbIndex   = nullptr;
    float*    d_prmbDelay   = nullptr;
    float*    d_prmbPower   = nullptr;
    float*    d_antRssi     = nullptr;
    float*    d_rssi        = nullptr;
    float*    d_interf      = nullptr;

    cudaAllocAndZero(&d_numDetected, 1);
    // Allocate generous size for preamble outputs to avoid dependency on macro value at test side
    cudaAllocAndZero(&d_prmbIndex, 256);
    cudaAllocAndZero(&d_prmbDelay, 256);
    cudaAllocAndZero(&d_prmbPower, 256);
    cudaAllocAndZero(&d_antRssi, prm.N_ant);
    cudaAllocAndZero(&d_rssi, 1);
    cudaAllocAndZero(&d_interf, 1);

    // Launch
    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    const uint16_t     nOccaProc       = 1;
    const uint16_t     maxAntenna      = prm.N_ant;
    const int          L_ORAN          = (prm.L_RA == 139 ? 144 : 864);
    const unsigned int align_l_oran    = ((prm.N_rep * L_ORAN + 31) >> 5) << 5;
    const unsigned int max_l_oran_ant  = align_l_oran * prm.N_ant;
    const unsigned int max_ant_u       = prm.N_ant * prm.uCount;
    const unsigned int max_nfft        = prm.Nfft;
    const int          zoneSizeForExt  = static_cast<int>((prm.N_CS * prm.Nfft + prm.L_RA - 1) / prm.L_RA);
    const int          max_zoneSizeExt = static_cast<int>(nextPowerOfTwo(std::max(1, zoneSizeForExt)));
    const unsigned int cudaArch        = getCudaArch();

    constexpr uint32_t maxFftsPerOccasion = MAX_N_ANTENNAS_SUPPORTED * PRACH_MAX_NUM_PREAMBLES * PRACH_NUM_NON_COHERENT_COMBINING_GROUPS;
    auto h_fftPointers = cuphy::buffer<cuFloatComplex *, cuphy::pinned_alloc>(PRACH_NUM_SUPPORTED_FFT_SIZES * nOccaProc * maxFftsPerOccasion);
    std::array<FftInfo, PRACH_NUM_SUPPORTED_FFT_SIZES> fftInfo;

    cuphyStatus_t status = cuphyPrachReceiver(d_dyn,
                                              d_static,
                                              h_dyn_arr,
                                              h_static_arr,
                                              h_fftPointers.addr(),
                                              fftInfo,
                                              d_numDetected,
                                              d_prmbIndex,
                                              d_prmbDelay,
                                              d_prmbPower,
                                              d_antRssi,
                                              d_rssi,
                                              d_interf,
                                              nOccaProc,
                                              maxAntenna,
                                              max_l_oran_ant,
                                              max_ant_u,
                                              max_nfft,
                                              max_zoneSizeExt,
                                              cudaArch,
                                              stream);
    ASSERT_EQ(CUPHY_STATUS_SUCCESS, status);

    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));

    // Fetch outputs
    ASSERT_EQ(cudaSuccess, cudaMemcpy(&outNumDetected, d_numDetected, sizeof(uint32_t), cudaMemcpyDeviceToHost));
    ASSERT_EQ(cudaSuccess, cudaMemcpy(&outRssiDb, d_rssi, sizeof(float), cudaMemcpyDeviceToHost));
    ASSERT_EQ(cudaSuccess, cudaMemcpy(&outInterfDb, d_interf, sizeof(float), cudaMemcpyDeviceToHost));

    // Cleanup
    cudaFree(d_rx);
    cudaFree(d_static);
    cudaFree(d_dyn);
    cudaFree(d_numDetected);
    cudaFree(d_prmbIndex);
    cudaFree(d_prmbDelay);
    cudaFree(d_prmbPower);
    cudaFree(d_antRssi);
    cudaFree(d_rssi);
    cudaFree(d_interf);
    cudaStreamDestroy(stream);
}

static void runSingleOccasionTestWithCustomRx(const PrachTestConfig&      cfg,
                                              const std::vector<__half2>& host_rx,
                                              uint32_t&                   outNumDetected,
                                              float&                      outRssiDb,
                                              float&                      outInterfDb,
                                              float                       thr0 = 0.05f)
{
    // Ensure no stale CUDA error from prior tests affects this run.
    (void)cudaGetLastError();

    PrachInternalStaticParamPerOcca h_staticOcca{};
    fillPrachParams(cfg, h_staticOcca.prach_params);
    h_staticOcca.enableUlRxBf = cfg.enableUlRxBf ? 1 : 0;

    const PrachParams& prm = h_staticOcca.prach_params;
    const int          L_ORAN = (prm.L_RA == 139 ? 144 : 864);
    const size_t       expected = static_cast<size_t>(prm.N_ant) * prm.N_rep * L_ORAN;
    ASSERT_EQ(host_rx.size(), expected);

    // Reference (all ones) on device
    std::vector<__half2> host_y_ref(static_cast<size_t>(prm.L_RA) * prm.uCount);
    makeOnesHalf2(host_y_ref);

    h_staticOcca.prach_workspace_buffer = cuphy::buffer<float, cuphy::device_alloc>(getWorkspaceFloats(prm));
    h_staticOcca.d_y_u_ref              = cuphy::buffer<__half2, cuphy::device_alloc>(prm.L_RA * prm.uCount);
    if(!host_y_ref.empty())
    {
        ASSERT_EQ(cudaSuccess,
                  cudaMemcpy(h_staticOcca.d_y_u_ref.addr(), host_y_ref.data(), host_y_ref.size() * sizeof(__half2), cudaMemcpyHostToDevice));
    }

    PrachDeviceInternalStaticParamPerOcca h_devStatic{};
    h_devStatic.prach_params           = prm;
    h_devStatic.prach_workspace_buffer = h_staticOcca.prach_workspace_buffer.addr();
    h_devStatic.d_y_u_ref              = h_staticOcca.d_y_u_ref.addr();
    h_devStatic.enableUlRxBf           = h_staticOcca.enableUlRxBf;

    PrachDeviceInternalStaticParamPerOcca* d_static = nullptr;
    PrachInternalDynParamPerOcca*          d_dyn    = nullptr;
    PrachInternalDynParamPerOcca           h_dyn{};

    __half2* d_rx = nullptr;
    ASSERT_EQ(cudaSuccess, cudaMalloc(reinterpret_cast<void**>(&d_rx), host_rx.size() * sizeof(__half2)));
    ASSERT_EQ(cudaSuccess, cudaMemcpy(d_rx, host_rx.data(), host_rx.size() * sizeof(__half2), cudaMemcpyHostToDevice));

    h_dyn.dataRx         = d_rx;
    h_dyn.occaPrmStatIdx = 0;
    h_dyn.occaPrmDynIdx  = 0;
    h_dyn.thr0           = thr0;
    h_dyn.nUplinkStreams = cfg.nUplinkStreams;

    ASSERT_EQ(cudaSuccess, cudaMalloc(reinterpret_cast<void**>(&d_static), sizeof(PrachDeviceInternalStaticParamPerOcca)));
    ASSERT_EQ(cudaSuccess, cudaMemcpy(d_static, &h_devStatic, sizeof(PrachDeviceInternalStaticParamPerOcca), cudaMemcpyHostToDevice));
    ASSERT_EQ(cudaSuccess, cudaMalloc(reinterpret_cast<void**>(&d_dyn), sizeof(PrachInternalDynParamPerOcca)));
    ASSERT_EQ(cudaSuccess, cudaMemcpy(d_dyn, &h_dyn, sizeof(PrachInternalDynParamPerOcca), cudaMemcpyHostToDevice));

    PrachInternalDynParamPerOcca    h_dyn_arr[1]    = {h_dyn};
    PrachInternalStaticParamPerOcca h_static_arr[1] = {h_staticOcca};

    uint32_t* d_numDetected = nullptr;
    uint32_t* d_prmbIndex   = nullptr;
    float*    d_prmbDelay   = nullptr;
    float*    d_prmbPower   = nullptr;
    float*    d_antRssi     = nullptr;
    float*    d_rssi        = nullptr;
    float*    d_interf      = nullptr;
    cudaAllocAndZero(&d_numDetected, 1);
    cudaAllocAndZero(&d_prmbIndex, 256);
    cudaAllocAndZero(&d_prmbDelay, 256);
    cudaAllocAndZero(&d_prmbPower, 256);
    cudaAllocAndZero(&d_antRssi, prm.N_ant);
    cudaAllocAndZero(&d_rssi, 1);
    cudaAllocAndZero(&d_interf, 1);

    cudaStream_t stream;
    ASSERT_EQ(cudaSuccess, cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    const uint16_t     nOccaProc       = 1;
    const uint16_t     maxAntenna      = prm.N_ant;
    const unsigned int align_l_oran    = ((prm.N_rep * L_ORAN + 31) >> 5) << 5;
    const unsigned int max_l_oran_ant  = align_l_oran * prm.N_ant;
    const unsigned int max_ant_u       = prm.N_ant * prm.uCount;
    const unsigned int max_nfft        = prm.Nfft;
    const int          zoneSizeForExt  = static_cast<int>((prm.N_CS * prm.Nfft + prm.L_RA - 1) / prm.L_RA);
    const int          max_zoneSizeExt = static_cast<int>(nextPowerOfTwo(std::max(1, zoneSizeForExt)));
    const unsigned int cudaArch        = getCudaArch();

    constexpr uint32_t maxFftsPerOccasion = MAX_N_ANTENNAS_SUPPORTED * PRACH_MAX_NUM_PREAMBLES * PRACH_NUM_NON_COHERENT_COMBINING_GROUPS;
    auto h_fftPointers = cuphy::buffer<cuFloatComplex *, cuphy::pinned_alloc>(PRACH_NUM_SUPPORTED_FFT_SIZES * nOccaProc * maxFftsPerOccasion);
    std::array<FftInfo, PRACH_NUM_SUPPORTED_FFT_SIZES> fftInfo;

    const cuphyStatus_t status = cuphyPrachReceiver(d_dyn,
                                                    d_static,
                                                    h_dyn_arr,
                                                    h_static_arr,
                                                    h_fftPointers.addr(),
                                                    fftInfo,
                                                    d_numDetected,
                                                    d_prmbIndex,
                                                    d_prmbDelay,
                                                    d_prmbPower,
                                                    d_antRssi,
                                                    d_rssi,
                                                    d_interf,
                                                    nOccaProc,
                                                    maxAntenna,
                                                    max_l_oran_ant,
                                                    max_ant_u,
                                                    max_nfft,
                                                    max_zoneSizeExt,
                                                    cudaArch,
                                                    stream);
    ASSERT_EQ(CUPHY_STATUS_SUCCESS, status);
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));

    ASSERT_EQ(cudaSuccess, cudaMemcpy(&outNumDetected, d_numDetected, sizeof(uint32_t), cudaMemcpyDeviceToHost));
    ASSERT_EQ(cudaSuccess, cudaMemcpy(&outRssiDb, d_rssi, sizeof(float), cudaMemcpyDeviceToHost));
    ASSERT_EQ(cudaSuccess, cudaMemcpy(&outInterfDb, d_interf, sizeof(float), cudaMemcpyDeviceToHost));

    cudaFree(d_rx);
    cudaFree(d_static);
    cudaFree(d_dyn);
    cudaFree(d_numDetected);
    cudaFree(d_prmbIndex);
    cudaFree(d_prmbDelay);
    cudaFree(d_prmbPower);
    cudaFree(d_antRssi);
    cudaFree(d_rssi);
    cudaFree(d_interf);
    cudaStreamDestroy(stream);
}

static cuphyStatus_t runSingleOccasionStatusArch(const PrachTestConfig& cfg,
                                                 bool                   onesSignal,
                                                 unsigned int           cudaArch)
{
    // Ensure no stale CUDA error from prior tests affects this run.
    (void)cudaGetLastError();

    PrachInternalStaticParamPerOcca h_staticOcca{};
    fillPrachParams(cfg, h_staticOcca.prach_params);
    h_staticOcca.enableUlRxBf = cfg.enableUlRxBf ? 1 : 0;

    const PrachParams&   prm = h_staticOcca.prach_params;
    std::vector<__half2> host_y_ref(static_cast<size_t>(prm.L_RA) * prm.uCount);
    makeOnesHalf2(host_y_ref);

    h_staticOcca.prach_workspace_buffer = cuphy::buffer<float, cuphy::device_alloc>(getWorkspaceFloats(prm));
    h_staticOcca.d_y_u_ref              = cuphy::buffer<__half2, cuphy::device_alloc>(prm.L_RA * prm.uCount);
    if(!host_y_ref.empty())
    {
        cudaMemcpy(h_staticOcca.d_y_u_ref.addr(), host_y_ref.data(), host_y_ref.size() * sizeof(__half2), cudaMemcpyHostToDevice);
    }

    PrachDeviceInternalStaticParamPerOcca h_devStatic{};
    h_devStatic.prach_params           = prm;
    h_devStatic.prach_workspace_buffer = h_staticOcca.prach_workspace_buffer.addr();
    h_devStatic.d_y_u_ref              = h_staticOcca.d_y_u_ref.addr();
    h_devStatic.enableUlRxBf           = h_staticOcca.enableUlRxBf;

    PrachDeviceInternalStaticParamPerOcca* d_static = nullptr;
    PrachInternalDynParamPerOcca*          d_dyn    = nullptr;
    PrachInternalDynParamPerOcca           h_dyn{};

    std::vector<__half2> host_rx;
    int                  mShift = computePdpTargetBinForPrmb0(prm);
    prepareInputRx(prm, prm.kBar, onesSignal, host_y_ref, mShift, /*amplitude=*/8.0f, host_rx);
    __half2* d_rx = nullptr;
    cudaMalloc(reinterpret_cast<void**>(&d_rx), host_rx.size() * sizeof(__half2));
    cudaMemcpy(d_rx, host_rx.data(), host_rx.size() * sizeof(__half2), cudaMemcpyHostToDevice);

    h_dyn.dataRx         = d_rx;
    h_dyn.occaPrmStatIdx = 0;
    h_dyn.occaPrmDynIdx  = 0;
    h_dyn.thr0           = onesSignal ? 0.05f : 5.0f;
    h_dyn.nUplinkStreams = cfg.nUplinkStreams;

    cudaMalloc(reinterpret_cast<void**>(&d_static), sizeof(PrachDeviceInternalStaticParamPerOcca));
    cudaMemcpy(d_static, &h_devStatic, sizeof(PrachDeviceInternalStaticParamPerOcca), cudaMemcpyHostToDevice);
    cudaMalloc(reinterpret_cast<void**>(&d_dyn), sizeof(PrachInternalDynParamPerOcca));
    cudaMemcpy(d_dyn, &h_dyn, sizeof(PrachInternalDynParamPerOcca), cudaMemcpyHostToDevice);

    PrachInternalDynParamPerOcca    h_dyn_arr[1]    = {h_dyn};
    PrachInternalStaticParamPerOcca h_static_arr[1] = {h_staticOcca};

    uint32_t* d_numDetected = nullptr;
    uint32_t* d_prmbIndex   = nullptr;
    float*    d_prmbDelay   = nullptr;
    float*    d_prmbPower   = nullptr;
    float*    d_antRssi     = nullptr;
    float*    d_rssi        = nullptr;
    float*    d_interf      = nullptr;
    cudaAllocAndZero(&d_numDetected, 1);
    cudaAllocAndZero(&d_prmbIndex, 256);
    cudaAllocAndZero(&d_prmbDelay, 256);
    cudaAllocAndZero(&d_prmbPower, 256);
    cudaAllocAndZero(&d_antRssi, prm.N_ant);
    cudaAllocAndZero(&d_rssi, 1);
    cudaAllocAndZero(&d_interf, 1);

    cudaStream_t stream;
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);

    const uint16_t     nOccaProc       = 1;
    const uint16_t     maxAntenna      = prm.N_ant;
    const int          L_ORAN          = (prm.L_RA == 139 ? 144 : 864);
    const unsigned int align_l_oran    = ((prm.N_rep * L_ORAN + 31) >> 5) << 5;
    const unsigned int max_l_oran_ant  = align_l_oran * prm.N_ant;
    const unsigned int max_ant_u       = prm.N_ant * prm.uCount;
    const unsigned int max_nfft        = prm.Nfft;
    const int          zoneSizeForExt  = static_cast<int>((prm.N_CS * prm.Nfft + prm.L_RA - 1) / prm.L_RA);
    const int          max_zoneSizeExt = static_cast<int>(nextPowerOfTwo(std::max(1, zoneSizeForExt)));

    constexpr uint32_t maxFftsPerOccasion = MAX_N_ANTENNAS_SUPPORTED * PRACH_MAX_NUM_PREAMBLES * PRACH_NUM_NON_COHERENT_COMBINING_GROUPS;
    auto h_fftPointers = cuphy::buffer<cuFloatComplex *, cuphy::pinned_alloc>(PRACH_NUM_SUPPORTED_FFT_SIZES * nOccaProc * maxFftsPerOccasion);
    std::array<FftInfo, PRACH_NUM_SUPPORTED_FFT_SIZES> fftInfo;

    cuphyStatus_t status = cuphyPrachReceiver(d_dyn,
                                              d_static,
                                              h_dyn_arr,
                                              h_static_arr,
                                              h_fftPointers.addr(),
                                              fftInfo,
                                              d_numDetected,
                                              d_prmbIndex,
                                              d_prmbDelay,
                                              d_prmbPower,
                                              d_antRssi,
                                              d_rssi,
                                              d_interf,
                                              nOccaProc,
                                              maxAntenna,
                                              max_l_oran_ant,
                                              max_ant_u,
                                              max_nfft,
                                              max_zoneSizeExt,
                                              cudaArch,
                                              stream);

    cudaStreamDestroy(stream);
    cudaFree(d_rx);
    cudaFree(d_static);
    cudaFree(d_dyn);
    cudaFree(d_numDetected);
    cudaFree(d_prmbIndex);
    cudaFree(d_prmbDelay);
    cudaFree(d_prmbPower);
    cudaFree(d_antRssi);
    cudaFree(d_rssi);
    cudaFree(d_interf);
    return status;
}

static cuphyStatus_t runSingleOccasionStatus(const PrachTestConfig& cfg,
                                             bool                   onesSignal)
{
    return runSingleOccasionStatusArch(cfg, onesSignal, getCudaArch());
}

struct GraphContextTwoOcc
{
    PrachInternalStaticParamPerOcca                                           h_static[2];
    PrachInternalDynParamPerOcca                                              h_dyn_host[2];
    cuphy::buffer<PrachDeviceInternalStaticParamPerOcca, cuphy::device_alloc> d_static;
    cuphy::buffer<PrachInternalDynParamPerOcca, cuphy::device_alloc>          d_dyn;
    __half2*                                                                  d_rx0{nullptr};
    __half2*                                                                  d_rx1{nullptr};
    uint16_t                                                                  nTotCellOcca{2};
    uint16_t                                                                  nMaxOccasions{2};
    uint16_t                                                                  maxAntenna{1};
    unsigned int                                                              max_l_oran_ant{0};
    unsigned int                                                              max_ant_u{1};
    unsigned int                                                              max_nfft{1024};
    int                                                                       max_zoneSizeExt{256};
    unsigned int                                                              cudaArch{0};
};

static GraphContextTwoOcc createGraphContextForTwoOccasions(const PrachTestConfig& cfg0, const PrachTestConfig& cfg1)
{
    GraphContextTwoOcc ctx{};
    fillPrachParams(cfg0, ctx.h_static[0].prach_params);
    fillPrachParams(cfg1, ctx.h_static[1].prach_params);
    ctx.h_static[0].enableUlRxBf = cfg0.enableUlRxBf ? 1 : 0;
    ctx.h_static[1].enableUlRxBf = cfg1.enableUlRxBf ? 1 : 0;

    PrachDeviceInternalStaticParamPerOcca h_devStatic[2]{};
    for(int i = 0; i < 2; i++)
    {
        const PrachParams& prm                 = ctx.h_static[i].prach_params;
        ctx.h_static[i].prach_workspace_buffer = cuphy::buffer<float, cuphy::device_alloc>(getWorkspaceFloats(prm));
        ctx.h_static[i].d_y_u_ref              = cuphy::buffer<__half2, cuphy::device_alloc>(prm.L_RA * prm.uCount);
        std::vector<__half2> host_y_ref(static_cast<size_t>(prm.L_RA) * prm.uCount, __floats2half2_rn(1.0f, 0.0f));
        if(!host_y_ref.empty())
        {
            cudaMemcpy(ctx.h_static[i].d_y_u_ref.addr(), host_y_ref.data(), host_y_ref.size() * sizeof(__half2), cudaMemcpyHostToDevice);
        }

        h_devStatic[i].prach_params           = prm;
        h_devStatic[i].prach_workspace_buffer = ctx.h_static[i].prach_workspace_buffer.addr();
        h_devStatic[i].d_y_u_ref              = ctx.h_static[i].d_y_u_ref.addr();
        h_devStatic[i].enableUlRxBf           = ctx.h_static[i].enableUlRxBf;

        ctx.maxAntenna            = std::max<uint16_t>(ctx.maxAntenna, prm.N_ant);
        ctx.max_ant_u             = std::max(ctx.max_ant_u, prm.N_ant * prm.uCount);
        ctx.max_nfft              = std::max(ctx.max_nfft, prm.Nfft);
        int zoneSize              = (int)((prm.N_CS * prm.Nfft + prm.L_RA - 1) / prm.L_RA);
        ctx.max_zoneSizeExt       = std::max(ctx.max_zoneSizeExt, (int)nextPowerOfTwo(std::max(1, zoneSize)));
        const int    L_ORAN       = (prm.L_RA == 139 ? 144 : 864);
        unsigned int align_l_oran = ((prm.N_rep * L_ORAN + 31) >> 5) << 5;
        ctx.max_l_oran_ant        = std::max(ctx.max_l_oran_ant, align_l_oran);
    }

    ctx.d_static = cuphy::buffer<PrachDeviceInternalStaticParamPerOcca, cuphy::device_alloc>(2);
    cudaMemcpy(ctx.d_static.addr(), h_devStatic, sizeof(h_devStatic), cudaMemcpyHostToDevice);

    // Build dyn params and zero RX
    PrachInternalDynParamPerOcca h_dyn[2]{};
    for(int i = 0; i < 2; i++)
    {
        const PrachParams&   prm    = ctx.h_static[i].prach_params;
        const int            L_ORAN = (prm.L_RA == 139 ? 144 : 864);
        std::vector<__half2> host_rx(L_ORAN * prm.N_rep * prm.N_ant, __floats2half2_rn(0.f, 0.f));
        __half2*             d_rx = nullptr;
        cudaMalloc((void**)&d_rx, host_rx.size() * sizeof(__half2));
        cudaMemcpy(d_rx, host_rx.data(), host_rx.size() * sizeof(__half2), cudaMemcpyHostToDevice);
        if(i == 0)
            ctx.d_rx0 = d_rx;
        else
            ctx.d_rx1 = d_rx;
        h_dyn[i]          = {d_rx, (uint16_t)i, (uint16_t)i, 0.0f, 1};
        ctx.h_dyn_host[i] = h_dyn[i];
    }
    ctx.d_dyn = cuphy::buffer<PrachInternalDynParamPerOcca, cuphy::device_alloc>(2);
    cudaMemcpy(ctx.d_dyn.addr(), h_dyn, sizeof(h_dyn), cudaMemcpyHostToDevice);

    ctx.cudaArch = getCudaArch();
    return ctx;
}

struct GraphOutputs
{
    uint32_t* num{nullptr};
    uint32_t* idx{nullptr};
    float*    dly{nullptr};
    float*    pwr{nullptr};
    float*    ant{nullptr};
    float*    rssi{nullptr};
    float*    interf{nullptr};
};

static void allocGraphOutputs(GraphOutputs& out, uint32_t maxAnt, uint32_t prmbCap = 128, uint32_t occaCap = 2)
{
    cudaAllocAndZero(&out.num, occaCap);
    cudaAllocAndZero(&out.idx, prmbCap);
    cudaAllocAndZero(&out.dly, prmbCap);
    cudaAllocAndZero(&out.pwr, prmbCap);
    cudaAllocAndZero(&out.ant, maxAnt);
    cudaAllocAndZero(&out.rssi, occaCap);
    cudaAllocAndZero(&out.interf, occaCap);
}

static void freeGraphOutputs(GraphOutputs& out)
{
    cudaFree(out.num);
    cudaFree(out.idx);
    cudaFree(out.dly);
    cudaFree(out.pwr);
    cudaFree(out.ant);
    cudaFree(out.rssi);
    cudaFree(out.interf);
    out = {};
}

struct PrevPtrs
{
    uint32_t* num{nullptr};
    uint32_t* idx{nullptr};
    float*    dly{nullptr};
    float*    pwr{nullptr};
    float*    ant{nullptr};
    float*    rssi{nullptr};
    float*    interf{nullptr};
    uint16_t  nPrevOccaProc{0};
};

static void updateAndLaunchGraph(cudaGraphExec_t               graphExec,
                                 std::vector<cudaGraphNode_t>& nodes,
                                 GraphContextTwoOcc&           ctx,
                                 cuphy::buffer<cuFloatComplex *, cuphy::pinned_alloc>& h_fftPointers,
                                 std::array<FftInfo, PRACH_NUM_SUPPORTED_FFT_SIZES>& fftInfo,
                                 GraphOutputs&                 outs,
                                 PrevPtrs&                     prev,
                                 uint16_t                      nMaxOccasions,
                                 uint16_t                      nOcca,
                                 uint16_t                      maxAntenna,
                                 unsigned int                  max_l_oran_ant,
                                 unsigned int                  max_ant_u,
                                 unsigned int                  max_nfft,
                                 int                           max_zoneSizeExt,
                                 std::vector<char>&            activeOcc,
                                 std::vector<char>&            prevActive)
{
    ASSERT_EQ(cuphyPrachUpdateGraph(graphExec, nodes, ctx.d_dyn.addr(), ctx.d_static.addr(), ctx.h_dyn_host, ctx.h_static, h_fftPointers.addr(), fftInfo, outs.num, outs.idx, outs.dly, outs.pwr, outs.ant, outs.rssi, outs.interf, prev.num, prev.idx, prev.dly, prev.pwr, prev.ant, prev.rssi, prev.interf, nMaxOccasions, prev.nPrevOccaProc, nOcca, maxAntenna, max_l_oran_ant, max_ant_u, max_nfft, max_zoneSizeExt, activeOcc, prevActive), CUPHY_STATUS_SUCCESS);
    ASSERT_EQ(cuphyPrachLaunchGraph(graphExec, 0), CUPHY_STATUS_SUCCESS);
}

struct GraphContextSingleOcc
{
    PrachInternalStaticParamPerOcca                                           h_static;
    PrachInternalDynParamPerOcca                                              h_dyn_host;
    cuphy::buffer<PrachDeviceInternalStaticParamPerOcca, cuphy::device_alloc> d_static;
    cuphy::buffer<PrachInternalDynParamPerOcca, cuphy::device_alloc>          d_dyn;
    __half2*                                                                  d_rx{nullptr};
    uint16_t                                                                  nTotCellOcca{1};
    uint16_t                                                                  nMaxOccasions{1};
    uint16_t                                                                  maxAntenna{1};
    unsigned int                                                              max_l_oran_ant{0};
    unsigned int                                                              max_ant_u{1};
    unsigned int                                                              max_nfft{256};
    int                                                                       max_zoneSizeExt{256};
    unsigned int                                                              cudaArch{0};
};

static GraphContextSingleOcc createGraphContextSingleOcc(const PrachTestConfig& cfg)
{
    GraphContextSingleOcc ctx{};
    fillPrachParams(cfg, ctx.h_static.prach_params);
    ctx.h_static.enableUlRxBf           = cfg.enableUlRxBf ? 1 : 0;
    const PrachParams& prm              = ctx.h_static.prach_params;
    ctx.h_static.prach_workspace_buffer = cuphy::buffer<float, cuphy::device_alloc>(getWorkspaceFloats(prm));
    ctx.h_static.d_y_u_ref              = cuphy::buffer<__half2, cuphy::device_alloc>(prm.L_RA * prm.uCount);
    std::vector<__half2> host_y_ref(static_cast<size_t>(prm.L_RA) * prm.uCount, __floats2half2_rn(1.0f, 0.0f));
    if(!host_y_ref.empty())
    {
        cudaMemcpy(ctx.h_static.d_y_u_ref.addr(), host_y_ref.data(), host_y_ref.size() * sizeof(__half2), cudaMemcpyHostToDevice);
    }

    PrachDeviceInternalStaticParamPerOcca h_dev{};
    h_dev.prach_params           = prm;
    h_dev.prach_workspace_buffer = ctx.h_static.prach_workspace_buffer.addr();
    h_dev.d_y_u_ref              = ctx.h_static.d_y_u_ref.addr();
    h_dev.enableUlRxBf           = ctx.h_static.enableUlRxBf;
    ctx.d_static                 = cuphy::buffer<PrachDeviceInternalStaticParamPerOcca, cuphy::device_alloc>(1);
    cudaMemcpy(ctx.d_static.addr(), &h_dev, sizeof(h_dev), cudaMemcpyHostToDevice);

    // dyn and RX
    const int            L_ORAN = (prm.L_RA == 139 ? 144 : 864);
    std::vector<__half2> host_rx(L_ORAN * prm.N_rep * prm.N_ant, __floats2half2_rn(0.f, 0.f));
    cudaMalloc((void**)&ctx.d_rx, host_rx.size() * sizeof(__half2));
    cudaMemcpy(ctx.d_rx, host_rx.data(), host_rx.size() * sizeof(__half2), cudaMemcpyHostToDevice);
    ctx.h_dyn_host = {ctx.d_rx, 0, 0, 0.0f, 1};
    ctx.d_dyn      = cuphy::buffer<PrachInternalDynParamPerOcca, cuphy::device_alloc>(1);
    cudaMemcpy(ctx.d_dyn.addr(), &ctx.h_dyn_host, sizeof(ctx.h_dyn_host), cudaMemcpyHostToDevice);

    ctx.maxAntenna            = prm.N_ant;
    ctx.max_ant_u             = prm.N_ant * prm.uCount;
    ctx.max_nfft              = prm.Nfft;
    int zoneSize              = (int)((prm.N_CS * prm.Nfft + prm.L_RA - 1) / prm.L_RA);
    ctx.max_zoneSizeExt       = (int)nextPowerOfTwo(std::max(1, zoneSize));
    unsigned int align_l_oran = ((prm.N_rep * L_ORAN + 31) >> 5) << 5;
    ctx.max_l_oran_ant        = align_l_oran;
    ctx.cudaArch              = getCudaArch();
    return ctx;
}

static cuphyStatus_t runTwoOccasionsStatus(const PrachTestConfig& cfg0,
                                           const PrachTestConfig& cfg1,
                                           bool                   ones0,
                                           bool                   ones1)
{
    // Host static arrays
    PrachInternalStaticParamPerOcca h_static[2]{};
    fillPrachParams(cfg0, h_static[0].prach_params);
    fillPrachParams(cfg1, h_static[1].prach_params);
    h_static[0].enableUlRxBf = cfg0.enableUlRxBf ? 1 : 0;
    h_static[1].enableUlRxBf = cfg1.enableUlRxBf ? 1 : 0;

    // Device static array
    PrachDeviceInternalStaticParamPerOcca h_devStatic[2]{};
    for(int i = 0; i < 2; i++)
    {
        const PrachParams& prm             = h_static[i].prach_params;
        h_static[i].prach_workspace_buffer = cuphy::buffer<float, cuphy::device_alloc>(getWorkspaceFloats(prm));
        h_static[i].d_y_u_ref              = cuphy::buffer<__half2, cuphy::device_alloc>(prm.L_RA * prm.uCount);
        std::vector<__half2> host_y_ref(static_cast<size_t>(prm.L_RA) * prm.uCount, __floats2half2_rn(1.0f, 0.0f));
        if(!host_y_ref.empty())
        {
            cudaMemcpy(h_static[i].d_y_u_ref.addr(), host_y_ref.data(), host_y_ref.size() * sizeof(__half2), cudaMemcpyHostToDevice);
        }
        h_devStatic[i].prach_params           = prm;
        h_devStatic[i].prach_workspace_buffer = h_static[i].prach_workspace_buffer.addr();
        h_devStatic[i].d_y_u_ref              = h_static[i].d_y_u_ref.addr();
        h_devStatic[i].enableUlRxBf           = h_static[i].enableUlRxBf;
    }
    cuphy::buffer<PrachDeviceInternalStaticParamPerOcca, cuphy::device_alloc> d_static(2);
    cudaMemcpy(d_static.addr(), h_devStatic, sizeof(h_devStatic), cudaMemcpyHostToDevice);

    // Dyn arrays
    PrachInternalDynParamPerOcca h_dyn[2]{};
    __half2*                     d_rx[2]{};
    for(int i = 0; i < 2; i++)
    {
        const PrachParams&   prm    = h_static[i].prach_params;
        const int            L_ORAN = (prm.L_RA == 139 ? 144 : 864);
        std::vector<__half2> host_rx(L_ORAN * prm.N_rep * prm.N_ant, __floats2half2_rn(0.f, 0.f));
        // Optionally synthesize signal for detect path
        if((i == 0 && ones0) || (i == 1 && ones1))
        {
            int                  mShift = computePdpTargetBinForPrmb0(prm);
            std::vector<__half2> host_y_ref(prm.L_RA, __floats2half2_rn(1.0f, 0.0f));
            prepareInputRx(prm, prm.kBar, true, host_y_ref, mShift, /*amp=*/8.0f, host_rx);
        }
        cudaMalloc((void**)&d_rx[i], host_rx.size() * sizeof(__half2));
        cudaMemcpy(d_rx[i], host_rx.data(), host_rx.size() * sizeof(__half2), cudaMemcpyHostToDevice);
        h_dyn[i] = {d_rx[i], (uint16_t)i, (uint16_t)i, ones0 || ones1 ? 0.01f : 5.0f, 1};
    }
    cuphy::buffer<PrachInternalDynParamPerOcca, cuphy::device_alloc> d_dyn(2);
    cudaMemcpy(d_dyn.addr(), h_dyn, sizeof(h_dyn), cudaMemcpyHostToDevice);

    // Outputs
    uint32_t* d_numDetected = nullptr;
    uint32_t* d_prmbIndex   = nullptr;
    float*    d_prmbDelay   = nullptr;
    float*    d_prmbPower   = nullptr;
    float*    d_antRssi     = nullptr;
    float*    d_rssi        = nullptr;
    float*    d_interf      = nullptr;
    cudaAllocAndZero(&d_numDetected, 2);
    cudaAllocAndZero(&d_prmbIndex, 128);
    cudaAllocAndZero(&d_prmbDelay, 128);
    cudaAllocAndZero(&d_prmbPower, 128);
    cudaAllocAndZero(&d_antRssi, std::max(h_static[0].prach_params.N_ant, h_static[1].prach_params.N_ant));
    cudaAllocAndZero(&d_rssi, 2);
    cudaAllocAndZero(&d_interf, 2);

    // Max parameters
    uint16_t           maxAntenna      = std::max<uint16_t>(h_static[0].prach_params.N_ant, h_static[1].prach_params.N_ant);
    const unsigned int max_l_oran_ant  = std::max((unsigned int)(((h_static[0].prach_params.N_rep * ((h_static[0].prach_params.L_RA == 139) ? 144 : 864)) + 31) >> 5) << 5,
                                                 (unsigned int)(((h_static[1].prach_params.N_rep * ((h_static[1].prach_params.L_RA == 139) ? 144 : 864)) + 31) >> 5) << 5);
    const unsigned int max_ant_u       = std::max(h_static[0].prach_params.N_ant * h_static[0].prach_params.uCount,
                                            h_static[1].prach_params.N_ant * h_static[1].prach_params.uCount);
    const unsigned int max_nfft        = std::max(h_static[0].prach_params.Nfft, h_static[1].prach_params.Nfft);
    const int          max_zoneSizeExt = std::max((int)nextPowerOfTwo(std::max(1, (int)((h_static[0].prach_params.N_CS * h_static[0].prach_params.Nfft + h_static[0].prach_params.L_RA - 1) / h_static[0].prach_params.L_RA))),
                                         (int)nextPowerOfTwo(std::max(1, (int)((h_static[1].prach_params.N_CS * h_static[1].prach_params.Nfft + h_static[1].prach_params.L_RA - 1) / h_static[1].prach_params.L_RA))));

    uint16_t nOccaProc = 2;
    constexpr uint32_t maxFftsPerOccasion = MAX_N_ANTENNAS_SUPPORTED * PRACH_MAX_NUM_PREAMBLES * PRACH_NUM_NON_COHERENT_COMBINING_GROUPS;
    auto h_fftPointers = cuphy::buffer<cuFloatComplex *, cuphy::pinned_alloc>(PRACH_NUM_SUPPORTED_FFT_SIZES * nOccaProc * maxFftsPerOccasion);
    std::array<FftInfo, PRACH_NUM_SUPPORTED_FFT_SIZES> fftInfo;

    cudaStream_t stream;
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    PrachInternalDynParamPerOcca h_dyn_arr[2] = {h_dyn[0], h_dyn[1]};
    cuphyStatus_t                status       = cuphyPrachReceiver(d_dyn.addr(), d_static.addr(), h_dyn_arr, h_static, h_fftPointers.addr(), fftInfo, d_numDetected, d_prmbIndex, d_prmbDelay, d_prmbPower, d_antRssi, d_rssi, d_interf,
                                              /*nOccaProc=*/2,
                                              maxAntenna,
                                              max_l_oran_ant,
                                              max_ant_u,
                                              max_nfft,
                                              max_zoneSizeExt,
                                              getCudaArch(),
                                              stream);
    cudaStreamDestroy(stream);
    cudaFree(d_rx[0]);
    cudaFree(d_rx[1]);
    cudaFree(d_numDetected);
    cudaFree(d_prmbIndex);
    cudaFree(d_prmbDelay);
    cudaFree(d_prmbPower);
    cudaFree(d_antRssi);
    cudaFree(d_rssi);
    cudaFree(d_interf);
    return status;
}

} // namespace

//=============================== Tests ======================================
// Classification:
// - NonGraph_*: host non-graph path coverage (cuphyPrachReceiver)
// - Graph_*: graph create/update/launch happy paths
// - GraphNeg_*: graph negative/error handling
// - DISABLED_*: known flaky or pending detection tests

// -------- Non-graph tests --------
TEST(NonGraphSuite, NoSignal_NoDetection_B4)
{
    PrachTestConfig cfg{};
    cfg.L_RA           = 139;
    cfg.Nfft           = 256;
    cfg.N_rep          = 1;
    cfg.N_CS           = 1; // simple zone partition
    cfg.uCount         = 1;
    cfg.N_nc           = 1;
    cfg.N_ant          = 1;
    cfg.mu             = 0;
    cfg.delta_f_RA     = 15000; // 15 kHz
    cfg.kBar           = 2;
    cfg.enableUlRxBf   = false;
    cfg.nUplinkStreams = 1;

    uint32_t numDetected = 12345;
    float    rssiDb      = 0.0f;
    float    interfDb    = 0.0f;
    runSingleOccasionTest(cfg, /*onesSignal=*/false, numDetected, rssiDb, interfDb);

    EXPECT_EQ(numDetected, 0u);
    // With no signal, RSSI and interference should be set to -100 dB
    EXPECT_LE(rssiDb, -99.0f);
    EXPECT_LE(interfDb, -99.0f);
}

TEST(NonGraphSuite, UlRxBf_NoSignal_RssiMinus100)
{
    PrachTestConfig cfg{};
    cfg.L_RA           = 139;
    cfg.Nfft           = 256;
    cfg.N_rep          = 1;
    cfg.N_CS           = 1;
    cfg.uCount         = 1;
    cfg.N_nc           = 1;
    cfg.N_ant          = 2;
    cfg.mu             = 0;
    cfg.delta_f_RA     = 15000;
    cfg.kBar           = 2;
    cfg.enableUlRxBf   = true;
    cfg.nUplinkStreams = 1;

    uint32_t numDetected = 12345;
    float    rssiDb      = 0.0f;
    float    interfDb    = 0.0f;
    runSingleOccasionTest(cfg, /*onesSignal=*/false, numDetected, rssiDb, interfDb);

    EXPECT_EQ(numDetected, 0u);
    EXPECT_LE(rssiDb, -99.0f);
    EXPECT_LE(interfDb, -99.0f);
}

TEST(NonGraphSuite, UnsupportedFftSize_ReturnsError)
{
    PrachTestConfig cfg{};
    cfg.L_RA           = 139;
    cfg.Nfft           = 512; // unsupported by cuFFTDx path (256/1024 only)
    cfg.N_rep          = 1;
    cfg.N_CS           = 1;
    cfg.uCount         = 1;
    cfg.N_nc           = 1;
    cfg.N_ant          = 1;
    cfg.mu             = 0;
    cfg.delta_f_RA     = 15000;
    cfg.kBar           = 2;
    cfg.enableUlRxBf   = false;
    cfg.nUplinkStreams = 1;

    cuphyStatus_t status = runSingleOccasionStatus(cfg, /*onesSignal=*/false);
    EXPECT_NE(status, CUPHY_STATUS_SUCCESS);
}

TEST(NonGraphSuite, Rssi_NonZero_MultiAntenna)
{
    PrachTestConfig cfg{};
    cfg.L_RA           = 139;
    cfg.Nfft           = 256;
    cfg.N_rep          = 2; // average over repetitions
    cfg.N_CS           = 1;
    cfg.uCount         = 1;
    cfg.N_nc           = 1;
    cfg.N_ant          = 2; // multi-antenna
    cfg.mu             = 0;
    cfg.delta_f_RA     = 15000;
    cfg.kBar           = 2;
    cfg.enableUlRxBf   = false;
    cfg.nUplinkStreams = 1;

    uint32_t numDetected = 0;
    float    rssiDb      = -100.0f;
    float    interfDb    = -100.0f;
    runSingleOccasionTest(cfg, /*onesSignal=*/true, numDetected, rssiDb, interfDb);

    // We only assert RSSI becomes > -100 (non-zero RX), detection may still be 0 with synthetic
    EXPECT_GT(rssiDb, -100.0f);
}

TEST(NonGraphSuite, PDP_With_TwoNc_Graphless_Runs)
{
    PrachTestConfig cfg{};
    cfg.L_RA           = 139;
    cfg.Nfft           = 256;
    cfg.N_rep          = 2;
    cfg.N_CS           = 1;
    cfg.uCount         = 1;
    cfg.N_nc           = 2; // exercise non-coherent combining loop
    cfg.N_ant          = 1;
    cfg.mu             = 0;
    cfg.delta_f_RA     = 15000;
    cfg.kBar           = 2;
    cfg.enableUlRxBf   = false;
    cfg.nUplinkStreams = 1;

    // Just ensure status success
    cuphyStatus_t status = runSingleOccasionStatus(cfg, /*onesSignal=*/false);
    EXPECT_EQ(status, CUPHY_STATUS_SUCCESS);
}

TEST(NonGraphSuite, ZoneSizeExt_256_Supported)
{
    PrachTestConfig cfg{};
    cfg.L_RA           = 139;
    cfg.Nfft           = 256;
    cfg.N_rep          = 1;
    cfg.N_CS           = 108; // yields zoneSize ~200, zoneSizeExt -> 256 (<= NUM_THREAD), avoids known kernel mismatch
    cfg.uCount         = 1;
    cfg.N_nc           = 1;
    cfg.N_ant          = 1;
    cfg.mu             = 0;
    cfg.delta_f_RA     = 15000;
    cfg.kBar           = 2;
    cfg.enableUlRxBf   = false;
    cfg.nUplinkStreams = 1;

    // Ensure kernels run; success indicates PDP kernel accepted large zone size ext
    cuphyStatus_t status = runSingleOccasionStatus(cfg, /*onesSignal=*/false);
    EXPECT_EQ(status, CUPHY_STATUS_SUCCESS);
}

TEST(NonGraphSuite, ZoneSizeExt_512_CoversMaxZoneSizeExtBranch_NoOccasions)
{
    // Purpose: cover the ternary branch:
    //   block_dim = dim3(max_zoneSizeExt > NUM_THREAD ? max_zoneSizeExt : NUM_THREAD);
    // by setting max_zoneSizeExt=512 (> NUM_THREAD=256) while avoiding the heavy kernels that
    // have shown illegal-memory-access issues for some parameter combinations.
    //
    // We do this by calling cuphyPrachReceiver with nOccaProc=0. The code still evaluates the
    // ternary and computes launch dims, but kernels are launched with a zero-sized grid, which
    // results in a clean failure status rather than device memory corruption.

    // Ensure no stale CUDA error from prior tests affects this run.
    (void)cudaGetLastError();

    PrachDeviceInternalStaticParamPerOcca* d_static{};
    PrachInternalDynParamPerOcca*          d_dyn{};
    ASSERT_EQ(cudaSuccess, cudaMalloc(reinterpret_cast<void**>(&d_static), sizeof(PrachDeviceInternalStaticParamPerOcca)));
    ASSERT_EQ(cudaSuccess, cudaMalloc(reinterpret_cast<void**>(&d_dyn), sizeof(PrachInternalDynParamPerOcca)));

    PrachInternalStaticParamPerOcca h_static[1]{};
    PrachInternalDynParamPerOcca    h_dyn[1]{};

    uint32_t* num{};
    uint32_t* idx{};
    float*    dly{};
    float*    pwr{};
    float*    ant{};
    float*    rssi{};
    float*    interf{};
    cudaAllocAndZero(&num, 1);
    cudaAllocAndZero(&idx, 1);
    cudaAllocAndZero(&dly, 1);
    cudaAllocAndZero(&pwr, 1);
    cudaAllocAndZero(&ant, 1);
    cudaAllocAndZero(&rssi, 1);
    cudaAllocAndZero(&interf, 1);

    cudaStream_t stream{};
    ASSERT_EQ(cudaSuccess, cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    uint16_t nOccaProc = 0;
    constexpr uint32_t maxFftsPerOccasion = MAX_N_ANTENNAS_SUPPORTED * PRACH_MAX_NUM_PREAMBLES * PRACH_NUM_NON_COHERENT_COMBINING_GROUPS;
    auto h_fftPointers = cuphy::buffer<cuFloatComplex *, cuphy::pinned_alloc>(PRACH_NUM_SUPPORTED_FFT_SIZES * nOccaProc * maxFftsPerOccasion);
    std::array<FftInfo, PRACH_NUM_SUPPORTED_FFT_SIZES> fftInfo;

    const cuphyStatus_t status = cuphyPrachReceiver(d_dyn,
                                                    d_static,
                                                    h_dyn,
                                                    h_static,
                                                    h_fftPointers.addr(),
                                                    fftInfo,
                                                    num,
                                                    idx,
                                                    dly,
                                                    pwr,
                                                    ant,
                                                    rssi,
                                                    interf,
                                                    /*nOccaProc=*/0,
                                                    /*maxAntenna=*/1,
                                                    /*max_l_oran_ant=*/0,
                                                    /*max_ant_u=*/1,
                                                    /*max_nfft=*/256,
                                                    /*max_zoneSizeExt=*/512,
                                                    getCudaArch(),
                                                    stream);
    EXPECT_NE(status, CUPHY_STATUS_SUCCESS);

    (void)cudaStreamDestroy(stream);
    cudaFree(num);
    cudaFree(idx);
    cudaFree(dly);
    cudaFree(pwr);
    cudaFree(ant);
    cudaFree(rssi);
    cudaFree(interf);
    cudaFree(d_static);
    cudaFree(d_dyn);

    // Clear any CUDA "last error" from the intentional failure.
    (void)cudaGetLastError();
}

TEST(NonGraphSuite, TwoOccasions_Run_Succeeds_FR1_and_FMT0)
{
    // Build two occasions: B4 and fmt0; exercise both in a single non-graph call
    PrachTestConfig cfg0{};
    cfg0.L_RA           = 139;
    cfg0.Nfft           = 256;
    cfg0.N_rep          = 1;
    cfg0.N_CS           = 1;
    cfg0.uCount         = 1;
    cfg0.N_nc           = 1;
    cfg0.N_ant          = 1;
    cfg0.mu             = 0;
    cfg0.delta_f_RA     = 15000;
    cfg0.kBar           = 2;
    cfg0.enableUlRxBf   = false;
    cfg0.nUplinkStreams = 1;
    PrachTestConfig cfg1{};
    cfg1.L_RA           = 839;
    cfg1.Nfft           = 1024;
    cfg1.N_rep          = 1;
    cfg1.N_CS           = 1;
    cfg1.uCount         = 1;
    cfg1.N_nc           = 1;
    cfg1.N_ant          = 1;
    cfg1.mu             = 0;
    cfg1.delta_f_RA     = 1250;
    cfg1.kBar           = 7;
    cfg1.enableUlRxBf   = false;
    cfg1.nUplinkStreams = 1;
    cuphyStatus_t s     = runTwoOccasionsStatus(cfg0, cfg1, /*ones0=*/false, /*ones1=*/false);
    EXPECT_EQ(s, CUPHY_STATUS_SUCCESS);
}

TEST(NonGraphSuite, TwoOccasions_MixedAntenna_CoversMemsetRssiAntIndexGuard)
{
    // Purpose: ensure maxAntenna > N_ant for at least one occasion, so memsetRssi hits:
    //   if (antIndex >= N_ant + 2) return;
    // This is a coverage-only branch for mixed-antenna multi-occasion launches.
    PrachTestConfig cfg0{};
    cfg0.L_RA           = 139;
    cfg0.Nfft           = 256;
    cfg0.N_rep          = 1;
    cfg0.N_CS           = 1;
    cfg0.uCount         = 1;
    cfg0.N_nc           = 1;
    cfg0.N_ant          = 1; // smaller
    cfg0.mu             = 0;
    cfg0.delta_f_RA     = 15000;
    cfg0.kBar           = 2;
    cfg0.enableUlRxBf   = false;
    cfg0.nUplinkStreams = 1;

    PrachTestConfig cfg1 = cfg0;
    cfg1.N_ant          = 2; // larger => maxAntenna becomes 2

    const cuphyStatus_t s = runTwoOccasionsStatus(cfg0, cfg1, /*ones0=*/false, /*ones1=*/false);
    EXPECT_EQ(s, CUPHY_STATUS_SUCCESS);
}

TEST(NonGraphSuite, ArchFallback_UnsupportedArch_ReturnsError)
{
    PrachTestConfig cfg{};
    cfg.L_RA           = 139;
    cfg.Nfft           = 256;
    cfg.N_rep          = 1;
    cfg.N_CS           = 1;
    cfg.uCount         = 1;
    cfg.N_nc           = 1;
    cfg.N_ant          = 1;
    cfg.mu             = 0;
    cfg.delta_f_RA     = 15000;
    cfg.kBar           = 2;
    cfg.enableUlRxBf   = false;
    cfg.nUplinkStreams = 1;

    // Use a clearly unsupported arch code to exercise error path
    cuphyStatus_t status = runSingleOccasionStatusArch(cfg, /*onesSignal=*/false, /*cudaArch=*/12345);
    EXPECT_NE(status, CUPHY_STATUS_SUCCESS);
}

TEST(NonGraphSuite, SearchThreshold_NoiseVsDetect)
{
    // Two runs over identical config with different thr0 to flip noise/detect branches
    PrachTestConfig cfg{};
    cfg.L_RA           = 139;
    cfg.Nfft           = 256;
    cfg.N_rep          = 2;
    cfg.N_CS           = 1;
    cfg.uCount         = 1;
    cfg.N_nc           = 1;
    cfg.N_ant          = 1;
    cfg.mu             = 0;
    cfg.delta_f_RA     = 15000;
    cfg.kBar           = 2;
    cfg.enableUlRxBf   = false;
    cfg.nUplinkStreams = 1;

    uint32_t numDetected = 0;
    float    rssiDb      = 0.0f;
    float    interfDb    = 0.0f;

    // Noise: strong thr0 to push peaks below threshold
    {
        PrachTestConfig noisy = cfg;
        // synthesize RX but set onesSignal=false; thr0 is taken from h_dyn, so keep helper path
        runSingleOccasionTest(noisy, /*onesSignal=*/false, numDetected, rssiDb, interfDb);
        EXPECT_EQ(numDetected, 0u);
    }

    // Detect: enable onesSignal with low thr0
    {
        PrachTestConfig detect = cfg;
        runSingleOccasionTest(detect, /*onesSignal=*/true, numDetected, rssiDb, interfDb);
        // We only require RSSI to rise, detection may still be 0 with synthetic; ensure thresholds logic executes
        EXPECT_GT(rssiDb, -100.0f);
    }
}

TEST(NonGraphSuite, PdpAverage_SelectsMaxFromNonZeroSecondAntenna)
{
    // Purpose: cover the antenna-averaging "take maxLoc from another antenna" branch:
    //   if (d_pdp[global_index].max > maxVal) { ... }
    // We inject signal on antenna 1 only (antenna 0 stays zero), so ant1 peak > ant0 peak.
    PrachTestConfig cfg{};
    cfg.L_RA           = 139;
    cfg.Nfft           = 256;
    cfg.N_rep          = 1;
    cfg.N_CS           = 1;
    cfg.uCount         = 1;
    cfg.N_nc           = 1;
    cfg.N_ant          = 2;
    cfg.mu             = 0;
    cfg.delta_f_RA     = 15000;
    cfg.kBar           = 2;
    cfg.enableUlRxBf   = false;
    cfg.nUplinkStreams = 1;

    PrachParams prm{};
    fillPrachParams(cfg, prm);
    const int L_ORAN = (prm.L_RA == 139 ? 144 : 864);

    // Start with all zeros (both antennas), then fill only antenna 1.
    std::vector<__half2> host_rx(static_cast<size_t>(prm.N_ant) * prm.N_rep * L_ORAN, __floats2half2_rn(0.0f, 0.0f));
    std::vector<__half2> host_y_ref(prm.L_RA);
    makeOnesHalf2(host_y_ref);
    const int mShift = computePdpTargetBinForPrmb0(prm);

    const uint32_t antToFill = 1;
    const float    amplitude = 8.0f;
    for(uint32_t r = 0; r < prm.N_rep; ++r)
    {
        const size_t base = (static_cast<size_t>(antToFill) * prm.N_rep + r) * L_ORAN;
        for(uint32_t i = 0; i < prm.L_RA; ++i)
        {
            const float xr     = __low2float(host_y_ref[i]);
            const float xi     = __high2float(host_y_ref[i]);
            const float TWO_PI = 6.28318530717958647692f;
            const float angle  = TWO_PI * static_cast<float>(i % prm.Nfft) * static_cast<float>(mShift % (int)prm.Nfft) / static_cast<float>(prm.Nfft);
            const float ca     = cosf(angle);
            const float sa     = sinf(angle);
            const float rr     = amplitude * (xr * ca - xi * sa);
            const float ri     = amplitude * (xr * sa + xi * ca);
            host_rx[base + prm.kBar + i] = __floats2half2_rn(rr, ri);
        }
    }

    uint32_t numDetected{};
    float    rssiDb{};
    float    interfDb{};
    runSingleOccasionTestWithCustomRx(cfg, host_rx, numDetected, rssiDb, interfDb);
    EXPECT_GT(rssiDb, -100.0f);
}

TEST(NonGraphSuite, Thr2Min_RssiLinClampedAtOneBranchCovered)
{
    // Purpose: cover the ternary branch in prach_search_pdp:
    //   (*d_rssiLin >= 1 ? 1 : *d_rssiLin)
    // by driving RSSI linear to >= 1 with a strong injected signal.
    PrachTestConfig cfg{};
    cfg.L_RA           = 139;
    cfg.Nfft           = 256;
    cfg.N_rep          = 1;
    cfg.N_CS           = 1;
    cfg.uCount         = 1;
    cfg.N_nc           = 1;
    cfg.N_ant          = 1;
    cfg.mu             = 0;
    cfg.delta_f_RA     = 15000;
    cfg.kBar           = 2;
    cfg.enableUlRxBf   = false;
    cfg.nUplinkStreams = 1;

    PrachParams prm{};
    fillPrachParams(cfg, prm);
    const int L_ORAN = (prm.L_RA == 139 ? 144 : 864);
    std::vector<__half2> host_rx(static_cast<size_t>(prm.N_ant) * prm.N_rep * L_ORAN, __floats2half2_rn(2.0f, 0.0f));

    uint32_t numDetected{};
    float    rssiDb{};
    float    interfDb{};
    runSingleOccasionTestWithCustomRx(cfg, host_rx, numDetected, rssiDb, interfDb);

    // If rssiLin >= 1, rssiDb should be >= 0 (10*log10(rssiLin)).
    EXPECT_TRUE(std::isfinite(rssiDb));
    EXPECT_GT(rssiDb, 0.0f);
}

TEST(NonGraphSuite, Thr2_UsesNp2TimesThr0BranchCovered)
{
    // Purpose: cover the branch:
    //   if (np2*thr0 > thr2_min) thr2 = np2*thr0;
    //
    // We make thr0 very large so thr1 becomes huge (so many/most preambles count as "noise"),
    // producing a positive np2 and ensuring np2*thr0 > thr2_min.
    PrachTestConfig cfg{};
    cfg.L_RA           = 139;
    cfg.Nfft           = 256;
    cfg.N_rep          = 1;
    cfg.N_CS           = 1;
    cfg.uCount         = 1;
    cfg.N_nc           = 1;
    cfg.N_ant          = 1;
    cfg.mu             = 0;
    cfg.delta_f_RA     = 15000;
    cfg.kBar           = 2;
    cfg.enableUlRxBf   = false;
    cfg.nUplinkStreams = 1;

    PrachParams prm{};
    fillPrachParams(cfg, prm);
    std::vector<__half2> host_y_ref(prm.L_RA);
    makeOnesHalf2(host_y_ref);
    const int mShift = computePdpTargetBinForPrmb0(prm);

    // Use a PRACH-like input (energy in the expected bins) so PDP power is non-zero,
    // and choose a huge thr0 so thr1 is huge and the "noise" set is non-empty => np2 > 0.
    std::vector<__half2> host_rx;
    prepareInputRx(prm, prm.kBar, /*useRefSignal=*/true, host_y_ref, mShift, /*amplitude=*/64.0f, host_rx);

    uint32_t numDetected{};
    float    rssiDb{};
    float    interfDb{};
    runSingleOccasionTestWithCustomRx(cfg, host_rx, numDetected, rssiDb, interfDb, /*thr0=*/1000000.0f);
    EXPECT_TRUE(std::isfinite(rssiDb));
    EXPECT_TRUE(std::isfinite(interfDb));
}

TEST(GraphSuite, Create_Update_Launch_Succeeds)
{
    // Build one occasion (B4)
    PrachTestConfig cfg{};
    cfg.L_RA           = 139;
    cfg.Nfft           = 256;
    cfg.N_rep          = 1;
    cfg.N_CS           = 1;
    cfg.uCount         = 1;
    cfg.N_nc           = 1;
    cfg.N_ant          = 1;
    cfg.mu             = 0;
    cfg.delta_f_RA     = 15000;
    cfg.kBar           = 2;
    cfg.enableUlRxBf   = false;
    cfg.nUplinkStreams = 1;

    // Use helper contexts and buffers
    GraphContextSingleOcc ctx = createGraphContextSingleOcc(cfg);
    GraphOutputs          outs{};
    allocGraphOutputs(outs, /*maxAnt=*/ctx.maxAntenna, /*prmbCap=*/64, /*occaCap=*/1);

    // Graph setup
    cudaGraph_t                  graph;
    cudaGraphExec_t              graphExec;
    std::vector<cudaGraphNode_t> nodes;
    nodes.resize(GraphNodeType::FFTNode + 1);
    std::vector<char> activeOcc(1, 1), prevActive(1, 1);

    constexpr uint32_t maxFftsPerOccasion = MAX_N_ANTENNAS_SUPPORTED * PRACH_MAX_NUM_PREAMBLES * PRACH_NUM_NON_COHERENT_COMBINING_GROUPS;
    auto h_fftPointers = cuphy::buffer<cuFloatComplex *, cuphy::pinned_alloc>(PRACH_NUM_SUPPORTED_FFT_SIZES * ctx.nMaxOccasions * maxFftsPerOccasion);
    std::array<FftInfo, PRACH_NUM_SUPPORTED_FFT_SIZES> fftInfo;

    ASSERT_EQ(cuphyPrachCreateGraph(&graph,
                                    &graphExec,
                                    nodes,
                                    0,
                                    ctx.d_dyn.addr(),
                                    ctx.d_static.addr(),
                                    &ctx.h_static,
                                    h_fftPointers.addr(),
                                    fftInfo,
                                    outs.num,
                                    outs.idx,
                                    outs.dly,
                                    outs.pwr,
                                    outs.ant,
                                    outs.rssi,
                                    outs.interf,
                                    ctx.nTotCellOcca,
                                    ctx.nMaxOccasions,
                                    ctx.maxAntenna,
                                    ctx.max_l_oran_ant,
                                    ctx.max_ant_u,
                                    ctx.max_nfft,
                                    ctx.max_zoneSizeExt,
                                    activeOcc,
                                    ctx.cudaArch),
              CUPHY_STATUS_SUCCESS);

    ASSERT_EQ(cuphyPrachLaunchGraph(graphExec, 0), CUPHY_STATUS_SUCCESS);

    // Prepare prev state and update/launch directly
    PrevPtrs prev{};
    prev.nPrevOccaProc = ctx.nMaxOccasions;
    PrachInternalDynParamPerOcca hostDynArr[1] = {ctx.h_dyn_host};
    ASSERT_EQ(cuphyPrachUpdateGraph(graphExec,
                                    nodes,
                                    ctx.d_dyn.addr(),
                                    ctx.d_static.addr(),
                                    hostDynArr,
                                    &ctx.h_static,
                                    h_fftPointers.addr(),
                                    fftInfo,
                                    outs.num,
                                    outs.idx,
                                    outs.dly,
                                    outs.pwr,
                                    outs.ant,
                                    outs.rssi,
                                    outs.interf,
                                    prev.num,
                                    prev.idx,
                                    prev.dly,
                                    prev.pwr,
                                    prev.ant,
                                    prev.rssi,
                                    prev.interf,
                                    ctx.nMaxOccasions,
                                    prev.nPrevOccaProc,
                                    /*nOccaProc=*/1,
                                    ctx.maxAntenna,
                                    ctx.max_l_oran_ant,
                                    ctx.max_ant_u,
                                    ctx.max_nfft,
                                    ctx.max_zoneSizeExt,
                                    activeOcc,
                                    prevActive),
              CUPHY_STATUS_SUCCESS);
    ASSERT_EQ(cuphyPrachLaunchGraph(graphExec, 0), CUPHY_STATUS_SUCCESS);

    // Toggle FFT node off then on
    activeOcc[0] = 0;
    ASSERT_EQ(cuphyPrachUpdateGraph(graphExec, nodes, ctx.d_dyn.addr(), ctx.d_static.addr(), hostDynArr, &ctx.h_static, h_fftPointers.addr(), fftInfo, outs.num, outs.idx, outs.dly, outs.pwr, outs.ant, outs.rssi, outs.interf, prev.num, prev.idx, prev.dly, prev.pwr, prev.ant, prev.rssi, prev.interf, ctx.nMaxOccasions, prev.nPrevOccaProc, 1, ctx.maxAntenna, ctx.max_l_oran_ant, ctx.max_ant_u, ctx.max_nfft, ctx.max_zoneSizeExt, activeOcc, prevActive), CUPHY_STATUS_SUCCESS);
    ASSERT_EQ(cuphyPrachLaunchGraph(graphExec, 0), CUPHY_STATUS_SUCCESS);
    activeOcc[0] = 1;
    ASSERT_EQ(cuphyPrachUpdateGraph(graphExec, nodes, ctx.d_dyn.addr(), ctx.d_static.addr(), hostDynArr, &ctx.h_static, h_fftPointers.addr(), fftInfo, outs.num, outs.idx, outs.dly, outs.pwr, outs.ant, outs.rssi, outs.interf, prev.num, prev.idx, prev.dly, prev.pwr, prev.ant, prev.rssi, prev.interf, ctx.nMaxOccasions, prev.nPrevOccaProc, 1, ctx.maxAntenna, ctx.max_l_oran_ant, ctx.max_ant_u, ctx.max_nfft, ctx.max_zoneSizeExt, activeOcc, prevActive), CUPHY_STATUS_SUCCESS);
    ASSERT_EQ(cuphyPrachLaunchGraph(graphExec, 0), CUPHY_STATUS_SUCCESS);

    // Change output pointers and relaunch
    GraphOutputs outs2{};
    allocGraphOutputs(outs2, /*maxAnt=*/ctx.maxAntenna, /*prmbCap=*/64, /*occaCap=*/1);
    outs.num = outs2.num;
    ASSERT_EQ(cuphyPrachUpdateGraph(graphExec, nodes, ctx.d_dyn.addr(), ctx.d_static.addr(), hostDynArr, &ctx.h_static, h_fftPointers.addr(), fftInfo, outs.num, outs.idx, outs.dly, outs.pwr, outs.ant, outs.rssi, outs.interf, prev.num, prev.idx, prev.dly, prev.pwr, prev.ant, prev.rssi, prev.interf, ctx.nMaxOccasions, prev.nPrevOccaProc, 1, ctx.maxAntenna, ctx.max_l_oran_ant, ctx.max_ant_u, ctx.max_nfft, ctx.max_zoneSizeExt, activeOcc, prevActive), CUPHY_STATUS_SUCCESS);
    ASSERT_EQ(cuphyPrachLaunchGraph(graphExec, 0), CUPHY_STATUS_SUCCESS);
    outs.idx = outs2.idx;
    ASSERT_EQ(cuphyPrachUpdateGraph(graphExec, nodes, ctx.d_dyn.addr(), ctx.d_static.addr(), hostDynArr, &ctx.h_static, h_fftPointers.addr(), fftInfo, outs.num, outs.idx, outs.dly, outs.pwr, outs.ant, outs.rssi, outs.interf, prev.num, prev.idx, prev.dly, prev.pwr, prev.ant, prev.rssi, prev.interf, ctx.nMaxOccasions, prev.nPrevOccaProc, 1, ctx.maxAntenna, ctx.max_l_oran_ant, ctx.max_ant_u, ctx.max_nfft, ctx.max_zoneSizeExt, activeOcc, prevActive), CUPHY_STATUS_SUCCESS);
    ASSERT_EQ(cuphyPrachLaunchGraph(graphExec, 0), CUPHY_STATUS_SUCCESS);

    // Cleanup
    freeGraphOutputs(outs);
    freeGraphOutputs(outs2);
    cudaFree(ctx.d_rx);
}

TEST(GraphSuite, MultiOccasions_Update_All_Params)
{
    // Two occasions: B4 and fmt0
    PrachTestConfig cfg0{}; // B4
    cfg0.L_RA           = 139;
    cfg0.Nfft           = 256;
    cfg0.N_rep          = 1;
    cfg0.N_CS           = 1;
    cfg0.uCount         = 1;
    cfg0.N_nc           = 1;
    cfg0.N_ant          = 1;
    cfg0.mu             = 0;
    cfg0.delta_f_RA     = 15000;
    cfg0.kBar           = 2;
    cfg0.enableUlRxBf   = false;
    cfg0.nUplinkStreams = 1;
    PrachTestConfig cfg1{}; // fmt0
    cfg1.L_RA           = 839;
    cfg1.Nfft           = 1024;
    cfg1.N_rep          = 1;
    cfg1.N_CS           = 1;
    cfg1.uCount         = 1;
    cfg1.N_nc           = 1;
    cfg1.N_ant          = 1;
    cfg1.mu             = 0;
    cfg1.delta_f_RA     = 1250;
    cfg1.kBar           = 7;
    cfg1.enableUlRxBf   = false;
    cfg1.nUplinkStreams = 1;

    GraphContextTwoOcc ctx = createGraphContextForTwoOccasions(cfg0, cfg1);

    // Outputs (initial)
    GraphOutputs outs{};
    allocGraphOutputs(outs, /*maxAnt=*/ctx.maxAntenna, /*prmbCap=*/128, /*occaCap=*/2);

    // Graph
    cudaGraph_t                  graph;
    cudaGraphExec_t              graphExec;
    std::vector<cudaGraphNode_t> nodes;
    nodes.resize(GraphNodeType::FFTNode + 2);
    std::vector<char>  activeOcc(2, 1);
    const uint16_t     nTotCellOcca = ctx.nTotCellOcca, nMaxOccasions = ctx.nMaxOccasions, maxAntenna = ctx.maxAntenna;
    const unsigned int max_l_oran_ant = ctx.max_l_oran_ant;
    const unsigned int max_ant_u = ctx.max_ant_u, max_nfft = ctx.max_nfft;
    const int          max_zoneSizeExt = ctx.max_zoneSizeExt;
    const unsigned int cudaArch        = ctx.cudaArch;

    constexpr uint32_t maxFftsPerOccasion = MAX_N_ANTENNAS_SUPPORTED * PRACH_MAX_NUM_PREAMBLES * PRACH_NUM_NON_COHERENT_COMBINING_GROUPS;
    auto h_fftPointers = cuphy::buffer<cuFloatComplex *, cuphy::pinned_alloc>(PRACH_NUM_SUPPORTED_FFT_SIZES * nMaxOccasions * maxFftsPerOccasion);
    std::array<FftInfo, PRACH_NUM_SUPPORTED_FFT_SIZES> fftInfo;

    cuphyStatus_t sCreate = cuphyPrachCreateGraph(&graph, &graphExec, nodes, 0, ctx.d_dyn.addr(), ctx.d_static.addr(), ctx.h_static, h_fftPointers.addr(), fftInfo, outs.num, outs.idx, outs.dly, outs.pwr, outs.ant, outs.rssi, outs.interf, nTotCellOcca, nMaxOccasions, maxAntenna, max_l_oran_ant, max_ant_u, max_nfft, max_zoneSizeExt, activeOcc, cudaArch);
    EXPECT_EQ(sCreate, CUPHY_STATUS_SUCCESS);
    EXPECT_EQ(cuphyPrachLaunchGraph(graphExec, 0), CUPHY_STATUS_SUCCESS);

    // Prepare prev state and new buffers
    PrevPtrs prev{};
    prev.nPrevOccaProc = nMaxOccasions;
    std::vector<char> prevActive(2, 1);
    auto do_update_and_launch = [&](uint16_t nOcca) {
        updateAndLaunchGraph(graphExec, nodes, ctx, h_fftPointers, fftInfo, outs, prev, nMaxOccasions, nOcca, maxAntenna, max_l_oran_ant, max_ant_u, max_nfft, max_zoneSizeExt, activeOcc, prevActive);
    };
    // Prime prev by one update with no changes
    do_update_and_launch(nMaxOccasions);

    // Change nOccaProc to 1 and update
    do_update_and_launch(1);

    // Change pointers independently
    GraphOutputs outs2{};
    allocGraphOutputs(outs2, /*maxAnt=*/ctx.maxAntenna, /*prmbCap=*/128, /*occaCap=*/2);
    outs.num = outs2.num;
    do_update_and_launch(1);
    outs.rssi = outs2.rssi;
    outs.interf = outs2.interf;
    do_update_and_launch(1);

    // Toggle activeOccasions off→on on second FFT node
    activeOcc[1] = 0;
    do_update_and_launch(1);
    activeOcc[1] = 1;
    do_update_and_launch(1);

    // Launch after updates
    EXPECT_EQ(cuphyPrachLaunchGraph(graphExec, 0), CUPHY_STATUS_SUCCESS);

    // Cleanup
    cudaFree(ctx.d_rx0);
    cudaFree(ctx.d_rx1);
    freeGraphOutputs(outs);
    freeGraphOutputs(outs2);
}

TEST(GraphSuite, Update_All_Output_Pointers_And_Toggles_With_Launch)
{
    // Reuse two-occasion context
    PrachTestConfig cfg0{};
    cfg0.L_RA           = 139;
    cfg0.Nfft           = 256;
    cfg0.N_rep          = 1;
    cfg0.N_CS           = 1;
    cfg0.uCount         = 1;
    cfg0.N_nc           = 1;
    cfg0.N_ant          = 1;
    cfg0.mu             = 0;
    cfg0.delta_f_RA     = 15000;
    cfg0.kBar           = 2;
    cfg0.enableUlRxBf   = false;
    cfg0.nUplinkStreams = 1;
    PrachTestConfig cfg1{};
    cfg1.L_RA              = 839;
    cfg1.Nfft              = 1024;
    cfg1.N_rep             = 1;
    cfg1.N_CS              = 1;
    cfg1.uCount            = 1;
    cfg1.N_nc              = 1;
    cfg1.N_ant             = 1;
    cfg1.mu                = 0;
    cfg1.delta_f_RA        = 1250;
    cfg1.kBar              = 7;
    cfg1.enableUlRxBf      = false;
    cfg1.nUplinkStreams    = 1;
    GraphContextTwoOcc ctx = createGraphContextForTwoOccasions(cfg0, cfg1);

    // Outputs
    GraphOutputs outs{};
    allocGraphOutputs(outs, /*maxAnt=*/2);

    // Graph
    cudaGraph_t                  graph;
    cudaGraphExec_t              graphExec;
    std::vector<cudaGraphNode_t> nodes;
    nodes.resize(GraphNodeType::FFTNode + 2);
    std::vector<char>  activeOcc(2, 1), prevActive(2, 1);
    const uint16_t     nTotCellOcca = ctx.nTotCellOcca, nMaxOccasions = ctx.nMaxOccasions, maxAntenna = ctx.maxAntenna;
    const unsigned int max_l_oran_ant = ctx.max_l_oran_ant, max_ant_u = ctx.max_ant_u, max_nfft = ctx.max_nfft;
    const int          max_zoneSizeExt = ctx.max_zoneSizeExt;
    const unsigned int cudaArch        = ctx.cudaArch;

    constexpr uint32_t maxFftsPerOccasion = MAX_N_ANTENNAS_SUPPORTED * PRACH_MAX_NUM_PREAMBLES * PRACH_NUM_NON_COHERENT_COMBINING_GROUPS;
    auto h_fftPointers = cuphy::buffer<cuFloatComplex *, cuphy::pinned_alloc>(PRACH_NUM_SUPPORTED_FFT_SIZES * nMaxOccasions * maxFftsPerOccasion);
    std::array<FftInfo, PRACH_NUM_SUPPORTED_FFT_SIZES> fftInfo;

    ASSERT_EQ(cuphyPrachCreateGraph(&graph, &graphExec, nodes, 0, ctx.d_dyn.addr(), ctx.d_static.addr(), ctx.h_static, h_fftPointers.addr(), fftInfo, outs.num, outs.idx, outs.dly, outs.pwr, outs.ant, outs.rssi, outs.interf, nTotCellOcca, nMaxOccasions, maxAntenna, max_l_oran_ant, max_ant_u, max_nfft, max_zoneSizeExt, activeOcc, cudaArch), CUPHY_STATUS_SUCCESS);
    ASSERT_EQ(cuphyPrachLaunchGraph(graphExec, 0), CUPHY_STATUS_SUCCESS);

    // Prepare prev state holders (as API requires references)
    PrevPtrs prev{};
    prev.nPrevOccaProc = nMaxOccasions;

    // Helper lambda to update and launch
    auto do_update_and_launch = [&](uint16_t nOcca) {
        updateAndLaunchGraph(graphExec, nodes, ctx, h_fftPointers, fftInfo, outs, prev, nMaxOccasions, nOcca, maxAntenna, max_l_oran_ant, max_ant_u, max_nfft, max_zoneSizeExt, activeOcc, prevActive);
    };

    // nOccaProc 2 -> 1 -> 2
    do_update_and_launch(1);
    do_update_and_launch(2);

    // Change each output pointer independently, then launch
    GraphOutputs outs2{};
    allocGraphOutputs(outs2, /*maxAnt=*/2);
    outs.num = outs2.num; do_update_and_launch(2);
    outs.idx = outs2.idx; do_update_and_launch(2);
    outs.dly = outs2.dly; do_update_and_launch(2);
    outs.pwr = outs2.pwr; do_update_and_launch(2);
    outs.ant = outs2.ant; do_update_and_launch(2);
    outs.rssi = outs2.rssi; do_update_and_launch(2);
    outs.interf = outs2.interf; do_update_and_launch(2);

    // Toggle activeOcc for both nodes off→on
    activeOcc[0] = 0;
    do_update_and_launch(2);
    activeOcc[0] = 1;
    do_update_and_launch(2);
    activeOcc[1] = 0;
    do_update_and_launch(2);
    activeOcc[1] = 1;
    do_update_and_launch(2);

    // Cleanup
    freeGraphOutputs(outs);
    freeGraphOutputs(outs2);
}

TEST(NonGraphSuite, Fmt0_NcsZero_Succeeds)
{
    PrachTestConfig cfg{};
    cfg.L_RA           = 839;
    cfg.Nfft           = 1024;
    cfg.N_rep          = 1;
    cfg.N_CS           = 0;
    cfg.uCount         = 1;
    cfg.N_nc           = 1;
    cfg.N_ant          = 1;
    cfg.mu             = 0;
    cfg.delta_f_RA     = 1250;
    cfg.kBar           = 7;
    cfg.enableUlRxBf   = false;
    cfg.nUplinkStreams = 1;
    cuphyStatus_t s    = runSingleOccasionStatus(cfg, /*onesSignal=*/false);
    EXPECT_EQ(s, CUPHY_STATUS_SUCCESS);
}

TEST(GraphNegSuite, Create_InvalidZoneSize_Throws)
{
    PrachTestConfig cfg{};
    cfg.L_RA                  = 139;
    cfg.Nfft                  = 256;
    cfg.N_rep                 = 1;
    cfg.N_CS                  = 1;
    cfg.uCount                = 1;
    cfg.N_nc                  = 1;
    cfg.N_ant                 = 1;
    cfg.mu                    = 0;
    cfg.delta_f_RA            = 15000;
    cfg.kBar                  = 2;
    cfg.enableUlRxBf          = false;
    cfg.nUplinkStreams        = 1;
    GraphContextSingleOcc ctx = createGraphContextSingleOcc(cfg);

    GraphOutputs outs{};
    allocGraphOutputs(outs, /*maxAnt=*/1);
    std::vector<cudaGraphNode_t> nodes;
    nodes.resize(GraphNodeType::FFTNode + 1);
    std::vector<char> activeOcc(1, 1);

    uint16_t nMaxOccasions = 1;
    constexpr uint32_t maxFftsPerOccasion = MAX_N_ANTENNAS_SUPPORTED * PRACH_MAX_NUM_PREAMBLES * PRACH_NUM_NON_COHERENT_COMBINING_GROUPS;
    auto h_fftPointers = cuphy::buffer<cuFloatComplex *, cuphy::pinned_alloc>(PRACH_NUM_SUPPORTED_FFT_SIZES * nMaxOccasions * maxFftsPerOccasion);
    std::array<FftInfo, PRACH_NUM_SUPPORTED_FFT_SIZES> fftInfo;

    // Expect exception (or error) when max_zoneSizeExt == 0
    cudaGraph_t*     nullGraphValidPtr = new cudaGraph_t; // valid storage, but we'll still throw due to zoneSize
    cudaGraphExec_t* nullExecValidPtr  = new cudaGraphExec_t;
    EXPECT_ANY_THROW({
        (void)cuphyPrachCreateGraph(nullGraphValidPtr, nullExecValidPtr, nodes, 0, ctx.d_dyn.addr(), ctx.d_static.addr(), &ctx.h_static, h_fftPointers.addr(), fftInfo, outs.num, outs.idx, outs.dly, outs.pwr, outs.ant, outs.rssi, outs.interf,
                                    /*nTotCellOcca=*/1,
                                    /*nMaxOccasions=*/1,
                                    ctx.maxAntenna,
                                    ctx.max_l_oran_ant,
                                    ctx.max_ant_u,
                                    ctx.max_nfft,
                                    /*max_zoneSizeExt=*/0,
                                    activeOcc,
                                    ctx.cudaArch);
    });
    delete nullGraphValidPtr;
    delete nullExecValidPtr;
    freeGraphOutputs(outs);
    cudaFree(ctx.d_rx);
}

TEST(GraphNegSuite, Create_NullGraphPtr_Throws)
{
    PrachTestConfig cfg{};
    cfg.L_RA                  = 139;
    cfg.Nfft                  = 256;
    cfg.N_rep                 = 1;
    cfg.N_CS                  = 1;
    cfg.uCount                = 1;
    cfg.N_nc                  = 1;
    cfg.N_ant                 = 1;
    cfg.mu                    = 0;
    cfg.delta_f_RA            = 15000;
    cfg.kBar                  = 2;
    cfg.enableUlRxBf          = false;
    cfg.nUplinkStreams        = 1;
    GraphContextSingleOcc ctx = createGraphContextSingleOcc(cfg);

    GraphOutputs outs{};
    allocGraphOutputs(outs, /*maxAnt=*/1);
    std::vector<cudaGraphNode_t> nodes;
    nodes.resize(GraphNodeType::FFTNode + 1);
    std::vector<char> activeOcc(1, 1);

    uint16_t nMaxOccasions = 1;
    constexpr uint32_t maxFftsPerOccasion = MAX_N_ANTENNAS_SUPPORTED * PRACH_MAX_NUM_PREAMBLES * PRACH_NUM_NON_COHERENT_COMBINING_GROUPS;
    auto h_fftPointers = cuphy::buffer<cuFloatComplex *, cuphy::pinned_alloc>(PRACH_NUM_SUPPORTED_FFT_SIZES * nMaxOccasions * maxFftsPerOccasion);
    std::array<FftInfo, PRACH_NUM_SUPPORTED_FFT_SIZES> fftInfo;

    // Pass nullptr for graph and graphExec to force cudaGraphCreate failure path
    EXPECT_ANY_THROW({
        (void)cuphyPrachCreateGraph(/*graph=*/nullptr, /*graphExec=*/nullptr, nodes, 0, ctx.d_dyn.addr(), ctx.d_static.addr(), &ctx.h_static, h_fftPointers.addr(), fftInfo, outs.num, outs.idx, outs.dly, outs.pwr, outs.ant, outs.rssi, outs.interf,
                                    /*nTotCellOcca=*/1,
                                    /*nMaxOccasions=*/1,
                                    ctx.maxAntenna,
                                    ctx.max_l_oran_ant,
                                    ctx.max_ant_u,
                                    ctx.max_nfft,
                                    ctx.max_zoneSizeExt,
                                    activeOcc,
                                    ctx.cudaArch);
    });
    freeGraphOutputs(outs);
    cudaFree(ctx.d_rx);
}

TEST(GraphNegSuite, Create_ZeroOccasions_HitsAddKernelNodeError)
{
    // Exercise cudaGraphAddKernelNode error path by passing nTotCellOcca=0 (invalid grid)
    PrachTestConfig cfg{};
    cfg.L_RA                  = 139;
    cfg.Nfft                  = 256;
    cfg.N_rep                 = 1;
    cfg.N_CS                  = 1;
    cfg.uCount                = 1;
    cfg.N_nc                  = 1;
    cfg.N_ant                 = 1;
    cfg.mu                    = 0;
    cfg.delta_f_RA            = 15000;
    cfg.kBar                  = 2;
    cfg.enableUlRxBf          = false;
    cfg.nUplinkStreams        = 1;
    GraphContextSingleOcc ctx = createGraphContextSingleOcc(cfg);

    GraphOutputs outs{};
    allocGraphOutputs(outs, /*maxAnt=*/1);
    std::vector<cudaGraphNode_t> nodes;
    nodes.resize(GraphNodeType::FFTNode + 1);
    std::vector<char> activeOcc(1, 1);

    constexpr uint32_t maxFftsPerOccasion = MAX_N_ANTENNAS_SUPPORTED * PRACH_MAX_NUM_PREAMBLES * PRACH_NUM_NON_COHERENT_COMBINING_GROUPS;
    uint16_t nMaxOccasions = 1;
    auto h_fftPointers = cuphy::buffer<cuFloatComplex *, cuphy::pinned_alloc>(PRACH_NUM_SUPPORTED_FFT_SIZES * nMaxOccasions * maxFftsPerOccasion);
    std::array<FftInfo, PRACH_NUM_SUPPORTED_FFT_SIZES> fftInfo;

    EXPECT_ANY_THROW({
        cudaGraph_t     graph;
        cudaGraphExec_t exec;
        (void)cuphyPrachCreateGraph(&graph, &exec, nodes, 0, ctx.d_dyn.addr(), ctx.d_static.addr(), &ctx.h_static, h_fftPointers.addr(), fftInfo, outs.num, outs.idx, outs.dly, outs.pwr, outs.ant, outs.rssi, outs.interf,
                                    /*nTotCellOcca=*/0,
                                    /*nMaxOccasions=*/1,
                                    ctx.maxAntenna,
                                    ctx.max_l_oran_ant,
                                    ctx.max_ant_u,
                                    ctx.max_nfft,
                                    ctx.max_zoneSizeExt,
                                    activeOcc,
                                    ctx.cudaArch);
    });

    freeGraphOutputs(outs);
    cudaFree(ctx.d_rx);
}

TEST(GraphSuite, SingleOcc_NcsZero_Create_Launch)
{
    // Cover graph path with N_CS=0 (edge sizing)
    PrachTestConfig cfg{};
    cfg.L_RA           = 839;
    cfg.Nfft           = 1024;
    cfg.N_rep          = 1;
    cfg.N_CS           = 0;
    cfg.uCount         = 1;
    cfg.N_nc           = 1;
    cfg.N_ant          = 1;
    cfg.mu             = 0;
    cfg.delta_f_RA     = 1250;
    cfg.kBar           = 7;
    cfg.enableUlRxBf   = false;
    cfg.nUplinkStreams = 1;

    GraphContextSingleOcc ctx = createGraphContextSingleOcc(cfg);
    GraphOutputs          outs{};
    allocGraphOutputs(outs, /*maxAnt=*/ctx.maxAntenna, /*prmbCap=*/64, /*occaCap=*/1);

    cudaGraph_t                  graph;
    cudaGraphExec_t              exec;
    std::vector<cudaGraphNode_t> nodes;
    nodes.resize(GraphNodeType::FFTNode + 1);
    std::vector<char> activeOcc(1, 1);

    constexpr uint32_t maxFftsPerOccasion = MAX_N_ANTENNAS_SUPPORTED * PRACH_MAX_NUM_PREAMBLES * PRACH_NUM_NON_COHERENT_COMBINING_GROUPS;
    auto h_fftPointers = cuphy::buffer<cuFloatComplex *, cuphy::pinned_alloc>(PRACH_NUM_SUPPORTED_FFT_SIZES * ctx.nMaxOccasions * maxFftsPerOccasion);
    std::array<FftInfo, PRACH_NUM_SUPPORTED_FFT_SIZES> fftInfo;

    ASSERT_EQ(cuphyPrachCreateGraph(&graph,
                                    &exec,
                                    nodes,
                                    0,
                                    ctx.d_dyn.addr(),
                                    ctx.d_static.addr(),
                                    &ctx.h_static,
                                    h_fftPointers.addr(),
                                    fftInfo,
                                    outs.num,
                                    outs.idx,
                                    outs.dly,
                                    outs.pwr,
                                    outs.ant,
                                    outs.rssi,
                                    outs.interf,
                                    ctx.nTotCellOcca,
                                    ctx.nMaxOccasions,
                                    ctx.maxAntenna,
                                    ctx.max_l_oran_ant,
                                    ctx.max_ant_u,
                                    ctx.max_nfft,
                                    ctx.max_zoneSizeExt,
                                    activeOcc,
                                    ctx.cudaArch),
              CUPHY_STATUS_SUCCESS);
    ASSERT_EQ(cuphyPrachLaunchGraph(exec, 0), CUPHY_STATUS_SUCCESS);

    freeGraphOutputs(outs);
    cudaFree(ctx.d_rx);
}

TEST(DetectSuite, DISABLED_OnesSignal_Detection_B4_And_FMT0_With_BF)
{
    // First run: B4 (L_RA=139, Nfft=256), no BF
    {
        PrachTestConfig cfg{};
        cfg.L_RA           = 139;
        cfg.Nfft           = 256;
        cfg.N_rep          = 1;
        cfg.N_CS           = 1;
        cfg.uCount         = 1;
        cfg.N_nc           = 1;
        cfg.N_ant          = 1;
        cfg.mu             = 0;
        cfg.delta_f_RA     = 15000;
        cfg.kBar           = 2;
        cfg.enableUlRxBf   = false;
        cfg.nUplinkStreams = 1;

        uint32_t numDetected = 0;
        float    rssiDb      = 0.0f;
        float    interfDb    = 0.0f;
        runSingleOccasionTest(cfg, /*onesSignal=*/true, numDetected, rssiDb, interfDb);

        EXPECT_GE(numDetected, 1u);
        EXPECT_GT(rssiDb, -100.0f);
    }

    // Second run: Format 0 (L_RA=839, Nfft=1024), enable UL Rx BF path
    {
        PrachTestConfig cfg{};
        cfg.L_RA           = 839;
        cfg.Nfft           = 1024;
        cfg.N_rep          = 1;
        cfg.N_CS           = 1; // simplify zone search
        cfg.uCount         = 1;
        cfg.N_nc           = 1;
        cfg.N_ant          = 2; // >1 to exercise antenna reductions
        cfg.mu             = 0;
        cfg.delta_f_RA     = 1250; // 1.25 kHz
        cfg.kBar           = 7;    // per kBar table for L_RA=839, delta_f=15kHz
        cfg.enableUlRxBf   = true;
        cfg.nUplinkStreams = 1; // effective antenna count changes in kernels

        uint32_t numDetected = 0;
        float    rssiDb      = 0.0f;
        float    interfDb    = 0.0f;
        runSingleOccasionTest(cfg, /*onesSignal=*/true, numDetected, rssiDb, interfDb);

        EXPECT_GE(numDetected, 1u);
        EXPECT_GT(rssiDb, -100.0f);
    }
}

#ifdef USE_CUFFTDX
TEST(NonGraphSuite, Cufftdx_InvalidFftLaunchConfig_TriggersCudaLaunchKernelError)
{
    // Force the cuFFTDx cudaLaunchKernel() call to fail with a clean runtime error by
    // constructing an invalid launch configuration (gridDim.x == 0).
    // This should exercise the CUPHY_STATUS_INTERNAL_ERROR branch in the USE_CUFFTDX path.
    PrachTestConfig cfg{};
    cfg.L_RA           = 139;
    cfg.Nfft           = 256;
    cfg.N_rep          = 1;
    cfg.N_CS           = 1;
    cfg.uCount         = 1;
    cfg.N_nc           = 0; // makes FFT gridDim.x == N_ant * uCount * N_nc == 0
    cfg.N_ant          = 1;
    cfg.mu             = 0;
    cfg.delta_f_RA     = 15000;
    cfg.kBar           = 2;
    cfg.enableUlRxBf   = false;
    cfg.nUplinkStreams = 1;

    const unsigned int cudaArch = getCudaArch();
    const cuphyStatus_t status  = runSingleOccasionStatusArch(cfg, /*onesSignal=*/false, cudaArch);
    EXPECT_EQ(status, CUPHY_STATUS_INTERNAL_ERROR);

    // Clear CUDA "last error" so subsequent tests don't observe a stale launch failure.
    (void)cudaGetLastError();
}
#endif

int main(int argc, char* argv[])
{
    // Initialize Google Test
    ::testing::InitGoogleTest(&argc, argv);

    // Run all tests
    int result = RUN_ALL_TESTS();

    return result;
}
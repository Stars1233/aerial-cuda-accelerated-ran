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
#include <cuda.h>
#include <cuda_runtime.h>
#include <memory>
#include <vector>
#include <cmath>
#include <cstring>

#include "cuphy.h"
#include "cuphy.hpp"
#include "constants.hpp"

namespace
{

struct DeviceBufferDeleter
{
    void operator()(void* p) const
    {
        if(p) cudaFree(p);
    }
};

using device_ptr = std::unique_ptr<void, DeviceBufferDeleter>;

static inline device_ptr alloc_device(size_t bytes)
{
    void* p = nullptr;
    EXPECT_EQ(cudaMalloc(&p, bytes), cudaSuccess);
    EXPECT_NE(p, nullptr);
    return device_ptr(p);
}

static inline void memset_device(void* p, int value, size_t bytes)
{
    EXPECT_EQ(cudaMemset(p, value, bytes), cudaSuccess);
}

static inline void memcpy_h2d(void* dst, const void* src, size_t bytes)
{
    EXPECT_EQ(cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice), cudaSuccess);
}

static inline void memcpy_d2h(void* dst, const void* src, size_t bytes)
{
    EXPECT_EQ(cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost), cudaSuccess);
}

// Helper to fill tensor info for 1D
static inline cuphyTensorInfo1_t make_tensor_info_1d(void* dptr, cuphyDataType_t dtype, int32_t stride0)
{
    cuphyTensorInfo1_t info{};
    info.pAddr      = dptr;
    info.elemType   = dtype;
    info.strides[0] = stride0;
    return info;
}

// Helper to fill tensor info for 2D
static inline cuphyTensorInfo2_t make_tensor_info_2d(void* dptr, cuphyDataType_t dtype, int32_t s0, int32_t s1)
{
    cuphyTensorInfo2_t info{};
    info.pAddr      = dptr;
    info.elemType   = dtype;
    info.strides[0] = s0;
    info.strides[1] = s1;
    return info;
}

// Helper to fill tensor info for 3D
static inline cuphyTensorInfo3_t make_tensor_info_3d(void* dptr, cuphyDataType_t dtype, int32_t s0, int32_t s1, int32_t s2)
{
    cuphyTensorInfo3_t info{};
    info.pAddr      = dptr;
    info.elemType   = dtype;
    info.strides[0] = s0;
    info.strides[1] = s1;
    info.strides[2] = s2;
    return info;
}

// Helper to fill tensor info for 4D
static inline cuphyTensorInfo4_t make_tensor_info_4d(void* dptr, cuphyDataType_t dtype, int32_t s0, int32_t s1, int32_t s2, int32_t s3)
{
    cuphyTensorInfo4_t info{};
    info.pAddr      = dptr;
    info.elemType   = dtype;
    info.strides[0] = s0;
    info.strides[1] = s1;
    info.strides[2] = s2;
    info.strides[3] = s3;
    return info;
}

struct TestConfig
{
    uint16_t                   nUeGrps{1};
    uint16_t                   nRxAnt{4};
    uint16_t                   nLayers{1};
    uint16_t                   nPrb{2};
    uint8_t                    nDmrsSyms{1};
    uint8_t                    dmrsMaxLen{1};
    uint8_t                    dmrsSym0{0};
    uint8_t                    enableDftSOfdm{0};
    uint8_t                    dmrsSymbolIdx{CUPHY_PUSCH_NOISE_EST_DMRS_FULL_SLOT};
    uint8_t                    nDmrsCdmGrpsNoData{1};
    cuphyPuschEqCoefAlgoType_t eqAlgo{PUSCH_EQ_ALGO_TYPE_NOISE_DIAG_MMSE};
    bool                       mismatchAntennaCounts{false}; // when true and nUeGrps>=2, group1 uses nRxAnt+4
};

struct AllocSet
{
    device_ptr dDataRx;
    device_ptr dHEst;
    device_ptr dNoiseVarPreEq;
    device_ptr dLwInv;
    device_ptr dInterCtaSyncCnt;
    size_t     dataRxBytes{0};
    size_t     hEstBytes{0};
    size_t     lwInvBytes{0};
    uint32_t   noiseVarElems{0};
    uint32_t   interCtaElems{0};
};

struct TfPrcdOverrides
{
    uint8_t  enableTfPrcd{1};
    uint8_t  optionalDftSOfdm{0};
    uint8_t  groupOrSequenceHopping{0}; // 0 none, 1 group, 2 sequence
    uint32_t puschIdentity{10};
    uint8_t  N_symb_slot{OFDM_SYMBOLS_PER_SLOT};
    uint16_t lowPaprGroupNumber{0};
    uint16_t lowPaprSequenceNumber{0};
};

static inline AllocSet allocate_buffers(const TestConfig& cfg)
{
    AllocSet       as{};
    const uint32_t NF      = cfg.nPrb * CUPHY_N_TONES_PER_PRB;
    const uint32_t ND      = OFDM_SYMBOLS_PER_SLOT;
    const uint32_t NL      = cfg.nLayers;
    const uint32_t NH      = cfg.nDmrsSyms;
    const uint16_t nGroups = cfg.nUeGrps;

    // Sum per-group sizes to avoid overlap across groups
    size_t totalDataRxElemsHalf = 0;
    size_t totalHEstElemsFloat  = 0;
    size_t totalLwInvElemsFloat = 0;
    for(uint16_t g = 0; g < nGroups; ++g)
    {
        const uint16_t nRxAntG = (cfg.mismatchAntennaCounts && g == 1) ? static_cast<uint16_t>(cfg.nRxAnt + 4) : cfg.nRxAnt;
        totalDataRxElemsHalf += static_cast<size_t>(NF) * ND * nRxAntG * 2;            // complex half
        totalHEstElemsFloat += static_cast<size_t>(nRxAntG) * NL * NF * NH * 2;        // complex float
        totalLwInvElemsFloat += static_cast<size_t>(nRxAntG) * nRxAntG * cfg.nPrb * 2; // complex float
    }

    as.dataRxBytes = totalDataRxElemsHalf * sizeof(__half);
    as.hEstBytes   = totalHEstElemsFloat * sizeof(float);
    as.lwInvBytes  = totalLwInvElemsFloat * sizeof(float);

    as.dDataRx = alloc_device(as.dataRxBytes);
    memset_device(as.dDataRx.get(), 0, as.dataRxBytes);
    as.dHEst = alloc_device(as.hEstBytes);
    memset_device(as.dHEst.get(), 0, as.hEstBytes);
    as.dLwInv = alloc_device(as.lwInvBytes);
    memset_device(as.dLwInv.get(), 0, as.lwInvBytes);

    // NoiseVar and inter-CTA are indexed by UE or UEGRP depending on build. Allocate at least nUeGrps entries.
    as.noiseVarElems  = cfg.nUeGrps;
    as.dNoiseVarPreEq = alloc_device(as.noiseVarElems * sizeof(float));
    memset_device(as.dNoiseVarPreEq.get(), 0, as.noiseVarElems * sizeof(float));
    as.interCtaElems    = cfg.nUeGrps;
    as.dInterCtaSyncCnt = alloc_device(as.interCtaElems * sizeof(uint32_t));
    memset_device(as.dInterCtaSyncCnt.get(), 0, as.interCtaElems * sizeof(uint32_t));

    return as;
}

static inline void fill_uegrp_prms(const TestConfig& cfg, const AllocSet& as, std::vector<cuphyPuschRxUeGrpPrms_t>& h_uegrps)
{
    h_uegrps.resize(cfg.nUeGrps);
    const uint32_t NF = cfg.nPrb * CUPHY_N_TONES_PER_PRB;
    const uint32_t ND = OFDM_SYMBOLS_PER_SLOT;
    const uint32_t NL = cfg.nLayers;
    const uint32_t NH = cfg.nDmrsSyms;

    // Strides for contiguous row-major tensors common by layout (per group dims embedded in strides below)
    const int32_t data_stride0 = 1;       // sc
    const int32_t data_stride1 = NF;      // sym
    const int32_t data_stride2 = NF * ND; // ant

    // Cumulative byte offsets for per-group base pointers
    size_t dataBaseOff = 0;
    size_t hBaseOff    = 0;
    size_t lwBaseOff   = 0;

    for(uint16_t g = 0; g < cfg.nUeGrps; ++g)
    {
        const uint16_t          nRxAntG = (cfg.mismatchAntennaCounts && g == 1) ? static_cast<uint16_t>(cfg.nRxAnt + 4) : cfg.nRxAnt;
        cuphyPuschRxUeGrpPrms_t prms{};
        prms.nRxAnt             = nRxAntG;
        prms.nLayers            = NL;
        prms.slotNum            = 0;
        prms.enableDftSOfdm     = cfg.enableDftSOfdm;
        prms.enableTfPrcd       = 0;
        prms.optionalDftSOfdm   = 0;
        prms.startPrb           = 0;
        prms.nPrb               = cfg.nPrb;
        prms.mu                 = 1;
        prms.nUes               = 1;
        prms.ueIdxs[0]          = g; // unique UE index per group
        prms.dmrsMaxLen         = cfg.dmrsMaxLen;
        prms.nDmrsSyms          = cfg.nDmrsSyms;
        prms.dmrsCnt            = cfg.nDmrsSyms;
        prms.dmrsSymLoc[0]      = cfg.dmrsSym0;
        prms.scid               = 0;
        prms.nDmrsCdmGrpsNoData = cfg.nDmrsCdmGrpsNoData;
        prms.eqCoeffAlgo        = cfg.eqAlgo;
        prms.dmrsPortIdxs[0]    = 0; // single layer -> port 0

        // Per-group strides
        const int32_t h_stride0 = 1;                 // ant
        const int32_t h_stride1 = nRxAntG;           // layer
        const int32_t h_stride2 = nRxAntG * NL;      // sc
        const int32_t h_stride3 = nRxAntG * NL * NF; // time

        const int32_t lw_stride0 = 1;                 // row
        const int32_t lw_stride1 = nRxAntG;           // col
        const int32_t lw_stride2 = nRxAntG * nRxAntG; // prb

        // Set per-group base pointers using cumulative offsets
        void* dataPtrG = static_cast<void*>(static_cast<char*>(as.dDataRx.get()) + dataBaseOff);
        void* hPtrG    = static_cast<void*>(static_cast<char*>(as.dHEst.get()) + hBaseOff);
        void* lwPtrG   = static_cast<void*>(static_cast<char*>(as.dLwInv.get()) + lwBaseOff);

        prms.tInfoDataRx                      = make_tensor_info_3d(dataPtrG, CUPHY_C_16F, data_stride0, data_stride1, data_stride2);
        prms.tInfoHEst                        = make_tensor_info_4d(hPtrG, CUPHY_C_32F, h_stride0, h_stride1, h_stride2, h_stride3);
        prms.tInfoLwInv                       = make_tensor_info_4d(lwPtrG, CUPHY_C_32F, lw_stride0, lw_stride1, lw_stride2, 0);
        prms.tInfoNoiseVarPreEq               = make_tensor_info_1d(as.dNoiseVarPreEq.get(), CUPHY_R_32F, 1);
        prms.tInfoNoiseIntfEstInterCtaSyncCnt = make_tensor_info_1d(as.dInterCtaSyncCnt.get(), CUPHY_R_32U, 1);

        h_uegrps[g] = prms;

        // Increment cumulative offsets for next group
        const size_t dataElemsG = static_cast<size_t>(NF) * ND * nRxAntG * 2;            // half elements
        const size_t hElemsG    = static_cast<size_t>(nRxAntG) * NL * NF * NH * 2;       // float elements
        const size_t lwElemsG   = static_cast<size_t>(nRxAntG) * nRxAntG * cfg.nPrb * 2; // float elements
        dataBaseOff += dataElemsG * sizeof(__half);
        hBaseOff += hElemsG * sizeof(float);
        lwBaseOff += lwElemsG * sizeof(float);
    }
}

static inline void fill_uegrp_prms_tprcd(const TestConfig&                     cfg,
                                         const AllocSet&                       as,
                                         std::vector<cuphyPuschRxUeGrpPrms_t>& h_uegrps,
                                         const TfPrcdOverrides&                ov)
{
    h_uegrps.resize(cfg.nUeGrps);
    const uint32_t NF = cfg.nPrb * CUPHY_N_TONES_PER_PRB;
    const uint32_t ND = OFDM_SYMBOLS_PER_SLOT;
    const uint32_t NL = cfg.nLayers;
    const uint32_t NH = cfg.nDmrsSyms;

    const int32_t data_stride0 = 1;
    const int32_t data_stride1 = NF;
    const int32_t data_stride2 = NF * ND;

    size_t dataBaseOff = 0, hBaseOff = 0, lwBaseOff = 0;
    for(uint16_t g = 0; g < cfg.nUeGrps; ++g)
    {
        const uint16_t          nRxAntG = cfg.nRxAnt;
        cuphyPuschRxUeGrpPrms_t prms{};
        prms.nRxAnt                 = nRxAntG;
        prms.nLayers                = NL;
        prms.slotNum                = 0;
        prms.enableDftSOfdm         = 1; // DFT-s-OFDM enabled
        prms.enableTfPrcd           = ov.enableTfPrcd;
        prms.optionalDftSOfdm       = ov.optionalDftSOfdm;
        prms.puschIdentity          = ov.puschIdentity;
        prms.groupOrSequenceHopping = ov.groupOrSequenceHopping;
        prms.N_symb_slot            = ov.N_symb_slot;
        prms.lowPaprGroupNumber     = ov.lowPaprGroupNumber;
        prms.lowPaprSequenceNumber  = ov.lowPaprSequenceNumber;
        prms.startPrb               = 0;
        prms.nPrb                   = cfg.nPrb;
        prms.mu                     = 1;
        prms.nUes                   = 1;
        prms.ueIdxs[0]              = g;
        prms.dmrsMaxLen             = cfg.dmrsMaxLen;
        prms.nDmrsSyms              = cfg.nDmrsSyms;
        prms.dmrsCnt                = cfg.nDmrsSyms;
        prms.dmrsSymLoc[0]          = cfg.dmrsSym0;
        prms.scid                   = 0;
        prms.nDmrsCdmGrpsNoData     = cfg.nDmrsCdmGrpsNoData;
        prms.eqCoeffAlgo            = cfg.eqAlgo;
        prms.dmrsPortIdxs[0]        = 0;

        const int32_t h_stride0  = 1;
        const int32_t h_stride1  = nRxAntG;
        const int32_t h_stride2  = nRxAntG * NL;
        const int32_t h_stride3  = nRxAntG * NL * NF;
        const int32_t lw_stride0 = 1;
        const int32_t lw_stride1 = nRxAntG;
        const int32_t lw_stride2 = nRxAntG * nRxAntG;

        void* dataPtrG = static_cast<void*>(static_cast<char*>(as.dDataRx.get()) + dataBaseOff);
        void* hPtrG    = static_cast<void*>(static_cast<char*>(as.dHEst.get()) + hBaseOff);
        void* lwPtrG   = static_cast<void*>(static_cast<char*>(as.dLwInv.get()) + lwBaseOff);

        prms.tInfoDataRx                      = make_tensor_info_3d(dataPtrG, CUPHY_C_16F, data_stride0, data_stride1, data_stride2);
        prms.tInfoHEst                        = make_tensor_info_4d(hPtrG, CUPHY_C_32F, h_stride0, h_stride1, h_stride2, h_stride3);
        prms.tInfoLwInv                       = make_tensor_info_4d(lwPtrG, CUPHY_C_32F, lw_stride0, lw_stride1, lw_stride2, 0);
        prms.tInfoNoiseVarPreEq               = make_tensor_info_1d(as.dNoiseVarPreEq.get(), CUPHY_R_32F, 1);
        prms.tInfoNoiseIntfEstInterCtaSyncCnt = make_tensor_info_1d(as.dInterCtaSyncCnt.get(), CUPHY_R_32U, 1);

        h_uegrps[g] = prms;

        const size_t dataElemsG = static_cast<size_t>(NF) * ND * nRxAntG * 2;
        const size_t hElemsG    = static_cast<size_t>(nRxAntG) * NL * NF * NH * 2;
        const size_t lwElemsG   = static_cast<size_t>(nRxAntG) * nRxAntG * cfg.nPrb * 2;
        dataBaseOff += dataElemsG * sizeof(__half);
        hBaseOff += hElemsG * sizeof(float);
        lwBaseOff += lwElemsG * sizeof(float);
    }
}

static inline float read_noise_var_db_idx(const AllocSet& as, uint16_t idx)
{
    float val_db = -999.0f;
    memcpy_d2h(&val_db, static_cast<const float*>(as.dNoiseVarPreEq.get()) + idx, sizeof(float));
    return val_db;
}

static inline float read_noise_var_db_for_group(uint16_t ueIdx)
{
    // Helper placeholder, kept for API symmetry
    return static_cast<float>(ueIdx);
}

// Returns noise dB for group g, trying UE index first, then group index
static inline float get_group_noise_db(uint16_t g, const AllocSet& as)
{
    float v = read_noise_var_db_idx(as, g);
    return v;
}

static inline void run_kernel_and_sync(const cuphyPuschRxNoiseIntfEstLaunchCfgs_t& launchCfgs, cudaStream_t stream)
{
    const CUDA_KERNEL_NODE_PARAMS& kp = launchCfgs.cfgs[0].kernelNodeParamsDriver;
    EXPECT_NE(kp.func, nullptr);
    CUresult launchResult = cuLaunchKernel(
        kp.func,
        kp.gridDimX,
        kp.gridDimY,
        kp.gridDimZ,
        kp.blockDimX,
        kp.blockDimY,
        kp.blockDimZ,
        kp.sharedMemBytes,
        static_cast<CUstream>(stream),
        kp.kernelParams,
        kp.extra);
    EXPECT_EQ(launchResult, CUDA_SUCCESS);
    EXPECT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
}

static inline float expected_noise_db()
{
    const float expected_lin = CUPHY_NOISE_REGULARIZER;
    return 10.0f * log10f(expected_lin) + 0.5f;
}

static inline void validate_outputs(const TestConfig& cfg, const AllocSet& as, bool expectLwInvComputed)
{
    const float exp_db = expected_noise_db();
    for(uint16_t g = 0; g < cfg.nUeGrps; ++g)
    {
        const float got_db = get_group_noise_db(g, as);
        EXPECT_TRUE(std::isfinite(got_db));
        EXPECT_NEAR(got_db, exp_db, 0.3f);
    }
    const uint32_t     nant = cfg.nRxAnt;
    const uint32_t     prb  = 0;
    std::vector<float> lw_host(nant * nant * 2, 0.0f);
    const size_t       elems_per_prb = static_cast<size_t>(nant) * nant * 2;
    memcpy_d2h(lw_host.data(), static_cast<const float*>(as.dLwInv.get()) + prb * elems_per_prb, elems_per_prb * sizeof(float));
    auto at = [&](uint32_t r, uint32_t c) -> std::pair<float, float> {
        const size_t idx = (static_cast<size_t>(c) * nant + r) * 2;
        return {lw_host[idx + 0], lw_host[idx + 1]};
    };
    if(!expectLwInvComputed)
    {
        float max_abs = 0.0f;
        for(size_t i = 0; i < lw_host.size(); ++i) max_abs = std::max(max_abs, std::fabs(lw_host[i]));
        EXPECT_LT(max_abs, 1e-6f);
    }
    else
    {
        for(uint32_t r = 0; r < nant; ++r)
        {
            for(uint32_t c = 0; c < nant; ++c)
            {
                auto [re, im] = at(r, c);
                EXPECT_TRUE(std::isfinite(re));
                EXPECT_TRUE(std::isfinite(im));
                if(r < c)
                {
                    EXPECT_NEAR(re, 0.0f, 1e-3f);
                    EXPECT_NEAR(im, 0.0f, 1e-3f);
                }
            }
            auto [dRe, dIm] = at(r, r);
            EXPECT_GE(dRe, 0.0f);
            EXPECT_NEAR(dIm, 0.0f, 1e-6f);
        }
    }
}

static inline void validate_noise_equal_across_groups(const TestConfig& cfg, const AllocSet& as, float max_db_diff)
{
    if(cfg.nUeGrps < 2) return;
    const float ref = get_group_noise_db(0, as);
    EXPECT_TRUE(std::isfinite(ref));
    for(uint16_t g = 1; g < cfg.nUeGrps; ++g)
    {
        const float got = get_group_noise_db(g, as);
        EXPECT_TRUE(std::isfinite(got));
        EXPECT_NEAR(got, ref, max_db_diff);
    }
}

static inline void run_and_check(const TestConfig& cfg, bool expectLwInvComputed)
{
    AllocSet                             as = allocate_buffers(cfg);
    std::vector<cuphyPuschRxUeGrpPrms_t> h_uegrps;
    fill_uegrp_prms(cfg, as, h_uegrps);
    device_ptr d_uegrps = alloc_device(sizeof(cuphyPuschRxUeGrpPrms_t) * cfg.nUeGrps);
    memcpy_h2d(d_uegrps.get(), h_uegrps.data(), sizeof(cuphyPuschRxUeGrpPrms_t) * cfg.nUeGrps);
    size_t dynSize{}, dynAlign{};
    ASSERT_EQ(cuphyPuschRxNoiseIntfEstGetDescrInfo(&dynSize, &dynAlign), CUPHY_STATUS_SUCCESS);
    std::vector<uint8_t> h_dyn(dynSize, 0);
    device_ptr           d_dyn = alloc_device(dynSize);
    memcpy_h2d(d_dyn.get(), h_dyn.data(), dynSize);
    cuphyPuschRxNoiseIntfEstLaunchCfgs_t launchCfgs{};
    launchCfgs.nCfgs = 1;
    cuphyPuschRxNoiseIntfEstHndl_t hndl{};
    cuphy::stream                  localStrm;
    ASSERT_EQ(cuphyCreatePuschRxNoiseIntfEst(&hndl), CUPHY_STATUS_SUCCESS);
    ASSERT_EQ(cuphySetupPuschRxNoiseIntfEst(
                  hndl,
                  h_uegrps.data(),
                  static_cast<cuphyPuschRxUeGrpPrms_t*>(d_uegrps.get()),
                  cfg.nUeGrps,
                  cfg.nPrb,
                  cfg.enableDftSOfdm,
                  cfg.dmrsSymbolIdx,
                  0,
                  h_dyn.data(),
                  d_dyn.get(),
                  &launchCfgs,
                  localStrm.handle(),
                  0),
              CUPHY_STATUS_SUCCESS);
    run_kernel_and_sync(launchCfgs, localStrm.handle());
    validate_outputs(cfg, as, expectLwInvComputed);
    EXPECT_EQ(cuphyDestroyPuschRxNoiseIntfEst(hndl), CUPHY_STATUS_SUCCESS);
}

static inline void run_and_check_tprcd(const TestConfig& cfg, const TfPrcdOverrides& ov, bool expectLwInvComputed)
{
    AllocSet                             as = allocate_buffers(cfg);
    std::vector<cuphyPuschRxUeGrpPrms_t> h_uegrps;
    fill_uegrp_prms_tprcd(cfg, as, h_uegrps, ov);
    device_ptr d_uegrps = alloc_device(sizeof(cuphyPuschRxUeGrpPrms_t) * cfg.nUeGrps);
    memcpy_h2d(d_uegrps.get(), h_uegrps.data(), sizeof(cuphyPuschRxUeGrpPrms_t) * cfg.nUeGrps);
    size_t dynSize{}, dynAlign{};
    ASSERT_EQ(cuphyPuschRxNoiseIntfEstGetDescrInfo(&dynSize, &dynAlign), CUPHY_STATUS_SUCCESS);
    std::vector<uint8_t> h_dyn(dynSize, 0);
    device_ptr           d_dyn = alloc_device(dynSize);
    memcpy_h2d(d_dyn.get(), h_dyn.data(), dynSize);
    cuphyPuschRxNoiseIntfEstLaunchCfgs_t launchCfgs{};
    launchCfgs.nCfgs = 1;
    cuphyPuschRxNoiseIntfEstHndl_t hndl{};
    cuphy::stream                  localStrm;
    ASSERT_EQ(cuphyCreatePuschRxNoiseIntfEst(&hndl), CUPHY_STATUS_SUCCESS);
    ASSERT_EQ(cuphySetupPuschRxNoiseIntfEst(
                  hndl,
                  h_uegrps.data(),
                  static_cast<cuphyPuschRxUeGrpPrms_t*>(d_uegrps.get()),
                  cfg.nUeGrps,
                  cfg.nPrb,
                  cfg.enableDftSOfdm,
                  cfg.dmrsSymbolIdx,
                  0,
                  h_dyn.data(),
                  d_dyn.get(),
                  &launchCfgs,
                  localStrm.handle(),
                  0),
              CUPHY_STATUS_SUCCESS);
    run_kernel_and_sync(launchCfgs, localStrm.handle());
    validate_outputs(cfg, as, expectLwInvComputed);
    EXPECT_EQ(cuphyDestroyPuschRxNoiseIntfEst(hndl), CUPHY_STATUS_SUCCESS);
}

static inline void fill_uegrp_prms_var_prb(const TestConfig&                     cfg,
                                           const AllocSet&                       as,
                                           std::vector<cuphyPuschRxUeGrpPrms_t>& h_uegrps,
                                           const std::vector<uint16_t>&          groupPrbs)
{
    const uint16_t nGroups = cfg.nUeGrps;
    ASSERT_EQ(groupPrbs.size(), static_cast<size_t>(nGroups));
    h_uegrps.resize(nGroups);
    const uint32_t NFmax        = cfg.nPrb * CUPHY_N_TONES_PER_PRB;
    const uint32_t ND           = OFDM_SYMBOLS_PER_SLOT;
    const uint32_t NL           = cfg.nLayers;
    const uint32_t NH           = cfg.nDmrsSyms;
    const int32_t  data_stride0 = 1;
    const int32_t  data_stride1 = NFmax;
    const int32_t  data_stride2 = NFmax * ND;
    size_t         dataBaseOff = 0, hBaseOff = 0, lwBaseOff = 0;
    for(uint16_t g = 0; g < nGroups; ++g)
    {
        const uint16_t          nRxAntG = cfg.nRxAnt;
        cuphyPuschRxUeGrpPrms_t prms{};
        prms.nRxAnt                           = nRxAntG;
        prms.nLayers                          = NL;
        prms.slotNum                          = 0;
        prms.enableDftSOfdm                   = cfg.enableDftSOfdm;
        prms.enableTfPrcd                     = 0;
        prms.optionalDftSOfdm                 = 0;
        prms.startPrb                         = 0;
        prms.nPrb                             = groupPrbs[g]; // per-group PRB size to trigger early-exit blocks
        prms.mu                               = 1;
        prms.nUes                             = 1;
        prms.ueIdxs[0]                        = g;
        prms.dmrsMaxLen                       = cfg.dmrsMaxLen;
        prms.nDmrsSyms                        = cfg.nDmrsSyms;
        prms.dmrsCnt                          = cfg.nDmrsSyms;
        prms.dmrsSymLoc[0]                    = cfg.dmrsSym0;
        prms.scid                             = 0;
        prms.nDmrsCdmGrpsNoData               = cfg.nDmrsCdmGrpsNoData;
        prms.eqCoeffAlgo                      = cfg.eqAlgo;
        prms.dmrsPortIdxs[0]                  = 0;
        const int32_t h_stride0               = 1;
        const int32_t h_stride1               = nRxAntG;
        const int32_t h_stride2               = nRxAntG * NL;
        const int32_t h_stride3               = nRxAntG * NL * NFmax;
        const int32_t lw_stride0              = 1;
        const int32_t lw_stride1              = nRxAntG;
        const int32_t lw_stride2              = nRxAntG * nRxAntG;
        void*         dataPtrG                = static_cast<void*>(static_cast<char*>(as.dDataRx.get()) + dataBaseOff);
        void*         hPtrG                   = static_cast<void*>(static_cast<char*>(as.dHEst.get()) + hBaseOff);
        void*         lwPtrG                  = static_cast<void*>(static_cast<char*>(as.dLwInv.get()) + lwBaseOff);
        prms.tInfoDataRx                      = make_tensor_info_3d(dataPtrG, CUPHY_C_16F, data_stride0, data_stride1, data_stride2);
        prms.tInfoHEst                        = make_tensor_info_4d(hPtrG, CUPHY_C_32F, h_stride0, h_stride1, h_stride2, h_stride3);
        prms.tInfoLwInv                       = make_tensor_info_4d(lwPtrG, CUPHY_C_32F, lw_stride0, lw_stride1, lw_stride2, 0);
        prms.tInfoNoiseVarPreEq               = make_tensor_info_1d(as.dNoiseVarPreEq.get(), CUPHY_R_32F, 1);
        prms.tInfoNoiseIntfEstInterCtaSyncCnt = make_tensor_info_1d(as.dInterCtaSyncCnt.get(), CUPHY_R_32U, 1);
        h_uegrps[g]                           = prms;
        const size_t dataElemsG               = static_cast<size_t>(NFmax) * ND * nRxAntG * 2;
        const size_t hElemsG                  = static_cast<size_t>(nRxAntG) * NL * NFmax * NH * 2;
        const size_t lwElemsG                 = static_cast<size_t>(nRxAntG) * nRxAntG * cfg.nPrb * 2;
        dataBaseOff += dataElemsG * sizeof(__half);
        hBaseOff += hElemsG * sizeof(float);
        lwBaseOff += lwElemsG * sizeof(float);
    }
}

static inline void run_and_check_var_prb(const TestConfig& cfg, const std::vector<uint16_t>& groupPrbs, bool expectLwInvComputed)
{
    AllocSet                             as = allocate_buffers(cfg);
    std::vector<cuphyPuschRxUeGrpPrms_t> h_uegrps;
    fill_uegrp_prms_var_prb(cfg, as, h_uegrps, groupPrbs);
    device_ptr d_uegrps = alloc_device(sizeof(cuphyPuschRxUeGrpPrms_t) * cfg.nUeGrps);
    memcpy_h2d(d_uegrps.get(), h_uegrps.data(), sizeof(cuphyPuschRxUeGrpPrms_t) * cfg.nUeGrps);
    size_t dynSize{}, dynAlign{};
    ASSERT_EQ(cuphyPuschRxNoiseIntfEstGetDescrInfo(&dynSize, &dynAlign), CUPHY_STATUS_SUCCESS);
    std::vector<uint8_t> h_dyn(dynSize, 0);
    device_ptr           d_dyn = alloc_device(dynSize);
    memcpy_h2d(d_dyn.get(), h_dyn.data(), dynSize);
    cuphyPuschRxNoiseIntfEstLaunchCfgs_t launchCfgs{};
    launchCfgs.nCfgs = 1;
    cuphyPuschRxNoiseIntfEstHndl_t hndl{};
    cuphy::stream                  localStrm;
    ASSERT_EQ(cuphyCreatePuschRxNoiseIntfEst(&hndl), CUPHY_STATUS_SUCCESS);
    ASSERT_EQ(cuphySetupPuschRxNoiseIntfEst(
                  hndl,
                  h_uegrps.data(),
                  static_cast<cuphyPuschRxUeGrpPrms_t*>(d_uegrps.get()),
                  cfg.nUeGrps,
                  cfg.nPrb,
                  cfg.enableDftSOfdm,
                  cfg.dmrsSymbolIdx,
                  0,
                  h_dyn.data(),
                  d_dyn.get(),
                  &launchCfgs,
                  localStrm.handle(),
                  0),
              CUPHY_STATUS_SUCCESS);
    run_kernel_and_sync(launchCfgs, localStrm.handle());
    validate_outputs(cfg, as, expectLwInvComputed);
    EXPECT_EQ(cuphyDestroyPuschRxNoiseIntfEst(hndl), CUPHY_STATUS_SUCCESS);
}

static inline TestConfig make_cfg(uint8_t                    enableDftSOfdm,
                                  uint8_t                    dmrsSymbolIdx,
                                  cuphyPuschEqCoefAlgoType_t eqAlgo,
                                  uint8_t                    nDmrsCdmGrpsNoData,
                                  uint16_t                   nRxAnt  = 4,
                                  uint16_t                   nUeGrps = 1,
                                  uint16_t                   nPrb    = 2)
{
    TestConfig cfg{};
    cfg.enableDftSOfdm     = enableDftSOfdm;
    cfg.dmrsSymbolIdx      = dmrsSymbolIdx;
    cfg.eqAlgo             = eqAlgo;
    cfg.nDmrsCdmGrpsNoData = nDmrsCdmGrpsNoData;
    cfg.nRxAnt             = nRxAnt;
    cfg.nUeGrps            = nUeGrps;
    cfg.nPrb               = nPrb;
    return cfg;
}

// Forward declarations for helpers defined later in this file.
static inline void run_and_check_with_ports(TestConfig cfg, const std::vector<uint8_t>& layerPorts, bool expectLwInvComputed);

class PuschNoiseIntfEstTest : public ::testing::Test {
protected:
    cuphy::stream cuStrm;
};

TEST_F(PuschNoiseIntfEstTest, NoDft_FullSlot_DiagOnly_OneGroup)
{
    TestConfig cfg{};
    cfg.enableDftSOfdm     = 0;
    cfg.dmrsSymbolIdx      = CUPHY_PUSCH_NOISE_EST_DMRS_FULL_SLOT;
    cfg.eqAlgo             = PUSCH_EQ_ALGO_TYPE_NOISE_DIAG_MMSE;
    cfg.nDmrsCdmGrpsNoData = 1;

    AllocSet as = allocate_buffers(cfg);

    std::vector<cuphyPuschRxUeGrpPrms_t> h_uegrps;
    fill_uegrp_prms(cfg, as, h_uegrps);
    device_ptr d_uegrps = alloc_device(sizeof(cuphyPuschRxUeGrpPrms_t) * cfg.nUeGrps);
    memcpy_h2d(d_uegrps.get(), h_uegrps.data(), sizeof(cuphyPuschRxUeGrpPrms_t) * cfg.nUeGrps);

    // Descriptors
    size_t dynSize{}, dynAlign{};
    ASSERT_EQ(cuphyPuschRxNoiseIntfEstGetDescrInfo(&dynSize, &dynAlign), CUPHY_STATUS_SUCCESS);
    std::vector<uint8_t> h_dyn(dynSize, 0);
    device_ptr           d_dyn = alloc_device(dynSize);
    memcpy_h2d(d_dyn.get(), h_dyn.data(), dynSize);

    // Launch config
    cuphyPuschRxNoiseIntfEstLaunchCfgs_t launchCfgs{};
    launchCfgs.nCfgs = 1;

    // Create and setup
    cuphyPuschRxNoiseIntfEstHndl_t hndl{};
    ASSERT_EQ(cuphyCreatePuschRxNoiseIntfEst(&hndl), CUPHY_STATUS_SUCCESS);
    ASSERT_EQ(cuphySetupPuschRxNoiseIntfEst(
                  hndl,
                  h_uegrps.data(),
                  static_cast<cuphyPuschRxUeGrpPrms_t*>(d_uegrps.get()),
                  cfg.nUeGrps,
                  cfg.nPrb,
                  cfg.enableDftSOfdm,
                  cfg.dmrsSymbolIdx,
                  0,
                  h_dyn.data(),
                  d_dyn.get(),
                  &launchCfgs,
                  cuStrm.handle(),
                  0),
              CUPHY_STATUS_SUCCESS);

    run_kernel_and_sync(launchCfgs, cuStrm.handle());

    validate_outputs(cfg, as, /*expectLwInvComputed=*/false);

    EXPECT_EQ(cuphyDestroyPuschRxNoiseIntfEst(hndl), CUPHY_STATUS_SUCCESS);
}

TEST_F(PuschNoiseIntfEstTest, NoDft_FullSlot_DiagOnly_LargeGrid_Batches2PrbPerCta)
{
    // Cover accumulation across multiple PRBs within a single CTA in the no-DFT kernel.
    // This forces kernel selection to choose N_PRB_PER_THRD_BLK==2 by making:
    //   max_grid_size = nMaxPrb * nUeGrps >= SMALL_GRID_THRESHOLD (1600).
    // With N_PRB_PER_THRD_BLK==2, nPrbThisThrdBlk becomes 2 and the loop
    //   for (p = 1; p < nPrbThisThrdBlk; ++p) { sumNoisePwrAllPrbs += shRwwTrace[p]; }
    // executes (line 1073 in the VectorCAST-instrumented kernel source).
    TestConfig cfg = make_cfg(/*enableDftSOfdm=*/0,
                              /*dmrsSymbolIdx=*/CUPHY_PUSCH_NOISE_EST_DMRS_FULL_SLOT,
                              /*eqAlgo=*/PUSCH_EQ_ALGO_TYPE_NOISE_DIAG_MMSE,
                              /*nDmrsCdmGrpsNoData=*/1,
                              /*nRxAnt=*/4,
                              /*nUeGrps=*/1,
                              /*nPrb=*/1600);
    run_and_check(cfg, /*expectLwInvComputed=*/false);
}

TEST_F(PuschNoiseIntfEstTest, NoDft_FullSlot_DiagOnly_nRxAntGt32_Forces1PrbPerCta)
{
    // Cover the kernel selection branch:
    //   if (nRxAnt > 32) return 1;  (GetKernelNumPrbsPerCta)
    // This must be exercised with enableDftSOfdm==0 so num_prbs_per_cta is honored.
    TestConfig cfg = make_cfg(/*enableDftSOfdm=*/0,
                              /*dmrsSymbolIdx=*/CUPHY_PUSCH_NOISE_EST_DMRS_FULL_SLOT,
                              /*eqAlgo=*/PUSCH_EQ_ALGO_TYPE_NOISE_DIAG_MMSE,
                              /*nDmrsCdmGrpsNoData=*/1,
                              /*nRxAnt=*/36,
                              /*nUeGrps=*/1,
                              /*nPrb=*/2);
    run_and_check(cfg, /*expectLwInvComputed=*/false);
}

TEST_F(PuschNoiseIntfEstTest, Dft_FullSlot_DiagOnly_OneGroup)
{
    TestConfig cfg{};
    cfg.enableDftSOfdm     = 1;
    cfg.dmrsSymbolIdx      = CUPHY_PUSCH_NOISE_EST_DMRS_FULL_SLOT;
    cfg.eqAlgo             = PUSCH_EQ_ALGO_TYPE_NOISE_DIAG_MMSE;
    cfg.nDmrsCdmGrpsNoData = 2; // alternate path (even subcarriers)

    AllocSet                             as = allocate_buffers(cfg);
    std::vector<cuphyPuschRxUeGrpPrms_t> h_uegrps;
    fill_uegrp_prms(cfg, as, h_uegrps);
    device_ptr d_uegrps = alloc_device(sizeof(cuphyPuschRxUeGrpPrms_t) * cfg.nUeGrps);
    memcpy_h2d(d_uegrps.get(), h_uegrps.data(), sizeof(cuphyPuschRxUeGrpPrms_t) * cfg.nUeGrps);

    size_t dynSize{}, dynAlign{};
    ASSERT_EQ(cuphyPuschRxNoiseIntfEstGetDescrInfo(&dynSize, &dynAlign), CUPHY_STATUS_SUCCESS);
    std::vector<uint8_t> h_dyn(dynSize, 0);
    device_ptr           d_dyn = alloc_device(dynSize);
    memcpy_h2d(d_dyn.get(), h_dyn.data(), dynSize);

    cuphyPuschRxNoiseIntfEstLaunchCfgs_t launchCfgs{};
    launchCfgs.nCfgs = 1;

    cuphyPuschRxNoiseIntfEstHndl_t hndl{};
    ASSERT_EQ(cuphyCreatePuschRxNoiseIntfEst(&hndl), CUPHY_STATUS_SUCCESS);
    ASSERT_EQ(cuphySetupPuschRxNoiseIntfEst(
                  hndl,
                  h_uegrps.data(),
                  static_cast<cuphyPuschRxUeGrpPrms_t*>(d_uegrps.get()),
                  cfg.nUeGrps,
                  cfg.nPrb,
                  cfg.enableDftSOfdm,
                  cfg.dmrsSymbolIdx,
                  0,
                  h_dyn.data(),
                  d_dyn.get(),
                  &launchCfgs,
                  cuStrm.handle(),
                  0),
              CUPHY_STATUS_SUCCESS);

    run_kernel_and_sync(launchCfgs, cuStrm.handle());
    validate_outputs(cfg, as, /*expectLwInvComputed=*/false);
    EXPECT_EQ(cuphyDestroyPuschRxNoiseIntfEst(hndl), CUPHY_STATUS_SUCCESS);
}

TEST_F(PuschNoiseIntfEstTest, NoDft_AdditionalPos0_Shrink_RBLW_Covariance)
{
    TestConfig cfg{};
    cfg.enableDftSOfdm     = 0;
    cfg.dmrsSymbolIdx      = CUPHY_PUSCH_NOISE_EST_DMRS_ADDITIONAL_POS_0;
    cfg.eqAlgo             = PUSCH_EQ_ALGO_TYPE_MMSE_IRC_SHRINK_RBLW; // triggers covariance + shrink
    cfg.nDmrsCdmGrpsNoData = 1;

    AllocSet                             as = allocate_buffers(cfg);
    std::vector<cuphyPuschRxUeGrpPrms_t> h_uegrps;
    fill_uegrp_prms(cfg, as, h_uegrps);
    device_ptr d_uegrps = alloc_device(sizeof(cuphyPuschRxUeGrpPrms_t) * cfg.nUeGrps);
    memcpy_h2d(d_uegrps.get(), h_uegrps.data(), sizeof(cuphyPuschRxUeGrpPrms_t) * cfg.nUeGrps);

    size_t dynSize{}, dynAlign{};
    ASSERT_EQ(cuphyPuschRxNoiseIntfEstGetDescrInfo(&dynSize, &dynAlign), CUPHY_STATUS_SUCCESS);
    std::vector<uint8_t> h_dyn(dynSize, 0);
    device_ptr           d_dyn = alloc_device(dynSize);
    memcpy_h2d(d_dyn.get(), h_dyn.data(), dynSize);

    cuphyPuschRxNoiseIntfEstLaunchCfgs_t launchCfgs{};
    launchCfgs.nCfgs = 1;

    cuphyPuschRxNoiseIntfEstHndl_t hndl{};
    ASSERT_EQ(cuphyCreatePuschRxNoiseIntfEst(&hndl), CUPHY_STATUS_SUCCESS);
    ASSERT_EQ(cuphySetupPuschRxNoiseIntfEst(
                  hndl,
                  h_uegrps.data(),
                  static_cast<cuphyPuschRxUeGrpPrms_t*>(d_uegrps.get()),
                  cfg.nUeGrps,
                  cfg.nPrb,
                  cfg.enableDftSOfdm,
                  cfg.dmrsSymbolIdx,
                  0,
                  h_dyn.data(),
                  d_dyn.get(),
                  &launchCfgs,
                  cuStrm.handle(),
                  0),
              CUPHY_STATUS_SUCCESS);

    run_kernel_and_sync(launchCfgs, cuStrm.handle());
    validate_outputs(cfg, as, /*expectLwInvComputed=*/true);
    EXPECT_EQ(cuphyDestroyPuschRxNoiseIntfEst(hndl), CUPHY_STATUS_SUCCESS);
}

TEST_F(PuschNoiseIntfEstTest, NoDft_FullSlot_DiagOnly_TwoGroups_MismatchAntenna_GenericPath)
{
    TestConfig cfg{};
    cfg.nUeGrps               = 2;
    cfg.mismatchAntennaCounts = true; // forces generic kernel path
    cfg.enableDftSOfdm        = 0;
    // Use additional pos 0 and covariance shrinkage for robust per-group finalization
    cfg.dmrsSymbolIdx      = CUPHY_PUSCH_NOISE_EST_DMRS_ADDITIONAL_POS_0;
    cfg.eqAlgo             = PUSCH_EQ_ALGO_TYPE_MMSE_IRC_SHRINK_OAS;
    cfg.nDmrsCdmGrpsNoData = 1;

    AllocSet                             as = allocate_buffers(cfg);
    std::vector<cuphyPuschRxUeGrpPrms_t> h_uegrps;
    fill_uegrp_prms(cfg, as, h_uegrps);
    device_ptr d_uegrps = alloc_device(sizeof(cuphyPuschRxUeGrpPrms_t) * cfg.nUeGrps);
    memcpy_h2d(d_uegrps.get(), h_uegrps.data(), sizeof(cuphyPuschRxUeGrpPrms_t) * cfg.nUeGrps);

    size_t dynSize{}, dynAlign{};
    ASSERT_EQ(cuphyPuschRxNoiseIntfEstGetDescrInfo(&dynSize, &dynAlign), CUPHY_STATUS_SUCCESS);
    std::vector<uint8_t> h_dyn(dynSize, 0);
    device_ptr           d_dyn = alloc_device(dynSize);
    memcpy_h2d(d_dyn.get(), h_dyn.data(), dynSize);

    cuphyPuschRxNoiseIntfEstLaunchCfgs_t launchCfgs{};
    launchCfgs.nCfgs = 1;

    cuphyPuschRxNoiseIntfEstHndl_t hndl{};
    ASSERT_EQ(cuphyCreatePuschRxNoiseIntfEst(&hndl), CUPHY_STATUS_SUCCESS);
    ASSERT_EQ(cuphySetupPuschRxNoiseIntfEst(
                  hndl,
                  h_uegrps.data(),
                  static_cast<cuphyPuschRxUeGrpPrms_t*>(d_uegrps.get()),
                  cfg.nUeGrps,
                  cfg.nPrb,
                  cfg.enableDftSOfdm,
                  cfg.dmrsSymbolIdx,
                  0,
                  h_dyn.data(),
                  d_dyn.get(),
                  &launchCfgs,
                  cuStrm.handle(),
                  0),
              CUPHY_STATUS_SUCCESS);

    run_kernel_and_sync(launchCfgs, cuStrm.handle());
    validate_outputs(cfg, as, /*expectLwInvComputed=*/true);
    validate_noise_equal_across_groups(cfg, as, 1e-3f);
    EXPECT_EQ(cuphyDestroyPuschRxNoiseIntfEst(hndl), CUPHY_STATUS_SUCCESS);
}

// New tests to cover tfPrcd optional DFT-s-OFDM and group/sequence hopping branches
TEST_F(PuschNoiseIntfEstTest, Dft_TfPrcd_OptionalLowPapr_UsesUV)
{
    TestConfig cfg{};
    cfg.enableDftSOfdm     = 1;
    cfg.dmrsSymbolIdx      = CUPHY_PUSCH_NOISE_EST_DMRS_FULL_SLOT;
    cfg.eqAlgo             = PUSCH_EQ_ALGO_TYPE_NOISE_DIAG_MMSE;
    cfg.nDmrsCdmGrpsNoData = 1;
    cfg.nPrb               = 13; // ensure M_ZC > 72 when needed

    AllocSet                             as = allocate_buffers(cfg);
    std::vector<cuphyPuschRxUeGrpPrms_t> h_uegrps;
    TfPrcdOverrides                      ov{};
    ov.enableTfPrcd          = 1;
    ov.optionalDftSOfdm      = 1;
    ov.lowPaprGroupNumber    = 3;
    ov.lowPaprSequenceNumber = 1;
    fill_uegrp_prms_tprcd(cfg, as, h_uegrps, ov);
    device_ptr d_uegrps = alloc_device(sizeof(cuphyPuschRxUeGrpPrms_t) * cfg.nUeGrps);
    memcpy_h2d(d_uegrps.get(), h_uegrps.data(), sizeof(cuphyPuschRxUeGrpPrms_t) * cfg.nUeGrps);

    size_t dynSize{}, dynAlign{};
    ASSERT_EQ(cuphyPuschRxNoiseIntfEstGetDescrInfo(&dynSize, &dynAlign), CUPHY_STATUS_SUCCESS);
    std::vector<uint8_t> h_dyn(dynSize, 0);
    device_ptr           d_dyn = alloc_device(dynSize);
    memcpy_h2d(d_dyn.get(), h_dyn.data(), dynSize);

    cuphyPuschRxNoiseIntfEstLaunchCfgs_t launchCfgs{};
    launchCfgs.nCfgs = 1;

    cuphyPuschRxNoiseIntfEstHndl_t hndl{};
    ASSERT_EQ(cuphyCreatePuschRxNoiseIntfEst(&hndl), CUPHY_STATUS_SUCCESS);
    ASSERT_EQ(cuphySetupPuschRxNoiseIntfEst(
                  hndl,
                  h_uegrps.data(),
                  static_cast<cuphyPuschRxUeGrpPrms_t*>(d_uegrps.get()),
                  cfg.nUeGrps,
                  cfg.nPrb,
                  cfg.enableDftSOfdm,
                  cfg.dmrsSymbolIdx,
                  0,
                  h_dyn.data(),
                  d_dyn.get(),
                  &launchCfgs,
                  cuStrm.handle(),
                  0),
              CUPHY_STATUS_SUCCESS);

    run_kernel_and_sync(launchCfgs, cuStrm.handle());
    validate_outputs(cfg, as, /*expectLwInvComputed=*/false);
    EXPECT_EQ(cuphyDestroyPuschRxNoiseIntfEst(hndl), CUPHY_STATUS_SUCCESS);
}

TEST_F(PuschNoiseIntfEstTest, Dft_TfPrcd_GroupSequenceHopping_1_2)
{
    TestConfig cfg{};
    cfg.enableDftSOfdm     = 1;
    cfg.dmrsSymbolIdx      = CUPHY_PUSCH_NOISE_EST_DMRS_FULL_SLOT;
    cfg.eqAlgo             = PUSCH_EQ_ALGO_TYPE_NOISE_DIAG_MMSE;
    cfg.nDmrsCdmGrpsNoData = 1;
    cfg.nPrb               = 13; // M_ZC = 6 * 13 > 72

    AllocSet                             as = allocate_buffers(cfg);
    std::vector<cuphyPuschRxUeGrpPrms_t> h_uegrps_g1;
    {
        TfPrcdOverrides ov{};
        ov.enableTfPrcd           = 1;
        ov.optionalDftSOfdm       = 0;
        ov.groupOrSequenceHopping = 1;
        ov.puschIdentity          = 100;
        ov.N_symb_slot            = OFDM_SYMBOLS_PER_SLOT;
        fill_uegrp_prms_tprcd(cfg, as, h_uegrps_g1, ov);
    }
    device_ptr d_uegrps_g1 = alloc_device(sizeof(cuphyPuschRxUeGrpPrms_t) * cfg.nUeGrps);
    memcpy_h2d(d_uegrps_g1.get(), h_uegrps_g1.data(), sizeof(cuphyPuschRxUeGrpPrms_t) * cfg.nUeGrps);

    size_t dynSize{}, dynAlign{};
    ASSERT_EQ(cuphyPuschRxNoiseIntfEstGetDescrInfo(&dynSize, &dynAlign), CUPHY_STATUS_SUCCESS);
    std::vector<uint8_t> h_dyn(dynSize, 0);
    device_ptr           d_dyn = alloc_device(dynSize);
    memcpy_h2d(d_dyn.get(), h_dyn.data(), dynSize);

    cuphyPuschRxNoiseIntfEstLaunchCfgs_t launchCfgs{};
    launchCfgs.nCfgs = 1;

    cuphyPuschRxNoiseIntfEstHndl_t hndl{};
    ASSERT_EQ(cuphyCreatePuschRxNoiseIntfEst(&hndl), CUPHY_STATUS_SUCCESS);
    ASSERT_EQ(cuphySetupPuschRxNoiseIntfEst(
                  hndl,
                  h_uegrps_g1.data(),
                  static_cast<cuphyPuschRxUeGrpPrms_t*>(d_uegrps_g1.get()),
                  cfg.nUeGrps,
                  cfg.nPrb,
                  cfg.enableDftSOfdm,
                  cfg.dmrsSymbolIdx,
                  0,
                  h_dyn.data(),
                  d_dyn.get(),
                  &launchCfgs,
                  cuStrm.handle(),
                  0),
              CUPHY_STATUS_SUCCESS);
    run_kernel_and_sync(launchCfgs, cuStrm.handle());
    validate_outputs(cfg, as, /*expectLwInvComputed=*/false);
    EXPECT_EQ(cuphyDestroyPuschRxNoiseIntfEst(hndl), CUPHY_STATUS_SUCCESS);

    // Now groupOrSequenceHopping == 2
    std::vector<cuphyPuschRxUeGrpPrms_t> h_uegrps_g2;
    {
        TfPrcdOverrides ov{};
        ov.enableTfPrcd           = 1;
        ov.optionalDftSOfdm       = 0;
        ov.groupOrSequenceHopping = 2; // exercise v computation branch
        ov.puschIdentity          = 77;
        ov.N_symb_slot            = OFDM_SYMBOLS_PER_SLOT;
        fill_uegrp_prms_tprcd(cfg, as, h_uegrps_g2, ov);
    }
    device_ptr d_uegrps_g2 = alloc_device(sizeof(cuphyPuschRxUeGrpPrms_t) * cfg.nUeGrps);
    memcpy_h2d(d_uegrps_g2.get(), h_uegrps_g2.data(), sizeof(cuphyPuschRxUeGrpPrms_t) * cfg.nUeGrps);

    ASSERT_EQ(cuphyCreatePuschRxNoiseIntfEst(&hndl), CUPHY_STATUS_SUCCESS);
    ASSERT_EQ(cuphySetupPuschRxNoiseIntfEst(
                  hndl,
                  h_uegrps_g2.data(),
                  static_cast<cuphyPuschRxUeGrpPrms_t*>(d_uegrps_g2.get()),
                  cfg.nUeGrps,
                  cfg.nPrb,
                  cfg.enableDftSOfdm,
                  cfg.dmrsSymbolIdx,
                  0,
                  h_dyn.data(),
                  d_dyn.get(),
                  &launchCfgs,
                  cuStrm.handle(),
                  0),
              CUPHY_STATUS_SUCCESS);
    run_kernel_and_sync(launchCfgs, cuStrm.handle());
    validate_outputs(cfg, as, /*expectLwInvComputed=*/false);
    EXPECT_EQ(cuphyDestroyPuschRxNoiseIntfEst(hndl), CUPHY_STATUS_SUCCESS);
}

TEST_F(PuschNoiseIntfEstTest, NoDft_Covariance_AllSc_CDMGRPS2)
{
    TestConfig cfg{};
    cfg.enableDftSOfdm     = 0;
    cfg.dmrsSymbolIdx      = CUPHY_PUSCH_NOISE_EST_DMRS_ADDITIONAL_POS_0;
    cfg.eqAlgo             = PUSCH_EQ_ALGO_TYPE_MMSE_IRC_SHRINK_OAS;
    cfg.nDmrsCdmGrpsNoData = 2; // triggers scIdx increment by 1 (all tones)
    run_and_check(cfg, /*expectLwInvComputed=*/true);
}

TEST_F(PuschNoiseIntfEstTest, NoDft_FullSlot_Covariance_EveryOtherSc_CDMGRPS1)
{
    TestConfig cfg{};
    cfg.enableDftSOfdm     = 0;
    cfg.dmrsSymbolIdx      = CUPHY_PUSCH_NOISE_EST_DMRS_FULL_SLOT;
    cfg.eqAlgo             = PUSCH_EQ_ALGO_TYPE_MMSE_IRC_SHRINK_RBLW;
    cfg.nDmrsCdmGrpsNoData = 1; // triggers scIdx += 2 branch
    run_and_check(cfg, /*expectLwInvComputed=*/true);
}

TEST_F(PuschNoiseIntfEstTest, NoDft_KernelSelect_Case8_Covariance_MultiPrb)
{
    // Cover covariance accumulation with 8-antenna specialization (switch case 8) and multiple PRBs
    TestConfig cfg = make_cfg(/*enableDftSOfdm=*/0,
                              /*dmrsSymbolIdx=*/CUPHY_PUSCH_NOISE_EST_DMRS_FULL_SLOT,
                              /*eqAlgo=*/PUSCH_EQ_ALGO_TYPE_MMSE_IRC_SHRINK_RBLW,
                              /*nDmrsCdmGrpsNoData=*/2,
                              /*nRxAnt=*/8,
                              /*nUeGrps=*/1,
                              /*nPrb=*/3);
    run_and_check(cfg, /*expectLwInvComputed=*/true);
}

TEST_F(PuschNoiseIntfEstTest, NoDft_KernelSelect_Case16_AdditionalPos0_Covariance)
{
    // Hit non-DFT kernel selection: dmrsSymbolIdx==ADDITIONAL_POS_0 and 16-antenna specialization (switch case 16)
    TestConfig cfg = make_cfg(/*enableDftSOfdm=*/0,
                              /*dmrsSymbolIdx=*/CUPHY_PUSCH_NOISE_EST_DMRS_ADDITIONAL_POS_0,
                              /*eqAlgo=*/PUSCH_EQ_ALGO_TYPE_MMSE_IRC_SHRINK_RBLW,
                              /*nDmrsCdmGrpsNoData=*/2,
                              /*nRxAnt=*/16,
                              /*nUeGrps=*/1,
                              /*nPrb=*/3);
    run_and_check(cfg, /*expectLwInvComputed=*/true);
}

TEST_F(PuschNoiseIntfEstTest, Dft_Covariance_RBLW_CDMGRPS1)
{
    // DFT kernel, shrinkage RBLW, every-other-tone accumulation branch
    TestConfig      cfg = make_cfg(/*enableDftSOfdm=*/1,
                              /*dmrsSymbolIdx=*/CUPHY_PUSCH_NOISE_EST_DMRS_FULL_SLOT,
                              /*eqAlgo=*/PUSCH_EQ_ALGO_TYPE_MMSE_IRC_SHRINK_RBLW,
                              /*nDmrsCdmGrpsNoData=*/1,
                              /*nRxAnt=*/4,
                              /*nUeGrps=*/1,
                              /*nPrb=*/3);
    TfPrcdOverrides ov{};
    ov.enableTfPrcd = 0; // use scrambling DMRS
    run_and_check_tprcd(cfg, ov, /*expectLwInvComputed=*/true);
}

TEST_F(PuschNoiseIntfEstTest, Dft_Covariance_OAS_CDMGRPS2)
{
    // DFT kernel, shrinkage OAS, all-tone accumulation branch
    TestConfig      cfg = make_cfg(/*enableDftSOfdm=*/1,
                              /*dmrsSymbolIdx=*/CUPHY_PUSCH_NOISE_EST_DMRS_FULL_SLOT,
                              /*eqAlgo=*/PUSCH_EQ_ALGO_TYPE_MMSE_IRC_SHRINK_OAS,
                              /*nDmrsCdmGrpsNoData=*/2,
                              /*nRxAnt=*/4,
                              /*nUeGrps=*/1,
                              /*nPrb=*/3);
    TfPrcdOverrides ov{};
    ov.enableTfPrcd = 0;
    run_and_check_tprcd(cfg, ov, /*expectLwInvComputed=*/true);
}

TEST_F(PuschNoiseIntfEstTest, NoDft_EarlyExit_ThreadBlocks_PerGroupPrb)
{
    // Grid sized by nMaxPrb causes early-return for smaller nPrb group
    TestConfig            cfg       = make_cfg(/*enableDftSOfdm=*/0,
                              /*dmrsSymbolIdx=*/CUPHY_PUSCH_NOISE_EST_DMRS_FULL_SLOT,
                              /*eqAlgo=*/PUSCH_EQ_ALGO_TYPE_NOISE_DIAG_MMSE,
                              /*nDmrsCdmGrpsNoData=*/2,
                              /*nRxAnt=*/4,
                              /*nUeGrps=*/2,
                              /*nPrb=*/4); // max PRB across groups
    std::vector<uint16_t> groupPrbs = {4, 1};
    run_and_check_var_prb(cfg, groupPrbs, /*expectLwInvComputed=*/false);
}

TEST_F(PuschNoiseIntfEstTest, Dft_EarlyExit_ThreadBlocks_PerGroupPrb)
{
    // Same early-exit scenario, but under DFT kernel
    TestConfig            cfg       = make_cfg(/*enableDftSOfdm=*/1,
                              /*dmrsSymbolIdx=*/CUPHY_PUSCH_NOISE_EST_DMRS_FULL_SLOT,
                              /*eqAlgo=*/PUSCH_EQ_ALGO_TYPE_NOISE_DIAG_MMSE,
                              /*nDmrsCdmGrpsNoData=*/1,
                              /*nRxAnt=*/4,
                              /*nUeGrps=*/2,
                              /*nPrb=*/4);
    std::vector<uint16_t> groupPrbs = {4, 2};
    run_and_check_var_prb(cfg, groupPrbs, /*expectLwInvComputed=*/false);
}

TEST_F(PuschNoiseIntfEstTest, Dft_AdditionalPos0_Sets_nDmrsSyms_to_MaxLen)
{
    // Exercise branch where DMRS_SYMBOL_IDX==ADDITIONAL_POS_0 sets nDmrsSyms = dmrsMaxLen
    TestConfig cfg = make_cfg(/*enableDftSOfdm=*/1,
                              /*dmrsSymbolIdx=*/CUPHY_PUSCH_NOISE_EST_DMRS_ADDITIONAL_POS_0,
                              /*eqAlgo=*/PUSCH_EQ_ALGO_TYPE_NOISE_DIAG_MMSE,
                              /*nDmrsCdmGrpsNoData=*/1,
                              /*nRxAnt=*/4,
                              /*nUeGrps=*/1,
                              /*nPrb=*/2);
    cfg.dmrsMaxLen = 2; // ensure loop iterates over dmrsMaxLen
    TfPrcdOverrides ov{};
    ov.enableTfPrcd = 0; // scrambling-based DMRS
    run_and_check_tprcd(cfg, ov, /*expectLwInvComputed=*/false);
}

TEST_F(PuschNoiseIntfEstTest, NoDft_AdditionalPos0_MatchedAnt16_Covariance)
{
    // Hit non-DFT path, dmrsSymbolIdx==ADDITIONAL_POS_0, antenna specialization case 16
    TestConfig cfg = make_cfg(/*enableDftSOfdm=*/0,
                              /*dmrsSymbolIdx=*/CUPHY_PUSCH_NOISE_EST_DMRS_ADDITIONAL_POS_0,
                              /*eqAlgo=*/PUSCH_EQ_ALGO_TYPE_MMSE_IRC_SHRINK_RBLW,
                              /*nDmrsCdmGrpsNoData=*/2,
                              /*nRxAnt=*/16,
                              /*nUeGrps=*/1,
                              /*nPrb=*/3);
    run_and_check(cfg, /*expectLwInvComputed=*/true);
}

TEST_F(PuschNoiseIntfEstTest, NoDft_AdditionalPos0_Sets_nDmrsSyms_to_MaxLen)
{
    // Exercise same nDmrsSyms = dmrsMaxLen branch under non-DFT kernel
    TestConfig cfg = make_cfg(/*enableDftSOfdm=*/0,
                              /*dmrsSymbolIdx=*/CUPHY_PUSCH_NOISE_EST_DMRS_ADDITIONAL_POS_0,
                              /*eqAlgo=*/PUSCH_EQ_ALGO_TYPE_NOISE_DIAG_MMSE,
                              /*nDmrsCdmGrpsNoData=*/1,
                              /*nRxAnt=*/4,
                              /*nUeGrps=*/1,
                              /*nPrb=*/2);
    cfg.dmrsMaxLen = 2;
    run_and_check(cfg, /*expectLwInvComputed=*/false);
}

TEST_F(PuschNoiseIntfEstTest, NoDft_DefaultKernel_AllAntennaCountsMatch_UnsupportedCount12)
{
    // allAntennaCountsMatch true, but unsupported nRxAnt (12) should go to default (generic) kernel
    TestConfig cfg = make_cfg(/*enableDftSOfdm=*/0,
                              /*dmrsSymbolIdx=*/CUPHY_PUSCH_NOISE_EST_DMRS_FULL_SLOT,
                              /*eqAlgo=*/PUSCH_EQ_ALGO_TYPE_NOISE_DIAG_MMSE,
                              /*nDmrsCdmGrpsNoData=*/2,
                              /*nRxAnt=*/12,
                              /*nUeGrps=*/1,
                              /*nPrb=*/2);
    run_and_check(cfg, /*expectLwInvComputed=*/false);
}

TEST_F(PuschNoiseIntfEstTest, NoDft_KernelSelect_Case8_FullSlot)
{
    // allAntennaCountsMatch=true, nRxAnt=8 should pick the 8-antenna specialization for FULL_SLOT
    TestConfig cfg = make_cfg(/*enableDftSOfdm=*/0,
                              /*dmrsSymbolIdx=*/CUPHY_PUSCH_NOISE_EST_DMRS_FULL_SLOT,
                              /*eqAlgo=*/PUSCH_EQ_ALGO_TYPE_NOISE_DIAG_MMSE,
                              /*nDmrsCdmGrpsNoData=*/2,
                              /*nRxAnt=*/8,
                              /*nUeGrps=*/1,
                              /*nPrb=*/2);
    run_and_check(cfg, /*expectLwInvComputed=*/false);
}

TEST_F(PuschNoiseIntfEstTest, NoDft_KernelSelect_Case8_AdditionalPos0)
{
    // Same specialization (8) for ADDITIONAL_POS_0
    TestConfig cfg = make_cfg(/*enableDftSOfdm=*/0,
                              /*dmrsSymbolIdx=*/CUPHY_PUSCH_NOISE_EST_DMRS_ADDITIONAL_POS_0,
                              /*eqAlgo=*/PUSCH_EQ_ALGO_TYPE_NOISE_DIAG_MMSE,
                              /*nDmrsCdmGrpsNoData=*/1,
                              /*nRxAnt=*/8,
                              /*nUeGrps=*/1,
                              /*nPrb=*/2);
    run_and_check(cfg, /*expectLwInvComputed=*/false);
}

TEST_F(PuschNoiseIntfEstTest, NoDft_KernelSelect_Default_FullSlot)
{
    // Unsupported 12-antenna configuration selects default generic kernel for FULL_SLOT
    TestConfig cfg = make_cfg(/*enableDftSOfdm=*/0,
                              /*dmrsSymbolIdx=*/CUPHY_PUSCH_NOISE_EST_DMRS_FULL_SLOT,
                              /*eqAlgo=*/PUSCH_EQ_ALGO_TYPE_NOISE_DIAG_MMSE,
                              /*nDmrsCdmGrpsNoData=*/2,
                              /*nRxAnt=*/12,
                              /*nUeGrps=*/1,
                              /*nPrb=*/2);
    run_and_check(cfg, /*expectLwInvComputed=*/false);
}

TEST_F(PuschNoiseIntfEstTest, NoDft_KernelSelect_Default_AdditionalPos0)
{
    // Unsupported 12-antenna configuration selects default generic kernel for ADDITIONAL_POS_0
    TestConfig cfg = make_cfg(/*enableDftSOfdm=*/0,
                              /*dmrsSymbolIdx=*/CUPHY_PUSCH_NOISE_EST_DMRS_ADDITIONAL_POS_0,
                              /*eqAlgo=*/PUSCH_EQ_ALGO_TYPE_NOISE_DIAG_MMSE,
                              /*nDmrsCdmGrpsNoData=*/1,
                              /*nRxAnt=*/12,
                              /*nUeGrps=*/1,
                              /*nPrb=*/2);
    run_and_check(cfg, /*expectLwInvComputed=*/false);
}

TEST_F(PuschNoiseIntfEstTest, NoDft_KernelSelect_Case16_FullSlot)
{
    // allAntennaCountsMatch=true with nRxAnt=16 should pick the 16-antenna specialization for FULL_SLOT
    TestConfig cfg = make_cfg(/*enableDftSOfdm=*/0,
                              /*dmrsSymbolIdx=*/CUPHY_PUSCH_NOISE_EST_DMRS_FULL_SLOT,
                              /*eqAlgo=*/PUSCH_EQ_ALGO_TYPE_NOISE_DIAG_MMSE,
                              /*nDmrsCdmGrpsNoData=*/2,
                              /*nRxAnt=*/16,
                              /*nUeGrps=*/1,
                              /*nPrb=*/3);
    run_and_check(cfg, /*expectLwInvComputed=*/false);
}

TEST_F(PuschNoiseIntfEstTest, NoDft_DmrsCoverCode_FOCC_OddPort)
{
    // Cover DMRS cover-code branches:
    //   enableFOCC = (portIdx & PORT_IDX_FOCC_MSK) ? true : false;
    //   fOCC = (enableFOCC && (dmrsGridToneIdx & 0x1)) ? -1 : 1;
    // Using an odd DMRS port enables FOCC; across tones, dmrsGridToneIdx parity varies,
    // exercising both fOCC == -1 and fOCC == +1.
    TestConfig cfg = make_cfg(/*enableDftSOfdm=*/0,
                              /*dmrsSymbolIdx=*/CUPHY_PUSCH_NOISE_EST_DMRS_FULL_SLOT,
                              /*eqAlgo=*/PUSCH_EQ_ALGO_TYPE_NOISE_DIAG_MMSE,
                              /*nDmrsCdmGrpsNoData=*/2,
                              /*nRxAnt=*/4,
                              /*nUeGrps=*/1,
                              /*nPrb=*/2);
    run_and_check_with_ports(cfg, /*layerPorts=*/{1}, /*expectLwInvComputed=*/false);
}

TEST_F(PuschNoiseIntfEstTest, NoDft_DmrsCoverCode_TOCC_Port4_MaxLen2)
{
    // Cover DMRS cover-code branches:
    //   enableTOCC = (portIdx & PORT_IDX_TOCC_MSK) ? true : false;
    //   tOCC = (enableTOCC && (dmrsMaxLen == 2 && dmrsSymIdx % 2 == 1)) ? -1 : 1;
    // Using a port in [4..7] enables TOCC. With dmrsMaxLen==2 and a loop over dmrsSymIdx==0,1,
    // we exercise both tOCC == +1 (dmrsSymIdx==0) and tOCC == -1 (dmrsSymIdx==1).
    TestConfig cfg = make_cfg(/*enableDftSOfdm=*/0,
                              /*dmrsSymbolIdx=*/CUPHY_PUSCH_NOISE_EST_DMRS_ADDITIONAL_POS_0,
                              /*eqAlgo=*/PUSCH_EQ_ALGO_TYPE_NOISE_DIAG_MMSE,
                              /*nDmrsCdmGrpsNoData=*/2,
                              /*nRxAnt=*/4,
                              /*nUeGrps=*/1,
                              /*nPrb=*/2);
    cfg.dmrsMaxLen = 2;
    cfg.nDmrsSyms  = 2;
    run_and_check_with_ports(cfg, /*layerPorts=*/{4}, /*expectLwInvComputed=*/false);
}

TEST_F(PuschNoiseIntfEstTest, NoDft_KernelSelect_Case4_FullSlot)
{
    // allAntennaCountsMatch=true with nRxAnt=4 should pick the 4-antenna specialization for FULL_SLOT
    TestConfig cfg = make_cfg(/*enableDftSOfdm=*/0,
                              /*dmrsSymbolIdx=*/CUPHY_PUSCH_NOISE_EST_DMRS_FULL_SLOT,
                              /*eqAlgo=*/PUSCH_EQ_ALGO_TYPE_NOISE_DIAG_MMSE,
                              /*nDmrsCdmGrpsNoData=*/2,
                              /*nRxAnt=*/4,
                              /*nUeGrps=*/1,
                              /*nPrb=*/2);
    run_and_check(cfg, /*expectLwInvComputed=*/false);
}

static inline void fill_uegrp_prms_with_ports(const TestConfig&                     cfg,
                                              const AllocSet&                       as,
                                              std::vector<cuphyPuschRxUeGrpPrms_t>& h_uegrps,
                                              const std::vector<uint8_t>&           layerPorts)
{
    h_uegrps.resize(cfg.nUeGrps);
    const uint32_t NF           = cfg.nPrb * CUPHY_N_TONES_PER_PRB;
    const uint32_t ND           = OFDM_SYMBOLS_PER_SLOT;
    const uint32_t NL           = static_cast<uint32_t>(layerPorts.size());
    const uint32_t NH           = cfg.nDmrsSyms;
    const int32_t  data_stride0 = 1;
    const int32_t  data_stride1 = NF;
    const int32_t  data_stride2 = NF * ND;
    size_t         dataBaseOff = 0, hBaseOff = 0, lwBaseOff = 0;
    for(uint16_t g = 0; g < cfg.nUeGrps; ++g)
    {
        cuphyPuschRxUeGrpPrms_t prms{};
        prms.nRxAnt             = cfg.nRxAnt;
        prms.nLayers            = NL;
        prms.slotNum            = 0;
        prms.enableDftSOfdm     = 0;
        prms.enableTfPrcd       = 0;
        prms.optionalDftSOfdm   = 0;
        prms.startPrb           = 0;
        prms.nPrb               = cfg.nPrb;
        prms.mu                 = 1;
        prms.nUes               = 1;
        prms.ueIdxs[0]          = g;
        prms.dmrsMaxLen         = cfg.dmrsMaxLen;
        prms.nDmrsSyms          = cfg.nDmrsSyms;
        prms.dmrsCnt            = cfg.nDmrsSyms;
        prms.dmrsSymLoc[0]      = cfg.dmrsSym0;
        prms.scid               = 0;
        prms.nDmrsCdmGrpsNoData = cfg.nDmrsCdmGrpsNoData;
        prms.eqCoeffAlgo        = cfg.eqAlgo;
        for(uint32_t l = 0; l < NL && l < MAX_N_LAYERS_PUSCH; ++l) prms.dmrsPortIdxs[l] = layerPorts[l];
        const int32_t h_stride0               = 1;
        const int32_t h_stride1               = cfg.nRxAnt;
        const int32_t h_stride2               = cfg.nRxAnt * NL;
        const int32_t h_stride3               = cfg.nRxAnt * NL * NF;
        const int32_t lw_stride0              = 1;
        const int32_t lw_stride1              = cfg.nRxAnt;
        const int32_t lw_stride2              = cfg.nRxAnt * cfg.nRxAnt;
        void*         dataPtrG                = static_cast<void*>(static_cast<char*>(as.dDataRx.get()) + dataBaseOff);
        void*         hPtrG                   = static_cast<void*>(static_cast<char*>(as.dHEst.get()) + hBaseOff);
        void*         lwPtrG                  = static_cast<void*>(static_cast<char*>(as.dLwInv.get()) + lwBaseOff);
        prms.tInfoDataRx                      = make_tensor_info_3d(dataPtrG, CUPHY_C_16F, data_stride0, data_stride1, data_stride2);
        prms.tInfoHEst                        = make_tensor_info_4d(hPtrG, CUPHY_C_32F, h_stride0, h_stride1, h_stride2, h_stride3);
        prms.tInfoLwInv                       = make_tensor_info_4d(lwPtrG, CUPHY_C_32F, lw_stride0, lw_stride1, lw_stride2, 0);
        prms.tInfoNoiseVarPreEq               = make_tensor_info_1d(as.dNoiseVarPreEq.get(), CUPHY_R_32F, 1);
        prms.tInfoNoiseIntfEstInterCtaSyncCnt = make_tensor_info_1d(as.dInterCtaSyncCnt.get(), CUPHY_R_32U, 1);
        h_uegrps[g]                           = prms;
        const size_t dataElemsG               = static_cast<size_t>(NF) * ND * cfg.nRxAnt * 2;
        const size_t hElemsG                  = static_cast<size_t>(cfg.nRxAnt) * NL * NF * NH * 2;
        const size_t lwElemsG                 = static_cast<size_t>(cfg.nRxAnt) * cfg.nRxAnt * cfg.nPrb * 2;
        dataBaseOff += dataElemsG * sizeof(__half);
        hBaseOff += hElemsG * sizeof(float);
        lwBaseOff += lwElemsG * sizeof(float);
    }
}

static inline void run_and_check_with_ports(TestConfig cfg, const std::vector<uint8_t>& layerPorts, bool expectLwInvComputed)
{
    AllocSet as = allocate_buffers(cfg);
    cfg.nLayers = static_cast<uint16_t>(layerPorts.size());
    std::vector<cuphyPuschRxUeGrpPrms_t> h_uegrps;
    fill_uegrp_prms_with_ports(cfg, as, h_uegrps, layerPorts);
    device_ptr d_uegrps = alloc_device(sizeof(cuphyPuschRxUeGrpPrms_t) * cfg.nUeGrps);
    memcpy_h2d(d_uegrps.get(), h_uegrps.data(), sizeof(cuphyPuschRxUeGrpPrms_t) * cfg.nUeGrps);
    size_t dynSize{}, dynAlign{};
    ASSERT_EQ(cuphyPuschRxNoiseIntfEstGetDescrInfo(&dynSize, &dynAlign), CUPHY_STATUS_SUCCESS);
    std::vector<uint8_t> h_dyn(dynSize, 0);
    device_ptr           d_dyn = alloc_device(dynSize);
    memcpy_h2d(d_dyn.get(), h_dyn.data(), dynSize);
    cuphyPuschRxNoiseIntfEstLaunchCfgs_t launchCfgs{};
    launchCfgs.nCfgs = 1;
    cuphyPuschRxNoiseIntfEstHndl_t hndl{};
    cuphy::stream                  localStrm;
    ASSERT_EQ(cuphyCreatePuschRxNoiseIntfEst(&hndl), CUPHY_STATUS_SUCCESS);
    ASSERT_EQ(cuphySetupPuschRxNoiseIntfEst(
                  hndl,
                  h_uegrps.data(),
                  static_cast<cuphyPuschRxUeGrpPrms_t*>(d_uegrps.get()),
                  cfg.nUeGrps,
                  cfg.nPrb,
                  /*enableDftSOfdm*/ 0,
                  cfg.dmrsSymbolIdx,
                  0,
                  h_dyn.data(),
                  d_dyn.get(),
                  &launchCfgs,
                  localStrm.handle(),
                  0),
              CUPHY_STATUS_SUCCESS);
    run_kernel_and_sync(launchCfgs, localStrm.handle());
    validate_outputs(cfg, as, expectLwInvComputed);
    EXPECT_EQ(cuphyDestroyPuschRxNoiseIntfEst(hndl), CUPHY_STATUS_SUCCESS);
}

//-------------------------------------------------------------------------
// Setup error-path coverage helpers and tests
//-------------------------------------------------------------------------
struct SetupCtx
{
    AllocSet                             as;
    std::vector<cuphyPuschRxUeGrpPrms_t> h_uegrps;
    device_ptr                           d_uegrps{nullptr};
    std::vector<uint8_t>                 h_dyn;
    device_ptr                           d_dyn{nullptr};
    cuphyPuschRxNoiseIntfEstLaunchCfgs_t launchCfgs{};
    cuphyPuschRxNoiseIntfEstHndl_t       hndl{};
    cuphy::stream                        strm;
};

static inline void build_setup_ctx(const TestConfig& cfg, SetupCtx& ctx)
{
    ctx.as = allocate_buffers(cfg);
    fill_uegrp_prms(cfg, ctx.as, ctx.h_uegrps);
    ctx.d_uegrps = alloc_device(sizeof(cuphyPuschRxUeGrpPrms_t) * cfg.nUeGrps);
    memcpy_h2d(ctx.d_uegrps.get(), ctx.h_uegrps.data(), sizeof(cuphyPuschRxUeGrpPrms_t) * cfg.nUeGrps);
    size_t dynSize{}, dynAlign{};
    ASSERT_EQ(cuphyPuschRxNoiseIntfEstGetDescrInfo(&dynSize, &dynAlign), CUPHY_STATUS_SUCCESS);
    ctx.h_dyn.assign(dynSize, 0);
    ctx.d_dyn = alloc_device(dynSize);
    memcpy_h2d(ctx.d_dyn.get(), ctx.h_dyn.data(), dynSize);
    ctx.launchCfgs.nCfgs = 1;
    ASSERT_EQ(cuphyCreatePuschRxNoiseIntfEst(&ctx.hndl), CUPHY_STATUS_SUCCESS);
}

static inline void destroy_setup_ctx(SetupCtx& ctx)
{
    EXPECT_EQ(cuphyDestroyPuschRxNoiseIntfEst(ctx.hndl), CUPHY_STATUS_SUCCESS);
}

TEST_F(PuschNoiseIntfEstTest, Setup_Invalid_EnableDftSOfdm_ReturnsInvalidArgument)
{
    TestConfig cfg = make_cfg(/*enableDftSOfdm=*/0,
                              /*dmrsSymbolIdx=*/CUPHY_PUSCH_NOISE_EST_DMRS_FULL_SLOT,
                              /*eqAlgo=*/PUSCH_EQ_ALGO_TYPE_NOISE_DIAG_MMSE,
                              /*nDmrsCdmGrpsNoData=*/2,
                              /*nRxAnt=*/4,
                              /*nUeGrps=*/1,
                              /*nPrb=*/2);
    SetupCtx   ctx;
    build_setup_ctx(cfg, ctx);
    const uint8_t       invalidEnableDft = 2; // > 1 should be invalid
    const cuphyStatus_t st               = cuphySetupPuschRxNoiseIntfEst(
        ctx.hndl,
        ctx.h_uegrps.data(),
        static_cast<cuphyPuschRxUeGrpPrms_t*>(ctx.d_uegrps.get()),
        cfg.nUeGrps,
        cfg.nPrb,
        invalidEnableDft,
        cfg.dmrsSymbolIdx,
        0,
        ctx.h_dyn.data(),
        ctx.d_dyn.get(),
        &ctx.launchCfgs,
        ctx.strm.handle(),
        0);
    EXPECT_EQ(st, CUPHY_STATUS_INVALID_ARGUMENT);
    destroy_setup_ctx(ctx);
}

TEST_F(PuschNoiseIntfEstTest, Setup_Invalid_DmrsSymbolIdx_ReturnsInvalidArgument)
{
    TestConfig cfg = make_cfg(/*enableDftSOfdm=*/0,
                              /*dmrsSymbolIdx=*/CUPHY_PUSCH_NOISE_EST_DMRS_FULL_SLOT,
                              /*eqAlgo=*/PUSCH_EQ_ALGO_TYPE_NOISE_DIAG_MMSE,
                              /*nDmrsCdmGrpsNoData=*/2,
                              /*nRxAnt=*/4,
                              /*nUeGrps=*/1,
                              /*nPrb=*/2);
    SetupCtx   ctx;
    build_setup_ctx(cfg, ctx);
    const uint8_t       invalidDmrs = 255; // not a supported value
    const cuphyStatus_t st          = cuphySetupPuschRxNoiseIntfEst(
        ctx.hndl,
        ctx.h_uegrps.data(),
        static_cast<cuphyPuschRxUeGrpPrms_t*>(ctx.d_uegrps.get()),
        cfg.nUeGrps,
        cfg.nPrb,
        cfg.enableDftSOfdm,
        invalidDmrs,
        0,
        ctx.h_dyn.data(),
        ctx.d_dyn.get(),
        &ctx.launchCfgs,
        ctx.strm.handle(),
        0);
    EXPECT_EQ(st, CUPHY_STATUS_INVALID_ARGUMENT);
    destroy_setup_ctx(ctx);
}

TEST_F(PuschNoiseIntfEstTest, Setup_Invalid_NullPointers_ReturnsInvalidArgument)
{
    TestConfig cfg = make_cfg(/*enableDftSOfdm=*/0,
                              /*dmrsSymbolIdx=*/CUPHY_PUSCH_NOISE_EST_DMRS_FULL_SLOT,
                              /*eqAlgo=*/PUSCH_EQ_ALGO_TYPE_NOISE_DIAG_MMSE,
                              /*nDmrsCdmGrpsNoData=*/2,
                              /*nRxAnt=*/4,
                              /*nUeGrps=*/1,
                              /*nPrb=*/2);
    SetupCtx   ctx;
    build_setup_ctx(cfg, ctx);
    // Pass nullptrs to trigger invalid argument return
    const cuphyStatus_t st = cuphySetupPuschRxNoiseIntfEst(
        ctx.hndl,
        /*pDrvdUeGrpPrmsCpu*/ nullptr,
        /*pDrvdUeGrpPrmsGpu*/ nullptr,
        cfg.nUeGrps,
        cfg.nPrb,
        cfg.enableDftSOfdm,
        cfg.dmrsSymbolIdx,
        0,
        /*pDynDescrsCpu*/ nullptr,
        /*pDynDescrsGpu*/ nullptr,
        /*pLaunchCfgs*/ nullptr,
        ctx.strm.handle(),
        0);
    EXPECT_EQ(st, CUPHY_STATUS_INVALID_ARGUMENT);
    destroy_setup_ctx(ctx);
}

TEST_F(PuschNoiseIntfEstTest, Setup_Valid_NoDft_FullSlot_SetsKernelFunc)
{
    // Exercise the non-DFT FULL_SLOT branch that assigns kernelFunc without relying on a launch
    TestConfig cfg = make_cfg(/*enableDftSOfdm=*/0,
                              /*dmrsSymbolIdx=*/CUPHY_PUSCH_NOISE_EST_DMRS_FULL_SLOT,
                              /*eqAlgo=*/PUSCH_EQ_ALGO_TYPE_NOISE_DIAG_MMSE,
                              /*nDmrsCdmGrpsNoData=*/2,
                              /*nRxAnt=*/4,
                              /*nUeGrps=*/1,
                              /*nPrb=*/2);
    SetupCtx   ctx;
    build_setup_ctx(cfg, ctx);
    cuphyStatus_t st = cuphySetupPuschRxNoiseIntfEst(
        ctx.hndl,
        ctx.h_uegrps.data(),
        static_cast<cuphyPuschRxUeGrpPrms_t*>(ctx.d_uegrps.get()),
        cfg.nUeGrps,
        cfg.nPrb,
        cfg.enableDftSOfdm,
        cfg.dmrsSymbolIdx,
        0,
        ctx.h_dyn.data(),
        ctx.d_dyn.get(),
        &ctx.launchCfgs,
        ctx.strm.handle(),
        0);
    ASSERT_EQ(st, CUPHY_STATUS_SUCCESS);
    // Verify kernel function pointer was set by the FULL_SLOT branch
    EXPECT_NE(ctx.launchCfgs.cfgs[0].kernelNodeParamsDriver.func, nullptr);
    destroy_setup_ctx(ctx);
}

TEST_F(PuschNoiseIntfEstTest, Setup_Valid_NoDft_FullSlot_GenericSetsKernelFunc)
{
    // Force allAntennaCountsMatch=false and verify FULL_SLOT branch sets kernel function (generic path)
    TestConfig cfg            = make_cfg(/*enableDftSOfdm=*/0,
                              /*dmrsSymbolIdx=*/CUPHY_PUSCH_NOISE_EST_DMRS_FULL_SLOT,
                              /*eqAlgo=*/PUSCH_EQ_ALGO_TYPE_NOISE_DIAG_MMSE,
                              /*nDmrsCdmGrpsNoData=*/2,
                              /*nRxAnt=*/4,
                              /*nUeGrps=*/2,
                              /*nPrb=*/2);
    cfg.mismatchAntennaCounts = true; // second group uses different nRxAnt -> generic path
    SetupCtx ctx;
    build_setup_ctx(cfg, ctx);
    cuphyStatus_t st = cuphySetupPuschRxNoiseIntfEst(
        ctx.hndl,
        ctx.h_uegrps.data(),
        static_cast<cuphyPuschRxUeGrpPrms_t*>(ctx.d_uegrps.get()),
        cfg.nUeGrps,
        cfg.nPrb,
        cfg.enableDftSOfdm,
        cfg.dmrsSymbolIdx,
        0,
        ctx.h_dyn.data(),
        ctx.d_dyn.get(),
        &ctx.launchCfgs,
        ctx.strm.handle(),
        0);
    ASSERT_EQ(st, CUPHY_STATUS_SUCCESS);
    EXPECT_NE(ctx.launchCfgs.cfgs[0].kernelNodeParamsDriver.func, nullptr);
    destroy_setup_ctx(ctx);
}

TEST_F(PuschNoiseIntfEstTest, Setup_Valid_NoDft_FullSlot_SpecializedSetsKernelFunc)
{
    // allAntennaCountsMatch=true, nRxAnt=8 specialization; verify FULL_SLOT branch sets kernel function
    TestConfig cfg = make_cfg(/*enableDftSOfdm=*/0,
                              /*dmrsSymbolIdx=*/CUPHY_PUSCH_NOISE_EST_DMRS_FULL_SLOT,
                              /*eqAlgo=*/PUSCH_EQ_ALGO_TYPE_NOISE_DIAG_MMSE,
                              /*nDmrsCdmGrpsNoData=*/2,
                              /*nRxAnt=*/8,
                              /*nUeGrps=*/1,
                              /*nPrb=*/2);
    SetupCtx   ctx;
    build_setup_ctx(cfg, ctx);
    cuphyStatus_t st = cuphySetupPuschRxNoiseIntfEst(
        ctx.hndl,
        ctx.h_uegrps.data(),
        static_cast<cuphyPuschRxUeGrpPrms_t*>(ctx.d_uegrps.get()),
        cfg.nUeGrps,
        cfg.nPrb,
        cfg.enableDftSOfdm,
        cfg.dmrsSymbolIdx,
        0,
        ctx.h_dyn.data(),
        ctx.d_dyn.get(),
        &ctx.launchCfgs,
        ctx.strm.handle(),
        0);
    ASSERT_EQ(st, CUPHY_STATUS_SUCCESS);
    EXPECT_NE(ctx.launchCfgs.cfgs[0].kernelNodeParamsDriver.func, nullptr);
    destroy_setup_ctx(ctx);
}

} // namespace

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    int result = RUN_ALL_TESTS();

    return result;
}

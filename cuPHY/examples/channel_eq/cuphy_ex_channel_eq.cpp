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
#include <string>
#include <dirent.h> // opendir, readdir

#include "CLI/CLI.hpp"

#include "hdf5hpp.hpp"
#include "cuphy_hdf5.hpp"
#include "pusch_rx.hpp"
#include "pusch_utils.hpp"
#include "datasets.hpp"
#include "nvlog.hpp"
#include "gsl-lite/gsl-lite.hpp"

#define PUSCH_NOISE_INTF_EST_EQ_TEST 0
#define PUSCH_CFO_TA_EST_EQ_TEST PUSCH_NOISE_INTF_EST_EQ_TEST + 1
#define PUSCH_CH_EQ_COEF_EQ_TEST PUSCH_CFO_TA_EST_EQ_TEST + 1
#define PUSCH_CH_EQ_SOFT_DEMAP_EQ_TEST PUSCH_CH_EQ_COEF_EQ_TEST + CUPHY_PUSCH_RX_MAX_N_TIME_CH_EQ
#define PUSCH_CH_EQ_IDFT_EQ_TEST PUSCH_CH_EQ_SOFT_DEMAP_EQ_TEST + 1
#define PUSCH_CH_EQ_AFTER_IDFT_EQ_TEST PUSCH_CH_EQ_IDFT_EQ_TEST + 1
#define PUSCH_FRONT_END_PARAMS_EQ_TEST PUSCH_CH_EQ_AFTER_IDFT_EQ_TEST + 1
#define N_PUSCH_DESCR_TYPES_EQ_TEST PUSCH_FRONT_END_PARAMS_EQ_TEST + 1

#define LINEAR_ALLOC_PAD_BYTES 128

template <typename T>
void copyTensorRef2Info(cuphy::tensor_ref& tRef, T& tInfo)
{
    tInfo.pAddr              = tRef.addr();
    const tensor_desc& tDesc = static_cast<const tensor_desc&>(*(tRef.desc().handle()));
    tInfo.elemType           = tDesc.type();
    std::copy_n(tDesc.layout().strides.begin(), std::extent<decltype(tInfo.strides)>::value, tInfo.strides);
}

size_t getBufferSizeBluesteinWorkspace(cuphyPuschStatPrms_t const* pStatPrms)
{
     if(pStatPrms->enableDftSOfdm==0)
     {
         return 0;
     }
     else
     {
         //////// 53 different DFT sizes for DFT-s-OFDM//////////////////////////////
         // 12 24 36 48 60:                                              FFT128
         // 72 96 108 120:                                               FFT256
         // 144 180 192 216 240:                                         FFT512
         // 288 300 324 360 384 432 480:                                 FFT1024
         // 540 576 600 648 720 768 864 900 960 972:                     FFT2048
         // 1080 1152 1200 1296 1440 1500 1536 1620 1728 1800 1920 1944: FFT4096
         // 2160 2304 2400 2592 2700 2880 2916 3000 3072 3240:           FFT8192
         // Memeory size for Bluestein Workspace in both time and frequency domains
         return 53*sizeof(data_type_traits<CUPHY_C_32F>::type)*FFT8192*2;
     }
}

// TODO for size reduction
size_t getBufferSize(cuphyPuschStatPrms_t const* pStatPrms)
{
    // data type sizes
    static constexpr uint32_t N_BYTES_C16        = sizeof(data_type_traits<CUPHY_C_16F>::type);
    static constexpr uint32_t N_BYTES_R16        = sizeof(data_type_traits<CUPHY_R_16F>::type);
    static constexpr uint32_t N_BYTES_R32        = sizeof(data_type_traits<CUPHY_R_32F>::type);
    static constexpr uint32_t N_BYTES_C32        = sizeof(data_type_traits<CUPHY_C_32F>::type);
    static constexpr uint32_t N_BYTES_PER_UINT32 = 4;
    static constexpr uint32_t MAX_N_UE           = MAX_N_TBS_SUPPORTED; // 1UE per TB for PUSCH

    //Find the max UL BWP and max layers across all cells
    uint32_t max_nPrbUlBwp = pStatPrms->nMaxPrb;
    uint32_t max_N_RX      = pStatPrms->nMaxRx;

    // if max parameters are zero set to max supported
    if(max_nPrbUlBwp == 0)
        max_nPrbUlBwp = MAX_N_PRBS_SUPPORTED;
    if(max_N_RX == 0)
        max_N_RX = MAX_N_ANTENNAS_SUPPORTED;

    if(max_nPrbUlBwp > MAX_N_PRBS_SUPPORTED)
    {
        NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT, "nMaxPrb provided {} is larger than supported max {}", max_nPrbUlBwp, MAX_N_PRBS_SUPPORTED);
        std::string err("PUSCH: nMaxPrb provided (" + std::to_string(max_nPrbUlBwp) + ") is larger than supported max (" + std::to_string(MAX_N_PRBS_SUPPORTED) + ")");
        throw std::out_of_range(err);
    }

    if(max_N_RX > MAX_N_ANTENNAS_SUPPORTED)
    {
        NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT, "nMaxRx provided {} is larger than supported max {}", max_N_RX, MAX_N_ANTENNAS_SUPPORTED);
        std::string err("PUSCH: nMaxRx provided (" + std::to_string(max_N_RX) + ") is larger than supported max (" + std::to_string(MAX_N_ANTENNAS_SUPPORTED) + ")");
        throw std::out_of_range(err);
    }

    //---------------------------------------------------------------------------------------------------------------------------
    // Buffer sizes compute which needs to be scaled up by maximum number of cells per slot

    // compute linear buffer size
    size_t        nBytesBuffer          = 0;
    uint32_t      NF                    = max_nPrbUlBwp * CUPHY_N_TONES_PER_PRB;
    uint32_t      N_RX                  = max_N_RX;
    uint32_t      N_MAX_LAYERS          = N_RX;
    const int32_t EXTRA_PADDING         = MAX_N_USER_GROUPS_SUPPORTED * LINEAR_ALLOC_PAD_BYTES; // upper bound for extra memory required per allocation

    uint32_t nTimeChEq = pStatPrms->enablePuschTdi ? CUPHY_PUSCH_RX_MAX_N_TIME_CH_EQ : 1;

    // max equalizer coefficent buffer
    uint32_t maxBytesEqualizer = N_BYTES_C32 * N_RX * N_RX * NF * nTimeChEq;
    nBytesBuffer += maxBytesEqualizer + EXTRA_PADDING;

    // max ReeDiagInv buffer (equalizer preceision)
    uint32_t maxBytesPrecesion = N_BYTES_R32 * N_RX * NF * nTimeChEq;
    nBytesBuffer += maxBytesPrecesion + EXTRA_PADDING;

    // max equalizer debug buffer
    if(pStatPrms->pDbg != nullptr)
    {
        if(pStatPrms->pDbg->pOutFileName != nullptr)
        {
            uint32_t maxBytesEqualizerDbg = N_BYTES_C32 * (2 * N_RX) * N_RX * NF * (OFDM_SYMBOLS_PER_SLOT - 1);
            nBytesBuffer += maxBytesEqualizerDbg + EXTRA_PADDING;
        }
    }

    if (pStatPrms->enableDebugEqOutput)
    {
        // max estiamted data (DataEq) buffer
        uint32_t maxBytesEstimatedData = N_BYTES_C16 * N_RX * NF * (OFDM_SYMBOLS_PER_SLOT - 1);
        nBytesBuffer += maxBytesEstimatedData + EXTRA_PADDING;
    }

    // max equalizer output LLR buffer
    uint32_t maxBitsPerQam     = 8;
    uint32_t maxBytesEqOutLLRs = N_BYTES_R16 * NF * maxBitsPerQam * N_MAX_LAYERS * (OFDM_SYMBOLS_PER_SLOT - 1);
    nBytesBuffer += maxBytesEqOutLLRs + EXTRA_PADDING;
    nBytesBuffer += maxBytesEqOutLLRs + EXTRA_PADDING; // for LLR CDM1

    // Scale up by max number of cells per slot
    nBytesBuffer = nBytesBuffer * pStatPrms->nMaxCellsPerSlot;

    //---------------------------------------------------------------------------------------------------------------------------
    // Buffer sizes compute using the constants MAX_N_USER_GROUPS_SUPPORTED, MAX_N_TBS_SUPPORTED
    // accounts for all cells

    uint32_t maxBytesCfoPhaseRot = N_BYTES_C32 * CUPHY_PUSCH_RX_MAX_N_TIME_CH_EST * CUPHY_PUSCH_RX_MAX_N_LAYERS_PER_UE_GROUP * MAX_N_USER_GROUPS_SUPPORTED;
    nBytesBuffer += maxBytesCfoPhaseRot + EXTRA_PADDING;

    uint32_t maxBytesTaPhaseRot = N_BYTES_C32 * CUPHY_PUSCH_RX_MAX_N_LAYERS_PER_UE_GROUP * MAX_N_USER_GROUPS_SUPPORTED;
    nBytesBuffer += maxBytesTaPhaseRot + EXTRA_PADDING;

    uint32_t maxBytesCfoTaEstInterCtaSyncCnt = N_BYTES_PER_UINT32 * MAX_N_USER_GROUPS_SUPPORTED;
    nBytesBuffer += maxBytesCfoTaEstInterCtaSyncCnt + EXTRA_PADDING;

    nBytesBuffer += N_BYTES_R32 * MAX_N_UE;  //CFO

    nBytesBuffer += N_BYTES_R32 * MAX_N_UE;  //TA

    uint32_t maxBytesCfoEst = N_BYTES_C32 * MAX_ND_SUPPORTED * CUPHY_PUSCH_RX_MAX_N_LAYERS_PER_UE_GROUP * MAX_N_USER_GROUPS_SUPPORTED;
    nBytesBuffer += maxBytesCfoEst + EXTRA_PADDING;

    // max DFT data buffer
    if(pStatPrms->enableDftSOfdm==1)
    {
        nBytesBuffer += (N_BYTES_C32 * MAX_N_USER_GROUPS_SUPPORTED * 3276 * (OFDM_SYMBOLS_PER_SLOT - 1) + EXTRA_PADDING);
        nBytesBuffer += (N_BYTES_C32 * MAX_N_USER_GROUPS_SUPPORTED * FFT8192 * (OFDM_SYMBOLS_PER_SLOT - 1) + EXTRA_PADDING); //for intermediate results in Bluestein's FFT
        nBytesBuffer += (N_BYTES_C32 * MAX_N_USER_GROUPS_SUPPORTED * FFT8192 + EXTRA_PADDING); //for time domain data in Bluestein's FFT Workspace
        nBytesBuffer += (N_BYTES_C32 * MAX_N_USER_GROUPS_SUPPORTED * FFT8192 + EXTRA_PADDING); //for freq domain data in Bluestein's FFT Workspace
    }

    if(pStatPrms->enableSinrMeasurement)// || EqCoeffAlgoIsMMSEVariant(m_chEstSettings.eqCoeffAlgo))
    {
        uint32_t maxBytesNoiseVarPreEq               = N_BYTES_R32 * MAX_N_UE + LINEAR_ALLOC_PAD_BYTES;
        uint32_t maxBytesNoiseIntfEstInterCtaSyncCnt = N_BYTES_PER_UINT32 * MAX_N_USER_GROUPS_SUPPORTED + LINEAR_ALLOC_PAD_BYTES;
        uint32_t maxBytesNoiseIntfEstLwInv           = N_BYTES_C32 * N_RX * N_RX * MAX_N_PRBS_SUPPORTED * pStatPrms->nMaxCellsPerSlot + (MAX_N_USER_GROUPS_SUPPORTED * LINEAR_ALLOC_PAD_BYTES);
        // 2x PreEq noise buffer (one in dB reported to L1C + one in linear domain used in DTX detection) fix it
        nBytesBuffer += ((2*maxBytesNoiseVarPreEq) + maxBytesNoiseIntfEstInterCtaSyncCnt + maxBytesNoiseIntfEstLwInv);
    }
    
    return nBytesBuffer;
}

////////////////////////////////////////////////////////////////////////
// main()
int main(int argc, char* argv[])
{
    cuphyNvlogFmtHelper nvlog_fmt("ch_eq.log");
    nvlog_set_log_level(NVLOG_DEBUG);
    try
    {
        //------------------------------------------------------------------
        // Parse command line arguments
        CLI::App app{"ch_eq"};
        std::string inputFilename, outputFilename, trtYamlInput;
        std::vector<std::string> inputFilenameVec;
        uint32_t    fp16Mode       = 0xBAD;
        std::ignore = app.add_option("-i", inputFilename, "Input HDF5 filename, which must contain the following datasets:\n"
                                                           "Data_rx      : received data (frequency-time) to be equalized\n"
                                                           "WFreq        : interpolation filter coefficients used in channel estimation\n"
                                                           "ShiftSeq     : sequence to be applied to DMRS tones containing descrambling code and delay shift for channel centering\n"
                                                           "UnShiftSeq   : sequence to remove the delay shift from estimated channel\n")->required();
        std::ignore = app.add_option("-o", outputFilename, "Output HDFS debug file");
        std::ignore = app.add_option("-H", fp16Mode, "0         : No FP16\n"
                                                     "1(default): FP16 format used for received data samples only\n"
                                                     "2         : FP16 format used for all front end params\n");
        std::ignore = app.add_option("--trt-yaml", trtYamlInput, "TRT Engine YAML File for input configurations\n");
        CLI11_PARSE(app, argc, argv)
        inputFilenameVec.push_back(inputFilename);
        cudaStream_t cuStream;
        cudaStreamCreateWithFlags(&cuStream, cudaStreamNonBlocking);
        if(0xBAD == fp16Mode) fp16Mode = 1;

        // Check FP16 mode of operation
        bool isChannelFp16 = false;
        switch(fp16Mode)
        {
        case 0:
            [[fallthrough]];
        case 1:
            isChannelFp16 = false;
            break;
        case 2:
            isChannelFp16 = true;
            break;
        default:
            isChannelFp16 = false;
            break;
        }

        hdf5hpp::hdf5_file fInput = hdf5hpp::hdf5_file::open(inputFilename.c_str());

        //------------------------------------------------------------------
        // Load API parameters

        cuphy::stream cuStrmMain;

        const uint64_t procModeBmsk = 0;
        const bool cpuCopyOn        = false;

        StaticApiDataset  staticApiDataset(inputFilenameVec, cuStrmMain.handle(), outputFilename);
        DynApiDataset     dynApiDataset(inputFilenameVec,   cuStrmMain.handle(), procModeBmsk, cpuCopyOn, fp16Mode);
        EvalDataset       evalDataset(inputFilenameVec, cuStrmMain.handle());

        cudaStreamSynchronize(cuStrmMain.handle()); // synch to ensure data copied

        //----------------------------------------------------------------
        // Initialize CPU/GPU memory

        const uint32_t nUeGrps = dynApiDataset.cellGrpDynPrm.nUeGrps;
        const uint32_t nUes    = dynApiDataset.cellGrpDynPrm.nUes;

        //------------------------------------------------------------------
        // Allocate descriptors

        std::array<size_t, N_PUSCH_DESCR_TYPES_EQ_TEST> statDescrSizeBytes{};
        std::array<size_t, N_PUSCH_DESCR_TYPES_EQ_TEST> statDescrAlignBytes{};
        std::array<size_t, N_PUSCH_DESCR_TYPES_EQ_TEST> dynDescrSizeBytes{};
        std::array<size_t, N_PUSCH_DESCR_TYPES_EQ_TEST> dynDescrAlignBytes{};

        size_t* pStatDescrSizeBytes  = statDescrSizeBytes.data();  
        size_t* pStatDescrAlignBytes = statDescrAlignBytes.data();
        size_t* pDynDescrSizeBytes   = dynDescrSizeBytes.data();
        size_t* pDynDescrAlignBytes  = dynDescrAlignBytes.data();
        
        cuphyStatus_t status = cuphyPuschRxNoiseIntfEstGetDescrInfo(&pDynDescrSizeBytes[PUSCH_NOISE_INTF_EST_EQ_TEST], &pDynDescrAlignBytes[PUSCH_NOISE_INTF_EST_EQ_TEST]);
        if(CUPHY_STATUS_SUCCESS != status)
        {
            throw cuphy::cuphy_fn_exception(status, "cuphyPuschRxNoiseIntfEstGetDescrInfo()");
        }
        
        status = cuphyPuschRxCfoTaEstGetDescrInfo(&pStatDescrSizeBytes[PUSCH_CFO_TA_EST_EQ_TEST],
                                                  &pStatDescrAlignBytes[PUSCH_CFO_TA_EST_EQ_TEST],
                                                  &pDynDescrSizeBytes[PUSCH_CFO_TA_EST_EQ_TEST],
                                                  &pDynDescrAlignBytes[PUSCH_CFO_TA_EST_EQ_TEST]);
        if(CUPHY_STATUS_SUCCESS != status)
        {
            throw cuphy::cuphy_fn_exception(status, "cuphyPuschRxCfoTaEstGetDescrInfo()");
        }
        
        status = cuphyPuschRxChEqGetDescrInfo(&pStatDescrSizeBytes[PUSCH_CH_EQ_COEF_EQ_TEST],
                                              &pStatDescrAlignBytes[PUSCH_CH_EQ_COEF_EQ_TEST],
                                              &pStatDescrSizeBytes[PUSCH_CH_EQ_IDFT_EQ_TEST],
                                              &pStatDescrAlignBytes[PUSCH_CH_EQ_IDFT_EQ_TEST],
                                              &pDynDescrSizeBytes[PUSCH_CH_EQ_COEF_EQ_TEST],
                                              &pDynDescrAlignBytes[PUSCH_CH_EQ_COEF_EQ_TEST],
                                              &pDynDescrSizeBytes[PUSCH_CH_EQ_SOFT_DEMAP_EQ_TEST],
                                              &pDynDescrAlignBytes[PUSCH_CH_EQ_SOFT_DEMAP_EQ_TEST]);
        if(CUPHY_STATUS_SUCCESS != status)
        {
            throw cuphy::cuphy_fn_exception(status, "cuphyPuschRxChEqGetDescrInfo()");
        }

        for(uint32_t chEqInstIdx = 1; chEqInstIdx < CUPHY_PUSCH_RX_MAX_N_TIME_CH_EQ; ++chEqInstIdx)
        {
            pStatDescrSizeBytes[PUSCH_CH_EQ_COEF_EQ_TEST + chEqInstIdx]  = pStatDescrSizeBytes[PUSCH_CH_EQ_COEF_EQ_TEST];
            pStatDescrAlignBytes[PUSCH_CH_EQ_COEF_EQ_TEST + chEqInstIdx] = pStatDescrAlignBytes[PUSCH_CH_EQ_COEF_EQ_TEST];
            pDynDescrSizeBytes[PUSCH_CH_EQ_COEF_EQ_TEST + chEqInstIdx]   = pDynDescrSizeBytes[PUSCH_CH_EQ_COEF_EQ_TEST];
            pDynDescrAlignBytes[PUSCH_CH_EQ_COEF_EQ_TEST + chEqInstIdx]  = pDynDescrAlignBytes[PUSCH_CH_EQ_COEF_EQ_TEST];
        }

        pDynDescrSizeBytes[PUSCH_CH_EQ_IDFT_EQ_TEST]        = pDynDescrSizeBytes[PUSCH_CH_EQ_SOFT_DEMAP_EQ_TEST];
        pDynDescrAlignBytes[PUSCH_CH_EQ_IDFT_EQ_TEST]       = pDynDescrAlignBytes[PUSCH_CH_EQ_SOFT_DEMAP_EQ_TEST];

        pDynDescrSizeBytes[PUSCH_CH_EQ_AFTER_IDFT_EQ_TEST]  = pDynDescrSizeBytes[PUSCH_CH_EQ_SOFT_DEMAP_EQ_TEST];
        pDynDescrAlignBytes[PUSCH_CH_EQ_AFTER_IDFT_EQ_TEST] = pDynDescrAlignBytes[PUSCH_CH_EQ_SOFT_DEMAP_EQ_TEST];
        
        pDynDescrSizeBytes[PUSCH_FRONT_END_PARAMS_EQ_TEST]  = sizeof(cuphyPuschRxUeGrpPrms_t) * MAX_N_USER_GROUPS_SUPPORTED;
        pDynDescrAlignBytes[PUSCH_FRONT_END_PARAMS_EQ_TEST] = alignof(cuphyPuschRxUeGrpPrms_t);
        
        
        cuphy::kernelDescrs<N_PUSCH_DESCR_TYPES_EQ_TEST>     m_kernelStatDescr("PuschStatDescrEqTest");
        cuphy::kernelDescrs<N_PUSCH_DESCR_TYPES_EQ_TEST>     m_kernelDynDescr("PuschDynDescrEqTest");
        cuphyMemoryFootprint m_memoryFootprint;
        m_kernelStatDescr.alloc(statDescrSizeBytes, statDescrAlignBytes, &m_memoryFootprint);
        m_kernelDynDescr.alloc(dynDescrSizeBytes, dynDescrAlignBytes, &m_memoryFootprint);

        //------------------------------------------------------------------
        // Create component objects
        cuphyChEstSettings chEstSettings(&staticApiDataset.puschStatPrms, cuStrmMain.handle(), &m_memoryFootprint);
        
        cuphy::linear_alloc<LINEAR_ALLOC_PAD_BYTES, cuphy::device_alloc> m_LinearAllocBluesteinWorkspace(getBufferSizeBluesteinWorkspace(&(staticApiDataset.puschStatPrms)), &m_memoryFootprint);
        cuphy::tensor_ref m_tRefBluesteinWorkspaceTime, m_tRefBluesteinWorkspaceFreq;
        cuphyTensorInfo2_t tInfoDftBluesteinWorkspaceTime, tInfoDftBluesteinWorkspaceFreq;
        
        auto statCpuDescrStartAddrs      = m_kernelStatDescr.getCpuStartAddrs();
        auto statGpuDescrStartAddrs      = m_kernelStatDescr.getGpuStartAddrs();
        
        cuphyPuschRxNoiseIntfEstHndl_t m_noiseIntfEstHndl;
        cuphyPuschRxCfoTaEstHndl_t     m_cfoTaEstHndl;
        cuphyPuschRxChEqHndl_t         m_chEqHndl;
        cuphy::context                 m_ctx;
        const cuphyDeviceArchInfo      m_cudaDeviceArchInfo(get_cuda_device_arch());
        bool enableCpuToGpuDescrAsyncCpy = true;

        status = cuphyCreatePuschRxNoiseIntfEst(&m_noiseIntfEstHndl);
        if(CUPHY_STATUS_SUCCESS != status)
        {
            throw cuphy::cuphy_fn_exception(status, "cuphyCreatePuschRxNoiseIntfEst()");
        }

        status = cuphyCreatePuschRxCfoTaEst(&m_cfoTaEstHndl,
                                            enableCpuToGpuDescrAsyncCpy ? static_cast<uint8_t>(1) : static_cast<uint8_t>(0),
                                            static_cast<void*>(statCpuDescrStartAddrs[PUSCH_CFO_TA_EST_EQ_TEST]),
                                            static_cast<void*>(statGpuDescrStartAddrs[PUSCH_CFO_TA_EST_EQ_TEST]),
                                            cuStrmMain.handle());
        if(CUPHY_STATUS_SUCCESS != status)
        {
            throw cuphy::cuphy_fn_exception(status, "cuphyCreatePuschRxCfoTaEst()");
        }

        if(chEstSettings.enableDftSOfdm==1)
        {
            m_LinearAllocBluesteinWorkspace.reset();
            m_tRefBluesteinWorkspaceTime.desc().set(CUPHY_C_32F, 53, FFT8192, cuphy::tensor_flags::align_tight);
            m_LinearAllocBluesteinWorkspace.alloc(m_tRefBluesteinWorkspaceTime);
            copyTensorRef2Info(m_tRefBluesteinWorkspaceTime, tInfoDftBluesteinWorkspaceTime);
    
            m_tRefBluesteinWorkspaceFreq.desc().set(CUPHY_C_32F, 53, FFT8192, cuphy::tensor_flags::align_tight);
            m_LinearAllocBluesteinWorkspace.alloc(m_tRefBluesteinWorkspaceFreq);
            copyTensorRef2Info(m_tRefBluesteinWorkspaceFreq, tInfoDftBluesteinWorkspaceFreq);
//    
//            if(!m_cudaDeviceArchInfo.cuPHYSupported)
//            {
//                NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT, "{}: cudaDeviceArch {} is not supported", __FUNCTION__, m_cudaDeviceArchInfo.computeCapability);
//                throw cuphy::cuphy_fn_exception(status, "cuphyCreatePuschRxChEq()");
//            }
        }

        status = cuphyCreatePuschRxChEq(m_ctx.handle(),
                                        &m_chEqHndl,
                                        tInfoDftBluesteinWorkspaceTime,
                                        tInfoDftBluesteinWorkspaceFreq,
                                        m_cudaDeviceArchInfo.computeCapability,
                                        chEstSettings.enableDftSOfdm,
                                        staticApiDataset.puschStatPrms.enableDebugEqOutput,
                                        enableCpuToGpuDescrAsyncCpy ? static_cast<uint8_t>(1) : static_cast<uint8_t>(0),
                                        reinterpret_cast<void**>(&statCpuDescrStartAddrs[PUSCH_CH_EQ_COEF_EQ_TEST]),
                                        reinterpret_cast<void**>(&statGpuDescrStartAddrs[PUSCH_CH_EQ_COEF_EQ_TEST]),
                                        reinterpret_cast<void**>(&statCpuDescrStartAddrs[PUSCH_CH_EQ_IDFT_EQ_TEST]),
                                        reinterpret_cast<void**>(&statGpuDescrStartAddrs[PUSCH_CH_EQ_IDFT_EQ_TEST]),
                                        cuStrmMain.handle());
        if(CUPHY_STATUS_SUCCESS != status)
        {
            throw cuphy::cuphy_fn_exception(status, "cuphyCreatePuschRxChEq()");
        }
        
        //------------------------------------------------------------------
        // setup phase 1
        bool bypass_soft_demapper_eval_flag = chEstSettings.enableDftSOfdm ? true : false;
        enableCpuToGpuDescrAsyncCpy = false; 
        uint8_t enableRssiMeasurement = 0;
        bool subSlotProcessingFrontLoadedDmrsEnabled = true;
        uint8_t maxDmrsMaxLen = 1;
        uint32_t maxNPrbAlloc = getMaxNPrbAlloc(&staticApiDataset.puschStatPrms);
        
        auto dynCpuDescrStartAddrs = m_kernelDynDescr.getCpuStartAddrs();
        auto dynGpuDescrStartAddrs = m_kernelDynDescr.getGpuStartAddrs();
        gsl_lite::span<cuphyPuschRxUeGrpPrms_t> m_drvdUeGrpPrmsCpu = gsl_lite::span(reinterpret_cast<cuphyPuschRxUeGrpPrms_t*>(dynCpuDescrStartAddrs[PUSCH_FRONT_END_PARAMS_EQ_TEST]), MAX_N_USER_GROUPS_SUPPORTED);
        gsl_lite::span<cuphyPuschRxUeGrpPrms_t> m_drvdUeGrpPrmsGpu = gsl_lite::span(reinterpret_cast<cuphyPuschRxUeGrpPrms_t*>(dynGpuDescrStartAddrs[PUSCH_FRONT_END_PARAMS_EQ_TEST]), MAX_N_USER_GROUPS_SUPPORTED);
        uint32_t m_nMaxPrb = PuschRx::expandFrontEndParameters(&dynApiDataset.puschDynPrm, &staticApiDataset.puschStatPrms, m_drvdUeGrpPrmsCpu.data(), subSlotProcessingFrontLoadedDmrsEnabled, maxDmrsMaxLen, enableRssiMeasurement, maxNPrbAlloc);
        
        ///////////////////////////////////////////
        cuphyPuschUePrm_t*        uePrmsArray   = dynApiDataset.puschDynPrm.pCellGrpDynPrm->pUePrms;
        uint32_t layerCount[MAX_N_USER_GROUPS_SUPPORTED] = {0};
        PerTbParams perTbPrms[MAX_N_TBS_SUPPORTED];

        for(int i = 0; i < nUes; i++)
        {
            cuphyPuschUeGrpPrm_t*    pUeGrpPrm     = uePrmsArray[i].pUeGrpPrm;
            cuphyPuschCellDynPrm_t*  pCellDynPrm   = pUeGrpPrm->pCellPrm;
            uint16_t                 ueGrpIdx      = uePrmsArray[i].ueGrpIdx;
            cuphyPuschRxUeGrpPrms_t* drvdUeGrpPrms = &m_drvdUeGrpPrmsCpu[ueGrpIdx];

            perTbPrms[i].Qm = uePrmsArray[i].qamModOrder;
            
            uint32_t nUeLayers = static_cast<uint32_t>(uePrmsArray[i].nUeLayers);
            perTbPrms[i].Nl = nUeLayers;
            for(int l = 0; l < nUeLayers; l++)
            {
                perTbPrms[i].layer_map_array[l] = layerCount[ueGrpIdx];
                layerCount[ueGrpIdx]++;
            }
        }
        
        for(uint32_t iterator = 0; iterator < nUeGrps; iterator++)
        {   
            if((bypass_soft_demapper_eval_flag==false)&&(m_drvdUeGrpPrmsCpu[iterator].nDmrsCdmGrpsNoData==1))
                bypass_soft_demapper_eval_flag = true;
            
            cuphyPuschUeGrpPrm_t ueGrpPrms = dynApiDataset.puschDynPrm.pCellGrpDynPrm->pUeGrpPrms[iterator];
            for(int i = 0; i < ueGrpPrms.nUes; i++)
            {
                uint16_t ueIdx = ueGrpPrms.pUePrmIdxs[i];
                uint8_t  Qm    = static_cast<uint8_t>(perTbPrms[ueIdx].Qm);
                for(int j = 0; j < perTbPrms[ueIdx].Nl; ++j)
                {
                    m_drvdUeGrpPrmsCpu[iterator].qam[perTbPrms[ueIdx].layer_map_array[j]] = Qm;
                }
            }
        }
        //////////////////////////////////////////////////////////////////////////////////
        //------------------------------------------------------------------
        // Allocate device memory
        std::vector<cuphy::tensor_ref> m_tRefLwInvVec, m_tRefCfoEstVec, m_tRefReeDiagInvVec;
        cuphy::tensor_ref m_tRefCfoPhaseRot, m_tRefTaPhaseRot, m_tRefTaEst, m_tRefCfoTaEstInterCtaSyncCnt, m_tRefCfoEstInterCtaSyncCnt, m_tRefNoiseIntfEstInterCtaSyncCnt, m_tRefCfoHz, m_tRefNoiseVarPreEq;
        std::vector<cuphy::tensor_device> tChEstVec;
        cuphyTensorPrm_t                  tPrmChEst;
        std::vector<cuphy::tensor_device> tChEqCoefVec;
        cuphyTensorPrm_t                  tPrmChEqCoef;
        std::vector<cuphy::tensor_device> tChEqLLRVec;
        cuphyTensorPrm_t                  tPrmChEqLLR;
        cuphy::linear_alloc<LINEAR_ALLOC_PAD_BYTES, cuphy::device_alloc> m_LinearAlloc(getBufferSize(&(staticApiDataset.puschStatPrms)), &m_memoryFootprint);
        
        m_tRefLwInvVec.resize(MAX_N_USER_GROUPS_SUPPORTED);
        m_tRefCfoEstVec.resize(MAX_N_USER_GROUPS_SUPPORTED);
        m_tRefReeDiagInvVec.resize(MAX_N_USER_GROUPS_SUPPORTED);
        
        // mark the initial address and offset
        size_t initOffset = m_LinearAlloc.offset();
        void*  initAddr   = static_cast<char*>(m_LinearAlloc.address()) + initOffset;

        // Tensor allocations common across UE groups
        m_tRefCfoPhaseRot.desc().set(CUPHY_C_32F, CUPHY_PUSCH_RX_MAX_N_TIME_CH_EST, CUPHY_PUSCH_RX_MAX_N_LAYERS_PER_UE_GROUP, MAX_N_USER_GROUPS_SUPPORTED, cuphy::tensor_flags::align_tight);
        m_LinearAlloc.alloc(m_tRefCfoPhaseRot);
        m_tRefTaPhaseRot.desc().set(CUPHY_C_32F, CUPHY_PUSCH_RX_MAX_N_LAYERS_PER_UE_GROUP, nUeGrps, cuphy::tensor_flags::align_tight);
        m_LinearAlloc.alloc(m_tRefTaPhaseRot);
        m_tRefCfoHz.desc().set(CUPHY_R_32F, nUes, cuphy::tensor_flags::align_tight);
        m_LinearAlloc.alloc(m_tRefCfoHz);
        m_tRefTaEst.desc().set(CUPHY_R_32F, nUes, cuphy::tensor_flags::align_tight);
        m_LinearAlloc.alloc(m_tRefTaEst);
        m_tRefCfoTaEstInterCtaSyncCnt.desc().set(CUPHY_R_32U, MAX_N_USER_GROUPS_SUPPORTED, cuphy::tensor_flags::align_tight);
        m_LinearAlloc.alloc(m_tRefCfoTaEstInterCtaSyncCnt);
        m_tRefNoiseIntfEstInterCtaSyncCnt.desc().set(CUPHY_R_32U, nUeGrps, cuphy::tensor_flags::align_tight);
        m_LinearAlloc.alloc(m_tRefNoiseIntfEstInterCtaSyncCnt);
        m_tRefNoiseVarPreEq.desc().set(CUPHY_R_32F, nUes, cuphy::tensor_flags::align_tight);
        m_LinearAlloc.alloc(m_tRefNoiseVarPreEq);

        // Per UE group tensor allocations
        int NUM_ANTENNAS, NUM_LAYERS, NUM_DMRS_SYMS, NUM_DATA_SYMS, NH, NUM_PRBS, NF;
        for(int i = 0; i < nUeGrps; i++)
        {
            uint16_t cellPrmDynIdx = (dynApiDataset.cellGrpDynPrm.pUeGrpPrms[i]).pCellPrm->cellPrmDynIdx;

            cuphyDataType_t cplxTypeCh = CUPHY_C_32F;
            cuphyDataType_t realTypeCh = scalar_type_from_complex_type(cplxTypeCh);

            cuphyPuschRxUeGrpPrms_t& drvdUeGrpPrmsCpu = m_drvdUeGrpPrmsCpu[i];

            NUM_ANTENNAS  = drvdUeGrpPrmsCpu.nRxAnt;
            NUM_LAYERS    = drvdUeGrpPrmsCpu.nLayers;
            NUM_DMRS_SYMS = drvdUeGrpPrmsCpu.nDmrsSyms;
            NUM_DATA_SYMS = drvdUeGrpPrmsCpu.nDataSym;
            NH            = m_drvdUeGrpPrmsCpu[i].dmrsAddlnPos + 1;
            NUM_PRBS      = drvdUeGrpPrmsCpu.nPrb;
            NF            = CUPHY_N_TONES_PER_PRB * drvdUeGrpPrmsCpu.nPrb;

            uint8_t nDmrsCdmGrpsNoData = dynApiDataset.cellGrpDynPrm.pUeGrpPrms[i].pDmrsDynPrm->nDmrsCdmGrpsNoData;
            if(nDmrsCdmGrpsNoData==1)
            {
                NUM_DATA_SYMS += NUM_DMRS_SYMS;
            }

            // Slot buffer buffer allocated by L1 control
            copyTensorPrm2Info(dynApiDataset.DataIn.pTDataRx[cellPrmDynIdx], drvdUeGrpPrmsCpu.tInfoDataRx);
            
            tChEstVec.push_back(cuphy::tensor_device(cplxTypeCh, evalDataset.HestRef[i].layout()));
            tChEstVec[i].convert(evalDataset.HestRef[i], cuStrmMain.handle());
            tPrmChEst.desc  = tChEstVec[i].desc().handle();
            tPrmChEst.pAddr = tChEstVec[i].addr();
            copyTensorPrm2Info(tPrmChEst, drvdUeGrpPrmsCpu.tInfoHEst);

            if(chEstSettings.enableSinrMeasurement)// || EqCoeffAlgoIsMMSEVariant(m_chEstSettings.eqCoeffAlgo))
            {
                m_tRefLwInvVec[i].desc().set(cplxTypeCh, NUM_ANTENNAS, NUM_ANTENNAS, drvdUeGrpPrmsCpu.nPrb, cuphy::tensor_flags::align_tight);
                m_LinearAlloc.alloc(m_tRefLwInvVec[i]);
                copyTensorRef2Info(m_tRefLwInvVec[i], drvdUeGrpPrmsCpu.tInfoLwInv);
                
                copyTensorRef2Info(m_tRefNoiseVarPreEq, drvdUeGrpPrmsCpu.tInfoNoiseVarPreEq);
                copyTensorRef2Info(m_tRefNoiseIntfEstInterCtaSyncCnt, drvdUeGrpPrmsCpu.tInfoNoiseIntfEstInterCtaSyncCnt);
            }

            // Construct CfoEst tensor dimensions and linear memory allocation
            m_tRefCfoEstVec[i].desc().set(cplxTypeCh, MAX_ND_SUPPORTED, drvdUeGrpPrmsCpu.nUes, cuphy::tensor_flags::align_tight);
            m_LinearAlloc.alloc(m_tRefCfoEstVec[i]);
            copyTensorRef2Info(m_tRefCfoEstVec[i], drvdUeGrpPrmsCpu.tInfoCfoEst);
            
            uint32_t nTimeChEq = chEstSettings.enablePuschTdi ? NH : 1;
            tChEqCoefVec.push_back(cuphy::tensor_device(cplxTypeCh, NUM_ANTENNAS, CUPHY_N_TONES_PER_PRB, NUM_LAYERS, NUM_PRBS, nTimeChEq, cuphy::tensor_flags::align_tight));
            tChEqCoefVec[i].fill<cuComplex>({0.0f, 0.0f}, cuStrmMain.handle());
            cudaStreamSynchronize(cuStrmMain.handle());
            tPrmChEqCoef.desc  = tChEqCoefVec[i].desc().handle();
            tPrmChEqCoef.pAddr = tChEqCoefVec[i].addr();
            copyTensorPrm2Info(tPrmChEqCoef, drvdUeGrpPrmsCpu.tInfoEqCoef);

            // Construct ReeDiagInv (inverse of equalizer output error variance) tensor dimensions and linear memory allocation
            m_tRefReeDiagInvVec[i].desc().set(realTypeCh, CUPHY_N_TONES_PER_PRB, NUM_LAYERS, NUM_PRBS, nTimeChEq, cuphy::tensor_flags::align_tight);
            m_LinearAlloc.alloc(m_tRefReeDiagInvVec[i]);
            copyTensorRef2Info(m_tRefReeDiagInvVec[i], drvdUeGrpPrmsCpu.tInfoReeDiagInv);
            
            tChEqLLRVec.push_back(cuphy::tensor_device(CUPHY_R_16F, CUPHY_QAM_256, NUM_LAYERS, NF, NUM_DATA_SYMS, cuphy::tensor_flags::align_tight));
            tChEqLLRVec[i].fill<__half>(10.0, cuStrmMain.handle());
            cudaStreamSynchronize(cuStrmMain.handle());
            tPrmChEqLLR.desc  = tChEqLLRVec[i].desc().handle();
            tPrmChEqLLR.pAddr = tChEqLLRVec[i].addr();
            copyTensorPrm2Info(tPrmChEqLLR, drvdUeGrpPrmsCpu.tInfoLLR);

            // CFO/TA Estimation
            copyTensorRef2Info(m_tRefCfoPhaseRot, drvdUeGrpPrmsCpu.tInfoCfoPhaseRot);
            copyTensorRef2Info(m_tRefTaPhaseRot, drvdUeGrpPrmsCpu.tInfoTaPhaseRot);
            copyTensorRef2Info(m_tRefCfoHz, drvdUeGrpPrmsCpu.tInfoCfoHz);
            copyTensorRef2Info(m_tRefTaEst, drvdUeGrpPrmsCpu.tInfoTaEst);
            copyTensorRef2Info(m_tRefCfoTaEstInterCtaSyncCnt, drvdUeGrpPrmsCpu.tInfoCfoTaEstInterCtaSyncCnt);
        }

        // mark the final offset and memset
        size_t finalOffset = m_LinearAlloc.offset();
        CUDA_CHECK(cudaMemsetAsync(initAddr, 0, finalOffset - initOffset, cuStrmMain.handle()));

        //------------------------------------------------------------------
        // setup component objects
  
        cuphyPuschRxNoiseIntfEstLaunchCfgs_t m_noiseIntfEstLaunchCfgs[CUPHY_MAX_PUSCH_EXECUTION_PATHS];
        cuphyPuschRxCfoTaEstLaunchCfgs_t     m_cfoTaEstLaunchCfgs;
        cuphyPuschRxChEqLaunchCfgs_t         m_chEqCoefCompLaunchCfgs[CUPHY_PUSCH_RX_MAX_N_TIME_CH_EST];
        cuphyPuschRxChEqLaunchCfgs_t         m_chEqSoftDemapLaunchCfgs[CUPHY_MAX_PUSCH_EXECUTION_PATHS];
        
        for(int32_t idx = 0; idx < CUPHY_MAX_PUSCH_EXECUTION_PATHS; ++idx)
        {
            m_noiseIntfEstLaunchCfgs[idx].nCfgs = CUPHY_PUSCH_RX_NOISE_INTF_EST_N_MAX_HET_CFGS;
        }
        m_cfoTaEstLaunchCfgs.nCfgs = 0;
        for(int32_t chEqTimeInst = 0; chEqTimeInst < CUPHY_PUSCH_RX_MAX_N_TIME_CH_EST; ++chEqTimeInst)
        {
            m_chEqCoefCompLaunchCfgs[chEqTimeInst].nCfgs = 0;
        }
  
        status = cuphySetupPuschRxNoiseIntfEst(m_noiseIntfEstHndl,
                                               m_drvdUeGrpPrmsCpu.data(),
                                               m_drvdUeGrpPrmsGpu.data(),
                                               nUeGrps,
                                               m_nMaxPrb,
                                               chEstSettings.enableDftSOfdm,
                                               CUPHY_PUSCH_NOISE_EST_DMRS_FULL_SLOT,        // for full-slot processing
                                               enableCpuToGpuDescrAsyncCpy ? static_cast<uint8_t>(1) : static_cast<uint8_t>(0),
                                               static_cast<void*>(dynCpuDescrStartAddrs[PUSCH_NOISE_INTF_EST_EQ_TEST]),
                                               static_cast<void*>(dynGpuDescrStartAddrs[PUSCH_NOISE_INTF_EST_EQ_TEST]),
                                               &m_noiseIntfEstLaunchCfgs[CUPHY_PUSCH_FULL_SLOT_PATH],
                                               cuStrmMain.handle());
                                               
        if(CUPHY_STATUS_SUCCESS != status)
        {
            throw cuphy::cuphy_fn_exception(status, "cuphySetupPuschRxNoiseIntfEst()");
        }
        
        cuphy::buffer<float*, cuphy::pinned_alloc>  bFoCompensationBufferPtrs = std::move(cuphy::buffer<float*, cuphy::pinned_alloc>(nUes));
        float**   m_pFoCompensationBuffers = bFoCompensationBufferPtrs.addr();                                                        
        if(chEstSettings.enableCfoCorrection || chEstSettings.enableToEstimation)
        {
            status = cuphySetupPuschRxCfoTaEst(m_cfoTaEstHndl,
                                               m_drvdUeGrpPrmsCpu.data(),
                                               m_drvdUeGrpPrmsGpu.data(),
                                               m_pFoCompensationBuffers,
                                               nUeGrps,
                                               m_nMaxPrb,
                                               0,
                                               enableCpuToGpuDescrAsyncCpy ? static_cast<uint8_t>(1) : static_cast<uint8_t>(0),
                                               static_cast<void*>(dynCpuDescrStartAddrs[PUSCH_CFO_TA_EST_EQ_TEST]),
                                               static_cast<void*>(dynGpuDescrStartAddrs[PUSCH_CFO_TA_EST_EQ_TEST]),
                                               &m_cfoTaEstLaunchCfgs,
                                               cuStrmMain.handle());
            if(CUPHY_STATUS_SUCCESS != status)
            {
                throw cuphy::cuphy_fn_exception(status, "cuphySetupPuschRxCfoTaEst()");
            }
        }
        
        status = cuphySetupPuschRxChEqCoefCompute(m_chEqHndl,
                                                  m_drvdUeGrpPrmsCpu.data(),
                                                  m_drvdUeGrpPrmsGpu.data(),
                                                  nUeGrps,
                                                  m_nMaxPrb,
                                                  chEstSettings.enableCfoCorrection,
                                                  chEstSettings.enablePuschTdi,
                                                  enableCpuToGpuDescrAsyncCpy ? static_cast<uint8_t>(1) : static_cast<uint8_t>(0),
                                                  reinterpret_cast<void**>(&dynCpuDescrStartAddrs[PUSCH_CH_EQ_COEF_EQ_TEST]),
                                                  reinterpret_cast<void**>(&dynGpuDescrStartAddrs[PUSCH_CH_EQ_COEF_EQ_TEST]),
                                                  m_chEqCoefCompLaunchCfgs,
                                                  cuStrmMain.handle());
        if(CUPHY_STATUS_SUCCESS != status)
        {
            throw cuphy::cuphy_fn_exception(status, "cuphySetupPuschRxChEqCoefCompute()");
        }
        
        status = cuphySetupPuschRxChEqSoftDemap(m_chEqHndl,
                                                m_drvdUeGrpPrmsCpu.data(),
                                                m_drvdUeGrpPrmsGpu.data(),
                                                nUeGrps,
                                                m_nMaxPrb,
                                                chEstSettings.enableCfoCorrection,
                                                chEstSettings.enablePuschTdi,
                                                CUPHY_PUSCH_RX_SOFT_DEMAPPER_FULL_SLOT_SYMBOL_BITMASK, // for full-slot processing
                                                enableCpuToGpuDescrAsyncCpy ? static_cast<uint8_t>(1) : static_cast<uint8_t>(0),
                                                static_cast<void*>(dynCpuDescrStartAddrs[PUSCH_CH_EQ_SOFT_DEMAP_EQ_TEST]),
                                                static_cast<void*>(dynGpuDescrStartAddrs[PUSCH_CH_EQ_SOFT_DEMAP_EQ_TEST]),
                                                &m_chEqSoftDemapLaunchCfgs[CUPHY_PUSCH_FULL_SLOT_PATH],
                                                cuStrmMain.handle());
        if(CUPHY_STATUS_SUCCESS != status)
        {
            throw cuphy::cuphy_fn_exception(status, "cuphySetupPuschRxChEqSoftDemap()");
        }
        
        if(!enableCpuToGpuDescrAsyncCpy)
        {
            m_kernelDynDescr.asyncCpuToGpuCpy(cuStrmMain.handle());
        }

        // ------------------------------------------------------------------
        // run components
        if(chEstSettings.enableSinrMeasurement)  //|| EqCoeffAlgoIsMMSEVariant(chEstSettings.eqCoeffAlgo))
        {
            for(uint32_t hetCfgIdx = 0; hetCfgIdx < m_noiseIntfEstLaunchCfgs[CUPHY_PUSCH_FULL_SLOT_PATH].nCfgs; ++hetCfgIdx)
            {
                const CUDA_KERNEL_NODE_PARAMS& kernelNodeParamsDriver = m_noiseIntfEstLaunchCfgs[CUPHY_PUSCH_FULL_SLOT_PATH].cfgs[hetCfgIdx].kernelNodeParamsDriver;
                CU_CHECK_EXCEPTION(launch_kernel(kernelNodeParamsDriver, cuStrmMain.handle()));
            }
        }
        
        for(uint32_t hetCfgIdx = 0; hetCfgIdx < m_cfoTaEstLaunchCfgs.nCfgs; ++hetCfgIdx)
        {
            const CUDA_KERNEL_NODE_PARAMS& kernelNodeParamsDriverCfo = m_cfoTaEstLaunchCfgs.cfgs[hetCfgIdx].kernelNodeParamsDriver;
            CU_CHECK_EXCEPTION(launch_kernel(kernelNodeParamsDriverCfo, cuStrmMain.handle()));
        }
        
        int32_t bound = 1;
        if(chEstSettings.enablePuschTdi)
        {
            bound = CUPHY_PUSCH_RX_MAX_N_TIME_CH_EQ;
        }
        for(uint32_t chEqInstIdx = 0; chEqInstIdx < bound; ++chEqInstIdx)
        {
            for(uint32_t hetCfgIdx = 0; hetCfgIdx < m_chEqCoefCompLaunchCfgs[chEqInstIdx].nCfgs; ++hetCfgIdx)
            {
                const CUDA_KERNEL_NODE_PARAMS& kernelNodeParamsDriver = m_chEqCoefCompLaunchCfgs[chEqInstIdx].cfgs[hetCfgIdx].kernelNodeParamsDriver;
                CU_CHECK_EXCEPTION(launch_kernel(kernelNodeParamsDriver, cuStrmMain.handle()));
            }
        }
        
        if(bypass_soft_demapper_eval_flag==false)
        {
            for(uint32_t hetCfgIdx = 0; hetCfgIdx < m_chEqSoftDemapLaunchCfgs[CUPHY_PUSCH_FULL_SLOT_PATH].nCfgs; ++hetCfgIdx)
            {
                const CUDA_KERNEL_NODE_PARAMS& kernelNodeParamsDriver = m_chEqSoftDemapLaunchCfgs[CUPHY_PUSCH_FULL_SLOT_PATH].cfgs[hetCfgIdx].kernelNodeParamsDriver;
                CU_CHECK_EXCEPTION(launch_kernel(kernelNodeParamsDriver, cuStrmMain.handle()));
            }
        }
        
        
        //------------------------------------------------------------------
        // cleanup
        cudaStreamSynchronize(cuStrmMain.handle());
        cudaDeviceSynchronize();
        
        //------------------------------------------------------------------
        // eval for channel equalization coefficients
        double SNR_THRESH_EQCOEFF = 65.0;

        auto eqCoeffSnrMeetsThreshold = [SNR_THRESH_EQCOEFF](double snr) -> bool {
            // It is possible for snr to be infinity if the reference and test values
            // are bit-wise equivalent.
            return !isnan(snr) && (snr >= SNR_THRESH_EQCOEFF);
        };

        bool passed = true;
        for(int i = 0; i < nUeGrps; ++i) 
        {
            const double chEqCoefSnr = evalDataset.evalChEqCoef(tChEqCoefVec, i, cuStrmMain.handle());
            NVLOGC_FMT(NVLOG_PUSCH, "UE group {}: ChEq Coef SNR: {:.3f} dB", i, chEqCoefSnr);
            passed = passed && eqCoeffSnrMeetsThreshold(chEqCoefSnr);
        }
        if (passed) {
            NVLOGC_FMT(NVLOG_PUSCH, "ChEqCoef test vector {} PASSED", inputFilenameVec[0]);
        } else {
            NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT, "ChEqCoef test vector {} FAILED", inputFilenameVec[0]);
            return 1;
        }
        
        
        if(bypass_soft_demapper_eval_flag==false)
        {
            double SNR_THRESH_SOFTDEMAPPER = 61.0;
    
            auto softDemapperSnrMeetsThreshold = [SNR_THRESH_SOFTDEMAPPER](double snr) -> bool {
                // It is possible for snr to be infinity if the reference and test values
                // are bit-wise equivalent.
                return !isnan(snr) && (snr >= SNR_THRESH_SOFTDEMAPPER);
            };
            
            for(int i = 0; i < nUeGrps; ++i) 
            {
                const double chEqLLRSnr = evalDataset.evalChEqLLR(tChEqLLRVec, i, cuStrmMain.handle());
                NVLOGC_FMT(NVLOG_PUSCH, "UE group {}: SoftDemapper SNR: {:.3f} dB", i, chEqLLRSnr);
                passed = passed && softDemapperSnrMeetsThreshold(chEqLLRSnr);
            }
            if (passed) {
                NVLOGC_FMT(NVLOG_PUSCH, "SoftDemapper test vector {} PASSED", inputFilenameVec[0]);
            } else {
                NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT, "SoftDemapper test vector {} FAILED", inputFilenameVec[0]);
                return 1;
            }
        }
        else
        {
            NVLOGC_FMT(NVLOG_PUSCH, "SoftDemapper test vector {} BYPASSED", inputFilenameVec[0]);
        }
    }
    catch(const std::exception& e)
    {
        NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT, "EXCEPTION: {}\n", e.what());
        return 1;
    }
    catch(...)
    {
        NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT, "UNKNOWN EXCEPTION\n");
        return 2;
    }
    return 0;
}

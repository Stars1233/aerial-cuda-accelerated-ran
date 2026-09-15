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

#include <gsl-lite/gsl-lite.hpp>

#include "hdf5hpp.hpp"
#include "cuphy_hdf5.hpp"
#include "pusch_rx.hpp"
#include "pusch_utils.hpp"
#include "datasets.hpp"
#include "nvlog.hpp"

#include "ch_est/chest_factory.hpp"

////////////////////////////////////////////////////////////////////////
// main()
int main(int argc, char* argv[])
{
    cuphyNvlogFmtHelper nvlog_fmt("ch_est.log");
    nvlog_set_log_level(NVLOG_DEBUG);
    try
    {
        //------------------------------------------------------------------
        // Parse command line arguments
        CLI::App app{"ch_est"};
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
        const cuphyDataType_t feCplxChannelType = isChannelFp16 ? CUPHY_C_16F : CUPHY_C_32F;

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
        if (not trtYamlInput.empty()) {
            if (not std::filesystem::exists(trtYamlInput)) {
                NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT, "TrtEngine YAML config file:{} - no such file",
                           trtYamlInput);
                return -1;
            }
            staticApiDataset.puschStatPrms.puschrxChestFactorySettingsFilename = trtYamlInput.c_str();
            
            NVLOGC_FMT(NVLOG_PUSCH, "TrtEngine YAML config file:{}", trtYamlInput);
        }

        //----------------------------------------------------------------
        // Initialize CPU/GPU memory

        const uint32_t nUeGrps = dynApiDataset.cellGrpDynPrm.nUeGrps;
        const uint16_t nRxAnt  = staticApiDataset.cellStatPrmVec[0].nRxAnt;
        const cuphyPuschChEstAlgoType_t chEstAlgo = staticApiDataset.puschStatPrms.chEstAlgo;
        // We currently only support DMRS type 1 grids, which have 6 tones per PRB (every
        // other tone).
        const uint32_t DMRS_TONES_PER_PRB = 12 / 2;

        cuphy::buffer<cuphyPuschRxUeGrpPrms_t, cuphy::pinned_alloc>
                drvdUeGrpPrmsCpuBuffer(nUeGrps);
        cuphy::buffer<cuphyPuschRxUeGrpPrms_t, cuphy::device_alloc> drvdUeGrpPrmsGpuBuffer(nUeGrps);

        //------------------------------------------------------------------
        // Derive API parameters

        cuphyChEstSettings chEstSettings(&staticApiDataset.puschStatPrms, cuStrmMain.handle());

        uint8_t enableRssiMeasurement = 0;
        bool subSlotProcessingFrontLoadedDmrsEnabled = true;
        uint8_t maxDmrsMaxLen = 1;
        uint32_t maxNPrbAlloc = getMaxNPrbAlloc(&staticApiDataset.puschStatPrms);
        PuschRx::expandFrontEndParameters(&dynApiDataset.puschDynPrm, &staticApiDataset.puschStatPrms,
                                          drvdUeGrpPrmsCpuBuffer.addr(), subSlotProcessingFrontLoadedDmrsEnabled, maxDmrsMaxLen,
                                          enableRssiMeasurement, maxNPrbAlloc);

        //------------------------------------------------------------------
        // Allocate ChEst output tensor arrays in device memory

        std::vector<cuphy::tensor_device> tHEstArray;
        std::vector<cuphy::tensor_device> tLSEstArray;
        std::vector<cuphy::tensor_device> tDmrsDelayMean;
        std::vector<cuphy::tensor_device> tDmrsAccumArray;

        ///////////////////////const uint32_t maxPrbBlocks = div_round_up<uint32_t>(MAX_N_PRBS_SUPPORTED, CUPHY_PUSCH_RX_CH_EST_DELAY_EST_PRB_CLUSTER_SIZE);

        cuphyTensorPrm_t tPrms;
        for(int i = 0; i < nUeGrps; ++i)
        {
            cuphyPuschRxUeGrpPrms_t& drvdUeGrpPrmsCpu = drvdUeGrpPrmsCpuBuffer[i];
            uint16_t cellPrmDynIdx = (dynApiDataset.cellGrpDynPrm.pUeGrpPrms[i]).pCellPrm->cellPrmDynIdx;

            tLSEstArray.push_back(
                cuphy::tensor_device(CUPHY_C_32F, DMRS_TONES_PER_PRB*drvdUeGrpPrmsCpu.nPrb, drvdUeGrpPrmsCpu.nLayers, nRxAnt,
                drvdUeGrpPrmsCpu.dmrsAddlnPos + 1,cuphy::tensor_flags::align_tight));
            // We want to prevent false positive test passes, so we set the output vector to all zeros.
            // This is not strictly required if things are working correctly, but not doing so would allow
            // some or all of the output values to not be written by the kernel due to an error without
            // detecting an issue when calculating the SNR.
            if (feCplxChannelType == CUPHY_C_32F) {
                tLSEstArray[i].fill<cuComplex>({0.0f, 0.0f}, cuStrmMain.handle());
            } else if (feCplxChannelType == CUPHY_C_16F) {
                tLSEstArray[i].fill<__half2>({0.0f, 0.0f}, cuStrmMain.handle());
            } else {
                throw cuphy::cuphy_exception(CUPHY_STATUS_UNSUPPORTED_TYPE);
            }
            cudaStreamSynchronize(cuStrmMain.handle());
            tPrms.desc  = tLSEstArray[i].desc().handle();
            tPrms.pAddr = tLSEstArray[i].addr();
            copyTensorPrm2Info(tPrms, drvdUeGrpPrmsCpu.tInfoDmrsLSEst);

            tHEstArray.push_back(
                cuphy::tensor_device(CUPHY_C_32F, nRxAnt, drvdUeGrpPrmsCpu.nLayers, 12*drvdUeGrpPrmsCpu.nPrb, 
                drvdUeGrpPrmsCpu.dmrsAddlnPos + 1,cuphy::tensor_flags::align_tight));
            // We want to prevent false positive test passes, so we set the output vector to all zeros.
            // This is not strictly required if things are working correctly, but not doing so would allow
            // some or all of the output values to not be written by the kernel due to an error without
            // detecting an issue when calculating the SNR.
            if (feCplxChannelType == CUPHY_C_32F) {
                tHEstArray[i].fill<cuComplex>({0.0f, 0.0f}, cuStrmMain.handle());
            } else if (feCplxChannelType == CUPHY_C_16F) {
                tHEstArray[i].fill<__half2>({0.0f, 0.0f}, cuStrmMain.handle());
            } else {
                throw cuphy::cuphy_exception(CUPHY_STATUS_UNSUPPORTED_TYPE);
            }
            cudaStreamSynchronize(cuStrmMain.handle());
            tPrms.desc  = tHEstArray[i].desc().handle();
            tPrms.pAddr = tHEstArray[i].addr();
            copyTensorPrm2Info(tPrms, drvdUeGrpPrmsCpu.tInfoHEst);

            tDmrsDelayMean.push_back(
                cuphy::tensor_device(CUPHY_R_32F, CUPHY_PUSCH_RX_MAX_N_TIME_CH_EST, cuphy::tensor_flags::align_tight));
            tDmrsDelayMean[i].fill<float>(0.0f, cuStrmMain.handle());
            tPrms.desc  = tDmrsDelayMean[i].desc().handle();
            tPrms.pAddr = tDmrsDelayMean[i].addr();
            copyTensorPrm2Info(tPrms, drvdUeGrpPrmsCpu.tInfoDmrsDelayMean);

            tDmrsAccumArray.push_back(
                cuphy::tensor_device(CUPHY_C_32F, 2, cuphy::tensor_flags::align_tight));
            tDmrsAccumArray[i].fill<cuComplex>({0.0f, 0.0f}, cuStrmMain.handle());
            tPrms.desc  = tDmrsAccumArray[i].desc().handle();
            tPrms.pAddr = tDmrsAccumArray[i].addr();
            copyTensorPrm2Info(tPrms, drvdUeGrpPrmsCpu.tInfoDmrsAccum);

            copyTensorPrm2Info(dynApiDataset.DataIn.pTDataRx[cellPrmDynIdx], drvdUeGrpPrmsCpu.tInfoDataRx);
        }

        //------------------------------------------------------------------
        // ChEst descriptors

        // descriptors hold Kernel parameters in GPU
        size_t        statDescrSizeBytes, statDescrAlignBytes, dynDescrSizeBytes, dynDescrAlignBytes;
        // TODO no instance is needed here. leave it as is for now.
        cuphyStatus_t statusGetWorkspaceSize = cuphyPuschRxChEstGetDescrInfo(&statDescrSizeBytes,
                                                                             &statDescrAlignBytes,
                                                                             &dynDescrSizeBytes,
                                                                             &dynDescrAlignBytes);
        if(CUPHY_STATUS_SUCCESS != statusGetWorkspaceSize) throw cuphy::cuphy_exception(statusGetWorkspaceSize);

        cuphy::buffer<uint8_t, cuphy::pinned_alloc> statDescrBufCpu{statDescrSizeBytes * CUPHY_PUSCH_RX_MAX_N_TIME_CH_EST};
        cuphy::buffer<uint8_t, cuphy::pinned_alloc> dynDescrBufCpu{dynDescrSizeBytes * CUPHY_PUSCH_RX_MAX_N_TIME_CH_EST};
        cuphy::buffer<uint8_t, cuphy::device_alloc> statDescrBufGpu{statDescrSizeBytes * CUPHY_PUSCH_RX_MAX_N_TIME_CH_EST};
        cuphy::buffer<uint8_t, cuphy::device_alloc> dynDescrBufGpu{dynDescrSizeBytes * CUPHY_PUSCH_RX_MAX_N_TIME_CH_EST};

        std::array<std::uint8_t *, CUPHY_PUSCH_RX_MAX_N_TIME_CH_EST> statDescrCpuAddrs{};
        std::array<std::uint8_t *, CUPHY_PUSCH_RX_MAX_N_TIME_CH_EST> dynDescrCpuAddrs{};
        std::array<std::uint8_t *, CUPHY_PUSCH_RX_MAX_N_TIME_CH_EST> statDescrGpuAddrs{};
        std::array<std::uint8_t *, CUPHY_PUSCH_RX_MAX_N_TIME_CH_EST> dynDescrGpuAddrs{};
        statDescrCpuAddrs[0] = statDescrBufCpu.addr();
        dynDescrCpuAddrs[0] = dynDescrBufCpu.addr();
        statDescrGpuAddrs[0] = statDescrBufGpu.addr();
        dynDescrGpuAddrs[0] = dynDescrBufGpu.addr();
        for (int32_t i = 1; i < CUPHY_PUSCH_RX_MAX_N_TIME_CH_EST; i++) {
            statDescrCpuAddrs[i] = (statDescrCpuAddrs[i-1]) + statDescrSizeBytes;
            dynDescrCpuAddrs[i]  = (dynDescrCpuAddrs[i-1]) + dynDescrSizeBytes;
            statDescrGpuAddrs[i] = (statDescrGpuAddrs[i-1]) + statDescrSizeBytes;
            dynDescrGpuAddrs[i]  = (dynDescrGpuAddrs[i-1]) + dynDescrSizeBytes;
        }


        //------------------------------------------------------------------
        // Create ChEst object

        constexpr bool enableCpuToGpuDescrAsyncCpy = false; // True: static descriptors copied from CPU to GPU at creation.
                                                            // False: static descriptors populated in CPU. Caller needs to copy.
        constexpr bool enableEarlyHarqProc = false;
        auto kernelBuilder = ch_est::factory::createPuschRxChEstKernelBuilder();
        auto [puschRxChEst, statusCreate] = ch_est::factory::createPuschRxChEst(kernelBuilder.get(),
                                                                               chEstSettings,
                                                                               enableEarlyHarqProc,
                                                                               enableCpuToGpuDescrAsyncCpy,
                                                                               statDescrCpuAddrs, // being populated in init (chest vanilla)
                                                                               statDescrGpuAddrs, // same
                                                                               cuStrmMain.handle());
        if(CUPHY_STATUS_SUCCESS != statusCreate) {
            throw cuphy::cuphy_exception(statusCreate);
        }

        if(!enableCpuToGpuDescrAsyncCpy){
            for(int32_t chEstTimeInstIdx = 0; chEstTimeInstIdx < CUPHY_PUSCH_RX_MAX_N_TIME_CH_EST; ++chEstTimeInstIdx)
            {
                //CUDA_CHECK(cudaMemcpyAsync(statDescrBufGpu[chEstTimeInstIdx].addr(), statDescrBufCpu[chEstTimeInstIdx].addr(), statDescrSizeBytes, cudaMemcpyHostToDevice, cuStrmMain.handle()));
                CUDA_CHECK(cudaMemcpyAsync(statDescrGpuAddrs[chEstTimeInstIdx], statDescrCpuAddrs[chEstTimeInstIdx],
                                           statDescrSizeBytes, cudaMemcpyHostToDevice, cuStrmMain.handle()));
            }
            cudaStreamSynchronize(cuStrmMain.handle());
        }

        //------------------------------------------------------------------
        // setup ChEst object

        // setup function populates dynamic descriptor and launch config
        const uint8_t enableFrontLoadedDmrsProc = 0;
        // FIXME: in pusch_rx, these are from pDynPrm
        uint8_t preEarlyHarqWaitKernelStatus_d = 0, postEarlyHarqWaitKernelStatus_d = 0;
        uint16_t waitTimeOutPreEarlyHarqUs = 0, waitTimeOutPostEarlyHarqUs = 0;
        gsl_Expects(drvdUeGrpPrmsCpuBuffer.size() == drvdUeGrpPrmsGpuBuffer.size());
        gsl_Expects(drvdUeGrpPrmsCpuBuffer.size() == nUeGrps);
        cuphyStatus_t chEstSetupStatus = puschRxChEst->setup(kernelBuilder.get(),
                                                                gsl_lite::span(drvdUeGrpPrmsCpuBuffer.addr(), drvdUeGrpPrmsCpuBuffer.size()),
                                                                gsl_lite::span(drvdUeGrpPrmsGpuBuffer.addr(), drvdUeGrpPrmsGpuBuffer.size()),
                                                                nUeGrps,
                                                                maxDmrsMaxLen,
                                                                &preEarlyHarqWaitKernelStatus_d,
                                                                &postEarlyHarqWaitKernelStatus_d,
                                                                waitTimeOutPreEarlyHarqUs,
                                                                waitTimeOutPostEarlyHarqUs,
                                                                enableCpuToGpuDescrAsyncCpy,
                                                                gsl_lite::span(dynDescrCpuAddrs.data(), dynDescrCpuAddrs.size()),
                                                                gsl_lite::span(dynDescrGpuAddrs.data(), dynDescrGpuAddrs.size()),
                                                                enableEarlyHarqProc,
                                                                enableFrontLoadedDmrsProc,
                                                                0,          // only enabled for graph processing mode in PUSCH pipeline
                                                                nullptr,    // exec graph for device launch of sub-slot processing in PUSCH pipeline
                                                                nullptr,    // exec graph for device launch of full-slot processing in PUSCH pipeline
                                                                nullptr,    // launch config ptr in PUSCH pipeline, not used when enableEarlyHarqProc==0 and enableEarlyHarqProc==0
                                                                nullptr,    // launch config ptr in PUSCH pipeline, not used when enableEarlyHarqProc==0 and enableEarlyHarqProc==0
                                                                nullptr,    // launch config ptr in PUSCH pipeline, not used when enableEarlyHarqProc==0 and enableEarlyHarqProc==0
                                                                nullptr,    // launch config ptr in PUSCH pipeline, not used when enableEarlyHarqProc==0 and enableEarlyHarqProc==0
                                                                cuStrmMain.handle());
        if(CUPHY_STATUS_SUCCESS != chEstSetupStatus) throw cuphy::cuphy_exception(chEstSetupStatus);

        if(!enableCpuToGpuDescrAsyncCpy) {
            cudaMemcpyAsync(drvdUeGrpPrmsGpuBuffer.addr(), drvdUeGrpPrmsCpuBuffer.addr(),
                            nUeGrps * sizeof(cuphyPuschRxUeGrpPrms_t), cudaMemcpyHostToDevice, cuStrmMain.handle());
            for(int32_t chEstTimeInstIdx = 0; chEstTimeInstIdx < CUPHY_PUSCH_RX_MAX_N_TIME_CH_EST; ++chEstTimeInstIdx)
            {
                CUDA_CHECK(cudaMemcpyAsync(dynDescrGpuAddrs[chEstTimeInstIdx], dynDescrCpuAddrs[chEstTimeInstIdx],
                                           dynDescrSizeBytes, cudaMemcpyHostToDevice, cuStrmMain.handle()));
            }
            cudaStreamSynchronize(cuStrmMain.handle());
        }

        //------------------------------------------------------------------
        // run ChEst
        puschRxChEst->chestStream().launchKernels(cuStrmMain.handle());
        puschRxChEst->chestStream().launchSecondaryKernels(cuStrmMain.handle());
        //------------------------------------------------------------------
        // cleanup

        puschRxChEst.reset();
        cudaStreamSynchronize(cuStrmMain.handle());
        cudaDeviceSynchronize();

        //------------------------------------------------------------------
        // save ChEst to h5
        std::unique_ptr<hdf5hpp::hdf5_file> dbgProbeUqPtr;
        if(!outputFilename.empty())
        {
            dbgProbeUqPtr.reset(new hdf5hpp::hdf5_file(hdf5hpp::hdf5_file::create(outputFilename.c_str())));
            for(uint32_t ueGrpIdx = 0; ueGrpIdx <  nUeGrps; ++ueGrpIdx)
            {
                // write
                cuphy::write_HDF5_dataset(*dbgProbeUqPtr, tHEstArray[ueGrpIdx],
                                          std::string("HEst" + std::to_string(ueGrpIdx)).c_str());
            }
        }

        //------------------------------------------------------------------
        // chEst snr

        double SNR_THRESH = 80.0;

        if(staticApiDataset.puschStatPrms.chEstAlgo == PUSCH_CH_EST_ALGO_TYPE_RKHS)
        {
            SNR_THRESH = 30.0;
        }

        auto snrMeetsThreshold = [SNR_THRESH](double snr) -> bool {
            // It is possible for snr to be infinity if the reference and test values
            // are bit-wise equivalent.
            return !isnan(snr) && (snr >= SNR_THRESH);
        };

        // Refs are indexed by UE group, and groups whose cell does not use the
        // delay-estimation algorithm leave an empty placeholder entry.
        auto hasRef = [](const auto& refs, int idx) {
            return (static_cast<size_t>(idx) < refs.size()) && (refs[idx].addr() != nullptr);
        };

        bool passed = true;
        for(int i = 0; i < nUeGrps; ++i) {
            if (chEstAlgo == PUSCH_CH_EST_ALGO_TYPE_MULTISTAGE_MMSE_WITH_DELAY_EST && hasRef(evalDataset.tRefChEstDelayMean, i)) {
                const double delayMeanSnr = evalDataset.evalChEstDelayMean(tDmrsDelayMean, i, cuStrmMain.handle());
                NVLOGC_FMT(NVLOG_PUSCH, "UE group {}: Delay mean SNR: {:.3f} dB", i, delayMeanSnr);
                //passed = passed && snrMeetsThreshold(delayMeanSnr);
            }

            if (chEstAlgo == PUSCH_CH_EST_ALGO_TYPE_MULTISTAGE_MMSE_WITH_DELAY_EST && hasRef(evalDataset.tRefChEstLSHest, i)) {
                const double LSSnr = evalDataset.evalChEstLS(tLSEstArray, i, cuStrmMain.handle());
                NVLOGC_FMT(NVLOG_PUSCH, "UE group {}: LSEst SNR: {:.3f} dB", i, LSSnr);
                passed = passed && snrMeetsThreshold(LSSnr);
            }

            const double chEstSnr = evalDataset.evalChEst(tHEstArray, i, cuStrmMain.handle());
            NVLOGC_FMT(NVLOG_PUSCH, "UE group {}: ChEst SNR: {:.3f} dB", i, chEstSnr);
            passed = passed && snrMeetsThreshold(chEstSnr);
        }
        if (passed) {
            NVLOGC_FMT(NVLOG_PUSCH, "ChEst test vector {} PASSED", inputFilenameVec[0]);
        } else {
            NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT, "ChEst test vector {} FAILED", inputFilenameVec[0]);
            return 1;
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

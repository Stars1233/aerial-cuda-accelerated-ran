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

#include "testBench.hpp"

// cuMAC namespace
namespace cumac {

// #define exitCheckFail_ //exit when solution check fails and create TV for debugging 

void testBench::initCuda(unsigned int gpuDev)
{
  // set GPU device
  cudaSetDevice(gpuDev);
  NVLOGC_FMT(NVLOG_TESTBENCH, "cuMAC multi-cell scheduler: Running on GPU device {}", gpuDev);

  // create stream
  cudaStreamCreate(&m_cuStrm);

}

testBench::testBench(schedulerParams params, unsigned int gpuDev, unsigned int seed)
{
    // setup randomness seed
    srand(seed);

    initCuda(gpuDev);
    m_schedParams = params;

    // create network 
    net = new network(m_schedParams.direction, m_schedParams.schedulerType, 1, 1, 0);

    net->genNetTopology();
    net->genLSFading();

    // create API
    net->createAPI();

    // Initialize traffic service
    TrafficType basic_traffic(5000, 0, 1);
    TrafficType low_traffic(100, 0, 1);
    TrafficConfig traf_cfg(basic_traffic,totNumUesConst/2);
    traf_cfg.AddFlows(low_traffic,totNumUesConst/2);
    trafSvc = std::make_unique<TrafficService>(traf_cfg,net->cellGrpUeStatusCpu.get());


    // determine the number of interfering cells
    uint16_t nInterfCell = net->simParam->totNumCell - net->cellGrpPrmsGpu->nCell;
    assert(nInterfCell == 0);

    // post-eq SINR calculation
    mcSinrCalGpu = new multiCellSinrCal(net->cellGrpPrmsGpu.get());

    // PF UE selection
    mcUeSelGpu = new multiCellUeSelection(net->cellGrpPrmsGpu.get());

    // create GPU multi-cell scheduler 
    mcSchGpu = new multiCellScheduler(net->cellGrpPrmsGpu.get());

    if(m_schedParams.enable_pdsch)
    {
        // GPU layer selection
        mcLayerSelGpu = new multiCellLayerSel(net->cellGrpPrmsGpu.get());

        // GPU MCS selection
        mcsSelGpu = new mcsSelectionLUT(net->cellGrpPrmsGpu.get(), m_cuStrm);
        cudaStreamSynchronize(m_cuStrm);

        // CPU layer selection
        mcLayerSelCpu = new multiCellLayerSelCpu(net->cellGrpPrmsCpu.get());

        // CPU MCS selection
        mcsSelCpu = new mcsSelectionLUTCpu(net->cellGrpPrmsCpu.get());
    }

    if (m_schedParams.baseline == 1) { // baseline CPU RR scheduler
        mcUeSelCpu  = nullptr;
        mcSchCpu    = nullptr;
        NVLOGC_FMT(NVLOG_TESTBENCH, "Using CPU RR UE selection");
        // create CPU Round Robin UE selection
        rrUeSelCpu = new roundRobinUeSelCpu(net->cellGrpPrmsCpu.get());

        // create CPU Round Robin scheduler
        NVLOGC_FMT(NVLOG_TESTBENCH, "Using CPU RR UE scheduler");
        rrSchCpu = new roundRobinSchedulerCpu(net->cellGrpPrmsCpu.get());
    } else { // CPU reference check
        rrUeSelCpu  = nullptr;
        rrSchCpu    = nullptr;
        NVLOGC_FMT(NVLOG_TESTBENCH, "Using CPU multi-cell PF UE selection");
        // create CPU multi-cell PF UE selection
        mcUeSelCpu = new multiCellUeSelectionCpu(net->cellGrpPrmsCpu.get());

        NVLOGC_FMT(NVLOG_TESTBENCH, "Using CPU multi-cell PF scheduler");
        // create CPU multi-cell scheduler
        mcSchCpu = new multiCellSchedulerCpu(net->cellGrpPrmsCpu.get());
    }
  
    // create SVD precoder
    svdPrd = nullptr;
    if (m_schedParams.precodingScheme) {
        svdPrd = new svdPrecoding(net->cellGrpPrmsGpu.get());

        // Setup SVD precoder
        svdPrd->setup(net->cellGrpPrmsGpu.get(), m_cuStrm);
    }



}

void testBench::setup(int ttiNum)
{
    int frame_tti = ttiNum % 16;
    uint8_t lightWeight = 0;     //!< Use Lightweight kernels
    bool    csi_update  = true;  //!< CSI Update Indicator

    if(m_schedParams.periodicLightWt)
    {
        lightWeight = (frame_tti != 0);     //!< Use Lightweight kernels
        csi_update  = (frame_tti == 0);     //!< CSI Update Indicator
    }
    generateChannel(ttiNum);

    // Generate traffic
    trafSvc->Update();

    // setup API 
    net->setupAPI(m_cuStrm);
    cudaStreamSynchronize(m_cuStrm);
    NVLOGC_FMT(NVLOG_TESTBENCH, "API setup completed");

    if (csi_update) {
        csiUpdate();
    }
    ueSelection();

    // only set coordinate cell IDs and perform cell assocation at the first time slot //
    if (ttiNum == 0) {
        net->execStatus->cellIdRenew    = false;
        net->execStatus->cellAssocRenew = false;
    }

    // setup GPU multi-cell scheduler
    if (lightWeight == 1) {
        NVLOGC_FMT(NVLOG_TESTBENCH, "Multi-cell scheduler: calling light-weight kernel");
    }
    mcSchGpu->setup(net->cellGrpUeStatusGpu.get(), net->schdSolGpu.get(), net->cellGrpPrmsGpu.get(), net->simParam.get(),
                    m_schedParams.columnMajor, m_schedParams.halfPrecision, lightWeight, 2.0, m_cuStrm);
    cudaStreamSynchronize(m_cuStrm);
    
    if (m_schedParams.baseline == 1) {
        // setup CPU Round Robin scheduler
        rrSchCpu->setup(net->cellGrpUeStatusCpu.get(), net->schdSolCpu.get(), net->cellGrpPrmsCpu.get());
    } else {
        // setup CPU multi-cell scheduler
        mcSchCpu->setup(net->cellGrpUeStatusCpu.get(), net->schdSolCpu.get(), net->cellGrpPrmsCpu.get(), net->simParam.get(), m_schedParams.columnMajor);
    }


    if(m_schedParams.enable_pdsch)
    {
        // setup GPU layer selection
        mcLayerSelGpu->setup(net->cellGrpUeStatusGpu.get(), net->schdSolGpu.get(), net->cellGrpPrmsGpu.get(), 0, m_cuStrm);
        cudaStreamSynchronize(m_cuStrm);

        // setup GPU MCS selection
        mcsSelGpu->setup(net->cellGrpUeStatusGpu.get(), net->schdSolGpu.get(), net->cellGrpPrmsGpu.get(), m_cuStrm);
        cudaStreamSynchronize(m_cuStrm);

        // setup CPU layer selection
        mcLayerSelCpu->setup(net->cellGrpUeStatusCpu.get(), net->schdSolCpu.get(), net->cellGrpPrmsCpu.get());

        // setup CPU MCS selection
        mcsSelCpu->setup(net->cellGrpUeStatusCpu.get(), net->schdSolCpu.get(), net->cellGrpPrmsCpu.get());
    }

    // Update Buffer based on scheduled traffic
    std::string buffer_str;
    for(int i=0; i<net->cellGrpPrmsCpu->nUe; i++)
    {
        auto allocSol = net->getGpuAllocSol();
        auto alloc = net->getAllocBytes(i);
        auto& bufferSize = net->cellGrpUeStatusCpu->bufferSize[i];
        int sched_bytes = std::min({alloc,bufferSize});
        bufferSize -= sched_bytes;
        buffer_str += fmt::format("{:05}:{:06}\t",alloc,bufferSize);
        if(i%numUePerCellConst==numUePerCellConst-1) buffer_str+="\n";//printf("\n");
    }
    NVLOGI_FMT(NVLOG_TESTBENCH, "Scheduled: \n{}",buffer_str);
    return;
}

void testBench::generateChannel(int tti)
{
    // generate channel
    if(net->execStatus->channelRenew)
    {
        net->genFastFadingGpu(tti);
        cudaStreamSynchronize(m_cuStrm);
        NVLOGC_FMT(NVLOG_TESTBENCH, "GPU channel generated");

        if (prdSchemeConst) {
            // run SVD precoder
            svdPrd->run(net->cellGrpPrmsGpu.get());
            cudaStreamSynchronize(m_cuStrm);
            NVLOGC_FMT(NVLOG_TESTBENCH, "SVD precoder and singular values computed");
        }
    }

}

void testBench::csiUpdate()
{
    // GPU post-eq SINR calculation
    mcSinrCalGpu->setup(net->cellGrpPrmsGpu.get(), m_schedParams.columnMajor, m_cuStrm);
    cudaStreamSynchronize(m_cuStrm);
    NVLOGC_FMT(NVLOG_TESTBENCH, "CSI update: subband SINR calculation setup completed");

    mcSinrCalGpu->run(m_cuStrm);
    cudaStreamSynchronize(m_cuStrm);
    NVLOGC_FMT(NVLOG_TESTBENCH, "CSI update: subband SINR calculation run completed");
    
    //if (t == 0)
    //  mcSinrCalGpu->debugLog();

    mcSinrCalGpu->setup_wbSinr(net->cellGrpPrmsGpu.get(), m_cuStrm);
    cudaStreamSynchronize(m_cuStrm);
    NVLOGC_FMT(NVLOG_TESTBENCH, "CSI update: wideband SINR calculation setup completed");

    mcSinrCalGpu->run(m_cuStrm);
    cudaStreamSynchronize(m_cuStrm);
    NVLOGC_FMT(NVLOG_TESTBENCH, "CSI update: wideband SINR calculation run completed");

    net->cpySinrGpu2Cpu();
    NVLOGC_FMT(NVLOG_TESTBENCH, "CSI update: subband and wideband SINRS copied to CPU structures");

}

void testBench::ueSelection()
{
    // GPU UE selection
    mcUeSelGpu->setup(net->cellGrpUeStatusGpu.get(), net->schdSolGpu.get(), net->cellGrpPrmsGpu.get(), m_cuStrm);
    cudaStreamSynchronize(m_cuStrm);
    NVLOGC_FMT(NVLOG_TESTBENCH, "GPU PF UE selection setup completed");

    mcUeSelGpu->run(m_cuStrm);
    cudaStreamSynchronize(m_cuStrm);
    NVLOGC_FMT(NVLOG_TESTBENCH, "GPU PF UE selection run completed");

    // mcUeSelGpu->debugLog();

    net->ueDownSelectGpu();
    NVLOGC_FMT(NVLOG_TESTBENCH, "GPU UE downselection completed");

    //CPU UE selection
    if (m_schedParams.baseline == 1) { 
    rrUeSelCpu->setup(net->cellGrpUeStatusCpu.get(), net->schdSolCpu.get(), net->cellGrpPrmsCpu.get());
    rrUeSelCpu->run();
    NVLOGC_FMT(NVLOG_TESTBENCH, "CPU RR UE selection completed");
    } else {
    mcUeSelCpu->setup(net->cellGrpUeStatusCpu.get(), net->schdSolCpu.get(), net->cellGrpPrmsCpu.get());
    mcUeSelCpu->run();
    NVLOGC_FMT(NVLOG_TESTBENCH, "CPU PF UE selection completed");
    }

    net->ueDownSelectCpu();
    NVLOGC_FMT(NVLOG_TESTBENCH, "CPU UE downselection completed");
}

void testBench::run()
{
    // run GPU multi-cell scheduler
    mcSchGpu->run(m_cuStrm);
    cudaStreamSynchronize(m_cuStrm);

    // debug parameter logging
    // mcSchGpu->debugLog();
    
    if (m_schedParams.baseline == 1) { 
        // run CPU Round Robin scheduler
        rrSchCpu->run();
    } else {
        // run CPU multi-cell scheduler
        mcSchCpu->run();

        // debug parameter logging
        // mcSchCpu->debugLog();
    }
    NVLOGC_FMT(NVLOG_TESTBENCH, "PRB scheduling solution computed");

    if(m_schedParams.enable_pdsch)
    {
        // run GPU layer selection
        mcLayerSelGpu->run(m_cuStrm);
        cudaStreamSynchronize(m_cuStrm);
        NVLOGC_FMT(NVLOG_TESTBENCH, "GPU Layer selection solution computed");

        // run CPU layer selection
        mcLayerSelCpu->run();
        NVLOGC_FMT(NVLOG_TESTBENCH, "CPU Layer selection solution computed");

        // run GPU MCS selection
        mcsSelGpu->run(m_cuStrm);
        cudaStreamSynchronize(m_cuStrm);
        NVLOGC_FMT(NVLOG_TESTBENCH, "GPU MCS selection solution computed");
        // mcsSelGpu->debugLog();

        // run CPU MCS selection
        mcsSelCpu->run();
        NVLOGC_FMT(NVLOG_TESTBENCH, "CPU MCS selection solution computed");
        // mcsSelCpu->debugLog();
    }

    // use scheduling solution
    net->run(m_cuStrm);
    cudaStreamSynchronize(m_cuStrm);
    NVLOGC_FMT(NVLOG_TESTBENCH, "Scheduling solution transferred to host");

    return;
}

void testBench::check()
{
    bool solCheckPass = true;
    if (m_schedParams.baseline == 0) { 
        solCheckPass = net->compareCpuGpuAllocSol();
    }

#ifdef exitCheckFail_
    if (!solCheckPass) {
        saveToH5("tvDebug.h5",
                    net->cellGrpUeStatusGpu.get(),
                    net->cellGrpPrmsGpu.get(),
                    net->schdSolGpu.get());
        throw std::logic_error("Scheduler Check Failed.  Exiting");
    }
#endif
}

void testBench::updateDataRate(int ttiNum)
{
    if(m_schedParams.enable_pdsch)
    {
        net->phyAbstract(2, ttiNum);
    } else {
        net->updateDataRateUeSelCpu(ttiNum);
        net->updateDataRateGpu(ttiNum);
    }

    net->updateDataRateAllActiveUeCpu(ttiNum);
    net->updateDataRateAllActiveUeGpu(ttiNum);

}

bool testBench::validate()
{
    bool cpuGpuPerfCheckPass = true;
    if (m_schedParams.baseline == 1) {
        cpuGpuPerfCheckPass = 1;
    } else {
        cpuGpuPerfCheckPass = net->compareCpuGpuSchdPerf();
        NVLOGC_FMT(NVLOG_TESTBENCH, "CPU and GPU scheduler performance check result: {}", cpuGpuPerfCheckPass ? "PASS" : "FAIL" );
    }
    return cpuGpuPerfCheckPass;
}

void testBench::save()
{
    std::string saveTvName = std::to_string(net->cellGrpPrmsGpu->nCell) +"PC_" + (m_schedParams.direction ? "DL" : "UL") + ".h5";
    switch (m_schedParams.saveTv)
    {
    case 1: // Save GPU TVs
        saveTvName = "TV_cumac_F08-MC-CC-" + saveTvName;
        saveToH5(saveTvName,
                net->cellGrpUeStatusGpu.get(),
                net->cellGrpPrmsGpu.get(),
                net->schdSolGpu.get());
        break;
    case 2: // Save CPU TVs
        saveTvName = "CPU_TV_cumac_F08-MC-CC-" + saveTvName;
        saveToH5_CPU(saveTvName,
                net->cellGrpUeStatusCpu.get(),
                net->cellGrpPrmsCpu.get(),
                net->schdSolCpu.get());
        break;
    default:
        break;
    }

    net->writeToFileLargeNumActUe();
    net->writetoFileLargeNumActUe_short();

}

testBench::~testBench()
{
    net->destroyAPI();
    delete net;
    delete mcSinrCalGpu;
    delete mcUeSelGpu;
    delete mcSchGpu;
    if(m_schedParams.enable_pdsch)
    {
        delete mcLayerSelGpu;
        delete mcsSelGpu;
        delete mcLayerSelCpu;
        delete mcsSelCpu;
    }
    if (rrSchCpu) delete rrSchCpu;
    if (mcSchCpu) delete mcSchCpu;
    if (rrUeSelCpu) delete rrUeSelCpu;
    if (mcUeSelCpu) delete mcUeSelCpu;

    if (m_schedParams.precodingScheme) {
        svdPrd->destroy();
        delete svdPrd;
    }
}
}

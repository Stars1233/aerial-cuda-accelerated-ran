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

#include <memory>
#include "cuda.h"
#include "network.h"
#include "trafficService.hpp"

// cuMAC namespace
namespace cumac {

#define NVLOG_TESTBENCH NVLOG_TAG_BASE_CUMAC + 10

class NvlogFmtHelper
{
public:
    NvlogFmtHelper(std::string nvlog_name="nvlog.log")
    {
        // Relative path from binary to default nvlog_config.yaml
        nvlog_yaml_file.reserve(1024);
        relative_path = std::string("../../../../").append(NVLOG_DEFAULT_CONFIG_FILE);
        nv_get_absolute_path(nvlog_yaml_file.data(), relative_path.c_str());
        log_thread_id = nvlog_fmtlog_init(nvlog_yaml_file.c_str(), nvlog_name.c_str(),NULL);
        nvlog_fmtlog_thread_init();
    }

    ~NvlogFmtHelper()
    {
        nvlog_fmtlog_close(log_thread_id);
    }
private:
    std::string nvlog_yaml_file;
    std::string relative_path;
    pthread_t log_thread_id;
};


struct schedulerParams
{
    uint8_t schedulerType;   //!< Specify Scheduler type
    uint8_t columnMajor;     //!< channel matrix access type: 0 - row major, 1 - column major
    uint8_t enableHarq;
    uint8_t mcsBaseline;
    uint8_t baseline;
    bool    cpuMcsBaseline;
    uint8_t direction;       //!< Data Direction: 1 - Downlink, 0 - Uplink
    uint8_t precodingScheme; //!< Precoding Scheme: 0 - No Precoding, 1 - SVD Precoding
    uint8_t halfPrecision;   //!< Half Precision: 1- use FP16 kernels, 0 - use FP32 kernels
    uint8_t saveTv;          //!< Save TVs: 0 - Don't save, 1 - Save GPU results, 2- Save CPU results
    bool    enable_pdsch;    //!< Enable PDSCH scheduling
    bool    periodicLightWt; //!< Enable periodic lightweight kernel (don't compute full CSI results every TTI)
    /* data */
};



class testBench
{
private:
    void initCuda(unsigned int gpuDev=0);
    void generateChannel(int tti);
    void csiUpdate();
    void ueSelection();
    /* data */
    cudaStream_t m_cuStrm;
    schedulerParams m_schedParams;

    network* net;
    
    std::unique_ptr<TrafficService> trafSvc;

    mcSinrCalHndl_t mcSinrCalGpu;
    mcUeSelHndl_t   mcUeSelGpu;
    mcSchdHndl_t    mcSchGpu;

// PDSCH
    mcLayerSelHndl_t    mcLayerSelGpu;
    mcsSelLUTHndl_t     mcsSelGpu;
    mcLayerSelCpuHndl_t mcLayerSelCpu;
    mcsSelLUTCpuHndl_t  mcsSelCpu;

    rrUeSelCpuHndl_t  rrUeSelCpu;
    rrSchdCpuHndl_t   rrSchCpu;
    mcUeSelCpuHndl_t  mcUeSelCpu;
    mcSchdCpuHndl_t   mcSchCpu;

    svdPrdHndl_t svdPrd;

public:
    testBench(schedulerParams params, unsigned int gpuDev=0, unsigned int seed = 0);
    void setup(int ttiNum);
    void run();
    void check();
    void updateDataRate(int ttiNum);
    bool validate();
    void save();
    ~testBench();
};
}


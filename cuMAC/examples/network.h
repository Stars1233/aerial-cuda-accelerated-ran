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

 #pragma once

 #include "../src/api.h"
 #include "../src/cumac.h"
 #include "H5Cpp.h"
 #include "parameters.h"
 #include <cmath>
 #include "curand_kernel.h"
 #include "fading_chan.cuh"

// cuMAC namespace
namespace cumac {
 
 // execution status structure /////////////////////////////////////////
 struct cumacExecStatus {
    bool channelRenew; // controls the per slot update of the CFR channels
    bool cellIdRenew; // controls the per slot update of the set of coordinated cell IDs
    bool cellAssocRenew; // controls the per slot update of the UE-cell association profile
    bool averageRateRenew; // controls the per slot update of the per-UE average data rates and the tbErr of each UE's last transmission
 };

 struct networkData {
    uint8_t     scenarioUma;  // indication for simulation scenario: 0 - UMi, 1 - UMa
    float       carrierFreq; // carrier frequency in unit of GHz
    float       bsHeight; // BS antenna height in unit of meters
    float       ueHeight; // UE antenna height in unit of meters
    float       bsAntDownTilt; // antenna downtilting degree
    float       GEmax; // maximum directional gain of an antenna element 
    float       cellRadius; // cell radius in unit of meters
    float       bsTxPower; // BS transmit power in unit of dBm
    float       bsTxPower_perAntPrg; // per antenna and per PRG BS transmit power in unit of dBm
    float       ueTxPower; // UE transmit power in unit of dBm
    float       ueTxPower_perAntPrg; // per antenna and per PRG UE transmit power in unit of dBm
    uint16_t    numActiveUesPerCell; // number of active UEs per cell
    uint8_t     numCell; // totel number of cells. Currently support a maximum number of 21 cells
    float       sfStd; // shadow fading STD in dB
    float       noiseFloor; // noise floor in dBm
    uint16_t    numThrdBlk;
    uint16_t    numThrdPerBlk;
    float       rho;
    float       rhoPrime;
    float       minD2Bs;
        
    float       sectorOrien[3];
        
    std::vector<std::vector<float>>   bsPos;
    std::vector<std::vector<float>>   uePos;
    std::unique_ptr<float []>         rxSigPowDB = nullptr; // for DL
    std::unique_ptr<float []>         rxSigPowDB_UL = nullptr; // for UL
    float*                            rxSigPowDBGpu = nullptr; // for DL
    float*                            rxSigPowDBGpu_UL = nullptr; // for UL
    curandState_t*                    states = nullptr;

    // UE selection
    std::unique_ptr<float []> avgRatesGPU = nullptr; // per-UE average data rate for GPU scheduler
    std::unique_ptr<float []> avgRatesCPU = nullptr; // per-UE average data rate for CPU scheduler
 };

 class network {
 public:
   // API structures
    std::unique_ptr<cumacCellGrpUeStatus> cellGrpUeStatusGpu = nullptr;
    std::unique_ptr<cumacSchdSol> schdSolGpu = nullptr;
    std::unique_ptr<cumacCellGrpPrms> cellGrpPrmsGpu = nullptr;

   // API for CPU scheduler
    std::unique_ptr<cumacCellGrpUeStatus> cellGrpUeStatusCpu = nullptr;
    std::unique_ptr<cumacSchdSol> schdSolCpu = nullptr;
    std::unique_ptr<cumacCellGrpPrms> cellGrpPrmsCpu = nullptr;

   // cuMAC simulation parameter structure
    std::unique_ptr<cumacSimParam> simParam = nullptr;
   
   // execution status structure
    std::unique_ptr<cumacExecStatus> execStatus = nullptr;

    // constructor
    network(uint8_t in_DL, uint8_t extSchedulerType, uint8_t fixCellAssoc = 1, uint8_t fastFadingType = 0, bool en_traffic_gen = false, cudaStream_t strm = 0);
    // in_DL: 0 - UL, 1 - DL
    // fastFadingType: 0 - Rayleigh fading, 1 - GPU TDL CFR on Prg, 2 - GPU TDL CFR on Sc and Prg, 3 - GPU CDL CFR on Prg, 4 - GPU CDL CFR on Sc and Prg (default 0)

    ~network();
    void createAPI();
    void setupAPI(cudaStream_t strm); // requires externel synchronization
    void destroyAPI();
    void updateCurrSlotIdxPerCell(uint32_t slotIdx);  // update current slot index for each cell in the coordinated cell group

    void phyAbstract(uint8_t gpuInd, int slotIdx); // wrapper function for PHY layer abstraction
    // gpuInd: indication of GPU or CPU, 0 - CPU, 1 - GPU, 2 - both CPU and GPU
    // slotIdx: time slot index

    void genRandomChannel(); // with internel synchronization

    void run(cudaStream_t strm); // requires externel synchronization
    void updateDataRateCpu(int slotIdx); // to be modified
    void updateDataRateUeSelCpu(int slotIdx); // working
    void updateDataRateGpu(int slotIdx); // working

    void updateDataRateUpperBndCpu(int slotIdx); // to be modified
    void updateDataRateUpperBndGpu(int slotIdx); // to be modified

    uint8_t*   getCellAssoc();
    cuComplex* getEstH_fr();
    uint16_t*  getCellID();
    uint16_t*  getInterfCellID();
    int16_t*   getGpuAllocSol();
    uint32_t   getAllocBytes(int ueIdx);

    // copy computed SVD precoders from GPU mem to CPU mem for CPU version single cell PF scheduler
    void copyPrdMatGpu2Cpu(cudaStream_t strm);
    // copy cell association results to CPU
    void copyCellAssocResGpu2Cpu(cudaStream_t strm);
    // compare CPU and GPU solutions of single-cell scheduler
    bool compareCpuGpuAllocSol(); // for 4T4R SU-MIMO scheduling
    bool compareCpuGpuSchdPerf();
    // compare CPU and GPU solutions of cell association
    void compareCpuGpuCellAssocSol();

    void writeToFile();
    void saveSolH5(); // save solution to H5 file called cumacSol.h5
    void saveSolMatx(); // save solution to matX format cumacSol.mat
    void convertPrbgAllocToPrbAlloc(); // conver prbg allocation to prb allocation

    void genNetTopology();
    void genLSFading();
    void rrUeSelectionCpu(const int TTIidx); // assume that the cell assocation of all active UEs are fixed and that the number of active UEs per cell is no less than numActiveUesPerCell
    void genFastFading();
    void genFastFadingGpu(const int TTIidx);
    void genSrsChanEstGpu();
    void updateDataRateAllActiveUeGpu(const int slotIdx);
    void updateDataRateAllActiveUeCpu(const int slotIdx);
    void ueDownSelectGpu();
    void ueDownSelectCpu();
    void cpySinrGpu2Cpu();
    void testChannGen();
    void writeToFileLargeNumActUe();
    void writetoFileLargeNumActUe_short();

 private:

   const bool m_en_traffic_gen;

    uint8_t     mimoMode;
    // MIMO mode for the simulation: 0 - 4TR, 1 - 64 TR, other mode numbers not supported yet.

    // scheduler type: 0 - single-cell scheduler, 1 - multi-cell scheduler
    uint8_t     schedulerType;

    // indicator for DL/UL
    uint8_t     DL; // 1 for DL, 0 for UL

    // indicator for heterogeneous UE selection config. across cells
    uint8_t     heteroUeSelCells;

    // whether to assume fixed cell association
    uint8_t     fixCellAssociation;

    // fast fading mode
    uint8_t     m_fastFadingType; // fastFadingType: 0 - Rayleigh fading, 1 - GPU TDL CFR on Prg, 2 - GPU TDL CFR on Sc and Prg, 3 - GPU CDL CFR on Prg, 4 - GPU CDL CFR on Sc and Prg (default 0)

    // GPU address of external fast fading CFR on Prg
    cuComplex * m_externFastFadingPrbgPtr;

    // GPU address of external fast fading CFR on Sc, pointer to pointers for all links
    cuComplex ** m_externFastFadingScPtr;

    // scaling factor for channel coefficients and noise variance; 
    float       scalingFactor; // noise variance will be AFTER_SCALING_SIGMA_CONST^2 after scaling

    // PF scheduling parameters
    float       pfAvgRateUpd;
    float       initAvgRate;

    // randomness
    unsigned    seed; // randomness seed
    std::default_random_engine randomEngine; // random number generation engine
    std::uniform_real_distribution<float> uniformRealDist;
    std::unique_ptr<float[]> floatRandomArr;

    // performance metric records
    std::unique_ptr<float[]>  sumCellThrRecordsCpu = nullptr;
    std::unique_ptr<float[]>  sumCellThrRecordsGpu = nullptr;
    std::unique_ptr<float[]>  sumInsThrRecordsCpu = nullptr;
    std::unique_ptr<float[]>  sumInsThrRecordsGpu = nullptr;
    std::unique_ptr<float[]>  sumCellPfRecordsCpu = nullptr;
    std::unique_ptr<float[]>  sumCellPfRecordsGpu = nullptr;

    // MCS selection records
    std::unique_ptr<std::unique_ptr<int16_t []> []>   mcsSelRecordsCpu = nullptr;
    std::unique_ptr<std::unique_ptr<int8_t []> []>    tbErrRecordsCpu = nullptr;
    std::unique_ptr<std::unique_ptr<uint8_t []> []>   layerSelRecordsCpu = nullptr;
    std::unique_ptr<std::unique_ptr<int16_t []> []>   mcsSelRecordsGpu = nullptr;
    std::unique_ptr<std::unique_ptr<int8_t []> []>    tbErrRecordsGpu = nullptr;
    std::unique_ptr<std::unique_ptr<uint8_t []> []>   layerSelRecordsGpu = nullptr;

    // SNR-BLER tables for all MCS levels
    std::vector<float*> snrMcsArr;
    std::vector<float*> blerMscArr;
    // buffers in GPU for CPU scheduler
    cuComplex*          estH_fr_GPUforCpuSchd = nullptr;
    __nv_bfloat162*     estH_fr_GPUforCpuSchd_half = nullptr;
    uint16_t*           setSchdUePerCellTTIGpuforCpuSchd = nullptr;
    float*              avgRatesGpuforCpuSchd = nullptr;
    float*              avgRatesActUeGpuforCpuSchd = nullptr;
    int8_t*             tbErrLastGpuforCpuSchd = nullptr;
    int8_t*             tbErrLastActUeGpuforCpuSchd = nullptr;
    uint8_t*            cellAssocGpuforCpuSchd = nullptr;
    cuComplex*          prdMatGpuforCpuSchd = nullptr;
    cuComplex*          detMatGpuforCpuSchd = nullptr;
    float*              sinValGpuforCpuSchd = nullptr;

    // buffers in CPU
    std::unique_ptr<uint32_t []> currSlotIdxPerCell = nullptr; // current slot index for each cell in the coordinated cell group  
    std::unique_ptr<cuComplex []> estH_fr = nullptr;
    std::unique_ptr<cuComplex []> estH_fr_actUe = nullptr;
    std::unique_ptr<std::unique_ptr<cuComplex []> []> estH_fr_perUeBuffer = nullptr;
    std::unique_ptr<cuComplex* []> estH_fr_perUeBufferGpu = nullptr;
    std::unique_ptr<cuComplex* []> srsEstChan = nullptr;
    std::unique_ptr<int32_t* []> srsUeMap = nullptr;
    std::unique_ptr<cuComplex []> prdMat = nullptr;
    std::unique_ptr<cuComplex []> detMat = nullptr;
    std::unique_ptr<float []> sinVal = nullptr;
    std::unique_ptr<float []> avgRates = nullptr;
    std::unique_ptr<float []> avgRatesActUe = nullptr;
    std::unique_ptr<int8_t []> newDataActUe = nullptr; 
    std::unique_ptr<int8_t []> tbErrLast = nullptr; // per-UE indicator for TB decoding error of the last transmission: 0 - decoded correctly, 1 - decoding error, -1 - not scheduled for the last time slot
    std::unique_ptr<int8_t []> tbErrLastActUe = nullptr; // per-active UE indicator for TB decoding error of the last transmission
    std::unique_ptr<uint16_t []> prioWeightActUe = nullptr;
    std::unique_ptr<uint16_t []> cellId = nullptr; // IDs of coordinated cells
    std::unique_ptr<uint16_t []> interfCellId = nullptr;
    std::unique_ptr<uint8_t []> cellAssoc = nullptr; // for storing GPU cell association results for all scheduled UEs per TTI in the coordinated cells in CPU memory
    std::unique_ptr<uint8_t []> cellAssocActUe = nullptr; // for storing GPU cell association results for all active UEs in the coordinated cells in CPU memory
    std::unique_ptr<uint8_t []> numUeSchdPerCellTTIArr = nullptr; // for storing GPU array of the numbers of UEs scheduled per TTI for each cell in CPU memory
    std::unique_ptr<int16_t []> allocSol = nullptr; // for storing GPU scheduler solution in CPU mem
    std::unique_ptr<int16_t []> allocSolPrb = nullptr; // for storing GPU scheduler solution in CPU mem
    std::unique_ptr<int16_t []> mcsSelSol = nullptr; // MCS selection solution
    std::unique_ptr<float []> postEqSinr = nullptr; // for storing GPU post-equalization SINR results for all UEs, PRGs, layers in the coordinated cells in CPU memory
    std::unique_ptr<float []> wbSinr = nullptr; // wideband SINRs of all active UEs in the coordinated cells
    std::unique_ptr<uint16_t []> setSchdUePerCellTTI = nullptr; // set of global IDs of the schedule UEs per cell per TTI
    std::unique_ptr<uint8_t []> layerSelSol = nullptr; // layer selection solution for the selected UEs per TTI in the coordinated cells
    std::unique_ptr<uint16_t* []> sortedUeListGpu = nullptr; // for 64TR MU-MIMO, UE sorting solution
    std::unique_ptr<multiCellMuGrpList> muGrpListGpu = nullptr;
    // sizes of buffers
    uint32_t    hSize;
    uint32_t    hHalfSize;
    uint32_t    hActUePrdSize;
    uint32_t    hActUeSize;
    uint32_t    cidSize;
    uint32_t    assocSize;
    uint32_t    assocActUeSize;
    uint32_t    gpuAllocSolSize;
    uint32_t    cpuAllocSolSize;
    uint32_t    mcsSelSolSize;
    uint32_t    setSchdUeSolSize;
    uint32_t    pfMetricSize;
    uint32_t    pfIdSize;
    uint32_t    arSize;
    uint32_t    arActUeSize;
    uint32_t    tbeSize;
    uint32_t    tbeActUeSize;
    uint32_t    ndActUeSize;
    uint32_t    prioActUeSize;
    uint32_t    prdActUeSize;
    uint32_t    detActUeSize;
    uint32_t    prdSize;
    uint32_t    detSize;
    uint32_t    sinValActUeSize;    
    uint32_t    sinValSize;
    uint32_t    hPerUeBufferPtrSize;
    uint32_t    hPerUeBufferSize;
    uint32_t    postEqSinrSize;
    uint32_t    wbSinrSize;
    uint32_t    layerSize;
    uint32_t    numUeSchdArrSize;
    uint32_t    srsWbSnrSize;
    uint32_t    muMimoIndSize;
    uint32_t    nSCIDSize;

    // parameters
    uint16_t    nUe; // total number of scheduled UEs in the coordinated cells
    uint16_t    nActiveUe; // total number of active UEs in the coordinated cells
    uint8_t     numUeSchdPerCellTTI; // number of UEs scheduled per TTI per cell
    uint16_t    nCell; // number of coordinated cells
    uint16_t    totNumCell; // total number of cells in the network, including coordinated cells and interfering cells
    uint16_t    nInterfCell;
    uint16_t    nPrbGrp;
    uint16_t    nPrbPerGrp;
    uint8_t     nBsAnt;
    std::vector<uint16_t> bsAntSize;
    std::vector<float> bsAntSpacing;
    std::vector<float> bsAntPolarAngles;
    uint8_t bsAntPattern;
    uint8_t     nUeAnt;
    std::vector<uint16_t> ueAntSize;
    std::vector<float> ueAntSpacing;
    std::vector<float> ueAntPolarAngles;
    uint8_t ueAntPattern;
    std::vector<float> vDirection;
    float       W;
    float       slotDuration;
    float       sigma;
    float       sigmaSqrd;
    float       Pt_Rbg;
    float       Pt_rbgAnt;
    uint8_t     precodingScheme;
    uint8_t     receiverScheme;  
    uint8_t     gpuAllocType;
    uint8_t     cpuAllocType;
    float       betaCoeff; // coefficient for improving cell edge UEs' performance in multi-cell scheduling
    float       sinValThr;
    uint16_t    prioWeightStep;
    float       corrThr;
    uint16_t    numUeForGrpPerCell;
    uint16_t    nMaxActUePerCell;
    float       srsSnrThr;
    float       muCoeff;
    float       chanCorrThr;

    // CPU matrix operation algorithms
    std::unique_ptr<cpuMatAlg> matAlg = nullptr;

    // network data
    std::unique_ptr<networkData> netData = nullptr;

    // private functions
    void updateDataRatePdschCpu(int slotIdx); // to be modified
    void updateDataRatePdschGpu(int slotIdx); // to be modified

    void popHfrToPerUeBuffer();
    // PDSCH parameters
    uint16_t m_pdschNrOfDataSymb;
    uint32_t determineTbsPdsch(int rbSize, int nDataSymb, int nrOfLayers, float codeRate, int qam);

    // GPU TDL channel
    std::unique_ptr<tdlConfig_t> m_tdlCfg = nullptr; // TDL configuration struct
    std::unique_ptr<tdlChan<float, cuComplex>> m_tdl_chan = nullptr;  // ptr to tdl channel class, TODO: currently hardcode to use FP32

    // GPU CDL channel
    std::unique_ptr<cdlConfig_t> m_cdlCfg = nullptr; // CDL configuration struct
    std::unique_ptr<cdlChan<float, cuComplex>> m_cdl_chan = nullptr;  // ptr to cdl channel class, TODO: currently hardcode to use FP32
 };

 __global__ void init_curand(unsigned int t_seed, int id_offset, curandState *state);

 __global__ void ueDownSel4TrKernel(uint16_t*        setSchdUePerCellTTIGpu,
                                    uint16_t*        cellIdArr,
                                    uint8_t*         cellAssocArr,
                                    cuComplex*       channMatGpu, 
                                    cuComplex*       estH_frGpu,
                                    __nv_bfloat162*  estH_frGpu_half,
                                    float*           avgRatesActUeGpu,
                                    float*           avgRatesGpu,
                                    int8_t*          tbErrLastActUe,
                                    int8_t*          tbErrLast,
                                    cuComplex*       prdMat_actUe,
                                    cuComplex*       prdMat,
                                    cuComplex*       detMat_actUe,
                                    cuComplex*       detMat,
                                    float*           sinVal_actUe,
                                    float*           sinVal,
                                    const int        nPrbGrp, 
                                    const int        numCell, 
                                    const int        nActiveUe,
                                    const int        numSchdUePerCell,
                                    const int        numTxAnt,
                                    const int        numRxAnt);                               
                                 
 __global__ void ueDownSel64TrKernel(uint16_t**       sortedUeList,
                                     cuComplex*       channMatGpu, 
                                     cuComplex**      srsEstChan,
                                     int32_t**        srsUeMap,
                                     const int        nPrbGrp, 
                                     const int        numCell, 
                                     const int        nActiveUe,
                                     const int        numUeForGrpPerCell,
                                     const int        numTxAnt,
                                     const int        numRxAnt); 

 __global__ void genChann4TrKernel(cuComplex*       channMatGpu, 
                                   cuComplex*       channMatPrdGpu,
                                   float*           rxSigPowDBGpu,
                                   const int        nPrbGrp, 
                                   const int        numCell, 
                                   const int        numActUePerCell, 
                                   const int        numBsAnt,
                                   const int        numUeAnt,
                                   const float      rho,
                                   const float      rhoPrime,
                                   const float      scalingFactor,
                                   const int        TTIidx,
                                   curandState_t*   states,
                                   uint8_t          DL,
                                   cuComplex*       freqChanPrgPtr); // obtain freq channel on Prg, nullptr if Rayleigh fading is used

 __global__ void genSrsChanEstGpuKernel(float*        wbSinr,
                                        float*        srsWbSnr,
                                        const int     nActiveUe,
                                        const int     numUeAnt);
}
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

#include "drlMcsSelection.h"
#include "mcsSelectionDRL.h"

void usage() {
    printf("cuMAC DRL MCS selection example [options]\n");
    printf("  Options:\n");
    printf("  -i  name (path) of the folder that contains the HDF5 TVs\n");
    printf("  -m  Input trained ONNX model file\n");
    printf("  -g  GPU device index\n");
    printf("  -h  Show this help\n");
    printf("Example: './drlMcsSelection -i <tv_folder> -m <onnx_model_file> -g <GPU index>'\n");
}

int main(int argc, char* argv[]) {
try {
    // Number of GPUs.
    int32_t nGPUs = 0;
    CUDA_CHECK_ERR(cudaGetDeviceCount(&nGPUs));

    // Parse arguments.
    int iArg = 1;
    int32_t gpuId = 0;
    std::string h5TvFolder;
    std::string modelFile;

    while(iArg < argc) {
        if('-' == argv[iArg][0]) {
            switch(argv[iArg][1]) {
                case 'i': // HDF5 TV folder name
                    if(++iArg >= argc) {
                        fprintf(stderr, "ERROR: No TV folder name given.\n");
                        exit(1);
                    }
                    h5TvFolder.assign(argv[iArg++]);
                    break;
                case 'm':  // Input trained ONNX model file name
                    if(++iArg >= argc) {
                        fprintf(stderr, "ERROR: No input trained ONNX model file name given.\n");
                        exit(1);
                    }
                    modelFile.assign(argv[iArg++]);
                    break;
                case 'g':  // Select GPU
                    if((++iArg >= argc) || (1 != sscanf(argv[iArg], "%i", &gpuId)) || ((gpuId < 0) || (gpuId >= nGPUs))) {
                        fprintf(stderr, "ERROR: Invalid GPU ID (should be within [0, %d])\n", nGPUs - 1);
                        exit(1);
                    }
                    iArg++;
                    break;
                case 'h':  // Print usage
                    usage();
                    exit(0);
                    break;
                default:
                    fprintf(stderr, "ERROR: Unknown option: %s\n", argv[iArg]);
                    usage();
                    exit(1);
                    break;
            }
        } else {
            fprintf(stderr, "ERROR: Invalid command line argument: %s\n", argv[iArg]);
            exit(1);
        }
    }

    CUDA_CHECK_ERR(cudaSetDevice(gpuId));
    printf("\n=========================================\n");
    printf("=========================================\n");
    printf("cuMAC DRL MCS selection: Running on GPU device #%d\n", gpuId);

    // Create a CUDA stream
    cudaStream_t cudaStream;
    CUDA_CHECK_ERR(cudaStreamCreate(&cudaStream));
    
    // ONNX model file name.
    printf("\n=========================================\n");
    std::cout << "Model file: " << modelFile << std::endl;

    // set up parameters
    uint16_t nCell                  = cumac_ml::nCellConst;
    uint16_t nActiveUePerCell       = cumac_ml::nActiveUePerCellConst;
    uint16_t nActiveUe              = nCell*nActiveUePerCell;
    uint8_t  numUeSchdPerCellTTI    = cumac_ml::nUeSchdPerCellTTIConst;
    uint16_t nUe                    = nCell*numUeSchdPerCellTTI;
    uint16_t cqiPeriod              = cumac_ml::defCqiPeriodConst;
    uint16_t pdschPeriod            = cumac_ml::defPdschPeriodConst;
    uint16_t eventQueLen            = cumac_ml::eventQueLenConst;
    uint16_t nFeaturesPerEvent      = cumac_ml::numFeaturesPerEvent;
    uint16_t inputSize              = eventQueLen*nFeaturesPerEvent;
    uint16_t outputSize             = cumac_ml::numMcsLevels;

    assert(nActiveUePerCell == numUeSchdPerCellTTI); // number of active UEs per cell must be equal to the number of UEs scheduled per cell/TTI

    // initialize cuMAC multi-cell scheduler data structures
    std::unique_ptr<cumac::cumacSchdSol>            schdSolGpu          = std::make_unique<cumac::cumacSchdSol>();
    std::unique_ptr<cumac::cumacCellGrpPrms>        cellGrpPrmsGpu      = std::make_unique<cumac::cumacCellGrpPrms>();
    std::unique_ptr<cumac::cumacCellGrpUeStatus>    cellGrpUeStatusGpu  = std::make_unique<cumac::cumacCellGrpUeStatus>();
    std::unique_ptr<int16_t[]> mcsSelSol = std::make_unique<int16_t[]>(nUe);
    CUDA_CHECK_ERR(cudaMalloc((void**)&schdSolGpu->mcsSelSol, sizeof(int16_t)*nUe));
    std::unique_ptr<uint16_t[]> setSchdUePerCellTTI = std::make_unique<uint16_t[]>(nUe);
    for (int uIdx = 0; uIdx < nUe; uIdx++) {
        setSchdUePerCellTTI[uIdx] = uIdx;
    }
    CUDA_CHECK_ERR(cudaMalloc((void**)&schdSolGpu->setSchdUePerCellTTI, sizeof(uint16_t)*nUe));
    CUDA_CHECK_ERR(cudaMemcpyAsync((void*)schdSolGpu->setSchdUePerCellTTI, (void*)setSchdUePerCellTTI.get(), sizeof(uint16_t)*nUe, cudaMemcpyHostToDevice, cudaStream));
    CUDA_CHECK_ERR(cudaStreamSynchronize(cudaStream));

    // initialize cuMAC ML data API structure
    std::unique_ptr<cumac_ml::cumacMlDataApi> mlData = std::make_unique<cumac_ml::cumacMlDataApi>();
    mlData->numUeMlData = 0;
    mlData->maxNumUeMlData = 24*nCell;
    std::unique_ptr<int16_t[]> setUeMlData = std::make_unique<int16_t[]>(mlData->maxNumUeMlData);
    std::unique_ptr<int8_t[]> mcsLevel = std::make_unique<int8_t[]>(mlData->maxNumUeMlData);
    std::unique_ptr<int16_t[]> mcsDeltaT = std::make_unique<int16_t[]>(mlData->maxNumUeMlData);
    std::unique_ptr<int8_t[]> tbErr = std::make_unique<int8_t[]>(mlData->maxNumUeMlData);
    std::unique_ptr<int16_t[]> tbErrDeltaT = std::make_unique<int16_t[]>(mlData->maxNumUeMlData);
    std::unique_ptr<int8_t[]> cqiLevel = std::make_unique<int8_t[]>(mlData->maxNumUeMlData);
    std::unique_ptr<int16_t[]> cqiDeltaT = std::make_unique<int16_t[]>(mlData->maxNumUeMlData);
    mlData->setUeMlData = setUeMlData.get();
    mlData->mcsLevel = mcsLevel.get();
    mlData->mcsDeltaT = mcsDeltaT.get();
    mlData->tbErr = tbErr.get();
    mlData->tbErrDeltaT = tbErrDeltaT.get();
    mlData->cqiLevel = cqiLevel.get();
    mlData->cqiDeltaT = cqiDeltaT.get();
// ---------------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------------
    // create DRL MCS selection module 
    
    std::unique_ptr<cumac_ml::mcsSelectionDRL> mcsSelDRL = std::make_unique<cumac_ml::mcsSelectionDRL>(nActiveUe, nUe, eventQueLen);    
    
    // build DRL model
    mcsSelDRL->build(modelFile);
// ---------------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------------

    // set up buffers/queues
    std::vector<std::deque<cumac_ml::drlMcsSelEvent>>         eventQue(nActiveUe);
    std::vector<std::deque<uint64_t>>                               drlInfSlots(nActiveUe);
    std::vector<std::vector<std::vector<float>>>                    inputBuffers(nActiveUe);
    std::vector<std::vector<std::vector<float>>>                    outputBuffers(nActiveUe);
    std::vector<std::vector<int8_t>>                                selectedMcs(nActiveUe);

    int totNumTimeSlots;

    bool tvLoadingTest;

    if (h5TvFolder.empty()) { // no H5 TV provided - use default test scenario
        printf("\n=========================================\n");
        printf("No HDF5 test vector provided. \nUse default test scenario setting to check implementation correctness (no performance benchmarking available)\n");
        tvLoadingTest = false;
        
        cumac_ml::genDefaultEventQue(eventQue, drlInfSlots, nActiveUe, cqiPeriod, pdschPeriod);

        totNumTimeSlots = cumac_ml::defNumSlotConst;
    } else {
        printf("\n=========================================\n");
        printf("Test based on the provided HDF5 test vectors\n");
        tvLoadingTest = true;

        cumac_ml::loadFromH5_ML(h5TvFolder, nActiveUe, eventQue, drlInfSlots, inputBuffers, inputSize, outputBuffers, outputSize, selectedMcs);

        totNumTimeSlots = 0;
        for (int uIdx = 0; uIdx < nActiveUe; uIdx++) {
            if (eventQue[uIdx].back().timeStamp > totNumTimeSlots) {
                totNumTimeSlots = eventQue[uIdx].back().timeStamp;
            }
        }
    }

    // sanity check
    int numInf = drlInfSlots[0].size();
    for (int uIdx = 1; uIdx < nActiveUe; uIdx++) {
        assert(numInf == drlInfSlots[uIdx].size());
    }

    for (int idx = 0; idx < numInf; idx++) {
        uint64_t infSlot = drlInfSlots[0][idx];
        for (int uIdx = 1; uIdx < nActiveUe; uIdx++) {
            assert(infSlot == drlInfSlots[uIdx][idx]);
        }
    }

    printf("\n=========================================\n");
    printf("Event queue lengths: ");
    for (int uIdx = 0; uIdx < nActiveUe; uIdx++) {
        printf("(UE %d, %lu) ", uIdx, eventQue[uIdx].size());
    }
    printf("\n\n");

    
    // per time slot processing
    printf("\n=========================================\n");
    printf("Start per time slot processing: \n");
    int infSlotIdx = 0;
    bool success = true;
    float inputPassThreshold = cumac_ml::inPassThr;
    float outputPassThreshold = cumac_ml::outPassThr;
    const std::deque<cumac_ml::drlMcsSelEvent>* eventQueDl = mcsSelDRL->getEventQueDl();

    for (int globalSlotIdx = 0; globalSlotIdx < totNumTimeSlots; globalSlotIdx++) {
        // determine slot type: D/S/U, D slot indexes are 0, 1, 2
        int slotIdxInTddCycle = globalSlotIdx % 5;

        // only call DRL MCS selection module in D slots
        if (slotIdxInTddCycle == 0 || slotIdxInTddCycle == 1 || slotIdxInTddCycle == 2) { // D slot
            if (infSlotIdx == numInf) {
                break;
            }

            // prepare cuMAC ML data
            mlData->numUeMlData = 0;
            for (uint16_t uIdx = 0; uIdx < nActiveUe; uIdx++) {
                if (eventQue[uIdx].empty()) {
                    continue;
                }

                if (eventQue[uIdx].front().timeStamp < globalSlotIdx) {
                    mlData->setUeMlData[mlData->numUeMlData] = uIdx;
                    
                    if (eventQue[uIdx].front().mcsLevel != -1) {
                        mlData->mcsLevel[mlData->numUeMlData] = eventQue[uIdx].front().mcsLevel;
                        mlData->tbErr[mlData->numUeMlData] = eventQue[uIdx].front().tbErr;
                        mlData->mcsDeltaT[mlData->numUeMlData] = globalSlotIdx - eventQue[uIdx].front().timeStamp;
                        mlData->tbErrDeltaT[mlData->numUeMlData] = globalSlotIdx - eventQue[uIdx].front().timeStamp;
                    } else {
                        mlData->mcsLevel[mlData->numUeMlData] = -1;
                        mlData->tbErr[mlData->numUeMlData] = -1;
                        mlData->mcsDeltaT[mlData->numUeMlData] = -1;
                        mlData->tbErrDeltaT[mlData->numUeMlData] = -1;
                    }

                    if (eventQue[uIdx].front().cqiLevel != -1) {
                        mlData->cqiLevel[mlData->numUeMlData] = eventQue[uIdx].front().cqiLevel;
                        mlData->cqiDeltaT[mlData->numUeMlData] = globalSlotIdx - eventQue[uIdx].front().timeStamp;
                    } else {
                        mlData->cqiLevel[mlData->numUeMlData] = -1;
                        mlData->cqiDeltaT[mlData->numUeMlData] = -1;
                    }

                    eventQue[uIdx].pop_front();
                    mlData->numUeMlData++;
                }
            }

            // set up DRL MCS selection module
            if (drlInfSlots[0][infSlotIdx] == globalSlotIdx) {
                mcsSelDRL->setup(globalSlotIdx, mlData.get(), cellGrpPrmsGpu.get(), schdSolGpu.get(), cellGrpUeStatusGpu.get(), cudaStream);
                CUDA_CHECK_ERR(cudaStreamSynchronize(cudaStream));

                // run DRL MCS selection module
                mcsSelDRL->run(cudaStream);
                CUDA_CHECK_ERR(cudaStreamSynchronize(cudaStream));

                CUDA_CHECK_ERR(cudaMemcpyAsync((void*)mcsSelSol.get(), (void*)schdSolGpu->mcsSelSol, sizeof(int16_t)*nUe, cudaMemcpyDeviceToHost, cudaStream));
                CUDA_CHECK_ERR(cudaStreamSynchronize(cudaStream));

                if (tvLoadingTest) { // H5 TV based test
                    // check against reference.
                    const float* inputTrt = mcsSelDRL->getInputBuffer();
                    const float* outputTrt = mcsSelDRL->getOutputBuffer();

                    for (uint16_t uIdx = 0; uIdx < nActiveUe; uIdx++) {
                        if (eventQueDl[uIdx].size() < eventQueLen)
                            continue;
                        
                        // check input
                        for (int idx = 0; idx < inputSize; idx++) {
                            float gap = fabs(inputBuffers[uIdx][infSlotIdx][idx] - inputTrt[uIdx*inputSize + idx]);
                            if (gap > inputPassThreshold) {
                                success = false;
                                printf("Model input not matching: UE: %d, inference time slot: %d (#%d), gap: %.4e\n", uIdx, globalSlotIdx, infSlotIdx, gap);
                                printf("TRT input: ");
                                for (int i = 0; i < inputSize; i++) {
                                    printf("%.4e ", inputTrt[uIdx*inputSize + i]);
                                }

                                printf("\n\nRef input: ");
                                for (int i = 0; i < inputSize; i++) {
                                    printf("%.4e ", inputBuffers[uIdx][infSlotIdx][i]);
                                }
                                printf("\n\n");
                                break;
                            }
                        }

                        // check output
                        for (int idx = 0; idx < outputSize; idx++) {
                            float gap = fabs(outputBuffers[uIdx][infSlotIdx][idx] - outputTrt[uIdx*outputSize + idx]);
                            if (gap > outputPassThreshold) {
                                success = false;
                                printf("Model output not matching: UE: %d, inference time slot: %d (#%d), gap: %.4e\n", uIdx, globalSlotIdx, infSlotIdx, gap);
                                printf("TRT output: ");
                                for (int i = 0; i < outputSize; i++) {
                                    printf("%.4e ", outputTrt[uIdx*outputSize + i]);
                                }

                                printf("\n\nRef output: ");
                                for (int i = 0; i < outputSize; i++) {
                                    printf("%.4e ", outputBuffers[uIdx][infSlotIdx][i]);
                                }
                                printf("\n\n");
                                break;
                            }
                        }

                        // check selected MCS
                        if (mcsSelSol[uIdx] != selectedMcs[uIdx][infSlotIdx]) {
                            success = false;
                            printf("MCS selection result not matching: UE: %d, inference time slot: %d (#%d), TRT selected MCS: %d, ref selected MCS: %d\n", uIdx, globalSlotIdx, infSlotIdx, mcsSelSol[uIdx], selectedMcs[uIdx][infSlotIdx]);
                        }
                    }

                    if (!success) {
                        throw std::runtime_error("Failure: model inference results do not match\n");
                    }
                } else { // no H5 TV provided, default test scenario
                    // print results
                    printf("Slot #%d - selected MCS: ", globalSlotIdx);
                    for (int uIdx = 0; uIdx < nUe; uIdx++) {
                        printf("(UE %d, %d) ", uIdx, mcsSelSol[uIdx]);
                    }
                    printf("\n");
                }

                infSlotIdx++;
            } else {
                mcsSelDRL->setup(globalSlotIdx, mlData.get());
            }
        }
    }
    printf("=========================================\n");
    printf("Testing complete\n");
// ---------------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------------

    if (schdSolGpu->mcsSelSol != nullptr) CUDA_CHECK_ERR(cudaFree(schdSolGpu->mcsSelSol));
    if (schdSolGpu->setSchdUePerCellTTI != nullptr) CUDA_CHECK_ERR(cudaFree(schdSolGpu->setSchdUePerCellTTI));

    if(!success) {
        std::cout << "\033[1;32mFAILED!\033[0m" << std::endl;
        exit(1);
    }
        

    std::cout << "\033[1;32mPASSED!\033[0m" << std::endl;
    exit(0);
} catch (const std::runtime_error& e) {
        // Catch the runtime error
        std::cerr << "Runtime error: " << e.what() << std::endl;
        return EXIT_FAILURE;  
}    
}



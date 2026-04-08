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

namespace cumac_ml {

void loadFromH5_ML(const std::string&                                               tvFolderName,
                   uint16_t                                                         nActiveUe,
                   std::vector<std::deque<cumac_ml::drlMcsSelEvent>>&         eventQue,
                   std::vector<std::deque<uint64_t>>&                               drlInfSlots,
                   std::vector<std::vector<std::vector<float>>>&                    inputBuffers,
                   uint16_t                                                         inputSize,
                   std::vector<std::vector<std::vector<float>>>&                    outputBuffers,
                   uint16_t                                                         outputSize,
                   std::vector<std::vector<int8_t>>&                                selectedMcs) 
{
    // loop through all UEs
    for (uint16_t uIdx = 0; uIdx < nActiveUe; uIdx++) {
        // open TV H5 file
        std::string tvName = tvFolderName + "/tv_drl_mcs_sel_UE" + std::to_string(uIdx) + ".h5";
        H5::H5File file(tvName, H5F_ACC_RDONLY);

        // Open the event queue dataset
        H5::DataSet dataset = file.openDataSet("queue_events");

        // Get the datatype and dataspace of the dataset
        H5::DataSpace dataspace = dataset.getSpace();

        // Get the dimensions of the dataset
        hsize_t dims[2];
        dataspace.getSimpleExtentDims(dims);

        // sanity check
        assert(dims[1] == 5);

        // Create a buffer to hold the data
        std::vector<int16_t> data(dims[0] * dims[1]);

        // Read the data into the buffer
        dataset.read(data.data(), H5::PredType::NATIVE_INT16, dataspace);
        for (int i = 0; i < static_cast<int>(dims[0]); ++i) {
            cumac_ml::drlMcsSelEvent tempEvent;
            uint64_t timeStamp = static_cast<uint64_t>(data[i * dims[1] + 0]);
            tempEvent.timeStamp = timeStamp;
            tempEvent.mcsLevel = static_cast<int8_t>(data[i * dims[1] + 2]);
            tempEvent.tbErr = static_cast<int8_t>(data[i * dims[1] + 3]);
            tempEvent.cqiLevel = static_cast<int8_t>(data[i * dims[1] + 4]);

            eventQue[uIdx].push_back(tempEvent);

            uint8_t eventType = static_cast<uint8_t>(data[i * dims[1] + 1]);
            if (eventType == 1 || eventType == 3) {
                drlInfSlots[uIdx].push_back(timeStamp);
            }
        }

        std::vector<float> input(inputSize);
        std::vector<float> output(outputSize);
        int8_t selMcs;

        for (size_t idx = 0; idx < drlInfSlots[uIdx].size(); idx++) {
            std::string inputDataSet = "model_input_" +  std::to_string(idx);
            std::string outputDataSet = "model_output_" +  std::to_string(idx);
            std::string selMcsDataSet = "sel_mcs_" +  std::to_string(idx);

            dataset = file.openDataSet(inputDataSet);
            dataspace = dataset.getSpace();
            dataset.read(input.data(), H5::PredType::NATIVE_FLOAT, dataspace);

            inputBuffers[uIdx].push_back(input);

            dataset = file.openDataSet(outputDataSet);
            dataspace = dataset.getSpace();
            dataset.read(output.data(), H5::PredType::NATIVE_FLOAT, dataspace);

            outputBuffers[uIdx].push_back(output);

            dataset = file.openDataSet(selMcsDataSet);
            dataspace = dataset.getSpace();
            dataset.read(&selMcs, H5::PredType::NATIVE_INT8, dataspace);

            selectedMcs[uIdx].push_back(selMcs);
        }
    }
}

void genDefaultEventQue(std::vector<std::deque<cumac_ml::drlMcsSelEvent>>&    eventQue, 
                        std::vector<std::deque<uint64_t>>&                          drlInfSlots, 
                        uint16_t                                                    nActiveUe, 
                        uint16_t                                                    cqiPeriod, 
                        uint16_t                                                    pdschPeriod)
{
    const int firstCqiSlot = 4;
    int cqiCounter = 0;
    constexpr int firstPdschSlot = 0;
    int pdschCounter = 0;

    for (int slotIdx = 0; slotIdx < defNumSlotConst; slotIdx++) {
        int slotIdxInTddCycle = slotIdx % 5;

        bool newCqiEvent = false;
        bool newPdschEvent = false;
        
        // determine CQI event occurrence
        if (slotIdx == firstCqiSlot) {
            newCqiEvent = true;
            cqiCounter = 0;
        } else if (slotIdx > firstCqiSlot) {
            cqiCounter++;

            if (cqiCounter == cqiPeriod) {
                newCqiEvent = true;
                cqiCounter = 0;
            }
        }

        // determine PDSCH event occurrence
        if (slotIdx == firstPdschSlot) {
            newPdschEvent = true;
            pdschCounter = 0;
        } else if (slotIdx > firstPdschSlot) {
            if (slotIdxInTddCycle == 0 || slotIdxInTddCycle == 1 || slotIdxInTddCycle == 2) {
                pdschCounter++;

                if (pdschCounter == pdschPeriod) {
                    newPdschEvent = true;
                    pdschCounter = 0;
                }
            }
        }

        if (newCqiEvent || newPdschEvent) {
            for (uint16_t uIdx = 0; uIdx < nActiveUe; uIdx++) {
                drlMcsSelEvent tempEvent{};
                tempEvent.timeStamp = static_cast<uint64_t>(slotIdx);

                if (newCqiEvent) {
                    tempEvent.cqiLevel = (uIdx+slotIdx) % defNumCqi;
                } else {
                    tempEvent.cqiLevel = -1;
                }

                if (newPdschEvent) {
                    tempEvent.mcsLevel = (uIdx+slotIdx) % defNumMcs;
                    tempEvent.tbErr = (uIdx+slotIdx) % 2;
                } else {
                    tempEvent.mcsLevel = -1;
                    tempEvent.tbErr = -1;
                }
                
                eventQue[uIdx].push_back(tempEvent);

                if (newPdschEvent) {
                    drlInfSlots[uIdx].push_back(tempEvent.timeStamp);
                }
            }
        }
    }
}

}  // namespace cumac_ml

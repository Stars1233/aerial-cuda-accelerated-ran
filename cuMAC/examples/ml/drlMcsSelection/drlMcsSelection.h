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

#include "api.h"
#include "cumac.h"
#include "mcsSelectionDRL.h"
#include "h5TvCreate.h"
#include "h5TvLoad.h"

namespace cumac_ml { 

    // default scenario setting
constexpr int nCellConst = 1; // number of cells
constexpr int nActiveUePerCellConst = 6; // number of active UEs per cell
constexpr int nUeSchdPerCellTTIConst = 6; // number of scheduled UEs per cell/TTI
constexpr int defNumSlotConst = 1000; // number of time slots
constexpr int defCqiPeriodConst = 40; // CQI measurement periodicity in unit of time slots
constexpr int defPdschPeriodConst = 17; // PDSCH scheduling periodicity in unit of time slots (assuming Round-Robin scheduling)
constexpr int defNumCqi = 4; // number of CQI levels considered (from the lowest CQI 0)
constexpr int defNumMcs = 6; // number of MCS levels considered (from the lowest MCS 0)
// TDD pattern: assuming DDDSU DDDSU

constexpr float inPassThr = 1.0e-9;
constexpr float outPassThr = 1.0e-2;

// DRL constants
constexpr int eventQueLenConst = 3;

void loadFromH5_ML(const std::string&                                               tvFolderName,
                   uint16_t                                                         nActiveUe,
                   std::vector<std::deque<cumac_ml::drlMcsSelEvent>>&         eventQue,
                   std::vector<std::deque<uint64_t>>&                               drlInfSlots,
                   std::vector<std::vector<std::vector<float>>>&                    inputBuffers,
                   uint16_t                                                         inputSize,
                   std::vector<std::vector<std::vector<float>>>&                    outputBuffers,
                   uint16_t                                                         outputSize,
                   std::vector<std::vector<int8_t>>&                                selectedMcs);

void genDefaultEventQue(std::vector<std::deque<cumac_ml::drlMcsSelEvent>>& eventQue, 
                        std::vector<std::deque<uint64_t>>&       drlInfSlots, 
                        uint16_t                                 nActiveUe, 
                        uint16_t                                 cqiPeriod, 
                        uint16_t                                 pdschPeriod);

} // namespace cumac_ml

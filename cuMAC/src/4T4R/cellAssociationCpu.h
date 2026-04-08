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

// cuMAC namespace
namespace cumac {

class cellAssociationCpu {
public:
    cellAssociationCpu();
    ~cellAssociationCpu();
    cellAssociationCpu(cellAssociationCpu const&)            = delete;
    cellAssociationCpu& operator=(cellAssociationCpu const&) = delete;

    void setup(cumacCellGrpPrms* cellGrpPrms, cumacSimParam* simParam);
    void run(); // no need for simParam that has total number of cells

private:
    // dynamic descriptor
    caDynDescr_t* m_pCpuDynDesc;
};
}
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

#include <utility>

namespace nv {

    struct phy_config_option
    {
        /* data */
        bool precoding_enabled;
        bool bf_enabled;
        int enableTickDynamicSfnSlot;
        int staticPuschSlotNum;
        int staticPdschSlotNum;
        int staticPdcchSlotNum;
        int staticCsiRsSlotNum;
        int staticSsbPcid;
        int staticSsbSFN;
        int staticSsbSlotNum;
        int staticPucchSlotNum;
        bool duplicateConfigAllCells;
        /// Number of cells per DL C-plane batch fired at EOM.
        /// 0 means "unspecified/invalid" and falls back to one EOM batch for all accumulated DL cells.
        uint8_t cplane_processing_dl_batch_size{};
        /// Number of cells per UL C-plane batch fired at EOM.
        /// 0 means "unspecified/invalid" and falls back to one EOM batch for all accumulated UL cells.
        uint8_t cplane_processing_ul_batch_size{};
        /// Precomputed max number of DL batches used as UL tx-id offset base.
        uint8_t cplane_max_num_dl_batches{1};
        explicit phy_config_option():
        precoding_enabled(false),
        bf_enabled(false),
        enableTickDynamicSfnSlot(1),
        staticPuschSlotNum(-1),
        staticPdschSlotNum(-1),
        staticPdcchSlotNum(-1),
        staticCsiRsSlotNum(-1),
        staticSsbPcid(-1),
        staticSsbSFN(-1),
        staticSsbSlotNum(-1),
        staticPucchSlotNum(-1),
        duplicateConfigAllCells(false)
        {}

        phy_config_option(phy_config_option&& other):
        precoding_enabled(std::move(other.precoding_enabled)),
        bf_enabled(std::move(other.bf_enabled)),
        enableTickDynamicSfnSlot(std::move(other.enableTickDynamicSfnSlot)),
        staticPuschSlotNum(std::move(other.staticPuschSlotNum)),
        staticPdschSlotNum(std::move(other.staticPdschSlotNum)),
        staticPdcchSlotNum(std::move(other.staticPdcchSlotNum)),
        staticCsiRsSlotNum(std::move(other.staticCsiRsSlotNum)),
        staticSsbPcid(std::move(other.staticSsbPcid)),
        staticSsbSFN(std::move(other.staticSsbSFN)),
        staticSsbSlotNum(std::move(other.staticSsbSlotNum)),
        staticPucchSlotNum(std::move(other.staticPucchSlotNum)),
        duplicateConfigAllCells(std::move(other.duplicateConfigAllCells)),
        cplane_processing_dl_batch_size(other.cplane_processing_dl_batch_size),
        cplane_processing_ul_batch_size(other.cplane_processing_ul_batch_size),
        cplane_max_num_dl_batches(other.cplane_max_num_dl_batches)
        {}

        phy_config_option(phy_config_option&) = delete;
        phy_config_option& operator=(phy_config_option&) = delete;

    };

}

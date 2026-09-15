/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "slot_command/slot_command.hpp"

namespace nv::detail {

/**
 * Mimic the FH usage-done callback for stub-consumer debug runs.
 *
 * @param[in,out] info BFW coefficient memory info whose header is marked FREE
 *                     when it is currently BUSY.
 * @return true when a BUSY header was changed to FREE; false when @p info,
 *         its header, or the header state did not require cleanup. Return
 *         value must be checked.
 */
[[nodiscard]] inline bool debug_free_busy_bfw_coeff_header(
    slot_command_api::bfw_coeff_mem_info_t* info) noexcept
{
    if (info == nullptr || info->header == nullptr)
    {
        return false;
    }

    if (*info->header != static_cast<uint8_t>(slot_command_api::BFW_COFF_MEM_BUSY))
    {
        return false;
    }

    *info->header = static_cast<uint8_t>(slot_command_api::BFW_COFF_MEM_FREE);
    return true;
}

} // namespace nv::detail

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

#ifndef NV_AGGR_TASK_COMMON_HPP
#define NV_AGGR_TASK_COMMON_HPP

#include <cstdint>
#include <type_traits>

namespace nv
{

// Forward declarations — avoids circular include with nv_phy_module.hpp
class PHY_module;
class PHY_instance;
struct phy_mac_msg_desc;

/**
 * @brief Type tag written into every aggr task arg struct at construction.
 *
 * checked_cast_dl_aggr / checked_cast_ul_aggr read this field to verify
 * that the void* received by a worker function points to the expected struct
 * type, catching wrong-arg bugs (e.g. a DlAggrTaskArg* passed to a UL worker)
 * before a null dispatch pointer or bad ring_idx is dereferenced.
 */
enum class TaskArgType : uint32_t
{
    DL_AGGR = 1u, ///< DlAggrTaskArg
    UL_AGGR = 2u, ///< UlAggrTaskArg
};

/**
 * @brief Bitmask snapshot of DL/UL/CSI-RS slot-type flags captured at EOM.
 *
 * Stored in DlAggrTaskArg::slot_type_mask, consistent with active_ch_mask
 * and extensible for future slot types.
 */
enum class SlotTypeMask : uint8_t
{
    NONE  = 0,        ///< No slot type flags set.
    DL    = 1U << 0, ///< Slot carries DL transmissions (is_dl_slot_).
    UL    = 1U << 1, ///< Slot carries UL receptions    (is_ul_slot_).
    CSIRS = 1U << 2, ///< Slot carries CSI-RS           (is_csirs_slot_).
};

/**
 * @brief Common prefix for every aggr task arg struct.
 *
 * Embedding TaskArgBase as the first member of DlAggrTaskArg / UlAggrTaskArg
 * lets checked_cast helpers read the type tag through a pointer-interconvertible
 * TaskArgBase* (well-defined for standard-layout structs per [basic.compound]/4)
 * *before* down-casting to the concrete type, eliminating the previous reliance
 * on an implicit "type_tag is at offset 0" contract.
 */
struct TaskArgBase final
{
    TaskArgType type_tag{};
};
static_assert(std::is_standard_layout_v<TaskArgBase>,
              "TaskArgBase must be standard-layout for pointer-interconvertible casts");

} // namespace nv

#endif // NV_AGGR_TASK_COMMON_HPP

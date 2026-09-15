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

#if !defined(SCF_5G_FAPI_DL_STATS_HPP_INCLUDED_)
#define SCF_5G_FAPI_DL_STATS_HPP_INCLUDED_

#include <array>
#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "cuphy.h"

namespace scf_5g_fapi
{

inline constexpr std::size_t k_fapi_stats_cache_line_size{64U};
inline constexpr std::size_t k_dl_pdsch_stats_max_cells{DL_MAX_CELLS_PER_SLOT};

struct DlPdschCellDelta final
{
    std::uint64_t bytes{};
    std::uint32_t slots{};
};

static_assert(std::is_trivially_copyable_v<DlPdschCellDelta>);
static_assert(k_dl_pdsch_stats_max_cells <= 64U,
              "DlPdschStatsBatch active-cell mask supports up to 64 cells");

class alignas(k_fapi_stats_cache_line_size) DlPdschStatsBatch final
{
public:
    void add_pdsch_bytes(std::uint32_t cell_id, std::uint64_t bytes) noexcept
    {
        if (cell_id >= deltas_.size()) [[unlikely]] {
            return;
        }
        deltas_[cell_id].bytes += bytes;
        if (bytes != 0U) {
            mark_active(cell_id);
        }
    }

    void add_pdsch_slot(std::uint32_t cell_id) noexcept
    {
        if (cell_id >= deltas_.size()) [[unlikely]] {
            return;
        }
        ++deltas_[cell_id].slots;
        mark_active(cell_id);
    }

    [[nodiscard]] DlPdschCellDelta cell_delta(std::uint32_t cell_id) const noexcept
    {
        if (cell_id >= deltas_.size()) [[unlikely]] {
            return {};
        }
        return deltas_[cell_id];
    }

    void set_cell_delta(std::uint32_t cell_id, DlPdschCellDelta delta) noexcept
    {
        if (cell_id >= deltas_.size()) [[unlikely]] {
            return;
        }
        deltas_[cell_id] = delta;
        if (delta.bytes != 0U || delta.slots != 0U) {
            mark_active(cell_id);
        } else {
            clear_active(cell_id);
        }
    }

    [[nodiscard]] const DlPdschCellDelta& delta(std::uint32_t cell_id) const noexcept
    {
        static constexpr DlPdschCellDelta k_empty{};
        if (cell_id >= deltas_.size()) [[unlikely]] {
            return k_empty;
        }
        return deltas_[cell_id];
    }

    [[nodiscard]] std::uint64_t active_mask() const noexcept
    {
        return active_mask_;
    }

    void reset() noexcept
    {
        deltas_ = {};
        active_mask_ = 0U;
    }

private:
    void mark_active(std::uint32_t cell_id) noexcept
    {
        active_mask_ |= (std::uint64_t{1U} << cell_id);
    }

    void clear_active(std::uint32_t cell_id) noexcept
    {
        active_mask_ &= ~(std::uint64_t{1U} << cell_id);
    }

    std::array<DlPdschCellDelta, k_dl_pdsch_stats_max_cells> deltas_{};
    std::uint64_t active_mask_{};
};

static_assert(std::is_trivially_copyable_v<DlPdschStatsBatch>);
static_assert(alignof(DlPdschStatsBatch) == k_fapi_stats_cache_line_size);

} // namespace scf_5g_fapi

#endif // SCF_5G_FAPI_DL_STATS_HPP_INCLUDED_

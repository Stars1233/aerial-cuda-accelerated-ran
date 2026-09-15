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

#define TAG (NVLOG_TAG_BASE_CUPHY_DRIVER + 3) // "DRV.CTX"

#include "prach_offload_reconfig.hpp"
#include "phyprach_aggr.hpp"
#include "cell.hpp"
#include "cuphydriver_api.hpp"  // cell_phy_info
#include "locks.hpp"            // Mutex
#include "nvlog.hpp"

#include <gsl-lite/gsl-lite.hpp>
#include <range/v3/view/enumerate.hpp>

#include <cstddef>
#include <cstdint>

namespace {
// Pending swaps are tracked in a 64-bit mask; guard the aggregator count so it
// cannot silently overflow.
inline constexpr std::size_t MAX_PRACH_AGGREGATORS{64};
} // namespace

tl::expected<void, PrachStageError> PrachOffloadReconfig::stageCellConfig(const Cell& cell, const cell_phy_info& cell_pinfo)
{
    staged_cell_occa_start_idx_.reset();

    for(auto&& [i, aggr] : ranges::views::enumerate(aggrs_))
    {
        const auto staged = aggr->stageConfig(cell.getId(), cell_pinfo);
        if(!staged)
        {
            NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT,
                "PrachOffloadReconfig::stageCellConfig: aggr {} stageConfig failed", i);
            discard();
            return tl::unexpected(staged.error());
        }
        if(*staged)
        {
            // Every PRACH aggregator holds all cells in identical layout (createPhyObj
            // populates each from the same cell list in the same order), so every
            // aggregator that takes the grow path reports the SAME occasion start index for
            // this cell. Enforce that agreement instead of silently letting a later writer
            // win -- a mismatch would mean the aggregators' occasion layouts have diverged.
            gsl_Expects(!staged_cell_occa_start_idx_ || *staged_cell_occa_start_idx_ == **staged);
            staged_cell_occa_start_idx_ = **staged;
        }
    }
    return {};
}

tl::expected<void, PrachStageError> PrachOffloadReconfig::createObjects()
{
    for(auto& aggr : aggrs_)
    {
        if(auto created = aggr->createNewPhyObjFromStagedConfig(); !created)
        {
            // deleteTempPhyObj() guards on handle_temp != nullptr, so aggregators
            // that never ran (or failed) are skipped safely.
            deleteTempObjects();
            return tl::unexpected(created.error());
        }
    }
    return {};
}

void PrachOffloadReconfig::commit(Cell& cell)
{
    for(auto& aggr : aggrs_)
    {
        aggr->commitStagedConfig();
    }

    if(staged_cell_occa_start_idx_)
    {
        // No cast: staged index is uint16_t, matching Cell::prachOccaPrmStatIdx.
        // The occasion start index is produced by applyPrachUpdate as a uint16
        // widening of the stored cuPHY field, so no narrowing reaches here.
        cell.setPrachOccaPrmStatIdx(*staged_cell_occa_start_idx_);
        staged_cell_occa_start_idx_.reset();
    }
}

void PrachOffloadReconfig::discard()
{
    for(auto& aggr : aggrs_)
    {
        aggr->discardStagedConfig();
    }
    staged_cell_occa_start_idx_.reset();
}

bool PrachOffloadReconfig::armable() const noexcept
{
    // PRACH aggregators per context are few (well under 64); guard the bound so
    // the offload-owned pending bitmask cannot silently overflow.
    return aggrs_.size() <= MAX_PRACH_AGGREGATORS;
}

void PrachOffloadReconfig::arm()
{
    const auto n_aggr = aggrs_.size();
    // Precondition: reconfigure() verified armable() before commit(), so the bound holds.
    gsl_Expects(n_aggr <= MAX_PRACH_AGGREGATORS);

    // Legacy parity: arm the incremental per-aggregator handle swap and return without
    // blocking. commit() has already published the live vectors; each aggregator swaps
    // its cuPHY handle on a later SLOT.IND tryCommit() when idle (our analog of legacy's
    // num_new_prach_handles countdown in getNextPrachAggr), and the config converges over
    // the next few slots. The legacy active-cell path (cell_update_config_func) never
    // waits for the swap to finish, so there is no bounded wait here and thus no timeout /
    // partial-commit state to surface to the caller.
    handover_.arm(n_aggr);
}

void PrachOffloadReconfig::tryCommit()
{
    // The handover state machine owns the phase/notify; this drain performs the
    // idle scan + incremental swap under the same PRACH lock as getNextPrachAggr()'s
    // reservation, so a swap cannot race a reserve. No early return inside the lock.
    handover_.tryCommit([this](std::uint64_t& pending_mask, std::int32_t& pending_count) {
        aggr_lock_.lock();
        auto _ = gsl_lite::finally([this] { aggr_lock_.unlock(); });
        for(std::size_t i = 0; i < aggrs_.size() && i < MAX_PRACH_AGGREGATORS; ++i)
        {
            const std::uint64_t bit = std::uint64_t{1} << i;
            if((pending_mask & bit) && !aggrs_[i]->isActive())
            {
                aggrs_[i]->changePhyObj();
                pending_mask &= ~bit;
                --pending_count;
            }
        }
    });
}

void PrachOffloadReconfig::cancel()
{
    handover_.cancel();
}

bool PrachOffloadReconfig::armed() const
{
    return handover_.armed();
}

void PrachOffloadReconfig::deleteTempObjects()
{
    for(auto& aggr : aggrs_)
    {
        aggr->deleteTempPhyObj();
    }
}

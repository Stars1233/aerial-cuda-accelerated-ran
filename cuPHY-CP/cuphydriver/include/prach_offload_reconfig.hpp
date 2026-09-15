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

#ifndef PRACH_OFFLOAD_RECONFIG_HPP
#define PRACH_OFFLOAD_RECONFIG_HPP

#include "prach_offload_handover.hpp"
#include "prach_stage_error.hpp"

#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include <tl/expected.hpp>

class PhyPrachAggr;
class Cell;
class Mutex;
struct cell_phy_info;

/**
 * Outcome of PrachOffloadReconfig::reconfigure() -- the lifecycle stage reached, so the
 * caller can map/log success vs the specific failure without knowing the internal order.
 */
enum class PrachReconfigOutcome
{
    Committed,           //!< Staged, temp handles created, static config applied, committed, swap armed.
    Busy,                //!< A prior handover is still draining; nothing staged (caller may retry later).
    StageFailed,         //!< stageCellConfig failed (staged config discarded).
    CreateFailed,        //!< Temp (PONG) handle creation failed (temp handles cleaned up).
    StaticConfigFailed,  //!< The injected static cell-config step failed (staged config discarded).
    ArmFailed,           //!< Aggregator count exceeds the 64-bit pending mask (build/config error); rejected before any staging/commit.
};

/**
 * Maps a staging failure cause to the caller-facing outcome. reconfigure() consumes the
 * typed PrachStageError through this (rather than inferring the outcome from which call
 * failed), so the outcome stays tied to the actual cause -- a future staging path that
 * produces a different error surfaces the right outcome without editing reconfigure().
 */
[[nodiscard]] inline constexpr PrachReconfigOutcome toReconfigOutcome(PrachStageError err) noexcept
{
    switch(err)
    {
        case PrachStageError::StageRejected:    return PrachReconfigOutcome::StageFailed;
        case PrachStageError::TempCreateFailed: return PrachReconfigOutcome::CreateFailed;
    }
    return PrachReconfigOutcome::StageFailed; // all cases return above; satisfies -Wreturn-type
}

/**
 * Drives the offload (`nonslot_lp`) active-cell PRACH reconfiguration.
 *
 * Encapsulates the whole offload reconfig feature — the handover state machine
 * and the per-aggregator stage/create/commit/discard orchestration — so it stays
 * out of the PhyDriverCtx god-object and the offload-owned state is private. It
 * holds non-owning references to the shared PRACH aggregator pool and its lock
 * (legacy `getNextPrachAggr()` shares them and must stay untouched). Lifetime:
 * constructed and owned by PhyDriverCtx, declared after the referenced members.
 * The pool reference stays valid as the pool is populated after construction.
 *
 * Threading: reconfigure() (and its internal stage/create/commit/discard/arm steps) runs
 * on the `nonslot_lp` worker; tryCommit runs on the slot (SLOT.INDICATION) thread; cancel
 * runs on the reset/shutdown thread. Lock order is fixed and is a property of this type:
 * the handover mutex (inside PrachOffloadHandover) is always taken before the PRACH
 * aggregator lock -- a hook reaching from inside aggr_lock_ back into the handover would
 * invert it.
 */
class PrachOffloadReconfig final
{
public:
    PrachOffloadReconfig(std::vector<std::unique_ptr<PhyPrachAggr>>& aggrs, Mutex& aggr_lock)
        : aggrs_(aggrs)
        , aggr_lock_(aggr_lock)
    {
    }

    PrachOffloadReconfig(const PrachOffloadReconfig&)            = delete;
    PrachOffloadReconfig& operator=(const PrachOffloadReconfig&) = delete;
    PrachOffloadReconfig(PrachOffloadReconfig&&)                 = delete;
    PrachOffloadReconfig& operator=(PrachOffloadReconfig&&)      = delete;
    ~PrachOffloadReconfig()                                      = default;

    /**
     * Runs the full active-cell PRACH reconfiguration as one operation, owning the step
     * ordering and the rollback rules so callers never re-derive them (Tell-Don't-Ask):
     *
     *   stage -> create temp handles -> commit_static_config -> publish (commit) -> arm
     *
     * Any pre-commit failure discards the staged config. The static cell-config step runs
     * after the temp handles exist and before the PRACH commit (it has no dependency on the
     * committed vectors, and its failure must leave the PRACH config unpublished). After
     * commit() the incremental handle swap is armed and drains on later SLOT.INDICATION
     * tryCommit() calls -- no blocking wait (legacy parity).
     *
     * Rejects re-entry (returns Busy, stages nothing) while a prior handover is still armed
     * and draining: a fresh stageConfig() would overwrite the staged_ buffers that the prior
     * handover's still-active (pre-swap) handle reads from, so the new reconfig must wait until
     * the in-flight swap has drained.
     *
     * @tparam CommitStaticConfig Invocable returning int (0 == success). Taken by forwarding
     *         reference (no std::function, no heap), and kept a parameter rather than a
     *         member because the static cell-config step touches PhyDriverCtx/Cell state,
     *         not PRACH state.
     * @param[in,out] cell The cell being reconfigured.
     * @param[in] cell_pinfo New cell PHY info with the updated PRACH configuration.
     * @param[in] commit_static_config Caller-injected static cell-config step.
     * @return PrachReconfigOutcome::Committed on full success; otherwise the failing stage.
     */
    template <typename CommitStaticConfig>
    [[nodiscard]] PrachReconfigOutcome reconfigure(Cell&                cell,
                                                   const cell_phy_info& cell_pinfo,
                                                   CommitStaticConfig&& commit_static_config)
    {
        // Aggregator-count bound is a build invariant (pool size is fixed for this reconfig).
        // Check it up front so a violation is a pre-commit rejection -- never a half-committed
        // state where commit() has published but arm() then reports failure.
        if(!armable())
        {
            return PrachReconfigOutcome::ArmFailed;
        }
        if(handover_.armed())
        {
            // A prior handover's per-slot tryCommit() swap is still in flight. Re-staging now
            // would overwrite the staged_ buffers the prior, not-yet-swapped handle still reads.
            return PrachReconfigOutcome::Busy;
        }
        if(const auto staged = stageCellConfig(cell, cell_pinfo); !staged.has_value())
        {
            return toReconfigOutcome(staged.error());
        }
        if(const auto created = createObjects(); !created.has_value())
        {
            return toReconfigOutcome(created.error());
        }
        if(commit_static_config() != 0)
        {
            discard();
            return PrachReconfigOutcome::StaticConfigFailed;
        }
        commit(cell);
        arm();  // guaranteed to succeed: armable() was verified above, before any commit
        return PrachReconfigOutcome::Committed;
    }

    /**
     * Slot-boundary hook: swaps each idle pending PRACH aggregator to its new handle.
     *
     * Lock-free no-op when no handover is armed. Called once per slot from the
     * SLOT.INDICATION handler.
     */
    void tryCommit();

    /**
     * Cancels an armed handover and wakes any waiter (for async reset/reconnect/shutdown).
     *
     * @note Currently unwired by design: PhyDriverCtx (and thus this handover) is long-lived,
     *       and the legacy in-flight swap counter (num_new_prach_handles) is likewise not
     *       reset on reconnect -- so leaving an armed handover to drain on resume matches
     *       legacy behavior. This hook exists so a future reset path can cancel without
     *       diverging from legacy first; wire it only alongside the legacy reset.
     */
    void cancel();

    /**
     * @return true if a handover is currently armed.
     */
    [[nodiscard]] bool armed() const;

private:
    // Lifecycle steps, orchestrated only by reconfigure() (ordering + rollback live there).

    /**
     * Stages the PRACH reconfiguration into per-aggregator copies (no live change).
     * @param[in] cell The cell being reconfigured.
     * @param[in] cell_pinfo New cell PHY info with the updated PRACH configuration.
     * @return Empty on success; tl::unexpected(PrachStageError::StageRejected) if any
     *         aggregator failed to stage (all discarded).
     */
    [[nodiscard]] tl::expected<void, PrachStageError> stageCellConfig(const Cell& cell, const cell_phy_info& cell_pinfo);

    /**
     * Creates the temporary (PONG) PRACH objects from the staged config. Does not set
     * num_new_prach_handles, so legacy getNextPrachAggr() stays inert.
     * @return Empty on success; tl::unexpected(PrachStageError::TempCreateFailed) on
     *         creation failure (temp handles cleaned up).
     */
    [[nodiscard]] tl::expected<void, PrachStageError> createObjects();

    /**
     * Commits staged PRACH config across all aggregators into live state.
     * @param[in,out] cell The cell whose PRACH occasion start index may be updated.
     */
    void commit(Cell& cell);

    /** Discards staged PRACH config across all aggregators (pre-commit rollback). */
    void discard();

    /**
     * True if the aggregator pool fits the 64-bit pending mask, i.e. arm() cannot fail.
     * Checked by reconfigure() before any commit so a bound violation stays pre-commit.
     */
    [[nodiscard]] bool armable() const noexcept;

    /**
     * Arms the incremental per-aggregator handle swap and returns WITHOUT blocking (legacy
     * parity: never waits for the swap; drains on later SLOT.IND tryCommit() calls, the
     * analog of legacy's num_new_prach_handles countdown in getNextPrachAggr).
     * Precondition: armable() -- reconfigure() verifies it before commit().
     */
    void arm();

    void deleteTempObjects();  //!< Free every aggregator's temp (PONG) handle.

    std::vector<std::unique_ptr<PhyPrachAggr>>& aggrs_;      //!< Shared PRACH aggregator pool (non-owning).
    Mutex&                                      aggr_lock_;  //!< Shared PRACH reservation lock (non-owning).
    PrachOffloadHandover                        handover_;   //!< Handover state machine.
    std::optional<std::uint16_t>                staged_cell_occa_start_idx_;  //!< Deferred cell occasion start index.
};

#endif // PRACH_OFFLOAD_RECONFIG_HPP

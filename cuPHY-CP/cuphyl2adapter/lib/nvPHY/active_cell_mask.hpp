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

#ifndef ACTIVE_CELL_MASK_HPP
#define ACTIVE_CELL_MASK_HPP

#include <atomic>
#include <bit>
#include <cstdint>

namespace nv {

/**
 * Active-cell bitmap with deferred, production-gated slot-boundary commit, so the slot
 * end-of-messages (EOM) expectation only ever contains cells that have actually started
 * producing -- a just-started, not-yet-producing cell never delays the cells already running.
 *
 * Three masks: STAGED (what L2 intends to run), PRODUCED (cells that have sent at least one
 * slot message since being staged), and COMMITTED (what the EOM expectation waits on; the
 * per-ring snapshot is taken from this):
 *
 *  - START   -> stageStarted(): cell enters STAGED only.
 *  - 1st msg -> markProduced(): cell enters PRODUCED (sticky). Records that the cell is
 *               actually producing; does not commit it.
 *  - STOP    -> stageStopped(): cell leaves STAGED and PRODUCED, so a later restart must
 *               re-prove production before being committed again.
 *  - slot    -> commitStaged(): committed = staged & produced. Synced once per SLOT.IND, then
 *    boundary   the per-slot snapshot is taken from committed -- so a cell's membership is fixed
 *               for the whole slot and cannot change mid-slot.
 *
 * Net effect (defer by at most one slot): a cell joins the EOM expectation at the first slot
 * boundary AFTER it has been both staged and seen producing, and leaves it at the boundary
 * after a STOP. Because a cell is committed only for slots it is genuinely producing:
 *  - established cells are never made to wait on a not-yet-producing new cell (no late
 *    finalize / C-plane timing hit on running cells), and
 *  - the per-slot snapshot is stable, so an established cell's EOM cannot fire and then be
 *    reopened by a late-arriving join (which would double-submit the ring slot).
 *
 * Cost: the new cell's FIRST produced slot is not yet committed (commit lands the next
 * boundary), so the established cells do not wait for it on that slot; at most that single
 * startup-transient slot of the new cell's data may be dropped. From the next slot it is
 * committed and runs normally.
 *
 * Invariants: committed is always a subset of (staged & produced).
 *
 * Mutated on the message-processing thread; STAGED is also written by the offload LP worker on
 * START/STOP. All members are atomic so cross-thread reads observe a consistent published value.
 */
class ActiveCellMask final
{
public:
    static constexpr std::uint16_t kMaxCells = 64; //!< Bit width of the masks.

    /** Constructs an empty mask (no staged, produced, or committed cells). */
    ActiveCellMask() = default;

    /**
     * Constructs from explicit mask values (used to move-rebuild an owning object whose
     * atomics cannot themselves be moved).
     *
     * @param[in] staged    Initial staged mask.
     * @param[in] produced  Initial produced mask.
     * @param[in] committed Initial committed mask.
     */
    explicit ActiveCellMask(std::uint64_t staged, std::uint64_t produced, std::uint64_t committed) noexcept
        : staged_(staged)
        , produced_(produced)
        , committed_(committed)
    {
    }

    ActiveCellMask(const ActiveCellMask&)            = delete;
    ActiveCellMask& operator=(const ActiveCellMask&) = delete;
    ActiveCellMask(ActiveCellMask&&)                 = delete;
    ActiveCellMask& operator=(ActiveCellMask&&)      = delete;
    ~ActiveCellMask()                                = default;

    /** @return The committed mask (what the EOM expectation waits on). */
    [[nodiscard]] std::uint64_t committed() const noexcept
    {
        return committed_.load(std::memory_order_acquire);
    }

    /** @return The staged mask (cells L2 intends to run). */
    [[nodiscard]] std::uint64_t staged() const noexcept
    {
        return staged_.load(std::memory_order_acquire);
    }

    /** @return The produced mask (staged cells that have sent at least one slot message). */
    [[nodiscard]] std::uint64_t produced() const noexcept
    {
        return produced_.load(std::memory_order_acquire);
    }

    /** @return Number of committed (active, producing) cells. */
    [[nodiscard]] std::uint32_t committedCount() const noexcept
    {
        return static_cast<std::uint32_t>(std::popcount(committed()));
    }

    /**
     * START: stage the cell. It joins COMMITTED (and the EOM expectation) at the first
     * commitStaged() boundary AFTER it has also produced a slot message (see markProduced()).
     *
     * @param[in] cell_id Cell to stage (ignored if >= kMaxCells).
     */
    void stageStarted(std::uint16_t cell_id) noexcept
    {
        const std::uint64_t bit = bitFor(cell_id);
        if(bit != 0ULL)
        {
            staged_.fetch_or(bit, std::memory_order_acq_rel);
        }
    }

    /**
     * STOP: remove the cell from STAGED and PRODUCED. It leaves COMMITTED at the next
     * commitStaged(); clearing PRODUCED forces a later restart to re-prove production before
     * it can be committed again.
     *
     * @param[in] cell_id Cell to stage-stop (ignored if >= kMaxCells).
     */
    void stageStopped(std::uint16_t cell_id) noexcept
    {
        const std::uint64_t bit = bitFor(cell_id);
        if(bit != 0ULL)
        {
            staged_.fetch_and(~bit, std::memory_order_acq_rel);
            produced_.fetch_and(~bit, std::memory_order_acq_rel);
        }
    }

    /**
     * First slot message: mark the cell as producing. Idempotent and gated on STAGED, so a
     * straggler message for a cell that is not currently staged (e.g. arriving after a STOP)
     * cannot pre-set PRODUCED and let a later restart commit without re-proving production.
     * Does NOT commit the cell -- commit happens at the next commitStaged() boundary.
     *
     * @param[in] cell_id Cell whose slot message was just stored (ignored if >= kMaxCells).
     */
    void markProduced(std::uint16_t cell_id) noexcept
    {
        const std::uint64_t bit = bitFor(cell_id);
        if(bit != 0ULL && (staged_.load(std::memory_order_acquire) & bit) != 0ULL)
        {
            produced_.fetch_or(bit, std::memory_order_acq_rel);
        }
    }

    /**
     * Slot-boundary commit: committed = staged & produced. A staged-and-producing cell joins
     * COMMITTED and a staged-out STOP leaves it, both at this boundary. Called once per slot
     * from the SLOT.IND handler, before the per-slot snapshot is taken from committed().
     *
     * @return The committed mask after the sync (equal to staged & produced).
     */
    [[nodiscard]] std::uint64_t commitStaged() noexcept
    {
        const std::uint64_t next = staged_.load(std::memory_order_acquire) &
                                   produced_.load(std::memory_order_acquire);
        committed_.store(next, std::memory_order_release);
        return next;
    }

    /**
     * Adds a cell to STAGED, PRODUCED, and COMMITTED at once (legacy serial START path, which
     * commits immediately with no defer). PRODUCED is set too so the invariant
     * committed subset of (staged & produced) holds even if commitStaged() later runs.
     *
     * @param[in] cell_id Cell to activate (ignored if >= kMaxCells).
     * @return The committed mask after the addition.
     */
    [[nodiscard]] std::uint64_t activateImmediate(std::uint16_t cell_id) noexcept
    {
        const std::uint64_t bit = bitFor(cell_id);
        if(bit == 0ULL)
        {
            return committed();
        }
        staged_.fetch_or(bit, std::memory_order_acq_rel);
        produced_.fetch_or(bit, std::memory_order_acq_rel);
        return committed_.fetch_or(bit, std::memory_order_acq_rel) | bit;
    }

    /**
     * Drops a cell from STAGED, PRODUCED, and COMMITTED at once (legacy serial STOP path).
     *
     * @param[in] cell_id Cell to deactivate (ignored if >= kMaxCells).
     * @return The committed mask after the removal.
     */
    [[nodiscard]] std::uint64_t deactivateImmediate(std::uint16_t cell_id) noexcept
    {
        const std::uint64_t bit = bitFor(cell_id);
        if(bit == 0ULL)
        {
            return committed();
        }
        staged_.fetch_and(~bit, std::memory_order_acq_rel);
        produced_.fetch_and(~bit, std::memory_order_acq_rel);
        return committed_.fetch_and(~bit, std::memory_order_acq_rel) & ~bit;
    }

private:
    /** @return Single-bit mask for @p cell_id, or 0 if out of range. */
    [[nodiscard]] static std::uint64_t bitFor(std::uint16_t cell_id) noexcept
    {
        return (cell_id < kMaxCells) ? (std::uint64_t{1} << cell_id) : 0ULL;
    }

    std::atomic<std::uint64_t> staged_{};     //!< Cells L2 intends to run.
    std::atomic<std::uint64_t> produced_{};   //!< Staged cells that have sent >= 1 slot message.
    std::atomic<std::uint64_t> committed_{};  //!< Cells the EOM expectation waits on (= staged & produced at last boundary).
};

} // namespace nv

#endif // ACTIVE_CELL_MASK_HPP

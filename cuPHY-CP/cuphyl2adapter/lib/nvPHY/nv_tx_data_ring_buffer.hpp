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

/**
 * @file nv_tx_data_ring_buffer.hpp
 * @brief TxDataSlotState and TxDataRingBuffer — cohesive ring-buffer for
 *        split-phase TX_DATA H2D staging state (Phase 1 Option A).
 *
 * Replaces five scattered per-ring arrays that were members of PHY_module:
 *   - tx_data_deferred_[]
 *   - expected_tx_data_count_[]
 *   - TxDataBatchState tx_batch_state_[]
 *   - txdata_staged_gpu_ptr_[][]
 *   - txdata_staged_msg_buf_[][]
 *
 * Depends on compile-time constants:
 *   - MAX_CELLS_PER_SLOT  — maximum cell index (from cuphy.h or CMake definition)
 *   - SLOT_STORAGE_DEPTH  — ring depth (from nv_slot_task_pool.hpp or CMake definition)
 */

#ifndef NV_TX_DATA_RING_BUFFER_HPP_
#define NV_TX_DATA_RING_BUFFER_HPP_

#include <array>
#include <atomic>
#include <cstdint>

namespace nv
{

/// Cache-line size for false-sharing prevention.
/// Defined as a compile-time constant to avoid GCC -Werror=interference-size,
/// which rejects std::hardware_destructive_interference_size because its value
/// can vary with -mtune/-mcpu.  Both ARM Grace and x86-64 use 64-byte lines.
inline constexpr std::size_t k_cpu_cache_line_size = 64;

/**
 * Per-slot staging state for split-phase TX_DATA H2D copy.
 *
 * Aligned to k_cpu_cache_line_size so that each ring slot starts on a
 * separate cache line, preventing false sharing between the message thread
 * (which writes via stage()) and worker threads (which read via gpu_ptrs()
 * / msg_bufs() in the adjacent slot).
 */
struct alignas(k_cpu_cache_line_size) TxDataSlotState
{
    uint8_t                                     cell_count{};     //!< Cells staged via stage(); reset by reset().
    uint8_t                                     expected_cells{}; //!< Set by arm() on success; 0 after reset().
    std::array<uint8_t*, MAX_CELLS_PER_SLOT>    gpu_ptr{};        //!< GPU TB pointers indexed by cell_id.
    std::array<const void*, MAX_CELLS_PER_SLOT> msg_buf{};        //!< TX_DATA FAPI msg_buf ptrs indexed by cell_id.

    /// Deferred-release ownership for this ring slot's TX_DATA lane. Written once
    /// per slot at arm time (set_deferred) on the message thread; read by both the
    /// message thread (inline release) and the H2D-completion callback thread.
    ///   true  — release owned by the tx_data_release_fn callback; inline skips.
    ///   false — released inline by the message thread; callback does not fire.
    /// Atomic (release store / acquire load) so the callback thread observes the
    /// arming write. Deliberately NOT touched by reset() (which is staging-only):
    /// the callback's staging reset must not flip ownership and let the message
    /// thread re-release the same NVIPC buffer. Each new slot re-arms it via
    /// set_deferred(); the abandon paths clear it via reset_txdata_h2d_state_for_ring.
    std::atomic<bool> deferred{false};

    /**
     * Reset staging state to zero (cell_count, expected_cells, all pointers).
     * Deliberately does NOT touch @c deferred — that is the cross-thread release
     * ownership token, cleared only by set_deferred() at (re-)arm or on the
     * enqueue-failure / collision abandon paths.
     */
    void reset() noexcept
    {
        cell_count     = 0;
        expected_cells = 0;
        gpu_ptr.fill(nullptr);
        msg_buf.fill(nullptr);
    }

    /**
     * True when the slot has been successfully armed (expected_cells > 0).
     *
     * @return true when armed; false after construction, reset(), or failed arm().
     *         Return value must be checked.
     */
    [[nodiscard]] bool valid() const noexcept { return expected_cells > 0; }
};

/**
 * Ring buffer of TxDataSlotState entries for SLOT_STORAGE_DEPTH concurrent slots.
 *
 * Lifecycle per slot:
 *   1. stage() — called once per TX_DATA.req on the message thread.
 *   2. arm()   — called at EOM; validates staged count == expected PDSCH cells.
 *   3. gpu_ptrs() / msg_bufs() — read by the PDSCH worker via DlAggrTaskArg snapshot.
 *   4. reset() — called after slot release (reset_txdata_h2d_state_for_ring).
 */
class TxDataRingBuffer {
public:
    /**
     * Stage one cell's GPU pointer and FAPI msg_buf for the given ring slot.
     *
     * @param[in] ring_idx  Ring slot index in [0, SLOT_STORAGE_DEPTH).
     * @param[in] cell_id   Physical cell ID; must be < MAX_CELLS_PER_SLOT.
     * @param[in] gpu       GPU TB pointer returned by l1_stage_tb_h2d().
     * @param[in] fapi_msg  TX_DATA.req msg_buf pointer from the NVIPC descriptor.
     */
    void stage(uint32_t ring_idx, uint32_t cell_id, uint8_t* gpu, const void* fapi_msg) noexcept
    {
        // Bounds guard — out-of-range indices would silently corrupt adjacent
        // memory in the fixed-size ring_ / gpu_ptr / msg_buf arrays.
        if (ring_idx >= SLOT_STORAGE_DEPTH || cell_id >= MAX_CELLS_PER_SLOT) [[unlikely]] { return; }
        auto& s = ring_[ring_idx];

        // Idempotent on duplicate cell_id: only bump cell_count on the
        // empty→non-empty transition so the count tracks DISTINCT cells.
        // A duplicate TX_DATA.req for the same cell_id silently overwrites
        // the prior gpu_ptr/msg_buf (last-write-wins) without inflating
        // cell_count — keeps arm()'s `cell_count == expected_pdsch` check
        // honest even if upstream re-stages.
        const bool first_stage_for_cell = (s.gpu_ptr[cell_id] == nullptr);
        s.gpu_ptr[cell_id] = gpu;
        s.msg_buf[cell_id] = fapi_msg;
        if (first_stage_for_cell)
        {
            ++s.cell_count;
        }
    }

    /**
     * Validate staged count and arm the slot for downstream readers.
     *
     * Returns false and resets the slot when @p expected_pdsch == 0 or when
     * the staged count does not match @p expected_pdsch.  Callers must save
     * staged_count() BEFORE calling arm() because reset() zeroes cell_count.
     *
     * @param[in] ring_idx       Ring slot index.
     * @param[in] expected_pdsch Expected PDSCH cell count (from DL_TTI scan).
     * @return true on success; false when validation failed (slot reset internally).
     *         Return value must be checked.
     */
    [[nodiscard]] bool arm(uint32_t ring_idx, uint8_t expected_pdsch) noexcept
    {
        auto& s = ring_[ring_idx];
        if(expected_pdsch == 0 || s.cell_count != expected_pdsch) [[unlikely]]
        {
            s.reset();
            return false;
        }
        s.expected_cells = expected_pdsch;
        return true;
    }

    /**
     * GPU TB pointer array for the slot; nullptr when not armed.
     *
     * @param[in] ring_idx  Ring slot index.
     * @return Pointer to the gpu_ptr[] array when is_valid(), nullptr otherwise.
     *         Return value must be checked.
     */
    [[nodiscard]] uint8_t* const* gpu_ptrs(uint32_t ring_idx) const noexcept
    {
        const auto& s = ring_[ring_idx];
        return s.valid() ? s.gpu_ptr.data() : nullptr;
    }

    /**
     * TX_DATA msg_buf pointer array for the slot; nullptr when not armed.
     *
     * @param[in] ring_idx  Ring slot index.
     * @return Pointer to the msg_buf[] array when is_valid(), nullptr otherwise.
     *         Return value must be checked.
     */
    [[nodiscard]] const void* const* msg_bufs(uint32_t ring_idx) const noexcept
    {
        const auto& s = ring_[ring_idx];
        return s.valid() ? s.msg_buf.data() : nullptr;
    }

    /**
     * Number of cells staged for the slot (0 after construction or reset()).
     *
     * @param[in] ring_idx  Ring slot index.
     * @return Cell count; 0 when not staged or after reset.
     *         Return value must be checked.
     */
    [[nodiscard]] uint8_t staged_count(uint32_t ring_idx) const noexcept
    {
        return ring_[ring_idx].cell_count;
    }

    /**
     * True after a successful arm(); false after construction, reset(), or failed arm().
     *
     * @param[in] ring_idx  Ring slot index.
     * @return true when the slot is armed and gpu_ptrs()/msg_bufs() are valid.
     *         Return value must be checked.
     */
    [[nodiscard]] bool is_valid(uint32_t ring_idx) const noexcept
    {
        return ring_[ring_idx].valid();
    }

    /**
     * True when TX_DATA release is deferred to the cuphydriver callback.
     *
     * @param[in] ring_idx  Ring slot index.
     * @return true when deferred; false otherwise.
     *         Return value must be checked.
     */
    [[nodiscard]] bool is_deferred(uint32_t ring_idx) const noexcept
    {
        return ring_[ring_idx].deferred.load(std::memory_order_acquire);
    }

    /**
     * Set or clear the deferred-release ownership for the given slot.
     *
     * Written once per slot at (re-)arm time on the message thread, before either
     * release path can run. The release-store pairs with is_deferred()'s
     * acquire-load so the H2D-completion callback thread observes the value.
     *
     * @param[in] ring_idx  Ring slot index.
     * @param[in] v         true => callback releases; false => inline release.
     */
    void set_deferred(uint32_t ring_idx, bool v) noexcept
    {
        ring_[ring_idx].deferred.store(v, std::memory_order_release);
    }

    /**
     * Reset all fields in the given ring slot (valid, staged, deferred, pointers).
     *
     * @param[in] ring_idx  Ring slot index.
     */
    void reset(uint32_t ring_idx) noexcept
    {
        ring_[ring_idx].reset();
    }

private:
    std::array<TxDataSlotState, SLOT_STORAGE_DEPTH> ring_{};
};

} // namespace nv

#endif // NV_TX_DATA_RING_BUFFER_HPP_

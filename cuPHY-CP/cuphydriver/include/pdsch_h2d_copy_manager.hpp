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

#ifndef PDSCH_H2D_COPY_MANAGER_HPP_
#define PDSCH_H2D_COPY_MANAGER_HPP_

#include <array>
#include <atomic>
#include <thread>
#include "constant.hpp"
#include "cuphydriver_api.hpp"
#include "locks.hpp"
#include "cuphy.hpp"

// CUPHYDRIVER_PDSCH_USE_BATCHED_COPY — compile-time master switch for the
// batched-memcpy code path inside PdschH2DCopyManager. Forms the AND of:
//
//   (CUPHYDRIVER_PDSCH_USE_BATCHED_COPY == 1) && (useBatchedMemcpy from YAML)
//
// Setting this macro to 0 forces the legacy per-TB cudaMemcpyAsync path
// regardless of the runtime YAML flag. Default is 1 (enabled). Override
// by passing `-DCUPHYDRIVER_PDSCH_USE_BATCHED_COPY=0` to the CMake configure
// step; no dedicated CMake option exposes this today.
#ifndef CUPHYDRIVER_PDSCH_USE_BATCHED_COPY
#define CUPHYDRIVER_PDSCH_USE_BATCHED_COPY 1
#endif

class MpsCtx;

/**
 * Tag type for the Null-Object constructor of PdschH2DCopyManager.
 *
 * Pass null_object to construct an inert instance that does no CUDA work —
 * used by the minimal/test PhyDriverCtx so callers never observe a null
 * manager pointer.
 */
struct null_object_t {};
inline constexpr null_object_t null_object{};

/**
 * Manages all PDSCH transport-block host-to-device DMA state.
 *
 * Owns the H2D CUDA stream, start/complete events, the batched-memcpy helper,
 * the prepone ring buffer, and the optional dedicated copy thread.  Extracted
 * from PhyDriverCtx so that H2D DMA mechanics are self-contained.
 */
class PdschH2DCopyManager {
public:
    /**
     * Construct manager — creates the CUDA stream and events.
     *
     * The MPS context pointer is borrowed (not owned); the caller must ensure
     * it outlives this object.
     *
     * @param[in] mpsCtx           PDSCH MPS/green context (non-owning).
     * @param[in] useBatchedMemcpy YAML flag for batched memcpy (0 or 1).
     * @param[in] copyWaitThNs     Timeout threshold for H2D event wait (ns).
     * @param[in] threadEnable     Whether to spawn a dedicated copy thread.
     */
    PdschH2DCopyManager(MpsCtx*  mpsCtx,
                        bool     useBatchedMemcpy,
                        uint32_t copyWaitThNs,
                        bool     threadEnable);

    /**
     * Null-Object constructor — does no CUDA work.
     *
     * Every member is left in its default-initialized, benign state: stream
     * and events are nullptr, the batched-memcpy helper is empty, flags are
     * zero. All accessors therefore return safe defaults and all mutating
     * methods become no-ops. Used by the minimal/test PhyDriverCtx so callers
     * never have to check `getH2DCopyManager() == nullptr`.
     */
    explicit PdschH2DCopyManager(null_object_t) noexcept
        : m_mpsCtx{nullptr}
        , m_useBatchedMemcpy{false}
        , m_batchedMemcpyHelper{0,
                                batchedMemcpySrcHint::srcIsHost,
                                batchedMemcpyDstHint::dstIsDevice,
                                false}
        , m_threadEnable{false}
        , m_copyWaitThNs{0}
    {
    }

    ~PdschH2DCopyManager();

    // Rule-of-five: this class owns CUDA stream/event handles and a jthread.
    // None of those are safely copyable or movable, so all four operations
    // are deleted explicitly (silences clang-tidy cppcoreguidelines-special-
    // member-functions and matches CodeRabbit / Greptile suggestions).
    PdschH2DCopyManager(const PdschH2DCopyManager&)            = delete;
    PdschH2DCopyManager& operator=(const PdschH2DCopyManager&) = delete;
    PdschH2DCopyManager(PdschH2DCopyManager&&)                 = delete;
    PdschH2DCopyManager& operator=(PdschH2DCopyManager&&)      = delete;

    // ── CUDA context ─────────────────────────────────────────────
    void         setCtx();

    // ── Stream / events ──────────────────────────────────────────
    [[nodiscard]] cudaStream_t getStream() const { return m_stream; }
    [[nodiscard]] cudaEvent_t  getStartEvent(uint8_t slot) { return m_startEvents[slot % MAX_PDSCH_TB_CPY_CUDA_EVENTS]; }
    [[nodiscard]] cudaEvent_t  getCompleteEvent(uint8_t slot) { return m_completeEvents[slot % MAX_PDSCH_TB_CPY_CUDA_EVENTS]; }

    // ── Batched memcpy ───────────────────────────────────────────
    void              updateBatchedMemcpyInfo(void* dst, const void* src, std::size_t count);
    [[nodiscard]] cuphyStatus_t performBatchedMemcpy();
    void              resetBatchedMemcpyBatches();
    [[nodiscard]] bool getUseBatchedMemcpy() const { return m_useBatchedMemcpy; }

    // ── Split-phase H2D guard ───────────────────────────────────
    // Synchronization: l1_launch_tb_h2d() stores true with release after
    // recording the start/complete CUDA events; l1_enqueue_phy_work() loads
    // with acquire in isBatchLaunchedForSlot() before deciding to skip
    // double-submit. clearBatchLaunched() stores false with release before
    // the modulo event slot is reused by a later slot.
    void markBatchLaunched(uint8_t slot)
    {
        m_batchLaunched[slot % MAX_PDSCH_TB_CPY_CUDA_EVENTS].store(true, std::memory_order_release);
    }
    [[nodiscard]] bool isBatchLaunchedForSlot(uint8_t slot) const
    {
        return m_batchLaunched[slot % MAX_PDSCH_TB_CPY_CUDA_EVENTS].load(std::memory_order_acquire);
    }
    void clearBatchLaunched(uint8_t slot)
    {
        m_batchLaunched[slot % MAX_PDSCH_TB_CPY_CUDA_EVENTS].store(false, std::memory_order_release);
    }

    // ── Prepone ring buffer ──────────────────────────────────────
    [[nodiscard]] h2d_copy_prepone_info_t* getPreponeInfo(uint16_t idx) { return &m_preponeRing[idx]; }
    void                                  resetPreponeInfo();

    // ── Flags / coordination ─────────────────────────────────────
    [[nodiscard]] bool     isPreponeEnabled() const { return m_preponeEnabled; }
    void                   setPreponeEnabled(bool v) { m_preponeEnabled = v; }

    [[nodiscard]] bool     isThreadEnabled() const { return m_threadEnable; }
    [[nodiscard]] uint32_t getCopyWaitThNs() const { return m_copyWaitThNs; }

    [[nodiscard]] uint8_t  getBuffCopyCount() const { return m_numBuffCopy; }
    void                   setBuffCopyCount(uint8_t v) { m_numBuffCopy = v; }
    [[nodiscard]] uint8_t  incBuffCopyCount() { return ++m_numBuffCopy; }

    std::atomic<uint16_t>& writeIdx() { return m_writeIdx; }
    uint16_t&              readIdx()  { return m_readIdx; }

    std::array<std::atomic<bool>, PDSCH_MAX_GPU_BUFFS>& cudaEventRecDone() { return m_cudaEventRecDone; }
    std::array<std::atomic<int>,  PDSCH_MAX_GPU_BUFFS>& doneCurSlotIdx()   { return m_doneCurSlotIdx; }
    uint8_t& doneCurSlotReadIdx()  { return m_doneCurSlotReadIdx; }
    uint8_t& doneCurSlotWriteIdx() { return m_doneCurSlotWriteIdx; }

    Mutex& preponeMutex() { return m_preponeMutex; }

    // ── Copy thread ──────────────────────────────────────────────
    std::jthread& copyThread() { return m_copyThread; }

    // ── Shutdown ─────────────────────────────────────────────────
    // Request cooperative cancellation via the jthread's stop_source and join.
    // Safe to call multiple times. The destructor invokes this before
    // destroying the CUDA stream/events so the worker exits cleanly without
    // touching destroyed handles (use-after-free).
    void stopThread();

private:
    MpsCtx*      m_mpsCtx;
    cudaStream_t m_stream{};

    std::array<cudaEvent_t, MAX_PDSCH_TB_CPY_CUDA_EVENTS> m_startEvents{};
    std::array<cudaEvent_t, MAX_PDSCH_TB_CPY_CUDA_EVENTS> m_completeEvents{};

    bool                     m_useBatchedMemcpy{false};
    cuphyBatchedMemcpyHelper m_batchedMemcpyHelper;

    std::array<h2d_copy_prepone_info_t, (DL_MAX_CELLS_PER_SLOT * PDSCH_MAX_GPU_BUFFS)> m_preponeRing{};

    bool     m_preponeEnabled{false};
    uint8_t  m_numBuffCopy{0};
    bool     m_threadEnable{false};
    uint32_t m_copyWaitThNs{0};

    std::atomic<uint16_t> m_writeIdx{0};
    uint16_t              m_readIdx{0};

    std::array<std::atomic<bool>, PDSCH_MAX_GPU_BUFFS> m_cudaEventRecDone{};
    std::array<std::atomic<int>,  PDSCH_MAX_GPU_BUFFS> m_doneCurSlotIdx{};
    uint8_t m_doneCurSlotReadIdx{0};
    uint8_t m_doneCurSlotWriteIdx{0};

    std::array<std::atomic<bool>, MAX_PDSCH_TB_CPY_CUDA_EVENTS> m_batchLaunched{};

    Mutex             m_preponeMutex{};
    std::jthread      m_copyThread{};
};

#endif // PDSCH_H2D_COPY_MANAGER_HPP_

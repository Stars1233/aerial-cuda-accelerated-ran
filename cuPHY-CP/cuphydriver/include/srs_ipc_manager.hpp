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

#ifndef SRS_IPC_MANAGER_HPP
#define SRS_IPC_MANAGER_HPP

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <thread>
#include <vector>

#include "nv_lockfree.hpp"

// `class GpuDevice` is forward-declared so this header can stay free of
// gpudevice.hpp. gpudevice.hpp defines a default `TAG` macro inside an
// `#ifndef TAG` self-guard, which collides with the per-TU `#define TAG`
// pattern used throughout cuMAC-CP and cuphydriver. Including it
// transitively from a frequently-pulled header reintroduces the `TAG`
// redefined errors fixed earlier in this branch.
class GpuDevice;

// `_CVSrsChestBuff` (alias `CVSrsChestBuff`) is forward-declared so the
// pool member `nv::lock_free_mem_pool<CVSrsChestBuff>*` can be expressed
// here without dragging cv_memory_bank_srs_chest.hpp in. The full
// definition is needed (for sizeof) at the point where the pool is
// `new`'d / `delete`'d in srs_ipc_manager.cpp; that TU includes the bank
// header explicitly.
struct _CVSrsChestBuff;
typedef struct _CVSrsChestBuff CVSrsChestBuff;

// TODO: Source from L1 config rather than hard-coding here.
#define MAX_NUM_UE_SRS_INFO_PER_SLOT (48U)

/**
 * @brief One SRS info-update record, allocated from `SrsIpcManager::info_pool()`.
 *
 * Populated by the cuphydriver SRS path (see `physrs_aggr.cpp::setup`)
 * with the active cell index, the per-cell info index, and the global
 * "real" buffer index returned by `CvSrsChestMemoryBank::preAllocateBuffer`.
 * Consumed by cuMAC across the IPC pool to find the matching channel
 * estimate buffer for each scheduled UE.
 */
typedef struct _SrsInfoUpdate
{
    uint32_t real_buff_idx; ///< Global SRS chest buffer index in `gpu_pool()`/`chest_pool()`
    uint16_t srs_info_idx;  ///< Per-cell info index within the slot (0..max_srs_info_per_cell-1)
    uint16_t cell_idx;      ///< Active runtime cell index
    uint16_t rnti;          ///< UE C-RNTI
    uint16_t _reserved;     ///< Padding; must be zero
} SrsInfoUpdate;

/**
 * @brief IPC-backed device (GPU) memory buffer.
 *
 * Wraps a single slot from an `nv::lock_free_mem_pool<uint8_t>` GPU pool
 * (opened with `cuda_device_id`). Provides the same interface as
 * `dev_buf` (addr(), size(), size_alloc, clear()) so it can be used as a
 * drop-in replacement, e.g. as the backing buffer of `CVSrsChestBuff`.
 *
 * Method bodies live in `srs_ipc_manager.cpp` to avoid pulling cuda.h /
 * cuda_driver_utils.hpp / gpudevice.hpp into every consumer of this
 * header.
 */
class ipc_dev_buf
{
public:
    /**
     * @brief Allocate one buffer from the IPC memory pool.
     *
     * @param pool   Open `nv::lock_free_mem_pool<uint8_t>` (GPU pool created with cuda_device_id)
     * @param _gDev  Optional GPU device pointer (used for `setDevice()` before `clear()`)
     */
    ipc_dev_buf(nv::lock_free_mem_pool<uint8_t>* pool, GpuDevice* _gDev = nullptr);

    /// Releases the buffer back to the pool (no-op if `pool` was null).
    ~ipc_dev_buf();

    ipc_dev_buf(const ipc_dev_buf&)            = delete;
    ipc_dev_buf& operator=(const ipc_dev_buf&) = delete;

    /// Pointer to the allocated GPU buffer (nullptr when pool was null or alloc failed).
    uint8_t* addr();

    /// Allocated size in bytes.
    size_t size() const;

    /**
     * @brief Zero the buffer on the GPU (cuMemsetD8).
     *
     * Calls `setDevice()` on the optional GpuDevice first. No-op when
     * the buffer is null or zero-sized.
     */
    void clear();

    size_t size_alloc; ///< Allocated size in bytes (for memory-footprint tracking; matches dev_buf API)

private:
    nv::lock_free_mem_pool<uint8_t>* mempool;  ///< IPC mempool (GPU-backed, raw uint8_t bytes); not owned
    GpuDevice*                       gDev;     ///< Optional GPU device for setDevice() before clear()
    uint8_t*                         buf_addr; ///< Pointer to the allocated GPU buffer (nullptr when not owned)
};

/**
 * @brief Owns the three SRS IPC memory pools and the [DEBUG] HDF5 dump path.
 *
 * Single-purpose container assembled by `CvSrsChestMemoryBank` once at
 * construction. Concentrates everything that is "the IPC layer" of the
 * SRS path so the bank itself can stay focused on per-UE bookkeeping:
 *
 *   - `gpu_pool()`   — raw GPU-side bytes for channel-estimate buffers
 *                      (contiguous; sized `total_num_buffers * gpu_buf_size`,
 *                      opened on `gpu_device->getId()`).
 *   - `chest_pool()` — `CVSrsChestBuff` slots used by the bank with
 *                      placement-new (one slot per channel-estimate buffer).
 *   - `info_pool()`  — per-slot `SrsInfoUpdate` records produced by the
 *                      cuphydriver SRS path and consumed by cuMAC over IPC.
 *
 * Also hosts the `DUMP_SRS_SLOT_NUM=N` per-slot HDF5 dump (formerly
 * `SrsBufH5Dumper`):
 *   - Phase 1 (synchronous, RT thread): `cuMemcpyDtoH` D2H of the GPU pool
 *     into a pinned host staging buffer + plain `memcpy` of the two CPU
 *     pools into heap staging buffers. Bounded latency.
 *   - Phase 2 (detached, SCHED_OTHER thread): all HDF5 file I/O drives
 *     off the staging copies, so disk / page-cache stalls never preempt
 *     the L1 data path.
 *
 * `DUMP_SRS_SLOT_NUM` is parsed once at construction as a non-negative
 * integer (default 0 = disabled, no staging is allocated, `dump_h5()` is
 * a no-op). N>0 sizes the staging vector to N entries and caps the
 * number of dumps performed in this process to N.
 *
 * Lifetime: owned as a value member of `CvSrsChestMemoryBank`. The bank
 * destroys all placement-new'd `CVSrsChestBuff` objects in its own
 * destructor body before the manager is destroyed, so the pools are
 * still alive when the slot destructors run.
 */
class SrsIpcManager
{
public:
    /**
     * Pool sizing + H5 attribute metadata. All values are run-fixed
     * after construction. The macros `CV_NUM_PRBG`, `CV_NUM_UE_LAYER`
     * etc. live in `cv_memory_bank_srs_chest.hpp`; the bank fills in
     * this struct from there so the manager header doesn't need to
     * include the bank header.
     */
    struct Config
    {
        uint32_t total_num_buffers{0};     ///< Length of `gpu_pool()` and `chest_pool()` (capped by bank to MAX_SRS_CHEST_BUFFERS)
        uint32_t srs_info_pool_len{0};     ///< Length of `info_pool()` (MAX_NUM_UE_SRS_INFO_PER_SLOT * MAX_CELLS_PER_SLOT * SLOTS_PER_FRAME)
        uint32_t gpu_buf_size{0};          ///< Per-buffer GPU bytes (PRG * gnb_ant * ue_layer * sizeof(uint32_t))
        uint32_t num_prg{0};               ///< CV_NUM_PRBG, dumped as `num_prg` H5 attribute
        uint32_t num_ue_layer{0};          ///< CV_NUM_UE_LAYER, dumped as `num_ue_layer` H5 attribute
        uint16_t max_srs_antenna_ports{0}; ///< Per-buffer GNB antenna count, dumped as `num_gnb_ant` attribute
    };

    /**
     * @brief Open the three IPC pools and pre-allocate H5 staging if the
     *        debug dump is enabled.
     *
     * Pool open failures are logged at NVLOGF (fatal). Staging-allocation
     * failures leave the dumper "not ready" so subsequent `dump_h5()`
     * calls reject with -1; the IPC pools themselves stay open and the
     * L1 data path is unaffected.
     *
     * @param cfg         Pool sizing + H5-attribute metadata.
     * @param gpu_device  GPU device (used to derive `cuda_device_id` for
     *                    the GPU pool, and to call `setDevice()` before
     *                    each `dump_h5()` D2H copy). Must be non-null.
     */
    SrsIpcManager(const Config& cfg, GpuDevice* gpu_device);

    /// Closes the three pools and releases pinned/heap H5 staging buffers.
    ~SrsIpcManager();

    SrsIpcManager(const SrsIpcManager&)            = delete;
    SrsIpcManager& operator=(const SrsIpcManager&) = delete;

    //-- Pool accessors ------------------------------------------------------
    nv::lock_free_mem_pool<uint8_t>*        gpu_pool() const   { return ipc_gpu_pool_; }
    nv::lock_free_mem_pool<CVSrsChestBuff>* chest_pool() const { return chest_buf_pool_; }
    nv::lock_free_mem_pool<SrsInfoUpdate>*  info_pool() const  { return srs_info_pool_; }

    //-- Convenience wrappers ------------------------------------------------
    /**
     * @brief Get the address of the SRS-info record at `buf_id` inside `info_pool()`.
     *
     * Equivalent to `info_pool()->get_buf_addr(buf_id)`. Used by
     * `physrs_aggr.cpp::PhySrsAggr::setup` to populate one record per
     * scheduled SRS UE.
     */
    SrsInfoUpdate* get_srs_info_update_buf(int buf_id) const;

    //-- Debug HDF5 dump -----------------------------------------------------
    /// Configured per-process H5 dump cap (`DUMP_SRS_SLOT_NUM` env var,
    /// default 0). Returns 0 when the dump path is disabled, otherwise N
    /// (and the manager has pre-allocated N staging sets).
    uint32_t h5_dump_slot_max() const { return h5_dump_slot_max_; }

    /**
     * @brief Snapshot all three pools and asynchronously write one
     *        `cubb_srs_buffers_<N>_SFN_<sfn>.<slot>.h5` file.
     *
     * Phase 1 runs synchronously on the caller's thread (intended to be
     * RT-priority). Phase 2 spawns a detached SCHED_OTHER thread that
     * writes attributes + datasets and then exits. Returns 0 in both the
     * no-op (disabled / capped) and the "snapshot taken" paths; -1 only
     * on hard snapshot errors. Detached-thread failures are logged but
     * do not propagate.
     *
     * @param sfn               System Frame Number (embedded in filename + log)
     * @param slot              Slot number (embedded in filename + log)
     * @param active_num_cells  Active runtime cell count; stored as the
     *                          `cell_num` attribute and used to trim
     *                          `info_pool()` to
     *                          `active_num_cells * MAX_NUM_UE_SRS_INFO_PER_SLOT * info_buf_size`
     *                          bytes (clamped to the full pool).
     * @param outputDir         Output directory (defaults to "/tmp" when null/empty).
     * @return 0 on success / disabled / capped; -1 on snapshot failure.
     */
    int dump_h5(uint16_t sfn, uint16_t slot, uint32_t active_num_cells, const char* outputDir = "/tmp");

private:
    /// Pinned + plain host staging for one pending H5 dump.
    struct H5Staging
    {
        uint8_t* gpu_pool_host{nullptr};   ///< cuMemAllocHost'd pinned buffer for GPU D2H copy
        uint8_t* chest_pool_host{nullptr}; ///< heap buffer for chest_buf_pool snapshot
        uint8_t* info_pool_host{nullptr};  ///< heap buffer for srs_info_pool snapshot
    };

    //-- Pools ---------------------------------------------------------------
    nv::lock_free_mem_pool<uint8_t>*        ipc_gpu_pool_{nullptr};
    nv::lock_free_mem_pool<CVSrsChestBuff>* chest_buf_pool_{nullptr};
    nv::lock_free_mem_pool<SrsInfoUpdate>*  srs_info_pool_{nullptr};
    GpuDevice*                              gpu_device_{nullptr};

    //-- Run-fixed config (mirrors Config; cached for dump_h5 attributes) ----
    Config                                  cfg_{};

    //-- H5 dump state -------------------------------------------------------
    /// Per-process dump cap, parsed from `DUMP_SRS_SLOT_NUM` at
    /// construction. 0 means the dump path is disabled (no staging
    /// allocated, `dump_h5()` is a no-op). Equal to `h5_staging_.size()`
    /// once `allocate_h5_staging()` has run successfully.
    uint32_t                                h5_dump_slot_max_{0};
    bool                                    h5_staging_ready_{false};
    /// Number of dumps already reserved (0..h5_dump_slot_max_). Atomic
    /// because `dump_h5()` is called from concurrent UlPhyDriver threads
    /// (one per cell group); plain `++` would race and could hand the
    /// same staging-slot index to two callers.
    std::atomic<uint32_t>                   h5_dump_slot_count_{0};
    std::vector<H5Staging>                  h5_staging_{};
    size_t                                  gpu_pool_total_bytes_{0};
    size_t                                  chest_pool_total_bytes_{0};
    size_t                                  info_pool_total_bytes_{0};

    /// Background H5 writer threads spawned by `dump_h5()`. We now track
    /// each instead of `detach()`-ing it so the destructor can join them
    /// before releasing the staging buffers they read from -- otherwise
    /// a quick teardown after the last dump can race with an in-flight
    /// HDF5 write. Bounded by `h5_dump_slot_max_` per process so the
    /// vector stays small. Guarded by `h5_threads_mtx_` against
    /// concurrent emplaces from multiple UlPhyDriver threads; the
    /// destructor reads it single-threaded.
    std::vector<std::thread>                h5_threads_{};
    std::mutex                              h5_threads_mtx_{};

    /// Allocate `h5_dump_slot_max_` staging sets. Called from the
    /// constructor when `h5_dump_slot_max_ > 0`. Sets `h5_staging_ready_`
    /// to false on any allocation failure; partial allocations are
    /// reclaimed in the destructor.
    void allocate_h5_staging();

    // Singleton guard: only one `SrsIpcManager` may exist at a time. The
    // primary (`LOCK_FREE_OPT_SHM_PRIMARY`) IPC pools we open are named
    // shared-memory segments which can only be created once per process,
    // so a second manager would race on pool open. Incremented in the
    // constructor (NVLOGF on collision) and decremented in the destructor.
    static std::atomic<int> instance_count_;
};

#endif // SRS_IPC_MANAGER_HPP

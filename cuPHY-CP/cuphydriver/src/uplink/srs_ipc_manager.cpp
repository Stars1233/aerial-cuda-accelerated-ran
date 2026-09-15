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

#define TAG (NVLOG_TAG_BASE_CUPHY_DRIVER + 32) // "DRV.CV_MEM_BNK" (shared with bank)

#include "srs_ipc_manager.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <pthread.h>
#include <sched.h>
#include <string>
#include <sys/stat.h>
#include <thread>

#include <cuda.h>
#include <hdf5.h>

// `cv_memory_bank_srs_chest.hpp` is included AFTER our own `#define TAG`
// above so gpudevice.hpp's default-`TAG` self-guard (`#ifndef TAG`) is a
// no-op for this TU. We need the full `CVSrsChestBuff` definition for
// `new lock_free_mem_pool<CVSrsChestBuff>(...)` (sizeof default arg).
#include "app_config.hpp"
#include "cv_memory_bank_srs_chest.hpp"
#include "gpudevice.hpp"
#include "nvlog.hpp"

//============================================================================
// ipc_dev_buf
//============================================================================
ipc_dev_buf::ipc_dev_buf(nv::lock_free_mem_pool<uint8_t>* pool, GpuDevice* _gDev)
    : size_alloc(0)
    , mempool(pool)
    , gDev(_gDev)
    , buf_addr(nullptr)
{
    if (mempool)
    {
        buf_addr = mempool->alloc();
        if (buf_addr != nullptr)
        {
            size_alloc = mempool->get_buf_size();
        }
    }
}

ipc_dev_buf::~ipc_dev_buf()
{
    if (mempool && buf_addr)
        mempool->free(buf_addr);
}

uint8_t* ipc_dev_buf::addr()
{
    return buf_addr;
}

size_t ipc_dev_buf::size() const
{
    return static_cast<size_t>(size_alloc);
}

void ipc_dev_buf::clear()
{
    if (buf_addr && size_alloc > 0)
    {
        if (gDev)
            gDev->setDevice();
        CUDA_DRIVER_CHECK(cuMemsetD8(reinterpret_cast<CUdeviceptr>(buf_addr), 0, size_alloc));
    }
}

//============================================================================
// HDF5 helpers (anonymous-namespace, used only by SrsIpcManager::dump_h5)
//============================================================================
namespace
{
//! Create a 1-D `H5T_NATIVE_UINT8` dataset of `num_bytes` bytes and write
//! `host_ptr` into it. Empty / null inputs are logged and treated as success
//! (caller may have a legitimately empty pool).
//! @return 0 on success or skip; -1 on HDF5 failure.
int writeBytesDataset(hid_t file, const char* ds_name, const void* host_ptr, size_t num_bytes)
{
    if (num_bytes == 0 || host_ptr == nullptr)
    {
        NVLOGI_FMT(TAG, "SrsIpcManager: skipping dataset '{}' (bytes={}, ptr=0x{:x})",
                   ds_name, num_bytes, reinterpret_cast<uintptr_t>(host_ptr));
        return 0;
    }

    hsize_t dim   = static_cast<hsize_t>(num_bytes);
    hid_t   space = H5Screate_simple(1, &dim, nullptr);
    if (space < 0)
    {
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "SrsIpcManager: H5Screate_simple failed for '{}'", ds_name);
        return -1;
    }
    hid_t dset = H5Dcreate2(file, ds_name, H5T_NATIVE_UINT8, space,
                            H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    if (dset < 0)
    {
        H5Sclose(space);
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "SrsIpcManager: H5Dcreate2 failed for '{}'", ds_name);
        return -1;
    }
    herr_t wr = H5Dwrite(dset, H5T_NATIVE_UINT8, H5S_ALL, H5S_ALL, H5P_DEFAULT, host_ptr);
    H5Dclose(dset);
    H5Sclose(space);
    if (wr < 0)
    {
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "SrsIpcManager: H5Dwrite failed for '{}'", ds_name);
        return -1;
    }
    return 0;
}

//! Bundle of parameters captured from the manager state and the snapshot
//! phase, forwarded into the detached H5-writer thread by value so the
//! detached worker never reads back into the manager.
struct H5WriteJob
{
    std::string fname;
    uint16_t    sfn;
    uint16_t    slot;
    uint32_t    dump_slot;       ///< 1-based dump index
    uint32_t    max_dump_slots;  ///< Snapshot of SrsIpcManager::h5_dump_slot_max_

    // Attribute payload
    uint32_t    gpu_pool_len;
    uint32_t    gpu_buf_size;
    uint32_t    num_prg;
    uint32_t    num_gnb_ant;
    uint32_t    num_ue_layer;
    uint32_t    cell_num;

    // Dataset payload pointers (live in manager's staging buffers, not owned)
    const uint8_t* gpu_h5;
    size_t         gpu_bytes;
    const uint8_t* chest_h5;
    size_t         chest_bytes;
    const uint8_t* info_h5;
    size_t         info_bytes;
};

//! Body of the detached H5-writer thread. Drops scheduling priority to
//! `SCHED_OTHER` so disk I/O cannot preempt the RT data path, then writes
//! attributes + datasets and logs timings. All errors are logged inside;
//! return value is ignored by the parent (`std::thread::detach()`).
void runH5WriteJob(H5WriteJob job)
{
    int         before_policy = -1;
    int         after_policy  = -1;
    sched_param before_sp{};
    sched_param after_sp{};
    pthread_t   self  = pthread_self();
    int         gp_rc = pthread_getschedparam(self, &before_policy, &before_sp);

    sched_param normal_sp{};
    normal_sp.sched_priority = 0; // SCHED_OTHER ignores prio; must be 0
    int sp_rc = pthread_setschedparam(self, SCHED_OTHER, &normal_sp);
    pthread_getschedparam(self, &after_policy, &after_sp);

    // Bind to the low-priority core to keep H5 disk I/O off real-time cores.
    const int lp_core = static_cast<int>(AppConfig::getInstance().getLowPriorityCore());
    nv_assign_thread_cpu_core(lp_core);
    const int cpu_actual = sched_getcpu();

    NVLOGI_FMT(TAG,
               "SrsIpcManager H5 thread: SFN {}.{} dump_slot={}/{} "
               "before(policy={}, prio={}, getrc={}) after(policy={}, prio={}, setrc={}) "
               "[SCHED_FIFO={}, SCHED_RR={}, SCHED_OTHER={}] "
               "cpu_bind_req={} cpu_actual={}",
               job.sfn, job.slot, job.dump_slot, job.max_dump_slots,
               before_policy, before_sp.sched_priority, gp_rc,
               after_policy,  after_sp.sched_priority,  sp_rc,
               SCHED_FIFO, SCHED_RR, SCHED_OTHER,
               lp_core, cpu_actual);

    // HDF5 is not thread-safe by default. Concurrent H5* calls from
    // multiple dump threads corrupt global HDF5 state, producing 96-byte
    // stub files or datasets with missing sections. Serialize all H5 I/O.
    static std::mutex s_hdf5_serialize;
    std::lock_guard<std::mutex> hdf5_lk(s_hdf5_serialize);

    const auto t_h5_start = std::chrono::steady_clock::now();

    int   ret  = 0;
    hid_t file = H5Fcreate(job.fname.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    if (file < 0)
    {
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT,
                   "SrsIpcManager: H5Fcreate failed for {} (SFN {}.{})",
                   job.fname, job.sfn, job.slot);
        return;
    }

    auto writeU32attr = [&](const char* name, uint32_t val) {
        hid_t sp   = H5Screate(H5S_SCALAR);
        hid_t attr = H5Acreate2(file, name, H5T_NATIVE_UINT32, sp, H5P_DEFAULT, H5P_DEFAULT);
        H5Awrite(attr, H5T_NATIVE_UINT32, &val);
        H5Aclose(attr);
        H5Sclose(sp);
    };
    writeU32attr("gpu_pool_len", job.gpu_pool_len);
    writeU32attr("gpu_buf_size", job.gpu_buf_size);
    writeU32attr("num_prg",      job.num_prg);
    writeU32attr("num_gnb_ant",  job.num_gnb_ant);
    writeU32attr("num_ue_layer", job.num_ue_layer);
    // Active runtime cell count (not the compile-time MAX_CELLS_PER_SLOT
    // used to size srs_info_pool). Consumers (cuMAC TV-replay) check this
    // against their NUM_CELL config to refuse mismatched replays.
    writeU32attr("cell_num", job.cell_num);

    if (writeBytesDataset(file, "ipc_gpu_pool",   job.gpu_h5,   job.gpu_bytes)   != 0) ret = -1;
    if (writeBytesDataset(file, "chest_buf_pool", job.chest_h5, job.chest_bytes) != 0) ret = -1;
    if (writeBytesDataset(file, "srs_info_pool",  job.info_h5,  job.info_bytes)  != 0) ret = -1;

    H5Fclose(file);

    const auto t_h5_end = std::chrono::steady_clock::now();
    const long h5_us    = std::chrono::duration_cast<std::chrono::microseconds>(t_h5_end - t_h5_start).count();
    const long h5_ms    = std::chrono::duration_cast<std::chrono::milliseconds>(t_h5_end - t_h5_start).count();

    // Report actual on-disk file size so logs confirm a complete write.
    long long file_size_bytes = -1;
    struct stat st{};
    if (::stat(job.fname.c_str(), &st) == 0)
    {
        file_size_bytes = static_cast<long long>(st.st_size);
    }

    NVLOGI_FMT(TAG,
               "SrsIpcManager H5 thread done: SFN {}.{} dump_slot={}/{} file={} "
               "size_bytes={} h5_us={} h5_ms={} ret={}",
               job.sfn, job.slot, job.dump_slot, job.max_dump_slots,
               job.fname, file_size_bytes, h5_us, h5_ms, ret);
}
} // anonymous namespace

//============================================================================
// SrsIpcManager
//============================================================================
std::atomic<int> SrsIpcManager::instance_count_{0};

SrsIpcManager::SrsIpcManager(const Config& cfg, GpuDevice* gpu_device)
    : gpu_device_(gpu_device)
    , cfg_(cfg)
{
    // Singleton guard: the primary SHM pools we open below can only be
    // created once per process; a second manager would race on pool open.
    // NVLOGF is fatal so this aborts immediately on collision.
    int prev = instance_count_.fetch_add(1, std::memory_order_relaxed);
    if (prev != 0)
    {
        NVLOGF_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "SrsIpcManager: only one instance is allowed, but instance #{} was attempted", prev + 1);
    }

    const int cuda_device_id = (gpu_device_ != nullptr) ? static_cast<int>(gpu_device_->getId()) : -1;

    // Open the three IPC pools. Names mirror the originals so the consumer
    // (cuMAC simple_srs_memory_bank, etc.) keeps finding them.
    char pool_name[32];

    // 1. GPU-backed raw byte pool: one slot per channel-estimate buffer.
    //    `buf_size` is supplied explicitly (default `sizeof(uint8_t)` is wrong here).
    std::snprintf(pool_name, sizeof(pool_name), "CvSrsChest_GPU%d", cuda_device_id);
    ipc_gpu_pool_ = new nv::lock_free_mem_pool<uint8_t>(cfg_.total_num_buffers,
                                                       LOCK_FREE_OPT_SHM_PRIMARY,
                                                       pool_name,
                                                       cuda_device_id,
                                                       cfg_.gpu_buf_size);
    if (ipc_gpu_pool_ == nullptr || ipc_gpu_pool_->get_pool_len() <= 0)
    {
        NVLOGF_FMT(TAG, AERIAL_NVIPC_API_EVENT,
                   "SrsIpcManager: lock_free_mem_pool<uint8_t> open failed name=%s buf_size=%u pool_len=%u cuda_device_id=%d",
                   pool_name, cfg_.gpu_buf_size, cfg_.total_num_buffers, cuda_device_id);
    }

    // 2. CPU pool of CVSrsChestBuff metadata objects. Used by the bank
    //    with placement new -- one slot per channel-estimate buffer.
    std::snprintf(pool_name, sizeof(pool_name), "SrsChest");
    chest_buf_pool_ = new nv::lock_free_mem_pool<CVSrsChestBuff>(cfg_.total_num_buffers,
                                                                 LOCK_FREE_OPT_SHM_PRIMARY,
                                                                 pool_name);

    // 3. CPU pool of SrsInfoUpdate records. Sized by the caller
    //    (typically MAX_NUM_UE_SRS_INFO_PER_SLOT * MAX_CELLS_PER_SLOT).
    std::snprintf(pool_name, sizeof(pool_name), "SrsInfo");
    srs_info_pool_ = new nv::lock_free_mem_pool<SrsInfoUpdate>(cfg_.srs_info_pool_len,
                                                               LOCK_FREE_OPT_SHM_PRIMARY,
                                                               pool_name);

    // Parse DUMP_SRS_SLOT_NUM exactly once. The integer value sets the
    // per-process dump cap (0 = disabled = staging skipped, dump_h5 is a
    // cheap early-return). Non-numeric / negative input falls back to 0.
    {
        const char* v     = std::getenv("DUMP_SRS_SLOT_NUM");
        h5_dump_slot_max_ = 0;
        if (v != nullptr && v[0] != '\0')
        {
            char*             end    = nullptr;
            const long parsed_signed = std::strtol(v, &end, 10);
            if (end != v && parsed_signed > 0)
            {
                h5_dump_slot_max_ = static_cast<uint32_t>(parsed_signed);
            }
        }
        NVLOGI_FMT(TAG, "SrsIpcManager: DUMP_SRS_SLOT_NUM={} (env='{}')",
                   h5_dump_slot_max_, (v != nullptr) ? v : "<unset>");
    }

    if (h5_dump_slot_max_ > 0)
    {
        allocate_h5_staging();
    }

    NVLOGC_FMT(TAG,
               "SrsIpcManager: pools opened (total_num_buffers={}, gpu_buf_size={} B, srs_info_pool_len={}, "
               "num_prg={}, num_gnb_ant={}, num_ue_layer={}, h5_dump_slot_max={}, h5_ready={})",
               cfg_.total_num_buffers, cfg_.gpu_buf_size, cfg_.srs_info_pool_len,
               cfg_.num_prg, cfg_.max_srs_antenna_ports, cfg_.num_ue_layer,
               h5_dump_slot_max_, h5_staging_ready_ ? 1 : 0);
}

SrsIpcManager::~SrsIpcManager()
{
    instance_count_.fetch_sub(1, std::memory_order_relaxed);

    // Wait for every spawned H5 writer thread to finish *before* freeing
    // the staging buffers it borrows from `h5_staging_`. We don't take
    // the mutex here -- the destructor runs single-threaded after all
    // dump_h5() callers have returned (no new emplaces can occur). The
    // vector is empty when DUMP_SRS_SLOT_NUM=0 so this loop is a no-op
    // for the common production path.
    for (std::thread& t : h5_threads_)
    {
        if (t.joinable())
        {
            t.join();
        }
    }
    h5_threads_.clear();

    // Now it's safe to release staging buffers -- no detached reader is
    // still touching them.
    for (H5Staging& s : h5_staging_)
    {
        if (s.gpu_pool_host)
        {
            CUDA_DRIVER_CHECK_NON_FATAL(cuMemFreeHost(s.gpu_pool_host));
            s.gpu_pool_host = nullptr;
        }
        delete[] s.chest_pool_host;
        s.chest_pool_host = nullptr;
        delete[] s.info_pool_host;
        s.info_pool_host = nullptr;
    }

    // Close pools. The bank already destroyed every placement-new'd
    // CVSrsChestBuff in its own destructor body before we got here, so
    // these `delete`s only release the underlying IPC backing memory.
    if (chest_buf_pool_)
    {
        delete chest_buf_pool_;
        chest_buf_pool_ = nullptr;
    }
    if (srs_info_pool_)
    {
        delete srs_info_pool_;
        srs_info_pool_ = nullptr;
    }
    if (ipc_gpu_pool_)
    {
        delete ipc_gpu_pool_;
        ipc_gpu_pool_ = nullptr;
    }
}

SrsInfoUpdate* SrsIpcManager::get_srs_info_update_buf(int buf_id) const
{
    return srs_info_pool_ ? srs_info_pool_->get_buf_addr(buf_id) : nullptr;
}

void SrsIpcManager::allocate_h5_staging()
{
    gpu_pool_total_bytes_   = static_cast<size_t>(ipc_gpu_pool_   ? ipc_gpu_pool_->get_pool_len()   : 0)
                            * static_cast<size_t>(ipc_gpu_pool_   ? ipc_gpu_pool_->get_buf_size()   : 0);
    chest_pool_total_bytes_ = static_cast<size_t>(chest_buf_pool_ ? chest_buf_pool_->get_pool_len() : 0)
                            * static_cast<size_t>(chest_buf_pool_ ? chest_buf_pool_->get_buf_size() : 0);
    info_pool_total_bytes_  = static_cast<size_t>(srs_info_pool_  ? srs_info_pool_->get_pool_len()  : 0)
                            * static_cast<size_t>(srs_info_pool_  ? srs_info_pool_->get_buf_size()  : 0);

    // Size the vector once; entries default-construct to all-nullptr,
    // matching the previous std::array<...> behaviour.
    h5_staging_.resize(h5_dump_slot_max_);

    bool alloc_ok = true;
    for (uint32_t i = 0; i < h5_dump_slot_max_; i++)
    {
        H5Staging& s = h5_staging_[i];
        if (gpu_pool_total_bytes_ > 0)
        {
            CUresult cerr = cuMemAllocHost(reinterpret_cast<void**>(&s.gpu_pool_host),
                                           gpu_pool_total_bytes_);
            if (cerr != CUDA_SUCCESS)
            {
                const char* cerr_str = nullptr;
                cuGetErrorString(cerr, &cerr_str);
                NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT,
                           "SrsIpcManager: cuMemAllocHost({}) for H5 staging slot {} failed: {} ({})",
                           gpu_pool_total_bytes_, i, static_cast<unsigned>(cerr), cerr_str ? cerr_str : "unknown");
                s.gpu_pool_host = nullptr;
                alloc_ok = false;
            }
        }
        if (chest_pool_total_bytes_ > 0)
        {
            s.chest_pool_host = new (std::nothrow) uint8_t[chest_pool_total_bytes_];
            if (s.chest_pool_host == nullptr) alloc_ok = false;
        }
        if (info_pool_total_bytes_ > 0)
        {
            s.info_pool_host = new (std::nothrow) uint8_t[info_pool_total_bytes_];
            if (s.info_pool_host == nullptr) alloc_ok = false;
        }
    }
    h5_staging_ready_ = alloc_ok;
    NVLOGC_FMT(TAG,
               "SrsIpcManager: H5 staging pre-allocated (slots={}, gpu={} B, chest={} B, info={} B, ready={})",
               h5_dump_slot_max_,
               gpu_pool_total_bytes_, chest_pool_total_bytes_, info_pool_total_bytes_,
               h5_staging_ready_ ? 1 : 0);
}

int SrsIpcManager::dump_h5(uint16_t sfn, uint16_t slot, uint32_t active_num_cells, const char* outputDir)
{
    if (h5_dump_slot_max_ == 0)
    {
        return 0;
    }
    // Conservative pre-check: avoids a needless fetch_add once we're capped.
    // The authoritative race-free check is the post-fetch_add bound below.
    if (h5_dump_slot_count_.load(std::memory_order_acquire) >= h5_dump_slot_max_)
    {
        return 0;
    }
    if (ipc_gpu_pool_ == nullptr || chest_buf_pool_ == nullptr || srs_info_pool_ == nullptr)
    {
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT,
                   "SrsIpcManager: pool not initialized (gpu=0x{:x}, chest=0x{:x}, info=0x{:x})",
                   reinterpret_cast<uintptr_t>(ipc_gpu_pool_),
                   reinterpret_cast<uintptr_t>(chest_buf_pool_),
                   reinterpret_cast<uintptr_t>(srs_info_pool_));
        return -1;
    }
    if (!h5_staging_ready_)
    {
        NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT,
                   "SrsIpcManager: H5 staging buffers not allocated (SFN {}.{})", sfn, slot);
        return -1;
    }

    // Reserve a staging slot before any I/O. fetch_add gives every
    // concurrent caller a unique index even when several UlPhyDriver
    // threads enter dump_h5() at once. If the post-add value lands past
    // the cap we treat the dump as a no-op (the pre-check filters the
    // common case; this branch handles the small race window between
    // the load above and the fetch_add).
    const uint32_t reserved  = h5_dump_slot_count_.fetch_add(1, std::memory_order_acq_rel);
    if (reserved >= h5_dump_slot_max_)
    {
        return 0;
    }
    const uint32_t dump_slot = reserved + 1;
    H5Staging&     staging   = h5_staging_[reserved];

    // Snapshot pool geometry. Pool pointers are stable for the lifetime of
    // the cuphycontroller process, so re-reading these per dump is fine.
    const uint32_t gpu_pool_len   = static_cast<uint32_t>(ipc_gpu_pool_->get_pool_len());
    const uint32_t gpu_buf_size   = ipc_gpu_pool_->get_buf_size();
    const void*    gpu_base       = ipc_gpu_pool_->get_buf_addr(0);
    const uint32_t chest_pool_len = static_cast<uint32_t>(chest_buf_pool_->get_pool_len());
    const uint32_t chest_buf_size = chest_buf_pool_->get_buf_size();
    const void*    chest_base     = chest_buf_pool_->get_buf_addr(0);
    const uint32_t info_buf_size  = srs_info_pool_->get_buf_size();
    // Pool is sized MAX_NUM_UE_SRS_INFO_PER_SLOT * MAX_CELLS_PER_SLOT * SLOTS_PER_FRAME * 16.
    // Each (sfn, slot) owns its own section indexed by (sfn & 0xF) * SLOTS_PER_FRAME + slot;
    // dump only the current section so the H5 dataset stays sized
    // NUM_CELL * MAX_NUM_UE_SRS_INFO_PER_SLOT (same as the cuMAC TV test-bench expects).
    const uint32_t info_pool_len_per_slot = MAX_NUM_UE_SRS_INFO_PER_SLOT * MAX_CELLS_PER_SLOT;
    const uint32_t info_slot_base_idx     = ((static_cast<uint32_t>(sfn) & 0xFU) * SLOTS_PER_FRAME
                                            + static_cast<uint32_t>(slot)) * info_pool_len_per_slot;
    const void*    info_base              = srs_info_pool_->get_buf_addr(info_slot_base_idx);

    const size_t gpu_bytes   = static_cast<size_t>(gpu_pool_len)   * static_cast<size_t>(gpu_buf_size);
    const size_t chest_bytes = static_cast<size_t>(chest_pool_len) * static_cast<size_t>(chest_buf_size);
    const size_t info_bytes_slot   = static_cast<size_t>(info_pool_len_per_slot) * static_cast<size_t>(info_buf_size);
    const size_t info_bytes_active = static_cast<size_t>(active_num_cells)
                                   * static_cast<size_t>(MAX_NUM_UE_SRS_INFO_PER_SLOT)
                                   * static_cast<size_t>(info_buf_size);
    const size_t info_bytes        = std::min(info_bytes_active, info_bytes_slot);

    //-----------------------------------------------------------------------
    // Phase 1 — synchronous snapshot in the caller's RT thread.
    //-----------------------------------------------------------------------
    using clock = std::chrono::steady_clock;
    const auto t_snap_start = clock::now();

    int snap_ret = 0;
    if (gpu_bytes > 0 && gpu_base != nullptr && staging.gpu_pool_host != nullptr)
    {
        if (gpu_device_) gpu_device_->setDevice();
        CUresult cerr = cuMemcpyDtoH(staging.gpu_pool_host, reinterpret_cast<CUdeviceptr>(gpu_base), gpu_bytes);
        if (cerr != CUDA_SUCCESS)
        {
            const char* cerr_str = nullptr;
            cuGetErrorString(cerr, &cerr_str);
            NVLOGE_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT,
                       "SrsIpcManager: cuMemcpyDtoH D2H failed (SFN {}.{}, bytes={}): {} ({})",
                       sfn, slot, gpu_bytes, static_cast<unsigned>(cerr), cerr_str ? cerr_str : "unknown");
            snap_ret = -1;
        }
    }
    if (chest_bytes > 0 && chest_base != nullptr && staging.chest_pool_host != nullptr)
    {
        std::memcpy(staging.chest_pool_host, chest_base, chest_bytes);
        for (uint32_t i = 0; i < chest_pool_len; i++)
        {
            auto *chest_buff_dump = reinterpret_cast<CVSrsChestBuff *>(
                staging.chest_pool_host + static_cast<size_t>(i) * static_cast<size_t>(chest_buf_size));
            if (chest_buff_dump != nullptr)
            {
                // Only scrub pointer-like/runtime descriptor fields in the dump copy.
                chest_buff_dump->scrubDumpOnlyMembers();
            }
        }
    }
    if (info_bytes > 0 && info_base != nullptr && staging.info_pool_host != nullptr)
    {
        std::memcpy(staging.info_pool_host, info_base, info_bytes);
    }

    const auto t_snap_end = clock::now();
    const long snap_us    = std::chrono::duration_cast<std::chrono::microseconds>(t_snap_end - t_snap_start).count();

    NVLOGI_FMT(TAG,
               "SrsIpcManager: RT snapshot done (SFN {}.{}, dump_slot={}/{}, gpu={} B, chest={} B, info={} B, snap_us={}, ret={})",
               sfn, slot, dump_slot, h5_dump_slot_max_,
               gpu_bytes, chest_bytes, info_bytes, snap_us, snap_ret);

    if (snap_ret != 0)
    {
        return snap_ret;
    }

    //-----------------------------------------------------------------------
    // Phase 2 — detached SCHED_OTHER thread writes the .h5 file.
    //-----------------------------------------------------------------------
    const char* dir = (outputDir != nullptr && outputDir[0] != '\0') ? outputDir : "/tmp";
    char fname[256];
    std::snprintf(fname, sizeof(fname), "%s/cubb_srs_buffers_%u_SFN_%u.%u.h5",
                  dir,
                  static_cast<unsigned>(dump_slot - 1),
                  static_cast<unsigned>(sfn),
                  static_cast<unsigned>(slot));

    H5WriteJob job{};
    job.fname          = fname;
    job.sfn            = sfn;
    job.slot           = slot;
    job.dump_slot      = dump_slot;
    job.max_dump_slots = h5_dump_slot_max_;
    job.gpu_pool_len   = gpu_pool_len;
    job.gpu_buf_size   = gpu_buf_size;
    job.num_prg        = cfg_.num_prg;
    job.num_gnb_ant    = static_cast<uint32_t>(cfg_.max_srs_antenna_ports);
    job.num_ue_layer   = cfg_.num_ue_layer;
    job.cell_num       = active_num_cells;
    job.gpu_h5         = staging.gpu_pool_host;
    job.gpu_bytes      = gpu_bytes;
    job.chest_h5       = staging.chest_pool_host;
    job.chest_bytes    = chest_bytes;
    job.info_h5        = staging.info_pool_host;
    job.info_bytes     = info_bytes;

    // Track the writer thread instead of detaching it: the destructor
    // joins all entries before releasing staging buffers, so even an
    // immediate teardown after the last dump can't race with an
    // in-flight HDF5 write.
    {
        std::lock_guard<std::mutex> lk(h5_threads_mtx_);
        h5_threads_.emplace_back(runH5WriteJob, std::move(job));
    }

    return 0;
}

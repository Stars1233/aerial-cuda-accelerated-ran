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

#define TAG (NVLOG_TAG_BASE_CUMAC_CP + 4) // "CUMCP.HANDLER"

#include <string.h>
#include <sys/time.h>

#include <new>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <pthread.h>
#include <sched.h>
#include <string>
#include <thread>

#include <cuda_runtime_api.h>
#include <hdf5.h>

#include "nvlog.hpp"
#include "nv_utils.h"

#include "nv_phy_utils.hpp"
#include "cumac_app.hpp"
#include "cumac.h"
#include "api.h"
#include "cumac_msg.h"
#include "cumac_cp_handler.hpp"
#include "cumac_cp_tv.hpp"
#include "cumac_muUeGrp.h"
#include <cuda_fp16.h>

#include "nv_phy_mac_transport.hpp"
#include "nv_phy_epoll_context.hpp"

#include "cv_memory_bank_srs_chest.hpp"

#include "nvlog.hpp"

using namespace std;

using namespace nv;
using namespace cumac;

using namespace std::chrono;

#define CHECK_PTR_NULL_FATAL(ptr)                                                                                  \
    do                                                                                                             \
    {                                                                                                              \
        if ((ptr) == nullptr)                                                                                      \
        {                                                                                                          \
            NVLOGF_FMT(TAG, AERIAL_CUMAC_CP_EVENT, "{} line {}: pointer {} is nullptr", __func__, __LINE__, #ptr); \
        }                                                                                                          \
    } while (0);

#define CHECK_VALUE_EQUAL_ERR(v1, v2)                                                                                                   \
    do                                                                                                                                  \
    {                                                                                                                                   \
        if ((v1) != (v2))                                                                                                               \
        {                                                                                                                               \
            NVLOGE_FMT(TAG, AERIAL_CUMAC_CP_EVENT, "{} line {}: values not equal: {}={}, {}={}", __func__, __LINE__, #v1, v1, #v2, v2); \
        }                                                                                                                               \
    } while (0);

#define CHECK_VALUE_MAX_ERR(val, max)                                                                                                                                                 \
    do                                                                                                                                                                                \
    {                                                                                                                                                                                 \
        if ((val) > (max))                                                                                                                                                            \
        {                                                                                                                                                                             \
            NVLOGE_FMT(TAG, AERIAL_CUMAC_CP_EVENT, "{} line {}: value > max: {}={} > {}={}", __func__, __LINE__, #val, static_cast<uint32_t>(val), #max, static_cast<uint32_t>(max)); \
        }                                                                                                                                                                             \
    } while (0);


//! Worst-case MU UE-pair GPU buffer sizes (same layout as cumac_cp_tv / L2 integration test).
size_t get_ue_pair_srs_chan_est_bytes(uint16_t n_bs_ant, uint16_t n_sub, uint16_t n_prg_samp, uint16_t cell_num)
{
    const uint64_t v = static_cast<uint64_t>(sizeof(__half2)) * static_cast<uint64_t>(n_bs_ant)
        * static_cast<uint64_t>(MAX_NUM_UE_ANT_PORT) * static_cast<uint64_t>(n_sub)
        * static_cast<uint64_t>(n_prg_samp) * static_cast<uint64_t>(MAX_NUM_SRS_UE_PER_CELL)
        * static_cast<uint64_t>(cell_num);
    return static_cast<size_t>(v);
}

size_t get_ue_pair_srs_snr_bytes(uint16_t cell_num)
{
    const uint64_t v = static_cast<uint64_t>(sizeof(float)) * static_cast<uint64_t>(MAX_NUM_SRS_UE_PER_CELL)
        * static_cast<uint64_t>(cell_num);
    return static_cast<size_t>(v);
}

size_t get_ue_pair_chan_orth_bytes(uint16_t n_sub, uint16_t n_prg_samp, uint16_t cell_num)
{
    const uint64_t n_pair = static_cast<uint64_t>(MAX_NUM_SRS_UE_PER_CELL) * static_cast<uint64_t>(MAX_NUM_UE_ANT_PORT);
    const uint64_t tri = n_pair * (n_pair + 1u) / 2u;
    const uint64_t v = static_cast<uint64_t>(sizeof(float)) * tri * static_cast<uint64_t>(n_sub)
        * static_cast<uint64_t>(n_prg_samp) * static_cast<uint64_t>(cell_num);
    return static_cast<size_t>(v);
}

inline constexpr int  MAX_SRS_CHEST_BUFFERS_PER_CELL = 1024;
inline constexpr uint32_t MAX_CELLS_MU_MIMO_ENABLE = 9;
inline constexpr uint32_t MAX_SRS_CHEST_BUFFERS = MAX_CELLS_MU_MIMO_ENABLE * MAX_SRS_CHEST_BUFFERS_PER_CELL;
inline constexpr uint32_t num_srs_buffers_per_cell = 1024;
size_t get_ue_pair_cubb_gpu_bytes(uint16_t n_prg, uint16_t n_bs_ant, uint16_t n_ue_layer, uint16_t cell_num)
{
    const size_t total_num_buffers = std::min(static_cast<size_t>(num_srs_buffers_per_cell) * cell_num, static_cast<size_t>(MAX_SRS_CHEST_BUFFERS));
    const size_t buffer_size = sizeof(uint32_t) * static_cast<size_t>(n_prg) * static_cast<size_t>(n_bs_ant) * static_cast<size_t>(n_ue_layer);
    NVLOGI_FMT(TAG, "{}: num_prg={} num_bs_ant={} num_ue_layer={} cell_num={} cubb_srs_gpu_buf_total_size: {} * {} = {}",
                __func__, n_prg, n_bs_ant, n_ue_layer, cell_num,
                buffer_size, total_num_buffers, buffer_size * total_num_buffers);
    return buffer_size * total_num_buffers;
}

template <typename T>
static T* cumac_init_msg_header(nv_ipc_msg_t* msg, int msg_id, int cell_id)
{
    size_t msg_size = sizeof(T);
    msg->msg_id = msg_id;
    msg->cell_id = cell_id;
    msg->msg_len = msg_size;
    msg->data_len = 0;

    cumac_msg_header_t *header = (cumac_msg_header_t*) msg->msg_buf;
    header->message_count = 1;
    header->handle_id = cell_id;
    header->type_id = msg_id;
    header->body_len = msg_size - sizeof(cumac_msg_header_t);
    return reinterpret_cast<T*>(msg->msg_buf);
}

void sched_slot_data::init_slot_data(cumac_cp_handler* _handler, uint32_t _cell_num) {
    cell_num = _cell_num;
    handler = _handler;
    slot_msgs.resize(cell_num);
    task = nullptr;
}

void sched_slot_data::reset_slot_data(sfn_slot_t ss) {
    NVLOGI_FMT(TAG, "SFN {}.{} init slot_data", ss.u16.sfn, ss.u16.slot);

    curr_cell_id = 0;

    if (task != nullptr) { // || ss_sched.u32 != SFN_SLOT_INVALID) {
        NVLOGE_FMT(TAG, AERIAL_CUMAC_CP_EVENT, "SFN {}.{} dropping previous incomplete slot SFN {}.{}",
                   ss.u16.sfn, ss.u16.slot, ss_sched.u16.sfn, ss_sched.u16.slot);

        // Free unhandled nvipc buffers
        nv::phy_mac_transport_wrapper &wrapper = handler->transport_wrapper();
        for (struct nv::phy_mac_msg_desc &msg_desc : task->tti_reqs)
        {
            if (msg_desc.msg_buf != nullptr)
            {
                NVLOGW_FMT(TAG, "SFN {}.{} dropping message SFN {}.{} cell_id={} msg_id=0x{:02X}", ss.u16.sfn, ss.u16.slot, ss_sched.u16.sfn, ss_sched.u16.slot, msg_desc.cell_id, msg_desc.msg_id);
                wrapper.rx_release(msg_desc);
                msg_desc.reset();
            }
        }

        // Free the cumac_task_t buffer
        handler->task_ring->free(task);
    }

    // TODO: for reorder
    for (uint32_t cell_id = 0; cell_id < cell_num; cell_id ++) {
        slot_msgs[cell_id].reset_cell_data();
    }

    if (handler == nullptr || handler->task_ring == nullptr) {
        NVLOGF_FMT(TAG, AERIAL_CUMAC_CP_EVENT, "Invalid pointer: handler or handler->task_ring is null");
        return;
    }

    ss_sched = ss;
    if ((task = handler->task_ring->alloc()) == nullptr) {
        NVLOGW_FMT(TAG, "SFN {}.{} task process can't catch up with enqueue, drop slot", ss.u16.sfn, ss.u16.slot);
        return;
    }

    task->reset_cumac_task(ss);
}

cumac_cp_handler::cumac_cp_handler(cumac_cp_configs& _configs, nv::phy_mac_transport_wrapper& wrapper) :
    configs(_configs),
    trans_wrapper(wrapper),
    task_order_sem_(std::max(
        std::size_t{1},
        static_cast<std::size_t>(_configs.worker_cores.size()) * static_cast<std::size_t>(_configs.thread_num_per_core)))
{
    configured_cell_num = 0;
    global_tick = 0;
    group_buf_size = configs.get_max_data_size() * configs.cell_num;

    task_ring = nullptr;
    ss_curr = {.u32 = SFN_SLOT_INVALID};

    cell_configs.resize(configs.cell_num);
    thrputs.resize(configs.cell_num);

    memset(&buf_num, 0, sizeof(cumac_buf_num_t));

    for (uint32_t buf_id = 0; buf_id < SCHED_SLOT_BUF_NUM; buf_id++)
    {
        sched_slots[buf_id].init_slot_data(this, configs.cell_num);
    }

    if (configs.enable_tv_test)
    {
        if (parse_group_tv(group_tv, configs.cell_num, configs.enable_gpu_share, configs.srs_slot_lag, configs.task_bitmask) < 0)
        {
            NVLOGW_FMT(TAG, "Failed to parse group TV");
        }
    }
    else
    {
        NVLOGC_FMT(TAG, "enable_tv_test is false: skipping group TV parse");
    }
}

cumac_cp_handler::~cumac_cp_handler() {
    free_mu_ue_pair_gpu();
    if (blerTargetActUe != nullptr) {
        free(blerTargetActUe);
    }
}

void cumac_cp_handler::free_mu_ue_pair_gpu()
{
    if (mu_ue_pair_gpu_ != nullptr)
    {
        delete mu_ue_pair_gpu_;
        mu_ue_pair_gpu_ = nullptr;
    }
    if (ue_pair_srs_chan_est_ != nullptr)
    {
        cudaFree(ue_pair_srs_chan_est_);
        ue_pair_srs_chan_est_ = nullptr;
    }
    if (ue_pair_srs_snr_ != nullptr)
    {
        cudaFree(ue_pair_srs_snr_);
        ue_pair_srs_snr_ = nullptr;
    }
    if (ue_pair_chan_orth_ != nullptr)
    {
        cudaFree(ue_pair_chan_orth_);
        ue_pair_chan_orth_ = nullptr;
    }
    if (cubb_gpu_pool_)
    {
        // Shared pool: ue_pair_cubb_gpu_ is a pointer into the pool, not a cudaMalloc'd buffer.
        cubb_gpu_pool_.reset();
        ue_pair_cubb_gpu_ = nullptr;
    }
    else if (ue_pair_cubb_gpu_ != nullptr)
    {
        cudaFree(ue_pair_cubb_gpu_);
        ue_pair_cubb_gpu_ = nullptr;
    }
    if (chest_buf_pool_)
    {
        chest_buf_pool_.reset();
    }
    if (srs_info_pool_)
    {
        srs_info_pool_.reset();
    }
    if (ue_pair_task_out_ != nullptr)
    {
        cudaFree(ue_pair_task_out_);
        ue_pair_task_out_ = nullptr;
    }

    free_ue_pair_h5_staging();
}

void cumac_cp_handler::init_ue_pair_gpu_resources()
{
    // Pick the first slot whose MU TV was actually loaded as the dimension
    // donor (slot 0 may be unscheduled now that schedule_slot_period drives
    // ue_pair sizing). Fall back to a default-constructed TV otherwise.
    const ue_pair_tv_t default_tv{};
    const ue_pair_tv_t *tv = &default_tv;
    for (const auto& slot_tv : group_tv.ue_pair)
    {
        if (slot_tv.mu_ue_pair_tv_loaded)
        {
            tv = &slot_tv;
            break;
        }
    }
    const bool tv_loaded = tv->mu_ue_pair_tv_loaded;

    // Parameters for maximum sizes required for muMimoUserPairing
    uint16_t num_prg = group_params.nPrbGrp;
    uint16_t num_bs_ant = static_cast<uint16_t>(group_params.nBsAnt);
    uint16_t num_subband = MAX_NUM_SUBBAND;
    uint16_t num_prg_samp = MAX_NUM_PRG_SAMP_PER_SUBBAND;
    uint16_t num_ue_layer = MAX_NUM_UE_ANT_PORT;

    if (configs.enable_tv_test && tv_loaded)
    {
        num_prg = tv->num_prg;
        num_subband = tv->num_subband;
        num_prg_samp = tv->num_prg_samp_per_subband;
        num_bs_ant = tv->num_bs_ant;
        num_ue_layer = tv->num_ue_ant;
        num_srs_info_ = tv->num_srs_ue_per_slot_cell;
    }

    // Save actual geometry so H5 dump attributes match what was allocated.
    ue_pair_num_prg_samp_per_subband_ = num_prg_samp;
    ue_pair_num_subband_              = num_subband;
    ue_pair_num_ue_ant_port_          = num_ue_layer;
    ue_pair_num_bs_ant_               = num_bs_ant;

    // Calculate static GPU buffer sizes for muMimoUserPairing
    ue_pair_chan_est_bytes_ = get_ue_pair_srs_chan_est_bytes(num_bs_ant, num_subband, num_prg_samp, group_params.nCell);
    ue_pair_snr_bytes_ = get_ue_pair_srs_snr_bytes(group_params.nCell);
    ue_pair_orth_bytes_ = get_ue_pair_chan_orth_bytes(num_subband, num_prg_samp, group_params.nCell);
    ue_pair_cubb_bytes_ = get_ue_pair_cubb_gpu_bytes(num_prg, num_bs_ant, num_ue_layer, group_params.nCell);

    // Calculate per slot task input buffer and output buffer sizes for muMimoUserPairing
    ue_pair_task_in_bytes_ = static_cast<size_t>(cumac_muUeGrp_req_info_size(configs.enable_gpu_share)) * static_cast<size_t>(group_params.nCell);
    ue_pair_out_bytes_ = sizeof(cumac_muUeGrp_resp_info_t) * static_cast<size_t>(group_params.nCell);

    NVLOGI_FMT(TAG, "{}: size: srs_chan_est={} srs_snr={} chan_orth={} cubb_gpu={} task_in_buf={} task_out_buf={}",
               __func__, ue_pair_chan_est_bytes_, ue_pair_snr_bytes_, ue_pair_orth_bytes_, ue_pair_cubb_bytes_, ue_pair_task_in_bytes_, ue_pair_out_bytes_);

    if (configs.enable_tv_test && tv_loaded)
    {
        CHECK_VALUE_EQUAL_ERR(ue_pair_chan_est_bytes_, tv->srs_chan_est_size);
        CHECK_VALUE_EQUAL_ERR(ue_pair_snr_bytes_, tv->srs_snr_size);
        CHECK_VALUE_EQUAL_ERR(ue_pair_orth_bytes_, tv->chan_orth_size);
        // cubb_srs_buf_size==0 means TV was generated without a CUBB dump (zeros stay from cudaMemset)
        if (tv->cubb_srs_buf_size > 0)
        {
            CHECK_VALUE_EQUAL_ERR(ue_pair_cubb_bytes_, tv->cubb_srs_buf_size);
        }
        CHECK_VALUE_EQUAL_ERR(ue_pair_task_in_bytes_, tv->task_in_buf_group_size);
        CHECK_VALUE_EQUAL_ERR(ue_pair_out_bytes_, sizeof(cumac_muUeGrp_resp_info_t) * static_cast<size_t>(group_params.nCell));
    }

    if (configs.run_in_cpu != 0)
    {
        NVLOGW_FMT(TAG, "{}: run_in_cpu={}, skip GPU module init", __func__, configs.run_in_cpu);
        return;
    }

    // Use num_bs_ant (set from TV when loaded) rather than group_params.nBsAnt (from CONFIG request),
    // so TV-only mode (no Cell_Configs H5) still proceeds when TV provides valid dims.
    if (num_prg == 0 || group_params.nCell == 0 || num_bs_ant == 0)
    {
        NVLOGE_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "{}: invalid MU dims: num_prg={} num_bs_ant={} nCell={}, skip GPU module init", __func__,
                   num_prg, num_bs_ant, group_params.nCell);
        return;
    }

    // Allocate static GPU buffers for muMimoUserPairing at most once
    if (mu_ue_pair_gpu_ == nullptr)
    {
        // Allocate static GPU buffers for muMimoUserPairing and initialize to 0
        CHECK_CUDA_ERR(cudaMalloc(reinterpret_cast<void**>(&ue_pair_srs_chan_est_), ue_pair_chan_est_bytes_));
        CHECK_CUDA_ERR(cudaMalloc(reinterpret_cast<void**>(&ue_pair_srs_snr_), ue_pair_snr_bytes_));
        CHECK_CUDA_ERR(cudaMalloc(reinterpret_cast<void**>(&ue_pair_chan_orth_), ue_pair_orth_bytes_));
        CHECK_CUDA_ERR(cudaMalloc(reinterpret_cast<void**>(&ue_pair_task_out_), ue_pair_out_bytes_));
        CHECK_CUDA_ERR(cudaMemset(ue_pair_srs_chan_est_, 0, ue_pair_chan_est_bytes_));
        CHECK_CUDA_ERR(cudaMemset(ue_pair_srs_snr_, 0, ue_pair_snr_bytes_));
        CHECK_CUDA_ERR(cudaMemset(ue_pair_chan_orth_, 0, ue_pair_orth_bytes_));
        CHECK_CUDA_ERR(cudaMemset(ue_pair_task_out_, 0, ue_pair_out_bytes_));

        // cubb_srs buffer: when enable_cubb=1 attach to cuphycontroller's shared pool;
        // otherwise cudaMalloc a local buffer (fallback below).
        if (configs.enable_cubb)
        {
            int cuda_dev = 0;
            cudaGetDevice(&cuda_dev);
            char pool_name[32];
            snprintf(pool_name, sizeof(pool_name), "CvSrsChest_GPU%d", cuda_dev);
            // Pool geometry must match cuphycontroller's CvSrsChestMemoryBank:
            //   pool_len  = min(num_srs_buffers_per_cell * nCell, MAX_SRS_CHEST_BUFFERS)
            //   buf_size  = sizeof(uint32_t) * n_prg * n_bs_ant * n_ue_layer = total / pool_len
            const size_t pool_len  = std::min(static_cast<size_t>(num_srs_buffers_per_cell) * static_cast<size_t>(group_params.nCell),
                                              static_cast<size_t>(MAX_SRS_CHEST_BUFFERS));
            const size_t pool_buf_size = (pool_len > 0) ? (ue_pair_cubb_bytes_ / pool_len) : ue_pair_cubb_bytes_;
            // unique_ptr ownership: if any of the three open() calls
            // throws, the already-constructed pools are released
            // automatically by stack unwinding (no manual delete chain
            // needed).
            cubb_gpu_pool_ = std::make_unique<nv::lock_free_mem_pool<uint8_t>>(
                static_cast<uint32_t>(pool_len), LOCK_FREE_OPT_SHM_SECONDARY, pool_name, cuda_dev, pool_buf_size);
            if (cubb_gpu_pool_->get_pool_len() > 0)
            {
                ue_pair_cubb_gpu_ = reinterpret_cast<__half2*>(cubb_gpu_pool_->get_buf_addr(0));
                NVLOGC_FMT(TAG, "{}: attached to shared GPU pool '{}' pool_len={} buf_addr={:p}",
                           __func__, pool_name, cubb_gpu_pool_->get_pool_len(), static_cast<void*>(ue_pair_cubb_gpu_));
            }
            else
            {
                NVLOGF_FMT(TAG, AERIAL_CUMAC_CP_EVENT, "{}: shared GPU pool '{}' not found, cannot attach to cuphycontroller's shared pool", __func__, pool_name);
            }

            snprintf(pool_name, sizeof(pool_name), "SrsChest");
            chest_buf_pool_ = std::make_unique<nv::lock_free_mem_pool<CVSrsChestBuff>>(
                static_cast<uint32_t>(pool_len), LOCK_FREE_OPT_SHM_SECONDARY, pool_name);
            if (chest_buf_pool_->get_pool_len() > 0)
            {
                NVLOGC_FMT(TAG, "{}: attached to shared CPU chest pool '{}' pool_len={}",
                           __func__, pool_name, chest_buf_pool_->get_pool_len());
            }
            else
            {
                NVLOGF_FMT(TAG, AERIAL_CUMAC_CP_EVENT, "{}: shared CPU chest pool '{}' not found, cannot attach to cuphycontroller's shared pool", __func__, pool_name);
            }

            snprintf(pool_name, sizeof(pool_name), "SrsInfo");
            uint32_t srs_info_pool_len = MAX_NUM_UE_SRS_INFO_PER_SLOT * MAX_CELLS_PER_SLOT * SLOT_NUM_PER_FRAME * 16;
            srs_info_pool_ = std::make_unique<nv::lock_free_mem_pool<SrsInfoUpdate>>(
                srs_info_pool_len, LOCK_FREE_OPT_SHM_SECONDARY, pool_name);
            if (srs_info_pool_->get_pool_len() > 0)
            {
                NVLOGC_FMT(TAG, "{}: attached to shared CPU srs_info pool '{}' pool_len={}",
                           __func__, pool_name, srs_info_pool_->get_pool_len());
            }
            else
            {
                NVLOGF_FMT(TAG, AERIAL_CUMAC_CP_EVENT, "{}: shared CPU srs_info pool '{}' not found, cannot attach to cuphycontroller's shared pool", __func__, pool_name);
            }
        }
        else
        {
            // Allocate local GPU buffer for cubb_srs_buf when enable_cubb=0
            CHECK_CUDA_ERR(cudaMalloc(reinterpret_cast<void**>(&ue_pair_cubb_gpu_), ue_pair_cubb_bytes_));
            CHECK_CUDA_ERR(cudaMemset(ue_pair_cubb_gpu_, 0, ue_pair_cubb_bytes_));
        }
    }
    else
    {
        delete mu_ue_pair_gpu_;
    }

    try
    {
        mu_ue_pair_gpu_ = new cumac::muMimoUserPairing(
            ue_pair_srs_chan_est_,
            ue_pair_srs_snr_,
            ue_pair_chan_orth_,
            ue_pair_cubb_gpu_,
            group_params.nCell,
            num_prg,
            num_subband,
            num_prg_samp,
            num_bs_ant);
    }
    catch (...)
    {
        NVLOGE_FMT(TAG, AERIAL_CUMAC_CP_EVENT, "{}: muMimoUserPairing construction failed", __func__);
        return;
    }

    NVLOGI_FMT(TAG, "{}: tv_loaded={} nCell={} num_prg={} num_subband={} num_prg_samp_per_subband={} num_bs_ant={} num_ue_layer={}",
               __func__, tv_loaded, group_params.nCell, num_prg, num_subband, num_prg_samp, num_bs_ant, num_ue_layer);
}

int cumac_cp_handler::load_ue_pair_static_buffers(sfn_slot_t ss, cudaStream_t strm)
{
    if (configs.run_in_cpu != 0)
    {
        NVLOGE_FMT(TAG, AERIAL_CUMAC_CP_EVENT, "{}: run_in_cpu={}, skip loading UE pair static buffers", __func__, configs.run_in_cpu);
        return -1;
    }

    if (group_tv.ue_pair.empty())
    {
        NVLOGE_FMT(TAG, AERIAL_CUMAC_CP_EVENT, "{}: ue_pair vector is empty, no MU TV loaded", __func__);
        return -1;
    }

    int tv_slot_idx = (ss.u16.sfn * SLOT_NUM_PER_FRAME + ss.u16.slot) % group_tv.ue_pair.size();
    if(first_ue_pair_tv_id < 0)
    {
        first_ue_pair_tv_id = tv_slot_idx;
    }

    const ue_pair_tv_t *src = &group_tv.ue_pair[tv_slot_idx];

    if(tv_slot_idx == first_ue_pair_tv_id)
    {
        // Reset task_out_buf at the first slot of each TV pattern period when enable_cubb=1
        CHECK_VALUE_EQUAL_ERR(sizeof(cumac_muUeGrp_resp_info_t) * group_params.nCell, ue_pair_out_bytes_);
        CHECK_CUDA_ERR(cudaMemsetAsync(ue_pair_task_out_, 0, ue_pair_out_bytes_, strm));

        if (!configs.enable_cubb)
        {
            CHECK_VALUE_EQUAL_ERR(src->chan_orth_size, ue_pair_orth_bytes_);
            CHECK_VALUE_EQUAL_ERR(src->srs_snr_size, ue_pair_snr_bytes_);
            CHECK_VALUE_EQUAL_ERR(src->srs_chan_est_size, ue_pair_chan_est_bytes_);

            CHECK_CUDA_ERR(cudaMemcpyAsync(ue_pair_chan_orth_, src->chan_orth_host, src->chan_orth_size, cudaMemcpyHostToDevice, strm));
            CHECK_CUDA_ERR(cudaMemcpyAsync(ue_pair_srs_snr_, src->srs_snr_host, src->srs_snr_size, cudaMemcpyHostToDevice, strm));
            CHECK_CUDA_ERR(cudaMemcpyAsync(ue_pair_srs_chan_est_, src->srs_chan_est_host, src->srs_chan_est_size, cudaMemcpyHostToDevice, strm));

            // Copy cubb_srs_buf from TV to local GPU buffer when enable_gpu_share != 0
            if (configs.enable_gpu_share)
            {
                CHECK_VALUE_EQUAL_ERR(src->cubb_srs_buf_size, ue_pair_cubb_bytes_);
                CHECK_CUDA_ERR(cudaMemcpyAsync(ue_pair_cubb_gpu_, src->cubb_srs_buf_host, src->cubb_srs_buf_size, cudaMemcpyHostToDevice, strm));
            }
            NVLOGI_FMT(TAG, "{}: SFN {}.{} gpu_share={} enable_cubb={} tv_slot_idx={} first_tv_slot={} - loaded all buffers",
                __func__, ss.u16.sfn, ss.u16.slot, configs.enable_gpu_share, configs.enable_cubb, tv_slot_idx, first_ue_pair_tv_id);
        }
        else
        {
            NVLOGI_FMT(TAG, "{}: SFN {}.{} gpu_share={} enable_cubb={} tv_slot_idx={} first_tv_slot={} - only reset task_out",
                __func__, ss.u16.sfn, ss.u16.slot, configs.enable_gpu_share, configs.enable_cubb, tv_slot_idx, first_ue_pair_tv_id);
        }
    }
    else
    {
        NVLOGI_FMT(TAG, "{}: SFN {}.{} gpu_share={} enable_cubb={} tv_slot_idx={} first_tv_slot={} - skipped",
            __func__, ss.u16.sfn, ss.u16.slot, configs.enable_gpu_share, configs.enable_cubb, tv_slot_idx, first_ue_pair_tv_id);
    }

    return 0;
}

int cumac_cp_handler::load_ue_pair_shared_memory(sfn_slot_t ss, std::vector<struct nv::phy_mac_msg_desc>& tti_reqs)
{
    nanoseconds ts_start = duration_cast<nanoseconds>(system_clock::now().time_since_epoch());

    // Print SRS info for debug
    if (!configs.enable_cubb)
    {
        uint32_t total_num_srs_ue = 0;
        for (int cell_id = 0; cell_id < configs.cell_num; cell_id++)
        {
            cumac_sch_tti_req_t& head = *reinterpret_cast<cumac_sch_tti_req_t*>(tti_reqs[cell_id].msg_buf);
            cumac_tti_req_payload_t& req = head.payload;
            uint8_t *data_buf_base = reinterpret_cast<uint8_t *>(tti_reqs[cell_id].data_buf);
            cumac_muUeGrp_req_info_t* muUeGrpInfo = reinterpret_cast<cumac_muUeGrp_req_info_t*>(data_buf_base + req.offsets.muUeGrpInfo);
            total_num_srs_ue += muUeGrpInfo->numSrsInfo;
            if (configs.enable_gpu_share)
            {
                cumac_muUeGrp_req_srs_info_msh_t* srsInfoMsh = reinterpret_cast<cumac_muUeGrp_req_srs_info_msh_t*>(muUeGrpInfo->payload);
                for (int srs_info_idx = 0; srs_info_idx < muUeGrpInfo->numSrsInfo; srs_info_idx++)
                {
                    cumac_muUeGrp_req_srs_info_msh_t* info = &srsInfoMsh[srs_info_idx];
                    NVLOGD_FMT(TAG, "{}: SFN {}.{} cell_id={} srs_info_idx={}-{} rnti={} id={} srsWbSnr={} real_buff_idx={}",
                        __func__, ss.u16.sfn, ss.u16.slot, cell_id, muUeGrpInfo->numSrsInfo, srs_info_idx, info->rnti, info->id, info->srsWbSnr, info->realBuffIdx);
                }
            }
            else
            {
                cumac_muUeGrp_req_srs_info_t* srsInfo = reinterpret_cast<cumac_muUeGrp_req_srs_info_t*>(muUeGrpInfo->payload);
                for (int srs_info_idx = 0; srs_info_idx < muUeGrpInfo->numSrsInfo; srs_info_idx++)
                {
                    cumac_muUeGrp_req_srs_info_t* info = &srsInfo[srs_info_idx];
                    NVLOGD_FMT(TAG, "{}: SFN {}.{} cell_id={} srs_info_idx={}-{} rnti={} id={} srsWbSnr={}",
                        __func__, ss.u16.sfn, ss.u16.slot, cell_id, muUeGrpInfo->numSrsInfo, srs_info_idx, info->rnti, info->id, info->srsWbSnr);
                }
            }
        }
        nanoseconds ts_end = duration_cast<nanoseconds>(system_clock::now().time_since_epoch());
        NVLOGI_FMT(TAG, "{}: SFN {}.{} total_num_srs_ue={} cell_num={} gpu_share={} enable_cubb={} time_ns={}",
            __func__, ss.u16.sfn, ss.u16.slot, total_num_srs_ue, configs.cell_num, configs.enable_gpu_share, configs.enable_cubb, ts_end.count() - ts_start.count());

        return 0;
    }

    if (!chest_buf_pool_ || chest_buf_pool_->get_pool_len() <= 0 ||
        !srs_info_pool_  || srs_info_pool_->get_pool_len()  <= 0)
    {
        NVLOGE_FMT(TAG, AERIAL_CUMAC_CP_EVENT, "{}: shared pools not ready", __func__);
        return -1;
    }

    // Load SRS info into shared memory pool.
    //
    // These two pools are produced by cuphydriver and only *read* by
    // cuMAC. The lock_free_mem_pool API is generic (returns a mutable
    // T*) because cuphydriver does write through it; we use a const
    // view locally here to enforce read-only access at the cuMAC-CP
    // consumer site -- any accidental write through these pointers is
    // now a compile error.
    CVSrsChestBuff* chest_buf = chest_buf_pool_->get_buf_addr(0);
    // Pool is indexed by (sfn, slot): section_id = (sfn & 0xF) * SLOTS_PER_FRAME + slot.
    // Apply srs_slot_lag inverse: MAC (sfn, slot) reads PHY write at
    //   phy_slot = (slot - lag + 20) % 20
    //   phy_sfn_mod16 = (sfn - (slot < lag ? 1 : 0) + 16) & 0xF
    const uint32_t srs_info_pool_len_per_slot = MAX_NUM_UE_SRS_INFO_PER_SLOT * MAX_CELLS_PER_SLOT;
    const uint32_t phy_slot = (static_cast<uint32_t>(ss.u16.slot) + SLOT_NUM_PER_FRAME
                                - static_cast<uint32_t>(configs.srs_slot_lag)) % SLOT_NUM_PER_FRAME;
    const uint32_t phy_sfn_mod16 = (static_cast<uint32_t>(ss.u16.sfn)
                                    + 16U
                                    - (ss.u16.slot < configs.srs_slot_lag ? 1U : 0U)) & 0xFU;
    const uint32_t srs_slot_section_idx = phy_sfn_mod16 * SLOT_NUM_PER_FRAME + phy_slot;
    const SrsInfoUpdate*  srs_info  = srs_info_pool_->get_buf_addr(srs_slot_section_idx * srs_info_pool_len_per_slot);
    uint32_t chest_buf_pool_len = chest_buf_pool_->get_pool_len();
    uint32_t srs_info_pool_len = srs_info_pool_len_per_slot;

    uint32_t total_num_srs_ue = 0;
    for (int cell_id = 0; cell_id < configs.cell_num; cell_id++)
    {
        cumac_sch_tti_req_t& head = *reinterpret_cast<cumac_sch_tti_req_t*>(tti_reqs[cell_id].msg_buf);
        cumac_tti_req_payload_t& req = head.payload;
        uint8_t *data_buf_base = reinterpret_cast<uint8_t *>(tti_reqs[cell_id].data_buf);
        cumac_muUeGrp_req_info_t* muUeGrpInfo = reinterpret_cast<cumac_muUeGrp_req_info_t*>(data_buf_base + req.offsets.muUeGrpInfo);
        total_num_srs_ue += muUeGrpInfo->numSrsInfo;
    }

    uint32_t err_count = 0;
    for (int group_ue_id = 0; group_ue_id < total_num_srs_ue; group_ue_id++) {
        if (group_ue_id >= srs_info_pool_len)
        {
            NVLOGE_FMT(TAG, AERIAL_CUMAC_CP_EVENT, "{}: SFN {}.{} srs_info_pool_len={} group_ue_id={} - out of range",
                __func__, ss.u16.sfn, ss.u16.slot, srs_info_pool_len, group_ue_id);
            err_count++;
            break;
        }

        uint16_t cIdx = srs_info[group_ue_id].cell_idx;
        uint16_t srs_info_idx = srs_info[group_ue_id].srs_info_idx;
        uint32_t real_buff_idx = srs_info[group_ue_id].real_buff_idx;

        if (cIdx >= configs.cell_num || real_buff_idx >= chest_buf_pool_len)
        {
            NVLOGE_FMT(TAG, AERIAL_CUMAC_CP_EVENT, "{}: SFN {}.{} cell_id={} group_ue_id={} srs_info_idx={} real_buff_idx={} - out of range",
                __func__, ss.u16.sfn, ss.u16.slot, cIdx, group_ue_id, srs_info_idx, real_buff_idx);
            err_count++;
            continue;
        }

        cumac_sch_tti_req_t& head = *reinterpret_cast<cumac_sch_tti_req_t*>(tti_reqs[cIdx].msg_buf);
        uint8_t *data_buf_base = reinterpret_cast<uint8_t *>(tti_reqs[cIdx].data_buf);
        cumac_tti_req_payload_t& req = head.payload;
        cumac_muUeGrp_req_info_t* muUeGrpInfo = reinterpret_cast<cumac_muUeGrp_req_info_t*>(data_buf_base + req.offsets.muUeGrpInfo);

        if (srs_info_idx >= muUeGrpInfo->numSrsInfo)
        {
            NVLOGE_FMT(TAG, AERIAL_CUMAC_CP_EVENT, "{}: SFN {}.{} cell_id={} group_ue_id={} srs_info_idx={} real_buff_idx={} muUeGrpInfo->numSrsInfo={} - out of range",
                __func__, ss.u16.sfn, ss.u16.slot, cIdx, group_ue_id, srs_info_idx, real_buff_idx, muUeGrpInfo->numSrsInfo);
            err_count++;
            continue;
        }

        CVSrsChestBuff* ue_buffer = chest_buf + real_buff_idx;
        uint16_t srsStartPrg;
        uint16_t srsStartValidPrg;
        uint16_t srsNValidPrg;
        uint8_t  srsPrgSize;
        ue_buffer->getSrsPrgInfo(&srsPrgSize, &srsStartPrg, &srsStartValidPrg, &srsNValidPrg);

        cumac_muUeGrp_req_srs_info_msh_t* srsInfoMsh = reinterpret_cast<cumac_muUeGrp_req_srs_info_msh_t*>(muUeGrpInfo->payload) + srs_info_idx;
        srsInfoMsh->realBuffIdx = real_buff_idx;
        srsInfoMsh->srsStartPrg = srsStartPrg;
        srsInfoMsh->srsStartValidPrg = srsStartValidPrg;
        srsInfoMsh->srsNValidPrg = srsNValidPrg;
        srsInfoMsh->flags = 0x01; // valid

        if (srs_info[group_ue_id].rnti != srsInfoMsh->rnti) {
            NVLOGE_FMT(TAG, AERIAL_CUMAC_CP_EVENT, "{}: SFN {}.{} cell_id={} group_ue_id={} srs_info_idx={} rnti mismatch: pool={} req={}",
                __func__, ss.u16.sfn, ss.u16.slot, cIdx, group_ue_id, srs_info_idx,
                srs_info[group_ue_id].rnti, srsInfoMsh->rnti);
        }

        NVLOGD_FMT(TAG, "{}: SFN {}.{} cell_id={} group_ue_id={} srs_info_idx={}-{} real_buff_idx={} rnti={} id={}",
            __func__, ss.u16.sfn, ss.u16.slot, cIdx, group_ue_id, muUeGrpInfo->numSrsInfo, srs_info_idx, real_buff_idx, srsInfoMsh->rnti, srsInfoMsh->id);
    }

    nanoseconds ts_end = duration_cast<nanoseconds>(system_clock::now().time_since_epoch());

    NVLOGI_FMT(TAG, "{}: SFN {}.{} total_num_srs_ue={} cell_num={} srs_info_pool_len={} chest_buf_pool_len={} err_count={} time_ns={}",
        __func__, ss.u16.sfn, ss.u16.slot, total_num_srs_ue, configs.cell_num, srs_info_pool_len, chest_buf_pool_len, err_count, ts_end.count() - ts_start.count());

    return 0;
}

int cumac_cp_handler::check_config_params()
{
    memset(&group_params, 0, sizeof(group_params));
    group_params.sigmaSqrd = 1.0; // Default; per-TTI value is taken from SCH_TTI.request in on_sch_tti_request

    for (int cell_id = 0; cell_id < configs.cell_num; cell_id++)
    {
        cumac_cell_configs_t &cell = cell_configs[cell_id];

        group_params.nUe += cell.nMaxSchUePerCell;
        group_params.nCell += 1;
        group_params.totNumCell += 1;
        group_params.nMaxSchdUePerRnd += cell.nMaxSchUePerCell;
        group_params.nActiveUe += cell.nMaxActUePerCell; // group_params.nActiveUe is used for max buffer size calculation, not the real active UEs count.

        if (cell_id == 0)
        {
            group_params.nPrbGrp = cell.nMaxPrg;

            group_params.nBsAnt = cell.nMaxBsAnt;
            group_params.nUeAnt = cell.nMaxUeAnt;

            // 12 * subcarrier spacing * number of PRBs per PRG
            group_params.W = 12 * cell.scSpacing * cell.nPrbPerPrg;

            group_params.maxNumUePerCell = cell.nMaxActUePerCell;
            group_params.betaCoeff = cell.betaCoeff;

            group_params.numUeSchdPerCellTTI = cell.nMaxSchUePerCell;
            group_params.precodingScheme = cell.precoderType;
            group_params.receiverScheme = cell.receiverType;
            group_params.allocType = cell.allocType;

            group_params.columnMajor = cell.colMajChanAccess;
            group_params.sinValThr = cell.sinValThr;

            group_params.mcsSelSinrCapThr = cell.mcsSelSinrCapThr;
            group_params.mcsSelLutType = cell.mcsSelLutType;
            group_params.harqEnabledInd = cell.harqEnabledInd;
            group_params.mcsSelCqi = cell.mcsSelCqi;
        }
        else
        {
            CHECK_VALUE_EQUAL_ERR(group_params.nPrbGrp, cell.nMaxPrg);

            CHECK_VALUE_EQUAL_ERR(group_params.nBsAnt, cell.nMaxBsAnt);
            CHECK_VALUE_EQUAL_ERR(group_params.nUeAnt, cell.nMaxUeAnt);

            CHECK_VALUE_EQUAL_ERR(group_params.W, 12 * cell.scSpacing * cell.nPrbPerPrg);

            CHECK_VALUE_EQUAL_ERR(group_params.maxNumUePerCell, cell.nMaxActUePerCell);
            CHECK_VALUE_EQUAL_ERR(group_params.betaCoeff, cell.betaCoeff);

            CHECK_VALUE_EQUAL_ERR(group_params.numUeSchdPerCellTTI, cell.nMaxSchUePerCell);
            CHECK_VALUE_EQUAL_ERR(group_params.precodingScheme, cell.precoderType);
            CHECK_VALUE_EQUAL_ERR(group_params.receiverScheme, cell.receiverType);
            CHECK_VALUE_EQUAL_ERR(group_params.allocType, cell.allocType);

            CHECK_VALUE_EQUAL_ERR(group_params.columnMajor, cell.colMajChanAccess);
            CHECK_VALUE_EQUAL_ERR(group_params.sinValThr, cell.sinValThr);

            CHECK_VALUE_EQUAL_ERR(group_params.mcsSelSinrCapThr, cell.mcsSelSinrCapThr);
            CHECK_VALUE_EQUAL_ERR(group_params.mcsSelLutType, cell.mcsSelLutType);
            CHECK_VALUE_EQUAL_ERR(group_params.harqEnabledInd, cell.harqEnabledInd);
            CHECK_VALUE_EQUAL_ERR(group_params.mcsSelCqi, cell.mcsSelCqi);

            CHECK_VALUE_EQUAL_ERR(cell_configs[0].blerTarget, cell.blerTarget);
        }
    }

    nanoseconds ts_start = duration_cast<nanoseconds>(system_clock::now().time_since_epoch());

    cumacSchedulerParam &p = group_params;
    NVLOGC_FMT(TAG, "GroupParams-1: moduleBitMask={} nUe={} nCell={} totNumCell={} nPrbGrp={} nBsAnt={} nUeAnt={} W={} sigmaSqrd={} maxNumUePerCell={} nMaxSchdUePerRnd={} betaCoeff={} harqEnabledInd={}",
        moduleBitMask, p.nUe, p.nCell, p.totNumCell, p.nPrbGrp, p.nBsAnt, p.nUeAnt, p.W, p.sigmaSqrd, p.maxNumUePerCell, p.nMaxSchdUePerRnd, p.betaCoeff, p.harqEnabledInd);
    NVLOGC_FMT(TAG, "GroupParams-2: nActiveUe={} numUeSchdPerCellTTI={} precodingScheme={} receiverScheme={} allocType={} columnMajor={} allocType={} columnMajor={} sinValThr={} mcsSelLutType={} mcsSelSinrCapThr={} mcsSelCqi={}",
        p.nActiveUe, p.numUeSchdPerCellTTI, p.precodingScheme, p.receiverScheme, p.allocType, p.columnMajor, p.allocType, p.columnMajor, p.sinValThr, p.mcsSelLutType, p.mcsSelSinrCapThr, p.mcsSelCqi);

    if (configs.group_buffer_enable && configs.run_in_cpu != 1)
    {
        const uint32_t need = compute_group_buf_need_bytes();
        const uint32_t legacy = static_cast<uint32_t>(configs.get_max_data_size()) * configs.cell_num;
        const uint32_t new_size = std::max(legacy, need);
        if (new_size > group_buf_size)
        {
            NVLOGI_FMT(TAG, "{}: group_buf_size increased to {} bytes (IPC/legacy hint={}, layout need={})", __func__, new_size, legacy, need);
            group_buf_size = new_size;
        }
    }

    if ((moduleBitMask & CUMAC_CP_TASK_MASK_MU_UE_GRP) != 0U)
    {
        init_ue_pair_gpu_resources();
    }

    uint32_t ring_len = task_ring->get_ring_len();
    
    for (int i = 0; i < ring_len; i++)
    {
        cumac_task *task = task_ring->get_buf_addr(i);
        if (task == nullptr)
        {
            NVLOGE_FMT(TAG, AERIAL_CUMAC_CP_EVENT, "Error cumac_task ring length: i={} length={}", i, ring_len);
            return -1;
        }

        // Initiate object at pre-allocated memory
        cumac_task *task_obj = new (task) cumac_task();
        if (task_obj != task)
        {
            NVLOGE_FMT(TAG, AERIAL_CUMAC_CP_EVENT, "{}: {} task_obj={} task={}", __func__, i, (void *)task_obj, (void *)task);
        }

        const int result = initiate_cumac_task(task);
        if (result != 0)
        {
            NVLOGE_FMT(TAG, AERIAL_CUMAC_CP_EVENT, "{}: initiate_cumac_task failed for task[{}]", __func__, i);
            return result;
        }
    }

    nanoseconds ts_end = duration_cast<nanoseconds>(system_clock::now().time_since_epoch());
    NVLOGC_FMT(TAG, "{}: buffer allocated: group_buf_size={} duration={}ns nCells={} nUe={} nPrbGrp={} nBsAnt={} nUeAnt={} betaCoeff={}", __func__,
            group_buf_size, ts_end.count() - ts_start.count(), group_params.nCell, group_params.nUe, group_params.nPrbGrp, group_params.nBsAnt, group_params.nUeAnt, group_params.betaCoeff);
    return 0;
}

int cumac_cp_handler::check_task_buf_size(cumac_task *task)
{
    if (task == nullptr)
    {
        return 0;
    }

    // UE_SELECTION buffers
    CHECK_VALUE_MAX_ERR(task->data_num.cellId, buf_num.cellId);
    CHECK_VALUE_MAX_ERR(task->data_num.prgMsk, buf_num.prgMsk);
    CHECK_VALUE_MAX_ERR(task->data_num.wbSinr, buf_num.wbSinr);

    CHECK_VALUE_MAX_ERR(task->data_num.avgRatesActUe, buf_num.avgRatesActUe);
    CHECK_VALUE_MAX_ERR(task->data_num.cellAssocActUe, buf_num.cellAssocActUe);

    CHECK_VALUE_MAX_ERR(task->data_num.setSchdUePerCellTTI, buf_num.setSchdUePerCellTTI);

    // PRB_ALLOCATION buffers
    CHECK_VALUE_MAX_ERR(task->data_num.cellAssoc, buf_num.cellAssoc);

    CHECK_VALUE_MAX_ERR(task->data_num.postEqSinr, buf_num.postEqSinr);
    CHECK_VALUE_MAX_ERR(task->data_num.sinVal, buf_num.sinVal);

    CHECK_VALUE_MAX_ERR(task->data_num.detMat, buf_num.detMat);
    CHECK_VALUE_MAX_ERR(task->data_num.prdMat, buf_num.prdMat);
    CHECK_VALUE_MAX_ERR(task->data_num.estH_fr, buf_num.estH_fr);

    return 0;
}

int cumac_task_callback_func(cumac_task* task, void* arg) {
    cumac_cp_handler* handler = static_cast<cumac_cp_handler*>(arg);
    handler->cumac_task_callback(task);
    return 0;
}

int cumac_cp_handler::cumac_task_callback(cumac_task *task)
{
    // Send regular scheduler responses for each cell
    for (int cell_id = 0; cell_id < group_params.nCell; cell_id ++) {
        send_sch_tti_response(task, cell_id);
    }
    return 0;
}

#define CUMAC_GPU_ALIGN_BYTES (16)

template <typename T>
int cumac_cp_handler::malloc_cumac_buf(cumac_task *task, T **ptr, uint32_t *num_save, uint32_t num, uint32_t force_host_mem)
{
    if (num == 0)
    {
        *ptr = nullptr;
        NVLOGE_FMT(TAG, AERIAL_CUMAC_CP_EVENT, "{}: memory allocation size is 0, num={}", __func__, num);
        return -1;
    }

    if (task->run_in_cpu || force_host_mem) // Allocate CPU memory
    {
        CHECK_CUDA_ERR(cudaMallocHost((void **)ptr, sizeof(T) * num));
    }
    else if (task->group_buf_enabled)
    {
        // Allocate a section from the contiguous group_buf
        *ptr = reinterpret_cast<T *>(task->group_buf + task->group_buf_offset);
        task->group_buf_offset += sizeof(T) * num;
        // Add padding bytes to align
        task->group_buf_offset = (task->group_buf_offset + CUMAC_GPU_ALIGN_BYTES - 1) & ~(CUMAC_GPU_ALIGN_BYTES - 1);
        if (task->group_buf_offset > group_buf_size)
        {
            NVLOGF_FMT(TAG, AERIAL_CUMAC_CP_EVENT, "{}: group_buf size={} exceeds allocated size={}", __func__, task->group_buf_offset, group_buf_size);
        }
    }
    else // Allocate GPU memory
    {
        CHECK_CUDA_ERR(cudaMalloc((void **)ptr, sizeof(T) * num));
    }

    if (num_save != nullptr) // Save the buffer number
    {
        *num_save = num;
    }
    return 0;
}

uint32_t cumac_cp_handler::compute_group_buf_need_bytes() const
{
    const cumacSchedulerParam &g = group_params;
    const uint32_t nCell = g.nCell;
    const uint32_t nUe = g.nUe;
    const uint32_t nActiveUe = g.nActiveUe;
    const uint32_t nPrbGrp = g.nPrbGrp;
    const uint32_t nBsAnt = g.nBsAnt;
    const uint32_t nUeAnt = g.nUeAnt;

    size_t off = 0;
    auto add_aligned = [&off](size_t nbytes) {
        off += nbytes;
        off = (off + static_cast<size_t>(CUMAC_GPU_ALIGN_BYTES - 1)) &
              ~(static_cast<size_t>(CUMAC_GPU_ALIGN_BYTES - 1));
    };

    for (uint32_t c = 0; c < nCell; c++)
    {
        add_aligned(static_cast<size_t>(nPrbGrp) * sizeof(uint8_t));
    }
    add_aligned(static_cast<size_t>(nCell) * sizeof(uint16_t));
    add_aligned(static_cast<size_t>(nCell) * static_cast<size_t>(nUe) * sizeof(uint8_t));
    add_aligned(static_cast<size_t>(nCell) * static_cast<size_t>(nActiveUe) * sizeof(uint8_t));
    add_aligned(static_cast<size_t>(nActiveUe) * sizeof(float));
    add_aligned(static_cast<size_t>(nActiveUe) * static_cast<size_t>(nUeAnt) * sizeof(float));
    add_aligned(static_cast<size_t>(nActiveUe) * static_cast<size_t>(nPrbGrp) * static_cast<size_t>(nUeAnt) *
                  sizeof(float));
    add_aligned(static_cast<size_t>(nUe) * static_cast<size_t>(nPrbGrp) * static_cast<size_t>(nUeAnt) *
                  sizeof(float));

    const uint32_t prdLen = nUe * nPrbGrp * nBsAnt * nBsAnt;
    const uint32_t detLen = prdLen;
    const uint32_t hLen = nPrbGrp * nUe * nCell * nBsAnt * nUeAnt;
    add_aligned(static_cast<size_t>(prdLen) * sizeof(cuComplex));
    add_aligned(static_cast<size_t>(detLen) * sizeof(cuComplex));
    add_aligned(static_cast<size_t>(hLen) * sizeof(cuComplex));

    add_aligned(static_cast<size_t>(nUe) * sizeof(uint16_t));
    if (g.allocType == 1)
    {
        add_aligned(static_cast<size_t>(nUe) * 2U * sizeof(int16_t));
        const uint32_t pfSize = static_cast<uint32_t>(nPrbGrp) * static_cast<uint32_t>(g.numUeSchdPerCellTTI);
        uint32_t pow2N = 2;
        while (pow2N < pfSize)
        {
            pow2N <<= 1;
        }
        add_aligned(static_cast<size_t>(nCell) * static_cast<size_t>(pow2N) * sizeof(float));
        add_aligned(static_cast<size_t>(nCell) * static_cast<size_t>(pow2N) * sizeof(uint16_t));
    }
    else
    {
        add_aligned(static_cast<size_t>(nCell) * static_cast<size_t>(nPrbGrp) * sizeof(int16_t));
    }
    add_aligned(static_cast<size_t>(nUe) * sizeof(uint8_t));
    add_aligned(static_cast<size_t>(nUe) * sizeof(int16_t));

    add_aligned(static_cast<size_t>(nActiveUe) * sizeof(float));
    add_aligned(static_cast<size_t>(nUe) * sizeof(float));
    add_aligned(static_cast<size_t>(nActiveUe) * sizeof(int8_t));
    add_aligned(static_cast<size_t>(nActiveUe) * sizeof(int8_t));
    add_aligned(static_cast<size_t>(nUe) * sizeof(int8_t));

    add_aligned(static_cast<size_t>(nCell) * sizeof(cumac_pfm_cell_info_t));

    const uint32_t ue_grp_stride =
        static_cast<uint32_t>(cumac_muUeGrp_req_info_size(configs.enable_gpu_share));
    add_aligned(static_cast<size_t>(nCell) * static_cast<size_t>(ue_grp_stride));

    add_aligned(static_cast<size_t>(nCell) * sizeof(uint8_t *));

    if (off > static_cast<size_t>(UINT32_MAX))
    {
        NVLOGE_FMT(TAG, AERIAL_CUMAC_CP_EVENT, "{}: computed group_buf need {} exceeds UINT32_MAX", __func__, off);
        return UINT32_MAX;
    }
    return static_cast<uint32_t>(off);
}

int cumac_cp_handler::initiate_cumac_task(cumac_task *task)
{
    if (task == nullptr)
    {
        NVLOGE_FMT(TAG, AERIAL_CUMAC_CP_EVENT, "{}: task pointer is NULL!", __func__);
        return -1;
    }

    task->callback_fun = cumac_task_callback_func;
    task->callback_args = this;
    task->cp_handler = this;
    task->cell_num = group_params.nCell;
    task->debug_option = configs.debug_option;
    task->group_buf_enabled = configs.group_buffer_enable;
    task->tti_reqs.resize(task->cell_num);
    task->group_buf_offset = 0;

    // Alloc host-pinned memory for debug log print
    CHECK_CUDA_ERR(cudaMallocHost((void **)&task->debug_buffer, CUMAC_TASK_DEBUG_BUF_MAX_SIZE));
    if (task->debug_buffer == nullptr)
    {
        NVLOGE_FMT(TAG, AERIAL_CUMAC_CP_EVENT, "{}: cudaMallocHost failed for debug_buffer", __func__);
        return -1;
    }

    // Allocate CPU and GPU memory for cell_desc_t: size = sizeof(cell_desc_t) * cell_num
    task->cpu_cell_descs = reinterpret_cast<cell_desc_t *>(malloc(sizeof(cell_desc_t) * task->cell_num));
    if (task->cpu_cell_descs == nullptr)
    {
        NVLOGE_FMT(TAG, AERIAL_CUMAC_CP_EVENT, "{}: malloc failed for cpu_cell_descs", __func__);
        return -1;
    }

    CHECK_CUDA_ERR(cudaMalloc(&task->gpu_cell_descs, sizeof(cell_desc_t) * task->cell_num));
    CHECK_CUDA_ERR(cudaMalloc(&task->gpu_task_info, sizeof(cumac_task_info_t)));

    // Allocate GPU memory for each cell: size = configs.get_max_data_size() for each cell
    CHECK_CUDA_ERR(cudaMalloc(&task->cells_buf, configs.get_max_data_size() * task->cell_num));
    CHECK_CUDA_ERR(cudaMalloc(&task->group_buf, group_buf_size));

    // One handler-owned muMimoUserPairing + GPU SRS/chan-orth/cuBB buffers (from slot-0 TV); all tasks share it.
    if (mu_ue_pair_gpu_ != nullptr)
    {
        task->muMimoUserPairingGpu = mu_ue_pair_gpu_;
        task->muUeGrpSol = reinterpret_cast<cumac_muUeGrp_resp_info_t*>(ue_pair_task_out_);
        task->max_num_srs_info = num_srs_info_;
    }

    for (int cell_id = 0; cell_id < task->cell_num; cell_id++)
    {
        task->tti_reqs[cell_id].reset();

        cell_desc_t *cell_desc = task->cpu_cell_descs + cell_id;
        cell_desc->home = task->cells_buf + configs.get_max_data_size() * cell_id;
    }

    // Set running in GPU or CPU
    if (configs.run_in_cpu == 1) // Force running in CPU
    {
        task->run_in_cpu = configs.run_in_cpu;
    }
    else if (configs.run_in_cpu == 2) // Run in GPU for even task_id, run in CPU for odd task_id
    {
        task->run_in_cpu = task->task_id & 0x1;
    }
    else // Default: Run in GPU
    {
        task->run_in_cpu = 0;
    }

    // Create CUDA stream for each task if multi-stream is enabled
    if (task->run_in_cpu == 0 && configs.multi_stream_enable)
    {
        CHECK_CUDA_ERR(cudaStreamCreate(&task->strm));
    }

    task->slot_concurrent_enable = configs.slot_concurrent_enable;

    if (group_tv.parsed)
    {
        task->tv = &group_tv;
    }

    // Group parameters
    struct cumac::cumacCellGrpPrms *grpPrms = &task->grpPrms;
    struct cumac::cumacCellGrpUeStatus *cellGrpUeStatus = &task->ueStatus;
    struct cumac::cumacSchdSol *schdSol = &task->schdSol;

    grpPrms->numUeSchdPerCellTTI = group_params.numUeSchdPerCellTTI;
    grpPrms->nUe = group_params.nUe;
    grpPrms->nActiveUe = group_params.nActiveUe; // nActiveUe can change per slot request
    grpPrms->nCell = group_params.nCell;

    grpPrms->nPrbGrp = group_params.nPrbGrp;
    grpPrms->nBsAnt = group_params.nBsAnt;
    grpPrms->nUeAnt = group_params.nUeAnt;
    grpPrms->W = group_params.W;

    grpPrms->sigmaSqrd = group_params.sigmaSqrd;
    grpPrms->Pt_Rbg = 79.4328 / group_params.nPrbGrp;
    grpPrms->Pt_rbgAnt = 79.4328 / group_params.nPrbGrp / group_params.nBsAnt; // 5.38e-43f
    grpPrms->precodingScheme = group_params.precodingScheme; // precoder type: 0 - no precoding, 1 - SVD precoding

    grpPrms->receiverScheme = group_params.receiverScheme; // receiver type: only support 1 - MMSE-IRC
    grpPrms->allocType = group_params.allocType;           // PRB allocation type: 0 - non-consecutive type 0 allocate, 1 - consecutive type 1 allocate
    grpPrms->betaCoeff = group_params.betaCoeff;           // coefficient for balancing cell-center and cell-edge UEs' performance in multi-cell scheduling. Default value is 1.0
    grpPrms->sinValThr = group_params.sinValThr;           // singular value threshold for layer selection, value is in (0, 1). Default value is 0.1

    grpPrms->corrThr = cell_configs[0].corrThr;               // channel vector correlation value threshold for layer selection,  value is in (0, 1). Default value is 0.5
    grpPrms->prioWeightStep = cell_configs[0].prioWeightStep; // step size for UE priority weight increment per TTI if UE does not get scheduled. Default is 100

    grpPrms->harqEnabledInd = group_params.harqEnabledInd;
    grpPrms->mcsSelCqi = group_params.mcsSelCqi;
    grpPrms->mcsSelSinrCapThr = group_params.mcsSelSinrCapThr;
    grpPrms->mcsSelLutType = group_params.mcsSelLutType;

    // Allocate buffers for 4T4R modules (UE_SEL, PRB_ALLOC, LAYER_SEL, MCS_SEL).
    // All sizes depend on nPrbGrp/nBsAnt/nUeAnt/nUe/nActiveUe from Cell_Configs; skip when not enabled.
    if ((moduleBitMask & CUMAC_CP_TASK_MASK_4T4R) != 0U)
    {
        // Alloc uint8_t** prgMsk pointer array in CPU memory
        CHECK_CUDA_ERR(cudaMallocHost((void **)&grpPrms->prgMsk, group_params.nCell * sizeof(uint8_t *)));
        for (int cIdx = 0; cIdx < group_params.nCell; cIdx++)
        {
            malloc_cumac_buf(task, &grpPrms->prgMsk[cIdx], &buf_num.prgMsk, group_params.nPrbGrp);
        }

        uint32_t prdLen = group_params.nUe * group_params.nPrbGrp * group_params.nBsAnt * group_params.nBsAnt;
        uint32_t detLen = group_params.nUe * group_params.nPrbGrp * group_params.nBsAnt * group_params.nBsAnt;
        uint32_t hLen = group_params.nPrbGrp * group_params.nUe * group_params.nCell * group_params.nBsAnt * group_params.nUeAnt;

        malloc_cumac_buf(task, &grpPrms->cellId, &buf_num.cellId, group_params.nCell);

        malloc_cumac_buf(task, &grpPrms->cellAssoc, &buf_num.cellAssoc, group_params.nCell * group_params.nUe);
        malloc_cumac_buf(task, &grpPrms->cellAssocActUe, &buf_num.cellAssocActUe, group_params.nCell * group_params.nActiveUe);

        CHECK_CUDA_ERR(cudaMemset(grpPrms->cellAssoc, 0, static_cast<size_t>(group_params.nCell) * group_params.nUe));
        CHECK_CUDA_ERR(cudaMemset(grpPrms->cellAssocActUe, 0, static_cast<size_t>(group_params.nCell) * group_params.nActiveUe));

        malloc_cumac_buf(task, &grpPrms->blerTargetActUe, &buf_num.blerTargetActUe, group_params.nActiveUe);
        CHECK_CUDA_ERR(cudaMemcpy(grpPrms->blerTargetActUe, blerTargetActUe, sizeof(float) * group_params.nActiveUe, cudaMemcpyHostToDevice));

        malloc_cumac_buf(task, &grpPrms->wbSinr, &buf_num.wbSinr, group_params.nActiveUe * group_params.nUeAnt);

        malloc_cumac_buf(task, &grpPrms->postEqSinr, &buf_num.postEqSinr, group_params.nActiveUe * group_params.nPrbGrp * group_params.nUeAnt);
        malloc_cumac_buf(task, &grpPrms->sinVal, &buf_num.sinVal, group_params.nUe * group_params.nPrbGrp * group_params.nUeAnt);

        malloc_cumac_buf(task, &grpPrms->prdMat, &buf_num.prdMat, prdLen);
        malloc_cumac_buf(task, &grpPrms->detMat, &buf_num.detMat, detLen);
        malloc_cumac_buf(task, &grpPrms->estH_fr, &buf_num.estH_fr, hLen);

        malloc_cumac_buf(task, &schdSol->setSchdUePerCellTTI, &buf_num.setSchdUePerCellTTI, group_params.nUe);
        if (group_params.allocType == 1) {
            malloc_cumac_buf(task, &schdSol->allocSol, &buf_num.allocSol, group_params.nUe * 2);

            uint32_t pfSize = group_params.nPrbGrp * group_params.numUeSchdPerCellTTI;
            uint32_t pow2N = 2;
            while (pow2N < pfSize)
            {
                pow2N = pow2N << 1;
            }
            malloc_cumac_buf(task, &schdSol->pfMetricArr, &buf_num.pfMetricArr, group_params.nCell * pow2N);
            malloc_cumac_buf(task, &schdSol->pfIdArr, &buf_num.pfIdArr, group_params.nCell * pow2N);
        } else {
            malloc_cumac_buf(task, &schdSol->allocSol, &buf_num.allocSol, group_params.nCell * group_params.nPrbGrp);
            schdSol->pfMetricArr = nullptr;
            schdSol->pfIdArr = nullptr;
        }

        malloc_cumac_buf(task, &schdSol->layerSelSol, &buf_num.layerSelSol, group_params.nUe);
        malloc_cumac_buf(task, &schdSol->mcsSelSol, &buf_num.mcsSelSol, group_params.nUe);

        malloc_cumac_buf(task, &cellGrpUeStatus->avgRatesActUe, &buf_num.avgRatesActUe, group_params.nActiveUe);
        malloc_cumac_buf(task, &cellGrpUeStatus->avgRates, &buf_num.avgRates, group_params.nUe);
        malloc_cumac_buf(task, &cellGrpUeStatus->newDataActUe, &buf_num.newDataActUe, group_params.nActiveUe);

        malloc_cumac_buf(task, &cellGrpUeStatus->tbErrLastActUe, &buf_num.tbErrLastActUe, group_params.nActiveUe);
        malloc_cumac_buf(task, &cellGrpUeStatus->tbErrLast, &buf_num.tbErrLast, group_params.nUe);

        // Init cellId static CUDA buffers
        for (uint16_t cell_id = 0; cell_id < grpPrms->nCell; cell_id++)
        {
            if (task->run_in_cpu)
            {
                *(grpPrms->cellId + cell_id) = cell_id;
            }
            else
            {
                CHECK_CUDA_ERR(cudaMemcpy(grpPrms->cellId + cell_id, &cell_id, sizeof(uint16_t), cudaMemcpyHostToDevice));
            }
        }

        // Alloc host-pinned memory for block copy, force in CPU memory
        malloc_cumac_buf(task, &task->input_avgRatesActUe, nullptr, buf_num.avgRatesActUe, 1);
        malloc_cumac_buf(task, &task->input_avgRates, nullptr, buf_num.avgRates, 1);
        malloc_cumac_buf(task, &task->input_tbErrLastActUe, nullptr, buf_num.tbErrLastActUe, 1);
        malloc_cumac_buf(task, &task->input_tbErrLast, nullptr, buf_num.tbErrLast, 1);
        malloc_cumac_buf(task, &task->input_estH_fr, nullptr, buf_num.estH_fr, 1);

        malloc_cumac_buf(task, &task->output_setSchdUePerCellTTI, nullptr, buf_num.setSchdUePerCellTTI, 1);
        malloc_cumac_buf(task, &task->output_allocSol, nullptr, buf_num.allocSol, 1);
        malloc_cumac_buf(task, &task->output_layerSelSol, nullptr, buf_num.layerSelSol, 1);
        malloc_cumac_buf(task, &task->output_mcsSelSol, nullptr, buf_num.mcsSelSol, 1);
    }

    // Allocate memory for pfmSort
    if ((moduleBitMask & CUMAC_CP_TASK_MASK_PFM_SORT) != 0U)
    {
        malloc_cumac_buf(task, &task->pfmCellInfo, &buf_num.pfmCellInfo, group_params.nCell);
        malloc_cumac_buf(task, &task->output_pfmSortSol, &buf_num.pfmSortSol, group_params.nCell, 1);
    }

    // Allocate memory for muUeGrp (MU-MIMO UE grouping); per-cell stride from enable_gpu_share
    if ((moduleBitMask & CUMAC_CP_TASK_MASK_MU_UE_GRP) != 0U)
    {
        const uint32_t ue_grp_info_size = static_cast<uint32_t>(cumac_muUeGrp_req_info_size(configs.enable_gpu_share));
        malloc_cumac_buf(task, reinterpret_cast<uint8_t**>(&task->muUeGrpInfo), &buf_num.muUeGrpInfo, group_params.nCell * ue_grp_info_size);
        malloc_cumac_buf(task, &task->output_muUeGrpSol, &buf_num.muUeGrpSol, group_params.nCell, 1);
    }

    CHECK_CUDA_ERR(cudaMemcpy(&task->gpu_task_info->grpPrms, &task->grpPrms, sizeof(struct cumac::cumacCellGrpPrms), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERR(cudaMemcpy(&task->gpu_task_info->ueStatus, &task->ueStatus, sizeof(struct cumac::cumacCellGrpUeStatus), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERR(cudaMemcpy(&task->gpu_task_info->schdSol, &task->schdSol, sizeof(struct cumac::cumacSchdSol), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERR(cudaMemcpy(&task->gpu_task_info->data_num, &buf_num, sizeof(cumac_buf_num_t), cudaMemcpyHostToDevice));

    // Copy pfmCellInfo and muUeGrpReqInfo GPU buffer pointers to gpu_task_info
    CHECK_CUDA_ERR(cudaMemcpy(&task->gpu_task_info->pfmCellInfo, &task->pfmCellInfo, sizeof(cumac_pfm_cell_info_t*), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERR(cudaMemcpy(&task->gpu_task_info->muUeGrpInfo, &task->muUeGrpInfo, sizeof(uint8_t*), cudaMemcpyHostToDevice));

    if (task->group_buf_enabled && (moduleBitMask & CUMAC_CP_TASK_MASK_4T4R) != 0U)
    {
        uint8_t**  tmp_prgMsk_array = nullptr;
        uint32_t tmp_size = group_params.nCell * sizeof(uint8_t *);
        malloc_cumac_buf(task, &tmp_prgMsk_array, nullptr, group_params.nCell);
        CHECK_CUDA_ERR(cudaMemcpy(&task->gpu_task_info->grpPrms.prgMsk, &tmp_prgMsk_array, sizeof(uint8_t**), cudaMemcpyHostToDevice));
        CHECK_CUDA_ERR(cudaMemcpy(tmp_prgMsk_array, grpPrms->prgMsk, tmp_size, cudaMemcpyHostToDevice));
    }

    task->module_bitmask = moduleBitMask;
    task->init_cumac_modules();

    NVLOGI_FMT(TAG, "{}: group_buf_size={} group_buf_offset={} group_buf_enabled={}", __func__, group_buf_size, task->group_buf_offset, task->group_buf_enabled);
    return 0;
}

void cumac_cp_handler::set_task_ring(nv::lock_free_ring_pool<cumac_task>* ring, sem_t* sem) {
    task_ring = ring;
    task_sem = sem;
}

void cumac_cp_handler::on_config_request(nv_ipc_msg_t& msg) {

    if (msg.msg_buf == nullptr || msg.cell_id < 0 || msg.cell_id >= static_cast<int>(cell_configs.size())
            || msg.msg_len < static_cast<int>(sizeof(cumac_config_req_t)))
    {
        NVLOGE_FMT(TAG, AERIAL_TEST_CUMAC_EVENT, "{}: invalid CONFIG.req cell_id={} msg_len={} cell_configs={}",
                __func__, msg.cell_id, msg.msg_len, cell_configs.size());
        return;
    }

    cumac_config_req_t* req = reinterpret_cast<cumac_config_req_t*>(msg.msg_buf);

    if (req->header.body_len < sizeof(cumac_cell_configs_t))
    {
        NVLOGE_FMT(TAG, AERIAL_TEST_CUMAC_EVENT, "{}: invalid CONFIG.req body_len={}", __func__, req->header.body_len);
        return;
    }

    cumac_cell_configs_t& cfg = cell_configs[msg.cell_id];

    // Copy cell configs first so we have access to the config data
    cfg = *reinterpret_cast<cumac_cell_configs_t*>(req->body);
    if (cfg.nMaxCell == 0 || cfg.nMaxCell > configs.cell_num)
    {
        NVLOGE_FMT(TAG, AERIAL_TEST_CUMAC_EVENT, "{}: invalid nMaxCell={} configured_cell_num={}",
                __func__, cfg.nMaxCell, configs.cell_num);
        return;
    }

    moduleBitMask |= cfg.moduleBitMask;

    if (configured_cell_num == 0) {
        // Allocate blerTargetActUe for all cells when first cell config is received
        const size_t blerTargetActUe_num = static_cast<size_t>(cfg.nMaxActUePerCell) * static_cast<size_t>(cfg.nMaxCell);
        blerTargetActUe = reinterpret_cast<float*>(malloc(sizeof(float) * blerTargetActUe_num));
        if (blerTargetActUe == nullptr) {
            NVLOGE_FMT(TAG, AERIAL_CUMAC_CP_EVENT, "{}: malloc failed for blerTargetActUe, num={}", __func__, blerTargetActUe_num);
            return;
        }
        for (size_t i = 0; i < blerTargetActUe_num; i++) {
            blerTargetActUe[i] = cfg.blerTarget;
        }
    }

    NVLOGC_FMT(TAG, "{}: cell_id={} nMaxCell={} nMaxPrg={} nPrbPerPrg={} nMaxBsAnt={} nMaxUeAnt={} harqEnabledInd={}",
            __func__, msg.cell_id, cfg.nMaxCell, cfg.nMaxPrg, cfg.nPrbPerPrg, cfg.nMaxBsAnt, cfg.nMaxUeAnt, cfg.harqEnabledInd);

    // TODO: handle re-config
    configured_cell_num++;
    if (configured_cell_num == configs.cell_num) {
        check_config_params();
    }

    // Send response after config
    nv::phy_mac_transport& transp = transport(msg.cell_id);
    nv::phy_mac_msg_desc msg_desc;
    if (transp.tx_alloc(msg_desc) < 0)
    {
        return;
    }

    auto resp = cumac_init_msg_header<cumac_config_resp_t>(&msg_desc, CUMAC_CONFIG_RESPONSE, msg.cell_id);
    resp->error_code = 0;

    NVLOGI_FMT(TAG, "SEND: cell_id={} msg_id=0x{:02X} {}", msg.cell_id, msg_desc.msg_id, get_cumac_msg_name(msg_desc.msg_id));

    transp.tx_send(msg_desc);
    transp.tx_post();
}

void cumac_cp_handler::on_start_request(nv_ipc_msg_t& msg)
 {
    nv::phy_mac_transport& transp = transport(msg.cell_id);
    nv::phy_mac_msg_desc msg_desc;
    if (transp.tx_alloc(msg_desc) < 0)
    {
        return;
    }

    auto resp = cumac_init_msg_header<cumac_start_resp_t>(&msg_desc, CUMAC_START_RESPONSE, msg.cell_id);
    resp->error_code = 0;

    NVLOGI_FMT(TAG, "SEND: cell_id={} msg_id=0x{:02X} {}", msg.cell_id, msg_desc.msg_id, get_cumac_msg_name(msg_desc.msg_id));

    transp.tx_send(msg_desc);
    transp.tx_post();
}

void cumac_cp_handler::on_stop_request(nv_ipc_msg_t& msg)
{
    nv::phy_mac_transport& transp = transport(msg.cell_id);
    nv::phy_mac_msg_desc msg_desc;
    if (transp.tx_alloc(msg_desc) < 0)
    {
        return;
    }

    auto resp = cumac_init_msg_header<cumac_stop_resp_t>(&msg_desc, CUMAC_STOP_RESPONSE, msg.cell_id);
    resp->error_code = 0;

    NVLOGI_FMT(TAG, "SEND: cell_id={} msg_id=0x{:02X} {}", msg.cell_id, msg_desc.msg_id, get_cumac_msg_name(msg_desc.msg_id));

    transp.tx_send(msg_desc);
    transp.tx_post();
}

template <typename T>
int copy_from_ipc_buf(nv::phy_mac_transport &transp, nv_ipc_msg_t &msg, cumac_task *task, T *dst_buf, const char *info, uint32_t &src_offset_in_bytes, uint32_t &dst_offset_in_num, uint32_t num)
{
    if (task->run_in_cpu)
    {
        transp.copy_from_data_buf(msg, src_offset_in_bytes, dst_buf + dst_offset_in_num, num * sizeof(T));
    }
    else if (task->group_buf_enabled == 0)
    {
        uint8_t *src = reinterpret_cast<uint8_t *>(msg.data_buf);
        CHECK_CUDA_ERR(cudaMemcpyAsync(dst_buf + dst_offset_in_num, src + src_offset_in_bytes, num * sizeof(T), cudaMemcpyHostToDevice, task->strm));
    }
    dst_offset_in_num += num;
    return 0;
}

void cumac_cp_handler::cell_copy_task(nv_ipc_msg_t &msg, cumac_task *task)
{
    cumac_sch_tti_req_t &head = *reinterpret_cast<cumac_sch_tti_req_t *>(msg.msg_buf);
    cumac_tti_req_payload_t &req = head.payload;
    uint8_t *src = reinterpret_cast<uint8_t *>(msg.data_buf);
    uint32_t nBlock = group_params.nUe * req.nBsAnt * req.nUeAnt;

    struct cumacCellGrpPrms &grpPrms = task->grpPrms;

    if (req.taskBitMask & (0x1 << CUMAC_TASK_PRB_ALLOCATION)) // multiCellScheduler buffers
    {
        // task->data_num.estH_fr += req.nPrbGrp * nBlock;

        // if (task->tv != nullptr && task->debug_option & DBG_OPT_WAR_COPY_GROUP_TV)
        // {
        // return;
        // }

        // copy_from_ipc_buf(transp, msg, task, grpPrms->estH_fr, "estH_fr", req.offsets.estH_fr, task->data_num.estH_fr, hLen);
        cuComplex(*dst_estH_fr)[grpPrms.nPrbGrp][grpPrms.nCell][grpPrms.nUe][grpPrms.nBsAnt][grpPrms.nUeAnt] = reinterpret_cast<cuComplex(*)[grpPrms.nPrbGrp][grpPrms.nCell][grpPrms.nUe][grpPrms.nBsAnt][grpPrms.nUeAnt]>(task->input_estH_fr);
        cuComplex(*src_estH_fr)[grpPrms.nPrbGrp][grpPrms.nUe][grpPrms.nBsAnt][grpPrms.nUeAnt] = reinterpret_cast<cuComplex(*)[grpPrms.nPrbGrp][grpPrms.nUe][grpPrms.nBsAnt][grpPrms.nUeAnt]>(src + req.offsets.estH_fr);
        for (int prgId = 0; prgId < req.nPrbGrp; prgId++)
        {
            memcpy((*dst_estH_fr)[prgId][msg.cell_id], (*src_estH_fr)[prgId], nBlock * sizeof(cuComplex));
        }

        // cuComplex *base_estH_fr = reinterpret_cast<cuComplex *>(src + req.offsets.estH_fr);
        // for (int prgId = 0; prgId < req.nPrbGrp; prgId++)
        // {
        //     int indexGroup = prgId * group_params.nCell * nBlock + msg.cell_id * nBlock;
        //     int indexCell = prgId * nBlock;
        //     memcpy(task->input_estH_fr + indexGroup, base_estH_fr + indexCell, nBlock * sizeof(cuComplex));
        // }
    }
}

// This function is called in cell_id order
void cumac_cp_handler::on_sch_tti_request(nv_ipc_msg_t& msg, cumac_task* task)
{
    uint16_t cell_id = msg.cell_id;
    sfn_slot_t ss_msg = nv_ipc_get_sfn_slot(&msg);

    if (task == nullptr)
    {
        NVLOGE_FMT(TAG, AERIAL_CUMAC_CP_EVENT, "SFN {}.{} cell_id={} task == nullptr", ss_msg.u16.sfn, ss_msg.u16.slot,
                cell_id);
        return;
    }

    if (msg.msg_buf == nullptr || msg.msg_len < static_cast<int>(sizeof(cumac_sch_tti_req_t)))
    {
        NVLOGE_FMT(TAG, AERIAL_CUMAC_CP_EVENT, "{}: invalid SCH_TTI.req cell_id={} msg_len={}", __func__, cell_id,
                msg.msg_len);
        return;
    }

    if (cell_id >= group_params.nCell || cell_id >= cell_configs.size())
    {
        NVLOGE_FMT(TAG, AERIAL_CUMAC_CP_EVENT, "{}: invalid SCH_TTI.req cell_id={} nCell={} cell_configs={}",
                __func__, cell_id, group_params.nCell, cell_configs.size());
        return;
    }

    nv::phy_mac_transport& transp = transport(cell_id);
    cumac_cell_configs_t& cfg = cell_configs[cell_id];

    cumac_sch_tti_req_t& head = *reinterpret_cast<cumac_sch_tti_req_t*>(msg.msg_buf);
    cumac_tti_req_payload_t& req = head.payload;

    // 4T4R path allocates buffers sized from these maxima; zero values skip the
    // checks below and then fail (or leave null pointers) in malloc_cumac_buf.
    if ((req.taskBitMask & CUMAC_CP_TASK_MASK_4T4R) != 0U &&
        (cfg.nMaxActUePerCell == 0 || cfg.nMaxBsAnt == 0 || cfg.nMaxUeAnt == 0 || group_params.nPrbGrp == 0))
    {
        NVLOGE_FMT(TAG, AERIAL_CUMAC_CP_EVENT,
                   "{}: SFN {}.{} cell_id={} invalid zero maxima for enabled 4T4R: nMaxActUePerCell={} nMaxBsAnt={} nMaxUeAnt={} nPrbGrp={}",
                   __func__, head.sfn, head.slot, cell_id, cfg.nMaxActUePerCell, cfg.nMaxBsAnt, cfg.nMaxUeAnt,
                   group_params.nPrbGrp);
        return;
    }

    if (cfg.nMaxActUePerCell > 0 && req.nActiveUe > cfg.nMaxActUePerCell)
    {
        NVLOGE_FMT(TAG, AERIAL_CUMAC_CP_EVENT,
                "{}: SFN {}.{} cell_id={} nActiveUe={} exceeds nMaxActUePerCell={}", __func__, head.sfn, head.slot,
                cell_id, req.nActiveUe, cfg.nMaxActUePerCell);
        return;
    }

    if ((cfg.nMaxBsAnt > 0 && req.nBsAnt > cfg.nMaxBsAnt) ||
        (cfg.nMaxUeAnt > 0 && req.nUeAnt > cfg.nMaxUeAnt) ||
        (group_params.nPrbGrp > 0 && req.nPrbGrp > group_params.nPrbGrp))
    {
        NVLOGE_FMT(TAG, AERIAL_CUMAC_CP_EVENT,
                "{}: SFN {}.{} cell_id={} invalid dims nBsAnt={}/{} nUeAnt={}/{} nPrbGrp={}/{}", __func__, head.sfn,
                head.slot, cell_id, req.nBsAnt, cfg.nMaxBsAnt, req.nUeAnt, cfg.nMaxUeAnt, req.nPrbGrp,
                group_params.nPrbGrp);
        return;
    }

    task->taskBitMask = req.taskBitMask;

    task->grpPrms.sigmaSqrd = req.sigmaSqrd; // Use request value; no longer hardcoded to 1.0
    task->grpPrms.nActiveUe += req.nActiveUe;

    struct cumacCellGrpPrms* grpPrms = &task->grpPrms;
    struct cumacCellGrpUeStatus* ueStatus = &task->ueStatus;
    struct cumacSchdSol* schdSol = &task->schdSol;

    CHECK_VALUE_MAX_ERR(cell_id, group_params.nCell - 1);

    if ((req.taskBitMask & CUMAC_CP_TASK_MASK_4T4R) != 0U) // Check common parameters for the four 4T4R modules
    {
        // Below parameter validations are for debugging
        CHECK_VALUE_EQUAL_ERR(cell_id, req.cellID);
        CHECK_VALUE_MAX_ERR(grpPrms->numUeSchdPerCellTTI, cfg.nMaxSchUePerCell);
        CHECK_VALUE_MAX_ERR(grpPrms->nUe, grpPrms->numUeSchdPerCellTTI * grpPrms->nCell);
        // CHECK_VALUE_EQUAL_ERR(grpPrms->nActiveUe, req.nActiveUe * grpPrms->nCell);
        CHECK_VALUE_MAX_ERR(req.nActiveUe, cfg.nMaxActUePerCell);
        CHECK_VALUE_EQUAL_ERR(grpPrms->nCell, configs.cell_num);

        CHECK_VALUE_MAX_ERR(grpPrms->nPrbGrp, group_params.nPrbGrp);
        CHECK_VALUE_MAX_ERR(grpPrms->nBsAnt, req.nBsAnt);
        CHECK_VALUE_MAX_ERR(grpPrms->nUeAnt, req.nUeAnt);
        CHECK_VALUE_EQUAL_ERR(grpPrms->W, group_params.W);

        // CHECK_VALUE_EQUAL_ERR(grpPrms->sigmaSqrd, req.sigmaSqrd);
        // CHECK_VALUE_EQUAL_ERR(grpPrms->Pt_Rbg, 0);
        // CHECK_VALUE_EQUAL_ERR(grpPrms->Pt_rbgAnt, 0);
        CHECK_VALUE_EQUAL_ERR(grpPrms->precodingScheme, cfg.precoderType);

        CHECK_VALUE_EQUAL_ERR(grpPrms->receiverScheme, cfg.receiverType);
        CHECK_VALUE_EQUAL_ERR(grpPrms->allocType, cfg.allocType);
        CHECK_VALUE_EQUAL_ERR(grpPrms->betaCoeff, cfg.betaCoeff);
        CHECK_VALUE_EQUAL_ERR(grpPrms->sinValThr, cfg.sinValThr);

        CHECK_VALUE_EQUAL_ERR(grpPrms->corrThr, cfg.corrThr);
        CHECK_VALUE_EQUAL_ERR(grpPrms->prioWeightStep, cfg.prioWeightStep);

        CHECK_PTR_NULL_FATAL(msg.data_buf);
        CHECK_PTR_NULL_FATAL(grpPrms->wbSinr);
        CHECK_PTR_NULL_FATAL(ueStatus->avgRatesActUe);
        CHECK_PTR_NULL_FATAL(grpPrms->prgMsk[cell_id]);
    }

    // Copy data buffers
    uint8_t *src = reinterpret_cast<uint8_t *>(msg.data_buf);

    if (req.taskBitMask & (0x1 << CUMAC_TASK_UE_SELECTION)) // multiCellUeSelection buffers
    {
        if (task->run_in_cpu)
        {
            // Populate cellAssocActUe buffer: (Assume each cell has the same nActiveUe)
            // ue_id_offset = req.nActiveUe * cell_id
            // cell_offset = group_params.nActiveUe * cell_id
            // home_offset_for_cell_cellAssocActUe = cell_offset + ue_id_offset = (group_params.nActiveUe + req.nActiveUe ) * cell_id
            uint32_t total_active_ue = group_params.nCell * req.nActiveUe;
            memset(grpPrms->cellAssocActUe + (total_active_ue + req.nActiveUe) * cell_id, 1, req.nActiveUe);
            memset(grpPrms->cellAssoc + (group_params.numUeSchdPerCellTTI * group_params.nCell + group_params.numUeSchdPerCellTTI) * cell_id, 1, group_params.numUeSchdPerCellTTI);
        }
        else if (task->group_buf_enabled == 0)
        {
            // Populate cellAssocActUe buffer: (Assume each cell has the same nActiveUe)
            // ue_id_offset = req.nActiveUe * cell_id
            // cell_offset = group_params.nActiveUe * cell_id
            // home_offset_for_cell_cellAssocActUe = cell_offset + ue_id_offset = (group_params.nActiveUe + req.nActiveUe ) * cell_id
            uint32_t total_active_ue = group_params.nCell * req.nActiveUe;
            CHECK_CUDA_ERR(cudaMemsetAsync(grpPrms->cellAssocActUe + (total_active_ue + req.nActiveUe) * cell_id, 1, req.nActiveUe, task->strm));
            CHECK_CUDA_ERR(cudaMemsetAsync(grpPrms->cellAssoc + (group_params.numUeSchdPerCellTTI * group_params.nCell + group_params.numUeSchdPerCellTTI) * cell_id, 1, group_params.numUeSchdPerCellTTI, task->strm));
        }

        task->data_num.cellId += 1;
        task->data_num.cellAssocActUe += group_params.nCell * req.nActiveUe; // Actual count this TTI, not configured max
        task->data_num.cellAssoc += group_params.nCell * group_params.numUeSchdPerCellTTI;

        uint32_t prgMsk_num = 0;
        copy_from_ipc_buf(transp, msg, task, grpPrms->prgMsk[cell_id], "prgMsk", req.offsets.prgMsk, prgMsk_num, req.nPrbGrp);
        task->data_num.prgMsk = prgMsk_num;

        copy_from_ipc_buf(transp, msg, task, grpPrms->wbSinr, "wbSinr", req.offsets.wbSinr, task->data_num.wbSinr, req.nActiveUe * req.nUeAnt);

        // Copy to a contiguous CPU buffer for later selection
        memcpy(task->input_avgRatesActUe + task->data_num.avgRatesActUe, src + req.offsets.avgRatesActUe, req.nActiveUe * sizeof(float));
        copy_from_ipc_buf(transp, msg, task, ueStatus->avgRatesActUe, "avgRatesActUe", req.offsets.avgRatesActUe, task->data_num.avgRatesActUe, req.nActiveUe);

        if (task->debug_option & DBG_OPT_PRINT_NVIPC_BUF) // Dump IPC buffers
        {
            NVLOGI_FMT_ARRAY(TAG, "NVIPC_avgRatesActUe", reinterpret_cast<float*>(src + req.offsets.avgRatesActUe), req.nActiveUe);
        }
    }

    if (req.taskBitMask & (0x1 << CUMAC_TASK_PRB_ALLOCATION)) // multiCellScheduler buffers
    {
        // estH_fr data_num
        task->data_num.estH_fr += req.nPrbGrp * group_params.nUe * req.nBsAnt * req.nUeAnt;

        copy_from_ipc_buf(transp, msg, task, grpPrms->postEqSinr, "postEqSinr", req.offsets.postEqSinr, task->data_num.postEqSinr, req.nActiveUe * req.nPrbGrp * req.nUeAnt);
        copy_from_ipc_buf(transp, msg, task, grpPrms->sinVal, "sinVal", req.offsets.sinVal, task->data_num.sinVal, group_params.numUeSchdPerCellTTI * req.nPrbGrp * req.nUeAnt);

        uint32_t prdLen = group_params.numUeSchdPerCellTTI * req.nPrbGrp * req.nBsAnt * req.nBsAnt;
        uint32_t detLen = group_params.numUeSchdPerCellTTI * req.nPrbGrp * req.nUeAnt * req.nUeAnt;

        copy_from_ipc_buf(transp, msg, task, grpPrms->detMat, "detMat", req.offsets.detMat, task->data_num.detMat, detLen);
        copy_from_ipc_buf(transp, msg, task, grpPrms->prdMat, "prdMat", req.offsets.prdMat, task->data_num.prdMat, prdLen);
    }

    if (req.taskBitMask & (0x1 << CUMAC_TASK_MCS_SELECTION)) // mcsSelectionLUT buffers
    {
        // Copy to a contiguous CPU buffer for later selection
        memcpy(task->input_tbErrLastActUe + task->data_num.tbErrLastActUe, src + req.offsets.tbErrLastActUe, req.nActiveUe * sizeof(int8_t));
        copy_from_ipc_buf(transp, msg, task, ueStatus->tbErrLastActUe, "tbErrLastActUe", req.offsets.tbErrLastActUe, task->data_num.tbErrLastActUe, req.nActiveUe);
    }

    // Handle pfmSort task setup
    if (req.taskBitMask & (0x1 << CUMAC_TASK_PFM_SORT))
    {
        if (task->debug_option & DBG_OPT_PRINT_NVIPC_BUF) // Dump IPC buffers
        {
            uint8_t* pfmCellInfo_buf = reinterpret_cast<uint8_t*>(msg.data_buf) + req.offsets.pfmCellInfo;
            NVLOGI_FMT_ARRAY(TAG, "NVIPC_pfmCellInfo", pfmCellInfo_buf, sizeof(cumac_pfm_cell_info_t));
        }
        copy_from_ipc_buf(transp, msg, task, task->pfmCellInfo, "pfmCellInfo", req.offsets.pfmCellInfo, task->data_num.pfmCellInfo, 1);
    }

    // Handle muUeGrp: per-cell copy size matches cumac_muUeGrp_req_info_size(enable_gpu_share).
    if (req.taskBitMask & (0x1 << CUMAC_TASK_MU_UE_GRP))
    {
        const uint32_t nbytes = static_cast<uint32_t>(cumac_muUeGrp_req_info_size(configs.enable_gpu_share));
        copy_from_ipc_buf<uint8_t>(transp, msg, task, reinterpret_cast<uint8_t*>(task->muUeGrpInfo),
            "muUeGrpInfo", req.offsets.muUeGrpInfo, task->data_num.muUeGrpInfo, nbytes);

        if (task->debug_option & DBG_OPT_PRINT_NVIPC_BUF) // Dump IPC buffers
        {
            uint8_t* muUeGrpInfo_buf = src + req.offsets.muUeGrpInfo;
            NVLOGI_FMT_ARRAY(TAG, "NVIPC_muUeGrpInfo", muUeGrpInfo_buf, nbytes);
        }
        cumac_muUeGrp_req_info_t* muUeGrpInfo = reinterpret_cast<cumac_muUeGrp_req_info_t*>(src + req.offsets.muUeGrpInfo);
        task->max_num_srs_info = std::max(task->max_num_srs_info, muUeGrpInfo->numSrsInfo);
        NVLOGI_FMT(TAG, "SFN {}.{} RECV: SCH_TTI.req cell_id={} num_srs_info={}", head.sfn, head.slot, cell_id, muUeGrpInfo->numSrsInfo);
    }

    if (task->group_buf_enabled) // Copy the whole msg.data_buf to the GPU group buffer
    {
        cell_desc_t *cell_desc = task->cpu_cell_descs + cell_id;
        CHECK_CUDA_ERR(cudaMemcpyAsync(cell_desc->home, src, msg.data_len, cudaMemcpyHostToDevice, task->strm));
        memcpy(&cell_desc->offsets, &req.offsets, sizeof(cumac_tti_req_buf_offsets_t));
    }

    NVLOGI_FMT(TAG, "SFN {}.{} RECV: SCH_TTI.req cell_id={} taskBitMask=0x{:X} cellID={} ULDLSch={} nActiveUe={} nBsAnt={} nUeAnt={} group: nCell={} nActiveUe={} nUe={}",
        head.sfn, head.slot, cell_id, req.taskBitMask, req.cellID, req.ULDLSch, req.nActiveUe, req.nBsAnt, req.nUeAnt, grpPrms->nCell, grpPrms->nActiveUe, grpPrms->nUe);
}

void cumac_cp_handler::print_cumac_cp_thrput(uint64_t slot_counter)
{
    for(int cell_id = 0; cell_id < configs.cell_num; cell_id++)
    {
        cumac_cp_thrput_t& thrput = thrputs[cell_id];
        // Console log print per second
        NVLOGC_FMT(TAG, "Cell {:2} | CUMAC {:4} | ERR {:4} | Slots {}",
                   cell_id,
                   thrput.cumac_slots.load(),
                   thrput.error.load(),
                   slot_counter);

        thrput.reset();
    }
}

void cumac_cp_handler::push_cumac_task(sfn_slot_t ss)
{
    cumac_task* task = get_cumac_task(ss);
    struct cumac::cumacCellGrpPrms& grpPrms = task->grpPrms;

    NVLOGI_FMT(TAG, "SFN {}.{} PUSH_TASK: 0x{:X} nUe={} nActiveUe={} numUeSchdPerCellTTI={} nCell={} nPrbGrp={} nBsAnt={} nUeAnt={} precodingScheme={} receiverScheme={} allocType={} prioWeightStep={}", ss.u16.sfn, ss.u16.slot,
               task->taskBitMask, grpPrms.nUe, grpPrms.nActiveUe, grpPrms.numUeSchdPerCellTTI, grpPrms.nCell, grpPrms.nPrbGrp, grpPrms.nBsAnt, grpPrms.nUeAnt, grpPrms.precodingScheme, grpPrms.receiverScheme, grpPrms.allocType, grpPrms.prioWeightStep);

    // Add buffer size check
    task->calculate_output_data_num();
    check_task_buf_size(task);

    task->ts_enqueue = std::chrono::system_clock::now().time_since_epoch().count();
    task_order_sem_.assign_for_enqueue(*task);
    task_ring->enqueue(task);
    sem_post(task_sem);
}

int cumac_cp_handler::send_sch_tti_response(cumac_task *task, int cell_id)
{
    size_t offset = 0;

    nv::phy_mac_transport& transp = transport(cell_id);

    nv::phy_mac_msg_desc msg;
    msg.data_pool = NV_IPC_MEMPOOL_CPU_DATA;

    cumac_cell_configs_t &cell_cfg = cell_configs[cell_id];
    uint32_t task_mask = task->taskBitMask;

    if (transp.tx_alloc(msg) < 0)
    {
        return -1;
    }

    // Get the request message header and payload
    nv::phy_mac_msg_desc& req_msg = task->tti_reqs[cell_id];
    cumac_sch_tti_req_t& req_head = *reinterpret_cast<cumac_sch_tti_req_t*>(req_msg.msg_buf);
    cumac_tti_req_payload_t& req = req_head.payload;

    cumac_sch_tti_resp_t& resp = *cumac_init_msg_header<cumac_sch_tti_resp_t>(&msg, CUMAC_SCH_TTI_RESPONSE, cell_id);
    resp.sfn = task->ss.u16.sfn;
    resp.slot = task->ss.u16.slot;
    resp.taskBitMask = task_mask;

    // Per-cell slice length in setSchdUePerCellTTI / allocSol / layer / MCS buffers (nCell * nMaxSch from CONFIG)
    const size_t nMaxSchUePerCell = static_cast<size_t>(task->grpPrms.numUeSchdPerCellTTI);

    if ((task_mask & (0x1u << CUMAC_TASK_UE_SELECTION)) && task->output_setSchdUePerCellTTI != nullptr)
    {
        const uint16_t* row = task->output_setSchdUePerCellTTI + static_cast<size_t>(cell_id) * nMaxSchUePerCell;
        uint32_t nActual = 0;
        for (uint32_t i = 0; i < nMaxSchUePerCell; i++)
        {
            if (row[i] != 0xFFFFu)
            {
                nActual++;
            }
        }
        resp.nUeSchd = static_cast<uint16_t>(nActual);
    }
    else
    {
        // UE selection not run or no CPU copy: keep prior max-slots semantics for L2
        resp.nUeSchd = static_cast<uint16_t>(nMaxSchUePerCell);
    }

    memset(&resp.offsets, INVALID_CUMAC_BUF_OFFSET, sizeof(resp.offsets));

    uint32_t ipc_offset = 0;

    if (task_mask & (0x1 << CUMAC_TASK_UE_SELECTION)) // UE_SELECTION result
    {
        size_t setSchdUePerCellTTI_offset = nMaxSchUePerCell * cell_id;
        transp.copy_to_data_buf(msg, ipc_offset, task->output_setSchdUePerCellTTI + setSchdUePerCellTTI_offset, sizeof(*task->output_setSchdUePerCellTTI) * resp.nUeSchd);
        // Map L1 group ue_id to L2 per cell ue_id
        uint16_t *ue_id_buf = reinterpret_cast<uint16_t *>(reinterpret_cast<uint8_t *>(msg.data_buf) + ipc_offset);
        uint32_t ue_id_base = cell_id * task->grpPrms.nActiveUe / task->grpPrms.nCell;
        for (uint32_t i = 0; i < resp.nUeSchd; i++)
        {
            if (ue_id_buf[i] != 0xFFFFu)
            {
                ue_id_buf[i] = static_cast<uint16_t>(ue_id_buf[i] - ue_id_base);
            }
        }
        resp.offsets.setSchdUePerCellTTI = ipc_offset;
        ipc_offset += resp.nUeSchd * sizeof(*task->output_setSchdUePerCellTTI);
        if (task->debug_option & DBG_OPT_PRINT_NVIPC_BUF) // Dump IPC buffers
        {
            NVLOGI_FMT_ARRAY(TAG, "ARRAY setSchdUePerCellTTI", ue_id_buf, resp.nUeSchd);
        }
    }

    if (task_mask & (0x1 << CUMAC_TASK_PRB_ALLOCATION)) // PRB_ALLOCATION result
    {
        uint32_t allocSol_num = cell_cfg.allocType == 1 ? 2 * resp.nUeSchd : task->grpPrms.nPrbGrp;
        size_t allocSol_offset = cell_cfg.allocType == 1 ? 2 * static_cast<size_t>(nMaxSchUePerCell) * cell_id : static_cast<size_t>(task->grpPrms.nPrbGrp) * cell_id;
        transp.copy_to_data_buf(msg, ipc_offset, task->output_allocSol + allocSol_offset, sizeof(*task->output_allocSol) * allocSol_num);
        resp.offsets.allocSol = ipc_offset;
        ipc_offset += allocSol_num * sizeof(*task->output_allocSol);
    }

    if (task_mask & (0x1 << CUMAC_TASK_LAYER_SELECTION)) // LAYER_SELECTION result
    {
        size_t layerSelSol_offset = static_cast<size_t>(nMaxSchUePerCell) * cell_id;
        transp.copy_to_data_buf(msg, ipc_offset, task->output_layerSelSol + layerSelSol_offset, sizeof(*task->output_layerSelSol) * resp.nUeSchd);
        resp.offsets.layerSelSol = ipc_offset;
        ipc_offset += resp.nUeSchd * sizeof(*task->output_layerSelSol);
    }

    if (task_mask & (0x1 << CUMAC_TASK_MCS_SELECTION)) // MCS_SELECTION result
    {
        size_t mcsSelSol_offset = static_cast<size_t>(nMaxSchUePerCell) * cell_id;
        transp.copy_to_data_buf(msg, ipc_offset, task->output_mcsSelSol + mcsSelSol_offset, sizeof(*task->output_mcsSelSol) * resp.nUeSchd);
        resp.offsets.mcsSelSol = ipc_offset;
        ipc_offset += resp.nUeSchd * sizeof(*task->output_mcsSelSol);
    }

    if (task_mask & (0x1 << CUMAC_TASK_PFM_SORT)) // PFM_SORT result
    {
        transp.copy_to_data_buf(msg, ipc_offset, task->output_pfmSortSol + cell_id, sizeof(*task->output_pfmSortSol));
        resp.offsets.pfmSortSol = ipc_offset;
        ipc_offset += sizeof(*task->output_pfmSortSol);
    }

    if (task_mask & (0x1 << CUMAC_TASK_MU_UE_GRP)) // muUeGrp result
    {
        transp.copy_to_data_buf(msg, ipc_offset, task->output_muUeGrpSol + cell_id, sizeof(*task->output_muUeGrpSol));
        resp.offsets.muUeGrpSol = ipc_offset;
        ipc_offset += sizeof(*task->output_muUeGrpSol);
    }

    msg.data_len = ipc_offset;

    NVLOGI_FMT(TAG, "SFN {}.{} SEND: SCH_TTI.resp cell_id={} msg_len={} data_len={} nActiveUe={} nMaxSchUePerCell={} nUeSchd={} allocSol_offset={} layerSelSol_offset={} mcsSelSol_offset={} setSchdUePerCellTTI_offset={}",
               resp.sfn, resp.slot, cell_id, msg.msg_len, msg.data_len, task->grpPrms.nActiveUe, nMaxSchUePerCell, resp.nUeSchd, resp.offsets.allocSol, resp.offsets.layerSelSol, resp.offsets.mcsSelSol, resp.offsets.setSchdUePerCellTTI);

    transp.tx_send(msg);
    transp.notify(1);

    thrputs[cell_id].cumac_slots ++;

    return 0;
}

sched_slot_data& cumac_cp_handler::get_sched_slot_data(sfn_slot_t ss) {
    return sched_slots[ss.u16.slot & 0x03];
}

void cumac_cp_handler::handle_slot_msg(nv::phy_mac_msg_desc &msg_desc, sfn_slot_t ss_msg)
{
    sfn_slot_t ss_curr = this->ss_curr.load();
    NVLOGI_FMT(TAG, "SFN {}.{} HANDLE: cell_id={} msg_id=0x{:02X} {} SFN {}.{}",
               ss_curr.u16.sfn, ss_curr.u16.slot, msg_desc.cell_id, msg_desc.msg_id, get_cumac_msg_name(msg_desc.msg_id), ss_msg.u16.sfn, ss_msg.u16.slot);

    nv::phy_mac_transport& transp = transport(msg_desc.cell_id);

    if (configured_cell_num < configs.cell_num)
    {
        NVLOGW_FMT(TAG, "SFN {}.{} cell_id={} msg_id=0x{:02X} {} skip before all cells configured",
            ss_msg.u16.sfn, ss_msg.u16.slot, msg_desc.cell_id, msg_desc.msg_id, get_cumac_msg_name(msg_desc.msg_id));
        transp.rx_release(msg_desc);
        return;
    }

    sched_slot_data &slot_data = get_sched_slot_data(ss_msg);
    if (ss_msg.u32 != slot_data.ss_sched.u32)
    {
        // This is the first message of a new slot, reset slot_data and cuda_task buffer
        NVLOGI_FMT(TAG, "SFN {}.{} received: SFN {}.{} cell_id={} msg_id=0x{:02X} {} for new slot", slot_data.ss_sched.u16.sfn, slot_data.ss_sched.u16.slot,
                   ss_msg.u16.sfn, ss_msg.u16.slot, msg_desc.cell_id, msg_desc.msg_id, get_cumac_msg_name(msg_desc.msg_id));
        global_tick ++;
        slot_data.reset_slot_data(ss_msg);
        if (slot_data.task != nullptr)
        {
            slot_data.task->ts_start = transp.get_ts_send(msg_desc);
        }
    }

    // Handle slot messages
    switch (msg_desc.msg_id)
    {
    case CUMAC_SCH_TTI_REQUEST:
        if (slot_data.task != nullptr)
        {
            // Do not release IPC buffer until async copy finished
            slot_data.task->tti_reqs[msg_desc.cell_id] = msg_desc;

            // Parse taskBitMask from SCH_TTI.req
            cumac_sch_tti_req_t& head = *reinterpret_cast<cumac_sch_tti_req_t*>(msg_desc.msg_buf);
            cumac_tti_req_payload_t& req = head.payload;
            slot_data.task->taskBitMask = req.taskBitMask;
        }
        else
        {
            // Drop the message if not assigned task buffer successfully
            transp.rx_release(msg_desc);
        }
        break;
    case CUMAC_TTI_END:
        // Start handling next cell after current cell ended
        slot_data.curr_cell_id++;
        if (slot_data.curr_cell_id >= slot_data.cell_num)
        {
            if (slot_data.task != nullptr)
            {
                // SLOT messages ended, create and push cumac_task into task queue
                slot_data.task->ts_last_send = transp.get_ts_send(msg_desc);
                slot_data.task->ts_last_recv = std::chrono::system_clock::now().time_since_epoch().count();
                push_cumac_task(ss_msg);
                slot_data.task = nullptr;
                slot_data.ss_sched = {.u32 = SFN_SLOT_INVALID};
            }

            // Print throughput every second
            uint64_t slot_index = ss_msg.u16.sfn * SLOT_NUM_PER_FRAME + ss_msg.u16.slot;
            if ((slot_index + SFN_SLOT_NUM_MAX - last_thrput_print_slot) % SFN_SLOT_NUM_MAX >= SLOTS_PER_SECOND)
            {
                print_cumac_cp_thrput(global_tick);
                last_thrput_print_slot = slot_index;
            }
        }
        transp.rx_release(msg_desc);
        break;
    default:
        break;
    }
}

void cumac_cp_handler::handle_slot_msg_reorder(nv::phy_mac_msg_desc& msg_desc, sfn_slot_t ss_msg)
{
    sfn_slot_t ss_curr = this->ss_curr.load();
    NVLOGI_FMT(TAG, "SFN {}.{} HANDLE_ORIGIN: cell_id={} msg_id=0x{:02X} {} SFN {}.{}",
        ss_curr.u16.sfn, ss_curr.u16.slot, msg_desc.cell_id, msg_desc.msg_id, get_cumac_msg_name(msg_desc.msg_id), ss_msg.u16.sfn, ss_msg.u16.slot);

    sched_slot_data& slot_data = get_sched_slot_data(ss_msg);
    if (ss_msg.u32 != slot_data.ss_sched.u32) {
        // This is the first message of a new slot, reset slot_data and cuda_task buffer
        NVLOGI_FMT(TAG, "SFN {}.{} received: SFN {}.{} cell_id={} msg_id=0x{:02X} {} for new slot", slot_data.ss_sched.u16.sfn, slot_data.ss_sched.u16.slot,
                   ss_msg.u16.sfn, ss_msg.u16.slot, msg_desc.cell_id, msg_desc.msg_id, get_cumac_msg_name(msg_desc.msg_id));
        slot_data.reset_slot_data(ss_msg);
        if (slot_data.task != nullptr)
        {
            slot_data.task->ts_start = transport(msg_desc.cell_id).get_ts_send(msg_desc);
        }
    }

    if (slot_data.task == nullptr)
    {
        NVLOGW_FMT(TAG, "SFN {}.{} handle_slot_msg_reorder: task is null, dropping cell_id={} msg_id=0x{:02X}",
                   ss_msg.u16.sfn, ss_msg.u16.slot, msg_desc.cell_id, msg_desc.msg_id);
        transport(msg_desc.cell_id).rx_release(msg_desc);
        return;
    }

    slot_data.slot_msgs[msg_desc.cell_id].push_msg(msg_desc);

    // Check if current cell_id message arrived
    nv::phy_mac_msg_desc* msg = nullptr;
    while (slot_data.curr_cell_id < slot_data.cell_num && ((msg = slot_data.slot_msgs[slot_data.curr_cell_id].pull_msg()) != nullptr)) {
        handle_slot_msg(*msg, ss_msg);
    }
}

//============================================================================
// MU UE-pair H5 dump (debug bit DBG_OPT_DUMP_UE_PAIR_H5)
//
// Mirrors cuphydriver/src/uplink/srs_ipc_manager.cpp dump_h5(): a two-phase
// design that keeps the RT worker thread cheap (synchronous D2H into
// pre-pinned staging) and pushes HDF5 file creation onto a detached
// SCHED_OTHER thread. Only the three MU UE-pair input buffers
// (chan_est / snr / chan_orth) are captured -- those are the buffers that
// muMimoUserPairing consumes and that the user wants to compare against TVs.
//============================================================================
namespace
{
//! Create a 1-D `H5T_NATIVE_UINT8` dataset of `num_bytes` bytes and write
//! `host_ptr` into it. Empty / null inputs are logged and treated as success.
//! @return 0 on success or skip; -1 on HDF5 failure.
int writeUePairBytesDataset(hid_t file, const char* ds_name, const void* host_ptr, size_t num_bytes)
{
    if (num_bytes == 0 || host_ptr == nullptr)
    {
        NVLOGI_FMT(TAG, "dump_ue_pair_h5: skipping dataset '{}' (bytes={}, ptr=0x{:x})",
                   ds_name, num_bytes, reinterpret_cast<uintptr_t>(host_ptr));
        return 0;
    }

    hsize_t dim   = static_cast<hsize_t>(num_bytes);
    hid_t   space = H5Screate_simple(1, &dim, nullptr);
    if (space < 0)
    {
        NVLOGE_FMT(TAG, AERIAL_CUMAC_CP_EVENT, "dump_ue_pair_h5: H5Screate_simple failed for '{}'", ds_name);
        return -1;
    }
    hid_t dset = H5Dcreate2(file, ds_name, H5T_NATIVE_UINT8, space,
                            H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    if (dset < 0)
    {
        H5Sclose(space);
        NVLOGE_FMT(TAG, AERIAL_CUMAC_CP_EVENT, "dump_ue_pair_h5: H5Dcreate2 failed for '{}'", ds_name);
        return -1;
    }
    herr_t wr = H5Dwrite(dset, H5T_NATIVE_UINT8, H5S_ALL, H5S_ALL, H5P_DEFAULT, host_ptr);
    H5Dclose(dset);
    H5Sclose(space);
    if (wr < 0)
    {
        NVLOGE_FMT(TAG, AERIAL_CUMAC_CP_EVENT, "dump_ue_pair_h5: H5Dwrite failed for '{}'", ds_name);
        return -1;
    }
    return 0;
}

//! Bundle of parameters forwarded by value into the detached H5-writer thread
//! so the worker never reads back into the handler.
struct UePairH5WriteJob
{
    std::string fname;
    uint16_t    sfn;
    uint16_t    slot;
    uint32_t    tv_slot_idx;     //!< Captured TV slot (log only; not written as H5 attribute)
    uint32_t    dump_index;      //!< 1-based ordinal of this dump in the per-process budget
    uint32_t    max_dump_slots;  //!< Total budget = count of loaded MU UE-pair TVs

    // Geometry attributes
    uint32_t    cell_num;
    uint32_t    num_bs_ant;
    uint32_t    num_ue_ant_port;
    uint32_t    num_subband;
    uint32_t    num_prg_samp_per_subband;
    uint32_t    num_srs_ue_per_cell;
    uint32_t    max_num_srs_ue_per_cell;

    // Dataset payload pointers (live in handler's staging buffers, not owned)
    const uint8_t* chan_est_h5;
    size_t         chan_est_bytes;
    const uint8_t* snr_h5;
    size_t         snr_bytes;
    const uint8_t* chan_orth_h5;
    size_t         chan_orth_bytes;
};

//! Body of the detached H5-writer thread. Drops scheduling priority to
//! SCHED_OTHER so disk I/O cannot preempt cuMAC's RT worker threads.
void runUePairH5WriteJob(UePairH5WriteJob job)
{
    int         before_policy = -1;
    int         after_policy  = -1;
    sched_param before_sp{};
    sched_param after_sp{};
    pthread_t   self  = pthread_self();
    int         gp_rc = pthread_getschedparam(self, &before_policy, &before_sp);

    sched_param normal_sp{};
    normal_sp.sched_priority = 0; // SCHED_OTHER requires prio 0
    int sp_rc = pthread_setschedparam(self, SCHED_OTHER, &normal_sp);
    pthread_getschedparam(self, &after_policy, &after_sp);

    NVLOGI_FMT(TAG,
               "dump_ue_pair_h5 thread: sfn={} slot={} tv_slot_idx={} dump={}/{} "
               "before(policy={}, prio={}, getrc={}) after(policy={}, prio={}, setrc={}) "
               "[SCHED_FIFO={}, SCHED_RR={}, SCHED_OTHER={}]",
               job.sfn, job.slot, job.tv_slot_idx, job.dump_index, job.max_dump_slots,
               before_policy, before_sp.sched_priority, gp_rc,
               after_policy,  after_sp.sched_priority,  sp_rc,
               SCHED_FIFO, SCHED_RR, SCHED_OTHER);

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
        NVLOGE_FMT(TAG, AERIAL_CUMAC_CP_EVENT,
                   "dump_ue_pair_h5: H5Fcreate failed for {} (sfn={}, slot={})",
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
    // Geometry attributes mirror the per-TV layout used by cumac_cp_tv and
    // the cuMAC L2-integration test bench; offline tools reuse these to
    // reshape the byte datasets into typed tensors.
    writeU32attr("cell_num",                 job.cell_num);
    writeU32attr("num_bs_ant",               job.num_bs_ant);
    writeU32attr("num_ue_ant_port",          job.num_ue_ant_port);
    writeU32attr("num_subband",              job.num_subband);
    writeU32attr("num_prg_samp_per_subband", job.num_prg_samp_per_subband);
    writeU32attr("num_srs_ue_per_cell",      job.num_srs_ue_per_cell);
    writeU32attr("max_num_srs_ue_per_cell",  job.max_num_srs_ue_per_cell);

    // Element-size attributes simplify reshape on the reader side
    writeU32attr("chan_est_elem_size",  static_cast<uint32_t>(sizeof(__half2)));
    writeU32attr("snr_elem_size",       static_cast<uint32_t>(sizeof(float)));
    writeU32attr("chan_orth_elem_size", static_cast<uint32_t>(sizeof(float)));

    if (writeUePairBytesDataset(file, "ue_pair_srs_chan_est", job.chan_est_h5,  job.chan_est_bytes)  != 0) ret = -1;
    if (writeUePairBytesDataset(file, "ue_pair_srs_snr",      job.snr_h5,       job.snr_bytes)       != 0) ret = -1;
    if (writeUePairBytesDataset(file, "ue_pair_chan_orth",    job.chan_orth_h5, job.chan_orth_bytes) != 0) ret = -1;

    H5Fclose(file);

    const auto t_h5_end = std::chrono::steady_clock::now();
    const long h5_us    = std::chrono::duration_cast<std::chrono::microseconds>(t_h5_end - t_h5_start).count();
    const long h5_ms    = std::chrono::duration_cast<std::chrono::milliseconds>(t_h5_end - t_h5_start).count();

    NVLOGI_FMT(TAG,
               "dump_ue_pair_h5 thread done: sfn={} slot={} tv_slot_idx={} dump={}/{} file={} h5_us={} h5_ms={} ret={}",
               job.sfn, job.slot, job.tv_slot_idx, job.dump_index, job.max_dump_slots,
               job.fname, h5_us, h5_ms, ret);
}
} // anonymous namespace

bool cumac_cp_handler::allocate_ue_pair_h5_staging()
{
    if (ue_pair_h5_staging_ready_.load(std::memory_order_acquire))
    {
        return true;
    }
    if (ue_pair_chan_est_bytes_ == 0 || ue_pair_snr_bytes_ == 0 || ue_pair_orth_bytes_ == 0)
    {
        // Geometry not finalized yet (init_ue_pair_gpu_resources hasn't run).
        return false;
    }

    // Cap = number of slots in group_tv.ue_pair that have a MU UE-pair TV
    // actually loaded. Sets the per-process dump budget so the file set
    // captures exactly one snapshot per scheduled MU UE-pair slot in the
    // TV pattern (no duplicates across SFNs).
    uint32_t num_loaded = 0;
    for (const auto& slot_tv : group_tv.ue_pair)
    {
        if (slot_tv.mu_ue_pair_tv_loaded) num_loaded++;
    }
    if (num_loaded == 0)
    {
        NVLOGW_FMT(TAG,
                   "dump_ue_pair_h5: no loaded MU UE-pair TV slots (group_tv.ue_pair.size={}); dumping disabled",
                   group_tv.ue_pair.size());
        ue_pair_h5_max_slots_ = 0;
        // ready stays false -> dump_ue_pair_h5 keeps retrying lazily, which
        // is fine because the loaded-count is stable after init.
        return false;
    }
    ue_pair_h5_max_slots_ = num_loaded;

    ue_pair_h5_staging_.resize(num_loaded);
    ue_pair_h5_dumped_tv_slot_.assign(group_tv.ue_pair.size(), 0);
    bool alloc_ok = true;
    for (uint32_t i = 0; i < num_loaded; i++)
    {
        UePairH5Staging& s = ue_pair_h5_staging_[i];
        if (s.chan_est_host == nullptr)
        {
            // Pinned host memory: target of cudaMemcpy D2H from a device pointer.
            cudaError_t cerr = cudaMallocHost(reinterpret_cast<void**>(&s.chan_est_host),
                                              ue_pair_chan_est_bytes_);
            if (cerr != cudaSuccess)
            {
                NVLOGE_FMT(TAG, AERIAL_CUMAC_CP_EVENT,
                           "dump_ue_pair_h5: cudaMallocHost({}) for staging slot {} failed: {} ({})",
                           ue_pair_chan_est_bytes_, i, static_cast<int>(cerr), cudaGetErrorString(cerr));
                s.chan_est_host = nullptr;
                alloc_ok = false;
            }
        }
        if (s.snr_host == nullptr)
        {
            // Also pinned: snr and chan_orth are GPU floats, so D2H benefits
            // from a pinned host buffer too.
            cudaError_t cerr = cudaMallocHost(reinterpret_cast<void**>(&s.snr_host),
                                              ue_pair_snr_bytes_);
            if (cerr != cudaSuccess)
            {
                NVLOGE_FMT(TAG, AERIAL_CUMAC_CP_EVENT,
                           "dump_ue_pair_h5: cudaMallocHost({}) for staging slot {} (snr) failed: {} ({})",
                           ue_pair_snr_bytes_, i, static_cast<int>(cerr), cudaGetErrorString(cerr));
                s.snr_host = nullptr;
                alloc_ok = false;
            }
        }
        if (s.chan_orth_host == nullptr)
        {
            cudaError_t cerr = cudaMallocHost(reinterpret_cast<void**>(&s.chan_orth_host),
                                              ue_pair_orth_bytes_);
            if (cerr != cudaSuccess)
            {
                NVLOGE_FMT(TAG, AERIAL_CUMAC_CP_EVENT,
                           "dump_ue_pair_h5: cudaMallocHost({}) for staging slot {} (chan_orth) failed: {} ({})",
                           ue_pair_orth_bytes_, i, static_cast<int>(cerr), cudaGetErrorString(cerr));
                s.chan_orth_host = nullptr;
                alloc_ok = false;
            }
        }
    }
    ue_pair_h5_staging_ready_.store(alloc_ok, std::memory_order_release);
    NVLOGC_FMT(TAG,
               "dump_ue_pair_h5: staging pre-allocated (slots={}, ue_pair_tv_size={}, chan_est={} B, snr={} B, chan_orth={} B, ready={})",
               ue_pair_h5_max_slots_, group_tv.ue_pair.size(),
               ue_pair_chan_est_bytes_, ue_pair_snr_bytes_, ue_pair_orth_bytes_,
               alloc_ok ? 1 : 0);
    return alloc_ok;
}

void cumac_cp_handler::free_ue_pair_h5_staging()
{
    // Join all H5 writer threads before releasing staging buffers so that
    // reconfiguration paths (not just process exit) are race-free.
    for (std::thread& t : ue_pair_h5_threads_)
    {
        if (t.joinable())
        {
            t.join();
        }
    }
    ue_pair_h5_threads_.clear();

    for (UePairH5Staging& s : ue_pair_h5_staging_)
    {
        if (s.chan_est_host)
        {
            cudaFreeHost(s.chan_est_host);
            s.chan_est_host = nullptr;
        }
        if (s.snr_host)
        {
            cudaFreeHost(s.snr_host);
            s.snr_host = nullptr;
        }
        if (s.chan_orth_host)
        {
            cudaFreeHost(s.chan_orth_host);
            s.chan_orth_host = nullptr;
        }
    }
    ue_pair_h5_staging_.clear();
    ue_pair_h5_dumped_tv_slot_.clear();
    ue_pair_h5_max_slots_ = 0;
    ue_pair_h5_staging_ready_.store(false, std::memory_order_release);
    ue_pair_h5_dump_count_.store(0, std::memory_order_release);
}

int cumac_cp_handler::dump_ue_pair_h5(uint16_t sfn, uint16_t slot, const char* outputDir)
{
    if (ue_pair_srs_chan_est_ == nullptr || ue_pair_srs_snr_ == nullptr || ue_pair_chan_orth_ == nullptr)
    {
        NVLOGW_FMT(TAG,
                   "dump_ue_pair_h5: ue_pair buffers not initialized (chan_est=0x{:x}, snr=0x{:x}, chan_orth=0x{:x}), skip sfn={} slot={}",
                   reinterpret_cast<uintptr_t>(ue_pair_srs_chan_est_),
                   reinterpret_cast<uintptr_t>(ue_pair_srs_snr_),
                   reinterpret_cast<uintptr_t>(ue_pair_chan_orth_),
                   sfn, slot);
        return 0;
    }
    if (group_tv.ue_pair.empty())
    {
        // No TV pattern loaded -> nothing to dedup against; skip rather than
        // dump unbounded snapshots from the live (cuphydriver-fed) buffers.
        return 0;
    }

    // Lazy staging allocation under a mutex so concurrent worker threads
    // see it exactly once. After ready=true the fast path skips the lock
    // via the atomic load in allocate_ue_pair_h5_staging().
    if (!ue_pair_h5_staging_ready_.load(std::memory_order_acquire))
    {
        std::lock_guard<std::mutex> lk(ue_pair_h5_init_mtx_);
        if (!allocate_ue_pair_h5_staging())
        {
            // allocate_ue_pair_h5_staging already logged the reason. Treat
            // as a non-error no-op so subsequent slots keep working.
            return 0;
        }
    }

    // Map (sfn, slot) to the TV slot it loaded from. Mirrors the formula
    // used in load_ue_pair_static_buffers / validate_buffer_setup so
    // tv_slot_idx is stable across re-visits of the same scheduled slot.
    const int tv_slot_idx_signed =
        (static_cast<int>(sfn) * SLOT_NUM_PER_FRAME + static_cast<int>(slot)) %
        static_cast<int>(group_tv.ue_pair.size());
    const uint32_t tv_slot_idx = static_cast<uint32_t>(tv_slot_idx_signed);

    if (!group_tv.ue_pair[tv_slot_idx].mu_ue_pair_tv_loaded)
    {
        // Not a scheduled MU UE-pair slot -> nothing meaningful to dump.
        return 0;
    }

    // Atomic claim: ensures each tv_slot_idx is captured at most once and
    // every successful claim gets a unique staging triple. Slow path
    // (D2H + thread launch) runs outside the mutex.
    uint32_t reserved_staging;
    {
        std::lock_guard<std::mutex> lk(ue_pair_h5_claim_mtx_);
        if (tv_slot_idx >= ue_pair_h5_dumped_tv_slot_.size())
        {
            NVLOGE_FMT(TAG, AERIAL_CUMAC_CP_EVENT,
                       "dump_ue_pair_h5: tv_slot_idx={} out of range (size={}), sfn={} slot={}",
                       tv_slot_idx, ue_pair_h5_dumped_tv_slot_.size(), sfn, slot);
            return -1;
        }
        if (ue_pair_h5_dumped_tv_slot_[tv_slot_idx])
        {
            return 0; // tv_slot already dumped -> no-op
        }
        if (ue_pair_h5_dump_count_.load(std::memory_order_relaxed) >= ue_pair_h5_max_slots_)
        {
            // Defense in depth: max_slots == count of loaded TVs, and we
            // mark each tv_slot at most once, so this should be unreachable.
            return 0;
        }
        ue_pair_h5_dumped_tv_slot_[tv_slot_idx] = 1;
        reserved_staging = ue_pair_h5_dump_count_.fetch_add(1, std::memory_order_acq_rel);
    }
    UePairH5Staging& staging = ue_pair_h5_staging_[reserved_staging];

    //-----------------------------------------------------------------------
    // Phase 1 -- synchronous D2H snapshot in the caller's worker thread.
    //-----------------------------------------------------------------------
    using clock = std::chrono::steady_clock;
    const auto t_snap_start = clock::now();

    int snap_ret = 0;
    cudaError_t cerr;
    cerr = cudaMemcpy(staging.chan_est_host, ue_pair_srs_chan_est_,
                      ue_pair_chan_est_bytes_, cudaMemcpyDeviceToHost);
    if (cerr != cudaSuccess)
    {
        NVLOGE_FMT(TAG, AERIAL_CUMAC_CP_EVENT,
                   "dump_ue_pair_h5: D2H chan_est failed (sfn={}, slot={}, bytes={}): {} ({})",
                   sfn, slot, ue_pair_chan_est_bytes_, static_cast<int>(cerr), cudaGetErrorString(cerr));
        snap_ret = -1;
    }
    cerr = cudaMemcpy(staging.snr_host, ue_pair_srs_snr_,
                      ue_pair_snr_bytes_, cudaMemcpyDeviceToHost);
    if (cerr != cudaSuccess)
    {
        NVLOGE_FMT(TAG, AERIAL_CUMAC_CP_EVENT,
                   "dump_ue_pair_h5: D2H snr failed (sfn={}, slot={}, bytes={}): {} ({})",
                   sfn, slot, ue_pair_snr_bytes_, static_cast<int>(cerr), cudaGetErrorString(cerr));
        snap_ret = -1;
    }
    cerr = cudaMemcpy(staging.chan_orth_host, ue_pair_chan_orth_,
                      ue_pair_orth_bytes_, cudaMemcpyDeviceToHost);
    if (cerr != cudaSuccess)
    {
        NVLOGE_FMT(TAG, AERIAL_CUMAC_CP_EVENT,
                   "dump_ue_pair_h5: D2H chan_orth failed (sfn={}, slot={}, bytes={}): {} ({})",
                   sfn, slot, ue_pair_orth_bytes_, static_cast<int>(cerr), cudaGetErrorString(cerr));
        snap_ret = -1;
    }

    const auto t_snap_end = clock::now();
    const long snap_us    = std::chrono::duration_cast<std::chrono::microseconds>(t_snap_end - t_snap_start).count();

    const uint32_t dump_so_far = reserved_staging + 1;

    NVLOGI_FMT(TAG,
               "dump_ue_pair_h5: RT snapshot done (sfn={}, slot={}, tv_slot_idx={}, dump={}/{}, chan_est={} B, snr={} B, chan_orth={} B, snap_us={}, ret={})",
               sfn, slot, tv_slot_idx, dump_so_far, ue_pair_h5_max_slots_,
               ue_pair_chan_est_bytes_, ue_pair_snr_bytes_, ue_pair_orth_bytes_,
               snap_us, snap_ret);

    if (snap_ret != 0)
    {
        return snap_ret;
    }

    //-----------------------------------------------------------------------
    // Phase 2 -- detached SCHED_OTHER thread writes the .h5 file.
    //-----------------------------------------------------------------------
    const char* dir = (outputDir != nullptr && outputDir[0] != '\0') ? outputDir : "/tmp";
    char fname[256];
    // Index prefix is the 0-based dump order within the session budget.
    std::snprintf(fname, sizeof(fname), "%s/cumac_ue_pair_buffers_%u_SFN_%u.%u.h5",
                  dir,
                  static_cast<unsigned>(reserved_staging),
                  static_cast<unsigned>(sfn),
                  static_cast<unsigned>(slot));

    UePairH5WriteJob job{};
    job.fname                    = fname;
    job.sfn                      = sfn;
    job.slot                     = slot;
    job.tv_slot_idx              = tv_slot_idx;
    job.dump_index               = dump_so_far;
    job.max_dump_slots           = ue_pair_h5_max_slots_;
    job.cell_num                 = group_params.nCell;
    job.num_bs_ant               = ue_pair_num_bs_ant_;
    job.num_ue_ant_port          = ue_pair_num_ue_ant_port_;
    job.num_subband              = ue_pair_num_subband_;
    job.num_prg_samp_per_subband = ue_pair_num_prg_samp_per_subband_;
    job.num_srs_ue_per_cell      = static_cast<uint32_t>(num_srs_info_);
    job.max_num_srs_ue_per_cell  = MAX_NUM_SRS_UE_PER_CELL;
    job.chan_est_h5              = staging.chan_est_host;
    job.chan_est_bytes           = ue_pair_chan_est_bytes_;
    job.snr_h5                   = staging.snr_host;
    job.snr_bytes                = ue_pair_snr_bytes_;
    job.chan_orth_h5             = staging.chan_orth_host;
    job.chan_orth_bytes          = ue_pair_orth_bytes_;

    {
        std::lock_guard<std::mutex> lk(ue_pair_h5_claim_mtx_);
        ue_pair_h5_threads_.emplace_back(runUePairH5WriteJob, std::move(job));
    }

    return 0;
}
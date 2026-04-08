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

#include <string.h>
#include <sys/time.h>

#include "nvlog.hpp"
#include "nv_utils.h"

#include "nv_phy_utils.hpp"
#include "cumac_app.hpp"
#include "cumac_msg.h"
#include "msg_recv.hpp"

#include "nv_phy_mac_transport.hpp"
#include "nv_phy_epoll_context.hpp"
#include "nvlog.hpp"

using namespace std;
using namespace nv;
using namespace cumac;

#define TAG (NVLOG_TAG_BASE_CUMAC_CP + 3) // "CUMCP.RECV"

using namespace std;
using namespace nv;
using namespace cumac;
using namespace std::chrono;

static void cumac_task_run(cumac_task *task)
{
    if (task == nullptr)
    {
        NVLOGE_FMT(TAG, AERIAL_TEST_CUMAC_EVENT, "null cumac_tas ptr");
        return;
    }

    task->setup();
    task->run();
    task->callback();
}

static void *cumac_worker_thread_func(void *arg)
{
    nvlog_fmtlog_thread_init();
    work_thread_arg_t *thread_arg = reinterpret_cast<work_thread_arg_t *>(arg);
    nv::lock_free_ring_pool<cumac_task> *task_ring = thread_arg->task_ring;

    if (assign_thread_name(thread_arg->thread_name) != 0)
    {
        NVLOGE_FMT(TAG, AERIAL_SYSTEM_API_EVENT, "{}: assign thread name [{}] failed", __FUNCTION__, thread_arg->thread_name);
    }

    if (assign_thread_cpu_core(thread_arg->cpu_core) != 0)
    {
        NVLOGE_FMT(TAG, AERIAL_SYSTEM_API_EVENT, "{}: assign CPU core [{}] failed", __FUNCTION__, thread_arg->cpu_core);
    }

    if (assign_thread_priority(80) != 0)
    {
        NVLOGE_FMT(TAG, AERIAL_SYSTEM_API_EVENT, "{}: set thread policy and priority failed", __FUNCTION__);
    }

    NVLOGC_FMT(TAG, "{}: thread [{}] running on CPU {} initialized fmtlog task_sem={}", __FUNCTION__,
               thread_arg->thread_name, sched_getcpu(), reinterpret_cast<void *>(thread_arg->task_sem));

    cumac_task *task = nullptr;
    while (sem_wait(thread_arg->task_sem) == 0)
    {
        NVLOGD_FMT(TAG, "[{}] thread got semophore", thread_arg->thread_name);

        while ((task = task_ring->dequeue()) != nullptr)
        {
            task->ts_dequeue = std::chrono::system_clock::now().time_since_epoch().count();
            NVLOGV_FMT(TAG, "[{}] SFN {}.{} start: ts_start={} process_delay={}", thread_arg->thread_name, task->ss.u16.sfn, task->ss.u16.slot,
                       task->ts_dequeue - task->ts_start, task->ts_dequeue - task->ts_start);
            cumac_task_run(task);
            task_ring->free(task);
        }
        NVLOGD_FMT(TAG, "[{}] thread waiting for next sem_post", thread_arg->thread_name);
    }

    NVLOGC_FMT(TAG, "Thread [{}] on CPU {} exiting", thread_arg->thread_name, sched_getcpu());
    return nullptr;
}

static void *cumac_receiver_thread_func(void *arg)
{
    cumac_receiver *receiver = reinterpret_cast<cumac_receiver *>(arg);
    receiver->start_epoll_loop();
    return nullptr;
}

cumac_receiver::cumac_receiver(yaml::node &yaml_node, cumac_cp_configs &_configs) :
    transp_wrapper(yaml_node, NV_IPC_MODULE_PRIMARY, _configs.cell_num),
    configs(_configs),
    _handler(_configs, transp_wrapper)
{
    // Initialize CUDA device before creating the ring pool
    if (configs.gpu_id < 0)
    {
        NVLOGE_FMT(TAG, AERIAL_CONFIG_EVENT, "Invalid GPU ID: gpu_id={}", configs.gpu_id);
    }

    CHECK_CUDA_ERR(cudaSetDevice(configs.gpu_id));

    task_ring = new lock_free_ring_pool<cumac_task>("cumac_task", configs.task_ring_len, sizeof(cumac_task));

    if (sem_init(&_task_sem, 0, 0) != 0)
    {
        NVLOGE_FMT(TAG, AERIAL_SYSTEM_API_EVENT, "{}: sem_init failed: {}", __FUNCTION__, strerror(errno));
    }

    cell_num = configs.cell_num;
    _handler.set_task_ring(task_ring, &_task_sem);

    // Config event_fd callback
    epoll_ctx_p.reset(new phy_epoll_context());
    std::unique_ptr<member_event_callback<cumac_receiver>> mcb_p(new member_event_callback<cumac_receiver>(this, &cumac_receiver::recv_msg));
    for (phy_mac_transport *ptransport : transp_wrapper.get_transports())
    {
        epoll_ctx_p->add_fd(ptransport->get_fd(), mcb_p.get());
    }
    msg_mcb_p = std::move(mcb_p);

    worker_thread_args.resize(configs.worker_cores.size());
    for (int core_id = 0; core_id < configs.worker_cores.size(); core_id++)
    {
        worker_thread_args[core_id].resize(configs.thread_num_per_core);
        for (int thread_id = 0; thread_id < configs.thread_num_per_core; thread_id++)
        {
            work_thread_arg_t &thread_arg = worker_thread_args[core_id][thread_id];
            thread_arg.task_sem = &_task_sem;
            thread_arg.task_ring = task_ring;
            thread_arg.cpu_core = configs.worker_cores[core_id];
            char name_buf[32];
            snprintf(name_buf, 32, "worker_%02d_%d", thread_arg.cpu_core, thread_id);
            nvlog_safe_strncpy(thread_arg.thread_name, name_buf, 16);

            if (pthread_create(&thread_arg.pthread_id, NULL, cumac_worker_thread_func, &thread_arg) != 0)
            {
                NVLOGE_FMT(TAG, AERIAL_SYSTEM_API_EVENT, "Create worker_[{:02}]_[{}] thread failed", thread_arg.cpu_core, thread_id);
            }
        }
    }

    NVLOGC_FMT(TAG, "{}: initialized. cell_num={} debug_option=0x{:X} configs.run_in_cpu={} gpu_id={} cuda_block_num={}",
            __func__, cell_num, configs.debug_option, configs.run_in_cpu, configs.gpu_id, configs.cuda_block_num);
}

cumac_receiver::~cumac_receiver()
{
}

void cumac_receiver::start_epoll_loop()
{
    struct nv::thread_config &thread_cfg = configs.get_recv_thread_config();
    if (config_thread_property(thread_cfg) < 0)
    {
        NVLOGE_FMT(TAG, AERIAL_SYSTEM_API_EVENT, "{}: config thread {} failed: cpu_core={} priority={}", __func__, thread_cfg.name, thread_cfg.cpu_affinity, thread_cfg.sched_priority);
    }

    epoll_ctx_p->start_event_loop();
}

void cumac_receiver::start()
{
    if (pthread_create(&recv_thread_id, NULL, cumac_receiver_thread_func, this) != 0)
    {
        NVLOGE_FMT(TAG, AERIAL_SYSTEM_API_EVENT, "Create cuMAC-CP receiver thread failed");
    }
}

void cumac_receiver::stop()
{
}

void cumac_receiver::join()
{
    if (pthread_join(recv_thread_id, NULL) != 0)
    {
        NVLOGE_FMT(TAG, AERIAL_SYSTEM_API_EVENT, "Join  cuMAC-CP receiver thread failed");
    }
}

void cumac_receiver::recv_msg()
{
    phy_mac_msg_desc msg;
    while (transport_wrapper().rx_recv(msg) >= 0)
    {
        if (on_msg(msg))
        {
            // Release the NVIPC buffers if FAPI message handle finished in on_msg()
            transport_wrapper().rx_release(msg);
        }
    }
}

// Return whether the nvipc message should be freed
bool cumac_receiver::on_msg(nv::phy_mac_msg_desc &msg)
{
    bool ready_to_free = true;
    if (msg.msg_id < 0 || msg.cell_id >= cell_num)
    {
        NVLOGE_FMT(TAG, AERIAL_TEST_CUMAC_EVENT, "{}: Invalid CUMAC MSG cell_id: cell_id={} msg_id={}", __func__, msg.cell_id, msg.msg_id);
        return ready_to_free;
    }

    sfn_slot_t ss_msg = nv_ipc_get_sfn_slot(&msg);
    NVLOGI_FMT(TAG, "SFN {}.{} RECV: cell_id={} msg_id=0x{:02X} {}",
               ss_msg.u16.sfn, ss_msg.u16.slot, msg.cell_id, msg.msg_id, get_cumac_msg_name(msg.msg_id));

    // For controller messages like CONFIG.req, START.req
    if (ss_msg.u32 == SFN_SLOT_INVALID)
    {
        switch (msg.msg_id)
        {
        case CUMAC_CONFIG_REQUEST:
            _handler.on_config_request(msg);
            break;
        case CUMAC_START_REQUEST:
            _handler.on_start_request(msg);
            break;
        case CUMAC_STOP_REQUEST:
            _handler.on_stop_request(msg);
            break;
        }
        return ready_to_free;
    }

    // For slot messages which have SFN/SLOT values
    // _handler.handle_slot_msg_reorder(msg, ss_msg);
    _handler.handle_slot_msg(msg, ss_msg);

    return false;
}
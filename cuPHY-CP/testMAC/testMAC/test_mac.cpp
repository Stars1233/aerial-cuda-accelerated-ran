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

#include <iostream>
#include <map>
#include <string>
#include <iterator>
#include <algorithm>
#include <unordered_map>
#include <exception>
#include <inttypes.h>
#include <signal.h>
#include <string.h>
#include <sys/time.h>

#include "test_mac.hpp"
#include "scf_fapi_handler.hpp"
#include "signal_handler.hpp"

#define TAG (NVLOG_TAG_BASE_TEST_MAC + 7) // "MAC.PROC"

using namespace std;
using namespace nv;

test_mac::test_mac(yaml::node yaml_root, uint32_t cell_num) :
        testmac_yaml(yaml_root) {

    NVLOGC_FMT(TAG, "{} construct start", __func__);

    _transport = new phy_mac_transport(yaml_root["transport"], NV_IPC_MODULE_MAC, cell_num);

    configs = NULL; // Will be set later by set_launch_pattern_and_configs()
    _fapi_handler = NULL; // Will be created later by set_launch_pattern_and_configs()

    mac_recv_tid = 0; // MAC receiver thread ID, initialized to 0

    NVLOGC_FMT(TAG, "{} construct finished", __func__);
}

test_mac::~test_mac()
{
    delete _transport;
}

void test_mac::set_launch_pattern_and_configs(test_mac_configs* _configs, launch_pattern* _lp)
{
    configs = _configs;

    // Query IPC buffer sizes from transport and update configuration
    configs->set_max_msg_size(nv_ipc_get_buf_size(transport().get_nv_ipc_config(), NV_IPC_MEMPOOL_CPU_MSG));
#ifdef ENABLE_32DL
    configs->set_max_data_size(nv_ipc_get_buf_size(transport().get_nv_ipc_config(), NV_IPC_MEMPOOL_CPU_LARGE));
#else
    configs->set_max_data_size(nv_ipc_get_buf_size(transport().get_nv_ipc_config(), NV_IPC_MEMPOOL_CPU_DATA));
#endif

    int data_buf_opt = configs->get_fapi_tb_loc();
    NVLOGC_FMT(TAG, "{}: create SCF FAPI interface. tb_loc={} max_msg_size={} max_data_size={} pdsch_align_bytes={}",
            __FUNCTION__, data_buf_opt, configs->get_max_msg_size(), configs->get_max_data_size(), configs->pdsch_align_bytes);

    // Create SCF FAPI handler with transport, configs, launch pattern, and conformance stats
    _fapi_handler = new scf_fapi_handler(transport(), _configs, _lp, &conformance_test_stats);
}

void* oam_thread_func(void* arg)
{
    nvlog_fmtlog_thread_init();
    NVLOGC_FMT(TAG, "Thread {} on CPU {} initialized fmtlog", __FUNCTION__, sched_getcpu());

    // nv_assign_thread_cpu_core(0);
    if(pthread_setname_np(pthread_self(), "oam_thread") != 0)
    {
        NVLOGW_FMT(TAG, "{}: set thread name failed", __func__);
    }

    fapi_handler *_fapi_handler = reinterpret_cast<fapi_handler*>(arg);

    CuphyOAM* oam = CuphyOAM::getInstance();
    while(1)
    {
        if (is_app_exiting())
        {
            NVLOGC_FMT(TAG, "OAM thread exiting due to app exiting");
            _fapi_handler->terminate();
            break;
        }

        CuphyOAMCellCtrlCmd* cmd;
        while((cmd = oam->get_cell_ctrl_cmd()) != nullptr)
        {
            if(cmd->target_cell_id >= 0)
            {
                NVLOGC_FMT(TAG,"cell_ctrl_cmd: {}, cell_id: {} target_cell_id: {}", cmd->cell_ctrl_cmd, cmd->cell_id, cmd->target_cell_id);
            }
            else
            {
                NVLOGC_FMT(TAG,"cell_ctrl_cmd: {}, cell_id: {} ", cmd->cell_ctrl_cmd, cmd->cell_id);
            }

            if(cmd->cell_id < 0 || cmd->cell_id >= _fapi_handler->get_cell_num())
            {
                NVLOGE_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "Invalid cell_id: {}", cmd->cell_id);
                oam->free_cell_ctrl_cmd(cmd);
                continue;
            }

            switch(cmd->cell_ctrl_cmd)
            {
            case 0: //stop cell
                _fapi_handler->cell_stop(cmd->cell_id);
                break;
            case 1: //start cell
                _fapi_handler->cell_start(cmd->cell_id);
                break;
            case 2: //Re-config cell
                if(cmd->target_cell_id >= 0 && cmd->target_cell_id < _fapi_handler->get_cell_num())
                {
                    if(_fapi_handler->cell_id_remap(cmd->cell_id, cmd->target_cell_id) != 0)
                    {
                        NVLOGE_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "A cell re-config request is already in process, please try again after it's finished");
                        break;
                    }
                    _fapi_handler->send_config_request(cmd->cell_id);
                }
                else
                {
                    NVLOGE_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "Invalid target_cell_id: {}", cmd->target_cell_id);
                }
                break;
            case 3: //Init config
                // Always send CONFIG.req with full parameters for OAM CONFIG command
                _fapi_handler->set_first_init_flag(cmd->cell_id, true);
                _fapi_handler->cell_init(cmd->cell_id);
                break;
            }
            oam->free_cell_ctrl_cmd(cmd);
        }

        CuphyOAMFapiDelayCmd* command;
        while((command = oam->get_fapi_delay_cmd()) != nullptr)
        {
            _fapi_handler->set_fapi_delay(command->cell_id, command->slot, command->fapi_mask, command->delay_us);
            oam->free_fapi_delay_cmd(command);
        }

        CuphyOAMGenericAsyncCmd* acmd;
        while((acmd = oam->get_generic_async_cmd()) != nullptr)
        {
            NVLOGI_FMT(TAG,"OAM Async CMD: cmd_id={} param_int1={} param_int2={} param_str={}",
                    acmd->cmd_id, acmd->param_int1, acmd->param_int2, acmd->param_str.c_str());
            switch (acmd->cmd_id)
            {
            case 1:
                NVLOGC_FMT(TAG, "OAM Set rnti_test_mode: {}", acmd->param_int1);
                _fapi_handler->set_rnti_test_mode(acmd->param_int1);
                break;
            default:
                NVLOGE_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "OAM cmd_id not supported: {}", acmd->cmd_id);
                break;
            }
            oam->free_generic_async_cmd(acmd);
        }

        // Sleep for 100ms before next poll to reduce CPU usage
        usleep(100 * 1000);
    }

    NVLOGI_FMT(TAG, "test_mac::oam_thread_func exit thread");
    return nullptr;
}

void* scheduler_thread_func(void* arg)
{
    fapi_handler *_fapi_handler = reinterpret_cast<fapi_handler*>(arg);
    config_thread_property(_fapi_handler->get_configs()->get_sched_thread_config());

    nvlog_fmtlog_thread_init();
    NVLOGC_FMT(TAG, "Thread {} on CPU {} initialized fmtlog", __FUNCTION__, sched_getcpu());

    _fapi_handler->scheduler_thread_func();
    return nullptr;
}

void* builder_thread_func(void* arg)
{
    fapi_handler *_fapi_handler = reinterpret_cast<fapi_handler*>(arg);
    config_thread_property(_fapi_handler->get_configs()->get_builder_thread_config());

    nvlog_fmtlog_thread_init();
    NVLOGC_FMT(TAG, "Thread {} on CPU {} initialized fmtlog", __FUNCTION__, sched_getcpu());

    _fapi_handler->builder_thread_func();
    return nullptr;
}

void* worker_thread_func(void* arg)
{
    fapi_handler *_fapi_handler = reinterpret_cast<fapi_handler*>(arg);
    _fapi_handler->worker_thread_func();
    return nullptr;
}

void* mac_recv_thread_func(void *arg)
{
    nvlog_fmtlog_thread_init();

    test_mac *testmac = reinterpret_cast<test_mac*>(arg);
    test_mac_configs* configs = testmac->get_configs();
    fapi_handler *_fapi_handler = testmac->get_fapi_handler();

    if (configs->builder_thread_enable != 0)
    {
        pthread_t thread_id;
        if(pthread_create(&thread_id, NULL, builder_thread_func, _fapi_handler) !=  0)
        {
            NVLOGE_FMT(TAG, AERIAL_SYSTEM_API_EVENT, "Create FAPI builder thread failed");
        }
    }

    pthread_t thread_id;
    if(pthread_create(&thread_id, NULL, oam_thread_func, _fapi_handler) !=  0)
    {
        NVLOGE_FMT(TAG, AERIAL_SYSTEM_API_EVENT, "Create thread oam_thread_func failed");
    }

    bool worker_threads_enabled = false;
    if(configs->worker_cores.size() > 0)
    {
        worker_threads_enabled = true;
        for(int i = 0; i < configs->worker_cores.size(); i++)
        {
            if(pthread_create(&thread_id, NULL, worker_thread_func, _fapi_handler) != 0)
            {
                NVLOGE_FMT(TAG, AERIAL_SYSTEM_API_EVENT, "Create thread worker_thread_func failed");
            }
        }
    }

    config_thread_property(configs->get_recv_thread_config());
    NVLOGC_FMT(TAG, "Thread {} on CPU {} initialized fmtlog", __FUNCTION__, sched_getcpu());

    // Wait for IPC connection to be fully established before proceeding
    sleep(1);

    // Send OAM cell update message if configured at the first slot
    _fapi_handler->schedule_cell_update(0);

    try
    {
        // Initialize all cells if not controlled by OAM
        if(configs->oam_cell_ctrl_cmd == 0)
        {
            for(int cell_id = 0; cell_id < _fapi_handler->get_cell_num(); cell_id++)
            {
                _fapi_handler->cell_init(cell_id);
            }
        }
    }
    catch(std::exception& e)
    {
        NVLOGF_FMT(TAG, AERIAL_TEST_MAC_EVENT, "IPC send failed, please check whether cuphycontroller is running properly", e.what());
        return nullptr;
    }
    catch(...)
    {
        NVLOGF_FMT(TAG, AERIAL_TEST_MAC_EVENT, "test_mac::thread_func() unknown exception");
        return nullptr;
    }

    phy_mac_transport& transport = testmac->transport();
    nv::phy_mac_msg_desc msg_desc;

    // Main message receive loop
    while (1) {
        try {
            // Wait for incoming messages from PHY
            transport.rx_wait();

            if(worker_threads_enabled)
            {
                // Notify worker threads to process messages
                _fapi_handler->notify_worker_threads();
            }
            else
            {
                // Process messages directly in this thread
                while(transport.rx_recv(msg_desc) >= 0)
                {
                    _fapi_handler->on_msg(msg_desc);
                    transport.rx_release(msg_desc);
                }
            }
        } catch (std::exception &e) {
            NVLOGF_FMT(TAG, AERIAL_TEST_MAC_EVENT, "mac_recv_thread_func: exception: {}", e.what());
        }
    }

    return nullptr;
}

void test_mac::start() {
    NVLOGC_FMT(TAG, "test_mac::start");

    if(pthread_create(&mac_sched_tid, NULL, scheduler_thread_func, get_fapi_handler()) !=  0)
    {
        NVLOGE_FMT(TAG, AERIAL_SYSTEM_API_EVENT, "Create mac_sched thread failed");
    }

    if (pthread_create(&mac_recv_tid, NULL, mac_recv_thread_func, this) != 0) {
        NVLOGE_FMT(TAG, AERIAL_SYSTEM_API_EVENT, "Create mac_recv thread failed");
    }
}

void test_mac::join() {
    if (mac_sched_tid != 0 && pthread_join(mac_sched_tid, NULL) != 0) {
        NVLOGE_FMT(TAG, AERIAL_SYSTEM_API_EVENT, "Join mac_sched thread failed");
    }

    if (mac_recv_tid != 0) {
        pthread_cancel(mac_recv_tid);
        pthread_join(mac_recv_tid, NULL);
    }

    NVLOGC_FMT(TAG, "test_mac: [mac_sched] and [mac_recv] threads joined");
}

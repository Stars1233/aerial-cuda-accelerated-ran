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

#include <chrono>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <iterator>
#include <algorithm>
#include <unordered_map>
#include <exception>
#include <filesystem>
#include <inttypes.h>
#include <signal.h>
#include <string.h>
#include <sys/time.h>

#include "test_mac.hpp"
#include "scf_fapi_handler.hpp"
#include "yaml_sdk_version.hpp"

#include <grpcpp/grpcpp.h>
#include "aerial_common.grpc.pb.h"

#define TAG (NVLOG_TAG_BASE_TEST_MAC + 7) // "MAC.PROC"

using namespace std;
using namespace nv;

test_mac::test_mac(const char* config_yaml_path)
{
    _fapi_handler = nullptr;

    mac_recv_tid = 0;
    mac_sched_tid = 0;

    NVLOGC_FMT(TAG, "test_mac_config_yaml={}", config_yaml_path);

    _config_yaml_parser = std::make_unique<yaml::file_parser>(config_yaml_path);
    _config_yaml_document = _config_yaml_parser->next_document();
    yaml::node yaml_root = _config_yaml_document.root();

    aerial::check_yaml_version(yaml_root, config_yaml_path);

    load_nv_ipc_yaml_config(&ipc_config, config_yaml_path, NV_IPC_MODULE_MAC);

    _configs = std::make_unique<test_mac_configs>(yaml_root);
    _configs->set_max_msg_size(default_max_msg_size);
    _configs->set_max_data_size(default_max_data_size);
}

test_mac::~test_mac() = default;

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
                // Disable CONFIG.req retry for OAM CONFIG command
                _fapi_handler->get_configs()->cell_config_retry = 0;
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

/// Wait until the RU emulator at @p host is reachable via gRPC, or @p timeout_secs elapses.
/// @param host       Hostname (or host:port) of the RU emulator.
/// @param timeout_secs  Maximum seconds to wait before continuing.
static void wait_for_ru_emulator(const std::string& host, int timeout_secs)
{
    static constexpr const char* kRuEmulatorPort = "50052";
    std::string address = (host.find(':') == std::string::npos) ? host + ":" + kRuEmulatorPort : host;
    NVLOGC_FMT(TAG, "Waiting for RU emulator at {} (timeout={}s)", address, timeout_secs);

    auto stub = aerial::Common::NewStub(
        grpc::CreateChannel(address, grpc::InsecureChannelCredentials()));

    aerial::GenericRequest request;
    aerial::CpuUtilizationReply reply;

    for(int elapsed = 0; elapsed < timeout_secs; elapsed++)
    {
        grpc::ClientContext ctx;
        ctx.set_deadline(std::chrono::system_clock::now() + std::chrono::seconds(5));
        grpc::Status status = stub->GetCpuUtilization(&ctx, request, &reply);
        if(status.ok())
        {
            NVLOGC_FMT(TAG, "RU emulator ready after {}s", elapsed);
            return;
        }
        NVLOGC_FMT(TAG, "RU emulator not ready ({}s elapsed), retrying...", elapsed);
        sleep(1);
    }
    NVLOGE_FMT(TAG, AERIAL_SYSTEM_API_EVENT,
        "RU emulator at {} not ready after {}s, continuing anyway", address, timeout_secs);
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

    if(!configs->ru_emulator_host.empty())
    {
        int timeout = 15 * (_fapi_handler->get_cell_num() + 1);
        wait_for_ru_emulator(configs->ru_emulator_host, timeout);
    }

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

void test_mac::start(bool enable_uplink, bool enable_downlink) {
    NVLOGC_FMT(TAG, "{}: enable_uplink={} enable_downlink={}", __func__, enable_uplink, enable_downlink);

    if (enable_uplink == false && enable_downlink == true) {
        NVLOGE_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "Downlink thread depends on uplink thread, please set enable_uplink=true while enable_downlink=true");
        return;
    }

    if (_fapi_handler == nullptr) {
        NVLOGE_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "Launch pattern is not loaded, please call load_launch_pattern first");
        return;
    }

    if (!_transport) {
        _transport = std::make_unique<phy_mac_transport>(ipc_config, _lp->get_cell_num());
    }
    _fapi_handler->set_transport(_transport.get());

    // Query IPC buffer sizes from transport and update configuration (nv_ipc_get_buf_size returns -1 for unknown transport)
    nv_ipc_config_t* ipc_cfg = transport().get_nv_ipc_config();
    int msg_sz = nv_ipc_get_buf_size(ipc_cfg, NV_IPC_MEMPOOL_CPU_MSG);
    if (msg_sz > 0)
    {
        _configs->set_max_msg_size(msg_sz);
    }
    else
    {
        NVLOGE_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "nv_ipc_get_buf_size(NV_IPC_MEMPOOL_CPU_MSG) returned {}; keeping max_msg_size={}",
            msg_sz, _configs->get_max_msg_size());
    }
    // TB data-pool sizing: 0/1 -> CPU_DATA, 2 -> CPU_LARGE, 3 -> GPU_DATA.
    nv_ipc_mempool_id_t data_pool = NV_IPC_MEMPOOL_CPU_DATA;
    if (_configs->get_fapi_tb_loc() == 2)
    {
        data_pool = NV_IPC_MEMPOOL_CPU_LARGE;
    }
    else if (_configs->get_fapi_tb_loc() == 3)
    {
        data_pool = NV_IPC_MEMPOOL_GPU_DATA;
    }
    int data_sz = nv_ipc_get_buf_size(ipc_cfg, data_pool);
    if (data_sz > 0)
    {
        _configs->set_max_data_size(data_sz);
    }
    else
    {
        NVLOGE_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "nv_ipc_get_buf_size(data pool) returned {}; keeping max_data_size={}",
            data_sz, _configs->get_max_data_size());
    }

    if(enable_downlink)
    {
        if(pthread_create(&mac_sched_tid, NULL, scheduler_thread_func, get_fapi_handler()) !=  0)
        {
            NVLOGE_FMT(TAG, AERIAL_SYSTEM_API_EVENT, "Create mac_sched thread failed");
        }
    }

    if(enable_uplink)
    {
        if (pthread_create(&mac_recv_tid, NULL, mac_recv_thread_func, this) != 0) {
            NVLOGE_FMT(TAG, AERIAL_SYSTEM_API_EVENT, "Create mac_recv thread failed");
        }
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

int test_mac::load_launch_pattern(const char* launch_pattern_path, uint64_t cell_mask, uint32_t channel_mask)
{
    NVLOGC_FMT(TAG, "{}: launch_pattern_yaml={} cell_mask=0x{:X} channel_mask=0x{:X}", __func__, launch_pattern_path, cell_mask, channel_mask);

    _lp = std::make_unique<launch_pattern>(_configs.get());
    if(_lp->launch_pattern_parsing(launch_pattern_path, channel_mask, cell_mask) < 0)
    {
        NVLOGE_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "Launch pattern parsing failed");
        return -1;
    }

    _fapi_handler = std::make_unique<scf_fapi_handler>(_configs.get(), _lp.get(), &conformance_test_stats);
    int data_buf_opt = _configs->get_fapi_tb_loc();
    NVLOGC_FMT(TAG, "{}: create SCF FAPI interface. tb_loc={} max_msg_size={} max_data_size={} pdsch_align_bytes={}",
            __FUNCTION__, data_buf_opt, _configs->get_max_msg_size(), _configs->get_max_data_size(), _configs->pdsch_align_bytes);

    return 0;
}

int test_mac::prebuild_fapi_messages()
{
    if (_fapi_handler == nullptr)
    {
        NVLOGE_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "Launch pattern is not loaded, please call load_launch_pattern first");
        return -1;
    }

    return _fapi_handler->prebuild_downlink_messages();
}

void test_mac::print_prebuilt_fapi_messages()
{
    if (_fapi_handler == nullptr)
    {
        NVLOGE_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "Launch pattern is not loaded, please call load_launch_pattern first");
        return;
    }

    int slot_num = _lp->get_sched_slot_num();
    int cell_num = _lp->get_cell_num();

    // Print CONFIG.req for all cells
    NVLOGC_FMT(TAG, "=============================================");
    NVLOGC_FMT(TAG, "Print CONFIG.req for all cells");
    NVLOGC_FMT(TAG, "---------------------------------------------");
    for (int cell_id = 0; cell_id < cell_num; cell_id++)
    {
        const nv::phy_mac_msg_desc* config_req = get_prebuilt_config_req(cell_id);
        if(config_req != nullptr)
        {
            NVLOGC_FMT(TAG, "Cell {} msg_id=0x{:02X} - {} msg_len={} data_len={}", cell_id, config_req->msg_id, get_scf_fapi_msg_name(config_req->msg_id), config_req->msg_len, config_req->data_len);
        }
    }

    NVLOGC_FMT(TAG, "=============================================");
    NVLOGC_FMT(TAG, "Print slot messages for all cells");
    NVLOGC_FMT(TAG, "---------------------------------------------");
    // Print slot messages for all cells
    sfn_slot_t ss = {.u16 = {0, 0}};
    for(int slot_idx = 0; slot_idx < slot_num; slot_idx++) {
        for(int cell_id = 0; cell_id < cell_num; cell_id++) {
            std::span<const nv::phy_mac_msg_desc> slot_msgs = get_prebuilt_slot_messages(cell_id, ss);
            if(!slot_msgs.empty()) {
                for(int msg_idx = 0; msg_idx < static_cast<int>(slot_msgs.size()); msg_idx++)
                {
                    const nv::phy_mac_msg_desc& m = slot_msgs[static_cast<size_t>(msg_idx)];
                    NVLOGC_FMT(TAG, "Slot {} SFN {}.{} Cell {} msg_id=0x{:02X} - {} msg_len={} data_len={}",
                        slot_idx, ss.u16.sfn, ss.u16.slot, cell_id, m.msg_id, get_scf_fapi_msg_name(m.msg_id), m.msg_len, m.data_len);
                }
            } else {
                NVLOGE_FMT(TAG, AERIAL_TEST_MAC_EVENT, "Slot {} SFN {}.{} Cell {} has no messages",
                    slot_idx, ss.u16.sfn, ss.u16.slot, cell_id);
            }
        }
        // Get the next slot SFN/SLOT number
        ss = _fapi_handler->get_next_sfn_slot(ss);
    }
    NVLOGC_FMT(TAG, "=============================================");
}

const nv::phy_mac_msg_desc* test_mac::get_prebuilt_config_req(int cell_id) const
{
    if (_fapi_handler == nullptr)
    {
        NVLOGE_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "Launch pattern is not loaded, please call load_launch_pattern first");
        return nullptr;
    }
    return _fapi_handler->get_prebuilt_config_req(cell_id);
}

std::span<const nv::phy_mac_msg_desc> test_mac::get_prebuilt_slot_messages(int cell_id, sfn_slot_t ss) const
{
    if (_fapi_handler == nullptr)
    {
        NVLOGE_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "Launch pattern is not loaded, please call load_launch_pattern first");
        return {};
    }
    return _fapi_handler->get_prebuilt_slot_messages(cell_id, ss);
}

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

#include <stdio.h>
#include <grpcpp/grpcpp.h>
#include <grpcpp/ext/proto_server_reflection_plugin.h>

#include "oam_services.hpp"
#include "cuphyoam.hpp"

#include "scf_fapi_oam_services.hpp"

class CuphyOAMRpc
{
    public:
        CuphyOAMRpc() {};
        // CuphyGreeterServiceImpl greeterSvc;
        P9MessagesService p9MsgSvc;
        CuphyCommonServiceImpl commonSvc;
        CuphyPushNotificationServiceImpl pushNotificationSvc;
        grpc::ServerBuilder builder;
        std::unique_ptr<grpc::Server> server;
};

CuphyOAM* CuphyOAM::instance = nullptr;

std::string CuphyOAM::server_addr = "0.0.0.0:50051";


CuphyOAM* CuphyOAM::getInstance()
{
    if (instance == nullptr)
    {
        instance = new CuphyOAM();
    }

    return instance;
}

void CuphyOAM::notifyClient(int clientId, std::string msg)
{
    CuphyOAMRpc *rpc = reinterpret_cast<CuphyOAMRpc*>(pRpc);
    if (rpc != nullptr) rpc->pushNotificationSvc.notifyClient(clientId, msg);
}

void CuphyOAM::notifyAll(std::string msg)
{
    CuphyOAMRpc *rpc = reinterpret_cast<CuphyOAMRpc*>(pRpc);
    if (rpc != nullptr) rpc->pushNotificationSvc.notifyAll(msg);
}

void CuphyOAM::set_server_address(std::string addr)
{
    server_addr = addr;
}

std::string CuphyOAM::get_server_address()
{
    return server_addr;
}

int CuphyOAM::init_rpc_server()
{
    CuphyOAMRpc *rpc = new CuphyOAMRpc();
    pRpc = reinterpret_cast<void*>(rpc);

    // Setup gRPC Server
    std::string server_address = CuphyOAM::get_server_address();

    grpc::EnableDefaultHealthCheckService(true);
    grpc::reflection::InitProtoReflectionServerBuilderPlugin();
    rpc->builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());

    // Register each service
    // rpc->builder.RegisterService(&rpc->greeterSvc);
    rpc->builder.RegisterService(&rpc->commonSvc).RegisterService(&rpc->p9MsgSvc).RegisterService(&rpc->pushNotificationSvc);

    rpc->server = std::unique_ptr<grpc::Server>(rpc->builder.BuildAndStart());
    std::cout << "gRPC Server listening on " << server_address << std::endl;

    return 0;
}

int CuphyOAM::init_pools()
{
#ifdef CUPHYOAM_WIP
    struct rte_mempool *local_p;
    char s[256];

    // Initialize oam control memory pool
    p = rte_mempool_create(
        "P_ctrl",           // name
        1023,               // n
        1024,               // elt_size
        0,                  // cache_size
        0,                  // private_data_size
        NULL,               // mp_init
        NULL,               // mp_init_arg
        NULL,               // obj_init
        NULL,               // obj_init_arg
        SOCKET_ID_ANY,      // socket_id
        0                   // flags
    );
    if (p == 0)
    {
        printf("error on pool create: %d\n",rte_errno);
        return -1;
    }
#endif
    return 0;
}

int CuphyOAM::init_rings()
{
#ifdef CUPHYOAM_WIP
    char s[256];

    for (int k=0; k<MAX_CORES; k++)
    {
        sprintf(s,"R_ctrl_rx_%d",k);
        q_rx[k] = rte_ring_create(s,                // name
                                  1024,             // count
                                  SOCKET_ID_ANY,    // socket
                                  0                 // flags
        );
        if (q_rx[k] == 0)
        {
            printf("error on ring %s create: %d\n",s,rte_errno);
            return -1;
        }

        sprintf(s,"R_ctrl_tx_%d",k);
        q_tx[k] = rte_ring_create(s,                // name
                                  1024,             // count
                                  SOCKET_ID_ANY,    // socket
                                  0                 // flags
        );
        if (q_tx[k] == 0)
        {
            printf("error on ring %s create: %d\n",s,rte_errno);
            return -1;
        }
    }
#endif
    return 0;
}

int CuphyOAM::init_cell_config_ring()
{
    char s[256];
    sprintf(s,"R_cell_cfg");

    cell_reconfig_requests = nv_ring_create(s, 16, 0);
    if (cell_reconfig_requests == 0)
    {
        printf("error on ring %s create\n",s);
        return -1;
    }

    sprintf(s,"R_cell_ctrl");
    cell_ctrl_requests = nv_ring_create(s, 16, 0);
    if (cell_ctrl_requests == 0)
    {
        printf("error on ring %s create\n",s);
        return -1;
    }

    sprintf(s,"R_generic_async_test");
    if ((generic_async_requests = nv_ring_create(s, 16, 0)) == 0)
    {
        printf("error on ring %s create\n",s);
        return -1;
    }

    sprintf(s,"R_cpu_stall");
    simulate_cpu_stall_requests = nv_ring_create(s, 16, 0);
    if (simulate_cpu_stall_requests == 0)
    {
        printf("error on ring %s create\n",s);
        return -1;
    }

    sprintf(s,"R_fapi_delay_test");
    if ((fapi_delay_requests = nv_ring_create(s, 16, 0)) == 0)
    {
        printf("error on ring %s create\n",s);
        return -1;
    }

    sprintf(s,"R_ul_u_plane_drop_requests");
    ul_u_plane_drop_requests = nv_ring_create(s, 32, 0);
    if (ul_u_plane_drop_requests == 0)
    {
        printf("error on ring %s create\n",s);
        return -1;
    }

    sprintf(s,"R_zero_ul_u_plane_requests");
    zero_ul_u_plane_requests = nv_ring_create(s, 32, 0);
    if (zero_ul_u_plane_requests == 0)
    {
        printf("error on ring %s create\n",s);
        return -1;
    }

    sprintf(s,"R_sfn_slot_sync_requests");
    sfn_slot_sync_requests = nv_ring_create(s, 16, 0);
    if (sfn_slot_sync_requests == 0)
    {
        printf("error on ring %s create\n",s);
        return -1;
    }    
    return 0;
}

int CuphyOAM::init_everything()
{
    int ret;

    ret = init_pools();
    if (ret != 0) return ret;

    ret = init_rings();
    if (ret != 0) return ret;

    ret = init_cell_config_ring();
    if (ret != 0) return ret;

    return init_rpc_server();
}

int CuphyOAM::shutdown()
{
    CuphyOAMRpc *rpc = reinterpret_cast<CuphyOAMRpc*>(pRpc);
    if (rpc != nullptr) rpc->server->Shutdown();
    return 0;
}

int CuphyOAM::wait_shutdown()
{
    CuphyOAMRpc *rpc = reinterpret_cast<CuphyOAMRpc*>(pRpc);
    if (rpc != nullptr) rpc->server->Wait();
    return 0;
}


int CuphyOAM::put_generic_async_cmd(CuphyOAMGenericAsyncCmd* cmd)
{
    return nv_ring_enqueue(generic_async_requests, cmd);
}

int CuphyOAM::free_generic_async_cmd(CuphyOAMGenericAsyncCmd* cmd)
{
    free(cmd);
    return 0;
}

CuphyOAMGenericAsyncCmd* CuphyOAM::get_generic_async_cmd()
{
    CuphyOAMGenericAsyncCmd* cmd;
    if (nv_ring_dequeue(generic_async_requests, reinterpret_cast<void**>(&cmd)))
    {
        return nullptr;
    }

    return cmd;
}

int CuphyOAM::put_cell_config(CuphyOAMCellConfig* config)
{
    return nv_ring_enqueue(cell_reconfig_requests, config);
}

int CuphyOAM::free_cell_config(CuphyOAMCellConfig* config)
{
    free(config);
    return 0;
}

CuphyOAMCellConfig* CuphyOAM::get_cell_config()
{
    CuphyOAMCellConfig* config;
    if (nv_ring_dequeue(cell_reconfig_requests, reinterpret_cast<void**>(&config)))
    {
        return nullptr;
    }

    return config;
}

int CuphyOAM::put_cell_ctrl_cmd(CuphyOAMCellCtrlCmd* config)
{
    return nv_ring_enqueue(cell_ctrl_requests, config);
}

int CuphyOAM::free_cell_ctrl_cmd(CuphyOAMCellCtrlCmd* config)
{
    free(config);
    return 0;
}

int CuphyOAM::put_sfn_slot_sync_cmd(CuphyOAMSfnSlotSyncCmd* config)
{
    return nv_ring_enqueue(sfn_slot_sync_requests, config);
}

int CuphyOAM::free_sfn_slot_sync_cmd(CuphyOAMSfnSlotSyncCmd* config)
{
    free(config);
    return 0;
}


int CuphyOAM::put_simulated_cpu_stall_cmd(SimulatedCPUStallCmd* cmd)
{
    return nv_ring_enqueue(simulate_cpu_stall_requests, cmd);
}

int CuphyOAM::free_simulated_cpu_stall_cmd(SimulatedCPUStallCmd* cmd)
{
    free(cmd);
    return 0;
}

SimulatedCPUStallCmd* CuphyOAM::get_simulated_cpu_stall_cmd(int32_t thread_id,int32_t task_id)
{
    SimulatedCPUStallCmd* cmd = nullptr;
    for (int i = 0; i < nv_ring_count(simulate_cpu_stall_requests); i++)
    {
        if (nv_ring_dequeue(simulate_cpu_stall_requests, reinterpret_cast<void**>(&cmd)))
        {
            // queue is empty, return null
            return nullptr;
        }

        if (cmd == nullptr)
        {
            return nullptr;
        }

        if ((cmd->thread_id == thread_id) && (task_id==-1 || cmd->task_id==task_id))
        {
            // thread_id matches, return
            return cmd;
        }
        else
        {
            // thread_id doesn't match, enqueue back to the tail
            nv_ring_enqueue(simulate_cpu_stall_requests, cmd);
            cmd = nullptr;
        }
    }

    return nullptr;
}




bool simulated_cpu_stall_checkpoint(int32_t thread_id, int32_t task_id)
{
        CuphyOAM *oam = CuphyOAM::getInstance();
        SimulatedCPUStallCmd *cpu_stall_cmd;
        SimulatedCPUStallCmd  cpu_stall_cmd_insert;
        bool res = false;
        while ((cpu_stall_cmd = oam->get_simulated_cpu_stall_cmd(thread_id,task_id)) != nullptr)
        {
            if(cpu_stall_cmd->usleep_duration <= 0)
            {
                continue;
            }
            if(thread_id == L2A_MSG_THREAD)
            {
                printf("Simulate CPU Stall on L2A msg thread for %d us\n", cpu_stall_cmd->usleep_duration);
            }
            else if(thread_id == L2A_TICK_THREAD)
            {
                printf("Simulate CPU Stall on L2A tick thread for %d us\n", cpu_stall_cmd->usleep_duration);
            }
            else if(thread_id == CUPHYDRIVER_DL_WORKER_THREAD)
            {
                printf("Simulate CPU Stall on DL worker thread task id %d for %d us\n",cpu_stall_cmd->task_id,cpu_stall_cmd->usleep_duration);
            }
            else if(thread_id == CUPHYDRIVER_UL_WORKER_THREAD)
            {
                printf("Simulate CPU Stall on UL worker thread task id %d for %d us\n",cpu_stall_cmd->task_id,cpu_stall_cmd->usleep_duration);
            }
            usleep(cpu_stall_cmd->usleep_duration);
            oam->free_simulated_cpu_stall_cmd(cpu_stall_cmd);
            res = true;
        }
        return res;
}

CuphyOAMCellCtrlCmd* CuphyOAM::get_cell_ctrl_cmd()
{
    CuphyOAMCellCtrlCmd* config;
    if (nv_ring_dequeue(cell_ctrl_requests, reinterpret_cast<void**>(&config)))
    {
        return nullptr;
    }

    return config;
}

int CuphyOAM::put_fapi_delay_cmd(CuphyOAMFapiDelayCmd* cmd)
{
    return nv_ring_enqueue(fapi_delay_requests, cmd);
}

int CuphyOAM::free_fapi_delay_cmd(CuphyOAMFapiDelayCmd* cmd)
{
    free(cmd);
    return 0;
}

CuphyOAMFapiDelayCmd* CuphyOAM::get_fapi_delay_cmd()
{
    CuphyOAMFapiDelayCmd* cmd;
    if (nv_ring_dequeue(fapi_delay_requests, reinterpret_cast<void**>(&cmd)))
    {
        return nullptr;
    }

    return cmd;
}

int CuphyOAM::put_ul_u_plane_drop_cmd(CuphyOAMULUPlaneDropCmd* cmd)
{
    return nv_ring_enqueue(ul_u_plane_drop_requests, cmd);
}

int CuphyOAM::free_ul_u_plane_drop_cmd(CuphyOAMULUPlaneDropCmd* cmd)
{
    free(cmd);
    return 0;
}

CuphyOAMULUPlaneDropCmd* CuphyOAM::get_ul_u_plane_drop_cmd()
{
    CuphyOAMULUPlaneDropCmd* cmd;
    if (nv_ring_dequeue(ul_u_plane_drop_requests, reinterpret_cast<void**>(&cmd)))
    {
        return nullptr;
    }

    return cmd;
}

CuphyOAMSfnSlotSyncCmd* CuphyOAM::get_sfn_slot_sync_cmd()
{
    CuphyOAMSfnSlotSyncCmd* cmd;
    if (nv_ring_dequeue(sfn_slot_sync_requests, reinterpret_cast<void**>(&cmd)))
    {
        return nullptr;
    }

    return cmd;
}

int CuphyOAM::put_zero_u_plane_request(CuphyOAMZeroUplaneRequest* cmd)
{
    return nv_ring_enqueue(zero_ul_u_plane_requests, cmd);
}

int CuphyOAM::free_zero_u_plane_request(CuphyOAMZeroUplaneRequest* cmd)
{
    free(cmd);
    return 0;
}

CuphyOAMZeroUplaneRequest* CuphyOAM::get_zero_u_plane_request()
{
    CuphyOAMZeroUplaneRequest* cmd;
    if (nv_ring_dequeue(zero_ul_u_plane_requests, reinterpret_cast<void**>(&cmd)))
    {
        return nullptr;
    }

    return cmd;
}

CuphyOAMCellConfig* CuphyOAM::get_cell_config(int32_t cell_id)
{
    CuphyOAMCellConfig* config = nullptr;
    for (int i = 0; i < nv_ring_count(cell_reconfig_requests); i++)
    {
        if (nv_ring_dequeue(cell_reconfig_requests, reinterpret_cast<void**>(&config)))
        {
            // queue is empty, return null
            return nullptr;
        }

        if (config == nullptr)
        {
            return nullptr;
        }

        if (config->cell_id == cell_id)
        {
            // cell_id matches, return
            return config;
        }
        else
        {
            // cell_id doesn't match, enqueue back to the tail
            nv_ring_enqueue(cell_reconfig_requests, config);
            config = nullptr;
        }
    }

    return nullptr;
}

#ifdef CUPHYOAM_WIP
cuphyoam_pool_t* CuphyOAM::getCtrlPool()
{
    return p;
}

cuphyoam_queue_t* CuphyOAM::getCtrlQueue(int coreId, bool rxNotTx)
{
    cuphyoam_queue_t* q;

    if (rxNotTx)
    {
        q = q_rx[coreId];
    }
    else
    {
        q = q_tx[coreId];
    }

    return q;
}
#endif

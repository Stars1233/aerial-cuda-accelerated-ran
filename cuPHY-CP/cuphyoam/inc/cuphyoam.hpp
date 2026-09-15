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

#pragma once
//#define CUPHYOAM_WIP

#include <atomic>
#include <mutex>
#include <unistd.h>
#include <unordered_map>
#include <functional>
#include <string>

#include "nv_ring.h"

#ifdef CUPHYOAM_WIP
using cuphyoam_queue_t = struct rte_ring;
using cuphyoam_pool_t = struct rte_mempool;
#endif

class CuphyOAMRpc;

const int MAX_CORES = 128;

static constexpr uint32_t L2A_MSG_THREAD = 0;
static constexpr uint32_t L2A_TICK_THREAD = 1;
static constexpr uint32_t CUPHYDRIVER_DL_WORKER_THREAD = 2;
static constexpr uint32_t CUPHYDRIVER_UL_WORKER_THREAD = 3;

struct CuphyOAMGenericAsyncCmd
{
    int32_t cmd_id;
    int32_t param_int1;
    int32_t param_int2;
    std::string param_str;
};

struct CuphyOAMCellConfig
{
    int32_t cell_id;

    bool update_network_cfg;
    std::string dst_mac_addr;
    uint16_t vlan_tci;

    bool update_cell_attenuation_cfg;
    float attenuation_dB;

    bool multi_attrs_cfg;
    std::unordered_map<std::string, double> attrs;
    std::unordered_map<std::string, int> res;

    bool eaxcid_update;
    std::unordered_map<int, std::vector<uint16_t>> ch_eaxcid_map;

    CuphyOAMCellConfig() : update_network_cfg(false), update_cell_attenuation_cfg(false), multi_attrs_cfg(false), eaxcid_update(false) {}
};

struct CuphyOAMCellCtrlCmd
{
  int32_t cell_id;
  int32_t cell_ctrl_cmd;
  int32_t target_cell_id;
};

struct SimulatedCPUStallCmd
{
  int32_t thread_id;
  int32_t task_id;
  int32_t usleep_duration;
};

struct CuphyOAMFapiDelayCmd
{
    int32_t cell_id;
    int32_t slot;
    int32_t fapi_mask;
    int32_t delay_us;
};

struct CuphyOAMULUPlaneDropCmd
{
  int32_t cell_id;
  int32_t channel_id;
  int32_t drop_rate;
  int32_t single_drop;
  int32_t drop_slot;
  int32_t frame_id;
  int32_t subframe_id;
  int32_t slot_id;
};

struct CuphyOAMSfnSlotSyncCmd
{
  int32_t sync_done;
  uint64_t time_anchor;
};

struct CuphyOAMZeroUplaneRequest
{
  int32_t cell_id;
  int32_t use_cell_mask;
  uint64_t cell_mask;
  int32_t channel_id;
};

typedef struct
{
    // De-initiate and destroy the nv_ipc_t instance
    int (*update_cell_attenuation)(int32_t mplane_id, float attenuation_db);
    void (*cell_multi_attr_update)(uint16_t mplane_id, std::unordered_map<std::string, double>& attrs);

} CuphyOAMCallback;

using CellMultiAttriUpdateCallBackFn = std::function<void(uint16_t mplane_id, std::unordered_map<std::string, double>& attrs, std::unordered_map<std::string, int>& res)>;
using CelleAxCIdsUpdateCallBackFn = std::function<void(uint16_t mplane_id, std::unordered_map<int, std::vector<uint16_t>>& eaxcids_ch_map)>;

class CuphyOAM
{
    public:
        static CuphyOAM* getInstance();

        virtual int init_rpc_server();
        virtual int init_pools();
        virtual int init_rings();
        virtual int init_cell_config_ring();
        virtual int init_everything();

        static void set_server_address(std::string addr);
        static std::string get_server_address();

        virtual int shutdown();
        virtual int wait_shutdown();

        virtual int put_generic_async_cmd(CuphyOAMGenericAsyncCmd* cmd);
        virtual int free_generic_async_cmd(CuphyOAMGenericAsyncCmd* cmd);
        virtual CuphyOAMGenericAsyncCmd* get_generic_async_cmd();

        virtual int put_cell_config(CuphyOAMCellConfig* config);
        virtual int free_cell_config(CuphyOAMCellConfig* config);
        virtual CuphyOAMCellConfig* get_cell_config();
        virtual CuphyOAMCellConfig* get_cell_config(int32_t cell_id);

        virtual int put_cell_ctrl_cmd(CuphyOAMCellCtrlCmd* config);
        virtual int free_cell_ctrl_cmd(CuphyOAMCellCtrlCmd* config);
        virtual CuphyOAMCellCtrlCmd* get_cell_ctrl_cmd();

        virtual int put_sfn_slot_sync_cmd(CuphyOAMSfnSlotSyncCmd* config);
        virtual int free_sfn_slot_sync_cmd(CuphyOAMSfnSlotSyncCmd* config);
        virtual CuphyOAMSfnSlotSyncCmd* get_sfn_slot_sync_cmd();

        virtual int put_simulated_cpu_stall_cmd(SimulatedCPUStallCmd* cmd);
        virtual int free_simulated_cpu_stall_cmd(SimulatedCPUStallCmd* cmd);
        virtual SimulatedCPUStallCmd*  get_simulated_cpu_stall_cmd(int32_t thread_id,int32_t task_id);

        virtual int put_fapi_delay_cmd(CuphyOAMFapiDelayCmd* cmd);
        virtual int free_fapi_delay_cmd(CuphyOAMFapiDelayCmd* cmd);
        virtual CuphyOAMFapiDelayCmd* get_fapi_delay_cmd();

        virtual int put_ul_u_plane_drop_cmd(CuphyOAMULUPlaneDropCmd* cmd);
        virtual int free_ul_u_plane_drop_cmd(CuphyOAMULUPlaneDropCmd* cmd);
        virtual CuphyOAMULUPlaneDropCmd* get_ul_u_plane_drop_cmd();

        virtual int put_zero_u_plane_request(CuphyOAMZeroUplaneRequest* cmd);
        virtual int free_zero_u_plane_request(CuphyOAMZeroUplaneRequest* cmd);
        virtual CuphyOAMZeroUplaneRequest* get_zero_u_plane_request();

        virtual void notifyClient(int clientId, std::string msg);
        virtual void notifyAll(std::string msg);

#ifdef CUPHYOAM_WIP
        cuphyoam_pool_t* getCtrlPool();
        cuphyoam_queue_t* getCtrlQueue(int coreId, bool rxNotTx);
#endif

        std::atomic<uint32_t> status_sfn_slot{0};
        std::atomic<uint32_t> core_active[MAX_CORES];
        std::atomic<uint32_t> cpu_utilization_x1000[MAX_CORES];
        std::mutex puschH5DumpMutex;
        std::atomic<bool> puschH5dumpNextCrc{false};
        std::atomic<bool> puschH5dumpInProgress{false};

        std::atomic<bool> ul_pcap_enabled{false};
        std::atomic<uint64_t> ul_pcap_arm_cell_bitmask{0};
        std::atomic<uint64_t> ul_pcap_flush_cell_bitmask{0};

        CuphyOAMCallback callback;
        CellMultiAttriUpdateCallBackFn cell_multi_attri_update_callback;
        CelleAxCIdsUpdateCallBackFn cell_eaxcids_update_callback;

    private:
        CuphyOAM() {
            pRpc = nullptr;
            cell_reconfig_requests = nullptr;
            cell_ctrl_requests = nullptr;
            simulate_cpu_stall_requests = nullptr;
            ul_u_plane_drop_requests = nullptr;
            zero_ul_u_plane_requests = nullptr;
            callback.update_cell_attenuation = nullptr;
        };

        virtual ~CuphyOAM() {};

        static CuphyOAM* instance;

        static std::string server_addr;

        void* pRpc;

        nv_ring* generic_async_requests = nullptr;
        nv_ring* cell_reconfig_requests;
        nv_ring* cell_ctrl_requests;
        nv_ring* sfn_slot_sync_requests;
        nv_ring* simulate_cpu_stall_requests;
        nv_ring* fapi_delay_requests = nullptr;
        nv_ring* ul_u_plane_drop_requests;
        nv_ring* zero_ul_u_plane_requests;

#ifdef CUPHYOAM_WIP
        cuphyoam_pool_t *p;
        cuphyoam_queue_t* q_rx[MAX_CORES];
        cuphyoam_queue_t* q_tx[MAX_CORES];
#endif
};

bool simulated_cpu_stall_checkpoint(int32_t thread_id,int32_t task_id);

void* cus_conn_mgr_thread_func(void *arg);

#ifdef CUPHYOAM_WIP
inline int cuphyoam_enqueue(cuphyoam_queue_t *q, void* msg)
{
   return rte_ring_enqueue(q, msg);
}

inline int cuphyoam_dequeue(cuphyoam_queue_t *q, void** msg)
{
   return rte_ring_dequeue(q, msg);
}

inline int cuphyoam_queue_isEmpty(cuphyoam_queue_t *q)
{
   return rte_ring_empty(q);
}

inline int cuphyoam_msg_alloc(cuphyoam_pool_t *p, void** msg)
{
   return rte_mempool_get(p, msg);
}

inline int cuphyoam_msg_free(cuphyoam_pool_t *p, void* msg)
{
   rte_mempool_put(p, msg);
   return 0;
}
#endif

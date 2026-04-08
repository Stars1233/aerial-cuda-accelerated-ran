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

// #define _GNU_SOURCE
#include <stdio.h>
#include <random>
#include <string.h>
#include <stdint.h>
#include <stdlib.h>
#include <errno.h>
#include <pthread.h>
#include <fcntl.h>
#include <inttypes.h>
#include <stdarg.h>
#include <errno.h>
#include <unistd.h>
#include <termios.h>
#include <stdatomic.h>
#include <sys/queue.h>
#include <sys/epoll.h>

#include "nv_utils.h"
#include "nv_ipc.h"
#include "nv_ipc_utils.h"
#include "cu_mac_api.h"

/*
* YAML config path: relative path based on this app build folder
* (1) App binary path: build/cuMAC-CP/examples/L2IntegrationExample/l2_test
* (2) YAML file path: ./cuMAC-CP/examples/L2IntegrationExample/l2_nvipc.yaml
*/ 
#define YAML_CONFIG_PATH "./cuMAC-CP/examples/L2IntegrationExample/l2_nvipc.yaml"

#define MAX_PATH_LEN (1024)

typedef std::pair<uint16_t, uint8_t> scheduled_ue_lc; // <UE index, LC index> for storing scheduling solution per slot

// Log TAG configured in nvlog
static int TAG = (NVLOG_TAG_BASE_NVIPC + 0);

nv_ipc_t* ipc = NULL;

#define SENDER_THREAD_CORE (8)

#define RECV_THREAD_CORE (9)

////////////////////////////////////////////////////////////////////////
// Build a TX message
static int l2_build_tx_msg(nv_ipc_t* ipc, nv_ipc_msg_t* nvipc_buf, uint32_t slot, uint32_t sfn, uint32_t num_ue, PFM_UE_INFO* test_data)
{
    if(ipc == NULL || nvipc_buf == NULL)
    {
        NVLOGE(TAG, AERIAL_NVIPC_API_EVENT, "%s: ipc or msg buffer is NULL", __func__);
        return -1;
    }

    cumac_pfm_tti_req_t* req = (cumac_pfm_tti_req_t*)nvipc_buf->msg_buf;
    uint8_t* data_buf = (uint8_t*)nvipc_buf->data_buf;

    req->header.type_id = CUMAC_SCH_TTI_REQUEST;
    req->sfn = sfn;
    req->slot = slot;
    req->num_ue = num_ue;
    for (int i = 0; i < 10; i++) {
        req->num_output_sorted_lc[i] = MAX_NUM_OUTPUT_SORTED_LC_PER_QOS;
    }

    // Copy data which need to be sent to GPU to the nvipc_buf->data_buf
    uint32_t offset = 0;
    uint32_t data_size = 0;

    // Set the offset and copy PFM_UE_INFO data to the data buffer
    req->offset_ue_info_arr = offset;
    data_size = sizeof(PFM_UE_INFO) * num_ue;
    memcpy(data_buf + offset, test_data, data_size);
    offset += data_size;

    // May have more data structures to copy

    // Update the msg_len and data_len of the NVIPC message header
    nvipc_buf->msg_id = CUMAC_SCH_TTI_REQUEST;
    nvipc_buf->cell_id = 0;
    nvipc_buf->msg_len = sizeof(cumac_pfm_tti_req_t);
    nvipc_buf->data_len = data_size;

    return 0;
}

static int test_l2_send_slot(nv_ipc_t* ipc, uint32_t slot, uint32_t sfn, uint32_t num_ue, PFM_UE_INFO* test_data)
{
    struct timespec nvipcAlloc_start, nvipcAlloc_end, buildMsg_start, buildMsg_end, sendMsg_start, sendMsg_end;

    clock_gettime(CLOCK_REALTIME, &nvipcAlloc_start);
    nv_ipc_msg_t send_msg;

    // send_msg.msg_buf will be allocated by default. Set data_pool to get send_msg.data_buf
    send_msg.data_pool = NV_IPC_MEMPOOL_CPU_DATA;

    // Allocate NVIPC buffer which contains MSG part and DATA part
    if(ipc->tx_allocate(ipc, &send_msg, 0) != 0)
    {
        NVLOGE(TAG, AERIAL_NVIPC_API_EVENT, "%s error: NVIPC memory pool is full", __func__);
        return -1;
    }
    clock_gettime(CLOCK_REALTIME, &nvipcAlloc_end);

    // Build the message
    clock_gettime(CLOCK_REALTIME, &buildMsg_start);
    if(l2_build_tx_msg(ipc, &send_msg, slot, sfn, num_ue, test_data) < 0)
    {
        NVLOGE(TAG, AERIAL_NVIPC_API_EVENT, "%s error: build message failed", __func__);
        // Free the buffer from sender side for error case. Normally will not happen.
        ipc->tx_release(ipc, &send_msg);
        return -1;
    }
    clock_gettime(CLOCK_REALTIME, &buildMsg_end);

    // Send the message
    clock_gettime(CLOCK_REALTIME, &sendMsg_start);
    if(ipc->tx_send_msg(ipc, &send_msg) < 0)
    {
        NVLOGE(TAG, AERIAL_NVIPC_API_EVENT, "%s error: send message failed", __func__);
        // Free the buffer from sender side for error case. Normally will not happen.
        ipc->tx_release(ipc, &send_msg);
        return -1;
    }

    if(ipc->tx_tti_sem_post(ipc) < 0)
    {
        NVLOGE(TAG, AERIAL_NVIPC_API_EVENT, "%s error: tx notification failed", __func__);
        return -1;
    }
    clock_gettime(CLOCK_REALTIME, &sendMsg_end);

    int64_t nvipcAlloc_duration = nvlog_timespec_interval(&nvipcAlloc_start, &nvipcAlloc_end);
    int64_t buildMsg_duration = nvlog_timespec_interval(&buildMsg_start, &buildMsg_end);
    int64_t sendMsg_duration = nvlog_timespec_interval(&sendMsg_start, &sendMsg_end);

    NVLOGI(TAG, "L2 NVIPC shared buffer allocation duration: %ld ns", nvipcAlloc_duration);
    NVLOGI(TAG, "L2 NVIPC message build duration: %ld ns", buildMsg_duration);
    NVLOGI(TAG, "L2 NVIPC message send duration: %ld ns", sendMsg_duration);

    printf("L2 NVIPC shared buffer allocation duration: %ld ns\n", nvipcAlloc_duration);
    printf("L2 NVIPC message build duration: %ld ns\n", buildMsg_duration);
    printf("L2 NVIPC message send duration: %ld ns\n", sendMsg_duration);

    return 0;
}

static void* l2_blocking_recv_task(void* arg)
{
    // Set thread name, max string length < 16
    pthread_setname_np(pthread_self(), "l2_recv");
    // Set thread schedule policy to SCHED_FIFO and set priority to 80
    nv_set_sched_fifo_priority(80);
    // Set the thread CPU core
    nv_assign_thread_cpu_core(RECV_THREAD_CORE);

    int num_recv_msg = 0;

    while (num_recv_msg < NUM_TIME_SLOTS) {
        NVLOGI(TAG, "%s: wait for incoming messages notification ...", __func__);

        // Wait for incoming messages notification
        ipc->rx_tti_sem_wait(ipc);

        nv_ipc_msg_t recv_msg;

        // Dequeue the incoming NVIPC message
        while (ipc->rx_recv_msg(ipc, &recv_msg) >= 0) {
            num_recv_msg++; // received a message, increment the counter

            cumac_pfm_tti_resp_t* resp = (cumac_pfm_tti_resp_t*)recv_msg.msg_buf;
            uint8_t* data_buf = (uint8_t*)recv_msg.data_buf;

            NVLOGC(TAG, "RECV: SFN %u.%u msg_id=0x%02X %s msg_len=%d data_len=%d",
                    resp->sfn, resp->slot, recv_msg.msg_id, get_cumac_msg_name(recv_msg.msg_id), recv_msg.msg_len, recv_msg.data_len);

            // Handle the cuMAC scheduling response
            PFM_DL_OUTPUT_INFO* dl_gbr_critical = (PFM_DL_OUTPUT_INFO*) (data_buf + resp->offset_output_sorted_lc[0]);
            PFM_DL_OUTPUT_INFO* dl_gbr_non_critical = (PFM_DL_OUTPUT_INFO*) (data_buf + resp->offset_output_sorted_lc[1]);
            PFM_DL_OUTPUT_INFO* dl_ngbr_critical = (PFM_DL_OUTPUT_INFO*) (data_buf + resp->offset_output_sorted_lc[2]);
            PFM_DL_OUTPUT_INFO* dl_ngbr_non_critical = (PFM_DL_OUTPUT_INFO*) (data_buf + resp->offset_output_sorted_lc[3]);
            PFM_DL_OUTPUT_INFO* dl_mbr_non_critical = (PFM_DL_OUTPUT_INFO*) (data_buf + resp->offset_output_sorted_lc[4]);

            PFM_UL_OUTPUT_INFO* ul_gbr_critical = (PFM_UL_OUTPUT_INFO*) (data_buf + resp->offset_output_sorted_lc[5]);
            PFM_UL_OUTPUT_INFO* ul_gbr_non_critical = (PFM_UL_OUTPUT_INFO*) (data_buf + resp->offset_output_sorted_lc[6]);
            PFM_UL_OUTPUT_INFO* ul_ngbr_critical = (PFM_UL_OUTPUT_INFO*) (data_buf + resp->offset_output_sorted_lc[7]);
            PFM_UL_OUTPUT_INFO* ul_ngbr_non_critical = (PFM_UL_OUTPUT_INFO*) (data_buf + resp->offset_output_sorted_lc[8]);
            PFM_UL_OUTPUT_INFO* ul_mbr_non_critical = (PFM_UL_OUTPUT_INFO*) (data_buf + resp->offset_output_sorted_lc[9]);

            uint32_t num_dl_gbr_critical_ue = resp->num_output_sorted_lc[0];
            uint32_t num_dl_gbr_non_critical_ue = resp->num_output_sorted_lc[1];
            uint32_t num_dl_ngbr_critical_ue = resp->num_output_sorted_lc[2];
            uint32_t num_dl_ngbr_non_critical_ue = resp->num_output_sorted_lc[3];
            uint32_t num_dl_mbr_non_critical_ue = resp->num_output_sorted_lc[4];

            uint32_t num_ul_gbr_critical_ue = resp->num_output_sorted_lc[5];
            uint32_t num_ul_gbr_non_critical_ue = resp->num_output_sorted_lc[6];
            uint32_t num_ul_ngbr_critical_ue = resp->num_output_sorted_lc[7];
            uint32_t num_ul_ngbr_non_critical_ue = resp->num_output_sorted_lc[8];
            uint32_t num_ul_mbr_non_critical_ue = resp->num_output_sorted_lc[9];

            // TODO: apply the cuMAC scheduling response

            // Release the NVIPC message buffer
            ipc->rx_release(ipc, &recv_msg);
        }
    }

    return NULL;
}

static void get_next_slot_timespec(struct timespec* ts, uint64_t interval_nsec)
{
    ts->tv_nsec += interval_nsec;
    while (ts->tv_nsec >= 1E9)
    {
        ts->tv_nsec -= 1E9;
        ts->tv_sec++;
    }
}

void prepare_slot_data_dl(std::vector<PFM_UE_INFO>& test_data, 
                          std::vector<scheduled_ue_lc>& scheduled_ue_dl_lc_list, 
                          const int slot_idx)
{
    // initialize random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> distrib(0, INT_MAX);

    if (slot_idx == 0) {
        for (int i = 0; i < MAX_NUM_SCHEDULED_UE; i++) {
            scheduled_ue_dl_lc_list[i].first = i;
            scheduled_ue_dl_lc_list[i].second = distrib(gen) % MAX_NUM_LC;
        }
        for (int uIdx = 0; uIdx < MAX_NUM_UE; uIdx++) {
            for (int lcIdx = 0; lcIdx < MAX_NUM_LC; lcIdx++) {
                PFM_DL_LC_INFO* tempPtr = &(test_data[uIdx].dl_lc_info[lcIdx]);
                tempPtr->pfm = 0;
                tempPtr->ravg = 1; 
                tempPtr->flags = 0x03; // valid, reset ravg to 1.0
                tempPtr->qos_type = distrib(gen) % 5;
                tempPtr->tbs_scheduled = 0;
            }

            test_data[uIdx].num_dl_lcs = MAX_NUM_LC;
            test_data[uIdx].ambr = 50000000;
            test_data[uIdx].rcurrent_dl = 1000000 + distrib(gen) % 20000000;
            test_data[uIdx].rnti = uIdx;
            test_data[uIdx].num_layers_dl = distrib(gen) % 2 + 1;
            test_data[uIdx].flags = 0x07; // valid, scheduled_dl, scheduled_ul
            test_data[uIdx].carrier_id = 0; 
        }
    } else {
        for (int uIdx = 0; uIdx < MAX_NUM_UE; uIdx++) {
            for (int lcIdx = 0; lcIdx < MAX_NUM_LC; lcIdx++) {
                test_data[uIdx].dl_lc_info[lcIdx].tbs_scheduled = 0;
            }
        }

        for (int i = 0; i < MAX_NUM_SCHEDULED_UE; i++) {
            uint16_t scheduled_ue = scheduled_ue_dl_lc_list[i].first;
            uint8_t scheduled_lc = scheduled_ue_dl_lc_list[i].second;

            test_data[scheduled_ue].dl_lc_info[scheduled_lc].tbs_scheduled = floor(static_cast<float>(test_data[scheduled_ue].rcurrent_dl) * SLOT_DURATION);

            scheduled_ue_dl_lc_list[i].first = (scheduled_ue + MAX_NUM_SCHEDULED_UE) % MAX_NUM_UE; 
            scheduled_ue_dl_lc_list[i].second = distrib(gen) % MAX_NUM_LC;
        }

        for (int uIdx = 0; uIdx < MAX_NUM_UE; uIdx++) {
            test_data[uIdx].rcurrent_dl = 1000000 + distrib(gen) % 20000000;
            test_data[uIdx].num_layers_dl = distrib(gen) % 2 + 1;
            for (int lcIdx = 0; lcIdx < MAX_NUM_LC; lcIdx++) {
                PFM_DL_LC_INFO* tempPtr = &(test_data[uIdx].dl_lc_info[lcIdx]);
                tempPtr->flags = 0x01; // valid, not reseting ravg
            }
        }
    }
}

void prepare_slot_data_ul(std::vector<PFM_UE_INFO>& test_data, 
                          std::vector<scheduled_ue_lc>& scheduled_ue_ul_lcg_list, 
                          const int slot_idx)

{
    // initialize random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> distrib(0, INT_MAX);

    if (slot_idx == 0) {
        for (int i = 0; i < MAX_NUM_SCHEDULED_UE; i++) {
            scheduled_ue_ul_lcg_list[i].first = i;
            scheduled_ue_ul_lcg_list[i].second = distrib(gen) % MAX_NUM_LCG;
        }
        for (int uIdx = 0; uIdx < MAX_NUM_UE; uIdx++) {
            for (int lcgIdx = 0; lcgIdx < MAX_NUM_LCG; lcgIdx++) {
                PFM_UL_LCG_INFO* tempPtr = &(test_data[uIdx].ul_lcg_info[lcgIdx]);
                tempPtr->pfm = 0;
                tempPtr->ravg = 1;
                tempPtr->flags = 0x03; // valid, reset ravg to 1.0
                tempPtr->qos_type = distrib(gen) % 5;
                tempPtr->tbs_scheduled = 0;
            }

            test_data[uIdx].num_ul_lcgs = MAX_NUM_LCG;
            test_data[uIdx].ambr = 20000000;
            test_data[uIdx].rcurrent_ul = 500000 + distrib(gen) % 10000000;
            test_data[uIdx].rnti = uIdx;
            test_data[uIdx].num_layers_ul = distrib(gen) % 2 + 1;
            test_data[uIdx].flags = 0x07; // valid, scheduled_dl, scheduled_ul
            test_data[uIdx].carrier_id = 0; 
        }
    } else {
        for (int uIdx = 0; uIdx < MAX_NUM_UE; uIdx++) {
            for (int lcgIdx = 0; lcgIdx < MAX_NUM_LCG; lcgIdx++) {
                test_data[uIdx].ul_lcg_info[lcgIdx].tbs_scheduled = 0;
            }
        }

        for (int i = 0; i < MAX_NUM_SCHEDULED_UE; i++) {
            uint16_t scheduled_ue = scheduled_ue_ul_lcg_list[i].first;
            uint8_t scheduled_lc = scheduled_ue_ul_lcg_list[i].second;

            test_data[scheduled_ue].ul_lcg_info[scheduled_lc].tbs_scheduled = floor(static_cast<float>(test_data[scheduled_ue].rcurrent_ul) * SLOT_DURATION);

            scheduled_ue_ul_lcg_list[i].first = (scheduled_ue + MAX_NUM_SCHEDULED_UE) % MAX_NUM_UE; 
            scheduled_ue_ul_lcg_list[i].second = distrib(gen) % MAX_NUM_LCG;
        }

        for (int uIdx = 0; uIdx < MAX_NUM_UE; uIdx++) {
            test_data[uIdx].rcurrent_ul = 500000 + distrib(gen) % 10000000;
            test_data[uIdx].num_layers_ul = distrib(gen) % 2 + 1;
            for (int lcgIdx = 0; lcgIdx < MAX_NUM_LCG; lcgIdx++) {
                PFM_UL_LCG_INFO* tempPtr = &(test_data[uIdx].ul_lcg_info[lcgIdx]);
                tempPtr->flags = 0x01; // valid, not reseting ravg
            }   
        }
    }
}

int main(int argc, char** argv)
{

    NVLOGC(TAG, "YAML file: %s", YAML_CONFIG_PATH);

    // Load nvipc configuration from YAML file
    nv_ipc_config_t config;
    load_nv_ipc_yaml_config(&config, YAML_CONFIG_PATH, NV_IPC_MODULE_SECONDARY);    

    // Initialize NVIPC interface and connect to the cuMAC-CP
    if ((ipc = create_nv_ipc_interface(&config)) == NULL)
    {
        NVLOGE(TAG, AERIAL_NVIPC_API_EVENT, "%s: create IPC interface failed", __func__);
        return -1;
    }

    // Sleep 1 seconds for NVIPC connection
    usleep(1000000);

    // Create L2 receiver thread
    pthread_t recv_thread_id;
    int ret = pthread_create(&recv_thread_id, NULL, l2_blocking_recv_task, NULL);
    if(ret != 0)
    {
        NVLOGE(TAG, AERIAL_NVIPC_API_EVENT, "%s failed, ret=%d", __func__, ret);
        return -1;
    }
  
    // test data
    std::vector<PFM_UE_INFO> test_data(MAX_NUM_UE);
    // scheduled UE list
    std::vector<scheduled_ue_lc> scheduled_ue_dl_lc_list(MAX_NUM_SCHEDULED_UE);
    std::vector<scheduled_ue_lc> scheduled_ue_ul_lcg_list(MAX_NUM_SCHEDULED_UE);

    // Inital SFN/SLOT = 0.0
    uint16_t sfn = 0, slot = 0;

    // Align the first slot timestamp to the next second
    struct timespec ts_slot, ts_remain;
    clock_gettime(CLOCK_REALTIME, &ts_slot);
    ts_slot.tv_nsec = 0;
    ts_slot.tv_sec ++;

    // Set thread name, max string length < 16
    pthread_setname_np(pthread_self(), "l2_sender");
    // Set thread schedule policy to SCHED_FIFO and set priority to 80
    nv_set_sched_fifo_priority(80);
    // Set the thread CPU core
    nv_assign_thread_cpu_core(SENDER_THREAD_CORE);

    // Main loop or sender thread
    for (int slotIdx = 0; slotIdx < NUM_TIME_SLOTS; slotIdx++) {
        // Sleep to the next slot timestamp
        int ret = clock_nanosleep(CLOCK_REALTIME, TIMER_ABSTIME, &ts_slot, &ts_remain);
        if(ret != 0)
        {
            NVLOGE(TAG, AERIAL_CLOCK_API_EVENT, "clock_nanosleep returned error ret: %d", ret);
        }

        uint32_t num_ue = MAX_NUM_UE;

        // prepare DL and UL data for the current slot
        prepare_slot_data_dl(test_data, scheduled_ue_dl_lc_list, slotIdx);
        prepare_slot_data_ul(test_data, scheduled_ue_ul_lcg_list, slotIdx);

        // Allocate an NVIPC buffer, build the message and send it
        test_l2_send_slot(ipc, slot, sfn, num_ue, test_data.data());

        // Update SFN/SLOT for next slot
        slot++;
        if (slot >= 20)
        {
            slot = 0;
            sfn++;
            if (sfn >= 1024)
            {
                sfn = 0;
            }
        }

        // Update timestamp for next slot
        get_next_slot_timespec(&ts_slot, SLOT_INTERVAL_NS);
    }

    if(recv_thread_id != 0)
    {
        // Cancel the receiver thread
        NVLOGI(TAG, "%s: canceling receiver thread", __func__);
        if(pthread_cancel(recv_thread_id) != 0)
        {
            NVLOGE(TAG, AERIAL_NVIPC_API_EVENT, "%s pthread_cancel failed, stderr=%s", __func__, strerror(errno));
        }

        // Wait for the receiver thread to finish
        NVLOGI(TAG, "%s: waiting for receiver thread to exit", __func__);
        int ret = pthread_join(recv_thread_id, NULL);
        if(ret != 0)
        {
            NVLOGE(TAG, AERIAL_NVIPC_API_EVENT, "%s pthread_join failed, stderr=%s", __func__, strerror(ret));
            return -1;
        }
    }

    return 0;
}
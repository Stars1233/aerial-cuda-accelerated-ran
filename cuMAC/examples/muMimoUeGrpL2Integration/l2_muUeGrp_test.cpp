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

#include "l2_muUeGrp_test.h"
#include <csignal>

// * For simplicity, this L2 stack model uses a single CPU thread for the L2 processing of multiple cells. A real L2 stack implementation may use different CPU threads to handle different cells.

// global variables
int L2_MAIN_THREAD_CORE; // L2 main thread core
int L2_CUMAC_RECV_THREAD_CORE; // L2 cuMAC receiver thread core
int NUM_TIME_SLOTS; // number of time slots
int NUM_CELL; // number of cells
bool PRINT_UE_PAIRING_SOLUTION; // print UE pairing solution

volatile sig_atomic_t g_shutdown = 0;

static void signal_handler(int signum)
{
    g_shutdown = 1;
}

// NVIPC interfaces
nv_ipc_t* ipc_l2_cumac = NULL; // NVIPC interface to the cuMAC-CP
nv_ipc_t* ipc_l2_l1 = NULL; // NVIPC interface to the L1 stack

// L2 work per slot with memory sharing
int l2_work_per_slot_mem_sharing(nv_ipc_t* ipc_l2_l1, nv_ipc_t* ipc_l2_cumac, const sys_param_t& sys_param, std::vector<std::unique_ptr<l2_connected_ue_list_t>>& connected_ue_list_vec, uint32_t slot, uint32_t sfn)
{
    // check if the NVIPC interfaces are valid
    if(ipc_l2_l1 == NULL || ipc_l2_cumac == NULL)
    {
        NVLOGE(MU_TEST_TAG, AERIAL_NVIPC_API_EVENT, "%s: ipc is NULL", __func__);
        return -1;
    }

    // for generating random test data 
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> distrib(0, INT_MAX);
    std::normal_distribution<> normal_distrib(0.0, 1.0); // Normal (Gaussian) distribution with mean 0 and stddev 1

    // L2 to L1 NVIPC TX message
    nv_ipc_msg_t l2_l1_msg;

    l2_l1_msg.data_pool = NV_IPC_MEMPOOL_CPU_DATA;

    // Allocate NVIPC buffer which contains MSG part and DATA part
    if(ipc_l2_l1->tx_allocate(ipc_l2_l1, &(l2_l1_msg), 0) != 0) {
        NVLOGE(MU_TEST_TAG, AERIAL_NVIPC_API_EVENT, "%s error: NVIPC memory pool is full", __func__);
        return -1;
    }

    l2_l1_message_t* l2_l1_msg_ptr;

    l2_l1_msg_ptr = (l2_l1_message_t*) l2_l1_msg.data_buf;
    
    // L2 to cuMAC NVIPC TX message
    nv_ipc_msg_t send_msg[sys_param.num_cell];
    
    std::vector<cumac_muUeGrp_req_info_t*> req_data_vec(sys_param.num_cell);

    for (int cIdx = 0; cIdx < sys_param.num_cell; cIdx++) {
        // send_msg.msg_buf will be allocated by default. Set data_pool to get send_msg.data_buf
        send_msg[cIdx].data_pool = NV_IPC_MEMPOOL_CPU_DATA;

        // Allocate NVIPC buffer which contains MSG part and DATA part
        if(ipc_l2_cumac->tx_allocate(ipc_l2_cumac, &(send_msg[cIdx]), 0) != 0) {
            NVLOGE(MU_TEST_TAG, AERIAL_NVIPC_API_EVENT, "%s error: NVIPC memory pool is full", __func__);
            return -1;
        }

        req_data_vec[cIdx] = (cumac_muUeGrp_req_info_t*) send_msg[cIdx].data_buf;
    }
    
    uint16_t nSrsUes = 0;
    std::vector<uint16_t> num_srsInfo(sys_param.num_cell, 0);

    for (size_t cIdx = 0; cIdx < sys_param.num_cell; cIdx++) {
        req_data_vec[cIdx]->srsInfoMsh = (cumac_muUeGrp_req_srs_info_msh_t*) (req_data_vec[cIdx]->payload);
    }

    char tdd_direction = sys_param.TDD_pattern[slot % sys_param.TDD_pattern.length()];
    if (tdd_direction == 'S') { // S-slot with SRS scheduled UEs
        // determine the number of SRS UEs and SRS info for each cell
        for (size_t cIdx = 0; cIdx < sys_param.num_cell; cIdx++) {
            uint16_t num_connected_srs_ue = connected_ue_list_vec[cIdx]->get_num_connected_srs_ue();
            if (num_connected_srs_ue == 0xFFFF) {
                throw std::runtime_error("Number of connected SRS UEs is not consistent with the number of SRS UEs in the SRS schedule queue in cell " + std::to_string(cIdx));
            }

            num_srsInfo[cIdx] = sys_param.num_srs_ue_per_slot < num_connected_srs_ue ? sys_param.num_srs_ue_per_slot : num_connected_srs_ue;

            nSrsUes += num_srsInfo[cIdx];
        }

        // prepare messages to both L1 and cuMAC
        uint16_t* arr_cell_idx = (uint16_t*) (l2_l1_msg_ptr->arr_usage + nSrsUes);
        uint16_t* arr_rnti = (uint16_t*) (arr_cell_idx + nSrsUes);
        uint16_t* arr_buffer_Idx = (uint16_t*) (arr_rnti + nSrsUes);
        uint16_t* arr_srs_info_idx = (uint16_t*) (arr_buffer_Idx + nSrsUes);

        uint16_t srs_ue_idx = 0;
        
        for (size_t cIdx = 0; cIdx < sys_param.num_cell; cIdx++) {
            for (uint16_t i = 0; i < num_srsInfo[cIdx]; i++) {
                ue_id_rnti_t ue_id_rnti = connected_ue_list_vec[cIdx]->get_next_srs_schedule();

                // prepare message to L1
                l2_l1_msg_ptr->arr_usage[srs_ue_idx] = 0x01; // valid
                arr_cell_idx[srs_ue_idx] = cIdx;
                arr_rnti[srs_ue_idx] = ue_id_rnti.rnti;
                arr_buffer_Idx[srs_ue_idx] = ue_id_rnti.id;
                arr_srs_info_idx[srs_ue_idx] = i;

                srs_ue_idx++;

                // prepare message to cuMAC
                req_data_vec[cIdx]->srsInfoMsh[i].srsWbSnr = std::bit_cast<uint32_t>(5.0f); // dB
                req_data_vec[cIdx]->srsInfoMsh[i].id = ue_id_rnti.id;
                req_data_vec[cIdx]->srsInfoMsh[i].rnti = ue_id_rnti.rnti;
                req_data_vec[cIdx]->srsInfoMsh[i].nUeAnt = sys_param.num_ue_ant_port; 

                connected_ue_list_vec[cIdx]->set_srs_info(ue_id_rnti.rnti, i); 
            }
        }  
    }

    l2_l1_msg_ptr->nSrsUes = nSrsUes;

    // prepare messages to cuMAC
    for (int cIdx = 0; cIdx < sys_param.num_cell; cIdx++) {
        uint16_t num_connected_srs_ue = connected_ue_list_vec[cIdx]->get_num_connected_srs_ue();
        if (num_connected_srs_ue == 0xFFFF) {
            throw std::runtime_error("Number of connected SRS UEs is not consistent with the number of SRS UEs in the SRS schedule queue in cell " + std::to_string(cIdx));
        }
        int num_ueInfo = num_connected_srs_ue;

        connected_ue_list_vec[cIdx]->set_num_srsInfo(num_srsInfo[cIdx]);

        cumac_muUeGrp_req_ue_info_t* ueInfo_ptr = (cumac_muUeGrp_req_ue_info_t*) (req_data_vec[cIdx]->srsInfoMsh + num_srsInfo[cIdx]);
        for (int i = 0; i < num_ueInfo; i++) {
            ue_id_rnti_t ue_id_rnti = connected_ue_list_vec[cIdx]->get_ue_id_rnti(i);

            ueInfo_ptr[i].flags = 0x03; // valid and new TX
            if (ue_id_rnti.flags == 0x03) {
                ueInfo_ptr[i].flags |= 0x0C; // SRS chanEst available and has updated SRS info in the current slot
                ueInfo_ptr[i].srsInfoIdx = ue_id_rnti.srsInfoIdx;
                connected_ue_list_vec[cIdx]->set_flags(ue_id_rnti.rnti, 0x01); // change to SRS chanEst available
            } else if (ue_id_rnti.flags == 0x01) {
                ueInfo_ptr[i].flags |= 0x04; // SRS chanEst available
            }
            ueInfo_ptr[i].avgRate = 1000000 + distrib(gen) % 20000000; // bits/s
            ueInfo_ptr[i].currRate = 1000000 + distrib(gen) % 20000000; // bits/s
            ueInfo_ptr[i].bufferSize = 20000000; // bits
            ueInfo_ptr[i].id = ue_id_rnti.id; // 0-based cell-specific UE ID used for cuMAC scheduling, ranging from 0 to MAX_NUM_SRS_UE_PER_CELL-1
            ueInfo_ptr[i].rnti = ue_id_rnti.rnti; // C-RNTI
            ueInfo_ptr[i].numAllocPrgLastTx = 0xFFFF;
            ueInfo_ptr[i].layerSelLastTx = 0xFF;
            ueInfo_ptr[i].nUeAnt = sys_param.num_ue_ant_port; // number of SRS TX antenna ports. Value: 2, 4
        }

        req_data_vec[cIdx]->betaCoeff = std::bit_cast<uint32_t>(1.0f); // exponent applied to the instantaneous rate for proportional-fair scheduling. Default value is 1.0.
        req_data_vec[cIdx]->muCoeff = std::bit_cast<uint32_t>(1.5f); // coefficient for prioritizing UEs feasible for MU-MIMO transmissions. Default value is 1.5.
        req_data_vec[cIdx]->chanCorrThr = std::bit_cast<uint32_t>(0.7f); // threshold on the channel vector correlation value for UE grouping. Value: a real number between 0 and 1.0. Default: 0.7
        req_data_vec[cIdx]->srsSnrThr = std::bit_cast<uint32_t>(-3.0f); // Threshold on measured SRS SNR in dB for determining the feasibility of MU-MIMO transmission. Default value is -3.0 (dB).
        req_data_vec[cIdx]->muGrpSrsSnrMaxGap = std::bit_cast<uint32_t>(100.0f); // maximum gap among the SRS SNRs of UEs in the same MU-MIMO UEG. Value: a real number greater than 0.0. Default: 100.0
        req_data_vec[cIdx]->muGrpSrsSnrSplitThr = std::bit_cast<uint32_t>(-100.0f); // threshold to split the SRS SNR range for grouping UEs for MU-MIMO separately. Value: a real number greater than 0.0. Default: -100.0
        req_data_vec[cIdx]->numUeInfo = num_ueInfo;
        req_data_vec[cIdx]->numSrsInfo = num_srsInfo[cIdx];
        req_data_vec[cIdx]->nBsAnt = sys_param.num_bs_ant_port; // Each RU’s number of TX & RX antenna ports.
        req_data_vec[cIdx]->nMaxUeSchdPerCellTTI = sys_param.max_num_ue_schd_per_cell_tti; // maximum number of UEs scheduled per cell per TTI. Default: 16
        req_data_vec[cIdx]->numUeForGrpPerCell = sys_param.max_num_ue_for_grp_per_cell;
        req_data_vec[cIdx]->numSubband = sys_param.num_subband;
        req_data_vec[cIdx]->numPrgSampPerSubband = sys_param.num_prg_samp_per_subband;
        req_data_vec[cIdx]->nPrbGrp = sys_param.num_prg_per_cell;
        req_data_vec[cIdx]->nMaxUePerGrp = 16; // maximum number of UEs per UEG. Default: 16
        req_data_vec[cIdx]->nMaxLayerPerGrp = 16; // maximium number of layers per UEG. Default: 16
        req_data_vec[cIdx]->nMaxLayerPerUeSu = 4; // maximium number of layers per UE for SU-MIMO. Default: 4
        req_data_vec[cIdx]->nMaxLayerPerUeMu = 4; // maximium number of layers per UE for MU-MIMO. Default: 4
        req_data_vec[cIdx]->nMaxUegPerCell = 4; // maximum number of UEGs per cell. Default: 4
        req_data_vec[cIdx]->allocType = 1; // PRB allocation type. Currently only support 1: consecutive type-1 allocation. 
    }

    // Update the msg_len and data_len of l2_l1_msg message
    l2_l1_msg.cell_id = 0;
    l2_l1_msg.msg_len = 0;
    l2_l1_msg.data_len = sizeof(l2_l1_message_t) + nSrsUes*(sizeof(uint32_t) + 4 * sizeof(uint16_t));

    // Send the message
    NVLOGC(MU_TEST_TAG, "L2-MAIN: L2-L1 NVIPC message sending - SFN = %u.%u, msg_len = %d, data_len = %d",
            sfn, slot, l2_l1_msg.msg_len, l2_l1_msg.data_len);
    if(ipc_l2_l1->tx_send_msg(ipc_l2_l1, &(l2_l1_msg)) < 0) {
        NVLOGE(MU_TEST_TAG, AERIAL_NVIPC_API_EVENT, "%s error: send message failed", __func__);
        ipc_l2_l1->tx_release(ipc_l2_l1, &(l2_l1_msg));
        return -1;
    }

    // Post the NVIPC TX notification
    if(ipc_l2_l1->tx_tti_sem_post(ipc_l2_l1) < 0) {
        NVLOGE(MU_TEST_TAG, AERIAL_NVIPC_API_EVENT, "%s error: tx notification failed", __func__);
        ipc_l2_l1->tx_release(ipc_l2_l1, &(l2_l1_msg));
        return -1;
    }

    struct timespec msg_send_start, msg_send_end;

    clock_gettime(CLOCK_REALTIME, &msg_send_start);

    // build and send the messages to cuMAC
    for (int cIdx = 0; cIdx < sys_param.num_cell; cIdx++) {
        cumac_muUeGrp_req_msg_t* req = (cumac_muUeGrp_req_msg_t*) send_msg[cIdx].msg_buf;
        req->sfn = sfn;
        req->slot = slot;
        req->offsetData = 0;

        send_msg[cIdx].msg_id = CUMAC_SCH_TTI_REQUEST;
        send_msg[cIdx].cell_id = cIdx;
        send_msg[cIdx].msg_len = sizeof(cumac_muUeGrp_req_msg_t);
        send_msg[cIdx].data_len = sizeof(cumac_muUeGrp_req_info_t) + 
                                  sizeof(cumac_muUeGrp_req_ue_info_t)*connected_ue_list_vec[cIdx]->get_num_connected_srs_ue() + 
                                  sizeof(cumac_muUeGrp_req_srs_info_msh_t)*connected_ue_list_vec[cIdx]->get_num_srsInfo();

        // Send the message
        NVLOGC(MU_TEST_TAG, "L2-MAIN: L2-cuMAC NVIPC message sending - SFN = %u.%u, cell ID = %u, msg_id = 0x%02X %s, msg_len = %d, data_len = %d",
            sfn, slot, cIdx, send_msg[cIdx].msg_id, get_cumac_msg_name(send_msg[cIdx].msg_id), send_msg[cIdx].msg_len, send_msg[cIdx].data_len);

        if(ipc_l2_cumac->tx_send_msg(ipc_l2_cumac, &(send_msg[cIdx])) < 0) {
            NVLOGE(MU_TEST_TAG, AERIAL_NVIPC_API_EVENT, "%s error: send message failed", __func__);
            // Free the buffer from sender side for error case. Normally will not happen.
            ipc_l2_cumac->tx_release(ipc_l2_cumac, &(send_msg[cIdx]));
            return -1;
        }
    }

    // Post the NVIPC TX notification
    if(ipc_l2_cumac->tx_tti_sem_post(ipc_l2_cumac) < 0) {
        NVLOGE(MU_TEST_TAG, AERIAL_NVIPC_API_EVENT, "%s error: tx notification failed", __func__);
        return -1;
    }
    
    clock_gettime(CLOCK_REALTIME, &msg_send_end);

    int64_t msg_send_duration = nvlog_timespec_interval(&msg_send_start, &msg_send_end);

    NVLOGC(MU_TEST_TAG, "L2-MAIN: L2-cuMAC NVIPC message sending duration: %f microseconds", msg_send_duration/1000.0);

    return 0;
}

// L2 work per slot without memory sharing
int l2_work_per_slot(nv_ipc_t* ipc_l2_l1, nv_ipc_t* ipc_l2_cumac, const sys_param_t& sys_param, std::vector<std::unique_ptr<l2_connected_ue_list_t>>& connected_ue_list_vec, uint32_t slot, uint32_t sfn)
{
    if(ipc_l2_l1 == NULL || ipc_l2_cumac == NULL)
    {
        NVLOGE(MU_TEST_TAG, AERIAL_NVIPC_API_EVENT, "%s: ipc_l2_l1 or ipc_l2_cumac is NULL", __func__);
        return -1;
    }

    // *********** L2 to L1 NVIPC TX message (dummy message) ***********
    nv_ipc_msg_t l2_l1_msg;
    l2_l1_msg.data_pool = NV_IPC_MEMPOOL_CPU_DATA;

    // Allocate NVIPC buffer which contains MSG part and DATA part
    if(ipc_l2_l1->tx_allocate(ipc_l2_l1, &(l2_l1_msg), 0) != 0) {
        NVLOGE(MU_TEST_TAG, AERIAL_NVIPC_API_EVENT, "%s error: NVIPC memory pool is full", __func__);
        return -1;
    }

    l2_l1_message_t* l2_l1_msg_ptr;
    l2_l1_msg_ptr = (l2_l1_message_t*) l2_l1_msg.data_buf;
    l2_l1_msg_ptr->nSrsUes = 0;
    
    // Update the msg_len and data_len of l2_l1_msg message
    l2_l1_msg.cell_id = 0;
    l2_l1_msg.msg_len = 0;
    l2_l1_msg.data_len = sizeof(l2_l1_message_t);

    // Send the message
    NVLOGC(MU_TEST_TAG, "L2-MAIN: L2-L1 NVIPC message sending - SFN = %u.%u, msg_len = %d, data_len = %d",
            sfn, slot, l2_l1_msg.msg_len, l2_l1_msg.data_len);
    if(ipc_l2_l1->tx_send_msg(ipc_l2_l1, &(l2_l1_msg)) < 0) {
        NVLOGE(MU_TEST_TAG, AERIAL_NVIPC_API_EVENT, "%s error: send message failed", __func__);
        ipc_l2_l1->tx_release(ipc_l2_l1, &(l2_l1_msg));
        return -1;
    }

    // Post the NVIPC TX notification
    if(ipc_l2_l1->tx_tti_sem_post(ipc_l2_l1) < 0) {
        NVLOGE(MU_TEST_TAG, AERIAL_NVIPC_API_EVENT, "%s error: tx notification failed", __func__);
        ipc_l2_l1->tx_release(ipc_l2_l1, &(l2_l1_msg));
        return -1;
    }
    // ****************************************************************

    // cuMAC to L2 NVIPC TX message
    nv_ipc_msg_t send_msg[sys_param.num_cell];

    std::vector<cumac_muUeGrp_req_info_t*> req_data_vec(sys_param.num_cell);

    for (int cIdx = 0; cIdx < sys_param.num_cell; cIdx++) {
        // send_msg.msg_buf will be allocated by default. Set data_pool to get send_msg.data_buf
        send_msg[cIdx].data_pool = NV_IPC_MEMPOOL_CPU_DATA;

        // Allocate NVIPC buffer which contains MSG part and DATA part
        if(ipc_l2_cumac->tx_allocate(ipc_l2_cumac, &(send_msg[cIdx]), 0) != 0) {
            NVLOGE(MU_TEST_TAG, AERIAL_NVIPC_API_EVENT, "%s error: NVIPC memory pool is full", __func__);
            return -1;
        }

        req_data_vec[cIdx] = (cumac_muUeGrp_req_info_t*) send_msg[cIdx].data_buf;
    }

    // prepare the slot data
    prepare_slot_data_l2_to_cumac(sys_param, connected_ue_list_vec, req_data_vec, slot);

    struct timespec msg_send_start, msg_send_end;

    clock_gettime(CLOCK_REALTIME, &msg_send_start);

    for (int cIdx = 0; cIdx < sys_param.num_cell; cIdx++) {
        // Build the message
        if(l2_cumac_build_tx_msg_slot(ipc_l2_cumac, &(send_msg[cIdx]), cIdx, slot, sfn, connected_ue_list_vec[cIdx]->get_num_srsInfo(), connected_ue_list_vec[cIdx]->get_num_connected_srs_ue()) < 0) {
            NVLOGE(MU_TEST_TAG, AERIAL_NVIPC_API_EVENT, "%s error: build message failed", __func__);
            // Free the buffer from sender side for error case. Normally will not happen.
            ipc_l2_cumac->tx_release(ipc_l2_cumac, &(send_msg[cIdx]));
            return -1;
        }    

        // Send the message
        NVLOGC(MU_TEST_TAG, "L2-MAIN: L2-cuMAC NVIPC message sending - SFN = %u.%u, cell ID = %u, msg_id = 0x%02X %s, msg_len = %d, data_len = %d",
                sfn, slot, cIdx, send_msg[cIdx].msg_id, get_cumac_msg_name(send_msg[cIdx].msg_id), send_msg[cIdx].msg_len, send_msg[cIdx].data_len);
        if(ipc_l2_cumac->tx_send_msg(ipc_l2_cumac, &(send_msg[cIdx])) < 0) {
            NVLOGE(MU_TEST_TAG, AERIAL_NVIPC_API_EVENT, "%s error: send message failed", __func__);
            // Free the buffer from sender side for error case. Normally will not happen.
            ipc_l2_cumac->tx_release(ipc_l2_cumac, &(send_msg[cIdx]));
            return -1;
        }
    }

    // Post the NVIPC TX notification
    if(ipc_l2_cumac->tx_tti_sem_post(ipc_l2_cumac) < 0) {
        NVLOGE(MU_TEST_TAG, AERIAL_NVIPC_API_EVENT, "%s error: tx notification failed", __func__);
        return -1;
    }

    clock_gettime(CLOCK_REALTIME, &msg_send_end);

    int64_t msg_send_duration = nvlog_timespec_interval(&msg_send_start, &msg_send_end);

    NVLOGC(MU_TEST_TAG, "L2-MAIN: L2-cuMAC NVIPC message sending duration: %f microseconds", msg_send_duration/1000.0);

    return 0;
}

// Prepare the slot data for an L2-to-cuMAC NVIPC TX message without L1-cuMAC memory sharing. 
void prepare_slot_data_l2_to_cumac(const sys_param_t& sys_param, std::vector<std::unique_ptr<l2_connected_ue_list_t>>& connected_ue_list_vec, std::vector<cumac_muUeGrp_req_info_t*> req_data, const int slot_idx)
{
    // initialize random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> distrib(0, INT_MAX);

    // Normal (Gaussian) distribution with mean 0 and stddev 1
    std::normal_distribution<> normal_distrib(0.0, 1.0);

    char tdd_direction = sys_param.TDD_pattern[slot_idx % sys_param.TDD_pattern.length()];

    std::vector<int> num_srsInfo(sys_param.num_cell, 0);

    for (int cIdx = 0; cIdx < sys_param.num_cell; cIdx++) {
        req_data[cIdx]->srsInfo = (cumac_muUeGrp_req_srs_info_t*) (req_data[cIdx]->payload);
    }

    if (tdd_direction == 'S') { // S-slot with SRS scheduled UEs
        for (int cIdx = 0; cIdx < sys_param.num_cell; cIdx++) {
            uint16_t num_connected_srs_ue = connected_ue_list_vec[cIdx]->get_num_connected_srs_ue();
            if (num_connected_srs_ue == 0xFFFF) {
                throw std::runtime_error("Number of connected SRS UEs is not consistent with the number of SRS UEs in the SRS schedule queue in cell " + std::to_string(cIdx));
            }

            num_srsInfo[cIdx] = sys_param.num_srs_ue_per_slot < num_connected_srs_ue ? sys_param.num_srs_ue_per_slot : num_connected_srs_ue;

            for (int i = 0; i < num_srsInfo[cIdx]; i++) {
                ue_id_rnti_t ue_id_rnti = connected_ue_list_vec[cIdx]->get_next_srs_schedule();

                req_data[cIdx]->srsInfo[i].flags = 0x01; // valid
                req_data[cIdx]->srsInfo[i].nUeAnt = sys_param.num_ue_ant_port;
                req_data[cIdx]->srsInfo[i].id = ue_id_rnti.id;
                req_data[cIdx]->srsInfo[i].rnti = ue_id_rnti.rnti;
                req_data[cIdx]->srsInfo[i].srsWbSnr = std::bit_cast<uint32_t>(5.0f); // dB
                
                for (int subband_idx = 0; subband_idx < sys_param.num_subband; subband_idx++) {
                    for (int prg_idx = 0; prg_idx < sys_param.num_prg_samp_per_subband; prg_idx++) {
                        int index = subband_idx*sys_param.num_prg_samp_per_subband*sys_param.num_bs_ant_port*req_data[cIdx]->srsInfo[i].nUeAnt + 
                                    prg_idx*sys_param.num_bs_ant_port*req_data[cIdx]->srsInfo[i].nUeAnt;
                        for (int chan_idx = 0; chan_idx < sys_param.num_bs_ant_port*req_data[cIdx]->srsInfo[i].nUeAnt; chan_idx++) {
                            req_data[cIdx]->srsInfo[i].srsChanEst[index+chan_idx].x = normal_distrib(gen)*sqrt(0.5*sys_param.srs_chan_est_coeff_var);
                            req_data[cIdx]->srsInfo[i].srsChanEst[index+chan_idx].y = normal_distrib(gen)*sqrt(0.5*sys_param.srs_chan_est_coeff_var);
                        }
                    }
                }

                connected_ue_list_vec[cIdx]->set_srs_info(ue_id_rnti.rnti, i); 
            }
        }
    }

    for (int cIdx = 0; cIdx < sys_param.num_cell; cIdx++) {
        uint16_t num_connected_srs_ue = connected_ue_list_vec[cIdx]->get_num_connected_srs_ue();
        if (num_connected_srs_ue == 0xFFFF) {
            throw std::runtime_error("Number of connected SRS UEs is not consistent with the number of SRS UEs in the SRS schedule queue in cell " + std::to_string(cIdx));
        }
        int num_ueInfo = num_connected_srs_ue;

        connected_ue_list_vec[cIdx]->set_num_srsInfo(num_srsInfo[cIdx]);

        cumac_muUeGrp_req_ue_info_t* ueInfo_ptr = (cumac_muUeGrp_req_ue_info_t*) (req_data[cIdx]->srsInfo + num_srsInfo[cIdx]); 
        for (int i = 0; i < num_ueInfo; i++) {
            ue_id_rnti_t ue_id_rnti = connected_ue_list_vec[cIdx]->get_ue_id_rnti(i);

            ueInfo_ptr[i].flags = 0x03; // valid and new TX
            if (ue_id_rnti.flags == 0x03) {
                ueInfo_ptr[i].flags |= 0x0C; // SRS chanEst available and has updated SRS info in the current slot
                ueInfo_ptr[i].srsInfoIdx = ue_id_rnti.srsInfoIdx;
                connected_ue_list_vec[cIdx]->set_flags(ue_id_rnti.rnti, 0x01); // change to SRS chanEst available
            } else if (ue_id_rnti.flags == 0x01) {
                ueInfo_ptr[i].flags |= 0x04; // SRS chanEst available
            }
            ueInfo_ptr[i].avgRate = 1000000 + distrib(gen) % 20000000; // bits/s
            ueInfo_ptr[i].currRate = 1000000 + distrib(gen) % 20000000; // bits/s
            ueInfo_ptr[i].bufferSize = 20000000; // bits
            ueInfo_ptr[i].id = ue_id_rnti.id; // 0-based cell-specific UE ID used for cuMAC scheduling, ranging from 0 to MAX_NUM_SRS_UE_PER_CELL-1
            ueInfo_ptr[i].rnti = ue_id_rnti.rnti; // C-RNTI
            ueInfo_ptr[i].numAllocPrgLastTx = 0xFFFF;
            ueInfo_ptr[i].layerSelLastTx = 0xFF;
            ueInfo_ptr[i].nUeAnt = sys_param.num_ue_ant_port; // number of SRS TX antenna ports. Value: 2, 4
        }

        req_data[cIdx]->betaCoeff = std::bit_cast<uint32_t>(1.0f); // exponent applied to the instantaneous rate for proportional-fair scheduling. Default value is 1.0.
        req_data[cIdx]->muCoeff = std::bit_cast<uint32_t>(1.5f); // coefficient for prioritizing UEs feasible for MU-MIMO transmissions. Default value is 1.5.
        req_data[cIdx]->chanCorrThr = std::bit_cast<uint32_t>(0.7f); // threshold on the channel vector correlation value for UE grouping. Value: a real number between 0 and 1.0. Default: 0.7
        req_data[cIdx]->srsSnrThr = std::bit_cast<uint32_t>(-3.0f); // Threshold on measured SRS SNR in dB for determining the feasibility of MU-MIMO transmission. Default value is -3.0 (dB).
        req_data[cIdx]->muGrpSrsSnrMaxGap = std::bit_cast<uint32_t>(100.0f); // maximum gap among the SRS SNRs of UEs in the same MU-MIMO UEG. Value: a real number greater than 0.0. Default: 100.0
        req_data[cIdx]->muGrpSrsSnrSplitThr = std::bit_cast<uint32_t>(-100.0f); // threshold to split the SRS SNR range for grouping UEs for MU-MIMO separately. Value: a real number greater than 0.0. Default: -100.0
        req_data[cIdx]->numUeInfo = num_ueInfo;
        req_data[cIdx]->numSrsInfo = num_srsInfo[cIdx];
        req_data[cIdx]->nBsAnt = sys_param.num_bs_ant_port; // Each RU’s number of TX & RX antenna ports.
        req_data[cIdx]->nMaxUeSchdPerCellTTI = sys_param.max_num_ue_schd_per_cell_tti; // maximum number of UEs scheduled per cell per TTI. Default: 16
        req_data[cIdx]->numUeForGrpPerCell = sys_param.max_num_ue_for_grp_per_cell;
        req_data[cIdx]->numSubband = sys_param.num_subband;
        req_data[cIdx]->numPrgSampPerSubband = sys_param.num_prg_samp_per_subband;
        req_data[cIdx]->nPrbGrp = sys_param.num_prg_per_cell;
        req_data[cIdx]->nMaxUePerGrp = 16; // maximum number of UEs per UEG. Default: 16
        req_data[cIdx]->nMaxLayerPerGrp = 16; // maximium number of layers per UEG. Default: 16
        req_data[cIdx]->nMaxLayerPerUeSu = 4; // maximium number of layers per UE for SU-MIMO. Default: 4
        req_data[cIdx]->nMaxLayerPerUeMu = 4; // maximium number of layers per UE for MU-MIMO. Default: 4
        req_data[cIdx]->nMaxUegPerCell = 4; // maximum number of UEGs per cell. Default: 4
        req_data[cIdx]->allocType = 1; // PRB allocation type. Currently only support 1: consecutive type-1 allocation.   
    }
}

// Build NVIPC request messages
int l2_cumac_build_tx_msg_slot(nv_ipc_t* ipc_l2_cumac, nv_ipc_msg_t* nvipc_buf, uint16_t cell_id, uint16_t slot, uint16_t sfn, int num_srsInfo, int num_ueInfo)
{
    if(ipc_l2_cumac == NULL || nvipc_buf == NULL)
    {
        NVLOGE(MU_TEST_TAG, AERIAL_NVIPC_API_EVENT, "%s: ipc or msg buffer is NULL", __func__);
        return -1;
    }

    cumac_muUeGrp_req_msg_t* req = (cumac_muUeGrp_req_msg_t*) nvipc_buf->msg_buf;

    req->sfn = sfn;
    req->slot = slot;
    req->offsetData = 0;

    // Update the msg_len and data_len of the NVIPC message header
    nvipc_buf->msg_id = CUMAC_SCH_TTI_REQUEST;
    nvipc_buf->cell_id = cell_id;
    nvipc_buf->msg_len = sizeof(cumac_muUeGrp_req_msg_t);
    nvipc_buf->data_len = sizeof(cumac_muUeGrp_req_info_t) + sizeof(cumac_muUeGrp_req_ue_info_t)*num_ueInfo + sizeof(cumac_muUeGrp_req_srs_info_t)*num_srsInfo;

    return 0;
}

// *****************************************************
// L2 receiver thread to receive the scheduling response messages from cuMAC-CP
void* l2_cumac_blocking_recv_task(void* arg)
{
    // Set thread name, max string length < 16
    pthread_setname_np(pthread_self(), "l2_cuMAC_recv");
    // Set thread schedule policy to SCHED_FIFO and set priority to 80
    nv_set_sched_fifo_priority(80);
    // Set the thread CPU core
    nv_assign_thread_cpu_core(L2_CUMAC_RECV_THREAD_CORE);

    int num_slot = 0;

    while (num_slot < NUM_TIME_SLOTS && !g_shutdown) {
        NVLOGI(MU_TEST_TAG, "%s: wait for incoming messages notification ...", __func__);

        // Wait for incoming messages notification from cuMAC-CP
        ipc_l2_cumac->rx_tti_sem_wait(ipc_l2_cumac);
        if (g_shutdown) break;
        num_slot++;

        struct timespec msg_recv_start, msg_recv_end;
        clock_gettime(CLOCK_REALTIME, &msg_recv_start);

        nv_ipc_msg_t recv_msg;

        int num_recv_msg = 0;

        // Dequeue the incoming NVIPC message
        while (ipc_l2_cumac->rx_recv_msg(ipc_l2_cumac, &recv_msg) >= 0) {
            cumac_muUeGrp_resp_msg_t* resp = (cumac_muUeGrp_resp_msg_t*) recv_msg.msg_buf;
            uint8_t* data_buf = (uint8_t*) recv_msg.data_buf;

            num_recv_msg++;

            NVLOGC(MU_TEST_TAG, "L2-cuMAC RECV: SFN = %u.%u, cell ID = %u, msg_id = 0x%02X %s, msg_len = %d, data_len = %d",
                    resp->sfn, resp->slot, recv_msg.cell_id, recv_msg.msg_id, get_cumac_msg_name(recv_msg.msg_id), recv_msg.msg_len, recv_msg.data_len);

            cumac_muUeGrp_resp_info_t* output_schd_uegInfo = (cumac_muUeGrp_resp_info_t*) (data_buf + resp->offsetData);

            if (PRINT_UE_PAIRING_SOLUTION) {
                print_ue_pairing_sol("L2", resp->sfn, resp->slot,
                                     reinterpret_cast<const uint8_t*>(output_schd_uegInfo), 1, recv_msg.cell_id);
            }

            // Release the NVIPC message buffer
            ipc_l2_cumac->rx_release(ipc_l2_cumac, &recv_msg);
        }

        clock_gettime(CLOCK_REALTIME, &msg_recv_end);
        int64_t msg_recv_duration = nvlog_timespec_interval(&msg_recv_start, &msg_recv_end);

        NVLOGC(MU_TEST_TAG, "L2-cuMAC RECV: NVIPC message receive duration: %f microseconds", msg_recv_duration/1000.0);

        NVLOGC(MU_TEST_TAG, "L2-cuMAC RECV: time slot %d, received %d messages, expected %d messages", num_slot-1, num_recv_msg, NUM_CELL);
    }

    NVLOGC(MU_TEST_TAG, "L2-cuMAC RECV: test completed successfully");
    return NULL;
}

int main(int argc, char** argv)
{
    struct sigaction sa = {};
    sa.sa_handler = signal_handler;
    sigemptyset(&sa.sa_mask);
    sigaction(SIGINT, &sa, nullptr);
    sigaction(SIGTERM, &sa, nullptr);

    NVLOGC(MU_TEST_TAG, "L2-MAIN: parameter config YAML file: %s", YAML_PARAM_CONFIG_PATH);
    NVLOGC(MU_TEST_TAG, "L2-MAIN: L2-cuMAC secondary NVIPC config YAML file: %s", YAML_L2_CUMAC_NVIPC_CONFIG_PATH);

    // system configuration parameters
    sys_param_t sys_param(YAML_PARAM_CONFIG_PATH);
    L2_MAIN_THREAD_CORE = sys_param.l2_main_thread_core;
    L2_CUMAC_RECV_THREAD_CORE = sys_param.l2_cumac_recv_thread_core;
    NUM_TIME_SLOTS = sys_param.num_time_slots;
    NUM_CELL = sys_param.num_cell;
    PRINT_UE_PAIRING_SOLUTION = sys_param.print_ue_pairing_solution;

    // initialize connected UE list for each cell
    std::vector<std::unique_ptr<l2_connected_ue_list_t>> connected_ue_list_vec;
    for (int i = 0; i < sys_param.num_cell; i++) {
        connected_ue_list_vec.push_back(std::make_unique<l2_connected_ue_list_t>(i, sys_param.num_srs_ue_per_cell));
    }

    // Load nvipc configuration from YAML file
    nv_ipc_config_t l2_cumac_config;
    load_nv_ipc_yaml_config(&l2_cumac_config, YAML_L2_CUMAC_NVIPC_CONFIG_PATH, NV_IPC_MODULE_SECONDARY);    

    nv_ipc_config_t l2_l1_config;
    load_nv_ipc_yaml_config(&l2_l1_config, YAML_L2_L1_NVIPC_CONFIG_PATH, NV_IPC_MODULE_PRIMARY);

    // Initialize NVIPC interface and connect to the cuMAC-CP
    if ((ipc_l2_cumac = create_nv_ipc_interface(&l2_cumac_config)) == NULL)
    {
        NVLOGE(MU_TEST_TAG, AERIAL_NVIPC_API_EVENT, "%s: create L2/cuMAC secondary NVIPC interface failed", __func__);
        return -1;
    }

    // Initialize NVIPC interface and connect to the L1 stack
    if ((ipc_l2_l1 = create_nv_ipc_interface(&l2_l1_config)) == NULL)
    {
        NVLOGE(MU_TEST_TAG, AERIAL_NVIPC_API_EVENT, "%s: create L2/L1 primary NVIPC interface failed", __func__);
        return -1;
    }

    // Sleep 1 seconds for NVIPC connections
    usleep(1000000);

    // Configure L2 main thread
    pthread_setname_np(pthread_self(), "l2_main");
    nv_set_sched_fifo_priority(80);
    nv_assign_thread_cpu_core(L2_MAIN_THREAD_CORE);

    // Create L2/cuMAC receiver thread
    pthread_t thread_id;
    int ret = pthread_create(&thread_id, NULL, l2_cumac_blocking_recv_task, NULL);
    if(ret != 0)
    {
        NVLOGE(MU_TEST_TAG, AERIAL_NVIPC_API_EVENT, "%s failed, ret=%d", __func__, ret);
        return -1;
    }

    // Initial SFN/SLOT = 0.0
    uint16_t sfn = 0, slot = 0;

    // Align the first slot timestamp to the next second
    struct timespec ts_slot, ts_remain;
    clock_gettime(CLOCK_REALTIME, &ts_slot);
    ts_slot.tv_nsec = 0;
    ts_slot.tv_sec ++;

    // Main loop or sender thread
    for (int slotIdx = 0; slotIdx < sys_param.num_time_slots && !g_shutdown; slotIdx++) {
        // Sleep to the next slot timestamp
        int ret = clock_nanosleep(CLOCK_REALTIME, TIMER_ABSTIME, &ts_slot, &ts_remain);
        if(ret != 0)
        {
            NVLOGE(MU_TEST_TAG, AERIAL_CLOCK_API_EVENT, "clock_nanosleep returned error ret: %d", ret);
        }

        // Send the UE grouping request message to cuMAC-CP
        if (sys_param.enable_l1_l2_mem_sharing) {
            l2_work_per_slot_mem_sharing(ipc_l2_l1, ipc_l2_cumac, sys_param, connected_ue_list_vec, slot, sfn);
        } else {
            l2_work_per_slot(ipc_l2_l1, ipc_l2_cumac, sys_param, connected_ue_list_vec, slot, sfn);
        }

        // Update SFN/SLOT for next slot
        advance_sfn_slot(sfn, slot);

        // Update timestamp for next slot
        get_next_slot_timespec(&ts_slot, sys_param.slot_interval_ns);
    }

    if (g_shutdown) {
        NVLOGC(MU_TEST_TAG, "L2-MAIN: received shutdown signal, cleaning up...");
    }

    // Wait for the L2 receiver thread to exit
    ret = pthread_join(thread_id, NULL);
    if(ret != 0)
    {
        NVLOGE(MU_TEST_TAG, AERIAL_NVIPC_API_EVENT, "%s pthread_join failed, stderr=%s", __func__, strerror(ret));
        return -1;
    }

    // release NVIPC interfaces
    ipc_l2_cumac->ipc_destroy(ipc_l2_cumac);
    ipc_l2_l1->ipc_destroy(ipc_l2_l1);

    NVLOGC(MU_TEST_TAG, "L2-MAIN: test completed successfully, ENABLE_L1_L2_MEM_SHARING: %s", (sys_param.enable_l1_l2_mem_sharing ? "true" : "false"));

    return 0;
}


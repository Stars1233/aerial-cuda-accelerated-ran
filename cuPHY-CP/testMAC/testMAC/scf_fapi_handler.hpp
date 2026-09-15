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

#ifndef _SCF_FAPI_HANDLER_HPP_
#define _SCF_FAPI_HANDLER_HPP_

#include <thread>
#include <memory>
#include <vector>

#include "scf_5g_fapi.h"

#include "fapi_handler.hpp"
#include "scf_5g_fapi_msg_helpers.hpp"

// Schedule item structure for each FAPI message.
typedef struct {
    int cell_id; // Cell ID
    fapi_group_t group_id; // FAPI group ID, represents the TX FAPI message
    int32_t exist; // Whether the FAPI message exists in current slot. 0 - not exist, 1 - exist
    int32_t remain_num; // Remaining number of messages to send within the same TX deadline.
    int64_t tx_deadline; // FAPI TX deadline time. Unit: ns.
} schedule_item_t;

/**
 * SCF 5G FAPI handler: builds slot requests from the launch pattern, schedules TX deadlines, and drives NVIPC.
 *
 * Construct without a transport; the owner must call set_transport() before runtime IPC (see test_mac::start()
 * or unit-test harnesses such as TestMacSetupService).
 */
class scf_fapi_handler : public fapi_handler {
public:
    /**
     * @param[in] configs                 Parsed test MAC / FAPI configuration (not owned).
     * @param[in] lp                      Launch pattern (not owned).
     * @param[in] conformance_test_stats Optional stats sink; may be nullptr.
     */
    scf_fapi_handler(test_mac_configs* configs, launch_pattern* lp, ch8_conformance_test_stats* conformance_test_stats);

    /** Releases prebuilt FAPI host buffers and destroys worker synchronization objects. */
    ~scf_fapi_handler() override;

    void cell_init(int cell_id) override;
    void cell_start(int cell_id) override;
    void cell_stop(int cell_id) override;
    void on_msg(nv_ipc_msg_t& msg) override;
    int  schedule_slot(sfn_slot_t ss) override;
    int  schedule_fapi_reqs(sfn_slot_t ss, int ts_offset);
    int  schedule_fapi_request(int cell_id, sfn_slot_t ss, fapi_group_t group_id, int32_t ts_offset);

    /**
     * Builds CONFIG.req and repeating-pattern slot downlink messages into host-backed caches (config_reqs, slot_msgs).
     * Uses configs->get_max_msg_size() / get_max_data_size() for temporary encode buffers.
     * @note Succeeds at most once per instance; a second call returns -1.
     */
    [[nodiscard]] int prebuild_downlink_messages() override;

    /**
     * Releases prebuilt FAPI host buffers.
     */
    void release_prebuilt_message_storage();

    /**
     * Stop all cells and exit the FAPI handler
     */
    void terminate() override;

    // FAPI builder thread function
    void builder_thread_func() override;

    // Scheduler thread function
    void scheduler_thread_func() override;

    // Worker thread function
    void worker_thread_func() override;

    void notify_worker_threads() override;

    int send_config_request(int cell_id) override;
    int send_mem_bank_cv_config_req(int cell_id);
    int send_start_request(int cell_id) override;
    int send_stop_request(int cell_id) override;

    int build_fapi_and_send(int cell_id, sfn_slot_t ss_curr);
    int send_fapi_from_queue(int cell_id, sfn_slot_t ss_curr);

protected:
    bool cell_id_sanity_check(int cell_id);
    int encode_tx_beamforming(uint8_t* buf, const tx_beamforming_data_t& beam_data, channel_type_t ch_type);

    int encode_rx_beamforming(uint8_t* buf, const rx_beamforming_data_t& beam_data, channel_type_t ch_type);

    int encode_srs_rx_beamforming(uint8_t* buf, const rx_srs_beamforming_data_t& beam_data);

    // Functions for building dynamic slot FAPI requests: DL_TTI.req, UL_TTI.req, TX_DATA.req
    int build_dyn_dl_tti_request(int cell_id, std::vector<fapi_req_t*>& fapi_reqs, scf_fapi_dl_tti_req_t& req, dyn_slot_param_t& param);
    int build_dyn_tx_data_request(int cell_id, std::vector<fapi_req_t*>& fapi_reqs, scf_fapi_tx_data_req_t& req, nv::phy_mac_msg_desc& msg_desc, dyn_slot_param_t& param);
    int build_dyn_ul_tti_request(int cell_id, std::vector<fapi_req_t*>& fapi_reqs, scf_fapi_ul_tti_req_t& req, dyn_slot_param_t& param);

    // Functions for building slot downlink FAPI requests: CONFIG.req, DL_TTI.req, UL_TTI.req, TX_DATA.req, UL_DCI.req, DL_BFW_CVI.req, UL_BFW_CVI.req
    int build_config_request(int cell_id, nv::phy_mac_msg_desc& msg_desc);
    int build_dl_tti_request(int cell_id, std::vector<fapi_req_t*>& fapi_reqs, scf_fapi_dl_tti_req_t& req);
    int build_tx_data_request(int cell_id, std::vector<fapi_req_t*>& fapi_reqs, scf_fapi_tx_data_req_t& req, nv::phy_mac_msg_desc& msg_desc);
    int build_ul_tti_request(int cell_id, std::vector<fapi_req_t*>& fapi_reqs, scf_fapi_ul_tti_req_t& req);
    int build_ul_dci_request(int cell_id, std::vector<fapi_req_t*>& fapi_reqs, scf_fapi_ul_dci_t& req);
    int build_bfw_cvi_request(int cell_id, vector<fapi_req_t*>& fapi_reqs, scf_fapi_dl_bfw_cvi_request_t* req);

    // Build slot FAPI request wrapper function
    int build_slot_fapi_request(int cell_id, sfn_slot_t ss, fapi_group_t group_id, nv::phy_mac_msg_desc& msg_desc);

    int send_slot_response(int cell_id, sfn_slot_t& ss);

    void validate_indication_timing(int cell_id, uint64_t handle_start_time, int sfn, int slot, int deadline_ns, timing_t& summary_timing, timing_t& thrputs_timing);

    int handle_uci_indication(int cell_id, scf_fapi_uci_ind_t& resp, fapi_validate& vald);

    void validate_timing_srs_indication(int cell_id, uint64_t handle_start_time, scf_fapi_srs_ind_t& resp);

    int parse_pf01_sr(int cell_id, uci_pdu_format_t format, uint8_t* payload);
    int parse_pf01_harq(int cell_id, uci_pdu_format_t format, uint8_t* payload);
    int parse_pf234_sr(int cell_id, uci_pdu_format_t format, uint8_t* payload);
    int parse_pf234_harq(int cell_id, uci_pdu_format_t format, uint8_t* payload);
    int parse_pf234_csi(int cell_id, uci_pdu_format_t format, int csi_id, uint8_t* payload);
    int parse_pusch_csi(int cell_id, int csi_id, uint8_t* payload);

    void update_prach_ocassion(const prach_tv_data_t& prach_pars);

    void update_pucch_ocassion(scf_fapi_pucch_pdu_t& pucch_pdu);

    int validate_rach_ind(int cell_id, uint16_t sfn, uint16_t slot, int pdu_id, scf_fapi_prach_ind_pdu_t* prmb, fapi_validate& vald);
    int validate_rx_data_ind(int cell_id, uint16_t sfn, uint16_t slot, int pdu_id, scf_fapi_rx_data_pdu_t* pdu, uint8_t* tb_data, fapi_validate& vald);
    int validate_crc_ind(int cell_id, uint16_t sfn, uint16_t slot, int pdu_id, scf_fapi_crc_info_t* crc, fapi_validate& vald);
    int validate_srs_ind(int cell_id, uint16_t sfn, uint16_t slot, int pdu_id, scf_fapi_srs_info_t* pdu, uint8_t* iq_report_buffer, uint32_t handle, int * pRbSnrOffset, fapi_validate& vald, bool free_srs_chest_buffer_index_l2);

    int validate_pe_noise_interference_ind(int cell_id, uint16_t sfn, uint16_t slot, int pdu_id, scf_fapi_meas_t* pdu, fapi_validate& vald);
    int validate_pf234_interference_ind(int cell_id, uint16_t sfn, uint16_t slot, int pdu_id, scf_fapi_meas_t* pdu, fapi_validate& vald);
    int validate_prach_interference_ind(int cell_id, uint16_t sfn, uint16_t slot, int pdu_id, scf_fapi_prach_interference_t* pdu, fapi_validate& vald);

    int compare_ul_measurement(fapi_validate* vald, ul_measurement_t& tv, uint8_t* payload, uint16_t pdu_type);
    int compare_ul_measurement_ehq(fapi_validate* vald, ul_measurement_t& tv, uint8_t* payload, uint16_t pdu_type);

    // Reorder schedule sequence per slot (only items with fapi_reqs.size() > 0), by deadline
    int reorder_schedule_sequence(sfn_slot_t ss);

    // Data members

    // Worker thread inscreasing index and synchronization semaphore
    std::atomic<int> worker_id = 0;
    sem_t            worker_sem;

    // Dynamic slot test parameters for spectral efficiency
    std::vector<uint8_t> dyn_tb_data_gen_buf;

    // Schedule item list. size = cell_num * FAPI_REQ_SIZE
    std::vector<schedule_item_t> schedule_item_list;

    // Schedule sequence (indices into schedule_item_list). Size = number of items with fapi_reqs.size() > 0 for current slot.
    std::vector<int> schedule_sequence;

    // Per-FAPI-message timing: first send per message type in the slot (ns since epoch, -1 if not set). Used for MAC.PROCESSING_TIMES.
    int64_t dl_tti_start_ns_ = -1;
    int64_t dl_tti_stop_ns_ = -1;
    int64_t tx_data_start_ns_ = -1;
    int64_t tx_data_stop_ns_ = -1;
    int64_t ul_tti_start_ns_ = -1;
    int64_t ul_tti_stop_ns_ = -1;
};

#endif /* _SCF_FAPI_HANDLER_HPP_ */

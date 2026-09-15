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

#ifndef APP_CONFIG_H
#define APP_CONFIG_H

#include <cstdint>
#include <string>

class AppConfig
{
public:
    static AppConfig &getInstance();
    void setLowPriorityCore(uint8_t low_priority_core);
    uint8_t getLowPriorityCore() const;
    void setNicTputAlertThreshold(uint32_t nic_tput_alert_threshold);
    uint32_t getNicTputAlertThreshold() const;
    void setCellGroupNum(uint8_t cell_group_num);
    uint8_t getCellGroupNum() const;
    bool isCellActive(uint16_t mplane_id) const;
    void cellActivated(uint16_t mplane_id);
    void cellDeactivated(uint16_t mplane_id);
    void setCUSPortFailover(bool failover);
    bool isCUSPortFailoverEnabled() const;
    uint64_t getTaiOffset() const;
    void setTaiOffset();

    void enablePtpSvcMonitoring(bool enable_ptp_svc_monitoring);
    bool isPtpSvcMonitoringEnabled() const;
    void setPtpRmsThreshold(uint8_t ptp_rms_threshold);
    uint8_t getPtpRmsThreshold() const;

    void enableRhocpPtpEventsMonitoring(bool enable_rhocp_ptp_events_monitoring);
    bool isRhocpPtpEventsMonitoringEnabled() const;
    void setRhocpPtpPublisher(const std::string& rhocp_ptp_publisher);
    void setRhocpPtpNodeName(const std::string& rhocp_ptp_node_name);
    void setRhocpPtpConsumer(const std::string& rhocp_ptp_consumer);
    const std::string& getRhocpPtpPublisher() const;
    const std::string& getRhocpPtpNodeName() const;
    const std::string& getRhocpPtpConsumer() const;

private:
    AppConfig() { active_cell_bitmap_ = 0; setTaiOffset(); }
    AppConfig(const AppConfig &other) = delete;
    AppConfig &operator=(const AppConfig &other) = delete;

    uint8_t low_priority_core_;
    uint32_t nic_tput_alert_threshold_;
    uint8_t cell_group_num_;
    uint32_t active_cell_bitmap_;
    uint64_t tai_offset_;
    bool cus_port_failover_;
    bool enable_ptp_svc_monitoring_;
    uint8_t ptp_rms_threshold_;
    bool enable_rhocp_ptp_events_monitoring_;
    std::string rhocp_ptp_publisher_;
    std::string rhocp_ptp_node_name_;
    std::string rhocp_ptp_consumer_;
};

#endif // APP_CONFIG_H
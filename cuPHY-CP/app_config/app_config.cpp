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

#include "app_config.hpp"
#include "nvlog.hpp"
#include <sys/timex.h>
#include <cstring>
#include <iostream>
#include <string>

#define TAG (NVLOG_TAG_BASE_APP_CFG_UTILS + 1) // "APP.CONFIG"

AppConfig &AppConfig::getInstance()
{
    static AppConfig instance;
    return instance;
}

void AppConfig::setLowPriorityCore(uint8_t low_priority_core)
{
    low_priority_core_ = low_priority_core;
}

uint8_t AppConfig::getLowPriorityCore() const
{
    return low_priority_core_;
}

void AppConfig::setNicTputAlertThreshold(uint32_t nic_tput_alert_threshold)
{
    nic_tput_alert_threshold_ = nic_tput_alert_threshold;
}

uint32_t AppConfig::getNicTputAlertThreshold() const
{
    return nic_tput_alert_threshold_;
}

void AppConfig::setCellGroupNum(uint8_t cell_group_num)
{
    cell_group_num_ = cell_group_num;
}

uint8_t AppConfig::getCellGroupNum() const
{
    return cell_group_num_;
}

bool AppConfig::isCellActive(uint16_t mplane_id) const
{
    return active_cell_bitmap_ & ((uint32_t)1 << mplane_id);
}

void AppConfig::cellActivated(uint16_t mplane_id)
{
    active_cell_bitmap_ |= ((uint32_t)1 << mplane_id);
}

void AppConfig::cellDeactivated(uint16_t mplane_id)
{
    active_cell_bitmap_ &= ~((uint32_t)1 << mplane_id);
}

void AppConfig::setCUSPortFailover(bool failover)
{
    cus_port_failover_ = failover;
}

bool AppConfig::isCUSPortFailoverEnabled() const
{
    return cus_port_failover_;
}

uint64_t AppConfig::getTaiOffset() const
{
    return tai_offset_;
}

void AppConfig::setTaiOffset()
{
    struct timex tmx;
    memset(&tmx, 0, sizeof(tmx));
    if (adjtimex(&tmx) == -1)
    {
        NVLOGE_FMT(TAG, AERIAL_SYSTEM_API_EVENT, "adjtimex failed: {}",  strerror(errno));
        return;
    }
    tai_offset_ = (uint64_t)tmx.tai * 1000000000;
    NVLOGC_FMT(TAG, "Current TAI offset: {}s", tmx.tai);
}

void AppConfig::enablePtpSvcMonitoring(bool enable_ptp_svc_monitoring)
{
    enable_ptp_svc_monitoring_ = enable_ptp_svc_monitoring;
}

bool AppConfig::isPtpSvcMonitoringEnabled() const
{
    return enable_ptp_svc_monitoring_;
}

void AppConfig::setPtpRmsThreshold(uint8_t ptp_rms_threshold)
{
    ptp_rms_threshold_ = ptp_rms_threshold;
}

uint8_t AppConfig::getPtpRmsThreshold() const
{
    return ptp_rms_threshold_;
}

void AppConfig::enableRhocpPtpEventsMonitoring(bool enable_rhocp_ptp_events_monitoring)
{
    enable_rhocp_ptp_events_monitoring_ = enable_rhocp_ptp_events_monitoring;
}

bool AppConfig::isRhocpPtpEventsMonitoringEnabled() const
{
    return enable_rhocp_ptp_events_monitoring_;
}

void AppConfig::setRhocpPtpPublisher(const std::string& rhocp_ptp_publisher)
{
    rhocp_ptp_publisher_ = rhocp_ptp_publisher;
}   

void AppConfig::setRhocpPtpNodeName(const std::string& rhocp_ptp_node_name)
{
    rhocp_ptp_node_name_ = rhocp_ptp_node_name;
}

void AppConfig::setRhocpPtpConsumer(const std::string& rhocp_ptp_consumer)
{
    rhocp_ptp_consumer_ = rhocp_ptp_consumer;
}

const std::string& AppConfig::getRhocpPtpPublisher() const
{
    return rhocp_ptp_publisher_;
}

const std::string& AppConfig::getRhocpPtpNodeName() const
{
    return rhocp_ptp_node_name_;
}

const std::string& AppConfig::getRhocpPtpConsumer() const
{
    return rhocp_ptp_consumer_;
}

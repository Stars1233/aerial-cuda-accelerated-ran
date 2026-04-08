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

#ifndef ptp_service_status_checking_H
#define ptp_service_status_checking_H

#include <ctime>
#include <string>

namespace AppUtils
{
    enum class ServiceStatus
    {
        RUNNING_SYNCED,
        RUNNING_UNSYNCED,
        STOPPED,
        ERROR
    };

    ServiceStatus checkPtpServiceStatus(const std::string &syslogPath, double rmsThreshold, const std::string &serviceName);
    int checkPtpServiceStatus(const std::string &syslogPath, double ptp4lRmsThreshold = 10.0, double phc2sysRmsThreshold = 10.0);

    /**
     * Get most recent PTP port link down/up event timestamps from syslog (ptp4l or systemd-networkd lines).
     * @param syslogPath Path to the syslog file to scan.
     * @param[out] outLastLinkDownTs On success, set to the timestamp of the most recent link-down event, or (time_t)-1 if none.
     * @param[out] outLastLinkUpTs On success, set to the timestamp of the most recent link-up event, or (time_t)-1 if none.
     * @return void
     */
    void getPtpPortLinkEvents(const std::string& syslogPath, time_t& outLastLinkDownTs, time_t& outLastLinkUpTs);
}

#endif

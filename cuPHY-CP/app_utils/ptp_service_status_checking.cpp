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
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <cstdlib>
#include <iomanip>
#include <cctype>
#include <cstring>

#include "nv_utils.h"
#include "nvlog_fmt.hpp"

#include "ptp_service_status_checking.hpp"
#include "nvlog.hpp"

using namespace AppUtils;

#define TAG (NVLOG_TAG_BASE_APP_CFG_UTILS + 2) // "APP.UTILS"

/** Sentinel for invalid or missing timestamp. */
static constexpr time_t kInvalidTimestamp = static_cast<time_t>(-1);
/** Max bytes from end of syslog to scan (tail only, for low latency). */
static constexpr long long kMaxSyslogScanBytes = 256LL * 1024;

/* Returns true if line contains needle (case-insensitive). Needle must be lowercase.
 * Uses in-place comparison to avoid allocating a copy. */
static inline bool containsSubstring(const std::string& line, const char* needle) {
    const size_t n = std::strlen(needle);
    if (line.size() < n) return false;
    for (size_t i = 0; i <= line.size() - n; ++i) {
        size_t j = 0;
        for (; j < n && std::tolower(static_cast<unsigned char>(line[i + j])) == needle[j]; ++j) {}
        if (j == n) {
            const size_t end = i + n;
            if (end >= line.size() || !std::isalpha(static_cast<unsigned char>(line[end])))
                return true;
        }
    }
    return false;
}

static bool checkIfLinkDown(const std::string& line) {
    return containsSubstring(line, "link down");
}

static bool checkIfLinkUp(const std::string& line) {
    return containsSubstring(line, "link up");
}

/* Check if syslog line indicates PTP port link down/up (ptp4l or systemd-networkd). */
static void checkPortLinkStatus(const std::string& line, bool& out_is_down, bool& out_is_up) {
    out_is_down = false;
    out_is_up = false;
    const bool has_ptp4l = (line.find("ptp4l:") != std::string::npos);
    const bool has_networkd = (line.find("systemd-networkd") != std::string::npos);
    if (has_ptp4l) {
        out_is_down = checkIfLinkDown(line);
        out_is_up = checkIfLinkUp(line);
    }
    if (has_networkd) {
        out_is_down |= checkIfLinkDown(line);
        out_is_up |= checkIfLinkUp(line);
    }
}

static time_t parseSyslogTimestamp(const std::string& line) {
    std::string timestampStr;
    size_t pos = 0;
    for (int i = 0; i < 3 && pos < line.length(); ++i) {
        while (pos < line.length() && std::isspace(line[pos])) ++pos;
        while (pos < line.length() && !std::isspace(line[pos])) {
            timestampStr += line[pos++];
        }
        if (i < 2) timestampStr += " ";
    }
    if (timestampStr.empty()) return kInvalidTimestamp;
    std::tm tm = {};
    std::istringstream ss(timestampStr);
    ss >> std::get_time(&tm, "%b %d %H:%M:%S");
    if (ss.fail()) return kInvalidTimestamp;
    time_t now = time(nullptr);
    std::tm local_tm = {};
    if (localtime_r(&now, &local_tm) != nullptr) {
        tm.tm_year = local_tm.tm_year;
        /* Syslog has no year; if parsed month is ahead of current month, entry is from previous year. */
        if (tm.tm_mon > local_tm.tm_mon)
            tm.tm_year -= 1;
        tm.tm_isdst = local_tm.tm_isdst;
    }
    return mktime(&tm);
}

void AppUtils::getPtpPortLinkEvents(const std::string& syslogPath, time_t& outLastLinkDownTs, time_t& outLastLinkUpTs) {
    outLastLinkDownTs = kInvalidTimestamp;
    outLastLinkUpTs = kInvalidTimestamp;
    try {
        std::ifstream file(syslogPath, std::ios::binary);
        if (!file.is_open()) return;
        file.seekg(0, std::ios::end);
        long long fileSize = file.tellg();
        if (fileSize <= 0) return;

        long long scanStart = (fileSize > kMaxSyslogScanBytes) ? (fileSize - kMaxSyslogScanBytes) : 0;

        std::string line;
        long long pos = fileSize - 1;
        while (pos >= scanStart) {
            file.seekg(pos, std::ios::beg);
            char c;
            if (!file.get(c)) {
                file.clear();
                --pos;
                continue;
            }
            if (c == '\n' || pos == 0) {
                long long lineStart = (pos == 0 ? 0 : pos + 1);
                file.seekg(lineStart, std::ios::beg);
                file.clear();
                if (std::getline(file, line)) {
                    bool is_down = false, is_up = false;
                    checkPortLinkStatus(line, is_down, is_up);
                    if (is_down || is_up) {
                        const time_t ts = parseSyslogTimestamp(line);
                        if (ts != kInvalidTimestamp) {
                            if (is_down && outLastLinkDownTs == kInvalidTimestamp) outLastLinkDownTs = ts;
                            if (is_up && outLastLinkUpTs == kInvalidTimestamp) outLastLinkUpTs = ts;
                            if (outLastLinkDownTs != kInvalidTimestamp && outLastLinkUpTs != kInvalidTimestamp) break;
                        }
                    }
                } else {
                    file.clear();
                }
                pos = (pos == scanStart ? scanStart - 1 : pos - 1);
            } else {
                --pos;
            }
        }
        file.close();
    } catch (const std::exception& e) {
        NVLOGW_FMT(TAG, "getPtpPortLinkEvents failed: {}", e.what());
        /* Leave out* as kInvalidTimestamp (already set above). */
    } catch (...) {
        NVLOGW_FMT(TAG, "getPtpPortLinkEvents failed: unknown exception");
        /* Leave out* as kInvalidTimestamp (already set above). */
    }
}

ServiceStatus AppUtils::checkPtpServiceStatus(const std::string& syslogPath, double rmsThreshold, const std::string& serviceName) {
    try {
        // Validate service name
        if (serviceName != "ptp4l" && serviceName != "phc2sys") {
            throw std::runtime_error("Invalid service name: " + serviceName);
        }

        // Step 1: Check service status
        std::string cmd = "systemctl is-active " + serviceName + ".service > /dev/null 2>&1";
        int systemctlResult = std::system(cmd.c_str());
        bool isRunning = (systemctlResult == 0);

        // if (!isRunning) {
        //     cmd = "busctl get-property org.freedesktop.systemd1 /org/freedesktop/systemd1/unit/" + 
        //           serviceName + "_2eservice org.freedesktop.systemd1.Unit ActiveState 2>/dev/null | grep -q 'active'";
        //     systemctlResult = std::system(cmd.c_str());
        //     isRunning = (systemctlResult == 0);
        //     if (!isRunning) {
        //         NVLOGW_FMT(TAG, "Warning: systemctl and busctl failed for {} , assuming stopped", serviceName);
        //     }
        // }

        if (!isRunning) {
            return ServiceStatus::STOPPED;
        }

        // Step 2: Open syslog file
        std::ifstream file(syslogPath, std::ios::binary);
        if (!file.is_open()) {
            throw std::runtime_error("Unable to open syslog file: " + syslogPath);
        }

        // Check file size
        file.seekg(0, std::ios::end);
        long long fileSize = file.tellg();
        if (fileSize <= 0) {
            file.close();
            throw std::runtime_error("Syslog file is empty or unreadable: " + syslogPath);
        }

        // Step 3: Parse syslog for the latest service RMS using reverse reading
        std::string lastLine;
        long long pos = fileSize - 1;
        std::string line;
        bool found = false;
        std::string servicePrefix = serviceName + ":";

        while (pos >= 0 && !found) {
            file.seekg(pos, std::ios::beg);
            char c;
            if (!file.get(c)) {
                file.clear();
                --pos;
                continue;
            }

            if (c == '\n' || pos == 0) {
                long long lineStart = (pos == 0 ? 0 : pos + 1);
                file.seekg(lineStart, std::ios::beg);
                file.clear();
                if (std::getline(file, line)) {
                    if (line.find(servicePrefix) != std::string::npos && line.find("rms") != std::string::npos) {
                        lastLine = line;
                        found = true;
                    }
                } else {
                    file.clear();
                }
                pos = (pos == 0 ? -1 : pos - 1);
            } else {
                --pos;
            }
        }

        file.close();

        // Step 4: Extract RMS from the last service line
        double rmsValue = -1.0;
        if (!lastLine.empty()) {
            try {
                time_t logTime = parseSyslogTimestamp(lastLine);
                if (logTime == kInvalidTimestamp) {
                    throw std::runtime_error("Failed to parse timestamp");
                }
                time_t currentTime = time(nullptr);
                double timeDiff = std::difftime(currentTime, logTime);
                // Use threshold > 1s to tolerate log updates every 1s and polling/IO jitter
                const double kLogStalenessThresholdSeconds = 1.5;
                if (timeDiff > kLogStalenessThresholdSeconds) {
                    throw std::runtime_error("Log timestamp is more than " + std::to_string(kLogStalenessThresholdSeconds) + " seconds old: " + std::to_string(timeDiff) + " seconds");
                }
            } catch (const std::exception& e) {
                throw std::runtime_error("Error parsing timestamp in " + serviceName + " line: (" + lastLine + ") : " + e.what());
            }


            size_t rmsPos = lastLine.find("rms");
            try {
                size_t pos = rmsPos + 3;
                while (pos < lastLine.length() && std::isspace(lastLine[pos])) {
                    ++pos;
                }
                std::string numStr;
                while (pos < lastLine.length() && (std::isdigit(lastLine[pos]) || lastLine[pos] == '.' || lastLine[pos] == '-')) {
                    numStr += lastLine[pos++];
                }
                if (numStr.empty()) {
                    throw std::runtime_error("No number found after 'rms' in " + serviceName + " line: " + lastLine);
                }
                rmsValue = std::stod(numStr);
            } catch (const std::exception& e) {
                throw std::runtime_error("Error parsing " + serviceName + " RMS in line: " + lastLine + ": " + e.what());
            }
        } else {
            NVLOGW_FMT(TAG, "Warning: No {} lines with 'rms' found in {}", serviceName, syslogPath);
            throw std::runtime_error("No " + serviceName + " synchronization data available");
        }

        // Step 5: Determine status
        if (rmsValue < rmsThreshold) {
            return ServiceStatus::RUNNING_SYNCED;
        }
        NVLOGI_FMT(TAG, "{}.service: current rms: {} ns, rmsThreshold {} ns)", serviceName, rmsValue, rmsThreshold);
        return ServiceStatus::RUNNING_UNSYNCED;

    } catch (const std::exception& e) {
        NVLOGW_FMT(TAG, "Error: {} ", e.what());
        return ServiceStatus::ERROR;
    }
}

int AppUtils::checkPtpServiceStatus(const std::string& syslogPath, double ptp4lRmsThreshold, double phc2sysRmsThreshold) {
    int ret1 = -1;
    // Check ptp4l.service
    //ServiceStatus ptp4lStatus = checkPtp4lStatus(syslogPath, ptp4lRmsThreshold);
    ServiceStatus ptp4lStatus = checkPtpServiceStatus(syslogPath, ptp4lRmsThreshold, "ptp4l");
    switch (ptp4lStatus)
    {
    case ServiceStatus::RUNNING_SYNCED:
        NVLOGI_FMT(TAG, "ptp4l.service: running and synchronized (RMS < {} ns)", ptp4lRmsThreshold);
        ret1 = 0;
        break;
    case ServiceStatus::RUNNING_UNSYNCED:
        NVLOGW_FMT(TAG, "ptp4l.service: running but rms larger than threshold (RMS >= {} ns)", ptp4lRmsThreshold);
        break;
    case ServiceStatus::STOPPED:
        NVLOGW_FMT(TAG, "ptp4l.service: stopped");
        break;
    case ServiceStatus::ERROR:
        NVLOGW_FMT(TAG, "ptp4l.service: error checking status");
        break;
    }

    int ret2 = -1;
    // Check phc2sys.service
    // ServiceStatus phc2sysStatus = checkPhc2sysStatus(syslogPath, phc2sysRmsThreshold);
    ServiceStatus phc2sysStatus = checkPtpServiceStatus(syslogPath, phc2sysRmsThreshold, "phc2sys");
    switch (phc2sysStatus)
    {
    case ServiceStatus::RUNNING_SYNCED:
        NVLOGI_FMT(TAG, "phc2sys.service: running and synchronized (RMS < {} ns)", phc2sysRmsThreshold);
        ret2 = 0;
        break;
    case ServiceStatus::RUNNING_UNSYNCED:
        NVLOGW_FMT(TAG, "phc2sys.service: running but rms larger than threshold (RMS >= {} ns)", phc2sysRmsThreshold);
        break;
    case ServiceStatus::STOPPED:
        NVLOGW_FMT(TAG, "phc2sys.service: stopped");
        break;
    case ServiceStatus::ERROR:
        NVLOGW_FMT(TAG, "phc2sys.service: error checking status");
        break;
    }

    return (ret1 || ret2) ? -1 : 0;
}

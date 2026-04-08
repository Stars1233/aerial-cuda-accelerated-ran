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

#ifndef _CUMAC_VALIDATE_HPP_
#define _CUMAC_VALIDATE_HPP_

#include "nvlog.hpp"
#include "cumac_defines.hpp"

// Max invalid items to print in VALD_LOG_PER_MSG log
#define MAX_PER_MSG_LOG_COUNT 3

#define VALD_LOG_BUF_SIZE 1000

class cumac_validate {
public:
    // 0 - disabled; 1 - report error level; 2 - report error and warning levels
    int enable = VALD_ENABLE_ERR;

    // 0 - no print; 1 - print per MSG; 2 - print per PDU
    int log_opt = 1;

    cumac_validate();
    cumac_validate(int _enable, int _log_opt);
    ~cumac_validate();

    const char* get_msg_name();

    // Call before start validation
    void msg_start(int cell_id, int msg_id, uint16_t sfn, uint16_t slot);

    int pdu_start(int pdu_id, int channel);
    int pdu_start(int pdu_id, int channel, cumac_req_t* req);

    // Call after one PDU ended
    int pdu_ended(int pdu_id, int ind_id);

    // Call after one CUMAC message ended
    int msg_ended();

    void set_cumac_req(cumac_req_t* req)
    {
        cumac_req = req;
    }

    cumac_req_t* get_cumac_req()
    {
        return cumac_req;
    }

    // Report a text string in log
    void log_text(const char* text);

    // Report a mismatched value or bytes in log
    void log_value(int level, const char* name1, const char* name2, uint32_t val1, uint32_t val2);
    void log_value(int level, const char* name1, const char* name2, uint16_t val1, uint16_t val2);
    void log_value(int level, const char* name1, const char* name2, uint8_t val1, uint8_t val2);
    void log_value(int level, const char* name1, const char* name2, int32_t val1, int32_t val2);
    void log_value(int level, const char* name1, const char* name2, int16_t val1, int16_t val2);
    void log_value(int level, const char* name1, const char* name2, int8_t val1, int8_t val2);
    void log_value(int level, const char* name1, const char* name2, float val1, float val2);
    void log_value(int level, const char* name1, const char* name2, float2* val1, float2* val2, uint len);
    void log_bytes(int level, const char* name1, const char* name2, void* buf1, void* buf2, vald_result_t result);

    // Check whether to log per configuration
    bool should_log(vald_result_t result)
    {
        // Force log all if configured: log_opt = VALD_LOG_PRINT_ALL
        if(log_opt == VALD_LOG_PRINT_ALL)
        {
            return true;
        }

        if(result == VALD_OK)
        {
            // Validate succeeded, no log
            return false;
        }
        else if(log_opt == VALD_LOG_PER_PDU || (log_opt == VALD_LOG_PER_MSG && log_count++ < MAX_PER_MSG_LOG_COUNT))
        {
            // Validate failed, enable log per PDU or log per message
            return true;
        }
        return false;
    }

    // Check whether to report error per configuration
    int report(int level)
    {
        if(enable >= level)
        {
            err_count++;
            return VALD_FAIL;
        }
        else
        {
            warn_count++;
            return VALD_OK;
        }
    }

    // Report with printf style message
    int report_text(int level, const char* fmt, ...)
    {
        if(should_log(VALD_FAIL))
        {
            if(offset < VALD_LOG_BUF_SIZE - 3)
            {
                offset += snprintf(errbuf + offset, VALD_LOG_BUF_SIZE - offset, " [");
                if(level == VALD_ENABLE_WARN) {
                    offset += snprintf(errbuf + offset, VALD_LOG_BUF_SIZE - offset, "wrn, ");
                } else if(level == VALD_ENABLE_ERR) {
                    offset += snprintf(errbuf + offset, VALD_LOG_BUF_SIZE - offset, "err, ");
                }

                va_list va;
                va_start(va, fmt);
                offset += vsnprintf(errbuf + offset, VALD_LOG_BUF_SIZE - offset, fmt, va);
                va_end(va);
                if(offset < VALD_LOG_BUF_SIZE)
                {
                    offset += snprintf(errbuf + offset, VALD_LOG_BUF_SIZE - offset, "]");
                }
            }
        }
        return report(level);
    }

    // Report if the 2 values don't match
    template <typename T>
    int check_value(int level, const char* name1, const char* name2, T val1, T val2, T tolerance = 0)
    {
        // Use int64_t to avoid overflow
        int64_t diff = val1 > val2 ? val1 : val2;
        diff -= val1 > val2 ? val2 : val1;

        vald_result_t check_result = VALD_OK;
        if(diff > tolerance)
        {
            check_result = VALD_FAIL;
        }

        if(should_log(check_result))
        {
            log_value(level, name1, name2, val1, val2);
        }

        return check_result == VALD_OK ? 0 : report(level);
    }

    int check_approx_value(int level, const char* name1, const char* name2, float val1, float val2, float tolerance = 0.0f)
    {
        // Use int64_t to avoid overflow
        // int64_t diff = val1 > val2 ? val1 : val2;
        // diff -= val1 > val2 ? val2 : val1;

        float       diff      = fabs(val1 - val2);
        vald_result_t check_result = VALD_OK;
        if(diff >= tolerance)
        {
            check_result = VALD_FAIL;
        }

        if(should_log(check_result))
        {
            log_value(level, name1, name2, val1, val2);
        }

        return check_result == VALD_OK ? 0 : report(level);
    }

    // check if E{|val1 - val2|^2 / |val2|^2} < tolerance
    int check_complex_approx_ratio(int level, const char* name1, const char* name2, float2* val1, float2* val2, uint len, float tolerance = 0.001f)
    {
        // use FP32 in comparison
        float ratioDiff = 0.0f;
        // calculate ratioDiff and avg over len
        for(uint idx = 0; idx < len; idx++)
        {
            float2 diff{val1[idx].x - val2[idx].x, val1[idx].y - val2[idx].y};
            ratioDiff += (diff.x * diff.x + diff.y * diff.y) / float(val2[idx].x * val2[idx].x + val2[idx].y * val2[idx].y);
        }
        ratioDiff /= len;

        vald_result_t check_result = VALD_OK;

        if(ratioDiff >= tolerance)
        {
            check_result = VALD_FAIL;
        }
           
        if(should_log(check_result))
        {
            log_value(level, name1, name2, val1, val2, len);
        }

        return check_result == VALD_OK ? 0 : report(level);
    }

    // Report if contents of the 2 buffers are not the same
    int check_bytes(int level, const char* name1, const char* name2, void* buf1, void* buf2, size_t nbytes)
    {
        vald_result_t check_result = VALD_OK;
        if(buf1 == nullptr || buf2 == nullptr || memcmp(buf1, buf2, nbytes))
        {
            check_result = VALD_FAIL;
        }

        if(should_log(check_result))
        {
            log_bytes(level, name1, name2, buf1, buf2, check_result);
        }

        return check_result == VALD_OK ? 0 : report(level);
    }

    int  cell_id = 0;    //!< Cell ID

protected:
    char errbuf[VALD_LOG_BUF_SIZE] = "";    //!< Error buffer
    int  offset = 0;        //!< Offset in error buffer
    int  msg_id = 0;        //!< Message ID

    uint16_t sfn = 0;        //!< SFN
    uint16_t slot = 0;        //!< Slot
    uint32_t err_count = 0;    //!< Error count
    uint32_t err_pdu = 0;    //!< Error PDU count
    uint32_t warn_count = 0; //!< Warning count
    uint32_t warn_pdu = 0;   //!< Warning PDU count
    uint32_t log_count = 0;  //!< Log count

    cumac_thrput_t* thrput = nullptr;    //!< Throughput
    cumac_req_t*    cumac_req = nullptr;    //!< CUMAC request
};

// Report fail with text string
#define CUMAC_VALIDATE_TEXT_ERR(vald, fmt, ...) \
    ((vald)->report_text(VALD_ENABLE_ERR, fmt, ##__VA_ARGS__))
#define CUMAC_VALIDATE_TEXT_WARN(vald, fmt, ...) \
    ((vald)->report_text(VALD_ENABLE_WARN, fmt, ##__VA_ARGS__))

// Validate uint32_t value
#define CUMAC_VALIDATE_U32_ERR(vald, val1, val2, ...) \
    ((vald)->check_value<uint32_t>(VALD_ENABLE_ERR, #val1, #val2, (val1), (val2), ##__VA_ARGS__))
#define CUMAC_VALIDATE_U32_WARN(vald, val1, val2, ...) \
    ((vald)->check_value<uint32_t>(VALD_ENABLE_WARN, #val1, #val2, (val1), (val2), ##__VA_ARGS__))

// Validate int32_t value
#define CUMAC_VALIDATE_I32_ERR(vald, val1, val2, ...) \
    ((vald)->check_value<int32_t>(VALD_ENABLE_ERR, #val1, #val2, (val1), (val2), ##__VA_ARGS__))
#define CUMAC_VALIDATE_I32_WARN(vald, val1, val2, ...) \
    ((vald)->check_value<int32_t>(VALD_ENABLE_WARN, #val1, #val2, (val1), (val2), ##__VA_ARGS__))

// Validate uint16_t value
#define CUMAC_VALIDATE_U16_ERR(vald, val1, val2, ...) \
    ((vald)->check_value<uint16_t>(VALD_ENABLE_ERR, #val1, #val2, (val1), (val2), ##__VA_ARGS__))
#define CUMAC_VALIDATE_U16_WARN(vald, val1, val2, ...) \
    ((vald)->check_value<uint16_t>(VALD_ENABLE_WARN, #val1, #val2, (val1), (val2), ##__VA_ARGS__))

// Validate int16_t value
#define CUMAC_VALIDATE_I16_ERR(vald, val1, val2, ...) \
    ((vald)->check_value<int16_t>(VALD_ENABLE_ERR, #val1, #val2, (val1), (val2), ##__VA_ARGS__))
#define CUMAC_VALIDATE_I16_WARN(vald, val1, val2, ...) \
    ((vald)->check_value<int16_t>(VALD_ENABLE_WARN, #val1, #val2, (val1), (val2), ##__VA_ARGS__))

// Validate uint8_t value
#define CUMAC_VALIDATE_U8_ERR(vald, val1, val2, ...) \
    ((vald)->check_value<uint8_t>(VALD_ENABLE_ERR, #val1, #val2, (val1), (val2), ##__VA_ARGS__))
#define CUMAC_VALIDATE_U8_WARN(vald, val1, val2, ...) \
    ((vald)->check_value<uint8_t>(VALD_ENABLE_WARN, #val1, #val2, (val1), (val2), ##__VA_ARGS__))

// Validate int8_t value
#define CUMAC_VALIDATE_I8_ERR(vald, val1, val2, ...) \
    ((vald)->check_value<int8_t>(VALD_ENABLE_ERR, #val1, #val2, (val1), (val2), ##__VA_ARGS__))
#define CUMAC_VALIDATE_I8_WARN(vald, val1, val2, ...) \
    ((vald)->check_value<int8_t>(VALD_ENABLE_WARN, #val1, #val2, (val1), (val2), ##__VA_ARGS__))

// Validate float value
#define CUMAC_VALIDATE_FLOAT_ERR(vald, val1, val2, ...) \
    ((vald)->check_value<float>(VALD_ENABLE_ERR, #val1, #val2, (val1), (val2), ##__VA_ARGS__))
#define CUMAC_VALIDATE_FLOAT_WARN(vald, val1, val2, ...) \
    ((vald)->check_value<float>(VALD_ENABLE_WARN, #val1, #val2, (val1), (val2), ##__VA_ARGS__))

// Validate float value
#define CUMAC_VALIDATE_FLOAT_APPROX_ERR(vald, val1, val2, ...) \
    ((vald)->check_approx_value(VALD_ENABLE_ERR, #val1, #val2, (val1), (val2), ##__VA_ARGS__))
#define CUMAC_VALIDATE_FLOAT_APPROX_WARN(vald, val1, val2, ...) \
    ((vald)->check_approx_value(VALD_ENABLE_WARN, #val1, #val2, (val1), (val2), ##__VA_ARGS__))

// Validate payload bytes
#define CUMAC_VALIDATE_BYTES_ERR(vald, buf1, buf2, nbytes) \
    (vald)->check_bytes(VALD_ENABLE_ERR, #buf1, #buf2, (buf1), (buf2), nbytes)
#define CUMAC_VALIDATE_BYTES_WARN(vald, buf1, buf2, nbytes) \
    (vald)->check_bytes(VALD_ENABLE_WARN, #buf1, #buf2, (buf1), (buf2), nbytes)

// Validate complex value
#define CUMAC_VALIDATE_COMPLEX_APPROX_RATIO_ERR(vald, val1, val2, len, ...) \
    ((vald)->check_complex_approx_ratio(VALD_ENABLE_ERR, #val1, #val2, (val1), (val2), len, ##__VA_ARGS__))
#define CUMAC_VALIDATE_COMPLEX_APPROX_RATIO_WARN(vald, val1, val2, ...) \
    ((vald)->check_complex_approx_ratio(VALD_ENABLE_WARN, #val1, #val2, (val1), (val2), len, ##__VA_ARGS__))

#endif /* _CUMAC_VALIDATE_HPP_ */

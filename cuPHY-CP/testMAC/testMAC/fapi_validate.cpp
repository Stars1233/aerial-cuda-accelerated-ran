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

#include "fapi_handler.hpp"
#include "fapi_validate.hpp"
#include "scf_5g_fapi.h"

#define TAG (NVLOG_TAG_BASE_TEST_MAC + 8) // "MAC.VALD"

const char* fapi_validate::get_msg_name()
{
    return get_scf_fapi_msg_name(msg_id);
}

fapi_validate::fapi_validate()
{
    // 0 - disabled; 1 - report error level; 2 - report error and warning level
    enable = VALD_ENABLE_NONE;

    // validate_log_opt: 0 - no print; 1 - print per MSG; 2 - print per PDU; 3 - force print all
    log_opt = 1;

    fapi_req = nullptr;
    thrput   = nullptr;
    cell_id  = 0;
    sfn      = 0;
    slot     = 0;
    offset   = 0;
    msg_id   = 0;

    err_count  = 0;
    err_pdu    = 0;
    warn_count = 0;
    warn_pdu   = 0;
    log_count  = 0;
}

fapi_validate::fapi_validate(int _enable, int _log_opt) : fapi_validate()
{
    // 0 - disabled; 1 - report error level; 2 - report error and warning level
    enable = _enable;

    // validate_log_opt: 0 - no print; 1 - print per MSG; 2 - print per PDU; 3 - force print all
    log_opt = _log_opt;
}

fapi_validate::~fapi_validate()
{
}

void fapi_validate::msg_start(int cell_id, int msg_id, uint16_t sfn, uint16_t slot)
{
    this->cell_id               = cell_id;
    this->msg_id                = msg_id;
    this->sfn                   = sfn;
    this->slot                  = slot;
    offset                      = 0;
    err_count                   = 0;
    err_pdu                     = 0;
    warn_count                  = 0;
    warn_pdu                    = 0;
    log_count                   = 0;
    fapi_handler* _fapi_handler = get_fapi_handler_instance();
    thrput                      = _fapi_handler->get_thrput(cell_id);
    fapi_req                    = nullptr;
}

int fapi_validate::pdu_start(int pdu_id, int channel)
{
    if(fapi_req == nullptr)
    {
        report_text(VALD_ENABLE_ERR, "%s TV not found", get_channel_name(channel));
        pdu_ended(pdu_id, -1);
        return -1;
    }

    if (fapi_req->tv_data == nullptr)
    {
        return 0;
    }

    int size = 0;
    switch(channel)
    {
    case channel_type_t::PUSCH:
        size = fapi_req->tv_data->pusch_tv.data.size();
        break;
    case channel_type_t::PUCCH:
        size = fapi_req->tv_data->pucch_tv.data.size();
        break;
    case channel_type_t::PRACH:
        size = fapi_req->tv_data->prach_tv.data.size();
        break;
    case channel_type_t::SRS:
        size = fapi_req->tv_data->srs_tv.data.size();
        break;
    default:
        break;
    }

    if(pdu_id >= size)
    {
        report_text(VALD_ENABLE_ERR, "%s pdu_id=%d exceeds size=%d", get_channel_name(channel), pdu_id, size);
        pdu_ended(pdu_id, -1);
        return -1;
    }
    return 0;
}

int fapi_validate::pdu_start(int pdu_id, int channel, fapi_req_t* req)
{
    fapi_req = req;
    return pdu_start(pdu_id, channel);
}

int fapi_validate::pdu_ended(int pdu_id, int ind_id)
{
    // Print validation message for VALD_LOG_PER_PDU and VALD_LOG_PRINT_ALL cases
    if(offset > 0 && log_opt >= VALD_LOG_PER_PDU)
    {
        NVLOGI_FMT(TAG, "SFN {}.{} Cell {} {} PDU{} mismatch: {}/IND{}{}", sfn, slot, cell_id, get_scf_fapi_msg_name(msg_id), pdu_id, fapi_req == nullptr ? "TV_Unknown" : fapi_req->tv_file.c_str(), ind_id, errbuf);
        offset = 0;
    }

    if(warn_count > 0)
    {
        warn_count = 0;
        warn_pdu++;
    }

    int ret = err_count;
    if(err_count > 0)
    {
        err_count = 0;
        err_pdu++;
    }

    return ret;
}

int fapi_validate::msg_ended()
{
    // End the last PDU if not ended
    pdu_ended(-1, -1);

    // Print validation message for VALD_LOG_PER_MSG case
    if(offset > 0 && log_opt == VALD_LOG_PER_MSG)
    {
        NVLOGI_FMT(TAG, "SFN {}.{} Cell {} {} mismatch: {} err {} warn{}", sfn, slot, cell_id, get_scf_fapi_msg_name(msg_id), err_pdu, warn_pdu, errbuf);
    }
    log_count = 0;

    int ret = err_pdu;
    if(err_pdu > 0)
    {
        thrput->invalid += err_pdu;
        err_pdu = 0;
    }

    return ret;
}

void fapi_validate::log_text(const char* text)
{
    if(offset < VALD_LOG_BUF_SIZE)
    {
        offset += snprintf(errbuf + offset, VALD_LOG_BUF_SIZE - offset, " [%s]", text);
    }
}

void fapi_validate::log_value(int level, const char* name1, const char* name2, uint32_t val1, uint32_t val2)
{
    if(offset < VALD_LOG_BUF_SIZE)
    {
        if(level == VALD_ENABLE_ERR) {
            offset += snprintf(errbuf + offset, VALD_LOG_BUF_SIZE - offset, " [err, ");
        } else if(level == VALD_ENABLE_WARN) {
            offset += snprintf(errbuf + offset, VALD_LOG_BUF_SIZE - offset, " [wrn, ");
        }
    }

    if(offset < VALD_LOG_BUF_SIZE)
    {
        offset += snprintf(errbuf + offset, VALD_LOG_BUF_SIZE - offset, "%s=%u %s=%u]", name1, val1, name2, val2);
    }
}

void fapi_validate::log_value(int level, const char* name1, const char* name2, uint16_t val1, uint16_t val2)
{
    if(offset < VALD_LOG_BUF_SIZE)
    {
        if(level == VALD_ENABLE_ERR) {
            offset += snprintf(errbuf + offset, VALD_LOG_BUF_SIZE - offset, " [err, ");
        } else if(level == VALD_ENABLE_WARN) {
            offset += snprintf(errbuf + offset, VALD_LOG_BUF_SIZE - offset, " [wrn, ");
        }
    }

    if(offset < VALD_LOG_BUF_SIZE)
    {
        offset += snprintf(errbuf + offset, VALD_LOG_BUF_SIZE - offset, "%s=%u %s=%u]", name1, val1, name2, val2);
    }
}

void fapi_validate::log_value(int level, const char* name1, const char* name2, uint8_t val1, uint8_t val2)
{
    if(offset < VALD_LOG_BUF_SIZE)
    {
        if(level == VALD_ENABLE_ERR) {
            offset += snprintf(errbuf + offset, VALD_LOG_BUF_SIZE - offset, " [err, ");
        } else if(level == VALD_ENABLE_WARN) {
            offset += snprintf(errbuf + offset, VALD_LOG_BUF_SIZE - offset, " [wrn, ");
        }
    }

    if(offset < VALD_LOG_BUF_SIZE)
    {
        offset += snprintf(errbuf + offset, VALD_LOG_BUF_SIZE - offset, "%s=%u %s=%u]", name1, val1, name2, val2);
    }
}

void fapi_validate::log_value(int level, const char* name1, const char* name2, int32_t val1, int32_t val2)
{
    if(offset < VALD_LOG_BUF_SIZE)
    {
        if(level == VALD_ENABLE_ERR) {
            offset += snprintf(errbuf + offset, VALD_LOG_BUF_SIZE - offset, " [err, ");
        } else if(level == VALD_ENABLE_WARN) {
            offset += snprintf(errbuf + offset, VALD_LOG_BUF_SIZE - offset, " [wrn, ");
        }
    }

    if(offset < VALD_LOG_BUF_SIZE)
    {
        offset += snprintf(errbuf + offset, VALD_LOG_BUF_SIZE - offset, "%s=%d %s=%d]", name1, val1, name2, val2);
    }
}

void fapi_validate::log_value(int level, const char* name1, const char* name2, int16_t val1, int16_t val2)
{
    if(offset < VALD_LOG_BUF_SIZE)
    {
        if(level == VALD_ENABLE_ERR) {
            offset += snprintf(errbuf + offset, VALD_LOG_BUF_SIZE - offset, " [err, ");
        } else if(level == VALD_ENABLE_WARN) {
            offset += snprintf(errbuf + offset, VALD_LOG_BUF_SIZE - offset, " [wrn, ");
        }
    }

    if(offset < VALD_LOG_BUF_SIZE)
    {
        offset += snprintf(errbuf + offset, VALD_LOG_BUF_SIZE - offset, "%s=%d %s=%d]", name1, val1, name2, val2);
    }
}

void fapi_validate::log_value(int level, const char* name1, const char* name2, int8_t val1, int8_t val2)
{
    if(offset < VALD_LOG_BUF_SIZE)
    {
        if(level == VALD_ENABLE_ERR) {
            offset += snprintf(errbuf + offset, VALD_LOG_BUF_SIZE - offset, " [err, ");
        } else if(level == VALD_ENABLE_WARN) {
            offset += snprintf(errbuf + offset, VALD_LOG_BUF_SIZE - offset, " [wrn, ");
        }
    }

    if(offset < VALD_LOG_BUF_SIZE)
    {
        offset += snprintf(errbuf + offset, VALD_LOG_BUF_SIZE - offset, "%s=%d %s=%d]", name1, val1, name2, val2);
    }
}

void fapi_validate::log_value(int level, const char* name1, const char* name2, float val1, float val2)
{
    if(offset < VALD_LOG_BUF_SIZE)
    {
        if(level == VALD_ENABLE_ERR) {
            offset += snprintf(errbuf + offset, VALD_LOG_BUF_SIZE - offset, " [err, ");
        } else if(level == VALD_ENABLE_WARN) {
            offset += snprintf(errbuf + offset, VALD_LOG_BUF_SIZE - offset, " [wrn, ");
        }
    }

    if(offset < VALD_LOG_BUF_SIZE)
    {
        offset += snprintf(errbuf + offset, VALD_LOG_BUF_SIZE - offset, "%s=%f %s=%f]", name1, val1, name2, val2);
    }
}


void fapi_validate::log_value(int level, const char* name1, const char* name2, float2* val1, float2* val2, uint len)
{
    for(uint idx = 0; idx < len; idx ++)
    {
        if(offset < VALD_LOG_BUF_SIZE)
        {
            if(level == VALD_ENABLE_ERR) {
                offset += snprintf(errbuf + offset, VALD_LOG_BUF_SIZE - offset, " [err, ");
            } else if(level == VALD_ENABLE_WARN) {
                offset += snprintf(errbuf + offset, VALD_LOG_BUF_SIZE - offset, " [wrn, ");
            }
        }

        if(offset < VALD_LOG_BUF_SIZE)
        {
            offset += snprintf(errbuf + offset, VALD_LOG_BUF_SIZE - offset, "%s[%d]=%f + %f i, %s[%d]=%f + %f i]", name1, idx, val1[idx].x, val1[idx].y, name2, idx, val2[idx].x, val2[idx].y);
        }
    }
}

void fapi_validate::log_value(int level, const char* name1, const char* name2, short2* val1, float2* val2, uint len)
{
    for(uint idx = 0; idx < len; idx ++)
    {
        if(offset < VALD_LOG_BUF_SIZE)
        {
            if(level == VALD_ENABLE_ERR) {
                offset += snprintf(errbuf + offset, VALD_LOG_BUF_SIZE - offset, " [err, ");
            } else if(level == VALD_ENABLE_WARN) {
                offset += snprintf(errbuf + offset, VALD_LOG_BUF_SIZE - offset, " [wrn, ");
            }
        }

        if(offset < VALD_LOG_BUF_SIZE)
        {
            offset += snprintf(errbuf + offset, VALD_LOG_BUF_SIZE - offset, "%s[%d]=%d + %d i, %s[%d]=%f + %f i]", name1, idx, val1[idx].x, val1[idx].y, name2, idx, val2[idx].x, val2[idx].y);
        }
    }
}

void fapi_validate::log_bytes(int level, const char* name1, const char* name2, void* buf1, void* buf2, vald_result_t result)
{
    if(offset < VALD_LOG_BUF_SIZE)
    {

        if(level == VALD_ENABLE_ERR) {
            offset += snprintf(errbuf + offset, VALD_LOG_BUF_SIZE - offset, " [err, ");
        } else if(level == VALD_ENABLE_WARN) {
            offset += snprintf(errbuf + offset, VALD_LOG_BUF_SIZE - offset, " [wrn, ");
        }
    }

    if(offset < VALD_LOG_BUF_SIZE)
    {
        if(buf1 == nullptr || buf2 == nullptr)
        {
            offset += snprintf(errbuf + offset, VALD_LOG_BUF_SIZE - offset, "%s=%p %s=%p]", name1, buf1, name2, buf2);
        }
        else
        {
            if(result == VALD_OK)
            {
                offset += snprintf(errbuf + offset, VALD_LOG_BUF_SIZE - offset, "bytes same: %s[0]=0x%02X]", name1, *(uint8_t*)buf1);
            }
            else
            {
                offset += snprintf(errbuf + offset, VALD_LOG_BUF_SIZE - offset, "bytes differ: %s[0]=0x%02X %s[0]=0x%02X]", name1, *(uint8_t*)buf1, name2, *(uint8_t*)buf2);
            }
        }
    }
}

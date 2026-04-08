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

#include <stdint.h>
#include <stddef.h>
#include <cumac_msg.h>

/**
 * Convert cuMAC message ID to human-readable string
 *
 * Provides a lookup table for cuMAC message type identifiers,
 * useful for logging and debugging purposes.
 *
 * @param[in] msg_id Message type identifier from cumac_msg_t enum
 *
 * @return Pointer to constant string containing message name.
 *         Returns "UNKNOWN_CUMAC_MSG" for unrecognized message IDs.
 */
const char* get_cumac_msg_name(int msg_id)
{
    switch(msg_id)
    {
    case CUMAC_PARAM_REQUEST:
        return "PARAM.req";
    case CUMAC_PARAM_RESPONSE:
        return "PARAM.resp";

    case CUMAC_CONFIG_REQUEST:
        return "CONFIG.req";
    case CUMAC_CONFIG_RESPONSE:
        return "CONFIG.resp";

    case CUMAC_START_REQUEST:
        return "START.req";
    case CUMAC_START_RESPONSE:
        return "START.resp";

    case CUMAC_STOP_REQUEST:
        return "STOP.req";
    case CUMAC_STOP_RESPONSE:
        return "STOP.resp";

    case CUMAC_ERROR_INDICATION:
        return "ERR.ind";

    case CUMAC_TTI_ERROR_INDICATION:
        return "TTI_ERR.ind";
    case CUMAC_DL_TTI_REQUEST:
        return "DL_TTI.req";
    case CUMAC_UL_TTI_REQUEST:
        return "UL_TTI.req";

    case CUMAC_SCH_TTI_REQUEST:
        return "SCH_TTI.req";
    case CUMAC_SCH_TTI_RESPONSE:
        return "SCH_TTI.resp";

    case CUMAC_TTI_END:
        return "TTI_END.req";

    default:
        return "UNKNOWN_CUMAC_MSG";
    }
}
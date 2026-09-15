/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef SCF_5G_FAPI_CONFIG_DECODE_HPP
#define SCF_5G_FAPI_CONFIG_DECODE_HPP

#include "scf_5g_fapi.h"               // scf_fapi_config_request_msg_t, scf_fapi_tl_t, CONFIG_TLV_*
#include "nv_phy_fapi_msg_common.hpp"  // nv::cell_update_config

namespace scf_5g_fapi
{

/**
 * Decodes the reconfigurable subset of CONFIG.request TLVs into @p out.
 *
 * Shared by the serial (phy::on_config_request) and offload
 * (phy::on_config_request_offload) reconfiguration paths so the two TLV
 * decoders can never silently diverge.
 *
 * This is a pure decode step: it writes only @p out (and returns whether a DBT
 * PDU is present) and touches no PHY/driver/global state, so it can be
 * unit-tested with a synthetic TLV buffer. The actual DBT-PDU store,
 * phyCellId-mismatch check, and l1_cell_update_cell_config dispatch remain in
 * the callers.
 *
 * @param[in]  config_request The CONFIG.request message.
 * @param[out] out           Reconfiguration target; only the reconfigurable
 *                            carrier/cell/PRACH fields present in the request are written.
 *                            Passed by reference (large struct) rather than returned.
 * @return true iff a VENDOR_DIGITAL_BEAM_TABLE_PDU TLV is present.
 */
[[nodiscard]] bool decode_reconfig_tlvs(const scf_fapi_config_request_msg_t& config_request,
                                        nv::cell_update_config&              out);

} // namespace scf_5g_fapi

#endif // SCF_5G_FAPI_CONFIG_DECODE_HPP

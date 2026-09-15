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

#include "scf_5g_fapi.hpp"
#include "scf_5g_fapi_mac.hpp"
#include "scf_5g_fapi_phy.hpp"
#include "nv_mac_factory.hpp"
#include "nv_phy_factory.hpp"
#include "nv_phy_module.hpp"


const char* get_fapi_msg_name(int32_t msg_id)
{
    switch(msg_id)
    {
    case SCF_FAPI_PARAM_REQUEST:
        return "PARAM.req";
    case SCF_FAPI_PARAM_RESPONSE:
        return "PARAM.resp";
    case SCF_FAPI_CONFIG_REQUEST:
        return "CONFIG.req";
    case SCF_FAPI_CONFIG_RESPONSE:
        return "CONFIG.resp";

    case SCF_FAPI_START_REQUEST:
        return "START.req";
    case SCF_FAPI_STOP_REQUEST:
        return "STOP.req";
    case SCF_FAPI_STOP_INDICATION:
        return "STOP.ind";
    case SCF_FAPI_ERROR_INDICATION:
        return "ERR.ind";

    case SCF_FAPI_SLOT_INDICATION:
        return "SLOT.ind";

    case SCF_FAPI_DL_TTI_REQUEST:
        return "DL_TTI.req";
    case SCF_FAPI_UL_TTI_REQUEST:
        return "UL_TTI.req";
    case SCF_FAPI_TX_DATA_REQUEST:
        return "TX_DATA.req";
    case SCF_FAPI_UL_DCI_REQUEST:
        return "UL_DCI.req";
    case SCF_FAPI_DL_BFW_CVI_REQUEST:
        return "DL_BFW_CVI.req";
    case SCF_FAPI_UL_BFW_CVI_REQUEST:
        return "UL_BFW_CVI.req";
    case SCF_FAPI_RX_DATA_INDICATION:
        return "RX_DATA.ind";
    case SCF_FAPI_CRC_INDICATION:
        return "CRC.ind";
    case SCF_FAPI_UCI_INDICATION:
        return "UCI.ind";
    case SCF_FAPI_SRS_INDICATION:
        return "SRS.ind";
    case SCF_FAPI_RACH_INDICATION:
        return "RACH.ind";

    case SCF_FAPI_RX_PE_NOISE_VARIANCE_INDICATION:
        return "PE_NOISE_VARIANCE.ind";
    case SCF_FAPI_RX_PF_234_INTEFERNCE_INDICATION:
        return "PF_234_INTERFERENCE.ind";
    case SCF_FAPI_RX_PRACH_INTEFERNCE_INDICATION:
        return "PRACH_INTERFERENCE.ind";

    case SCF_FAPI_SLOT_RESPONSE:
        return "SLOT.resp";
    case CV_MEM_BANK_CONFIG_REQUEST:
        return "CV_MEM_BANK_CONFIG.req";
    case CV_MEM_BANK_CONFIG_RESPONSE:
        return "CV_MEM_BANK_CONFIG.resp";

    default:
        return "UNKNOWN_SCF_FAPI";
    }
}

namespace
{

/**
 * @brief Create a SCF 5G FAPI MAC instance
 * @param node_config YAML configuration node
 * @param cell_num Cell number
 * @return Pointer to the created MAC instance
 */
nv::mac* create_scf_5g_fapi_mac(yaml::node node_config, uint32_t cell_num)
{
    return new scf_5g_fapi::mac(node_config, cell_num);
}

/// MAC creator structure for SCF 5G FAPI
nv::mac_creator scf_5g_fapi_mac_creator =
{
    "scf_5g_fapi",
    &create_scf_5g_fapi_mac
};

/**
 * @brief Create a SCF 5G FAPI PHY instance
 * @param phy_module Reference to the parent PHY module
 * @param node_config YAML configuration node
 * @return Unique pointer to the created PHY instance
 */
std::unique_ptr<nv::PHY_instance> create_scf_5g_fapi_phy(nv::PHY_module& phy_module,
                                         yaml::node      node_config)
{
    return std::make_unique<scf_5g_fapi::phy>(phy_module, node_config);
}

/// PHY creator structure for SCF 5G FAPI
nv::phy_creator scf_5g_fapi_phy_creator =
{
    "scf_5g_fapi",
    &create_scf_5g_fapi_phy
};

} // namespace


namespace scf_5g_fapi
{

/**
 * @brief Initialize SCF 5G FAPI module
 *
 * Registers MAC and PHY creators with their respective factories
 * to enable runtime instantiation.
 */
void init()
{
    nv::mac_factory::register_type(scf_5g_fapi_mac_creator);
    nv::phy_factory::register_type(scf_5g_fapi_phy_creator);
}
    
} // namespace scf_5g_fapi


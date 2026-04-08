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

#include <vector>
#include <queue>
#include <inttypes.h>

#include "nv_phy_driver_proxy.hpp"
#include "nv_simulate_phy_driver.hpp"
#include "nvlog.hpp"

#define TAG (NVLOG_TAG_BASE_L2_ADAPTER + 3) // "L2A.PROXY"

namespace nv
{
static std::unique_ptr<PHYDriverProxy> sProxy;
static std::once_flag                  sProxyFlag;

static std::unique_ptr<SimulatePhyDriver> sSimulateDrv;

PHYDriverProxy& PHYDriverProxy::getInstance()
{
    return *sProxy;
}

PHYDriverProxy* PHYDriverProxy::getInstancePtr()
{
    return sProxy.get();
}

void PHYDriverProxy::make(phydriver_handle driver, mplane_config& config)
{
    // Create instance with cuphydriver
    if(driver == nullptr)
    {
        NVLOGF_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "NULL phydriver_handle pointer");
    }
    std::call_once(sProxyFlag, [=]() {
        sProxy.reset(new PHYDriverProxy(driver, config));
    });
}

void PHYDriverProxy::make(thread_config* cfg, int max_cell_num)
{
    // Create simulate driver for l2adapter_standalone test
    std::call_once(sProxyFlag, [=]() {
        sProxy.reset(new PHYDriverProxy(max_cell_num));
        sSimulateDrv.reset(new SimulatePhyDriver(cfg));
    });
    NVLOGC_FMT(TAG, "{}: Created PHYDriverProxy instance for L2SA test", __func__);
}

int PHYDriverProxy::l1_cell_create(cell_phy_info& cell)
{
    return ::l1_cell_create(driver_, cell);
}

int PHYDriverProxy::l1_cell_create(uint16_t cell_id)
{
    return 0;
}

int PHYDriverProxy::l1_cell_destroy(uint16_t cell_id)
{
    if(driver_ != nullptr)
    {
        return ::l1_cell_destroy(driver_, cell_id);
    }
    else
    {
        // l2_adapter standalone mode
        return 0;
    }
}

int PHYDriverProxy::l1_cell_start(uint16_t cell_id)
{
    if(driver_ != nullptr)
    {
        NVLOGI_FMT(TAG, "PHYDriverProxy:: Starting cell ={}", cell_id);
        return ::l1_cell_start(driver_, cell_id);
    }
    else
    {
        // l2_adapter standalone mode
        return 0;
    }
}

int PHYDriverProxy::l1_cell_stop(uint16_t cell_id)
{
    if(driver_ != nullptr)
    {
        NVLOGI_FMT(TAG, "PHYDriverProxy:: Stopping cell ={}", cell_id);
        return ::l1_cell_stop(driver_, cell_id);
    }
    else
    {
        // l2_adapter standalone mode
        return 0;
    }
}

int PHYDriverProxy::l1_enqueue_phy_work(slot_command& command)
{
    if(driver_ != nullptr)
    {
        return ::l1_enqueue_phy_work(driver_, &command);
    }
    else
    {
        return sSimulateDrv->enqueue_phy_work(command);
    }
}

int PHYDriverProxy::l1_set_output_callback(callbacks& cb)
{
    if(driver_ != nullptr)
    {
        return ::l1_set_output_callback(driver_, cb);
    }
    else
    {
        return sSimulateDrv->set_output_callback(cb);
    }
}

int PHYDriverProxy::l1_cell_update_cell_config(uint16_t mplane_id, uint16_t grid_sz, bool dl)
{
    if(driver_ != nullptr)
    {
        return ::l1_cell_update_cell_config(driver_, mplane_id, grid_sz, dl);
    }
    else
    {
        // l2_adapter standalone mode
        return 0;
    }
}

int PHYDriverProxy::l1_cell_update_cell_config(uint16_t mplane_id, std::unordered_map<int, std::vector<uint16_t>>& eaxcids_ch_map)
{
    if(driver_ != nullptr)
    {
        return ::l1_cell_update_cell_config(driver_, mplane_id, eaxcids_ch_map);
    }
    else
    {
        // l2_adapter standalone mode
        return 0;
    }
}

int PHYDriverProxy::l1_cell_update_cell_config(uint16_t mplane_id, std::string dst_mac, uint16_t vlan_tci)
{
    if(driver_ != nullptr)
    {
        return ::l1_cell_update_cell_config(driver_, mplane_id, dst_mac, vlan_tci);
    }
    else
    {
        // l2_adapter standalone mode
        return 0;
    }
}

int PHYDriverProxy::l1_cell_update_cell_config(uint16_t mplane_id, std::unordered_map<std::string, double>& attrs, std::unordered_map<std::string, int>& res)
{
    if(driver_ != nullptr)
    {
        NVLOGC_FMT(TAG, "Update cell: mplane_id={} ", mplane_id);

        auto & cfg = getMPlaneConfigByMplaneId(mplane_id);
        if(attrs.find(CELL_PARAM_UL_GAIN_CALIBRATION) != attrs.end())
        {
            cfg.ul_gain_calibration = attrs[CELL_PARAM_UL_GAIN_CALIBRATION];
            NVLOGC_FMT(TAG, "{} updated to {} ", CELL_PARAM_UL_GAIN_CALIBRATION, cfg.ul_gain_calibration);
            attrs.erase(CELL_PARAM_UL_GAIN_CALIBRATION);
        }

        if(attrs.empty())
        {
            return 0;
        }

        int ret = ::l1_cell_update_cell_config(driver_, mplane_id, attrs, res);
        if(ret != 0)
        {
            return ret;
        }

        for(auto& p : attrs)
        {
            //NVLOGC_FMT(TAG, "{}: {} ", p.first.c_str(), p.second);
            if(strcmp(p.first.c_str(), CELL_PARAM_RU_TYPE) == 0)
            {
                cfg.ru = static_cast<ru_type>(p.second);
            }
            else if(strcmp(p.first.c_str(), CELL_PARAM_COMPRESSION_BITS) == 0)
            {
                cfg.dl_comp_meth = static_cast<aerial_fh::UserDataCompressionMethod>(p.second);
            }
            else if(strcmp(p.first.c_str(), CELL_PARAM_DECOMPRESSION_BITS) == 0)
            {
                cfg.ul_comp_meth = static_cast<aerial_fh::UserDataCompressionMethod>(p.second);
            }
            else if(strcmp(p.first.c_str(), CELL_PARAM_EXPONENT_DL) == 0)
            {
                cfg.exponent_dl = p.second;
            }
            else if(strcmp(p.first.c_str(), CELL_PARAM_EXPONENT_UL) == 0)
            {
                cfg.exponent_ul = p.second;
            }
            else if(strcmp(p.first.c_str(), CELL_PARAM_MAX_AMP_UL) == 0)
            {
                cfg.max_amp_ul = p.second;
            }
            else if(strcmp(p.first.c_str(), CELL_PARAM_PUSCH_PRB_STRIDE) == 0)
            {
                cfg.pusch_prb_stride = p.second;
            }
            else if(strcmp(p.first.c_str(), CELL_PARAM_PRACH_PRB_STRIDE) == 0)
            {
                cfg.prach_prb_stride = p.second;
            }
            else if(strcmp(p.first.c_str(), CELL_PARAM_SECTION_3_TIME_OFFSET) == 0)
            {
                cfg.section_3_time_offset = p.second;
            }
            else if(strcmp(p.first.c_str(), CELL_PARAM_FH_DISTANCE_RANGE) == 0)
            {
                cfg.fh_len_range = p.second;
            }
            else if(strcmp(p.first.c_str(), CELL_PARAM_LOWER_GUARD_BW) == 0)
            {
                cfg.lower_guard_bw = p.second;
                NVLOGC_FMT(TAG, "{} updated to {:.0f} ", CELL_PARAM_LOWER_GUARD_BW, p.second);
            }
        }
        return ret;
    }
    else
    {
        // l2_adapter standalone mode
        return 0;
    }
}

int PHYDriverProxy::l1_cell_update_attenuation(uint16_t mplane_id, float attenuation_dB)
{
    if(driver_ != nullptr)
    {
        return ::l1_cell_update_attenuation(driver_, mplane_id, attenuation_dB);
    }
    else
    {
        // l2_adapter standalone mode
        return 0;
    }
}

int PHYDriverProxy::l1_update_gps_alpha_beta(uint64_t alpha,int64_t beta)
{
    if(driver_ != nullptr)
    {
        return ::l1_update_gps_alpha_beta(driver_,alpha,beta);
    }
    else
    {
        // l2_adapter standalone mode
        return 0;
    }
}

int PHYDriverProxy::l1_cell_update_cell_config(struct cell_phy_info& cell_pinfo, ::CellUpdateCallBackFn& fn)
{
    if(driver_ != nullptr)
    {
        return ::l1_cell_update_cell_config(driver_, cell_pinfo, fn);
    }
    else
    {
        // l2_adapter standalone mode
        return 0;
    }
}

int PHYDriverProxy::l1_lock_update_cell_config_mutex()
{
    if(driver_ != nullptr)
    {
        return ::l1_lock_update_cell_config_mutex(driver_);
    }
    else
    {
        // l2_adapter standalone mode
        return 0;
    }
}

int PHYDriverProxy::l1_unlock_update_cell_config_mutex()
{
    if(driver_ != nullptr)
    {
        return ::l1_unlock_update_cell_config_mutex(driver_);
    }
    else
    {
        // l2_adapter standalone mode
        return 0;
    }
}

uint8_t PHYDriverProxy::l1_get_prach_start_ro_index(uint16_t phy_cell_id)
{
    if(driver_ != nullptr)
    {
        return ::l1_get_prach_start_ro_index(driver_, phy_cell_id);
    }
    else
    {
        // l2_adapter standalone mode
        return 0;
    }
}

bool PHYDriverProxy::allocSrsChesBuffPool(uint32_t requestedBy, uint16_t phy_cell_id, uint32_t poolSize)
{
    if(driver_ != nullptr)
    {
        return ::l1_allocSrsChesBuffPool(driver_, requestedBy, phy_cell_id, poolSize);
    }
    else
    {
        // l2_adapter standalone mode
        return false;
    }
}

bool PHYDriverProxy::deAllocSrsChesBuffPool(uint16_t phy_cell_id)
{
    if(driver_ != nullptr)
    {
        return ::l1_deAllocSrsChesBuffPool(driver_, phy_cell_id);
    }
    else
    {
        // l2_adapter standalone mode
        return false;
    }
}

// Get cell_mplane_info by cell_id
cell_mplane_info& PHYDriverProxy::getMPlaneConfig(int32_t cell_id)
{
    if(cell_id < 0 || cell_id >= config_.size())
    {
        throw std::runtime_error(
            std::string("Invalid index passed: index=") + std::to_string(cell_id) + std::string("config_.size=") + std::to_string(config_.size()));
    }
    return config_[cell_id];
}

// Get cell_mplane_info by mplane_id
cell_mplane_info& PHYDriverProxy::getMPlaneConfigByMplaneId(uint32_t mplane_id)
{
    for (cell_mplane_info& cfg : config_)
    {
        if (cfg.mplane_id == mplane_id)
        {
            return cfg;
        }
    }

    throw std::runtime_error(
        std::string("MPlaneConfig for mplane_id=") + std::to_string(mplane_id) + std::string("not found. config_size=") + std::to_string(config_.size()));
}

// Get cell_mplane_info list
std::vector<cell_mplane_info>& PHYDriverProxy::getMPlaneConfigList()
{
    return config_;
}

void PHYDriverProxy:: l1_copy_TB_to_gpu_buf(uint16_t phy_cell_id, uint8_t * tb_buff, uint8_t ** gpu_buff_ref,uint32_t tb_len, uint8_t slot_index, uint16_t sfn)
{
    /*Copy Tb from CPU buff to GPU*/
    //::l1_copy_TB_to_gpu_buf(driver_, phy_cell_id, tb_buff, gpu_buff_ref, tb_len, slot_index);
    ::l1_copy_TB_to_gpu_buf_thread_offload(driver_, phy_cell_id, tb_buff, gpu_buff_ref, tb_len, slot_index, sfn);
    return;
}

int PHYDriverProxy::l1_cv_mem_bank_update(uint32_t cell_id,uint16_t rnti, uint16_t buffer_idx, uint16_t reportType, uint16_t startPrbGrp,uint32_t srsPrbGrpSize,uint16_t numPrgs,
    uint8_t nGnbAnt,uint8_t nUeAnt,uint32_t offset, uint8_t* srsChEsts, uint16_t startValidPrg, uint16_t nValidPrg)
{
    if(driver_ != nullptr)
        return ::l1_cv_mem_bank_update(driver_,cell_id,rnti,buffer_idx,reportType,startPrbGrp,srsPrbGrpSize,numPrgs,nGnbAnt,nUeAnt,offset,srsChEsts, startValidPrg, nValidPrg);
    else
        return -1;
}

int PHYDriverProxy::l1_cv_mem_bank_retrieve_buffer(uint32_t cell_id, uint16_t rnti, uint16_t buffer_idx, uint16_t reportType, uint8_t *pSrsPrgSize, uint16_t* pSrsStartPrg, uint16_t* pSrsStartValidPrg, uint16_t* pSrsNValidPrg, cuphyTensorDescriptor_t* descr, uint8_t** ptr)
{
    if(driver_ != nullptr)
        return ::l1_cv_mem_bank_retrieve_buffer(driver_,cell_id,rnti,buffer_idx,reportType,pSrsPrgSize,pSrsStartPrg,pSrsStartValidPrg,pSrsNValidPrg,descr,ptr);
    else
        return -1;
}

int PHYDriverProxy::l1_cv_mem_bank_update_buffer_state(uint32_t cell_id, uint16_t buffer_idx, slot_command_api::srsChestBuffState srs_chest_buff_state)
{
    if(driver_ != nullptr)
        return ::l1_cv_mem_bank_update_buffer_state(driver_, cell_id, buffer_idx, srs_chest_buff_state);
    else
        return -1;
}

int PHYDriverProxy::l1_cv_mem_bank_get_buffer_state(uint32_t cell_id, uint16_t buffer_idx, slot_command_api::srsChestBuffState *srs_chest_buff_state)
{
    if(driver_ != nullptr)
        return ::l1_cv_mem_bank_get_buffer_state(driver_, cell_id, buffer_idx, srs_chest_buff_state);
    else
        return -1;
}

int PHYDriverProxy::l1_bfw_coeff_retrieve_buffer(uint32_t cell_id, bfw_buffer_info* ptr)
{
    if(driver_ != nullptr)
        return ::l1_bfw_coeff_retrieve_buffer(driver_, cell_id, ptr);
    else
        return -1;
}

int PHYDriverProxy::l1_mMIMO_enable_info(uint8_t *pMuMIMO_enable)
{
    if(driver_ != nullptr)
        return ::l1_mMIMO_enable_info(driver_,pMuMIMO_enable);
    else
        return -1;
}
int PHYDriverProxy::l1_enable_srs_info(uint8_t *pEnable_srs)
{
    if(driver_ != nullptr)
        return ::l1_enable_srs_info(driver_, pEnable_srs);
    else
        return -1;
}


uint32_t PHYDriverProxy::l1_get_cell_group_num() {
    uint8_t cell_group_num = 0;
    if(driver_ != nullptr)
    {
        if (::l1_get_cell_group_num(driver_, &cell_group_num) < 0)
        {
            NVLOGF_FMT(TAG, AERIAL_CUPHYDRV_API_EVENT, "Failed to get cell group number");
        }
    }
    else
    {
        cell_group_num = config_.size();
    }
    return cell_group_num;
}

int PHYDriverProxy::l1_get_ch_segment_proc_enable_info(uint8_t* ch_seg_proc_enable) {
    if(driver_ != nullptr)
        return ::l1_get_ch_segment_proc_enable_info(driver_, ch_seg_proc_enable);
    else
        return -1;
}

bool PHYDriverProxy::l1_incr_recovery_slots()
{
    if(driver_ != nullptr)
        return ::l1_incr_recovery_slots(driver_);
    else
        return false;
}

bool PHYDriverProxy::l1_incr_all_obj_free_slots()
{
    if(driver_ != nullptr)
        return ::l1_incr_all_obj_free_slots(driver_);
    else
        return false;
}

void PHYDriverProxy::l1_reset_all_obj_free_slots()
{
    if(driver_ != nullptr)
        return ::l1_reset_all_obj_free_slots(driver_);
}

void PHYDriverProxy::l1_reset_recovery_slots()
{
    if(driver_ != nullptr)
        return ::l1_reset_recovery_slots(driver_);
}

bool PHYDriverProxy::l1_get_aggr_obj_free_status()
{
    if(driver_ != nullptr)
        return ::l1_check_cuphy_objects_status(driver_);
    else
        return false;
}

int PHYDriverProxy::l1_storeDBTPdu(uint16_t cell_id, void* data_buf)
{
    if(driver_ != nullptr)
    {
        return ::l1_storeDBTPduInFH(driver_, cell_id, data_buf);
    }
    else
    {
        // l2_adapter standalone mode
        return -1;
    }
}

[[nodiscard]] int PHYDriverProxy::l1_resetDBTStorage(uint16_t cell_id)
{
    if(driver_ != nullptr)
    {
        return ::l1_resetDBTStorageInFH(driver_, cell_id);
    }
    else
    {
        // l2_adapter standalone mode
        return -1;
    }
}

int PHYDriverProxy::l1_getBeamWeightsSentFlag(uint16_t cell_id, uint16_t beamIdx)
{
    if(driver_ != nullptr)
    {
        return ::l1_getBeamWeightsSentFlagInFH(driver_, cell_id, beamIdx);
    }
    else
    {
        // l2_adapter standalone mode
        return -1;
    }
}

int PHYDriverProxy::l1_setBeamWeightsSentFlag(uint16_t cell_id, uint16_t beamIdx)
{
    if(driver_ != nullptr)
    {
        return ::l1_setBeamWeightsSentFlagInFH(driver_, cell_id, beamIdx);
    }
    else
    {
        // l2_adapter standalone mode
        return -1;
    }
}

int16_t PHYDriverProxy::l1_getDynamicBeamIdOffset(uint16_t cell_id)
{
    if(driver_ != nullptr)
    {
        return ::l1_getDynamicBeamIdOffset(driver_, cell_id);
    }
    else
    {
        // l2_adapter standalone mode
        return -1;
    }
}

int PHYDriverProxy::l1_staticBFWConfigured(uint16_t cell_id)
{
    if(driver_ != nullptr)
    {
        return ::l1_staticBFWConfiguredInFH(driver_, cell_id);
    }
    else
    {
        // l2_adapter standalone mode
        return -1;
    }
}

int PHYDriverProxy::l1_get_send_static_bfw_wt_all_cplane()
{
    if(driver_ != nullptr)
        return ::l1_get_send_static_bfw_wt_all_cplane(driver_);
    else
        return -1;
}

void PHYDriverProxy::l1_resetBatchedMemcpyBatches()
{
    if(driver_ != nullptr)
        ::l1_resetBatchedMemcpyBatches(driver_);
    // Assuming no explicit handling of else case is needed
}

uint8_t PHYDriverProxy::l1_get_enable_weighted_average_cfo()
{
    if(driver_ != nullptr)
        return ::l1_get_enable_weighted_average_cfo(driver_);
    else
        return 0;
}

bool PHYDriverProxy::l1_get_split_ul_cuda_streams()
{
    if(driver_ != nullptr)
        return ::l1_get_split_ul_cuda_streams(driver_);
    else
        return false;
}
bool PHYDriverProxy::l1_get_dl_tx_notification() const noexcept
{
    if (driver_ != nullptr)
        return ::l1_get_dl_tx_notification(driver_);
    else
        return false;
}

} // namespace nv

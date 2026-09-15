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

#include <unistd.h>

#include "hdf5hpp.hpp"
#include "hdf5.h"
#include "cuphy_hdf5.hpp"
#include "nvlog.hpp"

#include "cumac_pattern.hpp"
#include "cumac_defines.hpp"
#include "common_utils.hpp"

#include <chrono>
#include <vector>
#include <filesystem>

#define TAG (NVLOG_TAG_BASE_TEST_MAC + 23) // "CUMAC.PATTERN"

#define CHECK_VALUE_EQUAL_ERR(v1, v2)                                                                                              \
    do                                                                                                                             \
    {                                                                                                                              \
        if((v1) != (v2))                                                                                                           \
        {                                                                                                                          \
            NVLOGE_FMT(TAG, AERIAL_CUMAC_CP_EVENT, "{} line {}: values doesn't equal: v1={} > v2={}", __func__, __LINE__, v1, v2); \
        }                                                                                                                          \
    } while(0);

// Only used for calculate phyCellId from testMac_config_params_XXX.h5
#define CONFIG_PHY_CELL_ID_BASE 40 // phyCellId = CONFIG_PHY_CELL_ID_BASE + cell_number

#define CONFIG_CUMAC_TV_PATH "testVectors/cumac/"

//! Use with snprintf and (SFN, per-frame slot) to generate per-slot TV filename.
//! Multi-frame test patterns (schedule_slot_period spanning more than one
//! radio frame) save TVs as muUePairTV_sfn<S>_slot<N>_cell<C>.h5 where N is
//! the slot index within frame S, both derived from the period-relative
//! slot_idx via slots_per_frame.
#define CONFIG_MU_UE_PAIR_TV_FORMAT "muUePairTV_sfn%d_slot%d"

#define H5_N_PDU "nPdu"
#define H5_PDU "PDU"
#define H5_IND "IND"
#define H5_BFP "BFPforCuphy"

static const char* NUMPDU = "nPdu";
static const char* PDU    = "PDU";
static const char* CELL_CONFIG = "Cell_Config";

using namespace std;
using namespace std::chrono;

// Current parsing cell_id, slot_id, channel and TV file name for debug log
static int curr_cell;
static int curr_slot;
static int curr_task;
static std::string curr_tv;

template <typename T>
static int yaml_try_parse_list(yaml::node& parent_node, const char* name, std::vector<T>& values)
{
    yaml::node list_nodes = parent_node[name];
    if(list_nodes.type() != YAML_SEQUENCE_NODE)
    {
        NVLOGE_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "{}: failed to parse {}: error type {}\n", __func__, name, list_nodes.type());
        return -1;
    }

    size_t num = list_nodes.length();
    values.resize(num);

    for(size_t i = 0; i < num; i++)
    {
        yaml::node node = list_nodes[i];
        values[i]         = node.as<T>();
    }
    return 0;
}

static int h5dset_try_read(hdf5hpp::hdf5_file& file, const char* name, void* buf, size_t size)
{
    if(!file.is_valid_dataset(name))
    {
        NVLOGW_FMT(TAG, "TV cell {} slot {} {} {} dataset {} not exist",
                curr_cell, curr_slot, get_task_name(curr_task), curr_tv.c_str(), name);
        return -1;
    }

    try
    {
        hdf5hpp::hdf5_dataset h5dset = file.open_dataset(name);
        if(h5dset.get_buffer_size_bytes() != size)
        {
            NVLOGW_FMT(TAG, "TV cell {} slot {} {} {} dataset {} size doesn't match: dataset_size={} buf_size={}",
                    curr_cell, curr_slot, get_task_name(curr_task), curr_tv.c_str(),
                    name, h5dset.get_buffer_size_bytes(), size);
            return -1;
        }
        else
        {
            h5dset.read(buf);
            return 0;
        }
    }
    catch(std::exception& e)
    {
        NVLOGE_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "TV cell {} slot {} {} {} dataset {} exception: {}",
                curr_cell, curr_slot, get_task_name(curr_task), curr_tv.c_str(), name, e.what());
    }
    return -1;
}

template <typename Type>
static int h5dset_try_read_array(hdf5hpp::hdf5_file& file, const char* name, Type** buf_ptr, uint32_t elem_num)
{
    *buf_ptr = new Type[elem_num];
    if (h5dset_try_read(file, name, *buf_ptr, sizeof(Type) * elem_num) < 0) {
        delete *buf_ptr;
        return -1;
    }
 
    return 0;
}

static int h5dset_try_read_complex(hdf5hpp::hdf5_file& file, const char* name_real, const char* name_imag, cuComplex** complex_ptr, uint32_t elem_num)
{
    float* tmp_real = nullptr;
    float* tmp_imag = nullptr;

    if(h5dset_try_read_array(file, name_real, &tmp_real, elem_num) < 0)
    {
        return -1;
    }

    if(h5dset_try_read_array(file, name_imag, &tmp_imag, elem_num) < 0)
    {
        delete tmp_real;
        return -1;
    }

    *complex_ptr = new cuComplex[elem_num];
    for(int i = 0; i < elem_num; i++)
    {
        cuComplex* val = *complex_ptr + i;
        val->x         = *(tmp_real + i);
        val->y         = *(tmp_imag + i);
    }

    delete tmp_real;
    delete tmp_imag;

    return 0;
}

static int h5dset_try_read_u32_to_bits(hdf5hpp::hdf5_file& file, const char* name, std::vector<uint8_t>& dest, int num_bits)
{
    uint32_t* src = new uint32_t[num_bits];
    int ret = h5dset_try_read(file, name, src, num_bits * sizeof(uint32_t));
    if (ret == 0)
    {
        // Initiate bytes to 0
        int nbytes = (num_bits + 7) / 8;
        dest.resize(nbytes);
        for (int i = 0; i < nbytes; i ++)
        {
            dest[i] = 0;
        }
        // Convert bits to bytes
        for (int j = 0; j < num_bits; j ++)
        {
            dest[j / 8] |= src[j] == 0 ? 0 : 1 << j % 8;
        }
    }
    delete[] src;
    return ret;
}

template <typename TypeSrc, typename TypeDst>
static int h5dset_try_read_convert(hdf5hpp::hdf5_file& file, const char* name, TypeDst* dst, uint32_t num)
{
    if (num == 0)
    {
        NVLOGW_FMT(TAG, "TV cell {} slot {} {} {} dataset {} reading with num=0",
                curr_cell, curr_slot, get_task_name(curr_task), curr_tv.c_str(), name);
        return -1;
    }

    TypeSrc* src = new uint32_t[num];
    int ret = h5dset_try_read(file, name, src, num * sizeof(TypeSrc));
    if (ret == 0)
    {
        for (int i = 0; i < num; i ++)
        {
            dst[i] = src[i];
        }
    }
    delete[] src;
    return ret;
}

template <typename T>
static T h5dset_try_parse(const hdf5hpp::hdf5_dataset_elem& dset_elem, const char* name, T default_value, bool miss_warning = true)
{
    T value;
    try
    {
        value = dset_elem[name].as<T>();
    }
    catch(std::exception& e)
    {
        value = default_value;
        if (miss_warning)
        {
            NVLOGW_FMT(TAG, "TV cell {} slot {} {} {} key {} not exist",
                    curr_cell, curr_slot, get_task_name(curr_task), curr_tv.c_str(), name);
        }
    }
    return value;
}

template <typename T>
static T h5dset_try_parse(hdf5hpp::hdf5_dataset& h5dset, const char* name, T default_value, bool miss_warning = true)
{
    return h5dset_try_parse(h5dset[0], name, default_value, miss_warning);
}

template <typename T>
static T h5file_try_parse(const char* file_name, const char* dset_name, const char* var_name, T default_value, bool miss_warning = true, int dset_id = 0)
{
    char h5path[MAX_PATH_LEN];
    get_cubb_full_path(h5path, CONFIG_CUMAC_TV_PATH, file_name);
    if(access(h5path, F_OK) != 0)
    {
        NVLOGF_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "H5 file {} not exist", h5path);
    }

    hdf5hpp::hdf5_file hdf5file;
    try
    {
        hdf5file = hdf5hpp::hdf5_file::open(h5path);
    }
    catch(std::exception& e)
    {
        NVLOGE_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "Exception: {}: hdf5_file::open({}): {}", __FUNCTION__, h5path, e.what());
        return default_value;
    }

    if(hdf5file.is_valid_dataset(dset_name))
    {
        hdf5hpp::hdf5_dataset dset = hdf5file.open_dataset(dset_name);
        return h5dset_try_parse(dset[dset_id], var_name, default_value, miss_warning);
    }
    else
    {
        return default_value;
    }
}

cumac_pattern::cumac_pattern(test_mac_configs* configs)
{
    yaml_configs               = configs;
    cumac_type                 = 0;
    channel_mask               = 0;
    cell_num                   = 0;
    sched_slot_num             = 0;
    init_slot_num              = 0;
    slots_per_frame            = SLOTS_PER_FRAME;
    config_static_harq_proc_id = 0;
    negative_test              = 0;
    using_init_patterns        = false;
    cumac_configs              = configs->cumac_configs;
}

void cumac_pattern::free_test_vector(cumac_test_vector_t* tv)
{
    if (tv == nullptr)
    {
        return;
    }

    // Free the PFM sorting input buffer
    if (tv->req.pfmCellInfo != nullptr)
    {
        delete tv->req.pfmCellInfo;
        tv->req.pfmCellInfo = nullptr;
    }
    // Free the PFM sorting output buffer
    if (tv->resp.pfmSortSol != nullptr)
    {
        delete tv->resp.pfmSortSol;
        tv->resp.pfmSortSol = nullptr;
    }

    // Free the MU UE grouping request info buffer (allocated with new uint8_t[nbytes] in load_mu_ue_grp_tv)
    if (tv->req.muUeGrpInfo != nullptr)
    {
        delete[] reinterpret_cast<uint8_t*>(tv->req.muUeGrpInfo);
        tv->req.muUeGrpInfo = nullptr;
    }
    if (tv->resp.muUeGrpSol != nullptr)
    {
        delete tv->resp.muUeGrpSol;
        tv->resp.muUeGrpSol = nullptr;
    }

    // Free the test vector buffer
    delete tv;
}

cumac_pattern::~cumac_pattern()
{
    for(auto slot : init_slots_pattern)
    {
        for(auto cell : slot)
        {
            for(auto group : cell)
            {
                for(auto req : group)
                {
                    if (req->tv_data != nullptr && yaml_configs->tv_data_map_enable == 0)
                    {
                        free_test_vector(req->tv_data);
                        req->tv_data = nullptr;
                    }
                    delete req;
                }
            }
        }
    }
    for(auto slot : sched_slots_pattern)
    {
        for(auto cell : slot)
        {
            for(auto group : cell)
            {
                for(auto req : group)
                {
                    if (req->tv_data != nullptr && yaml_configs->tv_data_map_enable == 0)
                    {
                        free_test_vector(req->tv_data);
                        req->tv_data = nullptr;
                    }
                    delete req;
                }
            }
        }
    }
    for(auto cell_config : cumac_cell_configs_v)
    {
        free(cell_config);
    }
}

namespace {

template <typename T>
bool h5_read_attr_scalar(hid_t loc_id, const char* name, hid_t mem_type, T* out)
{
    if (H5Aexists(loc_id, name) <= 0)
        return false;
    hid_t a = H5Aopen(loc_id, name, H5P_DEFAULT);
    if (a < 0)
        return false;
    herr_t st = H5Aread(a, mem_type, out);
    H5Aclose(a);
    return st >= 0;
}

} // namespace

int cumac_pattern::probe_schedule_slot_period()
{
    namespace fs = std::filesystem;

    char probe_path[MAX_PATH_LEN];
    get_full_path_file(probe_path, CONFIG_CUMAC_TV_PATH, "probe", CONFIG_CUBB_ROOT_DIR_RELATIVE_NUM);
    const fs::path tv_dir = fs::path(probe_path).parent_path();

    std::error_code ec;
    if (!fs::is_directory(tv_dir, ec))
    {
        NVLOGI_FMT(TAG, "probe_schedule_slot_period: TV directory {} not present, using legacy slot-list sizing",
                   tv_dir.string());
        return 0;
    }

    const std::string tv_prefix = "muUePairTV_sfn";
    const std::string cell0_sfx = "_cell0.h5";
    for (const auto& entry : fs::directory_iterator(tv_dir, ec))
    {
        if (!entry.is_regular_file()) continue;
        const std::string fname = entry.path().filename().string();
        if (fname.rfind(tv_prefix, 0) != 0) continue;
        if (fname.size() <= cell0_sfx.size()) continue;
        if (fname.compare(fname.size() - cell0_sfx.size(), cell0_sfx.size(), cell0_sfx) != 0) continue;

        try
        {
            hdf5hpp::hdf5_file probe = hdf5hpp::hdf5_file::open(entry.path().string().c_str());
            int32_t period_attr = 0;
            if (h5_read_attr_scalar(probe.id(), "schedule_slot_period", H5T_NATIVE_INT32, &period_attr) && period_attr > 0)
            {
                NVLOGC_FMT(TAG, "probe_schedule_slot_period: read period={} from {}", period_attr, fname);
                return period_attr;
            }
        }
        catch (const std::exception& e)
        {
            NVLOGW_FMT(TAG, "probe_schedule_slot_period: failed to open {}: {}", fname, e.what());
        }
    }

    NVLOGI_FMT(TAG, "probe_schedule_slot_period: no muUePairTV with schedule_slot_period attr in {}, falling back to legacy sizing",
               tv_dir.string());
    return 0;
}

int cumac_pattern::probe_first_ue_group_slot()
{
    namespace fs = std::filesystem;

    char probe_path[MAX_PATH_LEN];
    get_full_path_file(probe_path, CONFIG_CUMAC_TV_PATH, "probe", CONFIG_CUBB_ROOT_DIR_RELATIVE_NUM);
    const fs::path tv_dir = fs::path(probe_path).parent_path();

    const std::string tv_prefix = "muUePairTV_sfn";
    const std::string cell0_sfx = "_cell0.h5";
    std::error_code ec;
    for (const auto& entry : fs::directory_iterator(tv_dir, ec))
    {
        if (!entry.is_regular_file()) continue;
        const std::string fname = entry.path().filename().string();
        if (fname.rfind(tv_prefix, 0) != 0) continue;
        if (fname.size() <= cell0_sfx.size()) continue;
        if (fname.compare(fname.size() - cell0_sfx.size(), cell0_sfx.size(), cell0_sfx) != 0) continue;

        int fsfn = 0, fslot = 0;
        if (sscanf(fname.c_str() + tv_prefix.size(), "%d_slot%d_cell0", &fsfn, &fslot) != 2) continue;

        try
        {
            hdf5hpp::hdf5_file probe = hdf5hpp::hdf5_file::open(entry.path().string().c_str());
            uint8_t first_slot_attr = 0;
            if (h5_read_attr_scalar(probe.id(), "first_slot", H5T_NATIVE_UINT8, &first_slot_attr)
                && first_slot_attr != 0)
            {
                const int tv_slot_id = (fsfn * SLOTS_PER_FRAME + fslot + cumac_configs->srs_slot_lag) % sched_slot_num;
                NVLOGC_FMT(TAG, "{}: srs_slot_lag={} first_slot at SFN {}.{} -> tv_slot_idx={}",
                           __func__, cumac_configs->srs_slot_lag, fsfn, fslot, tv_slot_id);
                return tv_slot_id;
            }
        }
        catch (const std::exception& e)
        {
            NVLOGW_FMT(TAG, "{}: failed to open {}: {}", __func__, fname, e.what());
        }
    }

    NVLOGI_FMT(TAG, "{}: no muUePairTV with first_slot attr in {}", __func__, tv_dir.string());
    return -1;
}

int cumac_pattern::load_mu_ue_grp_tv(cumac_req_t* req)
{
    if (req == nullptr || req->tv_data == nullptr)
        return -1;

    cumac_tti_req_tv_t& tv = req->tv_data->req;
    cumac_tti_resp_tv_t& tv_resp = req->tv_data->resp;

    tv.muUeGrpInfo = nullptr;
    tv.muUeGrpReqDataLen = 0;
    tv.muUeGrpIsMemSharing = false;
    tv_resp.muUeGrpSol = nullptr;

    const int cell_id = req->cell_idx;
    const int slot_id = req->slot_idx;

    // Derive SFN/SLOT from slot_id, applying srs_slot_lag offset with wrap-around.
    // Reduce srs_slot_lag mod sched_slot_num first so the intermediate value is always positive.
    const int tv_slot_id = (sched_slot_num > 0)
        ? (slot_id + sched_slot_num - cumac_configs->srs_slot_lag % sched_slot_num) % sched_slot_num
        : slot_id;

    const int sfn  = slot_id / SLOTS_PER_FRAME;
    const int slot = slot_id % SLOTS_PER_FRAME;

    const int tv_sfn  = tv_slot_id / SLOTS_PER_FRAME;
    const int tv_slot = tv_slot_id % SLOTS_PER_FRAME;

    char tv_base[128];
    snprintf(tv_base, sizeof(tv_base), CONFIG_MU_UE_PAIR_TV_FORMAT, tv_sfn, tv_slot);

    char cell_fname[MAX_PATH_LEN];
    snprintf(cell_fname, sizeof(cell_fname), "%s_cell%d.h5", tv_base, cell_id);
    char full_cell[MAX_PATH_LEN];
    get_cubb_full_path(full_cell, CONFIG_CUMAC_TV_PATH, cell_fname);
    if (access(full_cell, F_OK) != 0)
    {
        // Missing per-slot TV is expected when only some slots are scheduled
        // (see schedule_slot_period). Caller clears MU_UE_GRP bit in the
        // per-slot taskBitMask so cuMAC-CP knows to skip that slot.
        req->tv_data->req.taskBitMask &= ~static_cast<uint32_t>(0x1 << CUMAC_TASK_MU_UE_GRP);
        NVLOGI_FMT(TAG, "SFN {}.{} UE_PAIR_TV cell_id={} slot_idx={} tv_slot_idx={} srs_slot_lag={} {} not found",
            sfn, slot, cell_id, slot_id, tv_slot_id, cumac_configs->srs_slot_lag, tv_base);
        return 0;
    }

    try
    {
        hdf5hpp::hdf5_file cell_file = hdf5hpp::hdf5_file::open(full_cell);
        hid_t fid = cell_file.id();

        uint8_t mem_sharing_u8 = 0;
        h5_read_attr_scalar(fid, "is_mem_sharing", H5T_NATIVE_UINT8, &mem_sharing_u8);
        tv.muUeGrpIsMemSharing = (mem_sharing_u8 != 0);
        tv.muUeGrpReqDataLen = cumac_muUeGrp_req_info_size(tv.muUeGrpIsMemSharing);

        hdf5hpp::hdf5_dataset ds_in = cell_file.open_dataset("task_in_buf");
        const size_t nbytes = ds_in.get_buffer_size_bytes();
        if (nbytes != tv.muUeGrpReqDataLen)
        {
            NVLOGE_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "MU UE pair TV: cell_id={} task_in_buf size {} != expected {}", cell_id, nbytes, cumac_muUeGrp_req_info_size(tv.muUeGrpIsMemSharing));
            return -1;
        }

        // Allocate memory for the MU UE grouping request info structure and read the data from the H5 file.
        tv.muUeGrpInfo = reinterpret_cast<cumac_muUeGrp_req_info_t*>(new uint8_t[nbytes]);
        ds_in.read(tv.muUeGrpInfo);

         // Set the 3 pointers in to nullptr because it doesn't make sence in commnunication between 2 processes.
        cumac_muUeGrp_req_info_t* req_info = reinterpret_cast<cumac_muUeGrp_req_info_t*>(tv.muUeGrpInfo);
        req_info->srsInfo = nullptr;
        req_info->srsInfoMsh = nullptr;
        req_info->ueInfo = nullptr;

        // Load the MU UE grouping response TV file.
        char sol_fname[MAX_PATH_LEN];
        snprintf(sol_fname, sizeof(sol_fname), "%s_solution.h5", tv_base);
        char full_sol[MAX_PATH_LEN];
        get_cubb_full_path(full_sol, CONFIG_CUMAC_TV_PATH, sol_fname);
        if (access(full_sol, F_OK) != 0)
        {
            NVLOGW_FMT(TAG, "MU UE pair solution TV not found: {}, expected output validation disabled", full_sol);
            NVLOGI_FMT(TAG, "Loaded MU UE pair request TV: cell_id={} bytes={} mem_sharing={}", cell_id, tv.muUeGrpReqDataLen,
                       tv.muUeGrpIsMemSharing ? 1 : 0);
            return 0;
        }

        hdf5hpp::hdf5_file sol_file = hdf5hpp::hdf5_file::open(full_sol);
        hdf5hpp::hdf5_dataset ds_sol = sol_file.open_dataset("solution");
        const size_t sol_bytes = ds_sol.get_buffer_size_bytes();
        const size_t sol_size_per_cell = sizeof(cumac_muUeGrp_resp_info_t);

        if (sol_bytes != sol_size_per_cell * cell_num)
        {
            NVLOGE_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "MU UE pair solution size {} != expected {} * {}", sol_bytes, sol_size_per_cell, cell_num);
            return -1;
        }

        // Allocate memory for the whole cell group
        cumac_muUeGrp_resp_info_t* total_buffer = new cumac_muUeGrp_resp_info_t[cell_num];
        ds_sol.read(total_buffer);
        tv_resp.muUeGrpSol = new cumac_muUeGrp_resp_info_t;
        std::memcpy(tv_resp.muUeGrpSol, total_buffer + cell_id, sol_size_per_cell);
        delete[] total_buffer;

        NVLOGI_FMT(TAG, "SFN {}.{} UE_PAIR_TV cell_id={} slot_idx={} srs_slot_lag={} loaded {}: num_srs_info={} mem_sharing={} req_size={} sol_size={}",
            sfn, slot, cell_id, slot_id, cumac_configs->srs_slot_lag, tv_base, tv.muUeGrpInfo->numSrsInfo, tv.muUeGrpIsMemSharing, tv.muUeGrpReqDataLen, sol_size_per_cell);
        return 0;
    }
    catch (const std::exception& e)
    {
        NVLOGE_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "load_mu_ue_grp_tv cell_id={}: {}", cell_id, e.what());
        return -1;
    }
}

/**
 * Load PFM sorting test vector data from separate H5 file
 *
 * @param[in] req cuMAC request containing cell index, slot index, and test vector data
 * @return 0 on success, -1 on error
 */
int cumac_pattern::load_pfm_sorting_tv(cumac_req_t* req)
{
    if (req == nullptr || req->tv_data == nullptr)
    {
        return -1;
    }

    cumac_tti_req_tv_t& tv = req->tv_data->req;
    cumac_tti_resp_tv_t& tv_resp = req->tv_data->resp;
    int cell_id = req->cell_idx;
    int slot_idx = req->slot_idx;

    // TODO: use different PFM SORT TV for each slot
    int tv_id = 0; // slot_idx;

    // Construct filename based on number of cells and slot index: PFM_SORT_TV_<cell_num>CELLS_SLOT_<slot_idx>.h5
    char pfm_tv_filename[MAX_PATH_LEN];
    snprintf(pfm_tv_filename, MAX_PATH_LEN, "PFM_SORT_TV_%dCELLS_SLOT_%d.h5", cell_num, tv_id);
    char pfm_file_path[MAX_PATH_LEN];
    get_cubb_full_path(pfm_file_path, CONFIG_CUMAC_TV_PATH, pfm_tv_filename);

    // Check if file exists
    if(access(pfm_file_path, F_OK) != 0)
    {
        NVLOGF_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "PFM TV file not found: {}", pfm_file_path);
        return -1;
    }

    hdf5hpp::hdf5_file pfm_file;
    try
    {
        pfm_file = hdf5hpp::hdf5_file::open(pfm_file_path);
    }
    catch(const std::exception& e)
    {
        NVLOGE_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "Failed to open PFM TV: cell_id={}, slot_idx={} tv_file={}: {}", cell_id, slot_idx, pfm_tv_filename, e.what());
        tv.pfmCellInfo = nullptr;
        tv_resp.pfmSortSol = nullptr;
        return -1;
    }

    try
    {
        // Load input cell info for this cell
        const std::string input_dataset_name = "INPUT_CELL_INFO_" + std::to_string(cell_id);

        if(pfm_file.is_valid_dataset(input_dataset_name.c_str()))
        {
            tv.pfmCellInfo = new cumac_pfm_cell_info_t;
            hdf5hpp::hdf5_dataset input_dataset = pfm_file.open_dataset(input_dataset_name.c_str());

            const size_t expected_size = sizeof(cumac_pfm_cell_info_t);
            const size_t actual_size = input_dataset.get_buffer_size_bytes();

            if(actual_size != expected_size)
            {
                NVLOGW_FMT(TAG, "PFM input dataset: cell_id={}, slot_idx={} tv_file={} dataset={} size mismatch: expected={} actual={}",
                    cell_id, slot_idx, pfm_tv_filename, input_dataset_name, expected_size, actual_size);
                delete tv.pfmCellInfo;
                tv.pfmCellInfo = nullptr;
                return -1;
            }
            else
            {
                input_dataset.read(reinterpret_cast<uint8_t*>(tv.pfmCellInfo));
            }
        }
        else
        {
            NVLOGW_FMT(TAG, "PFM input dataset: cell_id={}, slot_idx={} tv_file={} dataset={} not found",
                    cell_id, slot_idx, pfm_tv_filename, input_dataset_name);
            tv.pfmCellInfo = nullptr;
            return -1;
        }

        // Load output cell info for this cell
        const std::string output_dataset_name = "OUTPUT_CELL_INFO_" + std::to_string(cell_id);

        if(pfm_file.is_valid_dataset(output_dataset_name.c_str()))
        {
            tv_resp.pfmSortSol = new cumac_pfm_output_cell_info_t;
            hdf5hpp::hdf5_dataset output_dataset = pfm_file.open_dataset(output_dataset_name.c_str());

            const size_t expected_size = sizeof(cumac_pfm_output_cell_info_t);
            const size_t actual_size = output_dataset.get_buffer_size_bytes();

            if(actual_size != expected_size)
            {
                NVLOGW_FMT(TAG, "PFM output dataset: cell_id={}, slot_idx={} tv_file={} dataset={} size mismatch: expected={} actual={}",
                          cell_id, slot_idx, pfm_tv_filename, output_dataset_name, expected_size, actual_size);
                delete tv_resp.pfmSortSol;
                tv_resp.pfmSortSol = nullptr;
                return -1;
            }
            else
            {
                output_dataset.read(reinterpret_cast<uint8_t*>(tv_resp.pfmSortSol));
            }
        }
        else
        {
            NVLOGW_FMT(TAG, "PFM output dataset: cell_id={}, slot_idx={} tv_file={} dataset={} not found",
                    cell_id, slot_idx, pfm_tv_filename, output_dataset_name);
            tv_resp.pfmSortSol = nullptr;
            return -1;
        }

        NVLOGI_FMT(TAG, "Loaded PFM TV: cell_id={}, slot_idx={} tv_file={} dataset={} num_ue={} num_lc_per_ue={} num_lcg_per_ue={}",
            cell_id, slot_idx, pfm_tv_filename, input_dataset_name, tv.pfmCellInfo->num_ue,
            static_cast<unsigned int>(tv.pfmCellInfo->num_lc_per_ue),
            static_cast<unsigned int>(tv.pfmCellInfo->num_lcg_per_ue));

        return 0;
    }
    catch(const std::exception& e)
    {
        NVLOGE_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "Exception loading PFM TV for cell {}: {}", cell_id, e.what());

        if(tv.pfmCellInfo != nullptr)
        {
            delete tv.pfmCellInfo;
            tv.pfmCellInfo = nullptr;
        }

        if(tv_resp.pfmSortSol != nullptr)
        {
            delete tv_resp.pfmSortSol;
            tv_resp.pfmSortSol = nullptr;
        }

        return -1;
    }
}

int cumac_pattern::load_4t4r_tv(cumac_req_t* req)
{
    if (req == nullptr || req->tv_data == nullptr)
    {
        return -1;
    }

    if(req->tv_file.length() == 0)
    {
        NVLOGF_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "TV file is not set for cell {} slot {} ", req->cell_idx, req->slot_idx);
        return -1;
    }

    cumac_tti_req_tv_t&   tv       = req->tv_data->req;
    cumac_cell_configs_t& cell_cfg = get_cumac_cell_configs(req->cell_idx);

    char file_path[MAX_PATH_LEN];
    get_cubb_full_path(file_path, CONFIG_CUMAC_TV_PATH, req->tv_file.c_str());
    if(access(file_path, F_OK) != 0)
    {
        NVLOGF_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "TV cell {} slot {} {} {} file not exist: {}",
                curr_cell, curr_slot, get_task_name(curr_task), curr_tv.c_str(), file_path);
        return -1;
    }

    hdf5hpp::hdf5_file hdf5file;
    try
    {
        hdf5file = hdf5hpp::hdf5_file::open(file_path);
    }
    catch(std::exception& e)
    {
        NVLOGF_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "TV cell {} slot {} {} {} hdf5_file::open failed",
                curr_cell, curr_slot, get_task_name(curr_task), curr_tv.c_str());
        return -1;
    }

    try
    {
        h5dset_try_read(hdf5file, "cellID", &tv.cellID, sizeof(tv.cellID));
        h5dset_try_read(hdf5file, "ULDLSch", &tv.ULDLSch, sizeof(tv.ULDLSch));
        h5dset_try_read(hdf5file, "nActiveUe", &tv.nActiveUe, sizeof(tv.nActiveUe));
        h5dset_try_read(hdf5file, "nSrsUe", &tv.nSrsUe, sizeof(tv.nSrsUe));

        h5dset_try_read(hdf5file, "nPrbGrp", &tv.nPrbGrp, sizeof(tv.nPrbGrp));
        h5dset_try_read(hdf5file, "nBsAnt", &tv.nBsAnt, sizeof(tv.nBsAnt));
        h5dset_try_read(hdf5file, "nUeAnt", &tv.nUeAnt, sizeof(tv.nUeAnt));
        h5dset_try_read(hdf5file, "sigmaSqrd", &tv.sigmaSqrd, sizeof(tv.sigmaSqrd));

        // Create and populate data buffers. Example size: nActiveUe=100, nPrbGrp=68, nBsAnt=4, nUeAnt=4, nMaxSchUePerCell=6
        uint32_t prdLen, detLen, hLen;
        if(tv.ULDLSch == 1)
        { // DL
            prdLen = cell_cfg.nMaxSchUePerCell * tv.nPrbGrp * tv.nBsAnt * tv.nBsAnt;
            detLen = cell_cfg.nMaxSchUePerCell * tv.nPrbGrp * tv.nUeAnt * tv.nUeAnt;
        }
        else
        { // UL
            prdLen = cell_cfg.nMaxSchUePerCell * tv.nPrbGrp * tv.nUeAnt * tv.nUeAnt;
            detLen = cell_cfg.nMaxSchUePerCell * tv.nPrbGrp * tv.nBsAnt * tv.nBsAnt;
        }
        hLen = tv.nPrbGrp * cell_cfg.nMaxSchUePerCell * cell_cfg.nMaxCell * tv.nBsAnt * tv.nUeAnt;

        if ((cumac_configs->task_bitmask & TESTMAC_CUMAC_TASK_MASK_4T4R) != 0U)
        {
            h5dset_try_read_array(hdf5file, "CRNTI", &tv.CRNTI, tv.nActiveUe);                                    // Dataset {100}
            h5dset_try_read_array(hdf5file, "srsCRNTI", &tv.srsCRNTI, cell_cfg.nMaxSchUePerCell);                 // Dataset {6}
            h5dset_try_read_array(hdf5file, "prgMsk", &tv.prgMsk, tv.nPrbGrp);                                    // Dataset {68}
            h5dset_try_read_array(hdf5file, "postEqSinr", &tv.postEqSinr, tv.nActiveUe * tv.nPrbGrp * tv.nUeAnt); // Dataset {27200}
            h5dset_try_read_array(hdf5file, "wbSinr", &tv.wbSinr, tv.nActiveUe * tv.nUeAnt);                      // Dataset {400}

            h5dset_try_read_complex(hdf5file, "detMat_real", "detMat_imag", &tv.detMat, detLen);  // Dataset {6528}
            h5dset_try_read_complex(hdf5file, "estH_fr_real", "estH_fr_imag", &tv.estH_fr, hLen); // Dataset {52224}
            h5dset_try_read_complex(hdf5file, "prdMat_real", "prdMat_imag", &tv.prdMat, prdLen);  // Dataset {6528}

            h5dset_try_read_array(hdf5file, "sinVal", &tv.sinVal, cell_cfg.nMaxSchUePerCell * tv.nPrbGrp * tv.nUeAnt); // Dataset {1632}
            h5dset_try_read_array(hdf5file, "avgRatesActUe", &tv.avgRatesActUe, tv.nActiveUe);                         // Dataset {100}

            h5dset_try_read_array(hdf5file, "tbErrLastActUe", &tv.tbErrLastActUe, tv.nActiveUe); // Dataset {100}

            // Parse TV RESPONSE
            cumac_tti_resp_tv_t& tv_resp = req->tv_data->resp;
            h5dset_try_read_array(hdf5file, "setSchdUePerCellTTI_resp", &tv_resp.setSchdUePerCellTTI, cell_cfg.nMaxSchUePerCell); // Dataset {6}
            h5dset_try_read_array(hdf5file, "mcsSelSol_resp", &tv_resp.mcsSelSol, cell_cfg.nMaxSchUePerCell); // Dataset {6}
            h5dset_try_read_array(hdf5file, "layerSelSol_resp", &tv_resp.layerSelSol, cell_cfg.nMaxSchUePerCell); // Dataset {6}

            uint32_t allocSol_num = cell_cfg.allocType == 0 ? tv.nPrbGrp : 2 * cell_cfg.nMaxSchUePerCell;
            h5dset_try_read_array(hdf5file, "allocSol_resp", &tv_resp.allocSol, allocSol_num); // Dataset {12}
        }
    }
    catch(std::exception& e)
    {
        if(req->tv_data != nullptr)
        {
            free_test_vector(req->tv_data);
            req->tv_data = nullptr;
        }
        NVLOGF_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "TV cell {} slot {} {} {} exception: {}", curr_cell, curr_slot, get_task_name(curr_task), curr_tv.c_str(), e.what());
        return -1;
    }

    return 0;
}

int cumac_pattern::parse_tv_file(cumac_req_t* req)
{
    if(req == nullptr || req->tv_file.length() == 0)
    {
        return -1;
    }

    // operator new throws on failure; no nullptr check needed.
    req->tv_data = new cumac_test_vector_t;

    // Initialize the test vector data
    memset(req->tv_data, 0, sizeof(cumac_test_vector_t));
    req->tv_data->req.taskBitMask = static_cast<uint32_t>(cumac_configs->task_bitmask);

    // Load 4T4R TV data if enabled
    if (cumac_configs->task_bitmask & TESTMAC_CUMAC_TASK_MASK_4T4R)
    {
        if (load_4t4r_tv(req) != 0)
        {
            free_test_vector(req->tv_data);
            req->tv_data = nullptr;
            delete req;
            return -1;
        }
    }

    // Load PFM sorting data from separate H5 file if enabled
    if(req->tv_data != nullptr && (cumac_configs->task_bitmask & (0x1 << CUMAC_TASK_PFM_SORT)))
    {
        if (load_pfm_sorting_tv(req) != 0)
        {
            free_test_vector(req->tv_data);
            req->tv_data = nullptr;
            delete req;
            return -1;
        }
    }

    // Load MU UE pairing / muUeGrp TV (per-cell + solution) when task enabled
    if (req->tv_data != nullptr && (cumac_configs->task_bitmask & (0x1 << CUMAC_TASK_MU_UE_GRP)))
    {
        if (load_mu_ue_grp_tv(req) != 0)
        {
            free_test_vector(req->tv_data);
            req->tv_data = nullptr;
            delete req;
            return -1;
        }
    }

    return 0;
}

int cumac_pattern::load_h5_config_params(int cell_id, const char* config_params_h5_file)
{
    char h5path[MAX_PATH_LEN];
    NVLOGC_FMT(TAG, "config params {} {:p}", config_params_h5_file, (void*)config_params_h5_file);
    get_cubb_full_path(h5path, CONFIG_CUMAC_TV_PATH, config_params_h5_file);
    if(access(h5path, F_OK) != 0)
    {

        NVLOGF_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "Cell_Configs {} file not exist", config_params_h5_file);
    }

    hdf5hpp::hdf5_file hdf5file;
    try
    {
        hdf5file = hdf5hpp::hdf5_file::open(h5path);
    }
    catch(std::exception& e)
    {
        NVLOGE_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "Exception: {}: hdf5_file::open({}): {}", __FUNCTION__, h5path, e.what());
        return -1;
    }

    if(!hdf5file.is_valid_dataset("blerTarget"))
    {
        NVLOGW_FMT(TAG, "Dataset blerTarget not exist in file {}", h5path);
        return -1;
    }

    cumac_cell_configs_t* cfgs = reinterpret_cast<cumac_cell_configs_t*>(malloc(sizeof(cumac_cell_configs_t)));
    if(cfgs == nullptr)
    {
        NVLOGE_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "Failed to allocate memory for cfgs");
        return -1;
    }

    // Per-slot taskBitMask to be enabled in cuMAC-CP
    cfgs->moduleBitMask = static_cast<uint32_t>(cumac_configs->task_bitmask);

    h5dset_try_read(hdf5file, "nMaxCell", &cfgs->nMaxCell, sizeof(cfgs->nMaxCell));
    h5dset_try_read(hdf5file, "nMaxActUePerCell", &cfgs->nMaxActUePerCell, sizeof(cfgs->nMaxActUePerCell));
    h5dset_try_read(hdf5file, "nMaxSchUePerCell", &cfgs->nMaxSchUePerCell, sizeof(cfgs->nMaxSchUePerCell));
    h5dset_try_read(hdf5file, "nMaxPrg", &cfgs->nMaxPrg, sizeof(cfgs->nMaxPrg));

    h5dset_try_read(hdf5file, "nPrbPerPrg", &cfgs->nPrbPerPrg, sizeof(cfgs->nPrbPerPrg));
    h5dset_try_read(hdf5file, "nMaxBsAnt", &cfgs->nMaxBsAnt, sizeof(cfgs->nMaxBsAnt));
    h5dset_try_read(hdf5file, "nMaxUeAnt", &cfgs->nMaxUeAnt, sizeof(cfgs->nMaxUeAnt));
    h5dset_try_read(hdf5file, "scSpacing", &cfgs->scSpacing, sizeof(cfgs->scSpacing));

    h5dset_try_read(hdf5file, "allocType", &cfgs->allocType, sizeof(cfgs->allocType));
    h5dset_try_read(hdf5file, "precoderType", &cfgs->precoderType, sizeof(cfgs->precoderType));
    h5dset_try_read(hdf5file, "receiverType", &cfgs->receiverType, sizeof(cfgs->receiverType));
    h5dset_try_read(hdf5file, "colMajChanAccess", &cfgs->colMajChanAccess, sizeof(cfgs->colMajChanAccess));

    h5dset_try_read(hdf5file, "betaCoeff", &cfgs->betaCoeff, sizeof(cfgs->betaCoeff));
    h5dset_try_read(hdf5file, "sinValThr", &cfgs->sinValThr, sizeof(cfgs->sinValThr));
    h5dset_try_read(hdf5file, "corrThr", &cfgs->corrThr, sizeof(cfgs->corrThr));
    h5dset_try_read(hdf5file, "prioWeightStep", &cfgs->prioWeightStep, sizeof(cfgs->prioWeightStep));

    h5dset_try_read(hdf5file, "harqEnabledInd", &cfgs->harqEnabledInd, sizeof(cfgs->harqEnabledInd));
    h5dset_try_read(hdf5file, "mcsSelCqi", &cfgs->mcsSelCqi, sizeof(cfgs->mcsSelCqi));
    h5dset_try_read(hdf5file, "mcsSelSinrCapThr", &cfgs->mcsSelSinrCapThr, sizeof(cfgs->mcsSelSinrCapThr));
    h5dset_try_read(hdf5file, "mcsSelLutType", &cfgs->mcsSelLutType, sizeof(cfgs->mcsSelLutType));

    h5dset_try_read(hdf5file, "blerTarget", &cfgs->blerTarget, sizeof(cfgs->blerTarget));

    NVLOGC_FMT(TAG, "{}: cell_id={} nMaxCell={}->{} nMaxPrg={} nPrbPerPrg={} nMaxBsAnt={} nMaxUeAnt={} harqEnabledInd={} h5file={}",
            __func__, cell_id, cfgs->nMaxCell, cell_num, cfgs->nMaxPrg, cfgs->nPrbPerPrg, cfgs->nMaxBsAnt, cfgs->nMaxUeAnt, cfgs->harqEnabledInd, config_params_h5_file);

    cfgs->nMaxCell = cell_num;

    // Check group common parameters
    if (cell_id > 0) {
        cumac_cell_configs_t& cfgs0 = *cumac_cell_configs_v[0];

        CHECK_VALUE_EQUAL_ERR(cfgs0.nMaxCell, cfgs->nMaxCell);
        CHECK_VALUE_EQUAL_ERR(cfgs0.nMaxActUePerCell, cfgs->nMaxActUePerCell);
        CHECK_VALUE_EQUAL_ERR(cfgs0.nMaxSchUePerCell, cfgs->nMaxSchUePerCell);
        CHECK_VALUE_EQUAL_ERR(cfgs0.nMaxPrg, cfgs->nMaxPrg);

        CHECK_VALUE_EQUAL_ERR(cfgs0.nPrbPerPrg, cfgs->nPrbPerPrg);
        CHECK_VALUE_EQUAL_ERR(cfgs0.nMaxBsAnt, cfgs->nMaxBsAnt);
        CHECK_VALUE_EQUAL_ERR(cfgs0.nMaxUeAnt, cfgs->nMaxUeAnt);
        CHECK_VALUE_EQUAL_ERR(cfgs0.scSpacing, cfgs->scSpacing);

        CHECK_VALUE_EQUAL_ERR(cfgs0.allocType, cfgs->allocType);
        CHECK_VALUE_EQUAL_ERR(cfgs0.precoderType, cfgs->precoderType);
        CHECK_VALUE_EQUAL_ERR(cfgs0.receiverType, cfgs->receiverType);
        CHECK_VALUE_EQUAL_ERR(cfgs0.colMajChanAccess, cfgs->colMajChanAccess);

        CHECK_VALUE_EQUAL_ERR(cfgs0.betaCoeff, cfgs->betaCoeff);
        CHECK_VALUE_EQUAL_ERR(cfgs0.sinValThr, cfgs->sinValThr);
        CHECK_VALUE_EQUAL_ERR(cfgs0.corrThr, cfgs->corrThr);
        CHECK_VALUE_EQUAL_ERR(cfgs0.prioWeightStep, cfgs->prioWeightStep);
        CHECK_VALUE_EQUAL_ERR(cfgs0.blerTarget, cfgs->blerTarget);
    }

    cumac_cell_configs_v[cell_id] = cfgs;
    hdf5file.close();
    return 0;
}

struct parsing_thread_arg_t {
    int thread_id;
    int thread_num;
    int cell_num;
    int cpu_core;
    int schedule_slot_period;

    std::vector<int32_t>* p_lp_cell_id_vec;

    pthread_t pid;
    cumac_pattern* lp;
    yaml::node* yaml_node;
    cumac_slot_pattern_t* slot_data;
};

static void* parsing_thread_func(void* arg)
{
    struct parsing_thread_arg_t* slot_parsing = (parsing_thread_arg_t*)arg;
    cumac_pattern* lp = slot_parsing->lp;
    yaml::node& slot_list = *(slot_parsing->yaml_node);
    cumac_slot_pattern_t& slots_data = *(slot_parsing->slot_data);
    size_t pattern_length = slot_list.length();

    char thread_name[32];
    snprintf(thread_name, 32, "cumac_lp_%02d", slot_parsing->thread_id);
    if(pthread_setname_np(pthread_self(), thread_name) != 0)
    {
        NVLOGW_FMT(TAG, "{}: set thread name {} failed", __func__, thread_name);
    }

    nv_assign_thread_cpu_core(slot_parsing->cpu_core);
    NVLOGC_FMT(TAG, "{}: thread {:02d} started on CPU core {:02d} pattern_length={} schedule_period={}",
        __FUNCTION__, slot_parsing->thread_id, slot_parsing->cpu_core, pattern_length, slot_parsing->schedule_slot_period);

    for(size_t id = 0; id < slot_parsing->schedule_slot_period; ++id)
    {
        if(is_app_exiting())
        {
            NVLOGC_FMT(TAG, "{}: thread [{}] exiting due to application exiting", __FUNCTION__, slot_parsing->thread_id);
            break;
        }

        int slot_id = id;
        curr_slot = slot_id;
        yaml::node cumac_cell_configs = slot_list[id % pattern_length]["config"];
        if(cumac_cell_configs.type() == YAML_SEQUENCE_NODE)
        {
            for (int cell_id = slot_parsing->thread_id; cell_id < slot_parsing->cell_num; cell_id += slot_parsing->thread_num)
            {
                std::vector<int32_t>& lp_cell_id = *slot_parsing->p_lp_cell_id_vec;
                if(lp->populate_cumac_pattern(slots_data, cumac_cell_configs[lp_cell_id[cell_id]], cell_id, slot_id) < 0)
                {
                    std::stringstream ss;
                    ss << __FUNCTION__ << " error: slot=" << slot_id << " lp_cell_id=" << lp_cell_id[cell_id] << std::endl;
                    throw std::runtime_error(ss.str().c_str());
                }
                NVLOGC_FMT(TAG, "{}: thread [{}] parsed TVs for cell {:02d} slot {:02d}/{}",
                    __FUNCTION__, slot_parsing->thread_id, cell_id, id, slot_parsing->schedule_slot_period);
            }
        }
    }

    return nullptr;
}

void cumac_pattern::parse_slots(yaml::node& slot_list, cumac_slot_pattern_t& slots_data)
{
    if (sched_slot_num == 0)
    {
        sched_slot_num = slot_list.length();
    }
    slots_data.resize(static_cast<size_t>(sched_slot_num));

    for(int slot_id = 0; slot_id < slots_data.size(); slot_id++)
    {
        slots_data[slot_id].resize(cell_num);
        for(int cell_id = 0; cell_id < cell_num; cell_id++)
        {
            slots_data[slot_id][cell_id].resize(CUMAC_REQ_SIZE);
        }
    }

    int thread_num = 1;

    struct parsing_thread_arg_t thread_args[thread_num];
    for(int thread_id = 0; thread_id < thread_num; thread_id++)
    {
        struct parsing_thread_arg_t& thread_arg = thread_args[thread_id];
        thread_arg.lp = this;
        thread_arg.yaml_node = &slot_list;
        thread_arg.slot_data = &slots_data;
        thread_arg.thread_id = thread_id;
        thread_arg.thread_num = thread_num;
        thread_arg.cell_num = cell_num;
        thread_arg.cpu_core = cumac_configs->get_recv_thread_config().cpu_affinity;
        thread_arg.schedule_slot_period = sched_slot_num;
        thread_arg.p_lp_cell_id_vec = &lp_cell_id_vec;

        if(pthread_create(&thread_arg.pid, NULL, parsing_thread_func, &thread_arg) != 0)
        {
            NVLOGE_FMT(TAG, AERIAL_THREAD_API_EVENT, "{}: task create failed: thread_id={}", __func__, thread_id);
        }
    }

    for(int thread_id = 0; thread_id < thread_num; thread_id++)
    {
        struct parsing_thread_arg_t& thread_arg = thread_args[thread_id];
        if(pthread_join(thread_arg.pid, NULL) != 0)
        {
            NVLOGE_FMT(TAG, AERIAL_THREAD_API_EVENT, "{}: task join failed: thread_id={}", __func__, thread_id);
        }
    }
}

int cumac_pattern::cumac_pattern_parsing(const char* lp_file_name, uint32_t ch_mask, uint64_t cell_mask)
{
    channel_mask = ch_mask;

    char pattern_file[MAX_PATH_LEN];
    get_cubb_full_path(pattern_file, CONFIG_LAUNCH_PATTERN_PATH, lp_file_name);
    if(access(pattern_file, F_OK) != 0)
    {
        if(cumac_configs->cumac_cell_num > 0)
        {
            // Debug with cumac_pattern_default.yaml
            get_cubb_full_path(pattern_file, CONFIG_LAUNCH_PATTERN_PATH, "cumac_pattern_default.yaml");
            if(access(pattern_file, F_OK) != 0)
            {
                NVLOGF_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "YAML file not exist: {}", pattern_file);
            }
        }
        else
        {
            NVLOGF_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "YAML file not exist: {}", pattern_file);
        }
    }

    yaml::file_parser fp(pattern_file);
    yaml::document    doc     = fp.next_document();
    yaml::node        pattern = doc.root();

    cell_num = pattern["Cell_Configs"].length();
    if(cumac_configs->cumac_cell_num > 0 && cumac_configs->cumac_cell_num < cell_num)
    {
        // Debug with cumac_pattern_default.yaml and test_cumac_config.yaml configured cumac_cell_num
        cell_num = cumac_configs->cumac_cell_num;
    }

    if (cell_mask != 0)
    {
        // Select some cells to start by --cells input parameter
        int32_t cell_id = 0;
        for (int lp_cell_id = 0; lp_cell_id < sizeof(cell_mask) * 8; lp_cell_id++)
        {
            if ((cell_mask & ((uint64_t)0x1 << lp_cell_id)) != 0)
            {
                lp_cell_id_vec.push_back(lp_cell_id);
                NVLOGC_FMT(TAG, "{}: cell_mask=0x{:X} map cell_id={} to lp_cell_id={}", __FUNCTION__, cell_mask, cell_id, lp_cell_id);
                cell_id ++;
            }
        }

        if (cell_num < lp_cell_id_vec.size()) {
            NVLOGE_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "cell_mask=0x{:X} size {} exceeds cumac_pattern cell_num {}", cell_mask, lp_cell_id_vec.size(), cell_num);
            return -1;
        }

        cell_num = lp_cell_id_vec.size();
    }
    else
    {
        // Enable all cells by default, no mapping
        for (int cell_id = 0; cell_id < cell_num; cell_id ++)
        {
            lp_cell_id_vec.push_back(cell_id);
        }
    }

    cumac_cell_configs_v.resize(cell_num);
    auto allocate_default_cell_configs = [&]() -> int {
        for(int cell_id = 0; cell_id < cell_num; cell_id++)
        {
            cumac_cell_configs_t* cfgs = static_cast<cumac_cell_configs_t*>(calloc(1, sizeof(cumac_cell_configs_t)));
            if (cfgs == nullptr)
            {
                NVLOGE_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "{}: failed to allocate default cell config for cell {}", __FUNCTION__, cell_id);
                return -1;
            }
            cfgs->moduleBitMask = static_cast<uint32_t>(cumac_configs->task_bitmask);
            cfgs->nMaxCell = static_cast<uint8_t>(cell_num);
            cumac_cell_configs_v[cell_id] = cfgs;
        }
        return 0;
    };

    if(pattern.has_key("Cell_Configs"))
    {
        // Load from list
        yaml::node cumac_cell_configs_list = pattern["Cell_Configs"];
        if(cumac_cell_configs_list.length() != cell_num)
        {
            NVLOGW_FMT(TAG, "{}: cell num doesn't match: cell_num={} config_list={}", __FUNCTION__, cell_num, cumac_cell_configs_list.length());
        }
        NVLOGC_FMT(TAG, "{}: config_list={} cell_mask={} cell_num={} lp_cell_id.size={}",
                __FUNCTION__, cumac_cell_configs_list.length(), cell_mask, cell_num, lp_cell_id_vec.size());

        if ((cumac_configs->task_bitmask & TESTMAC_CUMAC_TASK_MASK_4T4R) != 0U)
        {
            for(int cell_id = 0; cell_id < cell_num; cell_id++)
            {
                int32_t lp_cell_id = lp_cell_id_vec[cell_id];
                std::string config_tv = cumac_cell_configs_list[lp_cell_id].as<std::string>();
                if(load_h5_config_params(cell_id, config_tv.c_str()) < 0)
                {
                    return -1;
                }
            }
        }
        else
        {
            // No 4T4R tasks: allocate default cell configs so pointers are valid
            if (allocate_default_cell_configs() < 0)
            {
                return -1;
            }
        }
    }
    else if ((cumac_configs->task_bitmask & TESTMAC_CUMAC_TASK_MASK_4T4R) == 0U)
    {
        // No Cell_Configs key and no 4T4R tasks: still need valid pointers
        if (allocate_default_cell_configs() < 0)
        {
            return -1;
        }
    }

    // Expand expected throughput array size and initialize to 0
    expected.resize(cell_num);
    for(int i = 0; i < cell_num; i++)
    {
        expected[i].reset();
    }

    // Probe the schedule_slot_period attribute from the muUePairTV files
    if (cumac_configs->task_bitmask & (0x1 << CUMAC_TASK_MU_UE_GRP))
    {
        sched_slot_num = probe_schedule_slot_period();
        if (sched_slot_num > 0)
        {
            first_ue_group_slot = probe_first_ue_group_slot();
        }
    }

    if(yaml_configs->app_mode == 0 && pattern.has_key("SCHED"))
    {
        int64_t ts_start = system_clock::now().time_since_epoch().count();
        yaml::node slot_list = pattern["SCHED"];
        parse_slots(slot_list, sched_slots_pattern);
        int64_t ts_end = system_clock::now().time_since_epoch().count();
        float time_cost = (float)(ts_end - ts_start) / 1E9;
        NVLOGC_FMT(TAG, "{}: parsed SCHED slots: cell_num={} sched_slot_num={} taskBitMask=0x{:02X} srs_slot_lag={} first_ue_group_slot={} time={:.1f}s",
            __FUNCTION__, cell_num, sched_slot_num, cumac_configs->task_bitmask, cumac_configs->srs_slot_lag, first_ue_group_slot, time_cost);
    }

    if (sched_slot_num <= 0)
    {
        return -1;
    }

    // Update expected throughput values to per second total values
    for(int i = 0; i < cell_num; i++)
    {
        int repeat_per_second = SLOTS_PER_SECOND / sched_slot_num;
        cumac_thrput_t& thrput = expected[i];
    
        thrput.error = thrput.error * repeat_per_second;
        thrput.cumac_slots = thrput.cumac_slots * repeat_per_second;
        thrput.invalid = thrput.invalid * repeat_per_second;
        for (int task = 0; task < CUMAC_TASK_TOTAL_NUM; task++) {
            thrput.task_slots[task] = thrput.task_slots[task] * repeat_per_second;
        }

        // Set expected task slots based on task_bitmask
        const int task_bitmask = cumac_configs->task_bitmask;
        std::string exp_data = "CUMAC_TargetThrput: Cell=" + std::to_string(i);
        exp_data.append(" CUMAC_SLOT=").append(std::to_string(thrput.cumac_slots));

        // Add expected task slots for enabled tasks
        if (task_bitmask & (0x1 << CUMAC_TASK_UE_SELECTION)) {
            exp_data.append(" UE_SEL=").append(std::to_string(thrput.task_slots[CUMAC_TASK_UE_SELECTION]));
        }
        if (task_bitmask & (0x1 << CUMAC_TASK_PRB_ALLOCATION)) {
            exp_data.append(" PRB_ALLOC=").append(std::to_string(thrput.task_slots[CUMAC_TASK_PRB_ALLOCATION]));
        }
        if (task_bitmask & (0x1 << CUMAC_TASK_LAYER_SELECTION)) {
            exp_data.append(" LAYER_SEL=").append(std::to_string(thrput.task_slots[CUMAC_TASK_LAYER_SELECTION]));
        }
        if (task_bitmask & (0x1 << CUMAC_TASK_MCS_SELECTION)) {
            exp_data.append(" MCS_SEL=").append(std::to_string(thrput.task_slots[CUMAC_TASK_MCS_SELECTION]));
        }
        if (task_bitmask & (0x1 << CUMAC_TASK_PFM_SORT)) {
            exp_data.append(" PFM_SORT=").append(std::to_string(thrput.task_slots[CUMAC_TASK_PFM_SORT]));
        }
        if (task_bitmask & (0x1 << CUMAC_TASK_MU_UE_GRP)) {
            exp_data.append(" MU_UE_GRP=").append(std::to_string(thrput.task_slots[CUMAC_TASK_MU_UE_GRP]));
        }

        exp_data.append(" ERR=").append(std::to_string(thrput.error));
        exp_data.append(" INV=").append(std::to_string(thrput.invalid));
        NVLOGC_FMT(TAG, "{}", exp_data.c_str());
    }

    return 0;
}

int cumac_pattern::update_expected_values(int cell_id, cumac_req_t *req)
{
    if (req == nullptr || req->tv_data == nullptr)
    {
        return -1;
    }

    cumac_thrput_t& thrput = expected[cell_id];
    if (req->tv_data->req.taskBitMask != 0)
    {
        thrput.cumac_slots ++;
    }

    for (int task = 0; task < CUMAC_TASK_TOTAL_NUM; task++)
    {
        if (req->tv_data->req.taskBitMask & (0x1 << task))
        {
            thrput.task_slots[task] ++;
        }
    }
    return 0;
}

int cumac_pattern::populate_cumac_pattern(cumac_slot_pattern_t& slots_data, yaml::node cell_config, int cell_id, int slot_idx)
{
    if(cell_config.type() == YAML_SCALAR_NODE)
    {
        return 0;
    }

    int cell_index = cell_config["cell_index"].as<int32_t>();
    if (lp_cell_id_vec[cell_id] != cell_index)
    {
        NVLOGE_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "{}: slot {} cell_id={} lp_cell_id[cell_id]={} cell_index={}",
                __func__, slot_idx, cell_id, lp_cell_id_vec[cell_id], cell_index);
    }
    curr_cell = cell_id;

    yaml::node  channel_list = cell_config["channels"];
    std::string node_type    = channel_list.type_string();
    std::vector<std::string> channel_type_list;
    NVLOGV_FMT(TAG, "slot {} channel_list node type: {}", slot_idx, node_type.c_str());
    if(node_type == "YAML_SCALAR_NODE")
    {
        return 0;
    }

    for(int i = 0; i < channel_list.length(); ++i)
    {
        std::string tv_file = channel_list[i].as<std::string>();
    
        NVLOGV_FMT(TAG, "slot {} channel_list[{}].type_string(): {}", slot_idx, i, channel_list[i].type_string());
        channel_type_list.clear();

        cumac_task_type_t task_type = CUMAC_TASK_UE_SELECTION;
        cumac_req_t* req = new cumac_req_t;
        req->cell_idx = cell_id;
        req->slot_idx = slot_idx;
        req->tv_file = tv_file;
        curr_task = CUMAC_TASK_UE_SELECTION;
        if (parse_tv_file(req) < 0) {
            // delete req;
            continue;
        }
        update_expected_values(cell_id, req);
        slots_data[slot_idx][cell_id][CUMAC_SCH_TTI_REQ].push_back(req);
        NVLOGI_FMT(TAG, "{}: added tv: slot {} cell_id {} {:<8} {} task_bitmask=0x{:X}", __func__, slot_idx, cell_id,
            get_task_name(task_type), req->tv_file.c_str(), req->tv_data == nullptr ? 0 : req->tv_data->req.taskBitMask);
    }

    return 0;
}

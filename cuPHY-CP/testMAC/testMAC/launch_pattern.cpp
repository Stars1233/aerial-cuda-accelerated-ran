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
#include <filesystem>

#include "hdf5hpp.hpp"
#include "cuphy_hdf5.hpp"
#include "nvlog.hpp"

#include "launch_pattern.hpp"
#include "common_defines.hpp"
#include "common_utils.hpp"
#include "dyn_param_calc.hpp"

#include <chrono>

#define TAG (NVLOG_TAG_BASE_TEST_MAC + 1) // "MAC.LP"

// Only used for calculate phyCellId from testMac_config_params_XXX.h5
#define CONFIG_PHY_CELL_ID_BASE 40 // phyCellId = CONFIG_PHY_CELL_ID_BASE + cell_number

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
thread_local int curr_cell;
thread_local int curr_slot;
thread_local int curr_ch;
thread_local std::string curr_tv;

static uint32_t constexpr CV_MEM_BANK_MAX_UES = 192;

template <typename T>
static int yaml_try_parse_list(yaml::node& parent_node, const char* name, std::vector<T>& values)
{
    yaml::node list_nodes = parent_node[name];
    if(list_nodes.type() != YAML_SEQUENCE_NODE)
    {
        NVLOGE_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "{}: failed to parse {}: error type {}\n", __func__, name, +list_nodes.type());
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
                curr_cell, curr_slot, get_channel_name(curr_ch), curr_tv.c_str(), name);
        return -1;
    }

    try
    {
        hdf5hpp::hdf5_dataset h5dset = file.open_dataset(name);
        if(h5dset.get_buffer_size_bytes() != size)
        {
            NVLOGW_FMT(TAG, "TV cell {} slot {} {} {} dataset {} size doesn't match: dataset_size={} buf_size={}",
                    curr_cell, curr_slot, get_channel_name(curr_ch), curr_tv.c_str(),
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
                curr_cell, curr_slot, get_channel_name(curr_ch), curr_tv.c_str(), name, e.what());
    }
    return -1;
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
    delete [] src;
    return ret;
}

template <typename TypeSrc, typename TypeDst>
static int h5dset_try_read_convert(hdf5hpp::hdf5_file& file, const char* name, TypeDst* dst, uint32_t num)
{
    if (num == 0)
    {
        NVLOGW_FMT(TAG, "TV cell {} slot {} {} {} dataset {} reading with num=0",
                curr_cell, curr_slot, get_channel_name(curr_ch), curr_tv.c_str(), name);
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
    delete [] src;
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
                    curr_cell, curr_slot, get_channel_name(curr_ch), curr_tv.c_str(), name);
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
    char h5path_array[MAX_PATH_LEN];
    get_full_path_file(h5path_array, CONFIG_TEST_VECTOR_PATH, file_name, CONFIG_CUBB_ROOT_DIR_RELATIVE_NUM);
    std::filesystem::path h5path(h5path_array);

    if(access(h5path.c_str(), F_OK) != 0)
    {
        NVLOGF_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "H5 file {} not exist", h5path.c_str());
    }

    hdf5hpp::hdf5_file hdf5file;
    try
    {
        hdf5file = hdf5hpp::hdf5_file::open(h5path.c_str());
    }
    catch(std::exception& e)
    {
        NVLOGE_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "Exception: {}: hdf5_file::open({}): {}", __FUNCTION__, h5path.c_str(), e.what());
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

// Return whether it is a PRACH Msg 2~4, HARQ or not
tv_type_t get_tv_type(std::string& tv_file)
{
    if(strcasestr(tv_file.c_str(), "msg2") != NULL)
    {
        return tv_type_t::TV_PRACH_MSG2;
    }
    else if(strcasestr(tv_file.c_str(), "msg3") != NULL)
    {
        return tv_type_t::TV_PRACH_MSG3;
    }
    else if(strcasestr(tv_file.c_str(), "msg4") != NULL)
    {
        return tv_type_t::TV_PRACH_MSG4;
    }
    else if(strcasestr(tv_file.c_str(), "harq") != NULL)
    {
        return tv_type_t::TV_HARQ;
    }
    else
    {
        return tv_type_t::TV_GENERIC;
    }
}

std::vector<int> get_pdu_index_list(hdf5hpp::hdf5_file& file, channel_type_t channel)
{
    std::vector<int> list;
    tv_channel_type_t tv_type = get_tv_channel_type(channel);

    if(file.is_valid_dataset(H5_N_PDU))
    {
        uint32_t nPdu = 0;
        hdf5hpp::hdf5_dataset dset = file.open_dataset(H5_N_PDU);
        dset.read(&nPdu);
        for(uint i = 1; i <= nPdu; i++)
        {
            std::string name = H5_PDU + std::to_string(i);
            if(!file.is_valid_dataset(name.c_str()))
            {
                NVLOGE_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "H5 dataset {} doesn't exist", name.c_str());
                continue;
            }

            hdf5hpp::hdf5_dataset hdf5dset = file.open_dataset(name.c_str());
            uint32_t type = hdf5dset[0]["type"].as<uint32_t>();
            if (type == tv_type)
            {
                list.push_back(i);
            }
        }
    }
    return list;
}

launch_pattern::launch_pattern(test_mac_configs* configs)
{
    yaml_configs               = configs;
    fapi_type                  = 0;
    channel_mask               = 0;
    cell_num                   = 0;
    sched_slot_num             = 0;
    init_slot_num              = 0;
    slots_per_frame            = SLOTS_PER_FRAME;
    config_static_harq_proc_id = 0;
    negative_test              = 0;
    prach_reconfig_flag        = 0;
    using_init_patterns        = false;
    enable_dynamic_BF          = 0;
    enable_static_dynamic_BF   = 0;
    enable_srs                 = 0;

    cv_membank_config_read_dl_bfw.fill(false);
    cv_membank_config_read_ul_bfw.fill(false);
}

int launch_pattern::init_prach_config(int cell_num)
{
    // Load default PRACH from yaml file and h5 Cell_Config. It may be updated in PRACH TV parsing
    yaml::node& root_node             = yaml_configs->get_yaml_config();
    yaml::node  config_req_params     = root_node["data"];

    init_prach_configs.resize(cell_num);

    for (int cell_id = 0; cell_id < cell_num; cell_id ++)
    {
        prach_configs_t& prach_configs = get_prach_configs(cell_id);
        prach_configs.prachSubCSpacing    = 1;
        prach_configs.SsbPerRach          = config_req_params["nPrachSsbRach"].as<int32_t>();
        prach_configs.prachConfigIndex    = config_req_params["nPrachConfIdx"].as<int32_t>();
        prach_configs.prachSequenceLength = prach_configs.prachConfigIndex > 66 ? 1 : 0; //  38.211 Table 6.3.3.2-2
        prach_configs.numPrachFdOccasions = config_req_params["nPrachFdm"].as<int32_t>();
        prach_configs.restrictedSetConfig = config_req_params["nPrachRestrictSet"].as<uint32_t>();
        for(int i = 0; i < prach_configs.numPrachFdOccasions; i++)
        {
            prach_fd_occasion_config_t fd_occasion;
            fd_occasion.prachRootSequenceIndex = config_req_params["nPrachRootSeqIdx"].as<int32_t>();
            fd_occasion.numRootSequences       = 64;
            fd_occasion.k1                     = config_req_params["nPrachFreqStart"].as<int32_t>();
            fd_occasion.prachZeroCorrConf      = config_req_params["nPrachZeroCorrConf"].as<int32_t>();
            fd_occasion.numUnusedRootSequences = 0;
            prach_configs.prachFdOccasions.push_back(std::move(fd_occasion));
        }
    }

    return 0;
}

launch_pattern::~launch_pattern()
{
    if (yaml_configs->tv_data_map_enable != 0)
    {
        for(auto& per_channel_map: tv_data_maps)
        {
            for(auto& map_entry: per_channel_map)
            {
                delete map_entry.second;
            }
        }
    }

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
                        delete req->tv_data;
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
                        delete req->tv_data;
                        delete req;
                    }
                }
            }
        }
    }
    for(auto cell_config : cell_configs_v)
    {
        delete cell_config;
    }
}

int launch_pattern::parse_tv_pdcch(hdf5hpp::hdf5_file& file, fapi_req_t* req)
{
    pdcch_tv_t&   pdcch_tv = req->tv_data->pdcch_tv;
    tv_dset_list& list     = req->tv_data->dset_tv;

    channel_type_t type = req->channel;
    if(list.find(type) == list.end())
    {
        return -1;
    }

    auto& dsets = list[type];

    for(auto& dset : dsets)
    {
        hdf5hpp::hdf5_dataset pdu_dset = file.open_dataset(dset.c_str());
        coresetinfo_          cset;
        cset.BWPSize             = pdu_dset[0]["BWPSize"].as<uint32_t>();
        cset.BWPStart            = pdu_dset[0]["BWPStart"].as<uint32_t>();
        cset.SubcarrierSpacing   = pdu_dset[0]["SubcarrierSpacing"].as<uint32_t>();
        cset.CyclicPrefix        = pdu_dset[0]["CyclicPrefix"].as<uint32_t>();
        cset.StartSymbolIndex    = pdu_dset[0]["StartSymbolIndex"].as<uint32_t>();
        cset.DurationSymbols     = pdu_dset[0]["DurationSymbols"].as<uint32_t>();
        cset.CceRegMappingType   = pdu_dset[0]["CceRegMappingType"].as<uint32_t>();
        cset.RegBundleSize       = pdu_dset[0]["RegBundleSize"].as<uint32_t>();
        cset.InterleaverSize     = pdu_dset[0]["InterleaverSize"].as<uint32_t>();
        cset.CoreSetType         = pdu_dset[0]["CoreSetType"].as<uint32_t>();
        cset.ShiftIndex          = pdu_dset[0]["ShiftIndex"].as<uint32_t>();
        cset.precoderGranularity = pdu_dset[0]["precoderGranularity"].as<uint32_t>();
        try
        {
            cset.testModel           = pdu_dset[0]["testModel"].as<uint32_t>();
        }
        catch(...)
        {
            cset.testModel           = 0; // test model disabled
        }
        cset.numDlDci            = pdu_dset[0]["numDlDci"].as<uint32_t>();
        cset.FreqDomainResource  = static_cast<uint64_t>(pdu_dset[0]["FreqDomainResource0"].as<uint32_t>()) << 32 | pdu_dset[0]["FreqDomainResource1"].as<uint32_t>();
        cset.dciList.resize(cset.numDlDci);
        for(uint32_t i = 0; i < cset.numDlDci; i++)
        {
            std::string dcidsinfo      = dset + "_DCI" + std::to_string(i + 1);
            auto        hdf5dset       = file.open_dataset(dcidsinfo.c_str());
            auto&       dci            = cset.dciList[i];
            dci.RNTI                   = hdf5dset[0]["RNTI"].as<uint16_t>();
            dci.ScramblingId           = hdf5dset[0]["ScramblingId"].as<uint16_t>();
            dci.ScramblingRNTI         = hdf5dset[0]["ScramblingRNTI"].as<uint16_t>();
            dci.CceIndex               = hdf5dset[0]["CceIndex"].as<uint8_t>();
            dci.AggregationLevel       = hdf5dset[0]["AggregationLevel"].as<uint8_t>();
            dci.PayloadSizeBits        = hdf5dset[0]["PayloadSizeBits"].as<uint16_t>();
#ifdef SCF_FAPI_10_04
            dci.powerControlOffsetSSProfileNR   = hdf5dset[0]["powerControlOffsetSSProfileNR"].as<int8_t>();
#else
            dci.powerControlOffsetSS   = hdf5dset[0]["powerControlOffsetSS"].as<uint8_t>();
#endif
            dci.beta_PDCCH_1_0         = hdf5dset[0]["beta_PDCCH_1_0"].as<uint8_t>();

            std::string           dcids   = dcidsinfo + "_Payload";
            hdf5hpp::hdf5_dataset payload = file.open_dataset(dcids.c_str());
            payload.read(dci.Payload);

            for(int i = 0; i < (dci.PayloadSizeBits + 7) / 8; i++)
            {
                NVLOGD_FMT(TAG, "Payload[{}] : 0x{:x}", i, dci.Payload[i]);
            }
            parse_tx_beamforming(hdf5dset, dci.tx_beam_data);
        }
        pdcch_tv.coreset.push_back(std::move(cset));
    }

    return 0;
}

int launch_pattern::parse_tv_pdsch(hdf5hpp::hdf5_file& file, fapi_req_t* req)
{
    pdsch_tv_t&   pdsch_tv = req->tv_data->pdsch_tv;
    tv_dset_list& list     = req->tv_data->dset_tv;

    pdsch_tv.data_size = 0;

    if(list.find(channel_type_t::PDSCH) == list.end())
    {
        return -1;
    }

    std::size_t               tb_offset = 0;
    std::vector<std::string>& dsets     = list[channel_type_t::PDSCH];
    for(std::string& dset : dsets)
    {
        std::string dsPayload = dset + "_payload";
        if(!file.is_valid_dataset(dset.c_str()) || !file.is_valid_dataset(dsPayload.c_str()))
        {
            continue;
        }

        hdf5hpp::hdf5_dataset payload      = file.open_dataset(dsPayload.c_str());
        size_t                payload_size = payload.get_buffer_size_bytes();
        uint32_t padding_nbytes = (~payload_size + 1) & (yaml_configs->pdsch_align_bytes -1);
        pdsch_tv.data_size += payload_size + padding_nbytes;
    }
    pdsch_tv.data_buf.resize(pdsch_tv.data_size);

    pdsch_tv.type = get_tv_type(req->tv_file);

    for(std::string& dset : dsets)
    {
        std::string dsPayload = dset + "_payload";
        if(!file.is_valid_dataset(dset.c_str()) || !file.is_valid_dataset(dsPayload.c_str()))
        {
            continue;
        }

        hdf5hpp::hdf5_dataset hdf5dset = file.open_dataset(dset.c_str());

        pdsch_tv_data_t* tv_data = new pdsch_tv_data_t;
        tv_data->BWPStart  = hdf5dset[0]["BWPStart"].as<uint32_t>();
        tv_data->BWPSize   = hdf5dset[0]["BWPSize"].as<uint32_t>();
        tv_data->ref_point = hdf5dset[0]["refPoint"].as<uint32_t>();
        tv_data->SubcarrierSpacing = hdf5dset[0]["SubcarrierSpacing"].as<uint32_t>();
        tv_data->CyclicPrefix = hdf5dset[0]["CyclicPrefix"].as<uint32_t>();
        tv_data->resourceAlloc = hdf5dset[0]["resourceAlloc"].as<uint32_t>();

        tv_data->rbBitmap = h5dset_try_parse<std::vector<uint32_t>>(hdf5dset, "rbBitmap", {});

        // ue_pars
        tv_data->tbpars.nRnti       = hdf5dset[0]["RNTI"].as<uint32_t>();
        tv_data->tbpars.numLayers   = hdf5dset[0]["nrOfLayers"].as<uint32_t>();
        tv_data->tbpars.dataScramId = hdf5dset[0]["dataScramblingId"].as<uint32_t>(); // Cell ID 41
        tv_data->tbpars.nSCID       = hdf5dset[0]["SCID"].as<uint32_t>();
        uint32_t portValue          = hdf5dset[0]["dmrsPorts"].as<uint32_t>();
        tv_data->tbpars.nPortIndex  = portValue;
        int dmrsBitPos              = 0;
        for(int i = 0; i < 8; i++)
        {
            tv_data->portIndex[i] = 0;
        }

        for(int i = 0; i < 8; i++)
        {
            if(portValue & 0x1)
            {
                tv_data->portIndex[dmrsBitPos] = i;
                dmrsBitPos++;
            }
            portValue >>= 1;
        }

        // cw_pars
        tv_data->tbpars.rv            = hdf5dset[0]["rvIndex"].as<uint32_t>();
        tv_data->tbpars.mcsTableIndex  = hdf5dset[0]["mcsTable"].as<uint32_t>();
        tv_data->tbpars.mcsIndex       = hdf5dset[0]["mcsIndex"].as<uint32_t>();
        // targetCodeRate is uint16_t and qamModOrder uint8_t but the FAPI TV stores all as uint32_t
        tv_data->tbpars.targetCodeRate = hdf5dset[0]["targetCodeRate"].as<uint32_t>();
        tv_data->tbpars.qamModOrder    = hdf5dset[0]["qamModOrder"].as<uint32_t>();

        tv_data->tbpars.startPrb = hdf5dset[0]["rbStart"].as<uint32_t>();
        tv_data->tbpars.numPrb   = hdf5dset[0]["rbSize"].as<uint32_t>();

        tv_data->tbpars.startSym = hdf5dset[0]["StartSymbolIndex"].as<uint32_t>();
        tv_data->tbpars.numSym   = hdf5dset[0]["NrOfSymbols"].as<uint32_t>();

        tv_data->tbpars.dmrsType = hdf5dset[0]["dmrsConfigType"].as<uint32_t>();

        //"dmrsConfigType"
        tv_data->tbpars.dmrsScramId = hdf5dset[0]["DmrsScramblingId"].as<uint32_t>();

        // TB data of a PDU
        tv_data->tb_size              = hdf5dset[0]["TBSize"].as<uint32_t>();
        tv_data->tb_buf               = pdsch_tv.data_buf.data() + tb_offset;
        tv_data->numDmrsCdmGrpsNoData = hdf5dset[0]["numDmrsCdmGrpsNoData"].as<uint32_t>();
        tv_data->dmrsSymLocBmsk       = hdf5dset[0]["DmrsSymbPos"].as<uint32_t>();
        tv_data->tbpars.dmrsMaxLength = ((tv_data->dmrsSymLocBmsk & (tv_data->dmrsSymLocBmsk >> 1)) > 0) ? 2 : 1;

        tv_data->powerControlOffset  = hdf5dset[0]["powerControlOffset"].as<uint32_t>();
        tv_data->powerControlOffsetSS  = hdf5dset[0]["powerControlOffsetSS"].as<uint32_t>();

        tv_data->VRBtoPRBMapping    = hdf5dset[0]["VRBtoPRBMapping"].as<uint32_t>();
        tv_data->transmissionScheme = hdf5dset[0]["transmissionScheme"].as<uint32_t>();

        try
        {
            tv_data->testModel  = hdf5dset[0]["testModel"].as<uint32_t>();
        }
        catch(...)
        {
            tv_data->testModel  = 0; // test model disabled
        }

        int Ld         = tv_data->tbpars.numSym + tv_data->tbpars.startSym;
        int bit1_count = __builtin_popcount(tv_data->dmrsSymLocBmsk) / tv_data->tbpars.dmrsMaxLength;
        switch(tv_data->tbpars.dmrsType)
        {
        case 0: // Type A
            tv_data->tbpars.dmrsAddlPosition = bit1_count - 1;
            break;

        default:
            NVLOGE_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "Unsupported dmrsType {}", tv_data->tbpars.dmrsType);
            exit(1);
        }

        parse_tx_beamforming(hdf5dset, tv_data->tx_beam_data);

        // Read PDSCH TB data from TV
        hdf5hpp::hdf5_dataset payload = file.open_dataset(dsPayload.c_str());
        if (payload.get_buffer_size_bytes() != tv_data->tb_size) {
            NVLOGF_FMT(TAG, AERIAL_CONFIG_EVENT, "{}: size doesn't match: payload_size={} tb_size={}", __func__, payload.get_buffer_size_bytes(), tv_data->tb_size);
        }
        payload.read(tv_data->tb_buf);

        // Add padding bytes
        uint32_t padding_nbytes =  (~tv_data->tb_size + 1) & (yaml_configs->pdsch_align_bytes -1);
        if (padding_nbytes > 0) {
            memset(tv_data->tb_buf + tv_data->tb_size, 0, padding_nbytes);
        }
        NVLOGD_FMT(TAG, "{}: payload_size={} tb_size={} padding_nbytes={} tb_offset={}",
                __func__, payload.get_buffer_size_bytes(), tv_data->tb_size, padding_nbytes, tb_offset);

        pdsch_tv.data.push_back(tv_data);
        tb_offset += tv_data->tb_size + padding_nbytes;
    }

    if (pdsch_tv.data_size != tb_offset) {
        NVLOGF_FMT(TAG, AERIAL_CONFIG_EVENT, "{}: PDSCH TB size doesn't match: pdsch_tv.data_size={} tb_offset={}", __func__, pdsch_tv.data_size, tb_offset);
    } else {
        NVLOGI_FMT(TAG, "{}: PDSCH data_size={} nPDU={}", __FUNCTION__, pdsch_tv.data_size, pdsch_tv.data.size());
    }

#if 0
    for (auto& tv_data : pdsch_tv.data)
    {
	    uint8_t* tb = tv_data->tb_buf;
        if (tb != nullptr) {
            for (uint i = 0; i < 112; i+=16) {
              NVLOGC_FMT(TAG, "Data Buffer[{}] = 0x{:2x} 0x{:2x} 0x{:2x} 0x{:2x} 0x{:2x} 0x{:2x} 0x{:2x} 0x{:2x} 0x{:2x} 0x{:2x} 0x{:2x} 0x{:2x} 0x{:2x} 0x{:2x} 0x{:2x} 0x{:2x}", i, tb[i], tb[i+1], tb[i+2], tb[i+3],
               tb[i+4], tb[i+5], tb[i+ 6], tb[i+ 7],
               tb[i+8], tb[i+9], tb[i+10], tb[i+11],
               tb[i + 12], tb[i+13], tb[i+14], tb[i+15]);
            }
        }
    }
#endif



    return 0;
}

int launch_pattern::parse_tv_pucch(hdf5hpp::hdf5_file& file, fapi_req_t* req)
{
    pucch_tv_t&   pucch_tv = req->tv_data->pucch_tv;
    tv_dset_list& list     = req->tv_data->dset_tv;

    if(list.find(channel_type_t::PUCCH) == list.end())
    {
        return -1;
    }

    auto& dsets = list[channel_type_t::PUCCH];
    for(auto& dset : dsets)
    {
        std::string payloadDsetStr = dset + "_payload";

        if(!file.is_valid_dataset(dset.c_str()) || !file.is_valid_dataset(payloadDsetStr.c_str()))
        {
            continue;
        }
        hdf5hpp::hdf5_dataset payload     = file.open_dataset(payloadDsetStr.c_str());
        size_t                payloadSize = payload.get_buffer_size_bytes();
        if(!payloadSize)
        {
            continue;
        }
        pucch_tv.data.emplace_back();
        auto& pucch = pucch_tv.data.back();
        pucch.Payload.resize(payloadSize);
        payload.read(pucch.Payload.data());

        hdf5hpp::hdf5_dataset hdf5dset = file.open_dataset(dset.c_str());
        pucch.type                     = hdf5dset[0]["type"].as<uint32_t>();
        pucch.RNTI                     = hdf5dset[0]["RNTI"].as<uint32_t>();
        pucch.BWPSize                  = hdf5dset[0]["BWPSize"].as<uint32_t>();
        pucch.BWPStart                 = hdf5dset[0]["BWPStart"].as<uint32_t>();
        pucch.SubcarrierSpacing        = hdf5dset[0]["SubcarrierSpacing"].as<uint32_t>();
        pucch.CyclicPrefix             = hdf5dset[0]["CyclicPrefix"].as<uint32_t>();
        pucch.FormatType               = hdf5dset[0]["FormatType"].as<uint32_t>();
        pucch.multiSlotTxIndicator     = hdf5dset[0]["multiSlotTxIndicator"].as<uint32_t>();
        pucch.pi2Bpsk                  = hdf5dset[0]["pi2Bpsk"].as<uint32_t>();
        pucch.prbStart                 = hdf5dset[0]["prbStart"].as<uint32_t>();
        pucch.prbSize                  = hdf5dset[0]["prbSize"].as<uint32_t>();
        pucch.StartSymbolIndex         = hdf5dset[0]["StartSymbolIndex"].as<uint32_t>();
        pucch.NrOfSymbols              = hdf5dset[0]["NrOfSymbols"].as<uint32_t>();
        pucch.freqHopFlag              = hdf5dset[0]["freqHopFlag"].as<uint32_t>();
        pucch.secondHopPRB             = hdf5dset[0]["secondHopPRB"].as<uint32_t>();
        pucch.groupHopFlag             = hdf5dset[0]["groupHopFlag"].as<uint32_t>();
        pucch.sequenceHopFlag          = hdf5dset[0]["sequenceHopFlag"].as<uint32_t>();
        pucch.hoppingId                = hdf5dset[0]["hoppingId"].as<uint32_t>();
        pucch.InitialCyclicShift       = hdf5dset[0]["InitialCyclicShift"].as<uint32_t>();
        pucch.dataScramblingId         = hdf5dset[0]["dataScramblingId"].as<uint32_t>();
        pucch.TimeDomainOccIdx         = hdf5dset[0]["TimeDomainOccIdx"].as<uint32_t>();
        pucch.PreDftOccIdx             = hdf5dset[0]["PreDftOccIdx"].as<uint32_t>();
        pucch.PreDftOccLen             = hdf5dset[0]["PreDftOccLen"].as<uint32_t>();
        pucch.AddDmrsFlag              = hdf5dset[0]["AddDmrsFlag"].as<uint32_t>();
        pucch.DmrsScramblingId         = hdf5dset[0]["DmrsScramblingId"].as<uint32_t>();
        pucch.DMRScyclicshift          = hdf5dset[0]["DMRScyclicshift"].as<uint32_t>();
        pucch.BitLenHarq               = hdf5dset[0]["BitLenHarq"].as<uint32_t>();
        pucch.pucchPduIdx              = hdf5dset[0]["pucchPduIdx"].as<uint32_t>();
        pucch.RSRP                     = hdf5dset[0]["RSRP"].as<float>();
        pucch.snrdB                    = hdf5dset[0]["snrdB"].as<float>();

        // Mirroring cuPHY testbench, SNR estimates are less accurate above 20dB RSRP
        if (pucch.FormatType == 1 && pucch.RSRP > 20)
        {
            vald_tolerance_t& tolerance = get_cell_configs(req->cell_idx).tolerance;
            tolerance.ul_meas[UCI_PDU_TYPE_PF01].SNR = VALD_TOLERANCE_SINR_PF1_20DB;
        }

        switch(pucch.FormatType)
        {
        case 0:
        case 1:
            pucch.nBits          = h5dset_try_parse<uint32_t>(hdf5dset, "nBits", 0);
            pucch.SRFlag         = h5dset_try_parse<uint32_t>(hdf5dset, "SRFlag", 0);
            pucch.SRindication   = h5dset_try_parse<uint32_t>(hdf5dset, "SRindication", 0);
            pucch.maxCodeRate    = 0;
            pucch.BitLenCsiPart1 = 0;
            pucch.BitLenCsiPart2 = 0;
            if (pucch.snrdB <= -10)
            {
                vald_tolerance_t& tolerance = get_cell_configs(req->cell_idx).tolerance;
                tolerance.ul_meas[UCI_PDU_TYPE_PF01].TimingAdvance     = VALD_TOLERANCE_TIMING_ADVANCE_LOW_SNR;
                tolerance.ul_meas[UCI_PDU_TYPE_PF01].TimingAdvanceNs   = VALD_TOLERANCE_TIMING_ADVANCE_NS_LOW_SNR;
            }
            if ((pucch.RSRP < -40) && (get_cell_configs(req->cell_idx).BFP == 9))
            {
                vald_tolerance_t& tolerance = get_cell_configs(req->cell_idx).tolerance;
                tolerance.ul_meas[UCI_PDU_TYPE_PF01].RSRP = VALD_TOLERANCE_RSRP_LOW;
            }
	        //Pending logic to get expected HARQ payload.
            break;
        case 2:
        case 3:
        case 4:
            pucch.nBits          = 0;
            pucch.SRFlag         = 0;
            pucch.SRindication   = 0;
            pucch.bitLenSr       = hdf5dset[0]["BitLenSr"].as<uint32_t>();
            pucch.maxCodeRate    = hdf5dset[0]["maxCodeRate"].as<uint32_t>();
            pucch.BitLenCsiPart1 = hdf5dset[0]["BitLenCsiPart1"].as<uint32_t>();
            pucch.BitLenCsiPart2 = h5dset_try_parse<uint32_t>(hdf5dset[0], "BitLenCsiPart2", 0, false);
            if (pucch.snrdB <= -10)
            {
                vald_tolerance_t& tolerance = get_cell_configs(req->cell_idx).tolerance;
                tolerance.ul_meas[UCI_PDU_TYPE_PF234].TimingAdvance    = VALD_TOLERANCE_TIMING_ADVANCE_LOW_SNR;
                tolerance.ul_meas[UCI_PDU_TYPE_PF234].TimingAdvanceNs  = VALD_TOLERANCE_TIMING_ADVANCE_NS_LOW_SNR;
            }
            break;
        default:
            break;
        }

        parse_rx_beamforming(hdf5dset, pucch.rx_beam_data);

        // Indication validation
        uint32_t nInd = hdf5dset[0]["nInd"].as<uint32_t>();
        if (nInd != 1)
        {
            NVLOGE_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "Not supported indication count: nInd={}", nInd);
        }

        uint32_t idxInd = hdf5dset[0]["idxInd"].as<uint32_t>();
        std::string ind_name = H5_IND + std::to_string(idxInd);
        hdf5hpp::hdf5_dataset ind = file.open_dataset(ind_name.c_str());
        pucch_uci_ind_t& pucch_ind = pucch.uci_ind;
        // Common fields for all PUCCH format 0~5
        pucch_ind.idxInd = idxInd;
        pucch_ind.idxPdu = ind[0]["idxPdu"].as<uint32_t>();
        pucch_ind.PucchFormat = ind[0]["PucchFormat"].as<uint32_t>();
        pucch_ind.meas.UL_CQI = ind[0]["UL_CQI"].as<uint32_t>();
        pucch_ind.meas.TimingAdvance = h5dset_try_parse<uint32_t>(ind, "TimingAdvance", 0);
        pucch_ind.meas.TimingAdvanceNs = h5dset_try_parse<int16_t>(ind, "TimingAdvanceNano", 0);
        pucch_ind.meas.SNR = h5dset_try_parse<int16_t>(ind, "SNR", 0);
        pucch_ind.meas.RSRP = h5dset_try_parse<uint32_t>(ind, "RSRP", 0);
        pucch_ind.meas.RSSI = ind[0]["RSSI"].as<uint32_t>();

        switch(pucch.FormatType)
        {
        case 0:
        case 1:
            // PUCCH format 0~1 fields
            pucch_ind.noiseVardB = 0;
            pucch_ind.SRindication = ind[0]["SRindication"].as<uint32_t>();
            pucch_ind.SRconfidenceLevel = ind[0]["SRconfidenceLevel"].as<uint32_t>();
            pucch_ind.NumHarq = ind[0]["NumHarq"].as<uint32_t>();
            pucch_ind.HarqconfidenceLevel = ind[0]["HarqconfidenceLevel"].as<uint32_t>();
            if (pucch_ind.NumHarq > 0)
            {
                // Read from uint32_t array dataset and store to uint8_t vector
                pucch_ind.HarqValue.resize(pucch_ind.NumHarq);
#ifdef SCF_FAPI_10_04
                h5dset_try_read_convert<uint32_t, uint8_t>(file, std::string(ind_name + "_HarqValue").c_str(), pucch_ind.HarqValue.data(), pucch_ind.NumHarq);
#else
                h5dset_try_read_convert<uint32_t, uint8_t>(file, std::string(ind_name + "_HarqValueFapi1002").c_str(), pucch_ind.HarqValue.data(), pucch_ind.NumHarq);
#endif
            }
            break;
        case 2:
        case 3:
        case 4:
            // PUCCH format 2~4 fields
            pucch_ind.noiseVardB = ind[0]["noiseVardB"].as<int16_t>();;
            pucch_ind.SrBitLen = ind[0]["SrBitLen"].as<uint32_t>();
            pucch_ind.HarqCrc = ind[0]["HarqCrc"].as<uint32_t>();
            pucch_ind.HarqBitLen = ind[0]["BitLenHarq"].as<uint32_t>();
            pucch_ind.HarqDetectionStatus = ind[0]["HarqDetectionStatus"].as<uint32_t>();
            // CSI Part 1, 2 may not exist
            pucch_ind.csi_parts[0].Crc = h5dset_try_parse<uint32_t>(ind, "CsiPart1Crc", 0);
            pucch_ind.csi_parts[0].BitLen = h5dset_try_parse<uint32_t>(ind, "CsiPart1BitLen", 0);
            pucch_ind.csi_parts[0].DetectionStatus = h5dset_try_parse<uint32_t>(ind, "CsiPart1DetectionStatus", 0);
            pucch_ind.csi_parts[1].Crc = h5dset_try_parse<uint32_t>(ind, "CsiPart2Crc", 0, false);
            pucch_ind.csi_parts[1].BitLen = h5dset_try_parse<uint32_t>(ind, "CsiPart2BitLen", 0, false);
            pucch_ind.csi_parts[1].DetectionStatus = h5dset_try_parse<uint32_t>(ind, "CsiPart2DetectionStatus", 0);

            // Read SR payload for PUCCH
            if (pucch_ind.SrBitLen > 0)
            {
                h5dset_try_read_u32_to_bits(file, std::string(ind_name + "_SrPayload").c_str(), pucch_ind.SrPayload, pucch_ind.SrBitLen);
            }

            // Read HARQ payload for PUCCH
            if (pucch_ind.HarqBitLen > 0)
            {
                h5dset_try_read_u32_to_bits(file, std::string(ind_name + "_HarqPayload").c_str(), pucch_ind.HarqPayload, pucch_ind.HarqBitLen);
            }

            // Read CSI Part 1 payload for PUCCH
            if (pucch_ind.csi_parts[0].BitLen)
            {
                h5dset_try_read_u32_to_bits(file, std::string(ind_name + "_CsiPart1Payload").c_str(), pucch_ind.csi_parts[0].Payload, pucch_ind.csi_parts[0].BitLen);
            }

            // Read CSI Part 2 payload for PUCCH
            if (pucch_ind.csi_parts[1].BitLen)
            {
                h5dset_try_read_u32_to_bits(file, std::string(ind_name + "_CsiPart2Payload").c_str(), pucch_ind.csi_parts[1].Payload, pucch_ind.csi_parts[1].BitLen);
            }
            break;
        default:
            break;
        }
    }
    return 0;
}

int launch_pattern::parse_tv_pusch(hdf5hpp::hdf5_file& file, fapi_req_t* req)
{
    pusch_tv_t&   pusch_tv = req->tv_data->pusch_tv;
    std::size_t   tb_offset = 0;

    pusch_tv.data_size = 0;
    std::vector<int> list = get_pdu_index_list(file, channel_type_t::PUSCH);
    for (int i = 0; i < list.size(); i++)
    {
        int pdu_id = list[i];
        std::string dset = H5_PDU + std::to_string(pdu_id);
        std::string dsPayload = dset + "_payload";
        if(!file.is_valid_dataset(dset.c_str()) || !file.is_valid_dataset(dsPayload.c_str()))
        {
            continue;
        }

        hdf5hpp::hdf5_dataset payload      = file.open_dataset(dsPayload.c_str());
        size_t                payload_size = payload.get_buffer_size_bytes();
        pusch_tv.data_size += payload_size;
    }
    pusch_tv.data_buf.resize(pusch_tv.data_size);


    uint32_t puschBFP = 14;
    if(file.is_valid_dataset(H5_BFP))
    {
        hdf5hpp::hdf5_dataset dset = file.open_dataset(H5_BFP);
        dset.read(&puschBFP);
        NVLOGI_FMT(TAG, "{}: PUSCH BFP = {}", __FUNCTION__, puschBFP);
    }

    for (int i = 0; i < list.size(); i++)
    {
        int pdu_id = list[i];
        std::string dset = H5_PDU + std::to_string(pdu_id);
        std::string dsPayload = dset + "_payload";

        if(!file.is_valid_dataset(dset.c_str()) || !file.is_valid_dataset(dsPayload.c_str()))
        {
            continue;
        }

        hdf5hpp::hdf5_dataset payload = file.open_dataset(dsPayload.c_str());
        payload.read(pusch_tv.data_buf.data() + tb_offset);

        hdf5hpp::hdf5_dataset hdf5dset = file.open_dataset(dset.c_str());

        pusch_tv_data_t* tv_data = new pusch_tv_data_t;

        tv_data->pduBitmap         = hdf5dset[0]["pduBitmap"].as<uint32_t>();
        tv_data->BWPStart          = hdf5dset[0]["BWPStart"].as<uint32_t>();
        tv_data->BWPSize           = hdf5dset[0]["BWPSize"].as<uint32_t>();
        tv_data->pduBitmap         = hdf5dset[0]["pduBitmap"].as<uint32_t>();
        tv_data->harqAckBitLength  = hdf5dset[0]["harqAckBitLength"].as<uint32_t>();
        tv_data->csiPart1BitLength = hdf5dset[0]["csiPart1BitLength"].as<uint32_t>();
#ifdef SCF_FAPI_10_04
        tv_data->flagCsiPart2 = h5dset_try_parse<uint16_t>(hdf5dset, "flagCsiPart2", 0, 0);
#else
        tv_data->csiPart2BitLength = h5dset_try_parse<uint32_t>(hdf5dset, "csiPart2BitLength", 0);
#endif
        tv_data->alphaScaling      = hdf5dset[0]["alphaScaling"].as<uint32_t>();
        tv_data->betaOffsetHarqAck = hdf5dset[0]["betaOffsetHarqAck"].as<uint32_t>();
        tv_data->betaOffsetCsi1    = hdf5dset[0]["betaOffsetCsi1"].as<uint32_t>();
        tv_data->betaOffsetCsi2    = h5dset_try_parse<uint32_t>(hdf5dset, "betaOffsetCsi2", 0);
        tv_data->qamModOrder       = hdf5dset[0]["qamModOrder"].as<uint32_t>();
        tv_data->tbErr             = h5dset_try_parse<uint32_t>(hdf5dset, "tbErr", 0);
        tv_data->puschIdentity     = hdf5dset[0]["puschIdentity"].as<uint16_t>();
        tv_data->BFP               = BFP16;
        if(puschBFP==14)
        {
            tv_data->BFP = BFP14;
        }
        else if(puschBFP==9)
        {
            tv_data->BFP = BFP9;
        }

        //DFT-s-OFDM
        tv_data->TransformPrecoding      = h5dset_try_parse<uint8_t>(hdf5dset, "TransformPrecoding", 1);
        if(tv_data->TransformPrecoding==0)
        {
            if(tv_data->pduBitmap & PUSCH_BITMAP_DFTSOFDM)
            {
                tv_data->lowPaprGroupNumber = hdf5dset[0]["lowPaprGroupNumber"].as<uint8_t>();
                tv_data->lowPaprSequenceNumber = hdf5dset[0]["lowPaprSequenceNumber"].as<uint16_t>();
            }
            else
            {
            #ifdef SCF_FAPI_10_04
                tv_data->groupOrSequenceHopping = hdf5dset[0]["groupOrSequenceHopping"].as<uint8_t>();
            #endif
            }
        }
        
        tv_data->numPart2s = 0;

#ifdef SCF_FAPI_10_04
        if (tv_data->pduBitmap & PUSCH_BITMAP_UCI) {
           tv_data->numPart2s = h5dset_try_parse<uint16_t>(hdf5dset, "nCsi2Reports", 0);
            auto paramOffsets = h5dset_try_parse<std::vector<uint16_t>>(hdf5dset, "calcCsi2Size_prmOffsets", {});
            auto paramSizes = h5dset_try_parse<std::vector<uint8_t>>(hdf5dset, "calcCsi2Size_prmSizes", {});
            auto part2SizeMapIndex = h5dset_try_parse<std::vector<uint16_t>>(hdf5dset, "calcCsi2Size_csi2MapIdx", {});
            auto numPart1Parms = h5dset_try_parse<std::vector<uint8_t>>(hdf5dset, "calcCsi2Size_nPart1Prms", {});
            for (uint16_t i = 0; i < tv_data->numPart2s; i++) {
                tv_data->csip2_v3_parts.push_back(csi_part2_info_t());
                auto& csi_part = tv_data->csip2_v3_parts.back();
                csi_part.priority = 1;
                csi_part.numPart1Params = numPart1Parms[i];
                for (uint16_t j = 0; j < csi_part.numPart1Params; j++ ) {
                    csi_part.paramOffsets.push_back(paramOffsets[i * CUPHY_MAX_N_CSI1_PRMS + j]);
                    csi_part.paramSizes.push_back(paramSizes[i * CUPHY_MAX_N_CSI1_PRMS +j]);
                    NVLOGD_FMT(TAG, "paramOffsets [{}] paramSizes [{}]", paramOffsets[i+j], paramSizes[i+j]);
                }
                csi_part.part2SizeMapIndex = part2SizeMapIndex[i];
                csi_part.part2SizeMapScope = 1;
            }
            paramSizes.clear();
            paramOffsets.clear();
            part2SizeMapIndex.clear();
            numPart1Parms.clear();
        }
        // Weighted average CFO estimation
        cell_configs_t& cell_configs = get_cell_configs(req->cell_idx);
        if (cell_configs.enableWeightedAverageCfo == 1){
            tv_data->foForgetCoeff = h5dset_try_parse<float>(hdf5dset, "foForgetCoeff", 0.0f);
            tv_data->nIterations = h5dset_try_parse<uint8_t>(hdf5dset, "ldpcMaxNumItrPerUe", 10);
            tv_data->ldpcEarlyTermination = h5dset_try_parse<uint8_t>(hdf5dset, "ldpcEarlyTerminationPerUe", 0);
        } else { // Disabled
            tv_data->foForgetCoeff = 0.0f;
            tv_data->nIterations = 10;
            tv_data->ldpcEarlyTermination = 0;
        }
#endif

        // HARQ
        tv_data->harqProcessID    = hdf5dset[0]["harqProcessID"].as<uint32_t>();
        tv_data->newDataIndicator = hdf5dset[0]["newDataIndicator"].as<uint32_t>();


        // PUSCH SCF pars
        tv_data->SubcarrierSpacing          = hdf5dset[0]["SubcarrierSpacing"].as<uint32_t>();
        tv_data->CyclicPrefix               = hdf5dset[0]["CyclicPrefix"].as<uint32_t>();
        tv_data->targetCodeRate             = hdf5dset[0]["targetCodeRate"].as<uint32_t>();
        tv_data->FrequencyHopping           = hdf5dset[0]["FrequencyHopping"].as<uint32_t>();
        tv_data->uplinkFrequencyShift7p5khz = hdf5dset[0]["uplinkFrequencyShift7p5khz"].as<uint32_t>();
        tv_data->txDirectCurrentLocation    = hdf5dset[0]["txDirectCurrentLocation"].as<uint32_t>();

        // ue_pars
        tv_data->tbpars.nRnti       = hdf5dset[0]["RNTI"].as<uint32_t>();
        tv_data->tbpars.numLayers   = hdf5dset[0]["nrOfLayers"].as<uint32_t>();
        tv_data->tbpars.dataScramId = hdf5dset[0]["dataScramblingId"].as<uint32_t>(); // Cell ID 41
        tv_data->tbpars.nSCID       = hdf5dset[0]["SCID"].as<uint32_t>();
        tv_data->tbpars.nPortIndex  = hdf5dset[0]["dmrsPorts"].as<uint32_t>();

        // cw_pars
        tv_data->tbpars.rv            = hdf5dset[0]["rvIndex"].as<uint32_t>();
        tv_data->tbpars.mcsTableIndex = hdf5dset[0]["mcsTable"].as<uint32_t>();
        tv_data->tbpars.mcsIndex      = hdf5dset[0]["mcsIndex"].as<uint32_t>();

        tv_data->tbpars.startPrb = hdf5dset[0]["rbStart"].as<uint32_t>();
        tv_data->tbpars.numPrb   = hdf5dset[0]["rbSize"].as<uint32_t>();

        tv_data->tbpars.startSym = hdf5dset[0]["StartSymbolIndex"].as<uint32_t>();
        tv_data->tbpars.numSym   = hdf5dset[0]["NrOfSymbols"].as<uint32_t>();

        tv_data->tbpars.dmrsType      = hdf5dset[0]["dmrsConfigType"].as<uint32_t>();
        tv_data->tbpars.dmrsScramId   = hdf5dset[0]["DmrsScramblingId"].as<uint32_t>();
        tv_data->dmrsSymLocBmsk       = hdf5dset[0]["DmrsSymbPos"].as<uint32_t>();
        tv_data->tbpars.dmrsMaxLength = ((tv_data->dmrsSymLocBmsk & (tv_data->dmrsSymLocBmsk >> 1)) > 0) ? 2 : 1;
        tv_data->tbpars.qamModOrder   = hdf5dset[0]["qamModOrder"].as<uint32_t>();

        tv_data->numDmrsCdmGrpsNoData = hdf5dset[0]["numDmrsCdmGrpsNoData"].as<uint32_t>();

        // dmrsAddlPosition
        int Ld         = tv_data->tbpars.numSym + tv_data->tbpars.startSym;
        int bit1_count = __builtin_popcount(tv_data->dmrsSymLocBmsk) / tv_data->tbpars.dmrsMaxLength;
        switch(tv_data->tbpars.dmrsType)
        {
        case 0: // Type A
            tv_data->tbpars.dmrsAddlPosition = bit1_count - 1;
            break;

        default:
            NVLOGE_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "Unsupported dmrsType {}", tv_data->tbpars.dmrsType);
            exit(1);
        }

        // IND validation
        uint32_t nInd = hdf5dset[0]["nInd"].as<uint32_t>();
        uint32_t idxInd = hdf5dset[0]["idxInd"].as<uint32_t>();
        for (uint32_t n = 0; n < nInd; n++)
        {
            std::string ind_name = H5_IND + std::to_string(idxInd + n);
            hdf5hpp::hdf5_dataset ind = file.open_dataset(ind_name.c_str());
            uint32_t ind_type = ind[0]["type"].as<uint32_t>();
            if (ind_type == IND_PUSCH_DATA)
            {
                pusch_data_ind_t& data_ind = tv_data->data_ind;
                data_ind.idxInd = idxInd + n;
                data_ind.idxPdu = ind[0]["idxPdu"].as<uint32_t>();
                data_ind.TbCrcStatus = ind[0]["TbCrcStatus"].as<uint32_t>();
                data_ind.NumCb = ind[0]["NumCb"].as<uint32_t>();
                data_ind.meas.TimingAdvance = ind[0]["TimingAdvance"].as<uint32_t>();
                data_ind.meas.TimingAdvanceNs = h5dset_try_parse<int16_t>(ind, "TimingAdvanceNano", 0);
                // data_ind.meas.SNR = h5dset_try_parse<int16_t>(ind, "sinrdB", 0);
                data_ind.meas.RSSI = ind[0]["RSSI"].as<uint32_t>();
                data_ind.meas.RSSI_ehq = ind[0]["RSSI_ehq"].as<uint32_t>();
                data_ind.meas.RSRP = ind[0]["RSRP"].as<uint32_t>();
                data_ind.meas.RSRP_ehq = ind[0]["RSRP_ehq"].as<uint32_t>();

                data_ind.sinrdB = ind[0]["sinrdB"].as<int16_t>();
                data_ind.postEqSinrdB = ind[0]["postEqSinrdB"].as<int16_t>();
                data_ind.noiseVardB = ind[0]["noiseVardB"].as<int16_t>();
                data_ind.postEqNoiseVardB = ind[0]["postEqNoiseVardB"].as<int16_t>();

                // For validating SNR in PUSCH ul_measurement_t
                cell_configs_t& cell_configs = get_cell_configs(req->cell_idx);
                if (cell_configs.pusch_sinr_selector == 1) { // Post EQ
                    data_ind.meas.SNR = data_ind.postEqSinrdB;
                    data_ind.meas.UL_CQI = ind[0]["UL_CQI_POSTEQ"].as<uint32_t>();
                } else if (cell_configs.pusch_sinr_selector == 2) {  // PreEQ
                    data_ind.meas.SNR = data_ind.sinrdB;
                    data_ind.meas.UL_CQI = ind[0]["UL_CQI"].as<uint32_t>();
                } else { // Disabled
                    // Skip validating
                }

                // Read CbCrcStatus payload for PUSCH
                if (data_ind.NumCb > 0)
                {
                    int nbytes = (data_ind.NumCb + 7) / 8;
                    data_ind.CbCrcStatus.resize(nbytes);
                    h5dset_try_read(file, std::string(ind_name + "_CbCrcStatus").c_str(), data_ind.CbCrcStatus.data(), nbytes);
                }
            }
            else if (ind_type == IND_PUSCH_UCI)
            {
                pusch_uci_ind_t& uci_ind = tv_data->uci_ind;
                uci_ind.idxInd = idxInd + n;
                uci_ind.idxPdu = ind[0]["idxPdu"].as<uint32_t>();
                uci_ind.meas.TimingAdvance = ind[0]["TimingAdvance"].as<uint32_t>();
                uci_ind.meas.TimingAdvanceNs = h5dset_try_parse<int16_t>(ind, "TimingAdvanceNano", 0);
                uci_ind.meas.RSSI = ind[0]["RSSI"].as<uint32_t>();
                uci_ind.meas.RSSI_ehq = ind[0]["RSSI_ehq"].as<uint32_t>();
                uci_ind.meas.RSRP = ind[0]["RSRP"].as<uint32_t>();
                uci_ind.meas.RSRP_ehq = ind[0]["RSRP_ehq"].as<uint32_t>();
                uci_ind.HarqCrc = ind[0]["HarqCrc"].as<uint32_t>();
                uci_ind.HarqBitLen = ind[0]["HarqBitLen"].as<uint32_t>();
                uci_ind.csi_parts[0].Crc = ind[0]["CsiPart1Crc"].as<uint32_t>();
                uci_ind.csi_parts[0].BitLen = ind[0]["CsiPart1BitLen"].as<uint32_t>();
                uci_ind.csi_parts[1].Crc = ind[0]["CsiPart2Crc"].as<uint32_t>();
                uci_ind.csi_parts[1].BitLen = ind[0]["CsiPart2BitLen"].as<uint32_t>();

                // Get detection status values from PDUx
                uci_ind.HarqDetectionStatus = h5dset_try_parse<uint32_t>(hdf5dset, "harqDetStatus", 0);
                uci_ind.HarqDetStatus_earlyHarq = h5dset_try_parse<uint32_t>(hdf5dset, "harqDetStatus_earlyHarq", 0);
                uci_ind.isEarlyHarq = h5dset_try_parse<uint8_t>(hdf5dset, "isEarlyHarq", 0);
                uci_ind.csi_parts[0].DetectionStatus = h5dset_try_parse<uint32_t>(hdf5dset, "csi1DetStatus", 0);
                uci_ind.csi_parts[1].DetectionStatus = h5dset_try_parse<uint32_t>(hdf5dset, "csi2DetStatus", 0);

                uci_ind.sinrdB = ind[0]["sinrdB"].as<int16_t>();
                uci_ind.postEqSinrdB = ind[0]["postEqSinrdB"].as<int16_t>();
                uci_ind.noiseVardB = ind[0]["noiseVardB"].as<int16_t>();
                uci_ind.postEqNoiseVardB = ind[0]["postEqNoiseVardB"].as<int16_t>();

#ifdef SCF_FAPI_10_04
                if (uci_ind.isEarlyHarq) {
                    uci_ind.sinrdB_ehq = ind[0]["sinrdB_ehq"].as<int16_t>();
                } else {
                    uci_ind.sinrdB_ehq = std::numeric_limits<int16_t>::max();
                }
#endif
                // For validating SNR in PUSCH ul_measurement_t
                cell_configs_t& cell_configs = get_cell_configs(req->cell_idx);
                if (cell_configs.pusch_sinr_selector == 1) { // Post EQ
                    uci_ind.meas.SNR = uci_ind.postEqSinrdB;
                    uci_ind.meas.UL_CQI = ind[0]["UL_CQI_POSTEQ"].as<uint32_t>();
#ifdef SCF_FAPI_10_04
                    uci_ind.meas.SNR_ehq = std::numeric_limits<int16_t>::max();
#endif
                } else if (cell_configs.pusch_sinr_selector == 2) {  // PreEQ
                    uci_ind.meas.SNR = uci_ind.sinrdB;
                    uci_ind.meas.UL_CQI = ind[0]["UL_CQI"].as<uint32_t>();
#ifdef SCF_FAPI_10_04
                    uci_ind.meas.SNR_ehq = uci_ind.sinrdB_ehq;
#endif
                } else { // Disabled
                    // Skip validating
                }

                // Read HARQ payload for UCI on PUSCH
                if (uci_ind.HarqBitLen > 0)
                {
                    int nbytes = (uci_ind.HarqBitLen + 7) / 8;
                    uci_ind.HarqPayload.resize(nbytes);
                    h5dset_try_read(file, std::string(ind_name + "_HarqPayload").c_str(), uci_ind.HarqPayload.data(), nbytes);
                    if(uci_ind.isEarlyHarq)
                    {
                        uci_ind.HarqPayload_earlyHarq.resize(nbytes);
                        h5dset_try_read(file, std::string(ind_name + "_HarqPayload_earlyHarq").c_str(), uci_ind.HarqPayload_earlyHarq.data(), nbytes);
                    }
                }

                // Read CSI Part 1 payload for UCI on PUSCH
                if (uci_ind.csi_parts[0].BitLen)
                {
                    int nbytes = (uci_ind.csi_parts[0].BitLen + 7) / 8;
                    uci_ind.csi_parts[0].Payload.resize(nbytes);
                    h5dset_try_read(file, std::string(ind_name + "_CsiPart1Payload").c_str(), uci_ind.csi_parts[0].Payload.data(), nbytes);
                }

                // Read CSI Part 2 payload for UCI on PUSCH
                if (uci_ind.csi_parts[1].BitLen)
                {
                    int nbytes = (uci_ind.csi_parts[1].BitLen + 7) / 8;
                    uci_ind.csi_parts[1].Payload.resize(nbytes);
                    h5dset_try_read(file, std::string(ind_name + "_CsiPart2Payload").c_str(), uci_ind.csi_parts[1].Payload.data(), nbytes);
                }
            }
            else
            {
                NVLOGE_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "Unknown indication type {} in PUSCH TV", ind_type);
            }
        }

        parse_rx_beamforming(hdf5dset, tv_data->rx_beam_data);

        // TB data of a PDU
        tv_data->tb_size = hdf5dset[0]["TBSize"].as<uint32_t>();
        tv_data->tb_buf               = pusch_tv.data_buf.data() + tb_offset;
        pusch_tv.data.push_back(tv_data);
        tb_offset += tv_data->tb_size;
    }

    pusch_tv.negTV_enable = 0;
    if (file.is_valid_dataset(CELL_CONFIG)) {
        hdf5hpp::hdf5_dataset dset = file.open_dataset(CELL_CONFIG);
        pusch_tv.negTV_enable = dset[0]["negTV_enable"].as<uint32_t>();
        NVLOGI_FMT(TAG, "{}: PUSCH NEG TV = {}", __FUNCTION__,  pusch_tv.negTV_enable);
        if (pusch_tv.negTV_enable) {
            negative_test = pusch_tv.negTV_enable;
        }
    }
    return 0;
}

int launch_pattern::parse_tv_pbch(hdf5hpp::hdf5_file& file, fapi_req_t* req)
{
    pbch_tv_t&    pbch_tv = req->tv_data->pbch_tv;
    tv_dset_list& list    = req->tv_data->dset_tv;

    if(list.find(channel_type_t::PBCH) == list.end())
    {
        return -1;
    }
    auto& dsets = list[channel_type_t::PBCH];
    for(auto& dset : dsets)
    {
        if(!file.is_valid_dataset(dset.c_str()))
        {
            continue;
        }
        hdf5hpp::hdf5_dataset hdf5dset = file.open_dataset(dset.c_str());
        pbch_tv_data_t        pbch_pdu;
        pbch_pdu.betaPss             = hdf5dset[0]["betaPss"].as<uint32_t>();
        pbch_pdu.ssbBlockIndex       = hdf5dset[0]["ssbBlockIndex"].as<uint32_t>();
        pbch_pdu.ssbSubcarrierOffset = hdf5dset[0]["ssbSubcarrierOffset"].as<uint32_t>();
        pbch_pdu.bchPayloadFlag      = hdf5dset[0]["bchPayloadFlag"].as<uint32_t>();
        pbch_pdu.physCellId          = hdf5dset[0]["physCellId"].as<uint32_t>();
        pbch_pdu.SsbOffsetPointA     = hdf5dset[0]["SsbOffsetPointA"].as<uint32_t>();
        pbch_pdu.bchPayload          = hdf5dset[0]["bchPayload"].as<uint32_t>();
        NVLOGD_FMT(TAG, "{}: PBCH pbch_pdu.bchPayload=0x{:X}", __FUNCTION__, pbch_pdu.bchPayload);

        parse_tx_beamforming(hdf5dset, pbch_pdu.tx_beam_data);

        pbch_tv.data.push_back(std::move(pbch_pdu));
    }

    return 0;
}

int launch_pattern::parse_prach_config(hdf5hpp::hdf5_file& file, int cell_id)
{
    // PRACH configuration for CONFIG.request
    if(file.is_valid_dataset("Prach_Config"))
    {
        hdf5hpp::hdf5_dataset prach_config_dset = file.open_dataset("Prach_Config");
        prach_configs_t& prach_configs = get_prach_configs(cell_id);
        prach_configs.prachSequenceLength       = prach_config_dset[0]["prachSequenceLength"].as<uint32_t>();
        prach_configs.prachSubCSpacing          = prach_config_dset[0]["prachSubCSpacing"].as<uint32_t>();
        prach_configs.restrictedSetConfig       = prach_config_dset[0]["restrictedSetConfig"].as<uint32_t>();
        prach_configs.numPrachFdOccasions       = prach_config_dset[0]["numPrachFdOccasions"].as<uint32_t>();
        prach_configs.prachConfigIndex          = prach_config_dset[0]["prachConfigIndex"].as<uint32_t>();
        prach_configs.SsbPerRach                   = prach_config_dset[0]["SsbPerRach"].as<uint32_t>();
        prach_configs.prachMultipleCarriersInABand = prach_config_dset[0]["prachMultipleCarriersInABand"].as<uint32_t>();

        prach_configs.prachFdOccasions.clear();
        for(int i = 0; i < prach_configs.numPrachFdOccasions; i++)
        {
            std::string                dset_name = "RO_Config_" + std::to_string(i + 1);
            hdf5hpp::hdf5_dataset      cfg_dset  = file.open_dataset(dset_name.c_str());
            prach_fd_occasion_config_t fd_occasion;
            fd_occasion.prachRootSequenceIndex = cfg_dset[0]["prachRootSequenceIndex"].as<uint32_t>();
            fd_occasion.numRootSequences       = cfg_dset[0]["numRootSequences"].as<uint32_t>();
            fd_occasion.k1                     = cfg_dset[0]["k1"].as<uint32_t>();
            fd_occasion.prachZeroCorrConf      = cfg_dset[0]["prachZeroCorrConf"].as<uint32_t>();
            fd_occasion.numUnusedRootSequences = cfg_dset[0]["numUnusedRootSequences"].as<uint32_t>();
            if(fd_occasion.numUnusedRootSequences != 0)
            {
                NVLOGE_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "{}: numUnusedRootSequences={} not supported", __FUNCTION__, fd_occasion.numUnusedRootSequences);
            }
            prach_configs.prachFdOccasions.push_back(std::move(fd_occasion));
        }
    }

    return 0;
}

int launch_pattern::parse_tv_prach(hdf5hpp::hdf5_file& file, fapi_req_t* req)
{
    prach_tv_t&   prach_tv = req->tv_data->prach_tv;
    std::vector<int> list = get_pdu_index_list(file, channel_type_t::PRACH);
    for (int i = 0; i < list.size(); i++)
    {
        int pdu_id = list[i];
        std::string pdu_name = H5_PDU + std::to_string(pdu_id);

        if(file.is_valid_dataset(pdu_name.c_str()))
        {
            hdf5hpp::hdf5_dataset hdf5dset = file.open_dataset(pdu_name.c_str());

            prach_tv_data_t prach_pdu;
            prach_pdu.type               = hdf5dset[0]["type"].as<uint32_t>();
            prach_pdu.physCellID         = hdf5dset[0]["physCellID"].as<uint32_t>();
            prach_pdu.NumPrachOcas       = hdf5dset[0]["NumPrachOcas"].as<uint32_t>();
            prach_pdu.prachFormat        = hdf5dset[0]["prachFormat"].as<uint32_t>();
            prach_pdu.numRa              = hdf5dset[0]["numRa"].as<uint32_t>();
            prach_pdu.prachStartSymbol   = hdf5dset[0]["prachStartSymbol"].as<uint32_t>();
            prach_pdu.prachPduIdx        = hdf5dset[0]["prachPduIdx"].as<uint32_t>();
            prach_pdu.numCs              = hdf5dset[0]["numCs"].as<uint32_t>();

            // IND validation
            uint32_t idxInd = hdf5dset[0]["idxInd"].as<uint32_t>();
            std::string ind_name = H5_IND + std::to_string(idxInd);
            hdf5hpp::hdf5_dataset ind_dset = file.open_dataset(ind_name.c_str());
            prach_pdu.ind.idxInd = idxInd;
            prach_pdu.ind.idxPdu = ind_dset[0]["idxPdu"].as<uint32_t>();
            prach_pdu.ind.SymbolIndex = ind_dset[0]["SymbolIndex"].as<uint32_t>();
            prach_pdu.ind.SlotIndex = ind_dset[0]["SlotIndex"].as<uint32_t>();
            prach_pdu.ind.FreqIndex = ind_dset[0]["FreqIndex"].as<uint32_t>();
            prach_pdu.ind.avgRssi = ind_dset[0]["avgRssi"].as<uint32_t>();
            prach_pdu.ind.avgSnr = ind_dset[0]["avgSnr"].as<uint32_t>();
            prach_pdu.ind.avgNoise = ind_dset[0]["avgNoise"].as<uint32_t>();
            prach_pdu.ind.numPreamble = ind_dset[0]["numPreamble"].as<uint32_t>();

            prach_pdu.ind.PreamblePwr_v.resize(prach_pdu.ind.numPreamble);
            prach_pdu.ind.TimingAdvance_v.resize(prach_pdu.ind.numPreamble);
            prach_pdu.ind.TimingAdvanceNano_v.resize(prach_pdu.ind.numPreamble);
            prach_pdu.ind.preambleIndex_v.resize(prach_pdu.ind.numPreamble);

            size_t nbytes = prach_pdu.ind.numPreamble * sizeof(uint32_t);
            h5dset_try_read(file, std::string(ind_name + "_PreamblePwr").c_str(), prach_pdu.ind.PreamblePwr_v.data(), nbytes);
            h5dset_try_read(file, std::string(ind_name + "_TimingAdvance").c_str(), prach_pdu.ind.TimingAdvance_v.data(), nbytes);
            h5dset_try_read(file, std::string(ind_name + "_TimingAdvanceNano").c_str(), prach_pdu.ind.TimingAdvanceNano_v.data(), nbytes);
            h5dset_try_read(file, std::string(ind_name + "_preambleIndex").c_str(), prach_pdu.ind.preambleIndex_v.data(), nbytes);

            h5dset_try_read(file, std::string(pdu_name + "_numPrmb").c_str(), &prach_pdu.ref.numPrmb, sizeof(prach_pdu.ref.numPrmb));
            prach_pdu.ref.delay_v.resize(prach_pdu.ref.numPrmb * sizeof(float));
            prach_pdu.ref.peak_v.resize(prach_pdu.ref.numPrmb * sizeof(float));
            prach_pdu.ref.prmbIdx_v.resize(prach_pdu.ref.numPrmb * sizeof(uint32_t));
            h5dset_try_read(file, std::string(pdu_name + "_delay").c_str(), prach_pdu.ref.delay_v.data(), prach_pdu.ref.numPrmb * sizeof(float));
            h5dset_try_read(file, std::string(pdu_name + "_peak").c_str(), prach_pdu.ref.peak_v.data(), prach_pdu.ref.numPrmb * sizeof(float));
            h5dset_try_read(file, std::string(pdu_name + "_prmbIdx").c_str(), prach_pdu.ref.prmbIdx_v.data(), prach_pdu.ref.numPrmb * sizeof(uint32_t));

            parse_rx_beamforming(hdf5dset, prach_pdu.rx_beam_data);
            prach_tv.data.push_back(std::move(prach_pdu));
        }
    }
    return 0;
}

int launch_pattern::parse_tv_csirs(hdf5hpp::hdf5_file& file, fapi_req_t* req)
{
    csirs_tv_t&   csirs_tv = req->tv_data->csirs_tv;
    tv_dset_list& list     = req->tv_data->dset_tv;

    if(list.find(channel_type_t::CSI_RS) == list.end())
    {
        return -1;
    }
    auto& dsets = list[channel_type_t::CSI_RS];

    for(auto& dset : dsets)
    {
        if(file.is_valid_dataset(dset.c_str()))
        {
            hdf5hpp::hdf5_dataset hdf5dset = file.open_dataset(dset.c_str());

            csirs_tv_data_t csirs_pdu;
            csirs_pdu.type                 = hdf5dset[0]["type"].as<uint32_t>();
            csirs_pdu.BWPSize              = hdf5dset[0]["BWPSize"].as<uint32_t>();
            csirs_pdu.BWPStart             = hdf5dset[0]["BWPStart"].as<uint32_t>();
            csirs_pdu.SubcarrierSpacing    = hdf5dset[0]["SubcarrierSpacing"].as<uint32_t>();
            csirs_pdu.CyclicPrefix         = hdf5dset[0]["CyclicPrefix"].as<uint32_t>();
            csirs_pdu.StartRB              = hdf5dset[0]["StartRB"].as<uint32_t>();
            csirs_pdu.NrOfRBs              = hdf5dset[0]["NrOfRBs"].as<uint32_t>();
            csirs_pdu.CSIType              = hdf5dset[0]["CSIType"].as<uint32_t>();
            csirs_pdu.Row                  = hdf5dset[0]["Row"].as<uint32_t>();
            csirs_pdu.FreqDomain           = hdf5dset[0]["FreqDomain"].as<uint32_t>();
            csirs_pdu.SymbL0               = hdf5dset[0]["SymbL0"].as<uint32_t>();
            csirs_pdu.SymbL1               = hdf5dset[0]["SymbL1"].as<uint32_t>();
            csirs_pdu.CDMType              = hdf5dset[0]["CDMType"].as<uint32_t>();
            csirs_pdu.FreqDensity          = hdf5dset[0]["FreqDensity"].as<uint32_t>();
            csirs_pdu.ScrambId             = hdf5dset[0]["ScrambId"].as<uint32_t>();
            csirs_pdu.powerControlOffset   = hdf5dset[0]["powerControlOffset"].as<uint32_t>();
            csirs_pdu.powerControlOffsetSS = hdf5dset[0]["powerControlOffsetSS"].as<uint32_t>();
            csirs_pdu.csirsPduIdx          = hdf5dset[0]["csirsPduIdx"].as<uint32_t>();
            csirs_pdu.lastCsirsPdu         = hdf5dset[0]["lastCsirsPdu"].as<uint32_t>();
            parse_tx_beamforming(hdf5dset, csirs_pdu.tx_beam_data);
            csirs_tv.data.push_back(std::move(csirs_pdu));
        }
    }

    return 0;
}

int launch_pattern::parse_tv_srs(hdf5hpp::hdf5_file& file, fapi_req_t* req)
{
    srs_tv_t&   srs_tv = req->tv_data->srs_tv;
    tv_dset_list& list     = req->tv_data->dset_tv;

    if(list.find(channel_type_t::SRS) == list.end())
    {
        return -1;
    }
    auto& dsets = list[channel_type_t::SRS];

    for(auto& dset : dsets)
    {
        if(file.is_valid_dataset(dset.c_str()))
        {
            auto& cell_configs = get_cell_configs(req->cell_idx);
            enable_srs = 1;
            NVLOGD_FMT(TAG,"{} enable_srs = {}",__func__, enable_srs?"true":"false");
            hdf5hpp::hdf5_dataset hdf5dset = file.open_dataset(dset.c_str());
            srs_tv_data_t srs_pdu;
            srs_pdu.type = hdf5dset[0]["type"].as<uint32_t>();
            srs_pdu.RNTI = hdf5dset[0]["RNTI"].as<uint32_t>();
#ifdef SCF_FAPI_10_04
            srs_pdu.srsChestBufferIndex = hdf5dset[0]["srsChestBufferIndex"].as<uint32_t>();
#endif
            srs_pdu.BWPSize = hdf5dset[0]["BWPSize"].as<uint32_t>();
            srs_pdu.BWPStart = hdf5dset[0]["BWPStart"].as<uint32_t>();
            srs_pdu.SubcarrierSpacing = 1;//hdf5dset[0]["mu"].as<uint32_t>();
            srs_pdu.CyclicPrefix = 0; //hdf5dset[0]["CyclicPrefix"].as<uint32_t>();
            srs_pdu.numAntPorts = hdf5dset[0]["numAntPorts"].as<uint32_t>();
            srs_pdu.numSymbols = hdf5dset[0]["numSymbols"].as<uint32_t>();
            srs_pdu.numRepetitions = hdf5dset[0]["numRepetitions"].as<uint32_t>();
            srs_pdu.timeStartPosition = hdf5dset[0]["timeStartPosition"].as<uint32_t>();
            srs_pdu.configIndex = hdf5dset[0]["configIndex"].as<uint32_t>();
            srs_pdu.sequenceId = hdf5dset[0]["sequenceId"].as<uint32_t>();
            srs_pdu.bandwidthIndex = hdf5dset[0]["bandwidthIndex"].as<uint32_t>();
            srs_pdu.combSize = hdf5dset[0]["combSize"].as<uint32_t>();
            srs_pdu.combOffset = hdf5dset[0]["combOffset"].as<uint32_t>();
            srs_pdu.cyclicShift = hdf5dset[0]["cyclicShift"].as<uint32_t>();
            srs_pdu.frequencyPosition = hdf5dset[0]["frequencyPosition"].as<uint32_t>();
            srs_pdu.frequencyShift = hdf5dset[0]["frequencyShift"].as<uint32_t>();
            srs_pdu.frequencyHopping = hdf5dset[0]["frequencyHopping"].as<uint32_t>();
            srs_pdu.groupOrSequenceHopping = hdf5dset[0]["groupOrSequenceHopping"].as<uint32_t>();
            srs_pdu.resourceType = hdf5dset[0]["resourceType"].as<uint32_t>();
            srs_pdu.Tsrs = hdf5dset[0]["Tsrs"].as<uint32_t>();
            srs_pdu.Toffset = hdf5dset[0]["Toffset"].as<uint32_t>();
            srs_pdu.Beamforming = hdf5dset[0]["Beamforming"].as<uint32_t>();
            srs_pdu.numPRGs =  hdf5dset[0]["numPRGs"].as<uint32_t>();
            srs_pdu.prgSize = hdf5dset[0]["prgSize"].as<uint32_t>();
            srs_pdu.digBFInterfaces = hdf5dset[0]["digBFInterfaces"].as<uint32_t>();
            // srs_pdu.beamIdx = hdf5dset[0]["beamIdx"].as<std::vector<uint32_t>>();
            srs_pdu.srsPduIdx = hdf5dset[0]["srsPduIdx"].as<uint32_t>();
            srs_pdu.lastSrsPdu = hdf5dset[0]["lastSrsPdu"].as<uint32_t>();

            // TODO :
            // Bitmap indicating the type of report(s) expected at L2 from the SRS signaled by this PDU.
            // Bit positions:
            // - 0 beamManagement
            // - 1 codebook
            // - 2 nonCodebook
            // - 3 antennaSwitching
            // - 4 - 255: reserved.
            // For each of this bit positions: - 1 = requested
            // - 0 = not requested.
            // nUsage = sum(all bits in usage)
            srs_pdu.fapi_v4_params.usage =  hdf5dset[0]["usage"].as<uint32_t>();
            srs_pdu.fapi_v4_params.numTotalUeAntennas = hdf5dset[0]["numTotalUeAntennas"].as<uint32_t>();
            srs_pdu.fapi_v4_params.sampledUeAntennas = hdf5dset[0]["sampledUeAntennas"].as<uint32_t>();
            srs_pdu.fapi_v4_params.ueAntennasInThisSrsResourceSet = hdf5dset[0]["ueAntennasInThisSrsResourceSet"].as<uint32_t>();
            parse_srs_rx_beamforming(hdf5dset, srs_pdu.rx_beam_data);
            uint32_t idxInd = h5dset_try_parse<uint32_t>(hdf5dset, "idxInd", dset.back() - '0');
            std::string ind1RBSnrsetStr = H5_IND + std::to_string(idxInd) + "_rbSNR";
            std::string ind0dsetStr = H5_IND + std::to_string(idxInd) + "report0";
            std::string ind1dsetStr = H5_IND + std::to_string(idxInd) + "report1";
            //NVLOGC_FMT(TAG, "dset = {}, ind0dsetStr = {}, ind1dsetStr ={} ind1RBSnrsetStr {}", dset.c_str(), ind0dsetStr.c_str(), ind1dsetStr.c_str(),ind1RBSnrsetStr);
            hdf5hpp::hdf5_dataset ind1RBSnrset = file.open_dataset(ind1RBSnrsetStr.c_str());
            hdf5hpp::hdf5_dataset ind0dset = file.open_dataset(ind0dsetStr.c_str());
            hdf5hpp::hdf5_dataset ind1dset = file.open_dataset(ind1dsetStr.c_str());

            srs_pdu.ind.idxInd = idxInd;

            auto& ind0 = srs_pdu.ind.ind0;
            ind0.taOffset = ind0dset[0]["TimingAdvance"].as<uint32_t>();
            ind0.taOffsetNs =  ind0dset[0]["TimingAdvanceNano"].as<int16_t>();
            ind0.wideBandSNR =  ind0dset[0]["wideBandSNR"].as<uint8_t>();
            ind0.prgSize =  ind0dset[0]["prgSize"].as<uint8_t>();
            ind0.numSymbols =  ind0dset[0]["numSymbols"].as<uint8_t>();
            ind0.numReportedSymbols =  ind0dset[0]["numReportedSymbols"].as<uint8_t>();
            ind0.numPRGs =  ind0dset[0]["numPRGs"].as<uint16_t>();

            size_t nbytes = (size_t)ind0.numPRGs * ind0.prgSize;
            srs_pdu.SNRval.resize(nbytes);
            h5dset_try_read(file, ind1RBSnrsetStr.c_str(), srs_pdu.SNRval.data(), nbytes);

            auto& ind1 = srs_pdu.ind.ind1;
            ind1.numUeSrsAntPorts = ind1dset[0]["numUeSrsAntPorts"].as<uint16_t>();
            ind1.numGnbAntennaElements = ind1dset[0]["numGnbAntennaElements"].as<uint16_t>();
            ind1.prgSize = ind1dset[0]["prgSize"].as<uint16_t>();
            ind1.numPRGs =  ind1dset[0]["numPRGs"].as<uint16_t>();

            std::string hEstDsetStr = H5_IND + std::to_string(idxInd) + ("_HestNormToL2");
            hdf5hpp::hdf5_dataset hEstDset = file.open_dataset(hEstDsetStr.c_str());
            auto num = hEstDset.get_num_elements();
            ind1.report_iq_data.resize(num);
            hEstDset.read(ind1.report_iq_data.data());
            srs_tv.data.push_back(std::move(srs_pdu));
            // auto & tv_val= srs_tv.data.back();
            // auto iq = tv_val.ind.ind1.report_iq_data[0];
            // NVLOGC_FMT(TAG, "x = {} y = {}", iq.x, iq.y);
        }
    }
    return 0;
}

int launch_pattern::parse_tv_cv_membank_configs(hdf5hpp::hdf5_file& file, fapi_req_t* req) {

    int cell_id = req->cell_idx;
    auto&   bfw_tv = req->tv_data->bfw_tv;
    tv_dset_list& list     = req->tv_data->dset_tv;
    channel_type_t type = req->channel;
    if(list.find(type) == list.end())
    {
        return -1;
    }

    auto& dsets = list[type];
    uint8_t nUes = 0;
    std::map<uint32_t, bool> rntis_map;
    bool cv_membank_config_read_bfw = false;
    uint8_t ctr = 0;

    for (auto& dset : dsets) {
        ctr++;
        std::vector<uint32_t> rnti_val;
        if(file.is_valid_dataset(dset.c_str())) {
            hdf5hpp::hdf5_dataset hdf5dset = file.open_dataset(dset.c_str());
            nUes = hdf5dset[0]["nUes"].as<uint8_t>();
            rnti_val = hdf5dset[0]["RNTI"].as<std::vector<uint32_t>>();
            for (int i= 0; i < nUes; i++) {
                rntis_map[rnti_val[i]] = true;
                NVLOGD_FMT(TAG, "{} rnti_val {}", __FUNCTION__, rnti_val[i]);
            }
        }
        if (!cv_membank_config_read_bfw) {
            cv_membank_config_read_bfw = true;
            for (uint32_t i = 0; i < (CV_MEM_BANK_MAX_UES/2); i++) {
                std::string srsChestDest = "srsChEstInfo" + std::to_string(i);
                std::string srsChestData = "srsChEstHalf" + std::to_string(i);
                if (file.is_valid_dataset(srsChestDest.c_str()) && file.is_valid_dataset(srsChestData.c_str())) {
                    hdf5hpp::hdf5_dataset srsChestInfoDset = file.open_dataset(srsChestDest.c_str());
                    hdf5hpp::hdf5_dataset srsChestDataDest = file.open_dataset(srsChestData.c_str());
                    cv_membank_config_t info;
                    info.RNTI = i+1;
                    auto iter = rntis_map.find(i);
                    info.reportType = 2; //Support SRS Usage resports as input for UL & DL BFW calcaulation i.e. Usage=Codebook (1st positon bit set in Usage field) 
                    uint8_t temp = static_cast<uint8_t>(type);
                    NVLOGI_FMT(TAG, "{} type {}", __FUNCTION__, temp);
                    info.nGnbAnt = srsChestInfoDset[0]["nGnbAnt"].as<uint32_t>();
                    info.nPrbGrps = srsChestInfoDset[0]["nPrbGrps"].as<uint32_t>();
                    info.nUeAnt = srsChestInfoDset[0]["nUeAnt"].as<uint32_t>();
                    info.startPrbGrp = srsChestInfoDset[0]["startPrbGrp"].as<uint32_t>();
                    info.srsPrbGrpSize = srsChestInfoDset[0]["srsPrbGrpSize"].as<uint32_t>();

                    auto chest_buf_size = srsChestDataDest.get_buffer_size_bytes();
                    info.cv_samples.resize(chest_buf_size);
                    srsChestDataDest.read(info.cv_samples.data());
                    mem_bank_config_req[cell_id].data.push_back(std::move(info));
                }
            }
        }
        rntis_map.clear();
    }
    NVLOGI_FMT(TAG, "{} mem_bank_config_req[{}].data.size = {}", __FUNCTION__, cell_id, mem_bank_config_req[cell_id].data.size());
    return 0;
}

int launch_pattern::parse_tv_bfw(hdf5hpp::hdf5_file& file, fapi_req_t* req) {

    auto&   bfw_tv = req->tv_data->bfw_tv;
    tv_dset_list& list     = req->tv_data->dset_tv;
    channel_type_t type = req->channel;
    if(list.find(type) == list.end())
    {
        return -1;
    }

    auto& dsets = list[type];
    for(auto& dset : dsets)
    {
        if(file.is_valid_dataset(dset.c_str()))
        {
            hdf5hpp::hdf5_dataset hdf5dset = file.open_dataset(dset.c_str());
            bfw_tv.data.emplace_back();
            auto& bfw_pdu = bfw_tv.data.back();
            bfw_pdu.nUes = hdf5dset[0]["nUes"].as<uint8_t>();

            auto rnti_val = hdf5dset[0]["RNTI"].as<std::vector<uint32_t>>();
            auto srs_chest_buff_idx_val = hdf5dset[0]["srsChestBufferIndex"].as<std::vector<uint32_t>>();
            auto pduidx_val = hdf5dset[0]["pduIndex"].as<std::vector<uint32_t>>();
            auto numUeAnts = hdf5dset[0]["numOfUeAnt"].as<std::vector<uint32_t>>();
            auto gnbAnt_start_elem = hdf5dset[0]["gnbAntIdxStart"].as<std::vector<uint32_t>>();
            auto gnbAnt_end_elem = hdf5dset[0]["gnbAntIdxEnd"].as<std::vector<uint32_t>>();

            for (uint8_t idx = 0; idx < bfw_pdu.nUes; idx++) {
                bfw_pdu.ue_grp_data.emplace_back();
                auto& grp_data = bfw_pdu.ue_grp_data.back();
                grp_data.RNTI = rnti_val[idx];
                grp_data.srsChestBufferIndex = srs_chest_buff_idx_val[idx];
                grp_data.pduIndex = pduidx_val[idx];
                grp_data.numOfUeAnt = numUeAnts[idx];
                grp_data.gNbAntIdxStart = gnbAnt_start_elem[idx];
                // grp_data.gNbAntIdxStart = gnbAntIdxStart[0];
                 grp_data.gNbAntIdxEnd = gnbAnt_end_elem[idx];
                // grp_data.gNbAntIdxEnd = gnbAntIdxEnd[0];
                for (uint32_t antIdx = 0; antIdx < grp_data.numOfUeAnt; antIdx++) {
                    std::string dset_name = "ueAntIdx" + std::to_string(antIdx);
                    auto ant_vals =  hdf5dset[0][dset_name.c_str()].as<std::vector<uint32_t>>();
                    grp_data.ueAntIndexes.push_back(ant_vals[idx]);
                }
                NVLOGD_FMT(TAG, "RNTI {}, pduIndex = {}, numofUeAnt = {}, gNbAntIdxStart ={},  gNbAntIdxEnd={}\n", grp_data.RNTI, grp_data.pduIndex,
                    grp_data.numOfUeAnt, grp_data.gNbAntIdxStart, grp_data.gNbAntIdxEnd);
            }
            bfw_pdu.rbStart = hdf5dset[0]["rbStart"].as<uint16_t>();
            bfw_pdu.rbSize = hdf5dset[0]["rbSize"].as<uint16_t>();
            bfw_pdu.numPRGs = hdf5dset[0]["numPRGs"].as<uint16_t>();
            bfw_pdu.prgSize = hdf5dset[0]["bfwPrbGrpSize"].as<uint16_t>();
        }
    }

    return 0;
}

int launch_pattern::parse_precoding_matrix(hdf5hpp::hdf5_file& hdf5file, int cell_id)
{
    if(!hdf5file.is_valid_dataset("nPM"))
    {
        return -1;
    }

    double                nPM_double;
    hdf5hpp::hdf5_dataset nPM_dset = hdf5file.open_dataset("nPM");
    nPM_dset.read(&nPM_double);
    int nPM = nPM_double;
    if (nPM <= 0)
    {
        // Skip if not exist
        return 0;
    }

    // Overwrite previous parsed precoding parameters if nPM > 0
    precoding_matrix_v[cell_id].clear();
    for(int i = 0; i < nPM; i++)
    {
        precoding_matrix_t matrix;
        std::string        dset_name_dim     = "PM" + std::to_string(i + 1) + "_dim";
        std::string        dset_name_coef_re = "PM" + std::to_string(i + 1) + "_coef_real";
        std::string        dset_name_coef_im = "PM" + std::to_string(i + 1) + "_coef_imag";
        if(!hdf5file.is_valid_dataset(dset_name_dim.c_str()) || !hdf5file.is_valid_dataset(dset_name_coef_re.c_str()) || !hdf5file.is_valid_dataset(dset_name_coef_im.c_str()))
        {
            continue;
        }

        hdf5hpp::hdf5_dataset pm_dim  = hdf5file.open_dataset(dset_name_dim.c_str());
        hdf5hpp::hdf5_dataset coef_re = hdf5file.open_dataset(dset_name_coef_re.c_str());
        hdf5hpp::hdf5_dataset coef_im = hdf5file.open_dataset(dset_name_coef_im.c_str());

        matrix.PMidx       = i + 1;
        matrix.numLayers   = pm_dim[0]["nLayers"].as<uint32_t>();
        matrix.numAntPorts = pm_dim[0]["nPorts"].as<uint32_t>();

        int count = matrix.numLayers * matrix.numAntPorts;
        if(coef_re.get_buffer_size_bytes() != count * sizeof(int16_t))
        {
            NVLOGE_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "PM{}_coef_real size error: expected_size={} dataset_size={}", matrix.PMidx, count * 2, coef_re.get_buffer_size_bytes());
            continue;
        }
        if(coef_im.get_buffer_size_bytes() != count * sizeof(int16_t))
        {
            NVLOGE_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "PM{}_coef_imag count error: expected_size={} dataset_size={}", matrix.PMidx, count * 2, coef_im.get_buffer_size_bytes());
            continue;
        }

        int16_t* re_data = new int16_t[count];
        int16_t* im_data = new int16_t[count];
        coef_re.read(re_data);
        coef_im.read(im_data);
        matrix.precoderWeight_v.resize(count);
        for(int j = 0; j < count; j++)
        {
            matrix.precoderWeight_v[j].re = re_data[j];
            matrix.precoderWeight_v[j].im = im_data[j];
        }
        delete [] re_data;
        delete [] im_data;

        precoding_matrix_v[cell_id].push_back(std::move(matrix));
    }
    NVLOGD_FMT(TAG, "{}: cell_id={} nPM={}", __func__, cell_id, nPM);
    return 0;
}

int launch_pattern::set_beam_ids(uint8_t& digBFInterfaces, std::vector<uint16_t>& beamIdx_v)
{
    std::vector<uint16_t>* pBeamIdxV;
    if (current_beam_idx_v.size() > 0)
    {
        pBeamIdxV = &current_beam_idx_v;
    }
    else if (global_beam_idx_v.size() > 0)
    {
        pBeamIdxV = &global_beam_idx_v;
    }
    else
    {
        pBeamIdxV = NULL;
    }

    if (pBeamIdxV != NULL)
    {
        // Set beamIdx array from launch pattern
        digBFInterfaces = pBeamIdxV->size();
        beamIdx_v.resize(digBFInterfaces);
        for(uint16_t i = 0; i < digBFInterfaces; i++)
        {
            beamIdx_v[i] = pBeamIdxV->at(i);
        }
    }
    else
    {
        // Set beamIdx to empty by default
        digBFInterfaces = 0;
        beamIdx_v.resize(0);
    }
    return 0;
}

int launch_pattern::parse_tx_beamforming(hdf5hpp::hdf5_dataset& hdf5dset, tx_beamforming_data_t& beam_data)
{

    beam_data.numPRGs         = h5dset_try_parse<uint32_t>(hdf5dset, "numPRGs", 0);
    beam_data.prgSize         = h5dset_try_parse<uint32_t>(hdf5dset, "prgSize", 0);
    beam_data.digBFInterfaces = h5dset_try_parse<uint32_t>(hdf5dset, "digBFInterfaces", 0);

    NVLOGD_FMT(TAG,"{} enable_dynamic_BF {}", __FUNCTION__, enable_dynamic_BF);

    // Parse PMidx array
    beam_data.PMidx_v.resize(beam_data.numPRGs);
    for(uint16_t i = 0; i < beam_data.numPRGs; i++)
    {
        beam_data.PMidx_v[i] = h5dset_try_parse<uint32_t>(hdf5dset, "PMidx", 0);
    }

    beam_data.beamIdx_v.resize(beam_data.digBFInterfaces);
    if(beam_data.digBFInterfaces)
    {
        beam_data.beamIdx_v = h5dset_try_parse<std::vector<uint16_t>>(hdf5dset, "beamIdx", {});
        //Enable for debugging
        /*for(uint16_t i = 0; i < beam_data.digBFInterfaces; i++)
        {
            NVLOGI_FMT(TAG, "beam index [{}]: {}", i, beam_data.beamIdx_v[i]);
        } */
    }

    // Parse beamIdx array
    //set_beam_ids(beam_data.digBFInterfaces, beam_data.beamIdx_v);
    NVLOGD_FMT(TAG, "TV {} cell {} slot {} channel {}: tx_beamforming: numPRGs {} prgSize {} digBFInterfaces {}",
            curr_tv.c_str(), curr_cell, curr_slot, get_channel_name(curr_ch), beam_data.numPRGs, beam_data.prgSize, beam_data.digBFInterfaces);
    return 0;
}

int launch_pattern::parse_rx_beamforming(hdf5hpp::hdf5_dataset& hdf5dset, rx_beamforming_data_t& beam_data)
{
    beam_data.numPRGs         = h5dset_try_parse<uint32_t>(hdf5dset, "numPRGs", 0);
    beam_data.prgSize         = h5dset_try_parse<uint32_t>(hdf5dset, "prgSize", 0);
    beam_data.digBFInterfaces = h5dset_try_parse<uint32_t>(hdf5dset, "digBFInterfaces", 0);

    beam_data.beamIdx_v.resize(beam_data.digBFInterfaces);
    if(beam_data.digBFInterfaces)
    {
        beam_data.beamIdx_v = h5dset_try_parse<std::vector<uint16_t>>(hdf5dset, "beamIdx", {});
        //Enable for debugging
        /*for(uint16_t i = 0; i < beam_data.digBFInterfaces; i++)
        {
            NVLOGI_FMT(TAG, "beam index [{}]: {}", i, beam_data.beamIdx_v[i]);
        } */
    }

    // Parse beamIdx array
    //set_beam_ids(beam_data.digBFInterfaces, beam_data.beamIdx_v);
    NVLOGD_FMT(TAG, "rx_beamforming: numPRGs {} prgSize {} digBFInterfaces {}", beam_data.numPRGs, beam_data.prgSize, beam_data.digBFInterfaces);
    return 0;
}

int launch_pattern::parse_srs_rx_beamforming(hdf5hpp::hdf5_dataset& hdf5dset, rx_srs_beamforming_data_t& beam_data)
{
    beam_data.numPRGs         = h5dset_try_parse<uint32_t>(hdf5dset, "numPRGs", 0);
    beam_data.prgSize         = h5dset_try_parse<uint32_t>(hdf5dset, "prgSize", 0);
    beam_data.digBFInterfaces = h5dset_try_parse<uint32_t>(hdf5dset, "digBFInterfaces", 0);
#ifdef SCF_FAPI_10_04
    //std::vector<uint32_t> beamIdx_v;
    //beamIdx_v.resize(beam_data.digBFInterfaces);
    //beamIdx_v = h5dset_try_parse<std::vector<uint32_t>>(hdf5dset, "beamIdx", {});
    // Parse beamIdx array
    //set_beam_ids(beam_data.digBFInterfaces, beam_data.beamIdx_v, );

    beam_data.beamIdx_v.resize(beam_data.numPRGs, vector<uint16_t>(beam_data.digBFInterfaces));

    //beam_data.beamIdx_v = h5dset_try_parse<std::vector<uint16_t>>(hdf5dset, "beamIdx", 0);
    for (uint16_t prgIdx = 0; prgIdx < beam_data.numPRGs; prgIdx++)
    {
        //beam_data.beamIdx_v + prgIdx.resize(beam_data.digBFInterfaces);
        // TODO: Below code is crashing so need to fingure out an alternative:
        //beam_data.beamIdx_v[prgIdx] = h5dset_try_parse<std::vector<uint16_t>>(hdf5dset, "beamIdx", {});

        for (uint16_t digBfIdx = 0; digBfIdx < beam_data.digBFInterfaces; digBfIdx++)
        {
           beam_data.beamIdx_v[prgIdx][digBfIdx] = digBfIdx + 1; //TODO: introduce logic to read from TV file could be beamIdx_v[digBfIdx]
           //NVLOGD_FMT(TAG, "rx_beamforming: prgIdx: {}/{} beam_data.beamIdx_v = {}  beamIdx_v={}", prgIdx,beam_data.numPRGs,beam_data.beamIdx_v[prgIdx][digBfIdx], digBfIdx+1);
        }
    }
    NVLOGD_FMT(TAG, "rx_beamforming: numPRGs {} prgSize {} digBFInterfaces {} ", beam_data.numPRGs, beam_data.prgSize, beam_data.digBFInterfaces);
#endif
    return 0;
}
//#endif

int launch_pattern::parse_time_lines(hdf5hpp::hdf5_file& file, int cell_id) {
    if (!file.is_valid_dataset("nTimelines")) {
        return -1;
    }

    uint32_t nTimelines = 0;
    auto dset = file.open_dataset("nTimelines");
    dset.read(&nTimelines);

    if (nTimelines < 1) {
        return -1;
    }
    const char* timeline_dst_pref = "Timeline_";
    for (uint32_t i = 0; i < nTimelines; i++) {
        auto dset_str = std::string(timeline_dst_pref) + std::to_string(i);
        if (!file.is_valid_dataset(dset_str.c_str())) {
            continue;
        }
        auto tdset = file.open_dataset(dset_str.c_str());
        auto type = tdset[0]["pduType"].as<uint16_t>();
        auto chan_start_offset = tdset[0]["Tchan_start_offset"].as<uint16_t>();
        auto chan_duration = tdset[0]["Tchan_duration"].as<uint16_t>();
        ch_segment_map.try_emplace(cell_id, std::vector<channel_segment_t>());
        ch_segment_map[cell_id].push_back({type, chan_start_offset, chan_duration});
    }
    return 0;
}

int launch_pattern::parse_dbt_configs(hdf5hpp::hdf5_file& file, int cell_id) {

    double num_static_beamIdx = 0.0;
    double num_TRX_beamforming = 0.0;
    double enable_static_dynamic_beamforming = 0.0;

    if (!file.is_valid_dataset("enable_static_dynamic_beamforming") || 
        !file.is_valid_dataset("num_static_beamIdx") || 
        !file.is_valid_dataset("num_TRX_beamforming")) {
        return -1;
    }


    auto hdf5dset0{file.open_dataset("enable_static_dynamic_beamforming")};
    hdf5dset0.read(&enable_static_dynamic_beamforming);

    auto hdf5dset1{file.open_dataset("num_static_beamIdx")};
    hdf5dset1.read(&num_static_beamIdx);

    auto hdf5dset2{file.open_dataset("num_TRX_beamforming")};
    hdf5dset2.read(&num_TRX_beamforming);

    NVLOGC_FMT(TAG, "parse_dbt_configs enable_static_dynamic_beamforming {} num_static_beamIdx {} num_TRX_beamforming {}", enable_static_dynamic_beamforming, num_static_beamIdx, num_TRX_beamforming);

    enable_static_dynamic_BF = enable_static_dynamic_beamforming;

    dbt_md_t dbt_conf{!!static_cast<uint>(enable_static_dynamic_beamforming), static_cast<uint16_t>(num_static_beamIdx), static_cast<uint16_t>(num_TRX_beamforming), dbt_data_t{}};

    hdf5hpp::hdf5_dataset coef_re = file.open_dataset("DBT_real");
    hdf5hpp::hdf5_dataset coef_im = file.open_dataset("DBT_imag");

    auto count = dbt_conf.num_static_beamIdx * dbt_conf.num_TRX_beamforming;
    int16_t* re_data = new int16_t[count];
    int16_t* im_data = new int16_t[count];
    coef_re.read(re_data);
    coef_im.read(im_data);
    dbt_conf.dbt_data_buf.resize(count);
    for(int j = 0; j < count; j++)
    {
        dbt_conf.dbt_data_buf[j] = {re_data[j], im_data[j]};
    }
    delete [] re_data;
    delete [] im_data;
    dbt_per_cell_md_list.reserve(cell_id);
    dbt_per_cell_md_list.push_back(std::move(dbt_conf));

    return 0;
}

int launch_pattern::parse_tv_file(fapi_req_t* req)
{
    if(req == nullptr || req->tv_file.length() == 0)
    {
        return -1;
    }

    hdf5hpp::hdf5_file& file = *req->h5f;
    if(parse_pdu_dset_names(file, req->tv_data->dset_tv))
    {
        // No PDU in the TV
        NVLOGD_FMT(TAG, "TV cell {} slot {} {} {} TV has no PDU",
                curr_cell, curr_slot, get_channel_name(curr_ch), curr_tv.c_str());
        return -1;
    }

    if (req->tv_data->dset_tv[req->channel].size() == 0)
    {
        // No PDU for this channel
        NVLOGD_FMT(TAG, "TV cell {} slot {} {} {} not exist",
                curr_cell, curr_slot, get_channel_name(curr_ch), curr_tv.c_str());
        return -1;
    }

    try
    {
        if(req->channel == channel_type_t::PBCH)
        {
            parse_tv_pbch(file, req);
        }
        else if(req->channel == channel_type_t::PRACH)
        {
            parse_tv_prach(file, req);
        }
        else if(req->channel == channel_type_t::PUSCH)
        {
            parse_tv_pusch(file, req);
        }
        else if(req->channel == channel_type_t::PDSCH)
        {
            parse_tv_pdsch(file, req);
        }
        else if(req->channel == channel_type_t::PDCCH_DL)
        {
            parse_tv_pdcch(file, req);
        }
        else if(req->channel == channel_type_t::PDCCH_UL)
        {
            parse_tv_pdcch(file, req);
        }
        else if(req->channel == channel_type_t::PUCCH)
        {
            parse_tv_pucch(file, req);
        }
        else if(req->channel == channel_type_t::CSI_RS)
        {
            parse_tv_csirs(file, req);
        }
        else if(req->channel == channel_type_t::SRS)
        {
            parse_tv_srs(file, req);
        }
        else if (req->channel == channel_type_t::BFW_DL) {
            if (req->cell_idx < PDSCH_MAX_CELLS_PER_CELL_GROUP){
                if (!cv_membank_config_read_dl_bfw[req->cell_idx]){
                        cv_membank_config_read_dl_bfw[req->cell_idx] = true;
                        parse_tv_cv_membank_configs(file, req);
                    }
                }
            else{
                    NVLOGW_FMT(TAG,"BFW_DL curr_cell={} is greater than PDSCH_MAX_CELLS_PER_CELL_GROUP!", req->cell_idx);
                    return -1;
            }
            parse_tv_bfw(file, req);
        }
        else if (req->channel == channel_type_t::BFW_UL) {
            if (req->cell_idx < PDSCH_MAX_CELLS_PER_CELL_GROUP){
	            if (!cv_membank_config_read_ul_bfw[req->cell_idx]){
                    cv_membank_config_read_ul_bfw[req->cell_idx] = true;
                    parse_tv_cv_membank_configs(file, req);
                }
            }
            else{
                    NVLOGW_FMT(TAG,"BFW_UL curr_cell={} is greater than PDSCH_MAX_CELLS_PER_CELL_GROUP!", curr_cell);
                    return -1;
            }
            parse_tv_bfw(file, req);
        }

        else
        {
            NVLOGE_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "TV cell {} slot {} {} {} channel not supported",
                    curr_cell, curr_slot, get_channel_name(curr_ch), curr_tv.c_str());
            return -1;
        }
    }
    catch(std::exception& e)
    {
        NVLOGF_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "TV cell {} slot {} {} {} exception: {}",
                curr_cell, curr_slot, get_channel_name(curr_ch), curr_tv.c_str(), e.what());
        return -1;
    }
    return 0;
}

int launch_pattern::load_h5_config_params(int cell_id, const char* config_params_h5_file, const char* ul_params_h5_file)
{
    char h5path_array[MAX_PATH_LEN];
    get_full_path_file(h5path_array, CONFIG_TEST_VECTOR_PATH, config_params_h5_file, CONFIG_CUBB_ROOT_DIR_RELATIVE_NUM);
    std::filesystem::path h5path(h5path_array);
    NVLOGC_FMT(TAG, "config params {} {:p}", h5path.c_str(), (void*)config_params_h5_file);

    if(access(h5path.c_str(), F_OK) != 0)
    {
        NVLOGF_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "File {} does not exist", h5path.c_str());
    }

    hdf5hpp::hdf5_file hdf5file, ul_hdf5file;
    try
    {
        hdf5file = hdf5hpp::hdf5_file::open(h5path.c_str());
    }
    catch(std::exception& e)
    {
        NVLOGE_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "Exception: {}: hdf5_file::open({}): {}", __FUNCTION__, h5path.c_str(), e.what());
        return -1;
    }

    // Try opening file for UL parameters
    try
    {
        if(nullptr != ul_params_h5_file)
        {
            char ul_h5path_array[MAX_PATH_LEN];
            get_full_path_file(ul_h5path_array, CONFIG_TEST_VECTOR_PATH, ul_params_h5_file, CONFIG_CUBB_ROOT_DIR_RELATIVE_NUM);
            std::filesystem::path ul_h5path(ul_h5path_array);
            if(access(ul_h5path.c_str(), F_OK) != 0)
            {
                NVLOGF_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "File {} does not exist", ul_h5path.c_str());
            }
            ul_hdf5file = hdf5hpp::hdf5_file::open(ul_h5path.c_str());
        }
    }
    catch(const std::exception& e)
    {
        NVLOGE_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "Exception: {}: opening {}: {} Defaulting to contents in {}", __FUNCTION__, h5path.c_str(), e.what(),config_params_h5_file);
    }

    if(hdf5file.is_valid_dataset("Cell_Config"))
    {
        hdf5hpp::hdf5_dataset      dset         = hdf5file.open_dataset("Cell_Config");
        hdf5hpp::hdf5_dataset_elem dset_elem    = dset[0];
        cell_configs_t*            cell_configs = new cell_configs_t;
        cell_configs->dlGridSize                = dset_elem["dlGridSize"].as<unsigned int>();
        cell_configs->ulGridSize                = dset_elem["ulGridSize"].as<unsigned int>();
        cell_configs->dlBandwidth               = dset_elem["dlBandwidth"].as<unsigned int>();
        cell_configs->ulBandwidth               = dset_elem["ulBandwidth"].as<unsigned int>();
        cell_configs->numTxAnt                  = dset_elem["numTxAnt"].as<unsigned int>();
        cell_configs->numRxAnt                  = dset_elem["numRxAnt"].as<unsigned int>();
        cell_configs->numTxPort                 = dset_elem["numTxPort"].as<unsigned int>();
        cell_configs->numRxPort                 = dset_elem["numRxPort"].as<unsigned int>();
        cell_configs->mu                        = dset_elem["mu"].as<unsigned int>();
        cell_configs->phyCellId                 = dset_elem["phyCellId"].as<unsigned int>();
        cell_configs->negTV_enable              = dset_elem["negTV_enable"].as<uint>();
        cell_configs->enable_dynamic_BF         = h5dset_try_parse<uint>(dset_elem, "enable_dynamic_BF", 0);
        if(cell_configs->enable_dynamic_BF)
        {
            enable_dynamic_BF = 1;
        }

        NVLOGD_FMT(TAG, "cell_configs->enable_dynamic_BF {} enable_dynamic_BF {}", cell_configs->enable_dynamic_BF, enable_dynamic_BF);
        cell_configs->dmrsTypeAPos = h5dset_try_parse<uint32_t>(dset_elem, "dmrsTypeAPos", 0);
        cell_configs->frameDuplexType = h5dset_try_parse<uint32_t>(dset_elem, "FrameDuplexType", 1);

        hdf5hpp::hdf5_dataset dset_alg_cfg = hdf5file.open_dataset("Alg_Config");
        cell_configs->pusch_sinr_selector = h5dset_try_parse<uint32_t>(dset_alg_cfg, "pusch_sinr_selector", 2);
        cell_configs->enableWeightedAverageCfo = h5dset_try_parse<uint32_t>(dset_alg_cfg, "enableWeightedAverageCfo", 0);

        uint32_t bfp = 16;
        if(hdf5file.is_valid_dataset(H5_BFP))
        {
            hdf5hpp::hdf5_dataset dset = hdf5file.open_dataset(H5_BFP);
            dset.read(&bfp);
        }
        cell_configs->BFP = bfp;

        cell_configs_v[cell_id]    = cell_configs;
        if (cell_id == 0)
        {
            slots_per_frame = 10 * (1 << cell_configs->mu); // slots_per_frame = 10 * 2 ^ mu
        }

        // Initiate default tolerance for all: PUSCH, PUCCH PF01, PUCCH PF234
        vald_tolerance_t& tolerance = cell_configs->tolerance;
        for (int pdu_type = 0; pdu_type < UCI_PDU_TYPE_NUM; pdu_type ++)
        {
            ul_measurement_t& ul_meas = tolerance.ul_meas[pdu_type];
            ul_meas.UL_CQI = VALD_TOLERANCE_UL_CQI;
            ul_meas.SNR = VALD_TOLERANCE_SINR;
            ul_meas.TimingAdvance = VALD_TOLERANCE_TIMING_ADVANCE;
            ul_meas.TimingAdvanceNs = VALD_TOLERANCE_TIMING_ADVANCE_NS;
            ul_meas.RSSI = VALD_TOLERANCE_RSSI;
            ul_meas.RSRP = VALD_TOLERANCE_RSRP;
            tolerance.pusch_pe_noiseVardB = VALD_TOLERANCE_MEAS_PUSCH_NOISE;
        }

        // Overwrite PUSCH BFP tolerances
        if (cell_configs->BFP == 9)
        {
            // PUSCH: pdu_type 0
            tolerance.ul_meas[UCI_PDU_TYPE_PUSCH].UL_CQI = VALD_TOLERANCE_UL_CQI_BFP9;
            tolerance.ul_meas[UCI_PDU_TYPE_PUSCH].SNR = VALD_TOLERANCE_SINR_BFP9;
            tolerance.ul_meas[UCI_PDU_TYPE_PUSCH].RSSI = VALD_TOLERANCE_RSSI_BFP9;
            tolerance.ul_meas[UCI_PDU_TYPE_PUSCH].RSRP = VALD_TOLERANCE_RSRP_BFP9;
            tolerance.pusch_pe_noiseVardB = VALD_TOLERANCE_MEAS_PUSCH_NOISE_BFP9;
        }

        NVLOGC_FMT(TAG, "{}: loaded 'Cell_Config': cell_id={} phyCellId={} numTxAnt={} numTxPort={} numRxAnt={} numRxPort={} BFP={} h5file={}",
                __func__, cell_id, cell_configs->phyCellId, cell_configs->numTxAnt, cell_configs->numTxPort, cell_configs->numRxAnt, cell_configs->numTxPort, cell_configs->BFP, config_params_h5_file);
    }
    else
    {
        NVLOGE_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "{}: Cell_Config not found in TV {}", __FUNCTION__, h5path.c_str());
        return -1;
    }

    if (nullptr != ul_params_h5_file && ul_hdf5file.is_valid_dataset("Prach_Config"))
    {
        parse_prach_config(ul_hdf5file, cell_id);
    }
    else if (hdf5file.is_valid_dataset("Prach_Config"))
    {
        parse_prach_config(hdf5file, cell_id);
    }

    parse_precoding_matrix(hdf5file, cell_id);

    parse_time_lines(hdf5file, cell_id);

    parse_dbt_configs(hdf5file, cell_id);

    // For CSI2 Maps read from UL_Cell_Configs or Cell_Configs
    auto &hdf5flehdl = ul_params_h5_file ? ul_hdf5file: hdf5file;
    read_csip2_maps(hdf5flehdl, cell_id);

    return 0;
}

struct parsing_thread_arg_t {
    int thread_id;
    int thread_num;
    int cell_num;
    int cpu_core;

    std::vector<int32_t>* p_lp_cell_id_vec;

    pthread_t pid;
    launch_pattern* lp;
    yaml::node* yaml_node;
    slot_pattern_t* slot_data;
};

void* parsing_thread_func(void* arg)
{
    nvlog_fmtlog_thread_init();
    NVLOGC_FMT(TAG, "Thread {} on CPU {} initialized fmtlog", __FUNCTION__, sched_getcpu());

    struct parsing_thread_arg_t* slot_parsing = (parsing_thread_arg_t*)arg;
    launch_pattern* lp = slot_parsing->lp;
    yaml::node& slot_list = *(slot_parsing->yaml_node);
    slot_pattern_t& slots_data = *(slot_parsing->slot_data);

    char thread_name[32];
    snprintf(thread_name, 32, "lp_parse_%02d", slot_parsing->thread_id);
    if(pthread_setname_np(pthread_self(), thread_name) != 0)
    {
        NVLOGW_FMT(TAG, "{}: set thread name {} failed", __func__, thread_name);
    }

    nv_assign_thread_cpu_core(slot_parsing->cpu_core);
    NVLOGC_FMT(TAG, "{}: thread {:02d} started on CPU core {:02d}", __FUNCTION__, slot_parsing->thread_id, slot_parsing->cpu_core);

    for(size_t id = 0; id < slot_list.length(); ++id)
    {
        if(is_app_exiting())
        {
            NVLOGC_FMT(TAG, "{}: thread [{}] exiting due to application exiting", __FUNCTION__, slot_parsing->thread_id);
            break;
        }

        int slot_id = slot_list[id]["slot"].as<int>();
        curr_slot = slot_id;
        yaml::node cell_configs = slot_list[id]["config"];
        if(cell_configs.type() == YAML_SEQUENCE_NODE)
        {
            for (int cell_id = slot_parsing->thread_id; cell_id < slot_parsing->cell_num; cell_id += slot_parsing->thread_num)
            {
                std::vector<int32_t>& lp_cell_id = *slot_parsing->p_lp_cell_id_vec;
                if(lp->populate_launch_pattern(slots_data, cell_configs[lp_cell_id[cell_id]], cell_id, slot_id) < 0)
                {
                    std::stringstream ss;
                    ss << __FUNCTION__ << " error: slot=" << slot_id << " lp_cell_id=" << lp_cell_id[cell_id] << std::endl;
                    throw std::runtime_error(ss.str().c_str());
                }
                NVLOGC_FMT(TAG, "{}: thread [{}] parsed TVs for cell {:02d} slot {:02d}/{}", __FUNCTION__, slot_parsing->thread_id, cell_id, slot_id, slot_list.length());
            }
        }
    }

    return nullptr;
}

void launch_pattern::parse_slots(yaml::node& slot_list, slot_pattern_t& slots_data)
{
    if (prach_reconfig_flag != 0)
    {
        slots_data.resize(sched_slots_pattern.size());
    }
    else
    {
        slots_data.resize(slot_list.length());
    }

    for(int slot_id = 0; slot_id < slots_data.size(); slot_id++)
    {
        slots_data[slot_id].resize(cell_num);
        for(int cell_id = 0; cell_id < cell_num; cell_id++)
        {
            slots_data[slot_id][cell_id].resize(FAPI_REQ_SIZE);
        }
    }

    // int thread_num = yaml_configs->worker_cores.size();
    // thread_num = thread_num < cell_num ? thread_num : cell_num;

    // Force using 1 thread to load TV because in practice multi-threads have worse performance
    constexpr int thread_num = 1;

    std::array<struct parsing_thread_arg_t, thread_num> thread_args{};
    for(int thread_id = 0; thread_id < thread_num; thread_id++)
    {
        struct parsing_thread_arg_t& thread_arg = thread_args[thread_id];
        thread_arg.lp = this;
        thread_arg.yaml_node = &slot_list;
        thread_arg.slot_data = &slots_data;
        thread_arg.thread_id = thread_id;
        thread_arg.thread_num = thread_num;
        thread_arg.cell_num = cell_num;
        thread_arg.cpu_core = yaml_configs->get_recv_thread_config().cpu_affinity; // yaml_configs->worker_cores[thread_id];
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

int launch_pattern::launch_pattern_parsing(const char* lp_file_name, uint32_t ch_mask, uint64_t cell_mask)
{
    channel_mask = ch_mask;

    char pattern_file_array[MAX_PATH_LEN];
    get_full_path_file(pattern_file_array, CONFIG_LAUNCH_PATTERN_PATH, lp_file_name, CONFIG_CUBB_ROOT_DIR_RELATIVE_NUM);
    std::filesystem::path pattern_file(pattern_file_array);

    yaml::file_parser fp(pattern_file.c_str());
    yaml::document    doc     = fp.next_document();
    yaml::node        pattern = doc.root();

    if(pattern.has_key("beam_ids"))
    {
        yaml::node beam_ids = pattern["beam_ids"];
        global_beam_idx_v.resize(beam_ids.length());
        for(int n = 0; n < beam_ids.length(); n++)
        {
            int beam_id   = beam_ids[n].as<int>();
            global_beam_idx_v[n] = beam_id;
            NVLOGI_FMT(TAG, "global_beam_idx_v: {}-{} = {}", beam_ids.length(), n, beam_id);
        }
    }

    if(pattern.has_key("config_static_harq_proc_id"))
    {
        config_static_harq_proc_id = pattern["config_static_harq_proc_id"].as<int>();
    }
    else
    {
        config_static_harq_proc_id = 0;
    }

    if (pattern.has_key("Num_Cells"))
    {
        cell_num = pattern["Num_Cells"].as<int>();
    }
    else if (pattern.has_key("numCellsActive"))
    {
        // For JSON dynamic pattern
        cell_num = pattern["numCellsActive"].as<int>();
    }
    else
    {
        cell_num = pattern["Cell_Configs"].length();
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
            NVLOGE_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "cell_mask=0x{:X} size {} exceeds launch_pattern cell_num {}", cell_mask, lp_cell_id_vec.size(), cell_num);
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

    precoding_matrix_v.resize(cell_num);
    init_prach_config(cell_num);
    hpid_maps.resize(cell_num);
    cell_configs_v.resize(cell_num);
    if(pattern.has_key("Cell_Configs"))
    {
        // Load from list
        yaml::node cell_configs_list = pattern["Cell_Configs"];
        if(cell_configs_list.length() != cell_num)
        {
            NVLOGW_FMT(TAG, "{}: cell num doesn't match: cell_num={} config_list={}", __FUNCTION__, cell_num, cell_configs_list.length());
        }
        NVLOGC_FMT(TAG, "{}: config_list={} cell_mask={} cell_num={} lp_cell_id.size={}",
                __FUNCTION__, cell_configs_list.length(), cell_mask, cell_num, lp_cell_id_vec.size());

        int num_ul_config = 0;
        if(pattern.has_key("UL_Cell_Configs"))
        {
            num_ul_config = pattern["UL_Cell_Configs"].length();
        }

        for(int cell_id = 0; cell_id < cell_num; cell_id++)
        {
            int32_t lp_cell_id = lp_cell_id_vec[cell_id];
            std::string config_tv = cell_configs_list[lp_cell_id].as<std::string>();
            std::string ul_config_tv;
            const char* ul_config_str = nullptr;
            if(num_ul_config > lp_cell_id)
            {
                ul_config_tv = pattern["UL_Cell_Configs"][lp_cell_id].as<std::string>();
                ul_config_str = ul_config_tv.c_str();
            }
            if(load_h5_config_params(cell_id, config_tv.c_str(), ul_config_str) < 0)
            {
                return -1;
            }
        }

        // Workaround for same phyCellId in Cell_Config TV of different cells
        if (cell_configs_list.length() > 1) {
            uint32_t phyCellId_0 = h5file_try_parse<uint32_t>(cell_configs_list[0UL].as<std::string>().c_str(), "Cell_Config", "phyCellId", 0);
            uint32_t phyCellId_1 = h5file_try_parse<uint32_t>(cell_configs_list[1UL].as<std::string>().c_str(), "Cell_Config", "phyCellId", 0);
            if (phyCellId_0 == phyCellId_1) {
                for(int cell_id = 0; cell_id < cell_num; cell_id++)
                {
                    cell_configs_v[cell_id]->phyCellId += lp_cell_id_vec[cell_id];
                    NVLOGC_FMT(TAG, "{}: Updated phyCellId: cell_id={} phyCellId={}", __func__, cell_id, cell_configs_v[cell_id]->phyCellId);
                }
            }
        }
    }
    else
    {
        for(int cell_id = 0; cell_id < cell_num; cell_id++)
        {
            std::string config_params_file = pattern["Cells"].as<std::string>();
            if(load_h5_config_params(cell_id, config_params_file.c_str(), nullptr) < 0)
            {
                return -1;
            }
        }
    }

    if (pattern.has_key("TV"))
    {
        for(int ch = 0; ch < channel_type_t::CHANNEL_MAX; ch++)
        {
            if(pattern["TV"].has_key(get_channel_name(ch)))
            {
                std::unordered_map<std::string, std::string>* tv_map = new std::unordered_map<std::string, std::string>;
                tv_maps[ch]                                          = tv_map;
                yaml::node tv_list                                   = pattern["TV"][get_channel_name(ch)];
                for(size_t i = 0; i < tv_list.length(); ++i)
                {
                    std::string tv_name = tv_list[i]["name"].as<std::string>();
                    std::string tv_file = tv_list[i]["path"].as<std::string>();
                    tv_map->insert(make_pair(tv_name, tv_file));
                    NVLOGI_FMT(TAG, "{}: tv list: {:<8} {} {}", __FUNCTION__, get_channel_name(ch), tv_name.c_str(), tv_file.c_str());
                }
            }
        }
    }

    // Expand expected throughput array size
    expected.resize(cell_num);

    if(yaml_configs->app_mode == 0 && pattern.has_key("INIT"))
    {
        yaml::node slot_list = pattern["INIT"];
        init_slot_num        = slot_list.length();
        parse_slots(slot_list, init_slots_pattern);
        using_init_patterns = init_slot_num > 0 ? true : false;
        NVLOGC_FMT(TAG, "{}: parsed INIT slots: cell_num={} init_slot_num={}", __FUNCTION__, cell_num, init_slot_num);
    }

    for(int i = 0; i < cell_num; i++)
    {
        memset(&expected[i], 0, sizeof(thrput_t));
    }

    if(yaml_configs->app_mode == 0 && pattern.has_key("SCHED"))
    {
        int64_t ts_start = system_clock::now().time_since_epoch().count();
        yaml::node slot_list = pattern["SCHED"];
        sched_slot_num       = slot_list.length();
        parse_slots(slot_list, sched_slots_pattern);
        int64_t ts_end = system_clock::now().time_since_epoch().count();
        float time_cost = (float)(ts_end - ts_start) / 1E9;
        NVLOGC_FMT(TAG, "{}: parsed SCHED slots: cell_num={} sched_slot_num={} time={:.1f}s", __FUNCTION__, cell_num, sched_slot_num, time_cost);
    }

    if(yaml_configs->app_mode != 0)
    {
        // Parse dynamic slot pattern from JSON
        if(dynamic_pattern_parsing(pattern, channel_mask, cell_mask) < 0)
        {
            return -1;
        }
    }

//    if(pattern.has_key("RECONFIG"))
//    {
//        prach_reconfig_flag = 1;
//        yaml::node slot_list = pattern["RECONFIG"];
//        parse_slots(slot_list, reconfig_slots_pattern);
//        prach_reconfig_flag = 0;
//        NVLOGC_FMT(TAG, "{}: parsed RECONFIG slots: list_size={}", __FUNCTION__, slot_list.length());
//    }

    if (sched_slot_num <= 0)
    {
        return -1;
    }

    for(int i = 0; i < cell_num; i++)
    {
        int schedule_per_second = SLOTS_PER_SECOND / sched_slot_num;

        std::string exp_slot = "ExpectedSlots: Cell=" + std::to_string(i);
        for(int ch = 0; ch < channel_type_t::CHANNEL_MAX; ch++)
        {
            expected[i].slots[ch] = expected[i].lp_slots[ch] * schedule_per_second;
            exp_slot.append(" ").append(get_channel_name(ch)).append("=").append(std::to_string(expected[i].slots[ch]));
        }

        if (expected[i].error > 0) {
            expected[i].error = expected[i].error * schedule_per_second;
        } else {
            expected[i].error = 0;
        }
        expected[i].dl_thrput = expected[i].dl_thrput * schedule_per_second;
        expected[i].ul_thrput = expected[i].ul_thrput * schedule_per_second;
        expected[i].mac_ul_drop = expected[i].mac_ul_drop * schedule_per_second;
        expected[i].prmb = expected[i].prmb * schedule_per_second;
        expected[i].harq = expected[i].harq * schedule_per_second;
        expected[i].sr = expected[i].sr * schedule_per_second;
        expected[i].csi1 = expected[i].csi1 * schedule_per_second;
        expected[i].csi2 = expected[i].csi2 * schedule_per_second;
        expected[i].srs = expected[i].srs * schedule_per_second;
        expected[i].invalid = expected[i].invalid * schedule_per_second;
    
        std::string exp_data = "ExpectedData: Cell=" + std::to_string(i);
        exp_data.append(" DL=").append(std::to_string((double)expected[i].dl_thrput * 8 / 1000000));
        exp_data.append(" UL=").append(std::to_string((double)expected[i].ul_thrput * 8 / 1000000));
        exp_data.append(" MAC_UL_DROP=").append(std::to_string((double)expected[i].mac_ul_drop * 8 / 1000000));
        exp_data.append(" Prmb=").append(std::to_string(expected[i].prmb));
        exp_data.append(" HARQ=").append(std::to_string(expected[i].harq));
        exp_data.append(" SR=").append(std::to_string(expected[i].sr));
        exp_data.append(" CSI1=").append(std::to_string(expected[i].csi1));
        exp_data.append(" CSI2=").append(std::to_string(expected[i].csi2));
        exp_data.append(" SRS=").append(std::to_string(expected[i].srs));
        exp_data.append(" ERR=").append(std::to_string(expected[i].error));
        exp_data.append(" INV=").append(std::to_string(expected[i].invalid));

        NVLOGC_FMT(TAG, "{}", exp_slot.c_str());
        NVLOGC_FMT(TAG, "{}", exp_data.c_str());
    }

    return 0;
}

int launch_pattern::dynamic_pattern_parsing(yaml::node& root_pattern, uint32_t ch_mask, uint64_t cell_mask)
{
    channel_mask = ch_mask;

    // Disable other channels than PDSCH and PUSCH
    if(yaml_configs->app_mode == 1)
    {
        channel_mask &= ((0x01 << channel_type_t::PDSCH) | (0x01 << channel_type_t::PUSCH));
    }
    else
    {
        // UE dynamic mode, only enable PUSCH
        channel_mask &= (0x01 << channel_type_t::PUSCH);
    }

    // Parse static slot parameters from JSON Cell_Configs TV and ue_profile
    static_slot_params.resize(cell_num);
    yaml::node dl_profile = root_pattern["ue_profile"]["dl"];
    yaml::node ul_profile = root_pattern["ue_profile"]["ul"];
    for(int cell_id = 0; cell_id < cell_num; cell_id++)
    {
        cell_configs_t* cell_config = cell_configs_v[cell_id];
        static_slot_param_t& static_params = static_slot_params[cell_id];

        // PDSCH static parameters
        static_params.pdsch.pduBitmap = dl_profile["pduBitmap"].as<uint32_t>();
        static_params.pdsch.BWPSize = cell_config->dlGridSize;
        static_params.pdsch.BWPStart = 0;
        static_params.pdsch.SubCarrierSpacing = cell_config->mu;
        static_params.pdsch.CyclicPrefix = dl_profile["CyclicPrefix"].as<uint32_t>();
        static_params.pdsch.NrOfCodeWords = dl_profile["NrOfCodeWords"].as<uint32_t>();
        static_params.pdsch.rvIndex = dl_profile["rvIndex"].as<uint32_t>();
        static_params.pdsch.dataScramblingId = cell_config->phyCellId;
        static_params.pdsch.transmission = dl_profile["transmission"].as<uint32_t>();
        static_params.pdsch.refPoint = dl_profile["refPoint"].as<uint32_t>();
        static_params.pdsch.dlDmrsScrmablingId = cell_config->phyCellId;
        static_params.pdsch.scid = dl_profile["scid"].as<uint32_t>();
        static_params.pdsch.resourceAlloc = dl_profile["resourceAlloc"].as<uint32_t>();
        static_params.pdsch.VRBtoPRBMapping = dl_profile["VRBtoPRBMapping"].as<uint32_t>();
        static_params.pdsch.powerControlOffset = dl_profile["powerControlOffset"].as<uint32_t>();
        static_params.pdsch.powerControlOffsetSS = dl_profile["powerControlOffsetSS"].as<uint32_t>();

        yaml_try_parse_list(dl_profile, "numDmrsCdmGrpsNoData", static_params.pdsch.numDmrsCdmGrpsNoData);
        yaml_try_parse_list(dl_profile, "rbBitmap", static_params.pdsch.rbBitmap);

        // TODO: Hard-code beamforming to 0 currently
        tx_beamforming_data_t& tx_beam = static_params.pdsch.tx_beam_data;
        tx_beam.numPRGs = 0;
        tx_beam.prgSize = 0;
        tx_beam.digBFInterfaces = 0;

        // PUSCH static parameters
        static_params.pusch.pduBitmap = ul_profile["pduBitmap"].as<uint32_t>();
        static_params.pusch.BWPSize = cell_config->ulGridSize;
        static_params.pusch.BWPStart = 0;
        static_params.pusch.SubCarrierSpacing = cell_config->mu;
        static_params.pusch.CyclicPrefix = ul_profile["CyclicPrefix"].as<uint32_t>();
        static_params.pusch.dataScramblingId = cell_config->phyCellId;
        static_params.pusch.dmrsConfigType = ul_profile["dmrsConfigType"].as<uint32_t>();
        static_params.pusch.ulDmrsScramblingId = cell_config->phyCellId;
        static_params.pusch.puschIdentity = ul_profile["puschIdentity"].as<uint32_t>();
        static_params.pusch.scid = ul_profile["scid"].as<uint32_t>();
        static_params.pusch.resourceAlloc = ul_profile["resourceAlloc"].as<uint32_t>();
        static_params.pusch.VRBtoPRBMapping = ul_profile["VRBtoPRBMapping"].as<uint32_t>();
        static_params.pusch.FrequencyHopping = ul_profile["FrequencyHopping"].as<uint32_t>();
        static_params.pusch.txDirectCurrentLocation = ul_profile["txDirectCurrentLocation"].as<uint32_t>();
        static_params.pusch.uplinkFrequencyShift7p5khz = ul_profile["uplinkFrequencyShift7p5khz"].as<uint32_t>();
        static_params.pusch.rvIndex = ul_profile["rvIndex"].as<uint32_t>();
        static_params.pusch.harqProcessID = ul_profile["harqProcessID"].as<uint32_t>();
        static_params.pusch.newDataIndicator = ul_profile["newDataIndicator"].as<uint32_t>();
        static_params.pusch.numCb = ul_profile["numCb"].as<uint32_t>();
        static_params.pusch.cbPresentAndPosition = ul_profile["cbPresentAndPosition"].as<uint32_t>();

        yaml_try_parse_list(ul_profile, "numDmrsCdmGrpsNoData", static_params.pusch.numDmrsCdmGrpsNoData);
        yaml_try_parse_list(ul_profile, "rbBitmap", static_params.pusch.rbBitmap);

        // TODO: Hard-code beamforming to 0 currently
        rx_beamforming_data_t& rx_beam = static_params.pusch.rx_beam_data;
        rx_beam.numPRGs = 0;
        rx_beam.prgSize = 0;
        rx_beam.digBFInterfaces = 0;
    }

    yaml::node slot_pattern = root_pattern["slot_pattern"];
    sched_slot_num = slot_pattern.length();
    dynamic_slot_params.resize(sched_slot_num);
    NVLOGC_FMT(TAG, "{}: channels=0x{:02X} slot_num={}", __func__, channel_mask, sched_slot_num);

    for(int slot_idx = 0; slot_idx < sched_slot_num; slot_idx++)
    {
        std::string                    slot_name           = "Slot " + std::to_string(slot_idx);
        yaml::node                     slot_node           = slot_pattern[slot_name.c_str()];
        std::vector<dyn_slot_param_t>& dynamic_cell_params = dynamic_slot_params[slot_idx];
        size_t                         dyn_cell_num        = slot_node.length();
        dynamic_cell_params.resize(dyn_cell_num);
        for(int cell_idx = 0; cell_idx < dyn_cell_num; cell_idx++)
        {
            dyn_slot_param_t& dyn_param = dynamic_cell_params[cell_idx];
            yaml::node        cell_node = slot_node[std::to_string(cell_idx).c_str()];

            std::string dir = cell_node["Direction"].as<std::string>();
            if(strncmp(dir.c_str(), "DL", 2) == 0)
            {
                // Swap PDSCH and PUSCH for UE dynamic mode
                dyn_param.ch_type = yaml_configs->app_mode == 1 ? channel_type_t::PDSCH : channel_type_t::PUSCH;
            }
            else if(strncmp(dir.c_str(), "UL", 2) == 0)
            {
                // Swap PDSCH and PUSCH for UE dynamic mode
                dyn_param.ch_type = yaml_configs->app_mode == 1 ? channel_type_t::PUSCH : channel_type_t::PDSCH;
            }
            else
            {
                NVLOGE_FMT(TAG, AERIAL_YAML_PARSER_EVENT, "Error Direction: {}", dir.c_str());
            }

            yaml::node prbs_node = cell_node["PRB"];
            yaml::node rnti_node = cell_node["rnti"];
            yaml::node beam_node = cell_node["Beams"];
            yaml::node layer_node = cell_node["Layer"];
            yaml::node mcs_table_node = cell_node["MCS table"];
            yaml::node mcs_node = cell_node["MCS"];
            yaml::node dmrs_port_bmsk_node = cell_node["dmrsPortBmsk"];
            yaml::node dmrs_sym_loc_bmsk_node = cell_node["dmrsSymLocBmsk"];
            yaml::node nrOfSymbols_node = cell_node["nrOfSymbols"];

            dyn_param.pdus.resize(rnti_node.length());
            for (uint32_t pdu_id = 0; pdu_id < rnti_node.length(); pdu_id ++)
            {
                dyn_pdu_param_t& pdu_param = dyn_param.pdus[pdu_id];
                pdu_param.prb.prbStart = prbs_node[pdu_id][0UL].as<uint32_t>();
                pdu_param.prb.prbEnd   = prbs_node[pdu_id][1UL].as<uint32_t>();
                pdu_param.rnti = rnti_node[pdu_id].as<uint32_t>();
                pdu_param.beam = beam_node[pdu_id].as<uint32_t>();
                pdu_param.layer = layer_node[pdu_id].as<uint32_t>();
                pdu_param.mcs_table = mcs_table_node[pdu_id].as<uint32_t>();
                pdu_param.mcs = mcs_node[pdu_id].as<uint32_t>();
                pdu_param.dmrs_port_bmsk = dmrs_port_bmsk_node[pdu_id].as<uint32_t>();
                pdu_param.dmrs_sym_loc_bmsk = dmrs_sym_loc_bmsk_node[pdu_id].as<uint32_t>();
                pdu_param.nrOfSymbols = nrOfSymbols_node[pdu_id].as<uint32_t>();
            }

            // Calculate TB size and get modulation_order, target_code_rate
            calculate_dyn_slot_params(dyn_param);
        }
    }

    // Parse static slot parameters
    channel_type_t ch;
    dynamic_slots_pattern.resize(sched_slot_num);
    for(int32_t slot_id = 0; slot_id < sched_slot_num; slot_id++)
    {
        dynamic_slots_pattern[slot_id].resize(cell_num);
        std::vector<dyn_slot_param_t>& dynamic_cell_params = dynamic_slot_params[slot_id];
        for(int cell_id = 0; cell_id < cell_num; cell_id++)
        {
            dynamic_slots_pattern[slot_id][cell_id].resize(FAPI_REQ_SIZE);
            dyn_slot_param_t& dyn_param = dynamic_cell_params[cell_id];
            if(yaml_configs->app_mode != 1 && dyn_param.ch_type == channel_type_t::PUSCH)
            {
                // Skip PUSCH for UE dynamic mode
                continue;
            }

            if((channel_mask & (1 << dyn_param.ch_type )) == 0)
            {
                continue;
            }

            NVLOGC_FMT(TAG, "{}: slot {} cell {} PDUs: {} {}", __func__, slot_id, cell_id, dyn_param.pdus.size(), get_channel_name(dyn_param.ch_type));
            if(add_dyn_slot_channel(dynamic_slots_pattern, cell_id, slot_id, dyn_param.ch_type) < 0)
            {
                return -1;
            }
        }
    }

    return 0;
}

int launch_pattern::apply_reconfig_pattern(int cell_id)
{
    prach_reconfig_flag = 2;

    // Dimensions: slot, cell, fapi_group, channel
    for (std::vector<std::vector<std::vector<fapi_req_t*>>>& new_slot : reconfig_slots_pattern)
    {
        for (std::vector<std::vector<fapi_req_t*>>& new_cell : new_slot)
        {
            for (std::vector<fapi_req_t*> new_group : new_cell)
            {
                for (fapi_req_t* req : new_group)
                {
                    if (cell_id !=  req->cell_idx)
                    {
                        continue;
                    }

                    int slot_id = req->slot_idx;
                    channel_type_t channel = req->channel;

                    fapi_req_t* replaced_req = nullptr;
                    std::vector<std::vector<fapi_req_t*>>& old_groups = sched_slots_pattern[slot_id][cell_id];
                    for (std::vector<fapi_req_t*>& old_group : old_groups)
                    {
                        for (int n = 0; n < old_group.size(); n ++)
                        {
                            fapi_req_t* old_req = old_group[n];
                            if (old_req == req)
                            {
                                NVLOGI_FMT(TAG, "{}: already overwritten {} TV for slot {} cell {}: {} -> {}, skip", __func__, get_channel_name(req->channel), slot_id, cell_id, reinterpret_cast<void*>(old_req), reinterpret_cast<void*>(req));
                            }
                            else if (old_req->channel == req->channel)
                            {
                                if (replaced_req != nullptr && replaced_req != old_req)
                                {
                                    NVLOGE_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "Duplicate {} TV for slot {} cell {}", get_channel_name(req->channel), slot_id, cell_id);
                                }
                                NVLOGI_FMT(TAG, "{}: overwrite {} TV for slot {} cell {}: {} -> {}", __func__, get_channel_name(req->channel), slot_id, cell_id, reinterpret_cast<void*>(old_req), reinterpret_cast<void*>(req));
                                replaced_req = old_req;
                                old_group[n] = req;
                            }
                        }
                    }

                    if (replaced_req != nullptr) // Note: old_req == req at the second time
                    {
                        switch (replaced_req->channel)
                        {
                        case channel_type_t::PRACH:
                            for (prach_tv_data_t prach_pdu : replaced_req->tv_data->prach_tv.data)
                            {
                                expected[cell_id].prmb -= prach_pdu.ref.numPrmb;
                            }
                            for (prach_tv_data_t prach_pdu : req->tv_data->prach_tv.data)
                            {
                                expected[cell_id].prmb += prach_pdu.ref.numPrmb;
                            }
                            break;
                        default:
                            NVLOGE_FMT(TAG, AERIAL_NO_SUPPORT_EVENT, "reconfig {} TV for cell_id={} slot_id={} not supported", get_channel_name(req->channel), slot_id, cell_id);
                            break;
                        }
                        NVLOGD_FMT(TAG, "{}: Updated expected result for {} TV of slot {} cell {}", __func__, get_channel_name(req->channel), slot_id, cell_id);
                        if (replaced_req != req)
                        {
                            NVLOGI_FMT(TAG, "{}: Delete the old {} TV of slot {} cell {}", __func__, get_channel_name(req->channel), slot_id, cell_id);
                            delete replaced_req;
                        }
                    }
                }
            }
        }
    }

    return 0;
}

int launch_pattern::update_expected_values(int cell_id, fapi_req_t *req)
{
    // PUSCH
    if(req->channel == channel_type_t::PUSCH)
    {
        pusch_tv_t& pusch_tv = req->tv_data->pusch_tv;
        
        if (pusch_tv.negTV_enable==1) {
            // Expected One Error per cell per slot.
            expected[cell_id].error++;
        }
        else
        {
            expected[cell_id].error = 0;
        }
        for(const pusch_tv_data_t* tv_data: pusch_tv.data)
        {
            if (tv_data->pduBitmap & 0x01)
            {
                expected[cell_id].ul_thrput += tv_data->tb_size;

                // Add invalid number for CRC fail tbErr = 1 case
                if (tv_data->tbErr != 0)
                {
                    expected[cell_id].mac_ul_drop += tv_data->tb_size;
                    //// remove invalid++ and keep invalid=0////
                    //// expected[cell_id].invalid++;
                    //////////////////////////////////
                }
            }

            if (tv_data->pduBitmap & 0x02)
            {
                if (tv_data->tbErr == 0)
                {
                    if(tv_data->harqAckBitLength > 0)
                    {
                        expected[cell_id].harq += 1;
                    }
                    if(tv_data->csiPart1BitLength > 0)
                    {
                        expected[cell_id].csi1 += 1;
                    }
#ifdef SCF_FAPI_10_04
                    if (tv_data->flagCsiPart2 == std::numeric_limits<uint16_t>::max() && tv_data->numPart2s > 0) {
                        expected[cell_id].csi2 += 1;
                    }
#else
                    if((tv_data->csiPart2BitLength > 0) && (tv_data->csiPart2BitLength != 255) && (tv_data->csiPart2BitLength < 1707)) // check valid range of csiPart2BitLength
                    {
                        expected[cell_id].csi2 += 1;
                    }
#endif
                }
            }

            if (tv_data->pduBitmap & 0x03) {

            }
        }
    }

    // PDSCH
    if(req->channel == channel_type_t::PDSCH)
    {
        pdsch_tv_t& pdsch_tv = req->tv_data->pdsch_tv;
        expected[cell_id].dl_thrput += pdsch_tv.data_size;
    }

    // PRACH
    if(req->channel == channel_type_t::PRACH)
    {
        if (prach_reconfig_flag == 0)
        {
            prach_tv_t&   prach_tv = req->tv_data->prach_tv;
            for(const auto& prach: prach_tv.data)
            {
                expected[cell_id].prmb += prach.ref.numPrmb;
            }
        }
    }

    // PUCCH
    if(req->channel == channel_type_t::PUCCH)
    {
        pucch_tv_t& pucch_tv = req->tv_data->pucch_tv;
        for(const auto& pucch: pucch_tv.data)
        {
            if(pucch.SRFlag != 0 && pucch.SRindication != 0)
            {
                // PUCCH PF0/1
                expected[cell_id].sr += 1;
            }
            if(pucch.bitLenSr > 0)
            {
                // PUCCH PF2/3/4
                expected[cell_id].sr += 1;
            }
            if(pucch.BitLenHarq > 0)
            {
                expected[cell_id].harq += 1;
            }
            if(pucch.BitLenCsiPart1 > 0)
            {
                expected[cell_id].csi1 += 1;
            }
            if(pucch.BitLenCsiPart2 > 0)
            {
                expected[cell_id].csi2 += 1;
            }
        }
    }

    // SRS
    if(req->channel == channel_type_t::SRS)
    {
        srs_tv_t& srs_tv = req->tv_data->srs_tv;
        for(const auto& srs: srs_tv.data)
        {
            expected[cell_id].srs += 1;
        }
    }

    // Skip while parsing PRACH "RECONFIG"
    if (prach_reconfig_flag != 0)
    {
        return 0;
    }

    // CSI_RS: skip if CSIType is ZP_CSI_RS in all CSI_RS PDUs
    if (req->channel == channel_type_t::CSI_RS) {
        bool all_are_zp = true;
        csirs_tv_t& csirs_tv = req->tv_data->csirs_tv;
        for(const csirs_tv_data_t& csirs: csirs_tv.data)
        {
            if (csirs.CSIType != ZP_CSI_RS)
            {
                all_are_zp = false;
                break;
            }
        }
        if (all_are_zp) {
            return 0;
        }
    }

    // Add expected slots for this channel
    expected[cell_id].lp_slots[req->channel]++;

    return 0;
}


int launch_pattern::add_dyn_slot_channel(slot_pattern_t& slots_data, int cell_id, int slot_id, channel_type_t ch)
{
    fapi_req_t* req = new fapi_req_t;
    req->cell_idx   = cell_id;
    req->slot_idx   = slot_id;
    req->channel    = ch;
    req->tv_file    = "DYN_JSON_NO_TV";

    switch(req->channel)
    {
    case channel_type_t::PDSCH:
        slots_data[slot_id][cell_id][DL_TTI_REQ].push_back(req);
        slots_data[slot_id][cell_id][TX_DATA_REQ].push_back(req);
        break;
    case channel_type_t::PUSCH:
        slots_data[slot_id][cell_id][UL_TTI_REQ].push_back(req);
        // Pre-allocate vector buffer for saving harq_pid for validation
        hpid_maps[cell_id][slot_id].resize(dynamic_slot_params[slot_id][cell_id].pdus.size());
        break;
    default:
        NVLOGE_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "{}: channel not supported yet: {}",
                __FUNCTION__, get_channel_name(req->channel));
        delete req;
        return -1;
    }
    NVLOGI_FMT(TAG, "{}: added tv: slot {} {:<8} {}",
            __FUNCTION__, slot_id, get_channel_name(ch), req->tv_file.c_str());
    return 0;
}

int launch_pattern::parse_tv_channels(slot_pattern_t& slots_data, int cell_id, int slot_id, int ch_mask, std::string tv_file, std::vector<std::string>& channel_type_list)
{
    if(tv_file.length() == 0)
    {
        return -1;
    }

    for (int ch = 0; ch < CHANNEL_MAX; ch++)
    {
        if(((ch_mask & (1 << ch)) == 0) ||
          ((channel_type_list.size() > 0) &&
          (channel_type_list.end() == std::find (channel_type_list.begin(), channel_type_list.end(), get_channel_name(ch)))))
        {
            // Skip channel which not enabled
            continue;
        }

        fapi_req_t* req = new fapi_req_t;
        req->cell_idx   = cell_id;
        req->slot_idx   = slot_id;
        req->channel    = (channel_type_t)ch;
        req->tv_file    = tv_file;

        curr_ch = req->channel;
        curr_tv = req->tv_file;

        if (h5file_map.find(tv_file) == h5file_map.end())
        {
            char file_path_array[MAX_PATH_LEN];
            get_full_path_file(file_path_array, CONFIG_TEST_VECTOR_PATH, tv_file.c_str(), CONFIG_CUBB_ROOT_DIR_RELATIVE_NUM);
            std::filesystem::path file_path(file_path_array);
            if(access(file_path.c_str(), F_OK) != 0)
            {
                NVLOGF_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "TV cell {} slot {} {} {} file not exist: {}",
                        curr_cell, curr_slot, get_channel_name(curr_ch), curr_tv.c_str(), file_path.c_str());
                delete req;
                return -1;
            }

            try
            {
                h5file_map[tv_file] = hdf5hpp::hdf5_file::open(file_path.c_str());
            }
            catch(std::exception& e)
            {
                NVLOGE_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "TV cell {} slot {} {} {} hdf5_file::open failed",
                        curr_cell, curr_slot, get_channel_name(curr_ch), curr_tv.c_str());
                delete req;
                return -1;
            }
        }
        req->h5f = &h5file_map[tv_file];

        if(tv_data_maps[ch].find(tv_file) == tv_data_maps[ch].end())
        {
            req->tv_data = new test_vector_t;
            if(parse_tv_file(req) != 0)
            {
                delete req->tv_data;
                delete req;
                continue;
            }

            bool cache_channels = (
                                ch == channel_type_t::PBCH ||
                                ch == channel_type_t::PDCCH_DL ||
                                ch == channel_type_t::PDCCH_UL ||
                                // ch == channel_type_t::PDSCH ||
                                ch == channel_type_t::CSI_RS ||
                                // ch == channel_type_t::PUSCH ||
                                // ch == channel_type_t::PUCCH ||
                                // ch == channel_type_t::PRACH ||
                                ch == channel_type_t::SRS
                                );

            // Save to TV data map if enabled
            if (yaml_configs->tv_data_map_enable != 0 && cache_channels)
            {
                tv_data_maps[ch][tv_file] = req->tv_data;
            }
        }
        else
        {
            req->tv_data = tv_data_maps[ch][tv_file];
        }

        update_expected_values(cell_id, req);

        switch(req->channel)
        {
        case channel_type_t::PBCH:
            slots_data[slot_id][cell_id][DL_TTI_REQ].push_back(req);
            break;
        case channel_type_t::PDCCH_DL:
            slots_data[slot_id][cell_id][DL_TTI_REQ].push_back(req);
            break;
        case channel_type_t::PDSCH:
            slots_data[slot_id][cell_id][DL_TTI_REQ].push_back(req);
            slots_data[slot_id][cell_id][TX_DATA_REQ].push_back(req);
            break;
        case channel_type_t::PDCCH_UL:
            slots_data[slot_id][cell_id][UL_DCI_REQ].push_back(req);
            break;
        case channel_type_t::PUSCH:
            slots_data[slot_id][cell_id][UL_TTI_REQ].push_back(req);
            // Pre-allocate vector buffer for saving harq_pid for validation
            hpid_maps[cell_id][slot_id].resize(req->tv_data->pusch_tv.data.size());
            break;
        case channel_type_t::PRACH:
            slots_data[slot_id][cell_id][UL_TTI_REQ].push_back(req);
            break;
        case channel_type_t::PUCCH:
            slots_data[slot_id][cell_id][UL_TTI_REQ].push_back(req);
            break;
        case channel_type_t::SRS:
            slots_data[slot_id][cell_id][UL_TTI_REQ].push_back(req);
            break;
        case channel_type_t::BFW_DL:
            slots_data[slot_id][cell_id][DL_BFW_CVI_REQ].push_back(req);
            break;
        case channel_type_t::BFW_UL:
            slots_data[slot_id][cell_id][UL_BFW_CVI_REQ].push_back(req);
            break;
        case channel_type_t::CSI_RS:
            slots_data[slot_id][cell_id][DL_TTI_REQ].push_back(req);
            break;
        default:
            NVLOGE_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "{}: channel not supported yet: {}",
                    __FUNCTION__, get_channel_name(req->channel));
            delete req;
            break;
        }
        NVLOGI_FMT(TAG, "{}: added tv: slot {} {:<8} {}",
                __FUNCTION__, slot_id, get_channel_name(ch), req->tv_file.c_str());
    }
    return 0;

}

// Parse beam_ids if exists, else set current_beam_idx_v to empty
int launch_pattern::parse_beam_ids(yaml::node map_node)
{
    if (map_node.has_key("beam_ids"))
    {
        yaml::node beam_ids = map_node["beam_ids"];
        current_beam_idx_v.resize(beam_ids.length());
        for(int n = 0; n < beam_ids.length(); ++n)
        {
            uint16_t beam_id = beam_ids[n].as<unsigned int>();
            current_beam_idx_v[n] = beam_id;
            NVLOGI_FMT(TAG, "cell {} slot {} beam_ids[{}-{}]={}", curr_cell, curr_slot, beam_ids.length(), n, beam_id);
        }
    }
    else
    {
        current_beam_idx_v.resize(0);
    }
    return 0;
}

int launch_pattern::populate_launch_pattern(slot_pattern_t& slots_data, yaml::node cell_config, int cell_id, int slot_idx)
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

    parse_beam_ids(cell_config);

    for(int i = 0; i < channel_list.length(); ++i)
    {
        NVLOGV_FMT(TAG, "slot {} channel_list[{}].type_string(): {}", slot_idx, i, channel_list[i].type_string());
        channel_type_list.clear();
        // New launch pattern format
        if (channel_list[i].type() == YAML_SCALAR_NODE)
        {
            if(cell_config.has_key("type"))
            {
                yaml::node channel_type_node = cell_config["type"];
                for (std::size_t j = 0; j < channel_type_node.length(); j++)
                {
                    channel_type_t channel = get_channel_type(channel_type_node[j].as<std::string>().c_str());
                    channel_type_list.push_back(channel_type_node[j].as<std::string>());
                }
            }
            std::string tv_file = channel_list[i].as<std::string>();
            parse_tv_channels(slots_data, cell_id, slot_idx, channel_mask, tv_file, channel_type_list);
            continue;
        }

        // Old launch pattern format
        parse_beam_ids(channel_list[i]);

        std::string channel_name = channel_list[i]["type"].as<std::string>();
        std::string tv_key       = channel_list[i]["tv"].as<std::string>();
        channel_type_t channel = get_channel_type(channel_name.c_str());

        if(channel == channel_type_t::CHANNEL_MAX)
        {
            NVLOGE_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "{}: unknown channel_type in launch pattern file: {}", __FUNCTION__, channel_name.c_str());
            continue;
        }

        std::unordered_map<std::string, std::string>& tv_map  = *(tv_maps[channel]);
        std::string                                   tv_file = tv_map[tv_key];
        std::string                                   exempted_tv("demo_msg1_gNB_FAPI_s15.h5");
        if(0 == exempted_tv.compare(tv_file))
        {
            yaml_configs->validate_enable = 0;
        }

        if((channel_mask & (1 << channel)) == 0)
        {
            // Skip channel which not enabled
            NVLOGI_FMT(TAG, "{}: skip channel {} - {}", __FUNCTION__, get_channel_name(channel), tv_file.c_str());
            continue;
        }
        if (parse_tv_channels(slots_data, cell_id, slot_idx, 1 << channel, tv_file, channel_type_list) < 0)
        {
            return -1;
        }
    }

    return 0;
}

static inline channel_type_t to_channel_type(uint32_t val)
{
    switch(val)
    {
    case tv_channel_type_t::TV_PBCH: return channel_type_t::PBCH;
    case tv_channel_type_t::TV_PDCCH: return channel_type_t::PDCCH_DL;
    case tv_channel_type_t::TV_PDSCH: return channel_type_t::PDSCH;
    case tv_channel_type_t::TV_CSI_RS: return channel_type_t::CSI_RS;
    case tv_channel_type_t::TV_PRACH: return channel_type_t::PRACH;
    case tv_channel_type_t::TV_PUCCH: return channel_type_t::PUCCH;
    case tv_channel_type_t::TV_PUSCH: return channel_type_t::PUSCH;
    case tv_channel_type_t::TV_SRS: return channel_type_t::SRS;
    case tv_channel_type_t::TV_BFW_DL: return channel_type_t::BFW_DL;
    case tv_channel_type_t::TV_BFW_UL: return channel_type_t::BFW_UL;
    case tv_channel_type_t::TV_PDCCH_UL: return channel_type_t::PDCCH_UL;
    default:
        break;
    }
    return channel_type_t::CHANNEL_MAX;
}

int launch_pattern::parse_pdu_dset_names(hdf5hpp::hdf5_file& file, tv_dset_list& list)
{
    int ret = 0;

    if(!file.is_valid_dataset(NUMPDU))
    {
        return -1;
    }
    auto     dset = file.open_dataset(NUMPDU);
    uint32_t nPdu = 0;
    dset.read(&nPdu);
    for(uint i = 0; i < nPdu; i++)
    {
        std::string name = PDU + std::to_string(i + 1);
        if(!file.is_valid_dataset(name.c_str()))
        {
            continue;
        }
        auto     pdu_ds   = file.open_dataset(name.c_str())[0];
        uint32_t pdu_type = pdu_ds["type"].as<uint32_t>();
        if(pdu_type == tv_channel_type_t::TV_PDCCH)
        {
            uint32_t ulDCI = pdu_ds["dciUL"].as<uint32_t>();
            if(ulDCI == 1)
            {
                pdu_type = tv_channel_type_t::TV_PDCCH_UL;
            }
        }
        if(pdu_type == tv_channel_type_t::TV_BFW_DL)
        {
            uint32_t bfwUL = pdu_ds["bfwUL"].as<uint32_t>();
            //NVLOGW_FMT(TAG, "bfwUL= {}\n", bfwUL);
            if(bfwUL == 1)
            {
                pdu_type = tv_channel_type_t::TV_BFW_UL;
            }
        }
        channel_type_t chType = to_channel_type(pdu_type);
        if(chType >= channel_type_t::CHANNEL_MAX)
        {
            continue;
        }
        list[chType].push_back(name);
    }
    if(list.empty())
    {
        return -1;
    }
    return 0;
}

int launch_pattern::save_harq_pid(int cell_id, uint16_t sfn, uint16_t slot, int pdu_id, uint8_t hpid)
{
    int ret = 0;
    std::unordered_map<uint32_t, std::vector<uint8_t>>& m = hpid_maps[cell_id];
    uint32_t sfn_slot = get_slot_in_frame(sfn, slot) % get_slot_cell_patterns(0).size();

    if (m.find(sfn_slot) == m.end())
    {
        NVLOGW_FMT(TAG, "{}: SFN {}.{} cell {} hpid_maps[{}] not found",
                __func__, sfn, slot, cell_id, sfn_slot);
        ret = -1;
    }

    std::vector<uint8_t>& hpid_v = m[sfn_slot];
    if (pdu_id < hpid_v.size())
    {
        hpid_v[pdu_id] = hpid;
        NVLOGV_FMT(TAG, "{}: SFN {}.{} cell {} hpid_maps[{}][{}]={}",
                __func__, sfn, slot, cell_id, sfn_slot, pdu_id, hpid);
    }
    else
    {
        NVLOGW_FMT(TAG, "{}: SFN {}.{} cell {} hpid_maps[{}][{}] vector size overflow",
                __func__, sfn, slot, cell_id, sfn_slot, pdu_id);
        ret = -1;
    }
    return ret;
}

int launch_pattern::read_harq_pid(int cell_id, uint16_t sfn, uint16_t slot, int pdu_id, uint8_t* hpid)
{
    int ret = 0;
    std::unordered_map<uint32_t, std::vector<uint8_t>>& m = hpid_maps[cell_id];
    uint32_t sfn_slot = get_slot_in_frame(sfn, slot) % get_slot_cell_patterns(0).size();

    if (m.find(sfn_slot) == m.end())
    {
        NVLOGW_FMT(TAG, "{}: SFN {}.{} cell {} hpid_maps[{}] not found",
                __func__, sfn, slot, cell_id, sfn_slot);
        ret = -1;
    }

    std::vector<uint8_t>& hpid_v = m[sfn_slot];
    if (pdu_id < hpid_v.size())
    {
        *hpid = hpid_v[pdu_id];
        NVLOGV_FMT(TAG, "{}: SFN {}.{} cell {} hpid_maps[{}][{}]={}",
                __func__, sfn, slot, cell_id, sfn_slot, pdu_id, *hpid);
    }
    else
    {
        NVLOGW_FMT(TAG, "{}: SFN {}.{} cell {} hpid_maps[{}][{}] vector size overflow",
                __func__, sfn, slot, cell_id, sfn_slot, pdu_id);
        *hpid = 0xFE; // Set to an error value
        ret = -1;
    }
    return ret;
}

static const char* CSI2_MAPS = "nCsi2Maps";
static const char* CSI2_MAP_PRMS = "csi2MapPrms";

void launch_pattern::read_csip2_maps(hdf5hpp::hdf5_file & file, int cell_id) {
#ifdef SCF_FAPI_10_04

    NVLOGD_FMT(TAG, "is_valid_dataset(CSI2_MAPS) {} is_valid_dataset(CSI2_MAP_PRMS) {}", file.is_valid_dataset(CSI2_MAPS), file.is_valid_dataset(CSI2_MAP_PRMS));

    if (!file.is_valid_dataset(CSI2_MAPS) || !file.is_valid_dataset(CSI2_MAP_PRMS)) {
        return;
    }

    double nCsiMaps = 0;

    csi2_maps_t inst{0, {}, 0};

    hdf5hpp::hdf5_dataset nCsi2MapsDset = file.open_dataset(CSI2_MAPS);
    nCsi2MapsDset.read(&nCsiMaps);

    inst.nCsi2Maps = static_cast<uint32_t> (nCsiMaps);
    hdf5hpp::hdf5_dataset nCsi2MapParamsDset = file.open_dataset(CSI2_MAP_PRMS);

    for (uint32_t i = 0 ; i < inst.nCsi2Maps; i++) {
        auto nPart1Prms    = nCsi2MapParamsDset[i]["numPart1Params"].as<uint8_t>();
        if (!nPart1Prms) {
            continue;
        }
        auto prmSize = nCsi2MapParamsDset[i]["sizesPart1Params"].as<std::vector<uint8_t>>();
        auto mapBitWidth     = nCsi2MapParamsDset[i]["mapBitWidth"].as<uint8_t>();
        std::vector<uint16_t>maps = nCsi2MapParamsDset[i]["map"].as<std::vector<uint16_t>>();
        // for (size_t mapIdx = 0; mapIdx < maps.size(); mapIdx++) {
        //     NVLOGD_FMT(TAG, "csip2Map {} mapIdx {}, value {}", i, mapIdx, maps[mapIdx]);
        // }
        inst.mapParams.push_back({nPart1Prms, std::move(prmSize), mapBitWidth, std::move(maps)});
        auto& last = inst.mapParams.back();
        // Size of 2 * sizeof(uint8_t) for numPart1Params, sizePart1Params.. Not accouting mapBitWidth
        // since mapBitWidth is not a field in FAPI 10.04
        inst.totalSizeInBytes += sizeof(uint16_t) + 2 * sizeof(uint8_t) + sizeof(uint16_t) * last.map.size();
        NVLOGD_FMT(TAG, "nCsi2Maps {} numPart1Params {} sizesPart1Params {} mapBitWidth {} maps size {}", inst.nCsi2Maps, last.numPart1Params, last.sizePart1Params.size(), last.mapBitWidth, last.map.size());
    }
    csi2maps.reserve(cell_id);
    csi2maps.push_back(std::move(inst));
#endif
}

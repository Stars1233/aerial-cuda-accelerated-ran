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

#include <cuda_runtime_api.h>

#include "api.h"
#include "cumac_task.hpp"

#include "hdf5hpp.hpp"
#include "hdf5.h"
#include "cuphy_hdf5.hpp"
#include "nvlog.hpp"

#include "cumac_cp_tv.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <string>
#include <utility>
#include <vector>

#define TAG (NVLOG_TAG_BASE_CUMAC_CP + 2) // "CUMCP.CFG"

inline constexpr int  MAX_SRS_CHEST_BUFFERS_PER_CELL = 1024;
inline constexpr int  MAX_SRS_CHEST_BUFFERS_PER_4T4R_CELL = 256;
// TBD: Change to max cells once we support mutiple Cells with MU-MIMO
inline constexpr uint32_t MAX_CELLS_MU_MIMO_ENABLE = 9;
inline constexpr uint32_t MAX_SRS_PDU_PER_SLOT = 128;
inline constexpr uint32_t MAX_SRS_CHEST_BUFFERS = MAX_CELLS_MU_MIMO_ENABLE * MAX_SRS_CHEST_BUFFERS_PER_CELL;
inline constexpr uint32_t num_srs_buffers_per_cell = 1024;

void ue_pair_tv_release_mu_host_buffers(ue_pair_tv_t &t)
{
    if (t.srs_chan_est_host != nullptr)
    {
        cudaFreeHost(t.srs_chan_est_host);
    }
    t.srs_chan_est_host = nullptr;
    t.srs_chan_est_size = 0;

    if (t.srs_snr_host != nullptr)
    {
        cudaFreeHost(t.srs_snr_host);
    }
    t.srs_snr_host = nullptr;
    t.srs_snr_size = 0;

    if (t.chan_orth_host != nullptr)
    {
        cudaFreeHost(t.chan_orth_host);
    }
    t.chan_orth_host = nullptr;
    t.chan_orth_size = 0;

    if (t.cubb_srs_buf_host != nullptr)
    {
        cudaFreeHost(t.cubb_srs_buf_host);
    }
    t.cubb_srs_buf_host = nullptr;
    t.cubb_srs_buf_size = 0;

    if (t.task_in_buf_group_host != nullptr)
    {
        cudaFreeHost(t.task_in_buf_group_host);
    }
    t.task_in_buf_group_host = nullptr;
    t.task_in_buf_group_size = 0;
}

ue_pair_tv::~ue_pair_tv()
{
    ue_pair_tv_release_mu_host_buffers(*this);
}

ue_pair_tv::ue_pair_tv(ue_pair_tv &&o) noexcept
    : mu_ue_pair_tv_loaded(o.mu_ue_pair_tv_loaded),
      num_prg(o.num_prg),
      num_subband(o.num_subband),
      num_prg_samp_per_subband(o.num_prg_samp_per_subband),
      num_bs_ant(o.num_bs_ant),
      num_ue_ant(o.num_ue_ant),
      num_srs_ue_per_slot_cell(o.num_srs_ue_per_slot_cell),
      num_blocks_per_row_chanOrtMat(o.num_blocks_per_row_chanOrtMat),
      kernel_launch_flags(o.kernel_launch_flags),
      is_mem_sharing(o.is_mem_sharing),
      muUeGrpSol(std::move(o.muUeGrpSol)),
      srs_chan_est_host(o.srs_chan_est_host),
      srs_chan_est_size(o.srs_chan_est_size),
      srs_snr_host(o.srs_snr_host),
      srs_snr_size(o.srs_snr_size),
      chan_orth_host(o.chan_orth_host),
      chan_orth_size(o.chan_orth_size),
      cubb_srs_buf_host(o.cubb_srs_buf_host),
      cubb_srs_buf_size(o.cubb_srs_buf_size),
      task_in_buf_group_host(o.task_in_buf_group_host),
      task_in_buf_group_size(o.task_in_buf_group_size)
{
    o.mu_ue_pair_tv_loaded = false;
    o.num_ue_ant = 0;
    o.srs_chan_est_host = nullptr;
    o.srs_chan_est_size = 0;
    o.srs_snr_host = nullptr;
    o.srs_snr_size = 0;
    o.chan_orth_host = nullptr;
    o.chan_orth_size = 0;
    o.cubb_srs_buf_host = nullptr;
    o.cubb_srs_buf_size = 0;
    o.task_in_buf_group_host = nullptr;
    o.task_in_buf_group_size = 0;
}

ue_pair_tv &ue_pair_tv::operator=(ue_pair_tv &&o) noexcept
{
    if (this == &o)
    {
        return *this;
    }
    ue_pair_tv_release_mu_host_buffers(*this);
    mu_ue_pair_tv_loaded = o.mu_ue_pair_tv_loaded;
    num_prg = o.num_prg;
    num_subband = o.num_subband;
    num_prg_samp_per_subband = o.num_prg_samp_per_subband;
    num_bs_ant = o.num_bs_ant;
    num_ue_ant = o.num_ue_ant;
    num_srs_ue_per_slot_cell = o.num_srs_ue_per_slot_cell;
    num_blocks_per_row_chanOrtMat = o.num_blocks_per_row_chanOrtMat;
    kernel_launch_flags = o.kernel_launch_flags;
    is_mem_sharing = o.is_mem_sharing;
    muUeGrpSol = std::move(o.muUeGrpSol);
    srs_chan_est_host = o.srs_chan_est_host;
    srs_chan_est_size = o.srs_chan_est_size;
    srs_snr_host = o.srs_snr_host;
    srs_snr_size = o.srs_snr_size;
    chan_orth_host = o.chan_orth_host;
    chan_orth_size = o.chan_orth_size;
    cubb_srs_buf_host = o.cubb_srs_buf_host;
    cubb_srs_buf_size = o.cubb_srs_buf_size;
    task_in_buf_group_host = o.task_in_buf_group_host;
    task_in_buf_group_size = o.task_in_buf_group_size;

    o.mu_ue_pair_tv_loaded = false;
    o.num_ue_ant = 0;
    o.srs_chan_est_host = nullptr;
    o.srs_chan_est_size = 0;
    o.srs_snr_host = nullptr;
    o.srs_snr_size = 0;
    o.chan_orth_host = nullptr;
    o.chan_orth_size = 0;
    o.cubb_srs_buf_host = nullptr;
    o.cubb_srs_buf_size = 0;
    o.task_in_buf_group_host = nullptr;
    o.task_in_buf_group_size = 0;
    return *this;
}

namespace {

static void load_mu_ue_pair_fail(ue_pair_tv_t &u)
{
    u.mu_ue_pair_tv_loaded = false;
    u.muUeGrpSol.clear();
    ue_pair_tv_release_mu_host_buffers(u);
}

static int mu_pin_alloc(void **p, size_t *out_bytes, size_t nbytes)
{
    *p = nullptr;
    *out_bytes = 0;
    if (nbytes == 0)
    {
        return 0;
    }
    if (cudaMallocHost(p, nbytes) != cudaSuccess)
    {
        NVLOGE_FMT(TAG, AERIAL_CUMAC_CP_EVENT, "cudaMallocHost failed ({} bytes)", nbytes);
        return -1;
    }
    *out_bytes = nbytes;
    return 0;
}

} // namespace

#define CHECK_VALUE_EQUAL_ERR(v1, v2)                                                                                              \
    do                                                                                                                             \
    {                                                                                                                              \
        if ((v1) != (v2))                                                                                                          \
        {                                                                                                                          \
            NVLOGE_FMT(TAG, AERIAL_CUMAC_CP_EVENT, "{} line {}: values doesn't equal: v1={} > v2={}", __func__, __LINE__, v1, v2); \
        }                                                                                                                          \
    } while (0);

#define CONFIG_CUMAC_TV_PATH "testVectors/cumac/"
#define CONFIG_MU_UE_PAIR_TV_FORMAT "muUePairTV_sfn%d_slot%d"

using namespace std;

namespace {

template <typename T>
static bool h5_read_attr_scalar(hid_t loc_id, const char* name, hid_t mem_type, T* out)
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

//! Matches slot_command_api::MAX_SRS_CHEST_BUFFERS used in cumac_muUeGrp_test.cu for total_num_buffers cap.
constexpr uint32_t kMuTvMaxSrsChestBuffers = 6u * 1024u;

//! Expected srs_chan_est_buf byte count (cumac_muUeGrp_test.cu srs_chan_est_buf_total_size).
static size_t mu_tv_expected_srs_chan_est_bytes(uint16_t n_bs_ant, uint16_t n_sub, uint16_t n_prg_samp, int cell_num)
{
    const uint64_t v = static_cast<uint64_t>(sizeof(__half2)) * static_cast<uint64_t>(n_bs_ant)
        * static_cast<uint64_t>(MAX_NUM_UE_ANT_PORT) * static_cast<uint64_t>(n_sub)
        * static_cast<uint64_t>(n_prg_samp) * static_cast<uint64_t>(MAX_NUM_SRS_UE_PER_CELL)
        * static_cast<uint64_t>(cell_num);
    return static_cast<size_t>(v);
}

//! Expected srs_snr_buf byte count (cumac_muUeGrp_test.cu srs_snr_buf_total_size).
static size_t mu_tv_expected_srs_snr_bytes(int cell_num)
{
    const uint64_t v = static_cast<uint64_t>(sizeof(float)) * static_cast<uint64_t>(MAX_NUM_SRS_UE_PER_CELL)
        * static_cast<uint64_t>(cell_num);
    return static_cast<size_t>(v);
}

//! Expected chan_orth_mat_buf byte count (cumac_muUeGrp_test.cu chan_orth_mat_buf_total_size).
static size_t mu_tv_expected_chan_orth_bytes(uint16_t n_sub, uint16_t n_prg_samp, int cell_num)
{
    const uint64_t n_pair = static_cast<uint64_t>(MAX_NUM_SRS_UE_PER_CELL) * static_cast<uint64_t>(MAX_NUM_UE_ANT_PORT);
    const uint64_t tri = n_pair * (n_pair + 1u) / 2u;
    const uint64_t v = static_cast<uint64_t>(sizeof(float)) * tri * static_cast<uint64_t>(n_sub)
        * static_cast<uint64_t>(n_prg_samp) * static_cast<uint64_t>(cell_num);
    return static_cast<size_t>(v);
}

static bool mu_tv_check_dataset_bytes(const char *ds_name, size_t actual, size_t expected)
{
    if (actual == expected)
        return true;
    NVLOGE_FMT(TAG, AERIAL_INVALID_PARAM_EVENT,
               "load_mu_ue_pair_group_tv: {} size mismatch: TV has {} bytes, expected {} bytes "
               "(same formula as cuMAC/examples/muMimoUeGrpL2Integration/cumac_muUeGrp_test.cu)",
               ds_name, actual, expected);
    return false;
}

//! cubb_srs_gpu_buf_total_size = buffer_size * total_num_buffers with buffer_size = num_prg * num_gnb_ant * num_ue_layer * sizeof(uint32_t).
static bool mu_tv_validate_cubb_srs_gpu_buf_size(size_t dim, uint16_t n_prg, uint8_t n_bs_ant, int32_t num_ue_ant_port_from_tv,
                                                  bool has_num_ue_ant_tv, bool has_num_srs_buffers, int32_t num_srs_buffers)
{
    const uint32_t ue_layers =
        (has_num_ue_ant_tv && num_ue_ant_port_from_tv > 0) ? static_cast<uint32_t>(num_ue_ant_port_from_tv) : MAX_NUM_UE_ANT_PORT;
    const uint64_t buffer_size = static_cast<uint64_t>(n_prg) * static_cast<uint64_t>(n_bs_ant)
        * static_cast<uint64_t>(ue_layers) * sizeof(uint32_t);
    if (buffer_size == 0)
    {
        NVLOGE_FMT(TAG, AERIAL_INVALID_PARAM_EVENT,
                   "load_mu_ue_pair_group_tv: cubb_srs_gpu_buf derived buffer_size is 0 (n_prg={} n_bs_ant={} ue_layers={})",
                   n_prg, static_cast<unsigned>(n_bs_ant), ue_layers);
        return false;
    }
    if (dim % buffer_size != 0)
    {
        NVLOGE_FMT(TAG, AERIAL_INVALID_PARAM_EVENT,
                   "load_mu_ue_pair_group_tv: cubb_srs_gpu_buf {} bytes is not a multiple of per-buffer size {} "
                   "(num_prg*num_bs_ant*num_ue_layer*sizeof(uint32_t), cf. cumac_muUeGrp_test.cu)",
                   dim, buffer_size);
        return false;
    }
    const uint64_t nbuf = dim / buffer_size;
    if (nbuf > kMuTvMaxSrsChestBuffers)
    {
        NVLOGE_FMT(TAG, AERIAL_INVALID_PARAM_EVENT,
                   "load_mu_ue_pair_group_tv: cubb_srs_gpu_buf implies {} SRS chest buffers, exceeds cap {}",
                   nbuf, kMuTvMaxSrsChestBuffers);
        return false;
    }
    if (has_num_srs_buffers)
    {
        const int64_t capped =
            std::min(static_cast<int64_t>(num_srs_buffers), static_cast<int64_t>(kMuTvMaxSrsChestBuffers));
        if (capped < 0 || static_cast<uint64_t>(capped) != nbuf)
        {
            NVLOGE_FMT(TAG, AERIAL_INVALID_PARAM_EVENT,
                       "load_mu_ue_pair_group_tv: cubb_srs_gpu_buf implies {} buffers; expected {} = "
                       "min(num_srs_buffers HDF5 attr={}, kMuTvMaxSrsChestBuffers={})",
                       nbuf, capped, num_srs_buffers, kMuTvMaxSrsChestBuffers);
            return false;
        }
    }
    return true;
}

} // namespace

using namespace std::chrono;

// Current parsing cell_id, slot_id, channel and TV file name for debug log
static int curr_cell;
static int curr_slot;
static int curr_task;
static std::string curr_tv;

static const char *CUMAC_CP_TV = "CUMAC_CP";
static inline const char *get_task_name(int task_type)
{
    return CUMAC_CP_TV;
}

int check_bytes(const char *name1, const char *name2, void *buf1, void *buf2, size_t nbytes)
{
    int check_result = 0;
    if (buf1 == nullptr || buf2 == nullptr)
    {
        NVLOGE_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "bytes pointer is null: {}=0x{} {}=0x{}", name1, buf1, name2, buf2);
    }
    else if (memcmp(buf1, buf2, nbytes))
    {
        NVLOGE_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "bytes differ: byte[0] {}=0x{:02X} {}=0x{:02X}", name1, *(uint8_t *)buf1, name2, *(uint8_t *)buf2);

        char info_str[64];
        snprintf(info_str, 64, "ARRAY DIFF: %s", name1);
        NVLOGI_FMT_ARRAY(TAG, info_str, reinterpret_cast<uint8_t *>(buf1), nbytes);
        snprintf(info_str, 64, "ARRAY DIFF: %s", name2);
        NVLOGI_FMT_ARRAY(TAG, info_str, reinterpret_cast<uint8_t *>(buf2), nbytes);

        uint8_t *v1 = reinterpret_cast<uint8_t *>(buf1);
        uint8_t *v2 = reinterpret_cast<uint8_t *>(buf2);
        uint32_t i;
        for (i = 0; i < nbytes; i++)
        {
            if (*(v1 + i) != *(v2 + i))
            {
                break;
            }
        }
        v1 += i;
        v2 += i;
        snprintf(info_str, 64, "ARRAY DIFF from %s[%u]", name1, i);
        NVLOGI_FMT_ARRAY(TAG, info_str, v1, nbytes - i);
        snprintf(info_str, 64, "ARRAY DIFF from %s[%u]", name2, i);
        NVLOGI_FMT_ARRAY(TAG, info_str, v2, nbytes - i);
        check_result = -1;
    }
    else
    {
        NVLOGI_FMT(TAG, "bytes same: byte[0] {}={}=0x{:02X}", name1, name2, *(uint8_t *)buf1);
    }
    return check_result;
}

template <typename T>
static int yaml_try_parse_list(yaml::node &parent_node, const char *name, std::vector<T> &values)
{
    yaml::node list_nodes = parent_node[name];
    if (list_nodes.type() != YAML_SEQUENCE_NODE)
    {
        NVLOGE_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "{}: failed to parse {}: error type {}\n", __func__, name, list_nodes.type());
        return -1;
    }

    size_t num = list_nodes.length();
    values.resize(num);

    for (size_t i = 0; i < num; i++)
    {
        yaml::node node = list_nodes[i];
        values[i] = node.as<T>();
    }
    return 0;
}

static int h5dset_try_read(hdf5hpp::hdf5_file &file, const char *name, void *buf, size_t size)
{
    if (!file.is_valid_dataset(name))
    {
        NVLOGW_FMT(TAG, "TV cell {} slot {} {} {} dataset {} not exist",
                   curr_cell, curr_slot, get_task_name(curr_task), curr_tv.c_str(), name);
        return -1;
    }

    try
    {
        hdf5hpp::hdf5_dataset h5dset = file.open_dataset(name);
        if (h5dset.get_buffer_size_bytes() != size)
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
    catch (std::exception &e)
    {
        NVLOGE_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "TV cell {} slot {} {} {} dataset {} exception: {}",
                   curr_cell, curr_slot, get_task_name(curr_task), curr_tv.c_str(), name, e.what());
    }
    return -1;
}

template <typename Type>
static int h5dset_try_read_array(hdf5hpp::hdf5_file &file, const char *name, Type **buf_ptr, uint32_t elem_num)
{
    NVLOGC_FMT(TAG, "Loading buffer {} num={}", name, elem_num);
    *buf_ptr = new Type[elem_num];
    if (h5dset_try_read(file, name, *buf_ptr, sizeof(Type) * elem_num) < 0)
    {
        delete *buf_ptr;
        return -1;
    }

    return 0;
}

static int h5dset_try_read_complex(hdf5hpp::hdf5_file &file, const char *name_real, const char *name_imag, cuComplex **complex_ptr, uint32_t elem_num)
{
    float *tmp_real = nullptr;
    float *tmp_imag = nullptr;

    if (h5dset_try_read_array(file, name_real, &tmp_real, elem_num) < 0)
    {
        return -1;
    }

    if (h5dset_try_read_array(file, name_imag, &tmp_imag, elem_num) < 0)
    {
        delete tmp_real;
        return -1;
    }

    *complex_ptr = new cuComplex[elem_num];
    for (int i = 0; i < elem_num; i++)
    {
        cuComplex *val = *complex_ptr + i;
        val->x = *(tmp_real + i);
        val->y = *(tmp_imag + i);
    }

    delete tmp_real;
    delete tmp_imag;

    return 0;
}

static int h5dset_try_read_u32_to_bits(hdf5hpp::hdf5_file &file, const char *name, std::vector<uint8_t> &dest, int num_bits)
{
    uint32_t *src = new uint32_t[num_bits];
    int ret = h5dset_try_read(file, name, src, num_bits * sizeof(uint32_t));
    if (ret == 0)
    {
        // Initiate bytes to 0
        int nbytes = (num_bits + 7) / 8;
        dest.resize(nbytes);
        for (int i = 0; i < nbytes; i++)
        {
            dest[i] = 0;
        }
        // Convert bits to bytes
        for (int j = 0; j < num_bits; j++)
        {
            dest[j / 8] |= src[j] == 0 ? 0 : 1 << j % 8;
        }
    }
    delete src;
    return ret;
}

template <typename TypeSrc, typename TypeDst>
static int h5dset_try_read_convert(hdf5hpp::hdf5_file &file, const char *name, TypeDst *dst, uint32_t num)
{
    if (num == 0)
    {
        NVLOGW_FMT(TAG, "TV cell {} slot {} {} {} dataset {} reading with num=0",
                   curr_cell, curr_slot, get_task_name(curr_task), curr_tv.c_str(), name);
        return -1;
    }

    TypeSrc *src = new uint32_t[num];
    int ret = h5dset_try_read(file, name, src, num * sizeof(TypeSrc));
    if (ret == 0)
    {
        for (int i = 0; i < num; i++)
        {
            dst[i] = src[i];
        }
    }
    delete src;
    return ret;
}

template <typename T>
static T h5dset_try_parse(const hdf5hpp::hdf5_dataset_elem &dset_elem, const char *name, T default_value, bool miss_warning = true)
{
    T value;
    try
    {
        value = dset_elem[name].as<T>();
    }
    catch (std::exception &e)
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
static T h5dset_try_parse(hdf5hpp::hdf5_dataset &h5dset, const char *name, T default_value, bool miss_warning = true)
{
    // return h5dset_try_parse(h5dset[0], name, default_value, miss_warning);
    return h5dset_try_parse(h5dset[0], name, default_value, miss_warning);
}

template <typename T>
static T h5file_try_parse(const char *file_name, const char *dset_name, const char *var_name, T default_value, bool miss_warning = true, int dset_id = 0)
{
    char h5path[MAX_PATH_LEN];
    get_full_path_file(h5path, CONFIG_CUMAC_TV_PATH, file_name, CONFIG_CUBB_ROOT_DIR_RELATIVE_NUM);
    if (access(h5path, F_OK) != 0)
    {
        NVLOGF_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "H5 file {} not exist", h5path);
    }

    hdf5hpp::hdf5_file hdf5file;
    try
    {
        hdf5file = hdf5hpp::hdf5_file::open(h5path);
    }
    catch (std::exception &e)
    {
        NVLOGE_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "Exception: {}: hdf5_file::open({}): {}", __FUNCTION__, h5path, e.what());
        return default_value;
    }

    if (hdf5file.is_valid_dataset(dset_name))
    {
        hdf5hpp::hdf5_dataset dset = hdf5file.open_dataset(dset_name);
        return h5dset_try_parse(dset[dset_id], var_name, default_value, miss_warning);
    }
    else
    {
        return default_value;
    }
}

int calculate_buf_num(struct cumac::cumacSchedulerParam &param, cumac_buf_num_t &buf_num)
{
    uint32_t prdLen = param.nUe * param.nPrbGrp * param.nBsAnt * param.nBsAnt;
    uint32_t detLen = param.nUe * param.nPrbGrp * param.nBsAnt * param.nBsAnt;
    uint32_t hLen = param.nPrbGrp * param.nUe * param.nCell * param.nBsAnt * param.nUeAnt;

    uint32_t pfSize = param.nPrbGrp * param.numUeSchdPerCellTTI;
    uint32_t pow2N = 2;
    while (pow2N < pfSize)
    {
        pow2N = pow2N << 1;
    }
    buf_num.postEqSinr = param.nActiveUe * param.nPrbGrp * param.nUeAnt;
    buf_num.cellId = param.nCell;
    buf_num.cellAssoc = param.nCell * param.nUe;
    buf_num.cellAssocActUe = param.nCell * param.nActiveUe;
    buf_num.wbSinr = param.nActiveUe * param.nUeAnt;
    buf_num.sinVal = param.nUe * param.nPrbGrp * param.nUeAnt;
    buf_num.prdMat = prdLen;
    buf_num.detMat = detLen;
    buf_num.estH_fr = hLen;
    buf_num.setSchdUePerCellTTI = param.nUe;
    buf_num.allocSol = param.allocType == 1 ? param.nUe * 2 : param.nCell * param.nPrbGrp;
    buf_num.layerSelSol = param.nUe;
    buf_num.mcsSelSol = param.nUe;
    buf_num.pfMetricArr = param.nCell * pow2N;
    buf_num.pfIdArr = param.nCell * pow2N;
    buf_num.avgRatesActUe = param.nActiveUe;
    buf_num.avgRates = param.nUe;
    buf_num.newDataActUe = param.nActiveUe;
    buf_num.tbErrLastActUe = param.nActiveUe;
    buf_num.tbErrLast = param.nUe;
    return 0;
}

int parse_4t4r_tv(cumac_cp_tv_t &tv, std::string tv_file)
{
    char file_path[MAX_PATH_LEN];
    get_full_path_file(file_path, CONFIG_CUMAC_TV_PATH, tv_file.c_str(), CONFIG_CUBB_ROOT_DIR_RELATIVE_NUM);
    if (access(file_path, F_OK) != 0)
    {
        NVLOGF_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "TV file not exist: {}", file_path);
        return -1;
    }

    hdf5hpp::hdf5_file hdf5file;
    try
    {
        hdf5file = hdf5hpp::hdf5_file::open(file_path);
    }
    catch (std::exception &e)
    {
        NVLOGE_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "hdf5_file::open failed. file={}", file_path);
        return -1;
    }

    try
    {
        // Read cumacSchedulerParam
        std::string filepath_str = std::string(file_path);
        H5::H5File file(filepath_str, H5F_ACC_RDONLY);
        // Open the dataset
        H5::DataSet dataset = file.openDataSet("cumacSchedulerParam");
        // Get the compound data type
        H5::CompType compoundType = dataset.getCompType();
        // Read the data from the dataset
        dataset.read(&tv.params, compoundType);

        // nUe=48, nCell=8, totNumCell=8, nPrbGrp=68, nBsAnt=4, nUeAnt=4, W=1.44e+06, sigmaSqrd=1, betaCoeff=1,
        // precodingScheme=1, receiverScheme=1, allocType=1, columnMajor=1, nActiveUe=800, numUeSchdPerCellTTI=6, sinValThr=0.1

        calculate_buf_num(tv.params, tv.buf_num);

        // Create and populate data buffers. Example size: nActiveUe=800, nPrbGrp=68, nBsAnt=4, nUeAnt=4, nMaxSchUePerCell=6
        h5dset_try_read_array(hdf5file, "avgRates", &tv.avgRates, tv.buf_num.avgRates);                   // Dataset {27200}
        h5dset_try_read_array(hdf5file, "cellAssoc", &tv.cellAssoc, tv.buf_num.cellAssoc);                // Dataset {27200}
        h5dset_try_read_array(hdf5file, "cellAssocActUe", &tv.cellAssocActUe, tv.buf_num.cellAssocActUe); // Dataset {27200}
        h5dset_try_read_array(hdf5file, "tbErrLast", &tv.tbErrLast, tv.buf_num.tbErrLast);                // Dataset {27200}

        h5dset_try_read_array(hdf5file, "cellId", &tv.cellId, tv.buf_num.cellId);                           // Dataset {27200}
        h5dset_try_read_array(hdf5file, "postEqSinr", &tv.postEqSinr, tv.buf_num.postEqSinr);               // Dataset {27200}
        h5dset_try_read_array(hdf5file, "wbSinr", &tv.wbSinr, tv.buf_num.wbSinr);                           // Dataset {400}
        h5dset_try_read_complex(hdf5file, "detMat_real", "detMat_imag", &tv.detMat, tv.buf_num.detMat);     // Dataset {6528}
        h5dset_try_read_complex(hdf5file, "estH_fr_real", "estH_fr_imag", &tv.estH_fr, tv.buf_num.estH_fr); // Dataset {52224}
        h5dset_try_read_complex(hdf5file, "prdMat_real", "prdMat_imag", &tv.prdMat, tv.buf_num.prdMat);     // Dataset {6528}
        h5dset_try_read_array(hdf5file, "sinVal", &tv.sinVal, tv.buf_num.sinVal);                           // Dataset {1632}
        h5dset_try_read_array(hdf5file, "avgRatesActUe", &tv.avgRatesActUe, tv.buf_num.avgRatesActUe);      // Dataset {100}
        h5dset_try_read_array(hdf5file, "tbErrLastActUe", &tv.tbErrLastActUe, tv.buf_num.tbErrLastActUe);   // Dataset {100}

        // Parse TV RESPONSE
        h5dset_try_read_array(hdf5file, "setSchdUePerCellTTI", &tv.setSchdUePerCellTTI, tv.buf_num.setSchdUePerCellTTI); // Dataset {6}
        h5dset_try_read_array(hdf5file, "mcsSelSol", &tv.mcsSelSol, tv.buf_num.mcsSelSol);                               // Dataset {6}
        h5dset_try_read_array(hdf5file, "layerSelSol", &tv.layerSelSol, tv.buf_num.layerSelSol);                         // Dataset {6}
        h5dset_try_read_array(hdf5file, "allocSol", &tv.allocSol, tv.buf_num.allocSol);                                  // Dataset {12}

        tv.parsed = 1;

        struct cumac::cumacSchedulerParam &p = tv.params;
        NVLOGC_FMT(TAG, "Parsed TV: cumacSchedulerParam-1: nUe={} nCell={} totNumCell={} nPrbGrp={} nBsAnt={} nUeAnt={} W={} sigmaSqrd={} maxNumUePerCell={} nMaxSchdUePerRnd={} betaCoeff={}",
                   p.nUe, p.nCell, p.totNumCell, p.nPrbGrp, p.nBsAnt, p.nUeAnt, p.W, p.sigmaSqrd, p.maxNumUePerCell, p.nMaxSchdUePerRnd, p.betaCoeff);
        NVLOGC_FMT(TAG, "Parsed TV: cumacSchedulerParam-2: nActiveUe={} numUeSchdPerCellTTI={} precodingScheme={} receiverScheme={} allocType={} columnMajor={} allocType={} columnMajor={} sinValThr={}",
                   p.nActiveUe, p.numUeSchdPerCellTTI, p.precodingScheme, p.receiverScheme, p.allocType, p.columnMajor, p.allocType, p.columnMajor, p.sinValThr);
    }
    catch (std::exception &e)
    {
        NVLOGF_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "TV ERR: {}", e.what());
        return -1;
    }

    return 0;
}

bool pfm_load_tv_H5(const std::string& tv_name, std::vector<cumac_pfm_cell_info_t>& pfm_cell_info, std::vector<cumac_pfm_output_cell_info_t>& pfm_output_cell_info)
{
    NVLOGC_FMT(TAG, "Loading PFM sorting TV file {}", tv_name.c_str());

    char file_path[MAX_PATH_LEN];
    get_full_path_file(file_path, CONFIG_CUMAC_TV_PATH, tv_name.c_str(), CONFIG_CUBB_ROOT_DIR_RELATIVE_NUM);

    if (access(file_path, F_OK) != 0)
    {
        NVLOGE_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "PFM TV file not exist: {}", file_path);
        return false;
    }

    try
    {
        H5::H5File file(file_path, H5F_ACC_RDONLY);

        const int num_cell = pfm_cell_info.size();

        if (num_cell != pfm_output_cell_info.size())
        {
            NVLOGE_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "PFM sorting - number of cells in the output cell info array and the input cell info array are different: {} vs {}", num_cell, pfm_output_cell_info.size());
            return false;
        }

        for (int cIdx = 0; cIdx < num_cell; cIdx++)
        {
            const std::string datasetName = "INPUT_CELL_INFO_" + std::to_string(cIdx);
            // see if the dataset exists
            if (H5Lexists(file.getId(), datasetName.c_str(), H5P_DEFAULT) > 0)
            {
                H5::DataSet dataset = file.openDataSet(datasetName);
                dataset.read(reinterpret_cast<uint8_t*>(&pfm_cell_info[cIdx]), H5::PredType::NATIVE_UINT8);
            }
            else
            {
                NVLOGE_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "PFM sorting TV file {} does not contain input cell info for cell {}", tv_name.c_str(), cIdx);
                return false;
            }
        }

        for (int cIdx = 0; cIdx < num_cell; cIdx++)
        {
            const std::string datasetName = "OUTPUT_CELL_INFO_" + std::to_string(cIdx);
            // see if the dataset exists
            if (H5Lexists(file.getId(), datasetName.c_str(), H5P_DEFAULT) > 0)
            {
                H5::DataSet dataset = file.openDataSet(datasetName);
                dataset.read(reinterpret_cast<uint8_t*>(&pfm_output_cell_info[cIdx]), H5::PredType::NATIVE_UINT8);
            }
            else
            {
                NVLOGE_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "PFM sorting TV file {} does not contain output cell info for cell {}", tv_name.c_str(), cIdx);
                return false;
            }
        }

        NVLOGC_FMT(TAG, "PFM sorting TV file {} loaded successfully", tv_name.c_str());
        return true;
    }
    catch (const H5::FileIException &e)
    {
        NVLOGE_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "PFM TV file exception: {}", e.getDetailMsg());
        return false;
    }
    catch (const std::exception &e)
    {
        NVLOGE_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "PFM TV file exception: {}", e.what());
        return false;
    }
}

bool pfm_validate_tv_h5(const std::string& tv_name, const std::vector<cumac_pfm_cell_info_t>& pfm_cell_info, const std::vector<cumac_pfm_output_cell_info_t>& pfm_output_cell_info)
{
    // get number of cells from the TV name "PFM_SORT_TV_xxCELLS_SLOT_yyyy.h5"
    const std::size_t tv_pos = tv_name.find("TV_");
    const std::size_t cells_pos = tv_name.find("CELLS");

    if (tv_pos == std::string::npos || cells_pos == std::string::npos)
    {
        NVLOGE_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "PFM sorting - invalid TV file name format: {}", tv_name.c_str());
        return false;
    }

    const std::size_t num_start = tv_pos + 3;  // Position after "TV_"
    if (num_start >= cells_pos)
    {
        NVLOGE_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "PFM sorting - invalid TV file name format: {}", tv_name.c_str());
        return false;
    }

    int num_cell{};
    try
    {
        num_cell = std::stoi(tv_name.substr(num_start, cells_pos - num_start));
    }
    catch (const std::invalid_argument& e)
    {
        NVLOGE_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "PFM sorting - invalid number format in TV file name: {}", tv_name.c_str());
        return false;
    }
    catch (const std::out_of_range& e)
    {
        NVLOGE_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "PFM sorting - number out of range in TV file name: {}", tv_name.c_str());
        return false;
    }

    if (num_cell != pfm_cell_info.size())
    {
        NVLOGE_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "PFM sorting - number of cells in the TV file does not match: {} vs {}", num_cell, pfm_cell_info.size());
        return false;
    }

    std::vector<cumac_pfm_cell_info_t> temp_cell_info(num_cell);
    std::vector<cumac_pfm_output_cell_info_t> temp_output_cell_info(num_cell);

    if (!pfm_load_tv_H5(tv_name, temp_cell_info, temp_output_cell_info))
    {
        return false;
    }

    for (int cIdx = 0; cIdx < num_cell; cIdx++)
    {
        // check if temp_cell_info[cIdx] matches pfm_cell_info[cIdx]
        if (temp_cell_info[cIdx].num_ue != pfm_cell_info[cIdx].num_ue ||
            temp_cell_info[cIdx].num_lc_per_ue != pfm_cell_info[cIdx].num_lc_per_ue ||
            temp_cell_info[cIdx].num_lcg_per_ue != pfm_cell_info[cIdx].num_lcg_per_ue)
        {
            NVLOGE_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "PFM sorting - cell info in the TV file does not match for cell {}", cIdx);
            return false;
        }

        for (int idx = 0; idx < (CUMAC_PFM_NUM_QOS_TYPES_UL + CUMAC_PFM_NUM_QOS_TYPES_DL); idx++)
        {
            if (temp_cell_info[cIdx].num_output_sorted_lc[idx] != pfm_cell_info[cIdx].num_output_sorted_lc[idx])
            {
                NVLOGE_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "PFM sorting - number of output sorted LCs in the TV file does not match for cell {}", cIdx);
                return false;
            }
        }

        // check if temp_cell_info[cIdx].ue_info matches pfm_cell_info[cIdx].ue_info
        for (int ueIdx = 0; ueIdx < temp_cell_info[cIdx].num_ue; ueIdx++)
        {
            if (temp_cell_info[cIdx].ue_info[ueIdx].rcurrent_dl != pfm_cell_info[cIdx].ue_info[ueIdx].rcurrent_dl ||
                temp_cell_info[cIdx].ue_info[ueIdx].rcurrent_ul != pfm_cell_info[cIdx].ue_info[ueIdx].rcurrent_ul ||
                temp_cell_info[cIdx].ue_info[ueIdx].rnti != pfm_cell_info[cIdx].ue_info[ueIdx].rnti ||
                temp_cell_info[cIdx].ue_info[ueIdx].id != pfm_cell_info[cIdx].ue_info[ueIdx].id ||
                temp_cell_info[cIdx].ue_info[ueIdx].num_layers_dl != pfm_cell_info[cIdx].ue_info[ueIdx].num_layers_dl ||
                temp_cell_info[cIdx].ue_info[ueIdx].num_layers_ul != pfm_cell_info[cIdx].ue_info[ueIdx].num_layers_ul ||
                temp_cell_info[cIdx].ue_info[ueIdx].flags != pfm_cell_info[cIdx].ue_info[ueIdx].flags)
            {
                NVLOGE_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "PFM sorting - UE info in the TV file does not match for cell {}", cIdx);
                return false;
            }
            else
            {
                // check dl_lc_info
                for (int lcIdx = 0; lcIdx < temp_cell_info[cIdx].num_lc_per_ue; lcIdx++)
                {
                    if (temp_cell_info[cIdx].ue_info[ueIdx].dl_lc_info[lcIdx].tbs_scheduled != pfm_cell_info[cIdx].ue_info[ueIdx].dl_lc_info[lcIdx].tbs_scheduled ||
                        temp_cell_info[cIdx].ue_info[ueIdx].dl_lc_info[lcIdx].flags != pfm_cell_info[cIdx].ue_info[ueIdx].dl_lc_info[lcIdx].flags ||
                        temp_cell_info[cIdx].ue_info[ueIdx].dl_lc_info[lcIdx].qos_type != pfm_cell_info[cIdx].ue_info[ueIdx].dl_lc_info[lcIdx].qos_type)
                    {
                        NVLOGE_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "PFM sorting - DL LC info in the TV file does not match for cell {}", cIdx);
                        return false;
                    }
                }

                // check ul_lcg_info
                for (int lcgIdx = 0; lcgIdx < temp_cell_info[cIdx].num_lcg_per_ue; lcgIdx++)
                {
                    if (temp_cell_info[cIdx].ue_info[ueIdx].ul_lcg_info[lcgIdx].tbs_scheduled != pfm_cell_info[cIdx].ue_info[ueIdx].ul_lcg_info[lcgIdx].tbs_scheduled ||
                        temp_cell_info[cIdx].ue_info[ueIdx].ul_lcg_info[lcgIdx].flags != pfm_cell_info[cIdx].ue_info[ueIdx].ul_lcg_info[lcgIdx].flags ||
                        temp_cell_info[cIdx].ue_info[ueIdx].ul_lcg_info[lcgIdx].qos_type != pfm_cell_info[cIdx].ue_info[ueIdx].ul_lcg_info[lcgIdx].qos_type)
                    {
                        NVLOGE_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "PFM sorting - UL LCG info in the TV file does not match for cell {}", cIdx);
                        return false;
                    }
                }
            }
        }
    }

    // check if temp_output_cell_info matches pfm_output_cell_info
    for (int cIdx = 0; cIdx < num_cell; cIdx++)
    {
        for (int idx = 0; idx < temp_cell_info[cIdx].num_output_sorted_lc[0]; idx++)
        {
            if (temp_output_cell_info[cIdx].dl_gbr_critical[idx].rnti != pfm_output_cell_info[cIdx].dl_gbr_critical[idx].rnti ||
                temp_output_cell_info[cIdx].dl_gbr_critical[idx].lc_id != pfm_output_cell_info[cIdx].dl_gbr_critical[idx].lc_id)
            {
                NVLOGE_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "PFM sorting - DL GBR critical LC info in the TV file does not match for cell {}", cIdx);
                return false;
            }
        }

        for (int idx = 0; idx < temp_cell_info[cIdx].num_output_sorted_lc[1]; idx++)
        {
            if (temp_output_cell_info[cIdx].dl_gbr_non_critical[idx].rnti != pfm_output_cell_info[cIdx].dl_gbr_non_critical[idx].rnti ||
                temp_output_cell_info[cIdx].dl_gbr_non_critical[idx].lc_id != pfm_output_cell_info[cIdx].dl_gbr_non_critical[idx].lc_id)
            {
                NVLOGE_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "PFM sorting - DL GBR non-critical LC info in the TV file does not match for cell {}", cIdx);
                return false;
            }
        }

        for (int idx = 0; idx < temp_cell_info[cIdx].num_output_sorted_lc[2]; idx++)
        {
            if (temp_output_cell_info[cIdx].dl_ngbr_critical[idx].rnti != pfm_output_cell_info[cIdx].dl_ngbr_critical[idx].rnti ||
                temp_output_cell_info[cIdx].dl_ngbr_critical[idx].lc_id != pfm_output_cell_info[cIdx].dl_ngbr_critical[idx].lc_id)
            {
                NVLOGE_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "PFM sorting - DL NGBR critical LC info in the TV file does not match for cell {}", cIdx);
                return false;
            }
        }

        for (int idx = 0; idx < temp_cell_info[cIdx].num_output_sorted_lc[3]; idx++)
        {
            if (temp_output_cell_info[cIdx].dl_ngbr_non_critical[idx].rnti != pfm_output_cell_info[cIdx].dl_ngbr_non_critical[idx].rnti ||
                temp_output_cell_info[cIdx].dl_ngbr_non_critical[idx].lc_id != pfm_output_cell_info[cIdx].dl_ngbr_non_critical[idx].lc_id)
            {
                NVLOGE_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "PFM sorting - DL NGBR non-critical LC info in the TV file does not match for cell {}", cIdx);
                return false;
            }
        }

        for (int idx = 0; idx < temp_cell_info[cIdx].num_output_sorted_lc[4]; idx++)
        {
            if (temp_output_cell_info[cIdx].dl_mbr_non_critical[idx].rnti != pfm_output_cell_info[cIdx].dl_mbr_non_critical[idx].rnti ||
                temp_output_cell_info[cIdx].dl_mbr_non_critical[idx].lc_id != pfm_output_cell_info[cIdx].dl_mbr_non_critical[idx].lc_id)
            {
                NVLOGE_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "PFM sorting - DL MBR non-critical LC info in the TV file does not match for cell {}", cIdx);
                return false;
            }
        }

        for (int idx = 0; idx < temp_cell_info[cIdx].num_output_sorted_lc[5]; idx++)
        {
            if (temp_output_cell_info[cIdx].ul_gbr_critical[idx].rnti != pfm_output_cell_info[cIdx].ul_gbr_critical[idx].rnti ||
                temp_output_cell_info[cIdx].ul_gbr_critical[idx].lcg_id != pfm_output_cell_info[cIdx].ul_gbr_critical[idx].lcg_id)
            {
                NVLOGE_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "PFM sorting - UL GBR critical LCG info in the TV file does not match for cell {}", cIdx);
                return false;
            }
        }

        for (int idx = 0; idx < temp_cell_info[cIdx].num_output_sorted_lc[6]; idx++)
        {
            if (temp_output_cell_info[cIdx].ul_gbr_non_critical[idx].rnti != pfm_output_cell_info[cIdx].ul_gbr_non_critical[idx].rnti ||
                temp_output_cell_info[cIdx].ul_gbr_non_critical[idx].lcg_id != pfm_output_cell_info[cIdx].ul_gbr_non_critical[idx].lcg_id)
            {
                NVLOGE_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "PFM sorting - UL GBR non-critical LCG info in the TV file does not match for cell {}", cIdx);
                return false;
            }
        }

        for (int idx = 0; idx < temp_cell_info[cIdx].num_output_sorted_lc[7]; idx++)
        {
            if (temp_output_cell_info[cIdx].ul_ngbr_critical[idx].rnti != pfm_output_cell_info[cIdx].ul_ngbr_critical[idx].rnti ||
                temp_output_cell_info[cIdx].ul_ngbr_critical[idx].lcg_id != pfm_output_cell_info[cIdx].ul_ngbr_critical[idx].lcg_id)
            {
                NVLOGE_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "PFM sorting - UL NGBR critical LCG info in the TV file does not match for cell {}", cIdx);
                return false;
            }
        }

        for (int idx = 0; idx < temp_cell_info[cIdx].num_output_sorted_lc[8]; idx++)
        {
            if (temp_output_cell_info[cIdx].ul_ngbr_non_critical[idx].rnti != pfm_output_cell_info[cIdx].ul_ngbr_non_critical[idx].rnti ||
                temp_output_cell_info[cIdx].ul_ngbr_non_critical[idx].lcg_id != pfm_output_cell_info[cIdx].ul_ngbr_non_critical[idx].lcg_id)
            {
                NVLOGE_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "PFM sorting - UL NGBR non-critical LCG info in the TV file does not match for cell {}", cIdx);
                return false;
            }
        }

        for (int idx = 0; idx < temp_cell_info[cIdx].num_output_sorted_lc[9]; idx++)
        {
            if (temp_output_cell_info[cIdx].ul_mbr_non_critical[idx].rnti != pfm_output_cell_info[cIdx].ul_mbr_non_critical[idx].rnti ||
                temp_output_cell_info[cIdx].ul_mbr_non_critical[idx].lcg_id != pfm_output_cell_info[cIdx].ul_mbr_non_critical[idx].lcg_id)
            {
                NVLOGE_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "PFM sorting - UL MBR non-critical LCG info in the TV file does not match for cell {}", cIdx);
                return false;
            }
        }
    }

    NVLOGC_FMT(TAG, "PFM sorting TV file {} validated successfully", tv_name.c_str());
    return true;
}

int load_mu_ue_pair_group_tv(ue_pair_tv_t &ue_pair_tv, const int cell_num, const int sfn, const int slot, const bool enable_gpu_share)
{
    ue_pair_tv.mu_ue_pair_tv_loaded = false;
    ue_pair_tv_release_mu_host_buffers(ue_pair_tv);
    ue_pair_tv.muUeGrpSol.clear();
    ue_pair_tv.num_prg = 0;
    ue_pair_tv.num_subband = 0;
    ue_pair_tv.num_prg_samp_per_subband = 0;
    ue_pair_tv.num_bs_ant = 0;
    ue_pair_tv.num_ue_ant = 0;
    ue_pair_tv.num_srs_ue_per_slot_cell = 0;
    ue_pair_tv.num_blocks_per_row_chanOrtMat = 0;
    ue_pair_tv.kernel_launch_flags = 0;
    ue_pair_tv.is_mem_sharing = enable_gpu_share;

    char tv_base[128];
    snprintf(tv_base, sizeof(tv_base), CONFIG_MU_UE_PAIR_TV_FORMAT, sfn, slot);

    char path0[MAX_PATH_LEN];
    char fname0[256];
    snprintf(fname0, sizeof(fname0), "%s_cell0.h5", tv_base);
    get_full_path_file(path0, CONFIG_CUMAC_TV_PATH, fname0, CONFIG_CUBB_ROOT_DIR_RELATIVE_NUM);
    if (access(path0, F_OK) != 0) {
        NVLOGW_FMT(TAG, "MU UE pair TV not found ({}), skipping CP GPU TV staging", fname0);
        return -1;
    }

    try {
        bool cubb_loaded = false;
        size_t expected_sz = 0;
        int32_t tv_num_ue_ant_port = 0;
        bool tv_has_num_ue_ant = false;
        int32_t tv_num_srs_buffers = 0;
        bool tv_has_num_srs_buffers = false;
        for (int c = 0; c < cell_num; c++) {
            char fname[256];
            snprintf(fname, sizeof(fname), "%s_cell%d.h5", tv_base, c);
            char fpath[MAX_PATH_LEN];
            get_full_path_file(fpath, CONFIG_CUMAC_TV_PATH, fname, CONFIG_CUBB_ROOT_DIR_RELATIVE_NUM);

            hdf5hpp::hdf5_file file = hdf5hpp::hdf5_file::open(fpath);
            hid_t fid = file.id();

            if (c == 0) {
                uint8_t tv_is_mem_sharing = 0;
                h5_read_attr_scalar(fid, "is_mem_sharing", H5T_NATIVE_UINT8, &tv_is_mem_sharing);
                if (static_cast<bool>(tv_is_mem_sharing) != ue_pair_tv.is_mem_sharing)
                {
                    NVLOGW_FMT(TAG,
                               "load_mu_ue_pair_group_tv: HDF5 is_mem_sharing={} differs from enable_gpu_share={} (using "
                               "config for layout)",
                               tv_is_mem_sharing, enable_gpu_share ? 1 : 0);
                }

                uint16_t srs_ue = 0;
                h5_read_attr_scalar(fid, "num_srs_ue_per_slot_cell", H5T_NATIVE_UINT16, &srs_ue);
                ue_pair_tv.num_srs_ue_per_slot_cell = srs_ue;

                uint8_t flags = 0;
                h5_read_attr_scalar(fid, "kernel_launch_flags", H5T_NATIVE_UINT8, &flags);
                ue_pair_tv.kernel_launch_flags = flags;

                int32_t blocks_per_row = 0;
                h5_read_attr_scalar(fid, "num_blocks_per_row_chanOrtMat", H5T_NATIVE_INT32, &blocks_per_row);
                ue_pair_tv.num_blocks_per_row_chanOrtMat = static_cast<uint16_t>(blocks_per_row);

                tv_has_num_ue_ant = h5_read_attr_scalar(fid, "num_ue_ant_port", H5T_NATIVE_INT32, &tv_num_ue_ant_port);
                tv_has_num_srs_buffers = h5_read_attr_scalar(fid, "num_srs_buffers", H5T_NATIVE_INT32, &tv_num_srs_buffers);

                expected_sz = cumac_muUeGrp_req_info_size(ue_pair_tv.is_mem_sharing);
                if (mu_pin_alloc(reinterpret_cast<void **>(&ue_pair_tv.task_in_buf_group_host),
                                 &ue_pair_tv.task_in_buf_group_size, expected_sz * static_cast<size_t>(cell_num)) != 0)
                {
                    load_mu_ue_pair_fail(ue_pair_tv);
                    return -1;
                }
            }

            hdf5hpp::hdf5_dataset ds_t = file.open_dataset("task_in_buf");
            const size_t tbytes = ds_t.get_buffer_size_bytes();
            if (tbytes != expected_sz)
            {
                NVLOGE_FMT(TAG, AERIAL_INVALID_PARAM_EVENT,
                    "load_mu_ue_pair_group_tv: cell {} task_in_buf bytes {} != expected_size {}", c, tbytes, expected_sz);
                load_mu_ue_pair_fail(ue_pair_tv);
                return -1;
            }

            uint8_t *const task_dst = ue_pair_tv.task_in_buf_group_host + expected_sz * static_cast<size_t>(c);
            ds_t.read(task_dst);

            // Serialized layout uses inline payload; pointers are not valid across processes (see testMAC cumac_pattern).
            cumac_muUeGrp_req_info_t *req_info = reinterpret_cast<cumac_muUeGrp_req_info_t *>(task_dst);
            req_info->srsInfo = nullptr;
            req_info->srsInfoMsh = nullptr;
            req_info->ueInfo = nullptr;

            // Shared buffers (srs_chan_est, srs_snr, chan_orth_mat) are only in cell 0 file
            // These are shared across all cells, so only load them once
            if (c == 0) {
                ue_pair_tv.num_prg = req_info->nPrbGrp;
                ue_pair_tv.num_subband = req_info->numSubband;
                ue_pair_tv.num_prg_samp_per_subband = req_info->numPrgSampPerSubband;
                ue_pair_tv.num_bs_ant = req_info->nBsAnt;

                // Read num_ue_ant_port from TV file
                int32_t num_ue_ant_port = 0;
                h5_read_attr_scalar(fid, "num_ue_ant_port", H5T_NATIVE_INT32, &num_ue_ant_port);
                ue_pair_tv.num_ue_ant = static_cast<uint16_t>(num_ue_ant_port);

                {
                    hdf5hpp::hdf5_dataset ds = file.open_dataset("chan_orth_mat_buf");
                    const size_t b = ds.get_buffer_size_bytes();
                    const size_t exp_orth =
                        mu_tv_expected_chan_orth_bytes(ue_pair_tv.num_subband, ue_pair_tv.num_prg_samp_per_subband, cell_num);
                    if (!mu_tv_check_dataset_bytes("chan_orth_mat_buf", b, exp_orth))
                    {
                        load_mu_ue_pair_fail(ue_pair_tv);
                        return -1;
                    }
                    if (mu_pin_alloc(reinterpret_cast<void **>(&ue_pair_tv.chan_orth_host), &ue_pair_tv.chan_orth_size, b) != 0)
                    {
                        load_mu_ue_pair_fail(ue_pair_tv);
                        return -1;
                    }
                    ds.read(reinterpret_cast<uint8_t *>(ue_pair_tv.chan_orth_host));
                }
                {
                    hdf5hpp::hdf5_dataset ds = file.open_dataset("srs_chan_est_buf");
                    const size_t dim = ds.get_buffer_size_bytes();
                    const size_t exp_ce = mu_tv_expected_srs_chan_est_bytes(
                        ue_pair_tv.num_bs_ant, ue_pair_tv.num_subband, ue_pair_tv.num_prg_samp_per_subband, cell_num);
                    if (!mu_tv_check_dataset_bytes("srs_chan_est_buf", dim, exp_ce))
                    {
                        load_mu_ue_pair_fail(ue_pair_tv);
                        return -1;
                    }
                    if (mu_pin_alloc(reinterpret_cast<void **>(&ue_pair_tv.srs_chan_est_host), &ue_pair_tv.srs_chan_est_size, dim) != 0)
                    {
                        load_mu_ue_pair_fail(ue_pair_tv);
                        return -1;
                    }
                    ds.read(ue_pair_tv.srs_chan_est_host);
                }
                {
                    hdf5hpp::hdf5_dataset ds = file.open_dataset("srs_snr_buf");
                    const size_t b = ds.get_buffer_size_bytes();
                    const size_t exp_snr = mu_tv_expected_srs_snr_bytes(cell_num);
                    if (!mu_tv_check_dataset_bytes("srs_snr_buf", b, exp_snr))
                    {
                        load_mu_ue_pair_fail(ue_pair_tv);
                        return -1;
                    }
                    if (mu_pin_alloc(reinterpret_cast<void **>(&ue_pair_tv.srs_snr_host), &ue_pair_tv.srs_snr_size, b) != 0)
                    {
                        load_mu_ue_pair_fail(ue_pair_tv);
                        return -1;
                    }
                    ds.read(reinterpret_cast<uint8_t *>(ue_pair_tv.srs_snr_host));
                }

                // cubb_srs_gpu_buf: only allocate and load if the dataset is present in this TV.
                // When absent (TV generated without CUBB dump), cubb_srs_buf_size stays 0 and
                // ue_pair_cubb_gpu_ keeps its cudaMemset-zero value from handler init.
                if (!cubb_loaded && H5Lexists(fid, "cubb_srs_gpu_buf", H5P_DEFAULT) > 0)
                {
                    size_t total_num_buffers =
                        std::min(static_cast<size_t>(num_srs_buffers_per_cell) * cell_num, static_cast<size_t>(MAX_SRS_CHEST_BUFFERS));
                    size_t buffer_size =
                        sizeof(uint32_t) * ue_pair_tv.num_prg * ue_pair_tv.num_bs_ant * ue_pair_tv.num_ue_ant;
                    ue_pair_tv.cubb_srs_buf_size = total_num_buffers * buffer_size;
                    NVLOGI_FMT(TAG, "TV: {} cubb_srs_gpu_buf_total_size: {} * {} = {}",
                        fname, buffer_size, total_num_buffers, ue_pair_tv.cubb_srs_buf_size);
                    if (ue_pair_tv.cubb_srs_buf_size == 0)
                    {
                        NVLOGE_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "cubb_srs_gpu_buf_size is 0 (num_prg={} num_bs_ant={} num_ue_ant={})",
                                   ue_pair_tv.num_prg, ue_pair_tv.num_bs_ant, ue_pair_tv.num_ue_ant);
                        load_mu_ue_pair_fail(ue_pair_tv);
                        return -1;
                    }
                    if (mu_pin_alloc(reinterpret_cast<void **>(&ue_pair_tv.cubb_srs_buf_host), &ue_pair_tv.cubb_srs_buf_size,
                                     ue_pair_tv.cubb_srs_buf_size) != 0)
                    {
                        load_mu_ue_pair_fail(ue_pair_tv);
                        return -1;
                    }
                    hdf5hpp::hdf5_dataset ds = file.open_dataset("cubb_srs_gpu_buf");
                    const size_t dim = ds.get_buffer_size_bytes();
                    if (dim != ue_pair_tv.cubb_srs_buf_size)
                    {
                        NVLOGE_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "cubb_srs_gpu_buf size {} != expected {}", dim,
                                   ue_pair_tv.cubb_srs_buf_size);
                        load_mu_ue_pair_fail(ue_pair_tv);
                        return -1;
                    }
                    ds.read(ue_pair_tv.cubb_srs_buf_host);
                    cubb_loaded = true;
                }

                NVLOGC_FMT(TAG, "SFN {}.{} UE_PAIR_TV: {} size: task_in_buf={} srs_chan_est={} srs_snr={} chan_orth={} cubb_srs={}", sfn, slot, fname,
                    expected_sz, ue_pair_tv.srs_chan_est_size, ue_pair_tv.srs_snr_size, ue_pair_tv.chan_orth_size, ue_pair_tv.cubb_srs_buf_size);
            }
        }

        char solname[256];
        snprintf(solname, sizeof(solname), "%s_solution.h5", tv_base);
        char solpath[MAX_PATH_LEN];
        get_full_path_file(solpath, CONFIG_CUMAC_TV_PATH, solname, CONFIG_CUBB_ROOT_DIR_RELATIVE_NUM);
        if (access(solpath, F_OK) != 0)
        {
            NVLOGW_FMT(TAG, "MU UE pair solution TV not found: {}", solname);
            load_mu_ue_pair_fail(ue_pair_tv);
            return -1;
        }
        hdf5hpp::hdf5_file sfile = hdf5hpp::hdf5_file::open(solpath);
        hdf5hpp::hdf5_dataset ds_sol = sfile.open_dataset("solution");
        const size_t sol_bytes = ds_sol.get_buffer_size_bytes();
        const size_t need = sizeof(cumac_muUeGrp_resp_info_t) * static_cast<size_t>(cell_num);
        if (sol_bytes < need)
        {
            NVLOGE_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "MU UE pair solution bytes {} < expected {}", sol_bytes, need);
            load_mu_ue_pair_fail(ue_pair_tv);
            return -1;
        }
        ue_pair_tv.muUeGrpSol.resize(cell_num);
        if (ds_sol.get_buffer_size_bytes() != sizeof(cumac_muUeGrp_resp_info_t) * cell_num)
        {
            NVLOGE_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "MU UE pair solution bytes {} != expected {}", ds_sol.get_buffer_size_bytes(), sizeof(cumac_muUeGrp_resp_info_t) * cell_num);
            load_mu_ue_pair_fail(ue_pair_tv);
            return -1;
        }
        ds_sol.read(reinterpret_cast<uint8_t *>(ue_pair_tv.muUeGrpSol.data()));

        ue_pair_tv.mu_ue_pair_tv_loaded = true;
        NVLOGC_FMT(TAG, "TV: {} size: solution={}", solname, need);
        return 0;
    }
    catch (const std::exception &e)
    {
        NVLOGE_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "load_mu_ue_pair_group_tv: {}", e.what());
        load_mu_ue_pair_fail(ue_pair_tv);
        return -1;
    }
}

int parse_group_tv(cumac_cp_tv_t &tv, const int cell_num, const bool enable_gpu_share, const int srs_slot_lag, const uint32_t task_bitmask)
{
    char file_path[MAX_PATH_LEN];
    bool any_tv_loaded = false;

    if ((task_bitmask & CUMAC_CP_TASK_MASK_4T4R) != 0U)
    {
        const std::string tv_4t4r_file = std::string("TV_cumac_F08-MC-CC-") + std::to_string(cell_num) + "PC_DL.h5";
        get_full_path_file(file_path, CONFIG_CUMAC_TV_PATH, tv_4t4r_file.c_str(), CONFIG_CUBB_ROOT_DIR_RELATIVE_NUM);

        if (access(file_path, F_OK) != 0)
        {
            NVLOGE_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "Old TV file not found: {}", tv_4t4r_file.c_str());
            return -1;
        }

        NVLOGC_FMT(TAG, "Found old TV file: {} with {} cells", tv_4t4r_file.c_str(), cell_num);

        if (parse_4t4r_tv(tv, tv_4t4r_file) != 0)
        {
            NVLOGE_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "Failed to parse 4T4R TV file: {}", tv_4t4r_file.c_str());
            return -1;
        }
        any_tv_loaded = true;
    }
    else
    {
        NVLOGC_FMT(TAG, "parse_group_tv: task_bitmask=0x{:X} has no 4T4R tasks, skipping 4T4R TV", task_bitmask);
    }

    const std::string pfm_tv_file = "PFM_SORT_TV_" + std::to_string(cell_num) + "CELLS_SLOT_1000.h5";
    get_full_path_file(file_path, CONFIG_CUMAC_TV_PATH, pfm_tv_file.c_str(), CONFIG_CUBB_ROOT_DIR_RELATIVE_NUM);

    if (access(file_path, F_OK) == 0)
    {
        NVLOGC_FMT(TAG, "Found PFM TV file: {}", pfm_tv_file.c_str());

        tv.pfmCellInfo.resize(cell_num);
        std::memset(tv.pfmCellInfo.data(), 0, sizeof(cumac_pfm_cell_info_t) * cell_num);

        tv.pfmSortSol.resize(cell_num);
        std::memset(tv.pfmSortSol.data(), 0, sizeof(cumac_pfm_output_cell_info_t) * cell_num);

        if (!pfm_load_tv_H5(pfm_tv_file, tv.pfmCellInfo, tv.pfmSortSol))
        {
            NVLOGE_FMT(TAG, AERIAL_INVALID_PARAM_EVENT, "Failed to load PFM TV file: {}", pfm_tv_file.c_str());
            return -1;
        }

        NVLOGC_FMT(TAG, "Successfully loaded PFM TV file: {}", pfm_tv_file.c_str());
        any_tv_loaded = true;
    }
    else
    {
        NVLOGW_FMT(TAG, "PFM TV file not found: {}, skipping PFM TV loading", pfm_tv_file.c_str());
    }

    // Discover MU UE pair TVs by scanning the TV directory. Each TV file is
    // named muUePairTV_sfn<S>_slot<N>_cell0.h5; (SFN,slot) pairs collected
    // here drive both the schedule_slot_period probe (read from the first
    // matching file's H5 attribute) and per-slot loading.
    namespace fs = std::filesystem;
    const std::string tv_prefix = "muUePairTV_sfn";
    const std::string cell0_sfx = "_cell0.h5";
    std::vector<std::pair<int,int>> sfn_slots;
    {
        char probe_path[MAX_PATH_LEN];
        get_full_path_file(probe_path, CONFIG_CUMAC_TV_PATH, "probe", CONFIG_CUBB_ROOT_DIR_RELATIVE_NUM);
        const fs::path tv_dir = fs::path(probe_path).parent_path();
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
            sfn_slots.push_back({fsfn, fslot});
        }
        if (ec)
        {
            NVLOGW_FMT(TAG, "parse_group_tv: TV directory scan error ({}): {}", tv_dir.string(), ec.message());
        }
        std::sort(sfn_slots.begin(), sfn_slots.end());
    }

    // Read schedule_slot_period from the first TV file (default to the legacy
    // size when the attribute is missing).
    size_t schedule_slot_period = SLOT_NUM_PER_FRAME;
    if (!sfn_slots.empty())
    {
        char fname[256];
        snprintf(fname, sizeof(fname), "muUePairTV_sfn%d_slot%d_cell0.h5",
                 sfn_slots.front().first, sfn_slots.front().second);
        char fpath[MAX_PATH_LEN];
        get_full_path_file(fpath, CONFIG_CUMAC_TV_PATH, fname, CONFIG_CUBB_ROOT_DIR_RELATIVE_NUM);
        try
        {
            hdf5hpp::hdf5_file probe = hdf5hpp::hdf5_file::open(fpath);
            int32_t period_attr = 0;
            if (h5_read_attr_scalar(probe.id(), "schedule_slot_period", H5T_NATIVE_INT32, &period_attr) && period_attr > 0)
            {
                schedule_slot_period = static_cast<size_t>(period_attr);
            }
        }
        catch (const std::exception& e)
        {
            NVLOGW_FMT(TAG, "parse_group_tv: probe of '{}' failed: {}; using default period {}",
                       fname, e.what(), schedule_slot_period);
        }
    }

    NVLOGC_FMT(TAG, "UE_PAIR_TV schedule_slot_period={} scheduled_slots={} srs_slot_lag={}", schedule_slot_period, sfn_slots.size(), srs_slot_lag);

    tv.ue_pair.resize(schedule_slot_period);
    int num_slot_loaded = 0;
    for (const auto& [fsfn, fslot] : sfn_slots)
    {
        const size_t idx = (fsfn * SLOT_NUM_PER_FRAME + fslot + srs_slot_lag) % schedule_slot_period;
        if (tv.ue_pair[idx].mu_ue_pair_tv_loaded)
        {
            NVLOGW_FMT(TAG, "SFN {}.{} UE_PAIR_TV idx={} already loaded; ignoring duplicate slot", fsfn, fslot, idx);
            continue;
        }
        if (load_mu_ue_pair_group_tv(tv.ue_pair[idx], cell_num, fsfn, fslot, enable_gpu_share) == 0)
        {
            num_slot_loaded++;
            any_tv_loaded = true;
        }
    }

    // Surface every unfilled slot index at INFO level so partial-period
    // schedules are visible without flooding the log.
    for (size_t s = 0; s < schedule_slot_period; ++s)
    {
        if (!tv.ue_pair[s].mu_ue_pair_tv_loaded)
        {
            NVLOGI_FMT(TAG, "UE_PAIR_TV missing for slot idx={} (period={})", s, schedule_slot_period);
        }
    }

    if (num_slot_loaded > 0)
    {
        NVLOGC_FMT(TAG, "UE_PAIR_TV loaded for {} / {} scheduled slots ({} cells)",
                   num_slot_loaded, schedule_slot_period, cell_num);
    }
    else
    {
        NVLOGW_FMT(TAG, "UE_PAIR_TV not loaded for any slot (optional)");
    }

    tv.parsed = any_tv_loaded ? 1U : 0U;
    NVLOGC_FMT(TAG, "Parsed group TV files for {} cells", cell_num);
    return 0;
}

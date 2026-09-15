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

/*
 * dump_tv.cpp — cuMAC-CP MU UE-pair test vector dump utility
 *
 * Reads muUePairTV_sfn0_slotN_cellC.h5 files and prints all task_in_buf
 * fields (cumac_muUeGrp_req_info_t header, srsInfo / srsInfoMsh entries,
 * ueInfo entries) plus the expected solution from the companion
 * muUePairTV_sfn0_slotN_solution.h5 file in human-readable text.
 *
 * Usage:
 *   dump_tv [--tv-dir <path>] [--cells <N>] [--slot <N>]
 *           [--mem-sharing | --no-mem-sharing]
 *           [--solution-only] [--no-solution]
 *           [--max-chanest <K>]
 *
 * Options:
 *   --tv-dir <path>      Directory containing the HDF5 TV files.
 *                        Default: $CUBB_SDK/testVectors/cumac/
 *   --cells <N>          Number of cells (default: 1).
 *   --slot <N>           Dump only slot N (0-19). Default: all slots found.
 *   --mem-sharing        Override: treat buffers as GPU-share layout.
 *   --no-mem-sharing     Override: treat buffers as inline-chanest layout.
 *   --solution-only      Skip task_in_buf; only print solution.
 *   --no-solution        Skip the solution file.
 *   --max-chanest <K>    Max srsChanEst complex pairs to print per UE (default 8).
 */

// ---------------------------------------------------------------------------
// __half / __half2 shims must appear before cumac_muUeGrp.h includes cuda_fp16.h.
// Define the cuda_fp16 include guards first so the real header is skipped; then
// provide minimal structs that match CUDA's ABI (2-byte __half, 4-byte __half2).
// ---------------------------------------------------------------------------
#define __CUDA_FP16_H__
#define __CUDA_FP16_HPP__

struct alignas(2) __half  { unsigned short __x; };
struct alignas(4) __half2 { unsigned short x, y; };

static_assert(sizeof(__half)  == 2, "FP16 shim size mismatch");
static_assert(sizeof(__half2) == 4, "FP16 shim size mismatch");

// ---------------------------------------------------------------------------
// Standard / project includes
// ---------------------------------------------------------------------------
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cfloat>
#include <vector>
#include <string>
#include <stdexcept>
#include <algorithm>
#include <unistd.h>
#include <glob.h>

#include "cumac_muUeGrp.h"   // cumac_muUeGrp_req_info_t and friends
#include "hdf5hpp.hpp"        // hdf5_file / hdf5_dataset wrappers

// ---------------------------------------------------------------------------
// Compile-time layout sanity checks
// ---------------------------------------------------------------------------
static_assert(sizeof(cumac_muUeGrp_req_srs_info_msh_t) == 20,
              "srsInfoMsh layout unexpected");

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static float u32_to_f32(uint32_t v)
{
    float f;
    std::memcpy(&f, &v, sizeof(f));
    return f;
}

// IEEE 754-2008 binary16 → float, no CUDA runtime needed.
static float half_to_float(uint16_t h)
{
    const uint32_t sign = (h >> 15) & 1u;
    const uint32_t exp  = (h >> 10) & 0x1fu;
    const uint32_t mant = h & 0x3ffu;
    uint32_t f32;
    if (exp == 0) {
        if (mant == 0) {
            f32 = sign << 31;
        } else {
            // Denormal: normalise it
            uint32_t m = mant, e2 = 127u - 14u;
            while (!(m & 0x400u)) { m <<= 1; --e2; }
            f32 = (sign << 31) | (e2 << 23) | ((m & 0x3ffu) << 13);
        }
    } else if (exp == 0x1fu) {
        f32 = (sign << 31) | (0xffu << 23) | (mant << 13); // Inf / NaN
    } else {
        f32 = (sign << 31) | ((exp + 112u) << 23) | (mant << 13);
    }
    float r;
    std::memcpy(&r, &f32, sizeof(r));
    return r;
}

static bool h5_attr_u8(hid_t fid, const char *name, uint8_t &out)
{
    if (H5Aexists(fid, name) <= 0) return false;
    hid_t a = H5Aopen(fid, name, H5P_DEFAULT);
    if (a < 0) return false;
    H5Aread(a, H5T_NATIVE_UINT8, &out);
    H5Aclose(a);
    return true;
}

static bool h5_attr_u16(hid_t fid, const char *name, uint16_t &out)
{
    if (H5Aexists(fid, name) <= 0) return false;
    hid_t a = H5Aopen(fid, name, H5P_DEFAULT);
    if (a < 0) return false;
    H5Aread(a, H5T_NATIVE_UINT16, &out);
    H5Aclose(a);
    return true;
}

static bool h5_attr_i32(hid_t fid, const char *name, int32_t &out)
{
    if (H5Aexists(fid, name) <= 0) return false;
    hid_t a = H5Aopen(fid, name, H5P_DEFAULT);
    if (a < 0) return false;
    H5Aread(a, H5T_NATIVE_INT32, &out);
    H5Aclose(a);
    return true;
}

static bool h5_attr_u32(hid_t fid, const char *name, uint32_t &out)
{
    if (H5Aexists(fid, name) <= 0) return false;
    hid_t a = H5Aopen(fid, name, H5P_DEFAULT);
    if (a < 0) return false;
    H5Aread(a, H5T_NATIVE_UINT32, &out);
    H5Aclose(a);
    return true;
}

// ---------------------------------------------------------------------------
// Print helpers
// ---------------------------------------------------------------------------

static void print_sep(char c = '=', int w = 70)
{
    for (int i = 0; i < w; ++i) putchar(c);
    putchar('\n');
}

static void decode_ue_flags(uint8_t flags, char *buf, int bufsz)
{
    int pos = 0;
    if (flags & 0x01) pos += snprintf(buf + pos, bufsz - pos, "valid");
    if (flags & 0x02) pos += snprintf(buf + pos, bufsz - pos, "%snewTx",  pos > 0 ? "|" : "");
    if (flags & 0x04) pos += snprintf(buf + pos, bufsz - pos, "%ssrsEst", pos > 0 ? "|" : "");
    if (flags & 0x08) pos += snprintf(buf + pos, bufsz - pos, "%ssrsUpd", pos > 0 ? "|" : "");
    if (pos == 0) snprintf(buf, bufsz, "none");
}

static void print_req_info_header(const cumac_muUeGrp_req_info_t *r, bool is_mem_sharing)
{
    printf("  betaCoeff            = %.6g  (0x%08X)\n", u32_to_f32(r->betaCoeff),           r->betaCoeff);
    printf("  muCoeff              = %.6g  (0x%08X)\n", u32_to_f32(r->muCoeff),             r->muCoeff);
    printf("  chanCorrThr          = %.6g  (0x%08X)\n", u32_to_f32(r->chanCorrThr),         r->chanCorrThr);
    printf("  srsSnrThr            = %.6g dB  (0x%08X)\n", u32_to_f32(r->srsSnrThr),        r->srsSnrThr);
    printf("  muGrpSrsSnrMaxGap    = %.6g dB  (0x%08X)\n", u32_to_f32(r->muGrpSrsSnrMaxGap),   r->muGrpSrsSnrMaxGap);
    printf("  muGrpSrsSnrSplitThr  = %.6g dB  (0x%08X)\n", u32_to_f32(r->muGrpSrsSnrSplitThr), r->muGrpSrsSnrSplitThr);
    printf("  numUeInfo            = %u\n",  (unsigned)r->numUeInfo);
    printf("  numSrsInfo           = %u\n",  (unsigned)r->numSrsInfo);
    printf("  numSubband           = %u\n",  (unsigned)r->numSubband);
    printf("  numPrgSampPerSubband = %u\n",  (unsigned)r->numPrgSampPerSubband);
    printf("  numUeForGrpPerCell   = %u\n",  (unsigned)r->numUeForGrpPerCell);
    printf("  nPrbGrp              = %u\n",  (unsigned)r->nPrbGrp);
    printf("  nBsAnt               = %u\n",  (unsigned)r->nBsAnt);
    printf("  nMaxUeSchdPerCellTTI = %u\n",  (unsigned)r->nMaxUeSchdPerCellTTI);
    printf("  nMaxUePerGrp         = %u\n",  (unsigned)r->nMaxUePerGrp);
    printf("  nMaxLayerPerGrp      = %u\n",  (unsigned)r->nMaxLayerPerGrp);
    printf("  nMaxLayerPerUeSu     = %u\n",  (unsigned)r->nMaxLayerPerUeSu);
    printf("  nMaxLayerPerUeMu     = %u\n",  (unsigned)r->nMaxLayerPerUeMu);
    printf("  nMaxUegPerCell       = %u\n",  (unsigned)r->nMaxUegPerCell);
    printf("  allocType            = %u\n",  (unsigned)r->allocType);
    printf("  [is_mem_sharing      = %s]\n", is_mem_sharing ? "true" : "false");
}

static void print_srs_msh(const cumac_muUeGrp_req_srs_info_msh_t *s, int idx, const char *pfx)
{
    char flags_str[32] = "none";
    if (s->flags & 0x01) snprintf(flags_str, sizeof(flags_str), "valid");

    printf("%s srsInfoMsh[%3d]: rnti=%-5u id=%-4u nUeAnt=%-2u"
           " srsWbSnr=%.2f dB (0x%08X) flags=0x%02X(%s)"
           " realBuffIdx=%-4u srsStartPrg=%-4u srsStartValidPrg=%-4u srsNValidPrg=%-4u\n",
           pfx, idx, (unsigned)s->rnti, (unsigned)s->id, (unsigned)s->nUeAnt,
           u32_to_f32(s->srsWbSnr), s->srsWbSnr, s->flags, flags_str,
           (unsigned)s->realBuffIdx, (unsigned)s->srsStartPrg,
           (unsigned)s->srsStartValidPrg, (unsigned)s->srsNValidPrg);
}

static void print_srs_inline(const cumac_muUeGrp_req_srs_info_t *s, int idx,
                              int num_bs_ant, int num_ue_ant,
                              int num_subband, int num_prg_samp, int max_print,
                              const char *pfx)
{
    char flags_str[32] = "none";
    if (s->flags & 0x01) snprintf(flags_str, sizeof(flags_str), "valid");

    printf("%s srsInfo[%3d]: rnti=%-5u id=%-4u nUeAnt=%-2u"
           " srsWbSnr=%.2f dB (0x%08X) flags=0x%02X(%s)\n",
           pfx, idx, (unsigned)s->rnti, (unsigned)s->id, (unsigned)s->nUeAnt,
           u32_to_f32(s->srsWbSnr), s->srsWbSnr, s->flags, flags_str);

    int actual_ue_ant = (s->nUeAnt > 0) ? s->nUeAnt : num_ue_ant;
    int total = num_bs_ant * actual_ue_ant * num_subband * num_prg_samp;
    total = std::min(total, (int)CUMAC_MUUEGRP_SRS_CHAN_EST_LEN);
    int nprint = std::min(max_print, total);

    printf("%s   srsChanEst (total %d complex half2, showing first %d):\n",
           pfx, total, nprint);
    for (int i = 0; i < nprint; ++i) {
        float re = half_to_float(s->srsChanEst[i].x);
        float im = half_to_float(s->srsChanEst[i].y);
        printf("%s     [%4d] %+.4f %+.4fi  (raw 0x%04X 0x%04X)\n",
               pfx, i, re, im, s->srsChanEst[i].x, s->srsChanEst[i].y);
    }

    if (total > 0) {
        double sum_sq = 0.0;
        int nonzero = 0;
        for (int i = 0; i < total; ++i) {
            float re = half_to_float(s->srsChanEst[i].x);
            float im = half_to_float(s->srsChanEst[i].y);
            double mag2 = (double)re * re + (double)im * im;
            sum_sq += mag2;
            if (mag2 > 0.0) ++nonzero;
        }
        printf("%s   srsChanEst rms=%.4f  nonzero=%d/%d\n",
               pfx, sqrt(sum_sq / total), nonzero, total);
    }
}

static void print_ue_info(const cumac_muUeGrp_req_ue_info_t *u, int idx, const char *pfx)
{
    char flags_str[128];
    decode_ue_flags(u->flags, flags_str, sizeof(flags_str));

    printf("%s ueInfo[%3d]: rnti=%-5u id=%-4u nUeAnt=%-2u flags=0x%02X(%s)"
           " avgRate=%-10u currRate=%-10u bufSize=%-10u"
           " layerSel=0x%02X allocPrg=%-6u srsIdx=%u\n",
           pfx, idx, (unsigned)u->rnti, (unsigned)u->id, (unsigned)u->nUeAnt,
           u->flags, flags_str,
           u->avgRate, u->currRate, u->bufferSize,
           u->layerSelLastTx, u->numAllocPrgLastTx, (unsigned)u->srsInfoIdx);
}

static void print_solution(const cumac_muUeGrp_resp_info_t *sol, const char *pfx)
{
    printf("%s numSchdUeg=%u\n", pfx, sol->numSchdUeg);
    for (uint32_t g = 0; g < sol->numSchdUeg && g < MAX_NUM_UEG_PER_CELL; ++g) {
        const cumac_muUeGrp_resp_ueg_info_t &ueg = sol->schdUegInfo[g];
        if (!(ueg.flags & 0x01)) {
            printf("%s UEG[%u]: not valid (flags=0x%02X)\n", pfx, g, ueg.flags);
            continue;
        }
        printf("%s UEG[%u]: prgStart=%-4d prgEnd=%-4d numUeInGrp=%-2u flags=0x%02X\n",
               pfx, g, (int)ueg.allocPrgStart, (int)ueg.allocPrgEnd,
               (unsigned)ueg.numUeInGrp, ueg.flags);
        for (uint8_t u = 0; u < ueg.numUeInGrp && u < MAX_NUM_UE_PER_GRP; ++u) {
            const cumac_muUeGrp_resp_ue_info_t &ue = ueg.ueInfo[u];
            if (!(ue.flags & 0x01)) continue;
            const char *mimo = (ue.flags & 0x02) ? "MU-MIMO" : "SU-MIMO";
            printf("%s   UE[%u]: rnti=%-5u id=%-4u layerSel=0x%02X "
                   "ueOrder=%-2u nSCID=%u %s\n",
                   pfx, u, (unsigned)ue.rnti, (unsigned)ue.id, ue.layerSel,
                   (unsigned)ue.ueOrderInGrp, (unsigned)ue.nSCID, mimo);
        }
    }
}

// ---------------------------------------------------------------------------
// cuBB TV dump helpers
// ---------------------------------------------------------------------------

// Mirror of SrsInfoUpdate from srs_ipc_manager.hpp (12 bytes, no header deps).
struct CubbSrsInfoUpdate {
    uint32_t real_buff_idx;
    uint16_t srs_info_idx;
    uint16_t cell_idx;
    uint16_t rnti;
    uint16_t _pad;
};
static_assert(sizeof(CubbSrsInfoUpdate) == 12, "SrsInfoUpdate size mismatch");

// Flat view of CVSrsChestBuff binary layout (aarch64, sizeof=104).
// Offsets confirmed by layout probe on the actual build toolchain.
// Fields buffer (off=8, 8B) and buffDesc (off=16, 56B) are zeroed by
// scrubDumpOnlyMembers() before the H5 write, so they are not read here.
struct ChestBufFlat {
    int32_t  state;           // off  0  srs_chest_buff_state (enum int)
    int32_t  _pad0;           // off  4  padding
    uint64_t _buffer;         // off  8  unique_ptr (zeroed in H5)
    uint8_t  _buffDesc[56];   // off 16  tensor_desc (zeroed in H5)
    uint32_t rnti;            // off 72
    uint32_t buffer_idx;      // off 76
    uint32_t cell_id;         // off 80
    uint32_t usage;           // off 84  srs_chest_buff_usage
    uint16_t sfn;             // off 88
    uint16_t slot_f;          // off 90
    uint8_t  srsPrgSize;      // off 92
    uint8_t  _pad1;           // off 93  padding
    uint16_t srsStartPrg;     // off 94
    uint16_t srsStartValidPrg;// off 96
    uint16_t srsNValidPrg;    // off 98
    uint8_t  _pad2[4];        // off 100 trailing padding to 104
};
static_assert(sizeof(ChestBufFlat) == 104, "ChestBufFlat layout mismatch");

static const char *chest_state_str(int32_t s) {
    switch (s) {
        case 0: return "INIT";
        case 1: return "REQUESTED";
        case 2: return "READY";
        case 3: return "NONE";
        default: return "?";
    }
}
static constexpr uint32_t CV_INVALID_RNTI_VAL = 65535u;

static std::vector<std::string> glob_files(const std::string& pattern)
{
    std::vector<std::string> result;
    glob_t g{};
    if (glob(pattern.c_str(), GLOB_NOSORT, nullptr, &g) == 0) {
        for (size_t i = 0; i < g.gl_pathc; ++i)
            result.push_back(g.gl_pathv[i]);
    }
    globfree(&g);
    std::sort(result.begin(), result.end());
    return result;
}

static void dump_cubb_file(const std::string& path, int max_chanest)
{
    // Parse SFN and slot from filename: cubb_srs_buffers_<N>_SFN_<sfn>.<slot>.h5
    int file_sfn = -1, file_slot = -1;
    {
        const size_t slash = path.rfind('/');
        const char  *bn    = path.c_str() + (slash == std::string::npos ? 0 : slash + 1);
        sscanf(bn, "cubb_srs_buffers_%*d_SFN_%d.%d.h5", &file_sfn, &file_slot);
    }

    printf("\n[cuBB TV] SFN %d.%d  %s\n", file_sfn, file_slot, path.c_str());
    print_sep('-', 60);

    try {
        hdf5hpp::hdf5_file file = hdf5hpp::hdf5_file::open(path.c_str());
        hid_t fid = file.id();

        uint32_t gpu_pool_len = 0, gpu_buf_size = 0, num_prg = 0;
        uint32_t num_gnb_ant  = 0, num_ue_layer = 0, cell_num = 0;
        h5_attr_u32(fid, "gpu_pool_len", gpu_pool_len);
        h5_attr_u32(fid, "gpu_buf_size", gpu_buf_size);
        h5_attr_u32(fid, "num_prg",      num_prg);
        h5_attr_u32(fid, "num_gnb_ant",  num_gnb_ant);
        h5_attr_u32(fid, "num_ue_layer", num_ue_layer);
        h5_attr_u32(fid, "cell_num",     cell_num);

        // All attributes on one line
        printf("  gpu_pool_len=%u  gpu_buf_size=%u  num_prg=%u  num_gnb_ant=%u"
               "  num_ue_layer=%u  cell_num=%u\n",
               gpu_pool_len, gpu_buf_size, num_prg, num_gnb_ant, num_ue_layer, cell_num);

        // ---------------------------------------------------------------
        // srs_info_pool — one line per entry, prefix from entry.cell_idx
        // ---------------------------------------------------------------
        hdf5hpp::hdf5_dataset ds_info = file.open_dataset("srs_info_pool");
        const size_t   info_bytes = ds_info.get_buffer_size_bytes();
        const uint32_t n_info     = (uint32_t)(info_bytes / sizeof(CubbSrsInfoUpdate));

        printf("srs_info_pool: %zu B  %u entries\n", info_bytes, n_info);

        std::vector<uint8_t> info_raw(info_bytes);
        ds_info.read(info_raw.data());
        const auto *info_arr = reinterpret_cast<const CubbSrsInfoUpdate *>(info_raw.data());

        for (uint32_t i = 0; i < n_info; ++i) {
            const auto &e = info_arr[i];
            if (e.real_buff_idx >= gpu_pool_len && gpu_pool_len > 0)
                continue;  // index out of pool range — unused slot
            char pfx[40];
            snprintf(pfx, sizeof(pfx), "SFN %d.%d cell %2d",
                     file_sfn, file_slot, (int)e.cell_idx);
            printf("%s srsInfo[%2u]: real_buff_idx=%-4u  srs_info_idx=%-3u  rnti=%-5u\n",
                   pfx, i, e.real_buff_idx, e.srs_info_idx, e.rnti);
        }

        // ---------------------------------------------------------------
        // ipc_gpu_pool — one line per active buffer, prefix from matching
        // srs_info entry (first match by real_buff_idx gives cell_idx)
        // ---------------------------------------------------------------
        hdf5hpp::hdf5_dataset ds_gpu  = file.open_dataset("ipc_gpu_pool");
        const size_t   gpu_bytes      = ds_gpu.get_buffer_size_bytes();
        // Each fp16 complex pair is stored as one uint32 (lo=re, hi=im)
        const uint32_t vals_per_buf   = (gpu_buf_size > 0) ? gpu_buf_size / 4u : 0;

        printf("ipc_gpu_pool: %zu B  %u bufs x %u B"
               "  (%u fp16 pairs/buf = %u prg x %u ant x %u layer)\n",
               gpu_bytes, gpu_pool_len, gpu_buf_size,
               vals_per_buf, num_prg, num_gnb_ant, num_ue_layer);

        if (gpu_bytes == 0 || vals_per_buf == 0) {
            printf("  (empty)\n");
            return;
        }

        std::vector<uint8_t> gpu_raw(gpu_bytes);
        ds_gpu.read(gpu_raw.data());

        // Walk srs_info entries in order; for each new real_buff_idx print its buffer once.
        std::vector<uint32_t> seen;
        for (uint32_t i = 0; i < n_info; ++i) {
            const auto &e = info_arr[i];
            if (e.real_buff_idx >= gpu_pool_len && gpu_pool_len > 0) continue;
            if (std::find(seen.begin(), seen.end(), e.real_buff_idx) != seen.end()) continue;
            seen.push_back(e.real_buff_idx);

            const auto *data = reinterpret_cast<const uint32_t *>(
                gpu_raw.data() + (size_t)e.real_buff_idx * gpu_buf_size);

            double sum_sq  = 0.0;
            int    nonzero = 0;
            for (uint32_t j = 0; j < vals_per_buf; ++j) {
                float re = half_to_float((uint16_t)(data[j] & 0xFFFFu));
                float im = half_to_float((uint16_t)(data[j] >> 16));
                double m  = (double)re * re + (double)im * im;
                sum_sq += m;
                if (m > 0.0) ++nonzero;
            }

            char pfx[40];
            snprintf(pfx, sizeof(pfx), "SFN %d.%d cell %2d",
                     file_sfn, file_slot, (int)e.cell_idx);
            printf("%s gpuBuf[%3u]: rms=%.4f  nonzero=%u/%u",
                   pfx, e.real_buff_idx,
                   sqrt(sum_sq / vals_per_buf), nonzero, vals_per_buf);

            int nprint = std::min(max_chanest, (int)vals_per_buf);
            for (int j = 0; j < nprint; ++j) {
                float re = half_to_float((uint16_t)(data[j] & 0xFFFFu));
                float im = half_to_float((uint16_t)(data[j] >> 16));
                printf("  [%d]%+.3f%+.3fi", j, re, im);
            }
            printf("\n");
        }

        if (seen.empty())
            printf("  (no valid srs_info references into pool)\n");

        // ---------------------------------------------------------------
        // chest_buf_pool — one line per entry, prefix from entry.cell_id
        // Only entries with rnti != CV_INVALID_RNTI are shown.
        // ---------------------------------------------------------------
        hdf5hpp::hdf5_dataset ds_chest = file.open_dataset("chest_buf_pool");
        const size_t   chest_bytes  = ds_chest.get_buffer_size_bytes();
        const uint32_t n_chest      = (uint32_t)(chest_bytes / sizeof(ChestBufFlat));

        printf("chest_buf_pool: %zu B  %u entries (%zu B each)\n",
               chest_bytes, n_chest, sizeof(ChestBufFlat));

        std::vector<uint8_t> chest_raw(chest_bytes);
        ds_chest.read(chest_raw.data());
        const auto *chest_arr = reinterpret_cast<const ChestBufFlat *>(chest_raw.data());

        for (uint32_t i = 0; i < n_chest; ++i) {
            const auto &e = chest_arr[i];
            if (e.rnti == CV_INVALID_RNTI_VAL) continue;
            char pfx[40];
            snprintf(pfx, sizeof(pfx), "SFN %d.%d cell %2d",
                     file_sfn, file_slot, (int)e.cell_id);
            printf("%s chestBuf[%3u]: state=%-9s  rnti=%-5u  buf_idx=%-4u"
                   "  usage=%-2u  sfn=%-4u  slot=%-2u"
                   "  prgSize=%-2u  startPrg=%-4u  validPrg=%-4u  nPrg=%-4u\n",
                   pfx, i, chest_state_str(e.state),
                   e.rnti, e.buffer_idx, e.usage,
                   e.sfn, e.slot_f,
                   e.srsPrgSize, e.srsStartPrg, e.srsStartValidPrg, e.srsNValidPrg);
        }

    } catch (const std::exception &e) {
        printf("  ERROR: %s\n", e.what());
    }
}

// ---------------------------------------------------------------------------
// Usage
// ---------------------------------------------------------------------------

static void usage(const char *argv0)
{
    printf("Usage: %s [options]\n\n", argv0);
    printf("  --tv-dir <path>      TV directory (default: $CUBB_SDK/testVectors/cumac/)\n");
    printf("  --cells <N>          Number of cells to read (default: auto-detect)\n");
    printf("  --slot <N>           Only dump slot N (0-19); default: all present slots\n");
    printf("  --mem-sharing        Force GPU-share layout interpretation\n");
    printf("  --no-mem-sharing     Force inline-chanest layout interpretation\n");
    printf("  --solution-only      Skip task_in_buf; only print solution\n");
    printf("  --no-solution        Skip solution file\n");
    printf("  --max-chanest <K>    Max srsChanEst pairs to print per UE (default 8)\n");
    printf("  --cubb-dir <path>    Directory with cubb_srs_buffers_*.h5 files\n");
    printf("                       (auto-detects /tmp/ when not given)\n");
    printf("  --help               Show this message\n");
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char *argv[])
{
    std::string tv_dir;
    std::string cubb_dir;
    int cells             = -1;   // -1 = auto-detect per slot
    int slot_filter       = -1;   // -1 = all
    int mem_sharing_ovr   = -1;   // -1 = auto from HDF5 attr
    bool solution_only    = false;
    bool no_solution      = false;
    int max_chanest       = 8;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if      (arg == "--tv-dir"        && i+1 < argc) { tv_dir = argv[++i]; }
        else if (arg == "--cells"         && i+1 < argc) { cells = std::stoi(argv[++i]); }
        else if (arg == "--slot"          && i+1 < argc) { slot_filter = std::stoi(argv[++i]); }
        else if (arg == "--max-chanest"   && i+1 < argc) { max_chanest = std::stoi(argv[++i]); }
        else if (arg == "--mem-sharing")                  { mem_sharing_ovr = 1; }
        else if (arg == "--no-mem-sharing")               { mem_sharing_ovr = 0; }
        else if (arg == "--solution-only")                { solution_only = true; }
        else if (arg == "--no-solution")                  { no_solution = true; }
        else if (arg == "--cubb-dir"        && i+1 < argc) { cubb_dir = argv[++i]; }
        else if (arg == "--help" || arg == "-h")          { usage(argv[0]); return 0; }
        else { fprintf(stderr, "Unknown argument: %s\n", arg.c_str()); usage(argv[0]); return 1; }
    }

    // Default TV dir
    if (tv_dir.empty()) {
        const char *sdk = getenv("CUBB_SDK");
        tv_dir = sdk ? (std::string(sdk) + "/testVectors/cumac/") : "./testVectors/cumac/";
    }
    if (!tv_dir.empty() && tv_dir.back() != '/') tv_dir += '/';

    printf("TV directory : %s\n", tv_dir.c_str());
    printf("Cells        : %s\n", cells >= 0 ? std::to_string(cells).c_str() : "auto");
    printf("Slot filter  : %s\n", slot_filter >= 0 ? std::to_string(slot_filter).c_str() : "all");
    printf("is_mem_sharing: %s\n\n",
           mem_sharing_ovr < 0 ? "auto (from HDF5 attr)" :
           mem_sharing_ovr ? "forced true" : "forced false");

    int n_dumped = 0;

    // Discover all TV files regardless of SFN (sfn=0 assumption was wrong
    // when schedule_slot_ids spans multiple frames).
    struct TvKey { int sfn, slot; };
    std::vector<TvKey> tv_keys;
    {
        auto all_c0 = glob_files(tv_dir + "muUePairTV_sfn*_slot*_cell0.h5");
        for (const auto &f : all_c0) {
            int fsn = -1, fsl = -1;
            const size_t sep = f.rfind('/');
            const char  *bn  = f.c_str() + (sep == std::string::npos ? 0 : sep + 1);
            if (sscanf(bn, "muUePairTV_sfn%d_slot%d_cell0.h5", &fsn, &fsl) == 2) {
                if (slot_filter < 0 || fsl == slot_filter)
                    tv_keys.push_back({fsn, fsl});
            }
        }
        std::sort(tv_keys.begin(), tv_keys.end(), [](const TvKey &a, const TvKey &b){
            return a.sfn < b.sfn || (a.sfn == b.sfn && a.slot < b.slot);
        });
    }

    for (const auto &k : tv_keys) {
        const int sfn  = k.sfn;
        const int slot = k.slot;

        // Auto-detect number of cell files when --cells not given
        int n_cells = cells;
        if (n_cells < 0) {
            n_cells = 0;
            char cp[1024];
            while (true) {
                snprintf(cp, sizeof(cp), "%smuUePairTV_sfn%d_slot%d_cell%d.h5",
                         tv_dir.c_str(), sfn, slot, n_cells);
                if (access(cp, F_OK) != 0) break;
                ++n_cells;
            }
        }

        print_sep();
        printf("SFN %d.%d  (muUePairTV_sfn%d_slot%d_cell*.h5, %d cell(s))\n",
               sfn, slot, sfn, slot, n_cells);
        print_sep();

        for (int c = 0; c < n_cells; ++c) {
            char fpath[1024];
            snprintf(fpath, sizeof(fpath), "%smuUePairTV_sfn%d_slot%d_cell%d.h5",
                     tv_dir.c_str(), sfn, slot, c);

            if (access(fpath, F_OK) != 0) {
                printf("\n[cell %d] not found: %s\n", c, fpath);
                continue;
            }

            char pfx[32];
            snprintf(pfx, sizeof(pfx), "SFN %d.%d cell %2d", sfn, slot, c);

            printf("\n[cell %d] %s\n", c, fpath);
            print_sep('-', 60);

            try {
                hdf5hpp::hdf5_file file = hdf5hpp::hdf5_file::open(fpath);
                hid_t fid = file.id();

                uint8_t  a_mem_shr = 0;
                uint16_t a_srs_ue  = 0;
                uint8_t  a_flags   = 0;
                int32_t  a_blk_row = 0;
                int32_t  a_ue_ant  = 0;
                int32_t  a_srs_buf = 0;

                h5_attr_u8 (fid, "is_mem_sharing",              a_mem_shr);
                h5_attr_u16(fid, "num_srs_ue_per_slot_cell",    a_srs_ue);
                h5_attr_u8 (fid, "kernel_launch_flags",         a_flags);
                h5_attr_i32(fid, "num_blocks_per_row_chanOrtMat", a_blk_row);
                h5_attr_i32(fid, "num_ue_ant_port",             a_ue_ant);
                h5_attr_i32(fid, "num_srs_buffers",             a_srs_buf);

                printf("Attributes:\n");
                printf("  is_mem_sharing             = %d\n",    (int)a_mem_shr);
                printf("  num_srs_ue_per_slot_cell   = %u\n",    (unsigned)a_srs_ue);
                printf("  kernel_launch_flags        = 0x%02X\n",(unsigned)a_flags);
                printf("  num_blocks_per_row_chanOrtMat = %d\n", a_blk_row);
                printf("  num_ue_ant_port            = %d\n",    a_ue_ant);
                printf("  num_srs_buffers            = %d\n",    a_srs_buf);

                const bool is_msh = (mem_sharing_ovr >= 0) ? (mem_sharing_ovr == 1)
                                                            : (a_mem_shr != 0);
                printf("  [effective is_mem_sharing  = %s]\n", is_msh ? "true" : "false");

                if (solution_only) continue;

                // -----------------------------------------------------------
                // task_in_buf
                // -----------------------------------------------------------
                const size_t expected_sz = cumac_muUeGrp_req_info_size(is_msh);
                std::vector<uint8_t> buf(expected_sz, 0);

                hdf5hpp::hdf5_dataset ds = file.open_dataset("task_in_buf");
                const size_t actual_sz = ds.get_buffer_size_bytes();

                if (actual_sz != expected_sz) {
                    printf("\n  WARNING: task_in_buf HDF5 size %zu != expected %zu "
                           "(is_mem_sharing may be wrong)\n", actual_sz, expected_sz);
                }
                buf.resize(actual_sz);
                ds.read(buf.data());

                const auto *req = reinterpret_cast<const cumac_muUeGrp_req_info_t *>(buf.data());

                printf("\ntask_in_buf (%zu bytes):\n", actual_sz);
                print_req_info_header(req, is_msh);

                const uint16_t num_srs = req->numSrsInfo;
                const uint16_t num_ue  = req->numUeInfo;

                // payload layout: srsInfo[numSrsInfo] then ueInfo[numUeInfo]
                // (matches kernel in muMimoUserPairing.cu)
                if (is_msh) {
                    const auto *srs = reinterpret_cast<const cumac_muUeGrp_req_srs_info_msh_t *>(
                        req->payload);
                    printf("\nSRS info (mem-sharing, numSrsInfo=%u, max=%u):\n",
                           num_srs, (unsigned)MAX_NUM_UE_SRS_INFO_PER_SLOT);
                    for (int i = 0; i < (int)num_srs; ++i)
                        print_srs_msh(&srs[i], i, pfx);

                    const auto *ue = reinterpret_cast<const cumac_muUeGrp_req_ue_info_t *>(
                        srs + num_srs);
                    printf("\nUE info (numUeInfo=%u, max=%u):\n",
                           num_ue, (unsigned)MAX_NUM_SRS_UE_PER_CELL);
                    for (int i = 0; i < (int)num_ue; ++i)
                        print_ue_info(&ue[i], i, pfx);
                } else {
                    const auto *srs = reinterpret_cast<const cumac_muUeGrp_req_srs_info_t *>(
                        req->payload);
                    printf("\nSRS info (inline chanEst, numSrsInfo=%u, max=%u):\n",
                           num_srs, (unsigned)MAX_NUM_UE_SRS_INFO_PER_SLOT);
                    for (int i = 0; i < (int)num_srs; ++i) {
                        print_srs_inline(&srs[i], i,
                                         req->nBsAnt,
                                         a_ue_ant > 0 ? (int)a_ue_ant : (int)srs[i].nUeAnt,
                                         req->numSubband,
                                         req->numPrgSampPerSubband,
                                         max_chanest, pfx);
                    }

                    const auto *ue = reinterpret_cast<const cumac_muUeGrp_req_ue_info_t *>(
                        srs + num_srs);
                    printf("\nUE info (numUeInfo=%u, max=%u):\n",
                           num_ue, (unsigned)MAX_NUM_SRS_UE_PER_CELL);
                    for (int i = 0; i < (int)num_ue; ++i)
                        print_ue_info(&ue[i], i, pfx);
                }

            } catch (const std::exception &e) {
                printf("  ERROR processing %s: %s\n", fpath, e.what());
            }
        } // cells

        // -------------------------------------------------------------------
        // Solution file
        // -------------------------------------------------------------------
        if (!no_solution) {
            char sol_path[1024];
            snprintf(sol_path, sizeof(sol_path), "%smuUePairTV_sfn%d_slot%d_solution.h5",
                     tv_dir.c_str(), sfn, slot);

            printf("\nSolution: %s\n", sol_path);
            print_sep('-', 60);

            if (access(sol_path, F_OK) != 0) {
                printf("  (not found)\n");
            } else {
                try {
                    hdf5hpp::hdf5_file sfile = hdf5hpp::hdf5_file::open(sol_path);
                    hdf5hpp::hdf5_dataset ds_sol = sfile.open_dataset("solution");
                    const size_t sol_bytes = ds_sol.get_buffer_size_bytes();
                    const int ncells_sol = (int)(sol_bytes / sizeof(cumac_muUeGrp_resp_info_t));
                    printf("  solution dataset: %zu bytes → %d cell(s)\n", sol_bytes, ncells_sol);

                    std::vector<uint8_t> sol_buf(sol_bytes);
                    ds_sol.read(sol_buf.data());

                    for (int c = 0; c < ncells_sol; ++c) {
                        const auto *sol = reinterpret_cast<const cumac_muUeGrp_resp_info_t *>(
                            sol_buf.data() + c * sizeof(cumac_muUeGrp_resp_info_t));
                        char sol_pfx[32];
                        snprintf(sol_pfx, sizeof(sol_pfx), "SFN %d.%d cell %2d", sfn, slot, c);
                        print_solution(sol, sol_pfx);
                    }
                } catch (const std::exception &e) {
                    printf("  ERROR reading solution: %s\n", e.what());
                }
            }
        }

        printf("\n");
        ++n_dumped;
    }

    // -----------------------------------------------------------------------
    // cuBB TV dump
    // -----------------------------------------------------------------------
    {
        std::string search_dir = cubb_dir;
        if (search_dir.empty()) {
            if (!glob_files("/tmp/cubb_srs_buffers_*.h5").empty())
                search_dir = "/tmp/";
        }
        if (!search_dir.empty()) {
            if (search_dir.back() != '/') search_dir += '/';
            const auto cubb_files = glob_files(search_dir + "cubb_srs_buffers_*.h5");
            if (!cubb_files.empty()) {
                print_sep();
                printf("cuBB TV files: %zu found in %s\n",
                       cubb_files.size(), search_dir.c_str());
                print_sep();
                for (const auto &cf : cubb_files)
                    dump_cubb_file(cf, max_chanest);
                printf("\n");
            } else {
                printf("(no cubb_srs_buffers_*.h5 found in %s)\n", search_dir.c_str());
            }
        }
    }

    printf("Total slots dumped: %d\n", n_dumped);
    return (n_dumped > 0) ? 0 : 1;
}

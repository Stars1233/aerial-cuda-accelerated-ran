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

#ifndef _CUMAC_CP_TV_HPP_
#define _CUMAC_CP_TV_HPP_

#include <unistd.h>

#include "api.h"
#include "cumac_app.hpp"
#include "cumac_pfm_sort.h"
#include "cumac_msg.h"

#include "hdf5hpp.hpp"
#include "cuphy_hdf5.hpp"
#include "nvlog.hpp"

#include <array>
#include <chrono>
#include <cstddef>
#include <vector>

#define CONFIG_CUMAC_TV_PATH "testVectors/cumac/"

//! Host-staged MU UE grouping / UE-pair test vector (muUePairTV_* HDF5; see load_h5_tv_cumacCp / load_h5_solution_tv)
typedef struct ue_pair_tv
{
    bool mu_ue_pair_tv_loaded{false};
    uint16_t num_prg{0};
    uint16_t num_subband{0};
    uint16_t num_prg_samp_per_subband{0};
    uint16_t num_bs_ant{0};
    uint16_t num_ue_ant{0};
    uint16_t num_srs_ue_per_slot_cell{0};
    uint16_t num_blocks_per_row_chanOrtMat{0};
    uint8_t kernel_launch_flags{0};
    //! Mirrors YAML `enable_gpu_share` when MU TV is loaded (HDF5 `is_mem_sharing` is not used for layout).
    bool is_mem_sharing{false};
    std::vector<cumac_muUeGrp_resp_info_t> muUeGrpSol; //!< Expected output, size cell_num when loaded
    //! Pinned host buffers (cudaMallocHost); released in destructor / ue_pair_tv_release_mu_host_buffers
    uint8_t *srs_chan_est_host{nullptr};
    float *srs_snr_host{nullptr};
    float *chan_orth_host{nullptr};
    uint8_t *cubb_srs_buf_host{nullptr};
    uint8_t *task_in_buf_group_host{nullptr};

    size_t srs_chan_est_size{0};
    size_t srs_snr_size{0};
    size_t chan_orth_size{0};
    size_t cubb_srs_buf_size{0};
    size_t task_in_buf_group_size{0};

    ~ue_pair_tv();
    ue_pair_tv() = default;
    ue_pair_tv(const ue_pair_tv &) = delete;
    ue_pair_tv &operator=(const ue_pair_tv &) = delete;
    ue_pair_tv(ue_pair_tv &&) noexcept;
    ue_pair_tv &operator=(ue_pair_tv &&) noexcept;
} ue_pair_tv_t;

//! Free MU UE pair pinned host buffers (safe if already null).
void ue_pair_tv_release_mu_host_buffers(ue_pair_tv_t &tv);

typedef struct cumac_cp_tv
{
    uint32_t parsed = 0;

    struct cumac::cumacSchedulerParam params{};
    cumac_buf_num_t buf_num{};

    float *avgRates = nullptr;
    uint8_t *cellAssoc = nullptr;
    uint8_t *cellAssocActUe = nullptr;
    int8_t *tbErrLast = nullptr;
    uint16_t *cellId = nullptr;

    // data buffer pointers
    uint16_t *CRNTI = nullptr;                 // C-RNTIs of all active UEs in the cell
    uint16_t *srsCRNTI = nullptr;              // C-RNTIs of the UEs that have refreshed SRS channel estimates in the cell.
    uint8_t *prgMsk = nullptr;                 // Bit map for the availability of each PRG for allocation
    float *postEqSinr = nullptr;               // Array of the per-PRG per-layer post-equalizer SINRs of all active UEs in the cell
    float *wbSinr = nullptr;                   // Array of wideband per-layer post-equalizer SINRs of all active UEs in the cell
    cuComplex *estH_fr = nullptr;              // For FP32. Array of the subband (per-PRG) SRS channel estimate coefficients for all active UEs in the cell
    cuComplex *estH_fr_half = nullptr;         // For FP16. Array of the subband (per-PRG) SRS channel estimate coefficients for all active UEs in the cell
    cuComplex *prdMat = nullptr;               // Array of the precoder/beamforming weights for all active UEs in the cell
    cuComplex *detMat = nullptr;               // Array of the detector/beamforming weights for all active UEs in the cell
    float *sinVal = nullptr;                   // Array of the per-UE, per-PRG, per-layer singular values obtained from the SVD of the channel matrix
    float *avgRatesActUe = nullptr;            // Array of the long-term average data rates of all active UEs in the cell
    uint16_t *prioWeightActUe = nullptr;       // For priority-based UE selection. Priority weights of all active UEs in the cell
    int8_t *tbErrLastActUe = nullptr;          // TB decoding error indicators of all active UEs in the cell
    int8_t *newDataActUe = nullptr;            // Indicators of initial transmission/retransmission for all active UEs in the cell
    int16_t *allocSolLastTxActUe = nullptr;    // The PRG allocation solution for the last transmissions of all active UEs in the cell
    int16_t *mcsSelSolLastTxActUe = nullptr;   // MCS selection solution for the last transmissions of all active UEs in the cell
    uint8_t *layerSelSolLastTxActUe = nullptr; //

    uint16_t *setSchdUePerCellTTI = nullptr; // Set of IDs of the selected UEs for the cell
    int16_t *allocSol = nullptr;             // PRB group allocation solution for all active UEs in the cell
    int16_t *mcsSelSol = nullptr;            // MCS selection solution for all active UEs in the cell
    uint8_t *layerSelSol = nullptr;          // Layer selection solution for all active UEs in the cell

    std::vector<cumac_pfm_cell_info_t> pfmCellInfo; // PFM sorting input buffer
    std::vector<cumac_pfm_output_cell_info_t> pfmSortSol; // PFM sorting output buffer

    std::vector<ue_pair_tv_t> ue_pair; // Per-slot MU TV indexed by `sfn * SLOT_NUM_PER_FRAME + slot`
} cumac_cp_tv_t;

int parse_4t4r_tv(cumac_cp_tv_t &tv, std::string tv_file);
/**
 * @brief Load cuMAC group test vectors for the modules enabled by task_bitmask.
 *
 * @param tv Destination TV container populated with 4T4R, PFM sort, and/or MU UE grouping data.
 * @param cell_num Number of cells expected in the TV files.
 * @param enable_gpu_share Whether MU UE grouping TV parsing should use GPU-share buffer layout.
 * @param srs_slot_lag Slot offset applied when mapping MU UE grouping TV files; defaults to 0.
 * @param task_bitmask cuMAC task enable mask; defaults to CUMAC_CP_TASK_MASK_DEFAULT.
 * @return 0 on success, negative value when a required enabled TV cannot be loaded.
 */
int parse_group_tv(cumac_cp_tv_t &tv, int cell_num, bool enable_gpu_share, int srs_slot_lag = 0, uint32_t task_bitmask = CUMAC_CP_TASK_MASK_DEFAULT);
//! Load MU UE pairing HDF5 TVs for one radio slot into \a ue_pair_tv (mu_pair_*_host, muUeGrpSol).
int load_mu_ue_pair_group_tv(ue_pair_tv_t &ue_pair_tv, int cell_num, int sfn, int slot, bool enable_gpu_share);
int check_bytes(const char *name1, const char *name2, void *buf1, void *buf2, size_t nbytes);

bool pfm_load_tv_H5(const std::string& tv_name, std::vector<cumac_pfm_cell_info_t>& pfm_cell_info, std::vector<cumac_pfm_output_cell_info_t>& pfm_output_cell_info);
bool pfm_validate_tv_h5(const std::string& tv_name, const std::vector<cumac_pfm_cell_info_t>& pfm_cell_info, const std::vector<cumac_pfm_output_cell_info_t>& pfm_output_cell_info);

#define CUMAC_VALIDATE_BYTES(buf1, buf2, nbytes) check_bytes(#buf1, #buf2, (buf1), (buf2), nbytes)

#endif // _CUMAC_CP_TV_HPP_
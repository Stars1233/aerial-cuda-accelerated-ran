# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for pdsch_tx.py."""
import glob
import pytest
from pytest import TEST_VECTOR_DIR

import h5py as h5
import cupy as cp
import numpy as np

from aerial.phy5g.pdsch import PdschTx


def test_pdsch_tx(pdsch_tx):  # pylint: disable=unused-argument
    """Tests initialization and destruction of PdschTx."""
    # The fixture initializes and it gets destroyed when
    # it goes out of scope of this function.


test_case_numbers = [3001, 3002, 3003, 3004, 3005, 3006, 3007, 3008, 3009, 3010, 3011, 3012, 3013,
                     3014, 3015, 3018, 3019, 3020, 3021, 3022, 3023, 3026, 3027, 3028, 3029, 3030,
                     3031, 3033, 3034, 3035, 3036, 3037, 3038, 3040, 3041, 3042, 3043, 3044, 3046,
                     3047, 3048, 3049, 3050, 3051, 3052, 3053, 3054, 3055, 3056, 3058, 3059, 3060,
                     3061, 3062, 3063, 3064, 3065, 3066, 3067, 3068, 3069, 3070, 3071, 3072, 3073,
                     3074, 3075, 3076, 3077, 3078, 3079, 3080, 3081, 3082, 3083, 3084, 3085, 3086,
                     3087, 3088, 3089, 3090, 3091, 3092, 3093, 3094, 3095, 3096, 3097, 3098, 3099,
                     3100, 3101, 3102, 3103, 3104, 3105, 3106, 3107, 3108, 3109, 3110, 3111, 3112,
                     3113, 3114, 3115, 3116, 3117, 3118, 3119, 3120, 3121, 3122, 3123, 3124, 3125,
                     3126, 3127, 3128, 3129, 3130, 3131, 3132, 3133, 3134, 3135, 3136, 3137, 3138,
                     3139, 3140, 3141, 3142, 3143, 3144, 3145, 3146, 3147, 3148, 3149, 3150, 3151,
                     3152, 3153, 3154, 3201, 3202, 3203, 3204, 3205, 3206, 3207, 3208, 3209, 3210,
                     3211, 3212, 3213, 3214, 3215, 3216, 3217, 3218, 3219, 3220, 3221, 3222, 3223,
                     3224, 3225, 3226, 3227, 3228, 3229, 3230, 3231, 3232, 3233, 3234, 3235, 3236,
                     3237, 3238, 3239, 3240, 3241, 3242, 3243, 3244, 3246, 3247, 3248, 3249, 3250,
                     3251, 3252, 3253, 3254, 3255, 3256, 3257, 3258, 3259, 3260, 3261, 3262, 3264,
                     3265, 3266, 3267, 3268, 3269, 3271, 3272, 3273, 3274, 3275, 3276, 3277, 3278,
                     3279, 3280, 3281, 3282, 3283, 3284, 3285, 3286, 3287, 3288, 3289, 3290, 3292,
                     3293, 3294, 3295, 3297, 3298, 3299, 3301, 3302, 3303, 3304, 3305, 3306, 3307,
                     3308, 3309, 3310, 3311, 3312, 3313, 3314, 3315, 3316, 3317, 3318, 3319, 3320,
                     3321, 3322, 3323, 3324, 3325, 3326, 3327, 3328, 3329, 3330, 3331, 3332, 3333,
                     3334, 3335, 3336, 3337, 3338, 3339, 3340, 3341, 3343, 3344, 3350, 3351, 3352,
                     3353, 3401, 3402, 3403, 3404, 3405, 3406, 3407, 3408, 3409, 3410, 3411, 3412,
                     3413, 3422, 3501, 3502, 3504, 3505, 3506, 3509, 3510, 3513, 3521, 3801, 3802,
                     3803, 3804, 3805, 3901, 3902, 3903, 3904, 3906, 3907, 3908, 3909, 3910, 3911,
                     3912, 3913, 3914, 3915, 3916, 3917, 3918, 3919, 3920, 3921, 3922, 3923, 3924,
                     3925, 3926, 3927, 3928, 3929, 3930, 3931, 3932, 3933, 3934, 3935, 3936, 3937,
                     3938, 3939, 3940, 3941, 3942, 3943, 3944, 3945, 3946, 3947, 3948, 3949, 3950,
                     3951, 3952, 3953, 3954, 3955, 3956, 3957, 3958, 3959, 3960, 3961, 3962, 3963,
                     3964, 3965, 3966, 3967, 3968, 3969, 3970, 3971, 3972, 3973, 3974, 3975, 3976,
                     3977, 3978, 3979, 3980, 3981, 3982, 3983, 3984, 3985, 3986, 3987, 3988, 3989,
                     3990, 3991, 3992, 3993, 3994, 3995, 3996, 3997, 3998, 3999]

# TCs with multiple UE groups, only supported with the config-based API.
tc_fdm = [3233, 3234, 3235, 3239, 3240, 3246, 3247, 3256, 3257, 3259, 3260, 3261, 3262, 3263,
          3264, 3265, 3266, 3267, 3268, 3269, 3277, 3278, 3286, 3287, 3288, 3330, 3334, 3335,
          3336, 3337, 3338, 3339, 3340, 3341, 3971]

# Skipped TVs.
# Testing mode not supported in Python API.
tc_to_skip = [3296, 3354, 3355]
# Modulation compression not supported.
tc_to_skip += [3263, 3281, 3291]
test_case_numbers = list(set(list(test_case_numbers)) - set(tc_to_skip) - set(tc_fdm))


# pylint: disable=too-many-locals
@pytest.mark.parametrize(
    "test_case_number",
    test_case_numbers,
    ids=[int(test_case_number) for test_case_number in test_case_numbers]
)
def test_pdsch_tx_run(csi_rs_config, cuda_stream, test_case_number):
    """Test running PDSCH transmission against Aerial test vectors.

    This is testing the single UE group API where parameters are given separately.
    Input and output are Numpy arrays in this case.
    """
    filename = glob.glob(TEST_VECTOR_DIR + f"TVnr_{test_case_number}_PDSCH_gNB_CUPHY_s*.h5")[0]
    try:
        input_file = h5.File(filename, "r")
    except FileNotFoundError:
        pytest.skip("Test vector file not available, skipping...")
        return

    cell_id = input_file["cellStat_pars"]["phyCellId"][0]
    num_rx_ant = input_file["cellStat_pars"]["nRxAnt"][0]
    num_tx_ant = input_file["cellStat_pars"]["nTxAnt"][0]
    num_prb_dl_bwp = input_file["cellStat_pars"]["nPrbDlBwp"][0]
    num_prb_ul_bwp = input_file["cellStat_pars"]["nPrbUlBwp"][0]
    mu = input_file["cellStat_pars"]["mu"][0]

    num_ues = input_file["cellGrpDyn_pars"]["nUes"][0]
    tb_inputs = [np.packbits(np.array(input_file["tb" + str(ue_idx) +
                 "_inputdata"]).astype(np.uint8)) for ue_idx in range(num_ues)]
    slot = input_file["cellDyn_pars"]["slotNum"][0]

    num_dmrs_cdm_grps_no_data = input_file["dmrs_pars"]["nDmrsCdmGrpsNoData"][0]
    resource_alloc = input_file["ueGrp_pars"]["resourceAlloc"][0]
    prb_bitmap = input_file["ueGrp_pars"]["rbBitmap"][0]
    start_prb = input_file["ueGrp_pars"]["startPrb"][0]
    num_prbs = input_file["ueGrp_pars"]["nPrb"][0]
    dmrs_syms = input_file["ueGrp_pars"]["dmrsSymLocBmsk"][0]
    dmrs_syms = [int(b) for b in list(bin(dmrs_syms)[2:].zfill(14))[::-1]]
    start_sym = input_file["ueGrp_pars"]["pdschStartSym"][0]
    num_symbols = input_file["ueGrp_pars"]["nPdschSym"][0]

    scids = input_file["ue_pars"]["scid"]
    dmrs_scrm_ids = input_file["ue_pars"]["dmrsScramId"]
    layers = input_file["ue_pars"]["nUeLayers"]
    dmrs_ports = input_file["ue_pars"]["dmrsPortBmsk"]
    bwp_starts = input_file["ue_pars"]["BWPStart"]
    ref_points = input_file["ue_pars"]["refPoint"]
    rntis = input_file["ue_pars"]["rnti"]
    data_scids = input_file["ue_pars"]["dataScramId"]
    if np.array(input_file["tb0_PM_W"]).size > 0:
        precoding_matrices = \
            [np.array(input_file["tb" + str(ue_idx) + "_PM_W"]["re"]) +
             1j * np.array(input_file["tb" + str(ue_idx) + "_PM_W"]["im"])
             for ue_idx in range(num_ues)]
    else:
        precoding_matrices = None

    mcs_tables = input_file["cw_pars"]["mcsTableIndex"]
    mcs_indices = input_file["cw_pars"]["mcsIndex"]
    code_rates = input_file["cw_pars"]["targetCodeRate"]
    mod_orders = input_file["cw_pars"]["qamModOrder"]
    rvs = input_file["cw_pars"]["rv"]
    num_prb_lbrms = input_file["cw_pars"]["n_PRB_LBRM"]
    max_layers = input_file["cw_pars"]["maxLayers"]
    max_qms = input_file["cw_pars"]["maxQm"]

    csi_rs_configs = csi_rs_config(input_file)

    pdsch_tx = PdschTx(
        cell_id=cell_id,
        num_rx_ant=num_rx_ant,
        num_tx_ant=num_tx_ant,
        num_ul_bwp=num_prb_ul_bwp,
        num_dl_bwp=num_prb_dl_bwp,
        mu=mu,
        cuda_stream=cuda_stream
    )

    xtf = pdsch_tx.run(
        tb_inputs=tb_inputs,
        num_ues=num_ues,
        slot=slot,

        num_dmrs_cdm_grps_no_data=num_dmrs_cdm_grps_no_data,
        resource_alloc=resource_alloc,
        prb_bitmap=prb_bitmap,
        start_prb=start_prb,
        num_prbs=num_prbs,
        dmrs_syms=dmrs_syms,
        start_sym=start_sym,
        num_symbols=num_symbols,

        scids=scids,
        dmrs_scrm_ids=dmrs_scrm_ids,
        layers=layers,
        dmrs_ports=dmrs_ports,
        bwp_starts=bwp_starts,
        ref_points=ref_points,
        rntis=rntis,
        data_scids=data_scids,
        precoding_matrices=precoding_matrices,

        mcs_tables=mcs_tables,
        mcs_indices=mcs_indices,
        code_rates=code_rates,
        mod_orders=mod_orders,
        rvs=rvs,
        num_prb_lbrms=num_prb_lbrms,
        max_layers=max_layers,
        max_qms=max_qms,

        csi_rs_configs=csi_rs_configs
    )

    ref_xtf = np.array(input_file["Xtf"]["re"]) + 1j * np.array(input_file["Xtf"]["im"])
    if ref_xtf.ndim == 2:
        ref_xtf = ref_xtf[None]
    ref_xtf = ref_xtf.transpose(2, 1, 0)
    ref_xtf = PdschTx.cuphy_to_tx(
        tx_slot=ref_xtf,
        num_ues=num_ues,
        dmrs_ports=dmrs_ports,
        scids=scids,
        precoding_matrices=precoding_matrices
    )

    assert np.allclose(xtf, ref_xtf, atol=1e-3)


api_test_case_numbers = test_case_numbers + tc_fdm


@pytest.mark.parametrize(
    "test_case_number",
    api_test_case_numbers,
    ids=[int(test_case_number) for test_case_number in api_test_case_numbers]
)
def test_pdsch_config_api(csi_rs_config, pdsch_config, cuda_stream, test_case_number):
    """Test running PDSCH transmission against Aerial test vectors.

    This is testing the pipeline API where parameters are given as configs.
    Input and output are CuPy arrays in this case.
    """
    filename = glob.glob(TEST_VECTOR_DIR + f"TVnr_{test_case_number}_PDSCH_gNB_CUPHY_s*.h5")[0]
    try:
        input_file = h5.File(filename, "r")
    except FileNotFoundError:
        pytest.skip("Test vector file not available, skipping...")
        return

    cell_id = input_file["cellStat_pars"]["phyCellId"][0]
    num_rx_ant = input_file["cellStat_pars"]["nRxAnt"][0]
    num_tx_ant = input_file["cellStat_pars"]["nTxAnt"][0]
    num_prb_dl_bwp = input_file["cellStat_pars"]["nPrbDlBwp"][0]
    num_prb_ul_bwp = input_file["cellStat_pars"]["nPrbUlBwp"][0]
    mu = input_file["cellStat_pars"]["mu"][0]

    num_ues = input_file["cellGrpDyn_pars"]["nUes"][0]
    dmrs_ports = input_file["ue_pars"]["dmrsPortBmsk"]
    scids = input_file["ue_pars"]["scid"]
    if np.array(input_file["tb0_PM_W"]).size > 0:
        precoding_matrices = \
            [np.array(input_file["tb" + str(ue_idx) + "_PM_W"]["re"]) +
             1j * np.array(input_file["tb" + str(ue_idx) + "_PM_W"]["im"])
             for ue_idx in range(num_ues)]
    else:
        precoding_matrices = None

    tb_inputs = [np.packbits(np.array(input_file["tb" + str(ue_idx) +
                 "_inputdata"]).astype(np.uint8)) for ue_idx in range(num_ues)]

    slot = input_file["cellDyn_pars"]["slotNum"][0]

    csi_rs_configs = csi_rs_config(input_file)
    pdsch_configs = pdsch_config(input_file)

    pdsch_tx = PdschTx(
        cell_id=cell_id,
        num_rx_ant=num_rx_ant,
        num_tx_ant=num_tx_ant,
        num_ul_bwp=num_prb_ul_bwp,
        num_dl_bwp=num_prb_dl_bwp,
        mu=mu,
        cuda_stream=cuda_stream
    )

    with cuda_stream:
        tb_inputs = [cp.array(tb, dtype=cp.uint8, order='F') for tb in tb_inputs]

    xtf = pdsch_tx(
        slot=slot,
        tb_inputs=tb_inputs,
        config=pdsch_configs,
        csi_rs_config=csi_rs_configs
    )

    xtf = xtf.get(order='F')

    ref_xtf = np.array(input_file["Xtf"]["re"]) + 1j * np.array(input_file["Xtf"]["im"])
    if ref_xtf.ndim == 2:
        ref_xtf = ref_xtf[None]
    ref_xtf = ref_xtf.transpose(2, 1, 0)
    ref_xtf = PdschTx.cuphy_to_tx(
        tx_slot=ref_xtf,
        num_ues=num_ues,
        dmrs_ports=dmrs_ports,
        scids=scids,
        precoding_matrices=precoding_matrices
    )

    assert np.allclose(xtf, ref_xtf, atol=1e-3)

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

"""Tests for csirs_4x.py."""
import glob
import itertools
import pytest
from pytest import TEST_VECTOR_DIR

import cupy as cp
import h5py as h5
import numpy as np

from aerial.phy5g.csirs import CsiRsConfig
from aerial.phy5g.csirs import CsiRsRxConfig
from aerial.phy5g.csirs import CsiRsRx


# Test vector numbers for CSI-RS.
test_case_numbers = list(range(4001, 4063))
test_case_numbers += [4101, 4102, 4103]
test_case_numbers += list(range(4201, 4223))
test_case_numbers += list(range(4801, 4808))
test_case_numbers += list(range(4901, 4906))

# TV numbers to skip - no UE side TVs.
tc_to_skip = [4056, 4057, 4062, 4802, 4901, 4902]
test_case_numbers = list(set(test_case_numbers) - set(tc_to_skip))

# pylint: disable=too-many-locals
all_cases = list(itertools.product(test_case_numbers, [True, False]))


@pytest.mark.parametrize(
    "test_case_number, cupy",
    all_cases,
    ids=[f"{test_case_number} - cuPy: {cupy}" for test_case_number, cupy in all_cases]
)
def test_csirs_rx_run(cuda_stream, test_case_number, cupy):
    """Test running CSI-RS Rx against Aerial test vectors."""
    filename = glob.glob(TEST_VECTOR_DIR + f"TVnr_{test_case_number}_CSIRS_UE_CUPHY_s*.h5")[0]
    try:
        input_file = h5.File(filename, "r")
    except FileNotFoundError:
        pytest.skip("Test vector file not available, skipping...")
        return

    csirs_prms_list = input_file["CsirsParamsList"]

    csirs_configs = []
    for csirs_prms in csirs_prms_list:
        csirs_configs.append(CsiRsConfig(
            start_prb=np.uint16(csirs_prms["StartRB"]),
            num_prb=np.uint16(csirs_prms["NrOfRBs"]),
            freq_alloc=list(map(int, format(csirs_prms["FreqDomain"], "016b"))),
            row=np.uint8(csirs_prms["Row"]),
            symb_L0=np.uint8(csirs_prms["SymbL0"]),
            symb_L1=np.uint8(csirs_prms["SymbL1"]),
            freq_density=np.uint8(csirs_prms["FreqDensity"]),
            scramb_id=np.uint16(csirs_prms["ScrambId"]),
            idx_slot_in_frame=np.uint8(csirs_prms["idxSlotInFrame"]),
            cdm_type=np.uint8(csirs_prms["CDMType"]),
            beta=csirs_prms["beta"]
        ))

    rx_data = np.array(input_file["X_tf"])["re"] + 1j * np.array(input_file["X_tf"])["im"]
    rx_data = rx_data.transpose(2, 1, 0)
    if cupy:
        with cuda_stream:
            rx_data = cp.array(rx_data, order='F', dtype=cp.complex64)

    csirs_rx = CsiRsRx(num_prb_dl_bwp=[rx_data.shape[0] // 12], cuda_stream=cuda_stream)
    csirs_rx_config = CsiRsRxConfig(csirs_configs=[csirs_configs], ue_cell_association=[0])

    ch_est = csirs_rx(rx_data=[rx_data], config=csirs_rx_config)

    if cupy:
        ch_est = [[buf.get() for buf in ue_ch_est] for ue_ch_est in ch_est]

    ref_ch_est = np.array(input_file["Csirs_Hest0"])["re"] + \
        1j * np.array(input_file["Csirs_Hest0"])["im"]
    ref_ch_est = ref_ch_est.transpose(2, 1, 0)
    assert np.allclose(ch_est[0][0], ref_ch_est)

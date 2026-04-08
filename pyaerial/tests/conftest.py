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

"""The conftest.py file for pytest. Contains common fixtures etc."""
import gc
import os
import pytest

import cuda.bindings.runtime as cudart
import cupy as cp
import numpy as np

from aerial import pycuphy
from aerial.util.cuda import CudaStream
from aerial.util.fapi import dmrs_fapi_to_bit_array
from aerial.phy5g.pdsch import PdschTx
from aerial.phy5g.pusch import PuschRx
from aerial.phy5g.csirs import CsiRsConfig
from aerial.phy5g.config import PuschConfig
from aerial.phy5g.config import PuschUeConfig
from aerial.phy5g.config import PdschConfig
from aerial.phy5g.config import PdschUeConfig
from aerial.phy5g.config import PdschCwConfig


@pytest.fixture(scope="function", autouse=True)
def clear_session():
    """Force garbage collection at the end of a test."""
    # Yield and let test run to completion.
    yield
    cp._default_memory_pool.free_all_blocks()
    gc.collect()


def pytest_configure():
    """Configure pytest."""
    pycuphy.set_nvlog_level(0)
    pytest.TEST_VECTOR_DIR = os.environ.get(
        "TEST_VECTOR_DIR",
        "/mnt/cicd_tvs/develop/GPU_test_input/"
    )


@pytest.fixture(name="cuda_stream", scope="function")
def fixture_cuda_stream():
    """A fixture for setting up CUDA and creating a CUDA stream.

    Yields a CudaStream. Use ``with cuda_stream:`` in the test to set it as
    the current stream (CudaStream is not re-entrant). Stream is destroyed
    when the CudaStream is GC'd.
    """
    # TODO: Figure out how to set the GPU in CICD.
    gpu_id = 0
    cudart.cudaSetDevice(gpu_id)

    stream = CudaStream()
    yield stream


@pytest.fixture(name="pdsch_tx", scope="function")
def fixture_pdsch_tx():
    """Fixture for creating a PdschTx object."""
    pdsch_tx = PdschTx(
        cell_id=41,
        num_rx_ant=4,
        num_tx_ant=4,
    )
    return pdsch_tx


@pytest.fixture(name="pusch_rx", scope="function")
def fixture_pusch_rx():
    """Fixture for creating a PuschRx object."""
    pusch_rx = PuschRx(
        cell_id=41,
        num_rx_ant=4,
        num_tx_ant=4,
    )
    return pusch_rx


@pytest.fixture(name="pusch_config", scope="module")
def fixture_pusch_config():
    """Fixture that returns a function for getting PuschConfig out of a test vector file."""
    def _pusch_config(test_vector_file):

        num_ue_grps = len(test_vector_file["ueGrp_pars"])
        num_ues = len(test_vector_file["tb_pars"])

        pusch_configs = []
        pusch_ue_configs = [[] for _ in range(num_ue_grps)]

        tb_pars = test_vector_file["tb_pars"]
        for ue_idx in range(num_ues):
            scid = tb_pars["nSCID"][ue_idx]
            num_layers = tb_pars["numLayers"][ue_idx]
            dmrs_ports = tb_pars["dmrsPortBmsk"][ue_idx]
            rnti = tb_pars["nRnti"][ue_idx]
            data_scid = tb_pars["dataScramId"][ue_idx]
            mcs_table = tb_pars["mcsTableIndex"][ue_idx]
            mcs_index = tb_pars["mcsIndex"][ue_idx]
            code_rate = tb_pars["targetCodeRate"][ue_idx]
            mod_order = tb_pars["qamModOrder"][ue_idx]
            tb_size = tb_pars["nTbByte"][ue_idx]
            rv = tb_pars["rv"][ue_idx]
            ndi = tb_pars["ndi"][ue_idx]
            ue_grp_idx = tb_pars["userGroupIndex"][ue_idx]

            pusch_ue_config = PuschUeConfig(
                scid=scid,
                layers=num_layers,
                dmrs_ports=dmrs_ports,
                rnti=rnti,
                data_scid=data_scid,
                mcs_table=mcs_table,
                mcs_index=mcs_index,
                code_rate=code_rate,
                mod_order=mod_order,
                tb_size=tb_size,
                rv=rv,
                ndi=ndi,
                harq_process_id=0
            )
            pusch_ue_configs[ue_grp_idx].append(pusch_ue_config)

        for ue_grp_idx in range(num_ue_grps):
            first_ue_idx = test_vector_file["ueGrp_pars"]["UePrmIdxs"][ue_grp_idx]
            if isinstance(first_ue_idx, np.ndarray):
                first_ue_idx = first_ue_idx[0]

            num_dmrs_cdm_grps_no_data = tb_pars["numDmrsCdmGrpsNoData"][first_ue_idx]
            dmrs_scrm_id = tb_pars["dmrsScramId"][first_ue_idx]
            start_prb = test_vector_file["ueGrp_pars"]["startPrb"][ue_grp_idx]
            num_prbs = test_vector_file["ueGrp_pars"]["nPrb"][ue_grp_idx]
            prg_size = test_vector_file["ueGrp_pars"]["prgSize"][ue_grp_idx]
            num_ul_streams = int(test_vector_file["ueGrp_pars"]["nUplinkStreams"][ue_grp_idx])
            dmrs_sym_loc_bmsk = test_vector_file["ueGrp_pars"]["dmrsSymLocBmsk"][ue_grp_idx]
            dmrs_syms = dmrs_fapi_to_bit_array(dmrs_sym_loc_bmsk)
            dmrs_max_len = tb_pars["dmrsMaxLength"][first_ue_idx]
            dmrs_add_ln_pos = tb_pars["dmrsAddlPosition"][first_ue_idx]
            start_sym = test_vector_file["ueGrp_pars"]["StartSymbolIndex"][ue_grp_idx]
            num_symbols = test_vector_file["ueGrp_pars"]["NrOfSymbols"][ue_grp_idx]

            pusch_config = PuschConfig(
                num_dmrs_cdm_grps_no_data=num_dmrs_cdm_grps_no_data,
                dmrs_scrm_id=dmrs_scrm_id,
                start_prb=start_prb,
                num_prbs=num_prbs,
                prg_size=prg_size,
                num_ul_streams=num_ul_streams,
                dmrs_syms=dmrs_syms,
                dmrs_max_len=dmrs_max_len,
                dmrs_add_ln_pos=dmrs_add_ln_pos,
                start_sym=start_sym,
                num_symbols=num_symbols,
                ue_configs=pusch_ue_configs[ue_grp_idx]
            )

            pusch_configs.append(pusch_config)

        return pusch_configs

    return _pusch_config


@pytest.fixture(name="csi_rs_config", scope="module")
def fixture_csi_rs_config():
    """Fixture that returns a function for getting CsiRsConfig out of a test vector file."""
    def _csi_rs_config(test_vector_file):
        csi_rs_prms_list = test_vector_file["csirs_pars"]
        csi_rs_config = []
        for csi_rs_prms in csi_rs_prms_list:
            csi_rs_config.append(CsiRsConfig(
                start_prb=csi_rs_prms["StartRB"],
                num_prb=csi_rs_prms["NrOfRBs"],
                freq_alloc=list(map(int, format(csi_rs_prms["FreqDomain"], "016b"))),
                row=csi_rs_prms["Row"],
                symb_L0=csi_rs_prms["SymbL0"],
                symb_L1=csi_rs_prms["SymbL1"],
                freq_density=csi_rs_prms["FreqDensity"],
                # Just empty REs, no signal in this case.
                scramb_id=0,
                idx_slot_in_frame=0,
                cdm_type=1,
                beta=1.0,
                enable_precoding=False,
                precoding_matrix_index=0
            ))
        return csi_rs_config
    return _csi_rs_config


@pytest.fixture(name="pdsch_config", scope="module")
def fixture_pdsch_config():
    """Fixture that returns a function for getting PdschConfig out of a test vector file."""
    def _pdsch_config(test_vector_file):

        num_ue_grps = len(test_vector_file["ueGrp_pars"])
        num_ues = len(test_vector_file["ue_pars"])
        num_cws = len(test_vector_file["cw_pars"])

        pdsch_configs = []
        pdsch_ue_configs = [[] for _ in range(num_ue_grps)]
        pdsch_cw_configs = [[] for _ in range(num_ues)]

        # CW parameters.
        ue_indices = test_vector_file["cw_pars"]["ueIdx"]
        mcs_tables = test_vector_file["cw_pars"]["mcsTableIndex"]
        mcs_indices = test_vector_file["cw_pars"]["mcsIndex"]
        code_rates = test_vector_file["cw_pars"]["targetCodeRate"]
        mod_orders = test_vector_file["cw_pars"]["qamModOrder"]
        rvs = test_vector_file["cw_pars"]["rv"]
        num_prb_lbrms = test_vector_file["cw_pars"]["n_PRB_LBRM"]
        max_layers = test_vector_file["cw_pars"]["maxLayers"]
        max_qms = test_vector_file["cw_pars"]["maxQm"]
        for cw_idx in range(num_cws):
            pdsch_cw_config = PdschCwConfig(
                mcs_table=mcs_tables[cw_idx],
                mcs_index=mcs_indices[cw_idx],
                code_rate=code_rates[cw_idx],
                mod_order=mod_orders[cw_idx],
                rv=rvs[cw_idx],
                num_prb_lbrm=num_prb_lbrms[cw_idx],
                max_layers=max_layers[cw_idx],
                max_qm=max_qms[cw_idx]
            )
            pdsch_cw_configs[ue_indices[cw_idx]].append(pdsch_cw_config)

        # UE parameters.
        ue_grp_indices = test_vector_file["ue_pars"]["ueGrpIdx"]
        scids = test_vector_file["ue_pars"]["scid"]
        dmrs_scrm_ids = test_vector_file["ue_pars"]["dmrsScramId"]
        layers = test_vector_file["ue_pars"]["nUeLayers"]
        dmrs_ports = test_vector_file["ue_pars"]["dmrsPortBmsk"]
        bwp_starts = test_vector_file["ue_pars"]["BWPStart"]
        ref_points = test_vector_file["ue_pars"]["refPoint"]
        beta_qams = test_vector_file["ue_pars"]["beta_qam"]
        beta_dmrs = test_vector_file["ue_pars"]["beta_dmrs"]
        rntis = test_vector_file["ue_pars"]["rnti"]
        data_scids = test_vector_file["ue_pars"]["dataScramId"]
        if np.array(test_vector_file["tb0_PM_W"]).size > 0:
            precoding_matrices = [
                np.array(test_vector_file["tb" + str(ue_idx) + "_PM_W"]["re"]) +
                1j * np.array(test_vector_file["tb" + str(ue_idx) + "_PM_W"]["im"])
                for ue_idx in range(num_ues)
            ]
        else:
            precoding_matrices = [None for _ in range(num_ues)]

        for ue_idx in range(num_ues):

            pdsch_ue_config = PdschUeConfig(
                cw_configs=pdsch_cw_configs[ue_idx],
                scid=scids[ue_idx],
                dmrs_scrm_id=dmrs_scrm_ids[ue_idx],
                layers=layers[ue_idx],
                dmrs_ports=dmrs_ports[ue_idx],
                bwp_start=bwp_starts[ue_idx],
                ref_point=ref_points[ue_idx],
                beta_qam=beta_qams[ue_idx],
                beta_dmrs=beta_dmrs[ue_idx],
                rnti=rntis[ue_idx],
                data_scid=data_scids[ue_idx],
                precoding_matrix=precoding_matrices[ue_idx]
            )
            pdsch_ue_configs[ue_grp_indices[ue_idx]].append(pdsch_ue_config)

        for ue_grp_idx in range(num_ue_grps):
            num_dmrs_cdm_grps_no_data = \
                test_vector_file["dmrs_pars"]["nDmrsCdmGrpsNoData"][ue_grp_idx]
            resource_alloc = test_vector_file["ueGrp_pars"]["resourceAlloc"][ue_grp_idx]
            prb_bitmap = test_vector_file["ueGrp_pars"]["rbBitmap"][ue_grp_idx]
            start_prb = test_vector_file["ueGrp_pars"]["startPrb"][ue_grp_idx]
            num_prbs = test_vector_file["ueGrp_pars"]["nPrb"][ue_grp_idx]
            dmrs_syms = test_vector_file["ueGrp_pars"]["dmrsSymLocBmsk"][ue_grp_idx]
            dmrs_syms = dmrs_fapi_to_bit_array(dmrs_syms)
            start_sym = test_vector_file["ueGrp_pars"]["pdschStartSym"][ue_grp_idx]
            num_symbols = test_vector_file["ueGrp_pars"]["nPdschSym"][ue_grp_idx]

            pdsch_config = PdschConfig(
                ue_configs=pdsch_ue_configs[ue_grp_idx],
                num_dmrs_cdm_grps_no_data=num_dmrs_cdm_grps_no_data,
                resource_alloc=resource_alloc,
                prb_bitmap=prb_bitmap,
                start_prb=start_prb,
                num_prbs=num_prbs,
                dmrs_syms=dmrs_syms,
                start_sym=start_sym,
                num_symbols=num_symbols
            )

            pdsch_configs.append(pdsch_config)

        return pdsch_configs

    return _pdsch_config

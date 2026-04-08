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

"""pyAerial library - PDSCH transmitter."""
from typing import Any
from typing import List
from typing import Optional

import cuda.bindings.runtime as cudart  # type: ignore
import cupy as cp  # type: ignore
import numpy as np

from aerial import pycuphy  # type: ignore
from aerial.util.cuda import check_cuda_errors
from aerial.util.cuda import CudaStream
from aerial.pycuphy.util import get_pdsch_stat_prms
from aerial.phy5g.api import Array
from aerial.phy5g.api import Pipeline
from aerial.phy5g.api import PipelineFactory
from aerial.phy5g.config import AerialPdschTxConfig
from aerial.phy5g.config import PdschConfig
from aerial.phy5g.config import PdschUeConfig
from aerial.phy5g.config import PdschCwConfig
from aerial.phy5g.config import CsiRsConfig
from aerial.phy5g.config import _pdsch_config_to_cuphy
from aerial.phy5g.pdsch.pdsch_tx_base import PdschTxPipeline
from aerial.phy5g.pdsch.pdsch_tx_base import NUM_RE_PER_PRB
from aerial.phy5g.pdsch.pdsch_tx_base import NUM_PRB_MAX
from aerial.phy5g.pdsch.pdsch_tx_base import NUM_SYMBOLS
from aerial.phy5g.pdsch.pdsch_tx_base import MAX_DL_LAYERS


class PdschTxPipelineFactory(PipelineFactory[AerialPdschTxConfig]):
    """Factory for building a `PdschTx` pipeline."""

    def create(self,
               config: AerialPdschTxConfig,
               cuda_stream: CudaStream,
               **kwargs: Any) -> Pipeline:
        """Create the pipeline.

        Args:
            config (AerialPdschTxConfig): Pipeline configuration object.
            cuda_stream (CudaStream): CUDA stream used to run the pipeline.
                Use ``with stream:`` to scope work; call ``stream.synchronize()``
                explicitly when sync is needed.

        Returns:
            PdschTx: A `PdschTx` pipeline object.
        """
        pdsch_tx = PdschTx(
            cell_id=config.cell_id,
            num_rx_ant=config.num_tx_ant,
            num_tx_ant=config.num_tx_ant,
            num_ul_bwp=config.num_dl_bwp,
            num_dl_bwp=config.num_dl_bwp,
            mu=config.mu,
            cuda_stream=cuda_stream
        )
        return pdsch_tx


class PdschTx(PdschTxPipeline[PdschConfig, Array]):
    """PDSCH transmitter.

    This class implements the whole PDSCH transmission pipeline from the transmitted
    transport block to the transmitted frequency-domain symbols.
    """
    def __init__(
        self,
        *,
        cell_id: int,
        num_rx_ant: int,
        num_tx_ant: int,
        num_ul_bwp: int = NUM_PRB_MAX,
        num_dl_bwp: int = NUM_PRB_MAX,
        mu: int = 1,
        cuda_stream: Optional[CudaStream] = None
    ) -> None:
        """Initialize PdschTx.

        Args:
            cell_id (int): Physical cell ID.
            num_rx_ant (int): Number of receive antennas.
            num_tx_ant (int): Number of transmit antennas.
            num_ul_bwp (int): Number of PRBs in a uplink bandwidth part.
                Default: 273.
            num_dl_bwp (int): Number of PRBs in a downlink bandwidth part.
                Default: 273.
            mu (int): Numerology. Values in [0, 3]. Default: 1.
            cuda_stream (Optional[CudaStream]): CUDA stream. If not given, a new
                CudaStream is created. Use ``with stream:`` to scope work; call
                ``stream.synchronize()`` explicitly when sync is needed.
        """
        self._cuda_stream = CudaStream() if cuda_stream is None else cuda_stream

        pdsch_tx_stat_prms = get_pdsch_stat_prms(
            cell_id=cell_id,
            num_rx_ant=num_rx_ant,
            num_tx_ant=num_tx_ant,
            num_prb_ul_bwp=num_ul_bwp,
            num_prb_dl_bwp=num_dl_bwp,
            mu=mu,
        )
        self.pdsch_pipeline = pycuphy.PdschPipeline(pdsch_tx_stat_prms)
        self.output_dims = [num_dl_bwp * NUM_RE_PER_PRB, NUM_SYMBOLS, MAX_DL_LAYERS]
        # Use int() to prevent overflow with numpy int16/int32 types
        self.num_bytes = int(num_dl_bwp) * NUM_RE_PER_PRB * NUM_SYMBOLS * MAX_DL_LAYERS * 4

        # Complex half output.
        self.tx_output_mem = check_cuda_errors(cudart.cudaMalloc(self.num_bytes))

        # Complex float output.
        with self._cuda_stream:
            self.tx_output = cp.ndarray(
                shape=(num_dl_bwp * NUM_RE_PER_PRB, NUM_SYMBOLS, MAX_DL_LAYERS),
                dtype=cp.complex64,
                order='F'
            )

        self.num_ues = 0  # Make pylint happy.

    def __call__(self,
                 slot: int,
                 tb_inputs: List[Array],
                 config: List[PdschConfig],
                 csi_rs_config: List[CsiRsConfig] = None,
                 **kwargs: Any) -> Array:
        """Run the transmitter pipeline.

        Note: This implements the base class abstract method.

        Args:
            slot (int): Slot number.
            tb_inputs (List[Array]): Transport blocks to be transmitted, one per UE.
            config (List[PdschConfig]): Dynamic slot configuration in this slot.
            csi_rs_config (List[CsiRsConfig]): Optional parameters for CSI-RS. Note: This only
                leaves the CSI-RS REs empty. To actually add in the CSI-RS signals, one
                needs to call the CSI-RS transmitter separately.

        Returns:
            Array: Transmitted OFDM symbols in a frequency x time x antenna tensor.
        """
        return self.run(
            tb_inputs=tb_inputs,
            slot=slot,
            pdsch_configs=config,
            csi_rs_configs=csi_rs_config
        )

    def run(  # pylint: disable=too-many-arguments, too-many-locals
            self,
            *,
            tb_inputs: List[Array],
            slot: int = 0,
            pdsch_configs: List[PdschConfig] = None,

            # UE group parameters.
            num_ues: int = 1,
            num_dmrs_cdm_grps_no_data: int = 2,
            resource_alloc: int = 1,
            prb_bitmap: List[int] = None,
            start_prb: int = 0,
            num_prbs: int = 273,
            dmrs_syms: List[int] = None,
            start_sym: int = 2,
            num_symbols: int = 12,

            # UE parameters.
            scids: List[int] = None,
            dmrs_scrm_ids: List[int] = None,
            layers: List[int] = None,
            dmrs_ports: List[int] = None,
            bwp_starts: List[int] = None,
            ref_points: List[int] = None,
            rntis: List[int] = None,
            data_scids: List[int] = None,
            precoding_matrices: List[np.ndarray] = None,

            # CW parameters.
            mcs_tables: List[int] = None,
            mcs_indices: List[int] = None,
            code_rates: List[int] = None,
            mod_orders: List[int] = None,
            rvs: List[int] = None,
            num_prb_lbrms: List[int] = None,
            max_layers: List[int] = None,
            max_qms: List[int] = None,

            # CSI-RS parameters.
            csi_rs_configs: List[CsiRsConfig] = None) -> Array:
        """Run PDSCH transmission.

        Set dynamic PDSCH parameters and call cuPHY to run the PDSCH transmission.

        If the input transport blocks are on the GPU, also the output will be on the GPU.
        If they are on the host (NumPy arrays), also the output will be on the host.

        Args:
            tb_inputs (List[np.ndarray]): Transport blocks in bytes for each UE.
            num_ues (int): Number of UEs.
            slot (int): Slot number.
            num_dmrs_cdm_grps_no_data (int): Number of DMRS CDM groups without data
                [3GPP TS 38.212, sec 7.3.1.1]. Value: 1->3.
            resource_alloc (int): Resource allocation type.
            prb_bitmap (List[int]): Array of bits indicating bitmask for allocated RBs.
            start_prb (int): Start PRB index for the UE group.
            num_prbs (int): Number of allocated PRBs for the UE group.
            dmrs_syms (List[int]): For the UE group, a list of binary numbers each indicating
                whether the corresponding symbol is a DMRS symbol.
            start_sym (int): Start OFDM symbol index of the UE group allocation.
            num_symbols (int): Number of symbols in the allocation, starting from
                `start_sym`.
            scids (List[int]): DMRS sequence initialization for each UE
                [TS38.211, sec 7.4.1.1.2].
            dmrs_scrm_ids (List[int]): Downlink DMRS scrambling ID for each UE.
            layers (List[int]): Number of layers for each UE.
            dmrs_ports (List[int]): DMRS ports for each UE. The format of each entry is in the
                SCF FAPI format as follows: A bitmap (mask) starting from the LSB where each bit
                indicates whether the corresponding DMRS port index is used.
            bwp_starts (List[int]): Bandwidth part start (PRB number starting from 0).
                Used only if reference point is 1.
            ref_points (List[int]): DMRS reference point per UE. Value 0 or 1.
            rntis (List[int]) RNTI for each UE.
            data_scids (List[int]): Data scrambling IDs for each UE, more precisely
                `dataScramblingIdentityPdsch` [TS38.211, sec 7.3.1.1].
            precoding_matrices (List[np.ndarray]): Precoding matrices, one per UE.
                The shape of each precoding matrix is number of layers x number of Tx antennas.
                If set to None, precoding is disabled.
            mcs_tables (List[int]): MCS table per UE.
            mcs_indices (List[int]): MCS index per UE.
            code_rates (List[int]): Code rate, expressed as the number of information
                bits per 1024 coded bits expressed in 0.1 bit units.
            mod_orders (List[int]): Modulation order for each UE.
            rvs (List[int]): Redundancy version per UE (default: 0 for each UE).
            num_prb_lbrms (List[int]): Number of PRBs used for LBRM TB size computation.
                Possible values: {32, 66, 107, 135, 162, 217, 273}.
            max_layers (List[int]): Number of layers used for LBRM TB size computation (at most 4).
            max_qms (List[int]): Modulation order used for LBRM TB size computation. Value: 6 or 8.
            csi_rs_configs (List[CsiRsConfig]): List of CSI-RS RRC dynamic parameters, see
                `CsiRsConfig`. Note that no CSI-RS symbols get written, this is only to make
                sure that PDSCH does not get mapped to the CSI-RS resource elements.

        Returns:
            Array: Transmitted OFDM symbols in a frequency x time x antenna tensor.
        """
        if precoding_matrices is not None:
            precoding_matrices = [np.ascontiguousarray(m) for m in precoding_matrices]

        cpu_copy = isinstance(tb_inputs[0], np.ndarray)

        with self._cuda_stream:
            tb_inputs = [cp.array(tb, dtype=cp.uint8, order='F') for tb in tb_inputs]

        # Reset the output buffer.
        check_cuda_errors(
            cudart.cudaMemsetAsync(
                self.tx_output_mem, 0., self.num_bytes, self._cuda_stream.handle
            )
        )

        if pdsch_configs is None:
            pdsch_configs = _single_ue_grp_pdsch_config(
                num_ues=num_ues,

                # UE group parameters.
                num_dmrs_cdm_grps_no_data=num_dmrs_cdm_grps_no_data,
                resource_alloc=resource_alloc,
                prb_bitmap=prb_bitmap,
                start_prb=start_prb,
                num_prbs=num_prbs,
                dmrs_syms=dmrs_syms,
                start_sym=start_sym,
                num_symbols=num_symbols,

                # UE parameters.
                scids=scids,
                dmrs_scrm_ids=dmrs_scrm_ids,
                layers=layers,
                dmrs_ports=dmrs_ports,
                bwp_starts=bwp_starts,
                ref_points=ref_points,
                rntis=rntis,
                data_scids=data_scids,
                precoding_matrices=precoding_matrices,

                # CW parameters.
                mcs_tables=mcs_tables,
                mcs_indices=mcs_indices,
                target_code_rates=code_rates,
                mod_orders=mod_orders,
                rvs=rvs,
                num_prb_lbrms=num_prb_lbrms,
                max_layers=max_layers,
                max_qms=max_qms
            )
        else:
            num_ues = sum(len(pdsch_config.ue_configs) for pdsch_config in pdsch_configs)
            dmrs_ports = [ue_config.dmrs_ports for pdsch_config in pdsch_configs
                          for ue_config in pdsch_config.ue_configs]
            scids = [ue_config.scid for pdsch_config in pdsch_configs
                     for ue_config in pdsch_config.ue_configs]
            precoding_matrices = [ue_config.precoding_matrix for pdsch_config in pdsch_configs
                                  for ue_config in pdsch_config.ue_configs]
        # Create the dynamic params structure. Default parameters inserted for those
        # that are not given.
        pdsch_tx_dyn_prms = _pdsch_config_to_cuphy(
            cuda_stream=self._cuda_stream,
            tb_inputs=tb_inputs,
            tx_output_mem=self.tx_output_mem,
            slot=slot,
            pdsch_configs=pdsch_configs,
            csi_rs_configs=csi_rs_configs
        )

        self.pdsch_pipeline.setup_pdsch_tx(pdsch_tx_dyn_prms)
        self.pdsch_pipeline.run_pdsch_tx()

        pycuphy.convert_to_complex64(
            self.tx_output_mem,
            self.tx_output.data.ptr,
            self.output_dims,
            self._cuda_stream.handle
        )

        tx_slot = PdschTx.cuphy_to_tx(
            tx_slot=self.tx_output,
            num_ues=num_ues,
            dmrs_ports=dmrs_ports,
            scids=scids,
            precoding_matrices=precoding_matrices
        )

        if cpu_copy:
            with self._cuda_stream:
                tx_slot = tx_slot.get(order='F')

        self.num_ues = num_ues

        return tx_slot

    def ldpc_output(self) -> List[np.ndarray]:
        """Return the coded bits from LDPC encoder output.

        Note: This is returned always as a NumPy array, i.e. in host memory.

        Returns:
            List[np.array]: Coded bits in a num_codewords x num_bits_per_codeword tensor,
                one per UE.
        """
        ldpc_output = []
        # Get the coded bits i.e. the LDPC output.
        for ue_idx in range(self.num_ues):
            ldpc_output.append(self.pdsch_pipeline.get_ldpc_output(0, ue_idx))

        return ldpc_output

    def __del__(self) -> None:
        """Destructor."""
        if hasattr(self, 'tx_output_mem'):
            check_cuda_errors(cudart.cudaFree(self.tx_output_mem))

    @classmethod
    def cuphy_to_tx(
            cls,
            *,
            tx_slot: Array,
            num_ues: int,
            dmrs_ports: List[int],
            scids: List[int],
            precoding_matrices: List[np.ndarray] = None) -> Array:
        """Map cuPHY outputs to Tx antenna ports.

        Args:
            tx_slot (Array): Transmit buffer from cuPHY.
            num_ues (int): Number of UEs.
            dmrs_ports (List[int]): DMRS ports for each UE. The format of each entry is in the
                SCF FAPI format as follows: A bitmap (mask) starting from the LSB where each bit
                indicates whether the corresponding DMRS port index is used.
            scids (List[int]): DMRS sequence initialization for each UE [TS38.211, sec 7.4.1.1.2].
            precoding_matrices (List[np.ndarray]): Precoding matrices, one per UE.
                The shape of each precoding matrix is number of layers x number of Tx antennas.
                If set to None, precoding is disabled.

        Returns:
            Array: Transmitted OFDM symbols in a frequency x time x antenna tensor.
        """
        indices = []
        for ue in range(num_ues):
            if precoding_matrices is None or precoding_matrices[ue] is None or \
                    precoding_matrices[ue].size == 0:
                dmrs_port_indices = np.where(np.flipud(np.unpackbits(np.uint8(dmrs_ports[ue]))))[0]
                indices += list(dmrs_port_indices + 8 * scids[ue])
            else:
                indices += list(range(precoding_matrices[ue].shape[1]))
        indices = list(set(indices))

        return tx_slot[:, :, indices]


def _single_ue_grp_pdsch_config(  # noqa: C901 pylint: disable=too-many-arguments, too-many-locals
        *,
        num_ues: int,

        # UE group parameters.
        num_dmrs_cdm_grps_no_data: int = 2,
        resource_alloc: int = 1,
        prb_bitmap: List[int] = None,
        start_prb: int = 0,
        num_prbs: int = 273,
        dmrs_syms: List[int] = None,
        start_sym: int = 2,
        num_symbols: int = 12,

        # UE parameters.
        scids: List[int] = None,
        dmrs_scrm_ids: List[int] = None,
        layers: List[int] = None,
        dmrs_ports: List[int] = None,
        bwp_starts: List[int] = None,
        ref_points: List[int] = None,
        rntis: List[int] = None,
        data_scids: List[int] = None,
        precoding_matrices: List[np.ndarray] = None,

        # CW parameters.
        mcs_tables: List[int] = None,
        mcs_indices: List[int] = None,
        target_code_rates: List[int] = None,
        mod_orders: List[int] = None,
        rvs: List[int] = None,
        num_prb_lbrms: List[int] = None,
        max_layers: List[int] = None,
        max_qms: List[int] = None) -> List[PdschConfig]:
    """Helper to convert given parameters to `PdschConfig`.

    Args:
        num_dmrs_cdm_grps_no_data (int): Number of DMRS CDM groups without data
            [3GPP TS 38.212, sec 7.3.1.1]. Value: 1->3.
        resource_alloc (int): Resource allocation type.
        prb_bitmap (List[int]): Array of bytes indicating bitmask for allocated RBs.
        start_prb (int): Start PRB index for the UE group.
        num_prbs (int): Number of allocated PRBs for the UE group.
        dmrs_syms (List[int]): For the UE group, a list of binary numbers each indicating
            whether the corresponding symbol is a DMRS symbol.
        start_sym (int): Start OFDM symbol index of the UE group allocation.
        num_symbols (int): Number of symbols in the allocation, starting from
            `start_sym`.
        scids (List[int]): DMRS sequence initialization for each UE
            [TS38.211, sec 7.4.1.1.2].
        dmrs_scrm_ids (List[int]): Downlink DMRS scrambling ID for each UE.
        layers (List[int]): Number of layers for each UE.
        dmrs_ports (List[int]): DMRS ports for each UE. The format of each entry is in the
            SCF FAPI format as follows: A bitmap (mask) starting from the LSB where each bit
            indicates whether the corresponding DMRS port index is used.
        bwp_starts (List[int]): Bandwidth part start (PRB number starting from 0).
            Used only if reference point is 1.
        ref_points (List[int]): DMRS reference point per UE. Value 0 or 1.
        rntis (List[int]) RNTI for each UE.
        data_scids (List[int]): Data scrambling IDs for each UE, more precisely
            `dataScramblingIdentityPdsch` [TS38.211, sec 7.3.1.1].
        precoding_matrices (List[np.ndarray]): Precoding matrices, one per UE.
            The shape of each precoding matrix is number of layers x number of Tx antennas.
            If set to None, precoding is disabled.
        mcs_tables (List[int]): MCS table per UE.
        mcs_indices (List[int]): MCS index per UE.
        target_code_rates (List[int]): Code rate for each UE in SCF FAPI format,
            i.e. code rate x 1024 x 10.
        mod_orders (List[int]): Modulation order for each UE.
        rvs (List[int]): Redundancy version per UE (default: 0 for each UE).
        num_prb_lbrms (List[int]): Number of PRBs used for LBRM TB size computation.
            Possible values: {32, 66, 107, 135, 162, 217, 273}.
        max_layers (List[int]): Number of layers used for LBRM TB size computation (at most 4).
        max_qms (List[int]): Modulation order used for LBRM TB size computation. Value: 6 or 8.
    """
    # Set the default values.
    if prb_bitmap is None:
        prb_bitmap = 36 * [0, ]
    if dmrs_syms is None:
        dmrs_syms = [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0]

    if scids is None:
        valid_scids = [0, 0, 1, 1, 0, 0, 1, 1]
        scids = valid_scids[:num_ues]
    if dmrs_scrm_ids is None:
        valid_dmrs_scrm_ids = [41, 41, 41, 41, 41, 41, 41, 41]
        dmrs_scrm_ids = valid_dmrs_scrm_ids[:num_ues]
    if layers is None:
        layers = list(num_ues * [1, ])
    if dmrs_ports is None:
        valid_dmrs_ports = [1, 4, 2, 8, 16, 64, 32, 128]
        dmrs_ports = valid_dmrs_ports[:num_ues]
    if bwp_starts is None:
        bwp_starts = list(num_ues * [0, ])
    if ref_points is None:
        ref_points = list(num_ues * [0, ])
    if rntis is None:
        rntis = list(np.arange(1, num_ues + 1))
    if data_scids is None:
        data_scids = list(np.arange(1, num_ues + 1))

    if mcs_tables is None:
        mcs_tables = list(num_ues * [0, ])
    if mcs_indices is None:
        mcs_indices = list(num_ues * [0, ])
    if target_code_rates is None:
        target_code_rates = list(num_ues * [1930, ])
    if mod_orders is None:
        mod_orders = list(num_ues * [2, ])
    if rvs is None:
        rvs = list(num_ues * [0, ])
    if num_prb_lbrms is None:
        num_prb_lbrms = list(num_ues * [273, ])
    if max_layers is None:
        max_layers = list(num_ues * [4, ])
    if max_qms is None:
        max_qms = list(num_ues * [8, ])

    pdsch_ue_configs = []

    for ue_idx in range(num_ues):

        pdsch_cw_config = PdschCwConfig(
            mcs_table=mcs_tables[ue_idx],
            mcs_index=mcs_indices[ue_idx],
            code_rate=target_code_rates[ue_idx],
            mod_order=mod_orders[ue_idx],
            rv=rvs[ue_idx],
            num_prb_lbrm=num_prb_lbrms[ue_idx],
            max_layers=max_layers[ue_idx],
            max_qm=max_qms[ue_idx]
        )

        precoding_matrix = None
        if precoding_matrices is not None:
            precoding_matrix = precoding_matrices[ue_idx]

        pdsch_ue_config = PdschUeConfig(
            cw_configs=[pdsch_cw_config],
            scid=scids[ue_idx],
            dmrs_scrm_id=dmrs_scrm_ids[ue_idx],
            layers=layers[ue_idx],
            dmrs_ports=dmrs_ports[ue_idx],
            bwp_start=bwp_starts[ue_idx],
            ref_point=ref_points[ue_idx],
            rnti=rntis[ue_idx],
            data_scid=data_scids[ue_idx],
            precoding_matrix=precoding_matrix
        )

        pdsch_ue_configs.append(pdsch_ue_config)

    pdsch_config = PdschConfig(
        ue_configs=pdsch_ue_configs,
        num_dmrs_cdm_grps_no_data=num_dmrs_cdm_grps_no_data,
        resource_alloc=resource_alloc,
        prb_bitmap=prb_bitmap,
        start_prb=start_prb,
        num_prbs=num_prbs,
        dmrs_syms=dmrs_syms,
        start_sym=start_sym,
        num_symbols=num_symbols
    )

    return [pdsch_config]

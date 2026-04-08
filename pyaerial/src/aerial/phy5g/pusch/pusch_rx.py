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

"""pyAerial library - PUSCH receiver."""
from typing import Any
from typing import List
from typing import Optional
from typing import Tuple

import cuda.bindings.runtime as cudart  # type: ignore
import cupy as cp  # type: ignore
import numpy as np

from aerial import pycuphy  # type: ignore
from aerial.util.cuda import check_cuda_errors
from aerial.util.cuda import CudaStream
from aerial.phy5g.api import Array
from aerial.phy5g.api import Pipeline
from aerial.phy5g.api import PipelineFactory
from aerial.phy5g.config import AerialPuschRxConfig
from aerial.phy5g.config import PuschConfig
from aerial.phy5g.config import PuschUeConfig
from aerial.phy5g.config import _pusch_config_to_cuphy
from aerial.phy5g.pusch.pusch_rx_base import PuschRxPipeline
from aerial.pycuphy.util import get_pusch_stat_prms
from aerial.pycuphy.util import get_pusch_dyn_prms_phase_2
from aerial.pycuphy.types import PuschEqCoefAlgoType
from aerial.pycuphy.types import PuschLdpcKernelLaunch

# Constant definitions.
NUM_PRB_MAX = 273


class PuschRxPipelineFactory(PipelineFactory[AerialPuschRxConfig]):
    """Factory for building a `PuschRx` pipeline."""

    def create(self,
               config: AerialPuschRxConfig,
               cuda_stream: CudaStream,
               **kwargs: Any) -> Pipeline:
        """Create the pipeline.

        Args:
            config (AerialPuschRxConfig): Pipeline configuration object.
            cuda_stream (CudaStream): CUDA stream used to run the pipeline.
                Use ``with stream:`` to scope work; call ``stream.synchronize()``
                explicitly when sync is needed.

        Returns:
            PuschRx: A `PuschRx` pipeline object.
        """
        pusch_rx = PuschRx(
            cell_id=config.cell_id,
            num_tx_ant=config.num_rx_ant,
            num_rx_ant=config.num_rx_ant,
            enable_pusch_tdi=config.enable_pusch_tdi,
            eq_coeff_algo=config.eq_coeff_algo,
            ldpc_kernel_launch=config.ldpc_kernel_launch,
            chest_factory_settings_filename=config.chest_factory_settings_filename,
            cuda_stream=cuda_stream
        )
        return pusch_rx


class PuschRx(PuschRxPipeline[PuschConfig, Array]):
    """PUSCH receiver pipeline.

    This class implements the whole PUSCH reception pipeline from the received OFDM
    post-FFT symbols to the received transport block (along with CRC check).
    """

    def __init__(  # pylint: disable=too-many-arguments
        self,
        *,
        cell_id: int,
        num_rx_ant: int,
        num_tx_ant: int,
        num_ul_bwp: int = NUM_PRB_MAX,
        num_dl_bwp: int = NUM_PRB_MAX,
        mu: int = 1,
        enable_cfo_correction: int = 0,
        enable_weighted_ave_cfo_est: int = 0,
        enable_to_estimation: int = 0,
        enable_pusch_tdi: int = 0,
        eq_coeff_algo: int = 1,
        enable_per_prg_chest: int = 0,
        enable_ul_rx_bf: int = 0,
        ldpc_kernel_launch: PuschLdpcKernelLaunch = PuschLdpcKernelLaunch.PUSCH_RX_ENABLE_DRIVER_LDPC_LAUNCH,  # noqa: E501 # pylint: disable=line-too-long
        chest_factory_settings_filename: Optional[str] = None,
        cuda_stream: Optional[CudaStream] = None
    ) -> None:
        """Initialize PuschRx.

        Args:
            cell_id (int): Physical cell ID.
            num_rx_ant (int): Number of receive antennas.
            num_tx_ant (int): Number of transmit antennas.
            num_ul_bwp (int): Number of PRBs in a uplink bandwidth part.
                Default: 273.
            num_dl_bwp (int): Number of PRBs in a downlink bandwidth part.
                Default: 273.
            mu (int): Numerology. Values in [0, 3]. Default: 1.
            enable_cfo_correction (int): Enable/disable CFO correction:

                - 0: Disable (default).
                - 1: Enable.

            enable_weighted_ave_cfo_est (int): Enable/disable CFO weighted average estimation:

                - 0: Disable (default).
                - 1: Enable.

            enable_to_estimation (int): Enable/disable time offset estimation:

                - 0: Disable (default).
                - 1: Enable.

            enable_pusch_tdi (int): Time domain interpolation on PUSCH.

                - 0: Disable (default).
                - 1: Enable.

            eq_coeff_algo (int): Algorithm for equalizer coefficient computation.

                - 0 - ZF.
                - 1 - MMSE (default).
                - 2 - MMSE-IRC.
                - 3 - MMSE-IRC with RBLW-based covariance shrinkage.
                - 4 - MMSE-IRC with OAS-based covariance shrinkage.

            enable_per_prg_chest (int): Enable/disable PUSCH per-PRG channel estimation.

                - 0: Disable (default).
                - 1: Enable.

            enable_ul_rx_bf (int): Enable/disable beamforming for PUSCH.

                - 0: Disable (default).
                - 1: Enable.

            ldpc_kernel_launch (PuschLdpcKernelLaunch): LDPC kernel launch method.
            chest_factory_settings_filename (str): The chest factory settings filename.
            cuda_stream (Optional[CudaStream]): CUDA stream. If not given, a new CudaStream is
                created. Use ``with stream:`` to scope work; call ``stream.synchronize()``
                explicitly when sync is needed.
        """
        self._cuda_stream = CudaStream() if cuda_stream is None else cuda_stream

        self.num_tx_ant = num_tx_ant
        self.num_rx_ant = num_rx_ant

        # TODO: Enable arbitrary user-defined stat_prms, e.g. by exposing
        # more parameters through this function interface.
        self.pusch_rx_stat_prms = get_pusch_stat_prms(
            cell_id=cell_id,
            num_rx_ant=num_rx_ant,
            num_tx_ant=num_tx_ant,
            num_prb_ul_bwp=num_ul_bwp,
            num_prb_dl_bwp=num_dl_bwp,
            mu=mu,
            enable_cfo_correction=enable_cfo_correction,
            enable_weighted_ave_cfo_est=enable_weighted_ave_cfo_est,
            enable_to_estimation=enable_to_estimation,
            enable_pusch_tdi=enable_pusch_tdi,
            enable_per_prg_chest=enable_per_prg_chest,
            enable_ul_rx_bf=enable_ul_rx_bf,
            eq_coeff_algo=PuschEqCoefAlgoType(eq_coeff_algo),
            chest_factory_settings_filename=chest_factory_settings_filename,
            ldpc_kernel_launch=ldpc_kernel_launch
        )
        self.pusch_pipeline = pycuphy.PuschPipeline(self.pusch_rx_stat_prms, self._cuda_stream.handle)

    def __call__(self,
                 slot: int,
                 rx_slot: Array,
                 config: List[PuschConfig],
                 **kwargs: Any) -> Tuple[Array, List[Array]]:
        """Run the receiver pipeline.

        Note: This implements the base class abstract method.

        Args:
            slot (int): Slot number.
            rx_slot (Array): Received slot as an Array.
            config (List[PuschConfig]): Dynamic slot configuration in this slot.

        Returns:
            Array, List[Array]: A tuple containing:

            - *Array*: Transport block CRCs.

            - *List[Array]*: Transport blocks, one per UE, without CRC.
        """
        return self.run(rx_slot=rx_slot, slot=slot, pusch_configs=config)

    def run(  # pylint: disable=too-many-arguments,too-many-locals
            self,
            *,
            rx_slot: Array,
            slot: int = 0,
            pusch_configs: List[PuschConfig] = None,

            # UE group parameters.
            num_ues: int = 1,
            num_dmrs_cdm_grps_no_data: int = 2,
            dmrs_scrm_id: int = 41,
            start_prb: int = 0,
            num_prbs: int = 273,
            prg_size: int = 1,
            num_ul_streams: int = 1,
            dmrs_syms: List[int] = None,
            dmrs_max_len: int = 2,
            dmrs_add_ln_pos: int = 1,
            start_sym: int = 2,
            num_symbols: int = 12,

            # UE parameters.
            scids: List[int] = None,
            layers: List[int] = None,
            dmrs_ports: List[int] = None,
            rntis: List[int] = None,
            data_scids: List[int] = None,

            # CW parameters.
            mcs_tables: List[int] = None,
            mcs_indices: List[int] = None,
            code_rates: List[int] = None,
            mod_orders: List[int] = None,
            tb_sizes: List[int] = None,
            rvs: List[int] = None,
            ndis: List[int] = None) -> Tuple[Array, List[Array]]:
        """Run PUSCH Rx.

        This runs the cuPHY PUSCH receiver pipeline based on the given parameters. Multiple
        UE groups are supported if the `PuschConfig` based API is used. Otherwise, the pipeline
        gets run only for a single UE group sharing the same time-frequency resources, i.e.
        having the same PRB allocation, and the same start symbol and number of allocated symbols.
        In this case default values get filled for the parameters that are not given.

        Args:
            rx_slot (Array): A tensor representing the receive slot buffer of the cell.
            slot (int): Slot number.
            pusch_configs (List[PuschConfig]): List of PUSCH configuration objects, one per UE
                group. If this argument is given, the rest are ignored. If not given, the other
                arguments will be used (default values are used for the parameters that are not
                given). Only one UE group is supported in that case.
            num_ues (int): Number of UEs in the UE group.
            num_dmrs_cdm_grps_no_data (int): Number of DMRS CDM groups without data
                [3GPP TS 38.212, sec 7.3.1.1]. Value: 1->3.
            dmrs_scrm_id (int): DMRS scrambling ID.
            start_prb (int): Start PRB index of the UE group allocation.
            num_prbs (int): Number of allocated PRBs for the UE group.
            prg_size (int): The Size of PRG in PRB for PUSCH per-PRG channel estimation.
            nUplinkStreams (int): The number of active streams for this PUSCH.
            dmrs_syms (List[int]): For the UE group, a list of binary numbers each indicating
                whether the corresponding symbol is a DMRS symbol.
            dmrs_max_len (int): The `maxLength` parameter, value 1 or 2, meaning that DMRS are
                single-symbol DMRS or single- or double-symbol DMRS.
            dmrs_add_ln_pos (int): Number of additional DMRS positions.
            start_sym (int): Start OFDM symbol index for the UE group allocation.
            num_symbols (int): Number of symbols in the UE group allocation.
            scids (List[int]): DMRS sequence initialization for each UE
                [TS38.211, sec 7.4.1.1.2].
            layers (List[int]): Number of layers for each UE.
            dmrs_ports (List[int]): DMRS ports for each UE. The format of each entry is in the
                SCF FAPI format as follows: A bitmap (mask) starting from the LSB where each bit
                indicates whether the corresponding DMRS port index is used.
            rntis (List[int]) RNTI for each UE.
            data_scids (List[int]): Data scrambling IDs for each UE, more precisely
                `dataScramblingIdentityPdsch` [TS38.211, sec 7.3.1.1].
            mcs_tables (List[int]): MCS table to use for each UE (see TS 38.214).
            mcs_indices (List[int]): MCS indices for each UE.
            code_rates (List[float]): Code rate, expressed as the number of information
                bits per 1024 coded bits expressed in 0.1 bit units.
            mod_orders (List[int]): Modulation order for each UE.
            tb_sizes (List[int]): TB size in bytes for each UE.
            rvs (List[int]): Redundancy versions for each UE.
            ndis (List[int]): New data indicator per UE.

        Returns:
            Array, List[Array]: A tuple containing:

            - *Array*: Transport block CRCs.

            - *List[Array]*: Transport blocks, one per UE, without CRC.
        """
        with self._cuda_stream:
            rx_slot = cp.array(rx_slot, dtype=cp.complex64, order='F')

        if pusch_configs is None:
            pusch_configs = _single_ue_grp_pusch_config(
                num_ues=num_ues,
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
                scids=scids,
                layers=layers,
                dmrs_ports=dmrs_ports,
                rntis=rntis,
                data_scids=data_scids,
                mcs_tables=mcs_tables,
                mcs_indices=mcs_indices,
                target_code_rates=code_rates,
                mod_orders=mod_orders,
                tb_sizes=tb_sizes,
                rvs=rvs,
                ndis=ndis
            )

        pusch_rx_dyn_params = _pusch_config_to_cuphy(
            cuda_stream=self._cuda_stream,
            rx_data=[rx_slot],
            slot=slot,
            pusch_configs=pusch_configs
        )
        tb_sizes = []
        num_ues = 0
        for pusch_config in pusch_configs:
            tb_sizes += [ue_config.tb_size for ue_config in pusch_config.ue_configs]
            num_ues += len(pusch_config.ue_configs)

        # Run setup phase 1.
        self.pusch_pipeline.setup_pusch_rx(pusch_rx_dyn_params)

        # Run setup phase 2.
        harq_buffers = []
        for ue_idx in range(num_ues):
            harq_buffer_size = pusch_rx_dyn_params.dataOut.harqBufferSizeInBytes[ue_idx]

            # TODO: Move this out of the real-time pipeline.
            harq_buffer = check_cuda_errors(cudart.cudaMalloc(harq_buffer_size))
            check_cuda_errors(
                cudart.cudaMemsetAsync(
                    harq_buffer, 0, harq_buffer_size * 1, self._cuda_stream.handle
                )
            )
            self._cuda_stream.synchronize()
            harq_buffers.append(harq_buffer)

        pusch_rx_dyn_params = get_pusch_dyn_prms_phase_2(
            pusch_rx_dyn_params, harq_buffers
        )
        self.pusch_pipeline.setup_pusch_rx(pusch_rx_dyn_params)

        # Run pipeline.
        self.pusch_pipeline.run_pusch_rx()

        # Fetch outputs.
        # Please note that not all PUSCH features are propagated through to pyaerial and not
        # all PuschDataOut processing results are checked, e.g., pCbCrcs are not.
        tb_crcs = pusch_rx_dyn_params.dataOut.tbCrcs
        tb_payloads = pusch_rx_dyn_params.dataOut.tbPayloads
        tot_num_tb_bytes = pusch_rx_dyn_params.dataOut.totNumPayloadBytes
        start_offsets = list(pusch_rx_dyn_params.dataOut.startOffsetsTbPayload)
        start_offsets.append(tot_num_tb_bytes[0])
        tbs = []
        for ue_idx in range(num_ues):
            # Remove CRC and padding bytes (cuPHY aligns the output to 4-byte boundaries).
            tb = tb_payloads[start_offsets[ue_idx] : start_offsets[ue_idx + 1]]
            tb = tb[:tb_sizes[ue_idx]]
            tbs.append(tb)

        # TODO: Move this out of the real-time pipeline.
        for ue_idx in range(num_ues):
            check_cuda_errors(cudart.cudaFree(harq_buffers[ue_idx]))

        return tb_crcs, tbs


def _single_ue_grp_pusch_config(  # pylint: disable=too-many-arguments
        *,
        num_ues: int,

        # UE group parameters.
        num_dmrs_cdm_grps_no_data: int = 2,
        dmrs_scrm_id: int = 41,
        start_prb: int = 0,
        num_prbs: int = 273,
        prg_size: int = 1,
        num_ul_streams: int = 1,
        dmrs_syms: Optional[List[int]] = None,
        dmrs_max_len: int = 2,
        dmrs_add_ln_pos: int = 1,
        start_sym: int = 2,
        num_symbols: int = 12,

        # UE parameters.
        scids: Optional[List[int]] = None,
        layers: Optional[List[int]] = None,
        dmrs_ports: Optional[List[int]] = None,
        rntis: Optional[List[int]] = None,
        data_scids: Optional[List[int]] = None,

        # CW parameters.
        mcs_tables: Optional[List[int]] = None,
        mcs_indices: Optional[List[int]] = None,
        target_code_rates: Optional[List[int]] = None,
        mod_orders: Optional[List[int]] = None,
        tb_sizes: Optional[List[int]] = None,
        rvs: Optional[List[int]] = None,
        ndis: Optional[List[int]] = None) -> List[PuschConfig]:
    """Helper to convert given parameters to `PuschConfig`.

    Args:
        num_ues (int): Number of UEs.
        num_dmrs_cdm_grps_no_data (int): Number of DMRS CDM groups without data
            [3GPP TS 38.212, sec 7.3.1.1]. Value: 1->3.
        dmrs_scrm_id (int): DMRS scrambling ID.
        start_prb (int): Start PRB index of the UE group allocation.
        num_prbs (int): Number of allocated PRBs for the UE group.
        prg_size (int): Size of PRG in PRB for the UE group.
        num_ul_streams (int): Number of allocated streams for the UE group.
        dmrs_syms (List[int]): For the UE group, a list of binary numbers each indicating whether
            the corresponding symbol is a DMRS symbol.
        dmrs_max_len (int): The `maxLength` parameter, value 1 or 2, meaning that DMRS are
            single-symbol DMRS or single- or double-symbol DMRS.
        dmrs_add_ln_pos (int): Number of additional DMRS positions.
        start_sym (int): Start OFDM symbol index for the UE group allocation.
        num_symbols (int): Number of symbols in the UE group allocation.
        scids (List[int]): DMRS sequence initialization for each UE
            [TS38.211, sec 7.4.1.1.2].
        layers (List[int]): Number of layers for each UE.
        dmrs_ports (List[int]): DMRS ports for each UE.
        rntis (List[int]): RNTI for each UE.
        data_scids (List[int]): Data scrambling IDs for each UE, more precisely
            `dataScramblingIdentityPdsch` [TS38.211, sec 7.3.1.1].
        mcs_tables (List[int]): MCS table to use for each UE (see TS 38.214).
        mcs_indices (List[int]): MCS indices for each UE.
        target_code_rates (List[int]): Code rate for each UE. This is the number of information
            bits per 1024 coded bits expressed in 0.1 bit units.
        mod_orders (List[int]): Modulation order for each UE.
        tb_sizes (List[int]): TB size in bytes for each UE.
        rvs (List[int]): Redundancy version per UE (default: 0 for each UE).
        ndis (List[int]): New data indicator per UE (default: 1 for each UE).

    Returns:
        List[PuschConfig]: A list of `PuschConfig` configuration objects (with just one element).
    """
    # Set the default values.
    if dmrs_syms is None:
        dmrs_syms = [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0]

    if scids is None:
        valid_scids = [0, 0, 1, 1, 0, 0, 1, 1]
        scids = valid_scids[:num_ues]
    if layers is None:
        layers = list(num_ues * [1, ])
    if dmrs_ports is None:
        valid_dmrs_ports = [1, 4, 2, 8, 16, 64, 32, 128]
        dmrs_ports = valid_dmrs_ports[:num_ues]
    if rntis is None:
        rntis = list(np.arange(1, num_ues + 1))
    if data_scids is None:
        data_scids = list(np.arange(1, num_ues + 1))

    if mcs_tables is None:
        mcs_tables = [0, ] * num_ues
    if mcs_indices is None:
        mcs_indices = list(num_ues * [0, ])
    if target_code_rates is None:
        target_code_rates = list(num_ues * [1930, ])
    if mod_orders is None:
        mod_orders = list(num_ues * [2, ])
    if tb_sizes is None:
        tb_sizes = [96321, ] * num_ues
    if rvs is None:
        rvs = [0, ] * num_ues
    if ndis is None:
        ndis = [1, ] * num_ues

    pusch_ue_configs = []
    for ue_idx in range(num_ues):
        pusch_ue_config = PuschUeConfig(
            scid=scids[ue_idx],
            layers=layers[ue_idx],
            dmrs_ports=dmrs_ports[ue_idx],
            rnti=rntis[ue_idx],
            data_scid=data_scids[ue_idx],
            mcs_table=mcs_tables[ue_idx],
            mcs_index=mcs_indices[ue_idx],
            code_rate=target_code_rates[ue_idx],
            mod_order=mod_orders[ue_idx],
            tb_size=tb_sizes[ue_idx],
            rv=rvs[ue_idx],
            ndi=ndis[ue_idx]
        )
        pusch_ue_configs.append(pusch_ue_config)

    pusch_configs = [PuschConfig(
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
        ue_configs=pusch_ue_configs
    )]
    return pusch_configs

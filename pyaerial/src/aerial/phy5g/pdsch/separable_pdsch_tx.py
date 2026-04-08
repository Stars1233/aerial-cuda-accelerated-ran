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

"""pyAerial - Separable PDSCH Tx pipeline built from pyAerial components."""
from typing import Any
from typing import List
from typing import Optional

import cupy as cp  # type: ignore
import numpy as np

from aerial.phy5g.api import Array
from aerial.phy5g.api import Pipeline
from aerial.phy5g.api import PipelineFactory
from aerial.phy5g.config import AerialPdschTxConfig
from aerial.phy5g.config import PdschConfig
from aerial.phy5g.config import CsiRsConfig
from aerial.phy5g.pdsch.pdsch_tx_base import PdschTxPipeline
from aerial.phy5g.pdsch.pdsch_tx import PdschTx
from aerial.phy5g.pdsch.pdsch_tx_base import NUM_RE_PER_PRB
from aerial.phy5g.pdsch.pdsch_tx_base import NUM_PRB_MAX
from aerial.phy5g.pdsch.pdsch_tx_base import NUM_SYMBOLS
from aerial.phy5g.pdsch.pdsch_tx_base import MAX_DL_LAYERS
from aerial.phy5g.pdsch.dmrs_tx import DmrsTx
from aerial.phy5g.ldpc import LdpcRateMatch
from aerial.phy5g.ldpc import LdpcEncoder
from aerial.phy5g.ldpc import CrcEncoder
from aerial.phy5g.ldpc.util import get_pdsch_config_attrs
from aerial.util.cuda import CudaStream


class SeparablePdschTxPipelineFactory(PipelineFactory[AerialPdschTxConfig]):
    """Factory for building a `SeparablePdschTx` pipeline."""

    def create(self,
               config: AerialPdschTxConfig,
               cuda_stream: CudaStream,
               **kwargs: Any) -> Pipeline:
        """Create the pipeline.

        Args:
            config (AerialPdschTxConfig): Pipeline configuration object.
            cuda_stream (CudaStream): CUDA stream used to run the pipeline.

        Returns:
            SeparablePdschTx: A `SeparablePdschTx` pipeline object.
        """
        max_num_tbs = kwargs.get("max_num_tbs", 128)
        separable_pdsch_tx = SeparablePdschTx(
            cell_id=config.cell_id,
            num_rx_ant=config.num_rx_ant,
            num_tx_ant=config.num_tx_ant,
            num_ul_bwp=config.num_ul_bwp,
            num_dl_bwp=config.num_dl_bwp,
            max_num_tbs=max_num_tbs,
            cuda_stream=cuda_stream
        )
        return separable_pdsch_tx


class SeparablePdschTx(PdschTxPipeline[PdschConfig, Array]):
    """Separable PDSCH transmitter.

    This class implements the whole PDSCH transmission pipeline from the transmitted
    transport block to the transmitted frequency-domain symbols. As opposed to
    :class:`~aerial.phy5g.pdsch.pdsch_tx.PdschTx`, this class implements the pipeline using
    separable PDSCH transmitter components.
    """

    def __init__(self,
                 *,
                 cell_id: int,
                 num_rx_ant: int,
                 num_tx_ant: int,
                 num_ul_bwp: int = NUM_PRB_MAX,
                 num_dl_bwp: int = NUM_PRB_MAX,
                 max_num_cells: int = 1,
                 max_num_tbs: int = 1,
                 cuda_stream: Optional[CudaStream] = None) -> None:
        """Initialize SeparablePdschTx.

        Args:
            cell_id (int): Physical cell ID.
            num_rx_ant (int): Number of receive antennas.
            num_tx_ant (int): Number of transmit antennas.
            num_ul_bwp (int): Number of UL BWP.
            num_dl_bwp (int): Number of DL BWP.
            max_num_cells (int): Maximum number of cells per slot.
            max_num_tbs (int): Maximum number of transport blocks per cell group.
            cuda_stream (Optional[CudaStream]): CUDA stream used to run the pipeline. If not given,
                a new CudaStream is created. Use ``with stream:`` to scope work; call
                ``stream.synchronize()`` explicitly when sync is needed.
        """
        self._cuda_stream = CudaStream() if cuda_stream is None else cuda_stream

        self.cell_id = cell_id
        self.num_rx_ant = num_rx_ant
        self.num_tx_ant = num_tx_ant
        self.num_ul_bwp = num_ul_bwp
        self.num_dl_bwp = num_dl_bwp

        # Create the components.
        self.crc_encoder = CrcEncoder(
            max_num_tbs=max_num_tbs,
            cuda_stream=self._cuda_stream
        )
        self.ldpc_encoder = LdpcEncoder(cuda_stream=self._cuda_stream)
        self.ldpc_rate_match = LdpcRateMatch(
            enable_scrambling=True,
            max_num_tbs=max_num_tbs,
            num_dl_bwp_prbs=num_dl_bwp,
            cuda_stream=self._cuda_stream
        )
        self.dmrs_tx = DmrsTx(
            cuda_stream=self._cuda_stream,
            num_bwp_prbs=num_dl_bwp,
            max_num_cells=max_num_cells,
            max_num_tbs=max_num_tbs
        )

    def __call__(self,
                 slot: int,
                 tb_inputs: List[Array],
                 config: List[PdschConfig],
                 csi_rs_config: List[CsiRsConfig] = None,
                 **kwargs: Any) -> Array:
        """Run the pipeline.

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

    def run(self,
            tb_inputs: List[Array],
            slot: int,
            pdsch_configs: List[PdschConfig],
            csi_rs_configs: List[CsiRsConfig] = None) -> Array:
        """Run the pipeline.

        Args:
            tb_inputs (List[Array]): Transport blocks to be transmitted, one per UE.
            slot (int): Slot number.
            pdsch_configs (List[PdschConfig]): Dynamic slot configuration in this slot.
                One entry per UE group.
            csi_rs_configs (List[CsiRsConfig]): Optional parameters for CSI-RS. Note: This only
                leaves the CSI-RS REs empty. To actually add in the CSI-RS signals, one
                needs to call the CSI-RS transmitter separately.

        Returns:
            Array: Transmitted OFDM symbols in a frequency x time x antenna tensor.
        """
        code_blocks = self.crc_encoder.encode(
            tb_inputs=tb_inputs,
            pdsch_configs=pdsch_configs
        )

        coded_blocks = self.ldpc_encoder.encode(
            code_blocks=code_blocks,
            pdsch_configs=pdsch_configs
        )

        # Initialize the output buffer.
        with self._cuda_stream:
            tx_buffer = cp.zeros(
                (self.num_dl_bwp * NUM_RE_PER_PRB, NUM_SYMBOLS, MAX_DL_LAYERS),
                dtype=np.complex64,
            )

        # Fill in the modulation symbols.
        tx_buffer = self.ldpc_rate_match.rm_mod_layer_map(
            tx_buffer=tx_buffer,
            coded_blocks=coded_blocks,
            pdsch_configs=pdsch_configs,
            csi_rs_configs=csi_rs_configs
        )

        # Fill in DMRS symbols.
        tx_buffer = self.dmrs_tx.run(slot=slot,
                                     tx_buffers=[tx_buffer],
                                     pdsch_configs=pdsch_configs)[0]

        # Map cuPHY outputs to Tx antenna ports.
        num_ues = len(tb_inputs)
        attrs = ["dmrs_ports", "scid", "precoding_matrix"]
        pdsch_config_attrs = get_pdsch_config_attrs(pdsch_configs, attrs)
        tx_buffer = PdschTx.cuphy_to_tx(
            tx_slot=tx_buffer,
            num_ues=num_ues,
            dmrs_ports=pdsch_config_attrs["dmrs_ports"],
            scids=pdsch_config_attrs["scid"],
            precoding_matrices=pdsch_config_attrs["precoding_matrix"]
        )

        return tx_buffer

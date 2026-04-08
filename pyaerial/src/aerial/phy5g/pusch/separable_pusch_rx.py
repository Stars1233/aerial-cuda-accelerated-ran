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

"""pyAerial - Separable PUSCH Rx pipeline built from pyAerial components."""
from typing import Any, Optional
from typing import List
from typing import Tuple

import cupy as cp  # type: ignore
import numpy as np

from aerial.phy5g.api import Array
from aerial.util.cuda import CudaStream
from aerial.phy5g.api import Pipeline
from aerial.phy5g.api import PipelineFactory
from aerial.phy5g.config import AerialPuschRxConfig
from aerial.phy5g.config import PuschConfig
from aerial.phy5g.pusch.pusch_rx_base import PuschRxPipeline
from aerial.phy5g.algorithms import ChannelEstimator
from aerial.phy5g.algorithms import ChannelEqualizer
from aerial.phy5g.algorithms import NoiseIntfEstimator
from aerial.phy5g.ldpc import LdpcDeRateMatch
from aerial.phy5g.ldpc import LdpcDecoder
from aerial.phy5g.ldpc import CrcChecker


class SeparablePuschRxPipelineFactory(PipelineFactory[AerialPuschRxConfig]):
    """Factory for building a `SeparablePuschRx` pipeline."""

    def create(self,
               config: AerialPuschRxConfig,
               cuda_stream: CudaStream,
               **kwargs: Any) -> Pipeline:
        """Create the pipeline.

        Args:
            config (AerialPuschRxConfig): Pipeline configuration object.
            cuda_stream (CudaStream): CUDA stream used to run the pipeline.

        Returns:
            SeparablePuschRx: A `SeparablePuschRx` pipeline object.
        """
        separable_pusch_rx = SeparablePuschRx(
            num_rx_ant=config.num_rx_ant,
            enable_pusch_tdi=config.enable_pusch_tdi,
            eq_coeff_algo=config.eq_coeff_algo,
            chest_factory_settings_filename=config.chest_factory_settings_filename,
            cuda_stream=cuda_stream
        )
        return separable_pusch_rx


class SeparablePuschRx(PuschRxPipeline[PuschConfig, Array]):
    """Separable PUSCH receiver pipeline.

    This class implements the whole PUSCH reception pipeline from the received OFDM
    post-FFT symbols to the received transport block (along with CRC check). As opposed to
    :class:`~aerial.phy5g.pusch.pusch_rx.PuschRx`, this class implements the pipeline using
    separable PUSCH receiver components.
    """

    def __init__(self,
                 *,
                 num_rx_ant: int,
                 enable_pusch_tdi: int,
                 eq_coeff_algo: int,
                 chest_factory_settings_filename: Optional[str],
                 cuda_stream: CudaStream) -> None:
        """Initialize SeparablePuschRx.

        Args:
            num_rx_ant (int): Number of receive antennas.
            enable_pusch_tdi (int): Time domain interpolation on PUSCH.

                - 0: Disable (default).
                - 1: Enable.
            eq_coeff_algo (int): Algorithm for equalizer coefficient computation.

                - 0 - ZF.
                - 1 - MMSE (default).
                - 2 - MMSE-IRC.
            cuda_stream (CudaStream): CUDA stream. Use ``with stream:`` to scope work; call
                ``stream.synchronize()`` explicitly when sync is needed.
        """
        self._cuda_stream = cuda_stream

        # Build the components of the receiver.
        self.channel_estimator = ChannelEstimator(
            num_rx_ant=num_rx_ant,
            cuda_stream=self._cuda_stream,
            chest_factory_settings_filename=chest_factory_settings_filename
        )
        self.channel_equalizer = ChannelEqualizer(
            num_rx_ant=num_rx_ant,
            enable_pusch_tdi=enable_pusch_tdi,
            eq_coeff_algo=eq_coeff_algo,
            cuda_stream=self._cuda_stream
        )
        self.noise_intf_estimator = NoiseIntfEstimator(
            num_rx_ant=num_rx_ant,
            eq_coeff_algo=eq_coeff_algo,
            cuda_stream=self._cuda_stream
        )
        self.derate_match = LdpcDeRateMatch(
            enable_scrambling=True,
            cuda_stream=self._cuda_stream
        )
        self.decoder = LdpcDecoder(cuda_stream=self._cuda_stream)
        self.crc_checker = CrcChecker(cuda_stream=self._cuda_stream)

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
        use_cupy = isinstance(rx_slot, cp.ndarray)

        # Channel estimation.
        ch_est = self.channel_estimator.estimate(
            rx_slot=rx_slot,
            slot=slot,
            pusch_configs=config
        )

        # Noise and interference estimation.
        lw_inv, noise_var_pre_eq = self.noise_intf_estimator.estimate(
            rx_slot=rx_slot,
            channel_est=ch_est,
            slot=slot,
            pusch_configs=config
        )

        # Channel equalization and soft demapping. The first return value are the LLRs,
        # second are the equalized symbols. We only want the LLRs now.
        llrs = self.channel_equalizer.equalize(
            rx_slot=rx_slot,
            channel_est=ch_est,
            lw_inv=lw_inv,
            noise_var_pre_eq=noise_var_pre_eq,
            pusch_configs=config
        )[0]

        # (De)rate matching.
        coded_blocks = self.derate_match.derate_match(
            input_llrs=llrs,
            pusch_configs=config
        )

        # LDPC decoding.
        code_blocks = self.decoder.decode(
            input_llrs=coded_blocks,
            pusch_configs=config
        )

        # Code block desegmentation, CRC removal, CRC checking.
        tbs, tb_crcs = self.crc_checker.check_crc(
            input_bits=code_blocks,
            pusch_configs=config
        )

        if use_cupy:
            tbs = [tb.get(order='F') for tb in tbs]
            tb_crcs = [crc.get(order='F') for crc in tb_crcs]

        # To comply with the API.
        tb_crcs = np.concatenate(tb_crcs)

        return tb_crcs, tbs  # type: ignore

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

"""pyAerial library - Carrier frequency offset and timing advance estimation (for equalization)."""
from typing import Generic
from typing import List
from typing import Optional
from typing import Tuple

import cupy as cp  # type: ignore
import numpy as np

from aerial import pycuphy  # type: ignore
from aerial.util.cuda import CudaStream
from aerial.phy5g.api import Array
from aerial.phy5g.config import PuschConfig
from aerial.phy5g.config import PuschUeConfig
from aerial.phy5g.config import _pusch_config_to_cuphy
from aerial.pycuphy.util import get_pusch_stat_prms


class CfoTaEstimator(Generic[Array]):
    """CFO and TA estimator class.

    This class implements an algorithm for carrier frequency offset and timing advance
    estimation. It calls the corresponding cuPHY algorithms and provides the estimates
    as needed for other cuPHY algorithms.

    It needs channel estimates as its input.
    """
    def __init__(self,
                 *,
                 num_rx_ant: int,
                 mu: int = 1,
                 enable_cfo_correction: bool = True,
                 enable_weighted_ave_cfo_est: bool = False,
                 enable_to_estimation: bool = True,
                 cuda_stream: Optional[CudaStream] = None) -> None:
        """Initialize CfoTaEstimator.

        Args:
            num_rx_ant (int): Number of receive antennas.
            mu (int): Numerology. Values in [0, 3]. Default: 1.
            enable_cfo_correction (int): Enable/disable CFO correction:

                - 0: Disable.
                - 1: Enable (default).

            enable_weighted_ave_cfo_est (int): Enable/disable CFO weighted average estimation:

                - 0: Disable.
                - 1: Enable (default).

            enable_to_estimation (int): Enable/disable time offset estimation:

                - 0: Disable.
                - 1: Enable (default).

            cuda_stream (Optional[CudaStream]): CUDA stream. If not given, a new CudaStream is
                created. Use ``with stream:`` to scope work; call ``stream.synchronize()``
                explicitly when sync is needed.
        """
        self._cuda_stream = CudaStream() if cuda_stream is None else cuda_stream

        pusch_stat_prms = get_pusch_stat_prms(
            num_rx_ant=num_rx_ant,
            mu=mu,
            enable_cfo_correction=int(enable_cfo_correction),
            enable_weighted_ave_cfo_est=int(enable_weighted_ave_cfo_est),
            enable_to_estimation=int(enable_to_estimation)
        )
        self._pusch_params = pycuphy.PuschParams()
        self._pusch_params.set_stat_prms(pusch_stat_prms)

        self.cfo_ta_estimator = pycuphy.CfoTaEstimator(self._cuda_stream.handle)
        self.cfo_est = None  # type: List[Array]
        self.cfo_hz = None  # type: Array
        self.ta = None  # type: Array

    def estimate(self,
                 *,
                 channel_est: List[Array],
                 pusch_configs: List[PuschConfig] = None,
                 num_ues: int = None,
                 num_prbs: int = None,
                 dmrs_syms: List[int] = None,
                 dmrs_max_len: int = None,
                 dmrs_add_ln_pos: int = None,
                 layers: List[int] = None) -> Tuple[Array, Array]:
        """Estimate carrier frequency offset and timing advance.

        Args:
            channel_est (List[Array]): The channel estimates as a
                Rx ant x layer x frequency x time Numpy or CuPy array, per UE group.
            pusch_configs (List[PuschConfig]): List of PUSCH configuration objects, one per UE
                group. If this argument is given, the rest are ignored. If not given, all other
                arguments need to be given (only one UE group supported in that case).
            num_ues (int): Number of UEs in the UE group.
            num_prbs (int): Number of allocated PRBs for the UE group.
            dmrs_syms (List[int]): For the UE group, a list of binary numbers each indicating
                whether the corresponding symbol is a DMRS symbol. The length of the list equals
                the number of symbols in the slot. 0 means no DMRS in the symbol and 1 means
                the symbol is a DMRS symbol.
            dmrs_max_len (int): The `maxLength` parameter, value 1 or 2, meaning that DMRS are
                single-symbol DMRS or single- or double-symbol DMRS.
            dmrs_add_ln_pos (int): Number of additional DMRS positions.
            layers (List[int]): Number of layers for each UE. The length of the list equals the
                number of UEs.

        Returns:
            Array, Array: A tuple containing:

            - *Array*: Carrier frequency offset per UE, in Hz.

            - *Array*: Timing offset per UE, in microseconds.
        """
        cpu_copy = isinstance(channel_est[0], np.ndarray)
        with self._cuda_stream:
            channel_est = [cp.array(elem, order='F', dtype=cp.complex64) for elem in channel_est]
            # Dummy empty Rx data (Rx data not needed).
            rx_data = cp.zeros((3276, 14, 1), dtype=cp.complex64)

        # If pusch_configs is given, use that directly. Else all the other parameters
        # need to be given (only a single UE group).
        if pusch_configs is None:

            # In this case all the other parameters need to be set.
            if num_ues is None:
                raise ValueError("Argument num_ues is not set!")
            if num_prbs is None:
                raise ValueError("Argument num_prbs is not set!")
            if dmrs_syms is None:
                raise ValueError("Argument dmrs_syms is not set!")
            if dmrs_max_len is None:
                raise ValueError("Argument dmrs_max_len is not set!")
            if dmrs_add_ln_pos is None:
                raise ValueError("Argument dmrs_add_ln_pos is not set!")
            if layers is None:
                raise ValueError("Argument layers is not set!")

            pusch_ue_configs = [PuschUeConfig(layers=layers[ue]) for ue in range(num_ues)]
            pusch_configs = [PuschConfig(
                num_prbs=num_prbs,
                dmrs_syms=dmrs_syms,
                dmrs_max_len=dmrs_max_len,
                dmrs_add_ln_pos=dmrs_add_ln_pos,
                ue_configs=pusch_ue_configs
            )]

        # Wrap CuPy arrays into pycuphy types.
        channel_est = [pycuphy.CudaArrayComplexFloat(elem) for elem in channel_est]
        rx_data = pycuphy.CudaArrayComplexFloat(rx_data)

        pusch_dyn_prms = _pusch_config_to_cuphy(
            cuda_stream=self._cuda_stream,
            rx_data=[rx_data],
            slot=0,  # Not used.
            pusch_configs=pusch_configs
        )

        self._pusch_params.set_dyn_prms(pusch_dyn_prms)

        self.cfo_est = self.cfo_ta_estimator.estimate(channel_est, self._pusch_params)
        self.cfo_hz = self.cfo_ta_estimator.get_cfo_hz()
        self.ta = self.cfo_ta_estimator.get_ta()

        with self._cuda_stream:
            self.cfo_est = [cp.array(elem) for elem in self.cfo_est]
            self.cfo_hz = cp.array(self.cfo_hz)
            self.ta = cp.array(self.ta)

            if cpu_copy:
                self.cfo_est = \
                    [elem.get(order='F') for elem in self.cfo_est]  # type: ignore[union-attr]
                self.cfo_hz = self.cfo_hz.get(order='F')
                self.ta = self.ta.get(order='F')

        return self.cfo_hz, self.ta

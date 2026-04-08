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

"""pyAerial library - Noise and interference estimation (for equalization)."""
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
from aerial.pycuphy.types import PuschEqCoefAlgoType
from aerial.pycuphy.util import get_pusch_stat_prms


class NoiseIntfEstimator(Generic[Array]):
    """Noise and interference estimator class.

    This class implements an algorithm for noise and interference estimation.
    It calls the corresponding cuPHY algorithms and provides the estimates
    as needed for cuPHY equalization and soft demapping.

    It needs channel estimates as its input, along with the received data symbols.
    """
    def __init__(self,
                 *,
                 num_rx_ant: int,
                 eq_coeff_algo: int,
                 cuda_stream: Optional[CudaStream] = None) -> None:
        """Initialize NoiseIntfEstimator.

        Args:
            num_rx_ant (int): Number of receive antennas.
            eq_coeff_algo (int): Algorithm used to compute equalizer coefficients.

                - 0: Zero-forcing equalizer.
                - 1: MMSE with noise variance only.
                - 2: MMSE-IRC.
            cuda_stream (Optional[CudaStream]): CUDA stream. If not given, a new CudaStream is
                created. Use ``with stream:`` to scope work; call ``stream.synchronize()``
                explicitly when sync is needed.
        """
        self._cuda_stream = CudaStream() if cuda_stream is None else cuda_stream

        pusch_stat_prms = get_pusch_stat_prms(
            num_rx_ant=num_rx_ant,
            eq_coeff_algo=PuschEqCoefAlgoType(eq_coeff_algo)
        )
        self._pusch_params = pycuphy.PuschParams()
        self._pusch_params.set_stat_prms(pusch_stat_prms)

        self.noise_intf_estimator = pycuphy.NoiseIntfEstimator(self._cuda_stream.handle)
        self.lw_inv = None  # type: List[Array]
        self.noise_var_pre_eq = None  # type: Array

    def estimate(self,  # pylint: disable=too-many-arguments
                 *,
                 rx_slot: Array,
                 channel_est: List[Array],
                 slot: int,
                 pusch_configs: List[PuschConfig] = None,
                 num_ues: int = None,
                 num_dmrs_cdm_grps_no_data: int = None,
                 dmrs_scrm_id: int = None,
                 start_prb: int = None,
                 num_prbs: int = None,
                 dmrs_syms: List[int] = None,
                 dmrs_max_len: int = None,
                 dmrs_add_ln_pos: int = None,
                 start_sym: int = None,
                 num_symbols: int = None,
                 scids: List[int] = None,
                 layers: List[int] = None,
                 dmrs_ports: List[int] = None) -> Tuple[List[Array], Array]:
        """Estimate noise and interference.

        This runs the cuPHY noise and interference estimation for all UE groups included in
        `pusch_configs`. If this argument is not given, all the other arguments need to be
        given and cuPHY noise and interference estimation is run only for a single UE group
        sharing the same time-frequency resources, i.e. having the same PRB allocation, and the
        same start symbol and number of allocated symbols.

        The method can be called using either Numpy or CuPy arrays. In case the input arrays
        are located on the GPU (CuPy), the output will be on the GPU (CuPy). So the return type
        shall be the same as used for `rx_slot` when calling the method.

        Args:
            rx_slot (Array): Input received data as a frequency x time x Rx
                antenna Numpy or CuPy array with type `complex64` entries.
            channel_est (List[Array]): The channel estimates as a
                Rx ant x layer x frequency x time Numpy or CuPy array, per UE group.
            slot (int): Slot number.
            pusch_configs (List[PuschConfig]): List of PUSCH configuration objects, one per UE
                group. If this argument is given, the rest are ignored. If not given, all other
                arguments need to be given (only one UE group supported in that case).
            num_ues (int): Number of UEs in the UE group.
            num_dmrs_cdm_grps_no_data (int): Number of DMRS CDM groups without data
                [3GPP TS 38.212, sec 7.3.1.1]. Value: 1->3.
            dmrs_scrm_id (int): DMRS scrambling ID.
            start_prb (int): Start PRB index of the UE allocation.
            num_prbs (int): Number of allocated PRBs for the UE group.
            dmrs_syms (List[int]): For the UE group, a list of binary numbers each indicating
                whether the corresponding symbol is a DMRS symbol. The length of the list equals
                the number of symbols in the slot. 0 means no DMRS in the symbol and 1 means
                the symbol is a DMRS symbol.
            dmrs_max_len (int): The `maxLength` parameter, value 1 or 2, meaning that DMRS are
                single-symbol DMRS or single- or double-symbol DMRS.
            dmrs_add_ln_pos (int): Number of additional DMRS positions.
            start_sym (int): Start symbol index for the UE group allocation.
            num_symbols (int): Number of symbols in the UE group allocation.
            scids (List[int]): DMRS sequence initialization SCID [TS38.211, sec 7.4.1.1.2] for each
                UE. Value is 0 or 1.
            layers (List[int]): Number of layers for each UE. The length of the list equals the
                number of UEs.
            dmrs_ports (List[int]): DMRS ports for each UE. The format of each entry is in the
                SCF FAPI format as follows: A bitmap (mask) starting from the LSB where each bit
                indicates whether the corresponding DMRS port index is used.

        Returns:
            List[Array], Array: A tuple containing:

            - *List[Array]*:
              Inverse of the Cholesky decomposition of the noise/interference
              covariance matrix per PRB, per UE group. The size of each entry in this list is
              number of Rx antennas x number of Rx antennas x number of PRBs.

            - *Array*:
              Pre-equalization wideband noise variance estimate per UE, i.e.
              one value per UE averaged over the whole frequency allocation. This
              value is in dB.
        """
        cpu_copy = isinstance(rx_slot, np.ndarray)
        with self._cuda_stream:
            rx_slot = cp.array(rx_slot, order='F', dtype=cp.complex64)
            channel_est = [cp.array(elem, order='F', dtype=cp.complex64) for elem in channel_est]

        # If pusch_configs is given, use that directly. Else all the other parameters
        # need to be given (only a single UE group).
        if pusch_configs is None:

            pusch_configs = self._get_pusch_configs(
                num_ues=num_ues,
                num_dmrs_cdm_grps_no_data=num_dmrs_cdm_grps_no_data,
                dmrs_scrm_id=dmrs_scrm_id,
                start_prb=start_prb,
                num_prbs=num_prbs,
                dmrs_syms=dmrs_syms,
                dmrs_max_len=dmrs_max_len,
                dmrs_add_ln_pos=dmrs_add_ln_pos,
                start_sym=start_sym,
                num_symbols=num_symbols,
                scids=scids,
                layers=layers,
                dmrs_ports=dmrs_ports
            )

        # Wrap CuPy arrays into pycuphy types.
        rx_slot = pycuphy.CudaArrayComplexFloat(rx_slot)
        channel_est = [pycuphy.CudaArrayComplexFloat(elem) for elem in channel_est]

        pusch_dyn_prms = _pusch_config_to_cuphy(
            cuda_stream=self._cuda_stream,
            rx_data=[rx_slot],
            slot=slot,
            pusch_configs=pusch_configs
        )
        self._pusch_params.set_dyn_prms(pusch_dyn_prms)

        self.lw_inv = self.noise_intf_estimator.estimate(channel_est, self._pusch_params)
        self.noise_var_pre_eq = self.noise_intf_estimator.get_info_noise_var_pre_eq()

        # Transform the output to cupy array.
        with self._cuda_stream:
            self.lw_inv = [cp.array(elem) for elem in self.lw_inv]
            self.noise_var_pre_eq = cp.array(self.noise_var_pre_eq)

            if cpu_copy:
                self.lw_inv = \
                    [elem.get(order='F') for elem in self.lw_inv]  # type: ignore[union-attr]
                self.noise_var_pre_eq = self.noise_var_pre_eq.get(order='F')

        return self.lw_inv, self.noise_var_pre_eq

    def _get_pusch_configs(self,  # pylint: disable=too-many-arguments
                           *,
                           num_ues: int = None,
                           num_dmrs_cdm_grps_no_data: int = None,
                           dmrs_scrm_id: int = None,
                           start_prb: int = None,
                           num_prbs: int = None,
                           dmrs_syms: List[int] = None,
                           dmrs_max_len: int = None,
                           dmrs_add_ln_pos: int = None,
                           start_sym: int = None,
                           num_symbols: int = None,
                           scids: List[int] = None,
                           layers: List[int] = None,
                           dmrs_ports: List[int] = None) -> List[PuschConfig]:
        """Helper to get PUSCH configs based on other parameters."""
        # In this case all the other parameters need to be set.
        if num_ues is None:
            raise ValueError("Argument num_ues is not set!")
        if num_dmrs_cdm_grps_no_data is None:
            raise ValueError("Argument num_dmrs_cdm_grps_no_data is not set!")
        if dmrs_scrm_id is None:
            raise ValueError("Argument dmrs_scrm_id is not set!")
        if start_prb is None:
            raise ValueError("Argument start_prb is not set!")
        if num_prbs is None:
            raise ValueError("Argument num_prbs is not set!")
        if dmrs_syms is None:
            raise ValueError("Argument dmrs_syms is not set!")
        if dmrs_max_len is None:
            raise ValueError("Argument dmrs_max_len is not set!")
        if dmrs_add_ln_pos is None:
            raise ValueError("Argument dmrs_add_ln_pos is not set!")
        if start_sym is None:
            raise ValueError("Argument start_sym is not set!")
        if num_symbols is None:
            raise ValueError("Argument num_symbols is not set!")
        if scids is None:
            raise ValueError("Argument scids is not set!")
        if layers is None:
            raise ValueError("Argument layers is not set!")
        if dmrs_ports is None:
            raise ValueError("Argument dmrs_ports is not set!")

        pusch_ue_configs = []
        for ue_idx in range(num_ues):
            pusch_ue_config = PuschUeConfig(
                scid=scids[ue_idx],
                layers=layers[ue_idx],
                dmrs_ports=dmrs_ports[ue_idx]
            )
            pusch_ue_configs.append(pusch_ue_config)

        pusch_configs = [PuschConfig(
            num_dmrs_cdm_grps_no_data=num_dmrs_cdm_grps_no_data,
            dmrs_scrm_id=dmrs_scrm_id,
            start_prb=start_prb,
            num_prbs=num_prbs,
            dmrs_syms=dmrs_syms,
            dmrs_max_len=dmrs_max_len,
            dmrs_add_ln_pos=dmrs_add_ln_pos,
            start_sym=start_sym,
            num_symbols=num_symbols,
            ue_configs=pusch_ue_configs
        )]

        return pusch_configs

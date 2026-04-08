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

"""pyAerial library - RSRP and pre-/post-equalizer SINR estimation."""
from typing import Generic
from typing import List
from typing import Optional
from typing import Tuple

import cupy as cp  # type: ignore
import numpy as np

from aerial import pycuphy
from aerial.util.cuda import CudaStream
from aerial.phy5g.api import Array
from aerial.phy5g.config import PuschConfig
from aerial.phy5g.config import PuschUeConfig
from aerial.phy5g.config import _pusch_config_to_cuphy
from aerial.pycuphy.util import get_pusch_stat_prms


class RsrpEstimator(Generic[Array]):
    """RSRP, post- and pre-equalizer SINR estimator class.

    This class implements RSRP estimation as well as post- and pre-equalizer SINR
    estimation for PUSCH receiver pipeline.
    """
    def __init__(self,
                 num_rx_ant: int,
                 enable_pusch_tdi: int,
                 cuda_stream: Optional[CudaStream] = None) -> None:
        """Initialize RsrpEstimator.

        Args:
            num_rx_ant (int): Number of receive antennas.
            enable_pusch_tdi (int): Whether time-interpolation is used in computing equalizer
                coefficients. This impacts post-equalizer SINR.
            cuda_stream (Optional[CudaStream]): CUDA stream. If not given, a new CudaStream is
                created. Use ``with stream:`` to scope work; call ``stream.synchronize()``
                explicitly when sync is needed.
        """
        self._cuda_stream = CudaStream() if cuda_stream is None else cuda_stream

        pusch_stat_prms = get_pusch_stat_prms(
            num_rx_ant=num_rx_ant,
            enable_pusch_tdi=enable_pusch_tdi
        )
        self._pusch_params = pycuphy.PuschParams()
        self._pusch_params.set_stat_prms(pusch_stat_prms)

        self.rsrp_estimator = pycuphy.RsrpEstimator(self._cuda_stream.handle)
        self.rsrp = None  # type: Array
        self.pre_eq_sinr = None  # type: Array
        self.post_eq_sinr = None  # type: Array
        self.noise_var_post_eq = None  # type: Array

    def estimate(self,
                 *,
                 channel_est: List[Array],
                 ree_diag_inv: List[Array],
                 noise_var_pre_eq: Array,
                 pusch_configs: List[PuschConfig] = None,
                 num_ues: int = None,
                 num_prbs: int = None,
                 dmrs_add_ln_pos: int = None,
                 layers: List[int] = None) -> Tuple[Array, Array, Array]:
        """Run RSRP and post- and pre-equalizer SINR estimation.

        The method can be called using either Numpy or CuPy arrays. In case the input arrays
        are located on the GPU (CuPy), the output will be on the GPU (CuPy). So the return type
        shall be the same as used for `rx_slot` when calling the method.

        Args:
            channel_est (List[Array]):  The channel estimates as a
                Rx ant x layer x frequency x time Numpy or CuPy array, per UE group.
            ree_diag_inv  (List[Array]): Inverse of post-equalizer residual
                error covariance diagonal, per UE group.
            noise_var_pre_eq (Array): Average pre-equalizer noise variance in dB.
                One value per UE group.
            pusch_configs (List[PuschConfig]): List of PUSCH configuration objects, one per UE
                group. If this argument is given, the rest are ignored. If not given, all other
                arguments need to be given (only one UE group supported in that case).
            num_ues (int): Number of UEs in the UE group.
            num_prbs (int): Number of allocated PRBs for the UE group.
            dmrs_add_ln_pos (int): Number of additional DMRS positions. This is used to derive the
                total number of DMRS symbols.
            layers (List[int]): Number of layers for each UE.

        Returns:
            Array, Array, Array: A tuple containing:

            - *Array*: RSRP values per UE.

            - *Array*: Pre-equalization SINR values per UE.

            - *Array*: Post-equalization SINR values per UE.

        """
        cpu_copy = isinstance(channel_est[0], np.ndarray)
        with self._cuda_stream:
            channel_est = [cp.array(elem, order='F', dtype=cp.complex64) for elem in channel_est]
            ree_diag_inv = [cp.array(elem, order='F') for elem in ree_diag_inv]
            noise_var_pre_eq = cp.array(noise_var_pre_eq, dtype=cp.float32)
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
            if dmrs_add_ln_pos is None:
                raise ValueError("Argument dmrs_add_ln_pos is not set!")
            if layers is None:
                raise ValueError("Argument layers is not set!")

            pusch_ue_configs = []
            for ue_idx in range(num_ues):
                pusch_ue_config = PuschUeConfig(
                    layers=layers[ue_idx],
                )
                pusch_ue_configs.append(pusch_ue_config)

            pusch_configs = [PuschConfig(
                num_prbs=num_prbs,
                dmrs_add_ln_pos=dmrs_add_ln_pos,
                ue_configs=pusch_ue_configs
            )]

        # Wrap CuPy arrays into pycuphy types.
        rx_data = pycuphy.CudaArrayComplexFloat(rx_data)
        channel_est = [pycuphy.CudaArrayComplexFloat(elem) for elem in channel_est]
        ree_diag_inv = [pycuphy.CudaArrayFloat(elem) for elem in ree_diag_inv]
        noise_var_pre_eq = pycuphy.CudaArrayFloat(noise_var_pre_eq)

        pusch_dyn_prms = _pusch_config_to_cuphy(
            cuda_stream=self._cuda_stream,
            rx_data=[rx_data],
            slot=0,  # Not relevant here.
            pusch_configs=pusch_configs
        )
        self._pusch_params.set_dyn_prms(pusch_dyn_prms)

        self.rsrp = self.rsrp_estimator.estimate(
            channel_est,
            ree_diag_inv,
            noise_var_pre_eq,
            self._pusch_params
        )
        self.pre_eq_sinr = self.rsrp_estimator.get_sinr_pre_eq()
        self.post_eq_sinr = self.rsrp_estimator.get_sinr_post_eq()
        self.noise_var_post_eq = self.rsrp_estimator.get_info_noise_var_post_eq()

        # Transform the output to cupy array.
        with self._cuda_stream:
            self.rsrp = cp.array(self.rsrp)
            self.pre_eq_sinr = cp.array(self.pre_eq_sinr)
            self.post_eq_sinr = cp.array(self.post_eq_sinr)
            self.noise_var_post_eq = cp.array(self.noise_var_post_eq)
            if cpu_copy:
                self.rsrp = self.rsrp.get(order='F')
                self.pre_eq_sinr = self.pre_eq_sinr.get(order='F')
                self.post_eq_sinr = self.post_eq_sinr.get(order='F')
                self.noise_var_post_eq = self.noise_var_post_eq.get(order='F')

        return self.rsrp, self.pre_eq_sinr, self.post_eq_sinr

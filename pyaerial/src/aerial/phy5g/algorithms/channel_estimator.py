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

"""pyAerial library - Channel estimation."""
from typing import Generic, Optional
from typing import List

import cupy as cp  # type: ignore
import numpy as np

from aerial.util.cuda import CudaStream
from aerial.phy5g.api import Array
from aerial.phy5g.config import PuschConfig
from aerial.phy5g.config import PuschUeConfig
from aerial.phy5g.config import _pusch_config_to_cuphy
from aerial import pycuphy
from aerial.pycuphy.chest_filters import CUPHY_CHEST_COEFF_FILE
from aerial.pycuphy.chest_filters import pusch_chest_params_from_hdf5
from aerial.pycuphy.util import get_pusch_stat_prms


class ChannelEstimator(Generic[Array]):
    """Channel estimator class.

    This class implements traditional MMSE-based channel estimation on the
    DMRS symbols of the received slot signal. It outputs the channel estimates
    for all resource elements in the DMRS symbols. Similarly to many other classes
    in pyAerial, this class handles groups of UEs sharing the same time-frequency
    resources with one call, i.e. it supports MU-MIMO.
    """

    def __init__(self,  # pylint: disable=too-many-arguments
                 *,
                 num_rx_ant: int,
                 ch_est_algo: int = 1,
                 enable_per_prg_chest: int = 0,
                 enable_ul_rx_bf: int = 0,
                 cuda_stream: Optional[CudaStream] = None,
                 chest_filter_h5: str = None,
                 w_freq_array: np.ndarray = None,
                 w_freq4_array: np.ndarray = None,
                 w_freq_small_array: np.ndarray = None,
                 shift_seq_array: np.ndarray = None,
                 unshift_seq_array: np.ndarray = None,
                 shift_seq4_array: np.ndarray = None,
                 unshift_seq4_array: np.ndarray = None,
                 chest_factory_settings_filename: Optional[str] = None) -> None:
        """Initialize ChannelEstimator.

        The channel estimation filters can be given as an H5 file or directly as Numpy
        arrays. If neither is given, the channel estimator is using default filters.

        Args:
            num_rx_ant (int): Number of receive antennas.
            ch_est_algo (int): Channel estimation algorithm.

                - 0 - MMSE
                - 1 - Multi-stage MMSE with delay estimation (default)
                - 2 - RKHS not supported by pyAerial yet
                - 3 - LS channel estimation only

            enable_per_prg_chest (int): Enable/disable PUSCH per-PRG channel estimation.

                - 0: Disable (default).
                - 1: Enable.

            enable_ul_rx_bf (int): Enable/disable beamforming for PUSCH.

                - 0: Disable (default).
                - 1: Enable.

            cuda_stream (Optional[CudaStream]): CUDA stream. If not given, a new CudaStream is
                created. Use ``with stream:`` to scope work; call ``stream.synchronize()``
                explicitly when sync is needed.
            chest_filter_h5 (str): Filename of an HDF5 file containing channel estimation filters.
            w_freq_array (np.ndarray):
            w_freq4_array (np.ndarray):
            w_freq_small_array (np.ndarray):
            shift_seq_array (np.ndarray):
            unshift_seq_array (np.ndarray):
            shift_seq4_array (np.ndarray):
            unshift_seq4_array (np.ndarray):
        """
        self._cuda_stream = CudaStream() if cuda_stream is None else cuda_stream

        # Sanity check on the parameters.
        # pylint: disable=too-many-boolean-expressions
        if(chest_filter_h5 is not None and (  # noqa: E275
                (w_freq_array is not None) or
                (w_freq4_array is not None) or
                (w_freq_small_array is not None) or
                (shift_seq_array is not None) or
                (unshift_seq_array is not None) or
                (shift_seq4_array is not None) or
                (unshift_seq4_array is not None))):
            raise ValueError(
                "Either the channel estimation filter file or " +
                "the filters themselves need to be supplied but not both!"
            )

        if ((w_freq_array is not None) and
                (w_freq4_array is not None) and
                (w_freq_small_array is not None) and
                (shift_seq_array is not None) and
                (unshift_seq_array is not None) and
                (shift_seq4_array is not None) and
                (unshift_seq4_array is not None)):
            self._filters = dict(
                WFreq=w_freq_array,
                WFreq4=w_freq4_array,
                WFreqSmall=w_freq_small_array,
                ShiftSeq=shift_seq_array,
                UnShiftSeq=unshift_seq_array,
                ShiftSeq4=shift_seq4_array,
                UnShiftSeq4=unshift_seq4_array,
            )
        else:
            if chest_filter_h5 is None:
                # Use cuPHY filters.
                chest_filter_h5 = CUPHY_CHEST_COEFF_FILE
            self._filters = pusch_chest_params_from_hdf5(chest_filter_h5)

        pusch_stat_prms = get_pusch_stat_prms(
            num_rx_ant=num_rx_ant,
            ch_est_algo=ch_est_algo,
            enable_per_prg_chest=enable_per_prg_chest,
            enable_ul_rx_bf=enable_ul_rx_bf,
            chest_factory_settings_filename=chest_factory_settings_filename
        )

        self._pusch_params = pycuphy.PuschParams()
        self._pusch_params.set_stat_prms(pusch_stat_prms)
        self._pusch_params.set_filters(
            self._filters["WFreq"],
            self._filters["WFreq4"],
            self._filters["WFreqSmall"],
            self._filters["ShiftSeq"],
            self._filters["UnShiftSeq"],
            self._filters["ShiftSeq4"],
            self._filters["UnShiftSeq4"],
            self._cuda_stream.handle
        )
        if chest_factory_settings_filename is not None:
            self._pusch_params.set_chest_factory_settings_filename(chest_factory_settings_filename)
        self._channel_estimator = pycuphy.ChannelEstimator(self._pusch_params, self._cuda_stream.handle)

    def estimate(self,  # pylint: disable=too-many-arguments
                 *,
                 rx_slot: Array,
                 slot: int,
                 pusch_configs: List[PuschConfig] = None,
                 num_ues: int = None,
                 num_dmrs_cdm_grps_no_data: int = None,
                 dmrs_scrm_id: int = None,
                 start_prb: int = None,
                 num_prbs: int = None,
                 prg_size: int = None,
                 num_ul_streams: int = None,
                 dmrs_syms: List[int] = None,
                 dmrs_max_len: int = None,
                 dmrs_add_ln_pos: int = None,
                 start_sym: int = None,
                 num_symbols: int = None,
                 scids: List[int] = None,
                 layers: List[int] = None,
                 dmrs_ports: List[int] = None) -> List[np.ndarray] | List[cp.ndarray]:
        """Run channel estimation for multiple UE groups.

        This runs the cuPHY channel estimation for all UE groups included in `pusch_configs`.
        If this argument is not given, all the other arguments need to be given and cuPHY
        channel estimation is run only for a single UE group sharing the same
        time-frequency resources, i.e. having the same PRB allocation, and the same start
        symbol and number of allocated symbols. This single UE group is the parameterized
        by the all other arguments.

        The method can be called using either Numpy or CuPy arrays. In case the input arrays
        are located on the GPU (CuPy), the output will be on the GPU (CuPy). So the return type
        shall be the same as used for `rx_slot` when calling the method.

        Args:
            rx_slot (Array): Input received data as a frequency x time x Rx
                antenna Numpy or CuPy array with type `complex64` entries.
            slot (int): Slot number.
            pusch_configs (List[PuschConfig]): List of PUSCH configuration objects, one per UE
                group. If this argument is given, the rest are ignored. If not given, all other
                arguments need to be given (only one UE group supported in that case).
            num_ues (int): Number of UEs in the single UE group.
            num_dmrs_cdm_grps_no_data (int): Number of DMRS CDM groups without data
                [3GPP TS 38.212, sec 7.3.1.1]. Value: 1->3.
            dmrs_scrm_id (int): DMRS scrambling ID.
            start_prb (int): Start PRB index of the UE allocation.
            num_prbs (int): Number of allocated PRBs for the UE group.
            prg_size (int): The Size of PRG in PRB for PUSCH per-PRG channel estimation.
            num_ul_streams (int): The number of active streams for this PUSCH.
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
                UE in the UE group. Value is 0 or 1.
            layers (List[int]): Number of layers for each UE in the UE group. The length of the
                list equals the number of UEs.
            dmrs_ports (List[int]): DMRS ports for each UE in the UE group. The format of each
                entry is in the SCF FAPI format as follows: A bitmap (mask) starting from the LSB
                where each bit indicates whether the corresponding DMRS port index is used.

        Returns:
            List[Array]: The channel estimates as a
            Rx ant x layer x frequency x time Numpy or CuPy array, per UE group.
        """
        cpu_copy = isinstance(rx_slot, np.ndarray)
        with self._cuda_stream:
            rx_slot = cp.array(rx_slot, order='F', dtype=cp.complex64)

        # If pusch_configs is given, use that directly. Else all the other parameters
        # need to be given (only a single UE group).
        if pusch_configs is None:
            pusch_configs = self._get_pusch_configs(
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
                dmrs_ports=dmrs_ports
            )

        # Wrap CuPy arrays into pycuphy types.
        rx_slot = pycuphy.CudaArrayComplexFloat(rx_slot)

        pusch_dyn_prms = _pusch_config_to_cuphy(
            cuda_stream=self._cuda_stream,
            rx_data=[rx_slot],
            slot=slot,
            pusch_configs=pusch_configs
        )
        self._pusch_params.set_dyn_prms(pusch_dyn_prms)
        channel_est = self._channel_estimator.estimate(self._pusch_params)

        with self._cuda_stream:
            channel_est = [cp.array(elem) for elem in channel_est]
            if cpu_copy:
                channel_est = [elem.get(order='F') for elem in channel_est]

        return channel_est

    def _get_pusch_configs(self,  # pylint: disable=too-many-arguments
                           *,
                           num_ues: int = None,
                           num_dmrs_cdm_grps_no_data: int = None,
                           dmrs_scrm_id: int = None,
                           start_prb: int = None,
                           num_prbs: int = None,
                           prg_size: int = None,
                           num_ul_streams: int = None,
                           dmrs_syms: List[int] = None,
                           dmrs_max_len: int = None,
                           dmrs_add_ln_pos: int = None,
                           start_sym: int = None,
                           num_symbols: int = None,
                           scids: List[int] = None,
                           layers: List[int] = None,
                           dmrs_ports: List[int] = None) -> List[PuschConfig]:
        """Helper to get PUSCH configs based on other parameters."""
        # All these parameters need to be set.
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
        if prg_size is None:
            raise ValueError("Argument prg_size is not set!")
        if num_ul_streams is None:
            raise ValueError("Argument num_ul_streams is not set!")
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

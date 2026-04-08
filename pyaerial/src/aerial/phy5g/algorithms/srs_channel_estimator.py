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

"""pyAerial library - SRS channel estimation."""
from typing import Generic
from typing import List
from typing import NamedTuple
from typing import Optional
from typing import Tuple
import warnings

import cupy as cp  # type: ignore
import numpy as np

from aerial.phy5g.api import Array
from aerial import pycuphy  # type: ignore
from aerial.pycuphy.chest_filters import CUPHY_CHEST_COEFF_FILE
from aerial.pycuphy.chest_filters import srs_chest_params_from_hdf5
from aerial.util.cuda import CudaStream


class SrsCellPrms(NamedTuple):
    """SRS cell parameters.

    A list of SRS cell parameters is given to the SRS channel estimator as input,
    one entry per cell.

    Args:
        slot_num (np.uint16): Slot number.
        frame_num (np.uint16): Frame number.
        srs_start_sym (np.uint8): SRS start symbol.
        num_srs_sym (np.uint8): Number of SRS symbols.
        num_rx_ant_srs (np.uint16): Number of SRS Rx antennas.
        mu (np.uint8): Subcarrier spacing parameter, see TS 38.211.
    """

    slot_num: np.uint16
    frame_num: np.uint16
    srs_start_sym: np.uint8
    num_srs_sym: np.uint8
    num_rx_ant_srs: np.uint16
    mu: np.uint8


class UeSrsPrms(NamedTuple):
    """UE SRS parameters.

    A list of UE SRS parameters is given to the SRS channel estimator as input,
    one entry per UE.

    Args:

        cell_idx (np.uint16): Index of cell user belongs to.
        num_ant_ports (np.uint8): Number of SRS antenna ports. 1,2, or 4.
        num_syms (np.uint8): Number of SRS symbols. 1,2, or 4.
        num_repetitions (np.uint8): Number of repititions. 1,2, or 4.
        comb_size (np.uint8): SRS comb size. 2 or 4.
        start_sym (np.uint8): Starting SRS symbol. 0 - 13.
        sequence_id (np.uint16): SRS sequence ID. 0 - 1023.
        config_idx (np.uint8): SRS bandwidth configuration idndex. 0 - 63.
        bandwidth_idx (np.uint8): SRS bandwidth index. 0 - 3.
        comb_offset (np.uint8): SRS comb offset. 0 - 3.
        cyclic_shift (np.uint8): Cyclic shift. 0 - 11.
        frequency_position (np.uint8): Frequency domain position. 0 - 67.
        frequency_shift (np.uint16): Frequency domain shift. 0 - 268.
        frequency_hopping (np.uint8): Freuqnecy hopping options. 0 - 3.
        resource_type (np.uint8): Type of SRS allocation. 0:
            Aperiodic. 1: Semi-persistent. 2: Periodic.
        periodicity (np.uint16): SRS periodicity in slots.
            0, 2, 3, 5, 8, 10, 16, 20, 32, 40, 64, 80, 160, 320, 640, 1280, 2560.
        offset (np.uint16): Slot offset value. 0 - 2569.
        group_or_sequence_hopping (np.uint8): Hopping configuration.
            0: No hopping. 1: Group hopping. 2: Sequence hopping.
        ch_est_buff_idx (np.uint16): Index of which buffer to store SRS estimates into.
        srs_ant_port_to_ue_ant_map (np.ndarray): Mapping between SRS antenna ports and UE antennas
            in channel estimation buffer: Store estimates for SRS antenna port i in
            srs_ant_port_to_ue_ant_map[i].
        prg_size (np.uint8): Number of PRBs per PRG.
    """

    cell_idx: np.uint16
    num_ant_ports: np.uint8
    num_syms: np.uint8
    num_repetitions: np.uint8
    comb_size: np.uint8
    start_sym: np.uint8
    sequence_id: np.uint16
    config_idx: np.uint8
    bandwidth_idx: np.uint8
    comb_offset: np.uint8
    cyclic_shift: np.uint8
    frequency_position: np.uint8
    frequency_shift: np.uint16
    frequency_hopping: np.uint8
    resource_type: np.uint8
    periodicity: np.uint16
    offset: np.uint16
    group_or_sequence_hopping: np.uint8
    ch_est_buff_idx: np.uint16
    srs_ant_port_to_ue_ant_map: np.ndarray
    prg_size: np.uint8


class SrsReport(NamedTuple):
    """SRS output report.

    This report is returned by the SRS channel estimator.

    Args:
        to_est_micro_sec (np.float32): Time offset estimate in microseconds.
        wideband_snr (np.float3): Wideband SNR.
        wideband_noise_energy (np.float32): Wideband noise energy.
        wideband_signal_energy (np.float32): Wideband signal energy.
        wideband_sc_corr (np.complex64): Wideband subcarrier correlation.
        wideband_cs_corr_ratio_db (np.float32):
        wideband_cs_corr_use (np.float32):
        wideband_cs_corr_not_use (np.float32):
    """
    to_est_micro_sec: np.float32
    wideband_snr: np.float32
    wideband_noise_energy: np.float32
    wideband_signal_energy: np.float32
    wideband_sc_corr: np.complex64
    wideband_cs_corr_ratio_db: np.float32
    wideband_cs_corr_use: np.float32
    wideband_cs_corr_not_use: np.float32


class SrsChannelEstimator(Generic[Array]):
    """SrsChannelEstimator class.

    This class implements SRS channel sounding for 5G NR.
    """
    def __init__(self,
                 *,
                 chest_algo_idx: int = None,
                 enable_delay_offset_correction: int = None,
                 chest_params: dict = None,
                 cuda_stream: Optional[CudaStream] = None) -> None:
        """Initialize SrsChannelEstimator.

        Args:
            chest_algo_idx (int) : ChEst algorithm index. 0: MMSE, 1: RKHS.
            chest_params (dict): Dictionary of channel estimation filters and parameters.
                Set to None to use defaults.
            cuda_stream (Optional[CudaStream]): CUDA stream. If not given, a new CudaStream is
                created. Use ``with stream:`` to scope work; call ``stream.synchronize()``
                explicitly when sync is needed.
        """
        warnings.warn("This class is deprecated. Use the full SRS Rx pipeline SrsRx instead.",
                      DeprecationWarning)

        if chest_algo_idx is None:
            chest_algo_idx = 0
        self.chest_algo_idx = chest_algo_idx

        if enable_delay_offset_correction is None:
            enable_delay_offset_correction = 1
        self.enable_delay_offset_correction = enable_delay_offset_correction

        self._cuda_stream = CudaStream() if cuda_stream is None else cuda_stream

        if chest_params is None:
            # Default SRS channel estimation parameters are loaded from the cuPHY filter file.
            chest_params = srs_chest_params_from_hdf5(CUPHY_CHEST_COEFF_FILE)

        self.channel_estimator = pycuphy.SrsChannelEstimator(chest_algo_idx,
                                                             enable_delay_offset_correction,
                                                             chest_params,
                                                             self._cuda_stream.handle)

    def estimate(self,
                 *,
                 rx_data: Array,
                 num_srs_ues: int,
                 num_srs_cells: int,
                 num_prb_grps: int,
                 start_prb_grp: int,
                 srs_cell_prms: List[SrsCellPrms],
                 srs_ue_prms: List[UeSrsPrms]) -> Tuple[list, Array, list]:
        """Run SRS channel estimation.

        Args:
            rx_data (Array): Input RX data, size num_subcarriers x num_srs_sym x num_rx_ant.
            num_srs_ues (int): Number of UEs.
            num_srs_cells (int): Number of SRS cells.
            num_prb_grps (int): Number of PRB groups.
            start_prb_grp (int): Start PRB group.
            srs_cell_prms (List[SrsCellPrms]): List of SRS cell parameters, one per cell.
            srs_ue_prms (List[UeSrsPrms]): List of UE SRS parameters, one per UE.

        Returns:
            List[Array], Array, List[SrsReport]: A tuple containing:

            - *List[Array]*:
              A list of channel estimates, one per UE. The channel estimate is a
              num_prb_grps x num_rx_ant x num_tx_ant numpy array.

            - *Array*:
              SNRs per RB per UE.

            - *List[SrsReport]*:
              A list of SRS wideband statistics reports, one per UE. Note: This gets copied to
              host memory always.
        """
        cpu_copy = isinstance(rx_data, np.ndarray)
        with self._cuda_stream:
            rx_data = cp.array(rx_data, order='F', dtype=cp.complex64)

        # Wrap CuPy array into pycuphy types.
        rx_data = pycuphy.CudaArrayComplexFloat(rx_data)

        ch_est = self.channel_estimator.estimate(
            rx_data,
            np.uint16(num_srs_ues),
            np.uint16(num_srs_cells),
            np.uint16(num_prb_grps),
            np.uint16(start_prb_grp),
            srs_cell_prms,
            srs_ue_prms
        )
        rb_snr_buffer = self.channel_estimator.get_rb_snr_buffer()
        srs_report = self.channel_estimator.get_srs_report()

        with self._cuda_stream:
            rb_snr_buffer = cp.array(rb_snr_buffer)
            ch_est = [cp.array(elem) for elem in ch_est]
            if cpu_copy:
                rb_snr_buffer = rb_snr_buffer.get()
                ch_est = [elem.get(order='F') for elem in ch_est]

        return ch_est, rb_snr_buffer, srs_report

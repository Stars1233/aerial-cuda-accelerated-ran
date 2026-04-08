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

"""pyAerial - SRS receiver pipeline class definition."""
from typing import Any
from typing import List
from typing import Optional

import cupy as cp  # type: ignore

from aerial import pycuphy  # type: ignore
from aerial.pycuphy.chest_filters import srs_chest_params_from_hdf5
from aerial.pycuphy.chest_filters import CUPHY_CHEST_COEFF_FILE
from aerial.util.cuda import CudaStream
from aerial.phy5g.api import Array
from aerial.phy5g.srs.srs_api import SrsRxPipeline
from aerial.phy5g.srs.srs_api import SrsRxConfig
from aerial.phy5g.srs.srs_api import SrsReport


class SrsRx(SrsRxPipeline[SrsRxConfig, SrsReport, Array]):
    """SRS receiver pipeline.

    This class implements the sounding reference signal reception pipeline.
    The SRS transmissions can be received from multiple cells with a single API call.
    """

    def __init__(self,
                 *,
                 num_rx_ant: List[int],
                 chest_algo_idx: int = 0,
                 enable_delay_offset_correction: int = 1,
                 chest_params: dict = None,
                 num_max_srs_ues: int = 192,
                 cuda_stream: Optional[CudaStream] = None) -> None:
        """Initialize SrsRx.

        Args:
            num_rx_ant (List[int]): Number of receive antennas per cell.
            chest_algo_idx (int): Channel estimation algorithm. Default: 0 (MMSE).

                - 0: MMSE
                - 1: RKHS

            enable_delay_offset_correction (int): Enable/disable delay offset correction.
                Default: 1 (enabled).

            chest_params (dict): Dictionary of channel estimation filters and parameters.
                Set to None to use defaults.
            num_max_srs_ues (int): Maximum number of SRS UEs. This number is used in memory
                allocations. Default: 192.
            cuda_stream (Optional[CudaStream]): CUDA stream. If not given, a new CudaStream is
                created. Use ``with stream:`` to scope work; call ``stream.synchronize()``
                explicitly when sync is needed.
        """
        self._cuda_stream = CudaStream() if cuda_stream is None else cuda_stream

        num_cells = len(num_rx_ant)

        if chest_params is None:
            # Default SRS channel estimation parameters.
            chest_params = srs_chest_params_from_hdf5(CUPHY_CHEST_COEFF_FILE)

        self.srs_rx = pycuphy.SrsRx(num_cells,
                                    num_rx_ant,
                                    chest_algo_idx,
                                    enable_delay_offset_correction,
                                    chest_params,
                                    num_max_srs_ues,
                                    self._cuda_stream.handle)

    def __call__(self,
                 rx_data: List[Array],
                 config: SrsRxConfig,
                 **kwargs: Any) -> List[SrsReport]:
        """Run SRS reception.

        Note: This implements the base class abstract method.

        Args:
            rx_data (List[Array]): Received data slot as an Array (Numpy or CuPy).
            config (SrsRxConfig): SRS reception configuration. See `SrsRxConfig`.

        Returns:
            List[SrsReport]: The SRS reports per UE, see `SrsReport`.
        """
        with self._cuda_stream:
            rx_data = [cp.array(elem, order='F', dtype=cp.complex64) for elem in rx_data]
            rx_data = [pycuphy.CudaArrayComplexFloat(elem) for elem in rx_data]

        ch_est = self.srs_rx.run(rx_data,
                                 config.srs_ue_configs,
                                 config.srs_cell_configs)

        ch_est_to_L2 = self.srs_rx.get_ch_est_to_L2()
        rb_snr_buffer = self.srs_rx.get_rb_snr_buffer()
        srs_report_cuphy = self.srs_rx.get_srs_report()

        # Create the UE SRS reports
        srs_reports = []
        num_ues = len(config.srs_ue_configs)
        for ue_idx in range(num_ues):
            srs_report = SrsReport(
                ch_est=ch_est[ue_idx],
                ch_est_to_L2=ch_est_to_L2[ue_idx],
                rb_snr=rb_snr_buffer[:, ue_idx],
                to_est_ms=srs_report_cuphy[ue_idx].to_est_ms,
                wideband_snr=srs_report_cuphy[ue_idx].wideband_snr,
                wideband_noise_energy=srs_report_cuphy[ue_idx].wideband_noise_energy,
                wideband_signal_energy=srs_report_cuphy[ue_idx].wideband_signal_energy,
                wideband_sc_corr=srs_report_cuphy[ue_idx].wideband_sc_corr,
                wideband_cs_corr_ratio_db=srs_report_cuphy[ue_idx].wideband_cs_corr_ratio_db,
                wideband_cs_corr_use=srs_report_cuphy[ue_idx].wideband_cs_corr_use,
                wideband_cs_corr_not_use=srs_report_cuphy[ue_idx].wideband_cs_corr_not_use,
                high_density_ant_port_flag=srs_report_cuphy[ue_idx].high_density_ant_port_flag
            )
            srs_reports.append(srs_report)

        return srs_reports

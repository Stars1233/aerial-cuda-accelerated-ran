# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""pyAerial library - statistical channel model.

This module provides the StatisticalChannel class for GPU-accelerated statistical
channel simulation supporting system-level simulations with large-scale fading,
path loss, shadowing, and small-scale fading.
"""

from typing import List, Optional
import numpy as np
import cupy as cp  # type: ignore

from aerial import pycuphy
from aerial.util.cuda import CudaStream
from aerial.pycuphy import (  # pylint: disable=no-name-in-module
    SimConfig,
    SystemLevelConfig,
    LinkLevelConfig,
    ExternalConfig,
)
from aerial.phy5g.api import Array


class StatisticalChannel:
    """GPU-accelerated statistical channel model for 5G NR system-level simulations.

    Implements a comprehensive channel model with:
    - Large-scale effects: path loss, shadowing, LOS/NLOS determination
    - Small-scale fading: TDL/CDL models with Doppler effects
    - Multi-cell, multi-user support with dynamic UE mobility

    Supports both CuPy arrays (zero-copy GPU operation) and NumPy arrays
    (automatic GPU transfer).

    Example:
        >>> from aerial.phy5g.channel_models import (
        ...     StatisticalChannel, SimConfig, SystemLevelConfig,
        ...     LinkLevelConfig, ExternalConfig
        ... )
        >>>
        >>> sim_cfg = SimConfig(...)
        >>> sys_cfg = SystemLevelConfig(...)
        >>> link_cfg = LinkLevelConfig(...)
        >>> ext_cfg = ExternalConfig(...)
        >>>
        >>> channel = StatisticalChannel(
        ...     sim_config=sim_cfg,
        ...     system_level_config=sys_cfg,
        ...     link_level_config=link_cfg,
        ...     external_config=ext_cfg
        ... )
        >>> channel.run(ref_time=0.0, active_cell=[0, 1], active_ut=[[0, 1], [2, 3]])

    Args:
        sim_config: Simulation configuration (frequency, bandwidth, FFT size, etc.).
        system_level_config: System-level parameters (scenario, path loss, shadowing).
        link_level_config: Link-level parameters (fading type, delay profile, mobility).
        external_config: External configuration (cells, UTs, antenna panels).
        cuda_stream: CUDA stream for GPU operations. If None, a new CudaStream is
            created. Use ``with stream:`` to scope work; call ``stream.synchronize()``
            explicitly when sync is needed. Default: None.
        rand_seed: Random seed for channel generation. Default: 0.
    """

    def __init__(
        self,
        *,
        sim_config: SimConfig,
        system_level_config: SystemLevelConfig,
        link_level_config: LinkLevelConfig,
        external_config: ExternalConfig,
        cuda_stream: Optional[CudaStream] = None,
        rand_seed: int = 0
    ) -> None:
        """Initialize the statistical channel model.

        Args:
            sim_config: Simulation configuration (frequency, bandwidth, FFT size, etc.).
            system_level_config: System-level parameters (scenario, path loss, shadowing).
            link_level_config: Link-level parameters (fading type, delay profile, mobility).
            external_config: External configuration (cells, UTs, antenna panels).
            cuda_stream: CUDA stream. If None, a new CudaStream is created.
            rand_seed: Random seed for channel generation.
        """
        # CUDA stream management
        self._stream = CudaStream() if cuda_stream is None else cuda_stream

        # Store configuration
        self.sim_config = sim_config
        self.system_level_config = system_level_config
        self.link_level_config = link_level_config
        self.external_config = external_config

        self._impl = pycuphy.StatisChanModel(
            sim_config,
            system_level_config,
            link_level_config,
            external_config,
            rand_seed,
            self._stream.handle
        )

    def _convert_array_list(
        self,
        arrays: Optional[List[Array]]
    ) -> Optional[List[cp.ndarray]]:
        """Convert list of arrays to CuPy arrays if needed."""
        if arrays is None:
            return None
        return [
            cp.asarray(arr) if isinstance(arr, np.ndarray) else arr
            for arr in arrays
        ]

    def run(  # pylint: disable=too-many-arguments
        self,
        *,
        ref_time: float = 0.0,
        continuous_fading: int = 1,
        active_cell: Optional[List[int]] = None,
        active_ut: Optional[List[List[int]]] = None,
        ut_new_loc: Optional[Array] = None,
        ut_new_velocity: Optional[Array] = None,
        cir_coe: Optional[List[Array]] = None,
        cir_norm_delay: Optional[List[Array]] = None,
        cir_n_taps: Optional[List[Array]] = None,
        cfr_sc: Optional[List[Array]] = None,
        cfr_prbg: Optional[List[Array]] = None
    ) -> None:
        """Run channel simulation for current TTI.

        Generates channel coefficients based on current UE positions and velocities.
        Results are written to the provided output arrays.

        Args:
            ref_time: Reference time for CIR generation (typically tti_idx * slot_duration).
            continuous_fading: Fading mode. 0 = discontinuous (regenerate every TTI),
                1 = continuous (maintain time correlation).
            active_cell: List of active cell IDs.
            active_ut: List of active UT lists per sector. Each element is a list of
                active UT indices for that sector.
            ut_new_loc: New UT locations array of shape [n_ut, 3] to update positions.
            ut_new_velocity: New UT velocity array of shape [n_ut, 3] to update velocities.
            cir_coe: Output CIR coefficients. List of arrays per sector, each with shape
                [n_active_ut, n_snapshot, n_ut_ant, n_bs_ant, 24].
            cir_norm_delay: Output normalized CIR delays. List of arrays per sector,
                each with shape [n_active_ut, 24].
            cir_n_taps: Output number of CIR taps. List of arrays per sector,
                each with shape [n_active_ut].
            cfr_sc: Output CFR per subcarrier. List of arrays per sector, each with
                shape [n_active_ut, n_snapshot, n_ut_ant, n_bs_ant, fft_size].
            cfr_prbg: Output CFR per PRB group. List of arrays per sector, each with
                shape [n_active_ut, n_snapshot, n_ut_ant, n_bs_ant, n_prbg].
        """
        # Convert location/velocity arrays if needed, in the correct CUDA context
        with self._stream:
            ut_new_loc_gpu = None
            if ut_new_loc is not None:
                ut_new_loc_gpu = (
                    cp.asarray(ut_new_loc) if isinstance(ut_new_loc, np.ndarray)
                    else ut_new_loc
                )

            ut_new_velocity_gpu = None
            if ut_new_velocity is not None:
                ut_new_velocity_gpu = (
                    cp.asarray(ut_new_velocity) if isinstance(ut_new_velocity, np.ndarray)
                    else ut_new_velocity
                )

        # Call C++ implementation
        self._impl.run(
            ref_time=ref_time,
            continuous_fading=continuous_fading,
            active_cell=active_cell,
            active_ut=active_ut,
            ut_new_loc=ut_new_loc_gpu,
            ut_new_velocity=ut_new_velocity_gpu,
            cir_coe=cir_coe,
            cir_norm_delay=cir_norm_delay,
            cir_n_taps=cir_n_taps,
            cfr_sc=cfr_sc,
            cfr_prbg=cfr_prbg
        )

    def __call__(  # pylint: disable=too-many-arguments
        self,
        *,
        ref_time: float = 0.0,
        continuous_fading: int = 1,
        active_cell: Optional[List[int]] = None,
        active_ut: Optional[List[List[int]]] = None,
        ut_new_loc: Optional[Array] = None,
        ut_new_velocity: Optional[Array] = None,
        cir_coe: Optional[List[Array]] = None,
        cir_norm_delay: Optional[List[Array]] = None,
        cir_n_taps: Optional[List[Array]] = None,
        cfr_sc: Optional[List[Array]] = None,
        cfr_prbg: Optional[List[Array]] = None
    ) -> None:
        """Alias for run(). See run() for documentation."""
        return self.run(
            ref_time=ref_time,
            continuous_fading=continuous_fading,
            active_cell=active_cell,
            active_ut=active_ut,
            ut_new_loc=ut_new_loc,
            ut_new_velocity=ut_new_velocity,
            cir_coe=cir_coe,
            cir_norm_delay=cir_norm_delay,
            cir_n_taps=cir_n_taps,
            cfr_sc=cfr_sc,
            cfr_prbg=cfr_prbg
        )

    def reset(self) -> None:
        """Reset channel model state.

        Reinitializes the internal channel state. Call this when starting a new
        simulation or when channel coherence time has been exceeded.
        """
        self._impl.reset()

    def get_cir(
        self,
        *,
        cir_coe: Optional[List[Array]] = None,
        cir_norm_delay: Optional[List[Array]] = None,
        cir_n_taps: Optional[List[Array]] = None
    ) -> None:
        """Get Channel Impulse Response (CIR) data.

        Retrieves the CIR coefficients, delays, and number of taps from the
        last run() call.

        Args:
            cir_coe: Output CIR coefficients. List of arrays per sector, each with
                shape [n_active_ut, n_snapshot, n_ut_ant, n_bs_ant, 24].
            cir_norm_delay: Output normalized CIR delays. List of arrays per sector,
                each with shape [n_active_ut, 24].
            cir_n_taps: Output number of CIR taps. List of arrays per sector,
                each with shape [n_active_ut].
        """
        self._impl.get_cir(cir_coe, cir_norm_delay, cir_n_taps)

    def get_cfr(
        self,
        *,
        cfr_sc: Optional[List[Array]] = None,
        cfr_prbg: Optional[List[Array]] = None
    ) -> None:
        """Get Channel Frequency Response (CFR) data.

        Retrieves the CFR per subcarrier or per PRB group from the last run() call.

        Args:
            cfr_sc: Output CFR per subcarrier. List of arrays per sector, each with
                shape [n_active_ut, n_snapshot, n_ut_ant, n_bs_ant, fft_size].
            cfr_prbg: Output CFR per PRB group. List of arrays per sector, each with
                shape [n_active_ut, n_snapshot, n_ut_ant, n_bs_ant, n_prbg].
        """
        self._impl.get_cfr(cfr_sc, cfr_prbg)

    def dump_los_nlos_stats(
        self,
        los_nlos_stats: Optional[Array] = None
    ) -> None:
        """Dump LOS/NLOS statistics.

        Args:
            los_nlos_stats: Output array for LOS/NLOS statistics with shape
                [n_sector, n_ut].
        """
        self._impl.dump_los_nlos_stats(los_nlos_stats)

    def dump_pl_sf_stats(
        self,
        pl_sf: Array,
        active_cell: Optional[Array] = None,
        active_ut: Optional[Array] = None
    ) -> None:
        """Dump pathloss and shadowing statistics.

        Values are total loss = -(pathloss - shadow_fading) in dB. The sign of
        shadow fading is defined so that positive SF means more received power
        at UT than predicted by the path loss model.

        Args:
            pl_sf: Output array for pathloss and shadowing (required).
                Shape depends on active_cell/active_ut:
                - If both provided: [len(active_cell), len(active_ut)]
                - If one is None: uses n_sector*n_site or n_ut for that dimension
            active_cell: Optional array of active cell IDs.
            active_ut: Optional array of active UT IDs.
        """
        self._impl.dump_pl_sf_stats(
            pl_sf, active_cell, active_ut
        )

    def dump_pl_sf_ant_gain_stats(
        self,
        pl_sf_ant_gain: Array,
        active_cell: Optional[Array] = None,
        active_ut: Optional[Array] = None
    ) -> None:
        """Dump pathloss, shadowing and antenna gain statistics.

        Values are total channel gain in dB = antGain - pathloss + SF
        antGain is per antenna element only (no array gain); downstream may add array/beamforming gain.

        Args:
            pl_sf_ant_gain: Output array for pathloss, shadowing and antenna gain.
                Same shape rules as dump_pl_sf_stats.
            active_cell: Optional array of active cell IDs.
            active_ut: Optional array of active UT IDs.
        """
        self._impl.dump_pl_sf_ant_gain_stats(
            pl_sf_ant_gain, active_cell, active_ut
        )

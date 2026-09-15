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

"""gpu3gppchan — statistical channel model.

This module provides the StatisticalChannel class for GPU-accelerated statistical
channel simulation supporting system-level simulations with large-scale fading,
path loss, shadowing, and small-scale fading.
"""

from typing import List, Optional
import numpy as np
import cupy as cp  # type: ignore

from . import _gpu3gppchan as _native
from .cuda_utils import Array, CudaStream
from ._gpu3gppchan import (  # pylint: disable=no-name-in-module
    SimConfig,
    SystemLevelConfig,
    LinkLevelConfig,
    ExternalConfig,
)


def _as_host_xyz_array(value: Optional[Array], name: str) -> Optional[np.ndarray]:
    """Return location or velocity metadata as a host float32 ``[N, 3]`` array."""
    if value is None:
        return None

    if isinstance(value, cp.ndarray):
        value = cp.asnumpy(value)

    array = np.asarray(value, dtype=np.float32, order="C")
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError(f"{name} must have shape (N, 3)")
    return array


class StatisticalChannel:
    """GPU-accelerated statistical channel model for 5G NR system-level simulations.

    Implements a comprehensive channel model with:
    - Large-scale effects: path loss, shadowing, LOS/NLOS determination
    - Small-scale fading: TDL/CDL models with Doppler effects
    - Multi-cell, multi-user support with dynamic UE mobility

    Supports both CuPy arrays (zero-copy GPU operation) and NumPy arrays
    (automatic GPU transfer).

    Example:
        >>> from gpu3gppchan import (
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

        self._impl = _native.StatisChanModel(
            sim_config,
            system_level_config,
            link_level_config,
            external_config,
            rand_seed,
            self._stream
        )

    def _convert_array_list(
        self,
        arrays: Optional[List[Array]]
    ) -> Optional[List[Array]]:
        """Return buffers whose memory space matches the native channel mode."""
        if arrays is None or self.sim_config.cpu_only_mode != 0:
            return arrays
        return [
            cp.asarray(arr) if isinstance(arr, np.ndarray) else arr
            for arr in arrays
        ]

    @staticmethod
    def _copy_device_outputs_to_host(
        original: Optional[List[Array]],
        converted: Optional[List[Array]],
    ) -> None:
        """Copy converted GPU output buffers back to caller-owned NumPy arrays."""
        if original is None or converted is None:
            return
        for host_array, native_array in zip(original, converted):
            if isinstance(host_array, np.ndarray) and isinstance(
                native_array, cp.ndarray
            ):
                np.copyto(host_array, cp.asnumpy(native_array))

    def run(  # pylint: disable=too-many-arguments
        self,
        *,
        ref_time: float = 0.0,
        continuous_fading: int = 1,
        active_cell: Optional[List[int]] = None,
        active_ut: Optional[List[List[int]]] = None,
        ut_new_loc: Optional[Array] = None,
        ut_new_velocity: Optional[Array] = None,
        ut_update_stream: Optional[CudaStream] = None,
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
            ut_update_stream: Stream that produced the mobility updates. When supplied,
                host conversion waits on that stream before native execution.
            cir_coe: Output CIR coefficients. List of arrays per sector, each with shape
                [n_active_ut, n_snapshot, n_ut_ant, n_bs_ant, max_taps], where
                max_taps is the caller-allocated effective CIR capacity.
            cir_norm_delay: Output normalized CIR delays. List of arrays per sector,
                each with shape [n_active_ut, max_taps].
            cir_n_taps: Output number of CIR taps. List of arrays per sector,
                each with shape [n_active_ut].
            cfr_sc: Output CFR per subcarrier. List of arrays per sector, each with
                shape [n_active_ut, n_snapshot, n_ut_ant, n_bs_ant, fft_size].
            cfr_prbg: Output CFR per PRB group. List of arrays per sector, each with
                shape [n_active_ut, n_snapshot, n_ut_ant, n_bs_ant, n_prbg].

        Returns:
            None. Output arrays are populated in place.

        Raises:
            ValueError: If an output buffer has an incompatible shape, dtype, or
                configured memory location.

        Examples:
            Run one TTI using the model's default active-cell and active-UT sets::

                channel.run(ref_time=0.0)
        """
        # Locations and velocities are host-side metadata consumed by the native
        # wrapper. Convert CuPy inputs back to host arrays before calling it.
        if ut_update_stream is not None:
            ut_update_stream.synchronize()
        with self._stream:
            ut_new_loc_host = _as_host_xyz_array(ut_new_loc, "ut_new_loc")
            ut_new_velocity_host = _as_host_xyz_array(
                ut_new_velocity, "ut_new_velocity"
            )
            cir_coe_native = self._convert_array_list(cir_coe)
            cir_norm_delay_native = self._convert_array_list(cir_norm_delay)
            cir_n_taps_native = self._convert_array_list(cir_n_taps)
            cfr_sc_native = self._convert_array_list(cfr_sc)
            cfr_prbg_native = self._convert_array_list(cfr_prbg)

        # Call C++ implementation
        self._impl.run(
            ref_time=ref_time,
            continuous_fading=continuous_fading,
            active_cell=active_cell,
            active_ut=active_ut,
            ut_new_loc=ut_new_loc_host,
            ut_new_velocity=ut_new_velocity_host,
            cir_coe=cir_coe_native,
            cir_norm_delay=cir_norm_delay_native,
            cir_n_taps=cir_n_taps_native,
            cfr_sc=cfr_sc_native,
            cfr_prbg=cfr_prbg_native
        )

        # The native model retains raw output pointers for later HDF export.
        # Retain the corresponding Python arrays for as long as this wrapper lives.
        self._external_output_owners = tuple(
            array
            for arrays in (
                cir_coe_native,
                cir_norm_delay_native,
                cir_n_taps_native,
                cfr_sc_native,
                cfr_prbg_native,
            )
            if arrays is not None
            for array in arrays
        )

        if self.sim_config.cpu_only_mode == 0:
            with self._stream:
                self._copy_device_outputs_to_host(cir_coe, cir_coe_native)
                self._copy_device_outputs_to_host(
                    cir_norm_delay, cir_norm_delay_native
                )
                self._copy_device_outputs_to_host(
                    cir_n_taps, cir_n_taps_native
                )
                self._copy_device_outputs_to_host(cfr_sc, cfr_sc_native)
                self._copy_device_outputs_to_host(cfr_prbg, cfr_prbg_native)

    def __call__(  # pylint: disable=too-many-arguments
        self,
        *,
        ref_time: float = 0.0,
        continuous_fading: int = 1,
        active_cell: Optional[List[int]] = None,
        active_ut: Optional[List[List[int]]] = None,
        ut_new_loc: Optional[Array] = None,
        ut_new_velocity: Optional[Array] = None,
        ut_update_stream: Optional[CudaStream] = None,
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
            ut_update_stream=ut_update_stream,
            cir_coe=cir_coe,
            cir_norm_delay=cir_norm_delay,
            cir_n_taps=cir_n_taps,
            cfr_sc=cfr_sc,
            cfr_prbg=cfr_prbg
        )

    def set_los_override(self, los_ind: Array) -> None:
        """Override LOS/NLOS state for each site-UT link.

        Args:
            los_ind: One integer per site-UT link. Use 1 for LOS, 0 for NLOS,
                and 255 to retain the model's probability draw for that link.

        Raises:
            ValueError: If the input is not one-dimensional or contains values
                other than 0, 1, and 255. The native binding also verifies that
                the array length matches the model's site-UT link count.

        Returns:
            None.

        Examples:
            Apply explicit LOS/NLOS overrides::

                channel.set_los_override(np.array([1, 0, 255], dtype=np.uint8))
        """
        values = (
            cp.asnumpy(los_ind)
            if isinstance(los_ind, cp.ndarray)
            else np.asarray(los_ind)
        )
        if values.ndim != 1:
            raise ValueError("los_ind must be a one-dimensional array")
        if not np.issubdtype(values.dtype, np.integer):
            raise ValueError("los_ind must contain integer values")
        if not np.all(np.isin(values, (0, 1, 255))):
            raise ValueError(
                "los_ind entries must be 0 (NLOS), 1 (LOS), or 255 (model draw)"
            )
        self._impl.set_los_override(
            np.ascontiguousarray(values, dtype=np.uint8)
        )

    def clear_los_override(self) -> None:
        """Restore probability-based LOS/NLOS selection for every link.

        Args:
            None.

        Returns:
            None.

        Raises:
            RuntimeError: If the native model rejects the clear request.

        Examples:
            Remove all explicit LOS/NLOS overrides::

                channel.clear_los_override()
        """
        self._impl.clear_los_override()

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
        """Copy internally stored CIR data into caller-owned output arrays.

        Requires ``internal_memory_mode`` 1 or 2 and a preceding ``run()``
        with explicit ``active_ut`` selectors.

        Args:
            cir_coe: Optional per-active-cell CIR coefficient output arrays.
            cir_norm_delay: Optional per-active-cell normalized-delay output arrays.
            cir_n_taps: Optional per-active-cell tap-count output arrays.

        Returns:
            None. Provided output arrays are populated in place.

        Raises:
            ValueError: If internal CIR storage is unavailable or an output buffer
                does not satisfy the native buffer contract.

        Examples:
            Copy the CIR into caller-owned arrays after ``run()``::

                channel.get_cir(cir_coe=cir_coe, cir_norm_delay=cir_delay,
                                cir_n_taps=cir_n_taps)
        """
        with self._stream:
            cir_coe_native = self._convert_array_list(cir_coe)
            cir_norm_delay_native = self._convert_array_list(cir_norm_delay)
            cir_n_taps_native = self._convert_array_list(cir_n_taps)
            self._impl.get_cir(
                cir_coe_native,
                cir_norm_delay_native,
                cir_n_taps_native,
            )
            if self.sim_config.cpu_only_mode == 0:
                self._copy_device_outputs_to_host(cir_coe, cir_coe_native)
                self._copy_device_outputs_to_host(
                    cir_norm_delay, cir_norm_delay_native
                )
                self._copy_device_outputs_to_host(cir_n_taps, cir_n_taps_native)

    def get_cfr(
        self,
        *,
        cfr_sc: Optional[List[Array]] = None,
        cfr_prbg: Optional[List[Array]] = None
    ) -> None:
        """Copy internally stored CFR data into caller-owned output arrays.

        Requires ``internal_memory_mode`` 2 and a preceding ``run()`` with
        explicit ``active_ut`` selectors.

        Args:
            cfr_sc: Optional per-active-cell subcarrier-CFR output arrays.
            cfr_prbg: Optional per-active-cell PRBG-CFR output arrays.

        Returns:
            None. Provided output arrays are populated in place.

        Raises:
            ValueError: If internal CFR storage is unavailable or an output buffer
                does not satisfy the native buffer contract.

        Examples:
            Copy the CFR into caller-owned arrays after ``run()``::

                channel.get_cfr(cfr_sc=cfr_sc, cfr_prbg=cfr_prbg)
        """
        with self._stream:
            cfr_sc_native = self._convert_array_list(cfr_sc)
            cfr_prbg_native = self._convert_array_list(cfr_prbg)
            self._impl.get_cfr(cfr_sc_native, cfr_prbg_native)
            if self.sim_config.cpu_only_mode == 0:
                self._copy_device_outputs_to_host(cfr_sc, cfr_sc_native)
                self._copy_device_outputs_to_host(cfr_prbg, cfr_prbg_native)

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

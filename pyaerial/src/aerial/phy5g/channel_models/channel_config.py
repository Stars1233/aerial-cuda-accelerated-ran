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

"""pyAerial library - channel models API.

This module provides configuration classes for fading channel models:
- FadingChannelConfig: Base configuration for all fading channels
- TdlChannelConfig: TDL (Tapped Delay Line) channel configuration
- CdlChannelConfig: CDL (Clustered Delay Line) channel configuration

It also includes configuration classes for the statistical channel model (StatisChanModel).
"""

# pylint: disable=no-member,too-many-positional-arguments
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple, TypeVar, Union
import cupy as cp  # type: ignore
import numpy as np
from aerial import pycuphy


Array = TypeVar("Array", np.ndarray, cp.ndarray)


# =============================================================================
# Fading Channel Configuration Classes
# =============================================================================

@dataclass
class FadingChannelConfig(ABC):
    """Abstract base configuration for fading channel models.

    This is an abstract class containing parameters common to all fading channel types.
    Do not instantiate directly - use TdlChannelConfig or CdlChannelConfig instead.

    Args:
        delay_profile (str): 3GPP delay profile identifier [3GPP TR 38.901, Sec 7.7.2].
            Values: 'A', 'B', 'C', 'D', 'E'. Default: 'A'.
        delay_spread (float): RMS delay spread in nanoseconds [3GPP TR 38.901, Table 7.7.3-1].
            Typical values: 30 (short), 100 (nominal), 300 (long). Default: 30.
        max_doppler_shift (float): Maximum Doppler frequency shift in Hz. Determined by
            UE velocity (v) and carrier frequency (fc) as fd = v * fc / c.
            Default: 5.0.
        n_cell (int): Number of cells in the simulation. Default: 1.
        n_ue (int): Number of UEs per cell. Default: 1.
        cfo_hz (float): Carrier frequency offset in Hz. Models oscillator mismatch between
            transmitter and receiver. Default: 0.0.
        delay (float): Propagation delay in seconds. Default: 0.0.
        save_ant_pair_sample (int): Save per antenna pair samples for debugging:

            - 0: Disabled (default).
            - 1: Enabled.

        rand_seed (int): Random seed for reproducible channel generation. Default: 0.
    """
    delay_profile: str = 'A'
    delay_spread: float = 30.0
    max_doppler_shift: float = 5.0
    n_cell: int = 1
    n_ue: int = 1
    cfo_hz: float = 0.0
    delay: float = 0.0
    save_ant_pair_sample: int = 0
    rand_seed: int = 0

    @abstractmethod
    def _to_pycuphy(
        self, carrier_f_samp: float, carrier_n_sc: int,
        carrier_n_sc_prbg: int, carrier_sc_spacing_hz: float,
        carrier_n_symbol_slot: int
    ) -> Union[pycuphy.TdlConfig, pycuphy.CdlConfig]:  # type: ignore[name-defined]
        """Convert to pycuphy config object. Must be implemented by subclasses."""
        pass


@dataclass
class TdlChannelConfig(FadingChannelConfig):
    """TDL (Tapped Delay Line) channel configuration.

    TDL models are used for link-level simulations with simplified spatial characteristics.
    Suitable for SISO or basic MIMO without antenna array geometry
    [3GPP TR 38.901, Sec 7.7.2].

    The TDL model uses a tapped delay line structure where each tap represents a
    multipath component with specific delay and power. The model does not include
    spatial correlation between antenna elements.

    Example:
        >>> config = TdlChannelConfig(
        ...     delay_profile='A',
        ...     delay_spread=30.0,
        ...     max_doppler_shift=5.0,
        ...     n_bs_ant=4,
        ...     n_ue_ant=2
        ... )

    Args:
        n_bs_ant (int): Number of base station antennas. Default: 1.
        n_ue_ant (int): Number of UE antennas. Default: 1.
        n_path (int): Number of delay taps in the channel model. The number of taps
            depends on the delay profile (e.g., TDL-A has 23 taps). Default: 23.
        use_simplified_pdp (bool): Use simplified power delay profile with fewer taps
            for faster computation at the cost of reduced accuracy. Default: False.
    """
    n_bs_ant: int = 1
    n_ue_ant: int = 1
    n_path: int = 23
    use_simplified_pdp: bool = False

    def _to_pycuphy(
        self, carrier_f_samp: float, carrier_n_sc: int,
        carrier_n_sc_prbg: int, carrier_sc_spacing_hz: float,
        carrier_n_symbol_slot: int
    ) -> pycuphy.TdlConfig:  # type: ignore[name-defined]
        """Convert to pycuphy.TdlConfig for internal use."""
        cfg = pycuphy.TdlConfig()
        cfg.delay_profile = self.delay_profile.upper()
        cfg.delay_spread = self.delay_spread
        cfg.max_doppler_shift = self.max_doppler_shift
        cfg.n_cell = self.n_cell
        cfg.n_ue = self.n_ue
        cfg.n_bs_ant = self.n_bs_ant
        cfg.n_ue_ant = self.n_ue_ant
        cfg.n_sc = carrier_n_sc
        cfg.n_sc_prbg = carrier_n_sc_prbg
        cfg.sc_spacing_hz = carrier_sc_spacing_hz
        cfg.cfo_hz = self.cfo_hz
        cfg.delay = self.delay
        cfg.n_path = self.n_path
        cfg.use_simplified_pdp = int(self.use_simplified_pdp)
        cfg.save_ant_pair_sample = self.save_ant_pair_sample
        cfg.f_samp = int(carrier_f_samp)
        cfg.f_batch = carrier_n_symbol_slot
        cfg.run_mode = 2  # Required for frequency domain processing
        cfg.freq_convert_type = 0
        cfg.sc_sampling = 1
        return cfg


@dataclass
class CdlChannelConfig(FadingChannelConfig):
    """CDL (Clustered Delay Line) channel configuration.

    CDL models include full spatial characteristics with antenna array geometry and
    angular spreads. Suitable for MIMO simulations requiring accurate spatial correlation
    [3GPP TR 38.901, Sec 7.7.1].

    The CDL model extends TDL by adding angular information (arrival and departure angles)
    for each cluster, enabling accurate modeling of antenna array responses and spatial
    correlation.

    Example:
        >>> config = CdlChannelConfig(
        ...     delay_profile='A',
        ...     delay_spread=30.0,
        ...     bs_ant_size=(1, 1, 8, 4, 2),  # (M_g,N_g,M,N,P) = 64 antennas
        ...     ue_ant_size=(1, 1, 1, 1, 2)   # (M_g,N_g,M,N,P) = 2 antennas
        ... )

    Args:
        bs_ant_size (Tuple[int, int, int, int, int]): Base station antenna array dimensions
            as (M_g, N_g, M, N, P). Total antennas = M_g * N_g * M * N * P.
            Backward compatible input: 3-tuple (M, N, P) is accepted and normalized
            to (1, 1, M, N, P).
            Default: (1, 1, 1, 2, 2) = 4 antennas.
        bs_ant_spacing (Tuple[float, float, float, float]): BS antenna spacing in wavelengths
            as (d_g_h, d_g_v, d_h, d_v). Must be a 4-tuple.
            Default: (1.0, 1.0, 0.5, 0.5).
        bs_ant_polar_angles (Tuple[float, float]): BS antenna polarization slant angles
            in degrees for dual-polarized arrays. Default: (45.0, -45.0) for cross-polarized.
        bs_ant_pattern (int): BS antenna element radiation pattern type:

            - 0: Isotropic pattern (default).
            - 1: 3GPP directional pattern [3GPP TR 38.901, Table 7.3-1].

        ue_ant_size (Tuple[int, int, int, int, int]): UE antenna array dimensions
            as (M_g, N_g, M, N, P). Backward compatible input: 3-tuple (M, N, P)
            is accepted and normalized to (1, 1, M, N, P). Default: (1, 1, 2, 2, 1) = 4 antennas.
        ue_ant_spacing (Tuple[float, float, float, float]): UE antenna spacing in wavelengths
            as (d_g_h, d_g_v, d_h, d_v). Must be a 4-tuple.
            Default: (1.0, 1.0, 0.5, 0.5).
        ue_ant_polar_angles (Tuple[float, float]): UE antenna polarization slant angles
            in degrees. Default: (0.0, 90.0) for vertical/horizontal polarization.
        ue_ant_pattern (int): UE antenna element radiation pattern type. Default: 0.
        n_ray (int): Number of rays per cluster for angular spread modeling.
            Default: 20.
        v_direction (Tuple[float, float, float]): UE velocity direction vector as
            (azimuth_deg, elevation_deg, speed_m_s) for Doppler calculation.
            Default: (0.0, 0.0, 0.0).
    """
    bs_ant_size: Tuple[int, int, int, int, int] = (1, 1, 1, 2, 2)
    bs_ant_spacing: Tuple[float, float, float, float] = (1.0, 1.0, 0.5, 0.5)
    bs_ant_polar_angles: Tuple[float, float] = (45.0, -45.0)
    bs_ant_pattern: int = 0
    ue_ant_size: Tuple[int, int, int, int, int] = (1, 1, 2, 2, 1)
    ue_ant_spacing: Tuple[float, float, float, float] = (1.0, 1.0, 0.5, 0.5)
    ue_ant_polar_angles: Tuple[float, float] = (0.0, 90.0)
    ue_ant_pattern: int = 0
    n_ray: int = 20
    v_direction: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    @staticmethod
    def _normalize_ant_size(name: str, value: Tuple[int, ...]) -> Tuple[int, int, int, int, int]:
        """Normalize antenna-size input to 5 positive integers."""
        if len(value) == 3:
            normalized = (1, 1, int(value[0]), int(value[1]), int(value[2]))
        elif len(value) == 5:
            normalized = tuple(int(v) for v in value)  # type: ignore[assignment]
        else:
            raise ValueError(
                f"{name} must have 3 or 5 elements; got {len(value)}"
            )
        if any(v <= 0 for v in normalized):
            raise ValueError(f"{name} elements must be positive integers")
        return normalized

    @staticmethod
    def _normalize_ant_spacing(name: str, value: Tuple[float, ...]) -> Tuple[float, float, float, float]:
        """Validate antenna-spacing input as 4 positive floats. Accepts 2 (d_H, d_V) or 4 elements."""
        if len(value) == 2:
            normalized = (1.0, 1.0, float(value[0]), float(value[1]))
        elif len(value) == 4:
            normalized = tuple(float(v) for v in value)  # type: ignore[assignment]
        else:
            raise ValueError(
                f"{name} must have 2 or 4 elements; got {len(value)}"
            )
        if any(v <= 0.0 for v in normalized):
            raise ValueError(f"{name} elements must be positive floats")
        return normalized

    def __post_init__(self) -> None:
        """Normalize and validate antenna size/spacing fields after init."""
        self.bs_ant_size = self._normalize_ant_size("bs_ant_size", self.bs_ant_size)
        self.ue_ant_size = self._normalize_ant_size("ue_ant_size", self.ue_ant_size)
        self.bs_ant_spacing = self._normalize_ant_spacing("bs_ant_spacing", self.bs_ant_spacing)
        self.ue_ant_spacing = self._normalize_ant_spacing("ue_ant_spacing", self.ue_ant_spacing)

    @property
    def n_bs_ant(self) -> int:
        """Total number of base station antennas."""
        return (
            self.bs_ant_size[0]
            * self.bs_ant_size[1]
            * self.bs_ant_size[2]
            * self.bs_ant_size[3]
            * self.bs_ant_size[4]
        )

    @property
    def n_ue_ant(self) -> int:
        """Total number of UE antennas."""
        return (
            self.ue_ant_size[0]
            * self.ue_ant_size[1]
            * self.ue_ant_size[2]
            * self.ue_ant_size[3]
            * self.ue_ant_size[4]
        )

    def _to_pycuphy(
        self, carrier_f_samp: float, carrier_n_sc: int,
        carrier_n_sc_prbg: int, carrier_sc_spacing_hz: float,
        carrier_n_symbol_slot: int
    ) -> pycuphy.CdlConfig:  # type: ignore[name-defined]
        """Convert to pycuphy.CdlConfig for internal use."""
        cfg = pycuphy.CdlConfig()
        cfg.delay_profile = self.delay_profile.upper()
        cfg.delay_spread = self.delay_spread
        cfg.max_doppler_shift = self.max_doppler_shift
        cfg.n_cell = self.n_cell
        cfg.n_ue = self.n_ue
        cfg.bs_ant_size = list(self.bs_ant_size)
        cfg.bs_ant_spacing = self.bs_ant_spacing
        cfg.bs_ant_polar_angles = self.bs_ant_polar_angles
        cfg.bs_ant_pattern = self.bs_ant_pattern
        cfg.ue_ant_size = list(self.ue_ant_size)
        cfg.ue_ant_spacing = self.ue_ant_spacing
        cfg.ue_ant_polar_angles = self.ue_ant_polar_angles
        cfg.ue_ant_pattern = self.ue_ant_pattern
        cfg.n_sc = carrier_n_sc
        cfg.n_sc_prbg = carrier_n_sc_prbg
        cfg.sc_spacing_hz = carrier_sc_spacing_hz
        cfg.cfo_hz = self.cfo_hz
        cfg.delay = self.delay
        cfg.n_ray = self.n_ray
        cfg.v_direction = self.v_direction
        cfg.save_ant_pair_sample = self.save_ant_pair_sample
        cfg.f_samp = int(carrier_f_samp)
        cfg.f_batch = carrier_n_symbol_slot
        cfg.run_mode = 2  # Required for frequency domain processing
        cfg.freq_convert_type = 0
        cfg.sc_sampling = 1
        return cfg


# =============================================================================
# Statistical Channel Model Configuration Classes (Legacy)
# =============================================================================

# Summary of the API
#
# The `channel_config.py` module provides a comprehensive set of classes and
# configurations for simulating and modeling wireless communication channels,
# particularly in the context of 5G networks. The API is designed to be
# flexible and extensible, allowing users to configure various aspects of the
# channel model according to their specific needs.
#
# Key Components:
#
# 1. **Scenario Enum**:
#    - Defines different deployment scenarios such as Urban Macro (UMa), Urban
#      Micro (UMi), and Rural Macro (RMa).
#
# 2. **Coordinate Class**:
#    - Represents a 3D coordinate in the global coordinate system, used for
#      specifying locations of user terminals (UTs) and cells.
#
# 3. **AntPanelConfig Class**:
#    - Configures antenna panel parameters, including:
#      - Number of antennas and array dimensions (M_g, N_g, M, N, P)
#      - Antenna spacing in wavelengths
#      - Antenna patterns (theta and phi patterns in dB)
#      - Polarization angles
#      - Antenna model type (isotropic, directional, or direct pattern)
#
# 4. **UtParamCfg Class**:
#    - Defines parameters for user terminals, including:
#      - Unique ID and location
#      - Outdoor/indoor indicator
#      - Antenna panel configuration index
#      - Antenna panel orientation in GCS (theta, phi, slant offset)
#      - Mobility parameters
#      - Serving cell ID
#
# 5. **CellParam Class**:
#    - Specifies parameters for cells, including:
#      - Cell ID and site ID
#      - Location in GCS
#      - Antenna panel configuration index
#      - Antenna panel orientation in GCS (theta, phi, slant offset)
#      - Co-sited cells share the same site ID and LSP
#
# 6. **SimConfig Class**:
#    - Configures simulation parameters, including:
#      - Link simulation settings
#      - Frequency and bandwidth
#      - Subcarrier spacing and FFT size
#      - PRB and PRBG configurations
#      - Channel realization settings
#
# 7. **SystemLevelConfig Class**:
#    - Configures system-level parameters, including:
#      - Scenario type and inter-site distance
#      - Number of sites and sectors
#      - Number of UTs
#      - Path loss and shadowing options
#      - O2I penetration loss settings
#      - Near-field and non-stationarity effects
#
# 8. **LinkLevelConfig Class**:
#    - Configures link-level parameters, including:
#      - Fast fading type and delay profile
#      - Delay spread and mobility
#      - Number of rays and paths
#      - CFO and delay settings
#
# 9. **ExternalConfig Class**:
#    - Manages external configurations, including:
#      - Cell and UT configurations
#      - Channel buffers in sparse format:
#        - CIR coefficients and indices
#        - Number of non-zero taps
#        - CFR per subcarrier and PRB group
#
# 10. **StatisChanModel Class**:
#     - The main class that integrates all configurations and provides:
#       - Channel model simulation with active cells and UTs
#       - UT location and mobility updates
#       - LOS/NLOS and path loss/shadowing statistics
#       - Channel state reset functionality
#
# Usage:
# The API is designed to be used in simulations where detailed modeling of
# wireless channels is required. It allows for the configuration of various
# parameters to match real-world scenarios as specified by standards like
# 3GPP TR 38.901. The modular design enables easy extension and adaptation to
# different research and development needs in wireless communications.
#
# Key Features:
# - Support for both isotropic and directional antenna patterns
# - Flexible coordinate system (GCS and LCS)
# - Sparse format for efficient CIR storage
# - Configurable path loss and shadowing models
# - Support for O2I penetration losses
# - Near-field and non-stationarity effects
#
# This summary provides an overview of the key components and their roles
# within the API, offering a foundation for users to understand and utilize
# the module effectively.
#
# Note: Statistical channel configuration classes (Scenario, Coordinate,
# AntPanelConfig, UtParamCfg, CellParam, SimConfig, SystemLevelConfig,
# LinkLevelConfig, ExternalConfig) are provided by the C++ bindings and
# re-exported from aerial.phy5g.channel_models. See pybind11 docstrings
# for parameter documentation.


# =============================================================================
# Antenna Pattern Helper Functions
# =============================================================================

def create_antenna_pattern(ant_model: int) -> Tuple[List[float], List[float]]:
    """Generate antenna patterns for isotropic or directional antenna models.

    This helper function creates the antenna patterns required for AntPanelConfig
    when using antenna model 0 (isotropic) or 1 (directional/3GPP).

    Args:
        ant_model: Antenna model type:

            - 0: Isotropic pattern (constant 0 dB gain in all directions)
            - 1: Directional pattern (3GPP TR 38.901 compliant)

    Returns:
        Tuple of (ant_theta, ant_phi) where:

            - ant_theta: List of 181 floats for A(theta, phi=0) pattern in dB
            - ant_phi: List of 360 floats for A(theta=90, phi) pattern in dB

    Raises:
        ValueError: If ant_model is not 0 or 1.

    Example:
        >>> ant_theta, ant_phi = create_antenna_pattern(ant_model=1)
        >>> config = AntPanelConfig(
        ...     n_ant=4,
        ...     ant_size=[1, 1, 1, 2, 2],
        ...     ant_spacing=[0, 0, 0.5, 0.5],
        ...     ant_theta=ant_theta,
        ...     ant_phi=ant_phi,
        ...     ant_polar_angles=[45, -45],
        ...     ant_model=1
        ... )

    Note:
        For ant_model=2 (direct pattern), you must provide your own
        measured or custom antenna patterns.
    """
    if ant_model == 0:
        # Isotropic pattern: constant 0 dB gain in all directions
        ant_theta = [0.0] * 181
        ant_phi = [0.0] * 360
    elif ant_model == 1:
        # Directional pattern per 3GPP TR 38.901
        # A(theta) = -min[12*(theta/theta_3dB)^2, SLA_v]
        theta_3db = 65.0  # degrees
        sla_v = 30.0  # dB
        ant_theta = [
            -min(12.0 * ((theta - 90) / theta_3db) ** 2, sla_v)
            for theta in range(181)
        ]

        # A(phi) = -min[12*(phi/phi_3dB)^2, A_max]
        phi_3db = 65.0  # degrees
        a_max = 30.0  # dB
        ant_phi = []
        for phi in range(360):
            phi_wrap = phi - 360 if phi >= 180 else phi
            ant_phi.append(-min(12.0 * (phi_wrap / phi_3db) ** 2, a_max))
    else:
        raise ValueError(
            f"ant_model must be 0 (isotropic) or 1 (directional), got {ant_model}. "
            "For ant_model=2 (direct pattern), provide custom ant_theta/ant_phi arrays."
        )

    return ant_theta, ant_phi

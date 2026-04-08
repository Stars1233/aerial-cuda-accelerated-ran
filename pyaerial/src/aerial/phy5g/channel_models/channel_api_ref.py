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

"""
pyAerial library - channel models API reference.

DOCUMENTATION ONLY - This file is kept for reference and discussion.
The classes documented here are available in ``aerial.pycuphy`` and
``aerial.phy5g.channel_models``.
Attribute names are the Python ``snake_case`` names and may differ
from C++ ``camelCase`` names.

See ``pyaerial/tests/`` for working examples:

- ``test_chmod_sls.py`` -- system-level statistical channel model
- ``test_chmod_fading_chan.py`` -- fading channel with OFDM modulation
- ``test_chmod_isac.py`` -- ISAC integrated sensing and communication
- ``test_chmod_ofdm.py`` -- OFDM modulator and demodulator
"""

# pylint: disable=no-member,too-many-positional-arguments,too-many-lines
from typing import List, Optional
import numpy as np

# Summary of the API (consolidated; full prose lives in class docstrings below).
#
# Stubs mirror ``pycuphy`` / 3GPP TR 38.901: system-level (UMa, UMi, RMa),
# link-level (TDL/CDL), O2I, path loss / shadowing, and ISAC (targets, RCS,
# monostatic/bistatic hooks). Indoor / InF / SMa scenarios may be TODO in the
# library build.
#
# Key components (parameters are defined on each class; this is a table of contents):
#  1. ``Scenario`` — Deployment scenario type (38.901 UMa / UMi / RMa; some enums TODO).
#  2. ``SensingTargetType`` — ISAC sensing-target category enum (38.901 §7.9).
#  3. ``UeType`` — User-equipment category enum.
#  4. ``Coordinate`` — 3D point in the simulation global coordinate system.
#  5. ``SpstParam`` — One sub-pixel scattering point on an ISAC target.
#  6. ``StParam`` — One ISAC sensing target (ST) and its SPST list / RCS model.
#  7. ``AntPanelConfig`` — Antenna panel geometry, spacing, patterns, polarization.
#  8. ``UtParamCfg`` — Per-UE placement, antenna panel, mobility, UE type, ISAC flags.
#  9. ``CellParam`` — Per-sector cell/site placement, antenna panel, ISAC flags.
# 10. ``SimConfig`` — Waveform / FFT / PRB policy, run mode, CFR/CIR memory policy.
# 11. ``SystemLevelConfig`` — Scenario geometry, LSP options, ISAC ST rules, UT drop.
# 12. ``LinkLevelConfig`` — Small-scale fast fading (AWGN / TDL / CDL) parameters.
# 13. ``ExternalConfig`` — Explicit cell / UE / panel / ST parameters (vs auto-generated).
# 14. ``StatisChanModel`` — Documentation stub for the bound stochastic channel API.
#
# Usage: align these names with live ``pycuphy`` objects and YAML where
# applicable. CIR/CFR buffers are passed to ``run()``, not stored on
# ``ExternalConfig``. For ISAC, set ``SystemLevelConfig.isac_type`` and populate
# ``ExternalConfig.st_config`` as needed.
#
# Features called out in 38.901-style modeling: isotropic or Table 7.3-1 / custom
# patterns; GCS vs panel LCS for AoA/AoD; sparse CIR; configurable LSP and O2I;
# optional near-field and non-stationarity; ISAC multi-SPST / calibration flags.
# Binding quirks and defaults: see pybind11 docstrings for your SDK revision.

# =============================================================================
# Enums
# =============================================================================


class Scenario:
    """Deployment scenario per 3GPP TR 38.901 (Python: ``pycuphy.Scenario`` enum).

    Bindings export numeric enum members; assign e.g. ``system_level_config.scenario =
    Scenario.UMa`` (type/values match C++ ``scenario_t``).

    Members:
        UMa — Urban macro (default in many configs).
        UMi — Urban micro.
        RMa — Rural macro; pair with appropriate ``isd`` (m), often 1732 or 5000.
        Indoor — Reserved / not supported yet in library.
        InF — Indoor factory; not supported yet.
        SMa — Suburban macro; not supported yet.
    """
    UMa = 'UMa'    # Urban Macro
    UMi = 'UMi'    # Urban Micro
    RMa = 'RMa'    # Rural Macro
    Indoor = 'Indoor'  # TODO: Not supported yet
    InF = 'InF'  # TODO: Not supported yet (Indoor Factory)
    SMa = 'SMa'  # TODO: Not supported yet, currently in CR (Suburban Macro)


class SensingTargetType:
    """ISAC sensing target kind (Python: ``pycuphy.SensingTargetType`` enum).

    Used in ``StParam.target_type``, ``SystemLevelConfig.st_target_type``, etc.
    Each type corresponds to 38.901 tables for RCS / size.

    Members (int-backed in bindings):
        UAV — Table 7.9.1-1.
        AUTOMOTIVE — Table 7.9.1-2.
        HUMAN — Table 7.9.1-3.
        AGV — Table 7.9.1-4.
        HAZARD — Table 7.9.1-5.
    """
    UAV = 0         # UAV (drones), Table 7.9.1-1
    AUTOMOTIVE = 1  # Automotive, Table 7.9.1-2
    HUMAN = 2       # Human target, Table 7.9.1-3
    AGV = 3         # Automated Guided Vehicle, Table 7.9.1-4
    HAZARD = 4      # Hazards on roads/railways, Table 7.9.1-5


class UeType:
    """UE equipment category (Python: ``pycuphy.UeType`` enum).

    Set on ``UtParamCfg.ue_type``. AERIAL triggers 36.777-style behavior where
    applicable.

    Members:
        TERRESTRIAL — Handheld / fixed UE.
        VEHICLE — V2X-capable vehicular UE.
        AERIAL — Drones / UAV as communication UE.
        AGV — Industrial AGV.
        RSU — Road-side unit (fixed V2X).
    """
    TERRESTRIAL = 0  # Handheld/fixed (smartphones, tablets, CPE)
    VEHICLE = 1      # Vehicular UE for V2X (cars, trucks, buses)
    AERIAL = 2       # Aerial UE (drones, UAVs); uses TR 36.777
    AGV = 3          # Automated Guided Vehicle (industrial robots)
    RSU = 4          # Road Side Unit (fixed V2X infrastructure)


# =============================================================================
# Coordinate and ISAC Structures
# =============================================================================

class Coordinate:
    """3D point in the simulation global coordinate system (GCS).

    Python binding: ``pycuphy.Coordinate`` with float fields.

    Attributes (constructor args):
        x — Scalar, meters (axis convention shared with C++ layout / topology).
        y — Scalar, meters.
        z — Scalar, meters; height above reference (typically ground).
    """

    def __init__(self,
                 x: float = 0,  # x-coordinate in GCS, meters
                 y: float = 0,  # y-coordinate in GCS, meters
                 z: float = 0):  # z-coordinate (height), meters
        """Initialize coordinate with x, y, z values."""
        self.x = x
        self.y = y
        self.z = z


_DEFAULT_COORD = Coordinate(0, 0, 0)


class SpstParam:
    """One scattering point on a sensing target (Python: ``pycuphy.SpstParam``).

    Referenced from ``StParam.spst_configs``. TR 38.901 §7.9.2.1; RCS tables
    7.9.2.1-1. (Angular RCS fields may exist in C++ for model 2; not all are
    exposed in every binding—check ``pycuphy`` for extra properties.)

    Attributes:
        spst_id — Index of this SPST inside the parent ``StParam``.
        loc_in_st_lcs — ``Coordinate`` in the target’s local frame (offset from ST center).
        rcs_sigma_m_dbsm — Mean monostatic RCS σ_M in dBsm (10·log10(σ_M[m²])).
        rcs_sigma_d_dbsm — Mean monostatic RCS σ_D in dBsm.
        rcs_sigma_s_db — Standard deviation σ_s of RCS in dB.
        enable_forward_scattering — 0=disable, 1=enable forward term in Eq. 7.9.2-2.
    """

    def __init__(self,
                 spst_id: int = 0,  # SPST ID within the ST (0-indexed)
                 loc_in_st_lcs: Coordinate = _DEFAULT_COORD,  # Location of SPST in ST's
                 # local coordinate system; relative to ST center position
                 rcs_sigma_m_dbsm: float = -12.81,  # Mean monostatic RCS
                 # sigma_M in dBsm (10*log10(sigma_M)), Table 7.9.2.1-1
                 rcs_sigma_d_dbsm: float = 1.0,  # Mean monostatic RCS
                 # sigma_D in dBsm (10*log10(sigma_D))
                 rcs_sigma_s_db: float = 3.74,  # Standard deviation sigma_s_dB
                 # in dB
                 enable_forward_scattering: int = 1,  # Control forward-scattering
                 # effect (7.9.2-2): 0=disable, 1=enable (default)
                 ):
        """Initialize SPST parameters."""
        self.spst_id = spst_id
        self.loc_in_st_lcs = loc_in_st_lcs
        self.rcs_sigma_m_dbsm = rcs_sigma_m_dbsm
        self.rcs_sigma_d_dbsm = rcs_sigma_d_dbsm
        self.rcs_sigma_s_db = rcs_sigma_s_db
        self.enable_forward_scattering = enable_forward_scattering


class StParam:
    """One sensing target for ISAC (Python: ``pycuphy.StParam``).

    Listed in ``ExternalConfig.st_config``. TR 38.901 §7.9.2. LOS/NLOS is
    per-link (STX–SPST, SPST–SRX), not a single flag on the ST.

    Attributes:
        sid — Global sensing-target identifier.
        target_type — ``SensingTargetType`` value (UAV, AUTOMOTIVE, …).
        outdoor_ind — 0=indoor ST, 1=outdoor ST.
        loc — ``Coordinate`` in GCS (m).
        rcs_model — 1=deterministic monostatic; 2=angular-dependent σ_D (requires
            consistent ``n_spst`` and ``spst_configs``; binding validates).
        n_spst — Number of SPSTs; must be 1 for model 1; model 2 allows multiple
            (e.g. 5 for auto/AGV facets: front/left/back/right/roof).
        spst_configs — ``list[SpstParam]``, length must match ``n_spst``.
        velocity — ``list[float]`` length 3 ``[vx,vy,vz]`` m/s.
        target_orientation — ``list[float]`` length 2 ``[azimuth, elevation]`` deg in GCS.
        physical_size — ``list[float]`` length 3 ``[length, width, height]`` m.
    """

    def __init__(self,  # pylint: disable=too-many-arguments
                 sid: int = 0,  # Global ST ID (Sensing Target ID)
                 target_type: int = SensingTargetType.UAV,  # 0=UAV, 1=AUTOMOTIVE,
                 # 2=HUMAN, 3=AGV, 4=HAZARD (Tables 7.9.1-1 to 7.9.1-5)
                 outdoor_ind: int = 1,  # 0=indoor, 1=outdoor (ST-specific).
                 # LOS/NLOS assigned per-link, not per-ST
                 loc: Coordinate = _DEFAULT_COORD,  # ST location in GCS [x, y, z] meters
                 rcs_model: int = 1,  # 1=deterministic monostatic (angular
                 # independent sigma_D), 2=angular dependent sigma_D
                 n_spst: int = 1,  # Number of scattering points. Model 1: must
                 # be 1. Model 2: 1 or more (5 for automotive/AGV)
                 spst_configs: Optional[List[SpstParam]] = None,  # SPST parameter configs,
                 # one per scattering point
                 velocity: Optional[List[float]] = None,  # Velocity vector [vx, vy, vz] m/s
                 target_orientation: Optional[List[float]] = None,  # Target orientation in
                 # GCS [azimuth, elevation] degrees; for RCS calculation
                 physical_size: Optional[List[float]] = None,  # Physical dimensions
                 # [length, width, height] in meters
                 ):
        """Initialize sensing target parameters."""
        self.sid = sid
        self.target_type = target_type
        self.outdoor_ind = outdoor_ind
        self.loc = loc
        self.rcs_model = rcs_model
        self.n_spst = n_spst
        self.spst_configs = list(spst_configs) if spst_configs is not None else []
        self.velocity = list(velocity) if velocity is not None else [0.0, 0.0, 0.0]
        self.target_orientation = (
            list(target_orientation) if target_orientation is not None else [0.0, 0.0]
        )
        self.physical_size = (
            list(physical_size) if physical_size is not None else [0.0, 0.0, 0.0]
        )


# =============================================================================
# Antenna Panel
# =============================================================================

class AntPanelConfig:
    """Antenna panel definition (Python: ``pycuphy.AntPanelConfig``).

    Lives in ``ExternalConfig.ant_panel_config``; ``UtParamCfg.ant_panel_idx`` /
    ``CellParam.ant_panel_idx`` select which panel applies.

    **Python attribute names** (bindings use properties with length checks):
        n_ant — int; must equal ``M_g*N_g*P*M*N`` from ``ant_size``.
        ant_size — list[int] length **5**: ``[M_g, N_g, M, N, P]`` per 38.901 §7.3.
        ant_spacing — list[float] length **4**: ``[d_gh, d_gv, d_h, d_v]`` in wavelengths.
        ant_theta — list[float] length **181**: ``A(θ, φ=0)`` in dB, θ=0…180°.
        ant_phi — list[float] length **360**: ``A(90°, φ)`` in dB, φ=0…359°.
        ant_polar_angles — list[float] length **2**:
            ``[roll_first_pol, roll_second_pol]`` (deg); dual-pol often ±45.
        ant_model — int: **0** isotropic, **1** 38.901 Table 7.3-1 directional,
            **2** user tables ``ant_theta``/``ant_phi`` required.

    Antenna pattern calculation (ant_model 0 and 1):
    - Model 0 (isotropic): A(theta)=0 dB, A(phi)=0 dB for all angles.
    - Model 1 (directional, 3GPP TR 38.901 Table 7.3-1):
      A(theta) = -min[12*((theta-90)/theta_3dB)^2, SLA_v]
        theta_3dB = 65 degrees, SLA_v = 30 dB
        theta in [0, 180] degrees
      A(phi) = -min[12*(phi_wrap/phi_3dB)^2, A_max]
        phi_3dB = 65 degrees, A_max = 30 dB
        phi_wrap = phi - 360 if phi >= 180 else phi (phi in [0, 359])
    - Model 2 (direct pattern): user provides ant_theta (181), ant_phi (360).

    Field strength from pattern (Eq 7.3-4, 7.3-5):
      F_theta = sqrt(10^(A_theta/10)) * cos(slant_angle)
      F_phi   = sqrt(10^(A_phi/10))   * sin(slant_angle)
    slant_angle = additional_slant_offset + ant_polar_angles[pol_index]
    """

    def __init__(self,
                 n_ant: int = 4,  # Number of antennas in the array.
                 # n_ant = M_g * N_g * P * M * N
                 ant_size: Optional[List[int]] = None,  # Dimensions of the antenna array
                 # (M_g, N_g, M, N, P). TODO: only support one panel for now
                 # M_g = N_g = 1.
                 ant_spacing: Optional[List[float]] = None,  # Spacing between antennas
                 # in terms of wavelength (d_g_h, d_g_v, d_h, d_v)
                 ant_theta: Optional[List[float]] = None,  # Antenna pattern A(theta,
                 # phi=0) in dB, dimension 181 x 1, for theta in [0, 180].
                 # Required for ant_model=2 (direct pattern).
                 ant_phi: Optional[List[float]] = None,  # Antenna pattern A(theta=90,
                 # phi) in dB, dimension 360 x 1, for phi in [0, 359].
                 # Required for ant_model=2 (direct pattern).
                 ant_polar_angles: Optional[List[float]] = None,  # Antenna polar angles
                 # (roll_angle_first_polz, roll_angle_second_polz).
                 # Default ±45 deg for dual-pol, 0 deg for single-pol.
                 ant_model: int = 2,  # Antenna model type: 0=isotropic,
                 # 1=directional (3GPP TR 38.901 Table 7.3-1),
                 # 2=direct antenna pattern (user-provided ant_theta, ant_phi).
                 # For 0/1: F_theta=sqrt(10^(A_theta/10))*cos(slant_angle),
                 # F_phi=sqrt(10^(A_phi/10))*sin(slant_angle); slant_angle from
                 # ant_polar_angles and additional slant offset.
                 ):
        """Initialize antenna panel configuration."""
        size = list(ant_size) if ant_size is not None else [1, 1, 1, 2, 2]
        spacing = list(ant_spacing) if ant_spacing is not None else [0, 0, 0.5, 0.5]
        if not ant_polar_angles:
            ant_pol = [45.0, -45.0] if size[4] == 2 else [0.0]
        else:
            ant_pol = list(ant_polar_angles)
        self.n_ant = n_ant
        self.ant_size = size
        self.ant_spacing = spacing
        self.ant_theta = ant_theta
        self.ant_phi = ant_phi
        self.ant_polar_angles = ant_pol
        self.ant_model = ant_model

    def calc_ant_pattern(self) -> None:
        """Calculate antenna patterns for isotropic (0) and directional (1) models.
        This function is insider the C++ channel model library. Here is just to show
        the format of how theta and phi are calculated for directional model.
        For ant_model=2, ant_theta and ant_phi must be provided by the user.

        Model 0 (isotropic): ant_theta = [0]*181, ant_phi = [0]*360 (0 dB gain).
        Model 1 (3GPP directional):
            theta_deg = linspace(0, 180, 181)
            A(theta) = -min[12*((theta_deg-90)/65)^2, 30]
            phi_deg = linspace(0, 359, 360), phi_wrap = phi-360 if phi>=180 else phi
            A(phi) = -min[12*(phi_wrap/65)^2, 30]

        The actual antenna pattern will be calculated using:
            F_theta = sqrt(10^(A_theta/10)) * cos(slant_angle)
            F_phi = sqrt(10^(A_phi/10)) * sin(slant_angle)
            where slant_angle is determined by ant_polar_angles and
            additional slant offset
        """
        pass


# =============================================================================
# UT and Cell Parameters
# =============================================================================

class UtParamCfg:
    """Per-UE configuration (Python: ``pycuphy.UtParamCfg``).

    Typical container: ``ExternalConfig.ut_config`` with length ``n_ut`` (global
    UE index equals ``uid`` ordering as set up by the simulation).

    **Python attributes**
        uid — int; global UE id (0 … ``n_ut``-1 in simple layouts).
        loc — ``Coordinate``; initial/reference position in GCS (m).
        outdoor_ind — int; **0** indoor UT, **1** outdoor UT (drives O2I and tables).
        ant_panel_idx — int; index into ``ExternalConfig.ant_panel_config``.
        ant_panel_orientation — ``list[float]`` length 3:
            ``[θ_bearing, φ_tilt, extra_slant]`` deg in GCS for panel mount; see
            class comments below for LCS primes vs ZOA/AOA/ZOD/AOD.
        velocity — ``list[float]`` length 3 ``[vx,vy,vz]`` m/s; **vz** should
            follow 3GPP usage (often 0 for ground UE).
        ue_type — ``UeType`` enum value (TERRESTRIAL, VEHICLE, …).
        monostatic_ind — **0** normal UE; **1** UE acts as monostatic sensing RX.
        same_antenna_panel_ind — **0** use ``second_ant_panel_*`` for
            sensing RX; **1** reuse main panel for sensing.
        second_ant_panel_idx — int; index into panel pool when separate
            sensing array is used.
        second_ant_panel_orientation — ``list[float]`` length 3,
            same convention as ``ant_panel_orientation``.

    Detailed GCS→LCS angle / slant / pattern math: **#** comment block below.
    """

    # antenna panel orientation in GCS, dim: 3: (theta, phi,
    # additional slant offset) for each antenna element.
    # antenan angle calculation: (_prime is in LCS and _ is in GCS)
    # theta_n_m_ZOA_prime = theta_n_m_ZOA(n, m) -
    #     UtParamCfg.antPanelOrintation[0];
    # phi_n_m_AOA_prime = phi_n_m_AOA(n, m) -
    #     UtParamCfg.antPanelOrintation[1];
    # theta_n_m_ZOD_prime = theta_n_m_ZOD(n, m) -
    #     CellParam.antPanelOrientation[0];
    # phi_n_m_AOD_prime = phi_n_m_AOD(n, m) -
    #     CellParam.antPanelOrientation[1];
    # these angles are used to read the antenna pattern F_theta
    # and F_phi in LCS
    # for single polarization, slant angle = additional slant
    # offset + AntPanelConfig.ant_polar_angles[0]
    # for dual polarization, slant angle 0 = additional slant
    # offset + AntPanelConfig.ant_polar_angles[0]
    #                        slant angle 1 = additional slant
    # offset + AntPanelConfig.ant_polar_angles[1]
    # these angles are used to calculate the antenna pattern by
    # cos(slant angle *) * F_theta and sin(slant angle *) * F_phi,
    # Eq 7.3-4 and Eq 7.3-5 in 3GPP TR 38.901

    def __init__(self,  # pylint: disable=too-many-arguments
                 uid: int = 0,  # Global UE ID
                 loc: Coordinate = _DEFAULT_COORD,  # UE location at beginning [x,y,z] m
                 outdoor_ind: int = 0,  # Outdoor indicator: 0=indoor, 1=outdoor.
                 # Calculated at AODT side; generated at simulation start.
                 ant_panel_idx: int = 0,  # Antenna panel configuration index
                 ant_panel_orientation: Optional[List[float]] = None,  # Antenna panel
                 # orientation in GCS; see class comment block above for LCS transform
                 velocity: Optional[List[float]] = None,  # Mobility parameters (vx, vy, vz).
                 # abs(velocity) = speed in m/s; vz = 0 per 3GPP spec.
                 ue_type: int = UeType.TERRESTRIAL,  # 0=TERRESTRIAL, 1=VEHICLE,
                 # 2=AERIAL (3GPP TR 36.777), 3=AGV, 4=RSU
                 monostatic_ind: int = 0,  # ISAC: 0=not a monostatic sensing
                 # receiver, 1=monostatic (UE receives sensing reflections)
                 same_antenna_panel_ind: int = 0,  # 0=use second antenna
                 # panel for sensing RX, 1=use same antenna panel for sensing
                 second_ant_panel_idx: int = 0,  # Second antenna
                 # panel index for sensing RX (when monostatic_ind=1 and
                 # same_antenna_panel_ind=0)
                 # Second antenna panel orientation [theta, phi, slant_offset]
                 second_ant_panel_orientation: Optional[List[float]] = None):
        """Initialize User Terminal parameters."""
        self.uid = uid
        self.loc = loc
        self.outdoor_ind = outdoor_ind
        self.ant_panel_idx = ant_panel_idx
        zeros3 = [0.0, 0.0, 0.0]
        self.ant_panel_orientation = (
            list(ant_panel_orientation) if ant_panel_orientation is not None else list(zeros3)
        )
        self.velocity = list(velocity) if velocity is not None else list(zeros3)
        self.ue_type = ue_type
        self.monostatic_ind = monostatic_ind
        self.same_antenna_panel_ind = same_antenna_panel_ind
        self.second_ant_panel_idx = second_ant_panel_idx
        self.second_ant_panel_orientation = (
            list(second_ant_panel_orientation)
            if second_ant_panel_orientation is not None
            else list(zeros3)
        )


class CellParam:
    """Per-sector base station cell (Python: ``pycuphy.CellParam``).

    Typical container: ``ExternalConfig.cell_config`` length
    ``n_site * n_sector_per_site``; ``cid`` is the global sector index.

    **Python attributes**
        cid — int; global cell id, **0 … n_site·n_sector_per_site − 1**.
        site_id — int; **0 … n_site − 1**; co-sited sectors share LSP/state
            (same ``site_id``, different ``cid`` / bearing).
        loc — ``Coordinate``; e.g. BS position in GCS (m).
        ant_panel_idx — int; index into ``ExternalConfig.ant_panel_config``.
        ant_panel_orientation — ``list[float]`` length 3 GCS orientation
            as for UE; co-sited sectors often rotate by **120° / 240°** in the
            horizontal component (see # comments below).
        monostatic_ind — **0** comms-only gNB; **1** gNB TX/RX for monostatic ISAC.
        second_ant_panel_idx — int; sensing RX panel when enabled.
        second_ant_panel_orientation — ``list[float]`` length 3.

    Pattern / slant math matches ``UtParamCfg``; see **#** block below.
    """

    # antenna panel orientation in GCS, dim: 3: (theta, phi,
    # additional slant offset) for each antenna element.
    # the co-sites cells will have antPanelOrintation[1] separated
    # by 120, 240 degrees
    # antenan angle calculation: (_prime is in LCS and _ is in GCS)
    # theta_n_m_ZOA_prime = theta_n_m_ZOA(n, m) -
    #     UtParamCfg.antPanelOrientation[0];
    # phi_n_m_AOA_prime = phi_n_m_AOA(n, m) -
    #     UtParamCfg.antPanelOrientation[1];
    # theta_n_m_ZOD_prime = theta_n_m_ZOD(n, m) -
    #     CellParam.antPanelOrientation[0];
    # phi_n_m_AOD_prime = phi_n_m_AOD(n, m) -
    #     CellParam.antPanelOrientation[1];
    # these angles are used to read the antenna pattern F_theta
    # and F_phi in LCS
    # for single polarization, slant angle = additional slant
    # offset + AntPanelConfig.ant_polar_angles[0]
    # for dual polarization, slant angle 0 = additional slant
    # offset + AntPanelConfig.ant_polar_angles[0]
    # slant angle 1 = additional slant
    # offset + AntPanelConfig.ant_polar_angles[1]
    # these angles are used to calculate the antenna pattern by
    # cos(slant angle *) * F_theta and sin(slant angle *) * F_phi,
    # Eq 7.3-4 and Eq 7.3-5 in 3GPP TR 38.901

    def __init__(self,
                 cid: int,  # Global cell ID, 0 ~ n_site*n_sector_per_site-1
                 site_id: int,  # Site ID, 0 ~ n_site-1, used to access LSP.
                 # Cells with same site_id are co-sited, share same LSP.
                 loc: Coordinate,  # Cell location (x, y, z), constant during
                 # simulation
                 ant_panel_idx: int,  # Antenna parameters, index of panel config
                 ant_panel_orientation: Optional[List[float]] = None,  # Antenna panel
                 # orientation in GCS; see class comment block above for LCS transform
                 monostatic_ind: int = 0,  # ISAC: 0=communication only,
                 # 1=monostatic (BS acts as both TX and RX for sensing)
                 second_ant_panel_idx: int = 0,  # Second antenna
                 # panel index for sensing RX (when monostatic_ind=1)
                 # Second antenna panel orientation [theta, phi, slant_offset]
                 second_ant_panel_orientation: Optional[List[float]] = None):
        """Initialize base station cell parameters."""
        self.cid = cid
        self.site_id = site_id
        self.loc = loc
        self.ant_panel_idx = ant_panel_idx
        zeros3 = [0.0, 0.0, 0.0]
        self.ant_panel_orientation = (
            list(ant_panel_orientation) if ant_panel_orientation is not None else list(zeros3)
        )
        self.monostatic_ind = monostatic_ind
        self.second_ant_panel_idx = second_ant_panel_idx
        self.second_ant_panel_orientation = (
            list(second_ant_panel_orientation)
            if second_ant_panel_orientation is not None
            else list(zeros3)
        )


# =============================================================================
# Configuration Classes
# =============================================================================

class SystemLevelConfig:
    """Geometry, large-scale statistics, UT drop, and ISAC controls.

    Python: ``pycuphy.SystemLevelConfig``. Paired with ``SimConfig`` (waveform /
    FFT) and ``ExternalConfig`` (topology). Many fields are small integers
    selecting 38.901 equation branches.

    **Core geometry / deployment**
        scenario — ``Scenario`` enum (UMa / UMi / RMa / …).
        isd — float; inter-site distance **m** (RMa: commonly 1732 or 5000;
            less central for UMa/UMi fixed layouts).
        n_site — int; number of sites.
        n_sector_per_site — int; **1** (single sector) or **3** (hex sectors).
        n_ut — int; total UEs in simulation.

    **Path loss / fading / LSP control**
        optional_pl_ind — **0** standard PL model, **1** optional PL variant.
        o2i_building_penetr_loss_ind — building O2I: **0** none; **1** low-loss;
            **2** mixed low/high; **3** high-loss (full matrix depends on
            scenario—align with YAML / C++ for your build).
        o2i_car_penetr_loss_ind — vehicle O2I: **0** none; **1** basic; **2** mix;
            **3** metallized (RMa-focused in many builds).
        enable_near_field_effect — **0** off, **1** on.
        enable_non_stationarity — **0** off, **1** on.
        force_los_prob — ``list[float]`` length 2 ``[p_outdoor, p_indoor]``
            in **[0,1]** or **-1** to auto per table; **binding order is outdoor first**.
        force_ut_speed — ``list[float]`` ``[v_outdoor, v_indoor]`` m/s or **-1** auto.
        force_indoor_ratio — float in **[0,1]** or **-1** auto indoor fraction.
        disable_pl_shadowing — **0** compute PL+SF, **1** skip (calibration).
        disable_small_scale_fading — **0** full SSF; **1** only large-scale (fast
            fading scalars → 1).
        enable_per_tti_lsp — **0** static LSP; **1** refresh PL/O2I/SF per TTI;
            **2** refresh all LSP components (per C++ semantics).
        enable_propagation_delay — **0** omit **distance/c** term in tap times;
            **1** include in CIR (CFR via FFT of composite impulse response).

    **ISAC (when ``isac_type`` > 0)**
        isac_type — **0** communications only; **1** monostatic sensing;
            **2** bistatic (separate TX/RX roles per implementation).
        n_st — int; count of sensing targets when auto-generated; **0** disables.
        st_horizontal_speed — ``list[float]`` length 2 **[min, max]** m/s
            horizontal ST speed (binding default ≈ 30 km/h).
        st_vertical_velocity — float; mean vertical velocity m/s (binding default 0).
        st_distribution_option — ``list[int]`` length 2 ``[horiz_opt, vert_opt]``
            for ST placement rules (0=A, 1=B, 2=C on horizontal axis per TR).
        st_height — ``list[float]`` length 2 **[min, max]** ST height m;
            ``st_fixed_height`` may be a **deprecated alias** in bindings.
        st_minimum_distance — float; min spacing between STs (m); **0** = auto.
        st_size_ind — int; discrete size class for drawn targets (binding:
            0=small, 1=medium, 2=large—maps to tables/C++).
        st_min_dist_from_tx_rx — float; keep STs at least this far from any BS/UE (m).
        st_target_type — ``SensingTargetType`` default when generating STs.
        st_rcs_model — **1** deterministic monostatic RCS; **2** angle-dependent.
        path_drop_threshold_db — prune weak ISAC paths **max_power − this** (dB).
        isac_disable_background — **1** → target-only CIR (calibration).
        isac_disable_target — **1** → clutter-only CIR (calibration).

    **UT dropping**
        ut_drop_option — **0** random drop in service area; **1** equal UEs per
            site; **2** equal UEs per sector (remainder assignment rules in C++).
        ut_drop_cells — ``list[int]`` of allowed **cid** values; **[]** means all.
        ut_cell_2d_dist — ``list[float]`` length 2 **[min,max]** 2-D distance
            UE–serving cell (m); **[-1,-1]** → implementation default.

    Constructor convenience differs by binding; attributes above are what you
    read/write on the live object.
    """

    def __init__(self,  # pylint: disable=too-many-arguments,too-many-locals
                 scenario: str = Scenario.UMa,  # Scenario type: UMa, UMi, RMa
                 isd: float = 1732.0,  # Inter-site distance in meters. Only
                 # used for RMa scenario (1732 or 5000). Ignored for UMa/UMi.
                 n_site: int = 1,  # Number of sites in the simulation
                 n_sector_per_site: int = 3,  # 1 (single-cell) or 3 (hexagonal)
                 n_ut: int = 100,  # Total number of User Terminals
                 optional_pl_ind: int = 0,  # 0=standard pathloss equation,
                 # 1=optional path loss equation
                 o2i_building_penetr_loss_ind: int = 1,  # Outdoor-to-Indoor
                 # building penetration loss: 0=none, 1=low-loss, 2=50% low-loss
                 # 50% high-loss, 3=100% high-loss. UMa/UMi: 0,1,2,3;
                 # RMa: 0,1,2. Only for indoor UT.
                 o2i_car_penetr_loss_ind: int = 0,  # Outdoor-to-Indoor car
                 # penetration loss: 0=none, 1=basic, 2=50% basic 50% metallized,
                 # 3=100% metallized. Only applicable for RMa, not UMa/UMi.
                 enable_near_field_effect: int = 0,  # 0=disable near field
                 # effect, 1=enable near field effect
                 enable_non_stationarity: int = 0,  # 0=disable non-stationarity,
                 # 1=enable non-stationarity
                 force_los_prob: Optional[List[float]] = None,  # [outdoor, indoor] in [0,1];
                 # -1 = auto (Table 7.4.1-1). Order matches pycuphy binding.
                 force_ut_speed: Optional[List[float]] = None,  # [outdoor, indoor] m/s;
                 # -1 = auto per Table 7.4.1-1
                 force_indoor_ratio: float = -1,  # Force indoor ratio for all
                 # links. [0,1]=valid; -1=invalid, set from scenario
                 disable_pl_shadowing: int = 0,  # 0=calculate PL and shadowing,
                 # 1=disable PL and shadowing calculation
                 disable_small_scale_fading: int = 0,  # 0=calculate small scale
                 # fading, 1=disable (fast fading=1, only pathloss)
                 enable_per_tti_lsp: int = 0,  # Enable LSP per TTI: 0=disable,
                 # 1=update PL, O2I penetration, shadowing only, 2=update all
                 enable_propagation_delay: int = 1,  # 0=disable propagation
                 # delay in CIR generation, 1=enable. Propagation delay is
                 # link-specific, distance/speed_of_light. CIR: delay =
                 # cluster_delay + propagation_delay; CFR: FFT of CIR.
                 # ISAC parameters
                 isac_type: int = 0,  # 0=communication only, 1=monostatic
                 # (BS as TX/RX), 2=bistatic (separate TX/RX)
                 n_st: int = 0,  # Number of sensing targets (0 for comm-only)
                 st_horizontal_speed: Optional[List[float]] = None,  # [min,max] m/s;
                 # direction is random
                 st_vertical_velocity: float = 0.0,  # Vertical velocity m/s
                 st_distribution_option: Optional[List[int]] = None,  # [horizontal,
                 # vertical]; horizontal: 0=Option A, 1=Option B, 2=Option C;
                 # vertical: 0=uniform 1.5-300m, 1=height from st_height
                 st_height: Optional[List[float]] = None,  # [min,max] m for vertical
                 # Option B
                 st_minimum_distance: float = 0.0,  # Min between STs in meters
                 # (0=auto by physical size)
                 st_size_ind: int = 0,  # UAV: 0=large/1=small; AGV: 0=small/
                 # 1=large; HAZARD: 0=child/1=adult/2=animal
                 st_min_dist_from_tx_rx: float = 10.0,  # Min 2D dist from ST
                 # to any BS or UE in meters
                 st_target_type: int = SensingTargetType.UAV,  # 0=UAV,
                 # 1=AUTOMOTIVE, 2=HUMAN, 3=AGV, 4=HAZARD
                 st_rcs_model: int = 1,  # 1=deterministic (n_spst=1),
                 # 2=angular dependent (n_spst varies)
                 path_drop_threshold_db: float = 40.0,  # Drop paths weaker
                 # than max-40 dB
                 isac_disable_background: int = 0,  # 0=combine target with
                 # background, 1=target CIR only (calibration)
                 isac_disable_target: int = 0,  # 0=include target CIR,
                 # 1=background CIR only (calibration)
                 # UT drop
                 ut_drop_option: int = 0,  # UT drop control: 0=randomly across
                 # whole region; 1=same number of UTs per site; 2=same number
                 # per sector. If n_ut not divisible by n_sector/n_site, remainder
                 # assigned to first n_ut % (n_sector/n_site) sectors/sites.
                 ut_drop_cells: Optional[List[int]] = None,  # Allowed cell IDs for UE
                 # dropping, e.g. [0, 4, 8]. [] = all cells [0..n_site*
                 # n_sector_per_site-1]
                 ut_cell_2d_dist: Optional[List[float]] = None,  # UT-to-serving-cell 2D
                 # distance range [min, max] in meters; [-1, -1] = default
                 ):
        """Initialize system-level configuration parameters."""
        self.scenario = scenario
        self.isd = isd
        self.n_site = n_site
        self.n_sector_per_site = n_sector_per_site
        self.n_ut = n_ut
        self.optional_pl_ind = optional_pl_ind
        self.o2i_building_penetr_loss_ind = o2i_building_penetr_loss_ind
        self.o2i_car_penetr_loss_ind = o2i_car_penetr_loss_ind
        self.enable_near_field_effect = enable_near_field_effect
        self.enable_non_stationarity = enable_non_stationarity
        self.force_los_prob = (
            list(force_los_prob) if force_los_prob is not None else [-1.0, -1.0]
        )
        self.force_ut_speed = (
            list(force_ut_speed) if force_ut_speed is not None else [-1.0, -1.0]
        )
        self.force_indoor_ratio = force_indoor_ratio
        self.disable_pl_shadowing = disable_pl_shadowing
        self.disable_small_scale_fading = disable_small_scale_fading
        self.enable_per_tti_lsp = enable_per_tti_lsp
        self.enable_propagation_delay = enable_propagation_delay
        self.isac_type = isac_type
        self.n_st = n_st
        self.st_horizontal_speed = (
            list(st_horizontal_speed) if st_horizontal_speed is not None else [8.33, 8.33]
        )
        self.st_vertical_velocity = st_vertical_velocity
        self.st_distribution_option = (
            list(st_distribution_option) if st_distribution_option is not None else [0, 0]
        )
        self.st_height = list(st_height) if st_height is not None else [100.0, 100.0]
        self.st_minimum_distance = st_minimum_distance
        self.st_size_ind = st_size_ind
        self.st_min_dist_from_tx_rx = st_min_dist_from_tx_rx
        self.st_target_type = st_target_type
        self.st_rcs_model = st_rcs_model
        self.path_drop_threshold_db = path_drop_threshold_db
        self.isac_disable_background = isac_disable_background
        self.isac_disable_target = isac_disable_target
        self.ut_drop_option = ut_drop_option
        self.ut_drop_cells = list(ut_drop_cells) if ut_drop_cells is not None else []
        self.ut_cell_2d_dist = (
            list(ut_cell_2d_dist) if ut_cell_2d_dist is not None else [-1.0, -1.0]
        )


class LinkLevelConfig:
    """Small-scale model inside the statistical channel (Python: ``pycuphy.LinkLevelConfig``).

    Used when the SLS couples to **TDL**/**CDL** generators (see
    ``fast_fading_type``). For standalone link-level OFDM tests, use
    ``FadingChannel`` + ``TdlChannelConfig`` / ``CdlChannelConfig`` instead.

    **Python attributes**
        fast_fading_type — **0** AWGN only; **1** TDL; **2** CDL.
        delay_profile — str **'A'…'C'** selecting PDP/CDL table (D/E may be TODO).
        delay_spread — float; **nanoseconds** (RMS delay spread input to profile).
        velocity — ``list[float]`` length 3 ``[vx,vy,vz]`` m/s; binding name
            is ``velocity`` (not ``mobility``).
        num_ray — int; rays superposed per tap/cluster (**0** → library defaults:
            ~48 TDL, ~20 CDL).
        cfo_hz — float; residual carrier offset (Hz).
        delay — float; extra bulk delay **seconds** (implementation-defined use).
    """

    def __init__(self,
                 fast_fading_type: int = 0,  # Fast fading type: 0=AWGN,
                 # 1=TDL (Tapped Delay Line), 2=CDL (Clustered Delay Line)
                 delay_profile: str = 'A',  # Delay profile 'A' to 'C'.
                 # TODO: add support of 'D' and 'E'
                 delay_spread: float = 30.0,  # Delay spread in nanoseconds
                 velocity: Optional[List[float]] = None,  # [vx, vy, vz] m/s; vz=0 typical
                 num_ray: int = 0,  # Rays per path; 0 → lib default (48/20)
                 cfo_hz: float = 200.0,  # Carrier frequency offset in Hz
                 delay: float = 0.0):  # Bulk delay in seconds
        """Initialize link-level configuration parameters."""
        self.fast_fading_type = fast_fading_type
        self.delay_profile = delay_profile
        self.delay_spread = delay_spread
        self.velocity = list(velocity) if velocity is not None else [0.0, 0.0, 0.0]
        self.num_ray = num_ray
        self.cfo_hz = cfo_hz
        self.delay = delay


class SimConfig:
    """Waveform / OFDM / output tensor policy (Python: ``pycuphy.SimConfig``).

    **Python attributes**
        link_sim_ind — int; **1** when driving built-in link-sim path (legacy flag;
            many SLS runs leave **0**).
        center_freq_hz — float; carrier **Hz**.
        bandwidth_hz — float; occupied BW **Hz** (sets sampling with FFT/SCS).
        sc_spacing_hz — float; subcarrier spacing **Δf** Hz.
        fft_size — int; ``N_FFT`` (FFT length for OFDM / CFR axis).
        n_prb — int; number of PRBs ``N_PRB`` (related to used subcarriers).
        n_prbg — int; PRB **groups** for grouped CFR output.
        n_snapshot_per_slot — int; channel draws per slot (**1** or **14**).
        run_mode — int: **0** CIR taps only; **1** CIR+CFR(PRBG); **2** CIR+CFR(SC);
            **3** CIR+CFR both granularities; **4** CIR+CFR on full **N_FFT** grid
            (no PRBG aggregation). (Short labels in pybind docstrings may list a
            subset—align with your build.)
        internal_memory_mode — **0** outputs only to caller-provided tensors on
            ``run()``; **1** holds CIR on device, CFR external; **2** both internal
            (then use wrapper ``get_cir`` / ``get_cfr`` when bound).
        freq_convert_type — SC→PRBG aggregation: **0** first SC, **1** center,
            **2** last, **3** average, **4** average de-ramped.
        sc_sampling — int; stride **inside** PRBG for partial SC CFR when using
            modes **3**/**4** above.
        tx_sig_in — optional list of TX arrays (reserved / unused in many builds).
        proc_sig_freq — int; time vs frequency-domain signal processing path.
        optional_cfr_dim — **0** default layout ``[..., nPrbg|nSc]`` last;
            **1** swaps resource axis before antennas (see shapes in ``run()``).
        cpu_only_mode — **0** GPU, **1** CPU reference path if compiled in.
        h5_dump_level — **0** small debug H5; **1** richer dump.

    Sampling period for ``cir_norm_delay`` ties to ``1 / (fft_size * sc_spacing_hz)``
    in the ideal OFDM model (confirm against your TV / C++).
    """

    def __init__(self,  # pylint: disable=too-many-arguments
                 link_sim_ind: int = 0,  # Indicator for link simulation
                 center_freq_hz: float = 3e9,  # Center frequency in Hz
                 bandwidth_hz: float = 100e6,  # Bandwidth in Hz
                 sc_spacing_hz: float = 15e3 * 2,  # Subcarrier spacing in Hz
                 fft_size: int = 4096,  # FFT size for CFR calculation and OFDM processing
                 n_prb: int = 273,  # Number of PRBs (Physical Resource Blocks)
                 n_prbg: int = 137,  # Number of PRBGs (PRB Groups)
                 n_snapshot_per_slot: int = 1,  # Number of channel realizations
                 # per slot (1 or 14)
                 run_mode: int = 0,  # Run mode: 0=CIR only, 1=CIR and CFR on
                 # PRBG, 2=CIR and CFR on Sc, 3=CIR and CFR on PRBG/Sc,
                 # 4=CIR and CFR on all N_FFT subcarriers (no PRBG)
                 internal_memory_mode: int = 0,  # 0=external memory for CIR/CFR
                 # (buffer allocated outside, read ptr to put channel);
                 # 1=internal memory for CIR, external for CFR (use get*());
                 # 2=internal for CIR/CFR (use get*()); channel data still
                 # copied to external memory if given (e.g. cir_norm_delay)
                 freq_convert_type: int = 1,  # Frequency conversion for CFR on
                 # PRBG: 0=first SC, 1=center SC, 2=last SC, 3=avg SC,
                 # 4=avg SC with removing frequency ramping. Only valid when
                 # converting CFR on SC to PRBG.
                 sc_sampling: int = 1,  # Calculate CFR for subset of SCs within
                 # PRBG: only SCs for 0:scSampling:N_sc_Prbg-1. Applicable only
                 # when not using FFT and freq_convert_type=3 or 4
                 tx_sig_in: Optional[List[np.ndarray]] = None,  # Input signal
                 # for transmission. TODO: not used for now
                 proc_sig_freq: int = 0,  # Indicator for processing signal
                 optional_cfr_dim: int = 0,  # Optional CFR dimension:
                 # 0=[nActiveUt, n_snapshot, nUtAnt, nBsAnt, nPrbg/nSc],
                 # 1=[nActiveUt, n_snapshot, nPrbg/nSc, nUtAnt, nBsAnt]
                 cpu_only_mode: int = 0,  # 0=GPU mode, 1=CPU only mode
                 h5_dump_level: int = 1):  # H5 dump: 0=minimal, 1=full (default)
        """Initialize test configuration parameters."""
        self.link_sim_ind = link_sim_ind
        self.center_freq_hz = center_freq_hz
        self.bandwidth_hz = bandwidth_hz
        self.sc_spacing_hz = sc_spacing_hz
        self.fft_size = fft_size
        self.n_prb = n_prb
        self.n_prbg = n_prbg
        self.n_snapshot_per_slot = n_snapshot_per_slot
        self.run_mode = run_mode
        self.internal_memory_mode = internal_memory_mode
        self.freq_convert_type = freq_convert_type
        self.sc_sampling = sc_sampling
        self.tx_sig_in = tx_sig_in
        self.proc_sig_freq = proc_sig_freq
        self.optional_cfr_dim = optional_cfr_dim
        self.cpu_only_mode = cpu_only_mode
        self.h5_dump_level = h5_dump_level


class ExternalConfig:
    """Static topology for SLS (Python: ``pycuphy.ExternalConfig``).

    **Python attributes**
        cell_config — ``list[CellParam]``, length ``n_site * n_sector_per_site``.
        ut_config — ``list[UtParamCfg]``, length ``n_ut`` (order matches UE ids).
        ant_panel_config — ``list[AntPanelConfig]`` shared **pool**; indices referenced
            by cells and UEs.
        st_config — ``list[StParam]``; empty when ``SystemLevelConfig.isac_type==0``.

    Values are mirrored to GPU during ``StatisChanModel`` construction / updates.
    """

    def __init__(self,
                 cell_config: Optional[List[CellParam]] = None,  # Cell config
                 # per sector at start; dim n_site * n_sector_per_site
                 ut_config: Optional[List[UtParamCfg]] = None,  # UT config at
                 # start; dim n_ut
                 ant_panel_config: Optional[List[AntPanelConfig]] = None,  # Pool
                 # of antenna panel configs; cells/UTs reference by ant_panel_idx
                 st_config: Optional[List[StParam]] = None,  # Sensing target
                 # config for ISAC (when isac_type > 0)
                 ):
        """Initialize external configuration."""
        self.cell_config = list(cell_config) if cell_config is not None else []
        self.ut_config = list(ut_config) if ut_config is not None else []
        self.ant_panel_config = (
            list(ant_panel_config) if ant_panel_config is not None else []
        )
        self.st_config = list(st_config) if st_config is not None else []


# =============================================================================
# Statistical Channel Model
# =============================================================================

class StatisChanModel:
    """C++ stochastic channel backend exposed as ``pycuphy.StatisChanModel``.

    Construct with the four config objects; optional ``rand_seed`` and CUDA
    stream pointer in real bindings. ``StatisticalChannel`` in Python adds
    CuPy conversion and may expose ``get_cir`` / ``get_cfr`` even when this
    stub does not.

    **Methods (typical binding surface)**
        ``run(...)`` — system-level step; keyword args for time, fading mode,
            active cells/UEs, UE kinematics, CIR/CFR output lists (see method
            body comments).
        ``run_link_level(...)`` — TDL/CDL small-scale path with swap / layout flags.
        ``reset()`` — new small-scale draws / state as implemented in C++.
        ``dump_los_nlos_stats``, ``dump_pl_sf_stats``, ``dump_pl_sf_ant_gain_stats``
            — introspection arrays (see parameter comments).
        ``dump_topology_to_yaml(path)`` — export layout + parameters.
        ``save_sls_chan_to_h5_file(suffix='')`` — debug HDF5 dump.
        ``get_cir`` / ``get_cfr`` — present on ``StatisticalChannel``; copies last
            run results when ``internal_memory_mode`` keeps data on device (check
            your pybind build for direct ``StatisChanModel`` support).
    """

    def __init__(self,
                 sim_config: SimConfig,
                 system_level_config: SystemLevelConfig,
                 link_level_config: LinkLevelConfig,
                 external_config: ExternalConfig):
        """Initialize the statistical channel model."""
        self.sim_config = sim_config
        self.system_level_config = system_level_config
        self.link_level_config = link_level_config
        self.external_config = external_config

    def run(self,  # pylint: disable=too-many-arguments
            ref_time: float = 0.0,
            continuous_fading: int = 1,
            active_cell: Optional[List[int]] = None,
            active_ut: Optional[List[List[int]]] = None,
            ut_new_loc: Optional[np.ndarray] = None,
            ut_new_velocity: Optional[np.ndarray] = None,
            cir_coe: Optional[List[np.ndarray]] = None,
            cir_norm_delay: Optional[List[np.ndarray]] = None,
            cir_n_taps: Optional[List[np.ndarray]] = None,
            cfr_sc: Optional[List[np.ndarray]] = None,
            cfr_prbg: Optional[List[np.ndarray]] = None) -> None:
        """Run system-level channel simulation for current TTI.

        Output buffers cir_* / cfr_* are optional; when provided and
        internal_memory_mode allows, coefficients are written to host/device
        memory supplied by the caller (see SimConfig.internal_memory_mode).
        """
        pass

    def run_link_level(self,
                       ref_time0: float = 0.0,  # Reference time for CIR
                       continuous_fading: int = 1,  # 0=discontinuous,
                       # 1=continuous
                       enable_swap_tx_rx: int = 0,  # 0=DL mode, 1=UL mode
                       tx_column_major_ind: int = 0) -> None:
        """
        Run link-level channel model simulation (TDL/CDL).
        """
        pass

    def reset(self) -> None:
        """Reset channel model state."""
        pass

    def get_cir(self,
                cir_coe: Optional[List[np.ndarray]] = None,
                cir_norm_delay: Optional[List[np.ndarray]] = None,
                cir_n_taps: Optional[List[np.ndarray]] = None) -> None:
        """Fetch sparse CIR from last ``run`` when internal buffers hold it.

        Keyword-only in ``StatisticalChannel`` wrapper. Shapes match ``run()``.
        """
        pass

    def get_cfr(self,
                cfr_sc: Optional[List[np.ndarray]] = None,
                cfr_prbg: Optional[List[np.ndarray]] = None) -> None:
        """Fetch CFR from last ``run`` when internal buffers hold it."""
        pass

    def dump_los_nlos_stats(self,
                            lost_nlos_stats: Optional[np.ndarray] = None
                            ) -> None:
        """
        Dump LOS/NLOS statistics for all links.
        Dimension: [n_sector, n_ut].
        """
        pass

    def dump_pl_sf_stats(self,
                         pl_sf: np.ndarray,  # Output array (required). If
                         # active_cell and active_ut provided: dimension
                         # [active_cell.size(), active_ut.size()]. If either
                         # empty: use n_sector*n_site or n_ut for the empty one
                         active_cell: Optional[np.ndarray] = None,  # Active
                         # cell IDs, dim n_active_sector (optional)
                         active_ut: Optional[np.ndarray] = None) -> None:
        """
        Dump pathloss and shadowing statistics (negative value in dB).
        Total loss = -(pathloss - shadow_fading). Positive SF means more
        received power at UT than predicted by path loss model. Shadow fading
        sign: positive SF = more power at UT than path loss predicts.
        """
        pass

    def dump_pl_sf_ant_gain_stats(self,
                                  pl_sf_ant_gain: np.ndarray,  # Output array
                                  active_cell: Optional[np.ndarray] = None,
                                  active_ut: Optional[np.ndarray] = None
                                  ) -> None:
        """
        Dump pathloss, shadowing and antenna gain statistics.
        Antenna gain is per-element only (no array gain); downstream may add
        array/beamforming gain.
        """
        pass

    def dump_topology_to_yaml(self, filename: str) -> None:
        """Dump topology (cells, UTs, STs, etc.) to YAML file."""
        pass

    def save_sls_chan_to_h5_file(self,
                                 filename_ending: str = ""
                                 ) -> None:
        """Save SLS channel data to H5 file for debugging.
        filename_ending: optional string to append to filename.
        """
        pass

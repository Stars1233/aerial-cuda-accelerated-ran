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

"""Example usage of the statistic channel models API."""

import gc
import os
import time
import weakref
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, Tuple, List, Optional

import cupy as cp
import h5py
import numpy as np
import pytest
import yaml
from gpu3gppchan import (
    SimConfig, SystemLevelConfig, LinkLevelConfig, ExternalConfig,
    StatisticalChannel, Coordinate, AntPanelConfig, UtParamCfg, CellParam, Scenario
)
from gpu3gppchan.cuda_utils import CudaStream


def read_config(config_file: str) -> Tuple[SystemLevelConfig, LinkLevelConfig,
                                           SimConfig, ExternalConfig]:
    """
    Read configuration from YAML file and populate config structs.

    Mirrors the C++ ConfigReader::readConfig functionality.

    Args:
        config_file: Path to YAML configuration file

    Returns:
        Tuple of (system_config, link_config, sim_config, external_config)

    Raises:
        RuntimeError: If configuration file cannot be parsed or contains
                     invalid values
    """
    try:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        raise RuntimeError(f"Error loading YAML file: {e}")

    # Initialize configuration objects
    system_config = SystemLevelConfig()
    link_config = LinkLevelConfig()
    sim_config = SimConfig()
    external_config = ExternalConfig()

    try:
        # Read System Level Configuration
        if "system_level" in config:
            system_config = _read_system_level_config(config["system_level"])

        # Read Link Level Configuration
        if "link_level" in config:
            link_config = _read_link_level_config(config["link_level"])

        # Read Simulation Configuration
        if "simulation" in config:
            sim_config = _read_simulation_config(config["simulation"])

        # Read Antenna Panel Configurations
        if "antenna_panels" in config:
            _read_antenna_panel_configs(config["antenna_panels"],
                                        external_config)

    except Exception as e:
        raise RuntimeError(f"Error parsing YAML configuration: {e}")

    return system_config, link_config, sim_config, external_config


def _read_system_level_config(sl_config: dict) -> SystemLevelConfig:
    """Read system level configuration from YAML using direct constructor."""
    # Convert string to Scenario enum
    scenario_str = sl_config["scenario"]
    scenario_map = {
        "UMa": Scenario.UMa,
        "UMi": Scenario.UMi,
        "RMa": Scenario.RMa,
        "Indoor": Scenario.Indoor,
        "InF": Scenario.InF,
        "SMa": Scenario.SMa
    }

    if scenario_str not in scenario_map:
        raise RuntimeError(f"Invalid scenario: {scenario_str}")

    # Use new constructor with Python lists - much cleaner
    ut_cell_2d_dist = sl_config.get("ut_cell_2d_dist", [-1.0, -1.0])
    if len(ut_cell_2d_dist) != 2:
        raise RuntimeError(
            f"ut_cell_2d_dist must contain exactly 2 elements, got {len(ut_cell_2d_dist)}"
        )
    ut_drop_cells = [int(cid) for cid in sl_config.get("ut_drop_cells", [])]

    return SystemLevelConfig(
        scenario=scenario_map[scenario_str],
        n_site=int(sl_config["n_site"]),
        n_sector_per_site=int(sl_config["n_sector_per_site"]),  # Add missing parameter
        n_ut=int(sl_config["n_ut"]),
        isd=float(sl_config["isd"]),
        ut_drop_cells=ut_drop_cells,
        ut_drop_option=int(sl_config.get("ut_drop_option", 0)),
        ut_cell_2d_dist=[float(ut_cell_2d_dist[0]), float(ut_cell_2d_dist[1])],
        optional_pl_ind=int(sl_config["optional_pl_ind"]),
        o2i_building_penetr_loss_ind=int(sl_config["o2i_building_penetr_loss_ind"]),
        o2i_car_penetr_loss_ind=int(sl_config["o2i_car_penetr_loss_ind"]),
        enable_near_field_effect=int(sl_config["enable_near_field_effect"]),
        enable_non_stationarity=int(sl_config["enable_non_stationarity"]),
        force_los_prob=(
            lambda prob_list: [float(prob_list[0]), float(prob_list[1])]
            if len(prob_list) == 2
            else (_ for _ in ()).throw(RuntimeError(
                f"force_los_prob must contain exactly 2 elements, got {len(prob_list)}"
            ))
        )(sl_config["force_los_prob"]),
        force_ut_speed=(
            lambda speed_list: [float(speed_list[0]), float(speed_list[1])]
            if len(speed_list) == 2
            else (_ for _ in ()).throw(RuntimeError(
                f"force_ut_speed must contain exactly 2 elements, got {len(speed_list)}"
            ))
        )(sl_config["force_ut_speed"]),
        force_indoor_ratio=float(sl_config["force_indoor_ratio"]),
        disable_pl_shadowing=int(sl_config["disable_pl_shadowing"]),
        disable_small_scale_fading=int(sl_config["disable_small_scale_fading"]),
        enable_per_tti_lsp=int(sl_config["enable_per_tti_lsp"]),
        enable_propagation_delay=int(sl_config["enable_propagation_delay"])
    )


def _read_link_level_config(ll_config: dict) -> LinkLevelConfig:
    """Read link level configuration from YAML using direct constructor."""
    delay_str = str(ll_config["delay_profile"])
    delay_profile = delay_str[0] if delay_str else 'A'

    # Use basic constructor with only supported parameters
    link_config = LinkLevelConfig(
        fast_fading_type=int(ll_config["fast_fading_type"]),
        delay_profile=delay_profile,
        delay_spread=float(ll_config["delay_spread"])
    )

    # Assign remaining parameters as attributes after instantiation
    link_config.velocity = [
        float(ll_config["velocity"][0]),
        float(ll_config["velocity"][1]),
        float(ll_config["velocity"][2])
    ]
    link_config.num_ray = int(ll_config["num_ray"])
    link_config.cfo_hz = float(ll_config["cfo_hz"])
    link_config.delay = float(ll_config["delay"])

    return link_config


def _read_simulation_config(sim_config_dict: dict) -> SimConfig:
    """Read simulation configuration from YAML using direct constructor."""
    # Use new full constructor with all parameters in one call
    return SimConfig(
        link_sim_ind=int(sim_config_dict["link_sim_ind"]),
        center_freq_hz=float(sim_config_dict["center_freq_hz"]),
        bandwidth_hz=float(sim_config_dict["bandwidth_hz"]),
        sc_spacing_hz=float(sim_config_dict["sc_spacing_hz"]),
        fft_size=int(sim_config_dict["fft_size"]),
        n_prb=int(sim_config_dict["n_prb"]),
        n_prbg=int(sim_config_dict["n_prbg"]),
        n_snapshot_per_slot=int(sim_config_dict["n_snapshot_per_slot"]),
        run_mode=int(sim_config_dict["run_mode"]),
        internal_memory_mode=int(sim_config_dict["internal_memory_mode"]),
        freq_convert_type=int(sim_config_dict["freq_convert_type"]),
        sc_sampling=int(sim_config_dict["sc_sampling"]),
        proc_sig_freq=int(sim_config_dict["proc_sig_freq"]),
        optional_cfr_dim=int(sim_config_dict["optional_cfr_dim"]),
        cpu_only_mode=int(sim_config_dict["cpu_only_mode"])
    )


def _read_antenna_panel_configs(ap_config: dict,
                                external_config: ExternalConfig) -> None:
    """Read antenna panel configurations from YAML."""
    external_config.ant_panel_config = []

    # Read panel_0 (BS panel)
    if "panel_0" in ap_config:
        panel_0 = _read_single_panel_config(ap_config["panel_0"])
        external_config.ant_panel_config.append(panel_0)

    # Read panel_1 (UE panel)
    if "panel_1" in ap_config:
        panel_1 = _read_single_panel_config(ap_config["panel_1"])
        external_config.ant_panel_config.append(panel_1)


def _read_single_panel_config(panel_config: dict) -> AntPanelConfig:
    """Read a single antenna panel configuration using direct constructor."""
    n_ant = int(panel_config["n_ant"])
    ant_model = int(panel_config["ant_model"])

    # Read and validate ant_size array
    ant_size = panel_config["ant_size"]
    if len(ant_size) != 5:
        raise RuntimeError("ant_size must have exactly 5 elements")
    ant_size_list = [int(ant_size[i]) for i in range(5)]

    # Read and validate ant_spacing array
    ant_spacing = panel_config["ant_spacing"]
    if len(ant_spacing) != 4:
        raise RuntimeError("ant_spacing must have exactly 4 elements")
    ant_spacing_list = [float(ant_spacing[i]) for i in range(4)]

    # Read and validate ant_polar_angles array
    ant_polar_angles = panel_config["ant_polar_angles"]
    if len(ant_polar_angles) != 2:
        raise RuntimeError("ant_polar_angles must have exactly 2 elements")
    ant_polar_list = [float(ant_polar_angles[i]) for i in range(2)]

    # For model 2 with direct patterns, use full constructor
    if ant_model == 2 and "ant_theta" in panel_config and "ant_phi" in panel_config:
        # Read ant_theta array
        ant_theta = panel_config["ant_theta"]
        if len(ant_theta) != 181:
            raise RuntimeError("ant_theta must have exactly 181 elements when ant_model=2")
        ant_theta_list = [float(ant_theta[i]) for i in range(181)]

        # Read ant_phi array
        ant_phi = panel_config["ant_phi"]
        if len(ant_phi) != 360:
            raise RuntimeError("ant_phi must have exactly 360 elements when ant_model=2")
        ant_phi_list = [float(ant_phi[i]) for i in range(360)]

        # Use full constructor with direct patterns - Python lists
        return AntPanelConfig(
            n_ant=n_ant,
            ant_size=ant_size_list,
            ant_spacing=ant_spacing_list,
            ant_theta=ant_theta_list,
            ant_phi=ant_phi_list,
            ant_polar_angles=ant_polar_list,
            ant_model=ant_model
        )
    else:
        # Use basic constructor for models 0 and 1 - Python lists
        return AntPanelConfig(
            n_ant=n_ant,
            ant_size=ant_size_list,
            ant_spacing=ant_spacing_list,
            ant_polar_angles=ant_polar_list,
            ant_model=ant_model
        )


def create_channel_model_from_config(
        config_file: Optional[str] = None,
        n_site: Optional[int] = None,
        n_ut: Optional[int] = None,
        scenario: Optional[Scenario] = None) -> Tuple[StatisticalChannel, SimConfig,
                                                      SystemLevelConfig]:
    """
    Create a channel model configuration from YAML file with optional
    overrides.

    Args:
        config_file: Path to YAML configuration file. If None, uses the
                    default config file.
        n_site: Override number of sites
        n_ut: Override number of UTs
        scenario: Override scenario

    Returns:
        Tuple of (chan_model, sim_config, system_config)
    """

    if config_file is None:
        # Get the directory of this script file
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # Build path relative to the script location
        config_file = os.path.join(
            script_dir,
            "../../config/statistic_channel_config.yaml")
        config_file = os.path.normpath(config_file)

    system_config, link_config, sim_config, external_config = read_config(
        config_file)

    # Apply overrides
    if n_site is not None:
        system_config.n_site = n_site
    if n_ut is not None:
        system_config.n_ut = n_ut
    if scenario is not None:
        system_config.scenario = scenario

    # Generate cells and UTs if external config is empty
    if not external_config.cell_config:
        external_config.cell_config = generate_cell_configs(
            system_config, external_config)
    if not external_config.ut_config:
        external_config.ut_config = generate_ut_configs(
            system_config, external_config)

    # Seed initial UT configs on function attributes to avoid globals
    per_slot_update._initial_ut_config = external_config.ut_config

    # Create the channel model
    chan_model = StatisticalChannel(
        sim_config=sim_config,
        system_level_config=system_config,
        link_level_config=link_config,
        external_config=external_config
    )

    return chan_model, sim_config, system_config


def generate_cell_configs(system_config: SystemLevelConfig,
                          external_config: ExternalConfig) -> List[CellParam]:
    """Generate cell configurations based on system config."""
    n_sites = system_config.n_site
    n_sectors_per_site = system_config.n_sector_per_site
    cell_configs = []

    # Get ISD value based on scenario
    if system_config.scenario == Scenario.UMa:
        isd = 500.0
    elif system_config.scenario == Scenario.UMi:
        isd = 200.0
    elif system_config.scenario == Scenario.RMa:
        isd = system_config.isd
    else:
        raise RuntimeError(f"Invalid scenario: {system_config.scenario}")

    # Calculate cell radius for hexagonal layout
    cell_radius = isd / np.sqrt(3)

    for site_id in range(n_sites):
        # Calculate site center position using proper hexagonal topology
        # This matches the C++ logic in sls_chan_topology.cu
        if site_id < 1:
            # Center site
            site_x = 0.0
            site_y = 0.0
        elif site_id < 7:
            # First ring (6 sites) - hexagonal pattern around center
            site_angle = (site_id - 1) * np.pi / 3.0 + np.pi / 6.0
            site_x = np.cos(site_angle) * isd
            site_y = np.sin(site_angle) * isd
        elif site_id < 19:
            # Second ring (12 sites)
            ring_angle = (site_id - 7) * np.pi / 6.0
            if site_id % 2 == 1:
                site_x = np.cos(ring_angle) * 3.0 * cell_radius
                site_y = np.sin(ring_angle) * 3.0 * cell_radius
            else:
                site_x = np.cos(ring_angle) * 2.0 * isd
                site_y = np.sin(ring_angle) * 2.0 * isd
        else:
            # For sites beyond the second ring, use a simplified pattern
            # This extends the hexagonal pattern for larger deployments
            ring_num = int(np.sqrt(site_id))
            angle_step = 2 * np.pi / (6 * ring_num) if ring_num > 0 else 0
            site_angle = (site_id - (ring_num * ring_num)) * angle_step
            site_x = np.cos(site_angle) * ring_num * isd
            site_y = np.sin(site_angle) * ring_num * isd

        for sector_id in range(n_sectors_per_site):
            cid = site_id * n_sectors_per_site + sector_id

            # Calculate sector orientation angle (120-degree sectors)
            sector_angle = sector_id * 2 * np.pi / 3 + np.pi / 6

            # For multi-sector sites, cells are co-located at site center
            # but with different antenna orientations
            x = site_x
            y = site_y
            z = 25.0  # Base station height

            # Create coordinate using direct constructor
            coord = Coordinate(x, y, z)

            # Create cell config using constructor with Python list
            cell_config = CellParam(
                cid=cid,
                site_id=site_id,
                loc=coord,
                ant_panel_idx=0,  # Use the first antenna config
                ant_panel_orientation=[sector_angle, 0.0, 0.0]  # Sector-specific orientation
            )
            cell_configs.append(cell_config)

    return cell_configs


def generate_ut_configs(system_config: SystemLevelConfig,
                        external_config: ExternalConfig) -> List[UtParamCfg]:
    """Generate UT configurations based on system config - matches C++ implementation."""
    n_uts = system_config.n_ut
    ut_configs = []

    # Get cell configurations (should be generated first)
    if not external_config.cell_config:
        raise RuntimeError("Cell configs must be generated before UT configs")

    n_cells = len(external_config.cell_config)

    # Calculate simulation boundary limits based on cell locations
    # This matches C++ boundary checking logic
    cell_x_coords = [cell.loc.x for cell in external_config.cell_config]
    cell_y_coords = [cell.loc.y for cell in external_config.cell_config]

    # Calculate indoor UT percentage based on force_indoor_ratio
    if (system_config.force_indoor_ratio >= 0 and
       system_config.force_indoor_ratio <= 1):
        indoor_ut_percent = system_config.force_indoor_ratio
    else:
        # Default indoor percentages by scenario (typical values)
        if system_config.scenario == Scenario.UMa:
            indoor_ut_percent = 0.8  # 80% indoor in urban macro
            isd = 500.0
            min_bs_ue_dist = 10.0
        elif system_config.scenario == Scenario.UMi:
            indoor_ut_percent = 0.5  # 50% indoor in urban micro
            isd = 200.0
            min_bs_ue_dist = 35.0
        else:
            indoor_ut_percent = 0.2  # 20% indoor for rural
            isd = system_config.isd
            min_bs_ue_dist = 35.0

    cell_radius = isd / np.sqrt(3)
    min_x = min(cell_x_coords) - cell_radius
    max_x = max(cell_x_coords) + cell_radius
    min_y = min(cell_y_coords) - cell_radius
    max_y = max(cell_y_coords) + cell_radius

    uid = 0
    attempts = 0
    max_attempts = n_uts * 10  # Prevent infinite loops

    while uid < n_uts and attempts < max_attempts:
        attempts += 1

        # Select a random sector/cell for this UT
        # This matches C++ logic where UTs are distributed around cells
        sec_idx = np.random.randint(0, n_cells)
        cell_param = external_config.cell_config[sec_idx]

        # Generate UE location following C++ logic

        # Generate random angle within sector coverage (-60° to +60° from sector orientation)
        random_angle = 2.0 * np.pi * np.random.uniform() / 3.0 - np.pi / 3.0

        # Generate random distance using sqrt for better spatial distribution
        random_distance = ((cell_radius - min_bs_ue_dist) * np.sqrt(np.random.uniform()) +
                           min_bs_ue_dist)

        # Check if the UE is within the hexagonal cell boundary
        temp_angle = abs(random_angle)
        if temp_angle > np.pi / 6.0:
            temp_angle = np.pi / 3.0 - temp_angle
        # temp_angle should be in the range of [0, pi/6]
        max_distance_angle = isd / 2.0 / np.cos(temp_angle)

        if random_distance > max_distance_angle:
            continue  # Position is outside hexagonal cell, try again

        # Add sector orientation (antenna panel orientation)
        random_angle += cell_param.ant_panel_orientation[1] * np.pi / 180.0

        # Calculate UT position relative to cell location
        x = np.cos(random_angle) * random_distance + cell_param.loc.x
        y = np.sin(random_angle) * random_distance + cell_param.loc.y

        # Boundary checking - ensure UT is within simulation area
        if x < min_x or x > max_x or y < min_y or y > max_y:
            continue  # Try again with different position

        # Valid position found - proceed with UT generation
        # Determine indoor/outdoor status
        outdoor_ind = 1 if np.random.uniform(0, 1) > indoor_ut_percent else 0

        # Calculate height based on scenario and indoor/outdoor status
        if (system_config.scenario == Scenario.UMa or
           system_config.scenario == Scenario.UMi):
            # Generate random number of total floors (N_fl) between 4 and 8 (inclusive)
            # [4, 8] - multiply by 5 to include 8
            n_floors_total = 4 + int(np.random.uniform(0, 1) * 5)

            if outdoor_ind == 1:
                # Outdoor UT: ground floor (floor 1)
                n_floor = 1
            else:
                # Indoor UT: random floor between 1 and N_fl (inclusive)
                n_floor = 1 + int(np.random.uniform(0, 1) * n_floors_total)

            # Height calculation: 3m per floor + 1.5m base height
            z = 3.0 * (n_floor - 1) + 1.5
        else:
            # For other scenarios: fixed height
            z = 1.5

        # Generate velocity vector
        # Random direction
        velocity_direction = 2.0 * np.pi * np.random.uniform(0, 1)

        # Speed based on indoor/outdoor status and scenario
        if outdoor_ind == 1:  # Outdoor UT
            if system_config.force_ut_speed[0] >= 0:
                speed_kmh = system_config.force_ut_speed[0]
            else:
                # Default outdoor speeds
                if system_config.scenario == Scenario.RMa:
                    speed_kmh = 60.0  # Rural: higher speed
                else:
                    speed_kmh = 3.0   # Urban: walking speed
        else:  # Indoor UT
            if system_config.force_ut_speed[1] >= 0:
                speed_kmh = system_config.force_ut_speed[1]
            else:
                speed_kmh = 3.0  # Indoor: walking speed

        # Convert km/h to m/s
        speed_ms = speed_kmh / 3.6

        # Calculate velocity components
        velocity_x = speed_ms * np.cos(velocity_direction)
        velocity_y = speed_ms * np.sin(velocity_direction)
        velocity_z = 0.0

        # Calculate antenna orientation aligned with movement direction (matching C++)
        # Per 3GPP TR 38.901: UE antenna azimuth aligned with velocity direction
        # ant_panel_orientation = [zenith, azimuth, slant]
        orientation_azimuth_deg = np.arctan2(velocity_y, velocity_x) * 180.0 / np.pi
        # No need to wrap to [0, 360) - atan2 returns [-180, 180] which is valid

        ant_orientation = [
            90.0,                      # [0] zenith/downtilt: fixed at 90°
            orientation_azimuth_deg,   # [1] azimuth: aligned with movement direction
            0.0                        # [2] slant: fixed at 0°
        ]

        # Create coordinate using direct constructor
        coord = Coordinate(x, y, z)

        # Create UT config using constructor with Python lists
        ut_config = UtParamCfg(
            uid=uid,
            loc=coord,
            outdoor_ind=outdoor_ind,
            ant_panel_idx=1,  # Use the second antenna config (UE panel)
            ant_panel_orientation=ant_orientation,
            velocity=[velocity_x, velocity_y, velocity_z]
        )
        ut_configs.append(ut_config)
        uid += 1  # Move to next UT only after successful placement

    # Check if we generated all requested UTs
    if uid < n_uts:
        print(f"Warning: Could only generate {uid} UTs out of {n_uts} requested "
              f"after {attempts} attempts. Consider relaxing boundary constraints.")

    return ut_configs


def setup_simulation_parameters(sim_config, system_config):
    """Setup initial simulation parameters."""
    n_active_cells = system_config.n_site * system_config.n_sector_per_site
    n_snapshots = sim_config.n_snapshot_per_slot
    n_bs_ant = 4  # Default BS antenna count
    n_ut_ant = 4  # Default UE antenna count
    max_taps = 24

    return n_active_cells, n_snapshots, n_bs_ant, n_ut_ant, max_taps


def update_ut_mobility(system_config, prev_locations, prev_velocities, tslot_s: float = 0.0005):
    """Update UT locations and velocities for current TTI using random directions.

    Args:
        system_config: System config object with n_ut.
        prev_locations: np.ndarray shape (n_ut, 3) of previous UT [x,y,z].
        prev_velocities: np.ndarray shape (n_ut, 3) of previous UT [vx,vy,vz].
        tslot_s: slot duration in seconds (default 0.5 ms).

    Returns:
        (new_locations, new_velocities) as np.ndarrays.
    """
    # alwasy use total number of UTs
    # TODO: add support for only updating selected UTs
    n_uts = system_config.n_ut

    ut_new_locations = np.zeros((n_uts, 3), dtype=np.float32)
    ut_new_velocities = np.zeros((n_uts, 3), dtype=np.float32)

    speed_mps = 0.833  # constant speed in m/s, 3 km/h
    # TODO: add support for variable speed
    for uid in range(n_uts):
        theta = np.random.uniform(0.0, 2.0 * np.pi)
        vx = speed_mps * np.cos(theta)
        vy = speed_mps * np.sin(theta)
        # Zero vertical motion per 38.901
        vz = 0.0

        dx = vx * tslot_s
        dy = vy * tslot_s

        ut_new_locations[uid, 0] = prev_locations[uid, 0] + dx
        ut_new_locations[uid, 1] = prev_locations[uid, 1] + dy
        ut_new_locations[uid, 2] = prev_locations[uid, 2]

        ut_new_velocities[uid, 0] = vx
        ut_new_velocities[uid, 1] = vy
        ut_new_velocities[uid, 2] = vz

    return ut_new_locations, ut_new_velocities


def setup_active_links(n_active_cells, n_uts):
    """Setup active cells and UTs configuration."""
    active_cells = list(range(n_active_cells))

    # Generate active UTs - each cell connects to UT 0 and UT 1
    active_uts = []
    for cell_id in range(n_active_cells):
        cell_uts = list(range(min(10, n_uts)))
        active_uts.append(cell_uts)

    total_active_links = sum(len(cell_uts) for cell_uts in active_uts)

    return active_cells, active_uts, total_active_links


def allocate_cir_arrays(sim_config, active_cells, active_uts,
                        n_snapshots, n_ut_ant, n_bs_ant, max_taps):
    """Allocate CIR arrays for all cells."""
    cir_coe_per_cell = []
    cir_norm_delay_per_cell = []
    cir_n_taps_per_cell = []

    print(f"  Allocating per-cell arrays for {len(active_cells)} cells")

    for cell_id in range(len(active_cells)):
        n_uts_this_cell = len(active_uts[cell_id])
        if n_uts_this_cell == 0:
            print(f"  Warning: Cell {cell_id} has no active UTs, skipping")
            continue

        print(f"  Allocating arrays for Cell {cell_id}: {n_uts_this_cell} UTs")

        if sim_config.cpu_only_mode == 1:
            # Allocate NumPy arrays for CPU-only mode
            cir_coe_np = np.zeros((n_uts_this_cell, n_snapshots, n_ut_ant,
                                   n_bs_ant, max_taps), dtype=np.complex64, order='C')
            cir_norm_delay_np = np.zeros((n_uts_this_cell, max_taps), dtype=np.uint16, order='C')
            cir_n_taps_np = np.zeros(n_uts_this_cell, dtype=np.uint16, order='C')

            cir_coe_per_cell.append(cir_coe_np)
            cir_norm_delay_per_cell.append(cir_norm_delay_np)
            cir_n_taps_per_cell.append(cir_n_taps_np)
        else:
            # Allocate CuPy arrays for GPU mode
            cir_coe_cupy = cp.zeros((n_uts_this_cell, n_snapshots, n_ut_ant,
                                    n_bs_ant, max_taps), dtype=cp.complex64, order='C')
            cir_norm_delay_cupy = cp.zeros((n_uts_this_cell, max_taps),
                                           dtype=cp.uint16, order='C')
            cir_n_taps_cupy = cp.zeros(n_uts_this_cell, dtype=cp.uint16, order='C')

            # Synchronize and verify GPU pointers
            cp.cuda.Device().synchronize()

            cir_coe_ptr = cir_coe_cupy.data.ptr
            cir_norm_delay_ptr = cir_norm_delay_cupy.data.ptr
            cir_n_taps_ptr = cir_n_taps_cupy.data.ptr

            print(f"    Cell {cell_id} GPU pointers:")
            print(f"      cir_coe: 0x{cir_coe_ptr:x}")
            print(f"      cir_norm_delay: 0x{cir_norm_delay_ptr:x}")
            print(f"      cir_n_taps: 0x{cir_n_taps_ptr:x}")

            assert cir_coe_ptr != 0, f"Null cir_coe pointer for cell {cell_id}"
            assert cir_norm_delay_ptr != 0, f"Null cir_norm_delay pointer for cell {cell_id}"
            assert cir_n_taps_ptr != 0, f"Null cir_n_taps pointer for cell {cell_id}"

            cir_coe_per_cell.append(cir_coe_cupy)
            cir_norm_delay_per_cell.append(cir_norm_delay_cupy)
            cir_n_taps_per_cell.append(cir_n_taps_cupy)

    return cir_coe_per_cell, cir_norm_delay_per_cell, cir_n_taps_per_cell


def allocate_cfr_arrays(sim_config, active_cells, active_uts, n_snapshots, n_ut_ant, n_bs_ant):
    """Allocate CFR arrays based on run mode."""
    cfr_prbg_per_cell = []
    cfr_sc_per_cell = []

    if sim_config.run_mode >= 1:
        print(f"  Allocating CFR arrays for run mode {sim_config.run_mode}")

        for cell_id in range(len(active_cells)):
            n_uts_this_cell = len(active_uts[cell_id])
            if n_uts_this_cell == 0:
                print(f"  Warning: Cell {cell_id} has no active UTs, skipping CFR allocation")
                continue

            print(f"  Allocating CFR arrays for Cell {cell_id}: {n_uts_this_cell} UTs")

            # CFR PRBG allocation (used in run modes 1 and 3 only)
            if sim_config.run_mode in (1, 3):
                if sim_config.cpu_only_mode == 1:
                    cfr_prbg_np = np.zeros((n_uts_this_cell, n_snapshots, n_ut_ant,
                                            n_bs_ant, sim_config.n_prbg),
                                           dtype=np.complex64, order='C')
                    cfr_prbg_per_cell.append(cfr_prbg_np)
                else:
                    cfr_prbg_cupy = cp.zeros((n_uts_this_cell, n_snapshots, n_ut_ant,
                                              n_bs_ant, sim_config.n_prbg),
                                             dtype=cp.complex64, order='C')
                    cfr_prbg_per_cell.append(cfr_prbg_cupy)

                    cfr_prbg_ptr = cfr_prbg_cupy.data.ptr
                    print(f"    Cell {cell_id} CFR PRBG pointer: 0x{cfr_prbg_ptr:x}")
                    assert cfr_prbg_ptr != 0, f"Null cfr_prbg pointer for cell {cell_id}"

            # CFR SC allocation (used in run modes 2, 3, and 4 only)
            if sim_config.run_mode >= 2:
                if sim_config.run_mode == 4:
                    n_subcarriers = sim_config.fft_size  # Mode 4: Use N_FFT subcarriers
                else:
                    n_subcarriers = sim_config.n_prb * 12  # Modes 2, 3: Use N_PRB * 12
                if sim_config.cpu_only_mode == 1:
                    cfr_sc_np = np.zeros((n_uts_this_cell, n_snapshots, n_ut_ant,
                                          n_bs_ant, n_subcarriers),
                                         dtype=np.complex64,
                                         order='C')
                    cfr_sc_per_cell.append(cfr_sc_np)
                else:
                    cfr_sc_cupy = cp.zeros((n_uts_this_cell, n_snapshots, n_ut_ant,
                                            n_bs_ant, n_subcarriers),
                                           dtype=cp.complex64,
                                           order='C')
                    cfr_sc_per_cell.append(cfr_sc_cupy)

                    cfr_sc_ptr = cfr_sc_cupy.data.ptr
                    print(f"    Cell {cell_id} CFR SC pointer: 0x{cfr_sc_ptr:x}")
                    assert cfr_sc_ptr != 0, f"Null cfr_sc pointer for cell {cell_id}"

    return cfr_prbg_per_cell, cfr_sc_per_cell


def print_buffer_info(sim_config, active_cells, active_uts, total_active_links,
                      cir_coe_per_cell, cir_norm_delay_per_cell, cir_n_taps_per_cell,
                      cfr_prbg_per_cell, cfr_sc_per_cell):
    """Print detailed buffer allocation information."""
    print("  Active links configuration:")
    print(f"    active_cells: {active_cells}")
    print(f"    active_uts: {active_uts}")
    print(f"    total_active_links: {total_active_links}")
    link_list = [(cell_id, active_uts[cell_id]) for cell_id in range(len(active_cells))]
    print(f"    This creates links: {link_list}")

    print("  Final buffer info:")
    print(f"    Total links: {total_active_links}")
    print(f"    Number of allocated per-cell arrays: {len(cir_coe_per_cell)}")

    for i, arr in enumerate(cir_coe_per_cell):
        print(f"    Cell {i} cir_coe shape: {arr.shape}")
        print(f"    Cell {i} cir_norm_delay shape: {cir_norm_delay_per_cell[i].shape}")
        print(f"    Cell {i} cir_n_taps shape: {cir_n_taps_per_cell[i].shape}")

        # cfr_prbg is only allocated for run_mode 1 or 3
        if sim_config.run_mode in (1, 3) and len(cfr_prbg_per_cell) > i:
            print(f"    Cell {i} cfr_prbg shape: {cfr_prbg_per_cell[i].shape}")
        elif sim_config.run_mode == 4:
            print(f"    Cell {i} cfr_prbg: NOT ALLOCATED (run mode 4 uses only SC)")

        if sim_config.run_mode >= 2:
            print(f"    Cell {i} cfr_sc shape: {cfr_sc_per_cell[i].shape}")


def per_slot_update(sim_config, system_config,
                    n_active_cells, n_snapshots, n_ut_ant, n_bs_ant,
                    max_taps, tti_idx):
    """Perform per-TTI updates before running channel simulation.

    This computes active links each slot, reuses previously allocated
    buffers when the active sets are unchanged, and reallocates on change.
    """

    # Update UT mobility with persistent state
    if tti_idx == 0:
        # Slot 0: seed from initial UE dropping if available; otherwise default
        if hasattr(per_slot_update, '_initial_ut_config') and \
           len(per_slot_update._initial_ut_config) == system_config.n_ut:
            locs = np.zeros((system_config.n_ut, 3), dtype=np.float32)
            vels = np.zeros((system_config.n_ut, 3), dtype=np.float32)
            for i, ut in enumerate(per_slot_update._initial_ut_config):
                locs[i, 0] = ut.loc.x
                locs[i, 1] = ut.loc.y
                locs[i, 2] = ut.loc.z
                vels[i, 0] = ut.velocity[0]
                vels[i, 1] = ut.velocity[1]
                vels[i, 2] = ut.velocity[2]
            per_slot_update._prev_locs = locs
            per_slot_update._prev_vels = vels
        else:
            raise ValueError("Initial UT configuration not found")
    else:
        # Subsequent slots: ensure previous state exists
        if not hasattr(per_slot_update, "_prev_locs") or not hasattr(per_slot_update, "_prev_vels"):
            raise ValueError("Previous UT state not found")

    ut_new_locations, ut_new_velocities = update_ut_mobility(system_config,
                                                             per_slot_update._prev_locs,
                                                             per_slot_update._prev_vels,
                                                             tti_idx)
    per_slot_update._prev_locs = ut_new_locations.copy()
    per_slot_update._prev_vels = ut_new_velocities.copy()

    # Determine active links configuration for this TTI
    active_cells, active_uts, _ = setup_active_links(
        n_active_cells, system_config.n_ut)

    # Check if active sets have changed; reuse buffers when possible
    buffer_key = (
        sim_config.cpu_only_mode,
        sim_config.run_mode,
        sim_config.n_prb,
        sim_config.n_prbg,
        sim_config.fft_size,
        n_snapshots,
        n_ut_ant,
        n_bs_ant,
        max_taps,
    )
    reuse_buffers = (
        hasattr(per_slot_update, "_prev_active_cells") and
        hasattr(per_slot_update, "_prev_active_uts") and
        getattr(per_slot_update, "_buffer_key", None) == buffer_key and
        per_slot_update._prev_active_cells == active_cells and
        per_slot_update._prev_active_uts == active_uts and
        hasattr(per_slot_update, "_cir_coe_per_cell") and
        hasattr(per_slot_update, "_cir_norm_delay_per_cell") and
        hasattr(per_slot_update, "_cir_n_taps_per_cell") and
        hasattr(per_slot_update, "_cfr_prbg_per_cell") and
        hasattr(per_slot_update, "_cfr_sc_per_cell")
    )

    if reuse_buffers:
        cir_coe_per_cell = per_slot_update._cir_coe_per_cell
        cir_norm_delay_per_cell = per_slot_update._cir_norm_delay_per_cell
        cir_n_taps_per_cell = per_slot_update._cir_n_taps_per_cell
        cfr_prbg_per_cell = per_slot_update._cfr_prbg_per_cell
        cfr_sc_per_cell = per_slot_update._cfr_sc_per_cell
    else:
        # Allocate CIR arrays
        (cir_coe_per_cell,
         cir_norm_delay_per_cell,
         cir_n_taps_per_cell) = allocate_cir_arrays(
            sim_config, active_cells, active_uts,
            n_snapshots, n_ut_ant, n_bs_ant, max_taps)

        # Allocate CFR arrays
        cfr_prbg_per_cell, cfr_sc_per_cell = allocate_cfr_arrays(
            sim_config, active_cells, active_uts,
            n_snapshots, n_ut_ant, n_bs_ant)

        # Cache for reuse on subsequent TTIs when active sets match
        per_slot_update._cir_coe_per_cell = cir_coe_per_cell
        per_slot_update._cir_norm_delay_per_cell = cir_norm_delay_per_cell
        per_slot_update._cir_n_taps_per_cell = cir_n_taps_per_cell
        per_slot_update._cfr_prbg_per_cell = cfr_prbg_per_cell
        per_slot_update._cfr_sc_per_cell = cfr_sc_per_cell
        per_slot_update._prev_active_cells = active_cells
        per_slot_update._prev_active_uts = active_uts
        per_slot_update._buffer_key = buffer_key

    # Final GPU synchronization (GPU mode only)
    if sim_config.cpu_only_mode == 0:
        cp.cuda.get_current_stream().synchronize()

    # Calculate total active links
    total_active_links = sum(len(cell_uts) for cell_uts in active_uts)

    # Print buffer information
    print_buffer_info(sim_config, active_cells, active_uts, total_active_links,
                      cir_coe_per_cell, cir_norm_delay_per_cell, cir_n_taps_per_cell,
                      cfr_prbg_per_cell, cfr_sc_per_cell)

    return (active_cells, active_uts,
            ut_new_locations, ut_new_velocities,
            cir_coe_per_cell, cir_norm_delay_per_cell, cir_n_taps_per_cell,
            cfr_prbg_per_cell, cfr_sc_per_cell, total_active_links)


def run_channel_model_step(chan_model, tti, sim_config, active_cells, active_uts,
                           ut_new_locations, ut_new_velocities, cir_coe_per_cell,
                           cir_norm_delay_per_cell, cir_n_taps_per_cell,
                           cfr_prbg_per_cell, cfr_sc_per_cell):
    """Execute the channel model for one TTI.

    Returns:
        float: Execution time in milliseconds
    """

    # Prepare CFR arrays based on run mode
    cfr_prbg_args = (cfr_prbg_per_cell if sim_config.run_mode in (1, 3) else [])
    cfr_sc_args = cfr_sc_per_cell if sim_config.run_mode >= 2 else []

    print("  Passing to channel model:")
    print(f"    cfr_prbg_args: {len(cfr_prbg_args)} arrays "
          f"(mode 4 = 0, others = {len(active_cells)})")
    print(f"    cfr_sc_args: {len(cfr_sc_args)} arrays")

    # Start timing
    start_time = time.perf_counter()

    chan_model.run(
        ref_time=tti * 0.5e-3,  # TTI duration = 0.5ms
        continuous_fading=1,
        active_cell=active_cells,
        active_ut=active_uts,
        ut_new_loc=ut_new_locations,
        ut_new_velocity=ut_new_velocities,
        cir_coe=cir_coe_per_cell,
        cir_norm_delay=cir_norm_delay_per_cell,
        cir_n_taps=cir_n_taps_per_cell,
        cfr_prbg=cfr_prbg_args,
        cfr_sc=cfr_sc_args
    )

    # CRITICAL: Ensure all GPU kernels complete before Python reads results (GPU mode only)
    if sim_config.cpu_only_mode == 0:
        print("  Synchronizing GPU kernels before reading results...")
        chan_model._stream.synchronize()

    # End timing after synchronization
    end_time = time.perf_counter()
    elapsed_ms = (end_time - start_time) * 1000.0

    return elapsed_ms


def process_cir_statistics(cir_coe_per_cell, cir_n_taps_per_cell, cir_norm_delay_per_cell,
                           active_uts, max_taps):
    """Process and print CIR statistics."""
    print("  CIR Statistics:")

    all_n_taps = []
    total_non_zero_count = 0
    total_elements = 0
    first_non_zero_found = False

    for cell_id, (cir_coe_arr, cir_n_taps_arr) in enumerate(
        zip(cir_coe_per_cell, cir_n_taps_per_cell)
    ):
        # Collect tap counts and validate
        try:
            is_cupy = hasattr(cir_n_taps_arr, 'get')
        except Exception:
            is_cupy = False

        cell_n_taps = cir_n_taps_arr.get() if is_cupy else cir_n_taps_arr

        # Sanity check: tap counts should be finite and <= 24
        MAX_TAPS = 24
        invalid_mask = ~np.isfinite(cell_n_taps) | (cell_n_taps > MAX_TAPS) | (cell_n_taps < 0)
        valid_n_taps = cell_n_taps[~invalid_mask]

        # Report any invalid values
        if np.any(invalid_mask):
            invalid_count = np.sum(invalid_mask)
            invalid_values = cell_n_taps[invalid_mask]
            print(f"    ERROR: Cell {cell_id} has {invalid_count} invalid tap counts!")
            print(f"    Expected: 0 <= tap_count <= {MAX_TAPS}")
            print(f"    Invalid values: {invalid_values}")
            # Also show some statistics about the invalid values
            finite_invalid = invalid_values[np.isfinite(invalid_values)]
            if len(finite_invalid) > 0:
                print(f"    Invalid range: [{np.min(finite_invalid):.2e}, "
                      f"{np.max(finite_invalid):.2e}]")

        all_n_taps.extend(valid_n_taps)

        # Count non-zero elements
        if is_cupy:
            import cupy as cp  # type: ignore
            non_zero_mask = cp.abs(cir_coe_arr) > 1e-20
            cell_non_zero_count = int(cp.sum(non_zero_mask))
            total_elements += cir_coe_arr.size
        else:
            non_zero_mask = np.abs(cir_coe_arr) > 1e-20
            cell_non_zero_count = int(np.sum(non_zero_mask))
            total_elements += cir_coe_arr.size
        total_non_zero_count += cell_non_zero_count

        # Calculate per-cell statistics using safe summation
        avg_n_taps_cell = (np.sum(valid_n_taps, dtype=np.float64) / len(valid_n_taps)
                           if len(valid_n_taps) > 0 else 0)
        n_uts_this_cell = len(active_uts[cell_id])

        print(f"    Cell {cell_id} (N_UTs={n_uts_this_cell}, "
              f"max_taps={max_taps}): "
              f"non-zero={cell_non_zero_count}/{cir_coe_arr.size}")
        print(f"    Cell {cell_id} avg_taps: {avg_n_taps_cell:.1f}, "
              f"tap_details: {cell_n_taps}")

        # Find first non-zero element for debugging
        if not first_non_zero_found:
            if is_cupy:
                import cupy as cp  # type: ignore # noqa: F401
                non_zero_indices = cp.where(non_zero_mask)
                if len(non_zero_indices[0]) > 0:
                    first_idx = tuple(int(idx[0]) for idx in non_zero_indices)
                    first_value = cir_coe_arr[first_idx]
                    print(f"    First non-zero at Cell {cell_id}, index {first_idx}: {first_value}")
                    first_non_zero_found = True
            else:
                non_zero_indices = np.where(non_zero_mask)
                if len(non_zero_indices[0]) > 0:
                    first_idx = tuple(int(idx[0]) for idx in non_zero_indices)
                    first_value = cir_coe_arr[first_idx]
                    print(f"    First non-zero at Cell {cell_id}, index {first_idx}: {first_value}")
                    first_non_zero_found = True

        # Print the first normalized delay
        if cir_norm_delay_per_cell and len(cir_norm_delay_per_cell) > 0:
            first_cell_delays = cir_norm_delay_per_cell[cell_id]  # per-cell delay array
            if first_cell_delays.size > 0:
                delays_cpu = (first_cell_delays.get()
                              if hasattr(first_cell_delays, 'get') else first_cell_delays)
                first_norm_delay = delays_cpu[0, 0]  # First UT, first tap
                print(f"First normalized delay of cell {cell_id}: {first_norm_delay}")
            else:
                print(f"First normalized delay of cell {cell_id}: No data available")
        else:
            print(f"First normalized delay of cell {cell_id}: No cells available")

    # Calculate overall statistics using safe summation
    avg_n_taps = (np.sum(all_n_taps, dtype=np.float64) / len(all_n_taps)
                  if all_n_taps else 0)

    print(f"    CIR Summary - Average number of taps: {avg_n_taps}")
    print(f"    CIR Summary - Non-zero elements across all cells: "
          f"{total_non_zero_count} out of {total_elements}")


def process_cfr_statistics(sim_config, cfr_prbg_per_cell, cfr_sc_per_cell):
    """Process and print CFR statistics."""

    # CFR PRBG Statistics
    if cfr_prbg_per_cell:
        print("  CFR PRBG Statistics:")
        for cell_id, cfr_prbg_arr in enumerate(cfr_prbg_per_cell):
            if hasattr(cfr_prbg_arr, 'get'):
                import cupy as cp  # type: ignore
                non_zero_count = int(cp.sum(cp.abs(cfr_prbg_arr) > 1e-20))
                total = cfr_prbg_arr.size
            else:
                non_zero_count = int(np.sum(np.abs(cfr_prbg_arr) > 1e-20))
                total = cfr_prbg_arr.size

            print(f"    Cell {cell_id} (Mode {sim_config.run_mode}, N_PRBG={sim_config.n_prbg}): "
                  f"non-zero={non_zero_count}/{total}")

    # CFR SC Statistics
    if cfr_sc_per_cell:
        print("  CFR SC Statistics:")
        for cell_id, cfr_sc_arr in enumerate(cfr_sc_per_cell):
            if hasattr(cfr_sc_arr, 'get'):
                import cupy as cp  # type: ignore
                non_zero_count = int(cp.sum(cp.abs(cfr_sc_arr) > 1e-20))
                total = cfr_sc_arr.size
            else:
                non_zero_count = int(np.sum(np.abs(cfr_sc_arr) > 1e-20))
                total = cfr_sc_arr.size

            print(f"    Cell {cell_id} (Mode {sim_config.run_mode}, N_FFT={sim_config.fft_size}): "
                  f"non-zero={non_zero_count}/{total}")

    else:
        print("  CFR PRBG Statistics: SKIPPED (run mode 4 doesn't use PRBG)")


def dump_final_statistics(chan_model, system_config, active_cells):
    """Dump final simulation statistics."""
    print("\nDumping final statistics...")

    # LOS/NLOS statistics
    n_sectors = system_config.n_site * system_config.n_sector_per_site
    los_nlos_stats = np.zeros((n_sectors, system_config.n_ut), dtype=np.float32)
    chan_model.dump_los_nlos_stats(los_nlos_stats)
    print(f"LOS ratio: {np.mean(los_nlos_stats):.2f}")

    # Pathloss shadowing statistics for ALL UTs (full range)
    all_uts = list(range(system_config.n_ut))  # Use all UTs from 0 to n_ut-1
    pl_sf_stats = np.zeros((len(active_cells), len(all_uts)), dtype=np.float32)

    print(f"Dumping pathloss shadowing for {len(active_cells)} cells and {len(all_uts)} UTs...")
    chan_model.dump_pl_sf_stats(
        pl_sf=pl_sf_stats,
        active_cell=np.array(active_cells, dtype=np.int32),
        active_ut=np.array(all_uts, dtype=np.int32)
    )
    # convert to minus dB
    pl_sf_stats = -pl_sf_stats
    # Print basic statistics
    valid_pl_sf = pl_sf_stats[~np.isnan(pl_sf_stats)]
    valid_pl_sf = valid_pl_sf[np.isfinite(valid_pl_sf)]
    print(f"Total pathloss shadowing samples: {len(valid_pl_sf)} "
          f"(from {len(active_cells)} cells × {len(all_uts)} UTs)")
    print(f"Average pathloss shadowing: {np.mean(valid_pl_sf):.1f} dB")
    print(f"Pathloss shadowing std: {np.std(valid_pl_sf):.1f} dB")
    print(f"Pathloss shadowing range: [{np.min(valid_pl_sf):.1f}, {np.max(valid_pl_sf):.1f}] dB")

    # Additional percentile statistics
    percentiles = [5, 10, 25, 50, 75, 90, 95]
    percentile_values = np.percentile(valid_pl_sf, percentiles)
    print("Pathloss shadowing percentiles:")
    for p, val in zip(percentiles, percentile_values):
        print(f"  P{p}: {val:.1f} dB")


def run_channel_simulation(chan_model, sim_config, system_config, n_tti):
    """Run channel simulation for specified number of TTIs."""

    # Setup simulation parameters
    n_active_cells, n_snapshots, n_bs_ant, n_ut_ant, max_taps = (
        setup_simulation_parameters(sim_config, system_config))

    # Timing statistics
    tti_times = []
    total_start_time = time.perf_counter()

    # Simulation loop
    last_active_cells = []
    for tti_idx in range(n_tti):
        tti_start_time = time.perf_counter()
        print(f"\n{'=' * 80}")
        print(f"Running TTI {tti_idx}")
        print(f"{'=' * 80}")
        try:
            # Per-slot update: mobility, allocations, and buffer preparation
            (active_cells, active_uts,
             ut_new_locations, ut_new_velocities,
             cir_coe_per_cell, cir_norm_delay_per_cell, cir_n_taps_per_cell,
             cfr_prbg_per_cell, cfr_sc_per_cell, total_active_links) = per_slot_update(
                sim_config, system_config,
                n_active_cells, n_snapshots, n_ut_ant, n_bs_ant, max_taps,
                tti_idx)
            last_active_cells = active_cells

            # Run the channel model (returns execution time)
            run_channel_model_step(
                chan_model, tti_idx, sim_config, active_cells, active_uts,
                ut_new_locations, ut_new_velocities, cir_coe_per_cell,
                cir_norm_delay_per_cell, cir_n_taps_per_cell,
                cfr_prbg_per_cell, cfr_sc_per_cell)

            # Process results
            print(f"  Generated channels for {total_active_links} links "
                  f"across {len(active_cells)} cells")

            # Process CIR statistics
            process_cir_statistics(
                cir_coe_per_cell, cir_n_taps_per_cell,
                cir_norm_delay_per_cell,
                active_uts, max_taps)

            # Process CFR statistics
            process_cfr_statistics(sim_config, cfr_prbg_per_cell,
                                   cfr_sc_per_cell)

            # Additional GPU synchronization to ensure data consistency across TTIs (GPU mode only)
            if sim_config.cpu_only_mode == 0:
                print("  Final GPU sync for TTI completion...")
                cp.cuda.get_current_stream().synchronize()

            # Calculate TTI timing
            tti_end_time = time.perf_counter()
            tti_elapsed_ms = (tti_end_time - tti_start_time) * 1000.0
            tti_times.append(tti_elapsed_ms)

        except Exception as e:
            print(f"Error running simulation at TTI {tti_idx}: {e}")
            raise  # Re-raise to help with debugging

    # Calculate total simulation time
    total_end_time = time.perf_counter()
    total_elapsed_ms = (total_end_time - total_start_time) * 1000.0
    total_elapsed_s = total_elapsed_ms / 1000.0

    # Dump final statistics
    dump_final_statistics(chan_model, system_config, last_active_cells)

    # Print timing summary
    print(f"\n{'=' * 80}")
    print("TIMING SUMMARY")
    print(f"{'=' * 80}")
    print(f"Total TTIs simulated: {n_tti}")
    print(f"Total simulation time: {total_elapsed_ms:.3f} ms ({total_elapsed_s:.3f} s)")
    print("\nPer-TTI timing statistics:")
    print(f"  Average: {np.mean(tti_times):.3f} ms")
    print(f"  Minimum: {np.min(tti_times):.3f} ms")
    print(f"  Maximum: {np.max(tti_times):.3f} ms")
    print(f"  Std Dev: {np.std(tti_times):.3f} ms")
    if n_tti > 1:
        print(f"\nFirst TTI: {tti_times[0]:.3f} ms (includes initialization overhead)")
        print(f"Average excluding first TTI: {np.mean(tti_times[1:]):.3f} ms")
    print(f"{'=' * 80}\n")


def test_external_multisite_single_sector_topology(tmp_path: Path) -> None:
    """Drop UEs around external cells before deriving topology geometry."""
    config_file = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "../../config/statistic_channel_config.yaml"
    )
    system_config, link_config, sim_config, external_config = read_config(
        os.path.normpath(config_file)
    )
    system_config.n_site = 2
    system_config.n_sector_per_site = 1
    system_config.n_ut = 2
    system_config.ut_drop_option = 2
    system_config.force_indoor_ratio = 0.0
    sim_config.cpu_only_mode = 1

    external_config.cell_config = [
        CellParam(1, 1, Coordinate(2500.0, 1000.0, 25.0), 0,
                  [12.0, 30.0, 0.0]),
        CellParam(0, 0, Coordinate(2000.0, 1000.0, 25.0), 0,
                  [12.0, 30.0, 0.0]),
    ]
    external_config.ut_config = []

    chan_model = StatisticalChannel(
        sim_config=sim_config,
        system_level_config=system_config,
        link_level_config=link_config,
        external_config=external_config,
        rand_seed=7,
    )
    topology_path = tmp_path / "external_topology.yaml"
    chan_model._impl.dump_topology_to_yaml(str(topology_path))

    with topology_path.open("r", encoding="utf-8") as topology_file:
        topology = yaml.safe_load(topology_file)

    base_stations = sorted(topology["base_stations"], key=lambda cell: cell["cid"])
    user_equipment = sorted(topology["user_equipment"], key=lambda ue: ue["uid"])
    expected_xy = [(2000.0, 1000.0), (2500.0, 1000.0)]
    cell_radius = 500.0 / np.sqrt(3.0)

    assert topology["topology"]["nSector"] == 2
    for base_station, ue, (expected_x, expected_y) in zip(
            base_stations, user_equipment, expected_xy, strict=True):
        assert base_station["location"]["x"] == pytest.approx(expected_x)
        assert base_station["location"]["y"] == pytest.approx(expected_y)
        ue_distance = np.hypot(
            ue["location"]["x"] - expected_x,
            ue["location"]["y"] - expected_y,
        )
        assert ue_distance <= cell_radius


@pytest.mark.parametrize(
    "initial_indoor_ratio, external_outdoor_ind",
    [(1.0, 1), (0.0, 0)],
)
def test_external_outdoor_ind_recomputes_o2i_derived_state(
        tmp_path: Path,
        initial_indoor_ratio: float,
        external_outdoor_ind: int) -> None:
    """Keep O2I state consistent with a per-UE external override."""
    config_file = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "../../config/statistic_channel_config.yaml"
    )
    system_config, link_config, sim_config, external_config = read_config(
        os.path.normpath(config_file)
    )
    system_config.n_ut = 1
    system_config.force_indoor_ratio = initial_indoor_ratio
    sim_config.cpu_only_mode = 1
    external_config.ut_config = [
        UtParamCfg(
            uid=0,
            loc=Coordinate(50.0, 0.0, 1.5),
            outdoor_ind=external_outdoor_ind,
            ant_panel_idx=1,
            ant_panel_orientation=[90.0, 0.0, 0.0],
            velocity=[0.0, 0.0, 0.0],
        )
    ]

    chan_model = StatisticalChannel(
        sim_config=sim_config,
        system_level_config=system_config,
        link_level_config=link_config,
        external_config=external_config,
        rand_seed=7,
    )
    topology_path = tmp_path / f"external_o2i_{external_outdoor_ind}.yaml"
    chan_model._impl.dump_topology_to_yaml(str(topology_path))

    with topology_path.open("r", encoding="utf-8") as topology_file:
        ue = yaml.safe_load(topology_file)["user_equipment"][0]

    assert ue["outdoor_ind"] == external_outdoor_ind
    if external_outdoor_ind == 1:
        assert ue["d_2d_in"] == 0.0
        assert ue["o2i_penetration_loss"] == 0.0
    else:
        assert 0.0 < ue["d_2d_in"] <= 50.0
        assert ue["o2i_penetration_loss"] > 0.0


def test_location_update_copy_waits_on_explicit_producer_stream() -> None:
    """Copy CuPy mobility updates on the stream that produced them."""
    class RunCapture:
        """Capture arguments passed through the public channel-model seam."""

        def run(self, **kwargs) -> None:
            self.arguments = kwargs

    producer_stream = CudaStream()
    channel_stream = CudaStream()
    with producer_stream:
        locations = cp.full((2, 3), 7.5, dtype=cp.float32)

    channel = StatisticalChannel.__new__(StatisticalChannel)
    channel.sim_config = SimpleNamespace(cpu_only_mode=1)
    channel._stream = channel_stream
    capture = RunCapture()
    channel._impl = capture
    channel.run(
        ut_new_loc=locations,
        ut_update_stream=producer_stream,
    )

    np.testing.assert_array_equal(
        capture.arguments["ut_new_loc"],
        np.full((2, 3), 7.5, dtype=np.float32),
    )


def test_prbg_count_cannot_exceed_prb_count() -> None:
    """Reject a PRBG count that cannot partition the configured PRBs."""
    config_file = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "../../config/statistic_channel_config.yaml"
    )
    system_config, link_config, sim_config, external_config = read_config(
        os.path.normpath(config_file)
    )
    sim_config.cpu_only_mode = 1
    sim_config.n_prb = 10
    sim_config.n_prbg = 11

    with pytest.raises(RuntimeError, match=r"n_prbg must be in \[1, n_prb\]"):
        StatisticalChannel(
            sim_config=sim_config,
            system_level_config=system_config,
            link_level_config=link_config,
            external_config=external_config,
        )


def test_prb_count_cannot_overflow_native_subcarrier_count() -> None:
    """Reject a PRB count whose subcarrier count cannot fit native storage."""
    config_file = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "../../config/statistic_channel_config.yaml"
    )
    system_config, link_config, sim_config, external_config = read_config(
        os.path.normpath(config_file)
    )
    sim_config.cpu_only_mode = 1
    sim_config.n_prb = 6000
    sim_config.n_prbg = 1

    with pytest.raises(RuntimeError, match="more subcarriers than the channel supports"):
        StatisticalChannel(
            sim_config=sim_config,
            system_level_config=system_config,
            link_level_config=link_config,
            external_config=external_config,
        )


def test_deramped_prbg_conversion_preserves_per_subcarrier_cfr(
        tmp_path: Path) -> None:
    """Keep SC output frequency-selective when PRBGs use de-ramped averaging."""
    config_file = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "../../config/statistic_channel_config.yaml"
    )
    with open(os.path.normpath(config_file), "r", encoding="utf-8") as source:
        config = yaml.safe_load(source)
    config["system_level"].update({
        "n_site": 1,
        "n_sector_per_site": 1,
        "n_ut": 1,
    })
    config["simulation"].update({
        "n_prb": 4,
        "n_prbg": 2,
        "n_snapshot_per_slot": 1,
        "run_mode": 3,
        "internal_memory_mode": 0,
        "freq_convert_type": 4,
        "sc_sampling": 1,
        "cpu_only_mode": 0,
    })
    regression_config = tmp_path / "deramped_prbg.yaml"
    with regression_config.open("w", encoding="utf-8") as destination:
        yaml.safe_dump(config, destination)

    chan_model, sim_config, system_config = create_channel_model_from_config(
        str(regression_config)
    )
    simulation_params = setup_simulation_parameters(sim_config, system_config)
    (active_cells, active_uts, locations, velocities,
     cir_coe, cir_delay, cir_ntaps, cfr_prbg, cfr_sc, _) = per_slot_update(
        sim_config, system_config, *simulation_params, tti_idx=0
    )
    run_channel_model_step(
        chan_model, 0, sim_config, active_cells, active_uts,
        locations, velocities, cir_coe, cir_delay, cir_ntaps,
        cfr_prbg, cfr_sc
    )

    first_prbg_sc = cfr_sc[0][0, 0, 0, 0, :24]
    max_magnitude = cp.max(cp.abs(first_prbg_sc))
    # CFR magnitude varies with path loss; require frequency selectivity relative
    # to the generated channel power, not against a fixed absolute scale. Do
    # not let one anomalous subcarrier alone establish frequency selectivity.
    min_relative_variation = 1e-3
    median_real = cp.median(first_prbg_sc.real)
    median_imag = cp.median(first_prbg_sc.imag)
    varying_subcarrier_count = cp.count_nonzero(
        cp.hypot(
            first_prbg_sc.real - median_real,
            first_prbg_sc.imag - median_imag,
        )
        > min_relative_variation * max_magnitude
    )
    assert bool((varying_subcarrier_count >= first_prbg_sc.size // 2).get())
    assert bool(cp.allclose(cfr_prbg[0][0, 0, 0, 0, 0], first_prbg_sc[12]))


@pytest.mark.parametrize("run_mode", [1, 3])
def test_273_prb_cfr_tail_above_thread_block_limit(
        tmp_path: Path, run_mode: int) -> None:
    """Populate the 273-PRBG tail in PRBG-only and combined SC/PRBG modes."""
    config_file = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "../../config/statistic_channel_config.yaml"
    )
    with open(os.path.normpath(config_file), "r", encoding="utf-8") as source:
        config = yaml.safe_load(source)

    config["system_level"]["n_site"] = 1
    config["system_level"]["n_ut"] = 3
    config["simulation"].update({
        "bandwidth_hz": 100e6,
        "fft_size": 4096,
        "n_prb": 273,
        "n_prbg": 273,
        "run_mode": run_mode,
        "internal_memory_mode": 0,
    })
    regression_config = tmp_path / f"prbg273_run_mode_{run_mode}.yaml"
    with regression_config.open("w", encoding="utf-8") as destination:
        yaml.safe_dump(config, destination)

    chan_model, sim_config, system_config = create_channel_model_from_config(
        str(regression_config)
    )
    simulation_params = setup_simulation_parameters(sim_config, system_config)
    (active_cells, active_uts, locations, velocities,
     cir_coe, cir_delay, cir_ntaps, cfr_prbg, cfr_sc, _) = per_slot_update(
        sim_config, system_config, *simulation_params, tti_idx=0
    )

    # CUDA-DET-1.3: issue and synchronize the sentinel writes on the model's
    # explicit non-blocking stream; do not rely on a CUDA default stream.
    with chan_model._stream:
        for cell_cfr in cfr_prbg:
            cell_cfr[..., 256:] = cp.complex64(complex(np.nan, np.nan))
    chan_model._stream.synchronize()

    run_channel_model_step(
        chan_model, 0, sim_config, active_cells, active_uts,
        locations, velocities, cir_coe, cir_delay, cir_ntaps,
        cfr_prbg, cfr_sc
    )

    assert len(cfr_prbg) == 3
    for cell_cfr in cfr_prbg:
        assert cell_cfr.shape[-1] == 273
        assert bool(cp.all(cp.isfinite(cell_cfr[..., 256:])).get())


@pytest.mark.parametrize("memory_mode", [0, 1, 2])
def test_cpu_memory_modes_h5_export_retains_active_ut_subset_outputs(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        memory_mode: int) -> None:
    """Populate and retain CPU outputs in every memory ownership mode."""
    config_file = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "../../config/statistic_channel_config.yaml"
    )
    system_config, link_config, sim_config, external_config = read_config(
        os.path.normpath(config_file)
    )
    sim_config.cpu_only_mode = 1
    sim_config.internal_memory_mode = memory_mode
    sim_config.run_mode = 1
    external_config.cell_config = generate_cell_configs(
        system_config, external_config
    )
    external_config.ut_config = generate_ut_configs(
        system_config, external_config
    )
    chan_model = StatisticalChannel(
        sim_config=sim_config,
        system_level_config=system_config,
        link_level_config=link_config,
        external_config=external_config,
    )
    # Keep an empty first selection and a populated nonzero cell. This verifies
    # that HDF output is neither gated by selector zero nor named by ordinal.
    active_cells = [0, 1]
    active_uts = [[], [0, 2]]
    _, n_snapshots, n_bs_ant, n_ut_ant, max_taps = (
        setup_simulation_parameters(sim_config, system_config)
    )
    cir_coe, cir_delay, cir_ntaps = allocate_cir_arrays(
        sim_config, active_cells, active_uts,
        n_snapshots, n_ut_ant, n_bs_ant, max_taps
    )
    cfr_prbg, cfr_sc = allocate_cfr_arrays(
        sim_config, active_cells, active_uts,
        n_snapshots, n_ut_ant, n_bs_ant
    )
    cir_coe.insert(
        0, np.empty((0, n_snapshots, n_ut_ant, n_bs_ant, max_taps),
                    dtype=np.complex64)
    )
    cir_delay.insert(0, np.empty((0, max_taps), dtype=np.uint16))
    cir_ntaps.insert(0, np.empty(0, dtype=np.uint16))
    cfr_prbg.insert(
        0, np.empty((0, n_snapshots, n_ut_ant, n_bs_ant, sim_config.n_prbg),
                    dtype=np.complex64)
    )
    locations = np.asarray(
        [[ut.loc.x, ut.loc.y, ut.loc.z] for ut in external_config.ut_config],
        dtype=np.float32,
    )
    velocities = np.asarray(
        [list(ut.velocity) for ut in external_config.ut_config],
        dtype=np.float32,
    )
    locations[0, 0] += 0.25

    chan_model.run(
        active_cell=active_cells,
        active_ut=active_uts,
        ut_new_loc=locations,
        ut_new_velocity=velocities,
        cir_coe=cir_coe,
        cir_norm_delay=cir_delay,
        cir_n_taps=cir_ntaps,
        cfr_prbg=cfr_prbg,
        cfr_sc=cfr_sc,
    )
    assert np.all(np.isfinite(cir_coe[1]))
    assert np.count_nonzero(cir_coe[1]) > 0
    assert np.all(np.isfinite(cfr_prbg[1]))
    assert np.count_nonzero(cfr_prbg[1]) > 0
    cir_owner = weakref.ref(cir_coe[1])
    cfr_owner = weakref.ref(cfr_prbg[1])
    cir_coe.clear()
    cir_delay.clear()
    cir_ntaps.clear()
    cfr_prbg.clear()
    cfr_sc.clear()
    del cir_coe, cir_delay, cir_ntaps, cfr_prbg, cfr_sc
    gc.collect()
    assert cir_owner() is not None
    assert cfr_owner() is not None

    monkeypatch.chdir(tmp_path)
    chan_model._impl.save_sls_chan_to_h5_file("_owner_subset")
    output_path = tmp_path / (
        f"slsChanData_1sites_{system_config.n_ut}uts_owner_subset_seed0.h5"
    )
    with h5py.File(output_path, "r") as output_file:
        assert output_file.attrs["gpu3gppchan_format_version"] == 3
        assert output_file["cirPerCell/cirCoe_cell1"].shape[0] == 2
        assert output_file["cfrPrbgPerCell/cfrPrbg_cell1"].shape[0] == 2
        assert output_file["topology/utParams/loc_x"][0] == pytest.approx(
            locations[0, 0]
        )


@pytest.mark.parametrize("memory_mode", [1, 2])
@pytest.mark.parametrize("cpu_only_mode", [1, 0])
def test_internal_memory_getters_copy_cir_and_cfr(
    memory_mode: int, cpu_only_mode: int
) -> None:
    """Copy internally owned channel outputs into caller buffers."""
    config_file = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "../../config/statistic_channel_config.yaml",
    )
    system_config, link_config, sim_config, external_config = read_config(
        os.path.normpath(config_file)
    )
    system_config.n_site = 1
    system_config.n_sector_per_site = 1
    system_config.n_ut = 1
    sim_config.cpu_only_mode = cpu_only_mode
    sim_config.internal_memory_mode = memory_mode
    sim_config.run_mode = 3
    sim_config.n_prb = 4
    sim_config.n_prbg = 2
    external_config.cell_config = generate_cell_configs(
        system_config, external_config
    )
    external_config.ut_config = generate_ut_configs(
        system_config, external_config
    )
    channel = StatisticalChannel(
        sim_config=sim_config,
        system_level_config=system_config,
        link_level_config=link_config,
        external_config=external_config,
        rand_seed=19,
    )
    active_cells = [0]
    active_uts = [[0]]
    _, n_snapshots, n_bs_ant, n_ut_ant, max_taps = (
        setup_simulation_parameters(sim_config, system_config)
    )
    cir_coe, cir_delay, cir_ntaps = allocate_cir_arrays(
        sim_config, active_cells, active_uts,
        n_snapshots, n_ut_ant, n_bs_ant, max_taps,
    )
    cfr_prbg, cfr_sc = allocate_cfr_arrays(
        sim_config, active_cells, active_uts,
        n_snapshots, n_ut_ant, n_bs_ant,
    )

    channel.run(active_cell=active_cells, active_ut=active_uts)
    first_cir_delay = cir_delay[0]
    delay_sentinel = np.iinfo(first_cir_delay.dtype).max
    first_cir_delay.fill(delay_sentinel)
    channel.get_cir(
        cir_coe=cir_coe,
        cir_norm_delay=cir_delay,
        cir_n_taps=cir_ntaps,
    )
    if cpu_only_mode == 0:
        channel._stream.synchronize()

    array_module = np if cpu_only_mode == 1 else cp
    assert array_module.count_nonzero(cir_coe[0]) > 0
    assert array_module.any(first_cir_delay != delay_sentinel)
    assert array_module.count_nonzero(cir_ntaps[0]) > 0
    if memory_mode == 2:
        channel.get_cfr(cfr_sc=cfr_sc, cfr_prbg=cfr_prbg)
        if cpu_only_mode == 0:
            channel._stream.synchronize()
        assert array_module.count_nonzero(cfr_prbg[0]) > 0
        assert array_module.count_nonzero(cfr_sc[0]) > 0
    else:
        with pytest.raises(RuntimeError, match="internal_memory_mode 2"):
            channel.get_cfr(cfr_sc=cfr_sc)


def test_cpu_external_memory_rejects_device_buffers(tmp_path: Path) -> None:
    """Reject a device pointer before CPU channel code can dereference it."""
    config_file = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "../../config/statistic_channel_config.yaml"
    )
    with open(os.path.normpath(config_file), "r", encoding="utf-8") as source:
        config = yaml.safe_load(source)
    config["simulation"]["cpu_only_mode"] = 1
    config["simulation"]["internal_memory_mode"] = 0
    regression_config = tmp_path / "cpu_external.yaml"
    with regression_config.open("w", encoding="utf-8") as destination:
        yaml.safe_dump(config, destination)

    chan_model, _, _ = create_channel_model_from_config(str(regression_config))
    with pytest.raises(ValueError, match="cir_coe.*host-accessible"):
        chan_model.run(
            active_cell=[0],
            active_ut=[[]],
            cir_coe=[cp.empty(1, dtype=cp.complex64)],
        )


def _make_readonly_numpy_output(size: int) -> np.ndarray:
    """Create a read-only output candidate for native contract tests."""
    output = np.empty(size, dtype=np.complex64)
    output.flags.writeable = False
    return output


@pytest.mark.parametrize(
    "buffer_factory,expected_message",
    [
        (lambda: np.empty(1, dtype=np.complex64), "too small"),
        (lambda: np.empty(1_000_000, dtype=np.uint8), "incompatible dtype"),
        (
            lambda: np.empty(2_000_000, dtype=np.complex64)[::2],
            "C-contiguous",
        ),
        (
            lambda: _make_readonly_numpy_output(1_000_000),
            "writable",
        ),
    ],
)
def test_cpu_external_memory_validates_output_buffer_contract(
        tmp_path: Path, buffer_factory: Callable[[], np.ndarray],
        expected_message: str) -> None:
    """Reject buffers that cannot hold the native CIR write extent."""
    config_file = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "../../config/statistic_channel_config.yaml"
    )
    with open(os.path.normpath(config_file), "r", encoding="utf-8") as source:
        config = yaml.safe_load(source)
    config["simulation"]["cpu_only_mode"] = 1
    config["simulation"]["internal_memory_mode"] = 0
    regression_config = tmp_path / "cpu_buffer_contract.yaml"
    with regression_config.open("w", encoding="utf-8") as destination:
        yaml.safe_dump(config, destination)

    chan_model, _, _ = create_channel_model_from_config(str(regression_config))
    buffer = buffer_factory()
    with pytest.raises(ValueError, match=rf"cir_coe.*{expected_message}"):
        chan_model.run(
            active_cell=[0],
            active_ut=[[0]],
            cir_coe=[buffer],
        )


def test_native_statistical_channel_retains_external_output_owner(
        tmp_path: Path) -> None:
    """The exported native API must retain buffers reused by later exports."""
    config_file = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "../../config/statistic_channel_config.yaml"
    )
    with open(os.path.normpath(config_file), "r", encoding="utf-8") as source:
        config = yaml.safe_load(source)
    config["simulation"]["cpu_only_mode"] = 1
    config["simulation"]["internal_memory_mode"] = 0
    regression_config = tmp_path / "cpu_output_owner.yaml"
    with regression_config.open("w", encoding="utf-8") as destination:
        yaml.safe_dump(config, destination)

    chan_model, sim_config, system_config = create_channel_model_from_config(
        str(regression_config)
    )
    active_cells = [0]
    active_uts = [[0]]
    _, snapshots, n_bs_ant, n_ut_ant, max_taps = setup_simulation_parameters(
        sim_config, system_config
    )
    cir_coe, cir_delay, cir_ntaps = allocate_cir_arrays(
        sim_config, active_cells, active_uts,
        snapshots, n_ut_ant, n_bs_ant, max_taps,
    )
    cfr_prbg, cfr_sc = allocate_cfr_arrays(
        sim_config, active_cells, active_uts,
        snapshots, n_ut_ant, n_bs_ant,
    )
    output_references = [
        weakref.ref(output)
        for outputs in (cir_coe, cir_delay, cir_ntaps, cfr_prbg, cfr_sc)
        for output in outputs
    ]
    chan_model.run(
        active_cell=active_cells,
        active_ut=active_uts,
        cir_coe=cir_coe,
        cir_norm_delay=cir_delay,
        cir_n_taps=cir_ntaps,
        cfr_prbg=cfr_prbg,
        cfr_sc=cfr_sc,
    )

    chan_model._external_output_owners = ()
    del cir_coe, cir_delay, cir_ntaps, cfr_prbg, cfr_sc
    gc.collect()
    assert all(reference() is not None for reference in output_references)


def test_external_memory_requires_buffer_per_active_cell(tmp_path: Path) -> None:
    """Reject partial external buffer lists before native execution."""
    config_file = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "../../config/statistic_channel_config.yaml"
    )
    with open(os.path.normpath(config_file), "r", encoding="utf-8") as source:
        config = yaml.safe_load(source)
    config["system_level"]["n_site"] = 2
    config["simulation"]["cpu_only_mode"] = 1
    config["simulation"]["internal_memory_mode"] = 0
    regression_config = tmp_path / "partial_external_buffers.yaml"
    with regression_config.open("w", encoding="utf-8") as destination:
        yaml.safe_dump(config, destination)

    chan_model, _, _ = create_channel_model_from_config(str(regression_config))
    with pytest.raises(
            ValueError, match=r"cir_norm_delay must contain one buffer per active cell"):
        chan_model.run(
            active_cell=[0, 1],
            active_ut=[[], []],
            cir_norm_delay=[np.empty(1, dtype=np.uint16)],
        )


def test_gpu_external_memory_rejects_host_buffers(tmp_path: Path) -> None:
    """Reject a host pointer before GPU channel code can dereference it."""
    class CpuTensorLike:
        """Model a CPU tensor that exposes the generic data_ptr protocol."""

        is_cuda = False
        nbytes: int = 1_000_000

        @staticmethod
        def data_ptr() -> int:
            """Return a non-null host address placeholder."""
            return 1

    config_file = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "../../config/statistic_channel_config.yaml"
    )
    with open(os.path.normpath(config_file), "r", encoding="utf-8") as source:
        config = yaml.safe_load(source)
    config["simulation"]["cpu_only_mode"] = 0
    config["simulation"]["internal_memory_mode"] = 0
    regression_config = tmp_path / "gpu_external.yaml"
    with regression_config.open("w", encoding="utf-8") as destination:
        yaml.safe_dump(config, destination)

    chan_model, _, _ = create_channel_model_from_config(str(regression_config))
    # NumPy output arrays are intentionally converted to CuPy by the Python
    # wrapper. A CPU tensor-like object reaches the native fallback unchanged.
    with pytest.raises(ValueError, match=r"cir_coe.*CUDA-accessible"):
        chan_model.run(
            active_cell=[0],
            active_ut=[[]],
            cir_coe=[CpuTensorLike()],
        )


@pytest.mark.parametrize("invalid_layout", ["dtype", "strided"])
def test_gpu_external_memory_validates_output_buffer_contract(
        tmp_path: Path, invalid_layout: str) -> None:
    """Reject incompatible CUDA output dtype and layout metadata."""
    config_file = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "../../config/statistic_channel_config.yaml"
    )
    with open(os.path.normpath(config_file), "r", encoding="utf-8") as source:
        config = yaml.safe_load(source)
    config["simulation"]["cpu_only_mode"] = 0
    config["simulation"]["internal_memory_mode"] = 0
    regression_config = tmp_path / "gpu_buffer_dtype.yaml"
    with regression_config.open("w", encoding="utf-8") as destination:
        yaml.safe_dump(config, destination)

    chan_model, _, _ = create_channel_model_from_config(str(regression_config))
    if invalid_layout == "dtype":
        buffer = cp.empty(1_000_000, dtype=cp.uint8)
        expected_message = "incompatible dtype"
    else:
        buffer = cp.empty(2_000_000, dtype=cp.complex64)[::2]
        expected_message = "C-contiguous"
    with pytest.raises(ValueError, match=rf"cir_coe.*{expected_message}"):
        chan_model.run(
            active_cell=[0],
            active_ut=[[0]],
            cir_coe=[buffer],
        )


@pytest.mark.parametrize("n_site,n_ut,n_tti", [
    (1, 8, 5),       # Small test case
    (3, 80, 5),      # Medium test case
])
def test_channel_model_simulation(n_site, n_ut, n_tti):
    """Test channel model simulation with different configurations."""
    chan_model, sim_config, system_config = create_channel_model_from_config(
        n_site=n_site, n_ut=n_ut)

    # Run simulation with specified number of TTIs
    run_channel_simulation(chan_model, sim_config, system_config, n_tti)

    # Verify the model was created successfully
    assert chan_model is not None
    assert sim_config.n_snapshot_per_slot == 1
    assert system_config.n_site == n_site
    assert system_config.n_ut == n_ut


if __name__ == "__main__":
    # Run pytest tests
    try:
        import subprocess
        import sys
        print("Running pytest tests...")
        subprocess.run([sys.executable, "-m", "pytest", __file__, "-v"])
    except Exception as e:
        print(f"Could not run pytest: {e}")

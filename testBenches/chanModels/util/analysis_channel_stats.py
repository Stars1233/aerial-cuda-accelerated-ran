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

"""
H5 Channel Model Analysis Script

This script analyzes 3GPP channel model data from H5 files directly and computes:
1. Coupling loss for serving cell
2. Wideband SIR before receiver without noise
3. CDFs of Delay Spread and Angle Spreads (ASD, ZSD, ASA, ZSA) 
4. CDFs of PRB singular values (largest, smallest, ratio) at t=0

Cell Association Methods:
- RSRP-based (default): Uses received signal power (TX power - (path loss - shadow fading))
- CIR-based: Uses Channel Impulse Response power from H5 file (use --use-cir-association flag)
"""

import numpy as np
import matplotlib.pyplot as plt
import h5py
import logging
import json
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from pathlib import Path
from scipy import stats

# Set up logging
def setup_logging(log_file: Optional[str] = None, log_level: str = 'ERROR'):
    """Configure logging to output to console or file
    
    Args:
        log_file: Optional path to log file
        log_level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR')
    """
    format_str = '%(asctime)s - %(levelname)s - %(message)s'
    level = getattr(logging, log_level.upper(), logging.INFO)
    
    if log_file:
        # Create logs directory if it doesn't exist
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Set up file handler
        logging.basicConfig(
            level=level,
            format=format_str,
            handlers=[
                logging.FileHandler(log_file, mode='w'),
                logging.StreamHandler()  # Also log to console
            ]
        )
        print(f"Logging to file: {log_file}")
    else:
        logging.basicConfig(level=level, format=format_str)

# Logger will be configured in main() based on command line arguments
logger = logging.getLogger(__name__)

import re
import glob

def find_multi_seed_files(base_h5_file: str) -> List[str]:
    """
    Find all H5 files matching the base pattern with different random seeds.
    
    The H5 filename format is: slsChanData_{sites}sites_{uts}uts_{suffix}_seed{N}.h5
    Given one file, this function finds all files with the same base pattern but different seeds.
    
    Args:
        base_h5_file: Path to one of the H5 files (any seed)
        
    Returns:
        Sorted list of matching H5 file paths
    """
    base_path = Path(base_h5_file)
    
    # Extract the base pattern by removing _seed{N} suffix
    # Pattern: ...._seed123.h5 -> ...._seed*.h5
    filename = base_path.stem  # Without .h5
    
    # Match pattern: _seed followed by digits at the end
    seed_pattern = re.compile(r'^(.+)_seed(\d+)$')
    match = seed_pattern.match(filename)
    
    if match:
        base_name = match.group(1)
        # Create glob pattern to find all seed variants
        glob_pattern = str(base_path.parent / f"{base_name}_seed*.h5")
        matching_files = sorted(glob.glob(glob_pattern))
        
        if matching_files:
            logger.info(f"Found {len(matching_files)} H5 files with different seeds:")
            for f in matching_files:
                # Extract seed number for logging
                seed_match = seed_pattern.match(Path(f).stem)
                if seed_match:
                    logger.info(f"  - {f} (seed={seed_match.group(2)})")
            return matching_files
    
    # If no seed pattern found or no matches, return just the original file
    logger.warning(f"No seed pattern found in filename: {base_h5_file}")
    logger.warning("Expected format: ...._seed{N}.h5")
    logger.warning("Proceeding with single file analysis.")
    return [base_h5_file]

@dataclass
class ClusterParams:
    """Data structure for cluster parameters"""
    n_cluster: int
    n_ray_per_cluster: int
    delays: List[float]
    powers: List[float]
    strongest_2_clusters_idx: List[int]
    phi_n_aoa: List[float]
    phi_n_aod: List[float]
    theta_n_zod: List[float]
    theta_n_zoa: List[float]
    xpr: List[float]
    random_phases: List[float]
    # Per-ray angles (N_clusters * N_rays per array)
    phi_n_m_aoa: List[float]
    phi_n_m_aod: List[float]
    theta_n_m_zod: List[float]
    theta_n_m_zoa: List[float]

@dataclass
class LinkParams:
    """Data structure for link parameters"""
    d2d: float
    d2d_in: float
    d2d_out: float
    d3d: float
    d3d_in: float
    phi_los_aod: float
    theta_los_zod: float
    phi_los_aoa: float
    theta_los_zoa: float
    los_ind: int
    pathloss: float
    sf: float  # Shadow fading
    k_factor: float
    ds: float  # Delay spread
    asd: float  # Azimuth spread of departure
    asa: float  # Azimuth spread of arrival
    mu_lg_zsd: float
    sigma_lg_zsd: float
    mu_offset_zod: float
    zsd: float  # Zenith spread of departure
    zsa: float  # Zenith spread of arrival

@dataclass
class AntennaPanel:
    """Data structure for antenna panel configuration"""
    panel_idx: int
    n_antennas: int  # Total number of antennas (from nAnt field)
    ant_model: int  # Antenna model type
    ant_size: List[int]  # Antenna size array [5]
    ant_spacing: List[float]  # Antenna spacing array [4]
    ant_polar_angles: List[float]  # Polarization angles [2]
    # Note: antTheta and antPhi arrays are large (181 and 360 elements)
    # and not stored in this dataclass for memory efficiency
    
    @property
    def total_antennas(self) -> int:
        """Total number of antennas in the panel"""
        return self.n_antennas
    
    def antenna_gain_db(self, use_calibration_values: bool = False, calibration_phase: int = 2,
                       angle_aod_deg: float = 0.0, tilt_deg: float = 12.0, sector_orientation_deg: float = 0.0) -> float:
        """
        Calculate antenna gain in dB based on number of antennas with optional virtualization.
        
        Args:
            use_calibration_values: If True, use antenna virtualization per ITU-R M.2101
            calibration_phase: Calibration phase (1 or 2) for virtualization method
            angle_aod_deg: Azimuth angle of departure in degrees (for Phase 2 beamforming)
            tilt_deg: BS antenna tilt in degrees (default: 12 degrees)
            sector_orientation_deg: Sector orientation in degrees (default: 0 degrees)
            
        Returns:
            Antenna gain in dB
        """
        if self.total_antennas <= 0:
            logger.warning(f"Panel {self.panel_idx}: Invalid antenna count ({self.total_antennas}), using 0 dB gain")
            return 0.0
        
        if not use_calibration_values:
            # Simple gain calculation: 10*log10(N) where N is number of antennas
            gain_db = 10.0 * np.log10(self.total_antennas)
            logger.debug(f"Panel {self.panel_idx}: {self.total_antennas} antennas, gain = 10*log10({self.total_antennas}) = {gain_db:.2f} dB (no virtualization)")
            return gain_db
        
        # Apply antenna virtualization based on calibration phase
        if calibration_phase == 1:
            # Phase 1: Use simplified virtualization with tilt-based beamforming
            # w_virt = (1/sqrt(numM)) * exp(-1i*pi*cos(deg2rad(tilt_deg+90))*(0:numM-1).')
            # numM = 10 (number of antenna rows)
            # The gain comes from: pow = conj(w_virt) * H * H' * w_virt
            # For calibration, we assume H approximates identity for the array gain calculation
            num_m = 10  # Number of rows for Phase 1
            
            # Calculate beamforming weights
            tilt_rad = np.deg2rad(tilt_deg + 90.0)
            m_indices = np.arange(num_m)
            w_virt = (1.0 / np.sqrt(num_m)) * np.exp(-1j * np.pi * np.cos(tilt_rad) * m_indices)
            
            # For Phase 1, the beamforming is applied to a column of antennas
            # The array gain is effectively numM for coherent combining
            # Since w_virt is normalized (|w_virt|^2 = 1), the gain is numM
            # pow = |w_virt' * w_virt| * numM (coherent combining gain)
            power_gain_linear = num_m  # Coherent array gain for Phase 1
            gain_db = 10.0 * np.log10(power_gain_linear)
            
            logger.debug(f"Panel {self.panel_idx}: Phase 1 virtualization, numM={num_m}, tilt={tilt_deg}°, gain={gain_db:.2f} dB")
            
        elif calibration_phase == 2:
            # Phase 2: Group 16 elements into 1 CRS port, sweep angles, pick best gain
            # 64 elements -> 4 TX ports (64/16 = 4)
            if self.total_antennas < 16:
                # Fallback to simple gain if not enough antennas
                gain_db = 10.0 * np.log10(self.total_antennas)
                logger.warning(f"Panel {self.panel_idx}: Not enough antennas ({self.total_antennas}) for Phase 2 virtualization, using simple gain")
                return gain_db
            
            n_elements_per_port = 16
            n_tx_ports = self.total_antennas // n_elements_per_port
            
            # Sweep 12 angles, each covering 10 degrees, from -60 to 60 (relative to sector orientation)
            angle_sweep_deg = np.linspace(-60, 60, 12)  # 12 angles from -60° to 60°
            
            max_gain_linear = 0.0
            best_angle = 0.0
            
            for sweep_angle_deg in angle_sweep_deg:
                # Absolute angle = sector orientation + sweep angle
                absolute_angle_deg = sector_orientation_deg + sweep_angle_deg
                
                # Calculate relative angle to the arrival direction (AoD)
                relative_angle_deg = angle_aod_deg - absolute_angle_deg
                relative_angle_rad = np.deg2rad(relative_angle_deg)
                
                # Calculate beamforming gain for each TX port
                port_gains = []
                for port_idx in range(n_tx_ports):
                    # For each port, calculate beamforming weights for 16 elements
                    # Using uniform linear array with half-wavelength spacing
                    element_indices = np.arange(n_elements_per_port)
                    
                    # Beamforming weight with phase progression based on angle
                    # w = (1/sqrt(N)) * exp(1j * pi * cos(theta) * n)
                    w_port = (1.0 / np.sqrt(n_elements_per_port)) * np.exp(1j * np.pi * np.cos(relative_angle_rad) * element_indices)
                    
                    # Power gain for this port
                    port_gain_linear = np.sum(np.abs(w_port)**2) * n_elements_per_port
                    port_gains.append(port_gain_linear)
                
                # Pick the highest gain among all TX ports
                max_port_gain = np.max(port_gains)
                
                if max_port_gain > max_gain_linear:
                    max_gain_linear = max_port_gain
                    best_angle = absolute_angle_deg
            
            gain_db = 10.0 * np.log10(max_gain_linear)
            
            logger.debug(f"Panel {self.panel_idx}: Phase 2 virtualization, {n_tx_ports} TX ports, "
                        f"best_angle={best_angle:.1f}°, gain={gain_db:.2f} dB")
        
        else:
            # Unknown phase, use simple gain
            gain_db = 10.0 * np.log10(self.total_antennas)
            logger.warning(f"Panel {self.panel_idx}: Unknown calibration phase {calibration_phase}, using simple gain")
        
        return gain_db

@dataclass
class CellParams:
    """Data structure for cell parameters"""
    cid: int
    site_id: int
    location: Dict[str, float]
    ant_panel_idx: int
    ant_panel_orientation: List[float] = None  # [zenith, azimuth, slant] in degrees
    
    def __post_init__(self):
        """Initialize default orientation if not provided"""
        if self.ant_panel_orientation is None:
            self.ant_panel_orientation = [90.0, 0.0, 0.0]  # Default: horizontal, 0° azimuth

@dataclass
class ActiveLinkParams:
    """Data structure for active link parameters"""
    cid: int
    uid: int
    link_idx: int
    lsp_read_idx: int

@dataclass
class SystemLevelConfig:
    """Data structure for system-level configuration"""
    scenario: str
    isd: float
    n_site: int
    n_sector_per_site: int
    n_ut: int
    optional_pl_ind: int
    o2i_building_penetr_loss_ind: int
    o2i_car_penetr_loss_ind: int
    enable_near_field_effect: int
    enable_non_stationarity: int
    force_los_prob: List[float]
    force_ut_speed: List[float]
    force_indoor_ratio: float
    disable_pl_shadowing: int
    disable_small_scale_fading: int
    enable_per_tti_lsp: int
    enable_propagation_delay: int
    ut_drop_option: int = 0  # 0: random across region, 1: same UTs per site, 2: same UTs per sector

@dataclass
class ISACConfig:
    """Data structure for ISAC (Integrated Sensing and Communications) configuration"""
    isac_type: int = 0           # 0: comm only, 1: monostatic, 2: bistatic
    n_st: int = 0                # Number of sensing targets
    st_target_type: int = 0      # 0: UAV, 1: AUTOMOTIVE, 2: HUMAN, 3: AGV, 4: HAZARD
    st_rcs_model: int = 1        # 1: deterministic, 2: angular-dependent
    isac_disable_background: int = 0  # 0: combined, 1: target only
    isac_disable_target: int = 0      # 0: include target, 1: background only
    st_size_ind: int = 0         # Size index for the target
    
    @property
    def is_enabled(self) -> bool:
        """Check if ISAC mode is enabled"""
        return self.isac_type > 0 and self.n_st > 0
    
    @property
    def is_monostatic(self) -> bool:
        """Check if monostatic sensing mode"""
        return self.isac_type == 1
    
    @property
    def is_bistatic(self) -> bool:
        """Check if bistatic sensing mode"""
        return self.isac_type == 2
    
    @property
    def is_target_only(self) -> bool:
        """Check if target-only mode (background disabled)"""
        return self.isac_disable_background == 1 and self.isac_disable_target == 0
    
    @property
    def is_background_only(self) -> bool:
        """Check if background-only mode (target disabled)"""
        return self.isac_disable_target == 1 and self.isac_disable_background == 0
    
    @property
    def is_combined(self) -> bool:
        """Check if combined mode (both target and background)"""
        return self.isac_disable_background == 0 and self.isac_disable_target == 0
    
    @property
    def target_type_name(self) -> str:
        """Get human-readable target type name"""
        target_map = {0: 'UAV', 1: 'AUTOMOTIVE', 2: 'HUMAN', 3: 'AGV', 4: 'HAZARD'}
        return target_map.get(self.st_target_type, 'UNKNOWN')
    
    @property
    def sensing_mode_name(self) -> str:
        """Get sensing mode name for reference data lookup"""
        if self.isac_type == 1:
            return 'TRPmo'  # TRP monostatic
        elif self.isac_type == 2:
            return 'bistatic'
        return 'comm_only'
    
    def get_isac_scenario_key(self, freq_ghz: float, is_background: bool = False) -> str:
        """
        Get ISAC-specific scenario key for reference data lookup
        
        Args:
            freq_ghz: Frequency in GHz
            is_background: True for background channel, False for target channel
            
        Returns:
            Scenario key string (e.g., 'TRPmo(t)-6GHz' or 'TRPmo(b)-6GHz')
        """
        freq_int = int(round(freq_ghz))
        
        if self.isac_type == 1:  # Monostatic
            channel_suffix = '(b)' if is_background else '(t)'
            return f'TRPmo{channel_suffix}-{freq_int}GHz'
        elif self.isac_type == 2:  # Bistatic
            # For bistatic, we have TRP-TRP, TRP-UE, UE-UE modes
            # Default to TRP-TRP for now
            channel_suffix = '(b)' if is_background else ''
            return f'TRP-TRP{channel_suffix}-{freq_int}GHz'
        
        return f'UMa-{freq_int}GHz'  # Fallback to standard

@dataclass
class SimConfig:
    """Data structure for simulation configuration"""
    link_sim_ind: int
    center_freq_hz: float
    bandwidth_hz: float
    sc_spacing_hz: float
    fft_size: int
    n_prb: int
    n_prbg: int
    n_snapshot_per_slot: int
    run_mode: int
    internal_memory_mode: int
    freq_convert_type: int
    sc_sampling: int
    proc_sig_freq: int

class ThreeGPPReferenceData:
    """Class to load and manage 3GPP calibration reference data"""
    
    def __init__(self, json_file_path: str = None, calibration_phase: int = 2):
        """
        Initialize 3GPP reference data loader
        
        Args:
            json_file_path: Path to the 3GPP calibration JSON file
            calibration_phase: Calibration phase (1 or 2). Phase 1 has coupling_loss and geometry_sinr.
                             Phase 2 has coupling_loss, wideband_sir, and angle spreads.
        """
        self.json_file_path = json_file_path
        self.reference_data = None
        self.available_scenarios = []
        self.calibration_phase = calibration_phase
        
        if json_file_path and Path(json_file_path).exists():
            self.load_reference_data()
    
    def load_reference_data(self) -> bool:
        """Load 3GPP calibration data from JSON file"""
        try:
            with open(self.json_file_path, 'r') as f:
                self.reference_data = json.load(f)
            
            self.available_scenarios = list(self.reference_data.get('scenarios', {}).keys())
            logger.info(f"Loaded 3GPP Phase {self.calibration_phase} reference data from: {self.json_file_path}")
            logger.info(f"Available scenarios: {self.available_scenarios}")
            
            # Log available metrics for first scenario as a sample
            if self.available_scenarios:
                first_scenario = self.available_scenarios[0]
                metrics = list(self.reference_data['scenarios'][first_scenario].get('metrics', {}).keys())
                logger.info(f"Available metrics (Phase {self.calibration_phase}): {metrics}")
            
            return True
        except Exception as e:
            logger.error(f"Failed to load 3GPP reference data: {e}")
            return False
    
    def get_scenario_key(self, scenario: str, freq_ghz: float, 
                         isac_config: 'ISACConfig' = None, 
                         isac_channel_type: str = None) -> Optional[str]:
        """
        Get the scenario key for matching reference data
        
        Args:
            scenario: Scenario name (UMa, UMi, InH)
            freq_ghz: Frequency in GHz
            isac_config: Optional ISACConfig for ISAC mode
            isac_channel_type: For ISAC, 'target' or 'background'
            
        Returns:
            Scenario key string or None if not found
        """
        freq_int = int(round(freq_ghz))
        
        # Check if this is ISAC mode with valid config
        if isac_config and isac_config.is_enabled and isac_channel_type:
            # ISAC scenario keys have different format:
            # TRPmo(t)-6GHz, TRPmo(b)-6GHz for monostatic target/background
            # TRP-TRP-6GHz, TRP-TRP(b)-6GHz for bistatic
            is_background = (isac_channel_type.lower() == 'background')
            isac_scenario_key = isac_config.get_isac_scenario_key(freq_ghz, is_background)
            
            if isac_scenario_key in self.available_scenarios:
                logger.info(f"Using ISAC scenario key: {isac_scenario_key}")
                return isac_scenario_key
            
            # Try with trailing space (some Excel exports have this)
            isac_scenario_key_space = isac_scenario_key + ' '
            if isac_scenario_key_space in self.available_scenarios:
                logger.info(f"Using ISAC scenario key (with space): {isac_scenario_key_space}")
                return isac_scenario_key_space
            
            # Try finding closest match
            prefix = isac_scenario_key.split('-')[0]  # e.g., 'TRPmo(t)'
            for avail_scenario in self.available_scenarios:
                if avail_scenario.startswith(prefix) and f'{freq_int}GHz' in avail_scenario:
                    logger.warning(f"Exact match not found for {isac_scenario_key}, using {avail_scenario}")
                    return avail_scenario
            
            logger.warning(f"ISAC scenario key not found: {isac_scenario_key}")
            # Fall through to standard lookup
        
        # Standard (non-ISAC) scenario lookup
        # Map scenario names to expected format
        scenario_map = {
            'UMA': 'UMa',
            'UMI': 'UMi',
            'UMI-STREET CANYON': 'UMi',
            'UMI_STREET_CANYON': 'UMi',
            'INH': 'InH',
            'INDOOR HOTSPOT': 'InH'
        }
        
        scenario_normalized = scenario_map.get(scenario.upper(), scenario)
        
        # Try exact frequency match (6, 30, 60, 70 GHz)
        scenario_key = f"{scenario_normalized}-{freq_int}GHz"
        
        if scenario_key in self.available_scenarios:
            return scenario_key
        
        # Try finding closest frequency
        for avail_scenario in self.available_scenarios:
            if avail_scenario.startswith(f"{scenario_normalized}-"):
                logger.warning(f"Exact match not found for {scenario_key}, using {avail_scenario}")
                return avail_scenario
        
        logger.warning(f"No matching scenario found for {scenario} at {freq_ghz} GHz")
        return None
    
    def get_metric_reference(self, scenario_key: str, metric_name: str) -> Optional[Dict]:
        """
        Get reference data for a specific metric
        
        Args:
            scenario_key: Scenario key (e.g., "UMa-6GHz")
            metric_name: Metric name (coupling_loss, delay_spread, etc.)
            
        Returns:
            Dictionary with reference CDF data or None
        """
        if not self.reference_data or scenario_key not in self.available_scenarios:
            return None
        
        scenario_data = self.reference_data['scenarios'][scenario_key]
        metric_data = scenario_data.get('metrics', {}).get(metric_name)
        
        if not metric_data:
            logger.warning(f"Metric {metric_name} not found in scenario {scenario_key}")
            return None
        
        return metric_data
    
    def get_company_cdf(self, scenario_key: str, metric_name: str, company_name: str = None) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Get CDF data for a specific company or average of all companies
        
        Args:
            scenario_key: Scenario key (e.g., "UMa-6GHz")
            metric_name: Metric name
            company_name: Company name (None for average across all companies)
            
        Returns:
            Tuple of (x_values, cdf_percentiles) or (None, None)
        """
        metric_data = self.get_metric_reference(scenario_key, metric_name)
        if not metric_data:
            return None, None
        
        cdf_percentiles = np.array(metric_data.get('cdf_percentiles', []))
        # Always use data_dict keys as the source of truth for available companies
        data_dict = metric_data.get('data', {})
        
        if len(cdf_percentiles) == 0 or len(data_dict) == 0:
            return None, None
        
        companies = list(data_dict.keys())
        
        if company_name and company_name in data_dict:
            # Return specific company data (convert None to NaN)
            x_values = np.array(data_dict[company_name], dtype=float)
            return x_values, cdf_percentiles
        else:
            # Return average across all companies
            all_company_data = []
            for company in companies:
                if company in data_dict:
                    # Convert None to NaN for proper numpy handling
                    company_values = np.array(data_dict[company], dtype=float)
                    if len(company_values) == len(cdf_percentiles):
                        all_company_data.append(company_values)
            
            if len(all_company_data) == 0:
                return None, None
            
            # Average across companies (using nanmean to ignore NaN values)
            x_values = np.nanmean(all_company_data, axis=0)
            return x_values, cdf_percentiles
    
    def get_company_cdf_envelope(self, scenario_key: str, metric_name: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Get min/max envelope of CDF data across all companies
        
        Args:
            scenario_key: Scenario key (e.g., "UMa-6GHz")
            metric_name: Metric name
            
        Returns:
            Tuple of (x_min, x_max, cdf_percentiles) or (None, None, None)
        """
        metric_data = self.get_metric_reference(scenario_key, metric_name)
        if not metric_data:
            return None, None, None
        
        cdf_percentiles = np.array(metric_data.get('cdf_percentiles', []))
        # Always use data_dict keys as the source of truth for available companies
        data_dict = metric_data.get('data', {})
        
        if len(cdf_percentiles) == 0 or len(data_dict) == 0:
            return None, None, None
        
        companies = list(data_dict.keys())
        
        # Collect all company data
        all_company_data = []
        for company in companies:
            if company in data_dict:
                # Convert None to NaN for proper numpy handling
                company_values = np.array(data_dict[company], dtype=float)
                if len(company_values) == len(cdf_percentiles):
                    all_company_data.append(company_values)
        
        if len(all_company_data) == 0:
            return None, None, None
        
        # Compute min and max across companies at each CDF percentile (using nanmin/nanmax to ignore NaN)
        all_company_data = np.array(all_company_data)
        x_min = np.nanmin(all_company_data, axis=0)
        x_max = np.nanmax(all_company_data, axis=0)
        
        return x_min, x_max, cdf_percentiles

class H5ChannelAnalyzer:
    """Main analyzer class for H5 channel data"""
    
    def __init__(self, h5_file_path: str, reference_json_path: str = None, calibration_phase: int = 2,
                 isac_channel_type: str = None):
        """
        Initialize H5 Channel Analyzer
        
        Args:
            h5_file_path: Path to H5 file
            reference_json_path: Path to 3GPP reference JSON file
            calibration_phase: Calibration phase (1 or 2)
            isac_channel_type: For ISAC mode, specify 'target' or 'background' channel type
                              Required for ISAC calibration, ignored for communication-only mode
        """
        self.h5_file_path = Path(h5_file_path)
        self.cluster_params: Optional[ClusterParams] = None
        self.link_params: List[LinkParams] = []
        self.cell_params: List[CellParams] = []
        self.active_link_params: List[ActiveLinkParams] = []
        self.antenna_panels: Dict[int, AntennaPanel] = {}
        self.cir_per_cell: Dict[int, np.ndarray] = {}
        self.cir_ntaps_per_cell: Dict[int, np.ndarray] = {}  # {cell_id: ntaps array}
        self.system_level_config: Optional[SystemLevelConfig] = None
        self.sim_config: Optional[SimConfig] = None
        self.isac_config: Optional[ISACConfig] = None
        self.calibration_phase = calibration_phase
        self.isac_channel_type = isac_channel_type  # 'target' or 'background'
        # Store UE-to-row mappings if available from H5 file
        self.ue_row_mappings: Dict[int, Dict[int, int]] = {}  # {cell_id: {ue_id: row_idx}}
        # 3GPP reference data
        self.reference_data: Optional[ThreeGPPReferenceData] = None
        if reference_json_path:
            self.reference_data = ThreeGPPReferenceData(reference_json_path, calibration_phase)
        
    def _get_ue_row_mapping(self, cell_id: int, ue_idx: int) -> Optional[int]:
        """Get the row index for a specific UE in a cell's CIR data.
        
        Args:
            cell_id: The cell ID
            ue_idx: The UE index
            
        Returns:
            Row index if mapping exists, None otherwise
        """
        if cell_id in self.ue_row_mappings:
            return self.ue_row_mappings[cell_id].get(ue_idx)
        return None
        
    def _load_ue_row_mapping(self, cir_group: h5py.Group, cell_id: int) -> None:
        """Load UE-to-row mapping from H5 file for a specific cell.
        
        Args:
            cir_group: The CIR group from H5 file
            cell_id: The cell ID to load mapping for
        """
        # Look for UE mapping dataset (e.g., "ue_mapping_cell0" or "ue_indices_cell0")
        mapping_dataset_names = [
            f"ue_mapping_cell{cell_id}",
            f"ue_indices_cell{cell_id}",
            f"ue_row_mapping_cell{cell_id}"
        ]
        
        for dataset_name in mapping_dataset_names:
            if dataset_name in cir_group:
                try:
                    # Load UE indices array - should be ordered by row position
                    ue_indices = np.array(cir_group[dataset_name])
                    
                    # Create mapping: {ue_id: row_index}
                    if cell_id not in self.ue_row_mappings:
                        self.ue_row_mappings[cell_id] = {}
                    
                    for row_idx, ue_id in enumerate(ue_indices):
                        self.ue_row_mappings[cell_id][int(ue_id)] = row_idx
                    
                    logger.info(f"Loaded UE-to-row mapping for cell {cell_id}: {len(ue_indices)} UEs")
                    return
                    
                except Exception as e:
                    logger.warning(f"Could not load UE mapping from {dataset_name}: {e}")
        
        # If no explicit mapping found, log a one-time warning
        if not hasattr(self, '_ue_mapping_warning_shown'):
            self._ue_mapping_warning_shown = True
            logger.warning(f"  No explicit UE-to-row mapping found in H5 file. "
                          f"Using fallback ordering which may be less accurate. "
                          f"Consider adding 'ue_mapping_cellX' datasets to H5 file for reliable CIR access.")
        logger.debug(f"No explicit UE-to-row mapping found for cell {cell_id}.")
        
    def inspect_h5_structure(self) -> None:
        """Inspect and log the structure of the H5 file"""
        logger.info(f"Inspecting H5 file structure: {self.h5_file_path}")
        
        def print_structure(name, obj):
            if isinstance(obj, h5py.Group):
                logger.info(f"Group: {name}")
            elif isinstance(obj, h5py.Dataset):
                logger.info(f"Dataset: {name}, shape: {obj.shape}, dtype: {obj.dtype}")
        
        try:
            with h5py.File(self.h5_file_path, 'r') as f:
                logger.info("H5 file contents:")
                f.visititems(print_structure)
        except Exception as e:
            logger.error(f"Error inspecting H5 file: {e}")
    
    def parse_h5_file(self) -> None:
        """Parse the H5 file directly and extract all relevant data"""
        logger.info(f"Parsing H5 file: {self.h5_file_path}")
        
        # First inspect the structure
        self.inspect_h5_structure()
        
        try:
            with h5py.File(self.h5_file_path, 'r') as f:
                # Parse different datasets
                self._parse_active_link_params_h5(f)
                self._parse_cell_params_h5(f)
                self._parse_ut_params_h5(f)
                self._parse_cluster_params_h5(f)
                self._parse_link_params_h5(f)
                self._parse_antenna_panels_h5(f)
                self._parse_cir_data_h5(f)
                self._parse_system_config_h5(f)
                self._parse_sim_config_h5(f)
                
            logger.info(f"Parsed {len(self.active_link_params)} active links, "
                       f"{len(self.cell_params)} cells, {len(self.link_params)} link parameters, "
                       f"{len(self.antenna_panels)} antenna panels")
        except Exception as e:
            logger.error(f"Error parsing H5 file: {e}")
            raise
    
    def _parse_active_link_params_h5(self, h5_file: h5py.File) -> None:
        """Parse active link parameters from H5 file"""
        if 'activeLinkParams' in h5_file:
            dataset = h5_file['activeLinkParams']
            for i, record in enumerate(dataset):
                self.active_link_params.append(ActiveLinkParams(
                    cid=int(record['cid']),
                    uid=int(record['uid']),
                    link_idx=int(record['linkIdx']),
                    lsp_read_idx=int(record['lspReadIdx'])
                ))
        else:
            logger.warning("activeLinkParams dataset not found in H5 file")
    
    def _parse_cell_params_h5(self, h5_file: h5py.File) -> None:
        """Parse cell parameters from H5 file (now part of topology structure)"""
        # Try new topology structure first
        if 'topology' in h5_file and 'cellParams' in h5_file['topology']:
            dataset = h5_file['topology']['cellParams']
            for record in dataset:
                # Handle nested location structure
                if hasattr(record['loc'], 'dtype') and record['loc'].dtype.names:
                    # Structured array
                    loc = record['loc']
                    location = {'x': float(loc['x']), 'y': float(loc['y']), 'z': float(loc['z'])}
                else:
                    # Simple array [x, y, z]
                    loc = record['loc']
                    location = {'x': float(loc[0]), 'y': float(loc[1]), 'z': float(loc[2])}
                
                # Parse antenna orientation if available
                ant_orientation = None
                if 'antPanelOrientation' in record.dtype.names:
                    orient = record['antPanelOrientation']
                    ant_orientation = [float(orient[0]), float(orient[1]), float(orient[2])]
                
                self.cell_params.append(CellParams(
                    cid=int(record['cid']),
                    site_id=int(record['siteId']),
                    location=location,
                    ant_panel_idx=int(record['antPanelIdx']),
                    ant_panel_orientation=ant_orientation
                ))
        # Fallback to old structure for backward compatibility
        elif 'cellParams' in h5_file:
            dataset = h5_file['cellParams']
            for record in dataset:
                # Handle nested location structure
                if hasattr(record['loc'], 'dtype') and record['loc'].dtype.names:
                    # Structured array
                    loc = record['loc']
                    location = {'x': float(loc['x']), 'y': float(loc['y']), 'z': float(loc['z'])}
                else:
                    # Simple array [x, y, z]
                    loc = record['loc']
                    location = {'x': float(loc[0]), 'y': float(loc[1]), 'z': float(loc[2])}
                
                # Parse antenna orientation if available
                ant_orientation = None
                if 'antPanelOrientation' in record.dtype.names:
                    orient = record['antPanelOrientation']
                    ant_orientation = [float(orient[0]), float(orient[1]), float(orient[2])]
                
                self.cell_params.append(CellParams(
                    cid=int(record['cid']),
                    site_id=int(record['siteId']),
                    location=location,
                    ant_panel_idx=int(record['antPanelIdx']),
                    ant_panel_orientation=ant_orientation
                ))
        else:
            logger.warning("cellParams dataset not found in H5 file (checked both topology/cellParams and cellParams)")
    
    def _parse_cluster_params_h5(self, h5_file: h5py.File) -> None:
        """Parse cluster parameters from H5 file - stores only first record for backward compatibility"""
        if 'clusterParams' in h5_file:
            dataset = h5_file['clusterParams']
            if len(dataset) > 0:
                record = dataset[0]  # Assuming first record contains cluster parameters
                
                self.cluster_params = ClusterParams(
                    n_cluster=int(record['nCluster']),
                    n_ray_per_cluster=int(record['nRayPerCluster']),
                    delays=record['delays'].tolist(),
                    powers=record['powers'].tolist(),
                    strongest_2_clusters_idx=record['strongest2clustersIdx'].tolist() if 'strongest2clustersIdx' in record.dtype.names else [],
                    phi_n_aoa=record['phi_n_AoA'].tolist() if 'phi_n_AoA' in record.dtype.names else [],
                    phi_n_aod=record['phi_n_AoD'].tolist() if 'phi_n_AoD' in record.dtype.names else [],
                    theta_n_zod=record['theta_n_ZOD'].tolist() if 'theta_n_ZOD' in record.dtype.names else [],
                    theta_n_zoa=record['theta_n_ZOA'].tolist() if 'theta_n_ZOA' in record.dtype.names else [],
                    xpr=record['xpr'].tolist() if 'xpr' in record.dtype.names else [],
                    random_phases=record['randomPhases'].tolist() if 'randomPhases' in record.dtype.names else [],
                    # Per-ray angles
                    phi_n_m_aoa=record['phi_n_m_AoA'].tolist() if 'phi_n_m_AoA' in record.dtype.names else [],
                    phi_n_m_aod=record['phi_n_m_AoD'].tolist() if 'phi_n_m_AoD' in record.dtype.names else [],
                    theta_n_m_zod=record['theta_n_m_ZOD'].tolist() if 'theta_n_m_ZOD' in record.dtype.names else [],
                    theta_n_m_zoa=record['theta_n_m_ZOA'].tolist() if 'theta_n_m_ZOA' in record.dtype.names else []
                )
        else:
            logger.warning("clusterParams dataset not found in H5 file")
    
    def _load_all_cluster_params(self) -> List[Dict]:
        """Load all cluster parameters from H5 file (one per link)"""
        cluster_params_list = []
        
        with h5py.File(self.h5_file_path, 'r') as h5_file:
            if 'clusterParams' in h5_file:
                dataset = h5_file['clusterParams']
                
                for record in dataset:
                    cluster_param = {
                        'n_cluster': int(record['nCluster']),
                        'n_ray_per_cluster': int(record['nRayPerCluster']),
                        'delays': record['delays'].tolist(),
                        'powers': record['powers'].tolist(),
                        'strongest_2_clusters_idx': record['strongest2clustersIdx'].tolist() if 'strongest2clustersIdx' in record.dtype.names else [],
                        'phi_n_aoa': record['phi_n_AoA'].tolist() if 'phi_n_AoA' in record.dtype.names else [],
                        'phi_n_aod': record['phi_n_AoD'].tolist() if 'phi_n_AoD' in record.dtype.names else [],
                        'theta_n_zod': record['theta_n_ZOD'].tolist() if 'theta_n_ZOD' in record.dtype.names else [],
                        'theta_n_zoa': record['theta_n_ZOA'].tolist() if 'theta_n_ZOA' in record.dtype.names else [],
                        'xpr': record['xpr'].tolist() if 'xpr' in record.dtype.names else [],
                        'random_phases': record['randomPhases'].tolist() if 'randomPhases' in record.dtype.names else [],
                        # Per-ray angles
                        'phi_n_m_aoa': record['phi_n_m_AoA'].tolist() if 'phi_n_m_AoA' in record.dtype.names else [],
                        'phi_n_m_aod': record['phi_n_m_AoD'].tolist() if 'phi_n_m_AoD' in record.dtype.names else [],
                        'theta_n_m_zod': record['theta_n_m_ZOD'].tolist() if 'theta_n_m_ZOD' in record.dtype.names else [],
                        'theta_n_m_zoa': record['theta_n_m_ZOA'].tolist() if 'theta_n_m_ZOA' in record.dtype.names else []
                    }
                    cluster_params_list.append(cluster_param)
            else:
                logger.warning("clusterParams dataset not found in H5 file")
        
        return cluster_params_list
    
    def _parse_link_params_h5(self, h5_file: h5py.File) -> None:
        """Parse link parameters from H5 file"""
        if 'linkParams' in h5_file:
            dataset = h5_file['linkParams']
            for record in dataset:
                self.link_params.append(LinkParams(
                    d2d=float(record['d2d']),
                    d2d_in=float(record['d2d_in']),
                    d2d_out=float(record['d2d_out']),
                    d3d=float(record['d3d']),
                    d3d_in=float(record['d3d_in']),
                    phi_los_aod=float(record['phi_LOS_AOD']),
                    theta_los_zod=float(record['theta_LOS_ZOD']),
                    phi_los_aoa=float(record['phi_LOS_AOA']),
                    theta_los_zoa=float(record['theta_LOS_ZOA']),
                    los_ind=int(record['losInd']),
                    pathloss=float(record['pathloss']),
                    sf=float(record['SF']),
                    k_factor=float(record['K']),
                    ds=float(record['DS']),
                    asd=float(record['ASD']),
                    asa=float(record['ASA']),
                    mu_lg_zsd=float(record['mu_lgZSD']),
                    sigma_lg_zsd=float(record['sigma_lgZSD']),
                    mu_offset_zod=float(record['mu_offset_ZOD']),
                    zsd=float(record['ZSD']),
                    zsa=float(record['ZSA'])
                ))
        else:
            logger.warning("linkParams dataset not found in H5 file")
    
    def _parse_antenna_panels_h5(self, h5_file: h5py.File) -> None:
        """Parse antenna panel configurations from H5 file"""
        if 'antennaPanels' in h5_file:
            dataset = h5_file['antennaPanels']
            logger.info(f"Found antenna panels dataset with {len(dataset)} panels")
            
            for panel_idx, record in enumerate(dataset):
                antenna_panel = AntennaPanel(
                    panel_idx=panel_idx,
                    n_antennas=int(record['nAnt']),
                    ant_model=int(record['antModel']),
                    ant_size=record['antSize'].tolist(),
                    ant_spacing=record['antSpacing'].tolist(),
                    ant_polar_angles=record['antPolarAngles'].tolist()
                )
                self.antenna_panels[antenna_panel.panel_idx] = antenna_panel
                
                # Detailed antenna panel information
                gain_db = antenna_panel.antenna_gain_db()
                logger.info(f"Panel {antenna_panel.panel_idx} Configuration:")
                logger.info(f"  -> Antennas: {antenna_panel.total_antennas}")
                logger.info(f"  -> Model: {antenna_panel.ant_model}")
                logger.info(f"  -> Size array: {antenna_panel.ant_size}")
                logger.info(f"  -> Calculated gain: {gain_db:.1f} dB")
                
                if antenna_panel.total_antennas == 1:
                    logger.warning(f"  -> WARNING: Single antenna detected! This limits diversity and beamforming gains")
                    logger.info(f"  -> Potential improvements:")
                    logger.info(f"     - 2x2 MIMO would give ~3 dB gain (10*log10(4) = 6 dB)")
                    logger.info(f"     - 4x4 MIMO would give ~6 dB gain (10*log10(16) = 12 dB)")
                elif antenna_panel.total_antennas < 4:
                    logger.info(f"  -> Limited MIMO configuration - consider more antennas for better performance")
        else:
            # Create default antenna panel configurations if not present in H5
            logger.warning("antennaPanels dataset not found in H5 file, creating default configurations")
            self._create_default_antenna_panels()
    
    def _create_default_antenna_panels(self) -> None:
        """Create default antenna panel configurations based on typical 5G deployments"""
        # Default configurations for different panel types
        default_panels = [
            # Panel 0: Typical macro cell panel
            AntennaPanel(
                panel_idx=0,
                n_antennas=64,  # 8x8 array
                ant_model=1,
                ant_size=[8, 8, 1, 1, 1],
                ant_spacing=[0.0, 0.0, 0.5, 0.5],  # lambda/2 spacing
                ant_polar_angles=[45.0, -45.0]
            ),
            # Panel 1: Small cell panel
            AntennaPanel(
                panel_idx=1,
                n_antennas=16,  # 4x4 array
                ant_model=1,
                ant_size=[4, 4, 1, 1, 1],
                ant_spacing=[0.0, 0.0, 0.5, 0.5],
                ant_polar_angles=[45.0, -45.0]
            ),
            # Panel 2: High-gain panel
            AntennaPanel(
                panel_idx=2,
                n_antennas=64,  # 16x4 array
                ant_model=1,
                ant_size=[16, 4, 1, 1, 1],
                ant_spacing=[0.0, 0.0, 0.5, 0.5],
                ant_polar_angles=[45.0, -45.0]
            )
        ]
        
        for panel in default_panels:
            self.antenna_panels[panel.panel_idx] = panel
            logger.info(f"Created default panel {panel.panel_idx}: {panel.total_antennas} antennas, "
                       f"gain: {panel.antenna_gain_db():.1f} dB")
    
    def _parse_cir_data_h5(self, h5_file: h5py.File) -> None:
        """Parse CIR data from H5 file if available"""
        if 'cirPerCell' in h5_file:
            cir_group = h5_file['cirPerCell']
            for dataset_name in cir_group.keys():
                try:
                    # Extract cell ID from dataset name (e.g., "cirCoe_cell0" -> 0)
                    if 'cirCoe_cell' in dataset_name:
                        cell_id_str = dataset_name.replace('cirCoe_cell', '')
                        cell_id = int(cell_id_str)
                        
                        # Read the CIR coefficient data
                        cell_data = cir_group[dataset_name]
                        cir_array = np.array(cell_data)
                        
                        # Handle complex data format - convert from structured array to complex
                        if cir_array.dtype.names and 'real' in cir_array.dtype.names and 'imag' in cir_array.dtype.names:
                            # Convert structured array (real, imag) to complex array
                            cir_complex = cir_array['real'] + 1j * cir_array['imag']
                            self.cir_per_cell[cell_id] = cir_complex
                        else:
                            self.cir_per_cell[cell_id] = cir_array
                        
                        logger.info(f"Loaded CIR data for cell {cell_id}: shape {self.cir_per_cell[cell_id].shape}")
                        
                        # Try to load UE-to-row mapping for this cell
                        self._load_ue_row_mapping(cir_group, cell_id)
                        
                        # Try to load ntaps data for this cell
                        ntaps_dataset_name = f"cirNtaps_cell{cell_id}"
                        if ntaps_dataset_name in cir_group:
                            try:
                                ntaps_data = np.array(cir_group[ntaps_dataset_name])
                                self.cir_ntaps_per_cell[cell_id] = ntaps_data
                                logger.info(f"Loaded CIR ntaps for cell {cell_id}: shape {ntaps_data.shape}")
                            except Exception as e:
                                logger.warning(f"Could not load ntaps data for cell {cell_id}: {e}")
                        
                except Exception as e:
                    logger.warning(f"Could not parse CIR data for dataset {dataset_name}: {e}")
        else:
            logger.info("No cirPerCell data found in H5 file")
    
    def _parse_ut_params_h5(self, h5_file: h5py.File) -> None:
        """Parse UT parameters from H5 file (now part of topology structure)"""
        # Initialize ut_params list
        self.ut_params = []
        
        # Try new topology structure first
        if 'topology' in h5_file and 'utParams' in h5_file['topology']:
            ut_group = h5_file['topology']['utParams']
            # Read all arrays
            uids = ut_group['uid'][:]
            locs_x = ut_group['loc_x'][:]
            locs_y = ut_group['loc_y'][:]
            locs_z = ut_group['loc_z'][:]
            outdoor_inds = ut_group['outdoor_ind'][:]
            ant_panel_idxs = ut_group['antPanelIdx'][:]
            velocities_x = ut_group['velocity_x'][:]
            velocities_y = ut_group['velocity_y'][:]
            velocities_z = ut_group['velocity_z'][:]
            d_2d_ins = ut_group['d_2d_in'][:]
            
            # Create UT parameter dictionaries
            for i in range(len(uids)):
                ut_param = {
                    'uid': int(uids[i]),
                    'location': {'x': float(locs_x[i]), 'y': float(locs_y[i]), 'z': float(locs_z[i])},
                    'outdoor_ind': int(outdoor_inds[i]),
                    'ant_panel_idx': int(ant_panel_idxs[i]),
                    'velocity': {'x': float(velocities_x[i]), 'y': float(velocities_y[i]), 'z': float(velocities_z[i])},
                    'd_2d_in': float(d_2d_ins[i])
                }
                self.ut_params.append(ut_param)
            
            logger.info(f"Parsed {len(uids)} UT parameters from topology structure")
        # Fallback to old structure for backward compatibility
        elif 'utParams' in h5_file:
            ut_group = h5_file['utParams']
            # Read all arrays
            uids = ut_group['uid'][:]
            locs_x = ut_group['loc_x'][:]
            locs_y = ut_group['loc_y'][:]
            locs_z = ut_group['loc_z'][:]
            outdoor_inds = ut_group['outdoor_ind'][:]
            ant_panel_idxs = ut_group['antPanelIdx'][:]
            velocities_x = ut_group['velocity_x'][:]
            velocities_y = ut_group['velocity_y'][:]
            velocities_z = ut_group['velocity_z'][:]
            d_2d_ins = ut_group['d_2d_in'][:]
            
            # Create UT parameter dictionaries
            for i in range(len(uids)):
                ut_param = {
                    'uid': int(uids[i]),
                    'location': {'x': float(locs_x[i]), 'y': float(locs_y[i]), 'z': float(locs_z[i])},
                    'outdoor_ind': int(outdoor_inds[i]),
                    'ant_panel_idx': int(ant_panel_idxs[i]),
                    'velocity': {'x': float(velocities_x[i]), 'y': float(velocities_y[i]), 'z': float(velocities_z[i])},
                    'd_2d_in': float(d_2d_ins[i])
                }
                self.ut_params.append(ut_param)
            
            logger.info(f"Parsed {len(uids)} UT parameters from old structure")
        else:
            logger.warning("utParams dataset not found in H5 file (checked both topology/utParams and utParams)")
    
    def _parse_system_config_h5(self, h5_file: h5py.File) -> None:
        """Parse SystemLevelConfig from H5 file"""
        if 'systemLevelConfig' in h5_file:
            dataset = h5_file['systemLevelConfig']
            if len(dataset) > 0:
                record = dataset[0]  # Single instance
                
                # Handle scenario enum conversion
                scenario_map = {0: 'UMa', 1: 'UMi', 2: 'RMa', 3: 'InH', 4: 'InF'}
                scenario_val = int(record['scenario'])
                scenario_str = scenario_map.get(scenario_val, f'Unknown({scenario_val})')
                
                ut_drop_option = int(record['ut_drop_option']) if 'ut_drop_option' in record.dtype.names else 0
                self.system_level_config = SystemLevelConfig(
                    scenario=scenario_str,
                    isd=float(record['isd']),
                    n_site=int(record['n_site']),
                    n_sector_per_site=int(record['n_sector_per_site']),
                    n_ut=int(record['n_ut']),
                    optional_pl_ind=int(record['optional_pl_ind']),
                    o2i_building_penetr_loss_ind=int(record['o2i_building_penetr_loss_ind']),
                    o2i_car_penetr_loss_ind=int(record['o2i_car_penetr_loss_ind']),
                    enable_near_field_effect=int(record['enable_near_field_effect']),
                    enable_non_stationarity=int(record['enable_non_stationarity']),
                    force_los_prob=record['force_los_prob'].tolist(),
                    force_ut_speed=record['force_ut_speed'].tolist(),
                    force_indoor_ratio=float(record['force_indoor_ratio']),
                    disable_pl_shadowing=int(record['disable_pl_shadowing']),
                    disable_small_scale_fading=int(record['disable_small_scale_fading']),
                    enable_per_tti_lsp=int(record['enable_per_tti_lsp']),
                    enable_propagation_delay=int(record['enable_propagation_delay']),
                    ut_drop_option=ut_drop_option
                )
                logger.info(f"Parsed SystemLevelConfig: {self.system_level_config.scenario} scenario, {self.system_level_config.n_site} sites, {self.system_level_config.n_ut} UTs")
                
                # Parse ISAC configuration fields (optional - may not exist in older H5 files)
                try:
                    isac_type = int(record['isac_type']) if 'isac_type' in record.dtype.names else 0
                    n_st = int(record['n_st']) if 'n_st' in record.dtype.names else 0
                    st_target_type = int(record['st_target_type']) if 'st_target_type' in record.dtype.names else 0
                    st_rcs_model = int(record['st_rcs_model']) if 'st_rcs_model' in record.dtype.names else 1
                    isac_disable_background = int(record['isac_disable_background']) if 'isac_disable_background' in record.dtype.names else 0
                    isac_disable_target = int(record['isac_disable_target']) if 'isac_disable_target' in record.dtype.names else 0
                    st_size_ind = int(record['st_size_ind']) if 'st_size_ind' in record.dtype.names else 0
                    
                    self.isac_config = ISACConfig(
                        isac_type=isac_type,
                        n_st=n_st,
                        st_target_type=st_target_type,
                        st_rcs_model=st_rcs_model,
                        isac_disable_background=isac_disable_background,
                        isac_disable_target=isac_disable_target,
                        st_size_ind=st_size_ind
                    )
                    
                    if self.isac_config.is_enabled:
                        mode_str = 'target-only' if self.isac_config.is_target_only else \
                                   'background-only' if self.isac_config.is_background_only else 'combined'
                        logger.info(f"Parsed ISACConfig: type={self.isac_config.isac_type} "
                                   f"({'monostatic' if self.isac_config.is_monostatic else 'bistatic'}), "
                                   f"n_st={self.isac_config.n_st}, "
                                   f"target_type={self.isac_config.target_type_name}, "
                                   f"mode={mode_str}")
                except Exception as e:
                    logger.debug(f"ISAC config fields not found in H5 (this is normal for non-ISAC simulations): {e}")
                    self.isac_config = ISACConfig()  # Default: communication only
        else:
            logger.warning("systemLevelConfig dataset not found in H5 file")

    def _compute_phase2_coupling_loss_from_cir(self, use_virtualization: bool = False) -> Optional[np.ndarray]:
        """
        Compute Phase 2 coupling loss from CIR data (includes fast fading).
        
        Phase 1 and Phase 2 are independent:
        - Phase 1: Large-scale only (PL + SF formula)
        - Phase 2: Includes fast fading from CIR
        
        CIR format in H5 (supported):
        - Legacy: [n_ue, n_snapshots, n_cir_coeff] where n_cir_coeff = n_rx_ant × n_tx_ant × n_max_taps (flattened)
        - New:    [n_ue, n_snapshots, n_rx_ant, n_tx_ant, n_max_taps]
        
        Args:
            use_virtualization: If True, use antenna virtualization (keep array gain).
                              If False, apply antenna normalization (divide by n_ant).
                              
                              - UMa calibration: use_virtualization=True (matches 3GPP reference)
                              - ISAC calibration: use_virtualization=False (per-antenna normalization)
        
        Returns:
            Array of coupling loss values (dB)
        """
        try:
            with h5py.File(self.h5_file_path, 'r') as f:
                coupling_losses = []
                
                # Iterate through all cells
                for cell_idx in range(self.system_level_config.n_site * self.system_level_config.n_sector_per_site):
                    cir_key = f'cirPerCell/cirCoe_cell{cell_idx}'
                    
                    if cir_key not in f:
                        continue
                    
                    # CIR data:
                    # - Legacy: [n_ue, n_snapshots, n_cir_coeff]
                    # - New:    [n_ue, n_snapshots, n_rx_ant, n_tx_ant, n_max_taps]
                    cir_data = f[cir_key][:]
                    
                    # Convert to complex if needed
                    if cir_data.dtype.names and 'real' in cir_data.dtype.names:
                        cir_complex = cir_data['real'] + 1j * cir_data['imag']
                    else:
                        cir_complex = cir_data
                    
                    logger.debug(f"Cell {cell_idx}: CIR shape = {cir_complex.shape}")
                    
                    # For each UE/RP in this cell
                    for ue_idx in range(cir_complex.shape[0]):
                        # ue_cir has snapshot dimension first:
                        # - Legacy: [n_snapshots, n_cir_coeff]
                        # - New:    [n_snapshots, n_rx_ant, n_tx_ant, n_max_taps]
                        ue_cir = cir_complex[ue_idx]
                        
                        # Step 1: Compute total power per snapshot (sum over all CIR taps and antennas)
                        # |H|^2 for each snapshot: [n_snapshots]
                        if ue_cir.ndim < 2:
                            power_per_snapshot = np.array([np.sum(np.abs(ue_cir) ** 2)])
                        else:
                            power_per_snapshot = np.sum(np.abs(ue_cir) ** 2, axis=tuple(range(1, ue_cir.ndim)))
                        
                        # Step 2: Average power across snapshots (ALWAYS enabled)
                        # This normalizes over n_snapshot_per_slot (e.g., 14 snapshots)
                        # CRITICAL: Use mean, not sum, to get stable power estimate
                        n_snapshots = len(power_per_snapshot)
                        avg_power = np.mean(power_per_snapshot)
                        
                        # Step 3: Apply antenna virtualization or normalization
                        if use_virtualization:
                            # Antenna virtualization: Keep array gain (for UMa calibration)
                            # This matches 3GPP reference calibration which includes antenna gain
                            final_power = avg_power
                            n_rx_ant = self.system_level_config.n_ue_ant if hasattr(self.system_level_config, 'n_ue_ant') else 2
                            n_tx_ant = self.system_level_config.n_bs_ant if hasattr(self.system_level_config, 'n_bs_ant') else 2
                            logger.debug(f"Using antenna virtualization: n_rx={n_rx_ant}, n_tx={n_tx_ant}, final_power={final_power:.2e}")
                        else:
                            # Antenna normalization: Divide by number of antenna pairs (for ISAC calibration)
                            # For dual-pol isotropic: n_rx_ant = 2, n_tx_ant = 2 → divide by 4
                            n_rx_ant = self.system_level_config.n_ue_ant if hasattr(self.system_level_config, 'n_ue_ant') else 2
                            n_tx_ant = self.system_level_config.n_bs_ant if hasattr(self.system_level_config, 'n_bs_ant') else 2
                            final_power = avg_power / (n_rx_ant * n_tx_ant)
                            logger.debug(f"Using antenna normalization: n_rx={n_rx_ant}, n_tx={n_tx_ant}, "
                                       f"avg_power={avg_power:.2e}, final_power={final_power:.2e}")
                        
                        if final_power > 0:
                            # Coupling loss in dB (reported as negative per 3GPP convention)
                            coupling_loss_db = 10.0 * np.log10(final_power)
                            coupling_losses.append(coupling_loss_db)
                            
                            logger.debug(f"Cell {cell_idx}, UE {ue_idx}: "
                                       f"n_snapshots={n_snapshots}, "
                                       f"n_rx_ant={n_rx_ant}, n_tx_ant={n_tx_ant}, "
                                       f"Avg power (over snapshots)={avg_power:.2e}, "
                                       f"Final power={final_power:.2e}, "
                                       f"Coupling loss={coupling_loss_db:.2f} dB")
                
                if coupling_losses:
                    logger.info(f"Computed Phase 2 coupling loss from CIR: {len(coupling_losses)} samples")
                    logger.info(f"  CL range: [{min(coupling_losses):.2f}, {max(coupling_losses):.2f}] dB")
                    logger.info(f"  CL mean: {np.mean(coupling_losses):.2f} dB")
                    return np.array(coupling_losses)
                else:
                    logger.warning("No valid CIR data found for Phase 2 coupling loss computation")
                    return None
                    
        except Exception as e:
            logger.error(f"Error computing Phase 2 coupling loss from CIR: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def _get_isac_coupling_loss(self) -> Optional[np.ndarray]:
        """
        Get ISAC coupling loss from H5 file.
        
        For target channel (isac_channel_type='target'):
            Reads from topology/isacTargetLinks/coupling_loss_db
            L = PL(d1) + PL(d2) + 10*log10(c²/(4πf)²) - 10*log10(σ_RCS,A) + SF1 + SF2
        
        For background channel (isac_channel_type='background'):
            Reads from topology/isacBackgroundLinks/coupling_loss_db
            L = PL(d) + SF (one-way path to Reference Point)
        
        Returns:
            Array of coupling loss values in dB, or None if not available
        """
        try:
            with h5py.File(self.h5_file_path, 'r') as f:
                # Check if background channel is requested
                if self.isac_channel_type == 'background':
                    if 'topology/isacBackgroundLinks/coupling_loss_db' in f:
                        coupling_loss = f['topology/isacBackgroundLinks/coupling_loss_db'][:]
                        
                        # Log the components if available
                        if 'topology/isacBackgroundLinks/pathloss' in f:
                            pathloss = f['topology/isacBackgroundLinks/pathloss'][:]
                            sf = f['topology/isacBackgroundLinks/shadow_fading'][:]
                            rp_z = f['topology/isacBackgroundLinks/rp_loc_z'][:]
                            d_3d = f['topology/isacBackgroundLinks/d_3d'][:]
                            
                            logger.info("=== ISAC Background Link Parameters ===")
                            for i in range(min(5, len(coupling_loss))):
                                logger.info(f"  RP {i}: height={rp_z[i]:.1f}m, d_3d={d_3d[i]:.1f}m")
                                logger.info(f"    PL={pathloss[i]:.2f}dB, SF={sf[i]:.2f}dB")
                                logger.info(f"    Coupling loss={coupling_loss[i]:.2f}dB")
                        
                        logger.info(f"Loaded {len(coupling_loss)} ISAC background coupling loss values from H5")
                        return coupling_loss
                    else:
                        logger.warning("ISAC background coupling loss data not found in H5 file")
                        logger.warning("Please rebuild C++ and re-run simulation to generate background channel data")
                        return None
                
                # Target channel (default)
                if 'topology/isacTargetLinks/coupling_loss_db' in f:
                    coupling_loss = f['topology/isacTargetLinks/coupling_loss_db'][:]
                    
                    # Log the components if available
                    if 'topology/isacTargetLinks/incident_pathloss' in f:
                        incident_pl = f['topology/isacTargetLinks/incident_pathloss'][:]
                        scattered_pl = f['topology/isacTargetLinks/scattered_pathloss'][:]
                        incident_sf = f['topology/isacTargetLinks/incident_sf'][:]
                        scattered_sf = f['topology/isacTargetLinks/scattered_sf'][:]
                        rcs_dbsm = f['topology/isacTargetLinks/rcs_dbsm'][:]
                        target_z = f['topology/isacTargetLinks/target_loc_z'][:]
                        wavelength_term = f['topology/isacTargetLinks/wavelength_term_db'][()]
                        
                        logger.info("=== ISAC Target Link Parameters ===")
                        for i in range(min(5, len(coupling_loss))):
                            logger.info(f"  Link {i}: target_height={target_z[i]:.1f}m")
                            logger.info(f"    PL(d1)={incident_pl[i]:.2f}dB, SF1={incident_sf[i]:.2f}dB")
                            logger.info(f"    PL(d2)={scattered_pl[i]:.2f}dB, SF2={scattered_sf[i]:.2f}dB")
                            logger.info(f"    RCS={rcs_dbsm[i]:.2f}dBsm, wavelength_term={wavelength_term:.2f}dB")
                            logger.info(f"    Coupling loss={coupling_loss[i]:.2f}dB")
                    
                    logger.info(f"Loaded {len(coupling_loss)} ISAC coupling loss values from H5: {coupling_loss}")
                    return coupling_loss
                else:
                    logger.warning("ISAC coupling loss data not found in H5 file")
                    logger.warning("Please rebuild C++ and re-run simulation to generate ISAC data")
                    return None
        except Exception as e:
            logger.error(f"Error reading ISAC coupling loss: {e}")
            return None

    def _parse_sim_config_h5(self, h5_file: h5py.File) -> None:
        """Parse SimConfig from H5 file"""
        if 'simConfig' in h5_file:
            dataset = h5_file['simConfig']
            if len(dataset) > 0:
                record = dataset[0]  # Single instance
                
                self.sim_config = SimConfig(
                    link_sim_ind=int(record['link_sim_ind']),
                    center_freq_hz=float(record['center_freq_hz']),
                    bandwidth_hz=float(record['bandwidth_hz']),
                    sc_spacing_hz=float(record['sc_spacing_hz']),
                    fft_size=int(record['fft_size']),
                    n_prb=int(record['n_prb']),
                    n_prbg=int(record['n_prbg']),
                    n_snapshot_per_slot=int(record['n_snapshot_per_slot']),
                    run_mode=int(record['run_mode']),
                    internal_memory_mode=int(record['internal_memory_mode']),
                    freq_convert_type=int(record['freq_convert_type']),
                    sc_sampling=int(record['sc_sampling']),
                    proc_sig_freq=int(record['proc_sig_freq'])
                )
                logger.info(f"Parsed SimConfig: {self.sim_config.center_freq_hz/1e9:.1f} GHz, {self.sim_config.bandwidth_hz/1e6:.0f} MHz, run_mode={self.sim_config.run_mode}")
        else:
            logger.warning("simConfig dataset not found in H5 file")

    
    def compute_cir_power(self) -> Dict[int, float]:
        """
        Compute Channel Impulse Response (CIR) power for each cell.
        This is used for cell association as specified in the requirements.
        """
        logger.info("Computing CIR power for cell association")
        
        cir_powers = {}
        
        # If we have actual CIR data, use it
        if self.cir_per_cell:
            for cell_id, cir_data in self.cir_per_cell.items():
                # Compute power as sum of squared magnitudes
                if len(cir_data.shape) == 1:
                    # Real-valued CIR
                    power = np.sum(cir_data**2)
                else:
                    # Complex-valued CIR
                    power = np.sum(np.abs(cir_data)**2)
                cir_powers[cell_id] = float(power)
        elif self.cluster_params:
            # Fallback: use cluster powers as proxy for CIR power
            for cell in self.cell_params:
                total_power = sum(self.cluster_params.powers)
                cir_powers[cell.cid] = total_power
        else:
            # Last resort: use path loss
            for i, cell in enumerate(self.cell_params):
                if i < len(self.link_params):
                    # Convert path loss to received power (higher path loss = lower power)
                    power = 10**(-self.link_params[i].pathloss / 10)
                    cir_powers[cell.cid] = power
                else:
                    cir_powers[cell.cid] = 1.0  # Default value
            
        return cir_powers
    
    def find_serving_cell(self, association_method: str = 'rsrp') -> int:
        """
        Find the serving cell based on specified association method.
        
        Args:
            association_method: Association method to use:
                - 'rsrp': RSRP-based (pathloss - shadow fading) [default]
                - 'cir': CIR power-based (legacy)
                - 'distance': Minimal distance-based
            
        Returns:
            Cell ID of the serving cell
        """
        if association_method == 'cir':
            return self._find_serving_cell_by_cir()
        elif association_method == 'distance':
            return self._find_serving_cell_by_distance()
        else:  # default to 'rsrp'
            return self._find_serving_cell_by_rsrp()
    
    def _find_serving_cell_by_cir(self) -> int:
        """Find serving cell based on CIR power (legacy method)"""
        cir_powers = self.compute_cir_power()
        if not cir_powers:
            logger.warning("No CIR powers computed, using cell 0 as serving cell")
            return 0
            
        serving_cell = max(cir_powers.items(), key=lambda x: x[1])[0]
        logger.info(f"Serving cell (CIR-based): {serving_cell}")
        return serving_cell
    
    def _calculate_azimuth_angle(self, from_x: float, from_y: float, to_x: float, to_y: float) -> float:
        """
        Calculate azimuth angle from point (from_x, from_y) to point (to_x, to_y) in degrees.
        Returns angle in range [0, 360) degrees, where 0° = North, 90° = East.
        """
        dx = to_x - from_x
        dy = to_y - from_y
        azimuth_rad = np.arctan2(dx, dy)  # Note: (x, y) order for East=0, North=90 convention
        azimuth_deg = np.degrees(azimuth_rad)
        
        # Wrap to [0, 360) range
        if azimuth_deg < 0:
            azimuth_deg += 360.0
        
        return azimuth_deg
    
    def _calculate_azimuth_difference(self, angle1: float, angle2: float) -> float:
        """
        Calculate the absolute angular difference between two azimuth angles.
        Handles wrap-around correctly (e.g., 350° and 10° are 20° apart, not 340°).
        
        Args:
            angle1, angle2: Azimuth angles in degrees [0, 360)
            
        Returns:
            Absolute angular difference in degrees [0, 180]
        """
        diff = abs(angle1 - angle2)
        if diff > 180:
            diff = 360 - diff
        return diff
    
    def _find_serving_cell_by_rsrp(self) -> int:
        """
        Find serving cell based on RSRP with azimuth-based tie-breaking for co-sited cells.
        
        For co-sited cells (same pathloss and shadow fading), selects the cell whose
        antenna boresight is closest to the UE direction.
        """
        if not self.active_link_params or not self.link_params:
            logger.warning("No link parameters available, using cell 0 as serving cell")
            return 0
        
        # Group links by UE and find best serving cell for each UE
        ue_best_cells = {}
        
        for active_link in self.active_link_params:
            if active_link.lsp_read_idx < len(self.link_params):
                link = self.link_params[active_link.lsp_read_idx]
                
                # Calculate RSRP = -(Pathloss - Shadow_Fading) (higher is better)
                rsrp = -(link.pathloss - link.sf)
                
                ue_id = active_link.uid
                cid = active_link.cid
                
                # For co-sited cells with identical RSRP, use azimuth alignment
                should_update = False
                if ue_id not in ue_best_cells:
                    should_update = True
                else:
                    prev_cid, prev_rsrp, prev_az_diff = ue_best_cells[ue_id]
                    rsrp_diff = abs(rsrp - prev_rsrp)
                    
                    if rsrp_diff < 0.1:  # RSRP values within 0.1 dB (likely co-sited)
                        # Use azimuth alignment to break tie
                        az_diff = self._get_azimuth_alignment_for_link(active_link, link)
                        if az_diff is not None and (prev_az_diff is None or az_diff < prev_az_diff):
                            should_update = True
                            ue_best_cells[ue_id] = (cid, rsrp, az_diff)
                        elif prev_az_diff is None:
                            ue_best_cells[ue_id] = (prev_cid, prev_rsrp, az_diff)
                    elif rsrp > prev_rsrp:
                        # Clear winner by RSRP
                        az_diff = self._get_azimuth_alignment_for_link(active_link, link)
                        should_update = True
                        ue_best_cells[ue_id] = (cid, rsrp, az_diff)
                
                if should_update and ue_id not in ue_best_cells:
                    az_diff = self._get_azimuth_alignment_for_link(active_link, link)
                    ue_best_cells[ue_id] = (cid, rsrp, az_diff)
        
        # Find the most common serving cell
        if ue_best_cells:
            serving_cells = [cell_info[0] for cell_info in ue_best_cells.values()]
            serving_cell_id = max(set(serving_cells), key=serving_cells.count)
            avg_rsrp = np.mean([cell_info[1] for ue_id, cell_info in ue_best_cells.items() 
                               if cell_info[0] == serving_cell_id])
            logger.info(f"Serving cell (RSRP+Azimuth): Cell {serving_cell_id} (avg RSRP: {avg_rsrp:.2f} dBm)")
            return serving_cell_id
        else:
            logger.warning("No valid links found, using cell 0 as serving cell")
            return 0
    
    def _get_azimuth_alignment_for_link(self, active_link: ActiveLinkParams, link: LinkParams) -> float:
        """
        Calculate azimuth alignment between UE direction and BS antenna boresight.
        
        Uses phi_los_aod (azimuth of departure from BS) and compares with BS antenna
        boresight orientation. Smaller angular difference means better alignment.
        
        Returns:
            Angular difference in degrees [0, 180], or None if data unavailable
        """
        # Find cell parameters
        cell = next((c for c in self.cell_params if c.cid == active_link.cid), None)
        if cell is None or cell.ant_panel_orientation is None:
            return None
        
        # Get azimuth from BS to UE (phi_los_aod is already in degrees)
        ue_direction_azimuth = link.phi_los_aod
        
        # Normalize to [0, 360) range
        while ue_direction_azimuth < 0:
            ue_direction_azimuth += 360
        while ue_direction_azimuth >= 360:
            ue_direction_azimuth -= 360
        
        # Get BS antenna boresight azimuth (index 1 in orientation array)
        bs_boresight_azimuth = cell.ant_panel_orientation[1]
        
        # Calculate angular difference (minimum angle between two directions)
        angular_diff = self._calculate_azimuth_difference(ue_direction_azimuth, bs_boresight_azimuth)
        
        return angular_diff
    
    def _find_serving_cell_by_distance(self) -> int:
        """
        Find serving cell based on minimal 3D distance with azimuth-based tie-breaking.
        
        For co-sited cells (same distance), selects the cell whose antenna boresight
        is closest to the UE direction.
        """
        if not self.active_link_params or not self.link_params:
            logger.warning("No link parameters available, using cell 0 as serving cell")
            return 0
        
        # Group links by UE and find closest cell for each UE
        ue_closest_cells = {}
        
        for active_link in self.active_link_params:
            if active_link.lsp_read_idx < len(self.link_params):
                link = self.link_params[active_link.lsp_read_idx]
                
                # Use 3D distance (d3d) - lower is better
                distance = link.d3d
                
                ue_id = active_link.uid
                cid = active_link.cid
                
                # For co-sited cells with identical distance, use azimuth alignment
                should_update = False
                if ue_id not in ue_closest_cells:
                    should_update = True
                else:
                    prev_cid, prev_distance, prev_az_diff = ue_closest_cells[ue_id]
                    distance_diff = abs(distance - prev_distance)
                    
                    if distance_diff < 0.1:  # Distances within 0.1m (co-sited cells)
                        # Use azimuth alignment to break tie
                        az_diff = self._get_azimuth_alignment_for_link(active_link, link)
                        if az_diff is not None and (prev_az_diff is None or az_diff < prev_az_diff):
                            should_update = True
                            ue_closest_cells[ue_id] = (cid, distance, az_diff)
                        elif prev_az_diff is None:
                            ue_closest_cells[ue_id] = (prev_cid, prev_distance, az_diff)
                    elif distance < prev_distance:
                        # Clear winner by distance
                        az_diff = self._get_azimuth_alignment_for_link(active_link, link)
                        should_update = True
                        ue_closest_cells[ue_id] = (cid, distance, az_diff)
                
                if should_update and ue_id not in ue_closest_cells:
                    az_diff = self._get_azimuth_alignment_for_link(active_link, link)
                    ue_closest_cells[ue_id] = (cid, distance, az_diff)
        
        # Find the most common serving cell
        if ue_closest_cells:
            serving_cells = [cell_info[0] for cell_info in ue_closest_cells.values()]
            serving_cell_id = max(set(serving_cells), key=serving_cells.count)
            avg_distance = np.mean([cell_info[1] for ue_id, cell_info in ue_closest_cells.items() 
                                   if cell_info[0] == serving_cell_id])
            logger.info(f"Serving cell (Distance+Azimuth): Cell {serving_cell_id} (avg distance: {avg_distance:.2f} m)")
            return serving_cell_id
        else:
            logger.warning("No valid links found, using cell 0 as serving cell")
            return 0
    
    def get_antenna_gain_for_cell(self, cell_id: int, ue_id: int = None, 
                                  use_virtualization: bool = False) -> float:
        """
        Get antenna gain in dB for a specific cell based on its antenna panel configuration.
        
        Args:
            cell_id: Cell ID to get antenna gain for
            ue_id: UE ID for link-specific angle calculation (optional, used for Phase 2 virtualization)
            use_virtualization: If True, use antenna virtualization based on calibration phase
            
        Returns:
            Antenna gain in dB
        """
        # Find the cell parameters
        cell_param = None
        for cell in self.cell_params:
            if cell.cid == cell_id:
                cell_param = cell
                break
        
        if cell_param is None:
            logger.warning(f"Cell {cell_id} not found in cell parameters, using 0 dB antenna gain")
            return 0.0
        
        # Get antenna panel for this cell
        ant_panel_idx = cell_param.ant_panel_idx
        if ant_panel_idx not in self.antenna_panels:
            logger.warning(f"Antenna panel {ant_panel_idx} not found for cell {cell_id}, using 0 dB antenna gain")
            return 0.0
        
        antenna_panel = self.antenna_panels[ant_panel_idx]
        
        if not use_virtualization:
            # Simple gain without virtualization
            gain_db = antenna_panel.antenna_gain_db(use_calibration_values=False)
            logger.debug(f"Cell {cell_id} uses antenna panel {ant_panel_idx} with {antenna_panel.total_antennas} antennas, gain: {gain_db:.1f} dB (no virtualization)")
            return gain_db
        
        # Get link parameters for angle information (if ue_id is provided)
        angle_aod_deg = 0.0
        tilt_deg = 12.0  # Default tilt
        sector_orientation_deg = 0.0  # Default orientation
        
        # Get sector orientation from cell antenna panel orientation
        if cell_param.ant_panel_orientation is not None and len(cell_param.ant_panel_orientation) >= 2:
            sector_orientation_deg = cell_param.ant_panel_orientation[1]  # Azimuth angle
        
        # For Phase 2, get the angle of departure for this specific UE-cell link
        if self.calibration_phase == 2 and ue_id is not None:
            # Find the link parameters for this UE-cell pair
            for active_link in self.active_link_params:
                if active_link.cid == cell_id and active_link.uid == ue_id:
                    if active_link.lsp_read_idx < len(self.link_params):
                        link = self.link_params[active_link.lsp_read_idx]
                        angle_aod_deg = link.phi_los_aod  # Azimuth angle of departure
                        break
        
        # Calculate virtualized antenna gain
        gain_db = antenna_panel.antenna_gain_db(
            use_calibration_values=True,
            calibration_phase=self.calibration_phase,
            angle_aod_deg=angle_aod_deg,
            tilt_deg=tilt_deg,
            sector_orientation_deg=sector_orientation_deg
        )
        
        logger.debug(f"Cell {cell_id} (UE {ue_id}): Panel {ant_panel_idx}, Phase {self.calibration_phase} virtualization, "
                    f"AoD={angle_aod_deg:.1f}°, sector={sector_orientation_deg:.1f}°, gain={gain_db:.1f} dB")
        
        return gain_db

    def compute_coupling_loss_serving_cell(self, association_method: str = 'rsrp', 
                                          use_virtualization: bool = False) -> float:
        """
        Compute coupling loss for the serving cell.
        Coupling loss = Path loss - Shadow fading - BS Antenna gain
        
        Args:
            association_method: Association method to use:
                - 'rsrp': RSRP-based (pathloss - shadow fading) [default]
                - 'cir': CIR power-based (legacy)
                - 'distance': Minimal distance-based
            use_virtualization: If True, use antenna virtualization for gain calculation
        """
        serving_cell_id = self.find_serving_cell(association_method)
        
        # Find the link parameters for the serving cell
        serving_link = None
        serving_ue_id = None
        for active_link in self.active_link_params:
            if active_link.cid == serving_cell_id:
                # Get the corresponding link parameters
                if active_link.lsp_read_idx < len(self.link_params):
                    serving_link = self.link_params[active_link.lsp_read_idx]
                    serving_ue_id = active_link.uid
                    break
        
        if serving_link is None:
            logger.error(f"Could not find link parameters for serving cell {serving_cell_id}")
            return 0.0
        
        # Get antenna gain for the serving cell (with optional virtualization)
        antenna_gain_db = self.get_antenna_gain_for_cell(
            serving_cell_id, 
            ue_id=serving_ue_id,
            use_virtualization=use_virtualization
        )
        
        # Coupling loss = Path loss - Shadow fading - BS Antenna gain
        # (Lower coupling loss = better performance)
        coupling_loss = serving_link.pathloss - serving_link.sf - antenna_gain_db
        
        virt_label = f"(Phase {self.calibration_phase} virtualization)" if use_virtualization else "(no virtualization)"
        logger.info(f"Link Budget Analysis for Serving Cell {serving_cell_id} {virt_label}:")
        logger.info(f"  -> Path loss: {serving_link.pathloss:.2f} dB")
        logger.info(f"  -> Shadow fading: {serving_link.sf:.2f} dB")
        logger.info(f"  -> BS Antenna gain: {antenna_gain_db:.2f} dB")
        logger.info(f"  -> Coupling loss: {coupling_loss:.2f} dB")
        
        return coupling_loss
    
    def compute_wideband_sir(self, skip_isac_check: bool = False) -> float:
        """
        Compute wideband Signal-to-Interference Ratio (SIR) before receiver without noise.
        SIR = Signal power / Interference power
        """
        logger.info("Computing wideband SIR")
        
        serving_cell_id = self.find_serving_cell()
        
        # Get signal power from serving cell
        signal_power_db = None
        interference_power_db = []
        
        for active_link in self.active_link_params:
            if active_link.lsp_read_idx < len(self.link_params):
                link = self.link_params[active_link.lsp_read_idx]
                
                # Convert path loss to received power (assuming unit transmit power)
                received_power_db = -(link.pathloss - link.sf)
                
                if active_link.cid == serving_cell_id:
                    signal_power_db = received_power_db
                else:
                    interference_power_db.append(received_power_db)
        
        if signal_power_db is None:
            logger.error("Could not determine signal power")
            return 0.0
        
        if not interference_power_db:
            logger.warning("No interference sources found")
            return float('inf')
        
        # Convert to linear scale, sum interference, then back to dB
        signal_power_linear = 10**(signal_power_db / 10)
        interference_power_linear = sum(10**(p / 10) for p in interference_power_db)
        
        sir_linear = signal_power_linear / interference_power_linear
        sir_db = 10 * np.log10(sir_linear)
        
        logger.info(f"Wideband SIR: {sir_db:.2f} dB")
        return sir_db
    
    def compute_cluster_angle_spread_with_los(self, cluster_angles_deg: List[float], cluster_powers: List[float],
                                             los_angle_deg: float, los_power: float, is_zenith: bool = False) -> float:
        """
        Compute angle spread for LOS scenarios using TR 25.996 Annex A Equations A-1, A-3.
        
        This explicitly includes the LOS path power and angle separate from NLOS clusters.
        
        Per TR 25.996 Annex A:
        - Equation A-3: μ_θ = (Σ θ_i * P_i) / (Σ P_i)  [power-weighted mean]
        - Equation A-2: θ_{i,μ} = mod(θ_i - μ_θ + π, 2π) - π  [wrap angles relative to mean]
        - Equation A-1: σ_AS = sqrt(Σ (θ_{i,μ})² * P_i / Σ P_i)  [RMS spread]
        
        Args:
            cluster_angles_deg: List of NLOS cluster angles in degrees (not including LOS)
            cluster_powers: List of NLOS cluster powers in linear scale
            los_angle_deg: LOS path angle in degrees  
            los_power: LOS path power in linear scale
            is_zenith: If True, angles are zenith angles [0, 180]. If False, azimuth [-180, 180]
            
        Returns:
            Angle spread in degrees per TR 25.996 Annex A
        """
        if not cluster_angles_deg or len(cluster_angles_deg) == 0:
            # Only LOS, no spread
            return 0.0
        
        # Combine LOS and NLOS components
        all_angles_deg = np.concatenate([[los_angle_deg], np.array(cluster_angles_deg)])
        all_powers = np.concatenate([[los_power], np.array(cluster_powers)])
        
        # Convert to radians
        all_angles_rad = np.deg2rad(all_angles_deg)
        
        # Normalize powers
        total_power = np.sum(all_powers)
        if total_power == 0:
            return 0.0
        
        # Step 1: Calculate power-weighted mean angle (Equation A-3)
        mu_theta_rad = np.sum(all_angles_rad * all_powers) / total_power
        
        # Step 2: Wrap angles relative to mean (Equation A-2)
        # θ_{i,μ} = mod(θ_i - μ_θ + π, 2π) - π
        theta_wrapped_rad = np.mod(all_angles_rad - mu_theta_rad + np.pi, 2*np.pi) - np.pi
        
        # Step 3: Calculate RMS spread (Equation A-1)
        # σ_AS = sqrt(Σ (θ_{i,μ})² * P_i / Σ P_i)
        variance = np.sum((theta_wrapped_rad**2) * all_powers) / total_power
        rms_spread_rad = np.sqrt(variance)
        
        # Convert to degrees
        rms_spread_deg = np.rad2deg(rms_spread_rad)
        
        return rms_spread_deg
    
    def compute_circular_angle_spread(self, angles_deg: List[float], powers: List[float] = None, is_zenith: bool = False) -> float:
        """
        Compute circular angle spread according to 3GPP TR 25.996 Annex A specification.
        
        Implements the exact algorithm from equations (A-4) to (A-6):
        1. For each Delta: Calculate theta_n,m(Delta) = mod(theta_n,m + Delta + pi, 2*pi) - pi
        2. Calculate mu_theta(Delta) = Sum theta_n,m(Delta) * P_n,m / Sum P_n,m  
        3. Calculate theta_n,m,mu(Delta) = mod(theta_n,m(Delta) - mu_theta(Delta) + pi, 2*pi) - pi
        4. Calculate sigma_AS(Delta) = sqrt(Sum [theta_n,m,mu(Delta)]^2 * P_n,m / Sum P_n,m)
        5. Find sigma_AS = min_Delta sigma_AS(Delta)
        
        NOTE: This is for NLOS scenarios (Equation A-1). For LOS scenarios, use
        compute_cluster_angle_spread_with_los() which implements Equation A-3.
        
        Args:
            angles_deg: List of angles in degrees
            powers: Optional list of powers for weighted calculation (linear scale)
            is_zenith: If True, use Delta in [0, 180 degrees]. If False, use Delta in [-180, 180 degrees]
            
        Returns:
            Angle spread in degrees (pure 3GPP formula, no additional scaling)
        """
        if not angles_deg or len(angles_deg) < 2:
            return 0.0
        
        angles_deg = np.array(angles_deg)
        
        # Handle powers
        if powers is not None:
            powers = np.array(powers)
            if len(powers) != len(angles_deg):
                powers = powers[:len(angles_deg)]  # Truncate if needed
            # Normalize powers to sum to 1
            powers = powers / np.sum(powers) if np.sum(powers) > 0 else np.ones_like(powers) / len(powers)
        else:
            # Equal weights
            powers = np.ones_like(angles_deg) / len(angles_deg)
        
        # Apply 3GPP TR 25.996 Annex A circular angle spread calculation for ALL angles
        # Convert to radians for calculation
        angles_rad = angles_deg * np.pi / 180
        
        # Use 1-degree granularity for delta optimization as requested
        if is_zenith:
            # For zenith angles: delta range [0, 180 deg] since zenith is bounded [0, 180 deg]
            delta_values_deg = np.arange(0, 181, 1)  # 1-degree steps from 0 to 180 degrees
        else:
            # For azimuth angles: delta range [-180, 180 deg] for full circular coverage
            delta_values_deg = np.arange(-180, 181, 1)  # 1-degree steps from -180 to 180 degrees
        
        delta_values_rad = delta_values_deg * np.pi / 180
        
        min_spread = float('inf')
        
        # Equation (A-4): sigma_AS = min_Delta sigma_AS(Delta)
        for delta in delta_values_rad:
            
            # Step 1: Calculate mu_theta(Delta) - Equation (A-6)
            # mu_theta(Delta) = Sum Sum theta_n,m(Delta) * P_n,m / Sum Sum P_n,m
            # where theta_n,m(Delta) = mod(theta_n,m + Delta + pi, 2*pi) - pi
            
            # Apply delta shift and wrap: theta_n,m(Delta) = mod(theta_n,m + Delta + pi, 2*pi) - pi
            theta_shifted = np.mod(angles_rad + delta + np.pi, 2*np.pi) - np.pi
            
            # Calculate power-weighted mean mu_theta(Delta)
            mu_theta_delta = np.sum(powers * theta_shifted)
            
            # Step 2: Calculate theta_n,m,mu(Delta) - Equation (A-5)  
            # theta_n,m,mu(Delta) = mod(theta_n,m(Delta) - mu_theta(Delta) + pi, 2*pi) - pi
            theta_centered = np.mod(theta_shifted - mu_theta_delta + np.pi, 2*np.pi) - np.pi
            
            # Step 3: Calculate sigma_AS(Delta) - Equation (A-4)
            # sigma_AS(Delta) = sqrt(Sum Sum [theta_n,m,mu(Delta)]^2 * P_n,m / Sum Sum P_n,m)
            spread_squared = np.sum(powers * theta_centered**2)
            spread_val = np.sqrt(spread_squared)
            
            # Find minimum across all delta values
            if spread_val < min_spread:
                min_spread = spread_val
        
        # Convert back to degrees - NO scaling applied (pure 3GPP formula)
        spread = min_spread * 180 / np.pi
        
        return spread
    
    def _compute_serving_assignments(self, association_method: str = 'rsrp') -> Dict[int, int]:
        """
        Compute serving cell assignments for all UEs based on the specified method.
        Uses the SAME logic as compute_sir_sinr_geometry() to ensure consistency.
        
        Args:
            association_method: Association method to use:
                - 'rsrp': RSRP-based (CIR power with antenna virtualization)
                - 'cir': CIR power-based (legacy)
                - 'distance': Minimal distance-based
        
        Returns:
            Dictionary mapping UE index to serving cell ID {ue_idx: cell_id}
        """
        if association_method == 'cir':
            # Legacy CIR-based association
            return self.get_serving_cell_assignments()
        
        # Build serving assignments using RSRP or distance (same logic as SIR/SINR code)
        serving_assignments = {}
        
        # Get all unique UE indices
        ue_indices = set(link.uid for link in self.active_link_params)
        
        for ue_idx in ue_indices:
            # Get all links for this UE
            ue_links = [link for link in self.active_link_params if link.uid == ue_idx]
            
            if not ue_links:
                continue
            
            # Find serving cell using the specified method
            best_cell_id = None
            best_metric = None
            
            for link in ue_links:
                cell_id = link.cid
                
                # Calculate the metric based on association method
                if association_method == 'rsrp':
                    # RSRP: Use CIR power with antenna virtualization (SAME as SIR/SINR code)
                    metric = 0.0
                    if cell_id in self.cir_per_cell:
                        cir_data = self.cir_per_cell[cell_id]
                        
                        # Get the UE row mapping
                        ue_row_idx = self._get_ue_row_mapping(cell_id, ue_idx)
                        
                        if ue_row_idx is None:
                            # Fallback: Use ordering based on active links
                            cell_ue_indices = []
                            for l in self.active_link_params:
                                if l.cid == cell_id:
                                    cell_ue_indices.append(l.uid)
                            if ue_idx in cell_ue_indices:
                                ue_row_idx = cell_ue_indices.index(ue_idx)
                        
                        if ue_row_idx is not None and ue_row_idx < cir_data.shape[0]:
                            # Get ntaps if available
                            ntaps = None
                            if cell_id in self.cir_ntaps_per_cell:
                                ntaps_data = self.cir_ntaps_per_cell[cell_id]
                                if ue_row_idx < len(ntaps_data):
                                    ntaps = ntaps_data[ue_row_idx]
                            
                            # Compute CIR power with antenna virtualization for RSRP
                            apply_virt = (self.calibration_phase in [1, 2])
                            cir_power = self.compute_cir_power_for_link(
                                cir_data[ue_row_idx], 
                                ntaps,
                                apply_virtualization=apply_virt,
                                calibration_phase=self.calibration_phase,
                                tilt_deg=12.0
                            )
                            metric = cir_power
                elif association_method == 'distance':
                    # Distance-based: use negative distance (so closer is higher/better)
                    if link.lsp_read_idx < len(self.link_params):
                        lp = self.link_params[link.lsp_read_idx]
                        metric = -lp.d3d  # Negative so closer is better
                
                # Update best cell if this one is better
                if best_metric is None or (metric is not None and metric > best_metric):
                    best_metric = metric
                    best_cell_id = cell_id
            
            if best_cell_id is not None:
                serving_assignments[ue_idx] = best_cell_id
        
        return serving_assignments
    
    def analyze_delay_and_angle_spreads_isac(self) -> Dict[str, np.ndarray]:
        """
        Analyze delay spread and angle spreads for ISAC channels.
        For ISAC target/background channels, we analyze ALL links (not just serving cell).
        
        Per 3GPP 38.901 Table 7.9.6.2-1:
        - Delay spread and angle spreads are calculated separately for target/background channels
        - For monostatic: spreads computed separately for each reference point
        
        Returns CDFs for DS, ASD, ZSD, ASA, ZSA.
        """
        logger.info(f"Analyzing delay and angle spreads for ISAC {self.isac_channel_type} channel")
        
        # If multi-seed data is available, return it directly
        if hasattr(self, '_is_multi_seed') and self._is_multi_seed and hasattr(self, '_multi_seed_spreads'):
            logger.info(f"Using multi-seed combined spread data")
            return self._multi_seed_spreads
        
        # Collect all relevant parameters for ALL ISAC links
        ds_values = []
        asd_values = []
        zsd_values = []
        asa_values = []
        zsa_values = []
        
        # Load cluster parameters from H5 file if not already loaded
        cluster_params_per_link = self._load_all_cluster_params()
        
        # For ISAC target channel, load LOS status and K-factor from ISAC target links
        isac_target_los = None
        isac_target_k = None  # K-factor array (will be in dB)
        if self.isac_channel_type == 'target':
            try:
                with h5py.File(self.h5_file_path, 'r') as f:
                    if 'topology/isacTargetLinks/incident_los' in f:
                        isac_target_los = f['topology/isacTargetLinks/incident_los'][:]
                        logger.info(f"Loaded {len(isac_target_los)} ISAC target LOS indicators from H5")
                        
                        # Try to load K-factor from ISAC target links (new format)
                        if 'topology/isacTargetLinks/incident_k' in f:
                            isac_target_k = f['topology/isacTargetLinks/incident_k'][:]
                            logger.info(f"Loaded {len(isac_target_k)} ISAC target K-factors from H5 (incident path)")
                        else:
                            # Fallback: Check for K-factor override in system config (old method or if incident_k not saved)
                            if 'systemLevelConfig' in f:
                                sys_config = f['systemLevelConfig'][()]
                                if hasattr(sys_config, 'dtype') and sys_config.dtype.names:
                                    if 'st_override_k_db' in sys_config.dtype.names:
                                        st_override_k_db = sys_config['st_override_k_db']
                                        if isinstance(st_override_k_db, np.ndarray):
                                            st_override_k_db = st_override_k_db[0] if len(st_override_k_db) > 0 else st_override_k_db
                                        if not np.isnan(st_override_k_db):
                                            # Apply same K-factor to all links
                                            isac_target_k = np.full(len(isac_target_los), st_override_k_db)
                                            logger.info(f"Using overridden K-factor: {st_override_k_db:.2f} dB for all LOS ISAC targets (from systemLevelConfig)")
            except Exception as e:
                logger.warning(f"Could not load ISAC target link data: {e}")
                logger.warning("Falling back to linkParams data")
        
        # For ISAC, analyze ALL active links (no serving cell filtering)
        for idx, active_link in enumerate(self.active_link_params):
            if active_link.lsp_read_idx >= len(self.link_params):
                continue
            
            # Get cluster parameters for this link
            if (active_link.lsp_read_idx < len(cluster_params_per_link) and 
                cluster_params_per_link[active_link.lsp_read_idx] is not None):
                
                cluster_param = cluster_params_per_link[active_link.lsp_read_idx]
                n_clusters = cluster_param['n_cluster']
                n_rays_per_cluster = cluster_param['n_ray_per_cluster']
                total_rays = n_clusters * n_rays_per_cluster
                
                # Get LOS status and K-factor for this link
                # For ISAC target channel, use ISAC-specific data; otherwise use linkParams
                if self.isac_channel_type == 'target' and isac_target_los is not None and idx < len(isac_target_los):
                    # Use ISAC target link data (BS→ST→BS path)
                    is_los = (isac_target_los[idx] == 1)
                    if is_los and isac_target_k is not None and idx < len(isac_target_k):
                        # Use K-factor from ISAC target links (saved in H5 file)
                        k_factor_db = isac_target_k[idx]
                        k_factor_linear = np.power(10.0, k_factor_db / 10.0)
                    else:
                        # NLOS or K-factor not available: use linkParams K-factor
                        lp = self.link_params[active_link.lsp_read_idx]
                        k_factor_linear = lp.k_factor
                else:
                    # Use traditional link params (BS→UE path) for background or when ISAC data unavailable
                    lp = self.link_params[active_link.lsp_read_idx]
                    is_los = (lp.los_ind == 1)
                    k_factor_linear = lp.k_factor  # K-factor in linear scale
                
                # Compute per-ray powers according to 3GPP TR 25.996 Annex A
                cluster_powers = np.array(cluster_param['powers'][:n_clusters])
                ray_powers = np.repeat(cluster_powers / n_rays_per_cluster, n_rays_per_cluster)
                
                # Compute DS from actual cluster delays (RMS delay spread)
                if cluster_param['delays'] is not None and len(cluster_param['delays']) >= n_clusters:
                    cluster_delays = np.array(cluster_param['delays'][:n_clusters])
                    total_power = np.sum(cluster_powers)
                    if total_power > 0:
                        mean_delay = np.sum(cluster_powers * cluster_delays) / total_power
                        mean_delay_sq = np.sum(cluster_powers * cluster_delays**2) / total_power
                        ds_rms = np.sqrt(mean_delay_sq - mean_delay**2)
                        ds_values.append(ds_rms)
                
                # Compute angle spreads using TR 25.996 Annex A for LOS, Equation A-1 for NLOS
                if is_los and n_clusters > 1:
                    # LOS case: Use TR 25.996 Annex A with explicit LOS component
                    # Per 3GPP TR 38.901 Eq. 7.5-7: P_LOS = K_R / (K_R + 1), P_NLOS = 1 / (K_R + 1)
                    los_power = k_factor_linear / (k_factor_linear + 1)
                    nlos_total_power = 1.0 / (k_factor_linear + 1)
                    
                    # All NLOS cluster powers (stored in cluster_powers) are normalized and need to be
                    # scaled by the total NLOS power allocation: nlos_cluster_power_actual = cluster_power_normalized * P_NLOS
                    # The cluster_powers array is normalized such that sum(cluster_powers) ≈ 1.0
                    # For LOS scenarios: cluster_powers[0] = LOS cluster power, cluster_powers[1:] = NLOS cluster powers
                    # We need to scale ALL NLOS clusters (excluding cluster 0 which is LOS) by nlos_total_power
                    nlos_cluster_powers_actual = cluster_powers[1:] * nlos_total_power
                    
                    # LOS angles are from first ray of first cluster (geometric angles)
                    # NLOS cluster angles are from cluster centers (phi_n_aod, not phi_n_m_aod)
                    if (cluster_param['phi_n_m_aod'] is not None and len(cluster_param['phi_n_m_aod']) >= n_rays_per_cluster and
                        cluster_param['phi_n_aod'] is not None and len(cluster_param['phi_n_aod']) >= n_clusters):
                        los_aod = cluster_param['phi_n_m_aod'][0]  # First ray of first cluster
                        nlos_aod = np.array(cluster_param['phi_n_aod'][1:n_clusters])  # Cluster centers (skip first)
                        if len(nlos_aod) > 0:
                            asd_spread = self.compute_cluster_angle_spread_with_los(
                                nlos_aod.tolist(), nlos_cluster_powers_actual.tolist(),
                                los_aod, los_power, is_zenith=False
                            )
                            asd_values.append(asd_spread)
                    
                    if (cluster_param['phi_n_m_aoa'] is not None and len(cluster_param['phi_n_m_aoa']) >= n_rays_per_cluster and
                        cluster_param['phi_n_aoa'] is not None and len(cluster_param['phi_n_aoa']) >= n_clusters):
                        los_aoa = cluster_param['phi_n_m_aoa'][0]  # First ray of first cluster
                        nlos_aoa = np.array(cluster_param['phi_n_aoa'][1:n_clusters])  # Cluster centers (skip first)
                        if len(nlos_aoa) > 0:
                            asa_spread = self.compute_cluster_angle_spread_with_los(
                                nlos_aoa.tolist(), nlos_cluster_powers_actual.tolist(),
                                los_aoa, los_power, is_zenith=False
                            )
                            asa_values.append(asa_spread)
                    
                    if (cluster_param['theta_n_m_zod'] is not None and len(cluster_param['theta_n_m_zod']) >= n_rays_per_cluster and
                        cluster_param['theta_n_zod'] is not None and len(cluster_param['theta_n_zod']) >= n_clusters):
                        los_zod = cluster_param['theta_n_m_zod'][0]  # First ray of first cluster
                        nlos_zod = np.array(cluster_param['theta_n_zod'][1:n_clusters])  # Cluster centers (skip first)
                        if len(nlos_zod) > 0:
                            zsd_spread = self.compute_cluster_angle_spread_with_los(
                                nlos_zod.tolist(), nlos_cluster_powers_actual.tolist(),
                                los_zod, los_power, is_zenith=True
                            )
                            zsd_values.append(zsd_spread)
                    
                    if (cluster_param['theta_n_m_zoa'] is not None and len(cluster_param['theta_n_m_zoa']) >= n_rays_per_cluster and
                        cluster_param['theta_n_zoa'] is not None and len(cluster_param['theta_n_zoa']) >= n_clusters):
                        los_zoa = cluster_param['theta_n_m_zoa'][0]  # First ray of first cluster
                        nlos_zoa = np.array(cluster_param['theta_n_zoa'][1:n_clusters])  # Cluster centers (skip first)
                        if len(nlos_zoa) > 0:
                            zsa_spread = self.compute_cluster_angle_spread_with_los(
                                nlos_zoa.tolist(), nlos_cluster_powers_actual.tolist(),
                                los_zoa, los_power, is_zenith=True
                            )
                            zsa_values.append(zsa_spread)
                
                else:
                    # NLOS case or single cluster: Use per-ray calculation (Equation A-1)
                    if cluster_param['phi_n_m_aod'] is not None and len(cluster_param['phi_n_m_aod']) >= total_rays:
                        asd_circular = self.compute_circular_angle_spread(
                            cluster_param['phi_n_m_aod'][:total_rays],
                            ray_powers.tolist(),
                            is_zenith=False
                        )
                        asd_values.append(asd_circular)
                    
                    if cluster_param['phi_n_m_aoa'] is not None and len(cluster_param['phi_n_m_aoa']) >= total_rays:
                        asa_circular = self.compute_circular_angle_spread(
                            cluster_param['phi_n_m_aoa'][:total_rays],
                            ray_powers.tolist(),
                            is_zenith=False
                        )
                        asa_values.append(asa_circular)
                    
                    if cluster_param['theta_n_m_zod'] is not None and len(cluster_param['theta_n_m_zod']) >= total_rays:
                        zod_angles = np.array(cluster_param['theta_n_m_zod'][:total_rays])
                        zsd_spread = self.compute_circular_angle_spread(
                            zod_angles.tolist(),
                            ray_powers.tolist(),
                            is_zenith=True
                        )
                        zsd_values.append(zsd_spread)
                    
                    if cluster_param['theta_n_m_zoa'] is not None and len(cluster_param['theta_n_m_zoa']) >= total_rays:
                        zoa_angles = np.array(cluster_param['theta_n_m_zoa'][:total_rays])
                        zsa_spread = self.compute_circular_angle_spread(
                            zoa_angles.tolist(),
                            ray_powers.tolist(),
                            is_zenith=True
                        )
                        zsa_values.append(zsa_spread)
        
        logger.info(f"Collected {len(ds_values)} delay spread samples from ISAC links")
        logger.info(f"Collected {len(asd_values)} ASD samples, {len(asa_values)} ASA samples")
        logger.info(f"Collected {len(zsd_values)} ZSD samples, {len(zsa_values)} ZSA samples")
        
        # Create results dictionary
        results = {
            'DS': np.array(ds_values),
            'ASD': np.array(asd_values), 
            'ZSD': np.array(zsd_values),
            'ASA': np.array(asa_values),
            'ZSA': np.array(zsa_values)
        }
        
        # Save to JSON file
        self._save_spreads_to_json(results, f'isac_{self.isac_channel_type}')
        
        return results
    
    def analyze_delay_and_angle_spreads(self, association_method: str = 'rsrp', 
                                        serving_assignments: Dict[int, int] = None) -> Dict[str, np.ndarray]:
        """
        Analyze delay spread and angle spreads for the serving cell.
        Returns CDFs for DS, ASD, ZSD, ASA, ZSA.
        
        Args:
            association_method: Association method to use:
                - 'rsrp': RSRP-based (pathloss - shadow fading) [default]
                - 'cir': CIR power-based (legacy)
                - 'distance': Minimal distance-based
            serving_assignments: Pre-computed serving cell assignments {ue_idx: cell_id}.
                If None, will compute using association_method.
        """
        logger.info("Analyzing delay and angle spreads")
        
        # If multi-seed data is available, return it directly
        if hasattr(self, '_is_multi_seed') and self._is_multi_seed and hasattr(self, '_multi_seed_spreads'):
            logger.info(f"Using multi-seed combined spread data")
            return self._multi_seed_spreads
        
        # Use pre-computed serving assignments if provided, otherwise compute them
        if serving_assignments is None:
            serving_assignments = self._compute_serving_assignments(association_method)
        
        # Collect all relevant parameters for serving cell links
        ds_values = []
        asd_values = []
        zsd_values = []
        asa_values = []
        zsa_values = []
        
        # Load cluster parameters from H5 file if not already loaded
        cluster_params_per_link = self._load_all_cluster_params()
        
        # For each UE, analyze their serving link
        for ue_idx, serving_cell_id in serving_assignments.items():
            # Find the serving link for this UE
            serving_link = None
            for active_link in self.active_link_params:
                if active_link.uid == ue_idx and active_link.cid == serving_cell_id and active_link.lsp_read_idx < len(self.link_params):
                    serving_link = active_link
                    link = self.link_params[active_link.lsp_read_idx]
                    break
            
            if serving_link is None:
                continue
            
            # Get cluster parameters for this specific link to compute DS and angle spreads from actual per-ray data
            if (serving_link.lsp_read_idx < len(cluster_params_per_link) and 
                cluster_params_per_link[serving_link.lsp_read_idx] is not None):
                
                cluster_param = cluster_params_per_link[serving_link.lsp_read_idx]
                n_clusters = cluster_param['n_cluster']
                n_rays_per_cluster = cluster_param['n_ray_per_cluster']
                total_rays = n_clusters * n_rays_per_cluster
                
                # Get LOS status and K-factor for this link (already retrieved in line 2224)
                is_los = (link.los_ind == 1)
                k_factor_linear = link.k_factor  # K-factor in linear scale
                
                # Compute per-ray powers according to 3GPP TR 25.996 Annex A
                # Each ray within a cluster gets equal power (1/M of the cluster power)
                cluster_powers = np.array(cluster_param['powers'][:n_clusters])
                ray_powers = np.repeat(cluster_powers / n_rays_per_cluster, n_rays_per_cluster)
                
                # Compute DS from actual cluster delays (RMS delay spread)
                if cluster_param['delays'] is not None and len(cluster_param['delays']) >= n_clusters:
                    cluster_delays = np.array(cluster_param['delays'][:n_clusters])
                    # RMS delay spread formula: sqrt(E[tau^2] - E[tau]^2)
                    total_power = np.sum(cluster_powers)
                    if total_power > 0:
                        mean_delay = np.sum(cluster_powers * cluster_delays) / total_power
                        mean_delay_sq = np.sum(cluster_powers * cluster_delays**2) / total_power
                        ds_rms = np.sqrt(mean_delay_sq - mean_delay**2)
                        ds_values.append(ds_rms)
                
                # Compute angle spreads using TR 25.996 Annex A for LOS, Equation A-1 for NLOS
                if is_los and n_clusters > 1:
                    # LOS case: Use TR 25.996 Annex A with explicit LOS component
                    # Per 3GPP TR 38.901 Eq. 7.5-7: P_LOS = K_R / (K_R + 1), P_NLOS = 1 / (K_R + 1)
                    los_power = k_factor_linear / (k_factor_linear + 1)
                    nlos_total_power = 1.0 / (k_factor_linear + 1)
                    
                    # All NLOS cluster powers (stored in cluster_powers) are normalized and need to be
                    # scaled by the total NLOS power allocation: nlos_cluster_power_actual = cluster_power_normalized * P_NLOS
                    # The cluster_powers array is normalized such that sum(cluster_powers) ≈ 1.0
                    # For LOS scenarios: cluster_powers[0] = LOS cluster power, cluster_powers[1:] = NLOS cluster powers
                    # We need to scale ALL NLOS clusters (excluding cluster 0 which is LOS) by nlos_total_power
                    nlos_cluster_powers_actual = cluster_powers[1:] * nlos_total_power
                    
                    # LOS angles are from first ray of first cluster (geometric angles)
                    # NLOS cluster angles are from cluster centers (phi_n_aod, not phi_n_m_aod)
                    if (cluster_param['phi_n_m_aod'] is not None and len(cluster_param['phi_n_m_aod']) >= n_rays_per_cluster and
                        cluster_param['phi_n_aod'] is not None and len(cluster_param['phi_n_aod']) >= n_clusters):
                        los_aod = cluster_param['phi_n_m_aod'][0]  # First ray of first cluster
                        nlos_aod = np.array(cluster_param['phi_n_aod'][1:n_clusters])  # Cluster centers (skip first)
                        if len(nlos_aod) > 0:
                            asd_spread = self.compute_cluster_angle_spread_with_los(
                                nlos_aod.tolist(), nlos_cluster_powers_actual.tolist(),
                                los_aod, los_power, is_zenith=False
                            )
                            asd_values.append(asd_spread)
                    
                    if (cluster_param['phi_n_m_aoa'] is not None and len(cluster_param['phi_n_m_aoa']) >= n_rays_per_cluster and
                        cluster_param['phi_n_aoa'] is not None and len(cluster_param['phi_n_aoa']) >= n_clusters):
                        los_aoa = cluster_param['phi_n_m_aoa'][0]  # First ray of first cluster
                        nlos_aoa = np.array(cluster_param['phi_n_aoa'][1:n_clusters])  # Cluster centers (skip first)
                        if len(nlos_aoa) > 0:
                            asa_spread = self.compute_cluster_angle_spread_with_los(
                                nlos_aoa.tolist(), nlos_cluster_powers_actual.tolist(),
                                los_aoa, los_power, is_zenith=False
                            )
                            asa_values.append(asa_spread)
                    
                    if (cluster_param['theta_n_m_zod'] is not None and len(cluster_param['theta_n_m_zod']) >= n_rays_per_cluster and
                        cluster_param['theta_n_zod'] is not None and len(cluster_param['theta_n_zod']) >= n_clusters):
                        los_zod = cluster_param['theta_n_m_zod'][0]  # First ray of first cluster
                        nlos_zod = np.array(cluster_param['theta_n_zod'][1:n_clusters])  # Cluster centers (skip first)
                        if len(nlos_zod) > 0:
                            zsd_spread = self.compute_cluster_angle_spread_with_los(
                                nlos_zod.tolist(), nlos_cluster_powers_actual.tolist(),
                                los_zod, los_power, is_zenith=True
                            )
                            zsd_values.append(zsd_spread)
                    
                    if (cluster_param['theta_n_m_zoa'] is not None and len(cluster_param['theta_n_m_zoa']) >= n_rays_per_cluster and
                        cluster_param['theta_n_zoa'] is not None and len(cluster_param['theta_n_zoa']) >= n_clusters):
                        los_zoa = cluster_param['theta_n_m_zoa'][0]  # First ray of first cluster
                        nlos_zoa = np.array(cluster_param['theta_n_zoa'][1:n_clusters])  # Cluster centers (skip first)
                        if len(nlos_zoa) > 0:
                            zsa_spread = self.compute_cluster_angle_spread_with_los(
                                nlos_zoa.tolist(), nlos_cluster_powers_actual.tolist(),
                                los_zoa, los_power, is_zenith=True
                            )
                            zsa_values.append(zsa_spread)
                
                else:
                    # NLOS case or single cluster: Use per-ray calculation (Equation A-1)
                    if cluster_param['phi_n_m_aod'] is not None and len(cluster_param['phi_n_m_aod']) >= total_rays:
                        asd_circular = self.compute_circular_angle_spread(
                            cluster_param['phi_n_m_aod'][:total_rays],
                            ray_powers.tolist(),
                            is_zenith=False  # ASD is azimuth (circular)
                        )
                        asd_values.append(asd_circular)
                    
                    if cluster_param['phi_n_m_aoa'] is not None and len(cluster_param['phi_n_m_aoa']) >= total_rays:
                        asa_circular = self.compute_circular_angle_spread(
                            cluster_param['phi_n_m_aoa'][:total_rays],
                            ray_powers.tolist(),
                            is_zenith=False  # ASA is azimuth (circular)
                        )
                        asa_values.append(asa_circular)
                    
                    if cluster_param['theta_n_m_zod'] is not None and len(cluster_param['theta_n_m_zod']) >= total_rays:
                        # For ZSD: Use zenith-specific calculation (linear, not circular)
                        zod_angles = np.array(cluster_param['theta_n_m_zod'][:total_rays])
                        zsd_spread = self.compute_circular_angle_spread(
                            zod_angles.tolist(),
                            ray_powers.tolist(),
                            is_zenith=True  # ZSD is zenith (linear)
                        )
                        zsd_values.append(zsd_spread)
                    
                    if cluster_param['theta_n_m_zoa'] is not None and len(cluster_param['theta_n_m_zoa']) >= total_rays:
                        # For ZSA: Use zenith-specific calculation (linear, not circular)
                        zoa_angles = np.array(cluster_param['theta_n_m_zoa'][:total_rays])                        
                        zsa_spread = self.compute_circular_angle_spread(
                            zoa_angles.tolist(),
                            ray_powers.tolist(),
                            is_zenith=True  # ZSA is zenith (linear)
                        )                        
                        zsa_values.append(zsa_spread)

        # Create results dictionary
        results = {
            'DS': np.array(ds_values),
            'ASD': np.array(asd_values), 
            'ZSD': np.array(zsd_values),
            'ASA': np.array(asa_values),
            'ZSA': np.array(zsa_values)
        }
        
        # Save to JSON file
        self._save_spreads_to_json(results, association_method)
        
        return results
    
    def _save_spreads_to_json(self, results: Dict[str, np.ndarray], association_method: str = 'rsrp'):
        """Save angle spread results to JSON file"""
        import json
        from pathlib import Path
        
        # Convert numpy arrays to lists for JSON serialization
        json_results = {}
        for param, values in results.items():
            json_results[param] = values.tolist()
        
        # Add metadata
        json_results['metadata'] = {
            'total_samples': len(results['DS']),
            'cell_association_method': association_method,
            'h5_file': str(self.h5_file_path),
            'analysis_timestamp': str(np.datetime64('now')),
            'statistics': {}
        }
        
        # Add statistics for each parameter
        for param, values in results.items():
            if len(values) > 0:
                json_results['metadata']['statistics'][param] = {
                    'count': len(values),
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'percentiles': {
                        '5th': float(np.percentile(values, 5)),
                        '10th': float(np.percentile(values, 10)),
                        '25th': float(np.percentile(values, 25)),
                        '50th': float(np.percentile(values, 50)),
                        '75th': float(np.percentile(values, 75)),
                        '90th': float(np.percentile(values, 90)),
                        '95th': float(np.percentile(values, 95)),
                        '99th': float(np.percentile(values, 99))
                    }
                }
        
        # Generate filename
        association_suffix = f"_{association_method}"
        output_file = f"angle_spreads{association_suffix}.json"
        
        # Save to file
        with open(output_file, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        logger.info(f"Angle spread results saved to: {output_file}")
        print(f"Results saved to: {output_file}")
    
    def compute_prb_singular_values(self, fft_size: int = 2048) -> Dict[str, np.ndarray]:
        """
        Compute PRB singular values from CIR data using FFT to CFR (matching MATLAB approach).
        
        This function:
        1. Takes CIR (Channel Impulse Response) from H5 file
        2. FFTs to CFR (Channel Frequency Response) 
        3. Computes SVD for each frequency bin/PRB
        4. Normalizes SVs and collects statistics
        
        Args:
            fft_size: FFT size for converting CIR to CFR (default: 2048)
            
        Returns:
            Dictionary with largest, smallest singular values and their ratios
        """
        logger.info("Computing PRB singular values from CIR data (MATLAB-style)")

        # Phase 1 calibration is large-scale only; PRB singular value analysis is not applicable.
        if self.calibration_phase == 1:
            logger.info("Skipping PRB singular value computation for Phase 1 calibration (large-scale only)")
            return {}
        
        # Skip for ISAC mode - ISAC uses BS→ST→BS channels, not BS→UE
        if self.isac_config and self.isac_config.is_enabled:
            logger.info("Skipping PRB singular value computation for ISAC mode (no BS→UE links)")
            return {}
        
        if not self.cir_per_cell:
            logger.error("No CIR data available. Cannot compute singular values.")
            return {}
        
        # Initialize results - collect ALL normalized SVs (not just largest/smallest)
        all_normalized_svs: List[float] = []
        largest_sv: List[float] = []
        smallest_sv: List[float] = []
        sv_ratios: List[float] = []

        total_cells = len(self.cir_per_cell)
        skipped_cells = 0
        used_cells = 0
        
        # Process CIR data for each cell
        for cell_id, cir_data in self.cir_per_cell.items():
            logger.info(f"Processing CIR for cell {cell_id}, shape: {cir_data.shape}")
            
            # CIR data shape (Phase-2 typical): [nUE, nSnapshots, nCoeffs]
            # CIR data shape (legacy):         [nUE, nCoeffs]
            #
            # We compute PRB SVs at t=0, so if snapshot dimension exists we take snapshot 0.
            # Then reshape per UE as [nTaps, nRx, nTx].
            
            if cir_data.ndim not in (2, 3, 4, 5):
                logger.warning(f"Unexpected CIR ndim={cir_data.ndim} for cell {cell_id}, skipping")
                skipped_cells += 1
                continue

            n_ues = cir_data.shape[0]
            # Determine (nRx, nTx, nTaps) depending on dataset shape
            if cir_data.ndim == 5:
                # [nUE, nSnapshots, nRx, nTx, nTaps]
                n_snapshots = cir_data.shape[1]
                n_rx = cir_data.shape[2]
                n_tx = cir_data.shape[3]
                n_taps = cir_data.shape[4]
                total_elements = n_rx * n_tx * n_taps
            elif cir_data.ndim == 4:
                # [nUE, nRx, nTx, nTaps] (treat as nSnapshots=1)
                n_snapshots = 1
                n_rx = cir_data.shape[1]
                n_tx = cir_data.shape[2]
                n_taps = cir_data.shape[3]
                total_elements = n_rx * n_tx * n_taps
            else:
                # Legacy:
                # - [nUE, nSnapshots, nCoeffs]
                # - [nUE, nCoeffs]
                n_snapshots = cir_data.shape[1] if cir_data.ndim == 3 else 1
                total_elements = cir_data.shape[-1]
            
            # Get antenna counts from antenna panel configuration
            # Find the cell parameter for this cell to get BS antenna panel idx
            cell_param = next((c for c in self.cell_params if c.cid == cell_id), None)
            if cell_param is None:
                logger.warning(f"Cell {cell_id} not found in cell_params, skipping")
                skipped_cells += 1
                continue
            
            # Get BS antenna panel configuration
            bs_ant_panel = self.antenna_panels.get(cell_param.ant_panel_idx)
            if bs_ant_panel is None:
                logger.warning(f"BS antenna panel {cell_param.ant_panel_idx} not found, skipping cell {cell_id}")
                skipped_cells += 1
                continue
            
            n_tx = bs_ant_panel.n_antennas  # BS antennas
            
            if cir_data.ndim in (2, 3):
                # UE antenna count is not always reliably represented by the "default" UE panel index in the H5 metadata
                # (e.g., UMa datasets can store CIR as 24 taps × 2 Rx × 64 Tx = 3072 coefficients).
                # We prefer to infer nRx from the coefficient length and BS nTx when possible.
                n_taps = 24  # Default per 3GPP
                if total_elements % n_taps != 0:
                    logger.error(f"CIR coeff length {total_elements} not divisible by n_taps={n_taps}, skipping cell {cell_id}")
                    skipped_cells += 1
                    continue

                n_ant_pairs = total_elements // n_taps  # nRx * nTx

                # First try: infer nRx using BS nTx
                n_rx = None
                if n_tx > 0 and (n_ant_pairs % n_tx == 0):
                    n_rx = n_ant_pairs // n_tx
                else:
                    # Fallback: try UE panel metadata to infer nTx
                    ue_ant_panel_idx = 1 if 1 in self.antenna_panels else 0
                    ue_ant_panel = self.antenna_panels.get(ue_ant_panel_idx)
                    if ue_ant_panel is not None and ue_ant_panel.n_antennas > 0 and (n_ant_pairs % ue_ant_panel.n_antennas == 0):
                        n_rx = ue_ant_panel.n_antennas
                        n_tx = n_ant_pairs // n_rx
                    else:
                        # Final fallback: choose a factor pair close to BS nTx
                        best_pair = None
                        for cand_tx in range(1, n_ant_pairs + 1):
                            if n_ant_pairs % cand_tx != 0:
                                continue
                            cand_rx = n_ant_pairs // cand_tx
                            score = abs(cand_tx - n_tx)
                            if best_pair is None or score < best_pair[0]:
                                best_pair = (score, cand_rx, cand_tx)
                        if best_pair is None:
                            logger.error(f"Cannot factor n_ant_pairs={n_ant_pairs} for cell {cell_id}, skipping")
                            skipped_cells += 1
                            continue
                        _, n_rx, n_tx = best_pair
                        logger.warning(f"Inferred nRx={n_rx}, nTx={n_tx} from coeff length for cell {cell_id}")

                expected_elements = n_taps * n_rx * n_tx
                if expected_elements != total_elements:
                    logger.error(f"CIR dimension mismatch for cell {cell_id}: expected {expected_elements}, actual {total_elements}")
                    skipped_cells += 1
                    continue

            used_cells += 1
            
            logger.info(f"  Cell {cell_id}: nUE={n_ues}, nTaps={n_taps}, nRx={n_rx}, nTx={n_tx} (BS panel {cell_param.ant_panel_idx})")
            logger.info(f"  Total elements: {total_elements} = {n_taps}×{n_rx}×{n_tx}")
            
            # Process each UE
            for ue_idx in range(min(n_ues, 100)):  # Limit to first 100 UEs for performance
                # Extract CIR for this UE at t=0 and reshape to [nTaps, nRx, nTx]
                if cir_data.ndim == 5:
                    # [nUE, nSnapshots, nRx, nTx, nTaps] -> [nTaps, nRx, nTx]
                    cir_ue = np.transpose(cir_data[ue_idx, 0, :, :, :], (2, 0, 1))
                elif cir_data.ndim == 4:
                    # [nUE, nRx, nTx, nTaps] -> [nTaps, nRx, nTx]
                    cir_ue = np.transpose(cir_data[ue_idx, :, :, :], (2, 0, 1))
                else:
                    # Legacy flattened coefficients -> reshape [nTaps, nRx, nTx]
                    if cir_data.ndim == 3:
                        cir_vec = cir_data[ue_idx, 0, :]
                    else:
                        cir_vec = cir_data[ue_idx, :]
                    cir_ue = cir_vec.reshape(n_taps, n_rx, n_tx)
                
                # === STEP 1: FFT to frequency domain (matching MATLAB) ===
                # MATLAB: tmp_fd = (1./sqrt(sizeFFT)) * fftshift(fft(H, sizeFFT, 3), 3)
                # We FFT along the tap dimension (axis 0)
                H_fd = np.fft.fft(cir_ue, n=fft_size, axis=0) / np.sqrt(fft_size)
                H_fd = np.fft.fftshift(H_fd, axes=0)
                
                # === STEP 2: Compute SVD for each frequency bin ===
                # MATLAB: tmp_svd = squeeze(pagesvd(H_cir_fd(:,:,:,itrial)))
                # Process each frequency bin
                for freq_idx in range(fft_size):
                    # Extract channel matrix for this frequency bin: [nRx x nTx]
                    H_freq = H_fd[freq_idx, :, :]
                    
                    # Compute SVD
                    try:
                        U, sv_values, Vh = np.linalg.svd(H_freq, full_matrices=False)
                        
                        if len(sv_values) > 0 and sv_values[0] > 1e-12:
                            # === STEP 3: Normalize by largest SV ===
                            # MATLAB: tmp_svd_n = tmp_svd ./ tmp_svd(1,:)
                            sv_normalized = sv_values / sv_values[0]
                            
                            # Collect largest and smallest (absolute values)
                            largest_sv.append(sv_values[0])
                            if len(sv_values) > 1:
                                smallest_sv.append(sv_values[-1])
                                sv_ratios.append(sv_values[0] / sv_values[-1] if sv_values[-1] > 1e-12 else sv_values[0] / 1e-12)
                                
                                # === STEP 4: Collect all normalized SVs except the first (which is 1.0) ===
                                # MATLAB: svd_vals = [svd_vals tmp_svd_n(2:end,:)]
                                all_normalized_svs.extend(sv_normalized[1:])
                            
                    except np.linalg.LinAlgError:
                        # SVD failed for this frequency bin, skip it
                        continue
        
        logger.info(
            f"PRB SV summary: total_cells={total_cells}, used_cells={used_cells}, skipped_cells={skipped_cells}"
        )

        if len(largest_sv) == 0:
            logger.warning(
                "No singular values collected (all CIRs were skipped or numerically invalid). "
                "Skipping PRB singular value plots."
            )
            return {}

        logger.info(f"Collected {len(all_normalized_svs)} normalized singular values")
        logger.info(f"Largest SV range: [{np.min(largest_sv):.2e}, {np.max(largest_sv):.2e}]")
        
        return {
            'largest': np.array(largest_sv),
            'smallest': np.array(smallest_sv),
            'ratios': np.array(sv_ratios),
            'all_normalized': np.array(all_normalized_svs)  # NEW: All normalized SVs like MATLAB
        }

    def plot_cdf(self, data: np.ndarray, title: str, xlabel: str, ylabel: str = 'CDF (%)', 
                  log_scale: bool = False, filename: Optional[str] = None,
                  reference_data: Tuple[np.ndarray, np.ndarray] = None,
                  reference_envelope: Tuple[np.ndarray, np.ndarray, np.ndarray] = None,
                  metric_name: str = None) -> None:
        """
        Plot cumulative distribution function with optional 3GPP reference overlay.
        
        Args:
            data: Input data array (simulation results)
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
            log_scale: Whether to use log scale for x-axis
            filename: Optional filename to save the plot
            reference_data: Tuple of (x_values, cdf_percentiles) from 3GPP reference (average)
            reference_envelope: Tuple of (x_min, x_max, cdf_percentiles) showing min/max range
            metric_name: Metric name for statistics display
        """
        if len(data) == 0:
            logger.warning(f"No data available for {title}")
            return
        
        plt.figure(figsize=(12, 8))
        
        # Remove any invalid values from simulation data
        valid_data = data[np.isfinite(data)]
        if len(valid_data) == 0:
            logger.warning(f"No valid data for {title}")
            return
        
        # Sort simulation data and compute CDF
        sorted_data = np.sort(valid_data)
        cdf_values = (np.arange(1, len(sorted_data) + 1) / len(sorted_data)) * 100
        
        # Plot reference envelope first (so it's in the background)
        if reference_envelope is not None:
            ref_x_min, ref_x_max, ref_cdf = reference_envelope
            if ref_x_min is not None and ref_x_max is not None and ref_cdf is not None:
                if len(ref_x_min) > 0 and len(ref_x_max) > 0:
                    # Fill between min and max curves with light grey
                    plt.fill_betweenx(ref_cdf, ref_x_min, ref_x_max, 
                                     color='lightgrey', alpha=0.5, 
                                     label='3GPP Range (Min-Max)', zorder=1)
        
        # Plot simulation data
        plt.plot(sorted_data, cdf_values, linewidth=2.5, label='Simulation', color='blue', zorder=3)
        
        # Plot reference data if available (average curve)
        ks_statistic = None
        ks_pvalue = None
        if reference_data is not None:
            ref_x, ref_cdf = reference_data
            if ref_x is not None and ref_cdf is not None and len(ref_x) > 0:
                plt.plot(ref_x, ref_cdf, linewidth=2.5, label='3GPP Reference (Avg)', 
                        color='red', linestyle='--', alpha=0.8, zorder=2)
                
                # Compute KS statistic and p-value using two-sample KS test
                if len(ref_x) > 1 and len(sorted_data) > 1:
                    try:
                        # Use scipy's two-sample KS test to compare distributions
                        # This returns both the KS statistic and p-value
                        ks_statistic, ks_pvalue = stats.ks_2samp(sorted_data, ref_x)
                        
                        logger.info(f"{metric_name} KS statistic: {ks_statistic:.4f}, p-value: {ks_pvalue:.4f}")
                    except Exception as e:
                        logger.warning(f"Could not compute KS statistic and p-value: {e}")
        
        if log_scale:
            plt.xscale('log')
        
        plt.xlabel(xlabel, fontsize=12)
        plt.ylabel(ylabel, fontsize=12)
        plt.title(title, fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=11, loc='best')
        
        # Add statistics text box
        stats_text = f'Simulation Statistics:\n'
        stats_text += f'  Mean: {np.mean(valid_data):.2f}\n'
        stats_text += f'  Std: {np.std(valid_data):.2f}\n'
        stats_text += f'  Median: {np.median(valid_data):.2f}\n'
        stats_text += f'  Samples: {len(valid_data)}'
        
        if reference_data is not None and reference_data[0] is not None:
            ref_x, _ = reference_data
            stats_text += f'\n\n3GPP Reference:\n'
            stats_text += f'  Mean: {np.mean(ref_x):.2f}\n'
            stats_text += f'  Std: {np.std(ref_x):.2f}\n'
            stats_text += f'  Median: {np.median(ref_x):.2f}'
            
            if ks_statistic is not None:
                stats_text += f'\n\nKS Statistic: {ks_statistic:.4f}'
            if ks_pvalue is not None:
                stats_text += f'\np-value: {ks_pvalue:.4f}'
        
        plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
                 verticalalignment='top', fontsize=9,
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        
        if filename:
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            logger.info(f"Saved plot: {filename}")
        
        plt.close()
    
    def calculate_thermal_noise_power(self, bandwidth_hz: float, noise_figure_db: float = 9.0, temperature_k: float = 290.0) -> float:
        """
        Calculate thermal noise power in dBm using standard -174 dBm/Hz noise density.
        
        Args:
            bandwidth_hz: Bandwidth in Hz
            noise_figure_db: Noise figure in dB (default 9 dB from 3GPP table)
            temperature_k: Temperature in Kelvin (unused, kept for compatibility)
        
        Returns:
            Thermal noise power in dBm
            Formula: N = -174 + 10*log10(BW_Hz) + NF_dB
        """
        # Thermal noise using standard -174 dBm/Hz noise density
        # N = N0 + 10*log10(BW) + NF, where N0 = -174 dBm/Hz
        noise_density_dbm_hz = -174.0
        bandwidth_db_hz = 10 * np.log10(bandwidth_hz)
        total_noise_dbm = noise_density_dbm_hz + bandwidth_db_hz + noise_figure_db
        
        return total_noise_dbm

    def get_system_parameters(self, scenario_or_freq, carrier_freq_hz: float = None):
        """
        Get system parameters based on scenario and carrier frequency according to 3GPP Table 7.8-1.
        
        Args:
            scenario_or_freq: Deployment scenario (UMa, UMi, etc.) or carrier frequency in Hz for backward compatibility
            carrier_freq_hz: Carrier frequency in Hz (optional if scenario_or_freq is frequency)
        
        Returns:
            dict: System parameters including tx_power_dbm and bandwidth_hz
        """
        # Handle backward compatibility - if first arg is a number, treat as frequency
        if isinstance(scenario_or_freq, (int, float)):
            carrier_freq_hz = scenario_or_freq
            scenario = None
        else:
            scenario = scenario_or_freq
            
        if carrier_freq_hz is None:
            raise ValueError("Carrier frequency must be provided")
            
        carrier_freq_ghz = carrier_freq_hz / 1e9
        scenario_str = scenario.name if hasattr(scenario, 'name') else str(scenario) if scenario else 'UMa'
        
        # Default parameters
        params = {
            'tx_power_dbm': 44,  # Default
            'bandwidth_hz': 20e6,  # Default 20 MHz
            'noise_figure_db': 9.0
        }
        
        # Set parameters based on frequency and scenario according to Table 7.8-1
        if abs(carrier_freq_ghz - 6.0) < 0.1:  # 6 GHz
            params['bandwidth_hz'] = 20e6  # 20 MHz
            if scenario_str and scenario_str.upper() in ['UMI', 'UMI-STREET CANYON', 'UMI_STREET_CANYON']:
                params['tx_power_dbm'] = 44  # UMi-Street Canyon at 6GHz
            elif scenario_str and scenario_str.upper() == 'UMA':
                params['tx_power_dbm'] = 49  # UMa at 6GHz
            else:  # Indoor or default
                params['tx_power_dbm'] = 44 if not scenario_str else 24
        elif abs(carrier_freq_ghz - 30.0) < 0.1:  # 30 GHz
            params['bandwidth_hz'] = 100e6  # 100 MHz
            if scenario_str and scenario_str.upper() in ['UMI', 'UMI-STREET CANYON', 'UMI_STREET_CANYON', 'UMA']:
                params['tx_power_dbm'] = 35  # UMa and UMi-Street Canyon at 30GHz
            else:  # Indoor
                params['tx_power_dbm'] = 24
        elif abs(carrier_freq_ghz - 70.0) < 0.1:  # 70 GHz
            params['bandwidth_hz'] = 100e6  # 100 MHz
            if scenario_str and scenario_str.upper() in ['UMI', 'UMI-STREET CANYON', 'UMI_STREET_CANYON', 'UMA']:
                params['tx_power_dbm'] = 35  # UMa and UMi-Street Canyon at 70GHz
            else:  # Indoor
                params['tx_power_dbm'] = 24
        
        return params

    def compute_coupling_loss_all_links(self, use_virtualization: bool = False) -> np.ndarray:
        """
        Compute coupling loss for all links.
        Coupling loss = Path loss - Shadow fading - BS Antenna gain
        
        Args:
            use_virtualization: If True, use antenna virtualization for gain calculation
        """
        logger.info(f"Computing coupling loss for all links (virtualization={'ON' if use_virtualization else 'OFF'})")
        
        coupling_losses = []
        
        for active_link in self.active_link_params:
            if active_link.lsp_read_idx < len(self.link_params):
                link = self.link_params[active_link.lsp_read_idx]
                
                # Get antenna gain for this cell (with optional virtualization)
                antenna_gain_db = self.get_antenna_gain_for_cell(
                    active_link.cid,
                    ue_id=active_link.uid,
                    use_virtualization=use_virtualization
                )
                
                # Coupling loss = Path loss - Shadow fading - BS Antenna gain (use negative for loss)
                coupling_loss = -(link.pathloss - link.sf - antenna_gain_db)
                coupling_losses.append(coupling_loss)
        
        return np.array(coupling_losses)

    def get_serving_cell_assignments(self) -> Dict[int, Dict[str, any]]:
        """
        Determine serving cells for each UE based on CIR power from H5 data.
        
        Returns:
            Dictionary mapping UE index to serving cell info:
            {ue_idx: {'cell_id': int, 'cir_power': float, 'site_id': int, 'sector_id': int}}
        """
        logger.info("Determining serving cells based on CIR power from H5 data")
        
        serving_assignments = {}
        
        if not self.cir_per_cell:
            logger.warning("No CIR data available in H5 file")
            return serving_assignments
        
        # Get system topology info
        n_sites = self.system_level_config.n_site if self.system_level_config else 1
        n_sectors_per_site = self.system_level_config.n_sector_per_site if self.system_level_config else 3
        
        # Helper function to get site and sector from cell ID
        def get_site_sector(cell_id):
            site_id = cell_id // n_sectors_per_site
            sector_id = cell_id % n_sectors_per_site
            return site_id, sector_id
        
        # Get all UE indices from active link parameters
        all_ue_indices = set()
        for active_link in self.active_link_params:
            all_ue_indices.add(active_link.uid)
        
        # For each UE, find the cell with maximum CIR power
        for ue_idx in sorted(all_ue_indices):
            max_power = -np.inf
            best_cell = None
            power_per_cell = {}
            
            for cell_id, cir_data in self.cir_per_cell.items():
                # Find this UE's CIR data for this cell
                ue_cir_power = 0.0
                
                # Find the active link for this UE-cell pair
                active_link_found = False
                ue_link_index = None
                
                for active_link in self.active_link_params:
                    if active_link.uid == ue_idx and active_link.cid == cell_id:
                        ue_link_index = active_link.link_idx
                        active_link_found = True
                        break
                
                if active_link_found:
                    # Calculate CIR power for this UE-cell link
                    # CIR data shape can be:
                    #   2D: (n_ues, n_taps)
                    #   3D: (n_ues, n_snapshots, n_taps)
                    if len(cir_data.shape) >= 2:  # 2D or 3D
                        # Try to get explicit UE-to-row mapping from H5 file first
                        ue_row_idx = self._get_ue_row_mapping(cell_id, ue_idx)
                        
                        if ue_row_idx is None:
                            # Fallback: Use ordering based on active links (UNSAFE - ordering assumption)
                            logger.debug(f"No explicit UE-to-row mapping found for cell {cell_id}, UE {ue_idx}. "
                                         "Using fallback ordering which may be incorrect.")
                            cell_ue_indices = []
                            for link in self.active_link_params:
                                if link.cid == cell_id:
                                    cell_ue_indices.append(link.uid)
                            
                            if ue_idx in cell_ue_indices:
                                ue_row_idx = cell_ue_indices.index(ue_idx)
                        
                        if ue_row_idx is not None and ue_row_idx < cir_data.shape[0]:
                            # Sum power across all snapshots and taps for this UE
                            # For 2D: cir_data[ue_row_idx, :] -> (n_taps,)
                            # For 3D: cir_data[ue_row_idx, :, :] or cir_data[ue_row_idx] -> (n_snapshots, n_taps)
                            ue_cir_power = np.sum(np.abs(cir_data[ue_row_idx])**2)
                        else:
                            logger.debug(f"Could not find valid row mapping for cell {cell_id}, UE {ue_idx}")
                    else:
                        # For 1D or unexpected shapes, sum all power (fallback)
                        ue_cir_power = np.sum(np.abs(cir_data)**2) / cir_data.size  # Normalize by size
                
                power_per_cell[cell_id] = ue_cir_power
                if ue_cir_power > max_power:
                    max_power = ue_cir_power
                    best_cell = cell_id
            
            if best_cell is not None:
                site_id, sector_id = get_site_sector(best_cell)
                serving_assignments[ue_idx] = {
                    'cell_id': best_cell,
                    'cir_power': max_power,
                    'site_id': site_id,
                    'sector_id': sector_id,
                    'power_per_cell': power_per_cell
                }
                logger.debug(f"UE {ue_idx}: serving cell = {best_cell} (Site {site_id}, Sector {sector_id}), CIR power = {max_power:.2e}")
        
        logger.info(f"Determined serving cells for {len(serving_assignments)} UEs")
        return serving_assignments

    def print_serving_cell_assignments(self, serving_assignments: Dict[int, Dict[str, any]]):
        """
        Print serving cell assignments for visual verification.
        
        Args:
            serving_assignments: Dictionary from get_serving_cell_assignments()
        """
        if not serving_assignments:
            print("No serving cell assignments available")
            return
        
        print(f"\n=== SERVING CELL ASSIGNMENTS (CIR-based) ===")
        print(f"Total UEs: {len(serving_assignments)}")
        
        # Group by serving cell for summary
        cell_ue_counts = {}
        site_ue_counts = {}
        sector_ue_counts = {}
        
        print(f"\n{'UE':<4} {'Serving Cell':<12} {'Site':<6} {'Sector':<8} {'CIR Power':<15} {'Competition':<20}")
        print(f"{'-'*75}")
        
        for ue_idx in sorted(serving_assignments.keys()):
            assignment = serving_assignments[ue_idx]
            cell_id = assignment['cell_id']
            site_id = assignment['site_id']
            sector_id = assignment['sector_id']
            cir_power = assignment['cir_power']
            power_per_cell = assignment['power_per_cell']
            
            # Count assignments
            cell_ue_counts[cell_id] = cell_ue_counts.get(cell_id, 0) + 1
            site_ue_counts[site_id] = site_ue_counts.get(site_id, 0) + 1
            sector_ue_counts[sector_id] = sector_ue_counts.get(sector_id, 0) + 1
            
            # Find second best cell for competition analysis
            sorted_powers = sorted(power_per_cell.items(), key=lambda x: x[1], reverse=True)
            competition = ""
            if len(sorted_powers) > 1:
                second_best_cell = sorted_powers[1][0]
                second_best_power = sorted_powers[1][1]
                if second_best_power > 0:
                    power_ratio_db = 10 * np.log10(cir_power / second_best_power) if second_best_power > 0 else np.inf
                    competition = f"vs Cell{second_best_cell} (+{power_ratio_db:.1f}dB)"
            
            print(f"{ue_idx:<4} {cell_id:<12} {site_id:<6} {sector_id:<8} {cir_power:<15.2e} {competition:<20}")
            
            # Print detailed power breakdown for first few UEs
            if ue_idx < 3:
                print(f"  Power from UE {ue_idx} to all cells:")
                for cell_id_detail in sorted(power_per_cell.keys()):
                    power_detail = power_per_cell[cell_id_detail]
                    power_db = 10 * np.log10(power_detail) if power_detail > 0 else -np.inf
                    is_serving = "  [SERVING]" if cell_id_detail == cell_id else ""
                    print(f"    Cell {cell_id_detail}: {power_detail:.2e} ({power_db:.1f} dB){is_serving}")
        
        # Print summary statistics
        print(f"\\n=== SERVING CELL DISTRIBUTION ===")
        print(f"Cells serving UEs:")
        for cell_id in sorted(cell_ue_counts.keys()):
            site_id = cell_id // (self.system_level_config.n_sector_per_site if self.system_level_config else 3)
            sector_id = cell_id % (self.system_level_config.n_sector_per_site if self.system_level_config else 3)
            count = cell_ue_counts[cell_id]
            percentage = (count / len(serving_assignments)) * 100
            print(f"  Cell {cell_id} (Site {site_id}, Sector {sector_id}): {count} UEs ({percentage:.1f}%)")
        
        print(f"\\nSites serving UEs:")
        for site_id in sorted(site_ue_counts.keys()):
            count = site_ue_counts[site_id]
            percentage = (count / len(serving_assignments)) * 100
            print(f"  Site {site_id}: {count} UEs ({percentage:.1f}%)")
        
        print(f"\\nSectors serving UEs:")
        for sector_id in sorted(sector_ue_counts.keys()):
            count = sector_ue_counts[sector_id]
            percentage = (count / len(serving_assignments)) * 100
            print(f"  Sector {sector_id}: {count} UEs ({percentage:.1f}%)")

    def compute_serving_cells_from_cir(self) -> Dict[int, int]:
        """
        Legacy function - use get_serving_cell_assignments() instead.
        Returns simple mapping of UE index to serving cell index.
        """
        serving_assignments = self.get_serving_cell_assignments()
        return {ue_idx: assignment['cell_id'] for ue_idx, assignment in serving_assignments.items()}

    def compute_coupling_loss_serving_cells_only(self, use_virtualization: bool = False) -> np.ndarray:
        """
        Compute coupling loss for serving cell links only.
        Serving cells are determined by maximum CIR power.
        Coupling loss = Path loss - Shadow fading - BS Antenna gain
        
        Args:
            use_virtualization: If True, use antenna virtualization for gain calculation
        
        Returns:
            Array of coupling losses for serving cell links only (negative values for loss)
        """
        logger.info(f"Computing coupling loss for serving cells only (virtualization={'ON' if use_virtualization else 'OFF'})")
        
        # If multi-seed data is available, return it directly
        if hasattr(self, '_is_multi_seed') and self._is_multi_seed and hasattr(self, '_multi_seed_coupling_losses'):
            logger.info(f"Using multi-seed combined data: {len(self._multi_seed_coupling_losses)} samples")
            return self._multi_seed_coupling_losses
        
        # For ISAC mode, use the appropriate coupling loss based on calibration phase
        if self.isac_config and self.isac_config.is_enabled:
            # Phase 2: Compute from CIR to include all small-scale effects
            if self.calibration_phase == 2:
                # ISAC uses antenna normalization (divide by n_ant) for per-antenna power
                phase2_coupling = self._compute_phase2_coupling_loss_from_cir(use_virtualization=False)
                if phase2_coupling is not None and len(phase2_coupling) > 0:
                    logger.info(f"Using Phase 2 coupling loss from CIR: {len(phase2_coupling)} samples")
                    logger.info("Phase 2 includes: PL, SF, RCS (A+B1+B2), antenna patterns, polarization, ray powers")
                    logger.info("Using antenna normalization (per-antenna power)")
                    return phase2_coupling
                else:
                    logger.warning("Phase 2 CIR computation failed, falling back to Phase 1 formula")
            
            # Phase 1 or fallback: Use saved coupling loss (large-scale only)
            isac_coupling_losses = self._get_isac_coupling_loss()
            if isac_coupling_losses is not None and len(isac_coupling_losses) > 0:
                if self.calibration_phase == 1:
                    logger.info(f"Using Phase 1 coupling loss (large-scale only): {len(isac_coupling_losses)} samples")
                else:
                    logger.warning(f"Using Phase 1 formula as fallback: {len(isac_coupling_losses)} samples")
                return isac_coupling_losses
        
        # For UMa calibration (non-ISAC): Use formula-based approach for both Phase 1 and Phase 2
        # Coupling Loss = -(PL - SF - antenna_gain)
        # Note: Both phases use the same formula. Phase 2 calibration includes additional metrics
        # (delay spread, angle spreads) but coupling loss is still computed from large-scale parameters.
        
        # Get serving cell assignments based on CIR power
        serving_assignments = self.get_serving_cell_assignments()
        
        if not serving_assignments:
            logger.warning("No serving cells determined, falling back to all links")
            return self.compute_coupling_loss_all_links(use_virtualization=use_virtualization)
        
        serving_coupling_losses = []
        
        for ue_idx, assignment in serving_assignments.items():
            serving_cell_id = assignment['cell_id']
            # Find the link parameters for this UE-serving cell pair
            for active_link in self.active_link_params:
                if active_link.uid == ue_idx and active_link.cid == serving_cell_id:
                    if active_link.lsp_read_idx < len(self.link_params):
                        link = self.link_params[active_link.lsp_read_idx]
                        
                        # Get antenna gain for this cell (with optional virtualization)
                        antenna_gain_db = self.get_antenna_gain_for_cell(
                            active_link.cid,
                            ue_id=active_link.uid,
                            use_virtualization=use_virtualization
                        )
                        
                        # Coupling loss = Path loss - Shadow fading - BS Antenna gain (use negative for loss)
                        coupling_loss = -(link.pathloss - link.sf - antenna_gain_db)
                        serving_coupling_losses.append(coupling_loss)
                    break
        
        logger.info(f"Computed coupling loss for {len(serving_coupling_losses)} serving cell links")
        return np.array(serving_coupling_losses)

    def compute_cir_power_for_link(self, cir_coe, cir_ntaps, apply_virtualization=False, 
                                   calibration_phase=1, tilt_deg=12.0, sum_ports=False) -> float:
        """
        Compute CIR power for a single link.
        
        Args:
            cir_coe: CIR coefficients array
            cir_ntaps: Number of taps (can be scalar, array, or None)
            apply_virtualization: (Unused, kept for API compatibility)
            calibration_phase: (Unused, kept for API compatibility)
            tilt_deg: (Unused, kept for API compatibility)
            sum_ports: (Unused, kept for API compatibility)
        
        Returns:
            Total CIR power (linear scale) - sum of squared magnitudes
        """
        # Get the actual number of taps used
        # Handle both scalar and array cases
        if cir_ntaps is None:
            ntaps = 0
        elif isinstance(cir_ntaps, (int, np.integer)):
            ntaps = int(cir_ntaps)
        elif hasattr(cir_ntaps, '__len__'):
            ntaps = int(cir_ntaps[0]) if len(cir_ntaps) > 0 else 0
        else:
            ntaps = int(cir_ntaps)
        
        # Compute power as sum of squared magnitudes.
        # Supported CIR shapes for this helper:
        # - New:  [nSnapshots, nRx, nTx, nTaps] or [nRx, nTx, nTaps]
        # - Comm: [nSnapshots, nAnt, nTaps]
        # - Legacy flattened: [nSnapshots, nCoeffs] or [nCoeffs]
        if ntaps <= 0:
            return 0.0

        if not hasattr(cir_coe, 'shape'):
            return float(np.sum(np.abs(cir_coe) ** 2))

        arr = np.asarray(cir_coe)

        # Heuristic: very large last-dimension implies flattened coefficients, not taps.
        if arr.ndim <= 2 and arr.shape[-1] > 4096:
            return float(np.sum(np.abs(arr) ** 2))

        # If arr has a tap axis, it is expected to be the last axis.
        if arr.shape[-1] >= ntaps:
            slicer = (slice(None),) * (arr.ndim - 1) + (slice(0, ntaps),)
            return float(np.sum(np.abs(arr[slicer]) ** 2))

        # Fallback: sum everything
        return float(np.sum(np.abs(arr) ** 2))
    
    def _apply_virtualization_to_cir(self, H, calibration_phase, tilt_deg=12.0, sum_ports=False) -> float:
        """
        Apply DFT-based antenna virtualization beamforming to CIR.
        
        Args:
            H: CIR matrix (shape: n_snapshot x n_antenna x ntaps)
            calibration_phase: 1 or 2
            tilt_deg: BS antenna tilt in degrees
            sum_ports: If True, sum power over all ports. If False, pick max port (default: False)
            
        Returns:
            Power after beamforming (linear scale)
        """
        n_snapshot, n_antenna, ntaps = H.shape
        logger.debug(f"_apply_virtualization_to_cir: phase={calibration_phase}, sum_ports={sum_ports}, shape={H.shape}")
        
        if calibration_phase == 1:
            # Phase 1: Apply w_virt to 10 antenna elements (vertical array)
            # w_virt = (1/sqrt(numM)) * exp(-1i*pi*cos(deg2rad(tilt_deg+90))*(0:numM-1).')
            numM = min(10, n_antenna)  # Use up to 10 antennas
            
            if n_antenna < numM:
                # Not enough antennas, fall back to no virtualization
                return np.sum(np.abs(H)**2)
            
            # Calculate beamforming weights
            tilt_rad = np.deg2rad(tilt_deg + 90.0)
            m_indices = np.arange(numM)
            w_virt = (1.0 / np.sqrt(numM)) * np.exp(-1j * np.pi * np.cos(tilt_rad) * m_indices)
            
            # Apply beamforming: H_virt = H[:, :numM, :] @ conj(w_virt)
            # Note: conj because a_rx*a_tx' is used during CIR generation
            # Shape: (n_snapshot, numM, ntaps) @ (numM,) -> (n_snapshot, ntaps)
            H_subset = H[:, :numM, :]  # Use first numM antennas
            H_virt = np.einsum('ijk,j->ik', H_subset, np.conj(w_virt))
            
            # Compute power: pow = H_virt(:)' * H_virt(:)
            power = np.sum(np.abs(H_virt)**2)
            
        elif calibration_phase == 2:
            # Phase 2: Group 16 elements into 1 virtualized port
            # 64 antennas -> 4 TX ports (64/16 = 4)
            n_elements_per_port = 16
            
            if n_antenna < n_elements_per_port:
                # Not enough antennas, fall back to no virtualization
                return np.sum(np.abs(H)**2)
            
            n_tx_ports = n_antenna // n_elements_per_port
            
            # Apply virtualization to each port separately
            # w_virt = (1/sqrt(N)) * exp(1j * pi * cos(theta) * n) for uniform linear array
            # For simplicity, use downtilt beamforming for each port
            tilt_rad = np.deg2rad(tilt_deg + 90.0)
            element_indices = np.arange(n_elements_per_port)
            w_port = (1.0 / np.sqrt(n_elements_per_port)) * np.exp(-1j * np.pi * np.cos(tilt_rad) * element_indices)
            
            # Calculate power from each TX port
            port_powers = []
            for port_idx in range(n_tx_ports):
                # Extract antennas for this port
                start_idx = port_idx * n_elements_per_port
                end_idx = start_idx + n_elements_per_port
                H_port = H[:, start_idx:end_idx, :]  # Shape: (n_snapshot, 16, ntaps)
                
                # Apply beamforming to this port
                H_virt_port = np.einsum('ijk,j->ik', H_port, np.conj(w_port))
                
                # Calculate power from this port
                port_power = np.sum(np.abs(H_virt_port)**2)
                port_powers.append(port_power)
            
            # Sum over all ports or pick maximum
            if sum_ports:
                power = np.sum(port_powers) if port_powers else 0.0
            else:
                power = np.max(port_powers) if port_powers else 0.0
        else:
            # Unknown phase, no virtualization
            power = np.sum(np.abs(H)**2)
        
        return float(power)

    def compute_sir_sinr_geometry(self, center_freq_hz: float = 6e9, association_method: str = 'cir',
                                  serving_assignments: Dict[int, int] = None) -> Dict[str, np.ndarray]:
        """
        Compute SIR and SINR based on CIR power and geometry.
        
        **CIR-BASED APPROACH** (includes all channel effects):
        - CIR power includes: pathloss + shadow fading + antenna array gain + small-scale fading
        - This gives ACTUAL received power, not just large-scale average
        - For Phase 1: Includes deterministic antenna array gain (~10 dB for 10-element array)
        - For Phase 2: Includes full small-scale fading effects
        
        SIR = Signal CIR power / Interference CIR power (relative power ratio)
        SINR = Signal RX power / (Interference RX power + Noise power)
             where RX power (dBm) = TX power (dBm) + CIR gain (dB)
        
        Args:
            center_freq_hz: Carrier frequency in Hz
            association_method: Association method to use:
                - 'rsrp': RSRP-based (pathloss - shadow fading)
                - 'cir': CIR power-based (legacy) [default]
                - 'distance': Minimal distance-based
            serving_assignments: Pre-computed serving cell assignments {ue_idx: cell_id}.
                If None, will compute using association_method.
            
        Returns:
            Dictionary with SIR and SINR arrays
        """
        logger.info("="*80)
        logger.info(f"CIR-BASED SIR/SINR CALCULATION")
        logger.info("="*80)
        logger.info("Using raw CIR power (sum of all taps and antennas)")
        logger.info("This is the CORRECT approach for 3GPP calibration (not pathloss-only)")
        
        # Get system parameters (includes TX power based on scenario and frequency)
        sys_params = self.get_system_parameters(
            self.system_level_config.scenario if self.system_level_config else 'UMa',
            center_freq_hz
        )
        tx_power_dbm = sys_params['tx_power_dbm']
        bandwidth_hz = sys_params['bandwidth_hz']
        noise_figure_db = sys_params['noise_figure_db']
        
        logger.info(f"System parameters: TX={tx_power_dbm} dBm, BW={bandwidth_hz/1e6:.0f} MHz, NF={noise_figure_db} dB")
        
        # Calculate thermal noise power
        noise_power_dbm = self.calculate_thermal_noise_power(bandwidth_hz, noise_figure_db)
        logger.info(f"Noise power: {noise_power_dbm:.2f} dBm")
        
        # Use pre-computed serving assignments if provided, otherwise compute them
        if serving_assignments is None:
            serving_assignments = self._compute_serving_assignments(association_method)
        
        # Convert simple {ue_idx: cell_id} format to the format expected by this function
        # which is {ue_idx: {'cell_id': cell_id, 'power_per_cell': {...}}}
        if serving_assignments and isinstance(next(iter(serving_assignments.values())), int):
            # Simple format, need to compute power_per_cell for ALL cells
            logger.info("Computing CIR powers for all cells for SIR/SINR calculation...")
            full_assignments = {}
            for ue_idx, cell_id in serving_assignments.items():
                # Compute CIR power for all cells for this UE
                power_per_cell = {}
                for cell in self.cell_params:
                    cid = cell.cid
                    if cid in self.cir_per_cell:
                        cir_data = self.cir_per_cell[cid]
                        
                        # Get the UE row mapping
                        ue_row_idx = self._get_ue_row_mapping(cid, ue_idx)
                        
                        if ue_row_idx is None:
                            # Fallback: Use ordering based on active links
                            cell_ue_indices = []
                            for l in self.active_link_params:
                                if l.cid == cid:
                                    cell_ue_indices.append(l.uid)
                            if ue_idx in cell_ue_indices:
                                ue_row_idx = cell_ue_indices.index(ue_idx)
                        
                        if ue_row_idx is not None and ue_row_idx < cir_data.shape[0]:
                            # Get ntaps if available
                            ntaps = None
                            if cid in self.cir_ntaps_per_cell:
                                ntaps_data = self.cir_ntaps_per_cell[cid]
                                if ue_row_idx < len(ntaps_data):
                                    ntaps = ntaps_data[ue_row_idx]
                            
                            # Compute CIR power (sum of all taps and antennas, no virtualization)
                            cir_power = self.compute_cir_power_for_link(
                                cir_data[ue_row_idx], 
                                ntaps,
                                apply_virtualization=False,
                                calibration_phase=self.calibration_phase,
                                tilt_deg=12.0,
                                sum_ports=False
                            )
                            power_per_cell[cid] = cir_power
                
                full_assignments[ue_idx] = {
                    'cell_id': cell_id,
                    'power_per_cell': power_per_cell
                }
            serving_assignments = full_assignments
            logger.info(f"CIR powers computed for {len(serving_assignments)} UEs across {len(self.cell_params)} cells")
        
        if not serving_assignments:
            logger.warning("No serving cell assignments available for SIR/SINR calculation")
            return {'SIR': np.array([]), 'SINR': np.array([])}
        
        n_uts = len(serving_assignments)
        
        # Create power matrix from CIR power (linear scale)
        n_cells = len(self.cell_params)
        power_matrix_linear = np.full((n_cells, n_uts), 0.0)
        
        # Fill power matrix with CIR powers
        for ue_idx, assignment in serving_assignments.items():
            power_per_cell = assignment['power_per_cell']
            for cell_id, cir_power in power_per_cell.items():
                if cell_id < n_cells and ue_idx < n_uts:
                    power_matrix_linear[cell_id, ue_idx] = cir_power
        
        # Get serving cells and calculate SIR/SINR
        serving_cells = np.zeros(n_uts, dtype=int)
        sir_db = np.zeros(n_uts)
        sinr_db = np.zeros(n_uts)
        
        # Arrays to track power values for logging
        serving_powers_log = []
        interference_powers_log = []
        
        # Track interference breakdown for each UE (for debugging)
        interference_breakdown = {}  # {ue_idx: [(cell_id, power_dbm), ...]}
        
        # Noise power in linear scale (convert from dBm)
        noise_power_linear = 10**(noise_power_dbm / 10)
        
        for ue_idx in serving_assignments.keys():
            assignment = serving_assignments[ue_idx]
            serving_cell_idx = assignment['cell_id']
            serving_cells[ue_idx] = serving_cell_idx
            
            # Get CIR powers for this UE (already in linear scale)
            ue_powers_linear = power_matrix_linear[:, ue_idx]
            
            # Serving CIR power (channel gain)
            serving_cir_linear = ue_powers_linear[serving_cell_idx]
            
            if serving_cir_linear > 0:
                # Calculate interference from all other cells (sum of CIR powers)
                interfering_cir_linear = np.delete(ue_powers_linear, serving_cell_idx)
                total_interference_cir_linear = np.sum(interfering_cir_linear)
                
                # Store interference breakdown (cell_id -> power_dbm)
                interf_list = []
                for cell_idx in range(len(ue_powers_linear)):
                    if cell_idx != serving_cell_idx:
                        cir_power = ue_powers_linear[cell_idx]
                        if cir_power > 0:
                            cir_db = 10 * np.log10(cir_power)
                            rx_power_dbm = tx_power_dbm + cir_db
                            interf_list.append((cell_idx, rx_power_dbm, cir_db))
                # Sort by power (highest first)
                interf_list.sort(key=lambda x: x[1], reverse=True)
                interference_breakdown[ue_idx] = interf_list
                
                # Convert CIR to dB for logging
                serving_cir_db = 10 * np.log10(serving_cir_linear) if serving_cir_linear > 0 else -np.inf
                total_interference_cir_db = 10 * np.log10(total_interference_cir_linear) if total_interference_cir_linear > 0 else -np.inf
                
                # Calculate absolute received powers (dBm) = TX power (dBm) + CIR (dB)
                serving_rx_power_dbm = tx_power_dbm + serving_cir_db
                total_interference_rx_power_dbm = tx_power_dbm + total_interference_cir_db
                
                # Convert to linear for SINR calculation
                serving_rx_power_linear = 10**(serving_rx_power_dbm / 10)
                total_interference_rx_power_linear = 10**(total_interference_rx_power_dbm / 10)
                noise_power_linear = 10**(noise_power_dbm / 10)
                
                # Log power values for first few UEs with detailed breakdown
                if ue_idx < 5:
                    # Get pathloss for comparison
                    pathloss_based_gain_db = -np.inf
                    for active_link in self.active_link_params:
                        if active_link.uid == ue_idx and active_link.cid == serving_cell_idx:
                            if active_link.lsp_read_idx < len(self.link_params):
                                link = self.link_params[active_link.lsp_read_idx]
                                pathloss_based_gain_db = -(link.pathloss - link.sf)
                                antenna_gain_contribution = serving_cir_db - pathloss_based_gain_db
                                logger.info(f"UE {ue_idx}: Serving Cell {serving_cell_idx}")
                                logger.info(f"  === DISTANCE ===")
                                logger.info(f"  d_2d: {link.d2d:.2f} m")
                                logger.info(f"  d_3d: {link.d3d:.2f} m")
                                logger.info(f"  === LOS/K-FACTOR ===")
                                logger.info(f"  LOS: {'Yes' if link.los_ind else 'No'}")
                                # K-factor is already stored in dB in the H5 file
                                k_factor_db = link.k_factor  # Already in dB
                                k_factor_linear = 10**(k_factor_db / 10.0) if np.isfinite(k_factor_db) else 0.0
                                logger.info(f"  K-factor: {k_factor_linear:.6f} linear, {k_factor_db:.2f} dB")
                                logger.info(f"  === CIR-BASED (includes array gain) ===")
                                logger.info(f"  CIR gain: {serving_cir_db:.2f} dB")
                                logger.info(f"  === Pathloss-based (large-scale only) ===")
                                logger.info(f"  Pathloss gain: {pathloss_based_gain_db:.2f} dB")
                                logger.info(f"  === DIFFERENCE (antenna array + fading gain) ===")
                                logger.info(f"  Additional gain from CIR: {antenna_gain_contribution:.2f} dB")
                                logger.info(f"  Expected ~10 dB for 10-antenna array in Phase 1")
                                break
                    
                    logger.info(f"  Signal CIR: {serving_cir_db:.2f} dB, Interference CIR: {total_interference_cir_db:.2f} dB")
                    logger.info(f"  Signal RX Power: {serving_rx_power_dbm:.2f} dBm (TX={tx_power_dbm} dBm + CIR={serving_cir_db:.2f} dB)")
                    logger.info(f"  Interference RX Power: {total_interference_rx_power_dbm:.2f} dBm")
                    logger.info(f"  Noise Power: {noise_power_dbm:.2f} dBm")
                
                # Store for statistics (use RX power for logging)
                serving_powers_log.append(serving_rx_power_dbm)
                interference_powers_log.append(total_interference_rx_power_dbm)
                
                # Calculate SIR using relative CIR power (channel gain ratio)
                if total_interference_cir_linear > 0:
                    sir_linear = serving_cir_linear / total_interference_cir_linear
                    sir_db[ue_idx] = 10 * np.log10(sir_linear)
                else:
                    sir_db[ue_idx] = np.inf
                
                # Calculate SINR using absolute received power with noise
                sinr_linear = serving_rx_power_linear / (total_interference_rx_power_linear + noise_power_linear)
                sinr_db[ue_idx] = 10 * np.log10(sinr_linear)
                
                if ue_idx < 5:
                    logger.info(f"  SIR: {sir_db[ue_idx]:.2f} dB")
                    logger.info(f"  SINR: {sinr_db[ue_idx]:.2f} dB")
            else:
                sir_db[ue_idx] = np.nan
                sinr_db[ue_idx] = np.nan
        
        # Log statistics for all UEs with CIR vs Pathloss comparison
        if len(serving_powers_log) > 0:
            logger.info(f"\n" + "="*80)
            logger.info(f"POWER STATISTICS (all {len(serving_powers_log)} UEs) - CIR-BASED")
            logger.info("="*80)
            logger.info(f"  TX Power: {tx_power_dbm:.2f} dBm")
            logger.info(f"  Noise Power: {noise_power_dbm:.2f} dBm")
            logger.info(f"  ---")
            logger.info(f"  RX Signal Power (TX+CIR): Mean={np.mean(serving_powers_log):.2f} dBm, "
                       f"Range=[{np.min(serving_powers_log):.2f}, {np.max(serving_powers_log):.2f}] dBm")
            logger.info(f"  RX Interference Power (TX+CIR): Mean={np.mean(interference_powers_log):.2f} dBm, "
                       f"Range=[{np.min(interference_powers_log):.2f}, {np.max(interference_powers_log):.2f}] dBm")
            
            # Calculate average CIR gain vs pathloss gain for all UEs
            cir_gains = []
            pathloss_gains = []
            for ue_idx in serving_assignments.keys():
                if ue_idx < len(serving_cells):
                    serving_cell_idx = serving_cells[ue_idx]
                    ue_powers_linear = power_matrix_linear[:, ue_idx]
                    serving_cir_linear = ue_powers_linear[serving_cell_idx]
                    if serving_cir_linear > 0:
                        cir_gain_db = 10 * np.log10(serving_cir_linear)
                        cir_gains.append(cir_gain_db)
                        
                        # Get pathloss gain
                        for active_link in self.active_link_params:
                            if active_link.uid == ue_idx and active_link.cid == serving_cell_idx:
                                if active_link.lsp_read_idx < len(self.link_params):
                                    link = self.link_params[active_link.lsp_read_idx]
                                    pathloss_gain = -(link.pathloss - link.sf)
                                    pathloss_gains.append(pathloss_gain)
                                    break
            
            if len(cir_gains) > 0 and len(pathloss_gains) > 0:
                avg_antenna_gain = np.mean(np.array(cir_gains) - np.array(pathloss_gains))
                logger.info(f"  ---")
                logger.info(f"  Average CIR gain: {np.mean(cir_gains):.2f} dB")
                logger.info(f"  Average Pathloss gain: {np.mean(pathloss_gains):.2f} dB")
                logger.info(f"  Average Antenna Array + Fading Gain: {avg_antenna_gain:.2f} dB")
                logger.info(f"  (Expected ~10 dB for 10-antenna BS array in Phase 1)")
            logger.info("="*80)
        
        # Collect K-factor statistics for serving cells
        k_factors_linear = []
        k_factors_db = []
        k_factors_los = []
        k_factors_nlos = []
        k_factor_sinr_pairs = []  # (k_factor_db, sinr_db, distance, los_ind)
        
        for ue_idx in serving_assignments.keys():
            if ue_idx < len(serving_cells):
                serving_cell_idx = serving_cells[ue_idx]
                for active_link in self.active_link_params:
                    if active_link.uid == ue_idx and active_link.cid == serving_cell_idx:
                        if active_link.lsp_read_idx < len(self.link_params):
                            link = self.link_params[active_link.lsp_read_idx]
                            # K-factor is already stored in dB in the H5 file
                            k_db = link.k_factor  # Already in dB
                            k_lin = 10**(k_db / 10.0) if np.isfinite(k_db) else 0.0  # Convert dB to linear
                            k_factors_linear.append(k_lin)
                            k_factors_db.append(k_db)
                            
                            if link.los_ind:
                                k_factors_los.append(k_db)
                            else:
                                k_factors_nlos.append(k_db)
                            
                            if ue_idx < len(sinr_db) and np.isfinite(sinr_db[ue_idx]):
                                k_factor_sinr_pairs.append((k_db, sinr_db[ue_idx], link.d2d, link.los_ind))
                            break
        
        if len(k_factors_db) > 0:
            logger.info(f"\n" + "="*80)
            logger.info(f"K-FACTOR STATISTICS FOR SERVING CELLS")
            logger.info("="*80)
            
            # Overall statistics
            k_factors_db_finite = [k for k in k_factors_db if np.isfinite(k)]
            if len(k_factors_db_finite) > 0:
                logger.info(f"  Overall K-factor (dB):")
                logger.info(f"    Count: {len(k_factors_db_finite)} UEs")
                logger.info(f"    Mean: {np.mean(k_factors_db_finite):.2f} dB")
                logger.info(f"    Std: {np.std(k_factors_db_finite):.2f} dB")
                logger.info(f"    Median: {np.median(k_factors_db_finite):.2f} dB")
                logger.info(f"    Range: [{np.min(k_factors_db_finite):.2f}, {np.max(k_factors_db_finite):.2f}] dB")
                logger.info(f"    Percentiles: 10%={np.percentile(k_factors_db_finite, 10):.2f}, "
                           f"50%={np.percentile(k_factors_db_finite, 50):.2f}, "
                           f"90%={np.percentile(k_factors_db_finite, 90):.2f} dB")
            
            # LOS vs NLOS
            logger.info(f"\n  K-factor by LOS/NLOS:")
            if len(k_factors_los) > 0:
                logger.info(f"    LOS ({len(k_factors_los)} UEs):")
                logger.info(f"      Mean: {np.mean(k_factors_los):.2f} dB")
                logger.info(f"      Range: [{np.min(k_factors_los):.2f}, {np.max(k_factors_los):.2f}] dB")
            else:
                logger.info(f"    LOS: No LOS links")
            
            if len(k_factors_nlos) > 0:
                logger.info(f"    NLOS ({len(k_factors_nlos)} UEs):")
                logger.info(f"      Mean: {np.mean(k_factors_nlos):.2f} dB")
                logger.info(f"      Range: [{np.min(k_factors_nlos):.2f}, {np.max(k_factors_nlos):.2f}] dB")
            else:
                logger.info(f"    NLOS: No NLOS links")
            
            # K-factor vs SINR correlation for high SINR UEs
            if len(k_factor_sinr_pairs) > 0:
                logger.info(f"\n  K-factor vs SINR Analysis:")
                # Sort by SINR (descending)
                k_factor_sinr_pairs.sort(key=lambda x: x[1], reverse=True)
                
                # Top 10 SINR UEs
                top10 = k_factor_sinr_pairs[:10]
                logger.info(f"    Top 10 SINR UEs:")
                for i, (k_db, sinr, dist, los) in enumerate(top10):
                    logger.info(f"      #{i+1}: SINR={sinr:.2f} dB, K-factor={k_db:.2f} dB, "
                               f"d_2d={dist:.1f}m, LOS={'Yes' if los else 'No'}")
                
                # High SINR (>20 dB) statistics
                high_sinr = [pair for pair in k_factor_sinr_pairs if pair[1] > 20.0]
                if len(high_sinr) > 0:
                    k_high_sinr = [pair[0] for pair in high_sinr if np.isfinite(pair[0])]
                    logger.info(f"\n    High SINR (>20 dB) UEs ({len(high_sinr)} UEs):")
                    if len(k_high_sinr) > 0:
                        logger.info(f"      K-factor Mean: {np.mean(k_high_sinr):.2f} dB")
                        logger.info(f"      K-factor Range: [{np.min(k_high_sinr):.2f}, {np.max(k_high_sinr):.2f}] dB")
                    los_count = sum(1 for pair in high_sinr if pair[3])
                    logger.info(f"      LOS: {los_count}/{len(high_sinr)} ({100*los_count/len(high_sinr):.1f}%)")
                    avg_dist = np.mean([pair[2] for pair in high_sinr])
                    logger.info(f"      Average distance: {avg_dist:.1f}m")
            
            logger.info("="*80)
        
        # Print detailed K-factor for each UE's serving link
        print(f"\n=== PER-UE K-FACTOR (Serving Link) ===")
        print(f"\n{'UE':<6} {'Cell':<6} {'d_2d(m)':<10} {'LOS':<6} {'K (linear)':<15} {'K (dB)':<12} {'SINR (dB)':<12}")
        print("-" * 80)
        for ue_idx in range(min(50, n_uts)):  # Print first 50 UEs
            if ue_idx in serving_assignments and ue_idx < len(serving_cells):
                serving_cell = serving_cells[ue_idx]
                
                # Get K-factor and distance information
                k_linear = np.nan
                k_db = np.nan
                d_2d_val = np.nan
                los_status = "N/A"
                sinr_val = sinr_db[ue_idx] if ue_idx < len(sinr_db) else np.nan
                
                for active_link in self.active_link_params:
                    if active_link.uid == ue_idx and active_link.cid == serving_cell:
                        if active_link.lsp_read_idx < len(self.link_params):
                            link = self.link_params[active_link.lsp_read_idx]
                            # K-factor is already stored in dB in the H5 file
                            k_db = link.k_factor  # Already in dB
                            k_linear = 10**(k_db / 10.0) if np.isfinite(k_db) else 0.0  # Convert dB to linear
                            d_2d_val = link.d2d
                            los_status = "LOS" if link.los_ind else "NLOS"
                            break
                
                print(f"{ue_idx:<6} {serving_cell:<6} {d_2d_val:<10.1f} {los_status:<6} {k_linear:<15.6e} {k_db:<12.2f} {sinr_val:<12.2f}")
        
        # Print detailed SIR/SINR for all UEs
        print(f"\n=== PER-UE SIR/SINR BREAKDOWN (CIR-based with TX power) ===")
        print(f"TX Power: {tx_power_dbm} dBm, Noise Power: {noise_power_dbm:.2f} dBm")
        print(f"\n{'UE':<6} {'Cell':<6} {'d_2d(m)':<10} {'d_3d(m)':<10} {'RX Signal':<12} {'RX Interf':<12} {'SIR':<10} {'SINR':<10}")
        print("-" * 100)
        for ue_idx in range(min(20, n_uts)):  # Print first 20 UEs
            if ue_idx in serving_assignments and np.isfinite(sir_db[ue_idx]) and np.isfinite(sinr_db[ue_idx]):
                serving_cell = serving_cells[ue_idx]
                signal_pwr_dbm = serving_powers_log[ue_idx] if ue_idx < len(serving_powers_log) else np.nan
                interf_pwr_dbm = interference_powers_log[ue_idx] if ue_idx < len(interference_powers_log) else np.nan
                
                # Get distance information
                d_2d_val = np.nan
                d_3d_val = np.nan
                for active_link in self.active_link_params:
                    if active_link.uid == ue_idx and active_link.cid == serving_cell:
                        if active_link.lsp_read_idx < len(self.link_params):
                            link = self.link_params[active_link.lsp_read_idx]
                            d_2d_val = link.d2d
                            d_3d_val = link.d3d
                            break
                
                # Convert to linear for breakdown
                signal_mw = 10**(signal_pwr_dbm / 10)
                interf_mw = 10**(interf_pwr_dbm / 10)
                noise_mw = 10**(noise_power_dbm / 10)
                
                # Calculate SINR components
                interf_plus_noise_mw = interf_mw + noise_mw
                interf_plus_noise_dbm = 10 * np.log10(interf_plus_noise_mw)
                
                # Show breakdown
                breakdown = f"S={signal_pwr_dbm:.1f} / (I={interf_pwr_dbm:.1f} + N={noise_power_dbm:.1f}) = {signal_pwr_dbm:.1f} / {interf_plus_noise_dbm:.1f}"
                
                print(f"{ue_idx:<6} {serving_cell:<6} {d_2d_val:<10.1f} {d_3d_val:<10.1f} {signal_pwr_dbm:<12.2f} {interf_pwr_dbm:<12.2f} {sir_db[ue_idx]:<10.2f} {sinr_db[ue_idx]:<10.2f} {breakdown}")
        
        # Print detailed analysis for high SINR UEs (SINR > 25 dB)
        high_sinr_ues = [(ue_idx, sinr_db[ue_idx]) for ue_idx in range(n_uts) 
                         if ue_idx < len(sinr_db) and np.isfinite(sinr_db[ue_idx]) and sinr_db[ue_idx] > 25.0]
        
        if len(high_sinr_ues) > 0:
            print(f"\n{'='*100}")
            print(f"DETAILED ANALYSIS FOR HIGH SINR UEs (SINR > 25 dB) - Total: {len(high_sinr_ues)} UEs")
            print(f"{'='*100}")
            
            # Sort by SINR (highest first)
            high_sinr_ues.sort(key=lambda x: x[1], reverse=True)
            
            for ue_idx, sinr_val in high_sinr_ues[:20]:  # Print top 20 high SINR UEs
                if ue_idx in serving_assignments and ue_idx < len(serving_cells):
                    serving_cell = serving_cells[ue_idx]
                    
                    print(f"\n{'='*100}")
                    print(f"UE {ue_idx}: SINR = {sinr_val:.2f} dB, SIR = {sir_db[ue_idx]:.2f} dB")
                    print(f"{'='*100}")
                    
                    # Get link parameters for this UE-serving cell pair
                    for active_link in self.active_link_params:
                        if active_link.uid == ue_idx and active_link.cid == serving_cell:
                            if active_link.lsp_read_idx < len(self.link_params):
                                link = self.link_params[active_link.lsp_read_idx]
                                
                                # Get CIR and pathloss gains
                                ue_powers_linear = power_matrix_linear[:, ue_idx]
                                serving_cir_linear = ue_powers_linear[serving_cell]
                                cir_gain_db = 10 * np.log10(serving_cir_linear) if serving_cir_linear > 0 else -np.inf
                                pathloss_gain_db = -(link.pathloss - link.sf)
                                antenna_contribution = cir_gain_db - pathloss_gain_db
                                
                                # K-factor
                                k_db = link.k_factor
                                k_linear = 10**(k_db / 10.0) if np.isfinite(k_db) else 0.0
                                
                                # Power values
                                signal_pwr_dbm = serving_powers_log[ue_idx] if ue_idx < len(serving_powers_log) else np.nan
                                interf_pwr_dbm = interference_powers_log[ue_idx] if ue_idx < len(interference_powers_log) else np.nan
                                
                                # Get UE location from utParams if available
                                ue_location = None
                                if hasattr(self, 'ut_params') and ue_idx < len(self.ut_params):
                                    ue_location = self.ut_params[ue_idx].get('location') if isinstance(self.ut_params[ue_idx], dict) else None
                                
                                # Get serving cell location
                                serving_cell_location = None
                                serving_cell_site_id = None
                                for cell in self.cell_params:
                                    if cell.cid == serving_cell:
                                        serving_cell_location = cell.location
                                        serving_cell_site_id = cell.site_id
                                        break
                                
                                print(f"  Serving Cell: {serving_cell} (Site {serving_cell_site_id})" if serving_cell_site_id is not None else f"  Serving Cell: {serving_cell}")
                                if serving_cell_location:
                                    print(f"  Serving Cell Location: ({serving_cell_location['x']:.1f}, {serving_cell_location['y']:.1f}, {serving_cell_location['z']:.1f}) m")
                                if ue_location:
                                    print(f"  UE Location: ({ue_location['x']:.1f}, {ue_location['y']:.1f}, {ue_location['z']:.1f}) m")
                                print(f"  Distance: d_2d = {link.d2d:.1f} m, d_3d = {link.d3d:.1f} m")
                                print(f"  LOS State: {'LOS' if link.los_ind else 'NLOS'}")
                                print(f"  K-factor: {k_linear:.6e} linear ({k_db:.2f} dB)")
                                print(f"")
                                print(f"  === CHANNEL GAINS ===")
                                print(f"  CIR Gain (includes antenna array + fading): {cir_gain_db:.2f} dB")
                                print(f"  Pathloss Gain (large-scale only):          {pathloss_gain_db:.2f} dB")
                                print(f"  Antenna Array + Fading Contribution:        {antenna_contribution:.2f} dB")
                                print(f"")
                                print(f"  === RECEIVED POWERS ===")
                                print(f"  TX Power:           {tx_power_dbm:.2f} dBm")
                                print(f"  Signal RX Power:    {signal_pwr_dbm:.2f} dBm  (TX + CIR gain)")
                                print(f"  Interference Power: {interf_pwr_dbm:.2f} dBm")
                                print(f"  Noise Power:        {noise_power_dbm:.2f} dBm")
                                print(f"")
                                print(f"  === SIR/SINR ===")
                                print(f"  SIR  = {sir_db[ue_idx]:.2f} dB  (Signal / Interference, no noise)")
                                print(f"  SINR = {sinr_val:.2f} dB  (Signal / (Interference + Noise))")
                                
                                # Calculate noise contribution to total interference+noise
                                interf_linear = 10**(interf_pwr_dbm / 10)
                                noise_linear = 10**(noise_power_dbm / 10)
                                total_in = interf_linear + noise_linear
                                noise_contribution_pct = (noise_linear / total_in) * 100
                                interf_contribution_pct = (interf_linear / total_in) * 100
                                
                                print(f"")
                                print(f"  === INTERFERENCE + NOISE BREAKDOWN ===")
                                print(f"  Interference contribution: {interf_contribution_pct:.1f}%")
                                print(f"  Noise contribution:        {noise_contribution_pct:.1f}%")
                                
                                # Detailed interference breakdown from each cell with co-sited analysis
                                if ue_idx in interference_breakdown:
                                    interf_list = interference_breakdown[ue_idx]
                                    
                                    # Calculate co-sited vs other-site interference
                                    co_sited_interf_power_linear = 0.0
                                    other_site_interf_power_linear = 0.0
                                    
                                    for cell_idx, rx_pwr, cir_db in interf_list:
                                        # Get interfering cell site
                                        interf_site_id = None
                                        for cell in self.cell_params:
                                            if cell.cid == cell_idx:
                                                interf_site_id = cell.site_id
                                                break
                                        
                                        rx_pwr_linear = 10**(rx_pwr / 10)
                                        if interf_site_id == serving_cell_site_id:
                                            co_sited_interf_power_linear += rx_pwr_linear
                                        else:
                                            other_site_interf_power_linear += rx_pwr_linear
                                    
                                    total_interf_linear = co_sited_interf_power_linear + other_site_interf_power_linear
                                    co_sited_pct = (co_sited_interf_power_linear / total_interf_linear * 100) if total_interf_linear > 0 else 0
                                    other_site_pct = (other_site_interf_power_linear / total_interf_linear * 100) if total_interf_linear > 0 else 0
                                    
                                    print(f"")
                                    print(f"  === CO-SITED vs OTHER-SITE INTERFERENCE ===")
                                    if co_sited_interf_power_linear > 0:
                                        co_sited_dbm = 10 * np.log10(co_sited_interf_power_linear)
                                        print(f"  Co-sited interference (Site {serving_cell_site_id}): {co_sited_dbm:.2f} dBm ({co_sited_pct:.1f}%)")
                                    else:
                                        print(f"  Co-sited interference: None")
                                    
                                    if other_site_interf_power_linear > 0:
                                        other_site_dbm = 10 * np.log10(other_site_interf_power_linear)
                                        print(f"  Other-site interference: {other_site_dbm:.2f} dBm ({other_site_pct:.1f}%)")
                                    else:
                                        print(f"  Other-site interference: None")
                                    
                                    print(f"")
                                    print(f"  === TOP INTERFERING CELLS ===")
                                    print(f"  {'Cell':<8} {'Site':<6} {'Type':<10} {'RX Power':<12} {'CIR Gain':<12} {'Distance':<12} {'Rel to Signal':<15}")
                                    print(f"  {'-'*95}")
                                    for i, (cell_idx, rx_pwr, cir_db) in enumerate(interf_list[:5]):  # Top 5 interferers
                                        rel_db = rx_pwr - signal_pwr_dbm
                                        # Find distance to interfering cell and its location
                                        interf_dist = None
                                        interf_cell_location = None
                                        interf_site_id = None
                                        for al in self.active_link_params:
                                            if al.uid == ue_idx and al.cid == cell_idx:
                                                if al.lsp_read_idx < len(self.link_params):
                                                    interf_dist = self.link_params[al.lsp_read_idx].d2d
                                                    break
                                        # Get interfering cell location
                                        for cell in self.cell_params:
                                            if cell.cid == cell_idx:
                                                interf_cell_location = cell.location
                                                interf_site_id = cell.site_id
                                                break
                                        dist_str = f"{interf_dist:.1f} m" if interf_dist is not None else "N/A"
                                        site_str = f"{interf_site_id}" if interf_site_id is not None else "N/A"
                                        interf_type = "CO-SITED" if interf_site_id == serving_cell_site_id else "Other"
                                        print(f"  {cell_idx:<8} {site_str:<6} {interf_type:<10} {rx_pwr:<12.2f} {cir_db:<12.2f} {dist_str:<12} {rel_db:>+6.2f} dB")
                                        if interf_cell_location:
                                            print(f"    Location: ({interf_cell_location['x']:.1f}, {interf_cell_location['y']:.1f}, {interf_cell_location['z']:.1f}) m")
                                
                                break
            
            print(f"\n{'='*100}")
            print(f"SUMMARY: {len(high_sinr_ues)} UEs with SINR > 25 dB")
            if len(high_sinr_ues) > 0:
                high_sinrs = [sinr for _, sinr in high_sinr_ues]
                print(f"  SINR Range: [{np.min(high_sinrs):.2f}, {np.max(high_sinrs):.2f}] dB")
                print(f"  SINR Mean:  {np.mean(high_sinrs):.2f} dB")
                print(f"  SINR Median: {np.median(high_sinrs):.2f} dB")
                
                # Analyze why SINR might be limited
                print(f"\n  === ANALYSIS: Why SINR doesn't reach 35 dB ===")
                
                # Check K-factor distribution for high SINR UEs
                k_factors_high_sinr = []
                los_count = 0
                distances = []
                for ue_idx, _ in high_sinr_ues:
                    if ue_idx < len(serving_cells):
                        serving_cell = serving_cells[ue_idx]
                        for active_link in self.active_link_params:
                            if active_link.uid == ue_idx and active_link.cid == serving_cell:
                                if active_link.lsp_read_idx < len(self.link_params):
                                    link = self.link_params[active_link.lsp_read_idx]
                                    k_factors_high_sinr.append(link.k_factor)
                                    if link.los_ind:
                                        los_count += 1
                                    distances.append(link.d2d)
                                    break
                
                if len(k_factors_high_sinr) > 0:
                    print(f"  K-factor for high SINR UEs: Mean = {np.mean(k_factors_high_sinr):.2f} dB, "
                          f"Max = {np.max(k_factors_high_sinr):.2f} dB")
                    print(f"  LOS percentage: {100*los_count/len(high_sinr_ues):.1f}% ({los_count}/{len(high_sinr_ues)})")
                    print(f"  Distance: Mean = {np.mean(distances):.1f} m, Min = {np.min(distances):.1f} m")
                    print(f"")
                    print(f"  Possible reasons for SINR < 35 dB:")
                    print(f"  1. **CO-SITED INTERFERENCE** - Other sectors at same site cause strong interference")
                    print(f"  2. Interference from neighboring cells (check 'CO-SITED vs OTHER-SITE' above)")
                    print(f"  3. K-factor not high enough (typical LOS K ~ 5-10 dB for close UEs)")
                    print(f"  4. UE distances too far (should be < 50m for very high SINR)")
                    print(f"  5. Shadow fading variations reducing signal strength")
            print(f"{'='*100}\n")
        else:
            print(f"\n{'='*100}")
            print(f"NO UEs with SINR > 25 dB found in simulation")
            print(f"{'='*100}")
            print(f"Max SINR: {np.max(sinr_db[np.isfinite(sinr_db)]):.2f} dB")
            print(f"{'='*100}\n")
        
        # Print detailed analysis for UEs within 50m of serving cell
        close_ues = []
        for ue_idx in range(n_uts):
            if ue_idx in serving_assignments and ue_idx < len(serving_cells):
                serving_cell = serving_cells[ue_idx]
                for active_link in self.active_link_params:
                    if active_link.uid == ue_idx and active_link.cid == serving_cell:
                        if active_link.lsp_read_idx < len(self.link_params):
                            link = self.link_params[active_link.lsp_read_idx]
                            if link.d2d <= 50.0:
                                sinr_val = sinr_db[ue_idx] if ue_idx < len(sinr_db) and np.isfinite(sinr_db[ue_idx]) else np.nan
                                close_ues.append((ue_idx, link.d2d, sinr_val))
                            break
        
        if len(close_ues) > 0:
            print(f"\n{'='*100}")
            print(f"DETAILED ANALYSIS FOR CLOSE UEs (d_2d <= 50m) - Total: {len(close_ues)} UEs")
            print(f"{'='*100}")
            
            # Sort by distance (closest first)
            close_ues.sort(key=lambda x: x[1])
            
            for ue_idx, distance, sinr_val in close_ues[:20]:  # Print top 20 closest UEs
                if ue_idx in serving_assignments and ue_idx < len(serving_cells):
                    serving_cell = serving_cells[ue_idx]
                    
                    print(f"\n{'='*100}")
                    print(f"UE {ue_idx}: d_2d = {distance:.1f} m, SINR = {sinr_val:.2f} dB, SIR = {sir_db[ue_idx]:.2f} dB")
                    print(f"{'='*100}")
                    
                    # Get link parameters for this UE-serving cell pair
                    for active_link in self.active_link_params:
                        if active_link.uid == ue_idx and active_link.cid == serving_cell:
                            if active_link.lsp_read_idx < len(self.link_params):
                                link = self.link_params[active_link.lsp_read_idx]
                                
                                # Get CIR and pathloss gains
                                ue_powers_linear = power_matrix_linear[:, ue_idx]
                                serving_cir_linear = ue_powers_linear[serving_cell]
                                cir_gain_db = 10 * np.log10(serving_cir_linear) if serving_cir_linear > 0 else -np.inf
                                pathloss_gain_db = -(link.pathloss - link.sf)
                                antenna_contribution = cir_gain_db - pathloss_gain_db
                                
                                # K-factor
                                k_db = link.k_factor
                                k_linear = 10**(k_db / 10.0) if np.isfinite(k_db) else 0.0
                                
                                # Power values
                                signal_pwr_dbm = serving_powers_log[ue_idx] if ue_idx < len(serving_powers_log) else np.nan
                                interf_pwr_dbm = interference_powers_log[ue_idx] if ue_idx < len(interference_powers_log) else np.nan
                                
                                # Get UE location from utParams if available
                                ue_location = None
                                if hasattr(self, 'ut_params') and ue_idx < len(self.ut_params):
                                    ue_location = self.ut_params[ue_idx].get('location') if isinstance(self.ut_params[ue_idx], dict) else None
                                
                                # Get serving cell location
                                serving_cell_location = None
                                serving_cell_site_id = None
                                for cell in self.cell_params:
                                    if cell.cid == serving_cell:
                                        serving_cell_location = cell.location
                                        serving_cell_site_id = cell.site_id
                                        break
                                
                                print(f"  Serving Cell: {serving_cell} (Site {serving_cell_site_id})" if serving_cell_site_id is not None else f"  Serving Cell: {serving_cell}")
                                if serving_cell_location:
                                    print(f"  Serving Cell Location: ({serving_cell_location['x']:.1f}, {serving_cell_location['y']:.1f}, {serving_cell_location['z']:.1f}) m")
                                if ue_location:
                                    print(f"  UE Location: ({ue_location['x']:.1f}, {ue_location['y']:.1f}, {ue_location['z']:.1f}) m")
                                print(f"  Distance: d_2d = {link.d2d:.1f} m, d_3d = {link.d3d:.1f} m")
                                print(f"  LOS State: {'LOS' if link.los_ind else 'NLOS'}")
                                print(f"  K-factor: {k_linear:.6e} linear ({k_db:.2f} dB)")
                                print(f"  Shadow Fading: {link.sf:.2f} dB")
                                print(f"  Pathloss: {link.pathloss:.2f} dB")
                                print(f"")
                                print(f"  === CHANNEL GAINS ===")
                                print(f"  CIR Gain (includes antenna array + fading): {cir_gain_db:.2f} dB")
                                print(f"  Pathloss Gain (large-scale only):          {pathloss_gain_db:.2f} dB")
                                print(f"  Antenna Array + Fading Contribution:        {antenna_contribution:.2f} dB")
                                print(f"")
                                print(f"  === RECEIVED POWERS ===")
                                print(f"  TX Power:           {tx_power_dbm:.2f} dBm")
                                print(f"  Signal RX Power:    {signal_pwr_dbm:.2f} dBm  (TX + CIR gain)")
                                print(f"  Interference Power: {interf_pwr_dbm:.2f} dBm")
                                print(f"  Noise Power:        {noise_power_dbm:.2f} dBm")
                                print(f"")
                                print(f"  === SIR/SINR ===")
                                print(f"  SIR  = {sir_db[ue_idx]:.2f} dB  (Signal / Interference, no noise)")
                                print(f"  SINR = {sinr_val:.2f} dB  (Signal / (Interference + Noise))")
                                
                                # Calculate noise contribution to total interference+noise
                                interf_linear = 10**(interf_pwr_dbm / 10)
                                noise_linear = 10**(noise_power_dbm / 10)
                                total_in = interf_linear + noise_linear
                                noise_contribution_pct = (noise_linear / total_in) * 100
                                interf_contribution_pct = (interf_linear / total_in) * 100
                                
                                print(f"")
                                print(f"  === INTERFERENCE + NOISE BREAKDOWN ===")
                                print(f"  Interference contribution: {interf_contribution_pct:.1f}%")
                                print(f"  Noise contribution:        {noise_contribution_pct:.1f}%")
                                
                                # Detailed interference breakdown from each cell with co-sited analysis
                                if ue_idx in interference_breakdown:
                                    interf_list = interference_breakdown[ue_idx]
                                    
                                    # Calculate co-sited vs other-site interference
                                    co_sited_interf_power_linear = 0.0
                                    other_site_interf_power_linear = 0.0
                                    
                                    for cell_idx, rx_pwr, cir_db in interf_list:
                                        # Get interfering cell site
                                        interf_site_id = None
                                        for cell in self.cell_params:
                                            if cell.cid == cell_idx:
                                                interf_site_id = cell.site_id
                                                break
                                        
                                        rx_pwr_linear = 10**(rx_pwr / 10)
                                        if interf_site_id == serving_cell_site_id:
                                            co_sited_interf_power_linear += rx_pwr_linear
                                        else:
                                            other_site_interf_power_linear += rx_pwr_linear
                                    
                                    total_interf_linear = co_sited_interf_power_linear + other_site_interf_power_linear
                                    co_sited_pct = (co_sited_interf_power_linear / total_interf_linear * 100) if total_interf_linear > 0 else 0
                                    other_site_pct = (other_site_interf_power_linear / total_interf_linear * 100) if total_interf_linear > 0 else 0
                                    
                                    print(f"")
                                    print(f"  === CO-SITED vs OTHER-SITE INTERFERENCE ===")
                                    if co_sited_interf_power_linear > 0:
                                        co_sited_dbm = 10 * np.log10(co_sited_interf_power_linear)
                                        print(f"  Co-sited interference (Site {serving_cell_site_id}): {co_sited_dbm:.2f} dBm ({co_sited_pct:.1f}%)")
                                    else:
                                        print(f"  Co-sited interference: None")
                                    
                                    if other_site_interf_power_linear > 0:
                                        other_site_dbm = 10 * np.log10(other_site_interf_power_linear)
                                        print(f"  Other-site interference: {other_site_dbm:.2f} dBm ({other_site_pct:.1f}%)")
                                    else:
                                        print(f"  Other-site interference: None")
                                    
                                    print(f"")
                                    print(f"  === TOP INTERFERING CELLS ===")
                                    print(f"  {'Cell':<8} {'Site':<6} {'Type':<10} {'RX Power':<12} {'CIR Gain':<12} {'Distance':<12} {'Rel to Signal':<15}")
                                    print(f"  {'-'*95}")
                                    for i, (cell_idx, rx_pwr, cir_db) in enumerate(interf_list[:5]):  # Top 5 interferers
                                        rel_db = rx_pwr - signal_pwr_dbm
                                        # Find distance to interfering cell and its location
                                        interf_dist = None
                                        interf_cell_location = None
                                        interf_site_id = None
                                        for al in self.active_link_params:
                                            if al.uid == ue_idx and al.cid == cell_idx:
                                                if al.lsp_read_idx < len(self.link_params):
                                                    interf_dist = self.link_params[al.lsp_read_idx].d2d
                                                    break
                                        # Get interfering cell location
                                        for cell in self.cell_params:
                                            if cell.cid == cell_idx:
                                                interf_cell_location = cell.location
                                                interf_site_id = cell.site_id
                                                break
                                        dist_str = f"{interf_dist:.1f} m" if interf_dist is not None else "N/A"
                                        site_str = f"{interf_site_id}" if interf_site_id is not None else "N/A"
                                        interf_type = "CO-SITED" if interf_site_id == serving_cell_site_id else "Other"
                                        print(f"  {cell_idx:<8} {site_str:<6} {interf_type:<10} {rx_pwr:<12.2f} {cir_db:<12.2f} {dist_str:<12} {rel_db:>+6.2f} dB")
                                        if interf_cell_location:
                                            print(f"    Location: ({interf_cell_location['x']:.1f}, {interf_cell_location['y']:.1f}, {interf_cell_location['z']:.1f}) m")
                                
                                break
            
            print(f"\n{'='*100}")
            print(f"SUMMARY: {len(close_ues)} UEs within 50m of serving cell")
            if len(close_ues) > 0:
                distances = [d for _, d, _ in close_ues]
                sinrs = [s for _, _, s in close_ues if np.isfinite(s)]
                print(f"  Distance Range: [{np.min(distances):.1f}, {np.max(distances):.1f}] m")
                print(f"  Distance Mean:  {np.mean(distances):.1f} m")
                print(f"  Distance Median: {np.median(distances):.1f} m")
                if len(sinrs) > 0:
                    print(f"")
                    print(f"  SINR Range: [{np.min(sinrs):.2f}, {np.max(sinrs):.2f}] dB")
                    print(f"  SINR Mean:  {np.mean(sinrs):.2f} dB")
                    print(f"  SINR Median: {np.median(sinrs):.2f} dB")
                
                # Check LOS percentage and K-factor for close UEs
                los_count = 0
                k_factors_close = []
                for ue_idx, _, _ in close_ues:
                    if ue_idx < len(serving_cells):
                        serving_cell = serving_cells[ue_idx]
                        for active_link in self.active_link_params:
                            if active_link.uid == ue_idx and active_link.cid == serving_cell:
                                if active_link.lsp_read_idx < len(self.link_params):
                                    link = self.link_params[active_link.lsp_read_idx]
                                    if link.los_ind:
                                        los_count += 1
                                    k_factors_close.append(link.k_factor)
                                    break
                
                if len(k_factors_close) > 0:
                    print(f"")
                    print(f"  LOS percentage: {100*los_count/len(close_ues):.1f}% ({los_count}/{len(close_ues)})")
                    print(f"  K-factor: Mean = {np.mean(k_factors_close):.2f} dB, Max = {np.max(k_factors_close):.2f} dB, Min = {np.min(k_factors_close):.2f} dB")
                    
                    print(f"")
                    print(f"  === ANALYSIS: Why close UEs might not have very high SINR ===")
                    print(f"  1. **CO-SITED INTERFERENCE** - Other sectors at same site are equally close")
                    print(f"  2. Interference from neighboring cells (see 'CO-SITED vs OTHER-SITE' above)")
                    print(f"  3. Interfering cells may also be close, reducing SIR")
                    print(f"  4. Shadow fading variations (negative SF reduces signal)")
                    print(f"  5. K-factor might be low even for close LOS links")
                    print(f"  6. Small-scale fading effects captured in CIR")
            print(f"{'='*100}\n")
        else:
            print(f"\n{'='*100}")
            print(f"NO UEs within 50m of serving cell found")
            print(f"{'='*100}\n")
        
        # Filter out invalid values
        valid_sir = sir_db[np.isfinite(sir_db)]
        valid_sinr = sinr_db[np.isfinite(sinr_db)]
        
        return {
            'SIR': valid_sir,
            'SINR': valid_sinr,
            'serving_cells': serving_cells,
            'noise_power_dbm': noise_power_dbm
        }

    def calculate_received_power_from_cir(self, cir_coe_per_cell, cir_n_taps_per_cell, active_cells, active_uts, n_total_uts, scenario, center_freq_hz):
        """
        Calculate received power matrix from CIR coefficients including transmit power.
        
        Args:
            cir_coe_per_cell: List of CIR coefficient arrays per cell
            cir_n_taps_per_cell: List of number of taps arrays per cell
            active_cells: List of active cell indices
            active_uts: List of active UE indices per cell
            n_total_uts: Total number of UEs
            scenario: Deployment scenario (UMa, UMi, etc.)
            center_freq_hz: Center frequency in Hz
            
        Returns:
            power_matrix: Array of shape (n_cells, n_total_uts) with received power per (cell, UE) in dBm
        """
        logger.info("Calculating received power from CIR coefficients...")
        
        # Get system parameters for TX power based on scenario and frequency
        sys_params = self.get_system_parameters(scenario, center_freq_hz)
        tx_power_dbm = sys_params['tx_power_dbm']
        
        logger.info(f"TX Power: {tx_power_dbm} dBm (Scenario: {scenario}, Freq: {center_freq_hz/1e9:.1f} GHz)")
        
        n_cells = len(active_cells)
        
        # Initialize power matrix for all possible UEs
        power_matrix = np.full((n_cells, n_total_uts), -np.inf, dtype=np.float32)  # Use -inf for missing links
        
        for cell_idx, cell_id in enumerate(active_cells):
            if cell_idx >= len(cir_coe_per_cell) or cell_idx >= len(cir_n_taps_per_cell):
                logger.warning(f"No CIR data for cell {cell_id} (index {cell_idx})")
                continue
                
            cir_coe = cir_coe_per_cell[cell_idx]  # Shape: (n_active_uts, n_snapshots, n_ut_ant, n_bs_ant, max_taps)
            cir_n_taps = cir_n_taps_per_cell[cell_idx]  # Shape: (n_active_uts,)
            
            cell_active_uts = active_uts[cell_idx]
            if len(cell_active_uts) == 0:
                continue
                
            # Convert to numpy if needed
            if hasattr(cir_coe, 'get'):  # CuPy array
                cir_coe_np = cir_coe.get()
                cir_n_taps_np = cir_n_taps.get()
            else:
                cir_coe_np = cir_coe
                cir_n_taps_np = cir_n_taps
                
            n_active_uts_this_cell = cir_coe_np.shape[0]
            
            for cir_ue_idx in range(n_active_uts_this_cell):
                if cir_ue_idx < len(cell_active_uts):
                    global_ue_idx = cell_active_uts[cir_ue_idx]  # Map to global UE index
                    n_taps = cir_n_taps_np[cir_ue_idx]
                    
                    if n_taps > 0 and global_ue_idx < n_total_uts:
                        # Calculate channel power: sum |cir_coe|^2 across all dimensions except UE
                        # Sum across: snapshots, UT antennas, BS antennas, taps (up to n_taps)
                        cir_channel_power_linear = np.sum(np.abs(cir_coe_np[cir_ue_idx, :, :, :, :n_taps])**2)
                        
                        # Convert channel power to dB
                        if cir_channel_power_linear > 0:
                            channel_gain_db = 10 * np.log10(cir_channel_power_linear)
                            # Calculate total received power = TX Power + Channel Gain
                            received_power_dbm = tx_power_dbm + channel_gain_db
                            power_matrix[cell_idx, global_ue_idx] = 10**(received_power_dbm / 10)  # Store as linear for later dBm conversion
                        else:
                            power_matrix[cell_idx, global_ue_idx] = -np.inf
                    
            logger.info(f"Cell {cell_id}: processed {n_active_uts_this_cell} active UEs")
        
        # Convert to dBm (power_matrix now contains linear power including TX power)
        # Replace -inf with very small value, then convert to dBm
        power_matrix_linear = np.where(power_matrix == -np.inf, 1e-20, power_matrix)
        power_matrix_dbm = 10 * np.log10(power_matrix_linear)
        
        # Mark invalid links with very low power
        power_matrix_dbm = np.where(power_matrix == -np.inf, -200.0, power_matrix_dbm)
        
        valid_powers = power_matrix_dbm[power_matrix_dbm > -200]
        logger.info(f"Power matrix shape: {power_matrix_dbm.shape}")
        logger.info(f"Valid links: {len(valid_powers)} out of {power_matrix_dbm.size}")
        if len(valid_powers) > 0:
            logger.info(f"Power range: [{np.min(valid_powers):.1f}, {np.max(valid_powers):.1f}] dBm")
        
        return power_matrix_dbm

    def plot_pathloss_cdf(self, pathloss_values, scenario_name="UMa", save_path=None):
        """
        Plot Cumulative Distribution Function (CDF) of pathloss values.
        
        Args:
            pathloss_values: Array of pathloss values in dB
            scenario_name: Name of the scenario for the plot title
            save_path: Optional path to save the plot
        """
        # Remove any NaN or invalid values
        valid_pathloss = pathloss_values[~np.isnan(pathloss_values)]
        valid_pathloss = valid_pathloss[np.isfinite(valid_pathloss)]
        
        if len(valid_pathloss) == 0:
            logger.warning("No valid pathloss values found for CDF plot")
            return
        
        # Sort the data for CDF calculation
        sorted_pathloss = np.sort(valid_pathloss)
        
        # Calculate CDF values (percentiles)
        cdf = np.arange(1, len(sorted_pathloss) + 1) / len(sorted_pathloss)
        
        # Create the plot
        plt.figure(figsize=(10, 6))
        plt.plot(sorted_pathloss, cdf * 100, 'b-', linewidth=2, label=f'{scenario_name} Scenario')
        
        # Add percentile markers (5th, 50th, 95th percentiles)
        percentiles = [5, 50, 95]
        percentile_values = np.percentile(valid_pathloss, percentiles)
        
        for i, (p, val) in enumerate(zip(percentiles, percentile_values)):
            plt.axvline(x=val, color='red', linestyle='--', alpha=0.7)
            plt.axhline(y=p, color='red', linestyle='--', alpha=0.7)
            plt.plot(val, p, 'ro', markersize=8)
            plt.text(val + 2, p + 5, f'P{p}: {val:.1f} dB', 
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Formatting
        plt.xlabel('Pathloss (dB)', fontsize=12)
        plt.ylabel('CDF (%)', fontsize=12)
        plt.title(f'Pathloss CDF - {scenario_name} Scenario\n'
                  f'Mean: {np.mean(valid_pathloss):.1f} dB, '
                  f'Std: {np.std(valid_pathloss):.1f} dB, '
                  f'Links: {len(valid_pathloss)}', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Set reasonable axis limits
        plt.xlim([np.min(valid_pathloss) - 5, np.max(valid_pathloss) + 5])
        plt.ylim([0, 100])
        
        # Add statistical info
        stats_text = (f'Min: {np.min(valid_pathloss):.1f} dB\n'
                      f'Max: {np.max(valid_pathloss):.1f} dB\n'
                      f'Range: {np.max(valid_pathloss) - np.min(valid_pathloss):.1f} dB')
        plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        plt.tight_layout()
        
        # Save the plot
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Pathloss CDF plot saved to: {save_path}")
        else:
            # Save with default name
            default_path = f"pathloss_cdf_{scenario_name.lower()}.png"
            plt.savefig(default_path, dpi=300, bbox_inches='tight')
            logger.info(f"Pathloss CDF plot saved to: {default_path}")
        
        plt.close()  # Close the figure to free memory

    def plot_pathloss_comparison_cdf(self, all_links_pathloss, serving_cells_pathloss, scenario_name="UMa", save_path=None):
        """
        Plot comparison CDF of pathloss values for all links vs serving cells only.
        
        Args:
            all_links_pathloss: Array of pathloss values for all links in dB
            serving_cells_pathloss: Array of pathloss values for serving cells only in dB
            scenario_name: Name of the scenario for the plot title
            save_path: Optional path to save the plot
        """
        # Remove any NaN or invalid values
        valid_all = all_links_pathloss[~np.isnan(all_links_pathloss)]
        valid_all = valid_all[np.isfinite(valid_all)]
        
        valid_serving = serving_cells_pathloss[~np.isnan(serving_cells_pathloss)]
        valid_serving = valid_serving[np.isfinite(valid_serving)]
        
        if len(valid_all) == 0 and len(valid_serving) == 0:
            logger.warning("No valid pathloss values found for comparison CDF plot")
            return
        
        # Create the plot
        plt.figure(figsize=(12, 8))
        
        # Plot All Links CDF
        if len(valid_all) > 0:
            sorted_all = np.sort(valid_all)
            cdf_all = np.arange(1, len(sorted_all) + 1) / len(sorted_all)
            plt.plot(sorted_all, cdf_all * 100, 'b-', linewidth=2, label=f'All Links (N={len(valid_all)})')
            
            # Add percentile markers for all links
            percentiles = [5, 50, 95]
            percentile_values_all = np.percentile(valid_all, percentiles)
            for i, (p, val) in enumerate(zip(percentiles, percentile_values_all)):
                plt.axvline(x=val, color='blue', linestyle='--', alpha=0.5)
                plt.plot(val, p, 'bo', markersize=6)
                plt.text(val + 1, p + 2, f'P{p}: {val:.1f}', color='blue',
                        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
        
        # Plot Serving Cells CDF
        if len(valid_serving) > 0:
            sorted_serving = np.sort(valid_serving)
            cdf_serving = np.arange(1, len(sorted_serving) + 1) / len(sorted_serving)
            plt.plot(sorted_serving, cdf_serving * 100, 'r-', linewidth=2, label=f'Serving Cells Only (N={len(valid_serving)})')
            
            # Add percentile markers for serving cells
            percentiles = [5, 50, 95]
            percentile_values_serving = np.percentile(valid_serving, percentiles)
            for i, (p, val) in enumerate(zip(percentiles, percentile_values_serving)):
                plt.axvline(x=val, color='red', linestyle='--', alpha=0.5)
                plt.plot(val, p, 'ro', markersize=6)
                plt.text(val + 1, p - 5, f'P{p}: {val:.1f}', color='red',
                        bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))
        
        # Formatting
        plt.xlabel('Coupling Loss (dB)', fontsize=12)
        plt.ylabel('CDF (%)', fontsize=12)
        plt.title(f'Pathloss CDF Comparison - {scenario_name} Scenario\\n'
                  f'All Links vs Serving Cells (CIR-based)', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=11)
        
        # Set reasonable axis limits
        all_values = np.concatenate([valid_all, valid_serving]) if len(valid_all) > 0 and len(valid_serving) > 0 else (valid_all if len(valid_all) > 0 else valid_serving)
        if len(all_values) > 0:
            plt.xlim([np.min(all_values) - 5, np.max(all_values) + 5])
        plt.ylim([0, 100])
        
        # Add statistical comparison
        if len(valid_all) > 0 and len(valid_serving) > 0:
            stats_text = (f'All Links:\\n'
                          f'  Mean: {np.mean(valid_all):.1f} dB\\n'
                          f'  Std: {np.std(valid_all):.1f} dB\\n'
                          f'\\n'
                          f'Serving Cells:\\n'
                          f'  Mean: {np.mean(valid_serving):.1f} dB\\n'
                          f'  Std: {np.std(valid_serving):.1f} dB\\n'
                          f'\\n'
                          f'Difference (Serving - All):\\n'
                          f'  Mean: {np.mean(valid_serving) - np.mean(valid_all):.1f} dB')
            plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
                     verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        
        plt.tight_layout()
        
        # Save the plot
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Pathloss comparison CDF plot saved to: {save_path}")
        else:
            # Save with default name
            default_path = f"pathloss_cdf_comparison_{scenario_name.lower()}.png"
            plt.savefig(default_path, dpi=300, bbox_inches='tight')
            logger.info(f"Pathloss comparison CDF plot saved to: {default_path}")
        
        plt.close()  # Close the figure to free memory

    def calculate_and_plot_advanced_geometry_sir_sinr(self, cir_coe_per_cell=None, cir_n_taps_per_cell=None, 
                                                     active_cells=None, active_uts=None, scenario_name="UMa", 
                                                     center_freq_hz=None, n_total_uts=None, output_dir="."):
        """
        Calculate and plot advanced geometry SIR and SINR based on actual CIR power and LOS geometry.
        This is an enhanced version that works with both H5 data and simulation data.
        
        Args:
            cir_coe_per_cell: List of CIR coefficient arrays per cell (optional, uses H5 data if None)
            cir_n_taps_per_cell: List of number of taps arrays per cell (optional)
            active_cells: List of active cell indices (optional)
            active_uts: List of active UE indices per cell (optional)
            scenario_name: Scenario name for plot titles
            center_freq_hz: Center frequency in Hz (optional, uses SimConfig if available)
            n_total_uts: Total number of UEs (optional, uses SystemConfig if available)
            output_dir: Output directory for plots
        """
        logger.info(f"Calculating advanced geometry SIR/SINR for {scenario_name}...")
        
        # Use center frequency from SimConfig if available, otherwise use provided parameter or default
        if center_freq_hz is None:
            if self.sim_config is not None:
                center_freq_hz = self.sim_config.center_freq_hz
                logger.info(f"Using center frequency from SimConfig: {center_freq_hz/1e9:.1f} GHz")
            else:
                center_freq_hz = 6e9  # Default fallback
                logger.warning("No SimConfig found, using default center frequency: 6 GHz")
        
        # Use n_total_uts from SystemConfig if available
        if n_total_uts is None:
            if self.system_level_config is not None:
                n_total_uts = self.system_level_config.n_ut
                logger.info(f"Using n_total_uts from SystemConfig: {n_total_uts}")
        else:
                n_total_uts = len(set(link.uid for link in self.active_link_params)) if self.active_link_params else 10
                logger.warning(f"No SystemConfig found, estimating n_total_uts: {n_total_uts}")
        
        # Get system parameters
        sys_params = self.get_system_parameters(scenario_name, center_freq_hz)
        tx_power_dbm = sys_params['tx_power_dbm']
        bandwidth_hz = sys_params['bandwidth_hz']
        noise_figure_db = sys_params['noise_figure_db']
        
        logger.info(f"System parameters:")
        logger.info(f"  TX Power: {tx_power_dbm} dBm")
        logger.info(f"  Bandwidth: {bandwidth_hz/1e6:.0f} MHz")
        logger.info(f"  Noise Figure: {noise_figure_db} dB")
        
        # Calculate thermal noise power
        noise_power_dbm = self.calculate_thermal_noise_power(bandwidth_hz, noise_figure_db)
        logger.info(f"  Thermal Noise Power: {noise_power_dbm:.1f} dBm")
        
        # Calculate received power matrix
        if cir_coe_per_cell is not None and cir_n_taps_per_cell is not None:
            # Use provided CIR data (from simulation)
            rx_power_dbm = self.calculate_received_power_from_cir(
                cir_coe_per_cell, cir_n_taps_per_cell, active_cells, active_uts, 
                n_total_uts, scenario_name, center_freq_hz
            )
        else:
            # Use H5 data if available
            if self.link_params and self.active_link_params:
                n_cells = len(self.cell_params)
                rx_power_dbm = np.full((n_cells, n_total_uts), -200.0, dtype=np.float32)
                
                for active_link in self.active_link_params:
                    if active_link.lsp_read_idx < len(self.link_params):
                        link = self.link_params[active_link.lsp_read_idx]
                        # Calculate received power = TX power - (path loss - shadow fading)
                        rx_power = tx_power_dbm - (link.pathloss - link.sf)
                        rx_power_dbm[active_link.cid, active_link.uid] = rx_power
            else:
                logger.error("No CIR data or H5 link parameters available for power calculation")
                return
        
        logger.info(f"Received power matrix shape: {rx_power_dbm.shape}")
        valid_powers = rx_power_dbm[rx_power_dbm > -200]
        if len(valid_powers) > 0:
            logger.info(f"Received power range: [{np.min(valid_powers):.1f}, {np.max(valid_powers):.1f}] dBm")
        
        # Calculate SIR and SINR for each UE using geometry-based serving cell assignment
        n_cells, n_uts = rx_power_dbm.shape
        sir_db = np.zeros(n_uts)
        sinr_db = np.zeros(n_uts)
        
        # Use serving cell assignments from CIR power if available, otherwise use power matrix
        if self.cir_per_cell and not cir_coe_per_cell:
            # H5 data mode - use CIR-based serving cell assignments
            serving_assignments = self.get_serving_cell_assignments()
            serving_cells = np.zeros(n_uts, dtype=int)
            for ue_idx in range(n_uts):
                if ue_idx in serving_assignments:
                    serving_cells[ue_idx] = serving_assignments[ue_idx]['cell_id']
                else:
                    serving_cells[ue_idx] = 0  # Fallback
        else:
            # Simulation data mode - determine serving cell from power matrix
            serving_cells = np.zeros(n_uts, dtype=int)
            for ue_idx in range(n_uts):
                # Get received power from all cells for this UE
                ue_rx_power = rx_power_dbm[:, ue_idx]  # Shape: (n_cells,)
                valid_mask = ue_rx_power > -200  # Valid power threshold
                
                if np.any(valid_mask):
                    # Find serving cell with maximum received power
                    serving_cell_idx = np.argmax(ue_rx_power)
                    serving_cells[ue_idx] = serving_cell_idx
                else:
                    # Fallback: assign to cell 0 if no valid power data
                    serving_cells[ue_idx] = 0
        
        # Arrays to store power data for plotting
        serving_powers_dbm = np.zeros(n_uts)
        total_interference_powers_dbm = np.zeros(n_uts)
        
        for ue_idx in range(n_uts):
            # Get received powers from all cells for this UE
            rx_powers_ue = rx_power_dbm[:, ue_idx]  # Shape: (n_cells,)
            
            # Use geometry-determined serving cell
            serving_cell_idx = serving_cells[ue_idx]
            serving_power_dbm = rx_powers_ue[serving_cell_idx]
            
            # Calculate interference power (sum of all other cells)
            interfering_powers_dbm = np.delete(rx_powers_ue, serving_cell_idx)
            
            # Skip UEs with invalid serving power
            if serving_power_dbm <= -200:  # Invalid power
                sir_db[ue_idx] = np.nan
                sinr_db[ue_idx] = np.nan
                serving_powers_dbm[ue_idx] = np.nan
                total_interference_powers_dbm[ue_idx] = np.nan
                continue
            
            # Store serving power for plotting
            serving_powers_dbm[ue_idx] = serving_power_dbm
            
            # Convert to linear scale for summation
            serving_power_mw = 10**(serving_power_dbm / 10)
            
            # Calculate total interference
            interfering_powers_mw = []
            for interfering_power_dbm in interfering_powers_dbm:
                if interfering_power_dbm > -200:  # Valid power
                    interfering_power_mw = 10**(interfering_power_dbm / 10)
                    interfering_powers_mw.append(interfering_power_mw)
            
            if len(interfering_powers_mw) > 0:
                total_interference_mw = np.sum(interfering_powers_mw)
                total_interference_powers_dbm[ue_idx] = 10 * np.log10(total_interference_mw)
            else:
                total_interference_mw = 0
                total_interference_powers_dbm[ue_idx] = -200  # Very low for no interference
            
            # Calculate SIR (without noise)
            noise_power_mw = 10**(noise_power_dbm / 10)
            
            if total_interference_mw > 0:
                sir_linear = serving_power_mw / total_interference_mw
                sir_db[ue_idx] = 10 * np.log10(sir_linear)
            else:
                sir_db[ue_idx] = np.inf  # No interference
            
            # Calculate SINR (with noise)
            sinr_linear = serving_power_mw / (total_interference_mw + noise_power_mw)
            sinr_db[ue_idx] = 10 * np.log10(sinr_linear)
        
        # Filter out invalid values
        valid_sir = sir_db[np.isfinite(sir_db)]
        valid_sinr = sinr_db[np.isfinite(sinr_db)]
        
        logger.info(f"SIR statistics:")
        if len(valid_sir) > 0:
            logger.info(f"  Mean: {np.mean(valid_sir):.1f} dB")
            logger.info(f"  Std: {np.std(valid_sir):.1f} dB")
            logger.info(f"  Range: [{np.min(valid_sir):.1f}, {np.max(valid_sir):.1f}] dB")
        
        logger.info(f"SINR statistics:")
        if len(valid_sinr) > 0:
            logger.info(f"  Mean: {np.mean(valid_sinr):.1f} dB")
            logger.info(f"  Std: {np.std(valid_sinr):.1f} dB")
            logger.info(f"  Range: [{np.min(valid_sinr):.1f}, {np.max(valid_sinr):.1f}] dB")
        
        # Plot SIR and SINR
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Combined SIR/SINR plot
        plt.figure(figsize=(15, 10))
        
        # Plot 1: SIR Histogram
        plt.subplot(2, 3, 1)
        if len(valid_sir) > 0:
            plt.hist(valid_sir, bins=50, alpha=0.7, color='blue', edgecolor='black')
        plt.xlabel('SIR (dB)')
        plt.ylabel('Number of UEs')
        plt.title(f'SIR Distribution - {scenario_name}')
        plt.grid(True, alpha=0.3)
        
        # Plot 2: SINR Histogram
        plt.subplot(2, 3, 2)
        if len(valid_sinr) > 0:
            plt.hist(valid_sinr, bins=50, alpha=0.7, color='red', edgecolor='black')
        plt.xlabel('SINR (dB)')
        plt.ylabel('Number of UEs')
        plt.title(f'SINR Distribution - {scenario_name}')
        plt.grid(True, alpha=0.3)
        
        # Plot 3: SIR CDF
        plt.subplot(2, 3, 3)
        if len(valid_sir) > 0:
            sorted_sir = np.sort(valid_sir)
            cdf_sir = np.arange(1, len(sorted_sir) + 1) / len(sorted_sir)
            plt.plot(sorted_sir, cdf_sir * 100, 'b-', linewidth=2, label='SIR')
        plt.xlabel('SIR (dB)')
        plt.ylabel('CDF (%)')
        plt.title(f'SIR CDF - {scenario_name}')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Plot 4: SINR CDF
        plt.subplot(2, 3, 4)
        if len(valid_sinr) > 0:
            sorted_sinr = np.sort(valid_sinr)
            cdf_sinr = np.arange(1, len(sorted_sinr) + 1) / len(sorted_sinr)
            plt.plot(sorted_sinr, cdf_sinr * 100, 'r-', linewidth=2, label='SINR')
        plt.xlabel('SINR (dB)')
        plt.ylabel('CDF (%)')
        plt.title(f'SINR CDF - {scenario_name}')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Plot 5: SIR vs SINR comparison
        plt.subplot(2, 3, 5)
        if len(valid_sir) > 0 and len(valid_sinr) > 0:
            sorted_sir = np.sort(valid_sir)
            cdf_sir = np.arange(1, len(sorted_sir) + 1) / len(sorted_sir)
            sorted_sinr = np.sort(valid_sinr)
            cdf_sinr = np.arange(1, len(sorted_sinr) + 1) / len(sorted_sinr)
            plt.plot(sorted_sir, cdf_sir * 100, 'b-', linewidth=2, label='SIR')
            plt.plot(sorted_sinr, cdf_sinr * 100, 'r-', linewidth=2, label='SINR')
        plt.xlabel('Signal Ratio (dB)')
        plt.ylabel('CDF (%)')
        plt.title(f'SIR vs SINR Comparison - {scenario_name}')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Plot 6: Box plot comparison
        plt.subplot(2, 3, 6)
        data_to_plot = []
        labels = []
        if len(valid_sir) > 0:
            data_to_plot.append(valid_sir)
            labels.append('SIR')
        if len(valid_sinr) > 0:
            data_to_plot.append(valid_sinr)
            labels.append('SINR')
        if data_to_plot:
            plt.boxplot(data_to_plot, tick_labels=labels)
        plt.ylabel('Signal Ratio (dB)')
        plt.title(f'SIR vs SINR Box Plot - {scenario_name}')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save the plot
        sir_sinr_plot_path = output_path / f"sir_sinr_geometry_{scenario_name.lower()}_{center_freq_hz/1e9:.0f}ghz.png"
        plt.savefig(sir_sinr_plot_path, dpi=300, bbox_inches='tight')
        logger.info(f"SIR/SINR geometry plot saved to: {sir_sinr_plot_path}")
        
        plt.close()
        
        return {
            'SIR': valid_sir,
            'SINR': valid_sinr,
            'serving_cells': serving_cells,
            'noise_power_dbm': noise_power_dbm,
            'serving_powers_dbm': serving_powers_dbm,
            'total_interference_powers_dbm': total_interference_powers_dbm
        }

    def analyze_sf_and_los_statistics(self) -> Dict[str, Any]:
        """Analyze Shadow Fading (SF) and LOS/NLOS statistics from link parameters
        
        Returns:
            Dictionary containing SF and LOS/NLOS statistics
        """
        if not self.link_params:
            print("No link parameters available for SF/LOS analysis")
            return {}
        
        # Get serving cell assignments for filtering
        serving_assignments = self.get_serving_cell_assignments()
        serving_link_indices = set()
        for ue_idx, assignment in serving_assignments.items():
            for active_link in self.active_link_params:
                if active_link.uid == ue_idx and active_link.cid == assignment['cell_id']:
                    serving_link_indices.add(active_link.lsp_read_idx)
                    break
        
        # Extract SF and LOS data from all active links
        sf_values_all = []
        sf_values_serving = []
        los_counts_all = {'LOS': 0, 'NLOS': 0}
        los_counts_serving = {'LOS': 0, 'NLOS': 0}
        
        # For distance-based LOS probability
        distance_bins = [0, 50, 100, 200, 300, 500, 750, 1000, 1500, 2000, 5000]
        los_by_distance = {i: {'los': 0, 'total': 0} for i in range(len(distance_bins)-1)}
        
        for active_link in self.active_link_params:
            if active_link.lsp_read_idx < len(self.link_params):
                link = self.link_params[active_link.lsp_read_idx]
                is_serving = active_link.lsp_read_idx in serving_link_indices
                
                # SF statistics
                sf_values_all.append(link.sf)
                if is_serving:
                    sf_values_serving.append(link.sf)
                
                # LOS/NLOS counts
                if link.los_ind == 1:
                    los_counts_all['LOS'] += 1
                    if is_serving:
                        los_counts_serving['LOS'] += 1
                else:
                    los_counts_all['NLOS'] += 1
                    if is_serving:
                        los_counts_serving['NLOS'] += 1
                
                # Distance-based LOS probability
                d_2d = link.d2d
                for i in range(len(distance_bins)-1):
                    if distance_bins[i] <= d_2d < distance_bins[i+1]:
                        los_by_distance[i]['total'] += 1
                        if link.los_ind == 1:
                            los_by_distance[i]['los'] += 1
                        break
        
        sf_array_all = np.array(sf_values_all)
        sf_array_serving = np.array(sf_values_serving) if sf_values_serving else np.array([])
        total_links_all = len(sf_values_all)
        total_links_serving = len(sf_values_serving)
        
        results = {
            'sf_mean_all': np.mean(sf_array_all),
            'sf_std_all': np.std(sf_array_all),
            'sf_min_all': np.min(sf_array_all),
            'sf_max_all': np.max(sf_array_all),
            'sf_median_all': np.median(sf_array_all),
            'los_count_all': los_counts_all['LOS'],
            'nlos_count_all': los_counts_all['NLOS'],
            'los_percentage_all': (los_counts_all['LOS'] / total_links_all * 100) if total_links_all > 0 else 0,
            'total_links_all': total_links_all,
            'sf_mean_serving': np.mean(sf_array_serving) if len(sf_array_serving) > 0 else 0,
            'sf_std_serving': np.std(sf_array_serving) if len(sf_array_serving) > 0 else 0,
            'los_count_serving': los_counts_serving['LOS'],
            'nlos_count_serving': los_counts_serving['NLOS'],
            'los_percentage_serving': (los_counts_serving['LOS'] / total_links_serving * 100) if total_links_serving > 0 else 0,
            'total_links_serving': total_links_serving,
            'los_by_distance': los_by_distance,
            'distance_bins': distance_bins
        }
        
        print(f"\n{'='*80}")
        print(f"SHADOW FADING (SF) AND LOS/NLOS STATISTICS")
        print(f"{'='*80}")
        print(f"Shadow Fading (dB):")
        print(f"  ALL LINKS:")
        print(f"    Mean:     {results['sf_mean_all']:>8.2f} dB")
        print(f"    Std Dev:  {results['sf_std_all']:>8.2f} dB")
        print(f"    Median:   {results['sf_median_all']:>8.2f} dB")
        print(f"    Range:    [{results['sf_min_all']:>6.2f}, {results['sf_max_all']:>6.2f}] dB")
        if total_links_serving > 0:
            print(f"  SERVING LINKS ONLY:")
            print(f"    Mean:     {results['sf_mean_serving']:>8.2f} dB")
            print(f"    Std Dev:  {results['sf_std_serving']:>8.2f} dB")
        
        print(f"\nLOS/NLOS Distribution (ALL {total_links_all} active links):")
        print(f"  LOS:      {results['los_count_all']:>5} links ({results['los_percentage_all']:>5.1f}%)")
        print(f"  NLOS:     {results['nlos_count_all']:>5} links ({100-results['los_percentage_all']:>5.1f}%)")
        
        if total_links_serving > 0:
            print(f"\nLOS/NLOS Distribution (SERVING LINKS ONLY - {total_links_serving} links):")
            print(f"  LOS:      {results['los_count_serving']:>5} links ({results['los_percentage_serving']:>5.1f}%)")
            print(f"  NLOS:     {results['nlos_count_serving']:>5} links ({100-results['los_percentage_serving']:>5.1f}%)")
        
        print(f"\nLOS Probability by Distance:")
        print(f"  {'Distance (m)':<20} {'LOS Links':<12} {'Total Links':<12} {'LOS %':<10} {'3GPP Theory'}")
        print(f"  {'-'*78}")
        
        # Calculate 3GPP theoretical LOS probability for comparison
        scenario = self.system_level_config.scenario if self.system_level_config else 'UMa'
        for i in range(len(distance_bins)-1):
            d_min = distance_bins[i]
            d_max = distance_bins[i+1]
            d_mid = (d_min + d_max) / 2.0
            
            stats = los_by_distance[i]
            if stats['total'] > 0:
                los_pct = (stats['los'] / stats['total']) * 100
                
                # Calculate 3GPP theoretical LOS probability at midpoint
                if scenario == 'UMa':
                    if d_mid <= 18.0:
                        p_los_theory = 100.0
                    else:
                        p_los_theory = ((18.0 / d_mid) + np.exp(-d_mid / 63.0) * (1.0 - 18.0 / d_mid)) * 100.0
                elif scenario == 'UMi':
                    if d_mid <= 18.0:
                        p_los_theory = 100.0
                    else:
                        p_los_theory = ((18.0 / d_mid) + np.exp(-d_mid / 36.0) * (1.0 - 18.0 / d_mid)) * 100.0
                elif scenario == 'RMa':
                    if d_mid <= 10.0:
                        p_los_theory = 100.0
                    else:
                        p_los_theory = np.exp(-(d_mid - 10.0) / 1000.0) * 100.0
                else:
                    p_los_theory = 0.0
                
                print(f"  [{d_min:>5.0f}, {d_max:>5.0f})    {stats['los']:>5}/{stats['total']:<5}    "
                      f"{los_pct:>6.1f}%      {p_los_theory:>5.1f}%")
            else:
                print(f"  [{d_min:>5.0f}, {d_max:>5.0f})    No links in this range")
        
        print(f"{'='*80}\n")
        
        return results

    def generate_all_plots(self, output_dir: str = "channel_analysis_plots", association_method: str = 'rsrp',
                          use_virtualization: bool = True) -> None:
        """Generate all requested plots and analysis
        
        Args:
            output_dir: Directory to save plots and results
            association_method: Association method to use:
                - 'rsrp': RSRP-based (pathloss - shadow fading) [default]
                - 'cir': CIR power-based (legacy)
                - 'distance': Minimal distance-based
            use_virtualization: If True, use antenna virtualization per ITU-R M.2101 (default: True)
        """
        logger.info(f"Generating all analysis plots using {association_method} cell association")
        logger.info(f"Antenna virtualization: {'ENABLED' if use_virtualization else 'DISABLED'} (Phase {self.calibration_phase})")
        
        # Print cell association method information
        print(f"\n{'='*60}")
        print(f"CELL ASSOCIATION METHOD: {association_method.upper()}")
        print(f"{'='*60}")
        if association_method == 'rsrp':
            print("Using RSRP-based association (TX power - (pathloss - shadow fading))")
            print("This is the standard 3GPP cell association method.")
        elif association_method == 'cir':
            print("Using CIR power-based association from H5 file")
            print("Serving cells determined by maximum CIR power.")
        elif association_method == 'distance':
            print("Using distance-based association (minimal 3D distance)")
            print("Serving cells determined by closest distance.")
        print(f"{'='*60}\n")
        
        # Use center frequency from SimConfig - this is required
        if self.sim_config is not None:
            center_freq_hz = self.sim_config.center_freq_hz
            logger.info(f"Using center frequency from SimConfig: {center_freq_hz/1e9:.1f} GHz")
        else:
            raise ValueError("No SimConfig found in H5 file - center frequency is required for analysis")
        
        # Print configuration information if available
        if self.system_level_config is not None:
            print(f"\n=== SYSTEM LEVEL CONFIGURATION ===")
            print(f"Scenario: {self.system_level_config.scenario}")
            print(f"Number of sites: {self.system_level_config.n_site}")
            print(f"Sectors per site: {self.system_level_config.n_sector_per_site}")
            print(f"Number of UTs: {self.system_level_config.n_ut}")
            print(f"Inter-site distance: {self.system_level_config.isd:.1f} m")
            
        if self.sim_config is not None:
            print(f"\n=== SIMULATION CONFIGURATION ===")
            print(f"Center frequency: {self.sim_config.center_freq_hz/1e9:.1f} GHz")
            print(f"Bandwidth: {self.sim_config.bandwidth_hz/1e6:.0f} MHz")
            print(f"FFT size: {self.sim_config.fft_size}")
            print(f"Number of PRBs: {self.sim_config.n_prb}")
            print(f"Run mode: {self.sim_config.run_mode}")
        
        # Detect ISAC mode - affects what analyses are performed
        is_isac_mode = self.isac_config and self.isac_config.is_enabled
        if is_isac_mode:
            print(f"\n{'='*60}")
            print(f"ISAC SENSING MODE - LIMITED ANALYSIS")
            print(f"{'='*60}")
            print(f"  ISAC uses BS→ST→BS channels (not BS→UE)")
            if self.calibration_phase == 2:
                print(f"  Phase 2 {self.isac_channel_type} channel: Computing coupling loss, delay spread, angle spreads")
                print(f"  Skipping: SIR/SINR, K-factor, PRB singular values")
                print(f"  Note: Per 3GPP 38.901 Table 7.9.6.2-1, CDFs are generated separately for target/background channels")
            else:
                print(f"  Skipping: SIR/SINR, K-factor, PRB singular values, angle spreads")
                print(f"  Computing: Coupling loss only (primary ISAC calibration metric)")
            print(f"{'='*60}\n")
        
        # Analyze SF and LOS/NLOS statistics (skip for ISAC - no UE links)
        sf_los_stats = None
        if not is_isac_mode:
            sf_los_stats = self.analyze_sf_and_los_statistics()
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Print virtualization information
        print(f"\n{'='*60}")
        print(f"ANTENNA VIRTUALIZATION: {'ENABLED' if use_virtualization else 'DISABLED'}")
        print(f"{'='*60}")
        if use_virtualization:
            print(f"Using ITU-R M.2101 antenna virtualization for Phase {self.calibration_phase}")
            if self.calibration_phase == 1:
                print(f"Phase 1: Simplified beamforming with tilt=12°, numM=10")
            else:
                print(f"Phase 2: 16 elements/port grouping, angle sweep [-60°, 60°], 12 angles")
        else:
            print("Using simple antenna gain: 10*log10(N) where N = number of antennas")
        print(f"{'='*60}\n")
        
        # 1. Coupling loss for serving cell (single value)
        coupling_loss = self.compute_coupling_loss_serving_cell(association_method, use_virtualization)
        print(f"\n=== COUPLING LOSS - SERVING CELL ({association_method}) ===")
        print(f"Coupling Loss: {coupling_loss:.2f} dB")
        virt_label = f" (Phase {self.calibration_phase} virtualization)" if use_virtualization else " (no virtualization)"
        print(f"Antenna gain calculation: {virt_label}")
        
        # Get scenario key for reference data matching
        scenario_key = None
        if self.reference_data and self.system_level_config and self.sim_config:
            # For ISAC mode, use ISAC-specific scenario keys
            scenario_key = self.reference_data.get_scenario_key(
                self.system_level_config.scenario,
                self.sim_config.center_freq_hz / 1e9,
                isac_config=self.isac_config,
                isac_channel_type=self.isac_channel_type
            )
            if scenario_key:
                logger.info(f"Using 3GPP reference data for scenario: {scenario_key}")
                print(f"\n{'='*60}")
                if self.isac_config and self.isac_config.is_enabled:
                    print(f"3GPP ISAC REFERENCE DATA: ENABLED (Phase {self.calibration_phase})")
                    print(f"{'='*60}")
                    print(f"ISAC scenario key: {scenario_key}")
                    print(f"Target type: {self.isac_config.target_type_name}")
                    print(f"Sensing mode: {'Monostatic' if self.isac_config.is_monostatic else 'Bistatic'}")
                    print(f"Channel type: {self.isac_channel_type or 'auto-detected'}")
                else:
                    print(f"3GPP REFERENCE DATA: ENABLED (Phase {self.calibration_phase})")
                    print(f"{'='*60}")
                    print(f"Scenario key: {scenario_key}")
                if self.calibration_phase == 1:
                    print(f"Phase 1 metrics: coupling_loss")
                else:
                    print(f"Phase 2 metrics: coupling_loss, delay_spread, angle_spreads")
                print(f"Reference curves will be overlaid on relevant CDF plots.")
                print(f"{'='*60}\n")
            else:
                print(f"\n{'='*60}")
                print(f"3GPP REFERENCE DATA: Not available for this scenario/frequency")
                if self.isac_config and self.isac_config.is_enabled:
                    print(f"ISAC mode detected but no matching reference data found.")
                    print(f"Expected scenario key pattern: {self.isac_config.get_isac_scenario_key(self.sim_config.center_freq_hz / 1e9, self.isac_channel_type == 'background')}")
                print(f"{'='*60}\n")
        else:
            print(f"\n{'='*60}")
            print(f"3GPP REFERENCE DATA: Not loaded")
            print(f"Use --reference-json flag to enable reference comparisons")
            print(f"{'='*60}\n")
        
        # 1a. Coupling loss CDF for serving cells only (per 3GPP calibration methodology)
        serving_coupling_losses = self.compute_coupling_loss_serving_cells_only(use_virtualization)
        if len(serving_coupling_losses) > 0:
            print(f"Coupling Loss (Serving Cells): Mean = {np.mean(serving_coupling_losses):.2f} dB, Std = {np.std(serving_coupling_losses):.2f} dB")
            
            # Get reference data for coupling loss
            ref_data = None
            ref_envelope = None
            if scenario_key:
                ref_x, ref_cdf = self.reference_data.get_company_cdf(scenario_key, 'coupling_loss')
                if ref_x is not None:
                    ref_data = (ref_x, ref_cdf)
                    print(f"  -> 3GPP Reference: Mean = {np.mean(ref_x):.2f} dB, Std = {np.std(ref_x):.2f} dB")
                
                # Get min/max envelope
                ref_x_min, ref_x_max, ref_cdf_env = self.reference_data.get_company_cdf_envelope(scenario_key, 'coupling_loss')
                if ref_x_min is not None and ref_x_max is not None:
                    ref_envelope = (ref_x_min, ref_x_max, ref_cdf_env)
            
            # Create descriptive title with association method
            assoc_label = association_method.upper()
            title_suffix = f" ({assoc_label})" if ref_data else f" - {assoc_label}"
            ref_suffix = " vs 3GPP Ref" if ref_data else ""
            
            self.plot_cdf(
                serving_coupling_losses,
                f'CDF of Coupling Loss (Serving Cells){title_suffix}{ref_suffix}',
                'Coupling Loss (dB)',
                filename=output_path / f'cdf_coupling_loss_{association_method}.png',
                reference_data=ref_data,
                reference_envelope=ref_envelope,
                metric_name='Coupling Loss'
            )
        
        # 2. Wideband SIR (single value) - skip for ISAC (no UE links)
        serving_assignments = {}
        sir_sinr_results = {'SIR': [], 'SINR': [], 'noise_power_dbm': -92.0}
        
        if not is_isac_mode:
            sir_db = self.compute_wideband_sir()
            print(f"\n=== WIDEBAND SIR (Before Receiver, No Noise) ===")
            print(f"SIR: {sir_db:.2f} dB")
            
            # Compute serving assignments once for efficiency (reused by SIR/SINR and angle spread analysis)
            logger.info(f"Computing serving cell assignments using {association_method.upper()} method...")
            serving_assignments = self._compute_serving_assignments(association_method)
            logger.info(f"Serving assignments computed for {len(serving_assignments)} UEs")
            
            # 2a. SIR/SINR geometry analysis with CDFs
            print(f"\n=== SIR/SINR ({association_method.upper()}) ===")
            sir_sinr_results = self.compute_sir_sinr_geometry(center_freq_hz, association_method, serving_assignments)
        if not is_isac_mode and len(sir_sinr_results['SIR']) > 0:
            print(f"SIR: Mean = {np.mean(sir_sinr_results['SIR']):.2f} dB, Std = {np.std(sir_sinr_results['SIR']):.2f} dB")
            print(f"SINR: Mean = {np.mean(sir_sinr_results['SINR']):.2f} dB, Std = {np.std(sir_sinr_results['SINR']):.2f} dB")
            print(f"Noise Power: {sir_sinr_results['noise_power_dbm']:.1f} dBm")
            
            assoc_label = association_method.upper()
            
            # Get reference data for Phase 1 (geometry_sinr) or Phase 2 (wideband_sir)
            sir_ref_data = None
            sir_ref_envelope = None
            sinr_ref_data = None
            sinr_ref_envelope = None
            
            if scenario_key:
                if self.calibration_phase == 1:
                    # Phase 1: Use wideband_sir for SIR, geometry_sinr for SINR
                    # SIR reference from wideband_sir
                    ref_x_sir, ref_cdf_sir = self.reference_data.get_company_cdf(scenario_key, 'wideband_sir')
                    if ref_x_sir is not None:
                        sir_ref_data = (ref_x_sir, ref_cdf_sir)
                        print(f"  -> Phase 1 Wideband SIR Reference: Mean = {np.mean(ref_x_sir):.2f} dB, Std = {np.std(ref_x_sir):.2f} dB")
                    
                    # SIR envelope from wideband_sir
                    ref_x_min_sir, ref_x_max_sir, ref_cdf_env_sir = self.reference_data.get_company_cdf_envelope(scenario_key, 'wideband_sir')
                    if ref_x_min_sir is not None and ref_x_max_sir is not None:
                        sir_ref_envelope = (ref_x_min_sir, ref_x_max_sir, ref_cdf_env_sir)
                    
                    # SINR reference from geometry_sinr
                    ref_x_sinr, ref_cdf_sinr = self.reference_data.get_company_cdf(scenario_key, 'geometry_sinr')
                    if ref_x_sinr is not None:
                        sinr_ref_data = (ref_x_sinr, ref_cdf_sinr)
                        print(f"  -> Phase 1 Geometry SINR Reference: Mean = {np.mean(ref_x_sinr):.2f} dB, Std = {np.std(ref_x_sinr):.2f} dB")
                    
                    # SINR envelope from geometry_sinr
                    ref_x_min_sinr, ref_x_max_sinr, ref_cdf_env_sinr = self.reference_data.get_company_cdf_envelope(scenario_key, 'geometry_sinr')
                    if ref_x_min_sinr is not None and ref_x_max_sinr is not None:
                        sinr_ref_envelope = (ref_x_min_sinr, ref_x_max_sinr, ref_cdf_env_sinr)
                        
                elif self.calibration_phase == 2:
                    # Phase 2: Use wideband_sir (only for SIR, not SINR)
                    ref_x, ref_cdf = self.reference_data.get_company_cdf(scenario_key, 'wideband_sir')
                    if ref_x is not None:
                        sir_ref_data = (ref_x, ref_cdf)
                        print(f"  -> Phase 2 Wideband SIR Reference: Mean = {np.mean(ref_x):.2f} dB, Std = {np.std(ref_x):.2f} dB")
                    
                    # Get envelope
                    ref_x_min, ref_x_max, ref_cdf_env = self.reference_data.get_company_cdf_envelope(scenario_key, 'wideband_sir')
                    if ref_x_min is not None and ref_x_max is not None:
                        sir_ref_envelope = (ref_x_min, ref_x_max, ref_cdf_env)
            
            # Plot SIR
            sir_title = f'CDF of SIR ({assoc_label})'
            if sir_ref_data:
                phase_label = "Phase 1 Wideband SIR" if self.calibration_phase == 1 else "Phase 2 Wideband SIR"
                sir_title += f' vs 3GPP Ref ({phase_label})'
            
            self.plot_cdf(
                sir_sinr_results['SIR'],
                sir_title,
                'SIR (dB)',
                filename=output_path / f'cdf_sir_geometry_{association_method}.png',
                reference_data=sir_ref_data,
                reference_envelope=sir_ref_envelope,
                metric_name='SIR'
            )
            
            # Plot SINR
            sinr_title = f'CDF of SINR ({assoc_label})'
            if sinr_ref_data:
                sinr_title += ' vs 3GPP Ref (Phase 1 Geometry SINR)'
            
            self.plot_cdf(
                sir_sinr_results['SINR'],
                sinr_title,
                'SINR (dB)',
                filename=output_path / f'cdf_sinr_geometry_{association_method}.png',
                reference_data=sinr_ref_data,
                reference_envelope=sinr_ref_envelope,
                metric_name='SINR'
            )
        
        # 3. Delay and angle spreads CDFs (Phase 2 only)
        # Per 3GPP 38.901 Table 7.9.6.2-1 NOTE: "CDFs can be separately generated for target channel, background channel"
        # For ISAC Phase 2: compute for BOTH target and background channels separately
        should_compute_spreads = (
            self.calibration_phase == 2 and 
            (not is_isac_mode or (is_isac_mode and self.isac_channel_type in ['target', 'background']))
        )
        
        if should_compute_spreads:
            # Use ISAC-specific function for ISAC mode (analyzes all links, not just serving cell)
            if is_isac_mode:
                spreads = self.analyze_delay_and_angle_spreads_isac()
            else:
                spreads = self.analyze_delay_and_angle_spreads(association_method, serving_assignments)
            
            # Metric name mapping to reference JSON keys
            metric_map = {
                'DS': 'delay_spread',
                'ASD': 'angle_spread_asd',
                'ZSD': 'angle_spread_zsd',
                'ASA': 'angle_spread_asa',
                'ZSA': 'angle_spread_zsa'
            }
            
            # Set appropriate labels and suffixes based on mode
            if is_isac_mode:
                channel_label = f"ISAC {self.isac_channel_type.capitalize()} Channel"
                print(f"\n=== DELAY AND ANGLE SPREADS ({channel_label}) ===")
                association_suffix = f"_isac_{self.isac_channel_type}"
                assoc_label = channel_label
            else:
                print(f"\n=== DELAY AND ANGLE SPREADS ({association_method.upper()}) ===")
                association_suffix = f"_{association_method}"
                assoc_label = association_method.upper()
            
            for param_name, values in spreads.items():
                if len(values) > 0:
                    print(f"{param_name}: Mean = {np.mean(values):.2f}, Std = {np.std(values):.2f}")
                    
                    # Get reference data if available
                    ref_data = None
                    ref_envelope = None
                    if scenario_key and param_name in metric_map:
                        ref_metric_name = metric_map[param_name]
                        ref_x, ref_cdf = self.reference_data.get_company_cdf(scenario_key, ref_metric_name)
                        if ref_x is not None:
                            ref_data = (ref_x, ref_cdf)
                            print(f"  -> 3GPP Reference: Mean = {np.mean(ref_x):.2f}, Std = {np.std(ref_x):.2f}")
                        
                        # Get min/max envelope
                        ref_x_min, ref_x_max, ref_cdf_env = self.reference_data.get_company_cdf_envelope(scenario_key, ref_metric_name)
                        if ref_x_min is not None and ref_x_max is not None:
                            ref_envelope = (ref_x_min, ref_x_max, ref_cdf_env)
                    
                    # Plot CDF with reference data
                    units = "ns" if param_name == "DS" else "degrees"
                    
                    # Create clear title indicating channel type and reference comparison
                    if ref_data:
                        title = f'CDF of {param_name} ({assoc_label}) vs 3GPP Reference'
                    else:
                        title = f'CDF of {param_name} - Serving Cell ({assoc_label})'
                    
                    self.plot_cdf(
                        values, 
                        title,
                        f'{param_name} ({units})',
                        filename=output_path / f'cdf_{param_name.lower()}{association_suffix}.png',
                        reference_data=ref_data,
                        reference_envelope=ref_envelope,
                        metric_name=param_name
                    )
                else:
                    print(f"{param_name}: No data collected")
        elif is_isac_mode:
            print(f"\n=== DELAY AND ANGLE SPREADS ===")
            print(f"Skipped for ISAC {self.isac_channel_type} channel (unknown channel type or Phase 1)")
        else:
            print(f"\n=== DELAY AND ANGLE SPREADS ===")
            print("Skipped for Phase 1 calibration (not included in Phase 1 reference data)")
        
        # 4. PRB Singular Values CDFs
        # Not applicable for Phase 1 calibration (large-scale only).
        # Also skip for ISAC (no BS→UE links).
        if self.calibration_phase == 1:
            print(f"\n=== PRB SINGULAR VALUES (t=0) ===")
            print("Skipped for Phase 1 calibration (large-scale only)")
            sv_results = {}
        else:
            sv_results = self.compute_prb_singular_values()
        
        print(f"\n=== PRB SINGULAR VALUES (t=0) ===")
        if sv_results:
            # Plot largest singular values in dB
            if len(sv_results['largest']) > 0:
                largest_db = 10 * np.log10(sv_results['largest'])
                print(f"Largest SV: Mean = {np.mean(largest_db):.2f} dB, Std = {np.std(largest_db):.2f} dB")
                self.plot_cdf(
                    largest_db,
                    'CDF of Largest PRB Singular Values (Serving Cell, t=0)',
                    'Largest Singular Value (dB)',
                    filename=output_path / 'cdf_largest_sv.png'
                )
            
            # Plot smallest singular values in dB
            if len(sv_results['smallest']) > 0:
                smallest_db = 10 * np.log10(sv_results['smallest'])
                print(f"Smallest SV: Mean = {np.mean(smallest_db):.2f} dB, Std = {np.std(smallest_db):.2f} dB")
                self.plot_cdf(
                    smallest_db,
                    'CDF of Smallest PRB Singular Values (Serving Cell, t=0)',
                    'Smallest Singular Value (dB)',
                    filename=output_path / 'cdf_smallest_sv.png'
                )
            
            # Plot ratio in dB
            if len(sv_results['ratios']) > 0:
                ratios_db = 10 * np.log10(sv_results['ratios'])
                print(f"SV Ratio: Mean = {np.mean(ratios_db):.2f} dB, Std = {np.std(ratios_db):.2f} dB")
                self.plot_cdf(
                    ratios_db,
                    'CDF of PRB Singular Value Ratios (Largest/Smallest, Serving Cell, t=0)',
                    'Singular Value Ratio (dB)',
                    filename=output_path / 'cdf_sv_ratio.png'
                )
        
        # 5. Advanced pathloss CDF analysis
        print(f"\n=== PATHLOSS ANALYSIS ===")
        
        # 5a. Get and print serving cell assignments for visual verification
        serving_assignments = self.get_serving_cell_assignments()
        self.print_serving_cell_assignments(serving_assignments)
        
        # 5b. Pathloss CDF for all links
        all_coupling_losses = self.compute_coupling_loss_all_links(use_virtualization)
        if len(all_coupling_losses) > 0:
            print(f"Coupling Loss (All Links): Mean = {np.mean(all_coupling_losses):.2f} dB, Std = {np.std(all_coupling_losses):.2f} dB")
            self.plot_pathloss_cdf(
                all_coupling_losses,
                scenario_name=f"{self.system_level_config.scenario if self.system_level_config else 'UMa'} - All Links",
                save_path=output_path / 'pathloss_cdf_all_links.png'
            )
        
        # 5b. Pathloss CDF for serving cells only (determined by CIR power)
        serving_coupling_losses = self.compute_coupling_loss_serving_cells_only(use_virtualization)
        if len(serving_coupling_losses) > 0:
            print(f"Coupling Loss (Serving Cells Only): Mean = {np.mean(serving_coupling_losses):.2f} dB, Std = {np.std(serving_coupling_losses):.2f} dB")
            self.plot_pathloss_cdf(
                serving_coupling_losses,
                scenario_name=f"{self.system_level_config.scenario if self.system_level_config else 'UMa'} - Serving Cells",
                save_path=output_path / 'pathloss_cdf_serving_cells.png'
            )
        
        # 5c. Combined comparison plot
        if len(all_coupling_losses) > 0 and len(serving_coupling_losses) > 0:
            self.plot_pathloss_comparison_cdf(
                all_coupling_losses, 
                serving_coupling_losses,
                scenario_name=self.system_level_config.scenario if self.system_level_config else "UMa",
                save_path=output_path / 'pathloss_cdf_comparison.png'
            )
        
        # 6. Advanced geometry SIR/SINR analysis
        if self.active_link_params and self.link_params:
            print(f"\n=== ADVANCED GEOMETRY ANALYSIS ===")
            advanced_results = self.calculate_and_plot_advanced_geometry_sir_sinr(
                scenario_name=self.system_level_config.scenario if self.system_level_config else "UMa",
                center_freq_hz=center_freq_hz,
                output_dir=str(output_path)
            )
            if advanced_results and len(advanced_results['SIR']) > 0:
                print(f"Advanced SIR: Mean = {np.mean(advanced_results['SIR']):.2f} dB, Std = {np.std(advanced_results['SIR']):.2f} dB")
                print(f"Advanced SINR: Mean = {np.mean(advanced_results['SINR']):.2f} dB, Std = {np.std(advanced_results['SINR']):.2f} dB")
        
        print(f"\nAll plots saved to: {output_path}")
        logger.info("Analysis complete!")

def main():
    """Main function to run the analysis"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Analyze H5 Channel Model Data with 3GPP Reference Comparison',
        epilog=(
            'Cell Association Methods:\n'
            '  RSRP-based (default): Uses received signal power (TX power - (path loss - shadow fading))\n'
            '                        to determine serving cells. This is the standard 3GPP method.\n'
            '  CIR-based:           Uses Channel Impulse Response power from H5 file for cell association.\n'
            '                        This may give different results if CIR data is available.\n'
            '  Distance-based:      Uses minimal 3D distance for cell association.\n'
            '\n'
            'Antenna Virtualization (ITU-R M.2101):\n'
            '  Enabled by default. Use --no-virtualization to disable.\n'
            '  Phase 1: Simplified beamforming with tilt=12°, numM=10 antenna rows\n'
            '           w_virt = (1/sqrt(numM)) * exp(-1i*pi*cos(deg2rad(tilt+90))*(0:numM-1))\n'
            '           Provides ~10 dB antenna gain (10*log10(10))\n'
            '  Phase 2: Groups 16 elements into 1 CRS port (64 elements -> 4 TX ports)\n'
            '           Sweeps 12 angles from -60° to 60° (10° steps) relative to sector orientation\n'
            '           Selects best gain among all TX ports at all angles\n'
            '  Without virtualization: Simple gain = 10*log10(N) where N = number of antennas\n'
            '\n'
            '3GPP Reference Data:\n'
            '  Use --reference-json to provide 3GPP calibration data for comparison.\n'
            '  Use --calibration-phase to specify Phase 1 or Phase 2 calibration data:\n'
            '    Phase 1: coupling_loss, wideband_sir (SIR), geometry_sinr (SINR)\n'
            '    Phase 2: coupling_loss, wideband_sir, delay_spread, angle_spreads\n'
            '  The script will automatically match the scenario and frequency from the H5 file.\n'
            '  Reference CDFs will be overlaid on simulation plots for ALL association methods.\n'
            '  This allows you to compare different association methods against 3GPP reference.\n'
            '\n'
            'Example Usage:\n'
            '  # Phase 2 (default) with virtualization and all metrics:\n'
            '  python analysis_channel_stats.py input.h5 --reference-json 3gpp_calibration_phase2.json\n'
            '\n'
            '  # Phase 1 with virtualization and coupling loss:\n'
            '  python analysis_channel_stats.py input.h5 --reference-json 3gpp_calibration_phase1.json --calibration-phase 1\n'
            '\n'
            '  # Without virtualization (simple antenna gain):\n'
            '  python analysis_channel_stats.py input.h5 --no-virtualization\n'
            '\n'
            '  # CIR-based with Phase 2 reference curves:\n'
            '  python analysis_channel_stats.py input.h5 --assoc cir --reference-json 3gpp_calibration_phase2.json\n'
            '\n'
            'ISAC Calibration Mode:\n'
            '  For ISAC (Integrated Sensing and Communications) calibration, use:\n'
            '  --isac-channel: Specify channel type for ISAC reference data lookup\n'
            '                  Required for ISAC calibration, auto-detected from H5 file\n'
            '    target:     Use target channel reference (TRPmo(t), TRP-TRP, etc.)\n'
            '    background: Use background channel reference (TRPmo(b), TRP-TRP(b), etc.)\n'
            '\n'
            '  IMPORTANT: 3GPP calibrates target and background channels SEPARATELY.\n'
            '  If combined mode (isac_disable_background=0), an error is thrown.\n'
            '  Run with isac_disable_background=1 for target-only analysis.\n'
            '\n'
            'ISAC Example Usage:\n'
            '  # ISAC Phase 1 (Large Scale) - target channel:\n'
            '  python analysis_channel_stats.py isac_target.h5 \\\n'
            '      --reference-json 3gpp_calibration_isac_uav_phase1.json \\\n'
            '      --calibration-phase 1 --isac-channel target\n'
            '\n'
            '  # ISAC Phase 2 (Full) - background channel:\n'
            '  python analysis_channel_stats.py isac_background.h5 \\\n'
            '      --reference-json 3gpp_calibration_isac_uav_phase2.json \\\n'
            '      --calibration-phase 2 --isac-channel background'
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('h5_file', help='Path to H5 TV file')
    parser.add_argument('--output-dir', default='channel_analysis_plots', 
                       help='Output directory for plots (default: channel_analysis_plots)')
    parser.add_argument('--reference-json', type=str,
                       help='Path to 3GPP calibration reference JSON file (e.g., 3gpp_calibration_phase1.json or 3gpp_calibration_phase2.json)')
    parser.add_argument('--calibration-phase', type=int, default=2, choices=[1, 2],
                       help='3GPP calibration phase: 1 (coupling_loss, wideband_sir, geometry_sinr) or 2 (coupling_loss, wideband_sir, spreads) (default: 2)')
    parser.add_argument('--log-file', type=str,
                       help='Log file path (if not provided, logs to console only)')
    parser.add_argument('--log-level', type=str, default='ERROR',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level (default: ERROR)')
    parser.add_argument('--assoc', type=str, default='rsrp', 
                       choices=['rsrp', 'cir', 'distance'],
                       help='Cell association method: rsrp (RSRP-based, default), cir (CIR power-based), or distance (minimal distance-based)')
    parser.add_argument('--use-virtualization', action='store_true', default=True,
                       help='Use antenna virtualization per ITU-R M.2101 (default: True)')
    parser.add_argument('--no-virtualization', dest='use_virtualization', action='store_false',
                       help='Disable antenna virtualization (use simple 10*log10(N) gain)')
    parser.add_argument('--isac-channel', type=str, choices=['target', 'background'],
                       help='ISAC channel type for reference data lookup. Required for ISAC calibration.')
    parser.add_argument('--multi-seed', action='store_true',
                       help='Enable multi-seed analysis: find all H5 files matching the base pattern with different seeds '
                            '(e.g., file_seed10.h5, file_seed20.h5) and combine results for CDF analysis.')
    
    args = parser.parse_args()
    
    # Get association method from command line argument
    association_method = args.assoc
    calibration_phase = args.calibration_phase
    isac_channel_type = args.isac_channel
    
    # Append association method and phase number to output directory if using default
    output_dir = args.output_dir
    if output_dir == 'channel_analysis_plots':
        if isac_channel_type:
            output_dir = f'channel_analysis_plots_isac_{isac_channel_type}_phase{calibration_phase}'
        else:
            output_dir = f'channel_analysis_plots_{association_method}_phase{calibration_phase}'
        logger.info(f"Using phase and association-specific output directory: {output_dir}")
    
    # Set up logging based on command line arguments
    setup_logging(log_file=args.log_file, log_level=args.log_level)
    
    # Multi-seed mode: find and combine multiple H5 files with different seeds
    if args.multi_seed:
        h5_files = find_multi_seed_files(args.h5_file)
        if len(h5_files) > 1:
            print(f"\n{'='*60}")
            print(f"MULTI-SEED ANALYSIS MODE: {len(h5_files)} drops found")
            print(f"{'='*60}")
            for f in h5_files:
                print(f"  - {f}")
            print(f"{'='*60}\n")
            
            # Collect data from all files
            all_coupling_losses = []
            all_ds_values = []
            all_asd_values = []
            all_zsd_values = []
            all_asa_values = []
            all_zsa_values = []
            first_analyzer = None
            is_isac_mode = False
            
            for h5_file in h5_files:
                print(f"Processing: {h5_file}")
                temp_analyzer = H5ChannelAnalyzer(
                    h5_file,
                    reference_json_path=args.reference_json,
                    calibration_phase=calibration_phase,
                    isac_channel_type=isac_channel_type
                )
                temp_analyzer.parse_h5_file()
                
                # Store the first analyzer for config info
                if first_analyzer is None:
                    first_analyzer = temp_analyzer
                    is_isac_mode = temp_analyzer.isac_config and temp_analyzer.isac_config.is_enabled
                
                # Collect coupling loss data
                coupling_losses = temp_analyzer.compute_coupling_loss_serving_cells_only(args.use_virtualization)
                all_coupling_losses.extend(coupling_losses.tolist())
                
                # For Phase 2, indicate if using CIR-based computation
                if calibration_phase == 2 and is_isac_mode:
                    print(f"  -> Collected {len(coupling_losses)} Phase 2 coupling loss samples (from CIR)")
                else:
                    print("  -> Collected " + str(len(coupling_losses)) + " coupling loss samples")
                
                # Collect delay/angle spreads for Phase 2
                if calibration_phase == 2:
                    if is_isac_mode:
                        spreads = temp_analyzer.analyze_delay_and_angle_spreads_isac()
                    else:
                        spreads = temp_analyzer.analyze_delay_and_angle_spreads(association_method)
                    
                    if spreads['DS'] is not None and len(spreads['DS']) > 0:
                        all_ds_values.extend(spreads['DS'].tolist())
                    if spreads['ASD'] is not None and len(spreads['ASD']) > 0:
                        all_asd_values.extend(spreads['ASD'].tolist())
                    if spreads['ZSD'] is not None and len(spreads['ZSD']) > 0:
                        all_zsd_values.extend(spreads['ZSD'].tolist())
                    if spreads['ASA'] is not None and len(spreads['ASA']) > 0:
                        all_asa_values.extend(spreads['ASA'].tolist())
                    if spreads['ZSA'] is not None and len(spreads['ZSA']) > 0:
                        all_zsa_values.extend(spreads['ZSA'].tolist())
                    
                    print(f"  -> Collected {len(spreads['DS'])} delay spread samples")
                    print(f"  -> Collected {len(spreads['ASD'])} ASD, {len(spreads['ASA'])} ASA, {len(spreads['ZSD'])} ZSD, {len(spreads['ZSA'])} ZSA samples")
            
            # Update the first analyzer with combined data
            print(f"\nTotal samples collected:")
            print(f"  Coupling losses: {len(all_coupling_losses)}")
            if calibration_phase == 2:
                print(f"  Delay spreads: {len(all_ds_values)}")
                print(f"  Angle spreads: ASD={len(all_asd_values)}, ZSD={len(all_zsd_values)}, ASA={len(all_asa_values)}, ZSA={len(all_zsa_values)}")
            
            # Create combined analyzer using first file's config but with multi-seed data
            analyzer = first_analyzer
            analyzer._multi_seed_coupling_losses = np.array(all_coupling_losses)
            analyzer._multi_seed_spreads = {
                'DS': np.array(all_ds_values),
                'ASD': np.array(all_asd_values),
                'ZSD': np.array(all_zsd_values),
                'ASA': np.array(all_asa_values),
                'ZSA': np.array(all_zsa_values)
            }
            analyzer._is_multi_seed = True
            
        else:
            print("Multi-seed mode enabled but only 1 file found. Proceeding with single file.")
            analyzer = H5ChannelAnalyzer(
                args.h5_file, 
                reference_json_path=args.reference_json, 
                calibration_phase=calibration_phase,
                isac_channel_type=isac_channel_type
            )
            analyzer.parse_h5_file()
            analyzer._is_multi_seed = False
    else:
        # Standard single-file analysis
        analyzer = H5ChannelAnalyzer(
            args.h5_file, 
            reference_json_path=args.reference_json, 
            calibration_phase=calibration_phase,
            isac_channel_type=isac_channel_type
        )
        analyzer.parse_h5_file()
        analyzer._is_multi_seed = False
    
    # ISAC mode validation
    if analyzer.isac_config and analyzer.isac_config.is_enabled:
        logger.info("=" * 60)
        logger.info("ISAC MODE DETECTED")
        logger.info("=" * 60)
        logger.info(f"  Sensing mode: {'Monostatic' if analyzer.isac_config.is_monostatic else 'Bistatic'}")
        logger.info(f"  Target type: {analyzer.isac_config.target_type_name}")
        logger.info(f"  Number of STs: {analyzer.isac_config.n_st}")
        logger.info(f"  RCS model: {analyzer.isac_config.st_rcs_model}")
        logger.info(f"  Disable background: {analyzer.isac_config.isac_disable_background}")
        logger.info(f"  Disable target: {analyzer.isac_config.isac_disable_target}")
        
        # Determine CIR mode
        if analyzer.isac_config.is_target_only:
            cir_mode = "target-only"
        elif analyzer.isac_config.is_background_only:
            cir_mode = "background-only"
        else:
            cir_mode = "combined"
        logger.info(f"  CIR mode: {cir_mode}")
        
        # Check for combined mode - this is not supported for 3GPP calibration
        if analyzer.isac_config.is_combined:
            if args.reference_json:
                logger.error("=" * 60)
                logger.error("ERROR: ISAC COMBINED MODE NOT SUPPORTED FOR CALIBRATION")
                logger.error("=" * 60)
                logger.error("The H5 file contains combined target+background CIR data.")
                logger.error("3GPP TR 38.901 Section 7.9.6 requires SEPARATE calibration for:")
                logger.error("  - Target channel (coupling loss for target path)")
                logger.error("  - Background channel (coupling loss for reference points)")
                logger.error("")
                logger.error("To fix this, choose ONE of the following:")
                logger.error("  Option A (Target-only):")
                logger.error("    1. Set 'isac_disable_background: 1' in your YAML config")
                logger.error("    2. Re-run simulation to generate target-only CIR")
                logger.error("    3. Analyze with: --isac-channel target")
                logger.error("")
                logger.error("  Option B (Background-only):")
                logger.error("    1. Set 'isac_disable_target: 1' in your YAML config")
                logger.error("    2. Re-run simulation to generate background-only CIR")
                logger.error("    3. Analyze with: --isac-channel background")
                logger.error("=" * 60)
                raise ValueError("ISAC combined mode not supported for 3GPP calibration. "
                               "Set isac_disable_background=1 OR isac_disable_target=1 in YAML config.")
            else:
                logger.warning("ISAC combined mode detected. No reference JSON provided, proceeding with analysis.")
        
        # Auto-detect channel type if not specified
        if not isac_channel_type and args.reference_json:
            if analyzer.isac_config.is_target_only:
                isac_channel_type = 'target'
                analyzer.isac_channel_type = 'target'
                logger.info(f"Auto-detected ISAC channel type: target (background disabled)")
            elif analyzer.isac_config.is_background_only:
                isac_channel_type = 'background'
                analyzer.isac_channel_type = 'background'
                logger.info(f"Auto-detected ISAC channel type: background (target disabled)")
            else:
                logger.warning("Cannot auto-detect ISAC channel type for combined mode.")
        
        if isac_channel_type:
            logger.info(f"  Reference lookup: {isac_channel_type} channel")
        logger.info("=" * 60)
    
    # Log the cell association method being used
    logger.info(f"Using '{association_method}' cell association for analysis")
    logger.info(f"Antenna virtualization: {'ENABLED' if args.use_virtualization else 'DISABLED'}")
    
    if args.reference_json:
        logger.info(f"3GPP Phase {calibration_phase} reference comparison enabled from: {args.reference_json}")
        if calibration_phase == 1:
            logger.info("Phase 1 metrics: coupling_loss, wideband_sir (SIR), geometry_sinr (SINR)")
        else:
            logger.info("Phase 2 metrics: coupling_loss, wideband_sir, delay_spread, angle_spreads")
    else:
        logger.info("No 3GPP reference data provided - running without comparison")
    
    # Generate all analysis and plots
    analyzer.generate_all_plots(output_dir, association_method, use_virtualization=args.use_virtualization)
    
    # Also generate advanced geometry analysis if H5 data is available
    if analyzer.active_link_params and analyzer.link_params:
        logger.info("Generating advanced geometry analysis...")
        analyzer.calculate_and_plot_advanced_geometry_sir_sinr(
            scenario_name=analyzer.system_level_config.scenario if analyzer.system_level_config else "UMa",
            output_dir=output_dir
        )

if __name__ == "__main__":
    main()

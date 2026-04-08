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

"""pyAerial library - channel models.

This module provides channel models for 5G NR simulations:

Fading Channel Models:
- FadingChannel: GPU-accelerated fading channel with CuPy/NumPy support
- TdlChannelConfig: TDL (Tapped Delay Line) channel configuration
- CdlChannelConfig: CDL (Clustered Delay Line) channel configuration
- FadingChannelConfig: Base class for fading channel configurations

Statistical Channel Model:
- StatisticalChannel: GPU-accelerated statistical channel for system-level simulations
- SimConfig: Simulation configuration (frequency, FFT, run mode)
- SystemLevelConfig: System-level parameters (scenario, path loss, shadowing)
- LinkLevelConfig: Link-level parameters (fading type, delay profile)
- ExternalConfig: External configuration (cells, UTs, antenna panels)

Example (Fading Channel):
    >>> from aerial.phy5g.channel_models import TdlChannelConfig, FadingChannel
    >>>
    >>> config = TdlChannelConfig(
    ...     delay_profile='A',
    ...     delay_spread=30.0,
    ...     n_bs_ant=4,
    ...     n_ue_ant=2
    ... )
    >>> channel = FadingChannel(channel_config=config, n_sc=3276, numerology=1)
    >>> rx = channel(freq_in=tx_signal, tti_idx=0, snr_db=20.0)

Example (Statistical Channel):
    >>> from aerial.phy5g.channel_models import (
    ...     StatisticalChannel, SimConfig, SystemLevelConfig,
    ...     LinkLevelConfig, ExternalConfig
    ... )
    >>>
    >>> sim_cfg = SimConfig(center_freq_hz=3.5e9, fft_size=4096)
    >>> sys_cfg = SystemLevelConfig(scenario='UMa', n_site=7, n_ut=100)
    >>> link_cfg = LinkLevelConfig(fast_fading_type=2, delay_profile='A')
    >>> ext_cfg = ExternalConfig(cell_config=cells, ut_config=uts, ant_panel_config=panels)
    >>> channel = StatisticalChannel(
    ...     sim_config=sim_cfg,
    ...     system_level_config=sys_cfg,
    ...     link_level_config=link_cfg,
    ...     external_config=ext_cfg
    ... )
    >>> channel.run(ref_time=0.0, active_cell=[0, 1, 2])
"""
# pylint: disable=no-name-in-module

# Statistical channel configs (re-exported from C++ bindings)
# These are documented in pybind11
from aerial.pycuphy import (
    Scenario,
    SensingTargetType,
    UeType,
    Coordinate,
    AntPanelConfig,
    UtParamCfg,
    SpstParam,
    StParam,
    CellParam,
    SimConfig,
    SystemLevelConfig,
    LinkLevelConfig,
    ExternalConfig,
)

from .fading_channel import FadingChannel
from .statistical_channel import (
    StatisticalChannel,
)
from .channel_config import (
    # Fading channel configs (Python classes)
    FadingChannelConfig,
    TdlChannelConfig,
    CdlChannelConfig,
    # Helper functions
    create_antenna_pattern,
)

__all__ = [
    # Fading channel
    "FadingChannel",
    "FadingChannelConfig",
    "TdlChannelConfig",
    "CdlChannelConfig",
    # Statistical channel
    "StatisticalChannel",
    "Scenario",
    "SensingTargetType",
    "UeType",
    "Coordinate",
    "AntPanelConfig",
    "UtParamCfg",
    "SpstParam",
    "StParam",
    "CellParam",
    "SimConfig",
    "SystemLevelConfig",
    "LinkLevelConfig",
    "ExternalConfig",
    # Helper functions
    "create_antenna_pattern",
]

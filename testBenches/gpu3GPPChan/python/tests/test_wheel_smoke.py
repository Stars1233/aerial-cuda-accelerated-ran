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

"""Smoke + robustness tests for an installed gpu3gppchan wheel.

Tier 1 (import / config construction) runs anywhere — no cupy or GPU needed.
Tier 2 (FadingChannel / StatisticalChannel GPU runs) is gated by ``REQUIRES_GPU``
and auto-skips when cupy or a CUDA device is unavailable.
"""

from __future__ import annotations

import os
import sys

import pytest


def _gpu_available() -> bool:
    try:
        import cupy
        return cupy.cuda.runtime.getDeviceCount() > 0
    except (ImportError, RuntimeError):
        return False


REQUIRES_GPU = pytest.mark.skipif(
    not _gpu_available(), reason="no cupy / no CUDA device"
)


def test_import() -> None:
    gil_checker = getattr(sys, "_is_gil_enabled", None)
    gil_before = gil_checker() if gil_checker is not None else None

    import gpu3gppchan

    gil_after = gil_checker() if gil_checker is not None else None
    if gil_before is False and gil_after is not False:
        pytest.fail("importing gpu3gppchan re-enabled the GIL")

    assert gpu3gppchan.__version__
    assert os.path.exists(gpu3gppchan._gpu3gppchan.__file__)


def test_config_construction() -> None:
    import gpu3gppchan

    sim_cfg = gpu3gppchan.SimConfig(
        center_freq_hz=3.5e9, bandwidth_hz=100e6, run_mode=0
    )
    sys_cfg = gpu3gppchan.SystemLevelConfig(
        scenario=gpu3gppchan.Scenario.UMa, n_site=1, n_sector_per_site=1, n_ut=1
    )
    link_cfg = gpu3gppchan.LinkLevelConfig(
        fast_fading_type=1, delay_profile="A", delay_spread=30.0
    )

    assert sim_cfg.center_freq_hz == 3.5e9
    assert sys_cfg.n_site == 1
    assert link_cfg.fast_fading_type == 1


@REQUIRES_GPU
def test_fading_channel_tdl_run() -> None:
    import cupy as cp

    from gpu3gppchan import FadingChannel, TdlChannelConfig

    config = TdlChannelConfig(
        delay_profile="A", delay_spread=30.0, n_bs_ant=1, n_ue_ant=1
    )
    # Defaults: n_sc=3276, n_fft=4096, numerology=1, n_symbol_slot=14.
    chan = FadingChannel(channel_config=config)

    # freq_in shape: (n_cell, n_ue, n_tx_ant, n_symbol, n_sc)
    tx = cp.ones((1, 1, 1, 14, 3276), dtype=cp.complex64)
    rx = chan.run(freq_in=tx, tti_idx=0, snr_db=20.0)

    # DL output: (n_cell, n_ue, n_ue_ant, n_symbol, n_sc)
    assert rx.shape == (1, 1, 1, 14, 3276)
    assert bool(cp.all(cp.isfinite(rx)))


@REQUIRES_GPU
def test_statistical_channel_minimal_run() -> None:
    import gpu3gppchan as chan

    sim_cfg = chan.SimConfig(
        center_freq_hz=3.5e9, bandwidth_hz=100e6, run_mode=0
    )
    # This minimal smoke test supplies no external CIR/CFR buffers, so retain
    # CIR internally rather than selecting the caller-buffer mode.
    sim_cfg.internal_memory_mode = 1
    sys_cfg = chan.SystemLevelConfig(
        scenario=chan.Scenario.UMa,
        n_site=1,
        n_sector_per_site=1,
        n_ut=1,
    )
    link_cfg = chan.LinkLevelConfig(
        fast_fading_type=1, delay_profile="A", delay_spread=30.0
    )

    bs_panel = chan.AntPanelConfig(
        n_ant=1,
        ant_size=[1, 1, 1, 1, 1],
        ant_spacing=[0.0, 0.0, 0.5, 0.5],
        ant_polar_angles=[0.0, 0.0],
        ant_model=0,
    )
    ut_panel = chan.AntPanelConfig(
        n_ant=1,
        ant_size=[1, 1, 1, 1, 1],
        ant_spacing=[0.0, 0.0, 0.5, 0.5],
        ant_polar_angles=[0.0, 0.0],
        ant_model=0,
    )
    cell = chan.CellParam(
        0, 0, chan.Coordinate(0.0, 0.0, 25.0), 0, [0.0, 0.0, 0.0]
    )
    ut = chan.UtParamCfg(
        0,
        chan.Coordinate(100.0, 0.0, 1.5),
        1,
        1,
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        chan.UeType.TERRESTRIAL,
    )
    external_cfg = chan.ExternalConfig()
    external_cfg.cell_config = [cell]
    external_cfg.ut_config = [ut]
    external_cfg.ant_panel_config = [bs_panel, ut_panel]

    model = chan.StatisticalChannel(
        sim_config=sim_cfg,
        system_level_config=sys_cfg,
        link_level_config=link_cfg,
        external_config=external_cfg,
    )

    # Exercises installed wheel only; no imports from repository test helpers.
    model.run(ref_time=0.0)

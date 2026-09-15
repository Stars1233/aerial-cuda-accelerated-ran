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

"""Unit tests for statistical-channel analysis helpers."""

import sys
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

CHANNEL_UTIL_DIR = Path(__file__).resolve().parents[2] / "util"
sys.path.insert(0, str(CHANNEL_UTIL_DIR))

from analysis_channel_stats import (  # noqa: E402  # pylint: disable=wrong-import-position
    ActiveLinkParams,
    H5ChannelAnalyzer,
)


def _write_isac_calibration_paths(
        h5_path: Path,
        channel_group: str,
        group_ids: np.ndarray,
        delays_ns: np.ndarray) -> None:
    """Write a minimal generated-path group for ISAC spread analysis."""
    path_count = len(group_ids)
    with h5py.File(h5_path, "w") as h5_file:
        path_group = h5_file.create_group(
            f"topology/{channel_group}/calibrationPaths"
        )
        path_group.create_dataset("group_id", data=group_ids)
        path_group.create_dataset("delay_ns", data=delays_ns)
        path_group.create_dataset("power", data=np.ones(path_count))
        for name in ("aod_deg", "zod_deg", "aoa_deg", "zoa_deg"):
            path_group.create_dataset(name, data=np.arange(path_count) * 10.0)


def test_all_interference_free_ues_report_infinite_sir(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """Represent an all-interference-free deployment with positive infinity."""
    analyzer = H5ChannelAnalyzer("unused.h5")
    analyzer.active_link_params = [
        ActiveLinkParams(cid=0, uid=0, link_idx=0, lsp_read_idx=0)
    ]
    analyzer.link_params = [SimpleNamespace(pathloss=80.0, sf=0.0)]
    monkeypatch.setattr(
        analyzer,
        "_compute_serving_assignments",
        lambda _method: {0: 0},
    )
    monkeypatch.setattr(
        analyzer,
        "get_antenna_gain_for_cell",
        lambda *_args, **_kwargs: 0.0,
    )

    assert analyzer.compute_wideband_sir() == float("inf")


def test_missing_serving_signal_reports_zero_sir(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep zero as the sentinel when no serving signal can be determined."""
    analyzer = H5ChannelAnalyzer("unused.h5")
    analyzer.active_link_params = []
    analyzer.link_params = []
    monkeypatch.setattr(
        analyzer,
        "_compute_serving_assignments",
        lambda _method: {0: 0},
    )

    assert analyzer.compute_wideband_sir() == 0.0


def test_isac_target_spreads_use_generated_paths(
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch) -> None:
    """Measure a target channel from coupled ISAC paths, not comm clusters."""
    h5_path = tmp_path / "target.h5"
    _write_isac_calibration_paths(
        h5_path,
        "isacTargetLinks",
        np.array([0, 0], dtype=np.uint32),
        np.array([0.0, 10.0]),
    )
    with h5py.File(h5_path, "r+") as h5_file:
        h5_file["topology/isacTargetLinks/calibrationPaths/power"][:] = [1.0, 3.0]

    analyzer = H5ChannelAnalyzer(str(h5_path), isac_channel_type="target")
    monkeypatch.setattr(analyzer, "_save_spreads_to_json", lambda *_args: None)

    spreads = analyzer.analyze_delay_and_angle_spreads_isac()

    assert spreads["DS"] == pytest.approx([np.sqrt(18.75)])
    for metric_name in ("ASD", "ZSD", "ASA", "ZSA"):
        assert spreads[metric_name] == pytest.approx([np.sqrt(18.75)])


def test_isac_background_spreads_are_grouped_per_reference_point(
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch) -> None:
    """Produce one background spread sample for each monostatic RP."""
    h5_path = tmp_path / "background.h5"
    _write_isac_calibration_paths(
        h5_path,
        "isacBackgroundLinks",
        np.array([0, 0, 1, 1, 2, 2], dtype=np.uint32),
        np.array([0.0, 10.0, 0.0, 20.0, 0.0, 30.0]),
    )

    analyzer = H5ChannelAnalyzer(str(h5_path), isac_channel_type="background")
    monkeypatch.setattr(analyzer, "_save_spreads_to_json", lambda *_args: None)

    spreads = analyzer.analyze_delay_and_angle_spreads_isac()

    assert spreads["DS"] == pytest.approx([5.0, 10.0, 15.0])
    for metric_name in ("ASD", "ZSD", "ASA", "ZSA"):
        assert len(spreads[metric_name]) == 3


def test_isac_spread_analysis_rejects_legacy_h5_without_generated_paths(
        tmp_path: Path) -> None:
    """Do not substitute communication clusters for an ISAC channel."""
    h5_path = tmp_path / "legacy.h5"
    with h5py.File(h5_path, "w"):
        pass

    analyzer = H5ChannelAnalyzer(str(h5_path), isac_channel_type="target")

    with pytest.raises(ValueError, match="calibrationPaths are missing"):
        analyzer.analyze_delay_and_angle_spreads_isac()


def test_isac_spread_analysis_accepts_current_empty_channel(
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch) -> None:
    """Represent a current channel with no surviving paths as empty metrics."""
    h5_path = tmp_path / "empty.h5"
    with h5py.File(h5_path, "w") as h5_file:
        group = h5_file.create_group("topology/isacTargetLinks")
        group.create_dataset("calibration_paths_schema_version", data=1)

    analyzer = H5ChannelAnalyzer(str(h5_path), isac_channel_type="target")
    monkeypatch.setattr(analyzer, "_save_spreads_to_json", lambda *_args: None)

    spreads = analyzer.analyze_delay_and_angle_spreads_isac()

    assert all(len(values) == 0 for values in spreads.values())

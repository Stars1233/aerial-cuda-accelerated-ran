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

"""Compatibility checks for pyAerial's gpu3gppchan facade."""

from typing import Any, Callable

import numpy as np
import pytest

import gpu3gppchan
import aerial.phy5g.channel_models as aerial_channel_models
from gpu3gppchan.channel_config import Array as Gpu3gppchanArray


def _gpu_available() -> bool:
    try:
        import cupy

        return cupy.cuda.runtime.getDeviceCount() > 0
    # A missing optional CuPy dependency skips GPU-only tests. Do not catch
    # RuntimeError: a broken CUDA runtime must fail the test environment.
    except ImportError:
        print("CuPy is not installed; skipping GPU-only compatibility tests.")
        return False


REQUIRES_GPU = pytest.mark.skipif(
    not _gpu_available(), reason="no cupy / no CUDA device"
)


@pytest.mark.parametrize(
    "public_name",
    [
        "FadingChannel",
        "TdlChannelConfig",
        "CdlChannelConfig",
        "StatisticalChannel",
        "SimConfig",
        "SystemLevelConfig",
        "LinkLevelConfig",
        "ExternalConfig",
        "StatisChanModel",
        "LinkParams",
        "ClusterParams",
        "CarrierParams",
        "OfdmModulate",
        "OfdmDeModulate",
    ],
)
def test_channel_model_facade_reexports_package_api(public_name: str) -> None:
    """The pyAerial namespace must expose the component's public objects unchanged."""
    assert getattr(aerial_channel_models, public_name) is getattr(
        gpu3gppchan, public_name
    )


def test_channel_model_facade_reexports_array_protocol() -> None:
    """The facade must preserve the shared NumPy/CuPy array type alias."""
    assert aerial_channel_models.Array is Gpu3gppchanArray


def test_statistical_channel_call_forwards_ut_update_stream() -> None:
    """Call syntax must forward the mobility producer stream to ``run``."""
    channel = aerial_channel_models.StatisticalChannel.__new__(
        aerial_channel_models.StatisticalChannel
    )
    stream = object()
    received = {}

    def record_run(**kwargs: Any) -> None:
        received.update(kwargs)

    channel.run = record_run
    channel(ut_update_stream=stream)

    assert received["ut_update_stream"] is stream


def test_statistical_channel_forwards_los_override() -> None:
    """The public wrapper normalizes LOS values before forwarding them."""

    class FakeImpl:
        """Record calls without constructing the native CUDA-backed model."""

        def __init__(self) -> None:
            self.los_ind = None
            self.clear_count = 0

        def set_los_override(self, los_ind: np.ndarray) -> None:
            self.los_ind = los_ind

        def clear_los_override(self) -> None:
            self.clear_count += 1

    native_impl = FakeImpl()
    channel = aerial_channel_models.StatisticalChannel.__new__(
        aerial_channel_models.StatisticalChannel
    )
    channel._impl = native_impl

    channel.set_los_override(np.array([1, 0, 255], dtype=np.int64))
    channel.clear_los_override()

    assert native_impl.los_ind is not None
    assert native_impl.los_ind.dtype == np.uint8
    assert native_impl.los_ind.flags.c_contiguous
    np.testing.assert_array_equal(native_impl.los_ind, [1, 0, 255])
    assert native_impl.clear_count == 1


@pytest.mark.parametrize(
    ("los_ind", "match"),
    [
        (
            np.array([[1, 0]], dtype=np.uint8),
            r"^los_ind must be a one-dimensional array$",
        ),
        (
            np.array([1.0, 0.0], dtype=np.float32),
            r"^los_ind must contain integer values$",
        ),
        (
            np.array([1, 2], dtype=np.uint8),
            r"^los_ind entries must be 0 \(NLOS\), 1 \(LOS\), or 255 \(model draw\)$",
        ),
    ],
)
def test_statistical_channel_rejects_invalid_los_override(
    los_ind: np.ndarray,
    match: str,
) -> None:
    """Invalid LOS arrays fail before reaching the native model."""
    channel = aerial_channel_models.StatisticalChannel.__new__(
        aerial_channel_models.StatisticalChannel
    )
    with pytest.raises(ValueError, match=match):
        channel.set_los_override(los_ind)


@pytest.mark.parametrize(
    "config_factory",
    [
        lambda: aerial_channel_models.TdlChannelConfig(
            delay_profile="A",
            delay_spread=30.0,
            n_bs_ant=1,
            n_ue_ant=1,
        ),
        lambda: aerial_channel_models.CdlChannelConfig(
            delay_profile="A",
            delay_spread=30.0,
            bs_ant_size=(1, 1, 1, 1, 1),
            ue_ant_size=(1, 1, 1, 1, 1),
        ),
    ],
    ids=["tdl", "cdl"],
)
@REQUIRES_GPU
def test_channel_model_facade_runs_fading_channel(
    config_factory: Callable[[], Any],
) -> None:
    """A small fading-channel call must work through the pyAerial import path."""
    n_sc = 1632
    n_symbol_slot = 14
    channel = aerial_channel_models.FadingChannel(
        channel_config=config_factory(),
        n_sc=n_sc,
        n_symbol_slot=n_symbol_slot,
        disable_noise=True,
    )
    tx_signal = np.ones(
        (1, 1, 1, n_symbol_slot, n_sc), dtype=np.complex64
    )

    rx_signal = channel.run(freq_in=tx_signal, tti_idx=0, snr_db=20.0)

    assert rx_signal.shape == (1, 1, 1, n_symbol_slot, n_sc)
    assert np.all(np.isfinite(rx_signal))

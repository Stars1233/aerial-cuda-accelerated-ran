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

import gc
import sys
from typing import Any, Callable

import numpy as np
import cupy as cp
import pytest
import matplotlib.pyplot as plt
import gpu3gppchan._gpu3gppchan as _native
from gpu3gppchan import (
    FadingChannel,
    TdlChannelConfig,
    CdlChannelConfig,
)
from gpu3gppchan.cuda_utils import CudaStream


@pytest.mark.parametrize(
    "config_factory, channel_factory, changed_value",
    [
        (
            _native.TdlConfig,
            lambda cfg, stream: _native.TdlChan(
                tdl_cfg=cfg, rand_seed=0, stream_handle=stream
            ),
            1,
        ),
        (
            _native.CdlConfig,
            lambda cfg, stream: _native.CdlChan(
                cdl_cfg=cfg, rand_seed=0, stream_handle=stream
            ),
            [1, 1, 1, 1, 1],
        ),
    ],
    ids=["tdl", "cdl"],
)
def test_native_channel_copies_configuration(
    config_factory: Callable[[], Any],
    channel_factory: Callable[[Any, CudaStream], Any],
    changed_value: int | list[int],
    cuda_stream: CudaStream,
) -> None:
    """An existing channel must not follow later caller configuration changes."""
    config = config_factory()
    channel = channel_factory(config, cuda_stream)
    original_shape = channel.get_rx_signal_out_array().shape

    if isinstance(config, _native.TdlConfig):
        config.n_ue_ant = changed_value
    else:
        config.ue_ant_size = changed_value

    assert channel.get_rx_signal_out_array().shape == original_shape


@pytest.mark.parametrize(
    "config_factory, channel_factory",
    [
        (
            _native.TdlConfig,
            lambda cfg, stream: _native.TdlChan(
                tdl_cfg=cfg, rand_seed=0, stream_handle=stream
            ),
        ),
        (
            _native.CdlConfig,
            lambda cfg, stream: _native.CdlChan(
                cdl_cfg=cfg, rand_seed=0, stream_handle=stream
            ),
        ),
    ],
    ids=["tdl", "cdl"],
)
def test_native_channel_owns_configuration_lifetime(
    config_factory: Callable[[], Any],
    channel_factory: Callable[[Any, CudaStream], Any],
    cuda_stream: CudaStream,
) -> None:
    """A native channel remains usable after its caller configuration is released."""
    def create_channel() -> tuple[
        Any, tuple[int, int, int, int], tuple[int, ...]
    ]:
        config = config_factory()
        channel = channel_factory(config, cuda_stream)
        output_shape = channel.get_rx_signal_out_array().shape
        if isinstance(config, _native.TdlConfig):
            n_tx_ant = config.n_bs_ant
        else:
            n_tx_ant = int(np.prod(config.bs_ant_size))
        input_shape = (
            config.n_cell,
            config.n_ue,
            n_tx_ant,
            config.signal_length_per_ant,
        )
        return channel, input_shape, output_shape

    channel, input_shape, original_shape = create_channel()
    gc.collect()

    # Encourage allocator reuse of the released native configuration storage.
    replacements = [config_factory() for _ in range(64)]
    for config in replacements:
        if isinstance(config, _native.TdlConfig):
            config.n_ue_ant = 1
        else:
            config.ue_ant_size = [1, 1, 1, 1, 1]

    with cuda_stream:
        tx_signal = cp.ones(input_shape, dtype=cp.complex64)
    channel.run(tx_signal_in=_native.CudaArrayComplexFloat(tx_signal))
    rx_signal = cp.asarray(channel.get_rx_signal_out_array())
    cuda_stream.synchronize()

    assert rx_signal.shape == original_shape
    assert bool(cp.all(cp.isfinite(rx_signal)))


@pytest.mark.parametrize(
    "config_factory, channel_factory",
    [
        (
            _native.TdlConfig,
            lambda cfg, stream: _native.TdlChan(
                tdl_cfg=cfg, rand_seed=0, stream_handle=stream
            ),
        ),
        (
            _native.CdlConfig,
            lambda cfg, stream: _native.CdlChan(
                cdl_cfg=cfg, rand_seed=0, stream_handle=stream
            ),
        ),
    ],
    ids=["tdl", "cdl"],
)
def test_native_output_array_retains_channel(
    config_factory: Callable[[], Any],
    channel_factory: Callable[[Any, CudaStream], Any],
    cuda_stream: CudaStream,
) -> None:
    """A borrowed CUDA output view must retain its native channel owner."""
    channel = channel_factory(config_factory(), cuda_stream)
    original_refcount = sys.getrefcount(channel)

    output_array = channel.get_rx_signal_out_array()
    assert sys.getrefcount(channel) == original_refcount + 1

    del output_array
    gc.collect()
    assert sys.getrefcount(channel) == original_refcount


@pytest.mark.parametrize(
    "config_factory, channel_factory",
    [
        (
            _native.TdlConfig,
            lambda cfg, stream: _native.TdlChan(
                tdl_cfg=cfg, rand_seed=0, stream_handle=stream
            ),
        ),
        (
            _native.CdlConfig,
            lambda cfg, stream: _native.CdlChan(
                cdl_cfg=cfg, rand_seed=0, stream_handle=stream
            ),
        ),
    ],
    ids=["tdl", "cdl"],
)
def test_native_channel_retains_stream_owner(
    config_factory: Callable[[], Any],
    channel_factory: Callable[[Any, CudaStream], Any],
    cuda_stream: CudaStream,
) -> None:
    """A native wrapper must keep its CUDA stream alive."""
    original_refcount = sys.getrefcount(cuda_stream)
    channel = channel_factory(config_factory(), cuda_stream)
    assert sys.getrefcount(cuda_stream) == original_refcount + 1

    del channel
    gc.collect()
    assert sys.getrefcount(cuda_stream) == original_refcount


@pytest.mark.parametrize(
    "config_factory, channel_factory",
    [
        (
            _native.TdlConfig,
            lambda cfg, stream: _native.TdlChan(
                tdl_cfg=cfg, rand_seed=0, stream_handle=stream
            ),
        ),
        (
            _native.CdlConfig,
            lambda cfg, stream: _native.CdlChan(
                cdl_cfg=cfg, rand_seed=0, stream_handle=stream
            ),
        ),
    ],
    ids=["tdl", "cdl"],
)
@pytest.mark.parametrize("invalid_layout", ["undersized", "strided"])
def test_native_channel_validates_input_extent_and_layout(
    config_factory: Callable[[], Any],
    channel_factory: Callable[[Any, CudaStream], Any],
    invalid_layout: str,
    cuda_stream: CudaStream,
) -> None:
    """Reject inputs that native channel kernels cannot safely read linearly."""
    config = config_factory()
    n_tx_ant = (
        config.n_bs_ant
        if isinstance(config, _native.TdlConfig)
        else int(np.prod(config.bs_ant_size))
    )
    expected_size = (
        config.n_cell * config.n_ue * n_tx_ant * config.signal_length_per_ant
    )
    channel = channel_factory(config, cuda_stream)

    with cuda_stream:
        if invalid_layout == "undersized":
            tx_signal = cp.empty(expected_size - 1, dtype=cp.complex64)
            expected_message = "expected"
        else:
            storage = cp.empty(expected_size * 2, dtype=cp.complex64)
            tx_signal = storage[::2]
            expected_message = "layout"

    with pytest.raises(ValueError, match=expected_message):
        channel.run(tx_signal_in=_native.CudaArrayComplexFloat(tx_signal))


def test_fading_channel_waits_for_explicit_input_stream() -> None:
    """Order a CuPy producer stream before the channel consumes its output."""
    class TrackingStream(CudaStream):
        """Record synchronization while preserving the real CUDA behavior."""

        def __init__(self) -> None:
            super().__init__()
            self.was_synchronized = False

        def synchronize(self) -> None:
            """Record and perform stream synchronization."""
            self.was_synchronized = True
            super().synchronize()

    producer_stream = TrackingStream()
    channel = FadingChannel(
        channel_config=TdlChannelConfig(
            delay_profile="A",
            delay_spread=30.0,
            n_bs_ant=4,
            n_ue_ant=4,
        ),
        n_sc=1632,
        n_fft=4096,
        n_symbol_slot=14,
        disable_noise=True,
    )
    with producer_stream:
        freq_in = cp.ones((1, 1, 4, 14, 1632), dtype=cp.complex64)

    output = channel.run(
        freq_in=freq_in,
        tti_idx=0,
        snr_db=20.0,
        input_stream=producer_stream,
    )

    assert producer_stream.was_synchronized
    assert bool(cp.all(cp.isfinite(output)))


def freq_out_ref_check(
    n_symbol_slot: int,
    sim_params: list,
    freq_in: np.ndarray,
    freq_out: np.ndarray,
    cfr_sc: np.ndarray,
    enable_swap_tx_rx: bool = False
) -> tuple[float, float]:
    """
    Calculate and return the empirical SNR and signal power.

    Parameters:
        n_symbol_slot: Number of OFDM symbols per slot.
        sim_params: The simulation parameters [n_cell, n_ue, n_bs_ant, n_ue_ant, n_sc].
        freq_in: The input frequency domain signal.
        freq_out: The output frequency domain signal.
        cfr_sc: The channel frequency response (per subcarrier).
        enable_swap_tx_rx: A flag to enable swapping TX and RX.

    Returns:
        Tuple of (empirical SNR in dB, signal power).
    """
    freq_out_ref = np.zeros_like(freq_out)  # same dim with freq_out
    [n_cell, n_ue, n_bs_ant, n_ue_ant, n_sc] = sim_params
    for cell_idx in range(n_cell):
        for ue_idx in range(n_ue):
            for ofdm_sym_idx in range(n_symbol_slot):
                for ue_ant_idx in range(n_ue_ant):
                    for bs_ant_idx in range(n_bs_ant):
                        tmp_chan = cfr_sc[cell_idx][ue_idx][ofdm_sym_idx][ue_ant_idx][bs_ant_idx]
                        if enable_swap_tx_rx:  # UL
                            freq_out_ref[cell_idx][ue_idx][bs_ant_idx][ofdm_sym_idx] += (
                                freq_in[cell_idx][ue_idx][ue_ant_idx][ofdm_sym_idx] * tmp_chan
                            )
                        else:  # DL
                            freq_out_ref[cell_idx][ue_idx][ue_ant_idx][ofdm_sym_idx] += (
                                freq_in[cell_idx][ue_idx][bs_ant_idx][ofdm_sym_idx] * tmp_chan
                            )

    # Calculate noise (difference between noisy and reference signals)
    noise = freq_out - freq_out_ref
    # Calculate signal power (mean squared magnitude of reference signal)
    signal_power = np.mean(np.abs(freq_out_ref)**2)
    # Calculate noise power (mean squared magnitude of noise)
    noise_power = np.mean(np.abs(noise)**2)
    # Calculate SNR in decibels
    snr_db = 1000 if noise_power == 0 else 10 * \
        np.log10(signal_power / noise_power)

    return snr_db, signal_power


def plot_snr_hist(snrs: np.ndarray) -> None:
    """
    Plot CDF of input SNRs, save the plot into a PNG file snr_cdf_plot.png
    """
    # Sort data
    snr_sorted = np.sort(snrs)

    # Compute CDF
    cdf = np.arange(1, len(snr_sorted) + 1) / len(snr_sorted)

    # Plot CDF
    plt.figure(figsize=(8, 6))
    plt.plot(snr_sorted, cdf, marker='.', linestyle='none')
    plt.title("Cumulative Distribution Function (CDF)")
    plt.xlabel("SNR (dB)")
    plt.ylabel("Cumulative Probability")
    plt.grid(True)
    plt.savefig("snr_cdf_plot.png")
    plt.show()


@pytest.mark.parametrize("use_cupy", [False, True], ids=["numpy", "cupy"])
@pytest.mark.parametrize(
    "n_sc, delay_profile, n_tti, channel_type, disable_noise", [
        (1632, 'A', 100, 'tdl', True),
        (1632, 'C', 100, 'tdl', True),
        (3276, 'A', 100, 'tdl', True),
        (3276, 'C', 100, 'tdl', True),
        (1632, 'A', 100, 'tdl', False),
        (1632, 'C', 100, 'tdl', False),
        (3276, 'A', 100, 'tdl', False),
        (3276, 'C', 100, 'tdl', False),
        (1632, 'A', 100, 'cdl', True),
        (1632, 'C', 100, 'cdl', True),
        (3276, 'A', 100, 'cdl', True),
        (3276, 'C', 100, 'cdl', True),
        (1632, 'A', 100, 'cdl', False),
        (1632, 'C', 100, 'cdl', False),
        (3276, 'A', 100, 'cdl', False),
        (3276, 'C', 100, 'cdl', False)
    ]
)
def test_fading_chan(n_sc, delay_profile, n_tti, channel_type, disable_noise, use_cupy, snr_db=10):
    """
    Test the fading channel model with specified parameters.

    - n_sc: number of subcarriers
    - delay_profile: TDL/CDL channel model delay profile (e.g., 'A', 'B', 'C')
    - n_tti: number of TTIs in test
    - channel_type: 'tdl' or 'cdl'
    - disable_noise: disable noise addition for test purpose
    - use_cupy: use CuPy arrays for input (True) or NumPy arrays (False)
    - snr_db: SNR in dB
    """
    try:
        # Get delay spread and Doppler based on profile
        match delay_profile:
            case 'A':
                delay_spread = 30.0
                max_doppler_shift = 10.0
            case 'B':
                delay_spread = 100.0
                max_doppler_shift = 400.0
            case 'C':
                delay_spread = 300.0
                max_doppler_shift = 100.0
            case _:
                raise NotImplementedError(f"Unsupported delay profile: {delay_profile}")

        # Common parameters
        n_symbol_slot = 14
        numerology = 1
        enable_swap_tx_rx = True  # Test uplink

        if channel_type == 'tdl':
            n_bs_ant = 4
            n_ue_ant = 4

            config = TdlChannelConfig(
                delay_profile=delay_profile,
                delay_spread=delay_spread,
                max_doppler_shift=max_doppler_shift,
                n_bs_ant=n_bs_ant,
                n_ue_ant=n_ue_ant,
                cfo_hz=0.0,
                delay=1e-6,
                rand_seed=0
            )
        else:  # cdl
            # CDL antenna configuration
            bs_ant_size = (1, 1, 2, 2, 2)  # (M_g,N_g,M,N,P) = 8 BS antennas
            ue_ant_size = (1, 1, 1, 1, 1)  # (M_g,N_g,M,N,P) = 1 UE antenna
            n_bs_ant = int(np.prod(bs_ant_size))
            n_ue_ant = int(np.prod(ue_ant_size))

            config = CdlChannelConfig(
                delay_profile=delay_profile,
                delay_spread=delay_spread,
                max_doppler_shift=max_doppler_shift,
                bs_ant_size=bs_ant_size,
                ue_ant_size=ue_ant_size,
                ue_ant_polar_angles=(0.0, 90.0),
                ue_ant_pattern=0,
                cfo_hz=0.0,
                delay=1e-6,
                rand_seed=0
            )

        n_cell = config.n_cell
        n_ue = config.n_ue

        # Create FadingChannel
        channel = FadingChannel(
            channel_config=config,
            n_sc=n_sc,
            numerology=numerology,
            n_symbol_slot=n_symbol_slot,
            disable_noise=disable_noise
        )

        # Allocate input buffer
        # For uplink (enable_swap_tx_rx=True): TX from UE antennas
        n_tx_ant = n_ue_ant if enable_swap_tx_rx else n_bs_ant
        freq_data_in_size = [n_cell, n_ue, n_tx_ant, n_symbol_slot, n_sc]

        # Run channel for multiple TTIs
        snr_empirical = np.zeros(n_tti)
        signal_powers = np.zeros(n_tti)
        for tti_idx in range(n_tti):
            # Generate random input data (always generate with NumPy, then convert)
            normalize_factor = 1 / np.sqrt(2 * n_tx_ant)
            freq_in_np = np.empty(freq_data_in_size, dtype=np.complex64)
            freq_in_np.real = np.random.randn(*freq_data_in_size) * normalize_factor
            freq_in_np.imag = np.random.randn(*freq_data_in_size) * normalize_factor

            # Convert to CuPy if needed
            freq_in = cp.asarray(freq_in_np) if use_cupy else freq_in_np

            # Run fading channel
            freq_out = channel.run(
                freq_in=freq_in,
                tti_idx=tti_idx,
                snr_db=snr_db,
                enable_swap_tx_rx=enable_swap_tx_rx
            )
            assert freq_out.size > 0, "freq_out is empty"

            # Convert output to NumPy for reference check
            freq_out_np = freq_out.get() if use_cupy else freq_out

            # Get channel frequency response
            cfr_sc = channel.get_channel_frequency_response(granularity='subcarrier')

            # Calculate empirical SNR and signal power (using NumPy arrays)
            snr_empirical[tti_idx], signal_powers[tti_idx] = freq_out_ref_check(
                n_symbol_slot=n_symbol_slot,
                sim_params=[n_cell, n_ue, n_bs_ant, n_ue_ant, n_sc],
                freq_in=freq_in_np,
                freq_out=freq_out_np,
                cfr_sc=cfr_sc,
                enable_swap_tx_rx=enable_swap_tx_rx
            )

        avg_signal_power = np.mean(signal_powers)

        # Print statistics
        avg_snr = np.mean(snr_empirical)
        min_snr = np.min(snr_empirical)

        if disable_noise:
            # When noise is disabled, output should match reference exactly
            # SNR should be very high (limited by floating point precision)
            min_snr_threshold_db = 50.0
            assert min_snr > min_snr_threshold_db, (
                f"With noise disabled, minimum SNR {min_snr:.2f} dB should be > "
                f"{min_snr_threshold_db} dB (output should match reference exactly)"
            )
        else:
            # Assert empirical SNR matches expected SNR (based on actual signal power)
            expected_snr_db = 10 * np.log10(avg_signal_power) + snr_db
            snr_tolerance_db = 1.0
            assert abs(avg_snr - expected_snr_db) < snr_tolerance_db, (
                f"Average SNR {avg_snr:.2f} dB differs from expected {expected_snr_db:.2f} dB "
                f"by more than {snr_tolerance_db} dB"
            )

    except Exception as e:
        assert False, f"Error running fading channel test: {e}"

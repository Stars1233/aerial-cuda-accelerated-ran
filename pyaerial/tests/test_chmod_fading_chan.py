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

import numpy as np
import cupy as cp
import pytest
import matplotlib.pyplot as plt
from aerial.phy5g.channel_models import (
    FadingChannel,
    TdlChannelConfig,
    CdlChannelConfig,
)


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

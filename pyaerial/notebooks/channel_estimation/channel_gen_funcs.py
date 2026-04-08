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

"""Channel generation utilities for ML channel estimation training and testing.

Provides PyAerialChannelGenerator for generating training data (clean/noisy channel
pairs) and PyAerialChannelEstimateGenerator for generating LS/MMSE estimates from
a full PUSCH transmission through a fading channel.
"""

import math
import numpy as np

from aerial.phy5g.channel_models import FadingChannel, CdlChannelConfig, TdlChannelConfig
from aerial.phy5g.pdsch import PdschTx
from aerial.phy5g.algorithms import ChannelEstimator
from aerial.phy5g.ldpc import get_mcs, random_tb
from aerial.util.cuda import CudaStream


def _ant_count_to_size(n_ant: int) -> tuple:
    """Convert a scalar antenna count to (M_g, N_g, M, N, P).

    Uses dual-polarization (P=2) when n_ant is even, single-pol (P=1) otherwise.
    M and N are chosen so M*N*P == n_ant with M <= N.
    """
    n_ant = max(1, n_ant)
    if n_ant % 2 == 0:
        half = n_ant // 2
        m = int(math.isqrt(half))
        while m > 0:
            if half % m == 0:
                return (1, 1, m, half // m, 2)
            m -= 1
    m = int(math.isqrt(n_ant))
    while m > 0:
        if n_ant % m == 0:
            return (1, 1, m, n_ant // m, 1)
        m -= 1
    return (1, 1, 1, n_ant, 1)


def _make_channel_config(channel_model: str, n_bs_ant: int = None, n_ue_ant: int = None):
    """Create a FadingChannel config from a channel model string.

    Args:
        channel_model: Channel model identifier, e.g. 'CDL-C', 'TDL-A'.
        n_bs_ant: Number of BS antennas (for TDL or CDL). If None, uses defaults.
        n_ue_ant: Number of UE antennas (for TDL or CDL). If None, uses defaults.

    Returns:
        CdlChannelConfig or TdlChannelConfig instance.
    """
    model_upper = channel_model.upper()
    profile = channel_model.split('-')[-1].upper()

    if 'CDL' in model_upper:
        delay_spreads = {'A': 30.0, 'B': 100.0, 'C': 300.0, 'D': 30.0, 'E': 100.0}
        kwargs = dict(
            delay_profile=profile,
            delay_spread=delay_spreads.get(profile, 100.0),
            max_doppler_shift=5.0,
        )
        if n_bs_ant is not None:
            kwargs['bs_ant_size'] = _ant_count_to_size(n_bs_ant)
        if n_ue_ant is not None:
            kwargs['ue_ant_size'] = _ant_count_to_size(n_ue_ant)
        return CdlChannelConfig(**kwargs)
    elif 'TDL' in model_upper:
        delay_spreads = {'A': 30.0, 'B': 100.0, 'C': 300.0}
        return TdlChannelConfig(
            delay_profile=profile,
            delay_spread=delay_spreads.get(profile, 100.0),
            max_doppler_shift=5.0,
            n_bs_ant=n_bs_ant if n_bs_ant is not None else 1,
            n_ue_ant=n_ue_ant if n_ue_ant is not None else 1,
        )
    else:
        raise ValueError(
            f"Unsupported channel model: {channel_model}. Use 'CDL-x' or 'TDL-x'."
        )


class PyAerialChannelGenerator:
    """Channel generator using pyAerial FadingChannel for ML training.

    Generates pairs of (clean_channel, noisy_channel) suitable for training
    denoising-based channel estimation models. Each sample is an independent
    single-OFDM-symbol channel realization for one Tx-Rx antenna pair.
    With multiple Rx antennas the antenna dimension is flattened into the
    batch, so the effective output size is ``batch_size * num_rx_ant``.

    Args:
        num_prbs: Number of PRBs in the UE allocation.
        channel_model: Channel model identifier (e.g. 'CDL-C', 'TDL-A').
        batch_size: Number of independent channel realizations per call.
        num_rx_ant: Number of receive (BS) antennas. Default: 2.
    """

    def __init__(self, num_prbs: int, channel_model: str = 'CDL-C', batch_size: int = 32,
                 num_rx_ant: int = 2):
        """Initialize the channel generator.

        For simplicity, we currently hardcode for single user, single layer,
        and several other parameters like delay spread, link direction, etc.
        """
        self.num_prbs = num_prbs
        self.batch_size = batch_size
        self._num_rx_ant = num_rx_ant
        self.n_sc = num_prbs * 12

        config = _make_channel_config(channel_model, n_bs_ant=num_rx_ant, n_ue_ant=1)
        self.channel = FadingChannel(
            channel_config=config,
            n_sc=self.n_sc,
            numerology=1,           # 30 kHz SCS
            disable_noise=True,     # We add noise manually for controlled SNR
        )

        # Pre-allocate unit probe signal: (n_cell, n_ue, n_tx_ant, n_symbol, n_sc)
        n_bs_ant = self.channel.n_bs_ant
        self._probe = np.ones((1, 1, n_bs_ant, 14, self.n_sc), dtype=np.complex64)

    def gen_channel(self, snr_db: float):
        """Generate a batch of (clean_channel, noisy_channel) pairs.

        Runs the channel model batch_size times, each time generating an
        independent realization. The clean channel is the CFR, and the noisy
        version has AWGN added at the specified SNR.

        Args:
            snr_db: Signal-to-noise ratio in dB.

        Returns:
            Tuple of (h, h_noisy):
                h: Clean channel frequency response,
                    shape (batch_size, num_rx_ant, n_sc).
                h_noisy: Channel with AWGN noise,
                    shape (batch_size, num_rx_ant, n_sc).
        """
        no = 10.0 ** (-float(snr_db) / 10.0)
        std = np.sqrt(no / 2).astype(np.float32)

        n_rx = self._num_rx_ant
        h_all = np.empty((self.batch_size, n_rx, self.n_sc), dtype=np.complex64)
        h_noisy_all = np.empty((self.batch_size, n_rx, self.n_sc), dtype=np.complex64)

        for i in range(self.batch_size):
            # Run channel to generate fading coefficients
            self.channel.run(freq_in=self._probe, tti_idx=0, snr_db=0.0)

            # Get clean channel frequency response
            # Shape: (n_cell, n_ue, n_symbol, n_ue_ant, n_bs_ant, n_sc)
            # Extract all Rx (BS) antennas for UE ant 0 at first OFDM symbol
            cfr = self.channel.get_channel_frequency_response()
            h = cfr[0, 0, 0, 0, :, :]  # (n_rx_ant, n_sc)

            # Add independent AWGN noise per Rx antenna
            noise = std * (
                np.random.randn(n_rx, self.n_sc).astype(np.float32)
                + 1j * np.random.randn(n_rx, self.n_sc).astype(np.float32)
            )

            h_all[i] = h
            h_noisy_all[i] = h + noise

            # Reset for next independent realization
            self.channel.reset()

        return h_all, h_noisy_all


class PyAerialChannelEstimateGenerator:
    """Generator for PyAerial LS/MMSE channel estimates for ML testing.

    Transmits a PUSCH signal through a FadingChannel, then applies PyAerial's
    channel estimators (LS, MS-MMSE) to the received signal. Returns the
    estimates alongside the ground truth channel.

    Args:
        num_prbs: Number of PRBs in the UE allocation.
        channel_model: Channel model identifier (e.g. 'CDL-C', 'TDL-A').
        batch_size: Number of independent channel realizations per call.
        num_rx_ant: Number of receive (BS) antennas. Default: 2.
    """

    def __init__(self, num_prbs: int, channel_model: str = 'CDL-C', batch_size: int = 32,
                 num_rx_ant: int = 2):
        self.num_prbs = num_prbs
        self.num_subcarriers = num_prbs * 12
        self.batch_size = batch_size
        self._num_rx_ant = num_rx_ant

        # DMRS parameters (shared between Tx and channel estimation)
        self.dmrs_params = dict(
            num_ues=1,                    # We simulate only one UE
            slot=0,                       # Slot number
            num_dmrs_cdm_grps_no_data=2,  # Number of DMRS CDM groups without data
            start_prb=0,                  # Start PRB index
            num_prbs=self.num_prbs,       # Number of allocated PRBs
            dmrs_syms=[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Binary list indicating
                                                                   # which symbols are DMRS
            start_sym=2,                  # Start symbol index
            num_symbols=12,               # Number of symbols in the UE group allocation.
            scids=[0],                    # DMRS scrambling ID
            layers=[1],                   # Number of layers per user
            dmrs_ports=[1],               # DMRS port
        )

        # DMRS parameters only used for channel estimation
        self.ch_est_dmrs_params = dict(
            dmrs_scrm_id=41,              # DMRS scrambling ID (ChannelEstimator uses singular form)
            prg_size=1,                   # PRG size
            num_ul_streams=1,             # Number of UL streams
            dmrs_max_len=1,               # 1: single-symbol DMRS. 2: double-symbol
            dmrs_add_ln_pos=0,            # Number of additional DMRS positions.
        )

        # DMRS symbol index
        self.dmrs_idx = self.dmrs_params['dmrs_syms'].index(1)

        # Generate transmit signal, truncated to allocated subcarriers
        self.tx_tensor = self._make_tx_tensor()[:self.num_subcarriers]

        # Create FadingChannel: single Tx antenna, num_rx_ant Rx antennas.
        config = _make_channel_config(channel_model, n_bs_ant=num_rx_ant, n_ue_ant=1)
        self.channel = FadingChannel(
            channel_config=config,
            n_sc=self.num_subcarriers,
            numerology=1,
        )

        # Create PyAerial channel estimators with matching Rx antenna count
        self.ls_estimator = self._create_channel_estimator('LS')
        self.mmse_estimator = self._create_channel_estimator('MS MMSE')

    def _create_channel_estimator(self, estimator_type: str):
        """Create a PyAerial channel estimator.

        Args:
            estimator_type: One of 'MMSE', 'MS MMSE', 'LS'.
        """
        assert estimator_type in ['MMSE', 'MS MMSE', 'LS']
        ch_est_algo = {'MMSE': 0, 'MS MMSE': 1, 'LS': 3}[estimator_type]
        return ChannelEstimator(
            num_rx_ant=self._num_rx_ant,
            ch_est_algo=ch_est_algo,
            cuda_stream=CudaStream()
        )

    def _make_tx_tensor(self, mcs=1, seed=42):
        """Create a PUSCH transmit tensor for channel probing.

        Returns:
            np.ndarray: Transmitted signal, shape (subcarriers, symbols, tx_antennas).
        """
        pusch_tx = PdschTx(cell_id=41, num_rx_ant=1, num_tx_ant=1)

        mod_order, coderate = get_mcs(mcs)

        np.random.seed(seed)
        tb_input = random_tb(
            mod_order=mod_order,
            code_rate=coderate,
            dmrs_syms=self.dmrs_params['dmrs_syms'],
            num_prbs=self.num_prbs,
            start_sym=self.dmrs_params['start_sym'],
            num_symbols=self.dmrs_params['num_symbols'],
            num_layers=self.dmrs_params['layers'][0]
        )

        tx_tensor = pusch_tx.run(
            **self.dmrs_params,
            dmrs_scrm_ids=[41],
            rntis=[1234],
            data_scids=[0],
            tb_inputs=[tb_input],
            code_rates=[coderate],
            mod_orders=[mod_order]
        )  # Output: (subcarriers, symbols, tx_antennas)

        del pusch_tx
        return tx_tensor

    def _apply_channel(self, snr_db: float):
        """Apply fading channel to the transmit tensor.

        Follows the same pattern as example_pusch_simulation.ipynb:
        reshape (n_sc, n_symbol, n_tx_ant) -> (n_cell, n_ue, n_tx_ant, n_symbol, n_sc)
        for FadingChannel input, and reverse for output.

        Args:
            snr_db: Signal-to-noise ratio in dB.

        Returns:
            Tuple of (rx_slot, gt):
                rx_slot: Received signal, shape (n_sc, n_symbol, n_rx_ant).
                gt: Ground truth channel at DMRS symbol for all Rx antennas,
                    shape (n_rx_ant, n_sc).
        """
        # Reshape for FadingChannel: (n_cell, n_ue, n_tx_ant, n_symbol, n_sc)
        # TX has 1 UE antenna; with enable_swap_tx_rx the channel outputs
        # n_bs_ant (= num_rx_ant) receive antennas.
        tx_reshaped = self.tx_tensor.transpose((2, 1, 0))[None, None, ...]

        rx = self.channel.run(
            freq_in=tx_reshaped,
            tti_idx=0,
            snr_db=snr_db,
            enable_swap_tx_rx=True
        )

        # Get ground truth CFR at the DMRS symbol for all Rx antennas
        # Shape: (n_cell, n_ue, n_symbol, n_ue_ant, n_bs_ant, n_sc)
        cfr = self.channel.get_channel_frequency_response()
        gt = cfr[0, 0, self.dmrs_idx, 0, :, :]  # (n_rx_ant, n_sc)

        # Reshape to (n_sc, n_symbol, n_rx_ant)
        rx_slot = rx[0, 0].transpose((2, 1, 0))

        # Reset for next independent realization
        self.channel.reset()

        return rx_slot, gt

    def __call__(self, snr_db: float):
        """Generate LS, MMSE, and ground truth channel estimates.

        For each batch element, transmits the PUSCH signal through a new
        channel realization, then runs PyAerial LS and MS-MMSE estimators.

        Args:
            snr_db: Signal-to-noise ratio in dB.

        Returns:
            Tuple of (ls, mmse, gt):
                ls: LS estimates, shape (batch_size, num_rx_ant, n_sc / 2).
                mmse: MMSE estimates, shape (batch_size, num_rx_ant, n_sc).
                gt: Ground truth channel, shape (batch_size, num_rx_ant, n_sc).
        """
        n_rx = self._num_rx_ant
        n_sc = self.num_subcarriers
        ls = np.zeros((self.batch_size, n_rx, n_sc // 2), dtype=np.complex64)
        mmse = np.zeros((self.batch_size, n_rx, n_sc), dtype=np.complex64)
        gt = np.zeros((self.batch_size, n_rx, n_sc), dtype=np.complex64)

        for b in range(self.batch_size):
            rx_slot, gt_b = self._apply_channel(snr_db)
            gt[b] = gt_b  # (n_rx_ant, n_sc)

            # Run PyAerial channel estimators
            # Estimates have shape (n_rx_ant, n_layers, n_freq, n_time) per UE group.
            est_param = {'rx_slot': rx_slot, **self.dmrs_params, **self.ch_est_dmrs_params}
            ls_est = self.ls_estimator.estimate(**est_param)[0]
            mmse_est = self.mmse_estimator.estimate(**est_param)[0]
            # Extract all Rx antennas, squeeze layer and time dims.
            # LS output is (freq, layer, rx_ant, time) — needs transpose.
            # MMSE output is (rx_ant, layer, freq, time) — already correct.
            ls[b] = ls_est[:, 0, :, 0].T / np.sqrt(2)  # (n_rx_ant, n_freq)
            mmse[b] = mmse_est[:, 0, :, 0]              # (n_rx_ant, n_freq)

        return ls, mmse, gt

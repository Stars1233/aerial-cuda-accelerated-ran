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

"""pyAerial library - fading channel.

This module provides the FadingChannel class for GPU-accelerated fading channel
simulation supporting both TDL and CDL channel models.
"""

# pylint: disable=no-member
import logging
from typing import Generic, Optional

import numpy as np
import cupy as cp  # type: ignore

from aerial import pycuphy
from aerial.util.cuda import CudaStream
from aerial.phy5g.api import Array
from .channel_config import (
    FadingChannelConfig,
    TdlChannelConfig,
    CdlChannelConfig,
)

logger = logging.getLogger(__name__)


class FadingChannel(Generic[Array]):
    """GPU-accelerated fading channel for 5G NR simulations.

    Implements TDL and CDL channel models with OFDM modulation/demodulation and
    AWGN noise addition. Supports both CuPy arrays (zero-copy GPU operation) and
    NumPy arrays (automatic GPU transfer).

    The channel processes frequency-domain input signals and produces frequency-domain
    output signals with applied fading and optional noise.

    Example:
        >>> from aerial.phy5g.channel_models import TdlChannelConfig, FadingChannel
        >>>
        >>> config = TdlChannelConfig(
        ...     delay_profile='A',
        ...     delay_spread=30.0,
        ...     n_bs_ant=4,
        ...     n_ue_ant=2
        ... )
        >>> channel = FadingChannel(
        ...     channel_config=config,
        ...     n_sc=3276,
        ...     numerology=1
        ... )
        >>>
        >>> # Run with CuPy (zero-copy) or NumPy (auto-transfer)
        >>> rx_signal = channel(freq_in=tx_signal, tti_idx=0, snr_db=20.0)

    Args:
        channel_config (FadingChannelConfig): Channel model configuration. Must be either
            TdlChannelConfig or CdlChannelConfig.
        n_sc (int): Number of subcarriers. Default: 3276.
        numerology (int): 5G NR numerology index determining subcarrier spacing
            [3GPP TS 38.211, Sec 4.2]:

            - 0: 15 kHz subcarrier spacing.
            - 1: 30 kHz subcarrier spacing (default).
            - 2: 60 kHz subcarrier spacing.
            - 3: 120 kHz subcarrier spacing.

        n_fft (int): FFT size for OFDM processing. Default: 4096.
        n_symbol_slot (int): Number of OFDM symbols per slot. Default: 14 (normal CP).
        n_sc_prbg (int): Number of subcarriers per PRB group. Default: 48.
        cp_type (int): Cyclic prefix type:

            - 0: Normal CP (default).
            - 1: Extended CP.

        cuda_stream (Optional[CudaStream]): CUDA stream for GPU operations. If None, a new
            CudaStream is created. Use ``with stream:`` to scope work; call
            ``stream.synchronize()`` explicitly when sync is needed. Default: None.
        disable_noise (bool): If True, skip AWGN noise addition. Useful for debugging
            or when noise is added externally. Default: False.
    """

    def __init__(
        self,
        *,
        channel_config: FadingChannelConfig,
        n_sc: int = 3276,
        numerology: int = 1,
        n_fft: int = 4096,
        n_symbol_slot: int = 14,
        n_sc_prbg: int = 48,
        cp_type: int = 0,
        cuda_stream: Optional[CudaStream] = None,
        disable_noise: bool = False
    ) -> None:
        """Initialize FadingChannel.

        Args:
            channel_config: Channel model configuration. Must be either
                TdlChannelConfig or CdlChannelConfig.
            n_sc: Number of subcarriers.
            numerology: 5G NR numerology index (0-3) determining subcarrier spacing.
            n_fft: FFT size for OFDM processing.
            n_symbol_slot: Number of OFDM symbols per slot.
            n_sc_prbg: Number of subcarriers per PRB group.
            cp_type: Cyclic prefix type (0: normal, 1: extended).
            cuda_stream: CUDA stream. If None, a new CudaStream is created.
            disable_noise: If True, skip AWGN noise addition.
        """
        # Validate channel config type
        if not isinstance(channel_config, (TdlChannelConfig, CdlChannelConfig)):
            raise TypeError(
                f"channel_config must be TdlChannelConfig or CdlChannelConfig, "
                f"got {type(channel_config).__name__}"
            )

        # CUDA stream management
        self._stream = CudaStream() if cuda_stream is None else cuda_stream

        # Store configuration
        self.channel_config = channel_config
        self.disable_noise = disable_noise
        self.n_sc = n_sc
        self.numerology = numerology
        self.n_symbol_slot = n_symbol_slot
        self.n_sc_prbg = n_sc_prbg

        # Derived parameters
        self.sc_spacing_hz = 15e3 * (2 ** numerology)
        self.f_samp = n_fft * self.sc_spacing_hz
        self.slot_duration = 1e-3 / (2 ** numerology)

        # Get antenna counts from channel config
        self.n_bs_ant = channel_config.n_bs_ant
        self.n_ue_ant = channel_config.n_ue_ant

        # Determine channel type
        self._is_tdl = isinstance(channel_config, TdlChannelConfig)

        # Create internal pycuphy carrier config
        self._carrier = self._create_carrier_config(
            n_sc=n_sc,
            numerology=numerology,
            n_fft=n_fft,
            n_symbol_slot=n_symbol_slot,
            cp_type=cp_type
        )

        # Convert channel config to pycuphy format
        self._pycuphy_cfg = channel_config._to_pycuphy(
            carrier_f_samp=self.f_samp,
            carrier_n_sc=n_sc,
            carrier_n_sc_prbg=n_sc_prbg,
            carrier_sc_spacing_hz=self.sc_spacing_hz,
            carrier_n_symbol_slot=n_symbol_slot
        )

        # Set signal length and batch info for frequency domain processing
        # Always use frequency domain mode (proc_sig_freq=1)
        # (TdlConfig and CdlConfig from pybind11 both have these attributes)
        self._pycuphy_cfg.signal_length_per_ant = n_symbol_slot * n_sc  # type: ignore[union-attr]
        self._pycuphy_cfg.batch_len = [n_sc] * n_symbol_slot  # type: ignore[union-attr]
        self._pycuphy_cfg.proc_sig_freq = 1  # type: ignore[union-attr]

        # Create channel using new constructor without tx_signal_in
        if self._is_tdl:
            self._channel = pycuphy.TdlChan(
                tdl_cfg=self._pycuphy_cfg,
                rand_seed=channel_config.rand_seed,
                stream_handle=self._stream.handle
            )
        else:
            self._channel = pycuphy.CdlChan(
                cdl_cfg=self._pycuphy_cfg,
                rand_seed=channel_config.rand_seed,
                stream_handle=self._stream.handle
            )

        # Store output shapes for zero-copy view creation
        n_cell = channel_config.n_cell
        n_ue = channel_config.n_ue
        # DL output: (n_cell, n_ue, n_ue_ant, n_symbol, n_sc)
        self._output_shape_dl = (n_cell, n_ue, self.n_ue_ant, n_symbol_slot, n_sc)
        # UL output: (n_cell, n_ue, n_bs_ant, n_symbol, n_sc)
        self._output_shape_ul = (n_cell, n_ue, self.n_bs_ant, n_symbol_slot, n_sc)

        # Create noise adder
        if not disable_noise:
            self._noise_adder = pycuphy.GauNoiseAdder(
                num_threads=1024,
                rand_seed=channel_config.rand_seed,
                stream_handle=self._stream.handle
            )

    def _create_carrier_config(
        self, *, n_sc: int, numerology: int, n_fft: int,
        n_symbol_slot: int, cp_type: int
    ) -> pycuphy.CuphyCarrierPrms:  # type: ignore[name-defined]
        """Create internal pycuphy carrier config."""
        cfg = pycuphy.CuphyCarrierPrms()
        cfg.n_sc = n_sc
        cfg.mu = numerology
        cfg.n_fft = n_fft
        cfg.n_symbol_slot = n_symbol_slot
        cfg.cp_type = cp_type
        cfg.f_samp = int(n_fft * 15e3 * (2 ** numerology))
        # For channel model: layers = antennas (no beamforming)
        cfg.n_bs_layer = self.n_bs_ant
        cfg.n_ue_layer = self.n_ue_ant
        return cfg

    def run(
        self,
        *,
        freq_in: Array,
        tti_idx: int,
        snr_db: float,
        enable_swap_tx_rx: bool = False
    ) -> Array:
        """Run the fading channel on input signal.

        Processes the input frequency-domain signal through the channel model,
        applying fading effects and optionally adding AWGN noise.

        Args:
            freq_in (Array): Frequency-domain input signal. Shape depends on channel
                configuration: (n_cell, n_ue, n_tx_ant, n_symbol, n_sc). Accepts both
                CuPy arrays (zero-copy) and NumPy arrays (auto-transferred to GPU).
            tti_idx (int): Transmission Time Interval index. Used to calculate the
                reference time for time-varying channel coefficients.
            snr_db (float): Signal-to-Noise Ratio in dB for AWGN noise addition.
                Ignored if disable_noise=True.
            enable_swap_tx_rx (bool): Swap transmit and receive roles to simulate
                uplink using downlink channel. Default: False.

        Returns:
            Array: Frequency-domain output signal with fading and noise applied.
                Same type as input (CuPy if input was CuPy, NumPy if input was NumPy).
                Shape: (n_cell, n_ue, n_rx_ant, n_symbol, n_sc).
        """
        # Determine input type
        input_was_numpy = isinstance(freq_in, np.ndarray)

        with self._stream:
            freq_in_gpu = cp.asarray(freq_in, dtype=cp.complex64, order='C')

        # Wrap CuPy array into pycuphy type for C++ binding
        freq_in_cuda = pycuphy.CudaArrayComplexFloat(freq_in_gpu)

        # Calculate reference time
        ref_time = tti_idx * self.slot_duration

        # Run channel
        self._channel.run(
            tx_signal_in=freq_in_cuda,
            ref_time0=ref_time,
            enable_swap_tx_rx=int(enable_swap_tx_rx),
            tx_column_major_ind=0
        )

        # Get output as cuda_array_t, convert to CuPy
        output_array = self._channel.get_rx_signal_out_array(int(enable_swap_tx_rx))

        # Use stream context for CuPy operations
        with self._stream:
            output = cp.array(output_array)

            # Reshape to expected shape (C++ returns [n_cell, n_ue, n_ant, sigLenPerAnt])
            # We need [n_cell, n_ue, n_ant, n_symbol, n_sc]
            output_shape = self._output_shape_ul if enable_swap_tx_rx else self._output_shape_dl
            expected_size = int(np.prod(output_shape))
            if output.size != expected_size:
                raise RuntimeError(
                    f"Output size mismatch: output has {output.size} elements but "
                    f"expected shape {output_shape} requires {expected_size} elements"
                )
            output = output.reshape(output_shape)

            # Add noise if enabled
            if not self.disable_noise:
                self._add_noise(output, snr_db)

        # Return same type as input
        if input_was_numpy:
            return output.get(order='C')
        return output

    def __call__(
        self,
        *,
        freq_in: Array,
        tti_idx: int,
        snr_db: float,
        enable_swap_tx_rx: bool = False
    ) -> Array:
        """Run the fading channel. Alias for run().

        See run() for full documentation.
        """
        return self.run(
            freq_in=freq_in,
            tti_idx=tti_idx,
            snr_db=snr_db,
            enable_swap_tx_rx=enable_swap_tx_rx
        )

    def _add_noise(self, signal_buffer: cp.ndarray, snr_db: float) -> None:
        """Add Gaussian noise to signal buffer in-place on GPU."""
        d_signal = signal_buffer.__cuda_array_interface__['data'][0]
        self._noise_adder.add_noise(
            d_signal=d_signal,
            signal_size=signal_buffer.size,
            snr_db=snr_db
        )

    def reset(self) -> None:
        """Reset the channel state.

        Reinitializes the internal channel state, useful when starting a new
        simulation or when channel coherence time has been exceeded.
        """
        self._channel.reset()

    def get_channel_frequency_response(
        self,
        granularity: str = 'subcarrier'
    ) -> np.ndarray:
        """Get the channel frequency response (CFR).

        Args:
            granularity (str): CFR granularity:

                - 'subcarrier': Per-subcarrier CFR.
                - 'prbg': Per-PRB group CFR.

        Returns:
            np.ndarray: Channel frequency response array.

        Raises:
            ValueError: If granularity is not 'subcarrier' or 'prbg'.
        """
        if granularity not in ('subcarrier', 'prbg'):
            raise ValueError(
                f"granularity must be 'subcarrier' or 'prbg', got '{granularity}'"
            )

        n_cell = self.channel_config.n_cell
        n_ue = self.channel_config.n_ue

        if granularity == 'prbg':
            n_prbg = int(np.ceil(self.n_sc / self.n_sc_prbg))
            cfr = np.empty(
                (n_cell, n_ue, self.n_symbol_slot,
                 self.n_ue_ant, self.n_bs_ant, n_prbg),
                dtype=np.complex64
            )
            self._channel.dump_cfr_prbg(cfr)
        else:  # subcarrier
            cfr = np.empty(
                (n_cell, n_ue, self.n_symbol_slot,
                 self.n_ue_ant, self.n_bs_ant, self.n_sc),
                dtype=np.complex64
            )
            self._channel.dump_cfr_sc(cfr)

        self._stream.synchronize()
        return cfr

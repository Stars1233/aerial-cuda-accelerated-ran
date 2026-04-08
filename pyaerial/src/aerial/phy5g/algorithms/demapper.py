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

"""pyAerial library - QAM modulation mapper and soft demapper."""
from typing import Any
from typing import List

import cupy as cp  # type: ignore[import-untyped]
import numpy as np

from aerial.phy5g.api import Array


class ModulationMapper:
    """QAM modulation mapper and demapper following 3GPP TS 38.211.

    This class provides:
    - Mapping of bits to QAM symbols (modulation)
    - Demapping of symbols to log-likelihood ratios (soft demodulation)

    The demapping algorithm used is the exact log-MAP, which is computationally
    intensive. All processing is done on GPU using CuPy. NumPy inputs are
    automatically converted to CuPy and results are converted back to NumPy.

    Supported modulation orders:
    - 2: QPSK
    - 4: 16-QAM
    - 6: 64-QAM
    - 8: 256-QAM
    """

    def __init__(self, mod_order: int) -> None:
        """Initialize modulation mapper.

        Args:
            mod_order: Modulation order. Supported values: 2, 4, 6, 8.
        """
        if mod_order not in [2, 4, 6, 8]:
            raise ValueError(f"Unsupported modulation order: {mod_order}. "
                             f"Supported values: 2, 4, 6, 8.")
        self.mod_order = mod_order

        # Build constellation array on GPU, indexed by bit pattern integer
        self.constellation = self._build_constellation()

        # Build powers array for bit-to-index conversion (MSB first)
        self._powers = cp.array(
            [2 ** (self.mod_order - 1 - i) for i in range(self.mod_order)],
            dtype=cp.int32
        )

        # Build bit masks for demapping: which constellation indices have bit i = 0 or 1
        num_symbols = 2 ** self.mod_order
        self._zero_masks = []
        self._one_masks = []
        for bit_idx in range(self.mod_order):
            zero_indices = []
            one_indices = []
            for sym_idx in range(num_symbols):
                bit_val = (sym_idx >> (self.mod_order - 1 - bit_idx)) & 1
                if bit_val == 0:
                    zero_indices.append(sym_idx)
                else:
                    one_indices.append(sym_idx)
            self._zero_masks.append(cp.array(zero_indices, dtype=cp.int32))
            self._one_masks.append(cp.array(one_indices, dtype=cp.int32))

    def _build_constellation(self) -> cp.ndarray:
        """Build constellation array indexed by bit pattern integer.

        The constellation follows 3GPP TS 38.211 Gray coding where adjacent
        constellation points differ by only one bit.

        Returns:
            cp.ndarray: Complex constellation points of shape (2^mod_order,) on GPU.
        """
        num_symbols = 2 ** self.mod_order
        constellation = cp.zeros(num_symbols, dtype=cp.complex64)

        for idx in range(num_symbols):
            # Convert index to bit list (MSB first)
            bits = [(idx >> (self.mod_order - 1 - i)) & 1 for i in range(self.mod_order)]
            if self.mod_order == 2:
                constellation[idx] = self._map_qpsk(bits)
            elif self.mod_order == 4:
                constellation[idx] = self._map_16qam(bits)
            elif self.mod_order == 6:
                constellation[idx] = self._map_64qam(bits)
            elif self.mod_order == 8:
                constellation[idx] = self._map_256qam(bits)

        return constellation

    def _map_qpsk(self, bits: List[int]) -> complex:
        """Map bits to QPSK symbols per TS 38.211 Table 5.1.3-1.

        Gray coded: adjacent symbols differ by 1 bit.
        """
        symbol = (1 - 2 * bits[0]) + 1j * (1 - 2 * bits[1])
        symbol /= cp.sqrt(2)
        return symbol

    def _map_16qam(self, bits: List[int]) -> complex:
        """Map bits to 16QAM symbols per TS 38.211 Table 5.1.3-2.

        Gray coded using interleaved bit assignment (b0,b2 for I; b1,b3 for Q).
        """
        symbol = (1 - 2 * bits[0]) * (2 - (1 - 2 * bits[2])) + \
            1j * (1 - 2 * bits[1]) * (2 - (1 - 2 * bits[3]))
        symbol /= cp.sqrt(10)
        return symbol

    def _map_64qam(self, bits: List[int]) -> complex:
        """Map bits to 64QAM symbols per TS 38.211 Table 5.1.3-3.

        Gray coded using interleaved bit assignment.
        """
        symbol = (1 - 2 * bits[0]) * (4 - (1 - 2 * bits[2]) * (2 - (1 - 2 * bits[4]))) + \
            1j * (1 - 2 * bits[1]) * (4 - (1 - 2 * bits[3]) * (2 - (1 - 2 * bits[5])))
        symbol /= cp.sqrt(42)
        return symbol

    def _map_256qam(self, bits: List[int]) -> complex:
        """Map bits to 256QAM symbols per TS 38.211 Table 5.1.3-4.

        Gray coded using interleaved bit assignment.
        """
        symbol = (1 - 2 * bits[0]) * (8 - (1 - 2 * bits[2])) * \
            (4 - (1 - 2 * bits[4]) * (2 - (1 - 2 * bits[6]))) + \
            1j * (1 - 2 * bits[1]) * (8 - (1 - 2 * bits[3])) * \
            (4 - (1 - 2 * bits[5]) * (2 - (1 - 2 * bits[7])))
        symbol /= cp.sqrt(170)
        return symbol

    def map(self, bits: Array) -> Array:
        """Map bits to QAM symbols.

        Args:
            bits: Bit array. Will be flattened and grouped into mod_order bits per symbol.
                  The number of bits must be divisible by mod_order.
                  Can be NumPy or CuPy array.

        Returns:
            Array: Complex symbol array. Same type (NumPy/CuPy) as input.
        """
        cpu_copy = isinstance(bits, np.ndarray)

        # Convert to GPU
        bits_gpu = cp.asarray(bits).flatten()
        num_bits = len(bits_gpu)

        if num_bits % self.mod_order != 0:
            raise ValueError(f"Number of bits ({num_bits}) must be divisible by "
                             f"modulation order ({self.mod_order}).")

        num_symbols = num_bits // self.mod_order

        # Reshape bits into groups of mod_order
        bit_groups = bits_gpu.reshape(num_symbols, self.mod_order)

        # Convert bit groups to constellation indices (MSB first)
        indices = cp.sum(bit_groups.astype(cp.int32) * self._powers, axis=1)

        # Look up symbols from GPU constellation
        symbols = self.constellation[indices]

        if cpu_copy:
            return symbols.get()
        return symbols

    def demap(self, syms: Array, noise_var_inv: Array) -> Array:
        """Run demapping (soft demodulation).

        Args:
            syms: Array of modulation symbols. Can be NumPy or CuPy array.
            noise_var_inv: Inverse of noise variance per subcarrier. The size of this
                array must broadcast with `syms`. Can be NumPy or CuPy array.

        Returns:
            Array: Log-likelihood ratios. The first dimension is modulation order,
            otherwise the dimensions are the same as those of `syms`.
            Same type (NumPy/CuPy) as input.
        """
        cpu_copy = isinstance(syms, np.ndarray)

        # Convert to GPU
        syms_gpu: Any = cp.asarray(syms)
        noise_var_inv_gpu = cp.asarray(noise_var_inv)

        # Reshape constellation for broadcasting with N-dimensional inputs
        # constellation shape: (num_constellation_points,) -> (num_constellation_points, 1, 1, ...)
        num_const = len(self.constellation)
        const_shape = (num_const,) + (1,) * syms_gpu.ndim
        constellation_reshaped = self.constellation.reshape(const_shape)

        # Compute distances to all constellation points
        # Shape: (num_constellation_points, *syms.shape)
        distances_sq = cp.abs(syms_gpu - constellation_reshaped) ** 2

        # Compute exponentials for log-MAP: exp(-|y-s|^2 / sigma^2)
        # where noise_var_inv = 1/sigma^2
        exponents = -distances_sq * noise_var_inv_gpu
        exp_terms = cp.exp(exponents)

        # Compute LLRs for each bit position
        llr = cp.zeros((self.mod_order, *syms_gpu.shape), dtype=cp.float32)

        for bit_idx in range(self.mod_order):
            # Sum exp terms for constellation points where bit = 0
            zero_sum = cp.sum(exp_terms[self._zero_masks[bit_idx], ...], axis=0)
            # Sum exp terms for constellation points where bit = 1
            one_sum = cp.sum(exp_terms[self._one_masks[bit_idx], ...], axis=0)

            # Avoid log(0): epsilon relative to scale for numerical stability
            scale = cp.maximum(cp.maximum(zero_sum, one_sum), 1e-15)
            eps = cp.maximum(1e-15, 1e-7 * scale)
            zero_sum = cp.maximum(zero_sum, eps)
            one_sum = cp.maximum(one_sum, eps)

            llr[bit_idx, ...] = cp.log(zero_sum) - cp.log(one_sum)

        if cpu_copy:
            return llr.get()  # pylint: disable=no-member; llr is cupy.ndarray
        return llr


# Backward compatibility alias
Demapper = ModulationMapper

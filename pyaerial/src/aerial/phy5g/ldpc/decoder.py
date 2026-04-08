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

"""pyAerial library - LDPC decoding."""
from typing import Generic
from typing import List
from typing import Optional

import cupy as cp  # type: ignore
import numpy as np

from aerial import pycuphy  # type: ignore
from aerial.util.cuda import CudaStream
from aerial.phy5g.api import Array
from aerial.phy5g.config import PuschConfig
from aerial.phy5g.ldpc.util import get_code_block_size


class LdpcDecoder(Generic[Array]):
    """LDPC decoder.

    This class supports decoding of LDPC code blocks encoded following TS 38.212. It uses
    cuPHY accelerated LDPC decoding routines under the hood.
    """

    def __init__(self,
                 num_iterations: int = 10,
                 throughput_mode: bool = False,
                 cuda_stream: Optional[CudaStream] = None) -> None:
        """Initialize LdpcDecoder.

        Args:
            num_iterations (int): Number of LDPC decoder iterations. Default: 10.
            throughput_mode (bool): Enable throughput mode. Default: False.
            cuda_stream (Optional[CudaStream]): CUDA stream. If not given, a new CudaStream is
                created. Use ``with stream:`` to scope work; call ``stream.synchronize()``
                explicitly when sync is needed.
        """
        self.num_iterations = num_iterations
        self.throughput_mode = throughput_mode

        self._cuda_stream = CudaStream() if cuda_stream is None else cuda_stream

        # Create cuPHY LDPC decoder object.
        self.pycuphy_ldpc_decoder = pycuphy.LdpcDecoder(self._cuda_stream.handle)

        self.pycuphy_ldpc_decoder.set_num_iterations(num_iterations)
        self.pycuphy_ldpc_decoder.set_throughput_mode(throughput_mode)

        self.cpu_copy = False

    def decode(self,
               *,
               input_llrs: List[Array],
               pusch_configs: List[PuschConfig] = None,
               tb_sizes: List[int] = None,
               code_rates: List[float] = None,
               redundancy_versions: List[int] = None,
               rate_match_lengths: List[int] = None,
               num_iterations: int = None) -> List[Array]:
        """Decode function for LDPC decoder.

        The decoder outputs decoded code blocks which can be further concatenated into
        the received transport block using :class:`~aerial.phy5g.ldpc.crc_check.CrcChecker`.

        The method can be called using either Numpy or CuPy arrays. In case the input arrays
        are located on the GPU (CuPy), the output will be on the GPU (CuPy). So the return type
        shall be the same as used for `input_llrs` when calling the method.

        Args:
            input_llrs (List[Array]): Input LLRs per UE, each array
                is a N x C array of 32-bit floats, N being the number of LLRs per code block
                and C being the number of code blocks.
            pusch_configs (List[PuschConfig]): List of PUSCH configuration objects, one per UE
                group. If this argument is given, the rest are ignored. If not given, all other
                arguments need to be given.
            tb_sizes (List[int]): Transport block size in bits, without CRC, per UE.
            code_rates (List[float]): Target code rates per UE.
            redundancy_versions (List[int]): Redundancy version, i.e. 0, 1, 2, or 3, per UE.
            rate_match_lengths (int): Number of rate matching output bits of each UE.
                This is equal to N.
            num_iterations (int): Number of LDPC iterations. If not given, use the default from
                the constructor.

        Returns:
            List[Array]: The decoded bits.
        """
        self.cpu_copy = isinstance(input_llrs[0], np.ndarray)
        with self._cuda_stream:
            input_llrs = [cp.array(llr, dtype=cp.float16, order='F') for llr in input_llrs]

        # If PuschConfigs given, read the other parameters from that (the rest are ignored).
        if pusch_configs is not None:
            tb_sizes = []
            code_rates = []
            redundancy_versions = []
            rate_match_lengths = []
            for pusch_config in pusch_configs:
                tb_sizes += [ue_config.tb_size * 8 for ue_config in pusch_config.ue_configs]
                code_rates += [ue_config.code_rate / 10240.
                               for ue_config in pusch_config.ue_configs]
                redundancy_versions += [ue_config.rv for ue_config in pusch_config.ue_configs]
                num_data_sym = (np.array(pusch_config.dmrs_syms[
                    pusch_config.start_sym:pusch_config.start_sym + pusch_config.num_symbols
                ]) == 0).sum()
                num_ues = len(pusch_config.ue_configs)
                rate_match_lengths += [
                    num_data_sym * pusch_config.ue_configs[ue].mod_order * pusch_config.num_prbs *
                    12 * pusch_config.ue_configs[ue].layers for ue in range(num_ues)
                ]
        # Make sure all parameters are given in this case.
        else:
            if tb_sizes is None:
                raise ValueError("Argument tb_sizes is not set!")
            if code_rates is None:
                raise ValueError("Argument code_rates is not set!")
            if redundancy_versions is None:
                raise ValueError("Argument redundancy_versions is not set!")
            if rate_match_lengths is None:
                raise ValueError("Argument rate_match_lengths is not set!")

        # Reset the LDPC num of iterations if necessary
        if (num_iterations is not None) and (num_iterations != self.num_iterations):
            self.pycuphy_ldpc_decoder.set_num_iterations(num_iterations)
            self.num_iterations = num_iterations

        # Wrap CuPy arrays into pycuphy types.
        input_llrs = [pycuphy.CudaArrayHalf(llr) for llr in input_llrs]
        decoded_bits_ = self.pycuphy_ldpc_decoder.decode(
            input_llrs,
            [np.uint32(tb_size) for tb_size in tb_sizes],
            [np.float32(code_rate) for code_rate in code_rates],
            [np.uint32(rv) for rv in redundancy_versions],
            [np.uint32(rate_match_len) for rate_match_len in rate_match_lengths]
        )

        with self._cuda_stream:
            decoded_bits_ = [cp.array(elem) for elem in decoded_bits_]
            if self.cpu_copy:
                decoded_bits_ = [elem.get(order='F') for elem in decoded_bits_]

        # Remove the possible padding to maximum code block size.
        decoded_bits = []
        for ue_idx, bits in enumerate(decoded_bits_):
            cb_size = get_code_block_size(tb_sizes[ue_idx], code_rates[ue_idx])
            decoded_bits.append(bits[:cb_size, :])

        return decoded_bits

    def set_num_iterations(self, num_iterations: int) -> None:
        """Set a particular value for the number of iterations to be run.

        Args:
            num_iterations (int): Value of the number of iterations.
        """
        self.num_iterations = num_iterations
        self.pycuphy_ldpc_decoder.set_num_iterations(num_iterations)

    def set_throughput_mode(self, throughput_mode: bool) -> None:
        """Enable throughput mode.

        Args:
            throughput_mode (bool): Enable (True) throughput mode.
        """
        self.throughput_mode = throughput_mode
        self.pycuphy_ldpc_decoder.set_throughput_mode(throughput_mode)

    def get_soft_bits(self) -> List[Array]:
        """Get the soft bit output from the decoder.

        Returns:
            List[Array]: The soft bits in an array.
        """
        soft_bits = self.pycuphy_ldpc_decoder.get_soft_outputs()
        soft_bits = [cp.array(elem) for elem in soft_bits]
        if self.cpu_copy:
            soft_bits = [elem.get(order='F') for elem in soft_bits]

        return soft_bits

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

"""pyAerial library - CRC checking."""
from typing import Generic
from typing import List
from typing import Optional
from typing import Tuple

import cupy as cp  # type: ignore
import numpy as np

from aerial import pycuphy  # type: ignore
from aerial.util.cuda import CudaStream
from aerial.phy5g.api import Array
from aerial.phy5g.config import PuschConfig


class CrcChecker(Generic[Array]):
    """CRC checking.

    This class supports decoding the code block CRCs, desegmenting code blocks together,
    assembling the transport block and also finally decoding the transport block CRCs.
    It uses cuPHY accelerated CRC routines under the hood.
    """

    def __init__(self, cuda_stream: Optional[CudaStream] = None) -> None:
        """Initialize CrcChecker.

        Args:
            cuda_stream (Optional[CudaStream]): CUDA stream. If not given, a new
                CudaStream is created. Use ``with stream:`` to scope work; call
                ``stream.synchronize()`` explicitly when sync is needed.
        """
        self._cuda_stream = CudaStream() if cuda_stream is None else cuda_stream

        self.crc_checker = pycuphy.CrcChecker(self._cuda_stream.handle)

        self.cb_crcs = None  # type: List[Array]

    def check_crc(self,
                  *,
                  input_bits: List[Array],
                  pusch_configs: List[PuschConfig] = None,
                  tb_sizes: List[int] = None,
                  code_rates: List[float] = None) -> Tuple[List[Array], List[Array]]:
        """CRC checking.

        This method takes LDPC decoder output as its input, checks the code block CRCs,
        desegments code blocks, combines them into a transport block and checks the
        transport block CRC. It returns the transport block payloads without CRC, as well
        as the transport block CRC check results. The code block CRC results are stored
        as well and may be queried separately.

        Args:
            input_bits (List[Array]): LDPC decoder outputs per UE,
                each array is a K x C array of 32-bit floats, K being the number of bits per
                code block and C being the number of code blocks.
            pusch_configs (List[PuschConfig]): List of PUSCH configuration objects, one per UE
                group. If this argument is given, the rest are ignored. If not given, all other
                arguments need to be given.
            tb_sizes (List[int]): Transport block size in bits, without CRC, per UE.
            code_rates (List[float]): Target code rates per UE.

        Returns:
            List[Array], List[Array]: A tuple containing:

            - *List[Array]*:
              Transport block payloads in bytes, without CRC, for each UE.

            - *List[Array]*:
              Transport block CRC check results for each UE.
        """
        cpu_copy = isinstance(input_bits[0], np.ndarray)
        with self._cuda_stream:
            input_bits = [cp.array(elem, order='F', dtype=cp.float16) for elem in input_bits]

        max_code_block_size = 8448

        if pusch_configs is not None:
            tb_sizes = []
            code_rates = []
            for pusch_config in pusch_configs:
                tb_sizes += [ue_config.tb_size * 8 for ue_config in pusch_config.ue_configs]
                code_rates += [ue_config.code_rate / 10240.
                               for ue_config in pusch_config.ue_configs]
        # Make sure all parameters are given in this case.
        else:
            if tb_sizes is None:
                raise ValueError("Argument tb_sizes is not set!")
            if code_rates is None:
                raise ValueError("Argument code_rates is not set!")

        # cuPHY wants the LDPC output / CRC input extended to maximum number of info bits K
        # and stacked together.
        with self._cuda_stream:
            tot_num_code_blocks = sum(bits.shape[1] for bits in input_bits)
            crc_input = cp.zeros(
                (max_code_block_size, tot_num_code_blocks),
                dtype=cp.float16,
                order='F'
            )
            idx = 0
            for ue_bits in input_bits:
                crc_input[:ue_bits.shape[0], idx:idx + ue_bits.shape[1]] = \
                    ue_bits.astype(cp.float16)
                idx += ue_bits.shape[1]

        # Wrap CuPy arrays into pycuphy types.
        crc_input = pycuphy.CudaArrayHalf(crc_input)
        tb_payloads = self.crc_checker.check_crc(
            crc_input,
            [np.uint32(tb_size) for tb_size in tb_sizes],
            [np.float32(code_rate) for code_rate in code_rates],
        )

        self.cb_crcs = self.crc_checker.get_cb_crcs()
        tb_crcs = self.crc_checker.get_tb_crcs()

        with self._cuda_stream:
            tb_payloads = [cp.array(elem) for elem in tb_payloads]
            tb_crcs = [cp.array(elem) for elem in tb_crcs]
            self.cb_crcs = [cp.array(elem) for elem in self.cb_crcs]
            if cpu_copy:
                tb_payloads = [elem.get(order='F') for elem in tb_payloads]
                tb_crcs = [elem.get(order='F') for elem in tb_crcs]
                self.cb_crcs = \
                    [elem.get(order='F') for elem in self.cb_crcs]  # type: ignore[union-attr]

        return tb_payloads, tb_crcs

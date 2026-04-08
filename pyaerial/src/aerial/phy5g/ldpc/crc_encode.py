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

"""pyAerial library - CRC encoding."""
from typing import Generic
from typing import List
from typing import Optional

import cupy as cp  # type: ignore
import numpy as np

from aerial import pycuphy  # type: ignore
from aerial.util.cuda import CudaStream
from aerial.phy5g.api import Array
from aerial.phy5g.config import PdschConfig
from aerial.phy5g.ldpc.util import get_pdsch_config_attrs
from aerial.phy5g.ldpc.util import get_pdsch_tb_sizes


class CrcEncoder(Generic[Array]):
    """CRC encoding.

    This class supports computing and attaching transport block CRCs into the input
    transport blocks, segmenting the TB into code blocks and computing and attaching
    code block CRCs into the code blocks, if needed. It uses cuPHY accelerated CRC
    routines under the hood.
    """

    def __init__(self, max_num_tbs: int = 128, cuda_stream: Optional[CudaStream] = None) -> None:
        """initialize the CRC encoder.

        Args:
            max_num_tbs (int): The maximum number of transport blocks.
            cuda_stream (Optional[CudaStream]): CUDA stream. If not given, a new
                CudaStream is created. Use ``with stream:`` to scope work; call
                ``stream.synchronize()`` explicitly when sync is needed.
        """
        self._cuda_stream = CudaStream() if cuda_stream is None else cuda_stream

        self.crc_encoder = pycuphy.CrcEncoder(self._cuda_stream.handle, max_num_tbs)

    def encode(self,
               *,
               tb_inputs: List[Array],
               pdsch_configs: List[PdschConfig] = None,
               tb_sizes: List[int] = None,
               code_rates: List[float] = None) -> List[Array]:
        """Run the CRC encoding.

        The input is a list of transport blocks (TBs) in bytes. For each TB, transport block CRC
        gets computed and attached into the TB. Then, if needed, the transport block
        gets segmented into code blocks (as per 3GPP specifications), and each
        code blocks gets appended with a code block CRC. The output is code blocks.

        Note: If the input data is given as Numpy arrays, the output will be Numpy arrays. If it
        is CuPy arrays, the output will be CuPy arrays, i.e. no copies between host and device
        memory are done in that case.

        The parameters can be given in two ways:
        1. If `pdsch_configs` is given, the rest are ignored.
        2. If `pdsch_configs` is not given, all other arguments need to be given.

        Args:
            tb_inputs (List[Array]): The transport blocks in bytes.
            pdsch_configs (List[PdschConfig]): List of PDSCH configuration objects, one per UE
                group. If this argument is given, the rest are ignored. If not given, all other
                arguments need to be given.
            tb_sizes (List[int]): The size of each transport block in bits.
            code_rates (List[float]): Code rates as float per transport block. Code rate is needed
                to determine LDPC base graph.

        Returns:
            List[Array]: Output code blocks corresponding to each transport block.
        """
        if pdsch_configs is not None:
            # If PDSCH configs given, read the other parameters from that (the rest are ignored).
            pdsch_config_attrs = get_pdsch_config_attrs(pdsch_configs, ["code_rate"])
            code_rates = [c / 10240. for c in pdsch_config_attrs["code_rate"]]
            tb_sizes = get_pdsch_tb_sizes(pdsch_configs)

        # Make sure all parameters are given in this case.
        else:
            if tb_sizes is None:
                raise ValueError("Argument tb_sizes is not set!")
            if code_rates is None:
                raise ValueError("Argument code_rates is not set!")

        # Sanity-check that all per-TB vectors have equal length
        expected = len(tb_inputs)
        # pylint: disable=superfluous-parens
        if not (len(tb_sizes) == len(code_rates) == expected):
            raise ValueError(
                f"Mismatch in list lengths – "
                f"tb_inputs={expected}, tb_sizes={len(tb_sizes)}, "
                f"code_rates={len(code_rates)}"
            )

        cpu_copy = isinstance(tb_inputs[0], np.ndarray)
        with self._cuda_stream:
            tb_inputs = [cp.array(tb_input, order='F', dtype=np.uint8) for tb_input in tb_inputs]
            tb_inputs = cp.concatenate(tb_inputs, axis=0)

        tb_inputs = pycuphy.CudaArrayUint8(tb_inputs)
        crc_outs = self.crc_encoder.encode(tb_inputs, tb_sizes, code_rates)
        num_info_bits = self.crc_encoder.get_num_info_bits()
        with self._cuda_stream:
            for tb_idx, crc_out in enumerate(crc_outs):
                crc_out = cp.array(crc_out, order='F')
                num_cbs = crc_out.shape[1]
                crc_out = cp.unpackbits(crc_out.T, bitorder='little')
                crc_out = crc_out.reshape(-1, num_cbs, order='F')
                crc_out = crc_out.astype(cp.float32)
                crc_out = crc_out[:num_info_bits[tb_idx], :]
                crc_outs[tb_idx] = crc_out

        if cpu_copy:
            crc_outs = [crc_out.get(order='F') for crc_out in crc_outs]

        return crc_outs

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

"""pyAerial library - LDPC encoding."""
from typing import Generic
from typing import List
from typing import Optional

import cuda.bindings.runtime as cudart  # type: ignore
import cupy as cp  # type: ignore
import numpy as np

from aerial import pycuphy  # type: ignore
from aerial.phy5g.api import Array
from aerial.phy5g.config import PdschConfig
from aerial.util.cuda import check_cuda_errors
from aerial.util.cuda import CudaStream
from aerial.phy5g.ldpc.util import get_pdsch_config_attrs
from aerial.phy5g.ldpc.util import get_pdsch_tb_sizes


class LdpcEncoder(Generic[Array]):
    """LDPC encoder.

    This class provides encoding of transmitted transport block bits using LDPC coding
    following TS 38.212. The encoding process is GPU accelerated using cuPHY routines.
    As the input, the transport blocks are assumed to be attached with the CRC and
    segmented to code blocks (as per TS 38.212).
    """

    def __init__(self,
                 puncturing: bool = True,
                 max_num_code_blocks: int = 152,
                 cuda_stream: Optional[CudaStream] = None) -> None:
        """Initialize LdpcEncoder.

        Initialization does all the necessary memory allocations for cuPHY.

        Args:
            puncturing (bool): Whether to puncture the systematic bits (2Zc). Default: True.
            max_num_code_blocks (int): Maximum number of code blocks. Memory is allocated based
                on this. Default: 152.
            cuda_stream (Optional[CudaStream]): CUDA stream. If not given, a new CudaStream is
                created. Use ``with stream:`` to scope work; call ``stream.synchronize()``
                explicitly when sync is needed.
        """
        self._cuda_stream = CudaStream() if cuda_stream is None else cuda_stream

        self.puncturing = puncturing

        # Memory allocation. Allocate maximum possible for a single transport block.
        data_size = 4
        max_lifting_size = 384
        max_output_size = 68 * max_lifting_size * max_num_code_blocks * data_size

        self.output_device_ptr = check_cuda_errors(cudart.cudaMalloc(max_output_size))

        # Create LDPC encoder object.
        self.pycuphy_ldpc_encoder = pycuphy.LdpcEncoder(
            self.output_device_ptr,
            self._cuda_stream.handle
        )

        self.pycuphy_ldpc_encoder.set_puncturing(puncturing)

    def encode(self,
               *,
               code_blocks: List[Array],
               pdsch_configs: List[PdschConfig] = None,
               tb_sizes: List[int] = None,
               code_rates: List[float] = None,
               redundancy_versions: List[int] = None) -> List[Array]:
        """Encode function for LDPC encoder.

        The input to this function is a list of code blocks, one per transport block. Code block
        segmentation is expected to be done before calling this function. Code block segmentation
        can be done using :func:`~aerial.phy5g.ldpc.util.code_block_segment`, or together with
        CRC attachment using :class:`~aerial.phy5g.ldpc.crc_check.CrcChecker`.

        Note: If the input code blocks are given as Numpy arrays, the output will be in Numpy
        arrays. If it is given as CuPy arrays, the output will be in CuPy arrays, i.e. no copies
        between host and device memory are done in that case.

        The parameters can be given in two ways:
        1. If `pdsch_configs` is given, the rest are ignored.
        2. If `pdsch_configs` is not given, all other arguments need to be given.

        Args:
            code_blocks (List[Array]): The input code blocks as a K x C array where K is the
                number of input bits per code block (including CRCs) and C is the number
                of code blocks. One entry per transport block.
            pdsch_configs (List[PdschConfig]): PDSCH configurations. If given, the rest of the
                arguments are ignored.
            tb_sizes (List[int]): Transport block sizes in bits, without CRC.
            code_rates (List[float]): Code rates as float per transport block.
            redundancy_versions (List[int]): Redundancy versions, 0, 1, 2, or 3.

        Returns:
            List[Array]: Encoded bits as a N x C array where N is the number of
                encoded bits per code block. One entry per transport block.
        """
        cpu_copy = isinstance(code_blocks[0], np.ndarray)

        if pdsch_configs is not None:
            # If PDSCH configs given, read the other parameters from that (the rest are ignored).
            attrs = ["code_rate", "rv"]
            pdsch_config_attrs = get_pdsch_config_attrs(pdsch_configs, attrs)
            code_rates = [c / 10240. for c in pdsch_config_attrs["code_rate"]]
            redundancy_versions = pdsch_config_attrs["rv"]
            tb_sizes = get_pdsch_tb_sizes(pdsch_configs)

        # Make sure all parameters are given in this case.
        else:
            if tb_sizes is None:
                raise ValueError("Argument tb_sizes is not set!")
            if code_rates is None:
                raise ValueError("Argument code_rates is not set!")
            if redundancy_versions is None:
                raise ValueError("Argument redundancy_versions is not set!")

        # Sanity-check that all per-TB vectors have equal length
        expected = len(code_blocks)
        # pylint: disable=superfluous-parens
        if not (len(tb_sizes) == len(code_rates) == len(redundancy_versions) == expected):
            raise ValueError(
                f"Mismatch in list lengths – "
                f"code_blocks={expected}, tb_sizes={len(tb_sizes)}, "
                f"code_rates={len(code_rates)}, rvs={len(redundancy_versions)}"
            )

        # For now, we just run the cuPHY LDPC encoder separately for each transport block.
        coded_bits = []
        for tb_code_blocks, tb_size, code_rate, redundancy_version in zip(code_blocks,
                                                                          tb_sizes,
                                                                          code_rates,
                                                                          redundancy_versions):
            with self._cuda_stream:
                cb_size, num_cb = tb_code_blocks.shape
                if np.mod(cb_size, 32):
                    pad = 32 - np.mod(cb_size, 32)
                else:
                    pad = 0

                tb_code_blocks = cp.concatenate(
                    (cp.array(tb_code_blocks.T, order='F', dtype=cp.uint8),
                     cp.zeros((num_cb, pad), order='F', dtype=cp.uint8)),
                    axis=1
                )
                # Pack bits into uint32 for cuPHY.
                tb_code_blocks = cp.packbits(tb_code_blocks.reshape(-1, 8)[:, ::-1])\
                    .view(cp.uint32)\
                    .reshape(num_cb, -1).T

            tb_code_blocks = pycuphy.CudaArrayUint32(tb_code_blocks)
            tb_coded_bits = self.pycuphy_ldpc_encoder.encode(
                tb_code_blocks,
                np.uint32(tb_size),
                np.float32(code_rate),
                int(redundancy_version)
            )

            with self._cuda_stream:
                tb_coded_bits = cp.array(tb_coded_bits, order='F')

                # Unpack bits.
                tb_coded_bits = cp.unpackbits(tb_coded_bits.T.view(cp.uint8))\
                    .reshape(num_cb, -1, 8)[:, :, ::-1]\
                    .reshape(num_cb, -1).T
                tb_coded_bits = \
                    tb_coded_bits[:self.pycuphy_ldpc_encoder.get_cb_size(), :].astype(cp.float32)

            coded_bits.append(tb_coded_bits)

        if cpu_copy:
            coded_bits = [cb.get(order='F') for cb in coded_bits]

        return coded_bits

    def set_puncturing(self, puncturing: bool) -> None:
        """Set puncturing flag.

        Args:
            puncturing (bool): Whether to puncture the systematic bits (2*Zc). Default: True.
        """
        self.puncturing = puncturing
        self.pycuphy_ldpc_encoder.set_puncturing(puncturing)

    def __del__(self) -> None:
        """Destroy function."""
        # Free allocated memory.
        check_cuda_errors(cudart.cudaFree(self.output_device_ptr))

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

"""pyAerial library - DMRS transmitter."""
from typing import Generic
from typing import List
from typing import Optional

import cupy as cp  # type: ignore
import numpy as np

from aerial import pycuphy  # type: ignore
from aerial.phy5g.api import Array
from aerial.phy5g.config import PdschDmrsConfig
from aerial.phy5g.config import PdschConfig
from aerial.util.cuda import CudaStream


class DmrsTx(Generic[Array]):
    """DMRS transmitter.

    This class implements DMRS transmission within a slot. Transmit buffer
    is given as an argument, and gets filled with DMRS REs in correct positions
    based on the given DMRS configuration.
    """

    def __init__(self,
                 num_bwp_prbs: int = 273,
                 max_num_cells: int = 1,
                 max_num_tbs: int = 1,
                 cuda_stream: Optional[CudaStream] = None):
        """Initialize DMRS transmitter.

        Args:
            num_bwp_prbs (int): Number of PRBs in the BWP.
            max_num_cells (int): Maximum number of cells per slot. Memory will be allocated
                for this many cells.
            max_num_tbs (int): Maximum number of transport blocks per cell group. Memory will be
                allocated for this many transport blocks.
            cuda_stream (Optional[CudaStream]): CUDA stream. If not given, a new CudaStream is
                created. Use ``with stream:`` to scope work; call ``stream.synchronize()``
                explicitly when sync is needed.
        """
        self.num_bwp_prbs = num_bwp_prbs
        self._cuda_stream = CudaStream() if cuda_stream is None else cuda_stream

        self.dmrs_tx = pycuphy.DmrsTx(self._cuda_stream.handle, max_num_cells, max_num_tbs)

    def run(self,
            slot: int,
            tx_buffers: List[Array],
            dmrs_params: List[PdschDmrsConfig] = None,
            pdsch_configs: List[PdschConfig] = None) -> List[Array]:
        """Run DMRS transmission.

        The method can be called using either Numpy or CuPy arrays. In case the input arrays
        are located on the GPU (CuPy), the output will be on the GPU (CuPy). So the return type
        shall be the same as used for `tx_buffers` when calling the method.

        The method can be called with either `dmrs_params` or `pdsch_configs`.

        Args:
            slot (int): Slot number.
            tx_buffers (List[Array]): Transmit buffers per cell. The DMRS will be inserted in these
                buffers in the correct locations based on the DMRS parameters.
            dmrs_params (List[PdschDmrsParams]): DMRS parameters, one entry per transport block.
            pdsch_configs (List[PdschConfig]): PDSCH configuration, one entry per UE group.

        Returns:
            List[Array]: Transmit buffers with DMRS inserted into them.
        """
        if dmrs_params is None and pdsch_configs is None:
            raise ValueError("Either dmrs_params or pdsch_configs must be provided.")

        if dmrs_params is not None and pdsch_configs is not None:
            raise ValueError("Only one of dmrs_params or pdsch_configs must be provided.")

        if dmrs_params is None:
            dmrs_params = []
            for pdsch_config in pdsch_configs:
                dmrs_params += PdschDmrsConfig.from_pdsch_config(pdsch_config, self.num_bwp_prbs)

        cpu_copy = isinstance(tx_buffers[0], np.ndarray)
        with self._cuda_stream:
            tx_buffers = [cp.array(tx_buf, order='F', dtype=cp.complex64) for tx_buf in tx_buffers]

        tx_buffers = [pycuphy.CudaArrayComplexFloat(tx_buf) for tx_buf in tx_buffers]
        self.dmrs_tx.run(tx_buffers, slot, dmrs_params)
        with self._cuda_stream:
            tx_buffers = [cp.array(tx_buf) for tx_buf in tx_buffers]
            if cpu_copy:
                tx_buffers = [tx_buf.get(order='F') for tx_buf in tx_buffers]

        return tx_buffers

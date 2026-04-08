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

"""pyAerial library - CSI-RS transmitter."""
from typing import Any
from typing import List
from typing import Optional

import cupy as cp  # type: ignore
import numpy as np

from aerial import pycuphy  # type: ignore
from aerial.phy5g.api import Array
from aerial.phy5g.csirs.csirs_api import CsiRsConfig
from aerial.phy5g.csirs.csirs_api import CsiRsTxConfig
from aerial.phy5g.csirs.csirs_api import CsiRsTxPipeline
from aerial.util.cuda import CudaStream


class CsiRsTx(CsiRsTxPipeline[CsiRsTxConfig, Array]):
    """CSI-RS transmitter.

    This class implements CSI-RS transmission within a slot.
    """
    def __init__(self,
                 num_prb_dl_bwp: List[int],
                 num_ant_dl: Optional[List[int]] = None,
                 cuda_stream: Optional[CudaStream] = None) -> None:
        """Initialize CsiRsTx.

        Args:
            num_prb_dl_bwp (List[int]): Number of PRBs in DL BWP.
            num_ant_dl (List[int]): Number of antennas in DL.
            cuda_stream (Optional[CudaStream]): CUDA stream. If not given, a new CudaStream is
                created. Use ``with stream:`` to scope work; call ``stream.synchronize()``
                explicitly when sync is needed.
        """
        self._cuda_stream = CudaStream() if cuda_stream is None else cuda_stream
        if num_ant_dl is None:
            # Default of max CSI-RS ports; replicate per cell
            num_ant_dl = [32] * len(num_prb_dl_bwp)

        if len(num_prb_dl_bwp) != len(num_ant_dl):
            raise ValueError(
                "num_ant_dl must have the same length as num_prb_dl_bwp "
                f"(got {len(num_ant_dl)} vs {len(num_prb_dl_bwp)})"
            )

        self.csi_rs_tx = pycuphy.CsiRsTx(num_prb_dl_bwp, num_ant_dl)

    def run(self,
            csirs_configs: List[List[CsiRsConfig]],
            tx_buffers: List[Array],
            precoding_matrices: List[np.ndarray] = None) -> List[Array]:
        """Run CSI-RS transmission.

        Fills CSI-RS into the transmit buffers given as input, based on given CSI-RS
        parameters.

        The method can be called using either Numpy or CuPy arrays. In case the input arrays
        are located on the GPU (CuPy), the output will be on the GPU (CuPy). So the return type
        shall be the same as used for `tx_buffers` when calling the method.

        Args:
            csirs_configs (List[List[CsiRsConfig]]): A list of CSI-RS RRC parameters,
                one list per cell. See `CsiRsConfig`.
            tx_buffers (List[Array]): A list of transmit slot buffers, one per cell. These
                represent the slot buffers prior to inserting the CSI-RS.
            precoding_matrices (List[np.ndarray]): A list of precoding matrices. This list
                gets indexed by the `precoding_matrix_index` field in `CsiRsConfig`.

        Returns:
            List[Array]: Transmit buffers for the slot for each cell after inserting CSI-RS.
        """
        precoding_matrices = precoding_matrices or []
        cpu_copy = isinstance(tx_buffers[0], np.ndarray)
        with self._cuda_stream:
            tx_buffers = [cp.array(tx_buf, order='F', dtype=cp.complex64) for tx_buf in tx_buffers]
            tx_buffers = [pycuphy.CudaArrayComplexFloat(tx_buf) for tx_buf in tx_buffers]
        self.csi_rs_tx.run(csirs_configs,
                           precoding_matrices,
                           tx_buffers,
                           self._cuda_stream.handle)
        with self._cuda_stream:
            tx_buffers = [cp.array(tx_buf) for tx_buf in tx_buffers]
            if cpu_copy:
                tx_buffers = [tx_buf.get(order='F') for tx_buf in tx_buffers]

        return tx_buffers

    def __call__(self,
                 config: CsiRsTxConfig,
                 tx_buffers: List[Array],
                 **kwargs: Any) -> List[Array]:
        """Run CSI-RS transmission.

        Note: This implements the base class abstract method.

        Args:
            config (CsiRsTxConfig): CSI-RS transmission configuration.
            tx_buffers (List[Array]): A list of transmit slot buffers, one per cell. These
                represent the slot buffers prior to inserting the CSI-RS.

        Returns:
            List[Array]: Transmit buffers for the slot for each cell after inserting CSI-RS.
        """
        csirs_configs = config.csirs_configs
        precoding_matrices = config.precoding_matrices
        return self.run(csirs_configs, tx_buffers, precoding_matrices)

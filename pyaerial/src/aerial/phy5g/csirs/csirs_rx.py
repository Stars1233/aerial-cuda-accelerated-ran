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

"""pyAerial library - CSI-RS receiver."""
from typing import Any
from typing import List
from typing import Optional

import cupy as cp  # type: ignore
import numpy as np

from aerial import pycuphy  # type: ignore
from aerial.phy5g.api import Array
from aerial.phy5g.csirs.csirs_api import CsiRsConfig
from aerial.phy5g.csirs.csirs_api import CsiRsRxConfig
from aerial.phy5g.csirs.csirs_api import CsiRsRxPipeline
from aerial.util.cuda import CudaStream


class CsiRsRx(CsiRsRxPipeline[CsiRsRxConfig, Array]):
    """CSI-RS receiver.

    This class implements CSI-RS reception within a slot.
    """
    def __init__(self, num_prb_dl_bwp: List[int], cuda_stream: Optional[CudaStream] = None) -> None:
        """Initialize CsiRsRx.

        Args:
            num_prb_dl_bwp (List[int]): Number of PRBs in DL BWP.
            cuda_stream (Optional[CudaStream]): CUDA stream. If not given, a new CudaStream is
                created. Use ``with stream:`` to scope work; call ``stream.synchronize()``
                explicitly when sync is needed.
        """
        self._cuda_stream = CudaStream() if cuda_stream is None else cuda_stream

        self.csi_rs_rx = pycuphy.CsiRsRx(num_prb_dl_bwp)

    def run(self,
            csirs_configs: List[List[CsiRsConfig]],
            rx_data: List[Array],
            ue_cell_association: List[int]) -> List[List[Array]]:
        """Run CSI-RS reception.

        The method can be called using either Numpy or CuPy arrays. In case the input arrays
        are located on the GPU (CuPy), the output will be on the GPU (CuPy). So the return type
        shall be the same as used for `rx_data` when calling the method.

        Args:
            csirs_configs (List[List[CsiRsConfig]]): A list of CSI-RS RRC parameters,
                one list per cell. See `CsiRsConfig`.
            rx_data (List[Array]): A list of received data buffers, one per UE. The Rx data is
                given frequency x time x Rx antenna arrays.
            ue_cell_association (List[int]): Association of UEs to cells. Index of the cell
                per UE, used to index `csirs_configs`.

        Returns:
            List[List[[Array]]: Channel estimation buffers for the slot for each UE.
        """
        cpu_copy = isinstance(rx_data[0], np.ndarray)
        with self._cuda_stream:
            rx_data = [cp.array(buf, order='F', dtype=cp.complex64) for buf in rx_data]
            rx_data = [pycuphy.CudaArrayComplexFloat(buf) for buf in rx_data]
        ch_est = self.csi_rs_rx.run(csirs_configs,
                                    rx_data,
                                    ue_cell_association,
                                    self._cuda_stream.handle)

        with self._cuda_stream:
            ch_est = [[cp.array(buf) for buf in ue_ch_est] for ue_ch_est in ch_est]
            if cpu_copy:
                ch_est = [[buf.get(order='F') for buf in ue_ch_est] for ue_ch_est in ch_est]

        return ch_est

    def __call__(self,
                 rx_data: List[Array],
                 config: CsiRsRxConfig,
                 **kwargs: Any) -> List[List[Array]]:
        """Run CSI-RS reception.

        Note: This implements the base class abstract method.

        Args:
            rx_data (List[Array]): A list of received data buffers, one per UE. The Rx data is
                given frequency x time x Rx antenna arrays.
            config (CsiRsRxConfig): CSI-RS reception configuration.

        Returns:
            List[List[Array]]: Channel estimation buffers for the slot for each UE.
        """
        csirs_configs = config.csirs_configs
        ue_cell_association = config.ue_cell_association
        return self.run(csirs_configs, rx_data, ue_cell_association)

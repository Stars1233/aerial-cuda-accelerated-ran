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

"""pyAerial - SRS transmitter pipeline class definition."""
from typing import Any
from typing import List
from typing import Optional

import cupy as cp  # type: ignore

from aerial import pycuphy  # type: ignore
from aerial.util.cuda import CudaStream
from aerial.phy5g.api import Array
from aerial.phy5g.srs.srs_api import SrsTxConfig
from aerial.phy5g.srs.srs_api import SrsTxPipeline


class SrsTx(SrsTxPipeline[Array, SrsTxConfig]):
    """SRS transmitter pipeline.

    This class implements the sounding reference signal transmission.
    The signals can be generated for multiple UEs with a single API call.
    """

    def __init__(self,
                 num_max_srs_ues: int,
                 num_slot_per_frame: int,
                 num_symb_per_slot: int,
                 cuda_stream: Optional[CudaStream] = None) -> None:
        """Initialize SrsTx.

        Args:
            num_max_srs_ues (int): Maximum number of SRS UEs that this pipeline will
                handle. Memory allocation is based on this number.
            num_slot_per_frame (int): Number of slots per frame.
            num_symb_per_slot (int): Number of symbols in a slot.
            cuda_stream (Optional[CudaStream]): CUDA stream. If not given, a new
                CudaStream is created. Use ``with stream:`` to scope work; call
                ``stream.synchronize()`` explicitly when sync is needed.
        """
        self._cuda_stream = CudaStream() if cuda_stream is None else cuda_stream

        self.srs_tx = pycuphy.SrsTx(num_max_srs_ues,
                                    num_slot_per_frame,
                                    num_symb_per_slot,
                                    self._cuda_stream.handle)

    def __call__(self,
                 config: SrsTxConfig,
                 copy_to_cpu: bool = False,
                 **kwargs: Any) -> List[Array]:
        """Run SRS transmission.

        Note: This implements the base class abstract method.

        Args:
            config (SrsTxConfig): SRS transmission configuration. See `SrsTxConfig`.
            copy_to_cpu (bool): Whether to copy the transmit buffers to host memory
                as Numpy arrays. Default: False.

        Returns:
            List[Array]: The SRS transmit buffers per UE.
        """
        num_tx_ant = [cfg.num_ant_ports for cfg in config.srs_configs]
        tx_buffers = self.srs_tx.run(config.slot, config.frame, config.srs_configs)
        with self._cuda_stream:
            tx_buffers = [cp.array(buf) for buf in tx_buffers]
            tx_buffers = [buf[..., :num_tx_ant[idx]] for idx, buf in enumerate(tx_buffers)]

            if copy_to_cpu:
                tx_buffers = [buf.get(order='F') for buf in tx_buffers]

        return tx_buffers

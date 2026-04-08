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

"""pyAerial library - LDPC rate matching."""
from typing import Generic
from typing import List
from typing import Optional
import math

import cupy as cp  # type: ignore
import numpy as np

from aerial import pycuphy  # type: ignore
from aerial.phy5g.api import Array
from aerial.phy5g.config import PdschConfig
from aerial.phy5g.config import CsiRsConfig
from aerial.util.cuda import CudaStream


class LdpcRateMatch(Generic[Array]):
    """LDPC rate matching.

    This class is used to rate match LDPC code blocks. It is used to rate match the code blocks
    after LDPC encoding. It also supports modulation and layer mapping after rate matching.
    """

    def __init__(self,
                 *,
                 enable_scrambling: bool = True,
                 num_dl_bwp_prbs: int = 273,
                 max_num_code_blocks: int = 152,
                 max_num_tbs: int = 128,
                 cuda_stream: Optional[CudaStream] = None) -> None:
        """Initialize LdpcRateMatch.

        Args:
            enable_scrambling (bool): Whether to enable scrambling after code block concatenation.
                Default: True.
            num_dl_bwp_prbs (int): Number of PRBs in the DL BWP.
            max_num_code_blocks (int): Maximum number of code blocks. Memory will be allocated
                based on this number.
            max_num_tbs (int): Maximum number of transport blocks. Memory will be allocated
                based on this number.
            cuda_stream (Optional[CudaStream]): CUDA stream. If not given, a new CudaStream is
                created. Use ``with stream:`` to scope work; call ``stream.synchronize()``
                explicitly when sync is needed.
        """
        self._cuda_stream = CudaStream() if cuda_stream is None else cuda_stream

        # Create pycuphy LDPC rate match object.
        self.pycuphy_ldpc_rate_match = pycuphy.LdpcRateMatch(
            pycuphy.EnableScrambling(enable_scrambling),
            num_dl_bwp_prbs,
            max_num_tbs,
            max_num_code_blocks,
            self._cuda_stream.handle
        )

    def rate_match(self,
                   *,
                   coded_blocks: List[Array],
                   tb_sizes: List[int],
                   code_rates: List[float],
                   rate_match_lens: List[int],
                   mod_orders: List[int],
                   num_layers: List[int],
                   redundancy_versions: List[int],
                   cinits: List[int]) -> List[Array]:
        """LDPC rate matching function.

        This function does rate matching of LDPC code blocks following TS 38.212. If scrambling
        is enabled, it also scrambles the rate matched bits. In this case the `c_init` value
        needs to be set to an appropriate scrambling sequence initialization value.

        Note: If the input data is given as Numpy arrays, the output will be Numpy arrays. If it
        is in CuPy arrays, the output will be in CuPy arrays, i.e. no copies between host and device
        memory are done in that case.

        Args:
            coded_blocks (List[Array]): Input bits as N x C arrays per UE where N is the number of
                bits per code block and C is the number of code blocks.
            tb_sizes (List[int]): Transport block size in bits without CRC, per UE.
            code_rates (List[float]): Code rate, per UE.
            rate_match_lens (List[int]): Number of rate matching output bits, per UE.
            mod_orders (List[int]): Modulation order, per UE.
            num_layers (List[int]): Number of layers, per UE.
            redundancy_versions (List[int]): Redundancy version, i.e. 0, 1, 2, or 3, per UE.
            cinits (List[int]): The `c_init` value used for initializing scrambling, per UE.

        Returns:
            List[Array]: Rate matched bits, per UE.
        """
        cpu_copy = isinstance(coded_blocks[0], np.ndarray)
        num_tbs = len(coded_blocks)
        coded_blocks_to_cuphy = self._prepare_coded_blocks(coded_blocks)
        coded_blocks_to_cuphy = pycuphy.CudaArrayUint32(coded_blocks_to_cuphy)
        rate_matched_bits = self.pycuphy_ldpc_rate_match.rate_match(
            coded_blocks_to_cuphy,
            tb_sizes,
            code_rates,
            rate_match_lens,
            mod_orders,
            num_layers,
            redundancy_versions,
            cinits
        )

        num_rm_bits = self.pycuphy_ldpc_rate_match.get_num_rm_bits()
        max_num_rm_bits = math.ceil(max(num_rm_bits) / 32) * 32
        max_num_cbs = max(cb.shape[1] for cb in coded_blocks)

        rm_out = []
        with self._cuda_stream:
            rate_matched_bits = cp.array(rate_matched_bits)

            # Unpack bits.
            rate_matched_bits = cp.unpackbits(rate_matched_bits.view(cp.uint8))\
                .reshape(-1, 8)[:, ::-1]\
                .flatten()
            rate_matched_bits = rate_matched_bits.astype(cp.float32)

            cb_idx = 0
            for tb_idx in range(num_tbs):
                bit_index = tb_idx * max_num_cbs * max_num_rm_bits
                num_cb = coded_blocks[tb_idx].shape[1]
                rm_out_ue = []
                for _ in range(num_cb):
                    rm_out_ue += [rate_matched_bits[bit_index:bit_index + num_rm_bits[cb_idx]]]
                    bit_index += max_num_rm_bits
                    cb_idx += 1
                rm_out += [cp.concatenate(rm_out_ue)]

            if cpu_copy:
                rm_out = [rm_out_ue.get(order='F') for rm_out_ue in rm_out]

        return rm_out

    def rm_mod_layer_map(self,
                         *,
                         coded_blocks: List[Array],
                         tx_buffer: Array,
                         pdsch_configs: List[PdschConfig],
                         csi_rs_configs: Optional[List[CsiRsConfig]] = None) -> Array:
        """Rate match, modulate and map symbols to layers.

        This function does rate matching, modulation, and layer mapping of the input bits.
        The output modulation symbols are stored in the `tx_buffer` array, in the correct
        time frequency locations.

        Note: If the input data is given as Numpy arrays, the output will be Numpy arrays. If it
        is in CuPy arrays, the output will be in CuPy arrays, i.e. no copies between host and device
        memory are done in that case.

        Args:
            coded_blocks (List[Array]): Input bits as N x C arrays per UE where N is the number of
                bits per code block and C is the number of code blocks.
            tx_buffer (Array): Output buffer for the modulation symbols.
            pdsch_configs (List[PdschConfig]): PDSCH configuration, per UE group.
            csi_rs_configs (Optional[List[CsiRsConfig]]): Optional parameters for CSI-RS. Note:
                This only leaves the CSI-RS REs empty. To actually add in the CSI-RS signals, one
                needs to call the CSI-RS transmitter separately.

        Returns:
            Array: Transmitted symbols.
        """
        cpu_copy = isinstance(coded_blocks[0], np.ndarray)
        coded_blocks = self._prepare_coded_blocks(coded_blocks)

        with self._cuda_stream:
            tx_buffer = cp.array(tx_buffer, order='F')

        coded_blocks = pycuphy.CudaArrayUint32(coded_blocks)
        tx_buffer = pycuphy.CudaArrayComplexFloat(tx_buffer)
        csi_rs_configs = csi_rs_configs or []
        self.pycuphy_ldpc_rate_match.rm_mod_layer_map(coded_blocks,
                                                      tx_buffer,
                                                      pdsch_configs,
                                                      csi_rs_configs)

        with self._cuda_stream:
            tx_buffer = cp.array(tx_buffer)
            if cpu_copy:
                tx_buffer = tx_buffer.get(order='F')

        return tx_buffer

    def _prepare_coded_blocks(self, coded_blocks: List[Array]) -> cp.ndarray:
        """Prepare coded blocks for rate matching.

        This function prepares the coded blocks for rate matching. It concatenates the code blocks
        and pads them to 32-bit boundaries. It also packs the bits into uint32 for cuPHY.

        Args:
            coded_blocks (List[Array]): Input bits as N x C arrays per UE where N is the number of
                bits per code block and C is the number of code blocks.

        Returns:
            cp.ndarray: Prepared coded blocks in a single CuPy array.
        """
        with self._cuda_stream:
            coded_blocks = [cp.array(cb, order='F', dtype=cp.uint8) for cb in coded_blocks]
            num_tbs = len(coded_blocks)
            max_num_cbs = max(cb.shape[1] for cb in coded_blocks)
            max_cb_size = math.ceil(max(cb.shape[0] for cb in coded_blocks) / 32)
            num_elems = num_tbs * max_num_cbs * max_cb_size

            output = cp.zeros((num_elems,), dtype=cp.uint32)
            idx = 0
            for cbs in coded_blocks:
                cb_size, num_cb = cbs.shape
                if np.mod(cb_size, 32):
                    pad = 32 - np.mod(cb_size, 32)
                else:
                    pad = 0

                cbs = cp.concatenate((cp.array(cbs.T, order='F', dtype=cp.uint8),
                                      cp.zeros((num_cb, pad), order='F', dtype=cp.uint8)),
                                     axis=1)
                # Pack bits into uint32 for cuPHY.
                cbs = cp.packbits(cbs.reshape(-1, 8)[:, ::-1]).view(cp.uint32)

                output[idx:idx + cbs.shape[0]] = cbs
                idx += max_num_cbs * max_cb_size

        return output

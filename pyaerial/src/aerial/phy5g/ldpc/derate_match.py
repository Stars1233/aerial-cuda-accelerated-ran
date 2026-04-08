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

"""pyAerial library - LDPC derate matching."""
from typing import Generic
from typing import List
from typing import Optional

import cupy as cp  # type: ignore
import numpy as np

from aerial import pycuphy  # type: ignore
from aerial.util.cuda import CudaStream
from aerial.phy5g.api import Array
from aerial.phy5g.config import PuschConfig


class LdpcDeRateMatch(Generic[Array]):
    """LDPC derate matching."""

    def __init__(self,
                 enable_scrambling: bool = True,
                 cuda_stream: Optional[CudaStream] = None) -> None:
        """Initialize LdpcDeRateMatch.

        Initialization does all the necessary memory allocations for cuPHY.

        Args:
            enable_scrambling (bool): Whether to descramble the bits before derate matching.
                Default: True.
            cuda_stream (Optional[CudaStream]): CUDA stream. If not given, a new CudaStream is
                created. Use ``with stream:`` to scope work; call ``stream.synchronize()``
                explicitly when sync is needed.
        """
        self._cuda_stream = CudaStream() if cuda_stream is None else cuda_stream

        # Create the cuPHY LDPC derate match object.
        self.pycuphy_ldpc_derate_match = pycuphy.LdpcDerateMatch(
            enable_scrambling,
            self._cuda_stream.handle
        )

    def derate_match(  # pylint: disable=too-many-arguments
            self,
            *,
            input_llrs: List[Array],
            pusch_configs: List[PuschConfig] = None,
            tb_sizes: List[int] = None,
            code_rates: List[float] = None,
            rate_match_lengths: List[int] = None,
            mod_orders: List[int] = None,
            num_layers: List[int] = None,
            redundancy_versions: List[int] = None,
            ndis: List[int] = None,
            cinits: List[int] = None,
            ue_grp_idx: List[int] = None) -> List[Array]:
        """LDPC derate matching function.

        Args:
            input_llrs (List[Array]): Input LLRs as a N x 1 array with dtype
                `np.float32`, where N is the number of LLRs coming from the equalizer. Ordering
                of this input data is `bitsPerQam x numLayers x numSubcarriers x numDataSymbols`.
                One entry per UE group.
            pusch_configs (List[PuschConfig]): List of PUSCH configuration objects, one per UE
                group. If this argument is given, the rest are ignored. If not given, all other
                arguments need to be given.
            tb_sizes (List[int]): Transport block sizes in bits without CRC, per UE.
            code_rates (List[float]): Code rates per UE.
            rate_match_lengths (List[int]): Number of rate matching output bits, the same as N,
                per UE.
            mod_orders (List[int]): Modulation order per UE.
            num_layers (List[int]): Number of layers per UE.
            redundancy_versions (List[int]): Redundancy version, i.e. 0, 1, 2, or 3, per UE.
            ndis (List[int]): New data indicator per UE.
            cinits (List[int]): The `c_init` value used for initializing scrambling for each UE.
            ue_grp_idx (List[int]): The UE group index for each UE. Default is one-to-one mapping.

        Returns:
            List[Array]: Derate matched LLRs for each UE.
        """
        cpu_copy = isinstance(input_llrs[0], np.ndarray)
        with self._cuda_stream:
            input_llrs = [cp.array(llr, order='F', dtype=cp.float16) for llr in input_llrs]

        # If PuschConfigs given, read the other parameters from that (the rest are ignored).
        if pusch_configs is not None:
            tb_sizes = []
            code_rates = []
            mod_orders = []
            num_layers = []
            redundancy_versions = []
            ndis = []
            cinits = []
            rate_match_lengths = []
            ue_grp_idx = []
            for idx, pusch_config in enumerate(pusch_configs):
                tb_sizes += [ue_config.tb_size * 8 for ue_config in pusch_config.ue_configs]
                code_rates += [ue_config.code_rate / 10240.
                               for ue_config in pusch_config.ue_configs]
                mod_orders += [ue_config.mod_order for ue_config in pusch_config.ue_configs]
                num_layers += [ue_config.layers for ue_config in pusch_config.ue_configs]
                redundancy_versions += [ue_config.rv for ue_config in pusch_config.ue_configs]
                ndis += [ue_config.ndi for ue_config in pusch_config.ue_configs]
                cinits += [
                    (ue_config.rnti << 15) + ue_config.data_scid
                    for ue_config in pusch_config.ue_configs
                ]

                num_data_sym = (np.array(pusch_config.dmrs_syms[
                    pusch_config.start_sym:pusch_config.start_sym + pusch_config.num_symbols
                ]) == 0).sum()
                num_ues = len(pusch_config.ue_configs)
                rate_match_lengths += [
                    num_data_sym * pusch_config.ue_configs[ue].mod_order * pusch_config.num_prbs *
                    12 * pusch_config.ue_configs[ue].layers
                    for ue in range(num_ues)
                ]
                ue_grp_idx += [idx for _ in range(num_ues)]

        # Make sure all parameters are given in this case.
        else:
            if tb_sizes is None:
                raise ValueError("Argument tb_sizes is not set!")
            if code_rates is None:
                raise ValueError("Argument code_rates is not set!")
            if rate_match_lengths is None:
                raise ValueError("Argument rate_match_lengths is not set!")
            if mod_orders is None:
                raise ValueError("Argument mod_orders is not set!")
            if num_layers is None:
                raise ValueError("Argument num_layers is not set!")
            if redundancy_versions is None:
                raise ValueError("Argument redundancy_versions is not set!")
            if ndis is None:
                raise ValueError("Argument ndis is not set!")
            if cinits is None:
                raise ValueError("Argument cinits is not set!")
            if ue_grp_idx is None:
                num_ues = len(tb_sizes)
                ue_grp_idx = list(range(num_ues))  # Default is one-to-one mapping.

        # Arrange the input data as cuPHY wants it.
        derate_match_inputs = []
        for idx, input_llr in enumerate(input_llrs):
            # Find first UE corresponding to this UE group.
            ue_idx = next(i for (i, v) in enumerate(ue_grp_idx) if idx == v)

            assert rate_match_lengths[ue_idx] <= (4 * 273 * 12 * 14 * 8), \
                "Maximum rate matching length exceeded!"

            if input_llr.ndim in (1, 2) or (input_llr.ndim == 4 and
                                            input_llr.shape[0] == mod_orders[ue_idx]):
                if input_llr.ndim == 2 and input_llr.shape[1] > 1:
                    raise ValueError("Invalid input data dimensions!")

                # The cuPHY derate matcher pads all inputs to 256-QAM as it supports MU-MIMO, and
                # different users may be different modulation order. We add this padding here.
                with self._cuda_stream:
                    derate_match_input = \
                        cp.zeros(
                            (8, int(rate_match_lengths[ue_idx] / mod_orders[ue_idx])),
                            dtype=cp.float16, order='F'
                        )
                    derate_match_input[:mod_orders[ue_idx], :] = \
                        input_llr.reshape((mod_orders[ue_idx], -1), order='F')

            elif input_llr.ndim != 4 or input_llr.shape[0] != 8:
                raise ValueError("Invalid input data dimensions!")

            else:
                derate_match_input = input_llr

            derate_match_inputs.append(derate_match_input)

        # Wrap CuPy arrays into pycuphy types.
        derate_match_inputs = [pycuphy.CudaArrayHalf(llr) for llr in derate_match_inputs]

        derate_matched_llrs = self.pycuphy_ldpc_derate_match.derate_match(
            derate_match_inputs,
            [np.uint32(tb_size) for tb_size in tb_sizes],
            [np.float32(code_rate) for code_rate in code_rates],
            [np.uint32(rate_match_len) for rate_match_len in rate_match_lengths],
            [np.uint8(mod_order) for mod_order in mod_orders],
            [np.uint8(nl) for nl in num_layers],
            [np.uint32(rv) for rv in redundancy_versions],
            [np.uint32(ndi) for ndi in ndis],
            [np.uint32(cinit) for cinit in cinits],
            [np.uint32(idx) for idx in ue_grp_idx]
        )

        with self._cuda_stream:
            derate_matched_llrs = [cp.array(elem) for elem in derate_matched_llrs]
            if cpu_copy:
                derate_matched_llrs = [elem.get(order='F') for elem in derate_matched_llrs]

        return derate_matched_llrs

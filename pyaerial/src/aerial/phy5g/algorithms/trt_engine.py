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

"""pyAerial library - TensorRT engine."""
from dataclasses import dataclass
from typing import Generic
from typing import List
from typing import Optional

import cupy as cp  # type: ignore
import numpy as np

from aerial import pycuphy  # type: ignore
from aerial.util.cuda import CudaStream
from aerial.phy5g.api import Array
from aerial.pycuphy.types import DataType


@dataclass
class TrtTensorPrms:
    """Class to hold the TRT input and output tensor parameters."""

    # Name of the tensor as in the TRT file.
    name: str

    # Tensor dimensions without batch dimension.
    dims: List[int]

    # Data type. Supported: np.float32, np.int32 / cp.float32, cp.int32
    data_type: type = np.float32

    @property
    def cuphy_data_type(self) -> DataType:
        """Convert data type to cuPHY data type format."""
        if self.data_type in [np.float32, cp.float32]:
            cuphy_data_type = DataType.CUPHY_R_32F
        elif self.data_type in [np.int32, cp.int32]:
            cuphy_data_type = DataType.CUPHY_R_32I
        else:
            raise ValueError(
                "Invalid data type (supported: np.float32, np.int32, cp.float32, cp.int32)"
            )

        return cuphy_data_type


class TrtEngine(Generic[Array]):
    """TensorRT engine class.

    This class implements a simple wrapper around NVIDIA's TensorRT and its
    cuPHY API. It takes a TRT engine file as its input, along with the names
    and dimensions of the input and output tensors. The TRT engine file
    can be generated offline from an `.onnx` file using the `trtexec` tool.
    """

    def __init__(self,
                 *,
                 trt_model_file: str,
                 max_batch_size: int,
                 input_tensors: List[TrtTensorPrms],
                 output_tensors: List[TrtTensorPrms],
                 cuda_stream: Optional[CudaStream] = None) -> None:
        """Initialize TrtEngine.

        Args:
            trt_model_file (str): This is TRT engine (model) file.
            max_batch_size (int): Maximum batch size.
            input_tensors (List[TrtTensorPrms]): A mapping from tensor names to input tensor
                dimensions. The names are strings that must match with those found in the TRT model
                file, and the shapes are iterables of integers. The batch dimension is skipped.
            output_tensors (List[TrtTensorPrms]): A mapping from tensor names to output tensor
                dimensions. The names are strings that must match with those found in the TRT model
                file, and the shapes are iterables of integers. The batch dimension is skipped.
            cuda_stream (Optional[CudaStream]): CUDA stream. If not given, a new CudaStream is
                created. Use ``with stream:`` to scope work; call ``stream.synchronize()``
                explicitly when sync is needed.
        """
        self._cuda_stream = CudaStream() if cuda_stream is None else cuda_stream

        self.input_names = [tensor.name for tensor in input_tensors]
        self.input_dims = [tensor.dims for tensor in input_tensors]
        self.input_cuphy_data_types = [tensor.cuphy_data_type for tensor in input_tensors]
        self.input_data_types = [tensor.data_type for tensor in input_tensors]

        self.output_names = [tensor.name for tensor in output_tensors]
        self.output_dims = [tensor.dims for tensor in output_tensors]
        self.output_cuphy_data_types = [tensor.cuphy_data_type for tensor in output_tensors]
        self.output_data_types = [tensor.data_type for tensor in output_tensors]

        self.trt_engine = pycuphy.TrtEngine(
            trt_model_file,
            max_batch_size,
            self.input_names,
            self.input_dims,
            self.input_cuphy_data_types,
            self.output_names,
            self.output_dims,
            self.output_cuphy_data_types,
            self._cuda_stream.handle
        )

    def run(self, input_tensors: dict[str, Array]) -> dict[str, Array]:
        """Run the TensorRT model.

        This runs the model using NVIDIA TensorRT engine.

        Args:
            input_tensors (dict): A mapping from input tensor names to the actual
                input tensors. The tensor names must match with those given in the initialization,
                and with those found in the TRT model. Actual batch size is read from
                the tensor size. The tensors can be either Numpy or CuPy arrays.

        Returns:
            dict: A mapping from output tensor names to the actual output tensors.
        """
        trt_input = dict()
        # Track if any input is numpy - if so, return numpy outputs
        any_numpy_input = False
        for index, name in enumerate(self.input_names):
            try:
                input_tensor = input_tensors[name]

                if isinstance(input_tensor, np.ndarray):
                    any_numpy_input = True

                if input_tensor.dtype != self.input_data_types[index]:
                    print(
                        f"Warning! Tensor {name} is not of the configured data type, casting..."
                    )
                    input_tensor = input_tensor.astype(self.input_data_types[index])

                with self._cuda_stream:
                    input_tensor = cp.array(input_tensor, order='F')

                # Verify shape.
                input_dims = self.input_dims[index]
                if input_dims != input_tensor.shape[1:]:
                    raise ValueError(f"Tensor {name} has invalid shape!")

                trt_input[name] = input_tensor
            except KeyError:
                print(f"Tensor {name} not found in the inputs!")
                raise

        # Wrap CuPy arrays into pycuphy types.
        for name in self.input_names:
            if trt_input[name].dtype == cp.float32:
                trt_input[name] = pycuphy.CudaArrayFloat(trt_input[name])
            elif trt_input[name].dtype == cp.int32:
                trt_input[name] = pycuphy.CudaArrayInt(trt_input[name])
            else:
                raise ValueError(f"Tensor {name} has invalid data type!")

        trt_output = self.trt_engine.run(trt_input)

        for index, name in enumerate(self.output_names):
            try:
                with self._cuda_stream:
                    trt_output[name] = cp.array(trt_output[name])
                    if any_numpy_input:
                        trt_output[name] = trt_output[name].get(order='F')
                trt_output[name] = trt_output[name].astype(self.output_data_types[index])

            except KeyError:
                print(f"Tensor {name} not found in the outputs!")
                raise

        return trt_output

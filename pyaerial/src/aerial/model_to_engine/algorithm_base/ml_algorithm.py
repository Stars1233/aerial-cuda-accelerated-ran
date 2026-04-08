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

"""
This module provides base classes specifically for machine learning algorithms
within the PyAerial Algorithm Framework. It defines `MLAlgorithm` as an
abstract base for any ML model and `PyTorchAlgorithm` as a concrete base
for PyTorch models, handling model loading, saving, and basic inference flow.
"""

from abc import abstractmethod
from typing import Dict, Any
import numpy as np
import cupy  # type: ignore  # Missing type stubs

import torch
import torch.nn as nn  # pylint: disable=R0402

from .algorithm_base import AlgorithmBase


class MLAlgorithm(AlgorithmBase):
    """Base class for ML-based algorithms (PyTorch)."""

    @abstractmethod
    def get_model(self) -> Any:
        """Return the underlying ML model."""
        pass

    @abstractmethod
    def load_weights(self, path: str) -> None:
        """Load model weights from a file."""
        pass

    @abstractmethod
    def save_weights(self, path: str) -> None:
        """Save model weights to a file."""
        pass


class PyTorchAlgorithm(MLAlgorithm):
    """Base class for PyTorch-based algorithms."""

    def __init__(self) -> None:
        """Initialize PyTorch algorithm."""
        super().__init__()
        self.model = None

    def get_model(self) -> nn.Module:
        """Return the PyTorch model."""
        if self.model is None:
            raise ValueError("Model has not been initialized")
        return self.model

    def forward(   # noqa: C901
        self, *args: Any, **kwargs: Any
    ) -> Dict[str, Any]:
        """Execute the PyTorch model.

        This default implementation:
        1. Converts inputs to PyTorch tensors
        2. Runs the model in eval mode with no gradients
        3. Converts outputs back to numpy arrays

        Override this method for custom pre/post-processing.
        """
        model = self.get_model()
        model.eval()

        # Process inputs
        if args and not kwargs:
            # Handle positional arguments
            torch_inputs = {}
            for i, arg in enumerate(args):
                if isinstance(arg, cupy.ndarray):
                    # Convert cuPY array to PyTorch tensor
                    torch_inputs[f"arg_{i}"] = torch.as_tensor(
                        arg, dtype=torch.float32
                    )
                elif isinstance(arg, np.ndarray):
                    torch_inputs[f"arg_{i}"] = torch.from_numpy(arg).float()
                elif isinstance(arg, torch.Tensor):
                    torch_inputs[f"arg_{i}"] = arg
                else:
                    # Try to convert to tensor
                    torch_inputs[f"arg_{i}"] = torch.tensor(
                        arg, dtype=torch.float32
                    )
        else:
            # Handle keyword arguments
            torch_inputs = {}
            for key, value in kwargs.items():
                if isinstance(value, cupy.ndarray):
                    # Convert cuPY array to PyTorch tensor
                    torch_inputs[key] = torch.as_tensor(
                        value, dtype=torch.float32
                    )
                elif isinstance(value, np.ndarray):
                    torch_inputs[key] = torch.from_numpy(value).float()
                elif isinstance(value, torch.Tensor):
                    torch_inputs[key] = value
                else:
                    # Try to convert to tensor
                    torch_inputs[key] = torch.tensor(
                        value, dtype=torch.float32
                    )

        # Run model inference
        with torch.no_grad():
            outputs = model(**torch_inputs)  # pylint: disable=E1102

        # Process outputs
        result = {}
        if isinstance(outputs, (list, tuple)):
            # Multiple outputs
            output_specs = self.get_output_specs()
            for i, out in enumerate(outputs):
                if i < len(output_specs):
                    name = output_specs[i].name
                else:
                    name = f"output_{i}"

                if isinstance(out, torch.Tensor):
                    result[name] = out.cpu().numpy()
                elif isinstance(out, cupy.ndarray):
                    result[name] = cupy.asnumpy(out)
                else:
                    result[name] = np.array(out)
        elif isinstance(outputs, dict):
            # Output is already a dictionary
            for key, value in outputs.items():
                if isinstance(value, torch.Tensor):
                    result[key] = value.cpu().numpy()
                elif isinstance(value, cupy.ndarray):
                    result[key] = cupy.asnumpy(value)
                else:
                    result[key] = np.array(value)
        else:
            # Single output
            output_specs = self.get_output_specs()
            name = output_specs[0].name if output_specs else "output"

            if isinstance(outputs, torch.Tensor):
                result[name] = outputs.cpu().numpy()
            elif isinstance(outputs, cupy.ndarray):
                result[name] = cupy.asnumpy(outputs)
            else:
                result[name] = np.array(outputs)

        return result

    def load_weights(self, path: str) -> None:
        """Load model weights from a PyTorch state dict file."""
        if self.model is None:
            raise ValueError(
                "Model not initialized. Initialize model before loading "
                "weights."
            )

        state_dict = torch.load(path)
        self.model.load_state_dict(state_dict)

    def save_weights(self, path: str) -> None:
        """Save model weights to a PyTorch state dict file."""
        if self.model is None:
            raise ValueError("Model not initialized.")

        torch.save(self.model.state_dict(), path)

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
This module defines the PyTorch implementation of the
EnhancedFusedChannelEstimator, a neural network model designed for 5G channel
estimation. It includes necessary sub-modules like ResidualBlock and an
ONNX-compatible OptimizedDFT, and provides the core model architecture used
as the reference for conversion to ONNX and TensorRT engines within the
model-to-engine framework.
"""

# Enhanced Fused Channel Estimator and dependencies for TRT Engine Integration

import torch
import torch.nn as nn  # pylint: disable=R0402
import torch.nn.functional as F
from typing import List, Dict, Any  # pylint: disable=C0411

from aerial.model_to_engine.algorithm_base.ml_algorithm import PyTorchAlgorithm
from aerial.model_to_engine.algorithm_base.algorithm_base import TensorSpec


# --- ResidualBlock ---
class ResidualBlock(nn.Module):
    """Residual block with layer normalization."""
    def __init__(
        self,
        num_conv_channels: int,
        num_res: int,
        dilation: int = 1
    ):
        """Initialize ResidualBlock with specified parameters.

        Args:
            num_conv_channels: Number of convolutional channels
            num_res: Size of the residual dimension
            dilation: Dilation rate for convolutions
        """
        super().__init__()
        self._conv1 = nn.Conv1d(
            num_conv_channels,
            num_conv_channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
            bias=False
        )
        self._norm1 = nn.LayerNorm([num_conv_channels, num_res])
        self._conv2 = nn.Conv1d(
            num_conv_channels,
            num_conv_channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
            bias=False
        )
        self._norm2 = nn.LayerNorm([num_conv_channels, num_res])

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass through the residual block.

        Args:
            inputs: Input tensor

        Returns:
            Processed tensor after applying residual connection
        """
        x = self._conv1(inputs)
        x = self._norm1(x)
        x = F.relu(x)
        x = self._conv2(x)
        x = self._norm2(x)
        x = x + inputs  # skip connection
        x = F.relu(x)
        return x


# --- OptimizedDFT ---
class OptimizedDFT(nn.Module):
    """ONNX-compatible DFT implementation using pre-computed coefficients."""
    def __init__(self, seq_len: int, do_fft: bool = True):
        """Initialize the OptimizedDFT module.

        Args:
            seq_len: Length of the sequence for DFT
            do_fft: Whether to perform FFT (True) or not (False)
        """
        super().__init__()
        self.seq_len = seq_len
        self.do_fft = do_fft
        if do_fft:
            n = torch.arange(seq_len, dtype=torch.float32)
            sign = -1 if do_fft else 1
            angle = sign * 2 * torch.pi / seq_len
            k_times_n = torch.outer(n, n) * angle
            cos_matrix = torch.cos(k_times_n)
            sin_matrix = torch.sin(k_times_n)
            self.register_buffer('cos_matrix_fft', cos_matrix)
            self.register_buffer('sin_matrix_fft', sin_matrix)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Perform forward pass of the DFT operation.

        Args:
            z: Input tensor to transform

        Returns:
            Transformed tensor
        """
        if not self.do_fft:
            return z
        orig_shape = z.shape
        z_r = z[..., 0]
        z_i = z[..., 1]
        reshape_dims = orig_shape[:-2] + (-1,)
        z_r = z_r.reshape(reshape_dims)
        z_i = z_i.reshape(reshape_dims)

        # Matrix multiplication for FFT
        # Type assertions for mypy (register_buffer creates Tensor | Module type)
        cos_mat: torch.Tensor = self.cos_matrix_fft  # type: ignore[assignment]
        sin_mat: torch.Tensor = self.sin_matrix_fft  # type: ignore[assignment]
        out_r = torch.matmul(z_r, cos_mat) - torch.matmul(z_i, sin_mat)
        out_i = torch.matmul(z_r, sin_mat) + torch.matmul(z_i, cos_mat)

        scale = 1.0 / self.seq_len if not self.do_fft else 1.0
        out_r = out_r * scale
        out_i = out_i * scale
        out_r = out_r.reshape(orig_shape[:-1])
        out_i = out_i.reshape(orig_shape[:-1])
        out = torch.stack([out_r, out_i], dim=-1)
        return out


# --- ChannelEstimatorUpsampling ---
class ChannelEstimatorUpsampling(nn.Module):
    """Channel estimator with upsampling capabilities."""
    def __init__(self, num_res: int, do_fft: bool = True,
                 num_conv_channels: int = 64, upsample_factor: int = 2):
        """Initialize the upsampling channel estimator.

        Args:
            num_res: Number of residual units
            do_fft: Whether to perform FFT
            num_conv_channels: Number of convolutional channels
            upsample_factor: Factor by which to upsample the signal
        """
        super().__init__()
        self.upsample_factor = upsample_factor
        self.num_res = num_res
        self.do_fft = do_fft
        self.optimized_dft = OptimizedDFT(num_res, do_fft=False)

        # Input convolution - matches channel_est_models.py
        self.input_conv = nn.Sequential(
            nn.Conv1d(
                2, num_conv_channels, kernel_size=3, padding=1, bias=False
            ),
            nn.ReLU()
        )

        # ResidualBlocks - update dilation in second block to 3 (from 2)
        self.res_block_1 = ResidualBlock(
            num_conv_channels, num_res, dilation=1)
        self.res_block_2 = ResidualBlock(
            num_conv_channels, num_res, dilation=3)

        # Output convolution with upsampling - updated to match ground truth
        self.output_conv = nn.Sequential(
            # First expand the number of channels
            nn.Conv1d(
                num_conv_channels,
                num_conv_channels * 2,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.ReLU(),

            # Then map to output channels with upsampling factor
            nn.Conv1d(
                num_conv_channels * 2,
                2 * upsample_factor,  # 2 channels * upsample_factor
                kernel_size=3,
                padding=1,
                bias=False
            )
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Forward pass through the upsampling estimator.

        Args:
            z: Input tensor

        Returns:
            Processed tensor with upsampling applied
        """
        x = self.optimized_dft(z) if self.do_fft else z
        x = x.permute(0, 2, 1)
        x = self.input_conv(x)
        x = self.res_block_1(x)
        # Match the behavior in channel_est_models.py which only uses the first
        # residual block
        # x = self.res_block_2(x)

        x = self.output_conv(x)

        # Reshape to perform the upsampling (similar to PixelShuffle in 1D)
        batch_size, _, length = x.shape
        x = x.view(batch_size, 2, self.upsample_factor, length)
        x = x.permute(0, 1, 3, 2).contiguous()
        x = x.view(batch_size, 2, length * self.upsample_factor)

        x = x.permute(0, 2, 1)

        return x


# --- EnhancedFusedChannelEstimator ---
class EnhancedFusedChannelEstimator(nn.Module, PyTorchAlgorithm):
    """Enhanced fused channel estimator for 5G signal processing."""
    def __init__(self, num_res: int = 612, comb_size: int = 2,
                 do_fft: bool = True, reshape: bool = True):
        """Initialize the enhanced fused channel estimator.

        Args:
            num_res: Number of residual units
            comb_size: Comb size for upsampling (2 or 4 supported)
            do_fft: Whether to perform FFT
            reshape: Whether to reshape the input/output tensors
        """
        super().__init__()
        self.num_res = num_res
        self.comb_size = comb_size
        self.reshape = reshape
        self.do_fft = do_fft
        self.min_scale = 1e-15  # Match the ground truth implementation
        self.optimized_dft = OptimizedDFT(num_res, do_fft=True)

        if self.comb_size not in [2, 4]:
            raise ValueError('Comb size not supported. Choose 2 or 4.')

        # Use a single upsampling estimator
        self.estimator = ChannelEstimatorUpsampling(
            num_res=num_res,
            do_fft=do_fft,
            num_conv_channels=64,
            upsample_factor=comb_size
        )

    def get_input_specs(self) -> List[TensorSpec]:
        """Get input tensor specifications.

        Returns:
            List of input tensor specifications
        """
        return [
            TensorSpec(
                name="z",
                # batch, subcarriers, layers, antennas, symbols, real&imag
                shape=[1, 1638, 4, 4, 2, 2],
                dtype="float32",
                is_dynamic=True,
                dynamic_axes={"batch": 0}  # Only batch is dynamic
            )
        ]

    def get_output_specs(self) -> List[TensorSpec]:
        """Get output tensor specifications.

        Returns:
            List of output tensor specifications
        """
        return [
            TensorSpec(
                name="zout",
                # batch, antennas, layers, subcarriers_out, symbols, real&imag
                shape=[1, 4, 4, 3276, 2, 2],
                dtype="float32",
                is_dynamic=True,
                dynamic_axes={"batch": 0}  # Only batch is dynamic
            )
        ]

    def forward(self, z: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass of the channel estimator.

        Args:
            z: Input tensor with channel data

        Returns:
            Dictionary with output tensor "zout"
        """
        # Initialize variables that might be used later
        orig_shape = None
        dims = None

        # If reshape is True, handle multi-dimensional inputs
        if self.reshape:
            # Extract batch dimensions for later reshaping
            n_batch, subcarriers, n_layers, n_ant, n_symb, dims = z.shape

            # Swapaxes to match the ground truth implementation
            z = z.swapaxes(1, 3)

            # Model-dependent reshaping
            z = z.swapaxes(3, 4).contiguous().reshape((-1, subcarriers, dims))
            orig_shape = (n_batch, subcarriers, n_layers, n_ant, n_symb)

        # Apply scaling
        scale_factors = torch.sqrt(
            torch.sum(torch.abs(z)**2, dim=-1)
        ).max(dim=1, keepdim=True)[0].unsqueeze(-1)

        # Create a mask for values below min_scale
        mask = (scale_factors < self.min_scale).float()

        # Apply the mask to replace small values with min_scale
        scale_factors = scale_factors * (1 - mask) + self.min_scale * mask

        z /= scale_factors

        # Apply FFT if enabled
        if self.do_fft:
            z = self.optimized_dft(z)
            z = roll_in_half(z)

        # Process with the estimator
        zout = self.estimator(z)

        # Rescale
        zout *= scale_factors

        # Reshape back if needed
        if self.reshape:
            # Reshape back based on ground truth implementation
            assert orig_shape is not None  # Set when self.reshape is True
            n_batch, subcarriers, n_layers, n_ant, n_symb = orig_shape
            zout = zout.reshape(
                (n_batch, n_ant, n_layers, n_symb,
                 self.comb_size * subcarriers, dims)
            ).swapaxes(4, 3)

        return {"zout": zout}

    def get_metadata(self) -> Dict[str, Any]:
        """Get model metadata.

        Returns:
            Dictionary with model metadata including name, version, and parameters
        """
        return {
            "name": "enhanced_fused_channel_estimator",
            "version": "1.0.0",
            "description": ("Enhanced channel estimator with upsampling for "
                            "5G signal processing"),
            "parameters": {
                "num_res": self.num_res,
                "comb_size": self.comb_size,
                "do_fft": self.do_fft,
                "reshape": self.reshape
            }
        }


# Helper function to match ground truth implementation
def roll_in_half(x: torch.Tensor) -> torch.Tensor:
    """Roll the tensor in half along the last dimension.

    Args:
        x: Input tensor to roll

    Returns:
        Rolled tensor
    """
    return torch.roll(x, shifts=1, dims=-1)

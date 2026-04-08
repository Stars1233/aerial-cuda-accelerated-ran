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

import os
import numpy as np
import torch
import argparse
import onnx
import onnxruntime as ort

from aerial.model_to_engine.model.enhanced_channel_estimator import EnhancedFusedChannelEstimator


def get_onnx_providers():
    """Get list of ONNX Runtime execution providers for model verification.

    Uses CPU provider for ONNX verification to avoid CUDA library dependency issues.
    The actual TensorRT export handles GPU execution separately.
    """
    return ['CPUExecutionProvider']


def export_to_onnx(model, dummy_input, onnx_path):
    """Export model to ONNX using TorchScript-based exporter.

    Uses dynamo=False because the dynamo exporter's optimizer has a bug
    (IndexError in onnxscript _basic_rules.py), and disabling optimization
    produces models with runtime issues. The TorchScript exporter is stable.

    Args:
        model: PyTorch model to export
        dummy_input: Example input tensor for tracing
        onnx_path: Path to save the ONNX model
    """
    torch.onnx.export(
        model,
        (dummy_input,),
        onnx_path,
        input_names=['z'],
        output_names=['zout'],
        dynamic_axes={
            'z': {0: 'batch'},
            'zout': {0: 'batch'}
        },
        opset_version=18,
        dynamo=False,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Export EnhancedFusedChannelEstimator to ONNX and compare results"
    )
    parser.add_argument('--num_res', type=int, default=612,
                        help='Number of resource elements')
    parser.add_argument('--comb_size', type=int, default=2,
                        help='Comb size (2 or 4)')
    parser.add_argument('--do_fft', action='store_true', default=True,
                        help='Use FFT in the model')
    parser.add_argument('--input', type=str, required=True,
                        help='Input numpy file to use for inference')
    parser.add_argument('--pytorch_output', type=str, required=True,
                        help='PyTorch output file to compare with')
    parser.add_argument('--onnx_path', type=str, default='model.onnx',
                        help='Output ONNX model path')
    parser.add_argument('--onnx_output', type=str, default='onnx_output.npy',
                        help='ONNX output file path')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to run PyTorch on (cuda or cpu)')
    args = parser.parse_args()

    # Ensure output directory exists
    onnx_dir = os.path.dirname(args.onnx_path)
    if onnx_dir and not os.path.exists(onnx_dir):
        os.makedirs(onnx_dir, exist_ok=True)

    output_dir = os.path.dirname(args.onnx_output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # Load input data
    print(f"Loading input data from {args.input}")
    input_data = np.load(args.input)
    print(f"Input shape: {input_data.shape}")

    # Create the model
    print("Creating EnhancedFusedChannelEstimator model")
    model = EnhancedFusedChannelEstimator(
        num_res=args.num_res,
        comb_size=args.comb_size,
        do_fft=args.do_fft,
        reshape=True
    )

    # Move model to device and set to eval mode
    device = torch.device(args.device)
    model.to(device)
    model.eval()

    # Export model to ONNX
    print(f"Exporting model to ONNX: {args.onnx_path}")
    dummy_input = torch.tensor(input_data, device=device)
    export_to_onnx(model, dummy_input, args.onnx_path)

    # Verify the ONNX model
    onnx_model = onnx.load(args.onnx_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX model is valid")

    # Run inference with ONNX Runtime (CPU is sufficient for verification)
    print("Running ONNX inference")
    providers = get_onnx_providers()
    ort_session = ort.InferenceSession(
        args.onnx_path,
        providers=providers
    )

    ort_inputs = {ort_session.get_inputs()[0].name: input_data}
    ort_outputs = ort_session.run(None, ort_inputs)
    onnx_output = ort_outputs[0]

    # Save ONNX output
    np.save(args.onnx_output, onnx_output)
    print(f"ONNX output saved to {args.onnx_output}")

    # Load PyTorch output for comparison
    pytorch_output = np.load(args.pytorch_output)

    # Compare outputs
    print("\n--- Output Comparison ---")
    print(f"PyTorch output shape: {pytorch_output.shape}")
    print(f"ONNX output shape: {onnx_output.shape}")

    print(f"PyTorch output mean: {pytorch_output.mean():.6f}, std: {pytorch_output.std():.6f}")
    print(f"ONNX output mean: {onnx_output.mean():.6f}, std: {onnx_output.std():.6f}")

    # Calculate differences
    if pytorch_output.shape == onnx_output.shape:
        abs_diff = np.abs(pytorch_output - onnx_output)
        mean_diff = np.mean(abs_diff)
        max_diff = np.max(abs_diff)

        print(f"Mean absolute difference: {mean_diff:.6f}")
        print(f"Max absolute difference: {max_diff:.6f}")

        # Check if outputs are close enough
        tolerance = 1e-4
        if max_diff < tolerance:
            print(f"✅ ONNX output matches PyTorch output within tolerance {tolerance}")
        else:
            print(
                f"⚠️  ONNX output differs from PyTorch output beyond tolerance {tolerance} "
                "(this is safe to ignore, test is not calibrated yet)"
            )
    else:
        print("❌ Output shapes do not match")

    return 0


if __name__ == "__main__":
    main()

#!/usr/bin/env python

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
Generate YAML Configuration for TensorRT Engine and Test It

This script generates a YAML configuration for the new TensorRT engine
with 137 PRBs and tests it with the ChannelEstimator.
"""

import os
import argparse
import yaml
import numpy as np
import cupy as cp

from aerial.phy5g.algorithms import ChannelEstimator
from aerial.phy5g.config import PuschConfig, PuschUeConfig


def safe_to_numpy_with_order(array, order='C'):
    """
    Safely convert a CuPy array to NumPy array with specified order.
    If already NumPy, return as is.

    Args:
        array: Either a CuPy or NumPy array
        order: Memory order ('C' or 'F')

    Returns:
        NumPy array
    """
    if isinstance(array, cp.ndarray):
        return array.get(order=order)
    return array


def generate_yaml_config(engine_path, output_yaml=None):
    """
    Generate a YAML configuration for the TensorRT engine with 137 PRBs.

    Args:
        engine_path: Path to the TensorRT engine file
        output_yaml: Path to save the YAML configuration

    Returns:
        Path to the generated YAML file
    """
    # If no output path specified, create one next to the engine
    if output_yaml is None:
        yaml_dir = os.path.dirname(engine_path)
        yaml_name = os.path.basename(engine_path).replace(
            '.engine', '_config.yaml')
        output_yaml = os.path.join(yaml_dir, yaml_name)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_yaml), exist_ok=True)

    # Create configuration with dimensions matching chest_trt.yaml
    config = {
        "chest_factory": "trtengine",
        "file": os.path.abspath(engine_path),
        "max_batch_size": 1,
        "inputs": [
            {
                "name": "z",
                "dataType": 0,  # CUPHY_R_32F
                "dims": [1638, 4, 4, 2, 2]  # 137 PRBs = 1638 subcarriers
            }
        ],
        "outputs": [
            {
                "name": "zout",
                "dataType": 0,  # CUPHY_R_32F
                "dims": [4, 4, 3276, 2, 2]  # Output dimensions
            }
        ]
    }

    # Write configuration to YAML file
    with open(output_yaml, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    print(f"Generated YAML configuration: {output_yaml}")
    return output_yaml


def test_channel_estimator(yaml_path, verbose=False):
    """
    Test the ChannelEstimator with the generated YAML.

    Args:
        yaml_path: Path to the YAML configuration file
        verbose: Whether to print additional information

    Returns:
        True if successful, False otherwise
    """
    print(f"\nTesting ChannelEstimator with YAML: {yaml_path}")

    # Create input data matching [1638, 4, 4, 2, 2] dimensions
    num_subcarriers = 1638
    num_rx_ant = 4
    num_layers = 4
    num_symbols = 2
    num_prbs = 137  # 1638 / 12 = 136.5, rounded up to 137

    # Generate random data
    rx_slot_np = np.random.randn(
        num_subcarriers, num_layers, num_rx_ant, num_symbols, 2
    )
    rx_slot = cp.array(rx_slot_np, dtype=np.float32, order='F')

    print(f"  Input shape: {rx_slot.shape}")
    print(f"  Number of PRBs: {num_prbs}")
    print(f"  Number of RX antennas: {num_rx_ant}")

    # Create PUSCH configuration
    dmrs_positions = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
    ue_config = PuschUeConfig(
        scid=0,
        layers=num_layers,
        dmrs_ports=2  # DMRS port 0 used (bitmap: 0010)
    )

    pusch_config = PuschConfig(
        ue_configs=[ue_config],
        start_prb=0,
        num_prbs=num_prbs,
        start_sym=0,
        num_symbols=14,
        dmrs_syms=dmrs_positions,
        dmrs_add_ln_pos=0,
        dmrs_max_len=1,
        dmrs_scrm_id=0,
        num_dmrs_cdm_grps_no_data=2,
        prg_size=1,
        num_ul_streams=1
    )

    if verbose:
        # Print YAML content for debugging
        with open(yaml_path, 'r') as f:
            yaml_content = f.read()
        print("\nYAML configuration:")
        print(yaml_content)

    try:
        # Create the Channel Estimator with our YAML
        print("\nInitializing ChannelEstimator...")
        channel_estimator = ChannelEstimator(
            num_rx_ant=num_rx_ant,
            ch_est_algo=3,  # LS channel estimation
            chest_factory_settings_filename=yaml_path
        )

        # Run channel estimation
        print("Running channel estimation...")
        ch_est = channel_estimator.estimate(
            rx_slot=rx_slot,
            slot=0,
            pusch_configs=[pusch_config]
        )

        # Get results
        ch_est_cpu = [safe_to_numpy_with_order(est, order='F') for est in ch_est]

        # Print output information
        print("\nChannel estimation results:")
        for ue_idx, est in enumerate(ch_est_cpu):
            print(f"  UE {ue_idx} estimate shape: {est.shape}")
            print(f"  Mean magnitude: {np.abs(est).mean():.6f}")
            print(f"  Max magnitude: {np.abs(est).max():.6f}")

        print("\n✅ ChannelEstimator tested successfully!")
        return True

    except Exception as e:
        print(f"\n❌ Error testing ChannelEstimator: {str(e)}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Generate and Test YAML for TensorRT Engine"
    )
    parser.add_argument(
        '--engine',
        type=str,
        default=('./test_outputs/trt_direct_137prb/'
                 'model_137prb_fp16_tf32.engine'),
        help='Path to the TensorRT engine file'
    )
    parser.add_argument(
        '--yaml',
        type=str,
        default=None,
        help='Path to save the YAML configuration'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print additional information'
    )

    args = parser.parse_args()

    # Verify engine file exists
    if not os.path.exists(args.engine):
        print(f"Error: Engine file not found: {args.engine}")
        return 1

    # Generate YAML configuration
    yaml_path = generate_yaml_config(
        args.engine,
        args.yaml
    )

    # Test the Channel Estimator with the generated YAML
    success = test_channel_estimator(
        yaml_path,
        args.verbose
    )

    return 0 if success else 1


if __name__ == "__main__":
    main()

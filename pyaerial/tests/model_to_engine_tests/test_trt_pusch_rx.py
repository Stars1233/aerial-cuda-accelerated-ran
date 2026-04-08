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

"""Test the full PUSCH RX pipeline with TensorRT-based Channel Estimator."""
import os
import argparse
import time
import numpy as np
import cupy as cp

from aerial.phy5g.algorithms import ChannelEstimator
from aerial.phy5g.algorithms import ChannelEqualizer
from aerial.phy5g.algorithms import NoiseIntfEstimator
from aerial.phy5g.algorithms import RsrpEstimator
from aerial.phy5g.ldpc import LdpcDeRateMatch
from aerial.phy5g.ldpc import LdpcDecoder
from aerial.phy5g.ldpc import CrcChecker
from aerial.phy5g.config import PuschConfig, PuschUeConfig
from aerial.util.cuda import CudaStream


def safe_to_numpy_with_order(array, order='F'):
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


def create_pusch_config_for_pipeline(num_prbs=51, num_symbols=14, dmrs_positions=None):
    """
    Create a PUSCH configuration for the full RX pipeline test.

    Args:
        num_prbs: Number of PRBs
        num_symbols: Number of symbols
        dmrs_positions: List indicating DMRS positions (1 for DMRS, 0 for data)

    Returns:
        PuschConfig object with test parameters
    """
    if dmrs_positions is None:
        # Default: DMRS on symbols 2 and 11 (index starting from 0)
        dmrs_positions = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]

    # Create PUSCH UE config with payload parameters
    ue_config = PuschUeConfig(
        rnti=1,
        layers=1,
        dmrs_ports=2,  # DMRS port 0 used (bitmap: 0010)
        data_scid=0,
        scid=0,
        mcs_index=10,  # Mid-range MCS with reasonable code rate
        mcs_table=0,   # Table 1
        rv=0,
        ndi=1,
        harq_process_id=0,
        tb_size=1000,  # Some reasonable TB size
        code_rate=1930,  # Default code rate
        mod_order=2  # QPSK
    )

    # Create main PUSCH config
    pusch_config = PuschConfig(
        ue_configs=[ue_config],  # Use ue_config created above
        start_prb=0,
        num_prbs=num_prbs,
        start_sym=0,
        num_symbols=num_symbols,
        dmrs_syms=dmrs_positions,
        dmrs_add_ln_pos=0,
        dmrs_max_len=1,
        dmrs_scrm_id=0,
        num_dmrs_cdm_grps_no_data=2,
        prg_size=1,
        num_ul_streams=1
    )

    return pusch_config


def create_random_rx_slot(num_prbs, num_symbols, num_rx_ant, h2d=True, verbose=False):
    """
    Create random received slot data for testing.

    Args:
        num_prbs: Number of PRBs
        num_symbols: Number of symbols
        num_rx_ant: Number of receive antennas
        h2d: Whether to transfer data to GPU (CuPy)
        verbose: Whether to print debug information

    Returns:
        Array with random RX slot data
    """
    # Generate random complex data
    # Shape: [num_symbols, num_prbs*12, num_rx_ant]
    num_subcarriers = num_prbs * 12
    rx_slot_np = np.random.randn(num_symbols, num_subcarriers, num_rx_ant) + \
              1j * np.random.randn(num_symbols, num_subcarriers, num_rx_ant)  # noqa: E127
    rx_slot_np = rx_slot_np.astype(np.complex64)

    # DEBUG: Print rx_slot statistics
    if verbose:
        print("\n[DEBUG] RX Slot Generation:")
        print(f"  Shape: {rx_slot_np.shape}")
        print(f"  Mean power: {np.abs(rx_slot_np).mean():.6f}")
        print(f"  Max magnitude: {np.abs(rx_slot_np).max():.6f}")
        print(f"  Min magnitude: {np.abs(rx_slot_np).min():.6f}")

    if h2d:
        # Transfer to GPU
        rx_slot = cp.array(rx_slot_np, order='F')
    else:
        rx_slot = rx_slot_np

    return rx_slot


def adapt_tensorrt_channel_estimates(ch_est, verbose=False):
    """
    Adapt channel estimates from TensorRT format to the format expected
    by NoiseIntfEstimator and other downstream components.

    This implementation uses a DLPack-based approach similar to the reference
    implementation in ml_pusch_rx.py to ensure compatibility.

    Args:
        ch_est: Channel estimates from TensorRT-based channel estimator
        verbose: Whether to print debug information

    Returns:
        Adapted channel estimates with proper memory layout
    """
    adapted_ch_est = []

    # Ensure device synchronization before processing
    cp.cuda.Device().synchronize()

    for est in ch_est:
        # Check memory layout and shape
        if verbose:
            print(f"Original shape: {est.shape}, dtype: {est.dtype}")
            if isinstance(est, cp.ndarray):
                layout_msg = (
                    f"Memory layout: F-contiguous={est.flags.f_contiguous}, "
                    f"C-contiguous={est.flags.c_contiguous}"
                )
                print(layout_msg)

        # Use PyTorch as an intermediary for reliable memory handling
        try:
            import torch

            # Check if input is CuPy array
            is_cupy = isinstance(est, cp.ndarray)

            if is_cupy:
                # For CuPy input, use DLPack conversion path (more reliable)
                # First ensure contiguous array (C-contiguous for PyTorch)
                est_contiguous = cp.ascontiguousarray(est)
                # Convert to PyTorch using DLPack
                torch_tensor = torch.from_dlpack(
                    est_contiguous.toDlpack()
                ).to(torch.complex64)

                # Pass through PyTorch to standardize memory layout
                # Then convert back to CuPy using DLPack
                # This ensures proper memory layout and avoids corruption
                torch_tensor = torch_tensor.contiguous()
                est_adapted = cp.from_dlpack(torch.to_dlpack(torch_tensor))

                # Ensure proper dtype
                if est_adapted.dtype != cp.complex64:
                    est_adapted = est_adapted.astype(cp.complex64)

                # Ensure F-contiguous as expected by cuPHY
                if not est_adapted.flags.f_contiguous:
                    est_adapted = cp.asfortranarray(est_adapted)
            else:
                # For non-CuPy tensors
                tensor_torch = torch.tensor(est, dtype=torch.complex64).cuda()
                tensor_result = cp.from_dlpack(
                    torch.to_dlpack(tensor_torch)
                )
                est_adapted = cp.asfortranarray(tensor_result)

        except (ImportError, RuntimeError) as e:
            # Fallback if PyTorch not available or DLPack conversion fails
            if verbose:
                print(f"DLPack conversion failed, using fallback: {str(e)}")

            if isinstance(est, cp.ndarray):
                # For CuPy arrays, create a fresh copy with F-ordering
                est_adapted = cp.asfortranarray(est, dtype=cp.complex64)
            else:
                # For non-CuPy arrays, convert to CuPy
                est_adapted = cp.asfortranarray(est, dtype=cp.complex64)

        if verbose:
            print(f"Adapted shape: {est_adapted.shape}, dtype: {est_adapted.dtype}")
            layout_msg = (
                f"New memory layout: F-contiguous={est_adapted.flags.f_contiguous}, "
                f"C-contiguous={est_adapted.flags.c_contiguous}"
            )
            print(layout_msg)

        adapted_ch_est.append(est_adapted)

    # Final synchronization to ensure all operations are complete
    cp.cuda.Device().synchronize()

    return adapted_ch_est


def ensure_tensor_format_compatibility(tensor_list, verbose=False):
    """
    Ensure tensor format compatibility with NoiseIntfEstimator using a more
    robust approach through PyTorch and DLPack.

    Args:
        tensor_list: List of tensors to adapt
        verbose: Whether to print debug information

    Returns:
        List of tensors in the correct format
    """
    result = []

    # Import torch at the function level
    # This assumes torch must exist, so we don't need a fallback approach
    import torch

    for tensor in tensor_list:
        if verbose:
            print(f"Original tensor: shape={tensor.shape}, type={type(tensor)}")
            if isinstance(tensor, cp.ndarray):
                print(
                    f"  Memory layout: F-contiguous={tensor.flags.f_contiguous}, "
                    f"C-contiguous={tensor.flags.c_contiguous}"
                )

        # Use PyTorch as intermediary for safer memory handling
        if isinstance(tensor, cp.ndarray):
            # Use DLPack for zero-copy transfer
            tensor_contiguous = cp.ascontiguousarray(tensor)
            tensor_torch = torch.from_dlpack(tensor_contiguous.toDlpack()).to(torch.complex64)

            # Process and convert back using DLPack
            tensor_torch = tensor_torch.contiguous()
            tensor_result = cp.from_dlpack(torch.to_dlpack(tensor_torch))

            # Ensure F-contiguous which is expected by cuPHY
            if not tensor_result.flags.f_contiguous:
                tensor_result = cp.asfortranarray(tensor_result)
        else:
            # For non-CuPy tensors
            tensor_torch = torch.tensor(tensor, dtype=torch.complex64).cuda()
            tensor_result = cp.from_dlpack(torch.to_dlpack(tensor_torch))
            tensor_result = cp.asfortranarray(tensor_result)

        # Ensure complex64 type
        if tensor_result.dtype != cp.complex64:
            tensor_result = tensor_result.astype(cp.complex64)

        # Synchronize to ensure memory is ready
        cp.cuda.Device().synchronize()

        if verbose:
            print(
                f"Converted tensor: shape={tensor_result.shape}, "
                f"type={type(tensor_result)}"
            )
            print(
                f"  Memory layout: F-contiguous={tensor_result.flags.f_contiguous}, "
                f"C-contiguous={tensor_result.flags.c_contiguous}"
            )

        result.append(tensor_result)

    # Final synchronization
    cp.cuda.Device().synchronize()

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Test PUSCH RX pipeline with TensorRT ChEst"
    )
    parser.add_argument('--yaml', type=str, required=True,
                      help='Path to TensorRT channel estimator YAML config')  # noqa: E128
    parser.add_argument('--num_prbs', type=int, default=51,
                      help='Number of PRBs')  # noqa: E128
    parser.add_argument('--num_rx_ant', type=int, default=4,
                      help='Number of receive antennas')  # noqa: E128
    parser.add_argument('--output', type=str, default='pusch_rx_results',
                      help='Output directory for results')  # noqa: E128
    parser.add_argument('--verbose', action='store_true',
                      help='Enable verbose logging')  # noqa: E128
    args = parser.parse_args()

    # Check YAML file exists
    if not os.path.exists(args.yaml):
        print(f"Error: YAML file not found at {args.yaml}")
        return 1

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    # Create PUSCH configuration
    pusch_config = create_pusch_config_for_pipeline(
        num_prbs=args.num_prbs,
        num_symbols=14
    )

    print("\n=== Testing Full PUSCH RX Pipeline with TensorRT ChEst ===")
    print(f"YAML: {args.yaml}")
    print(f"Number of PRBs: {args.num_prbs}")
    print(f"Number of RX antennas: {args.num_rx_ant}")

    # Reset CUDA device to clear any previous memory issues
    try:
        cp.cuda.Device().synchronize()
        cp.cuda.runtime.deviceReset()  # Use runtime.deviceReset() instead of Device().reset()
        cp.cuda.runtime.setDevice(0)  # Ensure we're using device 0
    except Exception as e:
        print(f"Warning: CUDA device reset failed: {str(e)}")

    # Create random RX slot data
    rx_slot = create_random_rx_slot(
        num_prbs=args.num_prbs,
        num_symbols=14,
        num_rx_ant=args.num_rx_ant,
        verbose=args.verbose
    )

    # Create a proper CUDA stream instead of using the default (0)
    cuda_stream = CudaStream()

    # Dictionary to store timing information
    timings = {}

    # Instantiate all components for the PUSCH RX pipeline
    print("Initializing PUSCH RX components...")

    start_time = time.time()

    # Channel Estimator using TensorRT engine via YAML config
    # Note: Initialize using chest_factory_settings_filename to use the TRT engine
    channel_estimator = ChannelEstimator(
        num_rx_ant=args.num_rx_ant,
        ch_est_algo=3,  # This will be overridden by the YAML config
        chest_factory_settings_filename=args.yaml,
        cuda_stream=cuda_stream  # Use explicitly created stream
    )

    # Noise and Interference Estimator
    noise_intf_estimator = NoiseIntfEstimator(
        num_rx_ant=args.num_rx_ant,
        eq_coeff_algo=0,
        cuda_stream=cuda_stream
    )

    # Channel Equalizer
    channel_equalizer = ChannelEqualizer(
        num_rx_ant=args.num_rx_ant,
        eq_coeff_algo=0,
        enable_pusch_tdi=False,
        cuda_stream=cuda_stream
    )

    # RSRP Estimator
    rsrp_estimator = RsrpEstimator(
        num_rx_ant=args.num_rx_ant,
        enable_pusch_tdi=False,
        cuda_stream=cuda_stream
    )

    # LDPC De-Rate Matching
    derate_match = LdpcDeRateMatch(
        enable_scrambling=True,
        cuda_stream=cuda_stream
    )

    # LDPC Decoder
    decoder = LdpcDecoder(
        cuda_stream=cuda_stream
    )

    # CRC Checker
    crc_checker = CrcChecker(
        cuda_stream=cuda_stream
    )

    timings['initialization'] = time.time() - start_time

    # Run the full PUSCH RX pipeline
    print("Executing PUSCH RX pipeline...")

    try:
        # Main pipeline execution
        execute_pipeline(
            rx_slot=rx_slot,
            pusch_config=pusch_config,
            channel_estimator=channel_estimator,
            noise_intf_estimator=noise_intf_estimator,
            channel_equalizer=channel_equalizer,
            rsrp_estimator=rsrp_estimator,
            derate_match=derate_match,
            decoder=decoder,
            crc_checker=crc_checker,
            args=args,
            timings=timings,
            output_dir=args.output
        )
        return 0

    except Exception as e:
        print("\n❌ ERROR: Exception occurred during PUSCH RX pipeline execution:")
        print(f"   {str(e)}")
        return 1


def execute_pipeline(rx_slot, pusch_config, channel_estimator, noise_intf_estimator,  # noqa: C901, E501
                    channel_equalizer, rsrp_estimator, derate_match, decoder,  # noqa: E128
                    crc_checker, args, timings, output_dir):  # noqa: E128
    """
    Execute the PUSCH RX pipeline with the given components.

    This function is extracted from main() to allow for recovery attempts.
    """
    # Step 1: Channel Estimation
    print("Step 1: Running channel estimation...")
    start_time = time.time()

    # Explicitly synchronize all CUDA operations before channel estimation
    if isinstance(rx_slot, cp.ndarray):
        cp.cuda.Device().synchronize()

    ch_est = channel_estimator.estimate(
        rx_slot=rx_slot,
        slot=0,
        pusch_configs=[pusch_config]
    )

    # Synchronize after channel estimation
    if isinstance(rx_slot, cp.ndarray):
        cp.cuda.Device().synchronize()

    # Adapt channel estimates to ensure proper memory layout and format
    ch_est_adapted = adapt_tensorrt_channel_estimates(
        ch_est,
        verbose=args.verbose
    )

    timings['channel_estimation'] = time.time() - start_time

    if args.verbose:
        ch_est_cpu = [safe_to_numpy_with_order(est, order='F') for est in ch_est]
        print(f"Channel estimation output shape: {ch_est_cpu[0].shape}")
        print(f"Channel estimation mean magnitude: {np.abs(ch_est_cpu[0]).mean():.6f}")

    # DEBUG: Always print channel estimation details
    if args.verbose:
        ch_est_cpu = [safe_to_numpy_with_order(est, order='F') for est in ch_est_adapted]
        print("\n[DEBUG] Channel Estimation Results:")
        print(f"  Number of estimates: {len(ch_est_cpu)}")
        for i, est in enumerate(ch_est_cpu):
            print(f"  Estimate {i}: shape={est.shape}, dtype={est.dtype}")
            print(f"    Mean magnitude: {np.abs(est).mean():.6f}")
            print(f"    Max magnitude: {np.abs(est).max():.6f}")
            print(f"    Contains NaN: {np.isnan(est).any()}")
            print(f"    Contains Inf: {np.isinf(est).any()}")

    # Step 2: Noise and Interference Estimation
    print("Step 2: Running noise and interference estimation...")
    start_time = time.time()

    # Explicitly synchronize all CUDA operations before proceeding
    if isinstance(rx_slot, cp.ndarray):
        cp.cuda.Device().synchronize()

    # Verify that the adapted channel estimates are properly formatted
    if args.verbose:
        for i, est in enumerate(ch_est_adapted):
            print(f"Channel estimate {i} before noise estimation:")
            print(f"  Shape: {est.shape}, dtype: {est.dtype}")
            print(f"  F-contiguous: {est.flags.f_contiguous}")
            print(f"  Is on GPU: {isinstance(est, cp.ndarray)}")
            print(f"  Memory pointer: {est.data.ptr}")

    try:
        lw_inv, noise_var_pre_eq = noise_intf_estimator.estimate(
            rx_slot=rx_slot,
            channel_est=ch_est_adapted,  # Use adapted channel estimates
            slot=0,
            pusch_configs=[pusch_config]
        )

        # Synchronize after estimation to ensure memory is ready for next steps
        if isinstance(rx_slot, cp.ndarray):
            cp.cuda.Device().synchronize()

        if args.verbose:
            print("Noise estimation completed successfully")
            if isinstance(noise_var_pre_eq, cp.ndarray):
                print(f"Noise var shape: {noise_var_pre_eq.shape}, dtype: {noise_var_pre_eq.dtype}")
                print(f"Noise var mean: {noise_var_pre_eq.mean()}")

    except Exception as e:
        print(f"Error during noise estimation: {str(e)}")
        # If there was an error, try a different approach with explicit memory management
        print("Attempting recovery with explicit memory management...")

        # Force garbage collection to free any potentially corrupt GPU memory
        import gc
        gc.collect()
        if isinstance(rx_slot, cp.ndarray):
            cp.cuda.Device().synchronize()

        try:
            # Try with even more thorough tensor format compatibility conversion
            print("Attempting with thorough tensor format conversion...")
            ch_est_compatible = ensure_tensor_format_compatibility(ch_est_adapted)

            lw_inv, noise_var_pre_eq = noise_intf_estimator.estimate(
                rx_slot=rx_slot,
                channel_est=ch_est_compatible,
                slot=0,
                pusch_configs=[pusch_config]
            )
        except Exception as e2:
            print(f"Format conversion attempt failed: {str(e2)}")
            print("Attempting final recovery with fresh copies...")

            # Try again with fresh copies through CPU path
            ch_est_fresh = []
            for est in ch_est_adapted:
                if isinstance(est, cp.ndarray):
                    # Create a brand new copy with explicit memory allocation
                    est_fresh = cp.asfortranarray(est.get(), dtype=cp.complex64)
                else:
                    # For CPU arrays
                    est_fresh = cp.asfortranarray(est, dtype=cp.complex64)
                ch_est_fresh.append(est_fresh)

            # Retry the estimation with fresh memory
            lw_inv, noise_var_pre_eq = noise_intf_estimator.estimate(
                rx_slot=rx_slot,
                channel_est=ch_est_fresh,
                slot=0,
                pusch_configs=[pusch_config]
            )

        # Synchronize again
        if isinstance(rx_slot, cp.ndarray):
            cp.cuda.Device().synchronize()

    timings['noise_estimation'] = time.time() - start_time

    # DEBUG: Print noise estimation results
    if args.verbose:
        if isinstance(noise_var_pre_eq, cp.ndarray):
            noise_var_cpu = noise_var_pre_eq.get()
        else:
            noise_var_cpu = noise_var_pre_eq

        print("\n[DEBUG] Noise Estimation Results:")
        print(f"  Noise variance shape: {noise_var_cpu.shape}")
        print(f"  Mean noise variance: {noise_var_cpu.mean():.6f}")
        print(f"  Max noise variance: {noise_var_cpu.max():.6f}")
        print(f"  Min noise variance: {noise_var_cpu.min():.6f}")
        print(f"  Contains NaN: {np.isnan(noise_var_cpu).any()}")

        if isinstance(lw_inv, list):
            for i, lw in enumerate(lw_inv):
                lw_cpu = safe_to_numpy_with_order(lw, order='F')
                print(f"  LW_inv[{i}] shape: {lw_cpu.shape}, mean: {np.abs(lw_cpu).mean():.6f}")

    # Step 3: Channel Equalization
    print("Step 3: Running channel equalization...")
    start_time = time.time()
    llrs, _ = channel_equalizer.equalize(
        rx_slot=rx_slot,
        channel_est=ch_est_adapted,  # Use adapted channel estimates
        lw_inv=lw_inv,
        noise_var_pre_eq=noise_var_pre_eq,
        pusch_configs=[pusch_config]
    )
    timings['equalization'] = time.time() - start_time

    # DEBUG: Print equalization results
    if args.verbose:
        print("\n[DEBUG] Channel Equalization Results:")
        if isinstance(llrs, list):
            for i, llr in enumerate(llrs):
                llr_cpu = safe_to_numpy_with_order(llr, order='F')
                print(f"  LLR[{i}] shape: {llr_cpu.shape}")
                print(f"    Mean LLR magnitude: {np.abs(llr_cpu).mean():.6f}")
                print(f"    Max LLR: {llr_cpu.max():.6f}")
                print(f"    Min LLR: {llr_cpu.min():.6f}")
                print(f"    Contains NaN: {np.isnan(llr_cpu).any()}")
                print(f"    Contains Inf: {np.isinf(llr_cpu).any()}")
                # Check for all zeros
                print(f"    All zeros: {np.all(llr_cpu == 0)}")
                # Sample some LLR values
                flat_llr = llr_cpu.flatten()
                print(f"    Sample LLRs (first 10): {flat_llr[:10]}")

    # Step 4: RSRP Estimation (optional but good for debug)
    print("Step 4: Running RSRP estimation...")
    start_time = time.time()
    ree_diag_inv = channel_equalizer.ree_diag_inv
    rsrp, _, post_eq_sinr = rsrp_estimator.estimate(
        channel_est=ch_est_adapted,  # Use adapted channel estimates
        ree_diag_inv=ree_diag_inv,
        noise_var_pre_eq=noise_var_pre_eq,
        pusch_configs=[pusch_config]
    )
    timings['rsrp_estimation'] = time.time() - start_time

    # DEBUG: Print RSRP/SINR results
    if args.verbose:
        rsrp_cpu_debug = safe_to_numpy_with_order(rsrp, order='F')
        sinr_cpu_debug = safe_to_numpy_with_order(post_eq_sinr, order='F')
        print("\n[DEBUG] RSRP/SINR Estimation Results:")
        print(f"  RSRP: {rsrp_cpu_debug}")
        print(f"  Post-EQ SINR: {sinr_cpu_debug}")

    # Step 5: LDPC De-Rate Matching
    print("Step 5: Running LDPC de-rate matching...")
    start_time = time.time()
    coded_blocks = derate_match.derate_match(
        input_llrs=llrs,
        pusch_configs=[pusch_config]
    )
    timings['derate_matching'] = time.time() - start_time

    # DEBUG: Print de-rate matching results
    if args.verbose:
        print("\n[DEBUG] LDPC De-Rate Matching Results:")
        if isinstance(coded_blocks, list):
            for i, cb in enumerate(coded_blocks):
                cb_cpu = safe_to_numpy_with_order(cb, order='F')
                print(f"  Coded block[{i}] shape: {cb_cpu.shape}")
                print(f"    Mean magnitude: {np.abs(cb_cpu).mean():.6f}")
                print(f"    Max: {cb_cpu.max():.6f}")
                print(f"    Min: {cb_cpu.min():.6f}")
                print(f"    Contains NaN: {np.isnan(cb_cpu).any()}")
                print(f"    All zeros: {np.all(cb_cpu == 0)}")
                # Check distribution of values
                positive_ratio = np.sum(cb_cpu > 0) / cb_cpu.size
                print(f"    Positive LLR ratio: {positive_ratio:.3f}")

    # Step 6: LDPC Decoding
    print("Step 6: Running LDPC decoding...")
    start_time = time.time()
    code_blocks = decoder.decode(
        input_llrs=coded_blocks,
        pusch_configs=[pusch_config]
    )
    timings['decoding'] = time.time() - start_time

    # DEBUG: Print decoding results
    if args.verbose:
        print("\n[DEBUG] LDPC Decoding Results:")
        if isinstance(code_blocks, list):
            for i, cb in enumerate(code_blocks):
                cb_cpu = safe_to_numpy_with_order(cb, order='F')
                print(f"  Decoded block[{i}] shape: {cb_cpu.shape}")
                print(f"    Unique values: {np.unique(cb_cpu)}")
                print(f"    Num zeros: {np.sum(cb_cpu == 0)}")
                print(f"    Num ones: {np.sum(cb_cpu == 1)}")
                # Sample decoded bits
                if cb_cpu.size > 0:
                    print(f"    Sample bits (first 20): {cb_cpu.flatten()[:20]}")

    # Step 7: CRC Checking
    print("Step 7: Running CRC checking...")
    start_time = time.time()
    tb, tb_crcs = crc_checker.check_crc(
        input_bits=code_blocks,
        pusch_configs=[pusch_config]
    )
    timings['crc_checking'] = time.time() - start_time

    # Convert results to CPU if needed
    tb_cpu = [safe_to_numpy_with_order(elem, order='F') for elem in tb]
    tb_crcs_cpu = [safe_to_numpy_with_order(elem, order='F') for elem in tb_crcs]
    rsrp_cpu = safe_to_numpy_with_order(rsrp, order='F')
    post_eq_sinr_cpu = safe_to_numpy_with_order(post_eq_sinr, order='F')

    # Check CRC results
    crc_pass = tb_crcs_cpu[0][0] == 0

    # DEBUG: Print CRC results in detail
    if args.verbose:
        print("\n[DEBUG] CRC Checking Results:")
        print(f"  Number of TBs: {len(tb_cpu)}")
        print(f"  Number of CRC results: {len(tb_crcs_cpu)}")
        for i, (tb_data, crc_result) in enumerate(zip(tb_cpu, tb_crcs_cpu)):
            print(f"  TB[{i}]:")
            print(f"    Shape: {tb_data.shape}")
            print(f"    CRC result: {crc_result} (0=pass, non-zero=fail)")
            print(f"    TB size (bits): {tb_data.size}")
            if tb_data.size > 0:
                print(f"    Sample TB bits (first 20): {tb_data.flatten()[:20]}")
                print(f"    Unique values in TB: {np.unique(tb_data)}")
                print(f"    All zeros: {np.all(tb_data == 0)}")

        # DEBUG: Print PUSCH config details
        print("\n[DEBUG] PUSCH Configuration:")
        print(f"  Num PRBs: {pusch_config.num_prbs}")
        print(f"  Num symbols: {pusch_config.num_symbols}")
        print(f"  DMRS symbols: {pusch_config.dmrs_syms}")
        if pusch_config.ue_configs:
            ue_cfg = pusch_config.ue_configs[0]
            print("  UE Config:")
            print(f"    MCS index: {ue_cfg.mcs_index}")
            print(f"    Modulation order: {ue_cfg.mod_order}")
            print(f"    TB size: {ue_cfg.tb_size}")
            print(f"    Code rate: {ue_cfg.code_rate}")
            print(f"    RV: {ue_cfg.rv}")
            print(f"    Layers: {ue_cfg.layers}")

    # Save results
    print("Saving results...")
    os.makedirs(output_dir, exist_ok=True)
    result_file = os.path.join(output_dir, "pusch_rx_pipeline_results.npz")
    np.savez(
        result_file,
        tb=tb_cpu[0],
        tb_crc=tb_crcs_cpu[0],
        rsrp=rsrp_cpu,
        sinr=post_eq_sinr_cpu,
        crc_pass=crc_pass,
        timings=tuple(timings.items())  # Convert dict to tuple of (key, value) pairs
    )

    # Print final status
    if crc_pass:
        print("\n✅ PASS: PUSCH RX Pipeline successful!")
        print(f"   RSRP: {rsrp_cpu[0]:.2f} dB")
        print(f"   SINR: {post_eq_sinr_cpu[0]:.2f} dB")
    else:
        print("\n❌ FAIL: CRC check failed in PUSCH RX Pipeline!")

    # Print timing information
    print("\nPerformance Metrics:")
    total_time = sum(timings.values())
    for stage, duration in timings.items():
        percentage = (duration / total_time) * 100
        print(f"  {stage:<20}: {duration*1000:.2f} ms ({percentage:.1f}%)")
    print(f"  {'Total':<20}: {total_time*1000:.2f} ms (100%)")

    print(f"\nResults saved to: {result_file}")

    return crc_pass


if __name__ == "__main__":
    import sys
    sys.exit(main())

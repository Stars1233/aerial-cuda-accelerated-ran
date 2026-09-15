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

import gc
import numpy as np
import pytest
from cuda.bindings import runtime
from gpu3gppchan import (
    CarrierParams,
    CudaArrayComplexFloat,
    OfdmDeModulate,
    OfdmModulate,
)
from gpu3gppchan.cuda_utils import CudaStream


def compare_arrays(
    array1: np.ndarray | int,
    array2: np.ndarray | int,
    gpu_indicator: bool,
    size: int,
) -> None:
    """
    Compare two GPU or CPU arrays to check if they are close.

    Parameters:
    - array1: first array
    - array2: second array
    - gpu_indicator:
        - 0: array1 and array2 are numpy arraies
        - 1: array1 and array2 are GPU arrays, i.e., cuda device pointers
    - size: int, number of elements to compare

    Returns:
    - no returns, assert failure if arrays are not equal.
    """
    error_str = "OFDM modulation and demodulation results do not match, test FAIL!"
    if gpu_indicator:
        # allocate host memory
        host_array1 = np.empty(size, dtype=np.complex64)
        host_array2 = np.empty(size, dtype=np.complex64)

        # copy data from GPU to host
        err = runtime.cudaMemcpy(
            host_array1.ctypes.data, array1, host_array1.nbytes,
            runtime.cudaMemcpyKind.cudaMemcpyDeviceToHost
        )
        if err != (runtime.cudaError_t.cudaSuccess,):
            raise RuntimeError(f"Failed to copy array1 from GPU: {err}")

        err = runtime.cudaMemcpy(
            host_array2.ctypes.data, array2, host_array2.nbytes,
            runtime.cudaMemcpyKind.cudaMemcpyDeviceToHost
        )
        if err != (runtime.cudaError_t.cudaSuccess,):
            raise RuntimeError(f"Failed to copy array2 from GPU: {err}")

        assert np.allclose(host_array1, host_array2, rtol=1e-5, atol=1e-5), error_str
    else:
        assert np.allclose(array1, array2, rtol=1e-5, atol=1e-5), error_str


def ofdm_mod_demod_numpy(
    carrier_params: CarrierParams,
    n_tti: int,
    cuda_stream: CudaStream,
) -> None:
    """
        Test OfdmModulate and OfdmDeModulate using numpy for frequency data samples.
    """

    # Step 1: allocate numpy buffers
    freq_data_size = carrier_params.n_bs_layer * carrier_params.n_symbol_slot \
        * carrier_params.n_sc
    # since we only do OFDM mod + demod, input and output data have the same size
    freq_in = np.empty(freq_data_size, dtype=np.complex64)
    freq_out = np.empty(freq_data_size, dtype=np.complex64)

    # Step 2: create OFDM modulation and demodulation objects
    ofdm_mod = OfdmModulate(carrier_params=carrier_params,
                            freq_data_in_cpu=freq_in,
                            stream_handle=cuda_stream
                            )
    time_data_gpu = ofdm_mod.get_time_data_out()
    ofdm_demod = OfdmDeModulate(carrier_params=carrier_params,
                                time_data_in_gpu=time_data_gpu,
                                freq_data_out_cpu=freq_out,
                                prach=0,
                                stream_handle=cuda_stream
                                )
    # Step 3: run test and compare results
    for tti_idx in range(0, n_tti):
        # generate freq in data using numpy
        freq_in.real = np.random.randn(freq_data_size)
        freq_in.imag = np.random.randn(freq_data_size)
        # output frequency data will be stored in GPU memory by ofdmDeModulate

        # run OFDM modulation and demodulation
        ofdm_mod.run()  # or ofdm_mod.run(freq_in_new) with a numpy array (to be created)
        ofdm_demod.run()  # or ofdm_demod.run(freq_out_new) with a numpy array (to be created)
        compare_arrays(
            array1=freq_in,
            array2=freq_out,
            gpu_indicator=0,
            size=freq_data_size
        )


def ofdm_mod_demod_gpu_only(
    carrier_params: CarrierParams,
    n_tti: int,
    cuda_stream: CudaStream,
) -> None:
    """
        Test OfdmModulate and OfdmDeModulate using GPU memory address for frequency data samples.
    """

    # Step 1: allocate GPU buffers
    freq_data_size = carrier_params.n_bs_layer * carrier_params.n_symbol_slot \
        * carrier_params.n_sc
    freq_in = np.empty(freq_data_size, dtype=np.complex64)
    # since we only do OFDM mod + demod, input and output data have the same size
    # Allocate GPU memory
    err, freq_in_gpu = runtime.cudaMalloc(freq_in.nbytes)
    if err != runtime.cudaError_t.cudaSuccess:
        raise RuntimeError(f"Failed to allocate freq_in_gpu memory: {err}")

    freq_out_gpu = None
    try:
        err, freq_out_gpu = runtime.cudaMalloc(freq_in.nbytes)
        if err != runtime.cudaError_t.cudaSuccess:
            raise RuntimeError(f"Failed to allocate freq_out_gpu memory: {err}")

        freq_in_array = CudaArrayComplexFloat(
            freq_in_gpu, [freq_data_size], []
        )
        freq_out_array = CudaArrayComplexFloat(
            freq_out_gpu, [freq_data_size], []
        )
        ofdm_mod = OfdmModulate(
            carrier_params=carrier_params,
            freq_data_in_gpu=freq_in_array,
            stream_handle=cuda_stream,
        )
        time_data_in_gpu = ofdm_mod.get_time_data_out()
        ofdm_demod = OfdmDeModulate(
            carrier_params=carrier_params,
            time_data_in_gpu=time_data_in_gpu,
            freq_data_out_gpu=freq_out_array,
            prach=0,
            stream_handle=cuda_stream,
        )

        for _ in range(n_tti):
            freq_in.real = np.random.randn(freq_data_size)
            freq_in.imag = np.random.randn(freq_data_size)
            err = runtime.cudaMemcpyAsync(
                freq_in_gpu, freq_in.ctypes.data, freq_in.nbytes,
                runtime.cudaMemcpyKind.cudaMemcpyHostToDevice,
                cuda_stream.handle,
            )
            if err != (runtime.cudaError_t.cudaSuccess,):
                raise RuntimeError(f"Failed to copy data to GPU: {err}")

            ofdm_mod.run()
            ofdm_demod.run()
            cuda_stream.synchronize()
            compare_arrays(
                array1=freq_in_gpu,
                array2=freq_out_gpu,
                gpu_indicator=1,
                size=freq_data_size,
            )
    finally:
        if freq_out_gpu is not None:
            err = runtime.cudaFree(freq_out_gpu)
            if err != (runtime.cudaError_t.cudaSuccess,):
                print(f"Warning: Failed to free freq_out_gpu memory: {err}")
        err = runtime.cudaFree(freq_in_gpu)
        if err != (runtime.cudaError_t.cudaSuccess,):
            print(f"Warning: Failed to free freq_in_gpu memory: {err}")


@pytest.mark.parametrize("wrapper_name", ["modulate", "demodulate"])
def test_ofdm_numpy_constructor_rejects_undersized_buffer(
    wrapper_name: str,
    cuda_stream: CudaStream,
) -> None:
    """Reject constructor-owned host buffers before an async copy can overrun them."""
    carrier_params = CarrierParams()

    if wrapper_name == "modulate":
        required_size = (
            carrier_params.n_bs_layer
            * carrier_params.n_symbol_slot
            * carrier_params.n_sc
        )
        undersized = np.empty(required_size - 1, dtype=np.complex64)
        with pytest.raises(ValueError, match="expected at least"):
            OfdmModulate(
                carrier_params=carrier_params,
                freq_data_in_cpu=undersized,
                stream_handle=cuda_stream,
            )
    else:
        required_size = (
            carrier_params.n_ue_layer
            * carrier_params.n_symbol_slot
            * carrier_params.n_sc
        )
        input_buffer = np.empty(
            carrier_params.n_bs_layer
            * carrier_params.n_symbol_slot
            * carrier_params.n_sc,
            dtype=np.complex64,
        )
        modulator = OfdmModulate(
            carrier_params=carrier_params,
            freq_data_in_cpu=input_buffer,
            stream_handle=cuda_stream,
        )
        undersized = np.empty(required_size - 1, dtype=np.complex64)
        with pytest.raises(ValueError, match="expected at least"):
            OfdmDeModulate(
                carrier_params=carrier_params,
                time_data_in_gpu=modulator.get_time_data_out(),
                freq_data_out_cpu=undersized,
                prach=0,
                stream_handle=cuda_stream,
            )


def test_ofdm_demodulator_retains_modulator_output_owner(
    cuda_stream: CudaStream,
) -> None:
    """The demodulator input view must keep its producing modulator alive."""
    carrier_params = CarrierParams()
    freq_data_size = (
        carrier_params.n_bs_layer
        * carrier_params.n_symbol_slot
        * carrier_params.n_sc
    )
    freq_in = np.zeros(freq_data_size, dtype=np.complex64)
    freq_out = np.empty(freq_data_size, dtype=np.complex64)
    modulator = OfdmModulate(
        carrier_params=carrier_params,
        freq_data_in_cpu=freq_in,
        stream_handle=cuda_stream,
    )
    time_data = modulator.get_time_data_out()
    demodulator = OfdmDeModulate(
        carrier_params=carrier_params,
        time_data_in_gpu=time_data,
        freq_data_out_cpu=freq_out,
        prach=0,
        stream_handle=cuda_stream,
    )
    modulator.run()

    del time_data
    del modulator
    gc.collect()
    demodulator.run()

    assert np.allclose(freq_out, freq_in)


@pytest.mark.parametrize(
    "n_sc, n_tti, numpy_indicator", [
        (1632, 100, 0),
        (1632, 100, 1),
        (3276, 100, 0),
        (3276, 100, 1)
    ]
)
def test_ofdm_mod_demod(n_sc, n_tti, numpy_indicator, cuda_stream):
    """
    Main test function of OfdmModulate and OfdmDeModulate.

    Paremeters:
    - n_sc: number of sub-carriers
    - n_tti: number of TTIs in test
    - numpy_indicator: 1 - run test with numpy; 0 - run test with GPU momery directly
    - cuda_stream: cuda_stream to run test
    """
    try:
        # carrier configurations
        carrier_params = CarrierParams()
        carrier_params.n_sc = n_sc

        if numpy_indicator:
            ofdm_mod_demod_numpy(carrier_params, n_tti, cuda_stream)
        else:
            ofdm_mod_demod_gpu_only(carrier_params, n_tti, cuda_stream)

    except Exception as e:
        assert False, f"Error running OFDM test: {e}"

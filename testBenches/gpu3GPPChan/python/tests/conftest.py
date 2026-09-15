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

"""Pytest fixtures for the gpu3gppchan Python tests."""

import gc
from collections.abc import Generator
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from gpu3gppchan.cuda_utils import CudaStream


@pytest.fixture(scope="function", autouse=True)
def clear_session() -> Generator[None, None, None]:
    """Release cached GPU allocations and collect Python objects after each test."""
    yield
    try:
        import cupy as cp
    except ImportError:
        gc.collect()
        return
    cp.get_default_memory_pool().free_all_blocks()
    gc.collect()


@pytest.fixture(name="cuda_stream", scope="function")
def fixture_cuda_stream() -> Generator["CudaStream", None, None]:
    """Create a CUDA stream for a test and release it when the fixture exits."""
    pytest.importorskip("cuda.bindings.runtime")
    import cuda.bindings.runtime as cudart
    from gpu3gppchan.cuda_utils import CudaStream

    cudart.cudaSetDevice(0)
    stream = CudaStream()
    yield stream

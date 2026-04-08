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

"""CUDA-related utilities.

This module provides a :class:`CudaStream` wrapper for CUDA streams with
context-manager support. Use ``with stream:`` to scope work so that CuPy
operations run on this stream; the context manager does not synchronize or
destroy the stream on exit. Call :meth:`CudaStream.synchronize` explicitly
when synchronization is needed. The stream is destroyed when the
:class:`CudaStream` object is garbage-collected.
"""
from typing import Any, Literal, Optional

import cupy as cp  # type: ignore
import cuda.bindings.runtime as cudart  # type: ignore


def check_cuda_errors(result: cudart.cudaError_t) -> Any:
    """Check CUDA API result and raise on error.

    Args:
        result: CUDA error return value (e.g. from cudaStreamCreate).
            If the first element indicates an error, raises RuntimeError.

    Returns:
        The non-error part of the result (e.g. the created handle when
        len(result) == 2), or None when len(result) == 1.
    """
    if result[0].value:
        raise RuntimeError(f"CUDA error code={result[0].value}")
    if len(result) == 1:
        return None
    if len(result) == 2:
        return result[1]
    return result[1:]


class CudaStream:
    """RAII wrapper for a CUDA stream with context-manager support.

    Creates a CUDA stream on initialization and destroys it in :meth:`__del__`.
    Use ``with stream:`` to set CuPy's current stream for the block; the
    context manager does not synchronize or destroy on exit. Call
    :meth:`synchronize` explicitly when needed.
    """

    def __init__(self) -> None:
        """Create a new CUDA stream."""
        self._handle: Optional[Any] = None
        self._cupy_stream = None  # Set in __enter__, cleared in __exit__
        self._handle = check_cuda_errors(cudart.cudaStreamCreate())

    def __enter__(self) -> "CudaStream":
        """Set CuPy's current stream for the following block.

        Enters CuPy's ExternalStream so that ``with stream:`` has the same
        effect as ``with cp.cuda.ExternalStream(int(stream.handle)):``.
        Does not synchronize or destroy the stream on exit.
        Not re-entrant: nesting ``with stream:`` with the same instance raises.

        Returns:
            self (this CudaStream).
        """
        if self._cupy_stream is not None:
            raise RuntimeError(
                "CudaStream is not re-entrant; already used as a context manager"
            )
        if self._handle is None:
            raise RuntimeError("CudaStream handle is no longer valid (stream destroyed)")
        self._cupy_stream = cp.cuda.ExternalStream(int(self._handle))
        self._cupy_stream.__enter__()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> Literal[False]:
        """Restore CuPy's previous stream; do not synchronize or destroy."""
        if self._cupy_stream is not None:
            self._cupy_stream.__exit__(exc_type, exc_val, exc_tb)
            self._cupy_stream = None
        return False

    def __del__(self) -> None:
        """Destroy the CUDA stream.

        Force-exits the CuPy ExternalStream first (if still entered) so
        CuPy's stream stack stays balanced; then destroys the handle.
        """
        if self._cupy_stream is not None:
            self._cupy_stream.__exit__(None, None, None)
            self._cupy_stream = None
        if self._handle is not None:
            try:
                check_cuda_errors(cudart.cudaStreamDestroy(self._handle))
            except (RuntimeError, OSError):
                pass
            self._handle = None

    def synchronize(self) -> None:
        """Synchronize the CUDA stream.

        Call this explicitly when synchronization is needed. The context
        manager does not synchronize on exit.
        """
        if self._handle is not None:
            check_cuda_errors(cudart.cudaStreamSynchronize(self._handle))

    @property
    def handle(self) -> Any:
        """The raw CUDA stream handle.

        Valid for the lifetime of the object. Raises RuntimeError if the
        stream has already been destroyed (e.g. after __del__ ran).
        """
        if self._handle is None:
            raise RuntimeError("CudaStream handle is no longer valid (stream destroyed)")
        return self._handle

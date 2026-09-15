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

"""gpu3gppchan — GPU-accelerated 3GPP channel models (TR 38.901).

This package wraps the gpu3gppchan C++ / CUDA library via nanobind, providing:

- System-level stochastic channel modelling (``StatisticalChannel``)
- Link-level TDL / CDL fading channels (``FadingChannel``)
- OFDM modulation / demodulation helpers
- GPU array wrappers and noise utilities
"""

__version__ = "0.1.4"

__all__ = [
    "__version__",
    # Link-level TDL
    "TdlConfig",
    "TdlChan",
    # Link-level CDL
    "CdlConfig",
    "CdlChan",
    # Enums
    "Scenario",
    "SensingTargetType",
    "UeType",
    # ISAC
    "Coordinate",
    "SpstParam",
    "StParam",
    # Topology / antenna
    "AntPanelConfig",
    "UtParamCfg",
    "CellParam",
    # Configs
    "SimConfig",
    "SystemLevelConfig",
    "LinkLevelConfig",
    "ExternalConfig",
    # System-level channel
    "StatisChanModel",
    "LinkParams",
    "ClusterParams",
    "StatisticalChannel",
    # Noise
    "GauNoiseAdder",
    # OFDM
    "CarrierParams",
    "OfdmModulate",
    "OfdmDeModulate",
    # GPU arrays
    "CudaArrayComplexFloat",
    "CudaArrayFloat",
    "CudaArrayHalf",
    # High-level wrappers
    "FadingChannel",
    "TdlChannelConfig",
    "CdlChannelConfig",
]

try:
    from ._gpu3gppchan import (  # type: ignore[import-not-found]
        # Link-level TDL
        TdlConfig,
        TdlChan,
        # Link-level CDL
        CdlConfig,
        CdlChan,
        # Enums
        Scenario,
        SensingTargetType,
        UeType,
        # ISAC structures
        Coordinate,
        SpstParam,
        StParam,
        # Topology / antenna
        AntPanelConfig,
        UtParamCfg,
        CellParam,
        # Configuration objects
        SimConfig,
        SystemLevelConfig,
        LinkLevelConfig,
        ExternalConfig,
        # System-level channel
        StatisChanModel,
        LinkParams,
        ClusterParams,
        # Noise
        GauNoiseAdder,
        # OFDM
        CarrierParams,
        OfdmModulate,
        OfdmDeModulate,
        # GPU array wrappers
        CudaArrayComplexFloat,
        CudaArrayFloat,
        CudaArrayHalf,
    )
except ImportError as exc:
    raise ImportError(
        "Failed to import the _gpu3gppchan C++ extension. "
        "Make sure the package was built with CMake and the extension module "
        "(_gpu3gppchan*) is present in the gpu3gppchan package directory. "
        f"Original error: {exc}"
    ) from exc


# Top-level module names used by the wrappers, mapped to their PyPI distributions.
_GPU_DISTRIBUTION_FOR_MODULE = {
    "cupy": "cupy-cuda13x",
    "cuda": "cuda-bindings",
}


def _gpu_dependency_error(attr: str, exc: ImportError) -> ImportError:
    """Build an actionable error for a missing required GPU dependency."""
    root_module = (getattr(exc, "name", None) or "").split(".")[0]
    missing_distribution = (
        _GPU_DISTRIBUTION_FOR_MODULE.get(root_module)
        or "cupy-cuda13x/cuda-bindings"
    )
    return ImportError(
        f"gpu3gppchan.{attr} needs {missing_distribution}, which is a required "
        "dependency of gpu3gppchan but is not importable; this environment is "
        "incomplete.\n"
        "Repair it with: pip install --force-reinstall gpu3gppchan\n"
        f"Original error: {exc}"
    )


# Import the GPU-heavy wrappers lazily to keep the raw native API inexpensive.
def __getattr__(name: str) -> type:
    """Lazy-import high-level wrappers and report missing dependencies clearly."""
    if name not in (
        "StatisticalChannel",
        "FadingChannel",
        "TdlChannelConfig",
        "CdlChannelConfig",
    ):
        raise AttributeError(f"module 'gpu3gppchan' has no attribute {name!r}")
    try:
        if name == "StatisticalChannel":
            from .statistical_channel import StatisticalChannel
            return StatisticalChannel
        if name == "FadingChannel":
            from .fading_channel import FadingChannel
            return FadingChannel
        from .channel_config import CdlChannelConfig, TdlChannelConfig
        return TdlChannelConfig if name == "TdlChannelConfig" else CdlChannelConfig
    except ImportError as exc:
        root_module = (getattr(exc, "name", None) or "").split(".")[0]
        if root_module in _GPU_DISTRIBUTION_FOR_MODULE:
            raise _gpu_dependency_error(name, exc) from exc
        raise

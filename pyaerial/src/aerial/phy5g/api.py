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

"""pyAerial - Generic API definitions."""
from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from typing import Any
from typing import Generic
from typing import TypeVar

import numpy as np
import cupy as cp  # type: ignore

from aerial.util.cuda import CudaStream

Array = TypeVar("Array", np.ndarray, cp.ndarray)
_SlotConfigT = TypeVar("_SlotConfigT", bound="SlotConfig")
_PipelineConfigT = TypeVar("_PipelineConfigT", bound="PipelineConfig")


@dataclass
class SlotConfig:
    """An empty base class for all slot configuration data classes."""
    pass


@dataclass
class PipelineConfig:
    """An empty base class for all pipeline configuration data classes."""
    pass


class Pipeline(ABC, Generic[_SlotConfigT]):
    """A generic pipeline base class."""
    pass


class PipelineFactory(ABC, Generic[_PipelineConfigT]):
    """A generic pipeline factory defining the interface that the factories need to implement."""

    @abstractmethod
    def create(self, config: _PipelineConfigT, cuda_stream: CudaStream, **kwargs: Any) -> Pipeline:
        """Create the pipeline.

        Args:
            config (PipelineConfig): Pipeline configuration. Note that for the implementation of
                this method, a `PipelineConfig` may also be subclassed to implement an arbitrary
                pipeline configuration.
            cuda_stream (CudaStream): CUDA stream used to run the pipeline. Use ``with stream:``
                to scope work; call ``stream.synchronize()`` explicitly when sync is needed.

        Returns:
            Pipeline: A pipeline object, the class of which is derived from `Pipeline`.
        """
        raise NotImplementedError

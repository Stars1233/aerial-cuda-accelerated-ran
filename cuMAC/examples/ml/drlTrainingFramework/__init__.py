# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""DRL link adaptation training framework for full-buffer traffic.

A learned MCS selector that plugs into an existing system-level simulator. The
simulator keeps owning the channel, the resource allocation, the MU-MIMO
co-scheduling and the precoder; this package turns the MCS into the action of a
reinforcement learning agent and trains it through a four-phase curriculum
(teacher cloning, optional actor bootstrap, critic warmup, PPO).

Integrating it means implementing one interface,
:class:`~drlTrainingFramework.adapter.LinkAdaptationSimulator`, which publishes
a :class:`~drlTrainingFramework.adapter.SlotContext` per slot and consumes a
:class:`~drlTrainingFramework.adapter.SlotOutcome`. See ``INTEGRATION.md`` for
the field-by-field mapping and ``example_simulator.py`` for a working reference.
"""

from __future__ import annotations

__all__ = [
    "DEFAULT_PHY",
    "LA_OVERRIDE_SECTIONS",
    "MCS_SPECTRAL_EFF",
    "NUM_MCS",
    "PRBG_FEATURE_DIM",
    "PRBG_FEATURE_NAMES",
    "UE_FEATURE_DIM",
    "UE_FEATURE_NAMES",
    "BootstrapMetrics",
    "CarrierProfile",
    "CloneMetrics",
    "CriticWarmupMetrics",
    "EnvFactory",
    "LaEnvConfig",
    "LaModelConfig",
    "LaObservation",
    "LaRolloutBatch",
    "LaTrainingConfig",
    "LaTrainingResult",
    "LaValidationMetrics",
    "LinkAdaptationActorCritic",
    "LinkAdaptationEnv",
    "LinkAdaptationSimulator",
    "LinkAdaptationTrainingFramework",
    "NrPhyAbstraction",
    "ObservationNormalizer",
    "PPOMetrics",
    "PhyAbstraction",
    "SimulatorFactory",
    "SlotContext",
    "SlotOutcome",
    "build_la_observation",
    "candidate_bler_table",
    "debt_se_aware_mcs",
    "feature_names",
    "gaussian_mcs_targets",
    "greedy_mcs",
    "load_la_env_config",
    "load_la_model_config",
    "load_la_training_config",
    "observation_dim",
    "parameter_groups_for_weight_decay",
    "parse_la_config_overrides",
    "power_to_db",
    "simulator_env_factory",
]

from .adapter import (
    CarrierProfile,
    LinkAdaptationSimulator,
    SimulatorFactory,
    SlotContext,
    SlotOutcome,
)
from .config import (
    LA_OVERRIDE_SECTIONS,
    NUM_MCS,
    LaEnvConfig,
    LaModelConfig,
    LaTrainingConfig,
    load_la_env_config,
    load_la_model_config,
    load_la_training_config,
    parse_la_config_overrides,
)
from .env import LinkAdaptationEnv
from .features import (
    PRBG_FEATURE_DIM,
    PRBG_FEATURE_NAMES,
    UE_FEATURE_DIM,
    UE_FEATURE_NAMES,
    LaObservation,
    build_la_observation,
    feature_names,
    observation_dim,
)
from .model import (
    LinkAdaptationActorCritic,
    ObservationNormalizer,
    parameter_groups_for_weight_decay,
)
from .phy import (
    DEFAULT_PHY,
    MCS_SPECTRAL_EFF,
    NrPhyAbstraction,
    PhyAbstraction,
    power_to_db,
)
from .teacher import (
    candidate_bler_table,
    debt_se_aware_mcs,
    gaussian_mcs_targets,
    greedy_mcs,
)
from .training import (
    BootstrapMetrics,
    CloneMetrics,
    CriticWarmupMetrics,
    EnvFactory,
    LaRolloutBatch,
    LaTrainingResult,
    LaValidationMetrics,
    LinkAdaptationTrainingFramework,
    PPOMetrics,
    simulator_env_factory,
)

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

"""Actor-critic policy for DRL link adaptation.

One shared encoder feeds a categorical actor over MCS indices and a scalar
critic. The policy is applied per UE row, so the same weights serve however
many UEs the scheduler happens to admit in a slot.

Observations are standardized by a running normalizer that lives inside the
module. Keeping it in the ``state_dict`` rather than beside it means a
checkpoint is self-contained: the statistics a deployed encoder needs travel
with the weights that were trained under them.
"""

from __future__ import annotations

__all__ = [
    "LinkAdaptationActorCritic",
    "ObservationNormalizer",
    "parameter_groups_for_weight_decay",
]

from typing import Sequence

import torch
from torch import nn

from .config import LaModelConfig


class ObservationNormalizer(nn.Module):
    """Running per-feature standardization with Welford statistics.

    Only rows flagged active contribute, so padded UE slots cannot drag the
    mean toward the all-zero observation they carry.
    """

    def __init__(self, num_features: int, *, clip: float = 10.0) -> None:
        super().__init__()
        self.clip = float(clip)
        self.register_buffer("mean", torch.zeros(int(num_features)))
        # Sum of squared deviations from the running mean, not a variance, so
        # it starts at zero alongside count: the parallel-variance update in
        # update() then reduces to exactly the first batch's own sum. A nonzero
        # start would leave a permanent bias of m2_initial / count in the
        # variance that forward() divides by. The variance floor lives in
        # forward() instead, where it cannot accumulate.
        self.register_buffer("m2", torch.zeros(int(num_features)))
        self.register_buffer("count", torch.zeros((), dtype=torch.float64))

    @torch.no_grad()
    def update(
        self,
        observation: torch.Tensor,
        active: torch.Tensor | None = None,
    ) -> None:
        """Fold a batch of observations into the running statistics."""

        flat = observation.reshape(-1, observation.shape[-1])
        if active is not None:
            flat = flat[active.reshape(-1)]
        batch_count = flat.shape[0]
        if batch_count == 0:
            return

        batch_mean = flat.mean(dim=0)
        batch_m2 = flat.var(dim=0, unbiased=False) * batch_count
        delta = batch_mean - self.mean
        total = self.count + batch_count
        self.mean += delta * (batch_count / total).to(self.mean.dtype)
        self.m2 += batch_m2 + delta.square() * (
            self.count.to(delta.dtype) * batch_count / total.to(delta.dtype)
        )
        self.count.copy_(total)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        """Standardize and clamp, passing observations through until warm."""

        if float(self.count) < 2.0:
            return observation
        variance = (self.m2 / self.count.to(self.m2.dtype)).clamp_min(1e-8)
        standardized = (observation - self.mean) / variance.sqrt()
        return standardized.clamp(-self.clip, self.clip)


def _mlp(
    input_dim: int,
    hidden: Sequence[int],
    output_dim: int,
    *,
    output_gain: float,
) -> nn.Sequential:
    """Orthogonally initialized MLP with SiLU activations between layers."""

    layers: list[nn.Module] = []
    width = int(input_dim)
    for size in hidden:
        linear = nn.Linear(width, int(size))
        nn.init.orthogonal_(linear.weight, gain=2.0**0.5)
        nn.init.zeros_(linear.bias)
        layers.append(linear)
        layers.append(nn.SiLU())
        width = int(size)
    readout = nn.Linear(width, int(output_dim))
    nn.init.orthogonal_(readout.weight, gain=float(output_gain))
    nn.init.zeros_(readout.bias)
    layers.append(readout)
    return nn.Sequential(*layers)


class LinkAdaptationActorCritic(nn.Module):
    """Shared-encoder actor-critic over discrete MCS actions.

    Args:
        obs_dim: Observation width, from
            :func:`~drlTrainingFramework.features.observation_dim`.
        action_dim: Number of discrete MCS actions.
        config: Architecture and normalization settings.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        config: LaModelConfig | None = None,
    ) -> None:
        super().__init__()
        if obs_dim <= 0 or action_dim <= 0:
            raise ValueError("obs_dim and action_dim must be positive")
        self.config = config if config is not None else LaModelConfig()
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)

        widths = list(self.config.encoder_layers)
        self.encoder = nn.Sequential(
            _mlp(
                self.obs_dim,
                widths[:-1],
                widths[-1],
                output_gain=2.0**0.5,
            ),
            nn.LayerNorm(widths[-1]),
            nn.SiLU(),
        )
        self.actor = _mlp(
            widths[-1],
            self.config.actor_head_layers,
            self.action_dim,
            output_gain=float(self.config.actor_output_gain),
        )
        self.critic = _mlp(
            widths[-1],
            self.config.critic_head_layers,
            1,
            output_gain=float(self.config.critic_output_gain),
        )
        self.obs_normalizer = ObservationNormalizer(
            self.obs_dim,
            clip=float(self.config.obs_norm_clip),
        )
        self.normalize_observations = bool(self.config.obs_normalization)

    def encode(self, observation: torch.Tensor) -> torch.Tensor:
        """Shared representation for one batch of UE observations."""

        if self.normalize_observations:
            observation = self.obs_normalizer(observation)
        return self.encoder(observation)

    def forward(
        self,
        observation: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(logits, value)`` with value squeezed to the batch shape."""

        latent = self.encode(observation)
        return self.actor(latent), self.critic(latent).squeeze(-1)

    def act(
        self,
        observation: torch.Tensor,
        *,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample or take the modal action.

        Returns ``(action, log_prob, entropy, value)``.
        """

        logits, value = self.forward(observation)
        distribution = torch.distributions.Categorical(logits=logits)
        action = (
            logits.argmax(dim=-1) if deterministic else distribution.sample()
        )
        return action, distribution.log_prob(action), distribution.entropy(), value

    def evaluate_actions(
        self,
        observation: torch.Tensor,
        action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Score stored actions under the current policy for a PPO update."""

        logits, value = self.forward(observation)
        distribution = torch.distributions.Categorical(logits=logits)
        return (
            distribution.log_prob(action),
            distribution.entropy(),
            value,
        )

    def actor_parameters(self) -> list[nn.Parameter]:
        """Encoder and actor parameters, i.e. everything the policy uses."""

        return [*self.encoder.parameters(), *self.actor.parameters()]

    def critic_parameters(self) -> list[nn.Parameter]:
        """Value-head parameters only, for the frozen-encoder warmup."""

        return list(self.critic.parameters())


def parameter_groups_for_weight_decay(
    module: nn.Module,
    weight_decay: float,
) -> list[dict[str, object]]:
    """Split parameters so decay applies to weight matrices only.

    Decaying biases and LayerNorm gains shrinks the representation rather than
    regularizing it.
    """

    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []
    for parameter in module.parameters():
        if not parameter.requires_grad:
            continue
        (decay if parameter.ndim >= 2 else no_decay).append(parameter)
    return [
        {"params": decay, "weight_decay": float(weight_decay)},
        {"params": no_decay, "weight_decay": 0.0},
    ]

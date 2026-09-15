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

"""Observation features for DRL link adaptation.

The agent chooses one MCS per scheduled physical UE per slot, so an observation
row describes one UE. Each row has two lanes:

* A **per-PRBG lane** of :data:`PRBG_FEATURE_DIM` features repeated for every
  PRB group, describing what the UE was actually given on that PRB group.
* A **per-UE lane** of :data:`UE_FEATURE_DIM` wideband scalars describing state
  the UE reports once, not per PRB group.

The per-PRBG lane is what makes this work under non-contiguous (type-0)
resource allocation. Co-scheduling is decided independently on every PRB group,
so one UE can hold non-contiguous PRB groups and share each of them with a
different set of UEs, at a different rank and against a different total layer
count. Both of those are therefore per-PRBG features rather than the per-UE
scalars they would be under contiguous allocation with one co-scheduled set per
subband.

Everything here is derived from what a base station actually holds at decision
time: reported CQI and rank, feedback ages, the sliding BLER window, and the
link budget computed from the *channel estimate*. The realized channel is never
read, which is what separates the policy input from the teacher label.
"""

from __future__ import annotations

__all__ = [
    "PRBG_FEATURE_DIM",
    "PRBG_FEATURE_NAMES",
    "UE_FEATURE_DIM",
    "UE_FEATURE_NAMES",
    "LaObservation",
    "build_la_observation",
    "feature_names",
    "layer_correlation_features",
    "layer_counts",
    "observation_dim",
    "scatter_layers_to_ues",
]

from dataclasses import dataclass

import torch

from .adapter import CarrierProfile, SlotContext
from .config import LaEnvConfig
from .phy import DB_FLOOR, PhyAbstraction, power_to_db

# Per-PRBG features, in observation order. The last two are the co-scheduling
# descriptors that non-contiguous allocation forces down to PRBG granularity.
PRBG_FEATURE_NAMES = (
    "beamforming_signal_snr",
    "residual_interference_inr",
    "self_layer_max_corr",
    "other_ue_layer_max_corr",
    "mean_layer_corr",
    "allocated",
    "ue_layers",
    "coscheduled_layers",
)

# Per-UE wideband features, in observation order. Deliberately absent, because
# a production stack cannot supply them: per-UE inter-cell interference INR and
# its age, which a UE does not report, and UE speed.
UE_FEATURE_NAMES = (
    "wideband_sinr",
    "bler_windowed",
    "normalized_ue_rank",
    "normalized_cqi_feedback_age",
    "normalized_srs_feedback_age",
    "normalized_rank_feedback_age",
)

PRBG_FEATURE_DIM = len(PRBG_FEATURE_NAMES)
UE_FEATURE_DIM = len(UE_FEATURE_NAMES)


def observation_dim(num_prb_groups: int) -> int:
    """Observation width for a carrier of ``num_prb_groups`` PRB groups."""

    if num_prb_groups <= 0:
        raise ValueError("num_prb_groups must be positive")
    return PRBG_FEATURE_DIM * int(num_prb_groups) + UE_FEATURE_DIM


def feature_names(num_prb_groups: int) -> tuple[str, ...]:
    """Observation feature names in the exact order they are concatenated."""

    if num_prb_groups <= 0:
        raise ValueError("num_prb_groups must be positive")
    prbg_names = tuple(
        f"{name}_prbg_{index}"
        for index in range(int(num_prb_groups))
        for name in PRBG_FEATURE_NAMES
    )
    return (*prbg_names, *UE_FEATURE_NAMES)


@dataclass(frozen=True)
class LaObservation:
    """One slot of link adaptation observations.

    Attributes:
        features: ``[env, ue, obs_dim]`` policy input.
        active: ``[env, ue]`` mask of UEs holding a resource this slot. Only
            active rows produce a reward. This is the scheduled mask and keeps
            that meaning.
        resource_count: ``[env, ue]`` PRBG-layer resources the UE holds across
            all its PRB groups, i.e. the number of EESM terms behind its
            transport block.
        trainable: ``[env, ue]`` active rows whose MCS the policy actually
            chose. A retransmission replays the modulation its first attempt
            used, so the policy's action is discarded there and the sample
            carries no information about the decision being learned: training on
            it would fit the model to labels it did not produce. Every training
            stage filters on this rather than on ``active``.
    """

    features: torch.Tensor
    active: torch.Tensor
    resource_count: torch.Tensor
    trainable: torch.Tensor | None = None

    @property
    def learnable(self) -> torch.Tensor:
        """``trainable`` where the environment supplied it, else ``active``."""

        return self.active if self.trainable is None else self.trainable

    def validate(self, num_envs: int, max_ues: int, obs_dim: int) -> "LaObservation":
        """Check tensor shapes against the environment contract."""

        expected = (int(num_envs), int(max_ues))
        if tuple(self.features.shape) != (*expected, int(obs_dim)):
            raise ValueError(
                "features must have shape "
                f"{(*expected, int(obs_dim))}, got {tuple(self.features.shape)}"
            )
        if self.trainable is not None and tuple(self.trainable.shape) != expected:
            raise ValueError(
                f"trainable must have shape {expected}, "
                f"got {tuple(self.trainable.shape)}"
            )
        for name in ("active", "resource_count"):
            tensor = getattr(self, name)
            if tuple(tensor.shape) != expected:
                raise ValueError(
                    f"{name} must have shape {expected}, got {tuple(tensor.shape)}"
                )
        if self.active.dtype is not torch.bool:
            raise ValueError("active must be a boolean tensor")
        return self


def scatter_layers_to_ues(
    values: torch.Tensor,
    ue_index: torch.Tensor,
    stream_valid: torch.Tensor,
    num_ues: int,
) -> torch.Tensor:
    """Sum a per-layer quantity ``[env, prbg, layer]`` into ``[env, prbg, ue]``."""

    num_envs, num_prb_groups, _ = values.shape
    index = ue_index.clamp(0, num_ues - 1)
    out = torch.zeros(
        (num_envs, num_prb_groups, num_ues),
        device=values.device,
        dtype=values.dtype,
    )
    contribution = torch.where(stream_valid, values, torch.zeros_like(values))
    out.scatter_add_(2, index, contribution)
    return out


def _scatter_layer_max_to_ues(
    values: torch.Tensor,
    ue_index: torch.Tensor,
    stream_valid: torch.Tensor,
    num_ues: int,
) -> torch.Tensor:
    """Reduce a per-layer quantity to its max over each UE's layers."""

    num_envs, num_prb_groups, _ = values.shape
    index = ue_index.clamp(0, num_ues - 1)
    out = torch.zeros(
        (num_envs, num_prb_groups, num_ues),
        device=values.device,
        dtype=values.dtype,
    )
    contribution = torch.where(stream_valid, values, torch.zeros_like(values))
    out.scatter_reduce_(2, index, contribution, reduce="amax", include_self=True)
    return out


def layer_counts(
    context: SlotContext,
    num_ues: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-PRBG layer bookkeeping as ``[env, prbg, ue]``.

    Returns ``(ue_layers, coscheduled_layers)``: how many layers the UE holds
    on that PRB group, and how many layers are co-scheduled there in total.
    Both are zero where the UE is not allocated. Under non-contiguous
    allocation these vary across a single UE's own PRB groups, which is why they
    are per-PRBG quantities.
    """

    stream_valid = context.stream_valid
    dtype = torch.float32
    ue_layers = scatter_layers_to_ues(
        stream_valid.to(dtype),
        context.ue_index,
        stream_valid,
        num_ues,
    )
    allocated = (ue_layers > 0.0).to(dtype)
    total_layers = stream_valid.sum(dim=-1).to(dtype).unsqueeze(-1) * allocated
    return ue_layers, total_layers


def layer_correlation_features(
    effective_channel_est: torch.Tensor,
    ue_index: torch.Tensor,
    stream_valid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-layer spatial overlap with the other layers on the same PRB group.

    Returns ``(self_max, other_ue_max, mean)`` each shaped
    ``[env, prbg, layer]``. ``self_max`` is the largest overlap with another
    layer of the *same* UE, which is what limits that UE's own rank;
    ``other_ue_max`` and ``mean`` describe the multi-user interference the
    precoder has to suppress.
    """

    rows = effective_channel_est
    norm = rows.abs().square().sum(dim=-1, keepdim=True).clamp_min(1e-20).sqrt()
    unit = rows / norm.to(rows.dtype)
    correlation = (unit @ unit.conj().transpose(-1, -2)).abs()

    valid_pair = stream_valid.unsqueeze(-1) & stream_valid.unsqueeze(-2)
    eye = torch.eye(
        rows.shape[-2],
        device=rows.device,
        dtype=torch.bool,
    ).view(1, 1, rows.shape[-2], rows.shape[-2])
    valid_pair = valid_pair & ~eye

    same_ue = ue_index.unsqueeze(-1) == ue_index.unsqueeze(-2)
    self_pair = valid_pair & same_ue
    other_pair = valid_pair & ~same_ue

    zero = torch.zeros_like(correlation)
    self_max = torch.where(self_pair, correlation, zero).amax(dim=-1)
    other_max = torch.where(other_pair, correlation, zero).amax(dim=-1)
    pair_count = valid_pair.sum(dim=-1).clamp_min(1).to(correlation.dtype)
    mean = torch.where(valid_pair, correlation, zero).sum(dim=-1) / pair_count

    invalid = ~stream_valid
    self_max = self_max.masked_fill(invalid, 0.0)
    other_max = other_max.masked_fill(invalid, 0.0)
    mean = mean.masked_fill(invalid, 0.0)
    return self_max, other_max, mean


def build_la_observation(
    context: SlotContext,
    carrier: CarrierProfile,
    la_config: LaEnvConfig,
    *,
    bler_windowed: torch.Tensor,
    phy: PhyAbstraction,
) -> LaObservation:
    """Assemble the link adaptation observation for the current slot.

    Args:
        context: The slot the host simulator published.
        carrier: Static carrier description, supplying the feature normalizers.
        la_config: Feature scaling and clipping.
        bler_windowed: ``[env, ue]`` sliding BLER window before this slot. Owned
            by the environment, not the simulator.
        phy: PHY abstraction, used only to read the reported CQI as an SINR.
    """

    ue_index = context.ue_index
    stream_valid = context.stream_valid
    num_envs = int(carrier.num_envs)
    num_prb_groups = int(carrier.num_prb_groups)
    num_ues = int(carrier.num_ues)
    dtype = context.estimated_signal_watt.dtype

    ue_layers_ngu, coscheduled_ngu = layer_counts(context, num_ues)
    ue_layers_ngu = ue_layers_ngu.to(dtype)
    coscheduled_ngu = coscheduled_ngu.to(dtype)
    allocated_ngu = ue_layers_ngu > 0.0
    layer_norm = ue_layers_ngu.clamp_min(1.0)

    # Mean per-layer post-precoder signal and residual interference for this UE
    # on this PRB group, both from the channel estimate: the policy sees the
    # link budget the scheduler predicted, never the realized one.
    noise = float(carrier.noise_per_prbg_watt)
    signal_ngu = (
        scatter_layers_to_ues(
            context.estimated_signal_watt.to(dtype),
            ue_index,
            stream_valid,
            num_ues,
        )
        / layer_norm
    )
    interference_ngu = (
        scatter_layers_to_ues(
            context.estimated_residual_interference_watt.to(dtype),
            ue_index,
            stream_valid,
            num_ues,
        )
        / layer_norm
    )
    signal_snr_db = power_to_db(signal_ngu / noise, floor_db=DB_FLOOR)
    interference_inr_db = power_to_db(
        interference_ngu / noise,
        floor_db=DB_FLOOR,
    )
    signal_snr_db = torch.where(
        allocated_ngu,
        signal_snr_db,
        torch.zeros_like(signal_snr_db),
    )
    interference_inr_db = torch.where(
        allocated_ngu,
        interference_inr_db,
        torch.zeros_like(interference_inr_db),
    )

    self_max_l, other_max_l, mean_l = layer_correlation_features(
        context.effective_channel_est,
        ue_index,
        stream_valid,
    )
    self_max_ngu = _scatter_layer_max_to_ues(
        self_max_l.to(dtype),
        ue_index,
        stream_valid,
        num_ues,
    )
    other_max_ngu = _scatter_layer_max_to_ues(
        other_max_l.to(dtype),
        ue_index,
        stream_valid,
        num_ues,
    )
    mean_ngu = (
        scatter_layers_to_ues(mean_l.to(dtype), ue_index, stream_valid, num_ues)
        / layer_norm
    )

    max_coscheduled_layers = float(carrier.max_coscheduled_layers)
    max_ue_layers = float(carrier.max_layers_per_ue)
    sinr_norm = float(la_config.sinr_feature_norm_db)

    prbg_features = torch.stack(
        (
            signal_snr_db / sinr_norm,
            interference_inr_db / sinr_norm,
            self_max_ngu,
            other_max_ngu,
            mean_ngu,
            allocated_ngu.to(dtype),
            ue_layers_ngu / max_ue_layers,
            # Total layers co-scheduled on this PRB group. Under non-contiguous
            # allocation this differs across the UE's own PRB groups, which is
            # exactly the coupling the policy needs: the same reported CQI buys
            # a different post-precoder SINR against 4 co-scheduled layers than
            # against 16.
            coscheduled_ngu / max_coscheduled_layers,
        ),
        dim=-1,
    )
    # [env, prbg, ue, feature] -> [env, ue, prbg * feature], PRBG major.
    prbg_obs = prbg_features.permute(0, 2, 1, 3).reshape(
        num_envs,
        num_ues,
        num_prb_groups * PRBG_FEATURE_DIM,
    )

    age_norm = float(la_config.resolved_feedback_age_norm_slots(carrier))
    wideband_sinr = phy.cqi_to_sinr_db(context.wideband_cqi).to(dtype) / sinr_norm
    reported_rank = context.reported_rank.to(dtype) / max_ue_layers

    ue_obs = torch.stack(
        (
            wideband_sinr,
            bler_windowed.to(dtype),
            reported_rank,
            context.csi_age_slots.to(dtype) / age_norm,
            context.srs_age_slots.to(dtype) / age_norm,
            context.rank_age_slots.to(dtype) / age_norm,
        ),
        dim=-1,
    )

    clip = float(la_config.obs_clip)
    features = torch.cat((prbg_obs, ue_obs), dim=-1).clamp(-clip, clip)
    resource_count = ue_layers_ngu.sum(dim=1)
    active = resource_count > 0.0
    trainable = active
    if context.retransmitting is not None:
        trainable = active & ~context.retransmitting
    return LaObservation(
        features=features,
        active=active,
        resource_count=resource_count,
        trainable=trainable,
    )

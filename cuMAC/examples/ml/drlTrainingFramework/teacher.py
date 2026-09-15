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

"""Perfect-information teacher for link adaptation cloning.

The teacher is a perfect-information MCS selector: it reads the *realized*
per-layer SINR that the policy never sees. That makes it an upper reference
rather than a deployable scheme, and cloning it is what gives PPO a sane
starting point instead of a uniform policy over 28 MCS.

Its three regimes are what "debt-SE-aware" means:

* Below the BLER target the selector is unconstrained, so it maximizes expected
  goodput ``SE * (1 - BLER)`` and will happily overshoot the target this slot.
* At or above target it becomes bounded and takes the highest MCS still meeting
  the target.
* Once the sliding BLER window exceeds ``target + cushion`` the UE is in debt
  and the bound tightens to ``debt_bler_bound``, forcing repayment before the
  selector may chase spectral efficiency again.

The selectors here take a precomputed ``[env, ue, mcs]`` BLER table rather than
raw SINR, because the environment already needs that table to score whichever
MCS the agent picked. Building it costs one EESM pass over every scheduled
PRBG-layer resource, which dominates the slot, so it is computed once.

Everything below goes through :class:`~drlTrainingFramework.phy.PhyAbstraction`,
so substituting a host simulator's link-to-system mapping changes the teacher's
labels without touching this file.
"""

from __future__ import annotations

__all__ = [
    "candidate_bler_table",
    "debt_se_aware_mcs",
    "gaussian_mcs_targets",
    "greedy_mcs",
]

import torch

from .phy import PhyAbstraction


def candidate_bler_table(
    per_ue_sinr_db: torch.Tensor,
    per_ue_resource_mask: torch.Tensor,
    phy: PhyAbstraction,
) -> tuple[torch.Tensor, torch.Tensor]:
    """EESM effective SINR and BLER for every MCS.

    Args:
        per_ue_sinr_db: ``[env, ue, resource]`` per-layer SINR in dB.
        per_ue_resource_mask: ``[env, ue, resource]`` scheduled-resource mask.
        phy: PHY abstraction supplying the EESM aggregation and BLER curves.

    Returns:
        ``(effective_sinr_db, bler)`` both shaped ``[env, ue, mcs]``. The EESM
        beta is MCS dependent, so the effective SINR is too.
    """

    effective_sinr_db = phy.eesm_effective_sinr_all_mcs(
        per_ue_sinr_db,
        per_ue_resource_mask,
    )
    num_mcs = int(phy.num_mcs)
    candidate_mcs = (
        torch.arange(num_mcs, device=per_ue_sinr_db.device, dtype=torch.long)
        .view(1, 1, num_mcs)
        .expand_as(effective_sinr_db)
    )
    return effective_sinr_db, phy.bler_from_sinr_mcs(effective_sinr_db, candidate_mcs)


def debt_se_aware_mcs(
    candidate_bler: torch.Tensor,
    current_bler: torch.Tensor,
    scheduled: torch.Tensor,
    spectral_eff: torch.Tensor,
    *,
    target_bler: float,
    bler_cushion: float,
    debt_bler_bound: float,
) -> torch.Tensor:
    """Teacher MCS ``[env, ue]`` from a candidate BLER table.

    Args:
        candidate_bler: ``[env, ue, mcs]`` predicted BLER per MCS.
        current_bler: ``[env, ue]`` sliding BLER window before this slot.
        scheduled: ``[env, ue]`` mask of UEs holding a resource this slot.
        spectral_eff: ``[mcs]`` per-layer spectral efficiency.
        target_bler: Operating BLER target.
        bler_cushion: Slack above the target before a UE counts as in debt.
        debt_bler_bound: Tightened BLER bound applied while in debt.
    """

    num_mcs = candidate_bler.shape[-1]
    candidate_mcs = (
        torch.arange(num_mcs, device=candidate_bler.device, dtype=torch.long)
        .view(1, 1, num_mcs)
        .expand_as(candidate_bler)
    )
    target = max(float(target_bler), 1e-6)
    cushion = max(float(bler_cushion), 0.0)
    debt_bound = max(float(debt_bler_bound), 0.0)

    debt_region = current_bler > target + cushion
    bound = torch.where(
        debt_region,
        torch.full_like(current_bler, debt_bound),
        torch.full_like(current_bler, target),
    )
    feasible = candidate_bler <= bound.unsqueeze(-1)
    feasible_mcs = torch.where(
        feasible,
        candidate_mcs,
        torch.full_like(candidate_mcs, -1),
    )
    bounded_mcs = feasible_mcs.amax(dim=-1)
    min_bler_mcs = candidate_bler.argmin(dim=-1)
    bounded_mcs = torch.where(feasible.any(dim=-1), bounded_mcs, min_bler_mcs)

    goodput = spectral_eff.to(
        device=candidate_bler.device,
        dtype=candidate_bler.dtype,
    ).view(1, 1, num_mcs) * (1.0 - candidate_bler)
    goodput_mcs = goodput.argmax(dim=-1)
    selected = torch.where(current_bler < target, goodput_mcs, bounded_mcs)
    return torch.where(scheduled, selected, torch.zeros_like(selected))


def greedy_mcs(
    candidate_bler: torch.Tensor,
    scheduled: torch.Tensor,
    *,
    target_bler: float,
) -> torch.Tensor:
    """Highest MCS meeting ``target_bler``, with no BLER-history feedback.

    Falls back to the minimum-BLER MCS when the target is infeasible. This is
    the no-debt reference used to bound evaluation, not a cloning target.
    """

    num_mcs = candidate_bler.shape[-1]
    candidate_mcs = (
        torch.arange(num_mcs, device=candidate_bler.device, dtype=torch.long)
        .view(1, 1, num_mcs)
        .expand_as(candidate_bler)
    )
    feasible = candidate_bler <= float(target_bler)
    feasible_mcs = torch.where(
        feasible,
        candidate_mcs,
        torch.full_like(candidate_mcs, -1),
    )
    selected = torch.where(
        feasible.any(dim=-1),
        feasible_mcs.amax(dim=-1),
        candidate_bler.argmin(dim=-1),
    )
    return torch.where(scheduled, selected, torch.zeros_like(selected))


def gaussian_mcs_targets(
    target: torch.Tensor,
    action_dim: int,
    sigma: float,
) -> torch.Tensor:
    """Gaussian soft labels over the MCS index for teacher cloning.

    A one-hot target penalizes an off-by-one MCS as hard as an off-by-ten one,
    which is wrong: adjacent MCS differ by a fraction of a dB of required SINR.
    ``sigma <= 0`` recovers the one-hot target.
    """

    if sigma <= 0.0:
        return torch.nn.functional.one_hot(
            target.long(),
            num_classes=int(action_dim),
        ).to(torch.float32)
    index = torch.arange(
        int(action_dim),
        device=target.device,
        dtype=torch.float32,
    )
    distance = index.view((1,) * target.dim() + (int(action_dim),)) - target.to(
        torch.float32
    ).unsqueeze(-1)
    weights = torch.exp(-0.5 * (distance / float(sigma)).square())
    return weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)

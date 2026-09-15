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

"""DRL link adaptation environment over a host system-level simulator.

:class:`LinkAdaptationEnv` turns a
:class:`~drlTrainingFramework.adapter.LinkAdaptationSimulator` into a
reinforcement learning environment whose only action is the MCS. It owns no
radio state: the channel, the resource allocation, the co-scheduling and the
precoder all come from the simulator as a
:class:`~drlTrainingFramework.adapter.SlotContext`, and the environment adds the
link adaptation layer on top.

The slot is split where a scheduler would normally keep it joined::

    reset()/advance()  simulator publishes the slot -> observation, no commitment
    step(mcs)          BLER -> ACK -> sliding BLER window, outer loop, advance

so the agent observes the resources and the co-scheduling it was actually given
before committing to a modulation.

Non-contiguous (type-0) allocation is the general case here. A UE can hold
non-contiguous PRB groups and be co-scheduled with a different set of UEs on
each, so its layer count and the total co-scheduled layer count vary across its
own allocation. Those two are therefore per-PRBG observation features, while the
transport block spans the whole allocation through a single EESM aggregation.

Traffic is full buffer: every scheduled UE always has data, so a transport block
carries the rate its MCS and resource count imply and no queue state enters the
observation or the reward.
"""

from __future__ import annotations

__all__ = ["LinkAdaptationEnv"]

from typing import Any, Mapping

import torch

from .adapter import CarrierProfile, LinkAdaptationSimulator, SlotContext, SlotOutcome
from .config import LaEnvConfig
from .features import LaObservation, build_la_observation, observation_dim
from .teacher import candidate_bler_table, debt_se_aware_mcs, greedy_mcs

# SINR reported for a resource a UE does not hold. Far below any usable
# operating point, and never read because the resource mask excludes it.
_UNSCHEDULED_SINR_DB = -100.0


class LinkAdaptationEnv:
    """Per-slot link adaptation with an agent-selected MCS.

    Args:
        simulator: The host simulator publishing slots.
        la_config: Link adaptation loop settings. Defaults to
            :class:`~drlTrainingFramework.config.LaEnvConfig` defaults.
        seed: Seed for the environment's own ACK sampling. The simulator is
            seeded separately through ``reset(seed=...)``.
    """

    def __init__(
        self,
        simulator: LinkAdaptationSimulator,
        la_config: LaEnvConfig | None = None,
        *,
        seed: int = 0,
    ) -> None:
        self.simulator = simulator
        self.carrier: CarrierProfile = simulator.carrier.validate()
        self.phy = simulator.phy
        self.la_config = (la_config or LaEnvConfig()).validate()

        self.device = self.carrier.torch_device
        self.num_envs = int(self.carrier.num_envs)
        self.num_ues = int(self.carrier.num_ues)
        self.num_prb_groups = int(self.carrier.num_prb_groups)
        self.num_mcs = int(self.phy.num_mcs)
        self.obs_dim = observation_dim(self.num_prb_groups)
        self.action_dim = self.la_config.action_dim

        if self.la_config.resolved_max_scheduled_ues(self.carrier) != self.num_ues:
            raise ValueError(
                "max_scheduled_ues must equal CarrierProfile.num_ues; the action "
                "axis is the physical UE index so that a UE keeps its slot "
                "across PRB groups and across slots"
            )
        if self.la_config.drl_absolute_mcs and self.action_dim != self.num_mcs:
            raise ValueError(
                f"absolute MCS actions need action_dim == {self.num_mcs} to match "
                f"the PHY abstraction, got {self.action_dim}"
            )

        self.mcs_spectral_eff = self.phy.mcs_spectral_eff(
            device=self.device,
            dtype=torch.float32,
        )
        self.action_deltas = torch.arange(
            int(self.la_config.action_delta_min),
            int(self.la_config.action_delta_max) + 1,
            device=self.device,
            dtype=torch.long,
        )
        self.generator = torch.Generator(device=self.device)
        self.generator.manual_seed(int(seed))

        self.slot = 0
        # Sliding BLER window, owned here rather than by the simulator because
        # the reward, the teacher's debt regime and the observation must all
        # read the same value. Starting at the target means "assume on target
        # until measured", which avoids an opening transient in which the
        # teacher is unconstrained and maximally aggressive.
        self.bler_windowed = torch.full(
            (self.num_envs, self.num_ues),
            float(self.la_config.target_bler),
            device=self.device,
            dtype=torch.float32,
        )
        self.olla_offset_db = torch.full(
            (self.num_envs, self.num_ues),
            float(self.la_config.olla_initial_offset_db),
            device=self.device,
            dtype=torch.float32,
        )

        self._context: SlotContext | None = None
        self._per_ue_sinr_db: torch.Tensor | None = None
        self._per_ue_resource_mask: torch.Tensor | None = None
        self._candidate_bler: torch.Tensor | None = None
        self._scheduled: torch.Tensor | None = None
        self._resource_count: torch.Tensor | None = None
        self._illa_sinr_cache: torch.Tensor | None = None
        self._la_observation: LaObservation | None = None

    # ------------------------------------------------------------------
    # Public slot API
    # ------------------------------------------------------------------

    def reset(self, *, seed: int | None = None) -> LaObservation:
        """Start an episode and return the slot-zero link adaptation input.

        ``seed`` is forwarded to the simulator and also re-seeds the ACK draw,
        so two rollouts sharing a seed replay the same radio conditions and the
        same error realizations. That is what makes the student, the teacher and
        the outer-loop baseline comparable slot for slot.
        """

        if seed is not None:
            self.generator.manual_seed(int(seed))
        self.slot = 0
        self.bler_windowed = torch.full_like(
            self.bler_windowed,
            float(self.la_config.target_bler),
        )
        self.olla_offset_db = torch.full_like(
            self.olla_offset_db,
            float(self.la_config.olla_initial_offset_db),
        )
        context = self.simulator.reset(seed=seed)
        return self._begin_slot(context)

    def step(
        self,
        action: torch.Tensor | None,
    ) -> tuple[LaObservation, torch.Tensor, torch.Tensor, dict[str, Any]]:
        """Apply an MCS action and autoreset at the episode boundary."""

        return self._step_impl(action, auto_reset=True)

    def step_eval(
        self,
        action: torch.Tensor | None,
    ) -> tuple[LaObservation, torch.Tensor, torch.Tensor, dict[str, Any]]:
        """Apply an MCS action without episode autoreset."""

        return self._step_impl(action, auto_reset=False)

    @property
    def la_observation(self) -> LaObservation:
        """Link adaptation observation for the pending slot."""

        if self._la_observation is None:
            raise RuntimeError("reset() must be called before observing")
        return self._la_observation

    @property
    def context(self) -> SlotContext:
        """Slot the simulator published for the pending decision."""

        if self._context is None:
            raise RuntimeError("reset() must be called before observing")
        return self._context

    # ------------------------------------------------------------------
    # Reference policies
    # ------------------------------------------------------------------

    def teacher_mcs(self) -> torch.Tensor:
        """Perfect-information debt-SE-aware MCS ``[env, ue]`` for this slot.

        Reads the realized SINR the simulator published, which the observation
        never contains. This is the cloning label.
        """

        return debt_se_aware_mcs(
            self._require_candidate_bler(),
            self.bler_windowed,
            self._require_scheduled(),
            self.mcs_spectral_eff,
            target_bler=self.la_config.target_bler,
            bler_cushion=self.la_config.bler_cushion,
            debt_bler_bound=self.la_config.debt_bler_bound,
        )

    def genie_greedy_mcs(self) -> torch.Tensor:
        """Highest MCS meeting the BLER target, ignoring BLER history."""

        return greedy_mcs(
            self._require_candidate_bler(),
            self._require_scheduled(),
            target_bler=self.la_config.target_bler,
        )

    def illa_predicted_sinr_db(self) -> torch.Tensor:
        """Predicted post-precoder SINR ``[env, prbg, layer]`` before the outer loop.

        The reported CQI is a single-user, full-power measurement of
        ``P_prbg * ||H||_F^2 / (N + ICI)``. Handing it to the MCS table
        unchanged overshoots badly, because the scheduler then splits that power
        across the co-scheduled layers, steers a precoder that no longer matches
        the UE's channel, and leaves intra-cell leakage behind. All three follow
        from decisions the scheduler made, so all three can be priced from the
        channel estimate and the precoder built from it.

        The prediction is per resource rather than one number per UE because the
        link is graded by EESM, which is dominated by a UE's weakest resource. A
        scalar correction on a wideband CQI cannot express that, and throws away
        most of what the per-resource terms below know.
        """

        if self._illa_sinr_cache is None:
            raise RuntimeError("reset() must be called before observing")
        return self._illa_sinr_cache

    def illa_mcs(self) -> torch.Tensor:
        """Inner-loop MCS ``[env, ue]``, with no outer-loop correction."""

        return self._mcs_from_predicted_sinr(apply_olla=False)

    def olla_mcs(self) -> torch.Tensor:
        """Inner loop plus the outer-loop offset: the deployable baseline.

        The inner loop prices what the scheduler knows it did; the outer loop
        absorbs what neither the CQI nor the estimate can see, namely channel
        aging, estimation error and the neighbour cells' scheduling.
        """

        return self._mcs_from_predicted_sinr(apply_olla=True)

    def _mcs_from_predicted_sinr(self, *, apply_olla: bool) -> torch.Tensor:
        """Highest MCS whose predicted BLER meets the target.

        Identical in form to the perfect-information selector, differing only in
        that every input is an estimate: the same EESM aggregation and the same
        BLER curves are applied to the predicted SINR instead of the realized
        one.
        """

        context = self.context
        predicted = self.illa_predicted_sinr_db()
        predicted = predicted - float(self.la_config.conservative_mcs_margin_db)
        if apply_olla and self.la_config.olla_enabled:
            predicted = predicted + self._gather_per_ue(
                self.olla_offset_db,
                context.ue_index,
            )

        per_ue_sinr_db, per_ue_mask = self._scatter_stream_resources(
            predicted,
            context.ue_index,
            context.stream_valid,
        )
        _effective, candidate_bler = candidate_bler_table(
            per_ue_sinr_db,
            per_ue_mask,
            self.phy,
        )
        return greedy_mcs(
            candidate_bler,
            self._require_scheduled(),
            target_bler=self.la_config.target_bler,
        )

    def _illa_predicted_sinr_db(self, context: SlotContext) -> torch.Tensor:
        """Per-resource SINR prediction from estimates only, in dB.

        The reported CQI sets the absolute level, since it alone reflects the
        UE's own receiver and the interference it actually saw. The channel
        estimate and the precoder set the shape across resources. Levels are
        cumulative, each adding a term the previous one approximated away.
        """

        mode = self.la_config.illa_calibration
        stream_valid = context.stream_valid
        reported_sinr_db = self.phy.cqi_to_sinr_db(context.wideband_cqi).to(
            torch.float32
        )
        reported_layer_db = self._gather_per_ue(reported_sinr_db, context.ue_index)

        if mode == "power_split":
            # Equal power split only: the precoder is treated as a matched
            # filter and intra-cell leakage as zero.
            layers = stream_valid.sum(dim=-1, keepdim=True).to(torch.float32)
            return reported_layer_db - 10.0 * torch.log10(layers.clamp_min(1.0))

        # Reference what the CQI measured, per PRB group, from the same estimate
        # it came from: full power over all receive antennas against noise plus
        # the reported interference. Taking the ratio against this cancels the
        # estimate's own bias and leaves a pure single-user-to-co-scheduled
        # factor.
        su_signal = float(
            self.carrier.power_per_prbg_watt
        ) * context.csi_channel_energy.to(torch.float32)
        su_signal_layer = self._gather_per_ue_prbg(su_signal, context.ue_index)
        reported_ici_layer = self._gather_per_ue_prbg(
            context.estimated_ici_watt.to(torch.float32),
            context.ue_index,
        )
        noise_plus_ici = (
            reported_ici_layer + float(self.carrier.noise_per_prbg_watt)
        ).clamp_min(1e-30)

        # Post-precoder desired power on the estimated channel. A
        # column-normalized precoder makes this at most the matched-filter
        # energy, so the term is a loss that already carries the power split
        # inside it.
        mu_signal = context.estimated_signal_watt.to(torch.float32)
        denominator = noise_plus_ici
        if mode in {"beamforming_and_leakage", "estimation_error_floor"}:
            denominator = denominator + (
                context.estimated_residual_interference_watt.to(torch.float32)
            )
        if mode == "estimation_error_floor":
            denominator = denominator + self._estimation_error_leakage_watt(context)

        ratio = (mu_signal * noise_plus_ici) / (
            su_signal_layer * denominator
        ).clamp_min(1e-30)
        delta_db = 10.0 * torch.log10(ratio.clamp_min(1e-12))
        return reported_layer_db + delta_db

    def _estimation_error_leakage_watt(self, context: SlotContext) -> torch.Tensor:
        """Leakage ``[env, prbg, layer]`` the precoder's nulls will miss.

        The precoder is solved from the estimate, so it cancels interference
        exactly in estimate coordinates and
        ``estimated_residual_interference_watt`` reports almost none. Against
        the realized channel the nulls miss by the estimation error, which is
        invisible to any estimate-derived calculation. Its *magnitude* is not:
        the base station knows its own sounding quality, which the simulator
        publishes as :attr:`CarrierProfile.channel_est_error_nmse_db`.

        Writing the realized row as ``c = c_est + e`` with
        ``E|e|^2 = eps ||c||^2``, the cross terms average out and each
        co-scheduled unit-norm precoder column collects ``eps/n_tx`` of the
        error energy, giving ``eps * ||c_est||^2 * p_layer * (L - 1) / n_tx``.
        A regularized precoder does not null perfectly even in estimate space,
        so that argument is optimistic by a constant, which
        ``illa_leakage_floor_gain_db`` supplies.

        ``L - 1`` makes the term vanish on a single-layer PRB group, so it
        applies only where the UE is genuinely co-scheduled.
        """

        epsilon = 10.0 ** (float(self.carrier.channel_est_error_nmse_db) / 10.0)
        gain = 10.0 ** (float(self.la_config.illa_leakage_floor_gain_db) / 10.0)
        stream_valid = context.stream_valid
        estimate_energy = (
            context.effective_channel_est.abs().square().sum(dim=-1)
        ).to(torch.float32)
        co_scheduled = (
            stream_valid.sum(dim=-1, keepdim=True).to(torch.float32) - 1.0
        ).clamp_min(0.0)
        floor = (
            gain
            * epsilon
            * estimate_energy
            * context.power_per_layer_watt.to(torch.float32)
            * co_scheduled
            / float(self.carrier.num_tx_ant)
        )
        return torch.where(stream_valid, floor, torch.zeros_like(floor))

    # ------------------------------------------------------------------
    # Slot construction
    # ------------------------------------------------------------------

    def _begin_slot(self, context: SlotContext) -> LaObservation:
        """Score the published slot and build the observation, committing nothing."""

        context.validate(self.carrier)
        per_ue_sinr_db, per_ue_resource_mask = self._scatter_stream_resources(
            context.realized_sinr_db.to(torch.float32),
            context.ue_index,
            context.stream_valid,
        )
        _effective_sinr, candidate_bler = candidate_bler_table(
            per_ue_sinr_db,
            per_ue_resource_mask,
            self.phy,
        )

        self._context = context
        self._per_ue_sinr_db = per_ue_sinr_db
        self._per_ue_resource_mask = per_ue_resource_mask
        self._candidate_bler = candidate_bler
        self._resource_count = per_ue_resource_mask.sum(dim=-1).to(torch.float32)
        self._scheduled = self._resource_count > 0.0
        self._illa_sinr_cache = self._illa_predicted_sinr_db(context)
        self._la_observation = build_la_observation(
            context,
            self.carrier,
            self.la_config,
            bler_windowed=self.bler_windowed,
            phy=self.phy,
        )
        return self._la_observation

    def _gather_per_ue(
        self,
        per_ue: torch.Tensor,
        ue_index: torch.Tensor,
    ) -> torch.Tensor:
        """Broadcast ``[env, ue]`` onto the ``[env, prbg, layer]`` resource axis."""

        index = ue_index.clamp(0, self.num_ues - 1)
        expanded = per_ue.unsqueeze(1).expand(
            self.num_envs,
            self.num_prb_groups,
            self.num_ues,
        )
        return torch.gather(expanded, 2, index)

    def _gather_per_ue_prbg(
        self,
        per_ue_prbg: torch.Tensor,
        ue_index: torch.Tensor,
    ) -> torch.Tensor:
        """Gather ``[env, ue, prbg]`` onto the ``[env, prbg, layer]`` resource axis."""

        index = ue_index.clamp(0, self.num_ues - 1)
        return torch.gather(per_ue_prbg.transpose(1, 2), 2, index)

    def _scatter_stream_resources(
        self,
        sinr_db: torch.Tensor,
        ue_index: torch.Tensor,
        stream_valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Move ``[env, prbg, layer]`` SINR onto a per-physical-UE resource axis.

        Every ``(PRBG, layer)`` slot stays on its own trailing axis, so a UE
        appearing on several independently co-scheduled PRB groups contributes
        one EESM term per appearance. That is what a single transport block
        spanning a non-contiguous allocation actually experiences.
        """

        flat_index = ue_index.reshape(self.num_envs, -1)
        valid_resource = stream_valid.reshape(self.num_envs, -1)
        flat_sinr_db = sinr_db.reshape(self.num_envs, -1)
        num_resources = flat_index.shape[-1]

        per_ue_sinr_db = torch.full(
            (self.num_envs, self.num_ues, num_resources),
            _UNSCHEDULED_SINR_DB,
            device=self.device,
            dtype=flat_sinr_db.dtype,
        )
        per_ue_resource_mask = torch.zeros(
            (self.num_envs, self.num_ues, num_resources),
            device=self.device,
            dtype=torch.bool,
        )
        scatter_index = flat_index.clamp(0, self.num_ues - 1).unsqueeze(1)
        per_ue_sinr_db.scatter_(
            1,
            scatter_index,
            torch.where(
                valid_resource,
                flat_sinr_db,
                torch.full_like(flat_sinr_db, _UNSCHEDULED_SINR_DB),
            ).unsqueeze(1),
        )
        per_ue_resource_mask.scatter_(
            1,
            scatter_index,
            valid_resource.unsqueeze(1),
        )
        return per_ue_sinr_db, per_ue_resource_mask

    # ------------------------------------------------------------------
    # Action application and accounting
    # ------------------------------------------------------------------

    def _resolve_mcs(self, action: torch.Tensor | None) -> torch.Tensor:
        """Map a policy action to an MCS index, or fall back to the outer loop."""

        if action is None:
            return self.olla_mcs().clamp(0, self.num_mcs - 1)
        if tuple(action.shape) != (self.num_envs, self.num_ues):
            raise ValueError(
                "action must have shape "
                f"{(self.num_envs, self.num_ues)}, got {tuple(action.shape)}"
            )
        index = action.long()
        if self.la_config.drl_absolute_mcs:
            return index.clamp(0, self.num_mcs - 1)
        deltas = self.action_deltas[index.clamp(0, self.action_dim - 1)]
        return (self.olla_mcs() + deltas).clamp(0, self.num_mcs - 1)

    def _step_impl(
        self,
        action: torch.Tensor | None,
        *,
        auto_reset: bool,
    ) -> tuple[LaObservation, torch.Tensor, torch.Tensor, dict[str, Any]]:
        if self._la_observation is None or self._context is None:
            raise RuntimeError("reset() must be called before step()")

        mcs = self._resolve_mcs(action)
        link = self._apply_link_adaptation(mcs)
        reward = self._reward(link)
        self._update_olla(link)
        info = self._build_la_info(link, reward)
        outcome = SlotOutcome(
            slot=self.slot,
            scheduled=link["scheduled_mask"],
            mcs=link["mcs"],
            expected_bler=link["expected_bler"],
            ack=link["ack"],
            resource_count=link["resource_count"],
            raw_rate_bps=link["raw_rate_bps"],
            realized_rate_bps=link["realized_rate_bps"],
        )

        self.slot += 1
        episode_length = self.carrier.episode_length_slots
        episode_done = episode_length is not None and self.slot >= int(episode_length)
        done = torch.full(
            (self.num_envs,),
            bool(episode_done),
            device=self.device,
            dtype=torch.bool,
        )
        if auto_reset and episode_done:
            observation = self.reset()
        else:
            observation = self._begin_slot(self.simulator.advance(outcome))
        return observation, reward, done, info

    def _apply_link_adaptation(self, mcs: torch.Tensor) -> dict[str, torch.Tensor]:
        """Score the chosen MCS, draw one ACK per UE and roll the BLER window."""

        candidate_bler = self._require_candidate_bler()
        scheduled = self._require_scheduled()
        resource_count = self._resource_count
        assert resource_count is not None

        # A retransmission carries the modulation its first attempt chose, so
        # the policy has no action on it: its action is replaced by the replayed
        # MCS here, and the row is kept out of the reward and the learning
        # signal below.
        context = self._require_context()
        selected_mcs = mcs.clamp(0, self.num_mcs - 1)
        if context.retransmitting is None:
            retransmitting = torch.zeros_like(scheduled)
        else:
            retransmitting = context.retransmitting & scheduled
            if context.retransmission_mcs is not None:
                selected_mcs = torch.where(
                    retransmitting,
                    context.retransmission_mcs.long().clamp(0, self.num_mcs - 1),
                    selected_mcs,
                )
        selected_bler = candidate_bler.gather(
            -1,
            selected_mcs.unsqueeze(-1),
        ).squeeze(-1)
        selected_bler = torch.where(
            scheduled,
            selected_bler,
            torch.zeros_like(selected_bler),
        )

        spectral_eff = self.mcs_spectral_eff[selected_mcs]
        # Full buffer: the granted rate is the rate the MCS and the resource
        # count imply, with no queue to run short.
        raw_rate = torch.where(
            scheduled,
            spectral_eff
            * resource_count
            * float(self.carrier.prbg_bandwidth_hz),
            torch.zeros_like(selected_bler),
        )
        expected_rate = raw_rate * (1.0 - selected_bler)

        ack = (
            torch.rand(
                selected_bler.shape,
                device=self.device,
                generator=self.generator,
            )
            >= selected_bler
        ) & scheduled
        realized_rate = raw_rate * ack.to(raw_rate.dtype)

        bler_before = self.bler_windowed
        nack = (~ack).to(bler_before.dtype)
        alpha = float(self.la_config.bler_ema_alpha)
        updated = (1.0 - alpha) * bler_before + alpha * nack
        self.bler_windowed = torch.where(
            scheduled,
            updated,
            bler_before,
        ).detach()

        return {
            "scheduled_mask": scheduled,
            "retransmitting": retransmitting,
            "resource_count": resource_count,
            "mcs": torch.where(
                scheduled,
                selected_mcs,
                torch.full_like(selected_mcs, -1),
            ),
            "spectral_eff": spectral_eff,
            "expected_bler": selected_bler,
            "ack": ack,
            "nack": (~ack) & scheduled,
            "bler_before": bler_before.clone(),
            "bler_after": self.bler_windowed.clone(),
            "raw_rate_bps": raw_rate,
            "expected_rate_bps": expected_rate,
            "realized_rate_bps": realized_rate,
        }

    def _reward(self, link: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """Per-layer expected goodput, optionally shaped by BLER tracking.

        Normalizing by neither the rank nor the resource count is deliberate:
        the link adaptation policy chooses only the modulation, so paying it for
        resources the scheduler happened to grant would reward it for another
        component's decisions.
        """

        # A retransmission replays the modulation its first attempt chose, so
        # the policy took no action on it and must be neither paid nor charged
        # for the outcome.
        acted = link["scheduled_mask"] & ~link["retransmitting"]
        scheduled = acted.to(torch.float32)
        goodput = link["spectral_eff"] * (1.0 - link["expected_bler"])
        reward = float(self.la_config.goodput_reward_weight) * goodput

        weight = float(self.la_config.bler_adjust_award)
        if weight > 0.0:
            target = float(self.la_config.target_bler)
            before = (link["bler_before"] - target).abs()
            after = (link["bler_after"] - target).abs()
            reward = reward + weight * (before - after)
        return reward * scheduled

    def _update_olla(self, link: Mapping[str, torch.Tensor]) -> None:
        """Step the outer-loop offset on the realized ACK/NACK."""

        if not self.la_config.olla_enabled:
            return
        step = torch.where(
            link["ack"],
            torch.full_like(
                self.olla_offset_db,
                float(self.la_config.olla_ack_step_db),
            ),
            torch.full_like(
                self.olla_offset_db,
                -float(self.la_config.olla_nack_step_db),
            ),
        )
        updated = (self.olla_offset_db + step).clamp(
            float(self.la_config.olla_min_offset_db),
            float(self.la_config.olla_max_offset_db),
        )
        self.olla_offset_db = torch.where(
            link["scheduled_mask"],
            updated,
            self.olla_offset_db,
        ).detach()

    def _build_la_info(
        self,
        link: Mapping[str, torch.Tensor],
        reward: torch.Tensor,
    ) -> dict[str, Any]:
        """Per-slot KPIs, averaged over the UEs the scheduler admitted."""

        scheduled = link["scheduled_mask"]
        scheduled_count = scheduled.sum(dim=-1).clamp_min(1)
        scheduled_float = scheduled.to(torch.float32)
        expected_cell_rate = link["expected_rate_bps"].sum(dim=-1)
        realized_cell_rate = link["realized_rate_bps"].sum(dim=-1)
        bandwidth_hz = float(self.carrier.bandwidth_hz)
        return {
            "scheduled_mask": scheduled,
            "scheduled_ues": scheduled.sum(dim=-1).to(torch.float32),
            "mcs": link["mcs"],
            "mean_mcs": (
                link["mcs"].clamp_min(0).to(torch.float32) * scheduled_float
            ).sum(dim=-1)
            / scheduled_count,
            "expected_bler": link["expected_bler"],
            "mean_expected_bler": (
                link["expected_bler"] * scheduled_float
            ).sum(dim=-1)
            / scheduled_count,
            "nack_rate": (
                link["nack"].to(torch.float32) * scheduled_float
            ).sum(dim=-1)
            / scheduled_count,
            "bler_windowed": link["bler_after"],
            "mean_bler_windowed": (
                link["bler_after"] * scheduled_float
            ).sum(dim=-1)
            / scheduled_count,
            "expected_cell_rate_bps": expected_cell_rate,
            "realized_cell_rate_bps": realized_cell_rate,
            "expected_cell_spectral_eff": expected_cell_rate / bandwidth_hz,
            "realized_cell_spectral_eff": realized_cell_rate / bandwidth_hz,
            "expected_rate_bps_per_ue": link["expected_rate_bps"],
            "realized_rate_bps_per_ue": link["realized_rate_bps"],
            "resource_count_per_ue": link["resource_count"],
            "mean_reward": reward.sum(dim=-1) / scheduled_count,
            "scheduled_resources": link["resource_count"].sum(dim=-1),
        }

    # ------------------------------------------------------------------
    # Cached slot state
    # ------------------------------------------------------------------

    def _require_candidate_bler(self) -> torch.Tensor:
        if self._candidate_bler is None:
            raise RuntimeError("reset() must be called before observing")
        return self._candidate_bler

    def _require_scheduled(self) -> torch.Tensor:
        if self._scheduled is None:
            raise RuntimeError("reset() must be called before observing")
        return self._scheduled

    def _require_context(self) -> SlotContext:
        if self._context is None:
            raise RuntimeError("reset() must be called before observing")
        return self._context

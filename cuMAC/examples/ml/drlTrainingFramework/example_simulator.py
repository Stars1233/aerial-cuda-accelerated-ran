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

"""A minimal :class:`LinkAdaptationSimulator` for wiring up the integration.

**This is a plumbing test double, not a channel model.** It is here so the
framework can be imported, trained and evaluated end to end before a host
simulator is connected, and so an adapter author has one concrete, readable
example of every field in :class:`~drlTrainingFramework.adapter.SlotContext`.
Its statistics are made up: an AR(1) SINR process, a random co-scheduling
draw and closed-form steering and leakage penalties. Nothing about its absolute
throughput or BLER means anything.

What it *does* reproduce faithfully is the structure the framework depends on,
which is what makes it useful as a reference:

* The reported CQI is a single-user, full-power measurement, and
  :attr:`SlotContext.csi_channel_energy` is exactly the channel energy behind
  it, so the inner loop's single-user reference cancels correctly.
* The realized SINR is systematically *worse* than the estimate-derived
  prediction, by channel aging and by the interference the precoder's nulls miss
  because they were solved from the estimate. That gap is what the outer loop
  exists to absorb and what makes the perfect-information teacher worth cloning.
* CSI, sounding and rank reports are staggered across UEs and refreshed on a
  period, so the feedback-age features carry a real signal.
* Traffic is full buffer: every UE is always backlogged and eligible.

Replace it with an adapter over the host simulator. Keep
:meth:`ExampleSimulator.reset` honouring ``seed``, or the matched evaluation in
:mod:`~drlTrainingFramework.training` degenerates into noise.
"""

from __future__ import annotations

__all__ = [
    "ExampleSimulator",
    "ExampleSimulatorConfig",
    "build_simulator",
]

import math
from dataclasses import dataclass

import torch

from .adapter import CarrierProfile, SlotContext, SlotOutcome
from .phy import DEFAULT_PHY, PhyAbstraction
from .yaml_config import flatten_sections, load_document


@dataclass
class ExampleSimulatorConfig:
    """Shape and synthetic-statistics knobs for :class:`ExampleSimulator`."""

    num_envs: int = 4
    num_ues: int = 16
    num_prb_groups: int = 17
    max_coscheduled_layers: int = 8
    max_layers_per_ue: int = 2
    num_tx_ant: int = 64
    prbg_bandwidth_hz: float = 16 * 12 * 30.0e3
    #: Receiver noise power in one PRB group, in dBW, and the total transmit
    #: power the carrier splits across its PRB groups.
    noise_power_dbw: float = -95.0
    transmit_power_dbm: float = 46.0
    #: Mean and spread of the per-UE single-user SNR, i.e. the drop geometry.
    mean_su_snr_db: float = 14.0
    su_snr_spread_db: float = 8.0
    #: Frequency-selective and time-varying components of the fading.
    freq_selectivity_db: float = 3.0
    time_fading_db: float = 3.0
    #: AR(1) coefficient of the time-varying component; closer to 1 is slower.
    time_correlation: float = 0.95
    #: Inter-cell interference, as an interference-to-noise ratio.
    ici_inr_db: float = 3.0
    #: Precoder steering loss per co-scheduled layer, and the intra-cell
    #: leakage the precoder predicts it fails to null.
    steering_loss_per_layer: float = 0.12
    predicted_leakage_fraction: float = 0.004
    #: Channel estimate quality, published to the framework so the inner loop
    #: can price the nulls it will miss.
    channel_est_error_nmse_db: float = -15.0
    #: How far the realized missed-null leakage sits above the unit-gain
    #: first-principles floor, in dB. This is the modelling gap
    #: ``illa_leakage_floor_gain_db`` is meant to close.
    realized_leakage_excess_db: float = 12.0
    #: Signal lost to channel aging per slot of sounding age.
    aging_loss_db_per_slot: float = 0.15
    csi_report_period_slots: int = 5
    episode_length_slots: int | None = None
    device: str = "cpu"
    seed: int = 1234


class ExampleSimulator:
    """Synthetic full-buffer simulator implementing the adapter contract."""

    def __init__(self, config: ExampleSimulatorConfig | None = None) -> None:
        self.config = config or ExampleSimulatorConfig()
        device = torch.device(self.config.device)
        self._device = device
        self._generator = torch.Generator(device=device)
        self._generator.manual_seed(int(self.config.seed))

        noise_watt = 10.0 ** (float(self.config.noise_power_dbw) / 10.0)
        power_watt = 10.0 ** (
            (float(self.config.transmit_power_dbm) - 30.0) / 10.0
        ) / float(self.config.num_prb_groups)
        self._carrier = CarrierProfile(
            num_envs=int(self.config.num_envs),
            num_ues=int(self.config.num_ues),
            num_prb_groups=int(self.config.num_prb_groups),
            max_coscheduled_layers=int(self.config.max_coscheduled_layers),
            max_layers_per_ue=int(self.config.max_layers_per_ue),
            num_tx_ant=int(self.config.num_tx_ant),
            prbg_bandwidth_hz=float(self.config.prbg_bandwidth_hz),
            bandwidth_hz=float(self.config.prbg_bandwidth_hz)
            * float(self.config.num_prb_groups),
            power_per_prbg_watt=power_watt,
            noise_per_prbg_watt=noise_watt,
            device=device,
            csi_report_period_slots=int(self.config.csi_report_period_slots),
            channel_est_error_nmse_db=float(self.config.channel_est_error_nmse_db),
            episode_length_slots=self.config.episode_length_slots,
        )
        self._phy: PhyAbstraction = DEFAULT_PHY
        self._cqi_table = self._phy.cqi_to_sinr_db(
            torch.arange(16, device=device)
        ).contiguous()
        self._slot = 0
        self._draw_deployment()
        self._refresh_reports(force=True)

    # ------------------------------------------------------------------
    # Adapter contract
    # ------------------------------------------------------------------

    @property
    def carrier(self) -> CarrierProfile:
        return self._carrier

    @property
    def phy(self) -> PhyAbstraction:
        return self._phy

    def reset(self, *, seed: int | None = None) -> SlotContext:
        """Redraw the deployment and publish slot zero."""

        if seed is not None:
            self._generator.manual_seed(int(seed))
        self._slot = 0
        self._draw_deployment()
        self._refresh_reports(force=True)
        return self._publish()

    def advance(self, outcome: SlotOutcome) -> SlotContext:
        """Age the channel and the reports by one slot, then publish.

        A production adapter also updates its proportional-fair weights from
        ``outcome.realized_rate_bps`` here. This test double schedules at random
        and so ignores the outcome entirely.
        """

        del outcome
        self._slot += 1
        rho = float(self.config.time_correlation)
        self._time_db = rho * self._time_db + math.sqrt(
            max(1.0 - rho * rho, 0.0)
        ) * self._normal(self._time_db.shape, float(self.config.time_fading_db))
        self._refresh_reports()
        return self._publish()

    # ------------------------------------------------------------------
    # Synthetic radio state
    # ------------------------------------------------------------------

    def _normal(self, shape: tuple[int, ...], std: float) -> torch.Tensor:
        return (
            torch.randn(shape, device=self._device, generator=self._generator) * std
        )

    def _draw_deployment(self) -> None:
        """Fresh per-UE geometry, frequency selectivity and fading state."""

        carrier = self._carrier
        shape = (carrier.num_envs, carrier.num_ues, carrier.num_prb_groups)
        self._mean_snr_db = float(self.config.mean_su_snr_db) + self._normal(
            (carrier.num_envs, carrier.num_ues, 1),
            float(self.config.su_snr_spread_db),
        )
        self._freq_db = self._normal(shape, float(self.config.freq_selectivity_db))
        self._time_db = self._normal(shape, float(self.config.time_fading_db))
        self._report_slot = torch.zeros(
            (carrier.num_envs, carrier.num_ues),
            device=self._device,
            dtype=torch.long,
        )
        self._reported_cqi = torch.zeros_like(self._report_slot)
        self._reported_rank = torch.ones_like(self._report_slot)

    def _su_snr_db(self) -> torch.Tensor:
        """Single-user, full-power SNR ``[env, ue, prbg]``, the CQI's referent."""

        return self._mean_snr_db + self._freq_db + self._time_db

    def _refresh_reports(self, *, force: bool = False) -> None:
        """Re-measure CQI and rank for the UEs whose reporting period expired.

        Reports are staggered across the UE index, so at any slot the feedback
        ages span a whole reporting period rather than all being equal.
        """

        carrier = self._carrier
        period = max(int(carrier.csi_report_period_slots), 1)
        ue = torch.arange(carrier.num_ues, device=self._device)
        due = ((self._slot - ue) % period == 0).unsqueeze(0).expand(
            carrier.num_envs,
            carrier.num_ues,
        )
        if force:
            due = torch.ones_like(due)

        wideband_db = 10.0 * torch.log10(
            torch.pow(10.0, self._su_snr_db() / 10.0).mean(dim=-1).clamp_min(1e-12)
        )
        cqi = (
            torch.searchsorted(self._cqi_table, wideband_db.contiguous(), right=True)
            - 1
        ).clamp(0, self._cqi_table.numel() - 1)
        rank = (
            1 + ((wideband_db - 5.0) / 6.0).floor().long()
        ).clamp(1, carrier.max_layers_per_ue)

        self._reported_cqi = torch.where(due, cqi, self._reported_cqi)
        self._reported_rank = torch.where(due, rank, self._reported_rank)
        self._report_slot = torch.where(
            due,
            torch.full_like(self._report_slot, self._slot),
            self._report_slot,
        )

    def _schedule(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Random full-buffer co-scheduling, ``(ue_index, stream_valid)``.

        Every UE is backlogged, so eligibility is not modelled: a random
        priority order fills the layer slots of each PRB group, each selected UE
        taking its reported rank in layers, until the PRB group is full. A host
        simulator replaces this with its own scheduler and MU-MIMO pairing.
        """

        carrier = self._carrier
        envs, ues = carrier.num_envs, carrier.num_ues
        prbgs, layers = carrier.num_prb_groups, carrier.max_coscheduled_layers

        priority = torch.rand(
            (envs, prbgs, ues),
            device=self._device,
            generator=self._generator,
        )
        order = priority.argsort(dim=-1)
        rank = self._reported_rank.unsqueeze(1).expand(envs, prbgs, ues)
        ranks_ordered = torch.gather(rank, 2, order)
        ends = ranks_ordered.cumsum(dim=2)
        starts = ends - ranks_ordered

        slot = torch.arange(layers, device=self._device).view(1, 1, 1, layers)
        occupies = (slot >= starts.unsqueeze(-1)) & (slot < ends.unsqueeze(-1))
        ue_index = (order.unsqueeze(-1) * occupies).sum(dim=2)
        stream_valid = occupies.any(dim=2)
        return ue_index, stream_valid.contiguous()

    def _publish(self) -> SlotContext:
        """Assemble the slot the framework will decide on."""

        carrier = self._carrier
        envs, ues = carrier.num_envs, carrier.num_ues
        prbgs, layers = carrier.num_prb_groups, carrier.max_coscheduled_layers
        noise = float(carrier.noise_per_prbg_watt)
        ici = noise * 10.0 ** (float(self.config.ici_inr_db) / 10.0)

        ue_index, stream_valid = self._schedule()
        # The channel energy behind the reported CQI: publishing exactly this
        # is what lets the inner loop divide out the single-user reference.
        su_snr_linear = torch.pow(10.0, self._su_snr_db() / 10.0)
        csi_channel_energy = (
            su_snr_linear * (noise + ici) / float(carrier.power_per_prbg_watt)
        )
        estimated_ici_watt = torch.full_like(csi_channel_energy, ici)

        active_layers = stream_valid.sum(dim=-1, keepdim=True).clamp_min(1)
        power_per_layer = float(carrier.power_per_prbg_watt) / active_layers.to(
            torch.float32
        )
        power_per_layer = power_per_layer.expand(envs, prbgs, layers)

        index = ue_index.clamp(0, ues - 1)
        energy_layer = torch.gather(csi_channel_energy.transpose(1, 2), 2, index)
        rank_layer = torch.gather(
            self._reported_rank.unsqueeze(1)
            .expand(envs, prbgs, ues)
            .to(torch.float32),
            2,
            index,
        ).clamp_min(1.0)
        # A UE's channel energy is shared across the layers it was given.
        energy_layer = energy_layer / rank_layer
        ici_layer = torch.gather(estimated_ici_watt.transpose(1, 2), 2, index)

        co_scheduled = (active_layers.to(torch.float32) - 1.0).clamp_min(0.0)
        steering = 1.0 / (
            1.0 + float(self.config.steering_loss_per_layer) * co_scheduled
        )
        estimated_signal = power_per_layer * energy_layer * steering
        estimated_resid = (
            power_per_layer
            * energy_layer
            * float(self.config.predicted_leakage_fraction)
            * co_scheduled
        )

        # The realized link is worse than the estimate predicts, in the two ways
        # an estimate structurally cannot see: the channel moved since it was
        # sounded, and the nulls were solved in estimate coordinates.
        srs_age = (self._slot - self._report_slot).to(torch.float32)
        age_layer = torch.gather(
            srs_age.unsqueeze(1).expand(envs, prbgs, ues),
            2,
            index,
        )
        aging = torch.pow(
            10.0,
            -float(self.config.aging_loss_db_per_slot) * age_layer / 10.0,
        )
        epsilon = 10.0 ** (float(carrier.channel_est_error_nmse_db) / 10.0)
        excess = 10.0 ** (float(self.config.realized_leakage_excess_db) / 10.0)
        missed_nulls = (
            excess
            * epsilon
            * energy_layer
            * power_per_layer
            * co_scheduled
            / float(carrier.num_tx_ant)
        )
        realized_sinr = (estimated_signal * aging) / (
            noise + ici_layer + estimated_resid + missed_nulls
        ).clamp_min(1e-30)
        realized_sinr_db = 10.0 * torch.log10(realized_sinr.clamp_min(1e-12))

        # Random unit directions scaled to the per-layer channel energy. Only
        # the directions matter to the correlation features.
        direction = torch.randn(
            (envs, prbgs, layers, carrier.num_tx_ant),
            device=self._device,
            generator=self._generator,
            dtype=torch.complex64,
        )
        direction = direction / direction.abs().square().sum(
            dim=-1,
            keepdim=True,
        ).clamp_min(1e-20).sqrt().to(direction.dtype)
        effective_channel_est = direction * energy_layer.sqrt().unsqueeze(-1).to(
            direction.dtype
        )

        zero_out = ~stream_valid
        return SlotContext(
            slot=self._slot,
            ue_index=ue_index,
            stream_valid=stream_valid,
            estimated_signal_watt=estimated_signal.masked_fill(zero_out, 0.0),
            estimated_residual_interference_watt=estimated_resid.masked_fill(
                zero_out,
                0.0,
            ),
            power_per_layer_watt=power_per_layer.masked_fill(zero_out, 0.0),
            effective_channel_est=effective_channel_est,
            realized_sinr_db=realized_sinr_db,
            wideband_cqi=self._reported_cqi,
            reported_rank=self._reported_rank,
            csi_age_slots=srs_age,
            srs_age_slots=srs_age,
            rank_age_slots=srs_age,
            csi_channel_energy=csi_channel_energy,
            estimated_ici_watt=estimated_ici_watt,
            retransmitting=None,
        )


def build_simulator(
    config_path: str | None = None,
    *,
    seed: int | None = None,
    device: str | None = None,
    **overrides: object,
) -> ExampleSimulator:
    """Factory matching the launchers' ``--simulator module:function`` contract.

    ``config_path`` is a YAML file whose leaf keys name
    :class:`ExampleSimulatorConfig` fields; a host adapter is free to interpret
    it however its own simulator expects.
    """

    values: dict[str, object] = {}
    if config_path is not None:
        values.update(
            flatten_sections(load_document(config_path), ExampleSimulatorConfig)
        )
    values.update(overrides)
    if seed is not None:
        values["seed"] = int(seed)
    if device is not None:
        values["device"] = str(device)
    return ExampleSimulator(ExampleSimulatorConfig(**values))  # type: ignore[arg-type]

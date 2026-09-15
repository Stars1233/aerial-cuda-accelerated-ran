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

"""The interface a system-level simulator implements to train DRL link adaptation.

The split of responsibility is deliberate and narrow:

* **The host simulator owns the radio and the scheduler.** Deployment, channel,
  CSI feedback, resource allocation, MU-MIMO co-scheduling and precoding are all
  decided outside this package, and none of them are decisions the link
  adaptation policy makes. Each slot the simulator publishes the *result* of
  those decisions as a :class:`SlotContext`.
* **This package owns the MCS decision and everything downstream of it.**
  Observation construction, the inner and outer link adaptation loops, the
  EESM/BLER scoring of a candidate MCS, the teacher labels, the reward, the
  sliding BLER window and the training curriculum all live here.

So the integration work is: fill in a :class:`SlotContext` once per slot, and
consume a :class:`SlotOutcome` to advance your own state. Implement
:class:`LinkAdaptationSimulator` and the rest of the framework runs unchanged.

Axis conventions, used by every tensor below:

``env``
    Independent parallel simulation instances, batched for throughput. Use 1 if
    the host simulator is not vectorized.
``prbg``
    PRB groups across the carrier, the granularity at which resources are
    allocated and precoded.
``layer``
    Spatial layer slot within one PRB group, of width
    :attr:`CarrierProfile.max_coscheduled_layers`. A layer slot is owned by at
    most one UE, and one UE may own several of them.
``ue``
    Physical UE index, stable across PRB groups and across slots. The action
    axis is this index, so a UE keeps its policy row for its whole life.

Traffic model: the framework assumes **full buffer**, i.e. every scheduled UE
always has data to send, so the granted rate of a transport block is the rate
its MCS and resource count imply. No queue, no buffer occupancy and no
admission decision enter the observation or the reward.
"""

from __future__ import annotations

__all__ = [
    "DEFAULT_PHY",
    "CarrierProfile",
    "LinkAdaptationSimulator",
    "SimulatorFactory",
    "SlotContext",
    "SlotOutcome",
]

from dataclasses import dataclass, replace
from typing import Callable, Protocol, runtime_checkable

import torch

from .phy import DEFAULT_PHY, PhyAbstraction


@dataclass(frozen=True)
class CarrierProfile:
    """Static description of the carrier and the batch, published once.

    Everything here is fixed for the lifetime of a simulator instance; anything
    that changes per slot belongs in :class:`SlotContext`.

    Attributes:
        num_envs: Parallel simulation instances.
        num_ues: Physical UE slots, i.e. the width of the action axis.
        num_prb_groups: PRB groups across the carrier.
        max_coscheduled_layers: Width of the layer axis, i.e. the largest number
            of spatial layers the simulator will ever co-schedule on one PRB
            group. Also the normalizer for the co-scheduled-layer feature.
        max_layers_per_ue: Largest rank a single UE can be given.
        num_tx_ant: Transmit antennas, i.e. the width of
            :attr:`SlotContext.effective_channel_est`.
        prbg_bandwidth_hz: Bandwidth of one PRB group. A transport block's rate
            is ``spectral_eff * resource_count * prbg_bandwidth_hz``.
        bandwidth_hz: Carrier bandwidth, the denominator of cell spectral
            efficiency.
        power_per_prbg_watt: Transmit power allocated to one PRB group, before
            the split across co-scheduled layers.
        noise_per_prbg_watt: Receiver noise power in one PRB group.
        device: Device every published tensor lives on. The policy is placed
            here too, so keep the simulator and the training on one device.
        csi_report_period_slots: Slots between CSI reports. Feedback ages are
            normalized by this, so a normalized age of 1.0 means "one report
            old". Set to 1 if CSI is refreshed every slot.
        channel_est_error_nmse_db: Normalized mean-square error of the channel
            estimate the precoder was solved from, in dB. This is the
            simulator's own SRS/CSI quality figure, and the inner loop uses it
            to price the interference the precoder's nulls will miss. Less
            negative means a worse estimate.
        episode_length_slots: Slots before the framework asks for a reset.
            ``None`` never terminates, which is the natural choice for full
            buffer with a bandit discount.
    """

    num_envs: int
    num_ues: int
    num_prb_groups: int
    max_coscheduled_layers: int
    max_layers_per_ue: int
    num_tx_ant: int
    prbg_bandwidth_hz: float
    bandwidth_hz: float
    power_per_prbg_watt: float
    noise_per_prbg_watt: float
    device: torch.device = torch.device("cpu")
    csi_report_period_slots: int = 1
    channel_est_error_nmse_db: float = -15.0
    episode_length_slots: int | None = None

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> "CarrierProfile":
        """Check the sizes and powers a published slot will be measured against."""

        for name in (
            "num_envs",
            "num_ues",
            "num_prb_groups",
            "max_coscheduled_layers",
            "max_layers_per_ue",
            "num_tx_ant",
            "csi_report_period_slots",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"CarrierProfile.{name} must be a positive integer")
        if self.max_layers_per_ue > self.max_coscheduled_layers:
            raise ValueError(
                "max_layers_per_ue cannot exceed max_coscheduled_layers"
            )
        for name in (
            "prbg_bandwidth_hz",
            "bandwidth_hz",
            "power_per_prbg_watt",
            "noise_per_prbg_watt",
        ):
            value = float(getattr(self, name))
            if not value > 0.0:
                raise ValueError(f"CarrierProfile.{name} must be positive")
        if self.episode_length_slots is not None and self.episode_length_slots <= 0:
            raise ValueError(
                "episode_length_slots must be positive when set, or None to run "
                "without episode boundaries"
            )
        return self

    @property
    def torch_device(self) -> torch.device:
        """:attr:`device` as a :class:`torch.device`."""

        return torch.device(self.device)

    def with_device(self, device: str | torch.device) -> "CarrierProfile":
        """Copy of the profile targeting ``device``."""

        return replace(self, device=torch.device(device))


@dataclass(frozen=True)
class SlotContext:
    """Everything the simulator publishes for one slot, before the MCS is chosen.

    The framework never inspects the simulator itself; this is the whole input
    surface. All tensors live on :attr:`CarrierProfile.device`.

    Per-resource tensors are shaped ``[env, prbg, layer]`` and are only read
    where :attr:`stream_valid` is true. Fill invalid entries with anything.

    Attributes:
        slot: Monotonically increasing slot counter, reset to 0 by ``reset()``.
        ue_index: ``[env, prbg, layer]`` int64. Physical UE owning that layer
            slot. Ignored where ``stream_valid`` is false.
        stream_valid: ``[env, prbg, layer]`` bool. True where a layer carries a
            scheduled stream. This is the scheduling and MU-MIMO decision, taken
            by the host simulator; the framework treats it as given.
        estimated_signal_watt: ``[env, prbg, layer]``. Post-precoder desired
            signal power the scheduler *predicts* for that layer, computed from
            the channel estimate the precoder was solved from. This is what the
            policy is allowed to see.
        estimated_residual_interference_watt: ``[env, prbg, layer]``.
            Intra-cell interference the precoder predicts it fails to null, from
            the same estimate. Exactly zero on a single-layer PRB group.
        power_per_layer_watt: ``[env, prbg, layer]``. Transmit power the
            precoder assigned to that layer, after the split across co-scheduled
            layers.
        effective_channel_est: ``[env, prbg, layer, num_tx_ant]`` complex.
            Estimated effective channel row of that layer, i.e. the receive
            combiner applied to the estimated channel matrix. Feeds the spatial
            correlation features; only its direction is used, so any consistent
            scaling works.
        realized_sinr_db: ``[env, prbg, layer]``. Post-precoder SINR the layer
            *actually* experiences, against the realized channel and the true
            interference. **Teacher label only.** It never enters an
            observation, which is exactly what makes teacher cloning an
            imitation problem rather than a lookup.
        wideband_cqi: ``[env, ue]`` integer CQI index the UE last reported, on
            the CQI table the PHY abstraction converts from. Single-user,
            full-power, so the inner loop has to correct it for co-scheduling.
        reported_rank: ``[env, ue]`` rank the UE last reported, in layers.
        csi_age_slots: ``[env, ue]`` slots since the CQI report was refreshed.
        srs_age_slots: ``[env, ue]`` slots since the sounding the precoder was
            solved from. Equal to ``csi_age_slots`` when the two are refreshed
            together.
        rank_age_slots: ``[env, ue]`` slots since the rank report was refreshed.
        csi_channel_energy: ``[env, ue, prbg]``. Squared Frobenius norm of the
            estimated channel matrix, ``||H_est||_F^2``, for every candidate UE
            on every PRB group. This is the single-user reference the inner loop
            divides out to isolate the co-scheduling penalty, so it must come
            from the same estimate the reported CQI was measured on.
        estimated_ici_watt: ``[env, ue, prbg]``. Inter-cell interference power
            the UE reported for that PRB group. Set to zero for a single-cell
            study.
        retransmitting: ``[env, ue]`` bool, optional. True where the transport
            block is a retransmission replaying the modulation its first attempt
            chose. The policy's action is discarded on those rows: they are
            scored with :attr:`retransmission_mcs`, earn no reward, and are
            excluded from every loss, because the policy made no decision there
            and training on them would fit the model to labels it did not
            produce. Leave as ``None`` for a pure new-transmission study, which
            is the default full-buffer setup.
        retransmission_mcs: ``[env, ue]`` int64, required wherever
            :attr:`retransmitting` is true. The MCS the first attempt chose, so
            the framework's BLER, ACK, sliding BLER window and reported outcome
            all describe the modulation actually on the air. Note that the
            framework does not model HARQ soft-combining gain: it applies the
            PHY abstraction to :attr:`realized_sinr_db` as published, so an
            adapter modelling combining should publish the post-combining SINR
            on those layers.
    """

    slot: int
    ue_index: torch.Tensor
    stream_valid: torch.Tensor
    estimated_signal_watt: torch.Tensor
    estimated_residual_interference_watt: torch.Tensor
    power_per_layer_watt: torch.Tensor
    effective_channel_est: torch.Tensor
    realized_sinr_db: torch.Tensor
    wideband_cqi: torch.Tensor
    reported_rank: torch.Tensor
    csi_age_slots: torch.Tensor
    srs_age_slots: torch.Tensor
    rank_age_slots: torch.Tensor
    csi_channel_energy: torch.Tensor
    estimated_ici_watt: torch.Tensor
    retransmitting: torch.Tensor | None = None
    retransmission_mcs: torch.Tensor | None = None

    #: ``[env, prbg, layer]`` fields, validated as a group.
    RESOURCE_FIELDS = (
        "ue_index",
        "stream_valid",
        "estimated_signal_watt",
        "estimated_residual_interference_watt",
        "power_per_layer_watt",
        "realized_sinr_db",
    )

    #: ``[env, ue]`` fields, validated as a group.
    UE_FIELDS = (
        "wideband_cqi",
        "reported_rank",
        "csi_age_slots",
        "srs_age_slots",
        "rank_age_slots",
    )

    #: ``[env, ue, prbg]`` fields, validated as a group.
    UE_PRBG_FIELDS = (
        "csi_channel_energy",
        "estimated_ici_watt",
    )

    def validate(self, carrier: CarrierProfile) -> "SlotContext":
        """Check every published tensor against ``carrier``.

        Worth calling from an adapter's own tests: a shape or device mismatch
        surfaces here with the offending field named, instead of thousands of
        slots later inside a scatter or a matrix product.
        """

        device = carrier.torch_device
        resource_shape = (
            carrier.num_envs,
            carrier.num_prb_groups,
            carrier.max_coscheduled_layers,
        )
        ue_shape = (carrier.num_envs, carrier.num_ues)
        ue_prbg_shape = (carrier.num_envs, carrier.num_ues, carrier.num_prb_groups)

        groups = (
            (self.RESOURCE_FIELDS, resource_shape),
            (self.UE_FIELDS, ue_shape),
            (self.UE_PRBG_FIELDS, ue_prbg_shape),
        )
        for names, shape in groups:
            for name in names:
                _check_tensor(getattr(self, name), name, shape, device)
        _check_tensor(
            self.effective_channel_est,
            "effective_channel_est",
            (*resource_shape, carrier.num_tx_ant),
            device,
        )
        if self.retransmitting is not None:
            _check_tensor(self.retransmitting, "retransmitting", ue_shape, device)
            if self.retransmitting.dtype is not torch.bool:
                raise ValueError("SlotContext.retransmitting must be a bool tensor")
            if self.retransmission_mcs is None and bool(self.retransmitting.any()):
                raise ValueError(
                    "SlotContext.retransmission_mcs is required wherever "
                    "retransmitting is true: the framework scores those rows "
                    "with the modulation the first attempt chose, not with the "
                    "action the policy emitted"
                )
        if self.retransmission_mcs is not None:
            _check_tensor(
                self.retransmission_mcs,
                "retransmission_mcs",
                ue_shape,
                device,
            )

        if self.stream_valid.dtype is not torch.bool:
            raise ValueError("SlotContext.stream_valid must be a bool tensor")
        if self.ue_index.dtype not in (torch.int32, torch.int64):
            raise ValueError(
                "SlotContext.ue_index must be an integer tensor; it indexes the "
                f"action axis, got dtype {self.ue_index.dtype}"
            )
        if not self.effective_channel_est.is_complex():
            raise ValueError(
                "SlotContext.effective_channel_est must be a complex tensor"
            )
        valid_index = self.ue_index[self.stream_valid]
        if valid_index.numel() and (
            int(valid_index.min()) < 0 or int(valid_index.max()) >= carrier.num_ues
        ):
            raise ValueError(
                "SlotContext.ue_index must be in [0, num_ues) wherever "
                "stream_valid is true"
            )
        return self


@dataclass(frozen=True)
class SlotOutcome:
    """What the framework hands back after the MCS action has been applied.

    A simulator uses this to advance the state the framework does not own:
    proportional-fair weights, HARQ, per-UE throughput accounting and its own
    KPI collection.

    All tensors are ``[env, ue]`` on :attr:`CarrierProfile.device`.

    Attributes:
        slot: Slot the outcome belongs to, matching the context it came from.
        scheduled: True where the UE held at least one resource.
        mcs: Applied MCS index, ``-1`` where the UE was not scheduled.
        expected_bler: Block error rate predicted for the applied MCS at the
            UE's realized effective SINR.
        ack: Sampled acknowledgement. The framework draws it from
            ``expected_bler`` with its own generator so a rollout is
            reproducible from the training seed alone.
        resource_count: PRB-group-layer resources the UE held, i.e. the number
            of EESM terms behind its transport block.
        raw_rate_bps: Rate the transport block carried before the error draw,
            ``spectral_eff * resource_count * prbg_bandwidth_hz``. Under full
            buffer this is also the granted rate.
        realized_rate_bps: :attr:`raw_rate_bps` where acknowledged, zero
            otherwise.
    """

    slot: int
    scheduled: torch.Tensor
    mcs: torch.Tensor
    expected_bler: torch.Tensor
    ack: torch.Tensor
    resource_count: torch.Tensor
    raw_rate_bps: torch.Tensor
    realized_rate_bps: torch.Tensor


@runtime_checkable
class LinkAdaptationSimulator(Protocol):
    """A system-level simulator driving the link adaptation environment.

    Three members, and only one of them does any work per slot. See
    ``example_simulator.py`` for a minimal implementation that exercises the
    whole contract, and ``INTEGRATION.md`` for the field-by-field mapping.
    """

    @property
    def carrier(self) -> CarrierProfile:
        """Static carrier and batch description. Must not change after ``reset``."""

    @property
    def phy(self) -> PhyAbstraction:
        """Link-to-system mapping used for EESM, BLER and CQI conversion.

        Return the host simulator's calibrated PHY abstraction. Returning
        :data:`~drlTrainingFramework.phy.DEFAULT_PHY` gets a runnable reference
        model instead, which is fine for wiring up the integration and not fine
        for trusting the resulting throughput.
        """

    def reset(self, *, seed: int | None = None) -> SlotContext:
        """Start an episode and publish the slot-zero context.

        ``seed`` re-seeds the simulator's own randomness. Honouring it is what
        makes the framework's matched evaluation work: the student, the teacher
        and the outer-loop baseline are each replayed on the same seed, so the
        only difference left between them is the MCS decision. An adapter that
        ignores ``seed`` still trains, but its comparisons are noise.
        """

    def advance(self, outcome: SlotOutcome) -> SlotContext:
        """Apply ``outcome``, step the simulator one slot, publish the next context.

        This is where a host simulator advances the channel, updates
        proportional-fair weights from the delivered rate, refreshes CSI when
        the reporting period expires, and reschedules.
        """


#: Builds a simulator instance. ``seed`` is ``None`` to accept the simulator's
#: own default. The launchers resolve a ``module:function`` string to one of
#: these, which is the only place a host simulator is named.
SimulatorFactory = Callable[..., LinkAdaptationSimulator]


def _canonical_device(device: torch.device) -> torch.device:
    """Resolve a device to the concrete form a tensor would report.

    Needed because the two sides of a device comparison are spelled
    differently. ``torch.device("cuda")`` carries no index, but a tensor placed
    on it reports ``cuda:0``, so comparing the two verbatim would reject a
    correctly placed tensor. Resolving the index-less spelling to the current
    default device makes the comparison exact instead of merely comparing
    device *types*, which would let a ``cuda:0`` tensor pass against a
    ``cuda:1`` carrier.
    """

    if device.type == "cpu":
        # A CPU tensor always reports an index of None, so normalize the
        # ``cpu:0`` spelling onto it rather than treating them as different.
        return torch.device("cpu")
    if device.index is None and device.type == "cuda" and torch.cuda.is_available():
        return torch.device("cuda", torch.cuda.current_device())
    return device


def _check_tensor(
    tensor: object,
    name: str,
    shape: tuple[int, ...],
    device: torch.device,
) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise ValueError(f"SlotContext.{name} must be a torch.Tensor")
    if tuple(tensor.shape) != shape:
        raise ValueError(
            f"SlotContext.{name} must have shape {shape}, "
            f"got {tuple(tensor.shape)}"
        )
    expected = _canonical_device(device)
    if _canonical_device(tensor.device) != expected:
        raise ValueError(
            f"SlotContext.{name} must be on device {expected}, "
            f"got {tensor.device}"
        )

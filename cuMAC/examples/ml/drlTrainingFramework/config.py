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

"""Validated configuration for DRL link adaptation.

Three dataclasses, one per concern:

* :class:`LaEnvConfig` - the link adaptation loop itself: action space, BLER
  operating point, reward, feature normalization, and the inner/outer loop
  reference baselines.
* :class:`LaModelConfig` - actor-critic architecture and observation
  normalization.
* :class:`LaTrainingConfig` - the four-phase training curriculum.

Nothing here describes the radio, the channel or the scheduler. Those belong to
the host simulator and reach the framework through
:class:`~drlTrainingFramework.adapter.CarrierProfile`, so a configuration file
in ``config/`` is portable across simulators.

Every file under ``config/`` is loaded by :class:`YamlConfig`, so all three
support organizational sections, ``base:`` inheritance and unknown-key
rejection. See ``yaml_config.py``.
"""

from __future__ import annotations

__all__ = [
    "LA_OVERRIDE_SECTIONS",
    "NUM_MCS",
    "LaEnvConfig",
    "LaModelConfig",
    "LaTrainingConfig",
    "load_la_env_config",
    "load_la_model_config",
    "load_la_training_config",
    "parse_la_config_overrides",
]

import difflib
from dataclasses import Field, dataclass, field, fields
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Mapping

import yaml

from .phy import NUM_MCS
from .yaml_config import (
    YamlConfig,
    finite,
    nonnegative_int,
    positive_int,
)

if TYPE_CHECKING:  # pragma: no cover - import cycle only matters for typing
    from .adapter import CarrierProfile

# Cumulative inner-loop calibration ladder. Each level is a strict superset of
# the one before, and every level is computed from the channel estimate, the
# precoder built from it, the reported interference, and the known channel
# estimation quality. The realized channel is never read.
_ILLA_CALIBRATIONS = (
    "power_split",
    "beamforming_gain",
    "beamforming_and_leakage",
    "estimation_error_floor",
)

# Cell spectral efficiency over the KPI tail window of a closed-loop student
# rollout. ``expected`` folds in BLER; ``realized`` additionally samples
# ACK/NACK and is the noisier estimator of the same quantity.
_SELECTION_METRICS = frozenset(
    {"expected_cell_spectral_eff", "realized_cell_spectral_eff"}
)

_TRAINING_MODES = frozenset({"clone", "full"})


def _layer_widths(name: str, value: object) -> tuple[int, ...]:
    """Validate an MLP width list, allowing an empty list for a bare readout."""

    if isinstance(value, (str, bytes)) or not isinstance(value, Iterable):
        raise ValueError(f"{name} must be a list of positive integers")
    widths = tuple(value)
    for width in widths:
        positive_int(f"{name} entry", width)
    return tuple(int(width) for width in widths)


@dataclass
class LaEnvConfig(YamlConfig):
    """Link adaptation loop settings, independent of the host simulator."""

    # Action space. The agent emits one MCS per scheduled physical UE per slot,
    # which is the transport-block granularity: a UE's resources across every
    # PRB group it holds are aggregated by EESM into a single effective SINR,
    # so one transport block carries one modulation.
    drl_absolute_mcs: bool = True
    action_delta_min: int = 0
    action_delta_max: int = NUM_MCS - 1

    # BLER operating point. Shared by the environment, the teacher and the
    # reward so the three cannot disagree about what "on target" means.
    target_bler: float = 0.10
    # Slack above target before a UE counts as in debt and the teacher's bound
    # tightens.
    bler_cushion: float = 0.05
    # Tightened bound applied while a UE is in debt, forcing repayment before
    # the selector may chase spectral efficiency again.
    debt_bler_bound: float = 0.07
    # Decay of the per-UE sliding BLER window, updated once per scheduled slot.
    bler_ema_alpha: float = 0.02

    # Reward. Per-layer expected goodput keeps credit assignment independent of
    # the rank and the resource count, neither of which link adaptation
    # controls.
    goodput_reward_weight: float = 1.0
    bler_adjust_award: float = 0.0

    # Feature normalization. dB quantities are divided by sinr_feature_norm_db
    # and feedback ages by feedback_age_norm_slots, which defaults to the
    # simulator's CSI reporting period so a value of 1.0 means "one report old".
    sinr_feature_norm_db: float = 30.0
    feedback_age_norm_slots: int | None = None
    obs_clip: float = 5.0

    # Action slots per environment. Defaults to CarrierProfile.num_ues so the
    # action axis is the physical UE index and no compaction is needed.
    max_scheduled_ues: int | None = None

    # Inner-loop calibration converting the reported single-user, full-power
    # CQI into the co-scheduled operating point the scheduler actually created.
    # Cumulative, cheapest first:
    #   power_split             equal split of PRB-group power across the
    #                             co-scheduled layers
    #   beamforming_gain        + post-precoder desired power from the channel
    #                             estimate, so the precoder's steering loss is
    #                             priced instead of assuming a matched filter
    #   beamforming_and_leakage + intra-cell leakage the precoder fails to null,
    #                             from the same estimate
    #   estimation_error_floor  + the leakage those nulls will miss, which the
    #                             estimate structurally cannot see because the
    #                             precoder was solved from it
    #
    # The default stops one level short deliberately. estimation_error_floor
    # makes the inner loop usable on its own, but stacked with the outer loop it
    # double-counts the same missed nulls, and on the reference configuration
    # that cost a double-digit percentage of cell spectral efficiency with OLLA
    # enabled. Choose the floor when running the inner loop alone, and re-sweep
    # the ladder against your own precoder before shipping either choice.
    illa_calibration: str = "beamforming_and_leakage"

    # Gain on the estimation-error leakage floor, in dB. The floor itself is
    #   eps * ||c_est||^2 * p_layer * (L - 1) / n_tx
    # from a first-principles isotropic-error argument, which lands optimistic
    # because a regularized precoder does not null perfectly even in estimate
    # space. This constant closes that gap and is precoder specific: on the
    # reference configuration the realized leakage sat about 12 dB above the
    # unit-gain floor, and 15 dB was the plateau-robust pick across carrier
    # sizes. Re-sweep it for your own precoder.
    # (L - 1) makes the whole term vanish on a single-layer PRB group, so it
    # applies only where a UE is actually co-scheduled.
    illa_leakage_floor_gain_db: float = 15.0

    # Outer-loop link adaptation reference, used as an evaluation baseline and
    # as the anchor for relative (non-absolute) MCS actions.
    olla_enabled: bool = True
    olla_initial_offset_db: float = -2.0
    olla_ack_step_db: float = 0.1
    olla_nack_step_db: float = 1.0
    olla_min_offset_db: float = -20.0
    olla_max_offset_db: float = 6.0
    conservative_mcs_margin_db: float = 0.0

    def __post_init__(self) -> None:
        self.validate()

    @property
    def action_dim(self) -> int:
        """Number of discrete actions the policy chooses between."""

        return int(self.action_delta_max) - int(self.action_delta_min) + 1

    def resolved_feedback_age_norm_slots(self, carrier: "CarrierProfile") -> int:
        """Age normalizer in slots, defaulting to the CSI reporting period."""

        if self.feedback_age_norm_slots is not None:
            return int(self.feedback_age_norm_slots)
        return max(int(carrier.csi_report_period_slots), 1)

    def resolved_max_scheduled_ues(self, carrier: "CarrierProfile") -> int:
        """Action slots per environment, defaulting to every physical UE."""

        if self.max_scheduled_ues is None:
            return int(carrier.num_ues)
        return min(int(self.max_scheduled_ues), int(carrier.num_ues))

    def validate(self) -> "LaEnvConfig":
        """Validate the action space, BLER targets, reward and feature scaling."""

        for name in ("action_delta_min", "action_delta_max"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name} must be an integer")
        if self.action_delta_max < self.action_delta_min:
            raise ValueError(
                "action_delta_max must be >= action_delta_min"
            )
        if self.drl_absolute_mcs:
            if self.action_delta_min != 0:
                raise ValueError(
                    "absolute MCS actions require action_delta_min == 0"
                )
            if self.action_delta_max != NUM_MCS - 1:
                raise ValueError(
                    "absolute MCS actions require action_delta_max == "
                    f"{NUM_MCS - 1}"
                )
        for name in ("target_bler", "bler_cushion", "debt_bler_bound"):
            value = finite(name, getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if not 0.0 < finite("bler_ema_alpha", self.bler_ema_alpha) <= 1.0:
            raise ValueError("bler_ema_alpha must be in (0, 1]")
        if float(self.target_bler) <= 0.0:
            raise ValueError("target_bler must be positive")
        for name in ("goodput_reward_weight", "bler_adjust_award"):
            if finite(name, getattr(self, name)) < 0.0:
                raise ValueError(f"{name} must be nonnegative")
        if finite("sinr_feature_norm_db", self.sinr_feature_norm_db) <= 0.0:
            raise ValueError("sinr_feature_norm_db must be positive")
        if finite("obs_clip", self.obs_clip) <= 0.0:
            raise ValueError("obs_clip must be positive")
        if self.feedback_age_norm_slots is not None:
            positive_int("feedback_age_norm_slots", self.feedback_age_norm_slots)
        if self.max_scheduled_ues is not None:
            positive_int("max_scheduled_ues", self.max_scheduled_ues)
        calibration = self.illa_calibration.strip().lower().replace("-", "_")
        if calibration not in _ILLA_CALIBRATIONS:
            raise ValueError(
                "illa_calibration must be one of "
                f"{', '.join(_ILLA_CALIBRATIONS)}"
            )
        self.illa_calibration = calibration
        finite("illa_leakage_floor_gain_db", self.illa_leakage_floor_gain_db)
        for name in (
            "olla_initial_offset_db",
            "olla_ack_step_db",
            "olla_nack_step_db",
            "olla_min_offset_db",
            "olla_max_offset_db",
            "conservative_mcs_margin_db",
        ):
            finite(name, getattr(self, name))
        if self.olla_min_offset_db > self.olla_max_offset_db:
            raise ValueError(
                "olla_min_offset_db must be <= olla_max_offset_db"
            )
        for name in ("olla_ack_step_db", "olla_nack_step_db"):
            if float(getattr(self, name)) < 0.0:
                raise ValueError(f"{name} must be nonnegative")
        return self


@dataclass
class LaModelConfig(YamlConfig):
    """Actor-critic architecture for link adaptation.

    A shared encoder feeds separate actor and critic heads. An empty head list
    means a single linear readout on the encoder output.
    """

    encoder_layers: list[int] = field(
        default_factory=lambda: [256, 256, 256, 256]
    )
    actor_head_layers: list[int] = field(default_factory=lambda: [256])
    critic_head_layers: list[int] = field(default_factory=lambda: [256, 256])
    obs_normalization: bool = True
    # Standardized observations are clamped to this many standard deviations
    # before the encoder, so an unseen feature scale cannot saturate it.
    obs_norm_clip: float = 10.0
    # Near-zero actor output gain keeps the initial policy close to uniform, so
    # teacher cloning starts from maximum entropy rather than from an arbitrary
    # preferred MCS.
    actor_output_gain: float = 0.01
    critic_output_gain: float = 1.0

    def __post_init__(self) -> None:
        self.validate()

    @property
    def encoder_dim(self) -> int:
        """Width of the shared representation the two heads read."""

        return int(self.encoder_layers[-1])

    def validate(self) -> "LaModelConfig":
        """Validate layer widths and initialization gains."""

        encoder = _layer_widths("encoder_layers", self.encoder_layers)
        if not encoder:
            raise ValueError("encoder_layers must name at least one width")
        self.encoder_layers = list(encoder)
        self.actor_head_layers = list(
            _layer_widths("actor_head_layers", self.actor_head_layers)
        )
        self.critic_head_layers = list(
            _layer_widths("critic_head_layers", self.critic_head_layers)
        )
        if finite("obs_norm_clip", self.obs_norm_clip) <= 0.0:
            raise ValueError("obs_norm_clip must be positive")
        for name in ("actor_output_gain", "critic_output_gain"):
            if finite(name, getattr(self, name)) <= 0.0:
                raise ValueError(f"{name} must be positive")
        return self


@dataclass
class LaTrainingConfig(YamlConfig):
    """Four-phase link adaptation training curriculum.

    ``clone`` stops after teacher cloning and its checkpoint selection.
    ``full`` continues through the optional actor bootstrap, critic warmup and
    PPO fine-tuning.
    """

    training_mode: str = "clone"
    seed: int = 17
    device: str | None = None
    output_dir: str = "runs/drl_la"
    resume_checkpoint: str | None = None
    log_every: int = 100

    # Phase 1: teacher cloning. Gaussian label smoothing over the MCS index
    # encodes that neighbouring MCS are near-equivalent choices, which a
    # one-hot target would penalize as hard as a wildly wrong one.
    clone_updates: int = 10_000
    clone_learning_rate: float = 3.0e-4
    clone_weight_decay: float = 1.0e-2
    clone_label_smoothing_sigma: float = 1.0
    clone_replay_capacity: int = 262_144
    clone_batch_size: int = 2_048
    clone_collect_steps_per_update: int = 1
    # Slots between collections. A simulator slot is typically orders of
    # magnitude more expensive than a gradient step, so cloning at one slot per
    # gradient step spends nearly all its time in the simulator. Raising this
    # trains many times off each labelled slot, which the replay buffer already
    # decorrelates. Raise the simulator's env count alongside it: a wider batch
    # of slots is close to free and keeps the buffer diverse.
    clone_collect_every: int = 1
    clone_checkpoint_every: int | None = None

    # Phase 2: optional actor bootstrap. A single-step REINFORCE pass on the
    # cloned actor closes the imitation-to-on-policy distribution shift before
    # a randomly initialized critic is allowed to influence the actor.
    bootstrap_enabled: bool = False
    bootstrap_updates: int = 2_000
    bootstrap_learning_rate: float = 1.0e-4
    bootstrap_entropy_coef: float = 5.0e-3
    bootstrap_advantage_clip: float = 5.0
    bootstrap_checkpoint_every: int | None = None

    # Phase 3: critic warmup with the encoder and actor frozen, so fitting the
    # value head cannot corrupt the cloned representation.
    critic_warmup_enabled: bool = True
    critic_warmup_updates: int = 500
    critic_warmup_learning_rate: float = 3.0e-4
    critic_warmup_horizon: int = 32
    critic_warmup_epochs: int = 4
    critic_warmup_replay_capacity: int = 131_072
    critic_warmup_batch_size: int = 2_048
    critic_checkpoint_every: int | None = None
    freeze_actor_during_critic_warmup: bool = True

    # Phase 4: PPO fine-tuning. gamma defaults to 0 because the MCS choice for
    # one UE in one slot does not change the next state: the channel, the
    # resource allocation and the scheduler's own weights all evolve
    # independently of it. The only cross-slot coupling is the sliding BLER
    # window, which the reward already prices.
    ppo_enabled: bool = True
    ppo_updates: int = 10_000
    ppo_horizon: int = 32
    ppo_epochs: int = 4
    ppo_minibatch_size: int = 2_048
    ppo_learning_rate: float = 3.0e-4
    gamma: float = 0.0
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    value_clip_coef: float = 0.5
    value_loss_coef: float = 0.25
    entropy_coef: float = 5.0e-3
    max_grad_norm: float = 1.0
    advantage_normalization: bool = True
    reward_normalization: bool = True
    reward_norm_epsilon: float = 1.0e-6
    target_kl: float | None = 0.05
    ppo_checkpoint_every: int | None = None

    # Shared checkpointing and between-phase model selection.
    checkpoint_every: int = 1_000
    checkpoint_selection_steps: int = 0
    checkpoint_selection_seed: int | None = None
    checkpoint_selection_metric: str = "expected_cell_spectral_eff"

    # Validation rollouts scored against the teacher and the outer loop.
    validation_every: int = 0
    validation_steps: int = 100
    validation_seed: int = 10_001
    validation_warmup_slots: int = 0

    def __post_init__(self) -> None:
        self.validate()

    def phase_checkpoint_every(self, phase: str) -> int:
        """Checkpoint cadence for ``phase``, falling back to the shared value."""

        override = {
            "clone": self.clone_checkpoint_every,
            "bootstrap": self.bootstrap_checkpoint_every,
            "critic_warmup": self.critic_checkpoint_every,
            "ppo": self.ppo_checkpoint_every,
        }.get(phase)
        if override is None:
            return int(self.checkpoint_every)
        return int(override)

    def validate(self) -> "LaTrainingConfig":
        """Validate phase lengths, optimizer settings and selection metric."""

        mode = self.training_mode.strip().lower()
        if mode not in _TRAINING_MODES:
            raise ValueError(
                f"training_mode must be one of {', '.join(sorted(_TRAINING_MODES))}"
            )
        self.training_mode = mode
        for name in (
            "clone_updates",
            "bootstrap_updates",
            "critic_warmup_updates",
            "ppo_updates",
            "log_every",
            "checkpoint_selection_steps",
            "validation_every",
            "validation_warmup_slots",
        ):
            nonnegative_int(name, getattr(self, name))
        for name in (
            "seed",
            "clone_replay_capacity",
            "clone_batch_size",
            "clone_collect_steps_per_update",
            "clone_collect_every",
            "critic_warmup_horizon",
            "critic_warmup_epochs",
            "critic_warmup_replay_capacity",
            "critic_warmup_batch_size",
            "ppo_horizon",
            "ppo_epochs",
            "ppo_minibatch_size",
            "checkpoint_every",
            "validation_steps",
            "validation_seed",
        ):
            positive_int(name, getattr(self, name))
        for name in (
            "clone_checkpoint_every",
            "bootstrap_checkpoint_every",
            "critic_checkpoint_every",
            "ppo_checkpoint_every",
            "checkpoint_selection_seed",
        ):
            value = getattr(self, name)
            if value is not None:
                positive_int(name, value)
        for name in (
            "clone_learning_rate",
            "bootstrap_learning_rate",
            "critic_warmup_learning_rate",
            "ppo_learning_rate",
            "max_grad_norm",
        ):
            if finite(name, getattr(self, name)) <= 0.0:
                raise ValueError(f"{name} must be positive")
        for name in (
            "clone_weight_decay",
            "clone_label_smoothing_sigma",
            "bootstrap_entropy_coef",
            "bootstrap_advantage_clip",
            "clip_coef",
            "value_clip_coef",
            "value_loss_coef",
            "entropy_coef",
            "reward_norm_epsilon",
        ):
            if finite(name, getattr(self, name)) < 0.0:
                raise ValueError(f"{name} must be nonnegative")
        gamma = finite("gamma", self.gamma)
        if not 0.0 <= gamma <= 1.0:
            raise ValueError("gamma must be in [0, 1]")
        gae_lambda = finite("gae_lambda", self.gae_lambda)
        if not 0.0 <= gae_lambda <= 1.0:
            raise ValueError("gae_lambda must be in [0, 1]")
        if self.target_kl is not None and finite("target_kl", self.target_kl) <= 0.0:
            raise ValueError("target_kl must be positive when set")
        if self.checkpoint_selection_metric not in _SELECTION_METRICS:
            raise ValueError(
                "checkpoint_selection_metric must be one of "
                f"{', '.join(sorted(_SELECTION_METRICS))}"
            )
        return self


def load_la_env_config(
    path: str | Path,
    *,
    overrides: Mapping[str, Any] | None = None,
) -> LaEnvConfig:
    """Load :class:`LaEnvConfig` from YAML."""

    return LaEnvConfig.from_yaml(path, overrides=overrides)


def load_la_model_config(
    path: str | Path,
    *,
    overrides: Mapping[str, Any] | None = None,
) -> LaModelConfig:
    """Load :class:`LaModelConfig` from YAML."""

    return LaModelConfig.from_yaml(path, overrides=overrides)


def load_la_training_config(
    path: str | Path,
    *,
    overrides: Mapping[str, Any] | None = None,
) -> LaTrainingConfig:
    """Load :class:`LaTrainingConfig` from YAML."""

    return LaTrainingConfig.from_yaml(path, overrides=overrides)


LA_OVERRIDE_SECTIONS: Mapping[str, type] = {
    "env": LaEnvConfig,
    "model": LaModelConfig,
    "train": LaTrainingConfig,
}


def _coerce_override(text: str, config_field: Field) -> Any:
    """Parse an override value, honoring the target field's numeric type."""

    value = yaml.safe_load(text)
    if not isinstance(value, str):
        return value
    # YAML 1.1 reads bare scientific notation such as 3e-4 as a string.
    annotation = str(config_field.type)
    for name, converter in (("float", float), ("int", int)):
        if name in annotation:
            try:
                return converter(value)
            except ValueError:
                break
    return value


def parse_la_config_overrides(
    assignments: Iterable[str],
) -> dict[str, dict[str, Any]]:
    """Parse ``section.field=value`` strings into per-section override maps.

    Sections are the keys of :data:`LA_OVERRIDE_SECTIONS`. Values use YAML
    scalar syntax, so ``true``, ``null``, ``12`` and ``1.5`` keep their types.
    """

    parsed: dict[str, dict[str, Any]] = {
        name: {} for name in LA_OVERRIDE_SECTIONS
    }
    for assignment in assignments:
        if "=" not in assignment:
            raise ValueError(
                f"override {assignment!r} must use section.field=value syntax"
            )
        target, _, text = assignment.partition("=")
        section, dot, field_name = target.strip().partition(".")
        if not dot:
            raise ValueError(
                f"override {assignment!r} must name a section: "
                f"{', '.join(sorted(LA_OVERRIDE_SECTIONS))}"
            )
        if section not in LA_OVERRIDE_SECTIONS:
            raise ValueError(
                f"override {assignment!r} has unknown section {section!r}; "
                f"expected one of {', '.join(sorted(LA_OVERRIDE_SECTIONS))}"
            )
        available = {
            config_field.name: config_field
            for config_field in fields(LA_OVERRIDE_SECTIONS[section])
        }
        if field_name not in available:
            hint = difflib.get_close_matches(field_name, available, n=1)
            suffix = f"; did you mean {section}.{hint[0]}?" if hint else ""
            raise ValueError(
                f"override {assignment!r} has unknown "
                f"{LA_OVERRIDE_SECTIONS[section].__name__} field "
                f"{field_name!r}{suffix}"
            )
        if field_name in parsed[section]:
            raise ValueError(
                f"override {section}.{field_name} was given more than once"
            )
        parsed[section][field_name] = _coerce_override(
            text,
            available[field_name],
        )
    return parsed

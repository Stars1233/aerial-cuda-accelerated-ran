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

"""Four-phase training curriculum for DRL link adaptation.

The phases exist because each one removes a specific failure of the next::

    1 teacher cloning    a uniform policy over 28 MCS never survives contact
                         with a BLER target, so the actor is first regressed
                         onto the perfect-information debt-SE-aware selector
    2 actor bootstrap    imitation is evaluated on the teacher's own state
                         distribution; a short REINFORCE pass moves the actor
                         onto its own (optional, off by default)
    3 critic warmup      a randomly initialized value head would inject pure
                         noise into the first PPO advantages, so it is fitted
                         with the encoder and actor frozen
    4 PPO fine-tuning    only now is the full objective safe to optimize

Between the phases that change the actor, every checkpoint is replayed on a
held-out rollout and the best cell spectral efficiency is carried forward, so a
phase can never hand its successor a worse policy than one it already had.

``gamma`` defaults to zero, which makes phase 4 a contextual bandit. The MCS
chosen for one UE in one slot does not move the next state: the channel, the
resource allocation and the scheduler's own weights all evolve independently of
it. The one cross-slot coupling, the sliding BLER window, is already priced
inside the reward.

The framework never constructs a simulator. It takes an :data:`EnvFactory`,
calls it once for training and once for the held-out validation environment, and
seeds each rollout through ``reset(seed=...)``. Wiring a host simulator in
therefore means supplying one function; see ``INTEGRATION.md``.
"""

from __future__ import annotations

__all__ = [
    "ActionFn",
    "BootstrapMetrics",
    "CloneMetrics",
    "CriticWarmupMetrics",
    "EnvFactory",
    "LaCheckpointSelectionMetrics",
    "LaRolloutBatch",
    "LaTrainingResult",
    "LaValidationMetrics",
    "LinkAdaptationTrainingFramework",
    "PPOMetrics",
    "simulator_env_factory",
]

import csv
import math
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import torch
import yaml
from torch import nn
from torch.nn import functional as F

from .adapter import CarrierProfile, LinkAdaptationSimulator, SimulatorFactory
from .config import LaEnvConfig, LaModelConfig, LaTrainingConfig
from .env import LinkAdaptationEnv
from .model import (
    LinkAdaptationActorCritic,
    parameter_groups_for_weight_decay,
)
from .teacher import gaussian_mcs_targets

#: Builds a link adaptation environment around the host simulator. The
#: framework owns the authoritative :class:`LaEnvConfig` and passes it in, so a
#: factory only has to construct the simulator and wrap it.
EnvFactory = Callable[[LaEnvConfig], LinkAdaptationEnv]

#: Chooses an MCS action for one slot, or ``None`` to fall back to the
#: environment's outer-loop baseline.
ActionFn = Callable[[LinkAdaptationEnv, torch.Tensor], "torch.Tensor | None"]

#: Consecutive slots without a single policy-decided UE before the actor
#: bootstrap gives up. Generous enough to ride out an ordinary burst of
#: retransmission-only slots, tight enough that an environment which never
#: presents a learnable UE stops in seconds instead of consuming a slot per
#: requested update and learning nothing.
_BOOTSTRAP_MAX_CONSECUTIVE_SKIPS = 128


def simulator_env_factory(
    simulator_factory: SimulatorFactory,
    *,
    seed: int = 0,
) -> EnvFactory:
    """Wrap a simulator factory into an :data:`EnvFactory`.

    Convenience for the common case where the host simulator is constructed the
    same way for training and validation and only the rollout seed differs.
    """

    def build(env_config: LaEnvConfig) -> LinkAdaptationEnv:
        simulator: LinkAdaptationSimulator = simulator_factory()
        return LinkAdaptationEnv(simulator, env_config, seed=seed)

    return build


@dataclass(frozen=True)
class CloneMetrics:
    """One teacher-cloning gradient step."""

    update: int
    loss: float
    teacher_match: float
    mean_abs_mcs_error: float
    entropy: float
    gradient_norm: float
    replay_rows: int
    update_time_ms: float


@dataclass(frozen=True)
class BootstrapMetrics:
    """One actor-bootstrap gradient step."""

    update: int
    policy_loss: float
    entropy: float
    mean_reward: float
    mean_bler: float
    gradient_norm: float


@dataclass(frozen=True)
class CriticWarmupMetrics:
    """One critic-warmup outer update."""

    update: int
    value_loss: float
    explained_variance: float
    mean_return: float
    replay_rows: int


@dataclass(frozen=True)
class PPOMetrics:
    """One PPO outer update."""

    update: int
    policy_loss: float
    value_loss: float
    entropy: float
    approximate_kl: float
    clip_fraction: float
    mean_reward: float
    mean_cell_spectral_eff: float
    mean_bler: float
    epochs_run: int


@dataclass(frozen=True)
class LaValidationMetrics:
    """Held-out comparison of the student against its reference schemes."""

    update: int
    phase: str
    steps: int
    seed: int
    student_cell_spectral_eff: float
    teacher_cell_spectral_eff: float
    olla_cell_spectral_eff: float
    genie_cell_spectral_eff: float
    student_teacher_ratio: float
    student_olla_ratio: float
    illa_cell_spectral_eff: float
    student_bler: float
    teacher_bler: float
    olla_bler: float
    illa_bler: float
    student_mean_mcs: float
    teacher_mean_mcs: float
    teacher_action_match: float


@dataclass(frozen=True)
class LaCheckpointSelectionMetrics:
    """One candidate checkpoint scored during between-phase selection."""

    phase: str
    update: int
    steps: int
    seed: int
    expected_cell_spectral_eff: float
    realized_cell_spectral_eff: float
    mean_bler: float
    selected: int


@dataclass(frozen=True)
class LaTrainingResult:
    """Artifacts and terminal state of the configured curriculum."""

    completed_clone_updates: int
    completed_bootstrap_updates: int
    completed_critic_updates: int
    completed_ppo_updates: int
    phase: str
    output_dir: str
    observation_dim: int
    action_dim: int
    selected_clone_checkpoint: str | None
    selected_bootstrap_checkpoint: str | None
    selected_ppo_checkpoint: str | None


@dataclass
class LaRolloutBatch:
    """One on-policy rollout, time-major with an active-UE mask.

    ``active`` is what was scheduled and is what the reported metrics average
    over. ``learnable`` is the subset whose MCS the policy actually chose, so it
    excludes retransmissions, whose modulation is replayed from the first
    attempt; it is what every loss and replay filters on.
    """

    observations: torch.Tensor
    actions: torch.Tensor
    log_probs: torch.Tensor
    values: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor
    active: torch.Tensor
    last_value: torch.Tensor
    mean_cell_spectral_eff: float
    mean_bler: float
    learnable: torch.Tensor | None = None

    @property
    def trainable_mask(self) -> torch.Tensor:
        return self.active if self.learnable is None else self.learnable


def _write_csv_row(path: Path, row: Mapping[str, object]) -> None:
    """Append one row, reusing the header already on disk."""

    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    fieldnames = list(row)
    if exists:
        with path.open(newline="", encoding="utf-8") as stream:
            existing = next(csv.reader(stream), None)
        if existing is not None:
            fieldnames = existing
    with path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def _config_dict(config: Any) -> dict[str, Any]:
    """Shallow field mapping, avoiding the deep copy ``asdict`` would make."""

    return {
        config_field.name: getattr(config, config_field.name)
        for config_field in fields(config)
    }


def _carrier_dict(carrier: CarrierProfile) -> dict[str, Any]:
    """YAML-serializable view of the carrier the run was trained against."""

    record = _config_dict(carrier)
    record["device"] = str(record["device"])
    return record


def _load_checkpoint_file(
    path: str | Path,
    device: torch.device,
) -> dict[str, Any]:
    """Read a checkpoint with pickle code execution disabled.

    ``weights_only=True`` confines deserialization to tensors and plain Python
    containers, so reading a checkpoint cannot execute code as a side effect.
    That matters because checkpoints are ordinary files that get copied between
    machines, shared between teams and picked up by glob in
    :meth:`LinkAdaptationTrainingFramework.select_best_checkpoint`, which loads
    whatever matches the pattern in the output directory.

    Nothing is given up: :meth:`_checkpoint_payload` writes only tensors,
    primitives, dicts and lists. The flag is passed explicitly rather than left
    to the torch default, which only became ``True`` in 2.6 while this package
    supports 2.4 upward.
    """

    return torch.load(Path(path), map_location=device, weights_only=True)


class _RingReplay:
    """Fixed-capacity observation/target buffer with uniform sampling.

    Cloning and critic warmup both need to decorrelate a batch from the slot it
    was collected in: within one slot the UEs share a channel realization and a
    resource allocation, so an on-policy batch is far less diverse than its size
    suggests.
    """

    def __init__(
        self,
        capacity: int,
        obs_dim: int,
        *,
        device: torch.device,
        target_dtype: torch.dtype,
    ) -> None:
        self.capacity = int(capacity)
        self.observations = torch.zeros(
            (self.capacity, int(obs_dim)),
            device=device,
            dtype=torch.float32,
        )
        self.targets = torch.zeros(
            (self.capacity,),
            device=device,
            dtype=target_dtype,
        )
        self.cursor = 0
        self.size = 0

    def add(
        self,
        observations: torch.Tensor,
        targets: torch.Tensor,
        active: torch.Tensor,
    ) -> None:
        flat_obs = observations.reshape(-1, observations.shape[-1])
        flat_target = targets.reshape(-1)
        keep = active.reshape(-1)
        flat_obs = flat_obs[keep]
        flat_target = flat_target[keep]
        count = int(flat_obs.shape[0])
        if count == 0:
            return
        if count >= self.capacity:
            flat_obs = flat_obs[-self.capacity :]
            flat_target = flat_target[-self.capacity :]
            count = self.capacity
        end = self.cursor + count
        if end <= self.capacity:
            self.observations[self.cursor : end] = flat_obs
            self.targets[self.cursor : end] = flat_target
        else:
            head = self.capacity - self.cursor
            self.observations[self.cursor :] = flat_obs[:head]
            self.targets[self.cursor :] = flat_target[:head]
            self.observations[: end - self.capacity] = flat_obs[head:]
            self.targets[: end - self.capacity] = flat_target[head:]
        self.cursor = end % self.capacity
        self.size = min(self.size + count, self.capacity)

    def sample(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self.size == 0:
            raise RuntimeError("replay buffer is empty")
        index = torch.randint(
            0,
            self.size,
            (int(min(batch_size, self.size)),),
            device=self.observations.device,
        )
        return self.observations[index], self.targets[index]


class _RewardNormalizer:
    """Scale-only reward normalization over running discounted returns.

    The mean is deliberately not removed: subtracting it would change the sign
    of the advantage for a UE that is merely below average, and every MCS the
    policy can choose already earns nonnegative goodput.
    """

    def __init__(self, gamma: float, epsilon: float) -> None:
        self.gamma = float(gamma)
        self.epsilon = float(epsilon)
        self.count = 0.0
        self.mean = 0.0
        self.m2 = 0.0
        self._returns: torch.Tensor | None = None

    def update(self, rewards: torch.Tensor, done: torch.Tensor) -> None:
        if self._returns is None or self._returns.shape != rewards.shape:
            self._returns = torch.zeros_like(rewards)
        self._returns = self._returns * self.gamma + rewards
        if bool(done.any()):
            self._returns = torch.where(
                done.view(-1, *([1] * (rewards.dim() - 1))),
                torch.zeros_like(self._returns),
                self._returns,
            )
        flat = self._returns.reshape(-1)
        batch_count = float(flat.numel())
        if batch_count == 0.0:
            return
        batch_mean = float(flat.mean())
        batch_m2 = float(flat.var(unbiased=False)) * batch_count
        delta = batch_mean - self.mean
        total = self.count + batch_count
        self.mean += delta * batch_count / total
        self.m2 += batch_m2 + delta * delta * self.count * batch_count / total
        self.count = total

    def normalize(self, rewards: torch.Tensor) -> torch.Tensor:
        if self.count < 2.0:
            return rewards
        std = math.sqrt(max(self.m2 / self.count, 0.0))
        return rewards / max(std, self.epsilon)

    def state_dict(self) -> dict[str, float]:
        return {"count": self.count, "mean": self.mean, "m2": self.m2}

    def load_state_dict(self, state: Mapping[str, float]) -> None:
        self.count = float(state.get("count", 0.0))
        self.mean = float(state.get("mean", 0.0))
        self.m2 = float(state.get("m2", 0.0))


class LinkAdaptationTrainingFramework:
    """Run teacher cloning only, or the complete four-phase curriculum.

    Args:
        env_factory: Builds a link adaptation environment around the host
            simulator, given the environment configuration. Called once for
            training and once more for the held-out validation environment.
        env_config: Link adaptation loop settings, handed to ``env_factory``.
        model_config: Actor-critic architecture.
        training_config: Curriculum, optimizer and checkpointing settings.
        policy: Pre-built policy, for resuming or for architecture experiments.
        train_env: Pre-built training environment, bypassing ``env_factory`` for
            the training rollouts only.
    """

    CHECKPOINT_VERSION = 2

    def __init__(
        self,
        env_factory: EnvFactory,
        env_config: LaEnvConfig | None = None,
        model_config: LaModelConfig | None = None,
        training_config: LaTrainingConfig | None = None,
        *,
        policy: LinkAdaptationActorCritic | None = None,
        train_env: LinkAdaptationEnv | None = None,
    ) -> None:
        self.env_factory = env_factory
        self.env_config = (env_config or LaEnvConfig()).validate()
        self.model_config = (model_config or LaModelConfig()).validate()
        self.training_config = (training_config or LaTrainingConfig()).validate()

        torch.manual_seed(self.training_config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.training_config.seed)

        self.env = train_env if train_env is not None else self._make_env()
        self.device = self.env.device
        self.policy = (
            policy
            or LinkAdaptationActorCritic(
                self.env.obs_dim,
                self.env.action_dim,
                self.model_config,
            )
        ).to(self.device)

        self.output_dir = Path(self.training_config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.clone_metrics_path = self.output_dir / "clone_metrics.csv"
        self.bootstrap_metrics_path = self.output_dir / "bootstrap_metrics.csv"
        self.critic_metrics_path = self.output_dir / "critic_warmup_metrics.csv"
        self.ppo_metrics_path = self.output_dir / "ppo_metrics.csv"
        self.validation_metrics_path = self.output_dir / "validation_metrics.csv"
        self.latest_checkpoint_path = self.output_dir / "checkpoint_latest.pt"

        self.phase = "clone"
        self.completed_clone_updates = 0
        self.completed_bootstrap_updates = 0
        self.completed_critic_updates = 0
        self.completed_ppo_updates = 0

        self.optimizer = torch.optim.AdamW(
            parameter_groups_for_weight_decay(
                self.policy,
                self.training_config.clone_weight_decay,
            ),
            lr=self.training_config.clone_learning_rate,
        )
        self.reward_normalizer = _RewardNormalizer(
            self.training_config.gamma,
            self.training_config.reward_norm_epsilon,
        )
        self._validation_env: LinkAdaptationEnv | None = None
        self.observation = self.env.reset(seed=self.training_config.seed)
        self._write_run_metadata()

        if self.training_config.resume_checkpoint:
            self.load_checkpoint(self.training_config.resume_checkpoint)

    def _make_env(self) -> LinkAdaptationEnv:
        """Build an environment through the injected factory."""

        env = self.env_factory(self.env_config)
        if not isinstance(env, LinkAdaptationEnv):
            raise TypeError(
                "env_factory must return a LinkAdaptationEnv, got "
                f"{type(env).__name__}"
            )
        return env

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------

    def train(self) -> LaTrainingResult:
        """Run cloning only, or the complete curriculum, per training_mode."""

        self.teacher_clone()
        clone_selected = self.select_best_checkpoint(
            "checkpoint_clone_*.pt",
            phase="clone",
        )
        bootstrap_selected: Path | None = None
        ppo_selected: Path | None = None

        if self.training_config.training_mode == "full":
            if (
                self.training_config.bootstrap_enabled
                and self.training_config.bootstrap_updates > 0
            ):
                self.actor_bootstrap()
                bootstrap_selected = self.select_best_checkpoint(
                    "checkpoint_bootstrap_*.pt",
                    phase="bootstrap",
                )
            if (
                self.training_config.critic_warmup_enabled
                and self.training_config.critic_warmup_updates > 0
            ):
                # No selection after warmup: the actor is frozen, so every
                # candidate scores identically on a policy-driven rollout.
                self.critic_warmup()
            if (
                self.training_config.ppo_enabled
                and self.training_config.ppo_updates > 0
            ):
                self.ppo_fine_tune()
                ppo_selected = self.select_best_checkpoint(
                    "checkpoint_ppo_*.pt",
                    phase="ppo",
                )
            self.phase = "complete"

        return LaTrainingResult(
            completed_clone_updates=self.completed_clone_updates,
            completed_bootstrap_updates=self.completed_bootstrap_updates,
            completed_critic_updates=self.completed_critic_updates,
            completed_ppo_updates=self.completed_ppo_updates,
            phase=self.phase,
            output_dir=str(self.output_dir),
            observation_dim=self.env.obs_dim,
            action_dim=self.env.action_dim,
            selected_clone_checkpoint=(
                str(clone_selected) if clone_selected else None
            ),
            selected_bootstrap_checkpoint=(
                str(bootstrap_selected) if bootstrap_selected else None
            ),
            selected_ppo_checkpoint=(
                str(ppo_selected) if ppo_selected else None
            ),
        )

    # ------------------------------------------------------------------
    # Phase 1: teacher cloning
    # ------------------------------------------------------------------

    def teacher_clone(self) -> list[CloneMetrics]:
        """Regress the actor onto the perfect-information teacher."""

        config = self.training_config
        self.phase = "clone"
        self._set_optimizer(
            parameter_groups_for_weight_decay(
                self.policy,
                config.clone_weight_decay,
            ),
            config.clone_learning_rate,
        )
        replay = _RingReplay(
            config.clone_replay_capacity,
            self.env.obs_dim,
            device=self.device,
            target_dtype=torch.long,
        )
        history: list[CloneMetrics] = []
        start = self.completed_clone_updates + 1
        target = config.clone_updates
        collect_every = max(int(config.clone_collect_every), 1)
        for update in range(start, target + 1):
            step_start = time.perf_counter()
            # A simulator slot is typically far more expensive than a gradient
            # step, so collecting less often lets one labelled slot fund many
            # updates.
            if (update - start) % collect_every == 0:
                for _ in range(config.clone_collect_steps_per_update):
                    observation = self.env.la_observation
                    teacher = self.env.teacher_mcs()
                    # A retransmission replays its first attempt's modulation,
                    # so the teacher label there is not a decision the student
                    # can be asked to reproduce.
                    replay.add(
                        observation.features,
                        teacher,
                        observation.learnable,
                    )
                    self.policy.obs_normalizer.update(
                        observation.features,
                        observation.active,
                    )
                    with torch.no_grad():
                        self.observation, _r, _d, _i = self.env.step(teacher)

            batch_obs, batch_target = replay.sample(config.clone_batch_size)
            logits, _value = self.policy(batch_obs)
            soft_target = gaussian_mcs_targets(
                batch_target,
                self.env.action_dim,
                config.clone_label_smoothing_sigma,
            )
            log_prob = F.log_softmax(logits, dim=-1)
            loss = -(soft_target * log_prob).sum(dim=-1).mean()
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient_norm = nn.utils.clip_grad_norm_(
                self.policy.parameters(),
                config.max_grad_norm,
            )
            self.optimizer.step()
            self.completed_clone_updates = update

            with torch.no_grad():
                predicted = logits.argmax(dim=-1)
                match = (predicted == batch_target).to(torch.float32).mean()
                abs_error = (
                    (predicted - batch_target).abs().to(torch.float32).mean()
                )
                entropy = (
                    torch.distributions.Categorical(logits=logits)
                    .entropy()
                    .mean()
                )
            metrics = CloneMetrics(
                update=update,
                loss=float(loss.detach()),
                teacher_match=float(match),
                mean_abs_mcs_error=float(abs_error),
                entropy=float(entropy),
                gradient_norm=float(gradient_norm),
                replay_rows=replay.size,
                update_time_ms=1.0e3 * (time.perf_counter() - step_start),
            )
            history.append(metrics)
            _write_csv_row(self.clone_metrics_path, asdict(metrics))
            if self._should_log(update, start, target):
                print(
                    f"la_clone,{update},loss,{metrics.loss:.6f},"
                    f"teacher_match,{metrics.teacher_match:.4f},"
                    f"mcs_error,{metrics.mean_abs_mcs_error:.3f},"
                    f"entropy,{metrics.entropy:.4f},"
                    f"gradient_norm,{metrics.gradient_norm:.4f},"
                    f"replay_rows,{metrics.replay_rows}",
                    flush=True,
                )
            self._maybe_validate(update, target, "clone")
            self._maybe_checkpoint(update, target, "clone")
        return history

    # ------------------------------------------------------------------
    # Phase 2: optional actor bootstrap
    # ------------------------------------------------------------------

    def actor_bootstrap(self) -> list[BootstrapMetrics]:
        """Move the cloned actor onto its own state distribution."""

        config = self.training_config
        self.phase = "bootstrap"
        self._set_optimizer(
            [{"params": self.policy.actor_parameters(), "weight_decay": 0.0}],
            config.bootstrap_learning_rate,
        )
        history: list[BootstrapMetrics] = []
        start = self.completed_bootstrap_updates + 1
        target = config.bootstrap_updates
        # bootstrap_updates counts optimizer steps, as it does in every other
        # phase, so the loop advances on gradient steps rather than on slots
        # consumed. Slots that carry nothing for the policy to learn from are
        # spent but not counted.
        update = start - 1
        skipped = 0
        consecutive_skips = 0
        while update < target:
            observation = self.env.la_observation
            # The normalizer keeps every scheduled row, because the policy is
            # queried on a retransmitting UE too even though its answer is
            # discarded, so those rows are part of the input distribution. Only
            # the loss narrows to the rows the policy actually decided.
            self.policy.obs_normalizer.update(
                observation.features,
                observation.active,
            )
            active = observation.learnable
            logits, _value = self.policy(observation.features)
            distribution = torch.distributions.Categorical(logits=logits)
            action = distribution.sample()
            log_prob = distribution.log_prob(action)
            entropy = distribution.entropy()

            with torch.no_grad():
                self.observation, reward, _done, info = self.env.step(action)
                advantage = reward.detach()
                selected = advantage[active]
                if selected.numel() > 1:
                    advantage = (advantage - selected.mean()) / (
                        selected.std() + 1e-8
                    )
                clip = float(config.bootstrap_advantage_clip)
                if clip > 0.0:
                    advantage = advantage.clamp(-clip, clip)

            # A slot can legitimately contain nothing the policy decided: no UE
            # scheduled at all, or every scheduled UE replaying a
            # retransmission. There is no gradient to take from it, so it must
            # not consume one of the requested updates or record a metrics row
            # of zeros that would read as a real measurement. (A masked mean
            # over an empty selection is also nan, and one backward pass is
            # enough to write nan into the optimizer moments.) The environment
            # has already advanced, so the slot is spent, not replayed.
            if not bool(active.any()):
                skipped += 1
                consecutive_skips += 1
                if skipped == 1:
                    # Once, not per occurrence: the closing summary carries the
                    # total, and a per-slot line could bury the real output.
                    print(
                        f"la_bootstrap_skipped,completed,{update},"
                        f"reason,no_policy_decided_ue",
                        flush=True,
                    )
                if consecutive_skips >= _BOOTSTRAP_MAX_CONSECUTIVE_SKIPS:
                    print(
                        "la_bootstrap_aborted,consecutive_skipped_slots,"
                        f"{consecutive_skips},completed,{update},"
                        f"requested,{target}",
                        flush=True,
                    )
                    break
                continue

            consecutive_skips = 0
            update += 1
            policy_loss = -(log_prob[active] * advantage[active]).mean()
            entropy_bonus = entropy[active].mean()
            loss = policy_loss - config.bootstrap_entropy_coef * entropy_bonus
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient_norm = nn.utils.clip_grad_norm_(
                self.policy.parameters(),
                config.max_grad_norm,
            )
            self.optimizer.step()
            self.completed_bootstrap_updates = update

            metrics = BootstrapMetrics(
                update=update,
                policy_loss=float(policy_loss.detach()),
                entropy=float(entropy_bonus.detach()),
                mean_reward=float(reward[active].mean()),
                mean_bler=float(info["mean_expected_bler"].mean()),
                gradient_norm=float(gradient_norm),
            )
            history.append(metrics)
            _write_csv_row(self.bootstrap_metrics_path, asdict(metrics))
            if self._should_log(update, start, target):
                print(
                    f"la_bootstrap,{update},"
                    f"policy_loss,{metrics.policy_loss:.6f},"
                    f"entropy,{metrics.entropy:.4f},"
                    f"mean_reward,{metrics.mean_reward:.4f},"
                    f"mean_bler,{metrics.mean_bler:.4f}",
                    flush=True,
                )
            self._maybe_checkpoint(update, target, "bootstrap")

        if skipped:
            print(
                f"la_bootstrap_summary,optimizer_steps,{len(history)},"
                f"requested,{max(target - start + 1, 0)},"
                f"skipped_slots,{skipped}",
                flush=True,
            )
        return history

    # ------------------------------------------------------------------
    # Phase 3: critic warmup
    # ------------------------------------------------------------------

    def critic_warmup(self) -> list[CriticWarmupMetrics]:
        """Fit the value head with the encoder and actor frozen."""

        config = self.training_config
        self.phase = "critic_warmup"
        freeze = config.freeze_actor_during_critic_warmup
        frozen: list[nn.Parameter] = []
        if freeze:
            for parameter in self.policy.actor_parameters():
                if parameter.requires_grad:
                    parameter.requires_grad_(False)
                    frozen.append(parameter)
            self.policy.encoder.eval()
            self.policy.actor.eval()
        self._set_optimizer(
            [{"params": self.policy.critic_parameters(), "weight_decay": 0.0}],
            config.critic_warmup_learning_rate,
        )
        replay = _RingReplay(
            config.critic_warmup_replay_capacity,
            self.env.obs_dim,
            device=self.device,
            target_dtype=torch.float32,
        )
        history: list[CriticWarmupMetrics] = []
        start = self.completed_critic_updates + 1
        target = config.critic_warmup_updates
        try:
            for update in range(start, target + 1):
                rollout = self.collect_rollout(
                    config.critic_warmup_horizon,
                    update_normalizer=False,
                )
                returns = self._discounted_returns(rollout)
                replay.add(
                    rollout.observations,
                    returns,
                    rollout.trainable_mask,
                )
                value_loss = 0.0
                predictions: torch.Tensor | None = None
                targets: torch.Tensor | None = None
                for _ in range(config.critic_warmup_epochs):
                    batch_obs, batch_target = replay.sample(
                        config.critic_warmup_batch_size
                    )
                    _logits, value = self.policy(batch_obs)
                    loss = 0.5 * (value - batch_target).square().mean()
                    self.optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(
                        self.policy.critic_parameters(),
                        config.max_grad_norm,
                    )
                    self.optimizer.step()
                    value_loss = float(loss.detach())
                    predictions = value.detach()
                    targets = batch_target
                self.completed_critic_updates = update

                metrics = CriticWarmupMetrics(
                    update=update,
                    value_loss=value_loss,
                    explained_variance=_explained_variance(predictions, targets),
                    mean_return=float(returns[rollout.active].mean())
                    if bool(rollout.active.any())
                    else 0.0,
                    replay_rows=replay.size,
                )
                history.append(metrics)
                _write_csv_row(self.critic_metrics_path, asdict(metrics))
                if self._should_log(update, start, target):
                    print(
                        f"la_critic_warmup,{update},"
                        f"value_loss,{metrics.value_loss:.6f},"
                        f"explained_variance,{metrics.explained_variance:.4f},"
                        f"mean_return,{metrics.mean_return:.4f},"
                        f"replay_rows,{metrics.replay_rows}",
                        flush=True,
                    )
                self._maybe_checkpoint(update, target, "critic_warmup")
        finally:
            for parameter in frozen:
                parameter.requires_grad_(True)
            self.policy.train()
        return history

    # ------------------------------------------------------------------
    # Phase 4: PPO fine-tuning
    # ------------------------------------------------------------------

    def ppo_fine_tune(self) -> list[PPOMetrics]:
        """Optimize the clipped PPO objective over the whole policy."""

        config = self.training_config
        self.phase = "ppo"
        self._set_optimizer(
            parameter_groups_for_weight_decay(self.policy, 0.0),
            config.ppo_learning_rate,
        )
        history: list[PPOMetrics] = []
        start = self.completed_ppo_updates + 1
        target = config.ppo_updates
        for update in range(start, target + 1):
            rollout = self.collect_rollout(config.ppo_horizon)
            metrics = self.ppo_update(rollout, update=update)
            self.completed_ppo_updates = update
            history.append(metrics)
            _write_csv_row(self.ppo_metrics_path, asdict(metrics))
            if self._should_log(update, start, target):
                print(
                    f"la_ppo,{update},"
                    f"policy_loss,{metrics.policy_loss:.6f},"
                    f"value_loss,{metrics.value_loss:.6f},"
                    f"entropy,{metrics.entropy:.4f},"
                    f"approximate_kl,{metrics.approximate_kl:.6f},"
                    f"clip_fraction,{metrics.clip_fraction:.4f},"
                    f"cell_se,{metrics.mean_cell_spectral_eff:.4f},"
                    f"mean_bler,{metrics.mean_bler:.4f}",
                    flush=True,
                )
            self._maybe_validate(update, target, "ppo")
            self._maybe_checkpoint(update, target, "ppo")
        return history

    def collect_rollout(
        self,
        horizon: int,
        *,
        update_normalizer: bool = True,
    ) -> LaRolloutBatch:
        """Run ``horizon`` slots on the current policy, time-major."""

        observations: list[torch.Tensor] = []
        actions: list[torch.Tensor] = []
        log_probs: list[torch.Tensor] = []
        values: list[torch.Tensor] = []
        rewards: list[torch.Tensor] = []
        dones: list[torch.Tensor] = []
        active: list[torch.Tensor] = []
        learnable: list[torch.Tensor] = []
        cell_se: list[float] = []
        blers: list[float] = []

        for _ in range(int(horizon)):
            observation = self.env.la_observation
            if update_normalizer:
                self.policy.obs_normalizer.update(
                    observation.features,
                    observation.active,
                )
            with torch.no_grad():
                action, log_prob, _entropy, value = self.policy.act(
                    observation.features
                )
                next_observation, reward, done, info = self.env.step(action)
            observations.append(observation.features)
            actions.append(action)
            log_probs.append(log_prob)
            values.append(value)
            rewards.append(reward)
            dones.append(done)
            active.append(observation.active)
            learnable.append(observation.learnable)
            cell_se.append(float(info["expected_cell_spectral_eff"].mean()))
            blers.append(float(info["mean_expected_bler"].mean()))
            self.observation = next_observation

        with torch.no_grad():
            _logits, last_value = self.policy(self.env.la_observation.features)
        return LaRolloutBatch(
            observations=torch.stack(observations),
            actions=torch.stack(actions),
            log_probs=torch.stack(log_probs),
            values=torch.stack(values),
            rewards=torch.stack(rewards),
            dones=torch.stack(dones),
            active=torch.stack(active),
            last_value=last_value,
            mean_cell_spectral_eff=sum(cell_se) / max(len(cell_se), 1),
            mean_bler=sum(blers) / max(len(blers), 1),
            learnable=torch.stack(learnable),
        )

    def ppo_update(self, rollout: LaRolloutBatch, *, update: int) -> PPOMetrics:
        """One PPO outer update over the collected rollout."""

        config = self.training_config
        rewards = self._normalize_rewards(rollout)
        advantages, returns = self._generalized_advantage(rollout, rewards)

        keep = rollout.trainable_mask.reshape(-1)
        observations = rollout.observations
        flat_obs = observations.reshape(-1, observations.shape[-1])[keep]
        flat_actions = rollout.actions.reshape(-1)[keep]
        flat_log_probs = rollout.log_probs.reshape(-1)[keep]
        flat_values = rollout.values.reshape(-1)[keep]
        flat_advantages = advantages.reshape(-1)[keep]
        flat_returns = returns.reshape(-1)[keep]
        total = int(flat_obs.shape[0])
        if total == 0:
            raise RuntimeError("PPO rollout contained no scheduled UEs")

        policy_loss = value_loss = entropy_value = 0.0
        approximate_kl = clip_fraction = 0.0
        epochs_run = 0
        for _epoch in range(config.ppo_epochs):
            permutation = torch.randperm(total, device=flat_obs.device)
            epoch_kl: list[float] = []
            for begin in range(0, total, config.ppo_minibatch_size):
                index = permutation[begin : begin + config.ppo_minibatch_size]
                minibatch_advantage = flat_advantages[index]
                if config.advantage_normalization and index.numel() > 1:
                    minibatch_advantage = (
                        minibatch_advantage - minibatch_advantage.mean()
                    ) / (minibatch_advantage.std() + 1e-8)

                new_log_prob, entropy, value = self.policy.evaluate_actions(
                    flat_obs[index],
                    flat_actions[index],
                )
                log_ratio = new_log_prob - flat_log_probs[index]
                ratio = log_ratio.exp()
                unclipped = -minibatch_advantage * ratio
                clipped = -minibatch_advantage * ratio.clamp(
                    1.0 - config.clip_coef,
                    1.0 + config.clip_coef,
                )
                surrogate = torch.max(unclipped, clipped).mean()

                target_value = flat_returns[index]
                if config.value_clip_coef > 0.0:
                    old_value = flat_values[index]
                    clamped_value = old_value + (value - old_value).clamp(
                        -config.value_clip_coef,
                        config.value_clip_coef,
                    )
                    critic_loss = 0.5 * torch.max(
                        (value - target_value).square(),
                        (clamped_value - target_value).square(),
                    ).mean()
                else:
                    critic_loss = 0.5 * (value - target_value).square().mean()

                entropy_bonus = entropy.mean()
                loss = (
                    surrogate
                    + config.value_loss_coef * critic_loss
                    - config.entropy_coef * entropy_bonus
                )
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.policy.parameters(),
                    config.max_grad_norm,
                )
                self.optimizer.step()

                with torch.no_grad():
                    kl = ((ratio - 1.0) - log_ratio).mean()
                    clipped_fraction = (
                        (ratio - 1.0).abs() > config.clip_coef
                    ).to(torch.float32).mean()
                epoch_kl.append(float(kl))
                policy_loss = float(surrogate.detach())
                value_loss = float(critic_loss.detach())
                entropy_value = float(entropy_bonus.detach())
                approximate_kl = float(kl)
                clip_fraction = float(clipped_fraction)
            epochs_run += 1
            mean_kl = sum(epoch_kl) / max(len(epoch_kl), 1)
            if config.target_kl is not None and mean_kl > config.target_kl:
                break

        return PPOMetrics(
            update=update,
            policy_loss=policy_loss,
            value_loss=value_loss,
            entropy=entropy_value,
            approximate_kl=approximate_kl,
            clip_fraction=clip_fraction,
            mean_reward=float(rollout.rewards[rollout.active].mean())
            if bool(rollout.active.any())
            else 0.0,
            mean_cell_spectral_eff=rollout.mean_cell_spectral_eff,
            mean_bler=rollout.mean_bler,
            epochs_run=epochs_run,
        )

    # ------------------------------------------------------------------
    # Advantages and returns
    # ------------------------------------------------------------------

    def _normalize_rewards(self, rollout: LaRolloutBatch) -> torch.Tensor:
        if not self.training_config.reward_normalization:
            return rollout.rewards
        for step in range(rollout.rewards.shape[0]):
            self.reward_normalizer.update(
                rollout.rewards[step],
                rollout.dones[step],
            )
        return self.reward_normalizer.normalize(rollout.rewards)

    def _generalized_advantage(
        self,
        rollout: LaRolloutBatch,
        rewards: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        config = self.training_config
        horizon = rewards.shape[0]
        advantages = torch.zeros_like(rewards)
        running = torch.zeros_like(rewards[0])
        next_value = rollout.last_value
        for step in reversed(range(horizon)):
            not_done = (~rollout.dones[step]).to(rewards.dtype).unsqueeze(-1)
            delta = (
                rewards[step]
                + config.gamma * next_value * not_done
                - rollout.values[step]
            )
            running = delta + config.gamma * config.gae_lambda * not_done * running
            advantages[step] = running
            next_value = rollout.values[step]
        return advantages, advantages + rollout.values

    def _discounted_returns(self, rollout: LaRolloutBatch) -> torch.Tensor:
        rewards = self._normalize_rewards(rollout)
        gamma = self.training_config.gamma
        returns = torch.zeros_like(rewards)
        running = rollout.last_value
        for step in reversed(range(rewards.shape[0])):
            not_done = (~rollout.dones[step]).to(rewards.dtype).unsqueeze(-1)
            running = rewards[step] + gamma * running * not_done
            returns[step] = running
        return returns

    # ------------------------------------------------------------------
    # Evaluation and selection
    # ------------------------------------------------------------------

    def student_action(
        self,
        env: LinkAdaptationEnv,
        observation: torch.Tensor,
    ) -> torch.Tensor:
        """Deterministic MCS from the current policy."""

        with torch.no_grad():
            logits, _value = self.policy(observation)
        return logits.argmax(dim=-1)

    @torch.no_grad()
    def rollout_statistics(
        self,
        env: LinkAdaptationEnv,
        *,
        steps: int,
        action_fn: ActionFn,
        warmup_slots: int = 0,
        reference_fn: ActionFn | None = None,
        seed: int | None = None,
    ) -> dict[str, float]:
        """Average KPIs over a rollout, optionally matched against a reference.

        Re-seeding is what makes several schemes comparable. With the seed
        restored, every scheme replays the same radio conditions and the same
        error draws, and the only difference left is the MCS decision. This
        relies on the host simulator honouring the ``seed`` argument of
        ``reset``; an adapter that ignores it still trains, but its comparisons
        are noise.

        Stepping goes through ``step_eval``, which does not autoreset. A carrier
        with a finite ``episode_length_slots`` would otherwise re-randomize the
        simulator part-way through, unseeded, and the rollouts being compared
        would diverge from that slot onward.
        """

        was_training = self.policy.training
        self.policy.eval()
        try:
            observation = env.reset(seed=seed)
            for _ in range(int(warmup_slots)):
                observation, _r, _d, _i = env.step_eval(
                    action_fn(env, observation.features)
                )
            expected: list[float] = []
            realized: list[float] = []
            bler: list[float] = []
            mcs: list[float] = []
            matches: list[float] = []
            for _ in range(int(steps)):
                action = action_fn(env, observation.features)
                if reference_fn is not None and action is not None:
                    reference = reference_fn(env, observation.features)
                    active = observation.active
                    if bool(active.any()) and reference is not None:
                        matches.append(
                            float(
                                (action[active] == reference[active])
                                .to(torch.float32)
                                .mean()
                            )
                        )
                observation, _reward, _done, info = env.step_eval(action)
                expected.append(float(info["expected_cell_spectral_eff"].mean()))
                realized.append(float(info["realized_cell_spectral_eff"].mean()))
                bler.append(float(info["mean_expected_bler"].mean()))
                mcs.append(float(info["mean_mcs"].mean()))
        finally:
            self.policy.train(was_training)
        count = max(len(expected), 1)
        return {
            "expected_cell_spectral_eff": sum(expected) / count,
            "realized_cell_spectral_eff": sum(realized) / count,
            "mean_bler": sum(bler) / count,
            "mean_mcs": sum(mcs) / count,
            "action_match": sum(matches) / len(matches) if matches else 0.0,
        }

    def _evaluation_env(self) -> LinkAdaptationEnv:
        """Held-out environment, built once and re-seeded per rollout."""

        if self._validation_env is None:
            self._validation_env = self._make_env()
        return self._validation_env

    def evaluate(self, *, update: int, phase: str) -> LaValidationMetrics:
        """Score the student against the teacher, the outer loop and the genie."""

        config = self.training_config
        env = self._evaluation_env()
        steps = config.validation_steps
        warmup = config.validation_warmup_slots
        seed = config.validation_seed

        student = self.rollout_statistics(
            env,
            steps=steps,
            action_fn=self.student_action,
            warmup_slots=warmup,
            reference_fn=lambda e, _o: e.teacher_mcs(),
            seed=seed,
        )
        teacher = self.rollout_statistics(
            env,
            steps=steps,
            action_fn=lambda e, _o: e.teacher_mcs(),
            warmup_slots=warmup,
            seed=seed,
        )
        olla = self.rollout_statistics(
            env,
            steps=steps,
            action_fn=lambda _e, _o: None,
            warmup_slots=warmup,
            seed=seed,
        )
        genie = self.rollout_statistics(
            env,
            steps=steps,
            action_fn=lambda e, _o: e.genie_greedy_mcs(),
            warmup_slots=warmup,
            seed=seed,
        )
        # Inner loop alone, to separate what the calibration achieves from what
        # the outer loop absorbs.
        illa = self.rollout_statistics(
            env,
            steps=steps,
            action_fn=lambda e, _o: e.illa_mcs(),
            warmup_slots=warmup,
            seed=seed,
        )
        return LaValidationMetrics(
            update=update,
            phase=phase,
            steps=steps,
            seed=seed,
            student_cell_spectral_eff=student["expected_cell_spectral_eff"],
            teacher_cell_spectral_eff=teacher["expected_cell_spectral_eff"],
            olla_cell_spectral_eff=olla["expected_cell_spectral_eff"],
            genie_cell_spectral_eff=genie["expected_cell_spectral_eff"],
            student_teacher_ratio=_safe_ratio(
                student["expected_cell_spectral_eff"],
                teacher["expected_cell_spectral_eff"],
            ),
            student_olla_ratio=_safe_ratio(
                student["expected_cell_spectral_eff"],
                olla["expected_cell_spectral_eff"],
            ),
            illa_cell_spectral_eff=illa["expected_cell_spectral_eff"],
            student_bler=student["mean_bler"],
            teacher_bler=teacher["mean_bler"],
            olla_bler=olla["mean_bler"],
            illa_bler=illa["mean_bler"],
            student_mean_mcs=student["mean_mcs"],
            teacher_mean_mcs=teacher["mean_mcs"],
            teacher_action_match=student["action_match"],
        )

    def select_best_checkpoint(
        self,
        pattern: str,
        *,
        phase: str,
    ) -> Path | None:
        """Replay every phase checkpoint and keep the best cell SE.

        Training loss is a poor proxy here: cloning loss keeps falling while
        the closed-loop BLER window drifts, so the checkpoint with the lowest
        loss is routinely not the one with the best delivered throughput.
        """

        config = self.training_config
        steps = config.checkpoint_selection_steps
        if steps <= 0:
            return None
        candidates = sorted(self.output_dir.glob(pattern))
        if not candidates:
            return None

        seed = (
            config.checkpoint_selection_seed
            if config.checkpoint_selection_seed is not None
            else config.validation_seed
        )
        env = self._evaluation_env()
        metric = config.checkpoint_selection_metric
        selection_path = self.output_dir / f"{phase}_checkpoint_selection.csv"

        scored: list[tuple[float, int, Path, dict[str, float]]] = []
        for path in candidates:
            payload = _load_checkpoint_file(path, self.device)
            self.policy.load_state_dict(payload["policy"])
            stats = self.rollout_statistics(
                env,
                steps=steps,
                action_fn=self.student_action,
                warmup_slots=config.validation_warmup_slots,
                seed=seed,
            )
            update = int(payload.get("update", 0))
            scored.append((stats[metric], update, path, stats))
            print(
                f"la_checkpoint_selection,{phase},{update},"
                f"expected_cell_se,{stats['expected_cell_spectral_eff']:.4f},"
                f"realized_cell_se,{stats['realized_cell_spectral_eff']:.4f},"
                f"mean_bler,{stats['mean_bler']:.4f}",
                flush=True,
            )

        # Ties resolve toward the later checkpoint, which has trained longer.
        _best_score, _best_update, best_path = max(
            ((score, update, path) for score, update, path, _ in scored),
            key=lambda item: (item[0], item[1]),
        )
        for _score, update, path, stats in scored:
            _write_csv_row(
                selection_path,
                asdict(
                    LaCheckpointSelectionMetrics(
                        phase=phase,
                        update=update,
                        steps=steps,
                        seed=int(seed),
                        expected_cell_spectral_eff=stats[
                            "expected_cell_spectral_eff"
                        ],
                        realized_cell_spectral_eff=stats[
                            "realized_cell_spectral_eff"
                        ],
                        mean_bler=stats["mean_bler"],
                        selected=int(path == best_path),
                    )
                ),
            )

        payload = _load_checkpoint_file(best_path, self.device)
        self.policy.load_state_dict(payload["policy"])
        selected = self.output_dir / f"checkpoint_{phase}_selected.pt"
        self.save_checkpoint(selected, int(payload.get("update", 0)))
        return selected

    # ------------------------------------------------------------------
    # Checkpointing and bookkeeping
    # ------------------------------------------------------------------

    def save_checkpoint(self, path: str | Path, update: int) -> Path:
        """Atomically save policy, optimizer, RNG and resolved configuration.

        Written to a temporary file and then renamed, so an interrupted save
        leaves the previous checkpoint intact rather than a truncated one.
        That matters because :meth:`select_best_checkpoint` loads every file
        matching its glob.

        Args:
            path (str | Path): Destination. Parent directories are created.
            update (int): Update number to record in the payload, returned by
                :meth:`load_checkpoint` and used to order candidates during
                between-phase selection.

        Returns:
            Path: The written checkpoint path, i.e. ``path`` as a
            :class:`~pathlib.Path`.

        Raises:
            OSError: If the destination cannot be created or written, for
                example on a full or read-only filesystem.

        Examples:
            Save on a phase boundary and keep the path for later::

                latest = framework.save_checkpoint("runs/drl_la/latest.pt", 2000)
                framework.load_checkpoint(latest)
        """

        checkpoint_path = Path(path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
        torch.save(self._checkpoint_payload(update), temporary)
        temporary.replace(checkpoint_path)
        return checkpoint_path

    def load_checkpoint(self, path: str | Path) -> int:
        """Resume policy, optimizer, phase counters and RNG state.

        The simulator is *not* resumed: it restarts from the training seed, so
        this reproduces the saved policy rather than the interrupted
        trajectory. What restoring the RNG buys is that two resumes from one
        checkpoint agree, which they otherwise would not.

        Optimizer state is restored on a best-effort basis. Each phase rebuilds
        the optimizer over a different parameter set, so a checkpoint saved in
        one phase cannot hand its moments to another; that mismatch is
        swallowed and everything else still loads.

        Args:
            path (str | Path): Checkpoint written by :meth:`save_checkpoint`.
                Read with pickle execution disabled, so it must contain only
                tensors and plain Python containers.

        Returns:
            int: The ``update`` number the checkpoint was saved at. The phase
            counters are restored onto the instance separately, so this is a
            convenience for logging rather than the resume point itself.

        Raises:
            FileNotFoundError: If ``path`` does not exist.
            ValueError: If the payload's ``checkpoint_version`` does not equal
                :attr:`CHECKPOINT_VERSION`.
            KeyError: If the payload carries no ``policy`` entry.
            RuntimeError: If the stored ``policy`` does not fit the current
                architecture, which happens when the run was configured with a
                different :class:`~drlTrainingFramework.config.LaModelConfig`
                or against a carrier with a different PRB group count.
            pickle.UnpicklingError: If the file holds anything outside the
                safe-load allowlist. This is what stops a hostile checkpoint
                from executing code while being read.

        Examples:
            Resume explicitly, then continue the curriculum::

                framework = LinkAdaptationTrainingFramework(env_factory)
                update = framework.load_checkpoint("runs/drl_la/latest.pt")
                print(f"resuming from update {update}")
                framework.train()

            Or resume declaratively, which is what the launcher's ``--resume``
            flag sets. The constructor calls this method itself, after the
            policy is built::

                LaTrainingConfig(resume_checkpoint="runs/drl_la/latest.pt")
        """

        checkpoint = _load_checkpoint_file(path, self.device)
        version = int(checkpoint.get("checkpoint_version", -1))
        if version != self.CHECKPOINT_VERSION:
            raise ValueError(
                "unsupported link adaptation checkpoint version "
                f"{version}; expected {self.CHECKPOINT_VERSION}"
            )
        self.policy.load_state_dict(checkpoint["policy"])
        optimizer_state = checkpoint.get("optimizer")
        if optimizer_state is not None:
            try:
                self.optimizer.load_state_dict(optimizer_state)
            except ValueError:
                # Each phase rebuilds the optimizer over a different parameter
                # set, so a cross-phase resume legitimately cannot reuse it.
                pass
        self.phase = str(checkpoint.get("phase", self.phase))
        self.completed_clone_updates = int(
            checkpoint.get("completed_clone_updates", 0)
        )
        self.completed_bootstrap_updates = int(
            checkpoint.get("completed_bootstrap_updates", 0)
        )
        self.completed_critic_updates = int(
            checkpoint.get("completed_critic_updates", 0)
        )
        self.completed_ppo_updates = int(
            checkpoint.get("completed_ppo_updates", 0)
        )
        normalizer = checkpoint.get("reward_normalizer")
        if isinstance(normalizer, Mapping):
            self.reward_normalizer.load_state_dict(normalizer)
        self._restore_rng_state(checkpoint)
        return int(checkpoint.get("update", 0))

    def _restore_rng_state(self, checkpoint: Mapping[str, Any]) -> None:
        """Put the global generators back where the checkpoint left them.

        Without this, action sampling in :meth:`collect_rollout`, the replay
        draws and the PPO minibatch permutation all continue from wherever the
        process happened to be, so resuming twice from one checkpoint gives two
        different runs.

        The states are stored as tensors, so they arrive on ``map_location``'s
        device, while both setters require a CPU ``uint8`` tensor; hence the
        conversion. The CUDA generators matter as much as the CPU one here,
        because a policy on a CUDA device samples from them rather than from
        the CPU generator.
        """

        def to_cpu_bytes(state: torch.Tensor) -> torch.Tensor:
            return state.detach().to(device="cpu", dtype=torch.uint8)

        state = checkpoint.get("torch_rng_state")
        if isinstance(state, torch.Tensor):
            torch.set_rng_state(to_cpu_bytes(state))

        cuda_states = checkpoint.get("cuda_rng_state")
        if not torch.cuda.is_available() or not isinstance(cuda_states, (list, tuple)):
            return
        if len(cuda_states) != torch.cuda.device_count():
            # Saved on a machine with a different GPU count. Say so rather than
            # skipping quietly, because silent nondeterminism is the exact
            # failure this method exists to prevent.
            print(
                "la_resume_warning,cuda_rng_state_not_restored,saved_devices,"
                f"{len(cuda_states)},present_devices,{torch.cuda.device_count()}",
                flush=True,
            )
            return
        torch.cuda.set_rng_state_all([to_cpu_bytes(one) for one in cuda_states])

    def _checkpoint_payload(self, update: int) -> dict[str, Any]:
        return {
            "checkpoint_version": self.CHECKPOINT_VERSION,
            "update": int(update),
            "phase": self.phase,
            "completed_clone_updates": self.completed_clone_updates,
            "completed_bootstrap_updates": self.completed_bootstrap_updates,
            "completed_critic_updates": self.completed_critic_updates,
            "completed_ppo_updates": self.completed_ppo_updates,
            "policy": self.policy.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "reward_normalizer": self.reward_normalizer.state_dict(),
            "carrier": _carrier_dict(self.env.carrier),
            "env_config": _config_dict(self.env_config),
            "model_config": _config_dict(self.model_config),
            "training_config": _config_dict(self.training_config),
            "observation_dim": self.env.obs_dim,
            "action_dim": self.env.action_dim,
            # Both generator families are captured: a policy on a CUDA device
            # samples actions from the CUDA generators, not the CPU one.
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": (
                torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
            ),
        }

    def _write_run_metadata(self) -> None:
        parameters = sum(
            parameter.numel() for parameter in self.policy.parameters()
        )
        metadata = {
            "carrier": _carrier_dict(self.env.carrier),
            "env_config": _config_dict(self.env_config),
            "model_config": _config_dict(self.model_config),
            "training_config": _config_dict(self.training_config),
            "observation_dim": self.env.obs_dim,
            "action_dim": self.env.action_dim,
            "policy_parameters": int(parameters),
            "device": str(self.device),
        }
        path = self.output_dir / "run_config.yaml"
        with path.open("w", encoding="utf-8") as stream:
            yaml.safe_dump(metadata, stream, sort_keys=True)

    def _set_optimizer(
        self,
        parameter_groups: Sequence[Mapping[str, Any]],
        learning_rate: float,
    ) -> None:
        self.optimizer = torch.optim.AdamW(
            list(parameter_groups),
            lr=float(learning_rate),
        )

    def _should_log(self, update: int, start: int, target: int) -> bool:
        every = self.training_config.log_every
        return (
            update == start
            or update == target
            or (every > 0 and update % every == 0)
        )

    def _maybe_checkpoint(self, update: int, target: int, phase: str) -> None:
        cadence = self.training_config.phase_checkpoint_every(phase)
        if update != target and not (cadence > 0 and update % cadence == 0):
            return
        self.save_checkpoint(
            self.output_dir / f"checkpoint_{phase}_{update:06d}.pt",
            update,
        )
        self.save_checkpoint(self.latest_checkpoint_path, update)

    def _maybe_validate(self, update: int, target: int, phase: str) -> None:
        every = self.training_config.validation_every
        if every <= 0:
            return
        if update != target and update % every != 0:
            return
        metrics = self.evaluate(update=update, phase=phase)
        _write_csv_row(self.validation_metrics_path, asdict(metrics))
        print(
            f"la_validation,{phase},{update},"
            f"student_cell_se,{metrics.student_cell_spectral_eff:.4f},"
            f"teacher_cell_se,{metrics.teacher_cell_spectral_eff:.4f},"
            f"olla_cell_se,{metrics.olla_cell_spectral_eff:.4f},"
            f"illa_cell_se,{metrics.illa_cell_spectral_eff:.4f},"
            f"genie_cell_se,{metrics.genie_cell_spectral_eff:.4f},"
            f"teacher_ratio,{metrics.student_teacher_ratio:.4f},"
            f"olla_ratio,{metrics.student_olla_ratio:.4f},"
            f"student_bler,{metrics.student_bler:.4f},"
            f"action_match,{metrics.teacher_action_match:.4f}",
            flush=True,
        )


def _explained_variance(
    predictions: torch.Tensor | None,
    targets: torch.Tensor | None,
) -> float:
    if predictions is None or targets is None or targets.numel() < 2:
        return 0.0
    target_variance = float(targets.var())
    if target_variance <= 0.0:
        return 0.0
    return 1.0 - float((targets - predictions).var()) / target_variance

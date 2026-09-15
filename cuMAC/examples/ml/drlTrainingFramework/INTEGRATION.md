<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Integrating the DRL link adaptation training framework

This package trains a deep reinforcement learning MCS selector for **full-buffer
downlink traffic** and is designed to sit on top of an existing system-level
simulator (SLS). The SLS keeps owning the radio and the scheduler; the framework
owns the MCS decision and everything downstream of it.

Integration is one interface. Implement
[`LinkAdaptationSimulator`](adapter.py), which publishes a `SlotContext` once per
slot and consumes a `SlotOutcome`, and the four-phase curriculum runs unchanged.

---

## 1. Division of responsibility

| Owned by your SLS | Owned by this framework |
|---|---|
| Deployment, mobility, channel model | Observation construction |
| CSI/SRS feedback modelling and ageing | Inner loop (CQI → co-scheduled operating point) |
| Resource allocation, MU-MIMO co-scheduling | Outer loop (OLLA offset per UE) |
| Precoder design and the link budget it predicts | EESM/BLER scoring of a candidate MCS |
| The realized post-precoder SINR | Teacher labels, reward, sliding BLER window |
| PHY abstraction (EESM, BLER curves, CQI table) | ACK sampling, KPI aggregation |
| Per-UE throughput / PF weight accounting | Cloning, bootstrap, critic warmup, PPO |

The framework never asks the SLS to choose an MCS, and never chooses a UE, a
PRB group or a co-scheduling set itself. That boundary is what makes the package
portable.

## 2. Slot sequence

```text
                    ┌──────────────── your SLS ────────────────┐
reset(seed) ───────►│ drop, channel, CSI, schedule, precode    │
                    └──────────────────┬───────────────────────┘
                                       │ SlotContext
                    ┌──────────────────▼───────────────────────┐
                    │ observation → policy → one MCS per UE    │
                    │ EESM → BLER → ACK draw → reward, KPIs    │  framework
                    │ sliding BLER window, OLLA offset update  │
                    └──────────────────┬───────────────────────┘
                                       │ SlotOutcome
                    ┌──────────────────▼───────────────────────┐
advance(outcome) ──►│ PF weights, HARQ, age channel, reschedule│
                    └──────────────────────────────────────────┘
```

The split matters: the agent observes the resources and the co-scheduling it was
actually given **before** committing to a modulation.

## 3. Axis conventions

| Axis | Meaning |
|---|---|
| `env` | Independent parallel simulation instances, batched for throughput. Use `1` if your SLS is not vectorized. |
| `prbg` | PRB groups across the carrier, the granularity of allocation and precoding. |
| `layer` | Spatial layer slot within one PRB group, of width `max_coscheduled_layers`. Owned by at most one UE; one UE may own several. |
| `ue` | Physical UE index, stable across PRB groups and slots. The action axis, so a UE keeps its policy row for life. |

Non-contiguous (type-0) allocation is the general case. A UE may hold
non-contiguous PRB groups and be co-scheduled with a different set of UEs on
each, so its layer count and the total co-scheduled layer count are **per-PRBG**
observation features, while the transport block spans the whole allocation
through a single EESM aggregation.

## 4. What to implement

### 4.1 `carrier` — `CarrierProfile`, published once

Fixed for the lifetime of the simulator instance.

| Field | Meaning |
|---|---|
| `num_envs`, `num_ues`, `num_prb_groups` | Batch and carrier sizes. |
| `max_coscheduled_layers` | Width of the layer axis, i.e. the most layers you ever co-schedule on one PRB group. Also the feature normalizer. |
| `max_layers_per_ue` | Largest rank a single UE can be given. |
| `num_tx_ant` | Width of `effective_channel_est`. |
| `prbg_bandwidth_hz`, `bandwidth_hz` | Transport-block rate is `spectral_eff * resource_count * prbg_bandwidth_hz`; `bandwidth_hz` is the cell-SE denominator. |
| `power_per_prbg_watt`, `noise_per_prbg_watt` | Per-PRB-group transmit power before the layer split, and receiver noise. |
| `device` | Device every published tensor lives on. The policy is placed here too. |
| `csi_report_period_slots` | Feedback ages are normalized by this, so a normalized age of 1.0 means "one report old". |
| `channel_est_error_nmse_db` | Your own sounding quality, in dB. The inner loop uses it to price the interference the precoder's nulls will miss. |
| `episode_length_slots` | Slots before the framework resets. `None` never terminates, which is the natural choice for full buffer with a bandit discount. |

### 4.2 `reset(seed)` / `advance(outcome)` — `SlotContext`, published per slot

Per-resource tensors are `[env, prbg, layer]` and are read only where
`stream_valid` is true.

| Field | Shape | Meaning |
|---|---|---|
| `ue_index` | `[E,G,L]` int64 | Physical UE owning that layer slot. |
| `stream_valid` | `[E,G,L]` bool | **Your scheduling and MU-MIMO decision.** The framework treats it as given. |
| `estimated_signal_watt` | `[E,G,L]` | Post-precoder desired power your scheduler *predicts*, from the estimate the precoder was solved from. |
| `estimated_residual_interference_watt` | `[E,G,L]` | Intra-cell interference the precoder predicts it fails to null. Exactly zero on a single-layer PRB group. |
| `power_per_layer_watt` | `[E,G,L]` | Power assigned to that layer, after the split. |
| `effective_channel_est` | `[E,G,L,T]` complex | Estimated effective channel row (combiner applied to the estimated channel). Only its direction is used, so any consistent scaling works. |
| `realized_sinr_db` | `[E,G,L]` | SINR the layer *actually* experiences. **Teacher label only** — never enters an observation. |
| `wideband_cqi` | `[E,U]` int | Last reported CQI index. Single-user, full-power. |
| `reported_rank` | `[E,U]` | Last reported rank, in layers. |
| `csi_age_slots`, `srs_age_slots`, `rank_age_slots` | `[E,U]` | Slots since each report was refreshed. Equal when refreshed together. |
| `csi_channel_energy` | `[E,U,G]` | `‖H_est‖_F²` for every candidate UE on every PRB group. Must come from the same estimate the reported CQI was measured on. |
| `estimated_ici_watt` | `[E,U,G]` | Inter-cell interference the UE reported. Zero for a single-cell study. |
| `retransmitting` | `[E,U]` bool, optional | Rows replaying a first attempt's modulation. Scored with `retransmission_mcs`, excluded from every loss, earning no reward. `None` for a pure new-transmission study. |
| `retransmission_mcs` | `[E,U]` int64 | Required wherever `retransmitting` is true: the MCS the first attempt chose. The framework does **not** model HARQ soft-combining gain, so publish the post-combining SINR on those layers if you model it. |

Two consistency requirements are worth calling out, because getting them wrong
degrades training silently rather than raising:

1. **`csi_channel_energy` must be the energy behind `wideband_cqi`.** The inner
   loop forms the ratio of the predicted co-scheduled link budget to the
   single-user reference `power_per_prbg_watt * csi_channel_energy /
   (noise + ici)`. That ratio is what cancels the estimate's own bias and leaves
   a pure single-user-to-co-scheduled correction factor.
2. **`realized_sinr_db` must be genuinely worse than the estimate-derived
   prediction**, by the channel ageing and the missed nulls an estimate
   structurally cannot see. That gap is exactly what the outer loop exists to
   absorb and what makes the teacher worth cloning. If your SLS reports the
   realized SINR as equal to the predicted one, the teacher and the inner loop
   collapse onto each other and there is nothing to learn.

Call `context.validate(carrier)` from your own tests. A shape or device mismatch
then surfaces with the offending field named, instead of thousands of slots later
inside a scatter or a matrix product.

### 4.3 `phy` — `PhyAbstraction`

Return your calibrated link-to-system mapping, implementing four methods:
`mcs_spectral_eff`, `cqi_to_sinr_db`, `eesm_effective_sinr_all_mcs` and
`bler_from_sinr_mcs`. See [`phy.py`](phy.py) for exact signatures and semantics.

`DEFAULT_PHY` is a **reference model**, there so the package runs standalone. Its
3GPP MCS and CQI tables are exact; its required-SINR, EESM-beta and waterfall
values come from a documented analytic model and are not measurements. Use it to
bring the integration up, then substitute your own before trusting any absolute
BLER or throughput number.

### 4.4 `SlotOutcome` — what you get back

`slot`, `scheduled`, `mcs`, `expected_bler`, `ack`, `resource_count`,
`raw_rate_bps`, `realized_rate_bps`, all `[env, ue]`. Use it in `advance()` to
update PF weights, HARQ and your own KPI collection. The framework draws the ACK
itself, from its own generator, so a rollout is reproducible from the training
seed alone.

### 4.5 Honour the seed

`reset(seed=...)` must re-seed your simulator's randomness. The framework scores
the student, the teacher, the outer loop, the inner loop and the genie by
replaying each on the same seed, so the only difference left between them is the
MCS decision. An adapter that ignores `seed` still trains, but every reported
comparison becomes noise.

### 4.6 Full-buffer traffic

Configure your SLS so every UE is continuously backlogged. The framework assumes
a transport block always carries the rate its MCS and resource count imply; no
queue state, buffer occupancy or admission decision enters the observation or the
reward. If you want to model a finite buffer later, that changes the reward
definition, not the interface.

## 5. Minimal adapter skeleton

```python
from drlTrainingFramework import CarrierProfile, SlotContext, SlotOutcome

class MySlsAdapter:
    def __init__(self, sls, *, seed=None):
        self._sls = sls
        self._carrier = CarrierProfile(
            num_envs=1,
            num_ues=sls.num_ues,
            num_prb_groups=sls.num_prb_groups,
            max_coscheduled_layers=sls.max_layers_per_prbg,
            max_layers_per_ue=sls.max_rank,
            num_tx_ant=sls.num_tx_ant,
            prbg_bandwidth_hz=sls.prbg_bandwidth_hz,
            bandwidth_hz=sls.carrier_bandwidth_hz,
            power_per_prbg_watt=sls.power_per_prbg_watt,
            noise_per_prbg_watt=sls.noise_per_prbg_watt,
            device=torch.device("cuda:0"),
            csi_report_period_slots=sls.csi_period,
            channel_est_error_nmse_db=sls.srs_nmse_db,
        )

    @property
    def carrier(self):
        return self._carrier

    @property
    def phy(self):
        return self._phy          # your PhyAbstraction implementation

    def reset(self, *, seed=None):
        self._sls.reset(seed=seed)
        return self._publish()

    def advance(self, outcome: SlotOutcome):
        self._sls.report(
            mcs=outcome.mcs,
            ack=outcome.ack,
            delivered_bps=outcome.realized_rate_bps,
        )
        self._sls.step_slot()      # age channel, refresh CSI, reschedule
        return self._publish()

    def _publish(self) -> SlotContext:
        ...                        # fill in the table in section 4.2
```

Read [`example_simulator.py`](example_simulator.py) alongside this. It is a
**plumbing test double, not a channel model**, but it is a complete, readable
implementation of every field, and it is what the commands below run against.

## 6. Running it

```bash
pip install -r drlTrainingFramework/requirements.txt

# 1. Smoke-test the framework on the bundled test double (no SLS needed).
python drlTrainingFramework/train_drl_la.py --clone-updates 50 --output-dir runs/smoke

# 2. Same thing against your adapter.
python drlTrainingFramework/train_drl_la.py \
    --simulator my_sls.drl_la_adapter:build_simulator \
    --sim-config my_sls/full_buffer.yaml \
    --clone-updates 50 --output-dir runs/smoke_sls

# 3. The full curriculum. This is the one that produces a usable policy.
python drlTrainingFramework/train_drl_la.py \
    --simulator my_sls.drl_la_adapter:build_simulator \
    --sim-config my_sls/full_buffer.yaml \
    --training-config drlTrainingFramework/config/la_staged_training_config.yaml \
    --output-dir runs/drl_la

# 4. Score the selected checkpoint against every reference scheme.
python drlTrainingFramework/evaluate_drl_la.py \
    --simulator my_sls.drl_la_adapter:build_simulator \
    --sim-config my_sls/full_buffer.yaml \
    --checkpoint runs/drl_la/checkpoint_ppo_selected.pt \
    --steps 500 --warmup-slots 100
```

`--simulator` takes `module:function`; the factory must accept

```python
def build_simulator(config_path=None, *, seed=None, device=None, **overrides): ...
```

and may ignore any argument. Nothing else in the framework names a simulator, so
attaching a different one never edits framework code. If you would rather drive
the framework from your own script, build an `EnvFactory` directly:

```python
from drlTrainingFramework import (
    LinkAdaptationEnv, LinkAdaptationTrainingFramework, LaEnvConfig,
)

framework = LinkAdaptationTrainingFramework(
    lambda env_config: LinkAdaptationEnv(MySlsAdapter(build_sls()), env_config),
    LaEnvConfig.from_yaml("drlTrainingFramework/config/la_env_config.yaml"),
    ...,
)
framework.train()
```

Any field of any configuration can be overridden without editing YAML, using
`--set env.target_bler=0.05`, `--set model.encoder_layers=[128,128]`,
`--set train.ppo_updates=25`.

## 7. The training curriculum

Four phases, each removing a specific failure of the next.

| Phase | What it does | Why it is needed |
|---|---|---|
| 1. Teacher cloning | Regress the actor onto the perfect-information debt-SE-aware selector, with Gaussian soft labels over the MCS index. | A uniform policy over 28 MCS never survives contact with a BLER target. |
| 2. Actor bootstrap *(off by default)* | Short REINFORCE pass on the cloned actor. | Imitation is evaluated on the teacher's state distribution, not the actor's own. |
| 3. Critic warmup | Fit the value head with encoder and actor frozen. | A random value head injects pure noise into the first PPO advantages. |
| 4. PPO fine-tuning | Optimize the clipped objective over the whole policy. | Only now is the full objective safe to optimize. |

Between the phases that change the actor, every checkpoint is replayed on a
held-out rollout and the best cell spectral efficiency is carried forward, so a
phase can never hand its successor a worse policy than one it already had.

**Use `la_staged_training_config.yaml`.** It selects `training_mode: full` and
inherits everything else from `la_training_config.yaml`; training with the base
config alone stops after cloning. A cloned actor is an initialization, not a
deployable policy: its labels come from a teacher that reads the realized
channel while its own inputs are built from the estimate, so imitating it teaches
an aggression the student has no information to justify. Closed-loop BLER lands
well above target despite high action agreement, and PPO is what closes that gap.

`gamma` defaults to `0`, making phase 4 a contextual bandit. The MCS chosen for
one UE in one slot does not move the next state: the channel, the allocation and
the scheduler's weights all evolve independently of it. The one cross-slot
coupling, the sliding BLER window, is already priced inside the reward.

## 8. Reading the results

Everything lands in `--output-dir`: `run_config.yaml` (resolved configuration and
the carrier it trained against), per-phase metric CSVs, `validation_metrics.csv`,
`<phase>_checkpoint_selection.csv`, and the checkpoints.

The validation line to watch:

```text
la_validation,<phase>,<update>,student_cell_se,…,teacher_cell_se,…,olla_cell_se,…,
    illa_cell_se,…,genie_cell_se,…,teacher_ratio,…,olla_ratio,…,student_bler,…,
    action_match,…
```

| Scheme | What it is |
|---|---|
| `student` | The trained policy, acting greedily. |
| `olla` | Inner loop plus outer loop. **The deployable baseline to beat**, so `olla_ratio > 1` is the headline result. |
| `illa` | Inner loop alone, separating what the CQI calibration achieves from what the outer loop absorbs. |
| `teacher` | Perfect-information selector. An upper reference, not a scheme; `teacher_ratio` is a distance-to-oracle, not a target of 1.0. |
| `genie` | Highest MCS meeting the BLER target with no BLER-history feedback. Bounds the comparison. |

Sanity checks when bringing up a new adapter:

- `illa_cell_se` should be *below* `olla_cell_se`, and `illa_bler` well above
  target. The inner loop knowingly ignores what it cannot see.
- `teacher_bler` should sit close to `target_bler`. If it does not, your
  `realized_sinr_db` and your PHY abstraction's BLER curves disagree.
- `genie_cell_se` and `teacher_cell_se` should be close. Either can be higher:
  the teacher runs above the genie while UEs are below target, because it
  maximizes goodput unconstrained there, and below it once they are in BLER
  debt and its bound tightens.

## 9. Tuning for your simulator

| Setting | Where | Note |
|---|---|---|
| `target_bler`, `bler_cushion`, `debt_bler_bound` | `la_env_config.yaml` | Match the operating point your own outer loop is tuned for before comparing against it. |
| `illa_calibration` | `la_env_config.yaml` | Re-sweep the four-level ladder against your precoder. The default stops one level short of the full model on purpose: the last level double-counts missed nulls once the outer loop is also running. |
| `illa_leakage_floor_gain_db` | `la_env_config.yaml` | Precoder specific; only read at the `estimation_error_floor` level. |
| `clone_collect_every` | `la_training_config.yaml` | Slots per gradient step. Raise it in proportion to how expensive one of your slots is; the replay buffer decorrelates the batch. |
| `clone_updates` | `la_staged_training_config.yaml` | Cloning is initialization and should stop where `teacher_action_match` stops improving. Enable validation and measure it. |
| `encoder_layers` | `la_model_config.yaml` | The observation is `8 * num_prb_groups + 6` wide. The default encoder was sized for a much wider observation, so narrow it if your carrier has few PRB groups. |
| `num_envs` | your adapter | The single most effective throughput knob: batching independent instances makes wider slots close to free and keeps the replay buffer diverse. |

## 10. File map

| File | Role |
|---|---|
| `adapter.py` | **The interface you implement.** `CarrierProfile`, `SlotContext`, `SlotOutcome`, `LinkAdaptationSimulator`. |
| `phy.py` | `PhyAbstraction` protocol plus the reference NR model. |
| `env.py` | `LinkAdaptationEnv`: observation, MCS action, inner/outer loop, reward, KPIs. Simulator agnostic. |
| `features.py` | Per-PRBG and per-UE observation lanes. |
| `teacher.py` | Perfect-information labels and the soft-label targets used by cloning. |
| `model.py` | Actor-critic network and the observation normalizer that travels in the checkpoint. |
| `training.py` | The four-phase curriculum, validation, checkpoint selection. |
| `config.py`, `yaml_config.py` | Configuration dataclasses and the YAML loader (`base:` inheritance, unknown-key rejection). |
| `train_drl_la.py`, `evaluate_drl_la.py` | Launchers. |
| `launch.py` | `--simulator module:function` resolution, shared by both launchers. |
| `example_simulator.py` | Plumbing test double implementing the full contract. Not a channel model. |
| `config/*.yaml` | Environment, model, and base/staged training configurations. |

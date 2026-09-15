#!/usr/bin/env python3

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

"""Score a trained link adaptation policy against its reference schemes.

Five schemes are replayed on the same seed and warmed up identically, so the
comparison is matched slot for slot and any difference is the MCS decision and
nothing else:

* **student** - the trained policy, acting greedily.
* **olla** - inner loop plus outer loop, the deployable baseline to beat.
* **illa** - inner loop alone, to separate what the CQI calibration achieves
  from what the outer loop absorbs.
* **teacher** - the perfect-information debt-SE-aware selector the student was
  cloned from, an upper reference rather than a scheme.
* **genie** - highest MCS meeting the BLER target with no BLER-history
  feedback, bounding the whole comparison.

The matching relies on the host simulator honouring ``reset(seed=...)``.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path

# Runnable both as a plain script and as python -m drlTrainingFramework.evaluate_drl_la.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from drlTrainingFramework.config import (  # noqa: E402
    LA_OVERRIDE_SECTIONS,
    LaEnvConfig,
    LaModelConfig,
    LaTrainingConfig,
    parse_la_config_overrides,
)
from drlTrainingFramework.launch import (  # noqa: E402
    DEFAULT_CONFIG_DIR,
    add_simulator_arguments,
    build_env_factory,
    dump_yaml,
)
from drlTrainingFramework.training import (  # noqa: E402
    LinkAdaptationTrainingFramework,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a DRL link adaptation checkpoint against the OLLA and "
            "ILLA baselines, the perfect-information teacher and the genie "
            "upper reference."
        )
    )
    add_simulator_arguments(parser)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Checkpoint written by train_drl_la.py.",
    )
    parser.add_argument(
        "--env-config",
        type=Path,
        default=DEFAULT_CONFIG_DIR / "la_env_config.yaml",
        help="Link adaptation environment YAML.",
    )
    parser.add_argument(
        "--model-config",
        type=Path,
        default=DEFAULT_CONFIG_DIR / "la_model_config.yaml",
        help="Actor-critic YAML. Must match the trained architecture.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=500,
        help="Scored slots per scheme.",
    )
    parser.add_argument(
        "--warmup-slots",
        type=int,
        default=40,
        help=(
            "Unscored slots first. reset() zeroes every CSI age and restarts "
            "the outer loop at its initial offset, neither of which happens in "
            "operation; the warmup clears both transients."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=10_001,
        help="Seed for the held-out rollout, shared by every scheme.",
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        metavar="SECTION.FIELD=VALUE",
        action="append",
        default=[],
        help=(
            "Override any configuration field. Sections are "
            f"{', '.join(sorted(LA_OVERRIDE_SECTIONS))}. Repeatable."
        ),
    )
    parser.add_argument(
        "--device",
        help="Torch device for the policy and the simulator, e.g. cuda:0.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Write the comparison to this YAML file as well as stdout.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        explicit = parse_la_config_overrides(args.overrides)
    except ValueError as error:
        raise SystemExit(f"invalid --set override: {error}") from error

    env_config = LaEnvConfig.from_yaml(args.env_config, overrides=explicit["env"])
    model_config = LaModelConfig.from_yaml(
        args.model_config,
        overrides=explicit["model"],
    )
    training_config = LaTrainingConfig(
        training_mode="clone",
        clone_updates=0,
        output_dir=str(args.checkpoint.parent),
        validation_steps=args.steps,
        validation_seed=args.seed,
        validation_warmup_slots=args.warmup_slots,
    )

    framework = LinkAdaptationTrainingFramework(
        build_env_factory(
            args,
            seed=args.seed,
            device=args.device,
            env_seed=args.seed,
        ),
        env_config,
        model_config,
        training_config,
    )
    # weights_only=True keeps pickle from executing code out of a checkpoint
    # file; the payload is only tensors and primitives. Read directly rather
    # than through framework.load_checkpoint() because the observation width
    # has to be compared before any state is loaded.
    payload = torch.load(
        args.checkpoint,
        map_location=framework.device,
        weights_only=True,
    )
    # The observation width is 8 * num_prb_groups + 6, so evaluating on a
    # different carrier than the one trained on would fail deep inside
    # load_state_dict.
    trained_dim = int(payload.get("observation_dim", framework.env.obs_dim))
    if trained_dim != framework.env.obs_dim:
        raise SystemExit(
            f"checkpoint expects an observation width of {trained_dim} but this "
            f"configuration builds {framework.env.obs_dim}; evaluate with the "
            f"num_prb_groups the checkpoint was trained on"
        )
    framework.policy.load_state_dict(payload["policy"])
    metrics = framework.evaluate(
        update=int(payload.get("update", 0)),
        phase=str(payload.get("phase", "eval")),
    )
    text = dump_yaml({"checkpoint": str(args.checkpoint), **asdict(metrics)})
    print("la_evaluation_complete")
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
        print(f"la_evaluation_written,{args.output}")


if __name__ == "__main__":
    main()

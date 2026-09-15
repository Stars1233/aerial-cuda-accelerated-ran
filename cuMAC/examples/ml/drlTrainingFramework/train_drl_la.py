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

"""Train DRL link adaptation through the four-phase curriculum.

Point ``--simulator`` at a factory for the host system-level simulator. With no
``--simulator`` the bundled example simulator runs instead, which exercises the
whole pipeline but models no real radio.

    # Smoke-test the plumbing on the example simulator.
    python drlTrainingFramework/train_drl_la.py --clone-updates 50

    # Full curriculum against a host simulator.
    python drlTrainingFramework/train_drl_la.py \\
        --simulator my_sls.drl_la_adapter:build_simulator \\
        --sim-config my_sls/full_buffer.yaml \\
        --training-config drlTrainingFramework/config/la_staged_training_config.yaml
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path

# Runnable both as a plain script and as python -m drlTrainingFramework.train_drl_la.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

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
            "DRL link adaptation for full-buffer traffic: teacher cloning with "
            "optional actor bootstrap, critic warmup and PPO fine-tuning, on a "
            "host system-level simulator."
        )
    )
    add_simulator_arguments(parser)
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
        help="Actor-critic YAML.",
    )
    parser.add_argument(
        "--training-config",
        type=Path,
        default=DEFAULT_CONFIG_DIR / "la_training_config.yaml",
        help="Training YAML. Use la_staged_training_config.yaml for all phases.",
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        metavar="SECTION.FIELD=VALUE",
        action="append",
        default=[],
        help=(
            "Override any configuration field without writing a new YAML "
            "file, for example --set model.encoder_layers=[128,128] --set "
            "train.ppo_updates=25. Sections are "
            f"{', '.join(sorted(LA_OVERRIDE_SECTIONS))}. Repeatable."
        ),
    )
    parser.add_argument(
        "--device",
        help="Torch device for the policy and the simulator, e.g. cuda:0.",
    )
    parser.add_argument(
        "--clone-updates",
        type=int,
        help="Shorthand for --set train.clone_updates=VALUE.",
    )
    parser.add_argument(
        "--output-dir",
        help="Shorthand for --set train.output_dir=VALUE.",
    )
    parser.add_argument(
        "--resume",
        help="Shorthand for --set train.resume_checkpoint=VALUE.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    training_overrides: dict[str, object] = {}
    if args.clone_updates is not None:
        training_overrides["clone_updates"] = args.clone_updates
    if args.output_dir is not None:
        training_overrides["output_dir"] = args.output_dir
    if args.resume is not None:
        training_overrides["resume_checkpoint"] = args.resume

    try:
        explicit = parse_la_config_overrides(args.overrides)
    except ValueError as error:
        raise SystemExit(f"invalid --set override: {error}") from error
    training_overrides.update(explicit["train"])

    env_config = LaEnvConfig.from_yaml(args.env_config, overrides=explicit["env"])
    model_config = LaModelConfig.from_yaml(
        args.model_config,
        overrides=explicit["model"],
    )
    training_config = LaTrainingConfig.from_yaml(
        args.training_config,
        overrides=training_overrides,
    )
    device = args.device if args.device is not None else training_config.device

    framework = LinkAdaptationTrainingFramework(
        build_env_factory(
            args,
            seed=training_config.seed,
            device=device,
            env_seed=training_config.seed,
        ),
        env_config,
        model_config,
        training_config,
    )
    result = framework.train()
    print("la_training_complete")
    print(dump_yaml(asdict(result)))


if __name__ == "__main__":
    main()

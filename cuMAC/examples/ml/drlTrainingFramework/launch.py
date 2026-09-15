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

"""Shared plumbing for the training and evaluation launchers.

This is the only place a host simulator is named, and it is named by a
``module:function`` string on the command line rather than by an import, so
attaching a different simulator never edits framework code.

A simulator factory must accept::

    def build_simulator(
        config_path: str | None = None,
        *,
        seed: int | None = None,
        device: str | None = None,
        **overrides: object,
    ) -> LinkAdaptationSimulator

``config_path`` is whatever ``--sim-config`` pointed at, opaque to the
framework; ``overrides`` are the ``--simulator-set key=value`` pairs. A factory
is free to ignore any of them.
"""

from __future__ import annotations

__all__ = [
    "DEFAULT_CONFIG_DIR",
    "DEFAULT_SIMULATOR",
    "PACKAGE_ROOT",
    "add_simulator_arguments",
    "build_env_factory",
    "dump_yaml",
    "ensure_package_importable",
    "parse_simulator_options",
    "resolve_simulator_factory",
]

import argparse
import importlib
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Mapping

import yaml

if TYPE_CHECKING:  # pragma: no cover - annotations only
    # Imported for typing alone so that argument parsing and factory
    # resolution stay usable without pulling in torch, which every one of
    # these modules does at import time. LinkAdaptationEnv is deliberately
    # absent: build_env_factory imports it at call time, and the nested
    # build() resolves the name from that enclosing scope.
    from .adapter import SimulatorFactory
    from .config import LaEnvConfig
    from .training import EnvFactory

PACKAGE_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG_DIR = PACKAGE_ROOT / "config"
DEFAULT_SIMULATOR = f"{PACKAGE_ROOT.name}.example_simulator:build_simulator"


def ensure_package_importable() -> None:
    """Put the package's parent directory on ``sys.path``.

    Lets the launchers run as plain scripts (``python
    drlTrainingFramework/train_drl_la.py``) as well as through ``-m``. The path
    is derived from this file's location, so the working directory does not
    matter. Idempotent: a second call is a no-op rather than a duplicate entry.

    Returns:
        None: The effect is the ``sys.path`` mutation. Whether the entry was
        already present is deliberately not reported, since a caller that has
        to branch on it is working around the idempotence rather than using it.

    Raises:
        None: The body only reads ``sys.path`` and inserts one string, so this
        never fails and needs no guard. The path itself is resolved once at
        import time, into :data:`PACKAGE_ROOT`.

    Examples:
        >>> ensure_package_importable()
        >>> ensure_package_importable()  # idempotent
        >>> import drlTrainingFramework  # now importable regardless of cwd
    """

    parent = str(PACKAGE_ROOT.parent)
    if parent not in sys.path:
        sys.path.insert(0, parent)


def add_simulator_arguments(parser: argparse.ArgumentParser) -> None:
    """Register the simulator-selection flags shared by both launchers.

    Adds ``--simulator``, ``--sim-config`` and ``--simulator-set``, which
    together are everything :func:`build_env_factory` needs.

    Args:
        parser (argparse.ArgumentParser): Parser to extend, modified in place.

    Returns:
        None: The flags are registered on ``parser`` itself. The parsed values
        reach :func:`build_env_factory` through the resulting namespace, not
        through a return value.

    Raises:
        argparse.ArgumentError: If ``parser`` already defines any of the three
            option strings. Both launchers call this before adding their own
            flags, so a conflict means a caller has claimed one of these names.

    Examples:
        >>> import argparse
        >>> parser = argparse.ArgumentParser()
        >>> add_simulator_arguments(parser)
        >>> args = parser.parse_args(["--simulator-set", "num_ues=8"])
        >>> args.simulator_options
        ['num_ues=8']
        >>> conflicting = argparse.ArgumentParser()
        >>> conflicting.add_argument("--simulator")  # doctest: +ELLIPSIS
        _StoreAction(...)
        >>> add_simulator_arguments(conflicting)  # doctest: +ELLIPSIS
        Traceback (most recent call last):
        argparse.ArgumentError: argument --simulator: conflicting...
    """

    parser.add_argument(
        "--simulator",
        default=DEFAULT_SIMULATOR,
        help=(
            "Simulator factory as module:function. Defaults to the bundled "
            "example simulator, which is a plumbing test double rather than a "
            "channel model; point this at your own adapter."
        ),
    )
    parser.add_argument(
        "--sim-config",
        type=Path,
        help=(
            "Configuration file handed to the simulator factory unchanged. The "
            "framework does not read it."
        ),
    )
    parser.add_argument(
        "--simulator-set",
        dest="simulator_options",
        metavar="KEY=VALUE",
        action="append",
        default=[],
        help=(
            "Keyword argument for the simulator factory, in YAML scalar syntax. "
            "Repeatable."
        ),
    )


def resolve_simulator_factory(spec: str) -> SimulatorFactory:
    """Import ``module:function`` and return the callable it names.

    This is the only place a host simulator is named, so attaching a different
    one is a command-line change rather than a code change. The module must be
    importable from the current interpreter, which normally means it is on
    ``PYTHONPATH`` or installed.

    Every failure path exits rather than raising, because the sole caller is a
    command-line launcher and a stack trace would bury the actual problem: a
    misspelled module, a missing attribute, or a name that is not callable.

    Args:
        spec (str): Factory reference as ``module:function``, for example
            ``my_sls.drl_la_adapter:build_simulator``.

    Returns:
        SimulatorFactory: The named callable. It is expected to accept
        ``(config_path=None, *, seed=None, device=None, **overrides)`` and
        return a
        :class:`~drlTrainingFramework.adapter.LinkAdaptationSimulator`, though
        that signature is only exercised when the factory is called.

    Raises:
        SystemExit: If ``spec`` is not ``module:function``, if the module
            cannot be imported, if the module has no such attribute, or if the
            attribute is not callable.

    Examples:
        >>> factory = resolve_simulator_factory(
        ...     "drlTrainingFramework.example_simulator:build_simulator"
        ... )
        >>> factory.__name__
        'build_simulator'
        >>> resolve_simulator_factory("missing_colon")  # doctest: +ELLIPSIS
        Traceback (most recent call last):
        SystemExit: --simulator 'missing_colon' must use module:function...
    """

    module_name, separator, attribute = str(spec).partition(":")
    if not separator or not module_name or not attribute:
        raise SystemExit(
            f"--simulator {spec!r} must use module:function syntax, for example "
            f"{DEFAULT_SIMULATOR}"
        )
    try:
        module = importlib.import_module(module_name)
    except ImportError as error:
        raise SystemExit(
            f"--simulator {spec!r}: cannot import {module_name!r} ({error}). Make "
            "sure it is on PYTHONPATH."
        ) from error
    try:
        factory = getattr(module, attribute)
    except AttributeError as error:
        raise SystemExit(
            f"--simulator {spec!r}: {module_name!r} has no attribute "
            f"{attribute!r}"
        ) from error
    if not callable(factory):
        raise SystemExit(f"--simulator {spec!r} does not name a callable")
    return factory


def parse_simulator_options(assignments: Iterable[str]) -> dict[str, Any]:
    """Parse ``key=value`` strings into factory keyword arguments.

    Values are read as YAML scalars, so integers, floats, booleans and ``null``
    keep their types instead of arriving as strings. A repeated key wins on its
    last occurrence.

    Args:
        assignments (Iterable[str]): ``key=value`` strings, as collected by the
            repeatable ``--simulator-set`` flag.

    Returns:
        dict[str, Any]: Keyword arguments to pass to the simulator factory.

    Raises:
        SystemExit: If an entry contains no ``=``, or if its value is not
            parseable YAML. Both are command-line mistakes, so they exit with
            the offending entry quoted rather than surfacing a traceback.

    Examples:
        >>> parse_simulator_options(["num_ues=8", "device=cpu", "strict=true"])
        {'num_ues': 8, 'device': 'cpu', 'strict': True}
        >>> parse_simulator_options(["num_ues"])  # doctest: +ELLIPSIS
        Traceback (most recent call last):
        SystemExit: --simulator-set 'num_ues' must use key=value...
        >>> parse_simulator_options(["cfg={a: 1"])  # doctest: +ELLIPSIS
        Traceback (most recent call last):
        SystemExit: --simulator-set 'cfg={a: 1' has a malformed YAML value:...
    """

    options: dict[str, Any] = {}
    for assignment in assignments:
        key, separator, text = str(assignment).partition("=")
        if not separator:
            raise SystemExit(
                f"--simulator-set {assignment!r} must use key=value syntax"
            )
        try:
            value = yaml.safe_load(text)
        except yaml.YAMLError as error:
            # An unbalanced bracket or quote raises ParserError, ScannerError or
            # ComposerError, none of which a traceback helps the user act on.
            # MarkedYAMLError.problem is the one-line diagnosis; its position
            # marks are dropped because they point into the value fragment
            # rather than at the command line.
            reason = getattr(error, "problem", None) or str(error).splitlines()[0]
            raise SystemExit(
                f"--simulator-set {assignment!r} has a malformed YAML value: "
                f"{str(reason).strip()}"
            ) from error
        options[key.strip()] = value
    return options


def build_env_factory(
    args: argparse.Namespace,
    *,
    seed: int,
    device: str | None,
    env_seed: int = 0,
) -> EnvFactory:
    """Build the :data:`~drlTrainingFramework.training.EnvFactory` from CLI args.

    The returned factory is called once per environment the framework needs, so
    training and validation each get their own simulator instance, which the
    framework then re-seeds per rollout through ``reset(seed=...)``.

    The ``--simulator`` reference and the ``--simulator-set`` options are
    resolved eagerly, here, so a typo fails before any training starts rather
    than on the first call. Errors raised by the factory itself surface later,
    when the framework builds an environment.

    Args:
        args (argparse.Namespace): Parsed arguments carrying the three
            attributes registered by :func:`add_simulator_arguments`:
            ``simulator``, ``sim_config`` and ``simulator_options``.
        seed (int): Seed handed to the simulator factory. The framework
            overrides it per rollout, so it only sets the initial draw.
        device (str | None): Torch device for the simulator, or ``None`` to
            let it choose. The policy follows whatever device the simulator's
            :class:`~drlTrainingFramework.adapter.CarrierProfile` declares.
        env_seed (int): Seed for the environment's own ACK sampling, which is
            independent of the simulator's randomness.

    Returns:
        EnvFactory: A callable taking a
        :class:`~drlTrainingFramework.config.LaEnvConfig` and returning a
        :class:`~drlTrainingFramework.env.LinkAdaptationEnv` wrapped around a
        freshly constructed simulator.

    Raises:
        SystemExit: If ``--simulator`` cannot be resolved or a
            ``--simulator-set`` entry is not ``key=value``. Propagated from
            :func:`resolve_simulator_factory` and
            :func:`parse_simulator_options`.

    Examples:
        >>> import argparse
        >>> from drlTrainingFramework.config import LaEnvConfig
        >>> args = argparse.Namespace(
        ...     simulator="drlTrainingFramework.example_simulator:build_simulator",
        ...     sim_config=None,
        ...     simulator_options=["num_ues=8", "num_prb_groups=4"],
        ... )
        >>> env_factory = build_env_factory(args, seed=17, device="cpu")
        >>> env = env_factory(LaEnvConfig())
        >>> env.num_ues, env.num_prb_groups, env.obs_dim
        (8, 4, 38)
    """

    from .env import LinkAdaptationEnv

    factory = resolve_simulator_factory(args.simulator)
    options = parse_simulator_options(args.simulator_options)
    config_path = str(args.sim_config) if args.sim_config is not None else None

    def build(env_config: LaEnvConfig) -> LinkAdaptationEnv:
        simulator = factory(config_path, seed=seed, device=device, **options)
        return LinkAdaptationEnv(simulator, env_config, seed=env_seed)

    return build


def dump_yaml(report: Mapping[str, Any]) -> str:
    """Render a report as YAML for stdout, preserving key order.

    Insertion order is kept rather than sorted, so a report reads in the order
    its dataclass declares its fields.

    Args:
        report (Mapping[str, Any]): Values to render. Must contain only types
            :func:`yaml.safe_dump` accepts, i.e. primitives and plain
            containers.

    Returns:
        str: The YAML document, without a trailing newline.

    Raises:
        yaml.representer.RepresenterError: If a value has no safe YAML
            representation.

    Examples:
        >>> print(dump_yaml({"update": 3, "phase": "ppo", "seed": None}))
        update: 3
        phase: ppo
        seed: null
    """

    return yaml.safe_dump(dict(report), sort_keys=False).strip()

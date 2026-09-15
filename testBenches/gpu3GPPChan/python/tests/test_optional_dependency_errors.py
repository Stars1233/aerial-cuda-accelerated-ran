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

"""cupy and cuda-bindings are required dependencies, but the high-level wrappers
import them lazily (see ``__getattr__`` in ``gpu3gppchan/__init__.py``) so that
``import gpu3gppchan`` stays cheap. If one of them is missing the install is
broken, and the error must say so and name the package — a bare
``ModuleNotFoundError: No module named 'cupy'`` raised from deep inside the
package gives a user nothing to act on.
"""

from __future__ import annotations

import sys
from typing import Iterator

import pytest


# Every lazily-imported wrapper reaches cupy: fading_channel and
# statistical_channel import it directly, channel_config imports it at module
# scope, and all three pull in cuda_utils.
LAZY_ATTRS = ["FadingChannel", "StatisticalChannel", "TdlChannelConfig"]


def _purge_wrapper_modules() -> dict[str, object]:
    """Drop cached wrapper modules so a later import is genuinely re-attempted.

    The compiled extension is deliberately kept: it has no bearing on the
    optional-dependency path and reloading it is needless work.
    """
    doomed = [
        name
        for name in sys.modules
        if name.startswith("gpu3gppchan.") and name != "gpu3gppchan._gpu3gppchan"
    ]
    return {name: sys.modules.pop(name) for name in doomed}


@pytest.fixture(name="without_module")
def _without_module() -> Iterator:
    """Make a top-level module unimportable, then restore the import state."""
    saved_wrappers: dict[str, object] = {}
    saved_blocked: dict[str, object] = {}

    def block(module_name: str) -> None:
        nonlocal saved_wrappers
        saved_wrappers = _purge_wrapper_modules()
        for name in list(sys.modules):
            if name == module_name or name.startswith(module_name + "."):
                saved_blocked[name] = sys.modules.pop(name)
        # A None entry makes `import <module_name>` raise ImportError.
        sys.modules[module_name] = None  # type: ignore[assignment]

    yield block

    for name in list(sys.modules):
        if sys.modules[name] is None:
            del sys.modules[name]
    sys.modules.update(saved_blocked)
    _purge_wrapper_modules()
    sys.modules.update(saved_wrappers)


@pytest.mark.parametrize("attr", LAZY_ATTRS)
def test_missing_cupy_names_package_and_install_command(without_module, attr) -> None:
    import gpu3gppchan

    without_module("cupy")

    with pytest.raises(ImportError) as excinfo:
        getattr(gpu3gppchan, attr)

    message = str(excinfo.value)
    assert "cupy-cuda13x" in message, (
        f"error does not name the distribution to install: {message!r}"
    )
    assert "required dependency" in message, (
        f"error does not say the install is incomplete: {message!r}"
    )
    assert "pip install" in message, (
        f"error does not give a runnable repair command: {message!r}"
    )


def test_missing_cuda_bindings_names_package_and_install_command(
    without_module,
) -> None:
    import gpu3gppchan

    # Block only `cuda.bindings`, not the whole `cuda` namespace: cupy itself
    # imports `cuda.pathfinder`, so blocking `cuda` would fail inside cupy
    # before reaching the import this test is about.
    without_module("cuda.bindings")

    with pytest.raises(ImportError) as excinfo:
        gpu3gppchan.FadingChannel

    message = str(excinfo.value)
    assert "cuda-bindings" in message, (
        f"error does not name the missing distribution: {message!r}"
    )
    assert "required dependency" in message, (
        f"error does not say the install is incomplete: {message!r}"
    )


def test_unknown_attribute_still_raises_attribute_error() -> None:
    """The new error handling must not swallow ordinary attribute lookups."""
    import gpu3gppchan

    with pytest.raises(AttributeError):
        gpu3gppchan.NoSuchThing


def test_wrappers_import_when_dependencies_present() -> None:
    """Guards the inverse: with cupy and cuda-bindings installed, the lazy
    imports must still succeed. Without this, raising unconditionally would
    pass the tests above."""
    pytest.importorskip("cupy")
    pytest.importorskip("cuda.bindings.runtime")

    import gpu3gppchan

    for attr in LAZY_ATTRS:
        assert getattr(gpu3gppchan, attr) is not None

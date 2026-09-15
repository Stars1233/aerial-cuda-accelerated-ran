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

"""Package the gpu3gppchan artifacts already built by CMake."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import setuptools
from setuptools.command.build_py import build_py
from setuptools.errors import SetupError

RUNTIME_DEPENDENCIES = [
    "numpy",
    "h5py",
    "pyyaml",
    "nvidia-cuda-runtime>=13.0.88",
    "nvidia-curand>=10.4.0",
    "cuda-bindings",
    "cupy-cuda13x",
]


class BuildPyWithNativeArtifacts(build_py):
    """Copy the CMake-built extension and runtime library into the wheel."""

    def _native_artifacts(self) -> list[Path]:
        native_dir_value = os.environ.get("GPU3GPPCHAN_NATIVE_DIR")
        if not native_dir_value:
            raise SetupError(
                "GPU3GPPCHAN_NATIVE_DIR is required; build the "
                "gpu3gppchan_wheel CMake target"
            )
        native_dir = Path(native_dir_value).resolve()
        extensions = sorted(native_dir.glob("_gpu3gppchan*.so"))
        libraries = sorted(native_dir.glob("libgpu3gppchan.so*"))
        if len(extensions) != 1:
            raise SetupError(
                f"Expected one _gpu3gppchan extension under {native_dir}, "
                f"found {[path.name for path in extensions]}"
            )
        if not libraries:
            raise SetupError(
                f"No libgpu3gppchan.so runtime library found under {native_dir}"
            )
        return [extensions[0], *libraries]

    def run(self) -> None:
        super().run()
        package_dir = Path(self.build_lib) / "gpu3gppchan"
        package_dir.mkdir(parents=True, exist_ok=True)
        for artifact in self._native_artifacts():
            shutil.copy2(artifact, package_dir / artifact.name)

    def get_outputs(self, include_bytecode: bool = True) -> list[str]:
        outputs = super().get_outputs(include_bytecode)
        package_dir = Path(self.build_lib) / "gpu3gppchan"
        outputs.extend(
            str(package_dir / artifact.name)
            for artifact in self._native_artifacts()
        )
        return outputs


class BinaryDistribution(setuptools.Distribution):
    """Mark the wheel as platform-specific even though compilation is external."""

    def has_ext_modules(self) -> bool:
        return True


setuptools.setup(
    name="gpu3gppchan",
    version="0.1.4",
    description="GPU-accelerated 3GPP channel models (TR 38.901)",
    long_description=(Path(__file__).parent / "README.md").read_text(
        encoding="utf-8"
    ),
    long_description_content_type="text/markdown",
    license="Apache-2.0",
    python_requires=">=3.10",
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    install_requires=RUNTIME_DEPENDENCIES,
    extras_require={"test": ["pytest"]},
    cmdclass={"build_py": BuildPyWithNativeArtifacts},
    distclass=BinaryDistribution,
    package_data={"gpu3gppchan": ["py.typed"]},
    zip_safe=False,
)

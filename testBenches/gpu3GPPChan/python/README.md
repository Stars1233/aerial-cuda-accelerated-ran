<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# gpu3gppchan Python package

The `gpu3gppchan` package exposes the SDK channel-model library through a
nanobind extension. CMake owns compilation of both `libgpu3gppchan` and the
extension, then packages those pre-built artifacts into a wheel.

## CMake configure presets

GPU3GPP channel Python uses dedicated presets that build the component without
pyAerial:

| Preset | Platform | Inherits | Build directory |
|--------|----------|----------|-----------------|
| `gpu3gppchan-x86` | x86_64 | `minimal-x86` | `build-gpu3gppchan-x86/` |
| `gpu3gppchan-arm` | aarch64 | `minimal-arm` | `build-gpu3gppchan-arm/` |

Use `gpu3gppchan-arm` instead of `gpu3gppchan-x86` on an ARM server. Matching
build presets (`cmake --build --preset …`) use the same names.

Inside the Aerial container, configure the appropriate GPU3GPP Python preset
and build the wheel:

```bash
cmake --preset gpu3gppchan-x86
cmake --build --preset gpu3gppchan-x86 --target gpu3gppchan_wheel
```

The build uses the system Python interpreter found in the Aerial container
(currently Python 3.12). Set `ACAR_GPU3GPPCHAN_PYTHON_EXECUTABLE` at configure
time only when an equivalent environment is located elsewhere.

To create the component-local test environment and install the CMake-built
wheel into it, build `gpu3gppchan_python_setup`. It creates or reuses
`testBenches/gpu3GPPChan/python/.venv` from that system Python and reuses its
site packages:

```bash
cmake --build --preset gpu3gppchan-x86 --target gpu3gppchan_python_setup
```

The wheel is written below the configured build directory. When pyAerial is
enabled, its separate `pyaerial_setup` target also consumes this wheel.

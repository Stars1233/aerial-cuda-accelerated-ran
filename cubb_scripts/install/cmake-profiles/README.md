# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Build profiles for Aerial SDK

Build profiles set **BUILD_PRESET** and **PROFILE_CMAKE_FLAGS**. These are combined with platform-specific flags and the user’s **BUILD_CMAKE_FLAGS** (so user flags are not overridden by the profile). Use profiles via **PROFILE** or make targets.

## Usage

- **Script:** `PROFILE=fapi_10_02.conf ./quickstart-aerial.sh` or `BUILD_CMAKE_FLAGS="-Dfoo=ON" ./quickstart-aerial.sh` to add flags on top of profile + platform.
- **Make:** `make build_aerial PROFILE=fapi_10_02.conf` or `BUILD_CMAKE_FLAGS="-Dfoo=ON" make build_aerial`

## Profiles

| Profile          | Preset | Description |
|------------------|--------|-------------|
| **oai.conf**     | 10_02  | OAI L2+ (default). Enables 10.04 SRS |
| **fapi_10_02.conf** | 10_02  | FAPI 10_02 only |
| **fapi_10_04.conf** | 10_04  | FAPI 10_04 build (SCF_FAPI_10_04=ON). |

## Adding a profile

Add a file `cmake-profiles/<name>.conf` that exports:

- `BUILD_PRESET` – one of: `perf`, `10_02`, `10_04`, `10_04_32dl`
- `PROFILE_CMAKE_FLAGS` – CMake flags for this profile (combined with platform flags and user BUILD_CMAKE_FLAGS)

Example:

```bash
# cmake-profiles/my_variant.conf
export BUILD_PRESET="10_04"
export PROFILE_CMAKE_FLAGS="-DENABLE_CUMAC=OFF"
```

Then run: `PROFILE=my_variant.conf make build_aerial`. User flags: `BUILD_CMAKE_FLAGS="-Dfoo=ON" make build_aerial PROFILE=my_variant.conf`.

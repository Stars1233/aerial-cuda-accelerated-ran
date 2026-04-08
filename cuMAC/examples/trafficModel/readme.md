<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Traffic Test

This document provides instructions on how to use the `trafficModel` for testing traffic scenarios.

## Usage

```sh
./trafficModel [options]
```

## Options

- `-h` : Print help
- `-t` : Specify number of TTI (Transmission Time Intervals)
- `-n` : Specify number of UEs (User Equipments)
- `-r` : Specify traffic rate in bytes/TTI
- `-d` : Specify buffer drain rate/TTI
- `-s` : Specify ranodm seed

## Example

```sh
./trafficModel -t 100 -n 50 -r 1000 -d 500
```

This example runs the `trafficModel` with 100 TTIs, 50 UEs, a traffic rate of 1000 bytes/TTI, and a buffer drain rate of 500 bytes/TTI.



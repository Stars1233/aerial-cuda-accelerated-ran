# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import json
import matplotlib.pyplot as plt
import argparse
import numpy as np
import os

from util import to_float, to_int, well_formed_samples

base = argparse.ArgumentParser()
base.add_argument(
    "--filenames",
    type=str,
    nargs="+",
    dest="filenames",
    help="Specifies the files containing the results",
    required=True,
)
base.add_argument(
    "--cells",
    type=str,
    dest="cells",
    help="Specifies the number of cells to focus analysis on",
    required=True,
)
base.add_argument(
    "--short_legend",
    action="store_true",
    default=False,
    dest="is_short_legend",
    help="whether add platform and nSlots in each legend",
)
args = base.parse_args()

gpu_freq = []
mem_freq = []
power = []
mem_used = []
testConfig = []

if "+" in args.cells:
    cells = "+".join([str(x).zfill(2) for x in args.cells.split("+")])
else:
    cells = str(args.cells).zfill(2)

for filename in args.filenames:

    with open(filename, "r") as ifile:
        data = json.load(ifile)

    samples = data.get(cells)
    if samples is None:
        print(f"  [{filename}] missing cell key '{cells}', skipping")
        continue

    well_formed, skipped = well_formed_samples(samples, min_fields=4)
    if skipped:
        print(f"  [{filename}] {skipped} malformed sample row(s) skipped (need >=4 fields)")

    raw_gpu_freq = [to_int(x[0]) for x in well_formed]
    raw_mem_freq = [to_int(x[1]) for x in well_formed]
    na_gpu = sum(1 for v in raw_gpu_freq if v is None)
    na_mem = sum(1 for v in raw_mem_freq if v is None)
    if na_gpu:
        print(f"  [{filename}] {na_gpu} GPU freq sample(s) reported N/A")
    if na_mem:
        print(f"  [{filename}] {na_mem} mem freq sample(s) reported N/A")

    gpu_freq.extend([v for v in raw_gpu_freq if v is not None])
    mem_freq.extend([v for v in raw_mem_freq if v is not None])
    power.extend([to_float(x[2]) for x in well_formed])
    mem_used.extend([to_float(x[3]) for x in well_formed])
    testConfig.extend([data['testConfig']['gpuName'] + ' ' + str(data['testConfig']['sweeps']) + ' slots'])

valid_power = [v for v in power if not np.isnan(v)]
max_power = np.max(valid_power) if valid_power else None
std_freq = np.std(gpu_freq) if gpu_freq else None

parts = []
parts.append(f"Maximum power: {max_power}" if max_power is not None else "Maximum power: N/A")
parts.append(f"Frequency std.: {std_freq}" if std_freq is not None else "Frequency std.: N/A")
print(", ".join(parts))

plt.subplots(1, 2, figsize=(2 * 7.2, 4.8))
plt.subplot(1, 2, 1)
if gpu_freq:
    y, x = np.histogram(gpu_freq, bins=10000)
    cy = np.cumsum(y) / len(gpu_freq)

    plt.plot(x[1:], cy)
else:
    plt.text(0.5, 0.5, "N/A", ha="center", va="center", transform=plt.gca().transAxes)
plt.grid(True)
plt.xlabel("Frequency Cont. Load [MHz]")
plt.ylabel("CDF")

folder, _ = os.path.split(args.filenames[0])
if folder.split("/")[-1].isnumeric():
    plt.legend(["Freq. capped at " + folder.split("/")[-1] + " MHz"])

plt.subplot(1, 2, 2)
plt.grid(True)
valid_power_points = [(i, v) for i, v in enumerate(power) if not np.isnan(v)]
if valid_power_points:
    idx = [i for i, _ in valid_power_points]
    vals = [v for _, v in valid_power_points]
    plt.plot(np.array(idx) * 10e-3, vals)
else:
    plt.text(0.5, 0.5, "N/A", ha="center", va="center", transform=plt.gca().transAxes)
plt.xlabel("Time [s]")
plt.ylabel("Power [W]")

if not args.is_short_legend:
    plt.legend(testConfig) # add info for each data
    
# plt.figure()
# plt.grid(True)
# plt.plot(np.arange(0, len(mem_used)) * 10e-3, mem_used)
# plt.xlabel("Time [s]")
# plt.ylabel("Memory Used [MiB]")

plt.savefig("power.png")
# plt.show()

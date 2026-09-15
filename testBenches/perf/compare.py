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

import matplotlib.pyplot as plt
import numpy as np
import argparse
import itertools
import json
import os
import sys
import datetime

# convert latency to cdf and save to channels
def latency_to_cdf(ikey, latency, chanName, channels):
    if len(latency) > 0:
        y, x = np.histogram(latency, bins=10000)
        cy = np.cumsum(y) / len(latency)

        if args.is_filter:
            for idx, item in enumerate(x):
                if cy[idx] > threshold:
                    x = x[: idx + 1]
                    cy = cy[:idx]
                    break

        store = channels.get(chanName, {})

        store[ikey] = {}

        store[ikey]["x"] = x[1:]
        store[ikey]["y"] = cy

        channels[chanName] = store

base = argparse.ArgumentParser()
base.add_argument(
    "--filenames",
    type=str,
    nargs="+",
    dest="filenames",
    help="Specifies the files containing the results",
)
base.add_argument(
    "--filename",
    type=str,
    nargs="+",
    dest="filename",
    help="Alias for --filenames (same effect)",
)
base.add_argument(
    "--folder",
    type=str,
    dest="folder",
    help="Specifies the folder containing the results",
)
base.add_argument(
    "--cells",
    type=str,
    nargs="+",
    dest="cells",
    default=None,
    help="Specifies the number of cells to focus the comparison for. If omitted, all cells in the JSON are plotted.",
)
base.add_argument(
    "--filter",
    action="store_true",
    dest="is_filter",
    help="Specifies whether to remove values > mean + 3 sigma",
)
base.add_argument(
    "--percentile",
    type=float,
    dest="threshold",
    default=99.5,
    help="Specifies whether to remove values > mean + 3 sigma",
)
base.add_argument(
    "--fdd_pusch",
    action="store_true",
    dest="is_fdd_pusch",
    help="Internal",
)
base.add_argument(
    "--disable_uc",
    action="store_true",
    dest="is_disable_uc",
    help="Internal",
)
base.add_argument(
    "--mac_combine",
    action="store_true",
    dest="is_mac_combine",
    help="Internal",
)
base.add_argument(    
    "--short_legend",
    action="store_true",
    default=False,
    dest="is_short_legend",
    help="whether add platform and nSlots in each legend",
)
base.add_argument(
    "--short_title",
    action="store_true",
    default=False,
    dest="is_short_title",
    help="Omit SM allocation and TDD priorities from the plot title",
)
base.add_argument(
    "--legend",
    type=str,
    nargs="+",
    dest="legend_labels",
    default=None,
    help="Custom legend labels, one per input file",
)
base.add_argument(
    "--mac_title",
    type=str,
    dest="mac_title",
    default=None,
    help="Custom title for the MAC subplot",
)
def _png_path(value: str) -> str:
    if not value.endswith(".png"):
        raise argparse.ArgumentTypeError(f"Output path must end with '.png', got: {value!r}")
    return value

base.add_argument(
    "--output",
    type=_png_path,
    dest="output",
    default=None,
    help="Output PNG file path (must end with .png; overrides the default date-stamped name)",
)

args = base.parse_args()

# Merge --filename and --filenames, preserving order and duplicates
_fn  = args.filenames or []
_fa  = getattr(args, "filename", None) or []
merged = _fn + _fa
args.filenames = merged if merged else None

if args.filenames is None and args.folder is None:
    base.error("please specify which files or folder to analyze")

if args.cells is not None and len(args.cells) > 1:
    if args.folder is not None:
        base.error("multiple cell counts can only be specified when using --filenames")
    if args.filenames is None or len(args.cells) != len(args.filenames):
        base.error("when passing multiple --cells, provide the same number of --filenames")

data = {}
legend = []

offset = 0

usecase = None

filenames = []

if args.filenames is not None:

    for fn in args.filenames:
        if "F01" in fn:
            usecase = "F01"
            break
        elif "F14" in fn:
            usecase = "F14"
            break
        elif "F08" in fn:
            usecase = "F08"
            break
        elif "F09" in fn:
            usecase = "F09"
            break
        else:
            sys.exit(
                "error: currently only F01, F08, F09 and F14 use cases are supported"
            )

    filenames.extend(args.filenames)

    for idx, filename in enumerate(args.filenames):
        ifile = open(filename, "r")
        data[idx] = json.load(ifile)
        ifile.close()
        
        appenTestConfig = ''
        if not args.is_short_legend:
            appenTestConfig = ' ' + data[idx]['testConfig']['gpuName'] + ' ' + str(data[idx]['testConfig']['sweeps']) + ' slots'
                
        if len(filename.split("/")) > 1:
            legend.append(filename.split("/")[-2] + appenTestConfig)
        else:
            if args.cells is not None and len(args.cells) > 1:
                legend.append(args.cells[idx] + appenTestConfig)
            else:
                legend.append(f"Dataset #{idx+1}" + appenTestConfig)
        offset += 1

if args.folder is not None:
    folder = args.folder

    folder_filenames = []

    raw = os.walk(folder)

    for item in raw:
        c_root, c_flds, c_files = item
        for c_file in c_files:
            if ".json" in c_file:
                filename = os.path.join(c_root, c_file)
                folder_filenames.append(filename)

    folder_filenames = sorted(folder_filenames)

    for ffn in folder_filenames:
        if "F01" in ffn:
            usecase = "F01"
            break
        elif "F14" in ffn:
            usecase = "F14"
            break
        elif "F08" in ffn:
            usecase = "F08"
            break
        elif "F09" in ffn:
            usecase = "F09"
            break
        else:
            sys.exit(
                "error: currently only F01, F08, F09 and F14 use cases are supported"
            )

    filenames.extend(folder_filenames)

    for idx, filename in enumerate(folder_filenames):
        ifile = open(filename, "r")
        data[idx + offset] = json.load(ifile)
        ifile.close()

        if os.path.split(filename)[0] == folder:
            legend.append(f"Dataset #{idx+1}" + appenTestConfig)
        else:
            legend.append(os.path.split(os.path.split(filename)[0])[-1] + appenTestConfig)

    offset += len(folder_filenames)

if args.cells is None:
    expanded_data = {}
    expanded_legend = []
    expanded_key = []
    expanded_filenames = []
    eidx = 0
    for idx in sorted(data.keys()):
        cell_keys = sorted([k for k in data[idx] if k not in ('testConfig', 'constraints')])
        for ck in cell_keys:
            expanded_data[eidx] = data[idx].copy()
            appenTestConfig = ''
            if not args.is_short_legend:
                appenTestConfig = ' ' + data[idx]['testConfig']['gpuName'] + ' ' + str(data[idx]['testConfig']['sweeps']) + ' slots'
            expanded_legend.append(ck + appenTestConfig)
            expanded_key.append(ck)
            if idx < len(filenames):
                expanded_filenames.append(filenames[idx])
            eidx += 1
    data = expanded_data
    legend = expanded_legend
    filenames = expanded_filenames
    key = expanded_key

if args.legend_labels:
    if len(args.legend_labels) != len(legend):
        base.error(
            f"--legend count ({len(args.legend_labels)}) does not match "
            f"the number of data series ({len(legend)})"
        )
    legend = list(args.legend_labels)
legend.append("Constraint")

if args.cells is not None:
    if len(args.cells) > 1:

        key = []

        for cells in args.cells:
            if cells.isnumeric():
                key.append(cells.zfill(2))
            else:
                key.append("+".join([x.zfill(2) for x in cells.split("+")]))
    else:
        if args.cells[0].isnumeric():
            key = [args.cells[0].zfill(2)] * len(data.keys())
        else:
            key = ["+".join([x.zfill(2) for x in args.cells[0].split("+")])] * len(
                data.keys()
            )

threshold = args.threshold / 100

# plt.subplots(1, number_of_plots, figsize=(7.2 * number_of_plots, 4.8))
# plt.subplot(1, number_of_plots, 1)

channels = {}
constraints = {}

for kidx, ikey in enumerate(data.keys()):

    mMIMO_pattern = (data[ikey]['testConfig']['pattern'] == "dddsuudddd_mMIMO")
    if not constraints:
        constraints   = data[ikey]['constraints']  # only read constraint from the first cell setting
    else: # compare if constaints are the same for comparison
        all_keys = (set(constraints) | set(data[ikey]['constraints'])) - {'MAC'}
        diff = {k: (constraints.get(k), data[ikey]['constraints'].get(k))
                for k in all_keys if constraints.get(k) != data[ikey]['constraints'].get(k)}
        if diff:
            raise ValueError("Error: The constraints for channels among the input files are not equal.")       
    
    # --------------------           UL channels           --------------------
    if usecase != "F01":

        trace_type = data[ikey][key[kidx]].get("Mode", None)

        if trace_type is not None:
            # currently PUSCH1 and PUSCH2 are always separated
            # PUCC1 and PUCCH2 are separate only in new mMIMO pattern due to different budget
            ul_cells = data[ikey][key[kidx]].get("PUSCH1", [])
            
            ul_cells_subslotProc = data[ikey][key[kidx]].get("PUSCH1_SUBSLOT_PROC", [])
            
            ul_cells2 = data[ikey][key[kidx]].get("PUSCH2", [])
            
            ul_cells2_subslotProc = data[ikey][key[kidx]].get("PUSCH2_SUBSLOT_PROC", [])
            
            ul_cells2 = data[ikey][key[kidx]].get("PUSCH2", [])
            
            ulbfw_cells1 = data[ikey][key[kidx]].get("ULBFW1", [])
            
            ulbfw_cells2 = data[ikey][key[kidx]].get("ULBFW2", [])
            
            ulbfw_cells = ulbfw_cells1 +  ulbfw_cells2 # combine ULBFW

            cul_cells1 = data[ikey][key[kidx]].get("PUCCH1", [])
        
            cul_cells2 = data[ikey][key[kidx]].get("PUCCH2", [])

            cul_cells = cul_cells1 + cul_cells2 # compbine PUCCH
        else:
            ul_cells = data[ikey][key[kidx]].get("Total", [])
            if len(ul_cells) == 0:
                ul_cells = data[ikey][key[kidx]].get("PUSCH", [])
                ul_cells_subslotProc = data[ikey][key[kidx]].get("PUSCH_SUBSLOT_PROC", [])

            cul_cells = data[ikey][key[kidx]].get("PUCCH", [])
            
            ulbfw_cells = data[ikey][key[kidx]].get("ULBFW", [])

    else:
        if args.is_fdd_pusch:

            raw = data[ikey][key[kidx]].get("PDSCH", [])

            if type(raw[0]) == list:
                dl_cells = list(itertools.chain.from_iterable(raw))
            else:
                dl_cells = raw

            ul_cells = list(
                np.array(data[ikey][key[kidx]].get("PUSCH", [])) - np.array(dl_cells)
            )
        else:
            ul_cells = data[ikey][key[kidx]].get("Total", [])
            if len(ul_cells) == 0:
                ul_cells = data[ikey][key[kidx]].get("PUSCH", [])
        
    srs1 = data[ikey][key[kidx]].get("SRS1", [])
    srs2 = data[ikey][key[kidx]].get("SRS2", [])
    prach = data[ikey][key[kidx]].get("PRACH", [])
    
    # convert latency to CDF and save in channels
    latency_to_cdf(ikey, ul_cells_subslotProc, "PUSCH1_SUBSLOT_PROC", channels)
    latency_to_cdf(ikey, ul_cells, "PUSCH1", channels)
    latency_to_cdf(ikey, ul_cells2_subslotProc, "PUSCH2_SUBSLOT_PROC", channels)
    latency_to_cdf(ikey, ul_cells2, "PUSCH2", channels)
    if mMIMO_pattern:
        latency_to_cdf(ikey, cul_cells1, "PUCCH1", channels)
        latency_to_cdf(ikey, cul_cells2, "PUCCH2", channels)
        latency_to_cdf(ikey, ulbfw_cells1, "ULBFW1", channels)
        latency_to_cdf(ikey, ulbfw_cells2, "ULBFW2", channels)
    else:
        latency_to_cdf(ikey, cul_cells, "PUCCH", channels)
        latency_to_cdf(ikey, ulbfw_cells, "ULBFW", channels)
    
    latency_to_cdf(ikey, srs1, "SRS1", channels)
    latency_to_cdf(ikey, srs2, "SRS2", channels)
    latency_to_cdf(ikey, prach, "PRACH", channels)
    
    # --------------------           DL channels           --------------------
    # PDSCH and DLBFW
    # separate in new mMIMO pattern; add together in other patterns
    raw = data[ikey][key[kidx]].get("PDSCH", [])
    if len(raw) > 0:

        if usecase == "F01":
            if type(raw[0]) == list:
                dl_cells = list(itertools.chain.from_iterable(raw))
            else:
                dl_cells = raw
        else:
            dl_cells = []

            if type(raw) == dict:
                for ikey in raw.keys():
                    dl_cells.extend(itertools.chain.from_iterable(raw[ikey]))
            elif type(raw) == list:
                dl_cells = raw

    dlbfw_buffer = data[ikey][key[kidx]].get("DLBFW", [])
    
    if usecase != "F01" and len(dlbfw_buffer) > 0:
        
        if mMIMO_pattern: # no need to combine PDCCH + CSI-RS
            
            latency_to_cdf(ikey, dl_cells, "PDSCH", channels)
            latency_to_cdf(ikey, dlbfw_buffer, "DLBFW", channels)
        
        else:

            # add PDSCH + DLBFW
            y, x = np.histogram(np.array(dl_cells) + np.array(dlbfw_buffer), bins=10000)
            cy = np.cumsum(y) / len(dlbfw_buffer)

            if args.is_filter:
                for idx, item in enumerate(x):
                    if cy[idx] > threshold:
                        x = x[: idx + 1]
                        cy = cy[:idx]
                        break

            store = channels.get("PDSCH+DLBFW", {})

            store[ikey] = {}

            store[ikey]["x"] = x[1:]
            store[ikey]["y"] = cy

            channels["PDSCH+DLBFW"] = store
    else: # F01 or no DLBFW
        latency_to_cdf(ikey, dl_cells, "PDSCH", channels)
    
    # PDCCH and CSI-RS
    # separate in new mMIMO pattern; add together in other patterns
    raw = data[ikey][key[kidx]].get("PDCCH", [])

    if len(raw) > 0:

        if usecase == "F01":
            if type(raw[0]) == list:
                cdl_cells = list(itertools.chain.from_iterable(raw))
            else:
                cdl_cells = raw
        else:
            cdl_cells = raw

    csirs_buffer = data[ikey][key[kidx]].get("CSI-RS", [])

    if len(csirs_buffer) > 0:

        if mMIMO_pattern: # no need to combine PDCCH + CSI-RS
            latency_to_cdf(ikey, cdl_cells, "PDCCH", channels)
            latency_to_cdf(ikey, csirs_buffer, "CSI-RS", channels)
            
        else:
            y, x = np.histogram(np.array(cdl_cells) + np.array(csirs_buffer), bins=10000)
            cy = np.cumsum(y) / len(csirs_buffer)

            if args.is_filter:
                for idx, item in enumerate(x):
                    if cy[idx] > threshold:
                        x = x[: idx + 1]
                        cy = cy[:idx]
                        break

            store = channels.get("PDCCH+CSI-RS", {})

            store[ikey] = {}

            store[ikey]["x"] = x[1:]
            store[ikey]["y"] = cy

            channels["PDCCH+CSI-RS"] = store

    # SSB
    ssb_buffer   = data[ikey][key[kidx]].get("SSB", [])
    latency_to_cdf(ikey, ssb_buffer, "SSB", channels)

    # --------------------           cuMAC          --------------------
    mac_buffer = data[ikey][key[kidx]].get("MAC", [])

    if args.is_mac_combine: # combine all MAC slot latencies into one figure        
        
        latency_to_cdf(ikey, mac_buffer, "MAC", channels)
        
    else: # separate cuMAC heavy kernel and light kernel using saved mac_slot_config and cumac_light_weight_flag
        test_cfg = data[ikey].get("testConfig", {})
        mac_slot_config = test_cfg.get("mac_slot_config")
        cumac_light_weight_flag = test_cfg.get("cumac_light_weight_flag")
        pattern_len = test_cfg.get("pattern_len", 10)

        if mac_slot_config is not None and cumac_light_weight_flag is not None:
            lw = cumac_light_weight_flag
            if len(lw) == 0:
                mac_heavy_buffer = mac_buffer[::8]
                mac_light_buffer = [element for index, element in enumerate(mac_buffer) if index % 8 != 0]
            else:
                n_mac_per_pattern = sum(1 for j in range(min(len(mac_slot_config), pattern_len)) if mac_slot_config[j] == 1)
                n_lw = len(lw)
                mac_heavy_buffer = []
                mac_light_buffer = []
                for i in range(len(mac_buffer)):
                    if not n_mac_per_pattern:
                        break
                    mac_run_idx = i % n_mac_per_pattern
                    lw_idx = mac_run_idx % n_lw
                    if lw[lw_idx] == 0:
                        mac_heavy_buffer.append(mac_buffer[i])
                    else:
                        mac_light_buffer.append(mac_buffer[i])
        else:
            # backward compat: fallback to previous hardcoded split (heavy every 8th slot)
            mac_heavy_buffer = mac_buffer[::8]
            mac_light_buffer = [element for index, element in enumerate(mac_buffer) if index % 8 != 0]

        latency_to_cdf(ikey, mac_heavy_buffer, "MAC_heavy", channels)
        latency_to_cdf(ikey, mac_light_buffer, "MAC_light", channels)
        
    # --------------------           cuMAC2          --------------------
    mac2_buffer = data[ikey][key[kidx]].get("MAC2", [])

    if args.is_mac_combine: # combine all MAC slot latencies into one figure        
        
        latency_to_cdf(ikey, mac2_buffer, "MAC2", channels)
        
    else: # separate cuMAC2 heavy kernel and light kernel using same config as MAC1
        test_cfg = data[ikey].get("testConfig", {})
        mac_slot_config = test_cfg.get("mac_slot_config")
        cumac_light_weight_flag = test_cfg.get("cumac_light_weight_flag")
        pattern_len = test_cfg.get("pattern_len", 10)
        mac2_lw_offset = test_cfg.get("mac2_light_weight_flag_offset", 2)
        # the fallback split assumes an 8-run heavy period, so keep the offset in 0..7
        mac2_fallback_off = mac2_lw_offset % 8

        if mac_slot_config is not None and cumac_light_weight_flag is not None:
            lw = cumac_light_weight_flag
            n_lw = len(lw)
            if n_lw == 0:
                mac2_heavy_buffer = mac2_buffer[mac2_fallback_off::8]
                mac2_light_buffer = [element for index, element in enumerate(mac2_buffer) if index % 8 != mac2_fallback_off]
            else:
                mac2_slot_indices = [j for j in range(min(len(mac_slot_config), pattern_len)) if mac_slot_config[j] == 1]
                n_mac2_per_pattern = len(mac2_slot_indices) if mac2_slot_indices else 0
                mac2_heavy_buffer = []
                mac2_light_buffer = []
                for i in range(len(mac2_buffer)):
                    if not n_mac2_per_pattern:
                        break
                    mac_run_idx = i % n_mac2_per_pattern
                    lw_idx = (mac_run_idx - mac2_lw_offset + n_lw) % n_lw
                    if lw[lw_idx] == 0:
                        mac2_heavy_buffer.append(mac2_buffer[i])
                    else:
                        mac2_light_buffer.append(mac2_buffer[i])
        else:
            # backward compat: fallback to previous hardcoded split
            mac2_heavy_buffer = mac2_buffer[mac2_fallback_off::8]
            mac2_light_buffer = [element for index, element in enumerate(mac2_buffer) if index % 8 != mac2_fallback_off]

        latency_to_cdf(ikey, mac2_heavy_buffer, "MAC2_heavy", channels)
        latency_to_cdf(ikey, mac2_light_buffer, "MAC2_light", channels)

# --------------------           draw CDFs figures          --------------------
sz_channels = len(channels.keys())

cols = np.min([3, sz_channels])
rows = int(np.ceil(sz_channels / 3))

fig, _ = plt.subplots(rows, cols, figsize=(7.2 * cols, 4.8 * rows))

if not args.is_short_title:
    _tc = next(iter(data.values())).get("testConfig", {}) if data else {}
    _sm = _tc.get("target", [])
    _prio = _tc.get("tdd_priorities", {})
else:
    _tc = {}

for idx, key in enumerate(list(channels.keys())):

    local_legend = []

    plt.subplot(rows, cols, idx + 1)

    for iidx, ikey in enumerate(list(channels[key].keys())):

        x = channels[key][ikey]["x"]
        y = channels[key][ikey]["y"]

        label = legend[ikey]

        local_legend.append(label)

        plt.plot(x, y, color=f"C{ikey}")

    if key == "PUSCH1":

        if usecase == "F01":
            if args.is_fdd_pusch:
                if args.is_disable_uc:
                    plt.title("PUSCH1")
                else:
                    plt.title(f"{usecase}: PUSCH1")
            else:
                if args.is_disable_uc:
                    plt.title("PDSCH + PUSCH")
                else:
                    plt.title(f"{usecase}: PDSCH + PUSCH")
            plt.vlines(1000, 0, 1, color="k")
        else:
            if args.is_disable_uc:
                plt.title("PUSCH1")
            else:
                plt.title(f"{usecase}: PUSCH1")
            plt.vlines(constraints["PUSCH1"], 0, 1, color="k")

    elif key == "PUSCH1_SUBSLOT_PROC":

        if usecase == "F01":
            if args.is_fdd_pusch:
                if args.is_disable_uc:
                    plt.title("PUSCH1_SUBSLOT_PROC")
                else:
                    plt.title(f"{usecase}: PUSCH1_SUBSLOT_PROC")
            else:
                if args.is_disable_uc:
                    plt.title("PDSCH + PUSCH1_SUBSLOT_PROC")
                else:
                    plt.title(f"{usecase}: PDSCH + PUSCH1_SUBSLOT_PROC")
            plt.vlines(1000, 0, 1, color="k")
        else:
            if args.is_disable_uc:
                plt.title("PUSCH1_SUBSLOT_PROC")
            else:
                plt.title(f"{usecase}: PUSCH1_SUBSLOT_PROC")
            plt.vlines(constraints["PUSCH1_SUBSLOT_PROC"], 0, 1, color="k")
            
    elif key == "PUSCH2":

        if usecase == "F01":
            if args.is_fdd_pusch:
                if args.is_disable_uc:
                    plt.title("PUSCH2")
                else:
                    plt.title(f"{usecase}: PUSCH2")
            else:
                if args.is_disable_uc:
                    plt.title("PDSCH + PUSCH")
                else:
                    plt.title(f"{usecase}: PDSCH + PUSCH")
            plt.vlines(1000, 0, 1, color="k")
        else:
            if args.is_disable_uc:
                plt.title("PUSCH2")
            else:
                plt.title(f"{usecase}: PUSCH2")
            plt.vlines(constraints["PUSCH2"], 0, 1, color="k")
    
    elif key == "PUSCH2_SUBSLOT_PROC":

        if usecase == "F01":
            if args.is_fdd_pusch:
                if args.is_disable_uc:
                    plt.title("PUSCH2_SUBSLOT_PROC")
                else:
                    plt.title(f"{usecase}: PUSCH2_SUBSLOT_PROC")
            else:
                if args.is_disable_uc:
                    plt.title("PDSCH + PUSCH2_SUBSLOT_PROC")
                else:
                    plt.title(f"{usecase}: PDSCH + PUSCH2_SUBSLOT_PROC")
            plt.vlines(1000, 0, 1, color="k")
        else:
            if args.is_disable_uc:
                plt.title("PUSCH2_SUBSLOT_PROC")
            else:
                plt.title(f"{usecase}: PUSCH2_SUBSLOT_PROC")
            plt.vlines(constraints["PUSCH2_SUBSLOT_PROC"], 0, 1, color="k")
        
    elif key == "ULBFW":
        if args.is_disable_uc:
            plt.title("ULBFW")
        else:
            plt.title(f"{usecase}: ULBFW")
        plt.vlines(500, 0, 1, color="k")
        
    elif key == "ULBFW1":
        if args.is_disable_uc:
            plt.title("ULBFW1")
        else:
            plt.title(f"{usecase}: ULBFW1")
        plt.vlines(constraints["ULBFW1"], 0, 1, color="k")
    
    elif key == "ULBFW2":
        if args.is_disable_uc:
            plt.title("ULBFW2")
        else:
            plt.title(f"{usecase}: ULBFW2")
        plt.vlines(constraints["ULBFW2"], 0, 1, color="k")
        
    elif key == "PDSCH":
        if usecase == "F01":
            if args.is_disable_uc:
                plt.title("PDSCH")
            else:
                plt.title(f"{usecase}: PDSCH")
            plt.vlines(750, 0, 1, color="k")
        else:
            if args.is_disable_uc:
                plt.title("PDSCH")
            else:
                plt.title(f"{usecase}: PDSCH")
        plt.vlines(constraints["PDSCH"], 0, 1, color="k")

    elif key == "DLBFW":
        if args.is_disable_uc:
            plt.title("DLBFW")
        else:
            plt.title(f"{usecase}: DLBFW")
        plt.vlines(constraints["DLBFW"], 0, 1, color="k")
        
    elif key == "PDSCH+DLBFW":
        if args.is_disable_uc:
            plt.title("PDSCH+DLBFW")
        else:
            plt.title(f"{usecase}: PDSCH+DLBFW")
        plt.vlines(constraints["PDSCh+DLBFW"], 0, 1, color="k")

    elif key == "PDCCH":
        if usecase == "F01":
            if args.is_disable_uc:
                plt.title("PDCCH")
            else:
                plt.title(f"{usecase}: PDCCH")
            plt.vlines(750, 0, 1, color="k")
        else:
            if args.is_disable_uc:
                plt.title("PDCCH")
            else:
                plt.title(f"{usecase}: PDCCH")
            plt.vlines(constraints["PDCCH"], 0, 1, color="k")

    elif key == "CSI-RS":
        if args.is_disable_uc:
            plt.title("CSI-RS")
        else:
            plt.title(f"{usecase}: CSI-RS")
        plt.vlines(constraints["CSI-RS"], 0, 1, color="k")
        
    elif key == "PDCCH+CSI-RS":
        if usecase == "F01":
            if args.is_disable_uc:
                plt.title("PDCCH+CSI-RS")
            else:
                plt.title(f"{usecase}: PDCCH+CSI-RS")
            plt.vlines(750, 0, 1, color="k")
        else:
            if args.is_disable_uc:
                plt.title("PDCCH+CSI-RS")
            else:
                plt.title(f"{usecase}: PDCCH+CSI-RS")
            plt.vlines(constraints["PDCCH+CSI-RS"], 0, 1, color="k")

    elif key == "PUCCH":
        if usecase == "F01":
            raise NotImplementedError
        else:
            if args.is_disable_uc:
                plt.title("PUCCH")
            else:
                plt.title(f"{usecase}: PUCCH")
            plt.vlines(constraints.get("PUCCH1", constraints.get("PUCCH", 1500)), 0, 1, color="k")

    elif key == "PUCCH1":
        if usecase == "F01":
            raise NotImplementedError
        else:
            if args.is_disable_uc:
                plt.title("PUCCH1")
            else:
                plt.title(f"{usecase}: PUCCH1")
            plt.vlines(constraints["PUCCH1"], 0, 1, color="k")

    elif key == "PUCCH2":
        if usecase == "F01":
            raise NotImplementedError
        else:
            if args.is_disable_uc:
                plt.title("PUCCH2")
            else:
                plt.title(f"{usecase}: PUCCH2")
            plt.vlines(constraints["PUCCH2"], 0, 1, color="k")

    elif key == "SRS1":
        if args.is_disable_uc:
            plt.title("SRS1")
        else:
            plt.title(f"{usecase}: SRS1")
        plt.vlines(constraints["SRS1"], 0, 1, color="k")

    elif key == "SRS2":
        if args.is_disable_uc:
            plt.title("SRS1")
        else:
            plt.title(f"{usecase}: SRS1")
        plt.vlines(constraints["SRS2"], 0, 1, color="k")

    elif key == "SSB":
        if args.is_disable_uc:
            plt.title("SSB")
        else:
            plt.title(f"{usecase}: SSB")
        plt.vlines(constraints["SSB"], 0, 1, color="k")

    elif key == "PRACH":
        if args.is_disable_uc:
            plt.title("PRACH")
        else:
            plt.title(f"{usecase}: PRACH")
        plt.vlines(constraints["PRACH"], 0, 1, color="k")
    
    elif key == "MAC":
        if args.mac_title:
            plt.title(args.mac_title)
        elif args.is_disable_uc:
            plt.title("MAC")
        else:
            plt.title(f"{usecase}: MAC")
        plt.vlines(constraints["MAC"], 0, 1, color="k")
        
    elif key == "MAC_heavy":
        if args.mac_title:
            plt.title(args.mac_title)
        elif args.is_disable_uc:
            plt.title("MAC_heavy")
        else:
            plt.title(f"{usecase}: MAC_heavy")
        plt.vlines(constraints["MAC"], 0, 1, color="k")

    elif key == "MAC_light":
        if args.mac_title:
            plt.title(args.mac_title)
        elif args.is_disable_uc:
            plt.title("MAC_light")
        else:
            plt.title(f"{usecase}: MAC_light")
        plt.vlines(constraints["MAC"], 0, 1, color="k")

    elif key == "MAC2":
        if args.is_disable_uc:
            plt.title("MAC2")
        else:
            plt.title(f"{usecase}: MAC2")
        plt.vlines(constraints["MAC2"], 0, 1, color="k")

    elif key == "MAC2_heavy":
        if args.is_disable_uc:
            plt.title("MAC2_heavy")
        else:
            plt.title(f"{usecase}: MAC2_heavy")
        plt.vlines(constraints["MAC2"], 0, 1, color="k")

    elif key == "MAC2_light":
        if args.is_disable_uc:
            plt.title("MAC2_light")
        else:
            plt.title(f"{usecase}: MAC2_light")
        plt.vlines(constraints["MAC2"], 0, 1, color="k")
        
    else:
        raise NotImplementedError

    local_legend.append("Constraint")
    plt.legend(local_legend)

    plt.grid(True)
    plt.ylabel("CDF")
    plt.xlabel("Latency [us]")

# Enrich subplot titles with SM allocation and TDD priority per channel.
# Maps channel names from titles to SM sub-context index using testConfig flags.
if not args.is_short_title and _tc:
    _sm = _tc.get("target", [])
    _prio = _tc.get("tdd_priorities", {})
    # Build sub-context order from testConfig isolation flags
    _ctx_idx = 0
    _ch_sm_idx = {}
    if _tc.get("is_isolated_prach") or _tc.get("is_prach"):
        _ch_sm_idx["PRACH"] = _ctx_idx
        _ctx_idx += 1
    if _tc.get("is_isolated_pdcch"):
        for _c in ("PDCCH", "PDCCH+CSI-RS", "CSI-RS"):
            _ch_sm_idx[_c] = _ctx_idx
        _ctx_idx += 1
    if _tc.get("is_isolated_pucch"):
        for _c in ("PUCCH", "PUCCH1", "PUCCH2"):
            _ch_sm_idx[_c] = _ctx_idx
        _ctx_idx += 1
    if not _tc.get("is_no_pdsch"):
        for _c in ("PDSCH", "PDSCH+DLBFW", "DLBFW"):
            _ch_sm_idx[_c] = _ctx_idx
        if not _tc.get("is_isolated_pdcch"):
            for _c in ("PDCCH", "PDCCH+CSI-RS", "CSI-RS"):
                _ch_sm_idx.setdefault(_c, _ctx_idx)
        if not (_tc.get("is_ssb") and _tc.get("is_isolated_ssb")):
            _ch_sm_idx.setdefault("SSB", _ctx_idx)
        _ctx_idx += 1
    if not _tc.get("is_no_pusch"):
        for _c in (
            "PUSCH1",
            "PUSCH2",
            "PUSCH1_SUBSLOT_PROC",
            "PUSCH2_SUBSLOT_PROC",
            "ULBFW",
            "ULBFW1",
            "ULBFW2",
        ):
            _ch_sm_idx[_c] = _ctx_idx
        if not _tc.get("is_isolated_pucch"):
            for _c in ("PUCCH", "PUCCH1", "PUCCH2"):
                _ch_sm_idx.setdefault(_c, _ctx_idx)
        _ctx_idx += 1
    if _tc.get("is_ssb") and _tc.get("is_isolated_ssb"):
        _ch_sm_idx["SSB"] = _ctx_idx
        _ctx_idx += 1
    if _tc.get("is_srs_isolate"):
        for _c in ("SRS1", "SRS2"):
            _ch_sm_idx[_c] = _ctx_idx
        _ctx_idx += 1
    if _tc.get("is_mac"):
        _ch_sm_idx["MAC"] = _ctx_idx
        _ch_sm_idx["MAC_heavy"] = _ctx_idx
        _ch_sm_idx["MAC_light"] = _ctx_idx
        _ctx_idx += 1

    _prio_map = {
        "PUSCH1": "PUSCH", "PUSCH2": "PUSCH", "PUSCH1_SUBSLOT_PROC": "PUSCH",
        "PUSCH2_SUBSLOT_PROC": "PUSCH", "PUCCH1": "PUCCH", "PUCCH2": "PUCCH",
        "PUCCH": "PUCCH", "PDSCH": "PDSCH", "PDSCH+DLBFW": "PDSCH",
        "PDCCH": "PDCCH", "PDCCH+CSI-RS": "PDCCH", "CSI-RS": "CSIRS",
        "PRACH": "PRACH", "SSB": "SSB", "DLBFW": "PDSCH",
        "ULBFW": "PUSCH", "ULBFW1": "PUSCH", "ULBFW2": "PUSCH",
        "SRS1": "SRS", "SRS2": "SRS", "MAC": "MAC",
    }

    for ax in fig.get_axes():
        _title = ax.get_title()
        if not _title:
            continue
        _ch = _title.split(": ", 1)[-1] if ": " in _title else _title
        _suffix_parts = []
        _si = _ch_sm_idx.get(_ch)
        if _si is not None and _si < len(_sm):
            _suffix_parts.append(f"{_sm[_si]} SMs")
        _pk = _prio_map.get(_ch)
        if _pk and _pk in _prio:
            _suffix_parts.append(f"Prio {_prio[_pk]}")
        if _suffix_parts:
            ax.set_title(f"{_title}, {', '.join(_suffix_parts)}", fontsize=10)

time = datetime.datetime.now()
buffer = "_".join([str(time.year), str(time.month).zfill(2), str(time.day).zfill(2)])
plt.tight_layout()

plt.savefig(args.output if args.output else f"compare-{buffer}.png", bbox_inches="tight")

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

import logging
import os
import yaml
import json
import re

logger = logging.getLogger(__name__)

from ....analyze import extract
from ....traffic_utils import (
    drop_minus_one_overrides,
    expand_tvs_for_cells,
    is_truthy,
    load_tdd_yaml_overrides,
)


def run(args, vectors, testcases, filenames, sLotConfig):

    (
        testcases_dl,
        testcases_ul,
        testcases_dlbf,
        testcases_ulbf,
        testcases_sr,
        testcases_ra,
        testcases_cdl,
        testcases_cul,
        testcases_ssb,
        testcases_cr,
        testcases_mac,
        testcases_mac2,
    ) = testcases
    (
        filenames_dl,
        filenames_ul,
        filenames_dlbf,
        filenames_ulbf,
        filenames_sr,
        filenames_ra,
        filenames_cdl,
        filenames_cul,
        filenames_ssb,
        filenames_cr,
        filenames_mac,
        filenames_mac2,
    ) = filenames

    ofile = open(vectors, "w")

    payload = {}
    payload["cells"] = len(testcases_dl)
    payload["slots_per_pattern"] = args.pattern_len

    priorities, start_delay, tv_overrides, cumac_options = load_tdd_yaml_overrides(getattr(args, "yaml", None), logger)

    if priorities is None:
        with open("measure/TDD/priorities.json", "r", encoding="utf-8") as ifile:
            priorities = json.load(ifile)

    buffer = {}

    for key in priorities.keys():
        label = "_".join([key, "PRIO"])
        buffer[label] = priorities[key]

    payload.update(buffer)

    channels = []

    for sweep_idx in range(args.sweeps):

        channel = {}

        if not args.is_no_pdsch and sLotConfig["PDSCH"][sweep_idx % args.pattern_len]:
            channel["PDSCH"] = expand_tvs_for_cells(
                filenames_dl, testcases_dl, args.vfld
            )

        if testcases_cdl is not None and sLotConfig["PDCCH"][sweep_idx % args.pattern_len]:
            channel["PDCCH"] = expand_tvs_for_cells(
                filenames_cdl, testcases_cdl, args.vfld
            )

        if testcases_cr is not None and sLotConfig["CSIRS"][sweep_idx % args.pattern_len]:
            channel["CSIRS"] = expand_tvs_for_cells(
                filenames_cr, testcases_cr, args.vfld
            )

        if testcases_dlbf is not None and sLotConfig["PDSCH"][sweep_idx % args.pattern_len]:
            channel["DLBFW"] = expand_tvs_for_cells(
                filenames_dlbf, testcases_dlbf, args.vfld
            )

        if testcases_ssb is not None and sLotConfig["PBCH"][sweep_idx % args.pattern_len]:
            channel["SSB"] = expand_tvs_for_cells(
                filenames_ssb, testcases_ssb, args.vfld
            )

        if testcases_mac is not None and sLotConfig["MAC"][sweep_idx % args.pattern_len]:
            for testcase in testcases_mac:
                # Get the cumac TV name based on actual cell count
                # Example: in the input json config file, cumac TV defined as "TV_cumac_F08-MC-CC-8PC.h5". For test of 16 cells, we need to replace the "8" with 16, i.e., "TV_cumac_F08-MC-CC-16PC.h5"
                
                # Search for the default cell count in the TV name, using pattern '-(\d+)PC'
                macTvName = filenames_mac[testcase]
                matches = list(re.finditer(r'-(\d+)PC', macTvName))

                # If a cell count match is found, replace it with actual cell count
                if matches:
                    # Get the last match of '-(\d+)PC'
                    last_match = matches[-1]
                    # Replace the default cell count in TV name with actual cell count
                    macTvName = macTvName[:last_match.start(1)] + str(payload["cells"]) + macTvName[last_match.end(1):]
                    channel["MAC"] = [os.path.join(args.vfld, macTvName)]
                else:
                    raise ValueError("No match of cell count in cuMAC TV name")

        if testcases_mac2 is not None and args.mac2 > 0 and sLotConfig["MAC2"][sweep_idx % args.pattern_len]:
            for testcase in testcases_mac2:
                # Get the cumac TV name based on actual cell count
                # Example: in the input json config file, cumac TV defined as "TV_cumac_F08-MC-CC-8PC.h5". For test of 16 cells, we need to replace the "8" with 16, i.e., "TV_cumac_F08-MC-CC-16PC.h5"

                # Search for the default cell count in the TV name, using pattern '-(\d+)PC'
                mac2TvName = filenames_mac2[testcase]
                matches = list(re.finditer(r'-(\d+)PC', mac2TvName))

                # If a cell count match is found, replace it with actual cell count
                if matches:
                    # Get the last match of '-(\d+)PC'
                    last_match = matches[-1]
                    # Replace the default cell count in TV name with actual cell count
                    mac2TvName = mac2TvName[:last_match.start(1)] + str(args.mac2) + mac2TvName[last_match.end(1):] # mac2 using fixed number of cells
                    channel["MAC2"] = [os.path.join(args.vfld, mac2TvName)]
                else:
                    raise ValueError("No match of cell count in cuMAC2 TV name")

        if sweep_idx % args.pattern_len == 0:

            if not args.is_no_pusch:
                channel["PUSCH"] = expand_tvs_for_cells(
                    filenames_ul, testcases_ul, args.vfld
                )

            if testcases_ulbf is not None:
                channel["ULBFW"] = expand_tvs_for_cells(
                    filenames_ulbf, testcases_ulbf, args.vfld
                )

            if testcases_cul is not None:
                channel["PUCCH"] = expand_tvs_for_cells(
                    filenames_cul, testcases_cul, args.vfld
                )

            if testcases_ra is not None:
                channel["PRACH"] = expand_tvs_for_cells(
                    filenames_ra, testcases_ra, args.vfld
                )

            if testcases_sr is not None:
                channel["SRS"] = expand_tvs_for_cells(
                    filenames_sr, testcases_sr, args.vfld
                )

        if sweep_idx % args.pattern_len == 1:

            if not args.is_no_pusch:
                channel["PUSCH"] = expand_tvs_for_cells(
                    filenames_ul, testcases_ul, args.vfld
                )

            if testcases_ulbf is not None:
                channel["ULBFW"] = expand_tvs_for_cells(
                    filenames_ulbf, testcases_ulbf, args.vfld
                )

            if testcases_cul is not None:
                channel["PUCCH"] = expand_tvs_for_cells(
                    filenames_cul, testcases_cul, args.vfld
                )
                
        channels.append(channel)

    payload["slots"] = channels
    payload["parameters"] = extract(args, channels)
    # Keep optional YAML-only sections near the end for readability.
    if isinstance(start_delay, dict) and start_delay:
        payload["start_delay"] = start_delay

    if isinstance(cumac_options, dict) and cumac_options:
        payload["cumac_options"] = cumac_options

    cleaned_tv_overrides = drop_minus_one_overrides(tv_overrides)
    if isinstance(cleaned_tv_overrides, dict) and cleaned_tv_overrides:
        enabled = is_truthy(cleaned_tv_overrides.get("enable_override", False))
        meaningful_keys = [k for k in cleaned_tv_overrides.keys() if k != "enable_override"]
        if enabled and len(meaningful_keys) > 0:
            payload["override_test_vectors"] = cleaned_tv_overrides

    ofile = open(vectors, "w")
    yaml.dump(payload, ofile, sort_keys=False)
    ofile.close()

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

import numpy as np
import yaml

def calc_ontime_percentage(array, threshold):
    if len(array) == 0:
        return -1.0  # Set percentage to -1 if the array is empty
    else:
        ontimecount   = sum(1 for elem in array if elem <= threshold)
        ontimePercent = ontimecount / len(array)
        return ontimePercent

def parse_and_compare(s1, s2): # compare cell counts, sort by peak cell first
    # TODO: we currently use XY+00 for either peak or avg cell, support for mixed peak + avg cells to be added
    # "08+00" > "07+10" > "07+00"
    # Parse the strings into integers for comparison
    num1 = int(s1.split('+')[0]) * 100 + int(s1.split('+')[1])
    num2 = int(s2.split('+')[0]) * 100 + int(s2.split('+')[1])
    
    # Compare the parsed integers
    if num1 >= num2:
        return s1
    else:
        return s2


def _load_latency_budget_overrides(sweeps):
    test_config = sweeps.get("testConfig", {})
    yaml_path = test_config.get("yaml")
    if not yaml_path:
        return {}

    try:
        with open(yaml_path, "r", encoding="utf-8") as ifile:
            ycfg = yaml.safe_load(ifile) or {}
    except (OSError, UnicodeDecodeError):
        return {}
    except yaml.YAMLError:
        return {}

    if not isinstance(ycfg, dict):
        return {}

    latency_budget = None
    if isinstance(ycfg.get("latency_budget"), dict):
        latency_budget = ycfg["latency_budget"]
    elif isinstance(ycfg.get("config"), dict) and isinstance(
        ycfg["config"].get("latency_budget"), dict
    ):
        latency_budget = ycfg["config"]["latency_budget"]

    if not isinstance(latency_budget, dict):
        return {}

    alias_map = {
        "PDSCH_DLBFW": "PDSCH+DLBFW",
        "PDCCH_CSIRS": "PDCCH+CSI-RS",
        "CSIRS": "CSI-RS",
    }

    overrides = {}
    for key, value in latency_budget.items():
        if not isinstance(value, (int, float)):
            continue
        norm_key = alias_map.get(key, key)
        overrides[norm_key] = value

    return overrides


def check_cell_capacity(sweeps):

    usecase = sweeps['testConfig']['config'][-8:-5]
    mMIMO_pattern = (sweeps['testConfig']['pattern'] == "dddsuudddd_mMIMO")
    supported_patterns = {"dddsuudddd", "dddsuudddd_8slot", "dddsuudddd_mMIMO"}
    latency_budget_overrides = _load_latency_budget_overrides(sweeps)
    has_custom_latency_budget = len(latency_budget_overrides) > 0
    ontimePercent = {}
    cellCapacity  = "00+00"
    maxCellTested = "00+00"   
    successRunInd = False # whether a success run has been performed
    if sweeps['testConfig']['pattern'] not in supported_patterns:
        supported = ", ".join(sorted(supported_patterns))
        print(
            f"Auto check cell capacity only supports TDD long patterns: {supported}. "
            f"The test is using {sweeps['testConfig']['pattern']}"
        )
    elif usecase not in ["F08", "F09", "F14"] and not has_custom_latency_budget:
        print(f"Auto check cell capacity only supports F08, F09, F14. The test is using {usecase}")
    else:
        # (i) gather latency constraints (ii) check on time percentage per channel (iii) obtain cell capacity
        def budget(name, default):
            return latency_budget_overrides.get(name, default)

        # set latency budget, according to compare.py (used for drawing CDF)
        constraints = {}

        # hard coded latency constraints
        constraints['ULBFW1'] = budget("ULBFW1", 615)
        constraints['ULBFW2'] = budget("ULBFW2", 615)
        constraints['PUSCH1_SUBSLOT_PROC'] = budget("PUSCH1_SUBSLOT_PROC", 755)
        constraints['PUSCH1'] = budget("PUSCH1", 2000 if mMIMO_pattern else 1500)
        constraints['PUCCH1'] = budget("PUCCH1", 2000 if mMIMO_pattern else 1500)
        constraints['PUSCH2_SUBSLOT_PROC'] = budget("PUSCH2_SUBSLOT_PROC", 755)
        constraints['PUSCH2'] = budget("PUSCH2", 1855 if mMIMO_pattern else 1500)
        constraints['PUCCH2'] = budget("PUCCH2", 3000 if mMIMO_pattern else 1500)
        constraints['SRS1'] = budget("SRS1", 2500 if mMIMO_pattern else 500)
        constraints['SRS2'] = budget("SRS2", 2500 if mMIMO_pattern else 500)
        constraints['SSB'] = budget("SSB", 200 if mMIMO_pattern else 375)
        constraints['MAC'] = budget("MAC", 500)
        constraints['MAC2'] = budget("MAC2", 500)
                
        for key in sweeps.keys():
            if '+' in key:
                
                if mMIMO_pattern: # mMIMO new pattern
                    
                    constraints['PRACH'] = budget("PRACH", 3000)
                    constraints['DLBFW'] = budget("DLBFW", 250)
                    constraints['PDSCH'] = budget("PDSCH", 300)
                    constraints['PDCCH'] = budget("PDCCH", 300)
                    constraints['CSI-RS'] = budget("CSI-RS", 300)
                
                else:

                    # latency constraints based on test configs
                    if sweeps[key].get("DLBFW", None) is not None:
                        constraints['PDSCH+DLBFW'] = budget("PDSCH+DLBFW", 500)
                    else:
                        constraints['PDSCH'] = budget("PDSCH", 375)
                    
                    if sweeps[key].get("CSI-RS", None) is not None:
                        constraints['PDCCH+CSI-RS'] = budget("PDCCH+CSI-RS", 375)
                    else:
                        constraints['PDCCH'] = budget("PDCCH", 375)

                    if sweeps[key]['Mode'] == "Sequential":
                        constraints['PRACH'] = budget("PRACH", 1500)
                    elif sweeps[key]['Mode'] == "Parallel":
                        constraints['PRACH'] = budget("PRACH", 1250)
                    else:
                        constraints['PRACH'] = budget("PRACH", 2000)
            
                # extact latency from tested results and compare
                ontimePercent[key] = {}
                
                # PUSCH
                pusch1_subslotProc = sweeps[key].get("PUSCH1_SUBSLOT_PROC", [])
                ontimePercent[key]['PUSCH1_SUBSLOT_PROC'] = calc_ontime_percentage(pusch1_subslotProc, constraints['PUSCH1_SUBSLOT_PROC'])

                pusch1 = sweeps[key].get("PUSCH1", [])
                ontimePercent[key]['PUSCH1'] = calc_ontime_percentage(pusch1, constraints['PUSCH1'])

                pusch2_subslotProc = sweeps[key].get('PUSCH2_SUBSLOT_PROC', [])
                ontimePercent[key]['PUSCH2_SUBSLOT_PROC'] = calc_ontime_percentage(pusch2_subslotProc, constraints['PUSCH2_SUBSLOT_PROC'])
                
                pusch2 = sweeps[key].get("PUSCH2", [])
                ontimePercent[key]['PUSCH2'] = calc_ontime_percentage(pusch2, constraints['PUSCH2'])

                # ULBFW
                ulbfw2 = sweeps[key].get("ULBFW2", [])
                ontimePercent[key]['ULBFW2'] = calc_ontime_percentage(ulbfw2, constraints['ULBFW2'])

                # PUCCH
                pucch1 = sweeps[key].get("PUCCH1", [])
                pucch2 = sweeps[key].get("PUCCH2", [])
                ontimePercent[key]['PUCCH1'] = calc_ontime_percentage(pucch1, constraints['PUCCH1'])
                ontimePercent[key]['PUCCH2'] = calc_ontime_percentage(pucch2, constraints['PUCCH2'])
                
                # PRACH
                prach = sweeps[key].get("PRACH", [])
                ontimePercent[key]['PRACH'] = calc_ontime_percentage(prach, constraints['PRACH'])
                
                if mMIMO_pattern: # mMIMO new pattern
                    # DLBFW
                    if sweeps[key].get("DLBFW", None) is not None:
                        dlbfw = sweeps[key].get("DLBFW", [])
                        ontimePercent[key]['DLBFW'] = calc_ontime_percentage(dlbfw, constraints['DLBFW'])
                    
                    # PDSCH
                    pdsch = sweeps[key].get("PDSCH", [])
                    ontimePercent[key]['PDSCH'] = calc_ontime_percentage(pdsch, constraints['PDSCH'])
                    
                    # PDCCH
                    pdcch = sweeps[key].get("PDCCH", [])
                    ontimePercent[key]['PDCCH'] = calc_ontime_percentage(pdcch, constraints['PDCCH'])
                        
                    # CSI-RS          
                    csirs = sweeps[key].get("CSI-RS", [])
                    ontimePercent[key]['CSI-RS'] = calc_ontime_percentage(csirs, constraints['CSI-RS'])
                    
                else:
                    # PDSCH or PDSCH+DLBFW          
                    if sweeps[key].get("DLBFW", None) is not None:
                        pdsch_dlbfw = sweeps[key].get("PDSCH", []) + sweeps[key].get("DLBFW", [])
                        ontimePercent[key]['PDSCH+DLBFW'] = calc_ontime_percentage(pdsch_dlbfw, constraints['PDSCH+DLBFW'])
                    else:
                        pdsch = sweeps[key].get("PDSCH", [])
                        ontimePercent[key]['PDSCH'] = calc_ontime_percentage(pdsch, constraints['PDSCH'])
                        
                    # PDCCH or PDCCH+CSI-RS          
                    if sweeps[key].get("CSI-RS", None) is not None:
                        pdcch_csirs = sweeps[key].get("PDCCH", []) + sweeps[key].get("CSI-RS", [])
                        ontimePercent[key]['PDCCH+CSI-RS'] = calc_ontime_percentage(pdcch_csirs, constraints['PDCCH+CSI-RS'])
                    else:
                        pdcch = sweeps[key].get("PDCCH", [])
                        ontimePercent[key]['PDCCH'] = calc_ontime_percentage(pdcch, constraints['PDCCH'])
                
                # SSB
                ssb = sweeps[key].get("SSB", [])
                ontimePercent[key]['SSB'] = calc_ontime_percentage(ssb, constraints['SSB'])
                
                # SRS1
                srs1 = sweeps[key].get("SRS1", [])
                ontimePercent[key]['SRS1'] = calc_ontime_percentage(srs1, constraints['SRS1'])
                
                # SRS2
                srs2 = sweeps[key].get("SRS2", [])
                ontimePercent[key]['SRS2'] = calc_ontime_percentage(srs2, constraints['SRS2'])

                # MAC
                mac = sweeps[key].get("MAC", [])
                ontimePercent[key]['MAC'] = calc_ontime_percentage(mac, constraints['MAC'])

                # MAC2
                mac2 = sweeps[key].get("MAC2", [])
                ontimePercent[key]['MAC2'] = calc_ontime_percentage(mac2, constraints['MAC2'])
                
                # check if ontimePercent is 100% for all channels except the not tested SRS (has negative value
                ontimePercent[key] = {key: value for key, value in ontimePercent[key].items() if value >= 0}
                onTimeArray = np.array(list(ontimePercent[key].values()))

                successRunInd = successRunInd or (len(ontimePercent[key]) > 0)
                if(np.all(onTimeArray == 1.0) and (len(ontimePercent[key]) > 0)):
                    cellCapacity = parse_and_compare(cellCapacity, key)

                # save to tested results
                sweeps[key]['ontimePercent'] = ontimePercent[key]
                
                # update maxCellTested
                maxCellTested = parse_and_compare(maxCellTested, key)
        
        sweeps['constraints'] = constraints # save constraints into result json file
        
        if(not successRunInd):
            print("Warning: no successful run is done, unknown cell capacity (100% on time for all channels), please retry")
        elif(cellCapacity == maxCellTested):
            print(f"Warning: max cell count {maxCellTested} passed based on {sweeps['testConfig']['sweeps']} slots run, unknown cell capacity (100% on time for all channels), please try larger cell counts")
        elif (cellCapacity == "00+00"):
            print(f"Warning: no cell count passed based on {sweeps['testConfig']['sweeps']} slots run, unknown cell capacity (100% on time for all channels), please try smaller cell counts")
        else:
            print(f"Cell capacity is {cellCapacity} based on {sweeps['testConfig']['sweeps']} slots run (100% on time for all channels)")  
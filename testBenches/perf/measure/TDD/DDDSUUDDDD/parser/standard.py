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

from .common import _has_dl_bf, _has_srs, check, unpack


def run(args, lines):

    latencies = {}
    clat_ul1 = []
    clat_ul1_subslotProc = []
    clat_ul2 = []
    clat_ul2_subslotProc = []
    clat_dl = []
    clat_dlbf = []
    clat_ulbf1 = []
    clat_ulbf2 = []
    clat_sr1 = []
    clat_sr2 = []
    clat_ra = []
    clat_cdl = []
    clat_cul1 = []
    clat_cul2 = []
    clat_ssb = []
    clat_cr = []
    clat_mac = []
    clat_mac2 = []

    # count GPU memory usage
    memoryUseMB={}
    memoryUseMB["PUSCH"] = 0
    memoryUseMB["PUCCH"] = 0
    memoryUseMB["SRS"] = 0
    memoryUseMB["PRACH"] = 0
    
    memoryUseMB["PDSCH"] = 0
    memoryUseMB["PDCCH"] = 0
    memoryUseMB["SSB"] = 0
    memoryUseMB["CSIRS"] = 0
    
    # TODO: beamforming GPU memory, BFW does not separate DL and UL
    memoryUseMB["BFW"] = 0
     
    if args.is_pusch_cascaded:
        mode = "Sequential"
    else:
        mode = "Parallel"

    k = 0

    is_correct = False

    #### NOTE: this log is the stdout from C testbench. It's typically saved in buffer-XX.txt (XX = cell count) when running from Python interface
    ####       this log may not be up-to-date with the latest changes to the C code
    # -----------------------------------------------------------
    # 22:38:08.856418 WRN 3511 0 [CUPHY.MEMFOOT] cuphyMemoryFootprint - GPU allocation: 0.364 MiB for cuPHY PRACH channel object (0x7f38a4bc8440).
    # 22:38:08.864889 WRN 3513 0 [CUPHY.MEMFOOT] cuphyMemoryFootprint - GPU allocation: 21.245 MiB for cuPHY PUCCH channel object (0x7f3884abee90).
    
    # -----------------------------------------------------------
    # Slot pattern # 0
    # average slot pattern run time: 3702.40 us (averaged over 1 iterations) 
    # Average PUSCH run time: 1635.14 us from 548.48 (averaged over 1 iterations) 
    # Average PUSCH2 run time: 2675.65 us from 1640.29 (averaged over 1 iterations) 
    # Average PUCCH run time: 772.35 us from 561.63 (averaged over 1 iterations) 
    # Average PUCCH2 run time: 1828.93 us from 1646.50 (averaged over 1 iterations) 
    # Average PRACH run time: 1783.65 us from 1644.10 (averaged over 1 iterations) 
    # Slot # 0: average PDCCH run time: 165.28 us from 14.59 (averaged over 1 iterations) 
    # Slot # 1: average PDCCH run time: 664.26 us from 518.85 (averaged over 1 iterations) 
    # Slot # 2: average PDCCH run time: 1197.47 us from 1021.95 (averaged over 1 iterations) 
    # Slot # 3: average PDCCH run time: 1669.38 us from 1516.38 (averaged over 1 iterations) 
    # Slot # 4: average PDCCH run time: 2191.36 us from 2016.67 (averaged over 1 iterations) 
    # Slot # 5: average PDCCH run time: 2642.34 us from 2513.12 (averaged over 1 iterations) 
    # Slot # 6: average PDCCH run time: 3152.26 us from 3014.62 (averaged over 1 iterations) 
    # Slot # 7: average PDCCH run time: 3650.40 us from 3513.44 (averaged over 1 iterations) 
    # Slot # 0: average CSIRS run time: 199.74 us from 174.56 (averaged over 1 iterations) 
    # Slot # 1: average CSIRS run time: 690.78 us from 669.86 (averaged over 1 iterations) 
    # Slot # 2: average CSIRS run time: 1237.95 us from 1205.76 (averaged over 1 iterations) 
    # Slot # 3: average CSIRS run time: 1697.25 us from 1674.59 (averaged over 1 iterations) 
    # Slot # 4: average CSIRS run time: 2226.62 us from 2196.80 (averaged over 1 iterations) 
    # Slot # 5: average CSIRS run time: 2667.20 us from 2647.01 (averaged over 1 iterations) 
    # Slot # 6: average CSIRS run time: 3173.63 us from 3156.32 (averaged over 1 iterations) 
    # Slot # 7: average CSIRS run time: 3666.98 us from 3652.06 (averaged over 1 iterations) 
    # Slot # 0: average PDSCH run time: 129.73 us from 2.72 (averaged over 1 iterations) 
    # Slot # 1: average PDSCH run time: 632.58 us from 499.94 (averaged over 1 iterations) 
    # Slot # 2: average PDSCH run time: 1181.95 us from 1004.77 (averaged over 1 iterations) 
    # Slot # 3: average PDSCH run time: 1651.68 us from 1500.42 (averaged over 1 iterations) 
    # Slot # 4: average PDSCH run time: 2165.31 us from 1998.40 (averaged over 1 iterations) 
    # Slot # 5: average PDSCH run time: 2658.72 us from 2497.86 (averaged over 1 iterations) 
    # Slot # 6: average PDSCH run time: 3123.68 us from 2999.17 (averaged over 1 iterations) 
    # Slot # 7: average PDSCH run time: 3621.47 us from 3498.50 (averaged over 1 iterations) 
    # Slot # 4: average SSB   run time: 2060.83 us from 2022.85 (averaged over 1 iterations) 
    # Slot # 5: average SSB   run time: 2581.95 us from 2521.95 (averaged over 1 iterations) 
    # Slot # 6: average SSB   run time: 3061.28 us from 3023.65 (averaged over 1 iterations) 
    # Slot # 7: average SSB   run time: 3561.92 us from 3522.53 (averaged over 1 iterations) 

    while k < len(lines):
        lst = lines[k].split()

        # currently only GPU memory are traced
        if(len(lst) == 17 and lst[5] == 'cuphyMemoryFootprint' and lst[7] == 'GPU'):
            memoryUseMB[lst[13]] += float(lst[9])
            
        elif len(lst) == 4 and lst[0] == "Slot" and lst[1] == "pattern" and lst[2] == "#":

            k += 1

            pusch1 = 0
            pusch1_subslotProc = 0
            pusch2 = 0
            pusch2_subslotProc = 0
            pdsch = []
            dlbfw = []
            ulbfw1 = []
            ulbfw2 = []
            srs1 = 0
            srs2 = 0
            rach = 0
            pdcch = []
            pucch1 = 0
            pucch2 = 0
            ssb = []
            csirs = []
            mac = []
            mac2 = []

            while k < len(lines) and "----" not in lines[k]:
                lst = lines[k].split()

                if len(lst) > 3: # Uplink channels: use end - start for time measurement

                    if lst[0] == "Average" and lst[1] == "PUSCH" and lst[2] == "run":
                        pusch1 = float(lst[4]) - float(lst[7])
                        k += 1
                        continue

                    if lst[0] == "Average" and lst[1] == "PUSCH_subslotProc" and lst[2] == "run":
                        pusch1_subslotProc = float(lst[4]) - float(lst[7])
                        k += 1
                        continue

                    if lst[0] == "Average" and lst[1] == "PUSCH2" and lst[2] == "run":
                        pusch2 = float(lst[4]) - float(lst[7])
                        k += 1
                        continue

                    if lst[0] == "Average" and lst[1] == "PUSCH2_subslotProc" and lst[2] == "run":
                        pusch2_subslotProc = float(lst[4]) - float(lst[7])
                        k += 1
                        continue

                    if lst[0] == "Average" and lst[1] == "PUCCH" and lst[2] == "run":
                        pucch1 = float(lst[4]) - float(lst[7])
                        k += 1
                        continue

                    if lst[0] == "Average" and lst[1] == "PUCCH2" and lst[2] == "run":
                        pucch2 = float(lst[4]) - float(lst[7])
                        k += 1
                        continue

                    if lst[0] == "Average" and lst[1] == "ULBFW" and lst[2] == "run":
                        ulbfw1 = float(lst[4]) - float(lst[7])
                        k += 1
                        continue
                    
                    if lst[0] == "Average" and lst[1] == "ULBFW2" and lst[2] == "run":
                        ulbfw2 = float(lst[4]) - float(lst[7])
                        k += 1
                        continue
                    
                    if lst[0] == "Average" and lst[1] == "SRS1" and lst[2] == "run":
                        srs1 = float(lst[4]) - float(lst[7])
                        k += 1
                        continue

                    if lst[0] == "Average" and lst[1] == "SRS2" and lst[2] == "run":
                        srs2 = float(lst[4]) - float(lst[7])
                        k += 1
                        continue

                    if lst[0] == "Average" and lst[1] == "PRACH" and lst[2] == "run":
                        rach = float(lst[4]) - float(lst[7])
                        k += 1
                        continue
                
                if len(lst) > 4:  # Downlink channels: use end - start of slot boundaries for time measurement, also add timeslot index for checking start times
                    
                    if lst[0] == "Slot" and lst[4] == "PDSCH":
                        pdsch.append([float(lst[10]), float(lst[7]), float(lst[2][:-1])])
                        k += 1
                        continue

                    if lst[0] == "Slot" and lst[4] == "PDCCH":
                        pdcch.append([float(lst[10]), float(lst[7]), float(lst[2][:-1])])
                        k += 1
                        continue

                    if lst[0] == "Slot" and lst[4] == "CSIRS":
                        csirs.append([float(lst[10]), float(lst[7]), float(lst[2][:-1])])
                        k += 1
                        continue

                    if lst[0] == "Slot" and lst[4] == "SSB":
                        ssb.append([float(lst[10]), float(lst[7]), float(lst[2][:-1])])
                        k += 1
                        continue

                    if lst[0] == "Slot" and lst[4] == "DLBFW":
                        dlbfw.append([float(lst[10]), float(lst[7]), float(lst[2][:-1])])
                        k += 1
                        continue

                    if lst[0] == "Slot" and lst[4] == "MAC":
                        mac.append(float(lst[7]) - float(lst[10])) # calculate MAC latency here, not in unpack
                        k += 1
                        continue

                    if lst[0] == "Slot" and lst[4] == "MAC2":
                        mac2.append(float(lst[7]) - float(lst[10])) # calculate MAC2 latency here, not in unpack
                        k += 1
                        continue

                k += 1

            is_correct = check(
                args,
                pusch1,
                pusch1_subslotProc,
                pusch2,
                pusch2_subslotProc,
                pdsch,
                dlbfw,
                ulbfw1,
                ulbfw2,
                srs1,
                srs2,
                rach,
                pdcch,
                pucch1,
                pucch2,
                ssb,
                csirs,
            )

            (
                pusch1,
                pusch1_subslotProc,
                pusch2,
                pusch2_subslotProc,
                pdsch,
                dlbfw,
                ulbfw1,
                ulbfw2,
                srs1,
                srs2,
                rach,
                pdcch,
                pucch1,
                pucch2,
                ssb,
                csirs,
            ) = unpack(
                args,
                pusch1,
                pusch1_subslotProc,
                pusch2,
                pusch2_subslotProc,
                pdsch,
                dlbfw,
                ulbfw1,
                ulbfw2,
                srs1,
                srs2,
                rach,
                pdcch,
                pucch1,
                pucch2,
                ssb,
                csirs,
            )

            if pusch1 > 0:
                clat_ul1.append(pusch1)
            if pusch1_subslotProc > 0:
                clat_ul1_subslotProc.append(pusch1_subslotProc)
            if pusch2 > 0:
                clat_ul2.append(pusch2)
            if pusch2_subslotProc > 0:
                clat_ul2_subslotProc.append(pusch2_subslotProc)

            if pucch1 > 0:
                clat_cul1.append(pucch1)
            if pucch2 > 0:
                clat_cul2.append(pucch2)

            clat_ssb.extend(ssb)

            clat_dl.extend(pdsch)
            clat_cdl.extend(pdcch)
            clat_cr.extend(csirs)
            # Split what is_rec_bf used to gate together: DL-BFW vs SRS vs UL-BFW.
            if _has_dl_bf(args):
                clat_dlbf.extend(dlbfw)
                if args.is_rec_bf:
                    clat_ulbf1.append(ulbfw1)
                    clat_ulbf2.append(ulbfw2)
            if _has_srs(args):
                clat_sr1.append(srs1)
                if srs2 > 0:
                    clat_sr2.append(srs2)

            if args.is_prach:
                if rach > 0:
                    clat_ra.append(rach)

            if args.is_mac:
                clat_mac += mac # mac is per pattern latency, clat_mac is all slots latency

            if args.mac2 > 0:
                clat_mac2 += mac2 # mac2 is per pattern latency, clat_mac2 is all slots latency

        k += 1

    latencies["Structure"] = is_correct
    latencies["Mode"] = mode
    latencies["PDSCH"] = clat_dl
    latencies["PUSCH1"] = clat_ul1
    latencies["PUSCH1_SUBSLOT_PROC"] = clat_ul1_subslotProc
    latencies["PUSCH2"] = clat_ul2
    latencies["PUSCH2_SUBSLOT_PROC"] = clat_ul2_subslotProc

    if args.is_pdcch:
        latencies["PDCCH"] = clat_cdl

    if args.is_pucch:
        latencies["PUCCH1"] = clat_cul1
        latencies["PUCCH2"] = clat_cul2

    # UL-BFW remains reciprocal-BF-only; DL-BFW/SRS also available in isolation modes.
    if _has_dl_bf(args):
        latencies["DLBFW"] = clat_dlbf
    if args.is_rec_bf:
        latencies["ULBFW1"] = clat_ulbf1
        latencies["ULBFW2"] = clat_ulbf2
    if _has_srs(args):
        latencies["SRS1"] = clat_sr1
        latencies["SRS2"] = clat_sr2

    if args.is_prach:
        latencies["PRACH"] = clat_ra

    if args.is_ssb:
        latencies["SSB"] = clat_ssb

    if args.is_csirs:
        latencies["CSI-RS"] = clat_cr

    if args.is_mac:
        latencies["MAC"] = clat_mac

    if args.mac2 > 0:
        latencies["MAC2"] = clat_mac2

    # calculate all memory usage
    memoryUseMB["totalUlNoBFW"] = memoryUseMB["PUSCH"] + memoryUseMB["PUCCH"] + memoryUseMB["SRS"] + memoryUseMB["PRACH"]
    memoryUseMB["totalDlNoBFW"] = memoryUseMB["PDSCH"] + memoryUseMB["PDCCH"] + memoryUseMB["SSB"] + memoryUseMB["CSIRS"]
    memoryUseMB["totalNoBFW"] = memoryUseMB["totalUlNoBFW"] + memoryUseMB["totalDlNoBFW"]
    memoryUseMB["total"] = memoryUseMB["totalNoBFW"] + memoryUseMB["BFW"]
    
    results = latencies
    results['memoryUseMB'] = memoryUseMB
    
    return results

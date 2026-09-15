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

import argparse

import numpy as np


# --rec_bf bundles DL-BFW, SRS, and UL-BFW. Channel-only benchmarking needs
# DL-BFW or SRS in isolation; getattr keeps non-cli.py callers backward-compatible.
def _has_dl_bf(args: argparse.Namespace) -> bool:
    return args.is_rec_bf or getattr(args, "is_dl_bf_only", False)


def _has_srs(args: argparse.Namespace) -> bool:
    return args.is_rec_bf or getattr(args, "is_srs_only", False)


def check(
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
):

    if args.pattern != "dddsuudddd_mMIMO":
        if len(pdsch) < 8:
            return False

        offset = 0

        if _has_dl_bf(args):
            offset = 1

        for idx, itm in enumerate(pdsch):

            if np.abs(itm[0] - (itm[2] + offset) * 500) > 50:
                return False

        for idx, itm in enumerate(pdcch):

            if np.abs(itm[0] - (itm[2] + offset) * 500) > 50:
                return False
    
    else: 
        
        pdsch_pdcch_slot_offset = 225 # there is a 225 us delay for the start of PDSCH/PDCCH
        
        for idx, itm in enumerate(pdsch):

            if np.abs(itm[0] - itm[2] * 500 - pdsch_pdcch_slot_offset) > 50:
                return False

        for idx, itm in enumerate(pdcch):

            if np.abs(itm[0] - itm[2] * 500 - pdsch_pdcch_slot_offset) > 50:
                return False
            
    return True


def unpack(
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
):

    new_pdsch = []
    new_pdcch = []
    new_csirs = []
    new_dlbfw = []
    new_ssb = []

    if args.is_pusch_cascaded:

        if _has_dl_bf(args):

            for idx, itm in enumerate(pdsch):

                new_pdsch.append(itm[1] - itm[0])

            for idx, itm in enumerate(pdcch):

                new_pdcch.append(itm[1] - itm[0])

            for idx, itm in enumerate(csirs):

                new_csirs.append(itm[1] - itm[0])

            for idx, itm in enumerate(dlbfw):

                new_dlbfw.append(itm[1] - itm[0]) # use DLBFW standalone time measurement

            for idx, itm in enumerate(ssb):
                new_ssb.append(itm[1] - itm[0])

        else:

            for idx, itm in enumerate(pdsch):

                new_pdsch.append(itm[1] - itm[0])

            for idx, itm in enumerate(pdcch):

                new_pdcch.append(itm[1] - itm[0])

            for idx, itm in enumerate(csirs):

                new_csirs.append(itm[1] - itm[0])

            for idx, itm in enumerate(ssb):
                new_ssb.append(itm[1] - itm[0])

    else:
        # For better latency, we typically use the above is_pusch_cascaded = True
        # This avoids two PUSCH workloads to run together
        # For 4T4R (is_rec_bf = False) slot 0 doesn't have anything scheduled, simulation start at the begiing of slot 1; subtract 500 us for PUSCH2 delay
        # For 32T32TR (is_rec_bf or is_dl_bf_only) slot 0 has workload, simulation start at the begiing of slot 0; subtract 500 us for PUSCH1 delay, subtract 1000 for PUSCH2 delay
        if _has_dl_bf(args):

            for idx, itm in enumerate(pdsch):

                new_pdsch.append(itm[1] - itm[0])

            for idx, itm in enumerate(pdcch):

                new_pdcch.append(itm[1] - itm[0])

            for idx, itm in enumerate(csirs):

                new_csirs.append(itm[1] - itm[0])

            for idx, itm in enumerate(dlbfw):

                new_dlbfw.append(itm[1] - itm[0])
                
            for idx, itm in enumerate(ssb):
                new_ssb.append(itm[1] - itm[0])

        else:

            for idx, itm in enumerate(pdsch):

                new_pdsch.append(itm[1] - itm[0])

            for idx, itm in enumerate(pdcch):

                new_pdcch.append(itm[1] - itm[0])

            for idx, itm in enumerate(csirs):

                new_csirs.append(itm[1] - itm[0])

            for idx, itm in enumerate(ssb):
                new_ssb.append(itm[1] - itm[0])

    return (
        pusch1,
        pusch1_subslotProc,
        pusch2,
        pusch2_subslotProc,
        new_pdsch,
        new_dlbfw,
        ulbfw1,
        ulbfw2,
        srs1,
        srs2,
        rach,
        new_pdcch,
        pucch1,
        pucch2,
        new_ssb,
        new_csirs,
    )

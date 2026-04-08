#!/usr/bin/env python3

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

import pandas as pd
import numpy as np
from aerial_postproc.logparse import CICD_NONE_FORMAT

def retrieve_perf_metrics(perf_filename):
    df = pd.read_csv(perf_filename, dtype=np.float64, na_values=CICD_NONE_FORMAT)

    dlu_max_col = 'tick_to_dlu_completion'
    if not dlu_max_col in df.columns:
        #Allow this script to be compatible with the legacy format
        dlu_max_col = 'tick_to_dl_l1_completion'
    dlu_max_idx = df[dlu_max_col].idxmax()
    dlu_max = df[dlu_max_col].max()
    
    # Handle separate BFW and non-BFW DLC completion times
    if 'tick_to_dlc_bfw_completion' in df.columns and 'tick_to_dlc_nonbfw_completion' in df.columns:
        #Just use the BFW deadline for now
        dlc_max_idx = df.tick_to_dlc_bfw_completion.idxmax()
        dlc_max = df.tick_to_dlc_bfw_completion.max()
    else:
        # Fallback to old single field for backward compatibility
        dlc_max_idx = df.tick_to_dlc_completion.idxmax()
        dlc_max = df.tick_to_dlc_completion.max()
    
    # Handle separate BFW and non-BFW ULC completion times
    if 'tick_to_ulc_bfw_completion' in df.columns and 'tick_to_ulc_nonbfw_completion' in df.columns:
        #Just use the BFW deadline for now
        ulc_max_idx = df.tick_to_ulc_bfw_completion.idxmax()
        ulc_max = df.tick_to_ulc_bfw_completion.max()
    else:
        # Fallback to old single field for backward compatibility
        ulc_max_idx = df.tick_to_ulc_completion.idxmax()
        ulc_max = df.tick_to_ulc_completion.max()
        
    ul_pusch4_eh = df[df.slot%10==4].t0_to_pusch_eh_completion.max()
    ul_pucch4 = df[df.slot%10==4].t0_to_pucch_completion.max()
    ul_pusch4 = df[df.slot%10==4].t0_to_pusch_completion.max()
    ul_pusch5_eh = df[df.slot%10==5].t0_to_pusch_eh_completion.max()
    ul_pucch5 = df[df.slot%10==5].t0_to_pucch_completion.max()
    ul_pusch5 = df[df.slot%10==5].t0_to_pusch_completion.max()
    ul_prach = df.t0_to_prach_completion.max()
    ul_srs = df.t0_to_srs_completion.max()

    return df, {
        'dlu_max_idx': dlu_max_idx,
        'dlu_max'    : dlu_max,
        'dlc_max_idx': dlc_max_idx,
        'dlc_max'    : dlc_max,
        'ulc_max_idx': ulc_max_idx,
        'ulc_max'    : ulc_max,
        'ul_pusch4_eh': ul_pusch4_eh,
        'ul_pucch4'   : ul_pucch4,
        'ul_pusch4'   : ul_pusch4,
        'ul_pusch5_eh': ul_pusch5_eh,
        'ul_pucch5'   : ul_pucch5,
        'ul_pusch5'   : ul_pusch5,
        'ul_prach'    : ul_prach,
        'ul_srs'      : ul_srs
    }

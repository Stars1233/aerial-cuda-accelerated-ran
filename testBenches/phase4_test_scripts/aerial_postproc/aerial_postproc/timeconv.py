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

"""Time conversion utilities.

The time conversion module contains functions and utilities for converting between
different time definitions, e.g. from TAI time to 3GPP SFN and Slot.
"""

import math
from typing import NamedTuple

# TODO:
# 1. 240 kHz SCS isn't fully supported.

# FIRST tick scheduled for: tick=1642091687480000000, seconds since
# epoch = 1642091687, nanoseconds = 480000000.

TAI_TO_GPS_OFFSET_NS = (315964800 + 19) * 1000000000
FRAME_PERIOD_NS = 10000000
SFN_MAX_PLUS1 = 1024
ORAN_FN_MAX_PLUS1 = 256


class Sfn(NamedTuple):
    """SFN type tuple (10-bit System Frame Number, slot)."""

    sfn: int
    slot: int


class OranFrameNumber(NamedTuple):
    """O-RAN frame number type tuple (8-bit frame number, subframe, slot)."""

    frame_number: int
    subframe: int
    slot: int


def slot_period_ns(mu: int = 1) -> int:
    """Returns 3GPP slot period in nanoseconds, as function of subcarrier spacing factor mu."""
    slot_period_array_ns = [1000000, 500000, 250000, 125000, 62500]
    return slot_period_array_ns[mu]


def slot_max_plus1(mu: int = 1) -> int:
    """Returns number of 3GPP slots per frame, as function of subcarrier spacing factor mu."""
    # TODO: There are 2.5 slots in SCS 240 kHz -- code doesn't handle this correctly.
    slot_max_plus1_array = [40, 20, 10, 5]
    return slot_max_plus1_array[mu]


def oran_slot_max_plus1(mu: int = 1) -> int:
    """Returns number of O-RAN slots per subframe, as function of subcarrier spacing factor mu."""
    oran_slot_max_plus1_array = [1, 2, 4, 8, 16]
    return oran_slot_max_plus1_array[mu]


def tai_to_sfn(tai_time_ns: int, gps_alpha: int, gps_beta: int, mu: int = 1) -> Sfn:
    """Convert TAI time to 3GPP SFN and slot."""
    assert mu == 1
    gps_time_ns = tai_time_ns - TAI_TO_GPS_OFFSET_NS
    gps_offset = int(((gps_beta * 1000000000) / 100) + ((gps_alpha * 10000) / 12288))
    numerator = gps_time_ns - gps_offset
    sfn = math.floor(numerator / FRAME_PERIOD_NS) % SFN_MAX_PLUS1
    slot = math.floor(numerator / slot_period_ns(mu)) % slot_max_plus1(mu)

    return Sfn(sfn, slot)


def sfn_to_tai(
    sfn: int,
    slot: int,
    approx_tai_time_ns: int,
    gps_alpha: int,
    gps_beta: int,
    mu: int = 1,
) -> int:
    """Convert 3GPP SFN and Slot to precise TAI time, given an approximate TAI timestamp."""
    # First, figure out the base SFN
    approx_gps_time_ns = approx_tai_time_ns - TAI_TO_GPS_OFFSET_NS
    gps_offset = int(((gps_beta * 1000000000) / 100) + ((gps_alpha * 10000) / 12288))
    full_wrap_period_ns = FRAME_PERIOD_NS * SFN_MAX_PLUS1
    half_wrap_period_adjust_ns = (
        full_wrap_period_ns / 2 - sfn * FRAME_PERIOD_NS - slot * slot_period_ns(mu)
    )

    base_gps_time_ns = (
        math.floor(
            (approx_gps_time_ns - gps_offset + half_wrap_period_adjust_ns)
            / full_wrap_period_ns
        )
        * full_wrap_period_ns
    )
    base_tai_time_ns = base_gps_time_ns + TAI_TO_GPS_OFFSET_NS

    return base_tai_time_ns + sfn * FRAME_PERIOD_NS + slot * slot_period_ns(mu)


def oran_fn_to_tai(
    oran_fn: int,
    oran_sf: int,
    oran_slot: int,
    approx_tai_time_ns: int,
    gps_alpha: int,
    gps_beta: int,
    mu: int = 1,
) -> int:
    """Convert ORAN SFN, subframe, and slot to precise TAI time."""
    # First, figure out the base FN
    approx_gps_time_ns = approx_tai_time_ns - TAI_TO_GPS_OFFSET_NS
    gps_offset = ((gps_beta * 1000000000) / 100) + ((gps_alpha * 10000) / 12288)
    full_wrap_period_ns = FRAME_PERIOD_NS * ORAN_FN_MAX_PLUS1
    half_wrap_period_adjust_ns = (
        full_wrap_period_ns / 2
        - oran_fn * FRAME_PERIOD_NS
        - (oran_sf * oran_slot_max_plus1(mu) + oran_slot) * slot_period_ns(mu)
    )

    base_gps_time_ns = (
        math.floor(
            (approx_gps_time_ns - gps_offset + half_wrap_period_adjust_ns)
            / full_wrap_period_ns
        )
        * full_wrap_period_ns
    )
    base_tai_time_ns = base_gps_time_ns + TAI_TO_GPS_OFFSET_NS

    return (
        base_tai_time_ns
        + oran_fn * FRAME_PERIOD_NS
        + (oran_sf * oran_slot_max_plus1(mu) + oran_slot) * slot_period_ns(mu)
    )


def sfn_to_oran_fn(sfn: int, slot: int, mu: int = 1) -> OranFrameNumber:
    """Convert frame numbering from 3GPP to ORAN.

    Convert 3GPP SFN and slot numbering convention to ORAN frame, subframe, and slot
    numbering convention.

    Args:
        sfn (int): 3GPP System Frame Number.
        slot (int): 3GPP slot.
        mu (int): NR numerology parameter mu.
    """
    oran_fn = sfn % ORAN_FN_MAX_PLUS1
    oran_sf = int(slot / oran_slot_max_plus1(mu))
    oran_slot = slot % oran_slot_max_plus1(mu)

    return OranFrameNumber(oran_fn, oran_sf, oran_slot)

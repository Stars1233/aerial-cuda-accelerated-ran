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

"""Chrome Trace Event Format exporter for cpu_timeline_plot.

Renders the same lane layout as ``plot_all_tasks_cpu_timeline`` into a JSON
trace loadable in https://ui.perfetto.dev or chrome://tracing.
"""

from __future__ import annotations

import dataclasses
import datetime as _datetime
import json
import os
import shlex
import sys
from types import MappingProxyType
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

__all__ = [
    "PerfettoTraceConfig",
    "RunInputs",
    "build_chrome_trace_events",
    "write_chrome_trace",
    "DEFAULT_CNAME_PALETTE",
]

# ---------------------------------------------------------------------------
# Module-level constants. All "magic numbers" referenced in the spec live here.
# ---------------------------------------------------------------------------

# pid layout per run. Each run gets a disjoint 100-pid block so multiple
# runs in a single JSON never collide:
#   pid_base + 0 → CPU Timeline process
#   pid_base + 1 → GPU Timeline process
#   pid_base + 2 → CPU Utilization process (counter container)
PID_BASE_PER_RUN: int = 100
PID_STRIDE_PER_RUN: int = 100

# CPU-side tids start at 10000 (10000 + sorted_yy_index) so they're visually
# distinct from any small-integer tid scheme. GPU tids start at 1 within the
# GPU process.
TID_GPU_KERNEL_BASE: int = 1

# Deadlines live in their own collapsible process per run (pid_base + 3).
# Each channel (DLC, DLU, ULC, ULU) is a separate thread; window-open and
# window-close events for that channel land on the same thread row.
TID_DEADLINE_DLC: int = 1
TID_DEADLINE_DLU: int = 2
TID_DEADLINE_ULC: int = 3
TID_DEADLINE_ULU: int = 4

# process_sort_index multiplier — each run's processes get a sort_index range
# of [run_idx * SORT_RUN_STRIDE, ...) so all of run 0's processes sort before
# any of run 1's, etc.
SORT_RUN_STRIDE: int = 1000

# Per-event tuning constants.
NS_PER_US: int = 1000
CPU_BUSY_BUCKET_US: int = 500
MAX_EVENT_NAME_CHARS: int = 200
MIN_EVENT_DURATION_US: float = 1.0
TRACE_SIZE_WARN_BYTES: int = 200 * 1024 * 1024  # 200 MB

# otherData.trace_features tags.
_FEATURE_SORT_INDICES: str = "sort_indices"
_FEATURE_DISPLAY_TIME_UNIT_NS: str = "display_time_unit_ns"
_FEATURE_OTHER_DATA: str = "other_data"
_FEATURE_CNAME: str = "cname"
_FEATURE_SFN_MARKERS: str = "sfn_markers"
_FEATURE_COUNTERS: str = "counters"
_FEATURE_DEADLINE_MARKERS: str = "deadline_markers"

# Default category-to-cname palette.
DEFAULT_CNAME_PALETTE: Mapping[str, str] = MappingProxyType({
    "cpu.dl":          "thread_state_running",
    "cpu.ul":          "olive",
    "cpu.debug":       "yellow",
    "cpu.dl_comms":    "thread_state_runnable",
    "cpu.dl_comp":     "thread_state_iowait",
    "cpu.l2":          "thread_state_uninterruptible",
    "cpu.testmac":     "terrible",
    "cpu.tick":        "grey",
    "cpu.other":       "generic_work",
    "gpu.pusch":       "thread_state_running",
    "gpu.pucch":       "olive",
    "gpu.prach":       "yellow",
    "gpu.order":       "thread_state_runnable",
    "gpu.srs_order":   "thread_state_runnable",
    "gpu.compression": "thread_state_iowait",
    "slot.sfn":        "yellow",
    "slot.boundary":   "thread_state_runnable",
    "deadline.dlc":    "light_memory_dump",
    "deadline.dlu":    "light_memory_dump",
    "deadline.ulc":    "light_memory_dump",
    "deadline.ulu":    "light_memory_dump",
})

# Per-slot deadline channels. Each tuple is
#   (channel_label, tid_within_deadlines_process, cat, variants)
# where ``variants`` is a tuple of ``(variant_label, TrafficType_name)``
# describing the BFW vs non-BFW pairs for the channel (single-variant for
# DLU/ULU which have no BFW concept). Offsets are sourced from
# ``aerial_postproc.logparse.getReceptionWindow`` at build time so the
# values stay in sync with the rest of aerial_postproc (and adapt to
# mmimo via getTCType). For each tick we emit one ``ph='X'`` band per
# unique window (open..close) on the channel's lane — BFW and non-BFW
# collapse to a single band when their offsets match (4TR case for DLC
# and ULC).
_DEADLINE_CHANNELS: Tuple[Tuple[str, int, str, Tuple[Tuple[str, str], ...]], ...] = (
    ("DLC", TID_DEADLINE_DLC, "deadline.dlc",
     (("BFW", "DTT_DLC_BFW"), ("Non-BFW", "DTT_DLC_NONBFW"))),
    ("DLU", TID_DEADLINE_DLU, "deadline.dlu",
     (("", "DTT_DLU"),)),
    ("ULC", TID_DEADLINE_ULC, "deadline.ulc",
     (("BFW", "DTT_ULC_BFW"), ("Non-BFW", "DTT_ULC_NONBFW"))),
    ("ULU", TID_DEADLINE_ULU, "deadline.ulu",
     (("", "DTT_ULU"),)),
)

# GPU kernel-type prefix -> cname category. Used to color GPU envelope events
# by channel rather than by Perfetto's per-thread hash.
_GPU_PREFIX_TO_CAT: Tuple[Tuple[str, str], ...] = (
    ("PUSCH ",       "gpu.pusch"),
    ("PUCCH ",       "gpu.pucch"),
    ("PRACH ",       "gpu.prach"),
    ("SRS ORDER ",   "gpu.srs_order"),   # must precede "ORDER " because of prefix overlap
    ("ORDER ",       "gpu.order"),
    ("COMPRESSION",  "gpu.compression"),
)


# ---------------------------------------------------------------------------
# Public configuration / input dataclasses.
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class PerfettoTraceConfig:
    """User-facing configuration for one trace export call.

    Attributes:
        enable_sfn_markers: Emit instant events at SFN rollovers and tick slot
            boundaries on the Tick lane. Each tick slot marker's ``args`` also
            includes the corresponding OTA SFN/slot derived from the tick row's
            ``tick_timestamp`` via ``tai_to_sfn`` (see ``_ota_sfn_slot_from_tick_timestamp``).
            Defaults to True.
        enable_counters: Emit per-CPU busy % counter track. Defaults to True.
        enable_deadline_markers: Emit per-slot deadline instant markers
            (DLC, DLU, ULC, ULU) on a dedicated ``Deadlines`` lane at the top
            of the CPU process. Defaults to True.
        mmimo_enable: When True, deadlines are pulled from the GH_64TR
            (mmimo) window table instead of GH_4TR. Mirrors the
            ``-e / --mmimo_enable`` flag used by ``latency_summary.py``.
            Only affects deadline-marker placement; no other behavior.
    """
    enable_sfn_markers: bool = True
    enable_counters: bool = True
    enable_deadline_markers: bool = True
    mmimo_enable: bool = False


@dataclasses.dataclass(frozen=True)
class RunInputs:
    """Pre-shaped inputs for one input run.

    All DataFrames already have ``start_datetime`` / ``end_datetime`` columns
    populated (the caller does this once, exactly as the existing Bokeh path
    does in ``cpu_timeline_plot.add_datetimes``).

    Attributes:
        prep: Output of ``logplot.prepare_cpu_timeline_row_frames``.
        gpu_envelope_df: Concatenated CPU-derived GPU envelope DataFrame
            (compression_df + pusch_df + pucch_df + prach_df + order_df +
            srs_order_df from cpu_timeline_plot.main), with ``Kernel Type``
            and ``start_datetime`` / ``end_datetime`` columns populated.
        label: User-facing label for the run (from ``--labels`` or auto).
        input_paths: List of source paths (folder or phy/testmac/ru log set).
    """
    prep: Dict[str, Any]
    gpu_envelope_df: pd.DataFrame
    label: str
    input_paths: Sequence[str]


# ---------------------------------------------------------------------------
# Internal pid/tid layout.
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class _RunPidLayout:
    """pid/tid assignments for one run.

    Built once per run by ``_allocate_run_pids``; consumed by every event
    builder so duration / counter / marker events all land on the correct
    lanes. Layout is one CPU process + one GPU process + one CPU Utilization
    process per run.
    """
    run_idx: int
    pid_base: int
    label: str
    cpu_pid: int                                        # process holding CPU/L2/Testmac/Tick lanes
    cpu_util_pid: int                                   # dedicated counter process (collapsible)
    gpu_pid: Optional[int]                              # GPU envelopes (None when no GPU events)
    deadlines_pid: Optional[int]                        # dedicated deadlines process (None when disabled)
    cpu_event_tid_by_yy: Mapping[float, int]            # yy -> tid for CPU-side events
    gpu_tid_by_kernel_type: Mapping[str, int]           # "PUSCH 0" -> tid in GPU process
    metadata_events: List[Dict[str, Any]]
    feature_tags: List[str]


# ---------------------------------------------------------------------------
# Public API: build / write.
# ---------------------------------------------------------------------------

def build_chrome_trace_events(
    runs: Sequence[RunInputs],
    config: PerfettoTraceConfig,
    *,
    split_per_run: bool,
) -> List[Dict[str, Any]]:
    """Build a flat list of trace event dicts from one or more runs.

    Args:
        runs: One ``RunInputs`` per input log set. Must be non-empty.
        config: Feature toggles.
        split_per_run: When True, omits the ``"<label> / "`` prefix from
            process names. Caller is responsible for writing one file per run
            in that mode; this function returns events for whichever runs it
            was given.

    Returns:
        Flat list of trace event dicts. Caller wraps in the envelope.
    """
    if not runs:
        raise ValueError("runs must be non-empty")

    all_events: List[Dict[str, Any]] = []

    for run_idx, run_inputs in enumerate(runs):
        layout = _allocate_run_pids(
            run_idx=run_idx,
            run_inputs=run_inputs,
            config=config,
            split_per_run=split_per_run,
        )
        all_events.extend(layout.metadata_events)
        all_events.extend(_build_all_duration_events(run_inputs, layout))
        if config.enable_sfn_markers:
            all_events.extend(_build_sfn_slot_markers(run_inputs.prep, layout))
        if config.enable_counters:
            all_events.extend(_build_counter_events(run_inputs, layout))
        if config.enable_deadline_markers:
            all_events.extend(_build_deadline_markers(
                run_inputs.prep, layout, config.mmimo_enable,
            ))

    if len(runs) > 1 and not split_per_run:
        _rebase_runs_to_common_start(all_events)
    _resolve_lane_collisions(all_events)
    return all_events


def write_chrome_trace(
    runs: Sequence[RunInputs],
    out_path: str,
    config: PerfettoTraceConfig,
    *,
    split_per_run: bool = False,
) -> List[str]:
    """Build and write one or more Chrome Trace Event Format JSON files.

    Args:
        runs: One ``RunInputs`` per input log set.
        out_path: Output filename. When ``split_per_run`` is True and there is
            more than one run, ``.run{N}`` is inserted before ``.json``.
        config: Feature toggles.
        split_per_run: When True with ``len(runs) > 1``, emits one file per
            run with auto-suffixed filenames. With ``len(runs) <= 1``,
            behaves identically to False.

    Returns:
        List of paths actually written, in order.
    """
    if not runs:
        raise ValueError("runs must be non-empty")

    parent = os.path.dirname(out_path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    feature_tags = _compute_feature_tags(config)
    command_line = shlex.join(sys.argv)
    created_at = _datetime.datetime.now().astimezone().isoformat()
    parser_version = _resolve_parser_version()

    if split_per_run and len(runs) > 1:
        written: List[str] = []
        for run_idx, single_run in enumerate(runs):
            split_path = _split_path_for_run(out_path, run_idx)
            events = build_chrome_trace_events(
                runs=[single_run], config=config, split_per_run=True,
            )
            other_data = _build_other_data_block(
                runs=[single_run],
                config=config,
                feature_tags=feature_tags,
                command_line=command_line,
                created_at=created_at,
                parser_version=parser_version,
            )
            _dump_trace_json(split_path, events, other_data)
            written.append(split_path)
        return written

    events = build_chrome_trace_events(
        runs=runs, config=config, split_per_run=False,
    )
    other_data = _build_other_data_block(
        runs=runs,
        config=config,
        feature_tags=feature_tags,
        command_line=command_line,
        created_at=created_at,
        parser_version=parser_version,
    )
    _dump_trace_json(out_path, events, other_data)
    return [out_path]


# ---------------------------------------------------------------------------
# Internal helpers.
# ---------------------------------------------------------------------------

def _resolve_parser_version() -> str:
    """Return ``aerial_postproc.__version__`` when defined; ``"unknown"`` otherwise."""
    try:
        import aerial_postproc
    except ImportError:
        return "unknown"
    return getattr(aerial_postproc, "__version__", "unknown")


def _compute_feature_tags(config: PerfettoTraceConfig) -> List[str]:
    tags = [
        _FEATURE_SORT_INDICES,
        _FEATURE_DISPLAY_TIME_UNIT_NS,
        _FEATURE_OTHER_DATA,
        _FEATURE_CNAME,
    ]
    if config.enable_sfn_markers:
        tags.append(_FEATURE_SFN_MARKERS)
    if config.enable_counters:
        tags.append(_FEATURE_COUNTERS)
    if config.enable_deadline_markers:
        tags.append(_FEATURE_DEADLINE_MARKERS)
    return tags


def _build_other_data_block(
    *,
    runs: Sequence[RunInputs],
    config: PerfettoTraceConfig,
    feature_tags: Sequence[str],
    command_line: str,
    created_at: str,
    parser_version: str,
) -> Dict[str, Any]:
    """Build the ``otherData`` dict embedded at the top of each JSON file."""
    block: Dict[str, Any] = {
        "label":          runs[0].label if len(runs) == 1 else [r.label for r in runs],
        "input_paths": (
            list(runs[0].input_paths)
            if len(runs) == 1
            else [list(r.input_paths) for r in runs]
        ),
        "command_line":   command_line,
        "created_at":     created_at,
        "parser_version": parser_version,
        "config": {
            "enable_sfn_markers":      config.enable_sfn_markers,
            "enable_counters":         config.enable_counters,
            "enable_deadline_markers": config.enable_deadline_markers,
            "mmimo_enable":            config.mmimo_enable,
        },
        "trace_features": list(feature_tags),
    }
    if len(runs) > 1:
        block["runs"] = [
            {"label": r.label, "input_paths": list(r.input_paths)} for r in runs
        ]
    return block


def _split_path_for_run(out_path: str, run_idx: int) -> str:
    stem, ext = os.path.splitext(out_path)
    if ext.lower() == ".json":
        return "%s.run%d.json" % (stem, run_idx)
    return "%s.run%d.json" % (out_path, run_idx)


def _dump_trace_json(path: str, events: List[Dict[str, Any]], other_data: Dict[str, Any]) -> None:
    payload = {
        "displayTimeUnit": "ns",
        "otherData": other_data,
        "traceEvents": events,
    }
    with open(path, "w", encoding="utf-8") as fp:
        json.dump(payload, fp, default=_json_default_encoder, separators=(",", ":"))
    try:
        size_bytes = os.path.getsize(path)
    except OSError:
        size_bytes = 0
    if size_bytes > TRACE_SIZE_WARN_BYTES:
        size_mb = size_bytes / (1024.0 * 1024.0)
        print(
            "WARNING: trace JSON %s is %.1f MB; consider --trace-split-per-run "
            "for easier loading in Perfetto UI." % (path, size_mb)
        )


def _json_default_encoder(value: Any) -> Any:
    """``json.dump(default=...)`` callback. Convert non-JSON-native scalars."""
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        float_value = float(value)
        if np.isnan(float_value) or np.isinf(float_value):
            return None
        return float_value
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (pd.Timestamp, _datetime.datetime, _datetime.date)):
        return value.isoformat()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if hasattr(value, "item"):
        return value.item()
    raise TypeError("Object of type %s is not JSON serializable" % (type(value).__name__,))


def _datetime_series_to_microseconds(series: pd.Series) -> np.ndarray:
    """Vectorized datetime64 -> int64 µs since epoch. NaT -> -1."""
    if len(series) == 0:
        return np.empty(0, dtype=np.int64)
    nanoseconds = series.values.astype("datetime64[ns]").astype(np.int64)
    return nanoseconds // NS_PER_US


def _rebase_runs_to_common_start(events: List[Dict[str, Any]]) -> None:
    """Shift each run's events so all runs start at the same wall-clock.

    When users pass multiple log captures from different sessions (different
    wall-clock times, e.g. captured days apart), the absolute timestamps in
    the source data place each run far apart on the Perfetto time axis with
    a huge empty gap between them — useless for cross-run comparison.

    This post-pass picks the earliest run's data start as the reference and
    shifts each later run backwards by exactly enough µs so that every run
    appears to start at the same wall-clock moment. Wall-clock fidelity is
    lost on the time axis, but the per-event ``args`` (including future
    additions like ``log_timestamp``) preserve the original log context.

    Only called for multi-run single-JSON output. Runs split into separate
    JSON files via ``--trace-split-per-run`` are not rebased.

    Run identity is inferred from ``pid`` (each run gets a disjoint
    ``[PID_BASE_PER_RUN + i*PID_STRIDE_PER_RUN, ...)`` range).

    Mutates ``events`` in place. Metadata events (``ph: "M"``) have no
    ``ts`` field and are left untouched.
    """
    # Find the earliest event ts per run.
    run_min_ts: Dict[int, float] = {}
    for event in events:
        if "ts" not in event or "pid" not in event:
            continue
        run_idx = (event["pid"] - PID_BASE_PER_RUN) // PID_STRIDE_PER_RUN
        if run_idx < 0:
            continue
        ts_val = event["ts"]
        if run_idx not in run_min_ts or ts_val < run_min_ts[run_idx]:
            run_min_ts[run_idx] = ts_val
    if not run_min_ts:
        return

    # Pick reference = earliest among all runs' starts.
    reference_ts = min(run_min_ts.values())
    # Compute per-run shift: how much to subtract from each run's events so
    # they align to the reference.
    shifts: Dict[int, float] = {
        run_idx: reference_ts - min_ts for run_idx, min_ts in run_min_ts.items()
    }
    # Apply.
    for event in events:
        if "ts" not in event or "pid" not in event:
            continue
        run_idx = (event["pid"] - PID_BASE_PER_RUN) // PID_STRIDE_PER_RUN
        shift = shifts.get(run_idx, 0.0)
        if shift != 0.0:
            event["ts"] = event["ts"] + shift


def _resolve_lane_collisions(events: List[Dict[str, Any]]) -> None:
    """Post-process ph='X' events so no two on the same (pid, tid) lane overlap.

    Two subtasks logged sub-µs apart get rounded to the same integer µs.
    After MIN_EVENT_DURATION_US clamping, the events end up overlapping on
    the same lane, which Perfetto renders as vertical stacking.

    For each ``(pid, tid)`` lane, sort its X events by ``(ts, sequence,
    insertion order)`` and push any colliding event forward to the end of
    the previous one. Cumulative shift per chain is typically 1-3 µs --
    invisible at typical zoom (slot period is 500 µs). The (sequence,
    insertion-order) tiebreakers ensure later-in-source events stay later
    in the rendered horizontal sequence.

    Mutates ``events`` in place. Counter (``C``), instant (``i``), and
    metadata (``M``) events are not touched.
    """
    from collections import defaultdict

    # Skip GPU lanes: tids there are allocated per Kernel Type (not per CUDA
    # stream), so two same-type kernels can legitimately overlap (multi-cell
    # or future multi-stream scenarios). Pushing the later one to last_end
    # would silently serialize real GPU concurrency. CPU lanes are still
    # de-stacked because their overlaps are sub-µs rounding artifacts on a
    # single-threaded CPU.
    indices_by_lane: Dict[Tuple[int, int], List[int]] = defaultdict(list)
    for idx, event in enumerate(events):
        if event.get("ph") == "X" and not str(event.get("cat", "")).startswith("gpu"):
            indices_by_lane[(event["pid"], event["tid"])].append(idx)

    for lane_indices in indices_by_lane.values():
        def sort_key(i: int) -> Tuple[float, float, int]:
            ev = events[i]
            seq_val = ev.get("args", {}).get("sequence", 0)
            seq_num = float(seq_val) if seq_val is not None else 0.0
            return (float(ev["ts"]), seq_num, i)

        lane_indices.sort(key=sort_key)
        last_end = float("-inf")
        for i in lane_indices:
            ev = events[i]
            if ev["ts"] < last_end:
                ev["ts"] = float(last_end)
            last_end = ev["ts"] + ev["dur"]


# ---------------------------------------------------------------------------
# pid / tid allocation.
# ---------------------------------------------------------------------------

def _allocate_run_pids(
    *,
    run_idx: int,
    run_inputs: RunInputs,
    config: PerfettoTraceConfig,
    split_per_run: bool,
) -> _RunPidLayout:
    """Allocate pid/tid assignment for one run; emit metadata events.

    Layout per run is fixed: one CPU process, one CPU Utilization process,
    one GPU process. All assigned pids share a 100-pid block so multiple
    runs in a single JSON never collide:
        pid_base + 0 → CPU Timeline
        pid_base + 1 → GPU Timeline
        pid_base + 2 → CPU Utilization (counter container)

    Pre-computes:
      * pid for each of the three processes,
      * tid for every CPU yy lane (under CPU process),
      * tid for every GPU kernel-type lane (under GPU process),
      * the process_name / thread_name / *_sort_index metadata events.
    """
    prep = run_inputs.prep
    pid_base = PID_BASE_PER_RUN + run_idx * PID_STRIDE_PER_RUN
    feature_tags = _compute_feature_tags(config)
    metadata_events: List[Dict[str, Any]] = []
    label_prefix = "" if split_per_run else (run_inputs.label + " / ")
    sort_run_offset = run_idx * SORT_RUN_STRIDE

    cpu_pid = pid_base + 0
    gpu_pid = pid_base + 1
    cpu_util_pid = pid_base + 2
    deadlines_pid = pid_base + 3

    metadata_events.extend(_emit_process_metadata(
        pid=cpu_pid,
        name="PHY CPU Timeline — %s" % (run_inputs.label,) if label_prefix
             else "PHY CPU Timeline",
        sort_index=sort_run_offset,
    ))
    metadata_events.extend(_emit_process_metadata(
        pid=cpu_util_pid,
        name="PHY CPU Utilization — %s" % (run_inputs.label,) if label_prefix
             else "PHY CPU Utilization",
        sort_index=sort_run_offset + 1,
    ))
    if config.enable_deadline_markers:
        metadata_events.extend(_emit_process_metadata(
            pid=deadlines_pid,
            name="PHY Deadlines — %s" % (run_inputs.label,) if label_prefix
                 else "PHY Deadlines",
            sort_index=sort_run_offset + 2,
        ))
        for channel_label, channel_tid, _cat, _variants in _DEADLINE_CHANNELS:
            metadata_events.extend(_emit_thread_metadata(
                pid=deadlines_pid,
                tid=channel_tid,
                name=channel_label,
                sort_index=channel_tid,
            ))
    metadata_events.extend(_emit_process_metadata(
        pid=gpu_pid,
        name="PHY GPU Timeline — %s" % (run_inputs.label,) if label_prefix
             else "PHY GPU Timeline",
        sort_index=sort_run_offset + 3,
    ))

    # Collect every CPU-side yy value across the source DataFrames and assign
    # one tid per unique yy. Guard the column access — set-comprehension
    # ``if`` is evaluated AFTER ``prep[key]["yy"]`` lookup, which raises
    # KeyError for empty/missing dfs.
    yy_values_set: set = set()
    for source_key in ("dl_task_df", "ul_task_df", "debug_task_df",
                       "l2_copy_df", "tick_copy_df", "testmac_copy_df"):
        lane_df = prep[source_key]
        if len(lane_df) == 0 or "yy" not in lane_df.columns:
            continue
        for yy_value in lane_df["yy"].unique():
            yy_values_set.add(float(yy_value))
    cpu_event_tid_by_yy: Dict[float, int] = {}
    for tid_offset, yy_value in enumerate(sorted(yy_values_set)):
        tid = 10000 + tid_offset
        cpu_event_tid_by_yy[yy_value] = tid
        metadata_events.extend(_emit_thread_metadata(
            pid=cpu_pid,
            tid=tid,
            name=_resolve_lane_name(prep, yy_value),
            sort_index=int(yy_value * 1000),
        ))

    gpu_tid_by_kernel_type = _emit_gpu_metadata(
        gpu_pid=gpu_pid,
        gpu_envelope_df=run_inputs.gpu_envelope_df,
        metadata_events=metadata_events,
    )

    return _RunPidLayout(
        run_idx=run_idx,
        pid_base=pid_base,
        label=run_inputs.label,
        cpu_pid=cpu_pid,
        cpu_util_pid=cpu_util_pid,
        gpu_pid=gpu_pid if gpu_tid_by_kernel_type else None,
        deadlines_pid=deadlines_pid if config.enable_deadline_markers else None,
        cpu_event_tid_by_yy=cpu_event_tid_by_yy,
        gpu_tid_by_kernel_type=gpu_tid_by_kernel_type,
        metadata_events=metadata_events,
        feature_tags=list(feature_tags),
    )


def _emit_gpu_metadata(
    *,
    gpu_pid: int,
    gpu_envelope_df: pd.DataFrame,
    metadata_events: List[Dict[str, Any]],
) -> Dict[str, int]:
    """Declare one thread per unique ``Kernel Type`` value. Returns lookup map."""
    if len(gpu_envelope_df) == 0 or "Kernel Type" not in gpu_envelope_df.columns:
        return {}
    kernel_types: List[str] = sorted({str(k) for k in gpu_envelope_df["Kernel Type"].unique()})
    gpu_tid_by_kernel_type: Dict[str, int] = {}
    for tid_offset, kernel_type in enumerate(kernel_types):
        tid = TID_GPU_KERNEL_BASE + tid_offset
        gpu_tid_by_kernel_type[kernel_type] = tid
        metadata_events.extend(_emit_thread_metadata(
            pid=gpu_pid,
            tid=tid,
            name=kernel_type[:MAX_EVENT_NAME_CHARS],
            sort_index=tid,
        ))
    return gpu_tid_by_kernel_type


def _emit_process_metadata(*, pid: int, name: str, sort_index: int) -> List[Dict[str, Any]]:
    return [
        {"name": "process_name",       "ph": "M", "pid": pid, "args": {"name": name}},
        {"name": "process_sort_index", "ph": "M", "pid": pid, "args": {"sort_index": sort_index}},
    ]


def _emit_thread_metadata(*, pid: int, tid: int, name: str, sort_index: int) -> List[Dict[str, Any]]:
    return [
        {"name": "thread_name",       "ph": "M", "pid": pid, "tid": tid, "args": {"name": name}},
        {"name": "thread_sort_index", "ph": "M", "pid": pid, "tid": tid, "args": {"sort_index": sort_index}},
    ]


def _resolve_lane_name(prep: Mapping[str, Any], yy_value: float) -> str:
    """Map a yy lane value back to a human-readable thread name."""
    eps = 1e-5
    cpus = prep["cpus"]
    n_cpus = len(cpus)
    for cpu_idx in range(n_cpus):
        if abs(yy_value - cpu_idx) < eps:
            return "CPU index %d (cpu %s)" % (cpu_idx, cpus[cpu_idx])
    if prep["enable_l2"]:
        if abs(yy_value - n_cpus) < eps:
            return "L2A Processing (even slot)"
        if abs(yy_value - (n_cpus + 1)) < eps:
            return "L2A Processing (odd slot)"
    if prep["enable_testmac"] and abs(yy_value - (n_cpus + 2)) < eps:
        return "Testmac FAPI Send"
    if prep["enable_tick"] and abs(yy_value - (n_cpus + 3)) < eps:
        return "Tick"
    for df_key, legend in (
        ("dl_comms_df", "DL Comm Task Breakout"),
        ("dl_comp_df", "DL Compression Breakout"),
    ):
        df = prep[df_key]
        if len(df) == 0 or "yy" not in df.columns:
            continue
        match = df[(df["yy"].astype(float) - yy_value).abs() < eps]
        if len(match) == 0:
            continue
        row = match.iloc[0]
        cpu_str = row["cpu"] if "cpu" in row.index and pd.notna(row["cpu"]) else "?"
        seq_str = row["sequence"] if "sequence" in row.index and pd.notna(row["sequence"]) else "?"
        return "%s — cpu %s seq %s" % (legend, cpu_str, seq_str)
    return "cpu_track y=%.6g" % (yy_value,)


# ---------------------------------------------------------------------------
# Duration events (X-phase). The bulk of the trace.
# ---------------------------------------------------------------------------

def _build_all_duration_events(
    run_inputs: RunInputs,
    layout: _RunPidLayout,
) -> List[Dict[str, Any]]:
    """Build the ``X``-phase events for every CPU + GPU lane in one run."""
    prep = run_inputs.prep
    events: List[Dict[str, Any]] = []

    cpu_pid_resolver = _make_cpu_pid_resolver(prep, layout)

    # dl_comms_df and dl_comp_df are intentionally NOT emitted: those same
    # source rows already appear in dl_task_df (they all start with
    # "DL Task ..."). Bokeh draws them twice (coarse + per-sequence) as a UX
    # overview/detail; Perfetto's category coloring + name filtering makes
    # the per-sequence breakout redundant and visually noisy.
    for df_key, category, name_columns in (
        ("dl_task_df",      "cpu.dl",       ("task", "subtask")),
        ("ul_task_df",      "cpu.ul",       ("task", "subtask")),
        ("debug_task_df",   "cpu.debug",    ("task", "subtask")),
        ("l2_copy_df",      "cpu.l2",       None),
        ("testmac_copy_df", "cpu.testmac",  None),
    ):
        df = prep[df_key]
        if len(df) == 0:
            continue
        events.extend(_build_duration_events_for_df(
            events_df=df,
            category=category,
            name_columns=name_columns,
            cpu_pid_resolver=cpu_pid_resolver,
            tid_lookup=layout.cpu_event_tid_by_yy,
        ))

    # Catch-all for TI tasks whose top-level name doesn't match the DL/UL/Debug
    # prefixes the existing layout knows about. This ensures new TI task
    # families added in cuBB / Aerial SDK appear in the Perfetto trace
    # without a script update.
    other_tasks_df = _extract_other_tasks_df(prep)
    if len(other_tasks_df) > 0:
        events.extend(_build_duration_events_for_df(
            events_df=other_tasks_df,
            category="cpu.other",
            name_columns=("task", "subtask"),
            cpu_pid_resolver=cpu_pid_resolver,
            tid_lookup=layout.cpu_event_tid_by_yy,
        ))

    if layout.gpu_pid is not None and len(run_inputs.gpu_envelope_df) > 0:
        events.extend(_build_gpu_duration_events(
            gpu_envelope_df=run_inputs.gpu_envelope_df,
            gpu_pid=layout.gpu_pid,
            gpu_tid_by_kernel_type=layout.gpu_tid_by_kernel_type,
        ))

    return events


# Top-level TI task prefixes that the existing layout handles explicitly.
# Anything else is treated as cpu.other so new TI task families auto-appear
# in the trace.
_KNOWN_TI_TASK_PREFIXES: Tuple[str, ...] = ("DL Task", "UL Task", "Debug Task")


def _extract_other_tasks_df(prep: Mapping[str, Any]) -> pd.DataFrame:
    """Return TI rows whose ``task`` doesn't match the known prefixes.

    The existing Bokeh-side filter in ``logplot.py:prepare_cpu_timeline_row_frames``
    drops these rows. We resurrect them here so the Perfetto trace stays
    auto-extensible: any future TI task family added in cuBB code (e.g.
    "MAC Task ...", "FH Task ...") shows up on the appropriate CPU lane
    under category ``cpu.other`` without code changes.

    Args:
        prep: Output of ``prepare_cpu_timeline_row_frames``.

    Returns:
        A DataFrame with a populated ``yy`` column (matching the integer CPU
        index lane), or an empty DataFrame when there are no unknown rows.
    """
    temp_df = prep["temp_df"]
    if len(temp_df) == 0 or "task" not in temp_df.columns:
        return pd.DataFrame()
    other_mask = ~temp_df["task"].str.startswith(_KNOWN_TI_TASK_PREFIXES, na=False)
    if not other_mask.any():
        return pd.DataFrame()
    other_df = temp_df[other_mask]
    if "cpu" not in other_df.columns:
        return pd.DataFrame()
    cpus: Sequence = prep["cpus"]
    cpu_index_by_phys = {phys: idx for idx, phys in enumerate(cpus)}
    other_df = other_df[other_df["cpu"].isin(cpus)].copy(deep=True)
    if len(other_df) == 0:
        return pd.DataFrame()
    other_df["yy"] = other_df["cpu"].map(cpu_index_by_phys).astype(float)
    return other_df


def _make_cpu_pid_resolver(prep: Mapping[str, Any], layout: _RunPidLayout):
    """Return a function row->pid for CPU-side events.

    All CPU events on a run route to the same CPU process, so this is a
    trivial closure. Kept as a function (rather than inlining ``layout.cpu_pid``)
    so the call signatures in the duration-event builders stay stable.
    """
    cpu_pid = layout.cpu_pid
    return lambda phys_cpu: cpu_pid


# Columns extracted into args (when present and non-null) for CPU events.
_CPU_ARGS_COLUMNS: Tuple[str, ...] = (
    "slot", "cpu", "duration", "task", "subtask", "sfn", "sequence",
)


def _build_duration_events_for_df(
    *,
    events_df: pd.DataFrame,
    category: str,
    name_columns: Optional[Tuple[str, str]],
    cpu_pid_resolver,
    tid_lookup: Mapping[float, int],
) -> List[Dict[str, Any]]:
    """Vectorized ``X``-event builder for one source DataFrame."""
    if "start_datetime" not in events_df.columns or "end_datetime" not in events_df.columns:
        return []

    start_us = _datetime_series_to_microseconds(events_df["start_datetime"])
    end_us = _datetime_series_to_microseconds(events_df["end_datetime"])
    valid_mask = (start_us >= 0) & (end_us >= 0)
    dur_us = np.maximum(end_us - start_us, MIN_EVENT_DURATION_US)

    yy_array = events_df["yy"].astype(float).to_numpy()
    cpu_array = events_df["cpu"].to_numpy() if "cpu" in events_df.columns else None
    slot_array = events_df["slot"].to_numpy() if "slot" in events_df.columns else None

    if name_columns is not None:
        primary_names = events_df[name_columns[0]].astype(str).to_numpy()
        secondary_names = events_df[name_columns[1]].astype(str).to_numpy()
    else:
        primary_names = None
        secondary_names = None

    # Pre-extract args columns once.
    args_arrays: Dict[str, np.ndarray] = {}
    args_masks: Dict[str, np.ndarray] = {}
    for col in _CPU_ARGS_COLUMNS:
        if col in events_df.columns:
            series = events_df[col]
            args_arrays[col] = series.to_numpy()
            args_masks[col] = series.notna().to_numpy()

    cname = DEFAULT_CNAME_PALETTE.get(category)
    events: List[Dict[str, Any]] = []
    append_event = events.append

    for row_idx in np.flatnonzero(valid_mask):
        yy_value = float(yy_array[row_idx])
        tid = tid_lookup.get(yy_value)
        if tid is None:
            continue

        phys_cpu = cpu_array[row_idx] if cpu_array is not None else None
        pid = cpu_pid_resolver(phys_cpu)
        if pid is None:
            continue

        # Event name. category-specific fallback when source df has no task/subtask.
        if primary_names is not None:
            event_name = "%s | %s" % (primary_names[row_idx], secondary_names[row_idx])
        elif category == "cpu.l2" and slot_array is not None:
            slot_val = slot_array[row_idx]
            event_name = "L2A Processing | slot %s" % (slot_val,) if pd.notna(slot_val) else "L2A Processing"
        elif category == "cpu.testmac":
            event_name = "Testmac FAPI Send"
        else:
            event_name = category
        if len(event_name) > MAX_EVENT_NAME_CHARS:
            event_name = event_name[: MAX_EVENT_NAME_CHARS - 3] + "..."

        event_args: Dict[str, Any] = {"yy": yy_value}
        for col, col_array in args_arrays.items():
            if args_masks[col][row_idx]:
                event_args[col] = col_array[row_idx]

        event: Dict[str, Any] = {
            "name": event_name,
            "cat":  category,
            "ph":   "X",
            "ts":   float(start_us[row_idx]),
            "dur":  float(dur_us[row_idx]),
            "pid":  pid,
            "tid":  tid,
            "args": event_args,
        }
        if cname is not None:
            event["cname"] = cname
        append_event(event)

    return events


def _build_gpu_duration_events(
    *,
    gpu_envelope_df: pd.DataFrame,
    gpu_pid: int,
    gpu_tid_by_kernel_type: Mapping[str, int],
) -> List[Dict[str, Any]]:
    """Vectorized ``X``-event builder for GPU envelope rows."""
    if "start_datetime" not in gpu_envelope_df.columns or "end_datetime" not in gpu_envelope_df.columns:
        return []

    start_us = _datetime_series_to_microseconds(gpu_envelope_df["start_datetime"])
    end_us = _datetime_series_to_microseconds(gpu_envelope_df["end_datetime"])
    valid_mask = (start_us >= 0) & (end_us >= 0)
    dur_us = np.maximum(end_us - start_us, MIN_EVENT_DURATION_US)

    kernel_type_array = gpu_envelope_df["Kernel Type"].astype(str).to_numpy()

    extra_cols: Tuple[str, ...] = ("slot", "duration", "start_deadline", "end_deadline", "sfn")
    extra_arrays: Dict[str, np.ndarray] = {}
    extra_masks: Dict[str, np.ndarray] = {}
    for col in extra_cols:
        if col in gpu_envelope_df.columns:
            series = gpu_envelope_df[col]
            extra_arrays[col] = series.to_numpy()
            extra_masks[col] = series.notna().to_numpy()

    events: List[Dict[str, Any]] = []
    append_event = events.append

    for row_idx in np.flatnonzero(valid_mask):
        kernel_type = kernel_type_array[row_idx]
        tid = gpu_tid_by_kernel_type.get(kernel_type)
        if tid is None:
            continue
        cat = _gpu_cat_for_kernel_type(kernel_type)
        cname = DEFAULT_CNAME_PALETTE.get(cat)
        event_name = kernel_type
        if len(event_name) > MAX_EVENT_NAME_CHARS:
            event_name = event_name[: MAX_EVENT_NAME_CHARS - 3] + "..."

        event_args: Dict[str, Any] = {"kernel_type": kernel_type}
        for col, col_array in extra_arrays.items():
            if extra_masks[col][row_idx]:
                event_args[col] = col_array[row_idx]

        event: Dict[str, Any] = {
            "name": event_name,
            "cat":  cat,
            "ph":   "X",
            "ts":   float(start_us[row_idx]),
            "dur":  float(dur_us[row_idx]),
            "pid":  gpu_pid,
            "tid":  tid,
            "args": event_args,
        }
        if cname is not None:
            event["cname"] = cname
        append_event(event)

    return events


def _gpu_cat_for_kernel_type(kernel_type: str) -> str:
    for prefix, cat in _GPU_PREFIX_TO_CAT:
        if kernel_type.startswith(prefix):
            return cat
    return "gpu"


# ---------------------------------------------------------------------------
# SFN / slot markers (instant events, scope=global).
# ---------------------------------------------------------------------------

def _composite_sfn_slot_from_t0_ns(t0_ns: int, *, mu: int = 1) -> Tuple[int, int]:
    """Map a T0 timestamp (nanoseconds) to log-style composite ``(sfn, slot)``.

    Same ``tai_to_sfn`` + ``NUM_20SLOT_GROUPS`` composition as ``logparse``
    (e.g. ``parse_ru_tx_times``).
    """
    from aerial_postproc.logparse import NUM_20SLOT_GROUPS
    from aerial_postproc.timeconv import slot_max_plus1, tai_to_sfn

    sfn_t = tai_to_sfn(int(t0_ns), 0, 0, mu=mu)
    composite_slot = sfn_t.slot + (sfn_t.sfn % NUM_20SLOT_GROUPS) * slot_max_plus1(mu)
    return sfn_t.sfn, composite_slot


def _ota_sfn_slot_from_tick_timestamp(tick_timestamp_ns: int, *, mu: int = 1) -> Tuple[int, int]:
    """OTA ``(sfn, slot)`` from the tick row's ``tick_timestamp`` field.

    ``parse_tick_times`` stores the L2A hardware tick instant in ``tick_timestamp``.
    That instant aligns with the OTA air slot (the logged ``slot``/``sfn`` on the
    tick line include L2A ``slot_advance``). Converting the tick instant with
    ``tai_to_sfn`` yields the OTA composite slot; see ``nv_phy_module::tick_received``.
    """
    return _composite_sfn_slot_from_t0_ns(int(tick_timestamp_ns), mu=mu)


def _tick_slot_marker_args(
    slot_val: Any,
    sfn_val: Any,
    tick_timestamp_ns: Any,
) -> Dict[str, Any]:
    """Build Perfetto ``args`` for a tick-lane instant marker.

    Tick ``sfn``/``slot`` are the values logged on the L2A.TICK_TIMES line. OTA
    ``sfn``/``slot`` are derived from ``tick_timestamp`` via ``tai_to_sfn``.
    """
    args: Dict[str, Any] = {}
    if slot_val is not None and pd.notna(slot_val):
        args["slot"] = int(slot_val)
    if sfn_val is not None and pd.notna(sfn_val):
        args["sfn"] = int(sfn_val)
    if tick_timestamp_ns is not None and pd.notna(tick_timestamp_ns):
        ota_sfn_val, ota_slot_val = _ota_sfn_slot_from_tick_timestamp(
            int(tick_timestamp_ns),
        )
        args["ota_slot"] = ota_slot_val
        args["ota_sfn"] = ota_sfn_val
    return args


def _build_sfn_slot_markers(
    prep: Mapping[str, Any],
    layout: _RunPidLayout,
) -> List[Dict[str, Any]]:
    """Emit ph='i' instant events at tick slot boundaries and SFN rollovers.

    All instants use the ``Tick`` thread track. Tick slot markers include OTA
    SFN/slot in ``args`` (from ``tick_timestamp`` via ``tai_to_sfn``).
    """
    tick_df = prep["tick_copy_df"]
    if not prep["enable_tick"] or len(tick_df) == 0:
        return []
    if "cpu_timestamp" not in tick_df.columns:
        return []
    if "tick_timestamp" not in tick_df.columns:
        return []

    # The Tick lane sits at yy = len(cpus) + 3 (see
    # prepare_cpu_timeline_row_frames). Look up its tid + pid.
    tick_yy = float(len(prep["cpus"]) + 3)
    tick_tid = layout.cpu_event_tid_by_yy.get(tick_yy)
    tick_pid = layout.cpu_pid
    if tick_tid is None or tick_pid is None:
        return []

    # Anchor on cpu_timestamp (the wall-clock instant when L2A logged the
    # tick line) to match the Bokeh tick marker. ``t0_timestamp`` is the
    # canonical air-frame TAI of the slot; that's where the deadline
    # markers anchor, but it's NOT where the CPU/L2A actually processes
    # the slot. cpu_timestamp can lead or lag t0 depending on the cuBB
    # config (e.g., parallel_l2a leads t0 by ~1.5 ms).
    cpu_ns = tick_df["cpu_timestamp"].to_numpy().astype(np.int64)
    tick_us = cpu_ns // NS_PER_US
    valid_mask = tick_us >= 0
    if not valid_mask.any():
        return []

    slot_array = tick_df["slot"].to_numpy() if "slot" in tick_df.columns else None
    sfn_array = tick_df["sfn"].to_numpy() if "sfn" in tick_df.columns else None
    tick_ts_array = tick_df["tick_timestamp"].to_numpy()

    events: List[Dict[str, Any]] = []
    append_event = events.append
    cname_slot = DEFAULT_CNAME_PALETTE.get("slot.boundary")
    cname_sfn = DEFAULT_CNAME_PALETTE.get("slot.sfn")

    for row_idx in np.flatnonzero(valid_mask):
        ts_us = float(tick_us[row_idx])
        slot_val = slot_array[row_idx] if slot_array is not None else None
        sfn_val = sfn_array[row_idx] if sfn_array is not None else None
        marker_args = _tick_slot_marker_args(
            slot_val, sfn_val, tick_ts_array[row_idx],
        )

        slot_event: Dict[str, Any] = {
            "name": "slot %s" % (slot_val,) if slot_val is not None and pd.notna(slot_val) else "slot",
            "cat":  "slot.boundary",
            "ph":   "i",
            "s":    "t",
            "ts":   ts_us,
            "pid":  tick_pid,
            "tid":  tick_tid,
            "args": marker_args,
        }
        if cname_slot is not None:
            slot_event["cname"] = cname_slot
        append_event(slot_event)

        # SFN boundary marker (only when slot == 0).
        is_sfn_boundary = slot_val is not None and pd.notna(slot_val) and int(slot_val) == 0
        if is_sfn_boundary:
            sfn_event: Dict[str, Any] = {
                "name": "SFN %s" % (sfn_val,) if sfn_val is not None and pd.notna(sfn_val) else "SFN",
                "cat":  "slot.sfn",
                "ph":   "i",
                "s":    "t",
                "ts":   ts_us,
                "pid":  tick_pid,
                "tid":  tick_tid,
                "args": marker_args,
            }
            if cname_sfn is not None:
                sfn_event["cname"] = cname_sfn
            append_event(sfn_event)

    return events


# ---------------------------------------------------------------------------
# Per-slot deadline markers (instant events on a dedicated Deadlines lane).
# ---------------------------------------------------------------------------

def _build_deadline_markers(
    prep: Mapping[str, Any],
    layout: _RunPidLayout,
    mmimo_enable: bool,
) -> List[Dict[str, Any]]:
    """Emit per-slot deadline shading bands (DLC, DLU, ULC, ULU).

    For each slot tick in ``prep['tick_copy_df']`` and each channel variant
    (BFW, Non-BFW), emits one ``ph='X'`` duration event spanning the
    reception window from ``slot_t0 + OFFSET1`` to ``slot_t0 + OFFSET2``
    on the channel's dedicated lane inside the ``PHY Deadlines`` process.
    BFW and Non-BFW collapse to a single band when their offsets match
    (4TR case for DLC and ULC).

    Offsets are sourced from ``aerial_postproc.logparse.getReceptionWindow``
    and selected by ``mmimo_enable`` (False = GH_4TR, True = GH_64TR). For
    mmimo BFW channels, OFFSET1 (window open) shifts earlier than non-BFW.
    OFFSET2 (window close = deadline) is identical across BFW / non-BFW
    and across 4TR / 64TR for the same channel.

    Anchored on ``tick_df['t0_timestamp']`` (the canonical slot reference
    used by every other DataFrame in the parser), NOT on
    ``tick_df['cpu_timestamp']`` (which is the CPU's tick-processing time
    and typically lags ``t0_timestamp`` by ~500 µs — that gap is the
    ``cpu_deadline`` metric the parser produces).

    The band's ``name`` carries the channel, BFW / Non-BFW variant (when
    distinct), and ``sfn`` / ``slot`` so the Perfetto UI displays it on the
    bar. ``args`` also includes ``channel``, ``variant``, ``open_us``,
    ``close_us`` for tooltip use.
    """
    if layout.deadlines_pid is None:
        return []
    tick_df = prep["tick_copy_df"]
    if not prep["enable_tick"] or len(tick_df) == 0:
        return []
    if "t0_timestamp" not in tick_df.columns:
        return []

    # Lazy import keeps perfetto_trace importable in environments where
    # logparse may not be loadable (defensive — getReceptionWindow is the
    # source of truth for offsets).
    from aerial_postproc.logparse import getReceptionWindow, getTCType, TrafficType

    tc_type = getTCType(mmimo_enable)
    # Per channel, deduplicate variants whose offsets collapse to the same
    # window (the 4TR DLC and ULC case where BFW and Non-BFW are equal).
    # Resolved spec is (channel_label, channel_tid, cat, [(variant_label,
    # open_us, close_us)]).
    resolved_channels: List[Tuple[str, int, str, List[Tuple[str, float, float]]]] = []
    for channel_label, channel_tid, cat, variants in _DEADLINE_CHANNELS:
        seen_windows: Dict[Tuple[float, float], str] = {}
        variant_specs: List[Tuple[str, float, float]] = []
        for variant_label, traffic_type_name in variants:
            traffic_type = TrafficType[traffic_type_name]
            open_us, close_us = getReceptionWindow(traffic_type, tc_type)
            window = (float(open_us), float(close_us))
            if window in seen_windows:
                continue
            seen_windows[window] = variant_label
            variant_specs.append((variant_label, float(open_us), float(close_us)))
        # When only one variant remains, drop its label so the band reads
        # simply as "<channel> window" rather than "<channel> BFW window".
        if len(variant_specs) == 1:
            variant_specs = [("", variant_specs[0][1], variant_specs[0][2])]
        resolved_channels.append((channel_label, channel_tid, cat, variant_specs))

    # tick_df['t0_timestamp'] is the slot reference in nanoseconds.
    t0_ns = tick_df["t0_timestamp"].to_numpy().astype(np.int64)
    t0_us = t0_ns // NS_PER_US
    valid_mask = t0_us >= 0
    if not valid_mask.any():
        return []
    slot_array = tick_df["slot"].to_numpy() if "slot" in tick_df.columns else None
    sfn_array = tick_df["sfn"].to_numpy() if "sfn" in tick_df.columns else None

    events: List[Dict[str, Any]] = []
    append_event = events.append
    deadlines_pid = layout.deadlines_pid
    cname_by_cat = {cat: DEFAULT_CNAME_PALETTE.get(cat) for _, _, cat, _ in resolved_channels}

    for row_idx in np.flatnonzero(valid_mask):
        slot_t0_us = float(t0_us[row_idx])
        slot_val = slot_array[row_idx] if slot_array is not None else None
        sfn_val = sfn_array[row_idx] if sfn_array is not None else None
        slot_payload: Dict[str, Any] = {}
        if slot_val is not None and pd.notna(slot_val):
            slot_payload["slot"] = slot_val
        if sfn_val is not None and pd.notna(sfn_val):
            slot_payload["sfn"] = sfn_val
        sfn_slot_suffix = ""
        if "sfn" in slot_payload and "slot" in slot_payload:
            sfn_slot_suffix = " — sfn %s slot %s" % (slot_payload["sfn"], slot_payload["slot"])
        elif "slot" in slot_payload:
            sfn_slot_suffix = " — slot %s" % (slot_payload["slot"],)

        for channel_label, channel_tid, cat, variant_specs in resolved_channels:
            cname = cname_by_cat.get(cat)
            for variant_label, open_us, close_us in variant_specs:
                dur_us = max(close_us - open_us, MIN_EVENT_DURATION_US)
                if variant_label:
                    base_name = "%s %s window" % (channel_label, variant_label)
                else:
                    base_name = "%s window" % (channel_label,)
                event_args: Dict[str, Any] = {
                    "channel":  channel_label,
                    "variant":  variant_label if variant_label else "n/a",
                    "open_us":  open_us,
                    "close_us": close_us,
                }
                event_args.update(slot_payload)
                event: Dict[str, Any] = {
                    "name": base_name + sfn_slot_suffix,
                    "cat":  cat,
                    "ph":   "X",
                    "ts":   slot_t0_us + open_us,
                    "dur":  float(dur_us),
                    "pid":  deadlines_pid,
                    "tid":  channel_tid,
                    "args": event_args,
                }
                if cname is not None:
                    event["cname"] = cname
                append_event(event)

    return events


# ---------------------------------------------------------------------------
# Counter events (C-phase).
# ---------------------------------------------------------------------------

def _build_counter_events(
    run_inputs: RunInputs,
    layout: _RunPidLayout,
) -> List[Dict[str, Any]]:
    """Emit per-CPU busy% counter track."""
    events: List[Dict[str, Any]] = []
    if layout.cpu_util_pid is not None:
        events.extend(_build_cpu_busy_counter(run_inputs.prep, layout.cpu_util_pid))
    return events


def _build_cpu_busy_counter(
    prep: Mapping[str, Any],
    counter_pid: int,
) -> List[Dict[str, Any]]:
    """Per-CPU busy% sampled at CPU_BUSY_BUCKET_US, all CPUs as one stacked counter.

    Each event's duration is split across every bucket it overlaps using true
    interval-overlap accounting (legacy ``sum_overlaps_ns`` semantics): an event
    spanning [s, e] contributes ``min(bucket_end, e) - max(bucket_start, s)``
    to each bucket it intersects. An earlier version of this function attributed
    the entire duration to the start-bucket only, which inflated buckets where
    a long event began and zeroed buckets the event continued through.

    Caveats it does NOT fix (separate semantic concerns):
      * "Wait *" subtasks (Wait Slot End Task, Wait Order, etc.) still count
        as busy time. They're CPU-side waits, not real work, but they're
        inside TI brackets so the math treats them as activity.
      * Not true CPU utilization: kernel time, syscalls, other processes are
        invisible.

    Source rows: ``prep['task_df']`` (DL/UL/Debug tasks) plus
    ``_extract_other_tasks_df(prep)`` (any TI rows whose top-level family
    isn't DL/UL/Debug — e.g. future ``MAC Task`` or ``FH Task`` lines that
    cuBB might add). Including cpu.other rows keeps the busy counter
    consistent with what's actually visible on the timeline.

    Notably does NOT include ``dl_comms_df`` / ``dl_comp_df``: those are
    strict subsets of ``dl_task_df`` and including them would double-count.
    """
    task_df = prep["task_df"]
    other_df = _extract_other_tasks_df(prep)
    frames = [df for df in (task_df, other_df) if len(df) > 0]
    if not frames:
        return []
    all_cpu = pd.concat(frames, ignore_index=True, copy=False) if len(frames) > 1 else frames[0]
    if "start_datetime" not in all_cpu.columns or "end_datetime" not in all_cpu.columns:
        return []
    if "cpu" not in all_cpu.columns:
        return []

    start_us = _datetime_series_to_microseconds(all_cpu["start_datetime"])
    end_us = _datetime_series_to_microseconds(all_cpu["end_datetime"])
    valid_mask = (start_us >= 0) & (end_us >= 0) & (end_us > start_us)
    if not valid_mask.any():
        return []
    start_us = start_us[valid_mask]
    end_us = end_us[valid_mask]
    cpu_array = all_cpu.loc[valid_mask, "cpu"].to_numpy()

    bucket_us = CPU_BUSY_BUCKET_US
    # For each event, identify the range of buckets it overlaps: from the
    # bucket containing its start_us to the bucket containing end_us - epsilon.
    # Using ceil(end / bucket) - 1 to correctly handle float ts where the
    # event ends on or just past a bucket boundary.
    first_bucket = (start_us // bucket_us).astype(np.int64)
    last_bucket = (np.ceil(end_us / bucket_us).astype(np.int64) - 1)
    last_bucket = np.maximum(last_bucket, first_bucket)
    buckets_per_event = (last_bucket - first_bucket + 1).astype(np.int64)

    # Replicate each event by the number of buckets it touches so the
    # per-bucket overlap can be computed in a single vectorized pass.
    repeated_idx = np.repeat(np.arange(len(start_us), dtype=np.int64), buckets_per_event)
    # Per-row "position within this event's bucket list" (0, 1, 2, ... per event).
    event_row_starts = np.concatenate(([0], np.cumsum(buckets_per_event[:-1])))
    position_in_event = np.arange(len(repeated_idx), dtype=np.int64) - np.repeat(event_row_starts, buckets_per_event)
    bucket_ids = first_bucket[repeated_idx] + position_in_event
    bucket_starts = bucket_ids * bucket_us
    bucket_ends = bucket_starts + bucket_us
    overlap = (
        np.minimum(bucket_ends.astype(np.float64), end_us[repeated_idx])
        - np.maximum(bucket_starts.astype(np.float64), start_us[repeated_idx])
    )

    busy = pd.DataFrame({
        "bucket": bucket_starts,
        "cpu":    cpu_array[repeated_idx],
        "ovlp":   overlap.astype(np.float64),
    })
    pivot = (busy.groupby(["bucket", "cpu"])["ovlp"]
                 .sum()
                 .div(bucket_us)
                 .clip(upper=1.0)
                 .unstack(level="cpu", fill_value=0.0)
                 .sort_index())

    events: List[Dict[str, Any]] = []
    append_event = events.append
    columns = list(pivot.columns)
    for bucket_us_pos, row in pivot.iterrows():
        args = {"cpu_%s" % (col,): float(row[col]) for col in columns}
        append_event({
            "name": "cpu_busy_pct",
            "cat":  "counter.cpu",
            "ph":   "C",
            "ts":   float(bucket_us_pos),
            "pid":  counter_pid,
            "args": args,
        })
    return events


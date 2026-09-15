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

"""Generate per-cell scheduler QA plots from a cuMAC HDF5 time trace."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_TB_ERROR_SMOOTHING_WINDOW = 100
DEFAULT_INS_RATE_SMOOTHING_WINDOW = 100
DEFAULT_TB_ERROR_HIGH_VALUE_THRESHOLD = 0.2
DEFAULT_TB_ERROR_HIGH_RATIO_THRESHOLD = 0.2
DEFAULT_TB_ERROR_HIGH_RATIO_SKIP_INITIAL_SLOTS = 50
BPS_PER_GBPS = 1.0e9
MARKER_STYLES = ("o", "s", "d", "^", "v", ">", "<", "p", "h", "*", "+", "x")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot per-cell scheduler statistics from a cuMAC HDF5 time trace. "
            "Cell indices are one-based to match the MATLAB analysis script."
        )
    )
    parser.add_argument("filename", help="Path to the HDF5 time trace file.")
    parser.add_argument(
        "--cells",
        nargs="+",
        type=int,
        help="Optional one-based cell indices to plot. Defaults to all cells.",
    )
    parser.add_argument(
        "--tb-error-smoothing-window",
        type=int,
        default=DEFAULT_TB_ERROR_SMOOTHING_WINDOW,
        help=(
            "Causal smoothing window for per-cell TB error. "
            f"Defaults to {DEFAULT_TB_ERROR_SMOOTHING_WINDOW}."
        ),
    )
    parser.add_argument(
        "--tb-error-high-value-threshold",
        type=float,
        default=DEFAULT_TB_ERROR_HIGH_VALUE_THRESHOLD,
        help=(
            "Smoothed TB-error value threshold used by the high-error ratio check. "
            f"Defaults to {DEFAULT_TB_ERROR_HIGH_VALUE_THRESHOLD}."
        ),
    )
    parser.add_argument(
        "--tb-error-high-ratio-threshold",
        type=float,
        default=DEFAULT_TB_ERROR_HIGH_RATIO_THRESHOLD,
        help=(
            "Maximum allowed fraction of valid smoothed TB-error points above "
            f"--tb-error-high-value-threshold. Defaults to "
            f"{DEFAULT_TB_ERROR_HIGH_RATIO_THRESHOLD}."
        ),
    )
    parser.add_argument(
        "--tb-error-high-ratio-skip-initial-slots",
        type=int,
        default=DEFAULT_TB_ERROR_HIGH_RATIO_SKIP_INITIAL_SLOTS,
        help=(
            "Number of initial slots to exclude from the high TB-error ratio check. "
            f"Defaults to {DEFAULT_TB_ERROR_HIGH_RATIO_SKIP_INITIAL_SLOTS}."
        ),
    )
    parser.add_argument(
        "--ins-rate-smoothing-window",
        type=int,
        default=DEFAULT_INS_RATE_SMOOTHING_WINDOW,
        help=(
            "Causal smoothing window size for per-cell instantaneous rate; "
            "larger values increase smoothing. "
            f"Defaults to {DEFAULT_INS_RATE_SMOOTHING_WINDOW}."
        ),
    )
    parser.add_argument(
        "--save-plot",
        action="store_true",
        help="Save the generated plot.",
    )
    parser.add_argument(
        "--save-plot-dir",
        type=Path,
        default=Path("."),
        help="Directory for saved plots. Defaults to the current folder.",
    )
    parser.add_argument(
        "--show-plot",
        action="store_true",
        help="Show the generated plot.",
    )
    parser.add_argument(
        "--apply-slot-ratio-threshold",
        action="store_true",
        help="Enable dataset-wide scheduled-slot ratio validation.",
    )
    parser.add_argument(
        "--slot-ratio-threshold",
        type=float,
        default=0.8,
        help=(
            "Minimum required fraction of slots with at least one scheduled UE "
            "across all cells when --apply-slot-ratio-threshold is used. "
            "Defaults to 0.8."
        ),
    )
    return parser.parse_args()


def read_dataset(h5_file: h5py.File, dataset_name: str) -> np.ndarray:
    if dataset_name not in h5_file:
        raise KeyError(f"Required dataset '{dataset_name}' was not found.")
    return np.asarray(h5_file[dataset_name])


def validate_cell_indices(cells: list[int] | None, num_cells: int) -> list[int]:
    if cells is None:
        return list(range(1, num_cells + 1))

    invalid_cells = [cell_idx for cell_idx in cells if cell_idx < 1 or cell_idx > num_cells]
    if invalid_cells:
        invalid_str = ", ".join(str(cell_idx) for cell_idx in invalid_cells)
        raise ValueError(
            f"Invalid cell index(s): {invalid_str}. Must be between 1 and {num_cells}."
        )

    return cells


def make_plot_save_path(output_dir: Path, h5_filename: Path) -> Path:
    if output_dir.exists() and not output_dir.is_dir():
        raise ValueError(f"Plot output path is not a directory: {output_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / f"{h5_filename.stem}.png"


def causal_mean_ignore_value(
    data: np.ndarray, window_size: int, invalid_value: float = -1.0
) -> np.ndarray:
    if window_size <= 1:
        return data.copy()

    smoothed_data = np.zeros_like(data, dtype=float)
    for idx in range(data.size):
        window_start = max(0, idx - window_size + 1)
        window_values = data[window_start : idx + 1]
        valid_values = window_values[window_values != invalid_value]
        smoothed_data[idx] = np.mean(valid_values) if valid_values.size else invalid_value

    return smoothed_data


def find_per_cell_validation_failures(
    per_cell_num_sche_ues: np.ndarray,
    per_cell_tb_err: np.ndarray,
    per_cell_ins_rate: np.ndarray,
    cells_to_plot: list[int],
    tb_error_smoothing_window: int,
    ins_rate_smoothing_window: int,
) -> list[str]:
    failures: list[str] = []

    for cell_idx in cells_to_plot:
        cell_col = cell_idx - 1
        subplot_checks = (
            (
                "Number of Scheduled UEs",
                per_cell_num_sche_ues[:, cell_col],
                lambda values: np.isfinite(values) & (values >= 0),
            ),
            (
                "TB Error",
                causal_mean_ignore_value(
                    per_cell_tb_err[:, cell_col], tb_error_smoothing_window
                ),
                lambda values: (values == -1.0)
                | (np.isfinite(values) & (values >= 0) & (values <= 1)),
            ),
            (
                "Instantaneous Rate",
                causal_mean_ignore_value(
                    per_cell_ins_rate[:, cell_col], ins_rate_smoothing_window
                ),
                lambda values: np.isfinite(values) & (values >= 0),
            ),
        )

        for subplot_name, subplot_data, is_valid in subplot_checks:
            invalid_count = int(np.count_nonzero(~is_valid(subplot_data)))
            if invalid_count:
                failures.append(
                    f"Failed subplot: {subplot_name}, Cell {cell_idx}, "
                    f"invalid points: {invalid_count}"
                )

    return failures


def find_high_tb_error_ratio_failure(
    per_cell_tb_err: np.ndarray,
    tb_error_smoothing_window: int,
    tb_error_high_value_threshold: float,
    tb_error_high_ratio_threshold: float,
    tb_error_high_ratio_skip_initial_slots: int,
) -> str | None:
    smoothed_tb_err = np.column_stack(
        [
            causal_mean_ignore_value(per_cell_tb_err[:, cell_col], tb_error_smoothing_window)
            for cell_col in range(per_cell_tb_err.shape[1])
        ]
    )
    smoothed_tb_err = smoothed_tb_err[tb_error_high_ratio_skip_initial_slots:, :]
    valid_tb_err = (
        np.isfinite(smoothed_tb_err) & (smoothed_tb_err >= 0) & (smoothed_tb_err <= 1)
    )
    valid_count = int(np.count_nonzero(valid_tb_err))
    if valid_count == 0:
        return None

    high_count = int(
        np.count_nonzero(valid_tb_err & (smoothed_tb_err > tb_error_high_value_threshold))
    )
    high_ratio = high_count / valid_count
    if high_ratio > tb_error_high_ratio_threshold:
        return (
            "Failed high TB error ratio: "
            f"{high_ratio:.6g} ({high_count}/{valid_count} valid points) "
            f"exceeds threshold {tb_error_high_ratio_threshold:.6g}; "
            f"TB error value threshold is {tb_error_high_value_threshold:.6g}; "
            f"excluded initial slots: {tb_error_high_ratio_skip_initial_slots}"
        )

    return None


def find_scheduled_slot_ratio_failure(
    per_cell_num_sche_ues: np.ndarray, threshold: float
) -> str | None:
    total_scheduled_ues_per_slot = np.sum(per_cell_num_sche_ues, axis=1)
    scheduled_slot_count = int(np.count_nonzero(total_scheduled_ues_per_slot >= 1))
    scheduled_slot_ratio = scheduled_slot_count / per_cell_num_sche_ues.shape[0]

    if scheduled_slot_ratio < threshold:
        return (
            "Failed scheduled slot ratio: "
            f"{scheduled_slot_ratio:.6g} "
            f"({scheduled_slot_count}/{per_cell_num_sche_ues.shape[0]} slots) "
            f"is below threshold {threshold:.6g}"
        )

    return None


def plot_per_cell_statistics(
    per_cell_num_sche_ues: np.ndarray,
    per_cell_tb_err: np.ndarray,
    per_cell_ins_rate: np.ndarray,
    cells_to_plot: list[int],
    tb_error_smoothing_window: int,
    ins_rate_smoothing_window: int,
    save_path: Path | None = None,
    show_plot: bool = False,
) -> None:
    num_slots, _ = per_cell_num_sche_ues.shape
    slot_indices = np.arange(1, num_slots + 1)
    marker_interval = max(1, num_slots // 20)

    fig, axes = plt.subplots(3, 1, figsize=(10, 12), num="Per-Cell Statistics")

    for plot_idx, cell_idx in enumerate(cells_to_plot):
        cell_col = cell_idx - 1
        marker_style = MARKER_STYLES[plot_idx % len(MARKER_STYLES)]
        axes[0].plot(
            slot_indices,
            per_cell_num_sche_ues[:, cell_col],
            linewidth=1.5,
            marker=marker_style,
            markersize=6,
            markevery=marker_interval,
            label=f"Cell {cell_idx}",
        )

    axes[0].set_xlabel("Slot Index", fontsize=11)
    axes[0].set_ylabel("Number of Scheduled UEs", fontsize=11)
    axes[0].set_title(
        "Number of Scheduled UEs per Cell per Slot", fontsize=12, fontweight="bold"
    )
    axes[0].legend(loc="best")
    axes[0].grid(True)
    axes[0].set_xlim(1, num_slots)

    for plot_idx, cell_idx in enumerate(cells_to_plot):
        cell_col = cell_idx - 1
        marker_style = MARKER_STYLES[plot_idx % len(MARKER_STYLES)]
        tb_err_data = causal_mean_ignore_value(
            per_cell_tb_err[:, cell_col], tb_error_smoothing_window
        )
        axes[1].plot(
            slot_indices,
            tb_err_data,
            linewidth=1.5,
            marker=marker_style,
            markersize=6,
            markevery=marker_interval,
            label=f"Cell {cell_idx}",
        )

    axes[1].plot(
        [1, num_slots],
        [0.1, 0.1],
        "r--",
        linewidth=2,
        label="10% TB Error",
    )
    axes[1].set_xlabel("Slot Index", fontsize=11)
    axes[1].set_ylabel("TB Error", fontsize=11)
    axes[1].set_title(
        (
            "TB Error per Cell per Slot "
            f"(Smoothing: {tb_error_smoothing_window}, ignoring unscheduled slots)"
        ),
        fontsize=12,
        fontweight="bold",
    )
    axes[1].legend(loc="best")
    axes[1].grid(True)
    axes[1].set_xlim(1, num_slots)
    axes[1].set_ylim(0, 1)

    for plot_idx, cell_idx in enumerate(cells_to_plot):
        cell_col = cell_idx - 1
        marker_style = MARKER_STYLES[plot_idx % len(MARKER_STYLES)]
        ins_rate_data = causal_mean_ignore_value(
            per_cell_ins_rate[:, cell_col], ins_rate_smoothing_window
        )
        axes[2].plot(
            slot_indices,
            ins_rate_data / BPS_PER_GBPS,
            linewidth=1.5,
            marker=marker_style,
            markersize=6,
            markevery=marker_interval,
            label=f"Cell {cell_idx}",
        )

    axes[2].set_xlabel("Slot Index", fontsize=11)
    axes[2].set_ylabel("Instantaneous Rate (Gbps)", fontsize=11)
    if ins_rate_smoothing_window > 1:
        ins_rate_title = (
            "Instantaneous Rate per Cell per Slot "
            f"(Smoothing: {ins_rate_smoothing_window})"
        )
    else:
        ins_rate_title = "Instantaneous Rate per Cell per Slot"
    axes[2].set_title(ins_rate_title, fontsize=12, fontweight="bold")
    axes[2].legend(loc="best")
    axes[2].grid(True)
    axes[2].set_xlim(1, num_slots)

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def main() -> int:
    args = parse_args()
    h5_filename = Path(args.filename)

    if not h5_filename.is_file():
        raise FileNotFoundError(f"File not found: {h5_filename}")

    with h5py.File(h5_filename, "r") as h5_file:
        # h5py exposes the raw HDF5 dimension order, while MATLAB h5read
        # presents these datasets as (slot, cell). Transpose to match MATLAB.
        per_cell_num_sche_ues = read_dataset(h5_file, "/perCellperSlotNumScheUEs").T
        per_cell_tb_err = read_dataset(h5_file, "/perCellperSlotTbErr").T
        per_cell_ins_rate = read_dataset(h5_file, "/perCellperSlotInsRate").T

    if per_cell_num_sche_ues.ndim != 2:
        raise ValueError("/perCellperSlotNumScheUEs must be a 2D dataset.")
    if per_cell_tb_err.shape != per_cell_num_sche_ues.shape:
        raise ValueError("/perCellperSlotTbErr must match /perCellperSlotNumScheUEs shape.")
    if per_cell_ins_rate.shape != per_cell_num_sche_ues.shape:
        raise ValueError("/perCellperSlotInsRate must match /perCellperSlotNumScheUEs shape.")
    if args.tb_error_smoothing_window < 1:
        raise ValueError("--tb-error-smoothing-window must be at least 1.")
    if args.ins_rate_smoothing_window < 1:
        raise ValueError("--ins-rate-smoothing-window must be at least 1.")
    if not 0 <= args.tb_error_high_value_threshold <= 1:
        raise ValueError("--tb-error-high-value-threshold must be between 0 and 1, inclusive.")
    if not 0 <= args.tb_error_high_ratio_threshold <= 1:
        raise ValueError("--tb-error-high-ratio-threshold must be between 0 and 1, inclusive.")
    if not 0 <= args.slot_ratio_threshold <= 1:
        raise ValueError("--slot-ratio-threshold must be between 0 and 1, inclusive.")

    num_slots, num_cells = per_cell_num_sche_ues.shape
    if args.tb_error_high_ratio_skip_initial_slots < 0:
        raise ValueError("--tb-error-high-ratio-skip-initial-slots must be non-negative.")
    if args.tb_error_high_ratio_skip_initial_slots >= num_slots:
        raise ValueError(
            "--tb-error-high-ratio-skip-initial-slots must be smaller than "
            f"the number of slots in the dataset ({num_slots})."
        )
    cells_to_plot = validate_cell_indices(args.cells, num_cells)
    plot_save_path = make_plot_save_path(args.save_plot_dir, h5_filename) if args.save_plot else None
    validation_failures = find_per_cell_validation_failures(
        per_cell_num_sche_ues,
        per_cell_tb_err,
        per_cell_ins_rate,
        cells_to_plot,
        args.tb_error_smoothing_window,
        args.ins_rate_smoothing_window,
    )
    high_tb_error_ratio_failure = find_high_tb_error_ratio_failure(
        per_cell_tb_err,
        args.tb_error_smoothing_window,
        args.tb_error_high_value_threshold,
        args.tb_error_high_ratio_threshold,
        args.tb_error_high_ratio_skip_initial_slots,
    )
    if high_tb_error_ratio_failure is not None:
        validation_failures.append(high_tb_error_ratio_failure)
    if args.apply_slot_ratio_threshold:
        scheduled_slot_ratio_failure = find_scheduled_slot_ratio_failure(
            per_cell_num_sche_ues, args.slot_ratio_threshold
        )
        if scheduled_slot_ratio_failure is not None:
            validation_failures.append(scheduled_slot_ratio_failure)

    if args.show_plot or args.save_plot:
        plot_per_cell_statistics(
            per_cell_num_sche_ues,
            per_cell_tb_err,
            per_cell_ins_rate,
            cells_to_plot,
            args.tb_error_smoothing_window,
            args.ins_rate_smoothing_window,
            plot_save_path,
            args.show_plot,
        )

    if validation_failures:
        print("Summary - cuMAC multi-cell MU-MIMO scheduler simulation post analysis test: FAIL")
        for failure in validation_failures:
            print(failure)
        return 1
    else:
        print("Summary - cuMAC multi-cell MU-MIMO scheduler simulation post analysis test: PASS")
        return 0


if __name__ == "__main__":
    sys.exit(main())

<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# gpu3gppchan test harness

This folder holds the regression harness for the 3GPP statistical channel
models. It lets developers and SDK users check channel-model calibration and
basic robustness on a supported GPU system.

## What it does

Two independent legs, one entry point:

| Leg | Script | What it checks |
| --- | --- | --- |
| Calibration | `run_cicd_calibration.sh` | UMa phase 1/2 (seeds 0–4) + ISAC UAV phase 1/2, target & background (seeds 0–49), reusing `../util/run_sls_chan_multiseed.sh`. Gates on **KS statistics** of every metric CDF against `ks_thresholds.yaml`. |
| Parameter sweep | `run_param_sweep.sh` | One-parameter-at-a-time variants of the UMa 6 GHz baseline config (`../config/statistic_channel_config.yaml`), each run under **compute-sanitizer memcheck** with a single fixed seed. Pass = runs to completion, zero sanitizer errors. No curve/reference comparison. |

`run_cicd_all.sh` runs both legs and returns a single exit code. It can be run
manually or used as the entry point for a CI job. Pipeline registration is not
part of this folder.

## Layout

```text
test/
├── README.md                 # this file
├── run_cicd_all.sh           # CICD entry point: calibration + sweep
├── run_cicd_calibration.sh   # calibration leg (two builds + KS gate)
├── run_param_sweep.sh        # sweep leg (compute-sanitizer smoke test)
├── check_ks_gate.py          # KS gate / threshold derivation
├── apply_sweep_params.py     # expands sweep_params.yaml into variant configs
├── sweep_params.yaml         # EDITABLE sweep specification
└── ks_thresholds.yaml        # generated KS thresholds (see below)
```

Results land in `./gpu3gppchan_cicd_results/` under the current working
directory (gitignored).

## Prerequisites

- Run **inside the cuBB container** from the SDK root (same assumptions as
  `util/run_all_calibrations.sh`), with a GPU visible.
- The regular `build.aarch64` configured (Rel-19 parameters, i.e.
  `TR38901_REL18_PARAMS=OFF` — the default).
- `compute-sanitizer` in `PATH` (ships with the CUDA toolkit).
- Python dependencies from `../util/requirements.txt`, installed in the
  container's `/opt/nvidia/cuBB/.venv`.

## Usage

```bash
cd /opt/nvidia/cuBB                                   # SDK root, inside container
/opt/nvidia/cuBB/.venv/bin/pip install -r testBenches/gpu3GPPChan/util/requirements.txt
testBenches/gpu3GPPChan/test/run_cicd_all.sh           # full test (~25–35 min on GH200)

# individual legs
testBenches/gpu3GPPChan/test/run_cicd_calibration.sh --isac-only
testBenches/gpu3GPPChan/test/run_param_sweep.sh --tool racecheck
```

The scripts use `/opt/nvidia/cuBB/.venv/bin/python` by default. Set
`PYTHON_BIN` to another Python 3.10 virtual-environment interpreter when using
an external host environment.

## Calibration leg details

**Two builds are required** because the 3GPP parameter vintage is a
compile-time option:

- UMa legs compare against the legacy CMCC references
  (`util/3gpp_calibration_phase{1,2}.json`) which need **`TR38901_REL18_PARAMS=ON`**.
  The script maintains a dedicated `build.aarch64.gpu3gppchan_rel18/` for this
  (configured automatically on first run, `CMAKE_CUDA_ARCHITECTURES=native`).
- ISAC legs compare against Rel-19 references
  (`util/3gpp_calibration_isac_uav_phase{1,2}.json`) and use the regular
  `build.aarch64` (flag **OFF** — the script verifies this in `CMakeCache.txt`).
- ISAC phase-2 delay and angle spreads come from the generated target/RP
  paths saved under `topology/isac*Links/calibrationPaths`. Target paths form
  one channel sample per drop; monostatic background paths form one sample
  per reference point (three samples per drop), as required by 38.901
  Section 7.9.6.2.
- ISAC model delays are stored internally in nanoseconds. A seconds conversion
  factor is applied only at Hz-based sample/phase equations. HDF delay metadata
  is unit-explicit: `isacTargetLinks/total_delay_ns` and
  `calibrationPaths/delay_ns` are both nanoseconds. This schema is identified
  by `gpu3gppchan_format_version` 3.

Symptom of a vintage mix-up: the UMa phase-2 ASD CDF collapses (median ≈5°
instead of ≈20°).

The KS statistics are captured from the analysis logs
(`<leg>/analysis.log`, written because the harness passes
`--log-level INFO --log-file …` via `ANALYSIS_EXTRA_ARGS`) and compared
against `ks_thresholds.yaml` by `check_ks_gate.py`.

### Regenerating `ks_thresholds.yaml`

Thresholds are tied to the CICD seed counts (UMa 5, ISAC 50). After an
**intentional** model change that legitimately shifts the CDFs:

1. Confirm the new curves are good the usual way (inspect the CDF PNGs in the
   result dirs against the 3GPP ranges).
2. Re-derive: `run_cicd_calibration.sh --derive-thresholds`
   (threshold = min(1.0, max(observed × 1.5, observed + 0.02)) per metric).
3. Commit the updated `ks_thresholds.yaml` together with the model change.

## Sweep leg details

Edit `sweep_params.yaml` to decide what gets swept — one variant is generated
per (parameter, value) pair under `sweep:`, starting from the baseline config,
with global `overrides` applied to every variant. The optional `seeds:` list
runs every variant once per seed (`test_bench.rand_seed`, `_s<seed>` name
suffix). Sweep values equal to the base config's own default are skipped
automatically (noted on stderr), and list values in `overrides:` are rejected
with a pointer to `seeds:`/`sweep:`. Commented ready-to-enable examples are
included; the parameter list is owned by the channel-model team and is
expected to grow over time.

`--tool` selects the compute-sanitizer tool (`memcheck` default; `racecheck`,
`initcheck`, `synccheck`, or `none` for a plain run). The selected sanitizer is
applied only when `system_level.n_site <= 3`; variants with more sites run
plain to keep the sweep runtime bounded. RMa memcheck variants exclude
`convolveCRNKernel` from instrumentation because its large spatial convolution
dominates sanitizer runtime. The kernel still executes normally. Other
scenarios and sanitizer tools retain full kernel instrumentation. Each
variant's full output is kept in `runs/<variant>/run.log`; bulky H5 artifacts
are deleted on pass unless `--keep-artifacts` is given.

## Known quirks

- CPU-only variants (`cpu_only_mode: 1`) automatically run without
  compute-sanitizer — an application that makes no CUDA call is itself an
  error to the sanitizer.
- Variants with `system_level.n_site > 3` automatically run without
  compute-sanitizer because instrumentation makes the larger configurations
  prohibitively slow.
- RMa variants run `convolveCRNKernel` without memcheck instrumentation. A
  focused GH200 measurement reduced the RMa seed-0 case from approximately
  20m46s to 6m35s while retaining memcheck coverage for every other kernel.

- ISAC phase-2 spread analysis fails closed when `calibrationPaths` is absent;
  regenerate older H5 files before comparing target/background baselines.
- `run_sls_chan_multiseed.sh` seds the seed into the YAML config in place; the
  calibration leg therefore feeds it a **private copy** of each config so the
  tracked `config/` directory is never touched.
- Runtime on GH200: calibration ≈ 20–30 min (dominated by ISAC 4×50 seeds +
  the one-off REL18 build), sweep ≈ a few minutes with the default spec.

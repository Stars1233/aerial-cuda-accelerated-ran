# testBenches - Agent Guide

Testing and benchmarking tools for the Aerial SDK (NVIDIA 5G PHY baseband,
C++20/CUDA). This subsystem covers GPU-only latency/capacity benchmarking,
3GPP channel models, end-to-end (CPU+GPU) Aerial System Benchmark test automation, and the
CICD perf-case entry points. See `README.md` for full usage; this file is the
orientation map for agents.

## Layout

```text
testBenches/
├── cubb_gpu_test_bench/   # GPU-only perf testbench (C++/CUDA)
├── perf/                  # Python driver + analysis for cubb_gpu_test_bench
│   ├── cicd_cases/        # 80000-series Aerial GPU Benchmark CICD perf cases
│   └── nsys_analysis/     # Nsight trace parsing and local latency reports
├── gpu3GPPChan/            # 3GPP 38.901 channel models (TDL/CDL, UMa/UMi/RMa)
├── phase4_test_scripts/   # End-to-end cuBB test automation (DU+RU emulators)
├── doca_samples/          # DOCA GPUNetIO sample
├── CMakeLists.txt         # project(testBenches), C++20, CUDA
└── README.md
```

## Purpose by component

- **cubb_gpu_test_bench/**: GPU-only testbench. Runs multi-channel/multi-cell
  workloads under MPS or Green Context, measures per-channel GPU latency via CUDA events, builds
  latency CDFs, and determines achievable cell counts. Setup/CPU time is NOT
  measured. Core C++ sources: `cubb_gpu_test_bench.cpp`, `cuphy_testWrkr.*`,
  `cumac_testWrkr.*`, `testbench_common.*`.
- **perf/**: Python interface that drives the C testbench. `measure.py` is the
  main entry (takes a single YAML config; CLI flags override). `compare.py`,
  `analyze_*.py`, `capacity*.py`, `power.py`, `memory_plot.py`,
  `confirm_cell_capacity.py` handle visualization and capacity confirmation.
  `copy_tvs_from_yaml.sh <yaml> <src> <dst>` copies the .h5 TVs named in the
  YAML from src to dst (it does not auto-discover or sync).
- **gpu3GPPChan/**: 3GPP TR 38.901 channel-model library + example apps
  (`sls_chan_ex`, `tdl_chan_ex`, `cdl_chan_ex`). Used by cuPHY/cuMAC sim.
- **phase4_test_scripts/**: Orchestrates real end-to-end runs across DU and RU
  nodes (RU emulator + cuPHYcontroller + testMAC). See its own `README.md`.

## CI/CD relationship

- **perf/cicd_cases/** is the CICD entry point for the 80000-series Aerial GPU Benchmark perf
  tests. `cicd_cases.json` (schema_version 2, per-GPU `gpu_configs`) is the
  source of truth for SM allocation, expected cell capacity, freq, and power.
  `run_cicd_cases.py` reads it, generates standalone per-case shell scripts
  (`cubb_gpu_test_<case>.sh`, untracked), and runs them. Internal (NVIDIA CICD
  only): `internal/` publishes results to OpenSearch/Grafana Aerial GPU Benchmark dashboards;
  external users can ignore that directory.
- **phase4_test_scripts/parse_test_config_params.{sh,py}** turns a test-case
  string (e.g. `F08_6C_79_MODCOMP_STT480000_EH_1P`) plus a host config
  (`CG1_R750`, `CG1_CG1`, `GL4_R750`) into a `test_params.sh` of env vars
  consumed by the `build/setup/run/post_processing` scripts.
- **post_processing_cicd.sh** is the CICD wrapper for log analysis and threshold
  gating. Exit codes: `0` = pass, `1` = gating/absolute threshold failed or
  error, `2` = pass but warning threshold exceeded. `post_processing_parse.sh`
  (expensive parse) and `post_processing_analyze.sh` (downstream) are the
  granular pair; `post_processing_defaults.cfg` is the single source of truth
  for timing windows.
- **phase4_test_scripts/syscall_tracer/**: futex/CUDA-API and page-fault
  attribution used by the `PERF` test modifier and post-processing steps 11-12.
- **phase4_test_scripts/server_version_check/**: validates cluster node versions
  (BMC/BIOS/kernel/driver/CUDA) against manifest CSVs.

## Build

`cubb_gpu_test_bench` and channel-model examples build as part of the standard
Aerial SDK. Preferred:

```shell
$cuBB_SDK/testBenches/phase4_test_scripts/build_aerial_sdk.sh
# specific targets:
$cuBB_SDK/testBenches/phase4_test_scripts/build_aerial_sdk.sh --targets cubb_gpu_test_bench
```

Manual CMake target names: `testbenches_examples` (all), `cubb_gpu_test_bench`,
`sls_chan_ex`, `tdl_chan_ex`, `cdl_chan_ex`. C++20 is required (set in
`CMakeLists.txt`).

## Python environment

Post-processing scripts need the `aerial_postproc` venv. External-facing: create
it with `phase4_test_scripts/aerial_postproc/venv_create.sh` (default
`$HOME/.aerial_postproc_venv`, override via `AERIAL_POSTPROC_VENV`). Internal:
CICD images pre-install it at `/opt/aerial_postproc_venv`. For repo-general
Python work, first use an applicable component or task venv when it exists.
Otherwise use the cuBB root `.venv` when present (`/opt/nvidia/cuBB/.venv`),
then the container's system-wide Python 3.12 installation.

## Test vectors (TVs)

TVs are H5 files generated in MATLAB from `<aerial_sdk>/5GModel`. They are large
binary inputs, not committed source. External-facing: see `README.md` (Test
Vectors) for obtaining the full/compact perf TV sets or generating them from
`5GModel`. `perf/.gitattributes` and `perf/.gitignore`
govern what is tracked. Keep generated artifacts (per-case `cubb_gpu_test_*.sh`,
`runs/`, `vectors-*.yaml`, `buffer-*.txt`, results JSON) out of commits.

## Invariants - read before editing

- **Synthetic data only** (see root `AGENTS.md` -> `## Fixtures & data`).
  Delta for this subsystem: the phase-4 scripts SSH into DU/RU nodes, so
  **credentials** belong in env vars / local files (e.g. `~/aerial_pw` read via
  `SSHPASS`), never checked in.
- **`cicd_cases.json` is the source of truth** for Aerial GPU Benchmark CICD perf parameters.
  `expected_cell_capacity` values are owned by `phase3_statsupdate.py` and are not manually edited.
  SM allocation lives in `gpu_configs.<GPU>`; the README snapshot row and
  `cubb_gpu_test_config_<case>.yaml` document the same values and must stay consistent with it.
- **Don't break post-processing exit-code contract** (`0`/`1`/`2`) - CICD gating
  (and, internally, Slack notifications) depend on it.
- **Keep `post_processing_defaults.cfg` the single source** for timing windows;
  do not hardcode durations in the parse/analyze scripts.
- Preserve SPDX/Apache-2.0 license headers on new scripts and sources.
- Prefer the YAML-config + CLI-override path for `measure.py`; the explicit-JSON
  workflow is legacy.

## Documentation

This file and READMEs are read by both Claude Code and Codex. Keep additions
factual (only document what exists in these files) and in plain Markdown.

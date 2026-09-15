# Aerial PostProc Dashboard Scripts

This directory contains scripts for preparing and uploading Aerial performance data to the NVIDIA internal dashboard.

The code uses `phase_2` and `phase_4` as the `--test_phase` values. These names identify different dashboard data schemas, not the numbered steps in the upload workflow.

## Supported Data Paths

There are two dashboard data paths supported by these scripts today.

- `phase_2`: standalone NCU kernel profiling data. Use `dashboard_upload.sh` for this path. The script expects a directory with `report.ncu-rep`, creates `metadata.txt`, runs `metrics_extraction.py --test_phase phase_2`, and optionally runs `metrics_upload.py --test_phase phase_2 --upload_opensearch`.
- `phase_4`: system-level CI run data from the cuPHY control-plane test logs. Jenkins jobs call `metrics_extraction.py` and `metrics_upload.py` directly for this path. `dashboard_upload.sh` is not the `phase_4` uploader because it always invokes the scripts with `--test_phase phase_2`.

Both upload paths post through NVDataFlow to the OpenSearch-backed index used by Grafana. The default index is `swgpu-aerial-perflab-cicd`.

## Dashboard Definitions

The Grafana dashboards that visualize this data are defined in a separate repository: [`cicd-grafana-dashboards`](https://gitlab-master.nvidia.com/gputelecom/cicd-grafana-dashboards) (sibling of `aerial_sdk` in the workspace).

That repo holds the dashboard JSON sources and the provisioning script that publishes them to `https://grafana.nvidia.com`:

- `dashboards/Summary/db_summary_view.json` — Phase 4 system-level summary view (`/d/db_summary_view/summary-view`).
- `dashboards/Phase_2/db_phase2_engineering.json` — Phase 2 NCU kernel view (`/d/db_phase2_engineering/engineering-view`).
- `dashboards/Engineer/` — CCDF, clock, CPU, power, temperature, and engineering analysis dashboards.
- `dashboards/Timeline/db_timeline_system_phase4.json` — Phase 4 timeline view.
- `scripts/grafana_db_provisioning.sh` — uploads the JSONs via the Grafana API; uses the `yesoreyeram-infinity-datasource` pointed at OpenSearch `https://gpuwa.nvidia.com/opensearch/df-swgpu-aerial-perflab-*` (the same index family written by `metrics_upload.py`).
- `scripts/grafana_db_save.sh` — pulls dashboards back from Grafana into the repo.

End-to-end flow: CI artifacts → `metrics_extraction.py` → `metrics_upload.py` (NVDataFlow) → OpenSearch `swgpu-aerial-perflab-*` → Grafana Infinity datasource → dashboards defined in `cicd-grafana-dashboards`.

## Requirements

- **NVIDIA Internal Network**: Required for accessing NVIDIA's internal PyPI server (`sc-hw-artf.nvidia.com`) and download nvdataflow2 version 1.0.7.
- **uv**: Python package manager ([install](https://docs.astral.sh/uv/getting-started/installation/), verify with `uv --version`) (available in ACAR container)
- **NCU (Nsight Compute)**: For GPU kernel profiling (available in CUDA toolkit and ACAR container)

## Phase 2 NCU Kernel Metrics

Phase 2 dashboard publishing uses the `phase_2` mode of `metrics_extraction.py` and `metrics_upload.py`. This data path is intended for standalone NCU kernel profiling reports, not full CI scenario folders.

The scenario folder must contain `metadata.txt` at its root. This required input provides the `jenkins_pipeline`, `test_case`, and `jenkins_id` fields used to identify the dashboard documents. The `dashboard_upload.sh` wrapper generates `metadata.txt` automatically from its `--pipeline`, `--channel`, and `--tv` flags; the `test_case` is `run{channel}_{test_vector}`.

For Phase 2 metrics, the scenario folder should include:

- `metadata.txt` - required for extraction.
- `report.ncu-rep` - the NCU profiling report. Missing or unreadable `report.ncu-rep` does not abort extraction, but no kernel metrics are written and the run is marked with `report.ncu-rep was missing` in `info`.

The extractor writes (alongside the inputs in the scenario folder):

- `dashboard_data.json` - test run information.
- `kernel_metrics.csv` - per-kernel metrics post-processed from the NCU report.

`metrics_extraction.py` produces these via its `phase_2()` and `store()` flow. `metrics_upload.py` then reads `dashboard_data.json` and `kernel_metrics.csv` and posts to NVDataFlow.

The Phase 2 engineering view is at `https://grafana.nvidia.com/d/db_phase2_engineering/engineering-view?orgId=146`, defined in `cicd-grafana-dashboards/dashboards/Phase_2/` (see [Dashboard Definitions](#dashboard-definitions)).

### Workflow

Use this workflow for standalone NCU kernel profiling uploads.

#### Step 1: Run NCU Profiling

```bash
mkdir -p /tmp/my_results
ncu --set full -k 'regex:^(?!.*(convert_kernel|delay_kernel_us))' -o /tmp/my_results/report \
    ./build.x86_64/cuPHY/examples/pusch_rx_multi_pipe/cuphy_ex_pusch_rx_multi_pipe \
    -i GPU_test_input/TVnr_7201_PUSCH_gNB_CUPHY_s0p0.h5 -r 1
```

- `-k 'regex:...'` filters out irrelevant kernels (`convert_kernel`, `delay_kernel_us`)
- `-r 1` limits pipeline executables to a single iteration (ncu already repeats kernel runs)

> **Note**: If test vectors are not found, mount them when starting the container:
> ```bash
> export AERIAL_EXTRA_FLAGS="-v /mnt/cicd_tvs/develop/GPU_test_input:/opt/nvidia/cuBB/GPU_test_input:ro"
> ./cuPHY-CP/container/run_aerial.sh
> ```

#### Step 2: Upload to Dashboard

Choose either the **automated** or **manual** method below.

##### Option A: Automated (Recommended)

```bash
./testBenches/phase4_test_scripts/aerial_postproc/scripts/dashboard/dashboard_upload.sh \
    /tmp/my_results \
    --pipeline cuphy_bench_local \
    --channel 0 \
    --tv TVnr_7201_PUSCH \
    --upload
```

The script handles metadata creation, metric extraction, and upload.

**Script options:**
```bash
./testBenches/phase4_test_scripts/aerial_postproc/scripts/dashboard/dashboard_upload.sh <ncu_report_dir> [OPTIONS]

  -p, --pipeline NAME  Pipeline name (default: cuphy_bench_local)
  -c, --channel N      Channel number: 0, 1, 2, ... (default: 0)
  -t, --tv NAME        Test vector name (e.g., TVnr_7201_PUSCH)
  -u, --upload         Upload to NVDF (default: extract only)
  -h, --help           Show help
```

##### Option B: Manual

For more control, run each step individually from the repository root.

**1. Create metadata** (test_case format: `run{channel}_{test_vector}`):

```bash
cat > /tmp/my_results/metadata.txt << 'EOF'
jenkins_pipeline=cuphy_bench_local
test_case=run0_TVnr_7201_PUSCH
jenkins_id=1-20260205-120000
jenkins_test_result=PASS
EOF
```

**2. Extract metrics:**

```bash
PYTHONPATH=testBenches/phase4_test_scripts/aerial_postproc:$PYTHONPATH \
uv run --index https://sc-hw-artf.nvidia.com/artifactory/api/pypi/hwinf-gpuwa-pypi/simple \
    --with-requirements testBenches/phase4_test_scripts/aerial_postproc/scripts/dashboard/requirements.txt \
    testBenches/phase4_test_scripts/aerial_postproc/scripts/dashboard/metrics_extraction.py \
    /tmp/my_results --test_phase phase_2
```

**3. Upload:**

```bash
PYTHONPATH=testBenches/phase4_test_scripts/aerial_postproc:$PYTHONPATH \
uv run --index https://sc-hw-artf.nvidia.com/artifactory/api/pypi/hwinf-gpuwa-pypi/simple \
    --with-requirements testBenches/phase4_test_scripts/aerial_postproc/scripts/dashboard/requirements.txt \
    testBenches/phase4_test_scripts/aerial_postproc/scripts/dashboard/metrics_upload.py \
    /tmp/my_results --test_phase phase_2 --upload_opensearch
```

#### Step 3: View Results in Grafana

https://grafana.nvidia.com/d/db_phase2_engineering/engineering-view?orgId=146

Select your pipeline and channel to view kernel metrics.

## Phase 3 Status

The dashboard scripts currently accept only `phase_2` and `phase_4`; there is no `phase_3` dashboard extraction or upload mode in this directory yet. Phase 3 should be treated as a separate data path, not as the third step of either workflow above.

Existing repository code suggests Phase 3 would likely publish unit-test performance data from `Aerial_cuphy_container_unit_test_phase3_perf_pipeline`. The local Phase 3 analysis tools parse Phase 3 text output and Nsight Systems CSV timing data, including slot-pattern timings and kernel timelines. That is different from Phase 2 NCU kernel metrics and Phase 4 system-level log/perf CSV publishing.

Until a `phase_3` mode is added to `metrics_extraction.py`, `metrics_upload.py`, and the dashboard schema, Phase 3 dashboard publishing should be treated as planned or in-progress rather than supported by this uploader.

## Phase 4 System-Level CI Data

Phase 4 dashboard publishing uses the `phase_4` mode of `metrics_extraction.py` and `metrics_upload.py`. This data path is intended for complete CI scenario folders, not standalone NCU report directories.

The scenario folder must contain `metadata.txt` at its root. This required input provides the `jenkins_pipeline`, `test_case`, and `jenkins_id` fields used to identify the dashboard documents. Optional metadata such as `test_duration`, `mr`, and `jenkins_test_result` is included in the dashboard payload when present.

The `test_case` value is parsed for Phase 4 attributes using the shared `parse_tc_info()` helper. For names such as `F08_20C_59c_BFP9_STT455000_EH_GC_1P`, it reads `20C` as the cell count, `59c` as the traffic pattern, `BFP9` as the BFP mode, and the `EH` and `GC` flags.

For full Phase 4 metrics, the scenario folder should include:

- `metadata.txt` - required for extraction.
- `cuphy/perf_results/perf.csv` - required for full headroom and performance metrics. Missing or unreadable `perf.csv` causes an extraction error and prevents complete output.

The extractor can add more metrics when these files are present:

- `cuphy/phy.log` - used for reference timing and PHY/on-time parsing.
- `mac/testmac.log` - used for MAC throughput parsing.
- `ru/ru.log` - used for RU throughput and on-time parsing.
- `cuphy/perf_results/power.csv` - included in upload when present; missing or unreadable power CSVs disable power metrics.
- `cuphy/perf_results/power_summary.csv` - included in upload when present; missing or unreadable power CSVs disable power metrics.

When these files are unavailable, the corresponding derived metrics are omitted and the run information can record missing-file or extraction-stage details in `info`.

The extractor writes:

- `cuphy/perf_results/dashboard_data.json`
- generated CSVs under `cuphy/perf_results/binary/`, such as:
  - `mac_thr_obs.csv`
  - `ru_thr_obs.csv`
  - `allocation_sm.csv`
  - `allocation_cpu.csv`
  - `utilization_cpu.csv`

`metrics_extraction.py` creates these outputs through its `save_files_data` and `store()` flow. `metrics_upload.py` then reads `dashboard_data.json`, consumes generated allocation and utilization CSVs from `cuphy/perf_results/binary/`, and passes the binary folder to `PerfMetricsIO`.

The upload step also reads these existing postprocessed artifacts when present:

- `cuphy/perf_results/perf.csv`
- `cuphy/perf_results/power.csv`
- `cuphy/perf_results/power_summary.csv`
- `cuphy/perf_results/binary/`

Run extraction from the repository root:

```bash
PYTHONPATH=testBenches/phase4_test_scripts/aerial_postproc:$PYTHONPATH \
uv run --index https://sc-hw-artf.nvidia.com/artifactory/api/pypi/hwinf-gpuwa-pypi/simple \
    --with-requirements testBenches/phase4_test_scripts/aerial_postproc/scripts/dashboard/requirements.txt \
    testBenches/phase4_test_scripts/aerial_postproc/scripts/dashboard/metrics_extraction.py \
    /path/to/scenario --test_phase phase_4
```

Run upload after extraction:

```bash
PYTHONPATH=testBenches/phase4_test_scripts/aerial_postproc:$PYTHONPATH \
uv run --index https://sc-hw-artf.nvidia.com/artifactory/api/pypi/hwinf-gpuwa-pypi/simple \
    --with-requirements testBenches/phase4_test_scripts/aerial_postproc/scripts/dashboard/requirements.txt \
    testBenches/phase4_test_scripts/aerial_postproc/scripts/dashboard/metrics_upload.py \
    /path/to/scenario --test_phase phase_4 --upload_opensearch
```

The upload step writes `cuphy/perf_results/opensearch_payload.json` before posting to NVDataFlow. Pass `--index <name>` to override the default `swgpu-aerial-perflab-cicd` index.

In CI, Phase 4 upload is controlled by the Jenkins/shared-library path that decides whether to pass `--upload_opensearch`. If the test-case entry does not enable upload, the dashboard data can be extracted locally but will not be posted to OpenSearch.

The Phase 4 summary view is at `https://grafana.nvidia.com/d/db_summary_view/summary-view?orgId=146`; the timeline view is `https://grafana.nvidia.com/d/db_timeline_system_phase4/...`. Both are defined in `cicd-grafana-dashboards` (see [Dashboard Definitions](#dashboard-definitions)).

## Note

For users without NVIDIA internal access, the main aerial_postproc functionality remains available using the standard venv setup in the parent directory. Dashboard features require NVIDIA internal network access.

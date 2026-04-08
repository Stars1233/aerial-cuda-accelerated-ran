# Aerial PostProc Dashboard Scripts

This directory contains scripts for uploading NCU kernel metrics to the NVIDIA internal dashboard.

## Requirements

- **NVIDIA Internal Network**: Required for accessing NVIDIA's internal PyPI server (`sc-hw-artf.nvidia.com`) and download nvdataflow2 version 1.0.7.
- **uv**: Python package manager ([install](https://docs.astral.sh/uv/getting-started/installation/), verify with `uv --version`) (available in ACAR container)
- **NCU (Nsight Compute)**: For GPU kernel profiling (available in CUDA toolkit and ACAR container)

## Step 1: Run NCU Profiling

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

## Step 2: Upload to Dashboard

Choose either the **automated** or **manual** method below.

### Option A: Automated (Recommended)

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
./dashboard_upload.sh <ncu_report_dir> [OPTIONS]

  -p, --pipeline NAME  Pipeline name (default: cuphy_bench_local)
  -c, --channel N      Channel number: 0, 1, 2, ... (default: 0)
  -t, --tv NAME        Test vector name (e.g., TVnr_7201_PUSCH)
  -u, --upload         Upload to NVDF (default: extract only)
  -h, --help           Show help
```

### Option B: Manual

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

## Step 3: View Results in Grafana

https://grafana.nvidia.com/d/db_phase2_engineering/engineering-view?orgId=146

Select your pipeline and channel to view kernel metrics.

## Note

For users without NVIDIA internal access, the main aerial_postproc functionality remains available using the standard venv setup in the parent directory. Dashboard features require NVIDIA internal network access.

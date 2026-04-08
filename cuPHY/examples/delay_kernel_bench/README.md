# Delay Kernel Benchmark

This example provides a minimal benchmark for `gpu_us_delay(...)` used by cuPHY testbenches.

- Source: `cuphy_ex_delay_kernel_bench.cpp`
- Delay helper implementation: `cuPHY/examples/common/util.cu`
- Delay CUDA kernel: `delay_kernel_us` in `cuPHY/examples/common/util.cu`

## Build

This target is added to `cuPHY/examples/CMakeLists.txt` as:

- `cuphy_ex_delay_kernel_bench`

Build with your existing Aerial SDK build flow, or build this target directly in your configured build directory.

## CLI Arguments

```bash
cuphy_ex_delay_kernel_bench [-r <iterations>] [-d <delay_us>] [-t <period_us>] [-p <0|1>] [-s <sm_count>] [-e]
```

- `-r`: number of launches (default: `100`)
- `-d`: delay-kernel runtime in microseconds (default: `1000`)
- `-t`: launch period from CPU in microseconds (default: `5000`)
- `-p`: launch pattern
  - `0`: multi-SM / all blocks
  - `1`: single block (approximately 1 SM)
  - default: `1`
- `-s`: optional SM count request for MPS subcontext
  - default: `0`
  - omit `-s` (or use `-s 0`) to run without MPS subcontext
  - use `-s > 0` to request subcontext partitioning
- `-e`, `--event-timing`: optional CUDA event timing summary (min/mean/max kernel duration)
  - note: this mode performs per-launch `cudaEventSynchronize`, which serializes launches and may increase observed launch spacing versus runs without `-e`

Examples:

```bash
# Use all defaults
cuphy_ex_delay_kernel_bench

# Custom timing and launch pattern
cuphy_ex_delay_kernel_bench -r 2000 -d 120 -t 500 -p 1

# Enable optional event timing summary
cuphy_ex_delay_kernel_bench -e
```

## MPS Notes

If `-s > 0`, you should start CUDA MPS daemon manually before running. For example:

```bash
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps
export CUDA_LOG_DIRECTORY=/tmp/nvidia-mps
mkdir -p "${CUDA_MPS_PIPE_DIRECTORY}"
CUDA_VISIBLE_DEVICES=0 nvidia-cuda-mps-control -d
```

Stop it when done:

```bash
echo quit | CUDA_VISIBLE_DEVICES=0 CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps CUDA_LOG_DIRECTORY=/tmp/nvidia-mps nvidia-cuda-mps-control
```

## NVTX Markers

The benchmark emits NVTX ranges:

- `delay_kernel_benchmark` (whole run)
- `gpu_us_delay_launch` (per launch)

When `-e` is enabled, the benchmark also prints CUDA event timing summary
(`min/mean/max`) for kernel duration.

# Syscall Tracer – Perf Capture for Phase-4 cuBB

This directory contains scripts and utilities to capture **futex** (and related) syscalls during Phase-4 cuBB test runs, attribute them to call stacks (and thus to CUDA APIs where applicable), and produce summary statistics. The primary tool used is **perf**.

## Introduction

Understanding the impact of kernel syscalls—in particular **futex** (fast userspace mutex)—on real-time PHY workloads is important for validating and tuning the cuBB ACAR stack. Key goals include:

- **Quantifying** how often CUDA API calls lead to futex invocations in the critical path (e.g., DL/UL worker threads).
- **Evaluating** how a custom-built soft real-time drop from CUDA reduces or alleviates this syscall overhead and improves determinism.

To support this, we capture `sys_enter_futex` events during Phase-4 cuBB test runs, attribute them to call stacks (and thus to CUDA APIs where applicable), and produce summary statistics.

---

## Steps to Capture Syscalls (Futex) and Generate Summary Statistics

### 1. Enable Host PID Namespace in the Container

In the aerial container file, use **`--pid=host`** so that the host can see the same PIDs/TIDs as the processes running inside the container. This is required for `perf record -t <TID>` to attach to the correct threads.

### 2. Install perf and Kernel Tracing Tools

```bash
sudo apt-get update
sudo apt install linux-tools-generic linux-tools-$(uname -r) -y
```

### 3. Identify cuBB Worker Thread IDs (TIDs)

After **cuphycontroller_scf** is running, list the DL/UL worker TIDs (and PID) that you will pass to `perf record`.

**Important:** Make sure that **traffic flow is not started** before you start recording with perf.

```bash
./testBenches/phase4_test_scripts/syscall_tracer/list_cuphy_worker_tids.sh
```

- Default process name: **cuphycontroller_scf** (override with an optional argument).
- **Output:** Human-readable table of PID, DL worker TIDs, UL worker TIDs, and other threads spawned by the process.
- Use the displayed worker TIDs in the `perf record` step below (e.g. `-t TID1,TID2,...`).

### 4. [Optional] Check Default perf Ring Buffer and Kernel Limits

Use this when tuning buffer size or debugging empty captures.

**Tracepoint ID for futex:**

```bash
sudo cat /sys/kernel/debug/tracing/events/syscalls/sys_enter_futex/id
```

(Replace `sys_enter_futex` with any other event name if tracing different syscalls.)

**Build and run the small utility** that reports effective ring buffer layout for a given tracepoint and TID (utility is under `./testBenches/phase4_test_scripts/syscall_tracer`):

```bash
gcc -O2 -Wall perf_buf_size.c -o perf_buf_size
sudo ./perf_buf_size <trace_point_ID_from_above> <TID>
```

The `data_size` value reflects the ring buffer size used for that event.

**Per-CPU perf memory limit (mlock):**

```bash
cat /proc/sys/kernel/perf_event_mlock_kb
```

If captures are dropped, consider increasing this (with appropriate system limits) and/or using a larger `-m` value in `perf record`.

### 5. Record futex Syscalls with perf

Pin perf to a specific CPU, record only the worker TIDs (no child inheritance), and capture call stacks:

```bash
sudo taskset -c 58 perf record --no-inherit -e syscalls:sys_enter_futex -t <TID1>,<TID2>,<TID3>,<TID4>,<TID5> -m 4M -g --call-graph fp -o /tmp/perf_syscalls.data
```

| Option | Purpose |
|--------|--------|
| `taskset -c 58` | Run perf on CPU 58 to reduce interference with the workload. |
| `--no-inherit` | Do not attach to child threads; only the listed TIDs are traced. |
| `-e syscalls:sys_enter_futex` | Trace only futex entries. |
| `-t <TID1>,<TID2>,...` | Comma-separated list of worker TIDs from step 3. |
| `-m 4M` | Per-CPU ring buffer size (tune if you see drops). |
| `-g --call-graph fp` | Record call stacks using frame pointers (required for CUDA API attribution). |
| `-o /tmp/perf_syscalls.data` | Output perf data file. |

Adjust the CPU id, TIDs, buffer size, and output path as needed for your run.

**Once the above command is running, start the cuBB Phase-4 test** to initiate DL/UL traffic exchange. Based on multiple experiments so far, **single cell peak traffic pattern 59c for 5s (10K slots)** is seen to generate perf captures with reliability (see *Challenges and Issues with perf* below).

### 6. [Optional] Inspect Raw Events and Stack Frames

To verify that events and stacks were captured:

```bash
sudo perf script -i /tmp/perf_syscalls.data
```

This prints one line per `sys_enter_futex` event plus the corresponding stack trace. Use it to confirm that CUDA frames (e.g. `libcuda.so`, `libcudart.so`) appear in the stacks when expected.

### 7. Generate Per-Run Summary (Futex Counts per CUDA API, per Thread)

```bash
python3 testBenches/phase4_test_scripts/syscall_tracer/futex_cuda_summary_multi_thread.py -i /tmp/perf_data -o /tmp/perf_data/perf_data_summary.txt
```

| Option | Purpose |
|--------|--------|
| `-i` | Path to the folder containing the perf data file(s) (or path prefix; the script discovers `*.data` files under the folder). |
| `-o` | Output summary file path. |

The script:

- Runs `perf script` on each discovered perf data file (with `sudo` by default; use `--no-sudo` if not needed).
- Parses `sys_enter_futex` events and their stack traces.
- Attributes each futex to the `(comm, tid)` of the event and to the **topmost CUDA API** in the stack (e.g. `libcuda.so`, `libcudart.so`).
- Writes a per-thread summary: total futex, CUDA-attributed counts per API, non-CUDA count, and a summary table.

Output file name is arbitrary; the default, if `-o` is omitted, is `<folder>/futex_cuda_summary_multi_thread.txt`.

**Example snippet** from a 1C 59c run:

```
Futex call counts per CUDA API, per thread (multi-thread: one file may contain many threads)
Generated by futex_cuda_summary_multi_thread.py
--- Thread: UlPhyDriver07 (TID 314281) ---
  Total futex events: 5858  (CUDA-attributed: 360, Non-CUDA: 5498)
  cuEventQuery                       cuEventRecord                      cuGraphExecKernelNodeSetParams_v2  cuGraphLaunch                      cuGraphUpload                      cuKernelSetAttribute               cuLaunchKernel                     cuMemcpyBatchAsync_v2              cuMemcpyHtoDAsync_v2               cuMemsetD8Async                    cuStreamWaitEvent
  0                                  12                                 0                                  7                                  326                                0                                  0                                  15                                 0                                  0                                  0
--- Thread: UlPhyDriver06 (TID 314282) ---
  ...
========== Summary table (thread x CUDA API + CUDA + Non-CUDA + Total) ==========
Thread (name / TID)    cuEventQuery cuEventRecord ... CUDA     Non-CUDA Total
-----------------------------------------------------------------------------
UlPhyDriver07 / 314281   0            12            ... 360      5498     5858
...
TOTAL                 24           1100           ... 8092     49648    57740
```

### 8. Collate Summary Across Multiple Runs

When you have multiple test runs (e.g. one subfolder per run, each with its own per-run summary from the previous step):

```bash
python3 testBenches/phase4_test_scripts/syscall_tracer/futex_cuda_runs_summary.py -i /tmp/<parent_folder>
```

| Option | Purpose |
|--------|--------|
| `-i` | Parent folder whose direct subfolders are treated as one run each. |

The script looks in each subfolder for a summary file: first `futex_cuda_summary_multi_thread.txt`, then any file whose name contains `summary`. It parses the **TOTAL** row from each per-run summary and produces:

- **Run summary table:** Total futex, Total CUDA, Total Non-CUDA, % CUDA, % Non-CUDA per run.
- **CUDA API breakdown:** Per run, per CUDA API: count and % of that run’s CUDA-attributed futex calls.

Output defaults to `<parent>/futex_cuda_runs_summary.txt`; override with `-o <path>`.

### 9. [Optional] CUDA API Tracing for the Entire cuBB Run

In addition to perf-based futex capture, you can trace **all supported CUDA API calls** (Driver + Runtime) for the entire cuBB run using the **LD_PRELOAD tracer**. This gives direct call counts per API, complementary to the futex-attribution summary.

**Build the tracing library:**

```bash
cd testBenches/phase4_test_scripts/syscall_tracer
./build_cuda_api_tracer.sh
```

**Run cuBB with tracer enabled** (uses `LD_PRELOAD` via the `--cuda_tracer` option; optionally specify output path):

```bash
$cuBB_SDK/testBenches/phase4_test_scripts/run2_cuPHYcontroller.sh --cuda_tracer /tmp/cuda_no_env_set/run3/cuda_api_tracer.log
```

**Include the tracer log in the overall summary statistics:**

```bash
python3 testBenches/phase4_test_scripts/syscall_tracer/futex_cuda_runs_summary.py -i /tmp/cuda_env_set --include-tracer-log
```

The script looks for a tracer log file in each run subfolder (default: any file with `cuda_api` in the name, e.g. `cuda_api_tracer.log`). It adds a third table comparing **tracer-based API totals** vs **futex-attributed counts** per run.

---

## Challenges and Issues with perf

- **Reliability:** Based on experimentation, **single cell peak traffic pattern 59c for 5s (10K slots)** has been found to generate perf captures with good reliability.
- **Drops:** If you see dropped events, increase the per-CPU ring buffer (`-m`) and/or `perf_event_mlock_kb` as described in step 4.
- **Empty captures:** Ensure traffic is started **after** `perf record` is running, and that you are recording the correct worker TIDs from step 3.

---

## File Reference

| File | Purpose |
|------|--------|
| `list_cuphy_worker_tids.sh` | List PID and DL/UL worker TIDs for cuphycontroller_scf. |
| `perf_buf_size.c` | Utility to report effective ring buffer layout for a tracepoint and TID. |
| `build_cuda_api_tracer.sh` | Build the LD_PRELOAD CUDA API tracer library. |
| `futex_cuda_summary_multi_thread.py` | Per-run futex → CUDA API attribution and per-thread summary. |
| `futex_cuda_runs_summary.py` | Collate per-run summaries; optional tracer log inclusion. |

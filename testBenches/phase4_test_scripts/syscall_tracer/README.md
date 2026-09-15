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

## Tracing ALL Syscall Types with bpftrace

The perf flow above is scoped to **futex** (and page faults) on purpose. Casting a wide net over **every** syscall with `perf record` *or* `perf stat` puts too much per-event overhead on the real-time workers and causes **timing errors** — the hot syscalls (`clock_nanosleep`, `write`, `getcpu`) fire tens of millions of times per run, and perf's per-event handler can't keep up without perturbing the workload.

For all-syscall coverage we instead use **bpftrace**, which aggregates counts **in-kernel** (per-CPU, lock-free) with no per-event export to userspace. This is light enough to run alongside the RT workers, and can even run **simultaneously with the perf futex capture** (on an orthogonal CPU core).

### Why bpftrace and not perf for all syscalls

| | Per-event work | Data export | RT impact |
|---|---|---|---|
| `perf record` (all syscalls) | build + copy a sample record every syscall | ring buffer → disk, continuously | worst (stalls + drops) |
| `perf stat` (all syscalls) | heavy generic handler across ~300 events/thread | totals at end | still too heavy per-event |
| **bpftrace count** | one per-CPU counter bump | totals at end only | light enough to fit the RT slack |

### Automated capture wrapper

`bpftrace_trace_workers.sh` is the end-to-end wrapper (analogous to `perf_trace_workers.sh`). It discovers the worker TIDs, resolves them to thread names, mounts debugfs if needed, runs bpftrace, and post-processes the result:

```bash
sudo ./testBenches/phase4_test_scripts/syscall_tracer/bpftrace_trace_workers.sh -d 5 -o /tmp/run1/bpftrace_syscalls.txt
```

| Option | Purpose |
|--------|--------|
| `-o PATH` | bpftrace raw map dump (default `/tmp/bpftrace_syscalls.txt`). |
| `-s PATH` | Summary table file (default: alongside the dump). |
| `-d SECS` | Auto-stop after SECS seconds (default: run until Ctrl-C). |
| `-c CPU_ID` | CPU to pin bpftrace to (default: **59**, orthogonal to perf's 58). |
| `--raw` | Use the lighter `raw_syscalls:sys_enter` single probe (numeric ids) instead of the named `syscalls:sys_enter_*` wildcard. |
| `--no-summary` | Skip post-processing; only write the raw dump. |

Two things this wrapper bakes in that are easy to get wrong by hand:

1. **debugfs must be mounted** for bpftrace: `sudo mount -t debugfs none /sys/kernel/debug` (per container lifetime), else it errors with *"Could not read .../available_events"*.
2. **Filter by `comm`, not TID.** bpftrace's eBPF programs see **host/init-namespace** TIDs, while `list_cuphy_worker_tids.sh` reports **container-local** TIDs — they never match, so a TID predicate silently captures nothing. The wrapper resolves the worker `comm`s (thread names, namespace-independent) and filters on those. (perf's `-t <tid>` works with container TIDs only because `perf_event_open` translates namespaces; eBPF does not.)

### Summary and pass/fail

`bpftrace_syscall_summary.py` parses the map dump into a table: **one row per syscall type, one column per DL/UL worker, plus a Total column** (sorted by Total descending, with a TOTAL row). It emits a verdict and sets its exit code:

- **PASS** (exit 0) if any syscalls were captured.
- **FAIL** (exit 1) if the capture is **empty** (no syscalls recorded).

```bash
python3 testBenches/phase4_test_scripts/syscall_tracer/bpftrace_syscall_summary.py -i /tmp/run1/bpftrace_syscalls.txt
```

The wrapper runs this automatically at the end (unless `--no-summary`) and exits with the verdict code.

### Driving it from the test harness

`run2_cuPHYcontroller.sh` supports `--bpftrace`, mirroring `--perf_trace`: it installs bpftrace if missing, waits for L1 readiness, then launches `bpftrace_trace_workers.sh` on the worker threads. perf and bpftrace can be enabled together and run on orthogonal cores (58 / 59):

```bash
# bpftrace only (all syscalls)
./run2_cuPHYcontroller.sh --bpftrace --bpftrace_dir /tmp/bpf

# perf (futex, CPU 58) + bpftrace (all syscalls, CPU 59) together
./run2_cuPHYcontroller.sh --perf_trace --perf_trace_dir /tmp/perf --bpftrace --bpftrace_dir /tmp/bpf
```

| Option | Purpose |
|--------|--------|
| `--bpftrace` | Launch `bpftrace_trace_workers.sh` once L1 is ready. |
| `--bpftrace_opts "<opts>"` | Options forwarded to the wrapper (e.g. `"--raw"`). |
| `--bpftrace_dir <dir>` | Output directory (required). Dump + summary are written here. |

`--perf_trace_timeout` also governs the bpftrace readiness wait.

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
| `perf_trace_workers.sh` | End-to-end perf wrapper (futex): discover TIDs, set SCHED_OTHER, launch `perf record`. |
| `perf_buf_size.c` | Utility to report effective ring buffer layout for a tracepoint and TID. |
| `build_cuda_api_tracer.sh` | Build the LD_PRELOAD CUDA API tracer library. |
| `futex_cuda_summary_multi_thread.py` | Per-run futex → CUDA API attribution and per-thread summary. |
| `futex_cuda_runs_summary.py` | Collate per-run summaries; optional tracer log inclusion. |
| `bpftrace_trace_workers.sh` | End-to-end bpftrace wrapper (all syscalls): discover TIDs, comm-filter, mount debugfs, run bpftrace, summarize. |
| `bpftrace_syscall_summary.py` | Parse a bpftrace map dump into a per-syscall/per-worker table; PASS/FAIL verdict (FAIL if empty). |

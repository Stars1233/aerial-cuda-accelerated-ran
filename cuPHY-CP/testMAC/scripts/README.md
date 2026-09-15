# cuMAC-CP + cuBB 3-Step Test Orchestrator

Auto-test scripts for cuMAC-CP integrated with cuBB (phase4 tests).  
Covers SRS TV dump, cuMAC TV generation, and the combined 4-app test — all from a single command.

---

## Overview

The orchestrator supports five test modes built on three underlying steps:

```text
run_cubb()        — cuBB test (RU + cuPHYcontroller + testMAC)
                    └─ optionally dumps SRS H5 buffers to /tmp inside gNB container

run_cumcp_sa()    — cuMAC TV generation (run_cumcp_sa.sh inside gNB container)
                    └─ reads SRS H5 → writes cuMAC test vectors to NFS

run_cubb_cumcp()  — Combined 4-app test (cuMAC-CP + RU + cuPHYcontroller + testMAC)
                    └─ reads cuMAC TV, runs integrated test, checks results
```

| Mode | Steps run | Use case |
|---|---|---|
| `cubb` | `run_cubb` (no SRS dump) | cuBB standalone |
| `cubb-srs` | `run_cubb` (with SRS dump) | cuBB + SRS H5 capture, prereq for `ue-group` |
| `cumcp-sa` | `run_cumcp_sa` | cuMAC-CP standalone TV generation |
| `ue-group` | `run_cubb` → `run_cumcp_sa` → `run_cubb_cumcp` | Full UE group flow (`--cumac_task 0x20`) |
| `other` | `run_cubb_cumcp` | Combined test only, `enable_cubb=0` (non-UE-group tasks) |

All steps are orchestrated from the **top-script server** (typically the RU host).  
Scripts and logs are on a shared NFS path visible to all servers and containers.

---

## File Layout

Files used by this orchestrator (the directory also contains other, unrelated test scripts):

```text
cuPHY-CP/testMAC/scripts/
├── env.sh                    # Server addresses, SSH credentials, container name
├── config.sh                 # Test parameters and timeouts
├── run_cumcp_cubb_test.sh    # Main orchestrator
├── run_cumcp_sa.sh           # Standalone cuMAC-CP TV generator (used in cumcp-sa / ue-group)
├── cumac_cp_tv.sh            # cuMAC TV generator invoked by run_cumcp_sa.sh / run_cuMAC_CP.sh
├── check_result_cumcp.py     # Result checker (monitors logs, determines PASS/FAIL)
└── README.md                 # This file
```

The combined test (Step 3) also invokes `testBenches/phase4_test_scripts/run_cuMAC_CP.sh`,
along with the standard phase4 scripts (`copy_test_files.sh`, `build_aerial_sdk.sh`,
`setup1_DU.sh`, `setup2_RU.sh`, `test_config.sh`, `run1_RU.sh`, `run2_cuPHYcontroller.sh`,
`run3_testMAC.sh`, `parse_test_config_params.sh`).

---

## Quick Start

### 1. Configure `env.sh`

Edit `env.sh` and set your server addresses.

**ENG environment** (`gNB_RU_Share=1` — the NFS holds the full SDK, shared by gNB and RU). `CUBB_SDK` must still be set; it defaults to `/opt/nvidia/cuBB` and is different from `CUBB_HOST`:

```bash
export GNB_SERVER="dc7-gnb-001"
export RU_SERVER="dc7-ru-001"
export CUBB_HOST="/home/aerial/nfs/cubb/tot"   # NFS data path: YAML, logs, test vectors
export CUBB_SDK="/opt/nvidia/cuBB"             # in-container SDK path (default /opt/nvidia/cuBB); differs from CUBB_HOST
export HOST_CONFIG="CG1_R750"
export gNB_RU_Share=1                          # gNB and RU share the NFS SDK → skip redundant RU-side setup
```

**QA environment** (SDK installed in containers separately from NFS):

```bash
export GNB_SERVER="dc7-gnb-001"
export RU_SERVER="dc7-ru-001"
export CUBB_HOST="/home/aerial/nfs"            # NFS data path: YAML, logs, test vectors
export CUBB_SDK="/opt/nvidia/cuBB"             # cuBB_SDK env var exported inside containers
export HOST_CONFIG="CG1_R750"
export TV_SRC="${CUBB_HOST}/GPU_test_input"    # TV source for copy_test_files.sh --src
export LOG_BASE="${CUBB_HOST}/Log/Sanity/cuMAC_CP"

# SSH credentials — one shared pair (applies to both servers):
export SSH_USER="aerial"
export SSH_PASS="mypassword"

# Or per-server overrides when gNB and RU use different credentials:
export GNB_SSH_USER="aerial"
export GNB_SSH_PASS="gnb_password"
export RU_SSH_USER="aerial"
export RU_SSH_PASS="ru_password"
```

### 2. (Optional) Adjust `config.sh`

```bash
export SRS_SLOT_LAG=13         # SRS slot lag for combined test (default: 13)
export CUMAC_TASK_MASK=0xF     # Fallback task mask; --cumac_task overrides it per run
export DUMP_SRS_SLOT_NUM=4     # SRS slots to dump in cubb-srs / ue-group (default: 4)
export SHM_LEVEL=5             # nvlog shared-memory level (unset keeps YAML value)
export LOG_SIZE=32000000       # rotating nvlog file size in bytes (unset keeps YAML value)
```

### 3. Run the test

```bash
cd cuPHY-CP/testMAC/scripts

# Full UE group flow (default mode)
./run_cumcp_cubb_test.sh "F08_1C_66c_BFP9_STT455000_EH_1P" 300 1 2 --cumac_task 0x20

# Explicit mode
./run_cumcp_cubb_test.sh "F08_1C_66c_BFP9_STT455000_EH_1P" 300 1 2 --cumac_task 0x20 --test ue-group
```

---

## Usage

```text
./run_cumcp_cubb_test.sh <test_case_string> <duration> [alloc_type] [gpu_share] [options]

Arguments:
  test_case_string   Full cuBB test case string (passed to parse_test_config_params.sh), refer to testBenches/phase4_test_scripts/README.md
                     Format: F08_<N>C_<pattern>[_modifiers...]
                     E.g.  "F08_1C_66c_BFP9_STT455000_EH_1P"
                     E.g.  "F08_6C_79_MODCOMP_STT480000_EH_1P"

  duration           Duration in seconds  (e.g. 300)
  alloc_type         (optional) cuMAC allocation type for TV directory naming. Default: 1
                     Only used by cumcp-sa / ue-group / other; ignored for cubb / cubb-srs.
  gpu_share          (optional) GPU share mode. Default: 0  (value 2 enables SRS dump in ue-group)
                     Only used by cumcp-sa / ue-group / other; ignored for cubb / cubb-srs.

  --test <mode>      Test mode (default: ue-group):
    cubb             Normal cuBB Test — no SRS TV dump
    cubb-srs         cuBB Test + SRS TV Dump
    cumcp-sa         cuMAC-CP Standalone TV Generation only
    ue-group         Full UE group flow:
                       gpu_share=2 → cuBB+SRS Dump → cuMAC-CP SA → Combined Test
                       gpu_share≠2 → cuMAC-CP SA → Combined Test
    other            Combined Test only (enable_cubb=0)

  -b, --build_dir    Build directory under $CUBB_SDK containing the cumac_cp binary.
                     Default: auto-detected from RUN2_CUPHYCONTROLLER_PARAMS.
                     E.g. -b build.perf.x86_64

  --cumac_task <mask>
                     cuMAC-CP task bitmask in decimal or hexadecimal.
                     Overrides CUMAC_TASK_MASK for this run. Default: 0xF.
                     E.g. --cumac_task 0x20

  -h / --help        Show full help and exit
```

### Running individual modes

```bash
# cuBB standalone (no SRS dump) — alloc_type/gpu_share not needed
./run_cumcp_cubb_test.sh "F08_1C_66c_BFP9_STT455000_EH_1P" 300 --test cubb

# cuBB + SRS dump only (prerequisite for ue-group) — alloc_type/gpu_share not needed
./run_cumcp_cubb_test.sh "F08_1C_66c_BFP9_STT455000_EH_1P" 300 --test cubb-srs

# cuMAC-CP standalone TV generation only (requires SRS H5 from cubb-srs)
./run_cumcp_cubb_test.sh "F08_1C_66c_BFP9_STT455000_EH_1P" 300 1 2 --cumac_task 0x20 --test cumcp-sa

# Combined test only, Other Tasks (no SRS, enable_cubb=0)
./run_cumcp_cubb_test.sh "F08_1C_66c_BFP9_STT455000_EH_1P" 300 1 0 --cumac_task 0xF --test other
```

`run_sanity.sh` passes `--cumac_task` on each cuMAC-CP command, so different
task masks can be tested in one script without exporting and mutating
`CUMAC_TASK_MASK` between runs.

---

## Environment Variables

### Required (set in `env.sh` or export before running)

| Variable | Description | Example |
|----------|-------------|---------|
| `GNB_SERVER` | gNB / DU server hostname or IP | `dc7-gnb-001` |
| `RU_SERVER` | RU server hostname or IP | `dc7-ru-001` |
| `CUBB_HOST` | NFS path — must be identical on all hosts and inside containers | `/home/aerial/nfs` |
| `HOST_CONFIG` | Platform string for `parse_test_config_params.sh` | `CG1_R750` |

Valid `HOST_CONFIG` values: `CG1_R750` · `CG1_CG1` · `GL4_R750` · `SPRK_R750`

### Optional

| Variable | Default | Description |
|----------|---------|-------------|
| `CUBB_SDK` | `/opt/nvidia/cuBB` | `$cuBB_SDK` inside containers — the SDK install path. Always set in both ENG and QA; normally different from `CUBB_HOST` (the NFS data path). |
| `gNB_RU_Share` | `0` | Set `1` when gNB and RU share the same NFS-mounted SDK (ENG). The orchestrator then skips the redundant RU-side `copy_test_files`/`setup`/`test_config` and the NFS artifact exchange. `0` = separate hosts (RU-side steps run). |
| `TV_SRC` | *(auto)* | Source directory passed as `--src` to `copy_test_files.sh`. Leave unset for ENG (auto-detected via `get_uuid.sh`). QA: `${CUBB_HOST}/GPU_test_input` |
| `LOG_BASE` | `$CUBB_HOST/logs` | Parent directory for run logs. QA: `${CUBB_HOST}/Log/Sanity/cuMAC_CP` |
| `CUMAC_BUILD_DIR` | *(auto)* | Build subdirectory under `$CUBB_SDK` containing the `cumac_cp` binary. Auto-detected from `RUN2_CUPHYCONTROLLER_PARAMS`. Override for split-build environments: `export CUMAC_BUILD_DIR=build` |
| `SSH_USER` | `$USER` | SSH username shared default |
| `SSH_PASS` | *(unset)* | SSH password shared default — leave unset to use key auth |
| `GNB_SSH_USER` | `$SSH_USER` | gNB SSH username override |
| `GNB_SSH_PASS` | `$SSH_PASS` | gNB SSH password override |
| `RU_SSH_USER` | `$SSH_USER` | RU SSH username override |
| `RU_SSH_PASS` | `$SSH_PASS` | RU SSH password override |
| `CONTAINER` | `c_aerial_${SSH_USER}` | Docker container name on both servers |
| `DUMP_SRS_SLOT_NUM` | `4` | SRS slots to dump in `cubb-srs` / `ue-group` |
| `CUMAC_TASK_MASK` | `0xF` | Fallback cuMAC-CP task bitmask. `--cumac_task` overrides it for one invocation. |
| `SRS_SLOT_LAG` | `13` | SRS slot lag written to cuMAC-CP YAMLs in `run_cubb_cumcp` |
| `RU_READY_TIMEOUT` | `180` | Seconds to wait for RU ready signal |
| `PHY_READY_TIMEOUT` | `120` | Seconds to wait for cuPHYcontroller ready signal |
| `MAC_READY_TIMEOUT` | `300` | Seconds to wait for testMAC ready signal |
| `CUMAC_READY_TIMEOUT` | `120` | Seconds to wait for `cumac_receiver: initialized` after cuMAC-CP TV loading |
| `TIMEOUT_BASE` | `300` | Extra buffer seconds added to duration for cuMAC-CP timeout |
| `SHM_LEVEL` | *(YAML value)* | Global `shm_log_level` applied to nvlog configuration |
| `LOG_SIZE` | *(YAML value)* | Positive byte count applied to nvlog `max_file_size_bytes` |
| `TRY` | `0` | Set `1` to keep result checking active through transient throughput failures |
| `NVIPC_PCAP_ENABLE` | `0` | Set `1` to enable nvIPC packet capture and collect the generated PCAP |
| `RESTORE_CONFIG_BY_GIT` | `0` | Set `1` to restore baseline YAMLs with `git checkout` in the `CUBB_HOST` source tree |

---

## What Each Step Does

### `run_cubb` — cuBB Test (modes: `cubb`, `cubb-srs`, `ue-group`)

Log location: directly in the run directory for `--test cubb`; otherwise
`cubb/` (no SRS dump) or `cubb_srs_dump/` (with SRS dump).

1. Copies test vectors to gNB and RU containers (`copy_test_files.sh`)
2. **Builds only if needed** — checks each host for its own (arch-specific) build folder and runs `build_aerial_sdk.sh` on gNB and/or RU only where absent
3. Restores baseline config files from `config_ori/` (cuphycontroller, ru-emulator, testMAC, cuMAC-CP)
4. Runs `setup1_DU.sh` → exports artifacts to NFS → `setup2_RU.sh` → syncs back → `test_config.sh`
5. Launches **RU → cuPHYcontroller → testMAC** (each waits for ready signal before the next)
6. Runs `check_result_cumcp.py --no-fapi` for `duration` seconds
7. On success: verifies SRS H5 files exist (`/tmp/cubb_srs_buffers_*.h5`) when `DUMP_SRS_SLOT_NUM > 0`

### `run_cumcp_sa` — cuMAC-CP Standalone TV Generation (modes: `cumcp-sa`, `ue-group`)

Log subdir: `cumcp_sa/`

1. Prerequisite (only when `gpu_share=2`): verifies SRS H5 files exist in the gNB container; other `gpu_share` values skip this check
2. Runs `run_cumcp_sa.sh <duration> F08 <test_case> --alloc_type <a> --gpu_share <g> --task <mask>` inside the gNB container
3. On success: verifies the cuMAC TV symlink exists and is non-empty
   - Symlink: `$CUBB_SDK/testVectors/cumac` → `cumac.<N>c.type<a>.gpu_share<g>`
   - Check is via symlink only — `run_cumcp_sa.sh` may reuse an existing valid TV directory

> **Note:** `run_cumcp_sa.sh` sets `enable_cubb=0` and `srs_slot_lag=0`. `run_cubb_cumcp` restores these before launching.

### `run_cubb_cumcp` — Combined 4-app Test (modes: `ue-group`, `other`)

Log subdir: `cubb_cumcp/`

1. No cuMAC TV prerequisite check — `run_cuMAC_CP.sh` regenerates the cuMAC TV (via `cumac_cp_tv.sh`) if it is missing
2. **Builds only if needed** (same check as `run_cubb`)
3. Restores baseline config files, then re-runs `setup1_DU.sh` → `setup2_RU.sh` → `test_config.sh`
4. Applies cuMAC-CP yaml configuration:

   | `CUMAC_TASK_MASK` | Config applied |
   |---|---|
   | `0x20` (UE group) | `enable_cubb=1`, `enable_tv_test=1`, `srs_slot_lag=$SRS_SLOT_LAG` in both YAMLs |
   | Other | `enable_cubb=0` |

   `enable_gpu_share` is set automatically by `run_cuMAC_CP.sh` based on the `-g gpu_share` argument — no manual override needed.

5. Launches **RU → cuPHYcontroller → cuMAC-CP → testMAC**. testMAC is not
   launched until cuMAC-CP logs `cumac_receiver: initialized`, which occurs
   after its TV files are loaded.
6. Runs `check_result_cumcp.py` with `--cumcp-log`; the checker waits for both
   CUMAC and CUMCP throughput streams before starting the requested duration.

---

## Log Structure

Each run creates a timestamped directory under `LOG_BASE`, renamed `_PASS`,
`_FAIL`, `_ERR`, or `_CRASH` on completion. `_ERR` means the run finished but
error-level lines were found; `_CRASH` means an application exited before
result checking completed.

```text
$LOG_BASE/
├── latest -> <most recent run dir>
└── <YYYYMMDD_HHMMSS>_<test_case>_<mode>_TASK<mask>_TYPE<a>_SHARE<g>_<dur>s_PASS/
    ├── main.log              # Orchestrator log: [INFO/STEP/CFG/CMD/CHK/ERR] entries
    ├── screenlog_*.log       # directly here for --test cubb (no cubb/ subdir)
    ├── cubb_srs_dump/        # run_cubb output (--test cubb-srs / ue-group with gpu_share=2)
    │   ├── screenlog_ru.log
    │   ├── screenlog_phy.log
    │   ├── screenlog_mac.log
    │   └── configs/
    ├── cumcp_sa/             # run_cumcp_sa output
    │   └── configs/
    └── cubb_cumcp/           # run_cubb_cumcp output
        ├── screenlog_cum.log
        ├── screenlog_ru.log
        ├── screenlog_phy.log
        ├── screenlog_mac.log
        ├── check_result.log
        ├── phy.log* / ru.log* / testmac.log* / cumac_cp.log*
        ├── core.du.log / core.ru.log
        ├── smi.log
        ├── ul_packet_times.txt
        ├── ipc_dump.phy.nvipc.log
        ├── nvipc*.pcap               # when NVIPC_PCAP_ENABLE=1
        └── configs/                  # dump_configs snapshot
            ├── cuphycontroller_*.yaml # the specific yaml used (from phy screenlog)
            ├── test_mac_config.yaml
            ├── test_cumac_config.yaml
            ├── cumac_cp.yaml
            ├── nvlog_config.yaml
            ├── ru_emulator_config.yaml
            ├── test_params.sh
            └── sysinfo.txt            # uname, nvidia-smi, driver/CUDA versions
```

Only subdirs relevant to the modes actually run are created (e.g. `ue-group` with `gpu_share=0` skips `cubb_srs_dump/`).

To monitor a running test:

```bash
tail -f $LOG_BASE/latest/main.log
```

---

## Troubleshooting

**`env.sh failed — fix the variables above and retry`**  
One or more required variables (`GNB_SERVER`, `RU_SERVER`, `CUBB_HOST`, `HOST_CONFIG`) is not set. Also set `CUBB_SDK` (defaults to `/opt/nvidia/cuBB`) and, for QA, `LOG_BASE`.

**`parse_test_config_params.sh failed`**  
The test case string is invalid for the given `HOST_CONFIG`, or the gNB container cannot reach the phase4 scripts. Verify the string against `testBenches/phase4_test_scripts/README.md`.

**`SRS H5 files missing in gNB container`**  
`run_cubb` did not complete, or `DUMP_SRS_SLOT_NUM=0`. Re-run with `--test cubb-srs` and `DUMP_SRS_SLOT_NUM >= 1`.

**`Symlink $CUBB_SDK/testVectors/cumac does not exist or is empty`**  
`run_cumcp_sa` has not been run, or `run_cumcp_sa.sh` failed. Check `$LOG_BASE/latest/cumcp_sa/` logs.

**`cuPHYcontroller did NOT signal ready within 120s`**  
Increase `PHY_READY_TIMEOUT` or check `cubb_cumcp/screenlog_phy.log`.

**`testMAC did NOT signal ready within 60s`**  
Increase `MAC_READY_TIMEOUT` or check `cubb_cumcp/screenlog_mac.log`.

**`cuMAC-CP did NOT signal ready within 120s`**  
Increase `CUMAC_READY_TIMEOUT` or check `cubb_cumcp/screenlog_cum.log`.

**Combined test fails even though logs look OK**  
Check `main.log` for the `[CFG ]` lines — confirm `enable_cubb=1` and `srs_slot_lag=13` were applied before launch.

**`restore cuMAC-CP config failed`**  
Only fatal if `cuMAC-CP/config_ori/` exists in the container. On QA containers without `config_ori`, the restore is silently skipped — this is expected.

---

## Test Case String Format

```text
F08_<N>C_<pattern>[_modifiers...]

F08           Required prefix for performance test cases
<N>C          Number of cells (e.g. 1C, 2C, 6C, 8C)
<pattern>     Pattern number (e.g. 60, 66c, 79)
modifiers     Optional, any order:
              BFP9 / BFP14   BFP compression bits
              STT<value>     Schedule total time (e.g. STT455000)
              1P / 2P        Number of ports
              EH             Enable early HARQ
              GC             Enable green context
              MODCOMP        MODCOMP mode
              NS<value>      Number of slots (e.g. NS30000)
```

**Examples:**

```bash
"F08_1C_66c_BFP9_STT455000_EH_1P"      # 1-cell
"F08_2C_60c_BFP9_STT480000_EH_1P"      # 2-cell MU-MIMO, BFP9
"F08_6C_79_MODCOMP_STT480000_EH_1P"    # 6-cell MODCOMP
"F08_8C_66c_BFP9_STT455000_EH_1P"      # 8-cell
"F08_20C_59c_BFP14_EH_GC_NS30000_2P"  # 20-cell, dual port, green context
```

---

## References

- Phase4 test scripts: `$CUBB_HOST/testBenches/phase4_test_scripts/README.md`
- cuMAC-CP standalone runner: `run_cumcp_sa.sh --help`
- Phase4 parser: `parse_test_config_params.sh --help`

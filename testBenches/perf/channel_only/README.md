# Phase-3 channel-only isolation

Isolated per-channel GPU latency measurements for Phase-3 (`measure.py` / `cubb_gpu_test_bench`).

**Jira:** GT-12714

This directory adds **YAML configs and thin wrapper scripts** without modifying existing team run scripts outside `channel_only/`.

---

## Overview

Phase-3 normally runs multi-channel slot sweeps. For capacity planning and channel budgeting you often need **one channel at a time** with comparable slot/TDD context.

| Channel | How to isolate | Notes |
|---------|----------------|-------|
| PUSCH | `--no_pdsch` (YAML: `no_pdsch: true`) | Existing Phase-3 flag |
| PDSCH | `--no_pusch` (YAML: `no_pusch: true`) | Existing Phase-3 flag |
| DL BFW | `--dl_bf_only` | New: PDSCH context + DL BFW only; no PUSCH/SRS/UL BFW |
| SRS | `--srs_only` | New: isolated SRS; no PDSCH/PUSCH |

**Rules:**

- `--dl_bf_only` and `--srs_only` are mutually exclusive and cannot combine with `--rec_bf`.
- `--dl_bf_only` and `--srs_only` require **F09/F14 avg-cell** use cases (`uc_avg_F14_TDD.json`, etc.).
- PUSCH/PDSCH use existing `--no_pdsch` / `--no_pusch` via YAML (no new CLI flags).

Runs follow the Phase-3 YAML pattern:

```bash
python3 measure.py --yaml channel_only/yaml/<config>.yaml --cuphy <path> --vectors <path>
```

CLI flags override YAML values (e.g. `--freq`, `--target`).

### Phase-2 vs Phase-3 isolation

| | Phase-2 (`measure_srs.py`, `measure_avg_dlbfw.py`) | Phase-3 + channel-only YAML |
|--|-----------------------------------------------------|------------------------------|
| Bench | Single-kernel / legacy | Full slot sweep, CUDA graphs, cell capacity |
| Context | Minimal | TDD pattern, MPS, multi-cell |
| Use when | Kernel unit perf | System-level channel budgeting |

---

## Directory layout

```text
channel_only/
  README.md              ← this file
  common.env             ← paths and sweep defaults (override via env vars)
  yaml/                  ← Phase-3 YAML configs (one per channel/scenario)
    pusch_only_F14_64TR_run1.yaml
    pdsch_only_F14_64TR_run1.yaml
    pdsch_only_F14_64TR_run2.yaml
    dl_bf_only_F14_64TR_run1.yaml
    dl_bf_only_F14_64TR_run2.yaml
    srs_only_F14_64TR_run1.yaml
    srs_only_F14_64TR_srs_rkhs.yaml
  pusch_only.sh          ← wrapper → measure.py --yaml ...
  pdsch_only.sh
  dl_bf_only.sh
  srs_only.sh
  run_f14_stage1.sh      ← F14 64TR TVnr Stage-1 matrix (7 runs)
  verify_channel_only.sh ← Run + validate all channel-only tests
```
H5 TVs are listed inline in each YAML under `config.vector_files` (+ `usecase: F14`);
`measure.py` auto-generates the in-memory testcase config (same as `cubb_gpu_test_config.yaml`).

---

## Prerequisites

Inside the Aerial container (or host with cuBB built):

```bash
# Build perf bench once
cd /opt/nvidia/cuBB
cmake --preset perf.aarch64 && cmake --build build.perf.aarch64 -j

# Test vectors for F14 64TR campaign (example TV IDs)
ls /opt/nvidia/cuBB/testVectors/TVnr_ULMIX_21564_PUSCH_gNB_CUPHY_s4p7.h5
ls /opt/nvidia/cuBB/testVectors/TVnr_DLMIX_20566_PDSCH_gNB_CUPHY_s6p15.h5
ls /opt/nvidia/cuBB/testVectors/TVnr_DLMIX_20956_PDSCH_gNB_CUPHY_s6p31.h5
ls /opt/nvidia/cuBB/testVectors/TVnr_9366_BFW_gNB_CUPHY_s0.h5
ls /opt/nvidia/cuBB/testVectors/TVnr_9464_BFW_gNB_CUPHY_s0.h5
ls /opt/nvidia/cuBB/testVectors/TVnr_ULMIX_21673_SRS_gNB_CUPHY_s3p47.h5
ls /opt/nvidia/cuBB/testVectors/TVnr_ULMIX_21673_SRS_gNB_CUPHY_s3p47_rkhs.h5
```

Use case file must contain **`TDD`** in the filename (e.g. `uc_avg_F14_TDD.json`).

---

## Hardware settings (FREQ / POWER / TARGET_SM)

These vary by GPU and host. `common.env` **auto-detects** from `nvidia-smi` when `AUTO_DETECT_GPU=1` (default):

| Variable | Auto-detect source | Override example |
|----------|-------------------|------------------|
| `FREQ` | current graphics clock | `export FREQ=1980` |
| `POWER` | current power limit | `export POWER=900` |
| `TARGET_SM` | `nvidia-smi -q` Multiprocessors (or query field on newer drivers) | `export TARGET_SM=138` |

Using the **current** clock and power limit avoids `measure.py` changing GPU settings. To reproduce the F14 GH200 reference campaign exactly:

```bash
AUTO_DETECT_GPU=0 FREQ=1605 POWER=165 TARGET_SM=82 ./run_f14_stage1.sh
```

On a **138-SM machine with no caps**, defaults usually work with no exports:

```bash
./verify_channel_only.sh --quick   # uses detected 138 SM, current freq/power
```

---

## Quick start

```bash
cd /opt/nvidia/cuBB/testBenches/perf/channel_only

# Optional: override hardware-specific settings (auto-detected from nvidia-smi by default)
export AUTO_DETECT_GPU=1          # set 0 to use F14 reference defaults (1605 MHz, 165 W, 82 SM)
export TARGET_SM=138              # optional override (else GPU multiprocessor_count)
export FREQ=1980                  # optional override (else current graphics clock)
export POWER=900                  # optional override (else current power limit)
export VECTORS=/opt/nvidia/cuBB/testVectors
export CUPHY=/opt/nvidia/cuBB/build.perf.aarch64/testBenches

./pusch_only.sh
./pdsch_only.sh
./dl_bf_only.sh
./srs_only.sh
```

Each wrapper saves a JSON copy in `testBenches/perf/` (e.g. `pusch_only_F14_64TR_6cell.json`).

### Verify all tests together

```bash
cd /opt/nvidia/cuBB/testBenches/perf/channel_only
export FREQ=1605 POWER=165 TARGET_SM=82
export VECTORS=/opt/nvidia/cuBB/testVectors
export CUPHY=/opt/nvidia/cuBB/build.perf.aarch64/testBenches

./verify_channel_only.sh --quick          # 4 smoke runs + JSON checks
./verify_channel_only.sh                  # full Stage-1 (7 runs) + checks
./verify_channel_only.sh --check-only     # validate existing JSON only
VERIFY_TARGET_SM=1 ./verify_channel_only.sh --quick  # also 82 vs 16 SM SRS check
```

Or run directly with YAML:

```bash
cd /opt/nvidia/cuBB/testBenches/perf
python3 measure.py \
  --yaml channel_only/yaml/pusch_only_F14_64TR_run1.yaml \
  --cuphy /opt/nvidia/cuBB/build.perf.aarch64/testBenches \
  --vectors /opt/nvidia/cuBB/testVectors
```

---

## F14 64TR Stage-1 matrix

Full campaign repro (6 cells, 82 SMs, TVnr vectors):

```bash
cd /opt/nvidia/cuBB/testBenches/perf/channel_only
./run_f14_stage1.sh
```

| Run | YAML | TV (ID) |
|-----|------|---------|
| PUSCH | `yaml/pusch_only_F14_64TR_run1.yaml` | 21564 |
| PDSCH run1 | `yaml/pdsch_only_F14_64TR_run1.yaml` | 20566 (16L) |
| DL BFW 16L | `yaml/dl_bf_only_F14_64TR_run1.yaml` | 9366 |
| SRS MMSE | `yaml/srs_only_F14_64TR_run1.yaml` | 21673 |
| PDSCH run2 | `yaml/pdsch_only_F14_64TR_run2.yaml` | 20956 (32L) |
| DL BFW 32L | `yaml/dl_bf_only_F14_64TR_run2.yaml` | 9464 |
| SRS RKHS | `yaml/srs_only_F14_64TR_srs_rkhs.yaml` | 21673_rkhs |

**Common sweep settings** (in each YAML; override via CLI or env in wrappers):

| Parameter | Default |
|-----------|---------|
| Cells | 6 (`start: 6`, `cap: 6`) |
| TDD pattern | `dddsuudddd_mMIMO` |
| Slots | 30 |
| Iterations | 1 |
| Delay | 10000 µs |
| SMs | 82 (`TARGET_SM` in wrappers, or `--target` CLI override) |
| CUDA graphs | on (`graph: true`) |

For SRS SM scaling (16 / 24 SMs), pass `--target 16` as a CLI override:

```bash
./srs_only.sh yaml/srs_only_F14_64TR_run1.yaml srs_mmse_16SM_F14_64TR_6cell --target 16
```

---

## Test vectors and modulation

All PUSCH/PDSCH TVs in the F14 configs use **MCS 27 / 256QAM**.

| Channel | Test vector file | Key config |
|---------|------------------|------------|
| PUSCH | `TVnr_ULMIX_21564_PUSCH_gNB_CUPHY_s4p7.h5` | 64 gNB RX, 8 UEs × 1L, 247 RB |
| PDSCH 16L | `TVnr_DLMIX_20566_PDSCH_gNB_CUPHY_s6p15.h5` | 16 TX, 16 layers, 273 RB |
| PDSCH 32L | `TVnr_DLMIX_20956_PDSCH_gNB_CUPHY_s6p31.h5` | 32 TX, 32 layers, 273 RB |
| DL BFW 16L | `TVnr_9366_BFW_gNB_CUPHY_s0.h5` | 64 ports, 16 layers |
| DL BFW 32L | `TVnr_9464_BFW_gNB_CUPHY_s0.h5` | 64 ports, 32 layers |
| SRS MMSE | `TVnr_ULMIX_21673_SRS_gNB_CUPHY_s3p47.h5` | Comb4, 48 UEs × 2 ports, 4 sym |
| SRS RKHS | `TVnr_ULMIX_21673_SRS_gNB_CUPHY_s3p47_rkhs.h5` | Same TV, `chEstAlgoIdx=1` |

SRS config: Comb4, CS-6, 272 RB, 48 SRS PDUs (2×48 = 96 ports), 4 symbols.

---

## Manual `measure.py` examples

```bash
cd /opt/nvidia/cuBB/testBenches/perf

# PUSCH only @ 6 cells (existing --no_pdsch via YAML)
python3 measure.py \
  --yaml channel_only/yaml/pusch_only_F14_64TR_run1.yaml \
  --cuphy /opt/nvidia/cuBB/build.perf.aarch64/testBenches \
  --vectors /opt/nvidia/cuBB/testVectors \
  --target 82

# SRS only (RKHS TV)
python3 measure.py \
  --yaml channel_only/yaml/srs_only_F14_64TR_srs_rkhs.yaml \
  --cuphy /opt/nvidia/cuBB/build.perf.aarch64/testBenches \
  --vectors /opt/nvidia/cuBB/testVectors \
  --target 82
```

---

## Reading results

Sweep output defaults to `082_sweep_graphs_avg_F14.json`. Wrappers copy it to a tagged name.

Per-channel GPU times are under the `06+00` cell key:

```bash
python3 - <<'PY'
import json, statistics
d = json.load(open("1612_pusch_only_F14_64TR_6cell.json"))["06+00"]
for ch in ["PUSCH1", "PUSCH2", "PDSCH", "DLBFW", "SRS1"]:
    if ch in d and d[ch]:
        print(ch, round(statistics.mean(d[ch])), "us")
PY
```

Peak GPU memory: `d["memoryUseMB"]["total"]`.

---

## GPU clock note

Set `freq` in YAML or pass `--freq` to match the container's idle graphics clock:

```bash
nvidia-smi -i 0 --query-gpu=clocks.current.graphics --format=csv,noheader
# Set FREQ to match, e.g. FREQ=1605 ./run_f14_stage1.sh
```

On bare-metal hosts, lock clocks before long campaigns:

```bash
sudo nvidia-smi -i 0 -lgc 1605,1605
# ... run tests ...
sudo nvidia-smi -i 0 -rgc
```

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `Channel-only flags cannot be combined with --rec_bf` | Remove `--rec_bf` / set `rec_bf: false` in YAML |
| `--dl_bf_only requires F09/F14 avg-cell use cases` | Use `uc_avg_F14_TDD.json` (must contain `TDD`) |
| SRS RKHS segfault | Use `srs_only_F14_64TR_srs_rkhs.yaml` (`vector_files.SRS` = `*_rkhs.h5`), not a YAML override on the MMSE TV |
| `MISSING TV` | Copy TVs to `$VECTORS` |
| Empty container / no files | Mount aerial_sdk at `/opt/nvidia/cuBB` |

---

## References

- [Phase-3 Test Bench](https://nvidia.atlassian.net/wiki/spaces/5GV/pages/2089386561/Phase-3+Test+Bench)
- [How to Run Phase-3 Test Bench](https://nvidia.atlassian.net/wiki/spaces/5GV/pages/3317793174/How+to+Run+Phase-3+Test+Bench)
- Jira: GT-12714

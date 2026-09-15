# Performance Metrics CSV Reference

This document describes the columns produced by `cicd_performance_metrics.py` in the
per-slot performance CSV (`-p` output). Each row corresponds to one slot in the TDD
pattern. All timing values are **99th-percentile** (1% CCDF) statistics computed
across all observed frames, unless otherwise noted. The quantile is configurable
via `--quantile` (default 0.99).

---

## Timing Reference Points

Two reference points are used throughout:

| Reference | Definition |
|-----------|------------|
| **T0**    | OTA (over-the-air) start time of symbol 0 in the slot. |
| **tick**  | The scheduling tick: **T0 − 1500 us** (3 slots ahead of OTA at 30 kHz SCS). |

Metrics prefixed `tick_to_*` are measured **from tick**. Metrics prefixed `t0_to_*`
are measured **from T0**. This distinction matters because DL processing must
complete *before* OTA (relative to tick), while UL processing occurs *after* OTA
(relative to T0).

---

## Slot Structure and TDD Pattern

The CSV contains one row per slot in the TDD pattern. At **30 kHz SCS** each slot
is **500 us**. The default pattern spans **80 slots** (40 ms), with slot indices
0–79.

When **Early HARQ (EH)** is enabled, slots at positions **{4, 14, 24, 34, 44, 54,
64, 74}** (every 10th slot, offset 4) are designated as EH slots and have a tighter
UL completion deadline of **2000 us** from T0. All other UL slots use the standard
**4500 us** deadline. When EH is not enabled, all UL slots use 4500 us.

---

## Units

| Category | Unit |
|----------|------|
| All timing / headroom columns | **microseconds (us)** |
| Ontime percentage columns     | **percent (%)** |

---

## Column Reference

### Slot Identifier

| Column | Description |
|--------|-------------|
| `slot`  | Slot index within the TDD pattern (0–79). |

---

### Tick and L2 Start Timing

| Column | Reference | Description |
|--------|-----------|-------------|
| `tick_to_l2_start` | tick | Time from the tick event to the start of L2 processing on the CPU. Captures the scheduling jitter / wake-up latency between the tick interrupt and when the L2 stack begins work. |

---

### TestMAC / FAPI Timing

These metrics capture when the test MAC (L2 emulator) sends FAPI messages to the
PHY. All are measured **from tick**.

| Column | Reference | Percentile | Description |
|--------|-----------|------------|-------------|
| `testmac_start_min` | tick | 1st  | Earliest observed start of testmac FAPI message generation. |
| `testmac_start_max` | tick | 99th | Latest observed start of testmac FAPI message generation. |
| `testmac_end_min`   | tick | 1st  | Earliest observed completion of testmac FAPI message generation. |
| `testmac_end_max`   | tick | 99th | Latest observed completion of testmac FAPI message generation. |
| `testmac_duration`  | N/A  | 99th | Duration of testmac processing (end − start). Not referenced to tick or T0. |

The min/max pairs show the spread of testmac timing across frames for each slot.

---

### L2 Adapter Timing

The L2 Adapter (L2A) receives FAPI messages from the MAC and translates them into
cuPHY processing tasks for the slot. These metrics are derived from L2A processing
log messages (`L2A.PROCESSING_TIMES`), not from the TI instrumentation. All are
measured **from tick**.

| Column | Reference | Percentile | Description |
|--------|-----------|------------|-------------|
| `l2a_start_max` | tick | 99th | Time from tick to the start of L2A FAPI message translation for the slot. |
| `l2a_end_max`   | tick | 99th | Time from tick to the completion of L2A task dispatch for the slot. |
| `l2a_duration`  | N/A  | 99th | Duration of L2A processing (end − start). Not referenced to tick or T0. |

---

### DL Processing Timelines

These metrics track the DL (downlink) processing pipeline completion times relative
to **tick**. DL processing must complete before OTA so the fronthaul packets arrive
at the RU on time.

| Column | Reference | TI Task | TI Subtask | Description |
|--------|-----------|---------|------------|-------------|
| `tick_to_fhcb_completion` | tick | `DL Task FH Callback` | `Full Task` | Time from tick to Fronthaul Callback completion. This is when the DL fronthaul data transfer is acknowledged as complete. |
| `tick_to_dlu_completion`  | tick | `Debug Task` | `Trigger synchronize` | Time from tick to DL User-plane (DLU) processing completion. Covers the full DL L1 processing chain from L2A start through GPU completion. Requires the debug thread to be enabled in the cuPHY controller config (`debug_worker: 17`). |
| `tick_to_dlc_bfw_completion` | tick | `DL Task C-Plane N` | `bfw_prepare cell_X sym_0` | Time from tick to DL Control-plane BFW preparation completion. Covers the beamforming-weight (BFW) portion of DL C-plane message generation. When legacy subtask format is used, falls back to `CPlane Prepare`. |
| `tick_to_dlc_nonbfw_completion` | tick | `DL Task C-Plane N` | `nonbfw_prepare cell_X sym_0` | Time from tick to DL Control-plane non-BFW preparation completion. Covers the non-BFW portion of DL C-plane message generation (e.g., section headers, frequency-domain data). |
| `tick_to_dlbfw_completion` | tick | `DL Task BFW` | `Wait Run Completion` | Time from tick to DL beamforming weight computation completion. The `Wait Run Completion` subtask is an asynchronous wait for the GPU BFW kernel; the `end_deadline` of this subtask marks when the computation finishes. Only populated for MMIMO (64TR) test cases. |

---

### UL Control-Plane Timelines

These metrics track UL (uplink) control-plane preparation, measured from **tick**.
UL C-plane messages must be sent to the RU ahead of the UL reception window.

| Column | Reference | TI Task | TI Subtask | Description |
|--------|-----------|---------|------------|-------------|
| `tick_to_ulc_bfw_completion` | tick | `UL Task CPlane N` | `bfw_prepare cell_X sym_0` | Time from tick to UL Control-plane BFW preparation completion. Covers beamforming-weight portion of UL C-plane generation. |
| `tick_to_ulc_nonbfw_completion` | tick | `UL Task CPlane N` | `nonbfw_prepare cell_X sym_0` | Time from tick to UL Control-plane non-BFW preparation completion. Covers non-BFW portion of UL C-plane generation. |
| `tick_to_ulbfw_completion` | tick | `UL Task UL AGGR 3 [ULBFW]` | `Callback ULBFW` | Time from tick to UL beamforming weight computation completion. Currently not populated (reserved for future use). |

---

### UL User-Plane Timelines

These metrics track UL user-plane processing completion, measured from **T0**. UL
data arrives after OTA, so these timelines are naturally referenced to T0.

| Column | Reference | TI Task | TI Subtask | Description |
|--------|-----------|---------|------------|-------------|
| `t0_to_order_kernel_completion` | T0 | `UL Task UL AGGR 3` | `Wait Order` | Time from T0 to UL ordering kernel completion. This is an asynchronous wait for the ordering kernel to finish. |
| `t0_to_pucch_completion` | T0 | `UL Task UL AGGR 3` | `Callback PUCCH` | Time from T0 to PUCCH processing completion. |
| `t0_to_pusch_completion` | T0 | `UL Task UL AGGR 3` | `Callback PUSCH` | Time from T0 to PUSCH processing completion. |
| `t0_to_pusch_eh_completion` | T0 | `UL Task AGGR3 Early UCI IND` | `Signal Completion` | Time from T0 to PUSCH Early HARQ (EH) indication signaling. Unlike other `t0_to_*` metrics which use `end_deadline`, this metric uses `start_deadline` — the start of the `Signal Completion` section marks the moment the early UCI indication is signaled. Only populated when EH is enabled; `None` for non-EH runs. |
| `t0_to_prach_completion` | T0 | `UL Task UL AGGR 3` | `Callback PRACH` | Time from T0 to PRACH processing completion. |
| `t0_to_srs_completion` | T0 | `UL Task UL AGGR 3 [SRS]` | `Callback SRS` | Time from T0 to SRS processing completion. Only populated for MMIMO (64TR) test cases. |

---

### Headroom Metrics

Headroom = **deadline − completion time**. Positive values indicate the processing
finished before its deadline; negative values indicate a deadline violation.

#### DL Headroom (relative to tick)

Deadlines for DL metrics are derived from the fronthaul reception window offsets and
vary by test case type (4TR vs 64TR).

| Column | Completion Metric | Deadline Source | Description |
|--------|-------------------|-----------------|-------------|
| `dlu_headroom` | `tick_to_dlu_completion` | DLU reception window start | Headroom for DL user-plane delivery to the RU. |
| `dlc_bfw_headroom` | `tick_to_dlc_bfw_completion` | DLC BFW reception window start | Headroom for DL control-plane BFW delivery. |
| `dlc_nonbfw_headroom` | `tick_to_dlc_nonbfw_completion` | DLC non-BFW reception window start | Headroom for DL control-plane non-BFW delivery. |
| `dlbfw_headroom` | `tick_to_dlbfw_completion` | DLC BFW window start + 1 slot | Headroom for DL BFW computation. MMIMO only. |

#### UL Control-Plane Headroom (relative to tick)

| Column | Completion Metric | Deadline Source | Description |
|--------|-------------------|-----------------|-------------|
| `ulc_bfw_headroom` | `tick_to_ulc_bfw_completion` | ULC BFW reception window start | Headroom for UL control-plane BFW delivery. |
| `ulc_nonbfw_headroom` | `tick_to_ulc_nonbfw_completion` | ULC non-BFW reception window start | Headroom for UL control-plane non-BFW delivery. |
| `ulbfw_headroom` | `tick_to_ulbfw_completion` | ULC BFW window start + 1 slot | Headroom for UL BFW computation. Currently not populated. |

#### UL User-Plane Headroom (relative to T0)

| Column | Completion Metric | Deadline | Description |
|--------|-------------------|----------|-------------|
| `pucch_headroom` | `t0_to_pucch_completion` | 2000 us (EH slots) / 4500 us (non-EH slots) | Headroom for PUCCH processing. |
| `pusch_eh_headroom` | `t0_to_pusch_eh_completion` | 2000 us (EH slots) / 4500 us (non-EH slots) | Headroom for PUSCH Early HARQ indication. Only populated when EH is enabled. |
| `pusch_headroom` | `t0_to_pusch_completion` | 4500 us (all slots) | Headroom for PUSCH processing. |
| `prach_headroom` | `t0_to_prach_completion` | 4500 us (all slots) | Headroom for PRACH processing. |
| `srs_headroom` | `t0_to_srs_completion` | 5500 us (all slots) | Headroom for SRS processing. MMIMO only. |

---

### Ontime Percentages

Ontime percentage represents the fraction of fronthaul packets delivered on time for
a given traffic type. The system scores every packet as early, on-time, or late. The
value recorded per slot is the **worst (minimum) percentage across all cells**.

Initial warmup periods are excluded from the calculation (warmup slots on the RU
side, warmup frames on the DU side).

#### DU-Side Ontime (measured at the DU)

| Column | Traffic Type | Description |
|--------|-------------|-------------|
| `ulu_ontime_percentage` | UL user-plane (non-SRS) | Percentage of UL user-plane packets received on time at the DU. Covers PUCCH, PUSCH, and PRACH combined. |
| `srs_ontime_percentage` | SRS | Percentage of SRS packets received on time at the DU. MMIMO only. |

#### RU-Side Ontime (measured at the RU)

| Column | Traffic Type | Description |
|--------|-------------|-------------|
| `dlu_ontime_percentage` | DL user-plane | Percentage of DL user-plane packets delivered on time to the RU. |
| `dlc_ontime_percentage` | DL control-plane | Percentage of DL control-plane packets delivered on time to the RU. |
| `ulc_ontime_percentage` | UL control-plane | Percentage of UL control-plane packets delivered on time to the RU. |
| `ulutx_pucch_ontime_percentage` | UL TX PUCCH | Percentage of PUCCH UL-TX indication packets delivered on time. |
| `ulutx_pusch_ontime_percentage` | UL TX PUSCH | Percentage of PUSCH UL-TX indication packets delivered on time. |
| `ulutx_prach_ontime_percentage` | UL TX PRACH | Percentage of PRACH UL-TX indication packets delivered on time. |
| `ulutx_srs_ontime_percentage` | UL TX SRS | Percentage of SRS UL-TX indication packets delivered on time. MMIMO only. |

---

### GPU Duration Metrics

GPU duration metrics capture per-slot processing time on the GPU. There are two
flavors:

- **`total_duration`** — Wall-clock time including **setup stages** (e.g., memory
  allocation, descriptor preparation) and kernel execution. Used for **DL channels**
  where setup is on the critical path (processing must complete before OTA).
- **`run_duration`** — Actual GPU kernel execution time, **excluding setup**. Used
  for **UL channels** and **BFW** where setup occurs before data arrives over the
  fronthaul and is therefore not on the critical path.

#### DL GPU Durations (`total_duration`)

| Column | GPU Channel | Description |
|--------|-------------|-------------|
| `pdsch_gpu_total_duration` | `PDSCH Aggr` | PDSCH (Physical Downlink Shared Channel) total GPU processing time. |
| `pdcch_gpu_total_duration` | `Aggr PDCCH DL` | PDCCH (Physical Downlink Control Channel) total GPU processing time. |
| `csirs_gpu_total_duration` | `Aggr CSI-RS` | CSI-RS (Channel State Information Reference Signal) total GPU processing time. |
| `pbch_gpu_total_duration` | `Aggr PBCH` | PBCH (Physical Broadcast Channel) total GPU processing time. |
| `compression_gpu_total_duration` | `COMPRESSION DL` | DL fronthaul compression total GPU processing time. |

#### UL GPU Durations (`run_duration`)

| Column | GPU Channel | Description |
|--------|-------------|-------------|
| `pusch_gpu_run_duration` | `PUSCH Aggr` | PUSCH (Physical Uplink Shared Channel) GPU kernel execution time. |
| `pucch_gpu_run_duration` | `PUCCH Aggr` | PUCCH (Physical Uplink Control Channel) GPU kernel execution time. |
| `prach_gpu_run_duration` | `PRACH Aggr` | PRACH (Physical Random Access Channel) GPU kernel execution time. |
| `srs_gpu_run_duration` | `SRS Aggr` | SRS (Sounding Reference Signal) GPU kernel execution time. MMIMO only. |

#### BFW GPU Durations (`run_duration`)

BFW kernels use `run_duration` (not `total_duration`) because BFW setup is not on
the critical path.

| Column | GPU Channel | Description |
|--------|-------------|-------------|
| `dlbfw_gpu_run_duration` | `Aggr DL_BFW` | DL beamforming weight GPU kernel execution time. MMIMO only. |
| `ulbfw_gpu_run_duration` | `UL_BFW Aggr` | UL beamforming weight GPU kernel execution time. Currently not populated. |

---

## Log Source and Tag Dependencies

Every metric in the CSV is derived from one or more nvlog tags parsed from the PHY,
testmac, or RU log files. The table below maps each metric to its required nvlog tag
and, for TI-based metrics, the task/subtask that produces the measurement.

### Tick and L2 Start

| Metric | Log File | nvlog Tag |
|--------|----------|-----------|
| `tick_to_l2_start` | PHY | `L2A.TICK_TIMES` |

### TestMAC / FAPI

| Metric | Log File | nvlog Tag |
|--------|----------|-----------|
| `testmac_start_min` | testmac | `MAC.PROCESSING_TIMES` |
| `testmac_start_max` | testmac | `MAC.PROCESSING_TIMES` |
| `testmac_end_min` | testmac | `MAC.PROCESSING_TIMES` |
| `testmac_end_max` | testmac | `MAC.PROCESSING_TIMES` |
| `testmac_duration` | testmac | `MAC.PROCESSING_TIMES` |

### L2 Adapter

| Metric | Log File | nvlog Tag |
|--------|----------|-----------|
| `l2a_start_max` | PHY | `L2A.PROCESSING_TIMES` |
| `l2a_end_max` | PHY | `L2A.PROCESSING_TIMES` |
| `l2a_duration` | PHY | `L2A.PROCESSING_TIMES` |

### DL Processing Timelines (TI-based)

All TI metrics are parsed from `{TI}` markers in the PHY log. CPU task tracing must
be enabled in the cuPHY controller configuration.

| Metric | TI Task | TI Subtask | Completion Trigger |
|--------|---------|------------|--------------------|
| `tick_to_fhcb_completion` | `DL Task FH Callback` | `Full Task` | CPU — task scope (FH callback is CPU-only) |
| `tick_to_dlu_completion` | `Debug Task` | `Trigger synchronize` | CUDA event — non-blocking poll on TX end event (`getTxEndEvt`), or CPU atomic doorbell (`waitDlCpuDoorBellTaskDone`) when GPU-comm-via-CPU is enabled |
| `tick_to_dlc_bfw_completion` | `DL Task C-Plane N` | `bfw_prepare cell_X sym_0` (legacy: `CPlane Prepare`) | CPU — fronthaul C-plane message send (`send_cplane_mmimo`) |
| `tick_to_dlc_nonbfw_completion` | `DL Task C-Plane N` | `nonbfw_prepare cell_X sym_0` (legacy: `CPlane Prepare`) | CPU — fronthaul C-plane message send (`send_cplane_mmimo`) |
| `tick_to_dlbfw_completion` | `DL Task BFW` | `Wait Run Completion` | CUDA event — non-blocking poll on `run_completion` event (`cudaEventQuery`) |

### UL Control-Plane Timelines (TI-based)

| Metric | TI Task | TI Subtask | Completion Trigger |
|--------|---------|------------|--------------------|
| `tick_to_ulc_bfw_completion` | `UL Task CPlane N` | `bfw_prepare cell_X sym_0` (legacy: `CPlane Prepare` or `Send C Plane`) | CPU — fronthaul C-plane message send |
| `tick_to_ulc_nonbfw_completion` | `UL Task CPlane N` | `nonbfw_prepare cell_X sym_0` (legacy: `CPlane Prepare` or `Send C Plane`) | CPU — fronthaul C-plane message send |
| `tick_to_ulbfw_completion` | `UL Task UL AGGR 3 ULBFW` | `Callback ULBFW` (alt: `UL Task UL AGGR 3` / `Callback ULBFW`) | CUDA event — `PhyWaiter` polls `run_completion` event, then CPU callback |

### UL User-Plane Timelines (TI-based)

| Metric | TI Task | TI Subtask | Completion Trigger |
|--------|---------|------------|--------------------|
| `t0_to_order_kernel_completion` | `UL Task UL AGGR 3` | `Wait Order` | GPU host-pinned write — `OrderWaiter` polls a host-pinned flag written by the GPU order kernel |
| `t0_to_pucch_completion` | `UL Task UL AGGR 3` | `Callback PUCCH` | CUDA event — `PhyWaiter` polls `run_completion` event, then CPU validate + callback |
| `t0_to_pusch_completion` | `UL Task UL AGGR 3` | `Callback PUSCH` | CUDA event — `PhyWaiter` polls `run_completion` event, then CPU validate + callback |
| `t0_to_pusch_eh_completion` | `UL Task AGGR3 Early UCI IND` | `Signal Completion` | CUDA event — polls `subSlotCompletedEvent` for early UCI detection, then CPU callback to L2A, then GPU full-slot D2H copy launch |
| `t0_to_prach_completion` | `UL Task UL AGGR 3` | `Callback PRACH` | CUDA event — `PhyWaiter` polls `run_completion` event, then CPU validate + callback |
| `t0_to_srs_completion` | `UL Task UL AGGR 3 SRS` | `Callback SRS` (alt: `UL Task UL AGGR 3` / `Callback SRS`) | CUDA event — `PhyWaiter` polls `run_completion` event, then CPU validate + callback |

### Headroom Metrics (derived)

Headroom metrics are computed from the corresponding completion metric and a
deadline constant — they require no additional log tags beyond what the completion
metric needs.

| Metric | Derived From |
|--------|--------------|
| `dlu_headroom` | `tick_to_dlu_completion` |
| `dlc_bfw_headroom` | `tick_to_dlc_bfw_completion` |
| `dlc_nonbfw_headroom` | `tick_to_dlc_nonbfw_completion` |
| `dlbfw_headroom` | `tick_to_dlbfw_completion` |
| `ulc_bfw_headroom` | `tick_to_ulc_bfw_completion` |
| `ulc_nonbfw_headroom` | `tick_to_ulc_nonbfw_completion` |
| `ulbfw_headroom` | `tick_to_ulbfw_completion` |
| `pucch_headroom` | `t0_to_pucch_completion` |
| `pusch_eh_headroom` | `t0_to_pusch_eh_completion` |
| `pusch_headroom` | `t0_to_pusch_completion` |
| `prach_headroom` | `t0_to_prach_completion` |
| `srs_headroom` | `t0_to_srs_completion` |

### Ontime Percentages

| Metric | Log File | nvlog Tag |
|--------|----------|-----------|
| `ulu_ontime_percentage` | PHY | `DRV.UL_PACKET_SUMMARY` |
| `srs_ontime_percentage` | PHY | `DRV.SRS_PACKET_SUMMARY` |
| `dlu_ontime_percentage` | RU | `FH.PACKET_SUMMARY` |
| `dlc_ontime_percentage` | RU | `FH.PACKET_SUMMARY` |
| `ulc_ontime_percentage` | RU | `FH.PACKET_SUMMARY` |
| `ulutx_pucch_ontime_percentage` | RU | `FH.PACKET_SUMMARY` |
| `ulutx_pusch_ontime_percentage` | RU | `FH.PACKET_SUMMARY` |
| `ulutx_prach_ontime_percentage` | RU | `FH.PACKET_SUMMARY` |
| `ulutx_srs_ontime_percentage` | RU | `FH.PACKET_SUMMARY` |

### GPU Durations

| Metric | Log File | nvlog Tag | GPU Channel |
|--------|----------|-----------|-------------|
| `pdsch_gpu_total_duration` | PHY | `DRV.MAP_DL` | `PDSCH Aggr` |
| `pdcch_gpu_total_duration` | PHY | `DRV.MAP_DL` | `Aggr PDCCH DL` |
| `csirs_gpu_total_duration` | PHY | `DRV.MAP_DL` | `Aggr CSI-RS` |
| `pbch_gpu_total_duration` | PHY | `DRV.MAP_DL` | `Aggr PBCH` |
| `compression_gpu_total_duration` | PHY | `DRV.MAP_DL` | `COMPRESSION DL` |
| `pusch_gpu_run_duration` | PHY | `DRV.MAP_UL` | `PUSCH Aggr` |
| `pucch_gpu_run_duration` | PHY | `DRV.MAP_UL` | `PUCCH Aggr` |
| `prach_gpu_run_duration` | PHY | `DRV.MAP_UL` | `PRACH Aggr` |
| `srs_gpu_run_duration` | PHY | `DRV.MAP_UL` | `SRS Aggr` |
| `dlbfw_gpu_run_duration` | PHY | `DRV.MAP_DL` | `Aggr DL_BFW` |
| `ulbfw_gpu_run_duration` | PHY | `DRV.MAP_UL` | `UL_BFW Aggr` |

---

## Test Case Variations

### 4TR vs 64TR (MMIMO)

The `--mmimo_enable` (`-e`) flag selects between 4TR and 64TR configurations. This
affects:

- **Fronthaul reception window deadlines** — 64TR has different (generally tighter)
  BFW deadlines than 4TR.
- **MMIMO-only metrics** — The following columns are only populated for 64TR runs
  and will contain `None` for 4TR:
  `t0_to_srs_completion`, `srs_headroom`, `srs_ontime_percentage`,
  `tick_to_dlbfw_completion`, `dlbfw_headroom`, `srs_gpu_run_duration`,
  `dlbfw_gpu_run_duration`, `ulutx_srs_ontime_percentage`.

### Early HARQ (EH) vs Non-EH

When EH is enabled, `t0_to_pusch_eh_completion` and `pusch_eh_headroom` are
populated for UL slots. EH slots use a **2000 us** deadline from T0 while non-EH
slots use **4500 us**. When EH is not enabled, these columns contain `None`.

### PUSCH-Only Test Cases

Some test cases only schedule PUSCH traffic on the UL side (no PUCCH or PRACH). In
this configuration the following columns will contain `None` and are excluded from
validation: `t0_to_pucch_completion`, `t0_to_prach_completion`, `pucch_headroom`,
`prach_headroom`, `pucch_gpu_run_duration`, `prach_gpu_run_duration`.

### Ignored UL Channels

The `--ignore_ul_channels` argument allows selectively disabling validation for
specific UL channels (PUCCH, PRACH, SRS, PUSCH). All timeline, headroom, GPU
duration, and ontime metrics for the specified channels are excluded from validation.

---

## Max CSV Output

When the `-s` flag is provided, a separate CSV is written containing a single row
with the **maximum value across all slots** for every metric column. Note that the
interpretation of "maximum" depends on the metric type:

- **Timing / completion metrics** — higher = later completion = worse. The max
  represents the worst-case slot.
- **Headroom metrics** — higher = more margin = better. The max represents the
  best-case slot.
- **Ontime percentages** — higher = more packets on time = better. The max
  represents the best-case slot.

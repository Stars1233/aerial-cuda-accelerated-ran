# Aerial GPU Benchmark CICD Cases

This folder is the CICD entry point for the 80000-series Aerial GPU Benchmark
perf tests.
Use it to find the YAML input, semantic case name, and run command for each
case. SM allocations and expected capacities are maintained in
`cicd_cases.json`.

## Files


| File                                                                  | Purpose                                                                                                                                                           |
| --------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `cubb_gpu_test_config_<case>.yaml` | Aerial GPU Benchmark YAML inputs passed to `measure.py` (currently 80001, 80002, 80003, 80004, 80011, 80013, 80021, and 80023).                                                                      |
| `cicd_cases.json`                                                     | Case database for SM allocation, expected cell capacity, and run settings.                                                                                        |
| `run_cicd_cases.py`                                                   | Python entrypoint that reads `cicd_cases.json`, generates standalone per-case shell scripts, and can run them.                                                    |
| `README.md`                                                           | Human-readable case definitions and operating notes.                                                                                                              |


## Test Case Definitions


SM allocation and expected capacity are **per-GPU** under each case's
`gpu_configs` map in `cicd_cases.json` (schema_version 2), which is the source
of truth.

> **Disclaimer:** These values are based on initial SM allocations and the
> measurements/configuration captured at that time. They do not represent the
> latest performance of the Aerial SDK; rerun the cases after SDK or allocation
> changes before using them for a current performance comparison.

Case numbering: the **last digit selects the base case** and the **tens digit
selects the variant family** — `8000x` = base, `8001x` = base + early HARQ,
`8002x` = base + cuMAC in a green context. For example, 80013 is 80003 plus
early HARQ, and 80023 is 80003 plus cuMAC.

| Case  | Long name                                    | Delta                                                                               |
| ----- | -------------------------------------------- | ----------------------------------------------------------------------------------- |
| 80001 | `perf103_72a_dl200_pusch1250_o00000`         | 72a full-GPU baseline, no offload; PUSCH and PUCCH cascaded, 1250 us PUSCH budget. |
| 80002 | `perf103_72e_dl200_pusch700_o00000`          | 72e baseline; PUSCH budgets tightened to 700/719 us.                               |
| 80003 | `perf103_72e_dl200_pusch1200_o00000`         | 80002 with PUSCH budgets relaxed by 500 us to 1200/1219.                           |
| 80004 | `perf103_72e_dl200_pusch750_o00000_ovl500`   | 80002 with 750/750 budgets; PUSCH2 non-cascaded at a fixed 500 us offset (overlap). |
| 80011 | `perf103_72a_dl200_pusch1250_o00000_eh650`   | 80001 plus early HARQ with 650 us subslot budget.                                  |
| 80013 | `perf103_72e_dl200_pusch1200_o00000_eh216`   | 80003 plus early HARQ with 216 us subslot budget.                                  |
| 80021 | `perf103_72a_dl200_pusch1250_o00000_macgc`   | 80001 plus cuMAC in a dedicated green context (6th SM entry = MAC).                |
| 80023 | `perf103_72e_dl200_pusch1200_o00000_macgc`   | 80003 plus cuMAC in a dedicated green context (6th SM entry = MAC).                |

### GH200 1980 MHz operating points

The GH200 1980 MHz SM allocations and expected capacities are maintained in
`gpu_configs.GH200.alt_freqs.1980` in `cicd_cases.json`; that file is the
source of truth.

The expected capacities are seeded from the closest previously measured case
(or the YAML `cap` value) so CICD has a starting point. Update the matching
`gpu_configs.<GPU>.alt_freqs.<freq_mhz>.expected_cell_capacity` in
`cicd_cases.json` after the first qualified run confirms the stable capacity.

## Default GPU And Operating Points

**RTX PRO 4500 is the default GPU**: the `freq`/`power`/`target` values embedded
in each case YAML are the RTX PRO 4500 operating point (1610 MHz / 165 W), so a
direct `measure.py --yaml ...` run without overrides exercises the RTX PRO 4500
config. Check those values before running on a different GPU.

CICD itself always overrides `freq`/`power`/`target` per detected GPU from
`cicd_cases.json`. The default operating points there are:

- **GH200**: 1000 MHz (default). A developer-tuned 1980 MHz operating point is
  kept under `alt_freqs`; select it explicitly with `--freq-mhz 1980`.
- **RTX PRO 4500**: 1610 MHz (default).

## SM Allocation Order

The database uses this order:

```text
PRACH PDCCH PUCCH PDSCH PUSCH [MAC]
```

The trailing `MAC` slot is optional and present only for a case that enables
cuMAC: the green-context cases 80021 and 80023 carry six entries, all other
cases carry five. cuMAC requires green contexts to run: the runtime switch is
`use_green_contexts` in the case YAML's `config:` block (the same-named field
in `cicd_cases.json` is informational metadata used for tool filtering, not
the runtime toggle). Green contexts partition the GPU exclusively, so those SM
allocations are **manually tuned** — `run_sm_opt.py` does not apply to them.
The 80021/80023 RTX PRO 4500 entries carry the manually tuned overlapping
DL/UL layout; the GH200 entries are that layout scaled to 132 SMs at
granularity 8 and still need on-target tuning (see `tuning_status` in
`cicd_cases.json`).

### Green-context SM ranges (`green_context_sm_alloc`)

Green contexts are placed by **split ranges**, not just counts. Each
green-context `gpu_configs` entry in `cicd_cases.json` carries a
`green_context_sm_alloc` map of `CHANNEL: [split_start, sm_count]` alongside
the `sm_allocation` counts (keep the two consistent). The testbench reads
these ranges **only from its YAML** — they cannot be passed through
`measure.py` CLI args — so the generated wrapper script materializes a temp
YAML in the run dir (`<case>_gc_<GPU>.yaml`) with the detected GPU's ranges
appended at the YAML root, and runs all steps against it. A root-level block
overrides the config-level `green_context_sm_alloc` default embedded in the
case YAML (which holds the RTX PRO 4500 ranges for direct `measure.py` runs).
Starts/counts should stay aligned to the GPU's SM granularity (8). Ranges are
expected to overlap across DL and UL channels (e.g. PDSCH under PUSCH, MAC
under PDSCH, PUCCH sharing PRACH's block): in TDD the two directions are
active in different slots, so overlapping their splits lets each side use
more SMs than an exclusive partition would allow.

## Run Workflow

The Python entrypoint generates a standalone shell script per case and runs this
workflow:

1. 2C functionality test with `measure.py --enable_ref_check`.
2. Cell-capacity confirmation with `confirm_cell_capacity.py`, starting from `expected_cell_capacity` in `cicd_cases.json`.
3. Power sweep from `1C` through the confirmed capacity with `measure.py --measure_power`.
4. Nsys trace for each cell count swept during latency confirmation.

List the CICD cases:

```bash
python3 testBenches/perf/cicd_cases/run_cicd_cases.py list
```

Run one case:

```bash
python3 testBenches/perf/cicd_cases/run_cicd_cases.py --case 80001
```

That command also writes this offline wrapper:

```text
testBenches/perf/cicd_cases/cubb_gpu_test_80001.sh
```

Run the generated wrapper later:

```bash
testBenches/perf/cicd_cases/cubb_gpu_test_80001.sh
```

The generated script contains explicit `measure.py`, `confirm_cell_capacity.py`,
power plot, and nsys commands. It does not call back into `run_cicd_cases.py`.

Generate wrappers without running tests:

```bash
python3 testBenches/perf/cicd_cases/run_cicd_cases.py scripts --case all
```

Or use `--script_only` with the normal run syntax:

```bash
python3 testBenches/perf/cicd_cases/run_cicd_cases.py --case 80001 --script_only
```

Run case 80001 inside the standard container:

```bash
docker exec -it <container_name> bash -lc 'cd "${cuBB_SDK:-/workspace/aerial_sdk}" && testBenches/perf/cicd_cases/cubb_gpu_test_80001.sh'
```

Run all CICD cases:

```bash
python3 testBenches/perf/cicd_cases/run_cicd_cases.py run --case all
```

Each case in `cicd_cases.json` carries an `enabled` flag, so extra case YAMLs
can be kept in the database without being part of the default sweep. `--case
all` selects only enabled cases; add `--include-disabled` (alias
`--include_disabled`) to select every case:

```bash
python3 testBenches/perf/cicd_cases/run_cicd_cases.py run --case all --include-disabled
```

An explicitly selected case (e.g. `--case 80021`) always runs even when
disabled; `list` marks disabled cases with `[disabled]`.

Print commands without running hardware tests:

```bash
python3 testBenches/perf/cicd_cases/run_cicd_cases.py --case 80001 --dry-run
```

When a selected case ID starts with `8002*`, the runner reads the ONNX filename
from `config.cumac_options.airan.model_path` in the case YAML and checks for it
under `testVectors`. If it is absent, the runner invokes
`cuMAC/examples/ml/trtEngine/gen_models.py` with its default model config and
writes the generated artifacts directly to `testVectors`. Selected `8002*`
cases must name the same ONNX model. Existing ONNX files are reused by default;
force fresh generation with `--overwrite-onnx`:

```bash
python3 testBenches/perf/cicd_cases/run_cicd_cases.py run --case 80021 --overwrite-onnx
```

Override the database `defaults.confirm_slots` value for a shorter capacity
confirmation run:

```bash
python3 testBenches/perf/cicd_cases/run_cicd_cases.py --case 80001 --slots 30
```

The same override can be baked into a generated wrapper:

```bash
python3 testBenches/perf/cicd_cases/run_cicd_cases.py --case 80001 --script_only --slots 30
```

Or passed directly to an existing wrapper:

```bash
testBenches/perf/cicd_cases/cubb_gpu_test_80001.sh --slots 30
```

### GPU-metrics profiling (nsys runs under `sudo`)

Step 4's nsys trace collects **GPU hardware metrics** (`--gpu-metrics-devices`), which the NVIDIA
driver only permits for **admin** users when `RmProfilingAdminOnly=1` (the default). So the nsys
command is run under **`sudo -E`** by default — no flag needed; as a plain user it would otherwise
fail with `ERR_NVGPUCTRPERM` ("Insufficient privilege").

Because a `sudo` (root) process cannot write an **NFS `root_squash`** workspace, nsys writes its
report to local `/tmp`, which is then `chown`ed + `mv`ed back into the run dir as your user (the nsys
exit code is preserved). This is built into all cases — TDD and FDD
(`measure/TDD/configure_debug.py`, `measure/TDD/configure_power_debug.py`,
`measure/FDD/configure_debug.py`) — and requires passwordless `sudo` inside the container (provided by
the standard Aerial dev container).

Artifacts default to:

```text
testBenches/perf/cicd_cases/runs/<case>/<run_id>/
```

Use `--output-root <dir>` to redirect artifacts elsewhere.

## Database Updates

`cicd_cases.json` is `schema_version 2`: `sm_allocation`, `expected_cell_capacity`,
`freq_mhz`, and `power_w` live **per-GPU** under each case's `gpu_configs.<GPU>`
map (not at the case top level). When a confirmed capacity is accepted for a
case on a given GPU, update that GPU's entry:

```json
"gpu_configs": {
  "GH200": {
    "freq_mhz": 1000,
    "power_w": 900,
    "sm_allocation": [8, 12, 12, 106, 114],
    "expected_cell_capacity": 15
  }
}
```

Keep the YAML config and database aligned:

- `yaml` (case-level) points to the `cubb_gpu_test_config_<case>.yaml` file.
- `gpu_configs.<GPU>.sm_allocation` is the SM allocation passed to
  `measure.py --target` for that GPU.
- `gpu_configs.<GPU>.expected_cell_capacity` is the per-GPU seed used by
  `confirm_cell_capacity.py`.
- `gpu_configs.<GPU>.freq_mhz` / `power_w` are **required** in each entry (they
  are not inherited from top-level defaults). `defaults.fallback_gpu` names the
  GPU tag whose config is reused when the detected GPU has no own entry — it is
  not a set of field defaults.

## Add A New CICD Case

1. Add a `cubb_gpu_test_config_<case>.yaml` input.
2. Add a matching entry to `cicd_cases.json`.
3. Add one row to this README.
4. Run the case and update `expected_cell_capacity` after capacity is confirmed.

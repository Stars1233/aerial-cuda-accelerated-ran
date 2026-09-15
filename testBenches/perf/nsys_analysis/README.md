# Aerial GPU Benchmark Nsight Systems Analysis

These utilities parse Nsight Systems traces from the Aerial GPU Benchmark and
generate local per-channel kernel-latency reports.

| File | Purpose |
| --- | --- |
| `nsys_parser.py` | Parses an `.nsys-rep` or exported `.sqlite` file into per-channel kernel-latency JSON. |
| `nsys_gen_report.py` | Generates channel PNGs, a combined PDF, and parsed JSON data. |

An `.nsys-rep` input is recommended because the parser performs the required
SQLite export automatically. An existing Nsight Systems `.sqlite` export is
also supported. Input filename endings are matched case-insensitively.

## Dependencies

- Python 3
- Nsight Systems `nsys` command for `.nsys-rep` input
- Matplotlib for `nsys_gen_report.py`

## Generate a report

Run from the repository root:

```bash
python3 testBenches/perf/nsys_analysis/nsys_gen_report.py \
    path/to/run.nsys-rep
```

For an existing SQLite export:

```bash
python3 testBenches/perf/nsys_analysis/nsys_gen_report.py \
    path/to/run.sqlite
```

Without `-o`, the report is written to a directory named after the input file,
next to that file:

```text
path/to/run.nsys-rep -> path/to/run/
path/to/run.sqlite   -> path/to/run/
```

Use `-o` to select another output directory:

```bash
python3 testBenches/perf/nsys_analysis/nsys_gen_report.py \
    path/to/run.nsys-rep -o path/to/report
```

Each report contains:

- `YYYYMMDD_nsys_kernel_latency_report.pdf`
- `YYYYMMDD_nsys_kernel_latency_data.json`
- `01_pdsch_kernel_latency.png`, `02_pusch_kernel_latency.png`, and the
  remaining channel PNGs

The date is the local report-generation date. Re-running a report on the same
day overwrites that day's PDF, JSON, and report PNGs. Unrelated files in the
output directory are left untouched.

Use `--top-k N` to keep only the largest `N` kernel slices per channel and
combine the remainder as `Other`. The default `0` shows every kernel. Use
`--dpi` to control PNG resolution and `--nsys-bin` to select the `nsys`
executable.

## Parse JSON without generating figures

Write parsed data to stdout:

```bash
python3 testBenches/perf/nsys_analysis/nsys_parser.py path/to/run.nsys-rep
```

Write parsed data to a selected JSON file:

```bash
python3 testBenches/perf/nsys_analysis/nsys_parser.py \
    path/to/run.nsys-rep -o kernels.json
```

The parser normally removes the temporary SQLite export created for an
`.nsys-rep`. Add `--keep-sqlite` to preserve that export next to the trace.
When an existing `.sqlite` file is supplied directly, the parser never removes
the input file.

The Aerial GPU Benchmark extraction pipeline also loads `nsys_parser.py` to
embed kernel-latency data in its dashboard records. Parser failures remain
optional there and do not stop extraction of the other benchmark data.

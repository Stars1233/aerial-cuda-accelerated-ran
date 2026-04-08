# CICD Scripts

Scripts for parsing logs, extracting performance metrics, and applying threshold checks in CI/CD pipelines. These tools enable automated regression protection and performance validation for Aerial cuBB tests.

## Overview

The CICD workflow consists of two main phases:

1. **Log Parsing & Metrics Extraction**: Parse PHY/testMAC/RU logs and extract performance metrics
2. **Threshold Checking**: Compare metrics against perf_requirements files to detect regressions or validate requirements

All threshold checks (gating, warning, absolute) use the same validation script (`cicd_performance_threshold_absolute.py`) with the unified perf_requirements file format.

### Offline: Threshold Generation

Run on a small set of baseline runs (~3) to generate gating and warning requirements. Uses the absolute template which contains pre-configured `gating_mean_headroom` and `warning_mean_headroom` values that define the allowed buffer above/below the mean worst-case.

```
  N x perf.csv                     perf_requirements/absolute/*.csv
  (~3 baseline runs)               (with gating_mean_headroom, warning_mean_headroom)
        |                                       |
        +-------------------+  +-----------------+
                            |  |
                            v  v
                cicd_threshold_generate.py
                    |                    |
                    v                    v
  perf_requirements/                perf_requirements/
  warning_gating/                   warning_gating/
  gating_*.csv                      warning_*.csv
  (e.g. gating_perf_               (e.g. warning_perf_
   requirements_gh_                  requirements_gh_
   F08_59c_BFP9_EH_20C.csv)         F08_59c_BFP9_EH_20C.csv)
```

### Offline: Variance Characterization (Diagnostic)

Optional diagnostic tool. Run across historical runs (10+) to compute empirical standard deviation and compare against configured `gating_mean_headroom`/`warning_mean_headroom` values. Does not modify any files; produces console output only. Use to spot-check whether configured headroom values are appropriate.

```
  N x perf.csv                perf_requirements/absolute/*.csv
  (10+ historical runs)       (with gating_mean_headroom, warning_mean_headroom)
        |                              |
        +--------------+  +------------+
                       |  |
                       v  v
         cicd_variance_characterize.py
                        |
                        v
              Console: std comparison report
              (no file output)
```

### Online: CICD Validation Pipeline

Every CICD run validates the current `perf.csv` against gating, warning, and absolute requirements using the same script.

```
                              perf.csv (current run)
                                       |
  +------------------------------------+--------------------------------------+
  |                        post_processing_cicd.sh                            |
  |                                                                           |
  |  warning_gating/gating_*.csv ---> cicd_performance_threshold_absolute.py  |
  |                                   (gating: fail = PIPELINE FAIL)          |
  |                                                                           |
  |  warning_gating/warning_*.csv --> cicd_performance_threshold_absolute.py  |
  |                                   (warning: fail = WARNING)               |
  |                                                                           |
  |  absolute/*.csv ----------------> cicd_performance_threshold_absolute.py  |
  |                                   (absolute: fail = PIPELINE FAIL)        |
  |                                                                           |
  +------------------------------------+--------------------------------------+
                                       |
                            0=PASS, 1=FAIL, 2=WARN
```

## Scripts

### cicd_parse.py

Parses log files into binary format for efficient downstream processing. This is the most computationally expensive step and should only be run once per log set.

**Usage:**
```bash
# 64TR (mMIMO) - use -e flag
python cicd_parse.py phy.log testmac.log ru.log -o ./binary -f PerfMetrics -e

# 4TR (non-mMIMO) - omit -e flag
python cicd_parse.py phy.log testmac.log ru.log -o ./binary -f PerfMetrics
```

**Output formats (`-f`):**
- `PerfMetrics` - Parses timing data for performance metrics extraction. Output is used by `cicd_performance_metrics.py`.
- `LatencySummary` - Parses NIC latency data for latency summary visualization. Used for NICD analysis.

**Key options:**
- `-e, --mmimo_enable` - **Required for 64TR (mMIMO) configurations.** Adjusts timeline parsing for mMIMO-specific timing requirements. Omit for 4TR configurations.
- `-n, --num_proc` - Number of parallel parsing processes (default: 8). Increase for faster parsing on systems with more cores.
- `-m, --max_duration` - Maximum duration in seconds to process from logs.
- `-i, --ignore_duration` - Initial seconds to skip (useful to exclude startup transients).
- `-b, --bin_format` - Output binary format: `feather` (default), `parquet`, `hdf5`, `json`, or `csv`.

### cicd_performance_metrics.py

Extracts performance metrics from parsed binary data and outputs a `perf.csv` file containing 99th percentile (1% CCDF) timing measurements for all pipeline stages, organized by slot number.

**Usage:**
```bash
# 64TR (mMIMO) - use -e flag
python cicd_performance_metrics.py ./binary -p perf.csv -e

# 4TR (non-mMIMO) - omit -e flag
python cicd_performance_metrics.py ./binary -p perf.csv
```

**Key options:**
- `-e, --mmimo_enable` - **Required for 64TR (mMIMO) configurations.** Adjusts timeline calculations for mMIMO-specific timing requirements. Omit for 4TR configurations.
- `-p, --performance_metrics_outfile` - Output CSV file path for performance metrics (required for downstream threshold checking).
- `-s, --max_outfile` - Optional output CSV containing the worst-case (maximum) value across all slots for each metric.
- `-m, --max_duration` - Maximum duration in seconds to process from logs.
- `-i, --ignore_duration` - Initial seconds to skip (useful to exclude startup transients).
- `-q, --quantile` - Quantile for metrics calculation (default: 0.99 for 99th percentile).
- `-t, --ignore_ticks` - Skip tick message requirements (tick_to_l2_start will be empty in output).

### cicd_performance_threshold_absolute.py

Validates performance metrics against perf_requirements files. Used for all three threshold check types: gating, warning, and absolute. The distinction between check types is handled externally by the CI/CD wrapper scripts based on the return code.

**Usage:**
```bash
# Gating check
python cicd_performance_threshold_absolute.py perf.csv perf_requirements/warning_gating/gating_perf_requirements_gh_F08_59c_BFP9_EH_20C.csv \
    -o gating_threshold_results.csv

# Warning check
python cicd_performance_threshold_absolute.py perf.csv perf_requirements/warning_gating/warning_perf_requirements_gh_F08_59c_BFP9_EH_20C.csv \
    -o warning_threshold_results.csv

# Absolute check
python cicd_performance_threshold_absolute.py perf.csv perf_requirements/absolute/perf_requirements_gh_F08_59c_BFP9_EH_20C.csv \
    -o absolute_threshold_results.csv
```

**Arguments:**
- `performance_csv` - Input perf.csv file from `cicd_performance_metrics.py`.
- `requirements_csv` - perf_requirements file defining thresholds per metric.

**Key options:**
- `-o, --output_csv` - Output CSV file with detailed pass/fail results per metric.

**Return codes:**
- `0` = All requirements met (PASS)
- `1` = One or more requirements not met (FAIL)

**Requirements file format (`metric_name,required_value,headroom,slots`):**
- `metric_name` - Name of the metric to check (must match column in perf.csv).
- `required_value` - The timing requirement/deadline.
- `headroom` - Margin subtracted from required_value to get the threshold.
- `slots` (optional) - Comma-separated list of slots this requirement applies to. Empty = all slots.

Additional columns (`gating_mean_headroom`, `warning_mean_headroom`, `mean_worst`, `mean_headroom`) are ignored during validation.

### cicd_variance_characterize.py

Diagnostic tool that computes empirical standard deviation of worst-case performance across multiple historical perf.csv files, then compares against the `gating_mean_headroom` and `warning_mean_headroom` values configured in the absolute template. Does not produce any output files.

**Usage:**
```bash
python cicd_variance_characterize.py perf_requirements_4tr_eh.csv perf1.csv perf2.csv perf3.csv
```

**Arguments:**
- `input_requirements` - Perf_requirements CSV template (with `gating_mean_headroom`/`warning_mean_headroom` columns).
- `performance_csvs` - One or more perf.csv files from historical runs. At least 2 recommended.

**Output:** Console report showing computed std, suggested headroom (std * K), and current configured headroom for each metric. Flags metrics where suggested headroom exceeds configured values.

### cicd_threshold_generate.py

Generates gating and warning perf_requirements files from one or more baseline perf.csv files and an absolute template (with `gating_mean_headroom` and `warning_mean_headroom` columns).

**Usage:**
```bash
python cicd_threshold_generate.py perf_requirements.csv gating_output.csv warning_output.csv \
    perf1.csv perf2.csv ... perfN.csv
```

**Arguments:**
- `requirements_csv` - Perf_requirements CSV (with `gating_mean_headroom` and `warning_mean_headroom` columns).
- `gating_output_csv` - Output path for gating perf_requirements file.
- `warning_output_csv` - Output path for warning perf_requirements file.
- `performance_csvs` - One or more baseline perf.csv files. Mean of worst-cases across files is used as the baseline.

**Logic:** For each metric group, computes the worst-case value per perf.csv file, takes the mean across files, then applies `threshold = mean_worst + mean_headroom` (clamped against the absolute threshold so gating/warning never exceed absolute limits). The `mean_headroom` values are pre-configured in the absolute template and can be hand-tuned per metric. Higher-is-better metrics (`*_percentage`) use inverted direction (`threshold = mean_worst - mean_headroom`).

**Output format:** Generated files contain `metric_name,required_value,headroom,slots,mean_worst,mean_headroom` where `headroom` is the computed threshold headroom, `mean_worst` is the observed mean worst-case value from the baseline runs, and `mean_headroom` is a pass-through of the configured buffer used.

### cicd_threshold_generate_all.py

Batch generates gating/warning perf_requirements files for multiple test cases from a directory of CI/CD results. Automatically detects test configuration from folder names, groups `_RUN\d+` folders by base test case name, and matches against templates in `perf_requirements/absolute/`. All perf.csv files for the same test case are passed together to compute mean-based baselines.

**Usage:**
```bash
python cicd_threshold_generate_all.py /path/to/cicd_results /path/to/output --gh
```

**Arguments:**
- `input_folder` - Directory containing CI/CD result folders (expects `F08_*` pattern).
- `output_folder` - Output folder for generated gating/warning perf_requirements files.
- `-g, --gh` - Use GH platform naming prefix.
- `-l, --gl4` - Use GL4 platform naming prefix.

Templates are read from `perf_requirements/absolute/`, output is written to the specified output folder.

### cicd_threshold_summary.py

Displays threshold values from multiple perf_requirements files side-by-side with worst-case performance data from perf.csv. Used as a diagnostic tool to visualize where actual performance sits relative to each threshold tier.

**Usage:**
```bash
python cicd_threshold_summary.py perf.csv gating.csv warning.csv absolute.csv \
    -l gating warning absolute \
    -o summary.csv
```

**Arguments:**
- `performance_csv` - Input perf.csv file.
- `requirements_csvs` - One or more perf_requirements files to compare.
- `-l, --labels` - Labels for each requirements file (must match count). Defaults to filenames.
- `-o, --output_csv` - Optional CSV output file.

**Output:** Console table showing one row per metric group with the worst-case performance value and threshold values from each requirements file. Automatically called as an informational step in the CICD pipeline before gating/warning/absolute checks.

## perf_requirements Directory

### perf_requirements/absolute/

Per-pattern template files defining absolute performance requirements and mean headroom values for gating/warning threshold generation. Used for absolute checks and as input to threshold generation.

**File format:** `metric_name,required_value,headroom,slots,gating_mean_headroom,warning_mean_headroom`

- `headroom` - Absolute headroom (hard limit margin from required_value).
- `gating_mean_headroom` - Buffer above/below mean worst-case for gating thresholds. Hand-tunable per metric.
- `warning_mean_headroom` - Buffer above/below mean worst-case for warning thresholds. Hand-tunable per metric.

**Naming convention:** `perf_requirements_gh_F08_{pattern}_{compression}_{EH}_{cells}C.csv`

Also contains the 4 generic fallback files:
- `perf_requirements_4tr_eh.csv`
- `perf_requirements_4tr_noneh.csv`
- `perf_requirements_64tr_eh.csv`
- `perf_requirements_64tr_noneh.csv`

### perf_requirements/warning_gating/

Generated gating and warning perf_requirements files with computed headroom based on baseline performance + configured mean headroom. Used for gating and warning checks.

**File format:** `metric_name,required_value,headroom,slots,mean_worst,mean_headroom`

**Naming convention:** `{gating|warning}_perf_requirements_{platform}_F08_{pattern}_{compression}_{EH}_{cells}C.csv`

## Usage with Wrapper Scripts

These scripts are typically invoked through the post-processing wrapper scripts in the parent directory:

```bash
# Full CICD workflow
../../post_processing_cicd.sh phy.log testmac.log ru.log ./output \
    --gating-threshold perf_requirements/warning_gating/gating_perf_requirements_gh_F08_59c_BFP9_EH_20C.csv \
    --warning-threshold perf_requirements/warning_gating/warning_perf_requirements_gh_F08_59c_BFP9_EH_20C.csv \
    --absolute-threshold perf_requirements/absolute/perf_requirements_gh_F08_59c_BFP9_EH_20C.csv \
    --mmimo
```

See the [phase4_test_scripts README](../../../README.md#post-processing-scripts) for detailed workflow documentation.

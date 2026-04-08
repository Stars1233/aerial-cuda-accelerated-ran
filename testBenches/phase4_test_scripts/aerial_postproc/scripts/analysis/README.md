# Analysis Scripts

This folder contains Python scripts for post-processing and analyzing log files generated from Phase 4 test runs. The scripts parse log data and generate interactive HTML visualizations for performance metrics and latency analysis.

## Prerequisites

- Python 3.x
- The `aerial_postproc` package must be installed (see `testBenches/phase4_test_scripts/aerial_postproc/setup.py`)
- Log files generated from a Phase 4 test run: `phy.log`, `testmac.log`, and `ru.log`

## Script Overview

| Script | Description |
|--------|-------------|
| `compare_logs.py` | Generates box-whisker and CCDF plots for CPU/GPU performance metrics analysis |
| `cpu_timeline_plot.py` | Creates CPU and GPU timeline visualization views |
| `latency_summary.py` | Produces CCDF and bar plots for DU/RU latency analysis |
| `latency_timeline_plot.py` | Creates timeline visualizations for latency data |
| `cupti_parse.py` | Parses and visualizes CUPTI GPU kernel profiling data with timeline views, interactive statistics tables, and CSV export |

## Workflow

The typical workflow for analyzing Phase 4 test results involves:

1. **Configure test parameters** using `parse_test_config_params.sh`
2. **Run Phase 4 test** to generate log files (`phy.log`, `testmac.log`, `ru.log`)
3. **Pre-parse logs** using `cicd_parse.py` with the appropriate format
4. **Analyze data** using the visualization scripts

---

## Example 1: Performance Metrics Analysis (available in phase 4 with default settings)

This example demonstrates analyzing performance metrics from a MODCOMP test configuration.

### Step 1: Configure Test Parameters

```bash
cd testBenches/phase4_test_scripts
./parse_test_config_params.sh F08_6C_79_MODCOMP_STT480000_EH_1P CG1_R750 test_params.sh
source test_params.sh
```

### Step 2: Run Phase 4 Test

Run the Phase 4 test using the generated parameters. After the test completes, you will have:
- `phy.log`
- `testmac.log`
- `ru.log`

### Step 3: Pre-parse Logs with PerfMetrics Format

```bash
python3 ../aerial_postproc/scripts/cicd/cicd_parse.py \
    phy.log testmac.log ru.log \
    -f perfmetrics \
    -o ./parsed_perfmetrics/
```

### Step 4: Generate Performance Comparison Plots

```bash
python3 ../aerial_postproc/scripts/analysis/compare_logs.py \
    ./parsed_perfmetrics/ \
    -m 10 \
    -i 1 \
    -e \
    -o compare_results.html
```

**Parameters:**
- `-m 10` - Process up to 10 seconds of data
- `-i 1` - Ignore the first 1 second of data (warm-up period)
- `-e` - Enable mMIMO timeline requirements
- `-o compare_results.html` - Output filename

### Step 5: Generate CPU Timeline Plot

```bash
python3 ../aerial_postproc/scripts/analysis/cpu_timeline_plot.py \
    ./parsed_perfmetrics/ \
    -m 1.2 \
    -i 1 \
    -o cpu_timeline.html
```

**Parameters:**
- `-m 1.2` - Process up to 1.2 seconds of data (shorter duration for timeline visualization)
- `-i 1` - Ignore the first 1 second of data
- `-o cpu_timeline.html` - Output filename

---

## Example 2: NIC Latency Analysis (NICD setting in phase 4)

This example demonstrates analyzing NIC latency data from a NICD_MODCOMP test configuration.

### Step 1: Configure Test Parameters

```bash
cd testBenches/phase4_test_scripts
./parse_test_config_params.sh F08_6C_79_NICD_MODCOMP_STT480000_EH_1P CG1_R750 test_params.sh
source test_params.sh
```

### Step 2: Run Phase 4 Test

Run the Phase 4 test using the generated parameters. After the test completes, you will have:
- `phy.log`
- `testmac.log`
- `ru.log`

### Step 3: Pre-parse Logs with LatencySummary Format

```bash
python3 ../aerial_postproc/scripts/cicd/cicd_parse.py \
    phy.log testmac.log ru.log \
    -f latencysummary \
    -e \
    -o ./parsed_latency/
```

**Parameters:**
- `-f latencysummary` - Use LatencySummary parsing format
- `-e` - Enable mMIMO timeline requirements

### Step 4: Generate Latency Summary Plots

```bash
python3 ../aerial_postproc/scripts/analysis/latency_summary.py \
    ./parsed_latency/ \
    -m 10 \
    -i 1 \
    -e \
    -o latency_summary.html
```

**Parameters:**
- `-m 10` - Process up to 10 seconds of data
- `-i 1` - Ignore the first 1 second of data (warm-up period)
- `-e` - Enable mMIMO timeline requirements
- `-o latency_summary.html` - Output filename

### Step 5: Generate Latency Timeline Plot

```bash
python3 ../aerial_postproc/scripts/analysis/latency_timeline_plot.py \
    ./parsed_latency/ \
    -m 1.2 \
    -i 1 \
    -o latency_timeline.html
```

**Parameters:**
- `-m 1.2` - Process up to 1.2 seconds of data (shorter duration for timeline visualization)
- `-i 1` - Ignore the first 1 second of data
- `-o latency_timeline.html` - Output filename

> **Note:** The `-e` flag can also be added to `latency_timeline_plot.py` for mMIMO tests if needed for consistent timeline window requirements.

---

## Common Command-Line Options

### cicd_parse.py

| Option | Description | Default |
|--------|-------------|---------|
| `-f, --format` | Parse format: `perfmetrics` or `latencysummary` | `perfmetrics` |
| `-o, --output_folder` | Output folder for parsed data | `./` |
| `-m, --max_duration` | Maximum run duration to process (seconds) | 999999999.0 |
| `-i, --ignore_duration` | Seconds at beginning to ignore | 0.0 |
| `-n, --num_proc` | Number of parsing processes | 8 |
| `-e, --mmimo_enable` | Enable mMIMO timeline requirements | False |
| `-b, --bin_format` | Output format: `feather`, `parquet`, `hdf5`, `json`, `csv` | `feather` |

### compare_logs.py

| Option | Description | Default |
|--------|-------------|---------|
| `-m, --max_duration` | Maximum run duration to process (seconds) | 999999999.0 |
| `-i, --ignore_duration` | Seconds at beginning to ignore | 0.0 |
| `-o, --out_filename` | Output HTML filename | `result.html` |
| `-l, --labels` | Labels for each input dataset | Auto-generated |
| `-e, --mmimo_enable` | Enable mMIMO timeline requirements | False |
| `-a, --enable_subtask_breakdown` | Enable task/subtask breakdown for all UL/DL tasks | False |
| `-b, --enable_gpu_prep_breakdown` | Enable CPU durations for GPU setup | False |
| `-c, --enable_ulc_dlc_breakouts` | Enable per-task breakdowns for ULC/DLC | False |
| `-s, --slot_selection` | Slot selection (e.g., "1-4,7,9-11") | All slots |

### cpu_timeline_plot.py

| Option | Description | Default |
|--------|-------------|---------|
| `-m, --max_duration` | Maximum run duration to process (seconds) | 999999999.0 |
| `-i, --ignore_duration` | Seconds at beginning to ignore | 0.0 |
| `-o, --out_filename` | Output HTML filename | `result.html` |
| `-l, --labels` | Labels for each input dataset | Auto-generated |
| `-c, --enable_cpu_percentages` | Enable CPU percentage estimates | False |
| `-u, --ul_only` | Only show UL timeline data | False |

### latency_summary.py

| Option | Description | Default |
|--------|-------------|---------|
| `-m, --max_duration` | Maximum run duration to process (seconds) | 999999999.0 |
| `-i, --ignore_duration` | Seconds at beginning to ignore | 0.0 |
| `-o, --out_filename` | Output HTML filename | None (shows in browser) |
| `-l, --labels` | Labels for each input dataset | Auto-generated |
| `-e, --mmimo_enable` | Enable mMIMO timeline requirements | False |
| `-p, --percentile` | Percentile for statistical analysis (0-1) | 0.99 |
| `-s, --slot_selection` | Slot selection (e.g., "1-4,7,9-11") | All slots |

### latency_timeline_plot.py

| Option | Description | Default |
|--------|-------------|---------|
| `-m, --max_duration` | Maximum run duration to process (seconds) | 999999999.0 |
| `-i, --ignore_duration` | Seconds at beginning to ignore | 0.0 |
| `-o, --out_filename` | Output HTML filename | None (shows in browser) |
| `-l, --labels` | Labels for each input dataset | Auto-generated |
| `-e, --mmimo_enable` | Enable mMIMO timeline requirements | False |
| `-p, --per_symbol` | Enable per-symbol plot | False |
| `-d, --add_downlink` | Enable downlink on timeline plot | False |

### cupti_parse.py

| Option | Description | Default |
|--------|-------------|---------|
| `phy_filename` | PHY log file containing CUPTI data (positional argument) | Required |
| `-m, --max_duration` | Maximum run duration to process (seconds) | 999999999.0 |
| `-i, --ignore_duration` | Seconds at beginning to ignore | 0.0 |
| `-o, --out_filename` | Output HTML filename | `result.html` |
| `-n, --num_proc` | Number of processes for parsing | 8 |
| `-f, --name_filters` | Filter kernels by name (space-separated, partial match) | None (all kernels) |
| `-t, --enable_timeline` | Enable GPU and CPU timeline plots | False |
| `--enable_t0_plots` | Enable T0 timestamp coverage plots for data validation | False |
| `-c, --csv_output` | CSV output file for kernel duration statistics | None |
| `-p, --percentile` | Percentile for CSV output (0.0-1.0) | 0.50 (median) |

---

## Comparing Multiple Runs

All analysis scripts support comparing multiple datasets. You can provide multiple pre-parsed folders or sets of log files:

```bash
# Compare two runs using pre-parsed data
python3 compare_logs.py ./run1_parsed/ ./run2_parsed/ \
    -l "Baseline" "Optimized" \
    -m 10 -i 1 -e \
    -o comparison.html

# Or provide raw log files directly (alternating phy/testmac/ru)
python3 compare_logs.py \
    run1_phy.log run1_testmac.log run1_ru.log \
    run2_phy.log run2_testmac.log run2_ru.log \
    -l "Run1" "Run2" \
    -m 10 -i 1 -e \
    -o comparison.html
```

---

## Tips

1. **Warm-up Period**: Always use `-i` to ignore the first 1-2 seconds of data to exclude system warm-up effects.

2. **Timeline Plots**: Use a shorter `-m` value (1-2 seconds) for timeline plots to keep visualizations manageable.

3. **Summary Plots**: Use a longer `-m` value (10+ seconds) for CCDF and statistical plots to get meaningful percentile data.

4. **mMIMO Tests**: Remember to use the `-e` flag when analyzing mMIMO test configurations.

5. **Pre-parsing**: Pre-parse logs once with `cicd_parse.py`, then run multiple analysis scripts on the parsed data to save time.

---

## CUPTI Kernel Analysis (CUPTI setting in phase 4)

The `cupti_parse.py` script analyzes GPU kernel execution data from CUPTI instrumentation in PHY logs. It provides:

- **Interactive statistics table** with kernel duration percentiles per slot
- **GPU kernel timeline** showing kernel execution over time
- **CPU timeline** (optional) for correlating CPU and GPU activity
- **T0 coverage plots** (optional) for validating data completeness
- **CSV export** for further analysis

### Step 1: Enable CUPTI Tracing

CUPTI tracing must be enabled when configuring the test. Add the `CUPTI` modifier to your test case string:

```bash
cd testBenches/phase4_test_scripts

# Add CUPTI modifier to enable GPU kernel tracing
./parse_test_config_params.sh F08_6C_79_MODCOMP_STT480000_EH_CUPTI_1P CG1_R750 test_params.sh
source test_params.sh
```

This passes `--cupti` to `test_config.sh`, which sets `cuphydriver_config.cupti_enable_tracing = 1` in the CUPHY configuration.

### Step 2: Run Phase 4 Test

Run the Phase 4 test as usual. The `phy.log` will now contain `[CUPHY.CUPTI]` messages with GPU kernel execution data.

### Step 3: Analyze GPU Kernel Performance

```bash
# Basic analysis with interactive HTML output
python3 ../aerial_postproc/scripts/analysis/cupti_parse.py phy.log \
    -m 10 -i 1 \
    -o kernel_analysis.html

# Enable timeline plots for detailed visualization
python3 ../aerial_postproc/scripts/analysis/cupti_parse.py phy.log \
    -m 2 -i 1 \
    -t \
    -o kernel_timeline.html

# Export kernel statistics to CSV (99th percentile duration per slot)
python3 ../aerial_postproc/scripts/analysis/cupti_parse.py phy.log \
    -m 10 -i 1 \
    -c kernel_stats.csv -p 0.99

# Filter to specific kernels (partial name match)
python3 ../aerial_postproc/scripts/analysis/cupti_parse.py phy.log \
    -m 10 -i 1 \
    -f ldpc pusch \
    -o ldpc_pusch_analysis.html
```

> **Note:** CUPTI tracing adds overhead to test execution. Only enable it when GPU kernel profiling is needed.

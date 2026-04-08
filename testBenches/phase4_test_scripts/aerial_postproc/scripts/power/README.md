# Power Collection Utilities

This directory contains utilities for collecting and visualizing power consumption data from servers with NVIDIA GPUs.

## Overview

The power collection utilities gather power measurements from multiple sources:

- **GPU Power**: Data collected via `nvidia-smi` including power draw, temperatures, clocks, and utilization
- **Module Power**: Grace Hopper module power data from sysfs (CPU, SysIO, and total module power) - available on GH systems
- **PDU Power**: External PDU (Power Distribution Unit) measurements for total system power consumption

This data is useful for:
- Monitoring power consumption during any workload or benchmark
- Validating power efficiency optimizations
- Correlating power usage with workload characteristics
- Identifying thermal throttling or power limit issues

> **Note**: These utilities are used within the Aerial project for power monitoring during DU (Distributed Unit) CICD runs, but can be applied to any server workload requiring power measurement.

## Files

| File | Description |
|------|-------------|
| `collect_power.py` | Main data collection script |
| `power_plot.py` | Visualization tool for collected power data |
| `pdu_utils.py` | Utility library for Raritan PDU communication |

## Prerequisites

### Python Dependencies

```bash
pip install pandas bokeh ntplib
```

> **Note**: These dependencies are also included in the top-level `aerial_postproc/requirements.txt`.

### For Raritan PDU Support

The Raritan Python client bindings must be installed. Contact your system administrator for installation instructions.

### For CyberPower/TrippLite PDU Support

- `snmpget` command-line tool must be available
- Network access to the PDU

## Usage

### collect_power.py - Data Collection

Collects power measurements at regular intervals and writes them to a CSV file.

#### Basic Usage

```bash
# Collect GPU and module power only (no PDU)
python collect_power.py power_data.csv -d 300 --no-pdu

# Collect with verbose output for debugging
python collect_power.py power_data.csv -d 300 --no-pdu -v

# Collect data for 10 minutes with 0.5 second sampling period
python collect_power.py power_data.csv -d 600 -p 0.5 --no-pdu
```

#### PDU Configuration Examples

**Raritan PDU:**
```bash
python collect_power.py power_data.csv \
    -d 300 \
    -t raritan \
    -i 10.112.210.187 \
    -o 30,32 \
    --pdu-username admin \
    --pdu-password secret
```

**TrippLite PDU:**
```bash
python collect_power.py power_data.csv \
    -d 300 \
    -t tripplite \
    -i 192.168.1.100 \
    -o 1,2
```

**CyberPower PDU:**
```bash
python collect_power.py power_data.csv \
    -d 300 \
    -t cyberpower \
    -i 192.168.1.101 \
    -o 1,2,3,4
```

#### Collect CPU Clock Data

```bash
# Enable CPU clock frequency collection
python collect_power.py power_data.csv -d 300 -c cpu_clocks.csv
```

#### Full Command Reference

```
usage: collect_power.py [-h] [-c CPU_CLOCK_CSV] [-d DURATION] [-p PERIOD]
                        [-i PDU_IP] [-o PDU_OUTLETS] [-t PDU_TYPE] [-v]
                        [--no-pdu] [--pdu-username PDU_USERNAME]
                        [--pdu-password PDU_PASSWORD]
                        output_csv

Arguments:
  output_csv              Output csv containing raw power measurements

Options:
  -c, --cpu_clock_csv     Enable CPU clock collection and write to this csv
  -d, --duration          Duration in seconds to collect power data (default: 350)
  -p, --period            Approximate period between measurements (default: 1)
  -i, --pdu_ip            PDU IP or hostname
  -o, --pdu_outlets       Comma-separated list of PDU outlets
  -t, --pdu_type          PDU type: cyberpower, tripplite, or raritan
  -v, --verbose           Enable debug output
  --no-pdu                Skip PDU data collection
  --pdu-username          Username for Raritan PDU authentication
  --pdu-password          Password for Raritan PDU authentication
```

---

### power_plot.py - Visualization

Generates interactive HTML plots from collected power data using Bokeh.

#### Basic Usage

```bash
# Generate plots and open in browser
python power_plot.py power_data.csv

# Save plots to a specific HTML file
python power_plot.py power_data.csv -o power_report.html
```

#### With Power Summary Statistics

Use the GPU threshold option to automatically detect the active workload period and calculate power statistics:

```bash
# Calculate statistics when GPU power exceeds 200W
python power_plot.py power_data.csv -g 200

# Calculate statistics and save summary to CSV
python power_plot.py power_data.csv -g 200 -s power_summary.csv -o power_report.html

# Ignore first 10 seconds after workload starts (warm-up period)
python power_plot.py power_data.csv -g 200 -i 10 -o power_report.html
```

#### Full Command Reference

```
usage: power_plot.py [-h] [-g GPU_THRESHOLD] [-i IGNORE_DURATION]
                     [-s SUMMARY_CSV] [-o OUT_FILENAME]
                     [-e EXTERNAL_OVERLAY [EXTERNAL_OVERLAY ...]]
                     power_csv

Arguments:
  power_csv               Input power csv from collect_power.py

Options:
  -g, --gpu_threshold     GPU power threshold (W) to detect active period
  -i, --ignore_duration   Seconds to ignore after activity starts (default: 0)
  -s, --summary_csv       Output CSV summarizing power statistics
  -o, --out_filename      Output HTML filename (opens in browser if not specified)
  -e, --external_overlay  External perflab power datasets to overlay
```

## Output Data Format

### Power CSV Columns

The output CSV from `collect_power.py` contains the following column prefixes:

| Prefix | Source | Description |
|--------|--------|-------------|
| `[GPU]` | nvidia-smi | GPU power, temperature, clocks, utilization |
| `[MOD]` | sysfs | Grace Hopper module power (CPU, SysIO, total) |
| `[PDU]` | PDU | External PDU measurements per outlet and totals |

Example columns:
- `[GPU]power.draw.average` - GPU power draw in Watts
- `[GPU]temperature.gpu` - GPU temperature in Celsius
- `[MOD]Module Power Socket 0 Power` - Total module power in Watts
- `[PDU]total_power` - Total PDU power in Watts
- `[PDU]outlet_1_power` - Individual outlet power in Watts
- `collection_end_tir` - Time in run (seconds from start)

### Generated Plots

The HTML output from `power_plot.py` includes:

1. **Power Summary Table** (when using `-g` threshold): Min, median, average, and max power for each source
2. **Power vs Time Plot**: All power sources over time with interactive legend
3. **Temperature vs Time Plot**: GPU and memory temperatures
4. **Clock vs Time Plot**: GPU clock frequencies

## Example CICD Workflow

```bash
#!/bin/bash

# Start power collection in background before test
python collect_power.py power_data.csv \
    -d 600 \
    -t raritan \
    -i $PDU_IP \
    -o $PDU_OUTLETS \
    --pdu-username $PDU_USER \
    --pdu-password $PDU_PASS &
POWER_PID=$!

# Run the actual test
./run_test.sh

# Wait for power collection to complete
wait $POWER_PID

# Generate power report
python power_plot.py power_data.csv \
    -g 200 \
    -s power_summary.csv \
    -o power_report.html
```

## Troubleshooting

### "Raritan RPC module not found"

Install the Raritan Python client bindings. These are typically available from your organization's package repository.

### "MIB file not found" (CyberPower/TrippLite)

The script will automatically download required MIB files to `/tmp/`. Ensure network access to the MIB download URLs.

### GPU data shows 'N/A'

Some GPU metrics may not be available on all hardware configurations. The script will log warnings and use 0.0 as a fallback value.

### PDU connection timeout

Verify network connectivity to the PDU and correct credentials. For Raritan PDUs, ensure HTTPS access is enabled.

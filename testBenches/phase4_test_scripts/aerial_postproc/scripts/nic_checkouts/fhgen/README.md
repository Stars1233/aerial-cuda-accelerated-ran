# FH Generator Post-Processing Tools

## Overview

FH Generator is a test bench for simulating ORAN-compliant fronthaul traffic. This folder contains post-processing scripts for analyzing and thresholding FH Generator test results.

In the FH Generator application, there are two entities running:
- **DU (Distributed Unit)**: Transmitter of downlink traffic
- **RU (Radio Unit)**: Receiver of downlink traffic, transmitter of uplink traffic

## ORAN Traffic Fundamentals

### Timing Structure

| Unit | Duration | Description |
|------|----------|-------------|
| Slot | 500 us | Time slice containing 14 symbols |
| Symbol | ~35.7 us | Subdivisions within a slot |
| Frame | 10 ms | 20 slots |

### Traffic Types

- **C-Plane (Control Plane)**: Contains metadata for the DU to communicate U-plane expectations to the RU
- **U-Plane (User Plane)**: Contains actual user data

C-Plane traffic is much smaller as a percentage of total traffic compared to U-Plane.

### Traffic Directions

| Direction | Description |
|-----------|-------------|
| Downlink (DL) | Traffic from DU to RU |
| Uplink (UL) | Traffic from RU to DU |

### Example: 59C Pattern

In the 59C traffic pattern (defined over 4 frames / 80 slots):
- **Downlink slots**: 0-3, 6-13, 16-19 (traffic transmitted DU → RU)
- **Uplink slots**: 4-5, 14-15 (traffic transmitted RU → DU)

Each cell generates approximately **8 Gbps peak throughput**. A 20-cell test case produces 20x the traffic of a single cell.

## Timing Configuration

The following YAML configuration fields control traffic timing:

```yaml
tcp_adv_dl_ns: 125000
t1a_max_cp_ul_ns: 336000
t1a_max_up_ns: 345000
ta4_min_ns: 50000
ta4_max_ns: 331000
window_end_ns: 51000
ulu_tx_time_advance_ns: 280000
```

### Downlink Direction (DU → RU)

| Traffic Type | Destined Time |
|--------------|---------------|
| DL U-Plane | T0 - t1a_max_up_ns = T0 - 345000 ns |
| DL C-Plane | T0 - (t1a_max_up_ns + tcp_adv_dl_ns) = T0 - 470000 ns |
| UL C-Plane | T0 - t1a_max_cp_ul_ns = T0 - 336000 ns |

Packets received between (T + 0, T + window_end_ns) of their destined time are considered **on-time**.

### Uplink Direction (RU → DU)

| Traffic Type | Destined Time |
|--------------|---------------|
| UL U-Plane | T0 + ta4_max_ns = T0 + 331000 ns |

RU TX timestamp is set to T0 + ulu_tx_time_advance_ns = T0 + 280000 ns.

### Tuning Parameters

- **Enlarge passing window**: Increase `window_end_ns`
- **Adjust pass percentage**: Modify the following fields:

```yaml
dlc_ontime_pass_percentage: 99.99
dlu_ontime_pass_percentage: 99.99
ulc_ontime_pass_percentage: 99.99
ulu_ontime_pass_percentage: 99.99
```

## Example 22C 4TR Bi-Directional Test

### 1. Build cuBB

```bash
git clone ssh://git@gitlab-master.nvidia.com:12051/gputelecom/aerial_sdk.git --recurse-submodules cuBB_$(date +"%m%d")
cd cuBB_$(date +"%m%d")
export cuBB_SDK=$(pwd)
git lfs pull

# Run docker container
./cuPHY-CP/container/run_aerial.sh

# Build
./testBenches/phase4_test_scripts/build_aerial_sdk.sh
```

### 2. Generate FH Generator Config

```bash
python3 ./cuPHY-CP/aerial-fh-driver/app/fh_generator/scripts/fhgen_from_pattern.py \
    -c 22 \
    --du_nic_addrs "0000:01:00.0" \
    --ru_nic_addrs "0000:ca:00.0" \
    -i ./cuPHY-CP/traffic_pattern/POC2_59c.json \
    --sfn_slot_sync_ru "10.112.208.161" \
    --sfn_slot_sync_du "10.112.208.176" \
    --test_slots 120000
```

### 3. Run FH Generator

**On DU:**
```bash
sudo -E ./build.$(uname -m)/cuPHY-CP/aerial-fh-driver/app/fh_generator/fh_generator \
    ./cuPHY-CP/traffic_pattern/POC2_59c_22C_testslots_120000.yaml
```

**On RU:**
```bash
sudo -E ./build.$(uname -m)/cuPHY-CP/aerial-fh-driver/app/fh_generator/fh_generator \
    ./cuPHY-CP/traffic_pattern/POC2_59c_22C_testslots_120000.yaml -r
```

### 4. Collect Results

**On DU:**
```bash
mkdir -p ./results/
tar -C/tmp -cvzf ./results/fhgen_du_POC2_59c_22C_testslots_120000.tgz \
    fhgen_du_POC2_59c_22C_testslots_120000.log
```

**On RU:**
```bash
tar -C/tmp -cvzf ./results/fhgen_ru_POC2_59c_22C_testslots_120000.tgz \
    fhgen_ru_POC2_59c_22C_testslots_120000.log
```

### 5. Post-Processing

```bash
# Set up paths
export IO_PATH=./results/$(date +"%Y%m%d_%H%M%S")
mkdir -p $IO_PATH

# Extract logs
tar -xvf $IO_PATH/fhgen_du_POC2_59c_22C_testslots_120000.tgz -C $IO_PATH
tar -xvf $IO_PATH/fhgen_ru_POC2_59c_22C_testslots_120000.tgz -C $IO_PATH

# Parse data (requires aerial_postproc package)
touch $IO_PATH/blank.log
python3 scripts/cicd_parse.py \
    $IO_PATH/fhgen_du_POC2_59c_22C_testslots_120000.log \
    $IO_PATH/blank.log \
    $IO_PATH/fhgen_ru_POC2_59c_22C_testslots_120000.log \
    -f "latencysummary" \
    -o $IO_PATH/parsed/

# Generate visualizations
python3 scripts/nic_checkouts/fhgen/fhgen_compare.py \
    $IO_PATH/parsed/ \
    -t scripts/nic_checkouts/fhgen/fhgen_thesholds_22C_59c.csv \
    -m 60 -i 1 \
    -l "FHGen 22C 59c" \
    -o fhgen_compare_22C_59c.html

# Run thresholding
python3 scripts/nic_checkouts/fhgen/fhgen_threshold.py \
    $IO_PATH/parsed/ \
    -s $IO_PATH/stats.csv \
    -t scripts/nic_checkouts/fhgen/fhgen_thesholds_22C_59c.csv \
    -m 60 -i 1
```

### 6. Interpret Results

Successful output example:

```
Writing stats_csv /path/to/results/stats.csv...
Performing thresholding using scripts/nic_checkouts/fhgen/fhgen_thesholds_22C_59c.csv...
Thresholding for ulu_rx_start_q99.5 succeeded
Thresholding for ulu_rx_end_q99.5 succeeded
Thresholding for ulc_rx_start_q99.5 succeeded
Thresholding for ulc_rx_end_q99.5 succeeded
Thresholding for dlc_rx_start_q99.5 succeeded
Thresholding for dlc_rx_end_q99.5 succeeded
Thresholding for dlu_rx_start_q99.5 succeeded
Thresholding for dlu_rx_end_q99.5 succeeded
SUCCESS :: All results met input thresholds across all slot/symbol pairs
```

## Scripts in This Folder

### fhgen_compare.py

Produces visualization plots (box-whisker/CCDF) to analyze FH Generator timing data.

**Usage:**

```bash
python3 fhgen_compare.py <input_data_list> [options]
```

**Arguments:**

| Argument | Description |
|----------|-------------|
| `input_data_list` | List of input folders or rotating phy/testmac/ru logs |
| `-m, --max_duration` | Maximum run duration to process (default: 999999999.0) |
| `-i, --ignore_duration` | Seconds at beginning to ignore (default: 0.0) |
| `-n, --num_proc` | Number of parsing processes (default: 8) |
| `-w, --num_worst_slots` | Number of worst slots for CCDF (default: 1) |
| `-t, --input_threshold_csv` | Enable thresholding with specified CSV |
| `-l, --labels` | Name labels for each data set |
| `-o, --out_filename` | Output filename (default: result.html) |

### fhgen_threshold.py

Performs thresholding analysis on parsed FH Generator data.

**Usage:**

```bash
python3 fhgen_threshold.py <input_data_list> [options]
```

**Arguments:**

| Argument | Description |
|----------|-------------|
| `input_data_list` | List of input folders or rotating phy/testmac/ru logs |
| `-m, --max_duration` | Maximum run duration to process |
| `-i, --ignore_duration` | Seconds at beginning to ignore |
| `-n, --num_proc` | Number of parsing processes (default: 8) |
| `-t, --input_threshold_csv` | Threshold CSV for pass/fail determination |
| `-g, --output_threshold_csv` | Generate threshold CSV from input data |
| `-s, --stats_csv` | Output statistics CSV from mean results |

### fhgen_thesholds_22C_59c.csv

Pre-defined threshold values for 22-cell 59C pattern testing. Contains 99.5th percentile thresholds for:
- `ulu_rx_start_q99.5`, `ulu_rx_end_q99.5`
- `ulc_rx_start_q99.5`, `ulc_rx_end_q99.5`
- `dlc_rx_start_q99.5`, `dlc_rx_end_q99.5`
- `dlu_rx_start_q99.5`, `dlu_rx_end_q99.5`

## References

- [O-RAN SC Transport Layer and Fronthaul Protocol Implementation](https://docs.o-ran-sc.org/projects/o-ran-sc-o-du-phy/en/latest/Transport-Layer-and-ORAN-Fronthaul-Protocol-Implementation_fh.html#)

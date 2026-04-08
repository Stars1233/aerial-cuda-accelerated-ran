# netperftest Post-Processing Tools

## Overview

netperftest is a DPDK-only test used for bi-directional NIC checkouts. This folder contains post-processing scripts for analyzing and thresholding netperftest results.

## Test Setup

The typical test setup involves:
- **DU (Distributed Unit)**: e.g., GH200 server with BF3
- **RU (Radio Unit)**: e.g., R750 server with BF3

Both servers must be reachable on the same network and use NATS for synchronization.

## Example Bi-Directional Test

### Notes

- Power cycle servers between tests for more consistent performance
- The `-i` (ignore duration) parameter is recommended to skip the initial warmup period
- Results from `test_ru` represent **Downlink (DL)** traffic (DU → RU)
- Results from `test_du` represent **Uplink (UL)** traffic (RU → DU)

### 1. Container Deployment (on both DU and RU)

```bash
# Install prerequisites
sudo apt install pip
sudo pip install hpccm

# Enable nvidia-peermem (DU only)
sudo modprobe nvidia-peermem

# Clone netperftest (DU only, or use shared NFS)
export NPT_REPO=/path/to/repo (currently Nvidia internal)
git clone $NPT_REPO netperftest
cd netperftest

# Run container
./container/run.sh

# Build netperftest (inside container, on both DU and RU)
mkdir build.$(uname -m); cd build.$(uname -m)
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=90 ..
make -j $(nproc --all)
```

### 2. Execute Test

Requires 3 terminals on DU and 3 terminals on RU.

**Terminal 3, DU - Start NATS server:**
```bash
nats-server
```

**Terminal 1 & 2, DU - Start MPS server:**
```bash
export CUDA_MPS_PIPE_DIRECTORY=/var
export CUDA_MPS_LOG_DIRECTORY=/var
sudo -E echo quit | sudo -E nvidia-cuda-mps-control
export CUDA_MPS_PIPE_DIRECTORY=/var
export CUDA_MPS_LOG_DIRECTORY=/var
sudo -E nvidia-cuda-mps-control -d
sudo -E echo start_server -uid 0 | sudo -E nvidia-cuda-mps-control
```

**Terminal 1, DU - Run RX:**
```bash
sudo -E ./test_dpdk_tx -n <DU_HOSTNAME>:4222 -t du -p /tmp/test_du
```

**Terminal 2, DU - Run TX:**
```bash
cd build.$(uname -m)
sudo -E ./test_dpdk_rx -n <DU_HOSTNAME>:4222 -t du -p /tmp/test_du
```

**Terminal 1, RU - Run RX:**
```bash
sudo -E ./test_dpdk_rx -n <DU_HOSTNAME>:4222 -t ru -p /tmp/test_ru
```

**Terminal 2, RU - Run TX:**
```bash
cd build.$(uname -m)
sudo -E ./test_dpdk_tx -n <DU_HOSTNAME>:4222 -t ru -p /tmp/test_ru
```

**Terminal 3, RU - Execute test via NATS:**
```bash
cd scripts
python3 ./nats_run_test.py -c ../aerial_nic_checkouts/config_smc_r750_22C_59c_BFP9_1P.yaml -n <DU_HOSTNAME>:4222
```

### 3. Collect Output Files

**On DU:**
```bash
mkdir -p ../results/
tar -C/tmp -cvzf ../results/test_du.tgz test_du
```

**On RU:**
```bash
mkdir -p ../results/
tar -C/tmp -cvzf ../results/test_ru.tgz test_ru
```

**Consolidate results (on DU):**
```bash
export RES_FOLDER=../results/$(date +"%Y%m%d_%H%M%S")
mkdir -p $RES_FOLDER

# If using separate machines:
scp $USER@$RU_SERVER:$NETPERFTEST_PATH/results/test_ru.tgz $RES_FOLDER
mv ../results/test_du.tgz $RES_FOLDER

# If using shared NFS:
mv ../results/test_du.tgz $RES_FOLDER
mv ../results/test_ru.tgz $RES_FOLDER

# Extract
tar -C $RES_FOLDER -xvzf $RES_FOLDER/test_du.tgz
tar -C $RES_FOLDER -xvzf $RES_FOLDER/test_ru.tgz
```

### 4. Post-Processing

```bash
export INPUT_FOLDER=$RES_FOLDER
export IGNORE_DURATION=1

# Process RX logs to parsed CSV
pip3 install pandas
python3 netperftest_rxlogs2csv.py $INPUT_FOLDER/test_ru/log_rxq.csv -o $INPUT_FOLDER/test_ru/parsed.csv
python3 netperftest_rxlogs2csv.py $INPUT_FOLDER/test_du/log_rxq.csv -o $INPUT_FOLDER/test_du/parsed.csv

# Save summary statistics
python3 netperftest_threshold.py $INPUT_FOLDER/test_ru/parsed.csv -i $IGNORE_DURATION -s $INPUT_FOLDER/test_ru/stats.csv
python3 netperftest_threshold.py $INPUT_FOLDER/test_du/parsed.csv -i $IGNORE_DURATION -s $INPUT_FOLDER/test_du/stats.csv

# Generate analysis plots
pip3 install bokeh
python3 netperftest_compare.py $INPUT_FOLDER/test_ru/parsed.csv \
    -l "22C 59c BFP9 1P NIC Checkout (DL)" \
    -i $IGNORE_DURATION \
    -t netperftest_thresholds_dl_smc_r750_22C_59c_BFP9_1P_240522.csv \
    -o $INPUT_FOLDER/test_ru/dl_compare.html

python3 netperftest_compare.py $INPUT_FOLDER/test_du/parsed.csv \
    -l "22C 59c BFP9 1P NIC Checkout (UL)" \
    -i $IGNORE_DURATION \
    -t netperftest_thresholds_ul_smc_r750_22C_59c_BFP9_1P_240522.csv \
    -o $INPUT_FOLDER/test_du/ul_compare.html

# Run thresholding
python3 netperftest_threshold.py $INPUT_FOLDER/test_ru/parsed.csv \
    -i $IGNORE_DURATION \
    -t netperftest_thresholds_dl_smc_r750_22C_59c_BFP9_1P_240522.csv

python3 netperftest_threshold.py $INPUT_FOLDER/test_du/parsed.csv \
    -i $IGNORE_DURATION \
    -t netperftest_thresholds_ul_smc_r750_22C_59c_BFP9_1P_240522.csv

# Archive final results
tar -cvzf final_results.tgz $INPUT_FOLDER
```

### 5. Interpret Results

Successful thresholding output example:

```
THRESHOLD SUCCESS for key rx_start_q90.00, value=1.750000, threshold=1.830143
THRESHOLD SUCCESS for key rx_start_q99.00, value=1.980000, threshold=2.055286
THRESHOLD SUCCESS for key rx_start_q99.50, value=2.050000, threshold=2.159286
...
Returning success
```

Failed thresholding output example:

```
THRESHOLD FAILURE for key rx_end_q99.50, value=31.500000, threshold=30.735887
Returning failure
```

## Scripts in This Folder

### netperftest_rxlogs2csv.py

Converts netperftest RX logs to a summary CSV format for analysis and thresholding.

**Usage:**

```bash
python3 netperftest_rxlogs2csv.py <rx_log> [options]
```

**Arguments:**

| Argument | Description |
|----------|-------------|
| `rx_log` | RX log file from netperftest |
| `-o, --output_csv` | Output CSV file (default: result.csv) |
| `-p, --pkt_len` | Packet length used in netperftest (default: 1518) |

**Output Fields:**

| Field | Description |
|-------|-------------|
| `burst_idx` | Burst identifier |
| `desired_tir` | Desired time in run (seconds) |
| `rx_start_deadline` | First packet time relative to desired (usec) |
| `rx_end_deadline` | Last packet time relative to desired (usec) |
| `num_ques` | Number of TX queues |
| `num_packets` | Total packets in burst |
| `eth_frame_size` | Ethernet L1 frame size |
| `rx_duration` | Transfer duration (usec) |
| `estimated_burst_throughput_gbps` | Throughput including first packet latency |
| `estimated_sustained_throughput_gbps` | Throughput excluding first packet latency |

### netperftest_threshold.py

Performs pass/fail thresholding on netperftest results. Can also generate summary statistics and threshold files.

**Usage:**

```bash
python3 netperftest_threshold.py <input_csvs> [options]
```

**Arguments:**

| Argument | Description |
|----------|-------------|
| `input_csvs` | CSV file(s) parsed by rxlogs2csv |
| `-m, --max_duration` | Maximum run duration to process |
| `-i, --ignore_duration` | Seconds at beginning to ignore |
| `-t, --input_threshold_csv` | Threshold CSV for pass/fail (returns 0 on success, 1 on failure) |
| `-g, --output_threshold_csv` | Generate threshold CSV from input data |
| `-e, --easy_threshold` | Only use 99.5% threshold values |
| `-s, --stats_csv` | Generate stats CSV from input data |

**Thresholded Metrics:**

- `rx_start_q90.00`, `rx_start_q99.00`, `rx_start_q99.50`, `rx_start_q99.90`, `rx_start_q99.99`
- `rx_end_q90.00`, `rx_end_q99.00`, `rx_end_q99.50`, `rx_end_q99.90`, `rx_end_q99.99`

### netperftest_compare.py

Generates CCDF visualization plots for throughput and packet timing analysis.

**Usage:**

```bash
python3 netperftest_compare.py <input_csvs> [options]
```

**Arguments:**

| Argument | Description |
|----------|-------------|
| `input_csvs` | CSV file(s) parsed by rxlogs2csv |
| `-m, --max_duration` | Maximum run duration to process |
| `-i, --ignore_duration` | Seconds at beginning to ignore |
| `-o, --out_filename` | Output HTML file |
| `-l, --labels` | Labels for each dataset |
| `-t, --threshold_csv` | Overlay threshold data points on CCDFs |

**Generated Plots:**

- Estimated Sustained Throughput (CDF)
- First Packet Time Relative to Desired (CCDF)
- Last Packet Time Relative to Desired (CCDF)
- Transfer Duration (CCDF)

### netperftest_timeline_plot.py

Produces timeline plots showing throughput and latency metrics over time.

**Usage:**

```bash
python3 netperftest_timeline_plot.py <input_csv> [options]
```

**Arguments:**

| Argument | Description |
|----------|-------------|
| `input_csv` | CSV file parsed by rxlogs2csv |
| `-m, --max_duration` | Maximum run duration to process |
| `-i, --ignore_duration` | Seconds at beginning to ignore |
| `-t, --time_delta` | Time delta for computing statistics |
| `-o, --out_filename` | Output HTML file |

### netperftest_rxlogs_find_gaps.py

Identifies gaps in RX logs between first and last packet data.

**Usage:**

```bash
python3 netperftest_rxlogs_find_gaps.py <first_packet_rx_log> <last_packet_rx_log>
```

### netperftest_cqe2csv.py

Converts netperftest CQE (Completion Queue Entry) text logs to CSV format.

**Usage:**

```bash
python3 netperftest_cqe2csv.py <input_file> <output_file> [options]
```

**Arguments:**

| Argument | Description |
|----------|-------------|
| `input_file` | Input txt file with CQE logging |
| `output_file` | Output CSV file |
| `-l, --only_last_empw` | Only output last EMPW entry |
| `-r, --rte_file` | Write file with only RTE data |
| `-w, --wait_file` | Write file with only WAIT data |
| `-e, --empw_file` | Write file with only EMPW data |

### Threshold CSV Files

Pre-defined threshold values for 22-cell 59c BFP9 1P testing:

| File | Description |
|------|-------------|
| `netperftest_thresholds_dl_smc_r750_22C_59c_BFP9_1P_240522.csv` | Downlink thresholds (full) |
| `netperftest_thresholds_dl_smc_r750_22C_59c_BFP9_1P_240522_99.5_only.csv` | Downlink thresholds (99.5% only) |
| `netperftest_thresholds_ul_smc_r750_22C_59c_BFP9_1P_240522.csv` | Uplink thresholds (full) |
| `netperftest_thresholds_ul_smc_r750_22C_59c_BFP9_1P_240522_99.5_only.csv` | Uplink thresholds (99.5% only) |

## Optional: Custom DPDK Build

If an alternative DPDK version is required:

```bash
# Clone desired DPDK version
git clone ssh://git@gitlab-master.nvidia.com:12051/gputelecom-external/dpdk.org.git external/dpdk
cd external/dpdk
git checkout <desired_commit>
cd ../../

# Build container with custom DPDK
CONTAINER_VERSION_TAG=RELEASE ./container/build.sh
CONTAINER_VERSION_TAG=RELEASE ./container/run.sh

# Build and install DPDK inside container (on both DU and RU)
cd external/dpdk
sudo mkdir -p /opt/dpdk/build
sudo chown nvidia /opt/dpdk/build
meson setup /opt/dpdk/build
ninja -C /opt/dpdk/build
ninja -C /opt/dpdk/build install
sudo ldconfig
cd ../../
```

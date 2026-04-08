# 3GPP Channel Model Calibration Guide

This guide provides step-by-step instructions for calibrating the channel model against 3GPP TR 38.901 specifications.

**Last Updated**: January 2026

---

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Build Instructions](#build-instructions)
4. [Quick Start - Running All Calibrations](#quick-start---running-all-calibrations)
5. [UMa Calibration (6 GHz)](#uma-calibration-6-ghz)
6. [ISAC Calibration (UAV Sensing)](#isac-calibration-uav-sensing)
7. [Understanding Results](#understanding-results)
8. [Troubleshooting](#troubleshooting)

---

## Overview

The Aerial SDK supports multiple channel models with varying levels of CPU/GPU implementation and calibration. This guide focuses on calibrating system-level models (UMa and UMa-AV) against 3GPP reference data.

| Type | Channel | CPU version | GPU version | Calibration | Link |
|------|---------|-------------|-------------|-------------|------|
| Link | TDL (A/B/C) | No  | Yes | Yes | |
| Link | CDL (A/B/C) | No  | Yes | Yes | |
| System | UMa | Yes | Yes | Yes | [UMa Calibration](#uma-calibration-6-ghz) |
| System | UMi | Yes | Yes | No | |
| System | RMa | Yes | Yes | No | |
| System | UMa-AV (ISAC) | Yes | No | Done | [ISAC Calibration](#isac-uma-av-calibration) |

The calibration process validates the 3GPP TR 38.901 channel model implementation against 3GPP reference data through:

1. **Channel Generation**: Generate channel realizations (H5 files) using `sls_chan_ex`
2. **Statistical Analysis**: Extract metrics (coupling loss, delay spread, angle spreads, K-factor, etc.)
3. **CDF Comparison**: Compare CDFs with 3GPP reference curves

---

## Prerequisites

It's recommended to use Aerial container to run the tests. However, if a new test environment is preferrable, make sure to have the following dependencies.

### System Requirements
- CUDA-capable GPU (for GPU mode) or CPU
- Python 3.8+
- HDF5 library
- CMake 3.18+

### Python Dependencies
```bash
pip install numpy matplotlib h5py pyyaml scipy
```

---

## Build Instructions

Channel model examples can be built as part of the standard Aerial SDK build process. To build the complete Aerial SDK including channel model examples:

```bash
# Build full Aerial SDK
$cuBB_SDK/testBenches/phase4_test_scripts/build_aerial_sdk.sh
```
This script will automatically created a build folder `./build.aarch64` or `./build.x86_64` by detecting the CPU architecture.

Alternatively, you can manually configure the build options:

```bash
# Recommend running from the aerial_sdk root directory
cmake -B<build_dir> -GNinja -DCMAKE_TOOLCHAIN_FILE=cuPHY/cmake/toolchains/native
```

Once configured, you can build all testbench example targets (including channel models) using the `testbenches_examples` target:

```bash
# Run from the directory that contains <build_dir>
cmake --build <build_dir> --target testbenches_examples
```

To build the specific channel model target needed for calibration:

```bash
# Build SLS channel model example (used for UMa and UMa-AV calibration)
cmake --build <build_dir> --target sls_chan_ex
```

Usage instructions can be found by running:

```bash
<example_executable> -h
```

**Typical Executable Path**: `./build.aarch64/testBenches/chanModels/examples/sls_chan/sls_chan_ex`

---

## Quick Start - Running All Calibrations

For users who want to run both UMa and ISAC calibrations in one go, a convenience script is provided:

```bash
# Navigate to the util folder
cd $cuBB_SDK/testBenches/chanModels/util

# Run all calibrations (UMa Phase 1 & 2, ISAC Phase 1 & 2 for target and background)
./run_all_calibrations.sh
```

This script will execute:
- **UMa Phase 1**: Large-scale calibration (5 seeds by default: 0-4)
- **UMa Phase 2**: Full calibration with fast fading (5 seeds by default: 0-4)
- **ISAC Phase 1**: Target and background channel large-scale calibration (200 seeds by default: 0-199)
- **ISAC Phase 2**: Target and background channel full calibration (200 seeds by default: 0-199)

**Note on Seed Requirements**:
- **UMa calibration** uses 19 sites with 57 sectors (3 sectors/site) and 10 UEs per sector, totaling **570 UEs** per seed. This large number of UEs per seed provides sufficient statistical samples, requiring only **5 seeds** for good calibration results.
- **ISAC calibration** uses a single cell with 1 target, providing only **1 sample** per seed. Therefore, ISAC requires significantly more seeds (**200 seeds**) to achieve adequate statistical coverage.

**Options**:
- `--seed-start N --seed-end M`: Override default seed range for all calibrations
- `--uma-only`: Run only UMa calibrations (Phase 1 & 2)
- `--isac-only`: Run only ISAC calibrations (Phase 1 & 2 for target and background)

**Examples**:
```bash
# Run only UMa calibrations with default seeds (0-4)
./run_all_calibrations.sh --uma-only

# Run only ISAC calibrations with default seeds (0-199)
./run_all_calibrations.sh --isac-only

# Run all calibrations with custom seed range (quick test with 10 seeds)
./run_all_calibrations.sh --seed-start 0 --seed-end 9

# Run UMa only with custom seed range
./run_all_calibrations.sh --uma-only --seed-start 0 --seed-end 9
```

**Expected Results**:
- `uma_6ghz_phase1_results/`: UMa Phase 1 CDF plots and KS statistics
- `uma_6ghz_phase2_results/`: UMa Phase 2 CDF plots and KS statistics
- `isac_phase1_target_results/`: ISAC Phase 1 target channel results
- `isac_phase1_background_results/`: ISAC Phase 1 background channel results
- `isac_phase2_target_results/`: ISAC Phase 2 target channel results
- `isac_phase2_background_results/`: ISAC Phase 2 background channel results

**Note**: The full calibration process can take several hours depending on your hardware. For quicker validation, refer to the individual calibration sections below to run specific phases or reduce the number of seeds.

---



---
## UMa Calibration (6 GHz)

### Overview
Calibrates the traditional UMa channel model (BS-UE communication) against **3GPP TR 38.901 V18.0** at 6 GHz, using reference data from **R1-165974**.

The calibration follows the 3GPP methodology with two phases:

**Phase 1 (Large Scale Calibration)** - Section 7.8.1:
1. Coupling loss – serving cell (based on LOS pathloss)
2. Geometry (based on LOS pathloss) with and without white noise

**Phase 2 (Full Calibration)** - Section 7.8.2:
1. Coupling loss – serving cell
2. Wideband SIR before receiver without noise
3. CDF of Delay Spread and Angle Spread (ASD, ASA, ZSD, ZSA) from the serving cell (according to circular angle spread definition of TR 25.996)
4. CDF of PRB singular values (serving cell) at t=0 in 10×log₁₀ scale:
   - Largest (1st) singular value
   - Smallest (2nd) singular value  
   - Ratio between largest and smallest singular values

### Configuration Files

**Phase 1 Config**: `testBenches/chanModels/config/statistic_channel_config_phase1.yaml`

**Phase 2 Config**: `testBenches/chanModels/config/statistic_channel_config_phase2.yaml`

**Reference Data**: `testBenches/chanModels/util/3gpp_calibration_phase1.json` (Phase 1) and `testBenches/chanModels/util/3gpp_calibration_phase2.json` (Phase 2)

### Running UMa Calibration

The UMa calibration topology uses:
- **19 sites** with **3 sectors per site** = **57 sectors total**
- **10 UEs per sector** = **570 UEs total**
- **Total links**: 57 sectors × 570 UEs = 32,490 links

This large topology provides 570 statistical samples per seed. Therefore, UMa calibration uses a **small number of seeds** (default: **5 seeds, 0-4**) to achieve sufficient statistical coverage while keeping simulation time reasonable.

**Note**: All calibration scripts are located in `testBenches/chanModels/util/` and can be run from the util directory.

**Two Options for Running Calibrations**:

1. **Batch Mode** - Run multiple calibrations together:
   ```bash
   cd $cuBB_SDK/testBenches/chanModels/util
   
   # Run all calibrations (UMa + ISAC)
   ./run_all_calibrations.sh
   
   # Or run UMa only
   ./run_all_calibrations.sh --uma-only
   ```

2. **Individual Mode** - Run specific calibrations using `run_sls_chan_multiseed.sh` (examples below)

#### Phase 1: Large Scale Calibration (Default: 5 seeds, 0-4)

**YAML Config**: `testBenches/chanModels/config/statistic_channel_config_phase1.yaml` 

```bash
cd $cuBB_SDK/testBenches/chanModels/util

./run_sls_chan_multiseed.sh \
    --seed-start 0 \
    --seed-end 4 \
    --config ../config/statistic_channel_config_phase1.yaml \
    --dataset uma_6ghz \
    --reference-json 3gpp_calibration_phase1.json \
    --phase 1 \
    --output-dir uma_6ghz_phase1_results
```

#### Phase 2: Full Calibration (Default: 5 seeds, 0-4)

**YAML Config**: `testBenches/chanModels/config/statistic_channel_config_phase2.yaml` 

```bash
cd $cuBB_SDK/testBenches/chanModels/util

./run_sls_chan_multiseed.sh \
    --seed-start 0 \
    --seed-end 4 \
    --config ../config/statistic_channel_config_phase2.yaml \
    --dataset uma_6ghz \
    --reference-json 3gpp_calibration_phase2.json \
    --phase 2 \
    --output-dir uma_6ghz_phase2_results
```

### Expected Outputs

**Generated Files**:
- `slsChanData_19sites_570uts_TTI*_uma_6ghz_seed*.h5` - Channel realizations (H5 format)
- `uma_6ghz_phase1_results/` or `uma_6ghz_phase2_results/` - Analysis results and plots

**Phase 1 CDF Plots** (Large Scale Calibration):
- `cdf_coupling_loss_rsrp.png` - Coupling loss based on LOS pathloss
- `cdf_geometry.png` - Geometry (with/without white noise)

**Phase 2 CDF Plots** (Full Calibration):
- `cdf_coupling_loss_rsrp.png` - Coupling loss from serving cell
- `cdf_wideband_sir.png` - Wideband SIR before receiver (without noise)
- `cdf_delay_spread.png` - RMS delay spread
- `cdf_asd.png`, `cdf_asa.png` - Azimuth angle spreads (departure/arrival, per TR 25.996)
- `cdf_zsd.png`, `cdf_zsa.png` - Zenith angle spreads (departure/arrival, per TR 25.996)
- `cdf_prb_sv1.png` - Largest (1st) PRB singular value at t=0 (10×log₁₀ scale)
- `cdf_prb_sv2.png` - Smallest (2nd) PRB singular value at t=0 (10×log₁₀ scale)
- `cdf_prb_sv_ratio.png` - Ratio of largest to smallest PRB singular value (10×log₁₀ scale)

In each of the figures, we have three data
* blue solid curve: Aerial channel simulation
* red dotted curve: Average of 3GPP reference data
* grey shade: range of 3GPP reference data across different companies
  
---

## ISAC Calibration (UAV Sensing)

### Overview
Calibrates the ISAC (Integrated Sensing and Communication) channel model for UAV sensing targets against **3GPP TR 38.901 V19.0 Section 7.9**, using reference data from **R1-2509126**. Currently only BS monostatic sensing is calibrated.

**Two Channel Types**:
1. **Target Channel**: BS → UAV Target → BS (monostatic sensing)
2. **Background Channel**: BS → 3 reference points → BS (monostatic sensing)

**ISAC Topology and Seed Requirements**:
- ISAC calibration uses a **single cell** with **1 target** (or 3 reference points for background channel)
- This provides only **1 sample per seed** (compared to 570 samples per seed in UMa)
- Therefore, ISAC requires significantly **more seeds** (default: **200 seeds, 0-199**) to achieve adequate statistical coverage for CDF comparison with 3GPP reference data

**Phase 1 (Large Scale Calibration)** - Section 7.9.6.1:
- **Target**: Coupling loss only (monostatic path loss + shadow fading)
- **Background**: Coupling loss only (monostatic path loss + shadow fading)

**Phase 2 (Full Calibration)** - Section 7.9.6.2:
- **Target**: Coupling loss from CIR (includes RCS-scaled fast fading)
- **Background**: Full channel metrics (coupling loss, delay spread, angle spreads from CIR)

---

### ISAC Phase 1 Calibration

**Reference**: `testBenches/chanModels/util/3gpp_calibration_isac_uav_phase1.json`

#### Phase 1 Target Channel

**YAML Config**: `testBenches/chanModels/config/statistic_channel_config_isac_phase1.yaml`

**Run Command**:
```bash
cd $cuBB_SDK/testBenches/chanModels/util

# Full calibration (default: 200 seeds, 0-199)
./run_sls_chan_multiseed.sh \
    --seed-start 0 \
    --seed-end 199 \
    --config ../config/statistic_channel_config_isac_phase1.yaml \
    --dataset isac_uav_phase1_target \
    --reference-json 3gpp_calibration_isac_uav_phase1.json \
    --phase 1 \
    --isac-channel target \
    --output-dir isac_phase1_target_results

# Quick test (10 seeds)
./run_sls_chan_multiseed.sh \
    --seed-start 0 \
    --seed-end 9 \
    --config ../config/statistic_channel_config_isac_phase1.yaml \
    --dataset isac_uav_phase1_target \
    --reference-json 3gpp_calibration_isac_uav_phase1.json \
    --phase 1 \
    --isac-channel target \
    --output-dir isac_phase1_target_results
```

**Phase 1 Target Metrics**:
- Coupling Loss only (bistatic path loss + shadow fading for BS→UAV→BS)

**Expected Output**:
- `cdf_coupling_loss_rsrp.png` - Coupling loss for target channel

---

#### Phase 1 Background Channel

**YAML Config**: `testBenches/chanModels/config/statistic_channel_config_isac_phase1_background.yaml` The difference from target channel calibration is the `isac_disable_background` and `isac_disable_target` to only include background channel.

```
  isac_disable_background: 0     # 0=combined, 1=target-only (set to 1 for separate target analysis)
  isac_disable_target: 1         # 0=include target, 1=background-only (set to 1 for separate background analysis)
``` 

**Run Command**:
```bash
cd $cuBB_SDK/testBenches/chanModels/util

./run_sls_chan_multiseed.sh \
    --seed-start 0 \
    --seed-end 199 \
    --config ../config/statistic_channel_config_isac_phase1_background.yaml \
    --dataset isac_uav_phase1_background \
    --reference-json 3gpp_calibration_isac_uav_phase1.json \
    --phase 1 \
    --isac-channel background \
    --output-dir isac_phase1_background_results
```

**Phase 1 Background Metrics**:
- Coupling Loss (BS → 3 Reference Points → BS)
  
**Expected Outputs**:
- `cdf_coupling_loss_rsrp.png` - Coupling loss

---

### ISAC Phase 2 Calibration

#### Phase 2 Target Channel

**Config**: `testBenches/chanModels/config/statistic_channel_config_isac_phase2.yaml`

**Reference**: `testBenches/chanModels/util/3gpp_calibration_isac_uav_phase2.json`

**Run Command**:
```bash
cd $cuBB_SDK/testBenches/chanModels/util

# Full calibration (default: 200 seeds, 0-199)
./run_sls_chan_multiseed.sh \
    --seed-start 0 \
    --seed-end 199 \
    --config ../config/statistic_channel_config_isac_phase2.yaml \
    --dataset isac_uav_phase2_target \
    --reference-json 3gpp_calibration_isac_uav_phase2.json \
    --phase 2 \
    --isac-channel target \
    --output-dir isac_phase2_target_results

# Quick test (10 seeds)
./run_sls_chan_multiseed.sh \
    --seed-start 0 \
    --seed-end 9 \
    --config ../config/statistic_channel_config_isac_phase2.yaml \
    --dataset isac_uav_phase2_target \
    --reference-json 3gpp_calibration_isac_uav_phase2.json \
    --phase 2 \
    --isac-channel target \
    --output-dir isac_phase2_target_results
```

**Phase 2 Target Metrics** (from CIR with RCS):
- Coupling Loss (includes RCS-scaled fast fading)
- Delay Spread
- Angle Spreads (ASD, ASA, ZSD, ZSA)

**Expected Outputs**:
- `cdf_coupling_loss_rsrp.png` - Coupling loss from CIR
- `cdf_delay_spread.png` - RMS delay spread from CIR
- `cdf_asd.png`, `cdf_asa.png` - Azimuth angle spreads from CIR
- `cdf_zsd.png`, `cdf_zsa.png` - Zenith angle spreads from CIR

---

#### Phase 2 Background Channel

**Config**: `testBenches/chanModels/config/statistic_channel_config_isac_phase2_background.yaml`. The difference from target channel calibration is the `isac_disable_background` and `isac_disable_target` to only include background channel.

```
  isac_disable_background: 0     # 0=combined, 1=target-only (set to 1 for separate target analysis)
  isac_disable_target: 1         # 0=include target, 1=background-only (set to 1 for separate background analysis)
``` 

**Run Command**:
```bash
cd $cuBB_SDK/testBenches/chanModels/util

./run_sls_chan_multiseed.sh \
    --seed-start 0 \
    --seed-end 199 \
    --config ../config/statistic_channel_config_isac_phase2_background.yaml \
    --dataset isac_uav_phase2_background \
    --reference-json 3gpp_calibration_isac_uav_phase2.json \
    --phase 2 \
    --isac-channel background \
    --output-dir isac_phase2_background_results
```

**Phase 2 Background Metrics** (from CIR):
- Coupling Loss
- Delay Spread
- Angle Spreads (ASD, ASA, ZSD, ZSA)

**Expected Outputs**:
- `cdf_coupling_loss_rsrp.png` - Coupling loss from CIR
- `cdf_delay_spread.png` - RMS delay spread from CIR
- `cdf_asd.png`, `cdf_asa.png` - Azimuth angle spreads from CIR
- `cdf_zsd.png`, `cdf_zsa.png` - Zenith angle spreads from CIR

---

### Running All ISAC Calibrations at Once

To run all four ISAC calibration phases (Phase 1 & 2 for target and background channels) together:

```bash
cd $cuBB_SDK/testBenches/chanModels/util

# Run all ISAC calibrations with default seeds (0-199)
./run_all_calibrations.sh --isac-only

# Or with custom seed range (quick test with 10 seeds)
./run_all_calibrations.sh --isac-only --seed-start 0 --seed-end 9
```

This runs all four ISAC calibrations:
1. Phase 1 Target (default: 200 seeds) → `isac_phase1_target_results/`
2. Phase 1 Background (default: 200 seeds) → `isac_phase1_background_results/`
3. Phase 2 Target (default: 200 seeds) → `isac_phase2_target_results/`
4. Phase 2 Background (default: 200 seeds) → `isac_phase2_background_results/`

**Combined UMa + ISAC**: To run both UMa and ISAC calibrations together:
```bash
cd $cuBB_SDK/testBenches/chanModels/util
./run_all_calibrations.sh
```

---

## Workflow Details

### Step 1: Channel Generation

The `sls_chan_ex` executable reads a YAML configuration and generates channel realizations.

**Command**:
```bash
./build.aarch64/testBenches/chanModels/examples/sls_chan/sls_chan_ex <config.yaml>
```

**Process**:
1. Parse YAML configuration
2. Initialize topology (sites, UTs, sensing targets)
3. Generate large-scale parameters (LSP)
4. Generate small-scale fading (clusters, rays, phases)
5. Compute channel impulse response (CIR)
6. Save to HDF5 file

**Output Format** (`slsChanData_*.h5`):
```
/cirPerCell/
    cirCoe_cell0        # CIR coefficients [nUT, nSnapshot, nTaps] (complex)
    cirNormDelay_cell0  # Delay indices [nUT, nTaps]
    cirNtaps_cell0      # Number of active taps [nUT]

/topology/
    bsLoc               # BS locations
    utLoc               # UE locations
    isacTargetLinks/    # ISAC target link parameters
        coupling_loss_db
        rcs_linear
        rcs_dbsm
        ...

/commonLinkParams/
    pathLoss            # Path loss [nUT, nCell]
    shadowFading        # Shadow fading [nUT, nCell]
    k_factor            # K-factor [nUT]
    ...
```

---

### Step 2: Multi-Seed Generation

The `run_sls_chan_multiseed.sh` script automates:
1. Loop over seed range (e.g., 0-199)
2. Update `rand_seed` in YAML for each iteration
3. Run `sls_chan_ex` for each seed
4. Collect all H5 files

**Script Usage**:
```bash
./run_sls_chan_multiseed.sh \
    --seed-start <START> \
    --seed-end <END> \
    --config <YAML_FILE> \
    --dataset <DATASET_NAME> \
    --reference-json <REFERENCE_JSON> \
    --phase <1|2> \
    --isac-channel <target|background> \
    --output-dir <OUTPUT_DIR>
```

---

### Step 3: Statistical Analysis

The `analysis_channel_stats.py` script extracts metrics from H5 files and generates CDF plots.

**Command** (called automatically by `run_sls_chan_multiseed.sh`):
```bash
python3 testBenches/chanModels/util/analysis_channel_stats.py \
    slsChanData_*.h5 \
    --reference-json <REFERENCE_JSON> \
    --calibration-phase <1|2> \
    --isac-channel <target|background> \
    --multi-seed \
    --output-dir <OUTPUT_DIR>
```

**Process**:
1. Load all H5 files matching the pattern
2. Extract metrics per 3GPP definitions:
   - **Coupling Loss**: From CIR power (Phase 2) or PL+SF (Phase 1)
   - **Delay Spread**: RMS delay spread per TR 25.996 Annex A
   - **Angle Spreads**: RMS angle spreads per TR 25.996 Annex A
   - **K-factor**: Ratio of LOS to NLOS power
3. Compute empirical CDF for each metric
4. Load 3GPP reference CDFs from JSON
5. Calculate KS statistics (distance between CDFs)
6. Generate overlay plots (simulation vs. reference)

**Note**: The analysis uses only TTI 0 from each seed. For 3GPP calibration, one TTI snapshot per seed is sufficient because statistical diversity comes from multiple random seeds (each with different UE positions, shadow fading, and fast fading realizations).

---

### Step 4: Results Analysis

**Output Directory Structure**:
```
<output_dir>/
    cdf_coupling_loss_rsrp.png     # Coupling loss CDF
    cdf_delay_spread.png           # Delay spread CDF
    cdf_asd.png, cdf_asa.png       # Azimuth spread CDFs
    cdf_zsd.png, cdf_zsa.png       # Zenith spread CDFs
    cdf_k_factor.png               # K-factor CDF (UMa only)
    statistics_summary.txt         # KS statistics and metrics
```

**CDF Plot Features**:
- **Blue curve**: Simulation results
- **Red curve**: 3GPP reference mean
- **Grey shaded area**: Min-max envelope of all 3GPP companies

## Reference Files

### Configuration Files
- `testBenches/chanModels/config/statistic_channel_config_phase1.yaml` - UMa calibration
- `testBenches/chanModels/config/statistic_channel_config_isac_phase1.yaml` - ISAC Phase 1 target
- `testBenches/chanModels/config/statistic_channel_config_isac_phase1_background.yaml` - ISAC Phase 1 background
- `testBenches/chanModels/config/statistic_channel_config_isac_phase2.yaml` - ISAC Phase 2 target
- `testBenches/chanModels/config/statistic_channel_config_isac_phase2_background.yaml` - ISAC Phase 2 background

### Reference Data (3GPP)
- `testBenches/chanModels/util/3gpp_calibration_uma_6ghz.json` - UMa reference
- `testBenches/chanModels/util/3gpp_calibration_isac_uav_phase1.json` - ISAC Phase 1 reference
- `testBenches/chanModels/util/3gpp_calibration_isac_uav_phase2.json` - ISAC Phase 2 reference

### Scripts
- `run_sls_chan_multiseed.sh` - Multi-seed runner for individual calibrations (supports both UMa and ISAC)
- `run_all_calibrations.sh` - Master script to run multiple calibrations together (supports --uma-only, --isac-only, or both)
- `analysis_channel_stats.py` - Analysis and plotting

---

## Appendix: Key 3GPP Specifications

- **TR 38.901**: Study on channel model for frequencies from 0.5 to 100 GHz
  - Section 7.5: Channel model for 0.5 to 100 GHz
  - Section 7.9: Channel model for ISAC
  - Table 7.9.6.2-1: Calibration requirements for ISAC
- **TR 36.777**: Enhanced LTE support for aerial vehicles
  - Annex A: Evaluation assumptions
  - Annex B: Channel modelling details
- **TR 25.996**: Spatial channel model for Multiple Input Multiple Output (MIMO) simulations
  - Annex A: Calculation of angle spreads (RMS method)


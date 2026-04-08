# Running Phase-4 Tests

The scripts in `$cuBB_SDK/testBenches/phase4_test_scripts` can be used to run phase-4 tests (also known as cuBB tests).

## Quick Start with Test Configuration Parser

The recommended way to run tests is using the test configuration parser, which automatically generates all required parameters from a test case string.

### Prerequisites

1. **Reserve and log into a pair of lockable systems** (DU and RU nodes)
2. **Ensure codebase is on shared NFS** - All nodes must have access to the same `$cuBB_SDK` directory via shared network filesystem
3. **Deploy containers:**
   - DU node: Either two containers (DU1 for main operations, DU2 for testMAC) OR one container with two sessions
   - RU node: One container (RU)
4. **Prepare test information:**
   - Test case string (e.g., `F08_6C_79_MODCOMP_STT480000_EH_1P`)
   - Host configuration (e.g., `CG1_R750`, `CG1_CG1`, or `GL4_R750`)

### Test Configuration and Setup

| Container | Command | Description |
|-----------|---------|-------------|
| DU1 | `$cuBB_SDK/testBenches/phase4_test_scripts/parse_test_config_params.sh "F08_6C_79_MODCOMP_STT480000_EH_1P" "CG1_R750" test_params.sh` | Generate parameter file from test case string |
| DU1 | `source test_params.sh` | Load parameters into environment |
| RU | `source test_params.sh` | Load parameters into environment |

### Build and Configure

| Container | Command | Description |
|-----------|---------|-------------|
| DU1* | `$cuBB_SDK/testBenches/phase4_test_scripts/copy_test_files.sh $COPY_TEST_FILES_PARAMS` | Copy test vectors |
| DU1 | `$cuBB_SDK/testBenches/phase4_test_scripts/build_aerial_sdk.sh $BUILD_AERIAL_PARAMS` | Build Aerial SDK |
| RU  | `$cuBB_SDK/testBenches/phase4_test_scripts/build_aerial_sdk.sh $BUILD_AERIAL_PARAMS` | Build Aerial SDK (if different arch than DU) |
| DU1 | `$cuBB_SDK/testBenches/phase4_test_scripts/setup1_DU.sh $SETUP1_DU_PARAMS` | Configure DU setup |
| RU | `$cuBB_SDK/testBenches/phase4_test_scripts/setup2_RU.sh $SETUP2_RU_PARAMS` | Configure RU setup |
| DU1 | `$cuBB_SDK/testBenches/phase4_test_scripts/test_config.sh $TEST_CONFIG_PARAMS` | Configure test parameters |

*Note: `copy_test_files.sh` may need to be run outside the container if the container does not have access to the test vector directory.

### Run Test

| Container | Command | Description |
|-----------|---------|-------------|
| RU | `$cuBB_SDK/testBenches/phase4_test_scripts/run1_RU.sh $RUN1_RU_PARAMS` | Start RU emulator |
| DU1 | `$cuBB_SDK/testBenches/phase4_test_scripts/run2_cuPHYcontroller.sh $RUN2_CUPHYCONTROLLER_PARAMS` | Start cuPHY controller |
| DU2 | `source test_params.sh`<br>`$cuBB_SDK/testBenches/phase4_test_scripts/run3_testMAC.sh $RUN3_TESTMAC_PARAMS` | Start testMAC (separate container on DU) |

## Test Case String Format

Format: `F08_<cells>C_<pattern>_<modifiers>`

- `F08` - Required prefix for performance test cases
- `<cells>C` - Number of cells (e.g., `6C`, `20C`)
- `<pattern>` - Pattern number (e.g., `69`, `59c`)
- `<modifiers>` - Optional modifiers in any order:
  - `BFP9` or `BFP14` - BFP compression with specified bits
  - `STT<value>` - Schedule total time (e.g., `STT480000`)
  - `1P` or `2P` - Number of ports
  - `EH` - Enable early HARQ
  - `GC` - Enable green context
  - `WC<value>` - Work cancel mode (e.g., `WC2`)
  - `PMU<value>` - PMU metrics mode (e.g., `PMU3`)
  - `NS<value>` - Number of slots (e.g., `NS30000`)
  - `NICD` - Enable NIC timing logs
  - `RUWT` - Enable RU C-plane worker tracing logs
  - `NOPOST` - Enable reduced logging mode (disables detailed tracing and processing time logs)

### Host Configuration

Format: `<DU_HOST>_<RU_HOST>`

Valid combinations:
- `CG1_R750` - Grace (CG1) DU with x86 (R750) RU
- `CG1_CG1` - Grace DU with Grace RU
- `GL4_R750` - Grace+L4 (GL4) DU with x86 RU


## Parser Examples

### Basic 6-cell MODCOMP test with Early HARQ:
```bash
$cuBB_SDK/testBenches/phase4_test_scripts/parse_test_config_params.sh "F08_6C_79_MODCOMP_STT480000_EH_1P" "CG1_R750" test_params.sh
```

### 20-cell test with BFP14, Early HARQ, Green Context, Dual port, 30000 total slots in the run, with NIC-level debug:
```bash
$cuBB_SDK/testBenches/phase4_test_scripts/parse_test_config_params.sh "F08_20C_59c_BFP14_EH_GC_NS30000_NICD_2P" "GL4_R750" test_params.sh
```

### With comprehensive debugging (NIC timings + RU C-plane worker tracing):
```bash
$cuBB_SDK/testBenches/phase4_test_scripts/parse_test_config_params.sh "F08_6C_79_MODCOMP_STT480000_EH_NICD_RUWT_1P" "CG1_R750" test_params.sh
```

### With custom build directories:

By default, the parsing scripts will determine which preset configuration to run the test with and the build artifacts will be in build.$PRESET.$(uname -m).
However, if desired, the --custom-build-dir flag can be used, which will tell all of the scripts to use $CUSTOM.$(uname -m) instead.

```bash
$cuBB_SDK/testBenches/phase4_test_scripts/parse_test_config_params.sh "F08_6C_79_MODCOMP_STT480000_EH_1P" "CG1_R750" test_params.sh \
    --custom-build-dir custom
```

## Environment Variables Set by Parser

### Test Execution Variables
- `COPY_TEST_FILES_PARAMS` - Parameters for copy_test_files.sh
- `BUILD_AERIAL_PARAMS` - Parameters for build_aerial_sdk.sh
- `SETUP1_DU_PARAMS` - Parameters for setup1_DU.sh
- `SETUP2_RU_PARAMS` - Parameters for setup2_RU.sh
- `TEST_CONFIG_PARAMS` - Parameters for test_config.sh
- `RUN1_RU_PARAMS` - Parameters for run1_RU.sh
- `RUN2_CUPHYCONTROLLER_PARAMS` - Parameters for run2_cuPHYcontroller.sh
- `RUN3_TESTMAC_PARAMS` - Parameters for run3_testMAC.sh

### Post-Processing Variables
- `POST_PROCESSING_CICD_PARAMS` - All flags for post_processing_cicd.sh (exact flags used for CICD)
- `PARSE_LOGS_PARAMS` - Parameters for post_processing_parse.sh
- `POST_PROCESSING_PERF_PARAMS` - Parameters for post_processing_analyze.sh --perf-metrics
- `POST_PROCESSING_COMPARE_PARAMS` - Parameters for post_processing_analyze.sh --compare-logs
- `POST_PROCESSING_GATING_PARAMS` - Parameters for post_processing_analyze.sh --gating-threshold
- `POST_PROCESSING_WARNING_PARAMS` - Parameters for post_processing_analyze.sh --warning-threshold
- `POST_PROCESSING_ABSOLUTE_PARAMS` - Parameters for post_processing_analyze.sh --absolute-threshold
- `POST_PROCESSING_LATENCY_PARAMS` - Parameters for post_processing_analyze.sh --latency-summary (NICD only)

## MUMIMO Patterns

MUMIMO configuration is selected based on a fixed pattern list.  As of 9/17/2025 the following are MUMIMO patterns:
66a, 66b, 66c, 66d, 67a, 67b, 67c, 67d, 69, 69a, 69b, 69c, 71, 73, 79

## Manual Script Usage

All scripts support `-h` or `--help` options for detailed usage information. You can also run them manually without the parser:

```bash
# Example manual usage
$cuBB_SDK/testBenches/phase4_test_scripts/copy_test_files.sh 69 --max_cells 6
$cuBB_SDK/testBenches/phase4_test_scripts/setup1_DU.sh --mumimo 1
$cuBB_SDK/testBenches/phase4_test_scripts/setup2_RU.sh
$cuBB_SDK/testBenches/phase4_test_scripts/test_config.sh 69 --compression=4 --num-cells=6 --num-slots=600000
$cuBB_SDK/testBenches/phase4_test_scripts/run1_RU.sh
$cuBB_SDK/testBenches/phase4_test_scripts/run2_cuPHYcontroller.sh
$cuBB_SDK/testBenches/phase4_test_scripts/run3_testMAC.sh
```

## Multi-L2 test

Multi-L2 test means running two L2/testMAC instances with one cuphycontroller instance.
Below instruction use F08 8C 60 case as example.

### Configure

```bash
$cuBB_SDK/testBenches/phase4_test_scripts/setup1_DU.sh --ml2  # Enable Multi-L2 by "--ml2"
$cuBB_SDK/testBenches/phase4_test_scripts/setup2_RU.sh
$cuBB_SDK/testBenches/phase4_test_scripts/test_config.sh 60 --num-cells=8
```

The test_config.sh command automatically configure as below for Multi-L2 test.

(1) Enable Multi-L2 by setting `nvipc_config_file` in `l2_adapter_config_XXX.yaml`.

```yaml
nvipc_config_file: nvipc_multi_instances.yaml
```

(2) Configure nvipc_multi_instances.yaml to assign cell `0~3` for the first instance and cell `4~7` for the second instance, set the second instance prefix to nvipc1.

```yaml
transport:
- transport_id: 0
  phy_cells: [0, 1, 2, 3]
  shm_config:
    prefix: nvipc

- transport_id: 1
  phy_cells: [4, 5, 6, 7]
  shm_config:
    prefix: nvipc1
```

(3) Create a test_mac_config_1.yaml for the secondary testMAC instance and configure necessary values in it.

```yaml
# Copy test_mac_config.yaml and change below values to test_mac_config_1.yaml

transport:
  shm_config: {prefix: nvipc1}
log_name: testmac1.log
oam_server_addr: 0.0.0.0:50053

# Assign CPU cores for the second test_mac instance
low_priority_core: 30
sched_thread_config: {name: mac_sched, cpu_affinity: 31, sched_priority: 96}
recv_thread_config: {name: mac_recv, cpu_affinity: 32, sched_priority: 95}
builder_thread_config: {name: fapi_builder, cpu_affinity: 33, sched_priority: 95}
worker_cores: [34, 35, 36, 37, 38, 39]
```

### Run Multi-L2 test

```bash
$cuBB_SDK/testBenches/phase4_test_scripts/run1_RU.sh                 # Run ru_emulator
$cuBB_SDK/testBenches/phase4_test_scripts/run2_cuPHYcontroller.sh    # Run cuphycontroller_scf
$cuBB_SDK/testBenches/phase4_test_scripts/run3_testMAC.sh --ml2 0    # Run the first instance of test_mac
$cuBB_SDK/testBenches/phase4_test_scripts/run3_testMAC.sh --ml2 1    # Run the second instance of test_mac
```

The --ml2 0/1 is translated to below arguments for test_mac:

```bash
# First instance: enable cell 0~3 by cell_mask=0x0F, use default config file test_mac_config.yaml
--ml2 0  =>  --cells 0x0F
# Second instance: enable cell 4~7 by cell_mask=0xF0, explicitly select config file test_mac_config_1.yaml
--ml2 1  =>  --cells 0xF0 --config test_mac_config_1.yaml
```

## Post-Processing Scripts

### Prerequisites: Virtual Environment Setup

The post-processing scripts require a Python virtual environment with the `aerial_postproc` package installed. Before running any post-processing scripts, create the virtual environment:

```bash
# Create the virtual environment (one-time setup)
$cuBB_SDK/testBenches/phase4_test_scripts/aerial_postproc/venv_create.sh
```

By default, the virtual environment is created at `$HOME/.aerial_postproc_venv`. To use a custom location, set the `AERIAL_POSTPROC_VENV` environment variable before creating and using the venv:

```bash
# Create venv at custom location
export AERIAL_POSTPROC_VENV=/path/to/custom/venv
$cuBB_SDK/testBenches/phase4_test_scripts/aerial_postproc/venv_create.sh

# The post-processing scripts will automatically use this location
./post_processing_cicd.sh ...
```

Note: In CICD environments, the venv is typically pre-installed at `/opt/aerial_postproc_venv`.

### Script Overview

Three scripts are provided for post-processing test results:

- **post_processing_parse.sh** - Runs the expensive parsing step to generate binary output folders
- **post_processing_analyze.sh** - Runs downstream processing on already-parsed binary data
- **post_processing_cicd.sh** - CICD wrapper that runs the full sequence with proper return codes

### Using Post-Processing with Generated Parameters

**CICD Workflow (Simple - Recommended):**
```bash
source test_params.sh
./post_processing_cicd.sh phy.log testmac.log ru.log ./output $POST_PROCESSING_CICD_PARAMS
exit $?
```

**Developer Workflow (Granular):**
```bash
source test_params.sh

# Step 1: Parse logs (expensive, do once)
./post_processing_parse.sh phy.log testmac.log ru.log ./output $PARSE_LOGS_PARAMS

# Step 2: Run individual post-processing as needed
./post_processing_analyze.sh ./output/binary ./output $POST_PROCESSING_PERF_PARAMS
./post_processing_analyze.sh ./output/binary ./output $POST_PROCESSING_COMPARE_PARAMS
./post_processing_analyze.sh ./output/binary ./output $POST_PROCESSING_GATING_PARAMS
./post_processing_analyze.sh ./output/binary ./output $POST_PROCESSING_WARNING_PARAMS
./post_processing_analyze.sh ./output/binary ./output $POST_PROCESSING_ABSOLUTE_PARAMS

# NICD only (if POST_PROCESSING_LATENCY_PARAMS is set)
if [[ -n "$POST_PROCESSING_LATENCY_PARAMS" ]]; then
    ./post_processing_analyze.sh ./output/binary_ls ./output $POST_PROCESSING_LATENCY_PARAMS
fi
```

### Manual Script Usage

The following sections document the individual scripts and their options for manual invocation.

#### post_processing_parse.sh
Runs the expensive cicd_parse.py step to generate binary output folders.

```bash
# Parse for performance metrics only
./post_processing_parse.sh phy.log testmac.log ru.log ./output --perf-metrics --mmimo

# Parse for latency summary only (NICD)
./post_processing_parse.sh phy.log testmac.log ru.log ./output --latency-summary --mmimo

# Parse for both formats
./post_processing_parse.sh phy.log testmac.log ru.log ./output --perf-metrics --latency-summary --mmimo
```

#### post_processing_analyze.sh
Runs downstream processing on already-parsed binary data.

```bash
# Performance metrics extraction (creates perf.csv)
./post_processing_analyze.sh ./output/binary ./output --perf-metrics --mmimo

# Compare logs visualization
./post_processing_analyze.sh ./output/binary ./output --compare-logs --label "my_test" --mmimo

# Gating threshold check
./post_processing_analyze.sh ./output/binary ./output --gating-threshold /path/to/gating_perf_requirements.csv

# Warning threshold check
./post_processing_analyze.sh ./output/binary ./output --warning-threshold /path/to/warning_perf_requirements.csv

# Absolute threshold check
./post_processing_analyze.sh ./output/binary ./output --absolute-threshold /path/to/perf_requirements.csv

# Latency summary visualization (NICD)
./post_processing_analyze.sh ./output/binary_ls ./output --latency-summary --mmimo
```

#### post_processing_cicd.sh
CICD wrapper that runs the full post-processing sequence with proper return codes.

```bash
# Full CICD workflow (recommended)
./post_processing_cicd.sh phy.log testmac.log ru.log ./output \
    --gating-threshold /path/to/gating_perf_requirements.csv \
    --warning-threshold /path/to/warning_perf_requirements.csv \
    --absolute-threshold /path/to/perf_requirements.csv \
    --mmimo --label my_test
```

Return codes:
- 0 = All gating/absolute thresholds passed
- 1 = Gating threshold failed OR absolute threshold failed OR processing error
- Warning threshold failure does NOT cause return 1 (prints warning instead)

## Additional Notes

- Configuration files can be automatically generated for nrSim test cases. See [Generic steps to run nrSim test cases](https://docs.nvidia.com/aerial/cuda-accelerated-ran/latest/aerial_cubb/cubb_quickstart/running_cubb-end-to-end.html#generic-steps-to-run-nrsim-test-cases-using-phase4-scripts)
- The `yq` tool must be installed for test_config.sh to work outside the container
- Build directories may differ based on system architecture (aarch64 or x86_64)

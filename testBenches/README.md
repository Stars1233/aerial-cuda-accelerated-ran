# Aerial SDK TestBenches

## Overview

The `testBenches` directory contains various testing and benchmarking tools for the Aerial SDK components:

```
testBenches/
├── cubb_gpu_test_bench/     # GPU-only performance testbench
├── perf/                    # Python helper scripts for cubb_gpu_test_bench
├── chanModels/              # 3GPP 38.901 channel model
└── phase4_test_scripts/     # cuBB test (CPU+GPU) automation scripts
```

### Components

- **cubb_gpu_test_bench**: GPU-only performance testbench for measuring channel latencies, cell capacity, and GPU memory/power; simplified from the comprehensive cuBB testbench
- **perf**: Python helper scripts for `cubb_gpu_test_bench` to generate configurations, run tests, and visualize results
- **chanModels**: 3GPP 38.901 channel model library including link-level (TDL/CDL) and system-level (UMa/UMi/RMa) models
- **phase4_test_scripts**: Automated scripts for running end-to-end cuBB tests (also called Phase-4 tests in existing documents; see [phase4_test_scripts/README.md](phase4_test_scripts/README.md))

---

# Performance Testing with cubb_gpu_test_bench

## Introduction

`cubb_gpu_test_bench` is a GPU-only testbench that enables multiple channels sharing the same GPU through NVIDIA Multi-Process Service (MPS). It can execute the workloads, measure the latency of each workload over a specific number of time slots, and visualize the latency results. Here latency refers to the execution time of each workload on the GPU as measured by CUDA events. This means that the setup stage and CPU execution is not considered in cubb_gpu_test_bench. Additionally, it can collect Nsight traces for profiling purposes.

The core component is a C platform testbench at `<aerial_sdk>/testBenches/cubb_gpu_test_bench` that accepts command options and a YAML input file (`cubb_gpu_test_bench -h` for options). A Python interface in `<aerial_sdk>/testBenches/perf` drives the testbench: it can use a **single YAML config** for all defaults and **CLI flags to override** any parameter, optionally skipping separate JSON generation when the YAML defines test vectors and use case. It also visualizes results and can collect Nsight traces. Using the Python interface is recommended.

![Diagram of cubb_gpu_test_bench](perf/cubb_gpu_test_bench.png)

## Requirements

The tests can be run using a Linux environment making one of more GPU available. It's recommended to use the cuBB container since all the following requirements are automatically satisfied. The cuBB container is available on the [NVIDIA GPU Cloud (NGC)](https://registry.ngc.nvidia.com/orgs/qhrjhjrvlsbu/containers/aerial-cuda-accelerated-ran). Follow the instructions on that page to pull the container and to run the container.

If not using the cuBB container, make sure the following requirements are met:

* bash or zsh as default shell
* CUDA toolkit 12.6+ and properly configured so that nvidia-cuda-mps-control and nvidia-smi are in PATH
* Python 3.8+ and the following additional packages
  * numpy, pyCUDA, pyYAML, matplotlib

## Test Vectors (TVs)

A TV is an H5 file that includes the configurations, input data and reference output data for a specific channel. TVs are generated in MATLAB from 5GModel at `<aerial_sdk>/5GModel`. They should be available in either the **full** or **compact** set of performance TVs, depending on which generation path you use.

### Generating TVs in MATLAB

**When using `cubb_gpu_test_config.yaml` (ULMIX/DLMIX)** — Generate only the channels needed for that config. From `<aerial_sdk>/5GModel/nr_matlab` in MATLAB:

```matlab
testCompGenTV_dlmix([9481, 9696, 9905, 9906, 10047]);   % DLMIX: SSB, PDCCH, PDSCH, CSIRS
testCompGenTV_ulmix([4544, 4548, 4566]);                % ULMIX: PRACH, PUSCH, PUCCH
```

Outputs go to `GPU_test_input/`. The resulting filenames match the `vector_files` in `cubb_gpu_test_config.yaml` (e.g. `TVnr_DLMIX_9905_PDSCH_gNB_CUPHY_s0p5.h5`, `TVnr_ULMIX_4548_PUSCH_gNB_CUPHY_s0p5.h5`). Slot/PDU suffixes (s0p5, s0p3, etc.) depend on the test case; adjust the YAML `vector_files` if names differ. The MAC TV (e.g. `TV_cumac_F08-MC-CC-20PC_DL.h5`) may come from a separate flow or archive.

**Legacy perf TVs** — To use the older perf TV naming (e.g. for scripts that expect `example_100_testcases_avg_F08.json`), generate the full perf set in MATLAB:

```matlab
runRegression({'TestVector'}, {'perf_TVs'}, {'full'})
```

Wait for generation to finish. Legacy TV filenames follow patterns like `TV_cuphy_F08-RA-01.h5`, `TV_cuphy_V08-DS-02_slot0_MIMO4x4_PRB45_DataSyms12_qam256.h5`, etc.

### Copying TVs to your testvectors folder

**Recommended: shell script + YAML** — When using a YAML config (e.g. `cubb_gpu_test_config.yaml`) with `vector_files`, use the copy script so the set stays in sync with the YAML:

```shell
cd <aerial_sdk>/testBenches/perf
./copy_tvs_from_yaml.sh cubb_gpu_test_config.yaml <src_dir> <dst_dir>
```

Example: from MATLAB `GPU_test_input` to SDK testVectors:

```shell
./copy_tvs_from_yaml.sh cubb_gpu_test_config.yaml $cuBB_SDK/5GModel/nr_matlab/GPU_test_input $cuBB_SDK/testVectors
```

Example: from an archive to a container mount:

```shell
./copy_tvs_from_yaml.sh cubb_gpu_test_config.yaml /path/to/tv/archive /opt/nvidia/cuBB/testVectors
```

**Alternative (legacy TV names)** — `copyCubbGpuTestbenchTv.py` copies a fixed list of commonly used performance TVs; adjust `tvSource`, `tvDestination`, and `tvNames` (L98/L100) as needed.

* If you see the following error, it is most likely a TV issue. Please try pull the latest Aerial SDK changes and regenerate the latest TVs.
  ```shell
  terminate called after throwing an instance of 'cuphy::cuphyHDF5_exception' what(): No such scalar or structure field with the given name exists.
  ```

## Building cubb_gpu_test_bench

`cubb_gpu_test_bench` is built as part of the standard Aerial SDK. You can use the phase4 build script or configure and build with CMake directly.

**Using the build script (recommended)** — From the SDK root, build the full Aerial SDK (includes cubb_gpu_test_bench):

```shell
$cuBB_SDK/testBenches/phase4_test_scripts/build_aerial_sdk.sh
```

To build only `cubb_gpu_test_bench` with custom CMake options:

```shell
$cuBB_SDK/testBenches/phase4_test_scripts/build_aerial_sdk.sh --targets cubb_gpu_test_bench -- -DCMAKE_TOOLCHAIN_FILE=cuPHY/cmake/toolchains/<tool_chain_file> -DCMAKE_BUILD_TYPE=<Release|Debug>
```

**Manual CMake build** — Without the script, configure from the aerial_sdk root, then build:

```shell
# Configure (replace <build_dir>, <tool_chain_file>, and <Release|Debug> as needed)
cmake -B <build_dir> -GNinja -DCMAKE_TOOLCHAIN_FILE=cuPHY/cmake/toolchains/<tool_chain_file> -DCMAKE_BUILD_TYPE=<Release|Debug>

# Build all testbench examples (cubb_gpu_test_bench + channel models)
cmake --build <build_dir> --target testbenches_examples

# Or build only cubb_gpu_test_bench
cmake --build <build_dir> --target cubb_gpu_test_bench
```

## Run the Testbench

The main goal of this testbench is to measure the latency of multi-channel, multi-cell workloads. It repeatedly launches a pre-defined pattern on the GPU and measures latency per channel using CUDA events. In this latency measurement mode, channel objects are reset between patterns. Due to workload variation and interactions between multiple channels (e.g., different numbers of SMs assigned per channel over time), latency varies across iterations. This produces CDFs (Cumulative Distribution Functions) of latency for different channels. By comparing these latencies against the budgets, we can determine whether a specific cell count is achievable.

### Recommended: YAML config + CLI overrides

The preferred way to run tests is with a **single YAML file** that holds all defaults. You can **skip the JSON generation step** when the YAML defines `vector_files` and `usecase`: the Python interface will generate the test-case and use-case JSON in memory. Any **CLI flag overrides the YAML** for that option.

1. **Prepare a YAML config** (e.g. `cubb_gpu_test_config.yaml`, or `cubb_gpu_test_config_perf101.yaml` in `<testBenches>/perf`). In it set:
   * `config.testbench_folder` (or `cuphy`) and `config.testvectors_folder` (or `vectors`)
   * `config.vector_files`: per-channel lists of TV filenames under the testvectors folder
   * `config.usecase`: e.g. `F08` (used for auto cell-capacity detection and generated JSON names)
   * GPU and sweep settings: `gpu`, `freq`, `power`, `target`, `start`, `cap`, `slots`, `iterations`, `delay`, `graph`, etc.
   * Optional: `tdd_priorities`, `start_delay`, `override_test_vectors`, `latency_budget` (see comments in the sample YAML)

2. **Run from `<testBenches>/perf`**:

   ```shell
   python3 measure.py --yaml cubb_gpu_test_config.yaml --start 20 --cap 20 --slots 300
   ```

   Omit `--start`/`--cap` (and other flags) to use the values from the YAML. Any CLI input can override the corresponding parameter in the YAML file, e.g. `--freq 1980 --target 8 12 8 90 132 12`.

3. **Outputs**: `vectors-<N>.yaml`, `buffer-<N>.txt`, and a results JSON (e.g. `008_012_008_090_132_012_sweep_graphs_avg_F08.json`). The script prints detected GPU, cell count, and SM allocation; for TDD F08/F09/F14 it can auto-report cell capacity (100% on-time).

4. **Visualize latency**: `python3 compare.py --filename <results_json> --cells 20+0`. Use multiple `--filename` / `--cells` to compare runs.

   ```shell
   python3 compare.py --filename 008_012_008_090_132_012_sweep_graphs_avg_F08.json --cells 20+0
   ```


**Key options** (set in YAML or override via CLI):

| Option | Meaning |
|--------|--------|
| `--gpu` | GPU ordinal |
| `--freq` | GPU clock frequency (MHz); set and restored around the run |
| `--power` | GPU power limit (W); set and restored around the run |
| `--start`, `--cap` | Cell count sweep range [start, cap] |
| `--slots` | Slots per run (e.g. 8/16 debug, 120 quick, 400 capacity) |
| `--target` | SM allocation per subcontext (order depends on isolation flags) |
| `--graph` | Use CUDA graphs (recommended for lower latency) |

*Auto-detect cell capacity*: For TDD patterns dddsuudddd and usecases F08, F09, F14, the script checks on-time percentage of all channels; 100% on-time is PASS. Result is printed and stored in the results JSON under `ontimePercent`.

---

### Legacy instructions (JSON config)

If you prefer the older workflow with explicit JSON config and use-case files:

1. **Generate use-case JSON** (optional; not needed when using YAML with `vector_files`):

   ```shell
   python3 generate_avg_TDD.py --peak 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 --avg 0 --case F08 --exact --fdm
   ```

   This produces e.g. `uc_avg_F08_TDD.json`. For 20C, include 20 in `--peak`.

2. **Run with explicit config and UC**:

   ```shell
   python3 measure.py --cuphy <testBenches>/build --vectors <test_vectors> --config example_100_testcases_avg_F08.json --uc uc_avg_F08_TDD.json --delay 100000 --gpu 0 --freq 1980 --start 20 --cap 20 --iterations 1 --slots 300 --power 900 --target 8 12 8 90 132 12 --2cb_per_sm --save_buffer --priority --prach --prach_isolate --pdcch --pdcch_isolate --pucch --pucch_isolate --tdd_pattern dddsuudddd --pusch_cascaded --ssb --ssb_isolate --csirs --groups_dl --pack_pdsch --groups_pusch --ldpc_parallel --graph
   ```

Other steps are the same with above instructions.

### Other Usages

#### Collect Nsight Systems Traces

Add `--debug --debug_mode nsys` to your `measure.py` command (YAML or legacy). Use a small number of slots (8 or 16) when collecting traces. By default, traces include CUDA graph node information; use `--debug_mode nsys_simple` to omit details inside each graph.

#### GPU Power and Memory Measurement

**Quick GPU status check**: Run `nvidia-smi dmon` in a separate terminal to monitor GPU status in real-time.

**Detailed power and memory measurement**: Use the `--measure_power` option. The script queries GPU status every 10 ms during the test; a JSON file is written on success. Power/memory measurement needs the GPU continuously busy: use `--delay 0 --slots 10 --iterations 1000` and `--measure_power`.

With YAML config:

```shell
python3 measure.py --yaml cubb_gpu_test_config.yaml --delay 0 --start 8 --cap 8 --iterations 1000 --slots 10 --measure_power
```

**Analyze power**: `python3 power.py --filename <power_results_json> --cells 8+0` — plots GPU power over time and prints max power (keep GPU frequency locked during measurement).

**Analyze memory**: `python3 memory_plot.py --filename <power_results_json> --cells 8+0` — plots GPU memory over time and prints peak usage.

#### Running the C Platform Testbench Natively

For debugging, add `--test` to your `measure.py` command. The script then only prints the C testbench command and the generated vectors YAML path; run that command yourself. Start the MPS server manually when running the C testbench natively.

---

# Channel Models

The `chanModels` directory contains 3GPP 38.901 channel model implementations for wireless propagation simulation, including link-level (TDL/CDL) and system-level (UMa/UMi/RMa) models. It provides a library and unit test examples. These channel models are used by cuPHY and cuMAC for simulation.

## Channel Model Types

### System Level Channel Model
- **Location**: `chanModels/src/sls_chan_src/`
- **Functionality**: 3GPP TR 38.901 statistical channel model implementation
  - Supports UMa, UMi, and RMa
  - Supports ISAC UAV sensing targets (CPU-only)
  - For UMa/UMi/RMa, GPU-accelerated computation with CUDA kernels and a CPU-only mode if GPU is not available
  - Configurable via YAML files
- **Example**: `chanModels/examples/sls_chan/` - Demo application with H5 output support
- Currently, only 6 GHz - UMa, 6 GHz - UMa-AV monostatic sensing are calibrated with 3GPP reference curves. Calibration of other scenarios will be added later

### Link Level Channel Model
#### Tapped Delay Line (TDL) Channel Model
- **Location**: `chanModels/src/tdl_chan_src/`
- **Functionality**: 3GPP TDL channel model (TDL-A, TDL-B, TDL-C profiles)
  - Simplified delay profile models for link-level simulation
  - Fast fading with configurable Doppler effects
- **Example**: `chanModels/examples/tdl_chan/`
- Calibrated with MATLAB 5G toolbox and our own implementation inside `<aerial_sdk>/5GModel/nr_matlab/channel/genTDL.m`

#### Clustered Delay Line (CDL) Channel Model
- **Location**: `chanModels/src/cdl_chan_src/`
- **Functionality**: 3GPP CDL channel model (CDL-A through CDL-E profiles)
  - Cluster-based multipath propagation
  - Support for both LOS and NLOS scenarios
- **Example**: `chanModels/examples/cdl_chan/`
- Calibrated with MATLAB 5G toolbox and our own implementation inside `<aerial_sdk>/5GModel/nr_matlab/channel/genCDL.m`

## Additional Components

- **OFDM Modulator/Demodulator**: `chanModels/src/ofdm_src/` - OFDM Time-frequency domain conversion
- **Gaussian Noise Adder**: `chanModels/src/gauNoiseAdder.*` - AWGN channel implementation
- **Fading Channel**: `chanModels/src/fading_chan.*` - Generic fading channel implementation that includes OFDM modulation + fast fading + add noise + OFDM demodulation
- **Analysis Tools**: `chanModels/util/` - Python scripts for channel statistics analysis and visualization with H5 dumped from `sls_chan_ex`

## Building and Running Channel Model Examples

Channel model examples are built as part of the standard Aerial SDK. You can use the phase4 build script or configure and build with CMake directly (same options as [Building cubb_gpu_test_bench](#building-cubb_gpu_test_bench)).

**Using the build script (recommended)** — From the SDK root, build the full Aerial SDK (includes channel model examples):

```shell
$cuBB_SDK/testBenches/phase4_test_scripts/build_aerial_sdk.sh
```

To build only channel-model (and other testbench) targets with custom CMake options:

```shell
$cuBB_SDK/testBenches/phase4_test_scripts/build_aerial_sdk.sh --targets testbenches_examples -- -DCMAKE_TOOLCHAIN_FILE=cuPHY/cmake/toolchains/<tool_chain_file> -DCMAKE_BUILD_TYPE=<Release|Debug>
```

**Manual CMake build** — Without the script, configure from the aerial_sdk root, then build:

```shell
# Configure (replace <build_dir>, <tool_chain_file>, and <Release|Debug> as needed)
cmake -B <build_dir> -GNinja -DCMAKE_TOOLCHAIN_FILE=cuPHY/cmake/toolchains/<tool_chain_file> -DCMAKE_BUILD_TYPE=<Release|Debug>

# Build all testbench examples (cubb_gpu_test_bench + channel models)
cmake --build <build_dir> --target testbenches_examples

# Or build specific channel model examples
cmake --build <build_dir> --target sls_chan_ex
cmake --build <build_dir> --target tdl_chan_ex
cmake --build <build_dir> --target cdl_chan_ex
```

Run any example with `-h` for usage instructions (e.g. `sls_chan_ex -h`, `tdl_chan_ex -h`, `cdl_chan_ex -h`).

# Phase-4 Test Scripts (cuBB Tests)

The `phase4_test_scripts` directory contains automated scripts for running end-to-end cuBB tests (also called Phase-4 tests in existing documents) with DU and RU emulators.

These scripts help automate:
- Test configuration from test case strings
- Building Aerial SDK with proper presets
- Setting up DU and RU environments
- Running multi-component tests (RU emulator, cuPHYcontroller, testMAC)
- Multi-L2 test scenarios

**For detailed usage instructions**, see [phase4_test_scripts/README.md](phase4_test_scripts/README.md).

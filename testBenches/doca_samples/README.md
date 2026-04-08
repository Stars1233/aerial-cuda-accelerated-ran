# DOCA Samples

This directory contains build and run scripts for DOCA (Data Center on a Chip Architecture) sample applications.

## Directory Structure

```
doca_samples/
├── README.md
└── doca_gpunetio_send_wait_time/
    ├── build.sh    # Build script for the sample
    └── run.sh      # Run script with CLI argument support
```

## Prerequisites

- DOCA SDK installed (typically at `/opt/mellanox/doca/`)
- DPDK libraries installed (typically at `/opt/mellanox/dpdk/`)
- CUDA Toolkit installed (typically at `/usr/local/cuda/`)
- Mellanox NIC with GPUDirect RDMA support
- NVIDIA GPU

## Supported Architectures

The scripts automatically detect and support:
- **x86_64** (Intel/AMD)
- **aarch64** (ARM64/Grace)

## Sample Applications

### doca_gpunetio_send_wait_time

This sample demonstrates GPU-initiated network packet transmission with precise timing control using DOCA GPUNetIO.

#### Building

```bash
cd doca_gpunetio_send_wait_time
./build.sh
```

To clean and rebuild:
```bash
./build.sh --clean
./build.sh
```

#### Running

```bash
./run.sh -n <NIC_PCIE_ADDR> -g <GPU_PCIE_ADDR> [-t <WAIT_TIME_NS>]
```

**Arguments:**
- `-n`: PCIe address of the NIC port (required)
- `-g`: PCIe address of the GPU (required)
- `-t`: Wait time in nanoseconds (optional, default: 5000000)

**Example:**
```bash
./run.sh -n 0002:03:00.0 -g 0002:09:00.0 -t 5000000
```

#### Running with compute-sanitizer (memcheck, racecheck, synccheck, initcheck)

Use `run_sanitizer.sh <tool>` with the same arguments as `run.sh`:

```bash
# Memcheck – GPU memory access and leak detection
./run_sanitizer.sh memcheck  -n 0000:01:00.0 -g 0009:01:00.0 -t 5000000

# Racecheck – GPU shared memory race detection
./run_sanitizer.sh racecheck -n 0000:01:00.0 -g 0009:01:00.0 -t 5000000

# Synccheck – GPU synchronization correctness
./run_sanitizer.sh synccheck -n 0000:01:00.0 -g 0009:01:00.0 -t 5000000

# Initcheck – uninitialized GPU global memory access
./run_sanitizer.sh initcheck -n 0000:01:00.0 -g 0009:01:00.0 -t 5000000
```

Replace NIC/GPU addresses and `-t` value as needed. Each command runs the app under the corresponding compute-sanitizer tool (same `LD_LIBRARY_PATH` and `sudo -E` as `run.sh`).

#### Finding PCIe Addresses

To find NIC PCIe addresses:
```bash
lspci | grep -i mellanox
```

To find GPU PCIe addresses:
```bash
lspci | grep -i nvidia
```

## Adding New Samples

To add a new DOCA sample:

1. Create a new directory under `doca_samples/` with the sample name
2. Copy and modify `build.sh` and `run.sh` from an existing sample
3. Update the `DOCA_SAMPLE_DIR` variable to point to the correct source location
4. Adjust command-line arguments as needed for the specific sample

## Troubleshooting

### Build Errors

1. **"Could not find suitable CUDA compiler: nvcc"**
   - Ensure CUDA is in your PATH: `export PATH=/usr/local/cuda/bin:$PATH`

2. **"Package libdpdk was not found"**
   - The build script will automatically set PKG_CONFIG_PATH

3. **"Dependency doca-flow not found"**
   - The build script will attempt to install `libdoca-sdk-flow-dev`

4. **"Couldn't find requested CUDA module 'cuda'"**
   - The build script will create the required libcuda.so symlink

5. **"Dependency doca-rdma not found"**
   - The gpunetio_send_wait_time sample does not use doca_rdma; the upstream meson.build declares it. The build script automatically generates a stub doca-rdma.pc at build time when the DOCA RDMA library is not installed.

### Runtime Errors

1. **"error while loading shared libraries"**
   - The run script automatically sets LD_LIBRARY_PATH

2. **Permission denied**
   - The scripts use `sudo` where required for hardware access


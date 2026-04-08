<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

**Aerial cuMAC**

CUDA-based L2 scheduler acceleration over GPU.

All cuMAC data structures and scheduler module classes are included in the name space _cumac_. The header files _api.h_ and _cumac.h_ should be included in an application of cuMAC.  

**Prerequisites:** 

1. CMake (version 3.18 or newer) 

If you have a version of CMake installed, the version number can be determined as follows: 

```
cmake --version
```

You can download the latest version of CMake from the official CMake website. 

2. CUDA (version 12 or newer) 

CMake intrinsic CUDA support will automatically detect a CUDA installation using a CUDA compiler (nvcc), which is located via the PATH environment variable. To check for nvcc in your PATH: 

```
which nvcc
```

To use a non-standard CUDA installation path (or to use a specific version of CUDA): 

```
export PATH=/usr/local/cuda-12.0/bin:$PATH
```

For more information on CUDA support in CMake, see https://devblogs.nvidia.com/building-cuda-applications-cmake/. 

3. cuMAC requires a minimum GPU architecture of Ampere or newer. 

4. HDF5 (Hierarchical Data Format 5) 

The cuMAC CMake system currently checks for a specific version (1.10) of HDF5, for compatibility with binary-only distributions of cuMAC. 

To install a specific version of HDF5 from a source code archive: 

4.1. Remove the original hdf5 library (if necessary) 

```
dpkg -l | grep hdf5
sudo apt-get remove <name of these libraries> 
```

4.2. To build from source: 

```
wget  https://support.hdfgroup.org/ftp/HDF5/releases/hdf5-1.10/hdf5-1.10.5/src/hdf5-1.10.5.tar.gz 

tar -xzf hdf5-1.10.5.tar.gz 

cd hdf5-1.10.5 

./configure --prefix=/usr/local --enable-cxx --enable-build-mode=production 

sudo make install
```
 
**Building cuMAC Examples:**

`cuMAC` is built as part of the main `aerial_sdk` project. Building is performed from a top-level build directory (e.g., `<aerial_sdk>/build`), **not** by `cd`-ing into the `cuMAC` subdirectory. Refer to the main `README.md` in the `aerial_sdk` root for general CMake configuration and build instructions.

Once the project is configured according to the top-level `README.md`, you can build individual `cuMAC` executable targets from the same top-level build directory (e.g., `<aerial_sdk>/build`) using `cmake --build . --target <target_name>`.

To build a single target, for example `multiCellMuMimoScheduler`:
```shell
# Run from your top-level build directory (e.g., <aerial_sdk>/build)
cmake --build . --target multiCellMuMimoScheduler
```

To build multiple specific targets:
```shell
# Run from your top-level build directory (e.g., <aerial_sdk>/build)
# This command attempts to build all listed targets:
cmake --build . --target cellAssociateTest --target channInputTest --target multiCellMuMimoScheduler # ... and so on for other targets
```

Alternatively, to build all of the above example targets in one shot, you can use the `cumac_examples` target:
```shell
# Run from your top-level build directory (e.g., <aerial_sdk>/build)
cmake --build . --target cumac_examples
```

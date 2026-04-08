# NVIDIA Aerial SDK Container Configuration

This directory contains Docker container configurations for building and running the NVIDIA Aerial SDK. The containers use [HPCCM](https://github.com/NVIDIA/hpc-container-maker) (HPC Container Maker) to generate Dockerfiles with consistent, reproducible builds.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Container Types](#container-types)
- [Building Containers](#building-containers)
- [Running Containers](#running-containers)
- [Dependencies and Patches](#dependencies-and-patches)

## Prerequisites

### Install Docker and NVIDIA Container Toolkit

```bash
# Install Docker
curl https://get.docker.com | sh
sudo systemctl start docker && sudo systemctl enable docker

# Add your user to docker group
sudo usermod -aG docker $USER
newgrp docker

# Install NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

### Install GDRCopy

GDRCopy is required for GPU-direct memory access:

- [GDRCopy](https://github.com/NVIDIA/gdrcopy)

```bash
# Check the installed GDRCopy driver version
apt list --installed | grep gdrdrv-dkms

# Remove the driver, if you have the older version installed.
sudo apt purge gdrdrv-dkms
sudo apt autoremove

# Detect system architecture and install the matching GDRCopy driver
ARCH=$(uname -m)
case $ARCH in
  x86_64)
    wget https://developer.download.nvidia.com/compute/redist/gdrcopy/CUDA%2012.8/ubuntu22_04/x64/gdrdrv-dkms_2.5.1-1_amd64.Ubuntu22_04.deb
    sudo dpkg -i gdrdrv-dkms_2.5.1-1_amd64.Ubuntu22_04.deb
    ;;
  aarch64)
    wget https://developer.download.nvidia.com/compute/redist/gdrcopy/CUDA%2012.8/ubuntu22_04/aarch64/gdrdrv-dkms_2.5.1-1_arm64.Ubuntu22_04.deb
    sudo dpkg -i gdrdrv-dkms_2.5.1-1_arm64.Ubuntu22_04.deb
    ;;
  *)
    echo "Unsupported architecture: $ARCH"
    exit 1
    ;;
esac
```

### Install HPCCM

- [HPCCM Documentation](https://github.com/NVIDIA/hpc-container-maker)

```bash
pip3 install hpccm
```

### Verify GPU and Driver

```bash
nvidia-smi
```

## Container Types

This directory supports building multiple container types:

| Container Type | Script | Purpose |
|---------------|--------|---------|
| **aerial_base** | `build_base.sh` | Base container with CUDA, system libraries, and common dependencies |
| **aerial-cuda-accelerated-ran** | `build_devel.sh` | Development container for building and testing Aerial SDK |

## Building Containers

To build the Aerial development container, follow these instructions on the platform you will run Aerial on.

```bash
export AERIAL_VERSION_TAG=<custom_tag>
./build_base.sh
./build_devel.sh
```

- **`AERIAL_VERSION_TAG`**: Version tag for the container images (any custom tag to ensure the default tag is not overwritten)

This builds the base container, `aerial_base`, and the development container, `aerial-cuda-accelerated-ran`. The base container must be built first, as the development container depends on it.

When built on an x86_64 machine (e.g. the R750), then the container will be amd64 based. When built on an arm machine (e.g. the SMC), then the container will be arm64 based.

## Running Containers

### Interactive Development Container

Now that the containers are built, start an interactive bash shell in the development container:

```bash
export AERIAL_VERSION_TAG=<custom_tag>
./run_aerial.sh
```

You may modify the `setup.sh` script with the custom_tag once it is built to persist AERIAL_VERSION_TAG, instead of exporting it.

The source code from the repository is mounted at `/opt/nvidia/cuBB` inside the container. Changes made inside the container are available on the host and vice versa.

### Detached Mode

Run the container in the background:

```bash
./run_aerial.sh --detached
```

Attach to the running container:

```bash
./attach_aerial.sh
# or
docker exec -it c_aerial_$USER /bin/bash
```

Stop the detached container:

```bash
docker stop c_aerial_$USER
```

### Run Command in Container

Execute a specific command when container is not running:

```bash
./run_aerial.sh testBenches/phase4_test_scripts/build_aerial_sdk.sh
```

Execute a specific command when the container is running:

```bash
docker exec -it c_aerial_$USER /bin/bash -c 'echo CMD'
```

### Custom Prompt

Show the host SDK path in the container prompt:

```bash
./run_aerial.sh --modify-prompt
```

### Help

```bash
./run_aerial.sh --help
```

## Dependencies and Patches

### Python Dependencies

- `requirements.txt`: Common Python packages
- `requirements_x86_64.txt`: x86_64-specific packages
- `requirements_aarch64.txt`: ARM64-specific packages

### External GitHub Repositories

The development container builds several external repositories from GitHub directly into the container. Two of these repositories require patches for compatibility with the Aerial SDK:

The `patches/` directory contains modifications to these external dependencies:

- **mimalloc.patch**: Memory allocator optimizations and customizations
- **fmtlog.patch**: Logging library modifications for Aerial SDK

These patches are automatically applied during the container build process via HPCCM recipes.

## Building Aerial SDK from Inside Container

Once inside the development container, build the Aerial SDK:

```bash
# Container automatically mounts the host SDK at /opt/nvidia/cuBB
./testBenches/phase4_test_scripts/build_aerial_sdk.sh
```

# NVIDIA Aerial SDK Installation

## Quick start

First-time provisioning must be run in stages because the required reboot or
power cycle interrupts the installation. Do not expect `make all` to complete
in one invocation.

### DGX Spark

1. Install `make` if needed, run `make prepare` to install/select the platform kernel and GRUB configuration, then reboot:

   ```bash
   sudo apt update && sudo apt install -y build-essential
   make prepare
   sudo reboot
   ```

2. Install DOCA and the NIC firmware. For a PTP master node, replace the
   command below with `make install PTP_ROLE=master`:

   ```bash
   make install
   ```

   On a clean first-time installation, the installer normally stops with Make
   status 2 and explicitly requests an ordinary reboot so the NIC firmware
   update can take effect. This is an expected handoff when that prompt is
   present. Reboot, then continue:

   ```bash
   sudo reboot
   ```

3. Resume and complete the installation. For a PTP master node, replace the
   command below with `make install PTP_ROLE=master`:

   ```bash
   make install
   ```

4. Build and start the software:

   ```bash
   RU_MAC=<yourWncMac> make build
   RU_MAC=<yourWncMac> make start_all
   ```

### GH200

1. Install `make` if needed, run `make prepare` to install/select the platform kernel and GRUB configuration, then reboot:

   ```bash
   sudo apt update && sudo apt install -y build-essential
   make prepare
   sudo reboot
   ```

2. Install the BFB on both BF3 devices. For a PTP master node, replace the
   command below with `make install PTP_ROLE=master`:

   ```bash
   make install
   ```

   After installing a BFB, the installer stops with Make status 2 and explicitly
   requests a full cold power cycle. This is a normal handoff at this stage when
   that prompt is present. Perform the requested cold cycle, then power the host
   back on. A soft reboot is not sufficient.

3. Resume the installation to apply the BF3 NIC settings. For a PTP master
   node, replace the command below with `make install PTP_ROLE=master`:

   ```bash
   make install
   ```

   If the installer changes the BF3 `mlxconfig` settings, it stops with Make
   status 2 and explicitly requests another full cold power cycle. This is a
   normal handoff when that prompt is present. Perform the requested cold cycle,
   then power the host back on.

4. Resume and complete the installation. For a PTP master node, replace the
   command below with `make install PTP_ROLE=master`:

   ```bash
   make install
   ```

5. Build and start the software:

   ```bash
   RU_MAC=<yourWncMac> make build
   RU_MAC=<yourWncMac> make start_all
   ```

### MGX ARC Pro

MGX ARC Pro provisioning configures only `CX8-0` (`0002:03:00.0` and
`0002:03:00.1`) as `aerial00` and `aerial01`. `CX8-1`/`CX8-2` and LLS-C1
GNSS/DPLL/Telco-IO provisioning are outside this workflow.

1. Install `make` if needed, run `make prepare` to install/select the platform kernel and GRUB configuration, then reboot:

   ```bash
   sudo apt update && sudo apt install -y build-essential
   make prepare
   sudo reboot
   ```

2. Install the driver stack and verify or update all three ConnectX-8 devices.
   For a PTP master node, replace the command below with
   `make install PTP_ROLE=master`:

   ```bash
   make install
   ```

   If the installer updates the ConnectX-8 firmware, it stops with Make status
   2 and explicitly requests a full BMC/host cold power cycle. This is a normal
   handoff when that prompt is present. Perform the requested cold cycle, then
   power the host back on. A soft reboot is not sufficient.

3. Resume the installation to verify the active firmware and apply the required
   ConnectX-8 NIC settings. For a PTP master node, replace the command below
   with `make install PTP_ROLE=master`:

   ```bash
   make install
   ```

   If the installer changes the NIC settings, it stops with Make status 2 and
   explicitly requests another full BMC/host cold power cycle. This is a normal
   handoff when that prompt is present. Perform the requested cold cycle, then
   power the host back on.

4. After a second cold cycle, resume and complete the installation. For a PTP
   master node, replace the command below with
   `make install PTP_ROLE=master`:

   ```bash
   make install
   ```

### Dell R750

The R750 follows the same two-cold-cycle sequence as GH200, but provisions one
BF3 device instead of two.

1. Install `make` if needed, run `make prepare` to install/select the platform kernel and GRUB configuration, then reboot:

   ```bash
   sudo apt update && sudo apt install -y build-essential
   make prepare
   sudo reboot
   ```

2. Install the BFB on the BF3 device. For a PTP master node, replace the
   command below with `make install PTP_ROLE=master`:

   ```bash
   make install
   ```

   After installing the BFB, the installer stops with Make status 2 and
   explicitly requests a full cold power cycle. This is a normal handoff at
   this stage when that prompt is present. Perform the requested cold cycle,
   then power the host back on. A soft reboot is not sufficient.

3. Resume the installation to apply the BF3 NIC settings. For a PTP master
   node, replace the command below with `make install PTP_ROLE=master`:

   ```bash
   make install
   ```

   If the installer changes the BF3 `mlxconfig` settings, it stops with Make
   status 2 and explicitly requests another full cold power cycle. This is a
   normal handoff when that prompt is present. Perform the requested cold cycle,
   then power the host back on.

4. Resume and complete the installation. For a PTP master node, replace the
   command below with `make install PTP_ROLE=master`:

   ```bash
   make install
   ```

---

## Make targets

| Target | Function |
|--------|----------|
| **prepare** | Installs and selects the documented platform kernel and GRUB command line. **Reboot after this**, then run `make install`. |
| **all** | Full flow: `install` -> `build` -> `start_all`. Installs drivers and services, builds Aerial SDK and OAI, then starts gNB and CN5G. Use with `RU_MAC=<mac>`. |
| **install** | Resumable host provisioning: NIC stack/firmware, GPU software where applicable, network names, PTP, and system services. |
| **net** | Sets up network interfaces (e.g. `aerial0x`). |
| **kernel** | Installs the Aerial CUDA kernel. Reboot required after this if the kernel was updated. |
| **drivers** | Installs DOCA, OFED, and GPU drivers. Prompts for confirmation; ensure PTP, VLAN, RU peer MAC, and Docker login are configured first. |
| **services** | Installs PTP and system services. |
| **build** | Runs `build_aerial` and `build_oai`. |
| **build_aerial** | Builds the Aerial SDK (runs `quickstart-aerial.sh`). Use `PROFILE=` or `BUILD_PRESET`/`BUILD_CMAKE_FLAGS` for variants. |
| **build_oai** | Builds OAI (runs `quickstart-oai.sh --build-only`). Use `RU_MAC=<mac>` if needed. |
| **start_gnb** | Starts the gNB. Set `RU_MAC=<mac>` when invoking. |
| **start_cn** | Starts CN5G. |
| **start_all** | Starts both gNB and CN5G. Set `RU_MAC=<mac>` when invoking. |
| **check** | Strictly checks kernel, driver/network state, exact NIC firmware, services, and role-appropriate PTP status. |
| **help** | Prints target list and usage. |
| **clean** | Phony target (see `make help` for current behavior). |

## Options

- **DRYRUN=1** - Show commands without executing (e.g. `make all DRYRUN=1`).
- **VERBOSE=1** - Print commands before executing.
- **PTP_ROLE=client|master** - Override the default client role in `make install`.
- **PTP_INTERFACE=name** - Override the platform's PTP port. Omit this option to use the default `aerial00` interface.
- **ASSUME_YES=1** - Skip the driver-install confirmation prompt (for automation).
- **RU_MAC=aa:bb:cc:dd:ee:ff** - Set RU MAC address for gNB/OAI (required for `all`, `start_gnb`, `start_all`, and when building OAI with a specific MAC).
- **PROFILE=name** - Aerial build profile file: `oai.conf`, `fapi_10_02.conf`, `fapi_10_04.conf`, or a custom `<name>.conf` (see **Build profiles** below).
- **BUILD_PRESET=preset** - Override Aerial preset: `perf`, `10_02`, `10_04`, `10_04_TM`, `10_04_low_memory`.
- **BUILD_CMAKE_FLAGS="..."** - Override CMake flags for the Aerial build.

## Build profiles (Aerial configuration)

The Aerial build can use different configurations (OAI L2+ default, FAPI 10_02 only, or FAPI 10.04). Use either **make targets** or a **configuration profile**.

- **Make targets:**
  - `make build_aerial` — default (FAPI 10_02 + -DSCF_FAPI_10_04_SRS=ON).
  - `PROFILE=fapi_10_02.conf make build_aerial` — FAPI 10_02 only
  - `PROFILE=fapi_10_04.conf make build_aerial` — FAPI 10.04 (SCF_FAPI_10_04=ON).

- **Profile variable:**  
  Profiles are defined in `install/cmake-profiles/<name>.conf` (each sets `BUILD_PRESET` and `PROFILE_CMAKE_FLAGS`). See `install/cmake-profiles/README.md` for adding custom profiles.

## Examples

```bash
make prepare && sudo reboot
RU_MAC=e8:c7:cf:ac:58:32 make all
```

## Supported provisioning platforms

| Platform | Provisioning behavior |
|---|---|
| DGX Spark | Kernel `6.17.0-1018-nvidia`, 32 x 1-GiB hugepages, DOCA/OFED 3.3/26.01, ConnectX-7 firmware `28.47.1088`, GPU/CUDA, and PTP client on `aerial00` by default. |
| Grace Hopper MGX | Kernel `6.17.0-1018-nvidia-64k`, DOCA/OFED 3.3/26.01 for Ubuntu 24.04, GDRCopy 2.6-1, BF3 BFB installation through rshim0 and rshim1, MIG disabled, and PTP client on `aerial00` by default. |
| Dell PowerEdge R750 | Kernel `6.8.0-1058-nvidia-lowlatency`, 16 x 1-GiB hugepages, Ubuntu inbox mlx5, rshim/MFT, one BF3 BFB through rshim0, BF3 PSID `MT_0000000884`, `aerial00/01`, and PTP client on `aerial00` by default. GPU/A100X provisioning is outside this platform profile. |
| MGX ARC Pro | Kernel `6.17.0-1018-nvidia-64k`, 48 x 512-MiB hugepages, three ConnectX-8 devices targeting firmware `40.97.5452`, `aerial00/01` on ConnectX-8 adapter 0, and PTP client on `aerial00` by default. |

Host provisioning starts after the platform BIOS, Ubuntu installation, disk
layout, users, and management connectivity are ready.

## Scripts

The make targets run executable scripts in this directory (e.g. `install_aerial_kernel.sh`, `install_drivers.sh`, `install_services.sh`, `setup_net_ifs.sh`, `quickstart-aerial.sh`, `quickstart-oai.sh`). You can run any of these scripts directly. Each script supports a `-h` or `--help` option for usage and options.

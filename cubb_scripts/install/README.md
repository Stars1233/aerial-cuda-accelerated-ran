# NVIDIA Aerial SDK Installation

## Quick start

1. **Prepare and reboot** (sets up network interfaces and installs the Aerial CUDA kernel; a reboot is required to load the new kernel):

   ```bash
   sudo apt update && sudo apt install -y build-essential #Required on systems where make isn't part of the base install
   make prepare && sudo reboot
   ```


2. **After reboot**, run the full installation with your RU MAC address:

   ```bash
   RU_MAC=<yourWncMac> make all
   ```

Replace `<yourWncMac>` with your WNC RU MAC address (e.g. `e8:c7:cf:ac:58:32`).
If using a different RU, you may have to change PCP and VLAN in cuphycontroller_P5G_WNC_DGX.yaml before running `make all` or `make start` on the DGX-Spark

---

## Make targets

| Target | Function |
|--------|----------|
| **prepare** | Runs `net` and `kernel`: sets up network interfaces and installs the Aerial CUDA kernel. **Reboot after this**, then run `make all`. |
| **all** | Full flow: `install` -> `build` -> `start_all`. Installs drivers and services, builds Aerial SDK and OAI, then starts gNB and CN5G. Use with `RU_MAC=<mac>`. |
| **install** | Runs `drivers` and `services`: installs DOCA, OFED, GPU drivers, PTP, and system services. |
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
| **check** | Checks installation status (kernel and services). |
| **help** | Prints target list and usage. |
| **clean** | Phony target (see `make help` for current behavior). |

## Options

- **DRYRUN=1** - Show commands without executing (e.g. `make all DRYRUN=1`).
- **VERBOSE=1** - Print commands before executing.
- **RU_MAC=aa:bb:cc:dd:ee:ff** - Set RU MAC address for gNB/OAI (required for `all`, `start_gnb`, `start_all`, and when building OAI with a specific MAC).
- **PROFILE=name** - Aerial build profile file: `oai.conf`, `fapi_10_02.conf`, `fapi_10_04.conf`, or a custom `<name>.conf` (see **Build profiles** below).
- **BUILD_PRESET=preset** - Override Aerial preset: `perf`, `10_02`, `10_04`, `10_04_32dl`.
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

## Scripts

The make targets run executable scripts in this directory (e.g. `install_aerial_kernel.sh`, `install_drivers.sh`, `install_services.sh`, `setup_net_ifs.sh`, `quickstart-aerial.sh`, `quickstart-oai.sh`). You can run any of these scripts directly. Each script supports a `-h` or `--help` option for usage and options.

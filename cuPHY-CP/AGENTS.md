# cuPHY-CP - Control Plane

## Purpose

cuPHY-CP is the integrated control plane for NVIDIA Aerial's 5G L1 software
stack, bridging the 5G MAC layer to the GPU-accelerated cuPHY signal-processing
engine via three main components:

- **cuphydriver**: the PHY driver; manages GPU memory allocation, per-slot GPU
  task scheduling, IQ compression configuration, UL packet ordering, fronthaul
  I/O integration (aerial-fh-driver), and cell/worker lifecycle.
- **L2 adapter** (`cuphyl2adapter` / `scfl2adapter`): terminates the SCF FAPI
  MAC-PHY interface from L2 over nvIPC (which provides transport and memory
  buffers); handles both data path (TX_DATA / PDU transport) and control path
  (DL_TTI / UL_TTI scheduling); translates to the internal cuPHY API and the
  ORAN C-Plane.
- **Fronthaul driver** (`aerial-fh-driver`): implements the ORAN fronthaul
  interface (O-DU ↔ O-RU) using DOCA APIs to interface with the Mellanox NIC.

Also contains OAM and supporting libraries for slot commands and performance
metrics.

**Test framework (not production runtime):** `testMac` simulates the L2 MAC
layer; `ru-emulator` simulates the O-RU - both are for end-to-end Aerial
system validation only.

Key executables:

- `cuphycontroller_scf` (`cuphycontroller/examples/`): production cuPHY controller; full OAM, CUDA, DPDK, and PTP integration; the deployment artifact.
- `l2_adapter_cuphycontroller_scf` (`scfl2adapter/scf_app/cuphycontroller/`): lightweight scfl2adapter integration app; drives PHY via SCF FAPI without OAM/DPDK/CUDA profiling.

## Data flow (high level)

Per-slot tick: the `tti_gen` timer thread (`cuphyl2adapter/lib/nvPHY/nv_tick_generator.cpp`) fires each slot into `PHY_module::tick_received`, which sends SLOT.indication upstream to L2.
L2 replies with FAPI DL_TTI/UL_TTI/TX_DATA, dispatched by the SCF adapter `phy::` handlers (`scfl2adapter/lib/scf_5g_fapi/`), which translate the FAPI MAC-PHY interface to the internal cuPHY API and the ORAN C-Plane.
Those populate per-cell `slot_command` objects (`gt_common_libs/slot_command/.../slot_command.hpp`).
`PHY_module::process_phy_commands` hands each to the driver via `PHYDriverProxy::l1_enqueue_phy_work` -> `cuphydriver`.
The driver walks `sc->cells`, builds per-cell channel aggregates (`Phy*Aggr`) and schedules GPU L1 work + fronthaul TX (`aerial-fh-driver`).

## Key Files (read these first)

- `cuphydriver/include/cuphydriver_api.hpp` - PHY driver public API
- `cuphydriver/include/context.hpp` - driver context / state management
- `scfl2adapter/lib/scf_5g_fapi/scf_5g_fapi.cpp` - SCF FAPI adapter entry point
- `gt_common_libs/slot_command/include/slot_command/slot_command.hpp` - slot
  command model
- `cuphycontroller/include` - controller wiring

## Invariants (never violate)

- `phydriver_handle` and `phydriverwrk_handle` are opaque `void*` handles. Never
  cast or dereference them outside the driver implementation.
- `l1_task_work_fn_t` callbacks receive the cell range
  `[first_cell, first_cell + num_cells)` and must not access cells outside that
  range.
- cuPHY-CP accesses all cuPHY functionality exclusively through the `libcuphy`
  C ABI (`cuphy_api.h`). Never include cuPHY internal headers directly.
- When the cuPHY API changes (e.g. a struct gains a new field), explicitly
  initialize the new member to a stable value at every call site in cuPHY-CP;
  do not rely on implicit zero-initialization or aggregation defaults.

## Patterns Specific to This Area

- Opaque-handle pattern: adapters hold `void*` (e.g. `PHYDriverProxy`), not the
  concrete `PhyDriverCtx*`. Adding `l1_*` functions to the C-style API is the
  correct extension point.
- Interface classes (`IMPlaneConfigProvider`, `IBeamformingProvider`, etc.)
  follow C++ Core Guidelines C.67: suppress public copy/move, `protected`
  default constructor.

## Testing

Run inside the Aerial container (launch with `cuPHY-CP/container/run_aerial.sh`). cuPHY-CP
unit/integration tests carry the CTest label `cuphy-cp`. Configure a test
preset, build, then run the lane:

```bash
cmake --preset cicd-test-x86
cmake --build build-cicd-test-x86 -- -j32
ctest --preset cicd-test-x86        # runs the `cuphy-cp`-labelled tests
```

Sanitizer lanes (CPU-side: ASan/TSan) reuse the same label (`ctest --preset asan-x86-debug` /
`tsan-x86-debug`). FAPI-macro coverage varies by preset: the plain
`cicd-test-x86` / `-arm` presets build the SCF FAPI 10.04 macros OFF, but the
perf preset (`cicd-test-arm-perf`) sets `SCF_FAPI_10_04=ON`. The perf preset is
ARM-only - there is no x86 perf preset - so whether `#ifdef`-gated FAPI tests
are exercised depends on the lane, and locally-green can differ from a given CI
preset - see `scfl2adapter/lib/scf_5g_fapi/AGENTS.md` for the gating.

## Subdirectory guides

- `scfl2adapter/lib/scf_5g_fapi/AGENTS.md`: SCF 5G FAPI library build, error
  handling (`SlotParseResult`), packed-struct/TLV invariants, C++ patterns, and
  the `test_scfl2adapter` test binary.
- `data_lake/AGENTS.md`: telemetry capture + E3/dApp streaming; `e3::StreamType`
  and `SharedMemoryHeader` are a wire/ABI contract with an out-of-tree consumer
  and no in-tree test.

# cuPHY

GPU-accelerated 5G NR physical layer (PHY) library. Implements the 3GPP
uplink and downlink channel pipelines (PUSCH, PDSCH, PUCCH F0-F4, PDCCH,
PRACH, SRS, CSI-RS, SSB/PBCH, BFW) as CUDA kernels, including LDPC/polar coding, channel
estimation, equalization, rate matching, modulation mapping, and UCI on
PUSCH. Exposes a C API built around per-cell static/quasi-static state
handles with explicit CREATE/MODIFY/DESTROY lifecycle, and is consumed by
higher-level cuPHY-CP driver code.

## Data flow (high level)

This describes the full-stack cuPHY-CP integration path (as in end-to-end system tests). For standalone cuPHY usage, call the functions in `cuphy_api.h` directly and skip to `## Read these first`.

Entry: the cuPHY-CP driver calls the public C API `cuphySetup<Channel>` / `cuphyRun<Channel>` (e.g. `cuphySetupPdschTx`) in `cuphy_api.h`.
Payload: a per-slot `cuphy<Channel>DynPrms_t` struct (CUDA stream, `procModeBmsk`, cell-group config, I/O tensor pointers).
Populated by the cuphydriver aggregators (`cuPHY-CP/cuphydriver/src/{downlink,uplink}/phy*_aggr.cpp`) from L2/FAPI slot commands.
Consumer: one or more pipeline objects per channel type (e.g., `cuphyPdschTx`/`cuphyPuschRx` in `src/cuphy_channels/`); each owns its kernel sequence and its own `CUgraph`/`CUgraphExec` (always created at construction; no separate GraphManager type).
Execution: CUDA streams vs. CUDA graphs - controlled by `procModeBmsk`; CUDA graphs are favored for performance reasons. Using graphs vs. streams also affects the type of setup work that happens.

## Read these first

- `src/cuphy/cuphy_api.h` - public API: per-channel entry points (`cuphyCreate*`, `cuphySetup*`, `cuphyRun*`, `cuphyDestroy*`).
- `src/cuphy/cuphy.h` - types, constants, and sub-component C APIs.
- `src/cuphy/cuphy_internal.h` - internal helpers and shared definitions.
- `src/cuphy/cuphy_context.hpp` - context / state handle management.
- `CMakeLists.txt` - build configuration (requires a toolchain file; the
  toolchain dirs live under `cuPHY/cmake/toolchains/`, e.g. `x86-64`,
  `grace-cross` - or just use a preset from the repo-root `CMakePresets.json`).

## Layout

- `src/cuphy/` - public headers (`cuphy_api.h`, `cuphy.h`, `cuphy_internal.h`, `cuphy_context.hpp`).
- `src/cuphy_channels/` - per-channel pipeline objects (`cuphyPdschTx`, `cuphyPuschRx`, etc.).
- `test/` - unit and gtest coverage for the core library.
- `examples/` - reference and test-vector implementations for validation; not production runtime code.

## API Lifecycle

Channel pipelines follow `cuphyCreate*` → `cuphySetup*` → `cuphyRun*` → `cuphyDestroy*`:

- `cuphyCreate*` / `cuphyDestroy*`: once per pipeline lifetime; allocate/free resources. Not on the critical path.
- `cuphySetup*` / `cuphyRun*`: called per slot; **critical path with real-time constraints**. `setup` and `run` may execute many times between `create` and `destroy`.
- The caller must ensure the pipeline is **inactive** before calling `cuphySetup*` or `cuphyRun*` - both mutate internal state; calling `setup` while `run` is active on the same pipeline object instance is explicitly unsupported (cuphydriver uses multiple instances in round-robin; `setup` on one instance while another is running is valid).
- Exception: the PRACH API is partially implemented and uses `cuphyStateUpdateType_t` state transitions rather than this pattern (see Invariants).

## Invariants - never violate

- `cuphyStateUpdateType_t` state transitions (`CREATE_STATIC` → `CREATE_QUASI_STATIC` → `MODIFY*` → `DESTROY`) apply to the partially-implemented PRACH API only; most channel pipelines use the `cuphyCreate*`/`cuphySetup*`/`cuphyRun*`/`cuphyDestroy*` lifecycle instead. Mixing state-type values in PRACH causes undefined memory behavior.
- All CUDA kernels are enqueued onto caller-supplied CUDA streams. The
  library never synchronizes globally - callers own stream lifetime and
  must not destroy a stream while kernels are in flight.
- ABI versioning is not required. All callers (cuPHY-CP) live in the same
  source tree; there is no external ABI contract to preserve.
- When an API struct gains a new field, update every caller in cuPHY-CP to
  explicitly initialize that field to a stable value. Agents tend to miss
  cross-tree callers - always grep for all call sites before committing.

## Build

Target: `cuphy`. See root `AGENTS.md` for the container + preset build mechanics
(toolchain, `build-<preset>` dir, `CCACHE_DISABLE`); the only delta here is the
target name:

```bash
cmake --preset <preset>            # e.g. minimal-x86 or minimal-arm; see CMakePresets.json
cmake --build build-<preset> --target cuphy -- -j32
```

## Patterns specific to this area

- C API design: opaque per-cell handles, explicit lifecycle transitions
  rather than RAII across the API boundary.
- Stream-oriented async execution; no implicit synchronization (e.g., no
  `cudaDeviceSynchronize`). Treat any global sync added inside the library
  as a defect.
- CUDA code in this library is latency- and determinism-sensitive: prefer
  deterministic kernels, preload where possible, and keep an explicit
  stream/sync policy (see Invariants; avoid `cudaStreamSynchronize` inside
  channel functions on the critical path).
- C++ helpers around the C core follow the repo-root C++20/CMake conventions
  (see root `AGENTS.md`) - notably caller-owned streams and determinism above.

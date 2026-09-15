# cuMAC - CUDA MAC Scheduler

GPU-accelerated 5G NR L2 scheduling library and simulation/test framework for
multi-cell operation. Its 4T4R path implements cell association, PF/RR UE
selection, type-0/type-1 PRG allocation, optional SVD precoding, layer
selection, and LUT/OLLA-based MCS selection; its current 64T64R DL/type-1
simulation path adds MU-MIMO UE sorting/grouping and RZF beamforming.
Additional CUDA modules provide SINR calculation, multi-cell SRS scheduling,
stateful DL/UL PFM sorting, and L2 MU-MIMO user pairing, with control-plane
integration defined by the `lib/cumac_msg` NVIPC ABI. CUDA is the primary
execution path, while many algorithms have CPU reference implementations;
`cumacSubcontext::in_GPU` selects CPU versus GPU only for the HDF5-driven 4T4R
test wrapper. TensorRT/ONNX functionality is used by ML demonstrations rather
than the core scheduling stages, and HDF5 supports test-vector loading/replay,
result logging, and validation rather than the live scheduler interface.

Build target: `cumac`.

## Data flow (high level)

Entry: no monolithic entry - the caller instantiates a per-module scheduler class (`cellAssociation` / `multiCellScheduler` / `roundRobinScheduler` / `singleCellScheduler` / `muMimoUserPairing`; surface aggregated by `src/cumac.h`) and calls `setup()` then `run(strm)`.
`setup(cumacCellGrpPrms*, cumacSimParam*, stream)` binds coordinated-cell-group inputs (per-UE status, cell params, cellAssoc/UE-pairing buffers - all power-of-two sized), defined in `src/api.h`.
`run(strm)` enqueues the GPU scheduling kernels on the caller's stream (`cuLaunchKernel`; any CUDA error is fatal); no internal sync.
Results land in device buffers (`cumacCellGrpPrms::cellAssoc`, `cumacSchdSol`), read via getters (`getCellAssociaResGpu()`).
External-sync boundary: the caller owns `cudaStreamSynchronize(strm)` before reading results back.

## Scheduler pipelines

- 4T4R PF path: channel/SINR update -> UE selection -> PRG allocation -> layer
  selection -> MCS selection. The RR and single-cell schedulers replace selected
  stages. `examples/testBench.cpp` and
  `examples/multiCellSchedulerUeSelection/main.cpp` show the supported ordering.
- 64T64R simulation path: `multiCellMuUeSort` -> `multiCellMuUeGrp` ->
  `multiCellBeamform` -> `mcsSelectionLUT`, followed by PHY abstraction. See
  `examples/multiCellMuMimoScheduler/main.cpp`.

## Read first

- `src/api.h` - public API surface and the sizing constants that govern every
  array dimension and CUDA kernel launch config.
- `src/cumac.h` - core scheduler types and entry points.
- `src/cumacSubcontext.h` / `src/cumacSubcontext.cpp` - per-context scheduler
  state; GPU vs. CPU path selection and lifecycle.
- `CMakeLists.txt` (subsystem root) and `src/CMakeLists.txt` - build wiring,
  gating, and standalone `add_subdirectory` logic for the `cumac` target.
- `README.md` - standalone-build prerequisites (CMake 3.18+, CUDA 12+) and
  dependencies, relevant when building cuMAC outside the top-level cuBB tree.

## Invariants - never violate

- **Sizing constants are authoritative.** Array sizes are governed by constants
  in `api.h`: `maxNumCoorCells_=20`, `maxNumActUePerCell_=1024`,
  `maxNumUegPerCell_=128`, `maxNumSchdUePerCellTTI_=16`, `maxNumBsAnt_=64`.
  `maxNumActUePerCell_` (max 1024), `maxNumUegPerCell_`, and
  `maxNumUeForGrpPerCell_` (both max 128) must stay a **power of two** within
  that max - a non-power-of-two like 1000 obeys the cap yet still corrupts
  kernel indexing. The rest are hard upper bounds. Never change any of these
  without updating every dependent CUDA kernel launch configuration.
- **Sentinel values.** Invalid UE IDs are `0xFFFF` in 16-bit ID fields and
  `0xFFFFFFFF` in 32-bit ID arrays; invalid subband IDs are `-1`. Callers must
  check these sentinels before dereferencing per-UE data.
- **External synchronization required.** The `cumacSubcontext` constructor does
  not serialize itself - the caller must serialize concurrent access. The
  GPU/CPU scheduler path is fixed at construction time via the `in_GPU` flag
  and cannot be changed at runtime.
- **CUDA errors are fatal, not recoverable.** `CUDA_CHECK_ERR` and
  `CUDA_CHECK_RES` call `exit(EXIT_FAILURE)` immediately. Do not wrap cuMAC API
  calls in `catch` blocks expecting exceptions - there are none.

## Patterns specific to this area

- Keep the CPU reference/comparison path in sync with the GPU kernels: an
  algorithmic change to a kernel must also land in its CPU reference, unless the
  change is demonstrably launch-only or performance-only.
- New scheduling work follows the existing per-context model: allocate sizing
  from the `api.h` constants, choose the GPU or CPU path at construction, and
  keep kernel launch dimensions in lockstep with those constants.
- Test vectors flow through HDF5 (ingest and export); use that path for
  fixtures rather than hand-built buffers.
- Two antenna regimes (4T4R SU-MIMO, 64T64R MU-MIMO) share the API; verify any
  change against both rather than assuming a single configuration.

## Testing

cuMAC has its own CTest lane (`test_cellAssociation`), added from `test/` when
that directory is present - **not** gated by `ENABLE_TESTS`, unlike the rest of
the tree. Build, then run from the build dir:

```bash
ctest -R test_cellAssociation
```

See the repository root `AGENTS.md` for cuBB-wide build, C++20, and
contribution conventions.

# cuMAC-CP - cuMAC Control-Plane Application

Control-plane process that puts the `cuMAC` GPU scheduler on the wire. `cumac_cp`
is a standalone executable that receives MAC scheduling requests from an external
L2/MAC over nvIPC (shared-memory IPC), runs the cuMAC scheduler on the GPU, and
returns the results. Where `cuMAC/` is the scheduling library, `cuMAC-CP/` is the
service that hosts it, owns threading/affinity, IPC transport, config, and the
L2 integration contract.

Executable target: `cumac_cp`. Built with `cuMAC/` under the shared
`ENABLE_CUMAC` option (default ON); the top-level CMake adds this subdirectory
only when both `ENABLE_CUMAC` is set and `cuMAC-CP/` exists.

Invoke: `cumac_cp` (no arguments needed for default config; it picks up `launch_pattern` from the working directory). Positional args (`Fxx`, `xC`, `DL/UL`) select a launch-pattern variant; optional flags: `[--channels <names>]`, `[--cells <mask>]`, `[--mode <n>]`, `[--config <path>]`
(e.g. `cumac_cp F08 2C --channels PDSCH+PDCCH_DL+PDCCH_UL+PBCH`).

## Data flow (high level)

Entry: the nvIPC recv thread `cumac_receiver_thread_func` -> epoll -> `cumac_receiver::on_msg` -> `handle_slot_msg` (`src/msg_recv.cpp`, `src/cumac_cp_handler.cpp`).
Enqueue: `handle_slot_msg` allocs a `cumac_task` from `nv::lock_free_ring_pool<cumac_task>`, enqueues it, and `sem_post`s the shared `task_sem`.
GPU worker: `cumac_worker_thread_func` (config-driven affinity/priority) `sem_wait`s, dequeues the task, runs `setup()`/`run()`/`callback()`.
Scheduler: `cumac_task::run()` invokes the cuMAC scheduler on the GPU (`muMimoUserPairingGpu->run(...)`; `run_in_cpu` 0=GPU / 1=CPU / 2=mixed).
Response: `cumac_task::callback()` -> `send_sch_tti_response` per cell -> `transp.tx_send` back over nvIPC, then `rx_release`.

## Read first

- `src/main.cpp` - process entry, arg parsing, signal/backtrace setup, app bring-up.
- `src/cumac_cp_handler.{hpp,cpp}` - core message handling (CONFIG.req and
  per-slot scheduling), GPU buffer ownership, `muMimoUserPairing` integration.
- `src/cumac_task.{hpp,cpp}` - the unit of work handed to a GPU worker thread.
- `src/msg_recv.{hpp,cpp}` - receive thread and the worker-thread pool wiring.
- `src/cumac_cp_configs.{hpp,cpp}` - parsed view of `config/cumac_cp.yaml`.
- `config/cumac_cp.yaml` - runtime config (see the threading note below).
- `examples/L2IntegrationExample/` - a reference L2 integration; `cu_mac_api.h`
  defines the TTI request/response structs (`cumac_pfm_tti_req_t`,
  `cumac_pfm_tti_resp_t`, and the `PFM_*` info structs) an L2 exchanges over nvIPC.
  Builds the `l2_test` and `cumcp_test` example executables.

## Invariants - never violate

- **Threading and affinity are config-driven, not hardcoded.** A dedicated
  receive thread feeds a pool of GPU worker threads through a lock-free ring
  pool (`nv::lock_free_ring_pool<cumac_task>`) with semaphore signalling. Core
  affinities, scheduling priorities, worker cores, OAM address, GPU id, and CUDA
  block count all come from `config/cumac_cp.yaml`. The committed values are
  defaults tuned for one machine topology - treat them as tunables, never as
  fixed constants, and do not bake specific core numbers into code.
- **Execution mode is chosen by config.** `run_in_cpu` selects `0`=GPU,
  `1`=CPU, `2`=mixed. GPU is the real path; CPU mode is debug-only.
- **Slot pipelining depth is fixed.** `SCHED_SLOT_BUF_NUM` (=4) per-slot buffers
  pipeline scheduling; kernel and buffer indexing assume this depth. Keep any
  change in lockstep with the buffers it sizes.
- **cuMAC sizing constants still govern.** Array dimensions come from
  `cuMAC/src/api.h` (see `cuMAC/AGENTS.md`); this app allocates against them and
  must not exceed them.
- **The per-slot task set is a `taskBitMask` over `cumac_task_type_t`**
  (`cuMAC/lib/cumac_msg/cumac_msg.h`): the four 4T4R scheduler stages (UE
  selection, PRB allocation, layer selection, MCS selection) plus PFM sort and MU
  UE pairing, which are independent tasks rather than stages. The
  `CUMAC_TASK_MU_UE_GRP` bit invokes `muMimoUserPairing`, not `multiCellMuUeGrp`.

## Patterns specific to this area

- Messages are dispatched by `msg_id`/`cell_id`; `CONFIG.req` establishes cell
  config before per-slot scheduling requests flow. Validate cell/message ids
  before indexing per-cell state (the handler already rejects out-of-range ids).
- Test vectors flow through HDF5 via `cumac_cp_tv`; launch patterns live under
  `testVectors/multi-cell/`. Use that path for fixtures rather than hand-built
  buffers.
- The app links the cuMAC scheduler plus `cuphydriver`, `nvphy`, `nvipc`, and
  HDF5 - it is a full control-plane process, not a unit-testable library.

## Building and running

- There is no CTest lane here; `cuMAC-CP` produces the `cumac_cp` service plus
  the `l2_test` / `cumcp_test` integration examples. Exercise it by running the
  service against an L2 (or the example) over nvIPC.
- Note: this subsystem's CMake compiles at `-O0 -g` (unoptimized, with debug
  symbols) regardless of preset - expected, not a misconfiguration.

See the repository root `AGENTS.md` for cuBB-wide build, container, C++20, and
contribution conventions, and `cuMAC/AGENTS.md` for the scheduler library this
app hosts.

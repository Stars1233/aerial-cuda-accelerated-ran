# cuPHY-CP/testMAC - L2 MAC Emulator for Aerial System Validation

`testMAC` is a developer/test tool that emulates the L2 MAC layer to drive
the Aerial L1 (cuPHY + cuPHY-CP) in a controlled environment. It functions
as a **test harness, not a production scheduler**: it replays pre-captured
HDF5 test vectors (TVs) as SCF FAPI messages according to a static (YAML) or
dynamic (JSON) launch pattern. There is no live scheduling intelligence - the
"schedule" is a fixed per-slot table of TV files. For the real GPU-accelerated
MAC scheduler see `cuMAC/` and `cuMAC-CP/`.

GPU involvement is limited to IPC transport: testMAC optionally places FAPI
TB data directly into CUDA IPC buffers (`fapi_tb_loc: 3 -> GPU_DATA`) and
keeps a primary CUDA context alive for that purpose, not to run
signal-processing kernels.

testMAC sits at the **top of the DL spine** in a test rig: it produces
DL_TTI / UL_TTI / TX_DATA / UL_DCI / BFW FAPI over nvIPC to `cuphycontroller`,
and consumes the UL indications (CRC/RxData/UCI/RACH/SRS) coming back. It also
validates received PHY indications (`fapi_validate`) and measures throughput /
conformance (`--thrput`, `conf_test`).

## Entry point and invocation

Executable target: `test_mac`. Built as part of cuPHY-CP (requires
`scf_5g_fapi` and, optionally, `cumac`).

```bash
test_mac <Fxx> <xC> [--channels <names>] [--cells <mask>] [--mode <0|1>]
         [--config <yaml>] [--thrput] [--prebuild] [--no-validation]
         [--ru_emulator_host <host>]
```

Positional args (`Fxx`, `xC`, optional `DL/UL`) are concatenated into the
launch pattern filename (`launch_pattern_<args>.yaml` for mode 0,
`dynamic_pattern_<args>.json` for mode 1). Omit them to use the bare
`launch_pattern` default. `--channels` accepts `+`-joined names
(e.g. `PDSCH+PDCCH_DL+PDCCH_UL+PBCH`) or a numeric bitmask; 0 = all.

Wrapper script: `scripts/run_mac.sh <duration> <F08|F13|nrSim> <case>
--channels ...`. See `scripts/README.md` for the full cuMAC-CP orchestration
scripts.

## Read first

- `testMAC/main.cpp` - arg parsing, YAML/nvlog bootstrap, `PrimaryCtxGuard`
  (CUDA context), OAM init, optional cuMAC path, then `test_mac::start()/join()`.
- `testMAC/test_mac.{hpp,cpp}` - the `test_mac` app class: owns configs,
  launch pattern, FAPI handler, nvIPC transport; **defines the whole thread
  model** (scheduler + recv threads; recv spawns builder/OAM/worker threads).
- `testMAC/fapi_handler.hpp` - abstract base `fapi_handler`: scheduling
  contract, `cell_data_t`, `fapi_sched_t`, all timing constants (STT, FAPI TX
  deadline), per-cell state, SRS ChEst buffer bookkeeping.
- `testMAC/scf_fapi_handler.{hpp,cpp}` - concrete SCF-FAPI implementation:
  assembles CONFIG.req / DL_TTI / UL_TTI / TX_DATA / UL_DCI / BFW requests,
  TX-deadline scheduling, worker/builder threads, prebuild path.
- `testMAC/launch_pattern.{hpp,cpp}` - parses the launch-pattern YAML and
  per-channel HDF5 TVs into the 4-D `slot_pattern_t` schedule
  (slot x cell x fapi_group x channel).
- `testMAC/test_mac_configs.{hpp,cpp}` + `testMAC/test_mac_config.yaml` -
  the full config surface (thread cores/priorities, STT, deadlines, tb_loc,
  validation, transport). The YAML is the authoritative config reference.
- `testMAC/fapi_defines.hpp` - `channel_type_t` enum (PDSCH/PDCCH_DL/
  PDCCH_UL/PBCH/PUCCH/PUSCH/PRACH/CSI_RS/SRS/BFW_DL/BFW_UL; `CHANNEL_MAX=11`),
  `fapi_req_t`, TV dataset structs.
- `testMAC/tests/test_testmac_core.cpp` - unit test; also the clearest
  executable spec of the prebuild/getters contract and FAPI PDU wire layout.

## Data flow

`load_launch_pattern()` parses the YAML + HDF5 TVs into `launch_pattern`, then
constructs `scf_fapi_handler`. `start()` creates per-cell `phy_mac_transport`
(nvIPC, `transport.type: shm`, prefix `nvipc`), calls `set_transport()`, then
spawns threads (core/priority all config-driven):

- **`mac_sched`** - uplink and downlink scheduler: sends DL_TTI/UL_TTI/TX_DATA/UL_DCI/BFW
  per slot with STT / TX-deadline timing.
- **`mac_recv`** - uplink/IPC RX loop: `transport.rx_wait()` -> dispatch to
  worker threads or `_fapi_handler->on_msg()`; handles SLOT.indication and UL
  indications. Also spawns builder, OAM, and worker threads.
- **`fapi_builder`** (optional) - pre-builds FAPI messages off the scheduler
  thread; required when `fapi_tx_deadline_enable=1`.
- **`oam_thread`** - polls `CuphyOAM` (100 ms) for cell ctrl
  (start/stop/reconfig/init), FAPI-delay, and RNTI test-mode commands.
- **worker threads** (optional) - parallel message build/handle.

Startup sync: an optional gRPC barrier (`wait_for_ru_emulator`, port 50052)
synchronizes startup with `ru-emulator`.

Prebuild path (`--prebuild`): serializes CONFIG.req + one repeating slot TX
pattern into host buffers (`config_reqs`, `slot_msgs`) with no transport/IPC -
a transport-free way to exercise FAPI construction offline.

## Invariants - never violate

- **Downlink requires uplink.** `enable_downlink=true` with
  `enable_uplink=false` is rejected; the recv thread owns builder/OAM/worker
  spawning and IPC RX.
- **`load_launch_pattern()` before `prebuild` / `start()`.** `_fapi_handler`
  is null until the pattern loads; callers null-check it.
- **Prebuild is one-shot per handler.** A second `prebuild_downlink_messages()`
  returns -1 (`prebuild_downlink_done`). Unit test `tc_PrebuildIsIdempotent`
  guards this.
- **FAPI handler dimensions must match launch-pattern dimensions.** Handler
  `get_cell_num()` / `get_slots_per_frame()` must equal the loaded pattern.
  Guarded by `tc_FapiHandlerDimensionsMatchLaunchPattern`.
- **YAML lifetime ordering.** `_config_yaml_document` / parser are declared so
  they outlive `_configs`; `yaml::node` views must not outlive the document.
  Same constraint applies in unit-test `SuiteState`.
- **CUDA primary context must outlive `test_mac`.** Declare `PrimaryCtxGuard`
  (or `cuda_ctx`) before the `test_mac` object so that the GPU context remains
  valid during cleanup.
- **TX-deadline static_assert** (`fapi_handler.hpp`):
  `MAX_SCHEDULE_AHEAD_TIME_NS > AVG_SCHEDULE_AHEAD_TIME_NS + SYSTEM_WAKEUP_TIME_COST_NS`
  is a compile-time timing correctness guard; do not relax it.
- **`AERIAL_CUMAC_ENABLE` must be PUBLIC.** The cuMAC sidecar feature flag is
  `PUBLIC` on `testmac_core` so that class layouts (e.g. `test_mac_configs`)
  stay ABI-consistent across consumers even when the flag is off.

## Patterns specific to this area

- **Launch-pattern YAML = per-slot TV schedule.** Top-level `Cell_Configs` /
  `UL_Cell_Configs` list per-cell config HDF5s; `SCHED` is a list of
  `{slot, config:[{cell_index, channels:[TV .h5], type:[PDSCH,PBCH,...]}]}`.
  Parsed into a 4-D `slot_pattern_t` (slot x cell x fapi_group x channel).
- **Test vectors are HDF5 (`.h5`)**, one per (cell, slot, channel-group),
  fetched via git-lfs. They live under `testVectors/` (e.g.
  `testVectors/multi-cell/`), not in the testMAC directory itself. Run
  `git lfs pull` before using them.
- **Channel mask vocabulary**: names joined by any 1-char delimiter
  (`PDSCH+PDCCH_DL+...`) or numeric bitmask; `bfw_dl_str`/`bfw_ul_str`
  are special-cased. Mask 0 = all channels.
- **TB placement** (`fapi_tb_loc`): 0/1 -> CPU_DATA, 2 -> CPU_LARGE,
  3 -> GPU_DATA (CUDA IPC). IPC buffer sizes are fetched from
  `nv_ipc_get_buf_size()` at `start()`; prebuilt/encode buffers are sized
  from `get_max_msg_size()` / `get_max_data_size()`.
- **Optional CMake guards**: `cumac_*` sources compile only if
  `ENABLE_CUMAC AND TARGET cumac`; `scf_fapi_handler.cpp` only if
  `TARGET scf_5g_fapi`. The optional target `ENABLE_CONFORMANCE_TM_PDSCH_PDCCH`
  controls PDSCH/PDCCH conformance tests.
- **`-Werror` exemption on the unit-test target.** `sdk::warnings` is
  intentionally omitted on `test_testmac_core` because upstream
  `cuphy_hdf5.hpp` / `cuphy.hpp` carry warnings; this is documented and
  expected, not a misconfiguration.
- **`#ifdef SCF_FAPI_10_04`-gated cross-checks** in the unit tests match the
  cuPHY-CP CI preset coverage pattern; some tests are only active under the
  10_04 FAPI macro (see cuPHY-CP `AGENTS.md` for CI preset details).

## Building and testing

Build as part of cuPHY-CP (targets `test_mac` and lib `testmac_core`).

Unit tests (`CTest`, label `cuphy-cp;requires-tvs`):
- `testmac_core.copy_tv` - copies test vectors (FIXTURES_SETUP)
- `testmac_core.unit_tests` - GTest suite (FIXTURES_REQUIRED the TV copy);
  needs `cuBB_SDK` env and test vectors present. GPU is initialized only for
  pattern/prebuild suites; `FapiValidate.*` / `CommonUtils.*` are GPU-free.

System-level testing is script-driven (`scripts/`, and
`testBenches/phase4_test_scripts/`) running the full
`ru-emulator -> cuphycontroller -> testMAC` chain; not CTest.

See `cuPHY-CP/AGENTS.md` for the cuphy-cp CTest lane and container/build
conventions, and the repository root `AGENTS.md` for cuBB-wide standards.

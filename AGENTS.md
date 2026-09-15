# cuBB - NVIDIA Aerial CUDA-Accelerated RAN SDK

cuBB is NVIDIA's Aerial CUDA-Accelerated RAN SDK: a full 5G NR baseband stack
implementing Layer 1 (cuPHY) and Layer 2 (cuMAC) for real-time base-station
processing on NVIDIA GPU+CPU platforms (e.g., Hopper/Ampere, Grace Hopper). It is a
C++20/CUDA17 monorepo with optional Python bindings (pyAerial). cuPHY runs the
GPU-accelerated physical layer (LDPC/Polar coding, OFDM, MIMO, channel
estimation); cuPHY-CP integrates the control plane (PHY driver, L2 adapter, SCF
FAPI, fronthaul, RU emulator).

## Slot data-flow (how a slot crosses the whole stack)

DL spine: L2/MAC -> SCF FAPI messages (DL_TTI/UL_TTI/TX_DATA over nvIPC) -> cuPHY-CP.
cuPHY-CP: the SCF adapter (scfl2adapter) parses FAPI into per-cell `slot_command`s; a per-slot tick loop drives the PHY driver (cuphydriver).
cuphydriver launches the cuPHY GPU L1 channel pipelines (DL: PDSCH/PDCCH/PBCH/CSI-RS/BFW; UL: PUSCH/PUCCH/PRACH/SRS).
aerial-fh-driver (de)packetizes IQ as O-RAN fronthaul to/from the RU (or ru-emulator); UL reverses the spine (RU -> cuPHY -> FAPI indications -> L2).
cuMAC / cuMAC-CP are L2-side scheduling sidecars (`cumac_cp` over nvIPC) that feed the FAPI upstream - they are NOT in the per-slot L1 datapath.

## Subsystems

| Dir | Role |
|-----|------|
| `cuPHY/` | GPU-accelerated 5G NR physical layer (L1) in CUDA - e.g., channel coding, modulation, MIMO, channel estimation. |
| `cuPHY-CP/` | Control-plane integration: PHY driver, L2 adapter, SCF FAPI interface, fronthaul driver, RU emulator. CMake project `cuphy_ctrl_plane`. |
| `cuMAC/`, `cuMAC-CP/` | GPU-accelerated L2/MAC scheduling (built by default; disable with `ENABLE_CUMAC=OFF`). |
| `pyaerial/` | Optional Python bindings, pyAerial (built by default; disable with `ENABLE_PYAERIAL=OFF`). |
| `5GModel/` | MATLAB/Python 5G NR reference models that generate validation test vectors; see `5GModel/AGENTS.md`. |
| `testVectors/` | Repository test-vector entry point and fixture layout; see `testVectors/README.md`. |
| `testBenches/` | Test vectors, simulation harnesses, and CICD performance scripts. |
| `aerial_common/` | Header-only shared C++ utilities and GSL contract configuration used across the SDK. |
| `cmake/` | Shared CMake utilities (`cubb-utils.cmake`); options use the `ACAR_` prefix. |

## Fixtures & data

Never commit real usernames, hostnames, IPs, passwords, or other
personal/identifying data into fixtures, configs, manifests, or sample data -
use synthetic placeholders only.

## Container

Builds and tests run **inside the Aerial container**, which supplies the CUDA
toolchain and dependencies. Launch it with `run_aerial.sh` (it sources
`cuPHY-CP/container/setup.sh` for the image coordinates, then pulls and runs the
image):

```bash
git lfs pull                                # first time: fetch LFS payloads
./cuPHY-CP/container/run_aerial.sh          # interactive shell inside the container
```

`run_aerial.sh` resolves and pulls the image (the prebuilt one from NGC,
`nvcr.io/nvidia/aerial/aerial-cuda-accelerated-ran`; see repo `README.md`) - you
do not select or build it by hand. `setup.sh` is configuration sourced by the
launcher; do not run it directly.

For non-interactive (one-shot) builds and tests, pass the whole command as a
**single quoted string**; the script joins its arguments and hands them to the
container's `bash -c`, exiting when the command finishes. Select the CMake
preset that matches the target architecture; `run_aerial.sh` selects the
container platform, not the CMake preset:

```bash
# x86_64 host
./cuPHY-CP/container/run_aerial.sh 'cd /opt/nvidia/cuBB && cmake --preset minimal-x86 && cmake --build build-minimal-x86 -- -j8'

# aarch64/Grace host
./cuPHY-CP/container/run_aerial.sh 'cd /opt/nvidia/cuBB && cmake --preset minimal-arm && cmake --build build-minimal-arm -- -j8'
```

- **Quoting hazard:** do NOT wrap the command in an extra `bash -c '...'`.
  Word-splitting then drops a leading `cd`, so the command silently runs in
  `/opt/nvidia/cuBB` (the main checkout) and can clobber its build directory.
- **Pull latency:** every invocation runs `docker pull` first (can take
  minutes); there is no skip flag. Interactive mode (`-it`) is added only when
  stdin is a TTY.
- **Worktrees:** only the repo root is mounted, at `/opt/nvidia/cuBB`. A git
  worktree is reachable inside the container only if it lives under the repo
  directory (e.g. `<repo>/.worktrees/<name>` ->
  `/opt/nvidia/cuBB/.worktrees/<name>`) - `cd` there explicitly before
  building. Worktrees outside the repo root are not mounted; recreate them
  under the repo or pass an extra mount.
- **Parallelism:** `-j32` can OOM-kill `nvcc` on constrained hosts; start at
  `-j8` and raise only if memory allows. Keep `CCACHE_DISABLE=1` in sandboxes
  (see Build).

## Build

Run builds **inside the container**; a host build fails without the CUDA
toolchain. Two entry points exist, with **distinct** `--preset` vocabularies -
do not mix them:

- **Canonical** (wraps CMake; used by CI and the test benches):
  `./testBenches/phase4_test_scripts/build_aerial_sdk.sh`, whose `--preset` is
  one of `{perf, 10_02, 10_04, 10_04_low_memory, 10_02_SRS_10_04}` and
  `--toolchain` one of `{grace-cross, devkit, native}`.
- **Direct CMake:** `cmake --preset <p>` where `<p>` comes from
  `CMakePresets.json` (`minimal-x86`, `minimal-arm`, `cicd-test-x86`,
  `asan-x86-debug`, ...). A toolchain file is mandatory - configure FATAL_ERRORs
  without `CMAKE_TOOLCHAIN_FILE`; the presets set it. The build dir is
  `build-<preset>`:

```bash
cmake --preset minimal-x86
cmake --build build-minimal-x86 --target <target> -- -j8
```

- In sandboxed/restricted environments set `CCACHE_DISABLE=1` - the ccache tmp
  dir is read-only there and the build will otherwise fail.
- `-Werror` is enforced on CXX/C/CUDA. Every warning is a build error.

## CMake conventions

See `cmake/README.md` for the authoritative build-system details. New targets
should link `sdk::options` and `sdk::warnings`; these provide the project C++20
and warning policy plus active sanitizer options. Existing legacy targets are
not required to be migrated. Use the repository's `asan-*` or `tsan-*` presets
for sanitizer builds and the suppression files under `tools/` when applicable.

## Testing

Tests are CTest-based and wired per subsystem. Run them through the test presets
in `CMakePresets.json` (they set `ENABLE_TESTS=ON`) or a subsystem's own entry
point - the cuPHY-CP `cuphy-cp` ctest lane (`cuPHY-CP/AGENTS.md`), pyAerial's
`scripts/run_unit_tests.sh` (`pyaerial/AGENTS.md`), `test_scfl2adapter`
(`cuPHY-CP/scfl2adapter/lib/scf_5g_fapi/AGENTS.md`), and the testBenches perf
harness. Never delete or silently disable a failing test to green a build: mark
it `DISABLED TRUE` in its CMakeLists and open a tracking issue to fix it.

## C++20 / CUDA standards

- C++ standard is **C++20** (`CMAKE_CXX_STANDARD 20`, required); CUDA standard
  is **17**. Do not introduce C++23-only constructs.
- Error handling: use `tl::expected` - never roll a custom Result/Either type.
- Use `std::filesystem`, `std::string_view`, and `libfmt` (the fmt library -
  **not** `std::format`).
- Logging: NVLOG/libfmt **cannot bind a reference to a packed-struct field**.
  Copy the field into a local variable first, then pass the local to any
  `NVLOG*_FMT` macro.
- Prefer `final` classes, NSDMI (non-static data member initializers), and
  `inline static` members.
- Use **include guards**, not `#pragma once`.
- Non-void returns are typically `[[nodiscard]]` where ignoring the result is a
  bug (enforced most strictly in the SCF FAPI parsers; critical for `tl::expected`
  results). A convention, not a universal invariant - not every getter carries it.
- New CMake options use the `ACAR_` prefix, not `ENABLE_`/`BUILD_`/`CUPHY_` (this
  applies to newly introduced options; existing `ENABLE_*` such as `ENABLE_CUMAC`
  / `ENABLE_PYAERIAL` remain).
- Prefer `CMAKE_CURRENT_LIST_DIR` / `CMAKE_SOURCE_DIR` over `__FILE__`.

## Git conventions (NVIDIA-internal contributors)

- Commit first line: `[JIRA: GT-xxxxx] <description>` (note the colon-space
  inside the brackets; not a bare `GT-xxxxx:`).
- Main branch for PRs/MRs is `develop`.

**External-facing:** With no NVIDIA ticket, omit the `[JIRA: ...]` commit prefix
and use your own fork-local conventions.

## Local overrides

If a file named `AGENTS.local.md` exists at the repository root, read it for
environment-specific setup (personal tooling, MCP servers, machine paths). It is
optional; keep it out of version control (add it to `.gitignore` or
`.git/info/exclude`). A public clone will not have one, in which case there is
nothing to load.

## Python

- The current Aerial devel container uses Python 3.12. Select the interpreter
  in this order:
  1. Use a virtual environment owned by the component or task when it exists.
     For example, pyAerial uses `pyaerial/.venv`; other components may document
     their own environment.
  2. Otherwise, use the cuBB root `.venv` when it exists
     (`/opt/nvidia/cuBB/.venv` inside the container).
  3. If neither exists, use the system-wide `python3`/`pip3` installation.
- Do not create a new virtual environment solely for general repository work.
  Follow a component's README when that component requires or provides one.

```bash
# Component or root venv, when present:
<venv>/bin/python script.py
<venv>/bin/pip install <package>

# No applicable venv:
python3 script.py
pip3 install <package>
```

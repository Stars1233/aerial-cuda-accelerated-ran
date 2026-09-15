# cmake/ — Build System Modules

This directory contains reusable CMake modules. The root `CMakeLists.txt` includes
`CompileOptions.cmake`, `Cuda.cmake`, and `CMakeFormat.cmake`. `CompileOptions.cmake` includes
the modules needed for SDK compile options. Each module uses `include_guard(GLOBAL)` so CMake
loads it only once.

## Modules

| Module | Purpose |
|---|---|
| `CompileOptions.cmake` | Defines `sdk::warnings` and `sdk::options` INTERFACE targets. Entry point — includes all other modules. |
| `CompilerWarnings.cmake` | `set_project_warnings()` — full project warning set for GCC, Clang, MSVC, CUDA. |
| `SanitizerOptions.cmake` | `ACAR_ENABLE_SANITIZER_*` options wired into `sdk::options`. See **Sanitizers** below. |
| `Sanitizers.cmake` | `enable_sanitizers()` — applies `-fsanitize=` flags to C/CXX only (not CUDA). Called by `SanitizerOptions.cmake`. |
| `Cuda.cmake` | `target_link_cuda()`, `detect_gpu_architecture()`, `check_cuda_architectures()`. |
| `CpuArchitecture.cmake` | `detect_cpu_architecture()`, `get_arch_lib_suffix()`. |
| `SystemLink.cmake` | `target_link_system_library()`, `target_include_system_directories()` — suppresses warnings from external deps. |
| `Utilities.cmake` | Generic helpers: env parsing, target enumeration, verbosity checks. |
| `cubb-utils.cmake` | `read_aerial_sdk_version_file()` — compile-time/runtime version sync. |
| `StaticAnalyzers.cmake` | `enable_clang_tidy()` — target-scoped C++ clang-tidy with the Python cache wrapper. |
| `ClangFormat.cmake` | `enable_clang_format()` — opt-in target/file-scoped clang-format checks and rewrites. |
| `CMakeFormat.cmake` | `enable_cmake_format()` — opt-in CMake formatting checks and rewrites. |

## Clang-tidy

Clang-tidy is disabled by default and is never installed globally through
`CMAKE_CXX_CLANG_TIDY`. Enable it explicitly and opt targets in one at a time:

```bash
cmake ... -DENABLE_CLANG_TIDY=ON
```

```cmake
enable_clang_tidy(my_cpp_target)
```

The helper finds clang-tidy only when invoked for an opted-in target, requires
Python for `cmake/helpers/clang_tidy_cache.py`, uses the build compile database
and project C++ standard, and passes `-warnings-as-errors=*`. The cache can be
disabled with `-DENABLE_CLANG_TIDY_CACHE=OFF`. CUDA per-file clang-tidy and CI
clang-tidy jobs are intentionally outside this initial rollout; add targets
incrementally after local validation.

## Clang-format

Clang-format is disabled by default. Enable it explicitly and opt targets in
one at a time:

```bash
cmake ... -DENABLE_CLANG_FORMAT=ON
```

The repository has one `.clang-format` configuration in its root directory.

```cmake
enable_clang_format(my_cpp_target)
enable_clang_format(my_cpp_target FILES source.cpp include/header.hpp)
```

The helper creates `<target>-clang-format-check`, which runs
`clang-format --dry-run --Werror --style=file`, and
`<target>-clang-format-fix`, which rewrites the selected files. Without
`FILES`, all supported C/C++/CUDA sources listed by the target are selected.
When the option is enabled, `clang-format-all` checks every opted-in target,
and `clang-format-fix-all` rewrites every opted-in target. With no opt-ins,
these aggregate targets do nothing. The helper is intentionally target-scoped.

## CMake-format

CMake-format is disabled by default. Enable it when you want to check explicitly opted-in CMake
files:

```bash
cmake ... -DENABLE_CMAKE_FORMAT=ON
```

The repository has one `.cmake-format.yaml` configuration in its root directory. Opt files in
explicitly; CMake-format does not discover files automatically:

```cmake
enable_cmake_format(FILES cmake/CMakeFormat.cmake)
```

`cmake-format-all` checks every opted-in file without changing it.
`cmake-format-fix-all` rewrites every opted-in file and must be invoked explicitly.

## Using sdk:: Targets

New code should link both targets:

```cmake
target_link_libraries(<target> PRIVATE sdk::options sdk::warnings)
```

- `sdk::warnings` — enforces the full warning set (warnings-as-errors for new code)
- `sdk::options` — enforces C++20 + inherits any active sanitizer flags

Legacy targets do not need to link these.

## cppcheck

`ENABLE_CPPCHECK` defaults to `ON`. It requires cppcheck at configure time and
defines the shared checker command; it does not run cppcheck by itself. The
commented `nvphy_fapi_tasks` block is the target opt-in example; re-enable it
after its existing findings are addressed. No target receives cppcheck
automatically. Disable it for a local configure with:

```bash
cmake ... -DENABLE_CPPCHECK=OFF
```

With cppcheck enabled, configuration fails if the executable is unavailable.
When a target opts in, `--error-exitcode=2` makes cppcheck findings fail its
build with exit code 2.

## Sanitizers

### Options

All default `OFF`. Forced `OFF` on unsupported platforms (capability-probed at configure time).

| CMake Option | Sanitizer | Notes |
|---|---|---|
| `ACAR_ENABLE_SANITIZER_ADDRESS` | AddressSanitizer (ASan) | Detects heap/stack/global memory errors |
| `ACAR_ENABLE_SANITIZER_LEAK` | LeakSanitizer (LSan) | Shares ASan runtime; enable together |
| `ACAR_ENABLE_SANITIZER_UNDEFINED` | UndefinedBehaviorSanitizer (UBSan) | Detects signed overflow, misalignment, etc. |
| `ACAR_ENABLE_SANITIZER_THREAD` | ThreadSanitizer (TSan) | Mutually exclusive with ASan/LSan (enforced at configure time) |
| `ACAR_ENABLE_SANITIZER_MEMORY` | MemorySanitizer (MSan) | Clang only; mutually exclusive with all other sanitizers (enforced at configure time); requires fully instrumented stdlib |
| `ACAR_ENABLE_SANITIZER_POINTER_COMPARE` | ASan pointer-pair checks | GCC only; requires ASan to be active; enables `-fsanitize=pointer-compare,pointer-subtract`; needs `detect_invalid_pointer_pairs=2` in `ASAN_OPTIONS` (set automatically by the `asan-*` presets) |

### Usage

Use the CMake presets from the repository root. The presets enable the
required test options, sanitizer compile flags, and runtime environment
variables. Runtime variables live in the hidden `asan-env` and `tsan-env`
presets in `CMakePresets.json`; update those presets instead of copying
sanitizer option strings into CI scripts.

```bash
cmake --list-presets=all
```

The `asan-*` presets enable **AddressSanitizer + LeakSanitizer + UBSan + pointer-compare/subtract**
together in a single build. There is no standalone ubsan/lsan preset — those sanitizers always
run as part of the asan preset bundle.

Local x86 ASan validation for the lowercase `cuphy-cp` CTest set:

```bash
cmake --preset asan-x86-debug
cmake --build --preset asan-x86-debug
ctest --preset asan-x86-debug
```

TSan follows the same flow with `tsan-x86-debug` or `tsan-arm-debug`. TSan is
mutually exclusive with ASan/LSan.

Suppression files for known third-party false positives: `tools/lsan_suppressions.txt`,
`tools/ubsan_suppressions.txt`, `tools/tsan_suppressions.txt`.

# pyAerial (`pyaerial/`)

## Purpose

pyAerial is a documented Python API to a subset of cuPHY (the cuBB 5G NR
physical layer). It wraps cuPHY GPU components as pybind11 bindings and a
higher-level Python `aerial` package, targeted at ML researchers for model
validation/benchmarking, dataset generation, and producing ground-truth
signals (e.g. transmitted MAC PDUs, coded bits, transmitted symbols) for
over-the-air collected datasets.

## Data flow (high level)

Entry: a user calls a `phy5g` channel pipeline (e.g. `PdschTx.__call__`, built via `PdschTxPipelineFactory.create(config, cuda_stream)`).
`PipelineFactory.create` + a `CudaStream` (`aerial.util.cuda`) construct the pipeline, wrapping a `pycuphy.PdschPipeline`; `with stream:` scopes the GPU work.
NumPy/CuPy transport blocks cross column-major (`order='F'`) into the `aerial.pycuphy` (`_pycuphy`) pybind11 layer, loaded after the `.so` preload chain.
The `PYBIND11_MODULE` (`pybind11/pycuphy_pybind.cpp`) dispatches to the C++ wrapper `pycuphy::PdschPipeline::runPdschTx()`.
Lands in the cuPHY C pipeline: `cuphyRunPdschTx()` (`cuphy_api.h`); results convert back to `order='F'` on return.

## Read these first

- `README.md` - authoritative developer guide (build, install, test, notebooks). Defer to it.
- `CMakeLists.txt` - C++/CUDA build of the `_pycuphy` extension; lists every pybind11 source and consumes the CMake-built `gpu3gppchan` wheel artifact.
- `pyproject.toml` / `setup.py` - Python package `pyaerial` (package_dir `src`); PEP 621 `[project]` metadata plus optional-dependency extras (`ml`, `ml-amd64`/`ml-arm64`, `notebooks`, `dev`) and `interrogate`/`mypy` config; `setup.py` discovers the `_pycuphy` extension suffix dynamically.
- `src/aerial/__init__.py` - package entry; imports `aerial.pycuphy`.
- `src/aerial/pycuphy/__init__.py` - cuPHY backend entry; `from ._pycuphy import *` (shared-library deps resolve via the extension's RPATH).
- `src/aerial/phy5g/api.py` - generic base classes (`SlotConfig`, `PipelineConfig`, `Pipeline`, `PipelineFactory`) that the channel pipelines build on.
- `pybind11/` - C++ binding sources (e.g. `pycuphy_*` for SRS, CSI-RS, channel est/eq, LDPC, PUSCH params, CRC).

## Layout

- `src/aerial/` - Python package (`pycuphy` backend, `phy5g` PHY library, `util`, `model_to_engine`).
- `src/aerial/phy5g/` - 5G NR PHY: `pusch/`, `pdsch/`, `srs/`, `csirs/`, `ldpc/`, `algorithms/`, `channel_models/`, plus `api.py`, `config.py`. `channel_models/` is a compatibility facade over the standalone `gpu3gppchan` package.
- `src/aerial/model_to_engine/` - ML model export (`exporters/` ONNX + TensorRT, `model/`, `algorithm_base/`).
- `src/aerial/util/` - `cuda.py` (`CudaStream`), `data.py`, `fapi.py`, `visualization.py`.
- `pybind11/`, `cmake/` (`PythonEnv.cmake`, `CleanStagedLibs.cmake`), `scripts/`
  (`setup_venv.sh`, `_env.sh`, `run_pyaerial_ci.sh`, test runners), `models/`,
  `notebooks/`, `tests/`.

## Environment / install

- pyAerial builds and runs inside the **standard Aerial devel container** (via
  `cuPHY-CP/container/run_aerial.sh`); there is no separate pyAerial container.
- **pyAerial has its own required venv `pyaerial/.venv`**. Before executing
  pyAerial code, run `pyaerial_setup`; use `pyaerial/.venv` for all pyAerial
  development, testing, notebook, and script work. It carries the editable
  `_pycuphy` install plus the `ml`/`torch`/`notebooks` extras.
  `pyaerial_setup` creates or reuses it through `scripts/setup_venv.sh`
  (`pip install -e '.[dev,ml,ml-<arch>,notebooks]'`), and the test scripts resolve
  `$PYTHON`/`$PIP` from it via `scripts/_env.sh`. Do not substitute the cuBB
  root `.venv` or system Python for a missing pyAerial environment.

  ```bash
  source pyaerial/.venv/bin/activate      # after pyaerial_setup
  # or: pyaerial/.venv/bin/python script.py
  ```

- Build and set up the environment through the CMake presets (run from the cuBB
  root, inside the devel container):

  ```bash
  cmake --preset pyaerial-x86                              # or pyaerial-arm
  cmake --build --preset pyaerial-x86 --target _pycuphy
  cmake --build --preset pyaerial-x86 --target pyaerial_setup   # venv + editable install
  ```

  `scripts/run_pyaerial_ci.sh` wraps this (auto-selects the preset by arch, then
  runs `ctest --preset`) and is the entry point CI uses.
- Smoke test: `pyaerial/.venv/bin/python -c "import aerial"` must succeed with no
  errors.

## Build / enable flag

- Built only when the root cmake option `ENABLE_PYAERIAL` is ON (default ON): `OPTION(ENABLE_PYAERIAL "Enable pyaerial Python bindings" ON)` in `CMakeLists.txt`, which guards `add_subdirectory(pyaerial)`.
- The `ACAR_` prefix convention applies only to newly added options; `ENABLE_PYAERIAL` and `ENABLE_PUSCH_PER_UE_PREQ_NOISE_VAR` keep their existing names.
- `pyaerial/CMakeLists.txt` pins C++20 / CUDA 17 locally (it may be added via `add_subdirectory` without the root cmake), and defines `ENABLE_PUSCH_PER_UE_PREQ_NOISE_VAR` (default ON) → `-DUSE_PUSCH_PER_UE_PREQ_NOISE_VAR`.

## Patterns / invariants

- The compiled extension is `_pycuphy` (`_pycuphy.cpython-312-<machine>-linux-gnu.so`); the current container uses the Python 3.12 ABI. Its shared-library deps (`libcuphy`, `libnvlog`, `libgpu3gppchan`, `libfmtlog-shared`) resolve through the extension's RPATH (the build-tree path, for the editable install).
- Channel pipelines follow a factory pattern: subclass `SlotConfig`/`PipelineConfig`, implement `PipelineFactory.create(config, cuda_stream, ...)` returning a `Pipeline`. CUDA work is scoped via `aerial.util.cuda.CudaStream` (`with stream:`); synchronize explicitly with `stream.synchronize()`.
- Arrays are NumPy or CuPy (`Array = TypeVar(..., np.ndarray, cp.ndarray)`); GPU interchange uses the CUDA array interface (`pybind11/cuda_array_interface.cpp`).
- Most cuPHY wrappers hand `_pycuphy` **column-major** (Fortran-order) buffers, converting with `order='F'` before/after the crossing (pervasive across ~22 files, e.g. `phy5g/pusch/pusch_rx.py:300`, `phy5g/ldpc/encoder.py:157`, `phy5g/srs/srs_rx.py:100`); a new PUSCH/LDPC/SRS-style binding that omits it silently transposes data. Ordering is not universal, though: `py5g.FadingChannel` (imported from `gpu3gppchan.fading_channel.FadingChannel`) converts its input to C-order `cp.complex64`, calls the native API with `tx_column_major_ind=0`, and returns C-order output. Other `gpu3gppchan` entry points must match their explicit `*_column_major_ind` argument.
- Lint and tests run through the CMake targets / CTest: `pyaerial_lint`
  (`scripts/run_static_tests.sh`: `flake8`, `pylint`, `mypy`, `interrogate`) and
  `pyaerial_test` (`scripts/run_unit_tests.sh <trt|pytest|all>`), or all of it via
  `ctest --preset pyaerial-x86` (label `pyaerial`). Static tests are gating:
  `interrogate` docstring coverage is `fail_under=100`, so every public symbol
  needs a docstring. Keep changes lint/type/docstring clean.
- Unit tests need cuPHY test vectors. **Internal:** default mount
  `/mnt/cicd_tvs/develop/GPU_test_input/`. **External-facing:** set
  `$TEST_VECTOR_DIR` to your vector directory; tests that can't find vectors are
  skipped, not failed.
- Document only what exists in these files; verify against `README.md` and the source before adding claims.

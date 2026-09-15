# 5GModel

## Purpose

Reference implementation of the 5G NR physical layer in MATLAB and Python, used
to:

1. Validate waveform compliance against the MATLAB 5G Toolbox.
2. Generate test vectors (H5 files in `GPU_test_input/`) for cuPHY and cuBB
   verification.
3. Run receiver performance simulations based on 3GPP specifications.

This is a reference model implementation used primarily for offline testing and
analysis.

## Components

- **`nr_matlab/`**: MATLAB 5G NR reference implementation. Run `startup` then
  `runRegression` from `nr_matlab/` to generate TVs and verify waveform
  compliance. TVs land in `nr_matlab/GPU_test_input/`.
- **`aerial_mcore/`**: Python API bindings (`aerial_mcore` pip-installable wheel
  in `aerial_mcore/aerial_pkg/dist/`). Generates TVs programmatically via
  `aerial_mcore/examples/example_5GModel_regression.py`. See
  `aerial_mcore/QuickStart.md` for the install and run flow.

## Relationship to cuPHY / testBenches

TVs produced here (`GPU_test_input/*.h5`) are the test vectors consumed by cuPHY
unit tests and the testBenches perf harness (`testBenches/perf/`,
`$TEST_VECTOR_DIR`). Stale or missing TVs cause cuPHY tests to skip or fail.

## Key files

- `README.md` - overview and MATLAB regression workflow.
- `aerial_mcore/QuickStart.md` - Python API install and TV generation steps.
- `aerial_mcore/aerial_pkg/pyproject.toml` / `setup.py` - Python package
  metadata.
- `nr_matlab/cfg_template.yaml` - base config template for simulation runs.
- `nr_matlab/runRegression.m` - MATLAB regression entry point.
- `nr_matlab/test/` - test case definitions.
- `nr_matlab/gNBreceiver.m` / `nr_matlab/gNBtransmitter.m` - gNB transmitter
  and receiver reference implementations.

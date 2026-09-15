# scfl2adapter Unit Tests

Unit tests for the SCF 5G FAPI L2 adapter layer.

## Test suites

| Suite | File | Covers |
|-------|------|--------|
| `TtiDispatch` | `test_tti_dispatch.cpp` | `dispatch_pdus` routing and `dispatch_pdus_typed` active-type filtering (Groups A–B) |
| `TtiDispatch` | `test_tti_dispatch.cpp` | Per-parser body-size guard: `pdu_size < sizeof(header)+sizeof(pdu_t)` rejected before deref (Group D) |
| `DLSlotProcessor` | `test_tti_dispatch.cpp` | Null-buffer, empty-payload, batch first-failure-wins, sfn/slot propagation in errors, and (with `SCF_FAPI_10_04`) PDU-count mismatch (Group C) |
| `DLSlotProcessor` | `test_tti_dispatch.cpp` | `extract_req` `msg_len` guard: short message returns `NullBuffer` error (Group E) |
| `DLSlotProcessor` | `test_pdsch_pdu_parser.cpp` | Full PDSCH parse round-trip: single and multi-message batch, slot/rnti/LBRM/CW-offset assertions (Group F) |
| `PdschPduParser` | `test_pdsch_pdu_parser.cpp` | `setup_cell` (slot selection, CSI-RS count); `parse()` UE/CW/UE-group fields; LBRM lookup; beta LUT; UE-group reuse (Groups A–E) |
| `PdschPduParser` | `test_pdsch_pdu_parser.cpp` | `apply_pm_weights`: empty pm_map, hot-path weight insertion, `pm_idx==0` skip, `dig_bf_interfaces==0` skip (Group G) |
| `SrsPduParser` / `ULSlotProcessor` | `test_srs_pdu_parser.cpp` | SRS validation/drop coverage: disabled SRS, bad table indices, SINGLE_SECT_MODE, L1 limit, chest-buffer bad state, unsupported rep_scope/usage, SRS gating/filtering, no front-haul mutation on drops |

## Prerequisites

The tests require the `SCF_FAPI_10_04` CMake option to be **ON** (needed for
`nPDUsOfEachType` / `DL_TTI_NPDUS_IDX_*` macros used in the FAPI message
builders). This is set project-wide via `-DSCF_FAPI_10_04=ON` at configure time.

## Build

From the repo root (inside the development container):

```bash
# Configure — enable SCF_FAPI_10_04 if not already set project-wide
cmake -B /opt/nvidia/cuBB/build -GNinja \
  -DCMAKE_TOOLCHAIN_FILE=cuPHY/cmake/toolchains/native \
  -DSCF_FAPI_10_04=ON

# Build only the test executable
cmake --build /opt/nvidia/cuBB/build --target test_scfl2adapter
```

## Run

### Via ctest (recommended)

`gtest_discover_tests` registers each test case individually, so ctest
can filter, retry, and shard at the per-case level.

```bash
# All scfl2adapter tests with verbose output (matched by LABELS)
ctest --test-dir /opt/nvidia/cuBB/build \
  -L cuphy-cp \
  -R test_scfl2adapter \
  -V

# Show available tests without running
ctest --test-dir /opt/nvidia/cuBB/build \
  -R test_scfl2adapter \
  --show-only
```

### Run the binary directly

The binary is built to:
```
/opt/nvidia/cuBB/build/cuPHY-CP/scfl2adapter/lib/scf_5g_fapi/tests/test_scfl2adapter
```

```bash
cd /opt/nvidia/cuBB/build/cuPHY-CP/scfl2adapter/lib/scf_5g_fapi/tests

# All tests
./test_scfl2adapter

# List all test cases
./test_scfl2adapter --gtest_list_tests

# Filter by suite
./test_scfl2adapter --gtest_filter='TtiDispatch.*'
./test_scfl2adapter --gtest_filter='DLSlotProcessor.*'
./test_scfl2adapter --gtest_filter='PdschPduParser.*'

# TtiDispatch groups
./test_scfl2adapter --gtest_filter='TtiDispatch.Routing_TableDriven'
./test_scfl2adapter --gtest_filter='TtiDispatch.TypedDispatch_PdschActiveOnly_TableDriven'
./test_scfl2adapter --gtest_filter='TtiDispatch.TypedDispatch_PdschAndCsiRsActive_TableDriven'
./test_scfl2adapter --gtest_filter='TtiDispatch.BodySizeGuard_PduSizeBelowMinSizeof_Rejected'

# DLSlotProcessor groups
./test_scfl2adapter --gtest_filter='DLSlotProcessor.Process_TableDriven'
./test_scfl2adapter --gtest_filter='DLSlotProcessor.Process_Pdsch_TableDriven'
./test_scfl2adapter --gtest_filter='DLSlotProcessor.ExtractReq_ShortMsgLen_ReturnsNullBuffer'

# PdschPduParser groups
./test_scfl2adapter --gtest_filter='PdschPduParser.SetupCell_TableDriven'
./test_scfl2adapter --gtest_filter='PdschPduParser.Parse_UeFields_TableDriven'
./test_scfl2adapter --gtest_filter='PdschPduParser.Parse_CodewordFields_TableDriven'
./test_scfl2adapter --gtest_filter='PdschPduParser.Parse_UeGroupFields_TableDriven'
./test_scfl2adapter --gtest_filter='PdschPduParser.LbrmLookup_TableDriven'
./test_scfl2adapter --gtest_filter='PdschPduParser.PowerControlBeta_TableDriven'
./test_scfl2adapter --gtest_filter='PdschPduParser.UeGroupReuse_TableDriven'
./test_scfl2adapter --gtest_filter='PdschPduParser.ApplyPmWeights_TableDriven'

# Write JUnit XML output
./test_scfl2adapter --gtest_output=xml:test_scfl2adapter.xml
```

## Test structure

`test_stubs.cpp` provides the `check_bf_pc_params()` stub, allowing the tests
to link against the FAPI header-only target (`scf::scf_5g_fapi_h`) without
pulling in cuPHY CUDA kernels or gRPC.

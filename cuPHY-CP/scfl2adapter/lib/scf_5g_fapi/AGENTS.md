# SCF 5G FAPI Adapter (scfl2adapter / scf_5g_fapi)

## Purpose

`scfl2adapter` is the SCF-5G FAPI L2 adapter that translates between the
O-RAN/SCF FAPI MAC-PHY interface and the internal cuPHY-CP slot-command
pipeline. It receives packed SCF FAPI messages (DL_TTI.req, UL_TTI.req,
TX_DATA.req, UL_DCI.req, CONFIG.req, etc.) from L2/MAC over shared memory or
IPC, validates and parses them per the SCF 222 specification (both v10.02 and
v10.04 via compile-time guards), translates each PDU into internal
`slot_command` structures, and forwards them to the cuPHY driver. The core
parsing and slot-command building logic lives here in `lib/scf_5g_fapi` as a
static library; the `scf_app/cuphycontroller` executable
(`l2_adapter_cuphycontroller_scf`) is the top-level process that drives the
tick loop.

## Data flow (high level)

Entry: `phy::on_msg` (`scf_5g_fapi_phy.cpp`) pops a packed FAPI message off IPC/shm and switches on `msg_hdr.type_id` (the 84-line `scf_5g_fapi.cpp` is only the factory).
Per-message handler (`on_dl_tti_request` / `on_ul_tti_request` / `on_tx_data_request`) walks the PDU payload span.
Parse per SCF 222 (v10.02/v10.04, compile-gated): the `*_pdu_parser.hpp` decoders validate each PDU (TLV/bitmap layout).
Build: `prepare_dl_slot_command` / `prepare_ul_slot_command` -> `update_cell_command` accumulate into the `PHY_module`-owned `cell_group_command`.
Handoff: `PHY_module` submits that `cell_group_command` to cuphydriver (via `PHYDriverProxy` `l1_*`) at the slot boundary - outside this directory.

## Read These First

- `lib/scf_5g_fapi/scf_5g_fapi.h` - public adapter interface
- `lib/scf_5g_fapi/scf_5g_fapi.cpp` - message dispatch / entry points
- `lib/scf_5g_fapi/scf_5g_slot_commands.cpp` - PDU → slot_command translation
- `lib/scf_5g_fapi/scf_5g_fapi_phy.cpp` - PHY-side FAPI handling
- `scf_app/cuphycontroller/cuphycontroller.cpp` - process entry / tick loop
- `scf_5g_fapi_errors.hpp` - `SlotParseError`, `SlotParseResult`, `UeGroupErrorCode`

## Invariants (never violate)

- **Packed-struct field references**: all FAPI structs use
  `__attribute__((__packed__))`. Never bind a reference or pointer-to-member
  of a packed field directly into a libfmt/NVLOG macro - it is a GCC error.
  Copy the field to a local variable first.
- **Version gating is compile-time only**: SCF FAPI v10.02 vs v10.04
  differences are governed exclusively by `SCF_FAPI_10_04` and
  `SCF_FAPI_10_04_SRS` compile-time guards. Do not add runtime version
  switching. Coverage is preset-dependent: the option defaults OFF, plain
  `cicd-test-x86` / `-arm` build it OFF, and `cicd-test-arm-perf` builds it ON,
  so locally-green can differ from a given CI lane for `#ifdef`-gated code.
- **TLV 4-byte alignment**: TLV advancement aligns to 4 bytes per SCF 222
  section 3.3.1.4: `next = &val[0] + ((length + 3) / 4) * 4`.
- **Vendor TLV tag range**: vendor-specific TLV tags occupy `0xA000-0xAFFF`;
  standard SCF tags must never be placed in that range.

## Build

Configure a preset (see repo-root `AGENTS.md`), then build the `scf_5g_fapi`
target. `<build-dir>` is `build-<preset>` (e.g. `build-minimal-x86`). To build
the 10.04-gated code and its tests, use a preset that sets `SCF_FAPI_10_04=ON`
(the `cicd-test-arm-perf` lane) or pass `-DSCF_FAPI_10_04=ON` at configure time
(this auto-enables `SCF_FAPI_10_04_SRS`):

```bash
cmake --build <build-dir> --target scf_5g_fapi -- -v -j32
```

Outputs:

- Static library: `<build-dir>/cuPHY-CP/scfl2adapter/lib/scf_5g_fapi/libscf_5g_fapi.a`
- Header-only interface target: `scf_5g_fapi_h`

Dependencies: **WiseEnum** (reflective enums) and **tl::expected** - both provided
by the build toolchain/container as system headers (`<tl/expected.hpp>`, no
FetchContent, no custom `expected`) - plus internal **slot_command**, **nvphy**,
**nvipc** libraries.

## Error Handling: use tl::expected

`tl::expected` is provided by the build toolchain/container as a system header
(`<tl/expected.hpp>`). Do **not** use FetchContent for it and do **not** implement
a custom `expected` class.

Real types from `scf_5g_fapi_errors.hpp`:

```cpp
#include <tl/expected.hpp>

// SlotParseError is a struct with nested Code enum - NOT a wise_enum type
struct SlotParseError { enum class Code { NullBuffer, PduCountMismatch, PerTypeMismatch, BfParamsInvalid }; Code code; };
using SlotParseResult = tl::expected<void, SlotParseError>;

[[nodiscard]] SlotParseResult parse_slot(...) {
    if (error) return tl::unexpected(SlotParseError{SlotParseError::Code::NullBuffer});
    return {};
}
```

Caller pattern:

```cpp
auto result = parse_slot(pdu);
if (!result) {
    NVLOG_ERR_FMT("parse failed: code={}", static_cast<int>(result.error().code));
    return -1;
}
```

## C++ Patterns Specific to This Area

### wise_enum for error enums

`WiseEnum` provides automatic reflection (`to_string`, `from_string`, `range`,
`size`). In this library it is used for `UeGroupErrorCode` - **not** for
`SlotParseError` (which is a plain struct):

```cpp
#include <wise_enum/wise_enum.h>

WISE_ENUM_CLASS((UeGroupErrorCode, std::uint8_t), UeGroupsFull, UeGroupAtCapacity);
// wise_enum::to_string(UeGroupErrorCode::UeGroupsFull) → "UeGroupsFull"
```

Integration: WiseEnum headers arrive transitively from linked targets
(`aerial_common` et al.); this `CMakeLists.txt` does not call
`find_package(WiseEnum)`.

## Testing

Integration tests (`test_scfl2adapter`, CTest label `cuphy-cp`) are
built only when **both** `ENABLE_TESTS` and `SCF_FAPI_10_04` are ON. A test
preset supplies `ENABLE_TESTS`; the perf preset (or `-DSCF_FAPI_10_04=ON`) adds
the macro. The binary does not exist otherwise.

Configure a preset that sets both flags (e.g. `cicd-test-arm-perf`, or any test
preset plus `-DSCF_FAPI_10_04=ON`), build target `test_scfl2adapter`, then run it
via `ctest` (the `cuphy-cp` label selects it) or directly:

```bash
<build-dir>/cuPHY-CP/scfl2adapter/lib/scf_5g_fapi/tests/test_scfl2adapter
```

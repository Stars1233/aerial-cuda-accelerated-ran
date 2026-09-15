# data_lake - Telemetry Capture & E3/dApp Streaming

## Purpose

`data_lake` captures per-slot PHY telemetry (fronthaul IQ, PUSCH PDUs, channel
estimates, SRS) into ClickHouse and streams it to out-of-tree dApp clients over
a ZMQ control channel plus a POSIX shared-memory data channel (the E3 agent). It
is a `SHARED` library, always compiled and always linked into `cuphydriver`;
only runtime config (`datalake_samples` / `data_core`) decides whether it does
anything.

## Invariants (never violate)

- **`e3::StreamType` IDs and `SharedMemoryHeader` are a wire/ABI contract with an
  out-of-tree consumer** (`NVIDIA/aerial-sample-apps` dApps) and **no in-tree
  build or test exercises the E3 ZMQ/SHM path.** A renumber/rename/reorder that
  compiles and passes every local check still silently breaks dApps. The
  append-only / never-reorder / cap-128 rules that prevent this already live in
  the `e3_agent.hpp` comments at each type definition - read and obey them there
  before editing these types.

## Testing

Wire-path verification is integration-only (there is no unit test): run the
DATALAKE scenario (`ru_emulator` + `test_mac`) and diff captured DB rows against
the launch pattern - see `data_lake/tests/`. **External-facing:** that procedure
assumes internal container names (`c_aerial_$USER` / `pyaerial_$USER`);
substitute your own container name.

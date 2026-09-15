# Polar Decoder Example / Testbench

Standalone driver for the cuPHY polar decoder (`cuphyPolarDecoder`). It runs the
GPU decoder either on a stored test vector or on codewords it generates itself,
and can sweep BER/BLER versus SNR. It is a validation/benchmark tool, not
production runtime code.

- Source: `cuphy_ex_polar_decoder.cpp`
- Self-contained test-vector generator: `polar_decode_test_vec_gen.{hpp,cpp}`
- Target: `cuphy_ex_polar_decoder` (added in `cuPHY/examples/polar_decoder/CMakeLists.txt`)

Build with the usual Aerial SDK flow, or build the target directly in your
configured build directory (e.g. `ninja cuphy_ex_polar_decoder`).

## Input modes

The example needs exactly one input source:

- **File input** (`-i`): decode a stored HDF5 test vector, or a multi-cell YAML
  file listing several HDF5 vectors. The reference tree types and channel LLRs
  are read from the vector.
- **Generator input** (`-A <payloadBits>`): no input file needed. The built-in
  generator encodes random payloads with a host uplink polar encoder, runs them
  through a channel, and produces the per-codeword LLRs plus the `compCwTreeTypes`
  tree-types layout the decoder consumes. Passing `-A` selects this mode.

The generator produces a valid CRC per codeword, so list decoding (`-L > 1`),
which relies on CRC-based candidate selection, is exercised meaningfully.

The host encoder (`hostUplinkPolarEncodeRateMatch`, mirroring the 5GModel
`uplinkPolarCbEncoder` / `uplinkPolarRmItl` reference) is not bound by the cuPHY
DL polar-encoder caps, so the generated mother code reaches the decoder limit
**N = 1024** (38.212 `n_max = 10` for UCI), not 512. It is bit-exact with
`cuphyPolarEncRateMatch` for every K within that encoder's range (K &le; 164).
It covers the **full single-segment UCI payload range, `A` in `[12,1706]`**:

- `A` in `[12,19]`: CRC6 + 3 parity-check (PC) bits (the PC positions, including
  the weight-metric case, come from `compCwTreeTypes`).
- `A >= 20`: CRC11.
- `(A >= 360 & E >= 1088)` or `A >= 1013`: the segment splits into **two
  codeblocks** of `ceil(A/2)` info bits each (a zero is prepended to CB0 when `A`
  is odd), per 38.212 6.3.1.2 segmentation.

`A < 12` (Reed-Muller / simplex in NR) is rejected. The decoder processes at most
256 codewords per launch, so a segment count with `nCbs` codeblocks each that
would exceed 256 codewords is rejected (reduce `-w`); `--sweep` batches
automatically respect this.

## Channel

By default the generator runs the **full rate-matched RX chain**: it rate-matches
the `N_cw` coded bits to `E` transmitted bits, modulates, adds AWGN at the
requested SNR, then de-rate-matches / de-interleaves back to `N_cw` LLRs
(`E != N_cw` is the normal case). `--no-ratematch` modulates the `N_cw`
mother-code bits directly instead; `E` is still used to derive `N` (38.212
5.3.1) but is otherwise ignored, so the effective transmitted length is `N`,
not `E` (the config report prints this effective value).

## CLI arguments

```bash
cuphy_ex_polar_decoder (-i <file> | -A <payloadBits>) [options]
```

Input:
- `-i <file>`: HDF5 vector or multi-cell YAML file.

Generator (`-A` selects generator mode):
- `-A <bits>`: payload bits per segment, `12..1706` (excludes CRC; `A` in `[12,19]` is PC-polar; large `A`/`E` splits into 2 codeblocks).
- `-E <bits>`: rate-matched bits `nTxBits`, `1..8192` (default `100`).
- `-w <n>`: codewords per run (default `16`; in `--sweep` mode, the max codeword
  budget per SNR point, default `20000`).
- `-S <dB>`: channel SNR in dB (default `20.0`).
- `--seed <n>`: base RNG seed (default `0`).
- `--no-ratematch`: skip rate matching and transmit the raw `N` mother-code bits
  (`E` only derives `N`; effective transmitted length is `N`).

Sweep:
- `--sweep`: run a BER/BLER-vs-SNR sweep (requires `-A`).
- `--snr-start <dB>` / `--snr-stop <dB>` / `--snr-step <dB>`: SNR grid
  (defaults `-2.0` / `6.0` / `1.0`; stop is inclusive, step must be `> 0`).
- `-e, --min-block-errors <n>`: accumulate until this many block errors per SNR
  point before moving on (default `100`).
- `-o <file>`: also write the sweep rows to a CSV file (the console table is
  always printed).

Decoder & execution:
- `-L <1|2|4|8>`: list size (default `1`).
- `-r <n>`: number of timed decode launches (default `20`); one un-timed warmup
  launch always runs first.
- `--G <SMs>`: run the decoder in a green context limited to `<SMs>` SMs per
  context (default off, use all SMs). Requires a CUDA >= 12.4 build; on older
  toolkits `--G` reports an error and exits.

## Examples

```bash
# Decode a stored HDF5 vector (file input):
cuphy_ex_polar_decoder -i vector.h5

# Generate + decode one batch (rate-matched channel), list-8, 50 timed launches:
cuphy_ex_polar_decoder -A 32 -E 100 -w 256 -S 20 -L 8 -r 50

# Simplified E == N channel (skip rate matching):
cuphy_ex_polar_decoder -A 32 -E 100 -S 4 -L 8 --no-ratematch

# BER/BLER-vs-SNR sweep to a CSV file:
cuphy_ex_polar_decoder --sweep -A 32 -E 100 -L 8 --snr-start -3 --snr-stop 6 -e 200 -o sweep.csv
```

## Output

Single-run modes print a report modeled on `cuphy_ex_ldpc`: a GPU device line, a
banner-delimited configuration block (`A`, `K`, CRC bits, `N`, `E`, code rate,
list size, codeword count, channel, SNR, LLR type), a timing line with
human-readable throughput

```text
Average (100 runs) elapsed time in usec = 130.3, throughput = 230.22 Mbps (info bits), 1.535 Mcodewords/s
```

and an LDPC-style correctness summary (`bit error count`, `BER`, `BLER`) plus the
polar CRC-flag tally. For file input the `E` / code-rate / channel / SNR fields
are reported as `(from input vector)`.

Sweep mode instead prints one row per SNR point (`nCw`, `bitErrors`, `BER`,
`blockErrors`, `BLER`, `crcFlaggedBLER`) and optionally writes the same rows to
CSV.

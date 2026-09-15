# LDPC decoder test bench

Python harnesses that drive the LDPC example binaries (`cuphy_ex_ldpc`,
`cuphy_ex_ldpc_rm`) to answer two questions about a decoder:

- **Does it compute the right beliefs?** — per-iteration APP (LLR) fidelity
  against a reference, checked every iteration rather than only at the output.
- **Does it actually decode?** — transport-block BLER against SNR.

## Scripts

```text
script                     via ldpc_test.py     what it does
-------------------------  -------------------  ---------------------------------------------
ldpc_test.py               (entry)              Thin dispatcher for everything below.
llr_diff_sweep.py          llr-diff             Per-iteration APP vs a REFERENCE DECODER
                                                (default algo40 box-plus). TV-free: generates
                                                its own input. Sweeps the Z x p grid, BG1+BG2,
                                                prints a heatmap + PASS/FAIL. --csv records
                                                per-cell metrics.
llr_check_sweep.py         llr-check            Per-iteration APP vs the MATLAB GOLDEN
                                                (/APP_history in a test-vector .h5), one vector
                                                per p. Naming rules below.
ldpc_rm.py                 find-snr             BLER over the full PUSCH rate-matching path;
                           find-bler            UCI on PUSCH not supported. find-snr searches
                                                for the SNR at a target TB BLER, find-bler
                                                measures it over a fixed range. Both report
                                                per-point latency.
plot_ldpc_app.py           (standalone)         Plots APP trajectories from a ref/DUT .h5 pair.
                                                --llr_check writes ldpc_dut.h5 on every run;
                                                --llr_diff needs --dump-h5.
ldpc_bench_common.py       (library)            Not a test: locates the example binary under
                                                the usual build-dir names.
```

`ldpc2_decoder_bands.yaml` is the scope list for `llr-check`: which decoders,
over which p.

The binaries used by the above .py scripts live in `cuPHY/examples/error_correction/`:

```text
cuphy_ex_ldpc.cpp            LDPC CB-based unit test, driven by codeblock params
                             such as {Z, p, filler bits, puncture}, accepting SNR
                             and modulation. No BICM support.
cuphy_ex_ldpc_rm.cpp         LDPC TB-based unit test, driven by PUSCH allocation
                             parameters such as {num_PRB, layer, MCS, MCS table},
                             accepting SNR. Contains PUSCH rate-matching;
                             data-only, no UCI.
cuphy_ex_ldpc_util.{hpp,cu}  PUSCH rate-match / de-rate-match, CRC and MCS math
                             for the bench. The header depends only on public
                             cuphy.h / cuphy.hpp; the .cu reuses the product
                             rate-matching and ldpc_params headers.
ldpc_interm_check.{hpp,cpp}  The comparison engine shared by both
                             "cuphy_ex_ldpc" and "cuphy_ex_ldpc_rm":
                             check_APP, the gates, the TV /params reader, and the
                             DUT .h5 writer.
```

## Quick start

```bash
cd cuPHY/util/ldpc

python3 ldpc_test.py --help               # list commands

# APP fidelity: full BG1 Z x p grid vs the algo40 reference (~3 s, 1247 cells)
python3 ldpc_test.py llr-diff  --algos 35 --bg 1 --z 32-384 --p 4-46

# APP fidelity vs the MATLAB golden, one vector per p
python3 ldpc_test.py llr-check --algo 40 --testvectors-dir <dir>

# BLER: find the SNR at 10% TB BLER (bisects; ~6 s at --min-error-tb 20)
python3 ldpc_test.py find-snr  --n-prb 151 --nl 1 --mcs 27 --mcs-table 2 \
    --algo 40 -n 10 --target-bler 0.10 --min-error-tb 20

# BLER: waterfall over a fixed SNR range, START,STEP,END (~8 s)
python3 ldpc_test.py find-bler --n-prb 151 --nl 1 --mcs 27 --mcs-table 2 \
    --algo 40 -n 10 --snr-range 24.6,0.4,26.6 --num-tbs 500
```

Both BLER commands take `-o <file.csv>` for machine-readable output and
`--log-dir <dir>` for the per-point decoder logs.

`find-snr` derives its transport-block count as
`ceil(--min-error-tb / --target-bler)`, so the default `--min-error-tb 100`
means 1000 TBs per point at a 10% target. Lower it for a quick check.
`--target-bler` also takes a list (`0.2,0.1,0.01`), giving one operating point
per value in a single run.

### Early termination

`--et` turns on CRC-based early termination, so a codeblock stops when CRC
passes. `--iter-count` reports the per-CB iteration spread.

```bash
python3 ldpc_test.py find-bler --n-prb 151 --nl 1 --mcs 27 --mcs-table 2 \
    --algo 35 -n 10 --snr-range 25.4,1,25.4 --num-tbs 100 --et
```

The `iter` column then shows `min/mean/max` instead of `NA`:

```text
   SNR    TB_BLER   TB errors      iter
 25.400   0.00000     0/100      5/6.44/9
 25.400   0.00000     0/100         NA
```

In the `-o` CSV, `--et` also fills `cb_bler` / `cb_err` / `cb_total` from the
decoder's own CRC results; they are empty without it.

### Test-vector requirements for `llr-check`

`llr-check` discovers vectors **by filename**. Yours must be named

```text
<anything>_bg<BG>_Z<Z>_p<p>.h5        e.g. mytv_bg1_Z384_p21.h5
```

and sit directly in `--testvectors-dir`.
`BG` and `Z` come from `ldpc2_decoder_bands.yaml`; `p` is the parity-node
count. Vectors produced by the `genTV_ldpc.m` flow already follow this.

If **no** p finds a vector the sweep exits non-zero and names the pattern it
looked for, rather than reporting an empty pass. If only *some* p are missing
they are listed as `NO-TV` in the summary and the sweep still exits 0 — check
that line before reading a `PASS` count as full coverage.

The name is also checked against the file's own `/params`, so a mislabelled
vector is reported as `ERR` (and exits non-zero).

To check an arbitrarily-named vector, skip the sweep and call the binary
directly — it reads BG/Z/p from the file, not the name:

```bash
cuphy_ex_ldpc_rm -i /path/to/anything.h5 --llr_check -a 40 -n 10
```

The binary is auto-detected under `build.aarch64`, `build-minimal-arm`,
`build-minimal-x86` or `build/`. `llr-diff` and `llr-check` take `--bin` to
override; `find-snr` / `find-bler` always autodetect.

## Build prerequisite for the APP checks

`llr-diff` and `llr-check` read the decoder's per-iteration APP, which only
exists if that decoder's ET-free dump twin was compiled in:

```bash
cmake --preset <preset> -DCUPHY_LDPC_SPLIT_DUMP_KERNELS=ON   # OFF by default
```

Which decoders carry a twin is listed in `ldpc2_decoder_bands.yaml`.
`find-snr` / `find-bler` do not need the flag.

- No dump twin: `llr-check` reports `ERR`, exits non-zero.
- No dump twin: `llr-diff` reports `NO-DUMP`, exits non-zero.
- Config not supported by the decoder: `N/A`, stays a non-failure.
- Config not supported by the ref decoder: `REF-N/A`.

## Pass criteria (APP checks)

A cell passes when all three hold: `p99err_rms <= 0.07`, `SNR >= 30 dB`, no sign
flips in the last 3 iterations. Gates live in `ldpc_interm_check.hpp`; the
scripts use its defaults.

Only the core columns `(Kb+4)*Z` are compared; the kernels do not update the
extension-parity columns during iterations. These checks are a per-iteration
APP sanity check, not a substitute for BLER verification.

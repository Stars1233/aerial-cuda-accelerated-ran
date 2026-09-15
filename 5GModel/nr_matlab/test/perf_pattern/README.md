# PERF Pattern YAML

This directory stores the shared PERF pattern TV registry used by MATLAB TV
generation and POC2 launch-pattern generation.

`perfPatternTvYaml.m` loads `perf_pattern_helper.yaml`, expands the files listed
in `pattern_files`, validates the combined pattern set, and serves the data to:

- `testCompGenTV_dlmix.m` / `testCompGenTV_ulmix.m` through `tv_table`
- `genPerfPattern.m` through `genperf_pattern_names`, `genperf_commands`,
  `base_tv_only`, `pattern_number`, `pattern_id`, and `poc2_pattern`
- `genLP_POC2.m` through `compact_case_set` and `poc2_cfg`

## File Layout

- `perf_pattern_helper.yaml`: top-level registry settings, compact sets, TV table
  limits, POC2 compact pattern list, include list, and analysis group metadata.
- `perf_pattern_legacy.yaml`: legacy 4TR-style patterns that predate the newer
  4TR/64TR split.
- `perf_pattern_4tr.yaml`: 4TR pattern definitions.
- `perf_pattern_64tr.yaml`: 64TR and mMIMO pattern definitions.

All files in `pattern_files` are loaded into one logical `patterns` list.
Pattern `id` values must be unique enough for lookup. The numeric `pattern`
value is the source of truth for POC2 test selection and launch-pattern command
generation.

## Pattern Types

Each pattern entry should fall into exactly one of these groups.

### Runnable POC2 Pattern

A runnable POC2 pattern has `pattern_type` and `channel`. It is included in
`perfPatternTvYaml('poc2_cfg', channelMap)` and can generate launch-pattern
files through `genLP_POC2`.

Example:

```yaml
- id: 66a
  pattern: 66.1
  pattern_type: 64TR_nrSim_puschOnlyU
  channel: channels_O
  config_cells: 15
  full_cells: 1
  compact_cells: 1
```

### TV-Generation-Only Pattern

A TV-generation-only pattern has `genperf: true` but no `pattern_type`. It can
emit DLMIX/ULMIX/BFW commands through `genPerfPattern`, but does not generate
POC2 launch-pattern files.

### Base TV-Only Pattern

A base TV-only pattern has `base_tv_only: true` and no `pattern_type` or
`genperf: true`. It is a source range used by other patterns through `reuses`;
direct `genPerfPattern(<pattern>)` prints a skip message and does not generate
TVs or LP files.

Use this for entries such as pattern `66`, where `66a` through `66d` are the
actual runnable patterns.

## Top-Level Keys

### `pattern_files`

List of YAML files to include. The loader reads them in order and appends their
`patterns` entries to the helper file's pattern set.

### `genperf.compact_patterns`

Pattern ids used by `genPerfPattern('compact', ...)` and
`genPerfPattern('selected', ...)`.

Ids are matched as text against `patterns[].id`. Numeric ids and quoted string
ids are accepted, but quoted strings are clearer when an id might be parsed as a
number by YAML.

### `dlmix` / `ulmix`

Side-specific TV table settings consumed by `testCompGenTV_dlmix.m` and
`testCompGenTV_ulmix.m`.

- `compact_gpu_tvs`: curated TV numbers that must generate both cuPHY and FAPI
  in compact mode.
- `tc_ranges.selected_high`: upper bound for selected/compact test scans.
- `tc_ranges.full_high`: upper bound for full test scans.

### `poc2.compact_patterns`

Numeric POC2 pattern numbers used by `genLP_POC2('compact')`.

These values are numeric `patterns[].pattern` values, not necessarily the YAML
`id`. For example, id `66a` has `pattern: 66.1`.

### `analysis_groups`

Grouping metadata for TV space analysis reports. It does not change TV or LP
generation.

- `id`: group id for report output.
- `label`: human-readable group label.
- `patterns`: pattern ids included in that report group.

## Pattern Entry Keys

### `id`

Human-readable pattern identifier used for YAML lookup and `genPerfPattern`
selection. Letter suffixes such as `66a` are allowed.

### `pattern`

Exact numeric PERF/POC2 pattern number. This is used for POC2 CFG rows and
`genLP_POC2(...)` commands.

### `description`

Short human-readable description.

### `comment`

Longer human-readable summary. This is documentation only and is not consumed by
the loader.

### `config_cells`

The TV-config stride for the pattern. DLMIX/ULMIX ranges must be divisible by
this value unless a side or range override supplies a different `config_cells`.

For POC2, this is also the configured cell count in the launch-pattern CFG.

Every pattern entry must define `config_cells`.

### `full_cells`

Number of cells generated in full TV/LP generation modes. This can be smaller
than `config_cells` when the full perf set intentionally generates only a subset
of configured cells.

### `compact_cells`

Number of cells generated in compact TV/LP generation modes. This must be less
than or equal to `full_cells`. `0` means the pattern is omitted from compact
table generation for that range.

### `pattern_type`

POC2 pattern type string passed into `genLP_POC2` CFG generation. Presence of
this key marks the entry as a runnable POC2 pattern.

### `channel`

Name of a channel map variable defined in `genLP_POC2.m`, such as `channels_O`.
Required for runnable POC2 patterns.

### `genperf`

Boolean marker for patterns returned by
`perfPatternTvYaml('genperf_pattern_names', 'full')`.

If `genperf: true` is present without `pattern_type`, the entry is
TV-generation-only and does not generate LP files.

### `base_tv_only`

Boolean marker for source-only TV ranges. These ranges may be reused by other
patterns, but direct `genPerfPattern` selection skips them.

Do not combine `base_tv_only` with `pattern_type` or `genperf: true`.

### `ranges`

Side-specific TV ranges. Supported sides are `DLMIX`, `ULMIX`, and `BFW`.

Each side can define:

- `tvs`: list of TV range entries.
- `command_expr`: optional MATLAB expression used only for the printed
  `genPerfPattern` command. The `tvs` list remains the source of truth for table
  generation and validation.
- `config_cells`, `full_cells`, `compact_cells`: optional side-level overrides.

### `ranges.<side>.tvs`

List of TV entries for one side.

Common entry forms:

```yaml
- main: [11252, 11341]
- additional: 833
- additional: [4040, 4079]
```

`main` is the primary range used to derive default `dl_tv` or `ul_tv` for POC2.
`additional` marks extra or reused ranges that are part of TV generation but not
the default POC2 TV anchor.

Structured range entries may also use supported parser fields such as `tv`,
`range`, `start_tv`/`end_tv`, `include_in_table`, and per-range
`config_cells`/`full_cells`/`compact_cells`.

### `reuses`

Documentation for TV reuse between pattern entries. This is used to make shared
TV ownership explicit.

- `side`: `DLMIX`, `ULMIX`, or `BFW`.
- `from`: source pattern id.
- `tvs`: TV number or TV range reused by this pattern.
- `from_tvs`: optional wider source range when `tvs` is a subset.

### `dl_tv` / `ul_tv`

Optional explicit POC2 TV anchors. Usually omitted because POC2 derives them
from the side's `main` range.

Use explicit values for special cases such as CA, multi-TV 80-slot patterns, or
patterns without a usable `main` DLMIX/ULMIX range.

### `dl_tv_delta`

Optional POC2 DLMIX delta field used to populate CFG column 5 in `genLP_POC2`.

### `s_slot`

Optional special-slot TV configuration.

- `tv`: scalar or vector passed into the S-slot CFG.

There is no `s_slot.full_cells`. S-slot cell count comes from the parent
pattern's `config_cells`.

### `bfw_srs_bind`

Optional list of BFW/SRS binding rows consumed by `genLP_POC2`.

Each row can define:

- `srs_tv`
- `srs_slot`
- `srs_name_slot`
- `bfw_dl_tv`
- `bfw_dl_slots`
- `bfw_ul_tv`
- `bfw_ul_slots`
- `cell_idx`

### `overrides`

Optional launch-pattern override rows consumed by `genLP_POC2`.

Each row can define:

- `tv`
- `slot`
- `base_pattern`
- `cell_idx`
- `tv_increments`
- `tv_name_slot`
- `mix_tv`
- `target_dl`
- `verbose`

### `command_args`

Optional side-specific extra argument for `genPerfPattern` command text.

Example:

```yaml
command_args:
  ULMIX: genTV
  DLMIX: genTV
```

## Cell Count Rules

- `config_cells` must be positive.
- `full_cells` must be positive when present.
- `compact_cells` can be zero but cannot exceed `full_cells`.
- `full_cells` cannot exceed `config_cells`.
- TV range length must be divisible by the effective `config_cells`.
- Direct `genLP_POC2(pattern)` uses `config_cells`.
- Direct `genLP_POC2(pattern, nCells)` uses the requested count, clamped to
  `config_cells` with a warning when requested count is too large.

## Adding A Pattern

1. Add the entry to the appropriate included pattern file.
2. Set `id`, exact numeric `pattern`, `config_cells`, `full_cells`, and
   `compact_cells`.
3. Add `ranges` for each side that owns or generates TVs.
4. Add `pattern_type` and `channel` only if the pattern should generate POC2 LP
   files.
5. Use `genperf: true` for patterns that should be directly generated by
   `genPerfPattern`.
6. Use `base_tv_only: true` for source-only TV ranges that must not be directly
   generated.
7. Add `reuses` when this entry shares TV ranges with another pattern.
8. Add the numeric `pattern` to `poc2.compact_patterns` only when it belongs in
   compact LP generation.
9. Add the `id` to `genperf.compact_patterns` only when it belongs in compact TV
   generation.

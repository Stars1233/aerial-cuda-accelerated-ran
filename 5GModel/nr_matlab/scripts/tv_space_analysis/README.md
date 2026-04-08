# TV Space Analysis

Analyzes test vector (TV) `.h5` disk usage from directory listings. It parses `ls -alh`-style output, classifies files (DLMIX/ULMIX, FAPI/CUPHY/OTHER), and prints a report by pattern and category. Pattern ranges align with `5GModel/nr_matlab/test/genPerfPattern.m`.

## Requirements

- Python 3.6+
- Unix-like environment when using `--dir` (uses `ls -alh`)

## Usage

```bash
# Analyze a TV directory (runs ls -alh on it and reports)
python tv_space_analysis.py --dir /path/to/TV/folder

# Also use a log file: combine its contents with the directory listing, then analyze
# The log file is updated with the combined content for next time
python tv_space_analysis.py --dir /path/to/TV/folder --log my.log
```

### Options

| Option   | Required | Description |
|----------|----------|-------------|
| `--dir`  | Yes      | TV directory to analyze. The script runs `ls -alh` on this path. |
| `--log`  | No       | Log file path. If given, its lines are combined with the directory listing before analysis, and the file is overwritten with the combined content. |
| `--lp`   | No       | LP (pattern) ID(s) to analyze. Reports the size of child TVs for each LP. Multiple IDs separated by commas (e.g. `81a`, `81a,81c`). |

### Examples

```bash
# From this directory
python tv_space_analysis.py --dir ../../test/GPU_test_input/

# With a log file (e.g. to accumulate listings)
python tv_space_analysis.py --dir /data/tv_output --log tv_listing.log

# Analyze one or more LPs (comma-separated) and report size of their child TVs
python tv_space_analysis.py --dir ../../test/GPU_test_input/ --lp 81c
python tv_space_analysis.py --dir ../../test/GPU_test_input/ --lp 81a,81c
```

## Input format

The script expects lines in the format produced by `ls -alh` (or `ls -alhR`): each line should contain a size with unit (e.g. `1.2M`, `500K`) and a filename. It recognizes:

- **TVnr_DLMIX_*** and **TVnr_ULMIX_*** `.h5` files → counted by pattern and FAPI/CUPHY/OTHER
- Other **TVnr_*.h5** → "Other TVnr_ (not DLMIX/ULMIX)"
- Other **.h5** files → "Non-TVnr_ .h5 files"

## Output

- A table of pattern rows (DLMIX FAPI/cuPHY, ULMIX FAPI/cuPHY columns), subtotals, and totals
- "Other DLMIX/ULMIX (no pattern)" row
- DLMIX/ULMIX subtotal
- Other TVnr_ and non-TVnr_ .h5 rows
- Grand total and a short summary block
- Parsing warnings/errors if any

## Relation to shell script

The behavior matches `5GModel/nr_matlab/test/analyze_tv_space.sh`. This Python package can be used instead of the shell script for easier editing, testing, and running without a heredoc.

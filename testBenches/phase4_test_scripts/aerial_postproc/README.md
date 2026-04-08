# Aerial Post Processing (aerial_postproc)

A Python package for performance analysis and log processing of Aerial cuBB test results. This package provides tools for parsing log files, extracting performance metrics, generating visualizations, and performing threshold-based gating checks for CI/CD pipelines.

## Features

- **Log Parsing**: Parse PHY, testMAC, and RU log files into binary formats for efficient processing
- **Performance Metrics**: Extract and analyze performance metrics from test runs
- **Visualizations**: Generate interactive HTML reports (compare_logs, latency_summary, timeline plots)
- **Threshold Checking**: Gating, warning, and absolute threshold checks for CI/CD pipelines
- **NIC Checkout Tools**: Analysis tools for netperftest and fhgen results
- **Power Analysis**: Tools for collecting and plotting power measurements

## Virtual Environment Setup

The package requires a Python virtual environment with specific dependencies. Use the provided scripts to manage the environment.

### Creating the Virtual Environment

```bash
# Create the virtual environment (one-time setup)
./venv_create.sh
```

This creates the virtual environment at `$HOME/.aerial_postproc_venv` by default.

To use a custom location:

```bash
# Set custom location before creating
export AERIAL_POSTPROC_VENV=/path/to/custom/venv
./venv_create.sh
```

**Note**: In CI/CD environments, the venv is typically pre-installed at `/opt/aerial_postproc_venv`.

### Activating the Virtual Environment

```bash
# Activate the venv (for interactive use)
source ./venv_activate.sh
```

### Running Scripts Without Activation

For non-interactive use, run scripts directly using `run_isolated.sh`:

```bash
# Run a script without activating the venv
./run_isolated.sh scripts/cicd/cicd_parse.py --help
```

### Managing the Virtual Environment

```bash
# Deactivate (if activated)
source ./venv_deactivate.sh

# Delete the virtual environment
./venv_delete.sh
```

## Scripts Documentation

Each subdirectory under `scripts/` contains a README with detailed guidance for that category:

- `scripts/analysis/README.md` - Visualization and analysis scripts
- `scripts/cicd/README.md` - Log parsing, metrics extraction, and threshold checking for CI/CD
- `scripts/dashboard/README.md` - Metrics dashboard upload (uses separate venv)
- `scripts/nic_checkouts/fhgen/README.md` - FHGen validation tools
- `scripts/nic_checkouts/netperftest/README.md` - Netperftest validation tools
- `scripts/power/README.md` - Power measurement tools

## Requirements

- Python 3.8+
- See `requirements.txt` for Python package dependencies

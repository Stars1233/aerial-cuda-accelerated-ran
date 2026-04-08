# cubb_scripts

## Installation

See [install/README.md](install/README.md) for Aerial SDK installation and make targets. At the moment the installation only supports DGX Spark.

## Dependencies
<p>pyyaml (https://pypi.org/project/PyYAML/) <p>
<p>h5py (https://docs.h5py.org/en/latest/build.html)<p>

## Running
python3 auto_lp.py -i <input_dir> -t <launch_pattern_template.yaml> -c <numcells, default=0> -a (all slots)
### Example
python3 auto_lp.py -i /path/to/GPU_test_input -t launch_pattern_nrSim.yaml -c 8 -a 

produces launch_pattern_nrSim_xxxxx.yaml and out.txt with every line  = xxxxx which 
is the name of the TC 

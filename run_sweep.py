#!/usr/bin/env python
"""
Wrapper script to run Hydra sweeps in truly separate processes.
This ensures complete isolation between experiments, preventing memory leaks and library conflicts.

Usage:
    python run_sweep.py <config_name>
    
Example:
    python run_sweep.py debugging
"""
import subprocess
import sys
import itertools
from datetime import datetime
import os
import yaml

if len(sys.argv) < 2:
    print("Usage: python run_sweep.py <config_name>")
    print("Example: python run_sweep.py debugging")
    sys.exit(1)

# Get config name from command line
CONFIG_NAME = sys.argv[1]
config_path = f"conf/{CONFIG_NAME}.yaml"

if not os.path.exists(config_path):
    print(f"Error: Config file not found: {config_path}")
    sys.exit(1)

# Load the config file
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

# Extract sweep parameters from the config
if 'hydra' not in config or 'sweeper' not in config['hydra']:
    print(f"Error: Config file {config_path} does not contain hydra.sweeper")
    print("Expected format:")
    print("hydra:")
    print("  sweeper:")
    print("    params:  # OR grid_params and/or list_params")
    print("      param1: value1, value2")
    sys.exit(1)

sweeper = config['hydra']['sweeper']

# Parse sweep parameters - support both old 'params' and new 'grid_params'/'list_params'
GRID_PARAMS = {}
LIST_PARAMS = {}

if 'params' in sweeper:
    # Old format - treat as grid params
    for param_name, param_values in sweeper['params'].items():
        if isinstance(param_values, str):
            values = [v.strip() for v in param_values.split(',')]
        elif isinstance(param_values, list):
            values = param_values
        else:
            values = [param_values]
        GRID_PARAMS[param_name] = values
elif 'grid_params' in sweeper or 'list_params' in sweeper:
    # New format with grid_params and list_params
    if 'grid_params' in sweeper:
        for param_name, param_values in sweeper['grid_params'].items():
            if isinstance(param_values, str):
                values = [v.strip() for v in param_values.split(',')]
            elif isinstance(param_values, list):
                values = param_values
            else:
                values = [param_values]
            GRID_PARAMS[param_name] = values
    
    if 'list_params' in sweeper:
        for param_name, param_values in sweeper['list_params'].items():
            if isinstance(param_values, str):
                values = [v.strip() for v in param_values.split(',')]
            elif isinstance(param_values, list):
                values = param_values
            else:
                values = [param_values]
            LIST_PARAMS[param_name] = values
        
        # Validate that all list_params have the same length
        lengths = [len(v) for v in LIST_PARAMS.values()]
        if len(set(lengths)) > 1:
            print(f"Error: All list_params must have the same length")
            print(f"Found lengths: {dict(zip(LIST_PARAMS.keys(), lengths))}")
            sys.exit(1)
else:
    print(f"Error: Config file {config_path} must contain either 'params' or 'grid_params'/'list_params'")
    sys.exit(1)

# Detect config groups (directories in conf/)
CONFIG_GROUPS = set()
conf_dir = "conf"
for item in os.listdir(conf_dir):
    item_path = os.path.join(conf_dir, item)
    if os.path.isdir(item_path):
        CONFIG_GROUPS.add(item)

print(f"Loaded sweep configuration from: {config_path}")
if GRID_PARAMS:
    print(f"Grid parameters (cartesian product): {GRID_PARAMS}")
if LIST_PARAMS:
    print(f"List parameters (paired/zipped): {LIST_PARAMS}")
print(f"Detected config groups: {CONFIG_GROUPS}")
print()

# Prepare output directory path with timestamp (don't create it yet)
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
output_dir = f"test/{timestamp}"

# Generate all combinations of parameters
all_combinations = []

# First, generate list_params combinations (zipped/paired)
if LIST_PARAMS:
    list_param_names = list(LIST_PARAMS.keys())
    list_param_values = list(LIST_PARAMS.values())
    # Zip the values together (pair them by position)
    list_combinations = list(zip(*list_param_values))
else:
    list_param_names = []
    list_combinations = [()]  # Single empty combination if no list params

# Then, generate grid_params combinations (cartesian product)
if GRID_PARAMS:
    grid_param_names = list(GRID_PARAMS.keys())
    grid_param_values = list(GRID_PARAMS.values())
    grid_combinations = list(itertools.product(*grid_param_values))
else:
    grid_param_names = []
    grid_combinations = [()]  # Single empty combination if no grid params

# Combine list and grid combinations
# For each list combination, create all grid combinations
for list_combo in list_combinations:
    for grid_combo in grid_combinations:
        # Merge the two combinations
        combined = list_combo + grid_combo
        all_combinations.append(combined)

# Combined parameter names
param_names = list_param_names + grid_param_names

print(f"Running {len(all_combinations)} experiments in separate processes...")
print(f"Output directory: {output_dir}")
print("=" * 80)

failed_experiments = []

for i, combination in enumerate(all_combinations, 1):
    # Build the command - use simple key=value for all parameters
    # Hydra will automatically recognize dataset and model as config groups
    overrides = [f"{name}={value}" for name, value in zip(param_names, combination)]
    
    # Create a unique subdirectory for this experiment (without = signs to avoid parsing issues)
    exp_name = "_".join([f"{name}_{value}" for name, value in zip(param_names, combination)])
    exp_output_dir = os.path.join(output_dir, exp_name)
    
    # Build all overrides including hydra output directory
    all_overrides = overrides + [f"hydra.run.dir={exp_output_dir}"]
    
    # Build command to run the experiment
    cmd = [
        sys.executable,  # Use the same Python interpreter as the one running this script
        'main.py',
        f'--config-name={CONFIG_NAME}'
    ] + all_overrides
    
    print(f"\n[{i}/{len(all_combinations)}] Running experiment:")
    print(f"  Config: {', '.join(overrides)}")
    print(f"  Output: {exp_output_dir}")
    print("-" * 80)
    
    # Run the command in a separate process
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=False,  # Show output in real-time
            text=True,
            cwd=os.getcwd()
        )
        print(f"✓ Experiment {i}/{len(all_combinations)} completed successfully")
    except subprocess.CalledProcessError as e:
        print(f"✗ Experiment {i}/{len(all_combinations)} FAILED with exit code {e.returncode}")
        failed_experiments.append((i, exp_name, e.returncode))
    except Exception as e:
        print(f"✗ Experiment {i}/{len(all_combinations)} FAILED with error: {e}")
        failed_experiments.append((i, exp_name, str(e)))

print("\n" + "=" * 80)
print("SWEEP COMPLETE")
print("=" * 80)
print(f"Total experiments: {len(all_combinations)}")
print(f"Successful: {len(all_combinations) - len(failed_experiments)}")
print(f"Failed: {len(failed_experiments)}")

if failed_experiments:
    print("\nFailed experiments:")
    for idx, name, error in failed_experiments:
        print(f"  [{idx}] {name} - Error: {error}")
    sys.exit(1)
else:
    print("\n✓ All experiments completed successfully!")
    sys.exit(0)

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
if 'hydra' not in config or 'sweeper' not in config['hydra'] or 'params' not in config['hydra']['sweeper']:
    print(f"Error: Config file {config_path} does not contain hydra.sweeper.params")
    print("Expected format:")
    print("hydra:")
    print("  sweeper:")
    print("    params:")
    print("      param1: value1, value2")
    print("      param2: value3, value4")
    sys.exit(1)

# Parse sweep parameters
SWEEP_PARAMS = {}
for param_name, param_values in config['hydra']['sweeper']['params'].items():
    if isinstance(param_values, str):
        # Parse comma-separated values
        values = [v.strip() for v in param_values.split(',')]
    elif isinstance(param_values, list):
        values = param_values
    else:
        values = [param_values]
    SWEEP_PARAMS[param_name] = values

# Detect config groups (directories in conf/)
CONFIG_GROUPS = set()
conf_dir = "conf"
for item in os.listdir(conf_dir):
    item_path = os.path.join(conf_dir, item)
    if os.path.isdir(item_path):
        CONFIG_GROUPS.add(item)

# Also check which params in the defaults list are config groups
if 'defaults' in config:
    for default in config['defaults']:
        if isinstance(default, dict):
            for key in default.keys():
                if key in CONFIG_GROUPS:
                    # This confirms it's a config group
                    pass

print(f"Loaded sweep configuration from: {config_path}")
print(f"Sweep parameters: {SWEEP_PARAMS}")
print(f"Detected config groups: {CONFIG_GROUPS}")
print()

# Create output directory with timestamp
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
output_dir = f"test/{timestamp}"
os.makedirs(output_dir, exist_ok=True)

# Generate all combinations of parameters
param_names = list(SWEEP_PARAMS.keys())
param_values = list(SWEEP_PARAMS.values())
all_combinations = list(itertools.product(*param_values))

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
    
    # Build command that activates conda environment and runs the experiment
    python_cmd = f"python main.py --config-name={CONFIG_NAME} {' '.join(all_overrides)}"
    
    # Use bash to source conda and activate the environment
    cmd = [
        'bash',
        '-c',
        f'source $(conda info --base)/etc/profile.d/conda.sh && conda activate lmr && {python_cmd}'
    ]
    
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

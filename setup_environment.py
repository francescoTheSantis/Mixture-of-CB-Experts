#!/usr/bin/env python3
"""
Environment Setup Script

This script sets up the complete environment before running experiments:
1. Creates conda environment from environment.yml
2. Installs "models" branch of pytorch_concepts from GitHub
3. Pre-initializes Julia packages for PySR to avoid runtime conflicts

Usage:
    python setup_environment.py
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(cmd, description, shell=False):
    """Run a shell command and handle errors."""
    print(f"\n{'='*60}")
    print(f"{description}")
    print(f"{'='*60}")
    print(f"Running: {cmd if isinstance(cmd, str) else ' '.join(cmd)}\n")
    
    try:
        result = subprocess.run(
            cmd,
            shell=shell,
            check=True,
            text=True,
            capture_output=False
        )
        print(f"✓ {description} completed successfully")
        return result
    except subprocess.CalledProcessError as e:
        print(f"✗ Error during: {description}")
        print(f"Exit code: {e.returncode}")
        sys.exit(1)

def get_conda_executable():
    """Find conda or mamba executable."""
    for exe in ['mamba', 'conda']:
        try:
            subprocess.run([exe, '--version'], capture_output=True, check=True)
            print(f"Using {exe} for package management")
            return exe
        except (subprocess.CalledProcessError, FileNotFoundError):
            continue
    print("Error: Neither conda nor mamba found in PATH")
    sys.exit(1)

def main():
    # Get the project root directory
    project_root = Path(__file__).parent
    env_file = project_root / "environment.yml"
    
    if not env_file.exists():
        print(f"Error: environment.yml not found at {env_file}")
        sys.exit(1)
    
    print("="*60)
    print("Mixture of Concept Bottleneck Experts (M-CBEs) - Environment Setup")
    print("="*60)
    
    # Get conda executable
    conda_exe = get_conda_executable()
    
    # Read environment name from environment.yml
    env_name = "m_cbe"  # Default name
    try:
        with open(env_file) as f:
            for line in f:
                if line.strip().startswith('name:'):
                    env_name = line.split(':')[1].strip()
                    break
    except Exception as e:
        print(f"Warning: Could not read environment name, using default '{env_name}'")
    
    print(f"\nEnvironment name: {env_name}")
    
    # Step 1: Create/update conda environment
    run_command(
        [conda_exe, 'env', 'create', '-f', str(env_file)],
        f"Step 1/3: Creating conda environment '{env_name}'",
        shell=False
    )
    
    # Step 2: Install pytorch_concepts from GitHub
    # The package's shared.py imports pkg_resources at build time (a bug in the package),
    # which is unavailable in some Python 3.12 environments. We clone the repo, patch that
    # one import, and install locally to work around it.
    import tempfile
    import shutil
    tmp_dir = Path(tempfile.mkdtemp())
    pc_dir = tmp_dir / "pytorch_concepts"
    try:
        print(f"\n{'='*60}")
        print("Step 2/3: Installing pytorch_concepts from GitHub")
        print(f"{'='*60}")
        print("Cloning pytorch_concepts@models...")
        subprocess.run(
            ['git', 'clone', '--branch', 'models', '--depth', '1',
             'https://github.com/pyc-team/pytorch_concepts.git', str(pc_dir)],
            check=True
        )
        # Patch shared.py: replace pkg_resources with pathlib (pkg_resources is
        # not available at build time in Python 3.12 conda environments)
        shared_py = pc_dir / 'torch_concepts' / 'data' / 'traffic_construction' / 'shared.py'
        patched = (
            '"""\nShared global variables for this dataset generation.\n"""\n'
            'from pathlib import Path\n\n'
            '# Directory where all the useful sprites are stored\n'
            'SPRITES_DIRECTORY = lambda x: str(\n'
            '    Path(__file__).parent.parent.parent / "assets" / x\n'
            ')\n'
        )
        shared_py.write_text(patched)
        print("✓ Patched shared.py (replaced pkg_resources with pathlib)")

        run_command(
            [conda_exe, 'run', '-n', env_name, 'pip', 'install',
             '--no-build-isolation', str(pc_dir)],
            "Step 2/3: Installing pytorch_concepts from local clone",
            shell=False
        )
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    
    # Step 3: Pre-initialize Julia/PySR packages
    pysr_init_script = """
import os
# Prevent CUDA conflicts during Julia initialization
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['JULIA_CUDA_USE_BINARYBUILDER'] = 'false'

print("Initializing PySR and Julia packages...")
print("This will download and precompile Julia packages (may take a few minutes)...")

try:
    import pysr
    print("✓ PySR imported successfully")
    
    # Trigger Julia initialization
    from pysr import PySRRegressor
    print("✓ Julia packages initialized successfully")
    
    print("\\nSetup complete! Julia packages are ready for use.")
except Exception as e:
    print(f"Error during PySR initialization: {e}")
    import sys
    sys.exit(1)
"""
    
    # Write temporary script and execute it
    temp_script = project_root / "_temp_pysr_init.py"
    try:
        with open(temp_script, 'w') as f:
            f.write(pysr_init_script)
        
        run_command(
            [conda_exe, 'run', '-n', env_name, 'python', str(temp_script)],
            "Step 3/3: Pre-initializing Julia packages for PySR",
            shell=False
        )
    finally:
        # Clean up temporary script
        if temp_script.exists():
            temp_script.unlink()
    
    print("\n" + "="*60)
    print("SETUP COMPLETE!")
    print("="*60)
    print(f"\nTo activate the environment, run:")
    print(f"    conda activate {env_name}")
    print(f"\nThen you can run your experiments:")
    print(f"    python main.py")
    print("="*60)

if __name__ == "__main__":
    main()

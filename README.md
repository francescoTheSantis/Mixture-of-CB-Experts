# Linear-Memory-Reasoner

## Setup

```bash
python setup_environment.py
conda activate lmr
```

This script creates the environment, installs dependencies, and pre-initializes Julia packages for PySR.

**Configure environment variables in** `env.py`:

   * `HOME` - Path to the project root
   * `DATA_PATH` - Path to load datasets
   * `PROJECT_NAME` (optional) - Project identifier for logging

**Datasets**:
   Most datasets are automatically downloaded and processed when needed. For CIFAR10/CIFAR100, you need to manually download prerequisite text files (one-time setup):

   * Download `cifar10_filtered.txt` and `cifar10_classes.txt` from [Label-free-CBM](https://github.com/Trustworthy-ML-Lab/Label-free-CBM/tree/main)
   * Place them in `{DATA_PATH}/cifar10/`
   * Do the same for CIFAR100

## Running Experiments

To replicate all experiments from the paper:

1. **Run multiple experiments**:

   ```bash
   python run_sweep.py <specific_config>
   ```

   Where `"specific_config"` is one from the `conf/` directory.

2. **Plot results**:
   Add the result paths to the `paths` list in `show_results.py`, then run:

   ```bash
   python show_results.py
   ```

   Customize `custom_order` and `model_styles` as needed.

## Configuration

Experimental settings are defined in `conf/`:

* `dataset/` – dataset parameters
* `model/` – model architectures
* `encoder/` – input encoder configs
* `engine/` – training loop settings
* `common.yaml` – shared defaults






conda activate lmr && export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH" && python show_results.py